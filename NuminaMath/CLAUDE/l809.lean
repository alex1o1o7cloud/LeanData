import Mathlib

namespace NUMINAMATH_CALUDE_balloon_distribution_l809_80935

theorem balloon_distribution (red white green chartreuse : ℕ) 
  (h1 : red = 24)
  (h2 : white = 38)
  (h3 : green = 68)
  (h4 : chartreuse = 75)
  (friends : ℕ)
  (h5 : friends = 10) :
  (red + white + green + chartreuse) % friends = 5 := by
  sorry

end NUMINAMATH_CALUDE_balloon_distribution_l809_80935


namespace NUMINAMATH_CALUDE_triangle_line_equations_l809_80976

/-- Triangle ABC with given properties -/
structure Triangle where
  A : ℝ × ℝ
  CM : ℝ → ℝ → Prop
  BH : ℝ → ℝ → Prop

/-- The given triangle satisfies the problem conditions -/
def given_triangle : Triangle where
  A := (5, 1)
  CM := fun x y ↦ 2 * x - y - 5 = 0
  BH := fun x y ↦ x - 2 * y - 5 = 0

/-- Line equation represented as ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem stating the equations of line BC and its symmetric line -/
theorem triangle_line_equations (t : Triangle) 
  (h : t = given_triangle) : 
  ∃ (BC symmetric_BC : LineEquation),
    (BC.a = 6 ∧ BC.b = -5 ∧ BC.c = -9) ∧
    (symmetric_BC.a = 38 ∧ symmetric_BC.b = -9 ∧ symmetric_BC.c = -125) := by
  sorry

end NUMINAMATH_CALUDE_triangle_line_equations_l809_80976


namespace NUMINAMATH_CALUDE_parallel_vectors_tan_l809_80958

theorem parallel_vectors_tan (x : ℝ) : 
  let a : ℝ × ℝ := (Real.sin x, Real.cos x)
  let b : ℝ × ℝ := (2, -3)
  (a.1 * b.2 = a.2 * b.1) → Real.tan x = -2/3 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_tan_l809_80958


namespace NUMINAMATH_CALUDE_boys_camp_total_l809_80917

theorem boys_camp_total (total : ℕ) 
  (h1 : (total : ℚ) * (1 / 5) = (total : ℚ) * (20 / 100))
  (h2 : (total : ℚ) * (1 / 5) * (3 / 10) = (total : ℚ) * (1 / 5) * (30 / 100))
  (h3 : (total : ℚ) * (1 / 5) * (7 / 10) = 56) :
  total = 400 := by
sorry

end NUMINAMATH_CALUDE_boys_camp_total_l809_80917


namespace NUMINAMATH_CALUDE_dice_roll_sum_l809_80944

theorem dice_roll_sum (a b c d : ℕ) : 
  1 ≤ a ∧ a ≤ 6 ∧
  1 ≤ b ∧ b ≤ 6 ∧
  1 ≤ c ∧ c ≤ 6 ∧
  1 ≤ d ∧ d ≤ 6 ∧
  a * b * c * d = 360 →
  a + b + c + d ≠ 20 := by
sorry

end NUMINAMATH_CALUDE_dice_roll_sum_l809_80944


namespace NUMINAMATH_CALUDE_circle_properties_l809_80998

noncomputable def circle_equation (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 1)^2 = 1

theorem circle_properties :
  ∃ (c : ℝ × ℝ),
    (c.1 = c.2) ∧  -- Center is on the line y = x
    (∀ x y : ℝ, circle_equation x y → (x - c.1)^2 + (y - c.2)^2 = 1) ∧  -- Equation represents a circle
    (circle_equation 1 0) ∧  -- Circle passes through (1,0)
    (∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 1| ∧ |x - 1| < δ → ¬(circle_equation x 0)) :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l809_80998


namespace NUMINAMATH_CALUDE_smallest_alpha_inequality_l809_80924

theorem smallest_alpha_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  ∃ (α : ℝ), α > 0 ∧ α = 1/2 ∧
  ∀ (β : ℝ), β > 0 →
    ((x + y) / 2 ≥ β * Real.sqrt (x * y) + (1 - β) * Real.sqrt ((x^2 + y^2) / 2) →
     β ≥ α) :=
by sorry

end NUMINAMATH_CALUDE_smallest_alpha_inequality_l809_80924


namespace NUMINAMATH_CALUDE_min_cost_is_80_yuan_l809_80966

/-- Represents the swimming trip problem -/
structure SwimmingTripProblem where
  card_cost : ℕ            -- Cost of each swim card in yuan
  students : ℕ             -- Number of students
  swims_per_student : ℕ    -- Number of swims each student needs
  bus_cost : ℕ             -- Cost of bus rental per trip in yuan

/-- Calculates the minimum cost per student for the swimming trip -/
def min_cost_per_student (problem : SwimmingTripProblem) : ℚ :=
  let total_swims := problem.students * problem.swims_per_student
  let cards := 8  -- Optimal number of cards to buy
  let trips := total_swims / cards
  let total_cost := problem.card_cost * cards + problem.bus_cost * trips
  (total_cost : ℚ) / problem.students

/-- Theorem stating that the minimum cost per student is 80 yuan -/
theorem min_cost_is_80_yuan (problem : SwimmingTripProblem) 
    (h1 : problem.card_cost = 240)
    (h2 : problem.students = 48)
    (h3 : problem.swims_per_student = 8)
    (h4 : problem.bus_cost = 40) : 
  min_cost_per_student problem = 80 := by
  sorry

#eval min_cost_per_student { card_cost := 240, students := 48, swims_per_student := 8, bus_cost := 40 }

end NUMINAMATH_CALUDE_min_cost_is_80_yuan_l809_80966


namespace NUMINAMATH_CALUDE_remainder_101_pow_36_mod_100_l809_80909

theorem remainder_101_pow_36_mod_100 : 101^36 % 100 = 1 := by sorry

end NUMINAMATH_CALUDE_remainder_101_pow_36_mod_100_l809_80909


namespace NUMINAMATH_CALUDE_average_weight_problem_l809_80968

theorem average_weight_problem (A B C : ℝ) 
  (h1 : (A + B) / 2 = 40)
  (h2 : (B + C) / 2 = 43)
  (h3 : B = 31) :
  (A + B + C) / 3 = 45 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_problem_l809_80968


namespace NUMINAMATH_CALUDE_equation_solution_l809_80959

theorem equation_solution : ∃! x : ℚ, (10 - 2*x)^2 = 4*x^2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l809_80959


namespace NUMINAMATH_CALUDE_Q_always_perfect_square_l809_80902

theorem Q_always_perfect_square (x : ℤ) : ∃ (b : ℤ), x^4 + 4*x^3 + 8*x^2 + 6*x + 9 = b^2 := by
  sorry

end NUMINAMATH_CALUDE_Q_always_perfect_square_l809_80902


namespace NUMINAMATH_CALUDE_expression_evaluation_l809_80988

theorem expression_evaluation : 
  (2019^3 - 3 * 2019^2 * 2020 + 3 * 2019 * 2020^2 - 2020^3 + 6) / (2019 * 2020) = 5 / (2019 * 2020) := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l809_80988


namespace NUMINAMATH_CALUDE_binary_digit_difference_l809_80903

-- Define a function to calculate the number of digits in the binary representation of a number
def binaryDigits (n : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log2 n + 1

-- State the theorem
theorem binary_digit_difference : binaryDigits 950 - binaryDigits 150 = 2 := by
  sorry

end NUMINAMATH_CALUDE_binary_digit_difference_l809_80903


namespace NUMINAMATH_CALUDE_sum_of_series_equals_one_l809_80980

/-- The sum of the infinite series ∑(n=1 to ∞) (4n-3)/(3^n) is equal to 1 -/
theorem sum_of_series_equals_one :
  (∑' n : ℕ, (4 * n - 3 : ℝ) / (3 : ℝ) ^ n) = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_series_equals_one_l809_80980


namespace NUMINAMATH_CALUDE_savings_proof_l809_80921

/-- Calculates savings given income and expenditure ratio -/
def calculate_savings (income : ℕ) (income_ratio expenditure_ratio : ℕ) : ℕ :=
  income - (income * expenditure_ratio) / income_ratio

/-- Proves that given the specified conditions, the savings are 3400 -/
theorem savings_proof (income : ℕ) (income_ratio expenditure_ratio : ℕ) 
  (h1 : income = 17000)
  (h2 : income_ratio = 5)
  (h3 : expenditure_ratio = 4) :
  calculate_savings income income_ratio expenditure_ratio = 3400 := by
  sorry

#eval calculate_savings 17000 5 4

end NUMINAMATH_CALUDE_savings_proof_l809_80921


namespace NUMINAMATH_CALUDE_bobbys_shoe_cost_bobbys_shoe_cost_is_968_l809_80941

/-- Calculates the total cost of Bobby's handmade shoes -/
theorem bobbys_shoe_cost (mold_cost : ℝ) (labor_rate : ℝ) (work_hours : ℝ) 
  (labor_discount : ℝ) (materials_cost : ℝ) (tax_rate : ℝ) : ℝ :=
  let discounted_labor_cost := labor_rate * work_hours * labor_discount
  let total_before_tax := mold_cost + discounted_labor_cost + materials_cost
  let tax := total_before_tax * tax_rate
  let total_with_tax := total_before_tax + tax
  
  total_with_tax

theorem bobbys_shoe_cost_is_968 :
  bobbys_shoe_cost 250 75 8 0.8 150 0.1 = 968 := by
  sorry

end NUMINAMATH_CALUDE_bobbys_shoe_cost_bobbys_shoe_cost_is_968_l809_80941


namespace NUMINAMATH_CALUDE_smallest_k_for_square_l809_80962

theorem smallest_k_for_square : ∃ (m : ℕ), 
  2016 * 2017 * 2018 * 2019 + 1 = m^2 ∧ 
  ∀ (k : ℕ), k < 1 → ¬∃ (n : ℕ), 2016 * 2017 * 2018 * 2019 + k = n^2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_for_square_l809_80962


namespace NUMINAMATH_CALUDE_infinite_divisibility_l809_80946

theorem infinite_divisibility (a : ℕ) (h : a > 3) :
  ∃ (f : ℕ → ℕ), Monotone f ∧ (∀ i, (a + f i) ∣ (a^(f i) + 1)) :=
sorry

end NUMINAMATH_CALUDE_infinite_divisibility_l809_80946


namespace NUMINAMATH_CALUDE_job_completion_times_l809_80913

/-- Represents the productivity of a worker -/
structure Productivity where
  rate : ℝ
  rate_pos : rate > 0

/-- Represents a worker -/
structure Worker where
  productivity : Productivity

/-- Represents a job with three workers -/
structure Job where
  worker1 : Worker
  worker2 : Worker
  worker3 : Worker
  total_work : ℝ
  total_work_pos : total_work > 0
  third_worker_productivity : worker3.productivity.rate = (worker1.productivity.rate + worker2.productivity.rate) / 2
  work_condition : 48 * worker3.productivity.rate + 10 * worker1.productivity.rate = 
                   48 * worker3.productivity.rate + 15 * worker2.productivity.rate

/-- The theorem to be proved -/
theorem job_completion_times (job : Job) :
  let time1 := job.total_work / job.worker1.productivity.rate
  let time2 := job.total_work / job.worker2.productivity.rate
  let time3 := job.total_work / job.worker3.productivity.rate
  (time1 = 50 ∧ time2 = 75 ∧ time3 = 60) := by
  sorry

end NUMINAMATH_CALUDE_job_completion_times_l809_80913


namespace NUMINAMATH_CALUDE_chord_length_hyperbola_equation_l809_80994

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the line with slope 1 passing through the right focus
def line (x y : ℝ) : Prop := y = x - Real.sqrt 3

-- Define the hyperbola
def hyperbola (a b x y : ℝ) : Prop := x^2/a^2 - y^2/b^2 = 1

-- Theorem 1: Length of chord AB
theorem chord_length : 
  ∃ (A B : ℝ × ℝ), 
    ellipse A.1 A.2 ∧ 
    ellipse B.1 B.2 ∧ 
    line A.1 A.2 ∧ 
    line B.1 B.2 ∧ 
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8/5 :=
sorry

-- Theorem 2: Equation of the hyperbola
theorem hyperbola_equation :
  ∃ (a b : ℝ), 
    hyperbola 3 4 (-3) (2 * Real.sqrt 3) ∧
    ∀ (x y : ℝ), hyperbola a b x y ↔ 4*x^2/9 - y^2/4 = 1 :=
sorry

end NUMINAMATH_CALUDE_chord_length_hyperbola_equation_l809_80994


namespace NUMINAMATH_CALUDE_general_admission_price_general_admission_price_is_20_l809_80908

/-- Calculates the price of a general admission ticket given the total number of tickets sold,
    total revenue, VIP ticket price, and the difference between general and VIP tickets sold. -/
theorem general_admission_price 
  (total_tickets : ℕ) 
  (total_revenue : ℝ) 
  (vip_price : ℝ) 
  (ticket_difference : ℕ) : ℝ :=
  let general_tickets := (total_tickets + ticket_difference) / 2
  let vip_tickets := total_tickets - general_tickets
  let general_price := (total_revenue - vip_price * vip_tickets) / general_tickets
  general_price

/-- The price of a general admission ticket is $20 given the specific conditions. -/
theorem general_admission_price_is_20 : 
  general_admission_price 320 7500 40 212 = 20 := by
  sorry

end NUMINAMATH_CALUDE_general_admission_price_general_admission_price_is_20_l809_80908


namespace NUMINAMATH_CALUDE_three_tangent_circles_range_l809_80987

/-- Two circles with exactly three common tangents -/
structure ThreeTangentCircles where
  a : ℝ
  b : ℝ
  c1 : (x : ℝ) → (y : ℝ) → Prop := λ x y ↦ (x - a)^2 + y^2 = 1
  c2 : (x : ℝ) → (y : ℝ) → Prop := λ x y ↦ x^2 + y^2 - 2*b*y + b^2 - 4 = 0
  three_tangents : ∃! (p : ℝ × ℝ), c1 p.1 p.2 ∧ c2 p.1 p.2

/-- The range of a² + b² - 6a - 8b for circles with three common tangents -/
theorem three_tangent_circles_range (circles : ThreeTangentCircles) :
  -21 ≤ circles.a^2 + circles.b^2 - 6*circles.a - 8*circles.b ∧
  circles.a^2 + circles.b^2 - 6*circles.a - 8*circles.b ≤ 39 := by
  sorry

end NUMINAMATH_CALUDE_three_tangent_circles_range_l809_80987


namespace NUMINAMATH_CALUDE_percentage_both_correct_l809_80911

theorem percentage_both_correct (p_first : ℝ) (p_second : ℝ) (p_neither : ℝ) 
  (h1 : p_first = 0.75)
  (h2 : p_second = 0.70)
  (h3 : p_neither = 0.20) :
  p_first + p_second - (1 - p_neither) = 0.65 := by
  sorry

end NUMINAMATH_CALUDE_percentage_both_correct_l809_80911


namespace NUMINAMATH_CALUDE_square_cut_perimeter_sum_l809_80930

theorem square_cut_perimeter_sum (s : Real) 
  (h1 : s > 0) 
  (h2 : ∃ (a b c d : Real), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ 
        a + b = 1 ∧ c + d = 1 ∧
        s = 2*(a+b) + 2*(c+d) + 2*(a+c) + 2*(b+d)) :
  s = 8 ∨ s = 10 := by
sorry

end NUMINAMATH_CALUDE_square_cut_perimeter_sum_l809_80930


namespace NUMINAMATH_CALUDE_at_least_one_black_probability_l809_80971

def total_balls : ℕ := 4
def white_balls : ℕ := 2
def black_balls : ℕ := 2
def drawn_balls : ℕ := 2

def probability_at_least_one_black : ℚ := 5 / 6

theorem at_least_one_black_probability :
  probability_at_least_one_black = 
    (Nat.choose total_balls drawn_balls - Nat.choose white_balls drawn_balls) / 
    Nat.choose total_balls drawn_balls :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_black_probability_l809_80971


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l809_80925

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_first_term : a 1 = 1) 
  (h_sum : a 1 + a 2 + a 3 = 12) : 
  a 2 - a 1 = 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l809_80925


namespace NUMINAMATH_CALUDE_bianca_coloring_books_l809_80990

/-- The number of coloring books Bianca initially had -/
def initial_books : ℕ := 45

/-- The number of books Bianca gave away -/
def books_given_away : ℕ := 6

/-- The number of books Bianca bought -/
def books_bought : ℕ := 20

/-- The total number of books Bianca has after the transactions -/
def final_books : ℕ := 59

/-- Theorem stating that the initial number of books is correct -/
theorem bianca_coloring_books : 
  initial_books - books_given_away + books_bought = final_books := by
  sorry

#check bianca_coloring_books

end NUMINAMATH_CALUDE_bianca_coloring_books_l809_80990


namespace NUMINAMATH_CALUDE_deli_sandwich_count_l809_80977

-- Define the types of sandwich components
structure SandwichComponents where
  breads : Nat
  meats : Nat
  cheeses : Nat

-- Define the forbidden combinations
structure ForbiddenCombinations where
  ham_cheddar : Nat
  white_chicken : Nat
  turkey_swiss : Nat

-- Define the function to calculate the number of possible sandwiches
def calculate_sandwiches (components : SandwichComponents) (forbidden : ForbiddenCombinations) : Nat :=
  components.breads * components.meats * components.cheeses - 
  (forbidden.ham_cheddar + forbidden.white_chicken + forbidden.turkey_swiss)

-- Theorem statement
theorem deli_sandwich_count :
  let components := SandwichComponents.mk 5 7 6
  let forbidden := ForbiddenCombinations.mk 5 6 5
  calculate_sandwiches components forbidden = 194 := by
  sorry


end NUMINAMATH_CALUDE_deli_sandwich_count_l809_80977


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l809_80978

theorem sum_of_three_numbers (x y z : ℝ) 
  (sum_xy : x + y = 31)
  (sum_yz : y + z = 41)
  (sum_zx : z + x = 55) :
  x + y + z = 63.5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l809_80978


namespace NUMINAMATH_CALUDE_perfect_square_from_divisibility_l809_80932

theorem perfect_square_from_divisibility (n p : ℕ) : 
  n > 1 → 
  Nat.Prime p → 
  (p - 1) % n = 0 → 
  (n^3 - 1) % p = 0 → 
  ∃ (k : ℕ), 4*p - 3 = k^2 :=
by
  sorry

end NUMINAMATH_CALUDE_perfect_square_from_divisibility_l809_80932


namespace NUMINAMATH_CALUDE_inverse_f_75_l809_80916

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^3 - 6

-- State the theorem
theorem inverse_f_75 : f⁻¹ 75 = 3 := by sorry

end NUMINAMATH_CALUDE_inverse_f_75_l809_80916


namespace NUMINAMATH_CALUDE_median_of_consecutive_integers_with_sum_property_l809_80905

-- Define a set of consecutive integers
def ConsecutiveIntegers (a : ℤ) (n : ℕ) := {i : ℤ | ∃ k : ℕ, k < n ∧ i = a + k}

-- Define the property of sum of nth from beginning and end being 200
def SumProperty (s : Set ℤ) : Prop :=
  ∀ a n, s = ConsecutiveIntegers a n →
    ∀ k, k < n → (a + k) + (a + (n - 1 - k)) = 200

-- Theorem statement
theorem median_of_consecutive_integers_with_sum_property (s : Set ℤ) :
  SumProperty s → ∃ a n, s = ConsecutiveIntegers a n ∧ n % 2 = 1 ∧ 
  (∃ m : ℤ, m ∈ s ∧ (∀ x ∈ s, 2 * (x - m) ≤ n - 1 ∧ 2 * (m - x) ≤ n - 1) ∧ m = 100) :=
sorry

end NUMINAMATH_CALUDE_median_of_consecutive_integers_with_sum_property_l809_80905


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l809_80953

theorem profit_percentage_calculation (selling_price profit : ℝ) :
  selling_price = 850 →
  profit = 215 →
  let cost_price := selling_price - profit
  let profit_percentage := (profit / cost_price) * 100
  ∃ ε > 0, abs (profit_percentage - 33.86) < ε :=
by sorry

end NUMINAMATH_CALUDE_profit_percentage_calculation_l809_80953


namespace NUMINAMATH_CALUDE_no_seven_edge_polyhedron_l809_80999

/-- A polyhedron in three-dimensional space. -/
structure Polyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  euler : vertices - edges + faces = 2
  min_degree : edges * 2 ≥ vertices * 3

/-- Theorem stating that no polyhedron can have exactly seven edges. -/
theorem no_seven_edge_polyhedron :
  ¬∃ (p : Polyhedron), p.edges = 7 := by
  sorry

end NUMINAMATH_CALUDE_no_seven_edge_polyhedron_l809_80999


namespace NUMINAMATH_CALUDE_sin_cos_difference_simplification_l809_80979

theorem sin_cos_difference_simplification :
  Real.sin (72 * π / 180) * Real.cos (12 * π / 180) - 
  Real.cos (72 * π / 180) * Real.sin (12 * π / 180) = 
  Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_cos_difference_simplification_l809_80979


namespace NUMINAMATH_CALUDE_geometric_sequence_terms_l809_80918

theorem geometric_sequence_terms (a₁ aₙ q : ℚ) (n : ℕ) (h₁ : a₁ = 9/8) (h₂ : aₙ = 1/3) (h₃ : q = 2/3) :
  aₙ = a₁ * q^(n - 1) → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_terms_l809_80918


namespace NUMINAMATH_CALUDE_fourth_pentagon_dots_l809_80904

/-- Represents the number of dots in the nth pentagon of the sequence -/
def dots (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else if n = 2 then 6
  else if n = 3 then 16
  else dots (n - 1) + 5 * (n - 1)

/-- The theorem stating that the fourth pentagon has 31 dots -/
theorem fourth_pentagon_dots : dots 4 = 31 := by
  sorry


end NUMINAMATH_CALUDE_fourth_pentagon_dots_l809_80904


namespace NUMINAMATH_CALUDE_truck_departure_time_l809_80934

/-- Proves that given a car traveling at 55 mph and a truck traveling at 65 mph
    on the same road in the same direction, if it takes 6.5 hours for the truck
    to pass the car, then the truck left the station 1 hour after the car. -/
theorem truck_departure_time (car_speed truck_speed : ℝ) (passing_time : ℝ) :
  car_speed = 55 →
  truck_speed = 65 →
  passing_time = 6.5 →
  (truck_speed - car_speed) * passing_time / truck_speed = 1 :=
by sorry

end NUMINAMATH_CALUDE_truck_departure_time_l809_80934


namespace NUMINAMATH_CALUDE_units_digit_of_3968_pow_805_l809_80989

theorem units_digit_of_3968_pow_805 : (3968^805) % 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_3968_pow_805_l809_80989


namespace NUMINAMATH_CALUDE_cloth_sale_calculation_l809_80964

/-- Proves that the number of metres of cloth sold is 500 given the conditions -/
theorem cloth_sale_calculation (total_selling_price : ℕ) (loss_per_metre : ℕ) (cost_price_per_metre : ℕ)
  (h1 : total_selling_price = 18000)
  (h2 : loss_per_metre = 5)
  (h3 : cost_price_per_metre = 41) :
  total_selling_price / (cost_price_per_metre - loss_per_metre) = 500 := by
  sorry

end NUMINAMATH_CALUDE_cloth_sale_calculation_l809_80964


namespace NUMINAMATH_CALUDE_lunch_cost_l809_80938

theorem lunch_cost (x : ℝ) : 
  x + 0.04 * x + 0.06 * x = 110 → x = 100 := by
  sorry

end NUMINAMATH_CALUDE_lunch_cost_l809_80938


namespace NUMINAMATH_CALUDE_point_distance_to_line_l809_80907

/-- The distance from a point (a, 2) to the line x - y + 3 = 0 is 1, where a > 0 -/
def distance_to_line (a : ℝ) : Prop :=
  a > 0 ∧ |a + 1| / Real.sqrt 2 = 1

/-- Theorem: If the distance from (a, 2) to the line x - y + 3 = 0 is 1, then a = √2 - 1 -/
theorem point_distance_to_line (a : ℝ) (h : distance_to_line a) : a = Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_point_distance_to_line_l809_80907


namespace NUMINAMATH_CALUDE_payroll_threshold_proof_l809_80996

/-- Proves that the payroll threshold is $200,000 given the problem conditions --/
theorem payroll_threshold_proof 
  (total_payroll : ℝ) 
  (tax_paid : ℝ) 
  (tax_rate : ℝ) 
  (h1 : total_payroll = 400000)
  (h2 : tax_paid = 400)
  (h3 : tax_rate = 0.002) : 
  ∃ threshold : ℝ, 
    threshold = 200000 ∧ 
    tax_rate * (total_payroll - threshold) = tax_paid :=
by sorry

end NUMINAMATH_CALUDE_payroll_threshold_proof_l809_80996


namespace NUMINAMATH_CALUDE_max_third_side_of_triangle_l809_80949

/-- Given a triangle with two sides of 5 cm and 10 cm, 
    the maximum integer length of the third side is 14 cm. -/
theorem max_third_side_of_triangle (a b c : ℕ) : 
  a = 5 → b = 10 → c ≤ 14 → a + b > c → a + c > b → b + c > a → c ≤ a + b - 1 :=
by sorry

end NUMINAMATH_CALUDE_max_third_side_of_triangle_l809_80949


namespace NUMINAMATH_CALUDE_piggy_bank_coins_l809_80992

theorem piggy_bank_coins (sequence : Fin 6 → ℕ) 
  (h1 : sequence 0 = 72)
  (h2 : sequence 1 = 81)
  (h3 : sequence 2 = 90)
  (h5 : sequence 4 = 108)
  (h6 : sequence 5 = 117)
  (h_arithmetic : ∀ i : Fin 5, sequence (i + 1) - sequence i = sequence 1 - sequence 0) :
  sequence 3 = 99 := by
  sorry

end NUMINAMATH_CALUDE_piggy_bank_coins_l809_80992


namespace NUMINAMATH_CALUDE_winnie_the_pooh_honey_l809_80920

def honey_pot (initial_weight : ℝ) (empty_pot_weight : ℝ) : Prop :=
  ∃ (w1 w2 w3 w4 : ℝ),
    w1 = initial_weight / 2 ∧
    w2 = w1 / 2 ∧
    w3 = w2 / 2 ∧
    w4 = w3 / 2 ∧
    w4 = empty_pot_weight

theorem winnie_the_pooh_honey (empty_pot_weight : ℝ) 
  (h1 : empty_pot_weight = 200) : 
  ∃ (initial_weight : ℝ), 
    honey_pot initial_weight empty_pot_weight ∧ 
    initial_weight - empty_pot_weight = 3000 := by
  sorry

end NUMINAMATH_CALUDE_winnie_the_pooh_honey_l809_80920


namespace NUMINAMATH_CALUDE_max_threshold_price_l809_80982

/-- Represents a company with a product line -/
structure Company where
  num_products : ℕ
  avg_price : ℝ
  min_price : ℝ
  max_price : ℝ
  num_below_threshold : ℕ

/-- The threshold price for a given company -/
def threshold_price (c : Company) : ℝ := sorry

theorem max_threshold_price (c : Company) :
  c.num_products = 25 →
  c.avg_price = 1200 →
  c.min_price = 400 →
  c.max_price = 13200 →
  c.num_below_threshold = 12 →
  threshold_price c ≤ 700 ∧
  ∀ t, t > 700 → ¬(threshold_price c = t) := by
  sorry

#check max_threshold_price

end NUMINAMATH_CALUDE_max_threshold_price_l809_80982


namespace NUMINAMATH_CALUDE_arithmetic_sequence_before_negative_seventeen_l809_80963

/-- 
Given an arithmetic sequence with first term 88 and common difference -3,
prove that the number of terms that appear before -17 is 35.
-/
theorem arithmetic_sequence_before_negative_seventeen :
  let a : ℕ → ℤ := λ n => 88 - 3 * (n - 1)
  ∃ k : ℕ, a k = -17 ∧ k - 1 = 35 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_before_negative_seventeen_l809_80963


namespace NUMINAMATH_CALUDE_parabola_translation_theorem_l809_80940

/-- Represents a parabola of the form y = ax² -/
structure Parabola where
  a : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a translation in 2D space -/
structure Translation where
  right : ℝ
  up : ℝ

/-- Returns true if the given equation represents the parabola after translation -/
def is_translated_parabola (p : Parabola) (t : Translation) (eq : ℝ → ℝ) : Prop :=
  ∀ x, eq x = p.a * (x - t.right)^2 + t.up

/-- Returns true if the given point satisfies the equation -/
def satisfies_equation (pt : Point) (eq : ℝ → ℝ) : Prop :=
  eq pt.x = pt.y

theorem parabola_translation_theorem (p : Parabola) (t : Translation) (pt : Point) :
  is_translated_parabola p t (fun x => -4 * (x - 2)^2 + 3) ∧
  satisfies_equation pt (fun x => -4 * (x - 2)^2 + 3) ∧
  t.right = 2 ∧ t.up = 3 ∧ pt.x = 3 ∧ pt.y = -1 :=
sorry

end NUMINAMATH_CALUDE_parabola_translation_theorem_l809_80940


namespace NUMINAMATH_CALUDE_range_of_a_plus_3b_l809_80993

theorem range_of_a_plus_3b (a b : ℝ) 
  (h1 : -1 ≤ a + b ∧ a + b ≤ 1) 
  (h2 : 1 ≤ a - 2*b ∧ a - 2*b ≤ 3) : 
  -11/3 ≤ a + 3*b ∧ a + 3*b ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_plus_3b_l809_80993


namespace NUMINAMATH_CALUDE_equality_of_polynomials_l809_80943

theorem equality_of_polynomials (a b : ℝ) : 
  (∀ x : ℝ, x^2 - 3*x + 5 = (x - 2)^2 + a*(x - 2) + b) → 
  a + b = 4 := by
sorry

end NUMINAMATH_CALUDE_equality_of_polynomials_l809_80943


namespace NUMINAMATH_CALUDE_product_sequence_sum_l809_80984

theorem product_sequence_sum (a b : ℕ) (h1 : a / 3 = 18) (h2 : b = a - 1) : a + b = 107 := by
  sorry

end NUMINAMATH_CALUDE_product_sequence_sum_l809_80984


namespace NUMINAMATH_CALUDE_fairCoinThreeFlipsOneHead_l809_80923

def fairCoinProbability (n : ℕ) (k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * (1/2)^k * (1/2)^(n-k)

theorem fairCoinThreeFlipsOneHead :
  fairCoinProbability 3 1 = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_fairCoinThreeFlipsOneHead_l809_80923


namespace NUMINAMATH_CALUDE_no_solution_to_inequalities_l809_80901

theorem no_solution_to_inequalities : ¬∃ x : ℝ, (x / 2 ≥ 1 + x) ∧ (3 + 2*x > -3 - 3*x) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_to_inequalities_l809_80901


namespace NUMINAMATH_CALUDE_sally_payment_l809_80955

/-- Calculates the amount Sally needs to pay out of pocket for books -/
def sally_out_of_pocket (given_amount : ℕ) (book_cost : ℕ) (num_students : ℕ) : ℕ :=
  max 0 (book_cost * num_students - given_amount)

/-- Proves that Sally needs to pay $205 out of pocket -/
theorem sally_payment : sally_out_of_pocket 320 15 35 = 205 := by
  sorry

end NUMINAMATH_CALUDE_sally_payment_l809_80955


namespace NUMINAMATH_CALUDE_equation_solution_l809_80975

theorem equation_solution : 
  {x : ℝ | (x^3 - x^2)/(x^2 + 2*x + 1) + x = -2} = {-1/2, 2} := by sorry

end NUMINAMATH_CALUDE_equation_solution_l809_80975


namespace NUMINAMATH_CALUDE_problem_solution_l809_80970

theorem problem_solution (a b c d m n : ℕ+) 
  (h1 : a^2 + b^2 + c^2 + d^2 = 1989)
  (h2 : a + b + c + d = m^2)
  (h3 : max a (max b (max c d)) = n^2) :
  m = 9 ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l809_80970


namespace NUMINAMATH_CALUDE_weight_of_almonds_l809_80957

/-- Given the total weight of nuts and the weight of pecans, 
    calculate the weight of almonds. -/
theorem weight_of_almonds 
  (total_weight : ℝ) 
  (pecan_weight : ℝ) 
  (h1 : total_weight = 0.52) 
  (h2 : pecan_weight = 0.38) : 
  total_weight - pecan_weight = 0.14 := by
sorry

end NUMINAMATH_CALUDE_weight_of_almonds_l809_80957


namespace NUMINAMATH_CALUDE_point_on_line_angle_with_x_axis_line_equation_correct_l809_80961

/-- The equation of a line passing through (2, 2) and making a 60° angle with the x-axis -/
def line_equation (x y : ℝ) : Prop :=
  y = Real.sqrt 3 * x - 2 * Real.sqrt 3 + 2

/-- The point (2, 2) lies on the line -/
theorem point_on_line : line_equation 2 2 := by sorry

/-- The angle between the line and the x-axis is 60° -/
theorem angle_with_x_axis : 
  Real.arctan (Real.sqrt 3) = 60 * π / 180 := by sorry

/-- The line equation is correct -/
theorem line_equation_correct (x y : ℝ) :
  line_equation x y ↔ 
    (∃ k : ℝ, y - 2 = k * (x - 2) ∧ 
              k = Real.tan (60 * π / 180)) := by sorry

end NUMINAMATH_CALUDE_point_on_line_angle_with_x_axis_line_equation_correct_l809_80961


namespace NUMINAMATH_CALUDE_wasted_meat_pounds_l809_80997

def minimum_wage : ℝ := 8
def fruit_veg_cost_per_pound : ℝ := 4
def fruit_veg_wasted : ℝ := 15
def bread_cost_per_pound : ℝ := 1.5
def bread_wasted : ℝ := 60
def janitor_normal_wage : ℝ := 10
def janitor_hours : ℝ := 10
def meat_cost_per_pound : ℝ := 5
def james_work_hours : ℝ := 50

def total_cost : ℝ := james_work_hours * minimum_wage

def fruit_veg_cost : ℝ := fruit_veg_cost_per_pound * fruit_veg_wasted
def bread_cost : ℝ := bread_cost_per_pound * bread_wasted
def janitor_cost : ℝ := janitor_normal_wage * 1.5 * janitor_hours

def known_costs : ℝ := fruit_veg_cost + bread_cost + janitor_cost
def meat_cost : ℝ := total_cost - known_costs

theorem wasted_meat_pounds : meat_cost / meat_cost_per_pound = 20 := by
  sorry

end NUMINAMATH_CALUDE_wasted_meat_pounds_l809_80997


namespace NUMINAMATH_CALUDE_carls_cupcakes_l809_80942

/-- Carl's cupcake selling problem -/
theorem carls_cupcakes (days : ℕ) (cupcakes_per_day : ℕ) (cupcakes_for_bonnie : ℕ) : 
  days = 2 → cupcakes_per_day = 60 → cupcakes_for_bonnie = 24 →
  days * cupcakes_per_day + cupcakes_for_bonnie = 144 := by
  sorry

end NUMINAMATH_CALUDE_carls_cupcakes_l809_80942


namespace NUMINAMATH_CALUDE_difference_of_squares_division_l809_80900

theorem difference_of_squares_division : (245^2 - 205^2) / 40 = 450 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_division_l809_80900


namespace NUMINAMATH_CALUDE_number_calculation_l809_80919

theorem number_calculation (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 10 → (40/100 : ℝ) * N = 120 := by
  sorry

end NUMINAMATH_CALUDE_number_calculation_l809_80919


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l809_80928

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = x + 1) ↔ x ≥ -1 :=
sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l809_80928


namespace NUMINAMATH_CALUDE_complex_equation_solution_l809_80981

theorem complex_equation_solution (z : ℂ) : (Complex.I * z = 1 - Complex.I) → z = -1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l809_80981


namespace NUMINAMATH_CALUDE_same_and_different_signs_l809_80933

theorem same_and_different_signs (a b : ℝ) : 
  (a * b > 0 ↔ (a > 0 ∧ b > 0) ∨ (a < 0 ∧ b < 0)) ∧
  (a * b < 0 ↔ (a > 0 ∧ b < 0) ∨ (a < 0 ∧ b > 0)) :=
by sorry

end NUMINAMATH_CALUDE_same_and_different_signs_l809_80933


namespace NUMINAMATH_CALUDE_min_value_cyclic_fraction_min_value_cyclic_fraction_achievable_l809_80972

theorem min_value_cyclic_fraction (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) : 
  a / b + b / c + c / d + d / a ≥ 4 :=
by sorry

theorem min_value_cyclic_fraction_achievable :
  ∃ (a b c d : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
  a / b + b / c + c / d + d / a = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_cyclic_fraction_min_value_cyclic_fraction_achievable_l809_80972


namespace NUMINAMATH_CALUDE_factorization_of_2x_squared_minus_8_l809_80985

theorem factorization_of_2x_squared_minus_8 (x : ℝ) : 2 * x^2 - 8 = 2 * (x - 2) * (x + 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2x_squared_minus_8_l809_80985


namespace NUMINAMATH_CALUDE_radical_calculations_l809_80926

theorem radical_calculations :
  (∃ x y : ℝ, x^2 = 3 ∧ y^2 = 2 ∧
    (Real.sqrt 48 + Real.sqrt 8 - Real.sqrt 18 - Real.sqrt 12 = 2*x - y)) ∧
  (∃ a b c : ℝ, a^2 = 2 ∧ b^2 = 3 ∧ c^2 = 6 ∧
    (2*(a + b) - (b - a)^2 = 2*a + 2*b + 2*c - 5)) :=
by sorry

end NUMINAMATH_CALUDE_radical_calculations_l809_80926


namespace NUMINAMATH_CALUDE_flour_mass_acceptance_l809_80954

-- Define the labeled mass and uncertainty
def labeled_mass : ℝ := 35
def uncertainty : ℝ := 0.25

-- Define the acceptable range
def min_acceptable : ℝ := labeled_mass - uncertainty
def max_acceptable : ℝ := labeled_mass + uncertainty

-- Define the masses of the flour bags
def mass_A : ℝ := 34.70
def mass_B : ℝ := 34.80
def mass_C : ℝ := 35.30
def mass_D : ℝ := 35.51

-- Theorem to prove
theorem flour_mass_acceptance :
  (min_acceptable ≤ mass_B ∧ mass_B ≤ max_acceptable) ∧
  (mass_A < min_acceptable ∨ mass_A > max_acceptable) ∧
  (mass_C < min_acceptable ∨ mass_C > max_acceptable) ∧
  (mass_D < min_acceptable ∨ mass_D > max_acceptable) := by
  sorry

end NUMINAMATH_CALUDE_flour_mass_acceptance_l809_80954


namespace NUMINAMATH_CALUDE_empty_solution_set_range_l809_80991

theorem empty_solution_set_range (a : ℝ) : 
  (∀ x : ℝ, (a^2 - 4) * x^2 + (a + 2) * x - 1 < 0) ↔ (-2 < a ∧ a ≤ 6/5) :=
sorry

end NUMINAMATH_CALUDE_empty_solution_set_range_l809_80991


namespace NUMINAMATH_CALUDE_all_faces_dirty_l809_80965

/-- Represents the state of a wise man's face -/
inductive FaceState
| Clean
| Dirty

/-- Represents a wise man -/
structure WiseMan :=
  (id : Nat)
  (faceState : FaceState)

/-- Represents the knowledge of a wise man about the others' faces -/
def Knowledge := WiseMan → FaceState

/-- Represents whether a wise man is laughing -/
def isLaughing (w : WiseMan) (k : Knowledge) : Prop :=
  ∃ (other : WiseMan), k other = FaceState.Dirty

/-- The main theorem -/
theorem all_faces_dirty 
  (men : Finset WiseMan) 
  (h_three_men : men.card = 3) 
  (k : WiseMan → Knowledge) 
  (h_correct_knowledge : ∀ (w₁ w₂ : WiseMan), w₁ ≠ w₂ → k w₁ w₂ = w₂.faceState) 
  (h_all_laughing : ∀ (w : WiseMan), w ∈ men → isLaughing w (k w)) :
  ∀ (w : WiseMan), w ∈ men → w.faceState = FaceState.Dirty :=
sorry

end NUMINAMATH_CALUDE_all_faces_dirty_l809_80965


namespace NUMINAMATH_CALUDE_parabola_point_coordinates_l809_80922

theorem parabola_point_coordinates :
  ∀ (x y : ℝ),
  (y = 2 * x^2) →                          -- M is on the parabola y = 2x^2
  (x > 0 ∧ y > 0) →                        -- M is in the first quadrant
  ((x - 0)^2 + (y - 1/8)^2 = (1/4)^2) →    -- Distance from M to focus is 1/4
  (x = Real.sqrt 2 / 8 ∧ y = 1/16) := by
sorry

end NUMINAMATH_CALUDE_parabola_point_coordinates_l809_80922


namespace NUMINAMATH_CALUDE_roof_ratio_l809_80967

theorem roof_ratio (length width : ℝ) 
  (area_eq : length * width = 676)
  (diff_eq : length - width = 39) :
  length / width = 4 :=
by sorry

end NUMINAMATH_CALUDE_roof_ratio_l809_80967


namespace NUMINAMATH_CALUDE_paige_albums_l809_80950

def number_of_albums (total_pictures : ℕ) (first_album_pictures : ℕ) (pictures_per_album : ℕ) : ℕ :=
  (total_pictures - first_album_pictures) / pictures_per_album

theorem paige_albums : number_of_albums 35 14 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_paige_albums_l809_80950


namespace NUMINAMATH_CALUDE_x_cube_x_x_square_l809_80969

theorem x_cube_x_x_square (x : ℝ) (h : -1 < x ∧ x < 0) : x^3 < x ∧ x < x^2 := by
  sorry

end NUMINAMATH_CALUDE_x_cube_x_x_square_l809_80969


namespace NUMINAMATH_CALUDE_total_cost_is_598_l809_80939

/-- The cost of 1 kg of flour in dollars -/
def flour_cost : ℝ := 23

/-- The cost relationship between mangos and rice -/
def mango_rice_relation (mango_cost rice_cost : ℝ) : Prop :=
  10 * mango_cost = rice_cost * 10

/-- The cost relationship between flour and rice -/
def flour_rice_relation (rice_cost : ℝ) : Prop :=
  6 * flour_cost = 2 * rice_cost

/-- The total cost of the given quantities of mangos, rice, and flour -/
def total_cost (mango_cost rice_cost : ℝ) : ℝ :=
  4 * mango_cost + 3 * rice_cost + 5 * flour_cost

/-- Theorem stating the total cost is $598 given the conditions -/
theorem total_cost_is_598 (mango_cost rice_cost : ℝ) 
  (h1 : mango_rice_relation mango_cost rice_cost)
  (h2 : flour_rice_relation rice_cost) : 
  total_cost mango_cost rice_cost = 598 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_598_l809_80939


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l809_80947

theorem imaginary_part_of_complex_fraction (i : ℂ) :
  i * i = -1 →
  Complex.im ((1 + i) / (1 - i)) = 1 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l809_80947


namespace NUMINAMATH_CALUDE_sector_to_inscribed_circle_area_ratio_l809_80960

/-- Given a sector with a central angle of 120° and its inscribed circle,
    the ratio of the area of the sector to the area of the inscribed circle
    is (7 + 4√3) / 9. -/
theorem sector_to_inscribed_circle_area_ratio :
  ∀ (R r : ℝ), R > 0 → r > 0 →
  (2 * π / 3 : ℝ) = 2 * Real.arcsin (r / R) →
  (π * R^2 * (2 * π / 3) / (2 * π)) / (π * r^2) = (7 + 4 * Real.sqrt 3) / 9 := by
  sorry

end NUMINAMATH_CALUDE_sector_to_inscribed_circle_area_ratio_l809_80960


namespace NUMINAMATH_CALUDE_juan_reads_9000_pages_l809_80973

/-- Calculates the total pages Juan can read from three books given their page counts, reading rates, and lunch time constraints. -/
def total_pages_read (book1_pages book2_pages book3_pages : ℕ) 
                     (book1_rate book2_rate book3_rate : ℕ) 
                     (lunch_time : ℕ) : ℕ :=
  let book1_read_time := book1_pages / book1_rate
  let book2_read_time := book2_pages / book2_rate
  let book3_read_time := book3_pages / book3_rate
  let book1_lunch_time := book1_read_time / 2
  let book2_lunch_time := book2_read_time / 2
  let book3_lunch_time := book3_read_time / 2
  let total_lunch_time := book1_lunch_time + book2_lunch_time + book3_lunch_time
  let remaining_time1 := book1_lunch_time - lunch_time
  let remaining_time2 := book2_lunch_time
  let remaining_time3 := book3_lunch_time
  remaining_time1 * book1_rate + remaining_time2 * book2_rate + remaining_time3 * book3_rate

/-- Theorem stating that given the specific conditions in the problem, Juan can read 9000 pages. -/
theorem juan_reads_9000_pages : 
  total_pages_read 4000 6000 8000 60 40 30 4 = 9000 := by
  sorry

end NUMINAMATH_CALUDE_juan_reads_9000_pages_l809_80973


namespace NUMINAMATH_CALUDE_simplify_expression_l809_80974

theorem simplify_expression (x : ℝ) : 8*x + 15 - 3*x + 5 * 7 = 5*x + 50 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l809_80974


namespace NUMINAMATH_CALUDE_olivia_spent_25_dollars_l809_80906

/-- The amount Olivia spent at the supermarket -/
def amount_spent (initial_amount remaining_amount : ℕ) : ℕ :=
  initial_amount - remaining_amount

/-- Theorem stating that Olivia spent 25 dollars -/
theorem olivia_spent_25_dollars (initial_amount remaining_amount : ℕ) 
  (h1 : initial_amount = 54)
  (h2 : remaining_amount = 29) : 
  amount_spent initial_amount remaining_amount = 25 := by
  sorry

end NUMINAMATH_CALUDE_olivia_spent_25_dollars_l809_80906


namespace NUMINAMATH_CALUDE_roses_unchanged_l809_80915

def initial_roses : ℕ := 12
def initial_orchids : ℕ := 2
def final_orchids : ℕ := 21
def cut_orchids : ℕ := 19

theorem roses_unchanged (h : final_orchids - cut_orchids = initial_orchids) :
  initial_roses = initial_roses := by sorry

end NUMINAMATH_CALUDE_roses_unchanged_l809_80915


namespace NUMINAMATH_CALUDE_karen_homework_paragraphs_l809_80929

/-- Represents the homework assignment structure -/
structure HomeworkAssignment where
  shortAnswerTime : ℕ
  paragraphTime : ℕ
  essayTime : ℕ
  essayCount : ℕ
  shortAnswerCount : ℕ
  totalTime : ℕ

/-- Calculates the number of paragraphs in the homework assignment -/
def calculateParagraphs (hw : HomeworkAssignment) : ℕ :=
  (hw.totalTime - hw.essayCount * hw.essayTime - hw.shortAnswerCount * hw.shortAnswerTime) / hw.paragraphTime

/-- Theorem stating that Karen's homework assignment results in 5 paragraphs -/
theorem karen_homework_paragraphs :
  let hw : HomeworkAssignment := {
    shortAnswerTime := 3,
    paragraphTime := 15,
    essayTime := 60,
    essayCount := 2,
    shortAnswerCount := 15,
    totalTime := 240
  }
  calculateParagraphs hw = 5 := by sorry

end NUMINAMATH_CALUDE_karen_homework_paragraphs_l809_80929


namespace NUMINAMATH_CALUDE_power_of_p_is_one_l809_80912

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The property of being a positive even integer with a positive units digit -/
def isPositiveEvenWithPositiveUnitsDigit (p : ℕ) : Prop :=
  p > 0 ∧ p % 2 = 0 ∧ unitsDigit p > 0

theorem power_of_p_is_one (p : ℕ) (k : ℕ) 
  (h1 : isPositiveEvenWithPositiveUnitsDigit p)
  (h2 : unitsDigit (p + 1) = 7)
  (h3 : unitsDigit (p^3) - unitsDigit (p^k) = 0) :
  k = 1 := by sorry

end NUMINAMATH_CALUDE_power_of_p_is_one_l809_80912


namespace NUMINAMATH_CALUDE_negation_existential_real_l809_80983

theorem negation_existential_real (f : ℝ → ℝ) :
  (¬ ∃ x : ℝ, f x < 0) ↔ (∀ x : ℝ, f x ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_existential_real_l809_80983


namespace NUMINAMATH_CALUDE_maaza_liters_l809_80910

/-- The number of liters of Pepsi -/
def pepsi : ℕ := 144

/-- The number of liters of Sprite -/
def sprite : ℕ := 368

/-- The total number of cans required -/
def total_cans : ℕ := 261

/-- The capacity of each can in liters -/
def can_capacity : ℕ := Nat.gcd pepsi sprite

theorem maaza_liters : ∃ M : ℕ, 
  M + pepsi + sprite = total_cans * can_capacity ∧ 
  M = 3664 := by
  sorry

end NUMINAMATH_CALUDE_maaza_liters_l809_80910


namespace NUMINAMATH_CALUDE_right_square_prism_properties_l809_80931

/-- Right square prism -/
structure RightSquarePrism where
  base_edge : ℝ
  height : ℝ

/-- Calculates the lateral area of a right square prism -/
def lateral_area (p : RightSquarePrism) : ℝ :=
  4 * p.base_edge * p.height

/-- Calculates the volume of a right square prism -/
def volume (p : RightSquarePrism) : ℝ :=
  p.base_edge ^ 2 * p.height

theorem right_square_prism_properties :
  ∃ (p : RightSquarePrism), p.base_edge = 3 ∧ p.height = 2 ∧
    lateral_area p = 24 ∧ volume p = 18 := by
  sorry

end NUMINAMATH_CALUDE_right_square_prism_properties_l809_80931


namespace NUMINAMATH_CALUDE_minute_hand_distance_l809_80956

theorem minute_hand_distance (hand_length : ℝ) (time : ℝ) : 
  hand_length = 8 → time = 45 → 
  2 * π * hand_length * (time / 60) = 12 * π := by
sorry

end NUMINAMATH_CALUDE_minute_hand_distance_l809_80956


namespace NUMINAMATH_CALUDE_line_through_origin_and_intersection_l809_80986

/-- The equation of the line passing through the origin and the intersection of two given lines -/
theorem line_through_origin_and_intersection (x y : ℝ) : 
  (x - 2*y + 2 = 0) →  -- Line 1 equation
  (2*x - y - 2 = 0) →  -- Line 2 equation
  (∃ t : ℝ, x = t ∧ y = t) -- Equation of the line y = x in parametric form
  := by sorry

end NUMINAMATH_CALUDE_line_through_origin_and_intersection_l809_80986


namespace NUMINAMATH_CALUDE_pirate_treasure_problem_l809_80927

theorem pirate_treasure_problem :
  let n : ℕ := 8  -- Total number of islands
  let k : ℕ := 5  -- Number of islands with treasure
  let p_treasure : ℚ := 1/6  -- Probability of an island having treasure and no traps
  let p_neither : ℚ := 2/3  -- Probability of an island having neither treasure nor traps
  
  (Nat.choose n k : ℚ) * p_treasure^k * p_neither^(n - k) = 7/3328 := by
  sorry

end NUMINAMATH_CALUDE_pirate_treasure_problem_l809_80927


namespace NUMINAMATH_CALUDE_john_total_spend_l809_80936

-- Define the prices and quantities
def tshirt_price : ℝ := 20
def tshirt_quantity : ℕ := 3
def pants_price : ℝ := 50
def pants_quantity : ℕ := 2
def jacket_original_price : ℝ := 80
def jacket_discount : ℝ := 0.25
def hat_price : ℝ := 15
def shoes_original_price : ℝ := 60
def shoes_discount : ℝ := 0.10

-- Define the total cost function
def total_cost : ℝ :=
  (tshirt_price * tshirt_quantity) +
  (pants_price * pants_quantity) +
  (jacket_original_price * (1 - jacket_discount)) +
  hat_price +
  (shoes_original_price * (1 - shoes_discount))

-- Theorem to prove
theorem john_total_spend : total_cost = 289 := by
  sorry

end NUMINAMATH_CALUDE_john_total_spend_l809_80936


namespace NUMINAMATH_CALUDE_prob_TT_after_second_H_l809_80914

/-- A fair coin flip sequence that stops when two consecutive flips are the same -/
inductive CoinFlipSequence
  | HH
  | TT
  | HTH : CoinFlipSequence → CoinFlipSequence
  | HTT : CoinFlipSequence

/-- The probability of a coin flip sequence -/
def prob : CoinFlipSequence → ℚ
  | CoinFlipSequence.HH => 1/4
  | CoinFlipSequence.TT => 1/4
  | CoinFlipSequence.HTH s => (1/8) * prob s
  | CoinFlipSequence.HTT => 1/8

/-- The probability of getting two tails in a row but seeing a second head before seeing a second tail -/
def probTTAfterSecondH : ℚ := prob CoinFlipSequence.HTT

theorem prob_TT_after_second_H : probTTAfterSecondH = 1/24 := by
  sorry

end NUMINAMATH_CALUDE_prob_TT_after_second_H_l809_80914


namespace NUMINAMATH_CALUDE_ellipse_with_y_axis_focus_l809_80952

/-- Given that θ is an interior angle of a triangle ABC and sin θ + cos θ = 3/4,
    prove that x^2 * sin θ - y^2 * cos θ = 1 represents an ellipse with focus on the y-axis -/
theorem ellipse_with_y_axis_focus (θ : Real) (x y : Real) 
  (h1 : 0 < θ ∧ θ < π) -- θ is an interior angle of a triangle
  (h2 : Real.sin θ + Real.cos θ = 3/4) -- given condition
  (h3 : x^2 * Real.sin θ - y^2 * Real.cos θ = 1) -- equation of the curve
  : ∃ (a b : Real), 
    0 < b ∧ b < a ∧ 
    (x^2 / a^2) + (y^2 / b^2) = 1 ∧ 
    (a^2 - b^2) / a^2 > 0 :=
sorry

end NUMINAMATH_CALUDE_ellipse_with_y_axis_focus_l809_80952


namespace NUMINAMATH_CALUDE_find_y_value_l809_80995

theorem find_y_value (x : ℝ) (y : ℝ) (h1 : 3 * x = 0.75 * y) (h2 : x = 20) : y = 80 := by
  sorry

end NUMINAMATH_CALUDE_find_y_value_l809_80995


namespace NUMINAMATH_CALUDE_A_intersect_B_l809_80945

def A : Set ℤ := {-2, 0, 2}

def f (x : ℤ) : ℤ := Int.natAbs x

def B : Set ℤ := f '' A

theorem A_intersect_B : A ∩ B = {0, 2} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l809_80945


namespace NUMINAMATH_CALUDE_triangle_area_at_least_three_l809_80937

/-- A type representing a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The area of a triangle formed by three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ := sorry

/-- A set of five points in a plane -/
def FivePoints : Type := Fin 5 → Point

theorem triangle_area_at_least_three (points : FivePoints) 
  (h : ∀ (i j k : Fin 5), i ≠ j → j ≠ k → i ≠ k → 
       triangleArea (points i) (points j) (points k) ≥ 2) :
  ∃ (i j k : Fin 5), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    triangleArea (points i) (points j) (points k) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_at_least_three_l809_80937


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l809_80948

theorem hyperbola_focal_length (x y : ℝ) :
  x^2 / 7 - y^2 / 3 = 1 → ∃ (f : ℝ), f = 2 * Real.sqrt 10 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l809_80948


namespace NUMINAMATH_CALUDE_block_depth_l809_80951

theorem block_depth (cube_volume : ℕ) (length width : ℕ) (fewer_cubes : ℕ) (d : ℕ) : 
  cube_volume = 5 →
  length = 7 →
  width = 7 →
  fewer_cubes = 194 →
  length * width * d * cube_volume - fewer_cubes * cube_volume = length * width * (d - 1) * cube_volume →
  d = 5 :=
by sorry

end NUMINAMATH_CALUDE_block_depth_l809_80951
