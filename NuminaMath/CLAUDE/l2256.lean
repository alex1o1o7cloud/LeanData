import Mathlib

namespace NUMINAMATH_CALUDE_four_tangent_lines_with_equal_intercepts_l2256_225645

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in the 2D plane represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop :=
  let (x₀, y₀) := c.center
  let d := |l.a * x₀ + l.b * y₀ + l.c| / Real.sqrt (l.a^2 + l.b^2)
  d = c.radius

/-- Check if a line has equal intercepts on both axes -/
def hasEqualIntercepts (l : Line) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ -l.c/l.a = -l.c/l.b

/-- The main theorem -/
theorem four_tangent_lines_with_equal_intercepts :
  let c : Circle := { center := (3, 3), radius := Real.sqrt 8 }
  ∃ (lines : Finset Line),
    lines.card = 4 ∧
    ∀ l ∈ lines, isTangent l c ∧ hasEqualIntercepts l ∧
    ∀ l', isTangent l' c → hasEqualIntercepts l' → l' ∈ lines :=
sorry

end NUMINAMATH_CALUDE_four_tangent_lines_with_equal_intercepts_l2256_225645


namespace NUMINAMATH_CALUDE_interview_room_occupancy_l2256_225689

/-- Given a waiting room and an interview room, prove that the number of people in the interview room is 5. -/
theorem interview_room_occupancy (waiting_room interview_room : ℕ) : interview_room = 5 :=
  by
  -- Define the initial number of people in the waiting room
  have initial_waiting : waiting_room = 22 := by sorry
  
  -- Define the number of new arrivals
  have new_arrivals : ℕ := 3
  
  -- Define the relationship between waiting room and interview room after new arrivals
  have after_arrivals : waiting_room + new_arrivals = 5 * interview_room := by sorry
  
  sorry -- Proof goes here

end NUMINAMATH_CALUDE_interview_room_occupancy_l2256_225689


namespace NUMINAMATH_CALUDE_problem_solution_l2256_225653

-- Define the star operation
def star (a b : ℝ) : ℝ := a * b + 3 * b - 2 * a

-- State the theorem
theorem problem_solution :
  ∀ y : ℝ, star 4 y = 30 → y = 38 / 7 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2256_225653


namespace NUMINAMATH_CALUDE_factory_output_increase_l2256_225649

theorem factory_output_increase (planned_output actual_output : ℝ) 
  (h1 : planned_output = 20)
  (h2 : actual_output = 24) : 
  (actual_output - planned_output) / planned_output = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_factory_output_increase_l2256_225649


namespace NUMINAMATH_CALUDE_solution_set_when_m_is_one_inequality_holds_iff_m_in_range_l2256_225662

-- Define the function f
def f (x m : ℝ) : ℝ := |x - m| - |x + 3*m|

-- Theorem for part I
theorem solution_set_when_m_is_one :
  {x : ℝ | f x 1 ≥ 1} = {x : ℝ | x ≤ -3/2} := by sorry

-- Theorem for part II
theorem inequality_holds_iff_m_in_range :
  (∀ (x t : ℝ), f x m < |2 + t| + |t - 1|) ↔ (0 < m ∧ m < 3/4) := by sorry

end NUMINAMATH_CALUDE_solution_set_when_m_is_one_inequality_holds_iff_m_in_range_l2256_225662


namespace NUMINAMATH_CALUDE_pine_saplings_in_sample_l2256_225632

theorem pine_saplings_in_sample 
  (total_saplings : ℕ) 
  (pine_saplings : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_saplings = 20000) 
  (h2 : pine_saplings = 4000) 
  (h3 : sample_size = 100) : 
  (sample_size * pine_saplings) / total_saplings = 20 := by
sorry

end NUMINAMATH_CALUDE_pine_saplings_in_sample_l2256_225632


namespace NUMINAMATH_CALUDE_chess_tournament_games_l2256_225610

theorem chess_tournament_games (n : ℕ) (h : n = 14) : 
  (n.choose 2) = 91 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l2256_225610


namespace NUMINAMATH_CALUDE_coal_division_l2256_225603

/-- Given 3 tons of coal divided equally into 5 parts, prove the fraction and amount of each part -/
theorem coal_division (total_coal : ℝ) (num_parts : ℕ) 
  (h1 : total_coal = 3)
  (h2 : num_parts = 5) :
  (1 : ℝ) / num_parts = 1 / 5 ∧ 
  total_coal / num_parts = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_coal_division_l2256_225603


namespace NUMINAMATH_CALUDE_elisa_books_problem_l2256_225644

theorem elisa_books_problem :
  ∀ (total science math lit : ℕ),
  science = 24 →
  total = science + math + lit →
  total < 100 →
  (math + 1) * 9 = total + 1 →
  lit * 4 = total + 1 →
  math = 7 :=
by sorry

end NUMINAMATH_CALUDE_elisa_books_problem_l2256_225644


namespace NUMINAMATH_CALUDE_solution_implies_expression_value_l2256_225697

theorem solution_implies_expression_value
  (a b : ℝ)
  (h : a * (-2) - b = 1) :
  4 * a + 2 * b + 7 = 5 :=
by sorry

end NUMINAMATH_CALUDE_solution_implies_expression_value_l2256_225697


namespace NUMINAMATH_CALUDE_savings_percentage_l2256_225626

/-- Proves that a person saves 20% of their salary given specific conditions -/
theorem savings_percentage (salary : ℝ) (savings_after_increase : ℝ) 
  (h1 : salary = 6500)
  (h2 : savings_after_increase = 260)
  (h3 : ∃ (original_expenses : ℝ), 
    salary = original_expenses + (salary * 0.2) 
    ∧ savings_after_increase = salary - (original_expenses * 1.2)) :
  (salary - savings_after_increase) / salary * 100 = 80 := by
  sorry

end NUMINAMATH_CALUDE_savings_percentage_l2256_225626


namespace NUMINAMATH_CALUDE_problem_statement_l2256_225692

-- Define the sets A and B
def A : Set ℝ := Set.Ioo (-2) 2
def B (a : ℝ) : Set ℝ := Set.Ioo a (1 - a)

-- State the theorem
theorem problem_statement (a : ℝ) (h : a < 0) :
  (A ∪ B a = B a → a ≤ -2) ∧
  (A ∩ B a = B a → a ≥ -1) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2256_225692


namespace NUMINAMATH_CALUDE_circle_equation_proof_l2256_225671

/-- A circle with center (h, k) and radius r is represented by the equation (x - h)² + (y - k)² = r² --/
def is_circle (h k r : ℝ) (f : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, f x y ↔ (x - h)^2 + (y - k)^2 = r^2

/-- A circle is tangent to the x-axis if its distance from the x-axis equals its radius --/
def tangent_to_x_axis (h k r : ℝ) : Prop := k = r

theorem circle_equation_proof (x y : ℝ) :
  let h : ℝ := 2
  let k : ℝ := 1
  let f : ℝ → ℝ → Prop := λ x y ↦ (x - 2)^2 + (y - 1)^2 = 1
  is_circle h k 1 f ∧ tangent_to_x_axis h k 1 := by sorry

end NUMINAMATH_CALUDE_circle_equation_proof_l2256_225671


namespace NUMINAMATH_CALUDE_downstream_distance_l2256_225606

-- Define the given constants
def boat_speed : ℝ := 22
def stream_speed : ℝ := 5
def time_downstream : ℝ := 2

-- Define the theorem
theorem downstream_distance :
  let effective_speed := boat_speed + stream_speed
  effective_speed * time_downstream = 54 := by
  sorry

end NUMINAMATH_CALUDE_downstream_distance_l2256_225606


namespace NUMINAMATH_CALUDE_bob_improvement_percentage_l2256_225631

def bob_time : ℝ := 640
def sister_time : ℝ := 557

theorem bob_improvement_percentage :
  let time_difference := bob_time - sister_time
  let percentage_improvement := (time_difference / bob_time) * 100
  ∃ ε > 0, abs (percentage_improvement - 12.97) < ε :=
by sorry

end NUMINAMATH_CALUDE_bob_improvement_percentage_l2256_225631


namespace NUMINAMATH_CALUDE_rounding_and_multiplication_l2256_225614

/-- Round a number to the nearest significant figure -/
def roundToSignificantFigure (x : ℝ) : ℝ := sorry

/-- Round a number up to the nearest hundred -/
def roundUpToHundred (x : ℝ) : ℕ := sorry

/-- The main theorem -/
theorem rounding_and_multiplication :
  let a := 0.000025
  let b := 6546300
  let rounded_a := roundToSignificantFigure a
  let rounded_b := roundToSignificantFigure b
  let product := rounded_a * rounded_b
  roundUpToHundred product = 200 := by sorry

end NUMINAMATH_CALUDE_rounding_and_multiplication_l2256_225614


namespace NUMINAMATH_CALUDE_remainder_problem_l2256_225615

theorem remainder_problem (n : ℕ) (a b c d : ℕ) : 
  n > 0 → 
  n = 102 * a + b → 
  n = 103 * c + d → 
  0 ≤ b → b < 102 → 
  0 ≤ d → d < 103 → 
  a + d = 20 → 
  b = 20 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2256_225615


namespace NUMINAMATH_CALUDE_jose_share_of_profit_l2256_225605

/-- Calculates the share of profit for an investor based on their investment, time period, and total profit --/
def calculate_share_of_profit (investment : ℕ) (months : ℕ) (total_investment_months : ℕ) (total_profit : ℕ) : ℕ :=
  (investment * months * total_profit) / total_investment_months

theorem jose_share_of_profit (tom_investment : ℕ) (tom_months : ℕ) (jose_investment : ℕ) (jose_months : ℕ) (total_profit : ℕ)
  (h1 : tom_investment = 30000)
  (h2 : tom_months = 12)
  (h3 : jose_investment = 45000)
  (h4 : jose_months = 10)
  (h5 : total_profit = 63000) :
  calculate_share_of_profit jose_investment jose_months (tom_investment * tom_months + jose_investment * jose_months) total_profit = 35000 :=
by sorry

end NUMINAMATH_CALUDE_jose_share_of_profit_l2256_225605


namespace NUMINAMATH_CALUDE_jelly_bean_probability_l2256_225656

/-- The probability of selecting a non-red jelly bean from a bag -/
theorem jelly_bean_probability : 
  let red : ℕ := 7
  let green : ℕ := 9
  let yellow : ℕ := 10
  let blue : ℕ := 12
  let purple : ℕ := 5
  let total : ℕ := red + green + yellow + blue + purple
  let non_red : ℕ := green + yellow + blue + purple
  (non_red : ℚ) / total = 36 / 43 := by
  sorry

end NUMINAMATH_CALUDE_jelly_bean_probability_l2256_225656


namespace NUMINAMATH_CALUDE_betty_order_cost_l2256_225677

/-- The total cost of Betty's order -/
def total_cost (slippers_quantity : ℕ) (slippers_price : ℚ) 
               (lipstick_quantity : ℕ) (lipstick_price : ℚ) 
               (hair_color_quantity : ℕ) (hair_color_price : ℚ) : ℚ :=
  slippers_quantity * slippers_price + 
  lipstick_quantity * lipstick_price + 
  hair_color_quantity * hair_color_price

/-- Theorem stating that Betty's total order cost is $44 -/
theorem betty_order_cost : 
  total_cost 6 (5/2) 4 (5/4) 8 3 = 44 := by
  sorry

end NUMINAMATH_CALUDE_betty_order_cost_l2256_225677


namespace NUMINAMATH_CALUDE_square_sum_geq_product_sum_l2256_225695

theorem square_sum_geq_product_sum (a b c : ℝ) :
  a^2 + b^2 + c^2 ≥ a*b + b*c + c*a ∧
  (a^2 + b^2 + c^2 = a*b + b*c + c*a ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_square_sum_geq_product_sum_l2256_225695


namespace NUMINAMATH_CALUDE_units_digit_of_4539_pow_201_l2256_225687

theorem units_digit_of_4539_pow_201 : (4539^201) % 10 = 9 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_4539_pow_201_l2256_225687


namespace NUMINAMATH_CALUDE_quadratic_polynomial_conditions_l2256_225639

theorem quadratic_polynomial_conditions (p : ℝ → ℝ) : 
  (∀ x, p x = (7/8) * x^2 - (13/4) * x + 3) →
  p (-2) = 13 ∧ p 0 = 3 ∧ p 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_conditions_l2256_225639


namespace NUMINAMATH_CALUDE_jumping_rooks_remainder_l2256_225629

/-- The number of ways to place 2n jumping rooks on an n×n chessboard 
    such that each rook attacks exactly two other rooks. -/
def f (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | 1 => 0
  | 2 => 1
  | 3 => 6
  | n + 1 => n.choose 2 * (2 * f n + n * f (n - 1))

/-- The main theorem stating that the number of ways to place 16 jumping rooks
    on an 8×8 chessboard, with each rook attacking exactly two others,
    when divided by 1000, gives a remainder of 530. -/
theorem jumping_rooks_remainder : f 8 % 1000 = 530 := by
  sorry


end NUMINAMATH_CALUDE_jumping_rooks_remainder_l2256_225629


namespace NUMINAMATH_CALUDE_smallest_b_for_composite_l2256_225612

theorem smallest_b_for_composite (b : ℕ) (h : b = 8) :
  (∀ x : ℤ, ∃ y z : ℤ, y ≠ 1 ∧ z ≠ 1 ∧ y * z = x^4 + b^4) ∧
  (∀ b' : ℕ, 0 < b' ∧ b' < b →
    ∃ x : ℤ, ∀ y z : ℤ, (y * z = x^4 + b'^4) → (y = 1 ∨ z = 1)) :=
sorry

end NUMINAMATH_CALUDE_smallest_b_for_composite_l2256_225612


namespace NUMINAMATH_CALUDE_newspaper_recycling_profit_l2256_225669

/-- Calculates the amount of money made from recycling stolen newspapers over a period of time. -/
def recycling_profit (weekday_paper_weight : ℚ) (sunday_paper_weight : ℚ) 
  (papers_per_day : ℕ) (num_weeks : ℕ) (recycling_rate : ℚ) : ℚ :=
  let weekly_weight := (6 * weekday_paper_weight + sunday_paper_weight) * papers_per_day
  let total_weight := weekly_weight * num_weeks
  let total_tons := total_weight / 2000
  total_tons * recycling_rate

/-- Theorem stating that under the given conditions, the profit from recycling stolen newspapers is $100. -/
theorem newspaper_recycling_profit :
  recycling_profit (8/16) (16/16) 250 10 20 = 100 := by
  sorry

end NUMINAMATH_CALUDE_newspaper_recycling_profit_l2256_225669


namespace NUMINAMATH_CALUDE_function_equation_solution_l2256_225688

/-- A function satisfying the given functional equation -/
def satisfies_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (f x * f y - f (x * y)) / 2 = x + y + 1

/-- The main theorem stating that any function satisfying the equation must be f(x) = x + 2 -/
theorem function_equation_solution (f : ℝ → ℝ) (h : satisfies_equation f) : 
  ∀ x : ℝ, f x = x + 2 := by
  sorry

end NUMINAMATH_CALUDE_function_equation_solution_l2256_225688


namespace NUMINAMATH_CALUDE_function_property_l2256_225641

/-- Given a function f(x) = (ln x - k - 1)x where k is a real number and x > 1,
    prove that if x₁ ≠ x₂ and f(x₁) = f(x₂), then x₁x₂ < e^(2k) -/
theorem function_property (k : ℝ) :
  let f : ℝ → ℝ := λ x => (Real.log x - k - 1) * x
  ∀ x₁ x₂, x₁ > 1 → x₂ > 1 → x₁ ≠ x₂ → f x₁ = f x₂ → x₁ * x₂ < Real.exp (2 * k) := by
  sorry


end NUMINAMATH_CALUDE_function_property_l2256_225641


namespace NUMINAMATH_CALUDE_total_lateness_l2256_225696

/-- Given a student who is 20 minutes late and four other students who are each 10 minutes later than the first student, 
    the total time of lateness for all five students is 140 minutes. -/
theorem total_lateness (charlize_lateness : ℕ) (classmates_count : ℕ) (additional_lateness : ℕ) : 
  charlize_lateness = 20 →
  classmates_count = 4 →
  additional_lateness = 10 →
  charlize_lateness + classmates_count * (charlize_lateness + additional_lateness) = 140 :=
by sorry

end NUMINAMATH_CALUDE_total_lateness_l2256_225696


namespace NUMINAMATH_CALUDE_water_remaining_l2256_225634

/-- Given an initial amount of 3 gallons of water and using 5/4 gallons,
    the remaining amount is 7/4 gallons. -/
theorem water_remaining (initial : ℚ) (used : ℚ) (remaining : ℚ) : 
  initial = 3 → used = 5/4 → remaining = initial - used → remaining = 7/4 := by
  sorry

end NUMINAMATH_CALUDE_water_remaining_l2256_225634


namespace NUMINAMATH_CALUDE_smallest_with_eight_factors_l2256_225682

def factorCount (n : ℕ) : ℕ := (Nat.divisors n).card

theorem smallest_with_eight_factors :
  ∀ n : ℕ, n > 0 → factorCount n = 8 → n ≥ 24 :=
by sorry

end NUMINAMATH_CALUDE_smallest_with_eight_factors_l2256_225682


namespace NUMINAMATH_CALUDE_stealth_fighter_most_suitable_for_census_l2256_225681

/-- Represents a survey option -/
structure SurveyOption where
  name : String
  population_size : Nat
  feasibility_of_comprehensive_testing : Nat

/-- Defines the criteria for a survey to be suitable for a comprehensive survey (census) -/
def is_suitable_for_census (s : SurveyOption) : Prop :=
  s.population_size ≤ 1000 ∧ 
  s.importance_of_individual ≥ 9 ∧ 
  s.feasibility_of_comprehensive_testing ≥ 9

/-- The four survey options -/
def survey_options : List SurveyOption := [
  { name := "Car crash resistance", population_size := 10000, importance_of_individual := 5, feasibility_of_comprehensive_testing := 2 },
  { name := "Traffic regulation awareness", population_size := 1000000, importance_of_individual := 3, feasibility_of_comprehensive_testing := 1 },
  { name := "Light bulb service life", population_size := 100000, importance_of_individual := 2, feasibility_of_comprehensive_testing := 3 },
  { name := "Stealth fighter components", population_size := 100, importance_of_individual := 10, feasibility_of_comprehensive_testing := 10 }
]

/-- Theorem stating that the stealth fighter components survey is the most suitable for a comprehensive survey -/
theorem stealth_fighter_most_suitable_for_census :
  ∃ (s : SurveyOption), s ∈ survey_options ∧ 
  s.name = "Stealth fighter components" ∧
  is_suitable_for_census s ∧
  ∀ (t : SurveyOption), t ∈ survey_options → t.name ≠ "Stealth fighter components" → ¬(is_suitable_for_census t) :=
sorry

end NUMINAMATH_CALUDE_stealth_fighter_most_suitable_for_census_l2256_225681


namespace NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l2256_225648

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    if a + b + 2c = a² and a + b - 2c = -1, then the largest angle is 120°. -/
theorem largest_angle_in_special_triangle (a b c : ℝ) (h1 : a + b + 2*c = a^2) (h2 : a + b - 2*c = -1) :
  ∃ (A B C : ℝ), 
    0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
    A + B + C = Real.pi ∧    -- Sum of angles in a triangle
    max A (max B C) = 2*Real.pi/3 :=  -- Largest angle is 120°
by sorry

end NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l2256_225648


namespace NUMINAMATH_CALUDE_smallest_inverse_domain_l2256_225691

def g (x : ℝ) : ℝ := (x - 3)^2 - 1

theorem smallest_inverse_domain (d : ℝ) :
  (∀ x y, x ≥ d → y ≥ d → g x = g y → x = y) ↔ d ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_inverse_domain_l2256_225691


namespace NUMINAMATH_CALUDE_lego_storage_time_l2256_225651

/-- The time needed to store all Lego pieces -/
def storage_time (total_pieces : ℕ) (net_increase_per_minute : ℕ) : ℕ :=
  (total_pieces - 1) / net_increase_per_minute + 1

/-- Theorem: It takes 43 minutes to store 45 Lego pieces with a net increase of 1 piece per minute -/
theorem lego_storage_time :
  storage_time 45 1 = 43 := by
  sorry

end NUMINAMATH_CALUDE_lego_storage_time_l2256_225651


namespace NUMINAMATH_CALUDE_nuts_left_over_project_nuts_left_over_l2256_225699

theorem nuts_left_over (bolt_boxes : ℕ) (bolts_per_box : ℕ) (nut_boxes : ℕ) (nuts_per_box : ℕ) 
  (bolts_left : ℕ) (total_used : ℕ) : ℕ :=
  let total_bolts := bolt_boxes * bolts_per_box
  let total_nuts := nut_boxes * nuts_per_box
  let bolts_used := total_bolts - bolts_left
  let nuts_used := total_used - bolts_used
  let nuts_left := total_nuts - nuts_used
  nuts_left

theorem project_nuts_left_over : 
  nuts_left_over 7 11 3 15 3 113 = 6 := by
  sorry

end NUMINAMATH_CALUDE_nuts_left_over_project_nuts_left_over_l2256_225699


namespace NUMINAMATH_CALUDE_special_triangle_area_l2256_225694

/-- A triangle with specific properties -/
structure SpecialTriangle where
  /-- The height of the triangle -/
  height : ℝ
  /-- The smaller part of the base -/
  small_base : ℝ
  /-- The ratio of the divided angle -/
  angle_ratio : ℝ
  /-- The height divides the angle in the given ratio -/
  height_divides_angle : angle_ratio = 2
  /-- The height is 2 cm -/
  height_is_two : height = 2
  /-- The smaller part of the base is 1 cm -/
  small_base_is_one : small_base = 1

/-- The theorem stating the area of the special triangle -/
theorem special_triangle_area (t : SpecialTriangle) : 
  (1 / 2 : ℝ) * t.height * (t.small_base + 5 / 3) = 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_area_l2256_225694


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_on_circle_l2256_225683

theorem sum_of_x_and_y_on_circle (x y : ℝ) 
  (h : 4 * x^2 + 4 * y^2 = 40 * x - 24 * y + 64) : x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_on_circle_l2256_225683


namespace NUMINAMATH_CALUDE_average_daily_attendance_l2256_225628

def monday_attendance : ℕ := 10
def tuesday_attendance : ℕ := 15
def wednesday_to_friday_attendance : ℕ := 10
def total_days : ℕ := 5

def total_attendance : ℕ := 
  monday_attendance + tuesday_attendance + 3 * wednesday_to_friday_attendance

theorem average_daily_attendance : 
  total_attendance / total_days = 11 := by
  sorry

end NUMINAMATH_CALUDE_average_daily_attendance_l2256_225628


namespace NUMINAMATH_CALUDE_balance_four_hearts_l2256_225600

/-- Represents the weight of a symbol in the balance game -/
structure Weight (α : Type) where
  value : ℚ

/-- The balance game with three symbols -/
structure BalanceGame where
  star : Weight ℚ
  heart : Weight ℚ
  circle : Weight ℚ

/-- Defines the balance equations for the game -/
def balance_equations (game : BalanceGame) : Prop :=
  4 * game.star.value + 3 * game.heart.value = 12 * game.circle.value ∧
  2 * game.star.value = game.heart.value + 3 * game.circle.value

/-- The main theorem to prove -/
theorem balance_four_hearts (game : BalanceGame) :
  balance_equations game →
  4 * game.heart.value = 5 * game.circle.value :=
by sorry

end NUMINAMATH_CALUDE_balance_four_hearts_l2256_225600


namespace NUMINAMATH_CALUDE_food_bank_donation_l2256_225625

theorem food_bank_donation (first_week_donation : ℝ) : first_week_donation = 40 :=
  let second_week_donation := 2 * first_week_donation
  let total_donation := first_week_donation + second_week_donation
  let remaining_food := 36
  have h1 : remaining_food = 0.3 * total_donation := by sorry
  have h2 : 36 = 0.3 * (3 * first_week_donation) := by sorry
  have h3 : first_week_donation = 36 / 0.9 := by sorry
  sorry

#check food_bank_donation

end NUMINAMATH_CALUDE_food_bank_donation_l2256_225625


namespace NUMINAMATH_CALUDE_solution_set_min_value_min_value_expression_l2256_225611

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 2| + 2 * |x - 1|

-- Theorem stating the solution set for f(x) ≤ 4
theorem solution_set : {x : ℝ | f x ≤ 4} = {x : ℝ | 0 ≤ x ∧ x ≤ 4/3} := by sorry

-- Theorem stating the minimum value of f(x)
theorem min_value : ∃ (m : ℝ), ∀ (x : ℝ), f x ≥ m ∧ m = 3 := by sorry

-- Theorem for the minimum value of 1/(a-1) + 2/b
theorem min_value_expression :
  ∀ (a b : ℝ), a > 1 → b > 0 → a + 2*b = 3 →
  ∀ (y : ℝ), y = 1/(a-1) + 2/b → y ≥ 9/2 := by sorry

end NUMINAMATH_CALUDE_solution_set_min_value_min_value_expression_l2256_225611


namespace NUMINAMATH_CALUDE_face_value_of_shares_l2256_225690

/-- Proves that the face value of shares is 40, given the dividend rate, return on investment, and purchase price. -/
theorem face_value_of_shares (dividend_rate : ℝ) (roi_rate : ℝ) (purchase_price : ℝ) :
  dividend_rate = 0.125 →
  roi_rate = 0.25 →
  purchase_price = 20 →
  dividend_rate * (purchase_price / roi_rate) = 40 := by
  sorry

end NUMINAMATH_CALUDE_face_value_of_shares_l2256_225690


namespace NUMINAMATH_CALUDE_larger_integer_problem_l2256_225666

theorem larger_integer_problem (x y : ℕ+) 
  (h1 : y - x = 6) 
  (h2 : x * y = 135) : 
  y = 15 := by sorry

end NUMINAMATH_CALUDE_larger_integer_problem_l2256_225666


namespace NUMINAMATH_CALUDE_exist_five_naturals_sum_product_ten_l2256_225659

theorem exist_five_naturals_sum_product_ten : 
  ∃ (a b c d e : ℕ), a + b + c + d + e = 10 ∧ a * b * c * d * e = 10 :=
by sorry

end NUMINAMATH_CALUDE_exist_five_naturals_sum_product_ten_l2256_225659


namespace NUMINAMATH_CALUDE_cos_alpha_plus_seven_pi_twelfths_l2256_225636

theorem cos_alpha_plus_seven_pi_twelfths (α : ℝ) 
  (h : Real.sin (α + π / 12) = 1 / 3) : 
  Real.cos (α + 7 * π / 12) = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_plus_seven_pi_twelfths_l2256_225636


namespace NUMINAMATH_CALUDE_inequality_solution_implies_a_value_l2256_225698

theorem inequality_solution_implies_a_value (a : ℝ) : 
  (∀ x : ℝ, 2 * x - a ≤ -1 ↔ x ≤ 1) → a = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_a_value_l2256_225698


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2256_225619

theorem necessary_but_not_sufficient (a : ℝ) : 
  (∀ x : ℝ, x > 2 → x > 1) ∧ 
  (∃ y : ℝ, y > 1 ∧ y ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2256_225619


namespace NUMINAMATH_CALUDE_exists_polynomial_for_E_l2256_225650

/-- Definition of E(m) as described in the problem -/
def E (m : ℕ) : ℕ :=
  (Finset.univ.filter (fun s : Finset (Fin 6) => s.card = 6)).card

/-- The main theorem to be proved -/
theorem exists_polynomial_for_E :
  ∃ (c₄ c₃ c₂ c₁ c₀ : ℚ),
    ∀ (m : ℕ), m ≥ 6 → m % 2 = 0 →
      E m = c₄ * m^4 + c₃ * m^3 + c₂ * m^2 + c₁ * m + c₀ := by
  sorry

end NUMINAMATH_CALUDE_exists_polynomial_for_E_l2256_225650


namespace NUMINAMATH_CALUDE_max_value_of_expression_l2256_225609

theorem max_value_of_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let A := (a^4 + b^4 + c^4) / ((a + b + c)^4 - 80*(a*b*c)^(4/3))
  A ≤ 3 ∧ ∃ (x : ℝ), x > 0 ∧ 
    let A' := (x^4 + x^4 + x^4) / ((x + x + x)^4 - 80*(x*x*x)^(4/3))
    A' = 3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l2256_225609


namespace NUMINAMATH_CALUDE_max_value_theorem_l2256_225647

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

noncomputable def g (x : ℝ) : ℝ := -(Real.log x) / x

theorem max_value_theorem (x₁ x₂ t : ℝ) (h1 : f x₁ = t) (h2 : g x₂ = t) (h3 : t > 0) :
  (∀ y₁ y₂ s : ℝ, f y₁ = s → g y₂ = s → s > 0 → y₁ / (y₂ * Real.exp s) ≤ 1 / Real.exp 1) ∧
  (∃ z₁ z₂ r : ℝ, f z₁ = r ∧ g z₂ = r ∧ r > 0 ∧ z₁ / (z₂ * Real.exp r) = 1 / Real.exp 1) :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2256_225647


namespace NUMINAMATH_CALUDE_product_divisible_by_sum_implies_inequality_l2256_225652

theorem product_divisible_by_sum_implies_inequality (m n : ℕ) 
  (h : (m + n) ∣ (m * n)) : m + n ≤ n^2 := by
  sorry

end NUMINAMATH_CALUDE_product_divisible_by_sum_implies_inequality_l2256_225652


namespace NUMINAMATH_CALUDE_sequence_equality_l2256_225643

def sequence_property (a : ℕ → ℝ) : Prop :=
  (∀ n ∈ Finset.range 98, a n - 2022 * a (n + 1) + 2021 * a (n + 2) ≥ 0) ∧
  (a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0) ∧
  (a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0)

theorem sequence_equality (a : ℕ → ℝ) (h : sequence_property a) (h10 : a 10 = 10) : a 22 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sequence_equality_l2256_225643


namespace NUMINAMATH_CALUDE_range_of_a_for_two_negative_roots_l2256_225665

-- Define the quadratic equation
def quadratic (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + |a|

-- Define the condition for two negative roots
def has_two_negative_roots (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧ 
  quadratic a x₁ = 0 ∧ quadratic a x₂ = 0

-- State the theorem
theorem range_of_a_for_two_negative_roots :
  ∃ l u : ℝ, ∀ a : ℝ, has_two_negative_roots a ↔ l < a ∧ a < u :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_two_negative_roots_l2256_225665


namespace NUMINAMATH_CALUDE_sum_reciprocals_lower_bound_l2256_225622

theorem sum_reciprocals_lower_bound 
  (a b c d m : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (hm : m > 0)
  (eq1 : 1/a = (a + b + c + d + m)/a)
  (eq2 : 1/b = (a + b + c + d + m)/b)
  (eq3 : 1/c = (a + b + c + d + m)/c)
  (eq4 : 1/d = (a + b + c + d + m)/d)
  (eq5 : 1/m = (a + b + c + d + m)/m) :
  1/a + 1/b + 1/c + 1/d + 1/m ≥ 25 := by
sorry

end NUMINAMATH_CALUDE_sum_reciprocals_lower_bound_l2256_225622


namespace NUMINAMATH_CALUDE_quadratic_polynomial_value_l2256_225675

/-- A quadratic polynomial -/
def QuadraticPolynomial (a b c : ℚ) : ℚ → ℚ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_polynomial_value (a b c : ℚ) :
  let p : ℚ → ℚ := QuadraticPolynomial a b c
  (∀ x : ℚ, (x - 1) * (x + 1) * (x - 8) ∣ p x ^ 3 - x) →
  p 13 = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_value_l2256_225675


namespace NUMINAMATH_CALUDE_infinite_solutions_l2256_225630

/-- The equation (x-1)^2 + (x+1)^2 = y^2 + 1 -/
def is_solution (x y : ℕ) : Prop :=
  (x - 1)^2 + (x + 1)^2 = y^2 + 1

/-- The transformation function -/
def transform (x y : ℕ) : ℕ × ℕ :=
  (3*x + 2*y, 4*x + 3*y)

theorem infinite_solutions :
  (is_solution 0 1) ∧
  (is_solution 2 3) ∧
  (∀ x y : ℕ, is_solution x y → is_solution (transform x y).1 (transform x y).2) →
  ∃ f : ℕ → ℕ × ℕ, ∀ n : ℕ, is_solution (f n).1 (f n).2 ∧ f n ≠ f (n+1) :=
by sorry

end NUMINAMATH_CALUDE_infinite_solutions_l2256_225630


namespace NUMINAMATH_CALUDE_trapezoidal_dam_pressure_l2256_225613

/-- 
Represents a vertical trapezoidal dam with water pressure.
-/
structure TrapezoidalDam where
  ρ : ℝ  -- density of water
  g : ℝ  -- acceleration due to gravity
  h : ℝ  -- height of the dam
  a : ℝ  -- top width of the dam
  b : ℝ  -- bottom width of the dam
  h_pos : h > 0
  a_pos : a > 0
  b_pos : b > 0
  a_ge_b : a ≥ b

/-- 
The total water pressure on a vertical trapezoidal dam is ρg(h^2(2a + b))/6.
-/
theorem trapezoidal_dam_pressure (dam : TrapezoidalDam) :
  ∃ P : ℝ, P = dam.ρ * dam.g * (dam.h^2 * (2 * dam.a + dam.b)) / 6 :=
by
  sorry

end NUMINAMATH_CALUDE_trapezoidal_dam_pressure_l2256_225613


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcd_condition_l2256_225685

theorem greatest_integer_with_gcd_condition :
  ∃ n : ℕ, n < 200 ∧ n.gcd 30 = 10 ∧ ∀ m : ℕ, m < 200 → m.gcd 30 = 10 → m ≤ n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcd_condition_l2256_225685


namespace NUMINAMATH_CALUDE_handshakes_eight_people_l2256_225673

/-- The number of handshakes in a group where each person shakes hands with every other person exactly once. -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a group of 8 people, where each person shakes hands with every other person exactly once, the total number of handshakes is 28. -/
theorem handshakes_eight_people : handshakes 8 = 28 := by
  sorry

end NUMINAMATH_CALUDE_handshakes_eight_people_l2256_225673


namespace NUMINAMATH_CALUDE_line_passes_through_second_and_fourth_quadrants_l2256_225624

/-- A line with equation y = -2x + b (where b is a constant) always passes through the second and fourth quadrants. -/
theorem line_passes_through_second_and_fourth_quadrants (b : ℝ) :
  ∃ (x₁ x₂ y₁ y₂ : ℝ), 
    (x₁ < 0 ∧ y₁ > 0 ∧ y₁ = -2*x₁ + b) ∧ 
    (x₂ > 0 ∧ y₂ < 0 ∧ y₂ = -2*x₂ + b) :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_second_and_fourth_quadrants_l2256_225624


namespace NUMINAMATH_CALUDE_a_50_equals_6_5_l2256_225608

-- Define the sequence a_n
def a : ℕ → ℚ
| n => sorry

-- Theorem statement
theorem a_50_equals_6_5 : a 50 = 6/5 := by sorry

end NUMINAMATH_CALUDE_a_50_equals_6_5_l2256_225608


namespace NUMINAMATH_CALUDE_total_hamburgers_bought_l2256_225657

/-- Proves that the total number of hamburgers bought is 50 given the specified conditions. -/
theorem total_hamburgers_bought (total_spent : ℚ) (single_cost : ℚ) (double_cost : ℚ) (double_count : ℕ) : ℕ :=
  if total_spent = 70.5 ∧ single_cost = 1 ∧ double_cost = 1.5 ∧ double_count = 41 then
    50
  else
    0

#check total_hamburgers_bought

end NUMINAMATH_CALUDE_total_hamburgers_bought_l2256_225657


namespace NUMINAMATH_CALUDE_hyperbola_center_l2256_225602

theorem hyperbola_center (x y : ℝ) :
  9 * x^2 - 54 * x - 16 * y^2 + 128 * y - 71 = 0 →
  ∃ a b : ℝ, a = 3 ∧ b = 4 ∧
  ∀ x' y' : ℝ, 9 * (x' - a)^2 - 16 * (y' - b)^2 = 9 * x'^2 - 54 * x' - 16 * y'^2 + 128 * y' - 71 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_center_l2256_225602


namespace NUMINAMATH_CALUDE_system_solution_l2256_225678

theorem system_solution :
  ∃ (x y z : ℝ), 
    (x + 2*y = 4) ∧ 
    (2*x + 5*y - 2*z = 11) ∧ 
    (3*x - 5*y + 2*z = -1) ∧
    (x = 2) ∧ (y = 1) ∧ (z = -1) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2256_225678


namespace NUMINAMATH_CALUDE_subtraction_multiplication_problem_l2256_225672

theorem subtraction_multiplication_problem : 
  let initial_value : ℚ := 555.55
  let subtracted_value : ℚ := 111.11
  let multiplier : ℚ := 2
  let result : ℚ := (initial_value - subtracted_value) * multiplier
  result = 888.88 := by sorry

end NUMINAMATH_CALUDE_subtraction_multiplication_problem_l2256_225672


namespace NUMINAMATH_CALUDE_martha_cookies_theorem_l2256_225618

/-- Given that Martha can make 24 cookies with 3 cups of flour, this function
    calculates how many cookies she can make with a given number of cups. -/
def cookies_from_flour (cups : ℚ) : ℚ :=
  (24 / 3) * cups

/-- Given that Martha can make 24 cookies with 3 cups of flour, this function
    calculates how many cups of flour are needed to make a given number of cookies. -/
def flour_for_cookies (cookies : ℚ) : ℚ :=
  (3 / 24) * cookies

/-- Theorem stating that with 5 cups of flour, Martha can make 40 cookies,
    and 60 cookies require 7.5 cups of flour. -/
theorem martha_cookies_theorem :
  cookies_from_flour 5 = 40 ∧ flour_for_cookies 60 = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_martha_cookies_theorem_l2256_225618


namespace NUMINAMATH_CALUDE_camping_trip_percentage_l2256_225693

theorem camping_trip_percentage (total_students : ℕ) 
  (h1 : total_students > 0)
  (students_more_than_100 : ℕ) 
  (h2 : students_more_than_100 = (18 * total_students) / 100)
  (h3 : (75 * (students_more_than_100 * 100 / 18)) / 100 + students_more_than_100 = 
        (72 * total_students) / 100) :
  (72 * total_students) / 100 = total_students - 
    ((75 * (students_more_than_100 * 100 / 18)) / 100 + students_more_than_100) :=
by sorry

end NUMINAMATH_CALUDE_camping_trip_percentage_l2256_225693


namespace NUMINAMATH_CALUDE_marble_fraction_after_doubling_red_l2256_225661

theorem marble_fraction_after_doubling_red (total : ℚ) (h : total > 0) :
  let initial_blue := (3 / 5) * total
  let initial_red := total - initial_blue
  let new_red := 2 * initial_red
  let new_total := initial_blue + new_red
  new_red / new_total = 4 / 7 := by sorry

end NUMINAMATH_CALUDE_marble_fraction_after_doubling_red_l2256_225661


namespace NUMINAMATH_CALUDE_rosa_flower_count_l2256_225676

/-- The number of flowers Rosa has after receiving flowers from Andre -/
def total_flowers (initial : ℕ) (received : ℕ) : ℕ :=
  initial + received

/-- Theorem stating that Rosa's total flowers is the sum of her initial flowers and received flowers -/
theorem rosa_flower_count (initial : ℕ) (received : ℕ) :
  total_flowers initial received = initial + received :=
by sorry

end NUMINAMATH_CALUDE_rosa_flower_count_l2256_225676


namespace NUMINAMATH_CALUDE_exam_score_calculation_l2256_225670

theorem exam_score_calculation (total_questions : ℕ) (total_marks : ℤ) (correct_answers : ℕ) 
  (h1 : total_questions = 60)
  (h2 : total_marks = 130)
  (h3 : correct_answers = 38)
  (h4 : total_questions = correct_answers + (total_questions - correct_answers)) :
  ∃ (marks_per_correct : ℕ), 
    marks_per_correct * correct_answers - (total_questions - correct_answers) = total_marks ∧ 
    marks_per_correct = 4 := by
  sorry

end NUMINAMATH_CALUDE_exam_score_calculation_l2256_225670


namespace NUMINAMATH_CALUDE_triangle_theorem_l2256_225664

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.a > 0) (h2 : t.b > 0) (h3 : t.c > 0)
  (h4 : t.A > 0) (h5 : t.B > 0) (h6 : t.C > 0)
  (h7 : t.A + t.B + t.C = Real.pi)
  (h8 : Real.sqrt 3 * t.a * Real.sin t.C + t.a * Real.cos t.C = t.c + t.b) :
  t.A = Real.pi / 3 ∧ 
  (t.a = Real.sqrt 3 → Real.sqrt 3 < t.b + t.c ∧ t.b + t.c ≤ 2 * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_triangle_theorem_l2256_225664


namespace NUMINAMATH_CALUDE_trajectory_and_intersection_l2256_225684

/-- The trajectory C of point P in the Cartesian coordinate system xOy,
    where the sum of distances from P to (0, -√3) and (0, √3) equals 4 -/
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 / 4 + p.1^2 = 1}

/-- The line that intersects C -/
def line (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k * p.1 + 1}

/-- Theorem stating the properties of the trajectory C and its intersection with the line -/
theorem trajectory_and_intersection :
  ∀ k : ℝ,
  (∀ p : ℝ × ℝ, p ∈ C → (Real.sqrt ((p.1)^2 + (p.2 + Real.sqrt 3)^2) +
                         Real.sqrt ((p.1)^2 + (p.2 - Real.sqrt 3)^2) = 4)) ∧
  (∃ A B : ℝ × ℝ, A ∈ C ∧ B ∈ C ∧ A ∈ line k ∧ B ∈ line k ∧
    (k = 1/2 ∨ k = -1/2) ↔ (A.1 * B.1 + A.2 * B.2 = 0)) := by
  sorry

end NUMINAMATH_CALUDE_trajectory_and_intersection_l2256_225684


namespace NUMINAMATH_CALUDE_square_center_sum_l2256_225686

-- Define the square ABCD
structure Square where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

-- Define the conditions
def square_conditions (s : Square) : Prop :=
  -- Square is in the first quadrant
  s.A.1 ≥ 0 ∧ s.A.2 ≥ 0 ∧
  s.B.1 ≥ 0 ∧ s.B.2 ≥ 0 ∧
  s.C.1 ≥ 0 ∧ s.C.2 ≥ 0 ∧
  s.D.1 ≥ 0 ∧ s.D.2 ≥ 0 ∧
  -- Points on the lines
  (∃ t : ℝ, t ∈ Set.Icc 0 1 ∧ (4, 0) = s.A + t • (s.B - s.A)) ∧
  (∃ t : ℝ, t ∈ Set.Icc 0 1 ∧ (7, 0) = s.C + t • (s.D - s.C)) ∧
  (∃ t : ℝ, t ∈ Set.Icc 0 1 ∧ (9, 0) = s.B + t • (s.C - s.B)) ∧
  (∃ t : ℝ, t ∈ Set.Icc 0 1 ∧ (15, 0) = s.D + t • (s.A - s.D))

-- Theorem statement
theorem square_center_sum (s : Square) (h : square_conditions s) :
  (s.A.1 + s.B.1 + s.C.1 + s.D.1) / 4 + (s.A.2 + s.B.2 + s.C.2 + s.D.2) / 4 = 27 / 4 :=
by sorry

end NUMINAMATH_CALUDE_square_center_sum_l2256_225686


namespace NUMINAMATH_CALUDE_ellipse_k_range_l2256_225660

-- Define the equation
def ellipse_equation (x y k : ℝ) : Prop := x^2 + k*y^2 = 2

-- Define what it means for the equation to represent an ellipse with foci on the x-axis
def is_ellipse_with_foci_on_x_axis (k : ℝ) : Prop :=
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ 
  ∀ (x y : ℝ), ellipse_equation x y k ↔ (x^2 / (a^2) + y^2 / (b^2) = 1)

-- Theorem statement
theorem ellipse_k_range :
  ∀ k : ℝ, is_ellipse_with_foci_on_x_axis k ↔ k > 1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l2256_225660


namespace NUMINAMATH_CALUDE_vakha_always_wins_l2256_225623

/-- Represents a point on the circle -/
structure Point where
  index : Fin 99

/-- Represents a color (Red or Blue) -/
inductive Color
  | Red
  | Blue

/-- Represents the game state -/
structure GameState where
  coloredPoints : Fin 99 → Option Color

/-- Represents a player (Bjorn or Vakha) -/
inductive Player
  | Bjorn
  | Vakha

/-- Checks if three points form an equilateral triangle -/
def isEquilateralTriangle (p1 p2 p3 : Point) : Prop :=
  (p2.index - p1.index) % 33 = 0 ∧
  (p3.index - p2.index) % 33 = 0 ∧
  (p1.index - p3.index) % 33 = 0

/-- Checks if a monochromatic equilateral triangle exists in the game state -/
def existsMonochromaticTriangle (state : GameState) : Prop :=
  ∃ (p1 p2 p3 : Point) (c : Color),
    isEquilateralTriangle p1 p2 p3 ∧
    state.coloredPoints p1.index = some c ∧
    state.coloredPoints p2.index = some c ∧
    state.coloredPoints p3.index = some c

/-- Represents a valid move in the game -/
def validMove (state : GameState) (p : Point) (c : Color) : Prop :=
  state.coloredPoints p.index = none ∧
  (∃ (q : Point), state.coloredPoints q.index ≠ none ∧ (q.index + 1 = p.index ∨ q.index = p.index + 1))

/-- Represents a winning strategy for Vakha -/
def hasWinningStrategy (player : Player) : Prop :=
  ∀ (initialState : GameState),
    ∃ (finalState : GameState),
      (∀ (p : Point) (c : Color), validMove initialState p c → 
        ∃ (nextState : GameState), validMove nextState p c) ∧
      existsMonochromaticTriangle finalState

/-- The main theorem: Vakha always has a winning strategy -/
theorem vakha_always_wins : hasWinningStrategy Player.Vakha := by
  sorry

end NUMINAMATH_CALUDE_vakha_always_wins_l2256_225623


namespace NUMINAMATH_CALUDE_salon_buys_33_cans_l2256_225607

/-- Represents the number of cans of hairspray a salon buys daily. -/
def salon_hairspray_cans (customers : ℕ) (cans_per_customer : ℕ) (extra_cans : ℕ) : ℕ :=
  customers * cans_per_customer + extra_cans

/-- Theorem stating that the salon buys 33 cans of hairspray daily. -/
theorem salon_buys_33_cans :
  salon_hairspray_cans 14 2 5 = 33 := by
  sorry

#eval salon_hairspray_cans 14 2 5

end NUMINAMATH_CALUDE_salon_buys_33_cans_l2256_225607


namespace NUMINAMATH_CALUDE_f_decreasing_implies_a_range_l2256_225633

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3*a - 1)*x + 4*a else Real.log x / Real.log a

-- State the theorem
theorem f_decreasing_implies_a_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x > f a y) →
  a ∈ Set.Icc (1/7 : ℝ) (1/3 : ℝ) ∧ a ≠ 1/3 :=
sorry

end NUMINAMATH_CALUDE_f_decreasing_implies_a_range_l2256_225633


namespace NUMINAMATH_CALUDE_range_of_f_l2256_225601

-- Define the function
def f (x : ℝ) : ℝ := |x + 5| - |x - 3|

-- State the theorem
theorem range_of_f :
  Set.range f = Set.Icc (-8 : ℝ) 8 :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l2256_225601


namespace NUMINAMATH_CALUDE_headphone_price_reduction_l2256_225667

theorem headphone_price_reduction (original_price : ℝ) (first_discount_rate : ℝ) (second_discount_rate : ℝ) :
  original_price = 120 →
  first_discount_rate = 0.25 →
  second_discount_rate = 0.1 →
  let price_after_first_discount := original_price * (1 - first_discount_rate)
  let final_price := price_after_first_discount * (1 - second_discount_rate)
  final_price = 81 := by
sorry

end NUMINAMATH_CALUDE_headphone_price_reduction_l2256_225667


namespace NUMINAMATH_CALUDE_non_mundane_primes_characterization_l2256_225642

/-- A prime number is mundane if there exist positive integers a and b less than p/2 
    such that (ab - 1)/p is a positive integer. -/
def IsMundane (p : ℕ) : Prop :=
  ∃ a b : ℕ, 0 < a ∧ 0 < b ∧ a < p / 2 ∧ b < p / 2 ∧ 
  ∃ k : ℕ, 0 < k ∧ k * p = a * b - 1

/-- The set of non-mundane primes -/
def NonMundanePrimes : Set ℕ := {2, 3, 5, 7, 13}

/-- Theorem: A prime number is not mundane if and only if it belongs to the set {2, 3, 5, 7, 13} -/
theorem non_mundane_primes_characterization (p : ℕ) (hp : Nat.Prime p) : 
  ¬ IsMundane p ↔ p ∈ NonMundanePrimes := by
  sorry

end NUMINAMATH_CALUDE_non_mundane_primes_characterization_l2256_225642


namespace NUMINAMATH_CALUDE_absolute_value_difference_l2256_225658

theorem absolute_value_difference : |(8-(3^2))| - |((4^2) - (6*3))| = -1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_difference_l2256_225658


namespace NUMINAMATH_CALUDE_no_real_solution_exists_sum_equals_square_plus_twenty_l2256_225655

/-- Given three numbers with a difference of 4 between each,
    where their sum is 20 more than the square of the first number,
    prove that no real solution exists for the middle number. -/
theorem no_real_solution_exists (x : ℝ) : ¬ ∃ x : ℝ, x^2 - 3*x + 8 = 0 := by
  sorry

/-- Define the relationship between the three numbers -/
def second_number (x : ℝ) : ℝ := x + 4

/-- Define the relationship between the three numbers -/
def third_number (x : ℝ) : ℝ := x + 8

/-- Define the sum of the three numbers -/
def sum_of_numbers (x : ℝ) : ℝ := x + second_number x + third_number x

/-- Define the relationship between the sum and the square of the first number -/
theorem sum_equals_square_plus_twenty (x : ℝ) : sum_of_numbers x = x^2 + 20 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solution_exists_sum_equals_square_plus_twenty_l2256_225655


namespace NUMINAMATH_CALUDE_jelly_bean_division_l2256_225680

theorem jelly_bean_division (initial_amount : ℕ) (eaten_amount : ℕ) (num_piles : ℕ) :
  initial_amount = 72 →
  eaten_amount = 12 →
  num_piles = 5 →
  (initial_amount - eaten_amount) / num_piles = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_jelly_bean_division_l2256_225680


namespace NUMINAMATH_CALUDE_river_length_problem_l2256_225616

theorem river_length_problem (straight_length crooked_length total_length : ℝ) :
  straight_length * 3 = crooked_length →
  straight_length + crooked_length = total_length →
  total_length = 80 →
  straight_length = 20 := by
  sorry

end NUMINAMATH_CALUDE_river_length_problem_l2256_225616


namespace NUMINAMATH_CALUDE_bug_triangle_probability_l2256_225663

/-- Probability of the bug being at the starting vertex after n moves -/
def P : ℕ → ℚ
  | 0 => 1
  | n + 1 => 1/2 * (1 - P n)

/-- The problem statement -/
theorem bug_triangle_probability : P 10 = 171/512 := by
  sorry

end NUMINAMATH_CALUDE_bug_triangle_probability_l2256_225663


namespace NUMINAMATH_CALUDE_range_of_a_l2256_225620

-- Define p and q as predicates on real numbers
def p (a : ℝ) (x : ℝ) : Prop := x ≥ a
def q (x : ℝ) : Prop := x^2 - 2*x - 3 ≥ 0

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (∀ x, p a x → q x) ∧ (∃ x, q x ∧ ¬p a x) → a ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2256_225620


namespace NUMINAMATH_CALUDE_average_notebooks_sold_l2256_225635

/-- The average number of notebooks sold per day, given the total number of bundles,
    notebooks per bundle, and the number of days. -/
def average_notebooks_per_day (bundles : ℕ) (notebooks_per_bundle : ℕ) (days : ℕ) : ℚ :=
  (bundles * notebooks_per_bundle : ℚ) / days

/-- Theorem stating that the average number of notebooks sold per day is 120,
    given the conditions from the problem. -/
theorem average_notebooks_sold (bundles : ℕ) (notebooks_per_bundle : ℕ) (days : ℕ)
  (h1 : bundles = 15)
  (h2 : notebooks_per_bundle = 40)
  (h3 : days = 5) :
  average_notebooks_per_day bundles notebooks_per_bundle days = 120 := by
  sorry

#eval average_notebooks_per_day 15 40 5

end NUMINAMATH_CALUDE_average_notebooks_sold_l2256_225635


namespace NUMINAMATH_CALUDE_irrational_sqrt_7_rational_others_l2256_225668

theorem irrational_sqrt_7_rational_others : 
  (Irrational (Real.sqrt 7)) ∧ 
  (¬ Irrational 3.1415) ∧ 
  (¬ Irrational 3) ∧ 
  (¬ Irrational (1/3 : ℚ)) := by sorry

end NUMINAMATH_CALUDE_irrational_sqrt_7_rational_others_l2256_225668


namespace NUMINAMATH_CALUDE_cube_root_of_quarter_l2256_225679

theorem cube_root_of_quarter (t s : ℝ) : t = 15 * s^3 ∧ t = 3.75 → s = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_quarter_l2256_225679


namespace NUMINAMATH_CALUDE_circle_center_sum_l2256_225637

theorem circle_center_sum (h k : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = 6*x + 8*y + 9 ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 - 9)) →
  h + k = 7 := by
sorry

end NUMINAMATH_CALUDE_circle_center_sum_l2256_225637


namespace NUMINAMATH_CALUDE_circle_equation_with_radius_3_l2256_225654

theorem circle_equation_with_radius_3 (c : ℝ) : 
  (∀ x y : ℝ, x^2 - 8*x + y^2 + 10*y + c = 0 ↔ (x - 4)^2 + (y + 5)^2 = 3^2) → 
  c = 32 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_with_radius_3_l2256_225654


namespace NUMINAMATH_CALUDE_jakes_total_earnings_l2256_225617

/-- Calculates Jake's total earnings from selling baby snakes --/
def jakes_earnings (viper_count cobra_count python_count anaconda_count : ℕ)
  (viper_eggs cobra_eggs python_eggs anaconda_eggs : ℕ)
  (viper_price cobra_price python_price anaconda_price : ℚ)
  (viper_discount cobra_discount python_discount anaconda_discount : ℚ) : ℚ :=
  let viper_total := viper_count * viper_eggs * (viper_price * (1 - viper_discount))
  let cobra_total := cobra_count * cobra_eggs * (cobra_price * (1 - cobra_discount))
  let python_total := python_count * python_eggs * (python_price * (1 - python_discount))
  let anaconda_total := anaconda_count * anaconda_eggs * (anaconda_price * (1 - anaconda_discount))
  viper_total + cobra_total + python_total + anaconda_total

/-- Theorem stating Jake's total earnings --/
theorem jakes_total_earnings :
  jakes_earnings 3 2 1 1 3 2 4 5 300 250 450 500 (10/100) (5/100) (75/1000) (12/100) = 7245 := by
  sorry

end NUMINAMATH_CALUDE_jakes_total_earnings_l2256_225617


namespace NUMINAMATH_CALUDE_reseating_women_problem_l2256_225621

/-- Represents the number of ways n women can be reseated under the given conditions --/
def T : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | 2 => 2
  | n + 3 => T (n + 2) + T (n + 1) + if n = 0 then 1 else T n

/-- The problem statement --/
theorem reseating_women_problem :
  T 15 = 987 := by
  sorry

end NUMINAMATH_CALUDE_reseating_women_problem_l2256_225621


namespace NUMINAMATH_CALUDE_average_difference_l2256_225638

/-- Given that the average of a and b is 50, and the average of b and c is 70, prove that c - a = 40 -/
theorem average_difference (a b c : ℝ) 
  (h1 : (a + b) / 2 = 50) 
  (h2 : (b + c) / 2 = 70) : 
  c - a = 40 := by
  sorry

end NUMINAMATH_CALUDE_average_difference_l2256_225638


namespace NUMINAMATH_CALUDE_hyperbola_iff_product_negative_l2256_225646

/-- Definition of a hyperbola equation -/
def is_hyperbola_equation (m n : ℝ) : Prop :=
  ∃ (x y : ℝ → ℝ), ∀ t, (x t)^2 / m + (y t)^2 / n = 1 ∧
  (∃ t₁ t₂, (x t₁, y t₁) ≠ (x t₂, y t₂))

/-- The main theorem stating the condition for a hyperbola -/
theorem hyperbola_iff_product_negative (m n : ℝ) :
  is_hyperbola_equation m n ↔ m * n < 0 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_iff_product_negative_l2256_225646


namespace NUMINAMATH_CALUDE_interest_calculation_l2256_225604

/-- Calculates the simple interest and final amount given initial principal, annual rate, and time in years -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ × ℝ :=
  let interest := principal * rate * time
  let final_amount := principal + interest
  (interest, final_amount)

theorem interest_calculation (P : ℝ) :
  let (interest, final_amount) := simple_interest P 0.06 0.25
  final_amount = 510.60 → interest = 7.54 := by
sorry

end NUMINAMATH_CALUDE_interest_calculation_l2256_225604


namespace NUMINAMATH_CALUDE_consecutive_products_divisibility_l2256_225640

theorem consecutive_products_divisibility (a : ℤ) :
  ∃ k : ℤ, a * (a + 1) + (a + 1) * (a + 2) + (a + 2) * (a + 3) + a * (a + 3) + 1 = 12 * k :=
by sorry

end NUMINAMATH_CALUDE_consecutive_products_divisibility_l2256_225640


namespace NUMINAMATH_CALUDE_quadratic_intersects_x_axis_l2256_225627

/-- The quadratic function y = (k-1)x^2 + 2x - 1 intersects the x-axis if and only if k ≥ 0 and k ≠ 1 -/
theorem quadratic_intersects_x_axis (k : ℝ) : 
  (∃ x : ℝ, (k - 1) * x^2 + 2 * x - 1 = 0) ↔ (k ≥ 0 ∧ k ≠ 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_intersects_x_axis_l2256_225627


namespace NUMINAMATH_CALUDE_barbara_initial_candies_l2256_225674

/-- The number of candies Barbara bought -/
def candies_bought : ℕ := 18

/-- The total number of candies Barbara has after buying more -/
def total_candies : ℕ := 27

/-- The initial number of candies Barbara had -/
def initial_candies : ℕ := total_candies - candies_bought

theorem barbara_initial_candies : initial_candies = 9 := by
  sorry

end NUMINAMATH_CALUDE_barbara_initial_candies_l2256_225674
