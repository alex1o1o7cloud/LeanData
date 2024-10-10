import Mathlib

namespace pi_comparison_l2862_286220

theorem pi_comparison : -Real.pi < -3.14 := by sorry

end pi_comparison_l2862_286220


namespace paul_work_time_l2862_286218

-- Define the work rates and time
def george_work_rate : ℚ := 3 / (5 * 9)
def total_work : ℚ := 1
def george_paul_time : ℚ := 4
def george_initial_work : ℚ := 3 / 5

-- Theorem statement
theorem paul_work_time (paul_work_rate : ℚ) : 
  george_work_rate + paul_work_rate = (total_work - george_initial_work) / george_paul_time →
  total_work / paul_work_rate = 90 / 13 := by
  sorry

end paul_work_time_l2862_286218


namespace complex_sum_magnitude_l2862_286250

theorem complex_sum_magnitude (z₁ z₂ : ℂ) 
  (h₁ : Complex.abs z₁ = 1)
  (h₂ : Complex.abs z₂ = 1)
  (h₃ : Complex.abs (z₁ - z₂) = 1) :
  Complex.abs (z₁ + z₂) = Real.sqrt 3 := by
  sorry

end complex_sum_magnitude_l2862_286250


namespace min_distance_to_line_l2862_286285

/-- Given a point P(a,b) on the line y = √3x - √3, 
    the minimum value of (a+1)^2 + b^2 is 3 -/
theorem min_distance_to_line (a b : ℝ) : 
  b = Real.sqrt 3 * a - Real.sqrt 3 → 
  (∀ x y : ℝ, y = Real.sqrt 3 * x - Real.sqrt 3 → (a + 1)^2 + b^2 ≤ (x + 1)^2 + y^2) → 
  (a + 1)^2 + b^2 = 3 :=
by sorry

end min_distance_to_line_l2862_286285


namespace garden_ratio_l2862_286299

/-- Represents a rectangular garden with given perimeter and length -/
structure RectangularGarden where
  perimeter : ℝ
  length : ℝ
  width : ℝ
  perimeter_eq : perimeter = 2 * (length + width)
  length_positive : length > 0
  width_positive : width > 0

/-- The ratio of length to width for a rectangular garden with perimeter 150 and length 50 is 2:1 -/
theorem garden_ratio (garden : RectangularGarden) 
  (h_perimeter : garden.perimeter = 150) 
  (h_length : garden.length = 50) : 
  garden.length / garden.width = 2 := by
  sorry

end garden_ratio_l2862_286299


namespace find_m_value_l2862_286294

/-- Given functions f and g, prove that m = 4 when 3f(4) = g(4) -/
theorem find_m_value (m : ℝ) : 
  let f : ℝ → ℝ := λ x => x^2 - 3*x + m
  let g : ℝ → ℝ := λ x => x^2 - 3*x + 5*m
  3 * f 4 = g 4 → m = 4 := by
sorry

end find_m_value_l2862_286294


namespace comparison_of_scientific_notation_l2862_286241

theorem comparison_of_scientific_notation :
  (1.9 : ℝ) * (10 : ℝ) ^ 5 > (9.1 : ℝ) * (10 : ℝ) ^ 4 := by
  sorry

end comparison_of_scientific_notation_l2862_286241


namespace not_all_n_squared_plus_n_plus_41_prime_l2862_286274

theorem not_all_n_squared_plus_n_plus_41_prime :
  ∃ n : ℕ, ¬(Nat.Prime (n^2 + n + 41)) := by
  sorry

end not_all_n_squared_plus_n_plus_41_prime_l2862_286274


namespace flower_count_l2862_286247

theorem flower_count (yoojung_flowers namjoon_flowers : ℕ) : 
  yoojung_flowers = 32 → 
  yoojung_flowers = 4 * namjoon_flowers → 
  yoojung_flowers + namjoon_flowers = 40 := by
sorry

end flower_count_l2862_286247


namespace rectangle_width_l2862_286270

theorem rectangle_width (length width : ℝ) : 
  length / width = 6 / 5 → length = 24 → width = 20 := by
  sorry

end rectangle_width_l2862_286270


namespace system_solution_unique_l2862_286246

theorem system_solution_unique (x y : ℝ) : 
  (2 * x + 3 * y = -11) ∧ (6 * x - 5 * y = 9) ↔ (x = -1 ∧ y = -3) :=
by sorry

end system_solution_unique_l2862_286246


namespace inequality_solution_set_l2862_286266

theorem inequality_solution_set (a : ℝ) :
  let S := {x : ℝ | 12 * x^2 - a * x > a^2}
  (a > 0 → S = {x | x < -a/4} ∪ {x | x > a/3}) ∧
  (a = 0 → S = {x | x ≠ 0}) ∧
  (a < 0 → S = {x | x < a/3} ∪ {x | x > -a/4}) :=
by sorry

end inequality_solution_set_l2862_286266


namespace inequality_and_equality_condition_l2862_286217

theorem inequality_and_equality_condition (n : ℕ+) :
  (1/3 : ℝ) * n.val^2 + (1/2 : ℝ) * n.val + (1/6 : ℝ) ≥ (n.val.factorial : ℝ)^((2 : ℝ) / n.val) ∧
  ((1/3 : ℝ) * n.val^2 + (1/2 : ℝ) * n.val + (1/6 : ℝ) = (n.val.factorial : ℝ)^((2 : ℝ) / n.val) ↔ n = 1) :=
by sorry

end inequality_and_equality_condition_l2862_286217


namespace necessary_condition_inequality_l2862_286208

theorem necessary_condition_inequality (a b c : ℝ) (hc : c ≠ 0) :
  (∀ c, c ≠ 0 → a * c^2 > b * c^2) → a > b :=
sorry

end necessary_condition_inequality_l2862_286208


namespace expression_evaluation_l2862_286257

theorem expression_evaluation : 7500 + (1250 / 50) = 7525 := by
  sorry

end expression_evaluation_l2862_286257


namespace inequalities_for_negative_a_l2862_286225

theorem inequalities_for_negative_a (a b : ℝ) (ha : a < 0) :
  (a < b) ∧ (a^2 + b^2 > 2) ∧ 
  (∃ b, ¬(a + b < a*b)) ∧ (∃ b, ¬(|a| > |b|)) :=
sorry

end inequalities_for_negative_a_l2862_286225


namespace sin_50_plus_sqrt3_tan_10_equals_1_l2862_286239

theorem sin_50_plus_sqrt3_tan_10_equals_1 : 
  Real.sin (50 * π / 180) * (1 + Real.sqrt 3 * Real.tan (10 * π / 180)) = 1 := by
  sorry

end sin_50_plus_sqrt3_tan_10_equals_1_l2862_286239


namespace min_value_expression_min_value_attained_l2862_286216

theorem min_value_expression (x : ℝ) :
  (15 - x) * (8 - x) * (15 + x) * (8 + x) ≥ -6480.25 :=
by sorry

theorem min_value_attained (ε : ℝ) (hε : ε > 0) :
  ∃ x : ℝ, (15 - x) * (8 - x) * (15 + x) * (8 + x) < -6480.25 + ε :=
by sorry

end min_value_expression_min_value_attained_l2862_286216


namespace simplify_and_evaluate_l2862_286265

theorem simplify_and_evaluate (x : ℝ) (h : x = 2) : (x + 3) * (x - 2) + x * (4 - x) = 4 := by
  sorry

end simplify_and_evaluate_l2862_286265


namespace jane_inspection_fraction_l2862_286261

theorem jane_inspection_fraction 
  (total_rejection_rate : ℝ)
  (john_rejection_rate : ℝ)
  (jane_rejection_rate : ℝ)
  (h_total : total_rejection_rate = 0.0075)
  (h_john : john_rejection_rate = 0.007)
  (h_jane : jane_rejection_rate = 0.008)
  (h_all_inspected : ∀ x y : ℝ, x + y = 1 → 
    x * john_rejection_rate + y * jane_rejection_rate = total_rejection_rate) :
  ∃ y : ℝ, y = 1/2 ∧ ∃ x : ℝ, x + y = 1 ∧
    x * john_rejection_rate + y * jane_rejection_rate = total_rejection_rate :=
by sorry

end jane_inspection_fraction_l2862_286261


namespace f_differentiable_at_sqrt_non_square_l2862_286287

/-- A function f: ℝ → ℝ defined as follows:
    f(x) = 0 if x is irrational
    f(p/q) = 1/q³ if p ∈ ℤ, q ∈ ℕ, and p/q is in lowest terms -/
def f : ℝ → ℝ := sorry

/-- Predicate to check if a natural number is not a perfect square -/
def is_not_perfect_square (k : ℕ) : Prop := ∀ n : ℕ, n^2 ≠ k

theorem f_differentiable_at_sqrt_non_square (k : ℕ) (h : is_not_perfect_square k) :
  DifferentiableAt ℝ f (Real.sqrt k) ∧ deriv f (Real.sqrt k) = 0 := by sorry

end f_differentiable_at_sqrt_non_square_l2862_286287


namespace farm_animal_count_l2862_286223

theorem farm_animal_count :
  ∀ (cows chickens ducks : ℕ),
    (4 * cows + 2 * chickens + 2 * ducks = 20 + 2 * (cows + chickens + ducks)) →
    (chickens + ducks = 2 * cows) →
    cows = 10 := by
  sorry

end farm_animal_count_l2862_286223


namespace pencil_sharpening_theorem_l2862_286228

/-- Calculates the final length of a pencil after sharpening on two consecutive days. -/
def pencil_length (initial_length : ℕ) (day1_sharpening : ℕ) (day2_sharpening : ℕ) : ℕ :=
  initial_length - day1_sharpening - day2_sharpening

/-- Proves that a 22-inch pencil sharpened by 2 inches on two consecutive days will be 18 inches long. -/
theorem pencil_sharpening_theorem :
  pencil_length 22 2 2 = 18 := by
  sorry

end pencil_sharpening_theorem_l2862_286228


namespace min_value_x2_plus_y2_l2862_286221

theorem min_value_x2_plus_y2 (x y : ℝ) (h : 5 * x^2 * y^2 + y^4 = 1) :
  x^2 + y^2 ≥ 4/5 ∧ ∃ x y : ℝ, 5 * x^2 * y^2 + y^4 = 1 ∧ x^2 + y^2 = 4/5 := by
  sorry

end min_value_x2_plus_y2_l2862_286221


namespace unique_five_digit_multiple_of_6_l2862_286205

def is_divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem unique_five_digit_multiple_of_6 :
  ∃! d : ℕ, d < 10 ∧ is_divisible_by_6 (47360 + d) ∧ sum_of_digits (47360 + d) % 3 = 0 :=
by sorry

end unique_five_digit_multiple_of_6_l2862_286205


namespace intersection_points_problem_l2862_286268

/-- The number of intersection points in the first quadrant given points on x and y axes -/
def intersection_points (x_points y_points : ℕ) : ℕ :=
  (x_points.choose 2) * (y_points.choose 2)

/-- The theorem stating the number of intersection points for the given problem -/
theorem intersection_points_problem :
  intersection_points 15 10 = 4725 := by
  sorry

end intersection_points_problem_l2862_286268


namespace solve_for_y_l2862_286203

theorem solve_for_y (x y : ℝ) (h1 : x - y = 16) (h2 : x + y = 4) : y = -6 := by
  sorry

end solve_for_y_l2862_286203


namespace third_test_score_l2862_286262

def maria_scores (score3 : ℝ) : List ℝ := [80, 70, score3, 100]

theorem third_test_score (score3 : ℝ) : 
  (maria_scores score3).sum / (maria_scores score3).length = 85 → score3 = 90 := by
  sorry

end third_test_score_l2862_286262


namespace unique_solution_for_equation_l2862_286277

theorem unique_solution_for_equation : 
  ∃! (x y : ℕ+), 2 * (x : ℕ) ^ (y : ℕ) - (y : ℕ) = 2005 ∧ x = 1003 ∧ y = 1 := by
  sorry

end unique_solution_for_equation_l2862_286277


namespace ratio_equality_l2862_286204

theorem ratio_equality (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0)
  (h_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (h_eq : y / (x - z) = (x + 2*y) / z ∧ (x + 2*y) / z = x / (y + z)) :
  x / (y + z) = (2*y - z) / (y + z) := by
sorry

end ratio_equality_l2862_286204


namespace smallest_sum_of_factors_of_24_l2862_286292

theorem smallest_sum_of_factors_of_24 :
  (∀ a b : ℕ, a * b = 24 → a + b ≥ 10) ∧
  (∃ a b : ℕ, a * b = 24 ∧ a + b = 10) :=
by sorry

end smallest_sum_of_factors_of_24_l2862_286292


namespace polygon_vertices_from_diagonals_l2862_286214

/-- The number of diagonals that can be drawn from one vertex of a polygon. -/
def diagonals_from_vertex (n : ℕ) : ℕ := n - 3

/-- Theorem: A polygon with 6 diagonals from one vertex has 9 vertices. -/
theorem polygon_vertices_from_diagonals :
  ∃ (n : ℕ), n > 2 ∧ diagonals_from_vertex n = 6 → n = 9 :=
by sorry

end polygon_vertices_from_diagonals_l2862_286214


namespace square_and_rectangles_problem_l2862_286202

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a square with side length -/
structure Square where
  side : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- Calculates the area of a square -/
def Square.area (s : Square) : ℝ := s.side * s.side

/-- Theorem statement for the given problem -/
theorem square_and_rectangles_problem
  (small_square : Square)
  (large_rectangle : Rectangle)
  (R : Rectangle)
  (large_square : Square)
  (h1 : small_square.side = 2)
  (h2 : large_rectangle.width = 2 ∧ large_rectangle.height = 4)
  (h3 : small_square.area + large_rectangle.area + R.area = large_square.area)
  : large_square.side = 4 ∧ R.area = 4 := by
  sorry


end square_and_rectangles_problem_l2862_286202


namespace dot_product_parallel_l2862_286245

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define parallel vectors
def parallel (a b : V) : Prop := ∃ k : ℝ, a = k • b ∨ b = k • a

theorem dot_product_parallel (a b : V) (ha : a ≠ 0) (hb : b ≠ 0) :
  (inner a b = ‖a‖ * ‖b‖ → parallel a b) ∧
  ¬(parallel a b → inner a b = ‖a‖ * ‖b‖) :=
sorry

end dot_product_parallel_l2862_286245


namespace bank_interest_rate_is_five_percent_l2862_286272

/-- Proves that the bank interest rate is 5% given the investment conditions -/
theorem bank_interest_rate_is_five_percent 
  (total_investment : ℝ)
  (bank_investment : ℝ)
  (bond_investment : ℝ)
  (total_annual_income : ℝ)
  (bond_return_rate : ℝ)
  (h1 : total_investment = 10000)
  (h2 : bank_investment = 6000)
  (h3 : bond_investment = 4000)
  (h4 : total_annual_income = 660)
  (h5 : bond_return_rate = 0.09)
  (h6 : total_investment = bank_investment + bond_investment)
  (h7 : total_annual_income = bank_investment * bank_interest_rate + bond_investment * bond_return_rate) :
  bank_interest_rate = 0.05 := by
  sorry

#check bank_interest_rate_is_five_percent

end bank_interest_rate_is_five_percent_l2862_286272


namespace basketball_teams_count_l2862_286229

/-- The number of combinations of n items taken k at a time -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The total number of people available for the basketball game -/
def total_people : ℕ := 7

/-- The number of players needed for each team -/
def team_size : ℕ := 4

/-- Theorem: The number of different teams of 4 that can be formed from 7 people is 35 -/
theorem basketball_teams_count : binomial total_people team_size = 35 := by sorry

end basketball_teams_count_l2862_286229


namespace fence_cost_l2862_286243

/-- The cost of fencing a square plot -/
theorem fence_cost (area : ℝ) (price_per_foot : ℝ) (cost : ℝ) : 
  area = 49 → price_per_foot = 58 → cost = 1624 := by
  sorry

end fence_cost_l2862_286243


namespace stating_escalator_step_count_l2862_286230

/-- Represents the number of steps counted on an escalator under different conditions -/
structure EscalatorSteps where
  down : ℕ  -- steps counted running down
  up : ℕ    -- steps counted running up
  stationary : ℕ  -- steps counted on a stationary escalator

/-- 
Given the number of steps counted running down and up a moving escalator,
calculates the number of steps on a stationary escalator
-/
def calculateStationarySteps (e : EscalatorSteps) : Prop :=
  e.down = 30 ∧ e.up = 150 → e.stationary = 50

/-- 
Theorem stating that if a person counts 30 steps running down a moving escalator
and 150 steps running up the same escalator at the same speed relative to the escalator,
then they would count 50 steps on a stationary escalator
-/
theorem escalator_step_count : ∃ e : EscalatorSteps, calculateStationarySteps e :=
  sorry

end stating_escalator_step_count_l2862_286230


namespace simplify_expression_l2862_286227

theorem simplify_expression (x y : ℝ) :
  4 * x + 8 * x^2 + y^3 + 6 - (3 - 4 * x - 8 * x^2 - y^3) = 16 * x^2 + 8 * x + 2 * y^3 + 3 :=
by sorry

end simplify_expression_l2862_286227


namespace product_simplification_l2862_286222

theorem product_simplification : 
  (1 + 2 / 1) * (1 + 2 / 2) * (1 + 2 / 3) * (1 + 2 / 4) * (1 + 2 / 5) * (1 + 2 / 6) - 1 = 25 / 3 := by
  sorry

end product_simplification_l2862_286222


namespace rent_reduction_percentage_l2862_286207

-- Define the room prices
def cheap_room_price : ℕ := 40
def expensive_room_price : ℕ := 60

-- Define the total rent
def total_rent : ℕ := 1000

-- Define the number of rooms to be moved
def rooms_to_move : ℕ := 10

-- Define the function to calculate the new total rent
def new_total_rent : ℕ := total_rent - rooms_to_move * (expensive_room_price - cheap_room_price)

-- Define the reduction percentage
def reduction_percentage : ℚ := (total_rent - new_total_rent : ℚ) / total_rent * 100

-- Theorem statement
theorem rent_reduction_percentage :
  reduction_percentage = 20 :=
sorry

end rent_reduction_percentage_l2862_286207


namespace least_x_for_1894x_divisible_by_3_l2862_286206

theorem least_x_for_1894x_divisible_by_3 : 
  ∃ x : ℕ, (∀ y : ℕ, y < x → ¬(3 ∣ 1894 * y)) ∧ (3 ∣ 1894 * x) :=
by
  -- The proof goes here
  sorry

end least_x_for_1894x_divisible_by_3_l2862_286206


namespace find_N_l2862_286249

theorem find_N : ∃ N : ℚ, (5 + 6 + 7 + 8) / 4 = (2014 + 2015 + 2016 + 2017) / N ∧ N = 1240 := by
  sorry

end find_N_l2862_286249


namespace factor_and_divisor_statements_l2862_286236

-- Define what it means for a number to be a factor of another
def is_factor (a b : ℤ) : Prop := ∃ k : ℤ, b = a * k

-- Define what it means for a number to be a divisor of another
def is_divisor (a b : ℤ) : Prop := ∃ k : ℤ, b = a * k

theorem factor_and_divisor_statements :
  (is_factor 4 100) ∧
  (is_divisor 19 133 ∧ ¬ is_divisor 19 51) ∧
  (is_divisor 30 90 ∨ is_divisor 30 53) ∧
  (is_divisor 7 21 ∧ is_divisor 7 49) ∧
  (is_factor 10 200) := by sorry

end factor_and_divisor_statements_l2862_286236


namespace dorothy_interest_l2862_286281

/-- Calculates the interest earned on an investment with annual compound interest. -/
def interest_earned (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years - principal

/-- The interest earned on Dorothy's investment -/
theorem dorothy_interest : 
  let principal := 2000
  let rate := 0.02
  let years := 3
  ⌊interest_earned principal rate years⌋ = 122 := by
  sorry

end dorothy_interest_l2862_286281


namespace book_cost_proof_l2862_286286

/-- Given that Mark started with $85, bought 10 books, and was left with $35, prove that each book cost $5. -/
theorem book_cost_proof (initial_amount : ℕ) (books_bought : ℕ) (remaining_amount : ℕ) :
  initial_amount = 85 ∧ books_bought = 10 ∧ remaining_amount = 35 →
  (initial_amount - remaining_amount) / books_bought = 5 :=
by sorry

end book_cost_proof_l2862_286286


namespace line_intersection_triangle_l2862_286219

-- Define a Point type
structure Point :=
  (x : ℝ) (y : ℝ)

-- Define a Line type
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

-- Define a function to check if three points are collinear
def collinear (A B C : Point) : Prop :=
  (B.y - A.y) * (C.x - A.x) = (C.y - A.y) * (B.x - A.x)

-- Define a function to check if a point lies on a line
def pointOnLine (P : Point) (L : Line) : Prop :=
  L.a * P.x + L.b * P.y + L.c = 0

-- Define a function to check if a line intersects a segment
def lineIntersectsSegment (L : Line) (A B : Point) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧
    pointOnLine (Point.mk (A.x + t * (B.x - A.x)) (A.y + t * (B.y - A.y))) L

-- Main theorem
theorem line_intersection_triangle (A B C : Point) (L : Line) :
  ¬collinear A B C →
  ¬pointOnLine A L →
  ¬pointOnLine B L →
  ¬pointOnLine C L →
  (¬lineIntersectsSegment L B C ∧ ¬lineIntersectsSegment L C A ∧ ¬lineIntersectsSegment L A B) ∨
  (lineIntersectsSegment L B C ∧ lineIntersectsSegment L C A ∧ ¬lineIntersectsSegment L A B) ∨
  (lineIntersectsSegment L B C ∧ ¬lineIntersectsSegment L C A ∧ lineIntersectsSegment L A B) ∨
  (¬lineIntersectsSegment L B C ∧ lineIntersectsSegment L C A ∧ lineIntersectsSegment L A B) :=
by sorry

end line_intersection_triangle_l2862_286219


namespace correct_factorization_l2862_286273

theorem correct_factorization (x : ℝ) : x^2 - 3*x + 2 = (x - 1)*(x - 2) := by
  sorry

end correct_factorization_l2862_286273


namespace right_triangle_area_perimeter_l2862_286201

/-- A right triangle with hypotenuse 13 and one leg 5 has area 30 and perimeter 30 -/
theorem right_triangle_area_perimeter :
  ∀ (a b c : ℝ),
  a = 5 →
  c = 13 →
  a^2 + b^2 = c^2 →
  (1/2 * a * b = 30) ∧ (a + b + c = 30) :=
by sorry

end right_triangle_area_perimeter_l2862_286201


namespace imaginary_power_sum_product_l2862_286209

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Define the periodicity of i
axiom i_period (n : ℕ) : i^(n + 4) = i^n

-- State the theorem
theorem imaginary_power_sum_product : (i^22 + i^222) * i = -2 * i := by sorry

end imaginary_power_sum_product_l2862_286209


namespace min_sum_of_positive_integers_l2862_286289

theorem min_sum_of_positive_integers (a b : ℕ+) (h : a.val * b.val - 7 * a.val - 11 * b.val + 13 = 0) :
  ∃ (a₀ b₀ : ℕ+), a₀.val * b₀.val - 7 * a₀.val - 11 * b₀.val + 13 = 0 ∧
    ∀ (x y : ℕ+), x.val * y.val - 7 * x.val - 11 * y.val + 13 = 0 → a₀.val + b₀.val ≤ x.val + y.val ∧
    a₀.val + b₀.val = 34 :=
by sorry

end min_sum_of_positive_integers_l2862_286289


namespace marys_overtime_rate_increase_l2862_286233

/-- Represents Mary's work schedule and pay structure --/
structure WorkSchedule where
  maxHours : ℕ
  regularHours : ℕ
  regularRate : ℚ
  maxWeeklyEarnings : ℚ

/-- Calculates the percentage increase in overtime rate compared to regular rate --/
def overtimeRateIncrease (w : WorkSchedule) : ℚ :=
  let regularEarnings := w.regularRate * w.regularHours
  let overtimeEarnings := w.maxWeeklyEarnings - regularEarnings
  let overtimeHours := w.maxHours - w.regularHours
  let overtimeRate := overtimeEarnings / overtimeHours
  ((overtimeRate - w.regularRate) / w.regularRate) * 100

/-- Mary's work schedule --/
def marysSchedule : WorkSchedule :=
  { maxHours := 50
  , regularHours := 20
  , regularRate := 8
  , maxWeeklyEarnings := 460 }

/-- Theorem stating that Mary's overtime rate increase is 25% --/
theorem marys_overtime_rate_increase :
  overtimeRateIncrease marysSchedule = 25 := by
  sorry


end marys_overtime_rate_increase_l2862_286233


namespace hexahedron_faces_l2862_286263

/-- A hexahedron is a polyhedron with six faces -/
structure Hexahedron where
  -- We don't need to define the internal structure, just the concept

/-- The number of faces of a hexahedron -/
def num_faces (h : Hexahedron) : ℕ := sorry

/-- Theorem: The number of faces of a hexahedron is 6 -/
theorem hexahedron_faces (h : Hexahedron) : num_faces h = 6 := by
  sorry

end hexahedron_faces_l2862_286263


namespace more_spins_more_accurate_l2862_286291

/-- Represents a spinner used in random simulation -/
structure Spinner :=
  (radius : ℝ)

/-- Represents the result of a spinner simulation -/
structure SimulationResult :=
  (accuracy : ℝ)

/-- Represents a random simulation using a spinner -/
def SpinnerSimulation := Spinner → ℕ → SimulationResult

/-- Axiom: The spinner must be spun randomly for accurate estimation -/
axiom random_spinning_required (s : Spinner) (n : ℕ) (sim : SpinnerSimulation) :
  SimulationResult

/-- Axiom: The number of spins affects the estimation accuracy -/
axiom spins_affect_accuracy (s : Spinner) (n m : ℕ) (sim : SpinnerSimulation) :
  n ≠ m → sim s n ≠ sim s m

/-- Axiom: The spinner's radius does not affect the estimation accuracy -/
axiom radius_doesnt_affect_accuracy (s₁ s₂ : Spinner) (n : ℕ) (sim : SpinnerSimulation) :
  s₁.radius ≠ s₂.radius → sim s₁ n = sim s₂ n

/-- Theorem: Increasing the number of spins improves the accuracy of the estimation result -/
theorem more_spins_more_accurate (s : Spinner) (n m : ℕ) (sim : SpinnerSimulation) :
  n < m → (sim s m).accuracy > (sim s n).accuracy :=
sorry

end more_spins_more_accurate_l2862_286291


namespace line_through_point_l2862_286280

/-- Given a line equation ax + (a+4)y = a + 5 passing through (5, -10), prove a = -7.5 -/
theorem line_through_point (a : ℝ) : 
  (∀ x y : ℝ, a * x + (a + 4) * y = a + 5 → x = 5 ∧ y = -10) → 
  a = -7.5 := by sorry

end line_through_point_l2862_286280


namespace journey_speed_calculation_l2862_286298

theorem journey_speed_calculation (D : ℝ) (v : ℝ) (h1 : D > 0) (h2 : v > 0) : 
  (D / ((0.8 * D / 80) + (0.2 * D / v)) = 50) → v = 20 := by
  sorry

end journey_speed_calculation_l2862_286298


namespace f_roots_and_monotonicity_imply_b_range_l2862_286293

/-- The function f(x) = -x^3 + bx -/
def f (b : ℝ) (x : ℝ) : ℝ := -x^3 + b*x

/-- Theorem: If all roots of f(x) = 0 are within [-2, 2] and f(x) is monotonically increasing in (0, 1), then 3 ≤ b ≤ 4 -/
theorem f_roots_and_monotonicity_imply_b_range (b : ℝ) :
  (∀ x, f b x = 0 → x ∈ Set.Icc (-2) 2) →
  (∀ x ∈ Set.Ioo 0 1, ∀ y ∈ Set.Ioo 0 1, x < y → f b x < f b y) →
  b ∈ Set.Icc 3 4 := by
  sorry

end f_roots_and_monotonicity_imply_b_range_l2862_286293


namespace right_triangle_condition_l2862_286253

theorem right_triangle_condition (A B C : ℝ) (a b c : ℝ) :
  (A + B + C = π) →
  (a * Real.cos C + c * Real.cos A = b * Real.sin B) →
  (a * Real.sin A = b * Real.sin B) →
  (b * Real.sin B = c * Real.sin C) →
  (B = π / 2) :=
sorry

end right_triangle_condition_l2862_286253


namespace perpendicular_lines_condition_l2862_286296

/-- Two lines y = m₁x + b₁ and y = m₂x + b₂ are perpendicular if and only if m₁ * m₂ = -1 -/
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

/-- The statement "a = 2 is a sufficient but not necessary condition for the lines
    y = -ax + 2 and y = (a/4)x - 1 to be perpendicular" -/
theorem perpendicular_lines_condition (a : ℝ) :
  (a = 2 → perpendicular (-a) (a/4)) ∧ 
  ¬(perpendicular (-a) (a/4) → a = 2) :=
sorry

end perpendicular_lines_condition_l2862_286296


namespace target_breaking_sequences_l2862_286283

/-- The number of unique permutations of a string with repeated characters -/
def multinomial_permutations (char_counts : List Nat) : Nat :=
  Nat.factorial (char_counts.sum) / (char_counts.map Nat.factorial).prod

/-- The target arrangement represented as character counts -/
def target_arrangement : List Nat := [4, 3, 3]

theorem target_breaking_sequences :
  multinomial_permutations target_arrangement = 4200 := by
  sorry

end target_breaking_sequences_l2862_286283


namespace cube_volume_doubling_l2862_286259

theorem cube_volume_doubling (v : ℝ) (h : v = 27) :
  let new_volume := (2 * v^(1/3))^3
  new_volume = 216 := by
sorry

end cube_volume_doubling_l2862_286259


namespace power_of_square_l2862_286200

theorem power_of_square (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end power_of_square_l2862_286200


namespace distance_satisfies_conditions_l2862_286279

/-- The distance traveled by both the train and the ship -/
def distance : ℝ := 480

/-- The speed of the train in km/h -/
def train_speed : ℝ := 48

/-- The speed of the ship in km/h -/
def ship_speed : ℝ := 60

/-- The time difference between the train and ship journeys in hours -/
def time_difference : ℝ := 2

/-- Theorem stating that the given distance satisfies the problem conditions -/
theorem distance_satisfies_conditions :
  distance / train_speed = distance / ship_speed + time_difference :=
by sorry

end distance_satisfies_conditions_l2862_286279


namespace cube_root_equation_solution_l2862_286248

theorem cube_root_equation_solution :
  ∃ x : ℝ, (x^(1/3) * (x^5)^(1/4))^(1/3) = 4 ∧ x = 2^(8/3) := by
  sorry

end cube_root_equation_solution_l2862_286248


namespace train_length_l2862_286240

/-- The length of a train given its speed, time to cross a bridge, and the bridge's length -/
theorem train_length (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) :
  train_speed = 45 * (1000 / 3600) →
  crossing_time = 30 →
  bridge_length = 265 →
  train_speed * crossing_time - bridge_length = 110 := by
  sorry

end train_length_l2862_286240


namespace median_of_special_list_l2862_286276

def list_sum (n : ℕ) : ℕ := n * (n + 1) / 2

theorem median_of_special_list : 
  let total_elements : ℕ := list_sum 100
  let median_position : ℕ := total_elements / 2
  let cumulative_count (k : ℕ) : ℕ := list_sum k
  ∃ n : ℕ, 
    cumulative_count n ≥ median_position ∧ 
    cumulative_count (n-1) < median_position ∧
    n = 71 := by
  sorry

#check median_of_special_list

end median_of_special_list_l2862_286276


namespace inequality_solution_set_l2862_286275

theorem inequality_solution_set (a : ℝ) (h : a > 1) :
  {x : ℝ | (x - a) * (x - 1/a) > 0} = {x : ℝ | x < 1/a ∨ x > a} := by sorry

end inequality_solution_set_l2862_286275


namespace license_plate_count_l2862_286234

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of positions for letters on the license plate -/
def letter_positions : ℕ := 4

/-- The number of digits on the license plate -/
def digit_positions : ℕ := 2

/-- The number of possible digits (0-9) -/
def digit_options : ℕ := 10

/-- Calculates the number of license plate combinations -/
def license_plate_combinations : ℕ :=
  alphabet_size * (alphabet_size - 1).choose 2 * letter_positions.choose 2 * 2 * digit_options * (digit_options - 1)

theorem license_plate_count :
  license_plate_combinations = 8424000 :=
by sorry

end license_plate_count_l2862_286234


namespace work_completion_time_l2862_286290

theorem work_completion_time 
  (total_work : ℝ) 
  (a b c : ℝ) 
  (h1 : a + b + c = total_work / 4)  -- a, b, and c together finish in 4 days
  (h2 : b = total_work / 9)          -- b alone finishes in 9 days
  (h3 : c = total_work / 18)         -- c alone finishes in 18 days
  : a = total_work / 12 :=           -- a alone finishes in 12 days
by sorry

end work_completion_time_l2862_286290


namespace size_relationship_l2862_286213

theorem size_relationship : 5^30 < 3^50 ∧ 3^50 < 4^40 := by
  sorry

end size_relationship_l2862_286213


namespace octagon_area_l2862_286231

/-- The area of an octagon formed by the intersection of two concentric squares -/
theorem octagon_area (side_large : ℝ) (side_small : ℝ) (octagon_side : ℝ) : 
  side_large = 2 →
  side_small = 1 →
  octagon_side = 17/36 →
  let octagon_area := 8 * (1/2 * octagon_side * side_small)
  octagon_area = 17/9 := by
sorry

end octagon_area_l2862_286231


namespace janice_starting_sentences_janice_starting_sentences_proof_l2862_286242

/-- Proves the number of sentences Janice started with today -/
theorem janice_starting_sentences : ℕ :=
  let typing_speed : ℕ := 6  -- sentences per minute
  let first_session : ℕ := 20  -- minutes
  let second_session : ℕ := 15  -- minutes
  let third_session : ℕ := 18  -- minutes
  let erased_sentences : ℕ := 40
  let total_sentences : ℕ := 536

  let total_typed : ℕ := typing_speed * (first_session + second_session + third_session)
  let net_added : ℕ := total_typed - erased_sentences
  
  total_sentences - net_added

/-- The theorem statement -/
theorem janice_starting_sentences_proof : janice_starting_sentences = 258 := by
  sorry

end janice_starting_sentences_janice_starting_sentences_proof_l2862_286242


namespace expression_evaluation_l2862_286238

theorem expression_evaluation : 1 - (-2) - 3 - (-4) - 5 - (-6) = 5 := by
  sorry

end expression_evaluation_l2862_286238


namespace students_playing_all_sports_l2862_286278

/-- The number of students playing all three sports in a school with given sport participation data -/
theorem students_playing_all_sports (total : ℕ) (football cricket basketball : ℕ) 
  (neither : ℕ) (football_cricket football_basketball cricket_basketball : ℕ) :
  total = 580 →
  football = 300 →
  cricket = 250 →
  basketball = 180 →
  neither = 60 →
  football_cricket = 120 →
  football_basketball = 80 →
  cricket_basketball = 70 →
  ∃ (all_sports : ℕ), 
    all_sports = 140 ∧
    total = football + cricket + basketball - football_cricket - football_basketball - cricket_basketball + all_sports + neither :=
by sorry

end students_playing_all_sports_l2862_286278


namespace integral_sine_product_zero_and_no_beta_solution_l2862_286235

theorem integral_sine_product_zero_and_no_beta_solution 
  (m n : ℕ) (h_distinct : m ≠ n) (h_positive_m : m > 0) (h_positive_n : n > 0) :
  (∀ α : ℝ, |α| < 1 → ∫ x in -π..π, Real.sin ((m : ℝ) + α) * x * Real.sin ((n : ℝ) + α) * x = 0) ∧
  ¬ ∃ β : ℝ, (∫ x in -π..π, Real.sin ((m : ℝ) + β) * x ^ 2 = π + 2 / (4 * m - 1)) ∧
             (∫ x in -π..π, Real.sin ((n : ℝ) + β) * x ^ 2 = π + 2 / (4 * n - 1)) :=
by sorry

end integral_sine_product_zero_and_no_beta_solution_l2862_286235


namespace chord_length_for_60_degree_line_and_circle_l2862_286254

/-- The length of the chord formed by the intersection of a line passing through the origin
    with a slope angle of 60° and the circle x² + y² - 4y = 0 is equal to 2√3. -/
theorem chord_length_for_60_degree_line_and_circle : 
  let line := {(x, y) : ℝ × ℝ | y = Real.sqrt 3 * x}
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 - 4*y = 0}
  let chord := line ∩ circle
  ∃ p q : ℝ × ℝ, p ∈ chord ∧ q ∈ chord ∧ p ≠ q ∧ 
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = 2 * Real.sqrt 3 :=
by sorry

end chord_length_for_60_degree_line_and_circle_l2862_286254


namespace min_value_of_f_l2862_286264

/-- The function f(n) = n^2 - 8n + 5 -/
def f (n : ℝ) : ℝ := n^2 - 8*n + 5

/-- The minimum value of f(n) is -11 -/
theorem min_value_of_f : ∀ n : ℝ, f n ≥ -11 ∧ ∃ n₀ : ℝ, f n₀ = -11 := by
  sorry

end min_value_of_f_l2862_286264


namespace arccos_neg_one_eq_pi_l2862_286232

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = Real.pi := by sorry

end arccos_neg_one_eq_pi_l2862_286232


namespace complex_fraction_simplification_l2862_286252

theorem complex_fraction_simplification :
  1 + 1 / (1 + 1 / (2 + 2)) = 9 / 5 := by sorry

end complex_fraction_simplification_l2862_286252


namespace coconut_juice_unit_electric_water_heater_unit_l2862_286244

-- Define the types of containers
inductive Container
| CoconutJuiceBottle
| ElectricWaterHeater

-- Define the volume units
inductive VolumeUnit
| Milliliter
| Liter

-- Define a function to get the appropriate volume unit for a container
def appropriateUnit (container : Container) (volume : ℕ) : VolumeUnit :=
  match container with
  | Container.CoconutJuiceBottle => VolumeUnit.Milliliter
  | Container.ElectricWaterHeater => VolumeUnit.Liter

-- Theorem for coconut juice bottle
theorem coconut_juice_unit : 
  appropriateUnit Container.CoconutJuiceBottle 200 = VolumeUnit.Milliliter :=
by sorry

-- Theorem for electric water heater
theorem electric_water_heater_unit : 
  appropriateUnit Container.ElectricWaterHeater 50 = VolumeUnit.Liter :=
by sorry

end coconut_juice_unit_electric_water_heater_unit_l2862_286244


namespace mixed_fraction_calculation_l2862_286297

theorem mixed_fraction_calculation : 
  (-4 - 2/3) - (1 + 5/6) - (-18 - 1/2) + (-13 - 3/4) = -7/4 := by
  sorry

end mixed_fraction_calculation_l2862_286297


namespace tilly_star_count_l2862_286284

theorem tilly_star_count (stars_east : ℕ) (stars_west : ℕ) : 
  stars_east = 120 →
  stars_west = 6 * stars_east →
  stars_east + stars_west = 840 := by
sorry

end tilly_star_count_l2862_286284


namespace units_produced_l2862_286260

def fixed_costs : ℕ := 15000
def variable_cost_per_unit : ℕ := 300
def total_cost : ℕ := 27500

def total_cost_function (n : ℕ) : ℕ :=
  fixed_costs + n * variable_cost_per_unit

theorem units_produced : ∃ (n : ℕ), n > 0 ∧ n ≤ 50 ∧ total_cost_function n = total_cost :=
sorry

end units_produced_l2862_286260


namespace quadratic_solution_difference_squared_l2862_286256

theorem quadratic_solution_difference_squared :
  ∀ f g : ℝ,
  (2 * f^2 + 8 * f - 42 = 0) →
  (2 * g^2 + 8 * g - 42 = 0) →
  (f ≠ g) →
  (f - g)^2 = 100 := by
  sorry

end quadratic_solution_difference_squared_l2862_286256


namespace income_calculation_l2862_286226

theorem income_calculation (income expenditure savings : ℕ) : 
  income * 4 = expenditure * 5 →
  income - expenditure = savings →
  savings = 3400 →
  income = 17000 := by
sorry

end income_calculation_l2862_286226


namespace square_greater_than_self_when_less_than_negative_one_l2862_286295

theorem square_greater_than_self_when_less_than_negative_one (x : ℝ) : 
  x < -1 → x^2 > x := by
  sorry

end square_greater_than_self_when_less_than_negative_one_l2862_286295


namespace mean_temperature_l2862_286210

def temperatures : List ℝ := [-6.5, -2, -3.5, -1, 0.5, 4, 1.5]

theorem mean_temperature : (temperatures.sum / temperatures.length) = -1 := by
  sorry

end mean_temperature_l2862_286210


namespace winning_post_at_200m_l2862_286288

/-- Two runners A and B, where A is faster than B but gives B a head start -/
structure RaceScenario where
  /-- The speed ratio of runner A to runner B -/
  speed_ratio : ℚ
  /-- The head start given to runner B in meters -/
  head_start : ℚ

/-- The winning post distance for two runners to arrive simultaneously -/
def winning_post_distance (scenario : RaceScenario) : ℚ :=
  (scenario.speed_ratio * scenario.head_start) / (scenario.speed_ratio - 1)

/-- Theorem stating that for the given scenario, the winning post distance is 200 meters -/
theorem winning_post_at_200m (scenario : RaceScenario) 
  (h1 : scenario.speed_ratio = 5/3)
  (h2 : scenario.head_start = 80) :
  winning_post_distance scenario = 200 := by
  sorry

end winning_post_at_200m_l2862_286288


namespace intersecting_plane_theorem_l2862_286224

/-- Represents a 3D cube composed of unit cubes -/
structure Cube where
  side_length : ℕ
  total_units : ℕ

/-- Represents a plane intersecting the cube -/
structure IntersectingPlane where
  perpendicular_to_diagonal : Bool
  distance_ratio : ℚ

/-- Calculates the number of unit cubes intersected by the plane -/
def intersected_cubes (c : Cube) (p : IntersectingPlane) : ℕ :=
  sorry

/-- Theorem stating that a plane intersecting a 4x4x4 cube at 1/4 of its diagonal intersects 36 unit cubes -/
theorem intersecting_plane_theorem (c : Cube) (p : IntersectingPlane) :
  c.side_length = 4 ∧ c.total_units = 64 ∧ p.perpendicular_to_diagonal = true ∧ p.distance_ratio = 1/4 →
  intersected_cubes c p = 36 :=
sorry

end intersecting_plane_theorem_l2862_286224


namespace percentage_calculation_l2862_286215

theorem percentage_calculation : 
  (789524.37 : ℝ) * (7.5 / 100) = 59214.32825 := by sorry

end percentage_calculation_l2862_286215


namespace highway_length_is_500_l2862_286237

/-- The length of a highway where two cars meet after traveling from opposite ends -/
def highway_length (speed1 speed2 : ℝ) (time : ℝ) : ℝ :=
  speed1 * time + speed2 * time

/-- Theorem stating the length of the highway is 500 miles -/
theorem highway_length_is_500 :
  highway_length 40 60 5 = 500 := by
  sorry

end highway_length_is_500_l2862_286237


namespace train_speed_l2862_286251

/-- The speed of a train given its length and time to cross an electric pole -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 150) (h2 : time = 3) :
  length / time = 50 := by
  sorry

end train_speed_l2862_286251


namespace tangent_slope_condition_l2862_286269

/-- The curve function -/
def f (a : ℝ) (x : ℝ) : ℝ := x^5 - a*(x + 1)

/-- The derivative of the curve function -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 5*x^4 - a

theorem tangent_slope_condition (a : ℝ) :
  (f_derivative a 1 > 1) ↔ (a < 4) := by sorry

end tangent_slope_condition_l2862_286269


namespace quadratic_inequality_solution_set_l2862_286255

-- Define the quadratic function
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 - 5 * x + b

-- Define the solution set type
def SolutionSet := Set ℝ

-- State the theorem
theorem quadratic_inequality_solution_set 
  (a b : ℝ) 
  (h1 : a = -5) 
  (h2 : b = 30) 
  (h3 : {x : ℝ | f a b x > 0} = {x : ℝ | -3 < x ∧ x < 2}) :
  {x : ℝ | f b (-a) x > 0} = {x : ℝ | x < -1/3 ∨ x > 1/2} :=
sorry

end quadratic_inequality_solution_set_l2862_286255


namespace square_boundary_length_l2862_286258

/-- The total length of the boundary created by quarter-circle arcs and straight segments
    in a square with area 144, where each side is divided into thirds and quarters. -/
theorem square_boundary_length : ∃ (l : ℝ),
  l = 12 * Real.pi + 16 ∧ 
  (∃ (s : ℝ), s^2 = 144 ∧ 
    l = 4 * (2 * Real.pi * (s / 3) / 4 + Real.pi * (s / 6) / 4) + 4 * (s / 3)) :=
by sorry

end square_boundary_length_l2862_286258


namespace next_roll_for_average_three_l2862_286271

def rolls : List Nat := [1, 3, 2, 4, 3, 5, 3, 4, 4, 2]

theorem next_roll_for_average_three (rolls : List Nat) : 
  rolls.length = 10 → 
  rolls.sum = 31 → 
  ∃ (next_roll : Nat), 
    (rolls.sum + next_roll) / (rolls.length + 1 : Nat) = 3 ∧ 
    next_roll = 2 := by
  sorry

#check next_roll_for_average_three rolls

end next_roll_for_average_three_l2862_286271


namespace distance_center_to_origin_l2862_286211

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0

-- Define the center of the circle
def center_C : ℝ × ℝ := (1, 0)

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Theorem statement
theorem distance_center_to_origin :
  Real.sqrt ((center_C.1 - origin.1)^2 + (center_C.2 - origin.2)^2) = 1 := by
  sorry

end distance_center_to_origin_l2862_286211


namespace f_at_2_l2862_286212

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

-- State the theorem
theorem f_at_2 (a b : ℝ) : f a b (-2) = 3 → f a b 2 = -19 := by sorry

end f_at_2_l2862_286212


namespace rectangle_vertex_numbers_l2862_286282

theorem rectangle_vertex_numbers (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h1 : 2 * a ≥ b + d)
  (h2 : 2 * b ≥ a + c)
  (h3 : 2 * c ≥ b + d)
  (h4 : 2 * d ≥ a + c) :
  a = b ∧ b = c ∧ c = d :=
sorry

end rectangle_vertex_numbers_l2862_286282


namespace quadratic_equation_range_l2862_286267

theorem quadratic_equation_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + a*x + a = 0) → a ∈ Set.Iic 0 ∪ Set.Ici 4 := by
  sorry

end quadratic_equation_range_l2862_286267
