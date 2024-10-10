import Mathlib

namespace indefinite_integral_proof_l931_93158

open Real

theorem indefinite_integral_proof (x : ℝ) : 
  deriv (fun x => (1/2) * log (abs (x^2 - x + 1)) + 
                  Real.sqrt 3 * arctan ((2*x - 1) / Real.sqrt 3) + 
                  (1/2) * log (abs (x^2 + 1))) x = 
  (2*x^3 + 2*x + 1) / ((x^2 - x + 1) * (x^2 + 1)) :=
by sorry

end indefinite_integral_proof_l931_93158


namespace true_false_questions_count_l931_93144

/-- Proves that the number of true/false questions is 6 given the conditions of the problem -/
theorem true_false_questions_count :
  ∀ (T F M : ℕ),
  T + F + M = 45 →
  M = 2 * F →
  F = T + 7 →
  T = 6 := by
sorry

end true_false_questions_count_l931_93144


namespace curve_c_properties_l931_93165

/-- The curve C in a rectangular coordinate system -/
structure CurveC where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Point on the curve C -/
structure PointOnC (c : CurveC) where
  φ : ℝ
  x : ℝ
  y : ℝ
  h_x : x = c.a * Real.cos φ
  h_y : y = c.b * Real.sin φ

/-- Theorem about the curve C -/
theorem curve_c_properties (c : CurveC) 
  (m : PointOnC c) 
  (h_m_x : m.x = 2) 
  (h_m_y : m.y = Real.sqrt 3) 
  (h_m_φ : m.φ = π / 3) :
  (∀ x y, x^2 / 16 + y^2 / 4 = 1 ↔ ∃ φ, x = c.a * Real.cos φ ∧ y = c.b * Real.sin φ) ∧
  (∀ ρ₁ ρ₂ θ, 
    (∃ φ₁, ρ₁ * Real.cos θ = c.a * Real.cos φ₁ ∧ ρ₁ * Real.sin θ = c.b * Real.sin φ₁) →
    (∃ φ₂, ρ₂ * Real.cos (θ + π/2) = c.a * Real.cos φ₂ ∧ ρ₂ * Real.sin (θ + π/2) = c.b * Real.sin φ₂) →
    1 / ρ₁^2 + 1 / ρ₂^2 = 5 / 16) :=
by sorry

end curve_c_properties_l931_93165


namespace tank_weight_l931_93175

/-- Proves that the weight of a water tank filled to 80% capacity is 1360 pounds. -/
theorem tank_weight (tank_capacity : ℝ) (empty_tank_weight : ℝ) (water_weight_per_gallon : ℝ) :
  tank_capacity = 200 →
  empty_tank_weight = 80 →
  water_weight_per_gallon = 8 →
  empty_tank_weight + 0.8 * tank_capacity * water_weight_per_gallon = 1360 := by
  sorry

end tank_weight_l931_93175


namespace negation_equivalence_l931_93180

theorem negation_equivalence : 
  (¬ ∃ x₀ : ℝ, x₀ ≤ 0 ∧ x₀^2 ≥ 0) ↔ (∀ x : ℝ, x ≤ 0 → x^2 < 0) :=
by sorry

end negation_equivalence_l931_93180


namespace direction_cosines_sum_of_squares_l931_93117

/-- Direction cosines of a vector in 3D space -/
structure DirectionCosines where
  α : ℝ
  β : ℝ
  γ : ℝ

/-- Theorem: The sum of squares of direction cosines equals 1 -/
theorem direction_cosines_sum_of_squares (dc : DirectionCosines) : 
  dc.α^2 + dc.β^2 + dc.γ^2 = 1 := by
  sorry

end direction_cosines_sum_of_squares_l931_93117


namespace calculate_gross_profit_gross_profit_calculation_l931_93184

/-- Given a sales price and a gross profit percentage, calculate the gross profit --/
theorem calculate_gross_profit (sales_price : ℝ) (gross_profit_percentage : ℝ) : ℝ :=
  let cost := sales_price / (1 + gross_profit_percentage)
  let gross_profit := cost * gross_profit_percentage
  gross_profit

/-- Prove that given a sales price of $81 and a gross profit that is 170% of cost, 
    the gross profit is equal to $51 --/
theorem gross_profit_calculation :
  calculate_gross_profit 81 1.7 = 51 := by
  sorry

end calculate_gross_profit_gross_profit_calculation_l931_93184


namespace base_k_theorem_l931_93162

theorem base_k_theorem (k : ℕ) (h : k > 0) : 
  (1 * k^2 + 3 * k + 2 = 30) → k = 4 := by
  sorry

end base_k_theorem_l931_93162


namespace right_triangle_area_l931_93177

theorem right_triangle_area (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a = (4/3) * b) (h5 : a = (2/3) * c) (h6 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 2/3 :=
sorry

end right_triangle_area_l931_93177


namespace test_score_proof_l931_93152

theorem test_score_proof (total_questions : ℕ) (correct_points : ℕ) (incorrect_points : ℕ) (total_score : ℕ) :
  total_questions = 30 →
  correct_points = 20 →
  incorrect_points = 5 →
  total_score = 325 →
  ∃ (correct_answers : ℕ),
    correct_answers * correct_points - (total_questions - correct_answers) * incorrect_points = total_score ∧
    correct_answers = 19 :=
by sorry

end test_score_proof_l931_93152


namespace optimal_strategy_is_valid_l931_93113

/-- Represents a chain of links -/
structure Chain where
  links : ℕ

/-- Represents a cut in the chain -/
structure Cut where
  position : ℕ

/-- Represents a payment strategy for the hotel stay -/
structure PaymentStrategy where
  cut : Cut
  dailyPayments : List ℕ

/-- Checks if a payment strategy is valid for the given chain and number of days -/
def isValidPaymentStrategy (c : Chain) (days : ℕ) (s : PaymentStrategy) : Prop :=
  c.links = days ∧
  s.cut.position > 0 ∧
  s.cut.position < c.links ∧
  s.dailyPayments.length = days ∧
  s.dailyPayments.sum = c.links

/-- The optimal payment strategy for a 7-day stay with a 7-link chain -/
def optimalStrategy : PaymentStrategy :=
  { cut := { position := 3 },
    dailyPayments := [1, 1, 1, 1, 1, 1, 1] }

/-- Theorem stating that the optimal strategy is valid for a 7-day stay with a 7-link chain -/
theorem optimal_strategy_is_valid :
  isValidPaymentStrategy { links := 7 } 7 optimalStrategy := by sorry

end optimal_strategy_is_valid_l931_93113


namespace arcsin_negative_half_l931_93190

theorem arcsin_negative_half : Real.arcsin (-1/2) = -π/6 := by sorry

end arcsin_negative_half_l931_93190


namespace sheilas_weekly_earnings_l931_93128

/-- Represents the days of the week -/
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Sheila's work hours for each day -/
def workHours (d : Day) : ℕ :=
  match d with
  | Day.Monday => 8
  | Day.Tuesday => 6
  | Day.Wednesday => 8
  | Day.Thursday => 6
  | Day.Friday => 8
  | Day.Saturday => 0
  | Day.Sunday => 0

/-- Sheila's hourly wage -/
def hourlyWage : ℚ := 14

/-- Calculates daily earnings -/
def dailyEarnings (d : Day) : ℚ :=
  hourlyWage * (workHours d)

/-- Calculates weekly earnings -/
def weeklyEarnings : ℚ :=
  (dailyEarnings Day.Monday) +
  (dailyEarnings Day.Tuesday) +
  (dailyEarnings Day.Wednesday) +
  (dailyEarnings Day.Thursday) +
  (dailyEarnings Day.Friday) +
  (dailyEarnings Day.Saturday) +
  (dailyEarnings Day.Sunday)

/-- Theorem: Sheila's weekly earnings are $504 -/
theorem sheilas_weekly_earnings : weeklyEarnings = 504 := by
  sorry

end sheilas_weekly_earnings_l931_93128


namespace units_digit_of_29_power_8_7_l931_93108

theorem units_digit_of_29_power_8_7 : 29^(8^7) % 10 = 1 := by sorry

end units_digit_of_29_power_8_7_l931_93108


namespace six_digit_increase_characterization_l931_93122

def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

def last_digit (n : ℕ) : ℕ := n % 10

def move_last_to_first (n : ℕ) : ℕ :=
  (n / 10) + (last_digit n * 100000)

def increases_by_integer_factor (n : ℕ) : Prop :=
  ∃ k : ℕ, k > 0 ∧ move_last_to_first n = k * n

def S : Set ℕ := {111111, 222222, 333333, 444444, 555555, 666666, 777777, 888888, 999999, 
                  142857, 102564, 128205, 153846, 179487, 205128, 230769}

theorem six_digit_increase_characterization :
  ∀ n : ℕ, is_six_digit n ∧ increases_by_integer_factor n ↔ n ∈ S :=
sorry

end six_digit_increase_characterization_l931_93122


namespace equation_solutions_l931_93157

theorem equation_solutions : 
  {x : ℝ | (1 / (x^2 + 13*x - 12) + 1 / (x^2 + 3*x - 12) + 1 / (x^2 - 16*x - 12) = 0)} = 
  {1, -12, 3, -4} := by
sorry

end equation_solutions_l931_93157


namespace fraction_equality_l931_93181

theorem fraction_equality (a b : ℝ) (h : a ≠ b) (h1 : a / 4 = b / 3) : b / (a - b) = 3 := by
  sorry

end fraction_equality_l931_93181


namespace expenditure_ratio_l931_93107

theorem expenditure_ratio (rajan_income balan_income rajan_expenditure balan_expenditure : ℚ) : 
  (rajan_income / balan_income = 7 / 6) →
  (rajan_income = 7000) →
  (rajan_income - rajan_expenditure = 1000) →
  (balan_income - balan_expenditure = 1000) →
  (rajan_expenditure / balan_expenditure = 6 / 5) :=
by
  sorry

end expenditure_ratio_l931_93107


namespace min_sum_squares_l931_93123

theorem min_sum_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 8) :
  ∃ (min : ℝ), min = 16/3 ∧ x^2 + y^2 + z^2 ≥ min ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀^3 + y₀^3 + z₀^3 - 3*x₀*y₀*z₀ = 8 ∧ x₀^2 + y₀^2 + z₀^2 = min :=
sorry

end min_sum_squares_l931_93123


namespace dodecahedron_edge_probability_l931_93179

/-- A regular dodecahedron -/
structure RegularDodecahedron where
  vertices : Finset (Fin 20)
  edges : Finset (Fin 20 × Fin 20)
  vertex_count : vertices.card = 20
  edge_count : edges.card = 30
  vertex_degree : ∀ v ∈ vertices, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card = 3

/-- The probability of selecting two vertices that are endpoints of an edge in a regular dodecahedron -/
def edge_endpoint_probability (d : RegularDodecahedron) : ℚ :=
  3 / 19

/-- Theorem: The probability of randomly selecting two vertices that are endpoints of an edge in a regular dodecahedron is 3/19 -/
theorem dodecahedron_edge_probability (d : RegularDodecahedron) :
  edge_endpoint_probability d = 3 / 19 := by
  sorry

end dodecahedron_edge_probability_l931_93179


namespace boys_to_girls_ratio_l931_93126

theorem boys_to_girls_ratio : 
  let num_boys : ℕ := 40
  let num_girls : ℕ := num_boys + 64
  (num_boys : ℚ) / (num_girls : ℚ) = 5 / 13 :=
by sorry

end boys_to_girls_ratio_l931_93126


namespace beef_weight_before_processing_l931_93178

theorem beef_weight_before_processing 
  (weight_after : ℝ) 
  (percent_lost : ℝ) 
  (h1 : weight_after = 546) 
  (h2 : percent_lost = 35) : 
  ∃ weight_before : ℝ, 
    weight_before * (1 - percent_lost / 100) = weight_after ∧ 
    weight_before = 840 := by
  sorry

end beef_weight_before_processing_l931_93178


namespace diamond_calculation_l931_93127

-- Define the diamond operation
def diamond (a b : ℚ) : ℚ := a + 1 / b

-- Theorem statement
theorem diamond_calculation :
  (diamond (diamond 3 4) 5) - (diamond 3 (diamond 4 5)) = 89 / 420 := by
  sorry

end diamond_calculation_l931_93127


namespace circle_area_equilateral_triangle_l931_93159

/-- The area of a circle circumscribed about an equilateral triangle with side length 12 units is 48π square units. -/
theorem circle_area_equilateral_triangle : 
  ∀ (s : ℝ) (A : ℝ),
  s = 12 →  -- Side length of the equilateral triangle
  A = π * (s / Real.sqrt 3)^2 →  -- Area formula for circumscribed circle
  A = 48 * π := by
sorry

end circle_area_equilateral_triangle_l931_93159


namespace z_value_l931_93189

theorem z_value (x y z : ℝ) (h : 1/x + 1/y = 2/z) : z = x*y/2 := by
  sorry

end z_value_l931_93189


namespace chord_equation_l931_93193

noncomputable def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 2 = 1

theorem chord_equation (m n s t : ℝ) 
  (h_positive : m > 0 ∧ n > 0 ∧ s > 0 ∧ t > 0)
  (h_sum : m + n = 2)
  (h_ratio : m / s + n / t = 9)
  (h_min : s + t = 4 / 9)
  (h_midpoint : ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    ellipse x₁ y₁ ∧ 
    ellipse x₂ y₂ ∧ 
    m = (x₁ + x₂) / 2 ∧ 
    n = (y₁ + y₂) / 2) :
  ∃ (a b c : ℝ), a * m + b * n + c = 0 ∧ a = 1 ∧ b = 2 ∧ c = -3 :=
sorry

end chord_equation_l931_93193


namespace tims_income_percentage_l931_93145

theorem tims_income_percentage (tim mary juan : ℝ) 
  (h1 : mary = 1.6 * tim) 
  (h2 : mary = 1.12 * juan) : 
  tim = 0.7 * juan := by
sorry

end tims_income_percentage_l931_93145


namespace lisa_sock_collection_l931_93137

/-- The number of sock pairs Lisa ends up with after contributions from various sources. -/
def total_socks (lisa_initial : ℕ) (sandra : ℕ) (mom_extra : ℕ) : ℕ :=
  lisa_initial + sandra + (sandra / 5) + (3 * lisa_initial + mom_extra)

/-- Theorem stating the total number of sock pairs Lisa ends up with. -/
theorem lisa_sock_collection : total_socks 12 20 8 = 80 := by
  sorry

end lisa_sock_collection_l931_93137


namespace product_of_reals_l931_93130

theorem product_of_reals (a b : ℝ) (sum_eq : a + b = 8) (cube_sum_eq : a^3 + b^3 = 170) : a * b = 21.375 := by
  sorry

end product_of_reals_l931_93130


namespace radio_price_theorem_l931_93183

def original_price : ℕ := 5000
def final_amount : ℕ := 2468

def discount (price : ℕ) : ℚ :=
  let d1 := min price 2000 * (2 / 100)
  let d2 := min (max (price - 2000) 0) 2000 * (5 / 100)
  let d3 := max (price - 4000) 0 * (10 / 100)
  d1 + d2 + d3

def sales_tax (price : ℕ) : ℚ :=
  let t1 := min price 2500 * (4 / 100)
  let t2 := min (max (price - 2500) 0) 2000 * (7 / 100)
  let t3 := max (price - 4500) 0 * (9 / 100)
  t1 + t2 + t3

theorem radio_price_theorem (reduced_price : ℕ) :
  reduced_price - discount reduced_price + sales_tax original_price = final_amount :=
by sorry

end radio_price_theorem_l931_93183


namespace ellipse_equation_from_shared_focus_l931_93135

/-- Given a parabola and an ellipse with a shared focus, prove the equation of the ellipse -/
theorem ellipse_equation_from_shared_focus (a : ℝ) (h_a : a > 0) :
  (∃ (x y : ℝ), y^2 = 8*x ∧ x^2/a^2 + y^2 = 1 ∧ x = 2) →
  (∀ (x y : ℝ), x^2/8 + y^2/4 = 1 ↔ x^2/a^2 + y^2 = 1) :=
by sorry

end ellipse_equation_from_shared_focus_l931_93135


namespace quadratic_inequality_range_l931_93163

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, m * x^2 - 2 * m * x - 3 < 0) ↔ -3 < m ∧ m ≤ 0 := by
  sorry

end quadratic_inequality_range_l931_93163


namespace geometric_progression_solution_l931_93161

theorem geometric_progression_solution (b₁ q : ℝ) : 
  b₁ * (1 + q + q^2) = 21 ∧ 
  b₁^2 * (1 + q^2 + q^4) = 189 → 
  ((b₁ = 3 ∧ q = 2) ∨ (b₁ = 12 ∧ q = 1/2)) := by
  sorry

end geometric_progression_solution_l931_93161


namespace valid_seating_arrangements_l931_93142

/-- The number of seats in a row -/
def num_seats : ℕ := 7

/-- The number of persons to be seated -/
def num_persons : ℕ := 2

/-- Function to calculate the number of seating arrangements -/
def seating_arrangements (n : ℕ) (k : ℕ) : ℕ :=
  (n.factorial) / ((n - k).factorial)

/-- Theorem stating the number of valid seating arrangements -/
theorem valid_seating_arrangements :
  seating_arrangements num_seats num_persons - 
  (num_seats - 1) * seating_arrangements 2 2 = 30 := by
  sorry

end valid_seating_arrangements_l931_93142


namespace john_profit_l931_93141

/-- Calculates John's profit from selling woodburnings, metal sculptures, and paintings. -/
theorem john_profit : 
  let woodburnings_count : ℕ := 20
  let woodburnings_price : ℚ := 15
  let metal_sculptures_count : ℕ := 15
  let metal_sculptures_price : ℚ := 25
  let paintings_count : ℕ := 10
  let paintings_price : ℚ := 40
  let wood_cost : ℚ := 100
  let metal_cost : ℚ := 150
  let paint_cost : ℚ := 120
  let woodburnings_discount : ℚ := 0.1
  let sales_tax : ℚ := 0.05

  let woodburnings_revenue := woodburnings_count * woodburnings_price * (1 - woodburnings_discount)
  let metal_sculptures_revenue := metal_sculptures_count * metal_sculptures_price
  let paintings_revenue := paintings_count * paintings_price
  let total_revenue := woodburnings_revenue + metal_sculptures_revenue + paintings_revenue
  let total_revenue_with_tax := total_revenue * (1 + sales_tax)
  let total_cost := wood_cost + metal_cost + paint_cost
  let profit := total_revenue_with_tax - total_cost

  profit = 727.25
:= by sorry

end john_profit_l931_93141


namespace unit_circle_sector_arc_length_l931_93118

theorem unit_circle_sector_arc_length (θ : Real) :
  (1/2 * θ = 1) → (θ = 2) := by
  sorry

end unit_circle_sector_arc_length_l931_93118


namespace line_segment_endpoint_l931_93156

theorem line_segment_endpoint (y : ℝ) : y > 0 →
  (Real.sqrt (((-7) - 3)^2 + (y - (-2))^2) = 13) →
  y = -2 + Real.sqrt 69 := by
sorry

end line_segment_endpoint_l931_93156


namespace solve_linear_system_l931_93121

/-- Given a system of linear equations with parameters m and n,
    prove that m + n = -2 when x = 2 and y = 1 is a solution. -/
theorem solve_linear_system (m n : ℚ) : 
  (2 * m + 1 = -3) → (2 - 2 * 1 = 2 * n) → m + n = -2 := by
  sorry

end solve_linear_system_l931_93121


namespace equal_selection_probability_l931_93160

/-- Represents the selection process for a visiting group from a larger group of students. -/
structure SelectionProcess where
  total_students : ℕ
  selected_students : ℕ
  eliminated_students : ℕ

/-- The probability of a student being selected given the selection process. -/
def selection_probability (process : SelectionProcess) : ℚ :=
  (process.selected_students : ℚ) / (process.total_students : ℚ)

/-- Theorem stating that the selection probability is equal for all students. -/
theorem equal_selection_probability (process : SelectionProcess) 
  (h1 : process.total_students = 2006)
  (h2 : process.selected_students = 50)
  (h3 : process.eliminated_students = 6) :
  ∀ (student1 student2 : Fin process.total_students),
    selection_probability process = selection_probability process :=
by
  sorry

#check equal_selection_probability

end equal_selection_probability_l931_93160


namespace root_transformation_equation_l931_93185

theorem root_transformation_equation : 
  ∀ (p q r s : ℂ),
  (p^4 + 4*p^3 - 5 = 0) → 
  (q^4 + 4*q^3 - 5 = 0) → 
  (r^4 + 4*r^3 - 5 = 0) → 
  (s^4 + 4*s^3 - 5 = 0) → 
  ∃ (x : ℂ),
  (x = (p+q+r)/s^3 ∨ x = (p+q+s)/r^3 ∨ x = (p+r+s)/q^3 ∨ x = (q+r+s)/p^3) →
  (5*x^6 - x^2 + 4*x = 0) :=
by sorry

end root_transformation_equation_l931_93185


namespace sqrt_sum_2014_l931_93116

theorem sqrt_sum_2014 (a b c : ℕ) : 
  Real.sqrt a + Real.sqrt b + Real.sqrt c = Real.sqrt 2014 →
  ((a = 0 ∧ b = 0 ∧ c = 2014) ∨
   (a = 0 ∧ b = 2014 ∧ c = 0) ∨
   (a = 2014 ∧ b = 0 ∧ c = 0)) := by
  sorry

end sqrt_sum_2014_l931_93116


namespace negation_of_existence_l931_93153

theorem negation_of_existence (x : ℝ) : 
  (¬ ∃ x₀ : ℝ, x₀ > 0 ∧ Real.log x₀ > 3 - x₀) ↔ 
  (∀ x : ℝ, x > 0 → Real.log x ≤ 3 - x) := by sorry

end negation_of_existence_l931_93153


namespace collinear_sufficient_not_necessary_for_coplanar_l931_93170

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Check if three points are collinear -/
def collinear (p q r : Point3D) : Prop := sorry

/-- Check if four points are coplanar -/
def coplanar (p q r s : Point3D) : Prop := sorry

/-- Theorem stating that collinearity of three out of four points is sufficient but not necessary for coplanarity -/
theorem collinear_sufficient_not_necessary_for_coplanar :
  (∀ p q r s : Point3D, (collinear p q r ∨ collinear p q s ∨ collinear p r s ∨ collinear q r s) → coplanar p q r s) ∧
  (∃ p q r s : Point3D, coplanar p q r s ∧ ¬collinear p q r ∧ ¬collinear p q s ∧ ¬collinear p r s ∧ ¬collinear q r s) :=
sorry

end collinear_sufficient_not_necessary_for_coplanar_l931_93170


namespace special_function_property_l931_93192

/-- A function satisfying the given property for all real numbers -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, b^2 * f a = a^2 * f b

theorem special_function_property (f : ℝ → ℝ) (h1 : special_function f) (h2 : f 2 ≠ 0) :
  (f 5 - f 1) / f 2 = 6 := by
  sorry

end special_function_property_l931_93192


namespace expression_equals_36_l931_93114

theorem expression_equals_36 : ∃ (expr : ℝ), 
  (expr = 13 * (3 - 3 / 13)) ∧ (expr = 36) :=
by sorry

end expression_equals_36_l931_93114


namespace abs_sum_inequality_l931_93164

theorem abs_sum_inequality (x y z : ℝ) :
  |x| + |y| + |z| ≤ |x+y-z| + |y+z-x| + |z+x-y| := by
  sorry

end abs_sum_inequality_l931_93164


namespace square_side_irrational_l931_93120

theorem square_side_irrational (area : ℝ) (h : area = 3) :
  ∃ (side : ℝ), side * side = area ∧ Irrational side := by
  sorry

end square_side_irrational_l931_93120


namespace ellipse_and_fixed_point_l931_93125

noncomputable section

/-- The ellipse C with given conditions -/
structure Ellipse :=
  (a b c : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : c > 0)
  (h4 : (c / b) = Real.sqrt 3 / 3)
  (h5 : b + c + 2*c = 3 + Real.sqrt 3)

/-- The equation of the ellipse is x²/4 + y²/3 = 1 -/
def ellipse_equation (C : Ellipse) : Prop :=
  C.a = 2 ∧ C.b = Real.sqrt 3

/-- The fixed point on x-axis -/
def fixed_point : ℝ × ℝ := (5/2, 0)

/-- The line QM passes through the fixed point -/
def line_passes_through_fixed_point (C : Ellipse) (P : ℝ × ℝ) : Prop :=
  let F := (C.c, 0)
  let M := (4, P.2)
  let Q := sorry -- Intersection of PF with the ellipse
  ∃ t : ℝ, fixed_point = (1 - t) • Q + t • M

/-- Main theorem -/
theorem ellipse_and_fixed_point (C : Ellipse) :
  ellipse_equation C ∧
  ∀ P, P.1^2 / 4 + P.2^2 / 3 = 1 → line_passes_through_fixed_point C P :=
sorry

end

end ellipse_and_fixed_point_l931_93125


namespace triangle_problem_l931_93111

theorem triangle_problem (a b c A B C : ℝ) : 
  0 < A ∧ A < π / 2 →
  0 < B ∧ B < π / 2 →
  0 < C ∧ C < π / 2 →
  (2 * a - c) * Real.sin A + (2 * c - a) * Real.sin C = 2 * b * Real.sin B →
  b = 1 →
  B = π / 3 ∧ 
  ∃ (p : ℝ), p = a + b + c ∧ Real.sqrt 3 + 1 < p ∧ p ≤ 3 :=
by sorry

end triangle_problem_l931_93111


namespace proportion_solution_l931_93133

theorem proportion_solution (x : ℝ) (h : (3/4) / x = 5/6) : x = 9/10 := by
  sorry

end proportion_solution_l931_93133


namespace max_money_is_twelve_dollars_l931_93154

/-- Represents the recycling scenario with given rates and collected items -/
structure RecyclingScenario where
  can_rate : Rat -- Money received for 12 cans
  newspaper_rate : Rat -- Money received for 5 kg of newspapers
  bottle_rate : Rat -- Money received for 3 glass bottles
  weight_limit : Rat -- Weight limit in kg
  cans_collected : Nat -- Number of cans collected
  can_weight : Rat -- Weight of each can in kg
  newspapers_collected : Rat -- Weight of newspapers collected in kg
  bottles_collected : Nat -- Number of bottles collected
  bottle_weight : Rat -- Weight of each bottle in kg

/-- Calculates the maximum money received from recycling -/
noncomputable def max_money_received (scenario : RecyclingScenario) : Rat :=
  sorry

/-- Theorem stating that the maximum money received is $12.00 -/
theorem max_money_is_twelve_dollars (scenario : RecyclingScenario) 
  (h1 : scenario.can_rate = 1/2)
  (h2 : scenario.newspaper_rate = 3/2)
  (h3 : scenario.bottle_rate = 9/10)
  (h4 : scenario.weight_limit = 25)
  (h5 : scenario.cans_collected = 144)
  (h6 : scenario.can_weight = 3/100)
  (h7 : scenario.newspapers_collected = 20)
  (h8 : scenario.bottles_collected = 30)
  (h9 : scenario.bottle_weight = 1/2) :
  max_money_received scenario = 12 := by
  sorry

end max_money_is_twelve_dollars_l931_93154


namespace eight_balls_three_boxes_l931_93187

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Theorem: There are 45 ways to distribute 8 indistinguishable balls into 3 distinguishable boxes -/
theorem eight_balls_three_boxes : distribute_balls 8 3 = 45 := by
  sorry

end eight_balls_three_boxes_l931_93187


namespace stock_sale_loss_l931_93112

/-- Calculates the overall loss from selling a stock with given conditions -/
def calculate_overall_loss (stock_worth : ℝ) : ℝ :=
  let profit_portion := 0.2 * stock_worth
  let loss_portion := 0.8 * stock_worth
  let profit := 0.1 * profit_portion
  let loss := 0.05 * loss_portion
  loss - profit

/-- Theorem stating the overall loss for the given stock and conditions -/
theorem stock_sale_loss (stock_worth : ℝ) (h : stock_worth = 12500) :
  calculate_overall_loss stock_worth = 250 := by
  sorry

end stock_sale_loss_l931_93112


namespace constant_in_exponent_l931_93176

theorem constant_in_exponent (w : ℕ) (h1 : 2^(2*w) = 8^(w-4)) (h2 : w = 12) : 
  ∃ k : ℕ, 2^(2*w) = 8^(w-k) ∧ k = 4 := by
sorry

end constant_in_exponent_l931_93176


namespace supercomputer_additions_in_half_hour_l931_93136

/-- Proves that a supercomputer performing 20,000 additions per second can complete 36,000,000 additions in half an hour. -/
theorem supercomputer_additions_in_half_hour :
  let additions_per_second : ℕ := 20000
  let seconds_in_half_hour : ℕ := 1800
  additions_per_second * seconds_in_half_hour = 36000000 :=
by sorry

end supercomputer_additions_in_half_hour_l931_93136


namespace prob_change_approx_point_54_l931_93134

/-- The number of banks in the country of Alpha -/
def num_banks : ℕ := 5

/-- The initial probability of a bank closing -/
def initial_prob : ℝ := 0.05

/-- The probability of a bank closing after the crisis -/
def crisis_prob : ℝ := 0.25

/-- The probability that at least one bank will close -/
def prob_at_least_one_close (p : ℝ) : ℝ := 1 - (1 - p) ^ num_banks

/-- The change in probability of at least one bank closing -/
def prob_change : ℝ :=
  |prob_at_least_one_close crisis_prob - prob_at_least_one_close initial_prob|

/-- Theorem stating that the change in probability is approximately 0.54 -/
theorem prob_change_approx_point_54 :
  ∃ ε > 0, ε < 0.005 ∧ |prob_change - 0.54| < ε :=
sorry

end prob_change_approx_point_54_l931_93134


namespace machine_times_solution_l931_93140

/-- Represents the time taken by three machines to complete a task individually and together -/
structure MachineTimes where
  first : ℝ
  second : ℝ
  third : ℝ
  together : ℝ

/-- The conditions of the problem -/
def satisfies_conditions (t : MachineTimes) : Prop :=
  t.second = t.first + 2 ∧
  t.third = 2 * t.first ∧
  t.together = 8/3

/-- The theorem statement -/
theorem machine_times_solution (t : MachineTimes) :
  satisfies_conditions t → t.first = 6 ∧ t.second = 8 ∧ t.third = 12 := by
  sorry

end machine_times_solution_l931_93140


namespace bills_height_ratio_l931_93188

/-- Represents the heights of three siblings in inches -/
structure SiblingHeights where
  cary : ℕ
  jan : ℕ
  bill : ℕ

/-- Given the heights of Cary, Jan, and Bill, proves that Bill's height is half of Cary's -/
theorem bills_height_ratio (h : SiblingHeights) 
  (h_cary : h.cary = 72)
  (h_jan : h.jan = 42)
  (h_jan_bill : h.jan = h.bill + 6) :
  h.bill / h.cary = 1 / 2 := by sorry

end bills_height_ratio_l931_93188


namespace katya_magic_pen_problem_l931_93101

theorem katya_magic_pen_problem (p_katya : ℚ) (p_pen : ℚ) (total_problems : ℕ) (min_correct : ℚ) :
  p_katya = 4/5 →
  p_pen = 1/2 →
  total_problems = 20 →
  min_correct = 13 →
  ∃ x : ℕ, x ≥ 10 ∧
    x * p_katya + (total_problems - x) * p_pen ≥ min_correct ∧
    ∀ y : ℕ, y < 10 → y * p_katya + (total_problems - y) * p_pen < min_correct :=
by sorry

end katya_magic_pen_problem_l931_93101


namespace bridge_length_proof_l931_93100

/-- Given a train that crosses a bridge and passes a lamp post, prove the length of the bridge. -/
theorem bridge_length_proof (train_length : ℝ) (bridge_crossing_time : ℝ) (lamp_post_passing_time : ℝ)
  (h1 : train_length = 400)
  (h2 : bridge_crossing_time = 45)
  (h3 : lamp_post_passing_time = 15) :
  (bridge_crossing_time * train_length / lamp_post_passing_time) - train_length = 800 :=
by sorry

end bridge_length_proof_l931_93100


namespace container_initial_percentage_l931_93182

theorem container_initial_percentage (capacity : ℝ) (added_water : ℝ) (final_fraction : ℝ) :
  capacity = 80 →
  added_water = 20 →
  final_fraction = 3/4 →
  (capacity * final_fraction - added_water) / capacity = 1/2 := by
  sorry

end container_initial_percentage_l931_93182


namespace f_negative_alpha_l931_93139

noncomputable def f (x : ℝ) : ℝ := Real.tan x + 1 / Real.tan x

theorem f_negative_alpha (α : ℝ) (h : f α = 5) : f (-α) = -5 := by
  sorry

end f_negative_alpha_l931_93139


namespace optimal_box_volume_l931_93168

/-- The volume of an open box made from a 48m x 36m sheet by cutting squares from corners -/
def box_volume (x : ℝ) : ℝ := (48 - 2*x) * (36 - 2*x) * x

/-- The derivative of the box volume function -/
def box_volume_derivative (x : ℝ) : ℝ := 1728 - 336*x + 12*x^2

theorem optimal_box_volume :
  ∃ (x : ℝ),
    x = 12 ∧
    (∀ y : ℝ, 0 < y ∧ y < 24 → box_volume y ≤ box_volume x) ∧
    box_volume x = 3456 :=
by sorry

end optimal_box_volume_l931_93168


namespace extreme_value_interval_equation_solution_range_l931_93110

noncomputable def f (x : ℝ) : ℝ := (1 + Real.log x) / x

theorem extreme_value_interval (a : ℝ) (h : a > 0) :
  (∃ x ∈ Set.Ioo a (a + 1/2), ∀ y ∈ Set.Ioo a (a + 1/2), f x ≥ f y) →
  1/2 < a ∧ a < 1 :=
sorry

theorem equation_solution_range (k : ℝ) :
  (∃ x ≥ 1, f x = k / (x + 1)) →
  k ≥ 2 :=
sorry

end extreme_value_interval_equation_solution_range_l931_93110


namespace neznaika_contradiction_l931_93174

theorem neznaika_contradiction (S T : ℝ) 
  (h1 : S ≤ 50 * T) 
  (h2 : 60 * T ≤ S) 
  (h3 : T > 0) : 
  False :=
by sorry

end neznaika_contradiction_l931_93174


namespace pool_filling_time_l931_93151

/-- Proves that filling a 24,000-gallon pool with 5 hoses supplying 3 gallons per minute takes 27 hours (rounded) -/
theorem pool_filling_time :
  let pool_capacity : ℕ := 24000
  let num_hoses : ℕ := 5
  let flow_rate_per_hose : ℕ := 3
  let minutes_per_hour : ℕ := 60
  let total_flow_rate := num_hoses * flow_rate_per_hose * minutes_per_hour
  let filling_time := (pool_capacity + total_flow_rate - 1) / total_flow_rate
  filling_time = 27 := by
  sorry

end pool_filling_time_l931_93151


namespace complex_sum_power_l931_93103

theorem complex_sum_power (i : ℂ) : i * i = -1 → (1 - i)^2016 + (1 + i)^2016 = 2^1009 := by
  sorry

end complex_sum_power_l931_93103


namespace f_bound_and_g_monotonicity_l931_93138

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x + 1

theorem f_bound_and_g_monotonicity :
  (∃ c : ℝ, c = -1 ∧ ∀ x > 0, f x ≤ 2 * x + c) ∧
  (∀ a > 0, StrictMonoOn (fun x => (f x - f a) / (x - a)) (Set.Ioo 0 a)) ∧
  (∀ a > 0, StrictMonoOn (fun x => (f x - f a) / (x - a)) (Set.Ioi a)) :=
sorry

end f_bound_and_g_monotonicity_l931_93138


namespace mary_flour_added_l931_93143

def recipe_flour : ℕ := 12
def recipe_salt : ℕ := 7
def extra_flour : ℕ := 3

theorem mary_flour_added (flour_added : ℕ) : 
  flour_added = recipe_flour - (recipe_salt + extra_flour) → flour_added = 2 := by
  sorry

end mary_flour_added_l931_93143


namespace sphere_surface_area_with_inscribed_cube_l931_93169

/-- The surface area of a sphere that circumscribes a cube with edge length 4 -/
theorem sphere_surface_area_with_inscribed_cube : 
  ∀ (cube_edge_length : ℝ) (sphere_radius : ℝ),
    cube_edge_length = 4 →
    sphere_radius = 2 * Real.sqrt 3 →
    4 * Real.pi * sphere_radius^2 = 48 * Real.pi := by
  sorry

end sphere_surface_area_with_inscribed_cube_l931_93169


namespace propositions_analysis_l931_93173

theorem propositions_analysis :
  (∃ (a b c : ℝ), a > b ∧ b > 0 ∧ a * c^2 ≤ b * c^2) ∧
  (∃ (a b : ℝ), a < b ∧ 1/a ≤ 1/b) ∧
  (∀ (a b : ℝ), a > b ∧ b > 0 → a^2 > a*b ∧ a*b > b^2) ∧
  (∀ (a b : ℝ), a > abs b → a^2 > b^2) :=
by sorry

end propositions_analysis_l931_93173


namespace ducks_at_lake_michigan_l931_93106

theorem ducks_at_lake_michigan (ducks_north_pond : ℕ) (ducks_lake_michigan : ℕ) : 
  ducks_north_pond = 2 * ducks_lake_michigan + 6 →
  ducks_north_pond = 206 →
  ducks_lake_michigan = 100 := by
sorry

end ducks_at_lake_michigan_l931_93106


namespace books_per_box_is_fifteen_l931_93102

/-- Represents the number of books in Henry's collection at different stages --/
structure BookCollection where
  initial : Nat
  room : Nat
  coffeeTable : Nat
  kitchen : Nat
  final : Nat
  pickedUp : Nat

/-- Calculates the number of books in each donation box --/
def booksPerBox (collection : BookCollection) : Nat :=
  let totalDonated := collection.initial - collection.final + collection.pickedUp
  let outsideBoxes := collection.room + collection.coffeeTable + collection.kitchen
  let inBoxes := totalDonated - outsideBoxes
  inBoxes / 3

/-- Theorem stating that the number of books in each box is 15 --/
theorem books_per_box_is_fifteen (collection : BookCollection)
  (h1 : collection.initial = 99)
  (h2 : collection.room = 21)
  (h3 : collection.coffeeTable = 4)
  (h4 : collection.kitchen = 18)
  (h5 : collection.final = 23)
  (h6 : collection.pickedUp = 12) :
  booksPerBox collection = 15 := by
  sorry

end books_per_box_is_fifteen_l931_93102


namespace inequality_condition_l931_93191

theorem inequality_condition (b : ℝ) : 
  (b > 0) → (∃ x : ℝ, |x - 2| + |x - 5| < b) ↔ b > 3 :=
by sorry

end inequality_condition_l931_93191


namespace area_ratio_of_squares_l931_93198

/-- Given three square regions A, B, and C, where the perimeter of A is 16 units
    and the perimeter of B is 32 units, prove that the ratio of the area of
    region A to the area of region C is 1/9. -/
theorem area_ratio_of_squares (a b c : ℝ) : 
  (a > 0) → (b > 0) → (c > 0) →
  (4 * a = 16) → (4 * b = 32) → (c = 3 * a) →
  (a^2) / (c^2) = 1 / 9 := by
sorry

end area_ratio_of_squares_l931_93198


namespace arithmetic_sequence_2014_l931_93150

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem arithmetic_sequence_2014 :
  arithmetic_sequence 4 3 671 = 2014 := by
  sorry

end arithmetic_sequence_2014_l931_93150


namespace sum_of_coefficients_l931_93172

theorem sum_of_coefficients (C D : ℝ) :
  (∀ x : ℝ, x ≠ 3 → C / (x - 3) + D * (x - 2) = (5 * x^2 - 8 * x - 6) / (x - 3)) →
  C + D = 20 := by
  sorry

end sum_of_coefficients_l931_93172


namespace intersection_M_N_l931_93148

def M : Set ℤ := {x | ∃ a : ℤ, x = a^2 + 1}
def N : Set ℤ := {y | 1 ≤ y ∧ y ≤ 6}

theorem intersection_M_N : M ∩ N = {1, 2, 5} := by sorry

end intersection_M_N_l931_93148


namespace range_of_a_for_two_zeros_l931_93167

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then Real.log (1 - x) else Real.sqrt x - a

-- State the theorem
theorem range_of_a_for_two_zeros (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0 ∧
   (∀ z : ℝ, f a z = 0 → z = x ∨ z = y)) →
  a ∈ Set.Ici 1 :=
sorry

end range_of_a_for_two_zeros_l931_93167


namespace system_solution_l931_93129

theorem system_solution (x y z : ℝ) : 
  (x + y + z = 1 ∧ x^3 + y^3 + z^3 = 1 ∧ x*y*z = -16) ↔ 
  ((x = 1 ∧ y = 4 ∧ z = -4) ∨
   (x = 1 ∧ y = -4 ∧ z = 4) ∨
   (x = 4 ∧ y = 1 ∧ z = -4) ∨
   (x = 4 ∧ y = -4 ∧ z = 1) ∨
   (x = -4 ∧ y = 1 ∧ z = 4) ∨
   (x = -4 ∧ y = 4 ∧ z = 1)) :=
by sorry

end system_solution_l931_93129


namespace notebooks_in_scenario3_l931_93132

/-- Represents the production scenario in a factory --/
structure ProductionScenario where
  workers : ℕ
  hours : ℕ
  tablets : ℕ
  notebooks : ℕ

/-- The production rate for tablets (time to produce one tablet) --/
def tablet_rate : ℝ := 1

/-- The production rate for notebooks (time to produce one notebook) --/
def notebook_rate : ℝ := 2

/-- The given production scenarios --/
def scenario1 : ProductionScenario := ⟨120, 1, 360, 240⟩
def scenario2 : ProductionScenario := ⟨100, 2, 400, 500⟩
def scenario3 (n : ℕ) : ProductionScenario := ⟨80, 3, 480, n⟩

/-- Theorem stating that the number of notebooks produced in scenario3 is 120 --/
theorem notebooks_in_scenario3 : ∃ n : ℕ, scenario3 n = ⟨80, 3, 480, 120⟩ := by
  sorry


end notebooks_in_scenario3_l931_93132


namespace hcf_from_lcm_and_product_l931_93109

/-- Given three positive integers with LCM 45600 and product 109183500000, their HCF is 2393750 -/
theorem hcf_from_lcm_and_product (a b c : ℕ+) 
  (h_lcm : Nat.lcm (a.val) (Nat.lcm (b.val) (c.val)) = 45600)
  (h_product : a * b * c = 109183500000) :
  Nat.gcd (a.val) (Nat.gcd (b.val) (c.val)) = 2393750 := by
  sorry

end hcf_from_lcm_and_product_l931_93109


namespace right_triangle_leg_square_l931_93149

theorem right_triangle_leg_square (a b c : ℝ) : 
  (a^2 + b^2 = c^2) →  -- right triangle condition
  (c = a + 2) →        -- hypotenuse is 2 units longer than leg a
  b^2 = 4*(a + 1) :=   -- square of other leg b
by
  sorry

end right_triangle_leg_square_l931_93149


namespace quadratic_inequality_solution_set_l931_93186

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - 2*x - 3 < 0} = Set.Ioo (-1 : ℝ) 3 := by sorry

end quadratic_inequality_solution_set_l931_93186


namespace sum_of_squares_of_roots_l931_93195

theorem sum_of_squares_of_roots (a b c d : ℝ) : 
  (a^4 - 15*a^2 + 56 = 0) ∧ 
  (b^4 - 15*b^2 + 56 = 0) ∧ 
  (c^4 - 15*c^2 + 56 = 0) ∧ 
  (d^4 - 15*d^2 + 56 = 0) ∧ 
  (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (b ≠ c) ∧ (b ≠ d) ∧ (c ≠ d) →
  a^2 + b^2 + c^2 + d^2 = 30 := by
sorry

end sum_of_squares_of_roots_l931_93195


namespace harry_book_pages_l931_93115

/-- Given that Selena's book has x pages and Harry's book has y fewer pages than half of Selena's book,
    prove that the number of pages in Harry's book is (x/2) - y. -/
theorem harry_book_pages (x y : ℕ) (selena_pages : ℕ) (harry_pages : ℕ) 
    (h1 : selena_pages = x)
    (h2 : harry_pages = selena_pages / 2 - y) :
  harry_pages = x / 2 - y := by
  sorry

end harry_book_pages_l931_93115


namespace reflection_line_sum_l931_93119

/-- Given a reflection of point (2, -2) across line y = mx + b to point (8, 4), prove m + b = 5 -/
theorem reflection_line_sum (m b : ℝ) : 
  (∃ (x y : ℝ), 
    -- The reflected point (x, y) satisfies the reflection property
    (x - 2)^2 + (y + 2)^2 = (8 - 2)^2 + (4 + 2)^2 ∧
    -- The midpoint of the original and reflected points lies on y = mx + b
    (1 : ℝ) = m * 5 + b ∧
    -- The line y = mx + b is perpendicular to the line connecting the original and reflected points
    m * ((8 - 2) / (4 + 2)) = -1) →
  m + b = 5 := by
sorry

end reflection_line_sum_l931_93119


namespace walnut_trees_remaining_l931_93146

/-- The number of walnut trees remaining in the park after cutting down damaged trees. -/
def remaining_walnut_trees (initial : ℕ) (cut_down : ℕ) : ℕ :=
  initial - cut_down

/-- Theorem stating that 29 walnut trees remain after cutting down 13 from the initial 42. -/
theorem walnut_trees_remaining : remaining_walnut_trees 42 13 = 29 := by
  sorry

end walnut_trees_remaining_l931_93146


namespace susan_missed_pay_l931_93199

/-- Calculates the missed pay for Susan's vacation --/
def missed_pay (weeks : ℕ) (work_days_per_week : ℕ) (paid_vacation_days : ℕ) 
                (hourly_rate : ℚ) (hours_per_day : ℕ) : ℚ :=
  let total_work_days := weeks * work_days_per_week
  let unpaid_days := total_work_days - paid_vacation_days
  let daily_pay := hourly_rate * hours_per_day
  unpaid_days * daily_pay

/-- Proves that Susan will miss $480 on her vacation --/
theorem susan_missed_pay : 
  missed_pay 2 5 6 15 8 = 480 := by
  sorry

end susan_missed_pay_l931_93199


namespace nils_geese_count_l931_93131

/-- Represents the number of geese Nils initially has. -/
def initial_geese : ℕ := sorry

/-- Represents the number of days the feed lasts with the initial number of geese. -/
def initial_days : ℕ := sorry

/-- Represents the amount of feed one goose consumes per day. -/
def feed_per_goose_per_day : ℝ := sorry

/-- Represents the total amount of feed available. -/
def total_feed : ℝ := sorry

/-- The feed lasts 20 days longer when 50 geese are sold. -/
axiom sell_condition : total_feed = feed_per_goose_per_day * (initial_days + 20) * (initial_geese - 50)

/-- The feed lasts 10 days less when 100 geese are bought. -/
axiom buy_condition : total_feed = feed_per_goose_per_day * (initial_days - 10) * (initial_geese + 100)

/-- The initial amount of feed equals the product of initial days, initial geese, and feed per goose per day. -/
axiom initial_condition : total_feed = feed_per_goose_per_day * initial_days * initial_geese

/-- Theorem stating that Nils initially has 300 geese. -/
theorem nils_geese_count : initial_geese = 300 := by sorry

end nils_geese_count_l931_93131


namespace line_equation_proof_l931_93197

/-- Given a line that passes through the point (-2, 5) with a slope of -3/4,
    prove that its equation is 3x + 4y - 14 = 0 -/
theorem line_equation_proof (x y : ℝ) : 
  (y - 5 = -(3/4) * (x + 2)) ↔ (3*x + 4*y - 14 = 0) := by sorry

end line_equation_proof_l931_93197


namespace states_fraction_1800_1809_l931_93147

theorem states_fraction_1800_1809 (total_states : Nat) (states_1800_1809 : Nat) :
  total_states = 30 →
  states_1800_1809 = 5 →
  (states_1800_1809 : ℚ) / total_states = 1 / 6 := by
  sorry

end states_fraction_1800_1809_l931_93147


namespace hcd_7560_270_minus_4_l931_93155

theorem hcd_7560_270_minus_4 : Nat.gcd 7560 270 - 4 = 266 := by
  sorry

end hcd_7560_270_minus_4_l931_93155


namespace triangle_with_angle_ratio_2_3_4_is_acute_l931_93124

theorem triangle_with_angle_ratio_2_3_4_is_acute (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- angles are positive
  a + b + c = 180 →        -- sum of angles in a triangle
  b = (3/2) * a →          -- ratio between second and first angle
  c = 2 * a →              -- ratio between third and first angle
  a < 90 ∧ b < 90 ∧ c < 90 -- all angles are less than 90 degrees (acute triangle)
  := by sorry

end triangle_with_angle_ratio_2_3_4_is_acute_l931_93124


namespace kelly_points_l931_93194

def golden_state_team (kelly : ℕ) : Prop :=
  let draymond := 12
  let curry := 2 * draymond
  let durant := 2 * kelly
  let klay := draymond / 2
  draymond + curry + kelly + durant + klay = 69

theorem kelly_points : ∃ (k : ℕ), golden_state_team k ∧ k = 9 := by
  sorry

end kelly_points_l931_93194


namespace smallest_prime_12_less_than_square_l931_93105

theorem smallest_prime_12_less_than_square : ∃ (n : ℕ), 
  (∀ (m : ℕ), m < n → ¬(∃ (k : ℕ), Prime (k^2 - 12) ∧ k^2 - 12 > 0)) ∧ 
  Prime (n^2 - 12) ∧ 
  n^2 - 12 > 0 := by
  sorry

end smallest_prime_12_less_than_square_l931_93105


namespace cloth_sale_loss_per_metre_l931_93104

/-- Calculates the loss per metre for a cloth sale -/
theorem cloth_sale_loss_per_metre
  (total_metres : ℕ)
  (total_selling_price : ℕ)
  (cost_price_per_metre : ℕ)
  (h1 : total_metres = 600)
  (h2 : total_selling_price = 18000)
  (h3 : cost_price_per_metre = 35) :
  (cost_price_per_metre * total_metres - total_selling_price) / total_metres = 5 := by
  sorry

#check cloth_sale_loss_per_metre

end cloth_sale_loss_per_metre_l931_93104


namespace min_side_length_square_l931_93166

theorem min_side_length_square (s : ℝ) : s ≥ 0 → s ^ 2 ≥ 900 → s ≥ 30 := by
  sorry

end min_side_length_square_l931_93166


namespace line_slope_intercept_sum_l931_93196

/-- Represents a line in the form y = mx + b -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- Given a line with slope 4 passing through (-2, 5), prove m + b = 17 -/
theorem line_slope_intercept_sum (L : Line) 
  (slope_is_4 : L.m = 4)
  (passes_through : 5 = 4 * (-2) + L.b) : 
  L.m + L.b = 17 := by
  sorry

end line_slope_intercept_sum_l931_93196


namespace inequality_range_l931_93171

theorem inequality_range (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 2, |2*x - a| > x - 1) ↔ (a < 3 ∨ a > 5) := by
  sorry

end inequality_range_l931_93171
