import Mathlib

namespace ellipse_major_axis_length_l3492_349284

/-- An ellipse with foci at (15, 30) and (15, 90) that is tangent to the y-axis has a major axis of length 30√5 -/
theorem ellipse_major_axis_length : 
  ∀ (E : Set (ℝ × ℝ)) (F₁ F₂ Y : ℝ × ℝ),
  F₁ = (15, 30) →
  F₂ = (15, 90) →
  Y.1 = 0 →
  (∀ p ∈ E, dist p F₁ + dist p F₂ = dist Y F₁ + dist Y F₂) →
  (∀ q : ℝ × ℝ, q.1 = 0 → dist q F₁ + dist q F₂ ≥ dist Y F₁ + dist Y F₂) →
  dist Y F₁ + dist Y F₂ = 30 * Real.sqrt 5 :=
by sorry

end ellipse_major_axis_length_l3492_349284


namespace compute_expression_l3492_349251

theorem compute_expression : 2 * ((3 + 7)^2 + (3^2 + 7^2)) = 316 := by
  sorry

end compute_expression_l3492_349251


namespace inequalities_for_positive_reals_l3492_349214

theorem inequalities_for_positive_reals (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (-a < -b) ∧ ((b/a + a/b) > 2) := by
  sorry

end inequalities_for_positive_reals_l3492_349214


namespace find_t_value_l3492_349206

theorem find_t_value (t : ℝ) : 
  let A : Set ℝ := {-4, t^2}
  let B : Set ℝ := {t-5, 9, 1-t}
  9 ∈ A ∩ B → t = -3 :=
by sorry

end find_t_value_l3492_349206


namespace moores_law_decade_l3492_349229

/-- Moore's Law transistor growth over a decade -/
theorem moores_law_decade (initial_transistors : ℕ) (years : ℕ) : 
  initial_transistors = 250000 →
  years = 10 →
  initial_transistors * (2 ^ (years / 2)) = 8000000 :=
by sorry

end moores_law_decade_l3492_349229


namespace framing_for_enlarged_picture_l3492_349235

/-- Calculates the minimum number of linear feet of framing needed for an enlarged and bordered picture. -/
def min_framing_feet (original_width original_height enlargement_factor border_width : ℕ) : ℕ :=
  let enlarged_width := original_width * enlargement_factor
  let enlarged_height := original_height * enlargement_factor
  let total_width := enlarged_width + 2 * border_width
  let total_height := enlarged_height + 2 * border_width
  let perimeter_inches := 2 * (total_width + total_height)
  ((perimeter_inches + 11) / 12 : ℕ)

/-- Theorem stating that for a 4x6 inch picture, quadrupled and with a 3-inch border, 9 feet of framing is needed. -/
theorem framing_for_enlarged_picture :
  min_framing_feet 4 6 4 3 = 9 := by
  sorry

end framing_for_enlarged_picture_l3492_349235


namespace quadratic_inequality_solution_set_l3492_349210

theorem quadratic_inequality_solution_set :
  {x : ℝ | (x - 2) * (x + 2) < 5} = {x : ℝ | -3 < x ∧ x < 3} := by
  sorry

end quadratic_inequality_solution_set_l3492_349210


namespace fractional_equation_solution_l3492_349270

theorem fractional_equation_solution :
  ∃ x : ℚ, (1 / (2 * x) = 2 / (x - 3)) ∧ x = -1 :=
by
  sorry

end fractional_equation_solution_l3492_349270


namespace boy_age_problem_l3492_349205

theorem boy_age_problem (total_boys : ℕ) (avg_all : ℕ) (avg_first : ℕ) (avg_last : ℕ) 
  (h_total : total_boys = 11)
  (h_avg_all : avg_all = 50)
  (h_avg_first : avg_first = 49)
  (h_avg_last : avg_last = 52) :
  (total_boys * avg_all : ℕ) = 
  (6 * avg_first : ℕ) + (6 * avg_last : ℕ) - 56 := by
  sorry

#check boy_age_problem

end boy_age_problem_l3492_349205


namespace negation_of_existence_negation_of_inequality_l3492_349248

theorem negation_of_existence (p : ℝ → Prop) : 
  (¬ ∃ x, p x) ↔ ∀ x, ¬ p x :=
by sorry

theorem negation_of_inequality : 
  (¬ ∃ x : ℝ, 2^x ≥ 2*x + 1) ↔ (∀ x : ℝ, 2^x < 2*x + 1) :=
by sorry

end negation_of_existence_negation_of_inequality_l3492_349248


namespace framed_rectangle_dimensions_l3492_349264

/-- A rectangle on a grid with a one-cell-wide frame around it. -/
structure FramedRectangle where
  length : ℕ
  width : ℕ

/-- The area of the inner rectangle. -/
def FramedRectangle.inner_area (r : FramedRectangle) : ℕ :=
  r.length * r.width

/-- The area of the frame around the rectangle. -/
def FramedRectangle.frame_area (r : FramedRectangle) : ℕ :=
  (r.length + 2) * (r.width + 2) - r.length * r.width

/-- The property that the inner area equals the frame area. -/
def FramedRectangle.area_equality (r : FramedRectangle) : Prop :=
  r.inner_area = r.frame_area

/-- The theorem stating that if the inner area equals the frame area,
    then the dimensions are either 3 × 10 or 4 × 6. -/
theorem framed_rectangle_dimensions (r : FramedRectangle) :
  r.area_equality →
  ((r.length = 3 ∧ r.width = 10) ∨ (r.length = 4 ∧ r.width = 6) ∨
   (r.length = 10 ∧ r.width = 3) ∨ (r.length = 6 ∧ r.width = 4)) :=
by sorry

end framed_rectangle_dimensions_l3492_349264


namespace rectangle_area_is_220_l3492_349234

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of rectangle PQRS -/
structure Rectangle where
  P : Point
  Q : Point
  R : Point
  S : Point

/-- Calculate the area of a rectangle given three of its vertices -/
def rectangleArea (rect : Rectangle) : ℝ :=
  let width := abs (rect.Q.x - rect.P.x)
  let height := abs (rect.Q.y - rect.R.y)
  width * height

/-- Theorem: The area of rectangle PQRS with given vertices is 220 -/
theorem rectangle_area_is_220 : ∃ (S : Point),
  let rect : Rectangle := {
    P := { x := 15, y := 55 },
    Q := { x := 26, y := 55 },
    R := { x := 26, y := 35 },
    S := S
  }
  rectangleArea rect = 220 := by
  sorry

end rectangle_area_is_220_l3492_349234


namespace expression_simplification_l3492_349260

theorem expression_simplification (x : ℝ) :
  2*x*(4*x^2 - 3*x + 1) - 7*(2*x^2 - 3*x + 4) = 8*x^3 - 20*x^2 + 23*x - 28 := by
  sorry

end expression_simplification_l3492_349260


namespace dot_product_of_parallel_vectors_l3492_349294

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 = k * w.1 ∧ v.2 = k * w.2

theorem dot_product_of_parallel_vectors :
  let p : ℝ × ℝ := (1, -2)
  let q : ℝ × ℝ := (x, 4)
  ∀ x : ℝ, parallel p q → p.1 * q.1 + p.2 * q.2 = -10 := by
  sorry

end dot_product_of_parallel_vectors_l3492_349294


namespace polynomial_evaluation_and_coefficient_sum_l3492_349249

theorem polynomial_evaluation_and_coefficient_sum (d : ℝ) (h : d ≠ 0) :
  (10*d + 16 + 17*d^2 + 3*d^3) + (5*d + 4 + 2*d^2 + 2*d^3) = 5*d^3 + 19*d^2 + 15*d + 20 ∧
  5 + 19 + 15 + 20 = 59 := by
  sorry

end polynomial_evaluation_and_coefficient_sum_l3492_349249


namespace grey_perimeter_fraction_five_strips_l3492_349263

/-- A square divided into strips -/
structure StrippedSquare where
  num_strips : ℕ
  num_grey_strips : ℕ
  h_grey_strips : num_grey_strips ≤ num_strips

/-- The fraction of the perimeter that is grey -/
def grey_perimeter_fraction (s : StrippedSquare) : ℚ :=
  s.num_grey_strips / s.num_strips

/-- Theorem: In a square divided into 5 strips with 2 grey strips, 
    the fraction of the perimeter that is grey is 2/5 -/
theorem grey_perimeter_fraction_five_strips 
  (s : StrippedSquare) 
  (h_five_strips : s.num_strips = 5)
  (h_two_grey : s.num_grey_strips = 2) : 
  grey_perimeter_fraction s = 2 / 5 := by
  sorry

end grey_perimeter_fraction_five_strips_l3492_349263


namespace coin_difference_l3492_349204

def coin_values : List Nat := [5, 10, 25, 50]

def total_amount : Nat := 60

def min_coins (values : List Nat) (amount : Nat) : Nat :=
  sorry

def max_coins (values : List Nat) (amount : Nat) : Nat :=
  sorry

theorem coin_difference :
  max_coins coin_values total_amount - min_coins coin_values total_amount = 10 := by
  sorry

end coin_difference_l3492_349204


namespace farmer_remaining_apples_l3492_349250

def initial_apples : ℕ := 127
def apples_given_away : ℕ := 88

theorem farmer_remaining_apples : initial_apples - apples_given_away = 39 := by
  sorry

end farmer_remaining_apples_l3492_349250


namespace no_valid_solution_l3492_349242

theorem no_valid_solution : ¬∃ (Y : ℕ), Y > 0 ∧ 2*Y + Y + 3*Y = 14 := by
  sorry

end no_valid_solution_l3492_349242


namespace system_one_solution_system_two_solution_l3492_349296

-- System 1
theorem system_one_solution (x z : ℚ) : 
  (3 * x - 5 * z = 6 ∧ x + 4 * z = -15) ↔ (x = -3 ∧ z = -3) := by sorry

-- System 2
theorem system_two_solution (x y : ℚ) :
  ((2 * x - 1) / 5 + (3 * y - 2) / 4 = 2 ∧ 
   (3 * x + 1) / 5 - (3 * y + 2) / 4 = 0) ↔ (x = 3 ∧ y = 2) := by sorry

end system_one_solution_system_two_solution_l3492_349296


namespace tournament_games_count_l3492_349218

/-- A single-elimination tournament structure -/
structure Tournament :=
  (total_teams : ℕ)
  (bye_teams : ℕ)
  (h_bye : bye_teams ≤ total_teams)

/-- The number of games played in a single-elimination tournament -/
def games_played (t : Tournament) : ℕ :=
  t.total_teams - 1

theorem tournament_games_count (t : Tournament) 
  (h_total : t.total_teams = 32) 
  (h_bye : t.bye_teams = 8) : 
  games_played t = 32 := by
sorry

end tournament_games_count_l3492_349218


namespace triangle_count_proof_l3492_349291

/-- The number of triangles formed by 9 distinct lines in a plane -/
def num_triangles : ℕ := 23

/-- The total number of ways to choose 3 lines from 9 lines -/
def total_combinations : ℕ := Nat.choose 9 3

/-- The number of intersections where exactly three lines meet -/
def num_intersections : ℕ := 61

theorem triangle_count_proof :
  num_triangles = total_combinations - num_intersections :=
by sorry

end triangle_count_proof_l3492_349291


namespace percentage_with_neither_condition_l3492_349274

/-- Given a survey of teachers, calculate the percentage who have neither high blood pressure nor heart trouble. -/
theorem percentage_with_neither_condition
  (total : ℕ)
  (high_blood_pressure : ℕ)
  (heart_trouble : ℕ)
  (both : ℕ)
  (h1 : total = 150)
  (h2 : high_blood_pressure = 80)
  (h3 : heart_trouble = 60)
  (h4 : both = 30)
  : (total - (high_blood_pressure + heart_trouble - both)) / total * 100 = 800 / 30 := by
  sorry

end percentage_with_neither_condition_l3492_349274


namespace reflected_ray_equation_l3492_349278

/-- The equation of a line passing through two points (x₁, y₁) and (x₂, y₂) -/
def line_equation (x₁ y₁ x₂ y₂ : ℝ) (x y : ℝ) : Prop :=
  (y - y₁) * (x₂ - x₁) = (x - x₁) * (y₂ - y₁)

/-- A point (x, y) is a reflection of (x₀, y₀) across the x-axis if x = x₀ and y = -y₀ -/
def is_reflection_x_axis (x y x₀ y₀ : ℝ) : Prop :=
  x = x₀ ∧ y = -y₀

theorem reflected_ray_equation :
  is_reflection_x_axis 2 (-1) 2 1 →
  line_equation 2 (-1) 4 5 x y ↔ 3 * x - y - 7 = 0 :=
by sorry

end reflected_ray_equation_l3492_349278


namespace range_of_f_leq_3_l3492_349226

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 8 then x^(1/3) else 2 * Real.exp (x - 8)

-- Theorem statement
theorem range_of_f_leq_3 :
  {x : ℝ | f x ≤ 3} = {x : ℝ | x ≤ 27} := by sorry

end range_of_f_leq_3_l3492_349226


namespace intersection_dot_product_converse_not_always_true_l3492_349231

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 2*x

-- Define a line passing through (3,0)
def line_through_3_0 (l : ℝ → ℝ) : Prop := l 3 = 0

-- Define intersection points of a line and the parabola
def intersection_points (l : ℝ → ℝ) (A B : ℝ × ℝ) : Prop :=
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧ l A.1 = A.2 ∧ l B.1 = B.2

-- Define dot product of OA and OB
def dot_product (A B : ℝ × ℝ) : ℝ := A.1 * B.1 + A.2 * B.2

-- Theorem 1
theorem intersection_dot_product (l : ℝ → ℝ) (A B : ℝ × ℝ) :
  line_through_3_0 l → intersection_points l A B → dot_product A B = 3 :=
sorry

-- Theorem 2
theorem converse_not_always_true : 
  ∃ (A B : ℝ × ℝ) (l : ℝ → ℝ), parabola A.1 A.2 ∧ parabola B.1 B.2 ∧ 
  dot_product A B = 3 ∧ ¬(line_through_3_0 l) ∧ l A.1 = A.2 ∧ l B.1 = B.2 :=
sorry

end intersection_dot_product_converse_not_always_true_l3492_349231


namespace gravitational_force_at_distance_l3492_349258

/-- Gravitational force calculation -/
theorem gravitational_force_at_distance 
  (k : ℝ) -- Gravitational constant
  (d₁ d₂ : ℝ) -- Distances from Earth's center
  (f₁ : ℝ) -- Force at distance d₁
  (h₁ : d₁ > 0)
  (h₂ : d₂ > 0)
  (h₃ : f₁ > 0)
  (h₄ : k = f₁ * d₁^2) -- Force-distance relation at d₁
  (h₅ : d₁ = 4000) -- Distance to Earth's surface in miles
  (h₆ : f₁ = 500) -- Force at Earth's surface in Newtons
  (h₇ : d₂ = 40000) -- Distance to space station in miles
  : f₁ * (d₂ / d₁)^2 = 5 := by
  sorry

#check gravitational_force_at_distance

end gravitational_force_at_distance_l3492_349258


namespace unique_integer_divisibility_l3492_349212

theorem unique_integer_divisibility (n : ℕ) : 
  n > 1 → (∃ k : ℕ, (2^n + 1) = k * n^2) ↔ n = 3 := by
  sorry

end unique_integer_divisibility_l3492_349212


namespace m_divided_by_8_l3492_349207

theorem m_divided_by_8 (m : ℕ) (h : m = 16^1024) : m / 8 = 2^4093 := by
  sorry

end m_divided_by_8_l3492_349207


namespace a_equals_one_sufficient_not_necessary_for_abs_a_equals_one_l3492_349243

theorem a_equals_one_sufficient_not_necessary_for_abs_a_equals_one :
  (∀ a : ℝ, a = 1 → |a| = 1) ∧
  (∃ a : ℝ, a ≠ 1 ∧ |a| = 1) :=
by sorry

end a_equals_one_sufficient_not_necessary_for_abs_a_equals_one_l3492_349243


namespace exists_20_digit_singular_l3492_349298

/-- A number is singular if it's a 2n-digit perfect square, and both its first n digits
    and last n digits are also perfect squares. --/
def is_singular (x : ℕ) : Prop :=
  ∃ (n : ℕ), 
    (x ≥ 10^(2*n - 1)) ∧ 
    (x < 10^(2*n)) ∧
    (∃ (y : ℕ), x = y^2) ∧
    (∃ (a b : ℕ), 
      x = a * 10^n + b ∧
      (∃ (c : ℕ), a = c^2) ∧
      (∃ (d : ℕ), b = d^2) ∧
      (a ≥ 10^(n-1)) ∧
      (b > 0))

/-- There exists a 20-digit singular number. --/
theorem exists_20_digit_singular : ∃ (x : ℕ), is_singular x ∧ (x ≥ 10^19) ∧ (x < 10^20) :=
sorry

end exists_20_digit_singular_l3492_349298


namespace widget_selling_price_l3492_349288

-- Define the problem parameters
def widget_cost : ℝ := 3
def monthly_rent : ℝ := 10000
def tax_rate : ℝ := 0.20
def worker_salary : ℝ := 2500
def num_workers : ℕ := 4
def widgets_sold : ℕ := 5000
def total_profit : ℝ := 4000

-- Define the theorem
theorem widget_selling_price :
  let worker_expenses : ℝ := worker_salary * num_workers
  let total_expenses : ℝ := monthly_rent + worker_expenses
  let widget_expenses : ℝ := widget_cost * widgets_sold
  let taxes : ℝ := tax_rate * total_profit
  let total_expenses_with_taxes : ℝ := total_expenses + widget_expenses + taxes
  let total_revenue : ℝ := total_expenses_with_taxes + total_profit
  let selling_price : ℝ := total_revenue / widgets_sold
  selling_price = 7.96 := by
  sorry

end widget_selling_price_l3492_349288


namespace quadratic_root_implies_m_l3492_349232

theorem quadratic_root_implies_m (m : ℚ) : 
  ((-2 : ℚ)^2 - m*(-2) - 3 = 0) → m = -1/2 := by
  sorry

end quadratic_root_implies_m_l3492_349232


namespace inequality_proof_l3492_349259

theorem inequality_proof (x₁ x₂ y₁ y₂ z₁ z₂ : ℝ) 
  (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) 
  (hy₁ : x₁ * y₁ > z₁^2) (hy₂ : x₂ * y₂ > z₂^2) : 
  8 / ((x₁ + x₂) * (y₁ + y₂) - (z₁ + z₂)^2) ≤ 
  1 / (x₁ * y₁ - z₁^2) + 1 / (x₂ * y₂ - z₂^2) := by
  sorry

end inequality_proof_l3492_349259


namespace airplane_shot_down_probability_l3492_349273

def probability_airplane_shot_down : ℝ :=
  let p_A : ℝ := 0.4
  let p_B : ℝ := 0.5
  let p_C : ℝ := 0.8
  let p_one_hit : ℝ := 0.4
  let p_two_hit : ℝ := 0.7
  let p_three_hit : ℝ := 1

  let p_A_miss : ℝ := 1 - p_A
  let p_B_miss : ℝ := 1 - p_B
  let p_C_miss : ℝ := 1 - p_C

  let p_one_person_hits : ℝ := 
    (p_A * p_B_miss * p_C_miss + p_A_miss * p_B * p_C_miss + p_A_miss * p_B_miss * p_C) * p_one_hit

  let p_two_people_hit : ℝ := 
    (p_A * p_B * p_C_miss + p_A * p_B_miss * p_C + p_A_miss * p_B * p_C) * p_two_hit

  let p_all_hit : ℝ := p_A * p_B * p_C * p_three_hit

  p_one_person_hits + p_two_people_hit + p_all_hit

theorem airplane_shot_down_probability : 
  probability_airplane_shot_down = 0.604 := by sorry

end airplane_shot_down_probability_l3492_349273


namespace smartphone_price_l3492_349216

/-- The original sticker price of the smartphone -/
def sticker_price : ℝ := 950

/-- The price at store X after discount and rebate -/
def price_x (p : ℝ) : ℝ := 0.8 * p - 120

/-- The price at store Y after discount -/
def price_y (p : ℝ) : ℝ := 0.7 * p

/-- Theorem stating that the sticker price satisfies the given conditions -/
theorem smartphone_price :
  price_x sticker_price + 25 = price_y sticker_price :=
by sorry

end smartphone_price_l3492_349216


namespace fractional_equation_solution_l3492_349293

theorem fractional_equation_solution (k : ℝ) : 
  (k / 2 + (2 - 3) / (2 - 1) = 1) → k = 4 := by
  sorry

end fractional_equation_solution_l3492_349293


namespace rope_length_problem_l3492_349269

theorem rope_length_problem (L : ℝ) : 
  (L / 3 + 0.3 * (2 * L / 3)) - (L - (L / 3 + 0.3 * (2 * L / 3))) = 0.4 → L = 6 := by
  sorry

end rope_length_problem_l3492_349269


namespace find_x_l3492_349215

theorem find_x (x y z : ℝ) 
  (h1 : x * y / (x + y) = 4)
  (h2 : x * z / (x + z) = 5)
  (h3 : y * z / (y + z) = 6)
  : x = 40 / 9 := by
  sorry

end find_x_l3492_349215


namespace no_positive_solution_l3492_349230

theorem no_positive_solution :
  ¬ ∃ (x : ℝ), x > 0 ∧ (Real.log x / Real.log 4) * (Real.log 9 / Real.log x) = 2 * Real.log 9 / Real.log 4 :=
by sorry

end no_positive_solution_l3492_349230


namespace missing_number_in_mean_l3492_349220

theorem missing_number_in_mean (numbers : List ℕ) (missing : ℕ) : 
  numbers = [1, 22, 23, 24, 25, 26, 27] →
  numbers.length = 7 →
  (numbers.sum + missing) / 8 = 20 →
  missing = 12 := by
sorry

end missing_number_in_mean_l3492_349220


namespace skill_testing_question_l3492_349280

theorem skill_testing_question : 5 * (10 - 6) / 2 = 10 := by
  sorry

end skill_testing_question_l3492_349280


namespace jason_picked_ten_plums_l3492_349253

def alyssa_plums : ℕ := 17
def total_plums : ℕ := 27

def jason_plums : ℕ := total_plums - alyssa_plums

theorem jason_picked_ten_plums : jason_plums = 10 := by
  sorry

end jason_picked_ten_plums_l3492_349253


namespace hispanic_west_percentage_l3492_349227

def hispanic_ne : ℕ := 10
def hispanic_mw : ℕ := 8
def hispanic_south : ℕ := 22
def hispanic_west : ℕ := 15

def total_hispanic : ℕ := hispanic_ne + hispanic_mw + hispanic_south + hispanic_west

def percent_in_west : ℚ := hispanic_west / total_hispanic * 100

theorem hispanic_west_percentage :
  round percent_in_west = 27 :=
sorry

end hispanic_west_percentage_l3492_349227


namespace min_value_of_f_l3492_349257

-- Define the function f
def f (x m : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + m

-- State the theorem
theorem min_value_of_f (m : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f y m ≤ f x m) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x m = 3) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f x m ≤ f y m) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x m = -37) :=
by
  sorry

end min_value_of_f_l3492_349257


namespace rectangle_diagonal_l3492_349224

/-- Given a rectangle with side OA and diagonal OB, prove the value of k. -/
theorem rectangle_diagonal (OA OB : ℝ × ℝ) (k : ℝ) : 
  OA = (-3, 1) → 
  OB = (-2, k) → 
  (OA.1 * (OB.1 - OA.1) + OA.2 * (OB.2 - OA.2) = 0) → 
  k = 4 := by
  sorry

#check rectangle_diagonal

end rectangle_diagonal_l3492_349224


namespace appliance_cost_after_discount_l3492_349255

/-- Calculates the total cost of a washing machine and dryer after applying a discount -/
theorem appliance_cost_after_discount
  (washing_machine_cost : ℝ)
  (dryer_cost_difference : ℝ)
  (discount_percentage : ℝ)
  (h1 : washing_machine_cost = 100)
  (h2 : dryer_cost_difference = 30)
  (h3 : discount_percentage = 0.1) :
  let dryer_cost := washing_machine_cost - dryer_cost_difference
  let total_cost := washing_machine_cost + dryer_cost
  let discount_amount := discount_percentage * total_cost
  washing_machine_cost + dryer_cost - discount_amount = 153 := by
sorry

end appliance_cost_after_discount_l3492_349255


namespace abs_frac_gt_three_iff_x_in_intervals_l3492_349252

theorem abs_frac_gt_three_iff_x_in_intervals (x : ℝ) :
  x ≠ 2 →
  (|(3 * x - 2) / (x - 2)| > 3) ↔ (x > 4/3 ∧ x < 2) ∨ x > 2 :=
by sorry

end abs_frac_gt_three_iff_x_in_intervals_l3492_349252


namespace average_difference_theorem_l3492_349208

/-- A school with students and teachers -/
structure School where
  num_students : ℕ
  num_teachers : ℕ
  class_sizes : List ℕ

/-- Calculate the average class size from a teacher's perspective -/
def teacher_average (school : School) : ℚ :=
  (school.class_sizes.sum : ℚ) / school.num_teachers

/-- Calculate the average class size from a student's perspective -/
def student_average (school : School) : ℚ :=
  (school.class_sizes.map (λ size => size * size)).sum / school.num_students

/-- The main theorem to prove -/
theorem average_difference_theorem (school : School) 
    (h1 : school.num_students = 120)
    (h2 : school.num_teachers = 5)
    (h3 : school.class_sizes = [60, 30, 20, 5, 5])
    (h4 : school.class_sizes.sum = school.num_students) : 
    teacher_average school - student_average school = -17.25 := by
  sorry

end average_difference_theorem_l3492_349208


namespace fair_distribution_result_l3492_349282

/-- Represents the fair distribution of talers in the bread-sharing scenario -/
def fair_distribution (loaves1 loaves2 : ℕ) (total_talers : ℕ) : ℕ × ℕ :=
  let total_loaves := loaves1 + loaves2
  let loaves_per_person := total_loaves / 3
  let talers_per_loaf := total_talers / loaves_per_person
  let remaining_loaves1 := loaves1 - loaves_per_person
  let remaining_loaves2 := loaves2 - loaves_per_person
  let talers1 := remaining_loaves1 * talers_per_loaf
  let talers2 := remaining_loaves2 * talers_per_loaf
  (talers1, talers2)

/-- The fair distribution of talers in the given scenario is (1, 7) -/
theorem fair_distribution_result :
  fair_distribution 3 5 8 = (1, 7) := by
  sorry

end fair_distribution_result_l3492_349282


namespace elevator_problem_l3492_349239

def elevator_ways (n : ℕ) (k : ℕ) (max_per_floor : ℕ) : ℕ :=
  sorry

theorem elevator_problem : elevator_ways 3 5 2 = 120 := by
  sorry

end elevator_problem_l3492_349239


namespace football_game_cost_l3492_349222

def total_spent : ℚ := 35.52
def strategy_game_cost : ℚ := 9.46
def batman_game_cost : ℚ := 12.04

theorem football_game_cost :
  total_spent - strategy_game_cost - batman_game_cost = 13.02 := by
  sorry

end football_game_cost_l3492_349222


namespace tim_has_203_balloons_l3492_349247

/-- The number of violet balloons Dan has -/
def dan_balloons : ℕ := 29

/-- The factor by which Tim's balloons exceed Dan's -/
def tim_factor : ℕ := 7

/-- The number of violet balloons Tim has -/
def tim_balloons : ℕ := dan_balloons * tim_factor

/-- Theorem: Tim has 203 violet balloons -/
theorem tim_has_203_balloons : tim_balloons = 203 := by
  sorry

end tim_has_203_balloons_l3492_349247


namespace second_car_traveled_5km_l3492_349285

/-- Represents the distance traveled by the second car -/
def second_car_distance : ℝ := 5

/-- The initial distance between the two cars -/
def initial_distance : ℝ := 105

/-- The distance traveled by the first car before turning back -/
def first_car_distance : ℝ := 25 + 15 + 25

/-- The final distance between the two cars -/
def final_distance : ℝ := 20

/-- Theorem stating that the second car traveled 5 km -/
theorem second_car_traveled_5km :
  initial_distance - (first_car_distance + 15 + second_car_distance) = final_distance :=
by sorry

end second_car_traveled_5km_l3492_349285


namespace range_of_m_l3492_349225

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 - 4*x + 3
def g (m x : ℝ) : ℝ := m*x + 3 - 2*m

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, 
  (∀ x₁ ∈ Set.Icc 0 4, ∃ x₂ ∈ Set.Icc 0 4, f x₁ = g m x₂) ↔ 
  m ∈ Set.Iic (-2) ∪ Set.Ici 2 :=
sorry

end range_of_m_l3492_349225


namespace angle_measure_l3492_349211

theorem angle_measure (x : ℝ) : 
  (180 - x = 4 * (90 - x)) → x = 60 := by
  sorry

end angle_measure_l3492_349211


namespace probability_second_odd_given_first_odd_l3492_349245

theorem probability_second_odd_given_first_odd (n : ℕ) (odds evens : ℕ) 
  (h1 : n = odds + evens)
  (h2 : n = 9)
  (h3 : odds = 5)
  (h4 : evens = 4) :
  (odds - 1) / (n - 1) = 1 / 2 :=
sorry

end probability_second_odd_given_first_odd_l3492_349245


namespace jordan_rectangle_width_l3492_349217

theorem jordan_rectangle_width
  (carol_length : ℝ)
  (carol_width : ℝ)
  (jordan_length : ℝ)
  (jordan_width : ℝ)
  (h1 : carol_length = 5)
  (h2 : carol_width = 24)
  (h3 : jordan_length = 8)
  (h4 : carol_length * carol_width = jordan_length * jordan_width) :
  jordan_width = 15 := by
sorry

end jordan_rectangle_width_l3492_349217


namespace sequence_growth_l3492_349200

theorem sequence_growth (k : ℕ) (h : k ≥ 2) : 
  (2^(k+1) - 1) - (2^k - 1) = 2^k := by
  sorry

end sequence_growth_l3492_349200


namespace x_plus_y_values_l3492_349233

theorem x_plus_y_values (x y : ℝ) (h1 : -x = 3) (h2 : |y| = 5) :
  x + y = -8 ∨ x + y = 2 := by
  sorry

end x_plus_y_values_l3492_349233


namespace train_length_l3492_349297

theorem train_length (platform1_length platform2_length : ℝ)
                     (time1 time2 : ℝ)
                     (h1 : platform1_length = 110)
                     (h2 : platform2_length = 250)
                     (h3 : time1 = 15)
                     (h4 : time2 = 20)
                     (h5 : time1 > 0)
                     (h6 : time2 > 0) :
  let train_length := (platform2_length * time1 - platform1_length * time2) / (time2 - time1)
  train_length = 310 := by
sorry

end train_length_l3492_349297


namespace expected_value_specialized_coin_l3492_349287

/-- A specialized coin with given probabilities and payoffs -/
structure Coin where
  prob_heads : ℚ
  prob_tails : ℚ
  payoff_heads : ℚ
  payoff_tails : ℚ

/-- The expected value of a single flip of the coin -/
def expected_value (c : Coin) : ℚ :=
  c.prob_heads * c.payoff_heads + c.prob_tails * c.payoff_tails

/-- The expected value of two flips of the coin -/
def expected_value_two_flips (c : Coin) : ℚ :=
  2 * expected_value c

theorem expected_value_specialized_coin :
  let c : Coin := {
    prob_heads := 1/4,
    prob_tails := 3/4,
    payoff_heads := 4,
    payoff_tails := -3
  }
  expected_value_two_flips c = -5/2 := by
  sorry

end expected_value_specialized_coin_l3492_349287


namespace gabrielle_robins_count_l3492_349283

/-- The number of birds Gabrielle saw -/
def gabrielle_total : ℕ := sorry

/-- The number of robins Gabrielle saw -/
def gabrielle_robins : ℕ := sorry

/-- The number of cardinals Gabrielle saw -/
def gabrielle_cardinals : ℕ := 4

/-- The number of blue jays Gabrielle saw -/
def gabrielle_blue_jays : ℕ := 3

/-- The number of birds Chase saw -/
def chase_total : ℕ := 10

/-- The number of robins Chase saw -/
def chase_robins : ℕ := 2

/-- The number of blue jays Chase saw -/
def chase_blue_jays : ℕ := 3

/-- The number of cardinals Chase saw -/
def chase_cardinals : ℕ := 5

theorem gabrielle_robins_count :
  gabrielle_total = chase_total + chase_total / 5 ∧
  gabrielle_total = gabrielle_robins + gabrielle_cardinals + gabrielle_blue_jays ∧
  gabrielle_robins = 5 := by sorry

end gabrielle_robins_count_l3492_349283


namespace function_root_implies_a_range_l3492_349295

theorem function_root_implies_a_range (a : ℝ) :
  (∃ x : ℝ, -1 < x ∧ x < 1 ∧ a * x + 1 = 0) →
  (a < -1 ∨ a > 1) :=
by sorry

end function_root_implies_a_range_l3492_349295


namespace correct_mean_calculation_l3492_349219

theorem correct_mean_calculation (n : ℕ) (initial_mean : ℚ) (incorrect_value correct_value : ℚ) :
  n = 30 ∧ initial_mean = 180 ∧ incorrect_value = 135 ∧ correct_value = 155 →
  let total_sum := n * initial_mean
  let corrected_sum := total_sum + (correct_value - incorrect_value)
  corrected_sum / n = 180.67 := by
sorry

end correct_mean_calculation_l3492_349219


namespace unique_c_value_l3492_349237

theorem unique_c_value : ∃! c : ℝ, ∀ x : ℝ, x * (3 * x + 1) - c > 0 ↔ x > -5/3 ∧ x < 3 := by
  sorry

end unique_c_value_l3492_349237


namespace problem_solution_l3492_349261

theorem problem_solution (x : ℝ) (h : 1 - 6/x + 9/x^2 = 0) : 2/x = 2/3 := by
  sorry

end problem_solution_l3492_349261


namespace hikmet_seventh_l3492_349266

/-- Represents the position of a racer in a 12-person race -/
def Position := Fin 12

/-- The race results -/
structure RaceResult where
  david : Position
  hikmet : Position
  jack : Position
  marta : Position
  rand : Position
  todd : Position

/-- Conditions of the race -/
def race_conditions (result : RaceResult) : Prop :=
  result.marta.val = result.jack.val + 3 ∧
  result.jack.val = result.todd.val + 1 ∧
  result.todd.val = result.rand.val + 3 ∧
  result.rand.val + 5 = result.hikmet.val ∧
  result.hikmet.val + 4 = result.david.val ∧
  result.marta.val = 9

/-- Theorem stating that Hikmet finished in 7th place -/
theorem hikmet_seventh (result : RaceResult) 
  (h : race_conditions result) : result.hikmet.val = 7 := by
  sorry

end hikmet_seventh_l3492_349266


namespace not_p_or_q_l3492_349279

-- Define proposition p
def p : Prop := ∀ x : ℝ, x^2 + x - 1 > 0

-- Define proposition q
def q : Prop := ∃ x : ℝ, (2 : ℝ)^x > (3 : ℝ)^x

-- Theorem to prove
theorem not_p_or_q : (¬p) ∨ q := by sorry

end not_p_or_q_l3492_349279


namespace intersection_of_P_and_Q_l3492_349221

def P : Set ℝ := {x | 2 ≤ x ∧ x ≤ 7}
def Q : Set ℝ := {x | x^2 - x - 6 = 0}

theorem intersection_of_P_and_Q : P ∩ Q = {3} := by sorry

end intersection_of_P_and_Q_l3492_349221


namespace f_sum_zero_l3492_349202

-- Define the function f(x) = ax^2 + bx
def f (a b x : ℝ) : ℝ := a * x^2 + b * x

-- State the theorem
theorem f_sum_zero (a b x₁ x₂ : ℝ) 
  (h₁ : a * b ≠ 0) 
  (h₂ : f a b x₁ = f a b x₂) 
  (h₃ : x₁ ≠ x₂) : 
  f a b (x₁ + x₂) = 0 := by
sorry

end f_sum_zero_l3492_349202


namespace angle_EFG_is_60_degrees_l3492_349209

-- Define the angles as real numbers
variable (x : ℝ)
variable (angle_CFG angle_CEB angle_BEA angle_EFG : ℝ)

-- Define the parallel lines property
variable (AD_parallel_FG : Prop)

-- State the theorem
theorem angle_EFG_is_60_degrees 
  (h1 : AD_parallel_FG)
  (h2 : angle_CFG = 1.5 * x)
  (h3 : angle_CEB = x)
  (h4 : angle_BEA = 2 * x)
  (h5 : angle_EFG = angle_CFG) :
  angle_EFG = 60 := by
  sorry

end angle_EFG_is_60_degrees_l3492_349209


namespace parabola_closest_point_l3492_349262

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 2*y

-- Define the distance between two points
def distance_squared (x1 y1 x2 y2 : ℝ) : ℝ :=
  (x1 - x2)^2 + (y1 - y2)^2

-- Theorem statement
theorem parabola_closest_point (a : ℝ) :
  (∀ x y : ℝ, parabola x y →
    ∃ xv yv : ℝ, parabola xv yv ∧
      ∀ x' y' : ℝ, parabola x' y' →
        distance_squared xv yv 0 a ≤ distance_squared x' y' 0 a) →
  a ≤ 1 :=
sorry

end parabola_closest_point_l3492_349262


namespace motel_monthly_charge_l3492_349246

theorem motel_monthly_charge 
  (weeks_per_month : ℕ)
  (num_months : ℕ)
  (weekly_rate : ℕ)
  (total_savings : ℕ)
  (h1 : weeks_per_month = 4)
  (h2 : num_months = 3)
  (h3 : weekly_rate = 280)
  (h4 : total_savings = 360) :
  (num_months * weeks_per_month * weekly_rate - total_savings) / num_months = 1000 := by
  sorry

end motel_monthly_charge_l3492_349246


namespace m_range_l3492_349244

def f (x : ℝ) := x^3 + x

theorem m_range (m : ℝ) :
  (∀ θ : ℝ, 0 < θ ∧ θ < π/2 → f (m * Real.sin θ) + f (1 - m) > 0) →
  m ≤ 1 :=
by sorry

end m_range_l3492_349244


namespace grassy_plot_width_l3492_349268

/-- Given a rectangular grassy plot with a gravel path, calculate its width -/
theorem grassy_plot_width : ℝ :=
  let plot_length : ℝ := 110
  let path_width : ℝ := 2.5
  let gravelling_cost_per_sq_meter : ℝ := 0.80
  let total_gravelling_cost : ℝ := 680
  let plot_width : ℝ := 97.5
  
  have h1 : plot_length > 0 := by sorry
  have h2 : path_width > 0 := by sorry
  have h3 : gravelling_cost_per_sq_meter > 0 := by sorry
  have h4 : total_gravelling_cost > 0 := by sorry
  
  have path_area : ℝ := 
    (plot_length + 2 * path_width) * (plot_width + 2 * path_width) - 
    plot_length * plot_width
  
  have total_cost_equation : 
    gravelling_cost_per_sq_meter * path_area = total_gravelling_cost := by sorry
  
  plot_width

end grassy_plot_width_l3492_349268


namespace quadratic_shift_l3492_349289

def f (b : ℝ) (x : ℝ) : ℝ := 2 * x^2 - b * x + 3

def g (b : ℝ) (x : ℝ) : ℝ := 2 * x^2 + b * x + 3

theorem quadratic_shift (b : ℝ) : 
  (∀ x, g b x = f b (x + 6)) → b = 12 := by
  sorry

end quadratic_shift_l3492_349289


namespace f_derivative_roots_l3492_349275

-- Define the function f
def f (x : ℝ) : ℝ := (1 - x) * (2 - x) * (3 - x) * (4 - x)

-- State the theorem
theorem f_derivative_roots :
  ∃ (r₁ r₂ r₃ : ℝ),
    (1 < r₁ ∧ r₁ < 2) ∧
    (2 < r₂ ∧ r₂ < 3) ∧
    (3 < r₃ ∧ r₃ < 4) ∧
    (∀ x : ℝ, deriv f x = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) :=
sorry

end f_derivative_roots_l3492_349275


namespace fifteenth_odd_multiple_of_5_l3492_349213

/-- The nth positive integer that is both odd and a multiple of 5 -/
def oddMultipleOf5 (n : ℕ) : ℕ := 2 * n * 5 - 5

theorem fifteenth_odd_multiple_of_5 : oddMultipleOf5 15 = 145 := by sorry

end fifteenth_odd_multiple_of_5_l3492_349213


namespace max_value_and_min_sum_of_squares_l3492_349286

noncomputable def f (x a b : ℝ) : ℝ := |x - a| - |x + 2*b|

theorem max_value_and_min_sum_of_squares
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (∀ x, f x a b ≤ a + 2*b) ∧
  (a + 2*b = 1 → ∃ (a₀ b₀ : ℝ), a₀^2 + 4*b₀^2 = 1/2 ∧ ∀ a' b', a'^2 + 4*b'^2 ≥ 1/2) :=
by sorry

end max_value_and_min_sum_of_squares_l3492_349286


namespace curve_equation_l3492_349256

noncomputable def x (t : ℝ) : ℝ := 3 * Real.cos t - Real.sin t
noncomputable def y (t : ℝ) : ℝ := 5 * Real.sin t

theorem curve_equation :
  ∃ (a b c : ℝ), ∀ (t : ℝ),
    a * (x t)^2 + b * (x t) * (y t) + c * (y t)^2 = 1 ∧
    a = 1/9 ∧ b = 2/45 ∧ c = 4/45 := by
  sorry

end curve_equation_l3492_349256


namespace smallest_perfect_cube_divisor_l3492_349238

theorem smallest_perfect_cube_divisor (p q r : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r) 
  (hpq : p ≠ q) (hpr : p ≠ r) (hqr : q ≠ r) : 
  let n := p^2 * q^3 * r^5
  ∀ m : ℕ, m > 0 → (∃ k : ℕ, m = k^3) → n ∣ m → p^6 * q^9 * r^15 ≤ m :=
by sorry

end smallest_perfect_cube_divisor_l3492_349238


namespace hyperbola_center_correct_l3492_349240

/-- The equation of a hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  (4 * x - 8)^2 / 9^2 - (5 * y + 5)^2 / 7^2 = 1

/-- The center of the hyperbola -/
def hyperbola_center : ℝ × ℝ := (2, -1)

/-- Theorem stating that hyperbola_center is the center of the hyperbola defined by hyperbola_equation -/
theorem hyperbola_center_correct :
  ∀ (x y : ℝ), hyperbola_equation x y ↔ 
    hyperbola_equation (x - hyperbola_center.1) (y - hyperbola_center.2) := by
  sorry

end hyperbola_center_correct_l3492_349240


namespace barbara_shopping_cost_l3492_349265

-- Define the quantities and prices
def tuna_packs : ℕ := 5
def tuna_price : ℚ := 2
def water_bottles : ℕ := 4
def water_price : ℚ := 3/2
def other_goods_cost : ℚ := 40

-- Define the total cost function
def total_cost : ℚ :=
  (tuna_packs * tuna_price) + (water_bottles * water_price) + other_goods_cost

-- Theorem statement
theorem barbara_shopping_cost :
  total_cost = 56 := by
  sorry

end barbara_shopping_cost_l3492_349265


namespace min_value_of_squares_l3492_349271

def S : Finset Int := {-8, -6, -4, -1, 1, 3, 5, 14}

theorem min_value_of_squares (p q r s t u v w : Int)
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧
                q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧
                r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧
                s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧
                t ≠ u ∧ t ≠ v ∧ t ≠ w ∧
                u ≠ v ∧ u ≠ w ∧
                v ≠ w)
  (h_in_set : p ∈ S ∧ q ∈ S ∧ r ∈ S ∧ s ∈ S ∧ t ∈ S ∧ u ∈ S ∧ v ∈ S ∧ w ∈ S)
  (h_sum : p + q + r + s ≥ 5) :
  (∀ a b c d e f g h : Int,
    a ∈ S → b ∈ S → c ∈ S → d ∈ S → e ∈ S → f ∈ S → g ∈ S → h ∈ S →
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
    e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
    f ≠ g ∧ f ≠ h ∧
    g ≠ h →
    a + b + c + d ≥ 5 →
    (a + b + c + d)^2 + (e + f + g + h)^2 ≥ 26) ∧
  (∃ a b c d e f g h : Int,
    a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ e ∈ S ∧ f ∈ S ∧ g ∈ S ∧ h ∈ S ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
    e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
    f ≠ g ∧ f ≠ h ∧
    g ≠ h ∧
    a + b + c + d ≥ 5 ∧
    (a + b + c + d)^2 + (e + f + g + h)^2 = 26) :=
by sorry

end min_value_of_squares_l3492_349271


namespace cubic_root_sum_squares_l3492_349277

theorem cubic_root_sum_squares (p q r : ℝ) (x : ℝ → ℝ) :
  (∀ t, x t = 0 ↔ t^3 - p*t^2 + q*t - r = 0) →
  ∃ r s t, (x r = 0 ∧ x s = 0 ∧ x t = 0) ∧ 
           (r^2 + s^2 + t^2 = p^2 - 2*q) :=
by sorry

end cubic_root_sum_squares_l3492_349277


namespace max_value_of_sin_cos_ratio_l3492_349267

theorem max_value_of_sin_cos_ratio (α β γ : Real) 
  (h_acute_α : 0 < α ∧ α < π/2)
  (h_acute_β : 0 < β ∧ β < π/2)
  (h_acute_γ : 0 < γ ∧ γ < π/2)
  (h_sum_sin_sq : Real.sin α ^ 2 + Real.sin β ^ 2 + Real.sin γ ^ 2 = 1) :
  (Real.sin α + Real.sin β + Real.sin γ) / (Real.cos α + Real.cos β + Real.cos γ) ≤ Real.sqrt 2 / 2 := by
  sorry


end max_value_of_sin_cos_ratio_l3492_349267


namespace sandy_potatoes_l3492_349281

theorem sandy_potatoes (nancy_potatoes : ℕ) (total_potatoes : ℕ) (sandy_potatoes : ℕ) : 
  nancy_potatoes = 6 → 
  total_potatoes = 13 → 
  total_potatoes = nancy_potatoes + sandy_potatoes → 
  sandy_potatoes = 7 := by
sorry

end sandy_potatoes_l3492_349281


namespace max_value_d_l3492_349272

theorem max_value_d (a b c d : ℝ) 
  (sum_condition : a + b + c + d = 10)
  (product_condition : a * b + a * c + a * d + b * c + b * d + c * d = 20) :
  d ≤ (5 + Real.sqrt 105) / 2 :=
sorry

end max_value_d_l3492_349272


namespace line_intersection_with_x_axis_l3492_349203

/-- A line parallel to y = -3x that passes through (0, -2) intersects the x-axis at (-2/3, 0) -/
theorem line_intersection_with_x_axis :
  ∀ (k b : ℝ),
  (∀ x y : ℝ, y = k * x + b ↔ y = -3 * x + b) →  -- Line is parallel to y = -3x
  -2 = k * 0 + b →                               -- Line passes through (0, -2)
  ∃ x : ℝ, x = -2/3 ∧ 0 = k * x + b :=           -- Intersection point with x-axis
by sorry

end line_intersection_with_x_axis_l3492_349203


namespace sets_partition_l3492_349241

-- Define the set of positive integers
def PositiveIntegers : Set ℕ := {n : ℕ | n > 0}

-- Define a property for primes greater than 2013
def IsPrimeGreaterThan2013 (n : ℕ) : Prop :=
  Nat.Prime n ∧ n > 2013

-- Define the property for the special difference condition
def SpecialDifference (A B : Set ℕ) : Prop :=
  ∀ (x y : ℕ), x ∈ PositiveIntegers → y ∈ PositiveIntegers →
    IsPrimeGreaterThan2013 (x - y) →
    ((x ∈ A ∧ y ∈ B) ∨ (x ∈ B ∧ y ∈ A))

theorem sets_partition (A B : Set ℕ) :
  (A ∪ B = PositiveIntegers) →
  (A ∩ B = ∅) →
  SpecialDifference A B →
  ((∀ n : ℕ, n ∈ A ↔ n ∈ PositiveIntegers ∧ Even n) ∧
   (∀ n : ℕ, n ∈ B ↔ n ∈ PositiveIntegers ∧ Odd n)) :=
by sorry

end sets_partition_l3492_349241


namespace four_line_segment_lengths_exists_distinct_positive_integer_lengths_l3492_349290

-- Define a type for lines in a plane
def Line : Type := ℝ → ℝ → Prop

-- Define a type for points in a plane
def Point : Type := ℝ × ℝ

-- Define a function to check if three lines are concurrent
def areConcurrent (l₁ l₂ l₃ : Line) : Prop := sorry

-- Define a function to check if two lines intersect
def intersect (l₁ l₂ : Line) : Prop := sorry

-- Define a function to get the length of a line segment
def segmentLength (p₁ p₂ : Point) : ℝ := sorry

-- Define the configuration of four lines
structure FourLineConfiguration :=
  (lines : Fin 4 → Line)
  (intersectionPoints : Fin 6 → Point)
  (segmentLengths : Fin 8 → ℝ)
  (twoLinesIntersect : ∀ i j, i ≠ j → intersect (lines i) (lines j))
  (noThreeConcurrent : ∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → ¬areConcurrent (lines i) (lines j) (lines k))
  (eightSegments : ∀ i, segmentLengths i > 0)
  (distinctSegments : ∀ i j, i ≠ j → segmentLengths i ≠ segmentLengths j)

theorem four_line_segment_lengths 
  (config : FourLineConfiguration) : 
  (∀ i : Fin 8, config.segmentLengths i = i.val + 1) → False :=
sorry

theorem exists_distinct_positive_integer_lengths 
  (config : FourLineConfiguration) :
  ∃ (lengths : Fin 8 → ℕ), ∀ i : Fin 8, config.segmentLengths i = lengths i ∧ lengths i > 0 :=
sorry

end four_line_segment_lengths_exists_distinct_positive_integer_lengths_l3492_349290


namespace det_special_matrix_l3492_349236

theorem det_special_matrix (x y : ℝ) : 
  Matrix.det !![0, Real.cos x, Real.sin x; 
                -Real.cos x, 0, Real.cos y; 
                -Real.sin x, -Real.cos y, 0] = 0 := by
  sorry

end det_special_matrix_l3492_349236


namespace sequence_properties_l3492_349223

def a (n : ℕ) : ℕ := 3 * (n^2 + n) + 7

theorem sequence_properties :
  (∀ k : ℕ, 
    5 ∣ a (5*k + 2) ∧ 
    ¬(5 ∣ a (5*k)) ∧ 
    ¬(5 ∣ a (5*k + 1)) ∧ 
    ¬(5 ∣ a (5*k + 3)) ∧ 
    ¬(5 ∣ a (5*k + 4))) ∧
  (∀ n t : ℕ, a n ≠ t^3) := by
  sorry

end sequence_properties_l3492_349223


namespace trash_can_problem_l3492_349201

/-- Represents the unit price of trash can A -/
def price_A : ℝ := 60

/-- Represents the unit price of trash can B -/
def price_B : ℝ := 100

/-- Represents the total number of trash cans needed -/
def total_cans : ℕ := 200

/-- Represents the maximum total cost allowed -/
def max_cost : ℝ := 15000

theorem trash_can_problem :
  (3 * price_A + 4 * price_B = 580) ∧
  (6 * price_A + 5 * price_B = 860) ∧
  (∀ a : ℕ, a ≥ 125 → 
    (price_A * a + price_B * (total_cans - a) ≤ max_cost)) ∧
  (∀ a : ℕ, a < 125 → 
    (price_A * a + price_B * (total_cans - a) > max_cost)) :=
by sorry

end trash_can_problem_l3492_349201


namespace sara_frosting_cans_l3492_349299

/-- Represents the data for each day's baking and frosting --/
structure DayData where
  baked : ℕ
  eaten : ℕ
  frostingPerCake : ℕ

/-- Calculates the total frosting cans needed for the remaining cakes --/
def totalFrostingCans (data : List DayData) : ℕ :=
  data.foldl (fun acc day => acc + (day.baked - day.eaten) * day.frostingPerCake) 0

/-- The main theorem stating the total number of frosting cans needed --/
theorem sara_frosting_cans : 
  let bakingData : List DayData := [
    ⟨7, 4, 2⟩,
    ⟨12, 6, 3⟩,
    ⟨8, 3, 4⟩,
    ⟨10, 2, 3⟩,
    ⟨15, 3, 2⟩
  ]
  totalFrostingCans bakingData = 92 := by
  sorry

#eval totalFrostingCans [
  ⟨7, 4, 2⟩,
  ⟨12, 6, 3⟩,
  ⟨8, 3, 4⟩,
  ⟨10, 2, 3⟩,
  ⟨15, 3, 2⟩
]

end sara_frosting_cans_l3492_349299


namespace rectangle_diagonal_ratio_l3492_349254

theorem rectangle_diagonal_ratio (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a ≤ b) :
  (a + b - Real.sqrt (a^2 + b^2) = b / 3) → (a / b = 5 / 12) := by
  sorry

end rectangle_diagonal_ratio_l3492_349254


namespace emilys_friends_with_color_boxes_l3492_349276

def rainbow_colors : ℕ := 7
def total_pencils : ℕ := 56

theorem emilys_friends_with_color_boxes :
  ∀ (pencils_per_box : ℕ) (total_boxes : ℕ),
    pencils_per_box = rainbow_colors →
    total_pencils = pencils_per_box * total_boxes →
    total_boxes - 1 = 7 := by
  sorry

end emilys_friends_with_color_boxes_l3492_349276


namespace square_problem_l3492_349228

/-- Square with side length 800 -/
structure Square :=
  (side : ℝ)
  (is_800 : side = 800)

/-- Point on the side of the square -/
structure PointOnSide :=
  (x : ℝ)
  (in_range : 0 ≤ x ∧ x ≤ 800)

/-- Expression of the form p + q√r -/
structure SurdExpression :=
  (p q r : ℕ)
  (r_not_perfect_square : ∀ (n : ℕ), n > 1 → ¬(r.gcd (n^2) > 1))

/-- Main theorem -/
theorem square_problem (S : Square) (E F : PointOnSide) (BF : SurdExpression) :
  S.side = 800 →
  E.x < F.x →
  F.x - E.x = 300 →
  Real.cos (60 * π / 180) * (F.x - 400) = Real.sin (60 * π / 180) * 400 →
  800 - F.x = BF.p + BF.q * Real.sqrt BF.r →
  BF.p + BF.q + BF.r = 334 := by
  sorry

end square_problem_l3492_349228


namespace josh_marbles_l3492_349292

theorem josh_marbles (initial_marbles final_marbles received_marbles : ℕ) :
  final_marbles = initial_marbles + received_marbles →
  final_marbles = 42 →
  received_marbles = 20 →
  initial_marbles = 22 := by sorry

end josh_marbles_l3492_349292
