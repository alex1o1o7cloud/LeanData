import Mathlib

namespace NUMINAMATH_CALUDE_cornelia_age_l2677_267717

theorem cornelia_age (kilee_age : ℕ) (future_years : ℕ) :
  kilee_age = 20 →
  future_years = 10 →
  ∃ (cornelia_age : ℕ),
    cornelia_age + future_years = 3 * (kilee_age + future_years) ∧
    cornelia_age = 80 :=
by sorry

end NUMINAMATH_CALUDE_cornelia_age_l2677_267717


namespace NUMINAMATH_CALUDE_bug_path_tiles_l2677_267705

/-- Represents a rectangular floor with integer dimensions -/
structure RectangularFloor where
  width : ℕ
  length : ℕ

/-- Calculates the number of tiles a bug visits when walking diagonally across a rectangular floor -/
def tilesVisited (floor : RectangularFloor) : ℕ :=
  floor.width + floor.length - Nat.gcd floor.width floor.length

theorem bug_path_tiles (floor : RectangularFloor) 
  (h_width : floor.width = 9) 
  (h_length : floor.length = 13) : 
  tilesVisited floor = 21 := by
sorry

#eval tilesVisited ⟨9, 13⟩

end NUMINAMATH_CALUDE_bug_path_tiles_l2677_267705


namespace NUMINAMATH_CALUDE_train_length_l2677_267751

theorem train_length (time : Real) (speed_kmh : Real) (length : Real) : 
  time = 2.222044458665529 →
  speed_kmh = 162 →
  length = speed_kmh * (1000 / 3600) * time →
  length = 100 := by
sorry


end NUMINAMATH_CALUDE_train_length_l2677_267751


namespace NUMINAMATH_CALUDE_congruence_problem_l2677_267734

theorem congruence_problem (x : ℤ) :
  (4 * x + 9) % 17 = 3 → (3 * x + 12) % 17 = 16 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l2677_267734


namespace NUMINAMATH_CALUDE_b_age_is_eight_l2677_267788

/-- Given three people a, b, and c, where:
    - a is two years older than b
    - b is twice as old as c
    - The total of their ages is 22
    Prove that b is 8 years old. -/
theorem b_age_is_eight (a b c : ℕ) 
  (h1 : a = b + 2)
  (h2 : b = 2 * c)
  (h3 : a + b + c = 22) : 
  b = 8 := by
  sorry

end NUMINAMATH_CALUDE_b_age_is_eight_l2677_267788


namespace NUMINAMATH_CALUDE_triangle_count_l2677_267780

/-- The number of points on the circumference of the circle -/
def n : ℕ := 7

/-- The number of points needed to form a triangle -/
def k : ℕ := 3

/-- The number of different triangles that can be formed -/
def num_triangles : ℕ := Nat.choose n k

theorem triangle_count : num_triangles = 35 := by sorry

end NUMINAMATH_CALUDE_triangle_count_l2677_267780


namespace NUMINAMATH_CALUDE_jakes_weight_l2677_267712

theorem jakes_weight (j k : ℝ) 
  (h1 : j - 8 = 2 * k)  -- If Jake loses 8 pounds, he will weigh twice as much as Kendra
  (h2 : j + k = 290)    -- Together they now weigh 290 pounds
  : j = 196 :=          -- Jake's present weight is 196 pounds
by sorry

end NUMINAMATH_CALUDE_jakes_weight_l2677_267712


namespace NUMINAMATH_CALUDE_large_birdhouses_sold_l2677_267725

/-- Represents the number of large birdhouses sold -/
def large_birdhouses : ℕ := sorry

/-- The price of a large birdhouse in dollars -/
def large_price : ℕ := 22

/-- The price of a medium birdhouse in dollars -/
def medium_price : ℕ := 16

/-- The price of a small birdhouse in dollars -/
def small_price : ℕ := 7

/-- The number of medium birdhouses sold -/
def medium_sold : ℕ := 2

/-- The number of small birdhouses sold -/
def small_sold : ℕ := 3

/-- The total sales in dollars -/
def total_sales : ℕ := 97

/-- Theorem stating that the number of large birdhouses sold is 2 -/
theorem large_birdhouses_sold : large_birdhouses = 2 := by
  sorry

end NUMINAMATH_CALUDE_large_birdhouses_sold_l2677_267725


namespace NUMINAMATH_CALUDE_equilateral_triangle_filling_l2677_267748

theorem equilateral_triangle_filling :
  let large_side : ℝ := 15
  let small_side : ℝ := 3
  let area (side : ℝ) := (Real.sqrt 3 / 4) * side^2
  let large_area := area large_side
  let small_area := area small_side
  let num_small_triangles := large_area / small_area
  num_small_triangles = 25 := by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_filling_l2677_267748


namespace NUMINAMATH_CALUDE_staircase_perimeter_l2677_267707

/-- A staircase-shaped region with right angles -/
structure StaircaseRegion where
  /-- The number of 1-foot sides in the staircase -/
  num_sides : ℕ
  /-- The area of the region in square feet -/
  area : ℝ
  /-- Assumption that the number of sides is 10 -/
  sides_eq_ten : num_sides = 10
  /-- Assumption that the area is 85 square feet -/
  area_eq_85 : area = 85

/-- Calculate the perimeter of a staircase region -/
def perimeter (r : StaircaseRegion) : ℝ := sorry

/-- Theorem stating that the perimeter of the given staircase region is 30.5 feet -/
theorem staircase_perimeter (r : StaircaseRegion) : perimeter r = 30.5 := by sorry

end NUMINAMATH_CALUDE_staircase_perimeter_l2677_267707


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2677_267718

theorem complex_fraction_equality : (1 - I) * (1 + 2*I) / (1 + I) = 2 - I := by sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2677_267718


namespace NUMINAMATH_CALUDE_no_m_exists_for_all_x_inequality_l2677_267743

theorem no_m_exists_for_all_x_inequality :
  ¬ ∃ m : ℝ, ∀ x : ℝ, m * x^2 - 2*x - m + 1 < 0 := by
  sorry

end NUMINAMATH_CALUDE_no_m_exists_for_all_x_inequality_l2677_267743


namespace NUMINAMATH_CALUDE_parabola_with_origin_vertex_and_neg_two_directrix_l2677_267738

/-- A parabola is defined by its vertex and directrix -/
structure Parabola where
  vertex : ℝ × ℝ
  directrix : ℝ

/-- The equation of a parabola given its vertex and directrix -/
def parabola_equation (p : Parabola) : ℝ → ℝ → Prop :=
  fun x y => y^2 = 4 * (x - p.vertex.1) * (p.vertex.1 - p.directrix)

theorem parabola_with_origin_vertex_and_neg_two_directrix :
  let p : Parabola := { vertex := (0, 0), directrix := -2 }
  ∀ x y : ℝ, parabola_equation p x y ↔ y^2 = 8*x := by sorry

end NUMINAMATH_CALUDE_parabola_with_origin_vertex_and_neg_two_directrix_l2677_267738


namespace NUMINAMATH_CALUDE_cubic_polynomial_sum_l2677_267755

/-- A cubic polynomial with coefficients in ℝ -/
structure CubicPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Evaluation of a cubic polynomial at a point -/
def CubicPolynomial.eval (Q : CubicPolynomial) (x : ℝ) : ℝ :=
  Q.a * x^3 + Q.b * x^2 + Q.c * x + Q.d

theorem cubic_polynomial_sum (k : ℝ) (Q : CubicPolynomial) 
    (h0 : Q.eval 0 = k)
    (h1 : Q.eval 1 = 3*k)
    (h2 : Q.eval (-1) = 4*k) :
  Q.eval 2 + Q.eval (-2) = 22*k := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_sum_l2677_267755


namespace NUMINAMATH_CALUDE_total_fish_in_lake_l2677_267733

/-- The number of fish per white duck -/
def fishPerWhiteDuck : ℕ := 5

/-- The number of fish per black duck -/
def fishPerBlackDuck : ℕ := 10

/-- The number of fish per multicolor duck -/
def fishPerMulticolorDuck : ℕ := 12

/-- The number of white ducks -/
def numWhiteDucks : ℕ := 3

/-- The number of black ducks -/
def numBlackDucks : ℕ := 7

/-- The number of multicolor ducks -/
def numMulticolorDucks : ℕ := 6

/-- The total number of fish in the lake -/
def totalFish : ℕ := fishPerWhiteDuck * numWhiteDucks + 
                     fishPerBlackDuck * numBlackDucks + 
                     fishPerMulticolorDuck * numMulticolorDucks

theorem total_fish_in_lake : totalFish = 157 := by
  sorry

end NUMINAMATH_CALUDE_total_fish_in_lake_l2677_267733


namespace NUMINAMATH_CALUDE_mean_proportional_existence_l2677_267746

theorem mean_proportional_existence (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃ x : ℝ, x ^ 2 = a * b :=
by sorry

end NUMINAMATH_CALUDE_mean_proportional_existence_l2677_267746


namespace NUMINAMATH_CALUDE_quadratic_root_existence_l2677_267782

theorem quadratic_root_existence (a b c x₁ x₂ : ℝ) 
  (ha : a ≠ 0)
  (hx₁ : a * x₁^2 + b * x₁ + c = 0)
  (hx₂ : -a * x₂^2 + b * x₂ + c = 0) :
  ∃ x₃, (x₃ ∈ Set.Icc x₁ x₂ ∨ x₃ ∈ Set.Icc x₂ x₁) ∧ 
        (a / 2) * x₃^2 + b * x₃ + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_existence_l2677_267782


namespace NUMINAMATH_CALUDE_discount_percentage_calculation_l2677_267724

/-- Given the cost price, marked price, and profit percentage, calculate the discount percentage. -/
theorem discount_percentage_calculation (cost_price marked_price : ℝ) (profit_percentage : ℝ) :
  cost_price = 47.50 →
  marked_price = 64.54 →
  profit_percentage = 25 →
  ∃ (discount_percentage : ℝ), 
    (abs (discount_percentage - 8) < 0.1) ∧ 
    (cost_price * (1 + profit_percentage / 100) = marked_price * (1 - discount_percentage / 100)) := by
  sorry


end NUMINAMATH_CALUDE_discount_percentage_calculation_l2677_267724


namespace NUMINAMATH_CALUDE_soap_box_width_maximizes_boxes_l2677_267710

/-- The width of a soap box that maximizes the number of boxes in a carton. -/
def soap_box_width : ℝ :=
  let carton_volume : ℝ := 30 * 42 * 60
  let max_boxes : ℕ := 360
  let soap_box_length : ℝ := 7
  let soap_box_height : ℝ := 5
  6

/-- Theorem stating that the calculated width maximizes the number of soap boxes in the carton. -/
theorem soap_box_width_maximizes_boxes (carton_volume : ℝ) (max_boxes : ℕ) 
    (soap_box_length soap_box_height : ℝ) :
  carton_volume = 30 * 42 * 60 →
  max_boxes = 360 →
  soap_box_length = 7 →
  soap_box_height = 5 →
  soap_box_width * soap_box_length * soap_box_height * max_boxes = carton_volume :=
by sorry

end NUMINAMATH_CALUDE_soap_box_width_maximizes_boxes_l2677_267710


namespace NUMINAMATH_CALUDE_probability_bounds_l2677_267760

theorem probability_bounds (n : ℕ) (m₀ : ℕ) (p : ℝ) 
  (h_n : n = 120) 
  (h_m₀ : m₀ = 32) 
  (h_most_probable : m₀ = ⌊n * p + 0.5⌋) : 
  32 / 121 ≤ p ∧ p ≤ 33 / 121 := by
  sorry

end NUMINAMATH_CALUDE_probability_bounds_l2677_267760


namespace NUMINAMATH_CALUDE_total_numbers_correct_l2677_267726

/-- Represents a student in the talent show -/
inductive Student : Type
| Sarah : Student
| Ben : Student
| Jake : Student
| Lily : Student

/-- The total number of musical numbers in the show -/
def total_numbers : ℕ := 7

/-- The number of songs Sarah performed -/
def sarah_songs : ℕ := 6

/-- The number of songs Ben performed -/
def ben_songs : ℕ := sarah_songs - 3

/-- The number of songs Jake performed -/
def jake_songs : ℕ := 6

/-- The number of songs Lily performed -/
def lily_songs : ℕ := 6

/-- The number of duo shows Jake and Lily performed together -/
def jake_lily_duo : ℕ := 1

/-- The number of shows Jake and Lily performed together -/
def jake_lily_together : ℕ := 6

/-- Theorem stating that the total number of musical numbers is correct -/
theorem total_numbers_correct : 
  (sarah_songs = total_numbers - 2) ∧ 
  (ben_songs = sarah_songs - 3) ∧
  (jake_songs = lily_songs) ∧
  (jake_lily_together ≤ 7) ∧
  (jake_lily_together > jake_songs - jake_lily_duo) ∧
  (total_numbers = jake_songs + 1) := by
  sorry

#check total_numbers_correct

end NUMINAMATH_CALUDE_total_numbers_correct_l2677_267726


namespace NUMINAMATH_CALUDE_polynomial_g_forms_l2677_267709

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- Define the property that g must satisfy
def g_property (g : ℝ → ℝ) : Prop :=
  ∀ x, f (g x) = 9*x^2 - 6*x + 1

-- State the theorem
theorem polynomial_g_forms :
  ∀ g : ℝ → ℝ, g_property g →
  (∀ x, g x = 3*x - 1) ∨ (∀ x, g x = -3*x + 1) :=
sorry

end NUMINAMATH_CALUDE_polynomial_g_forms_l2677_267709


namespace NUMINAMATH_CALUDE_sector_radius_l2677_267766

theorem sector_radius (area : Real) (angle : Real) (π : Real) (h1 : area = 36.67) (h2 : angle = 42) (h3 : π = 3.14159) :
  ∃ r : Real, r = 10 ∧ area = (angle / 360) * π * r^2 := by
  sorry

end NUMINAMATH_CALUDE_sector_radius_l2677_267766


namespace NUMINAMATH_CALUDE_rainfall_problem_l2677_267775

theorem rainfall_problem (total_rainfall : ℝ) (ratio : ℝ) :
  total_rainfall = 30 →
  ratio = 1.5 →
  ∃ (first_week : ℝ) (second_week : ℝ),
    first_week + second_week = total_rainfall ∧
    second_week = ratio * first_week ∧
    second_week = 18 := by
  sorry

end NUMINAMATH_CALUDE_rainfall_problem_l2677_267775


namespace NUMINAMATH_CALUDE_existence_of_three_similar_numbers_l2677_267779

def is_1995_digit (n : ℕ) : Prop := n ≥ 10^1994 ∧ n < 10^1995

def composed_of_4_5_9 (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 4 ∨ d = 5 ∨ d = 9

def similar (a b : ℕ) : Prop :=
  ∀ d : ℕ, (d ∈ a.digits 10 ↔ d ∈ b.digits 10)

theorem existence_of_three_similar_numbers :
  ∃ (A B C : ℕ),
    is_1995_digit A ∧
    is_1995_digit B ∧
    is_1995_digit C ∧
    composed_of_4_5_9 A ∧
    composed_of_4_5_9 B ∧
    composed_of_4_5_9 C ∧
    similar A B ∧
    similar B C ∧
    similar A C ∧
    A + B = C :=
  sorry

end NUMINAMATH_CALUDE_existence_of_three_similar_numbers_l2677_267779


namespace NUMINAMATH_CALUDE_expected_remaining_balls_l2677_267735

/-- Represents the number of red balls initially in the bag -/
def redBalls : ℕ := 100

/-- Represents the number of blue balls initially in the bag -/
def blueBalls : ℕ := 100

/-- Represents the total number of balls initially in the bag -/
def totalBalls : ℕ := redBalls + blueBalls

/-- Represents the process of drawing balls without replacement until all red balls are drawn -/
def drawUntilAllRed (red blue : ℕ) : ℝ := sorry

/-- Theorem stating the expected number of remaining balls after drawing all red balls -/
theorem expected_remaining_balls :
  drawUntilAllRed redBalls blueBalls = blueBalls / (totalBalls : ℝ) := by sorry

end NUMINAMATH_CALUDE_expected_remaining_balls_l2677_267735


namespace NUMINAMATH_CALUDE_unique_n_satisfying_conditions_l2677_267794

theorem unique_n_satisfying_conditions : ∃! n : ℤ,
  50 < n ∧ n < 120 ∧
  n % 8 = 0 ∧
  n % 9 = 5 ∧
  n % 7 = 3 ∧
  n = 104 := by
  sorry

end NUMINAMATH_CALUDE_unique_n_satisfying_conditions_l2677_267794


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_decrease_l2677_267704

/-- Calculates the percentage decrease in area of an equilateral triangle 
    when its side length is decreased by a fixed amount. -/
theorem equilateral_triangle_area_decrease (initial_area : ℝ) (side_decrease : ℝ) : 
  initial_area = 100 * Real.sqrt 3 →
  side_decrease = 6 →
  let initial_side := Real.sqrt ((4 * initial_area) / (Real.sqrt 3))
  let new_side := initial_side - side_decrease
  let new_area := (Real.sqrt 3 / 4) * new_side ^ 2
  let area_decrease_percentage := (initial_area - new_area) / initial_area * 100
  area_decrease_percentage = 51 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_decrease_l2677_267704


namespace NUMINAMATH_CALUDE_monotonically_decreasing_interval_of_f_shifted_l2677_267753

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the derivative of f
axiom f_derivative (x : ℝ) : deriv f x = 2 * x - 4

-- Theorem statement
theorem monotonically_decreasing_interval_of_f_shifted :
  ∀ x : ℝ, (∀ y : ℝ, y < 3 → deriv (fun z ↦ f (z - 1)) y < 0) ∧
           (∀ y : ℝ, y ≥ 3 → deriv (fun z ↦ f (z - 1)) y ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_monotonically_decreasing_interval_of_f_shifted_l2677_267753


namespace NUMINAMATH_CALUDE_basketball_substitutions_l2677_267749

def substitution_ways (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | k + 1 => (5 - k) * (11 + k) * substitution_ways k

def total_substitution_ways : ℕ :=
  (List.range 6).map substitution_ways |> List.sum

theorem basketball_substitutions :
  total_substitution_ways % 1000 = 736 := by
  sorry

end NUMINAMATH_CALUDE_basketball_substitutions_l2677_267749


namespace NUMINAMATH_CALUDE_sofa_purchase_sum_l2677_267720

/-- The sum of Joan and Karl's sofa purchases -/
def total_purchase (joan_price karl_price : ℝ) : ℝ := joan_price + karl_price

/-- Theorem: Given the conditions, the sum of Joan and Karl's sofa purchases is $600 -/
theorem sofa_purchase_sum :
  ∀ (joan_price karl_price : ℝ),
  joan_price = 230 →
  2 * joan_price = karl_price + 90 →
  total_purchase joan_price karl_price = 600 := by
sorry

end NUMINAMATH_CALUDE_sofa_purchase_sum_l2677_267720


namespace NUMINAMATH_CALUDE_pages_per_day_l2677_267768

/-- Given a book with 240 pages read over 12 days with equal pages per day, prove that 20 pages are read daily. -/
theorem pages_per_day (total_pages : ℕ) (days : ℕ) (pages_per_day : ℕ) : 
  total_pages = 240 → days = 12 → total_pages = days * pages_per_day → pages_per_day = 20 := by
  sorry

end NUMINAMATH_CALUDE_pages_per_day_l2677_267768


namespace NUMINAMATH_CALUDE_elderly_sample_count_l2677_267799

/-- Represents the number of employees in different age groups and samples --/
structure EmployeeCount where
  total : ℕ
  young : ℕ
  elderly : ℕ
  youngSample : ℕ
  elderlySample : ℕ

/-- Conditions of the employee distribution and sampling --/
def validEmployeeCount (ec : EmployeeCount) : Prop :=
  ec.total = 430 ∧
  ec.young = 160 ∧
  ec.total = ec.young + 2 * ec.elderly + ec.elderly ∧
  ec.youngSample = 32 ∧
  ec.youngSample / ec.young = ec.elderlySample / ec.elderly

theorem elderly_sample_count (ec : EmployeeCount) (h : validEmployeeCount ec) :
  ec.elderlySample = 18 := by
  sorry


end NUMINAMATH_CALUDE_elderly_sample_count_l2677_267799


namespace NUMINAMATH_CALUDE_complex_imaginary_operation_l2677_267723

theorem complex_imaginary_operation : Complex.I - (1 / Complex.I) = 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_imaginary_operation_l2677_267723


namespace NUMINAMATH_CALUDE_amount_to_find_l2677_267796

def water_bottles : ℕ := 5 * 12
def energy_bars : ℕ := 4 * 12
def original_water_price : ℚ := 2
def original_energy_price : ℚ := 3
def market_water_price : ℚ := 185/100
def market_energy_price : ℚ := 275/100
def discount_rate : ℚ := 1/10

def original_total : ℚ := water_bottles * original_water_price + energy_bars * original_energy_price

def discounted_water_price : ℚ := market_water_price * (1 - discount_rate)
def discounted_energy_price : ℚ := market_energy_price * (1 - discount_rate)

def discounted_total : ℚ := water_bottles * discounted_water_price + energy_bars * discounted_energy_price

theorem amount_to_find : original_total - discounted_total = 453/10 := by sorry

end NUMINAMATH_CALUDE_amount_to_find_l2677_267796


namespace NUMINAMATH_CALUDE_sum_of_three_circles_l2677_267795

-- Define the values for triangles and circles
variable (triangle : ℝ)
variable (circle : ℝ)

-- Define the conditions
axiom condition1 : 3 * triangle + 2 * circle = 21
axiom condition2 : 2 * triangle + 3 * circle = 19

-- Theorem to prove
theorem sum_of_three_circles : 3 * circle = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_circles_l2677_267795


namespace NUMINAMATH_CALUDE_star_equation_solution_l2677_267702

-- Define the star operation
def star (a b : ℚ) : ℚ := a * b + 3 * b - a

-- State the theorem
theorem star_equation_solution :
  ∀ y : ℚ, star 4 y = 40 → y = 44 / 7 := by
  sorry

end NUMINAMATH_CALUDE_star_equation_solution_l2677_267702


namespace NUMINAMATH_CALUDE_max_a_value_l2677_267713

open Real

theorem max_a_value (e : ℝ) (h_e : e = exp 1) :
  let a_max := 1/2 + log 2/2 - e
  ∀ a : ℝ, (∀ x : ℝ, x ∈ Set.Icc (1/e) 2 → (a + e) * x - 1 - log x ≤ 0) →
  a ≤ a_max ∧
  ∃ x : ℝ, x ∈ Set.Icc (1/e) 2 ∧ (a_max + e) * x - 1 - log x = 0 :=
sorry

end NUMINAMATH_CALUDE_max_a_value_l2677_267713


namespace NUMINAMATH_CALUDE_average_speed_proof_l2677_267773

/-- Proves that the average speed of a trip is 32 km/h given the specified conditions -/
theorem average_speed_proof (total_distance : ℝ) (distance1 : ℝ) (speed1 : ℝ) (distance2 : ℝ) (speed2 : ℝ) :
  total_distance = 60 →
  distance1 = 30 →
  speed1 = 48 →
  distance2 = 30 →
  speed2 = 24 →
  (total_distance / ((distance1 / speed1) + (distance2 / speed2))) = 32 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_proof_l2677_267773


namespace NUMINAMATH_CALUDE_sequence_properties_l2677_267745

/-- Definition of the sequence a_n -/
def a : ℕ → ℝ
  | 0 => 4
  | n + 1 => 2 * a n - 2 * (n + 1) + 1

/-- Definition of the sequence b_n -/
def b (t : ℝ) (n : ℕ) : ℝ := t * n + 2

/-- Theorem statement -/
theorem sequence_properties :
  (∀ n : ℕ, a (n + 1) - 2 * (n + 1) - 1 = 2 * (a n - 2 * n - 1)) ∧
  (∀ t : ℝ, (∀ n : ℕ, b t (n + 1) < 2 * a (n + 1)) → t < 6) := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l2677_267745


namespace NUMINAMATH_CALUDE_athlete_speed_l2677_267714

/-- Given an athlete running 200 meters in 24 seconds, prove their speed is approximately 30 km/h -/
theorem athlete_speed (distance : Real) (time : Real) (h1 : distance = 200) (h2 : time = 24) :
  ∃ (speed : Real), abs (speed - 30) < 0.1 ∧ speed = (distance / 1000) / (time / 3600) := by
  sorry

end NUMINAMATH_CALUDE_athlete_speed_l2677_267714


namespace NUMINAMATH_CALUDE_scarves_per_box_chloes_scarves_l2677_267791

theorem scarves_per_box (num_boxes : ℕ) (mittens_per_box : ℕ) (total_clothing : ℕ) : ℕ :=
  let total_mittens := num_boxes * mittens_per_box
  let total_scarves := total_clothing - total_mittens
  let scarves_per_box := total_scarves / num_boxes
  scarves_per_box

theorem chloes_scarves :
  scarves_per_box 4 6 32 = 2 := by
  sorry

end NUMINAMATH_CALUDE_scarves_per_box_chloes_scarves_l2677_267791


namespace NUMINAMATH_CALUDE_divisibility_condition_l2677_267759

theorem divisibility_condition (x y : ℕ+) :
  (∃ (k : ℕ+), k * (2 * x + 7 * y) = 7 * x + 2 * y) ↔
  (∃ (a : ℕ+), (x = a ∧ y = a) ∨ (x = 4 * a ∧ y = a) ∨ (x = 19 * a ∧ y = a)) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_condition_l2677_267759


namespace NUMINAMATH_CALUDE_max_x_minus_y_is_half_l2677_267776

theorem max_x_minus_y_is_half :
  ∀ x y : ℝ, 2 * (x^2 + y^2 - x*y) = x + y →
  ∀ z : ℝ, z = x - y → z ≤ (1/2 : ℝ) ∧ ∃ x₀ y₀ : ℝ, 2 * (x₀^2 + y₀^2 - x₀*y₀) = x₀ + y₀ ∧ x₀ - y₀ = (1/2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_max_x_minus_y_is_half_l2677_267776


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2677_267762

/-- A positive geometric sequence -/
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

/-- The given arithmetic sequence condition -/
def arithmetic_sequence_condition (a : ℕ → ℝ) : Prop :=
  2 * ((1/2) * a 3) = 3 * a 1 + 2 * a 2

theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  is_positive_geometric_sequence a →
  arithmetic_sequence_condition a →
  (a 2014 - a 2015) / (a 2016 - a 2017) = 1/9 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2677_267762


namespace NUMINAMATH_CALUDE_train_average_speed_l2677_267769

theorem train_average_speed (distance1 distance2 time1 time2 : ℝ) 
  (h1 : distance1 = 250)
  (h2 : distance2 = 350)
  (h3 : time1 = 2)
  (h4 : time2 = 4) :
  (distance1 + distance2) / (time1 + time2) = 100 := by
  sorry

end NUMINAMATH_CALUDE_train_average_speed_l2677_267769


namespace NUMINAMATH_CALUDE_schoolyard_area_increase_l2677_267737

theorem schoolyard_area_increase :
  ∀ l w : ℝ,
  l > 0 → w > 0 →
  2 * l + 2 * w = 160 →
  (l + 10) * (w + 10) - l * w = 900 :=
by sorry

end NUMINAMATH_CALUDE_schoolyard_area_increase_l2677_267737


namespace NUMINAMATH_CALUDE_range_of_m_l2677_267722

/-- The set A defined by a quadratic inequality -/
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

/-- The set B defined by a quadratic inequality with parameter m -/
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*m*x + m^2 - 4 ≤ 0}

/-- The theorem stating the range of m given A is a subset of the complement of B -/
theorem range_of_m (m : ℝ) : A ⊆ (Set.univ \ B m) → m < -3 ∨ m > 5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l2677_267722


namespace NUMINAMATH_CALUDE_not_all_new_releases_implies_exists_not_new_and_not_all_new_l2677_267701

-- Define the universe of books in the library
variable (Book : Type)

-- Define the property of being a new release
variable (is_new_release : Book → Prop)

-- Define the library as a set of books
variable (library : Set Book)

-- Theorem stating that if not all books are new releases, 
-- then there exists a book that is not a new release and not all books are new releases
theorem not_all_new_releases_implies_exists_not_new_and_not_all_new
  (h : ¬(∀ b ∈ library, is_new_release b)) :
  (∃ b ∈ library, ¬(is_new_release b)) ∧ ¬(∀ b ∈ library, is_new_release b) := by
sorry

end NUMINAMATH_CALUDE_not_all_new_releases_implies_exists_not_new_and_not_all_new_l2677_267701


namespace NUMINAMATH_CALUDE_simplify_expression_l2677_267797

theorem simplify_expression (x : ℝ) : (x + 1)^2 + x*(x - 2) = 2*x^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2677_267797


namespace NUMINAMATH_CALUDE_perpendicular_lines_l2677_267736

theorem perpendicular_lines (b : ℝ) : 
  let v1 : Fin 2 → ℝ := ![4, -9]
  let v2 : Fin 2 → ℝ := ![b, 3]
  (∀ i : Fin 2, v1 i * v2 i = 0) → b = 27/4 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_l2677_267736


namespace NUMINAMATH_CALUDE_series_sum_ln2_series_sum_1_minus_ln2_l2677_267785

/-- The sum of the series where the nth term is 1/((2n-1)(2n)) converges to ln 2 -/
theorem series_sum_ln2 : ∑' n, 1 / ((2 * n - 1) * (2 * n)) = Real.log 2 := by sorry

/-- The sum of the series where the nth term is 1/((2n)(2n+1)) converges to 1 - ln 2 -/
theorem series_sum_1_minus_ln2 : ∑' n, 1 / ((2 * n) * (2 * n + 1)) = 1 - Real.log 2 := by sorry

end NUMINAMATH_CALUDE_series_sum_ln2_series_sum_1_minus_ln2_l2677_267785


namespace NUMINAMATH_CALUDE_ice_cream_flavors_count_l2677_267793

/-- The number of ways to distribute n indistinguishable items into k distinguishable containers -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of flavors that can be created by combining 4 scoops of 3 basic flavors -/
def ice_cream_flavors : ℕ := distribute 4 3

theorem ice_cream_flavors_count : ice_cream_flavors = 15 := by sorry

end NUMINAMATH_CALUDE_ice_cream_flavors_count_l2677_267793


namespace NUMINAMATH_CALUDE_no_solution_to_diophantine_equation_l2677_267798

theorem no_solution_to_diophantine_equation :
  ∀ (x y z t : ℕ), 3 * x^4 + 5 * y^4 + 7 * z^4 ≠ 11 * t^4 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_to_diophantine_equation_l2677_267798


namespace NUMINAMATH_CALUDE_distribute_seven_balls_two_boxes_l2677_267784

/-- The number of ways to distribute distinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  num_boxes ^ num_balls

/-- Theorem: The number of ways to distribute 7 distinguishable balls into 2 distinguishable boxes is 128 -/
theorem distribute_seven_balls_two_boxes : 
  distribute_balls 7 2 = 128 := by
  sorry

end NUMINAMATH_CALUDE_distribute_seven_balls_two_boxes_l2677_267784


namespace NUMINAMATH_CALUDE_linear_equation_condition_l2677_267777

theorem linear_equation_condition (a : ℝ) : 
  (∀ x, ∃ k m, (a - 1) * x^(|a|) + 4 = k * x + m) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_condition_l2677_267777


namespace NUMINAMATH_CALUDE_ratio_problem_l2677_267763

/-- Given two numbers with a 20:1 ratio where the first number is 200, 
    the second number is 10. -/
theorem ratio_problem (a b : ℝ) : 
  (a / b = 20) → (a = 200) → (b = 10) := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l2677_267763


namespace NUMINAMATH_CALUDE_supermarket_spend_correct_l2677_267774

def supermarket_spend (initial_amount left_amount showroom_spend : ℕ) : ℕ :=
  initial_amount - left_amount - showroom_spend

theorem supermarket_spend_correct (initial_amount left_amount showroom_spend : ℕ) 
  (h1 : initial_amount ≥ left_amount + showroom_spend) :
  supermarket_spend initial_amount left_amount showroom_spend = 
    initial_amount - left_amount - showroom_spend :=
by
  sorry

#eval supermarket_spend 106 26 49

end NUMINAMATH_CALUDE_supermarket_spend_correct_l2677_267774


namespace NUMINAMATH_CALUDE_apple_picking_theorem_l2677_267765

/-- The number of apples Lexie picked -/
def lexie_apples : ℕ := 12

/-- Tom picked twice as many apples as Lexie -/
def tom_apples : ℕ := 2 * lexie_apples

/-- The total number of apples collected -/
def total_apples : ℕ := lexie_apples + tom_apples

theorem apple_picking_theorem : total_apples = 36 := by
  sorry

end NUMINAMATH_CALUDE_apple_picking_theorem_l2677_267765


namespace NUMINAMATH_CALUDE_function_passes_through_point_l2677_267741

theorem function_passes_through_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x - 1) + 1
  f 1 = 2 := by sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l2677_267741


namespace NUMINAMATH_CALUDE_probability_at_least_two_black_l2677_267761

/-- The number of white balls in the bag -/
def white_balls : ℕ := 5

/-- The number of black balls in the bag -/
def black_balls : ℕ := 3

/-- The total number of balls in the bag -/
def total_balls : ℕ := white_balls + black_balls

/-- The number of balls drawn from the bag -/
def drawn_balls : ℕ := 3

/-- The probability of drawing at least 2 black balls when drawing 3 balls from a bag 
    containing 5 white balls and 3 black balls -/
theorem probability_at_least_two_black : 
  (Nat.choose black_balls 2 * Nat.choose white_balls 1 + Nat.choose black_balls 3) / 
  Nat.choose total_balls drawn_balls = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_two_black_l2677_267761


namespace NUMINAMATH_CALUDE_correct_calculation_l2677_267789

theorem correct_calculation (a b : ℝ) : 3 * a * b + 2 * a * b = 5 * a * b := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2677_267789


namespace NUMINAMATH_CALUDE_range_of_a_for_decreasing_f_l2677_267747

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (a - 2) * x + 3 else 2 * a / x

-- State the theorem
theorem range_of_a_for_decreasing_f :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x > f a y) → 0 < a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_for_decreasing_f_l2677_267747


namespace NUMINAMATH_CALUDE_coefficient_of_x_squared_l2677_267754

theorem coefficient_of_x_squared (z : ℂ) (a₀ a₁ a₂ a₃ a₄ : ℂ) :
  z = 1 + I →
  (∀ x : ℂ, (x + z)^4 = a₀*x^4 + a₁*x^3 + a₂*x^2 + a₃*x + a₄) →
  a₂ = 12*I :=
by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_squared_l2677_267754


namespace NUMINAMATH_CALUDE_flowchart_output_l2677_267781

def iterate_add_two (x : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n + 1 => iterate_add_two (x + 2) n

theorem flowchart_output : iterate_add_two 10 3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_flowchart_output_l2677_267781


namespace NUMINAMATH_CALUDE_probability_two_red_balls_l2677_267750

/-- The probability of picking 2 red balls from a bag containing 3 red balls, 4 blue balls, and 4 green balls. -/
theorem probability_two_red_balls (total_balls : ℕ) (red_balls : ℕ) (blue_balls : ℕ) (green_balls : ℕ) : 
  total_balls = red_balls + blue_balls + green_balls →
  red_balls = 3 →
  blue_balls = 4 →
  green_balls = 4 →
  (red_balls.choose 2 : ℚ) / (total_balls.choose 2) = 3 / 55 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_red_balls_l2677_267750


namespace NUMINAMATH_CALUDE_arcade_tickets_difference_l2677_267778

theorem arcade_tickets_difference (tickets_won tickets_left : ℕ) 
  (h1 : tickets_won = 48) 
  (h2 : tickets_left = 32) : 
  tickets_won - tickets_left = 16 := by
  sorry

end NUMINAMATH_CALUDE_arcade_tickets_difference_l2677_267778


namespace NUMINAMATH_CALUDE_digit_product_existence_l2677_267752

/-- A single-digit integer is a natural number from 1 to 9. -/
def SingleDigit : Type := { n : ℕ // 1 ≤ n ∧ n ≤ 9 }

/-- The product of a list of single-digit integers -/
def product_of_digits (digits : List SingleDigit) : ℕ :=
  digits.foldl (fun acc d => acc * d.val) 1

/-- Theorem stating the existence or non-existence of integers with specific digit products -/
theorem digit_product_existence :
  (¬ ∃ (digits : List SingleDigit), product_of_digits digits = 1980) ∧
  (¬ ∃ (digits : List SingleDigit), product_of_digits digits = 1990) ∧
  (∃ (digits : List SingleDigit), product_of_digits digits = 2000) :=
sorry

end NUMINAMATH_CALUDE_digit_product_existence_l2677_267752


namespace NUMINAMATH_CALUDE_sum_of_digits_greatest_prime_divisor_16385_l2677_267757

def greatest_prime_divisor (n : ℕ) : ℕ := sorry

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_greatest_prime_divisor_16385 :
  sum_of_digits (greatest_prime_divisor 16385) = 19 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_greatest_prime_divisor_16385_l2677_267757


namespace NUMINAMATH_CALUDE_loss_fraction_l2677_267786

theorem loss_fraction (cost_price selling_price : ℚ) 
  (h1 : cost_price = 21)
  (h2 : selling_price = 20) :
  (cost_price - selling_price) / cost_price = 1 / 21 := by
  sorry

end NUMINAMATH_CALUDE_loss_fraction_l2677_267786


namespace NUMINAMATH_CALUDE_brothers_age_in_6_years_l2677_267771

/-- The combined age of 4 brothers in a given number of years from now -/
def combined_age (years_from_now : ℕ) : ℕ :=
  sorry

theorem brothers_age_in_6_years :
  combined_age 15 = 107 → combined_age 6 = 71 :=
by sorry

end NUMINAMATH_CALUDE_brothers_age_in_6_years_l2677_267771


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2677_267783

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 + 3*x - 4 < 0} = Set.Ioo (-4 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2677_267783


namespace NUMINAMATH_CALUDE_jane_apple_purchase_l2677_267742

/-- The price of one apple in dollars -/
def apple_price : ℝ := 2

/-- The amount Jane has to spend in dollars -/
def jane_budget : ℝ := 2

/-- There is no bulk discount -/
axiom no_bulk_discount : ∀ (n : ℕ), n * apple_price = jane_budget → n = 1

/-- The number of apples Jane can buy with her budget -/
def apples_bought : ℕ := 1

theorem jane_apple_purchase :
  apples_bought * apple_price = jane_budget :=
sorry

end NUMINAMATH_CALUDE_jane_apple_purchase_l2677_267742


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l2677_267716

/-- The common ratio of the infinite geometric series 8/10 - 12/25 + 36/125 - ... is -3/5 -/
theorem geometric_series_common_ratio :
  let a₁ : ℚ := 8 / 10
  let a₂ : ℚ := -12 / 25
  let a₃ : ℚ := 36 / 125
  ∃ r : ℚ, r = a₂ / a₁ ∧ r = a₃ / a₂ ∧ r = -3 / 5 :=
by sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l2677_267716


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2677_267729

theorem quadratic_equation_roots (a m : ℤ) : 
  (∃ x : ℤ, (a - 1) * x^2 + a * x + 1 = 0 ∧ (m^2 + m) * x^2 + 3 * m * x - 3 = 0) →
  (a = -2 ∧ (m = -1 ∨ m = 3)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2677_267729


namespace NUMINAMATH_CALUDE_inequality_proof_l2677_267744

theorem inequality_proof (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq_3 : a + b + c + d = 3) :
  1 / a^3 + 1 / b^3 + 1 / c^3 + 1 / d^3 ≤ 1 / (a^3 * b^3 * c^3 * d^3) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2677_267744


namespace NUMINAMATH_CALUDE_fraction_value_l2677_267728

theorem fraction_value (a b : ℝ) (h : Real.sqrt (a + 2) + |b - 3| = 0) : a / b = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l2677_267728


namespace NUMINAMATH_CALUDE_man_son_age_difference_l2677_267767

/-- Represents the age difference between a man and his son -/
def AgeDifference (sonAge manAge : ℕ) : ℕ := manAge - sonAge

theorem man_son_age_difference :
  ∀ (sonAge manAge : ℕ),
  sonAge = 22 →
  manAge + 2 = 2 * (sonAge + 2) →
  AgeDifference sonAge manAge = 24 := by
  sorry

end NUMINAMATH_CALUDE_man_son_age_difference_l2677_267767


namespace NUMINAMATH_CALUDE_merchant_markup_l2677_267700

theorem merchant_markup (C : ℝ) (x : ℝ) : 
  (C * (1 + x / 100) * 0.7 = C * 1.225) → x = 75 := by
  sorry

end NUMINAMATH_CALUDE_merchant_markup_l2677_267700


namespace NUMINAMATH_CALUDE_ball_hit_ground_time_l2677_267730

/-- The time when a ball hits the ground, given its height equation -/
theorem ball_hit_ground_time (t : ℝ) : t ≥ 0 → -8*t^2 - 12*t + 72 = 0 → t = 3 := by
  sorry

#check ball_hit_ground_time

end NUMINAMATH_CALUDE_ball_hit_ground_time_l2677_267730


namespace NUMINAMATH_CALUDE_find_x_l2677_267706

theorem find_x : ∃ x : ℕ, 
  (∃ k : ℕ, x = 8 * k) ∧ 
  x^2 > 100 ∧ 
  x < 20 ∧ 
  x = 16 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l2677_267706


namespace NUMINAMATH_CALUDE_lcm_24_36_40_l2677_267727

theorem lcm_24_36_40 : Nat.lcm (Nat.lcm 24 36) 40 = 360 := by
  sorry

end NUMINAMATH_CALUDE_lcm_24_36_40_l2677_267727


namespace NUMINAMATH_CALUDE_adams_shelves_capacity_l2677_267708

/-- The number of action figures that can be held on Adam's shelves -/
def total_action_figures (figures_per_shelf : ℕ) (num_shelves : ℕ) : ℕ :=
  figures_per_shelf * num_shelves

/-- Theorem stating that the total number of action figures on Adam's shelves is 44 -/
theorem adams_shelves_capacity :
  total_action_figures 11 4 = 44 := by
  sorry

end NUMINAMATH_CALUDE_adams_shelves_capacity_l2677_267708


namespace NUMINAMATH_CALUDE_seth_purchase_difference_l2677_267703

/-- Calculates the difference in cost between discounted ice cream and yogurt purchases. -/
def ice_cream_yogurt_cost_difference (
  ice_cream_cartons : ℕ)
  (yogurt_cartons : ℕ)
  (ice_cream_price : ℚ)
  (yogurt_price : ℚ)
  (ice_cream_discount : ℚ)
  (yogurt_discount : ℚ) : ℚ :=
  let ice_cream_cost := ice_cream_cartons * ice_cream_price * (1 - ice_cream_discount)
  let yogurt_cost := yogurt_cartons * yogurt_price * (1 - yogurt_discount)
  ice_cream_cost - yogurt_cost

theorem seth_purchase_difference :
  ice_cream_yogurt_cost_difference 20 2 6 1 (1/10) (1/5) = 1064/10 := by
  sorry

end NUMINAMATH_CALUDE_seth_purchase_difference_l2677_267703


namespace NUMINAMATH_CALUDE_equality_of_negative_powers_l2677_267740

theorem equality_of_negative_powers : -(-1)^99 = (-1)^100 := by
  sorry

end NUMINAMATH_CALUDE_equality_of_negative_powers_l2677_267740


namespace NUMINAMATH_CALUDE_exponential_graph_quadrants_l2677_267732

theorem exponential_graph_quadrants (a b : ℝ) (ha : 0 < a) (ha' : a < 1) (hb : b < -1) :
  ∀ x y : ℝ, y = a^x + b → ¬(x > 0 ∧ y > 0) :=
by sorry

end NUMINAMATH_CALUDE_exponential_graph_quadrants_l2677_267732


namespace NUMINAMATH_CALUDE_product_inspection_probabilities_l2677_267792

def total_units : ℕ := 6
def inspected_units : ℕ := 2
def first_grade_units : ℕ := 3
def second_grade_units : ℕ := 2
def defective_units : ℕ := 1

def probability_both_first_grade : ℚ := 1 / 5
def probability_one_second_grade : ℚ := 8 / 15

def probability_at_most_one_defective (x : ℕ) : ℚ :=
  (Nat.choose x 1 * Nat.choose (total_units - x) 1 + Nat.choose (total_units - x) 2) /
  Nat.choose total_units inspected_units

theorem product_inspection_probabilities :
  (probability_both_first_grade = 1 / 5) ∧
  (probability_one_second_grade = 8 / 15) ∧
  (∀ x : ℕ, x ≤ total_units →
    (probability_at_most_one_defective x ≥ 4 / 5 → x ≤ 3)) :=
by sorry

end NUMINAMATH_CALUDE_product_inspection_probabilities_l2677_267792


namespace NUMINAMATH_CALUDE_four_day_viewing_time_l2677_267787

/-- Calculates the total hours watched given the number of days and minutes per episode -/
def totalHoursWatched (daysWatched : ℕ) (minutesPerEpisode : ℕ) : ℚ :=
  (daysWatched * minutesPerEpisode : ℚ) / 60

/-- Theorem: Watching a 30-minute show for 4 days results in 2 hours of total viewing time -/
theorem four_day_viewing_time :
  totalHoursWatched 4 30 = 2 := by
  sorry

#eval totalHoursWatched 4 30

end NUMINAMATH_CALUDE_four_day_viewing_time_l2677_267787


namespace NUMINAMATH_CALUDE_eugene_payment_l2677_267715

/-- The cost of a single T-shirt in dollars -/
def tshirt_cost : ℚ := 20

/-- The cost of a single pair of pants in dollars -/
def pants_cost : ℚ := 80

/-- The cost of a single pair of shoes in dollars -/
def shoes_cost : ℚ := 150

/-- The discount rate as a decimal -/
def discount_rate : ℚ := 0.1

/-- The number of T-shirts Eugene buys -/
def num_tshirts : ℕ := 4

/-- The number of pairs of pants Eugene buys -/
def num_pants : ℕ := 3

/-- The number of pairs of shoes Eugene buys -/
def num_shoes : ℕ := 2

/-- The total cost before discount -/
def total_cost_before_discount : ℚ :=
  tshirt_cost * num_tshirts + pants_cost * num_pants + shoes_cost * num_shoes

/-- The amount Eugene has to pay after the discount -/
def amount_to_pay : ℚ := total_cost_before_discount * (1 - discount_rate)

theorem eugene_payment :
  amount_to_pay = 558 := by sorry

end NUMINAMATH_CALUDE_eugene_payment_l2677_267715


namespace NUMINAMATH_CALUDE_trip_time_calculation_l2677_267731

/-- Proves that if a trip takes 4.5 hours at 70 mph, it will take 5.25 hours at 60 mph -/
theorem trip_time_calculation (distance : ℝ) : 
  distance = 70 * 4.5 → distance = 60 * 5.25 := by
  sorry

end NUMINAMATH_CALUDE_trip_time_calculation_l2677_267731


namespace NUMINAMATH_CALUDE_license_plate_theorem_l2677_267764

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of letter positions in the license plate -/
def letter_positions : ℕ := 4

/-- The number of digit positions in the license plate -/
def digit_positions : ℕ := 3

/-- The number of possible digits (0-9) -/
def digit_options : ℕ := 10

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- Calculates the number of license plate combinations -/
def license_plate_combinations : ℕ :=
  alphabet_size *
  (choose (alphabet_size - 1) 2) *
  (choose letter_positions 2) *
  (digit_options * (digit_options - 1) * (digit_options - 2))

theorem license_plate_theorem :
  license_plate_combinations = 33696000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_theorem_l2677_267764


namespace NUMINAMATH_CALUDE_two_over_x_values_l2677_267719

theorem two_over_x_values (x : ℝ) (hx : 3 - 9/x + 6/x^2 = 0) :
  2/x = 1 ∨ 2/x = 2 :=
by sorry

end NUMINAMATH_CALUDE_two_over_x_values_l2677_267719


namespace NUMINAMATH_CALUDE_B_power_15_minus_3_power_14_l2677_267770

def B : Matrix (Fin 2) (Fin 2) ℝ := !![4, 1; 0, 3]

theorem B_power_15_minus_3_power_14 : 
  B^15 - 3 • B^14 = !![4^14, 4^14; 0, 0] := by sorry

end NUMINAMATH_CALUDE_B_power_15_minus_3_power_14_l2677_267770


namespace NUMINAMATH_CALUDE_grade_assignment_count_l2677_267711

/-- The number of possible grades a professor can assign to each student. -/
def num_grades : ℕ := 4

/-- The number of students in the class. -/
def num_students : ℕ := 12

/-- The number of ways to assign grades to all students. -/
def num_ways : ℕ := num_grades ^ num_students

/-- Theorem stating that the number of ways to assign grades is 16,777,216. -/
theorem grade_assignment_count : num_ways = 16777216 := by
  sorry

end NUMINAMATH_CALUDE_grade_assignment_count_l2677_267711


namespace NUMINAMATH_CALUDE_cost_price_per_meter_l2677_267772

/-- 
Given a trader who sells cloth with the following conditions:
- total_meters: The total number of meters of cloth sold
- selling_price: The total selling price for all meters of cloth
- profit_per_meter: The profit made per meter of cloth

This theorem proves that the cost price per meter of cloth is equal to
(selling_price - (total_meters * profit_per_meter)) / total_meters
-/
theorem cost_price_per_meter 
  (total_meters : ℕ) 
  (selling_price profit_per_meter : ℚ) 
  (h1 : total_meters = 85)
  (h2 : selling_price = 8925)
  (h3 : profit_per_meter = 5) :
  (selling_price - (total_meters : ℚ) * profit_per_meter) / total_meters = 100 :=
by sorry

end NUMINAMATH_CALUDE_cost_price_per_meter_l2677_267772


namespace NUMINAMATH_CALUDE_paper_crane_folding_time_l2677_267758

theorem paper_crane_folding_time (time_A time_B : ℝ) (h1 : time_A = 30) (h2 : time_B = 45) :
  (1 / time_A + 1 / time_B)⁻¹ = 18 := by sorry

end NUMINAMATH_CALUDE_paper_crane_folding_time_l2677_267758


namespace NUMINAMATH_CALUDE_square_sum_geq_root_three_l2677_267739

theorem square_sum_geq_root_three (a b c : ℝ) 
  (h : a^2 * b * c + a * b^2 * c + a * b * c^2 = 1) : 
  a^2 + b^2 + c^2 ≥ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_geq_root_three_l2677_267739


namespace NUMINAMATH_CALUDE_line_direction_vector_l2677_267756

/-- Given a line y = (5x - 7) / 2 parameterized as (x, y) = v + t * d,
    where the distance between (x, y) and (4, 2) is t for x ≥ 4,
    prove that the direction vector d is (2/√29, 5/√29). -/
theorem line_direction_vector (v d : ℝ × ℝ) :
  (∀ x y t : ℝ, x ≥ 4 →
    y = (5 * x - 7) / 2 →
    (x, y) = v + t • d →
    ‖(x, y) - (4, 2)‖ = t) →
  d = (2 / Real.sqrt 29, 5 / Real.sqrt 29) :=
by sorry

end NUMINAMATH_CALUDE_line_direction_vector_l2677_267756


namespace NUMINAMATH_CALUDE_expand_expression_l2677_267790

theorem expand_expression (x : ℝ) : -2 * (x + 3) * (x - 2) * (x + 1) = -2*x^3 - 4*x^2 + 10*x + 12 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2677_267790


namespace NUMINAMATH_CALUDE_jills_sales_goal_l2677_267721

/-- Represents the number of boxes sold to each customer --/
structure CustomerSales where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ
  fifth : ℕ

/-- Calculates Jill's sales goal based on customer purchases and boxes left to sell --/
def salesGoal (sales : CustomerSales) (boxesLeft : ℕ) : ℕ :=
  sales.first + sales.second + sales.third + sales.fourth + sales.fifth + boxesLeft

/-- Theorem stating Jill's sales goal --/
theorem jills_sales_goal :
  ∀ (sales : CustomerSales) (boxesLeft : ℕ),
    sales.first = 5 →
    sales.second = 4 * sales.first →
    sales.third = sales.second / 2 →
    sales.fourth = 3 * sales.third →
    sales.fifth = 10 →
    boxesLeft = 75 →
    salesGoal sales boxesLeft = 150 := by
  sorry

end NUMINAMATH_CALUDE_jills_sales_goal_l2677_267721
