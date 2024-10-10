import Mathlib

namespace sum_of_divisors_2i3j_l1293_129315

/-- Sum of divisors function -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- Theorem: If the sum of divisors of 2^i * 3^j is 960, then i + j = 5 -/
theorem sum_of_divisors_2i3j (i j : ℕ) :
  sum_of_divisors (2^i * 3^j) = 960 → i + j = 5 := by sorry

end sum_of_divisors_2i3j_l1293_129315


namespace police_officers_on_duty_l1293_129396

theorem police_officers_on_duty 
  (total_female_officers : ℕ)
  (female_duty_percentage : ℚ)
  (female_duty_ratio : ℚ)
  (h1 : total_female_officers = 600)
  (h2 : female_duty_percentage = 17 / 100)
  (h3 : female_duty_ratio = 1 / 2) :
  ∃ (officers_on_duty : ℕ), 
    officers_on_duty = 204 ∧ 
    (officers_on_duty : ℚ) * female_duty_ratio = (total_female_officers : ℚ) * female_duty_percentage :=
by
  sorry

end police_officers_on_duty_l1293_129396


namespace swimmer_speed_is_4_l1293_129363

/-- The swimmer's speed in still water -/
def swimmer_speed : ℝ := 4

/-- The speed of the water current -/
def current_speed : ℝ := 1

/-- The time taken to swim against the current -/
def swim_time : ℝ := 2

/-- The distance swum against the current -/
def swim_distance : ℝ := 6

/-- Theorem stating that the swimmer's speed in still water is 4 km/h -/
theorem swimmer_speed_is_4 :
  swimmer_speed = 4 ∧
  current_speed = 1 ∧
  swim_time = 2 ∧
  swim_distance = 6 →
  swimmer_speed = swim_distance / swim_time + current_speed :=
by sorry

end swimmer_speed_is_4_l1293_129363


namespace tangent_ellipse_hyperbola_parameter_l1293_129311

/-- Given an ellipse and a hyperbola that are tangent, prove that the parameter m of the hyperbola is 5/9 -/
theorem tangent_ellipse_hyperbola_parameter (x y m : ℝ) : 
  (∃ x y, x^2 + 9*y^2 = 9 ∧ x^2 - m*(y+3)^2 = 4) →  -- Existence of points satisfying both equations
  (∀ x y, x^2 + 9*y^2 = 9 → x^2 - m*(y+3)^2 ≥ 4) →  -- Hyperbola does not intersect interior of ellipse
  (∃ x y, x^2 + 9*y^2 = 9 ∧ x^2 - m*(y+3)^2 = 4) →  -- Existence of a common point
  m = 5/9 := by sorry

end tangent_ellipse_hyperbola_parameter_l1293_129311


namespace books_read_difference_result_l1293_129371

/-- The number of books Peter has read more than his brother and Sarah combined -/
def books_read_difference (total_books : ℕ) (peter_percent : ℚ) (brother_percent : ℚ) (sarah_percent : ℚ) : ℚ :=
  (peter_percent * total_books) - ((brother_percent + sarah_percent) * total_books)

/-- Theorem stating the difference in books read -/
theorem books_read_difference_result :
  books_read_difference 50 (60 / 100) (25 / 100) (15 / 100) = 10 := by
  sorry

end books_read_difference_result_l1293_129371


namespace product_of_ratios_l1293_129317

theorem product_of_ratios (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 2005) (h₂ : y₁^3 - 3*x₁^2*y₁ = 2004)
  (h₃ : x₂^3 - 3*x₂*y₂^2 = 2005) (h₄ : y₂^3 - 3*x₂^2*y₂ = 2004)
  (h₅ : x₃^3 - 3*x₃*y₃^2 = 2005) (h₆ : y₃^3 - 3*x₃^2*y₃ = 2004)
  (h₇ : y₁ ≠ 0) (h₈ : y₂ ≠ 0) (h₉ : y₃ ≠ 0) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = 1/1002 := by
sorry

end product_of_ratios_l1293_129317


namespace right_triangle_perimeter_l1293_129333

theorem right_triangle_perimeter (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a * b / 2 = 180 →
  a = 30 →
  c^2 = a^2 + b^2 →
  a + b + c = 42 + 2 * Real.sqrt 261 :=
by sorry

end right_triangle_perimeter_l1293_129333


namespace line_slope_l1293_129328

/-- Given a line passing through points (1,2) and (4,2+√3), its slope is √3/3 -/
theorem line_slope : ∃ (k : ℝ), k = (2 + Real.sqrt 3 - 2) / (4 - 1) ∧ k = Real.sqrt 3 / 3 := by
  sorry

end line_slope_l1293_129328


namespace divisible_by_fifteen_l1293_129305

theorem divisible_by_fifteen (n : ℤ) : 15 ∣ (7*n + 5*n^3 + 3*n^5) := by
  sorry

end divisible_by_fifteen_l1293_129305


namespace equation_solution_l1293_129392

theorem equation_solution :
  ∀ y : ℝ, (45 : ℝ) / 75 = Real.sqrt (y / 25) → y = 9 := by
  sorry

end equation_solution_l1293_129392


namespace sqrt_nine_equals_three_l1293_129327

theorem sqrt_nine_equals_three : Real.sqrt 9 = 3 := by
  sorry

end sqrt_nine_equals_three_l1293_129327


namespace specific_lot_volume_l1293_129381

/-- The volume of a rectangular lot -/
def lot_volume (length width height : ℝ) : ℝ := length * width * height

/-- Theorem stating that the volume of the specific lot is 1600 cubic meters -/
theorem specific_lot_volume : lot_volume 40 20 2 = 1600 := by
  sorry

end specific_lot_volume_l1293_129381


namespace solve_euro_equation_l1293_129386

-- Define the operation €
def euro (x y : ℝ) : ℝ := 2 * x * y

-- Theorem statement
theorem solve_euro_equation : 
  ∀ x : ℝ, euro x (euro 4 5) = 720 → x = 9 := by
  sorry

end solve_euro_equation_l1293_129386


namespace polar_to_rectangular_l1293_129346

theorem polar_to_rectangular (r θ : ℝ) (h : r = 7 ∧ θ = π/3) :
  (r * Real.cos θ, r * Real.sin θ) = (3.5, 7 * Real.sqrt 3 / 2) := by
  sorry

end polar_to_rectangular_l1293_129346


namespace least_common_multiple_first_ten_l1293_129307

theorem least_common_multiple_first_ten : ∃ n : ℕ, n > 0 ∧ (∀ i : ℕ, i > 0 ∧ i ≤ 10 → i ∣ n) ∧ (∀ m : ℕ, m > 0 ∧ (∀ i : ℕ, i > 0 ∧ i ≤ 10 → i ∣ m) → n ≤ m) ∧ n = 2520 := by
  sorry

end least_common_multiple_first_ten_l1293_129307


namespace condition_relationship_l1293_129343

theorem condition_relationship :
  (∀ x : ℝ, (x - 1) / (x + 2) ≥ 0 → (x - 1) * (x + 2) ≥ 0) ∧
  (∃ x : ℝ, (x - 1) * (x + 2) ≥ 0 ∧ ¬((x - 1) / (x + 2) ≥ 0)) :=
by sorry

end condition_relationship_l1293_129343


namespace triangle_max_side_length_l1293_129318

theorem triangle_max_side_length (P Q R : Real) (a b : Real) :
  -- Triangle angles
  P + Q + R = Real.pi →
  -- Given condition
  Real.cos (3 * P) + Real.cos (3 * Q) + Real.cos (3 * R) = 1 →
  -- Two sides have lengths 12 and 15
  a = 12 ∧ b = 15 →
  -- Maximum length of the third side
  ∃ c : Real, c ≤ 27 ∧ 
    ∀ c' : Real, (c' ^ 2 ≤ a ^ 2 + b ^ 2 - 2 * a * b * Real.cos R) → c' ≤ c :=
by sorry

end triangle_max_side_length_l1293_129318


namespace smallest_constant_term_l1293_129303

theorem smallest_constant_term (a b c d e : ℤ) :
  (∀ x : ℚ, a * x^4 + b * x^3 + c * x^2 + d * x + e = 0 ↔ 
    x = 1 ∨ x = -3 ∨ x = 7 ∨ x = -2/5) →
  e > 0 →
  (∀ e' : ℤ, e' > 0 → 
    (∃ a' b' c' d' : ℤ, ∀ x : ℚ, a' * x^4 + b' * x^3 + c' * x^2 + d' * x + e' = 0 ↔ 
      x = 1 ∨ x = -3 ∨ x = 7 ∨ x = -2/5) →
    e ≤ e') →
  e = 42 := by
sorry

end smallest_constant_term_l1293_129303


namespace sqrt_5x_plus_y_squared_l1293_129322

theorem sqrt_5x_plus_y_squared (x y : ℝ) 
  (h : Real.sqrt (x - 1) + (3 * x + y - 1)^2 = 0) : 
  Real.sqrt (5 * x + y^2) = 3 := by
  sorry

end sqrt_5x_plus_y_squared_l1293_129322


namespace maggies_total_earnings_l1293_129355

/-- Calculates Maggie's earnings from selling magazine subscriptions -/
def maggies_earnings (price_per_subscription : ℕ) 
  (parents_subscriptions : ℕ) 
  (grandfather_subscriptions : ℕ) 
  (nextdoor_subscriptions : ℕ) : ℕ :=
  let total_subscriptions := 
    parents_subscriptions + 
    grandfather_subscriptions + 
    nextdoor_subscriptions + 
    (2 * nextdoor_subscriptions)
  price_per_subscription * total_subscriptions

theorem maggies_total_earnings : 
  maggies_earnings 5 4 1 2 = 55 := by
  sorry

end maggies_total_earnings_l1293_129355


namespace negative_a_exponent_division_l1293_129387

theorem negative_a_exponent_division (a : ℝ) : (-a)^6 / (-a)^3 = -a^3 := by sorry

end negative_a_exponent_division_l1293_129387


namespace box_volume_less_than_500_l1293_129362

def box_volume (x : ℕ) : ℕ := (x + 3) * (x - 3) * (x^2 + 9)

theorem box_volume_less_than_500 :
  ∀ x : ℕ, x > 0 → (box_volume x < 500 ↔ x = 4 ∨ x = 5) :=
by sorry

end box_volume_less_than_500_l1293_129362


namespace dodecahedron_interior_diagonals_l1293_129302

/-- Represents a dodecahedron -/
structure Dodecahedron where
  /-- The number of faces in a dodecahedron -/
  faces : ℕ
  /-- The number of vertices in a dodecahedron -/
  vertices : ℕ
  /-- The number of faces meeting at each vertex -/
  faces_per_vertex : ℕ
  /-- Assertion that the dodecahedron has 12 faces -/
  faces_eq : faces = 12
  /-- Assertion that the dodecahedron has 20 vertices -/
  vertices_eq : vertices = 20
  /-- Assertion that 3 faces meet at each vertex -/
  faces_per_vertex_eq : faces_per_vertex = 3

/-- Calculates the number of interior diagonals in a dodecahedron -/
def interior_diagonals (d : Dodecahedron) : ℕ :=
  (d.vertices * (d.vertices - d.faces_per_vertex - 1)) / 2

/-- Theorem stating that a dodecahedron has 160 interior diagonals -/
theorem dodecahedron_interior_diagonals (d : Dodecahedron) : 
  interior_diagonals d = 160 := by
  sorry

end dodecahedron_interior_diagonals_l1293_129302


namespace labor_market_effects_l1293_129385

-- Define the labor market for doctors
structure LaborMarket where
  supply : ℝ → ℝ  -- Supply function
  demand : ℝ → ℝ  -- Demand function
  equilibriumWage : ℝ  -- Equilibrium wage

-- Define the commercial healthcare market
structure HealthcareMarket where
  supply : ℝ → ℝ  -- Supply function
  demand : ℝ → ℝ  -- Demand function
  equilibriumPrice : ℝ  -- Equilibrium price

-- Define the government policy
def governmentPolicy (minYears : ℕ) : Prop :=
  ∃ (requirement : ℕ), requirement ≥ minYears

-- Theorem statement
theorem labor_market_effects
  (initialMarket : LaborMarket)
  (initialHealthcare : HealthcareMarket)
  (policy : governmentPolicy 1)
  (newMarket : LaborMarket)
  (newHealthcare : HealthcareMarket) :
  (newMarket.equilibriumWage > initialMarket.equilibriumWage) ∧
  (newHealthcare.equilibriumPrice < initialHealthcare.equilibriumPrice) :=
sorry

end labor_market_effects_l1293_129385


namespace rhombus_area_l1293_129334

/-- The area of a rhombus with side length 13 and diagonals differing by 10 units is 208 square units. -/
theorem rhombus_area (s d₁ d₂ : ℝ) (h₁ : s = 13) (h₂ : d₂ - d₁ = 10) 
    (h₃ : s^2 = (d₁/2)^2 + (d₂/2)^2) : d₁ * d₂ / 2 = 208 := by
  sorry

end rhombus_area_l1293_129334


namespace quadratic_roots_property_l1293_129376

theorem quadratic_roots_property (d e : ℝ) : 
  (2 * d^2 + 3 * d - 5 = 0) → 
  (2 * e^2 + 3 * e - 5 = 0) → 
  (d - 1) * (e - 1) = 0 := by
sorry

end quadratic_roots_property_l1293_129376


namespace complex_exponent_calculation_l1293_129380

theorem complex_exponent_calculation : 3 * 3^6 - 9^60 / 9^58 + 4^3 = 2170 := by
  sorry

end complex_exponent_calculation_l1293_129380


namespace product_of_common_ratios_l1293_129345

/-- Given two nonconstant geometric sequences with different common ratios
    satisfying a specific equation, prove that the product of their common ratios is 9. -/
theorem product_of_common_ratios (x p r : ℝ) (hx : x ≠ 0) (hp : p ≠ 1) (hr : r ≠ 1) (hpr : p ≠ r) :
  3 * x * p^2 - 4 * x * r^2 = 5 * (3 * x * p - 4 * x * r) →
  p * r = 9 := by
sorry

end product_of_common_ratios_l1293_129345


namespace quadratic_equation_roots_l1293_129399

theorem quadratic_equation_roots (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   a * x₁^2 - (3*a + 1) * x₁ + 2*(a + 1) = 0 ∧
   a * x₂^2 - (3*a + 1) * x₂ + 2*(a + 1) = 0 ∧
   x₁ - x₁*x₂ + x₂ = 1 - a) →
  a = -1 := by
sorry

end quadratic_equation_roots_l1293_129399


namespace student_difference_l1293_129321

/-- Given that the sum of students in grades 1 and 2 is 30 more than the sum of students in grades 2 and 5,
    prove that the difference between the number of students in grade 1 and grade 5 is 30. -/
theorem student_difference (g1 g2 g5 : ℕ) (h : g1 + g2 = g2 + g5 + 30) : g1 - g5 = 30 := by
  sorry

end student_difference_l1293_129321


namespace probability_not_losing_l1293_129335

theorem probability_not_losing (p_draw p_win : ℝ) :
  p_draw = 1/2 →
  p_win = 1/3 →
  p_draw + p_win = 5/6 :=
by sorry

end probability_not_losing_l1293_129335


namespace inequality_proof_l1293_129332

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  a / (b * (1 + c)) + b / (c * (1 + a)) + c / (a * (1 + b)) ≥ 3 / 2 := by
  sorry

end inequality_proof_l1293_129332


namespace social_science_papers_selected_l1293_129378

/-- Proves the number of social science papers selected in stratified sampling -/
theorem social_science_papers_selected
  (total_papers : ℕ)
  (social_science_papers : ℕ)
  (selected_papers : ℕ)
  (h1 : total_papers = 153)
  (h2 : social_science_papers = 54)
  (h3 : selected_papers = 51)
  : (social_science_papers * selected_papers) / total_papers = 18 := by
  sorry

end social_science_papers_selected_l1293_129378


namespace sum_of_coefficients_of_expanded_f_l1293_129356

-- Define the polynomial expression
def f (c : ℝ) : ℝ := 2 * (c - 2) * (c^2 + c * (4 - c))

-- Define the sum of coefficients function
def sumOfCoefficients (p : ℝ → ℝ) : ℝ := sorry

-- Theorem statement
theorem sum_of_coefficients_of_expanded_f :
  sumOfCoefficients f = -8 := by sorry

end sum_of_coefficients_of_expanded_f_l1293_129356


namespace smallest_b_value_l1293_129338

theorem smallest_b_value (a b : ℕ) : 
  (∃ x y z : ℕ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    (2 * a - b = x^2) ∧ 
    (a - 2 * b = y^2) ∧ 
    (a + b = z^2)) →
  (∀ b' : ℕ, b' < b → 
    ¬(∃ a' x' y' z' : ℕ, x' ≠ y' ∧ y' ≠ z' ∧ x' ≠ z' ∧
      (2 * a' - b' = x'^2) ∧ 
      (a' - 2 * b' = y'^2) ∧ 
      (a' + b' = z'^2))) →
  b = 3 :=
sorry

end smallest_b_value_l1293_129338


namespace college_students_count_l1293_129339

/-- Calculates the total number of students in a college given the ratio of boys to girls and the number of girls -/
def total_students (boys_ratio : ℕ) (girls_ratio : ℕ) (num_girls : ℕ) : ℕ :=
  let num_boys := boys_ratio * num_girls / girls_ratio
  num_boys + num_girls

/-- Proves that in a college with a boys to girls ratio of 8:4 and 200 girls, the total number of students is 600 -/
theorem college_students_count : total_students 8 4 200 = 600 := by
  sorry

end college_students_count_l1293_129339


namespace percentage_increase_l1293_129397

theorem percentage_increase (x : ℝ) (base : ℝ) (percentage : ℝ) : 
  x = base + (percentage / 100) * base →
  x = 110 →
  base = 88 →
  percentage = 25 := by
sorry

end percentage_increase_l1293_129397


namespace sum_of_angles_equals_360_l1293_129326

-- Define the angles as real numbers
variable (A B C D F G : ℝ)

-- Define the property of being a quadrilateral
def is_quadrilateral (A B C D : ℝ) : Prop :=
  A + B + C + D = 360

-- State the theorem
theorem sum_of_angles_equals_360 
  (h : is_quadrilateral A B C D) : A + B + C + D + F + G = 360 := by
  sorry

end sum_of_angles_equals_360_l1293_129326


namespace exist_four_numbers_perfect_squares_l1293_129301

theorem exist_four_numbers_perfect_squares :
  ∃ (a b c d : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    ∃ (m n : ℕ), a^2 + 2*c*d + b^2 = m^2 ∧ c^2 + 2*a*b + d^2 = n^2 := by
  sorry

end exist_four_numbers_perfect_squares_l1293_129301


namespace perfect_square_trinomial_l1293_129382

theorem perfect_square_trinomial (k : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, x^2 - 20*x + k = (a*x + b)^2) ↔ k = 100 :=
sorry

end perfect_square_trinomial_l1293_129382


namespace four_circles_max_regions_l1293_129373

/-- The maximum number of regions that n circles can divide a plane into -/
def max_regions (n : ℕ) : ℕ :=
  n * (n - 1) + 2

/-- Assumption that for n = 1, 2, 3, n circles divide the plane into at most 2^n parts -/
axiom max_regions_small (n : ℕ) (h : n ≤ 3) : max_regions n ≤ 2^n

/-- Theorem: The maximum number of regions that four circles can divide a plane into is 14 -/
theorem four_circles_max_regions : max_regions 4 = 14 := by
  sorry

end four_circles_max_regions_l1293_129373


namespace equation_solution_l1293_129300

theorem equation_solution : ∃ (x₁ x₂ : ℝ), x₁ = -3 ∧ x₂ = 2/3 ∧
  (∀ x : ℝ, 3*x*(x+3) = 2*(x+3) ↔ x = x₁ ∨ x = x₂) := by
  sorry

end equation_solution_l1293_129300


namespace unique_square_multiple_of_five_in_range_l1293_129350

theorem unique_square_multiple_of_five_in_range : 
  ∃! x : ℕ, 
    (∃ n : ℕ, x = n^2) ∧ 
    (x % 5 = 0) ∧ 
    (50 < x) ∧ 
    (x < 120) :=
by
  sorry

end unique_square_multiple_of_five_in_range_l1293_129350


namespace evaluate_expression_l1293_129349

theorem evaluate_expression : 6 - 5 * (10 - (2 + 1)^2) * 3 = -9 := by
  sorry

end evaluate_expression_l1293_129349


namespace equation_solution_l1293_129344

theorem equation_solution (a b : ℝ) :
  (a + b - 1)^2 = a^2 + b^2 - 1 ↔ a = 1 ∨ b = 1 := by
  sorry

end equation_solution_l1293_129344


namespace geometric_series_sum_times_four_fifths_l1293_129390

theorem geometric_series_sum_times_four_fifths :
  let a : ℚ := 1/4
  let r : ℚ := 1/4
  let n : ℕ := 5
  let S := (a * (1 - r^n)) / (1 - r)
  (S * 4/5 : ℚ) = 21/80 := by sorry

end geometric_series_sum_times_four_fifths_l1293_129390


namespace peters_horse_food_l1293_129358

/-- Calculates the total food required for horses over a given number of days -/
def total_food_required (num_horses : ℕ) (oats_per_meal : ℕ) (oats_meals_per_day : ℕ) 
                        (grain_per_day : ℕ) (num_days : ℕ) : ℕ :=
  let total_oats := num_horses * oats_per_meal * oats_meals_per_day * num_days
  let total_grain := num_horses * grain_per_day * num_days
  total_oats + total_grain

/-- Theorem: Peter needs 132 pounds of food to feed his horses for 3 days -/
theorem peters_horse_food : total_food_required 4 4 2 3 3 = 132 := by
  sorry

end peters_horse_food_l1293_129358


namespace circus_performers_standing_time_l1293_129383

/-- The combined time that Pulsar, Polly, and Petra stand on their back legs is 45 minutes. -/
theorem circus_performers_standing_time : 
  let pulsar_time : ℕ := 10
  let polly_time : ℕ := 3 * pulsar_time
  let petra_time : ℕ := polly_time / 6
  pulsar_time + polly_time + petra_time = 45 :=
by sorry

end circus_performers_standing_time_l1293_129383


namespace polynomial_divisibility_l1293_129372

theorem polynomial_divisibility (p q : ℤ) : 
  (∀ x : ℝ, (x - 3) * (x + 1) ∣ (x^5 - 2*x^4 + 3*x^3 - p*x^2 + q*x - 12)) →
  p = -8 ∧ q = -10 := by
sorry

end polynomial_divisibility_l1293_129372


namespace remainder_171_pow_2147_mod_52_l1293_129394

theorem remainder_171_pow_2147_mod_52 : ∃ k : ℕ, 171^2147 = 52 * k + 7 := by sorry

end remainder_171_pow_2147_mod_52_l1293_129394


namespace intersection_A_complement_B_l1293_129320

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -3 < x ∧ x < 6}
def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 7}

-- Define the theorem
theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B) = {x : ℝ | -3 < x ∧ x ≤ 2} := by sorry

end intersection_A_complement_B_l1293_129320


namespace sin_cos_identity_l1293_129336

theorem sin_cos_identity (x : ℝ) : 
  (Real.sin (2 * x) + Real.sqrt 3 * Real.cos (2 * x))^2 = 2 - 2 * Real.cos ((2 / 3) * Real.pi - x) ↔ 
  (∃ n : ℤ, x = (2 * Real.pi / 5) * ↑n) ∨ 
  (∃ k : ℤ, x = (2 * Real.pi / 9) * (3 * ↑k + 1)) :=
sorry

end sin_cos_identity_l1293_129336


namespace closest_reps_20_eq_12_or_13_l1293_129351

def weight_25 : ℕ := 25
def weight_20 : ℕ := 20
def reps_25 : ℕ := 10

def total_weight : ℕ := 2 * weight_25 * reps_25

def closest_reps (w : ℕ) : Set ℕ :=
  {n : ℕ | n * 2 * w ≥ total_weight ∧ 
    ∀ m : ℕ, m * 2 * w ≥ total_weight → n ≤ m}

theorem closest_reps_20_eq_12_or_13 : 
  closest_reps weight_20 = {12, 13} :=
sorry

end closest_reps_20_eq_12_or_13_l1293_129351


namespace cone_base_circumference_l1293_129306

/-- The circumference of the base of a right circular cone with volume 24π cubic centimeters and height 6 cm is 4√3π cm. -/
theorem cone_base_circumference :
  ∀ (V h r : ℝ),
  V = 24 * Real.pi ∧
  h = 6 ∧
  V = (1/3) * Real.pi * r^2 * h →
  2 * Real.pi * r = 4 * Real.sqrt 3 * Real.pi :=
by sorry

end cone_base_circumference_l1293_129306


namespace equation_roots_l1293_129323

theorem equation_roots (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧
    (0.5 : ℝ)^(x^2 - m*x + 0.5*m - 1.5) = (Real.sqrt 8)^(m - 1) ∧
    (0.5 : ℝ)^(y^2 - m*y + 0.5*m - 1.5) = (Real.sqrt 8)^(m - 1))
  ↔ (m < 2 ∨ m > 6) :=
by sorry

end equation_roots_l1293_129323


namespace additional_grazing_area_l1293_129347

theorem additional_grazing_area (π : ℝ) (h : π > 0) : 
  π * 23^2 - π * 12^2 = 385 * π := by
  sorry

end additional_grazing_area_l1293_129347


namespace prob_three_two_digit_dice_l1293_129313

-- Define the number of dice
def num_dice : ℕ := 6

-- Define the number of sides on each die
def num_sides : ℕ := 12

-- Define the number of two-digit outcomes on a single die
def two_digit_outcomes : ℕ := 3

-- Define the probability of rolling a two-digit number on a single die
def prob_two_digit : ℚ := two_digit_outcomes / num_sides

-- Define the probability of rolling a one-digit number on a single die
def prob_one_digit : ℚ := 1 - prob_two_digit

-- Define the number of dice we want to show two-digit numbers
def target_two_digit : ℕ := 3

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := sorry

-- State the theorem
theorem prob_three_two_digit_dice :
  (binomial num_dice target_two_digit : ℚ) * prob_two_digit ^ target_two_digit * prob_one_digit ^ (num_dice - target_two_digit) = 135 / 1024 := by
  sorry

end prob_three_two_digit_dice_l1293_129313


namespace range_of_m_unbounded_below_m_characterization_of_m_range_l1293_129337

def A : Set ℝ := { x | -3 ≤ x ∧ x ≤ 4 }
def B (m : ℝ) : Set ℝ := { x | m + 1 ≤ x ∧ x ≤ 2*m - 1 }

theorem range_of_m (m : ℝ) : (A ∪ B m = A) → m ≤ 5/2 :=
by sorry

theorem unbounded_below_m : ∀ (k : ℝ), ∃ (m : ℝ), m < k ∧ (A ∪ B m = A) :=
by sorry

theorem characterization_of_m_range : 
  ∀ (m : ℝ), (A ∪ B m = A) ↔ m ≤ 5/2 :=
by sorry

end range_of_m_unbounded_below_m_characterization_of_m_range_l1293_129337


namespace sum_of_integers_l1293_129308

theorem sum_of_integers (m n p q : ℕ+) : 
  m ≠ n ∧ m ≠ p ∧ m ≠ q ∧ n ≠ p ∧ n ≠ q ∧ p ≠ q →
  (6 - m) * (6 - n) * (6 - p) * (6 - q) = 4 →
  m + n + p + q = 24 := by sorry

end sum_of_integers_l1293_129308


namespace correct_equation_l1293_129352

theorem correct_equation : 4 - 4 / 2 = 2 := by
  sorry

end correct_equation_l1293_129352


namespace system_solution_l1293_129316

theorem system_solution : ∃ (x y : ℚ), 
  (7 * x = -10 - 3 * y) ∧ 
  (4 * x = 6 * y - 38) ∧ 
  (x = -29/9) ∧ 
  (y = -113/27) := by
sorry

end system_solution_l1293_129316


namespace sum_m_n_equals_34_l1293_129331

theorem sum_m_n_equals_34 (m n : ℕ+) (p : ℚ) : 
  m + 15 < n + 5 →
  (m + (m + 5) + (m + 15) + (n + 5) + (n + 6) + (2 * n - 1)) / 6 = p →
  ((m + 15) + (n + 5)) / 2 = p →
  m + n = 34 := by sorry

end sum_m_n_equals_34_l1293_129331


namespace target_line_is_correct_l1293_129314

-- Define the line we're looking for
def target_line (x y : ℝ) : Prop := y = x + 1

-- Define the given line x + y = 0
def given_line (x y : ℝ) : Prop := x + y = 0

-- Define perpendicularity of two lines
def perpendicular (f g : ℝ → ℝ → Prop) : Prop :=
  ∀ x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄, 
    f x₁ y₁ ∧ f x₂ y₂ ∧ g x₃ y₃ ∧ g x₄ y₄ ∧ 
    x₁ ≠ x₂ ∧ x₃ ≠ x₄ → 
    (y₂ - y₁) / (x₂ - x₁) * (y₄ - y₃) / (x₄ - x₃) = -1

-- Theorem statement
theorem target_line_is_correct : 
  target_line (-1) 0 ∧ 
  perpendicular target_line given_line :=
sorry

end target_line_is_correct_l1293_129314


namespace calculation_proof_l1293_129309

theorem calculation_proof : 
  1.2008 * 0.2008 * 2.4016 - 1.2008^3 - 1.2008 * 0.2008^2 = -1.2008 := by
sorry

end calculation_proof_l1293_129309


namespace constant_function_l1293_129330

variable (f : ℝ → ℝ)

theorem constant_function
  (h1 : Continuous f')
  (h2 : f 0 = 0)
  (h3 : ∀ x, |f' x| ≤ |f x|) :
  ∃ c, ∀ x, f x = c :=
sorry

end constant_function_l1293_129330


namespace quadratic_equation_prime_roots_l1293_129361

theorem quadratic_equation_prime_roots (p q : ℕ) 
  (h1 : ∃ (x y : ℕ), Prime x ∧ Prime y ∧ p * x^2 - q * x + 1985 = 0 ∧ p * y^2 - q * y + 1985 = 0) :
  12 * p^2 + q = 414 := by
  sorry

end quadratic_equation_prime_roots_l1293_129361


namespace arrangement_count_is_518400_l1293_129312

/-- The number of ways to arrange 4 math books and 6 history books with specific conditions -/
def arrangement_count : ℕ :=
  let math_books : ℕ := 4
  let history_books : ℕ := 6
  let math_ends : ℕ := math_books * (math_books - 1)
  let consecutive_history : ℕ := Nat.choose history_books 2
  let remaining_units : ℕ := 5  -- 4 single history books + 1 double-history unit
  let middle_arrangements : ℕ := Nat.factorial remaining_units
  let remaining_math_placements : ℕ := Nat.choose remaining_units 2 * Nat.factorial 2
  math_ends * consecutive_history * middle_arrangements * remaining_math_placements

/-- Theorem stating that the number of arrangements is 518,400 -/
theorem arrangement_count_is_518400 : arrangement_count = 518400 := by
  sorry

end arrangement_count_is_518400_l1293_129312


namespace profit_maximized_at_70_l1293_129359

/-- Represents the store's helmet sales scenario -/
structure HelmetStore where
  initialPrice : ℝ
  initialSales : ℝ
  priceReductionEffect : ℝ
  costPrice : ℝ

/-- Calculates the monthly profit for a given selling price -/
def monthlyProfit (store : HelmetStore) (sellingPrice : ℝ) : ℝ :=
  let salesVolume := store.initialSales + (store.initialPrice - sellingPrice) * store.priceReductionEffect
  (sellingPrice - store.costPrice) * salesVolume

/-- Theorem stating that 70 yuan maximizes the monthly profit -/
theorem profit_maximized_at_70 (store : HelmetStore) 
    (h1 : store.initialPrice = 80)
    (h2 : store.initialSales = 200)
    (h3 : store.priceReductionEffect = 20)
    (h4 : store.costPrice = 50) :
    ∀ x, monthlyProfit store 70 ≥ monthlyProfit store x := by
  sorry

#check profit_maximized_at_70

end profit_maximized_at_70_l1293_129359


namespace complex_sixth_root_of_negative_eight_l1293_129329

theorem complex_sixth_root_of_negative_eight :
  {z : ℂ | z^6 = -8} = {Complex.I * Real.rpow 2 (1/3), -Complex.I * Real.rpow 2 (1/3)} := by
  sorry

end complex_sixth_root_of_negative_eight_l1293_129329


namespace ellipse_equation_equiv_standard_form_l1293_129377

/-- The equation of an ellipse given the sum of distances from any point to two fixed points -/
def ellipse_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 2)^2 + y^2) + Real.sqrt ((x + 2)^2 + y^2) = 10

/-- The standard form of an ellipse equation -/
def ellipse_standard_form (x y : ℝ) : Prop :=
  x^2 / 25 + y^2 / 21 = 1

/-- Theorem stating that the ellipse equation is equivalent to its standard form -/
theorem ellipse_equation_equiv_standard_form :
  ∀ x y : ℝ, ellipse_equation x y ↔ ellipse_standard_form x y :=
sorry

end ellipse_equation_equiv_standard_form_l1293_129377


namespace inequality_range_l1293_129370

theorem inequality_range (m : ℝ) :
  (∀ x : ℝ, m * x^2 - m * x + 1 > 0) ↔ (0 ≤ m ∧ m < 4) :=
sorry

end inequality_range_l1293_129370


namespace tangent_line_to_circle_l1293_129369

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop := sorry

/-- Check if a line passes through a point -/
def passesThrough (l : Line) (p : Point) : Prop := sorry

theorem tangent_line_to_circle (c : Circle) (p : Point) :
  c.center = Point.mk 2 0 →
  c.radius = 2 →
  p = Point.mk 4 5 →
  ∀ l : Line, (isTangent l c ∧ passesThrough l p) ↔ 
    (l = Line.mk 21 (-20) 16 ∨ l = Line.mk 1 0 (-4)) := by sorry

end tangent_line_to_circle_l1293_129369


namespace opposite_sides_of_line_l1293_129319

theorem opposite_sides_of_line (m : ℝ) : 
  (∀ (x y : ℝ), 2*x + y - 2 = 0 → (2*(-2) + m - 2) * (2*m + 4 - 2) < 0) → 
  -1 < m ∧ m < 6 :=
sorry

end opposite_sides_of_line_l1293_129319


namespace sum_of_squares_rational_l1293_129391

theorem sum_of_squares_rational (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∃ q : ℚ, a + b = q) → 
  (∃ r : ℚ, a^3 + b^3 = r) → 
  (∃ s : ℚ, a^2 + b^2 = s) ∧ 
  ¬(∀ t u : ℚ, a = t ∧ b = u) :=
by sorry

end sum_of_squares_rational_l1293_129391


namespace triangle_sine_relation_l1293_129366

theorem triangle_sine_relation (A B C : ℝ) (h : 3 * Real.sin B ^ 2 + 7 * Real.sin C ^ 2 = 2 * Real.sin A * Real.sin B * Real.sin C + 2 * Real.sin A ^ 2) :
  Real.sin (A + π / 4) = Real.sqrt 10 / 10 := by
sorry

end triangle_sine_relation_l1293_129366


namespace point_on_segment_vector_relation_l1293_129365

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

-- Define points
variable (A B M O C D : V)

-- Define conditions
variable (h1 : M ∈ closedSegment A B)
variable (h2 : O ∉ line_through A B)
variable (h3 : C = 2 • O - A)  -- C is symmetric to A with respect to O
variable (h4 : D = 2 • C - B)  -- D is symmetric to B with respect to C
variable (x y : ℝ)
variable (h5 : O - M = x • (O - C) + y • (O - D))

-- Theorem statement
theorem point_on_segment_vector_relation :
  x + 3 * y = -1 :=
sorry

end point_on_segment_vector_relation_l1293_129365


namespace lily_bouquet_cost_l1293_129340

/-- The cost of a bouquet of lilies given the number of lilies -/
def bouquet_cost (n : ℕ) : ℚ :=
  sorry

/-- The property that the price is directly proportional to the number of lilies -/
axiom price_proportional (n m : ℕ) :
  n ≠ 0 → m ≠ 0 → bouquet_cost n / n = bouquet_cost m / m

theorem lily_bouquet_cost :
  bouquet_cost 18 = 30 →
  bouquet_cost 45 = 75 :=
by sorry

end lily_bouquet_cost_l1293_129340


namespace fahrenheit_to_celsius_l1293_129341

theorem fahrenheit_to_celsius (C F : ℝ) : 
  C = (4 / 7) * (F - 40) → C = 35 → F = 101.25 := by
  sorry

end fahrenheit_to_celsius_l1293_129341


namespace eight_digit_increasing_integers_mod_1000_l1293_129310

theorem eight_digit_increasing_integers_mod_1000 : 
  (Nat.choose 17 8) % 1000 = 310 := by sorry

end eight_digit_increasing_integers_mod_1000_l1293_129310


namespace senior_citizens_average_age_l1293_129360

theorem senior_citizens_average_age
  (total_members : ℕ)
  (overall_average_age : ℚ)
  (num_women : ℕ)
  (num_men : ℕ)
  (num_seniors : ℕ)
  (women_average_age : ℚ)
  (men_average_age : ℚ)
  (h1 : total_members = 60)
  (h2 : overall_average_age = 30)
  (h3 : num_women = 25)
  (h4 : num_men = 20)
  (h5 : num_seniors = 15)
  (h6 : women_average_age = 28)
  (h7 : men_average_age = 35)
  (h8 : total_members = num_women + num_men + num_seniors) :
  (total_members * overall_average_age - num_women * women_average_age - num_men * men_average_age) / num_seniors = 80 / 3 :=
by sorry

end senior_citizens_average_age_l1293_129360


namespace ghee_mixture_volume_l1293_129375

/-- Prove that the volume of a mixture of two brands of vegetable ghee is 4 liters -/
theorem ghee_mixture_volume :
  ∀ (weight_a weight_b : ℝ) (ratio_a ratio_b : ℕ) (total_weight : ℝ),
    weight_a = 900 →
    weight_b = 700 →
    ratio_a = 3 →
    ratio_b = 2 →
    total_weight = 3280 →
    ∃ (volume_a volume_b : ℝ),
      volume_a / volume_b = ratio_a / ratio_b ∧
      weight_a * volume_a + weight_b * volume_b = total_weight ∧
      volume_a + volume_b = 4 := by
  sorry

end ghee_mixture_volume_l1293_129375


namespace min_plane_spotlights_theorem_min_space_spotlights_theorem_l1293_129348

/-- A spotlight that illuminates a 90° plane angle --/
structure PlaneSpotlight where
  angle : ℝ
  angle_eq : angle = 90

/-- A spotlight that illuminates a trihedral angle with all plane angles of 90° --/
structure SpaceSpotlight where
  angle : ℝ
  angle_eq : angle = 90

/-- The minimum number of spotlights required to illuminate the entire plane --/
def min_plane_spotlights : ℕ := 4

/-- The minimum number of spotlights required to illuminate the entire space --/
def min_space_spotlights : ℕ := 8

/-- Theorem stating the minimum number of spotlights required for full plane illumination --/
theorem min_plane_spotlights_theorem (s : PlaneSpotlight) :
  min_plane_spotlights = 4 := by sorry

/-- Theorem stating the minimum number of spotlights required for full space illumination --/
theorem min_space_spotlights_theorem (s : SpaceSpotlight) :
  min_space_spotlights = 8 := by sorry

end min_plane_spotlights_theorem_min_space_spotlights_theorem_l1293_129348


namespace seven_power_minus_two_power_l1293_129342

theorem seven_power_minus_two_power : 
  ∀ x y : ℕ+, 7^(x.val) - 3 * 2^(y.val) = 1 ↔ (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 4) :=
by sorry

end seven_power_minus_two_power_l1293_129342


namespace alberto_bjorn_distance_difference_l1293_129364

/-- Proves that the difference between Alberto's and Bjorn's biking distances is 10 miles -/
theorem alberto_bjorn_distance_difference : 
  ∀ (alberto_distance bjorn_distance : ℕ), 
  alberto_distance = 75 → 
  bjorn_distance = 65 → 
  alberto_distance - bjorn_distance = 10 := by
sorry

end alberto_bjorn_distance_difference_l1293_129364


namespace quadratic_always_positive_l1293_129354

theorem quadratic_always_positive : ∀ x : ℝ, x^2 + x + 2 > 0 := by
  sorry

end quadratic_always_positive_l1293_129354


namespace ratio_x_to_y_l1293_129304

theorem ratio_x_to_y (x y : ℝ) (h : (12*x - 5*y) / (15*x - 3*y) = 3/5) : 
  x / y = 16/15 := by sorry

end ratio_x_to_y_l1293_129304


namespace smallest_six_digit_divisible_by_100011_l1293_129384

theorem smallest_six_digit_divisible_by_100011 :
  ∀ n : ℕ, 100000 ≤ n ∧ n < 1000000 → n % 100011 = 0 → n ≥ 100011 :=
by
  sorry

end smallest_six_digit_divisible_by_100011_l1293_129384


namespace inscribed_sphere_radius_l1293_129388

/-- A perfect sphere inscribed in a cube -/
structure InscribedSphere where
  cube_side_length : ℝ
  touches_face_centers : Bool
  radius : ℝ

/-- Theorem: The radius of a perfect sphere inscribed in a cube with side length 2,
    such that it touches the center of each face, is equal to 1 -/
theorem inscribed_sphere_radius
  (s : InscribedSphere)
  (h1 : s.cube_side_length = 2)
  (h2 : s.touches_face_centers = true) :
  s.radius = 1 := by
  sorry

end inscribed_sphere_radius_l1293_129388


namespace fourth_circle_radius_is_p_l1293_129367

-- Define the right triangle
structure RightTriangle :=
  (a b c : ℝ)
  (right_angle : a^2 + b^2 = c^2)
  (perimeter : a + b + c = 2 * p)

-- Define the circles
structure Circles (t : RightTriangle) :=
  (r1 r2 r3 : ℝ)
  (externally_tangent : t.a = r2 + r3 ∧ t.b = r1 + r3 ∧ t.c = r1 + r2)
  (fourth_circle_radius : ℝ)
  (internally_tangent : 
    t.a = fourth_circle_radius - r3 + (fourth_circle_radius - r2) ∧
    t.b = fourth_circle_radius - r1 + (fourth_circle_radius - r3) ∧
    t.c = fourth_circle_radius - r1 + (fourth_circle_radius - r2))

-- The theorem to prove
theorem fourth_circle_radius_is_p (t : RightTriangle) (c : Circles t) : 
  c.fourth_circle_radius = p :=
sorry

end fourth_circle_radius_is_p_l1293_129367


namespace powers_of_two_in_arithmetic_sequence_l1293_129368

theorem powers_of_two_in_arithmetic_sequence (k : ℕ) :
  (∃ n : ℕ, 2^k = 6*n + 8) ↔ (k > 1 ∧ k % 2 = 1) :=
sorry

end powers_of_two_in_arithmetic_sequence_l1293_129368


namespace double_reflection_of_D_l1293_129389

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def reflect_line (p : ℝ × ℝ) : ℝ × ℝ :=
  let p' := (p.1, p.2 - 1)  -- Translate down by 1
  let reflected := (-p'.2, -p'.1)  -- Reflect across y = -x
  (reflected.1, reflected.2 + 1)  -- Translate up by 1

def double_reflection (p : ℝ × ℝ) : ℝ × ℝ :=
  reflect_line (reflect_x p)

theorem double_reflection_of_D :
  double_reflection (4, 1) = (2, -3) := by
  sorry

end double_reflection_of_D_l1293_129389


namespace abc_divisibility_problem_l1293_129393

theorem abc_divisibility_problem :
  ∀ a b c : ℕ,
    a > 1 → b > 1 → c > 1 →
    (c ∣ (a * b + 1)) →
    (a ∣ (b * c + 1)) →
    (b ∣ (c * a + 1)) →
    ((a = 2 ∧ b = 3 ∧ c = 7) ∨
     (a = 2 ∧ b = 7 ∧ c = 3) ∨
     (a = 3 ∧ b = 2 ∧ c = 7) ∨
     (a = 3 ∧ b = 7 ∧ c = 2) ∨
     (a = 7 ∧ b = 2 ∧ c = 3) ∨
     (a = 7 ∧ b = 3 ∧ c = 2)) :=
by sorry


end abc_divisibility_problem_l1293_129393


namespace base4_division_theorem_l1293_129324

/-- Converts a base 4 number to base 10 --/
def base4_to_base10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4^i)) 0

/-- Converts a base 10 number to base 4 --/
def base10_to_base4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
  aux n []

theorem base4_division_theorem :
  let dividend := [3, 1, 2, 3]  -- 3213₄ in reverse order
  let divisor := [3, 1]         -- 13₄ in reverse order
  let quotient := [1, 0, 2]     -- 201₄ in reverse order
  (base4_to_base10 dividend) / (base4_to_base10 divisor) = base4_to_base10 quotient :=
by sorry

end base4_division_theorem_l1293_129324


namespace initial_orchids_is_three_l1293_129395

/-- Represents the number of flowers in a vase -/
structure FlowerVase where
  initialRoses : ℕ
  finalRoses : ℕ
  finalOrchids : ℕ
  orchidsCut : ℕ

/-- Calculates the initial number of orchids in the vase -/
def initialOrchids (v : FlowerVase) : ℕ :=
  v.finalOrchids - v.orchidsCut

/-- Theorem stating that the initial number of orchids is 3 -/
theorem initial_orchids_is_three (v : FlowerVase) 
  (h1 : v.initialRoses = 16)
  (h2 : v.finalRoses = 13)
  (h3 : v.finalOrchids = 7)
  (h4 : v.orchidsCut = 4) : 
  initialOrchids v = 3 := by
  sorry

end initial_orchids_is_three_l1293_129395


namespace march_first_is_monday_l1293_129398

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Returns the day of the week that is n days before the given day -/
def daysBefore (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => daysBefore (match d with
    | DayOfWeek.Sunday => DayOfWeek.Saturday
    | DayOfWeek.Monday => DayOfWeek.Sunday
    | DayOfWeek.Tuesday => DayOfWeek.Monday
    | DayOfWeek.Wednesday => DayOfWeek.Tuesday
    | DayOfWeek.Thursday => DayOfWeek.Wednesday
    | DayOfWeek.Friday => DayOfWeek.Thursday
    | DayOfWeek.Saturday => DayOfWeek.Friday) n

theorem march_first_is_monday (march13 : DayOfWeek) 
    (h : march13 = DayOfWeek.Saturday) : 
    daysBefore march13 12 = DayOfWeek.Monday := by
  sorry

end march_first_is_monday_l1293_129398


namespace percentage_difference_l1293_129379

theorem percentage_difference (A B C y : ℝ) : 
  C > A ∧ A > B ∧ B > 0 → 
  C = 2 * B → 
  A = C * (1 - y / 100) → 
  y = 100 - 50 * (A / B) :=
by
  sorry

end percentage_difference_l1293_129379


namespace max_perimeter_of_rectangle_from_triangles_l1293_129374

theorem max_perimeter_of_rectangle_from_triangles :
  ∀ (L W : ℝ),
  L > 0 → W > 0 →
  L * W = 60 * (1/2 * 2 * 3) →
  2 * (L + W) ≤ 184 :=
by
  sorry

end max_perimeter_of_rectangle_from_triangles_l1293_129374


namespace trapezoid_perimeter_l1293_129357

/-- Perimeter of a trapezoid EFGH with given properties -/
theorem trapezoid_perimeter (EF GH EG FH : ℝ) (h1 : EF = 40) (h2 : GH = 20) 
  (h3 : EG = 30) (h4 : FH = 45) : 
  EF + GH + Real.sqrt (EF ^ 2 - EG ^ 2) + Real.sqrt (FH ^ 2 - GH ^ 2) = 60 + 10 * Real.sqrt 7 + 5 * Real.sqrt 65 := by
  sorry

end trapezoid_perimeter_l1293_129357


namespace school_choir_robe_cost_l1293_129353

/-- Calculates the total cost of buying additional robes for a school choir, including discount and sales tax. -/
theorem school_choir_robe_cost
  (total_robes_needed : ℕ)
  (robes_owned : ℕ)
  (cost_per_robe : ℚ)
  (discount_rate : ℚ)
  (discount_threshold : ℕ)
  (sales_tax_rate : ℚ)
  (h1 : total_robes_needed = 30)
  (h2 : robes_owned = 12)
  (h3 : cost_per_robe = 2)
  (h4 : discount_rate = 15 / 100)
  (h5 : discount_threshold = 10)
  (h6 : sales_tax_rate = 8 / 100)
  : ∃ (final_cost : ℚ), final_cost = 3305 / 100 :=
by
  sorry

end school_choir_robe_cost_l1293_129353


namespace reciprocal_of_proper_fraction_greater_than_one_l1293_129325

-- Define a proper fraction
def ProperFraction (n d : ℕ) : Prop := 0 < n ∧ n < d

-- Theorem statement
theorem reciprocal_of_proper_fraction_greater_than_one {n d : ℕ} (h : ProperFraction n d) :
  (d : ℝ) / (n : ℝ) > 1 := by
  sorry


end reciprocal_of_proper_fraction_greater_than_one_l1293_129325
