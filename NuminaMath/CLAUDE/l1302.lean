import Mathlib

namespace NUMINAMATH_CALUDE_at_most_one_point_inside_plane_l1302_130207

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- A plane in 3D space -/
structure Plane3D where
  point : Point3D
  normal : Point3D

/-- Checks if a point is on a line -/
def isPointOnLine (p : Point3D) (l : Line3D) : Prop := sorry

/-- Checks if a point is on a plane -/
def isPointOnPlane (p : Point3D) (pl : Plane3D) : Prop := sorry

/-- Checks if a point is outside a plane -/
def isPointOutsidePlane (p : Point3D) (pl : Plane3D) : Prop := 
  ¬(isPointOnPlane p pl)

/-- The main theorem -/
theorem at_most_one_point_inside_plane 
  (l : Line3D) (pl : Plane3D) 
  (p1 p2 : Point3D) 
  (h1 : isPointOnLine p1 l) 
  (h2 : isPointOnLine p2 l) 
  (h3 : isPointOutsidePlane p1 pl) 
  (h4 : isPointOutsidePlane p2 pl) : 
  ∃! p, isPointOnLine p l ∧ isPointOnPlane p pl :=
sorry

end NUMINAMATH_CALUDE_at_most_one_point_inside_plane_l1302_130207


namespace NUMINAMATH_CALUDE_system_solution_triangle_side_range_l1302_130282

-- Problem 1
theorem system_solution (m : ℤ) : 
  (∃ x y : ℝ, 2*x + y = -3*m + 2 ∧ x + 2*y = 4 ∧ x + y > -3/2) ↔ 
  (m = 1 ∨ m = 2 ∨ m = 3) :=
sorry

-- Problem 2
theorem triangle_side_range (a b c : ℝ) :
  (a^2 + b^2 = 10*a + 8*b - 41) ∧
  (c ≥ a ∧ c ≥ b) →
  (5 ≤ c ∧ c < 9) :=
sorry

end NUMINAMATH_CALUDE_system_solution_triangle_side_range_l1302_130282


namespace NUMINAMATH_CALUDE_expression_evaluation_l1302_130276

theorem expression_evaluation :
  let a : ℚ := 2
  let b : ℚ := 1
  let expr := -1/3 * (a^3*b - a*b) + a*b^3 - (a*b - b)/2 - 1/2*b + 1/3*a^3*b
  expr = 5/3 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1302_130276


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l1302_130272

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x > 0, P x) ↔ (∀ x > 0, ¬ P x) := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x > 0, x^2 - 2*x + 1 < 0) ↔ (∀ x > 0, x^2 - 2*x + 1 ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l1302_130272


namespace NUMINAMATH_CALUDE_pizza_coverage_l1302_130269

/-- Represents a circular pizza with cheese circles on it. -/
structure CheesePizza where
  pizza_diameter : ℝ
  cheese_circles_across : ℕ
  total_cheese_circles : ℕ

/-- Calculates the fraction of the pizza covered by cheese. -/
def fraction_covered (pizza : CheesePizza) : ℚ :=
  sorry

/-- Theorem stating the fraction of pizza covered by cheese -/
theorem pizza_coverage (pizza : CheesePizza) 
  (h1 : pizza.pizza_diameter = 15)
  (h2 : pizza.cheese_circles_across = 9)
  (h3 : pizza.total_cheese_circles = 36) : 
  fraction_covered pizza = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_pizza_coverage_l1302_130269


namespace NUMINAMATH_CALUDE_sarah_copies_360_pages_l1302_130214

/-- The number of pages Sarah will copy for a meeting -/
def total_pages (num_people : ℕ) (copies_per_person : ℕ) (pages_per_contract : ℕ) : ℕ :=
  num_people * copies_per_person * pages_per_contract

/-- Proof that Sarah will copy 360 pages for the meeting -/
theorem sarah_copies_360_pages : 
  total_pages 9 2 20 = 360 := by
  sorry

end NUMINAMATH_CALUDE_sarah_copies_360_pages_l1302_130214


namespace NUMINAMATH_CALUDE_distance_to_black_planet_l1302_130250

/-- The distance to a black planet given spaceship and light travel times -/
theorem distance_to_black_planet 
  (v_ship : ℝ) -- speed of spaceship
  (v_light : ℝ) -- speed of light
  (t_total : ℝ) -- total time of travel and light reflection
  (h_v_ship : v_ship = 100000) -- spaceship speed in km/s
  (h_v_light : v_light = 300000) -- light speed in km/s
  (h_t_total : t_total = 100) -- total time in seconds
  : ∃ d : ℝ, d = 1500 * 10000 ∧ t_total = (d + v_ship * t_total) / v_light + d / v_light :=
by sorry

end NUMINAMATH_CALUDE_distance_to_black_planet_l1302_130250


namespace NUMINAMATH_CALUDE_matrix_equation_equivalence_l1302_130233

theorem matrix_equation_equivalence 
  (n : ℕ) 
  (A B C : Matrix (Fin n) (Fin n) ℝ) 
  (h_inv : IsUnit A) 
  (h_eq : (A - B) * C = B * A⁻¹) : 
  C * (A - B) = A⁻¹ * B := by
  sorry

end NUMINAMATH_CALUDE_matrix_equation_equivalence_l1302_130233


namespace NUMINAMATH_CALUDE_manager_average_salary_l1302_130201

/-- Proves that the average salary of managers is $90,000 given the company's employee structure and salary information --/
theorem manager_average_salary 
  (num_managers : ℕ) 
  (num_associates : ℕ) 
  (associate_avg_salary : ℝ) 
  (company_avg_salary : ℝ) : 
  num_managers = 15 → 
  num_associates = 75 → 
  associate_avg_salary = 30000 → 
  company_avg_salary = 40000 → 
  (num_managers * (num_managers * company_avg_salary - num_associates * associate_avg_salary) / 
   (num_managers * (num_managers + num_associates))) = 90000 := by
  sorry

end NUMINAMATH_CALUDE_manager_average_salary_l1302_130201


namespace NUMINAMATH_CALUDE_toms_age_ratio_l1302_130236

theorem toms_age_ratio (T N : ℚ) : 
  (T > 0) →  -- Tom's age is positive
  (N > 0) →  -- N is positive (years in the past)
  (∃ (a b c d : ℚ), a + b + c + d = T ∧ a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) →  -- Four children's ages sum to T
  (T - N = 3 * (T - 4 * N)) →  -- Condition from N years ago
  T / N = 11 / 2 := by
sorry

end NUMINAMATH_CALUDE_toms_age_ratio_l1302_130236


namespace NUMINAMATH_CALUDE_binomial_square_constant_l1302_130299

theorem binomial_square_constant (c : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, 9*x^2 - 24*x + c = (a*x + b)^2) → c = 16 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_constant_l1302_130299


namespace NUMINAMATH_CALUDE_total_seashells_l1302_130275

def sam_seashells : ℕ := 35
def joan_seashells : ℕ := 18

theorem total_seashells : sam_seashells + joan_seashells = 53 := by
  sorry

end NUMINAMATH_CALUDE_total_seashells_l1302_130275


namespace NUMINAMATH_CALUDE_power_multiplication_l1302_130229

theorem power_multiplication (a : ℝ) : a * a^3 = a^4 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l1302_130229


namespace NUMINAMATH_CALUDE_mouse_ratio_l1302_130204

/-- Represents the mouse distribution problem --/
def mouse_distribution (total_mice : ℕ) (robbie_fraction : ℚ) (store_multiple : ℕ) (feeder_fraction : ℚ) (remaining : ℕ) : Prop :=
  let robbie_mice := (total_mice : ℚ) * robbie_fraction
  let store_mice := (robbie_mice * store_multiple : ℚ)
  let before_feeder := (total_mice : ℚ) - robbie_mice - store_mice
  (before_feeder * feeder_fraction = (remaining : ℚ)) ∧
  (store_mice / robbie_mice = 3)

/-- Theorem stating the ratio of mice sold to pet store vs given to Robbie --/
theorem mouse_ratio :
  ∃ (store_multiple : ℕ),
    mouse_distribution 24 (1/6 : ℚ) store_multiple (1/2 : ℚ) 4 :=
sorry

end NUMINAMATH_CALUDE_mouse_ratio_l1302_130204


namespace NUMINAMATH_CALUDE_root_equation_implies_expression_value_l1302_130291

theorem root_equation_implies_expression_value (a : ℝ) : 
  a^2 + a - 1 = 0 → 2021 - 2*a^2 - 2*a = 2019 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_implies_expression_value_l1302_130291


namespace NUMINAMATH_CALUDE_integer_pairs_satisfying_equation_l1302_130224

theorem integer_pairs_satisfying_equation :
  {(x, y) : ℤ × ℤ | x^2 = y^2 + 2*y + 13} =
  {(4, -3), (4, 1), (-4, 1), (-4, -3)} := by
  sorry

end NUMINAMATH_CALUDE_integer_pairs_satisfying_equation_l1302_130224


namespace NUMINAMATH_CALUDE_saree_sale_price_l1302_130223

/-- The sale price of a saree after successive discounts -/
theorem saree_sale_price (original_price : ℝ) (discount1 discount2 discount3 discount4 : ℝ) :
  original_price = 400 →
  discount1 = 0.20 →
  discount2 = 0.05 →
  discount3 = 0.10 →
  discount4 = 0.15 →
  original_price * (1 - discount1) * (1 - discount2) * (1 - discount3) * (1 - discount4) = 232.56 := by
  sorry

end NUMINAMATH_CALUDE_saree_sale_price_l1302_130223


namespace NUMINAMATH_CALUDE_max_notebooks_is_11_l1302_130240

/-- Represents the number of notebooks in a pack -/
inductive NotebookPack
  | Single
  | Pack4
  | Pack7

/-- The cost of a notebook pack in dollars -/
def cost (pack : NotebookPack) : ℕ :=
  match pack with
  | .Single => 2
  | .Pack4 => 6
  | .Pack7 => 9

/-- The number of notebooks in a pack -/
def notebooks (pack : NotebookPack) : ℕ :=
  match pack with
  | .Single => 1
  | .Pack4 => 4
  | .Pack7 => 7

/-- Maria's budget in dollars -/
def budget : ℕ := 15

/-- A purchase combination is valid if it doesn't exceed the budget -/
def isValidPurchase (singles pack4s pack7s : ℕ) : Prop :=
  singles * cost .Single + pack4s * cost .Pack4 + pack7s * cost .Pack7 ≤ budget

/-- The total number of notebooks for a given purchase combination -/
def totalNotebooks (singles pack4s pack7s : ℕ) : ℕ :=
  singles * notebooks .Single + pack4s * notebooks .Pack4 + pack7s * notebooks .Pack7

/-- Theorem: The maximum number of notebooks that can be purchased with the given budget is 11 -/
theorem max_notebooks_is_11 :
    ∀ singles pack4s pack7s : ℕ,
      isValidPurchase singles pack4s pack7s →
      totalNotebooks singles pack4s pack7s ≤ 11 ∧
      ∃ s p4 p7 : ℕ, isValidPurchase s p4 p7 ∧ totalNotebooks s p4 p7 = 11 :=
  sorry


end NUMINAMATH_CALUDE_max_notebooks_is_11_l1302_130240


namespace NUMINAMATH_CALUDE_money_left_calculation_l1302_130248

/-- The amount of money Mary has left after purchasing pizzas and drinks -/
def money_left (p : ℝ) : ℝ :=
  let drink_cost := p
  let medium_pizza_cost := 3 * p
  let large_pizza_cost := 4 * p
  let total_spent := 5 * drink_cost + 2 * medium_pizza_cost + large_pizza_cost
  50 - total_spent

/-- Theorem stating that the money left is equal to 50 - 15p -/
theorem money_left_calculation (p : ℝ) : money_left p = 50 - 15 * p := by
  sorry

end NUMINAMATH_CALUDE_money_left_calculation_l1302_130248


namespace NUMINAMATH_CALUDE_radius_of_circle_B_l1302_130297

/-- A configuration of four circles A, B, C, and D with specific properties -/
structure CircleConfiguration where
  /-- Radius of circle A -/
  radiusA : ℝ
  /-- Radius of circle B -/
  radiusB : ℝ
  /-- Radius of circle D -/
  radiusD : ℝ
  /-- Circles A, B, and C are externally tangent to each other -/
  externallyTangent : Bool
  /-- Circles A, B, and C are internally tangent to circle D -/
  internallyTangentD : Bool
  /-- Circles B and C are congruent -/
  bCongruentC : Bool
  /-- Circle A passes through the center of D -/
  aPassesThroughCenterD : Bool

/-- The theorem stating that given the specific configuration, the radius of circle B is 7/3 -/
theorem radius_of_circle_B (config : CircleConfiguration) 
  (h1 : config.radiusA = 2)
  (h2 : config.externallyTangent = true)
  (h3 : config.internallyTangentD = true)
  (h4 : config.bCongruentC = true)
  (h5 : config.aPassesThroughCenterD = true) :
  config.radiusB = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_radius_of_circle_B_l1302_130297


namespace NUMINAMATH_CALUDE_sin_15_degrees_l1302_130203

theorem sin_15_degrees : Real.sin (15 * π / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_15_degrees_l1302_130203


namespace NUMINAMATH_CALUDE_arithmetic_mean_after_removal_l1302_130289

theorem arithmetic_mean_after_removal (arr : Finset ℤ) (sum : ℤ) : 
  Finset.card arr = 40 →
  sum = Finset.sum arr id →
  sum / 40 = 45 →
  60 ∈ arr →
  70 ∈ arr →
  ((sum - 60 - 70) : ℚ) / 38 = 43.95 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_after_removal_l1302_130289


namespace NUMINAMATH_CALUDE_evaluate_expression_l1302_130253

theorem evaluate_expression : 7 - 5 * (6 - 2^3) * 3 = -23 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1302_130253


namespace NUMINAMATH_CALUDE_soccer_team_lineup_count_l1302_130263

def total_players : ℕ := 18
def goalie_needed : ℕ := 1
def field_players_needed : ℕ := 10

theorem soccer_team_lineup_count :
  (total_players.choose goalie_needed) * ((total_players - goalie_needed).choose field_players_needed) = 349864 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_lineup_count_l1302_130263


namespace NUMINAMATH_CALUDE_distance_product_zero_l1302_130216

-- Define the curve C
def C : Set (ℝ × ℝ) := {(x, y) | x^2 / 36 + y^2 / 16 = 1}

-- Define the line l
def l : Set (ℝ × ℝ) := {(x, y) | ∃ t : ℝ, x = 1 - t/2 ∧ y = 1 + (Real.sqrt 3 * t)/2}

-- Define point P
def P : ℝ × ℝ := (1, 1)

-- Theorem statement
theorem distance_product_zero (A B : ℝ × ℝ) 
  (hA : A ∈ C ∩ l) (hB : B ∈ C ∩ l) (hAB : A ≠ B) :
  ‖P‖ * ‖P - B‖ = 0 := by
  sorry

end NUMINAMATH_CALUDE_distance_product_zero_l1302_130216


namespace NUMINAMATH_CALUDE_no_four_binomial_coeff_arithmetic_progression_l1302_130294

theorem no_four_binomial_coeff_arithmetic_progression :
  ∀ n m : ℕ, n > 0 → m > 0 → m + 3 ≤ n →
  ¬∃ d : ℕ, 
    (Nat.choose n (m+1) = Nat.choose n m + d) ∧
    (Nat.choose n (m+2) = Nat.choose n (m+1) + d) ∧
    (Nat.choose n (m+3) = Nat.choose n (m+2) + d) :=
by sorry

end NUMINAMATH_CALUDE_no_four_binomial_coeff_arithmetic_progression_l1302_130294


namespace NUMINAMATH_CALUDE_range_of_a_l1302_130218

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 + 4*x else 4*x - x^2

theorem range_of_a (a : ℝ) (h : f (a - 2) + f a > 0) : a > 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1302_130218


namespace NUMINAMATH_CALUDE_position_3_1_is_B_l1302_130268

-- Define the grid type
def Grid := Fin 5 → Fin 5 → Char

-- Define the valid letters
def ValidLetter (c : Char) : Prop := c ∈ ['A', 'B', 'C', 'D', 'E']

-- Define the property of a valid grid
def ValidGrid (g : Grid) : Prop :=
  (∀ r c, ValidLetter (g r c)) ∧
  (∀ r, ∀ c₁ c₂, c₁ ≠ c₂ → g r c₁ ≠ g r c₂) ∧
  (∀ c, ∀ r₁ r₂, r₁ ≠ r₂ → g r₁ c ≠ g r₂ c) ∧
  (∀ i j, i ≠ j → g i i ≠ g j j) ∧
  (∀ i j, i ≠ j → g i (4 - i) ≠ g j (4 - j))

-- Define the theorem
theorem position_3_1_is_B (g : Grid) (h : ValidGrid g)
  (h1 : g 0 0 = 'A') (h2 : g 3 0 = 'D') (h3 : g 4 0 = 'E') :
  g 2 0 = 'B' := by
  sorry

end NUMINAMATH_CALUDE_position_3_1_is_B_l1302_130268


namespace NUMINAMATH_CALUDE_first_three_decimal_digits_l1302_130219

theorem first_three_decimal_digits (n : ℕ) (x : ℝ) : 
  n = 2003 → 
  x = (10^n + 1)^(11/7) → 
  ∃ (k : ℕ), x = k + 0.571 + r ∧ 0 ≤ r ∧ r < 0.001 :=
sorry

end NUMINAMATH_CALUDE_first_three_decimal_digits_l1302_130219


namespace NUMINAMATH_CALUDE_johns_allowance_l1302_130274

theorem johns_allowance (A : ℚ) : 
  (7/15 + 3/10 + 1/6 : ℚ) * A +  -- Spent on arcade, books, and clothes
  2/5 * (1 - (7/15 + 3/10 + 1/6 : ℚ)) * A +  -- Spent at toy store
  (6/5 : ℚ) = A  -- Last $1.20 spent at candy store (represented as 6/5)
  → A = 30 := by
sorry

end NUMINAMATH_CALUDE_johns_allowance_l1302_130274


namespace NUMINAMATH_CALUDE_kevin_collected_18_frisbees_l1302_130295

/-- The number of frisbees Kevin collected for prizes at the fair. -/
def num_frisbees (total_prizes stuffed_animals yo_yos : ℕ) : ℕ :=
  total_prizes - (stuffed_animals + yo_yos)

/-- Theorem stating that Kevin collected 18 frisbees. -/
theorem kevin_collected_18_frisbees : 
  num_frisbees 50 14 18 = 18 := by
  sorry

end NUMINAMATH_CALUDE_kevin_collected_18_frisbees_l1302_130295


namespace NUMINAMATH_CALUDE_square_field_area_l1302_130237

/-- The area of a square field with side length 14 meters is 196 square meters. -/
theorem square_field_area : 
  ∀ (side_length area : ℝ), 
  side_length = 14 → 
  area = side_length ^ 2 → 
  area = 196 :=
by
  sorry

end NUMINAMATH_CALUDE_square_field_area_l1302_130237


namespace NUMINAMATH_CALUDE_sum_of_cube_difference_l1302_130255

theorem sum_of_cube_difference (a b c : ℕ+) :
  (a + b + c)^3 - a^3 - b^3 - c^3 = 150 →
  a + b + c = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cube_difference_l1302_130255


namespace NUMINAMATH_CALUDE_dagger_example_l1302_130254

-- Define the † operation
def dagger (m n p q : ℚ) : ℚ := m * p * ((n + q) / n)

-- Theorem statement
theorem dagger_example : dagger (9/5) (7/2) = 441/5 := by
  sorry

end NUMINAMATH_CALUDE_dagger_example_l1302_130254


namespace NUMINAMATH_CALUDE_comic_books_problem_l1302_130292

theorem comic_books_problem (sold : ℕ) (left : ℕ) (h1 : sold = 65) (h2 : left = 25) :
  sold + left = 90 := by
  sorry

end NUMINAMATH_CALUDE_comic_books_problem_l1302_130292


namespace NUMINAMATH_CALUDE_x_plus_2y_equals_8_l1302_130202

theorem x_plus_2y_equals_8 (x y : ℝ) 
  (h1 : (x + y) / 3 = 1.6666666666666667)
  (h2 : 2 * x + y = 7) : 
  x + 2 * y = 8 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_2y_equals_8_l1302_130202


namespace NUMINAMATH_CALUDE_complex_equality_implies_a_value_l1302_130249

theorem complex_equality_implies_a_value (a : ℝ) : 
  (Complex.re ((1 + 2*I) * (2*a + I)) = Complex.im ((1 + 2*I) * (2*a + I))) → 
  a = -5/2 := by
sorry

end NUMINAMATH_CALUDE_complex_equality_implies_a_value_l1302_130249


namespace NUMINAMATH_CALUDE_bridgets_skittles_proof_l1302_130286

/-- The number of Skittles Henry has -/
def henrys_skittles : ℕ := 4

/-- The total number of Skittles after Henry gives his to Bridget -/
def total_skittles : ℕ := 8

/-- Bridget's initial number of Skittles -/
def bridgets_initial_skittles : ℕ := total_skittles - henrys_skittles

theorem bridgets_skittles_proof :
  bridgets_initial_skittles = 4 :=
by sorry

end NUMINAMATH_CALUDE_bridgets_skittles_proof_l1302_130286


namespace NUMINAMATH_CALUDE_real_root_quadratic_l1302_130277

theorem real_root_quadratic (m : ℝ) : 
  (∃ x : ℝ, x^2 - (1 - Complex.I) * x + m + 2 * Complex.I = 0) → m = -6 := by
  sorry

end NUMINAMATH_CALUDE_real_root_quadratic_l1302_130277


namespace NUMINAMATH_CALUDE_march_greatest_drop_l1302_130208

/-- Represents the months from January to August --/
inductive Month
| January
| February
| March
| April
| May
| June
| July
| August

/-- Price change for each month --/
def price_change (m : Month) : ℝ :=
  match m with
  | Month.January => -1.00
  | Month.February => 1.50
  | Month.March => -3.00
  | Month.April => 2.00
  | Month.May => -0.75
  | Month.June => 1.00
  | Month.July => -2.50
  | Month.August => -2.00

/-- Definition of a price drop --/
def is_price_drop (m : Month) : Prop :=
  price_change m < 0

/-- Theorem: March has the greatest monthly drop in price --/
theorem march_greatest_drop :
  ∀ m : Month, is_price_drop m → price_change Month.March ≤ price_change m :=
by sorry

end NUMINAMATH_CALUDE_march_greatest_drop_l1302_130208


namespace NUMINAMATH_CALUDE_parallel_tangents_ordinates_l1302_130215

/-- The curve function y = x³ - 3x² + 6x + 2 -/
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 6*x + 2

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x + 6

theorem parallel_tangents_ordinates (P Q : ℝ × ℝ) :
  P.2 = f P.1 →
  Q.2 = f Q.1 →
  f' P.1 = f' Q.1 →
  P.2 = 1 →
  Q.2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_parallel_tangents_ordinates_l1302_130215


namespace NUMINAMATH_CALUDE_largest_integer_square_sum_l1302_130260

theorem largest_integer_square_sum : ∃ (x y z : ℤ),
  6^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 5*x + 5*y + 5*z - 10 ∧
  ∀ (n : ℤ), n > 6 → ¬∃ (x y z : ℤ),
    n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 5*x + 5*y + 5*z - 10 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_square_sum_l1302_130260


namespace NUMINAMATH_CALUDE_company_manager_fraction_l1302_130296

/-- Fraction of employees who are managers -/
def manager_fraction (total_employees : ℕ) (total_managers : ℕ) : ℚ :=
  total_managers / total_employees

theorem company_manager_fraction :
  ∀ (total_employees : ℕ) (total_managers : ℕ) (male_employees : ℕ) (male_managers : ℕ),
    total_employees > 0 →
    male_employees > 0 →
    total_employees = 625 + male_employees →
    total_managers = 250 + male_managers →
    manager_fraction total_employees total_managers = manager_fraction 625 250 →
    manager_fraction total_employees total_managers = manager_fraction male_employees male_managers →
    manager_fraction total_employees total_managers = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_company_manager_fraction_l1302_130296


namespace NUMINAMATH_CALUDE_zoey_lottery_split_l1302_130230

theorem zoey_lottery_split (lottery_amount : ℕ) (h1 : lottery_amount = 7348340) :
  ∃ (num_friends : ℕ), 
    (lottery_amount + 1) % (num_friends + 1) = 0 ∧ 
    num_friends = 7348340 := by
  sorry

end NUMINAMATH_CALUDE_zoey_lottery_split_l1302_130230


namespace NUMINAMATH_CALUDE_extremum_implies_f_2_eq_18_l1302_130242

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

-- Define the derivative of f
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extremum_implies_f_2_eq_18 (a b : ℝ) :
  f' a b 1 = 0 →  -- f has a critical point at x = 1
  f a b 1 = 10 →  -- The value of f at x = 1 is 10
  f a b 2 = 18    -- Then f(2) = 18
:= by sorry

end NUMINAMATH_CALUDE_extremum_implies_f_2_eq_18_l1302_130242


namespace NUMINAMATH_CALUDE_existence_of_abc_l1302_130221

theorem existence_of_abc (p : ℕ) (hp : p.Prime) (hp_gt_2011 : p > 2011) :
  ∃ (a b c : ℕ+), (¬(p ∣ a) ∨ ¬(p ∣ b) ∨ ¬(p ∣ c)) ∧
    ∀ (n : ℕ+), p ∣ (n^4 - 2*n^2 + 9) → p ∣ (24*a*n^2 + 5*b*n + 2011*c) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_abc_l1302_130221


namespace NUMINAMATH_CALUDE_fraction_gt_one_not_equivalent_to_a_gt_b_l1302_130278

theorem fraction_gt_one_not_equivalent_to_a_gt_b :
  ¬(∀ (a b : ℝ), a / b > 1 ↔ a > b) :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_gt_one_not_equivalent_to_a_gt_b_l1302_130278


namespace NUMINAMATH_CALUDE_product_of_roots_l1302_130281

theorem product_of_roots (x : ℝ) : 
  (3 * x^2 + 6 * x - 81 = 0) → 
  ∃ y : ℝ, (3 * y^2 + 6 * y - 81 = 0) ∧ (x * y = -27) := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_l1302_130281


namespace NUMINAMATH_CALUDE_abs_one_point_five_minus_sqrt_two_l1302_130220

theorem abs_one_point_five_minus_sqrt_two :
  |1.5 - Real.sqrt 2| = 1.5 - Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_abs_one_point_five_minus_sqrt_two_l1302_130220


namespace NUMINAMATH_CALUDE_average_difference_l1302_130226

theorem average_difference (a b c : ℝ) 
  (hab : (a + b) / 2 = 110) 
  (hbc : (b + c) / 2 = 150) : 
  a - c = -80 := by sorry

end NUMINAMATH_CALUDE_average_difference_l1302_130226


namespace NUMINAMATH_CALUDE_exclusive_math_enrollment_is_29_l1302_130256

/-- Represents the number of students in each class or combination of classes --/
structure ClassEnrollment where
  total : ℕ
  math : ℕ
  foreign : ℕ
  musicOnly : ℕ

/-- Calculates the number of students enrolled exclusively in math classes --/
def exclusiveMathEnrollment (e : ClassEnrollment) : ℕ :=
  e.math - (e.math + e.foreign - (e.total - e.musicOnly))

/-- Theorem stating that given the conditions, 29 students are enrolled exclusively in math --/
theorem exclusive_math_enrollment_is_29 (e : ClassEnrollment)
  (h1 : e.total = 120)
  (h2 : e.math = 82)
  (h3 : e.foreign = 71)
  (h4 : e.musicOnly = 20) :
  exclusiveMathEnrollment e = 29 := by
  sorry

#eval exclusiveMathEnrollment ⟨120, 82, 71, 20⟩

end NUMINAMATH_CALUDE_exclusive_math_enrollment_is_29_l1302_130256


namespace NUMINAMATH_CALUDE_correct_assignment_count_l1302_130217

def num_rooms : ℕ := 6
def num_friends : ℕ := 6
def max_occupancy : ℕ := 3
def min_occupancy : ℕ := 1
def num_inseparable_friends : ℕ := 2

-- Function to calculate the number of ways to assign friends to rooms
def assignment_ways : ℕ := sorry

-- Theorem statement
theorem correct_assignment_count :
  assignment_ways = 3600 :=
sorry

end NUMINAMATH_CALUDE_correct_assignment_count_l1302_130217


namespace NUMINAMATH_CALUDE_optimal_fare_and_passenger_change_l1302_130238

/-- Demand function -/
def demand (p : ℝ) : ℝ := 4200 - 100 * p

/-- Train fare -/
def train_fare : ℝ := 4

/-- Train capacity -/
def train_capacity : ℝ := 800

/-- Bus company cost function -/
def bus_cost (y : ℝ) : ℝ := 10 * y + 225

/-- Optimal bus fare -/
def optimal_bus_fare : ℝ := 22

/-- Change in total passengers if train service closes -/
def passenger_change : ℝ := -400

/-- Theorem stating the optimal bus fare and passenger change -/
theorem optimal_fare_and_passenger_change :
  (∃ (p : ℝ), p = optimal_bus_fare ∧
    ∀ (p' : ℝ), p' > train_fare →
      p * (demand p - train_capacity) - bus_cost (demand p - train_capacity) ≥
      p' * (demand p' - train_capacity) - bus_cost (demand p' - train_capacity)) ∧
  (demand (26) - (demand optimal_bus_fare - train_capacity + train_capacity) = passenger_change) :=
sorry

end NUMINAMATH_CALUDE_optimal_fare_and_passenger_change_l1302_130238


namespace NUMINAMATH_CALUDE_complex_equal_parts_l1302_130225

theorem complex_equal_parts (a : ℝ) : 
  let z : ℂ := (1 - a * Complex.I) / (2 + Complex.I)
  (z.re = z.im) → a = -3 := by
sorry

end NUMINAMATH_CALUDE_complex_equal_parts_l1302_130225


namespace NUMINAMATH_CALUDE_solve_equation_l1302_130279

theorem solve_equation (y : ℝ) : 
  5 * y^(1/4) - 3 * (y / y^(3/4)) = 9 + y^(1/4) ↔ y = 6561 :=
by sorry

end NUMINAMATH_CALUDE_solve_equation_l1302_130279


namespace NUMINAMATH_CALUDE_parabola_directrix_l1302_130232

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = (x^2 - 8*x + 12) / 16

/-- The directrix equation -/
def directrix (y : ℝ) : Prop := y = -1/2

/-- Theorem: The directrix of the given parabola is y = -1/2 -/
theorem parabola_directrix :
  ∀ (x y : ℝ), parabola x y → (∃ (d : ℝ), directrix d ∧ 
    d = y - (1/4) ∧ 
    ∀ (p : ℝ × ℝ), p.1 = x ∧ p.2 = y → 
      (p.1 - 4)^2 + (p.2 - 0)^2 = (p.2 - d)^2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l1302_130232


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l1302_130241

theorem arithmetic_calculations :
  ((-6) - 3 + (-7) - (-2) = -14) ∧
  ((-1)^2023 + 5 * (-2) - 12 / (-4) = -8) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l1302_130241


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1302_130298

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  1 / x + 4 / y ≥ 9 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + y₀ = 1 ∧ 1 / x₀ + 4 / y₀ = 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1302_130298


namespace NUMINAMATH_CALUDE_unique_solution_system_l1302_130243

theorem unique_solution_system (x y z : ℝ) : 
  y^3 - 6*x^2 + 12*x - 8 = 0 ∧
  z^3 - 6*y^2 + 12*y - 8 = 0 ∧
  x^3 - 6*z^2 + 12*z - 8 = 0 →
  x = 2 ∧ y = 2 ∧ z = 2 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_system_l1302_130243


namespace NUMINAMATH_CALUDE_garden_area_increase_l1302_130293

/-- Represents a rectangular garden with given length and width -/
structure RectGarden where
  length : ℝ
  width : ℝ

/-- Represents a square garden with given side length -/
structure SquareGarden where
  side : ℝ

/-- Calculates the perimeter of a rectangular garden -/
def RectGarden.perimeter (g : RectGarden) : ℝ :=
  2 * (g.length + g.width)

/-- Calculates the area of a rectangular garden -/
def RectGarden.area (g : RectGarden) : ℝ :=
  g.length * g.width

/-- Calculates the perimeter of a square garden -/
def SquareGarden.perimeter (g : SquareGarden) : ℝ :=
  4 * g.side

/-- Calculates the area of a square garden -/
def SquareGarden.area (g : SquareGarden) : ℝ :=
  g.side * g.side

/-- Theorem: Changing a 60 ft by 20 ft rectangular garden to a square garden 
    with the same perimeter increases the area by 400 square feet -/
theorem garden_area_increase :
  let rect := RectGarden.mk 60 20
  let square := SquareGarden.mk (rect.perimeter / 4)
  square.area - rect.area = 400 := by
  sorry


end NUMINAMATH_CALUDE_garden_area_increase_l1302_130293


namespace NUMINAMATH_CALUDE_power_three_multiplication_l1302_130271

theorem power_three_multiplication : 6^3 * 7^3 = 74088 := by
  sorry

end NUMINAMATH_CALUDE_power_three_multiplication_l1302_130271


namespace NUMINAMATH_CALUDE_least_divisible_by_7_11_13_l1302_130280

theorem least_divisible_by_7_11_13 : ∃ n : ℕ, n > 0 ∧ 7 ∣ n ∧ 11 ∣ n ∧ 13 ∣ n ∧ ∀ m : ℕ, m > 0 → 7 ∣ m → 11 ∣ m → 13 ∣ m → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_least_divisible_by_7_11_13_l1302_130280


namespace NUMINAMATH_CALUDE_square_sum_given_sum_and_product_l1302_130222

theorem square_sum_given_sum_and_product (a b : ℝ) 
  (h1 : a + b = 12) (h2 : a * b = 20) : a^2 + b^2 = 104 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_sum_and_product_l1302_130222


namespace NUMINAMATH_CALUDE_tan_pi_4_plus_2alpha_l1302_130212

open Real

theorem tan_pi_4_plus_2alpha (α : ℝ) : 
  π < α ∧ α < 3*π/2 →  -- α is in the third quadrant
  cos (2*α) = -3/5 → 
  tan (π/4 + 2*α) = -1/7 := by sorry

end NUMINAMATH_CALUDE_tan_pi_4_plus_2alpha_l1302_130212


namespace NUMINAMATH_CALUDE_x_value_proof_l1302_130258

theorem x_value_proof (x : ℚ) 
  (eq1 : 8 * x^2 + 7 * x - 1 = 0) 
  (eq2 : 24 * x^2 + 53 * x - 7 = 0) : 
  x = 1/8 := by sorry

end NUMINAMATH_CALUDE_x_value_proof_l1302_130258


namespace NUMINAMATH_CALUDE_stamp_collection_total_l1302_130273

/-- Represents a stamp collection with various categories of stamps. -/
structure StampCollection where
  foreign : ℕ
  old : ℕ
  both_foreign_and_old : ℕ
  neither_foreign_nor_old : ℕ

/-- Calculates the total number of stamps in the collection. -/
def total_stamps (collection : StampCollection) : ℕ :=
  collection.foreign + collection.old - collection.both_foreign_and_old + collection.neither_foreign_nor_old

/-- Theorem stating that the total number of stamps in the given collection is 220. -/
theorem stamp_collection_total :
  ∃ (collection : StampCollection),
    collection.foreign = 90 ∧
    collection.old = 70 ∧
    collection.both_foreign_and_old = 20 ∧
    collection.neither_foreign_nor_old = 60 ∧
    total_stamps collection = 220 := by
  sorry

end NUMINAMATH_CALUDE_stamp_collection_total_l1302_130273


namespace NUMINAMATH_CALUDE_profit_at_selling_price_185_l1302_130270

/-- Represents the daily sales volume as a function of price reduction --/
def sales_volume (x : ℝ) : ℝ := 4 * x + 100

/-- Represents the selling price as a function of price reduction --/
def selling_price (x : ℝ) : ℝ := 200 - x

/-- Represents the daily profit as a function of price reduction --/
def daily_profit (x : ℝ) : ℝ := (selling_price x - 100) * sales_volume x

theorem profit_at_selling_price_185 :
  ∃ x : ℝ, 
    daily_profit x = 13600 ∧ 
    selling_price x = 185 ∧ 
    selling_price x ≥ 150 := by sorry

end NUMINAMATH_CALUDE_profit_at_selling_price_185_l1302_130270


namespace NUMINAMATH_CALUDE_nested_expression_value_l1302_130239

def nested_expression : ℕ :=
  4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4)))))))

theorem nested_expression_value : nested_expression = 87380 := by
  sorry

end NUMINAMATH_CALUDE_nested_expression_value_l1302_130239


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1302_130259

theorem inequality_solution_set (x : ℝ) :
  (x^2 - |x| - 2 < 0) ↔ (-2 < x ∧ x < 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1302_130259


namespace NUMINAMATH_CALUDE_sum_a_b_equals_negative_one_l1302_130234

theorem sum_a_b_equals_negative_one (a b : ℝ) :
  |a - 2| + (b + 3)^2 = 0 → a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_sum_a_b_equals_negative_one_l1302_130234


namespace NUMINAMATH_CALUDE_first_four_super_nice_sum_l1302_130288

def is_super_nice (n : ℕ) : Prop :=
  n > 1 ∧
  (∃ (divisors : Finset ℕ),
    divisors = {d : ℕ | d ∣ n ∧ d ≠ 1 ∧ d ≠ n} ∧
    n = (Finset.prod divisors id) ∧
    n = (Finset.sum divisors id))

theorem first_four_super_nice_sum :
  ∃ (a b c d : ℕ),
    a < b ∧ b < c ∧ c < d ∧
    is_super_nice a ∧
    is_super_nice b ∧
    is_super_nice c ∧
    is_super_nice d ∧
    a + b + c + d = 45 :=
  sorry

end NUMINAMATH_CALUDE_first_four_super_nice_sum_l1302_130288


namespace NUMINAMATH_CALUDE_xiaoming_savings_l1302_130267

/-- Represents the number of coins in each pile -/
structure CoinCount where
  pile1_2cent : ℕ
  pile1_5cent : ℕ
  pile2_2cent : ℕ
  pile2_5cent : ℕ

/-- Calculates the total value of coins in cents -/
def totalValue (coins : CoinCount) : ℕ :=
  2 * (coins.pile1_2cent + coins.pile2_2cent) + 5 * (coins.pile1_5cent + coins.pile2_5cent)

theorem xiaoming_savings (coins : CoinCount) :
  coins.pile1_2cent = coins.pile1_5cent →
  2 * coins.pile2_2cent = 5 * coins.pile2_5cent →
  2 * coins.pile1_2cent + 5 * coins.pile1_5cent = 2 * coins.pile2_2cent + 5 * coins.pile2_5cent →
  500 ≤ totalValue coins →
  totalValue coins ≤ 600 →
  totalValue coins = 560 := by
  sorry

#check xiaoming_savings

end NUMINAMATH_CALUDE_xiaoming_savings_l1302_130267


namespace NUMINAMATH_CALUDE_green_ball_probability_l1302_130206

/-- Represents a container with red and green balls -/
structure Container where
  red : ℕ
  green : ℕ

/-- Calculates the probability of selecting a green ball from a container -/
def greenProbability (c : Container) : ℚ :=
  c.green / (c.red + c.green)

/-- Theorem: The probability of selecting a green ball is 53/96 -/
theorem green_ball_probability :
  let containers : List Container := [
    ⟨10, 5⟩,  -- Container I
    ⟨3, 5⟩,   -- Container II
    ⟨2, 6⟩,   -- Container III
    ⟨4, 4⟩    -- Container IV
  ]
  let totalContainers : ℕ := containers.length
  let containerProbability : ℚ := 1 / totalContainers
  let totalProbability : ℚ := (containers.map greenProbability).sum * containerProbability
  totalProbability = 53 / 96 := by
  sorry

end NUMINAMATH_CALUDE_green_ball_probability_l1302_130206


namespace NUMINAMATH_CALUDE_roots_cube_theorem_l1302_130284

theorem roots_cube_theorem (a b c d x₁ x₂ : ℝ) :
  (x₁^2 - (a + d)*x₁ + ad - bc = 0) →
  (x₂^2 - (a + d)*x₂ + ad - bc = 0) →
  (x₁^3)^2 - (a^3 + d^3 + 3*a*b*c + 3*b*c*d)*(x₁^3) + (a*d - b*c)^3 = 0 ∧
  (x₂^3)^2 - (a^3 + d^3 + 3*a*b*c + 3*b*c*d)*(x₂^3) + (a*d - b*c)^3 = 0 :=
by sorry

end NUMINAMATH_CALUDE_roots_cube_theorem_l1302_130284


namespace NUMINAMATH_CALUDE_sum_of_digits_1024_base5_l1302_130247

/-- Converts a natural number to its base 5 representation -/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

/-- Sums the elements of a list of natural numbers -/
def sumList (l : List ℕ) : ℕ :=
  sorry

theorem sum_of_digits_1024_base5 :
  sumList (toBase5 1024) = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_1024_base5_l1302_130247


namespace NUMINAMATH_CALUDE_system_solution_l1302_130252

theorem system_solution :
  let x₁ := Real.sqrt 2 / Real.sqrt 5
  let x₂ := -Real.sqrt 2 / Real.sqrt 5
  let y₁ := 2 * Real.sqrt 2 / Real.sqrt 5
  let y₂ := -2 * Real.sqrt 2 / Real.sqrt 5
  let condition₁ (x y : ℝ) := x^2 + y^2 ≤ 2
  let condition₂ (x y : ℝ) := x^4 - 8*x^2*y^2 + 16*y^4 - 20*x^2 - 80*y^2 + 100 = 0
  (condition₁ x₁ y₁ ∧ condition₂ x₁ y₁) ∧
  (condition₁ x₁ y₂ ∧ condition₂ x₁ y₂) ∧
  (condition₁ x₂ y₁ ∧ condition₂ x₂ y₁) ∧
  (condition₁ x₂ y₂ ∧ condition₂ x₂ y₂) :=
by sorry


end NUMINAMATH_CALUDE_system_solution_l1302_130252


namespace NUMINAMATH_CALUDE_diane_poker_loss_l1302_130262

/-- The total amount of money Diane lost in her poker game -/
def total_loss (initial_amount won_amount final_debt : ℝ) : ℝ :=
  initial_amount + won_amount + final_debt

/-- Theorem stating that Diane's total loss is $215 -/
theorem diane_poker_loss :
  let initial_amount : ℝ := 100
  let won_amount : ℝ := 65
  let final_debt : ℝ := 50
  total_loss initial_amount won_amount final_debt = 215 := by
  sorry

end NUMINAMATH_CALUDE_diane_poker_loss_l1302_130262


namespace NUMINAMATH_CALUDE_coefficient_of_x_cubed_in_expansion_l1302_130251

-- Define n such that 2^n = 64
def n : ℕ := 6

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Theorem statement
theorem coefficient_of_x_cubed_in_expansion :
  ∃ (coeff : ℤ), 
    (2^n = 64) ∧
    (coeff = (-1)^3 * binomial n 3 * 2^(n-3)) ∧
    (coeff = -160) := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_x_cubed_in_expansion_l1302_130251


namespace NUMINAMATH_CALUDE_expression_factorization_l1302_130213

theorem expression_factorization (b : ℝ) :
  (4 * b^3 + 126 * b^2 - 9) - (-9 * b^3 + 2 * b^2 - 9) = b^2 * (13 * b + 124) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l1302_130213


namespace NUMINAMATH_CALUDE_probability_not_red_marble_l1302_130266

theorem probability_not_red_marble (orange purple red yellow : ℕ) : 
  orange = 4 → purple = 7 → red = 8 → yellow = 5 → 
  (orange + purple + yellow) / (orange + purple + red + yellow) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_red_marble_l1302_130266


namespace NUMINAMATH_CALUDE_initial_boarders_l1302_130211

theorem initial_boarders (initial_boarders day_students new_boarders : ℕ) : 
  initial_boarders > 0 →
  day_students > 0 →
  new_boarders = 44 →
  initial_boarders * 12 = day_students * 5 →
  (initial_boarders + new_boarders) * 2 = day_students * 1 →
  initial_boarders = 220 := by
sorry

end NUMINAMATH_CALUDE_initial_boarders_l1302_130211


namespace NUMINAMATH_CALUDE_no_solution_to_equation_l1302_130245

theorem no_solution_to_equation : ¬∃ x : ℝ, x ≠ 0 ∧ x ≠ 5 ∧ (3*x^2 - 15*x) / (x^2 - 5*x) = x - 2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_to_equation_l1302_130245


namespace NUMINAMATH_CALUDE_A_intersect_B_l1302_130244

def A : Set ℝ := {0, 1, 2}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}

theorem A_intersect_B : A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l1302_130244


namespace NUMINAMATH_CALUDE_class_average_score_l1302_130265

theorem class_average_score (total_students : ℕ) 
  (assigned_day_percent : ℚ) (makeup_date_percent : ℚ) (later_date_percent : ℚ)
  (assigned_day_score : ℚ) (makeup_date_score : ℚ) (later_date_score : ℚ) :
  total_students = 100 →
  assigned_day_percent = 60 / 100 →
  makeup_date_percent = 30 / 100 →
  later_date_percent = 10 / 100 →
  assigned_day_score = 60 / 100 →
  makeup_date_score = 80 / 100 →
  later_date_score = 75 / 100 →
  (assigned_day_percent * assigned_day_score * total_students +
   makeup_date_percent * makeup_date_score * total_students +
   later_date_percent * later_date_score * total_students) / total_students = 675 / 1000 := by
  sorry

#eval (60 * 60 + 30 * 80 + 10 * 75) / 100  -- Expected output: 67.5

end NUMINAMATH_CALUDE_class_average_score_l1302_130265


namespace NUMINAMATH_CALUDE_triangle_theorem_l1302_130209

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the given conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a / Real.sin t.A = t.c / (Real.sqrt 3 * Real.cos t.C) ∧
  t.a + t.b = 6 ∧
  t.a * t.b * Real.cos t.C = 4

-- State the theorem
theorem triangle_theorem (t : Triangle) (h : triangle_conditions t) :
  t.C = π / 3 ∧ t.c = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l1302_130209


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1302_130200

theorem polynomial_simplification (a b c x : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) :
  (x - a)^4 / ((a - b) * (a - c)) + (x - b)^4 / ((b - a) * (b - c)) + (x - c)^4 / ((c - a) * (c - b)) =
  x^4 - 2*(a+b+c)*x^3 + (a^2+b^2+c^2+2*a*b+2*b*c+2*c*a)*x^2 - 2*a*b*c*x := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1302_130200


namespace NUMINAMATH_CALUDE_unique_b_for_two_integer_solutions_l1302_130264

theorem unique_b_for_two_integer_solutions :
  ∃! b : ℤ, ∃! (s : Finset ℤ), s.card = 2 ∧ ∀ x : ℤ, x ∈ s ↔ x^2 + b*x - 2 ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_b_for_two_integer_solutions_l1302_130264


namespace NUMINAMATH_CALUDE_tangent_line_equation_l1302_130231

-- Define the curve
def f (x : ℝ) : ℝ := 2 * x^2 + 3

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 4 * x

-- Define the point of tangency
def x₀ : ℝ := -1

-- Define the slope of the tangent line at x₀
def m : ℝ := f' x₀

-- Define a point on the curve at x₀
def p : ℝ × ℝ := (x₀, f x₀)

-- Theorem statement
theorem tangent_line_equation :
  ∀ x y : ℝ, y - p.2 = m * (x - p.1) ↔ y = -4 * x + 1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l1302_130231


namespace NUMINAMATH_CALUDE_parallel_line_equation_l1302_130285

/-- Given a line L passing through the point (1, 0) and parallel to the line x - 2y - 2 = 0,
    prove that the equation of L is x - 2y - 1 = 0 -/
theorem parallel_line_equation :
  ∀ (L : Set (ℝ × ℝ)),
  (∀ p ∈ L, ∃ x y : ℝ, p = (x, y) ∧ x - 2*y - 1 = 0) →
  (1, 0) ∈ L →
  (∀ p q : ℝ × ℝ, p ∈ L → q ∈ L → p.1 - q.1 = 2*(p.2 - q.2)) →
  ∀ p ∈ L, ∃ x y : ℝ, p = (x, y) ∧ x - 2*y - 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_equation_l1302_130285


namespace NUMINAMATH_CALUDE_unique_double_rectangle_with_perimeter_72_l1302_130210

/-- A rectangle with integer dimensions where one side is twice the other. -/
structure DoubleRectangle where
  shorter : ℕ
  longer : ℕ
  longer_is_double : longer = 2 * shorter

/-- The perimeter of a DoubleRectangle. -/
def perimeter (r : DoubleRectangle) : ℕ := 2 * (r.shorter + r.longer)

/-- The set of all DoubleRectangles with a perimeter of 72 inches. -/
def rectangles_with_perimeter_72 : Set DoubleRectangle :=
  {r : DoubleRectangle | perimeter r = 72}

theorem unique_double_rectangle_with_perimeter_72 :
  ∃! (r : DoubleRectangle), r ∈ rectangles_with_perimeter_72 := by
  sorry

#check unique_double_rectangle_with_perimeter_72

end NUMINAMATH_CALUDE_unique_double_rectangle_with_perimeter_72_l1302_130210


namespace NUMINAMATH_CALUDE_negative_product_of_negatives_l1302_130228

theorem negative_product_of_negatives (a b : ℝ) (h1 : a < b) (h2 : b < 0) : -a * b < 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_product_of_negatives_l1302_130228


namespace NUMINAMATH_CALUDE_bacon_suggestion_l1302_130246

/-- The number of students who suggested adding mashed potatoes -/
def mashed_potatoes : ℕ := 408

/-- The difference between the number of students who suggested mashed potatoes and bacon -/
def difference : ℕ := 366

/-- The number of students who suggested adding bacon -/
def bacon : ℕ := mashed_potatoes - difference

theorem bacon_suggestion :
  bacon = 42 :=
by sorry

end NUMINAMATH_CALUDE_bacon_suggestion_l1302_130246


namespace NUMINAMATH_CALUDE_square_minus_nine_l1302_130283

theorem square_minus_nine (x : ℤ) (h : x^2 = 1681) : (x + 3) * (x - 3) = 1672 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_nine_l1302_130283


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1302_130261

theorem polynomial_factorization (x : ℝ) : 
  9 * (x + 6) * (x + 12) * (x + 5) * (x + 15) - 8 * x^2 = 
  (3 * x^2 + 52 * x + 210) * (3 * x^2 + 56 * x + 222) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1302_130261


namespace NUMINAMATH_CALUDE_cubic_factorization_l1302_130205

theorem cubic_factorization (x : ℝ) : x^3 - 4*x = x*(x+2)*(x-2) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l1302_130205


namespace NUMINAMATH_CALUDE_anne_tom_age_sum_l1302_130235

theorem anne_tom_age_sum : 
  ∀ (A T : ℝ),
  A = T + 9 →
  A + 7 = 5 * (T - 3) →
  A + T = 24.5 :=
by
  sorry

end NUMINAMATH_CALUDE_anne_tom_age_sum_l1302_130235


namespace NUMINAMATH_CALUDE_sum_of_third_and_fourth_terms_l1302_130257

theorem sum_of_third_and_fourth_terms (a : ℕ → ℕ) (S : ℕ → ℕ) : 
  (∀ n : ℕ, S n = n^2 + n) → a 3 + a 4 = 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_third_and_fourth_terms_l1302_130257


namespace NUMINAMATH_CALUDE_boat_speed_problem_l1302_130227

/-- Proves that given a lake of width 60 miles, a boat traveling at 30 mph,
    and a waiting time of 3 hours for another boat to arrive,
    the speed of the second boat is 12 mph. -/
theorem boat_speed_problem (lake_width : ℝ) (janet_speed : ℝ) (waiting_time : ℝ) :
  lake_width = 60 →
  janet_speed = 30 →
  waiting_time = 3 →
  ∃ (sister_speed : ℝ),
    sister_speed = lake_width / (lake_width / janet_speed + waiting_time) ∧
    sister_speed = 12 := by sorry

end NUMINAMATH_CALUDE_boat_speed_problem_l1302_130227


namespace NUMINAMATH_CALUDE_special_decimal_value_l1302_130290

/-- A two-digit decimal number with specific digit placements -/
def special_decimal (n : ℚ) : Prop :=
  ∃ (w : ℕ), w < 100 ∧ n = w + 0.55

/-- The theorem stating that the special decimal number is equal to 50.05 -/
theorem special_decimal_value :
  ∀ n : ℚ, special_decimal n → n = 50.05 := by
sorry

end NUMINAMATH_CALUDE_special_decimal_value_l1302_130290


namespace NUMINAMATH_CALUDE_walnut_trees_planted_l1302_130287

/-- The number of walnut trees planted in a park --/
theorem walnut_trees_planted (initial_trees final_trees : ℕ) :
  initial_trees < final_trees →
  final_trees - initial_trees = final_trees - initial_trees :=
by sorry

end NUMINAMATH_CALUDE_walnut_trees_planted_l1302_130287
