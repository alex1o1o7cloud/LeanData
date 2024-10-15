import Mathlib

namespace NUMINAMATH_CALUDE_z_in_first_quadrant_l103_10377

def z : ℂ := (4 + 3*Complex.I) * (2 + Complex.I)

theorem z_in_first_quadrant : 
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = 1 :=
by sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_l103_10377


namespace NUMINAMATH_CALUDE_pure_imaginary_m_value_l103_10332

/-- A complex number is pure imaginary if its real part is zero and its imaginary part is non-zero -/
def IsPureImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- The complex number z defined in terms of a real number m -/
def z (m : ℝ) : ℂ := Complex.mk (m^2 + 2*m - 3) (m - 1)

theorem pure_imaginary_m_value :
  ∀ m : ℝ, IsPureImaginary (z m) → m = -3 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_m_value_l103_10332


namespace NUMINAMATH_CALUDE_principal_is_12000_l103_10352

/-- Calculates the principal amount given the interest rate, time, and total interest. -/
def calculate_principal (rate : ℚ) (time : ℕ) (interest : ℚ) : ℚ :=
  interest / (rate * time.cast / 100)

/-- Theorem stating that given the specified conditions, the principal amount is $12000. -/
theorem principal_is_12000 (rate : ℚ) (time : ℕ) (interest : ℚ) 
  (h_rate : rate = 12)
  (h_time : time = 3)
  (h_interest : interest = 4320) :
  calculate_principal rate time interest = 12000 := by
  sorry

#eval calculate_principal 12 3 4320

end NUMINAMATH_CALUDE_principal_is_12000_l103_10352


namespace NUMINAMATH_CALUDE_fixed_point_quadratic_fixed_point_satisfies_equation_l103_10360

/-- The fixed point on the graph of y = 9x^2 + kx - 5k -/
theorem fixed_point_quadratic (k : ℝ) : 
  9 * (-3)^2 + k * (-3) - 5 * k = 81 := by
  sorry

/-- The fixed point (-3, 81) satisfies the equation for all k -/
theorem fixed_point_satisfies_equation (k : ℝ) :
  ∃ (x y : ℝ), x = -3 ∧ y = 81 ∧ y = 9 * x^2 + k * x - 5 * k := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_quadratic_fixed_point_satisfies_equation_l103_10360


namespace NUMINAMATH_CALUDE_quotient_problem_l103_10396

theorem quotient_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) 
  (h1 : dividend = 165)
  (h2 : divisor = 18)
  (h3 : remainder = 3)
  (h4 : dividend = quotient * divisor + remainder) :
  quotient = 9 := by
  sorry

end NUMINAMATH_CALUDE_quotient_problem_l103_10396


namespace NUMINAMATH_CALUDE_binomial_half_variance_l103_10317

/-- A random variable following a binomial distribution -/
structure BinomialVariable where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The variance of a binomial variable -/
def variance (X : BinomialVariable) : ℝ := X.n * X.p * (1 - X.p)

/-- The main theorem -/
theorem binomial_half_variance (X : BinomialVariable) 
  (h2 : X.n = 8) (h3 : X.p = 3/5) : 
  variance X * (1/2)^2 = 12/25 := by sorry

end NUMINAMATH_CALUDE_binomial_half_variance_l103_10317


namespace NUMINAMATH_CALUDE_set_D_forms_triangle_l103_10313

-- Define a function to check if three lengths can form a triangle
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Theorem statement
theorem set_D_forms_triangle :
  can_form_triangle 10 10 5 := by
  sorry

end NUMINAMATH_CALUDE_set_D_forms_triangle_l103_10313


namespace NUMINAMATH_CALUDE_sheep_buying_problem_l103_10311

theorem sheep_buying_problem (x : ℝ) : 
  (∃ n : ℕ, n * 5 + 45 = x ∧ n * 7 + 3 = x) → (x - 45) / 5 = (x - 3) / 7 := by
  sorry

end NUMINAMATH_CALUDE_sheep_buying_problem_l103_10311


namespace NUMINAMATH_CALUDE_two_tangent_lines_l103_10323

/-- A line that intersects a parabola at exactly one point -/
structure TangentLine where
  slope : ℝ
  y_intercept : ℝ

/-- The parabola y^2 = 8x -/
def parabola (x y : ℝ) : Prop := y^2 = 8*x

/-- A point on the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The point M(2, 4) -/
def M : Point := ⟨2, 4⟩

/-- A line passes through a point -/
def passes_through (l : TangentLine) (p : Point) : Prop :=
  p.y = l.slope * p.x + l.y_intercept

/-- A line intersects the parabola at exactly one point -/
def intersects_once (l : TangentLine) : Prop :=
  ∃! (p : Point), passes_through l p ∧ parabola p.x p.y

/-- There are exactly two lines passing through M(2, 4) that intersect the parabola at exactly one point -/
theorem two_tangent_lines : ∃! (l1 l2 : TangentLine), 
  l1 ≠ l2 ∧ 
  passes_through l1 M ∧ 
  passes_through l2 M ∧ 
  intersects_once l1 ∧ 
  intersects_once l2 :=
sorry

end NUMINAMATH_CALUDE_two_tangent_lines_l103_10323


namespace NUMINAMATH_CALUDE_prob_green_face_specific_cube_l103_10342

/-- A cube with colored faces -/
structure ColoredCube where
  total_faces : ℕ
  green_faces : ℕ
  yellow_faces : ℕ

/-- The probability of rolling a green face on a colored cube -/
def prob_green_face (cube : ColoredCube) : ℚ :=
  cube.green_faces / cube.total_faces

/-- Theorem: The probability of rolling a green face on a cube with 5 green faces and 1 yellow face is 5/6 -/
theorem prob_green_face_specific_cube :
  let cube : ColoredCube := ⟨6, 5, 1⟩
  prob_green_face cube = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_prob_green_face_specific_cube_l103_10342


namespace NUMINAMATH_CALUDE_notebook_cost_per_page_l103_10355

/-- Calculates the cost per page in cents given the number of notebooks, pages per notebook, and total cost in dollars. -/
def cost_per_page (notebooks : ℕ) (pages_per_notebook : ℕ) (total_cost_dollars : ℕ) : ℚ :=
  (total_cost_dollars * 100) / (notebooks * pages_per_notebook)

/-- Proves that for 2 notebooks with 50 pages each, purchased for $5, the cost per page is 5 cents. -/
theorem notebook_cost_per_page :
  cost_per_page 2 50 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_notebook_cost_per_page_l103_10355


namespace NUMINAMATH_CALUDE_max_gcd_14n_plus_5_9n_plus_2_l103_10373

theorem max_gcd_14n_plus_5_9n_plus_2 :
  (∃ (k : ℕ+), ∀ (n : ℕ+), Nat.gcd (14 * n + 5) (9 * n + 2) ≤ k) ∧
  (∃ (n : ℕ+), Nat.gcd (14 * n + 5) (9 * n + 2) = 4) := by
  sorry

end NUMINAMATH_CALUDE_max_gcd_14n_plus_5_9n_plus_2_l103_10373


namespace NUMINAMATH_CALUDE_number_of_children_l103_10369

/-- Given a person with some children and money to distribute, prove the number of children. -/
theorem number_of_children (total_money : ℕ) (share_d_and_e : ℕ) (children : List String) : 
  total_money = 12000 → share_d_and_e = 4800 → 
  children = ["a", "b", "c", "d", "e"] → 
  children.length = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_of_children_l103_10369


namespace NUMINAMATH_CALUDE_solution_set_abs_inequality_l103_10389

theorem solution_set_abs_inequality (x : ℝ) :
  (Set.Icc 1 3 : Set ℝ) = {x | |2 - x| ≤ 1} :=
sorry

end NUMINAMATH_CALUDE_solution_set_abs_inequality_l103_10389


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l103_10395

theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 4/7
  let a₂ : ℚ := 12/7
  let a₃ : ℚ := 36/7
  let r : ℚ := a₂ / a₁
  (∀ n : ℕ, n ≥ 1 → (a₁ * r^(n-1) : ℚ) = 4/7 * 3^(n-1)) →
  r = 3 :=
by sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l103_10395


namespace NUMINAMATH_CALUDE_krista_hens_count_l103_10359

def egg_price_per_dozen : ℚ := 3
def total_sales : ℚ := 120
def weeks : ℕ := 4
def eggs_per_hen_per_week : ℕ := 12

def num_hens : ℕ := 10

theorem krista_hens_count :
  (egg_price_per_dozen * (total_sales / egg_price_per_dozen) = 
   ↑num_hens * ↑eggs_per_hen_per_week * ↑weeks) := by sorry

end NUMINAMATH_CALUDE_krista_hens_count_l103_10359


namespace NUMINAMATH_CALUDE_paiges_pencils_l103_10357

/-- Paige's pencil problem -/
theorem paiges_pencils (P : ℕ) : 
  P - (P - 15) / 4 + 16 - 12 + 23 = 84 → P = 71 := by
  sorry

end NUMINAMATH_CALUDE_paiges_pencils_l103_10357


namespace NUMINAMATH_CALUDE_notebook_cost_l103_10393

theorem notebook_cost (total_cost cover_cost notebook_cost : ℚ) : 
  total_cost = 2.4 →
  notebook_cost = cover_cost + 2 →
  total_cost = notebook_cost + cover_cost →
  notebook_cost = 2.2 := by
sorry

end NUMINAMATH_CALUDE_notebook_cost_l103_10393


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l103_10376

theorem polynomial_division_remainder : ∃ q : Polynomial ℂ, 
  (X^4 - 1) * (X^3 - 1) = (X^2 + 1) * q + (2 + X) := by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l103_10376


namespace NUMINAMATH_CALUDE_max_profit_thermos_l103_10398

/-- Thermos cup prices and quantities -/
structure ThermosCups where
  price_a : ℝ
  price_b : ℝ
  quantity_a : ℕ
  quantity_b : ℕ

/-- Conditions for thermos cup problem -/
def thermos_conditions (t : ThermosCups) : Prop :=
  t.price_b = t.price_a + 10 ∧
  600 / t.price_b = 480 / t.price_a ∧
  t.quantity_a + t.quantity_b = 120 ∧
  t.quantity_a ≥ t.quantity_b / 2 ∧
  t.quantity_a ≤ t.quantity_b

/-- Profit calculation -/
def profit (t : ThermosCups) : ℝ :=
  (t.price_a - 30) * t.quantity_a + (t.price_b * 0.9 - 30) * t.quantity_b

/-- Theorem: Maximum profit for thermos cup sales -/
theorem max_profit_thermos :
  ∃ t : ThermosCups, thermos_conditions t ∧
    profit t = 1600 ∧
    (∀ t' : ThermosCups, thermos_conditions t' → profit t' ≤ profit t) :=
  sorry

end NUMINAMATH_CALUDE_max_profit_thermos_l103_10398


namespace NUMINAMATH_CALUDE_angle_range_theorem_l103_10346

theorem angle_range_theorem (θ : Real) 
  (h1 : 0 ≤ θ) (h2 : θ < 2 * Real.pi) 
  (h3 : Real.sin θ ^ 3 - Real.cos θ ^ 3 ≥ Real.cos θ - Real.sin θ) : 
  Real.pi / 4 ≤ θ ∧ θ ≤ 5 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_angle_range_theorem_l103_10346


namespace NUMINAMATH_CALUDE_uncle_welly_roses_l103_10328

/-- The number of roses Uncle Welly planted two days ago -/
def roses_two_days_ago : ℕ := 50

/-- The number of roses Uncle Welly planted yesterday -/
def roses_yesterday : ℕ := roses_two_days_ago + 20

/-- The number of roses Uncle Welly planted today -/
def roses_today : ℕ := 2 * roses_two_days_ago

/-- The total number of roses Uncle Welly planted in his vacant lot -/
def total_roses : ℕ := roses_two_days_ago + roses_yesterday + roses_today

theorem uncle_welly_roses : total_roses = 220 := by
  sorry

end NUMINAMATH_CALUDE_uncle_welly_roses_l103_10328


namespace NUMINAMATH_CALUDE_congruence_systems_solvability_l103_10314

theorem congruence_systems_solvability :
  (∃ x : ℤ, x ≡ 2 [ZMOD 3] ∧ x ≡ 6 [ZMOD 14]) ∧
  (¬ ∃ x : ℤ, x ≡ 5 [ZMOD 12] ∧ x ≡ 7 [ZMOD 15]) ∧
  (∃ x : ℤ, x ≡ 10 [ZMOD 12] ∧ x ≡ 16 [ZMOD 21]) :=
by sorry

end NUMINAMATH_CALUDE_congruence_systems_solvability_l103_10314


namespace NUMINAMATH_CALUDE_consecutive_numbers_digit_sum_l103_10307

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def sum_of_digits_range (start : ℕ) (count : ℕ) : ℕ :=
  List.range count |>.map (fun i => sum_of_digits (start + i)) |>.sum

theorem consecutive_numbers_digit_sum :
  ∃! start : ℕ, sum_of_digits_range start 10 = 145 ∧ start ≥ 100 ∧ start < 1000 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_numbers_digit_sum_l103_10307


namespace NUMINAMATH_CALUDE_car_original_price_verify_car_price_l103_10383

/-- Calculates the original price of a car given the final price after discounts, taxes, and fees. -/
theorem car_original_price (final_price : ℝ) (doc_fee : ℝ) 
  (discount1 discount2 discount3 tax_rate : ℝ) : ℝ :=
  let remaining_after_discounts := (1 - discount1) * (1 - discount2) * (1 - discount3)
  let price_with_tax := remaining_after_discounts * (1 + tax_rate)
  (final_price - doc_fee) / price_with_tax

/-- Proves that the calculated original price satisfies the given conditions. -/
theorem verify_car_price : 
  let original_price := car_original_price 7500 200 0.15 0.20 0.25 0.10
  0.561 * original_price + 200 = 7500 := by
  sorry

end NUMINAMATH_CALUDE_car_original_price_verify_car_price_l103_10383


namespace NUMINAMATH_CALUDE_min_dot_product_planar_vectors_l103_10341

/-- Given planar vectors a and b satisfying |2a - b| ≤ 3, 
    the minimum value of a · b is -9/8 -/
theorem min_dot_product_planar_vectors 
  (a b : ℝ × ℝ) 
  (h : ‖(2 : ℝ) • a - b‖ ≤ 3) : 
  ∃ (m : ℝ), m = -9/8 ∧ ∀ (x : ℝ), x = a.1 * b.1 + a.2 * b.2 → m ≤ x :=
sorry

end NUMINAMATH_CALUDE_min_dot_product_planar_vectors_l103_10341


namespace NUMINAMATH_CALUDE_evaluate_expression_l103_10312

theorem evaluate_expression : (0.5^4 / 0.05^3) + 3 = 503 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l103_10312


namespace NUMINAMATH_CALUDE_floor_equality_l103_10335

theorem floor_equality (n : ℤ) (h : n > 2) :
  ⌊(n * (n + 1) : ℚ) / (4 * n - 2)⌋ = ⌊(n + 1 : ℚ) / 4⌋ := by
  sorry

end NUMINAMATH_CALUDE_floor_equality_l103_10335


namespace NUMINAMATH_CALUDE_max_got_more_candy_l103_10372

/-- The number of candy pieces Frankie got -/
def frankies_candy : ℕ := 74

/-- The number of candy pieces Max got -/
def maxs_candy : ℕ := 92

/-- The difference in candy pieces between Max and Frankie -/
def candy_difference : ℕ := maxs_candy - frankies_candy

theorem max_got_more_candy : candy_difference = 18 := by
  sorry

end NUMINAMATH_CALUDE_max_got_more_candy_l103_10372


namespace NUMINAMATH_CALUDE_time_per_furniture_piece_l103_10303

theorem time_per_furniture_piece (chairs tables total_time : ℕ) 
  (h1 : chairs = 4)
  (h2 : tables = 2)
  (h3 : total_time = 48) : 
  total_time / (chairs + tables) = 8 := by
  sorry

end NUMINAMATH_CALUDE_time_per_furniture_piece_l103_10303


namespace NUMINAMATH_CALUDE_line_bisects_circle_l103_10379

/-- The equation of a circle in the xy-plane -/
def Circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + 1 = 0

/-- The equation of a line in the xy-plane -/
def Line (x y : ℝ) : Prop :=
  x - y + 1 = 0

/-- The center of the circle -/
def center : ℝ × ℝ := (1, 2)

/-- Theorem stating that the line bisects the circle -/
theorem line_bisects_circle :
  ∀ x y : ℝ, Circle x y → Line x y → (x, y) = center :=
sorry

end NUMINAMATH_CALUDE_line_bisects_circle_l103_10379


namespace NUMINAMATH_CALUDE_greatest_value_quadratic_inequality_l103_10321

theorem greatest_value_quadratic_inequality :
  ∀ a : ℝ, a^2 - 12*a + 35 ≤ 0 → a ≤ 7 ∧
  ∃ a : ℝ, a^2 - 12*a + 35 ≤ 0 ∧ a = 7 :=
by sorry

end NUMINAMATH_CALUDE_greatest_value_quadratic_inequality_l103_10321


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l103_10358

theorem quadratic_equation_solution :
  let a : ℝ := 2
  let b : ℝ := -5
  let c : ℝ := 3
  let x₁ : ℝ := 3/2
  let x₂ : ℝ := 1
  (a * x₁^2 + b * x₁ + c = 0) ∧ (a * x₂^2 + b * x₂ + c = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l103_10358


namespace NUMINAMATH_CALUDE_infinite_cube_differences_l103_10363

theorem infinite_cube_differences (n : ℕ+) : 
  (∃ p : ℕ+, 3 * p + 1 = (n + 1)^3 - n^3) ∧ 
  (∃ q : ℕ+, 5 * q + 1 = (5 * n + 1)^3 - (5 * n)^3) := by
  sorry

end NUMINAMATH_CALUDE_infinite_cube_differences_l103_10363


namespace NUMINAMATH_CALUDE_f_extrema_l103_10381

open Real

noncomputable def f (x : ℝ) : ℝ := cos x + (x + 1) * sin x + 1

theorem f_extrema :
  ∃ (min max : ℝ),
    (∀ x ∈ Set.Icc 0 (2 * π), f x ≥ min) ∧
    (∃ x ∈ Set.Icc 0 (2 * π), f x = min) ∧
    (∀ x ∈ Set.Icc 0 (2 * π), f x ≤ max) ∧
    (∃ x ∈ Set.Icc 0 (2 * π), f x = max) ∧
    min = -3 * π / 2 ∧
    max = π / 2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_f_extrema_l103_10381


namespace NUMINAMATH_CALUDE_bobby_has_more_books_l103_10354

/-- Given that Bobby has 142 books and Kristi has 78 books, 
    prove that Bobby has 64 more books than Kristi. -/
theorem bobby_has_more_books : 
  let bobby_books : ℕ := 142
  let kristi_books : ℕ := 78
  bobby_books - kristi_books = 64 := by sorry

end NUMINAMATH_CALUDE_bobby_has_more_books_l103_10354


namespace NUMINAMATH_CALUDE_total_spider_legs_l103_10382

/-- The number of spiders in Christopher's room -/
def num_spiders : ℕ := 4

/-- The number of legs each spider has -/
def legs_per_spider : ℕ := 8

/-- Theorem: The total number of spider legs is 32 -/
theorem total_spider_legs : num_spiders * legs_per_spider = 32 := by
  sorry

end NUMINAMATH_CALUDE_total_spider_legs_l103_10382


namespace NUMINAMATH_CALUDE_model2_best_fit_l103_10324

/-- Represents a regression model with its coefficient of determination -/
structure RegressionModel where
  name : String
  r_squared : Float

/-- Determines if a model has the best fitting effect among a list of models -/
def has_best_fit (model : RegressionModel) (models : List RegressionModel) : Prop :=
  ∀ m ∈ models, model.r_squared ≥ m.r_squared

/-- The list of regression models with their R² values -/
def regression_models : List RegressionModel := [
  ⟨"Model 1", 0.78⟩,
  ⟨"Model 2", 0.85⟩,
  ⟨"Model 3", 0.61⟩,
  ⟨"Model 4", 0.31⟩
]

/-- Theorem stating that Model 2 has the best fitting effect -/
theorem model2_best_fit :
  ∃ model ∈ regression_models, model.name = "Model 2" ∧ has_best_fit model regression_models :=
by
  sorry

end NUMINAMATH_CALUDE_model2_best_fit_l103_10324


namespace NUMINAMATH_CALUDE_line_x_intercept_l103_10306

/-- Given a line passing through the point (3, 4) with slope 2, its x-intercept is 1. -/
theorem line_x_intercept : 
  ∀ (f : ℝ → ℝ), 
  (∀ x, f x = 2 * x + (4 - 2 * 3)) →  -- Line equation derived from point-slope form
  f 4 = 3 →                           -- Line passes through (3, 4)
  f 0 = 1 :=                          -- x-intercept is at (1, 0)
by
  sorry

end NUMINAMATH_CALUDE_line_x_intercept_l103_10306


namespace NUMINAMATH_CALUDE_class_artworks_l103_10394

theorem class_artworks (num_students : ℕ) (artworks_group1 : ℕ) (artworks_group2 : ℕ) : 
  num_students = 10 →
  artworks_group1 = 3 →
  artworks_group2 = 4 →
  (num_students / 2 : ℕ) * artworks_group1 + (num_students / 2 : ℕ) * artworks_group2 = 35 := by
  sorry

end NUMINAMATH_CALUDE_class_artworks_l103_10394


namespace NUMINAMATH_CALUDE_jesse_pencils_l103_10327

theorem jesse_pencils (initial : ℕ) (given : ℕ) (remaining : ℕ) : 
  given = 44 → remaining = 34 → initial = given + remaining :=
by
  sorry

end NUMINAMATH_CALUDE_jesse_pencils_l103_10327


namespace NUMINAMATH_CALUDE_elephant_pig_equivalence_l103_10320

variable (P Q : Prop)

theorem elephant_pig_equivalence :
  (P → Q) →
  ((P → Q) ↔ (¬Q → ¬P)) ∧
  ((P → Q) ↔ (¬P ∨ Q)) ∧
  ¬((P → Q) ↔ (Q → P)) :=
by sorry

end NUMINAMATH_CALUDE_elephant_pig_equivalence_l103_10320


namespace NUMINAMATH_CALUDE_female_officers_count_l103_10322

theorem female_officers_count (total_on_duty : ℕ) (female_on_duty_ratio : ℚ) 
  (female_duty_percentage : ℚ) (h1 : total_on_duty = 300) 
  (h2 : female_on_duty_ratio = 1/2) (h3 : female_duty_percentage = 15/100) : 
  ∃ (total_female : ℕ), total_female = 1000 ∧ 
  (total_on_duty : ℚ) * female_on_duty_ratio * (1/female_duty_percentage) = total_female := by
sorry

end NUMINAMATH_CALUDE_female_officers_count_l103_10322


namespace NUMINAMATH_CALUDE_max_attendance_l103_10378

-- Define the days of the week
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday

-- Define the people
inductive Person
  | Anna
  | Bill
  | Carl
  | Diana

-- Define the availability function
def isAvailable (p : Person) (d : Day) : Bool :=
  match p, d with
  | Person.Anna, Day.Monday => false
  | Person.Anna, Day.Tuesday => true
  | Person.Anna, Day.Wednesday => false
  | Person.Anna, Day.Thursday => true
  | Person.Anna, Day.Friday => false
  | Person.Bill, Day.Monday => true
  | Person.Bill, Day.Tuesday => false
  | Person.Bill, Day.Wednesday => true
  | Person.Bill, Day.Thursday => false
  | Person.Bill, Day.Friday => false
  | Person.Carl, Day.Monday => false
  | Person.Carl, Day.Tuesday => false
  | Person.Carl, Day.Wednesday => true
  | Person.Carl, Day.Thursday => false
  | Person.Carl, Day.Friday => true
  | Person.Diana, Day.Monday => true
  | Person.Diana, Day.Tuesday => true
  | Person.Diana, Day.Wednesday => false
  | Person.Diana, Day.Thursday => true
  | Person.Diana, Day.Friday => false

-- Define the function to count available people on a given day
def countAvailable (d : Day) : Nat :=
  (List.filter (fun p => isAvailable p d) [Person.Anna, Person.Bill, Person.Carl, Person.Diana]).length

-- Theorem statement
theorem max_attendance :
  (∀ d : Day, countAvailable d ≤ 2) ∧
  (countAvailable Day.Monday = 2) ∧
  (countAvailable Day.Tuesday = 2) ∧
  (countAvailable Day.Wednesday = 2) ∧
  (countAvailable Day.Thursday < 2) ∧
  (countAvailable Day.Friday < 2) :=
sorry

end NUMINAMATH_CALUDE_max_attendance_l103_10378


namespace NUMINAMATH_CALUDE_thirteen_to_six_div_three_l103_10365

theorem thirteen_to_six_div_three (x : ℕ) : 13^6 / 13^3 = 2197 := by
  sorry

end NUMINAMATH_CALUDE_thirteen_to_six_div_three_l103_10365


namespace NUMINAMATH_CALUDE_shaded_area_sum_l103_10399

/-- The sum of the areas of two pie-shaped regions in a circle with an inscribed square --/
theorem shaded_area_sum (d : ℝ) (h : d = 16) : 
  let r := d / 2
  let sector_area := 2 * (π * r^2 * (45 / 360))
  let triangle_area := 2 * (1 / 2 * r^2)
  sector_area - triangle_area = 32 * π - 64 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_sum_l103_10399


namespace NUMINAMATH_CALUDE_trout_division_l103_10348

theorem trout_division (total_trout : ℕ) (num_people : ℕ) (trout_per_person : ℕ) : 
  total_trout = 52 → num_people = 4 → trout_per_person = total_trout / num_people → trout_per_person = 13 := by
  sorry

end NUMINAMATH_CALUDE_trout_division_l103_10348


namespace NUMINAMATH_CALUDE_min_value_of_f_in_interval_l103_10351

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the interval [-1, 2]
def interval : Set ℝ := Set.Icc (-1) 2

-- State the theorem
theorem min_value_of_f_in_interval : 
  ∃ (x : ℝ), x ∈ interval ∧ f x = -2 ∧ ∀ (y : ℝ), y ∈ interval → f y ≥ f x :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_in_interval_l103_10351


namespace NUMINAMATH_CALUDE_function_equation_implies_constant_l103_10364

/-- A function satisfying the given functional equation is constant -/
theorem function_equation_implies_constant
  (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, 2 * f x = f (x + y) + f (x + 2 * y)) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c :=
sorry

end NUMINAMATH_CALUDE_function_equation_implies_constant_l103_10364


namespace NUMINAMATH_CALUDE_inscribed_sphere_radius_in_pyramid_inscribed_sphere_radius_is_correct_l103_10391

/-- The radius of the sphere inscribed in a pyramid PMKC, where:
  - PABCD is a regular quadrilateral pyramid
  - PO is the height of the pyramid and equals 4
  - ABCD is the base of the pyramid with side length 6
  - M is the midpoint of BC
  - K is the midpoint of CD
-/
theorem inscribed_sphere_radius_in_pyramid (PO : ℝ) (side_length : ℝ) : ℝ :=
  let PMKC_volume := (1/8) * (1/3) * side_length^2 * PO
  let CMK_area := (1/4) * (1/2) * side_length^2
  let ON := (1/4) * side_length * Real.sqrt 2
  let PN := Real.sqrt ((PO^2) + (ON^2))
  let OK := (1/2) * side_length
  let PK := Real.sqrt ((PO^2) + (OK^2))
  let PKC_area := (1/2) * OK * PK
  let PMK_area := (1/2) * (side_length * Real.sqrt 2 / 2) * PN
  let surface_area := 2 * PKC_area + PMK_area + CMK_area
  let radius := 3 * PMKC_volume / surface_area
  12 / (13 + Real.sqrt 41)

theorem inscribed_sphere_radius_is_correct (PO : ℝ) (side_length : ℝ)
  (h1 : PO = 4)
  (h2 : side_length = 6) :
  inscribed_sphere_radius_in_pyramid PO side_length = 12 / (13 + Real.sqrt 41) := by
  sorry

#check inscribed_sphere_radius_is_correct

end NUMINAMATH_CALUDE_inscribed_sphere_radius_in_pyramid_inscribed_sphere_radius_is_correct_l103_10391


namespace NUMINAMATH_CALUDE_only_setB_proportional_l103_10301

/-- A set of four line segments --/
structure LineSegmentSet where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Checks if a set of line segments is proportional --/
def isProportional (s : LineSegmentSet) : Prop :=
  s.a * s.d = s.b * s.c

/-- The given sets of line segments --/
def setA : LineSegmentSet := ⟨3, 4, 5, 6⟩
def setB : LineSegmentSet := ⟨5, 15, 2, 6⟩
def setC : LineSegmentSet := ⟨4, 8, 3, 5⟩
def setD : LineSegmentSet := ⟨8, 4, 1, 3⟩

/-- Theorem stating that only set B is proportional --/
theorem only_setB_proportional :
  ¬ isProportional setA ∧
  isProportional setB ∧
  ¬ isProportional setC ∧
  ¬ isProportional setD :=
sorry

end NUMINAMATH_CALUDE_only_setB_proportional_l103_10301


namespace NUMINAMATH_CALUDE_sqrt_15_minus_1_range_l103_10388

theorem sqrt_15_minus_1_range : 2 < Real.sqrt 15 - 1 ∧ Real.sqrt 15 - 1 < 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_15_minus_1_range_l103_10388


namespace NUMINAMATH_CALUDE_karen_average_speed_l103_10362

/-- Calculates the time difference in hours between two times given in hours and minutes -/
def timeDifference (start_hour start_minute end_hour end_minute : ℕ) : ℚ :=
  (end_hour - start_hour : ℚ) + (end_minute - start_minute : ℚ) / 60

/-- Calculates the average speed given distance and time -/
def averageSpeed (distance : ℚ) (time : ℚ) : ℚ :=
  distance / time

theorem karen_average_speed :
  let start_time : ℕ × ℕ := (9, 40)  -- (hour, minute)
  let end_time : ℕ × ℕ := (13, 20)   -- (hour, minute)
  let distance : ℚ := 198
  let time := timeDifference start_time.1 start_time.2 end_time.1 end_time.2
  averageSpeed distance time = 54 := by sorry

end NUMINAMATH_CALUDE_karen_average_speed_l103_10362


namespace NUMINAMATH_CALUDE_parabola_x_intercepts_l103_10336

theorem parabola_x_intercepts :
  let f (x : ℝ) := 3 * x^2 + 5 * x - 8
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) ∧
  (∀ (x₁ x₂ x₃ : ℝ), f x₁ = 0 → f x₂ = 0 → f x₃ = 0 → x₁ = x₂ ∨ x₁ = x₃ ∨ x₂ = x₃) := by
  sorry

end NUMINAMATH_CALUDE_parabola_x_intercepts_l103_10336


namespace NUMINAMATH_CALUDE_water_tank_problem_l103_10334

/-- A water tank problem during the rainy season -/
theorem water_tank_problem (tank_capacity : ℝ) (initial_fill_fraction : ℝ) 
  (day1_collection : ℝ) (day2_extra : ℝ) :
  tank_capacity = 100 →
  initial_fill_fraction = 2/5 →
  day1_collection = 15 →
  day2_extra = 5 →
  let initial_water := initial_fill_fraction * tank_capacity
  let day1_total := initial_water + day1_collection
  let day2_collection := day1_collection + day2_extra
  let day2_total := day1_total + day2_collection
  let day3_collection := tank_capacity - day2_total
  day3_collection = 25 := by
sorry


end NUMINAMATH_CALUDE_water_tank_problem_l103_10334


namespace NUMINAMATH_CALUDE_det_of_specific_matrix_l103_10353

theorem det_of_specific_matrix :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![8, 4; -2, 3]
  Matrix.det A = 32 := by
sorry

end NUMINAMATH_CALUDE_det_of_specific_matrix_l103_10353


namespace NUMINAMATH_CALUDE_ice_cream_bill_l103_10339

/-- The final bill for four ice cream sundaes with a 20% tip -/
theorem ice_cream_bill (alicia_cost brant_cost josh_cost yvette_cost : ℚ) 
  (h1 : alicia_cost = 7.5)
  (h2 : brant_cost = 10)
  (h3 : josh_cost = 8.5)
  (h4 : yvette_cost = 9)
  (tip_rate : ℚ)
  (h5 : tip_rate = 0.2) :
  alicia_cost + brant_cost + josh_cost + yvette_cost + 
  (alicia_cost + brant_cost + josh_cost + yvette_cost) * tip_rate = 42 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_bill_l103_10339


namespace NUMINAMATH_CALUDE_no_integer_solutions_l103_10349

theorem no_integer_solutions : ¬∃ (x y : ℤ), x^4 + y^2 = 4*y + 4 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l103_10349


namespace NUMINAMATH_CALUDE_fruit_arrangement_problem_l103_10309

def number_of_arrangements (n : ℕ) (a b c d : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial a * Nat.factorial b * Nat.factorial c * Nat.factorial d)

theorem fruit_arrangement_problem : number_of_arrangements 10 4 3 2 1 = 12600 := by
  sorry

end NUMINAMATH_CALUDE_fruit_arrangement_problem_l103_10309


namespace NUMINAMATH_CALUDE_speed_increase_for_time_reduction_car_speed_increase_l103_10343

/-- Calculates the required speed increase for a car to reduce its travel time --/
theorem speed_increase_for_time_reduction 
  (initial_speed : ℝ) 
  (distance : ℝ) 
  (time_reduction : ℝ) : ℝ :=
  let initial_time := distance / initial_speed
  let final_time := initial_time - time_reduction
  let final_speed := distance / final_time
  final_speed - initial_speed

/-- Proves that a car traveling at 60 km/h needs to increase its speed by 60 km/h
    to travel 1 km in half a minute less time --/
theorem car_speed_increase : 
  speed_increase_for_time_reduction 60 1 (1/120) = 60 := by
  sorry

end NUMINAMATH_CALUDE_speed_increase_for_time_reduction_car_speed_increase_l103_10343


namespace NUMINAMATH_CALUDE_gcd_digits_bound_l103_10325

theorem gcd_digits_bound (a b : ℕ) (ha : 10^6 ≤ a ∧ a < 10^7) (hb : 10^6 ≤ b ∧ b < 10^7)
  (hlcm : 10^11 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^12) :
  Nat.gcd a b < 10^3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_digits_bound_l103_10325


namespace NUMINAMATH_CALUDE_bert_spending_l103_10330

theorem bert_spending (n : ℚ) : 
  (1/2 * ((3/4 * n) - 9)) = 15 → n = 52 := by
  sorry

end NUMINAMATH_CALUDE_bert_spending_l103_10330


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_seven_l103_10331

theorem sqrt_sum_equals_seven (y : ℝ) 
  (h : Real.sqrt (64 - y^2) - Real.sqrt (36 - y^2) = 4) : 
  Real.sqrt (64 - y^2) + Real.sqrt (36 - y^2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_seven_l103_10331


namespace NUMINAMATH_CALUDE_no_real_roots_l103_10350

theorem no_real_roots : ¬∃ x : ℝ, Real.sqrt (x + 4) - Real.sqrt (x - 3) + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l103_10350


namespace NUMINAMATH_CALUDE_ten_thousand_equals_10000_l103_10367

theorem ten_thousand_equals_10000 : (10 * 1000 : ℕ) = 10000 := by
  sorry

end NUMINAMATH_CALUDE_ten_thousand_equals_10000_l103_10367


namespace NUMINAMATH_CALUDE_sphere_volume_ratio_l103_10375

theorem sphere_volume_ratio (r R : ℝ) (h : r > 0) (H : R > 0) : 
  (4 * π * r^2) / (4 * π * R^2) = 4 / 9 → 
  ((4 / 3) * π * r^3) / ((4 / 3) * π * R^3) = 8 / 27 := by
sorry

end NUMINAMATH_CALUDE_sphere_volume_ratio_l103_10375


namespace NUMINAMATH_CALUDE_investment_growth_period_l103_10397

/-- The annual interest rate as a real number between 0 and 1 -/
def interest_rate : ℝ := 0.341

/-- The target multiple of the initial investment -/
def target_multiple : ℝ := 3

/-- The function to calculate the investment value after n years -/
def investment_value (n : ℕ) : ℝ := (1 + interest_rate) ^ n

/-- The smallest investment period in years -/
def smallest_period : ℕ := 4

theorem investment_growth_period :
  (∀ k : ℕ, k < smallest_period → investment_value k ≤ target_multiple) ∧
  target_multiple < investment_value smallest_period :=
sorry

end NUMINAMATH_CALUDE_investment_growth_period_l103_10397


namespace NUMINAMATH_CALUDE_distance_AB_l103_10386

/-- The equation of the line -/
def line_equation (x y : ℝ) : Prop := x - 2*y + 6 = 0

/-- The x-coordinate of point A (x-axis intersection) -/
def point_A : ℝ := -6

/-- The y-coordinate of point B (y-axis intersection) -/
def point_B : ℝ := 3

/-- Theorem stating that the distance between points A and B is 3√5 -/
theorem distance_AB :
  let A : ℝ × ℝ := (point_A, 0)
  let B : ℝ × ℝ := (0, point_B)
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 3 * Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_distance_AB_l103_10386


namespace NUMINAMATH_CALUDE_factorization_of_polynomial_l103_10310

theorem factorization_of_polynomial (x : ℝ) :
  29 * 40 * x^4 + 64 = 29 * 40 * ((x^2 - 4*x + 8) * (x^2 + 4*x + 8)) :=
by sorry

end NUMINAMATH_CALUDE_factorization_of_polynomial_l103_10310


namespace NUMINAMATH_CALUDE_first_player_winning_strategy_l103_10392

/-- A rectangular game board -/
structure GameBoard where
  m : ℝ
  n : ℝ

/-- A penny with radius 1 -/
structure Penny where
  center : ℝ × ℝ

/-- The game state -/
structure GameState where
  board : GameBoard
  pennies : List Penny

/-- Check if a new penny placement is valid -/
def is_valid_placement (state : GameState) (new_penny : Penny) : Prop :=
  ∀ p ∈ state.pennies, (new_penny.center.1 - p.center.1)^2 + (new_penny.center.2 - p.center.2)^2 > 4

/-- The winning condition for the first player -/
def first_player_wins (board : GameBoard) : Prop :=
  board.m ≥ 2 ∧ board.n ≥ 2

/-- The main theorem -/
theorem first_player_winning_strategy (board : GameBoard) :
  first_player_wins board ↔ ∃ (strategy : GameState → Penny), 
    ∀ (game : GameState), game.board = board → 
      (is_valid_placement game (strategy game) → 
        ∀ (opponent_move : Penny), is_valid_placement (GameState.mk board (strategy game :: game.pennies)) opponent_move → 
          ∃ (next_move : Penny), is_valid_placement (GameState.mk board (opponent_move :: strategy game :: game.pennies)) next_move) :=
sorry

end NUMINAMATH_CALUDE_first_player_winning_strategy_l103_10392


namespace NUMINAMATH_CALUDE_walnut_trees_in_park_l103_10308

theorem walnut_trees_in_park (initial_trees new_trees : ℕ) : 
  initial_trees = 4 → new_trees = 6 → initial_trees + new_trees = 10 := by
  sorry

end NUMINAMATH_CALUDE_walnut_trees_in_park_l103_10308


namespace NUMINAMATH_CALUDE_tripled_division_l103_10344

theorem tripled_division (a b q r : ℤ) 
  (h1 : a = b * q + r) 
  (h2 : 0 ≤ r ∧ r < b) : 
  ∃ (r' : ℤ), 3 * a = (3 * b) * q + r' ∧ r' = 3 * r := by
sorry

end NUMINAMATH_CALUDE_tripled_division_l103_10344


namespace NUMINAMATH_CALUDE_inequality_implies_equality_l103_10384

theorem inequality_implies_equality (a b : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0)
  (h_ineq : Real.log a + Real.log (b^2) ≥ 2*a + b^2/2 - 2) :
  a - 2*b = 1/2 - 2*Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_implies_equality_l103_10384


namespace NUMINAMATH_CALUDE_jump_rope_challenge_l103_10315

structure Jumper where
  initialRate : ℝ
  breakPatterns : List (ℝ × ℝ)
  speedChanges : List ℝ

def calculateSkips (j : Jumper) (totalTime : ℝ) : ℝ :=
  sorry

theorem jump_rope_challenge :
  let leah : Jumper := {
    initialRate := 5,
    breakPatterns := [(120, 20), (120, 25), (120, 30)],
    speedChanges := [0.5, 0.5, 0.5]
  }
  let matt : Jumper := {
    initialRate := 3,
    breakPatterns := [(180, 15), (180, 15)],
    speedChanges := [-0.25, -0.25]
  }
  let linda : Jumper := {
    initialRate := 4,
    breakPatterns := [(240, 10), (240, 15)],
    speedChanges := [-0.1, 0.2]
  }
  let totalTime : ℝ := 600
  (calculateSkips leah totalTime = 3540) ∧
  (calculateSkips matt totalTime = 1635) ∧
  (calculateSkips linda totalTime = 2412) ∧
  (calculateSkips leah totalTime + calculateSkips matt totalTime + calculateSkips linda totalTime = 7587) :=
by
  sorry

end NUMINAMATH_CALUDE_jump_rope_challenge_l103_10315


namespace NUMINAMATH_CALUDE_negative_two_hash_negative_seven_l103_10338

/-- The # operation for rational numbers -/
def hash (a b : ℚ) : ℚ := a * b + 1

/-- Theorem stating that (-2) # (-7) = 15 -/
theorem negative_two_hash_negative_seven :
  hash (-2) (-7) = 15 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_hash_negative_seven_l103_10338


namespace NUMINAMATH_CALUDE_unique_albums_count_l103_10300

/-- Represents a music collection -/
structure MusicCollection where
  total : ℕ
  shared : ℕ
  unique : ℕ

/-- Theorem about the number of unique albums in two collections -/
theorem unique_albums_count
  (andrew : MusicCollection)
  (john : MusicCollection)
  (h1 : andrew.total = 23)
  (h2 : andrew.shared = 11)
  (h3 : john.shared = 11)
  (h4 : john.unique = 8)
  : andrew.unique + john.unique = 20 := by
  sorry

end NUMINAMATH_CALUDE_unique_albums_count_l103_10300


namespace NUMINAMATH_CALUDE_cube_root_function_l103_10304

theorem cube_root_function (k : ℝ) :
  (∃ y : ℝ, y = k * (64 : ℝ)^(1/3) ∧ y = 8) →
  k * (27 : ℝ)^(1/3) = 6 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_function_l103_10304


namespace NUMINAMATH_CALUDE_integral_even_function_l103_10333

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- State the theorem
theorem integral_even_function 
  (f : ℝ → ℝ) 
  (h1 : EvenFunction f) 
  (h2 : ∫ x in (0:ℝ)..6, f x = 8) : 
  ∫ x in (-6:ℝ)..6, f x = 16 := by
  sorry

end NUMINAMATH_CALUDE_integral_even_function_l103_10333


namespace NUMINAMATH_CALUDE_odometer_sum_squares_l103_10316

/-- Represents the odometer reading as a three-digit number -/
structure OdometerReading where
  hundreds : Nat
  tens : Nat
  ones : Nat
  valid : hundreds ≥ 1 ∧ hundreds + tens + ones ≤ 9

/-- Represents a car trip -/
structure CarTrip where
  hours : Nat
  speed : Nat
  initial : OdometerReading
  final : OdometerReading
  valid : speed = 65 ∧
          final.hundreds = initial.ones ∧
          final.tens = initial.tens ∧
          final.ones = initial.hundreds

theorem odometer_sum_squares (trip : CarTrip) :
  trip.initial.hundreds ^ 2 + trip.initial.tens ^ 2 + trip.initial.ones ^ 2 = 41 :=
sorry

end NUMINAMATH_CALUDE_odometer_sum_squares_l103_10316


namespace NUMINAMATH_CALUDE_sequence_properties_l103_10387

/-- Given a sequence {a_n} with partial sum S_n satisfying 3a_n - 2S_n = 2 for all n,
    prove the general term formula and a property of partial sums. -/
theorem sequence_properties (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h : ∀ n, 3 * a n - 2 * S n = 2) : 
    (∀ n, a n = 2 * 3^(n-1)) ∧ 
    (∀ n, S (n+1)^2 - S n * S (n+2) = 4 * 3^n) := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l103_10387


namespace NUMINAMATH_CALUDE_system_solution_l103_10345

theorem system_solution (x y : ℝ) (eq1 : 2 * x - y = -1) (eq2 : x + 4 * y = 22) : x + y = 7 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l103_10345


namespace NUMINAMATH_CALUDE_calculator_cost_ratio_l103_10305

theorem calculator_cost_ratio :
  ∀ (basic scientific graphing : ℝ),
  basic = 8 →
  graphing = 3 * scientific →
  100 - (basic + scientific + graphing) = 28 →
  scientific / basic = 2 := by
sorry

end NUMINAMATH_CALUDE_calculator_cost_ratio_l103_10305


namespace NUMINAMATH_CALUDE_books_per_shelf_l103_10374

theorem books_per_shelf (total_books : ℕ) (books_taken : ℕ) (shelves : ℕ) : 
  total_books = 14 → books_taken = 2 → shelves = 4 → 
  (total_books - books_taken) / shelves = 3 := by
  sorry

end NUMINAMATH_CALUDE_books_per_shelf_l103_10374


namespace NUMINAMATH_CALUDE_sum_to_all_ones_implies_digit_five_or_greater_l103_10329

/-- A function that checks if a natural number has no zero digits -/
def hasNoZeroDigits (n : ℕ) : Prop := sorry

/-- A function that generates all digit permutations of a natural number -/
def digitPermutations (n : ℕ) : Finset ℕ := sorry

/-- A function that checks if a natural number consists only of digit 1 -/
def isAllOnes (n : ℕ) : Prop := sorry

/-- A function that checks if a natural number has at least one digit 5 or greater -/
def hasDigitFiveOrGreater (n : ℕ) : Prop := sorry

/-- Theorem stating that if a number without zero digits and three of its permutations sum to all ones, it must have a digit 5 or greater -/
theorem sum_to_all_ones_implies_digit_five_or_greater (n : ℕ) :
  hasNoZeroDigits n →
  ∃ (p q r : ℕ), p ∈ digitPermutations n ∧ q ∈ digitPermutations n ∧ r ∈ digitPermutations n ∧
  isAllOnes (n + p + q + r) →
  hasDigitFiveOrGreater n :=
sorry

end NUMINAMATH_CALUDE_sum_to_all_ones_implies_digit_five_or_greater_l103_10329


namespace NUMINAMATH_CALUDE_dog_catches_rabbit_l103_10302

/-- Represents the chase scenario between a dog and a rabbit -/
structure ChaseScenario where
  rabbit_head_start : ℕ
  rabbit_distance_ratio : ℕ
  dog_distance_ratio : ℕ
  rabbit_time_ratio : ℕ
  dog_time_ratio : ℕ

/-- Calculates the minimum number of steps the dog must run to catch the rabbit -/
def min_steps_to_catch (scenario : ChaseScenario) : ℕ :=
  sorry

/-- Theorem stating that given the specific chase scenario, the dog needs 240 steps to catch the rabbit -/
theorem dog_catches_rabbit :
  let scenario : ChaseScenario := {
    rabbit_head_start := 100,
    rabbit_distance_ratio := 8,
    dog_distance_ratio := 3,
    rabbit_time_ratio := 9,
    dog_time_ratio := 4
  }
  min_steps_to_catch scenario = 240 := by
  sorry

end NUMINAMATH_CALUDE_dog_catches_rabbit_l103_10302


namespace NUMINAMATH_CALUDE_new_student_weight_l103_10368

theorem new_student_weight
  (n : ℕ)
  (initial_avg : ℝ)
  (new_avg : ℝ)
  (h1 : n = 29)
  (h2 : initial_avg = 28)
  (h3 : new_avg = 27.3) :
  (n + 1) * new_avg - n * initial_avg = 7 :=
by sorry

end NUMINAMATH_CALUDE_new_student_weight_l103_10368


namespace NUMINAMATH_CALUDE_mars_bars_count_l103_10366

theorem mars_bars_count (total : ℕ) (snickers : ℕ) (butterfingers : ℕ) 
  (h1 : total = 12)
  (h2 : snickers = 3)
  (h3 : butterfingers = 7) :
  total - snickers - butterfingers = 2 := by
  sorry

end NUMINAMATH_CALUDE_mars_bars_count_l103_10366


namespace NUMINAMATH_CALUDE_three_letter_initials_count_l103_10371

theorem three_letter_initials_count (n : ℕ) (h : n = 10) : n ^ 3 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_three_letter_initials_count_l103_10371


namespace NUMINAMATH_CALUDE_equation_real_solution_l103_10337

theorem equation_real_solution (x : ℝ) :
  (∀ y : ℝ, ∃ z : ℝ, x^2 + y^2 + z^2 + 2*x*y*z = 1) ↔ (x = 1 ∨ x = -1) := by
  sorry

end NUMINAMATH_CALUDE_equation_real_solution_l103_10337


namespace NUMINAMATH_CALUDE_square_perimeter_ratio_l103_10340

theorem square_perimeter_ratio (s S : ℝ) (hs : s > 0) (hS : S > 0) : 
  S * Real.sqrt 2 = 3 * (s * Real.sqrt 2) → 4 * S / (4 * s) = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_ratio_l103_10340


namespace NUMINAMATH_CALUDE_intersection_angle_relation_l103_10390

-- Define the circles and points
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def Circle.intersect (c1 c2 : Circle) : Prop :=
  ∃ p : ℝ × ℝ, (p.1 - c1.center.1)^2 + (p.2 - c1.center.2)^2 = c1.radius^2 ∧
                (p.1 - c2.center.1)^2 + (p.2 - c2.center.2)^2 = c2.radius^2

-- Define the theorem
theorem intersection_angle_relation (c1 c2 : Circle) (α β : ℝ) :
  c1.radius = c2.radius →
  c1.radius > 0 →
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 > c1.radius^2 →
  Circle.intersect c1 c2 →
  -- Assume α and β are the angles formed at the intersection points
  -- (We don't formally define these angles as it would require more complex geometry)
  β = 3 * α :=
sorry

end NUMINAMATH_CALUDE_intersection_angle_relation_l103_10390


namespace NUMINAMATH_CALUDE_max_pages_proof_l103_10361

/-- The cost in cents to copy 5 pages -/
def cost_per_5_pages : ℚ := 8

/-- The discount rate as a decimal -/
def discount_rate : ℚ := 1 / 10

/-- The available money in dollars -/
def available_money : ℚ := 30

/-- The maximum number of pages that can be copied -/
def max_pages : ℕ := 1687

theorem max_pages_proof :
  let discounted_money : ℚ := available_money * 100 * (1 - discount_rate)
  let pages_per_cent : ℚ := 5 / cost_per_5_pages
  ⌊discounted_money * pages_per_cent⌋ = max_pages := by
  sorry

end NUMINAMATH_CALUDE_max_pages_proof_l103_10361


namespace NUMINAMATH_CALUDE_math_class_students_count_l103_10347

theorem math_class_students_count :
  ∃! n : ℕ, 0 < n ∧ n < 50 ∧ n % 8 = 6 ∧ n % 5 = 1 ∧ n = 46 := by
  sorry

end NUMINAMATH_CALUDE_math_class_students_count_l103_10347


namespace NUMINAMATH_CALUDE_concentric_circles_k_value_l103_10326

/-- Two concentric circles with center at the origin --/
structure ConcentricCircles where
  largeRadius : ℝ
  smallRadius : ℝ

/-- The point P on the larger circle --/
def P : ℝ × ℝ := (10, 6)

/-- The point S on the smaller circle --/
def S (k : ℝ) : ℝ × ℝ := (0, k)

/-- The distance QR --/
def QR : ℝ := 4

theorem concentric_circles_k_value (circles : ConcentricCircles) 
  (h1 : circles.largeRadius ^ 2 = P.1 ^ 2 + P.2 ^ 2)
  (h2 : circles.smallRadius = circles.largeRadius - QR)
  (h3 : (S k).2 = circles.smallRadius) :
  k = 2 * Real.sqrt 34 - 4 := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_k_value_l103_10326


namespace NUMINAMATH_CALUDE_four_square_figure_perimeter_l103_10370

/-- A figure consisting of four identical squares -/
structure FourSquareFigure where
  /-- The side length of each square in the figure -/
  side_length : ℝ
  /-- The total area of the figure is 144 cm² -/
  area_eq : 4 * side_length ^ 2 = 144

/-- The perimeter of a four-square figure is 60 cm -/
theorem four_square_figure_perimeter (fig : FourSquareFigure) : 
  10 * fig.side_length = 60 := by
  sorry

#check four_square_figure_perimeter

end NUMINAMATH_CALUDE_four_square_figure_perimeter_l103_10370


namespace NUMINAMATH_CALUDE_total_minutes_played_l103_10356

/-- The number of days in 2 weeks -/
def days_in_two_weeks : ℕ := 14

/-- The number of gigs Mark does in 2 weeks -/
def gigs_in_two_weeks : ℕ := days_in_two_weeks / 2

/-- The number of songs Mark plays in each gig -/
def songs_per_gig : ℕ := 3

/-- The duration of the first two songs in minutes -/
def short_song_duration : ℕ := 5

/-- The duration of the last song in minutes -/
def long_song_duration : ℕ := 2 * short_song_duration

/-- The total duration of all songs in one gig in minutes -/
def duration_per_gig : ℕ := 2 * short_song_duration + long_song_duration

/-- The theorem stating the total number of minutes Mark played in 2 weeks -/
theorem total_minutes_played : gigs_in_two_weeks * duration_per_gig = 140 := by
  sorry

end NUMINAMATH_CALUDE_total_minutes_played_l103_10356


namespace NUMINAMATH_CALUDE_trigonometric_identities_l103_10318

open Real

theorem trigonometric_identities (α : ℝ) (h : tan α = 3) :
  (sin α + 3 * cos α) / (2 * sin α + 5 * cos α) = 6 / 11 ∧
  sin α ^ 2 + sin α * cos α + 3 * cos α ^ 2 = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l103_10318


namespace NUMINAMATH_CALUDE_geometric_series_sum_specific_l103_10380

def geometric_series_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_series_sum_specific : 
  let a : ℚ := 1/4
  let r : ℚ := 1/4
  let n : ℕ := 8
  geometric_series_sum a r n = 65535/196608 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_specific_l103_10380


namespace NUMINAMATH_CALUDE_max_sum_xy_l103_10385

def associated_numbers (m : ℕ) : List ℕ :=
  sorry

def P (m : ℕ) : ℚ :=
  (associated_numbers m).sum / 22

def x (a b : ℕ) : ℕ := 100 * a + 10 * b + 3

def y (b : ℕ) : ℕ := 400 + 10 * b + 5

theorem max_sum_xy :
  ∀ a b : ℕ,
    1 ≤ a ∧ a ≤ 9 →
    1 ≤ b ∧ b ≤ 9 →
    (∀ d : ℕ, d ∈ associated_numbers (x a b) → d ≠ 0) →
    (∀ d : ℕ, d ∈ associated_numbers (y b) → d ≠ 0) →
    P (x a b) + P (y b) = 20 →
    x a b + y b ≤ 1028 :=
  sorry

end NUMINAMATH_CALUDE_max_sum_xy_l103_10385


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l103_10319

theorem pure_imaginary_condition (i a : ℂ) : 
  i^2 = -1 →
  (((i^2 + a*i) / (1 + i)).re = 0) →
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l103_10319
