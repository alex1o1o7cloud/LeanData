import Mathlib

namespace NUMINAMATH_CALUDE_division_problem_l1788_178833

theorem division_problem (a b c : ℚ) 
  (h1 : a / b = 3) 
  (h2 : b / c = 2 / 5) : 
  c / a = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1788_178833


namespace NUMINAMATH_CALUDE_max_sum_of_squares_of_roots_l1788_178857

theorem max_sum_of_squares_of_roots (k : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x : ℝ, x^2 - (k-2)*x + (k^2+3*k+5) = 0 ↔ x = x₁ ∨ x = x₂) →
  (∃ k : ℝ, x₁^2 + x₂^2 = 18) ∧
  (∀ k : ℝ, x₁^2 + x₂^2 ≤ 18) :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_of_roots_l1788_178857


namespace NUMINAMATH_CALUDE_champagne_bottle_volume_l1788_178856

theorem champagne_bottle_volume
  (hot_tub_volume : ℚ)
  (quarts_per_gallon : ℚ)
  (bottle_cost : ℚ)
  (discount_rate : ℚ)
  (total_spent : ℚ)
  (h1 : hot_tub_volume = 40)
  (h2 : quarts_per_gallon = 4)
  (h3 : bottle_cost = 50)
  (h4 : discount_rate = 0.2)
  (h5 : total_spent = 6400) :
  (hot_tub_volume * quarts_per_gallon) / ((total_spent / (1 - discount_rate)) / bottle_cost) = 1 :=
by sorry

end NUMINAMATH_CALUDE_champagne_bottle_volume_l1788_178856


namespace NUMINAMATH_CALUDE_rectangle_area_l1788_178867

theorem rectangle_area (L B : ℝ) (h1 : L - B = 23) (h2 : 2 * (L + B) = 206) : L * B = 2520 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1788_178867


namespace NUMINAMATH_CALUDE_product_of_fractions_l1788_178861

theorem product_of_fractions : 
  (8 / 4) * (14 / 7) * (20 / 10) * (25 / 50) * (9 / 18) * (12 / 6) * (21 / 42) * (16 / 8) = 8 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l1788_178861


namespace NUMINAMATH_CALUDE_product_of_slopes_l1788_178849

theorem product_of_slopes (m n : ℝ) : 
  (∃ θ₁ θ₂ : ℝ, θ₁ = 3 * θ₂ ∧ m = Real.tan θ₁ ∧ n = Real.tan θ₂) →  -- L1 makes three times the angle with horizontal as L2
  m = 3 * n →                                                      -- L1 has 3 times the slope of L2
  m ≠ 0 →                                                          -- L1 is not vertical
  m * n = 0 :=                                                     -- Conclusion: mn = 0
by sorry

end NUMINAMATH_CALUDE_product_of_slopes_l1788_178849


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l1788_178822

theorem absolute_value_equation_solution :
  ∃! x : ℚ, |x + 5| = 3 * x - 2 :=
by
  use 7/2
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l1788_178822


namespace NUMINAMATH_CALUDE_gus_ate_fourteen_eggs_l1788_178878

/-- Represents the number of eggs in each dish Gus ate throughout the day -/
def eggs_per_dish : List Nat := [2, 1, 3, 2, 1, 2, 3]

/-- The total number of eggs Gus ate -/
def total_eggs : Nat := eggs_per_dish.sum

/-- Theorem stating that the total number of eggs Gus ate is 14 -/
theorem gus_ate_fourteen_eggs : total_eggs = 14 := by sorry

end NUMINAMATH_CALUDE_gus_ate_fourteen_eggs_l1788_178878


namespace NUMINAMATH_CALUDE_ellipse_m_range_l1788_178855

/-- Represents an ellipse with the given equation and foci on the y-axis -/
structure Ellipse where
  m : ℝ
  eq : ∀ (x y : ℝ), x^2 / (25 - m) + y^2 / (m + 9) = 1
  foci_on_y_axis : True  -- This is a placeholder for the foci condition

/-- The range of valid m values for the given ellipse -/
theorem ellipse_m_range (e : Ellipse) : 8 < e.m ∧ e.m < 25 := by
  sorry

#check ellipse_m_range

end NUMINAMATH_CALUDE_ellipse_m_range_l1788_178855


namespace NUMINAMATH_CALUDE_line_equation_through_ellipse_midpoint_l1788_178865

/-- Given an ellipse and a line passing through a point P, prove the equation of the line --/
theorem line_equation_through_ellipse_midpoint (x y : ℝ) :
  let ellipse := (y^2 / 9 + x^2 = 1)
  let P := (1/2, 1/2)
  let line_passes_through_P := (∃ (t : ℝ), (x, y) = P + t • (1, -9))
  let A := (x₁, y₁)
  let B := (x₂, y₂)
  let line_intersects_ellipse := (y₁^2 / 9 + x₁^2 = 1) ∧ (y₂^2 / 9 + x₂^2 = 1)
  let P_bisects_AB := (x₁ + x₂ = 1) ∧ (y₁ + y₂ = 1)
  ellipse →
  line_passes_through_P →
  line_intersects_ellipse →
  P_bisects_AB →
  (9*x + y - 5 = 0) := by
sorry

end NUMINAMATH_CALUDE_line_equation_through_ellipse_midpoint_l1788_178865


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l1788_178837

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 9/16) 
  (h2 : x - y = 5/16) : 
  x^2 - y^2 = 45/256 := by
sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l1788_178837


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1788_178817

theorem complex_equation_solution (z : ℂ) :
  (z - 2) * (1 + Complex.I) = 1 - Complex.I → z = 2 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1788_178817


namespace NUMINAMATH_CALUDE_min_profit_is_266_l1788_178811

/-- Represents the production plan for the clothing factory -/
structure ProductionPlan where
  typeA : ℕ
  typeB : ℕ

/-- Calculates the total cost for a given production plan -/
def totalCost (plan : ProductionPlan) : ℕ :=
  34 * plan.typeA + 42 * plan.typeB

/-- Calculates the total revenue for a given production plan -/
def totalRevenue (plan : ProductionPlan) : ℕ :=
  39 * plan.typeA + 50 * plan.typeB

/-- Calculates the profit for a given production plan -/
def profit (plan : ProductionPlan) : ℤ :=
  totalRevenue plan - totalCost plan

/-- Theorem: The minimum profit is 266 yuan -/
theorem min_profit_is_266 :
  ∃ (minProfit : ℕ), minProfit = 266 ∧
  ∀ (plan : ProductionPlan),
    plan.typeA + plan.typeB = 40 →
    1536 ≤ totalCost plan →
    totalCost plan ≤ 1552 →
    minProfit ≤ profit plan := by
  sorry

#check min_profit_is_266

end NUMINAMATH_CALUDE_min_profit_is_266_l1788_178811


namespace NUMINAMATH_CALUDE_derivative_of_y_l1788_178832

noncomputable def y (x : ℝ) : ℝ := (1 + Real.cos (2 * x))^3

theorem derivative_of_y (x : ℝ) :
  deriv y x = -48 * (Real.cos x)^5 * Real.sin x := by sorry

end NUMINAMATH_CALUDE_derivative_of_y_l1788_178832


namespace NUMINAMATH_CALUDE_smallest_n0_for_inequality_l1788_178896

theorem smallest_n0_for_inequality : ∃ (n0 : ℕ), n0 = 5 ∧ 
  (∀ n : ℕ, n ≥ n0 → 2^n > n^2 + 1) ∧ 
  (∀ m : ℕ, m < n0 → ¬(2^m > m^2 + 1)) :=
sorry

end NUMINAMATH_CALUDE_smallest_n0_for_inequality_l1788_178896


namespace NUMINAMATH_CALUDE_square_minus_product_equals_one_l1788_178881

theorem square_minus_product_equals_one : 2014^2 - 2013 * 2015 = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_product_equals_one_l1788_178881


namespace NUMINAMATH_CALUDE_cookie_radius_cookie_is_circle_l1788_178851

theorem cookie_radius (x y : ℝ) :
  (x^2 + y^2 + 17 = 6*x + 10*y) ↔ ((x - 3)^2 + (y - 5)^2 = 17) :=
by sorry

theorem cookie_is_circle (x y : ℝ) :
  (x^2 + y^2 + 17 = 6*x + 10*y) → ∃ (center_x center_y radius : ℝ),
    ((x - center_x)^2 + (y - center_y)^2 = radius^2) ∧ (radius = Real.sqrt 17) :=
by sorry

end NUMINAMATH_CALUDE_cookie_radius_cookie_is_circle_l1788_178851


namespace NUMINAMATH_CALUDE_system_solution_l1788_178850

theorem system_solution (x y z : ℝ) : 
  (x * y = 1 ∧ y * z = 2 ∧ z * x = 8) ↔ 
  ((x = 2 ∧ y = (1/2) ∧ z = 4) ∨ (x = -2 ∧ y = -(1/2) ∧ z = -4)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1788_178850


namespace NUMINAMATH_CALUDE_triangle_properties_l1788_178806

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  b = 2 * Real.sqrt 3 →
  c = 2 →
  b * Real.sin C - 2 * c * Real.sin B * Real.cos A = 0 →
  let S := (1 / 2) * b * c * Real.sin A
  let f := fun x => 4 * Real.cos x * (Real.sin x * Real.cos A + Real.cos x * Real.sin A)
  (S = 3) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 3), f x ≤ 2 + Real.sqrt 3) ∧
  (f (Real.pi / 12) = 2 + Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1788_178806


namespace NUMINAMATH_CALUDE_three_pumps_fill_time_l1788_178853

-- Define the pumps and tank
variable (T : ℝ) -- Volume of the tank
variable (X Y Z : ℝ) -- Rates at which pumps X, Y, and Z fill the tank

-- Define the conditions
axiom cond1 : T = 3 * (X + Y)
axiom cond2 : T = 6 * (X + Z)
axiom cond3 : T = 4.5 * (Y + Z)

-- Define the theorem
theorem three_pumps_fill_time : 
  T / (X + Y + Z) = 36 / 13 := by sorry

end NUMINAMATH_CALUDE_three_pumps_fill_time_l1788_178853


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l1788_178869

theorem quadratic_roots_sum_of_squares (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, 
    (2 * x₁^2 + k * x₁ - 2 * k + 1 = 0) ∧ 
    (2 * x₂^2 + k * x₂ - 2 * k + 1 = 0) ∧ 
    (x₁ ≠ x₂) ∧
    (x₁^2 + x₂^2 = 29/4)) → 
  (k = 3 ∨ k = -11) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l1788_178869


namespace NUMINAMATH_CALUDE_sum_of_roots_product_polynomials_l1788_178882

theorem sum_of_roots_product_polynomials :
  let p₁ : Polynomial ℝ := 3 * X^3 - 2 * X^2 + 9 * X - 15
  let p₂ : Polynomial ℝ := 4 * X^3 + 8 * X^2 - 4 * X + 24
  let roots := (p₁.roots.toFinset ∪ p₂.roots.toFinset).toList
  List.sum roots = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_product_polynomials_l1788_178882


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1788_178808

theorem quadratic_inequality_solution (x : ℝ) : x^2 + 7*x < 12 ↔ -4 < x ∧ x < -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1788_178808


namespace NUMINAMATH_CALUDE_line_plane_perpendicularity_l1788_178859

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (contained_in : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (c : Line) (α β : Plane) :
  contained_in c α → perpendicular c β → plane_perpendicular α β :=
by sorry

end NUMINAMATH_CALUDE_line_plane_perpendicularity_l1788_178859


namespace NUMINAMATH_CALUDE_constant_value_l1788_178870

-- Define the function f
def f (x : ℝ) : ℝ := x + 4

-- Define the equation
def equation (c : ℝ) (x : ℝ) : Prop :=
  (3 * f (x - 2)) / f 0 + 4 = f (c * x + 1)

-- Theorem statement
theorem constant_value :
  ∀ c : ℝ, equation c 0.4 → c = 2 := by
  sorry

end NUMINAMATH_CALUDE_constant_value_l1788_178870


namespace NUMINAMATH_CALUDE_range_of_m_l1788_178898

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, x^2 + 2*x + m > 0) ↔ m > 1 := by sorry

end NUMINAMATH_CALUDE_range_of_m_l1788_178898


namespace NUMINAMATH_CALUDE_gcf_of_72_and_90_l1788_178835

theorem gcf_of_72_and_90 : Nat.gcd 72 90 = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_72_and_90_l1788_178835


namespace NUMINAMATH_CALUDE_stationery_cost_l1788_178805

theorem stationery_cost (p e : ℕ) : 
  15 * p + 7 * e = 170 →
  p < e →
  2 * p ≠ e →
  e ≠ 2 * p →
  p + e = 16 :=
by sorry

end NUMINAMATH_CALUDE_stationery_cost_l1788_178805


namespace NUMINAMATH_CALUDE_math_test_problem_count_l1788_178884

theorem math_test_problem_count :
  ∀ (total_points three_point_count four_point_count : ℕ),
    total_points = 100 →
    four_point_count = 10 →
    total_points = 3 * three_point_count + 4 * four_point_count →
    three_point_count + four_point_count = 30 :=
by sorry

end NUMINAMATH_CALUDE_math_test_problem_count_l1788_178884


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1788_178836

theorem complex_modulus_problem (z : ℂ) (h : Complex.I * z = 3 - 4 * Complex.I) : 
  Complex.abs z = 5 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1788_178836


namespace NUMINAMATH_CALUDE_sin_cos_value_f_minus_cos_value_l1788_178866

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.tan x) * Real.cos x / (1 + Real.cos (-x))

-- Theorem 1
theorem sin_cos_value (θ : ℝ) (h : f θ * Real.sin (π/6) - Real.cos θ = 0) :
  Real.sin θ * Real.cos θ = 2/5 := by sorry

-- Theorem 2
theorem f_minus_cos_value (θ : ℝ) (h1 : f θ * Real.cos θ = 1/8) (h2 : π/4 < θ ∧ θ < 3*π/4) :
  f (2019*π - θ) - Real.cos (2018*π - θ) = Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_cos_value_f_minus_cos_value_l1788_178866


namespace NUMINAMATH_CALUDE_second_caterer_more_cost_effective_l1788_178825

/-- Represents the cost function for a caterer -/
structure Caterer where
  base_fee : ℕ
  per_person : ℕ

/-- Calculates the total cost for a given number of people -/
def total_cost (c : Caterer) (people : ℕ) : ℕ :=
  c.base_fee + c.per_person * people

/-- The first caterer's pricing structure -/
def caterer1 : Caterer :=
  { base_fee := 120, per_person := 14 }

/-- The second caterer's pricing structure -/
def caterer2 : Caterer :=
  { base_fee := 210, per_person := 11 }

/-- Theorem stating the minimum number of people for the second caterer to be more cost-effective -/
theorem second_caterer_more_cost_effective :
  (∀ n : ℕ, n ≥ 31 → total_cost caterer2 n < total_cost caterer1 n) ∧
  (∀ n : ℕ, n < 31 → total_cost caterer2 n ≥ total_cost caterer1 n) :=
sorry

end NUMINAMATH_CALUDE_second_caterer_more_cost_effective_l1788_178825


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1788_178872

theorem quadratic_inequality_solution_set (m : ℝ) : 
  (∀ x : ℝ, x^2 + m*x + 1 > 0) → -2 < m ∧ m < 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1788_178872


namespace NUMINAMATH_CALUDE_cupcakes_eaten_correct_l1788_178824

/-- Calculates the number of cupcakes Todd ate given the initial number of cupcakes,
    the number of packages, and the number of cupcakes per package. -/
def cupcakes_eaten (initial : ℕ) (packages : ℕ) (per_package : ℕ) : ℕ :=
  initial - (packages * per_package)

/-- Proves that the number of cupcakes Todd ate is correct -/
theorem cupcakes_eaten_correct (initial : ℕ) (packages : ℕ) (per_package : ℕ) :
  cupcakes_eaten initial packages per_package = initial - (packages * per_package) :=
by
  sorry

#eval cupcakes_eaten 39 6 3  -- Should evaluate to 21

end NUMINAMATH_CALUDE_cupcakes_eaten_correct_l1788_178824


namespace NUMINAMATH_CALUDE_sum_inequality_l1788_178813

theorem sum_inequality (a b c d : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0)
  (h5 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h6 : a ≥ b ∧ a ≥ c ∧ a ≥ d)
  (h7 : d ≤ b ∧ d ≤ c)
  (h8 : a * d = b * c) :
  a + d > b + c := by
sorry

end NUMINAMATH_CALUDE_sum_inequality_l1788_178813


namespace NUMINAMATH_CALUDE_prob_three_odds_eq_4_35_l1788_178854

/-- The set of numbers from which we select -/
def S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

/-- The set of odd numbers in S -/
def odds : Finset ℕ := S.filter (fun n => n % 2 = 1)

/-- The number of elements to select -/
def k : ℕ := 3

/-- The probability of selecting three distinct odd numbers from S -/
theorem prob_three_odds_eq_4_35 : 
  (Finset.card (odds.powersetCard k)) / (Finset.card (S.powersetCard k)) = 4 / 35 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_odds_eq_4_35_l1788_178854


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1788_178846

theorem inequality_equivalence (x y : ℝ) (h : x > 0) :
  (Real.sqrt (y - x) / x ≤ 1) ↔ (x ≤ y ∧ y ≤ x^2 + x) :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1788_178846


namespace NUMINAMATH_CALUDE_original_deck_size_l1788_178829

/-- Represents a deck of playing cards -/
structure Deck where
  total_cards : ℕ

/-- Represents the game setup -/
structure GameSetup where
  original_deck : Deck
  cards_kept_away : ℕ
  cards_in_play : ℕ

/-- The number of cards in a standard deck -/
def standard_deck_size : ℕ := 52

/-- Theorem: The original deck had 52 cards -/
theorem original_deck_size (setup : GameSetup) 
  (h1 : setup.cards_kept_away = 2) 
  (h2 : setup.cards_in_play + setup.cards_kept_away = setup.original_deck.total_cards) : 
  setup.original_deck.total_cards = standard_deck_size := by
  sorry

end NUMINAMATH_CALUDE_original_deck_size_l1788_178829


namespace NUMINAMATH_CALUDE_max_a4b4_l1788_178895

/-- Given an arithmetic sequence a and a geometric sequence b satisfying
    certain conditions, the maximum value of a₄b₄ is 37/4 -/
theorem max_a4b4 (a b : ℕ → ℝ) 
  (h_arith : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
  (h_geom : ∀ n, b (n + 1) / b n = b (n + 2) / b (n + 1))
  (h1 : a 1 * b 1 = 20)
  (h2 : a 2 * b 2 = 19)
  (h3 : a 3 * b 3 = 14) :
  (∀ x, a 4 * b 4 ≤ x) → x = 37/4 :=
sorry

end NUMINAMATH_CALUDE_max_a4b4_l1788_178895


namespace NUMINAMATH_CALUDE_sally_peaches_count_l1788_178819

def initial_peaches : ℕ := 13
def first_orchard_peaches : ℕ := 55

def peaches_after_giving : ℕ := initial_peaches - (initial_peaches / 2)
def peaches_after_first_orchard : ℕ := peaches_after_giving + first_orchard_peaches
def second_orchard_peaches : ℕ := 2 * first_orchard_peaches
def total_peaches : ℕ := peaches_after_first_orchard + second_orchard_peaches

theorem sally_peaches_count : total_peaches = 172 := by
  sorry

end NUMINAMATH_CALUDE_sally_peaches_count_l1788_178819


namespace NUMINAMATH_CALUDE_min_value_expression_l1788_178874

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 1) :
  ∃ m : ℝ, m = 3 ∧ ∀ x y : ℝ, x > 0 → y > 0 → x * y = 1 →
    x^2 + y^2 + 4 / (x + y)^2 ≥ m :=
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1788_178874


namespace NUMINAMATH_CALUDE_discriminant_irrational_l1788_178858

/-- A quadratic polynomial without roots -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℚ
  c : ℝ
  no_roots : ∀ x : ℝ, a * x^2 + b * x + c ≠ 0

/-- The function f(x) for a QuadraticPolynomial -/
def f (p : QuadraticPolynomial) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- The discriminant of a QuadraticPolynomial -/
def discriminant (p : QuadraticPolynomial) : ℝ :=
  p.b^2 - 4 * p.a * p.c

/-- Exactly one of c or f(c) is irrational -/
axiom one_irrational (p : QuadraticPolynomial) :
  (¬ Irrational p.c ∧ Irrational (f p p.c)) ∨
  (Irrational p.c ∧ ¬ Irrational (f p p.c))

theorem discriminant_irrational (p : QuadraticPolynomial) :
  Irrational (discriminant p) :=
sorry

end NUMINAMATH_CALUDE_discriminant_irrational_l1788_178858


namespace NUMINAMATH_CALUDE_martha_cards_l1788_178883

/-- The number of cards Martha ends up with after receiving more cards -/
def total_cards (initial : ℕ) (received : ℕ) : ℕ :=
  initial + received

/-- Theorem stating that Martha ends up with 79 cards -/
theorem martha_cards : total_cards 3 76 = 79 := by
  sorry

end NUMINAMATH_CALUDE_martha_cards_l1788_178883


namespace NUMINAMATH_CALUDE_arithmetic_sequence_geometric_mean_l1788_178823

/-- Given an arithmetic sequence {a_n} with first term a_1 = 1 and common difference d ≠ 0,
    if a_2 is the geometric mean of a_1 and a_4, then d = 1. -/
theorem arithmetic_sequence_geometric_mean (d : ℝ) (hd : d ≠ 0) : 
  let a : ℕ → ℝ := λ n => 1 + (n - 1) * d
  (a 2)^2 = a 1 * a 4 → d = 1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_geometric_mean_l1788_178823


namespace NUMINAMATH_CALUDE_total_points_is_94_bonus_points_is_7_l1788_178885

/-- Represents the points system and creature counts in the video game --/
structure GameState where
  goblin_points : ℕ := 3
  troll_points : ℕ := 5
  dragon_points : ℕ := 10
  combo_bonus : ℕ := 7
  total_goblins : ℕ := 14
  total_trolls : ℕ := 15
  total_dragons : ℕ := 4
  defeated_goblins : ℕ := 9  -- 70% of 14 rounded down
  defeated_trolls : ℕ := 10  -- 2/3 of 15
  defeated_dragons : ℕ := 1

/-- Calculates the total points earned in the game --/
def calculate_points (state : GameState) : ℕ :=
  state.goblin_points * state.defeated_goblins +
  state.troll_points * state.defeated_trolls +
  state.dragon_points * state.defeated_dragons +
  state.combo_bonus * (min state.defeated_goblins (min state.defeated_trolls state.defeated_dragons))

/-- Theorem stating that the total points earned is 94 --/
theorem total_points_is_94 (state : GameState) : calculate_points state = 94 := by
  sorry

/-- Theorem stating that the bonus points earned is 7 --/
theorem bonus_points_is_7 (state : GameState) : 
  state.combo_bonus * (min state.defeated_goblins (min state.defeated_trolls state.defeated_dragons)) = 7 := by
  sorry

end NUMINAMATH_CALUDE_total_points_is_94_bonus_points_is_7_l1788_178885


namespace NUMINAMATH_CALUDE_perpendicular_to_parallel_plane_l1788_178890

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relationships between planes and lines
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_to_parallel_plane
  (α β : Plane) (l : Line)
  (h1 : perpendicular l α)
  (h2 : parallel α β) :
  perpendicular l β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_to_parallel_plane_l1788_178890


namespace NUMINAMATH_CALUDE_locus_is_parabolic_arc_l1788_178839

-- Define the semicircle
structure Semicircle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define tangency between a circle and a semicircle
def is_tangent_to_semicircle (c : Circle) (s : Semicircle) : Prop :=
  ∃ p : ℝ × ℝ, 
    (p.1 - s.center.1)^2 + (p.2 - s.center.2)^2 = s.radius^2 ∧
    (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- Define tangency between a circle and a line (diameter)
def is_tangent_to_diameter (c : Circle) (s : Semicircle) : Prop :=
  ∃ p : ℝ × ℝ,
    p.2 = s.center.2 - s.radius ∧
    (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- Define a parabola
structure Parabola where
  focus : ℝ × ℝ
  directrix : ℝ  -- y-coordinate of the directrix

-- Define a point being on a parabola
def on_parabola (p : ℝ × ℝ) (para : Parabola) : Prop :=
  (p.1 - para.focus.1)^2 + (p.2 - para.focus.2)^2 = (p.2 - para.directrix)^2

-- Main theorem
theorem locus_is_parabolic_arc (s : Semicircle) :
  ∀ c : Circle, 
    is_tangent_to_semicircle c s → 
    is_tangent_to_diameter c s → 
    ∃ para : Parabola, 
      para.focus = s.center ∧ 
      para.directrix = s.center.2 - 2 * s.radius ∧
      on_parabola c.center para ∧
      (c.center.1 - s.center.1)^2 + (c.center.2 - s.center.2)^2 < s.radius^2 :=
sorry

end NUMINAMATH_CALUDE_locus_is_parabolic_arc_l1788_178839


namespace NUMINAMATH_CALUDE_consecutive_integers_product_plus_one_is_square_l1788_178838

theorem consecutive_integers_product_plus_one_is_square (n : ℤ) :
  n * (n + 1) * (n + 2) * (n + 3) + 1 = (n^2 + 3*n + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_plus_one_is_square_l1788_178838


namespace NUMINAMATH_CALUDE_effective_annual_rate_l1788_178875

/-- The effective annual compound interest rate for a 4-year investment -/
theorem effective_annual_rate (initial_investment final_amount : ℝ)
  (rate1 rate2 rate3 rate4 : ℝ) (h_initial : initial_investment = 810)
  (h_final : final_amount = 1550) (h_rate1 : rate1 = 0.05)
  (h_rate2 : rate2 = 0.07) (h_rate3 : rate3 = 0.06) (h_rate4 : rate4 = 0.04) :
  ∃ (r : ℝ), (abs (r - 0.1755) < 0.0001 ∧
  final_amount = initial_investment * ((1 + rate1) * (1 + rate2) * (1 + rate3) * (1 + rate4)) ∧
  final_amount = initial_investment * (1 + r)^4) :=
sorry

end NUMINAMATH_CALUDE_effective_annual_rate_l1788_178875


namespace NUMINAMATH_CALUDE_cos_five_pi_thirds_plus_two_alpha_l1788_178887

theorem cos_five_pi_thirds_plus_two_alpha (α : ℝ) 
  (h : Real.sin (π / 6 - α) = 1 / 3) : 
  Real.cos (5 * π / 3 + 2 * α) = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cos_five_pi_thirds_plus_two_alpha_l1788_178887


namespace NUMINAMATH_CALUDE_system_solution_ratio_l1788_178800

theorem system_solution_ratio (x y z a b : ℝ) 
  (eq1 : 4 * x - 3 * y + z = a)
  (eq2 : 6 * y - 8 * x - 2 * z = b)
  (b_nonzero : b ≠ 0)
  (has_solution : ∃ (x y z : ℝ), 4 * x - 3 * y + z = a ∧ 6 * y - 8 * x - 2 * z = b) :
  a / b = -1 / 2 := by
sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l1788_178800


namespace NUMINAMATH_CALUDE_merchant_markup_percentage_l1788_178877

theorem merchant_markup_percentage (markup_percentage : ℝ) : 
  (∀ cost_price : ℝ, cost_price > 0 →
    let marked_price := cost_price * (1 + markup_percentage / 100)
    let discounted_price := marked_price * 0.9
    let profit_percentage := (discounted_price - cost_price) / cost_price * 100
    profit_percentage = 57.5) →
  markup_percentage = 75 := by
sorry

end NUMINAMATH_CALUDE_merchant_markup_percentage_l1788_178877


namespace NUMINAMATH_CALUDE_angle_ratio_not_implies_right_triangle_l1788_178814

/-- Triangle ABC with angles A, B, and C -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_angles : A + B + C = 180
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

/-- The condition that angles are in the ratio 3:4:5 -/
def angle_ratio (t : Triangle) : Prop :=
  ∃ (x : ℝ), t.A = 3*x ∧ t.B = 4*x ∧ t.C = 5*x

/-- A triangle is right if one of its angles is 90 degrees -/
def is_right_triangle (t : Triangle) : Prop :=
  t.A = 90 ∨ t.B = 90 ∨ t.C = 90

/-- The main theorem: a triangle with angles in ratio 3:4:5 is not necessarily right -/
theorem angle_ratio_not_implies_right_triangle :
  ∃ (t : Triangle), angle_ratio t ∧ ¬is_right_triangle t :=
sorry

end NUMINAMATH_CALUDE_angle_ratio_not_implies_right_triangle_l1788_178814


namespace NUMINAMATH_CALUDE_integer_pair_solution_l1788_178816

theorem integer_pair_solution (m n : ℤ) :
  (m - n)^2 = 4 * m * n / (m + n - 1) →
  ∃ k : ℕ, k ≠ 1 ∧
    ((m = (k^2 + k) / 2 ∧ n = (k^2 - k) / 2) ∨
     (m = (k^2 - k) / 2 ∧ n = (k^2 + k) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_integer_pair_solution_l1788_178816


namespace NUMINAMATH_CALUDE_triangle_transformation_correct_l1788_178827

def initial_triangle : List (ℝ × ℝ) := [(1, -2), (-1, -2), (1, 1)]

def rotate_180 (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

def reflect_x_axis (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def rotate_270_clockwise (p : ℝ × ℝ) : ℝ × ℝ := (p.2, -p.1)

def transform_triangle (triangle : List (ℝ × ℝ)) : List (ℝ × ℝ) :=
  triangle.map (rotate_270_clockwise ∘ reflect_x_axis ∘ rotate_180)

theorem triangle_transformation_correct :
  transform_triangle initial_triangle = [(2, 1), (2, -1), (-1, -1)] := by
  sorry

end NUMINAMATH_CALUDE_triangle_transformation_correct_l1788_178827


namespace NUMINAMATH_CALUDE_phil_initial_money_l1788_178852

/-- The amount of money Phil started with, given his purchases and remaining quarters. -/
theorem phil_initial_money (pizza_cost soda_cost jeans_cost : ℚ)
  (quarters_left : ℕ) (quarter_value : ℚ) :
  pizza_cost = 2.75 →
  soda_cost = 1.50 →
  jeans_cost = 11.50 →
  quarters_left = 97 →
  quarter_value = 0.25 →
  pizza_cost + soda_cost + jeans_cost + (quarters_left : ℚ) * quarter_value = 40 :=
by sorry

end NUMINAMATH_CALUDE_phil_initial_money_l1788_178852


namespace NUMINAMATH_CALUDE_quadratic_equation_factor_l1788_178830

theorem quadratic_equation_factor (a : ℝ) : 
  (∀ x, 2 * x^2 - 8 * x + a = 0 ↔ 2 * (x - 2)^2 = 4) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_factor_l1788_178830


namespace NUMINAMATH_CALUDE_regular_rate_is_three_l1788_178891

/-- Represents a worker's pay structure and hours worked -/
structure PayStructure where
  regularRate : ℝ
  overtimeHours : ℝ
  totalPay : ℝ

/-- Calculates the total pay for a given pay structure -/
def calculateTotalPay (p : PayStructure) : ℝ :=
  40 * p.regularRate + p.overtimeHours * (2 * p.regularRate)

/-- Theorem stating that given the conditions, the regular rate is $3 per hour -/
theorem regular_rate_is_three (p : PayStructure) 
  (h1 : p.overtimeHours = 8)
  (h2 : p.totalPay = 168)
  (h3 : calculateTotalPay p = p.totalPay) : 
  p.regularRate = 3 := by
  sorry


end NUMINAMATH_CALUDE_regular_rate_is_three_l1788_178891


namespace NUMINAMATH_CALUDE_same_num_digits_l1788_178826

/-- The number of digits in the decimal representation of a positive integer -/
def num_digits (n : ℕ) : ℕ := sorry

/-- Theorem: If 10^b < a^b and 2^b < 10^b, then a^b and a^b + 2^b have the same number of digits -/
theorem same_num_digits (a b : ℕ) (h1 : 10^b < a^b) (h2 : 2^b < 10^b) :
  num_digits (a^b) = num_digits (a^b + 2^b) := by sorry

end NUMINAMATH_CALUDE_same_num_digits_l1788_178826


namespace NUMINAMATH_CALUDE_solve_cake_baking_l1788_178821

def cake_baking_problem (jane_rate roy_rate : ℚ) (jane_remaining_time : ℚ) (jane_remaining_work : ℚ) : Prop :=
  let combined_rate := jane_rate + roy_rate
  let total_work := 1
  ∃ t : ℚ, 
    t > 0 ∧
    combined_rate * t + jane_remaining_work = total_work ∧
    jane_rate * jane_remaining_time = jane_remaining_work ∧
    t = 2

theorem solve_cake_baking :
  cake_baking_problem (1/4) (1/5) (2/5) (1/10) :=
sorry

end NUMINAMATH_CALUDE_solve_cake_baking_l1788_178821


namespace NUMINAMATH_CALUDE_quadratic_polynomial_satisfies_conditions_l1788_178809

-- Define the quadratic polynomial
def p (x : ℚ) : ℚ := (13/6) * x^2 - (7/6) * x + 2

-- State the theorem
theorem quadratic_polynomial_satisfies_conditions :
  p 1 = 3 ∧ p 0 = 2 ∧ p 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_satisfies_conditions_l1788_178809


namespace NUMINAMATH_CALUDE_remaining_bag_weight_l1788_178886

def bag_weights : List ℕ := [15, 16, 18, 19, 20, 31]

def is_valid_partition (partition : List ℕ × List ℕ) : Prop :=
  let (group1, group2) := partition
  group1.length + group2.length = 5 ∧
  group1.sum = 2 * group2.sum ∧
  (∀ w ∈ group1, w ∈ bag_weights) ∧
  (∀ w ∈ group2, w ∈ bag_weights) ∧
  (∀ w ∈ group1, w ∉ group2) ∧
  (∀ w ∈ group2, w ∉ group1)

theorem remaining_bag_weight :
  ∃ (partition : List ℕ × List ℕ), is_valid_partition partition →
  bag_weights.sum - (partition.1.sum + partition.2.sum) = 20 :=
sorry

end NUMINAMATH_CALUDE_remaining_bag_weight_l1788_178886


namespace NUMINAMATH_CALUDE_roots_have_unit_modulus_l1788_178844

theorem roots_have_unit_modulus (z : ℂ) : 
  11 * z^10 + 10 * Complex.I * z^9 + 10 * Complex.I * z - 11 = 0 → Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_roots_have_unit_modulus_l1788_178844


namespace NUMINAMATH_CALUDE_kim_payment_share_l1788_178899

/-- Represents the time (in days) it takes a person to complete the work alone -/
structure WorkTime where
  days : ℚ
  days_positive : days > 0

/-- Calculates the work rate (portion of work done per day) given the work time -/
def work_rate (wt : WorkTime) : ℚ := 1 / wt.days

/-- Calculates the share of payment for a person given their work rate and the total work rate -/
def payment_share (individual_rate total_rate : ℚ) : ℚ := individual_rate / total_rate

theorem kim_payment_share 
  (kim : WorkTime)
  (david : WorkTime)
  (lisa : WorkTime)
  (h_kim : kim.days = 3)
  (h_david : david.days = 2)
  (h_lisa : lisa.days = 4)
  (total_payment : ℚ)
  (h_total_payment : total_payment = 200) :
  payment_share (work_rate kim) (work_rate kim + work_rate david + work_rate lisa) * total_payment = 800 / 13 :=
sorry

end NUMINAMATH_CALUDE_kim_payment_share_l1788_178899


namespace NUMINAMATH_CALUDE_cyclist_distance_l1788_178845

/-- Cyclist's travel problem -/
theorem cyclist_distance :
  ∀ (v t : ℝ),
  v > 0 →
  t > 0 →
  (v + 1) * (3/4 * t) = v * t →
  (v - 1) * (t + 3) = v * t →
  v * t = 18 :=
by sorry

end NUMINAMATH_CALUDE_cyclist_distance_l1788_178845


namespace NUMINAMATH_CALUDE_special_function_is_odd_and_even_l1788_178843

/-- A function satisfying the given property -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x + y ≠ 0 → f (x * y) = (f x + f y) / (x + y)

/-- A function is both odd and even -/
def odd_and_even (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x : ℝ, f (-x) = f x)

/-- The main theorem -/
theorem special_function_is_odd_and_even (f : ℝ → ℝ) (h : special_function f) :
  odd_and_even f :=
sorry

end NUMINAMATH_CALUDE_special_function_is_odd_and_even_l1788_178843


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l1788_178879

theorem cubic_equation_roots (p q : ℝ) : 
  (∃ a b c : ℕ+, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    ∀ x : ℝ, x^3 - 10*x^2 + p*x - q = 0 ↔ (x = a ∨ x = b ∨ x = c)) →
  p + q = 37 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l1788_178879


namespace NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l1788_178804

theorem largest_integer_satisfying_inequality :
  ∀ n : ℕ, n^200 < 5^300 ↔ n ≤ 11 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l1788_178804


namespace NUMINAMATH_CALUDE_three_times_root_equation_iff_roots_l1788_178873

/-- A quadratic equation ax^2 + bx + c = 0 (a ≠ 0) with two distinct real roots -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- Definition of a "3 times root equation" -/
def is_three_times_root_equation (eq : QuadraticEquation) : Prop :=
  ∃ (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ 
    eq.a * r₁^2 + eq.b * r₁ + eq.c = 0 ∧
    eq.a * r₂^2 + eq.b * r₂ + eq.c = 0 ∧
    r₂ = 3 * r₁

/-- Theorem: A quadratic equation is a "3 times root equation" iff its roots satisfy r2 = 3r1 -/
theorem three_times_root_equation_iff_roots (eq : QuadraticEquation) :
  is_three_times_root_equation eq ↔
  ∃ (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ 
    eq.a * r₁^2 + eq.b * r₁ + eq.c = 0 ∧
    eq.a * r₂^2 + eq.b * r₂ + eq.c = 0 ∧
    r₂ = 3 * r₁ :=
by sorry


end NUMINAMATH_CALUDE_three_times_root_equation_iff_roots_l1788_178873


namespace NUMINAMATH_CALUDE_sum_with_radical_conjugate_l1788_178828

theorem sum_with_radical_conjugate : 
  let x : ℝ := 15 - Real.sqrt 5000
  let y : ℝ := 15 + Real.sqrt 5000
  x + y = 30 := by
sorry

end NUMINAMATH_CALUDE_sum_with_radical_conjugate_l1788_178828


namespace NUMINAMATH_CALUDE_ellipse_parabola_intersection_bounds_l1788_178868

/-- If an ellipse and a parabola have a common point, then the parameter 'a' of the ellipse is bounded. -/
theorem ellipse_parabola_intersection_bounds (a : ℝ) : 
  (∃ x y : ℝ, x^2 + 4*(y - a)^2 = 4 ∧ x^2 = 2*y) → 
  -1 ≤ a ∧ a ≤ 17/8 := by
sorry

end NUMINAMATH_CALUDE_ellipse_parabola_intersection_bounds_l1788_178868


namespace NUMINAMATH_CALUDE_derivative_ln_plus_x_l1788_178889

open Real

theorem derivative_ln_plus_x (x : ℝ) (h : x > 0) : 
  deriv (fun x => log x + x) x = (x + 1) / x := by
sorry

end NUMINAMATH_CALUDE_derivative_ln_plus_x_l1788_178889


namespace NUMINAMATH_CALUDE_product_first_three_is_960_l1788_178807

/-- An arithmetic sequence with seventh term 20 and common difference 2 -/
def arithmetic_sequence (n : ℕ) : ℚ :=
  8 + 2 * (n - 1)

/-- The product of the first three terms of the arithmetic sequence -/
def product_first_three : ℚ :=
  (arithmetic_sequence 1) * (arithmetic_sequence 2) * (arithmetic_sequence 3)

theorem product_first_three_is_960 :
  product_first_three = 960 :=
by sorry

end NUMINAMATH_CALUDE_product_first_three_is_960_l1788_178807


namespace NUMINAMATH_CALUDE_speed_in_still_water_l1788_178820

theorem speed_in_still_water 
  (upstream_speed : ℝ) 
  (downstream_speed : ℝ) 
  (h1 : upstream_speed = 26) 
  (h2 : downstream_speed = 30) : 
  (upstream_speed + downstream_speed) / 2 = 28 := by
  sorry

end NUMINAMATH_CALUDE_speed_in_still_water_l1788_178820


namespace NUMINAMATH_CALUDE_system_solution_and_M_minimum_l1788_178801

-- Define the system of equations
def system (x y t : ℝ) : Prop :=
  x - 3*y = 4 - t ∧ x + y = 3*t

-- Define the range of t
def t_range (t : ℝ) : Prop :=
  -3 ≤ t ∧ t ≤ 1

-- Define M
def M (x y t : ℝ) : ℝ :=
  2*x - y - t

theorem system_solution_and_M_minimum :
  (∃ t, t_range t ∧ system 1 (-1) t) ∧
  (∀ x y t, t_range t → system x y t → M x y t ≥ -3) ∧
  (∃ x y t, t_range t ∧ system x y t ∧ M x y t = -3) :=
sorry

end NUMINAMATH_CALUDE_system_solution_and_M_minimum_l1788_178801


namespace NUMINAMATH_CALUDE_largest_and_smallest_results_l1788_178834

/-- The type representing our expression with parentheses -/
inductive Expr
  | num : ℕ → Expr
  | op : Expr → Expr → Expr

/-- Evaluate an expression to a rational number -/
def eval : Expr → ℚ
  | Expr.num n => n
  | Expr.op e₁ e₂ => (eval e₁) / (eval e₂)

/-- Check if a rational number is an integer -/
def isInteger (q : ℚ) : Prop := ∃ n : ℤ, q = n

/-- The set of all possible expressions using numbers 1 to 10 -/
def validExpr : Set Expr := sorry

/-- The theorem stating the largest and smallest possible integer results -/
theorem largest_and_smallest_results :
  (∃ e ∈ validExpr, eval e = 44800 ∧ isInteger (eval e)) ∧
  (∃ e ∈ validExpr, eval e = 7 ∧ isInteger (eval e)) ∧
  (∀ e ∈ validExpr, isInteger (eval e) → 7 ≤ eval e ∧ eval e ≤ 44800) :=
sorry

end NUMINAMATH_CALUDE_largest_and_smallest_results_l1788_178834


namespace NUMINAMATH_CALUDE_remainder_9_1995_mod_7_l1788_178818

theorem remainder_9_1995_mod_7 : 9^1995 % 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_9_1995_mod_7_l1788_178818


namespace NUMINAMATH_CALUDE_linear_congruence_intercepts_l1788_178862

/-- Proves the properties of x-intercept and y-intercept for the linear congruence equation 5x ≡ 3y + 2 (mod 27) -/
theorem linear_congruence_intercepts :
  ∃ (x₀ y₀ : ℕ),
    x₀ < 27 ∧
    y₀ < 27 ∧
    (5 * x₀) % 27 = 2 ∧
    (3 * y₀) % 27 = 25 ∧
    x₀ + y₀ = 40 := by
  sorry

end NUMINAMATH_CALUDE_linear_congruence_intercepts_l1788_178862


namespace NUMINAMATH_CALUDE_paths_through_point_c_l1788_178897

/-- The number of paths on a grid from (0,0) to (x,y) moving only right or up -/
def gridPaths (x y : ℕ) : ℕ := Nat.choose (x + y) y

/-- The total number of paths from A(0,0) to B(7,6) passing through C(3,2) on a 7x6 grid -/
def totalPaths : ℕ :=
  gridPaths 3 2 * gridPaths 4 3

theorem paths_through_point_c :
  totalPaths = 200 := by sorry

end NUMINAMATH_CALUDE_paths_through_point_c_l1788_178897


namespace NUMINAMATH_CALUDE_pepper_remaining_l1788_178802

theorem pepper_remaining (initial : Real) (used : Real) (remaining : Real) : 
  initial = 0.25 → used = 0.16 → remaining = initial - used → remaining = 0.09 := by
  sorry

end NUMINAMATH_CALUDE_pepper_remaining_l1788_178802


namespace NUMINAMATH_CALUDE_susan_work_hours_l1788_178847

/-- Susan's work problem -/
theorem susan_work_hours 
  (summer_weeks : ℕ) 
  (summer_hours_per_week : ℕ) 
  (summer_earnings : ℕ) 
  (school_weeks : ℕ) 
  (school_earnings : ℕ) 
  (h1 : summer_weeks = 10)
  (h2 : summer_hours_per_week = 60)
  (h3 : summer_earnings = 6000)
  (h4 : school_weeks = 50)
  (h5 : school_earnings = 6000) :
  ∃ (school_hours_per_week : ℕ),
    (summer_earnings : ℚ) / (summer_weeks * summer_hours_per_week : ℚ) * 
    (school_weeks * school_hours_per_week : ℚ) = school_earnings ∧
    school_hours_per_week = 12 :=
by sorry

end NUMINAMATH_CALUDE_susan_work_hours_l1788_178847


namespace NUMINAMATH_CALUDE_optimal_allocation_l1788_178831

/-- Represents an investment project --/
structure Project where
  maxProfitRate : ℝ
  maxLossRate : ℝ

/-- Represents an investment allocation --/
structure Allocation where
  projectA : ℝ
  projectB : ℝ

/-- Calculates the potential profit for a given allocation --/
def potentialProfit (projects : Project × Project) (alloc : Allocation) : ℝ :=
  alloc.projectA * projects.1.maxProfitRate + alloc.projectB * projects.2.maxProfitRate

/-- Calculates the potential loss for a given allocation --/
def potentialLoss (projects : Project × Project) (alloc : Allocation) : ℝ :=
  alloc.projectA * projects.1.maxLossRate + alloc.projectB * projects.2.maxLossRate

/-- Theorem: The optimal allocation maximizes profit while satisfying constraints --/
theorem optimal_allocation
  (projectA : Project)
  (projectB : Project)
  (totalLimit : ℝ)
  (lossLimit : ℝ)
  (h1 : projectA.maxProfitRate = 1)
  (h2 : projectB.maxProfitRate = 0.5)
  (h3 : projectA.maxLossRate = 0.3)
  (h4 : projectB.maxLossRate = 0.1)
  (h5 : totalLimit = 100000)
  (h6 : lossLimit = 18000) :
  ∃ (alloc : Allocation),
    alloc.projectA = 40000 ∧
    alloc.projectB = 60000 ∧
    alloc.projectA + alloc.projectB ≤ totalLimit ∧
    potentialLoss (projectA, projectB) alloc ≤ lossLimit ∧
    ∀ (otherAlloc : Allocation),
      otherAlloc.projectA + otherAlloc.projectB ≤ totalLimit →
      potentialLoss (projectA, projectB) otherAlloc ≤ lossLimit →
      potentialProfit (projectA, projectB) alloc ≥ potentialProfit (projectA, projectB) otherAlloc :=
by sorry

end NUMINAMATH_CALUDE_optimal_allocation_l1788_178831


namespace NUMINAMATH_CALUDE_smallest_prime_twelve_less_square_l1788_178848

theorem smallest_prime_twelve_less_square : 
  ∃ (n : ℕ), 
    (n > 0) ∧ 
    (Nat.Prime n) ∧ 
    (∃ (m : ℕ), n = m^2 - 12) ∧
    (∀ (k : ℕ), k > 0 → Nat.Prime k → (∃ (l : ℕ), k = l^2 - 12) → k ≥ n) ∧
    n = 13 :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_twelve_less_square_l1788_178848


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l1788_178863

theorem diophantine_equation_solution : 
  ∀ a b : ℤ, a > 0 ∧ b > 0 → (1 : ℚ) / a - (1 : ℚ) / b = (1 : ℚ) / 37 → (a = 38 ∧ b = 1332) :=
by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l1788_178863


namespace NUMINAMATH_CALUDE_mildred_spending_l1788_178860

def total_given : ℕ := 100
def amount_left : ℕ := 40
def candice_spent : ℕ := 35

theorem mildred_spending :
  total_given - amount_left - candice_spent = 25 :=
by sorry

end NUMINAMATH_CALUDE_mildred_spending_l1788_178860


namespace NUMINAMATH_CALUDE_ant_path_count_l1788_178893

/-- The number of paths from A to B -/
def paths_AB : ℕ := 3

/-- The number of paths from B to C -/
def paths_BC : ℕ := 3

/-- The total number of paths from A to C through B -/
def total_paths : ℕ := paths_AB * paths_BC

/-- Theorem stating that the total number of paths from A to C through B is 9 -/
theorem ant_path_count : total_paths = 9 := by
  sorry

end NUMINAMATH_CALUDE_ant_path_count_l1788_178893


namespace NUMINAMATH_CALUDE_correct_sum_and_digit_change_l1788_178864

theorem correct_sum_and_digit_change : ∃ (d e : ℕ), 
  (d ≤ 9 ∧ e ≤ 9) ∧ 
  (553672 + 637528 = 1511200) ∧ 
  (d + e = 14) ∧
  (953672 + 637528 ≠ 1511200) := by
sorry

end NUMINAMATH_CALUDE_correct_sum_and_digit_change_l1788_178864


namespace NUMINAMATH_CALUDE_statement_S_holds_for_options_l1788_178840

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def options : List ℕ := [90, 99, 108, 117]

theorem statement_S_holds_for_options : ∀ n ∈ options, 
  (sum_of_digits n) % 9 = 0 → n % 3 = 0 := by sorry

end NUMINAMATH_CALUDE_statement_S_holds_for_options_l1788_178840


namespace NUMINAMATH_CALUDE_wheel_distance_l1788_178894

/-- Given two wheels with different perimeters, prove that the distance traveled
    is 315 feet when the front wheel makes 10 more revolutions than the back wheel. -/
theorem wheel_distance (back_perimeter front_perimeter : ℝ) 
  (h1 : back_perimeter = 9)
  (h2 : front_perimeter = 7)
  (h3 : ∃ (back_revs front_revs : ℝ), 
    front_revs = back_revs + 10 ∧ 
    back_revs * back_perimeter = front_revs * front_perimeter) :
  ∃ (distance : ℝ), distance = 315 := by
  sorry

end NUMINAMATH_CALUDE_wheel_distance_l1788_178894


namespace NUMINAMATH_CALUDE_buccaneer_loot_sum_l1788_178812

def base5ToBase10 (n : List Nat) : Nat :=
  n.enum.foldr (fun (i, d) acc => acc + d * (5 ^ i)) 0

theorem buccaneer_loot_sum : 
  let pearls := base5ToBase10 [1, 2, 3, 4]
  let silk := base5ToBase10 [1, 1, 1, 1]
  let spices := base5ToBase10 [1, 2, 2]
  let maps := base5ToBase10 [0, 1]
  pearls + silk + spices + maps = 808 := by sorry

end NUMINAMATH_CALUDE_buccaneer_loot_sum_l1788_178812


namespace NUMINAMATH_CALUDE_quadratic_roots_average_l1788_178842

theorem quadratic_roots_average (c : ℝ) : 
  ∃ (x₁ x₂ : ℝ), (2 * x₁^2 - 4 * x₁ + c = 0) ∧ 
                 (2 * x₂^2 - 4 * x₂ + c = 0) ∧ 
                 (x₁ ≠ x₂) → 
                 (x₁ + x₂) / 2 = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_average_l1788_178842


namespace NUMINAMATH_CALUDE_monkey_peach_problem_l1788_178810

theorem monkey_peach_problem :
  ∀ (num_monkeys num_peaches : ℕ),
    (num_peaches = 14 * num_monkeys + 48) →
    (num_peaches = 18 * num_monkeys - 64) →
    (num_monkeys = 28 ∧ num_peaches = 440) :=
by
  sorry

end NUMINAMATH_CALUDE_monkey_peach_problem_l1788_178810


namespace NUMINAMATH_CALUDE_sum_odd_numbers_100_to_200_l1788_178871

def sum_odd_numbers_between (a b : ℕ) : ℕ :=
  let first_odd := if a % 2 = 0 then a + 1 else a
  let last_odd := if b % 2 = 0 then b - 1 else b
  let n := (last_odd - first_odd) / 2 + 1
  n * (first_odd + last_odd) / 2

theorem sum_odd_numbers_100_to_200 :
  sum_odd_numbers_between 100 200 = 7500 := by
  sorry

end NUMINAMATH_CALUDE_sum_odd_numbers_100_to_200_l1788_178871


namespace NUMINAMATH_CALUDE_nested_fraction_simplification_l1788_178892

theorem nested_fraction_simplification :
  2 + (3 / (4 + (5 / (6 + (7/8))))) = 137/52 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_simplification_l1788_178892


namespace NUMINAMATH_CALUDE_nested_squares_segment_length_l1788_178888

/-- Given four nested squares with known segment lengths, prove that the length of GH
    is the sum of lengths AB, CD, and FE. -/
theorem nested_squares_segment_length 
  (AB CD FE : ℝ) 
  (h1 : AB = 11) 
  (h2 : CD = 5) 
  (h3 : FE = 13) : 
  ∃ GH : ℝ, GH = AB + CD + FE :=
by sorry

end NUMINAMATH_CALUDE_nested_squares_segment_length_l1788_178888


namespace NUMINAMATH_CALUDE_pencils_leftover_l1788_178815

theorem pencils_leftover : Int.mod 33333332 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_pencils_leftover_l1788_178815


namespace NUMINAMATH_CALUDE_distribute_five_to_two_nonempty_l1788_178803

theorem distribute_five_to_two_nonempty (n : Nat) (k : Nat) : 
  n = 5 → k = 2 → (Finset.sum (Finset.range (n - 1)) (λ i => Nat.choose n (i + 1) * 2)) = 30 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_to_two_nonempty_l1788_178803


namespace NUMINAMATH_CALUDE_karens_class_size_l1788_178841

def total_cookies : ℕ := 50
def kept_cookies : ℕ := 10
def grandparents_cookies : ℕ := 8
def cookies_per_classmate : ℕ := 2

theorem karens_class_size :
  (total_cookies - kept_cookies - grandparents_cookies) / cookies_per_classmate = 16 := by
  sorry

end NUMINAMATH_CALUDE_karens_class_size_l1788_178841


namespace NUMINAMATH_CALUDE_sanitizer_sales_theorem_l1788_178880

/-- Represents the hand sanitizer sales problem -/
structure SanitizerSales where
  cost : ℝ  -- Cost per bottle in yuan
  initial_price : ℝ  -- Initial selling price per bottle in yuan
  initial_volume : ℝ  -- Initial daily sales volume
  price_sensitivity : ℝ  -- Decrease in sales for every 1 yuan increase in price
  x : ℝ  -- Increase in selling price

/-- Calculates the daily sales volume given the price increase -/
def daily_volume (s : SanitizerSales) : ℝ :=
  s.initial_volume - s.price_sensitivity * s.x

/-- Calculates the profit per bottle given the price increase -/
def profit_per_bottle (s : SanitizerSales) : ℝ :=
  (s.initial_price - s.cost) + s.x

/-- Calculates the daily profit given the price increase -/
def daily_profit (s : SanitizerSales) : ℝ :=
  (daily_volume s) * (profit_per_bottle s)

/-- The main theorem about the sanitizer sales problem -/
theorem sanitizer_sales_theorem (s : SanitizerSales) 
  (h1 : s.cost = 16)
  (h2 : s.initial_price = 20)
  (h3 : s.initial_volume = 60)
  (h4 : s.price_sensitivity = 5) :
  (daily_volume s = 60 - 5 * s.x) ∧
  (profit_per_bottle s = 4 + s.x) ∧
  (daily_profit s = 300 → s.x = 2 ∨ s.x = 6) ∧
  (∃ (max_profit : ℝ), max_profit = 320 ∧ 
    ∀ (y : ℝ), y = daily_profit s → y ≤ max_profit ∧
    (y = max_profit ↔ s.x = 4)) := by
  sorry


end NUMINAMATH_CALUDE_sanitizer_sales_theorem_l1788_178880


namespace NUMINAMATH_CALUDE_segment_ratio_l1788_178876

/-- Given four distinct points on a plane with segments of lengths a, a, b, a+√3b, 2a, and 2b
    connecting them, the ratio of b to a is 2 + √3. -/
theorem segment_ratio (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  ∃ (p1 p2 p3 p4 : ℝ × ℝ), 
    p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 ∧
    ({dist p1 p2, dist p1 p3, dist p1 p4, dist p2 p3, dist p2 p4, dist p3 p4} : Finset ℝ) = 
      {a, a, b, a + Real.sqrt 3 * b, 2 * a, 2 * b} →
    b / a = 2 + Real.sqrt 3 := by
  sorry

#check segment_ratio

end NUMINAMATH_CALUDE_segment_ratio_l1788_178876
