import Mathlib

namespace NUMINAMATH_CALUDE_percentage_sum_l3522_352297

theorem percentage_sum : 
  (20 / 100 * 30) + (15 / 100 * 50) + (25 / 100 * 120) + (-10 / 100 * 45) = 39 := by
  sorry

end NUMINAMATH_CALUDE_percentage_sum_l3522_352297


namespace NUMINAMATH_CALUDE_jenny_project_hours_l3522_352244

/-- The total hours Jenny has to work on her school project -/
def total_project_hours (research_hours proposal_hours report_hours : ℕ) : ℕ :=
  research_hours + proposal_hours + report_hours

/-- Theorem stating that Jenny's total project hours is 20 -/
theorem jenny_project_hours :
  total_project_hours 10 2 8 = 20 := by
  sorry

end NUMINAMATH_CALUDE_jenny_project_hours_l3522_352244


namespace NUMINAMATH_CALUDE_apples_per_crate_value_l3522_352270

/-- The number of apples in each crate -/
def apples_per_crate : ℕ := sorry

/-- The total number of crates -/
def total_crates : ℕ := 12

/-- The number of rotten apples -/
def rotten_apples : ℕ := 160

/-- The number of boxes filled with good apples -/
def filled_boxes : ℕ := 100

/-- The number of apples in each box -/
def apples_per_box : ℕ := 20

theorem apples_per_crate_value : apples_per_crate = 180 := by sorry

end NUMINAMATH_CALUDE_apples_per_crate_value_l3522_352270


namespace NUMINAMATH_CALUDE_x_zero_necessary_not_sufficient_l3522_352250

def a : ℝ × ℝ := (1, 1)
def b (x : ℝ) : ℝ × ℝ := (x, -1)

theorem x_zero_necessary_not_sufficient :
  ∃ (x : ℝ), x ≠ 0 ∧ (a + b x) • (b x) = 0 ∧
  ∀ (y : ℝ), (a + b y) • (b y) = 0 → y = 0 ∨ y = -1 :=
by sorry

end NUMINAMATH_CALUDE_x_zero_necessary_not_sufficient_l3522_352250


namespace NUMINAMATH_CALUDE_basketball_probability_l3522_352224

theorem basketball_probability (p_no_make : ℝ) (num_tries : ℕ) : 
  p_no_make = 1/3 → num_tries = 3 → 
  let p_make := 1 - p_no_make
  (num_tries.choose 1) * p_make * p_no_make^2 = 2/9 := by
sorry

end NUMINAMATH_CALUDE_basketball_probability_l3522_352224


namespace NUMINAMATH_CALUDE_no_three_rational_solutions_l3522_352203

theorem no_three_rational_solutions :
  ¬ ∃ (r : ℝ), ∃ (x y z : ℚ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    (x^3 - 2023*x^2 - 2023*x + r = 0) ∧
    (y^3 - 2023*y^2 - 2023*y + r = 0) ∧
    (z^3 - 2023*z^2 - 2023*z + r = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_three_rational_solutions_l3522_352203


namespace NUMINAMATH_CALUDE_nine_seats_six_people_arrangement_l3522_352248

/-- The number of ways to arrange people and empty seats in a row -/
def seating_arrangements (total_seats : ℕ) (people : ℕ) : ℕ :=
  (Nat.factorial people) * (Nat.choose (people - 1) (total_seats - people))

/-- Theorem: There are 7200 ways to arrange 6 people and 3 empty seats in a row of 9 seats,
    where every empty seat is flanked by people on both sides -/
theorem nine_seats_six_people_arrangement :
  seating_arrangements 9 6 = 7200 := by
  sorry

end NUMINAMATH_CALUDE_nine_seats_six_people_arrangement_l3522_352248


namespace NUMINAMATH_CALUDE_cube_volume_from_space_diagonal_l3522_352266

theorem cube_volume_from_space_diagonal :
  ∀ s : ℝ,
  s > 0 →
  s * Real.sqrt 3 = 10 * Real.sqrt 3 →
  s^3 = 1000 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_space_diagonal_l3522_352266


namespace NUMINAMATH_CALUDE_distance_to_focus_l3522_352268

/-- A point on a parabola with a specific distance property -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 4*x
  distance_to_line : |x + 3| = 5

/-- The theorem stating the distance from the point to the focus -/
theorem distance_to_focus (P : ParabolaPoint) :
  Real.sqrt ((P.x - 1)^2 + P.y^2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_focus_l3522_352268


namespace NUMINAMATH_CALUDE_industrial_lubricants_percentage_l3522_352211

theorem industrial_lubricants_percentage 
  (microphotonics : ℝ) 
  (home_electronics : ℝ) 
  (food_additives : ℝ) 
  (genetically_modified_microorganisms : ℝ) 
  (basic_astrophysics_degrees : ℝ) :
  microphotonics = 14 →
  home_electronics = 24 →
  food_additives = 20 →
  genetically_modified_microorganisms = 29 →
  basic_astrophysics_degrees = 18 →
  let basic_astrophysics := (basic_astrophysics_degrees / 360) * 100
  let total_known := microphotonics + home_electronics + food_additives + 
                     genetically_modified_microorganisms + basic_astrophysics
  let industrial_lubricants := 100 - total_known
  industrial_lubricants = 8 :=
by sorry

end NUMINAMATH_CALUDE_industrial_lubricants_percentage_l3522_352211


namespace NUMINAMATH_CALUDE_no_simultaneous_squares_l3522_352261

theorem no_simultaneous_squares (x y : ℕ) : ¬(∃ (a b : ℕ), x^2 + y = a^2 ∧ y^2 + x = b^2) := by
  sorry

end NUMINAMATH_CALUDE_no_simultaneous_squares_l3522_352261


namespace NUMINAMATH_CALUDE_store_a_more_cost_effective_l3522_352292

/-- Represents the cost of purchasing tennis equipment from two different stores -/
def tennis_purchase_cost (x : ℝ) : Prop :=
  x > 40 ∧ 
  (25 * x + 3000 < 22.5 * x + 3600) = (x > 120)

/-- Theorem stating that Store A is more cost-effective when x = 100 -/
theorem store_a_more_cost_effective : tennis_purchase_cost 100 := by
  sorry

end NUMINAMATH_CALUDE_store_a_more_cost_effective_l3522_352292


namespace NUMINAMATH_CALUDE_intersection_points_theorem_l3522_352246

theorem intersection_points_theorem : 
  ∃ (k₁ k₂ k₃ k₄ : ℕ+), 
    k₁ + k₂ + k₃ + k₄ = 100 ∧ 
    k₁^2 + k₂^2 + k₃^2 + k₄^2 = 5996 ∧
    k₁ * k₂ + k₁ * k₃ + k₁ * k₄ + k₂ * k₃ + k₂ * k₄ + k₃ * k₄ = 2002 :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_theorem_l3522_352246


namespace NUMINAMATH_CALUDE_rectangle_width_decrease_l3522_352232

theorem rectangle_width_decrease (L W : ℝ) (L_new W_new A_new : ℝ) 
  (h1 : L_new = 1.6 * L)
  (h2 : A_new = 1.36 * (L * W))
  (h3 : A_new = L_new * W_new) :
  W_new = 0.85 * W := by
sorry

end NUMINAMATH_CALUDE_rectangle_width_decrease_l3522_352232


namespace NUMINAMATH_CALUDE_quadratic_roots_less_than_one_l3522_352286

theorem quadratic_roots_less_than_one (a b : ℝ) 
  (h1 : abs a + abs b < 1) 
  (h2 : a^2 - 4*b ≥ 0) : 
  ∀ x, x^2 + a*x + b = 0 → abs x < 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_less_than_one_l3522_352286


namespace NUMINAMATH_CALUDE_fraction_equality_l3522_352281

theorem fraction_equality (m n p q : ℚ) 
  (h1 : m / n = 25)
  (h2 : p / n = 5)
  (h3 : p / q = 1 / 15) :
  m / q = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l3522_352281


namespace NUMINAMATH_CALUDE_inequality_range_l3522_352265

theorem inequality_range (a : ℝ) : 
  (∀ x ∈ Set.Icc (-1) 2, x^2 - a*x - 3 < 0) → 
  a ∈ Set.Ioo (1/2) 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l3522_352265


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3522_352242

/-- Given a = 2 and b = -1/2, prove that a - 2(a - b^2) + 3(-a + b^2) = -27/4 -/
theorem simplify_and_evaluate (a b : ℚ) (ha : a = 2) (hb : b = -1/2) :
  a - 2 * (a - b^2) + 3 * (-a + b^2) = -27/4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3522_352242


namespace NUMINAMATH_CALUDE_triangle_problem_l3522_352201

theorem triangle_problem (A B C : Real) (a b c : Real) :
  -- Conditions
  (0 < A ∧ A < π) →
  (0 < B ∧ B < π) →
  (0 < C ∧ C < π) →
  (A + B + C = π) →
  (a * Real.sin B + b * Real.cos A = c) →
  (a = Real.sqrt 2 * c) →
  (b = 2) →
  -- Conclusions
  (B = π / 4 ∧ c = 2) :=
by sorry


end NUMINAMATH_CALUDE_triangle_problem_l3522_352201


namespace NUMINAMATH_CALUDE_circle_equation_for_given_points_l3522_352276

/-- Given two points A and B, this function returns the standard equation of the circle
    with AB as its diameter in the form (x - h)^2 + (y - k)^2 = r^2,
    where (h, k) is the center and r is the radius. -/
def circleEquationFromDiameter (A B : ℝ × ℝ) : ℝ × ℝ × ℝ := by sorry

/-- Theorem stating that for points A(1, -4) and B(-5, 4),
    the standard equation of the circle with AB as its diameter is (x + 2)^2 + y^2 = 25 -/
theorem circle_equation_for_given_points :
  let A : ℝ × ℝ := (1, -4)
  let B : ℝ × ℝ := (-5, 4)
  let (h, k, r) := circleEquationFromDiameter A B
  h = -2 ∧ k = 0 ∧ r^2 = 25 := by sorry

end NUMINAMATH_CALUDE_circle_equation_for_given_points_l3522_352276


namespace NUMINAMATH_CALUDE_quadratic_equation_root_l3522_352285

theorem quadratic_equation_root (k : ℚ) : 
  (∃ x : ℚ, x - x^2 = k*x^2 + 1) → 
  (2 - 2^2 = k*2^2 + 1) → 
  k = -3/4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_l3522_352285


namespace NUMINAMATH_CALUDE_max_square_plots_l3522_352237

/-- Represents the dimensions of the rectangular field -/
structure FieldDimensions where
  length : ℕ
  width : ℕ

/-- Represents the available internal fencing -/
def availableFence : ℕ := 1500

/-- Calculates the number of square plots given the side length of a square -/
def numPlots (d : FieldDimensions) (side : ℕ) : ℕ :=
  (d.length / side) * (d.width / side)

/-- Calculates the amount of internal fencing needed for a given number of squares per side -/
def fencingNeeded (d : FieldDimensions) (squaresPerSide : ℕ) : ℕ :=
  (d.length * (squaresPerSide - 1)) + (d.width * (squaresPerSide - 1))

/-- Theorem stating that 576 is the maximum number of square plots -/
theorem max_square_plots (d : FieldDimensions) (h1 : d.length = 30) (h2 : d.width = 45) :
  ∀ n : ℕ, numPlots d n ≤ 576 ∧ fencingNeeded d (d.width / (d.width / 24)) ≤ availableFence :=
sorry

end NUMINAMATH_CALUDE_max_square_plots_l3522_352237


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l3522_352245

/-- A rhombus with given diagonal lengths has a specific perimeter. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) :
  let side := Real.sqrt ((d1/2)^2 + (d2/2)^2)
  4 * side = 52 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l3522_352245


namespace NUMINAMATH_CALUDE_f_properties_l3522_352225

/-- An odd function f(x) with a parameter a ≠ 0 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log ((a * x) / (1 - x) + 1)

/-- The function f is odd -/
axiom f_odd (a : ℝ) (h : a ≠ 0) : ∀ x, f a (-x) = -(f a x)

/-- Theorem stating the properties of the function f -/
theorem f_properties :
  ∃ (a : ℝ), a ≠ 0 ∧ 
  (a = 2) ∧ 
  (∀ x, f a x ≠ 0 ↔ -1 < x ∧ x < 1) ∧
  (∀ x y, -1 < x ∧ x < y ∧ y < 1 → f a x < f a y) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l3522_352225


namespace NUMINAMATH_CALUDE_doug_fires_count_l3522_352240

theorem doug_fires_count (doug kai eli total : ℕ) 
  (h1 : kai = 3 * doug)
  (h2 : eli = kai / 2)
  (h3 : doug + kai + eli = total)
  (h4 : total = 110) : doug = 20 := by
  sorry

end NUMINAMATH_CALUDE_doug_fires_count_l3522_352240


namespace NUMINAMATH_CALUDE_parabola_through_negative_x_l3522_352278

/-- A parabola passing through the point (-2, 3) cannot have a standard equation of the form y^2 = 2px where p > 0 -/
theorem parabola_through_negative_x (p : ℝ) (h : p > 0) : ¬ (3^2 = 2 * p * (-2)) := by
  sorry

end NUMINAMATH_CALUDE_parabola_through_negative_x_l3522_352278


namespace NUMINAMATH_CALUDE_hyperbola_range_l3522_352288

theorem hyperbola_range (m : ℝ) : 
  (∃ x y : ℝ, x^2 / (m - 2) + y^2 / (m + 3) = 1) ↔ -3 < m ∧ m < 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_range_l3522_352288


namespace NUMINAMATH_CALUDE_solve_a_and_b_l3522_352257

theorem solve_a_and_b : ∃ (a b : ℝ), 
  (b^2 - 2*b = 24) ∧ 
  (4*(1:ℝ)^2 + a = 2) ∧ 
  (4*b^2 - 2*b = 72) ∧ 
  (a = -2) ∧ 
  (b = -4) := by
  sorry

end NUMINAMATH_CALUDE_solve_a_and_b_l3522_352257


namespace NUMINAMATH_CALUDE_sum_of_15th_set_l3522_352274

/-- The first element of the nth set in the sequence -/
def first_element (n : ℕ) : ℕ :=
  1 + (n * (n - 1)) / 2

/-- The number of elements in the nth set -/
def set_size (n : ℕ) : ℕ := n

/-- The sum of elements in the nth set -/
def S (n : ℕ) : ℕ :=
  let a := first_element n
  let l := a + set_size n - 1
  (set_size n * (a + l)) / 2

/-- Theorem: The sum of elements in the 15th set is 1695 -/
theorem sum_of_15th_set : S 15 = 1695 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_15th_set_l3522_352274


namespace NUMINAMATH_CALUDE_extreme_values_imply_a_range_l3522_352279

/-- A function f(x) with parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - 2*x + a * Real.log x

/-- The derivative of f(x) with respect to x -/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := x - 2 + a / x

/-- Theorem stating that if f(x) has two distinct extreme values, then 0 < a < 1 -/
theorem extreme_values_imply_a_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ 
    f_derivative a x₁ = 0 ∧ f_derivative a x₂ = 0) →
  0 < a ∧ a < 1 := by sorry

end NUMINAMATH_CALUDE_extreme_values_imply_a_range_l3522_352279


namespace NUMINAMATH_CALUDE_cubic_expression_factorization_l3522_352262

theorem cubic_expression_factorization (a b c : ℝ) :
  ((a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3) / ((a - b)^3 + (b - c)^3 + (c - a)^3) =
  (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) :=
by sorry

end NUMINAMATH_CALUDE_cubic_expression_factorization_l3522_352262


namespace NUMINAMATH_CALUDE_student_wrestling_match_l3522_352247

theorem student_wrestling_match (n : ℕ) : n * (n - 1) / 2 = 91 → n = 14 := by
  sorry

end NUMINAMATH_CALUDE_student_wrestling_match_l3522_352247


namespace NUMINAMATH_CALUDE_decagon_diagonals_l3522_352269

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A decagon has 10 sides -/
def decagon_sides : ℕ := 10

theorem decagon_diagonals : num_diagonals decagon_sides = 35 := by
  sorry

end NUMINAMATH_CALUDE_decagon_diagonals_l3522_352269


namespace NUMINAMATH_CALUDE_factorial_500_trailing_zeroes_l3522_352228

def trailingZeroes (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

theorem factorial_500_trailing_zeroes :
  trailingZeroes 500 = 124 := by
  sorry

end NUMINAMATH_CALUDE_factorial_500_trailing_zeroes_l3522_352228


namespace NUMINAMATH_CALUDE_black_squares_21st_row_l3522_352291

/-- Represents the number of squares in a row of the stair-step figure -/
def squares_in_row (n : ℕ) : ℕ := 2 * n

/-- Represents the number of black squares in a row of the stair-step figure -/
def black_squares_in_row (n : ℕ) : ℕ := 2 * (squares_in_row n / 4)

theorem black_squares_21st_row :
  black_squares_in_row 21 = 20 := by
  sorry

end NUMINAMATH_CALUDE_black_squares_21st_row_l3522_352291


namespace NUMINAMATH_CALUDE_bus_seating_capacity_l3522_352277

theorem bus_seating_capacity 
  (total_students : ℕ) 
  (bus_capacity : ℕ) 
  (h1 : 4 * bus_capacity + 30 = total_students) 
  (h2 : 5 * bus_capacity = total_students + 10) : 
  bus_capacity = 40 := by
sorry

end NUMINAMATH_CALUDE_bus_seating_capacity_l3522_352277


namespace NUMINAMATH_CALUDE_inequality_always_holds_l3522_352206

theorem inequality_always_holds (a b c : ℝ) (h : b > c) : a^2 + b > a^2 + c := by
  sorry

end NUMINAMATH_CALUDE_inequality_always_holds_l3522_352206


namespace NUMINAMATH_CALUDE_circle_equation_l3522_352230

theorem circle_equation (center : ℝ × ℝ) (p1 p2 : ℝ × ℝ) :
  center.1 + center.2 = 0 →
  p1 = (0, 2) →
  p2 = (-4, 0) →
  ∀ x y : ℝ, ((x - center.1)^2 + (y - center.2)^2 = (p1.1 - center.1)^2 + (p1.2 - center.2)^2) ↔
              ((x - center.1)^2 + (y - center.2)^2 = (p2.1 - center.1)^2 + (p2.2 - center.2)^2) →
  ∃ a b r : ℝ, (x + a)^2 + (y - b)^2 = r ∧ a = 3 ∧ b = 3 ∧ r = 10 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l3522_352230


namespace NUMINAMATH_CALUDE_cookie_difference_l3522_352213

/-- Proves that Cristian had 50 more black cookies than white cookies initially -/
theorem cookie_difference (black_cookies white_cookies : ℕ) : 
  white_cookies = 80 →
  black_cookies > white_cookies →
  black_cookies / 2 + white_cookies / 4 = 85 →
  black_cookies - white_cookies = 50 :=
by
  sorry

#check cookie_difference

end NUMINAMATH_CALUDE_cookie_difference_l3522_352213


namespace NUMINAMATH_CALUDE_positive_sum_squares_bound_l3522_352222

theorem positive_sum_squares_bound (x y z a : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (ha : a > 0)
  (sum_eq : x + y + z = a)
  (sum_squares_eq : x^2 + y^2 + z^2 = a^2 / 2) :
  x ≤ 2*a/3 ∧ y ≤ 2*a/3 ∧ z ≤ 2*a/3 :=
by sorry

end NUMINAMATH_CALUDE_positive_sum_squares_bound_l3522_352222


namespace NUMINAMATH_CALUDE_basketball_prices_and_discounts_l3522_352209

/-- Represents the prices and quantities of basketballs -/
structure BasketballPrices where
  price_a : ℝ
  price_b : ℝ
  quantity_a : ℝ
  quantity_b : ℝ

/-- Represents the discount options -/
inductive DiscountOption
  | Percent10
  | Buy3Get1Free

/-- The main theorem about basketball prices and discount options -/
theorem basketball_prices_and_discounts 
  (prices : BasketballPrices)
  (x : ℝ) :
  prices.price_a = prices.price_b + 40 →
  1200 / prices.price_a = 600 / prices.price_b →
  x ≥ 5 →
  (prices.price_a = 80 ∧ prices.price_b = 40) ∧
  (∀ y : ℝ, 
    y > 20 → (0.9 * (80 * 15 + 40 * y) < 80 * 15 + 40 * (y - 15 / 3)) ∧
    y = 20 → (0.9 * (80 * 15 + 40 * y) = 80 * 15 + 40 * (y - 15 / 3)) ∧
    y < 20 → (0.9 * (80 * 15 + 40 * y) > 80 * 15 + 40 * (y - 15 / 3))) := by
  sorry

#check basketball_prices_and_discounts

end NUMINAMATH_CALUDE_basketball_prices_and_discounts_l3522_352209


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3522_352264

theorem polynomial_simplification (x : ℝ) :
  3 - 5*x - 7*x^2 + 9 + 11*x - 13*x^2 - 15 + 17*x + 19*x^2 = -x^2 + 23*x - 3 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3522_352264


namespace NUMINAMATH_CALUDE_inverse_value_equivalence_l3522_352238

-- Define the function f
def f (x : ℝ) : ℝ := 7 * x^3 - 2 * x^2 + 5 * x - 9

-- Theorem stating that finding f⁻¹(-3.5) is equivalent to solving 7x³ - 2x² + 5x - 5.5 = 0
theorem inverse_value_equivalence :
  ∀ x : ℝ, f x = -3.5 ↔ 7 * x^3 - 2 * x^2 + 5 * x - 5.5 = 0 :=
by
  sorry

-- Note: The actual inverse function is not defined as it's not expressible in elementary functions

end NUMINAMATH_CALUDE_inverse_value_equivalence_l3522_352238


namespace NUMINAMATH_CALUDE_polygon_sides_l3522_352289

theorem polygon_sides (S : ℕ) (h : S = 2160) : ∃ n : ℕ, n = 14 ∧ S = 180 * (n - 2) := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l3522_352289


namespace NUMINAMATH_CALUDE_arithmetic_series_sum_l3522_352205

theorem arithmetic_series_sum : 
  ∀ (a₁ aₙ d : ℚ) (n : ℕ),
    a₁ = 25 →
    aₙ = 50 →
    d = 2/5 →
    aₙ = a₁ + (n - 1) * d →
    (n : ℚ) * (a₁ + aₙ) / 2 = 2400 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_series_sum_l3522_352205


namespace NUMINAMATH_CALUDE_m_plus_2n_equals_neg_one_l3522_352260

theorem m_plus_2n_equals_neg_one (m n : ℝ) (h : |m - 3| + (n + 2)^2 = 0) : m + 2*n = -1 := by
  sorry

end NUMINAMATH_CALUDE_m_plus_2n_equals_neg_one_l3522_352260


namespace NUMINAMATH_CALUDE_product_is_three_digit_l3522_352229

def smallest_three_digit_number : ℕ := 100
def largest_single_digit_number : ℕ := 9

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem product_is_three_digit : 
  is_three_digit (smallest_three_digit_number * largest_single_digit_number) := by
  sorry

end NUMINAMATH_CALUDE_product_is_three_digit_l3522_352229


namespace NUMINAMATH_CALUDE_school_committee_formation_l3522_352214

theorem school_committee_formation (n_children : ℕ) (n_teachers : ℕ) (committee_size : ℕ) :
  n_children = 12 →
  n_teachers = 3 →
  committee_size = 9 →
  (Nat.choose (n_children + n_teachers) committee_size) - (Nat.choose n_children committee_size) = 4785 :=
by sorry

end NUMINAMATH_CALUDE_school_committee_formation_l3522_352214


namespace NUMINAMATH_CALUDE_number_of_combinations_is_736_l3522_352293

/-- Represents the number of different ways to occupy planets given the specified conditions --/
def number_of_combinations : ℕ :=
  let earth_like_planets : ℕ := 7
  let mars_like_planets : ℕ := 6
  let earth_like_units : ℕ := 3
  let mars_like_units : ℕ := 1
  let total_units : ℕ := 15

  -- The actual calculation of combinations
  0 -- placeholder, replace with actual calculation

/-- Theorem stating that the number of combinations is 736 --/
theorem number_of_combinations_is_736 : number_of_combinations = 736 := by
  sorry


end NUMINAMATH_CALUDE_number_of_combinations_is_736_l3522_352293


namespace NUMINAMATH_CALUDE_unique_solution_for_abc_l3522_352275

theorem unique_solution_for_abc (a b c : ℝ) 
  (ha : a > 2) (hb : b > 2) (hc : c > 2)
  (heq : (a - 1)^2 / (b + c + 1) + (b + 1)^2 / (c + a - 1) + (c + 5)^2 / (a + b - 5) = 49) :
  a = 10.5 ∧ b = 10 ∧ c = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_abc_l3522_352275


namespace NUMINAMATH_CALUDE_opposite_of_negative_2011_l3522_352231

theorem opposite_of_negative_2011 : 
  -((-2011) : ℤ) = (2011 : ℤ) := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2011_l3522_352231


namespace NUMINAMATH_CALUDE_show_attendance_l3522_352299

theorem show_attendance (adult_price children_price total_receipts : ℚ)
  (h1 : adult_price = 5.5)
  (h2 : children_price = 2.5)
  (h3 : total_receipts = 1026) :
  ∃ (adults children : ℕ),
    adults = 2 * children ∧
    adult_price * adults + children_price * children = total_receipts ∧
    adults = 152 :=
by sorry

end NUMINAMATH_CALUDE_show_attendance_l3522_352299


namespace NUMINAMATH_CALUDE_max_red_socks_l3522_352287

theorem max_red_socks (r b : ℕ) : 
  let t := r + b
  r + b ≤ 2000 →
  (r * (r - 1) + b * (b - 1)) / (t * (t - 1)) = 5 / 12 →
  r ≤ 109 :=
by sorry

end NUMINAMATH_CALUDE_max_red_socks_l3522_352287


namespace NUMINAMATH_CALUDE_perpendicular_length_l3522_352221

/-- Given two oblique lines and their projections, find the perpendicular length -/
theorem perpendicular_length
  (oblique1 oblique2 : ℝ)
  (projection_ratio : ℚ)
  (h1 : oblique1 = 41)
  (h2 : oblique2 = 50)
  (h3 : projection_ratio = 3 / 10) :
  ∃ (perpendicular : ℝ), perpendicular = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_length_l3522_352221


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l3522_352217

def M : Set ℝ := {x | x^2 - x = 0}
def N : Set ℝ := {y | y^2 + y = 0}

theorem union_of_M_and_N : M ∪ N = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l3522_352217


namespace NUMINAMATH_CALUDE_three_digit_divisible_by_26_l3522_352263

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def first_digit (n : ℕ) : ℕ := n / 100

def second_digit (n : ℕ) : ℕ := (n / 10) % 10

def third_digit (n : ℕ) : ℕ := n % 10

def sum_of_squared_digits (n : ℕ) : ℕ :=
  (first_digit n)^2 + (second_digit n)^2 + (third_digit n)^2

def valid_number (n : ℕ) : Prop :=
  is_three_digit n ∧ 
  first_digit n ≠ 0 ∧
  26 % (sum_of_squared_digits n) = 0

theorem three_digit_divisible_by_26 :
  {n : ℕ | valid_number n} = 
  {100, 110, 101, 302, 320, 230, 203, 431, 413, 314, 341, 134, 143, 510, 501, 150, 105} :=
by sorry

end NUMINAMATH_CALUDE_three_digit_divisible_by_26_l3522_352263


namespace NUMINAMATH_CALUDE_intersection_nonempty_condition_l3522_352216

theorem intersection_nonempty_condition (m n : ℝ) :
  let A := {x : ℝ | m - 1 < x ∧ x < m + 1}
  let B := {x : ℝ | 3 - n < x ∧ x < 4 - n}
  (∃ x, x ∈ A ∩ B) ↔ 2 < m + n ∧ m + n < 5 :=
by sorry

end NUMINAMATH_CALUDE_intersection_nonempty_condition_l3522_352216


namespace NUMINAMATH_CALUDE_product_relation_l3522_352243

theorem product_relation (x y z : ℝ) (h : x^2 + y^2 = x*y*(z + 1/z)) :
  x = y*z ∨ y = x*z := by sorry

end NUMINAMATH_CALUDE_product_relation_l3522_352243


namespace NUMINAMATH_CALUDE_complex_calculation_l3522_352236

theorem complex_calculation : 
  (((3.242^2 * (16 + 8)) / (100 - (3 * 25))) + (32 - 10)^2) = 494.09014144 := by
  sorry

end NUMINAMATH_CALUDE_complex_calculation_l3522_352236


namespace NUMINAMATH_CALUDE_budget_allocation_l3522_352210

theorem budget_allocation (microphotonics : ℝ) (home_electronics : ℝ) (gm_microorganisms : ℝ) (industrial_lubricants : ℝ) (astrophysics_degrees : ℝ) :
  microphotonics = 12 ∧
  home_electronics = 24 ∧
  gm_microorganisms = 29 ∧
  industrial_lubricants = 8 ∧
  astrophysics_degrees = 43.2 →
  100 - (microphotonics + home_electronics + gm_microorganisms + industrial_lubricants + (astrophysics_degrees / 360 * 100)) = 15 := by
  sorry

end NUMINAMATH_CALUDE_budget_allocation_l3522_352210


namespace NUMINAMATH_CALUDE_elephant_distribution_l3522_352271

theorem elephant_distribution (union_members non_union_members : ℕ) 
  (h1 : union_members = 28)
  (h2 : non_union_members = 37) :
  let total_elephants := 2072
  let elephants_per_union := total_elephants / union_members
  let elephants_per_non_union := total_elephants / non_union_members
  (elephants_per_union * union_members = elephants_per_non_union * non_union_members) ∧
  (elephants_per_union ≥ 1) ∧
  (elephants_per_non_union ≥ 1) ∧
  (∀ n : ℕ, n > total_elephants → 
    ¬(n / union_members * union_members = n / non_union_members * non_union_members ∧
      n / union_members ≥ 1 ∧
      n / non_union_members ≥ 1)) :=
by sorry

end NUMINAMATH_CALUDE_elephant_distribution_l3522_352271


namespace NUMINAMATH_CALUDE_ravenswood_remaining_gnomes_l3522_352253

/-- The number of gnomes in Westerville woods -/
def westerville_gnomes : ℕ := 20

/-- The ratio of gnomes in Ravenswood forest compared to Westerville woods -/
def ravenswood_ratio : ℕ := 4

/-- The percentage of gnomes taken from Ravenswood forest -/
def taken_percentage : ℚ := 40 / 100

/-- The number of gnomes remaining in Ravenswood forest after some are taken -/
def remaining_gnomes : ℕ := 48

theorem ravenswood_remaining_gnomes :
  (ravenswood_ratio * westerville_gnomes : ℚ) * (1 - taken_percentage) = remaining_gnomes := by
  sorry

end NUMINAMATH_CALUDE_ravenswood_remaining_gnomes_l3522_352253


namespace NUMINAMATH_CALUDE_expression_value_l3522_352218

theorem expression_value (α : Real) (h : Real.tan α = -3/4) :
  (Real.cos (π/2 + α) * Real.sin (-π - α)) / 
  (Real.cos (11*π/12 - α) * Real.sin (9*π/2 + α)) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3522_352218


namespace NUMINAMATH_CALUDE_inequality_solutions_count_l3522_352212

theorem inequality_solutions_count : 
  ∃ (S : Finset ℤ), (∀ x : ℤ, x ∈ S ↔ 5*x^2 + 19*x + 12 ≤ 20) ∧ Finset.card S = 8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solutions_count_l3522_352212


namespace NUMINAMATH_CALUDE_parabola_latus_rectum_l3522_352241

/-- A parabola passing through a specific point has a specific latus rectum equation -/
theorem parabola_latus_rectum (p : ℝ) (h1 : p > 0) :
  (∀ x y, y^2 = 2*p*x → x = 1 ∧ y = 1/2) →
  (∃ x, x = -1/16 ∧ ∀ y, y^2 = 2*p*x) := by
  sorry

end NUMINAMATH_CALUDE_parabola_latus_rectum_l3522_352241


namespace NUMINAMATH_CALUDE_parallelogram_inscribed_circles_l3522_352215

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a circle -/
structure Circle :=
  (center : Point) (radius : ℝ)

/-- Represents a parallelogram ABCD -/
structure Parallelogram :=
  (A B C D : Point)

/-- Checks if a circle is inscribed in a triangle -/
def is_inscribed (c : Circle) (p1 p2 p3 : Point) : Prop := sorry

/-- Checks if a point lies on a line segment -/
def on_segment (p : Point) (p1 p2 : Point) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Main theorem -/
theorem parallelogram_inscribed_circles 
  (ABCD : Parallelogram) 
  (P : Point) 
  (c_ABC : Circle) 
  (c_DAP : Circle) 
  (c_DCP : Circle) :
  on_segment P ABCD.A ABCD.C →
  is_inscribed c_ABC ABCD.A ABCD.B ABCD.C →
  is_inscribed c_DAP ABCD.D ABCD.A P →
  is_inscribed c_DCP ABCD.D ABCD.C P →
  distance ABCD.D ABCD.A + distance ABCD.D ABCD.C = 3 * distance ABCD.A ABCD.C →
  distance ABCD.D ABCD.A = distance ABCD.D P →
  (distance ABCD.D ABCD.A + distance ABCD.A P = distance ABCD.D ABCD.C + distance ABCD.C P) ∧
  (c_DAP.radius / c_DCP.radius = distance ABCD.A P / distance P ABCD.C) ∧
  (c_DAP.radius / c_DCP.radius = 4 / 3) := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_inscribed_circles_l3522_352215


namespace NUMINAMATH_CALUDE_machine_working_time_l3522_352273

/-- The number of shirts made by the machine -/
def total_shirts : ℕ := 12

/-- The number of shirts the machine can make per minute -/
def shirts_per_minute : ℕ := 2

/-- The time the machine was working in minutes -/
def working_time : ℕ := total_shirts / shirts_per_minute

theorem machine_working_time : working_time = 6 := by sorry

end NUMINAMATH_CALUDE_machine_working_time_l3522_352273


namespace NUMINAMATH_CALUDE_equality_theorem_l3522_352294

theorem equality_theorem (a b c d e f : ℝ) 
  (h1 : a + b + c = d + e + f)
  (h2 : a^2 + b^2 + c^2 = d^2 + e^2 + f^2)
  (h3 : a^3 + b^3 + c^3 ≠ d^3 + e^3 + f^3) :
  (∀ k : ℝ, 
    (a + b + c + (d+k) + (e+k) + (f+k) = d + e + f + (a+k) + (b+k) + (c+k)) ∧
    (a^2 + b^2 + c^2 + (d+k)^2 + (e+k)^2 + (f+k)^2 = d^2 + e^2 + f^2 + (a+k)^2 + (b+k)^2 + (c+k)^2) ∧
    (a^3 + b^3 + c^3 + (d+k)^3 + (e+k)^3 + (f+k)^3 = d^3 + e^3 + f^3 + (a+k)^3 + (b+k)^3 + (c+k)^3)) ∧
  (∀ k : ℝ, k ≠ 0 → 
    a^4 + b^4 + c^4 + (d+k)^4 + (e+k)^4 + (f+k)^4 ≠ d^4 + e^4 + f^4 + (a+k)^4 + (b+k)^4 + (c+k)^4) :=
by sorry

end NUMINAMATH_CALUDE_equality_theorem_l3522_352294


namespace NUMINAMATH_CALUDE_original_equals_scientific_l3522_352252

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The number to be expressed in scientific notation -/
def original_number : ℕ := 1030000000

/-- The scientific notation representation of the original number -/
def scientific_form : ScientificNotation :=
  { coefficient := 1.03
  , exponent := 9
  , is_valid := by sorry }

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific :
  (original_number : ℝ) = scientific_form.coefficient * (10 : ℝ) ^ scientific_form.exponent :=
by sorry

end NUMINAMATH_CALUDE_original_equals_scientific_l3522_352252


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3522_352280

def A : Set ℝ := {-1, 0, 1}
def B : Set ℝ := {x : ℝ | x^2 + x ≤ 0}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3522_352280


namespace NUMINAMATH_CALUDE_basketball_players_count_l3522_352267

/-- The number of boys playing basketball in a group with given conditions -/
def boys_playing_basketball (total : ℕ) (football : ℕ) (neither : ℕ) (both : ℕ) : ℕ :=
  total - neither

theorem basketball_players_count :
  boys_playing_basketball 22 15 3 18 = 19 :=
by sorry

end NUMINAMATH_CALUDE_basketball_players_count_l3522_352267


namespace NUMINAMATH_CALUDE_cost_increase_percentage_l3522_352290

/-- Represents the initial ratio of costs for raw material, labor, and overheads -/
def initial_ratio : Fin 3 → ℚ
  | 0 => 4
  | 1 => 3
  | 2 => 2

/-- Represents the percentage changes in costs for raw material, labor, and overheads -/
def cost_changes : Fin 3 → ℚ
  | 0 => 110 / 100  -- 10% increase
  | 1 => 108 / 100  -- 8% increase
  | 2 => 95 / 100   -- 5% decrease

/-- Theorem stating that the overall percentage increase in cost is 6% -/
theorem cost_increase_percentage : 
  let initial_total := (Finset.sum Finset.univ initial_ratio)
  let new_total := (Finset.sum Finset.univ (λ i => initial_ratio i * cost_changes i))
  (new_total - initial_total) / initial_total * 100 = 6 := by
  sorry

end NUMINAMATH_CALUDE_cost_increase_percentage_l3522_352290


namespace NUMINAMATH_CALUDE_cut_pyramid_volume_ratio_l3522_352296

/-- Represents a pyramid cut by a plane parallel to its base -/
structure CutPyramid where
  lateralAreaRatio : ℚ  -- Ratio of lateral surface areas (small pyramid : frustum)
  volumeRatio : ℚ       -- Ratio of volumes (small pyramid : frustum)

/-- Theorem: If the lateral area ratio is 9:16, then the volume ratio is 27:98 -/
theorem cut_pyramid_volume_ratio (p : CutPyramid) 
  (h : p.lateralAreaRatio = 9 / 16) : p.volumeRatio = 27 / 98 := by
  sorry

end NUMINAMATH_CALUDE_cut_pyramid_volume_ratio_l3522_352296


namespace NUMINAMATH_CALUDE_number_equation_solution_l3522_352223

theorem number_equation_solution :
  ∃ x : ℝ, x + 5 * 12 / (180 / 3) = 51 ∧ x = 50 := by
sorry

end NUMINAMATH_CALUDE_number_equation_solution_l3522_352223


namespace NUMINAMATH_CALUDE_fixed_point_on_linear_function_l3522_352208

theorem fixed_point_on_linear_function (m : ℝ) (h : m ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ m * x - (3 * m + 2)
  f 3 = -2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_linear_function_l3522_352208


namespace NUMINAMATH_CALUDE_cookie_production_l3522_352219

def initial_cookies : ℕ := 24
def initial_flour : ℕ := 3
def efficiency_improvement : ℚ := 1/10
def new_flour : ℕ := 4

def improved_cookies : ℕ := 35

theorem cookie_production : 
  let initial_efficiency : ℚ := initial_cookies / initial_flour
  let improved_efficiency : ℚ := initial_efficiency * (1 + efficiency_improvement)
  let theoretical_cookies : ℚ := improved_efficiency * new_flour
  ⌊theoretical_cookies⌋ = improved_cookies := by sorry

end NUMINAMATH_CALUDE_cookie_production_l3522_352219


namespace NUMINAMATH_CALUDE_original_not_imply_converse_converse_implies_negation_l3522_352256

-- Define a proposition P and Q
variable (P Q : Prop)

-- Statement 1: The truth of an original statement does not necessarily imply the truth of its converse
theorem original_not_imply_converse : ∃ P Q, (P → Q) ∧ ¬(Q → P) := by sorry

-- Statement 2: If the converse of a statement is true, then its negation is also true
theorem converse_implies_negation : ∀ P Q, (Q → P) → (¬P → ¬Q) := by sorry

end NUMINAMATH_CALUDE_original_not_imply_converse_converse_implies_negation_l3522_352256


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l3522_352235

theorem line_segment_endpoint (x : ℝ) : 
  x > 0 → 
  (x - 2)^2 + 2^2 = 10^2 → 
  x = 2 + 4 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l3522_352235


namespace NUMINAMATH_CALUDE_no_additional_painters_needed_l3522_352255

/-- Represents the painting job scenario -/
structure PaintingJob where
  initialPainters : ℕ
  initialDays : ℚ
  initialRate : ℚ
  newDays : ℕ
  newRate : ℚ

/-- Calculates the total work required for the job -/
def totalWork (job : PaintingJob) : ℚ :=
  job.initialPainters * job.initialDays * job.initialRate

/-- Calculates the number of painters needed for the new conditions -/
def paintersNeeded (job : PaintingJob) : ℚ :=
  (totalWork job) / (job.newDays * job.newRate)

/-- Theorem stating that no additional painters are needed -/
theorem no_additional_painters_needed (job : PaintingJob) 
  (h1 : job.initialPainters = 6)
  (h2 : job.initialDays = 5/2)
  (h3 : job.initialRate = 2)
  (h4 : job.newDays = 2)
  (h5 : job.newRate = 5/2) :
  paintersNeeded job = job.initialPainters :=
by sorry

#check no_additional_painters_needed

end NUMINAMATH_CALUDE_no_additional_painters_needed_l3522_352255


namespace NUMINAMATH_CALUDE_remainder_of_product_product_remainder_l3522_352282

theorem remainder_of_product (a b c : ℕ) : (a * b * c) % 12 = ((a % 12) * (b % 12) * (c % 12)) % 12 := by sorry

theorem product_remainder : (1625 * 1627 * 1629) % 12 = 3 := by
  have h1 : 1625 % 12 = 5 := by sorry
  have h2 : 1627 % 12 = 7 := by sorry
  have h3 : 1629 % 12 = 9 := by sorry
  have h4 : (5 * 7 * 9) % 12 = 3 := by sorry
  exact calc
    (1625 * 1627 * 1629) % 12 = ((1625 % 12) * (1627 % 12) * (1629 % 12)) % 12 := by apply remainder_of_product
    _ = (5 * 7 * 9) % 12 := by rw [h1, h2, h3]
    _ = 3 := by exact h4

end NUMINAMATH_CALUDE_remainder_of_product_product_remainder_l3522_352282


namespace NUMINAMATH_CALUDE_num_ways_to_form_triangles_l3522_352227

/-- The number of distinguishable balls -/
def num_balls : ℕ := 6

/-- The number of distinguishable sticks -/
def num_sticks : ℕ := 6

/-- The number of balls required to form a triangle -/
def balls_per_triangle : ℕ := 3

/-- The number of sticks required to form a triangle -/
def sticks_per_triangle : ℕ := 3

/-- The number of triangles to be formed -/
def num_triangles : ℕ := 2

/-- The number of symmetries for each triangle (rotations and reflections) -/
def symmetries_per_triangle : ℕ := 6

/-- Theorem stating the number of ways to form two disjoint non-interlocking triangles -/
theorem num_ways_to_form_triangles : 
  (Nat.choose num_balls balls_per_triangle * Nat.factorial num_sticks) / 
  (Nat.factorial num_triangles * symmetries_per_triangle ^ num_triangles) = 200 :=
sorry

end NUMINAMATH_CALUDE_num_ways_to_form_triangles_l3522_352227


namespace NUMINAMATH_CALUDE_expression_simplification_l3522_352258

theorem expression_simplification (a b : ℤ) (h1 : a = -2) (h2 : b = 1) :
  a^3 * (-b^3)^2 + (-1/2 * a * b^2)^3 = -7 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3522_352258


namespace NUMINAMATH_CALUDE_duty_roster_arrangements_l3522_352202

def number_of_arrangements (n : ℕ) : ℕ := n.factorial

def adjacent_arrangements (n : ℕ) : ℕ := 2 * (n - 1).factorial

def double_adjacent_arrangements (n : ℕ) : ℕ := 2 * 2 * (n - 2).factorial

theorem duty_roster_arrangements :
  let total := number_of_arrangements 6
  let adjacent_ab := adjacent_arrangements 6
  let adjacent_cd := adjacent_arrangements 6
  let both_adjacent := double_adjacent_arrangements 6
  total - adjacent_ab - adjacent_cd + both_adjacent = 336 := by sorry

end NUMINAMATH_CALUDE_duty_roster_arrangements_l3522_352202


namespace NUMINAMATH_CALUDE_ellipse_and_line_properties_l3522_352200

-- Define the ellipse
structure Ellipse where
  center : ℝ × ℝ
  vertex : ℝ × ℝ
  focus : ℝ × ℝ
  b_point : ℝ × ℝ

-- Define the line
structure Line where
  slope : ℝ
  y_intercept : ℝ

-- Define the problem conditions
def ellipse_conditions (e : Ellipse) : Prop :=
  e.center = (0, 0) ∧
  e.vertex = (0, 2) ∧
  e.b_point = (Real.sqrt 2, Real.sqrt 2) ∧
  Real.sqrt ((e.focus.1 - Real.sqrt 2)^2 + (e.focus.2 - Real.sqrt 2)^2) = 2

-- Define the theorem
theorem ellipse_and_line_properties (e : Ellipse) (l : Line) :
  ellipse_conditions e →
  (∀ x y, y = l.slope * x + l.y_intercept → x^2 / 12 + y^2 / 4 = 1) →
  (0, -3) ∈ {(x, y) | y = l.slope * x + l.y_intercept} →
  (∃ m n : ℝ × ℝ, m ≠ n ∧
    m ∈ {(x, y) | x^2 / 12 + y^2 / 4 = 1} ∧
    n ∈ {(x, y) | x^2 / 12 + y^2 / 4 = 1} ∧
    m ∈ {(x, y) | y = l.slope * x + l.y_intercept} ∧
    n ∈ {(x, y) | y = l.slope * x + l.y_intercept} ∧
    (m.1 - 0)^2 + (m.2 - 2)^2 = (n.1 - 0)^2 + (n.2 - 2)^2) →
  (x^2 / 12 + y^2 / 4 = 1 ∧ (l.slope = Real.sqrt 6 / 3 ∨ l.slope = -Real.sqrt 6 / 3) ∧ l.y_intercept = -3) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_and_line_properties_l3522_352200


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3522_352283

theorem polynomial_simplification (x : ℝ) :
  (3 * x^3 + 4 * x^2 + 9 * x - 5) - (2 * x^3 + 2 * x^2 + 6 * x - 15) =
  x^3 + 2 * x^2 + 3 * x + 10 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3522_352283


namespace NUMINAMATH_CALUDE_product_equals_square_l3522_352220

theorem product_equals_square : 
  1000 * 2.998 * 2.998 * 100 = (29980 : ℝ)^2 := by sorry

end NUMINAMATH_CALUDE_product_equals_square_l3522_352220


namespace NUMINAMATH_CALUDE_repeating_decimal_division_l3522_352251

/-- Represents a repeating decimal with a whole number part and a repeating fractional part. -/
structure RepeatingDecimal where
  whole : ℕ
  repeating : ℕ

/-- Converts a RepeatingDecimal to a rational number. -/
def repeatingDecimalToRational (d : RepeatingDecimal) : ℚ :=
  d.whole + d.repeating / (999 : ℚ)

/-- The main theorem: proving that 0.714714... divided by 2.857857... equals 119/476 -/
theorem repeating_decimal_division :
  let x : RepeatingDecimal := ⟨0, 714⟩
  let y : RepeatingDecimal := ⟨2, 857⟩
  (repeatingDecimalToRational x) / (repeatingDecimalToRational y) = 119 / 476 := by
  sorry


end NUMINAMATH_CALUDE_repeating_decimal_division_l3522_352251


namespace NUMINAMATH_CALUDE_lewis_harvest_earnings_l3522_352295

/-- Calculates the total earnings during harvest season --/
def harvest_earnings (regular_weekly : ℕ) (overtime_weekly : ℕ) (weeks : ℕ) : ℕ :=
  (regular_weekly + overtime_weekly) * weeks

/-- Theorem stating Lewis's total earnings during harvest season --/
theorem lewis_harvest_earnings :
  harvest_earnings 28 939 1091 = 1055497 := by
  sorry

end NUMINAMATH_CALUDE_lewis_harvest_earnings_l3522_352295


namespace NUMINAMATH_CALUDE_playground_area_l3522_352259

/-- 
A rectangular playground has a perimeter of 100 meters and its length is twice its width. 
This theorem proves that the area of such a playground is 5000/9 square meters.
-/
theorem playground_area (width : ℝ) (length : ℝ) : 
  (2 * length + 2 * width = 100) →  -- Perimeter condition
  (length = 2 * width) →            -- Length-width relation
  (length * width = 5000 / 9) :=    -- Area calculation
by sorry

end NUMINAMATH_CALUDE_playground_area_l3522_352259


namespace NUMINAMATH_CALUDE_remainder_theorem_l3522_352234

-- Define the polynomial Q
variable (Q : ℝ → ℝ)

-- Define the conditions
axiom Q_at_20 : Q 20 = 100
axiom Q_at_100 : Q 100 = 20

-- Define the remainder function
def remainder (f : ℝ → ℝ) (x : ℝ) : ℝ := -x + 120

-- State the theorem
theorem remainder_theorem :
  ∃ (R : ℝ → ℝ), ∀ x, Q x = (x - 20) * (x - 100) * R x + remainder Q x :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3522_352234


namespace NUMINAMATH_CALUDE_cube_sum_problem_l3522_352298

theorem cube_sum_problem (x y z : ℝ) 
  (sum_eq : x + y + z = 2)
  (sum_prod_eq : x * y + y * z + z * x = -6)
  (prod_eq : x * y * z = -6) :
  x^3 + y^3 + z^3 = 25 := by sorry

end NUMINAMATH_CALUDE_cube_sum_problem_l3522_352298


namespace NUMINAMATH_CALUDE_cubic_root_sum_squares_l3522_352207

/-- Given that a, b, and c are the roots of x^3 - 3x - 2 = 0,
    prove that a(b+c)^2 + b(c+a)^2 + c(a+b)^2 = -6 -/
theorem cubic_root_sum_squares (a b c : ℝ) : 
  (∀ x : ℝ, x^3 - 3*x - 2 = 0 ↔ x = a ∨ x = b ∨ x = c) →
  a*(b+c)^2 + b*(c+a)^2 + c*(a+b)^2 = -6 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_squares_l3522_352207


namespace NUMINAMATH_CALUDE_probability_is_24_1107_l3522_352204

/-- Represents a 5x5x5 cube with one face painted red and an internal diagonal painted green -/
structure PaintedCube where
  size : Nat
  size_eq : size = 5

/-- The number of unit cubes with exactly three painted faces -/
def three_painted_faces (cube : PaintedCube) : Nat := 8

/-- The number of unit cubes with exactly one painted face -/
def one_painted_face (cube : PaintedCube) : Nat := 21

/-- The total number of unit cubes in the larger cube -/
def total_cubes (cube : PaintedCube) : Nat := cube.size ^ 3

/-- The probability of selecting one cube with exactly three painted faces
    and one cube with exactly one painted face when choosing two cubes uniformly at random -/
def probability (cube : PaintedCube) : Rat :=
  (three_painted_faces cube * one_painted_face cube : Rat) / (total_cubes cube).choose 2

/-- The main theorem stating the probability is 24/1107 -/
theorem probability_is_24_1107 (cube : PaintedCube) : probability cube = 24 / 1107 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_24_1107_l3522_352204


namespace NUMINAMATH_CALUDE_insurance_coverage_percentage_l3522_352254

def xray_cost : ℝ := 250
def mri_cost : ℝ := 3 * xray_cost
def total_cost : ℝ := xray_cost + mri_cost
def mike_payment : ℝ := 200
def insurance_coverage : ℝ := total_cost - mike_payment

theorem insurance_coverage_percentage : (insurance_coverage / total_cost) * 100 = 80 :=
by sorry

end NUMINAMATH_CALUDE_insurance_coverage_percentage_l3522_352254


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l3522_352233

/-- Given a principal sum and conditions on simple interest, prove the annual interest rate -/
theorem interest_rate_calculation (P : ℝ) (P_pos : P > 0) : 
  (P * (20 / 7) * 7) / 100 = P / 5 → (20 / 7 : ℝ) = 20 / 7 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l3522_352233


namespace NUMINAMATH_CALUDE_rectangle_diagonal_shorter_percentage_rectangle_diagonal_shorter_approx_25_percent_l3522_352249

/-- The percentage difference between the sum of two sides of a 2x1 rectangle
    and its diagonal, relative to the sum of the sides. -/
theorem rectangle_diagonal_shorter_percentage : ℝ :=
  let side_sum := 2 + 1
  let diagonal := Real.sqrt (2^2 + 1^2)
  (side_sum - diagonal) / side_sum * 100

/-- The percentage difference is approximately 25%. -/
theorem rectangle_diagonal_shorter_approx_25_percent :
  ∃ ε > 0, abs (rectangle_diagonal_shorter_percentage - 25) < ε :=
sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_shorter_percentage_rectangle_diagonal_shorter_approx_25_percent_l3522_352249


namespace NUMINAMATH_CALUDE_distribute_8_balls_4_boxes_l3522_352272

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem stating that there are 139 ways to distribute 8 indistinguishable balls into 4 distinguishable boxes -/
theorem distribute_8_balls_4_boxes : distribute_balls 8 4 = 139 := by sorry

end NUMINAMATH_CALUDE_distribute_8_balls_4_boxes_l3522_352272


namespace NUMINAMATH_CALUDE_duanes_initial_pages_l3522_352226

theorem duanes_initial_pages (lana_initial : ℕ) (lana_final : ℕ) (duane_initial : ℕ) : 
  lana_initial = 8 → 
  lana_final = 29 → 
  lana_final = lana_initial + duane_initial / 2 →
  duane_initial = 42 := by
sorry

end NUMINAMATH_CALUDE_duanes_initial_pages_l3522_352226


namespace NUMINAMATH_CALUDE_sequence_sum_property_l3522_352239

theorem sequence_sum_property (a : ℕ+ → ℚ) (S : ℕ+ → ℚ) :
  (∀ n : ℕ+, S n = 1 - n * a n) →
  (∀ n : ℕ+, a n = 1 / (n * (n + 1))) :=
by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_property_l3522_352239


namespace NUMINAMATH_CALUDE_gas_pressure_volume_relationship_l3522_352284

/-- Given inverse proportionality of pressure and volume at constant temperature,
    calculate the new pressure when the volume changes. -/
theorem gas_pressure_volume_relationship
  (initial_volume initial_pressure new_volume : ℝ)
  (h_positive : initial_volume > 0 ∧ initial_pressure > 0 ∧ new_volume > 0)
  (h_inverse_prop : ∀ (v p : ℝ), v > 0 → p > 0 → v * p = initial_volume * initial_pressure) :
  let new_pressure := (initial_volume * initial_pressure) / new_volume
  new_pressure = 2 ∧ initial_volume = 2.28 ∧ initial_pressure = 5 ∧ new_volume = 5.7 := by
sorry

end NUMINAMATH_CALUDE_gas_pressure_volume_relationship_l3522_352284
