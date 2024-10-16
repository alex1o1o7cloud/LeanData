import Mathlib

namespace NUMINAMATH_CALUDE_angle_between_given_lines_l2533_253396

def line1 (x y : ℝ) : Prop := 2 * x - y + 1 = 0
def line2 (x y : ℝ) : Prop := x - y - 2 = 0

def angle_between_lines (l1 l2 : (ℝ → ℝ → Prop)) : ℝ := sorry

theorem angle_between_given_lines :
  angle_between_lines line1 line2 = Real.arctan (1/3) := by sorry

end NUMINAMATH_CALUDE_angle_between_given_lines_l2533_253396


namespace NUMINAMATH_CALUDE_window_width_is_four_l2533_253317

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangular surface -/
def rectangleArea (d : Dimensions) : ℝ := d.length * d.width

/-- Calculates the perimeter of a rectangular surface -/
def rectanglePerimeter (d : Dimensions) : ℝ := 2 * (d.length + d.width)

/-- Represents the properties of the room and whitewashing job -/
structure RoomProperties where
  roomDimensions : Dimensions
  doorDimensions : Dimensions
  windowHeight : ℝ
  numWindows : ℕ
  costPerSquareFoot : ℝ
  totalCost : ℝ

/-- Theorem: The width of each window is 4 feet -/
theorem window_width_is_four (props : RoomProperties) 
  (h1 : props.roomDimensions = ⟨25, 15, 12⟩)
  (h2 : props.doorDimensions = ⟨6, 3, 0⟩)
  (h3 : props.windowHeight = 3)
  (h4 : props.numWindows = 3)
  (h5 : props.costPerSquareFoot = 8)
  (h6 : props.totalCost = 7248) : 
  ∃ w : ℝ, w = 4 ∧ 
    props.totalCost = props.costPerSquareFoot * 
      (rectanglePerimeter props.roomDimensions * props.roomDimensions.height - 
       rectangleArea props.doorDimensions - 
       props.numWindows * (w * props.windowHeight)) := by
  sorry

end NUMINAMATH_CALUDE_window_width_is_four_l2533_253317


namespace NUMINAMATH_CALUDE_tan_alpha_fourth_quadrant_l2533_253373

theorem tan_alpha_fourth_quadrant (α : Real) : 
  (π / 2 < α ∧ α < 2 * π) →  -- α is in the fourth quadrant
  (Real.cos (π / 2 + α) = 4 / 5) → 
  Real.tan α = -4 / 3 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_fourth_quadrant_l2533_253373


namespace NUMINAMATH_CALUDE_words_in_page_l2533_253329

/-- The number of words Tom can type per minute -/
def words_per_minute : ℕ := 90

/-- The number of minutes it takes Tom to type 10 pages -/
def minutes_for_ten_pages : ℕ := 50

/-- The number of pages Tom types in the given time -/
def number_of_pages : ℕ := 10

/-- Calculates the number of words in a page -/
def words_per_page : ℕ := (words_per_minute * minutes_for_ten_pages) / number_of_pages

/-- Theorem stating that there are 450 words in a page -/
theorem words_in_page : words_per_page = 450 := by
  sorry

end NUMINAMATH_CALUDE_words_in_page_l2533_253329


namespace NUMINAMATH_CALUDE_sum_a_plus_d_equals_six_l2533_253347

theorem sum_a_plus_d_equals_six (a b c d e : ℝ)
  (eq1 : a + b = 12)
  (eq2 : b + c = 9)
  (eq3 : c + d = 3)
  (eq4 : d + e = 7)
  (eq5 : e + a = 10) :
  a + d = 6 := by sorry

end NUMINAMATH_CALUDE_sum_a_plus_d_equals_six_l2533_253347


namespace NUMINAMATH_CALUDE_min_value_quadratic_expression_l2533_253364

theorem min_value_quadratic_expression :
  ∃ (min_val : ℝ), min_val = -7208 ∧
  ∀ (x y : ℝ), 2*x^2 + 3*x*y + 4*y^2 - 8*x - 10*y ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_expression_l2533_253364


namespace NUMINAMATH_CALUDE_sets_intersection_union_l2533_253341

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}
def B (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b ≤ 0}

-- State the theorem
theorem sets_intersection_union (a b : ℝ) : 
  (A ∪ B a b = Set.univ) ∧ (A ∩ B a b = {x | 3 < x ∧ x ≤ 4}) → a + b = -7 := by
  sorry

end NUMINAMATH_CALUDE_sets_intersection_union_l2533_253341


namespace NUMINAMATH_CALUDE_min_value_of_f_l2533_253345

/-- The function f(x) = 2x^3 - 6x^2 + a -/
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + a

theorem min_value_of_f (a : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f a x = 3) ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f a x ≤ 3) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f a x = -37) ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f a x ≥ -37) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2533_253345


namespace NUMINAMATH_CALUDE_factory_growth_rate_l2533_253331

theorem factory_growth_rate (P a b x : ℝ) (h1 : P > 0) (h2 : a > -1) (h3 : b > -1)
  (h4 : (1 + x)^2 = (1 + a) * (1 + b)) : x ≤ max a b := by
  sorry

end NUMINAMATH_CALUDE_factory_growth_rate_l2533_253331


namespace NUMINAMATH_CALUDE_april_rainfall_calculation_l2533_253377

/-- Given the March rainfall and the difference between March and April rainfall,
    calculate the April rainfall. -/
def april_rainfall (march_rainfall : ℝ) (rainfall_difference : ℝ) : ℝ :=
  march_rainfall - rainfall_difference

/-- Theorem stating that given the specific March rainfall and difference,
    the April rainfall is 0.46 inches. -/
theorem april_rainfall_calculation :
  april_rainfall 0.81 0.35 = 0.46 := by
  sorry

end NUMINAMATH_CALUDE_april_rainfall_calculation_l2533_253377


namespace NUMINAMATH_CALUDE_inverse_of_10_mod_997_l2533_253383

theorem inverse_of_10_mod_997 : 
  ∃ x : ℕ, x < 997 ∧ (10 * x) % 997 = 1 :=
by
  use 709
  sorry

end NUMINAMATH_CALUDE_inverse_of_10_mod_997_l2533_253383


namespace NUMINAMATH_CALUDE_mary_earnings_l2533_253307

def cleaning_rate : ℕ := 46
def babysitting_rate : ℕ := 35
def pet_care_rate : ℕ := 60

def homes_cleaned : ℕ := 4
def days_babysat : ℕ := 5
def days_pet_care : ℕ := 3

def total_earnings : ℕ := 
  cleaning_rate * homes_cleaned + 
  babysitting_rate * days_babysat + 
  pet_care_rate * days_pet_care

theorem mary_earnings : total_earnings = 539 := by
  sorry

end NUMINAMATH_CALUDE_mary_earnings_l2533_253307


namespace NUMINAMATH_CALUDE_distance_between_points_l2533_253342

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (1, 2)
  let p2 : ℝ × ℝ := (-3, -4)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l2533_253342


namespace NUMINAMATH_CALUDE_only_cylinder_has_rectangular_front_view_l2533_253310

-- Define the solid figures
inductive SolidFigure
  | Cylinder
  | TriangularPyramid
  | Sphere
  | Cone

-- Define the property of having a rectangular front view
def hasRectangularFrontView (figure : SolidFigure) : Prop :=
  match figure with
  | SolidFigure.Cylinder => True
  | _ => False

-- Theorem statement
theorem only_cylinder_has_rectangular_front_view :
  ∀ (figure : SolidFigure), hasRectangularFrontView figure ↔ figure = SolidFigure.Cylinder :=
by sorry

end NUMINAMATH_CALUDE_only_cylinder_has_rectangular_front_view_l2533_253310


namespace NUMINAMATH_CALUDE_sheep_count_l2533_253380

/-- The number of sheep in the meadow -/
def num_sheep : ℕ := 36

/-- The number of cows in the meadow -/
def num_cows : ℕ := 12

/-- The number of ears per cow -/
def ears_per_cow : ℕ := 2

/-- The number of legs per cow -/
def legs_per_cow : ℕ := 4

/-- Theorem stating that the number of sheep is 36 given the conditions -/
theorem sheep_count :
  num_sheep > num_cows * ears_per_cow ∧
  num_sheep < num_cows * legs_per_cow ∧
  num_sheep % 12 = 0 →
  num_sheep = 36 :=
by sorry

end NUMINAMATH_CALUDE_sheep_count_l2533_253380


namespace NUMINAMATH_CALUDE_quadratic_one_solution_m_quadratic_one_solution_positive_m_l2533_253390

/-- The positive value of m for which the quadratic equation 9x^2 + mx + 36 = 0 has exactly one solution -/
theorem quadratic_one_solution_m (m : ℝ) : 
  (∃! x, 9 * x^2 + m * x + 36 = 0) → m = 36 ∨ m = -36 :=
by sorry

/-- The positive value of m for which the quadratic equation 9x^2 + mx + 36 = 0 has exactly one solution is 36 -/
theorem quadratic_one_solution_positive_m :
  ∃ m : ℝ, m > 0 ∧ (∃! x, 9 * x^2 + m * x + 36 = 0) ∧ m = 36 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_m_quadratic_one_solution_positive_m_l2533_253390


namespace NUMINAMATH_CALUDE_callum_max_score_l2533_253324

/-- Calculates the score for n consecutive wins with a base score and multiplier -/
def consecutiveWinScore (baseScore : ℕ) (n : ℕ) : ℕ :=
  baseScore * 2^(n - 1)

/-- Calculates the total score for a given number of wins -/
def totalScore (wins : ℕ) : ℕ :=
  (List.range wins).map (consecutiveWinScore 10) |> List.sum

theorem callum_max_score (totalMatches : ℕ) (krishnaWins : ℕ) 
    (h1 : totalMatches = 12)
    (h2 : krishnaWins = 2 * totalMatches / 3)
    (h3 : krishnaWins < totalMatches) : 
  totalScore (totalMatches - krishnaWins) = 150 := by
  sorry

end NUMINAMATH_CALUDE_callum_max_score_l2533_253324


namespace NUMINAMATH_CALUDE_polynomial_remainder_l2533_253332

theorem polynomial_remainder (q : ℝ → ℝ) (h1 : ∃ r1 : ℝ → ℝ, ∀ x, q x = (x - 1) * (r1 x) + 10)
  (h2 : ∃ r2 : ℝ → ℝ, ∀ x, q x = (x + 3) * (r2 x) - 8) :
  ∃ r : ℝ → ℝ, ∀ x, q x = (x - 1) * (x + 3) * (r x) + 4.5 * x + 5.5 := by
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l2533_253332


namespace NUMINAMATH_CALUDE_train_overtake_l2533_253399

/-- The speed of Train A in miles per hour -/
def speed_a : ℝ := 30

/-- The speed of Train B in miles per hour -/
def speed_b : ℝ := 38

/-- The time difference between Train A and Train B's departure in hours -/
def time_diff : ℝ := 2

/-- The distance at which Train B overtakes Train A -/
def overtake_distance : ℝ := 285

theorem train_overtake :
  ∃ t : ℝ, t > 0 ∧ speed_b * t = speed_a * (t + time_diff) ∧ 
  overtake_distance = speed_b * t :=
sorry

end NUMINAMATH_CALUDE_train_overtake_l2533_253399


namespace NUMINAMATH_CALUDE_least_possible_bc_length_l2533_253340

theorem least_possible_bc_length 
  (AB AC DC BD : ℝ) 
  (hAB : AB = 8) 
  (hAC : AC = 10) 
  (hDC : DC = 7) 
  (hBD : BD = 15) : 
  ∃ (BC : ℕ), BC = 9 ∧ 
    BC > AC - AB ∧ 
    BC > BD - DC ∧ 
    ∀ (n : ℕ), n < 9 → (n ≤ AC - AB ∨ n ≤ BD - DC) :=
by sorry

end NUMINAMATH_CALUDE_least_possible_bc_length_l2533_253340


namespace NUMINAMATH_CALUDE_store_earnings_calculation_l2533_253325

/-- Represents the earnings from a day's sale of drinks at a country store. -/
def store_earnings (cola_price : ℚ) (juice_price : ℚ) (water_price : ℚ) (sports_drink_price : ℚ)
                   (cola_sold : ℕ) (juice_sold : ℕ) (water_sold : ℕ) (sports_drink_sold : ℕ)
                   (sports_drink_paid : ℕ) : ℚ :=
  cola_price * cola_sold + juice_price * juice_sold + water_price * water_sold +
  sports_drink_price * sports_drink_paid

/-- Theorem stating the total earnings of the store given the specific conditions. -/
theorem store_earnings_calculation :
  let cola_price : ℚ := 3
  let juice_price : ℚ := 3/2
  let water_price : ℚ := 1
  let sports_drink_price : ℚ := 5/2
  let cola_sold : ℕ := 18
  let juice_sold : ℕ := 15
  let water_sold : ℕ := 30
  let sports_drink_sold : ℕ := 44
  let sports_drink_paid : ℕ := 22
  store_earnings cola_price juice_price water_price sports_drink_price
                 cola_sold juice_sold water_sold sports_drink_sold sports_drink_paid = 161.5 := by
  sorry


end NUMINAMATH_CALUDE_store_earnings_calculation_l2533_253325


namespace NUMINAMATH_CALUDE_composition_of_transformations_l2533_253343

-- Define the transformations
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)
def g (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

-- State the theorem
theorem composition_of_transformations :
  g (f (-6, 7)) = (-7, 6) := by sorry

end NUMINAMATH_CALUDE_composition_of_transformations_l2533_253343


namespace NUMINAMATH_CALUDE_lcm_is_perfect_square_l2533_253361

theorem lcm_is_perfect_square (a b : ℕ) (h : (a^3 + b^3 + a*b) % (a*b*(a - b)) = 0) :
  Nat.lcm a b = (Nat.gcd a b)^2 := by
  sorry

end NUMINAMATH_CALUDE_lcm_is_perfect_square_l2533_253361


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2533_253326

/-- For an arithmetic sequence with first term a₁ and common difference d,
    if the sum of the first 10 terms is 4 times the sum of the first 5 terms,
    then a₁/d = 1/2. -/
theorem arithmetic_sequence_ratio (a₁ d : ℝ) :
  let S : ℕ → ℝ := λ n => n * a₁ + (n * (n - 1) / 2) * d
  S 10 = 4 * S 5 → a₁ / d = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2533_253326


namespace NUMINAMATH_CALUDE_product_odd_implies_sum_even_l2533_253393

theorem product_odd_implies_sum_even (a b : ℤ) : 
  Odd (a * b) → Even (a + b) := by
  sorry

end NUMINAMATH_CALUDE_product_odd_implies_sum_even_l2533_253393


namespace NUMINAMATH_CALUDE_inheritance_tax_equation_l2533_253378

/-- The inheritance amount in dollars -/
def inheritance : ℝ := 35300

/-- The federal tax rate as a decimal -/
def federal_tax_rate : ℝ := 0.25

/-- The state tax rate as a decimal -/
def state_tax_rate : ℝ := 0.12

/-- The total tax paid in dollars -/
def total_tax_paid : ℝ := 12000

theorem inheritance_tax_equation :
  federal_tax_rate * inheritance + 
  state_tax_rate * (inheritance - federal_tax_rate * inheritance) = 
  total_tax_paid := by sorry

end NUMINAMATH_CALUDE_inheritance_tax_equation_l2533_253378


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l2533_253306

/-- Given a geometric sequence with first term 512 and 8th term 2, the 6th term is 16 -/
theorem geometric_sequence_sixth_term : 
  ∀ (a : ℕ → ℝ), 
  (∀ n, a (n + 1) = a n * (a 8 / a 7)) →  -- Geometric sequence property
  a 1 = 512 →                            -- First term is 512
  a 8 = 2 →                              -- 8th term is 2
  a 6 = 16 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l2533_253306


namespace NUMINAMATH_CALUDE_solution_to_system_l2533_253338

/-- The system of equations -/
def equation1 (x y : ℝ) : Prop := x^2*y - x*y^2 - 5*x + 5*y + 3 = 0
def equation2 (x y : ℝ) : Prop := x^3*y - x*y^3 - 5*x^2 + 5*y^2 + 15 = 0

/-- The theorem stating that (4, 1) is the solution to the system of equations -/
theorem solution_to_system : equation1 4 1 ∧ equation2 4 1 := by sorry

end NUMINAMATH_CALUDE_solution_to_system_l2533_253338


namespace NUMINAMATH_CALUDE_sandy_shopping_l2533_253314

def shopping_equation (X Y : ℝ) : Prop :=
  let pie_cost : ℝ := 6
  let sandwich_cost : ℝ := 3
  let book_cost : ℝ := 10
  let book_discount : ℝ := 0.2
  let sales_tax : ℝ := 0.05
  let discounted_book_cost : ℝ := book_cost * (1 - book_discount)
  let subtotal : ℝ := pie_cost + sandwich_cost + discounted_book_cost
  let total_cost : ℝ := subtotal * (1 + sales_tax)
  Y = X - total_cost

theorem sandy_shopping : 
  ∀ X Y : ℝ, shopping_equation X Y ↔ Y = X - 17.85 := by sorry

end NUMINAMATH_CALUDE_sandy_shopping_l2533_253314


namespace NUMINAMATH_CALUDE_third_fraction_numerator_l2533_253372

/-- Given three fractions where the sum is 3.0035428163476343,
    the first fraction is 2007/2999, the second is 8001/5998,
    and the third has a denominator of 3999,
    prove that the numerator of the third fraction is 4002. -/
theorem third_fraction_numerator :
  let sum : ℚ := 3.0035428163476343
  let frac1 : ℚ := 2007 / 2999
  let frac2 : ℚ := 8001 / 5998
  let denom3 : ℕ := 3999
  ∃ (num3 : ℕ), (frac1 + frac2 + (num3 : ℚ) / denom3 = sum) ∧ num3 = 4002 := by
  sorry


end NUMINAMATH_CALUDE_third_fraction_numerator_l2533_253372


namespace NUMINAMATH_CALUDE_halfway_point_l2533_253318

theorem halfway_point (a b : ℚ) (ha : a = 1/8) (hb : b = 1/10) :
  (a + b) / 2 = 9/80 := by
  sorry

end NUMINAMATH_CALUDE_halfway_point_l2533_253318


namespace NUMINAMATH_CALUDE_product_equality_l2533_253335

theorem product_equality (a b c d e f : ℝ) 
  (sum_zero : a + b + c + d + e + f = 0)
  (sum_cubes_zero : a^3 + b^3 + c^3 + d^3 + e^3 + f^3 = 0) :
  (a+c)*(a+d)*(a+e)*(a+f) = (b+c)*(b+d)*(b+e)*(b+f) := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l2533_253335


namespace NUMINAMATH_CALUDE_equation_solution_l2533_253398

theorem equation_solution : ∃ x : ℝ, 
  x * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 2800.0000000000005 ∧ 
  abs (x - 1.4) < 0.00000000000001 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2533_253398


namespace NUMINAMATH_CALUDE_derivative_of_exp_sin_l2533_253309

theorem derivative_of_exp_sin (x : ℝ) : 
  deriv (fun x => Real.exp x * Real.sin x) x = Real.exp x * (Real.sin x + Real.cos x) := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_exp_sin_l2533_253309


namespace NUMINAMATH_CALUDE_tangent_circle_radius_l2533_253321

/-- Right isosceles triangle with legs of length 2 -/
structure RightIsoscelesTriangle where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  is_right_angle : (Q.1 - P.1) * (R.1 - Q.1) + (Q.2 - P.2) * (R.2 - Q.2) = 0
  is_isosceles : (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = (R.1 - Q.1)^2 + (R.2 - Q.2)^2
  leg_length : (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 4
  Q_at_origin : Q = (0, 0)
  PQ_on_x_axis : P.2 = 0 ∧ P.1 > 0
  QR_on_y_axis : R.1 = 0 ∧ R.2 > 0

/-- Circle tangent to the hypotenuse and coordinate axes -/
structure TangentCircle where
  S : ℝ × ℝ
  radius : ℝ
  tangent_to_hypotenuse : (S.1 - 2)^2 + (S.2 - 2)^2 = radius^2
  tangent_to_x_axis : S.2 = radius
  tangent_to_y_axis : S.1 = radius

/-- Theorem: The radius of the tangent circle is 4 -/
theorem tangent_circle_radius (t : RightIsoscelesTriangle) (c : TangentCircle) : c.radius = 4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circle_radius_l2533_253321


namespace NUMINAMATH_CALUDE_fraction_B_is_02_l2533_253351

-- Define the fractions of students receiving grades
def fraction_A : ℝ := 0.7
def fraction_A_or_B : ℝ := 0.9

-- Theorem statement
theorem fraction_B_is_02 : 
  fraction_A_or_B - fraction_A = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_B_is_02_l2533_253351


namespace NUMINAMATH_CALUDE_largest_angle_in_triangle_l2533_253360

theorem largest_angle_in_triangle (x y z : ℝ) : 
  x = 60 → y = 70 → x + y + z = 180 → 
  ∃ max_angle : ℝ, max_angle = 70 ∧ max_angle ≥ x ∧ max_angle ≥ y ∧ max_angle ≥ z :=
by sorry

end NUMINAMATH_CALUDE_largest_angle_in_triangle_l2533_253360


namespace NUMINAMATH_CALUDE_car_sales_third_day_l2533_253319

theorem car_sales_third_day 
  (total_sales : ℕ) 
  (first_day : ℕ) 
  (second_day : ℕ) 
  (h1 : total_sales = 57) 
  (h2 : first_day = 14) 
  (h3 : second_day = 16) : 
  total_sales - (first_day + second_day) = 27 := by
sorry

end NUMINAMATH_CALUDE_car_sales_third_day_l2533_253319


namespace NUMINAMATH_CALUDE_every_algorithm_has_sequential_structure_l2533_253382

/-- An algorithm is a sequence of well-defined instructions for solving a problem or performing a task. -/
def Algorithm : Type := Unit

/-- A sequential structure is a series of steps executed in a specific order. -/
def SequentialStructure : Type := Unit

/-- Every algorithm has a sequential structure. -/
theorem every_algorithm_has_sequential_structure :
  ∀ (a : Algorithm), ∃ (s : SequentialStructure), True :=
sorry

end NUMINAMATH_CALUDE_every_algorithm_has_sequential_structure_l2533_253382


namespace NUMINAMATH_CALUDE_square_difference_of_integers_l2533_253304

theorem square_difference_of_integers (a b : ℕ) (h1 : a + b = 70) (h2 : a - b = 20) :
  a^2 - b^2 = 1400 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_of_integers_l2533_253304


namespace NUMINAMATH_CALUDE_determinant_special_matrix_l2533_253311

/-- The determinant of the matrix [[1, x, z], [1, x+z, z], [1, x, x+z]] is equal to xz - z^2 -/
theorem determinant_special_matrix (x z : ℝ) :
  Matrix.det !![1, x, z; 1, x + z, z; 1, x, x + z] = x * z - z^2 := by
  sorry

end NUMINAMATH_CALUDE_determinant_special_matrix_l2533_253311


namespace NUMINAMATH_CALUDE_hyperbola_line_slope_l2533_253388

/-- Given a hyperbola and a line intersecting it, prove that the slope of the line is 6 -/
theorem hyperbola_line_slope :
  ∀ (A B : ℝ × ℝ),
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let P := (2, 1)
  -- Hyperbola equation
  (x₁^2 - y₁^2/3 = 1) →
  (x₂^2 - y₂^2/3 = 1) →
  -- P is the midpoint of AB
  (2 = (x₁ + x₂)/2) →
  (1 = (y₁ + y₂)/2) →
  -- Slope of AB
  ((y₁ - y₂)/(x₁ - x₂) = 6) :=
by
  sorry


end NUMINAMATH_CALUDE_hyperbola_line_slope_l2533_253388


namespace NUMINAMATH_CALUDE_sphere_surface_area_ratio_l2533_253328

theorem sphere_surface_area_ratio (r₁ r₂ : ℝ) (h : (4 / 3 * Real.pi * r₁^3) / (4 / 3 * Real.pi * r₂^3) = 1 / 27) :
  (4 * Real.pi * r₁^2) / (4 * Real.pi * r₂^2) = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_ratio_l2533_253328


namespace NUMINAMATH_CALUDE_stack_probability_exact_l2533_253379

/-- Represents the dimensions of a crate -/
structure CrateDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- Calculates n! (n factorial) -/
def factorial (n : ℕ) : ℕ := sorry

/-- The number of crates in the stack -/
def numCrates : ℕ := 15

/-- The dimensions of each crate -/
def crateDim : CrateDimensions := ⟨2, 5, 7⟩

/-- The target height of the stack -/
def targetHeight : ℕ := 60

/-- The total number of possible orientations for the stack -/
def totalOrientations : ℕ := 3^numCrates

/-- The number of valid orientations that result in the target height -/
def validOrientations : ℕ := 
  choose numCrates 5 * choose 10 10 +
  choose numCrates 7 * choose 8 5 * choose 3 3 +
  choose numCrates 9 * choose 6 6

/-- The probability of the stack being exactly 60ft tall -/
def stackProbability : ℚ := validOrientations / totalOrientations

theorem stack_probability_exact : 
  stackProbability = 158158 / 14348907 := by sorry

end NUMINAMATH_CALUDE_stack_probability_exact_l2533_253379


namespace NUMINAMATH_CALUDE_difference_of_squares_l2533_253303

theorem difference_of_squares (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2533_253303


namespace NUMINAMATH_CALUDE_rectangle_length_proof_l2533_253374

/-- Proves that a rectangle with length double its width, when modified as described, has an original length of 40. -/
theorem rectangle_length_proof (w : ℝ) (h1 : w > 0) : 
  (2*w - 5) * (w + 5) = 2*w*w + 75 → 2*w = 40 := by
  sorry

#check rectangle_length_proof

end NUMINAMATH_CALUDE_rectangle_length_proof_l2533_253374


namespace NUMINAMATH_CALUDE_carolyn_sum_is_18_l2533_253352

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def carolyn_game (n : ℕ) (first_remove : ℕ) : ℕ → Prop
| 0 => true
| (k+1) => ∃ (removed : ℕ), removed ≤ n ∧
           (k = 0 → ¬is_prime removed) ∧
           (k = 0 → removed = first_remove) ∧
           carolyn_game n first_remove k

theorem carolyn_sum_is_18 (n : ℕ) (h1 : n = 10) (first_remove : ℕ) (h2 : first_remove = 4) :
  ∃ k, carolyn_game n first_remove k ∧
  (∃ (removed : List ℕ), 
    (∀ x ∈ removed, x ≤ n) ∧
    (List.length removed = k) ∧
    (List.sum removed = 18)) :=
sorry

end NUMINAMATH_CALUDE_carolyn_sum_is_18_l2533_253352


namespace NUMINAMATH_CALUDE_min_value_reciprocal_product_l2533_253392

theorem min_value_reciprocal_product (a b : ℝ) : 
  a > 0 → b > 0 → 2*a + b = 4 → (∀ x y : ℝ, x > 0 → y > 0 → 2*x + y = 4 → 1/(a*b) ≤ 1/(x*y)) ∧ (∃ a₀ b₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ 2*a₀ + b₀ = 4 ∧ 1/(a₀*b₀) = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_product_l2533_253392


namespace NUMINAMATH_CALUDE_reciprocal_of_sum_diff_fractions_l2533_253320

theorem reciprocal_of_sum_diff_fractions : 
  (1 / (1/3 + 1/4 - 1/12) : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_sum_diff_fractions_l2533_253320


namespace NUMINAMATH_CALUDE_instantaneous_velocity_one_l2533_253301

-- Define the motion equation
def s (t : ℝ) : ℝ := 3 * t^2 - 2

-- Define the instantaneous velocity (derivative of s with respect to t)
def v (t : ℝ) : ℝ := 6 * t

-- Theorem: The time at which the instantaneous velocity is 1 is 1/6
theorem instantaneous_velocity_one (t : ℝ) : v t = 1 ↔ t = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_one_l2533_253301


namespace NUMINAMATH_CALUDE_continuity_at_4_l2533_253362

def f (x : ℝ) : ℝ := 2 * x^2 - 3

theorem continuity_at_4 :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 4| < δ → |f x - f 4| < ε :=
by sorry

end NUMINAMATH_CALUDE_continuity_at_4_l2533_253362


namespace NUMINAMATH_CALUDE_x_square_plus_inverse_value_l2533_253302

theorem x_square_plus_inverse_value (x : ℝ) (h : 49 = x^6 + 1 / x^6) :
  x^2 + 1 / x^2 = (51 : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_x_square_plus_inverse_value_l2533_253302


namespace NUMINAMATH_CALUDE_z_equals_four_when_x_is_five_l2533_253391

/-- The inverse relationship between 7z and x² -/
def inverse_relation (z x : ℝ) : Prop := ∃ k : ℝ, 7 * z = k / (x ^ 2)

/-- The theorem stating that given the inverse relationship and initial condition, z = 4 when x = 5 -/
theorem z_equals_four_when_x_is_five :
  ∀ z₀ : ℝ, inverse_relation z₀ 2 ∧ z₀ = 25 →
  ∃ z : ℝ, inverse_relation z 5 ∧ z = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_z_equals_four_when_x_is_five_l2533_253391


namespace NUMINAMATH_CALUDE_max_ratio_concentric_circles_polyline_l2533_253305

/-- The maximum common ratio for concentric circles allowing a closed polyline -/
theorem max_ratio_concentric_circles_polyline :
  ∃ (q : ℝ), q = (Real.sqrt 5 + 1) / 2 ∧
  ∀ (r : ℝ) (A : Fin 5 → ℝ × ℝ),
  (∀ i : Fin 5, ‖A i‖ = r * q ^ i.val) →
  (∀ i : Fin 5, ‖A i - A (i + 1)‖ = ‖A 0 - A 1‖) →
  (A 0 = A 4) →
  ∀ q' > q, ¬∃ (A' : Fin 5 → ℝ × ℝ),
    (∀ i : Fin 5, ‖A' i‖ = r * q' ^ i.val) ∧
    (∀ i : Fin 5, ‖A' i - A' (i + 1)‖ = ‖A' 0 - A' 1‖) ∧
    (A' 0 = A' 4) :=
by sorry

end NUMINAMATH_CALUDE_max_ratio_concentric_circles_polyline_l2533_253305


namespace NUMINAMATH_CALUDE_cube_equation_solution_l2533_253313

theorem cube_equation_solution (a d : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * 45 * d) : d = 49 := by
  sorry

end NUMINAMATH_CALUDE_cube_equation_solution_l2533_253313


namespace NUMINAMATH_CALUDE_trajectory_and_symmetry_l2533_253368

-- Define the fixed circle F
def F (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

-- Define the line L that the moving circle is tangent to
def L (x : ℝ) : Prop := x = -1

-- Define the trajectory C of the center P
def C (x y : ℝ) : Prop := y^2 = 8*x

-- Define symmetry about the line y = x - 1
def symmetric_about_line (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (x₁ + x₂)/2 - ((y₁ + y₂)/2 + 1) = 0 ∧ y₁ + y₂ = x₁ + x₂ - 2

theorem trajectory_and_symmetry :
  (∀ x y, C x y ↔ ∃ r, (∀ xf yf, F xf yf → (x - xf)^2 + (y - yf)^2 = (r + 1)^2) ∧
                       (∀ xl, L xl → |x - xl| = r)) ∧
  ¬(∃ x₁ y₁ x₂ y₂, C x₁ y₁ ∧ C x₂ y₂ ∧ symmetric_about_line x₁ y₁ x₂ y₂) :=
sorry

end NUMINAMATH_CALUDE_trajectory_and_symmetry_l2533_253368


namespace NUMINAMATH_CALUDE_principal_calculation_l2533_253339

/-- Proves that the principal amount is 1500 given the specified conditions --/
theorem principal_calculation (rate : ℝ) (time : ℝ) (amount : ℝ) :
  rate = 0.05 →
  time = 2.4 →
  amount = 1680 →
  (1 + rate * time) * 1500 = amount :=
by sorry

end NUMINAMATH_CALUDE_principal_calculation_l2533_253339


namespace NUMINAMATH_CALUDE_marks_car_repair_cost_l2533_253350

/-- The total cost of fixing Mark's car -/
def total_cost (part_cost : ℕ) (num_parts : ℕ) (labor_rate : ℚ) (hours_worked : ℕ) : ℚ :=
  (part_cost * num_parts : ℚ) + labor_rate * (hours_worked * 60)

/-- Theorem stating that the total cost of fixing Mark's car is $220 -/
theorem marks_car_repair_cost :
  total_cost 20 2 0.5 6 = 220 := by
  sorry

end NUMINAMATH_CALUDE_marks_car_repair_cost_l2533_253350


namespace NUMINAMATH_CALUDE_expression_evaluation_l2533_253359

theorem expression_evaluation (x y : ℝ) (h : x > y ∧ y > 0) :
  (x^(2*y) * y^x) / (y^(2*x) * x^y) = (x/y)^(y-x) := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2533_253359


namespace NUMINAMATH_CALUDE_right_triangle_consecutive_sides_l2533_253395

theorem right_triangle_consecutive_sides (a c : ℕ) (h1 : c = a + 1) :
  ∃ b : ℕ, b * b = c + a ∧ c * c = a * a + b * b := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_consecutive_sides_l2533_253395


namespace NUMINAMATH_CALUDE_inequality_solution_interval_l2533_253349

theorem inequality_solution_interval (a : ℝ) : 
  (∀ x, Real.sqrt (x + a) ≥ x) ∧ 
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ 
    Real.sqrt (x₁ + a) = x₁ ∧ 
    Real.sqrt (x₂ + a) = x₂ ∧ 
    |x₁ - x₂| = 4 * |a|) →
  a = 4/9 ∨ a = (1 - Real.sqrt 5) / 8 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_interval_l2533_253349


namespace NUMINAMATH_CALUDE_complex_square_value_l2533_253348

theorem complex_square_value : ((1 - Complex.I * Real.sqrt 3) / Complex.I) ^ 2 = 2 + Complex.I * (2 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_complex_square_value_l2533_253348


namespace NUMINAMATH_CALUDE_g_lower_bound_l2533_253394

theorem g_lower_bound (x m : ℝ) (hx : x > 0) (hm : 0 < m) (hm1 : m < 1) :
  Real.exp (m * x - 1) - (Real.log x + 1) / m > m^(1/m) - m^(-1/m) := by
  sorry

end NUMINAMATH_CALUDE_g_lower_bound_l2533_253394


namespace NUMINAMATH_CALUDE_tangent_slope_at_point_A_l2533_253363

-- Define the function
def f (x : ℝ) : ℝ := x^2 + 3*x

-- Define the point A
def point_A : ℝ × ℝ := (2, 10)

-- Theorem statement
theorem tangent_slope_at_point_A :
  (deriv f) point_A.1 = 7 :=
sorry

end NUMINAMATH_CALUDE_tangent_slope_at_point_A_l2533_253363


namespace NUMINAMATH_CALUDE_smallest_whole_multiple_651_l2533_253300

def is_whole_multiple (n m : ℕ) : Prop := n % m = 0

def digit_sum (n : ℕ) : ℕ := 
  (n / 100) + ((n / 10) % 10) + (n % 10)

def P (n : ℕ) : ℕ := max ((n / 100) * 10 + (n / 10) % 10) (max ((n / 100) * 10 + n % 10) ((n / 10) % 10 * 10 + n % 10))

def Q (n : ℕ) : ℕ := min ((n / 100) * 10 + (n / 10) % 10) (min ((n / 100) * 10 + n % 10) ((n / 10) % 10 * 10 + n % 10))

theorem smallest_whole_multiple_651 : 
  ∃ (A : ℕ), 
    100 ≤ A ∧ A < 1000 ∧ 
    is_whole_multiple A 12 ∧
    digit_sum A = 12 ∧ 
    (A % 10 < (A / 10) % 10) ∧ ((A / 10) % 10 < A / 100) ∧
    ((P A + Q A) % 2 = 0) ∧
    (∀ (B : ℕ), 
      100 ≤ B ∧ B < 1000 ∧ 
      is_whole_multiple B 12 ∧
      digit_sum B = 12 ∧ 
      (B % 10 < (B / 10) % 10) ∧ ((B / 10) % 10 < B / 100) ∧
      ((P B + Q B) % 2 = 0) →
      A ≤ B) ∧
    A = 651 := by
  sorry

end NUMINAMATH_CALUDE_smallest_whole_multiple_651_l2533_253300


namespace NUMINAMATH_CALUDE_alice_coin_difference_l2533_253336

/-- Proves that given the conditions of Alice's coin collection, she has 3 more 10-cent coins than 25-cent coins -/
theorem alice_coin_difference :
  ∀ (n d q : ℕ),
  n + d + q = 30 →
  5 * n + 10 * d + 25 * q = 435 →
  d = n + 6 →
  q = 10 →
  d - q = 3 := by
sorry

end NUMINAMATH_CALUDE_alice_coin_difference_l2533_253336


namespace NUMINAMATH_CALUDE_youtube_time_is_17_minutes_l2533_253367

/-- The total time spent on YouTube per day -/
def total_youtube_time (num_videos : ℕ) (video_length : ℕ) (ad_time : ℕ) : ℕ :=
  num_videos * video_length + ad_time

/-- Theorem stating that the total time spent on YouTube is 17 minutes -/
theorem youtube_time_is_17_minutes :
  total_youtube_time 2 7 3 = 17 := by
  sorry

end NUMINAMATH_CALUDE_youtube_time_is_17_minutes_l2533_253367


namespace NUMINAMATH_CALUDE_min_value_a_squared_l2533_253337

/-- In an acute-angled triangle ABC with sides a, b, c opposite angles A, B, C respectively,
    if b^2 * sin(C) = 4√2 * sin(B) and the area of triangle ABC is 8/3,
    then the minimum value of a^2 is 16√2/3. -/
theorem min_value_a_squared (a b c A B C : ℝ) : 
  0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2 →  -- Acute-angled triangle
  b^2 * Real.sin C = 4 * Real.sqrt 2 * Real.sin B →     -- Given condition
  (1/2) * b * c * Real.sin A = 8/3 →                    -- Area of triangle
  ∀ x, x^2 ≥ a^2 → x^2 ≥ (16 * Real.sqrt 2) / 3 :=      -- Minimum value of a^2
by sorry

end NUMINAMATH_CALUDE_min_value_a_squared_l2533_253337


namespace NUMINAMATH_CALUDE_store_rooms_problem_l2533_253330

/-- The number of rooms in Li Sangong's store -/
def num_rooms : ℕ := 8

/-- The total number of people visiting the store -/
def total_people : ℕ := 7 * num_rooms + 7

theorem store_rooms_problem :
  (total_people = 7 * num_rooms + 7) ∧
  (total_people = 9 * (num_rooms - 1)) ∧
  (num_rooms = 8) := by
  sorry

end NUMINAMATH_CALUDE_store_rooms_problem_l2533_253330


namespace NUMINAMATH_CALUDE_cubic_sum_of_roots_l2533_253322

theorem cubic_sum_of_roots (a b r s : ℝ) : 
  (r^2 - a*r + b = 0) → (s^2 - a*s + b = 0) → (r^3 + s^3 = a^3 - 3*a*b) := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_of_roots_l2533_253322


namespace NUMINAMATH_CALUDE_sandy_correct_sums_l2533_253366

theorem sandy_correct_sums :
  ∀ (c i : ℕ),
    c + i = 50 →
    3 * c - 2 * i - (50 - c) = 100 →
    c = 25 := by
  sorry

end NUMINAMATH_CALUDE_sandy_correct_sums_l2533_253366


namespace NUMINAMATH_CALUDE_jack_afternoon_emails_l2533_253354

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := 4

/-- The total number of emails Jack received in the day -/
def total_emails : ℕ := 5

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := total_emails - morning_emails

theorem jack_afternoon_emails :
  afternoon_emails = 1 :=
sorry

end NUMINAMATH_CALUDE_jack_afternoon_emails_l2533_253354


namespace NUMINAMATH_CALUDE_custom_mult_chain_l2533_253365

/-- Custom multiplication operation -/
def star_mult (a b : ℚ) : ℚ := (a - b) / (1 - a * b)

/-- Main theorem -/
theorem custom_mult_chain : star_mult 5 (star_mult 6 (star_mult 7 (star_mult 8 9))) = 3588 / 587 := by
  sorry

end NUMINAMATH_CALUDE_custom_mult_chain_l2533_253365


namespace NUMINAMATH_CALUDE_running_speed_calculation_l2533_253369

/-- Proves that the running speed is 8 km/hr given the problem conditions --/
theorem running_speed_calculation (walking_speed : ℝ) (total_distance : ℝ) (total_time : ℝ) 
  (h1 : walking_speed = 4)
  (h2 : total_distance = 8)
  (h3 : total_time = 1.5)
  (h4 : total_distance / 2 / walking_speed + total_distance / 2 / running_speed = total_time) :
  running_speed = 8 := by
  sorry


end NUMINAMATH_CALUDE_running_speed_calculation_l2533_253369


namespace NUMINAMATH_CALUDE_race_average_time_l2533_253323

theorem race_average_time (fastest_time last_three_avg : ℝ) 
  (h1 : fastest_time = 15)
  (h2 : last_three_avg = 35) : 
  (fastest_time + 3 * last_three_avg) / 4 = 30 := by
  sorry

end NUMINAMATH_CALUDE_race_average_time_l2533_253323


namespace NUMINAMATH_CALUDE_faster_train_length_l2533_253371

/-- Proves that the length of a faster train is 340 meters given the specified conditions -/
theorem faster_train_length (faster_speed slower_speed : ℝ) (crossing_time : ℝ) : 
  faster_speed = 108 →
  slower_speed = 36 →
  crossing_time = 17 →
  (faster_speed - slower_speed) * crossing_time * (5/18) = 340 :=
by sorry

end NUMINAMATH_CALUDE_faster_train_length_l2533_253371


namespace NUMINAMATH_CALUDE_people_per_column_l2533_253355

theorem people_per_column (total_people : ℕ) 
  (h1 : total_people = 30 * 16) 
  (h2 : total_people = 15 * (total_people / 15)) : 
  total_people / 15 = 32 := by
  sorry

end NUMINAMATH_CALUDE_people_per_column_l2533_253355


namespace NUMINAMATH_CALUDE_chef_pies_total_l2533_253376

theorem chef_pies_total (apple : ℕ) (pecan : ℕ) (pumpkin : ℕ) 
  (h1 : apple = 2) (h2 : pecan = 4) (h3 : pumpkin = 7) : 
  apple + pecan + pumpkin = 13 := by
  sorry

end NUMINAMATH_CALUDE_chef_pies_total_l2533_253376


namespace NUMINAMATH_CALUDE_min_abs_diff_sqrt_30_l2533_253386

theorem min_abs_diff_sqrt_30 (x : ℤ) : |x - Real.sqrt 30| ≥ |5 - Real.sqrt 30| := by
  sorry

end NUMINAMATH_CALUDE_min_abs_diff_sqrt_30_l2533_253386


namespace NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l2533_253308

theorem infinite_geometric_series_first_term
  (r : ℝ) (S : ℝ) (a : ℝ)
  (h_r : r = 1 / 4)
  (h_S : S = 40)
  (h_sum : S = a / (1 - r)) :
  a = 30 := by
sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l2533_253308


namespace NUMINAMATH_CALUDE_andy_wrong_questions_l2533_253315

theorem andy_wrong_questions (a b c d : ℕ) : 
  a + b = c + d →
  a + d = b + c + 6 →
  c = 6 →
  a = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_andy_wrong_questions_l2533_253315


namespace NUMINAMATH_CALUDE_hyperbola_y_foci_coeff_signs_l2533_253334

/-- A curve represented by the equation ax^2 + by^2 = 1 -/
structure Curve where
  a : ℝ
  b : ℝ

/-- Predicate to check if a curve is a hyperbola with foci on the y-axis -/
def is_hyperbola_y_foci (c : Curve) : Prop :=
  ∃ (p q : ℝ), p > 0 ∧ q > 0 ∧ ∀ (x y : ℝ), c.a * x^2 + c.b * y^2 = 1 ↔ x^2/p - y^2/q = 1

/-- Theorem stating that if a curve is a hyperbola with foci on the y-axis,
    then its 'a' coefficient is negative and 'b' coefficient is positive -/
theorem hyperbola_y_foci_coeff_signs (c : Curve) :
  is_hyperbola_y_foci c → c.a < 0 ∧ c.b > 0 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_y_foci_coeff_signs_l2533_253334


namespace NUMINAMATH_CALUDE_two_preserving_transformations_l2533_253375

/-- Represents the regular, infinite pattern of squares and line segments along a line ℓ -/
structure RegularPattern :=
  (ℓ : Line)
  (square_size : ℝ)
  (diagonal_length : ℝ)

/-- Enumeration of the four types of rigid motion transformations -/
inductive RigidMotion
  | Rotation
  | Translation
  | ReflectionAcross
  | ReflectionPerpendicular

/-- Predicate to check if a rigid motion maps the pattern onto itself -/
def preserves_pattern (r : RegularPattern) (m : RigidMotion) : Prop :=
  sorry

/-- The main theorem stating that exactly two rigid motions preserve the pattern -/
theorem two_preserving_transformations (r : RegularPattern) :
  ∃! (s : Finset RigidMotion), s.card = 2 ∧ ∀ m ∈ s, preserves_pattern r m :=
sorry

end NUMINAMATH_CALUDE_two_preserving_transformations_l2533_253375


namespace NUMINAMATH_CALUDE_power_sum_equality_l2533_253356

theorem power_sum_equality : (-2)^2005 + (-2)^2006 = 2^2005 := by sorry

end NUMINAMATH_CALUDE_power_sum_equality_l2533_253356


namespace NUMINAMATH_CALUDE_circle_equation_l2533_253389

/-- The equation of a circle with center (-1, 2) and radius √5 is x² + y² + 2x - 4y = 0 -/
theorem circle_equation (x y : ℝ) :
  let center : ℝ × ℝ := (-1, 2)
  let radius : ℝ := Real.sqrt 5
  (x - center.1)^2 + (y - center.2)^2 = radius^2 ↔ x^2 + y^2 + 2*x - 4*y = 0 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l2533_253389


namespace NUMINAMATH_CALUDE_reims_to_chaumont_distance_l2533_253327

/-- Represents a city in the polygon -/
inductive City
  | Chalons
  | Vitry
  | Chaumont
  | SaintQuentin
  | Reims

/-- Represents the distance between two cities -/
def distance (a b : City) : ℕ :=
  match a, b with
  | City.Chalons, City.Vitry => 30
  | City.Vitry, City.Chaumont => 80
  | City.Chaumont, City.SaintQuentin => 236
  | City.SaintQuentin, City.Reims => 86
  | City.Reims, City.Chalons => 40
  | _, _ => 0  -- For simplicity, we set other distances to 0

/-- The theorem stating the distance from Reims to Chaumont -/
theorem reims_to_chaumont_distance :
  distance City.Reims City.Chaumont = 150 :=
by sorry

end NUMINAMATH_CALUDE_reims_to_chaumont_distance_l2533_253327


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2533_253387

/-- Given a geometric sequence {a_n} where a_2020 = 8a_2017, prove that the common ratio q is 2. -/
theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- geometric sequence condition
  a 2020 = 8 * a 2017 →         -- given condition
  q = 2 :=                      -- conclusion to prove
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2533_253387


namespace NUMINAMATH_CALUDE_coloring_book_shelves_l2533_253357

theorem coloring_book_shelves (initial_stock : ℕ) (books_sold : ℕ) (books_per_shelf : ℕ) : 
  initial_stock = 435 → books_sold = 218 → books_per_shelf = 17 →
  (initial_stock - books_sold + books_per_shelf - 1) / books_per_shelf = 13 := by
sorry


end NUMINAMATH_CALUDE_coloring_book_shelves_l2533_253357


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2533_253370

theorem simplify_and_evaluate (a : ℝ) (h : a = 19) :
  (1 + 2 / (a - 1)) / ((a^2 + 2*a + 1) / (a - 1)) = 1 / 20 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2533_253370


namespace NUMINAMATH_CALUDE_absolute_value_square_inequality_l2533_253316

theorem absolute_value_square_inequality {a b : ℝ} (h : |a| < b) : a^2 < b^2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_square_inequality_l2533_253316


namespace NUMINAMATH_CALUDE_rectangular_box_volume_l2533_253344

theorem rectangular_box_volume (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  (∃ (k : ℕ), k > 0 ∧ a = 2 * k ∧ b = 3 * k ∧ c = 5 * k) →
  a * b * c = 240 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_box_volume_l2533_253344


namespace NUMINAMATH_CALUDE_binary_10011_equals_19_l2533_253381

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_10011_equals_19 :
  binary_to_decimal [true, true, false, false, true] = 19 := by
  sorry

end NUMINAMATH_CALUDE_binary_10011_equals_19_l2533_253381


namespace NUMINAMATH_CALUDE_zero_product_property_l2533_253353

theorem zero_product_property (x : ℤ) : (∀ y : ℤ, x * y = 0) → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_product_property_l2533_253353


namespace NUMINAMATH_CALUDE_solve_percentage_equation_l2533_253312

theorem solve_percentage_equation (x : ℝ) : 
  (70 / 100) * 600 = (40 / 100) * x → x = 1050 := by
  sorry

end NUMINAMATH_CALUDE_solve_percentage_equation_l2533_253312


namespace NUMINAMATH_CALUDE_half_circle_roll_midpoint_path_length_l2533_253397

/-- The length of the path traveled by the midpoint of a half-circle's diameter when rolled along a straight line -/
theorem half_circle_roll_midpoint_path_length 
  (diameter : ℝ) 
  (h_diameter : diameter = 4 / Real.pi) : 
  let radius := diameter / 2
  let circumference := 2 * Real.pi * radius
  let path_length := circumference / 2
  path_length = 2 := by sorry

end NUMINAMATH_CALUDE_half_circle_roll_midpoint_path_length_l2533_253397


namespace NUMINAMATH_CALUDE_smallest_d_l2533_253385

/-- The smallest positive value of d that satisfies the equation √((4√3)² + (d+4)²) = 2d -/
theorem smallest_d : ∃ d : ℝ, d > 0 ∧ 
  (∀ d' : ℝ, d' > 0 → (4 * Real.sqrt 3)^2 + (d' + 4)^2 = (2 * d')^2 → d ≤ d') ∧
  (4 * Real.sqrt 3)^2 + (d + 4)^2 = (2 * d)^2 ∧
  d = (2 * (2 - Real.sqrt 52)) / 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_d_l2533_253385


namespace NUMINAMATH_CALUDE_collatz_100th_term_l2533_253346

def collatz (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else 3 * n + 1

def collatzSequence (n : ℕ) : ℕ → ℕ
  | 0 => 6
  | m + 1 => collatz (collatzSequence n m)

theorem collatz_100th_term :
  collatzSequence 100 99 = 4 := by sorry

end NUMINAMATH_CALUDE_collatz_100th_term_l2533_253346


namespace NUMINAMATH_CALUDE_sum_highest_powers_12_18_divides_20_factorial_l2533_253384

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def highest_power_divides (base k : ℕ) : ℕ → ℕ
| 0 => 0
| n + 1 => if (factorial n) % (base ^ (k + 1)) = 0 then highest_power_divides base (k + 1) n else k

theorem sum_highest_powers_12_18_divides_20_factorial :
  (highest_power_divides 12 0 20) + (highest_power_divides 18 0 20) = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_highest_powers_12_18_divides_20_factorial_l2533_253384


namespace NUMINAMATH_CALUDE_farm_feet_count_l2533_253358

/-- A farm with hens and cows -/
structure Farm where
  hens : ℕ
  cows : ℕ

/-- The total number of heads in the farm -/
def total_heads (f : Farm) : ℕ := f.hens + f.cows

/-- The total number of feet in the farm -/
def total_feet (f : Farm) : ℕ := 2 * f.hens + 4 * f.cows

/-- Theorem: Given a farm with 48 heads and 28 hens, the total number of feet is 136 -/
theorem farm_feet_count :
  ∀ f : Farm, total_heads f = 48 → f.hens = 28 → total_feet f = 136 :=
by
  sorry

end NUMINAMATH_CALUDE_farm_feet_count_l2533_253358


namespace NUMINAMATH_CALUDE_abs_neg_two_l2533_253333

theorem abs_neg_two : |(-2 : ℝ)| = 2 := by sorry

end NUMINAMATH_CALUDE_abs_neg_two_l2533_253333
