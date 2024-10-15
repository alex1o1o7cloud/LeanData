import Mathlib

namespace NUMINAMATH_CALUDE_trapezoid_existence_l3668_366860

/-- Represents a trapezoid ABCD with AB parallel to CD -/
structure Trapezoid where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- Checks if the given points form a valid trapezoid -/
def is_valid_trapezoid (t : Trapezoid) : Prop := sorry

/-- Calculates the perimeter of a trapezoid -/
def perimeter (t : Trapezoid) : ℝ := sorry

/-- Calculates the length of diagonal AC -/
def diagonal_ac (t : Trapezoid) : ℝ := sorry

/-- Calculates the angle DAB -/
def angle_dab (t : Trapezoid) : ℝ := sorry

/-- Calculates the angle ABC -/
def angle_abc (t : Trapezoid) : ℝ := sorry

/-- Theorem: Given a perimeter k, diagonal e, and angles α and β,
    there exists 0, 1, or 2 trapezoids satisfying these conditions -/
theorem trapezoid_existence (k e α β : ℝ) :
  ∃ n : Fin 3, ∃ ts : Finset Trapezoid,
    ts.card = n ∧
    ∀ t ∈ ts, is_valid_trapezoid t ∧
               perimeter t = k ∧
               diagonal_ac t = e ∧
               angle_dab t = α ∧
               angle_abc t = β := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_existence_l3668_366860


namespace NUMINAMATH_CALUDE_houses_before_boom_count_l3668_366834

/-- The number of houses in Lawrence County before the housing boom -/
def houses_before_boom : ℕ := 2000 - 574

/-- The current number of houses in Lawrence County -/
def current_houses : ℕ := 2000

/-- The number of houses built during the housing boom -/
def houses_built : ℕ := 574

theorem houses_before_boom_count : houses_before_boom = 1426 := by
  sorry

end NUMINAMATH_CALUDE_houses_before_boom_count_l3668_366834


namespace NUMINAMATH_CALUDE_total_cost_is_360_l3668_366892

def calculate_total_cost (sale_prices : List ℝ) (discounts : List ℝ) 
  (installation_fee : ℝ) (disposal_fee : ℝ) : ℝ :=
  let discounted_prices := List.zipWith (·-·) sale_prices discounts
  let with_installation := List.map (·+installation_fee) discounted_prices
  let total_per_tire := List.map (·+disposal_fee) with_installation
  List.sum total_per_tire

theorem total_cost_is_360 :
  let sale_prices := [75, 90, 120, 150]
  let discounts := [20, 30, 45, 60]
  let installation_fee := 15
  let disposal_fee := 5
  calculate_total_cost sale_prices discounts installation_fee disposal_fee = 360 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_360_l3668_366892


namespace NUMINAMATH_CALUDE_company_uses_systematic_sampling_l3668_366894

/-- Represents a sampling method -/
inductive SamplingMethod
| LotteryMethod
| RandomNumberTableMethod
| SystematicSampling
| StratifiedSampling

/-- Represents a production line -/
structure ProductionLine :=
  (uniform : Bool)

/-- Represents a sampling process -/
structure SamplingProcess :=
  (line : ProductionLine)
  (interval : ℕ)

/-- Determines if a sampling process is systematic -/
def is_systematic (process : SamplingProcess) : Prop :=
  process.line.uniform ∧ process.interval > 0

/-- The company's sampling method -/
def company_sampling : SamplingProcess :=
  { line := { uniform := true },
    interval := 10 }

/-- Theorem stating that the company's sampling method is systematic sampling -/
theorem company_uses_systematic_sampling :
  is_systematic company_sampling ∧ 
  SamplingMethod.SystematicSampling = 
    (match company_sampling with
     | { line := { uniform := true }, interval := 10 } => SamplingMethod.SystematicSampling
     | _ => SamplingMethod.LotteryMethod) :=
sorry

end NUMINAMATH_CALUDE_company_uses_systematic_sampling_l3668_366894


namespace NUMINAMATH_CALUDE_not_p_or_q_false_implies_p_and_q_false_l3668_366898

theorem not_p_or_q_false_implies_p_and_q_false (p q : Prop) :
  (¬(¬p ∨ q)) → ¬(p ∧ q) := by
  sorry

end NUMINAMATH_CALUDE_not_p_or_q_false_implies_p_and_q_false_l3668_366898


namespace NUMINAMATH_CALUDE_circle_area_increase_l3668_366882

theorem circle_area_increase (r : ℝ) (h : r > 0) :
  let new_radius := 1.5 * r
  let original_area := π * r^2
  let new_area := π * new_radius^2
  (new_area - original_area) / original_area = 1.25 := by
sorry

end NUMINAMATH_CALUDE_circle_area_increase_l3668_366882


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3668_366874

/-- An arithmetic sequence with specific properties -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_problem (a : ℕ → ℚ) 
    (h_arith : ArithmeticSequence a)
    (h_a1 : a 1 = 1/3)
    (h_sum : a 2 + a 5 = 4)
    (h_an : ∃ n : ℕ, a n = 33) :
  ∃ n : ℕ, a n = 33 ∧ n = 50 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3668_366874


namespace NUMINAMATH_CALUDE_inequality_solution_l3668_366827

theorem inequality_solution (x : ℝ) :
  (3*x + 4 ≠ 0) →
  (3 - 2 / (3*x + 4) < 5 ↔ x ∈ Set.Ioo (-5/3) (-4/3) ∪ Set.Ioi (-4/3)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3668_366827


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l3668_366849

theorem inverse_variation_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_inverse : ∃ k : ℝ, ∀ x y, x^2 * y = k) 
  (h_initial : 3^2 * 8 = 9 * 8) 
  (h_final : y = 648) : x = 1/3 := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l3668_366849


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3668_366842

/-- An isosceles triangle with sides a, b, and c, where two sides are equal. -/
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  isosceles : (a = b) ∨ (b = c) ∨ (a = c)
  positive : a > 0 ∧ b > 0 ∧ c > 0

/-- The perimeter of a triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ := t.a + t.b + t.c

/-- Theorem stating that an isosceles triangle with sides 3 and 4 has perimeter 10 or 11 -/
theorem isosceles_triangle_perimeter :
  ∀ t : IsoscelesTriangle, 
  ((t.a = 3 ∧ t.b = 4) ∨ (t.a = 4 ∧ t.b = 3) ∨ 
   (t.b = 3 ∧ t.c = 4) ∨ (t.b = 4 ∧ t.c = 3) ∨ 
   (t.a = 3 ∧ t.c = 4) ∨ (t.a = 4 ∧ t.c = 3)) →
  (perimeter t = 10 ∨ perimeter t = 11) :=
by sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3668_366842


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l3668_366852

-- Define the given line
def given_line (x y : ℝ) : Prop := 3 * x - 4 * y + 6 = 0

-- Define the point P
def point_P : ℝ × ℝ := (4, -1)

-- Define the parallel line
def parallel_line (x y : ℝ) : Prop := 3 * x - 4 * y - 16 = 0

-- Theorem statement
theorem parallel_line_through_point :
  (parallel_line point_P.1 point_P.2) ∧ 
  (∀ (x y : ℝ), parallel_line x y ↔ ∃ (k : ℝ), given_line (x + k) (y + (3/4) * k)) :=
sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l3668_366852


namespace NUMINAMATH_CALUDE_sequence_formula_l3668_366888

def sequence_a (n : ℕ) : ℕ := 2^n - 1

def sum_S : ℕ → ℕ
  | 0 => 0
  | n + 1 => 2 * sum_S n + n + 1

theorem sequence_formula (n : ℕ) :
  n > 0 →
  sequence_a 1 = 1 ∧
  (∀ k, k > 0 → sum_S (k + 1) = 2 * sum_S k + k + 1) →
  sequence_a n = sum_S n - sum_S (n - 1) :=
sorry

end NUMINAMATH_CALUDE_sequence_formula_l3668_366888


namespace NUMINAMATH_CALUDE_f_derivative_l3668_366875

/-- The function f(x) = (5x - 4)^3 -/
def f (x : ℝ) : ℝ := (5 * x - 4) ^ 3

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 15 * (5 * x - 4) ^ 2

theorem f_derivative :
  ∀ x : ℝ, deriv f x = f' x :=
by
  sorry

end NUMINAMATH_CALUDE_f_derivative_l3668_366875


namespace NUMINAMATH_CALUDE_square_area_ratio_l3668_366800

theorem square_area_ratio (s₂ : ℝ) (h : s₂ > 0) : 
  let s₁ := (s₂ * Real.sqrt 2) / 2
  (s₁^2) / (s₂^2) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_square_area_ratio_l3668_366800


namespace NUMINAMATH_CALUDE_adam_jackie_apple_difference_l3668_366854

theorem adam_jackie_apple_difference :
  ∀ (adam_apples jackie_apples : ℕ),
    adam_apples = 10 →
    jackie_apples = 2 →
    adam_apples - jackie_apples = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_adam_jackie_apple_difference_l3668_366854


namespace NUMINAMATH_CALUDE_rightmost_three_digits_of_7_to_1987_l3668_366866

theorem rightmost_three_digits_of_7_to_1987 :
  7^1987 % 1000 = 543 := by
  sorry

end NUMINAMATH_CALUDE_rightmost_three_digits_of_7_to_1987_l3668_366866


namespace NUMINAMATH_CALUDE_equation_solution_l3668_366801

theorem equation_solution :
  ∃ x : ℚ, (x + 3*x = 300 - (4*x + 5*x)) ∧ (x = 300/13) := by sorry

end NUMINAMATH_CALUDE_equation_solution_l3668_366801


namespace NUMINAMATH_CALUDE_turnip_bag_weights_l3668_366869

def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]

def is_valid_turnip_weight (t : ℕ) : Prop :=
  t ∈ bag_weights ∧
  ∃ (onion_weight carrot_weight : ℕ),
    onion_weight + carrot_weight = (bag_weights.sum - t) ∧
    carrot_weight = 2 * onion_weight ∧
    ∃ (onion_bags carrot_bags : List ℕ),
      onion_bags ++ carrot_bags = bag_weights.filter (λ x => x ≠ t) ∧
      onion_bags.sum = onion_weight ∧
      carrot_bags.sum = carrot_weight

theorem turnip_bag_weights :
  {t : ℕ | is_valid_turnip_weight t} = {13, 16} := by
  sorry

end NUMINAMATH_CALUDE_turnip_bag_weights_l3668_366869


namespace NUMINAMATH_CALUDE_min_value_on_negative_interval_l3668_366859

/-- A function is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- The function F defined in terms of f and g -/
def F (f g : ℝ → ℝ) (a b : ℝ) (x : ℝ) : ℝ := a * f x + b * g x + 3

theorem min_value_on_negative_interval
  (f g : ℝ → ℝ) (a b : ℝ)
  (hf : IsOdd f) (hg : IsOdd g)
  (hmax : ∀ x > 0, F f g a b x ≤ 10) :
  ∀ x < 0, F f g a b x ≥ -4 :=
sorry

end NUMINAMATH_CALUDE_min_value_on_negative_interval_l3668_366859


namespace NUMINAMATH_CALUDE_tan_276_equals_96_l3668_366813

theorem tan_276_equals_96 : 
  ∃! (n : ℤ), -180 < n ∧ n < 180 ∧ Real.tan (n * π / 180) = Real.tan (276 * π / 180) :=
by
  sorry

end NUMINAMATH_CALUDE_tan_276_equals_96_l3668_366813


namespace NUMINAMATH_CALUDE_blue_whale_tongue_weight_l3668_366824

/-- The weight of an adult blue whale's tongue in pounds -/
def tongue_weight : ℕ := 6000

/-- The number of pounds in one ton -/
def pounds_per_ton : ℕ := 2000

/-- The weight of an adult blue whale's tongue in tons -/
def tongue_weight_in_tons : ℚ := tongue_weight / pounds_per_ton

theorem blue_whale_tongue_weight : tongue_weight_in_tons = 3 := by
  sorry

end NUMINAMATH_CALUDE_blue_whale_tongue_weight_l3668_366824


namespace NUMINAMATH_CALUDE_fox_can_eat_80_fox_cannot_eat_65_l3668_366887

/-- Represents a distribution of candies into three piles -/
structure CandyDistribution :=
  (pile1 pile2 pile3 : ℕ)
  (sum_eq_100 : pile1 + pile2 + pile3 = 100)

/-- Calculates the number of candies the fox eats given a distribution -/
def fox_candies (d : CandyDistribution) : ℕ :=
  if d.pile1 = d.pile2 ∨ d.pile1 = d.pile3 ∨ d.pile2 = d.pile3
  then max d.pile1 (max d.pile2 d.pile3)
  else d.pile1 + d.pile2 + d.pile3 - 2 * min d.pile1 (min d.pile2 d.pile3)

theorem fox_can_eat_80 : ∃ d : CandyDistribution, fox_candies d = 80 := by
  sorry

theorem fox_cannot_eat_65 : ¬ ∃ d : CandyDistribution, fox_candies d = 65 := by
  sorry

end NUMINAMATH_CALUDE_fox_can_eat_80_fox_cannot_eat_65_l3668_366887


namespace NUMINAMATH_CALUDE_derivative_log_base_3_l3668_366806

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 3

theorem derivative_log_base_3 (x : ℝ) (h : x > 0) :
  deriv f x = 1 / (x * Real.log 3) :=
by sorry

end NUMINAMATH_CALUDE_derivative_log_base_3_l3668_366806


namespace NUMINAMATH_CALUDE_daily_reading_goal_l3668_366830

def september_days : ℕ := 30
def total_pages : ℕ := 600
def unavailable_days : ℕ := 10
def flight_day_pages : ℕ := 100

def available_days : ℕ := september_days - unavailable_days - 1
def remaining_pages : ℕ := total_pages - flight_day_pages

theorem daily_reading_goal :
  ∃ (pages_per_day : ℕ),
    pages_per_day * available_days ≥ remaining_pages ∧
    pages_per_day = 27 := by
  sorry

end NUMINAMATH_CALUDE_daily_reading_goal_l3668_366830


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3668_366841

theorem absolute_value_inequality_solution_set (x : ℝ) :
  (|x - 1| < 1) ↔ (x ∈ Set.Ioo 0 2) :=
sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3668_366841


namespace NUMINAMATH_CALUDE_floor_add_two_floor_sum_inequality_floor_square_inequality_exists_l3668_366864

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Theorem 1
theorem floor_add_two (x : ℝ) : floor (x + 2) = floor x + 2 := by sorry

-- Theorem 2
theorem floor_sum_inequality (x y : ℝ) : floor (x + y) ≤ floor x + floor y := by sorry

-- Theorem 3
theorem floor_square_inequality_exists :
  ∃ x : ℝ, floor (x^2) ≠ (floor x)^2 := by sorry

end NUMINAMATH_CALUDE_floor_add_two_floor_sum_inequality_floor_square_inequality_exists_l3668_366864


namespace NUMINAMATH_CALUDE_sine_cosine_equality_l3668_366810

theorem sine_cosine_equality (n : ℤ) (h1 : -180 ≤ n) (h2 : n ≤ 180) :
  Real.sin (n * Real.pi / 180) = Real.cos (510 * Real.pi / 180) → n = -60 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_equality_l3668_366810


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3668_366881

theorem complex_equation_solution (x y : ℝ) :
  (x + y - 3 : ℂ) + (x - 4 : ℂ) * I = 0 → x = 4 ∧ y = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3668_366881


namespace NUMINAMATH_CALUDE_average_visitors_is_750_l3668_366840

/-- Calculates the average number of visitors per day in a 30-day month starting on Sunday -/
def average_visitors (sunday_visitors : ℕ) (other_day_visitors : ℕ) : ℚ :=
  let total_sundays : ℕ := 5
  let total_other_days : ℕ := 30 - total_sundays
  let total_visitors : ℕ := sunday_visitors * total_sundays + other_day_visitors * total_other_days
  (total_visitors : ℚ) / 30

/-- Theorem stating that the average number of visitors per day is 750 -/
theorem average_visitors_is_750 :
  average_visitors 1000 700 = 750 := by sorry

end NUMINAMATH_CALUDE_average_visitors_is_750_l3668_366840


namespace NUMINAMATH_CALUDE_conic_is_ellipse_l3668_366838

/-- The equation of the conic section -/
def conic_equation (x y : ℝ) : Prop :=
  x^2 + 4*y^2 - 6*x + 8*y + 9 = 0

/-- Definition of an ellipse in standard form -/
def is_ellipse (h k a b : ℝ) (x y : ℝ) : Prop :=
  ((x - h)^2 / a^2) + ((y - k)^2 / b^2) = 1

/-- Theorem stating that the given equation represents an ellipse -/
theorem conic_is_ellipse :
  ∃ h k a b : ℝ, ∀ x y : ℝ, conic_equation x y ↔ is_ellipse h k a b x y :=
sorry

end NUMINAMATH_CALUDE_conic_is_ellipse_l3668_366838


namespace NUMINAMATH_CALUDE_function_properties_l3668_366885

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def periodic_negative (f : ℝ → ℝ) : Prop := ∀ x, f (x + 1) = -f x

def increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y

def symmetric_about (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

theorem function_properties (f : ℝ → ℝ) 
  (h1 : is_even f)
  (h2 : periodic_negative f)
  (h3 : increasing_on f (-1) 0) :
  (f 2 = f 0) ∧ (symmetric_about f 1) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l3668_366885


namespace NUMINAMATH_CALUDE_max_regions_three_triangles_is_20_l3668_366853

/-- The maximum number of regions formed by three triangles on a plane -/
def max_regions_three_triangles : ℕ := 20

/-- The number of triangles drawn on the plane -/
def num_triangles : ℕ := 3

/-- Theorem stating that the maximum number of regions formed by three triangles is 20 -/
theorem max_regions_three_triangles_is_20 :
  max_regions_three_triangles = 20 ∧ num_triangles = 3 := by sorry

end NUMINAMATH_CALUDE_max_regions_three_triangles_is_20_l3668_366853


namespace NUMINAMATH_CALUDE_max_value_on_parabola_l3668_366896

/-- The maximum value of m + n for a point (m, n) on the graph of y = -x^2 + 3 is 13/4 -/
theorem max_value_on_parabola : 
  ∃ (max : ℝ), max = 13/4 ∧ 
  ∀ (m n : ℝ), n = -m^2 + 3 → m + n ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_on_parabola_l3668_366896


namespace NUMINAMATH_CALUDE_total_meows_eq_286_l3668_366812

/-- The number of meows for eight cats over 12 minutes -/
def total_meows : ℕ :=
  let cat1_meows := 3 * 12
  let cat2_meows := (3 * 2) * 12
  let cat3_meows := ((3 * 2) / 3) * 12
  let cat4_meows := 4 * 12
  let cat5_meows := (60 / 45) * 12
  let cat6_meows := (5 / 2) * 12
  let cat7_meows := ((3 * 2) / 2) * 12
  let cat8_meows := (6 / 3) * 12
  cat1_meows + cat2_meows + cat3_meows + cat4_meows + 
  cat5_meows + cat6_meows + cat7_meows + cat8_meows

theorem total_meows_eq_286 : total_meows = 286 := by
  sorry

#eval total_meows

end NUMINAMATH_CALUDE_total_meows_eq_286_l3668_366812


namespace NUMINAMATH_CALUDE_sum_of_integers_with_product_5_4_l3668_366809

theorem sum_of_integers_with_product_5_4 (a b c : ℕ+) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a * b * c = 625 →
  (a : ℕ) + (b : ℕ) + (c : ℕ) = 131 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_with_product_5_4_l3668_366809


namespace NUMINAMATH_CALUDE_folded_rectangle_area_l3668_366844

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  topLeft : Point
  bottomRight : Point

/-- Represents the folding scenario -/
structure FoldedRectangle where
  original : Rectangle
  t : Point
  u : Point
  qPrime : Point
  pPrime : Point

theorem folded_rectangle_area (fold : FoldedRectangle) 
  (h1 : fold.t.x - fold.original.topLeft.x < fold.original.bottomRight.x - fold.u.x)
  (h2 : (fold.pPrime.x - fold.qPrime.x)^2 + (fold.pPrime.y - fold.qPrime.y)^2 = 
        (fold.original.bottomRight.y - fold.original.topLeft.y)^2)
  (h3 : fold.qPrime.x - fold.original.topLeft.x = 8)
  (h4 : fold.t.x - fold.original.topLeft.x = 36) :
  (fold.original.bottomRight.x - fold.original.topLeft.x) * 
  (fold.original.bottomRight.y - fold.original.topLeft.y) = 288 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_folded_rectangle_area_l3668_366844


namespace NUMINAMATH_CALUDE_scooter_final_price_l3668_366835

/-- The final sale price of a scooter after two consecutive discounts -/
theorem scooter_final_price (initial_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : 
  initial_price = 150 ∧ discount1 = 0.4 ∧ discount2 = 0.35 →
  initial_price * (1 - discount1) * (1 - discount2) = 58.50 := by
sorry

end NUMINAMATH_CALUDE_scooter_final_price_l3668_366835


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l3668_366884

theorem polynomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x, (1 - 2*x)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  |a₀| + |a₁| + |a₃| = 41 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l3668_366884


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l3668_366814

theorem condition_necessary_not_sufficient : 
  (∃ x : ℝ, (x - 1) * (x + 2) = 0 ∧ x ≠ 1) ∧ 
  (∀ x : ℝ, x = 1 → (x - 1) * (x + 2) = 0) := by
  sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l3668_366814


namespace NUMINAMATH_CALUDE_square_area_theorem_l3668_366837

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a rectangle -/
structure Rectangle :=
  (topLeft : Point)
  (bottomRight : Point)

/-- Represents a square -/
structure Square :=
  (topLeft : Point)
  (sideLength : ℝ)

/-- Calculates the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℝ :=
  (r.bottomRight.x - r.topLeft.x) * (r.topLeft.y - r.bottomRight.y)

/-- Theorem: If a square is divided into four rectangles of equal area and MN = 3,
    then the area of the square is 64 -/
theorem square_area_theorem (s : Square) 
  (r1 r2 r3 r4 : Rectangle) 
  (h1 : rectangleArea r1 = rectangleArea r2)
  (h2 : rectangleArea r2 = rectangleArea r3)
  (h3 : rectangleArea r3 = rectangleArea r4)
  (h4 : r1.topLeft = s.topLeft)
  (h5 : r4.bottomRight.x = s.topLeft.x + s.sideLength)
  (h6 : r4.bottomRight.y = s.topLeft.y - s.sideLength)
  (h7 : r1.bottomRight.x - r1.topLeft.x = 3) : 
  s.sideLength * s.sideLength = 64 :=
sorry

end NUMINAMATH_CALUDE_square_area_theorem_l3668_366837


namespace NUMINAMATH_CALUDE_mary_score_unique_l3668_366858

/-- Represents the scoring system for the AHSME -/
structure AHSMEScore where
  correct : ℕ
  wrong : ℕ
  score : ℕ
  total_problems : ℕ := 30
  score_formula : score = 35 + 5 * correct - wrong
  valid_answers : correct + wrong ≤ total_problems

/-- Represents the condition for John to uniquely determine Mary's score -/
def uniquely_determinable (s : AHSMEScore) : Prop :=
  ∀ s' : AHSMEScore, s'.score > 90 → s'.score ≤ s.score → s' = s

/-- Mary's AHSME score satisfies all conditions and is uniquely determinable -/
theorem mary_score_unique : 
  ∃! s : AHSMEScore, s.score > 90 ∧ uniquely_determinable s ∧ 
  s.correct = 12 ∧ s.wrong = 0 ∧ s.score = 95 := by
  sorry


end NUMINAMATH_CALUDE_mary_score_unique_l3668_366858


namespace NUMINAMATH_CALUDE_apples_left_over_l3668_366848

theorem apples_left_over (liam mia noah : ℕ) (h1 : liam = 53) (h2 : mia = 68) (h3 : noah = 22) : 
  (liam + mia + noah) % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_apples_left_over_l3668_366848


namespace NUMINAMATH_CALUDE_contrapositive_proof_l3668_366817

theorem contrapositive_proof (a : ℝ) : 
  a < 1 → ∀ x : ℝ, x^2 + (2*a+1)*x + a^2 + 2 > 0 := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_proof_l3668_366817


namespace NUMINAMATH_CALUDE_min_value_of_f_l3668_366828

-- Define the function f(x)
def f (x : ℝ) : ℝ := 12 * x - x^3

-- State the theorem
theorem min_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Icc (-3) 3 ∧ f x = -16 ∧ ∀ y ∈ Set.Icc (-3) 3, f y ≥ f x :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3668_366828


namespace NUMINAMATH_CALUDE_company_blocks_l3668_366863

/-- Represents the number of workers in each block -/
def workers_per_block : ℕ := 200

/-- Represents the total budget for gifts in dollars -/
def total_budget : ℕ := 6000

/-- Represents the cost of each gift in dollars -/
def gift_cost : ℕ := 2

/-- Calculates the number of blocks in the company -/
def number_of_blocks : ℕ := total_budget / (workers_per_block * gift_cost)

/-- Theorem stating that the number of blocks in the company is 15 -/
theorem company_blocks : number_of_blocks = 15 := by
  sorry

end NUMINAMATH_CALUDE_company_blocks_l3668_366863


namespace NUMINAMATH_CALUDE_range_of_a_l3668_366820

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x + 4 > 0
def q (a : ℝ) : Prop := ∀ x : ℝ, Monotone (fun x => (3 - 2*a)^x)

-- Define the theorem
theorem range_of_a (a : ℝ) : p a ∧ ¬(q a) → a ∈ Set.Icc 1 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3668_366820


namespace NUMINAMATH_CALUDE_hexagonal_pyramid_base_neq_slant_l3668_366878

/-- A regular hexagonal pyramid -/
structure RegularHexagonalPyramid where
  /-- The edge length of the base -/
  baseEdge : ℝ
  /-- The slant height of the pyramid -/
  slantHeight : ℝ
  /-- The apex angle of each lateral face -/
  apexAngle : ℝ
  /-- Condition: baseEdge and slantHeight are positive -/
  baseEdge_pos : baseEdge > 0
  slantHeight_pos : slantHeight > 0
  /-- Condition: The apex angle is determined by the baseEdge and slantHeight -/
  apexAngle_eq : apexAngle = 2 * Real.arcsin (baseEdge / (2 * slantHeight))

/-- Theorem: It's impossible for a regular hexagonal pyramid to have its base edge length equal to its slant height -/
theorem hexagonal_pyramid_base_neq_slant (p : RegularHexagonalPyramid) : 
  p.baseEdge ≠ p.slantHeight := by
  sorry

end NUMINAMATH_CALUDE_hexagonal_pyramid_base_neq_slant_l3668_366878


namespace NUMINAMATH_CALUDE_equal_triplet_solution_l3668_366895

theorem equal_triplet_solution {a b c : ℝ} (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0)
  (h1 : a * (a + b) = b * (b + c)) (h2 : b * (b + c) = c * (c + a)) :
  a = b ∧ b = c := by
sorry

end NUMINAMATH_CALUDE_equal_triplet_solution_l3668_366895


namespace NUMINAMATH_CALUDE_evaluate_expression_l3668_366891

theorem evaluate_expression : (0.5^4 / 0.05^3) = 500 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3668_366891


namespace NUMINAMATH_CALUDE_addition_subtraction_proof_l3668_366857

theorem addition_subtraction_proof : 987 + 113 - 1000 = 100 := by
  sorry

end NUMINAMATH_CALUDE_addition_subtraction_proof_l3668_366857


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l3668_366823

/-- A line is tangent to a parabola if and only if the discriminant of the resulting quadratic equation is zero -/
axiom tangent_condition (a b c : ℝ) : 
  b^2 - 4*a*c = 0 ↔ ∃ k, (∀ x y, a*y^2 + b*y + c = 0 ∧ y^2 = 16*x ∧ 6*x - 4*y + k = 0)

/-- The value of k for which the line 6x - 4y + k = 0 is tangent to the parabola y^2 = 16x -/
theorem line_tangent_to_parabola : 
  ∃! k, ∀ x y, (y^2 = 16*x ∧ 6*x - 4*y + k = 0) → k = 32/3 := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l3668_366823


namespace NUMINAMATH_CALUDE_alpha_range_l3668_366868

noncomputable def f (α : Real) (x : Real) : Real := Real.log x + Real.tan α

theorem alpha_range (α : Real) (x₀ : Real) :
  α ∈ Set.Ioo 0 (Real.pi / 2) →
  x₀ < 1 →
  x₀ > 0 →
  (fun x => 1 / x) x₀ = f α x₀ →
  α ∈ Set.Ioo (Real.pi / 4) (Real.pi / 2) :=
by sorry

end NUMINAMATH_CALUDE_alpha_range_l3668_366868


namespace NUMINAMATH_CALUDE_number_of_teams_in_league_l3668_366846

theorem number_of_teams_in_league : ∃ n : ℕ, n > 0 ∧ n * (n - 1) / 2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_number_of_teams_in_league_l3668_366846


namespace NUMINAMATH_CALUDE_ant_path_distance_l3668_366815

/-- Represents the rectangle in which the ant walks --/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents the ant's path --/
structure AntPath where
  start : ℝ  -- Distance from the nearest corner to the starting point X
  angle : ℝ  -- Angle of the path with respect to the sides of the rectangle

/-- Theorem stating the conditions and the result to be proved --/
theorem ant_path_distance (rect : Rectangle) (path : AntPath) :
  rect.width = 18 ∧ 
  rect.height = 150 ∧ 
  path.angle = 45 ∧ 
  path.start ≥ 0 ∧ 
  path.start ≤ rect.width ∧
  (∃ n : ℕ, n * rect.width = rect.height / 2) →
  path.start = 3 := by
  sorry

end NUMINAMATH_CALUDE_ant_path_distance_l3668_366815


namespace NUMINAMATH_CALUDE_number_of_ferries_divisible_by_four_l3668_366883

/-- Represents a ferry route between two points across a lake. -/
structure FerryRoute where
  /-- Time interval between ferry departures -/
  departureInterval : ℕ
  /-- Time taken to cross the lake -/
  crossingTime : ℕ
  /-- Number of ferries arriving during docking time -/
  arrivingFerries : ℕ

/-- Theorem stating that the number of ferries on a route with given conditions is divisible by 4 -/
theorem number_of_ferries_divisible_by_four (route : FerryRoute) 
  (h1 : route.crossingTime = route.arrivingFerries * route.departureInterval)
  (h2 : route.crossingTime > 0) : 
  ∃ (n : ℕ), (4 * route.crossingTime) / route.departureInterval = 4 * n := by
  sorry


end NUMINAMATH_CALUDE_number_of_ferries_divisible_by_four_l3668_366883


namespace NUMINAMATH_CALUDE_parallelogram_condition_l3668_366890

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0
  h_a_gt_b : a > b

/-- Checks if a point lies on the unit circle -/
def onUnitCircle (p : Point) : Prop :=
  p.x^2 + p.y^2 = 1

/-- Checks if a point lies on the given ellipse -/
def onEllipse (p : Point) (e : Ellipse) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- The main theorem stating the condition for the parallelogram property -/
theorem parallelogram_condition (e : Ellipse) :
  (∀ p : Point, onEllipse p e → 
    ∃ q r s : Point, 
      onEllipse q e ∧ onEllipse r e ∧ onEllipse s e ∧
      onUnitCircle q ∧ onUnitCircle s ∧
      -- Additional conditions for parallelogram property would be defined here
      True) → 
  1 / e.a^2 + 1 / e.b^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_parallelogram_condition_l3668_366890


namespace NUMINAMATH_CALUDE_distance_calculation_l3668_366899

/-- The distance between Maxwell's and Brad's homes -/
def distance_between_homes : ℝ := 94

/-- Maxwell's walking speed in km/h -/
def maxwell_speed : ℝ := 4

/-- Brad's running speed in km/h -/
def brad_speed : ℝ := 6

/-- Time Maxwell walks before meeting Brad, in hours -/
def maxwell_time : ℝ := 10

/-- Time difference between Maxwell's start and Brad's start, in hours -/
def time_difference : ℝ := 1

theorem distance_calculation :
  distance_between_homes = 
    maxwell_speed * maxwell_time + 
    brad_speed * (maxwell_time - time_difference) :=
by sorry

end NUMINAMATH_CALUDE_distance_calculation_l3668_366899


namespace NUMINAMATH_CALUDE_orange_juice_serving_size_l3668_366850

/-- Given the conditions for preparing orange juice, prove that each serving is 6 ounces. -/
theorem orange_juice_serving_size :
  -- Conditions
  (concentrate_to_water_ratio : ℚ) →
  (concentrate_cans : ℕ) →
  (concentrate_size : ℚ) →
  (total_servings : ℕ) →
  -- Assumptions
  concentrate_to_water_ratio = 1 / 4 →
  concentrate_cans = 45 →
  concentrate_size = 12 →
  total_servings = 360 →
  -- Conclusion
  (total_volume : ℚ) →
  total_volume = concentrate_cans * concentrate_size * (1 + 1 / concentrate_to_water_ratio) →
  total_volume / total_servings = 6 :=
by sorry

end NUMINAMATH_CALUDE_orange_juice_serving_size_l3668_366850


namespace NUMINAMATH_CALUDE_not_perfect_square_for_prime_l3668_366897

theorem not_perfect_square_for_prime (p : ℕ) (h : Prime p) : ¬ ∃ t : ℤ, (7 * p + 3^p - 4 : ℤ) = t^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_for_prime_l3668_366897


namespace NUMINAMATH_CALUDE_special_triangle_line_BC_l3668_366804

/-- A triangle ABC with vertex A at (-4, 2) and two medians on specific lines -/
structure SpecialTriangle where
  /-- Vertex A of the triangle -/
  A : ℝ × ℝ
  /-- The line containing one median -/
  median1 : ℝ → ℝ → ℝ
  /-- The line containing another median -/
  median2 : ℝ → ℝ → ℝ
  /-- Condition: A is at (-4, 2) -/
  h_A : A = (-4, 2)
  /-- Condition: One median lies on 3x - 2y + 2 = 0 -/
  h_median1 : median1 x y = 3*x - 2*y + 2
  /-- Condition: Another median lies on 3x + 5y - 12 = 0 -/
  h_median2 : median2 x y = 3*x + 5*y - 12

/-- The equation of line BC in the special triangle -/
def lineBCEq (t : SpecialTriangle) (x y : ℝ) : ℝ := 2*x + y - 8

/-- Theorem: The equation of line BC in the special triangle is 2x + y - 8 = 0 -/
theorem special_triangle_line_BC (t : SpecialTriangle) :
  ∀ x y, lineBCEq t x y = 0 ↔ y = -2*x + 8 :=
sorry

end NUMINAMATH_CALUDE_special_triangle_line_BC_l3668_366804


namespace NUMINAMATH_CALUDE_cistern_filling_time_l3668_366851

/-- The time it takes to fill a cistern when two taps (one filling, one emptying) are opened simultaneously -/
theorem cistern_filling_time 
  (fill_time : ℝ) 
  (empty_time : ℝ) 
  (fill_time_pos : 0 < fill_time)
  (empty_time_pos : 0 < empty_time) : 
  (fill_time * empty_time) / (empty_time - fill_time) = 
    1 / (1 / fill_time - 1 / empty_time) :=
by sorry

end NUMINAMATH_CALUDE_cistern_filling_time_l3668_366851


namespace NUMINAMATH_CALUDE_odd_integers_count_odd_integers_three_different_digits_count_l3668_366877

theorem odd_integers_count : ℕ := by
  -- Define the range of integers
  let lower_bound : ℕ := 2000
  let upper_bound : ℕ := 3000

  -- Define the set of possible odd units digits
  let odd_units : Finset ℕ := {1, 3, 5, 7, 9}

  -- Define the count of choices for each digit position
  let thousands_choices : ℕ := 1  -- Always 2
  let hundreds_choices : ℕ := 8   -- Excluding 2 and the chosen units digit
  let tens_choices : ℕ := 7       -- Excluding 2, hundreds digit, and units digit
  let units_choices : ℕ := Finset.card odd_units

  -- Calculate the total count
  let total_count : ℕ := thousands_choices * hundreds_choices * tens_choices * units_choices

  -- Prove that the count equals 280
  sorry

-- The theorem statement
theorem odd_integers_three_different_digits_count :
  (odd_integers_count : ℕ) = 280 := by sorry

end NUMINAMATH_CALUDE_odd_integers_count_odd_integers_three_different_digits_count_l3668_366877


namespace NUMINAMATH_CALUDE_octahedron_volume_with_unit_inscribed_sphere_l3668_366822

/-- An octahedron is a polyhedron with 8 equilateral triangular faces. -/
structure Octahedron where
  -- We don't need to define the full structure, just what we need for this problem
  volume : ℝ

/-- A sphere is a three-dimensional geometric object. -/
structure Sphere where
  radius : ℝ

/-- An octahedron with an inscribed sphere. -/
structure OctahedronWithInscribedSphere where
  octahedron : Octahedron
  sphere : Sphere
  inscribed : sphere.radius = 1  -- The sphere is inscribed and has radius 1

/-- The volume of an octahedron with an inscribed sphere of radius 1 is √6. -/
theorem octahedron_volume_with_unit_inscribed_sphere
  (o : OctahedronWithInscribedSphere) :
  o.octahedron.volume = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_octahedron_volume_with_unit_inscribed_sphere_l3668_366822


namespace NUMINAMATH_CALUDE_green_home_construction_l3668_366826

theorem green_home_construction (x : ℝ) (h : x > 50) : (300 : ℝ) / (x - 50) = 400 / x := by
  sorry

end NUMINAMATH_CALUDE_green_home_construction_l3668_366826


namespace NUMINAMATH_CALUDE_parallel_lines_condition_l3668_366805

/-- Given two lines l₁ and l₂ defined by linear equations with parameter m,
    prove that m = -2 is a sufficient but not necessary condition for l₁ // l₂ -/
theorem parallel_lines_condition (m : ℝ) :
  let l₁ := {(x, y) : ℝ × ℝ | (m - 4) * x - (2 * m + 4) * y + 2 * m - 4 = 0}
  let l₂ := {(x, y) : ℝ × ℝ | (m - 1) * x + (m + 2) * y + 1 = 0}
  (m = -2 → l₁ = l₂) ∧ ¬(l₁ = l₂ → m = -2) :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_condition_l3668_366805


namespace NUMINAMATH_CALUDE_least_positive_integer_congruence_l3668_366821

theorem least_positive_integer_congruence :
  ∃! x : ℕ+, x.val + 3567 ≡ 1543 [ZMOD 14] ∧
  ∀ y : ℕ+, y.val + 3567 ≡ 1543 [ZMOD 14] → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_congruence_l3668_366821


namespace NUMINAMATH_CALUDE_thirteenth_square_vs_first_twelve_l3668_366876

def grains (k : ℕ) : ℕ := 2^k

def sum_grains (n : ℕ) : ℕ := (grains (n + 1)) - 2

theorem thirteenth_square_vs_first_twelve :
  grains 13 = sum_grains 12 + 2 := by
  sorry

end NUMINAMATH_CALUDE_thirteenth_square_vs_first_twelve_l3668_366876


namespace NUMINAMATH_CALUDE_absolute_value_equation_simplification_l3668_366880

theorem absolute_value_equation_simplification (a b c : ℝ) :
  (∀ x : ℝ, |5 * x - 4| + a ≠ 0) →
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ |4 * x₁ - 3| + b = 0 ∧ |4 * x₂ - 3| + b = 0) →
  (∃! x : ℝ, |3 * x - 2| + c = 0) →
  |a - c| + |c - b| - |a - b| = 0 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_equation_simplification_l3668_366880


namespace NUMINAMATH_CALUDE_area_of_triangle_with_given_conditions_l3668_366856

-- Define the triangle ABC and point P
structure Triangle :=
  (A B C : ℝ × ℝ)

structure TriangleWithPoint extends Triangle :=
  (P : ℝ × ℝ)

-- Define the conditions
def isScaleneRightTriangle (t : Triangle) : Prop := sorry

def isPointOnHypotenuse (t : TriangleWithPoint) : Prop := sorry

def angleABP30 (t : TriangleWithPoint) : Prop := sorry

def lengthAP3 (t : TriangleWithPoint) : Prop := sorry

def lengthCP1 (t : TriangleWithPoint) : Prop := sorry

-- Define the area function
def triangleArea (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem area_of_triangle_with_given_conditions (t : TriangleWithPoint) 
  (h1 : isScaleneRightTriangle t.toTriangle)
  (h2 : isPointOnHypotenuse t)
  (h3 : angleABP30 t)
  (h4 : lengthAP3 t)
  (h5 : lengthCP1 t) :
  triangleArea t.toTriangle = 12/5 := by sorry

end NUMINAMATH_CALUDE_area_of_triangle_with_given_conditions_l3668_366856


namespace NUMINAMATH_CALUDE_birth_probability_l3668_366819

-- Define the number of children
def n : ℕ := 5

-- Define the probability of a child being a boy or a girl
def p : ℚ := 1/2

-- Define the probability of all children being the same gender
def prob_all_same : ℚ := p^n

-- Define the probability of having 3 of one gender and 2 of the other
def prob_three_two : ℚ := Nat.choose n 3 * p^n

-- Define the probability of having 4 of one gender and 1 of the other
def prob_four_one : ℚ := 2 * Nat.choose n 1 * p^n

theorem birth_probability :
  prob_three_two > prob_all_same ∧
  prob_four_one > prob_all_same ∧
  prob_three_two = prob_four_one :=
by sorry

end NUMINAMATH_CALUDE_birth_probability_l3668_366819


namespace NUMINAMATH_CALUDE_x0_value_l3668_366833

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem x0_value (x₀ : ℝ) (h : x₀ > 0) :
  (deriv f x₀ = 3) → x₀ = Real.exp 2 := by
  sorry

end NUMINAMATH_CALUDE_x0_value_l3668_366833


namespace NUMINAMATH_CALUDE_parallelogram_not_symmetrical_l3668_366807

-- Define a type for shapes
inductive Shape
  | Circle
  | Rectangle
  | Parallelogram
  | IsoscelesTrapezoid

-- Define a property for symmetry
def isSymmetrical (s : Shape) : Prop :=
  match s with
  | Shape.Circle => True
  | Shape.Rectangle => True
  | Shape.Parallelogram => False
  | Shape.IsoscelesTrapezoid => True

-- Theorem statement
theorem parallelogram_not_symmetrical :
  ¬(isSymmetrical Shape.Parallelogram) :=
by
  sorry

#check parallelogram_not_symmetrical

end NUMINAMATH_CALUDE_parallelogram_not_symmetrical_l3668_366807


namespace NUMINAMATH_CALUDE_cos_range_theorem_l3668_366861

theorem cos_range_theorem (ω : ℝ) (h_ω : ω > 0) :
  (∀ x ∈ Set.Icc 0 (π / 3), 3 * Real.sin (ω * x) + 4 * Real.cos (ω * x) ∈ Set.Icc 4 5) →
  (∃ y ∈ Set.Icc (7 / 25) (4 / 5), y = Real.cos (π * ω / 3)) ∧
  (∀ y, y = Real.cos (π * ω / 3) → y ∈ Set.Icc (7 / 25) (4 / 5)) :=
by sorry

end NUMINAMATH_CALUDE_cos_range_theorem_l3668_366861


namespace NUMINAMATH_CALUDE_notebook_cost_l3668_366831

theorem notebook_cost (total_students : Nat) (buyers : Nat) (notebooks_per_student : Nat) (cost_per_notebook : Nat) :
  total_students = 42 →
  buyers > total_students / 2 →
  notebooks_per_student.Prime →
  cost_per_notebook > notebooks_per_student →
  buyers * notebooks_per_student * cost_per_notebook = 2310 →
  cost_per_notebook = 22 := by
  sorry

end NUMINAMATH_CALUDE_notebook_cost_l3668_366831


namespace NUMINAMATH_CALUDE_initial_juice_percentage_l3668_366825

/-- Proves that the initial percentage of pure fruit juice in a 2-liter mixture is 10% -/
theorem initial_juice_percentage :
  let initial_volume : ℝ := 2
  let added_juice : ℝ := 0.4
  let final_percentage : ℝ := 25
  let final_volume : ℝ := initial_volume + added_juice
  ∀ initial_percentage : ℝ,
    (initial_percentage / 100 * initial_volume + added_juice) / final_volume * 100 = final_percentage →
    initial_percentage = 10 := by
  sorry

end NUMINAMATH_CALUDE_initial_juice_percentage_l3668_366825


namespace NUMINAMATH_CALUDE_point_inside_ellipse_l3668_366870

theorem point_inside_ellipse (a : ℝ) : 
  (a^2 / 4 + 1 / 2 < 1) → (-Real.sqrt 2 < a ∧ a < Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_point_inside_ellipse_l3668_366870


namespace NUMINAMATH_CALUDE_task_completion_probability_l3668_366803

theorem task_completion_probability (p1 p2 : ℚ) (h1 : p1 = 2/3) (h2 : p2 = 3/5) :
  p1 * (1 - p2) = 4/15 := by
  sorry

end NUMINAMATH_CALUDE_task_completion_probability_l3668_366803


namespace NUMINAMATH_CALUDE_three_dice_not_one_or_six_l3668_366862

/-- The probability of a single die not showing 1 or 6 -/
def single_die_prob : ℚ := 4 / 6

/-- The number of dice tossed -/
def num_dice : ℕ := 3

/-- The probability that none of the three dice show 1 or 6 -/
def three_dice_prob : ℚ := single_die_prob ^ num_dice

theorem three_dice_not_one_or_six :
  three_dice_prob = 8 / 27 := by
  sorry

end NUMINAMATH_CALUDE_three_dice_not_one_or_six_l3668_366862


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l3668_366829

-- Define the sets A and B
def A : Set ℝ := {x | -3 < x ∧ x < 3}
def B : Set ℝ := {x | x < -2}

-- State the theorem
theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B) = {x : ℝ | -2 ≤ x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l3668_366829


namespace NUMINAMATH_CALUDE_three_speakers_from_different_companies_l3668_366886

/-- The number of companies -/
def num_companies : ℕ := 5

/-- The number of representatives for Company A -/
def company_a_reps : ℕ := 2

/-- The number of representatives for each of the other companies -/
def other_company_reps : ℕ := 1

/-- The number of speakers at the meeting -/
def num_speakers : ℕ := 3

/-- The number of ways to select 3 speakers from 3 different companies -/
def num_ways : ℕ := 16

theorem three_speakers_from_different_companies :
  let total_reps := company_a_reps + (num_companies - 1) * other_company_reps
  (Nat.choose total_reps num_speakers) = num_ways := by sorry

end NUMINAMATH_CALUDE_three_speakers_from_different_companies_l3668_366886


namespace NUMINAMATH_CALUDE_sara_movie_expenses_l3668_366855

/-- The total amount Sara spent on movies -/
def total_spent (ticket_price : ℚ) (num_tickets : ℕ) (rental_cost : ℚ) (purchase_cost : ℚ) : ℚ :=
  ticket_price * num_tickets + rental_cost + purchase_cost

/-- Proof that Sara spent $36.78 on movies -/
theorem sara_movie_expenses : 
  total_spent 10.62 2 1.59 13.95 = 36.78 := by
  sorry

end NUMINAMATH_CALUDE_sara_movie_expenses_l3668_366855


namespace NUMINAMATH_CALUDE_color_change_probability_l3668_366873

/-- Represents the duration of each color in the traffic light cycle -/
structure TrafficLightCycle where
  green : ℕ
  yellow : ℕ
  red : ℕ

/-- Calculates the total cycle duration -/
def cycleDuration (cycle : TrafficLightCycle) : ℕ :=
  cycle.green + cycle.yellow + cycle.red

/-- Calculates the number of seconds where a color change can be observed -/
def changeObservationDuration (cycle : TrafficLightCycle) (observationInterval : ℕ) : ℕ :=
  3 * observationInterval  -- 3 color changes per cycle

/-- Theorem: The probability of observing a color change is 12/85 -/
theorem color_change_probability (cycle : TrafficLightCycle) 
  (h1 : cycle.green = 45)
  (h2 : cycle.yellow = 5)
  (h3 : cycle.red = 35)
  (observationInterval : ℕ)
  (h4 : observationInterval = 4) :
  (changeObservationDuration cycle observationInterval : ℚ) / 
  (cycleDuration cycle : ℚ) = 12 / 85 := by
  sorry

end NUMINAMATH_CALUDE_color_change_probability_l3668_366873


namespace NUMINAMATH_CALUDE_intersection_point_l3668_366816

/-- The line equation -/
def line_equation (x y z : ℝ) : Prop :=
  (x - 1) / 2 = (y - 1) / (-1) ∧ (y - 1) / (-1) = (z + 2) / 3

/-- The plane equation -/
def plane_equation (x y z : ℝ) : Prop :=
  4 * x + 2 * y - z - 11 = 0

/-- The theorem stating that (3, 0, 1) is the unique intersection point -/
theorem intersection_point :
  ∃! (x y z : ℝ), line_equation x y z ∧ plane_equation x y z ∧ x = 3 ∧ y = 0 ∧ z = 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l3668_366816


namespace NUMINAMATH_CALUDE_pet_store_choices_l3668_366872

def num_puppies : ℕ := 10
def num_kittens : ℕ := 8
def num_bunnies : ℕ := 12

def alice_choices : ℕ := num_kittens + num_bunnies
def bob_choices (alice_choice : ℕ) : ℕ :=
  if alice_choice ≤ num_kittens then num_puppies + num_bunnies
  else num_puppies + (num_bunnies - 1)
def charlie_choices (alice_choice bob_choice : ℕ) : ℕ :=
  num_puppies + num_kittens + num_bunnies - alice_choice - bob_choice

def total_choices : ℕ :=
  (alice_choices * bob_choices num_kittens * charlie_choices num_kittens num_puppies) +
  (alice_choices * bob_choices num_kittens * charlie_choices num_kittens num_bunnies) +
  (alice_choices * bob_choices num_bunnies * charlie_choices num_bunnies num_puppies) +
  (alice_choices * bob_choices num_bunnies * charlie_choices num_bunnies (num_bunnies - 1))

theorem pet_store_choices : total_choices = 4120 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_choices_l3668_366872


namespace NUMINAMATH_CALUDE_number_of_divisors_sum_of_divisors_l3668_366871

def n : ℕ := 2310

-- Function to count positive divisors
def count_divisors (m : ℕ) : ℕ := sorry

-- Function to sum positive divisors
def sum_divisors (m : ℕ) : ℕ := sorry

-- Theorem stating the number of positive divisors of 2310
theorem number_of_divisors : count_divisors n = 32 := by sorry

-- Theorem stating the sum of positive divisors of 2310
theorem sum_of_divisors : sum_divisors n = 6912 := by sorry

end NUMINAMATH_CALUDE_number_of_divisors_sum_of_divisors_l3668_366871


namespace NUMINAMATH_CALUDE_M_is_range_of_f_l3668_366893

def M : Set ℝ := {y | ∃ x, y = x^2}

def f (x : ℝ) : ℝ := x^2

theorem M_is_range_of_f : M = Set.range f := by
  sorry

end NUMINAMATH_CALUDE_M_is_range_of_f_l3668_366893


namespace NUMINAMATH_CALUDE_subtract_from_zero_l3668_366889

theorem subtract_from_zero (x : ℚ) : 0 - x = -x := by sorry

end NUMINAMATH_CALUDE_subtract_from_zero_l3668_366889


namespace NUMINAMATH_CALUDE_stamp_costs_l3668_366867

theorem stamp_costs (a b c d : ℝ) : 
  a + b + c + d = 84 →                   -- sum is 84
  b - a = c - b ∧ c - b = d - c →        -- arithmetic progression
  d = 2.5 * a →                          -- largest is 2.5 times smallest
  a = 12 ∧ b = 18 ∧ c = 24 ∧ d = 30 :=   -- prove the values
by sorry

end NUMINAMATH_CALUDE_stamp_costs_l3668_366867


namespace NUMINAMATH_CALUDE_basket_balls_count_l3668_366839

/-- Given a basket of balls where the ratio of white to red balls is 5:3 and there are 15 white balls, prove that there are 9 red balls. -/
theorem basket_balls_count (white_balls : ℕ) (red_balls : ℕ) : 
  (white_balls : ℚ) / red_balls = 5 / 3 → white_balls = 15 → red_balls = 9 := by
  sorry

end NUMINAMATH_CALUDE_basket_balls_count_l3668_366839


namespace NUMINAMATH_CALUDE_min_bilingual_students_l3668_366879

theorem min_bilingual_students (total : ℕ) (hindi : ℕ) (english : ℕ) 
  (h_total : total = 40)
  (h_hindi : hindi = 30)
  (h_english : english = 20) :
  ∃ (both : ℕ), both ≥ hindi + english - total ∧ 
  (∀ (x : ℕ), x ≥ hindi + english - total → x ≥ both) :=
by sorry

end NUMINAMATH_CALUDE_min_bilingual_students_l3668_366879


namespace NUMINAMATH_CALUDE_dilation_rotation_theorem_l3668_366802

/-- The matrix representing a dilation by scale factor 4 centered at the origin -/
def dilation_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![4, 0; 0, 4]

/-- The matrix representing a 90-degree counterclockwise rotation -/
def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![0, -1; 1, 0]

/-- The combined transformation matrix -/
def combined_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![0, -4; 4, 0]

theorem dilation_rotation_theorem :
  rotation_matrix * dilation_matrix = combined_matrix :=
sorry

end NUMINAMATH_CALUDE_dilation_rotation_theorem_l3668_366802


namespace NUMINAMATH_CALUDE_marble_bag_count_l3668_366808

/-- Given a bag of marbles with red, blue, and green marbles in the ratio 2:4:6,
    and 36 green marbles, prove that the total number of marbles is 72. -/
theorem marble_bag_count (red blue green total : ℕ) : 
  red + blue + green = total →
  2 * blue = 4 * red →
  3 * blue = 2 * green →
  green = 36 →
  total = 72 := by
sorry

end NUMINAMATH_CALUDE_marble_bag_count_l3668_366808


namespace NUMINAMATH_CALUDE_article_count_l3668_366865

theorem article_count (x : ℕ) (cost_price selling_price : ℝ) : 
  (cost_price * x = selling_price * 16) →
  (selling_price = 1.5 * cost_price) →
  x = 24 := by
sorry

end NUMINAMATH_CALUDE_article_count_l3668_366865


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3668_366832

/-- Given a geometric sequence {aₙ} with positive terms where a₁a₅ + 2a₃a₅ + a₃a₇ = 25, prove that a₃ + a₅ = 5. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h1 : ∀ n, a n > 0) 
    (h2 : ∃ r : ℝ, ∀ n, a (n + 1) = r * a n) 
    (h3 : a 1 * a 5 + 2 * a 3 * a 5 + a 3 * a 7 = 25) : 
  a 3 + a 5 = 5 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3668_366832


namespace NUMINAMATH_CALUDE_least_possible_lcm_a_c_l3668_366847

theorem least_possible_lcm_a_c (a b c : ℕ) 
  (h1 : Nat.lcm a b = 18) 
  (h2 : Nat.lcm b c = 20) : 
  ∃ (a' c' : ℕ), Nat.lcm a' c' = 90 ∧ 
    (∀ (x y : ℕ), Nat.lcm x b = 18 → Nat.lcm b y = 20 → Nat.lcm a' c' ≤ Nat.lcm x y) := by
  sorry

end NUMINAMATH_CALUDE_least_possible_lcm_a_c_l3668_366847


namespace NUMINAMATH_CALUDE_walking_problem_l3668_366843

/-- The walking problem theorem -/
theorem walking_problem (total_distance : ℝ) (yolanda_rate : ℝ) (bob_distance : ℝ) : 
  total_distance = 24 →
  yolanda_rate = 3 →
  bob_distance = 12 →
  ∃ (bob_rate : ℝ), bob_rate = 12 ∧ bob_distance = bob_rate * 1 := by
  sorry

end NUMINAMATH_CALUDE_walking_problem_l3668_366843


namespace NUMINAMATH_CALUDE_three_digit_power_ending_theorem_l3668_366818

/-- A three-digit number is between 100 and 999, inclusive. -/
def ThreeDigitNumber (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- A number N satisfies the property if for all k ≥ 1, N^k ≡ N (mod 1000) -/
def SatisfiesProperty (N : ℕ) : Prop :=
  ∀ k : ℕ, k ≥ 1 → N^k ≡ N [MOD 1000]

theorem three_digit_power_ending_theorem :
  ∀ N : ℕ, ThreeDigitNumber N → SatisfiesProperty N ↔ (N = 625 ∨ N = 376) :=
sorry

end NUMINAMATH_CALUDE_three_digit_power_ending_theorem_l3668_366818


namespace NUMINAMATH_CALUDE_union_A_complement_B_equals_interval_l3668_366811

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x < 2}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = x^2 + 1}

-- State the theorem
theorem union_A_complement_B_equals_interval :
  A ∪ (U \ B) = Set.Iio 2 := by sorry

end NUMINAMATH_CALUDE_union_A_complement_B_equals_interval_l3668_366811


namespace NUMINAMATH_CALUDE_expected_full_circles_l3668_366845

/-- Represents the tiling of an equilateral triangle -/
structure TriangleTiling where
  n : ℕ
  sideLength : n > 2

/-- Expected number of full circles in a triangle tiling -/
def expectedFullCircles (t : TriangleTiling) : ℚ :=
  (t.n - 2) * (t.n - 1) / 1458

/-- Theorem stating the expected number of full circles in a triangle tiling -/
theorem expected_full_circles (t : TriangleTiling) :
  expectedFullCircles t = (t.n - 2) * (t.n - 1) / 1458 :=
by sorry

end NUMINAMATH_CALUDE_expected_full_circles_l3668_366845


namespace NUMINAMATH_CALUDE_root_sum_reciprocals_l3668_366836

-- Define the polynomial
def f (x : ℂ) : ℂ := x^4 + 10*x^3 + 20*x^2 + 15*x + 6

-- Define the roots
def p : ℂ := sorry
def q : ℂ := sorry
def r : ℂ := sorry
def s : ℂ := sorry

-- State the theorem
theorem root_sum_reciprocals :
  f p = 0 ∧ f q = 0 ∧ f r = 0 ∧ f s = 0 →
  1 / (p * q) + 1 / (p * r) + 1 / (p * s) + 1 / (q * r) + 1 / (q * s) + 1 / (r * s) = 10 / 3 :=
by sorry

end NUMINAMATH_CALUDE_root_sum_reciprocals_l3668_366836
