import Mathlib

namespace NUMINAMATH_CALUDE_parrot_count_l3206_320674

/-- Represents the number of animals in a zoo --/
structure ZooCount where
  parrots : ℕ
  snakes : ℕ
  monkeys : ℕ
  elephants : ℕ
  zebras : ℕ

/-- Checks if the zoo count satisfies the given conditions --/
def isValidZooCount (z : ZooCount) : Prop :=
  z.snakes = 3 * z.parrots ∧
  z.monkeys = 2 * z.snakes ∧
  z.elephants = (z.parrots + z.snakes) / 2 ∧
  z.zebras = z.elephants - 3 ∧
  z.monkeys - z.zebras = 35

/-- Theorem stating that there are 8 parrots in the zoo --/
theorem parrot_count : ∃ z : ZooCount, isValidZooCount z ∧ z.parrots = 8 := by
  sorry

end NUMINAMATH_CALUDE_parrot_count_l3206_320674


namespace NUMINAMATH_CALUDE_maths_fraction_in_class_l3206_320660

theorem maths_fraction_in_class (total_students : ℕ) 
  (maths_and_history_students : ℕ) :
  total_students = 25 →
  maths_and_history_students = 20 →
  ∃ (maths_fraction : ℚ),
    maths_fraction * total_students +
    (1 / 3 : ℚ) * (total_students - maths_fraction * total_students) +
    (total_students - maths_fraction * total_students - 
     (1 / 3 : ℚ) * (total_students - maths_fraction * total_students)) = total_students ∧
    maths_fraction * total_students +
    (total_students - maths_fraction * total_students - 
     (1 / 3 : ℚ) * (total_students - maths_fraction * total_students)) = maths_and_history_students ∧
    maths_fraction = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_maths_fraction_in_class_l3206_320660


namespace NUMINAMATH_CALUDE_third_group_frequency_count_l3206_320664

theorem third_group_frequency_count :
  ∀ (n₁ n₂ n₃ n₄ n₅ : ℕ),
  n₁ + n₂ + n₃ = 160 →
  n₃ + n₄ + n₅ = 260 →
  (n₃ : ℝ) / (n₁ + n₂ + n₃ + n₄ + n₅ : ℝ) = 0.20 →
  n₃ = 70 :=
by sorry

end NUMINAMATH_CALUDE_third_group_frequency_count_l3206_320664


namespace NUMINAMATH_CALUDE_smallest_gcd_value_l3206_320652

theorem smallest_gcd_value (a b c d : ℕ) : 
  (∃ (gcd_list : List ℕ), 
    gcd_list.length = 6 ∧ 
    1 ∈ gcd_list ∧ 
    2 ∈ gcd_list ∧ 
    3 ∈ gcd_list ∧ 
    4 ∈ gcd_list ∧ 
    5 ∈ gcd_list ∧
    (∃ (N : ℕ), N > 5 ∧ N ∈ gcd_list) ∧
    (∀ (x : ℕ), x ∈ gcd_list → 
      x = Nat.gcd a b ∨ 
      x = Nat.gcd a c ∨ 
      x = Nat.gcd a d ∨ 
      x = Nat.gcd b c ∨ 
      x = Nat.gcd b d ∨ 
      x = Nat.gcd c d)) →
  (∀ (M : ℕ), M > 5 ∧ 
    (∃ (gcd_list : List ℕ), 
      gcd_list.length = 6 ∧ 
      1 ∈ gcd_list ∧ 
      2 ∈ gcd_list ∧ 
      3 ∈ gcd_list ∧ 
      4 ∈ gcd_list ∧ 
      5 ∈ gcd_list ∧
      M ∈ gcd_list ∧
      (∀ (x : ℕ), x ∈ gcd_list → 
        x = Nat.gcd a b ∨ 
        x = Nat.gcd a c ∨ 
        x = Nat.gcd a d ∨ 
        x = Nat.gcd b c ∨ 
        x = Nat.gcd b d ∨ 
        x = Nat.gcd c d)) →
    M ≥ 14) :=
by sorry

end NUMINAMATH_CALUDE_smallest_gcd_value_l3206_320652


namespace NUMINAMATH_CALUDE_complex_expression_equality_l3206_320691

theorem complex_expression_equality : 
  (2 + 7/9)^(1/2 : ℝ) + (1/10)^(-2 : ℝ) + (2 + 10/27)^(-(2/3) : ℝ) - Real.pi^(0 : ℝ) + 37/48 = 807/8 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l3206_320691


namespace NUMINAMATH_CALUDE_convenience_store_syrup_cost_l3206_320623

/-- Calculates the weekly syrup cost for a convenience store. -/
def weekly_syrup_cost (soda_sold : ℕ) (gallons_per_box : ℕ) (cost_per_box : ℕ) : ℕ :=
  (soda_sold / gallons_per_box) * cost_per_box

/-- Theorem stating the weekly syrup cost for the given conditions. -/
theorem convenience_store_syrup_cost :
  weekly_syrup_cost 180 30 40 = 240 := by
  sorry

end NUMINAMATH_CALUDE_convenience_store_syrup_cost_l3206_320623


namespace NUMINAMATH_CALUDE_round_37_259_to_thousandth_l3206_320645

/-- Represents a repeating decimal with a whole number part and a repeating fractional part. -/
structure RepeatingDecimal where
  wholePart : ℕ
  repeatingPart : ℕ

/-- Rounds a RepeatingDecimal to the nearest thousandth. -/
def roundToThousandth (x : RepeatingDecimal) : ℚ :=
  sorry

/-- The repeating decimal 37.259259... -/
def num : RepeatingDecimal :=
  { wholePart := 37, repeatingPart := 259 }

theorem round_37_259_to_thousandth :
  roundToThousandth num = 37259 / 1000 :=
sorry

end NUMINAMATH_CALUDE_round_37_259_to_thousandth_l3206_320645


namespace NUMINAMATH_CALUDE_plant_branches_problem_l3206_320699

theorem plant_branches_problem (x : ℕ) : 
  (1 + x + x^2 = 43) → (x = 6) :=
by
  sorry

end NUMINAMATH_CALUDE_plant_branches_problem_l3206_320699


namespace NUMINAMATH_CALUDE_parabola_focus_at_hyperbola_vertex_l3206_320656

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

-- Define the right vertex of the hyperbola
def right_vertex (x y : ℝ) : Prop := hyperbola x y ∧ y = 0 ∧ x > 0

-- Define the standard form of a parabola
def parabola (x y p : ℝ) : Prop := y^2 = 2 * p * x

-- Theorem statement
theorem parabola_focus_at_hyperbola_vertex :
  ∃ (x₀ y₀ : ℝ), right_vertex x₀ y₀ →
  ∃ (p : ℝ), p > 0 ∧ ∀ (x y : ℝ), parabola (x - x₀) y p ↔ y^2 = 16 * x :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_at_hyperbola_vertex_l3206_320656


namespace NUMINAMATH_CALUDE_todd_snow_cone_profit_l3206_320653

/-- Calculates Todd's profit from his snow-cone stand business --/
theorem todd_snow_cone_profit :
  let loan := 300
  let repayment := 330
  let equipment_cost := 120
  let initial_ingredient_cost := 60
  let marketing_cost := 40
  let misc_cost := 10
  let snow_cone_price := 1.75
  let snow_cone_sales := 500
  let custom_cup_price := 2
  let custom_cup_sales := 250
  let ingredient_cost_increase_rate := 0.2
  let snow_cones_before_increase := 300

  let total_initial_expenses := equipment_cost + initial_ingredient_cost + marketing_cost + misc_cost + repayment
  let total_revenue := snow_cone_price * snow_cone_sales + custom_cup_price * custom_cup_sales
  let increased_ingredient_cost := initial_ingredient_cost * ingredient_cost_increase_rate
  let snow_cones_after_increase := snow_cone_sales - snow_cones_before_increase
  let total_expenses := total_initial_expenses + increased_ingredient_cost

  let profit := total_revenue - total_expenses

  profit = 803 := by sorry

end NUMINAMATH_CALUDE_todd_snow_cone_profit_l3206_320653


namespace NUMINAMATH_CALUDE_two_visitors_theorem_l3206_320696

/-- Represents a friend's visiting pattern -/
structure VisitPattern where
  period : ℕ
  start : ℕ

/-- Calculates the number of days with exactly two visitors -/
def exactTwoVisitors (alice beatrix claire : VisitPattern) (totalDays : ℕ) : ℕ :=
  sorry

theorem two_visitors_theorem (alice beatrix claire : VisitPattern) :
  alice.period = 2 ∧ alice.start = 1 ∧
  beatrix.period = 6 ∧ beatrix.start = 2 ∧
  claire.period = 5 ∧ claire.start = 2 →
  exactTwoVisitors alice beatrix claire 400 = 80 := by
  sorry

end NUMINAMATH_CALUDE_two_visitors_theorem_l3206_320696


namespace NUMINAMATH_CALUDE_product_of_differences_l3206_320604

theorem product_of_differences (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 2005) (h₂ : y₁^3 - 3*x₁^2*y₁ = 2004)
  (h₃ : x₂^3 - 3*x₂*y₂^2 = 2005) (h₄ : y₂^3 - 3*x₂^2*y₂ = 2004)
  (h₅ : x₃^3 - 3*x₃*y₃^2 = 2005) (h₆ : y₃^3 - 3*x₃^2*y₃ = 2004) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = 1/1002 := by
  sorry

end NUMINAMATH_CALUDE_product_of_differences_l3206_320604


namespace NUMINAMATH_CALUDE_equation_has_three_solutions_l3206_320684

-- Define the complex polynomial in the numerator
def numerator (z : ℂ) : ℂ := z^4 - 1

-- Define the complex polynomial in the denominator
def denominator (z : ℂ) : ℂ := z^3 - 3*z + 2

-- Define the equation
def equation (z : ℂ) : Prop := numerator z = 0 ∧ denominator z ≠ 0

-- Theorem statement
theorem equation_has_three_solutions :
  ∃ (s : Finset ℂ), s.card = 3 ∧ (∀ z ∈ s, equation z) ∧ (∀ z, equation z → z ∈ s) :=
sorry

end NUMINAMATH_CALUDE_equation_has_three_solutions_l3206_320684


namespace NUMINAMATH_CALUDE_difference_divisible_by_18_l3206_320672

theorem difference_divisible_by_18 (a b : ℤ) : 
  18 ∣ ((3*a + 2)^2 - (3*b + 2)^2) := by sorry

end NUMINAMATH_CALUDE_difference_divisible_by_18_l3206_320672


namespace NUMINAMATH_CALUDE_product_of_fractions_equals_negative_one_l3206_320637

theorem product_of_fractions_equals_negative_one 
  (x y z : ℝ) 
  (hx : x ≠ 3) 
  (hy : y ≠ 5) 
  (hz : z ≠ 7) : 
  ((x - 3) / (7 - z)) * ((y - 5) / (3 - x)) * ((z - 7) / (5 - y)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_equals_negative_one_l3206_320637


namespace NUMINAMATH_CALUDE_integer_sum_problem_l3206_320608

theorem integer_sum_problem (x y : ℤ) : 
  x > 0 → y > 0 → x - y = 8 → x * y = 120 → x + y = 4 * Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_integer_sum_problem_l3206_320608


namespace NUMINAMATH_CALUDE_exam_results_l3206_320693

theorem exam_results (total : ℕ) (failed_hindi : ℕ) (failed_english : ℕ) (failed_both : ℕ)
  (h1 : failed_hindi = total / 4)
  (h2 : failed_english = total / 2)
  (h3 : failed_both = total / 4)
  : (total - (failed_hindi + failed_english - failed_both)) = total / 2 := by
  sorry

end NUMINAMATH_CALUDE_exam_results_l3206_320693


namespace NUMINAMATH_CALUDE_tabitha_honey_nights_l3206_320640

/-- Calculates the number of nights Tabitha can enjoy honey in her tea before running out. -/
def honey_nights (servings_per_cup : ℕ) (cups_per_night : ℕ) (container_size : ℕ) (servings_per_ounce : ℕ) : ℕ :=
  let total_servings := container_size * servings_per_ounce
  let servings_per_night := servings_per_cup * cups_per_night
  total_servings / servings_per_night

/-- Proves that Tabitha can enjoy honey in her tea for 48 nights before running out. -/
theorem tabitha_honey_nights : 
  honey_nights 1 2 16 6 = 48 := by
  sorry

end NUMINAMATH_CALUDE_tabitha_honey_nights_l3206_320640


namespace NUMINAMATH_CALUDE_min_type1_figures_l3206_320673

/-- The side length of the equilateral triangle T -/
def side_length : ℕ := 2022

/-- The total number of unit triangles in T -/
def total_triangles : ℕ := side_length * (side_length + 1) / 2

/-- The number of upward-pointing unit triangles in T -/
def upward_triangles : ℕ := (total_triangles + side_length) / 2

/-- The number of downward-pointing unit triangles in T -/
def downward_triangles : ℕ := (total_triangles - side_length) / 2

/-- The excess of upward-pointing unit triangles -/
def excess_upward : ℕ := upward_triangles - downward_triangles

/-- A figure consisting of 4 equilateral unit triangles -/
inductive Figure
| Type1 : Figure  -- Has an excess of ±2 upward-pointing unit triangles
| Type2 : Figure  -- Has equal number of upward and downward-pointing unit triangles
| Type3 : Figure  -- Has equal number of upward and downward-pointing unit triangles
| Type4 : Figure  -- Has equal number of upward and downward-pointing unit triangles

/-- A covering of the triangle T with figures -/
def Covering := List Figure

/-- Predicate to check if a covering is valid -/
def is_valid_covering (c : Covering) : Prop := sorry

/-- The number of Type1 figures in a covering -/
def count_type1 (c : Covering) : ℕ := sorry

theorem min_type1_figures :
  ∃ (c : Covering), is_valid_covering c ∧
  count_type1 c = 1011 ∧
  ∀ (c' : Covering), is_valid_covering c' → count_type1 c' ≥ 1011 := by sorry

end NUMINAMATH_CALUDE_min_type1_figures_l3206_320673


namespace NUMINAMATH_CALUDE_simplify_expression_l3206_320606

theorem simplify_expression (x : ℝ) : 3*x + 6*x + 9*x + 12*x + 15*x + 18 + 9 = 45*x + 27 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3206_320606


namespace NUMINAMATH_CALUDE_equation_root_range_l3206_320689

theorem equation_root_range (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ x^2 - a*x + a^2 - 3 = 0) → 
  -Real.sqrt 3 < a ∧ a ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_equation_root_range_l3206_320689


namespace NUMINAMATH_CALUDE_f_composed_three_roots_l3206_320658

/-- A quadratic function f(x) = x^2 + 6x + c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 6*x + c

/-- The composition of f with itself -/
def f_composed (c : ℝ) (x : ℝ) : ℝ := f c (f c x)

/-- Predicate to check if a function has exactly 3 distinct real roots -/
def has_three_distinct_real_roots (g : ℝ → ℝ) : Prop :=
  ∃ (r₁ r₂ r₃ : ℝ), (∀ x : ℝ, g x = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) ∧ r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃

theorem f_composed_three_roots :
  ∀ c : ℝ, has_three_distinct_real_roots (f_composed c) ↔ c = (11 - Real.sqrt 13) / 2 :=
sorry

end NUMINAMATH_CALUDE_f_composed_three_roots_l3206_320658


namespace NUMINAMATH_CALUDE_function_properties_l3206_320649

/-- The function f(x) = x^2 - 6x + 10 -/
def f (x : ℝ) : ℝ := x^2 - 6*x + 10

/-- The interval [2, 5) -/
def I : Set ℝ := Set.Icc 2 5

theorem function_properties :
  (∃ (m : ℝ), m = 1 ∧ ∀ x ∈ I, f x ≥ m) ∧
  (¬∃ (M : ℝ), ∀ x ∈ I, f x ≤ M) :=
sorry

end NUMINAMATH_CALUDE_function_properties_l3206_320649


namespace NUMINAMATH_CALUDE_tourist_count_l3206_320648

theorem tourist_count : 
  ∃ (n : ℕ), 
    (1/2 : ℚ) * n + (1/3 : ℚ) * n + (1/4 : ℚ) * n = 39 ∧ 
    n = 36 := by
  sorry

end NUMINAMATH_CALUDE_tourist_count_l3206_320648


namespace NUMINAMATH_CALUDE_exists_unresolved_conjecture_l3206_320667

/-- A structure representing a mathematical conjecture -/
structure Conjecture where
  statement : Prop
  is_proven : Prop
  is_disproven : Prop

/-- A predicate that determines if a conjecture is unresolved -/
def is_unresolved (c : Conjecture) : Prop :=
  ¬c.is_proven ∧ ¬c.is_disproven

/-- There exists at least one unresolved conjecture in mathematics -/
theorem exists_unresolved_conjecture : ∃ c : Conjecture, is_unresolved c := by
  sorry

#check exists_unresolved_conjecture

end NUMINAMATH_CALUDE_exists_unresolved_conjecture_l3206_320667


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3206_320616

theorem quadratic_inequality (x : ℝ) : x^2 + 3*x < 10 ↔ -5 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3206_320616


namespace NUMINAMATH_CALUDE_largest_common_divisor_problem_l3206_320614

theorem largest_common_divisor_problem : Nat.gcd (69 - 5) (86 - 6) = 16 := by
  sorry

end NUMINAMATH_CALUDE_largest_common_divisor_problem_l3206_320614


namespace NUMINAMATH_CALUDE_min_diagonal_of_rectangle_l3206_320646

/-- Given a rectangle ABCD with perimeter 24 inches, 
    the minimum possible length of its diagonal AC is 6√2 inches. -/
theorem min_diagonal_of_rectangle (l w : ℝ) : 
  (l + w = 12) →  -- perimeter condition: 2l + 2w = 24, simplified
  ∃ (d : ℝ), d = Real.sqrt (l^2 + w^2) ∧ 
              d ≥ 6 * Real.sqrt 2 ∧
              (∀ l' w' : ℝ, l' + w' = 12 → 
                Real.sqrt (l'^2 + w'^2) ≥ 6 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_min_diagonal_of_rectangle_l3206_320646


namespace NUMINAMATH_CALUDE_shoes_theorem_l3206_320609

/-- The number of pairs of shoes Ellie, Riley, and Jordan have in total -/
def total_shoes (ellie riley jordan : ℕ) : ℕ := ellie + riley + jordan

/-- The theorem stating the total number of shoes given the conditions -/
theorem shoes_theorem (ellie riley jordan : ℕ) 
  (h1 : ellie = 8)
  (h2 : riley = ellie - 3)
  (h3 : jordan = ((ellie + riley) * 3) / 2) :
  total_shoes ellie riley jordan = 32 := by
  sorry

end NUMINAMATH_CALUDE_shoes_theorem_l3206_320609


namespace NUMINAMATH_CALUDE_walking_time_calculation_l3206_320601

theorem walking_time_calculation (walking_speed run_speed : ℝ) (run_time : ℝ) (h1 : walking_speed = 5) (h2 : run_speed = 15) (h3 : run_time = 36 / 60) :
  let distance := run_speed * run_time
  walking_speed * (distance / walking_speed) = 1.8 := by sorry

end NUMINAMATH_CALUDE_walking_time_calculation_l3206_320601


namespace NUMINAMATH_CALUDE_fraction_sum_equals_point_four_l3206_320633

theorem fraction_sum_equals_point_four :
  2 / 20 + 3 / 30 + 4 / 40 + 5 / 50 = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_point_four_l3206_320633


namespace NUMINAMATH_CALUDE_inequality_range_l3206_320638

theorem inequality_range (a : ℝ) : 
  (∀ x > a, 2 * x + 1 / (x - a) ≥ 2 * Real.sqrt 2) ↔ a ≥ 0 := by sorry

end NUMINAMATH_CALUDE_inequality_range_l3206_320638


namespace NUMINAMATH_CALUDE_technician_avg_salary_l3206_320644

def total_workers : ℕ := 24
def avg_salary_all : ℕ := 8000
def num_technicians : ℕ := 8
def avg_salary_non_tech : ℕ := 6000

theorem technician_avg_salary :
  let num_non_tech := total_workers - num_technicians
  let total_salary := avg_salary_all * total_workers
  let total_salary_non_tech := avg_salary_non_tech * num_non_tech
  let total_salary_tech := total_salary - total_salary_non_tech
  total_salary_tech / num_technicians = 12000 := by
  sorry

end NUMINAMATH_CALUDE_technician_avg_salary_l3206_320644


namespace NUMINAMATH_CALUDE_pen_price_calculation_l3206_320615

/-- Given the total number of pens purchased -/
def num_pens : ℕ := 30

/-- Given the total number of pencils purchased -/
def num_pencils : ℕ := 75

/-- Given the total cost of pens and pencils -/
def total_cost : ℝ := 750

/-- Given the average price of a pencil -/
def avg_price_pencil : ℝ := 2

/-- The average price of a pen -/
def avg_price_pen : ℝ := 20

theorem pen_price_calculation :
  (num_pens : ℝ) * avg_price_pen + (num_pencils : ℝ) * avg_price_pencil = total_cost :=
by sorry

end NUMINAMATH_CALUDE_pen_price_calculation_l3206_320615


namespace NUMINAMATH_CALUDE_largest_expression_l3206_320630

def expr_a : ℕ := 2 + 3 + 1 + 7
def expr_b : ℕ := 2 * 3 + 1 + 7
def expr_c : ℕ := 2 + 3 * 1 + 7
def expr_d : ℕ := 2 + 3 + 1 * 7
def expr_e : ℕ := 2 * 3 * 1 * 7

theorem largest_expression : 
  expr_e > expr_a ∧ 
  expr_e > expr_b ∧ 
  expr_e > expr_c ∧ 
  expr_e > expr_d := by
  sorry

end NUMINAMATH_CALUDE_largest_expression_l3206_320630


namespace NUMINAMATH_CALUDE_discounted_soda_price_l3206_320600

/-- Calculate the price of discounted soda cans -/
theorem discounted_soda_price
  (regular_price : ℝ)
  (discount_percent : ℝ)
  (num_cans : ℕ)
  (h1 : regular_price = 0.60)
  (h2 : discount_percent = 20)
  (h3 : num_cans = 72) :
  let discounted_price := regular_price * (1 - discount_percent / 100)
  num_cans * discounted_price = 34.56 :=
by sorry

end NUMINAMATH_CALUDE_discounted_soda_price_l3206_320600


namespace NUMINAMATH_CALUDE_three_lines_intersection_l3206_320675

-- Define the lines
def line1 (x y : ℝ) := x - y + 1 = 0
def line2 (x y : ℝ) := 2*x + y - 4 = 0
def line3 (a x y : ℝ) := a*x - y + 2 = 0

-- Define the condition of exactly two intersection points
def has_two_intersections (a : ℝ) :=
  ∃ (x1 y1 x2 y2 : ℝ), 
    (line1 x1 y1 ∧ line2 x1 y1 ∧ line3 a x1 y1) ∧
    (line1 x2 y2 ∧ line2 x2 y2 ∧ line3 a x2 y2) ∧
    (x1 ≠ x2 ∨ y1 ≠ y2) ∧
    (∀ x y, line1 x y ∧ line2 x y ∧ line3 a x y → (x = x1 ∧ y = y1) ∨ (x = x2 ∧ y = y2))

-- Theorem statement
theorem three_lines_intersection (a : ℝ) :
  has_two_intersections a → a = 1 ∨ a = -2 :=
by sorry

end NUMINAMATH_CALUDE_three_lines_intersection_l3206_320675


namespace NUMINAMATH_CALUDE_roller_coaster_probability_l3206_320602

/-- The number of cars in the roller coaster -/
def num_cars : ℕ := 5

/-- The number of times the passenger rides the roller coaster -/
def num_rides : ℕ := 5

/-- The probability of riding in a specific car on a single ride -/
def prob_single_car : ℚ := 1 / num_cars

/-- The probability of riding in each of the cars over the given number of rides -/
def prob_all_cars : ℚ := (num_cars.factorial : ℚ) / num_cars ^ num_rides

theorem roller_coaster_probability :
  prob_all_cars = 24 / 625 := by
  sorry

end NUMINAMATH_CALUDE_roller_coaster_probability_l3206_320602


namespace NUMINAMATH_CALUDE_max_radius_of_circle_l3206_320612

-- Define a circle in 2D space
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the theorem
theorem max_radius_of_circle (C : Set (ℝ × ℝ)) (center : ℝ × ℝ) (radius : ℝ)
  (h1 : C = Circle center radius)
  (h2 : (4, 0) ∈ C)
  (h3 : (-4, 0) ∈ C)
  (h4 : ∃ (x y : ℝ), (x, y) ∈ C) :
  radius ≤ 4 := by
sorry

end NUMINAMATH_CALUDE_max_radius_of_circle_l3206_320612


namespace NUMINAMATH_CALUDE_complex_calculation_l3206_320662

theorem complex_calculation : 
  let z : ℂ := 1 - Complex.I
  2 / z + z^2 = 1 - Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_calculation_l3206_320662


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3206_320655

/-- Given a complex number z = (a+1) - ai where a is real,
    prove that a = -1 is a sufficient but not necessary condition for |z| = 1 -/
theorem sufficient_not_necessary_condition (a : ℝ) :
  let z : ℂ := (a + 1) - a * I
  (a = -1 → Complex.abs z = 1) ∧
  ¬(Complex.abs z = 1 → a = -1) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3206_320655


namespace NUMINAMATH_CALUDE_yellow_parrots_count_l3206_320688

theorem yellow_parrots_count (total : ℕ) (red_fraction : ℚ) : 
  total = 108 → red_fraction = 5/6 → (1 - red_fraction) * total = 18 := by
  sorry

end NUMINAMATH_CALUDE_yellow_parrots_count_l3206_320688


namespace NUMINAMATH_CALUDE_probability_at_least_one_one_is_correct_l3206_320631

/-- The number of sides on each die -/
def sides : ℕ := 6

/-- The probability of at least one die showing a 1 when two fair 6-sided dice are rolled -/
def probability_at_least_one_one : ℚ := 11 / 36

/-- Theorem stating that the probability of at least one die showing a 1 
    when two fair 6-sided dice are rolled is 11/36 -/
theorem probability_at_least_one_one_is_correct : 
  probability_at_least_one_one = 11 / 36 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_one_is_correct_l3206_320631


namespace NUMINAMATH_CALUDE_sum_digits_first_1500_even_l3206_320618

/-- The number of digits in a positive integer -/
def num_digits (n : ℕ) : ℕ := sorry

/-- The sum of digits for all even numbers from 2 to n -/
def sum_digits_even (n : ℕ) : ℕ := sorry

/-- The 1500th positive even integer -/
def nth_even (n : ℕ) : ℕ := 2 * n

theorem sum_digits_first_1500_even :
  sum_digits_even (nth_even 1500) = 5448 := by sorry

end NUMINAMATH_CALUDE_sum_digits_first_1500_even_l3206_320618


namespace NUMINAMATH_CALUDE_ferris_wheel_seats_l3206_320605

theorem ferris_wheel_seats (people_per_seat : ℕ) (total_people : ℕ) (h1 : people_per_seat = 9) (h2 : total_people = 18) :
  total_people / people_per_seat = 2 :=
by sorry

end NUMINAMATH_CALUDE_ferris_wheel_seats_l3206_320605


namespace NUMINAMATH_CALUDE_pedal_triangle_area_l3206_320686

/-- Given a triangle with area S and circumradius R, and a point at distance d from the circumcenter,
    the area S₁ of the triangle formed by the feet of perpendiculars from this point to the sides of
    the original triangle is equal to (S/4) * |1 - (d²/R²)| -/
theorem pedal_triangle_area (S R d S₁ : ℝ) (h₁ : S > 0) (h₂ : R > 0) :
  S₁ = (S / 4) * |1 - (d^2 / R^2)| := by
  sorry

end NUMINAMATH_CALUDE_pedal_triangle_area_l3206_320686


namespace NUMINAMATH_CALUDE_blithe_toys_proof_l3206_320607

/-- The number of toys Blithe lost -/
def lost_toys : ℕ := 6

/-- The number of toys Blithe found -/
def found_toys : ℕ := 9

/-- The number of toys Blithe had after losing and finding toys -/
def final_toys : ℕ := 43

/-- The initial number of toys Blithe had -/
def initial_toys : ℕ := 40

theorem blithe_toys_proof :
  initial_toys - lost_toys + found_toys = final_toys :=
by sorry

end NUMINAMATH_CALUDE_blithe_toys_proof_l3206_320607


namespace NUMINAMATH_CALUDE_num_true_propositions_l3206_320628

-- Define the original proposition
def original_proposition (x y : ℝ) : Prop := x^2 > y^2 → x > y

-- Define the converse
def converse (x y : ℝ) : Prop := x > y → x^2 > y^2

-- Define the inverse
def inverse (x y : ℝ) : Prop := ¬(x^2 > y^2) → ¬(x > y)

-- Define the contrapositive
def contrapositive (x y : ℝ) : Prop := ¬(x > y) → ¬(x^2 > y^2)

-- Theorem statement
theorem num_true_propositions : 
  (∃ x y : ℝ, ¬(original_proposition x y)) ∧ 
  (∃ x y : ℝ, ¬(converse x y)) ∧ 
  (∃ x y : ℝ, ¬(inverse x y)) ∧ 
  (∃ x y : ℝ, ¬(contrapositive x y)) := by
  sorry

end NUMINAMATH_CALUDE_num_true_propositions_l3206_320628


namespace NUMINAMATH_CALUDE_catherine_bottle_caps_l3206_320613

def number_of_friends : ℕ := 6
def bottle_caps_per_friend : ℕ := 3

theorem catherine_bottle_caps : 
  number_of_friends * bottle_caps_per_friend = 18 := by
  sorry

end NUMINAMATH_CALUDE_catherine_bottle_caps_l3206_320613


namespace NUMINAMATH_CALUDE_teachers_percentage_of_boys_l3206_320634

/-- Proves that the percentage of teachers to boys is 20% given the specified conditions -/
theorem teachers_percentage_of_boys (boys girls teachers : ℕ) : 
  (boys : ℚ) / (girls : ℚ) = 3 / 4 →
  girls = 60 →
  boys + girls + teachers = 114 →
  (teachers : ℚ) / (boys : ℚ) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_teachers_percentage_of_boys_l3206_320634


namespace NUMINAMATH_CALUDE_set_operations_l3206_320682

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B : Set ℝ := {x | 0 < x ∧ x ≤ 4}

-- Define the intervals for the expected results
def open_interval (a b : ℝ) : Set ℝ := {x | a < x ∧ x < b}
def closed_open_interval (a b : ℝ) : Set ℝ := {x | a ≤ x ∧ x < b}
def open_closed_interval (a b : ℝ) : Set ℝ := {x | a < x ∧ x ≤ b}
def left_ray (a : ℝ) : Set ℝ := {x | x ≤ a}
def right_ray (a : ℝ) : Set ℝ := {x | a < x}

-- State the theorem
theorem set_operations :
  (A ∩ B = open_interval 0 3) ∧
  (A ∪ B = open_interval (-1) 4) ∧
  ((Aᶜ ∩ Bᶜ) = left_ray (-1) ∪ right_ray 4) :=
by sorry

end NUMINAMATH_CALUDE_set_operations_l3206_320682


namespace NUMINAMATH_CALUDE_swordfish_difference_l3206_320639

/-- The number of times Shelly and Sam go fishing -/
def fishing_trips : ℕ := 5

/-- The total number of swordfish caught in all trips -/
def total_swordfish : ℕ := 25

/-- The number of swordfish Shelly catches each time -/
def shelly_catch : ℕ := 5 - 2

/-- The number of swordfish Sam catches each time -/
def sam_catch : ℕ := (total_swordfish - fishing_trips * shelly_catch) / fishing_trips

theorem swordfish_difference : shelly_catch - sam_catch = 1 := by
  sorry

end NUMINAMATH_CALUDE_swordfish_difference_l3206_320639


namespace NUMINAMATH_CALUDE_complex_sum_theorem_l3206_320666

theorem complex_sum_theorem (a b c d e f g h : ℝ) : 
  b = 2 → 
  g = -a - c - e → 
  3 * ((a + b * I) + (c + d * I) + (e + f * I) + (g + h * I)) = 2 * I → 
  d + f + h = -4/3 := by
sorry

end NUMINAMATH_CALUDE_complex_sum_theorem_l3206_320666


namespace NUMINAMATH_CALUDE_susan_first_turn_spaces_l3206_320685

/-- The number of spaces on the board game --/
def total_spaces : ℕ := 48

/-- The number of spaces Susan moves forward on the second turn --/
def second_turn_forward : ℤ := 2

/-- The number of spaces Susan moves backward on the second turn --/
def second_turn_backward : ℤ := 5

/-- The number of spaces Susan moves on the third turn --/
def third_turn : ℕ := 6

/-- The number of spaces Susan still needs to move after three turns --/
def remaining_spaces : ℕ := 37

/-- The number of spaces Susan moved on the first turn --/
def first_turn : ℕ := 8

theorem susan_first_turn_spaces :
  (first_turn : ℤ) + (second_turn_forward - second_turn_backward) + third_turn + remaining_spaces = total_spaces :=
sorry

end NUMINAMATH_CALUDE_susan_first_turn_spaces_l3206_320685


namespace NUMINAMATH_CALUDE_stadium_length_yards_l3206_320694

-- Define the length of the stadium in feet
def stadium_length_feet : ℕ := 186

-- Define the number of feet in a yard
def feet_per_yard : ℕ := 3

-- Theorem to prove the length of the stadium in yards
theorem stadium_length_yards : 
  stadium_length_feet / feet_per_yard = 62 := by
  sorry

end NUMINAMATH_CALUDE_stadium_length_yards_l3206_320694


namespace NUMINAMATH_CALUDE_root_product_equals_twenty_l3206_320625

theorem root_product_equals_twenty :
  (32 : ℝ) ^ (1/5) * (16 : ℝ) ^ (1/4) * (25 : ℝ) ^ (1/2) = 20 := by
  sorry

end NUMINAMATH_CALUDE_root_product_equals_twenty_l3206_320625


namespace NUMINAMATH_CALUDE_largest_common_divisor_360_315_l3206_320695

theorem largest_common_divisor_360_315 : Nat.gcd 360 315 = 45 := by
  sorry

end NUMINAMATH_CALUDE_largest_common_divisor_360_315_l3206_320695


namespace NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l3206_320692

def i : ℝ × ℝ := (1, 0)
def j : ℝ × ℝ := (0, 1)
def a : ℝ × ℝ := (2 * i.1 + 0 * i.2, 0 * j.1 + 3 * j.2)
def b (k : ℝ) : ℝ × ℝ := (k * i.1 + 0 * i.2, 0 * j.1 - 4 * j.2)

theorem perpendicular_vectors_k_value :
  ∀ k : ℝ, (a.1 * (b k).1 + a.2 * (b k).2 = 0) → k = 6 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l3206_320692


namespace NUMINAMATH_CALUDE_train_passengers_l3206_320690

theorem train_passengers (initial_passengers : ℕ) (stops : ℕ) : 
  initial_passengers = 64 → stops = 4 → 
  (initial_passengers : ℚ) * ((2 : ℚ) / 3) ^ stops = 1024 / 81 := by
  sorry

end NUMINAMATH_CALUDE_train_passengers_l3206_320690


namespace NUMINAMATH_CALUDE_inverse_of_five_mod_221_l3206_320619

theorem inverse_of_five_mod_221 : ∃! x : ℕ, x ∈ Finset.range 221 ∧ (5 * x) % 221 = 1 :=
by
  use 177
  sorry

end NUMINAMATH_CALUDE_inverse_of_five_mod_221_l3206_320619


namespace NUMINAMATH_CALUDE_clouds_weather_relationship_l3206_320661

/-- Represents the contingency table data --/
structure ContingencyTable where
  clouds_rain : Nat
  clouds_no_rain : Nat
  no_clouds_rain : Nat
  no_clouds_no_rain : Nat

/-- Represents the χ² test result --/
structure ChiSquareTest where
  calculated_value : Real
  critical_value : Real

/-- Theorem stating the relationship between clouds at sunset and nighttime weather --/
theorem clouds_weather_relationship (data : ContingencyTable) (test : ChiSquareTest) :
  data.clouds_rain + data.clouds_no_rain + data.no_clouds_rain + data.no_clouds_no_rain = 100 →
  data.clouds_rain + data.no_clouds_rain = 50 →
  data.clouds_no_rain + data.no_clouds_no_rain = 50 →
  test.calculated_value > test.critical_value →
  ∃ (relationship : Prop), relationship := by
  sorry

#check clouds_weather_relationship

end NUMINAMATH_CALUDE_clouds_weather_relationship_l3206_320661


namespace NUMINAMATH_CALUDE_f_neg_two_equals_thirteen_l3206_320617

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^4 + b * x^2 - x + 1

-- State the theorem
theorem f_neg_two_equals_thirteen (a b : ℝ) (h : f a b 2 = 9) : f a b (-2) = 13 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_two_equals_thirteen_l3206_320617


namespace NUMINAMATH_CALUDE_root_of_equation_l3206_320642

theorem root_of_equation (x : ℝ) : 
  (18 / (x^2 - 9) - 3 / (x - 3) = 2) ↔ (x = -4.5) :=
by sorry

end NUMINAMATH_CALUDE_root_of_equation_l3206_320642


namespace NUMINAMATH_CALUDE_solution_in_interval_l3206_320697

noncomputable def f (x : ℝ) : ℝ := Real.log x + x - 4

theorem solution_in_interval :
  ∃ x₀ : ℝ, x₀ ∈ Set.Ioo 2 3 ∧ f x₀ = 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_in_interval_l3206_320697


namespace NUMINAMATH_CALUDE_min_sum_squares_l3206_320683

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*y = 0

-- Define the line
def line (x y : ℝ) : Prop := y = x - 1

-- Define a point on the line
structure PointOnLine where
  x : ℝ
  y : ℝ
  on_line : line x y

-- Define the diameter AB
structure Diameter where
  A : ℝ × ℝ
  B : ℝ × ℝ
  is_diameter : ∀ (x y : ℝ), circle_C x y → 
    (x - A.1)^2 + (y - A.2)^2 + (x - B.1)^2 + (y - B.2)^2 = 4

-- Theorem statement
theorem min_sum_squares (d : Diameter) :
  ∃ (min : ℝ), min = 6 ∧ 
  ∀ (P : PointOnLine), 
    (P.x - d.A.1)^2 + (P.y - d.A.2)^2 + (P.x - d.B.1)^2 + (P.y - d.B.2)^2 ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l3206_320683


namespace NUMINAMATH_CALUDE_percent_within_one_std_dev_l3206_320677

-- Define a symmetric distribution
structure SymmetricDistribution where
  mean : ℝ
  std_dev : ℝ
  is_symmetric : Bool
  percent_less_than_mean_plus_std : ℝ

-- Theorem statement
theorem percent_within_one_std_dev 
  (dist : SymmetricDistribution) 
  (h1 : dist.is_symmetric = true) 
  (h2 : dist.percent_less_than_mean_plus_std = 92) : 
  (100 - 2 * (100 - dist.percent_less_than_mean_plus_std)) = 84 := by
  sorry

end NUMINAMATH_CALUDE_percent_within_one_std_dev_l3206_320677


namespace NUMINAMATH_CALUDE_hotel_outlets_count_l3206_320635

/-- Represents the number of outlets required for different room types and the distribution of outlet types -/
structure HotelOutlets where
  standardRoomOutlets : ℕ
  suiteOutlets : ℕ
  standardRoomCount : ℕ
  suiteCount : ℕ
  typeAPercentage : ℚ
  typeBPercentage : ℚ
  typeCPercentage : ℚ

/-- Calculates the total number of outlets needed for a hotel -/
def totalOutlets (h : HotelOutlets) : ℕ :=
  h.standardRoomCount * h.standardRoomOutlets +
  h.suiteCount * h.suiteOutlets

/-- Theorem stating that the total number of outlets for the given hotel configuration is 650 -/
theorem hotel_outlets_count (h : HotelOutlets)
    (h_standard : h.standardRoomOutlets = 10)
    (h_suite : h.suiteOutlets = 15)
    (h_standard_count : h.standardRoomCount = 50)
    (h_suite_count : h.suiteCount = 10)
    (h_typeA : h.typeAPercentage = 2/5)
    (h_typeB : h.typeBPercentage = 3/5)
    (h_typeC : h.typeCPercentage = 1) :
  totalOutlets h = 650 := by
  sorry

end NUMINAMATH_CALUDE_hotel_outlets_count_l3206_320635


namespace NUMINAMATH_CALUDE_base12Addition_l3206_320651

/-- Converts a base 12 number represented as a list of digits to its decimal equivalent -/
def base12ToDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => acc * 12 + d) 0

/-- Converts a decimal number to its base 12 representation -/
def decimalToBase12 (n : Nat) : List Nat :=
  if n < 12 then [n]
  else (n % 12) :: decimalToBase12 (n / 12)

/-- Represents the base 12 number 857₁₂ -/
def num1 : List Nat := [7, 5, 8]

/-- Represents the base 12 number 296₁₂ -/
def num2 : List Nat := [6, 9, 2]

/-- Represents the base 12 number B31₁₂ -/
def result : List Nat := [1, 3, 11]

theorem base12Addition :
  decimalToBase12 (base12ToDecimal num1 + base12ToDecimal num2) = result := by
  sorry

#eval base12ToDecimal num1
#eval base12ToDecimal num2
#eval base12ToDecimal result
#eval decimalToBase12 (base12ToDecimal num1 + base12ToDecimal num2)

end NUMINAMATH_CALUDE_base12Addition_l3206_320651


namespace NUMINAMATH_CALUDE_peace_treaty_day_l3206_320679

def day_of_week : Fin 7 → String
| 0 => "Sunday"
| 1 => "Monday"
| 2 => "Tuesday"
| 3 => "Wednesday"
| 4 => "Thursday"
| 5 => "Friday"
| 6 => "Saturday"

def days_between : Nat := 919

theorem peace_treaty_day :
  let start_day : Fin 7 := 4  -- Thursday
  let end_day : Fin 7 := (start_day + days_between) % 7
  day_of_week end_day = "Saturday" := by
  sorry


end NUMINAMATH_CALUDE_peace_treaty_day_l3206_320679


namespace NUMINAMATH_CALUDE_smallest_sum_of_five_consecutive_primes_l3206_320687

/-- A function that returns true if a number is prime, false otherwise -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that returns the nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

/-- A function that returns true if five consecutive primes starting from the nth prime sum to a multiple of 3, false otherwise -/
def sumDivisibleByThree (n : ℕ) : Prop :=
  (nthPrime n + nthPrime (n+1) + nthPrime (n+2) + nthPrime (n+3) + nthPrime (n+4)) % 3 = 0

/-- The index of the first prime in the sequence of five consecutive primes that sum to 39 -/
def firstPrimeIndex : ℕ := sorry

theorem smallest_sum_of_five_consecutive_primes :
  (∀ k < firstPrimeIndex, ¬sumDivisibleByThree k) ∧
  sumDivisibleByThree firstPrimeIndex ∧
  nthPrime firstPrimeIndex + nthPrime (firstPrimeIndex+1) + nthPrime (firstPrimeIndex+2) +
  nthPrime (firstPrimeIndex+3) + nthPrime (firstPrimeIndex+4) = 39 := by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_five_consecutive_primes_l3206_320687


namespace NUMINAMATH_CALUDE_social_event_handshakes_l3206_320624

/-- Represents the social event setup -/
structure SocialEvent where
  total_people : Nat
  group_a_size : Nat
  group_b_size : Nat
  group_b_connections : Nat

/-- Calculate the number of handshakes in the social event -/
def calculate_handshakes (event : SocialEvent) : Nat :=
  let group_b_internal := event.group_b_size.choose 2
  let group_ab_handshakes := event.group_b_size * event.group_b_connections
  group_b_internal + group_ab_handshakes

/-- Theorem stating the number of handshakes in the given social event -/
theorem social_event_handshakes :
  ∃ (event : SocialEvent),
    event.total_people = 40 ∧
    event.group_a_size = 25 ∧
    event.group_b_size = 15 ∧
    event.group_b_connections = 5 ∧
    calculate_handshakes event = 180 := by
  sorry

end NUMINAMATH_CALUDE_social_event_handshakes_l3206_320624


namespace NUMINAMATH_CALUDE_maximize_product_l3206_320629

theorem maximize_product (A : ℝ) (h : A > 0) :
  ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b + c = A ∧
    ∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 → x + y + z = A →
      x * y^2 * z^3 ≤ a * b^2 * c^3 ∧
    a = A / 6 ∧ b = A / 3 ∧ c = A / 2 :=
sorry

end NUMINAMATH_CALUDE_maximize_product_l3206_320629


namespace NUMINAMATH_CALUDE_price_decrease_l3206_320622

/-- Given an article with an original price of 700 rupees and a price decrease of 24%,
    the new price after the decrease is 532 rupees. -/
theorem price_decrease (original_price : ℝ) (decrease_percentage : ℝ) (new_price : ℝ) :
  original_price = 700 →
  decrease_percentage = 24 →
  new_price = original_price * (1 - decrease_percentage / 100) →
  new_price = 532 := by
sorry

end NUMINAMATH_CALUDE_price_decrease_l3206_320622


namespace NUMINAMATH_CALUDE_digit_difference_digit_difference_proof_l3206_320678

theorem digit_difference : ℕ → Prop :=
  fun n =>
    (∀ m : ℕ, m < 1000 → m < n) ∧
    (n < 10000) ∧
    (∀ k : ℕ, k < 1000) →
    n - 999 = 1

-- The proof
theorem digit_difference_proof : digit_difference 1000 := by
  sorry

end NUMINAMATH_CALUDE_digit_difference_digit_difference_proof_l3206_320678


namespace NUMINAMATH_CALUDE_diamond_expression_evaluation_l3206_320620

-- Define the diamond operation
def diamond (a b : ℚ) : ℚ := a - 1 / b

-- State the theorem
theorem diamond_expression_evaluation :
  let x := diamond (diamond 2 3) 4
  let y := diamond 2 (diamond 3 4)
  x - y = -29/132 := by sorry

end NUMINAMATH_CALUDE_diamond_expression_evaluation_l3206_320620


namespace NUMINAMATH_CALUDE_max_x_value_l3206_320621

theorem max_x_value : 
  ∃ (x_max : ℝ), 
    (∀ x : ℝ, ((5*x - 20)/(4*x - 5))^2 + ((5*x - 20)/(4*x - 5)) = 20 → x ≤ x_max) ∧
    ((5*x_max - 20)/(4*x_max - 5))^2 + ((5*x_max - 20)/(4*x_max - 5)) = 20 ∧
    x_max = 9/5 :=
by sorry

end NUMINAMATH_CALUDE_max_x_value_l3206_320621


namespace NUMINAMATH_CALUDE_gcd_of_special_powers_l3206_320668

theorem gcd_of_special_powers :
  Nat.gcd (2^2020 - 1) (2^2000 - 1) = 2^20 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_special_powers_l3206_320668


namespace NUMINAMATH_CALUDE_fourth_root_simplification_l3206_320663

theorem fourth_root_simplification (x : ℝ) (hx : x > 0) :
  (x^3 * (x^5)^(1/2))^(1/4) = x^(11/8) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_simplification_l3206_320663


namespace NUMINAMATH_CALUDE_inequality_holds_l3206_320680

theorem inequality_holds (a b c d : ℝ) (h1 : b < a) (h2 : d < c) : a + c > b + d := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_l3206_320680


namespace NUMINAMATH_CALUDE_backpack_price_l3206_320603

/-- The price of a backpack and three ring-binders, given price changes and total spent --/
theorem backpack_price (B : ℕ) : 
  (∃ (new_backpack_price new_binder_price : ℕ),
    -- Original price of each ring-binder
    20 = 20 ∧
    -- New backpack price is $5 more than original
    new_backpack_price = B + 5 ∧
    -- New ring-binder price is $2 less than original
    new_binder_price = 20 - 2 ∧
    -- Total spent is $109
    new_backpack_price + 3 * new_binder_price = 109) →
  -- Original backpack price was $50
  B = 50 := by
sorry

end NUMINAMATH_CALUDE_backpack_price_l3206_320603


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l3206_320636

theorem quadratic_roots_sum (α β : ℝ) : 
  (α^2 - 7*α + 3 = 0) → 
  (β^2 - 7*β + 3 = 0) → 
  (α > β) → 
  (α^2 + 7*β = 46) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l3206_320636


namespace NUMINAMATH_CALUDE_board_officer_selection_ways_l3206_320611

def board_size : ℕ := 30
def num_officers : ℕ := 4

def ways_without_special_members : ℕ := 26 * 25 * 24 * 23
def ways_with_one_pair : ℕ := 4 * 3 * 26 * 25
def ways_with_both_pairs : ℕ := 4 * 3 * 2 * 1

theorem board_officer_selection_ways :
  ways_without_special_members + 2 * ways_with_one_pair + ways_with_both_pairs = 374424 :=
sorry

end NUMINAMATH_CALUDE_board_officer_selection_ways_l3206_320611


namespace NUMINAMATH_CALUDE_constant_function_l3206_320698

/-- A function satisfying the given functional equation -/
def FunctionalEq (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x y : ℝ, f (x + y) = f x * f (a - y) + f y * f (a - x)

theorem constant_function (f : ℝ → ℝ) (h1 : f 0 = 1/2) (h2 : FunctionalEq f) :
    ∀ x : ℝ, f x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_constant_function_l3206_320698


namespace NUMINAMATH_CALUDE_era_burgers_l3206_320643

theorem era_burgers (num_friends : ℕ) (slices_per_burger : ℕ) 
  (friend1_slices : ℕ) (friend2_slices : ℕ) (friend3_slices : ℕ) 
  (friend4_slices : ℕ) (era_slices : ℕ) :
  num_friends = 4 →
  slices_per_burger = 2 →
  friend1_slices = 1 →
  friend2_slices = 2 →
  friend3_slices = 3 →
  friend4_slices = 3 →
  era_slices = 1 →
  (friend1_slices + friend2_slices + friend3_slices + friend4_slices + era_slices) / slices_per_burger = 5 := by
  sorry

end NUMINAMATH_CALUDE_era_burgers_l3206_320643


namespace NUMINAMATH_CALUDE_somu_father_age_ratio_l3206_320676

/-- Proves the ratio of Somu's age to his father's age -/
theorem somu_father_age_ratio :
  ∀ (S F : ℕ),
  S = 12 →
  S - 6 = (F - 6) / 5 →
  ∃ (a b : ℕ), a ≠ 0 ∧ b ≠ 0 ∧ S * b = F * a ∧ a = 1 ∧ b = 3 :=
by sorry

end NUMINAMATH_CALUDE_somu_father_age_ratio_l3206_320676


namespace NUMINAMATH_CALUDE_jellybean_box_capacity_l3206_320627

/-- Represents a rectangular box with a certain capacity of jellybeans -/
structure JellyBean_Box where
  height : ℝ
  width : ℝ
  length : ℝ
  capacity : ℕ

/-- Calculates the volume of a JellyBean_Box -/
def box_volume (box : JellyBean_Box) : ℝ :=
  box.height * box.width * box.length

/-- Theorem stating the relationship between box sizes and jellybean capacities -/
theorem jellybean_box_capacity 
  (box_b box_c : JellyBean_Box)
  (h_capacity_b : box_b.capacity = 125)
  (h_height : box_c.height = 2 * box_b.height)
  (h_width : box_c.width = 2 * box_b.width)
  (h_length : box_c.length = 2 * box_b.length) :
  box_c.capacity = 1000 :=
by sorry


end NUMINAMATH_CALUDE_jellybean_box_capacity_l3206_320627


namespace NUMINAMATH_CALUDE_field_area_diminished_l3206_320670

theorem field_area_diminished (L W : ℝ) (h : L > 0 ∧ W > 0) :
  let original_area := L * W
  let new_length := L * (1 - 0.4)
  let new_width := W * (1 - 0.4)
  let new_area := new_length * new_width
  (original_area - new_area) / original_area = 0.64 := by sorry

end NUMINAMATH_CALUDE_field_area_diminished_l3206_320670


namespace NUMINAMATH_CALUDE_bulb_probability_l3206_320647

/-- The probability that a bulb from factory X works for over 4000 hours -/
def prob_x : ℝ := 0.59

/-- The probability that a bulb from factory Y works for over 4000 hours -/
def prob_y : ℝ := 0.65

/-- The probability that a bulb from factory Z works for over 4000 hours -/
def prob_z : ℝ := 0.70

/-- The proportion of bulbs supplied by factory X -/
def supply_x : ℝ := 0.5

/-- The proportion of bulbs supplied by factory Y -/
def supply_y : ℝ := 0.3

/-- The proportion of bulbs supplied by factory Z -/
def supply_z : ℝ := 0.2

/-- The overall probability that a randomly selected bulb will work for over 4000 hours -/
def overall_prob : ℝ := supply_x * prob_x + supply_y * prob_y + supply_z * prob_z

theorem bulb_probability : overall_prob = 0.63 := by sorry

end NUMINAMATH_CALUDE_bulb_probability_l3206_320647


namespace NUMINAMATH_CALUDE_initial_time_is_six_hours_l3206_320626

/-- Proves that the initial time to cover 288 km is 6 hours -/
theorem initial_time_is_six_hours (distance : ℝ) (speed_new : ℝ) (time_factor : ℝ) :
  distance = 288 →
  speed_new = 32 →
  time_factor = 3 / 2 →
  ∃ (time_initial : ℝ),
    distance = speed_new * (time_factor * time_initial) ∧
    time_initial = 6 := by
  sorry


end NUMINAMATH_CALUDE_initial_time_is_six_hours_l3206_320626


namespace NUMINAMATH_CALUDE_total_brass_l3206_320669

def brass_composition (copper zinc : ℝ) : Prop :=
  copper / zinc = 13 / 7

theorem total_brass (zinc : ℝ) (h : zinc = 35) :
  ∃ total : ℝ, brass_composition (total - zinc) zinc ∧ total = 100 :=
sorry

end NUMINAMATH_CALUDE_total_brass_l3206_320669


namespace NUMINAMATH_CALUDE_spider_dressing_8_pairs_l3206_320650

/-- The number of ways a spider can put on n pairs of socks and shoes -/
def spiderDressingWays (n : ℕ) : ℕ :=
  Nat.factorial (2 * n) / (2^n)

/-- Theorem: For 8 pairs of socks and shoes, the number of ways is 81729648000 -/
theorem spider_dressing_8_pairs :
  spiderDressingWays 8 = 81729648000 := by
  sorry

end NUMINAMATH_CALUDE_spider_dressing_8_pairs_l3206_320650


namespace NUMINAMATH_CALUDE_power_of_power_l3206_320657

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l3206_320657


namespace NUMINAMATH_CALUDE_line_through_point_equal_intercepts_l3206_320671

theorem line_through_point_equal_intercepts :
  ∃ (m b : ℝ), (3 = m * 2 + b) ∧ (∃ (a : ℝ), a ≠ 0 ∧ (a = -b/m ∧ a = b)) :=
by sorry

end NUMINAMATH_CALUDE_line_through_point_equal_intercepts_l3206_320671


namespace NUMINAMATH_CALUDE_divisibility_of_m_l3206_320665

theorem divisibility_of_m (m : ℤ) : m = 76^2006 - 76 → 100 ∣ m := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_m_l3206_320665


namespace NUMINAMATH_CALUDE_min_value_expression_l3206_320659

theorem min_value_expression (x₁ x₂ : ℝ) (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₁ + x₂ = 1) :
  3 * x₁ / x₂ + 1 / (x₁ * x₂) ≥ 6 ∧ 
  ∃ x₁' x₂' : ℝ, x₁' > 0 ∧ x₂' > 0 ∧ x₁' + x₂' = 1 ∧ 3 * x₁' / x₂' + 1 / (x₁' * x₂') = 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3206_320659


namespace NUMINAMATH_CALUDE_trig_expression_equals_one_l3206_320641

theorem trig_expression_equals_one : 
  Real.sqrt 3 * Real.tan (30 * π / 180) * Real.cos (60 * π / 180) + Real.sin (45 * π / 180) ^ 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_one_l3206_320641


namespace NUMINAMATH_CALUDE_bus_seat_difference_l3206_320681

/-- Represents a bus with seats on both sides and a special seat at the back. -/
structure Bus where
  leftSeats : Nat
  rightSeats : Nat
  backSeatCapacity : Nat
  regularSeatCapacity : Nat
  totalCapacity : Nat

/-- The difference in the number of seats between the left and right sides of the bus. -/
def seatDifference (bus : Bus) : Nat :=
  bus.leftSeats - bus.rightSeats

/-- Theorem stating the difference in seats for a specific bus configuration. -/
theorem bus_seat_difference :
  ∃ (bus : Bus),
    bus.leftSeats = 15 ∧
    bus.backSeatCapacity = 11 ∧
    bus.regularSeatCapacity = 3 ∧
    bus.totalCapacity = 92 ∧
    seatDifference bus = 3 := by
  sorry

#check bus_seat_difference

end NUMINAMATH_CALUDE_bus_seat_difference_l3206_320681


namespace NUMINAMATH_CALUDE_correct_oranges_put_back_l3206_320654

/-- Represents the fruit selection problem with given prices and quantities -/
structure FruitSelection where
  apple_price : ℚ
  orange_price : ℚ
  total_fruits : ℕ
  initial_avg_price : ℚ
  desired_avg_price : ℚ

/-- Calculates the number of oranges to put back to achieve the desired average price -/
def oranges_to_put_back (fs : FruitSelection) : ℕ :=
  2

/-- Theorem stating that putting back the calculated number of oranges achieves the desired average price -/
theorem correct_oranges_put_back (fs : FruitSelection)
  (h1 : fs.apple_price = 40/100)
  (h2 : fs.orange_price = 60/100)
  (h3 : fs.total_fruits = 10)
  (h4 : fs.initial_avg_price = 48/100)
  (h5 : fs.desired_avg_price = 45/100) :
  let num_oranges_back := oranges_to_put_back fs
  let remaining_fruits := fs.total_fruits - num_oranges_back
  let num_apples := 6  -- Derived from the problem's solution
  let num_oranges := 4 -- Derived from the problem's solution
  fs.apple_price * num_apples + fs.orange_price * (num_oranges - num_oranges_back) =
    fs.desired_avg_price * remaining_fruits :=
by
  sorry

end NUMINAMATH_CALUDE_correct_oranges_put_back_l3206_320654


namespace NUMINAMATH_CALUDE_arithmetic_sequence_n_equals_5_l3206_320610

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  a_pos : ∀ n, a n > 0
  a_1 : a 1 = 3
  S_3 : (a 1) + (a 2) + (a 3) = 21
  a_n : ∃ n, a n = 48

/-- The theorem stating that n = 5 for the given arithmetic sequence -/
theorem arithmetic_sequence_n_equals_5 (seq : ArithmeticSequence) :
  ∃ n, seq.a n = 48 ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_n_equals_5_l3206_320610


namespace NUMINAMATH_CALUDE_stockholm_uppsala_distance_l3206_320632

/-- The scale of the map in km per cm -/
def map_scale : ℝ := 10

/-- The distance between Stockholm and Uppsala on the map in cm -/
def map_distance : ℝ := 45

/-- The actual distance between Stockholm and Uppsala in km -/
def actual_distance : ℝ := map_distance * map_scale

theorem stockholm_uppsala_distance :
  actual_distance = 450 := by sorry

end NUMINAMATH_CALUDE_stockholm_uppsala_distance_l3206_320632
