import Mathlib

namespace NUMINAMATH_CALUDE_pencil_cost_l1470_147055

theorem pencil_cost (total_students : Nat) (total_cost : Nat) 
  (h1 : total_students = 30)
  (h2 : total_cost = 1771)
  (h3 : ∃ (s n c : Nat), 
    s > total_students / 2 ∧ 
    n > 1 ∧ 
    c > n ∧ 
    s * n * c = total_cost) :
  ∃ (s n : Nat), s * n * 11 = total_cost :=
by sorry

end NUMINAMATH_CALUDE_pencil_cost_l1470_147055


namespace NUMINAMATH_CALUDE_total_baked_goods_is_338_l1470_147059

/-- The total number of baked goods Diane makes -/
def total_baked_goods : ℕ :=
  let gingerbread_trays : ℕ := 4
  let gingerbread_per_tray : ℕ := 25
  let chocolate_chip_trays : ℕ := 3
  let chocolate_chip_per_tray : ℕ := 30
  let oatmeal_trays : ℕ := 2
  let oatmeal_per_tray : ℕ := 20
  let sugar_trays : ℕ := 6
  let sugar_per_tray : ℕ := 18
  gingerbread_trays * gingerbread_per_tray +
  chocolate_chip_trays * chocolate_chip_per_tray +
  oatmeal_trays * oatmeal_per_tray +
  sugar_trays * sugar_per_tray

theorem total_baked_goods_is_338 : total_baked_goods = 338 := by
  sorry

end NUMINAMATH_CALUDE_total_baked_goods_is_338_l1470_147059


namespace NUMINAMATH_CALUDE_no_quadratic_polynomials_with_special_roots_l1470_147025

theorem no_quadratic_polynomials_with_special_roots : 
  ¬ ∃ (a b c : ℝ), a ≠ 0 ∧ 
  ∃ (x y : ℝ), x + y = -b / a ∧ x * y = c / a ∧
  ((x = a + b + c ∧ y = a * b * c) ∨ (y = a + b + c ∧ x = a * b * c)) :=
sorry

end NUMINAMATH_CALUDE_no_quadratic_polynomials_with_special_roots_l1470_147025


namespace NUMINAMATH_CALUDE_not_divisible_by_five_l1470_147044

theorem not_divisible_by_five (a : ℤ) (h : ¬(5 ∣ a)) : ¬(5 ∣ (3 * a^4 + 1)) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_five_l1470_147044


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1470_147093

def f (a b x : ℝ) : ℝ := x^5 - 3*x^4 + a*x^3 + b*x^2 - 5*x - 5

theorem polynomial_divisibility (a b : ℝ) :
  (∀ x, (x^2 - 1) ∣ f a b x) ↔ (a = 4 ∧ b = 8) :=
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l1470_147093


namespace NUMINAMATH_CALUDE_parallel_vectors_iff_m_values_l1470_147078

/-- Two 2D vectors are parallel if and only if their cross product is zero -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

/-- The vector a as a function of m -/
def a (m : ℝ) : ℝ × ℝ := (2*m + 1, 3)

/-- The vector b as a function of m -/
def b (m : ℝ) : ℝ × ℝ := (2, m)

/-- Theorem stating that vectors a and b are parallel if and only if m = 3/2 or m = -2 -/
theorem parallel_vectors_iff_m_values :
  ∀ m : ℝ, are_parallel (a m) (b m) ↔ m = 3/2 ∨ m = -2 := by sorry

end NUMINAMATH_CALUDE_parallel_vectors_iff_m_values_l1470_147078


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1470_147057

theorem partial_fraction_decomposition (x P Q R : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 4) :
  5 * x / ((x - 4) * (x - 2)^2) = P / (x - 4) + Q / (x - 2) + R / (x - 2)^2 ↔ P = 5 ∧ Q = -5 ∧ R = -5 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1470_147057


namespace NUMINAMATH_CALUDE_trapezoid_side_length_l1470_147040

-- Define the trapezoid ABCD
structure Trapezoid :=
  (AB : ℝ)
  (CD : ℝ)
  (AD : ℝ)
  (BC : ℝ)

-- Define the properties of the trapezoid
def is_valid_trapezoid (t : Trapezoid) : Prop :=
  t.AB = 10 ∧ 
  t.CD = 2 * t.AB ∧ 
  t.AD = t.BC ∧ 
  t.AB + t.BC + t.CD + t.AD = 42

-- Theorem statement
theorem trapezoid_side_length 
  (t : Trapezoid) 
  (h : is_valid_trapezoid t) : 
  t.AD = 6 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_side_length_l1470_147040


namespace NUMINAMATH_CALUDE_x_minus_y_times_x_plus_y_equals_95_l1470_147022

theorem x_minus_y_times_x_plus_y_equals_95 (x y : ℤ) (h1 : x = 12) (h2 : y = 7) : 
  (x - y) * (x + y) = 95 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_times_x_plus_y_equals_95_l1470_147022


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l1470_147060

theorem hyperbola_asymptote (m : ℝ) :
  (∀ x y : ℝ, x^2 / |m| - y^2 / (|m| + 3) = 1) →
  (2 * Real.sqrt 5 = Real.sqrt (2 * |m| + 3)) →
  (∃ k : ℝ, k = 2 ∧ ∀ x : ℝ, k * x = 2 * x) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l1470_147060


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l1470_147014

/-- Parabola intersecting with a line -/
theorem parabola_line_intersection 
  (a : ℝ) 
  (h_a : a ≠ 0) 
  (b : ℝ) 
  (h_intersection : b = 2 * 1 - 3 ∧ b = a * 1^2) :
  (a = -1 ∧ b = -1) ∧ 
  ∃ x y : ℝ, x = -3 ∧ y = -9 ∧ y = a * x^2 ∧ y = 2 * x - 3 :=
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l1470_147014


namespace NUMINAMATH_CALUDE_line_equation_proof_l1470_147041

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 4 * x + y + 6 = 0
def l₂ (x y : ℝ) : Prop := 3 * x - 5 * y - 6 = 0

-- Define the midpoint P
def P : ℝ × ℝ := (0, 0)

-- Define the line l
def l (x y : ℝ) : Prop := y = -1/6 * x

-- Theorem statement
theorem line_equation_proof :
  ∀ (A B : ℝ × ℝ),
  l₁ A.1 A.2 ∧ l₂ B.1 B.2 ∧  -- A is on l₁, B is on l₂
  (A.1 + B.1) / 2 = P.1 ∧ (A.2 + B.2) / 2 = P.2 →  -- P is the midpoint of AB
  ∀ (x y : ℝ), l x y  -- The equation of line l is y = -1/6 * x
  := by sorry

end NUMINAMATH_CALUDE_line_equation_proof_l1470_147041


namespace NUMINAMATH_CALUDE_average_breadth_is_18_l1470_147004

/-- Represents a trapezoidal plot with equal diagonal distances -/
structure TrapezoidalPlot where
  averageBreadth : ℝ
  maximumLength : ℝ
  area : ℝ

/-- The conditions of the problem -/
def PlotConditions (plot : TrapezoidalPlot) : Prop :=
  plot.area = 23 * plot.averageBreadth ∧
  plot.maximumLength - plot.averageBreadth = 10 ∧
  plot.area = (1/2) * (plot.maximumLength + plot.averageBreadth) * plot.averageBreadth

/-- The theorem to be proved -/
theorem average_breadth_is_18 (plot : TrapezoidalPlot) 
  (h : PlotConditions plot) : plot.averageBreadth = 18 := by
  sorry

end NUMINAMATH_CALUDE_average_breadth_is_18_l1470_147004


namespace NUMINAMATH_CALUDE_triangle_max_perimeter_l1470_147069

theorem triangle_max_perimeter :
  ∀ x : ℕ,
  x > 0 →
  x < 4*x →
  x + 4*x > 20 →
  4*x + 20 > x →
  x + 4*x + 20 ≤ 50 :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_perimeter_l1470_147069


namespace NUMINAMATH_CALUDE_magne_currency_conversion_l1470_147042

/-- Currency conversion rates on planet Magne -/
structure MagneCurrency where
  migs_per_mags : ℕ
  mags_per_mogs : ℕ

/-- Convert a combination of mogs and mags to migs -/
def convert_to_migs (c : MagneCurrency) (mogs : ℕ) (mags : ℕ) : ℕ :=
  mogs * c.mags_per_mogs * c.migs_per_mags + mags * c.migs_per_mags

/-- Theorem: 10 mogs + 6 mags is equal to 528 migs on planet Magne -/
theorem magne_currency_conversion (c : MagneCurrency) 
    (h1 : c.migs_per_mags = 8) 
    (h2 : c.mags_per_mogs = 6) : 
  convert_to_migs c 10 6 = 528 := by
  sorry

end NUMINAMATH_CALUDE_magne_currency_conversion_l1470_147042


namespace NUMINAMATH_CALUDE_baba_yaga_students_l1470_147076

theorem baba_yaga_students (total : ℕ) (boys girls : ℕ) : 
  total = 33 →
  boys + girls = total →
  22 = (2 * total) / 3 := by
  sorry

end NUMINAMATH_CALUDE_baba_yaga_students_l1470_147076


namespace NUMINAMATH_CALUDE_birth_rate_calculation_l1470_147038

/-- The number of people born every two seconds in a city -/
def birth_rate : ℕ := sorry

/-- The death rate in the city (people per two seconds) -/
def death_rate : ℕ := 1

/-- The net population increase in one day -/
def daily_net_increase : ℕ := 259200

/-- The number of two-second intervals in a day -/
def intervals_per_day : ℕ := 24 * 60 * 60 / 2

theorem birth_rate_calculation : 
  birth_rate = (daily_net_increase / intervals_per_day) + death_rate := by
  sorry

end NUMINAMATH_CALUDE_birth_rate_calculation_l1470_147038


namespace NUMINAMATH_CALUDE_complex_ratio_l1470_147037

theorem complex_ratio (z : ℂ) (a b : ℝ) (h1 : z = Complex.mk a b) (h2 : z * (1 - Complex.I) = Complex.I) :
  a / b = -1 := by sorry

end NUMINAMATH_CALUDE_complex_ratio_l1470_147037


namespace NUMINAMATH_CALUDE_probability_at_least_one_defective_l1470_147091

theorem probability_at_least_one_defective (total_bulbs : ℕ) (defective_bulbs : ℕ) 
  (h1 : total_bulbs = 23) (h2 : defective_bulbs = 4) :
  let non_defective := total_bulbs - defective_bulbs
  let prob_both_non_defective := (non_defective / total_bulbs) * ((non_defective - 1) / (total_bulbs - 1))
  1 - prob_both_non_defective = 164 / 506 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_defective_l1470_147091


namespace NUMINAMATH_CALUDE_robert_nickel_vs_ashley_l1470_147082

/-- The number of chocolates eaten by Robert -/
def robert_chocolates : ℕ := 10

/-- The number of chocolates eaten by Nickel -/
def nickel_chocolates : ℕ := 5

/-- The number of chocolates eaten by Ashley -/
def ashley_chocolates : ℕ := 15

/-- Theorem stating that Robert and Nickel together ate the same number of chocolates as Ashley -/
theorem robert_nickel_vs_ashley : 
  robert_chocolates + nickel_chocolates - ashley_chocolates = 0 :=
by sorry

end NUMINAMATH_CALUDE_robert_nickel_vs_ashley_l1470_147082


namespace NUMINAMATH_CALUDE_exists_diverse_line_l1470_147001

/-- Represents a 17x17 table with integers from 1 to 17 -/
def Table := Fin 17 → Fin 17 → Fin 17

/-- Predicate to check if a table is valid according to the problem conditions -/
def is_valid_table (t : Table) : Prop :=
  ∀ n : Fin 17, (Finset.univ.filter (λ (i : Fin 17 × Fin 17) => t i.1 i.2 = n)).card = 17

/-- Counts the number of different elements in a list -/
def count_different (l : List (Fin 17)) : Nat :=
  (l.toFinset).card

/-- Theorem stating the existence of a row or column with at least 5 different numbers -/
theorem exists_diverse_line (t : Table) (h : is_valid_table t) :
  (∃ i : Fin 17, count_different (List.ofFn (λ j => t i j)) ≥ 5) ∨
  (∃ j : Fin 17, count_different (List.ofFn (λ i => t i j)) ≥ 5) := by
  sorry

end NUMINAMATH_CALUDE_exists_diverse_line_l1470_147001


namespace NUMINAMATH_CALUDE_sum_of_roots_cubic_l1470_147084

theorem sum_of_roots_cubic : ∃ (a b c : ℝ), 
  (a^3 + 2*a^2 + a - 1 = 0) ∧ 
  (b^3 + 2*b^2 + b - 1 = 0) ∧ 
  (c^3 + 2*c^2 + c - 1 = 0) ∧ 
  (a + b + c = -2) := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_cubic_l1470_147084


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l1470_147009

theorem simplify_and_evaluate_expression :
  let x : ℝ := Real.sqrt 5 - 2
  (2 / (x^2 - 4)) / (1 - x / (x - 2)) = -Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l1470_147009


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1470_147036

theorem sufficient_not_necessary (a b : ℝ) :
  (a < b ∧ b < 0 → a^2 > b^2) ∧
  ¬(a^2 > b^2 → a < b ∧ b < 0) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1470_147036


namespace NUMINAMATH_CALUDE_triangle_area_l1470_147099

theorem triangle_area (a b c : ℝ) (A B C : ℝ) : 
  c^2 = (a - b)^2 + 6 →
  C = π/3 →
  (1/2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1470_147099


namespace NUMINAMATH_CALUDE_double_yolked_eggs_l1470_147095

/-- Given a carton of eggs with some double-yolked eggs, calculate the number of double-yolked eggs. -/
theorem double_yolked_eggs (total_eggs : ℕ) (total_yolks : ℕ) (double_yolked : ℕ) : 
  total_eggs = 12 → total_yolks = 17 → double_yolked = 5 → 
  2 * double_yolked + (total_eggs - double_yolked) = total_yolks := by
  sorry

#check double_yolked_eggs

end NUMINAMATH_CALUDE_double_yolked_eggs_l1470_147095


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l1470_147003

/-- Given a geometric series with first term a and common ratio r,
    prove that if the sum of the series is 20 and the sum of terms
    with odd powers of r is 8, then r = √(11/12) -/
theorem geometric_series_ratio (a r : ℝ) (h₁ : a ≠ 0) (h₂ : |r| < 1) :
  (a / (1 - r) = 20) →
  (a * r / (1 - r^2) = 8) →
  r = Real.sqrt (11/12) := by
sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l1470_147003


namespace NUMINAMATH_CALUDE_orientation_count_equals_product_of_combinations_l1470_147098

/-- The number of ways to orient 40 unit segments for zero sum --/
def orientationCount : ℕ := sorry

/-- The total number of unit segments --/
def totalSegments : ℕ := 40

/-- The number of horizontal (or vertical) segments --/
def segmentsPerDirection : ℕ := 20

/-- The number of segments that need to be positive in each direction for zero sum --/
def positiveSegmentsPerDirection : ℕ := 10

theorem orientation_count_equals_product_of_combinations : 
  orientationCount = Nat.choose segmentsPerDirection positiveSegmentsPerDirection * 
                     Nat.choose segmentsPerDirection positiveSegmentsPerDirection := by sorry

end NUMINAMATH_CALUDE_orientation_count_equals_product_of_combinations_l1470_147098


namespace NUMINAMATH_CALUDE_quadratic_sum_l1470_147002

/-- The quadratic function we're working with -/
def f (x : ℝ) : ℝ := 8*x^2 + 48*x + 200

/-- The general form of a quadratic after completing the square -/
def g (a b c x : ℝ) : ℝ := a*(x+b)^2 + c

theorem quadratic_sum (a b c : ℝ) : 
  (∀ x, f x = g a b c x) → a + b + c = 139 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l1470_147002


namespace NUMINAMATH_CALUDE_cyclist_speed_problem_l1470_147079

theorem cyclist_speed_problem (distance : ℝ) (speed_diff : ℝ) (time_diff : ℝ) :
  distance = 195 →
  speed_diff = 4 →
  time_diff = 1 →
  ∃ (v : ℝ),
    v > 0 ∧
    distance / v = distance / (v - speed_diff) - time_diff ∧
    v = 30 :=
by sorry

end NUMINAMATH_CALUDE_cyclist_speed_problem_l1470_147079


namespace NUMINAMATH_CALUDE_profit_sharing_theorem_l1470_147073

/-- Represents the profit share of a business partner -/
structure ProfitShare where
  ratio : Float
  amount : Float

/-- Calculates the remaining amount after a purchase -/
def remainingAmount (share : ProfitShare) (purchase : Float) : Float :=
  share.amount - purchase

theorem profit_sharing_theorem 
  (mike johnson amy : ProfitShare)
  (mike_purchase amy_purchase : Float)
  (h1 : mike.ratio = 2.5)
  (h2 : johnson.ratio = 5.2)
  (h3 : amy.ratio = 3.8)
  (h4 : johnson.amount = 3120)
  (h5 : mike_purchase = 200)
  (h6 : amy_purchase = 150)
  : remainingAmount mike mike_purchase + remainingAmount amy amy_purchase = 3430 := by
  sorry

end NUMINAMATH_CALUDE_profit_sharing_theorem_l1470_147073


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1470_147027

theorem p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, x > 2 → x^2 > 4) ∧
  (∃ x : ℝ, x^2 > 4 ∧ x ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1470_147027


namespace NUMINAMATH_CALUDE_largest_positive_root_l1470_147072

theorem largest_positive_root (a₀ a₁ a₂ a₃ : ℝ) 
  (h₀ : |a₀| ≤ 3) (h₁ : |a₁| ≤ 3) (h₂ : |a₂| ≤ 3) (h₃ : |a₃| ≤ 3) :
  ∃ (r : ℝ), r = 3 ∧ 
  (∀ (x : ℝ), x > r → ∀ (b₀ b₁ b₂ b₃ : ℝ), 
    |b₀| ≤ 3 → |b₁| ≤ 3 → |b₂| ≤ 3 → |b₃| ≤ 3 → 
    x^4 + b₃*x^3 + b₂*x^2 + b₁*x + b₀ ≠ 0) ∧
  (∃ (c₀ c₁ c₂ c₃ : ℝ), 
    |c₀| ≤ 3 ∧ |c₁| ≤ 3 ∧ |c₂| ≤ 3 ∧ |c₃| ≤ 3 ∧ 
    r^4 + c₃*r^3 + c₂*r^2 + c₁*r + c₀ = 0) :=
by sorry

end NUMINAMATH_CALUDE_largest_positive_root_l1470_147072


namespace NUMINAMATH_CALUDE_compute_custom_op_l1470_147018

-- Define the custom operation
def custom_op (a b : ℚ) : ℚ := (a + b) / (a - b)

-- State the theorem
theorem compute_custom_op : custom_op (custom_op 8 6) 2 = 9 / 5 := by
  sorry

end NUMINAMATH_CALUDE_compute_custom_op_l1470_147018


namespace NUMINAMATH_CALUDE_mod_31_equivalence_l1470_147029

theorem mod_31_equivalence : ∃! n : ℤ, 0 ≤ n ∧ n < 31 ∧ 78256 ≡ n [ZMOD 31] ∧ n = 19 := by
  sorry

end NUMINAMATH_CALUDE_mod_31_equivalence_l1470_147029


namespace NUMINAMATH_CALUDE_outside_circle_inequality_l1470_147012

/-- A circle in the xy-plane -/
structure Circle where
  D : ℝ
  E : ℝ
  F : ℝ

/-- A point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point is outside a circle -/
def is_outside (p : Point) (c : Circle) : Prop :=
  (p.x + c.D/2)^2 + (p.y + c.E/2)^2 > (c.D^2 + c.E^2 - 4*c.F)/4

/-- The main theorem -/
theorem outside_circle_inequality (p : Point) (c : Circle) :
  is_outside p c → p.x^2 + p.y^2 + c.D*p.x + c.E*p.y + c.F > 0 := by
  sorry

end NUMINAMATH_CALUDE_outside_circle_inequality_l1470_147012


namespace NUMINAMATH_CALUDE_parabola_vertex_coordinates_l1470_147062

/-- The vertex coordinates of the parabola y = x^2 - 4x + 3 are (2, -1) -/
theorem parabola_vertex_coordinates :
  let f : ℝ → ℝ := λ x ↦ x^2 - 4*x + 3
  ∃ x y : ℝ, (x = 2 ∧ y = -1) ∧
    (∀ t : ℝ, f t ≥ f x) ∧
    (y = f x) :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_coordinates_l1470_147062


namespace NUMINAMATH_CALUDE_max_a_value_exists_max_a_l1470_147010

theorem max_a_value (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 12, x^2 + 25 + |x^3 - 5*x^2| ≥ a*x) → 
  a ≤ 10 :=
by sorry

theorem exists_max_a : 
  ∃ a : ℝ, a = 10 ∧ 
  (∀ x ∈ Set.Icc 1 12, x^2 + 25 + |x^3 - 5*x^2| ≥ a*x) ∧
  ∀ b > a, ∃ x ∈ Set.Icc 1 12, x^2 + 25 + |x^3 - 5*x^2| < b*x :=
by sorry

end NUMINAMATH_CALUDE_max_a_value_exists_max_a_l1470_147010


namespace NUMINAMATH_CALUDE_people_in_room_l1470_147005

theorem people_in_room (empty_chairs : ℕ) 
  (h1 : empty_chairs = 5)
  (h2 : ∃ (total_chairs : ℕ), empty_chairs = total_chairs / 5)
  (h3 : ∃ (seated_people : ℕ) (total_people : ℕ), 
    seated_people = 4 * total_chairs / 5 ∧
    seated_people = 5 * total_people / 8) :
  ∃ (total_people : ℕ), total_people = 32 := by
sorry

end NUMINAMATH_CALUDE_people_in_room_l1470_147005


namespace NUMINAMATH_CALUDE_electricity_price_correct_l1470_147034

/-- The electricity price per kWh in Coco's town -/
def electricity_price : ℝ := 0.1

/-- Coco's oven consumption rate in kWh -/
def oven_consumption_rate : ℝ := 2.4

/-- The number of hours Coco used his oven -/
def hours_used : ℝ := 25

/-- The amount Coco paid for using his oven -/
def amount_paid : ℝ := 6

/-- Theorem stating that the electricity price is correct -/
theorem electricity_price_correct : 
  electricity_price = amount_paid / (oven_consumption_rate * hours_used) :=
by sorry

end NUMINAMATH_CALUDE_electricity_price_correct_l1470_147034


namespace NUMINAMATH_CALUDE_dave_total_rides_l1470_147052

/-- The number of rides Dave went on the first day -/
def first_day_rides : ℕ := 4

/-- The number of rides Dave went on the second day -/
def second_day_rides : ℕ := 3

/-- The total number of rides Dave went on over two days -/
def total_rides : ℕ := first_day_rides + second_day_rides

theorem dave_total_rides :
  total_rides = 7 := by sorry

end NUMINAMATH_CALUDE_dave_total_rides_l1470_147052


namespace NUMINAMATH_CALUDE_chess_team_photo_arrangements_l1470_147024

/-- The number of ways to arrange a chess team in a line -/
def chessTeamArrangements (numBoys numGirls : ℕ) : ℕ :=
  Nat.factorial numGirls * Nat.factorial numBoys

/-- Theorem: There are 12 ways to arrange 2 boys and 3 girls in a line
    with girls in the middle and boys on the ends -/
theorem chess_team_photo_arrangements :
  chessTeamArrangements 2 3 = 12 := by
  sorry

#eval chessTeamArrangements 2 3

end NUMINAMATH_CALUDE_chess_team_photo_arrangements_l1470_147024


namespace NUMINAMATH_CALUDE_possible_values_of_a_l1470_147087

theorem possible_values_of_a (a b c : ℤ) :
  (∀ x : ℤ, (x - a) * (x - 10) + 1 = (x + b) * (x + c)) →
  (a = 8 ∨ a = 12) :=
by sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l1470_147087


namespace NUMINAMATH_CALUDE_valid_pairs_characterization_l1470_147068

def is_single_digit (n : ℕ) : Prop := 1 < n ∧ n < 10

def product_contains_factor (a b : ℕ) : Prop :=
  let product := a * b
  (product % 10 = a ∨ product % 10 = b ∨ product / 10 = a ∨ product / 10 = b)

def valid_pairs : Set (ℕ × ℕ) :=
  {p | is_single_digit p.1 ∧ is_single_digit p.2 ∧ product_contains_factor p.1 p.2}

theorem valid_pairs_characterization :
  valid_pairs = {(5, 3), (5, 5), (5, 7), (5, 9), (6, 2), (6, 4), (6, 6), (6, 8)} := by sorry

end NUMINAMATH_CALUDE_valid_pairs_characterization_l1470_147068


namespace NUMINAMATH_CALUDE_gilbert_herb_count_l1470_147067

/-- Represents the number of herb plants Gilbert has at different stages of spring -/
structure HerbGarden where
  initial_basil : ℕ
  initial_parsley : ℕ
  initial_mint : ℕ
  extra_basil : ℕ
  eaten_mint : ℕ

/-- Calculates the final number of herb plants in Gilbert's garden -/
def final_herb_count (garden : HerbGarden) : ℕ :=
  garden.initial_basil + garden.initial_parsley + garden.initial_mint + garden.extra_basil - garden.eaten_mint

/-- Theorem stating that Gilbert had 5 herb plants at the end of spring -/
theorem gilbert_herb_count :
  ∀ (garden : HerbGarden),
    garden.initial_basil = 3 →
    garden.initial_parsley = 1 →
    garden.initial_mint = 2 →
    garden.extra_basil = 1 →
    garden.eaten_mint = 2 →
    final_herb_count garden = 5 := by
  sorry


end NUMINAMATH_CALUDE_gilbert_herb_count_l1470_147067


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l1470_147061

/-- Given an arithmetic sequence {a_n} with common difference d ≠ 0,
    if a_1, a_3, and a_9 form a geometric sequence,
    then (a_1 + a_3 + a_9) / (a_2 + a_4 + a_10) = 13/16 -/
theorem arithmetic_geometric_ratio 
  (a : ℕ → ℚ) 
  (d : ℚ)
  (h1 : d ≠ 0)
  (h2 : ∀ n : ℕ, a (n + 1) = a n + d)
  (h3 : (a 3 - a 1) * (a 9 - a 3) = (a 3 - a 1) * (a 3 - a 1)) :
  (a 1 + a 3 + a 9) / (a 2 + a 4 + a 10) = 13 / 16 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l1470_147061


namespace NUMINAMATH_CALUDE_binomial_12_3_l1470_147094

theorem binomial_12_3 : Nat.choose 12 3 = 220 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_3_l1470_147094


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1470_147030

theorem quadratic_equation_roots : ∃ (p q : ℤ),
  (∃ (x₁ x₂ : ℤ), 
    x₁ > 0 ∧ x₂ > 0 ∧
    x₁^2 + p*x₁ + q = 0 ∧
    x₂^2 + p*x₂ + q = 0 ∧
    p + q = 28) →
  (∃ (x₁ x₂ : ℤ), 
    (x₁ = 30 ∧ x₂ = 2) ∨ (x₁ = 2 ∧ x₂ = 30)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1470_147030


namespace NUMINAMATH_CALUDE_roots_count_lower_bound_l1470_147089

def count_roots (f : ℝ → ℝ) (a b : ℝ) : ℕ :=
  sorry

theorem roots_count_lower_bound
  (f : ℝ → ℝ)
  (h1 : ∀ x, f (3 + x) = f (3 - x))
  (h2 : ∀ x, f (9 + x) = f (9 - x))
  (h3 : f 1 = 0) :
  count_roots f (-1000) 1000 ≥ 334 := by
  sorry

end NUMINAMATH_CALUDE_roots_count_lower_bound_l1470_147089


namespace NUMINAMATH_CALUDE_fourth_root_of_polynomial_l1470_147019

theorem fourth_root_of_polynomial (a b : ℝ) : 
  (∀ x : ℝ, a * x^4 + (a + 2*b) * x^3 + (b - 3*a) * x^2 + (2*a - 6) * x + (7 - a) = 0 ↔ 
    x = 1 ∨ x = -1 ∨ x = 2 ∨ x = -2) → 
  ∃ x : ℝ, x = -2 ∧ a * x^4 + (a + 2*b) * x^3 + (b - 3*a) * x^2 + (2*a - 6) * x + (7 - a) = 0 :=
by sorry

end NUMINAMATH_CALUDE_fourth_root_of_polynomial_l1470_147019


namespace NUMINAMATH_CALUDE_power_problem_l1470_147008

theorem power_problem (a m n : ℕ) (h1 : a ^ m = 3) (h2 : a ^ n = 2) :
  a ^ (3 * m + 2 * n) = 108 := by
  sorry

end NUMINAMATH_CALUDE_power_problem_l1470_147008


namespace NUMINAMATH_CALUDE_triangle_area_inequality_l1470_147046

/-- Triangle type with side lengths and area -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  S : ℝ
  valid : 0 < a ∧ 0 < b ∧ 0 < c ∧ a < b + c ∧ b < a + c ∧ c < a + b

/-- Theorem statement for triangle inequality -/
theorem triangle_area_inequality (t : Triangle) :
  t.S ≤ (t.a^2 + t.b^2 + t.c^2) / (4 * Real.sqrt 3) ∧
  (t.S = (t.a^2 + t.b^2 + t.c^2) / (4 * Real.sqrt 3) ↔ t.a = t.b ∧ t.b = t.c) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_inequality_l1470_147046


namespace NUMINAMATH_CALUDE_day_crew_loading_fraction_l1470_147085

/-- The fraction of boxes loaded by the day crew given the relative capacities of night and day crews -/
theorem day_crew_loading_fraction 
  (D : ℝ) -- number of boxes loaded by each day crew worker
  (W : ℝ) -- number of workers in the day crew
  (h1 : D > 0) -- assume positive number of boxes
  (h2 : W > 0) -- assume positive number of workers
  : (D * W) / ((D * W) + ((3/4 * D) * (5/6 * W))) = 8/13 := by
  sorry

end NUMINAMATH_CALUDE_day_crew_loading_fraction_l1470_147085


namespace NUMINAMATH_CALUDE_units_digit_17_39_l1470_147058

theorem units_digit_17_39 : (17^39) % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_17_39_l1470_147058


namespace NUMINAMATH_CALUDE_union_equals_interval_l1470_147083

-- Define sets A and B
def A : Set ℝ := {x | 0 < x ∧ x < 16}
def B : Set ℝ := {y | -4 < 4 * y ∧ 4 * y < 16}

-- Define the open interval (-1, 16)
def openInterval : Set ℝ := {z | -1 < z ∧ z < 16}

-- Theorem statement
theorem union_equals_interval : A ∪ B = openInterval := by
  sorry

end NUMINAMATH_CALUDE_union_equals_interval_l1470_147083


namespace NUMINAMATH_CALUDE_rope_cutting_problem_l1470_147051

theorem rope_cutting_problem :
  let rope1 : ℕ := 44
  let rope2 : ℕ := 54
  let rope3 : ℕ := 74
  Nat.gcd rope1 (Nat.gcd rope2 rope3) = 2 := by
sorry

end NUMINAMATH_CALUDE_rope_cutting_problem_l1470_147051


namespace NUMINAMATH_CALUDE_circle_line_intersection_l1470_147031

theorem circle_line_intersection :
  ∃! p : ℝ × ℝ, p.1^2 + p.2^2 = 16 ∧ p.1 = 4 :=
by sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l1470_147031


namespace NUMINAMATH_CALUDE_jacobs_gift_budget_l1470_147097

theorem jacobs_gift_budget (total_budget : ℕ) (num_friends : ℕ) (friend_gift_cost : ℕ) (num_parents : ℕ) :
  total_budget = 100 →
  num_friends = 8 →
  friend_gift_cost = 9 →
  num_parents = 2 →
  (total_budget - num_friends * friend_gift_cost) / num_parents = 14 := by
  sorry

end NUMINAMATH_CALUDE_jacobs_gift_budget_l1470_147097


namespace NUMINAMATH_CALUDE_baseball_runs_proof_l1470_147066

theorem baseball_runs_proof (sequence : Fin 6 → ℕ) 
  (h1 : ∃ i, sequence i = 1)
  (h2 : ∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ sequence i = 5 ∧ sequence j = 5 ∧ sequence k = 5)
  (h3 : ∃ i j, i ≠ j ∧ sequence i = sequence j)
  (h4 : (Finset.sum Finset.univ (λ i => sequence i)) / 6 = 4) :
  ∃ i j, i ≠ j ∧ sequence i = sequence j ∧ sequence i = 4 := by
  sorry

end NUMINAMATH_CALUDE_baseball_runs_proof_l1470_147066


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l1470_147032

def is_prime (n : ℤ) : Prop := n > 1 ∧ ∀ m : ℤ, 1 < m → m < n → ¬(n % m = 0)

theorem quadratic_root_problem (a b c : ℤ) (m n : ℕ) :
  (∀ x : ℤ, a * x^2 + b * x + c = 0 ↔ x = m ∨ x = n) →
  m ≠ n →
  m > 0 →
  n > 0 →
  is_prime (a + b + c) →
  (∃ x : ℤ, a * x^2 + b * x + c = -55) →
  m = 2 →
  n = 17 :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l1470_147032


namespace NUMINAMATH_CALUDE_hugo_climb_count_l1470_147028

def hugo_mountain_elevation : ℕ := 10000
def boris_mountain_elevation : ℕ := hugo_mountain_elevation - 2500
def boris_climb_count : ℕ := 4

theorem hugo_climb_count : 
  ∃ (x : ℕ), x * hugo_mountain_elevation = boris_climb_count * boris_mountain_elevation ∧ x = 3 :=
by sorry

end NUMINAMATH_CALUDE_hugo_climb_count_l1470_147028


namespace NUMINAMATH_CALUDE_greatest_n_for_perfect_cube_l1470_147090

def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

def product_of_sums (n : ℕ) : ℕ := 
  (sum_of_squares n) * (sum_of_squares (2*n) - sum_of_squares n)

theorem greatest_n_for_perfect_cube : 
  ∃ n : ℕ, n = 2016 ∧ 
    n ≤ 2050 ∧ 
    is_perfect_cube (product_of_sums n) ∧
    ∀ m : ℕ, m > n → m ≤ 2050 → ¬(is_perfect_cube (product_of_sums m)) :=
sorry

end NUMINAMATH_CALUDE_greatest_n_for_perfect_cube_l1470_147090


namespace NUMINAMATH_CALUDE_root_value_theorem_l1470_147064

theorem root_value_theorem (m : ℝ) : 2 * m^2 - 3 * m - 1 = 0 → 4 * m^2 - 6 * m + 2021 = 2023 := by
  sorry

end NUMINAMATH_CALUDE_root_value_theorem_l1470_147064


namespace NUMINAMATH_CALUDE_line_through_points_l1470_147070

/-- Given a line y = ax + b passing through points (3, 7) and (6, 19), prove that a - b = 9 -/
theorem line_through_points (a b : ℝ) : 
  (7 : ℝ) = a * 3 + b ∧ (19 : ℝ) = a * 6 + b → a - b = 9 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l1470_147070


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1470_147092

/-- In a right triangle with one angle of 30° and the side opposite to this angle
    having length 6, the length of the hypotenuse is 12. -/
theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →  -- Pythagorean theorem for right triangle
  a = 6 →  -- Length of the side opposite to 30° angle
  Real.cos (30 * π / 180) = b / c →  -- Cosine of 30° in terms of adjacent side and hypotenuse
  c = 12 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1470_147092


namespace NUMINAMATH_CALUDE_proposition_truth_values_l1470_147054

theorem proposition_truth_values (p q : Prop) 
  (h1 : p ∨ q)
  (h2 : ¬p)
  (h3 : ¬p) :
  ¬p ∧ q := by
  sorry

end NUMINAMATH_CALUDE_proposition_truth_values_l1470_147054


namespace NUMINAMATH_CALUDE_jeff_payment_correct_l1470_147049

/-- The amount Jeff paid when picking up the Halloween costumes -/
def jeff_payment (num_costumes : ℕ) 
                 (deposit_rate : ℝ) 
                 (price_increase : ℝ) 
                 (last_year_price : ℝ) 
                 (jeff_discount : ℝ) 
                 (friend_discount : ℝ) : ℝ :=
  let this_year_price := last_year_price * (1 + price_increase)
  let total_cost := num_costumes * this_year_price
  let discounts := jeff_discount * this_year_price + friend_discount * this_year_price
  let adjusted_cost := total_cost - discounts
  let deposit := deposit_rate * adjusted_cost
  adjusted_cost - deposit

/-- Theorem stating that Jeff's payment matches the calculated amount -/
theorem jeff_payment_correct : 
  jeff_payment 3 0.1 0.4 250 0.15 0.1 = 866.25 := by
  sorry


end NUMINAMATH_CALUDE_jeff_payment_correct_l1470_147049


namespace NUMINAMATH_CALUDE_election_votes_count_l1470_147053

theorem election_votes_count :
  ∀ (total_votes : ℕ) 
    (candidate1_valid_votes candidate2_valid_votes invalid_votes : ℕ),
  candidate1_valid_votes + candidate2_valid_votes + invalid_votes = total_votes →
  candidate1_valid_votes = (55 * (candidate1_valid_votes + candidate2_valid_votes)) / 100 →
  invalid_votes = (20 * total_votes) / 100 →
  candidate2_valid_votes = 2700 →
  total_votes = 7500 :=
by
  sorry

end NUMINAMATH_CALUDE_election_votes_count_l1470_147053


namespace NUMINAMATH_CALUDE_elderly_arrangement_theorem_l1470_147056

/-- The number of ways to arrange n distinct objects in a row -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to choose k objects from n distinct objects, where order matters -/
def arrangements (n k : ℕ) : ℕ := 
  if k ≤ n then
    Nat.factorial n / Nat.factorial (n - k)
  else
    0

/-- The number of ways to arrange volunteers and elderly people with given constraints -/
def arrangement_count (volunteers elderly : ℕ) : ℕ :=
  permutations volunteers * arrangements (volunteers + 1) elderly

theorem elderly_arrangement_theorem :
  arrangement_count 4 2 = 480 := by
  sorry

end NUMINAMATH_CALUDE_elderly_arrangement_theorem_l1470_147056


namespace NUMINAMATH_CALUDE_fraction_problem_l1470_147016

theorem fraction_problem (N : ℚ) (h : (1/4) * (1/3) * (2/5) * N = 30) :
  ∃ F : ℚ, F * N = 120 ∧ F = 2/15 := by sorry

end NUMINAMATH_CALUDE_fraction_problem_l1470_147016


namespace NUMINAMATH_CALUDE_symmetric_point_l1470_147023

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The origin point (0,0) -/
def origin : Point2D := ⟨0, 0⟩

/-- Function to check if a point is the midpoint of two other points -/
def isMidpoint (m : Point2D) (p1 : Point2D) (p2 : Point2D) : Prop :=
  m.x = (p1.x + p2.x) / 2 ∧ m.y = (p1.y + p2.y) / 2

/-- Function to check if two points are symmetric with respect to the origin -/
def isSymmetricToOrigin (p1 : Point2D) (p2 : Point2D) : Prop :=
  isMidpoint origin p1 p2

/-- Theorem: The point (2,-3) is symmetric to (-2,3) with respect to the origin -/
theorem symmetric_point : 
  isSymmetricToOrigin ⟨-2, 3⟩ ⟨2, -3⟩ := by
  sorry


end NUMINAMATH_CALUDE_symmetric_point_l1470_147023


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l1470_147013

theorem triangle_angle_measure (A B C : Real) (BC AC : Real) :
  BC = Real.sqrt 3 →
  AC = Real.sqrt 2 →
  A = π / 3 →
  B = π / 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l1470_147013


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l1470_147020

-- Problem 1
theorem problem_1 (a b : ℝ) : 
  (abs a = 5) → 
  (abs b = 3) → 
  (abs (a - b) = b - a) → 
  ((a - b = -8) ∨ (a - b = -2)) :=
sorry

-- Problem 2
theorem problem_2 (a b c d m : ℝ) :
  (a + b = 0) →
  (c * d = 1) →
  (abs m = 2) →
  (abs (a + b) / m - c * d + m^2 = 3) :=
sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l1470_147020


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_bound_l1470_147039

theorem quadratic_roots_sum_bound (p : ℝ) (r₁ r₂ : ℝ) : 
  (∀ x : ℝ, x^2 + p*x + 12 = 0 ↔ x = r₁ ∨ x = r₂) →
  |r₁ + r₂| > 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_bound_l1470_147039


namespace NUMINAMATH_CALUDE_valid_hexagonal_star_exists_l1470_147048

/-- A configuration of numbers in the hexagonal star format -/
structure HexagonalStar where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  e : Nat
  f : Nat
  g : Nat

/-- Check if a number is 2, 3, or 5 -/
def isValidNumber (n : Nat) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5

/-- Check if all numbers in the configuration are valid -/
def allValidNumbers (hs : HexagonalStar) : Prop :=
  isValidNumber hs.a ∧ isValidNumber hs.b ∧ isValidNumber hs.c ∧
  isValidNumber hs.d ∧ isValidNumber hs.e ∧ isValidNumber hs.f ∧
  isValidNumber hs.g

/-- Check if all triangles have the same sum -/
def allTrianglesSameSum (hs : HexagonalStar) : Prop :=
  let sum1 := hs.a + hs.b + hs.g
  let sum2 := hs.b + hs.c + hs.g
  let sum3 := hs.c + hs.d + hs.g
  let sum4 := hs.d + hs.e + hs.g
  let sum5 := hs.e + hs.f + hs.g
  let sum6 := hs.f + hs.a + hs.g
  sum1 = sum2 ∧ sum2 = sum3 ∧ sum3 = sum4 ∧ sum4 = sum5 ∧ sum5 = sum6

/-- Theorem: There exists a valid hexagonal star configuration -/
theorem valid_hexagonal_star_exists : ∃ (hs : HexagonalStar), 
  allValidNumbers hs ∧ allTrianglesSameSum hs := by
  sorry

end NUMINAMATH_CALUDE_valid_hexagonal_star_exists_l1470_147048


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1470_147081

/-- The function f(x) = α^(x-2) - 1 always passes through the point (2, 0) for any α > 0 and α ≠ 1 -/
theorem fixed_point_of_exponential_function (α : ℝ) (h1 : α > 0) (h2 : α ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ α^(x - 2) - 1
  f 2 = 0 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1470_147081


namespace NUMINAMATH_CALUDE_hair_cut_first_day_l1470_147080

/-- The amount of hair cut off on the first day, given the total amount cut off and the amount cut off on the second day. -/
theorem hair_cut_first_day (total : ℚ) (second_day : ℚ) (h1 : total = 0.875) (h2 : second_day = 0.5) :
  total - second_day = 0.375 := by
  sorry

end NUMINAMATH_CALUDE_hair_cut_first_day_l1470_147080


namespace NUMINAMATH_CALUDE_ellipse_equation_l1470_147011

/-- Given an ellipse with center at the origin, foci on the x-axis, 
    major axis length of 4, and minor axis length of 2, 
    its equation is x²/4 + y² = 1 -/
theorem ellipse_equation (x y : ℝ) : 
  let center := (0 : ℝ × ℝ)
  let major_axis := 4
  let minor_axis := 2
  let foci_on_x_axis := true
  x^2 / 4 + y^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l1470_147011


namespace NUMINAMATH_CALUDE_square_window_side_length_l1470_147021

/-- Given a square window opening formed by two rectangular frames, 
    prove that the side length of the square is 5 when the perimeter 
    of the left frame is 14 and the perimeter of the right frame is 16. -/
theorem square_window_side_length 
  (a : ℝ) -- side length of the square window
  (b : ℝ) -- width of the left rectangular frame
  (h1 : 2 * a + 2 * b = 14) -- perimeter of the left frame
  (h2 : 4 * a - 2 * b = 16) -- perimeter of the right frame
  : a = 5 := by
  sorry

end NUMINAMATH_CALUDE_square_window_side_length_l1470_147021


namespace NUMINAMATH_CALUDE_sam_tuna_change_sam_change_proof_l1470_147026

/-- Calculates the change Sam received when buying tuna cans. -/
theorem sam_tuna_change (num_cans : ℕ) (num_coupons : ℕ) (coupon_value : ℕ) 
  (can_cost : ℕ) (paid_amount : ℕ) : ℕ :=
  let total_discount := num_coupons * coupon_value
  let total_cost := num_cans * can_cost
  let actual_paid := total_cost - total_discount
  paid_amount - actual_paid

/-- Proves that Sam received $5.50 in change. -/
theorem sam_change_proof : 
  sam_tuna_change 9 5 25 175 2000 = 550 := by
  sorry

end NUMINAMATH_CALUDE_sam_tuna_change_sam_change_proof_l1470_147026


namespace NUMINAMATH_CALUDE_total_rainfall_sum_l1470_147045

/-- The rainfall recorded on Monday in centimeters -/
def monday_rainfall : ℝ := 0.17

/-- The rainfall recorded on Tuesday in centimeters -/
def tuesday_rainfall : ℝ := 0.42

/-- The rainfall recorded on Wednesday in centimeters -/
def wednesday_rainfall : ℝ := 0.08

/-- The total rainfall recorded over the three days -/
def total_rainfall : ℝ := monday_rainfall + tuesday_rainfall + wednesday_rainfall

/-- Theorem stating that the total rainfall is equal to 0.67 cm -/
theorem total_rainfall_sum : total_rainfall = 0.67 := by sorry

end NUMINAMATH_CALUDE_total_rainfall_sum_l1470_147045


namespace NUMINAMATH_CALUDE_chord_length_squared_l1470_147075

/-- Two circles with given properties and intersecting chords --/
structure CircleConfiguration where
  -- First circle radius
  r1 : ℝ
  -- Second circle radius
  r2 : ℝ
  -- Distance between circle centers
  d : ℝ
  -- Length of chord QP
  x : ℝ
  -- Ensure the configuration is valid
  h1 : r1 = 10
  h2 : r2 = 7
  h3 : d = 15
  -- QP = PR = PS = PT
  h4 : ∀ (chord : ℝ), chord = x → (chord = QP ∨ chord = PR ∨ chord = PS ∨ chord = PT)

/-- The theorem stating that the square of QP's length is 265 --/
theorem chord_length_squared (config : CircleConfiguration) : config.x^2 = 265 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_squared_l1470_147075


namespace NUMINAMATH_CALUDE_infinitely_many_perfect_squares_l1470_147043

/-- An arithmetic sequence of natural numbers -/
def arithmeticSequence (a d : ℕ) (n : ℕ) : ℕ := a + n * d

/-- Predicate for perfect squares -/
def isPerfectSquare (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

theorem infinitely_many_perfect_squares
  (a d : ℕ) -- First term and common difference of the sequence
  (h : ∃ n₀ : ℕ, isPerfectSquare (arithmeticSequence a d n₀)) :
  ∀ m : ℕ, ∃ n > m, isPerfectSquare (arithmeticSequence a d n) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_perfect_squares_l1470_147043


namespace NUMINAMATH_CALUDE_num_selections_with_A_or_B_l1470_147000

/-- The number of key projects -/
def num_key_projects : ℕ := 4

/-- The number of general projects -/
def num_general_projects : ℕ := 6

/-- The number of key projects to be selected -/
def select_key : ℕ := 2

/-- The number of general projects to be selected -/
def select_general : ℕ := 2

/-- Theorem stating the number of selection methods with at least one of A or B selected -/
theorem num_selections_with_A_or_B : 
  (Nat.choose (num_key_projects - 1) (select_key - 1) * Nat.choose (num_general_projects - 1) select_general) +
  (Nat.choose (num_key_projects - 1) select_key * Nat.choose (num_general_projects - 1) (select_general - 1)) +
  (Nat.choose (num_key_projects - 1) (select_key - 1) * Nat.choose (num_general_projects - 1) (select_general - 1)) = 60 := by
  sorry


end NUMINAMATH_CALUDE_num_selections_with_A_or_B_l1470_147000


namespace NUMINAMATH_CALUDE_largest_x_abs_equation_l1470_147077

theorem largest_x_abs_equation : 
  ∀ x : ℝ, |x - 3| = 14.5 → x ≤ 17.5 ∧ ∃ y : ℝ, |y - 3| = 14.5 ∧ y = 17.5 := by
  sorry

end NUMINAMATH_CALUDE_largest_x_abs_equation_l1470_147077


namespace NUMINAMATH_CALUDE_final_points_count_l1470_147015

/-- The number of points after performing the insertion operation n times -/
def points_after_operations (initial_points : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0 => initial_points
  | k + 1 => 2 * points_after_operations initial_points k - 1

theorem final_points_count : points_after_operations 2010 3 = 16073 := by
  sorry

end NUMINAMATH_CALUDE_final_points_count_l1470_147015


namespace NUMINAMATH_CALUDE_consecutive_episodes_probability_l1470_147065

theorem consecutive_episodes_probability (n : ℕ) (h : n = 6) :
  let total_combinations := n.choose 2
  let consecutive_pairs := n - 1
  (consecutive_pairs : ℚ) / total_combinations = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_consecutive_episodes_probability_l1470_147065


namespace NUMINAMATH_CALUDE_innings_count_l1470_147088

/-- Represents the batting statistics of a batsman -/
structure BattingStats where
  n : ℕ                -- Total number of innings
  highest : ℕ          -- Highest score
  lowest : ℕ           -- Lowest score
  average : ℚ          -- Average score
  newAverage : ℚ       -- Average after excluding highest and lowest scores

/-- Theorem stating the conditions and the result to be proved -/
theorem innings_count (stats : BattingStats) : 
  stats.average = 50 ∧ 
  stats.highest - stats.lowest = 172 ∧
  stats.newAverage = stats.average - 2 ∧
  stats.highest = 174 →
  stats.n = 40 := by
  sorry


end NUMINAMATH_CALUDE_innings_count_l1470_147088


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1470_147063

theorem solution_set_inequality (x : ℝ) :
  (((1 - 2*x) / (3*x^2 - 4*x + 7)) ≥ 0) ↔ (x ≤ 1/2) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1470_147063


namespace NUMINAMATH_CALUDE_three_number_set_range_l1470_147086

theorem three_number_set_range (a b c : ℝ) : 
  a ≤ b ∧ b ≤ c ∧  -- Ordered set
  (a + b + c) / 3 = 6 ∧  -- Mean is 6
  b = 6 ∧  -- Median is 6
  a = 2  -- Smallest number is 2
  →
  c - a = 8 :=  -- Range is 8
by sorry

end NUMINAMATH_CALUDE_three_number_set_range_l1470_147086


namespace NUMINAMATH_CALUDE_star_polygon_external_intersection_angle_l1470_147096

/-- 
The angle at each intersection point outside a star-polygon with n points (n > 4) 
inscribed in a circle, given that each internal angle is (180(n-4))/n degrees.
-/
theorem star_polygon_external_intersection_angle (n : ℕ) (h : n > 4) : 
  let internal_angle := (180 * (n - 4)) / n
  (360 * (n - 4)) / n = 360 - 2 * (180 - internal_angle) := by
  sorry

#check star_polygon_external_intersection_angle

end NUMINAMATH_CALUDE_star_polygon_external_intersection_angle_l1470_147096


namespace NUMINAMATH_CALUDE_percentage_difference_l1470_147071

theorem percentage_difference (x y z : ℝ) (h1 : y = 1.2 * z) (h2 : z = 150) (h3 : x + y + z = 555) :
  (x - y) / y * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l1470_147071


namespace NUMINAMATH_CALUDE_cylinder_cone_surface_area_l1470_147047

/-- The total surface area of a cylinder topped with a cone -/
theorem cylinder_cone_surface_area (h_cyl h_cone r : ℝ) (h_cyl_pos : h_cyl > 0) (h_cone_pos : h_cone > 0) (r_pos : r > 0) :
  let cylinder_base_area := π * r^2
  let cylinder_lateral_area := 2 * π * r * h_cyl
  let cone_slant_height := Real.sqrt (r^2 + h_cone^2)
  let cone_lateral_area := π * r * cone_slant_height
  cylinder_base_area + cylinder_lateral_area + cone_lateral_area = 175 * π + 5 * π * Real.sqrt 89 :=
by sorry

end NUMINAMATH_CALUDE_cylinder_cone_surface_area_l1470_147047


namespace NUMINAMATH_CALUDE_solution_characterization_l1470_147033

def system_equations (n : ℕ) (x : ℕ → ℝ) : Prop :=
  (∀ i ∈ Finset.range n, 1 - x i * x ((i + 1) % n) = 0)

theorem solution_characterization (n : ℕ) (x : ℕ → ℝ) (hn : n > 0) :
  system_equations n x →
  (n % 2 = 1 ∧ (∀ i ∈ Finset.range n, x i = 1 ∨ x i = -1)) ∨
  (n % 2 = 0 ∧ ∃ a : ℝ, a ≠ 0 ∧
    x 0 = a ∧ x 1 = 1 / a ∧
    ∀ i ∈ Finset.range (n - 2), x (i + 2) = x i) :=
by sorry

end NUMINAMATH_CALUDE_solution_characterization_l1470_147033


namespace NUMINAMATH_CALUDE_harkamal_payment_l1470_147050

/-- Calculates the final amount paid after discount and tax --/
def calculate_final_amount (fruits : List (String × ℕ × ℕ)) (discount_rate : ℚ) (tax_rate : ℚ) : ℚ :=
  let total_cost := (fruits.map (λ (_, quantity, price) => quantity * price)).sum
  let discounted_total := total_cost * (1 - discount_rate)
  let final_amount := discounted_total * (1 + tax_rate)
  final_amount

/-- Theorem stating the final amount Harkamal paid --/
theorem harkamal_payment : 
  let fruits := [
    ("Grapes", 8, 70),
    ("Mangoes", 9, 55),
    ("Apples", 4, 40),
    ("Oranges", 6, 30),
    ("Pineapples", 2, 90),
    ("Cherries", 5, 100)
  ]
  let discount_rate : ℚ := 5 / 100
  let tax_rate : ℚ := 10 / 100
  calculate_final_amount fruits discount_rate tax_rate = 2168375 / 1000 := by
  sorry

#eval calculate_final_amount [
  ("Grapes", 8, 70),
  ("Mangoes", 9, 55),
  ("Apples", 4, 40),
  ("Oranges", 6, 30),
  ("Pineapples", 2, 90),
  ("Cherries", 5, 100)
] (5 / 100) (10 / 100)

end NUMINAMATH_CALUDE_harkamal_payment_l1470_147050


namespace NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l1470_147074

theorem condition_sufficient_not_necessary (a b : ℝ) :
  (∀ a b : ℝ, (a - b) * a^2 > 0 → a > b) ∧
  (∃ a b : ℝ, a > b ∧ ¬((a - b) * a^2 > 0)) := by
  sorry

end NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l1470_147074


namespace NUMINAMATH_CALUDE_second_scenario_cost_l1470_147035

/-- The cost of a single shirt -/
def shirt_cost : ℝ := sorry

/-- The cost of a single trouser -/
def trouser_cost : ℝ := sorry

/-- The cost of a single tie -/
def tie_cost : ℝ := sorry

/-- The first scenario: 6 shirts, 4 trousers, and 2 ties cost $80 -/
def scenario1 : Prop := 6 * shirt_cost + 4 * trouser_cost + 2 * tie_cost = 80

/-- The third scenario: 5 shirts, 3 trousers, and 2 ties cost $110 -/
def scenario3 : Prop := 5 * shirt_cost + 3 * trouser_cost + 2 * tie_cost = 110

/-- Theorem: Given scenario1 and scenario3, the cost of 4 shirts, 2 trousers, and 2 ties is $50 -/
theorem second_scenario_cost (h1 : scenario1) (h3 : scenario3) : 
  4 * shirt_cost + 2 * trouser_cost + 2 * tie_cost = 50 := by sorry

end NUMINAMATH_CALUDE_second_scenario_cost_l1470_147035


namespace NUMINAMATH_CALUDE_orthocenters_collinear_l1470_147006

-- Define the basic geometric objects
variable (A B C D O : Point)

-- Define the quadrilateral ABCD
def quadrilateral (A B C D : Point) : Prop := sorry

-- Define the inscribed circle O
def inscribedCircle (O : Point) (A B C D : Point) : Prop := sorry

-- Define the orthocenter of a triangle
def orthocenter (P Q R : Point) : Point := sorry

-- Define collinearity of points
def collinear (P Q R : Point) : Prop := sorry

-- State the theorem
theorem orthocenters_collinear 
  (h1 : quadrilateral A B C D) 
  (h2 : inscribedCircle O A B C D) : 
  collinear 
    (orthocenter O A B) 
    (orthocenter O B C) 
    (orthocenter O C D) ∧ 
  collinear 
    (orthocenter O C D) 
    (orthocenter O D A) 
    (orthocenter O A B) :=
sorry

end NUMINAMATH_CALUDE_orthocenters_collinear_l1470_147006


namespace NUMINAMATH_CALUDE_triangle_longest_side_l1470_147007

theorem triangle_longest_side (y : ℝ) : 
  10 + (y + 6) + (3 * y + 5) = 49 →
  max 10 (max (y + 6) (3 * y + 5)) = 26 :=
by sorry

end NUMINAMATH_CALUDE_triangle_longest_side_l1470_147007


namespace NUMINAMATH_CALUDE_intersection_M_N_l1470_147017

def M : Set ℝ := {x | ∃ y, y = Real.sqrt (2 - x^2)}
def N : Set ℝ := {x | ∃ y, y = x^2 - 1}

theorem intersection_M_N : M ∩ N = Set.Icc (-1) (Real.sqrt 2) := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1470_147017
