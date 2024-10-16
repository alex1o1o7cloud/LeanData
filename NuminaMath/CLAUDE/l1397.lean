import Mathlib

namespace NUMINAMATH_CALUDE_max_gcd_of_sum_1071_l1397_139737

theorem max_gcd_of_sum_1071 :
  ∃ (m : ℕ), m > 0 ∧ 
  (∀ (x y : ℕ), x > 0 → y > 0 → x + y = 1071 → Nat.gcd x y ≤ m) ∧
  (∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x + y = 1071 ∧ Nat.gcd x y = m) ∧
  m = 357 :=
sorry

end NUMINAMATH_CALUDE_max_gcd_of_sum_1071_l1397_139737


namespace NUMINAMATH_CALUDE_matinee_children_count_l1397_139714

/-- Proves the number of children at a movie theater matinee --/
theorem matinee_children_count :
  let child_price : ℚ := 9/2
  let adult_price : ℚ := 27/4
  let total_receipts : ℚ := 405
  ∀ (num_adults : ℕ),
    (child_price * (num_adults + 20 : ℚ) + adult_price * num_adults = total_receipts) →
    (num_adults + 20 = 48) :=
by
  sorry

#check matinee_children_count

end NUMINAMATH_CALUDE_matinee_children_count_l1397_139714


namespace NUMINAMATH_CALUDE_decreasing_cubic_implies_m_leq_neg_three_exists_m_leq_neg_three_not_decreasing_l1397_139773

/-- A function f : ℝ → ℝ is decreasing if for all x y, x < y implies f x > f y -/
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

/-- The cubic function f(x) = mx³ + 3x² - x + 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^3 + 3 * x^2 - x + 1

theorem decreasing_cubic_implies_m_leq_neg_three :
  ∀ m : ℝ, DecreasingFunction (f m) → m ≤ -3 :=
sorry

theorem exists_m_leq_neg_three_not_decreasing :
  ∃ m : ℝ, m ≤ -3 ∧ ¬(DecreasingFunction (f m)) :=
sorry

end NUMINAMATH_CALUDE_decreasing_cubic_implies_m_leq_neg_three_exists_m_leq_neg_three_not_decreasing_l1397_139773


namespace NUMINAMATH_CALUDE_sams_books_l1397_139735

theorem sams_books (tim_books sam_books total_books : ℕ) : 
  tim_books = 44 → 
  total_books = 96 → 
  total_books = tim_books + sam_books → 
  sam_books = 52 := by
sorry

end NUMINAMATH_CALUDE_sams_books_l1397_139735


namespace NUMINAMATH_CALUDE_adam_initial_money_l1397_139761

/-- The cost of the airplane in dollars -/
def airplane_cost : ℚ := 4.28

/-- The change Adam received in dollars -/
def change_received : ℚ := 0.72

/-- Adam's initial amount of money in dollars -/
def initial_money : ℚ := airplane_cost + change_received

theorem adam_initial_money : initial_money = 5 := by
  sorry

end NUMINAMATH_CALUDE_adam_initial_money_l1397_139761


namespace NUMINAMATH_CALUDE_product_equals_32_over_9_l1397_139721

/-- The repeating decimal 0.4444... --/
def repeating_four : ℚ := 4/9

/-- The product of the repeating decimal 0.4444... and 8 --/
def product : ℚ := repeating_four * 8

theorem product_equals_32_over_9 : product = 32/9 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_32_over_9_l1397_139721


namespace NUMINAMATH_CALUDE_cafeteria_sales_comparison_l1397_139768

def arithmetic_growth (initial : ℝ) (increment : ℝ) (periods : ℕ) : ℝ :=
  initial + increment * periods

def geometric_growth (initial : ℝ) (rate : ℝ) (periods : ℕ) : ℝ :=
  initial * (1 + rate) ^ periods

theorem cafeteria_sales_comparison
  (initial : ℝ)
  (increment : ℝ)
  (rate : ℝ)
  (h1 : initial > 0)
  (h2 : increment > 0)
  (h3 : rate > 0)
  (h4 : arithmetic_growth initial increment 8 = geometric_growth initial rate 8) :
  arithmetic_growth initial increment 4 > geometric_growth initial rate 4 :=
by sorry

end NUMINAMATH_CALUDE_cafeteria_sales_comparison_l1397_139768


namespace NUMINAMATH_CALUDE_triangle_area_l1397_139765

/-- The area of a triangle with one side of length 12 cm and an adjacent angle of 30° is 36 square centimeters. -/
theorem triangle_area (BC : ℝ) (angle_C : ℝ) : 
  BC = 12 → angle_C = 30 * (π / 180) → 
  (1/2) * BC * (BC * Real.sin angle_C) = 36 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1397_139765


namespace NUMINAMATH_CALUDE_profit_difference_l1397_139774

/-- The profit difference between selling a certain house and a standard house -/
theorem profit_difference (C : ℝ) : 
  let certain_house_cost : ℝ := C + 100000
  let standard_house_price : ℝ := 320000
  let certain_house_price : ℝ := 1.5 * standard_house_price
  let certain_house_profit : ℝ := certain_house_price - certain_house_cost
  let standard_house_profit : ℝ := standard_house_price - C
  certain_house_profit - standard_house_profit = 60000 := by
  sorry

#check profit_difference

end NUMINAMATH_CALUDE_profit_difference_l1397_139774


namespace NUMINAMATH_CALUDE_largest_trifecta_sum_l1397_139703

/-- A trifecta is an ordered triple of positive integers (a, b, c) with a < b < c
    such that a divides b, b divides c, and c divides ab. --/
def is_trifecta (a b c : ℕ) : Prop :=
  a < b ∧ b < c ∧ b % a = 0 ∧ c % b = 0 ∧ (a * b) % c = 0

/-- The sum of a trifecta (a, b, c) --/
def trifecta_sum (a b c : ℕ) : ℕ := a + b + c

/-- The largest possible sum of a trifecta of three-digit integers is 700 --/
theorem largest_trifecta_sum :
  (∃ a b c : ℕ, 100 ≤ a ∧ a < b ∧ b < c ∧ c ≤ 999 ∧ is_trifecta a b c ∧
    trifecta_sum a b c = 700) ∧
  (∀ a b c : ℕ, 100 ≤ a → a < b → b < c → c ≤ 999 → is_trifecta a b c →
    trifecta_sum a b c ≤ 700) :=
by sorry

end NUMINAMATH_CALUDE_largest_trifecta_sum_l1397_139703


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_angle_l1397_139797

-- Define an isosceles triangle with one interior angle of 110°
structure IsoscelesTriangle where
  base_angle : ℝ
  vertex_angle : ℝ
  is_isosceles : base_angle = base_angle  -- Both base angles are equal
  vertex_angle_value : vertex_angle = 110

-- Theorem: The base angle of this isosceles triangle is 35°
theorem isosceles_triangle_base_angle (t : IsoscelesTriangle) : t.base_angle = 35 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_angle_l1397_139797


namespace NUMINAMATH_CALUDE_integral_x_cubed_minus_three_to_three_l1397_139796

theorem integral_x_cubed_minus_three_to_three : 
  ∫ x in (-3)..3, x^3 = 0 := by sorry

end NUMINAMATH_CALUDE_integral_x_cubed_minus_three_to_three_l1397_139796


namespace NUMINAMATH_CALUDE_ben_marbles_count_l1397_139756

theorem ben_marbles_count (ben_marbles : ℕ) (leo_marbles : ℕ) : 
  (leo_marbles = ben_marbles + 20) →
  (ben_marbles + leo_marbles = 132) →
  (ben_marbles = 56) := by
sorry

end NUMINAMATH_CALUDE_ben_marbles_count_l1397_139756


namespace NUMINAMATH_CALUDE_ana_dress_count_l1397_139704

/-- The number of dresses Ana has -/
def ana_dresses : ℕ := 15

/-- The number of dresses Lisa has -/
def lisa_dresses : ℕ := ana_dresses + 18

/-- The total number of dresses Ana and Lisa have combined -/
def total_dresses : ℕ := 48

theorem ana_dress_count : ana_dresses = 15 := by sorry

end NUMINAMATH_CALUDE_ana_dress_count_l1397_139704


namespace NUMINAMATH_CALUDE_four_point_triangles_l1397_139733

/-- A point in a plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A set of four points in a plane -/
structure FourPoints :=
  (a b c d : Point)

/-- Predicate to check if three points are collinear -/
def collinear (p q r : Point) : Prop := sorry

/-- Predicate to check if no three points in a set of four points are collinear -/
def no_three_collinear (points : FourPoints) : Prop :=
  ¬(collinear points.a points.b points.c) ∧
  ¬(collinear points.a points.b points.d) ∧
  ¬(collinear points.a points.c points.d) ∧
  ¬(collinear points.b points.c points.d)

/-- The number of distinct triangles that can be formed from four points -/
def num_triangles (points : FourPoints) : ℕ := sorry

/-- Theorem: Given four points on a plane where no three points are collinear,
    the number of distinct triangles that can be formed is 4 -/
theorem four_point_triangles (points : FourPoints) 
  (h : no_three_collinear points) : num_triangles points = 4 := by
  sorry

end NUMINAMATH_CALUDE_four_point_triangles_l1397_139733


namespace NUMINAMATH_CALUDE_fraction_equality_implies_numerator_equality_l1397_139766

theorem fraction_equality_implies_numerator_equality 
  (a b c : ℝ) (h1 : c ≠ 0) (h2 : a / c = b / c) : a = b := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_numerator_equality_l1397_139766


namespace NUMINAMATH_CALUDE_spherical_to_rectangular_conversion_l1397_139725

theorem spherical_to_rectangular_conversion :
  let ρ : ℝ := 3
  let θ : ℝ := 5 * π / 12
  let φ : ℝ := π / 4
  let x : ℝ := ρ * Real.sin φ * Real.cos θ
  let y : ℝ := ρ * Real.sin φ * Real.sin θ
  let z : ℝ := ρ * Real.cos φ
  (x, y, z) = (3 * (Real.sqrt 12 + 2) / 8, 3 * (Real.sqrt 12 - 2) / 8, 3 * Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_spherical_to_rectangular_conversion_l1397_139725


namespace NUMINAMATH_CALUDE_farm_animals_l1397_139749

theorem farm_animals (cows chickens ducks : ℕ) : 
  (4 * cows + 2 * chickens + 2 * ducks = 24 + 2 * (cows + chickens + ducks)) →
  (ducks = chickens / 2) →
  (cows = 12) := by
sorry

end NUMINAMATH_CALUDE_farm_animals_l1397_139749


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1397_139705

theorem quadratic_inequality_range (k : ℝ) : 
  (∀ x : ℝ, k * x^2 - 6 * k * x + k + 8 ≥ 0) ↔ (0 ≤ k ∧ k ≤ 1) := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1397_139705


namespace NUMINAMATH_CALUDE_not_p_or_not_q_false_implies_l1397_139779

theorem not_p_or_not_q_false_implies (p q : Prop) 
  (h : ¬(¬p ∨ ¬q)) : 
  (p ∧ q) ∧ (p ∨ q) := by
  sorry

end NUMINAMATH_CALUDE_not_p_or_not_q_false_implies_l1397_139779


namespace NUMINAMATH_CALUDE_pentagon_angle_measure_l1397_139734

theorem pentagon_angle_measure (a b c d e : ℝ) : 
  -- Pentagon ABCDE is convex (sum of angles is 540°)
  a + b + c + d + e = 540 →
  -- Angle D is 30° more than angle A
  d = a + 30 →
  -- Angle E is 50° more than angle A
  e = a + 50 →
  -- Angles B and C are equal
  b = c →
  -- Angle A is 45° less than angle B
  a + 45 = b →
  -- Conclusion: Angle D measures 104°
  d = 104 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_angle_measure_l1397_139734


namespace NUMINAMATH_CALUDE_share_of_A_l1397_139729

theorem share_of_A (total : ℚ) (a b c : ℚ) : 
  total = 510 →
  a = (2 / 3) * b →
  b = (1 / 4) * c →
  total = a + b + c →
  a = 60 := by
sorry

end NUMINAMATH_CALUDE_share_of_A_l1397_139729


namespace NUMINAMATH_CALUDE_pizzas_ordered_proof_l1397_139710

/-- The number of students in the class -/
def num_students : ℕ := 32

/-- The number of cheese pieces each student gets -/
def cheese_per_student : ℕ := 2

/-- The number of onion pieces each student gets -/
def onion_per_student : ℕ := 1

/-- The number of slices in a large pizza -/
def slices_per_pizza : ℕ := 18

/-- The number of leftover cheese pieces -/
def leftover_cheese : ℕ := 8

/-- The number of leftover onion pieces -/
def leftover_onion : ℕ := 4

/-- The minimum number of pizzas ordered -/
def min_pizzas_ordered : ℕ := 5

theorem pizzas_ordered_proof :
  let total_cheese := num_students * cheese_per_student + leftover_cheese
  let total_onion := num_students * onion_per_student + leftover_onion
  let total_slices := total_cheese + total_onion
  (total_slices + slices_per_pizza - 1) / slices_per_pizza = min_pizzas_ordered :=
by sorry

end NUMINAMATH_CALUDE_pizzas_ordered_proof_l1397_139710


namespace NUMINAMATH_CALUDE_max_value_of_product_l1397_139727

theorem max_value_of_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 1) :
  x^2 * y^3 * z ≤ 1 / 3888 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_product_l1397_139727


namespace NUMINAMATH_CALUDE_right_triangle_has_multiple_altitudes_l1397_139785

/-- A right triangle is a triangle with one right angle. -/
structure RightTriangle where
  vertices : Fin 3 → ℝ × ℝ
  is_right_angle : sorry

/-- An altitude of a triangle is a line segment from a vertex perpendicular to the opposite side. -/
def altitude (t : RightTriangle) (v : Fin 3) : ℝ × ℝ := sorry

/-- The number of altitudes in a right triangle -/
def num_altitudes (t : RightTriangle) : ℕ := sorry

theorem right_triangle_has_multiple_altitudes (t : RightTriangle) : num_altitudes t > 1 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_has_multiple_altitudes_l1397_139785


namespace NUMINAMATH_CALUDE_bushel_weight_is_56_l1397_139724

/-- The weight of a bushel of corn in pounds -/
def bushel_weight : ℝ := 56

/-- The weight of an individual ear of corn in pounds -/
def ear_weight : ℝ := 0.5

/-- The number of bushels Clyde picked -/
def bushels_picked : ℕ := 2

/-- The number of individual corn cobs Clyde picked -/
def cobs_picked : ℕ := 224

/-- Theorem: The weight of a bushel of corn is 56 pounds -/
theorem bushel_weight_is_56 : 
  bushel_weight = (ear_weight * cobs_picked) / bushels_picked :=
sorry

end NUMINAMATH_CALUDE_bushel_weight_is_56_l1397_139724


namespace NUMINAMATH_CALUDE_no_solution_implies_a_geq_6_l1397_139794

theorem no_solution_implies_a_geq_6 (a : ℝ) : 
  (∀ x : ℝ, ¬(2*x - a > 0 ∧ 3*x - 4 < 5)) → a ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_a_geq_6_l1397_139794


namespace NUMINAMATH_CALUDE_total_age_problem_l1397_139739

theorem total_age_problem (a b c : ℕ) : 
  b = 4 → a = b + 2 → b = 2 * c → a + b + c = 12 := by
  sorry

end NUMINAMATH_CALUDE_total_age_problem_l1397_139739


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1397_139778

theorem regular_polygon_sides (interior_angle : ℝ) : 
  interior_angle = 140 → (360 / (180 - interior_angle) : ℝ) = 9 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1397_139778


namespace NUMINAMATH_CALUDE_tangent_line_at_one_f_positive_iff_a_le_two_l1397_139752

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := (x + 1) * Real.log x - a * (x - 1)

theorem tangent_line_at_one (a : ℝ) :
  a = 4 →
  ∃ k b : ℝ, ∀ x : ℝ, 
    (k * x + b = -2 * x + 2) ∧ 
    (k * 1 + b = f a 1) ∧
    (k = (deriv (f a)) 1) := by sorry

theorem f_positive_iff_a_le_two (a : ℝ) :
  (∀ x : ℝ, x > 1 → f a x > 0) ↔ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_f_positive_iff_a_le_two_l1397_139752


namespace NUMINAMATH_CALUDE_apple_distribution_l1397_139759

theorem apple_distribution (x : ℕ) (total_apples : ℕ) : 
  (total_apples = 5 * x + 12) → 
  (total_apples < 8 * x) →
  (0 ≤ 5 * x + 12 - 8 * (x - 1) ∧ 5 * x + 12 - 8 * (x - 1) < 8) :=
by sorry

end NUMINAMATH_CALUDE_apple_distribution_l1397_139759


namespace NUMINAMATH_CALUDE_chinese_chess_draw_probability_l1397_139707

theorem chinese_chess_draw_probability (p_win p_not_lose : ℝ) 
  (h_win : p_win = 0.6) 
  (h_not_lose : p_not_lose = 0.9) : 
  p_not_lose - p_win = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_chinese_chess_draw_probability_l1397_139707


namespace NUMINAMATH_CALUDE_f_min_value_l1397_139754

/-- The function f(x) defined in the problem -/
def f (x : ℝ) : ℝ := (x^2 + 4*x + 5)*(x^2 + 4*x + 2) + 2*x^2 + 8*x + 1

/-- Theorem stating that the minimum value of f(x) is -9 -/
theorem f_min_value : ∀ x : ℝ, f x ≥ -9 := by sorry

end NUMINAMATH_CALUDE_f_min_value_l1397_139754


namespace NUMINAMATH_CALUDE_student_average_score_l1397_139709

/-- Given a student's scores in physics, chemistry, and mathematics, prove that the average of all three subjects is 60. -/
theorem student_average_score (P C M : ℝ) : 
  P = 140 →                -- Physics score
  (P + M) / 2 = 90 →       -- Average of physics and mathematics
  (P + C) / 2 = 70 →       -- Average of physics and chemistry
  (P + C + M) / 3 = 60 :=  -- Average of all three subjects
by
  sorry


end NUMINAMATH_CALUDE_student_average_score_l1397_139709


namespace NUMINAMATH_CALUDE_complex_equation_difference_l1397_139745

theorem complex_equation_difference (x y : ℝ) : 
  (x : ℂ) + y * I = 1 + 2 * x * I → x - y = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_difference_l1397_139745


namespace NUMINAMATH_CALUDE_max_amount_received_back_l1397_139777

/-- Represents the casino chip denominations -/
inductive ChipDenomination
  | twenty
  | hundred

/-- Calculates the value of a chip -/
def chipValue : ChipDenomination → ℕ
  | ChipDenomination.twenty => 20
  | ChipDenomination.hundred => 100

/-- Represents the number of chips lost for each denomination -/
structure ChipsLost where
  twenty : ℕ
  hundred : ℕ

/-- Calculates the total value of chips lost -/
def totalLost (chips : ChipsLost) : ℕ :=
  chips.twenty * chipValue ChipDenomination.twenty +
  chips.hundred * chipValue ChipDenomination.hundred

/-- Represents the casino scenario -/
structure CasinoScenario where
  totalBought : ℕ
  chipsLost : ChipsLost

/-- Calculates the amount received back -/
def amountReceivedBack (scenario : CasinoScenario) : ℕ :=
  scenario.totalBought - totalLost scenario.chipsLost

/-- The main theorem to prove -/
theorem max_amount_received_back :
  ∀ (scenario : CasinoScenario),
    scenario.totalBought = 3000 ∧
    scenario.chipsLost.twenty + scenario.chipsLost.hundred = 13 ∧
    (scenario.chipsLost.twenty = scenario.chipsLost.hundred + 3 ∨
     scenario.chipsLost.twenty = scenario.chipsLost.hundred - 3) →
    amountReceivedBack scenario ≤ 2340 :=
by
  sorry

#check max_amount_received_back

end NUMINAMATH_CALUDE_max_amount_received_back_l1397_139777


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l1397_139742

theorem unique_quadratic_solution (p : ℝ) : 
  (p ≠ 0 ∧ ∃! x, p * x^2 - 10 * x + 2 = 0) ↔ p = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l1397_139742


namespace NUMINAMATH_CALUDE_john_popcorn_profit_l1397_139736

/-- Calculates the profit John makes from selling popcorn bags -/
theorem john_popcorn_profit :
  let regular_price : ℚ := 4
  let discount_rate : ℚ := 0.1
  let adult_price : ℚ := 8
  let child_price : ℚ := 6
  let adult_bags : ℕ := 20
  let child_bags : ℕ := 10
  let total_bags : ℕ := adult_bags + child_bags
  let discounted_price : ℚ := regular_price * (1 - discount_rate)
  let total_cost : ℚ := (total_bags : ℚ) * discounted_price
  let total_revenue : ℚ := (adult_bags : ℚ) * adult_price + (child_bags : ℚ) * child_price
  let profit : ℚ := total_revenue - total_cost
  profit = 112 :=
by
  sorry


end NUMINAMATH_CALUDE_john_popcorn_profit_l1397_139736


namespace NUMINAMATH_CALUDE_fractional_equation_solution_range_l1397_139726

theorem fractional_equation_solution_range (x m : ℝ) :
  (3 * x) / (x - 1) = m / (x - 1) + 2 →
  x ≥ 0 →
  x ≠ 1 →
  m ≥ 2 ∧ m ≠ 3 :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_range_l1397_139726


namespace NUMINAMATH_CALUDE_min_value_theorem_l1397_139716

theorem min_value_theorem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2*b = 1) :
  1/a + 2/b ≥ 9 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 2*b₀ = 1 ∧ 1/a₀ + 2/b₀ = 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1397_139716


namespace NUMINAMATH_CALUDE_students_exceed_rabbits_l1397_139728

theorem students_exceed_rabbits :
  let classrooms : ℕ := 5
  let students_per_classroom : ℕ := 23
  let rabbits_per_classroom : ℕ := 3
  let total_students : ℕ := classrooms * students_per_classroom
  let total_rabbits : ℕ := classrooms * rabbits_per_classroom
  total_students - total_rabbits = 100 := by
sorry

end NUMINAMATH_CALUDE_students_exceed_rabbits_l1397_139728


namespace NUMINAMATH_CALUDE_probability_of_specific_match_l1397_139744

/-- Calculates the probability of two specific players facing each other in a tournament. -/
theorem probability_of_specific_match (n : ℕ) (h : n = 26) : 
  (n - 1 : ℚ) / (n * (n - 1) / 2) = 1 / 13 := by
  sorry

#check probability_of_specific_match

end NUMINAMATH_CALUDE_probability_of_specific_match_l1397_139744


namespace NUMINAMATH_CALUDE_shirt_cost_problem_l1397_139715

/-- Proves that the original cost of one of the remaining shirts is $12.50 -/
theorem shirt_cost_problem (total_original_cost : ℝ) (discounted_shirt_price : ℝ) 
  (discount_rate : ℝ) (current_total_cost : ℝ) :
  total_original_cost = 100 →
  discounted_shirt_price = 25 →
  discount_rate = 0.4 →
  current_total_cost = 85 →
  ∃ (remaining_shirt_cost : ℝ),
    remaining_shirt_cost = 12.5 ∧
    3 * discounted_shirt_price * (1 - discount_rate) + 2 * remaining_shirt_cost = current_total_cost ∧
    3 * discounted_shirt_price + 2 * remaining_shirt_cost = total_original_cost :=
by
  sorry


end NUMINAMATH_CALUDE_shirt_cost_problem_l1397_139715


namespace NUMINAMATH_CALUDE_weight_difference_e_d_l1397_139767

/-- Given weights of individuals A, B, C, D, and E, prove that E weighs 3 kg more than D -/
theorem weight_difference_e_d (w_a w_b w_c w_d w_e : ℝ) : 
  (w_a + w_b + w_c) / 3 = 60 →
  (w_a + w_b + w_c + w_d) / 4 = 65 →
  (w_b + w_c + w_d + w_e) / 4 = 64 →
  w_a = 87 →
  w_e - w_d = 3 := by
sorry

end NUMINAMATH_CALUDE_weight_difference_e_d_l1397_139767


namespace NUMINAMATH_CALUDE_max_dinner_income_is_136_80_l1397_139712

/-- Represents the chef's restaurant scenario -/
structure RestaurantScenario where
  -- Lunch meals
  pasta_lunch : ℕ
  chicken_lunch : ℕ
  fish_lunch : ℕ
  -- Prices
  pasta_price : ℚ
  chicken_price : ℚ
  fish_price : ℚ
  -- Sold during lunch
  pasta_sold_lunch : ℕ
  chicken_sold_lunch : ℕ
  fish_sold_lunch : ℕ
  -- Dinner meals
  pasta_dinner : ℕ
  chicken_dinner : ℕ
  fish_dinner : ℕ
  -- Discount rate
  discount_rate : ℚ

/-- Calculates the maximum total income during dinner -/
def max_dinner_income (s : RestaurantScenario) : ℚ :=
  let pasta_unsold := s.pasta_lunch - s.pasta_sold_lunch
  let chicken_unsold := s.chicken_lunch - s.chicken_sold_lunch
  let fish_unsold := s.fish_lunch - s.fish_sold_lunch
  let discounted_pasta_price := s.pasta_price * (1 - s.discount_rate)
  let discounted_chicken_price := s.chicken_price * (1 - s.discount_rate)
  let discounted_fish_price := s.fish_price * (1 - s.discount_rate)
  (s.pasta_dinner * s.pasta_price + pasta_unsold * discounted_pasta_price) +
  (s.chicken_dinner * s.chicken_price + chicken_unsold * discounted_chicken_price) +
  (s.fish_dinner * s.fish_price + fish_unsold * discounted_fish_price)

/-- The chef's restaurant scenario -/
def chef_scenario : RestaurantScenario := {
  pasta_lunch := 8
  chicken_lunch := 5
  fish_lunch := 4
  pasta_price := 12
  chicken_price := 15
  fish_price := 18
  pasta_sold_lunch := 6
  chicken_sold_lunch := 3
  fish_sold_lunch := 3
  pasta_dinner := 2
  chicken_dinner := 2
  fish_dinner := 1
  discount_rate := 1/10
}

/-- Theorem stating the maximum total income during dinner -/
theorem max_dinner_income_is_136_80 :
  max_dinner_income chef_scenario = 136.8 := by sorry


end NUMINAMATH_CALUDE_max_dinner_income_is_136_80_l1397_139712


namespace NUMINAMATH_CALUDE_range_f_a2_values_of_a_min_3_l1397_139793

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := 4 * x^2 - 4 * a * x + (a^2 - 2 * a + 2)

-- Part 1: Range of f(x) when a = 2 in [1, 2]
theorem range_f_a2 :
  ∀ y ∈ Set.Icc (-2) 2, ∃ x ∈ Set.Icc 1 2, f 2 x = y :=
sorry

-- Part 2: Values of a when minimum of f(x) in [0, 2] is 3
theorem values_of_a_min_3 :
  (∀ x ∈ Set.Icc 0 2, f a x ≥ 3) ∧ (∃ x ∈ Set.Icc 0 2, f a x = 3) →
  a = 1 - Real.sqrt 2 ∨ a = 5 + Real.sqrt 10 :=
sorry

end NUMINAMATH_CALUDE_range_f_a2_values_of_a_min_3_l1397_139793


namespace NUMINAMATH_CALUDE_zero_neither_positive_nor_negative_l1397_139799

theorem zero_neither_positive_nor_negative : ¬(0 > 0 ∨ 0 < 0) := by
  sorry

end NUMINAMATH_CALUDE_zero_neither_positive_nor_negative_l1397_139799


namespace NUMINAMATH_CALUDE_smallest_three_digit_power_of_two_plus_one_multiple_of_five_l1397_139730

theorem smallest_three_digit_power_of_two_plus_one_multiple_of_five :
  ∃ (N : ℕ), 
    (100 ≤ N ∧ N ≤ 999) ∧ 
    (2^N + 1) % 5 = 0 ∧
    (∀ (M : ℕ), (100 ≤ M ∧ M < N) → (2^M + 1) % 5 ≠ 0) ∧
    N = 102 :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_power_of_two_plus_one_multiple_of_five_l1397_139730


namespace NUMINAMATH_CALUDE_inequality_proof_l1397_139762

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c ≤ 3) :
  1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1397_139762


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1397_139718

theorem absolute_value_inequality (x : ℝ) : 
  |((3 * x - 2) / (x - 2))| > 3 ↔ x ∈ Set.Ioo (4/3) 2 ∪ Set.Ioi 2 :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1397_139718


namespace NUMINAMATH_CALUDE_square_of_1024_l1397_139791

theorem square_of_1024 : (1024 : ℕ)^2 = 1048576 := by
  sorry

end NUMINAMATH_CALUDE_square_of_1024_l1397_139791


namespace NUMINAMATH_CALUDE_pizza_eaten_fraction_l1397_139787

theorem pizza_eaten_fraction (n : Nat) : 
  let r : ℚ := 1/3
  let sum : ℚ := (1 - r^n) / (1 - r)
  n = 6 → sum = 364/729 := by
sorry

end NUMINAMATH_CALUDE_pizza_eaten_fraction_l1397_139787


namespace NUMINAMATH_CALUDE_parabola_point_ordering_l1397_139731

-- Define the function
def f (x : ℝ) : ℝ := -x^2 + 5

-- Define the theorem
theorem parabola_point_ordering :
  ∀ (y₁ y₂ y₃ : ℝ),
  f (-4) = y₁ →
  f (-1) = y₂ →
  f 2 = y₃ →
  y₂ > y₃ ∧ y₃ > y₁ :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_point_ordering_l1397_139731


namespace NUMINAMATH_CALUDE_gcd_of_squares_sum_l1397_139770

theorem gcd_of_squares_sum : Nat.gcd 
  (122^2 + 234^2 + 346^2 + 458^2) 
  (121^2 + 233^2 + 345^2 + 457^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_squares_sum_l1397_139770


namespace NUMINAMATH_CALUDE_min_distance_parabola_to_line_l1397_139723

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y^2 = 2px -/
structure Parabola where
  p : ℝ

/-- Theorem: Minimum distance from a point on the parabola to the line y = x + 3 -/
theorem min_distance_parabola_to_line (para : Parabola) (P : Point) :
  para.p = 4 →  -- Derived from directrix x = -2
  P.y^2 = 2 * para.p * P.x →  -- Point P is on the parabola
  ∃ (d : ℝ), d = |P.x - P.y + 3| / Real.sqrt 2 ∧  -- Distance formula
  d ≥ Real.sqrt 2 / 2 ∧  -- Minimum distance
  (∃ (Q : Point), Q.y^2 = 2 * para.p * Q.x ∧  -- Another point on parabola
    |Q.x - Q.y + 3| / Real.sqrt 2 = Real.sqrt 2 / 2) :=  -- Achieving minimum distance
by sorry


end NUMINAMATH_CALUDE_min_distance_parabola_to_line_l1397_139723


namespace NUMINAMATH_CALUDE_rectangular_field_length_l1397_139740

theorem rectangular_field_length (w l : ℝ) (h1 : l = 2 * w) (h2 : 81 = (1 / 8) * (l * w)) :
  l = 36 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_field_length_l1397_139740


namespace NUMINAMATH_CALUDE_curve_in_second_quadrant_l1397_139713

-- Define the curve C
def C (a : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*a*x - 4*a*y + 5*a^2 - 4 = 0

-- Define the second quadrant
def second_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

-- Theorem statement
theorem curve_in_second_quadrant :
  (∀ a : ℝ, ∀ x y : ℝ, C a x y → second_quadrant x y) →
  (∀ a : ℝ, a ∈ Set.Ioi 2) :=
sorry

end NUMINAMATH_CALUDE_curve_in_second_quadrant_l1397_139713


namespace NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l1397_139764

theorem min_value_of_reciprocal_sum (p q r : ℝ) (a b : ℝ) : 
  0 < p ∧ 0 < q ∧ 0 < r →
  p < q ∧ q < r →
  p^3 - a*p^2 + b*p - 48 = 0 →
  q^3 - a*q^2 + b*q - 48 = 0 →
  r^3 - a*r^2 + b*r - 48 = 0 →
  1/p + 2/q + 3/r ≥ 3/2 ∧ ∃ p' q' r' a' b', 
    0 < p' ∧ 0 < q' ∧ 0 < r' ∧
    p' < q' ∧ q' < r' ∧
    p'^3 - a'*p'^2 + b'*p' - 48 = 0 ∧
    q'^3 - a'*q'^2 + b'*q' - 48 = 0 ∧
    r'^3 - a'*r'^2 + b'*r' - 48 = 0 ∧
    1/p' + 2/q' + 3/r' = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l1397_139764


namespace NUMINAMATH_CALUDE_lcm_count_theorem_l1397_139798

theorem lcm_count_theorem : 
  ∃ (S : Finset ℕ), 
    (∀ k ∈ S, Nat.lcm (Nat.lcm (9^9) (12^12)) k = 18^18) ∧ 
    (∀ k ∉ S, Nat.lcm (Nat.lcm (9^9) (12^12)) k ≠ 18^18) ∧ 
    S.card = 19 := by
  sorry

end NUMINAMATH_CALUDE_lcm_count_theorem_l1397_139798


namespace NUMINAMATH_CALUDE_sports_club_members_l1397_139771

theorem sports_club_members (total : ℕ) (badminton : ℕ) (tennis : ℕ) (both : ℕ) :
  total = 30 →
  badminton = 18 →
  tennis = 19 →
  both = 9 →
  total - (badminton + tennis - both) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_members_l1397_139771


namespace NUMINAMATH_CALUDE_total_campers_is_150_l1397_139732

/-- The total number of campers recorded for the past three weeks -/
def total_campers (three_weeks_ago two_weeks_ago last_week : ℕ) : ℕ :=
  three_weeks_ago + two_weeks_ago + last_week

/-- Proof that the total number of campers is 150 -/
theorem total_campers_is_150 :
  ∃ (three_weeks_ago two_weeks_ago last_week : ℕ),
    two_weeks_ago = 40 ∧
    two_weeks_ago = three_weeks_ago + 10 ∧
    last_week = 80 ∧
    total_campers three_weeks_ago two_weeks_ago last_week = 150 :=
by
  sorry

end NUMINAMATH_CALUDE_total_campers_is_150_l1397_139732


namespace NUMINAMATH_CALUDE_ellipse_properties_l1397_139780

-- Define the ellipse C
def ellipse_C (x y a b : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1 ∧ a > b ∧ b > 0

-- Define the focal distance
def focal_distance (c : ℝ) : Prop := c = 2

-- Define the eccentricity relation
def eccentricity_relation (a b : ℝ) : Prop :=
  (2 / a)^2 = 1 / 2

-- Define the line l
def line_l (x y k : ℝ) : Prop := y = k * x + 1

-- Define the intersection condition
def intersects_at_two_points (k : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂, x₁ ≠ x₂ ∧ 
    ellipse_C x₁ y₁ (2 * Real.sqrt 2) 2 ∧
    ellipse_C x₂ y₂ (2 * Real.sqrt 2) 2 ∧
    line_l x₁ y₁ k ∧ line_l x₂ y₂ k

-- Define the focus inside circle condition
def focus_inside_circle (k : ℝ) : Prop :=
  ∀ x₁ y₁ x₂ y₂, 
    ellipse_C x₁ y₁ (2 * Real.sqrt 2) 2 →
    ellipse_C x₂ y₂ (2 * Real.sqrt 2) 2 →
    line_l x₁ y₁ k → line_l x₂ y₂ k →
    (x₁ - 2) * (x₂ - 2) + y₁ * y₂ < 0

theorem ellipse_properties :
  ∀ a b : ℝ,
    ellipse_C 0 0 a b →
    focal_distance 2 →
    eccentricity_relation a b →
    (a = 2 * Real.sqrt 2 ∧ b = 2) ∧
    (∀ k : ℝ, intersects_at_two_points k →
      (focus_inside_circle k ↔ k < 1/8)) := by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l1397_139780


namespace NUMINAMATH_CALUDE_operation_result_l1397_139786

-- Define the operations
def op1 (m n : ℤ) : ℤ := n^2 - m
def op2 (m k : ℚ) : ℚ := (k + 2*m) / 3

-- Theorem statement
theorem operation_result : (op2 (op1 3 3) (op1 2 5)) = 35/3 := by
  sorry

end NUMINAMATH_CALUDE_operation_result_l1397_139786


namespace NUMINAMATH_CALUDE_garden_dimensions_and_walkway_area_l1397_139760

/-- A rectangular garden with a surrounding walkway. -/
structure Garden where
  breadth : ℝ
  length : ℝ
  walkwayWidth : ℝ

/-- Properties of the garden based on the problem conditions. -/
def GardenProperties (g : Garden) : Prop :=
  g.length = 3 * g.breadth ∧
  2 * (g.length + g.breadth) = 40 ∧
  g.walkwayWidth = 1 ∧
  (g.length + 2 * g.walkwayWidth) * (g.breadth + 2 * g.walkwayWidth) = 120

theorem garden_dimensions_and_walkway_area 
  (g : Garden) 
  (h : GardenProperties g) : 
  g.length = 15 ∧ g.breadth = 5 ∧ 
  ((g.length + 2 * g.walkwayWidth) * (g.breadth + 2 * g.walkwayWidth) - g.length * g.breadth) = 45 :=
by sorry

end NUMINAMATH_CALUDE_garden_dimensions_and_walkway_area_l1397_139760


namespace NUMINAMATH_CALUDE_event_committee_count_l1397_139720

/-- The number of teams in the league -/
def num_teams : ℕ := 5

/-- The number of members in each team -/
def team_size : ℕ := 8

/-- The number of members selected from the host team -/
def host_selection : ℕ := 4

/-- The number of members selected from each non-host team -/
def non_host_selection : ℕ := 3

/-- The total number of possible event committees -/
def total_committees : ℕ := 3442073600

/-- Theorem stating the number of possible event committees -/
theorem event_committee_count :
  (num_teams : ℕ) *
  (Nat.choose team_size host_selection) *
  (Nat.choose team_size non_host_selection)^(num_teams - 1) =
  total_committees := by sorry

end NUMINAMATH_CALUDE_event_committee_count_l1397_139720


namespace NUMINAMATH_CALUDE_smallest_among_given_numbers_l1397_139747

theorem smallest_among_given_numbers :
  let a := 1
  let b := Real.sqrt 2 / 2
  let c := Real.sqrt 3 / 3
  let d := Real.sqrt 5 / 5
  d < c ∧ d < b ∧ d < a := by
  sorry

end NUMINAMATH_CALUDE_smallest_among_given_numbers_l1397_139747


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1397_139775

/-- A hyperbola with foci at (-3,0) and (3,0), and a vertex at (2,0) has the equation x²/4 - y²/5 = 1 -/
theorem hyperbola_equation (x y : ℝ) : 
  let foci_1 : ℝ × ℝ := (-3, 0)
  let foci_2 : ℝ × ℝ := (3, 0)
  let vertex : ℝ × ℝ := (2, 0)
  (x^2 / 4 - y^2 / 5 = 1) ↔ 
    (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
      x^2 / a^2 - y^2 / b^2 = 1 ∧
      vertex.1 = a ∧
      (foci_2.1 - foci_1.1)^2 / 4 = a^2 + b^2) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1397_139775


namespace NUMINAMATH_CALUDE_tv_show_main_characters_l1397_139738

/-- Represents the TV show payment structure and calculates the number of main characters -/
def tv_show_characters : ℕ := by
  -- Define the number of minor characters
  let minor_characters : ℕ := 4
  -- Define the payment for each minor character
  let minor_payment : ℕ := 15000
  -- Define the total payment per episode
  let total_payment : ℕ := 285000
  -- Calculate the payment for each main character (3 times minor payment)
  let main_payment : ℕ := 3 * minor_payment
  -- Calculate the total payment for minor characters
  let minor_total : ℕ := minor_characters * minor_payment
  -- Calculate the remaining payment for main characters
  let main_total : ℕ := total_payment - minor_total
  -- Calculate the number of main characters
  exact main_total / main_payment

/-- Theorem stating that the number of main characters in the TV show is 5 -/
theorem tv_show_main_characters :
  tv_show_characters = 5 := by
  sorry

end NUMINAMATH_CALUDE_tv_show_main_characters_l1397_139738


namespace NUMINAMATH_CALUDE_prime_pairs_congruence_l1397_139753

theorem prime_pairs_congruence (p : ℕ) (hp : Nat.Prime p) : 
  (∃ S : Finset (ℕ × ℕ), S.card = p ∧ 
    (∀ (x y : ℕ), (x, y) ∈ S ↔ 
      (x ≤ p ∧ y ≤ p ∧ (y^2 : ZMod p) = (x^3 - x : ZMod p))))
  ↔ (p = 2 ∨ p % 4 = 3) :=
sorry

end NUMINAMATH_CALUDE_prime_pairs_congruence_l1397_139753


namespace NUMINAMATH_CALUDE_sum_of_coordinates_on_h_l1397_139741

def g (x : ℝ) : ℝ := x + 3

def h (x : ℝ) : ℝ := (g x)^2

theorem sum_of_coordinates_on_h : ∃ (x y : ℝ), 
  (2, 5) = (2, g 2) ∧ 
  (x, y) = (2, h 2) ∧ 
  x + y = 27 := by sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_on_h_l1397_139741


namespace NUMINAMATH_CALUDE_interleave_sequences_count_l1397_139706

def interleave_sequences (n₁ n₂ n₃ : ℕ) : ℕ :=
  Nat.factorial (n₁ + n₂ + n₃) / (Nat.factorial n₁ * Nat.factorial n₂ * Nat.factorial n₃)

theorem interleave_sequences_count (n₁ n₂ n₃ : ℕ) :
  interleave_sequences n₁ n₂ n₃ = 
    Nat.choose (n₁ + n₂ + n₃) n₁ * Nat.choose (n₂ + n₃) n₂ :=
by sorry

end NUMINAMATH_CALUDE_interleave_sequences_count_l1397_139706


namespace NUMINAMATH_CALUDE_three_power_gt_cube_l1397_139789

theorem three_power_gt_cube (n : ℕ) (h : n ≠ 3) : 3^n > n^3 := by
  sorry

end NUMINAMATH_CALUDE_three_power_gt_cube_l1397_139789


namespace NUMINAMATH_CALUDE_power_of_product_l1397_139751

theorem power_of_product (a : ℝ) : (2 * a) ^ 3 = 8 * a ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l1397_139751


namespace NUMINAMATH_CALUDE_mixture_percentage_l1397_139708

theorem mixture_percentage (solution1 solution2 : ℝ) 
  (percent1 percent2 : ℝ) (h1 : solution1 = 6) 
  (h2 : solution2 = 4) (h3 : percent1 = 0.2) 
  (h4 : percent2 = 0.6) : 
  (percent1 * solution1 + percent2 * solution2) / (solution1 + solution2) = 0.36 := by
  sorry

end NUMINAMATH_CALUDE_mixture_percentage_l1397_139708


namespace NUMINAMATH_CALUDE_tim_picked_five_pears_l1397_139784

/-- The number of pears Sara picked -/
def sara_pears : ℕ := 6

/-- The total number of pears picked by Sara and Tim -/
def total_pears : ℕ := 11

/-- The number of pears Tim picked -/
def tim_pears : ℕ := total_pears - sara_pears

theorem tim_picked_five_pears : tim_pears = 5 := by
  sorry

end NUMINAMATH_CALUDE_tim_picked_five_pears_l1397_139784


namespace NUMINAMATH_CALUDE_amount_with_r_l1397_139757

theorem amount_with_r (total : ℝ) (amount_r : ℝ) : 
  total = 7000 →
  amount_r = (2/3) * (total - amount_r) →
  amount_r = 2800 := by
sorry

end NUMINAMATH_CALUDE_amount_with_r_l1397_139757


namespace NUMINAMATH_CALUDE_composite_ratio_l1397_139788

def first_six_composites : List Nat := [4, 6, 8, 9, 10, 12]
def next_six_composites : List Nat := [14, 15, 16, 18, 20, 21]

def product_list (l : List Nat) : Nat :=
  l.foldl (·*·) 1

theorem composite_ratio :
  (product_list first_six_composites : Rat) / (product_list next_six_composites) = 1 / 49 := by
  sorry

end NUMINAMATH_CALUDE_composite_ratio_l1397_139788


namespace NUMINAMATH_CALUDE_equation_solutions_l1397_139755

def solution_set : Set ℝ := {12, 1, -1, -12}

def equation (x : ℝ) : Prop :=
  1 / (x^2 + 9*x - 12) + 1 / (x^2 + 3*x - 18) + 1 / (x^2 - 15*x - 12) = 0

theorem equation_solutions :
  {x : ℝ | equation x} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1397_139755


namespace NUMINAMATH_CALUDE_base_conversion_sum_l1397_139790

/-- Converts a number from base b to base 10 -/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * b^i) 0

/-- The main theorem -/
theorem base_conversion_sum :
  let x₁ := to_base_10 [3, 5, 2] 8
  let y₁ := to_base_10 [3, 1] 4
  let x₂ := to_base_10 [2, 3, 1] 5
  let y₂ := to_base_10 [3, 2] 3
  (x₁ : ℚ) / y₁ + (x₂ : ℚ) / y₂ = 28.67 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_sum_l1397_139790


namespace NUMINAMATH_CALUDE_zoe_spent_30_dollars_l1397_139719

/-- The price of a single flower in dollars -/
def flower_price : ℕ := 3

/-- The number of roses Zoe bought -/
def roses_bought : ℕ := 8

/-- The number of daisies Zoe bought -/
def daisies_bought : ℕ := 2

/-- Theorem: Given the conditions, Zoe spent 30 dollars -/
theorem zoe_spent_30_dollars : 
  (roses_bought + daisies_bought) * flower_price = 30 := by
  sorry

end NUMINAMATH_CALUDE_zoe_spent_30_dollars_l1397_139719


namespace NUMINAMATH_CALUDE_passengers_left_is_200_l1397_139792

/-- The number of minutes between train arrivals -/
def train_interval : ℕ := 5

/-- The number of passengers each train takes -/
def passengers_taken : ℕ := 320

/-- The total number of different passengers stepping on and off trains in one hour -/
def total_passengers : ℕ := 6240

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- The number of passengers each train leaves at the station -/
def passengers_left : ℕ := (total_passengers - (minutes_per_hour / train_interval * passengers_taken)) / (minutes_per_hour / train_interval)

theorem passengers_left_is_200 : passengers_left = 200 := by
  sorry

end NUMINAMATH_CALUDE_passengers_left_is_200_l1397_139792


namespace NUMINAMATH_CALUDE_min_value_A_l1397_139701

theorem min_value_A (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_abc : a + b + c = 3) :
  let A := (a^3 + b^3)/(8*a*b + 9 - c^2) + (b^3 + c^3)/(8*b*c + 9 - a^2) + (c^3 + a^3)/(8*c*a + 9 - b^2)
  A ≥ 3/8 ∧ ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ a₀ + b₀ + c₀ = 3 ∧
    (a₀^3 + b₀^3)/(8*a₀*b₀ + 9 - c₀^2) + (b₀^3 + c₀^3)/(8*b₀*c₀ + 9 - a₀^2) + (c₀^3 + a₀^3)/(8*c₀*a₀ + 9 - b₀^2) = 3/8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_A_l1397_139701


namespace NUMINAMATH_CALUDE_ratio_equation_solution_l1397_139769

theorem ratio_equation_solution (a b : ℚ) 
  (h1 : b / a = 4)
  (h2 : b = 20 - 7 * a) : 
  a = 20 / 11 := by
sorry

end NUMINAMATH_CALUDE_ratio_equation_solution_l1397_139769


namespace NUMINAMATH_CALUDE_system_solution_l1397_139700

theorem system_solution :
  ∀ x y z : ℝ,
  x = Real.sqrt (2 * y + 3) →
  y = Real.sqrt (2 * z + 3) →
  z = Real.sqrt (2 * x + 3) →
  x = 3 ∧ y = 3 ∧ z = 3 :=
by
  sorry

#check system_solution

end NUMINAMATH_CALUDE_system_solution_l1397_139700


namespace NUMINAMATH_CALUDE_range_of_a_l1397_139746

/-- The curve C in the Cartesian coordinate system -/
def C (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 + 2*a*p.1 - 4*a*p.2 + 5*a^2 - 4 = 0}

/-- All points on curve C are in the second quadrant -/
def all_points_in_second_quadrant (a : ℝ) : Prop :=
  ∀ p : ℝ × ℝ, p ∈ C a → p.1 < 0 ∧ p.2 > 0

/-- The main theorem -/
theorem range_of_a : 
  ∀ a : ℝ, all_points_in_second_quadrant a → a > 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1397_139746


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1397_139795

theorem simplify_and_evaluate (x y : ℝ) (h1 : x = -1) (h2 : y = 1) :
  2 * (x^2 * y + x * y) - 3 * (x^2 * y - x * y) - 5 * x * y = -1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1397_139795


namespace NUMINAMATH_CALUDE_employee_count_l1397_139783

theorem employee_count (avg_salary : ℝ) (new_avg_salary : ℝ) (manager_salary : ℝ) : 
  avg_salary = 1500 →
  new_avg_salary = 2500 →
  manager_salary = 22500 →
  ∃ (E : ℕ), (E : ℝ) * avg_salary + manager_salary = new_avg_salary * ((E : ℝ) + 1) ∧ E = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_employee_count_l1397_139783


namespace NUMINAMATH_CALUDE_intersection_implies_sum_l1397_139711

-- Define the sets A and B
def A (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = a * p.1 + 1}
def B (b : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1 + b}

-- State the theorem
theorem intersection_implies_sum (a b : ℝ) :
  A a ∩ B b = {(2, 5)} → a + b = 5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_sum_l1397_139711


namespace NUMINAMATH_CALUDE_elevator_problem_l1397_139722

theorem elevator_problem (initial_avg : ℝ) (new_avg : ℝ) (new_person_weight : ℝ) :
  initial_avg = 152 →
  new_avg = 151 →
  new_person_weight = 145 →
  ∃ n : ℕ, n > 0 ∧ 
    n * initial_avg + new_person_weight = (n + 1) * new_avg ∧
    n = 6 := by
  sorry

end NUMINAMATH_CALUDE_elevator_problem_l1397_139722


namespace NUMINAMATH_CALUDE_supermarket_spending_l1397_139743

theorem supermarket_spending (total : ℚ) :
  (1/2 : ℚ) * total +
  (1/3 : ℚ) * total +
  (1/10 : ℚ) * total +
  8 = total →
  total = 120 := by
sorry

end NUMINAMATH_CALUDE_supermarket_spending_l1397_139743


namespace NUMINAMATH_CALUDE_quadratic_solution_l1397_139758

theorem quadratic_solution (h : 81 * (4/9)^2 - 145 * (4/9) + 64 = 0) :
  81 * (-16/9)^2 - 145 * (-16/9) + 64 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l1397_139758


namespace NUMINAMATH_CALUDE_ring_toss_revenue_l1397_139702

/-- The daily revenue of a ring toss game at a carnival -/
def daily_revenue (total_revenue : ℕ) (num_days : ℕ) : ℚ :=
  total_revenue / num_days

/-- Theorem stating that the daily revenue is 140 given the conditions -/
theorem ring_toss_revenue :
  daily_revenue 420 3 = 140 := by
  sorry

end NUMINAMATH_CALUDE_ring_toss_revenue_l1397_139702


namespace NUMINAMATH_CALUDE_parabola_sum_l1397_139772

/-- A parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The vertex of a parabola -/
def vertex (p : Parabola) : ℝ × ℝ := sorry

/-- Check if a point is on the parabola -/
def contains_point (p : Parabola) (x y : ℝ) : Prop :=
  y = p.a * x^2 + p.b * x + p.c

/-- Check if the parabola has a vertical axis of symmetry -/
def has_vertical_axis (p : Parabola) : Prop := sorry

theorem parabola_sum (p : Parabola) :
  vertex p = (3, 7) →
  has_vertical_axis p →
  contains_point p 0 4 →
  p.a + p.b + p.c = 5.666 := by sorry

end NUMINAMATH_CALUDE_parabola_sum_l1397_139772


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1397_139750

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_property (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 1 * a 19 = 16) →
  (a 1 + a 19 = 10) →
  a 8 * a 10 * a 12 = 64 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1397_139750


namespace NUMINAMATH_CALUDE_triangle_angle_expression_minimum_l1397_139717

theorem triangle_angle_expression_minimum (A B C : Real) 
  (h_triangle : A + B + C = π) 
  (h_positive : 0 < A ∧ 0 < B ∧ 0 < C) : 
  (1 / (Real.sin A)^2) + (1 / (Real.sin B)^2) + (4 / (1 + Real.sin C)) ≥ 16 - 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_expression_minimum_l1397_139717


namespace NUMINAMATH_CALUDE_linear_function_quadrants_l1397_139748

/-- A linear function passing through the first, second, and third quadrants implies positive slope and y-intercept -/
theorem linear_function_quadrants (k b : ℝ) : 
  (∀ x y : ℝ, y = k * x + b → 
    (∃ x₁ y₁, x₁ > 0 ∧ y₁ > 0 ∧ y₁ = k * x₁ + b) ∧ 
    (∃ x₂ y₂, x₂ < 0 ∧ y₂ > 0 ∧ y₂ = k * x₂ + b) ∧ 
    (∃ x₃ y₃, x₃ < 0 ∧ y₃ < 0 ∧ y₃ = k * x₃ + b)) →
  k > 0 ∧ b > 0 := by
sorry

end NUMINAMATH_CALUDE_linear_function_quadrants_l1397_139748


namespace NUMINAMATH_CALUDE_sqrt_18_minus_sqrt_2_equality_l1397_139782

theorem sqrt_18_minus_sqrt_2_equality (a b : ℝ) :
  Real.sqrt 18 - Real.sqrt 2 = a * Real.sqrt 2 - Real.sqrt 2 ∧
  a * Real.sqrt 2 - Real.sqrt 2 = b * Real.sqrt 2 →
  a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_18_minus_sqrt_2_equality_l1397_139782


namespace NUMINAMATH_CALUDE_prob_at_least_one_woman_pair_value_l1397_139763

/-- The number of young men in the group -/
def num_men : ℕ := 6

/-- The number of young women in the group -/
def num_women : ℕ := 6

/-- The total number of people in the group -/
def total_people : ℕ := num_men + num_women

/-- The number of pairs formed -/
def num_pairs : ℕ := total_people / 2

/-- The total number of ways to pair up all people -/
def total_pairings : ℕ := (total_people.factorial) / (2^num_pairs * num_pairs.factorial)

/-- The number of ways to pair up without any woman-woman pairs -/
def pairings_without_woman_pairs : ℕ := num_women.factorial

/-- The probability of at least one woman-woman pair -/
def prob_at_least_one_woman_pair : ℚ :=
  (total_pairings - pairings_without_woman_pairs : ℚ) / total_pairings

theorem prob_at_least_one_woman_pair_value :
  prob_at_least_one_woman_pair = (10395 - 720) / 10395 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_woman_pair_value_l1397_139763


namespace NUMINAMATH_CALUDE_largest_non_sum_of_three_distinct_composites_l1397_139776

/-- A number is composite if it's greater than 1 and not prime -/
def IsComposite (n : ℕ) : Prop := n > 1 ∧ ¬Nat.Prime n

/-- A function that checks if a natural number can be expressed as the sum of three distinct composite numbers -/
def IsSumOfThreeDistinctComposites (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), IsComposite a ∧ IsComposite b ∧ IsComposite c ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b + c = n

/-- The theorem stating that 17 is the largest integer that cannot be expressed as the sum of three distinct composite numbers -/
theorem largest_non_sum_of_three_distinct_composites :
  (∀ n > 17, IsSumOfThreeDistinctComposites n) ∧
  ¬IsSumOfThreeDistinctComposites 17 ∧
  (∀ n < 17, ¬IsSumOfThreeDistinctComposites n → ¬IsSumOfThreeDistinctComposites 17) :=
sorry

end NUMINAMATH_CALUDE_largest_non_sum_of_three_distinct_composites_l1397_139776


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l1397_139781

theorem unique_solution_for_equation (y : ℝ) : y + 49 / y = 14 ↔ y = 7 := by sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l1397_139781
