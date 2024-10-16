import Mathlib

namespace NUMINAMATH_CALUDE_f_max_value_l1817_181747

/-- The function f(z) = -6z^2 + 24z - 12 -/
def f (z : ℝ) : ℝ := -6 * z^2 + 24 * z - 12

theorem f_max_value :
  (∀ z : ℝ, f z ≤ 12) ∧ (∃ z : ℝ, f z = 12) := by sorry

end NUMINAMATH_CALUDE_f_max_value_l1817_181747


namespace NUMINAMATH_CALUDE_complex_power_sum_l1817_181739

theorem complex_power_sum (z : ℂ) (h : z + 1/z = 2 * Real.cos (5 * π / 180)) :
  z^1800 + 1/(z^1800) = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l1817_181739


namespace NUMINAMATH_CALUDE_unique_base_property_l1817_181751

theorem unique_base_property (a : ℕ) (b : ℕ) (h1 : a > 0) (h2 : b > 1) :
  (a * b + (a + 1) = (a + 2) * (a + 3)) → b = 10 :=
by sorry

end NUMINAMATH_CALUDE_unique_base_property_l1817_181751


namespace NUMINAMATH_CALUDE_pages_used_l1817_181704

def cards_per_page : ℕ := 3
def new_cards : ℕ := 8
def old_cards : ℕ := 16

theorem pages_used (total_cards : ℕ) (h : total_cards = new_cards + old_cards) :
  total_cards / cards_per_page = 8 :=
sorry

end NUMINAMATH_CALUDE_pages_used_l1817_181704


namespace NUMINAMATH_CALUDE_circus_tent_capacity_l1817_181713

/-- The number of sections in the circus tent -/
def num_sections : ℕ := 4

/-- The capacity of each section in the circus tent -/
def section_capacity : ℕ := 246

/-- The total capacity of the circus tent -/
def total_capacity : ℕ := num_sections * section_capacity

theorem circus_tent_capacity : total_capacity = 984 := by
  sorry

end NUMINAMATH_CALUDE_circus_tent_capacity_l1817_181713


namespace NUMINAMATH_CALUDE_sequence_perfect_square_property_l1817_181796

/-- Given two sequences of natural numbers satisfying a specific equation,
    prove that yₙ - 1 is a perfect square for all n. -/
theorem sequence_perfect_square_property
  (x y : ℕ → ℕ)
  (h : ∀ n : ℕ, (x n : ℝ) + Real.sqrt 2 * (y n : ℝ) = Real.sqrt 2 * (3 + 2 * Real.sqrt 2) ^ (2 ^ n)) :
  ∀ n : ℕ, ∃ k : ℕ, y n - 1 = k ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_perfect_square_property_l1817_181796


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1817_181763

theorem polynomial_simplification (p : ℝ) :
  (5 * p^4 - 7 * p^2 + 3 * p + 9) + (-3 * p^3 + 2 * p^2 - 4 * p + 6) =
  5 * p^4 - 3 * p^3 - 5 * p^2 - p + 15 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1817_181763


namespace NUMINAMATH_CALUDE_event_relationship_l1817_181706

-- Define the critical value
def critical_value : ℝ := 6.635

-- Define the confidence level
def confidence_level : ℝ := 0.99

-- Define the relationship between K^2 and the confidence level
theorem event_relationship (K : ℝ) :
  K^2 > critical_value → confidence_level = 0.99 := by
  sorry

#check event_relationship

end NUMINAMATH_CALUDE_event_relationship_l1817_181706


namespace NUMINAMATH_CALUDE_chocolate_chip_calculation_l1817_181710

/-- The number of cups of chocolate chips needed for one recipe -/
def chips_per_recipe : ℝ := 3.5

/-- The number of recipes to be made -/
def num_recipes : ℕ := 37

/-- The total number of cups of chocolate chips needed -/
def total_chips : ℝ := chips_per_recipe * num_recipes

theorem chocolate_chip_calculation : total_chips = 129.5 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_chip_calculation_l1817_181710


namespace NUMINAMATH_CALUDE_pencil_cost_l1817_181700

theorem pencil_cost (initial_amount : ℕ) (amount_left : ℕ) (candy_cost : ℕ) : 
  initial_amount = 43 → amount_left = 18 → candy_cost = 5 → 
  initial_amount - amount_left - candy_cost = 20 := by
  sorry

end NUMINAMATH_CALUDE_pencil_cost_l1817_181700


namespace NUMINAMATH_CALUDE_binomial_variance_determines_n_l1817_181715

/-- A random variable following a binomial distribution -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The variance of a binomial distribution -/
def variance (ξ : BinomialDistribution) : ℝ := ξ.n * ξ.p * (1 - ξ.p)

theorem binomial_variance_determines_n (ξ : BinomialDistribution) 
  (h_p : ξ.p = 0.3) 
  (h_var : variance ξ = 2.1) : 
  ξ.n = 10 := by
sorry

end NUMINAMATH_CALUDE_binomial_variance_determines_n_l1817_181715


namespace NUMINAMATH_CALUDE_chris_earnings_june_l1817_181729

/-- Chris's earnings for the first two weeks of June --/
def chrisEarnings (hoursWeek1 hoursWeek2 : ℕ) (extraEarnings : ℚ) : ℚ :=
  let hourlyWage := extraEarnings / (hoursWeek2 - hoursWeek1)
  hourlyWage * (hoursWeek1 + hoursWeek2)

/-- Theorem stating Chris's earnings for the first two weeks of June --/
theorem chris_earnings_june :
  chrisEarnings 18 30 (65.40 : ℚ) = (261.60 : ℚ) := by
  sorry

#eval chrisEarnings 18 30 (65.40 : ℚ)

end NUMINAMATH_CALUDE_chris_earnings_june_l1817_181729


namespace NUMINAMATH_CALUDE_intersection_not_empty_l1817_181786

theorem intersection_not_empty :
  ∃ (n : ℕ) (k : ℕ), n > 1 ∧ 2^n - n = k^2 := by sorry

end NUMINAMATH_CALUDE_intersection_not_empty_l1817_181786


namespace NUMINAMATH_CALUDE_game_show_probability_l1817_181774

/-- Represents the amount of money in each box -/
def box_values : Fin 3 → ℕ
  | 0 => 4
  | 1 => 400
  | 2 => 4000

/-- The total number of ways to assign 3 keys to 3 boxes -/
def total_assignments : ℕ := 6

/-- The number of assignments that result in winning more than $4000 -/
def winning_assignments : ℕ := 1

/-- The probability of winning more than $4000 -/
def win_probability : ℚ := winning_assignments / total_assignments

theorem game_show_probability :
  win_probability = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_game_show_probability_l1817_181774


namespace NUMINAMATH_CALUDE_positive_integer_sum_greater_than_product_l1817_181738

theorem positive_integer_sum_greater_than_product (a b : ℕ+) :
  a + b > a * b ↔ a = 1 ∨ b = 1 := by sorry

end NUMINAMATH_CALUDE_positive_integer_sum_greater_than_product_l1817_181738


namespace NUMINAMATH_CALUDE_no_seven_edge_polyhedron_l1817_181705

/-- A polyhedron is a structure with vertices, edges, and faces. -/
structure Polyhedron where
  V : ℕ  -- number of vertices
  E : ℕ  -- number of edges
  F : ℕ  -- number of faces

/-- Euler's formula for polyhedra -/
axiom euler_formula (p : Polyhedron) : p.V - p.E + p.F = 2

/-- Each vertex in a polyhedron has at least 3 edges -/
axiom vertex_edge_count (p : Polyhedron) : p.E * 2 ≥ p.V * 3

/-- A polyhedron must have at least 4 vertices -/
axiom min_vertices (p : Polyhedron) : p.V ≥ 4

/-- Theorem: No polyhedron can have exactly 7 edges -/
theorem no_seven_edge_polyhedron :
  ¬∃ (p : Polyhedron), p.E = 7 := by sorry

end NUMINAMATH_CALUDE_no_seven_edge_polyhedron_l1817_181705


namespace NUMINAMATH_CALUDE_vegetable_field_area_l1817_181767

theorem vegetable_field_area (V W : ℝ) 
  (h1 : (1/2) * V + (1/3) * W = 13)
  (h2 : (1/2) * W + (1/3) * V = 12) : 
  V = 18 := by
sorry

end NUMINAMATH_CALUDE_vegetable_field_area_l1817_181767


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l1817_181764

/-- Represents a parabola in the form y = a(x - h)^2 + k --/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Shifts a parabola horizontally and vertically --/
def shift (p : Parabola) (dx dy : ℝ) : Parabola :=
  { a := p.a
    h := p.h - dx
    k := p.k + dy }

theorem parabola_shift_theorem (p : Parabola) :
  p.a = 2 ∧ p.h = 1 ∧ p.k = 3 →
  (shift (shift p 2 0) 0 (-1)) = { a := 2, h := -1, k := 2 } := by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_theorem_l1817_181764


namespace NUMINAMATH_CALUDE_function_inequality_implies_positive_a_l1817_181782

open Real

theorem function_inequality_implies_positive_a (a : ℝ) :
  (∃ x₀ ∈ Set.Icc 1 (Real.exp 1), a * (x₀ - 1 / x₀) - 2 * log x₀ > -a / x₀) →
  a > 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_positive_a_l1817_181782


namespace NUMINAMATH_CALUDE_expression_factorization_l1817_181777

theorem expression_factorization (a : ℝ) : 
  (8 * a^3 + 105 * a^2 + 7) - (-9 * a^3 + 16 * a^2 - 14) = a^2 * (17 * a + 89) + 21 := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l1817_181777


namespace NUMINAMATH_CALUDE_greatest_distance_between_circle_centers_l1817_181719

theorem greatest_distance_between_circle_centers 
  (rectangle_width : ℝ) 
  (rectangle_height : ℝ) 
  (circle_diameter : ℝ) 
  (h1 : rectangle_width = 16) 
  (h2 : rectangle_height = 20) 
  (h3 : circle_diameter = 8) :
  ∃ (d : ℝ), d = 4 * Real.sqrt 13 ∧ 
  ∀ (d' : ℝ), d' ≤ d ∧ 
  ∃ (x1 y1 x2 y2 : ℝ),
    0 ≤ x1 ∧ x1 ≤ rectangle_width ∧
    0 ≤ y1 ∧ y1 ≤ rectangle_height ∧
    0 ≤ x2 ∧ x2 ≤ rectangle_width ∧
    0 ≤ y2 ∧ y2 ≤ rectangle_height ∧
    (x1 - circle_diameter / 2 ≥ 0) ∧ (x1 + circle_diameter / 2 ≤ rectangle_width) ∧
    (y1 - circle_diameter / 2 ≥ 0) ∧ (y1 + circle_diameter / 2 ≤ rectangle_height) ∧
    (x2 - circle_diameter / 2 ≥ 0) ∧ (x2 + circle_diameter / 2 ≤ rectangle_width) ∧
    (y2 - circle_diameter / 2 ≥ 0) ∧ (y2 + circle_diameter / 2 ≤ rectangle_height) ∧
    d' = Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) := by
  sorry


end NUMINAMATH_CALUDE_greatest_distance_between_circle_centers_l1817_181719


namespace NUMINAMATH_CALUDE_total_amount_collected_l1817_181798

/-- Represents the ratio of passengers in I and II class -/
def passenger_ratio : ℚ := 1 / 50

/-- Represents the ratio of fares for I and II class -/
def fare_ratio : ℚ := 3 / 1

/-- Amount collected from II class passengers in rupees -/
def amount_II : ℕ := 1250

/-- Theorem stating the total amount collected from all passengers -/
theorem total_amount_collected :
  let amount_I := (amount_II : ℚ) * passenger_ratio * fare_ratio
  (amount_I + amount_II : ℚ) = 1325 := by sorry

end NUMINAMATH_CALUDE_total_amount_collected_l1817_181798


namespace NUMINAMATH_CALUDE_perimeter_gt_four_times_circumradius_l1817_181794

/-- An acute-angled triangle with its perimeter and circumradius -/
structure AcuteTriangle where
  -- The perimeter of the triangle
  perimeter : ℝ
  -- The circumradius of the triangle
  circumradius : ℝ
  -- Condition ensuring the triangle is acute-angled (this is a simplification)
  is_acute : perimeter > 0 ∧ circumradius > 0

/-- Theorem stating that for any acute-angled triangle, its perimeter is greater than 4 times its circumradius -/
theorem perimeter_gt_four_times_circumradius (t : AcuteTriangle) : t.perimeter > 4 * t.circumradius := by
  sorry


end NUMINAMATH_CALUDE_perimeter_gt_four_times_circumradius_l1817_181794


namespace NUMINAMATH_CALUDE_youngest_child_age_l1817_181799

def restaurant_problem (father_charge : ℝ) (child_charge_per_year : ℝ) (total_bill : ℝ) : Prop :=
  ∃ (twin_age youngest_age : ℕ),
    father_charge = 4.95 ∧
    child_charge_per_year = 0.45 ∧
    total_bill = 9.45 ∧
    twin_age > youngest_age ∧
    total_bill = father_charge + child_charge_per_year * (2 * twin_age + youngest_age) ∧
    youngest_age = 2

theorem youngest_child_age :
  restaurant_problem 4.95 0.45 9.45
  := by sorry

end NUMINAMATH_CALUDE_youngest_child_age_l1817_181799


namespace NUMINAMATH_CALUDE_circle_tangent_to_line_l1817_181735

/-- The value of 'a' for which the circle (x - a)² + (y - 1)² = 16 is tangent to the line 3x + 4y - 5 = 0 -/
theorem circle_tangent_to_line (a : ℝ) (h : a > 0) :
  (∃! p : ℝ × ℝ, (p.1 - a)^2 + (p.2 - 1)^2 = 16 ∧ 3*p.1 + 4*p.2 - 5 = 0) →
  a = 7 :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_to_line_l1817_181735


namespace NUMINAMATH_CALUDE_max_third_altitude_is_six_l1817_181727

/-- A scalene triangle with two known altitudes -/
structure ScaleneTriangle where
  /-- The length of the first known altitude -/
  altitude1 : ℝ
  /-- The length of the second known altitude -/
  altitude2 : ℝ
  /-- The triangle is scalene -/
  scalene : altitude1 ≠ altitude2
  /-- The altitudes are positive -/
  positive1 : altitude1 > 0
  positive2 : altitude2 > 0

/-- The maximum possible integer length of the third altitude -/
def max_third_altitude (t : ScaleneTriangle) : ℕ :=
  6

/-- Theorem stating that the maximum possible integer length of the third altitude is 6 -/
theorem max_third_altitude_is_six (t : ScaleneTriangle) 
  (h1 : t.altitude1 = 6 ∨ t.altitude2 = 6) 
  (h2 : t.altitude1 = 18 ∨ t.altitude2 = 18) : 
  max_third_altitude t = 6 := by
  sorry

#check max_third_altitude_is_six

end NUMINAMATH_CALUDE_max_third_altitude_is_six_l1817_181727


namespace NUMINAMATH_CALUDE_equation_solution_l1817_181772

theorem equation_solution :
  let f : ℝ → ℝ := λ x => (2*x + 1)*(3*x + 1)*(5*x + 1)*(30*x + 1)
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    f x₁ = 10 ∧ f x₂ = 10 ∧
    x₁ = (-4 + Real.sqrt 31) / 15 ∧
    x₂ = (-4 - Real.sqrt 31) / 15 ∧
    ∀ x : ℝ, f x = 10 → (x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1817_181772


namespace NUMINAMATH_CALUDE_pet_store_birds_l1817_181768

/-- The number of bird cages in the pet store -/
def num_cages : ℕ := 4

/-- The number of parrots in each cage -/
def parrots_per_cage : ℕ := 8

/-- The number of parakeets in each cage -/
def parakeets_per_cage : ℕ := 2

/-- The total number of birds in the pet store -/
def total_birds : ℕ := num_cages * (parrots_per_cage + parakeets_per_cage)

theorem pet_store_birds :
  total_birds = 40 :=
sorry

end NUMINAMATH_CALUDE_pet_store_birds_l1817_181768


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1817_181795

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicularLines : Line → Line → Prop)
variable (lineParallelPlane : Line → Plane → Prop)

-- Define the theorem
theorem sufficient_but_not_necessary
  (a b : Line) (α β : Plane)
  (h_diff : a ≠ b)
  (h_parallel : parallel α β)
  (h_perp : perpendicular a α) :
  (∃ (c : Line), c ≠ b ∧ perpendicularLines a c ∧ ¬lineParallelPlane c β) ∧
  (lineParallelPlane b β → perpendicularLines a b) ∧
  ¬(perpendicularLines a b → lineParallelPlane b β) :=
sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1817_181795


namespace NUMINAMATH_CALUDE_gcd_9125_4277_l1817_181718

theorem gcd_9125_4277 : Nat.gcd 9125 4277 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_9125_4277_l1817_181718


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l1817_181725

theorem simplify_trig_expression :
  (Real.cos (5 * π / 180))^2 - (Real.sin (5 * π / 180))^2 =
  2 * Real.sin (40 * π / 180) * Real.cos (40 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l1817_181725


namespace NUMINAMATH_CALUDE_unique_magnitude_of_quadratic_root_l1817_181712

theorem unique_magnitude_of_quadratic_root : ∃! m : ℝ, ∃ z : ℂ, z^2 - 10*z + 52 = 0 ∧ Complex.abs z = m :=
by sorry

end NUMINAMATH_CALUDE_unique_magnitude_of_quadratic_root_l1817_181712


namespace NUMINAMATH_CALUDE_composite_odd_number_characterization_l1817_181757

theorem composite_odd_number_characterization (c : ℕ) (h_odd : Odd c) :
  (∃ (a : ℕ), a ≤ c / 3 - 1 ∧ ∃ (k : ℕ), (2 * a - 1)^2 + 8 * c = (2 * k + 1)^2) ↔
  (∃ (p q : ℕ), p > 1 ∧ q > 1 ∧ c = p * q) :=
sorry

end NUMINAMATH_CALUDE_composite_odd_number_characterization_l1817_181757


namespace NUMINAMATH_CALUDE_square_numbers_between_24_and_150_divisible_by_6_l1817_181792

def is_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

theorem square_numbers_between_24_and_150_divisible_by_6 :
  {x : ℕ | 24 < x ∧ x < 150 ∧ is_square x ∧ x % 6 = 0} = {36, 144} := by
  sorry

end NUMINAMATH_CALUDE_square_numbers_between_24_and_150_divisible_by_6_l1817_181792


namespace NUMINAMATH_CALUDE_number_and_multiple_l1817_181765

theorem number_and_multiple (x k : ℝ) : x = -7.0 ∧ 3 * x = k * x - 7 → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_number_and_multiple_l1817_181765


namespace NUMINAMATH_CALUDE_volunteer_selection_l1817_181770

/-- The number of ways to select 5 people out of 9 (5 male and 4 female), 
    ensuring both genders are included. -/
theorem volunteer_selection (n m f : ℕ) 
  (h1 : n = 5) -- Total number to be selected
  (h2 : m = 5) -- Number of male students
  (h3 : f = 4) -- Number of female students
  : Nat.choose (m + f) n - Nat.choose m n = 125 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_selection_l1817_181770


namespace NUMINAMATH_CALUDE_mark_payment_l1817_181776

def hours : ℕ := 3
def hourly_rate : ℚ := 15
def tip_percentage : ℚ := 20 / 100

def total_paid : ℚ :=
  let base_cost := hours * hourly_rate
  let tip := base_cost * tip_percentage
  base_cost + tip

theorem mark_payment : total_paid = 54 := by
  sorry

end NUMINAMATH_CALUDE_mark_payment_l1817_181776


namespace NUMINAMATH_CALUDE_max_value_sin_tan_function_l1817_181701

theorem max_value_sin_tan_function :
  ∀ x : ℝ, 2 * Real.sin x ^ 2 - Real.tan x ^ 2 ≤ 3 - 2 * Real.sqrt 2 ∧
  ∃ x : ℝ, 2 * Real.sin x ^ 2 - Real.tan x ^ 2 = 3 - 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sin_tan_function_l1817_181701


namespace NUMINAMATH_CALUDE_factorize_xm_minus_xn_l1817_181724

theorem factorize_xm_minus_xn (x m n : ℝ) : x * m - x * n = x * (m - n) := by
  sorry

end NUMINAMATH_CALUDE_factorize_xm_minus_xn_l1817_181724


namespace NUMINAMATH_CALUDE_max_product_under_constraint_l1817_181709

theorem max_product_under_constraint (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 4*b = 8) :
  ab ≤ 4 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 4*b₀ = 8 ∧ a₀*b₀ = 4 :=
sorry

end NUMINAMATH_CALUDE_max_product_under_constraint_l1817_181709


namespace NUMINAMATH_CALUDE_expression_evaluation_l1817_181742

theorem expression_evaluation :
  let a : ℝ := 1
  let b : ℝ := -2
  a * (a - 2*b) + (a + b)^2 - (a + b)*(a - b) = 9 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1817_181742


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l1817_181783

theorem rectangle_diagonal (perimeter : ℝ) (ratio_length : ℝ) (ratio_width : ℝ) :
  perimeter = 72 →
  ratio_length = 3 →
  ratio_width = 2 →
  let length := (perimeter / 2) * (ratio_length / (ratio_length + ratio_width))
  let width := (perimeter / 2) * (ratio_width / (ratio_length + ratio_width))
  (length ^ 2 + width ^ 2) = 673.92 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l1817_181783


namespace NUMINAMATH_CALUDE_sheep_to_cow_ratio_is_ten_to_one_l1817_181703

/-- Represents the farm owned by Mr. Reyansh -/
structure Farm where
  num_cows : ℕ
  cow_water_daily : ℕ
  sheep_water_ratio : ℚ
  total_water_weekly : ℕ

/-- Calculates the ratio of sheep to cows on the farm -/
def sheep_to_cow_ratio (f : Farm) : ℚ :=
  let cow_water_weekly := f.num_cows * f.cow_water_daily * 7
  let sheep_water_weekly := f.total_water_weekly - cow_water_weekly
  let sheep_water_daily := sheep_water_weekly / 7
  let num_sheep := sheep_water_daily / (f.cow_water_daily * f.sheep_water_ratio)
  num_sheep / f.num_cows

/-- Theorem stating that the ratio of sheep to cows is 10:1 -/
theorem sheep_to_cow_ratio_is_ten_to_one (f : Farm) 
    (h1 : f.num_cows = 40)
    (h2 : f.cow_water_daily = 80)
    (h3 : f.sheep_water_ratio = 1/4)
    (h4 : f.total_water_weekly = 78400) :
  sheep_to_cow_ratio f = 10 := by
  sorry

#eval sheep_to_cow_ratio { num_cows := 40, cow_water_daily := 80, sheep_water_ratio := 1/4, total_water_weekly := 78400 }

end NUMINAMATH_CALUDE_sheep_to_cow_ratio_is_ten_to_one_l1817_181703


namespace NUMINAMATH_CALUDE_inscribed_circle_area_ratio_l1817_181717

theorem inscribed_circle_area_ratio (α : Real) (h : 0 < α ∧ α < π / 2) :
  let rhombus_area (a : Real) := a^2 * Real.sin α
  let circle_area (r : Real) := π * r^2
  let inscribed_circle_radius (a : Real) := (a * Real.sin α) / 2
  ∀ a > 0, circle_area (inscribed_circle_radius a) / rhombus_area a = (π / 4) * Real.sin α :=
sorry

end NUMINAMATH_CALUDE_inscribed_circle_area_ratio_l1817_181717


namespace NUMINAMATH_CALUDE_same_solution_implies_k_equals_two_l1817_181761

theorem same_solution_implies_k_equals_two (k : ℝ) :
  (∃ x : ℝ, (2 * x - 1) / 3 = 5 ∧ k * x - 1 = 15) →
  k = 2 :=
by sorry

end NUMINAMATH_CALUDE_same_solution_implies_k_equals_two_l1817_181761


namespace NUMINAMATH_CALUDE_max_profit_week3_l1817_181771

/-- Represents the sales and profit data for a bicycle store. -/
structure BicycleStore where
  profit_a : ℕ  -- Profit per type A bicycle
  profit_b : ℕ  -- Profit per type B bicycle
  week1_a : ℕ   -- Week 1 sales of type A
  week1_b : ℕ   -- Week 1 sales of type B
  week1_profit : ℕ  -- Week 1 total profit
  week2_a : ℕ   -- Week 2 sales of type A
  week2_b : ℕ   -- Week 2 sales of type B
  week2_profit : ℕ  -- Week 2 total profit
  week3_total : ℕ  -- Week 3 total sales

/-- Determines if the given sales distribution for Week 3 is valid. -/
def validWeek3Sales (store : BicycleStore) (a : ℕ) (b : ℕ) : Prop :=
  a + b = store.week3_total ∧ b > a ∧ b ≤ 2 * a

/-- Calculates the profit for a given sales distribution. -/
def calculateProfit (store : BicycleStore) (a : ℕ) (b : ℕ) : ℕ :=
  a * store.profit_a + b * store.profit_b

/-- Theorem stating that the maximum profit in Week 3 is achieved by selling 9 type A and 16 type B bicycles. -/
theorem max_profit_week3 (store : BicycleStore) 
    (h1 : store.week1_a * store.profit_a + store.week1_b * store.profit_b = store.week1_profit)
    (h2 : store.week2_a * store.profit_a + store.week2_b * store.profit_b = store.week2_profit)
    (h3 : store.week3_total = 25)
    (h4 : store.profit_a = 80)
    (h5 : store.profit_b = 100) :
    (∀ a b, validWeek3Sales store a b → calculateProfit store a b ≤ 2320) ∧
    validWeek3Sales store 9 16 ∧
    calculateProfit store 9 16 = 2320 := by
  sorry


end NUMINAMATH_CALUDE_max_profit_week3_l1817_181771


namespace NUMINAMATH_CALUDE_moe_eating_time_l1817_181791

/-- Given that a lizard named Moe eats 40 pieces of cuttlebone in 10 seconds, 
    this theorem proves that it takes 200 seconds for Moe to eat 800 pieces. -/
theorem moe_eating_time : ∀ (rate : ℝ) (pieces : ℕ) (time : ℝ),
  rate = 40 / 10 →
  pieces = 800 →
  time = pieces / rate →
  time = 200 := by sorry

end NUMINAMATH_CALUDE_moe_eating_time_l1817_181791


namespace NUMINAMATH_CALUDE_nancy_age_proof_l1817_181760

/-- Nancy's age in years -/
def nancy_age : ℕ := 5

/-- Nancy's grandmother's age in years -/
def grandmother_age : ℕ := 10 * nancy_age

/-- Age difference between Nancy's grandmother and Nancy at Nancy's birth -/
def age_difference : ℕ := 45

theorem nancy_age_proof :
  nancy_age = 5 ∧
  grandmother_age = 10 * nancy_age ∧
  grandmother_age - nancy_age = age_difference :=
by sorry

end NUMINAMATH_CALUDE_nancy_age_proof_l1817_181760


namespace NUMINAMATH_CALUDE_muffin_sale_total_l1817_181793

theorem muffin_sale_total (boys : ℕ) (girls : ℕ) (boys_muffins : ℕ) (girls_muffins : ℕ) : 
  boys = 3 → 
  girls = 2 → 
  boys_muffins = 12 → 
  girls_muffins = 20 → 
  boys * boys_muffins + girls * girls_muffins = 76 := by
sorry

end NUMINAMATH_CALUDE_muffin_sale_total_l1817_181793


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1817_181722

theorem geometric_sequence_problem (a : ℕ → ℝ) :
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  a 1 = 1/4 →
  a 3 * a 5 = 4 * (a 4 - 1) →
  a 2 = 1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1817_181722


namespace NUMINAMATH_CALUDE_shaded_area_square_with_circles_l1817_181730

/-- The shaded area of a square with side length 36 inches containing 9 tangent circles -/
theorem shaded_area_square_with_circles : 
  let square_side : ℝ := 36
  let num_circles : ℕ := 9
  let circle_radius : ℝ := square_side / 6

  let square_area : ℝ := square_side ^ 2
  let total_circles_area : ℝ := num_circles * Real.pi * circle_radius ^ 2
  let shaded_area : ℝ := square_area - total_circles_area

  shaded_area = 1296 - 324 * Real.pi :=
by
  sorry


end NUMINAMATH_CALUDE_shaded_area_square_with_circles_l1817_181730


namespace NUMINAMATH_CALUDE_two_numbers_difference_l1817_181716

theorem two_numbers_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 20) : 
  |x - y| = 2 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l1817_181716


namespace NUMINAMATH_CALUDE_triple_nested_log_sum_l1817_181787

-- Define the logarithm function
noncomputable def log (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

-- Define the theorem
theorem triple_nested_log_sum (x y z : ℝ) :
  log 3 (log 4 (log 5 x)) = 0 ∧
  log 4 (log 5 (log 3 y)) = 0 ∧
  log 5 (log 3 (log 4 z)) = 0 →
  x + y + z = 932 := by
  sorry

end NUMINAMATH_CALUDE_triple_nested_log_sum_l1817_181787


namespace NUMINAMATH_CALUDE_count_even_perfect_square_factors_l1817_181746

/-- The number of even perfect square factors of 2^6 * 7^3 * 3^8 -/
def evenPerfectSquareFactors : ℕ := 30

/-- The original number -/
def originalNumber : ℕ := 2^6 * 7^3 * 3^8

/-- A function that counts the number of even perfect square factors of originalNumber -/
def countEvenPerfectSquareFactors : ℕ := sorry

theorem count_even_perfect_square_factors :
  countEvenPerfectSquareFactors = evenPerfectSquareFactors := by sorry

end NUMINAMATH_CALUDE_count_even_perfect_square_factors_l1817_181746


namespace NUMINAMATH_CALUDE_problem_statement_l1817_181732

theorem problem_statement (x y : ℚ) (hx : x = 5/6) (hy : y = 6/5) : 
  (1/3) * x^8 * y^9 = 2/5 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l1817_181732


namespace NUMINAMATH_CALUDE_complex_fraction_equals_neg_i_l1817_181726

theorem complex_fraction_equals_neg_i : (1 - I) / (1 + I) = -I := by sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_neg_i_l1817_181726


namespace NUMINAMATH_CALUDE_questions_left_blank_l1817_181745

/-- Given a math test with a total number of questions and the number of questions answered,
    prove that the number of questions left blank is the difference between the total and answered questions. -/
theorem questions_left_blank (total : ℕ) (answered : ℕ) (h : answered ≤ total) :
  total - answered = total - answered :=
by sorry

end NUMINAMATH_CALUDE_questions_left_blank_l1817_181745


namespace NUMINAMATH_CALUDE_sum_of_squares_extremes_l1817_181797

theorem sum_of_squares_extremes (a b c : ℝ) : 
  a / b = 2 / 3 ∧ b / c = 3 / 4 ∧ b = 9 → a^2 + c^2 = 180 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_extremes_l1817_181797


namespace NUMINAMATH_CALUDE_employee_hourly_rate_l1817_181773

/-- Proves that the hourly rate for the first 40 hours is $11.25 given the conditions -/
theorem employee_hourly_rate 
  (x : ℝ) -- hourly rate for the first 40 hours
  (overtime_hours : ℝ) -- number of overtime hours
  (overtime_rate : ℝ) -- overtime hourly rate
  (gross_pay : ℝ) -- total gross pay
  (h1 : overtime_hours = 10.75)
  (h2 : overtime_rate = 16)
  (h3 : gross_pay = 622)
  (h4 : 40 * x + overtime_hours * overtime_rate = gross_pay) :
  x = 11.25 :=
by sorry

end NUMINAMATH_CALUDE_employee_hourly_rate_l1817_181773


namespace NUMINAMATH_CALUDE_smallest_n_with_g_having_8_or_higher_l1817_181758

/-- Sum of digits in base b representation of n -/
def sumDigits (n : ℕ) (b : ℕ) : ℕ := sorry

/-- f(n) is the sum of digits in base-five representation of n -/
def f (n : ℕ) : ℕ := sumDigits n 5

/-- g(n) is the sum of digits in base-nine representation of f(n) -/
def g (n : ℕ) : ℕ := sumDigits (f n) 9

/-- A number has a digit '8' or higher in base-nine if it's greater than or equal to 8 -/
def hasDigit8OrHigher (n : ℕ) : Prop := n ≥ 8

theorem smallest_n_with_g_having_8_or_higher :
  (∀ m : ℕ, m < 248 → ¬hasDigit8OrHigher (g m)) ∧ hasDigit8OrHigher (g 248) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_g_having_8_or_higher_l1817_181758


namespace NUMINAMATH_CALUDE_tan_plus_3sin_30_deg_l1817_181734

theorem tan_plus_3sin_30_deg :
  let sin_30 : ℝ := 1/2
  let cos_30 : ℝ := Real.sqrt 3 / 2
  (sin_30 / cos_30) + 3 * sin_30 = 2 + 3 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_tan_plus_3sin_30_deg_l1817_181734


namespace NUMINAMATH_CALUDE_inequality_proof_l1817_181762

theorem inequality_proof (a b c : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)
  (sum_cond : a + b + c = 2) :
  (1 / (1 + a*b)) + (1 / (1 + b*c)) + (1 / (1 + c*a)) ≥ 27/13 ∧
  ((1 / (1 + a*b)) + (1 / (1 + b*c)) + (1 / (1 + c*a)) = 27/13 ↔ a = 2/3 ∧ b = 2/3 ∧ c = 2/3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1817_181762


namespace NUMINAMATH_CALUDE_prime_square_minus_one_divisible_by_twelve_l1817_181754

theorem prime_square_minus_one_divisible_by_twelve (p : ℕ) 
  (h_prime : Nat.Prime p) (h_ge_five : p ≥ 5) : 
  12 ∣ p^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_prime_square_minus_one_divisible_by_twelve_l1817_181754


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1817_181759

/-- A polynomial of degree 4 with coefficients a, b, and c. -/
def polynomial (a b c : ℝ) (x : ℝ) : ℝ :=
  x^4 + a*x^2 + b*x + c

/-- The condition for divisibility by (x-1)^3. -/
def isDivisibleByXMinusOneCubed (a b c : ℝ) : Prop :=
  ∃ q : ℝ → ℝ, ∀ x, polynomial a b c x = (x - 1)^3 * q x

/-- Theorem stating the necessary and sufficient conditions for divisibility. -/
theorem polynomial_divisibility (a b c : ℝ) :
  isDivisibleByXMinusOneCubed a b c ↔ a = 0 ∧ b = 2 ∧ c = -1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l1817_181759


namespace NUMINAMATH_CALUDE_positive_rational_cube_sum_representation_l1817_181744

theorem positive_rational_cube_sum_representation (r : ℚ) (hr : 0 < r) :
  ∃ (a b c d : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ r = (a^3 + b^3 : ℚ) / (c^3 + d^3 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_positive_rational_cube_sum_representation_l1817_181744


namespace NUMINAMATH_CALUDE_peter_statement_consistency_l1817_181775

/-- Represents the day of the week -/
inductive Day
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday

/-- Represents whether a person is telling the truth or lying -/
inductive TruthState
| Truthful | Lying

/-- Represents a statement that can be made -/
inductive Statement
| A | B | C | D | E

/-- Function to determine if a day follows another -/
def follows (d1 d2 : Day) : Prop := sorry

/-- Function to determine if a number is divisible by another -/
def is_divisible_by (n m : Nat) : Prop := sorry

/-- Peter's truth-telling state on a given day -/
def peter_truth_state (d : Day) : TruthState := sorry

/-- The content of each statement -/
def statement_content (s : Statement) (today : Day) : Prop :=
  match s with
  | Statement.A => peter_truth_state (sorry : Day) = TruthState.Lying ∧ 
                   peter_truth_state (sorry : Day) = TruthState.Lying
  | Statement.B => peter_truth_state today = TruthState.Truthful ∧ 
                   peter_truth_state (sorry : Day) = TruthState.Truthful
  | Statement.C => is_divisible_by 2024 11
  | Statement.D => (sorry : Day) = Day.Wednesday
  | Statement.E => follows (sorry : Day) Day.Saturday

/-- The main theorem -/
theorem peter_statement_consistency 
  (today : Day) 
  (statements : Finset Statement) 
  (h1 : statements.card = 4) 
  (h2 : Statement.C ∉ statements) :
  ∀ s ∈ statements, 
    (peter_truth_state today = TruthState.Truthful → statement_content s today) ∧
    (peter_truth_state today = TruthState.Lying → ¬statement_content s today) := by
  sorry


end NUMINAMATH_CALUDE_peter_statement_consistency_l1817_181775


namespace NUMINAMATH_CALUDE_cubic_sum_greater_than_mixed_product_l1817_181779

theorem cubic_sum_greater_than_mixed_product (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a ≠ b) : a^3 + b^3 > a^2*b + a*b^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_greater_than_mixed_product_l1817_181779


namespace NUMINAMATH_CALUDE_amy_balloons_l1817_181740

theorem amy_balloons (james_balloons : ℕ) (difference : ℕ) (h1 : james_balloons = 232) (h2 : difference = 131) :
  james_balloons - difference = 101 :=
sorry

end NUMINAMATH_CALUDE_amy_balloons_l1817_181740


namespace NUMINAMATH_CALUDE_davids_weighted_average_l1817_181790

def english_mark : ℝ := 76
def math_mark : ℝ := 65
def physics_mark : ℝ := 82
def chemistry_mark : ℝ := 67
def biology_mark : ℝ := 85
def history_mark : ℝ := 78
def cs_mark : ℝ := 81

def english_weight : ℝ := 0.10
def math_weight : ℝ := 0.20
def physics_weight : ℝ := 0.15
def chemistry_weight : ℝ := 0.15
def biology_weight : ℝ := 0.10
def history_weight : ℝ := 0.20
def cs_weight : ℝ := 0.10

def weighted_average : ℝ := 
  english_mark * english_weight +
  math_mark * math_weight +
  physics_mark * physics_weight +
  chemistry_mark * chemistry_weight +
  biology_mark * biology_weight +
  history_mark * history_weight +
  cs_mark * cs_weight

theorem davids_weighted_average : weighted_average = 75.15 := by
  sorry

end NUMINAMATH_CALUDE_davids_weighted_average_l1817_181790


namespace NUMINAMATH_CALUDE_min_voters_for_tall_victory_l1817_181737

/-- Represents the voting structure and outcome of the giraffe beauty contest -/
structure GiraffeContest where
  total_voters : ℕ
  num_districts : ℕ
  num_sections_per_district : ℕ
  num_voters_per_section : ℕ
  winner_name : String

/-- Calculates the minimum number of voters required for the winner to secure victory -/
def min_voters_for_victory (contest : GiraffeContest) : ℕ :=
  let districts_to_win := contest.num_districts / 2 + 1
  let sections_to_win_per_district := contest.num_sections_per_district / 2 + 1
  let voters_to_win_per_section := contest.num_voters_per_section / 2 + 1
  districts_to_win * sections_to_win_per_district * voters_to_win_per_section

/-- The main theorem stating the minimum number of voters required for Tall to win -/
theorem min_voters_for_tall_victory (contest : GiraffeContest)
  (h1 : contest.total_voters = 105)
  (h2 : contest.num_districts = 5)
  (h3 : contest.num_sections_per_district = 7)
  (h4 : contest.num_voters_per_section = 3)
  (h5 : contest.winner_name = "Tall")
  : min_voters_for_victory contest = 24 := by
  sorry

#eval min_voters_for_victory {
  total_voters := 105,
  num_districts := 5,
  num_sections_per_district := 7,
  num_voters_per_section := 3,
  winner_name := "Tall"
}

end NUMINAMATH_CALUDE_min_voters_for_tall_victory_l1817_181737


namespace NUMINAMATH_CALUDE_acid_dilution_l1817_181733

/-- Proves that adding 30 ounces of pure water to 50 ounces of a 40% acid solution 
    results in a 25% acid solution -/
theorem acid_dilution (initial_volume : ℝ) (initial_concentration : ℝ) 
    (added_water : ℝ) (final_concentration : ℝ) :
  initial_volume = 50 →
  initial_concentration = 0.4 →
  added_water = 30 →
  final_concentration = 0.25 →
  (initial_volume * initial_concentration) / (initial_volume + added_water) = final_concentration :=
by
  sorry

#check acid_dilution

end NUMINAMATH_CALUDE_acid_dilution_l1817_181733


namespace NUMINAMATH_CALUDE_cubic_polynomial_property_l1817_181766

/-- Represents a cubic polynomial of the form x³ + px² + qx + r -/
structure CubicPolynomial where
  p : ℝ
  q : ℝ
  r : ℝ

/-- The sum of zeros of a cubic polynomial -/
def sumOfZeros (poly : CubicPolynomial) : ℝ := -poly.p

/-- The product of zeros of a cubic polynomial -/
def productOfZeros (poly : CubicPolynomial) : ℝ := -poly.r

/-- The sum of coefficients of a cubic polynomial -/
def sumOfCoefficients (poly : CubicPolynomial) : ℝ := 1 + poly.p + poly.q + poly.r

/-- The y-intercept of a cubic polynomial -/
def yIntercept (poly : CubicPolynomial) : ℝ := poly.r

theorem cubic_polynomial_property (poly : CubicPolynomial) :
  sumOfZeros poly = 2 * productOfZeros poly ∧
  sumOfZeros poly = sumOfCoefficients poly ∧
  yIntercept poly = 5 →
  poly.q = -24 := by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_property_l1817_181766


namespace NUMINAMATH_CALUDE_cone_sphere_ratio_l1817_181789

theorem cone_sphere_ratio (r h : ℝ) (hr : r > 0) :
  (1 / 3 * π * r^2 * h = 1 / 2 * (4 / 3 * π * r^3)) →
  h / r = 2 := by
  sorry

end NUMINAMATH_CALUDE_cone_sphere_ratio_l1817_181789


namespace NUMINAMATH_CALUDE_almond_croissant_price_l1817_181711

def white_bread_price : ℝ := 3.50
def baguette_price : ℝ := 1.50
def sourdough_price : ℝ := 4.50
def total_spent : ℝ := 78.00
def num_weeks : ℕ := 4

def weekly_bread_cost : ℝ := 2 * white_bread_price + baguette_price + 2 * sourdough_price

theorem almond_croissant_price :
  ∃ (croissant_price : ℝ),
    croissant_price * num_weeks + weekly_bread_cost * num_weeks = total_spent ∧
    croissant_price = 8.00 := by
  sorry

end NUMINAMATH_CALUDE_almond_croissant_price_l1817_181711


namespace NUMINAMATH_CALUDE_simplify_cube_root_l1817_181741

theorem simplify_cube_root : ∃ (c d : ℕ+), 
  (2^10 * 5^6 : ℝ)^(1/3) = c * (2 : ℝ)^(1/3) ∧ 
  c.val + d.val = 202 := by
  sorry

end NUMINAMATH_CALUDE_simplify_cube_root_l1817_181741


namespace NUMINAMATH_CALUDE_tournament_prize_interval_l1817_181708

def total_prize : ℕ := 4800
def first_place_prize : ℕ := 2000

def prize_interval (x : ℕ) : Prop :=
  first_place_prize + (first_place_prize - x) + (first_place_prize - 2*x) = total_prize

theorem tournament_prize_interval : ∃ (x : ℕ), prize_interval x ∧ x = 400 := by
  sorry

end NUMINAMATH_CALUDE_tournament_prize_interval_l1817_181708


namespace NUMINAMATH_CALUDE_georges_new_socks_l1817_181781

theorem georges_new_socks (initial_socks : ℝ) (dad_socks : ℝ) (total_socks : ℕ) 
  (h1 : initial_socks = 28)
  (h2 : dad_socks = 4)
  (h3 : total_socks = 68) :
  ↑total_socks - initial_socks - dad_socks = 36 :=
by sorry

end NUMINAMATH_CALUDE_georges_new_socks_l1817_181781


namespace NUMINAMATH_CALUDE_gate_width_scientific_notation_l1817_181748

theorem gate_width_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 0.000000014 = a * 10^n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.4 ∧ n = -8 := by
  sorry

end NUMINAMATH_CALUDE_gate_width_scientific_notation_l1817_181748


namespace NUMINAMATH_CALUDE_common_number_in_list_l1817_181778

theorem common_number_in_list (l : List ℝ) : 
  l.length = 7 →
  (l.take 4).sum / 4 = 6 →
  (l.drop 3).sum / 4 = 9 →
  l.sum / 7 = 55 / 7 →
  ∃ x ∈ l.take 4 ∩ l.drop 3, x = 5 :=
by sorry

end NUMINAMATH_CALUDE_common_number_in_list_l1817_181778


namespace NUMINAMATH_CALUDE_two_a_minus_two_d_is_zero_l1817_181731

/-- Given a function g and constants a, b, c, d, prove that 2a - 2d = 0 -/
theorem two_a_minus_two_d_is_zero
  (a b c d : ℝ)
  (h_abcd : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
  (g : ℝ → ℝ)
  (h_g : ∀ x, g x = (2*a*x - b) / (c*x - 2*d))
  (h_inv : ∀ x, g (g x) = x) :
  2*a - 2*d = 0 := by
  sorry

end NUMINAMATH_CALUDE_two_a_minus_two_d_is_zero_l1817_181731


namespace NUMINAMATH_CALUDE_gcd_2134_1455_ternary_l1817_181788

theorem gcd_2134_1455_ternary : 
  ∃ m : ℕ, 
    Nat.gcd 2134 1455 = m ∧ 
    (Nat.digits 3 m).reverse = [1, 0, 1, 2, 1] :=
by sorry

end NUMINAMATH_CALUDE_gcd_2134_1455_ternary_l1817_181788


namespace NUMINAMATH_CALUDE_total_loaves_served_l1817_181714

/-- Given that a restaurant served 0.5 loaf of wheat bread and 0.4 loaf of white bread,
    prove that the total number of loaves served is 0.9. -/
theorem total_loaves_served (wheat_bread : ℝ) (white_bread : ℝ)
    (h1 : wheat_bread = 0.5)
    (h2 : white_bread = 0.4) :
    wheat_bread + white_bread = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_total_loaves_served_l1817_181714


namespace NUMINAMATH_CALUDE_complex_simplification_l1817_181720

/-- The imaginary unit i -/
def i : ℂ := Complex.I

/-- Theorem stating the equality of the complex expression and its simplified form -/
theorem complex_simplification :
  3 * (2 + i) - i * (3 - i) + 2 * (1 - 2*i) = 7 - 4*i :=
by sorry

end NUMINAMATH_CALUDE_complex_simplification_l1817_181720


namespace NUMINAMATH_CALUDE_inequality_solution_l1817_181723

theorem inequality_solution (x : ℝ) : 
  (6*x^2 + 18*x - 64) / ((3*x - 2)*(x + 5)) < 2 ↔ -5 < x ∧ x < 2/3 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1817_181723


namespace NUMINAMATH_CALUDE_no_valid_n_l1817_181756

theorem no_valid_n : ¬∃ (n : ℕ), n > 0 ∧ 
  (∃ (x : ℕ), n^2 - 21*n + 110 = x^2) ∧ 
  (15 % n = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_valid_n_l1817_181756


namespace NUMINAMATH_CALUDE_complementary_angles_ratio_l1817_181702

/-- Two complementary angles in a ratio of 5:4 have the larger angle measuring 50 degrees -/
theorem complementary_angles_ratio (a b : ℝ) : 
  a + b = 90 →  -- angles are complementary
  a / b = 5 / 4 →  -- ratio of angles is 5:4
  max a b = 50 :=  -- larger angle measures 50 degrees
by sorry

end NUMINAMATH_CALUDE_complementary_angles_ratio_l1817_181702


namespace NUMINAMATH_CALUDE_unequal_probabilities_for_asymmetric_intervals_l1817_181743

/-- Represents a normal distribution with mean μ and variance σ² -/
structure NormalDistribution (μ : ℝ) (σ : ℝ) where
  mean : ℝ := μ
  variance : ℝ := σ^2

/-- Probability of a measurement falling within a given interval for a normal distribution -/
def probability_in_interval (d : NormalDistribution μ σ) (a b : ℝ) : ℝ :=
  sorry

theorem unequal_probabilities_for_asymmetric_intervals 
  (d : NormalDistribution 10 σ) (σ : ℝ) :
  probability_in_interval d 9.9 10.2 ≠ probability_in_interval d 10 10.3 :=
sorry

end NUMINAMATH_CALUDE_unequal_probabilities_for_asymmetric_intervals_l1817_181743


namespace NUMINAMATH_CALUDE_no_consecutive_sum_for_2_14_l1817_181749

theorem no_consecutive_sum_for_2_14 : 
  ¬ ∃ (k : ℕ+) (n : ℕ), 2^14 = (k * (2*n + k + 1)) / 2 := by
sorry

end NUMINAMATH_CALUDE_no_consecutive_sum_for_2_14_l1817_181749


namespace NUMINAMATH_CALUDE_disaster_relief_team_selection_part1_disaster_relief_team_selection_part2_l1817_181755

-- Define the number of internal medicine doctors and surgeons
def num_internal_med : ℕ := 12
def num_surgeons : ℕ := 8

-- Define the number of doctors needed for the team
def team_size : ℕ := 5

-- Theorem for part (1)
theorem disaster_relief_team_selection_part1 :
  (Nat.choose (num_internal_med + num_surgeons - 2) (team_size - 1)) = 3060 :=
sorry

-- Theorem for part (2)
theorem disaster_relief_team_selection_part2 :
  (Nat.choose (num_internal_med + num_surgeons) team_size) -
  (Nat.choose num_surgeons team_size) -
  (Nat.choose num_internal_med team_size) = 14656 :=
sorry

end NUMINAMATH_CALUDE_disaster_relief_team_selection_part1_disaster_relief_team_selection_part2_l1817_181755


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l1817_181780

/-- Given a line and a parabola in a Cartesian coordinate system, 
    this theorem states the conditions for the parabola to intersect 
    the line segment between two points on the line at two distinct points. -/
theorem parabola_line_intersection 
  (a : ℝ) 
  (h_a_neq_zero : a ≠ 0) 
  (h_line : ∀ x y : ℝ, y = (1/2) * x + 1/2 ↔ (x = -1 ∧ y = 0) ∨ (x = 1 ∧ y = 1)) 
  (h_parabola : ∀ x y : ℝ, y = a * x^2 - x + 1) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 
   -1 ≤ x1 ∧ x1 ≤ 1 ∧ -1 ≤ x2 ∧ x2 ≤ 1 ∧
   ((1/2) * x1 + 1/2 = a * x1^2 - x1 + 1) ∧
   ((1/2) * x2 + 1/2 = a * x2^2 - x2 + 1)) ↔
  (1 ≤ a ∧ a < 9/8) :=
by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l1817_181780


namespace NUMINAMATH_CALUDE_marble_arrangement_count_l1817_181707

def total_arrangements (n : ℕ) : ℕ := Nat.factorial n

def adjacent_arrangements (n : ℕ) : ℕ := 2 * Nat.factorial (n - 1)

theorem marble_arrangement_count :
  total_arrangements 5 - adjacent_arrangements 5 = 72 := by
  sorry

end NUMINAMATH_CALUDE_marble_arrangement_count_l1817_181707


namespace NUMINAMATH_CALUDE_quadratic_inequality_minimum_l1817_181785

theorem quadratic_inequality_minimum (a b c : ℝ) 
  (h1 : ∀ x, 3 < x ∧ x < 4 → a * x^2 + b * x + c > 0)
  (h2 : ∀ x, x ≤ 3 ∨ x ≥ 4 → a * x^2 + b * x + c ≤ 0) :
  ∃ m, m = (c^2 + 5) / (a + b) ∧ 
    (∀ k, k = (c^2 + 5) / (a + b) → m ≤ k) ∧
    m = 4 * Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_minimum_l1817_181785


namespace NUMINAMATH_CALUDE_comic_book_arrangement_l1817_181728

def arrange_comic_books (batman : Nat) (superman : Nat) (wonder_woman : Nat) (flash : Nat) : Nat :=
  (Nat.factorial batman) * (Nat.factorial superman) * (Nat.factorial wonder_woman) * (Nat.factorial flash) * (Nat.factorial 4)

theorem comic_book_arrangement :
  arrange_comic_books 8 7 6 5 = 421275894176000 := by
  sorry

end NUMINAMATH_CALUDE_comic_book_arrangement_l1817_181728


namespace NUMINAMATH_CALUDE_divisibility_by_five_l1817_181752

theorem divisibility_by_five (a b : ℕ) (h : 5 ∣ (a * b)) : 5 ∣ a ∨ 5 ∣ b := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_five_l1817_181752


namespace NUMINAMATH_CALUDE_line_through_123_quadrants_l1817_181750

-- Define a line in 2D space
structure Line where
  k : ℝ
  b : ℝ

-- Define the property of a line passing through the first, second, and third quadrants
def passesThrough123Quadrants (l : Line) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    (x₁ > 0 ∧ y₁ > 0) ∧  -- First quadrant
    (x₂ < 0 ∧ y₂ > 0) ∧  -- Second quadrant
    (x₃ < 0 ∧ y₃ < 0) ∧  -- Third quadrant
    (y₁ = l.k * x₁ + l.b) ∧
    (y₂ = l.k * x₂ + l.b) ∧
    (y₃ = l.k * x₃ + l.b)

-- Theorem statement
theorem line_through_123_quadrants (l : Line) :
  passesThrough123Quadrants l → l.k > 0 ∧ l.b > 0 := by
  sorry

end NUMINAMATH_CALUDE_line_through_123_quadrants_l1817_181750


namespace NUMINAMATH_CALUDE_inequality_proof_equality_condition_l1817_181753

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 1) :
  y * z + z * x + x * y ≥ 4 * (y^2 * z^2 + z^2 * x^2 + x^2 * y^2) + 5 * x * y * z :=
by sorry

theorem equality_condition (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 1) :
  (y * z + z * x + x * y = 4 * (y^2 * z^2 + z^2 * x^2 + x^2 * y^2) + 5 * x * y * z) ↔ 
  (x = 1/3 ∧ y = 1/3 ∧ z = 1/3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_equality_condition_l1817_181753


namespace NUMINAMATH_CALUDE_prob_three_tails_correct_l1817_181784

/-- Represents a coin with a given probability of heads -/
structure Coin where
  prob_heads : ℚ
  prob_heads_nonneg : 0 ≤ prob_heads
  prob_heads_le_one : prob_heads ≤ 1

/-- A fair coin with probability of heads = 1/2 -/
def fair_coin : Coin where
  prob_heads := 1/2
  prob_heads_nonneg := by norm_num
  prob_heads_le_one := by norm_num

/-- A biased coin with probability of heads = 2/3 -/
def biased_coin : Coin where
  prob_heads := 2/3
  prob_heads_nonneg := by norm_num
  prob_heads_le_one := by norm_num

/-- Sequence of coins: two fair coins, one biased coin, two fair coins -/
def coin_sequence : List Coin :=
  [fair_coin, fair_coin, biased_coin, fair_coin, fair_coin]

/-- Calculates the probability of getting at least 3 tails in a row -/
def prob_three_tails_in_row (coins : List Coin) : ℚ :=
  sorry

theorem prob_three_tails_correct :
  prob_three_tails_in_row coin_sequence = 13/48 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_tails_correct_l1817_181784


namespace NUMINAMATH_CALUDE_cos_4theta_l1817_181736

theorem cos_4theta (θ : ℝ) (h : Complex.exp (Complex.I * θ) = (3 + Complex.I * Real.sqrt 8) / 5) :
  Real.cos (4 * θ) = -287 / 625 := by
  sorry

end NUMINAMATH_CALUDE_cos_4theta_l1817_181736


namespace NUMINAMATH_CALUDE_prime_triple_problem_l1817_181769

theorem prime_triple_problem (p q r : ℕ) : 
  Prime p → Prime q → Prime r →
  5 ≤ p → p < q → q < r →
  2 * p^2 - r^2 ≥ 49 →
  2 * q^2 - r^2 ≤ 193 →
  p = 17 ∧ q = 19 ∧ r = 23 :=
by sorry

end NUMINAMATH_CALUDE_prime_triple_problem_l1817_181769


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_roots_difference_condition_l1817_181721

theorem quadratic_equation_roots (m : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 + (m + 3) * x + m + 1
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 :=
by sorry

theorem roots_difference_condition (m : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 + (m + 3) * x + m + 1
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ |x₁ - x₂| = 2 * Real.sqrt 2) →
  (m = 1 ∨ m = -3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_roots_difference_condition_l1817_181721
