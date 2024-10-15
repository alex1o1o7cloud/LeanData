import Mathlib

namespace NUMINAMATH_CALUDE_factor_implies_c_value_l2288_228895

theorem factor_implies_c_value (c : ℝ) : 
  (∀ x : ℝ, (x + 5) ∣ (c * x^3 + 23 * x^2 - 5 * c * x + 55)) → c = 6.3 := by
sorry

end NUMINAMATH_CALUDE_factor_implies_c_value_l2288_228895


namespace NUMINAMATH_CALUDE_monomial_count_l2288_228814

/-- An algebraic expression is a monomial if it consists of a single term. -/
def isMonomial (expr : String) : Bool := sorry

/-- The set of given algebraic expressions. -/
def expressions : List String := [
  "3a^2 + b",
  "-2",
  "3xy^3/5",
  "a^2b/3 + 1",
  "a^2 - 3b^2",
  "2abc"
]

/-- Counts the number of monomials in a list of expressions. -/
def countMonomials (exprs : List String) : Nat :=
  exprs.filter isMonomial |>.length

theorem monomial_count :
  countMonomials expressions = 3 := by sorry

end NUMINAMATH_CALUDE_monomial_count_l2288_228814


namespace NUMINAMATH_CALUDE_least_distinct_values_is_184_l2288_228862

/-- Represents a list of positive integers with a unique mode -/
structure IntegerList where
  elements : List Nat
  size : elements.length = 2023
  mode_frequency : Nat
  mode_unique : mode_frequency = 12
  is_mode : elements.count mode_frequency = mode_frequency
  other_frequencies : ∀ n, n ≠ mode_frequency → elements.count n < mode_frequency

/-- The least number of distinct values in the list -/
def leastDistinctValues (list : IntegerList) : Nat :=
  list.elements.toFinset.card

/-- Theorem: The least number of distinct values in the list is 184 -/
theorem least_distinct_values_is_184 (list : IntegerList) :
  leastDistinctValues list = 184 := by
  sorry


end NUMINAMATH_CALUDE_least_distinct_values_is_184_l2288_228862


namespace NUMINAMATH_CALUDE_skt_lineup_count_l2288_228847

/-- The total number of StarCraft programmers -/
def total_programmers : ℕ := 111

/-- The number of programmers in SKT's initial team -/
def initial_team_size : ℕ := 11

/-- The number of programmers needed for the lineup -/
def lineup_size : ℕ := 5

/-- The number of different lineups for SKT's second season opening match -/
def number_of_lineups : ℕ := 
  initial_team_size * (total_programmers - initial_team_size + 1) * 
  (Nat.choose initial_team_size lineup_size) * (Nat.factorial lineup_size)

theorem skt_lineup_count : number_of_lineups = 61593840 := by
  sorry

end NUMINAMATH_CALUDE_skt_lineup_count_l2288_228847


namespace NUMINAMATH_CALUDE_rockham_soccer_league_members_l2288_228863

theorem rockham_soccer_league_members : 
  let sock_cost : ℕ := 6
  let tshirt_cost : ℕ := sock_cost + 7
  let member_cost : ℕ := 2 * (sock_cost + tshirt_cost)
  let custom_fee : ℕ := 200
  let total_cost : ℕ := 2892
  ∃ (n : ℕ), n * member_cost + custom_fee = total_cost ∧ n = 70 :=
by sorry

end NUMINAMATH_CALUDE_rockham_soccer_league_members_l2288_228863


namespace NUMINAMATH_CALUDE_hall_to_cube_edge_l2288_228855

/-- Represents the dimensions of a rectangular hall --/
structure HallDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular hall --/
def hallVolume (d : HallDimensions) : ℝ := d.length * d.width * d.height

/-- Theorem: Given a rectangular hall with specific wall areas, 
    the edge of a cube with the same volume is the cube root of 40 --/
theorem hall_to_cube_edge 
  (d : HallDimensions) 
  (floor_area : d.length * d.width = 20)
  (long_wall_area : d.width * d.height = 10)
  (short_wall_area : d.length * d.height = 8) :
  ∃ (edge : ℝ), edge^3 = hallVolume d ∧ edge^3 = 40 := by
  sorry

end NUMINAMATH_CALUDE_hall_to_cube_edge_l2288_228855


namespace NUMINAMATH_CALUDE_cube_root_equation_solutions_l2288_228853

theorem cube_root_equation_solutions :
  let f (x : ℝ) := (18 * x - 3)^(1/3) + (12 * x + 3)^(1/3) - 5 * x^(1/3)
  { x : ℝ | f x = 0 } = 
    { 0, (-27 + Real.sqrt 18477) / 1026, (-27 - Real.sqrt 18477) / 1026 } := by
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solutions_l2288_228853


namespace NUMINAMATH_CALUDE_firm_employs_100_looms_l2288_228887

/-- Represents the number of looms employed by the textile manufacturing firm. -/
def number_of_looms : ℕ := sorry

/-- The aggregate sales value of the output of the looms in rupees. -/
def aggregate_sales : ℕ := 500000

/-- The monthly manufacturing expenses in rupees. -/
def manufacturing_expenses : ℕ := 150000

/-- The monthly establishment charges in rupees. -/
def establishment_charges : ℕ := 75000

/-- The decrease in profit when one loom breaks down for one month, in rupees. -/
def profit_decrease : ℕ := 3500

theorem firm_employs_100_looms :
  number_of_looms = 100 ∧
  aggregate_sales / number_of_looms - manufacturing_expenses / number_of_looms = profit_decrease :=
sorry

end NUMINAMATH_CALUDE_firm_employs_100_looms_l2288_228887


namespace NUMINAMATH_CALUDE_lowest_price_calculation_l2288_228821

/-- Calculates the lowest price per component to avoid loss --/
def lowest_price_per_component (production_cost shipping_cost : ℚ) 
  (fixed_monthly_cost : ℚ) (production_volume : ℕ) : ℚ :=
  let total_cost := production_volume * (production_cost + shipping_cost) + fixed_monthly_cost
  total_cost / production_volume

/-- Theorem: The lowest price per component is the total cost divided by the number of components --/
theorem lowest_price_calculation (production_cost shipping_cost : ℚ) 
  (fixed_monthly_cost : ℚ) (production_volume : ℕ) :
  lowest_price_per_component production_cost shipping_cost fixed_monthly_cost production_volume = 
  (production_volume * (production_cost + shipping_cost) + fixed_monthly_cost) / production_volume :=
by
  sorry

#eval lowest_price_per_component 80 4 16500 150

end NUMINAMATH_CALUDE_lowest_price_calculation_l2288_228821


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_T_l2288_228803

/-- Represents a three-digit integer -/
structure ThreeDigitInt where
  value : Nat
  is_three_digit : 100 ≤ value ∧ value < 1000

/-- Generates the next term in the sequence based on the current term -/
def next_term (n : ThreeDigitInt) : ThreeDigitInt :=
  { value := (n.value % 100) * 10 + (n.value / 100),
    is_three_digit := sorry }

/-- Generates a sequence of three terms starting with the given number -/
def generate_sequence (start : ThreeDigitInt) : Fin 3 → ThreeDigitInt
| 0 => start
| 1 => next_term start
| 2 => next_term (next_term start)

/-- Calculates the sum of all terms in a sequence -/
def sequence_sum (start : ThreeDigitInt) : Nat :=
  (generate_sequence start 0).value +
  (generate_sequence start 1).value +
  (generate_sequence start 2).value

/-- The starting number for the first sequence -/
def start1 : ThreeDigitInt :=
  { value := 312,
    is_three_digit := sorry }

/-- The starting number for the second sequence -/
def start2 : ThreeDigitInt :=
  { value := 231,
    is_three_digit := sorry }

/-- The sum of all terms from both sequences -/
def T : Nat := sequence_sum start1 + sequence_sum start2

theorem largest_prime_factor_of_T :
  ∃ (p : Nat), Nat.Prime p ∧ p ∣ T ∧ ∀ (q : Nat), Nat.Prime q → q ∣ T → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_T_l2288_228803


namespace NUMINAMATH_CALUDE_pie_eating_contest_l2288_228894

theorem pie_eating_contest (first_student third_student : ℚ) : 
  first_student = 7/8 → third_student = 3/4 → first_student - third_student = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_pie_eating_contest_l2288_228894


namespace NUMINAMATH_CALUDE_chastity_final_money_is_16_49_l2288_228820

/-- Calculates the final amount of money Chastity has after buying candies and giving some to a friend --/
def chastity_final_money (
  lollipop_price : ℚ)
  (gummies_price : ℚ)
  (chips_price : ℚ)
  (chocolate_price : ℚ)
  (discount_rate : ℚ)
  (tax_rate : ℚ)
  (initial_money : ℚ) : ℚ :=
  let total_cost := 4 * lollipop_price + gummies_price + 3 * chips_price + chocolate_price
  let discounted_cost := total_cost * (1 - discount_rate)
  let taxed_cost := discounted_cost * (1 + tax_rate)
  let money_after_purchase := initial_money - taxed_cost
  let friend_payback := 2 * lollipop_price + chips_price
  money_after_purchase + friend_payback

/-- Theorem stating that Chastity's final amount of money is $16.49 --/
theorem chastity_final_money_is_16_49 :
  chastity_final_money 1.5 2 1.25 1.75 0.1 0.05 25 = 16.49 := by
  sorry

end NUMINAMATH_CALUDE_chastity_final_money_is_16_49_l2288_228820


namespace NUMINAMATH_CALUDE_equation_has_four_solutions_l2288_228845

/-- The number of integer solutions to the equation 6y^2 + 3xy + x + 2y - 72 = 0 -/
def num_solutions : ℕ := 4

/-- The equation 6y^2 + 3xy + x + 2y - 72 = 0 -/
def equation (x y : ℤ) : Prop :=
  6 * y^2 + 3 * x * y + x + 2 * y - 72 = 0

theorem equation_has_four_solutions :
  ∃ (s : Finset (ℤ × ℤ)), s.card = num_solutions ∧
  (∀ (p : ℤ × ℤ), p ∈ s ↔ equation p.1 p.2) :=
sorry

end NUMINAMATH_CALUDE_equation_has_four_solutions_l2288_228845


namespace NUMINAMATH_CALUDE_solution_implies_m_value_l2288_228876

theorem solution_implies_m_value (m : ℝ) : 
  (∃ x : ℝ, x^2 + 12*x - m^2 = 0 ∧ x = 2) → m = 2 ∨ m = -2 := by
  sorry

end NUMINAMATH_CALUDE_solution_implies_m_value_l2288_228876


namespace NUMINAMATH_CALUDE_monotonic_absolute_value_function_l2288_228885

def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

theorem monotonic_absolute_value_function (a : ℝ) :
  (∀ x y, x < y ∧ x < -1 ∧ y < -1 → f a x ≤ f a y ∨ f a x ≥ f a y) →
  a ≥ -1 :=
by sorry

end NUMINAMATH_CALUDE_monotonic_absolute_value_function_l2288_228885


namespace NUMINAMATH_CALUDE_x_eighteenth_equals_negative_one_l2288_228882

theorem x_eighteenth_equals_negative_one (x : ℂ) (h : x + 1/x = Real.sqrt 3) : x^18 = -1 := by
  sorry

end NUMINAMATH_CALUDE_x_eighteenth_equals_negative_one_l2288_228882


namespace NUMINAMATH_CALUDE_kite_area_in_square_l2288_228805

/-- Given a 10 cm by 10 cm square with diagonals and a vertical line segment from
    the midpoint of the bottom side to the top side, the area of the kite-shaped
    region formed around the vertical line segment is 25 cm². -/
theorem kite_area_in_square (square_side : ℝ) (kite_area : ℝ) : 
  square_side = 10 → kite_area = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_kite_area_in_square_l2288_228805


namespace NUMINAMATH_CALUDE_angle_sum_is_180_l2288_228886

-- Define angles as real numbers (representing degrees)
variable (angle1 angle2 angle3 : ℝ)

-- Define the properties of vertical angles and supplementary angles
def vertical_angles (a b : ℝ) : Prop := a = b
def supplementary_angles (a b : ℝ) : Prop := a + b = 180

-- State the theorem
theorem angle_sum_is_180 
  (h1 : vertical_angles angle1 angle2) 
  (h2 : supplementary_angles angle2 angle3) : 
  angle1 + angle3 = 180 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_is_180_l2288_228886


namespace NUMINAMATH_CALUDE_chessboard_pawn_placement_l2288_228834

/-- The number of columns on the chessboard -/
def n : ℕ := 8

/-- The number of rows on the chessboard -/
def m : ℕ := 8

/-- The number of ways to place a pawn in a single row -/
def ways_per_row : ℕ := n + 1

/-- The total number of ways to place pawns on the chessboard -/
def total_ways : ℕ := ways_per_row ^ m

theorem chessboard_pawn_placement :
  total_ways = 3^16 :=
sorry

end NUMINAMATH_CALUDE_chessboard_pawn_placement_l2288_228834


namespace NUMINAMATH_CALUDE_tony_grocery_distance_l2288_228838

/-- Represents the distance Tony needs to drive for his errands -/
structure TonyErrands where
  halfway_distance : ℝ
  haircut_distance : ℝ
  doctor_distance : ℝ

/-- Calculates the distance Tony needs to drive for groceries -/
def grocery_distance (e : TonyErrands) : ℝ :=
  2 * e.halfway_distance - (e.haircut_distance + e.doctor_distance)

/-- Theorem stating that Tony needs to drive 10 miles for groceries -/
theorem tony_grocery_distance :
  ∀ (e : TonyErrands),
    e.halfway_distance = 15 →
    e.haircut_distance = 15 →
    e.doctor_distance = 5 →
    grocery_distance e = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_tony_grocery_distance_l2288_228838


namespace NUMINAMATH_CALUDE_intersection_empty_implies_a_nonnegative_l2288_228809

theorem intersection_empty_implies_a_nonnegative 
  (A : Set ℝ) (B : Set ℝ) (a : ℝ) 
  (h1 : A = {x : ℝ | x - a > 0})
  (h2 : B = {x : ℝ | x ≤ 0})
  (h3 : A ∩ B = ∅) :
  a ≥ 0 := by
sorry

end NUMINAMATH_CALUDE_intersection_empty_implies_a_nonnegative_l2288_228809


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2288_228843

open Complex

theorem complex_equation_solution (a : ℝ) : 
  (1 - I)^3 / (1 + I) = a + 3*I → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2288_228843


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l2288_228880

theorem solution_set_of_inequality (x : ℝ) :
  (2 * x^2 - x > 0) ↔ (x < 0 ∨ x > 1/2) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l2288_228880


namespace NUMINAMATH_CALUDE_transport_cost_effectiveness_l2288_228888

/-- Represents the transportation cost functions and conditions for fruit shipping --/
structure FruitTransport where
  x : ℝ  -- distance in kilometers
  truck_cost : ℝ → ℝ  -- trucking company cost function
  train_cost : ℝ → ℝ  -- train freight station cost function

/-- Theorem stating the cost-effectiveness of different transportation methods --/
theorem transport_cost_effectiveness (ft : FruitTransport) 
  (h_truck : ft.truck_cost = λ x => 94 * x + 4000)
  (h_train : ft.train_cost = λ x => 81 * x + 6600) :
  (∀ x, x > 0 ∧ x < 200 → ft.truck_cost x < ft.train_cost x) ∧
  (∀ x, x > 200 → ft.train_cost x < ft.truck_cost x) := by
  sorry

#check transport_cost_effectiveness

end NUMINAMATH_CALUDE_transport_cost_effectiveness_l2288_228888


namespace NUMINAMATH_CALUDE_stratified_sampling_school_a_l2288_228861

theorem stratified_sampling_school_a (school_a : ℕ) (school_b : ℕ) (school_c : ℕ) (sample_size : ℕ) :
  school_a = 3600 →
  school_b = 5400 →
  school_c = 1800 →
  sample_size = 90 →
  (school_a * sample_size) / (school_a + school_b + school_c) = 30 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_school_a_l2288_228861


namespace NUMINAMATH_CALUDE_geometric_series_sum_l2288_228824

theorem geometric_series_sum : 
  let a : ℚ := 2/3
  let r : ℚ := -1/2
  let n : ℕ := 6
  let S : ℚ := a * (1 - r^n) / (1 - r)
  S = 7/16 := by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l2288_228824


namespace NUMINAMATH_CALUDE_cos_sin_power_relation_l2288_228828

theorem cos_sin_power_relation (x a : Real) (h : Real.cos x ^ 6 + Real.sin x ^ 6 = a) :
  Real.cos x ^ 4 + Real.sin x ^ 4 = (1 + 2 * a) / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_power_relation_l2288_228828


namespace NUMINAMATH_CALUDE_prairie_area_l2288_228808

/-- The total area of a prairie given the dust-covered and untouched areas -/
theorem prairie_area (dust_covered : ℕ) (untouched : ℕ) 
  (h1 : dust_covered = 64535) 
  (h2 : untouched = 522) : 
  dust_covered + untouched = 65057 := by
  sorry

end NUMINAMATH_CALUDE_prairie_area_l2288_228808


namespace NUMINAMATH_CALUDE_eighty_six_million_scientific_notation_l2288_228846

/-- Expresses 86 million in scientific notation -/
theorem eighty_six_million_scientific_notation :
  (86000000 : ℝ) = 8.6 * 10^7 := by
  sorry

end NUMINAMATH_CALUDE_eighty_six_million_scientific_notation_l2288_228846


namespace NUMINAMATH_CALUDE_quadratic_shift_l2288_228879

/-- Represents a quadratic function of the form y = (x - h)² + k --/
structure QuadraticFunction where
  h : ℝ
  k : ℝ

/-- Shifts a quadratic function horizontally --/
def shift_horizontal (f : QuadraticFunction) (d : ℝ) : QuadraticFunction :=
  { h := f.h - d, k := f.k }

/-- Shifts a quadratic function vertically --/
def shift_vertical (f : QuadraticFunction) (d : ℝ) : QuadraticFunction :=
  { h := f.h, k := f.k - d }

/-- The main theorem stating that shifting y = (x + 1)² + 3 by 2 units right and 1 unit down
    results in y = (x - 1)² + 2 --/
theorem quadratic_shift :
  let f := QuadraticFunction.mk (-1) 3
  let g := shift_vertical (shift_horizontal f 2) 1
  g = QuadraticFunction.mk 1 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_shift_l2288_228879


namespace NUMINAMATH_CALUDE_triangle_max_perimeter_l2288_228869

theorem triangle_max_perimeter :
  ∀ x : ℕ,
  x > 0 →
  x ≤ 20 →
  x + 4*x > 20 →
  x + 20 > 4*x →
  4*x + 20 > x →
  ∀ y : ℕ,
  y > 0 →
  y ≤ 20 →
  y + 4*y > 20 →
  y + 20 > 4*y →
  4*y + 20 > y →
  x + 4*x + 20 ≥ y + 4*y + 20 →
  x + 4*x + 20 ≤ 50 :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_perimeter_l2288_228869


namespace NUMINAMATH_CALUDE_rectangle_length_l2288_228813

/-- Proves that a rectangle with length 2 cm more than width and perimeter 20 cm has length 6 cm -/
theorem rectangle_length (width : ℝ) (length : ℝ) (perimeter : ℝ) : 
  length = width + 2 →
  perimeter = 2 * length + 2 * width →
  perimeter = 20 →
  length = 6 := by
sorry

end NUMINAMATH_CALUDE_rectangle_length_l2288_228813


namespace NUMINAMATH_CALUDE_smallest_integer_above_sqrt5_plus_sqrt3_to_6th_l2288_228830

theorem smallest_integer_above_sqrt5_plus_sqrt3_to_6th (x : ℝ) :
  x = (Real.sqrt 5 + Real.sqrt 3)^6 → ⌈x⌉ = 3323 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_above_sqrt5_plus_sqrt3_to_6th_l2288_228830


namespace NUMINAMATH_CALUDE_smallest_M_for_inequality_l2288_228875

theorem smallest_M_for_inequality : 
  ∃ M : ℝ, (∀ a b c : ℝ, 
    |a*b*(a^2 - b^2) + b*c*(b^2 - c^2) + c*a*(c^2 - a^2)| ≤ M * (a^2 + b^2 + c^2)^2) ∧ 
  (∀ M' : ℝ, (∀ a b c : ℝ, 
    |a*b*(a^2 - b^2) + b*c*(b^2 - c^2) + c*a*(c^2 - a^2)| ≤ M' * (a^2 + b^2 + c^2)^2) → M ≤ M') ∧
  M = (9 * Real.sqrt 2) / 32 := by
sorry

end NUMINAMATH_CALUDE_smallest_M_for_inequality_l2288_228875


namespace NUMINAMATH_CALUDE_number_of_petunia_flats_l2288_228856

/-- The number of petunias per flat of petunias -/
def petunias_per_flat : ℕ := 8

/-- The amount of fertilizer needed for each petunia in ounces -/
def fertilizer_per_petunia : ℕ := 8

/-- The number of flats of roses -/
def rose_flats : ℕ := 3

/-- The number of roses per flat of roses -/
def roses_per_flat : ℕ := 6

/-- The amount of fertilizer needed for each rose in ounces -/
def fertilizer_per_rose : ℕ := 3

/-- The number of Venus flytraps -/
def venus_flytraps : ℕ := 2

/-- The amount of fertilizer needed for each Venus flytrap in ounces -/
def fertilizer_per_venus_flytrap : ℕ := 2

/-- The total amount of fertilizer needed in ounces -/
def total_fertilizer : ℕ := 314

/-- The theorem stating that the number of flats of petunias is 4 -/
theorem number_of_petunia_flats : 
  ∃ (P : ℕ), P * (petunias_per_flat * fertilizer_per_petunia) + 
             (rose_flats * roses_per_flat * fertilizer_per_rose) + 
             (venus_flytraps * fertilizer_per_venus_flytrap) = total_fertilizer ∧ 
             P = 4 :=
by sorry

end NUMINAMATH_CALUDE_number_of_petunia_flats_l2288_228856


namespace NUMINAMATH_CALUDE_ac_less_than_bc_l2288_228833

theorem ac_less_than_bc (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) : a * c < b * c := by
  sorry

end NUMINAMATH_CALUDE_ac_less_than_bc_l2288_228833


namespace NUMINAMATH_CALUDE_tangent_line_at_one_monotone_condition_equivalent_to_range_l2288_228872

noncomputable def f (a x : ℝ) : ℝ := a * x^2 - (a + 2) * x + Real.log x

theorem tangent_line_at_one (a : ℝ) (h : a = 1) :
  ∃ (k b : ℝ), ∀ x, k * x + b = f a x + (f a 1 - f a x) * (x - 1) / (1 - x) ∧ k = 0 ∧ b = -2 :=
sorry

theorem monotone_condition_equivalent_to_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → f a x₁ + 2*x₁ < f a x₂ + 2*x₂) ↔ 0 ≤ a ∧ a ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_monotone_condition_equivalent_to_range_l2288_228872


namespace NUMINAMATH_CALUDE_inverse_matrix_problem_l2288_228835

theorem inverse_matrix_problem (A : Matrix (Fin 2) (Fin 2) ℝ) :
  A⁻¹ = !![1, 0; 0, 2] → A = !![1, 0; 0, (1/2)] := by
  sorry

end NUMINAMATH_CALUDE_inverse_matrix_problem_l2288_228835


namespace NUMINAMATH_CALUDE_circle_tangent_line_l2288_228871

/-- Given a circle x^2 + y^2 = r^2 and a point P(x₀, y₀) on the circle,
    the tangent line at P has the equation x₀x + y₀y = r^2 -/
theorem circle_tangent_line (r x₀ y₀ : ℝ) (h : x₀^2 + y₀^2 = r^2) :
  ∀ x y : ℝ, (x - x₀)^2 + (y - y₀)^2 = 0 ↔ x₀*x + y₀*y = r^2 := by
sorry

end NUMINAMATH_CALUDE_circle_tangent_line_l2288_228871


namespace NUMINAMATH_CALUDE_even_odd_function_sum_l2288_228829

-- Define an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define an odd function
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem even_odd_function_sum (f g : ℝ → ℝ) 
  (hf : IsEven f) (hg : IsOdd g) 
  (h : ∀ x, f x + g x = x^2 + 3*x + 1) : 
  ∀ x, f x = x^2 + 1 := by
sorry

end NUMINAMATH_CALUDE_even_odd_function_sum_l2288_228829


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2288_228844

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, x ≥ 0 → 2^x + x - 1 ≥ 0)) ↔ (∃ x : ℝ, x ≥ 0 ∧ 2^x + x - 1 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2288_228844


namespace NUMINAMATH_CALUDE_highest_score_is_96_l2288_228873

def standard_score : ℝ := 85

def deviations : List ℝ := [-9, -4, 11, -7, 0]

def actual_scores : List ℝ := deviations.map (λ x => x + standard_score)

theorem highest_score_is_96 : 
  ∀ (score : ℝ), score ∈ actual_scores → score ≤ 96 :=
by sorry

end NUMINAMATH_CALUDE_highest_score_is_96_l2288_228873


namespace NUMINAMATH_CALUDE_inequality_and_minimum_l2288_228831

theorem inequality_and_minimum (a b x y : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0) (hy : y > 0) :
  let f := fun (t : ℝ) => 2 / t + 9 / (1 - 2 * t)
  -- Part I: Inequality and equality condition
  (a^2 / x + b^2 / y ≥ (a + b)^2 / (x + y)) ∧
  (a^2 / x + b^2 / y = (a + b)^2 / (x + y) ↔ a * y = b * x) ∧
  -- Part II: Minimum value and x value for minimum
  (∀ t ∈ Set.Ioo 0 (1/2), f t ≥ 25) ∧
  (f (1/5) = 25) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_minimum_l2288_228831


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l2288_228851

/-- Given two quadratic equations with a specific relationship between their roots,
    prove that the ratio of certain coefficients is 27. -/
theorem quadratic_root_relation (k n p : ℝ) : 
  k ≠ 0 → n ≠ 0 → p ≠ 0 →
  (∃ r₁ r₂ : ℝ, r₁ + r₂ = -p ∧ r₁ * r₂ = k) →
  (∃ s₁ s₂ : ℝ, s₁ + s₂ = -k ∧ s₁ * s₂ = n ∧ s₁ = 3*r₁ ∧ s₂ = 3*r₂) →
  n / p = 27 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l2288_228851


namespace NUMINAMATH_CALUDE_minimum_games_for_95_percent_win_rate_l2288_228883

theorem minimum_games_for_95_percent_win_rate : 
  ∃ N : ℕ, (N = 37 ∧ (1 + N : ℚ) / (3 + N) ≥ 95 / 100) ∧
  ∀ M : ℕ, M < N → (1 + M : ℚ) / (3 + M) < 95 / 100 := by
  sorry

end NUMINAMATH_CALUDE_minimum_games_for_95_percent_win_rate_l2288_228883


namespace NUMINAMATH_CALUDE_mean_height_of_players_l2288_228832

def heights : List ℕ := [47, 48, 50, 51, 51, 54, 55, 56, 56, 57, 61, 63, 64, 64, 65, 67]

theorem mean_height_of_players : 
  (heights.sum : ℚ) / heights.length = 56.8125 := by
  sorry

end NUMINAMATH_CALUDE_mean_height_of_players_l2288_228832


namespace NUMINAMATH_CALUDE_number_puzzle_l2288_228812

theorem number_puzzle : ∃ x : ℚ, (x / 4 + 15 = 4 * x - 15) ∧ (x = 8) := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l2288_228812


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l2288_228866

/-- The discriminant of a quadratic equation ax^2 + bx + c = 0 is b^2 - 4ac -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- The quadratic equation x^2 - 4x - 11 = 0 has discriminant 60 -/
theorem quadratic_discriminant :
  discriminant 1 (-4) (-11) = 60 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l2288_228866


namespace NUMINAMATH_CALUDE_rachel_lost_lives_l2288_228881

/- Define the initial number of lives -/
def initial_lives : ℕ := 10

/- Define the number of lives gained -/
def lives_gained : ℕ := 26

/- Define the final number of lives -/
def final_lives : ℕ := 32

/- Theorem: Rachel lost 4 lives in the hard part -/
theorem rachel_lost_lives :
  ∃ (lives_lost : ℕ), initial_lives - lives_lost + lives_gained = final_lives ∧ lives_lost = 4 := by
  sorry

end NUMINAMATH_CALUDE_rachel_lost_lives_l2288_228881


namespace NUMINAMATH_CALUDE_airplane_seats_proof_l2288_228818

-- Define the total number of seats
def total_seats : ℕ := 540

-- Define the number of First Class seats
def first_class_seats : ℕ := 54

-- Define the proportion of Business Class seats
def business_class_proportion : ℚ := 3 / 10

-- Define the proportion of Economy Class seats
def economy_class_proportion : ℚ := 6 / 10

-- Theorem statement
theorem airplane_seats_proof :
  (first_class_seats : ℚ) + 
  (business_class_proportion * total_seats) + 
  (economy_class_proportion * total_seats) = total_seats ∧
  economy_class_proportion = 2 * business_class_proportion :=
by sorry


end NUMINAMATH_CALUDE_airplane_seats_proof_l2288_228818


namespace NUMINAMATH_CALUDE_linear_equation_k_value_l2288_228852

theorem linear_equation_k_value (k : ℤ) : 
  (∀ x : ℝ, ∃ a b : ℝ, (k - 3) * x^(abs k - 2) + 5 = a * x + b) → 
  k = -3 := by
sorry

end NUMINAMATH_CALUDE_linear_equation_k_value_l2288_228852


namespace NUMINAMATH_CALUDE_kris_age_l2288_228849

/-- Herbert's age next year -/
def herbert_next_year : ℕ := 15

/-- Age difference between Kris and Herbert -/
def age_difference : ℕ := 10

/-- Herbert's current age -/
def herbert_current : ℕ := herbert_next_year - 1

/-- Kris's current age -/
def kris_current : ℕ := herbert_current + age_difference

theorem kris_age : kris_current = 24 := by
  sorry

end NUMINAMATH_CALUDE_kris_age_l2288_228849


namespace NUMINAMATH_CALUDE_minimum_games_for_percentage_l2288_228806

theorem minimum_games_for_percentage (N : ℕ) : N = 7 ↔ 
  (N ≥ 0) ∧ 
  (∀ k : ℕ, k ≥ 0 → (2 : ℚ) / (3 + k) ≥ (9 : ℚ) / 10 → k ≥ N) ∧
  ((2 : ℚ) / (3 + N) ≥ (9 : ℚ) / 10) :=
by sorry

end NUMINAMATH_CALUDE_minimum_games_for_percentage_l2288_228806


namespace NUMINAMATH_CALUDE_final_expression_l2288_228858

theorem final_expression (x y : ℕ) : x + 2*y + x + 3*y + x + 4*y + x + y = 4*x + 10*y := by
  sorry

end NUMINAMATH_CALUDE_final_expression_l2288_228858


namespace NUMINAMATH_CALUDE_only_f₂_is_saturated_l2288_228898

/-- Definition of a "saturated function of 1" -/
def is_saturated_function_of_1 (f : ℝ → ℝ) : Prop :=
  ∃ x₀ : ℝ, f (x₀ + 1) = f x₀ + f 1

/-- Function f₁(x) = 1/x -/
noncomputable def f₁ (x : ℝ) : ℝ := 1 / x

/-- Function f₂(x) = 2^x -/
noncomputable def f₂ (x : ℝ) : ℝ := 2^x

/-- Function f₃(x) = log(x² + 2) -/
noncomputable def f₃ (x : ℝ) : ℝ := Real.log (x^2 + 2)

/-- Theorem stating that only f₂ is a "saturated function of 1" -/
theorem only_f₂_is_saturated :
  ¬ is_saturated_function_of_1 f₁ ∧
  is_saturated_function_of_1 f₂ ∧
  ¬ is_saturated_function_of_1 f₃ :=
sorry

end NUMINAMATH_CALUDE_only_f₂_is_saturated_l2288_228898


namespace NUMINAMATH_CALUDE_max_rectangle_area_garden_max_area_l2288_228857

/-- The maximum area of a rectangle with a fixed perimeter -/
theorem max_rectangle_area (p : ℝ) (h : p > 0) : 
  (∃ l w : ℝ, l > 0 ∧ w > 0 ∧ 2 * (l + w) = p ∧ 
    ∀ l' w' : ℝ, l' > 0 → w' > 0 → 2 * (l' + w') = p → l * w ≥ l' * w') →
  ∃ l w : ℝ, l > 0 ∧ w > 0 ∧ 2 * (l + w) = p ∧ l * w = (p / 4) ^ 2 :=
by sorry

/-- The maximum area of a rectangle with perimeter 400 feet is 10000 square feet -/
theorem garden_max_area : 
  ∃ l w : ℝ, l > 0 ∧ w > 0 ∧ 2 * (l + w) = 400 ∧ l * w = 10000 :=
by sorry

end NUMINAMATH_CALUDE_max_rectangle_area_garden_max_area_l2288_228857


namespace NUMINAMATH_CALUDE_f_one_third_bounds_l2288_228848

def f_conditions (f : ℝ → ℝ) : Prop :=
  (∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ f x ∧ f x ≤ 1) ∧
  f 0 = 0 ∧
  f 1 = 1 ∧
  ∀ x y z, 0 ≤ x ∧ x < y ∧ y < z ∧ z ≤ 1 ∧ z - y = y - x →
    (1/2 : ℝ) ≤ (f z - f y) / (f y - f x) ∧ (f z - f y) / (f y - f x) ≤ 2

theorem f_one_third_bounds (f : ℝ → ℝ) (h : f_conditions f) :
  (1/7 : ℝ) ≤ f (1/3) ∧ f (1/3) ≤ 4/7 := by
  sorry

end NUMINAMATH_CALUDE_f_one_third_bounds_l2288_228848


namespace NUMINAMATH_CALUDE_stickers_per_pack_l2288_228897

/-- Proves that the number of stickers in each pack is 30 --/
theorem stickers_per_pack (
  num_packs : ℕ)
  (cost_per_sticker : ℚ)
  (total_cost : ℚ)
  (h1 : num_packs = 4)
  (h2 : cost_per_sticker = 1/10)
  (h3 : total_cost = 12) :
  (total_cost / cost_per_sticker) / num_packs = 30 := by
  sorry

#check stickers_per_pack

end NUMINAMATH_CALUDE_stickers_per_pack_l2288_228897


namespace NUMINAMATH_CALUDE_trig_identity_l2288_228896

theorem trig_identity (α : Real) 
  (h : Real.cos (π / 6 - α) = Real.sqrt 3 / 3) : 
  Real.cos (5 * π / 6 + α) + Real.sin (α - π / 6) ^ 2 = (2 - Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2288_228896


namespace NUMINAMATH_CALUDE_profit_per_meter_cloth_l2288_228836

theorem profit_per_meter_cloth (cloth_length : ℝ) (selling_price : ℝ) (cost_price_per_meter : ℝ)
  (h1 : cloth_length = 80)
  (h2 : selling_price = 6900)
  (h3 : cost_price_per_meter = 66.25) :
  (selling_price - cloth_length * cost_price_per_meter) / cloth_length = 20 := by
  sorry

end NUMINAMATH_CALUDE_profit_per_meter_cloth_l2288_228836


namespace NUMINAMATH_CALUDE_solution_s_l2288_228877

theorem solution_s (s : ℝ) : 
  Real.sqrt (3 * Real.sqrt (s - 3)) = (9 - s) ^ (1/4) → s = 3.6 := by
  sorry

end NUMINAMATH_CALUDE_solution_s_l2288_228877


namespace NUMINAMATH_CALUDE_incorrect_reasoning_l2288_228817

-- Define the types for points, lines, and planes
variable (Point Line Plane : Type)

-- Define the belonging relation
variable (belongs_to : Point → Line → Prop)
variable (belongs_to_plane : Point → Plane → Prop)

-- Define the subset relation for a line and a plane
variable (line_subset_plane : Line → Plane → Prop)

-- State the theorem
theorem incorrect_reasoning 
  (l : Line) (α : Plane) (A : Point) :
  ¬(∀ (l : Line) (α : Plane) (A : Point), 
    (¬(line_subset_plane l α) ∧ belongs_to A l) → ¬(belongs_to_plane A α)) :=
sorry

end NUMINAMATH_CALUDE_incorrect_reasoning_l2288_228817


namespace NUMINAMATH_CALUDE_hyperbola_foci_distance_l2288_228867

/-- The distance between the foci of a hyperbola defined by xy = 2 is 4. -/
theorem hyperbola_foci_distance :
  ∃ (f₁ f₂ : ℝ × ℝ), 
    (∀ (x y : ℝ), x * y = 2 → 
      (Real.sqrt ((x - f₁.1)^2 + (y - f₁.2)^2) + Real.sqrt ((x - f₂.1)^2 + (y - f₂.2)^2)) = 
      (Real.sqrt ((x + f₁.1)^2 + (y + f₁.2)^2) + Real.sqrt ((x + f₂.1)^2 + (y + f₂.2)^2))) ∧
    Real.sqrt ((f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2) = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_hyperbola_foci_distance_l2288_228867


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2288_228816

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The 7th term of the sequence -/
def a_7 (a : ℕ → ℝ) (m : ℝ) : Prop := a 7 = m

/-- The 14th term of the sequence -/
def a_14 (a : ℕ → ℝ) (n : ℝ) : Prop := a 14 = n

/-- Theorem: In an arithmetic sequence, if a₇ = m and a₁₄ = n, then a₂₁ = 2n - m -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) (m n : ℝ) 
  (h1 : arithmetic_sequence a) (h2 : a_7 a m) (h3 : a_14 a n) : 
  a 21 = 2 * n - m := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2288_228816


namespace NUMINAMATH_CALUDE_stating_plane_landing_time_l2288_228807

/-- Represents the scenario of a mail delivery between a post office and an airfield -/
structure MailDeliveryScenario where
  usual_travel_time : ℕ  -- Usual one-way travel time in minutes
  early_arrival : ℕ      -- How many minutes earlier the Moskvich arrived
  truck_travel_time : ℕ  -- How long the truck traveled before meeting Moskvich

/-- 
Theorem stating that under the given conditions, the plane must have landed 40 minutes early.
-/
theorem plane_landing_time (scenario : MailDeliveryScenario) 
  (h1 : scenario.early_arrival = 20)
  (h2 : scenario.truck_travel_time = 30) :
  40 = (scenario.truck_travel_time + (scenario.early_arrival / 2)) :=
by sorry

end NUMINAMATH_CALUDE_stating_plane_landing_time_l2288_228807


namespace NUMINAMATH_CALUDE_part1_part2_l2288_228822

-- Define the function f(x) = x / (e^x)
noncomputable def f (x : ℝ) : ℝ := x / Real.exp x

-- Define the function g(x) = f(x) - m
noncomputable def g (x m : ℝ) : ℝ := f x - m

-- Theorem for part 1
theorem part1 (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ g x₁ m = 0 ∧ g x₂ m = 0 ∧
   ∀ x, x > 0 → g x m = 0 → x = x₁ ∨ x = x₂) →
  0 < m ∧ m < 1 / Real.exp 1 :=
sorry

-- Theorem for part 2
theorem part2 (a : ℝ) :
  (∃! n : ℤ, (f n)^2 - a * f n > 0 ∧ ∀ x : ℝ, x > 0 → (f x)^2 - a * f x > 0 → ⌊x⌋ = n) →
  2 / Real.exp 2 ≤ a ∧ a < 1 / Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_part1_part2_l2288_228822


namespace NUMINAMATH_CALUDE_coffee_beans_remaining_l2288_228892

theorem coffee_beans_remaining (J B B_remaining : ℝ) 
  (h1 : J = 0.25 * (J + B))
  (h2 : J + B_remaining = 0.60 * (J + B))
  (h3 : J > 0)
  (h4 : B > 0) :
  B_remaining / B = 7 / 15 := by
sorry

end NUMINAMATH_CALUDE_coffee_beans_remaining_l2288_228892


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2288_228899

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ) :
  (∀ x : ℝ, (x^2 + 1) * (2*x + 1)^9 = a + a₁*(x+2) + a₂*(x+2)^2 + a₃*(x+2)^3 + 
    a₄*(x+2)^4 + a₅*(x+2)^5 + a₆*(x+2)^6 + a₇*(x+2)^7 + a₈*(x+2)^8 + 
    a₉*(x+2)^9 + a₁₀*(x+2)^10 + a₁₁*(x+2)^11) →
  a + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁ = -2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2288_228899


namespace NUMINAMATH_CALUDE_dime_position_l2288_228889

/-- Represents the two possible coin values -/
inductive CoinValue : Type
  | nickel : CoinValue
  | dime : CoinValue

/-- Represents the two possible pocket locations -/
inductive Pocket : Type
  | left : Pocket
  | right : Pocket

/-- Returns the value of a coin in cents -/
def coinValue (c : CoinValue) : Nat :=
  match c with
  | CoinValue.nickel => 5
  | CoinValue.dime => 10

/-- Represents the arrangement of coins in pockets -/
structure CoinArrangement :=
  (leftCoin : CoinValue)
  (rightCoin : CoinValue)

/-- Calculates the sum based on the given formula -/
def calculateSum (arr : CoinArrangement) : Nat :=
  3 * (coinValue arr.rightCoin) + 2 * (coinValue arr.leftCoin)

/-- The main theorem to prove -/
theorem dime_position (arr : CoinArrangement) :
  Even (calculateSum arr) ↔ arr.rightCoin = CoinValue.dime :=
sorry

end NUMINAMATH_CALUDE_dime_position_l2288_228889


namespace NUMINAMATH_CALUDE_problem_solution_l2288_228819

theorem problem_solution (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) :
  x^2 - y^2 = 80 ∧ x * y = 96 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2288_228819


namespace NUMINAMATH_CALUDE_sanity_indeterminable_likely_vampire_l2288_228884

-- Define the types of beings
inductive Being
| Human
| Vampire

-- Define the mental state
inductive MentalState
| Sane
| Insane

-- Define the claim
def claimsLostMind (b : Being) : Prop := true

-- Theorem 1: It's impossible to determine sanity from the claim
theorem sanity_indeterminable (b : Being) (claim : claimsLostMind b) :
  ¬ ∃ (state : MentalState), (b = Being.Human → state = MentalState.Sane) ∧
                             (b = Being.Vampire → state = MentalState.Sane) :=
sorry

-- Theorem 2: The being is most likely a vampire
theorem likely_vampire (b : Being) (claim : claimsLostMind b) :
  b = Being.Vampire :=
sorry

end NUMINAMATH_CALUDE_sanity_indeterminable_likely_vampire_l2288_228884


namespace NUMINAMATH_CALUDE_visibility_time_correct_l2288_228802

/-- The time when Steve and Laura can see each other again -/
def visibility_time : ℝ := 45

/-- Steve's walking speed in feet per second -/
def steve_speed : ℝ := 3

/-- Laura's walking speed in feet per second -/
def laura_speed : ℝ := 1

/-- Distance between Steve and Laura's parallel paths in feet -/
def path_distance : ℝ := 240

/-- Diameter of the circular art installation in feet -/
def installation_diameter : ℝ := 80

/-- Initial separation between Steve and Laura when hidden by the art installation in feet -/
def initial_separation : ℝ := 230

/-- Theorem stating that the visibility time is correct given the problem conditions -/
theorem visibility_time_correct :
  ∃ (steve_pos laura_pos : ℝ × ℝ),
    let steve_final := (steve_pos.1 + steve_speed * visibility_time, steve_pos.2)
    let laura_final := (laura_pos.1 + laura_speed * visibility_time, laura_pos.2)
    (steve_pos.2 - laura_pos.2 = path_distance) ∧
    ((steve_pos.1 - laura_pos.1)^2 + (steve_pos.2 - laura_pos.2)^2 = initial_separation^2) ∧
    (∃ (center : ℝ × ℝ), 
      (center.1 - steve_pos.1)^2 + ((center.2 - steve_pos.2) - path_distance/2)^2 = (installation_diameter/2)^2 ∧
      (center.1 - laura_pos.1)^2 + ((center.2 - laura_pos.2) + path_distance/2)^2 = (installation_diameter/2)^2) ∧
    ((steve_final.1 - laura_final.1)^2 + (steve_final.2 - laura_final.2)^2 > 
     (steve_pos.1 - laura_pos.1)^2 + (steve_pos.2 - laura_pos.2)^2) ∧
    (∀ t : ℝ, 0 < t → t < visibility_time →
      ∃ (x y : ℝ), 
        x^2 + y^2 = (installation_diameter/2)^2 ∧
        (y - steve_pos.2) * (steve_pos.1 + steve_speed * t - x) = 
        (x - steve_pos.1 - steve_speed * t) * (steve_pos.2 - y) ∧
        (y - laura_pos.2) * (laura_pos.1 + laura_speed * t - x) = 
        (x - laura_pos.1 - laura_speed * t) * (laura_pos.2 - y)) :=
by sorry

end NUMINAMATH_CALUDE_visibility_time_correct_l2288_228802


namespace NUMINAMATH_CALUDE_rotate_minus_six_minus_three_i_l2288_228827

/-- Rotate a complex number by 180 degrees counter-clockwise around the origin -/
def rotate180 (z : ℂ) : ℂ := -z

/-- The theorem stating that rotating -6 - 3i by 180 degrees results in 6 + 3i -/
theorem rotate_minus_six_minus_three_i :
  rotate180 (-6 - 3*I) = (6 + 3*I) := by
  sorry

end NUMINAMATH_CALUDE_rotate_minus_six_minus_three_i_l2288_228827


namespace NUMINAMATH_CALUDE_simplify_fraction_l2288_228874

theorem simplify_fraction :
  (140 : ℚ) / 2100 = 1 / 15 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2288_228874


namespace NUMINAMATH_CALUDE_positive_root_irrational_l2288_228850

-- Define the equation
def f (x : ℝ) : ℝ := x^5 + x

-- Define the property of being a solution to the equation
def is_solution (x : ℝ) : Prop := f x = 10

-- State the theorem
theorem positive_root_irrational :
  ∃ x > 0, is_solution x ∧ ¬ (∃ (p q : ℤ), q ≠ 0 ∧ x = p / q) :=
by sorry

end NUMINAMATH_CALUDE_positive_root_irrational_l2288_228850


namespace NUMINAMATH_CALUDE_original_average_l2288_228891

theorem original_average (n : ℕ) (A : ℚ) (h1 : n = 15) (h2 : (n : ℚ) * (A + 15) = n * 55) : A = 40 := by
  sorry

end NUMINAMATH_CALUDE_original_average_l2288_228891


namespace NUMINAMATH_CALUDE_benzoic_acid_molecular_weight_l2288_228800

/-- The molecular weight of Benzoic acid -/
def molecular_weight_benzoic_acid : ℝ := 122

/-- The number of moles given in the problem -/
def moles_given : ℝ := 4

/-- The total molecular weight for the given number of moles -/
def total_molecular_weight : ℝ := 488

/-- Theorem stating that the molecular weight of Benzoic acid is correct -/
theorem benzoic_acid_molecular_weight :
  molecular_weight_benzoic_acid = total_molecular_weight / moles_given :=
sorry

end NUMINAMATH_CALUDE_benzoic_acid_molecular_weight_l2288_228800


namespace NUMINAMATH_CALUDE_gcd_of_198_and_286_l2288_228825

theorem gcd_of_198_and_286 : Nat.gcd 198 286 = 22 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_198_and_286_l2288_228825


namespace NUMINAMATH_CALUDE_dislike_food_count_problem_solution_l2288_228837

/-- Calculates the number of students who didn't like the food -/
def students_dislike (total participants : ℕ) (students_like : ℕ) : ℕ :=
  total - students_like

/-- Proves that the number of students who didn't like the food is correct -/
theorem dislike_food_count (total : ℕ) (like : ℕ) (h : total ≥ like) :
  students_dislike total like = total - like :=
by sorry

/-- Verifies the solution for the specific problem -/
theorem problem_solution :
  students_dislike 814 383 = 431 :=
by sorry

end NUMINAMATH_CALUDE_dislike_food_count_problem_solution_l2288_228837


namespace NUMINAMATH_CALUDE_sierpinski_carpet_area_sum_l2288_228865

/-- Sierpinski carpet area calculation -/
theorem sierpinski_carpet_area_sum (n : ℕ) : 
  let initial_area : ℝ := Real.sqrt 3 / 4
  let removed_area_sum : ℝ → ℕ → ℝ := λ a k => a * (1 - (3/4)^k)
  removed_area_sum initial_area n = (Real.sqrt 3 / 4) * (1 - (3/4)^n) := by
  sorry

end NUMINAMATH_CALUDE_sierpinski_carpet_area_sum_l2288_228865


namespace NUMINAMATH_CALUDE_positive_real_product_and_sum_squares_l2288_228815

theorem positive_real_product_and_sum_squares (m n : ℝ) 
  (hm : m > 0) (hn : n > 0) (h_sum : m + n = 2 * m * n) : 
  m * n ≥ 1 ∧ m^2 + n^2 ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_positive_real_product_and_sum_squares_l2288_228815


namespace NUMINAMATH_CALUDE_tangent_chord_existence_l2288_228870

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in 2D space -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Checks if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop := sorry

/-- Checks if a line intersects a circle to form a chord of given length -/
def formsChord (l : Line) (c : Circle) (length : ℝ) : Prop := sorry

/-- Main theorem: Given two circles and a length, there exists a tangent to the larger circle
    that forms a chord of the given length in the smaller circle -/
theorem tangent_chord_existence (largeCircle smallCircle : Circle) (chordLength : ℝ) :
  ∃ (tangentLine : Line),
    isTangent tangentLine largeCircle ∧
    formsChord tangentLine smallCircle chordLength :=
  sorry

end NUMINAMATH_CALUDE_tangent_chord_existence_l2288_228870


namespace NUMINAMATH_CALUDE_conference_support_percentage_l2288_228811

theorem conference_support_percentage
  (total_attendees : ℕ)
  (male_attendees : ℕ)
  (female_attendees : ℕ)
  (male_support_rate : ℚ)
  (female_support_rate : ℚ)
  (h1 : total_attendees = 1000)
  (h2 : male_attendees = 150)
  (h3 : female_attendees = 850)
  (h4 : male_support_rate = 70 / 100)
  (h5 : female_support_rate = 75 / 100)
  (h6 : total_attendees = male_attendees + female_attendees) :
  let total_supporters : ℚ :=
    male_support_rate * male_attendees + female_support_rate * female_attendees
  (total_supporters / total_attendees) * 100 = 74.2 := by
  sorry


end NUMINAMATH_CALUDE_conference_support_percentage_l2288_228811


namespace NUMINAMATH_CALUDE_remove_500th_digit_of_3_7_is_greater_l2288_228810

/-- Represents a decimal expansion with a finite number of digits -/
def DecimalExpansion := List Nat

/-- Converts a rational number to its decimal expansion with a given number of digits -/
def rationalToDecimal (n d : Nat) (digits : Nat) : DecimalExpansion :=
  sorry

/-- Removes the nth digit from a decimal expansion -/
def removeNthDigit (n : Nat) (d : DecimalExpansion) : DecimalExpansion :=
  sorry

/-- Converts a decimal expansion back to a rational number -/
def decimalToRational (d : DecimalExpansion) : Rat :=
  sorry

theorem remove_500th_digit_of_3_7_is_greater :
  let original := (3 : Rat) / 7
  let decimalExp := rationalToDecimal 3 7 1000
  let modified := removeNthDigit 500 decimalExp
  decimalToRational modified > original := by
  sorry

end NUMINAMATH_CALUDE_remove_500th_digit_of_3_7_is_greater_l2288_228810


namespace NUMINAMATH_CALUDE_weight_loss_difference_l2288_228859

-- Define weight loss patterns
def barbi_loss_year1 : ℝ := 1.5 * 12
def barbi_loss_year2_3 : ℝ := 2.2 * 12 * 2

def luca_loss_year1 : ℝ := 9
def luca_loss_year2 : ℝ := 12
def luca_loss_year3_7 : ℝ := (12 + 3 * 5)

def kim_loss_year1 : ℝ := 2 * 12
def kim_loss_year2_3 : ℝ := 3 * 12 * 2
def kim_loss_year4_6 : ℝ := 1 * 12 * 3

-- Calculate total weight loss for each person
def barbi_total_loss : ℝ := barbi_loss_year1 + barbi_loss_year2_3
def luca_total_loss : ℝ := luca_loss_year1 + luca_loss_year2 + 5 * luca_loss_year3_7
def kim_total_loss : ℝ := kim_loss_year1 + kim_loss_year2_3 + kim_loss_year4_6

-- Theorem to prove
theorem weight_loss_difference :
  luca_total_loss + kim_total_loss - barbi_total_loss = 217.2 := by
  sorry

end NUMINAMATH_CALUDE_weight_loss_difference_l2288_228859


namespace NUMINAMATH_CALUDE_divisibility_of_power_plus_one_l2288_228826

theorem divisibility_of_power_plus_one (n : ℕ) :
  ∃ k : ℤ, 2^(3^n) + 1 = k * 3^(n + 1) := by sorry

end NUMINAMATH_CALUDE_divisibility_of_power_plus_one_l2288_228826


namespace NUMINAMATH_CALUDE_no_mem_is_veen_l2288_228823

-- Define the universe of discourse
variable {U : Type}

-- Define predicates for Mem, En, and Veen
variable (Mem En Veen : U → Prop)

-- Theorem statement
theorem no_mem_is_veen 
  (h1 : ∀ x, Mem x → En x)  -- All Mems are Ens
  (h2 : ∀ x, En x → ¬Veen x)  -- No Ens are Veens
  : ∀ x, Mem x → ¬Veen x :=  -- No Mem is a Veen
by
  sorry  -- Proof is omitted as per instructions

end NUMINAMATH_CALUDE_no_mem_is_veen_l2288_228823


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2288_228839

theorem purely_imaginary_complex_number (m : ℝ) :
  (2 * m^2 - 3 * m - 2 : ℂ) + (6 * m^2 + 5 * m + 1 : ℂ) * Complex.I = Complex.I * ((6 * m^2 + 5 * m + 1 : ℝ) : ℂ) →
  m = -1 ∨ m = 2 :=
by sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2288_228839


namespace NUMINAMATH_CALUDE_max_volume_after_dilutions_l2288_228854

/-- The maximum volume of a bucket that satisfies the given dilution conditions -/
theorem max_volume_after_dilutions : 
  ∃ (V : ℝ), V > 0 ∧ 
  (V - 10 - 8 * (V - 10) / V) / V ≤ 0.6 ∧
  ∀ (W : ℝ), W > 0 → (W - 10 - 8 * (W - 10) / W) / W ≤ 0.6 → W ≤ V ∧
  V = 40 :=
sorry

end NUMINAMATH_CALUDE_max_volume_after_dilutions_l2288_228854


namespace NUMINAMATH_CALUDE_cos_shift_equivalence_l2288_228878

open Real

theorem cos_shift_equivalence (x : ℝ) :
  cos (2 * x - π / 6) = sin (2 * (x - π / 6) + π / 2) := by sorry

end NUMINAMATH_CALUDE_cos_shift_equivalence_l2288_228878


namespace NUMINAMATH_CALUDE_square_from_relation_l2288_228864

theorem square_from_relation (a b : ℕ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : b^2 = a^2 + a*b + b) : 
  ∃ k : ℕ, k > 0 ∧ b = k^2 := by
  sorry

end NUMINAMATH_CALUDE_square_from_relation_l2288_228864


namespace NUMINAMATH_CALUDE_paul_strawberries_l2288_228804

/-- The number of strawberries Paul has after picking more -/
def total_strawberries (initial : ℕ) (picked : ℕ) : ℕ := initial + picked

/-- Theorem: Paul has 63 strawberries after picking more -/
theorem paul_strawberries : total_strawberries 28 35 = 63 := by
  sorry

end NUMINAMATH_CALUDE_paul_strawberries_l2288_228804


namespace NUMINAMATH_CALUDE_cookie_difference_l2288_228868

/-- Given that Alyssa has 129 cookies and Aiyanna has 140 cookies,
    prove that Aiyanna has 11 more cookies than Alyssa. -/
theorem cookie_difference (alyssa_cookies : ℕ) (aiyanna_cookies : ℕ)
    (h1 : alyssa_cookies = 129)
    (h2 : aiyanna_cookies = 140) :
    aiyanna_cookies - alyssa_cookies = 11 := by
  sorry

end NUMINAMATH_CALUDE_cookie_difference_l2288_228868


namespace NUMINAMATH_CALUDE_olivias_cans_l2288_228801

/-- The number of bags Olivia had -/
def num_bags : ℕ := 4

/-- The number of cans in each bag -/
def cans_per_bag : ℕ := 5

/-- The total number of cans Olivia had -/
def total_cans : ℕ := num_bags * cans_per_bag

theorem olivias_cans : total_cans = 20 := by
  sorry

end NUMINAMATH_CALUDE_olivias_cans_l2288_228801


namespace NUMINAMATH_CALUDE_expense_recording_l2288_228860

/-- Represents the recording of a financial transaction -/
inductive FinancialRecord
  | income (amount : ℤ)
  | expense (amount : ℤ)

/-- Records an income of 5 yuan as +5 -/
def record_income : FinancialRecord := FinancialRecord.income 5

/-- Theorem: If income of 5 yuan is recorded as +5, then expenses of 5 yuan should be recorded as -5 -/
theorem expense_recording (h : record_income = FinancialRecord.income 5) :
  FinancialRecord.expense 5 = FinancialRecord.expense (-5) :=
sorry

end NUMINAMATH_CALUDE_expense_recording_l2288_228860


namespace NUMINAMATH_CALUDE_quadratic_roots_determine_c_l2288_228893

-- Define the quadratic function
def f (c : ℝ) (x : ℝ) : ℝ := -3 * x^2 + c * x - 8

-- State the theorem
theorem quadratic_roots_determine_c :
  (∀ x : ℝ, f c x < 0 ↔ x < 2 ∨ x > 4) →
  (f c 2 = 0 ∧ f c 4 = 0) →
  c = 18 := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_determine_c_l2288_228893


namespace NUMINAMATH_CALUDE_expression_equals_36_l2288_228840

theorem expression_equals_36 (x : ℝ) : (x + 2)^2 + 2*(x + 2)*(4 - x) + (4 - x)^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_36_l2288_228840


namespace NUMINAMATH_CALUDE_product_remainder_l2288_228890

theorem product_remainder (a b c : ℕ) (h : a = 1625 ∧ b = 1627 ∧ c = 1629) : 
  (a * b * c) % 12 = 3 := by
sorry

end NUMINAMATH_CALUDE_product_remainder_l2288_228890


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achieved_l2288_228841

theorem min_value_expression (x₁ x₂ : ℝ) (h1 : x₁ + x₂ = 16) (h2 : x₁ > x₂) :
  (x₁^2 + x₂^2) / (x₁ - x₂) ≥ 16 :=
by sorry

theorem min_value_achieved (x₁ x₂ : ℝ) (h1 : x₁ + x₂ = 16) (h2 : x₁ > x₂) :
  ∃ x₁' x₂' : ℝ, x₁' + x₂' = 16 ∧ x₁' > x₂' ∧ (x₁'^2 + x₂'^2) / (x₁' - x₂') = 16 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achieved_l2288_228841


namespace NUMINAMATH_CALUDE_pythagoras_field_planted_fraction_l2288_228842

theorem pythagoras_field_planted_fraction :
  ∀ (a b c x : ℝ),
  a = 5 ∧ b = 12 ∧ c^2 = a^2 + b^2 →
  (a - x)^2 + (b - x)^2 = 4^2 →
  (a * b / 2 - x^2) / (a * b / 2) = 734 / 750 := by
sorry

end NUMINAMATH_CALUDE_pythagoras_field_planted_fraction_l2288_228842
