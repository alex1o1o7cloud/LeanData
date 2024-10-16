import Mathlib

namespace NUMINAMATH_CALUDE_tim_initial_amount_l635_63566

/-- Tim's initial amount of money in cents -/
def initial_amount : ℕ := sorry

/-- Amount Tim paid for the candy bar in cents -/
def candy_bar_cost : ℕ := 45

/-- Amount Tim received as change in cents -/
def change_received : ℕ := 5

/-- Theorem stating that Tim's initial amount equals 50 cents -/
theorem tim_initial_amount : initial_amount = candy_bar_cost + change_received := by sorry

end NUMINAMATH_CALUDE_tim_initial_amount_l635_63566


namespace NUMINAMATH_CALUDE_triangle_area_l635_63593

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfiesConditions (t : Triangle) : Prop :=
  t.a * Real.sin t.A = t.b * Real.sin t.B + (t.c - t.b) * Real.sin t.C ∧
  t.b * t.c = 4

-- Theorem statement
theorem triangle_area (t : Triangle) (h : satisfiesConditions t) : 
  (1/2) * t.b * t.c * Real.sin t.A = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l635_63593


namespace NUMINAMATH_CALUDE_payment_ways_formula_l635_63547

/-- The number of ways to pay n euros using 1-euro and 2-euro coins -/
def paymentWays (n : ℕ) : ℕ := n / 2 + 1

/-- Theorem: The number of ways to pay n euros using 1-euro and 2-euro coins
    is equal to ⌊n/2⌋ + 1 -/
theorem payment_ways_formula (n : ℕ) :
  paymentWays n = n / 2 + 1 := by
  sorry

#check payment_ways_formula

end NUMINAMATH_CALUDE_payment_ways_formula_l635_63547


namespace NUMINAMATH_CALUDE_right_triangle_sin_c_l635_63534

theorem right_triangle_sin_c (A B C : Real) (h1 : A + B + C = Real.pi)
  (h2 : B = Real.pi / 2) (h3 : Real.sin A = 7 / 25) :
  Real.sin C = 24 / 25 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sin_c_l635_63534


namespace NUMINAMATH_CALUDE_meaningful_fraction_l635_63521

theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = 5 / (x - 2)) ↔ x ≠ 2 := by
sorry

end NUMINAMATH_CALUDE_meaningful_fraction_l635_63521


namespace NUMINAMATH_CALUDE_functional_equation_implies_g_50_eq_0_l635_63584

/-- A function satisfying the given functional equation for all positive real numbers -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), x > 0 → y > 0 → x * g y - y * g x = g (x / y) + g (x + y)

/-- The main theorem stating that any function satisfying the functional equation must have g(50) = 0 -/
theorem functional_equation_implies_g_50_eq_0 (g : ℝ → ℝ) (h : FunctionalEquation g) : g 50 = 0 := by
  sorry

#check functional_equation_implies_g_50_eq_0

end NUMINAMATH_CALUDE_functional_equation_implies_g_50_eq_0_l635_63584


namespace NUMINAMATH_CALUDE_exactly_three_sets_sum_to_30_l635_63535

/-- A set of consecutive positive integers -/
structure ConsecutiveSet where
  start : ℕ
  length : ℕ
  length_ge_two : length ≥ 2

/-- The sum of a ConsecutiveSet -/
def sum_consecutive_set (s : ConsecutiveSet) : ℕ :=
  (s.length * (2 * s.start + s.length - 1)) / 2

/-- Predicate for a ConsecutiveSet summing to 30 -/
def sums_to_30 (s : ConsecutiveSet) : Prop :=
  sum_consecutive_set s = 30

theorem exactly_three_sets_sum_to_30 :
  ∃! (sets : Finset ConsecutiveSet), 
    Finset.card sets = 3 ∧ 
    (∀ s ∈ sets, sums_to_30 s) ∧
    (∀ s : ConsecutiveSet, sums_to_30 s → s ∈ sets) :=
sorry

end NUMINAMATH_CALUDE_exactly_three_sets_sum_to_30_l635_63535


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l635_63598

theorem sqrt_product_equality : 3 * Real.sqrt 2 * (2 * Real.sqrt 3) = 6 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l635_63598


namespace NUMINAMATH_CALUDE_like_terms_exponent_sum_l635_63575

theorem like_terms_exponent_sum (m n : ℤ) : 
  (∃ (x y : ℝ), -5 * x^m * y^(m+1) = x^(n-1) * y^3) → m + n = 5 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_exponent_sum_l635_63575


namespace NUMINAMATH_CALUDE_inequality_implication_l635_63549

theorem inequality_implication (a b : ℝ) : a < b → -a + 3 > -b + 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l635_63549


namespace NUMINAMATH_CALUDE_discounted_price_l635_63551

theorem discounted_price (original_price : ℝ) : 
  original_price * (1 - 0.20) * (1 - 0.10) * (1 - 0.05) = 6840 → 
  original_price = 10000 := by
sorry

end NUMINAMATH_CALUDE_discounted_price_l635_63551


namespace NUMINAMATH_CALUDE_complex_number_range_l635_63538

/-- The range of y/x for a complex number (x-2) + yi with modulus 1 -/
theorem complex_number_range (x y : ℝ) : 
  (x - 2)^2 + y^2 = 1 → 
  y ≠ 0 → 
  ∃ k : ℝ, y = k * x ∧ 
    (-Real.sqrt 3 / 3 ≤ k ∧ k < 0) ∨ (0 < k ∧ k ≤ Real.sqrt 3 / 3) :=
sorry

end NUMINAMATH_CALUDE_complex_number_range_l635_63538


namespace NUMINAMATH_CALUDE_tetrahedrons_from_cube_l635_63511

/-- A cube has 8 vertices -/
def cube_vertices : ℕ := 8

/-- The number of tetrahedrons that can be formed using the vertices of a cube -/
def num_tetrahedrons : ℕ := 58

/-- Theorem: The number of tetrahedrons that can be formed using the vertices of a cube is 58 -/
theorem tetrahedrons_from_cube : num_tetrahedrons = 58 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedrons_from_cube_l635_63511


namespace NUMINAMATH_CALUDE_lowest_power_x4_l635_63568

theorem lowest_power_x4 (x : ℝ) : 
  let A : ℝ := 1/3
  let B : ℝ := -1/9
  let C : ℝ := 5/81
  let f : ℝ → ℝ := λ x => (1 + A*x + B*x^2 + C*x^3)^3 - (1 + x)
  ∃ (D E F G H I : ℝ), f x = D*x^4 + E*x^5 + F*x^6 + G*x^7 + H*x^8 + I*x^9 ∧ D ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_lowest_power_x4_l635_63568


namespace NUMINAMATH_CALUDE_intersection_count_l635_63574

/-- A point in the Cartesian plane -/
structure Point where
  x : ℤ
  y : ℤ

/-- Checks if a point is the intersection of two lines -/
def is_intersection (p : Point) (k : ℤ) : Prop :=
  p.y = p.x - 3 ∧ p.y = k * p.x - k

/-- The theorem statement -/
theorem intersection_count : 
  ∃ (s : Finset ℤ), s.card = 3 ∧ 
  (∀ k : ℤ, k ∈ s ↔ ∃ p : Point, is_intersection p k) :=
sorry

end NUMINAMATH_CALUDE_intersection_count_l635_63574


namespace NUMINAMATH_CALUDE_austin_work_hours_on_monday_l635_63572

/-- Proves that Austin works 2 hours on Mondays to earn enough for a $180 bicycle in 6 weeks -/
theorem austin_work_hours_on_monday : 
  let hourly_rate : ℕ := 5
  let bicycle_cost : ℕ := 180
  let weeks : ℕ := 6
  let wednesday_hours : ℕ := 1
  let friday_hours : ℕ := 3
  ∃ (monday_hours : ℕ), 
    weeks * (hourly_rate * (monday_hours + wednesday_hours + friday_hours)) = bicycle_cost ∧ 
    monday_hours = 2 := by
  sorry

end NUMINAMATH_CALUDE_austin_work_hours_on_monday_l635_63572


namespace NUMINAMATH_CALUDE_range_of_m_l635_63510

-- Define the function f
def f (x : ℝ) := x^3 + x

-- State the theorem
theorem range_of_m (m : ℝ) :
  (∀ θ : ℝ, 0 < θ ∧ θ < Real.pi / 2 → f (m * Real.sin θ) + f (1 - m) > 0) →
  m ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l635_63510


namespace NUMINAMATH_CALUDE_five_digit_number_product_l635_63509

theorem five_digit_number_product (a b c d e : Nat) : 
  a ≠ 0 ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
  c ≠ d ∧ c ≠ e ∧ 
  d ≠ e ∧
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧
  (10 * a + b + 10 * b + c) * 
  (10 * b + c + 10 * c + d) * 
  (10 * c + d + 10 * d + e) = 157605 →
  (a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 4 ∧ e = 5) ∨
  (a = 2 ∧ b = 1 ∧ c = 4 ∧ d = 3 ∧ e = 6) := by
  sorry

end NUMINAMATH_CALUDE_five_digit_number_product_l635_63509


namespace NUMINAMATH_CALUDE_daves_phone_files_l635_63561

theorem daves_phone_files :
  ∀ (initial_apps initial_files current_apps : ℕ),
    initial_apps = 24 →
    initial_files = 9 →
    current_apps = 12 →
    current_apps = (current_apps - 7) + 7 →
    current_apps - 7 = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_daves_phone_files_l635_63561


namespace NUMINAMATH_CALUDE_garden_area_ratio_l635_63590

/-- Given a rectangular garden with initial length and width, and increase percentages for both dimensions, prove that the ratio of the original area to the redesigned area is 1/3. -/
theorem garden_area_ratio (initial_length initial_width : ℝ) 
  (length_increase width_increase : ℝ) : 
  initial_length = 10 →
  initial_width = 5 →
  length_increase = 0.5 →
  width_increase = 1 →
  (initial_length * initial_width) / 
  ((initial_length * (1 + length_increase)) * (initial_width * (1 + width_increase))) = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_garden_area_ratio_l635_63590


namespace NUMINAMATH_CALUDE_xy_eq_x_plus_y_plus_3_l635_63553

theorem xy_eq_x_plus_y_plus_3 (x y : ℕ) : 
  x * y = x + y + 3 ↔ (x = 2 ∧ y = 5) ∨ (x = 5 ∧ y = 2) ∨ (x = 3 ∧ y = 3) := by
  sorry

end NUMINAMATH_CALUDE_xy_eq_x_plus_y_plus_3_l635_63553


namespace NUMINAMATH_CALUDE_ant_travel_distance_l635_63579

theorem ant_travel_distance (planet_radius : ℝ) (observer_height : ℝ) : 
  planet_radius = 156 → observer_height = 13 → 
  let horizon_distance := Real.sqrt ((planet_radius + observer_height)^2 - planet_radius^2)
  (2 * Real.pi * horizon_distance) = 130 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ant_travel_distance_l635_63579


namespace NUMINAMATH_CALUDE_problem_solution_l635_63523

theorem problem_solution (x : ℝ) (h : x + 1/x = 5) :
  (x - 3)^2 + 36/((x - 3)^2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l635_63523


namespace NUMINAMATH_CALUDE_calculate_expression_l635_63563

theorem calculate_expression : (235 - 2 * 3 * 5) * 7 / 5 = 287 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l635_63563


namespace NUMINAMATH_CALUDE_special_hexagon_side_length_l635_63545

/-- An equilateral hexagon with specific properties -/
structure SpecialHexagon where
  -- Side length of the hexagon
  side_length : ℝ
  -- Three nonadjacent acute interior angles measure 45°
  has_45_degree_angles : Prop
  -- The enclosed area of the hexagon
  area : ℝ
  -- The hexagon is equilateral
  is_equilateral : Prop
  -- The area is 9√2
  area_is_9_sqrt_2 : area = 9 * Real.sqrt 2

/-- Theorem stating that a hexagon with the given properties has a side length of 2√3 -/
theorem special_hexagon_side_length 
  (h : SpecialHexagon) : h.side_length = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_special_hexagon_side_length_l635_63545


namespace NUMINAMATH_CALUDE_find_n_l635_63597

theorem find_n : ∃ n : ℕ, (1/5 : ℝ)^n * (1/4 : ℝ)^18 = 1/(2*(10^35)) ∧ n = 35 := by
  sorry

end NUMINAMATH_CALUDE_find_n_l635_63597


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l635_63506

theorem max_sum_of_squares (a b c d : ℝ) : 
  a + b = 12 →
  a * b + c + d = 52 →
  a * d + b * c = 83 →
  c * d = 42 →
  a^2 + b^2 + c^2 + d^2 ≤ 38 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l635_63506


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l635_63555

theorem nested_fraction_equality : 
  1 + (1 / (1 + (1 / (2 + (1 / 3))))) = 17 / 10 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l635_63555


namespace NUMINAMATH_CALUDE_notebook_savings_correct_l635_63583

def notebook_savings (quantity : ℕ) (original_price : ℝ) (individual_discount_rate : ℝ) (bulk_discount_rate : ℝ) (bulk_discount_threshold : ℕ) : ℝ :=
  let discounted_price := original_price * (1 - individual_discount_rate)
  let total_without_discount := quantity * original_price
  let total_with_individual_discount := quantity * discounted_price
  let final_total := if quantity > bulk_discount_threshold
                     then total_with_individual_discount * (1 - bulk_discount_rate)
                     else total_with_individual_discount
  total_without_discount - final_total

theorem notebook_savings_correct :
  notebook_savings 8 3 0.1 0.05 6 = 3.48 :=
sorry

end NUMINAMATH_CALUDE_notebook_savings_correct_l635_63583


namespace NUMINAMATH_CALUDE_sanctuary_animal_pairs_l635_63505

theorem sanctuary_animal_pairs : 
  let bird_species : ℕ := 29
  let bird_pairs_per_species : ℕ := 7
  let marine_species : ℕ := 15
  let marine_pairs_per_species : ℕ := 9
  let mammal_species : ℕ := 22
  let mammal_pairs_per_species : ℕ := 6
  
  bird_species * bird_pairs_per_species + 
  marine_species * marine_pairs_per_species + 
  mammal_species * mammal_pairs_per_species = 470 :=
by
  sorry

end NUMINAMATH_CALUDE_sanctuary_animal_pairs_l635_63505


namespace NUMINAMATH_CALUDE_find_divisor_l635_63503

theorem find_divisor (dividend quotient remainder : ℕ) (h : dividend = quotient * 23 + remainder) :
  dividend = 997 → quotient = 43 → remainder = 8 → 23 = dividend / quotient :=
by sorry

end NUMINAMATH_CALUDE_find_divisor_l635_63503


namespace NUMINAMATH_CALUDE_right_isosceles_triangle_special_property_l635_63564

/-- A right isosceles triangle with the given property has 45° acute angles -/
theorem right_isosceles_triangle_special_property (a h : ℝ) (θ : ℝ) : 
  a > 0 → -- The leg length is positive
  h > 0 → -- The hypotenuse length is positive
  h = a * Real.sqrt 2 → -- Right isosceles triangle property
  h^2 = 3 * a * Real.sin θ → -- Given special property
  θ = π/4 := by -- Conclusion: acute angle is 45° (π/4 radians)
sorry

end NUMINAMATH_CALUDE_right_isosceles_triangle_special_property_l635_63564


namespace NUMINAMATH_CALUDE_solve_quadratic_equation_solve_cubic_equation_l635_63522

-- Problem 1
theorem solve_quadratic_equation (x : ℝ) :
  4 * x^2 - 81 = 0 ↔ x = 9/2 ∨ x = -9/2 := by
sorry

-- Problem 2
theorem solve_cubic_equation (x : ℝ) :
  8 * (x + 1)^3 = 27 ↔ x = 1/2 := by
sorry

end NUMINAMATH_CALUDE_solve_quadratic_equation_solve_cubic_equation_l635_63522


namespace NUMINAMATH_CALUDE_semicircle_perimeter_l635_63519

/-- The perimeter of a semi-circle with radius r is πr + 2r -/
theorem semicircle_perimeter (r : ℝ) (hr : r > 0) : 
  let P := π * r + 2 * r
  P = π * r + 2 * r := by
  sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_l635_63519


namespace NUMINAMATH_CALUDE_sqrt_15_minus_1_range_l635_63516

theorem sqrt_15_minus_1_range : 2 < Real.sqrt 15 - 1 ∧ Real.sqrt 15 - 1 < 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_15_minus_1_range_l635_63516


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l635_63550

/-- An arithmetic sequence with common difference d and first term a₁ -/
def arithmetic_sequence (d a₁ : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

/-- Theorem: For an arithmetic sequence {aₙ} where the common difference d ≠ 0 and
    the first term a₁ ≠ 0, if a₂, a₄, a₈ form a geometric sequence,
    then (a₁ + a₅ + a₉) / (a₂ + a₃) = 3 -/
theorem arithmetic_geometric_ratio
  (d a₁ : ℝ)
  (hd : d ≠ 0)
  (ha₁ : a₁ ≠ 0)
  (h_geom : (arithmetic_sequence d a₁ 4) ^ 2 = 
            (arithmetic_sequence d a₁ 2) * (arithmetic_sequence d a₁ 8)) :
  (arithmetic_sequence d a₁ 1 + arithmetic_sequence d a₁ 5 + arithmetic_sequence d a₁ 9) /
  (arithmetic_sequence d a₁ 2 + arithmetic_sequence d a₁ 3) = 3 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l635_63550


namespace NUMINAMATH_CALUDE_factorization_proof_l635_63528

theorem factorization_proof (b : ℝ) : 56 * b^3 + 168 * b^2 = 56 * b^2 * (b + 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l635_63528


namespace NUMINAMATH_CALUDE_remainder_theorem_l635_63585

theorem remainder_theorem (n : ℤ) (k : ℤ) (h : n = 40 * k - 1) :
  (n^2 - 3*n + 5) % 40 = 9 := by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l635_63585


namespace NUMINAMATH_CALUDE_investment_problem_l635_63576

/-- Given two investment projects with specific conditions, 
    prove the minimum distance between them and the profitability of the deal. -/
theorem investment_problem 
  (p₁ x₁ p₂ x₂ : ℝ) 
  (h₁ : 4 * x₁ - 3 * p₁ - 44 = 0) 
  (h₂ : p₂^2 - 12 * p₂ + x₂^2 - 8 * x₂ + 43 = 0) 
  (h₃ : p₁ > 0) 
  (h₄ : p₂ > 0) : 
  let d := Real.sqrt ((x₁ - x₂)^2 + (p₁ - p₂)^2)
  ∃ (min_d : ℝ), 
    (∀ p₁' x₁' p₂' x₂', 
      4 * x₁' - 3 * p₁' - 44 = 0 → 
      p₂'^2 - 12 * p₂' + x₂'^2 - 8 * x₂' + 43 = 0 → 
      p₁' > 0 → 
      p₂' > 0 → 
      Real.sqrt ((x₁' - x₂')^2 + (p₁' - p₂')^2) ≥ min_d) ∧ 
    d = min_d ∧ 
    min_d = 6.2 ∧ 
    x₁ + x₂ - p₁ - p₂ > 0 := by
  sorry

end NUMINAMATH_CALUDE_investment_problem_l635_63576


namespace NUMINAMATH_CALUDE_blue_paint_cans_l635_63591

theorem blue_paint_cans (total_cans : ℕ) (blue_ratio green_ratio : ℕ) 
  (h1 : total_cans = 42)
  (h2 : blue_ratio = 4)
  (h3 : green_ratio = 3) :
  (blue_ratio * total_cans) / (blue_ratio + green_ratio) = 24 := by
  sorry

end NUMINAMATH_CALUDE_blue_paint_cans_l635_63591


namespace NUMINAMATH_CALUDE_order_of_powers_l635_63546

theorem order_of_powers : (3/5)^(1/5 : ℝ) > (1/5 : ℝ)^(1/5 : ℝ) ∧ (1/5 : ℝ)^(1/5 : ℝ) > (1/5 : ℝ)^(3/5 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_order_of_powers_l635_63546


namespace NUMINAMATH_CALUDE_tetrahedron_edge_relation_l635_63537

/-- Given a tetrahedron ABCD with edge lengths and angles, prove that among t₁, t₂, t₃,
    there is at least one number equal to the sum of the other two. -/
theorem tetrahedron_edge_relation (a₁ a₂ a₃ b₁ b₂ b₃ θ₁ θ₂ θ₃ : ℝ) 
  (h_pos : a₁ > 0 ∧ a₂ > 0 ∧ a₃ > 0 ∧ b₁ > 0 ∧ b₂ > 0 ∧ b₃ > 0)
  (h_angle : 0 < θ₁ ∧ θ₁ < π ∧ 0 < θ₂ ∧ θ₂ < π ∧ 0 < θ₃ ∧ θ₃ < π) :
  ∃ (i j k : Fin 3), i ≠ j ∧ i ≠ k ∧ j ≠ k ∧
    let t : Fin 3 → ℝ := λ n => match n with
      | 0 => a₁ * b₁ * Real.cos θ₁
      | 1 => a₂ * b₂ * Real.cos θ₂
      | 2 => a₃ * b₃ * Real.cos θ₃
    t i = t j + t k :=
by sorry

end NUMINAMATH_CALUDE_tetrahedron_edge_relation_l635_63537


namespace NUMINAMATH_CALUDE_reading_time_reduction_xiao_yu_reading_time_l635_63565

/-- Represents the number of days to read a book given the pages per day -/
def days_to_read (total_pages : ℕ) (pages_per_day : ℕ) : ℕ :=
  total_pages / pages_per_day

/-- The theorem stating the relationship between reading rates and days to finish the book -/
theorem reading_time_reduction (initial_pages_per_day : ℕ) (initial_days : ℕ) (additional_pages : ℕ) :
  initial_pages_per_day > 0 →
  initial_days > 0 →
  additional_pages > 0 →
  days_to_read (initial_pages_per_day * initial_days) (initial_pages_per_day + additional_pages) =
    initial_days * initial_pages_per_day / (initial_pages_per_day + additional_pages) :=
by
  sorry

/-- The specific instance of the theorem for the given problem -/
theorem xiao_yu_reading_time :
  days_to_read (15 * 24) (15 + 3) = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_reading_time_reduction_xiao_yu_reading_time_l635_63565


namespace NUMINAMATH_CALUDE_tan_fifteen_ratio_equals_sqrt_three_l635_63589

theorem tan_fifteen_ratio_equals_sqrt_three : 
  (1 + Real.tan (15 * π / 180)) / (1 - Real.tan (15 * π / 180)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_fifteen_ratio_equals_sqrt_three_l635_63589


namespace NUMINAMATH_CALUDE_apple_calculation_correct_l635_63541

/-- Represents the weight difference from the standard weight and its frequency --/
structure WeightDifference :=
  (difference : ℝ)
  (frequency : ℕ)

/-- Calculates the total weight and profit from a batch of apples --/
def apple_calculation (total_boxes : ℕ) (price_per_box : ℝ) (selling_price_per_kg : ℝ) 
  (weight_differences : List WeightDifference) : ℝ × ℝ :=
  sorry

/-- The main theorem stating the correctness of the calculation --/
theorem apple_calculation_correct : 
  let weight_differences := [
    ⟨-0.2, 5⟩, ⟨-0.1, 8⟩, ⟨0, 2⟩, ⟨0.1, 6⟩, ⟨0.2, 8⟩, ⟨0.5, 1⟩
  ]
  let (total_weight, profit) := apple_calculation 400 60 10 weight_differences
  total_weight = 300.9 ∧ profit = 16120 :=
by sorry

end NUMINAMATH_CALUDE_apple_calculation_correct_l635_63541


namespace NUMINAMATH_CALUDE_sum_of_integers_from_1_to_3_l635_63532

theorem sum_of_integers_from_1_to_3 : 
  (Finset.range 3).sum (fun i => i + 1) = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_from_1_to_3_l635_63532


namespace NUMINAMATH_CALUDE_child_wage_is_eight_l635_63507

/-- Represents the daily wage structure and worker composition of a building contractor. -/
structure ContractorData where
  male_workers : ℕ
  female_workers : ℕ
  child_workers : ℕ
  male_wage : ℕ
  female_wage : ℕ
  average_wage : ℕ

/-- Calculates the daily wage of a child worker given the contractor's data. -/
def child_worker_wage (data : ContractorData) : ℕ :=
  ((data.average_wage * (data.male_workers + data.female_workers + data.child_workers)) -
   (data.male_wage * data.male_workers + data.female_wage * data.female_workers)) / data.child_workers

/-- Theorem stating that given the specific conditions, the child worker's daily wage is 8 rupees. -/
theorem child_wage_is_eight (data : ContractorData)
  (h1 : data.male_workers = 20)
  (h2 : data.female_workers = 15)
  (h3 : data.child_workers = 5)
  (h4 : data.male_wage = 25)
  (h5 : data.female_wage = 20)
  (h6 : data.average_wage = 21) :
  child_worker_wage data = 8 := by
  sorry


end NUMINAMATH_CALUDE_child_wage_is_eight_l635_63507


namespace NUMINAMATH_CALUDE_angies_age_ratio_l635_63562

theorem angies_age_ratio : 
  ∀ (A : ℕ), A + 4 = 20 → (A : ℚ) / 20 = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_angies_age_ratio_l635_63562


namespace NUMINAMATH_CALUDE_tournament_size_l635_63508

/-- Represents a tournament with the given conditions -/
structure Tournament where
  n : ℕ  -- Number of players not in the weakest 15
  total_players : ℕ := n + 15
  total_games : ℕ := (total_players * (total_players - 1)) / 2
  weak_player_games : ℕ := 15 * 14 / 2
  strong_player_games : ℕ := n * (n - 1) / 2
  cross_games : ℕ := 15 * n

/-- The theorem stating that the tournament must have 36 players -/
theorem tournament_size (t : Tournament) : t.total_players = 36 := by
  sorry

end NUMINAMATH_CALUDE_tournament_size_l635_63508


namespace NUMINAMATH_CALUDE_triangle_sine_product_inequality_l635_63512

theorem triangle_sine_product_inequality (A B C : ℝ) (h_triangle : A + B + C = π) :
  Real.sin (A / 2) * Real.sin (B / 2) * Real.sin (C / 2) < 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_product_inequality_l635_63512


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_same_foci_l635_63592

theorem ellipse_hyperbola_same_foci (a : ℝ) :
  (∀ x y : ℝ, x^2/4 + y^2/a^2 = 1 ↔ (x, y) ∈ {p : ℝ × ℝ | p.1^2/4 + p.2^2/a^2 = 1}) ∧
  (∀ x y : ℝ, x^2/a^2 - y^2/2 = 1 ↔ (x, y) ∈ {p : ℝ × ℝ | p.1^2/a^2 - p.2^2/2 = 1}) ∧
  (∃ c : ℝ, c^2 = 4 - a^2 ∧ c^2 = a^2 + 2) →
  a = 1 ∨ a = -1 := by
sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_same_foci_l635_63592


namespace NUMINAMATH_CALUDE_two_green_marbles_probability_l635_63569

/-- The probability of drawing two green marbles without replacement from a jar -/
theorem two_green_marbles_probability
  (red : ℕ) (green : ℕ) (white : ℕ)
  (h_red : red = 4)
  (h_green : green = 5)
  (h_white : white = 12)
  : (green / (red + green + white)) * ((green - 1) / (red + green + white - 1)) = 1 / 21 :=
by sorry

end NUMINAMATH_CALUDE_two_green_marbles_probability_l635_63569


namespace NUMINAMATH_CALUDE_x_equals_y_l635_63588

theorem x_equals_y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (eq1 : x = 2 + 1 / y) (eq2 : y = 2 + 1 / x) : y = x := by
  sorry

end NUMINAMATH_CALUDE_x_equals_y_l635_63588


namespace NUMINAMATH_CALUDE_total_football_games_l635_63554

def football_games_this_year : ℕ := 4
def football_games_last_year : ℕ := 9

theorem total_football_games : 
  football_games_this_year + football_games_last_year = 13 := by
  sorry

end NUMINAMATH_CALUDE_total_football_games_l635_63554


namespace NUMINAMATH_CALUDE_rohans_age_puzzle_l635_63596

/-- 
Given that Rohan is currently 25 years old, this theorem proves that the number of years 
into the future when Rohan will be 4 times as old as he was the same number of years ago is 15.
-/
theorem rohans_age_puzzle : 
  ∃ x : ℕ, (25 + x = 4 * (25 - x)) ∧ x = 15 :=
by sorry

end NUMINAMATH_CALUDE_rohans_age_puzzle_l635_63596


namespace NUMINAMATH_CALUDE_zero_only_number_unchanged_by_integer_multiplication_l635_63556

theorem zero_only_number_unchanged_by_integer_multiplication :
  ∀ n : ℤ, (∀ m : ℤ, n * m = n) → n = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_only_number_unchanged_by_integer_multiplication_l635_63556


namespace NUMINAMATH_CALUDE_distance_traveled_l635_63573

/-- Proves that given a speed of 20 km/hr and a time of 2.5 hours, the distance traveled is 50 km. -/
theorem distance_traveled (speed : ℝ) (time : ℝ) (h1 : speed = 20) (h2 : time = 2.5) :
  speed * time = 50 := by
  sorry

end NUMINAMATH_CALUDE_distance_traveled_l635_63573


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l635_63558

/-- Two points are symmetric with respect to the x-axis if their x-coordinates are equal
    and their y-coordinates are negatives of each other -/
def symmetric_x_axis (A B : ℝ × ℝ) : Prop :=
  A.1 = B.1 ∧ A.2 = -B.2

/-- Given that point A(m, 1) is symmetric to point B(2, n) with respect to the x-axis,
    prove that m + n = 1 -/
theorem symmetric_points_sum (m n : ℝ) :
  symmetric_x_axis (m, 1) (2, n) → m + n = 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l635_63558


namespace NUMINAMATH_CALUDE_portrait_price_ratio_l635_63531

def price_8inch : ℝ := 5
def daily_8inch_sales : ℕ := 3
def daily_16inch_sales : ℕ := 5
def earnings_3days : ℝ := 195

def price_ratio : ℝ := 2

theorem portrait_price_ratio :
  let daily_earnings := daily_8inch_sales * price_8inch + daily_16inch_sales * (price_ratio * price_8inch)
  earnings_3days = 3 * daily_earnings :=
by sorry

end NUMINAMATH_CALUDE_portrait_price_ratio_l635_63531


namespace NUMINAMATH_CALUDE_smallest_N_for_P_less_than_four_fifths_l635_63567

/-- The probability function P(N) as described in the problem -/
def P (N : ℕ) : ℚ :=
  (2 * N * N) / (9 * (N + 2) * (N + 3))

/-- Predicate to check if a number is a multiple of 6 -/
def isMultipleOf6 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 6 * k

theorem smallest_N_for_P_less_than_four_fifths :
  ∀ N : ℕ, isMultipleOf6 N → N < 600 → P N ≥ 4/5 ∧
  isMultipleOf6 600 ∧ P 600 < 4/5 := by
  sorry

#eval P 600 -- To verify that P(600) is indeed less than 4/5

end NUMINAMATH_CALUDE_smallest_N_for_P_less_than_four_fifths_l635_63567


namespace NUMINAMATH_CALUDE_profit_decrease_l635_63514

theorem profit_decrease (march_profit : ℝ) (april_may_decrease : ℝ) : 
  (1 + 0.35) * (1 - april_may_decrease / 100) * (1 + 0.5) = 1.62000000000000014 →
  april_may_decrease = 20 := by
sorry

end NUMINAMATH_CALUDE_profit_decrease_l635_63514


namespace NUMINAMATH_CALUDE_student_weight_l635_63517

theorem student_weight (student_weight sister_weight : ℝ) :
  student_weight - 5 = 2 * sister_weight →
  student_weight + sister_weight = 104 →
  student_weight = 71 := by
sorry

end NUMINAMATH_CALUDE_student_weight_l635_63517


namespace NUMINAMATH_CALUDE_simplify_fraction_l635_63580

theorem simplify_fraction : (123 : ℚ) / 999 * 27 = 123 / 37 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l635_63580


namespace NUMINAMATH_CALUDE_order_of_magnitude_l635_63500

theorem order_of_magnitude : 70.3 > 70.2 ∧ 70.2 > Real.log 0.3 := by
  have h : 0 < 0.3 ∧ 0.3 < 1 := by sorry
  sorry

end NUMINAMATH_CALUDE_order_of_magnitude_l635_63500


namespace NUMINAMATH_CALUDE_terminal_side_in_third_quadrant_l635_63504

/-- Given a point P with coordinates (cosθ, tanθ) in the second quadrant,
    prove that the terminal side of angle θ is in the third quadrant. -/
theorem terminal_side_in_third_quadrant (θ : Real) :
  (cosθ < 0 ∧ tanθ > 0) →  -- Point P is in the second quadrant
  (cosθ < 0 ∧ sinθ < 0)    -- Terminal side of θ is in the third quadrant
:= by sorry

end NUMINAMATH_CALUDE_terminal_side_in_third_quadrant_l635_63504


namespace NUMINAMATH_CALUDE_rose_group_size_l635_63578

theorem rose_group_size (n : ℕ+) (h : Nat.lcm n 19 = 171) : n = 9 := by
  sorry

end NUMINAMATH_CALUDE_rose_group_size_l635_63578


namespace NUMINAMATH_CALUDE_quadratic_roots_properties_l635_63536

theorem quadratic_roots_properties (x₁ x₂ : ℝ) 
  (h₁ : x₁^2 - 5*x₁ - 3 = 0) 
  (h₂ : x₂^2 - 5*x₂ - 3 = 0) :
  (x₁^2 + x₂^2 = 31) ∧ (1/x₁ - 1/x₂ = Real.sqrt 37 / 3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_properties_l635_63536


namespace NUMINAMATH_CALUDE_m_eq_neg_one_iff_pure_imaginary_l635_63582

/-- A complex number z is defined as m² - 1 + (m² - 3m + 2)i, where m is a real number and i is the imaginary unit. -/
def z (m : ℝ) : ℂ := (m^2 - 1) + (m^2 - 3*m + 2)*Complex.I

/-- A complex number is pure imaginary if its real part is zero and its imaginary part is non-zero. -/
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- Theorem: m = -1 is both sufficient and necessary for z to be a pure imaginary number. -/
theorem m_eq_neg_one_iff_pure_imaginary (m : ℝ) :
  m = -1 ↔ is_pure_imaginary (z m) := by sorry

end NUMINAMATH_CALUDE_m_eq_neg_one_iff_pure_imaginary_l635_63582


namespace NUMINAMATH_CALUDE_quadratic_equation_condition_l635_63587

theorem quadratic_equation_condition (m : ℝ) : 
  (∀ x, ∃ a b c : ℝ, a ≠ 0 ∧ (m - 2) * x^2 + (2*m + 1) * x - m = a * x^2 + b * x + c) →
  m = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_condition_l635_63587


namespace NUMINAMATH_CALUDE_playground_insects_l635_63530

/-- Calculates the number of remaining insects in the playground --/
def remaining_insects (initial_bees initial_beetles initial_ants initial_termites
                       initial_praying_mantises initial_ladybugs initial_butterflies
                       initial_dragonflies : ℕ)
                      (bees_left beetles_taken ants_left termites_moved
                       ladybugs_left butterflies_left dragonflies_left : ℕ) : ℕ :=
  (initial_bees - bees_left) +
  (initial_beetles - beetles_taken) +
  (initial_ants - ants_left) +
  (initial_termites - termites_moved) +
  initial_praying_mantises +
  (initial_ladybugs - ladybugs_left) +
  (initial_butterflies - butterflies_left) +
  (initial_dragonflies - dragonflies_left)

/-- Theorem stating that the number of remaining insects is 54 --/
theorem playground_insects :
  remaining_insects 15 7 12 10 2 10 11 8 6 2 4 3 2 3 1 = 54 := by
  sorry

end NUMINAMATH_CALUDE_playground_insects_l635_63530


namespace NUMINAMATH_CALUDE_fraction_inequality_implies_numerator_inequality_l635_63543

theorem fraction_inequality_implies_numerator_inequality
  (a b c : ℝ) (hc : c ≠ 0) :
  a / c^2 > b / c^2 → a > b := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_implies_numerator_inequality_l635_63543


namespace NUMINAMATH_CALUDE_quadratic_minimum_ratio_bound_l635_63560

-- Define a quadratic function
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the derivative of the quadratic function
def quadratic_derivative (a b : ℝ) (x : ℝ) : ℝ := 2 * a * x + b

-- Define the second derivative of the quadratic function
def quadratic_second_derivative (a : ℝ) : ℝ := 2 * a

theorem quadratic_minimum_ratio_bound (a b c : ℝ) :
  a > 0 →  -- Ensures the function is concave up
  quadratic_derivative a b 0 > 0 →  -- f'(0) > 0
  (∀ x : ℝ, quadratic a b c x ≥ 0) →  -- f(x) ≥ 0 for all real x
  (quadratic a b c 1) / (quadratic_second_derivative a) ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_minimum_ratio_bound_l635_63560


namespace NUMINAMATH_CALUDE_rotated_angle_measure_l635_63544

/-- Given an angle of 60 degrees rotated 540 degrees clockwise, 
    the resulting new acute angle is also 60 degrees. -/
theorem rotated_angle_measure (initial_angle rotation : ℝ) : 
  initial_angle = 60 → 
  rotation = 540 → 
  (rotation % 360 - initial_angle) % 180 = 60 := by
sorry

end NUMINAMATH_CALUDE_rotated_angle_measure_l635_63544


namespace NUMINAMATH_CALUDE_M_inter_N_eq_M_l635_63577

def M : Set ℝ := {x | x^2 - x < 0}
def N : Set ℝ := {x | |x| < 2}

theorem M_inter_N_eq_M : M ∩ N = M := by
  sorry

end NUMINAMATH_CALUDE_M_inter_N_eq_M_l635_63577


namespace NUMINAMATH_CALUDE_sum_of_odd_and_multiples_of_five_l635_63524

/-- The number of five-digit odd numbers -/
def A : ℕ := 45000

/-- The number of five-digit multiples of 5 -/
def B : ℕ := 18000

/-- The sum of five-digit odd numbers and five-digit multiples of 5 is 63000 -/
theorem sum_of_odd_and_multiples_of_five : A + B = 63000 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_odd_and_multiples_of_five_l635_63524


namespace NUMINAMATH_CALUDE_nancy_carrots_count_l635_63525

/-- The number of carrots Nancy's mother picked -/
def mother_carrots : ℕ := 47

/-- The number of good carrots -/
def good_carrots : ℕ := 71

/-- The number of bad carrots -/
def bad_carrots : ℕ := 14

/-- The number of carrots Nancy picked -/
def nancy_carrots : ℕ := 38

theorem nancy_carrots_count : 
  nancy_carrots = (good_carrots + bad_carrots) - mother_carrots := by
  sorry

end NUMINAMATH_CALUDE_nancy_carrots_count_l635_63525


namespace NUMINAMATH_CALUDE_parabola_properties_l635_63501

-- Define the parabola
def parabola (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 3

-- Define the conditions
theorem parabola_properties :
  ∃ (a b : ℝ),
    (parabola a b 3 = 0) ∧
    (parabola a b 4 = 3) ∧
    (∀ x, parabola a b x = x^2 - 4*x + 3) ∧
    (a > 0) ∧
    (∀ x, parabola a b x ≥ parabola a b 2) ∧
    (parabola a b 2 = -1) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l635_63501


namespace NUMINAMATH_CALUDE_calculate_loss_percentage_l635_63513

/-- Calculates the percentage of loss given the selling prices and profit percentage --/
theorem calculate_loss_percentage
  (sp_profit : ℝ)       -- Selling price with profit
  (profit_percent : ℝ)  -- Profit percentage
  (sp_loss : ℝ)         -- Selling price with loss
  (h1 : sp_profit = 800)
  (h2 : profit_percent = 25)
  (h3 : sp_loss = 512)
  : (sp_profit * (100 / (100 + profit_percent)) - sp_loss) / 
    (sp_profit * (100 / (100 + profit_percent))) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_calculate_loss_percentage_l635_63513


namespace NUMINAMATH_CALUDE_round_trip_speed_l635_63527

/-- Proves that given specific conditions for a round trip, the return speed must be 48 mph -/
theorem round_trip_speed (distance : ℝ) (speed_ab : ℝ) (avg_speed : ℝ) (speed_ba : ℝ) : 
  distance = 120 →
  speed_ab = 80 →
  avg_speed = 60 →
  (2 * distance) / (distance / speed_ab + distance / speed_ba) = avg_speed →
  speed_ba = 48 := by
sorry

end NUMINAMATH_CALUDE_round_trip_speed_l635_63527


namespace NUMINAMATH_CALUDE_f_properties_l635_63502

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 / x + a * Real.log x

theorem f_properties (a : ℝ) (h_a : a > 0) :
  (∃ (x_min : ℝ), x_min > 0 ∧ x_min = 1/2 ∧ 
    (∀ (x : ℝ), x > 0 → f a x ≥ f a x_min)) ∧
  (¬∃ (x_max : ℝ), x_max > 0 ∧ 
    (∀ (x : ℝ), x > 0 → f a x ≤ f a x_max)) ∧
  ((∃ (x : ℝ), x > 0 ∧ f a x < 2) ↔ (a > 0 ∧ a ≠ 2)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l635_63502


namespace NUMINAMATH_CALUDE_sin_330_degrees_l635_63526

theorem sin_330_degrees : Real.sin (330 * π / 180) = -(1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l635_63526


namespace NUMINAMATH_CALUDE_problem_statement_l635_63520

theorem problem_statement : (3.14 - Real.pi) ^ 0 - 2 ^ (-1 : ℤ) = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l635_63520


namespace NUMINAMATH_CALUDE_equation_solution_l635_63533

theorem equation_solution : ∃! x : ℚ, (3 * x - 15) / 4 = (x + 9) / 5 ∧ x = 10 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l635_63533


namespace NUMINAMATH_CALUDE_right_triangle_legs_l635_63529

theorem right_triangle_legs (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  a^2 + b^2 = 37^2 → 
  a * b = (a + 7) * (b - 2) →
  (a = 35 ∧ b = 12) ∨ (a = 12 ∧ b = 35) := by
sorry

end NUMINAMATH_CALUDE_right_triangle_legs_l635_63529


namespace NUMINAMATH_CALUDE_problem_statement_l635_63539

theorem problem_statement (a b c : ℝ) (h : a + b = ab ∧ ab = c) :
  (c ≠ 0 → (2*a - 3*a*b + 2*b) / (5*a + 7*a*b + 5*b) = -1/12) ∧
  (a = 3 → b + c = 6) ∧
  (c ≠ 0 → (1-a)*(1-b) = 1/a + 1/b) ∧
  (c = 4 → a^2 + b^2 = 8) := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l635_63539


namespace NUMINAMATH_CALUDE_system_solution_l635_63552

theorem system_solution : 
  ∀ x y : ℝ, 
    x^2 + 3*x*y = 18 ∧ x*y + 3*y^2 = 6 → 
      (x = 3 ∧ y = 1) ∨ (x = -3 ∧ y = -1) :=
by
  sorry

end NUMINAMATH_CALUDE_system_solution_l635_63552


namespace NUMINAMATH_CALUDE_certain_number_l635_63594

theorem certain_number : ∃ x : ℤ, x - 9 = 5 ∧ x = 14 := by sorry

end NUMINAMATH_CALUDE_certain_number_l635_63594


namespace NUMINAMATH_CALUDE_particular_solutions_l635_63518

/-- The differential equation -/
def diff_eq (x y y' : ℝ) : Prop :=
  x * y'^2 - 2 * y * y' + 4 * x = 0

/-- The general integral -/
def general_integral (x y C : ℝ) : Prop :=
  x^2 = C * (y - C)

/-- Theorem stating that y = 2x and y = -2x are particular solutions -/
theorem particular_solutions (x : ℝ) (hx : x > 0) :
  (diff_eq x (2*x) 2 ∧ diff_eq x (-2*x) (-2)) ∧
  (∃ C, general_integral x (2*x) C) ∧
  (∃ C, general_integral x (-2*x) C) :=
sorry

end NUMINAMATH_CALUDE_particular_solutions_l635_63518


namespace NUMINAMATH_CALUDE_power_calculations_l635_63557

theorem power_calculations :
  ((-2 : ℤ) ^ (0 : ℕ) = 1) ∧
  ((-3 : ℚ) ^ (-3 : ℤ) = -1/27) := by
  sorry

end NUMINAMATH_CALUDE_power_calculations_l635_63557


namespace NUMINAMATH_CALUDE_trailing_zeros_mod_500_l635_63581

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def trailingZeros (n : ℕ) : ℕ :=
  (List.range 51).map factorial
    |> List.foldl (·*·) 1
    |> Nat.digits 10
    |> List.reverse
    |> List.takeWhile (·==0)
    |> List.length

theorem trailing_zeros_mod_500 :
  trailingZeros 50 % 500 = 12 := by sorry

end NUMINAMATH_CALUDE_trailing_zeros_mod_500_l635_63581


namespace NUMINAMATH_CALUDE_height_difference_l635_63571

/-- Heights of people in centimeters -/
structure Heights where
  janet : ℝ
  charlene : ℝ
  pablo : ℝ
  ruby : ℝ

/-- Problem conditions -/
def problem_conditions (h : Heights) : Prop :=
  h.janet = 62 ∧
  h.charlene = 2 * h.janet ∧
  h.pablo = h.charlene + 70 ∧
  h.ruby = 192 ∧
  h.pablo > h.ruby

/-- Theorem stating the height difference between Pablo and Ruby -/
theorem height_difference (h : Heights) 
  (hc : problem_conditions h) : h.pablo - h.ruby = 2 := by
  sorry

end NUMINAMATH_CALUDE_height_difference_l635_63571


namespace NUMINAMATH_CALUDE_investment_pays_off_after_9_months_l635_63559

/-- Cumulative net income function for the first 5 months after improvement -/
def g (n : ℕ) : ℚ :=
  if n ≤ 5 then n^2 + 100*n else 109*n - 20

/-- Monthly income without improvement (in 10,000 yuan) -/
def monthly_income : ℚ := 70

/-- Fine function without improvement (in 10,000 yuan) -/
def fine (n : ℕ) : ℚ := n^2 + 2*n

/-- Initial investment (in 10,000 yuan) -/
def investment : ℚ := 500

/-- One-time reward after improvement (in 10,000 yuan) -/
def reward : ℚ := 100

/-- Cumulative net income with improvement (in 10,000 yuan) -/
def income_with_improvement (n : ℕ) : ℚ :=
  g n - investment + reward

/-- Cumulative net income without improvement (in 10,000 yuan) -/
def income_without_improvement (n : ℕ) : ℚ :=
  n * monthly_income - fine n

theorem investment_pays_off_after_9_months :
  ∀ n : ℕ, n ≥ 9 → income_with_improvement n > income_without_improvement n :=
sorry

end NUMINAMATH_CALUDE_investment_pays_off_after_9_months_l635_63559


namespace NUMINAMATH_CALUDE_correct_contributions_l635_63540

/-- Represents the business contribution problem -/
structure BusinessContribution where
  total : ℝ
  a_months : ℝ
  b_months : ℝ
  a_received : ℝ
  b_received : ℝ

/-- Theorem stating the correct contributions of A and B -/
theorem correct_contributions (bc : BusinessContribution)
  (h_total : bc.total = 3400)
  (h_a_months : bc.a_months = 12)
  (h_b_months : bc.b_months = 16)
  (h_a_received : bc.a_received = 2070)
  (h_b_received : bc.b_received = 1920) :
  ∃ (a_contribution b_contribution : ℝ),
    a_contribution = 1800 ∧
    b_contribution = 1600 ∧
    a_contribution + b_contribution = bc.total ∧
    (bc.a_received - a_contribution) / (bc.b_received - (bc.total - a_contribution)) =
      (bc.a_months * a_contribution) / (bc.b_months * (bc.total - a_contribution)) :=
by sorry

end NUMINAMATH_CALUDE_correct_contributions_l635_63540


namespace NUMINAMATH_CALUDE_amp_neg_eight_five_l635_63599

def amp (a b : Int) : Int := (a + b) * (a - b)

theorem amp_neg_eight_five : amp (-8) 5 = 39 := by sorry

end NUMINAMATH_CALUDE_amp_neg_eight_five_l635_63599


namespace NUMINAMATH_CALUDE_seconds_in_12_5_minutes_l635_63542

/-- The number of seconds in one minute -/
def seconds_per_minute : ℕ := 60

/-- The number of minutes to convert -/
def minutes_to_convert : ℚ := 12.5

/-- Theorem: The number of seconds in 12.5 minutes is 750 -/
theorem seconds_in_12_5_minutes :
  (minutes_to_convert * seconds_per_minute : ℚ) = 750 := by sorry

end NUMINAMATH_CALUDE_seconds_in_12_5_minutes_l635_63542


namespace NUMINAMATH_CALUDE_no_real_roots_quadratic_l635_63595

theorem no_real_roots_quadratic (k : ℝ) : 
  (∀ x : ℝ, x^2 + 2*x - 2*k + 3 ≠ 0) → k < 1 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_quadratic_l635_63595


namespace NUMINAMATH_CALUDE_ellipse_equation_1_l635_63548

/-- Given an ellipse with semi-major axis a = 6 and eccentricity e = 1/3,
    prove that its standard equation is x²/36 + y²/32 = 1 -/
theorem ellipse_equation_1 (x y : ℝ) (a b c : ℝ) (h1 : a = 6) (h2 : c/a = 1/3) :
  x^2/a^2 + y^2/b^2 = 1 ↔ x^2/36 + y^2/32 = 1 := by sorry

end NUMINAMATH_CALUDE_ellipse_equation_1_l635_63548


namespace NUMINAMATH_CALUDE_smallest_n_below_threshold_l635_63515

/-- The probability of drawing a red marble on the nth draw -/
def P (n : ℕ) : ℚ := 1 / (n * (n + 1))

/-- The number of boxes -/
def num_boxes : ℕ := 1000

/-- The threshold probability -/
def threshold : ℚ := 1 / 1000

theorem smallest_n_below_threshold :
  (∀ k < 32, P k ≥ threshold) ∧ P 32 < threshold := by sorry

end NUMINAMATH_CALUDE_smallest_n_below_threshold_l635_63515


namespace NUMINAMATH_CALUDE_fixed_point_of_logarithmic_function_l635_63586

/-- The logarithm function with base a -/
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

/-- The function f(x) = log_a(x + 3) - 1 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log a (x + 3) - 1

theorem fixed_point_of_logarithmic_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a (-2) = -1 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_logarithmic_function_l635_63586


namespace NUMINAMATH_CALUDE_transformations_correctness_l635_63570

-- Define the transformations
def transformation_A (a b c : ℝ) : Prop := (c ≠ 0) → (a * c) / (b * c) = a / b

def transformation_B (a b : ℝ) : Prop := (a + b ≠ 0) → (-a - b) / (a + b) = -1

def transformation_C (m n : ℝ) : Prop := 
  (0.2 * m - 0.3 * n ≠ 0) → (0.5 * m + n) / (0.2 * m - 0.3 * n) = (5 * m + 10 * n) / (2 * m - 3 * n)

def transformation_D (x : ℝ) : Prop := (x + 1 ≠ 0) → (2 - x) / (x + 1) = (x - 2) / (1 + x)

-- Theorem stating which transformations are correct and which is incorrect
theorem transformations_correctness :
  (∀ a b c, transformation_A a b c) ∧
  (∀ a b, transformation_B a b) ∧
  (∀ m n, transformation_C m n) ∧
  ¬(∀ x, transformation_D x) := by
  sorry

end NUMINAMATH_CALUDE_transformations_correctness_l635_63570
