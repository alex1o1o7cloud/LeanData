import Mathlib

namespace NUMINAMATH_CALUDE_science_fiction_books_l1743_174308

theorem science_fiction_books (pages_per_book : ℕ) (total_pages : ℕ) (h1 : pages_per_book = 478) (h2 : total_pages = 3824) :
  total_pages / pages_per_book = 8 := by
  sorry

end NUMINAMATH_CALUDE_science_fiction_books_l1743_174308


namespace NUMINAMATH_CALUDE_intersection_sum_l1743_174379

def M : Set ℝ := {x | x^2 - 4*x < 0}
def N (m : ℝ) : Set ℝ := {x | m < x ∧ x < 5}

theorem intersection_sum (m n : ℝ) : 
  M ∩ N m = {x | 3 < x ∧ x < n} → m + n = 7 := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l1743_174379


namespace NUMINAMATH_CALUDE_complete_square_sum_l1743_174384

theorem complete_square_sum (a b c : ℤ) : 
  (∀ x : ℝ, 25 * x^2 + 30 * x - 75 = 0 ↔ (a * x + b)^2 = c) →
  a > 0 →
  a + b + c = -58 :=
by sorry

end NUMINAMATH_CALUDE_complete_square_sum_l1743_174384


namespace NUMINAMATH_CALUDE_machine_work_time_l1743_174322

theorem machine_work_time (x : ℝ) : 
  (x > 0) →
  (1 / (x + 4) + 1 / (x + 2) + 1 / (2 * x + 2) = 1 / x) →
  x = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_machine_work_time_l1743_174322


namespace NUMINAMATH_CALUDE_bus_empty_seats_l1743_174396

/-- Calculates the number of empty seats on a bus after a series of boarding and disembarking events -/
def empty_seats_after_events (rows : ℕ) (seats_per_row : ℕ) 
  (initial_boarding : ℕ) 
  (stop1_board : ℕ) (stop1_disembark : ℕ)
  (stop2_board : ℕ) (stop2_disembark : ℕ)
  (stop3_board : ℕ) (stop3_disembark : ℕ) : ℕ :=
  let total_seats := rows * seats_per_row
  let after_initial := total_seats - initial_boarding
  let after_stop1 := after_initial - (stop1_board - stop1_disembark)
  let after_stop2 := after_stop1 - (stop2_board - stop2_disembark)
  let after_stop3 := after_stop2 - (stop3_board - stop3_disembark)
  after_stop3

theorem bus_empty_seats : 
  empty_seats_after_events 23 4 16 15 3 17 10 12 8 = 53 := by
  sorry

end NUMINAMATH_CALUDE_bus_empty_seats_l1743_174396


namespace NUMINAMATH_CALUDE_balloon_height_theorem_l1743_174328

/-- Calculates the maximum height a balloon can fly given the budget and costs --/
def maxBalloonHeight (budget initialCost heliumPrice1 heliumPrice2 heliumPrice3 : ℚ) 
  (threshold1 threshold2 : ℚ) (heightPerOunce : ℚ) : ℚ :=
  let remainingBudget := budget - initialCost
  let ounces1 := min (remainingBudget / heliumPrice1) threshold1
  let ounces2 := min ((remainingBudget - ounces1 * heliumPrice1) / heliumPrice2) (threshold2 - threshold1)
  let totalOunces := ounces1 + ounces2
  totalOunces * heightPerOunce

/-- The maximum height the balloon can fly is 11,000 feet --/
theorem balloon_height_theorem : 
  maxBalloonHeight 200 74 1.2 1.1 1 50 120 100 = 11000 := by
  sorry

end NUMINAMATH_CALUDE_balloon_height_theorem_l1743_174328


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l1743_174320

theorem sum_of_fractions_equals_one
  (a b c x y z : ℝ)
  (eq1 : 11 * x + b * y + c * z = 0)
  (eq2 : a * x + 17 * y + c * z = 0)
  (eq3 : a * x + b * y + 37 * z = 0)
  (ha : a ≠ 11)
  (hx : x ≠ 0) :
  a / (a - 11) + b / (b - 17) + c / (c - 37) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l1743_174320


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l1743_174303

theorem purely_imaginary_complex_number (a : ℝ) :
  let z : ℂ := (1 - Complex.I) * (-2 + a * Complex.I)
  (z.re = 0) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l1743_174303


namespace NUMINAMATH_CALUDE_joseph_birth_year_is_1978_l1743_174343

/-- The year of the first revised AMC 8 -/
def first_revised_amc8_year : ℕ := 1987

/-- The year Joseph took the seventh AMC 8 after revision -/
def seventh_amc8_year : ℕ := first_revised_amc8_year + 6

/-- Joseph's age when he took the seventh AMC 8 after revision -/
def joseph_age_at_seventh_amc8 : ℕ := 15

/-- Joseph's birth year -/
def joseph_birth_year : ℕ := seventh_amc8_year - joseph_age_at_seventh_amc8

theorem joseph_birth_year_is_1978 : joseph_birth_year = 1978 := by
  sorry

end NUMINAMATH_CALUDE_joseph_birth_year_is_1978_l1743_174343


namespace NUMINAMATH_CALUDE_division_by_reciprocal_five_divided_by_one_fifth_l1743_174331

theorem division_by_reciprocal (a : ℝ) (b : ℝ) (hb : b ≠ 0) :
  a / (1 / b) = a * b := by sorry

theorem five_divided_by_one_fifth :
  5 / (1 / 5) = 25 := by sorry

end NUMINAMATH_CALUDE_division_by_reciprocal_five_divided_by_one_fifth_l1743_174331


namespace NUMINAMATH_CALUDE_mary_savings_problem_l1743_174330

theorem mary_savings_problem (S : ℝ) (x : ℝ) (h1 : S > 0) (h2 : 0 ≤ x ∧ x ≤ 1) 
  (h3 : 12 * S * x = 7 * S * (1 - x)) : 
  1 - x = 12 / 19 := by sorry

end NUMINAMATH_CALUDE_mary_savings_problem_l1743_174330


namespace NUMINAMATH_CALUDE_milk_discount_l1743_174392

/-- Calculates the discount on milk given grocery prices and remaining money --/
theorem milk_discount (initial_money : ℝ) (milk_price bread_price detergent_price banana_price_per_pound : ℝ)
  (banana_pounds : ℝ) (detergent_coupon : ℝ) (money_left : ℝ) :
  initial_money = 20 ∧
  milk_price = 4 ∧
  bread_price = 3.5 ∧
  detergent_price = 10.25 ∧
  banana_price_per_pound = 0.75 ∧
  banana_pounds = 2 ∧
  detergent_coupon = 1.25 ∧
  money_left = 4 →
  initial_money - (bread_price + (detergent_price - detergent_coupon) + 
    (banana_price_per_pound * banana_pounds) + money_left) = 2 :=
by sorry

end NUMINAMATH_CALUDE_milk_discount_l1743_174392


namespace NUMINAMATH_CALUDE_firewood_sacks_filled_l1743_174356

/-- Calculates the number of sacks filled with firewood -/
def sacks_filled (wood_per_sack : ℕ) (total_wood : ℕ) : ℕ :=
  total_wood / wood_per_sack

/-- Theorem stating that the number of sacks filled is 4 -/
theorem firewood_sacks_filled :
  let wood_per_sack : ℕ := 20
  let total_wood : ℕ := 80
  sacks_filled wood_per_sack total_wood = 4 := by
  sorry

end NUMINAMATH_CALUDE_firewood_sacks_filled_l1743_174356


namespace NUMINAMATH_CALUDE_cell_count_after_12_days_first_six_days_growth_next_six_days_growth_l1743_174350

/-- Represents the cell growth model over 12 days -/
def CellGrowth : Nat → Nat
| 0 => 5  -- Initial cell count
| n + 1 =>
  if n < 6 then
    CellGrowth n * 3  -- Growth rate for first 6 days
  else if n < 12 then
    CellGrowth n * 2  -- Growth rate for next 6 days
  else
    CellGrowth n      -- No growth after 12 days

/-- Theorem stating the number of cells after 12 days -/
theorem cell_count_after_12_days :
  CellGrowth 12 = 180 := by
  sorry

/-- Verifies the growth pattern for the first 6 days -/
theorem first_six_days_growth (n : Nat) (h : n < 6) :
  CellGrowth (n + 1) = CellGrowth n * 3 := by
  sorry

/-- Verifies the growth pattern for days 7 to 12 -/
theorem next_six_days_growth (n : Nat) (h1 : 6 ≤ n) (h2 : n < 12) :
  CellGrowth (n + 1) = CellGrowth n * 2 := by
  sorry

end NUMINAMATH_CALUDE_cell_count_after_12_days_first_six_days_growth_next_six_days_growth_l1743_174350


namespace NUMINAMATH_CALUDE_enhanced_square_triangle_count_l1743_174311

/-- A square with diagonals, midpoint connections, and additional bisections -/
structure EnhancedSquare where
  /-- The original square -/
  square : Set (ℝ × ℝ)
  /-- The diagonals of the square -/
  diagonals : Set (Set (ℝ × ℝ))
  /-- The segments connecting midpoints of opposite sides -/
  midpoint_connections : Set (Set (ℝ × ℝ))
  /-- The additional bisections of midpoint connections -/
  bisections : Set (Set (ℝ × ℝ))

/-- A triangle in the enhanced square -/
structure Triangle where
  vertices : Fin 3 → (ℝ × ℝ)

/-- Count the number of triangles in the enhanced square -/
def countTriangles (es : EnhancedSquare) : ℕ :=
  sorry

/-- The main theorem: The number of triangles in the enhanced square is 28 -/
theorem enhanced_square_triangle_count (es : EnhancedSquare) : 
  countTriangles es = 28 := by
  sorry

end NUMINAMATH_CALUDE_enhanced_square_triangle_count_l1743_174311


namespace NUMINAMATH_CALUDE_walter_exceptional_days_l1743_174352

/-- Represents the number of days Walter performed his chores in each category -/
structure ChorePerformance where
  poor : ℕ
  adequate : ℕ
  exceptional : ℕ

/-- Theorem stating that given the conditions, Walter performed exceptionally well for 6 days -/
theorem walter_exceptional_days :
  ∃ (perf : ChorePerformance),
    perf.poor + perf.adequate + perf.exceptional = 15 ∧
    2 * perf.poor + 4 * perf.adequate + 7 * perf.exceptional = 70 ∧
    perf.exceptional = 6 := by
  sorry


end NUMINAMATH_CALUDE_walter_exceptional_days_l1743_174352


namespace NUMINAMATH_CALUDE_gcd_of_polynomials_l1743_174364

theorem gcd_of_polynomials (x : ℤ) (h : ∃ k : ℤ, x = 2 * k * 2027) :
  Int.gcd (3 * x^2 + 47 * x + 101) (x + 23) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_polynomials_l1743_174364


namespace NUMINAMATH_CALUDE_prob_at_least_one_2_or_3_is_7_16_l1743_174378

/-- The probability of at least one of two fair 8-sided dice showing a 2 or a 3 -/
def prob_at_least_one_2_or_3 : ℚ := 7/16

/-- Two fair 8-sided dice are rolled -/
axiom fair_8_sided_dice : True

theorem prob_at_least_one_2_or_3_is_7_16 :
  prob_at_least_one_2_or_3 = 7/16 :=
sorry

end NUMINAMATH_CALUDE_prob_at_least_one_2_or_3_is_7_16_l1743_174378


namespace NUMINAMATH_CALUDE_kyles_profit_is_99_l1743_174309

/-- The amount of money Kyle will make by selling all his remaining baked goods -/
def kyles_profit (initial_cookies initial_brownies : ℕ) 
                 (kyle_eats_cookies kyle_eats_brownies : ℕ) 
                 (mom_eats_cookies mom_eats_brownies : ℕ) 
                 (cookie_price brownie_price : ℚ) : ℚ :=
  let remaining_cookies := initial_cookies - kyle_eats_cookies - mom_eats_cookies
  let remaining_brownies := initial_brownies - kyle_eats_brownies - mom_eats_brownies
  remaining_cookies * cookie_price + remaining_brownies * brownie_price

/-- Theorem stating that Kyle will make $99 by selling all his remaining baked goods -/
theorem kyles_profit_is_99 : 
  kyles_profit 60 32 2 2 1 2 1 (3/2) = 99 := by
  sorry

end NUMINAMATH_CALUDE_kyles_profit_is_99_l1743_174309


namespace NUMINAMATH_CALUDE_specific_sequence_common_difference_l1743_174326

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  first_term : ℝ
  last_term : ℝ
  sum : ℝ
  is_arithmetic : ℝ → ℝ → ℝ → Prop

/-- The common difference of an arithmetic sequence -/
def common_difference (seq : ArithmeticSequence) : ℝ := 
  sorry

/-- Theorem stating the common difference of the specific sequence -/
theorem specific_sequence_common_difference :
  ∃ (seq : ArithmeticSequence), 
    seq.first_term = 5 ∧ 
    seq.last_term = 50 ∧ 
    seq.sum = 495 ∧ 
    common_difference seq = 45 / 17 := by
  sorry

end NUMINAMATH_CALUDE_specific_sequence_common_difference_l1743_174326


namespace NUMINAMATH_CALUDE_lcm_18_45_l1743_174338

theorem lcm_18_45 : Nat.lcm 18 45 = 90 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_45_l1743_174338


namespace NUMINAMATH_CALUDE_derivative_sin_2x_l1743_174339

theorem derivative_sin_2x (x : ℝ) : 
  deriv (λ x => Real.sin (2 * x)) x = 2 * Real.cos (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_derivative_sin_2x_l1743_174339


namespace NUMINAMATH_CALUDE_F_minimum_at_negative_one_F_monotonic_intervals_t_ge_3_F_monotonic_intervals_t_between_F_monotonic_intervals_t_le_neg_one_l1743_174301

-- Define the function F(x, t)
def F (x t : ℝ) : ℝ := |2*x + t| + x^2 + x + 1

-- Theorem for the minimum value when t = -1
theorem F_minimum_at_negative_one :
  ∃ (x_min : ℝ), F x_min (-1) = 7/4 ∧ ∀ (x : ℝ), F x (-1) ≥ 7/4 :=
sorry

-- Theorems for monotonic intervals
theorem F_monotonic_intervals_t_ge_3 (t : ℝ) (h : t ≥ 3) :
  (∀ x y : ℝ, x ≤ y ∧ y ≤ -3/2 → F x t ≥ F y t) ∧
  (∀ x y : ℝ, -3/2 ≤ x ∧ x ≤ y → F x t ≤ F y t) :=
sorry

theorem F_monotonic_intervals_t_between (t : ℝ) (h1 : -1 < t) (h2 : t < 3) :
  (∀ x y : ℝ, x ≤ y ∧ y ≤ -t/2 → F x t ≥ F y t) ∧
  (∀ x y : ℝ, -t/2 ≤ x ∧ x ≤ y → F x t ≤ F y t) :=
sorry

theorem F_monotonic_intervals_t_le_neg_one (t : ℝ) (h : t ≤ -1) :
  (∀ x y : ℝ, x ≤ y ∧ y ≤ 1/2 → F x t ≥ F y t) ∧
  (∀ x y : ℝ, 1/2 ≤ x ∧ x ≤ y → F x t ≤ F y t) :=
sorry

end NUMINAMATH_CALUDE_F_minimum_at_negative_one_F_monotonic_intervals_t_ge_3_F_monotonic_intervals_t_between_F_monotonic_intervals_t_le_neg_one_l1743_174301


namespace NUMINAMATH_CALUDE_y_derivative_l1743_174380

noncomputable def y (x : ℝ) : ℝ :=
  (Real.cos x) / (3 * (2 + Real.sin x)) + (4 / (3 * Real.sqrt 3)) * Real.arctan ((2 * Real.tan (x / 2) + 1) / Real.sqrt 3)

theorem y_derivative (x : ℝ) :
  deriv y x = (2 * Real.sin x + 7) / (3 * (2 + Real.sin x)^2) :=
sorry

end NUMINAMATH_CALUDE_y_derivative_l1743_174380


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1743_174306

/-- A quadratic function with zeros at -2 and 3 -/
def f (x : ℝ) : ℝ := x^2 + a*x + b
  where
  a : ℝ := -1  -- Derived from the zeros, but not explicitly using the solution
  b : ℝ := -6  -- Derived from the zeros, but not explicitly using the solution

/-- The theorem statement -/
theorem solution_set_of_inequality (x : ℝ) :
  (f (-2*x) * (-1) > 0) ↔ (-3/2 < x ∧ x < 1) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1743_174306


namespace NUMINAMATH_CALUDE_roots_of_g_l1743_174325

def f (a b x : ℝ) : ℝ := a * x - b

def g (a b x : ℝ) : ℝ := b * x^2 + 3 * a * x

theorem roots_of_g (a b : ℝ) (h : f a b 3 = 0) :
  {x : ℝ | g a b x = 0} = {-1, 0} := by sorry

end NUMINAMATH_CALUDE_roots_of_g_l1743_174325


namespace NUMINAMATH_CALUDE_pipes_remaining_proof_l1743_174346

/-- The number of pipes in a triangular pyramid with n layers -/
def triangular_pyramid (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The total number of pipes available -/
def total_pipes : ℕ := 200

/-- The maximum number of complete layers in the pyramid -/
def max_layers : ℕ := 19

/-- The number of pipes left over -/
def pipes_left_over : ℕ := total_pipes - triangular_pyramid max_layers

theorem pipes_remaining_proof :
  pipes_left_over = 10 :=
sorry

end NUMINAMATH_CALUDE_pipes_remaining_proof_l1743_174346


namespace NUMINAMATH_CALUDE_max_x5_value_l1743_174390

theorem max_x5_value (x₁ x₂ x₃ x₄ x₅ : ℕ) 
  (h : x₁ + x₂ + x₃ + x₄ + x₅ = x₁ * x₂ * x₃ * x₄ * x₅) :
  x₅ ≤ 5 ∧ ∃ y₁ y₂ y₃ y₄ : ℕ, y₁ + y₂ + y₃ + y₄ + 5 = y₁ * y₂ * y₃ * y₄ * 5 :=
by sorry

end NUMINAMATH_CALUDE_max_x5_value_l1743_174390


namespace NUMINAMATH_CALUDE_scientific_notation_of_34_million_l1743_174383

theorem scientific_notation_of_34_million :
  ∃ (a : ℝ) (n : ℤ), 34000000 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 3.4 ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_34_million_l1743_174383


namespace NUMINAMATH_CALUDE_angle_in_second_quadrant_l1743_174300

/-- Given an angle α in the second quadrant with P(x,4) on its terminal side and cos α = (1/5)x,
    prove that x = -3 and tan α = -4/3 -/
theorem angle_in_second_quadrant (α : Real) (x : Real) 
    (h1 : π / 2 < α ∧ α < π) -- α is in the second quadrant
    (h2 : x < 0) -- P(x,4) is on the terminal side in the second quadrant
    (h3 : Real.cos α = (1/5) * x) -- Given condition
    : x = -3 ∧ Real.tan α = -4/3 := by
  sorry


end NUMINAMATH_CALUDE_angle_in_second_quadrant_l1743_174300


namespace NUMINAMATH_CALUDE_local_max_implies_c_eq_six_l1743_174355

/-- Given a function f(x) = x(x-c)² where c is a constant, 
    if f has a local maximum at x = 2, then c = 6 -/
theorem local_max_implies_c_eq_six (c : ℝ) : 
  let f : ℝ → ℝ := λ x => x * (x - c)^2
  (∃ δ > 0, ∀ x ∈ Set.Ioo (2 - δ) (2 + δ), f x ≤ f 2) →
  c = 6 := by
  sorry

end NUMINAMATH_CALUDE_local_max_implies_c_eq_six_l1743_174355


namespace NUMINAMATH_CALUDE_distance_between_opposite_faces_of_unit_octahedron_l1743_174316

/-- A regular octahedron is a polyhedron with 8 faces, where each face is an equilateral triangle -/
structure RegularOctahedron where
  side_length : ℝ

/-- The distance between two opposite faces of a regular octahedron -/
def distance_between_opposite_faces (o : RegularOctahedron) : ℝ :=
  sorry

/-- Theorem: In a regular octahedron with side length 1, the distance between two opposite faces is √6/3 -/
theorem distance_between_opposite_faces_of_unit_octahedron :
  let o : RegularOctahedron := ⟨1⟩
  distance_between_opposite_faces o = Real.sqrt 6 / 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_opposite_faces_of_unit_octahedron_l1743_174316


namespace NUMINAMATH_CALUDE_pencil_box_sequence_l1743_174304

theorem pencil_box_sequence (a : ℕ → ℕ) (h1 : a 1 = 78) (h2 : a 2 = 87) (h3 : a 3 = 96) (h5 : a 5 = 114)
  (h_arithmetic : ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + 9) : a 4 = 105 := by
  sorry

end NUMINAMATH_CALUDE_pencil_box_sequence_l1743_174304


namespace NUMINAMATH_CALUDE_cube_cross_sections_l1743_174387

/-- A regular polygon obtained by cutting a cube with a plane. -/
inductive CubeCrossSection
  | Triangle
  | Square
  | Hexagon

/-- The set of all possible regular polygons obtained by cutting a cube with a plane. -/
def ValidCubeCrossSections : Set CubeCrossSection :=
  {CubeCrossSection.Triangle, CubeCrossSection.Square, CubeCrossSection.Hexagon}

/-- Theorem: The only regular polygons that can be obtained by cutting a cube with a plane
    are triangles, squares, and hexagons. -/
theorem cube_cross_sections (cs : CubeCrossSection) :
  cs ∈ ValidCubeCrossSections := by sorry

end NUMINAMATH_CALUDE_cube_cross_sections_l1743_174387


namespace NUMINAMATH_CALUDE_geometric_sequence_2010th_term_l1743_174319

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  p : ℝ
  q : ℝ
  first_term : ℝ
  second_term : ℝ
  third_term : ℝ
  fourth_term : ℝ
  h1 : first_term = p
  h2 : second_term = 9
  h3 : third_term = 3 * p / q
  h4 : fourth_term = 3 * p * q

/-- The 2010th term of the geometric sequence is 9 -/
theorem geometric_sequence_2010th_term (seq : GeometricSequence) :
  let r := seq.second_term / seq.first_term
  seq.first_term * r^(2009 : ℕ) = 9 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_2010th_term_l1743_174319


namespace NUMINAMATH_CALUDE_turban_count_is_one_l1743_174313

/-- The number of turbans given as part of the annual salary -/
def turban_count : ℕ := sorry

/-- The price of one turban in Rupees -/
def turban_price : ℕ := 30

/-- The base salary in Rupees -/
def base_salary : ℕ := 90

/-- The total annual salary in Rupees -/
def annual_salary : ℕ := base_salary + turban_count * turban_price

/-- The fraction of the year worked by the servant -/
def fraction_worked : ℚ := 3/4

/-- The amount received by the servant after 9 months in Rupees -/
def amount_received : ℕ := 60 + turban_price

theorem turban_count_is_one :
  (fraction_worked * annual_salary = amount_received) → turban_count = 1 := by
  sorry

end NUMINAMATH_CALUDE_turban_count_is_one_l1743_174313


namespace NUMINAMATH_CALUDE_smallest_k_for_tangent_circle_l1743_174377

theorem smallest_k_for_tangent_circle : ∃ (h : ℕ+), 
  (1 - h.val)^2 + (1000 + 58 - h.val)^2 = h.val^2 ∧
  ∀ (k : ℕ), k < 58 → ¬∃ (h : ℕ+), (1 - h.val)^2 + (1000 + k - h.val)^2 = h.val^2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_for_tangent_circle_l1743_174377


namespace NUMINAMATH_CALUDE_tickets_won_later_l1743_174366

/-- Given Cody's initial tickets, tickets spent on a beanie, and final ticket count,
    prove the number of tickets he won later. -/
theorem tickets_won_later
  (initial_tickets : ℕ)
  (tickets_spent : ℕ)
  (final_tickets : ℕ)
  (h1 : initial_tickets = 49)
  (h2 : tickets_spent = 25)
  (h3 : final_tickets = 30) :
  final_tickets - (initial_tickets - tickets_spent) = 6 := by
  sorry

end NUMINAMATH_CALUDE_tickets_won_later_l1743_174366


namespace NUMINAMATH_CALUDE_lcm_from_hcf_and_product_l1743_174388

theorem lcm_from_hcf_and_product (a b : ℕ+) : 
  Nat.gcd a b = 21 → a * b = 138567 → Nat.lcm a b = 6603 := by
  sorry

end NUMINAMATH_CALUDE_lcm_from_hcf_and_product_l1743_174388


namespace NUMINAMATH_CALUDE_buddy_cards_l1743_174305

/-- Calculates the number of baseball cards Buddy has on Saturday --/
def saturday_cards (initial : ℕ) : ℕ :=
  let tuesday := initial - (initial * 30 / 100)
  let wednesday := tuesday + (tuesday * 20 / 100)
  let thursday := wednesday - (wednesday / 4)
  let friday := thursday + (thursday / 3)
  friday + (friday * 2)

/-- Theorem stating that Buddy will have 252 cards on Saturday --/
theorem buddy_cards : saturday_cards 100 = 252 := by
  sorry

end NUMINAMATH_CALUDE_buddy_cards_l1743_174305


namespace NUMINAMATH_CALUDE_first_train_speed_l1743_174318

/-- Given two trains with a speed ratio of 7:8, where the second train travels 400 km in 4 hours,
    prove that the speed of the first train is 87.5 km/h. -/
theorem first_train_speed
  (speed_ratio : ℚ) -- Ratio of speeds between the two trains
  (distance : ℝ) -- Distance traveled by the second train
  (time : ℝ) -- Time taken by the second train
  (h1 : speed_ratio = 7 / 8) -- The ratio of speeds is 7:8
  (h2 : distance = 400) -- The second train travels 400 km
  (h3 : time = 4) -- The second train takes 4 hours
  : ∃ (speed1 : ℝ), speed1 = 87.5 := by
  sorry

end NUMINAMATH_CALUDE_first_train_speed_l1743_174318


namespace NUMINAMATH_CALUDE_no_primes_in_sequence_infinitely_many_x_with_no_primes_l1743_174307

/-- Definition of the sequence a_n -/
def a (x : ℕ) : ℕ → ℕ
| 0 => 1
| 1 => x + 1
| (n + 2) => x * a x (n + 1) - a x n

/-- Theorem stating that for any c ≥ 3, the sequence contains no primes when x = c² - 2 -/
theorem no_primes_in_sequence (c : ℕ) (h : c ≥ 3) :
  ∀ n : ℕ, ¬ Nat.Prime (a (c^2 - 2) n) := by
  sorry

/-- Corollary: There exist infinitely many x such that the sequence contains no primes -/
theorem infinitely_many_x_with_no_primes :
  ∃ f : ℕ → ℕ, Monotone f ∧ ∀ k : ℕ, ∀ n : ℕ, ¬ Nat.Prime (a (f k) n) := by
  sorry

end NUMINAMATH_CALUDE_no_primes_in_sequence_infinitely_many_x_with_no_primes_l1743_174307


namespace NUMINAMATH_CALUDE_smallest_n_with_equal_digits_sum_l1743_174365

/-- The sum of the first n positive integers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Check if all digits of a number are equal -/
def all_digits_equal (n : ℕ) : Prop := 
  ∃ (d : ℕ) (k : ℕ), d ∈ Finset.range 10 ∧ n = d * (10^k - 1) / 9

/-- Sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

theorem smallest_n_with_equal_digits_sum : 
  ∃ (N : ℕ), 
    (∀ m : ℕ, m < N → ¬(all_digits_equal (m * sum_first_n 9))) ∧
    (all_digits_equal (N * sum_first_n 9)) ∧
    (digit_sum N = 37) := by sorry

end NUMINAMATH_CALUDE_smallest_n_with_equal_digits_sum_l1743_174365


namespace NUMINAMATH_CALUDE_red_apple_count_l1743_174342

/-- Represents the number of apples of each color --/
structure AppleCount where
  red : ℕ
  yellow : ℕ
  green : ℕ

/-- Represents Eric's apple boxes --/
def eric_apples : AppleCount :=
  { red := 0, yellow := 0, green := 0 }

theorem red_apple_count : eric_apples.red = 7 := by
  sorry

/-- The total number of apples --/
def total_apples : ℕ := 28

/-- Red apples make up 1/2 of the apples in the first box --/
axiom red_half_first_box : eric_apples.red = eric_apples.yellow

/-- After moving red apples, they make up 1/3 of the second box --/
axiom red_third_second_box : 3 * eric_apples.red = eric_apples.red + eric_apples.green

/-- The total number of apples is the sum of all colors --/
axiom total_apples_sum : total_apples = eric_apples.red + eric_apples.yellow + eric_apples.green

end NUMINAMATH_CALUDE_red_apple_count_l1743_174342


namespace NUMINAMATH_CALUDE_survey_respondents_l1743_174363

theorem survey_respondents (preferred_x : ℕ) (ratio_x : ℕ) (ratio_y : ℕ) : 
  preferred_x = 360 → ratio_x = 9 → ratio_y = 1 → 
  ∃ (total : ℕ), total = preferred_x + (preferred_x * ratio_y) / ratio_x ∧ total = 400 :=
by
  sorry

end NUMINAMATH_CALUDE_survey_respondents_l1743_174363


namespace NUMINAMATH_CALUDE_weighted_arithmetic_geometric_mean_inequality_l1743_174347

theorem weighted_arithmetic_geometric_mean_inequality 
  {n : ℕ} (a w : Fin n → ℝ) (h_pos_a : ∀ i, a i > 0) (h_pos_w : ∀ i, w i > 0) :
  let W := (Finset.univ.sum w)
  (W⁻¹ * Finset.univ.sum (λ i => w i * a i)) ≥ 
    (Finset.univ.prod (λ i => (a i) ^ (w i))) ^ (W⁻¹) := by
  sorry

end NUMINAMATH_CALUDE_weighted_arithmetic_geometric_mean_inequality_l1743_174347


namespace NUMINAMATH_CALUDE_shopkeeper_profit_l1743_174399

theorem shopkeeper_profit (total_apples : ℝ) (profit_rate1 profit_rate2 : ℝ) 
  (portion1 portion2 : ℝ) :
  total_apples = 280 ∧ 
  profit_rate1 = 0.1 ∧ 
  profit_rate2 = 0.3 ∧ 
  portion1 = 0.4 ∧ 
  portion2 = 0.6 ∧ 
  portion1 + portion2 = 1 →
  let selling_price1 := portion1 * total_apples * (1 + profit_rate1)
  let selling_price2 := portion2 * total_apples * (1 + profit_rate2)
  let total_selling_price := selling_price1 + selling_price2
  let total_profit := total_selling_price - total_apples
  let percentage_profit := (total_profit / total_apples) * 100
  percentage_profit = 22 := by
sorry

end NUMINAMATH_CALUDE_shopkeeper_profit_l1743_174399


namespace NUMINAMATH_CALUDE_problem_solution_l1743_174389

theorem problem_solution (y₁ y₂ y₃ y₄ y₅ y₆ y₇ : ℝ) 
  (h₁ : y₁ + 3*y₂ + 5*y₃ + 7*y₄ + 9*y₅ + 11*y₆ + 13*y₇ = 0)
  (h₂ : 3*y₁ + 5*y₂ + 7*y₃ + 9*y₄ + 11*y₅ + 13*y₆ + 15*y₇ = 10)
  (h₃ : 5*y₁ + 7*y₂ + 9*y₃ + 11*y₄ + 13*y₅ + 15*y₆ + 17*y₇ = 104) :
  7*y₁ + 9*y₂ + 11*y₃ + 13*y₄ + 15*y₅ + 17*y₆ + 19*y₇ = 282 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1743_174389


namespace NUMINAMATH_CALUDE_chess_tournament_schedule_ways_l1743_174329

/-- Represents a chess tournament between two schools -/
structure ChessTournament where
  /-- Number of players per school -/
  players_per_school : Nat
  /-- Number of games each player plays against each opponent -/
  games_per_opponent : Nat
  /-- Number of rounds in the tournament -/
  num_rounds : Nat
  /-- Number of games played simultaneously in each round -/
  games_per_round : Nat

/-- Calculate the number of ways to schedule a chess tournament -/
def scheduleWays (t : ChessTournament) : Nat :=
  (t.num_rounds.factorial) + (t.num_rounds.factorial / (2^(t.num_rounds / 2)))

/-- Theorem stating the number of ways to schedule the specific chess tournament -/
theorem chess_tournament_schedule_ways :
  let t : ChessTournament := {
    players_per_school := 4,
    games_per_opponent := 2,
    num_rounds := 8,
    games_per_round := 4
  }
  scheduleWays t = 42840 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_schedule_ways_l1743_174329


namespace NUMINAMATH_CALUDE_dividend_calculation_l1743_174376

theorem dividend_calculation (divisor remainder quotient : ℕ) : 
  divisor = 17 → remainder = 8 → quotient = 4 → 
  divisor * quotient + remainder = 76 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l1743_174376


namespace NUMINAMATH_CALUDE_yellow_score_mixture_l1743_174370

theorem yellow_score_mixture (white_ratio : ℕ) (black_ratio : ℕ) (total_yellow : ℕ) 
  (h1 : white_ratio = 7)
  (h2 : black_ratio = 6)
  (h3 : total_yellow = 78) :
  (white_ratio * (total_yellow / (white_ratio + black_ratio)) - 
   black_ratio * (total_yellow / (white_ratio + black_ratio))) / total_yellow = 1 / 13 := by
  sorry

end NUMINAMATH_CALUDE_yellow_score_mixture_l1743_174370


namespace NUMINAMATH_CALUDE_monotone_increasing_iff_t_geq_five_l1743_174382

def a (x : ℝ) : ℝ × ℝ := (x^2, x + 1)
def b (x t : ℝ) : ℝ × ℝ := (1 - x, t)

def f (x t : ℝ) : ℝ := (a x).1 * (b x t).1 + (a x).2 * (b x t).2

theorem monotone_increasing_iff_t_geq_five :
  ∀ t : ℝ, (∀ x ∈ Set.Ioo (-1 : ℝ) 1, ∀ y ∈ Set.Ioo (-1 : ℝ) 1, x < y → f x t < f y t) ↔ t ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_monotone_increasing_iff_t_geq_five_l1743_174382


namespace NUMINAMATH_CALUDE_unique_cube_difference_61_l1743_174358

theorem unique_cube_difference_61 :
  ∃! (n k : ℕ), n^3 - k^3 = 61 :=
by sorry

end NUMINAMATH_CALUDE_unique_cube_difference_61_l1743_174358


namespace NUMINAMATH_CALUDE_inequality_one_inequality_two_l1743_174345

-- Statement 1
theorem inequality_one (a : ℝ) (ha : a > 0) :
  Real.sqrt (a + 2) - Real.sqrt (a + 6) > Real.sqrt a - Real.sqrt (a + 4) := by
  sorry

-- Statement 2
theorem inequality_two (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  Real.sqrt (a + 1/2) + Real.sqrt (b + 1/2) ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_one_inequality_two_l1743_174345


namespace NUMINAMATH_CALUDE_conditional_equivalence_l1743_174302

theorem conditional_equivalence (P Q : Prop) :
  (P → ¬Q) ↔ (Q → ¬P) := by sorry

end NUMINAMATH_CALUDE_conditional_equivalence_l1743_174302


namespace NUMINAMATH_CALUDE_derivative_of_sin_minus_cos_l1743_174336

theorem derivative_of_sin_minus_cos (α : ℝ) :
  let f : ℝ → ℝ := λ x => Real.sin α - Real.cos x
  (deriv f) α = Real.sin α :=
by
  sorry

end NUMINAMATH_CALUDE_derivative_of_sin_minus_cos_l1743_174336


namespace NUMINAMATH_CALUDE_salt_solution_percentage_l1743_174351

def is_valid_salt_solution (initial_salt_percent : ℝ) : Prop :=
  let replaced_volume : ℝ := 1/4
  let final_salt_percent : ℝ := 16
  let replacing_salt_percent : ℝ := 31
  (1 - replaced_volume) * initial_salt_percent + replaced_volume * replacing_salt_percent = final_salt_percent

theorem salt_solution_percentage :
  ∃ (x : ℝ), is_valid_salt_solution x ∧ x = 11 :=
sorry

end NUMINAMATH_CALUDE_salt_solution_percentage_l1743_174351


namespace NUMINAMATH_CALUDE_danny_bottle_caps_indeterminate_l1743_174381

/-- Represents Danny's collection of bottle caps and wrappers --/
structure Collection where
  bottle_caps : ℕ
  wrappers : ℕ

/-- The problem statement --/
theorem danny_bottle_caps_indeterminate 
  (initial : Collection) 
  (park_found : Collection)
  (final_wrappers : ℕ) :
  park_found.bottle_caps = 22 →
  park_found.wrappers = 30 →
  final_wrappers = initial.wrappers + park_found.wrappers →
  final_wrappers = 57 →
  ¬∃ (n : ℕ), ∀ (x : ℕ), initial.bottle_caps = n ∨ initial.bottle_caps ≠ x :=
by sorry

end NUMINAMATH_CALUDE_danny_bottle_caps_indeterminate_l1743_174381


namespace NUMINAMATH_CALUDE_zoo_animals_l1743_174367

theorem zoo_animals (sea_lions : ℕ) (penguins : ℕ) : 
  sea_lions = 48 →
  sea_lions * 11 = penguins * 4 →
  penguins > sea_lions →
  penguins - sea_lions = 84 := by
sorry

end NUMINAMATH_CALUDE_zoo_animals_l1743_174367


namespace NUMINAMATH_CALUDE_parallel_planes_from_perpendicular_lines_l1743_174393

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- State the theorem
theorem parallel_planes_from_perpendicular_lines 
  (m n : Line) (α β : Plane) :
  parallel m n → 
  perpendicular_line_plane m α → 
  perpendicular_line_plane n β → 
  parallel_planes α β :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_from_perpendicular_lines_l1743_174393


namespace NUMINAMATH_CALUDE_softball_team_size_l1743_174317

theorem softball_team_size :
  ∀ (men women : ℕ),
  women = men + 6 →
  (men : ℚ) / (women : ℚ) = 6 / 10 →
  men + women = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_softball_team_size_l1743_174317


namespace NUMINAMATH_CALUDE_defective_probability_l1743_174353

theorem defective_probability (total_output : ℝ) : 
  let machine_a_output := 0.40 * total_output
  let machine_b_output := 0.35 * total_output
  let machine_c_output := 0.25 * total_output
  let machine_a_defective_rate := 14 / 2000
  let machine_b_defective_rate := 9 / 1500
  let machine_c_defective_rate := 7 / 1000
  let total_defective := 
    machine_a_defective_rate * machine_a_output +
    machine_b_defective_rate * machine_b_output +
    machine_c_defective_rate * machine_c_output
  total_defective / total_output = 0.00665 := by
sorry

end NUMINAMATH_CALUDE_defective_probability_l1743_174353


namespace NUMINAMATH_CALUDE_average_age_decrease_l1743_174357

theorem average_age_decrease (N : ℕ) : 
  let original_avg : ℚ := 40
  let new_students : ℕ := 12
  let new_students_avg : ℚ := 34
  let total_original_age : ℚ := N * original_avg
  let total_new_age : ℚ := new_students * new_students_avg
  let new_total_students : ℕ := N + new_students
  let new_avg : ℚ := (total_original_age + total_new_age) / new_total_students
  original_avg - new_avg = 6 := by
sorry

end NUMINAMATH_CALUDE_average_age_decrease_l1743_174357


namespace NUMINAMATH_CALUDE_gasoline_cost_calculation_l1743_174314

/-- Represents the cost of gasoline per liter -/
def gasoline_cost : ℝ := sorry

/-- Represents the trip distance one way in kilometers -/
def one_way_distance : ℝ := 150

/-- Represents the cost of the first car rental option per day, excluding gasoline -/
def first_option_cost : ℝ := 50

/-- Represents the cost of the second car rental option per day, including gasoline -/
def second_option_cost : ℝ := 90

/-- Represents the distance a liter of gasoline can cover in kilometers -/
def km_per_liter : ℝ := 15

/-- Represents the amount saved by choosing the first option over the second option -/
def savings : ℝ := 22

theorem gasoline_cost_calculation : gasoline_cost = 3.4 := by
  sorry

end NUMINAMATH_CALUDE_gasoline_cost_calculation_l1743_174314


namespace NUMINAMATH_CALUDE_frustum_radius_l1743_174340

theorem frustum_radius (r : ℝ) :
  (r > 0) →
  (3 * (2 * π * r) = 2 * π * (3 * r)) →
  (π * (r + 3 * r) * 3 = 84 * π) →
  r = 7 := by sorry

end NUMINAMATH_CALUDE_frustum_radius_l1743_174340


namespace NUMINAMATH_CALUDE_range_of_expression_l1743_174372

theorem range_of_expression (x y z : ℝ) (h : x^2 + y^2 + z^2 = 4) :
  -(6 : ℝ) ≤ x + 2*y - 2*z ∧ x + 2*y - 2*z ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_range_of_expression_l1743_174372


namespace NUMINAMATH_CALUDE_two_satisfying_functions_l1743_174362

/-- A function satisfying the given functional equation -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  ∃ c : ℝ, ∀ x y : ℝ, f (x + y) * f (x - y) = (f x + f y)^2 - 4 * x^2 * f y + c

/-- The set of all functions satisfying the functional equation -/
def SatisfyingFunctions : Set (ℝ → ℝ) :=
  {f | SatisfyingFunction f}

/-- The constant zero function -/
def ZeroFunction : ℝ → ℝ := λ _ => 0

/-- The square function -/
def SquareFunction : ℝ → ℝ := λ x => x^2

theorem two_satisfying_functions :
  SatisfyingFunctions = {ZeroFunction, SquareFunction} := by sorry

end NUMINAMATH_CALUDE_two_satisfying_functions_l1743_174362


namespace NUMINAMATH_CALUDE_jerrys_cartridge_cost_l1743_174323

/-- The total cost of printer cartridges for Jerry -/
def total_cost (color_cartridge_cost : ℕ) (bw_cartridge_cost : ℕ) (color_cartridge_count : ℕ) (bw_cartridge_count : ℕ) : ℕ :=
  color_cartridge_cost * color_cartridge_count + bw_cartridge_cost * bw_cartridge_count

/-- Theorem: Jerry's total cost for printer cartridges is $123 -/
theorem jerrys_cartridge_cost :
  total_cost 32 27 3 1 = 123 := by
  sorry

end NUMINAMATH_CALUDE_jerrys_cartridge_cost_l1743_174323


namespace NUMINAMATH_CALUDE_not_prime_special_polynomial_l1743_174361

theorem not_prime_special_polynomial (n : ℕ+) : 
  ¬ Nat.Prime (n.val^2 - 2^2014 * 2014 * n.val + 4^2013 * (2014^2 - 1)) := by
  sorry

end NUMINAMATH_CALUDE_not_prime_special_polynomial_l1743_174361


namespace NUMINAMATH_CALUDE_smallest_cube_ending_544_l1743_174398

theorem smallest_cube_ending_544 :
  ∃ (n : ℕ), n > 0 ∧ n^3 % 1000 = 544 ∧ ∀ (m : ℕ), m > 0 ∧ m^3 % 1000 = 544 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_cube_ending_544_l1743_174398


namespace NUMINAMATH_CALUDE_odd_perfect_square_theorem_l1743_174368

/-- 
The sum of divisors function σ(n) is the sum of all positive divisors of n, including n itself.
-/
def sum_of_divisors (n : ℕ+) : ℕ := sorry

/-- 
A number is a perfect square if it is the product of an integer with itself.
-/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem odd_perfect_square_theorem (n : ℕ+) : 
  sum_of_divisors n = 2 * n.val + 1 → Odd n.val ∧ is_perfect_square n.val :=
sorry

end NUMINAMATH_CALUDE_odd_perfect_square_theorem_l1743_174368


namespace NUMINAMATH_CALUDE_largest_divisor_five_consecutive_integers_l1743_174385

theorem largest_divisor_five_consecutive_integers :
  ∀ n : ℤ, ∃ m : ℤ, m > 24 ∧ ¬(m ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) ∧
  ∀ k : ℤ, k ≤ 24 → k ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_five_consecutive_integers_l1743_174385


namespace NUMINAMATH_CALUDE_max_quotient_value_l1743_174394

theorem max_quotient_value (a b : ℝ) (ha : 300 ≤ a ∧ a ≤ 500) (hb : 800 ≤ b ∧ b ≤ 1600) :
  (∀ x y, 300 ≤ x ∧ x ≤ 500 → 800 ≤ y ∧ y ≤ 1600 → 2 * y / x ≤ 32 / 3) ∧
  (∃ x y, 300 ≤ x ∧ x ≤ 500 ∧ 800 ≤ y ∧ y ≤ 1600 ∧ 2 * y / x = 32 / 3) :=
by sorry

end NUMINAMATH_CALUDE_max_quotient_value_l1743_174394


namespace NUMINAMATH_CALUDE_cards_rick_keeps_rick_keeps_fifteen_cards_l1743_174334

/-- The number of cards Rick keeps for himself given the initial number of cards and the distribution to others. -/
theorem cards_rick_keeps (initial_cards : ℕ) (cards_to_miguel : ℕ) (num_friends : ℕ) (cards_per_friend : ℕ) (num_sisters : ℕ) (cards_per_sister : ℕ) : ℕ :=
  initial_cards - cards_to_miguel - (num_friends * cards_per_friend) - (num_sisters * cards_per_sister)

/-- Proof that Rick keeps 15 cards for himself -/
theorem rick_keeps_fifteen_cards :
  cards_rick_keeps 130 13 8 12 2 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_cards_rick_keeps_rick_keeps_fifteen_cards_l1743_174334


namespace NUMINAMATH_CALUDE_hamster_lifespan_l1743_174348

theorem hamster_lifespan (fish_lifespan dog_lifespan hamster_lifespan : ℝ) 
  (h1 : fish_lifespan = dog_lifespan + 2)
  (h2 : dog_lifespan = 4 * hamster_lifespan)
  (h3 : fish_lifespan = 12) : 
  hamster_lifespan = 2.5 := by
sorry

end NUMINAMATH_CALUDE_hamster_lifespan_l1743_174348


namespace NUMINAMATH_CALUDE_abs_x_minus_one_lt_two_sufficient_not_necessary_for_x_lt_three_negation_true_implies_converse_true_l1743_174315

-- Statement 1
theorem abs_x_minus_one_lt_two_sufficient_not_necessary_for_x_lt_three :
  (∀ x : ℝ, |x - 1| < 2 → x < 3) ∧
  ¬(∀ x : ℝ, x < 3 → |x - 1| < 2) :=
sorry

-- Statement 2
theorem negation_true_implies_converse_true (P Q : Prop) :
  (¬(P → Q) → (Q → P)) :=
sorry

end NUMINAMATH_CALUDE_abs_x_minus_one_lt_two_sufficient_not_necessary_for_x_lt_three_negation_true_implies_converse_true_l1743_174315


namespace NUMINAMATH_CALUDE_equilateral_triangle_side_length_l1743_174327

-- Define the cubic equation
def cubic_equation (x : ℝ) : Prop := x^3 - 9*x^2 + 10*x + 5 = 0

-- Define the property of distinct roots
def distinct_roots (a b c : ℝ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

-- Theorem statement
theorem equilateral_triangle_side_length
  (a b c : ℝ)
  (ha : cubic_equation a)
  (hb : cubic_equation b)
  (hc : cubic_equation c)
  (hdistinct : distinct_roots a b c) :
  ∃ (side_length : ℝ), side_length = 2 * Real.sqrt 17 ∧
  side_length^2 = (a - b)^2 + (b - c)^2 + (c - a)^2 :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_side_length_l1743_174327


namespace NUMINAMATH_CALUDE_hot_sauce_duration_l1743_174359

-- Define the size of a quart in ounces
def quart_size : ℝ := 32

-- Define the size of the hot sauce jar
def jar_size : ℝ := quart_size - 2

-- Define the size of each serving
def serving_size : ℝ := 0.5

-- Define the number of servings used daily
def daily_servings : ℕ := 3

-- Define the daily consumption
def daily_consumption : ℝ := serving_size * daily_servings

-- Theorem to prove
theorem hot_sauce_duration : 
  (jar_size / daily_consumption : ℝ) = 20 := by sorry

end NUMINAMATH_CALUDE_hot_sauce_duration_l1743_174359


namespace NUMINAMATH_CALUDE_sequence_inequality_l1743_174354

def is_increasing (a : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

def not_in_sequence (a : ℕ → ℕ) (x : ℕ) : Prop :=
  ∀ n : ℕ, a n ≠ x

def representable (a : ℕ → ℕ) : Prop :=
  ∀ x : ℕ, not_in_sequence a x → ∃ k : ℕ, x = a k + 2 * k

theorem sequence_inequality (a : ℕ → ℕ) 
  (h1 : is_increasing a) 
  (h2 : representable a) : 
  ∀ k : ℕ, (a k : ℝ) < Real.sqrt (2 * k) := by
  sorry

end NUMINAMATH_CALUDE_sequence_inequality_l1743_174354


namespace NUMINAMATH_CALUDE_log_equation_solution_l1743_174321

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log x^2 / Real.log 4 + Real.log x / Real.log (1/4) = 7 → x = 16384 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1743_174321


namespace NUMINAMATH_CALUDE_snowboard_discount_l1743_174337

theorem snowboard_discount (original_price : ℝ) (friday_discount : ℝ) (monday_discount : ℝ) :
  original_price = 100 ∧ 
  friday_discount = 0.5 ∧ 
  monday_discount = 0.3 →
  (1 - monday_discount) * ((1 - friday_discount) * original_price) = 35 := by
sorry

end NUMINAMATH_CALUDE_snowboard_discount_l1743_174337


namespace NUMINAMATH_CALUDE_no_abc_divisible_by_9_l1743_174386

theorem no_abc_divisible_by_9 :
  ∀ (a b c : ℤ), ∃ (x : ℤ), ¬ (9 ∣ ((x + a) * (x + b) * (x + c) - x^3 - 1)) :=
by sorry

end NUMINAMATH_CALUDE_no_abc_divisible_by_9_l1743_174386


namespace NUMINAMATH_CALUDE_min_value_theorem_l1743_174375

theorem min_value_theorem (x y : ℝ) (h1 : x * y + 3 * x = 3) (h2 : 0 < x) (h3 : x < 1/2) :
  ∀ z, z = (3 / x) + (1 / (y - 3)) → z ≥ 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1743_174375


namespace NUMINAMATH_CALUDE_special_line_properties_l1743_174344

/-- A line passing through (5, 2) with x-intercept twice the y-intercept -/
def special_line (x y : ℝ) : Prop :=
  x + 2 * y - 9 = 0

theorem special_line_properties :
  (special_line 5 2) ∧ 
  (∃ (a : ℝ), a ≠ 0 ∧ special_line (2*a) 0 ∧ special_line 0 a) :=
by sorry

end NUMINAMATH_CALUDE_special_line_properties_l1743_174344


namespace NUMINAMATH_CALUDE_prime_has_property_P_infinitely_many_composite_with_property_P_l1743_174333

-- Define property P
def has_property_P (n : ℕ) : Prop :=
  ∀ a : ℕ, a > 0 → (n ∣ a^n - 1) → (n^2 ∣ a^n - 1)

-- Theorem 1: Every prime number has property P
theorem prime_has_property_P (p : ℕ) (hp : Prime p) : has_property_P p := by
  sorry

-- Theorem 2: There are infinitely many composite numbers with property P
theorem infinitely_many_composite_with_property_P :
  ∀ n : ℕ, ∃ m : ℕ, m > n ∧ ¬Prime m ∧ has_property_P m := by
  sorry

end NUMINAMATH_CALUDE_prime_has_property_P_infinitely_many_composite_with_property_P_l1743_174333


namespace NUMINAMATH_CALUDE_constant_term_expansion_l1743_174371

theorem constant_term_expansion (x : ℝ) : 
  ∃ (f : ℝ → ℝ), (∀ x ≠ 0, f x = (1/x^2 - 2*x)^6) ∧ 
  (∃ c : ℝ, ∀ ε > 0, ∃ δ > 0, ∀ x, |x| < δ → |f x - c| < ε) ∧
  (∃! c : ℝ, ∀ ε > 0, ∃ δ > 0, ∀ x, |x| < δ → |f x - c| < ε) ∧
  c = 240 :=
sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l1743_174371


namespace NUMINAMATH_CALUDE_fruit_price_adjustment_l1743_174349

/-- Represents the problem of adjusting fruit quantities to achieve a desired average price --/
theorem fruit_price_adjustment
  (apple_price : ℚ)
  (orange_price : ℚ)
  (total_fruits : ℕ)
  (initial_avg_price : ℚ)
  (desired_avg_price : ℚ)
  (h1 : apple_price = 40/100)
  (h2 : orange_price = 60/100)
  (h3 : total_fruits = 10)
  (h4 : initial_avg_price = 52/100)
  (h5 : desired_avg_price = 44/100)
  : ∃ (oranges_to_remove : ℕ),
    oranges_to_remove = 5 ∧
    ∃ (apples : ℕ) (oranges : ℕ),
      apples + oranges = total_fruits ∧
      (apple_price * apples + orange_price * oranges) / total_fruits = initial_avg_price ∧
      (apple_price * apples + orange_price * (oranges - oranges_to_remove)) / (total_fruits - oranges_to_remove) = desired_avg_price :=
by
  sorry

end NUMINAMATH_CALUDE_fruit_price_adjustment_l1743_174349


namespace NUMINAMATH_CALUDE_G_is_leftmost_l1743_174360

/-- Represents a square with four integer labels -/
structure Square where
  name : Char
  w : Int
  x : Int
  y : Int
  z : Int

/-- The set of all squares -/
def squares : Finset Square := sorry

/-- Predicate to check if a square is leftmost -/
def is_leftmost (s : Square) : Prop := sorry

/-- The squares are arranged in a row without rotating or reflecting -/
axiom squares_in_row : sorry

/-- All squares are distinct -/
axiom squares_distinct : sorry

/-- The specific squares given in the problem -/
def F : Square := ⟨'F', 5, 1, 7, 9⟩
def G : Square := ⟨'G', 1, 0, 4, 6⟩
def H : Square := ⟨'H', 4, 8, 6, 2⟩
def I : Square := ⟨'I', 8, 5, 3, 7⟩
def J : Square := ⟨'J', 9, 2, 8, 0⟩

/-- All given squares are in the set of squares -/
axiom all_squares_in_set : F ∈ squares ∧ G ∈ squares ∧ H ∈ squares ∧ I ∈ squares ∧ J ∈ squares

/-- Theorem: Square G is the leftmost square -/
theorem G_is_leftmost : is_leftmost G := by sorry

end NUMINAMATH_CALUDE_G_is_leftmost_l1743_174360


namespace NUMINAMATH_CALUDE_total_songs_in_june_l1743_174395

def june_days : ℕ := 30
def weekend_days : ℕ := 8
def holiday_days : ℕ := 1
def vivian_songs_per_day : ℕ := 10
def clara_songs_per_day : ℕ := vivian_songs_per_day - 2
def lucas_songs_per_day : ℕ := vivian_songs_per_day + 5

theorem total_songs_in_june :
  let playing_days : ℕ := june_days - weekend_days - holiday_days
  let vivian_total : ℕ := playing_days * vivian_songs_per_day
  let clara_total : ℕ := playing_days * clara_songs_per_day
  let lucas_total : ℕ := playing_days * lucas_songs_per_day
  vivian_total + clara_total + lucas_total = 693 := by
  sorry

end NUMINAMATH_CALUDE_total_songs_in_june_l1743_174395


namespace NUMINAMATH_CALUDE_dan_found_no_money_l1743_174332

/-- The amount of money Dan spent on a snake toy -/
def snake_toy_cost : ℚ := 11.76

/-- The amount of money Dan spent on a cage -/
def cage_cost : ℚ := 14.54

/-- The total cost of Dan's purchases -/
def total_cost : ℚ := 26.3

/-- The amount of money Dan found on the ground -/
def money_found : ℚ := total_cost - (snake_toy_cost + cage_cost)

theorem dan_found_no_money : money_found = 0 := by sorry

end NUMINAMATH_CALUDE_dan_found_no_money_l1743_174332


namespace NUMINAMATH_CALUDE_max_value_of_e_l1743_174391

def b (n : ℕ) : ℤ := (5^n - 1) / 4

def e (n : ℕ) : ℕ := Nat.gcd (Int.natAbs (b n)) (Int.natAbs (b (n + 1)))

theorem max_value_of_e (n : ℕ) : e n = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_e_l1743_174391


namespace NUMINAMATH_CALUDE_candied_apples_count_l1743_174397

/-- The number of candied apples that were made -/
def num_apples : ℕ := 15

/-- The price of each candied apple in dollars -/
def apple_price : ℚ := 2

/-- The number of candied grapes -/
def num_grapes : ℕ := 12

/-- The price of each candied grape in dollars -/
def grape_price : ℚ := 3/2

/-- The total earnings from selling all items in dollars -/
def total_earnings : ℚ := 48

theorem candied_apples_count :
  num_apples * apple_price + num_grapes * grape_price = total_earnings :=
sorry

end NUMINAMATH_CALUDE_candied_apples_count_l1743_174397


namespace NUMINAMATH_CALUDE_apples_left_in_basket_l1743_174310

/-- Given a basket of apples, calculate the number of apples left after Ricki and Samson remove some. -/
theorem apples_left_in_basket 
  (initial_apples : ℕ) 
  (ricki_removes : ℕ) 
  (h1 : initial_apples = 184) 
  (h2 : ricki_removes = 34) :
  initial_apples - (ricki_removes + 3 * ricki_removes) = 48 := by
  sorry

#check apples_left_in_basket

end NUMINAMATH_CALUDE_apples_left_in_basket_l1743_174310


namespace NUMINAMATH_CALUDE_subtract_like_terms_l1743_174341

theorem subtract_like_terms (x y : ℝ) : 5 * x * y - 4 * x * y = x * y := by
  sorry

end NUMINAMATH_CALUDE_subtract_like_terms_l1743_174341


namespace NUMINAMATH_CALUDE_square_sum_problem_l1743_174335

theorem square_sum_problem (x y : ℕ+) 
  (h1 : x.val * y.val + x.val + y.val = 35)
  (h2 : x.val * y.val * (x.val + y.val) = 360) :
  x.val^2 + y.val^2 = 185 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_problem_l1743_174335


namespace NUMINAMATH_CALUDE_polynomial_division_l1743_174324

theorem polynomial_division (x : ℝ) :
  x^6 + 3 = (x - 2) * (x^5 + 2*x^4 + 4*x^3 + 8*x^2 + 16*x + 32) + 67 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_l1743_174324


namespace NUMINAMATH_CALUDE_evaluate_nested_square_roots_l1743_174312

theorem evaluate_nested_square_roots : 
  Real.sqrt (64 * Real.sqrt (32 * Real.sqrt (4^3))) = 64 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_nested_square_roots_l1743_174312


namespace NUMINAMATH_CALUDE_unacceptable_weight_l1743_174369

def acceptable_range (x : ℝ) : Prop := 49.7 ≤ x ∧ x ≤ 50.3

theorem unacceptable_weight : ¬(acceptable_range 49.6) := by
  sorry

end NUMINAMATH_CALUDE_unacceptable_weight_l1743_174369


namespace NUMINAMATH_CALUDE_sum_of_zeros_less_than_two_ln_a_l1743_174374

/-- Given a function f(x) = e^x - ax + a, where a ∈ ℝ, if f has two zeros, their sum is less than 2 ln a -/
theorem sum_of_zeros_less_than_two_ln_a (a : ℝ) (x₁ x₂ : ℝ) :
  let f := fun x => Real.exp x - a * x + a
  (f x₁ = 0) → (f x₂ = 0) → (x₁ + x₂ < 2 * Real.log a) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_zeros_less_than_two_ln_a_l1743_174374


namespace NUMINAMATH_CALUDE_tim_younger_than_jenny_l1743_174373

/-- Given the ages of Tim, Rommel, and Jenny, prove that Tim is 12 years younger than Jenny. -/
theorem tim_younger_than_jenny (tim_age rommel_age jenny_age : ℕ) : 
  tim_age = 5 →
  rommel_age = 3 * tim_age →
  jenny_age = rommel_age + 2 →
  jenny_age - tim_age = 12 := by
sorry

end NUMINAMATH_CALUDE_tim_younger_than_jenny_l1743_174373
