import Mathlib

namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3129_312956

theorem right_triangle_hypotenuse (a b c : ℝ) (h1 : a = 90) (h2 : b = 120) 
  (h3 : c^2 = a^2 + b^2) : c = 150 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3129_312956


namespace NUMINAMATH_CALUDE_daniel_earnings_l3129_312959

-- Define the delivery schedule and prices
def monday_fabric : ℕ := 20
def monday_yarn : ℕ := 15
def tuesday_fabric : ℕ := 2 * monday_fabric
def tuesday_yarn : ℕ := monday_yarn + 10
def wednesday_fabric : ℕ := tuesday_fabric / 4
def wednesday_yarn : ℕ := tuesday_yarn / 2 + 1  -- Rounded up

def fabric_price : ℕ := 2
def yarn_price : ℕ := 3

-- Calculate total yards of fabric and yarn
def total_fabric : ℕ := monday_fabric + tuesday_fabric + wednesday_fabric
def total_yarn : ℕ := monday_yarn + tuesday_yarn + wednesday_yarn

-- Calculate total earnings
def total_earnings : ℕ := fabric_price * total_fabric + yarn_price * total_yarn

-- Theorem to prove
theorem daniel_earnings : total_earnings = 299 := by
  sorry

end NUMINAMATH_CALUDE_daniel_earnings_l3129_312959


namespace NUMINAMATH_CALUDE_beavers_working_on_home_l3129_312920

/-- The number of beavers initially working on their home -/
def initial_beavers : ℕ := 2

/-- The number of beavers that went for a swim -/
def swimming_beavers : ℕ := 1

/-- The number of beavers still working on their home -/
def remaining_beavers : ℕ := initial_beavers - swimming_beavers

theorem beavers_working_on_home :
  remaining_beavers = 1 :=
by sorry

end NUMINAMATH_CALUDE_beavers_working_on_home_l3129_312920


namespace NUMINAMATH_CALUDE_base_five_3214_equals_434_l3129_312967

def base_five_to_ten (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

theorem base_five_3214_equals_434 :
  base_five_to_ten [4, 1, 2, 3] = 434 := by
  sorry

end NUMINAMATH_CALUDE_base_five_3214_equals_434_l3129_312967


namespace NUMINAMATH_CALUDE_real_solutions_condition_l3129_312913

theorem real_solutions_condition (a : ℝ) :
  (∃ x y : ℝ, x + y^2 = a ∧ y + x^2 = a) ↔ a ≥ 3/4 := by
  sorry

end NUMINAMATH_CALUDE_real_solutions_condition_l3129_312913


namespace NUMINAMATH_CALUDE_lisa_age_l3129_312988

theorem lisa_age :
  ∀ (L N : ℕ),
  L = N + 8 →
  L - 2 = 3 * (N - 2) →
  L = 14 :=
by sorry

end NUMINAMATH_CALUDE_lisa_age_l3129_312988


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l3129_312963

theorem fraction_sum_equality (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) 
  (h_sum : a / (b - c) + b / (c - a) + c / (a - b) = 1) :
  a / (b - c)^2 + b / (c - a)^2 + c / (a - b)^2 = 
  1 / (b - c)^2 + 1 / (c - a)^2 + 1 / (a - b)^2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l3129_312963


namespace NUMINAMATH_CALUDE_red_chips_probability_l3129_312925

theorem red_chips_probability (total_chips : ℕ) (red_chips : ℕ) (green_chips : ℕ) 
  (h1 : total_chips = red_chips + green_chips)
  (h2 : red_chips = 5)
  (h3 : green_chips = 3) :
  (Nat.choose (total_chips - 1) (green_chips - 1) : ℚ) / (Nat.choose total_chips green_chips) = 3/8 :=
sorry

end NUMINAMATH_CALUDE_red_chips_probability_l3129_312925


namespace NUMINAMATH_CALUDE_find_number_l3129_312910

theorem find_number : ∃! x : ℤ, (x + 12) / 4 = 12 ∧ (x + 12) % 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l3129_312910


namespace NUMINAMATH_CALUDE_bicycle_speed_calculation_l3129_312961

theorem bicycle_speed_calculation (distance : ℝ) (speed_difference : ℝ) (time_ratio : ℝ) :
  distance = 10 ∧ 
  speed_difference = 45 ∧ 
  time_ratio = 4 →
  ∃ x : ℝ, x = 15 ∧ 
    distance / x = time_ratio * (distance / (x + speed_difference)) :=
by sorry

end NUMINAMATH_CALUDE_bicycle_speed_calculation_l3129_312961


namespace NUMINAMATH_CALUDE_infinite_series_sum_l3129_312908

theorem infinite_series_sum : 
  (∑' k : ℕ, (k : ℝ) / (3 : ℝ) ^ k) = (1 : ℝ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l3129_312908


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l3129_312916

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 5| = 3 * x - 2 :=
by
  -- The unique solution is x = 7/4
  use 7/4
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l3129_312916


namespace NUMINAMATH_CALUDE_unique_solution_condition_l3129_312974

/-- The equation (x + 3) / (mx - 2) = x + 1 has exactly one solution if and only if m = -8 ± 2√15 -/
theorem unique_solution_condition (m : ℝ) : 
  (∃! x : ℝ, (x + 3) / (m * x - 2) = x + 1) ↔ 
  (m = -8 + 2 * Real.sqrt 15 ∨ m = -8 - 2 * Real.sqrt 15) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l3129_312974


namespace NUMINAMATH_CALUDE_sqrt_x_minus_3_real_l3129_312999

theorem sqrt_x_minus_3_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 3) ↔ x ≥ 3 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_3_real_l3129_312999


namespace NUMINAMATH_CALUDE_range_of_a_l3129_312909

theorem range_of_a (a b c d : ℝ) 
  (sum_condition : a + b + c + d = 3) 
  (square_condition : a^2 + 2*b^2 + 3*c^2 + 6*d^2 = 5) : 
  1 ≤ a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l3129_312909


namespace NUMINAMATH_CALUDE_f_sum_symmetric_l3129_312958

/-- A function f(x) = ax^4 + bx^2 + 5 where a and b are real constants -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^4 + b * x^2 + 5

/-- Theorem: If f(20) = 3, then f(20) + f(-20) = 6 -/
theorem f_sum_symmetric (a b : ℝ) (h : f a b 20 = 3) : f a b 20 + f a b (-20) = 6 := by
  sorry

end NUMINAMATH_CALUDE_f_sum_symmetric_l3129_312958


namespace NUMINAMATH_CALUDE_mango_ratio_proof_l3129_312994

/-- Proves that the ratio of mangoes sold at the market to total mangoes harvested is 1:2 -/
theorem mango_ratio_proof (total_mangoes : ℕ) (num_neighbors : ℕ) (mangoes_per_neighbor : ℕ)
  (h1 : total_mangoes = 560)
  (h2 : num_neighbors = 8)
  (h3 : mangoes_per_neighbor = 35) :
  (total_mangoes - num_neighbors * mangoes_per_neighbor) / total_mangoes = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_mango_ratio_proof_l3129_312994


namespace NUMINAMATH_CALUDE_min_a_value_l3129_312986

theorem min_a_value (a : ℝ) : 
  (∀ x ∈ Set.Ioo 0 (1/2), x^2 + a*x + 1 ≥ 0) → a ≥ -5/2 := by
  sorry

end NUMINAMATH_CALUDE_min_a_value_l3129_312986


namespace NUMINAMATH_CALUDE_sin_390_degrees_l3129_312970

theorem sin_390_degrees (h1 : ∀ θ, Real.sin (θ + 2 * Real.pi) = Real.sin θ) 
                        (h2 : Real.sin (Real.pi / 6) = 1 / 2) : 
  Real.sin (13 * Real.pi / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_390_degrees_l3129_312970


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l3129_312901

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 * b.2 = k * a.2 * b.1

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (1, -3)
  let b : ℝ × ℝ := (m, 6)
  parallel a b → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l3129_312901


namespace NUMINAMATH_CALUDE_cube_units_digits_eq_all_digits_l3129_312932

/-- The set of all single digits -/
def AllDigits : Set Nat := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

/-- The set of units digits of integral perfect cubes -/
def CubeUnitsDigits : Set Nat :=
  {d | ∃ n : Nat, d = (n^3) % 10}

/-- Theorem: The set of units digits of integral perfect cubes
    is equal to the set of all single digits -/
theorem cube_units_digits_eq_all_digits :
  CubeUnitsDigits = AllDigits := by sorry

end NUMINAMATH_CALUDE_cube_units_digits_eq_all_digits_l3129_312932


namespace NUMINAMATH_CALUDE_haley_tv_watching_time_l3129_312921

theorem haley_tv_watching_time (saturday_hours sunday_hours : ℕ) 
  (h1 : saturday_hours = 6) 
  (h2 : sunday_hours = 3) : 
  saturday_hours + sunday_hours = 9 := by
sorry

end NUMINAMATH_CALUDE_haley_tv_watching_time_l3129_312921


namespace NUMINAMATH_CALUDE_inequality_solution_l3129_312936

theorem inequality_solution (x : ℝ) :
  x ≠ 4 →
  (x * (x + 1) / (x - 4)^2 ≥ 15 ↔ x ∈ Set.Iic 3 ∪ Set.Ioo (40/7) 4 ∪ Set.Ioi 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3129_312936


namespace NUMINAMATH_CALUDE_non_shaded_perimeter_l3129_312976

/-- Given a rectangle with dimensions 15 inches by 10 inches and area 150 square inches,
    containing a shaded rectangle with area 110 square inches,
    the perimeter of the non-shaded region is 26 inches. -/
theorem non_shaded_perimeter (large_width large_height : ℝ)
                              (large_area shaded_area : ℝ)
                              (non_shaded_width non_shaded_height : ℝ) :
  large_width = 15 →
  large_height = 10 →
  large_area = 150 →
  shaded_area = 110 →
  large_area = large_width * large_height →
  non_shaded_width * non_shaded_height = large_area - shaded_area →
  non_shaded_width ≤ large_width →
  non_shaded_height ≤ large_height →
  2 * (non_shaded_width + non_shaded_height) = 26 :=
by sorry

end NUMINAMATH_CALUDE_non_shaded_perimeter_l3129_312976


namespace NUMINAMATH_CALUDE_other_side_length_l3129_312996

/-- Represents a right triangle with given side lengths -/
structure RightTriangle where
  hypotenuse : ℝ
  side1 : ℝ
  side2 : ℝ
  hypotenuse_positive : hypotenuse > 0
  side1_positive : side1 > 0
  side2_positive : side2 > 0
  pythagorean : hypotenuse^2 = side1^2 + side2^2

/-- The length of the other side in a right triangle with hypotenuse 10 and one side 6 is 8 -/
theorem other_side_length (t : RightTriangle) (h1 : t.hypotenuse = 10) (h2 : t.side1 = 6) :
  t.side2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_other_side_length_l3129_312996


namespace NUMINAMATH_CALUDE_min_pieces_for_special_l3129_312906

/-- Represents a piece of the pie -/
inductive PieceType
| Empty
| Fish
| Sausage
| Special

/-- Represents the 8x8 pie grid -/
def Pie := Fin 8 → Fin 8 → PieceType

/-- Checks if a 6x6 square in the pie has at least 2 fish pieces -/
def has_two_fish (p : Pie) (i j : Fin 8) : Prop :=
  ∃ (i1 j1 i2 j2 : Fin 8),
    i1 < i + 6 ∧ j1 < j + 6 ∧ i2 < i + 6 ∧ j2 < j + 6 ∧
    (i1 ≠ i2 ∨ j1 ≠ j2) ∧
    p i1 j1 = PieceType.Fish ∧ p i2 j2 = PieceType.Fish

/-- Checks if a 3x3 square in the pie has at most 1 sausage piece -/
def at_most_one_sausage (p : Pie) (i j : Fin 8) : Prop :=
  ∀ (i1 j1 i2 j2 : Fin 8),
    i1 < i + 3 → j1 < j + 3 → i2 < i + 3 → j2 < j + 3 →
    p i1 j1 = PieceType.Sausage → p i2 j2 = PieceType.Sausage →
    i1 = i2 ∧ j1 = j2

/-- Defines a valid pie configuration -/
def valid_pie (p : Pie) : Prop :=
  (∃ (i1 j1 i2 j2 i3 j3 : Fin 8),
     p i1 j1 = PieceType.Fish ∧ p i2 j2 = PieceType.Fish ∧ p i3 j3 = PieceType.Fish ∧
     (i1 ≠ i2 ∨ j1 ≠ j2) ∧ (i1 ≠ i3 ∨ j1 ≠ j3) ∧ (i2 ≠ i3 ∨ j2 ≠ j3)) ∧
  (∃ (i1 j1 i2 j2 : Fin 8),
     p i1 j1 = PieceType.Sausage ∧ p i2 j2 = PieceType.Sausage ∧
     (i1 ≠ i2 ∨ j1 ≠ j2)) ∧
  (∃! (i j : Fin 8), p i j = PieceType.Special) ∧
  (∀ (i j : Fin 8), has_two_fish p i j) ∧
  (∀ (i j : Fin 8), at_most_one_sausage p i j)

/-- Theorem: The minimum number of pieces to guarantee getting the special piece is 5 -/
theorem min_pieces_for_special (p : Pie) (h : valid_pie p) :
  ∀ (s : Finset (Fin 8 × Fin 8)),
    s.card < 5 → ∃ (i j : Fin 8), p i j = PieceType.Special ∧ (i, j) ∉ s :=
sorry

end NUMINAMATH_CALUDE_min_pieces_for_special_l3129_312906


namespace NUMINAMATH_CALUDE_ariel_age_quadruples_l3129_312937

/-- Proves that it takes 15 years for Ariel to be four times her current age -/
theorem ariel_age_quadruples (current_age : ℕ) (years_passed : ℕ) : current_age = 5 →
  current_age + years_passed = 4 * current_age →
  years_passed = 15 := by
  sorry

end NUMINAMATH_CALUDE_ariel_age_quadruples_l3129_312937


namespace NUMINAMATH_CALUDE_number_problem_l3129_312982

theorem number_problem : ∃ x : ℝ, x = 580 ∧ 0.2 * x = 0.3 * 120 + 80 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3129_312982


namespace NUMINAMATH_CALUDE_square_difference_fraction_l3129_312918

theorem square_difference_fraction (x y : ℚ) 
  (sum_eq : x + y = 8/15) 
  (diff_eq : x - y = 1/35) : 
  x^2 - y^2 = 1/75 := by
sorry

end NUMINAMATH_CALUDE_square_difference_fraction_l3129_312918


namespace NUMINAMATH_CALUDE_arctan_less_arcsin_iff_l3129_312983

theorem arctan_less_arcsin_iff (x : ℝ) : Real.arctan x < Real.arcsin x ↔ -1 < x ∧ x ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_arctan_less_arcsin_iff_l3129_312983


namespace NUMINAMATH_CALUDE_exists_nonnegative_product_polynomial_l3129_312940

theorem exists_nonnegative_product_polynomial (f : Polynomial ℝ) 
  (h_no_nonneg_root : ∀ x : ℝ, x ≥ 0 → f.eval x ≠ 0) :
  ∃ h : Polynomial ℝ, ∀ i : ℕ, (f * h).coeff i ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_nonnegative_product_polynomial_l3129_312940


namespace NUMINAMATH_CALUDE_smallest_a_value_l3129_312907

theorem smallest_a_value (a b : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0)
  (h3 : ∀ (x : ℤ), Real.sin (a * (x : ℝ) + b) = Real.sin (17 * (x : ℝ))) :
  a ≥ 17 ∧ ∃ (a₀ : ℝ), a₀ ≥ 0 ∧ a₀ < 17 ∧ 
    (∀ (x : ℤ), Real.sin (a₀ * (x : ℝ) + b) = Real.sin (17 * (x : ℝ))) → False :=
by sorry

end NUMINAMATH_CALUDE_smallest_a_value_l3129_312907


namespace NUMINAMATH_CALUDE_sum_of_multiples_l3129_312904

def smallest_two_digit_multiple_of_5 : ℕ := 10

def smallest_three_digit_multiple_of_7 : ℕ := 105

theorem sum_of_multiples : 
  smallest_two_digit_multiple_of_5 + smallest_three_digit_multiple_of_7 = 115 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_multiples_l3129_312904


namespace NUMINAMATH_CALUDE_g_properties_l3129_312955

def f (n : ℕ) : ℕ := (Nat.factorial n)^2

def g (x : ℕ+) : ℚ := (f (x + 1) : ℚ) / (f x : ℚ)

theorem g_properties :
  (g 1 = 4) ∧
  (g 2 = 9) ∧
  (g 3 = 16) ∧
  (∀ ε > 0, ∃ N : ℕ+, ∀ x ≥ N, g x > ε) :=
sorry

end NUMINAMATH_CALUDE_g_properties_l3129_312955


namespace NUMINAMATH_CALUDE_certain_number_proof_l3129_312968

theorem certain_number_proof : ∃ x : ℝ, 45 * 12 = 0.60 * x ∧ x = 900 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3129_312968


namespace NUMINAMATH_CALUDE_distribute_five_books_three_students_l3129_312942

/-- The number of ways to distribute n different books among k students,
    with each student receiving at least one book -/
def distribute_books (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 different books among 3 students,
    with each student receiving at least one book, is 150 -/
theorem distribute_five_books_three_students :
  distribute_books 5 3 = 150 := by sorry

end NUMINAMATH_CALUDE_distribute_five_books_three_students_l3129_312942


namespace NUMINAMATH_CALUDE_cricket_run_rate_theorem_l3129_312917

/-- Represents a cricket game situation -/
structure CricketGame where
  totalOvers : ℕ
  firstPeriodOvers : ℕ
  firstPeriodRunRate : ℚ
  targetRuns : ℕ

/-- Calculates the required run rate for the remaining overs -/
def requiredRunRate (game : CricketGame) : ℚ :=
  let remainingOvers := game.totalOvers - game.firstPeriodOvers
  let runsScored := game.firstPeriodRunRate * game.firstPeriodOvers
  let runsNeeded := game.targetRuns - runsScored
  runsNeeded / remainingOvers

/-- Theorem stating the required run rate for the given game situation -/
theorem cricket_run_rate_theorem (game : CricketGame) 
  (h1 : game.totalOvers = 50)
  (h2 : game.firstPeriodOvers = 10)
  (h3 : game.firstPeriodRunRate = 21/5)  -- 4.2 as a fraction
  (h4 : game.targetRuns = 282) :
  requiredRunRate game = 6 := by
  sorry

end NUMINAMATH_CALUDE_cricket_run_rate_theorem_l3129_312917


namespace NUMINAMATH_CALUDE_complex_expression_equals_five_l3129_312960

theorem complex_expression_equals_five :
  (3 * Real.sqrt 12 - 2 * Real.sqrt (1/3) + Real.sqrt 48) / (2 * Real.sqrt 3) + (Real.sqrt (1/3))^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equals_five_l3129_312960


namespace NUMINAMATH_CALUDE_fish_purchase_total_l3129_312975

theorem fish_purchase_total (yesterday_fish : ℕ) (yesterday_cost : ℕ) (today_extra_cost : ℕ) : 
  yesterday_fish = 10 →
  yesterday_cost = 3000 →
  today_extra_cost = 6000 →
  ∃ (today_fish : ℕ), 
    (yesterday_fish + today_fish = 40 ∧ 
     yesterday_cost + today_extra_cost = (yesterday_cost / yesterday_fish) * (yesterday_fish + today_fish)) := by
  sorry

#check fish_purchase_total

end NUMINAMATH_CALUDE_fish_purchase_total_l3129_312975


namespace NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l3129_312992

theorem regular_polygon_interior_angle_sum 
  (n : ℕ) 
  (h_exterior : (360 : ℝ) / n = 45) : 
  (n - 2 : ℝ) * 180 = 1080 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l3129_312992


namespace NUMINAMATH_CALUDE_inscribed_circles_area_l3129_312951

theorem inscribed_circles_area (R : ℝ) (d : ℝ) : 
  R = 10 ∧ d = 6 → 
  let h := R - d / 2
  let r := R - d / 2
  2 * Real.pi * r^2 = 98 * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_inscribed_circles_area_l3129_312951


namespace NUMINAMATH_CALUDE_tree_height_problem_l3129_312981

theorem tree_height_problem (h₁ h₂ : ℝ) : 
  h₁ = h₂ + 20 →  -- One tree is 20 feet taller than the other
  h₂ / h₁ = 5 / 7 →  -- The heights are in the ratio 5:7
  h₁ = 70 :=  -- The height of the taller tree is 70 feet
by sorry

end NUMINAMATH_CALUDE_tree_height_problem_l3129_312981


namespace NUMINAMATH_CALUDE_inequality_problem_l3129_312971

theorem inequality_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y ≤ 4) :
  1 / (x * y) ≥ 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_problem_l3129_312971


namespace NUMINAMATH_CALUDE_fry_all_cutlets_in_15_minutes_l3129_312950

/-- Represents a cutlet that needs to be fried -/
structure Cutlet where
  sides_fried : Fin 2 → Bool
  deriving Repr

/-- Represents the state of frying cutlets -/
structure FryingState where
  time : ℕ
  cutlets : Fin 3 → Cutlet
  pan : Fin 2 → Option (Fin 3)
  deriving Repr

/-- Checks if all cutlets are fully fried -/
def all_fried (state : FryingState) : Prop :=
  ∀ i : Fin 3, (state.cutlets i).sides_fried 0 ∧ (state.cutlets i).sides_fried 1

/-- Represents a valid frying step -/
def valid_step (before after : FryingState) : Prop :=
  after.time = before.time + 5 ∧
  (∀ i : Fin 3, 
    (after.cutlets i).sides_fried 0 = (before.cutlets i).sides_fried 0 ∨
    (after.cutlets i).sides_fried 1 = (before.cutlets i).sides_fried 1) ∧
  (∀ i : Fin 2, after.pan i ≠ none → 
    (∃ j : Fin 3, after.pan i = some j ∧ 
      ((before.cutlets j).sides_fried 0 ≠ (after.cutlets j).sides_fried 0 ∨
       (before.cutlets j).sides_fried 1 ≠ (after.cutlets j).sides_fried 1)))

/-- The initial state of frying -/
def initial_state : FryingState := {
  time := 0,
  cutlets := λ _ ↦ { sides_fried := λ _ ↦ false },
  pan := λ _ ↦ none
}

/-- Theorem stating that it's possible to fry all cutlets in 15 minutes -/
theorem fry_all_cutlets_in_15_minutes : 
  ∃ (final_state : FryingState), 
    final_state.time ≤ 15 ∧ 
    all_fried final_state ∧
    ∃ (step1 step2 : FryingState), 
      valid_step initial_state step1 ∧
      valid_step step1 step2 ∧
      valid_step step2 final_state :=
sorry

end NUMINAMATH_CALUDE_fry_all_cutlets_in_15_minutes_l3129_312950


namespace NUMINAMATH_CALUDE_abc_inequalities_l3129_312957

theorem abc_inequalities (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_prod : (a + 1) * (b + 1) * (c + 1) = 8) :
  a + b + c ≥ 3 ∧ a * b * c ≤ 1 ∧
  (a + b + c = 3 ∧ a * b * c = 1 ↔ a = 1 ∧ b = 1 ∧ c = 1) :=
by sorry

end NUMINAMATH_CALUDE_abc_inequalities_l3129_312957


namespace NUMINAMATH_CALUDE_sabrina_leaves_l3129_312943

/-- The number of basil leaves Sabrina needs -/
def basil : ℕ := 12

/-- The number of sage leaves Sabrina needs -/
def sage : ℕ := basil / 2

/-- The number of verbena leaves Sabrina needs -/
def verbena : ℕ := sage + 5

/-- The total number of leaves Sabrina needs -/
def total : ℕ := basil + sage + verbena

theorem sabrina_leaves : total = 29 := by
  sorry

end NUMINAMATH_CALUDE_sabrina_leaves_l3129_312943


namespace NUMINAMATH_CALUDE_mary_pies_count_l3129_312977

theorem mary_pies_count (apples_per_pie : ℕ) (harvested_apples : ℕ) (apples_to_buy : ℕ) :
  apples_per_pie = 8 →
  harvested_apples = 50 →
  apples_to_buy = 30 →
  (harvested_apples + apples_to_buy) / apples_per_pie = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_mary_pies_count_l3129_312977


namespace NUMINAMATH_CALUDE_vector_magnitude_proof_l3129_312978

open Real

theorem vector_magnitude_proof (a b : ℝ × ℝ) :
  let angle := 60 * π / 180
  norm a = 2 ∧ norm b = 5 ∧ 
  a.1 * b.1 + a.2 * b.2 = norm a * norm b * cos angle →
  norm (2 • a - b) = sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_proof_l3129_312978


namespace NUMINAMATH_CALUDE_binomial_probability_two_successes_l3129_312966

/-- A random variable X follows a binomial distribution with parameters n and p -/
structure BinomialDistribution (n : ℕ) (p : ℝ) where
  X : ℝ → ℝ

/-- The probability mass function of a binomial distribution -/
def probability_mass_function (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
  (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

/-- Theorem: For a binomial distribution B(4, 1/2), P(X=2) = 3/8 -/
theorem binomial_probability_two_successes :
  ∀ (X : BinomialDistribution 4 (1/2)),
  probability_mass_function 4 (1/2) 2 = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_binomial_probability_two_successes_l3129_312966


namespace NUMINAMATH_CALUDE_water_heater_capacity_l3129_312944

/-- Represents a water heater with given parameters -/
structure WaterHeater where
  initialCapacity : ℝ
  addRate : ℝ → ℝ
  dischargeRate : ℝ → ℝ
  maxPersonUsage : ℝ

/-- Calculates the water volume as a function of time -/
def waterVolume (heater : WaterHeater) (t : ℝ) : ℝ :=
  heater.initialCapacity + heater.addRate t - heater.dischargeRate t

/-- Theorem: The given water heater can supply at least 4 people for continuous showers -/
theorem water_heater_capacity (heater : WaterHeater) 
  (h1 : heater.initialCapacity = 200)
  (h2 : ∀ t, heater.addRate t = 2 * t^2)
  (h3 : ∀ t, heater.dischargeRate t = 34 * t)
  (h4 : heater.maxPersonUsage = 60) :
  ∃ n : ℕ, n ≥ 4 ∧ 
    (∃ t : ℝ, t > 0 ∧ 
      heater.dischargeRate t / heater.maxPersonUsage ≥ n ∧
      ∀ s, 0 ≤ s ∧ s ≤ t → waterVolume heater s ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_water_heater_capacity_l3129_312944


namespace NUMINAMATH_CALUDE_china_population_scientific_notation_l3129_312922

/-- Represents the population of China in millions -/
def china_population : ℝ := 1412.60

/-- The scientific notation representation of the population -/
def scientific_notation : ℝ := 1.4126 * (10 ^ 5)

/-- Theorem stating that the scientific notation representation is correct -/
theorem china_population_scientific_notation :
  china_population = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_china_population_scientific_notation_l3129_312922


namespace NUMINAMATH_CALUDE_janet_needs_775_l3129_312969

/-- The amount of additional money Janet needs to rent an apartment -/
def additional_money_needed (savings : ℕ) (monthly_rent : ℕ) (advance_months : ℕ) (deposit : ℕ) : ℕ :=
  (monthly_rent * advance_months + deposit) - savings

/-- Proof that Janet needs $775 more to rent the apartment -/
theorem janet_needs_775 : 
  additional_money_needed 2225 1250 2 500 = 775 := by
  sorry

end NUMINAMATH_CALUDE_janet_needs_775_l3129_312969


namespace NUMINAMATH_CALUDE_car_A_original_speed_l3129_312964

/-- Represents the speed and position of a car --/
structure Car where
  speed : ℝ
  position : ℝ

/-- Represents the scenario of two cars meeting --/
structure MeetingScenario where
  carA : Car
  carB : Car
  meetingTime : ℝ
  meetingPosition : ℝ

/-- The original scenario where cars meet at point C --/
def originalScenario : MeetingScenario := sorry

/-- Scenario where car B increases speed by 5 km/h --/
def scenarioBFaster : MeetingScenario := sorry

/-- Scenario where car A increases speed by 5 km/h --/
def scenarioAFaster : MeetingScenario := sorry

theorem car_A_original_speed :
  ∃ (s : ℝ),
    (originalScenario.carA.speed = s) ∧
    (originalScenario.meetingTime = 6) ∧
    (scenarioBFaster.carB.speed = originalScenario.carB.speed + 5) ∧
    (scenarioBFaster.meetingPosition = originalScenario.meetingPosition - 12) ∧
    (scenarioAFaster.carA.speed = originalScenario.carA.speed + 5) ∧
    (scenarioAFaster.meetingPosition = originalScenario.meetingPosition + 16) ∧
    (s = 30) := by
  sorry

end NUMINAMATH_CALUDE_car_A_original_speed_l3129_312964


namespace NUMINAMATH_CALUDE_hyperbola_condition_equivalence_l3129_312991

/-- The equation represents a hyperbola -/
def is_hyperbola (k : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (k - 1) + y^2 / (k + 2) = 1 ∧ (k - 1) * (k + 2) < 0

/-- The condition 0 < k < 1 -/
def condition (k : ℝ) : Prop := 0 < k ∧ k < 1

theorem hyperbola_condition_equivalence :
  ∀ k : ℝ, is_hyperbola k ↔ condition k := by sorry

end NUMINAMATH_CALUDE_hyperbola_condition_equivalence_l3129_312991


namespace NUMINAMATH_CALUDE_units_digit_E_1000_l3129_312930

def E (n : ℕ) : ℕ := 3^(3^n) + 1

theorem units_digit_E_1000 : E 1000 % 10 = 4 := by sorry

end NUMINAMATH_CALUDE_units_digit_E_1000_l3129_312930


namespace NUMINAMATH_CALUDE_purchase_combinations_eq_545_l3129_312900

/-- Represents the number of oreo flavors available -/
def oreo_flavors : ℕ := 6

/-- Represents the number of milk flavors available -/
def milk_flavors : ℕ := 4

/-- Represents the total number of products purchased -/
def total_products : ℕ := 3

/-- Represents the number of flavors Alpha can choose from (excluding chocolate) -/
def alpha_flavors : ℕ := oreo_flavors - 1 + milk_flavors

/-- Function to calculate the number of ways Alpha and Beta can purchase products -/
def purchase_combinations : ℕ := sorry

/-- Theorem stating the correct number of purchase combinations -/
theorem purchase_combinations_eq_545 : purchase_combinations = 545 := by sorry

end NUMINAMATH_CALUDE_purchase_combinations_eq_545_l3129_312900


namespace NUMINAMATH_CALUDE_divisors_of_2744_l3129_312953

-- Define 2744 as the number we're interested in
def n : ℕ := 2744

-- Define the function that counts the number of positive divisors
def count_divisors (m : ℕ) : ℕ := (Finset.filter (· ∣ m) (Finset.range (m + 1))).card

-- State the theorem
theorem divisors_of_2744 : count_divisors n = 16 := by sorry

end NUMINAMATH_CALUDE_divisors_of_2744_l3129_312953


namespace NUMINAMATH_CALUDE_hyperbola_a_value_l3129_312990

theorem hyperbola_a_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∀ x y, x^2 / a^2 - y^2 / b^2 = 1) →
  (b / a = 2) →
  (a^2 + b^2 = 20) →
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_a_value_l3129_312990


namespace NUMINAMATH_CALUDE_problem_solution_l3129_312979

theorem problem_solution :
  ∀ (a b : ℝ),
  let A := 2 * a^2 + 3 * a * b - 2 * a - 1
  let B := -a^2 + a * b + a + 3
  (a = -1 ∧ b = 10 → 4 * A - (3 * A - 2 * B) = -45) ∧
  (a * b = 1 → 4 * A - (3 * A - 2 * B) = 10) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3129_312979


namespace NUMINAMATH_CALUDE_max_ab_linear_function_l3129_312972

/-- Given a linear function f(x) = ax + b where a and b are real numbers,
    if |f(x)| ≤ 1 for all x in [0, 1], then the maximum value of ab is 1/4. -/
theorem max_ab_linear_function (a b : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → |a * x + b| ≤ 1) →
  ab ≤ (1 : ℝ) / 4 ∧ ∃ a' b' : ℝ, (∀ x : ℝ, x ∈ Set.Icc 0 1 → |a' * x + b'| ≤ 1) ∧ a' * b' = (1 : ℝ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_max_ab_linear_function_l3129_312972


namespace NUMINAMATH_CALUDE_equation_solution_l3129_312929

theorem equation_solution (x : ℝ) :
  Real.sqrt ((3 / x) + 3) = 5 / 3 → x = -27 / 2 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3129_312929


namespace NUMINAMATH_CALUDE_factorial_1200_trailing_zeroes_l3129_312997

/-- The number of trailing zeroes in n! -/
def trailingZeroes (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625)

/-- Theorem: 1200! has 298 trailing zeroes -/
theorem factorial_1200_trailing_zeroes :
  trailingZeroes 1200 = 298 := by
  sorry

end NUMINAMATH_CALUDE_factorial_1200_trailing_zeroes_l3129_312997


namespace NUMINAMATH_CALUDE_boric_acid_solution_percentage_l3129_312914

/-- Proves that the percentage of boric acid in the first solution must be 1% 
    to create a 3% boric acid solution under the given conditions -/
theorem boric_acid_solution_percentage 
  (total_volume : ℝ) 
  (final_concentration : ℝ) 
  (volume1 : ℝ) 
  (volume2 : ℝ) 
  (concentration2 : ℝ) 
  (h1 : total_volume = 30)
  (h2 : final_concentration = 0.03)
  (h3 : volume1 = 15)
  (h4 : volume2 = 15)
  (h5 : concentration2 = 0.05)
  (h6 : volume1 + volume2 = total_volume)
  : ∃ (concentration1 : ℝ), 
    concentration1 = 0.01 ∧ 
    concentration1 * volume1 + concentration2 * volume2 = final_concentration * total_volume :=
by sorry

end NUMINAMATH_CALUDE_boric_acid_solution_percentage_l3129_312914


namespace NUMINAMATH_CALUDE_sin_70_in_terms_of_sin_10_l3129_312965

theorem sin_70_in_terms_of_sin_10 (k : ℝ) (h : Real.sin (10 * π / 180) = k) :
  Real.sin (70 * π / 180) = 1 - 2 * k^2 := by
  sorry

end NUMINAMATH_CALUDE_sin_70_in_terms_of_sin_10_l3129_312965


namespace NUMINAMATH_CALUDE_optimal_gasoline_percentage_l3129_312926

/-- Calculates the optimal gasoline percentage for a car's fuel mixture --/
theorem optimal_gasoline_percentage
  (initial_volume : ℝ)
  (initial_ethanol_percentage : ℝ)
  (initial_gasoline_percentage : ℝ)
  (added_ethanol : ℝ)
  (optimal_ethanol_percentage : ℝ)
  (h1 : initial_volume = 36)
  (h2 : initial_ethanol_percentage = 5)
  (h3 : initial_gasoline_percentage = 95)
  (h4 : added_ethanol = 2)
  (h5 : optimal_ethanol_percentage = 10)
  (h6 : initial_ethanol_percentage + initial_gasoline_percentage = 100) :
  let final_volume := initial_volume + added_ethanol
  let final_ethanol := initial_volume * (initial_ethanol_percentage / 100) + added_ethanol
  let final_ethanol_percentage := (final_ethanol / final_volume) * 100
  100 - optimal_ethanol_percentage = 90 ∧ final_ethanol_percentage = optimal_ethanol_percentage :=
by sorry

end NUMINAMATH_CALUDE_optimal_gasoline_percentage_l3129_312926


namespace NUMINAMATH_CALUDE_circle_center_l3129_312912

/-- The center of the circle with equation x^2 + y^2 - x + 2y = 0 has coordinates (1/2, -1) -/
theorem circle_center (x y : ℝ) : 
  x^2 + y^2 - x + 2*y = 0 → (x - 1/2)^2 + (y + 1)^2 = 5/4 := by
sorry

end NUMINAMATH_CALUDE_circle_center_l3129_312912


namespace NUMINAMATH_CALUDE_deck_size_l3129_312939

theorem deck_size (r b : ℕ) : 
  r ≠ 0 → 
  b ≠ 0 → 
  r / (r + b) = 1 / 4 → 
  r / (r + b + 6) = 1 / 6 → 
  r + b = 12 := by
sorry

end NUMINAMATH_CALUDE_deck_size_l3129_312939


namespace NUMINAMATH_CALUDE_planes_parallel_to_line_are_parallel_planes_parallel_to_plane_are_parallel_l3129_312989

-- Define a type for planes
variable (Plane : Type)

-- Define a type for lines
variable (Line : Type)

-- Define a relation for parallelism between planes
variable (parallel_planes : Plane → Plane → Prop)

-- Define a relation for parallelism between a plane and a line
variable (parallel_plane_line : Plane → Line → Prop)

-- Define a relation for parallelism between a plane and another plane
variable (parallel_plane_plane : Plane → Plane → Prop)

-- Theorem 1: Two planes parallel to the same line are parallel
theorem planes_parallel_to_line_are_parallel 
  (P Q : Plane) (L : Line) 
  (h1 : parallel_plane_line P L) 
  (h2 : parallel_plane_line Q L) : 
  parallel_planes P Q :=
sorry

-- Theorem 2: Two planes parallel to the same plane are parallel
theorem planes_parallel_to_plane_are_parallel 
  (P Q R : Plane) 
  (h1 : parallel_plane_plane P R) 
  (h2 : parallel_plane_plane Q R) : 
  parallel_planes P Q :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_to_line_are_parallel_planes_parallel_to_plane_are_parallel_l3129_312989


namespace NUMINAMATH_CALUDE_binomSum_not_div_five_l3129_312903

def binomSum (n : ℕ) : ℕ :=
  Finset.sum (Finset.range (n + 1)) (fun k => Nat.choose (2 * n + 1) (2 * k + 1) * 2^(3 * k))

theorem binomSum_not_div_five (n : ℕ) : ¬(5 ∣ binomSum n) := by
  sorry

end NUMINAMATH_CALUDE_binomSum_not_div_five_l3129_312903


namespace NUMINAMATH_CALUDE_sphere_volume_in_cube_l3129_312945

/-- Given a cube with edge length a and two congruent spheres inscribed in opposite trihedral angles
    that touch each other, this theorem states the volume of each sphere. -/
theorem sphere_volume_in_cube (a : ℝ) (a_pos : 0 < a) : 
  ∃ (r : ℝ), r = (3 * a - a * Real.sqrt 3) / 4 ∧ 
              (4 / 3 : ℝ) * Real.pi * r^3 = (4 / 3 : ℝ) * Real.pi * ((3 * a - a * Real.sqrt 3) / 4)^3 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_in_cube_l3129_312945


namespace NUMINAMATH_CALUDE_complex_equality_l3129_312935

theorem complex_equality (a b : ℝ) : (1 + Complex.I) + (2 - 3 * Complex.I) = a + b * Complex.I → a = 3 ∧ b = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_l3129_312935


namespace NUMINAMATH_CALUDE_specific_rhombus_area_l3129_312995

/-- Represents a rhombus with given properties -/
structure Rhombus where
  side_length : ℝ
  diagonal_difference : ℝ
  diagonals_perpendicular : Bool

/-- Calculates the area of a rhombus given its properties -/
def rhombus_area (r : Rhombus) : ℝ :=
  sorry

/-- Theorem stating the area of a specific rhombus -/
theorem specific_rhombus_area :
  let r : Rhombus := { 
    side_length := Real.sqrt 145,
    diagonal_difference := 8,
    diagonals_perpendicular := true
  }
  rhombus_area r = (Real.sqrt 274 * (Real.sqrt 274 - 4)) / 4 := by
  sorry

end NUMINAMATH_CALUDE_specific_rhombus_area_l3129_312995


namespace NUMINAMATH_CALUDE_greatest_power_of_two_factor_l3129_312927

theorem greatest_power_of_two_factor (n : ℕ) : 
  2^1200 ∣ (15^600 - 3^600) ∧ 
  ∀ k > 1200, ¬(2^k ∣ (15^600 - 3^600)) :=
sorry

end NUMINAMATH_CALUDE_greatest_power_of_two_factor_l3129_312927


namespace NUMINAMATH_CALUDE_function_satisfying_inequality_is_constant_two_l3129_312933

/-- A function satisfying the given inequality for all real x and y is constant and equal to 2. -/
theorem function_satisfying_inequality_is_constant_two 
  (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, 2 * f x + 2 * f y - f x * f y ≥ 4) : 
  ∀ x : ℝ, f x = 2 := by
  sorry

end NUMINAMATH_CALUDE_function_satisfying_inequality_is_constant_two_l3129_312933


namespace NUMINAMATH_CALUDE_no_integer_pairs_with_square_diff_30_l3129_312993

theorem no_integer_pairs_with_square_diff_30 :
  ¬∃ (m n : ℕ), m ≥ n ∧ m * m - n * n = 30 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_pairs_with_square_diff_30_l3129_312993


namespace NUMINAMATH_CALUDE_total_weight_is_20_2_l3129_312941

-- Define the capacities of the jugs
def jug1_capacity : ℝ := 2
def jug2_capacity : ℝ := 3
def jug3_capacity : ℝ := 4

-- Define the fill percentages
def jug1_fill_percent : ℝ := 0.7
def jug2_fill_percent : ℝ := 0.6
def jug3_fill_percent : ℝ := 0.5

-- Define the sand densities
def jug1_density : ℝ := 5
def jug2_density : ℝ := 4
def jug3_density : ℝ := 3

-- Calculate the weight of sand in each jug
def jug1_weight : ℝ := jug1_capacity * jug1_fill_percent * jug1_density
def jug2_weight : ℝ := jug2_capacity * jug2_fill_percent * jug2_density
def jug3_weight : ℝ := jug3_capacity * jug3_fill_percent * jug3_density

-- Total weight of sand in all jugs
def total_weight : ℝ := jug1_weight + jug2_weight + jug3_weight

theorem total_weight_is_20_2 : total_weight = 20.2 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_is_20_2_l3129_312941


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3129_312980

theorem sufficient_not_necessary (a b : ℝ) :
  (∀ a b : ℝ, a > 1 ∧ b > 2 → a + b > 3 ∧ a * b > 2) ∧
  (∃ a b : ℝ, a + b > 3 ∧ a * b > 2 ∧ ¬(a > 1 ∧ b > 2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3129_312980


namespace NUMINAMATH_CALUDE_binomial_60_3_l3129_312973

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end NUMINAMATH_CALUDE_binomial_60_3_l3129_312973


namespace NUMINAMATH_CALUDE_cyclic_inequality_l3129_312948

theorem cyclic_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 1) :
  (x^3 / (x^2 + y)) + (y^3 / (y^2 + z)) + (z^3 / (z^2 + x)) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_inequality_l3129_312948


namespace NUMINAMATH_CALUDE_exists_valid_path_2020_l3129_312984

/-- Represents a square grid with diagonals drawn in each cell. -/
structure DiagonalGrid (n : ℕ) where
  size : n > 0

/-- Represents a path on the diagonal grid. -/
structure DiagonalPath (n : ℕ) where
  grid : DiagonalGrid n
  is_closed : Bool
  visits_all_cells : Bool
  no_repeated_diagonals : Bool

/-- Theorem stating the existence of a valid path in a 2020x2020 grid. -/
theorem exists_valid_path_2020 :
  ∃ (path : DiagonalPath 2020),
    path.is_closed ∧
    path.visits_all_cells ∧
    path.no_repeated_diagonals :=
  sorry

end NUMINAMATH_CALUDE_exists_valid_path_2020_l3129_312984


namespace NUMINAMATH_CALUDE_baseball_cards_difference_l3129_312949

theorem baseball_cards_difference (jorge matias carlos : ℕ) : 
  jorge = matias → 
  carlos = 20 → 
  jorge + matias + carlos = 48 → 
  carlos - matias = 6 := by
sorry

end NUMINAMATH_CALUDE_baseball_cards_difference_l3129_312949


namespace NUMINAMATH_CALUDE_orange_juice_problem_l3129_312946

theorem orange_juice_problem (jug_volume : ℚ) (portion_drunk : ℚ) :
  jug_volume = 2/7 →
  portion_drunk = 5/8 →
  portion_drunk * jug_volume = 5/28 := by
  sorry

end NUMINAMATH_CALUDE_orange_juice_problem_l3129_312946


namespace NUMINAMATH_CALUDE_total_money_l3129_312952

def sam_money : ℕ := 38
def erica_money : ℕ := 53

theorem total_money : sam_money + erica_money = 91 := by
  sorry

end NUMINAMATH_CALUDE_total_money_l3129_312952


namespace NUMINAMATH_CALUDE_tim_balloon_count_l3129_312947

theorem tim_balloon_count (dan_balloons : ℕ) (tim_multiplier : ℕ) (h1 : dan_balloons = 29) (h2 : tim_multiplier = 7) : 
  dan_balloons * tim_multiplier = 203 := by
  sorry

end NUMINAMATH_CALUDE_tim_balloon_count_l3129_312947


namespace NUMINAMATH_CALUDE_new_person_weight_l3129_312905

/-- Given a group of 8 persons where the average weight increases by 2.5 kg
    when a person weighing 50 kg is replaced, the weight of the new person is 70 kg. -/
theorem new_person_weight (n : ℕ) (avg_increase : ℝ) (old_weight : ℝ) :
  n = 8 →
  avg_increase = 2.5 →
  old_weight = 50 →
  n * avg_increase + old_weight = 70 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l3129_312905


namespace NUMINAMATH_CALUDE_polygon_angle_theorem_l3129_312934

/-- 
Theorem: For a convex n-sided polygon with one interior angle x° and 
the sum of the remaining interior angles 2180°, x = 160° and n = 15.
-/
theorem polygon_angle_theorem (n : ℕ) (x : ℝ) 
  (h_convex : n ≥ 3)
  (h_sum : x + 2180 = 180 * (n - 2)) :
  x = 160 ∧ n = 15 := by
  sorry

end NUMINAMATH_CALUDE_polygon_angle_theorem_l3129_312934


namespace NUMINAMATH_CALUDE_committee_selection_count_l3129_312924

/-- The number of ways to choose a committee of size k from n people -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The size of the club -/
def club_size : ℕ := 30

/-- The size of the committee -/
def committee_size : ℕ := 5

/-- Theorem: The number of ways to choose a 5-person committee from a 30-person club is 142506 -/
theorem committee_selection_count : choose club_size committee_size = 142506 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_count_l3129_312924


namespace NUMINAMATH_CALUDE_protege_zero_implies_two_and_five_l3129_312902

/-- A digit is a protégé of a natural number if it is the units digit of some divisor of that number. -/
def isProtege (d : Nat) (n : Nat) : Prop :=
  ∃ k : Nat, k ∣ n ∧ k % 10 = d

/-- Theorem: If 0 is a protégé of a natural number, then 2 and 5 are also protégés of that number. -/
theorem protege_zero_implies_two_and_five (n : Nat) :
  isProtege 0 n → isProtege 2 n ∧ isProtege 5 n := by
  sorry


end NUMINAMATH_CALUDE_protege_zero_implies_two_and_five_l3129_312902


namespace NUMINAMATH_CALUDE_xiao_jun_pictures_xiao_jun_pictures_proof_l3129_312911

theorem xiao_jun_pictures : ℕ → Prop :=
  fun original : ℕ =>
    let half := original / 2
    let given_away := half - 1
    let remaining := original - given_away
    remaining = 25 → original = 48

-- The proof is omitted
theorem xiao_jun_pictures_proof : xiao_jun_pictures 48 := by
  sorry

end NUMINAMATH_CALUDE_xiao_jun_pictures_xiao_jun_pictures_proof_l3129_312911


namespace NUMINAMATH_CALUDE_sum_of_radii_l3129_312954

/-- A circle with center C(r, r) is tangent to the positive x-axis and y-axis,
    and externally tangent to a circle centered at (4,0) with radius 2. -/
def CircleTangency (r : ℝ) : Prop :=
  r > 0 ∧ (r - 4)^2 + r^2 = (r + 2)^2

/-- The sum of all possible radii of the circle with center C is 12. -/
theorem sum_of_radii :
  ∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ CircleTangency r₁ ∧ CircleTangency r₂ ∧ r₁ + r₂ = 12 :=
sorry

end NUMINAMATH_CALUDE_sum_of_radii_l3129_312954


namespace NUMINAMATH_CALUDE_ratio_difference_problem_l3129_312915

theorem ratio_difference_problem (A B : ℚ) : 
  A / B = 3 / 5 → B - A = 12 → A = 18 := by
  sorry

end NUMINAMATH_CALUDE_ratio_difference_problem_l3129_312915


namespace NUMINAMATH_CALUDE_intersection_M_N_l3129_312923

def M : Set ℝ := {x | x^2 + 3*x + 2 > 0}

def N : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_M_N : M ∩ N = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3129_312923


namespace NUMINAMATH_CALUDE_polynomial_remainder_l3129_312938

theorem polynomial_remainder (x : ℂ) : 
  x^2 - x + 1 = 0 → (2*x^5 - x^4 + x^2 - 1)*(x^3 - 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l3129_312938


namespace NUMINAMATH_CALUDE_additional_cats_needed_prove_additional_cats_l3129_312962

theorem additional_cats_needed (total_mice : ℕ) (initial_cats : ℕ) (initial_days : ℕ) (total_days : ℕ) : ℕ :=
  let initial_work := total_mice / 2
  let remaining_work := total_mice - initial_work
  let initial_rate := initial_work / (initial_cats * initial_days)
  let remaining_days := total_days - initial_days
  let additional_cats := (remaining_work / (initial_rate * remaining_days)) - initial_cats
  additional_cats

theorem prove_additional_cats :
  additional_cats_needed 100 2 5 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_additional_cats_needed_prove_additional_cats_l3129_312962


namespace NUMINAMATH_CALUDE_equal_expressions_l3129_312985

theorem equal_expressions : (-2)^3 = -2^3 := by
  sorry

end NUMINAMATH_CALUDE_equal_expressions_l3129_312985


namespace NUMINAMATH_CALUDE_exists_all_strawberry_day_l3129_312998

-- Define the type for our matrix
def WorkSchedule := Matrix (Fin 7) (Fin 16) Bool

-- Define the conditions
def first_day_all_mine (schedule : WorkSchedule) : Prop :=
  ∀ i : Fin 7, schedule i 0 = false

def at_least_three_different (schedule : WorkSchedule) : Prop :=
  ∀ j k : Fin 16, j ≠ k → 
    (∃ (s : Finset (Fin 7)), s.card ≥ 3 ∧ 
      (∀ i ∈ s, schedule i j ≠ schedule i k))

-- The main theorem
theorem exists_all_strawberry_day (schedule : WorkSchedule) 
  (h1 : first_day_all_mine schedule)
  (h2 : at_least_three_different schedule) : 
  ∃ j : Fin 16, ∀ i : Fin 7, schedule i j = true :=
sorry

end NUMINAMATH_CALUDE_exists_all_strawberry_day_l3129_312998


namespace NUMINAMATH_CALUDE_games_last_month_l3129_312987

def games_this_month : ℕ := 9
def games_next_month : ℕ := 7
def total_games : ℕ := 24

theorem games_last_month : total_games - (games_this_month + games_next_month) = 8 := by
  sorry

end NUMINAMATH_CALUDE_games_last_month_l3129_312987


namespace NUMINAMATH_CALUDE_function_identity_l3129_312919

theorem function_identity (f : ℝ → ℝ) :
  (∀ x, f x + 2 * f (3 - x) = x^2) →
  (∀ x, f x = (1/3) * x^2 - 4 * x + 6) :=
by sorry

end NUMINAMATH_CALUDE_function_identity_l3129_312919


namespace NUMINAMATH_CALUDE_inequality_of_positive_reals_l3129_312928

theorem inequality_of_positive_reals (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x^x * y^y * z^z ≥ (x*y*z)^((x+y+z)/3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_positive_reals_l3129_312928


namespace NUMINAMATH_CALUDE_xy_square_sum_l3129_312931

theorem xy_square_sum (x y : ℝ) (h1 : x + y = -2) (h2 : x * y = -3) :
  x^2 * y + x * y^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_xy_square_sum_l3129_312931
