import Mathlib

namespace NUMINAMATH_CALUDE_smallest_number_in_specific_set_l1289_128985

theorem smallest_number_in_specific_set (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Three positive integers
  (a + b + c) / 3 = 30 →   -- Arithmetic mean is 30
  (min (max a b) (max b c)) = 31 →  -- Median is 31
  max a (max b c) = 31 + 8 →  -- Largest number is 8 more than median
  min a (min b c) = 20 := by  -- Smallest number is 20
sorry

end NUMINAMATH_CALUDE_smallest_number_in_specific_set_l1289_128985


namespace NUMINAMATH_CALUDE_circle_chords_and_triangles_l1289_128970

/-- Given 10 points on the circumference of a circle, prove the number of chords and triangles -/
theorem circle_chords_and_triangles (n : ℕ) (hn : n = 10) :
  (Nat.choose n 2 = 45) ∧ (Nat.choose n 3 = 120) := by
  sorry

#check circle_chords_and_triangles

end NUMINAMATH_CALUDE_circle_chords_and_triangles_l1289_128970


namespace NUMINAMATH_CALUDE_set_operations_l1289_128932

-- Define the sets A and B
def A : Set ℝ := {x | x ≥ 1 ∨ x ≤ -3}
def B : Set ℝ := {x | -4 < x ∧ x < 0}

-- State the theorem
theorem set_operations :
  (A ∩ B = {x | -4 < x ∧ x ≤ -3}) ∧
  (A ∪ B = {x | x < 0 ∨ x ≥ 1}) ∧
  (A ∪ (Set.univ \ B) = {x | x ≤ -3 ∨ x ≥ 0}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l1289_128932


namespace NUMINAMATH_CALUDE_two_numbers_problem_l1289_128959

theorem two_numbers_problem (x y : ℕ) (h1 : x + y = 60) (h2 : Nat.gcd x y + Nat.lcm x y = 84) :
  (x = 24 ∧ y = 36) ∨ (x = 36 ∧ y = 24) := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_problem_l1289_128959


namespace NUMINAMATH_CALUDE_total_glasses_at_restaurant_l1289_128979

/-- Represents the number of glasses in a small box -/
def small_box : ℕ := 12

/-- Represents the number of glasses in a large box -/
def large_box : ℕ := 16

/-- Represents the difference in the number of large boxes compared to small boxes -/
def box_difference : ℕ := 16

/-- Represents the average number of glasses per box -/
def average_glasses : ℕ := 15

theorem total_glasses_at_restaurant :
  ∃ (small_boxes large_boxes : ℕ),
    large_boxes = small_boxes + box_difference ∧
    (small_box * small_boxes + large_box * large_boxes) / (small_boxes + large_boxes) = average_glasses ∧
    small_box * small_boxes + large_box * large_boxes = 480 :=
sorry

end NUMINAMATH_CALUDE_total_glasses_at_restaurant_l1289_128979


namespace NUMINAMATH_CALUDE_student_age_problem_l1289_128988

theorem student_age_problem (total_students : ℕ) (avg_age : ℕ) 
  (group1_size : ℕ) (group1_avg : ℕ) 
  (group2_size : ℕ) (group2_avg : ℕ) 
  (group3_size : ℕ) (group3_avg : ℕ) : 
  total_students = 25 →
  avg_age = 24 →
  group1_size = 8 →
  group1_avg = 22 →
  group2_size = 10 →
  group2_avg = 20 →
  group3_size = 6 →
  group3_avg = 28 →
  group1_size + group2_size + group3_size + 1 = total_students →
  (total_students * avg_age) - 
  (group1_size * group1_avg + group2_size * group2_avg + group3_size * group3_avg) = 56 :=
by sorry

end NUMINAMATH_CALUDE_student_age_problem_l1289_128988


namespace NUMINAMATH_CALUDE_exam_candidates_count_l1289_128994

theorem exam_candidates_count : 
  ∀ (T P F : ℕ) (total_avg passed_avg failed_avg : ℚ),
    P = 100 →
    total_avg = 35 →
    passed_avg = 39 →
    failed_avg = 15 →
    T = P + F →
    (total_avg * T : ℚ) = (passed_avg * P : ℚ) + (failed_avg * F : ℚ) →
    T = 120 := by
  sorry

end NUMINAMATH_CALUDE_exam_candidates_count_l1289_128994


namespace NUMINAMATH_CALUDE_range_of_x_l1289_128977

-- Define the determinant operation
def det (a b c d : ℝ) : ℝ := a * d - b * c

-- Define the theorem
theorem range_of_x (x : ℝ) : 
  det x 3 (-x) x < det 2 0 1 2 → -4 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l1289_128977


namespace NUMINAMATH_CALUDE_power_of_three_division_l1289_128951

theorem power_of_three_division : (3 : ℕ) ^ 2023 / 9 = (3 : ℕ) ^ 2021 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_division_l1289_128951


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1289_128913

/-- An isosceles triangle with two sides of length 3 and one side of length 1 -/
structure IsoscelesTriangle :=
  (side1 : ℝ)
  (side2 : ℝ)
  (base : ℝ)
  (isIsosceles : side1 = side2)
  (side1_eq_3 : side1 = 3)
  (base_eq_1 : base = 1)

/-- The perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ :=
  t.side1 + t.side2 + t.base

/-- Theorem: The perimeter of the specified isosceles triangle is 7 -/
theorem isosceles_triangle_perimeter :
  ∀ t : IsoscelesTriangle, perimeter t = 7 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1289_128913


namespace NUMINAMATH_CALUDE_min_pool_cost_l1289_128966

/-- Represents the construction cost of a rectangular pool -/
def pool_cost (length width depth : ℝ) (wall_price : ℝ) : ℝ :=
  (2 * (length + width) * depth * wall_price) + (length * width * 1.5 * wall_price)

/-- Theorem stating the minimum cost for the pool construction -/
theorem min_pool_cost (a : ℝ) (h_a : a > 0) :
  let volume := 4800
  let depth := 3
  ∃ (length width : ℝ),
    length > 0 ∧ 
    width > 0 ∧
    length * width * depth = volume ∧
    ∀ (l w : ℝ), l > 0 → w > 0 → l * w * depth = volume →
      pool_cost length width depth a ≤ pool_cost l w depth a ∧
      pool_cost length width depth a = 2880 * a :=
sorry

end NUMINAMATH_CALUDE_min_pool_cost_l1289_128966


namespace NUMINAMATH_CALUDE_other_number_value_l1289_128975

theorem other_number_value (x y : ℝ) : 
  y = 125 * 1.1 →
  x = y * 0.9 →
  x = 123.75 →
  y = 137.5 := by
sorry

end NUMINAMATH_CALUDE_other_number_value_l1289_128975


namespace NUMINAMATH_CALUDE_sum_of_squares_of_quadratic_roots_l1289_128945

theorem sum_of_squares_of_quadratic_roots : ∀ (s₁ s₂ : ℝ), 
  s₁^2 - 20*s₁ + 32 = 0 → 
  s₂^2 - 20*s₂ + 32 = 0 → 
  s₁^2 + s₂^2 = 336 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_quadratic_roots_l1289_128945


namespace NUMINAMATH_CALUDE_f_max_at_three_halves_l1289_128920

/-- The quadratic function f(x) = -3x^2 + 9x - 1 -/
def f (x : ℝ) : ℝ := -3 * x^2 + 9 * x - 1

/-- The theorem states that f(x) attains its maximum value when x = 3/2 -/
theorem f_max_at_three_halves :
  ∃ (c : ℝ), c = 3/2 ∧ ∀ (x : ℝ), f x ≤ f c :=
by
  sorry

end NUMINAMATH_CALUDE_f_max_at_three_halves_l1289_128920


namespace NUMINAMATH_CALUDE_toms_initial_investment_l1289_128922

theorem toms_initial_investment (t j k : ℝ) : 
  t + j + k = 1200 →
  t - 150 + 3*j + 3*k = 1800 →
  t = 825 := by
sorry

end NUMINAMATH_CALUDE_toms_initial_investment_l1289_128922


namespace NUMINAMATH_CALUDE_probability_all_odd_is_correct_l1289_128965

def total_slips : ℕ := 10
def odd_slips : ℕ := 5
def drawn_slips : ℕ := 4

def probability_all_odd : ℚ := (odd_slips.choose drawn_slips) / (total_slips.choose drawn_slips)

theorem probability_all_odd_is_correct : 
  probability_all_odd = 1 / 42 := by sorry

end NUMINAMATH_CALUDE_probability_all_odd_is_correct_l1289_128965


namespace NUMINAMATH_CALUDE_expand_product_l1289_128971

theorem expand_product (x : ℝ) : 4 * (x - 5) * (x + 8) = 4 * x^2 + 12 * x - 160 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l1289_128971


namespace NUMINAMATH_CALUDE_expansion_equals_power_l1289_128981

theorem expansion_equals_power (x : ℝ) :
  (x - 2)^5 + 5*(x - 2)^4 + 10*(x - 2)^3 + 10*(x - 2)^2 + 5*(x - 2) + 1 = (x - 1)^5 := by
  sorry

end NUMINAMATH_CALUDE_expansion_equals_power_l1289_128981


namespace NUMINAMATH_CALUDE_mateo_absent_days_l1289_128998

/-- Calculates the number of days not worked given weekly salary, work days per week, and deducted salary -/
def daysNotWorked (weeklySalary workDaysPerWeek deductedSalary : ℚ) : ℕ :=
  let dailySalary := weeklySalary / workDaysPerWeek
  let exactDaysNotWorked := deductedSalary / dailySalary
  (exactDaysNotWorked + 1/2).floor.toNat

/-- Proves that given the specific conditions, the number of days not worked is 2 -/
theorem mateo_absent_days :
  daysNotWorked 791 5 339 = 2 := by
  sorry

#eval daysNotWorked 791 5 339

end NUMINAMATH_CALUDE_mateo_absent_days_l1289_128998


namespace NUMINAMATH_CALUDE_f_values_l1289_128938

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sqrt ((1 - Real.sin x) / (1 + Real.sin x)) - Real.sqrt ((1 + Real.sin x) / (1 - Real.sin x))) *
  (Real.sqrt ((1 - Real.cos x) / (1 + Real.cos x)) - Real.sqrt ((1 + Real.cos x) / (1 - Real.cos x)))

theorem f_values (x : ℝ) 
  (h1 : x ∈ Set.Ioo 0 (2 * Real.pi))
  (h2 : x ≠ Real.pi / 2 ∧ x ≠ Real.pi ∧ x ≠ 3 * Real.pi / 2) :
  (0 < x ∧ x < Real.pi / 2 ∨ Real.pi < x ∧ x < 3 * Real.pi / 2) → f x = 4 ∧
  (Real.pi / 2 < x ∧ x < Real.pi ∨ 3 * Real.pi / 2 < x ∧ x < 2 * Real.pi) → f x = -4 := by
  sorry

#check f_values

end NUMINAMATH_CALUDE_f_values_l1289_128938


namespace NUMINAMATH_CALUDE_max_visible_cubes_11_l1289_128967

/-- Represents a cube made of unit cubes --/
structure UnitCube where
  size : ℕ

/-- Calculates the maximum number of visible unit cubes from a single point --/
def max_visible_cubes (cube : UnitCube) : ℕ :=
  3 * cube.size^2 - 3 * (cube.size - 1) + 1

/-- Theorem stating that for an 11x11x11 cube, the maximum number of visible unit cubes is 331 --/
theorem max_visible_cubes_11 :
  max_visible_cubes ⟨11⟩ = 331 := by
  sorry

#eval max_visible_cubes ⟨11⟩

end NUMINAMATH_CALUDE_max_visible_cubes_11_l1289_128967


namespace NUMINAMATH_CALUDE_hundred_three_square_partitions_l1289_128905

/-- A function that returns the number of ways to write a given number as the sum of three positive perfect squares, where the order doesn't matter. -/
def count_three_square_partitions (n : ℕ) : ℕ :=
  sorry

/-- The theorem stating that there is exactly one way to write 100 as the sum of three positive perfect squares, where the order doesn't matter. -/
theorem hundred_three_square_partitions : count_three_square_partitions 100 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hundred_three_square_partitions_l1289_128905


namespace NUMINAMATH_CALUDE_smallest_number_divisible_l1289_128910

theorem smallest_number_divisible (n : ℕ) : n = 6297 ↔ 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 3) = 18 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 3) = 70 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 3) = 100 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 3) = 84 * k)) ∧
  (∃ k₁ k₂ k₃ k₄ : ℕ, (n + 3) = 18 * k₁ ∧ (n + 3) = 70 * k₂ ∧ (n + 3) = 100 * k₃ ∧ (n + 3) = 84 * k₄) :=
by sorry

#check smallest_number_divisible

end NUMINAMATH_CALUDE_smallest_number_divisible_l1289_128910


namespace NUMINAMATH_CALUDE_frog_jump_distance_l1289_128942

/-- The jumping contest problem -/
theorem frog_jump_distance 
  (grasshopper_jump : ℕ) 
  (grasshopper_frog_diff : ℕ) 
  (h1 : grasshopper_jump = 19)
  (h2 : grasshopper_jump = grasshopper_frog_diff + frog_jump) :
  frog_jump = 15 :=
by
  sorry

#check frog_jump_distance

end NUMINAMATH_CALUDE_frog_jump_distance_l1289_128942


namespace NUMINAMATH_CALUDE_tangent_line_equation_l1289_128950

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 4*x^2 + 4

-- Define the point of tangency
def point : ℝ × ℝ := (1, 1)

-- State the theorem
theorem tangent_line_equation :
  ∃ (m b : ℝ), 
    (∀ x, (deriv f) x = 3*x^2 - 8*x) ∧
    (deriv f) (point.1) = m ∧
    f point.1 = point.2 ∧
    (∀ x, m * (x - point.1) + point.2 = -5 * x + 6) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l1289_128950


namespace NUMINAMATH_CALUDE_simplify_radicals_l1289_128973

theorem simplify_radicals : Real.sqrt (5 * 3) * Real.sqrt (3^3 * 5^3) = 225 := by
  sorry

end NUMINAMATH_CALUDE_simplify_radicals_l1289_128973


namespace NUMINAMATH_CALUDE_bicycle_problem_l1289_128900

/-- The time when two people traveling perpendicular to each other at different speeds are 100 miles apart -/
theorem bicycle_problem (jenny_speed mark_speed : ℝ) (h1 : jenny_speed = 10) (h2 : mark_speed = 15) :
  let t := (20 * Real.sqrt 13) / 13
  (t * jenny_speed) ^ 2 + (t * mark_speed) ^ 2 = 100 ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_problem_l1289_128900


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l1289_128928

theorem quadratic_roots_relation (a b p q : ℝ) (r₁ r₂ : ℂ) : 
  (r₁ + r₂ = -a ∧ r₁ * r₂ = b) →  -- roots of x² + ax + b = 0
  (r₁^2 + r₂^2 = -p ∧ r₁^2 * r₂^2 = q) →  -- r₁² and r₂² are roots of x² + px + q = 0
  p = -a^2 + 2*b :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l1289_128928


namespace NUMINAMATH_CALUDE_p_costs_more_after_10_years_l1289_128903

/-- Represents the yearly price increase in paise -/
structure PriceIncrease where
  p : ℚ  -- Price increase for commodity P
  q : ℚ  -- Price increase for commodity Q

/-- Represents the initial prices in rupees -/
structure InitialPrice where
  p : ℚ  -- Initial price for commodity P
  q : ℚ  -- Initial price for commodity Q

/-- Calculates the year when commodity P costs 40 paise more than commodity Q -/
def yearWhenPCostsMoreThanQ (increase : PriceIncrease) (initial : InitialPrice) : ℕ :=
  sorry

/-- The theorem stating that P costs 40 paise more than Q after 10 years -/
theorem p_costs_more_after_10_years 
  (increase : PriceIncrease) 
  (initial : InitialPrice) 
  (h1 : increase.p = 40/100) 
  (h2 : increase.q = 15/100) 
  (h3 : initial.p = 420/100) 
  (h4 : initial.q = 630/100) : 
  yearWhenPCostsMoreThanQ increase initial = 10 := by sorry

end NUMINAMATH_CALUDE_p_costs_more_after_10_years_l1289_128903


namespace NUMINAMATH_CALUDE_arithmetic_sequence_n_value_l1289_128987

/-- An arithmetic sequence with first term a₁, common difference d, and nth term aₙ -/
structure ArithmeticSequence where
  a₁ : ℝ
  d : ℝ
  n : ℕ
  aₙ : ℝ
  seq_def : aₙ = a₁ + (n - 1) * d

/-- The theorem stating that for the given arithmetic sequence, n = 100 -/
theorem arithmetic_sequence_n_value
  (seq : ArithmeticSequence)
  (h1 : seq.a₁ = 1)
  (h2 : seq.d = 3)
  (h3 : seq.aₙ = 298) :
  seq.n = 100 := by
  sorry

#check arithmetic_sequence_n_value

end NUMINAMATH_CALUDE_arithmetic_sequence_n_value_l1289_128987


namespace NUMINAMATH_CALUDE_sum_congruence_l1289_128909

theorem sum_congruence : ∃ k : ℤ, (82 + 83 + 84 + 85 + 86 + 87 + 88 + 89) = 17 * k + 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_congruence_l1289_128909


namespace NUMINAMATH_CALUDE_muffin_banana_cost_ratio_l1289_128918

/-- The cost ratio of muffins to bananas --/
def cost_ratio (muffin_cost banana_cost : ℚ) : ℚ := muffin_cost / banana_cost

/-- Susie's purchase --/
def susie_purchase (muffin_cost banana_cost : ℚ) : ℚ := 5 * muffin_cost + 4 * banana_cost

/-- Calvin's purchase --/
def calvin_purchase (muffin_cost banana_cost : ℚ) : ℚ := 3 * muffin_cost + 20 * banana_cost

theorem muffin_banana_cost_ratio :
  ∀ (muffin_cost banana_cost : ℚ),
    muffin_cost > 0 →
    banana_cost > 0 →
    calvin_purchase muffin_cost banana_cost = 3 * susie_purchase muffin_cost banana_cost →
    cost_ratio muffin_cost banana_cost = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_muffin_banana_cost_ratio_l1289_128918


namespace NUMINAMATH_CALUDE_parabola_triangle_area_l1289_128902

/-- Given a parabola y = x^2 - 20x + c (c ≠ 0) that intersects the x-axis at points A and B
    and the y-axis at point C, where A and C are symmetrical with respect to the line y = -x,
    the area of triangle ABC is 231. -/
theorem parabola_triangle_area (c : ℝ) (hc : c ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ x^2 - 20*x + c
  let A := (21 : ℝ)
  let B := (-1 : ℝ)
  let C := (0, c)
  (∀ x, f x = 0 → x = A ∨ x = B) →
  (f 0 = c) →
  (A, 0) = (-C.2, 0) →
  (1/2 : ℝ) * (A - B) * (-C.2) = 231 :=
by sorry

end NUMINAMATH_CALUDE_parabola_triangle_area_l1289_128902


namespace NUMINAMATH_CALUDE_incorrect_inequality_l1289_128976

-- Define the conditions
variable (a b : ℝ)
variable (h1 : b < a)
variable (h2 : a < 0)

-- Define the theorem
theorem incorrect_inequality :
  ¬((1/2:ℝ)^b < (1/2:ℝ)^a) :=
by sorry

end NUMINAMATH_CALUDE_incorrect_inequality_l1289_128976


namespace NUMINAMATH_CALUDE_line_translation_l1289_128997

/-- A line in the 2D plane represented by its slope and y-intercept. -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Vertical translation of a line. -/
def verticalTranslate (l : Line) (d : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept - d }

theorem line_translation (x : ℝ) :
  let original := Line.mk 2 0
  let transformed := Line.mk 2 (-3)
  transformed = verticalTranslate original 3 := by sorry

end NUMINAMATH_CALUDE_line_translation_l1289_128997


namespace NUMINAMATH_CALUDE_tv_sets_in_shop_a_l1289_128978

/-- The number of electronic shops in the Naza market -/
def num_shops : ℕ := 5

/-- The average number of TV sets in each shop -/
def average_tv_sets : ℕ := 48

/-- The number of TV sets in shop b -/
def tv_sets_b : ℕ := 30

/-- The number of TV sets in shop c -/
def tv_sets_c : ℕ := 60

/-- The number of TV sets in shop d -/
def tv_sets_d : ℕ := 80

/-- The number of TV sets in shop e -/
def tv_sets_e : ℕ := 50

/-- Theorem: Given the conditions, shop a must have 20 TV sets -/
theorem tv_sets_in_shop_a : 
  (num_shops * average_tv_sets) - (tv_sets_b + tv_sets_c + tv_sets_d + tv_sets_e) = 20 := by
  sorry

end NUMINAMATH_CALUDE_tv_sets_in_shop_a_l1289_128978


namespace NUMINAMATH_CALUDE_square_side_length_l1289_128939

theorem square_side_length (diagonal_inches : ℝ) (h : diagonal_inches = 2 * Real.sqrt 2) :
  let diagonal_feet := diagonal_inches / 12
  let side_feet := diagonal_feet / Real.sqrt 2
  side_feet = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_square_side_length_l1289_128939


namespace NUMINAMATH_CALUDE_total_tape_theorem_l1289_128907

/-- The amount of tape needed for a rectangular box -/
def tape_for_rect_box (length width : ℕ) : ℕ := 2 * width + length

/-- The amount of tape needed for a square box -/
def tape_for_square_box (side : ℕ) : ℕ := 3 * side

/-- The total amount of tape needed for multiple boxes -/
def total_tape_needed (rect_boxes square_boxes : ℕ) (rect_length rect_width square_side : ℕ) : ℕ :=
  rect_boxes * tape_for_rect_box rect_length rect_width +
  square_boxes * tape_for_square_box square_side

theorem total_tape_theorem :
  total_tape_needed 5 2 30 15 40 = 540 :=
by sorry

end NUMINAMATH_CALUDE_total_tape_theorem_l1289_128907


namespace NUMINAMATH_CALUDE_sqrt_54_minus_4_bounds_l1289_128969

theorem sqrt_54_minus_4_bounds : 3 < Real.sqrt 54 - 4 ∧ Real.sqrt 54 - 4 < 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_54_minus_4_bounds_l1289_128969


namespace NUMINAMATH_CALUDE_function_inequality_l1289_128924

theorem function_inequality (f : ℝ → ℝ) (a : ℝ) : 
  (∀ x ≥ 1, f x = x * Real.log x) →
  (∀ x ≥ 1, f x ≥ a * x - 1) →
  a ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_function_inequality_l1289_128924


namespace NUMINAMATH_CALUDE_inequality_proof_l1289_128934

theorem inequality_proof (e : ℝ) (h : e > 0) : 
  (1 : ℝ) / e > Real.log ((1 + e^2) / e^2) ∧ 
  Real.log ((1 + e^2) / e^2) > 1 / (1 + e^2) :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1289_128934


namespace NUMINAMATH_CALUDE_john_remaining_money_l1289_128931

/-- Converts a base 8 number to base 10 --/
def base8_to_base10 (n : ℕ) : ℕ := sorry

/-- The amount John has saved in base 8 --/
def john_savings : ℕ := 5555

/-- The cost of the round-trip airline ticket in base 10 --/
def ticket_cost : ℕ := 1200

/-- The amount John will have left after buying the ticket --/
def remaining_money : ℕ := base8_to_base10 john_savings - ticket_cost

theorem john_remaining_money :
  remaining_money = 1725 := by sorry

end NUMINAMATH_CALUDE_john_remaining_money_l1289_128931


namespace NUMINAMATH_CALUDE_middle_number_problem_l1289_128921

theorem middle_number_problem (a b c : ℕ) : 
  a < b ∧ b < c ∧ 
  a + b = 16 ∧ 
  a + c = 21 ∧ 
  b + c = 27 → 
  b = 11 := by
sorry

end NUMINAMATH_CALUDE_middle_number_problem_l1289_128921


namespace NUMINAMATH_CALUDE_gene_mutation_not_valid_reason_l1289_128915

/-- Represents a genotype --/
inductive Genotype
  | AA
  | Aa
  | BB
  | Bb
  | AaBB
  | AaBb
  | AAB
  | AaB
  | AABb

/-- Represents possible reasons for missing genes --/
inductive MissingGeneReason
  | GeneMutation
  | ChromosomeNumberVariation
  | ChromosomeStructureVariation
  | MaleSexLinked

/-- Defines the genotypes of individuals A and B --/
def individualA : Genotype := Genotype.AaB
def individualB : Genotype := Genotype.AABb

/-- Determines if a reason is valid for explaining the missing gene --/
def isValidReason (reason : MissingGeneReason) (genotypeA : Genotype) (genotypeB : Genotype) : Prop :=
  match reason with
  | MissingGeneReason.GeneMutation => False
  | _ => True

/-- Theorem stating that gene mutation is not a valid reason for the missing gene --/
theorem gene_mutation_not_valid_reason :
  ¬(isValidReason MissingGeneReason.GeneMutation individualA individualB) := by
  sorry


end NUMINAMATH_CALUDE_gene_mutation_not_valid_reason_l1289_128915


namespace NUMINAMATH_CALUDE_opposite_of_neg_five_l1289_128984

/-- The opposite of a number is the number that, when added to the original number, results in zero. -/
def opposite (a : ℤ) : ℤ := -a

/-- Theorem: The opposite of -5 is 5. -/
theorem opposite_of_neg_five : opposite (-5) = 5 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_neg_five_l1289_128984


namespace NUMINAMATH_CALUDE_mary_initial_money_l1289_128990

def initial_money : ℕ := 58
def pie_cost : ℕ := 6
def remaining_money : ℕ := 52

theorem mary_initial_money : 
  initial_money = remaining_money + pie_cost :=
by sorry

end NUMINAMATH_CALUDE_mary_initial_money_l1289_128990


namespace NUMINAMATH_CALUDE_no_k_for_always_negative_quadratic_l1289_128961

theorem no_k_for_always_negative_quadratic :
  ¬ ∃ k : ℝ, ∀ x : ℝ, x^2 - (k + 4) * x + k - 3 < 0 := by
  sorry

end NUMINAMATH_CALUDE_no_k_for_always_negative_quadratic_l1289_128961


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1289_128963

theorem inequality_solution_set (x : ℝ) : -x + 1 > 7*x - 3 ↔ x < 1/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1289_128963


namespace NUMINAMATH_CALUDE_arithmetic_progression_relationship_l1289_128954

theorem arithmetic_progression_relationship (x y z d : ℝ) : 
  (x + (y - z) ≠ y + (z - x) ∧ 
   y + (z - x) ≠ z + (x - y) ∧ 
   x + (y - z) ≠ z + (x - y)) →
  (x + (y - z) ≠ 0 ∧ y + (z - x) ≠ 0 ∧ z + (x - y) ≠ 0) →
  (y + (z - x)) - (x + (y - z)) = d →
  (z + (x - y)) - (y + (z - x)) = d →
  (x = y + d / 2 ∧ z = y + d) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_relationship_l1289_128954


namespace NUMINAMATH_CALUDE_four_vertex_cycle_exists_l1289_128940

/-- A graph with n ≥ 4 vertices where each vertex has degree between 1 and n-2 (inclusive) --/
structure CompanyGraph (n : ℕ) where
  (vertices : Finset (Fin n))
  (edges : Finset (Fin n × Fin n))
  (h1 : n ≥ 4)
  (h2 : ∀ v ∈ vertices, 1 ≤ (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card)
  (h3 : ∀ v ∈ vertices, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card ≤ n - 2)
  (h4 : ∀ e ∈ edges, e.1 ∈ vertices ∧ e.2 ∈ vertices)
  (h5 : ∀ e ∈ edges, (e.2, e.1) ∈ edges)  -- Knowledge is mutual

/-- A cycle of four vertices in the graph --/
structure FourVertexCycle (n : ℕ) (G : CompanyGraph n) where
  (v1 v2 v3 v4 : Fin n)
  (h1 : v1 ∈ G.vertices ∧ v2 ∈ G.vertices ∧ v3 ∈ G.vertices ∧ v4 ∈ G.vertices)
  (h2 : (v1, v2) ∈ G.edges ∧ (v2, v3) ∈ G.edges ∧ (v3, v4) ∈ G.edges ∧ (v4, v1) ∈ G.edges)
  (h3 : (v1, v3) ∉ G.edges ∧ (v2, v4) ∉ G.edges)

/-- The main theorem --/
theorem four_vertex_cycle_exists (n : ℕ) (G : CompanyGraph n) : 
  ∃ c : FourVertexCycle n G, True :=
sorry

end NUMINAMATH_CALUDE_four_vertex_cycle_exists_l1289_128940


namespace NUMINAMATH_CALUDE_cubic_equation_one_root_strategy_l1289_128964

theorem cubic_equation_one_root_strategy :
  ∃ (strategy : ℝ → ℝ → ℝ),
    ∀ (a b c : ℝ),
      ∃ (root : ℝ),
        (root^3 + a*root^2 + b*root + c = 0) ∧
        (∀ x : ℝ, x^3 + a*x^2 + b*x + c = 0 → x = root) :=
sorry

end NUMINAMATH_CALUDE_cubic_equation_one_root_strategy_l1289_128964


namespace NUMINAMATH_CALUDE_ronald_banana_count_l1289_128946

/-- The number of times Ronald went to the store last month -/
def store_visits : ℕ := 2

/-- The number of bananas Ronald buys each time he goes to the store -/
def bananas_per_visit : ℕ := 10

/-- The total number of bananas Ronald bought last month -/
def total_bananas : ℕ := store_visits * bananas_per_visit

theorem ronald_banana_count : total_bananas = 20 := by
  sorry

end NUMINAMATH_CALUDE_ronald_banana_count_l1289_128946


namespace NUMINAMATH_CALUDE_max_soda_bottles_problem_l1289_128930

/-- Represents the maximum number of soda bottles that can be consumed given a certain amount of money, cost per bottle, and exchange rate for empty bottles. -/
def max_soda_bottles (total_money : ℚ) (cost_per_bottle : ℚ) (exchange_rate : ℕ) : ℕ :=
  sorry

/-- Theorem stating that given 30 yuan, a soda cost of 2.5 yuan per bottle, and the ability to exchange 3 empty bottles for 1 new bottle, the maximum number of soda bottles that can be consumed is 18. -/
theorem max_soda_bottles_problem :
  max_soda_bottles 30 2.5 3 = 18 :=
sorry

end NUMINAMATH_CALUDE_max_soda_bottles_problem_l1289_128930


namespace NUMINAMATH_CALUDE_isosceles_triangle_largest_angle_l1289_128929

theorem isosceles_triangle_largest_angle (α β γ : ℝ) : 
  α + β + γ = 180 →  -- Sum of angles in a triangle is 180°
  α = β →            -- The triangle is isosceles (two angles are equal)
  α = 50 →           -- One of the equal angles is 50°
  max α (max β γ) = 80 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_largest_angle_l1289_128929


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l1289_128957

theorem right_triangle_perimeter : ∀ (a b c : ℕ),
  a > 0 → b > 0 → c > 0 →
  a = 11 →
  a * a + b * b = c * c →
  a + b + c = 132 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l1289_128957


namespace NUMINAMATH_CALUDE_multiples_properties_l1289_128960

theorem multiples_properties (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 4 * k) 
  (hb : ∃ m : ℤ, b = 8 * m) : 
  (∃ n : ℤ, b = 4 * n) ∧ 
  (∃ p : ℤ, a - b = 4 * p) ∧ 
  (∃ q : ℤ, a - b = 2 * q) := by
sorry

end NUMINAMATH_CALUDE_multiples_properties_l1289_128960


namespace NUMINAMATH_CALUDE_parallelogram_product_l1289_128906

structure Parallelogram where
  EF : ℝ
  FG : ℝ → ℝ
  GH : ℝ → ℝ
  HE : ℝ
  x : ℝ
  z : ℝ
  h_EF : EF = 46
  h_FG : FG z = 4 * z^3 + 1
  h_GH : GH x = 3 * x + 6
  h_HE : HE = 35
  h_opposite_sides_equal : EF = GH x ∧ FG z = HE

theorem parallelogram_product (p : Parallelogram) :
  p.x * p.z = (40/3) * Real.rpow 8.5 (1/3) := by sorry

end NUMINAMATH_CALUDE_parallelogram_product_l1289_128906


namespace NUMINAMATH_CALUDE_rebecca_eggs_l1289_128923

/-- The number of groups Rebecca wants to split her eggs into -/
def num_groups : ℕ := 4

/-- The number of eggs in each group -/
def eggs_per_group : ℕ := 2

/-- Theorem: Rebecca has 8 eggs in total -/
theorem rebecca_eggs : num_groups * eggs_per_group = 8 := by
  sorry

end NUMINAMATH_CALUDE_rebecca_eggs_l1289_128923


namespace NUMINAMATH_CALUDE_camel_cost_l1289_128948

/-- The cost of animals in rupees -/
structure AnimalCosts where
  camel : ℝ
  horse : ℝ
  ox : ℝ
  elephant : ℝ

/-- The conditions given in the problem -/
def problem_conditions (costs : AnimalCosts) : Prop :=
  10 * costs.camel = 24 * costs.horse ∧
  16 * costs.horse = 4 * costs.ox ∧
  6 * costs.ox = 4 * costs.elephant ∧
  10 * costs.elephant = 140000

/-- The theorem stating that under the given conditions, a camel costs 5600 rupees -/
theorem camel_cost (costs : AnimalCosts) : 
  problem_conditions costs → costs.camel = 5600 := by
  sorry

end NUMINAMATH_CALUDE_camel_cost_l1289_128948


namespace NUMINAMATH_CALUDE_euler_family_mean_age_l1289_128904

def euler_family_children : ℕ := 7
def girls_aged_8 : ℕ := 4
def boys_aged_11 : ℕ := 2
def girl_aged_16 : ℕ := 1

def total_age : ℕ := girls_aged_8 * 8 + boys_aged_11 * 11 + girl_aged_16 * 16

theorem euler_family_mean_age :
  (total_age : ℚ) / euler_family_children = 10 := by sorry

end NUMINAMATH_CALUDE_euler_family_mean_age_l1289_128904


namespace NUMINAMATH_CALUDE_unique_base_eight_l1289_128983

/-- Converts a list of digits in base b to its decimal representation -/
def toDecimal (digits : List Nat) (b : Nat) : Nat :=
  digits.foldl (fun acc d => acc * b + d) 0

/-- Checks if the equation 243₍ᵦ₎ + 152₍ᵦ₎ = 415₍ᵦ₎ holds for a given base b -/
def equationHolds (b : Nat) : Prop :=
  toDecimal [2, 4, 3] b + toDecimal [1, 5, 2] b = toDecimal [4, 1, 5] b

theorem unique_base_eight :
  ∃! b, b > 5 ∧ equationHolds b :=
sorry

end NUMINAMATH_CALUDE_unique_base_eight_l1289_128983


namespace NUMINAMATH_CALUDE_book_page_words_l1289_128952

theorem book_page_words (total_pages : ℕ) (words_per_page : ℕ) : 
  total_pages = 150 →
  50 ≤ words_per_page →
  words_per_page ≤ 150 →
  (total_pages * words_per_page) % 221 = 217 →
  words_per_page = 135 := by
sorry

end NUMINAMATH_CALUDE_book_page_words_l1289_128952


namespace NUMINAMATH_CALUDE_arctan_equality_l1289_128993

theorem arctan_equality : 4 * Real.arctan (1/5) - Real.arctan (1/239) = π/4 := by
  sorry

end NUMINAMATH_CALUDE_arctan_equality_l1289_128993


namespace NUMINAMATH_CALUDE_largest_integer_in_interval_l1289_128986

theorem largest_integer_in_interval : 
  ∃ (x : ℤ), (1/4 : ℚ) < (x : ℚ)/6 ∧ (x : ℚ)/6 < 7/9 ∧ 
  ∀ (y : ℤ), ((1/4 : ℚ) < (y : ℚ)/6 ∧ (y : ℚ)/6 < 7/9) → y ≤ x :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_in_interval_l1289_128986


namespace NUMINAMATH_CALUDE_total_books_l1289_128901

theorem total_books (tim_books sam_books alice_books : ℕ) 
  (h1 : tim_books = 44)
  (h2 : sam_books = 52)
  (h3 : alice_books = 38) :
  tim_books + sam_books + alice_books = 134 := by
  sorry

end NUMINAMATH_CALUDE_total_books_l1289_128901


namespace NUMINAMATH_CALUDE_polynomial_value_constraint_l1289_128925

theorem polynomial_value_constraint 
  (P : ℤ → ℤ) 
  (h_poly : ∀ x y : ℤ, (P x - P y) ∣ (x - y))
  (h_distinct : ∃ a b c : ℤ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ P a = 2 ∧ P b = 2 ∧ P c = 2) :
  ∀ x : ℤ, P x ≠ 3 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_value_constraint_l1289_128925


namespace NUMINAMATH_CALUDE_arithmetic_arrangement_l1289_128991

theorem arithmetic_arrangement :
  (1 / 8 * 1 / 9 * 1 / 28 = 1 / 2016) ∧
  ((1 / 8 - 1 / 9) * 1 / 28 = 1 / 2016) := by sorry

end NUMINAMATH_CALUDE_arithmetic_arrangement_l1289_128991


namespace NUMINAMATH_CALUDE_polynomial_roots_equivalence_l1289_128914

theorem polynomial_roots_equivalence :
  let p (x : ℝ) := 7 * x^4 - 48 * x^3 + 93 * x^2 - 48 * x + 7
  let y (x : ℝ) := x + 2 / x
  let q (y : ℝ) := 7 * y^2 - 48 * y + 47
  ∀ x : ℝ, x ≠ 0 →
    (p x = 0 ↔ ∃ y : ℝ, q y = 0 ∧ (x + 2 / x = y ∨ x + 2 / x = y)) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_roots_equivalence_l1289_128914


namespace NUMINAMATH_CALUDE_points_separated_by_line_l1289_128982

/-- Definition of a line in 2D space --/
def Line (a b c : ℝ) : ℝ × ℝ → ℝ :=
  fun p => a * p.1 + b * p.2 + c

/-- Definition of η for two points with respect to a line --/
def eta (l : ℝ × ℝ → ℝ) (p1 p2 : ℝ × ℝ) : ℝ :=
  (l p1) * (l p2)

/-- Definition of two points being separated by a line --/
def separatedByLine (l : ℝ × ℝ → ℝ) (p1 p2 : ℝ × ℝ) : Prop :=
  eta l p1 p2 < 0

/-- Theorem: Points A(1,2) and B(-1,0) are separated by the line x+y-1=0 --/
theorem points_separated_by_line :
  let l := Line 1 1 (-1)
  let A := (1, 2)
  let B := (-1, 0)
  separatedByLine l A B := by
  sorry


end NUMINAMATH_CALUDE_points_separated_by_line_l1289_128982


namespace NUMINAMATH_CALUDE_bedroom_set_price_l1289_128916

def original_price : ℝ := 2000
def gift_card : ℝ := 200
def first_discount_rate : ℝ := 0.15
def second_discount_rate : ℝ := 0.10

def final_price : ℝ :=
  let price_after_first_discount := original_price * (1 - first_discount_rate)
  let price_after_second_discount := price_after_first_discount * (1 - second_discount_rate)
  price_after_second_discount - gift_card

theorem bedroom_set_price : final_price = 1330 := by
  sorry

end NUMINAMATH_CALUDE_bedroom_set_price_l1289_128916


namespace NUMINAMATH_CALUDE_train_length_l1289_128926

/-- The length of a train given specific crossing times and platform length -/
theorem train_length (platform_cross_time signal_cross_time : ℝ) (platform_length : ℝ) : 
  platform_cross_time = 54 →
  signal_cross_time = 18 →
  platform_length = 600.0000000000001 →
  ∃ (train_length : ℝ), train_length = 300.00000000000005 :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_l1289_128926


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_expression_evaluation_at_3_l1289_128933

theorem expression_simplification_and_evaluation (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) :
  ((2 * x - 1) / (x - 2) - 1) / ((x + 1) / (x^2 - 4)) = x + 2 :=
by sorry

theorem expression_evaluation_at_3 :
  let x : ℝ := 3
  ((2 * x - 1) / (x - 2) - 1) / ((x + 1) / (x^2 - 4)) = 5 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_expression_evaluation_at_3_l1289_128933


namespace NUMINAMATH_CALUDE_rotary_club_omelet_eggs_rotary_club_omelet_eggs_proof_l1289_128943

/-- Calculate the number of eggs needed for the Rotary Club Omelet Breakfast -/
theorem rotary_club_omelet_eggs : ℕ :=
  let small_children := 53
  let older_children := 35
  let adults := 75
  let seniors := 37
  let small_children_omelets := 0.5
  let older_children_omelets := 1
  let adults_omelets := 2
  let seniors_omelets := 1.5
  let extra_omelets := 25
  let eggs_per_omelet := 2

  let total_omelets := small_children * small_children_omelets +
                       older_children * older_children_omelets +
                       adults * adults_omelets +
                       seniors * seniors_omelets +
                       extra_omelets

  let total_eggs := total_omelets * eggs_per_omelet

  584

theorem rotary_club_omelet_eggs_proof : rotary_club_omelet_eggs = 584 := by
  sorry

end NUMINAMATH_CALUDE_rotary_club_omelet_eggs_rotary_club_omelet_eggs_proof_l1289_128943


namespace NUMINAMATH_CALUDE_folded_rectangle_perimeter_l1289_128927

/-- A rectangle ABCD with a fold from A to A' on CD creating a crease EF -/
structure FoldedRectangle where
  -- Length of AE
  ae : ℝ
  -- Length of EB
  eb : ℝ
  -- Length of CF
  cf : ℝ

/-- The perimeter of the folded rectangle -/
def perimeter (r : FoldedRectangle) : ℝ :=
  2 * (r.ae + r.eb + r.cf + (r.ae + r.eb - r.cf))

/-- Theorem stating that the perimeter of the specific folded rectangle is 82 -/
theorem folded_rectangle_perimeter :
  let r : FoldedRectangle := { ae := 3, eb := 15, cf := 8 }
  perimeter r = 82 := by sorry

end NUMINAMATH_CALUDE_folded_rectangle_perimeter_l1289_128927


namespace NUMINAMATH_CALUDE_reading_difference_l1289_128989

/-- The number of pages Liza reads in one hour -/
def liza_rate : ℕ := 20

/-- The number of pages Suzie reads in one hour -/
def suzie_rate : ℕ := 15

/-- The number of hours they read -/
def hours : ℕ := 3

/-- The difference in pages read between Liza and Suzie over the given time period -/
def page_difference : ℕ := liza_rate * hours - suzie_rate * hours

theorem reading_difference : page_difference = 15 := by
  sorry

end NUMINAMATH_CALUDE_reading_difference_l1289_128989


namespace NUMINAMATH_CALUDE_solution_is_two_l1289_128947

-- Define the base 10 logarithm
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the equation
def equation (x : ℝ) : Prop :=
  log10 (x^2 - 3) = log10 (3*x - 5) ∧ x^2 - 3 > 0 ∧ 3*x - 5 > 0

-- Theorem stating that 2 is the solution to the equation
theorem solution_is_two : equation 2 := by sorry

end NUMINAMATH_CALUDE_solution_is_two_l1289_128947


namespace NUMINAMATH_CALUDE_paper_area_problem_l1289_128935

theorem paper_area_problem (L : ℝ) : 
  2 * (11 * L) = 2 * (8.5 * 11) + 100 ↔ L = 287 / 22 := by sorry

end NUMINAMATH_CALUDE_paper_area_problem_l1289_128935


namespace NUMINAMATH_CALUDE_entrance_exam_marks_l1289_128937

/-- Proves that the number of marks awarded for each correct answer is 3 -/
theorem entrance_exam_marks : 
  ∀ (total_questions correct_answers total_marks : ℕ) 
    (wrong_answer_penalty : ℤ),
  total_questions = 70 →
  correct_answers = 27 →
  total_marks = 38 →
  wrong_answer_penalty = -1 →
  ∃ (marks_per_correct_answer : ℕ),
    marks_per_correct_answer * correct_answers + 
    wrong_answer_penalty * (total_questions - correct_answers) = total_marks ∧
    marks_per_correct_answer = 3 :=
by sorry

end NUMINAMATH_CALUDE_entrance_exam_marks_l1289_128937


namespace NUMINAMATH_CALUDE_smallest_square_containing_circle_l1289_128968

theorem smallest_square_containing_circle (r : ℝ) (h : r = 5) :
  (2 * r) ^ 2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_containing_circle_l1289_128968


namespace NUMINAMATH_CALUDE_cookie_box_count_l1289_128999

/-- The number of cookies in a bag -/
def cookies_per_bag : ℕ := 7

/-- The number of cookies in a box -/
def cookies_per_box : ℕ := 12

/-- The number of bags used for comparison -/
def num_bags : ℕ := 9

/-- The additional number of cookies in boxes compared to bags -/
def extra_cookies : ℕ := 33

/-- The number of boxes -/
def num_boxes : ℕ := 8

theorem cookie_box_count :
  num_boxes * cookies_per_box = num_bags * cookies_per_bag + extra_cookies :=
by sorry

end NUMINAMATH_CALUDE_cookie_box_count_l1289_128999


namespace NUMINAMATH_CALUDE_multiplicative_inverse_123_mod_400_l1289_128958

theorem multiplicative_inverse_123_mod_400 : ∃ a : ℕ, a < 400 ∧ (123 * a) % 400 = 1 :=
by
  use 387
  sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_123_mod_400_l1289_128958


namespace NUMINAMATH_CALUDE_can_form_triangle_l1289_128941

theorem can_form_triangle (a b c : ℝ) (ha : a = 6) (hb : b = 8) (hc : c = 12) :
  a + b > c ∧ b + c > a ∧ c + a > b := by
  sorry

end NUMINAMATH_CALUDE_can_form_triangle_l1289_128941


namespace NUMINAMATH_CALUDE_fifth_number_13th_row_is_715_l1289_128911

/-- The fifth number in the 13th row of Pascal's triangle -/
def fifth_number_13th_row : ℕ :=
  Nat.choose 13 4

/-- Theorem stating that the fifth number in the 13th row of Pascal's triangle is 715 -/
theorem fifth_number_13th_row_is_715 : fifth_number_13th_row = 715 := by
  sorry

end NUMINAMATH_CALUDE_fifth_number_13th_row_is_715_l1289_128911


namespace NUMINAMATH_CALUDE_range_of_m_l1289_128992

def elliptical_region (x y : ℝ) : Prop := x^2 / 4 + y^2 ≤ 1

def dividing_lines (x y m : ℝ) : Prop :=
  (y = Real.sqrt 2 * x) ∨ (y = -Real.sqrt 2 * x) ∨ (x = m)

def valid_coloring (n : ℕ) : Prop :=
  n = 720 ∧ ∃ (colors : Fin 6 → Type) (parts : Type) (coloring : parts → Fin 6),
    ∀ (p1 p2 : parts), p1 ≠ p2 → coloring p1 ≠ coloring p2

theorem range_of_m :
  ∀ m : ℝ,
    (∀ x y : ℝ, elliptical_region x y → dividing_lines x y m → valid_coloring 720) ↔
    ((-2 < m ∧ m ≤ -2/3) ∨ m = 0 ∨ (2/3 ≤ m ∧ m < 2)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1289_128992


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l1289_128995

theorem complex_fraction_evaluation :
  let i : ℂ := Complex.I
  (3 + i) / (1 + i) = 2 - i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l1289_128995


namespace NUMINAMATH_CALUDE_calculate_birth_rate_l1289_128917

/-- Given a death rate and population increase rate, calculate the birth rate. -/
theorem calculate_birth_rate (death_rate : ℝ) (population_increase_rate : ℝ) : 
  death_rate = 11 → population_increase_rate = 2.1 → 
  ∃ (birth_rate : ℝ), birth_rate = 32 ∧ birth_rate - death_rate = population_increase_rate / 100 * 1000 := by
  sorry

#check calculate_birth_rate

end NUMINAMATH_CALUDE_calculate_birth_rate_l1289_128917


namespace NUMINAMATH_CALUDE_special_triangle_side_length_l1289_128944

/-- An equilateral triangle with a special interior point -/
structure SpecialTriangle where
  /-- The side length of the equilateral triangle -/
  t : ℝ
  /-- The distance from vertex D to point Q -/
  DQ : ℝ
  /-- The distance from vertex E to point Q -/
  EQ : ℝ
  /-- The distance from vertex F to point Q -/
  FQ : ℝ
  /-- The triangle is equilateral -/
  equilateral : t > 0
  /-- The point Q is inside the triangle -/
  interior : DQ > 0 ∧ EQ > 0 ∧ FQ > 0
  /-- The distances from Q to the vertices -/
  distances : DQ = 2 ∧ EQ = Real.sqrt 5 ∧ FQ = 3

/-- The theorem stating that the side length of the special triangle is 2√3 -/
theorem special_triangle_side_length (T : SpecialTriangle) : T.t = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_side_length_l1289_128944


namespace NUMINAMATH_CALUDE_modulus_of_z_l1289_128980

open Complex

theorem modulus_of_z (i : ℂ) (h : i * i = -1) : 
  abs (2 * i + 2 / (1 + i)) = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_modulus_of_z_l1289_128980


namespace NUMINAMATH_CALUDE_grid_block_selection_l1289_128956

theorem grid_block_selection (n : ℕ) (k : ℕ) : 
  n = 7 → k = 4 → (n.choose k) * (n.choose k) * k.factorial = 29400 := by
  sorry

end NUMINAMATH_CALUDE_grid_block_selection_l1289_128956


namespace NUMINAMATH_CALUDE_value_calculation_l1289_128936

theorem value_calculation (number : ℕ) (value : ℕ) 
  (h1 : value = 5 * number) 
  (h2 : number = 20) : 
  value = 100 := by
sorry

end NUMINAMATH_CALUDE_value_calculation_l1289_128936


namespace NUMINAMATH_CALUDE_pencils_bought_on_monday_l1289_128955

theorem pencils_bought_on_monday (P : ℕ) : P = 20 :=
  by
  -- Define the number of pencils bought on Tuesday
  let tuesday_pencils := 18

  -- Define the number of pencils bought on Wednesday
  let wednesday_pencils := 3 * tuesday_pencils

  -- Define the total number of pencils
  let total_pencils := 92

  -- Assert that the sum of pencils from all days equals the total
  have h : P + tuesday_pencils + wednesday_pencils = total_pencils := by sorry

  -- Prove that P equals 20
  sorry

end NUMINAMATH_CALUDE_pencils_bought_on_monday_l1289_128955


namespace NUMINAMATH_CALUDE_mean_of_cubic_solutions_l1289_128949

theorem mean_of_cubic_solutions (x : ℝ) : 
  (x^3 + 3*x^2 - 44*x = 0) → 
  (∃ s : Finset ℝ, (∀ y ∈ s, y^3 + 3*y^2 - 44*y = 0) ∧ 
                   (s.card = 3) ∧ 
                   (s.sum id / s.card = -1)) :=
by sorry

end NUMINAMATH_CALUDE_mean_of_cubic_solutions_l1289_128949


namespace NUMINAMATH_CALUDE_nancy_albums_l1289_128953

theorem nancy_albums (total_pictures : ℕ) (first_album : ℕ) (pics_per_album : ℕ) 
  (h1 : total_pictures = 51)
  (h2 : first_album = 11)
  (h3 : pics_per_album = 5) :
  (total_pictures - first_album) / pics_per_album = 8 := by
  sorry

end NUMINAMATH_CALUDE_nancy_albums_l1289_128953


namespace NUMINAMATH_CALUDE_products_not_equal_l1289_128919

def is_valid_table (t : Fin 10 → Fin 10 → ℕ) : Prop :=
  ∀ i j, 102 ≤ t i j ∧ t i j ≤ 201 ∧ (∀ i' j', (i ≠ i' ∨ j ≠ j') → t i j ≠ t i' j')

def row_product (t : Fin 10 → Fin 10 → ℕ) (i : Fin 10) : ℕ :=
  (Finset.univ.prod fun j => t i j)

def col_product (t : Fin 10 → Fin 10 → ℕ) (j : Fin 10) : ℕ :=
  (Finset.univ.prod fun i => t i j)

def row_products (t : Fin 10 → Fin 10 → ℕ) : Finset ℕ :=
  Finset.image (row_product t) Finset.univ

def col_products (t : Fin 10 → Fin 10 → ℕ) : Finset ℕ :=
  Finset.image (col_product t) Finset.univ

theorem products_not_equal :
  ∀ t : Fin 10 → Fin 10 → ℕ, is_valid_table t → row_products t ≠ col_products t :=
sorry

end NUMINAMATH_CALUDE_products_not_equal_l1289_128919


namespace NUMINAMATH_CALUDE_water_displaced_squared_value_l1289_128974

/-- The square of the volume of water displaced by a fully submerged cube in a cylindrical barrel -/
def water_displaced_squared (cube_side : ℝ) (barrel_radius : ℝ) (barrel_height : ℝ) : ℝ :=
  (cube_side ^ 3) ^ 2

/-- Theorem stating that the square of the volume of water displaced by a fully submerged cube
    with side length 7 feet in a cylindrical barrel with radius 5 feet and height 15 feet is 117649 cubic feet -/
theorem water_displaced_squared_value :
  water_displaced_squared 7 5 15 = 117649 := by
  sorry

end NUMINAMATH_CALUDE_water_displaced_squared_value_l1289_128974


namespace NUMINAMATH_CALUDE_range_of_a_l1289_128912

-- Define the condition that x^2 > 1 is necessary but not sufficient for x < a
def necessary_not_sufficient (a : ℝ) : Prop :=
  (∀ x : ℝ, x < a → x^2 > 1) ∧ 
  (∃ x : ℝ, x^2 > 1 ∧ x ≥ a)

-- Theorem stating the range of values for a
theorem range_of_a (a : ℝ) : 
  necessary_not_sufficient a ↔ a ≤ -1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1289_128912


namespace NUMINAMATH_CALUDE_smallest_integers_difference_l1289_128908

theorem smallest_integers_difference : ∃ n₁ n₂ : ℕ,
  (∀ k : ℕ, 2 ≤ k → k ≤ 12 → n₁ % k = 1) ∧
  (∀ k : ℕ, 2 ≤ k → k ≤ 12 → n₂ % k = 1) ∧
  n₁ > 1 ∧ n₂ > 1 ∧ n₂ > n₁ ∧
  (∀ m : ℕ, m > 1 → (∀ k : ℕ, 2 ≤ k → k ≤ 12 → m % k = 1) → m ≥ n₁) ∧
  n₂ - n₁ = 27720 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integers_difference_l1289_128908


namespace NUMINAMATH_CALUDE_max_product_sum_180_l1289_128996

theorem max_product_sum_180 : 
  ∀ a b : ℤ, a + b = 180 → a * b ≤ 8100 := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_180_l1289_128996


namespace NUMINAMATH_CALUDE_no_real_solutions_l1289_128972

theorem no_real_solutions :
  ∀ x : ℝ, (x^10 + 1) * (x^8 + x^6 + x^4 + x^2 + 1) ≠ 12 * x^9 :=
by sorry

end NUMINAMATH_CALUDE_no_real_solutions_l1289_128972


namespace NUMINAMATH_CALUDE_black_white_area_ratio_l1289_128962

/-- The ratio of black to white areas in concentric circles -/
theorem black_white_area_ratio :
  let r₁ : ℝ := 2
  let r₂ : ℝ := 4
  let r₃ : ℝ := 6
  let r₄ : ℝ := 8
  let black_area := (r₂^2 - r₁^2) * Real.pi + (r₄^2 - r₃^2) * Real.pi
  let white_area := r₁^2 * Real.pi + (r₃^2 - r₂^2) * Real.pi
  black_area / white_area = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_black_white_area_ratio_l1289_128962
