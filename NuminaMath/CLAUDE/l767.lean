import Mathlib

namespace NUMINAMATH_CALUDE_min_value_theorem_l767_76748

/-- The function f(x) defined as |x+a| + |x+3a| -/
def f (a : ℝ) (x : ℝ) : ℝ := |x + a| + |x + 3*a|

/-- The theorem stating the minimum value of 1/m^2 + n^2 given conditions -/
theorem min_value_theorem (a m n : ℝ) :
  (∃ (x : ℝ), ∀ (y : ℝ), f a y ≥ f a x) →  -- minimum of f(x) exists
  (∀ (x : ℝ), f a x ≥ 2) →  -- minimum value of f(x) is 2
  (a - m) * (a + m) = 4 / n^2 →  -- given condition
  (∃ (k : ℝ), ∀ (p q : ℝ), 1 / p^2 + q^2 ≥ k ∧ (1 / m^2 + n^2 = k)) →  -- minimum of 1/m^2 + n^2 exists
  (1 / m^2 + n^2 = 9)  -- conclusion: minimum value is 9
:= by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l767_76748


namespace NUMINAMATH_CALUDE_arithmetic_geometric_harmonic_means_l767_76772

theorem arithmetic_geometric_harmonic_means (p q r : ℝ) : 
  ((p + q) / 2 = 10) →
  (Real.sqrt (p * q) = 12) →
  ((q + r) / 2 = 26) →
  (2 / (1 / p + 1 / r) = 8) →
  (r - p = 32) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_harmonic_means_l767_76772


namespace NUMINAMATH_CALUDE_sin_150_degrees_l767_76703

theorem sin_150_degrees : Real.sin (150 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_150_degrees_l767_76703


namespace NUMINAMATH_CALUDE_intercepts_congruence_l767_76722

/-- Proof of x-intercept and y-intercept properties for the congruence 6x ≡ 5y - 1 (mod 28) --/
theorem intercepts_congruence :
  ∃ (x₀ y₀ : ℕ),
    x₀ < 28 ∧ y₀ < 28 ∧
    (6 * x₀) % 28 = 27 ∧
    (5 * y₀) % 28 = 1 ∧
    x₀ + y₀ = 20 := by
  sorry


end NUMINAMATH_CALUDE_intercepts_congruence_l767_76722


namespace NUMINAMATH_CALUDE_michael_passes_donovan_in_four_laps_l767_76751

/-- Represents the race conditions and calculates the number of laps for Michael to pass Donovan -/
def raceLaps (trackLength : ℕ) (donovanNormalTime : ℕ) (michaelNormalTime : ℕ) 
              (obstacles : ℕ) (donovanObstacleTime : ℕ) (michaelObstacleTime : ℕ) : ℕ :=
  let donovanLapTime := donovanNormalTime + obstacles * donovanObstacleTime
  let michaelLapTime := michaelNormalTime + obstacles * michaelObstacleTime
  let timeDiffPerLap := donovanLapTime - michaelLapTime
  let lapsToPass := (donovanLapTime + timeDiffPerLap - 1) / timeDiffPerLap
  lapsToPass

/-- Theorem stating that Michael needs 4 laps to pass Donovan under the given conditions -/
theorem michael_passes_donovan_in_four_laps :
  raceLaps 300 45 40 3 10 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_michael_passes_donovan_in_four_laps_l767_76751


namespace NUMINAMATH_CALUDE_product_of_solutions_with_positive_real_part_l767_76798

theorem product_of_solutions_with_positive_real_part (x : ℂ) : 
  (x^8 = -256) → 
  (∃ (S : Finset ℂ), 
    (∀ z ∈ S, z^8 = -256 ∧ z.re > 0) ∧ 
    (∀ z, z^8 = -256 ∧ z.re > 0 → z ∈ S) ∧ 
    (S.prod id = 8)) :=
by sorry

end NUMINAMATH_CALUDE_product_of_solutions_with_positive_real_part_l767_76798


namespace NUMINAMATH_CALUDE_dave_bought_26_tshirts_l767_76793

/-- The number of T-shirts Dave bought -/
def total_tshirts (white_packs blue_packs : ℕ) (white_per_pack blue_per_pack : ℕ) : ℕ :=
  white_packs * white_per_pack + blue_packs * blue_per_pack

/-- Proof that Dave bought 26 T-shirts -/
theorem dave_bought_26_tshirts :
  total_tshirts 3 2 6 4 = 26 := by
  sorry

end NUMINAMATH_CALUDE_dave_bought_26_tshirts_l767_76793


namespace NUMINAMATH_CALUDE_jerrys_painting_time_l767_76765

theorem jerrys_painting_time (fixing_time painting_time mowing_time hourly_rate total_payment : ℝ) :
  fixing_time = 3 * painting_time →
  mowing_time = 6 →
  hourly_rate = 15 →
  total_payment = 570 →
  hourly_rate * (painting_time + fixing_time + mowing_time) = total_payment →
  painting_time = 8 := by
  sorry

end NUMINAMATH_CALUDE_jerrys_painting_time_l767_76765


namespace NUMINAMATH_CALUDE_whistlers_count_l767_76780

/-- The number of whistlers in each of Koby's boxes -/
def whistlers_per_box : ℕ := sorry

/-- The total number of fireworks Koby and Cherie have -/
def total_fireworks : ℕ := 33

/-- The number of boxes Koby has -/
def koby_boxes : ℕ := 2

/-- The number of sparklers in each of Koby's boxes -/
def koby_sparklers_per_box : ℕ := 3

/-- The number of sparklers in Cherie's box -/
def cherie_sparklers : ℕ := 8

/-- The number of whistlers in Cherie's box -/
def cherie_whistlers : ℕ := 9

theorem whistlers_count : whistlers_per_box = 5 := by
  have h1 : total_fireworks = koby_boxes * (koby_sparklers_per_box + whistlers_per_box) + cherie_sparklers + cherie_whistlers := by sorry
  sorry

end NUMINAMATH_CALUDE_whistlers_count_l767_76780


namespace NUMINAMATH_CALUDE_perpendicular_slopes_not_always_negative_one_l767_76795

/-- Two lines in a 2D plane --/
structure Line where
  slope : Option ℝ

/-- Perpendicularity relation between two lines --/
def perpendicular (l₁ l₂ : Line) : Prop :=
  match l₁.slope, l₂.slope with
  | some m₁, some m₂ => m₁ * m₂ = -1
  | none, some m => m = 0
  | some m, none => m = 0
  | none, none => False

/-- Theorem stating that there exist perpendicular lines whose slopes do not multiply to -1 --/
theorem perpendicular_slopes_not_always_negative_one :
  ∃ (l₁ l₂ : Line), perpendicular l₁ l₂ ∧
    (l₁.slope.isNone ∨ l₂.slope.isNone ∨
     ∃ (m₁ m₂ : ℝ), l₁.slope = some m₁ ∧ l₂.slope = some m₂ ∧ m₁ * m₂ ≠ -1) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_slopes_not_always_negative_one_l767_76795


namespace NUMINAMATH_CALUDE_max_value_x_plus_inverse_l767_76717

theorem max_value_x_plus_inverse (x : ℝ) (h : 13 = x^2 + 1/x^2) :
  (∀ y : ℝ, y > 0 → 13 = y^2 + 1/y^2 → x + 1/x ≥ y + 1/y) ∧ x + 1/x = Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_max_value_x_plus_inverse_l767_76717


namespace NUMINAMATH_CALUDE_sum_of_integers_l767_76746

theorem sum_of_integers : (-9) + 18 + 2 + (-1) = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l767_76746


namespace NUMINAMATH_CALUDE_sin_sum_equality_l767_76749

theorem sin_sum_equality : Real.sin (163 * π / 180) * Real.sin (223 * π / 180) + 
                           Real.sin (253 * π / 180) * Real.sin (313 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_equality_l767_76749


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l767_76756

theorem imaginary_part_of_complex_fraction : 
  Complex.im ((1 - Complex.I) / (1 + Complex.I)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l767_76756


namespace NUMINAMATH_CALUDE_cricketer_average_score_l767_76741

theorem cricketer_average_score 
  (total_matches : ℕ) 
  (first_part_matches : ℕ) 
  (overall_average : ℝ) 
  (first_part_average : ℝ) 
  (h1 : total_matches = 12) 
  (h2 : first_part_matches = 8) 
  (h3 : overall_average = 48) 
  (h4 : first_part_average = 40) :
  let last_part_matches := total_matches - first_part_matches
  let total_runs := total_matches * overall_average
  let first_part_runs := first_part_matches * first_part_average
  let last_part_runs := total_runs - first_part_runs
  last_part_runs / last_part_matches = 64 := by
sorry

end NUMINAMATH_CALUDE_cricketer_average_score_l767_76741


namespace NUMINAMATH_CALUDE_grocery_total_l767_76734

/-- The number of cookie packs Lucy bought -/
def cookie_packs : ℕ := 23

/-- The number of cake packs Lucy bought -/
def cake_packs : ℕ := 4

/-- The total number of grocery packs Lucy bought -/
def total_packs : ℕ := cookie_packs + cake_packs

theorem grocery_total : total_packs = 27 := by
  sorry

end NUMINAMATH_CALUDE_grocery_total_l767_76734


namespace NUMINAMATH_CALUDE_f_is_odd_l767_76713

def f (x : ℝ) : ℝ := x^(1/3)

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x := by sorry

end NUMINAMATH_CALUDE_f_is_odd_l767_76713


namespace NUMINAMATH_CALUDE_history_not_statistics_l767_76754

theorem history_not_statistics (total : ℕ) (history : ℕ) (statistics : ℕ) (history_or_statistics : ℕ)
  (h_total : total = 90)
  (h_history : history = 36)
  (h_statistics : statistics = 32)
  (h_history_or_statistics : history_or_statistics = 57) :
  history - (history + statistics - history_or_statistics) = 25 := by
  sorry

end NUMINAMATH_CALUDE_history_not_statistics_l767_76754


namespace NUMINAMATH_CALUDE_georgia_muffins_per_batch_l767_76723

/-- Calculates the number of muffins per batch given the total number of students,
    total batches made, and the number of months. -/
def muffins_per_batch (students : ℕ) (total_batches : ℕ) (months : ℕ) : ℕ :=
  students * months / total_batches

/-- Proves that given 24 students and 36 batches of muffins made in 9 months,
    the number of muffins per batch is 6. -/
theorem georgia_muffins_per_batch :
  muffins_per_batch 24 36 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_georgia_muffins_per_batch_l767_76723


namespace NUMINAMATH_CALUDE_parabola_equation_l767_76797

/-- A parabola with vertex at the origin, axis of symmetry along the x-axis,
    and focus on the line 3x - 4y - 12 = 0 has the equation y² = 16x -/
theorem parabola_equation (x y : ℝ) :
  (∀ a b : ℝ, 3 * a - 4 * b - 12 = 0 → (a = 4 ∧ b = 0)) →
  y^2 = 16 * x := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l767_76797


namespace NUMINAMATH_CALUDE_min_value_of_f_l767_76784

/-- The quadratic expression in x and y with parameter k -/
def f (k x y : ℝ) : ℝ := 9*x^2 - 6*k*x*y + (3*k^2 + 1)*y^2 - 6*x - 6*y + 7

/-- The theorem stating that k = 3 is the unique value for which f has a minimum of 1 -/
theorem min_value_of_f :
  ∃! k : ℝ, (∀ x y : ℝ, f k x y ≥ 1) ∧ (∃ x y : ℝ, f k x y = 1) ∧ k = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l767_76784


namespace NUMINAMATH_CALUDE_parallel_line_y_intercept_l767_76786

/-- A line in the xy-plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Returns true if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop := l1.slope = l2.slope

/-- Returns true if a point (x, y) is on the given line -/
def pointOnLine (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.yIntercept

theorem parallel_line_y_intercept (b : Line) :
  parallel b { slope := -3, yIntercept := 6 } →
  pointOnLine b 4 (-1) →
  b.yIntercept = 11 := by sorry

end NUMINAMATH_CALUDE_parallel_line_y_intercept_l767_76786


namespace NUMINAMATH_CALUDE_angle_terminal_side_set_l767_76776

/-- 
Given an angle α whose terminal side, when rotated counterclockwise by 30°, 
coincides with the terminal side of 120°, the set of all angles β that have 
the same terminal side as α is {β | β = k × 360° + 90°, k ∈ ℤ}.
-/
theorem angle_terminal_side_set (α : Real) 
  (h : α + 30 = 120 + 360 * (⌊(α + 30 - 120) / 360⌋ : ℤ)) :
  {β : Real | ∃ k : ℤ, β = k * 360 + 90} = 
  {β : Real | ∃ k : ℤ, β = k * 360 + α} :=
by sorry


end NUMINAMATH_CALUDE_angle_terminal_side_set_l767_76776


namespace NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l767_76706

/-- The area of the unfolded lateral surface of a cylinder with base radius 2 and height 2 is 8π. -/
theorem cylinder_lateral_surface_area : 
  ∀ (r h : ℝ), r = 2 → h = 2 → 2 * π * r * h = 8 * π :=
by
  sorry

end NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l767_76706


namespace NUMINAMATH_CALUDE_no_solution_implies_a_less_than_two_l767_76769

theorem no_solution_implies_a_less_than_two (a : ℝ) :
  (∀ x : ℝ, |x - 3| + |x - 1| > a) → a < 2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_a_less_than_two_l767_76769


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l767_76720

theorem partial_fraction_decomposition (C D : ℚ) :
  (∀ x : ℚ, x ≠ 7 ∧ x ≠ -2 →
    (5 * x - 3) / (x^2 - 5*x - 14) = C / (x - 7) + D / (x + 2)) →
  C = 32/9 ∧ D = 13/9 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l767_76720


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l767_76716

/-- Given three non-zero real numbers x, y, and z forming a geometric sequence
    x(y-z), y(z-x), and z(y-x), prove that the common ratio q satisfies q^2 - q - 1 = 0 -/
theorem geometric_sequence_common_ratio (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (hseq : ∃ q : ℝ, q ≠ 0 ∧ y * (z - x) = q * (x * (y - z)) ∧ z * (y - x) = q * (y * (z - x))) :
  ∃ q : ℝ, q^2 - q - 1 = 0 ∧ y * (z - x) = q * (x * (y - z)) ∧ z * (y - x) = q * (y * (z - x)) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l767_76716


namespace NUMINAMATH_CALUDE_tan_difference_sum_l767_76796

theorem tan_difference_sum (α β γ : Real) 
  (h1 : Real.tan α = 5)
  (h2 : Real.tan β = 2)
  (h3 : Real.tan γ = 3) :
  Real.tan (α - β + γ) = 18 := by
  sorry

end NUMINAMATH_CALUDE_tan_difference_sum_l767_76796


namespace NUMINAMATH_CALUDE_second_race_length_l767_76727

/-- Given a 100 m race where A beats B by 10 m and C by 13 m, and another race where B beats C by 6 m,
    prove that the length of the second race is 180 meters. -/
theorem second_race_length (vA vB vC : ℝ) (t : ℝ) (h1 : vA * t = 100)
                            (h2 : vB * t = 90) (h3 : vC * t = 87) : 
  ∃ (L : ℝ), L / vB = (L - 6) / vC ∧ L = 180 := by
  sorry

end NUMINAMATH_CALUDE_second_race_length_l767_76727


namespace NUMINAMATH_CALUDE_geometric_sum_value_l767_76721

theorem geometric_sum_value (x : ℝ) (h1 : x^2023 - 3*x + 2 = 0) (h2 : x ≠ 1) :
  x^2022 + x^2021 + x^2020 + x^2019 + x^2018 + x^2017 + x^2016 + x^2015 + x^2014 + x^2013 + 
  x^2012 + x^2011 + x^2010 + x^2009 + x^2008 + x^2007 + x^2006 + x^2005 + x^2004 + x^2003 + 
  x^2002 + x^2001 + x^2000 + x^1999 + x^1998 + x^1997 + x^1996 + x^1995 + x^1994 + x^1993 + 
  x^1992 + x^1991 + x^1990 + x^1989 + x^1988 + x^1987 + x^1986 + x^1985 + x^1984 + x^1983 + 
  x^1982 + x^1981 + x^1980 + x^1979 + x^1978 + x^1977 + x^1976 + x^1975 + x^1974 + x^1973 + 
  -- ... (continuing the pattern)
  x^22 + x^21 + x^20 + x^19 + x^18 + x^17 + x^16 + x^15 + x^14 + x^13 + 
  x^12 + x^11 + x^10 + x^9 + x^8 + x^7 + x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sum_value_l767_76721


namespace NUMINAMATH_CALUDE_tv_show_minor_characters_l767_76733

/-- The problem of determining the number of minor characters in a TV show. -/
theorem tv_show_minor_characters :
  let main_characters : ℕ := 5
  let minor_character_pay : ℕ := 15000
  let main_character_pay : ℕ := 3 * minor_character_pay
  let total_pay : ℕ := 285000
  let minor_characters : ℕ := (total_pay - main_characters * main_character_pay) / minor_character_pay
  minor_characters = 4 := by sorry

end NUMINAMATH_CALUDE_tv_show_minor_characters_l767_76733


namespace NUMINAMATH_CALUDE_solution_satisfies_system_l767_76745

theorem solution_satisfies_system :
  let solutions : List (Int × Int) := [(-3, -1), (-1, -3), (1, 3), (3, 1)]
  ∀ (x y : Int), (x, y) ∈ solutions →
    (x^2 - x*y + y^2 = 7 ∧ x^4 + x^2*y^2 + y^4 = 91) := by
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_l767_76745


namespace NUMINAMATH_CALUDE_polar_coords_of_negative_one_negative_one_l767_76712

/-- Prove that the polar coordinates of the point P(-1, -1) are (√2, 5π/4) -/
theorem polar_coords_of_negative_one_negative_one :
  let x : ℝ := -1
  let y : ℝ := -1
  let ρ : ℝ := Real.sqrt 2
  let θ : ℝ := 5 * Real.pi / 4
  x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ := by
  sorry

end NUMINAMATH_CALUDE_polar_coords_of_negative_one_negative_one_l767_76712


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l767_76759

theorem decimal_to_fraction : 
  ∃ (n d : ℕ), n / d = (38 : ℚ) / 100 ∧ gcd n d = 1 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l767_76759


namespace NUMINAMATH_CALUDE_function_symmetry_l767_76708

theorem function_symmetry (f : ℝ → ℝ) (h : ∀ x ≠ 0, f x + 2 * f (1 / x) = 3 * x) :
  ∀ x ≠ 0, f x = f (-x) ↔ x = Real.sqrt 2 ∨ x = -Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_function_symmetry_l767_76708


namespace NUMINAMATH_CALUDE_vertical_translation_of_linear_function_l767_76757

/-- Represents a linear function of the form y = mx + b -/
structure LinearFunction where
  m : ℝ
  b : ℝ

/-- Translates a linear function vertically by a given amount -/
def translate_vertical (f : LinearFunction) (k : ℝ) : LinearFunction :=
  { m := f.m, b := f.b + k }

/-- The original function y = -3x -/
def original_function : LinearFunction :=
  { m := -3, b := 0 }

/-- The amount of vertical translation -/
def translation_amount : ℝ := 2

theorem vertical_translation_of_linear_function :
  translate_vertical original_function translation_amount =
  { m := -3, b := 2 } :=
sorry

end NUMINAMATH_CALUDE_vertical_translation_of_linear_function_l767_76757


namespace NUMINAMATH_CALUDE_expected_marbles_theorem_l767_76775

/-- The expected number of marbles drawn until the special marble is picked, given that no ugly marbles were drawn -/
def expected_marbles_drawn (blue_marbles : ℕ) (ugly_marbles : ℕ) (special_marbles : ℕ) : ℚ :=
  let total_marbles := blue_marbles + ugly_marbles + special_marbles
  let prob_blue := blue_marbles / total_marbles
  let prob_special := special_marbles / total_marbles
  let expected_draws := (prob_blue / (1 - prob_blue)) / prob_special
  expected_draws

/-- Theorem stating that the expected number of marbles drawn is 20/11 -/
theorem expected_marbles_theorem :
  expected_marbles_drawn 9 10 1 = 20 / 11 := by
  sorry

end NUMINAMATH_CALUDE_expected_marbles_theorem_l767_76775


namespace NUMINAMATH_CALUDE_simplify_expression_l767_76770

theorem simplify_expression : 2^2 + 2^2 + 2^2 + 2^2 = 2^4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l767_76770


namespace NUMINAMATH_CALUDE_solve_equation_l767_76789

theorem solve_equation (x y : ℝ) (h : 3 * x^2 - 2 * y = 1) :
  2025 + 2 * y - 3 * x^2 = 2024 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l767_76789


namespace NUMINAMATH_CALUDE_fourth_term_max_implies_n_six_l767_76771

theorem fourth_term_max_implies_n_six (n : ℕ) : 
  (∀ k : ℕ, k ≠ 3 → (n.choose k) ≤ (n.choose 3)) → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_max_implies_n_six_l767_76771


namespace NUMINAMATH_CALUDE_geometric_mean_of_3_and_12_l767_76714

theorem geometric_mean_of_3_and_12 : 
  ∃ (x : ℝ), x > 0 ∧ x^2 = 3 * 12 ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_geometric_mean_of_3_and_12_l767_76714


namespace NUMINAMATH_CALUDE_total_unique_polygons_l767_76778

/-- Represents a regular polyhedron --/
inductive RegularPolyhedron
  | Tetrahedron
  | Cube
  | Octahedron
  | Dodecahedron
  | Icosahedron

/-- Returns the number of unique non-planar polygons for a given regular polyhedron --/
def num_unique_polygons (p : RegularPolyhedron) : Nat :=
  match p with
  | .Tetrahedron => 1
  | .Cube => 1
  | .Octahedron => 3
  | .Dodecahedron => 2
  | .Icosahedron => 3

/-- The list of all regular polyhedra --/
def all_polyhedra : List RegularPolyhedron :=
  [RegularPolyhedron.Tetrahedron, RegularPolyhedron.Cube, RegularPolyhedron.Octahedron,
   RegularPolyhedron.Dodecahedron, RegularPolyhedron.Icosahedron]

/-- Theorem stating that the total number of unique non-planar polygons for all regular polyhedra is 10 --/
theorem total_unique_polygons :
  (all_polyhedra.map num_unique_polygons).sum = 10 := by
  sorry

#eval (all_polyhedra.map num_unique_polygons).sum

end NUMINAMATH_CALUDE_total_unique_polygons_l767_76778


namespace NUMINAMATH_CALUDE_probability_of_two_packages_l767_76711

/-- The number of tablets in a new package -/
def n : ℕ := 10

/-- The probability of having exactly two packages of tablets -/
def probability_two_packages : ℚ := (2^n - 1) / (2^(n-1) * n)

/-- Theorem stating the probability of having exactly two packages of tablets -/
theorem probability_of_two_packages :
  probability_two_packages = (2^n - 1) / (2^(n-1) * n) := by sorry

end NUMINAMATH_CALUDE_probability_of_two_packages_l767_76711


namespace NUMINAMATH_CALUDE_production_days_calculation_l767_76728

theorem production_days_calculation (n : ℕ) 
  (h1 : (n * 50 : ℝ) / n = 50)  -- Average of 50 units for n days
  (h2 : ((n * 50 + 105 : ℝ) / (n + 1) = 55)) -- New average of 55 units including today
  : n = 10 := by
  sorry

end NUMINAMATH_CALUDE_production_days_calculation_l767_76728


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l767_76755

theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (h : Set.Ioo 1 2 = {x | a * x^2 + b * x + c < 0}) : 
  Set.Ioo (1/2) 1 = {x | c * x^2 + b * x + a < 0} :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l767_76755


namespace NUMINAMATH_CALUDE_senate_committee_arrangement_l767_76719

/-- The number of ways to arrange senators around a circular table. -/
def arrange_senators (num_democrats : ℕ) (num_republicans : ℕ) : ℕ :=
  if num_democrats = num_republicans ∧ num_democrats > 0 then
    (num_democrats.factorial) * ((num_democrats - 1).factorial)
  else
    0

/-- Theorem: The number of ways to arrange 6 Democrats and 6 Republicans
    around a circular table, with Democrats and Republicans alternating,
    is equal to 86,400. -/
theorem senate_committee_arrangement :
  arrange_senators 6 6 = 86400 := by
  sorry

end NUMINAMATH_CALUDE_senate_committee_arrangement_l767_76719


namespace NUMINAMATH_CALUDE_largest_consecutive_even_integer_l767_76783

theorem largest_consecutive_even_integer (n : ℕ) : 
  n % 2 = 0 ∧ 
  n * (n + 2) * (n + 4) * (n + 6) = 6720 →
  n + 6 = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_consecutive_even_integer_l767_76783


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l767_76791

theorem inequality_and_equality_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b ≥ 1) :
  (1 / (1 + a) + 1 / (1 + b) ≤ 1) ∧
  (1 / (1 + a) + 1 / (1 + b) = 1 ↔ a * b = 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l767_76791


namespace NUMINAMATH_CALUDE_square_perimeter_increase_l767_76779

theorem square_perimeter_increase (x : ℝ) (h : x > 0) :
  let original_side := x / 4
  let new_perimeter := 4 * x
  let new_side := new_perimeter / 4
  new_side / original_side = 4 := by sorry

end NUMINAMATH_CALUDE_square_perimeter_increase_l767_76779


namespace NUMINAMATH_CALUDE_complex_equation_solution_l767_76773

theorem complex_equation_solution (z : ℂ) :
  z * Complex.I = Complex.abs (1/2 - Complex.I * (Real.sqrt 3 / 2)) →
  z = -Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l767_76773


namespace NUMINAMATH_CALUDE_exam_attendance_l767_76788

theorem exam_attendance (passed_percentage : ℚ) (failed_count : ℕ) : 
  passed_percentage = 35/100 →
  failed_count = 546 →
  (failed_count : ℚ) / (1 - passed_percentage) = 840 :=
by
  sorry

end NUMINAMATH_CALUDE_exam_attendance_l767_76788


namespace NUMINAMATH_CALUDE_unique_solution_l767_76738

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- Theorem: 2001 is the only natural number n that satisfies n + S(n) = 2004 -/
theorem unique_solution : ∀ n : ℕ, n + S n = 2004 ↔ n = 2001 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l767_76738


namespace NUMINAMATH_CALUDE_base4_multiplication_division_l767_76737

-- Define a function to convert base 4 to base 10
def base4ToBase10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, digit) => acc + digit * (4 ^ i)) 0

-- Define a function to convert base 10 to base 4
def base10ToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
  aux n []

-- Define the numbers in base 4
def n1 : List Nat := [1, 3, 2]  -- 231₄
def n2 : List Nat := [1, 2]     -- 21₄
def n3 : List Nat := [3]        -- 3₄

-- State the theorem
theorem base4_multiplication_division :
  base10ToBase4 ((base4ToBase10 n1 * base4ToBase10 n2) / base4ToBase10 n3) = [3, 1, 2] := by
  sorry

end NUMINAMATH_CALUDE_base4_multiplication_division_l767_76737


namespace NUMINAMATH_CALUDE_pentagon_area_sum_l767_76732

/-- Represents a pentagon with given side lengths and angle -/
structure Pentagon where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DE : ℝ
  EA : ℝ
  angleCDE : ℝ
  ABparallelDE : Prop

/-- Represents the area of a pentagon in the form √a + b·√c -/
structure PentagonArea where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Function to calculate the area of a pentagon -/
noncomputable def calculatePentagonArea (p : Pentagon) : ℝ := sorry

/-- Function to express the pentagon area in the form √a + b·√c -/
noncomputable def expressAreaAsSum (area : ℝ) : PentagonArea := sorry

theorem pentagon_area_sum (p : Pentagon) 
  (h1 : p.AB = 8)
  (h2 : p.BC = 4)
  (h3 : p.CD = 10)
  (h4 : p.DE = 7)
  (h5 : p.EA = 10)
  (h6 : p.angleCDE = π / 3)  -- 60° in radians
  (h7 : p.ABparallelDE) :
  let area := calculatePentagonArea p
  let expression := expressAreaAsSum area
  expression.a + expression.b + expression.c = 39 := by sorry

end NUMINAMATH_CALUDE_pentagon_area_sum_l767_76732


namespace NUMINAMATH_CALUDE_probability_three_or_more_smile_l767_76744

def probability_single_baby_smile : ℚ := 1 / 3

def number_of_babies : ℕ := 6

def probability_at_least_three_smile (p : ℚ) (n : ℕ) : ℚ :=
  1 - (Finset.sum (Finset.range 3) (λ k => (n.choose k : ℚ) * p^k * (1 - p)^(n - k)))

theorem probability_three_or_more_smile :
  probability_at_least_three_smile probability_single_baby_smile number_of_babies = 353 / 729 :=
sorry

end NUMINAMATH_CALUDE_probability_three_or_more_smile_l767_76744


namespace NUMINAMATH_CALUDE_power_calculation_l767_76725

theorem power_calculation : 8^15 / 64^7 * 16 = 512 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l767_76725


namespace NUMINAMATH_CALUDE_stock_price_calculation_abc_stock_price_l767_76715

theorem stock_price_calculation (initial_price : ℝ) 
  (first_year_increase : ℝ) (second_year_decrease : ℝ) : ℝ :=
  let price_after_first_year := initial_price * (1 + first_year_increase)
  let price_after_second_year := price_after_first_year * (1 - second_year_decrease)
  price_after_second_year

theorem abc_stock_price : 
  stock_price_calculation 100 0.5 0.3 = 105 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_calculation_abc_stock_price_l767_76715


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l767_76764

/-- The parabola P with equation y = x^2 -/
def P : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1^2}

/-- The point Q -/
def Q : ℝ × ℝ := (20, 14)

/-- The line through Q with slope m -/
def line_through_Q (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 - Q.2 = m * (p.1 - Q.1)}

/-- The condition for non-intersection -/
def no_intersection (m : ℝ) : Prop :=
  line_through_Q m ∩ P = ∅

theorem parabola_line_intersection :
  ∃ (r s : ℝ), (∀ m : ℝ, no_intersection m ↔ r < m ∧ m < s) ∧ r + s = 80 := by
  sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l767_76764


namespace NUMINAMATH_CALUDE_max_sum_of_roots_l767_76774

theorem max_sum_of_roots (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (hsum : a + b + c = 8) :
  ∃ (x y z : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z = 8 ∧
  ∀ (a' b' c' : ℝ), a' ≥ 0 → b' ≥ 0 → c' ≥ 0 → a' + b' + c' = 8 →
  Real.sqrt (3 * a' + 2) + Real.sqrt (3 * b' + 2) + Real.sqrt (3 * c' + 2) ≤
  Real.sqrt (3 * x + 2) + Real.sqrt (3 * y + 2) + Real.sqrt (3 * z + 2) ∧
  Real.sqrt (3 * x + 2) + Real.sqrt (3 * y + 2) + Real.sqrt (3 * z + 2) = 3 * Real.sqrt 10 :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_roots_l767_76774


namespace NUMINAMATH_CALUDE_hyperbola_k_range_l767_76705

/-- The equation of a hyperbola with parameter k -/
def hyperbola_equation (x y k : ℝ) : Prop :=
  x^2 / (1 - 2*k) - y^2 / (k - 2) = 1

/-- The condition that the hyperbola has foci on the y-axis -/
def foci_on_y_axis (k : ℝ) : Prop :=
  1 - 2*k < 0 ∧ k - 2 < 0

/-- Theorem: If the equation represents a hyperbola with foci on the y-axis,
    then k is in the open interval (1/2, 2) -/
theorem hyperbola_k_range (k : ℝ) :
  (∃ x y : ℝ, hyperbola_equation x y k) →
  foci_on_y_axis k →
  k ∈ Set.Ioo (1/2 : ℝ) 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_k_range_l767_76705


namespace NUMINAMATH_CALUDE_cuboid_surface_area_example_l767_76758

/-- The surface area of a cuboid given its dimensions -/
def cuboidSurfaceArea (length width height : ℝ) : ℝ :=
  2 * (length * width + width * height + height * length)

/-- Theorem: The surface area of a cuboid with length 4, width 5, and height 6 is 148 -/
theorem cuboid_surface_area_example : cuboidSurfaceArea 4 5 6 = 148 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_surface_area_example_l767_76758


namespace NUMINAMATH_CALUDE_combination_sum_equality_l767_76767

theorem combination_sum_equality (n m k : ℕ) (h1 : 1 ≤ k) (h2 : k < m) (h3 : m ≤ n) :
  (Nat.choose n m) + (Finset.range (k + 1)).sum (λ i => (Nat.choose k i) * (Nat.choose n (m - i))) = Nat.choose (n + k) m := by
  sorry

end NUMINAMATH_CALUDE_combination_sum_equality_l767_76767


namespace NUMINAMATH_CALUDE_apple_tree_width_proof_l767_76726

/-- The width of an apple tree in Quinton's backyard -/
def apple_tree_width : ℝ := 10

/-- The space between apple trees -/
def apple_tree_space : ℝ := 12

/-- The width of a peach tree -/
def peach_tree_width : ℝ := 12

/-- The space between peach trees -/
def peach_tree_space : ℝ := 15

/-- The total space taken by all trees -/
def total_space : ℝ := 71

theorem apple_tree_width_proof :
  2 * apple_tree_width + apple_tree_space + 2 * peach_tree_width + peach_tree_space = total_space :=
by sorry

end NUMINAMATH_CALUDE_apple_tree_width_proof_l767_76726


namespace NUMINAMATH_CALUDE_cos_inequality_range_l767_76799

theorem cos_inequality_range (x : Real) :
  x ∈ Set.Icc 0 (2 * Real.pi) →
  (Real.cos x ≤ 1 / 2 ↔ x ∈ Set.Icc (Real.pi / 3) (5 * Real.pi / 3)) := by
  sorry

end NUMINAMATH_CALUDE_cos_inequality_range_l767_76799


namespace NUMINAMATH_CALUDE_angle_side_ratio_angle_sine_relation_two_solutions_l767_76763

-- Define a triangle
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Theorem 1
theorem angle_side_ratio (t : Triangle) :
  t.A / t.B = 1 / 2 ∧ t.B / t.C = 2 / 3 →
  t.a / t.b = 1 / Real.sqrt 3 ∧ t.b / t.c = Real.sqrt 3 / 2 := by sorry

-- Theorem 2
theorem angle_sine_relation (t : Triangle) :
  t.A > t.B → Real.sin t.A > Real.sin t.B := by sorry

-- Theorem 3
theorem two_solutions (t : Triangle) :
  t.A = π / 6 ∧ t.a = 3 ∧ t.b = 4 →
  ∃ (t1 t2 : Triangle), t1 ≠ t2 ∧
    t1.A = t.A ∧ t1.a = t.a ∧ t1.b = t.b ∧
    t2.A = t.A ∧ t2.a = t.a ∧ t2.b = t.b := by sorry

end NUMINAMATH_CALUDE_angle_side_ratio_angle_sine_relation_two_solutions_l767_76763


namespace NUMINAMATH_CALUDE_bus_dispatch_interval_l767_76768

/-- The speed of the bus -/
def bus_speed : ℝ := sorry

/-- The speed of Xiao Wang -/
def person_speed : ℝ := sorry

/-- The interval between each bus dispatch in minutes -/
def dispatch_interval : ℝ := sorry

/-- The time between a bus passing Xiao Wang from behind in minutes -/
def overtake_time : ℝ := 6

/-- The time between a bus coming towards Xiao Wang in minutes -/
def approach_time : ℝ := 3

/-- Theorem stating that given the conditions, the dispatch interval is 4 minutes -/
theorem bus_dispatch_interval : 
  bus_speed > 0 ∧ 
  person_speed > 0 ∧ 
  person_speed < bus_speed ∧
  overtake_time * (bus_speed - person_speed) = dispatch_interval * bus_speed ∧
  approach_time * (bus_speed + person_speed) = dispatch_interval * bus_speed →
  dispatch_interval = 4 := by sorry

end NUMINAMATH_CALUDE_bus_dispatch_interval_l767_76768


namespace NUMINAMATH_CALUDE_min_perimeter_triangle_l767_76731

theorem min_perimeter_triangle (a b c : ℕ) (A B C : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  Real.cos A = 3/5 →
  Real.cos B = 5/13 →
  Real.cos C = -1/3 →
  a + b > c ∧ b + c > a ∧ c + a > b →
  (∀ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 →
    Real.cos A = 3/5 →
    Real.cos B = 5/13 →
    Real.cos C = -1/3 →
    x + y > z ∧ y + z > x ∧ z + x > y →
    a + b + c ≤ x + y + z) →
  a + b + c = 192 :=
sorry

end NUMINAMATH_CALUDE_min_perimeter_triangle_l767_76731


namespace NUMINAMATH_CALUDE_original_circle_area_l767_76702

/-- Given a circle whose area increases by 8 times and whose circumference
    increases by 50.24 centimeters, prove that its original area is 50.24 square centimeters. -/
theorem original_circle_area (r : ℝ) (h1 : r > 0) : 
  (π * (r + 50.24 / (2 * π))^2 = 9 * π * r^2) ∧ 
  (2 * π * (r + 50.24 / (2 * π)) = 2 * π * r + 50.24) → 
  π * r^2 = 50.24 := by
  sorry

end NUMINAMATH_CALUDE_original_circle_area_l767_76702


namespace NUMINAMATH_CALUDE_four_numbers_problem_l767_76742

theorem four_numbers_problem (a b c d : ℝ) : 
  a + b + c + d = 45 ∧ 
  a + 2 = b - 2 ∧ 
  a + 2 = 2 * c ∧ 
  a + 2 = d / 2 → 
  a = 8 ∧ b = 12 ∧ c = 5 ∧ d = 20 := by
sorry

end NUMINAMATH_CALUDE_four_numbers_problem_l767_76742


namespace NUMINAMATH_CALUDE_hundreds_digit_of_factorial_difference_is_zero_l767_76730

-- Define factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Theorem statement
theorem hundreds_digit_of_factorial_difference_is_zero :
  ∃ k : ℕ, factorial 25 - factorial 20 = 1000 * k :=
sorry

end NUMINAMATH_CALUDE_hundreds_digit_of_factorial_difference_is_zero_l767_76730


namespace NUMINAMATH_CALUDE_scaling_transforms_line_l767_76762

-- Define the original line
def original_line (x y : ℝ) : Prop := x - 2*y = 2

-- Define the transformed line
def transformed_line (x' y' : ℝ) : Prop := 2*x' - y' = 4

-- Define the scaling transformation
def scaling_transformation (x y x' y' : ℝ) : Prop := x' = x ∧ y' = 4*y

-- Theorem statement
theorem scaling_transforms_line :
  ∀ (x y x' y' : ℝ),
    original_line x y →
    scaling_transformation x y x' y' →
    transformed_line x' y' := by
  sorry

end NUMINAMATH_CALUDE_scaling_transforms_line_l767_76762


namespace NUMINAMATH_CALUDE_new_york_to_cape_town_duration_l767_76710

/-- Represents a time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : minutes < 60

/-- Represents a day of the week -/
inductive Day
| Monday
| Tuesday

/-- Calculates the time difference between two times in hours -/
def timeDifference (t1 t2 : Time) (d1 d2 : Day) : ℕ :=
  sorry

/-- The departure time from London -/
def londonDeparture : Time := { hours := 6, minutes := 0, valid := by simp }

/-- The arrival time in Cape Town -/
def capeTownArrival : Time := { hours := 10, minutes := 0, valid := by simp }

/-- Theorem stating the duration of the New York to Cape Town flight -/
theorem new_york_to_cape_town_duration :
  let londonToNewYorkDuration : ℕ := 18
  let newYorkArrival : Time := 
    { hours := 0, minutes := 0, valid := by simp }
  let newYorkToCapeArrivalDay : Day := Day.Tuesday
  timeDifference newYorkArrival capeTownArrival Day.Tuesday newYorkToCapeArrivalDay = 10 :=
sorry

end NUMINAMATH_CALUDE_new_york_to_cape_town_duration_l767_76710


namespace NUMINAMATH_CALUDE_sin_cos_relation_in_triangle_l767_76782

theorem sin_cos_relation_in_triangle (A B C : ℝ) (h_triangle : A + B + C = π) :
  (Real.sin A > Real.sin B) ↔ (Real.cos A < Real.cos B) :=
sorry

end NUMINAMATH_CALUDE_sin_cos_relation_in_triangle_l767_76782


namespace NUMINAMATH_CALUDE_shoe_shopping_cost_l767_76785

theorem shoe_shopping_cost 
  (price1 price2 price3 : ℝ) 
  (half_off_discount : ℝ → ℝ)
  (third_pair_discount : ℝ → ℝ)
  (extra_discount : ℝ → ℝ)
  (sales_tax : ℝ → ℝ)
  (h1 : price1 = 40)
  (h2 : price2 = 60)
  (h3 : price3 = 80)
  (h4 : half_off_discount x = x / 2)
  (h5 : third_pair_discount x = x * 0.7)
  (h6 : extra_discount x = x * 0.75)
  (h7 : sales_tax x = x * 1.08)
  : sales_tax (extra_discount (price1 + (price2 - half_off_discount price1) + third_pair_discount price3)) = 110.16 := by
  sorry

end NUMINAMATH_CALUDE_shoe_shopping_cost_l767_76785


namespace NUMINAMATH_CALUDE_min_value_constrained_l767_76735

/-- Given that x + 2y + 3z = 1, the minimum value of x^2 + y^2 + z^2 is 1/14 -/
theorem min_value_constrained (x y z : ℝ) (h : x + 2*y + 3*z = 1) :
  ∃ (min : ℝ), min = (1 : ℝ) / 14 ∧ 
  ∀ (a b c : ℝ), a + 2*b + 3*c = 1 → x^2 + y^2 + z^2 ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_constrained_l767_76735


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l767_76750

theorem sum_of_x_and_y : ∀ (x y : ℤ), x - y = 18 → x = 14 → x + y = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l767_76750


namespace NUMINAMATH_CALUDE_max_profit_at_three_percent_l767_76752

-- Define the bank's profit function
def bank_profit (k : ℝ) (x : ℝ) : ℝ :=
  0.045 * k * x^2 - k * x^3

-- Theorem statement
theorem max_profit_at_three_percent (k : ℝ) (h : k > 0) :
  ∃ (max_x : ℝ), max_x > 0 ∧ max_x = 0.03 ∧
  ∀ (x : ℝ), x > 0 → bank_profit k x ≤ bank_profit k max_x :=
sorry

end NUMINAMATH_CALUDE_max_profit_at_three_percent_l767_76752


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l767_76747

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of specific terms in the sequence equals 120 -/
def SumCondition (a : ℕ → ℝ) : Prop :=
  a 4 + a 6 + a 8 + a 10 + a 12 = 120

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h1 : ArithmeticSequence a) (h2 : SumCondition a) : 
  2 * a 10 - a 12 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l767_76747


namespace NUMINAMATH_CALUDE_polygon_number_formula_l767_76794

/-- N(n, k) represents the n-th k-sided polygon number -/
def N (n k : ℕ) : ℚ :=
  match k with
  | 3 => (1/2 : ℚ) * n^2 + (1/2 : ℚ) * n
  | 4 => n^2
  | 5 => (3/2 : ℚ) * n^2 - (1/2 : ℚ) * n
  | 6 => 2 * n^2 - n
  | _ => 0  -- placeholder for other k values

/-- The general formula for N(n, k) -/
def N_general (n k : ℕ) : ℚ :=
  ((k - 2 : ℚ) / 2) * n^2 + ((4 - k : ℚ) / 2) * n

theorem polygon_number_formula (n k : ℕ) (h1 : k ≥ 3) (h2 : n ≥ 1) :
  N n k = N_general n k :=
by sorry

end NUMINAMATH_CALUDE_polygon_number_formula_l767_76794


namespace NUMINAMATH_CALUDE_children_on_bus_l767_76787

theorem children_on_bus (total : ℕ) (men : ℕ) (women : ℕ) 
  (h1 : total = 54)
  (h2 : men = 18)
  (h3 : women = 26) :
  total - men - women = 10 := by
  sorry

end NUMINAMATH_CALUDE_children_on_bus_l767_76787


namespace NUMINAMATH_CALUDE_sector_arc_length_l767_76704

theorem sector_arc_length (θ : Real) (r : Real) (l : Real) : 
  θ = 2 * Real.pi / 3 → r = 2 → l = θ * r → l = 4 * Real.pi / 3 := by
  sorry

#check sector_arc_length

end NUMINAMATH_CALUDE_sector_arc_length_l767_76704


namespace NUMINAMATH_CALUDE_janet_lives_count_janet_final_lives_l767_76781

theorem janet_lives_count (initial_lives lost_lives gained_lives : ℕ) : 
  initial_lives - lost_lives + gained_lives = (initial_lives - lost_lives) + gained_lives :=
by sorry

theorem janet_final_lives : 
  38 - 16 + 32 = 54 :=
by sorry

end NUMINAMATH_CALUDE_janet_lives_count_janet_final_lives_l767_76781


namespace NUMINAMATH_CALUDE_boots_cost_ratio_l767_76729

theorem boots_cost_ratio (initial_amount : ℚ) (toilet_paper_cost : ℚ) (additional_money : ℚ) :
  initial_amount = 50 →
  toilet_paper_cost = 12 →
  additional_money = 35 →
  let remaining_after_toilet_paper := initial_amount - toilet_paper_cost
  let groceries_cost := 2 * toilet_paper_cost
  let remaining_after_groceries := remaining_after_toilet_paper - groceries_cost
  let total_boot_cost := remaining_after_groceries + 2 * additional_money
  let single_boot_cost := total_boot_cost / 2
  (single_boot_cost / remaining_after_groceries : ℚ) = 3 := by
sorry

end NUMINAMATH_CALUDE_boots_cost_ratio_l767_76729


namespace NUMINAMATH_CALUDE_exists_periodic_product_l767_76709

/-- A function f: ℝ → ℝ is periodic with period p if it's not constant and
    f(x) = f(x + p) for all x ∈ ℝ -/
def IsPeriodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ (∃ x y, f x ≠ f y) ∧ ∀ x, f x = f (x + p)

/-- The period of a periodic function is the smallest positive p satisfying the periodicity condition -/
def Period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  IsPeriodic f p ∧ ∀ q, 0 < q ∧ q < p → ¬ IsPeriodic f q

/-- Given any two positive real numbers a and b, there exist two periodic functions
    f₁ and f₂ with periods a and b respectively, such that their product f₁(x) · f₂(x)
    is also a periodic function -/
theorem exists_periodic_product (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (f₁ f₂ : ℝ → ℝ), Period f₁ a ∧ Period f₂ b ∧
  ∃ p, p > 0 ∧ IsPeriodic (fun x ↦ f₁ x * f₂ x) p := by
  sorry

end NUMINAMATH_CALUDE_exists_periodic_product_l767_76709


namespace NUMINAMATH_CALUDE_circle_transformation_l767_76753

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

def translate_up (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + d)

theorem circle_transformation :
  let S : ℝ × ℝ := (3, -4)
  let reflected := reflect_x S
  let final := translate_up reflected 5
  final = (3, 9) := by sorry

end NUMINAMATH_CALUDE_circle_transformation_l767_76753


namespace NUMINAMATH_CALUDE_reflection_coordinate_sum_l767_76700

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point over the x-axis -/
def reflect_over_x (p : Point) : Point :=
  { x := p.x, y := -p.y }

/-- The sum of coordinates of two points -/
def sum_coordinates (p1 p2 : Point) : ℝ :=
  p1.x + p1.y + p2.x + p2.y

theorem reflection_coordinate_sum :
  ∀ y : ℝ, 
  let A : Point := { x := 3, y := y }
  let B : Point := reflect_over_x A
  sum_coordinates A B = 6 := by
sorry

end NUMINAMATH_CALUDE_reflection_coordinate_sum_l767_76700


namespace NUMINAMATH_CALUDE_a_bounds_l767_76761

theorem a_bounds (a b c d : ℝ) 
  (sum_eq : a + b + c + d = 3) 
  (sum_squares_eq : a^2 + 2*b^2 + 3*c^2 + 6*d^2 = 5) : 
  1 ≤ a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_a_bounds_l767_76761


namespace NUMINAMATH_CALUDE_coefficient_of_negative_2pi_ab_squared_l767_76739

/-- The coefficient of a monomial is the numerical factor that multiplies the variable part. -/
def coefficient (m : ℝ) (x : String) : ℝ := sorry

/-- A monomial is an algebraic expression consisting of a single term. -/
def is_monomial (x : String) : Prop := sorry

theorem coefficient_of_negative_2pi_ab_squared :
  is_monomial "-2πab²" → coefficient (-2 * Real.pi) "ab²" = -2 * Real.pi := by sorry

end NUMINAMATH_CALUDE_coefficient_of_negative_2pi_ab_squared_l767_76739


namespace NUMINAMATH_CALUDE_inequality_proof_l767_76736

theorem inequality_proof (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (sum_one : x + y + z = 1) :
  (1 / x) + (4 / y) + (9 / z) ≥ 36 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l767_76736


namespace NUMINAMATH_CALUDE_line_parameterization_l767_76777

/-- Given a line y = 2x - 3 parameterized as (x, y) = (-8, s) + t(l, -7),
    prove that s = -19 and l = -7/2 -/
theorem line_parameterization (s l : ℝ) : 
  (∀ (x y t : ℝ), y = 2*x - 3 ↔ (x, y) = (-8, s) + t • (l, -7)) →
  s = -19 ∧ l = -7/2 := by
sorry

end NUMINAMATH_CALUDE_line_parameterization_l767_76777


namespace NUMINAMATH_CALUDE_students_not_enrolled_l767_76701

theorem students_not_enrolled (total : ℕ) (biology_frac : ℚ) (chemistry_frac : ℚ) (physics_frac : ℚ) 
  (h_total : total = 1500)
  (h_biology : biology_frac = 2/5)
  (h_chemistry : chemistry_frac = 3/8)
  (h_physics : physics_frac = 1/10)
  (h_no_overlap : biology_frac + chemistry_frac + physics_frac ≤ 1) :
  total - (⌊biology_frac * total⌋ + ⌊chemistry_frac * total⌋ + ⌊physics_frac * total⌋) = 188 := by
  sorry

end NUMINAMATH_CALUDE_students_not_enrolled_l767_76701


namespace NUMINAMATH_CALUDE_blue_tiles_count_l767_76790

/-- Given a pool that needs tiles, this theorem proves the number of blue tiles. -/
theorem blue_tiles_count 
  (total_needed : ℕ) 
  (additional_needed : ℕ) 
  (red_tiles : ℕ) 
  (h1 : total_needed = 100) 
  (h2 : additional_needed = 20) 
  (h3 : red_tiles = 32) : 
  total_needed - additional_needed - red_tiles = 48 := by
  sorry

end NUMINAMATH_CALUDE_blue_tiles_count_l767_76790


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l767_76707

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 and asymptotes y = ±2x is √5 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : b / a = 2) : 
  Real.sqrt (1 + (b / a)^2) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l767_76707


namespace NUMINAMATH_CALUDE_workshop_average_salary_l767_76760

theorem workshop_average_salary
  (total_workers : ℕ)
  (technicians : ℕ)
  (avg_salary_technicians : ℕ)
  (avg_salary_rest : ℕ)
  (h1 : total_workers = 12)
  (h2 : technicians = 6)
  (h3 : avg_salary_technicians = 12000)
  (h4 : avg_salary_rest = 6000) :
  (technicians * avg_salary_technicians + (total_workers - technicians) * avg_salary_rest) / total_workers = 9000 :=
by
  sorry

#check workshop_average_salary

end NUMINAMATH_CALUDE_workshop_average_salary_l767_76760


namespace NUMINAMATH_CALUDE_not_neighboring_root_eq1_neighboring_root_eq2_neighboring_root_eq3_l767_76718

/-- Definition of a neighboring root equation -/
def is_neighboring_root_equation (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ ∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ (x - y = 1 ∨ y - x = 1)

/-- Theorem for the first equation -/
theorem not_neighboring_root_eq1 : ¬ is_neighboring_root_equation 1 (-1) (-6) :=
sorry

/-- Theorem for the second equation -/
theorem neighboring_root_eq2 : is_neighboring_root_equation 2 (-2 * Real.sqrt 3) 1 :=
sorry

/-- Theorem for the third equation -/
theorem neighboring_root_eq3 (m : ℝ) : 
  is_neighboring_root_equation 1 (-(m-2)) (-2*m) ↔ m = -1 ∨ m = -3 :=
sorry

end NUMINAMATH_CALUDE_not_neighboring_root_eq1_neighboring_root_eq2_neighboring_root_eq3_l767_76718


namespace NUMINAMATH_CALUDE_complement_union_problem_l767_76740

theorem complement_union_problem (U A B : Set Nat) : 
  U = {1, 2, 3, 4} →
  A = {1, 2} →
  B = {2, 3} →
  (Aᶜ ∪ B) = {2, 3, 4} := by
  sorry

end NUMINAMATH_CALUDE_complement_union_problem_l767_76740


namespace NUMINAMATH_CALUDE_work_ratio_is_three_to_eight_l767_76743

/-- Represents a worker's weekly schedule and earnings -/
structure WorkerSchedule where
  days_per_week : ℕ
  weekly_earnings : ℚ
  daily_salary : ℚ

/-- Calculates the ratio of time worked on the last three days to the first four days -/
def work_ratio (w : WorkerSchedule) : ℚ × ℚ :=
  let last_three_days_earnings := w.weekly_earnings - (4 * w.daily_salary)
  let last_three_days_time := last_three_days_earnings / w.daily_salary
  let first_four_days_time := 4
  (last_three_days_time * 2, first_four_days_time * 2)

/-- Theorem stating the work ratio for a specific worker schedule -/
theorem work_ratio_is_three_to_eight (w : WorkerSchedule) 
  (h1 : w.days_per_week = 7)
  (h2 : w.weekly_earnings = 55)
  (h3 : w.daily_salary = 10) :
  work_ratio w = (3, 8) := by
  sorry

#eval work_ratio ⟨7, 55, 10⟩

end NUMINAMATH_CALUDE_work_ratio_is_three_to_eight_l767_76743


namespace NUMINAMATH_CALUDE_area_of_quadrilateral_l767_76766

/-- A line with slope -3 intersecting positive x and y axes -/
structure Line1 where
  slope : ℝ
  x_intercept : ℝ
  y_intercept : ℝ

/-- Another line intersecting x and y axes -/
structure Line2 where
  x_intercept : ℝ
  y_intercept : ℝ

/-- Point of intersection of the two lines -/
structure IntersectionPoint where
  x : ℝ
  y : ℝ

/-- The problem setup -/
def ProblemSetup (l1 : Line1) (l2 : Line2) (e : IntersectionPoint) : Prop :=
  l1.slope = -3 ∧
  l1.x_intercept > 0 ∧
  l1.y_intercept > 0 ∧
  l2.x_intercept = 10 ∧
  e.x = 5 ∧
  e.y = 5

/-- The area of quadrilateral OBEC -/
def QuadrilateralArea (l1 : Line1) (l2 : Line2) (e : IntersectionPoint) : ℝ := 
  sorry  -- The actual calculation would go here

theorem area_of_quadrilateral (l1 : Line1) (l2 : Line2) (e : IntersectionPoint) 
  (h : ProblemSetup l1 l2 e) : QuadrilateralArea l1 l2 e = 75 := by
  sorry

end NUMINAMATH_CALUDE_area_of_quadrilateral_l767_76766


namespace NUMINAMATH_CALUDE_Q_equals_G_l767_76792

-- Define the sets Q and G
def Q : Set ℝ := {y | ∃ x, y = x^2 + 1}
def G : Set ℝ := {x | x ≥ 1}

-- State the theorem
theorem Q_equals_G : Q = G := by sorry

end NUMINAMATH_CALUDE_Q_equals_G_l767_76792


namespace NUMINAMATH_CALUDE_solve_equation_l767_76724

theorem solve_equation : ∃ y : ℝ, (60 / 100 = Real.sqrt ((y + 20) / 100)) ∧ y = 16 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l767_76724
