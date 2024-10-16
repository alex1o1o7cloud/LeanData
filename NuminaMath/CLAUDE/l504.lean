import Mathlib

namespace NUMINAMATH_CALUDE_min_ab_value_l504_50462

theorem min_ab_value (a b : ℕ+) 
  (h1 : ¬ (7 ∣ (a * b * (a + b))))
  (h2 : (7 ∣ ((a + b)^7 - a^7 - b^7))) :
  ∀ x y : ℕ+, 
    (¬ (7 ∣ (x * y * (x + y)))) → 
    ((7 ∣ ((x + y)^7 - x^7 - y^7))) → 
    a * b ≤ x * y :=
by sorry

end NUMINAMATH_CALUDE_min_ab_value_l504_50462


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l504_50497

theorem sum_of_fractions_equals_one
  (a b c d w x y z : ℝ)
  (eq1 : 17*w + b*x + c*y + d*z = 0)
  (eq2 : a*w + 29*x + c*y + d*z = 0)
  (eq3 : a*w + b*x + 37*y + d*z = 0)
  (eq4 : a*w + b*x + c*y + 53*z = 0)
  (ha : a ≠ 17)
  (hb : b ≠ 29)
  (hc : c ≠ 37)
  (h_not_all_zero : ¬(w = 0 ∧ x = 0 ∧ y = 0)) :
  a / (a - 17) + b / (b - 29) + c / (c - 37) + d / (d - 53) = 1 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l504_50497


namespace NUMINAMATH_CALUDE_problem_statement_l504_50489

theorem problem_statement (x y : ℝ) (h : 3 * y - x^2 = -5) :
  6 * y - 2 * x^2 - 6 = -16 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l504_50489


namespace NUMINAMATH_CALUDE_umbrella_count_l504_50470

theorem umbrella_count (y b r : ℕ) 
  (h1 : b = (y + r) / 2)
  (h2 : r = (y + b) / 3)
  (h3 : y = 45) :
  b = 36 ∧ r = 27 := by
  sorry

end NUMINAMATH_CALUDE_umbrella_count_l504_50470


namespace NUMINAMATH_CALUDE_equation_solution_l504_50403

theorem equation_solution :
  ∃! y : ℚ, y + 4/5 = 2/3 + y/6 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_equation_solution_l504_50403


namespace NUMINAMATH_CALUDE_binomial_coefficient_10_3_l504_50465

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_10_3_l504_50465


namespace NUMINAMATH_CALUDE_change_amount_l504_50406

-- Define the given conditions
def pants_price : ℚ := 60
def shirt_price : ℚ := 45
def tie_price : ℚ := 20
def discount_rate : ℚ := 0.1
def tax_rate : ℚ := 0.075
def paid_amount : ℚ := 500

-- Define the calculation steps
def pants_total : ℚ := 3 * pants_price
def shirts_total : ℚ := 2 * shirt_price
def discount_amount : ℚ := discount_rate * shirt_price
def discounted_shirts_total : ℚ := shirts_total - discount_amount
def subtotal : ℚ := pants_total + discounted_shirts_total + tie_price
def tax_amount : ℚ := tax_rate * subtotal
def total_purchase : ℚ := subtotal + tax_amount
def change : ℚ := paid_amount - total_purchase

-- Theorem to prove
theorem change_amount : change = 193.09 := by
  sorry

end NUMINAMATH_CALUDE_change_amount_l504_50406


namespace NUMINAMATH_CALUDE_uniform_profit_percentage_clock_sales_l504_50495

/-- Uniform profit percentage calculation for clock sales --/
theorem uniform_profit_percentage_clock_sales
  (total_clocks : ℕ)
  (clocks_10_percent : ℕ)
  (clocks_20_percent : ℕ)
  (cost_price : ℚ)
  (revenue_difference : ℚ)
  (h1 : total_clocks = clocks_10_percent + clocks_20_percent)
  (h2 : total_clocks = 90)
  (h3 : clocks_10_percent = 40)
  (h4 : clocks_20_percent = 50)
  (h5 : cost_price = 79.99999999999773)
  (h6 : revenue_difference = 40) :
  let actual_revenue := clocks_10_percent * (cost_price * (1 + 10 / 100)) +
                        clocks_20_percent * (cost_price * (1 + 20 / 100))
  let uniform_revenue := actual_revenue - revenue_difference
  let uniform_profit_percentage := (uniform_revenue / (total_clocks * cost_price) - 1) * 100
  uniform_profit_percentage = 15 :=
sorry

end NUMINAMATH_CALUDE_uniform_profit_percentage_clock_sales_l504_50495


namespace NUMINAMATH_CALUDE_truncated_cone_height_l504_50475

/-- The height of a circular truncated cone with given top and bottom surface areas and volume. -/
theorem truncated_cone_height (S₁ S₂ V : ℝ) (h : ℝ) 
    (hS₁ : S₁ = 4 * Real.pi)
    (hS₂ : S₂ = 9 * Real.pi)
    (hV : V = 19 * Real.pi)
    (h_def : V = (1/3) * h * (S₁ + Real.sqrt (S₁ * S₂) + S₂)) :
  h = 3 := by
  sorry

#check truncated_cone_height

end NUMINAMATH_CALUDE_truncated_cone_height_l504_50475


namespace NUMINAMATH_CALUDE_jodi_walking_days_l504_50471

def weekly_distance (days_per_week : ℕ) : ℕ := 
  1 * days_per_week + 2 * days_per_week + 3 * days_per_week + 4 * days_per_week

theorem jodi_walking_days : 
  ∃ (days_per_week : ℕ), weekly_distance days_per_week = 60 ∧ days_per_week = 6 := by
  sorry

end NUMINAMATH_CALUDE_jodi_walking_days_l504_50471


namespace NUMINAMATH_CALUDE_specific_mixture_problem_l504_50435

/-- Represents a mixture of three components -/
structure Mixture where
  a : ℝ
  b : ℝ
  c : ℝ
  sum_to_100 : a + b + c = 100

/-- The problem of finding coefficients for mixing three mixtures to obtain a desired mixture -/
def mixture_problem (m₁ m₂ m₃ : Mixture) (desired : Mixture) :=
  ∃ (k₁ k₂ k₃ : ℝ),
    k₁ ≥ 0 ∧ k₂ ≥ 0 ∧ k₃ ≥ 0 ∧
    k₁ + k₂ + k₃ = 1 ∧
    k₁ * m₁.a + k₂ * m₂.a + k₃ * m₃.a = desired.a ∧
    k₁ * m₁.b + k₂ * m₂.b + k₃ * m₃.b = desired.b ∧
    k₁ * m₁.c + k₂ * m₂.c + k₃ * m₃.c = desired.c

/-- The specific mixture problem instance -/
theorem specific_mixture_problem :
  let m₁ : Mixture := ⟨10, 30, 60, by norm_num⟩
  let m₂ : Mixture := ⟨20, 60, 20, by norm_num⟩
  let m₃ : Mixture := ⟨80, 10, 10, by norm_num⟩
  let desired : Mixture := ⟨50, 30, 20, by norm_num⟩
  mixture_problem m₁ m₂ m₃ desired := by
    sorry

end NUMINAMATH_CALUDE_specific_mixture_problem_l504_50435


namespace NUMINAMATH_CALUDE_jimmy_can_lose_five_more_points_l504_50472

def passing_score : ℕ := 50
def exams_count : ℕ := 3
def points_per_exam : ℕ := 20
def points_lost : ℕ := 5

def max_additional_points_to_lose : ℕ :=
  exams_count * points_per_exam - points_lost - passing_score

theorem jimmy_can_lose_five_more_points :
  max_additional_points_to_lose = 5 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_can_lose_five_more_points_l504_50472


namespace NUMINAMATH_CALUDE_sum_and_equal_numbers_l504_50447

theorem sum_and_equal_numbers (a b c : ℚ) 
  (sum_eq : a + b + c = 108)
  (equal_after_changes : a + 8 = b - 4 ∧ b - 4 = 6 * c) :
  b = 724 / 13 := by
sorry

end NUMINAMATH_CALUDE_sum_and_equal_numbers_l504_50447


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l504_50455

theorem negative_fraction_comparison : -3/5 > -3/4 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l504_50455


namespace NUMINAMATH_CALUDE_multiply_mixed_number_l504_50426

theorem multiply_mixed_number : (7 : ℚ) * (9 + 2/5) = 65 + 4/5 := by
  sorry

end NUMINAMATH_CALUDE_multiply_mixed_number_l504_50426


namespace NUMINAMATH_CALUDE_reflection_across_origin_l504_50439

/-- Reflects a point across the origin -/
def reflect_origin (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

/-- The original point P -/
def P : ℝ × ℝ := (-2, -3)

/-- The reflected point Q -/
def Q : ℝ × ℝ := (2, 3)

theorem reflection_across_origin :
  reflect_origin P = Q := by sorry

end NUMINAMATH_CALUDE_reflection_across_origin_l504_50439


namespace NUMINAMATH_CALUDE_triangle_area_from_square_areas_l504_50476

theorem triangle_area_from_square_areas (a b c : ℝ) (h1 : a^2 = 36) (h2 : b^2 = 64) (h3 : c^2 = 100) :
  (1/2) * a * b = 24 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_from_square_areas_l504_50476


namespace NUMINAMATH_CALUDE_closest_axis_of_symmetry_l504_50425

theorem closest_axis_of_symmetry (ω : ℝ) (h1 : 0 < ω) (h2 : ω < π) :
  let f := fun x ↦ Real.sin (ω * x + 5 * π / 6)
  (f 0 = 1 / 2) →
  (f (1 / 2) = 0) →
  (∃ k : ℤ, -1 = 3 * k - 1 ∧ 
    ∀ m : ℤ, m ≠ k → |3 * m - 1| > |3 * k - 1|) :=
by sorry

end NUMINAMATH_CALUDE_closest_axis_of_symmetry_l504_50425


namespace NUMINAMATH_CALUDE_coffee_shop_sales_l504_50444

theorem coffee_shop_sales (teas : ℕ) (lattes : ℕ) : 
  teas = 6 → lattes = 4 * teas + 8 → lattes = 32 := by
  sorry

end NUMINAMATH_CALUDE_coffee_shop_sales_l504_50444


namespace NUMINAMATH_CALUDE_max_ab_min_a2_b2_l504_50459

theorem max_ab_min_a2_b2 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + 2 * b = 2) :
  (∀ x y : ℝ, 0 < x ∧ 0 < y ∧ x + 2 * y = 2 → x * y ≤ a * b) ∧
  (∀ x y : ℝ, 0 < x ∧ 0 < y ∧ x + 2 * y = 2 → a^2 + b^2 ≤ x^2 + y^2) ∧
  a * b = 1/2 ∧ a^2 + b^2 = 4/5 := by
sorry

end NUMINAMATH_CALUDE_max_ab_min_a2_b2_l504_50459


namespace NUMINAMATH_CALUDE_unique_row_with_41_l504_50456

/-- The number of rows in Pascal's Triangle containing 41 -/
def rows_containing_41 : ℕ := 1

/-- 41 is prime -/
axiom prime_41 : Nat.Prime 41

/-- Definition of binomial coefficient -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- 41 appears as a binomial coefficient -/
axiom exists_41_binomial : ∃ n k : ℕ, binomial n k = 41

theorem unique_row_with_41 : 
  (∃! r : ℕ, ∃ k : ℕ, binomial r k = 41) ∧ rows_containing_41 = 1 :=
sorry

end NUMINAMATH_CALUDE_unique_row_with_41_l504_50456


namespace NUMINAMATH_CALUDE_larger_number_problem_l504_50424

theorem larger_number_problem (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 6) : x = 23 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l504_50424


namespace NUMINAMATH_CALUDE_right_triangle_leg_sum_equals_circle_diameters_sum_l504_50492

/-- A right-angled triangle with its inscribed and circumscribed circles -/
structure RightTriangle where
  /-- The length of one leg of the right triangle -/
  leg1 : ℝ
  /-- The length of the other leg of the right triangle -/
  leg2 : ℝ
  /-- The radius of the inscribed circle -/
  inradius : ℝ
  /-- The radius of the circumscribed circle -/
  circumradius : ℝ
  /-- All lengths are positive -/
  leg1_pos : 0 < leg1
  leg2_pos : 0 < leg2
  inradius_pos : 0 < inradius
  circumradius_pos : 0 < circumradius

/-- 
In a right-angled triangle, the sum of the lengths of the two legs 
is equal to the sum of the diameters of the inscribed and circumscribed circles
-/
theorem right_triangle_leg_sum_equals_circle_diameters_sum (t : RightTriangle) :
  t.leg1 + t.leg2 = 2 * t.inradius + 2 * t.circumradius := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_leg_sum_equals_circle_diameters_sum_l504_50492


namespace NUMINAMATH_CALUDE_mans_speed_against_current_l504_50420

/-- Given a man's speed with the current and the speed of the current, 
    calculate the man's speed against the current. -/
def speedAgainstCurrent (speedWithCurrent : ℝ) (currentSpeed : ℝ) : ℝ :=
  speedWithCurrent - 2 * currentSpeed

/-- Theorem stating that given the specific speeds mentioned in the problem, 
    the man's speed against the current is 10 km/hr. -/
theorem mans_speed_against_current :
  speedAgainstCurrent 15 2.5 = 10 := by
  sorry

#eval speedAgainstCurrent 15 2.5

end NUMINAMATH_CALUDE_mans_speed_against_current_l504_50420


namespace NUMINAMATH_CALUDE_min_value_of_expression_l504_50412

theorem min_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y ≤ 1) :
  x^4 + y^4 - x^2*y - x*y^2 ≥ -1/8 ∧ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b ≤ 1 ∧ a^4 + b^4 - a^2*b - a*b^2 = -1/8 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l504_50412


namespace NUMINAMATH_CALUDE_min_framing_for_enlarged_picture_l504_50468

/-- Calculates the minimum number of linear feet of framing needed for an enlarged picture with a border. -/
def min_framing_feet (original_width original_height enlarge_factor border_width : ℕ) : ℕ :=
  let enlarged_width := original_width * enlarge_factor
  let enlarged_height := original_height * enlarge_factor
  let final_width := enlarged_width + 2 * border_width
  let final_height := enlarged_height + 2 * border_width
  let perimeter_inches := 2 * (final_width + final_height)
  let perimeter_feet := (perimeter_inches + 11) / 12  -- Round up to nearest foot
  perimeter_feet

/-- The theorem states that for a 4-inch by 6-inch picture enlarged by quadrupling its dimensions
    and adding a 3-inch border on each side, the minimum number of linear feet of framing needed is 9. -/
theorem min_framing_for_enlarged_picture :
  min_framing_feet 4 6 4 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_min_framing_for_enlarged_picture_l504_50468


namespace NUMINAMATH_CALUDE_regular_polygon_interior_angle_sides_l504_50461

theorem regular_polygon_interior_angle_sides : ∀ n : ℕ,
  n > 2 →
  (180 * (n - 2) : ℝ) / n = 150 →
  n = 12 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_interior_angle_sides_l504_50461


namespace NUMINAMATH_CALUDE_sin_cos_sum_11_19_l504_50466

theorem sin_cos_sum_11_19 : 
  Real.sin (11 * π / 180) * Real.cos (19 * π / 180) + 
  Real.cos (11 * π / 180) * Real.sin (19 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_11_19_l504_50466


namespace NUMINAMATH_CALUDE_log_equation_root_range_l504_50484

theorem log_equation_root_range (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 1 < x ∧ x < 3 ∧ 1 < y ∧ y < 3 ∧ 
   Real.log (x - 1) + Real.log (3 - x) = Real.log (a - x) ∧
   Real.log (y - 1) + Real.log (3 - y) = Real.log (a - y)) →
  (3 < a ∧ a < 13/4) :=
by sorry

end NUMINAMATH_CALUDE_log_equation_root_range_l504_50484


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l504_50463

universe u

def U : Set (Fin 5) := {1, 2, 3, 4, 5}
def M : Set (Fin 5) := {1, 2}
def N : Set (Fin 5) := {3, 5}

theorem complement_intersection_theorem :
  (U \ M) ∩ N = {3, 5} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l504_50463


namespace NUMINAMATH_CALUDE_cistern_problem_l504_50482

/-- Calculates the total wet surface area of a rectangular cistern -/
def cistern_wet_surface_area (length width depth : ℝ) : ℝ :=
  length * width + 2 * length * depth + 2 * width * depth

theorem cistern_problem :
  let length : ℝ := 8
  let width : ℝ := 6
  let depth : ℝ := 1.85
  cistern_wet_surface_area length width depth = 99.8 := by
  sorry

end NUMINAMATH_CALUDE_cistern_problem_l504_50482


namespace NUMINAMATH_CALUDE_shaded_cells_after_five_minutes_l504_50449

/-- Represents the state of the grid at a given minute -/
def GridState := Nat → Nat → Bool

/-- The initial state of the grid with a 1 × 5 shaded rectangle -/
def initial_state : GridState := sorry

/-- The rule for shading cells in the next minute -/
def shade_rule (state : GridState) : GridState := sorry

/-- The state of the grid after n minutes -/
def state_after (n : Nat) : GridState := sorry

/-- Counts the number of shaded cells in a given state -/
def count_shaded (state : GridState) : Nat := sorry

/-- The main theorem: after 5 minutes, 105 cells are shaded -/
theorem shaded_cells_after_five_minutes :
  count_shaded (state_after 5) = 105 := by sorry

end NUMINAMATH_CALUDE_shaded_cells_after_five_minutes_l504_50449


namespace NUMINAMATH_CALUDE_mobile_bus_uses_gps_and_gis_mobile_bus_not_uses_remote_sensing_l504_50413

/-- Represents the technologies used in the "Mobile Bus" app --/
inductive MobileBusTechnology
  | GPS        : MobileBusTechnology
  | GIS        : MobileBusTechnology
  | RemoteSensing : MobileBusTechnology
  | DigitalEarth  : MobileBusTechnology

/-- The set of technologies used in the "Mobile Bus" app --/
def mobileBusTechnologies : Set MobileBusTechnology :=
  {MobileBusTechnology.GPS, MobileBusTechnology.GIS}

/-- Theorem stating that the "Mobile Bus" app uses GPS and GIS --/
theorem mobile_bus_uses_gps_and_gis :
  MobileBusTechnology.GPS ∈ mobileBusTechnologies ∧
  MobileBusTechnology.GIS ∈ mobileBusTechnologies :=
by sorry

/-- Theorem stating that the "Mobile Bus" app does not use Remote Sensing --/
theorem mobile_bus_not_uses_remote_sensing :
  MobileBusTechnology.RemoteSensing ∉ mobileBusTechnologies :=
by sorry

end NUMINAMATH_CALUDE_mobile_bus_uses_gps_and_gis_mobile_bus_not_uses_remote_sensing_l504_50413


namespace NUMINAMATH_CALUDE_complex_modulus_range_l504_50417

theorem complex_modulus_range (a : ℝ) : 
  (∀ θ : ℝ, Complex.abs ((a + Real.cos θ) + (2 * a - Real.sin θ) * Complex.I) ≤ 2) ↔ 
  a ∈ Set.Icc (-(Real.sqrt 5) / 5) ((Real.sqrt 5) / 5) := by sorry

end NUMINAMATH_CALUDE_complex_modulus_range_l504_50417


namespace NUMINAMATH_CALUDE_paris_weekday_study_hours_l504_50458

/-- The number of hours Paris studies each weekday. -/
def weekday_study_hours : ℝ := 3

/-- The number of weeks in the fall semester. -/
def semester_weeks : ℕ := 15

/-- The number of hours Paris studies on Saturday. -/
def saturday_study_hours : ℝ := 4

/-- The number of hours Paris studies on Sunday. -/
def sunday_study_hours : ℝ := 5

/-- The total number of hours Paris studies during the semester. -/
def total_study_hours : ℝ := 360

/-- Theorem stating that the number of hours Paris studies each weekday is 3. -/
theorem paris_weekday_study_hours :
  weekday_study_hours * (5 * semester_weeks) +
  (saturday_study_hours + sunday_study_hours) * semester_weeks =
  total_study_hours :=
sorry

end NUMINAMATH_CALUDE_paris_weekday_study_hours_l504_50458


namespace NUMINAMATH_CALUDE_simons_age_in_2010_l504_50457

/-- Given that Jorge is 24 years younger than Simon and Jorge is 16 years old in 2005,
    prove that Simon's age in 2010 is 45 years. -/
theorem simons_age_in_2010 (jorge_age_2005 : ℕ) (simon_jorge_age_diff : ℕ) :
  jorge_age_2005 = 16 →
  simon_jorge_age_diff = 24 →
  jorge_age_2005 + simon_jorge_age_diff + (2010 - 2005) = 45 := by
sorry

end NUMINAMATH_CALUDE_simons_age_in_2010_l504_50457


namespace NUMINAMATH_CALUDE_exam_sections_percentage_l504_50411

theorem exam_sections_percentage :
  let total_candidates : ℕ := 1200
  let all_sections_percent : ℚ := 5 / 100
  let no_sections_percent : ℚ := 5 / 100
  let one_section_percent : ℚ := 25 / 100
  let four_sections_percent : ℚ := 20 / 100
  let three_sections_count : ℕ := 300
  
  ∃ (two_sections_percent : ℚ),
    two_sections_percent = 20 / 100 ∧
    (all_sections_percent + no_sections_percent + one_section_percent + 
     four_sections_percent + two_sections_percent + 
     (three_sections_count : ℚ) / total_candidates) = 1 :=
by sorry

end NUMINAMATH_CALUDE_exam_sections_percentage_l504_50411


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l504_50410

-- Define the set A
def A : Set ℤ := {x : ℤ | -1 < x ∧ x ≤ 3}

-- Define the set B
def B : Set ℤ := {1, 2}

-- Theorem statement
theorem intersection_A_complement_B : A ∩ Bᶜ = {0, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l504_50410


namespace NUMINAMATH_CALUDE_triangle_base_calculation_l504_50454

/-- Given a triangle with area 46 cm² and height 10 cm, prove its base is 9.2 cm -/
theorem triangle_base_calculation (area : ℝ) (height : ℝ) (base : ℝ) :
  area = 46 →
  height = 10 →
  area = (base * height) / 2 →
  base = 9.2 := by
sorry

end NUMINAMATH_CALUDE_triangle_base_calculation_l504_50454


namespace NUMINAMATH_CALUDE_sqrt_sum_max_value_l504_50477

theorem sqrt_sum_max_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) :
  ∃ (max : ℝ), max = 2 ∧ ∀ (x y : ℝ), 0 < x → 0 < y → x + y = 2 → Real.sqrt x + Real.sqrt y ≤ max :=
sorry

end NUMINAMATH_CALUDE_sqrt_sum_max_value_l504_50477


namespace NUMINAMATH_CALUDE_positive_real_inequality_l504_50443

theorem positive_real_inequality (a : ℝ) (h : a > 0) : a + 1/a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_positive_real_inequality_l504_50443


namespace NUMINAMATH_CALUDE_basketball_baseball_volume_ratio_l504_50436

theorem basketball_baseball_volume_ratio : 
  ∀ (r R : ℝ), r > 0 → R = 4 * r → 
  (4 / 3 * Real.pi * R^3) / (4 / 3 * Real.pi * r^3) = 64 := by
sorry

end NUMINAMATH_CALUDE_basketball_baseball_volume_ratio_l504_50436


namespace NUMINAMATH_CALUDE_radical_simplification_l504_50446

theorem radical_simplification (q : ℝ) (hq : q > 0) :
  Real.sqrt (45 * q) * Real.sqrt (10 * q) * Real.sqrt (15 * q) = 675 * q * Real.sqrt q :=
by sorry

end NUMINAMATH_CALUDE_radical_simplification_l504_50446


namespace NUMINAMATH_CALUDE_election_vote_difference_l504_50407

theorem election_vote_difference (total_votes : ℕ) (candidate_percentage : ℚ) : 
  total_votes = 4400 →
  candidate_percentage = 30 / 100 →
  (total_votes : ℚ) * candidate_percentage - (total_votes : ℚ) * (1 - candidate_percentage) = -1760 := by
  sorry

end NUMINAMATH_CALUDE_election_vote_difference_l504_50407


namespace NUMINAMATH_CALUDE_min_correct_answers_for_min_score_l504_50422

/-- Represents the scoring system and conditions of the AMC 12 test -/
structure AMC12Test where
  total_questions : Nat
  attempted_questions : Nat
  correct_points : Int
  incorrect_points : Int
  unanswered_points : Int
  min_score : Int

/-- Calculates the score based on the number of correct answers -/
def calculate_score (test : AMC12Test) (correct_answers : Nat) : Int :=
  let incorrect_answers := test.attempted_questions - correct_answers
  let unanswered_questions := test.total_questions - test.attempted_questions
  correct_answers * test.correct_points +
  incorrect_answers * test.incorrect_points +
  unanswered_questions * test.unanswered_points

/-- Theorem stating the minimum number of correct answers needed to achieve the minimum score -/
theorem min_correct_answers_for_min_score (test : AMC12Test)
  (h1 : test.total_questions = 35)
  (h2 : test.attempted_questions = 30)
  (h3 : test.correct_points = 7)
  (h4 : test.incorrect_points = -1)
  (h5 : test.unanswered_points = 2)
  (h6 : test.min_score = 150) :
  ∃ (n : Nat), n = 20 ∧ 
    (∀ (m : Nat), m < n → calculate_score test m < test.min_score) ∧
    calculate_score test n ≥ test.min_score :=
  sorry

end NUMINAMATH_CALUDE_min_correct_answers_for_min_score_l504_50422


namespace NUMINAMATH_CALUDE_fixed_salary_is_400_l504_50423

/-- Represents the fixed salary in the new commission scheme -/
def fixed_salary : ℕ := sorry

/-- Represents the total sales amount -/
def total_sales : ℕ := 12000

/-- Represents the threshold for commission in the new scheme -/
def commission_threshold : ℕ := 4000

/-- Calculates the commission under the old scheme -/
def old_commission (sales : ℕ) : ℕ :=
  (sales * 5) / 100

/-- Calculates the commission under the new scheme -/
def new_commission (sales : ℕ) : ℕ :=
  ((sales - commission_threshold) * 25) / 1000

/-- States that the new scheme pays 600 more than the old scheme -/
axiom new_scheme_difference : 
  fixed_salary + new_commission total_sales = old_commission total_sales + 600

theorem fixed_salary_is_400 : fixed_salary = 400 := by
  sorry

end NUMINAMATH_CALUDE_fixed_salary_is_400_l504_50423


namespace NUMINAMATH_CALUDE_unique_integer_solution_l504_50419

theorem unique_integer_solution :
  ∃! (x y z : ℤ), x^2 + y^2 + z^2 = x^2 * y^2 :=
by sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l504_50419


namespace NUMINAMATH_CALUDE_tangent_line_is_perpendicular_and_tangent_l504_50460

-- Define the given line
def given_line (x y : ℝ) : Prop := 2 * x - 6 * y + 1 = 0

-- Define the given curve
def given_curve (x y : ℝ) : Prop := y = x^3 + 3 * x^2 - 5

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := 3 * x + y + 6 = 0

-- Theorem statement
theorem tangent_line_is_perpendicular_and_tangent :
  ∃ (x₀ y₀ : ℝ),
    -- The point (x₀, y₀) is on the curve
    given_curve x₀ y₀ ∧
    -- The tangent line passes through (x₀, y₀)
    tangent_line x₀ y₀ ∧
    -- The tangent line is perpendicular to the given line
    (∀ (x₁ y₁ x₂ y₂ : ℝ),
      given_line x₁ y₁ ∧ given_line x₂ y₂ ∧ x₁ ≠ x₂ →
      (y₂ - y₁) / (x₂ - x₁) * ((y₀ + 6) / (-3) - y₀) / (((y₀ + 6) / (-3)) - x₀) = -1) ∧
    -- The tangent line is indeed tangent to the curve
    (∀ (x : ℝ), x ≠ x₀ → ∃ (y : ℝ), given_curve x y ∧ ¬tangent_line x y) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_is_perpendicular_and_tangent_l504_50460


namespace NUMINAMATH_CALUDE_set_operations_l504_50409

-- Define the sets A and B
def A : Set ℝ := {x | x - 2 ≥ 0}
def B : Set ℝ := {x | x < 3}

-- Define the theorem
theorem set_operations :
  (A ∪ B = Set.univ) ∧
  (A ∩ B = {x | 2 ≤ x ∧ x < 3}) ∧
  ((Aᶜ ∪ Bᶜ) = {x | x < 2 ∨ x ≥ 3}) := by
  sorry


end NUMINAMATH_CALUDE_set_operations_l504_50409


namespace NUMINAMATH_CALUDE_exactly_three_proper_sets_l504_50429

/-- A set of weights is proper if it can balance any weight from 1 to 200 grams uniquely -/
def IsProperSet (s : Multiset ℕ) : Prop :=
  (s.sum = 200) ∧
  (∀ w : ℕ, w ≥ 1 ∧ w ≤ 200 → ∃! subset : Multiset ℕ, subset ⊆ s ∧ subset.sum = w)

/-- The number of different proper sets of weights -/
def NumberOfProperSets : ℕ := 3

/-- Theorem stating that there are exactly 3 different proper sets of weights -/
theorem exactly_three_proper_sets :
  (∃ (sets : Finset (Multiset ℕ)), sets.card = NumberOfProperSets ∧
    (∀ s : Multiset ℕ, s ∈ sets ↔ IsProperSet s)) ∧
  (¬∃ (sets : Finset (Multiset ℕ)), sets.card > NumberOfProperSets ∧
    (∀ s : Multiset ℕ, s ∈ sets ↔ IsProperSet s)) :=
sorry

end NUMINAMATH_CALUDE_exactly_three_proper_sets_l504_50429


namespace NUMINAMATH_CALUDE_max_a_value_l504_50450

-- Define the quadratic polynomial f(x) = x^2 + ax + b
def f (a b x : ℝ) : ℝ := x^2 + a*x + b

-- State the theorem
theorem max_a_value (a b : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, f a b y = f a b x + y) →
  a ≤ (1/2 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_max_a_value_l504_50450


namespace NUMINAMATH_CALUDE_ski_lift_time_l504_50452

theorem ski_lift_time (ski_down_time : ℝ) (num_trips : ℕ) (total_time : ℝ) 
  (h1 : ski_down_time = 5)
  (h2 : num_trips = 6)
  (h3 : total_time = 120) : 
  (total_time - num_trips * ski_down_time) / num_trips = 15 := by
sorry

end NUMINAMATH_CALUDE_ski_lift_time_l504_50452


namespace NUMINAMATH_CALUDE_hyperbola_C_properties_l504_50496

-- Define the hyperbola C
def C (x y : ℝ) : Prop := x^2/3 - y^2/12 = 1

-- Define the reference hyperbola
def ref_hyperbola (x y : ℝ) : Prop := y^2/4 - x^2 = 1

-- Theorem statement
theorem hyperbola_C_properties :
  -- C passes through (2,2)
  C 2 2 ∧
  -- C has the same asymptotes as the reference hyperbola
  (∀ x y : ℝ, C x y ↔ ∃ k : ℝ, k ≠ 0 ∧ ref_hyperbola (x/k) (y/k)) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_C_properties_l504_50496


namespace NUMINAMATH_CALUDE_city_population_ratio_l504_50427

theorem city_population_ratio (pop_x pop_y pop_z : ℝ) 
  (h1 : pop_x = 5 * pop_y) 
  (h2 : pop_x / pop_z = 10) : 
  pop_y / pop_z = 2 := by
sorry

end NUMINAMATH_CALUDE_city_population_ratio_l504_50427


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l504_50404

theorem necessary_but_not_sufficient : 
  (∀ x : ℝ, (x - 1) * (x - 2) ≤ 0 → x^2 - 3*x ≤ 0) ∧ 
  (∃ x : ℝ, x^2 - 3*x ≤ 0 ∧ (x - 1) * (x - 2) > 0) := by
  sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l504_50404


namespace NUMINAMATH_CALUDE_second_class_average_l504_50445

theorem second_class_average (n₁ n₂ : ℕ) (avg₁ avg_total : ℚ) : 
  n₁ = 12 → 
  n₂ = 28 → 
  avg₁ = 40 → 
  avg_total = 54 → 
  ∃ avg₂ : ℚ, 
    (n₁ * avg₁ + n₂ * avg₂) / (n₁ + n₂) = avg_total ∧ 
    avg₂ = 60 :=
by sorry

end NUMINAMATH_CALUDE_second_class_average_l504_50445


namespace NUMINAMATH_CALUDE_max_third_term_arithmetic_sequence_l504_50451

theorem max_third_term_arithmetic_sequence (a d : ℕ) : 
  a > 0 → d > 0 → a + (a + d) + (a + 2*d) + (a + 3*d) = 50 → 
  ∀ (b e : ℕ), b > 0 → e > 0 → b + (b + e) + (b + 2*e) + (b + 3*e) = 50 → 
  (a + 2*d) ≤ 16 := by
sorry

end NUMINAMATH_CALUDE_max_third_term_arithmetic_sequence_l504_50451


namespace NUMINAMATH_CALUDE_max_value_of_expression_l504_50416

theorem max_value_of_expression (x y a : ℝ) (hx : x > 0) (hy : y > 0) (ha : a > 0) :
  (x + y + a)^2 / (x^2 + y^2 + a^2) ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l504_50416


namespace NUMINAMATH_CALUDE_sin_45_degrees_l504_50467

theorem sin_45_degrees : Real.sin (π / 4) = 1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_45_degrees_l504_50467


namespace NUMINAMATH_CALUDE_sum_first_n_even_sum_even_2_to_34_sum_even_28_to_50_l504_50441

-- Define the sum of first n even numbers
def sumFirstNEven (n : ℕ) : ℕ := n * (n + 1)

-- Define the sum of even numbers from a to b
def sumEvenRange (a b : ℕ) : ℕ := 
  sumFirstNEven (b / 2) - sumFirstNEven ((a - 2) / 2)

-- Theorem 1: The sum of the first n even numbers is equal to n * (n+1)
theorem sum_first_n_even (n : ℕ) : 
  2 * (n * (n + 1) / 2) = n * (n + 1) := by sorry

-- Theorem 2: The sum of even numbers from 2 to 34 is 306
theorem sum_even_2_to_34 : sumEvenRange 2 34 = 306 := by sorry

-- Theorem 3: The sum of even numbers from 28 to 50 is 468
theorem sum_even_28_to_50 : sumEvenRange 28 50 = 468 := by sorry

end NUMINAMATH_CALUDE_sum_first_n_even_sum_even_2_to_34_sum_even_28_to_50_l504_50441


namespace NUMINAMATH_CALUDE_new_average_production_theorem_l504_50473

/-- Calculates the new average daily production after adding a new day's production -/
def newAverageDailyProduction (n : ℕ) (pastAverage : ℚ) (todayProduction : ℚ) : ℚ :=
  ((n : ℚ) * pastAverage + todayProduction) / ((n : ℚ) + 1)

theorem new_average_production_theorem :
  let n : ℕ := 8
  let pastAverage : ℚ := 50
  let todayProduction : ℚ := 95
  newAverageDailyProduction n pastAverage todayProduction = 55 := by
  sorry

end NUMINAMATH_CALUDE_new_average_production_theorem_l504_50473


namespace NUMINAMATH_CALUDE_relationship_fg_l504_50418

noncomputable def f (x : ℝ) := Real.exp x + x - 2
noncomputable def g (x : ℝ) := Real.log x + x^2 - 3

theorem relationship_fg (a b : ℝ) (h1 : f a = 0) (h2 : g b = 0) :
  g a < 0 ∧ 0 < f b := by sorry

end NUMINAMATH_CALUDE_relationship_fg_l504_50418


namespace NUMINAMATH_CALUDE_junk_items_after_transactions_l504_50405

/-- Represents the composition of items in the attic -/
structure AtticComposition where
  useful : Rat
  valuable : Rat
  junk : Rat

/-- Represents the number of items in each category -/
structure AtticItems where
  useful : ℕ
  valuable : ℕ
  junk : ℕ

/-- The theorem to prove -/
theorem junk_items_after_transactions 
  (initial_composition : AtticComposition)
  (initial_items : AtticItems)
  (items_removed : AtticItems)
  (final_composition : AtticComposition)
  (final_useful_items : ℕ) :
  (initial_composition.useful = 1/5) →
  (initial_composition.valuable = 1/10) →
  (initial_composition.junk = 7/10) →
  (items_removed.useful = 4) →
  (items_removed.valuable = 3) →
  (final_composition.useful = 1/4) →
  (final_composition.valuable = 3/20) →
  (final_composition.junk = 3/5) →
  (final_useful_items = 20) →
  ∃ (final_items : AtticItems), final_items.junk = 48 := by
  sorry

end NUMINAMATH_CALUDE_junk_items_after_transactions_l504_50405


namespace NUMINAMATH_CALUDE_product_difference_equals_2019_l504_50402

theorem product_difference_equals_2019 : 672 * 673 * 674 - 671 * 673 * 675 = 2019 := by
  sorry

end NUMINAMATH_CALUDE_product_difference_equals_2019_l504_50402


namespace NUMINAMATH_CALUDE_no_universal_divisor_l504_50408

-- Define a function to represent the concatenation of digits
def concat_digits (a b : ℕ) : ℕ := sorry

-- Define a function to represent the concatenation of three digits
def concat_three_digits (a n b : ℕ) : ℕ := sorry

-- Theorem statement
theorem no_universal_divisor :
  ¬ ∃ n : ℕ, ∀ a b : ℕ, 
    a ≠ 0 → b ≠ 0 → a < 10 → b < 10 → 
    (concat_three_digits a n b) % (concat_digits a b) = 0 := by sorry

end NUMINAMATH_CALUDE_no_universal_divisor_l504_50408


namespace NUMINAMATH_CALUDE_dvd_packs_after_discount_l504_50479

theorem dvd_packs_after_discount (original_price discount available : ℕ) : 
  original_price = 107 → 
  discount = 106 → 
  available = 93 → 
  (available / (original_price - discount) : ℕ) = 93 := by
sorry

end NUMINAMATH_CALUDE_dvd_packs_after_discount_l504_50479


namespace NUMINAMATH_CALUDE_max_visible_cuboids_6x6x6_l504_50494

/-- Represents a cuboid with integer dimensions -/
structure Cuboid where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents a cube composed of smaller cuboids -/
structure CompositeCube where
  side_length : ℕ
  small_cuboid : Cuboid
  num_small_cuboids : ℕ

/-- Function to calculate the maximum number of visible small cuboids -/
def max_visible_cuboids (cube : CompositeCube) : ℕ :=
  sorry

/-- Theorem stating the maximum number of visible small cuboids for the given problem -/
theorem max_visible_cuboids_6x6x6 :
  let small_cuboid : Cuboid := ⟨3, 2, 1⟩
  let large_cube : CompositeCube := ⟨6, small_cuboid, 36⟩
  max_visible_cuboids large_cube = 31 :=
by sorry

end NUMINAMATH_CALUDE_max_visible_cuboids_6x6x6_l504_50494


namespace NUMINAMATH_CALUDE_class_composition_l504_50448

theorem class_composition (d m : ℕ) : 
  (d : ℚ) / (d + m : ℚ) = 3/5 →
  ((d - 1 : ℚ) / (d + m - 3 : ℚ) = 5/8) →
  d = 21 ∧ m = 14 := by
sorry

end NUMINAMATH_CALUDE_class_composition_l504_50448


namespace NUMINAMATH_CALUDE_original_triangle_area_l504_50400

theorem original_triangle_area (original_area new_area : ℝ) : 
  (∀ side, new_side = 5 * side) → 
  new_area = 125 → 
  original_area = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_original_triangle_area_l504_50400


namespace NUMINAMATH_CALUDE_average_of_data_set_l504_50486

def data_set : List ℤ := [3, -2, 4, 1, 4]

theorem average_of_data_set :
  (data_set.sum : ℚ) / data_set.length = 2 := by
  sorry

end NUMINAMATH_CALUDE_average_of_data_set_l504_50486


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_3x_plus_2_to_8_l504_50433

theorem coefficient_x_cubed_3x_plus_2_to_8 : 
  (Finset.range 9).sum (λ k => Nat.choose 8 k * (3 ^ k) * (2 ^ (8 - k)) * if k = 3 then 1 else 0) = 48384 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_3x_plus_2_to_8_l504_50433


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l504_50401

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

/-- Given a geometric sequence {aₙ} where a₂ = 4 and a₆a₇ = 16a₉, prove that a₅ = ±32. -/
theorem geometric_sequence_problem (a : ℕ → ℝ) 
    (h_geo : IsGeometricSequence a) 
    (h_a2 : a 2 = 4)
    (h_a6a7 : a 6 * a 7 = 16 * a 9) : 
  a 5 = 32 ∨ a 5 = -32 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l504_50401


namespace NUMINAMATH_CALUDE_sandbox_cost_l504_50432

/-- The cost to fill a rectangular sandbox with sand -/
theorem sandbox_cost (length width depth price_per_cubic_foot : ℝ) 
  (h_length : length = 4)
  (h_width : width = 3)
  (h_depth : depth = 1.5)
  (h_price : price_per_cubic_foot = 3) : 
  length * width * depth * price_per_cubic_foot = 54 := by
  sorry

#check sandbox_cost

end NUMINAMATH_CALUDE_sandbox_cost_l504_50432


namespace NUMINAMATH_CALUDE_expected_balls_in_original_position_l504_50485

/-- The number of balls arranged in a circle -/
def n : ℕ := 7

/-- The probability of a ball being swapped twice -/
def p_twice : ℚ := 2 / (n * n)

/-- The probability of a ball never being swapped -/
def p_never : ℚ := (n - 2)^2 / (n * n)

/-- The probability of a ball being in its original position after two transpositions -/
def p_original : ℚ := p_twice + p_never

/-- The expected number of balls in their original positions after two transpositions -/
def expected_original : ℚ := n * p_original

theorem expected_balls_in_original_position :
  expected_original = 189 / 49 := by sorry

end NUMINAMATH_CALUDE_expected_balls_in_original_position_l504_50485


namespace NUMINAMATH_CALUDE_charlottes_age_l504_50478

theorem charlottes_age (B E C : ℚ) 
  (h1 : B = 4 * C)
  (h2 : E = C + 5)
  (h3 : B = E) :
  C = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_charlottes_age_l504_50478


namespace NUMINAMATH_CALUDE_total_distance_traveled_l504_50415

theorem total_distance_traveled (XY XZ : ℝ) (h1 : XY = 4500) (h2 : XZ = 4000) : 
  XY + Real.sqrt (XY^2 - XZ^2) + XZ = 10562 := by
sorry

end NUMINAMATH_CALUDE_total_distance_traveled_l504_50415


namespace NUMINAMATH_CALUDE_prime_product_minus_sum_l504_50488

theorem prime_product_minus_sum : ∃ p q : ℕ, 
  Prime p ∧ Prime q ∧ 
  p ≠ q ∧ 
  4 < p ∧ p < 18 ∧ 
  4 < q ∧ q < 18 ∧ 
  p * q - (p + q) = 119 := by
sorry

end NUMINAMATH_CALUDE_prime_product_minus_sum_l504_50488


namespace NUMINAMATH_CALUDE_bowtie_equation_l504_50483

-- Define the ⋈ operation
noncomputable def bowtie (a b : ℝ) : ℝ :=
  a + Real.sqrt (b + 3 + Real.sqrt (b + 3 + Real.sqrt (b + 3 + Real.sqrt (b + 3))))

-- State the theorem
theorem bowtie_equation (g : ℝ) : bowtie 8 g = 11 → g = 3 := by
  sorry

end NUMINAMATH_CALUDE_bowtie_equation_l504_50483


namespace NUMINAMATH_CALUDE_subtraction_result_l504_50438

def largest_3digit_number : ℕ := 999
def smallest_5digit_number : ℕ := 10000

theorem subtraction_result : 
  smallest_5digit_number - largest_3digit_number = 9001 := by sorry

end NUMINAMATH_CALUDE_subtraction_result_l504_50438


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l504_50491

/-- Given a sequence a_n where a_2 = 2, a_6 = 0, and {1 / (a_n + 1)} is an arithmetic sequence,
    prove that a_4 = 1/2 -/
theorem arithmetic_sequence_problem (a : ℕ → ℚ) 
  (h1 : a 2 = 2)
  (h2 : a 6 = 0)
  (h3 : ∃ d : ℚ, ∀ n : ℕ, 1 / (a (n + 1) + 1) - 1 / (a n + 1) = d) :
  a 4 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l504_50491


namespace NUMINAMATH_CALUDE_other_integer_17_or_21_l504_50493

/-- Two consecutive odd integers with a sum of at least 36, one being 19 -/
structure ConsecutiveOddIntegers where
  n : ℤ
  sum_at_least_36 : n + (n + 2) ≥ 36
  one_is_19 : n = 19 ∨ n + 2 = 19

/-- The other integer is either 17 or 21 -/
theorem other_integer_17_or_21 (x : ConsecutiveOddIntegers) : 
  x.n = 21 ∨ x.n = 17 := by
  sorry


end NUMINAMATH_CALUDE_other_integer_17_or_21_l504_50493


namespace NUMINAMATH_CALUDE_sequence_inequality_existence_l504_50428

theorem sequence_inequality_existence (a b : ℕ → ℕ) :
  ∃ p q : ℕ, p < q ∧ a p ≤ a q ∧ b p ≤ b q := by
  sorry

end NUMINAMATH_CALUDE_sequence_inequality_existence_l504_50428


namespace NUMINAMATH_CALUDE_not_perfect_square_l504_50480

theorem not_perfect_square (n : ℤ) (h : n > 11) :
  ¬ ∃ m : ℤ, n^2 - 19*n + 89 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l504_50480


namespace NUMINAMATH_CALUDE_solution_product_log_l504_50421

-- Define the system of equations
def system_of_equations (x y : ℝ) : Prop :=
  (Real.log x / Real.log 225 + Real.log y / Real.log 64 = 4) ∧
  (Real.log 225 / Real.log x - Real.log 64 / Real.log y = 1)

-- State the theorem
theorem solution_product_log (x₁ y₁ x₂ y₂ : ℝ) :
  system_of_equations x₁ y₁ ∧ system_of_equations x₂ y₂ →
  Real.log (x₁ * y₁ * x₂ * y₂) / Real.log 30 = 12 :=
sorry

end NUMINAMATH_CALUDE_solution_product_log_l504_50421


namespace NUMINAMATH_CALUDE_unique_solution_l504_50434

/-- The product of all digits of a positive integer -/
def digit_product (n : ℕ+) : ℕ := sorry

/-- Theorem stating that 12 is the only positive integer solution -/
theorem unique_solution : 
  ∃! (x : ℕ+), digit_product x = x^2 - 10*x - 22 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l504_50434


namespace NUMINAMATH_CALUDE_collinear_points_x_value_l504_50490

/-- Given three collinear points A(-1, 2), B(2, 4), and C(x, 3), prove that x = 1/2 --/
theorem collinear_points_x_value :
  let A : ℝ × ℝ := (-1, 2)
  let B : ℝ × ℝ := (2, 4)
  let C : ℝ × ℝ := (x, 3)
  (∀ t : ℝ, ∃ u v : ℝ, u * (B.1 - A.1) + v * (C.1 - A.1) = t * (B.1 - A.1) ∧
                       u * (B.2 - A.2) + v * (C.2 - A.2) = t * (B.2 - A.2)) →
  x = 1/2 := by
sorry

end NUMINAMATH_CALUDE_collinear_points_x_value_l504_50490


namespace NUMINAMATH_CALUDE_right_triangle_inscribed_in_equilateral_l504_50487

theorem right_triangle_inscribed_in_equilateral (XC BX CZ : ℝ) :
  XC = 4 →
  BX = 3 →
  CZ = 3 →
  let XZ := XC + CZ
  let XY := XZ
  let YZ := XZ
  let BC := Real.sqrt (BX^2 + XC^2 - 2 * BX * XC * Real.cos (π/3))
  let AB := Real.sqrt (BX^2 + BC^2)
  let AZ := Real.sqrt (CZ^2 + BC^2)
  AB^2 = BC^2 + AZ^2 →
  AZ = 3 := by sorry

end NUMINAMATH_CALUDE_right_triangle_inscribed_in_equilateral_l504_50487


namespace NUMINAMATH_CALUDE_product_of_terms_l504_50499

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The roots of the quadratic equation 2x^2 + 5x + 1 = 0 -/
def roots_of_equation (x y : ℝ) : Prop :=
  2 * x^2 + 5 * x + 1 = 0 ∧ 2 * y^2 + 5 * y + 1 = 0

theorem product_of_terms (a : ℕ → ℝ) :
  geometric_sequence a →
  roots_of_equation (a 1) (a 10) →
  a 4 * a 7 = 1/2 := by sorry

end NUMINAMATH_CALUDE_product_of_terms_l504_50499


namespace NUMINAMATH_CALUDE_square_hall_tiles_l504_50481

theorem square_hall_tiles (black_tiles : ℕ) (miscounted_tiles : ℕ) : 
  black_tiles = 153 ∧ miscounted_tiles = 3 →
  ∃ (side_length : ℕ), 
    side_length * 2 = black_tiles + miscounted_tiles ∧
    side_length * side_length = 6084 :=
by sorry

end NUMINAMATH_CALUDE_square_hall_tiles_l504_50481


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l504_50430

theorem smallest_solution_of_equation : ∃ x : ℝ, 
  (∀ y : ℝ, y^4 - 26*y^2 + 169 = 0 → x ≤ y) ∧ 
  x^4 - 26*x^2 + 169 = 0 ∧ 
  x = -Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l504_50430


namespace NUMINAMATH_CALUDE_election_vote_count_l504_50474

/-- Given that the ratio of votes for candidate A to candidate B is 2:1,
    and candidate A received 14 votes, prove that the total number of
    votes for both candidates is 21. -/
theorem election_vote_count (votes_A : ℕ) (votes_B : ℕ) : 
  votes_A = 14 → 
  votes_A = 2 * votes_B → 
  votes_A + votes_B = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_election_vote_count_l504_50474


namespace NUMINAMATH_CALUDE_truthful_dwarfs_count_l504_50442

theorem truthful_dwarfs_count :
  ∀ (total_dwarfs : ℕ) 
    (vanilla_hands chocolate_hands fruit_hands : ℕ),
  total_dwarfs = 10 →
  vanilla_hands = total_dwarfs →
  chocolate_hands = total_dwarfs / 2 →
  fruit_hands = 1 →
  ∃ (truthful_dwarfs : ℕ),
    truthful_dwarfs = 4 ∧
    truthful_dwarfs + (total_dwarfs - truthful_dwarfs) = total_dwarfs ∧
    vanilla_hands + chocolate_hands + fruit_hands = 
      total_dwarfs + (total_dwarfs - truthful_dwarfs) :=
by sorry

end NUMINAMATH_CALUDE_truthful_dwarfs_count_l504_50442


namespace NUMINAMATH_CALUDE_cubic_roots_inequality_l504_50437

theorem cubic_roots_inequality (a b c d : ℝ) (h1 : a ≠ 0) 
  (h2 : ∃ (x₁ x₂ x₃ : ℝ), x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    a * x₁^3 + b * x₁^2 + c * x₁ + d = 0 ∧
    a * x₂^3 + b * x₂^2 + c * x₂ + d = 0 ∧
    a * x₃^3 + b * x₃^2 + c * x₃ + d = 0) :
  b * c < 3 * a * d := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_inequality_l504_50437


namespace NUMINAMATH_CALUDE_trigonometric_identity_l504_50453

theorem trigonometric_identity : 
  let a : Real := 2 * Real.pi / 3
  Real.sin (Real.pi - a / 2) + Real.tan (a - 5 * Real.pi / 12) = (2 + Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l504_50453


namespace NUMINAMATH_CALUDE_compute_alpha_l504_50498

theorem compute_alpha (α β : ℂ) 
  (h1 : (α + β).re > 0)
  (h2 : (Complex.I * (2 * α - 3 * β)).re > 0)
  (h3 : β = 4 + 3 * Complex.I) :
  α = 6 + (3 / 2) * Complex.I := by sorry

end NUMINAMATH_CALUDE_compute_alpha_l504_50498


namespace NUMINAMATH_CALUDE_mary_flour_calculation_l504_50440

/-- The number of cups of flour Mary put in -/
def flour_put_in : ℕ := 2

/-- The total number of cups of flour required by the recipe -/
def total_flour : ℕ := 10

/-- The number of cups of sugar required by the recipe -/
def sugar : ℕ := 3

/-- The additional cups of flour needed compared to sugar -/
def extra_flour : ℕ := 5

theorem mary_flour_calculation :
  flour_put_in = total_flour - (sugar + extra_flour) :=
by sorry

end NUMINAMATH_CALUDE_mary_flour_calculation_l504_50440


namespace NUMINAMATH_CALUDE_problem_statement_l504_50431

noncomputable def f (x : ℝ) := Real.exp x + Real.exp (-x)

theorem problem_statement :
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ m : ℝ, (∀ x > 0, m * f x ≤ Real.exp (-x) + m - 1) → m ≤ -1/3) ∧
  (∀ a > 0, (∃ x₀ ≥ 1, f x₀ < a * (-x₀^2 + 3*x₀)) → 
    ((1/2*(Real.exp 1 + Real.exp (-1)) < a ∧ a < Real.exp 1 → a^(Real.exp 1 - 1) > Real.exp (a - 1)) ∧
     (a > Real.exp 1 → a^(Real.exp 1 - 1) < Real.exp (a - 1)))) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l504_50431


namespace NUMINAMATH_CALUDE_tire_swap_optimal_l504_50469

/-- Represents the wear rate of a tire in km^(-1) -/
def WearRate := ℝ

/-- Calculates the remaining life of a tire after driving a certain distance -/
def remaining_life (total_life : ℝ) (distance_driven : ℝ) : ℝ :=
  total_life - distance_driven

/-- Theorem: Swapping tires at 9375 km results in simultaneous wear-out -/
theorem tire_swap_optimal (front_life rear_life swap_distance : ℝ)
  (h_front : front_life = 25000)
  (h_rear : rear_life = 15000)
  (h_swap : swap_distance = 9375) :
  remaining_life front_life swap_distance / rear_life =
  remaining_life rear_life swap_distance / front_life := by
  sorry

#check tire_swap_optimal

end NUMINAMATH_CALUDE_tire_swap_optimal_l504_50469


namespace NUMINAMATH_CALUDE_largest_number_given_hcf_and_lcm_factors_l504_50414

theorem largest_number_given_hcf_and_lcm_factors (a b : ℕ+) : 
  (Nat.gcd a b = 40) → 
  (∃ (k : ℕ+), Nat.lcm a b = 40 * 11 * 12 * k) → 
  (max a b = 480) := by
sorry

end NUMINAMATH_CALUDE_largest_number_given_hcf_and_lcm_factors_l504_50414


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l504_50464

theorem fixed_point_of_exponential_function 
  (a : ℝ) (ha : a > 0 ∧ a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ 7 + a^(x - 3)
  f 3 = 8 := by
sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l504_50464
