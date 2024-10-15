import Mathlib

namespace NUMINAMATH_CALUDE_intersection_M_N_l1038_103806

def M : Set ℤ := {-1, 1}
def N : Set ℤ := {x | x^2 + x = 0}

theorem intersection_M_N : M ∩ N = {-1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1038_103806


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l1038_103881

theorem perfect_square_trinomial (b : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, 16 * x^2 - b * x + 9 = (a * x + 3)^2) ↔ b = 24 ∨ b = -24 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l1038_103881


namespace NUMINAMATH_CALUDE_race_finish_times_l1038_103835

/-- Race parameters and runner speeds -/
def race_distance : ℝ := 15
def malcolm_speed : ℝ := 5
def joshua_speed : ℝ := 7
def emily_speed : ℝ := 6

/-- Calculate finish time for a runner given their speed -/
def finish_time (speed : ℝ) : ℝ := race_distance * speed

/-- Calculate time difference between two runners -/
def time_difference (speed1 speed2 : ℝ) : ℝ := finish_time speed1 - finish_time speed2

/-- Theorem stating the time differences for Joshua and Emily relative to Malcolm -/
theorem race_finish_times :
  (time_difference joshua_speed malcolm_speed = 30) ∧
  (time_difference emily_speed malcolm_speed = 15) := by
  sorry

end NUMINAMATH_CALUDE_race_finish_times_l1038_103835


namespace NUMINAMATH_CALUDE_difference_15x_x_squared_l1038_103813

theorem difference_15x_x_squared (x : ℕ) (h : x = 8) : 15 * x - x^2 = 56 := by
  sorry

end NUMINAMATH_CALUDE_difference_15x_x_squared_l1038_103813


namespace NUMINAMATH_CALUDE_p_properties_l1038_103837

/-- The product of digits function -/
def p (n : ℕ+) : ℕ := sorry

/-- Theorem stating the properties of p(n) -/
theorem p_properties (n : ℕ+) : 
  (p n ≤ n) ∧ (10 * p n = n^2 + 4*n - 2005 ↔ n = 45) := by sorry

end NUMINAMATH_CALUDE_p_properties_l1038_103837


namespace NUMINAMATH_CALUDE_garrison_provision_days_l1038_103892

/-- The number of days provisions last for a garrison --/
def provisionDays (initialMen : ℕ) (reinforcementMen : ℕ) (daysBeforeReinforcement : ℕ) (daysAfterReinforcement : ℕ) : ℕ :=
  (initialMen * daysBeforeReinforcement + (initialMen + reinforcementMen) * daysAfterReinforcement) / initialMen

theorem garrison_provision_days :
  provisionDays 2000 2700 15 20 = 62 := by
  sorry

#eval provisionDays 2000 2700 15 20

end NUMINAMATH_CALUDE_garrison_provision_days_l1038_103892


namespace NUMINAMATH_CALUDE_min_value_theorem_l1038_103874

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + b = 2) :
  (2 / a) + (1 / b) ≥ 9 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1038_103874


namespace NUMINAMATH_CALUDE_second_to_last_digit_even_for_valid_numbers_l1038_103800

def ends_in_valid_digit (k : ℕ) : Prop :=
  k % 10 = 1 ∨ k % 10 = 3 ∨ k % 10 = 7 ∨ k % 10 = 9 ∨ k % 10 = 5 ∨ k % 10 = 0

def second_to_last_digit (n : ℕ) : ℕ :=
  (n / 10) % 10

theorem second_to_last_digit_even_for_valid_numbers (k n : ℕ) 
  (h : ends_in_valid_digit k) : 
  Even (second_to_last_digit (k^n)) :=
sorry

end NUMINAMATH_CALUDE_second_to_last_digit_even_for_valid_numbers_l1038_103800


namespace NUMINAMATH_CALUDE_notification_completeness_l1038_103897

/-- Represents a point in the kingdom --/
structure Point where
  x : Real
  y : Real

/-- Represents the kingdom --/
structure Kingdom where
  side_length : Real
  residents : Set Point

/-- Represents the notification process --/
def NotificationProcess (k : Kingdom) (speed : Real) (start_time : Real) (end_time : Real) :=
  ∀ p ∈ k.residents, ∃ t : Real, start_time ≤ t ∧ t ≤ end_time ∧
    ∃ q : Point, q ∈ k.residents ∧ 
    Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2) ≤ speed * (t - start_time)

theorem notification_completeness 
  (k : Kingdom) 
  (h1 : k.side_length = 2) 
  (h2 : ∀ p ∈ k.residents, 0 ≤ p.x ∧ p.x ≤ k.side_length ∧ 0 ≤ p.y ∧ p.y ≤ k.side_length)
  (speed : Real) 
  (h3 : speed = 3) 
  (start_time end_time : Real) 
  (h4 : start_time = 12) 
  (h5 : end_time = 18) :
  NotificationProcess k speed start_time end_time :=
sorry

end NUMINAMATH_CALUDE_notification_completeness_l1038_103897


namespace NUMINAMATH_CALUDE_star_polygon_angle_sum_l1038_103838

/-- Represents a star polygon created from an n-sided convex polygon. -/
structure StarPolygon where
  n : ℕ
  h_n : n ≥ 6

/-- Calculates the sum of internal angles at the intersections of a star polygon. -/
def sum_of_internal_angles (sp : StarPolygon) : ℝ :=
  180 * (sp.n - 4)

/-- Theorem stating that the sum of internal angles at the intersections
    of a star polygon is 180(n-4) degrees. -/
theorem star_polygon_angle_sum (sp : StarPolygon) :
  sum_of_internal_angles sp = 180 * (sp.n - 4) := by
  sorry

end NUMINAMATH_CALUDE_star_polygon_angle_sum_l1038_103838


namespace NUMINAMATH_CALUDE_exp_equals_derivative_l1038_103894

-- Define the exponential function
noncomputable def f (x : ℝ) : ℝ := Real.exp x

-- State the theorem
theorem exp_equals_derivative :
  ∀ x : ℝ, f x = deriv f x :=
by sorry

end NUMINAMATH_CALUDE_exp_equals_derivative_l1038_103894


namespace NUMINAMATH_CALUDE_solve_x_l1038_103888

def symbol_value (a b c d : ℤ) : ℤ := a * d - b * c

theorem solve_x : ∃ x : ℤ, symbol_value (x - 1) 2 3 (-5) = 9 ∧ x = -2 := by sorry

end NUMINAMATH_CALUDE_solve_x_l1038_103888


namespace NUMINAMATH_CALUDE_vector_problem_l1038_103808

theorem vector_problem (x y : ℝ) (hx : x > 0) : 
  let a : ℝ × ℝ × ℝ := (2, 4, x)
  let b : ℝ × ℝ × ℝ := (2, y, 2)
  (2^2 + 4^2 + x^2 = (3*Real.sqrt 5)^2) →
  (2*2 + 4*y + x*2 = 0) →
  x + 2*y = -2 := by
sorry

end NUMINAMATH_CALUDE_vector_problem_l1038_103808


namespace NUMINAMATH_CALUDE_polynomial_equality_l1038_103873

theorem polynomial_equality : 7^4 + 4*(7^3) + 6*(7^2) + 4*7 + 1 = 4096 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l1038_103873


namespace NUMINAMATH_CALUDE_unique_w_value_l1038_103878

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def digit_sum (n : ℕ) : ℕ := sorry

def consecutive_digit_sums_prime (n : ℕ) : Prop := sorry

theorem unique_w_value (w : ℕ) :
  w > 0 →
  digit_sum (10^w - 74) = 440 →
  consecutive_digit_sums_prime (10^w - 74) →
  w = 50 := by sorry

end NUMINAMATH_CALUDE_unique_w_value_l1038_103878


namespace NUMINAMATH_CALUDE_greatest_integer_fraction_inequality_l1038_103801

theorem greatest_integer_fraction_inequality :
  ∀ y : ℤ, (8 : ℚ) / 11 > (y : ℚ) / 17 ↔ y ≤ 12 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_fraction_inequality_l1038_103801


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1038_103828

theorem fixed_point_of_exponential_function 
  (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x + 2) - 2
  f (-2) = -1 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1038_103828


namespace NUMINAMATH_CALUDE_partner_a_share_l1038_103885

/-- Calculates a partner's share of the annual gain in a partnership --/
def calculate_share (x : ℚ) (annual_gain : ℚ) : ℚ :=
  let a_share := 12 * x
  let b_share := 12 * x
  let c_share := 12 * x
  let d_share := 36 * x
  let e_share := 35 * x
  let f_share := 30 * x
  let total_investment := a_share + b_share + c_share + d_share + e_share + f_share
  (a_share / total_investment) * annual_gain

/-- The problem statement --/
theorem partner_a_share :
  ∃ (x : ℚ), calculate_share x 38400 = 3360 := by
  sorry

end NUMINAMATH_CALUDE_partner_a_share_l1038_103885


namespace NUMINAMATH_CALUDE_larger_solution_of_quadratic_l1038_103839

theorem larger_solution_of_quadratic (x : ℝ) :
  x^2 - 13*x + 30 = 0 ∧ x ≠ 3 → x = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_larger_solution_of_quadratic_l1038_103839


namespace NUMINAMATH_CALUDE_halloween_bags_l1038_103845

theorem halloween_bags (total_students : ℕ) (pumpkin_students : ℕ) (pack_size : ℕ) (pack_price : ℕ) (individual_price : ℕ) (total_spent : ℕ) : 
  total_students = 25 →
  pumpkin_students = 14 →
  pack_size = 5 →
  pack_price = 3 →
  individual_price = 1 →
  total_spent = 17 →
  total_students - pumpkin_students = 11 :=
by sorry

end NUMINAMATH_CALUDE_halloween_bags_l1038_103845


namespace NUMINAMATH_CALUDE_cuboid_face_area_l1038_103819

theorem cuboid_face_area (small_face_area : ℝ) 
  (h1 : small_face_area > 0)
  (h2 : ∃ (large_face_area : ℝ), large_face_area = 4 * small_face_area)
  (h3 : 2 * small_face_area + 4 * (4 * small_face_area) = 72) :
  ∃ (large_face_area : ℝ), large_face_area = 16 := by
sorry

end NUMINAMATH_CALUDE_cuboid_face_area_l1038_103819


namespace NUMINAMATH_CALUDE_f_properties_l1038_103889

noncomputable def f (x : ℝ) : ℝ := Real.sin (x + 7 * Real.pi / 4) + Real.cos (x - 3 * Real.pi / 4)

theorem f_properties :
  ∃ (p : ℝ),
    (∀ (x : ℝ), f (x + p) = f x) ∧
    (∀ (q : ℝ), (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
    (p = 2 * Real.pi) ∧
    (∀ (x : ℝ), f x ≤ 2) ∧
    (∃ (x : ℝ), f x = 2) ∧
    (∀ (k : ℤ),
      ∀ (x : ℝ),
        (5 * Real.pi / 4 + 2 * ↑k * Real.pi ≤ x ∧ x ≤ 9 * Real.pi / 4 + 2 * ↑k * Real.pi) →
        (∀ (y : ℝ), x < y → f (-y) < f (-x))) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1038_103889


namespace NUMINAMATH_CALUDE_marys_income_percentage_l1038_103815

theorem marys_income_percentage (juan tim mary : ℝ) 
  (h1 : tim = juan * 0.7)
  (h2 : mary = tim * 1.6) :
  mary = juan * 1.12 := by
sorry

end NUMINAMATH_CALUDE_marys_income_percentage_l1038_103815


namespace NUMINAMATH_CALUDE_smallest_survey_size_l1038_103886

theorem smallest_survey_size : ∀ n : ℕ, 
  n > 0 → 
  (∃ y n_yes n_no : ℕ, 
    n_yes = (76 * n) / 100 ∧ 
    n_no = (24 * n) / 100 ∧ 
    n_yes + n_no = n) → 
  n ≥ 25 := by sorry

end NUMINAMATH_CALUDE_smallest_survey_size_l1038_103886


namespace NUMINAMATH_CALUDE_expansion_sum_coefficients_l1038_103807

-- Define the binomial coefficient function
def binomial_coeff (n k : ℕ) : ℕ := sorry

-- Define the expression
def expansion_sum (x : ℕ) : ℕ :=
  (binomial_coeff x 1 + binomial_coeff x 2 + binomial_coeff x 3 + binomial_coeff x 4) ^ 2

-- Theorem statement
theorem expansion_sum_coefficients :
  ∃ x, expansion_sum x = 225 := by sorry

end NUMINAMATH_CALUDE_expansion_sum_coefficients_l1038_103807


namespace NUMINAMATH_CALUDE_quadratic_inequality_proof_l1038_103858

theorem quadratic_inequality_proof (a : ℝ) 
  (h : ∀ x : ℝ, x^2 - 2*a*x + a > 0) : 
  (0 < a ∧ a < 1) ∧ 
  (∀ x : ℝ, (a^(x^2 - 3) < a^(2*x) ∧ a^(2*x) < 1) ↔ x > 3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_proof_l1038_103858


namespace NUMINAMATH_CALUDE_max_value_even_function_l1038_103852

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem max_value_even_function 
  (f : ℝ → ℝ) 
  (h_even : is_even_function f) 
  (h_max : ∃ x ∈ Set.Icc (-3) (-1), ∀ y ∈ Set.Icc (-3) (-1), f y ≤ f x ∧ f x = 6) :
  ∃ x ∈ Set.Icc 1 3, ∀ y ∈ Set.Icc 1 3, f y ≤ f x ∧ f x = 6 :=
sorry

end NUMINAMATH_CALUDE_max_value_even_function_l1038_103852


namespace NUMINAMATH_CALUDE_glue_per_clipping_l1038_103899

theorem glue_per_clipping 
  (num_friends : ℕ) 
  (clippings_per_friend : ℕ) 
  (total_glue_drops : ℕ) : 
  num_friends = 7 → 
  clippings_per_friend = 3 → 
  total_glue_drops = 126 → 
  total_glue_drops / (num_friends * clippings_per_friend) = 6 := by
  sorry

end NUMINAMATH_CALUDE_glue_per_clipping_l1038_103899


namespace NUMINAMATH_CALUDE_inverse_proposition_is_correct_l1038_103821

/-- The original proposition -/
def original_proposition (n : ℕ) : Prop :=
  n % 10 = 5 → n % 5 = 0

/-- The inverse proposition -/
def inverse_proposition (n : ℕ) : Prop :=
  n % 5 = 0 → n % 10 = 5

/-- Theorem stating that the inverse_proposition is indeed the inverse of the original_proposition -/
theorem inverse_proposition_is_correct :
  inverse_proposition = λ n => ¬(original_proposition n) → ¬(n % 10 = 5) :=
by sorry

end NUMINAMATH_CALUDE_inverse_proposition_is_correct_l1038_103821


namespace NUMINAMATH_CALUDE_negation_equivalence_l1038_103853

theorem negation_equivalence (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + a*x + 1 < 0) ↔ (∀ x : ℝ, x^2 + a*x + 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1038_103853


namespace NUMINAMATH_CALUDE_square_difference_symmetry_l1038_103826

theorem square_difference_symmetry (x y : ℝ) : (x - y)^2 = (y - x)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_symmetry_l1038_103826


namespace NUMINAMATH_CALUDE_length_AB_is_6_l1038_103827

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  -- Right angle at A
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0 ∧
  -- Angle B is 45°
  (C.2 - B.2) / (C.1 - B.1) = 1 ∧
  -- BC = 6√2
  Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 6 * Real.sqrt 2

-- Theorem statement
theorem length_AB_is_6 (A B C : ℝ × ℝ) (h : Triangle A B C) :
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 6 :=
sorry

end NUMINAMATH_CALUDE_length_AB_is_6_l1038_103827


namespace NUMINAMATH_CALUDE_choir_members_count_l1038_103868

theorem choir_members_count :
  ∃! n : ℕ, 150 ≤ n ∧ n ≤ 300 ∧ n % 10 = 6 ∧ n % 11 = 6 ∧ n = 226 := by
  sorry

end NUMINAMATH_CALUDE_choir_members_count_l1038_103868


namespace NUMINAMATH_CALUDE_jo_equals_alex_sum_l1038_103832

def roundToNearestMultipleOf5 (n : ℕ) : ℕ :=
  5 * ((n + 2) / 5)

def joSum (n : ℕ) : ℕ :=
  n * (n + 1) / 2

def alexSum (n : ℕ) : ℕ :=
  (Finset.range n).sum (roundToNearestMultipleOf5 ∘ (· + 1))

theorem jo_equals_alex_sum :
  joSum 100 = alexSum 100 := by
  sorry

end NUMINAMATH_CALUDE_jo_equals_alex_sum_l1038_103832


namespace NUMINAMATH_CALUDE_rational_equality_l1038_103861

theorem rational_equality (n : ℕ) (x y : ℚ) 
  (h_odd : Odd n) 
  (h_pos : 0 < n) 
  (h_eq : x^n + 2*y = y^n + 2*x) : 
  x = y := by sorry

end NUMINAMATH_CALUDE_rational_equality_l1038_103861


namespace NUMINAMATH_CALUDE_function_properties_l1038_103867

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * x - a

theorem function_properties (a : ℝ) (h : a ≠ 0) :
  (f a 0 = 2 → a = -1) ∧
  (a = -1 → 
    (∀ x y : ℝ, x < y → x < 0 → f a x > f a y) ∧
    (∀ x y : ℝ, x < y → 0 < x → f a x < f a y) ∧
    (∀ x : ℝ, f a x ≥ 2) ∧
    (f a 0 = 2)) ∧
  ((∀ x : ℝ, f a x ≠ 0) → -Real.exp 2 < a ∧ a < 0) :=
by sorry

#check function_properties

end NUMINAMATH_CALUDE_function_properties_l1038_103867


namespace NUMINAMATH_CALUDE_monthly_fee_calculation_l1038_103818

/-- Represents the long distance phone service billing structure and usage -/
structure PhoneBill where
  monthlyFee : ℝ
  ratePerMinute : ℝ
  minutesUsed : ℕ
  totalBill : ℝ

/-- Theorem stating that given the specific conditions, the monthly fee is $2.00 -/
theorem monthly_fee_calculation (bill : PhoneBill) 
    (h1 : bill.ratePerMinute = 0.12)
    (h2 : bill.minutesUsed = 178)
    (h3 : bill.totalBill = 23.36) :
    bill.monthlyFee = 2.00 := by
  sorry

end NUMINAMATH_CALUDE_monthly_fee_calculation_l1038_103818


namespace NUMINAMATH_CALUDE_solve_plane_problem_l1038_103891

def plane_problem (distance : ℝ) (time_with_wind : ℝ) (time_against_wind : ℝ) : Prop :=
  ∃ (plane_speed : ℝ) (wind_speed : ℝ),
    (plane_speed + wind_speed) * time_with_wind = distance ∧
    (plane_speed - wind_speed) * time_against_wind = distance ∧
    plane_speed = 262.5

theorem solve_plane_problem :
  plane_problem 900 3 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_plane_problem_l1038_103891


namespace NUMINAMATH_CALUDE_smallest_collection_l1038_103865

def yoongi_collection : ℕ := 4
def yuna_collection : ℕ := 5
def jungkook_collection : ℕ := 6 + 3

theorem smallest_collection : 
  yoongi_collection < yuna_collection ∧ 
  yoongi_collection < jungkook_collection := by
sorry

end NUMINAMATH_CALUDE_smallest_collection_l1038_103865


namespace NUMINAMATH_CALUDE_dining_bill_calculation_l1038_103864

theorem dining_bill_calculation (total : ℝ) (tip_rate : ℝ) (tax_rate : ℝ) 
  (h1 : total = 132)
  (h2 : tip_rate = 0.20)
  (h3 : tax_rate = 0.10) :
  ∃ (original_price : ℝ), 
    original_price * (1 + tax_rate) * (1 + tip_rate) = total ∧ 
    original_price = 100 := by
  sorry

end NUMINAMATH_CALUDE_dining_bill_calculation_l1038_103864


namespace NUMINAMATH_CALUDE_intersection_line_slope_l1038_103831

-- Define the equations of the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y - 8 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 6*y + 9 = 0

-- Define the intersection points
def intersection (C D : ℝ × ℝ) : Prop :=
  circle1 C.1 C.2 ∧ circle1 D.1 D.2 ∧ circle2 C.1 C.2 ∧ circle2 D.1 D.2 ∧ C ≠ D

-- Theorem statement
theorem intersection_line_slope (C D : ℝ × ℝ) (h : intersection C D) :
  (D.2 - C.2) / (D.1 - C.1) = -1 := by sorry

end NUMINAMATH_CALUDE_intersection_line_slope_l1038_103831


namespace NUMINAMATH_CALUDE_diving_survey_contradiction_l1038_103830

structure Survey where
  population : ℕ
  sample : ℕ
  topic : String

def is_sampling_survey (s : Survey) : Prop :=
  s.sample < s.population

theorem diving_survey_contradiction (s : Survey) 
  (h1 : s.population = 2000)
  (h2 : s.sample = 150)
  (h3 : s.topic = "interest in diving")
  (h4 : is_sampling_survey s) : 
  s.sample ≠ 150 := by
  sorry

end NUMINAMATH_CALUDE_diving_survey_contradiction_l1038_103830


namespace NUMINAMATH_CALUDE_systematic_sample_theorem_l1038_103805

/-- Represents a systematic sample from a population --/
structure SystematicSample where
  population_size : Nat
  sample_size : Nat
  first_element : Nat
  interval : Nat

/-- Checks if a number is in the systematic sample --/
def SystematicSample.contains (s : SystematicSample) (n : Nat) : Prop :=
  ∃ k : Nat, k < s.sample_size ∧ n = s.first_element + k * s.interval

/-- The main theorem --/
theorem systematic_sample_theorem (sample : SystematicSample)
    (h_pop : sample.population_size = 56)
    (h_size : sample.sample_size = 4)
    (h_first : sample.first_element = 6)
    (h_contains_34 : sample.contains 34)
    (h_contains_48 : sample.contains 48) :
    sample.contains 20 :=
  sorry

end NUMINAMATH_CALUDE_systematic_sample_theorem_l1038_103805


namespace NUMINAMATH_CALUDE_multiples_17_sums_l1038_103803

/-- The sum of the first n positive integers -/
def sum_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sum of the squares of the first n positive integers -/
def sum_squares_n (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

/-- The sum of the first twelve positive multiples of 17 -/
def sum_multiples_17 : ℕ := 17 * sum_n 12

/-- The sum of the squares of the first twelve positive multiples of 17 -/
def sum_squares_multiples_17 : ℕ := 17^2 * sum_squares_n 12

theorem multiples_17_sums :
  sum_multiples_17 = 1326 ∧ sum_squares_multiples_17 = 187850 := by
  sorry

end NUMINAMATH_CALUDE_multiples_17_sums_l1038_103803


namespace NUMINAMATH_CALUDE_sports_equipment_store_problem_l1038_103810

/-- Sports equipment store problem -/
theorem sports_equipment_store_problem 
  (total_balls : ℕ) 
  (budget : ℕ) 
  (basketball_cost : ℕ) 
  (volleyball_cost : ℕ) 
  (basketball_price_ratio : ℚ) 
  (school_basketball_purchase : ℕ) 
  (school_volleyball_purchase : ℕ) 
  (school_basketball_count : ℕ) 
  (school_volleyball_count : ℕ) :
  total_balls = 200 →
  budget ≤ 5000 →
  basketball_cost = 30 →
  volleyball_cost = 24 →
  basketball_price_ratio = 3/2 →
  school_basketball_purchase = 1800 →
  school_volleyball_purchase = 1500 →
  school_volleyball_count = school_basketball_count + 10 →
  ∃ (basketball_price volleyball_price : ℕ) 
    (optimal_basketball optimal_volleyball : ℕ),
    basketball_price = 45 ∧
    volleyball_price = 30 ∧
    optimal_basketball = 33 ∧
    optimal_volleyball = 167 ∧
    optimal_basketball + optimal_volleyball = total_balls ∧
    optimal_basketball * basketball_cost + optimal_volleyball * volleyball_cost ≤ budget ∧
    ∀ (b v : ℕ), 
      b + v = total_balls →
      b * basketball_cost + v * volleyball_cost ≤ budget →
      (basketball_price - 3 - basketball_cost) * b + (volleyball_price - 2 - volleyball_cost) * v ≤
      (basketball_price - 3 - basketball_cost) * optimal_basketball + 
      (volleyball_price - 2 - volleyball_cost) * optimal_volleyball :=
by sorry

end NUMINAMATH_CALUDE_sports_equipment_store_problem_l1038_103810


namespace NUMINAMATH_CALUDE_trays_from_second_table_l1038_103851

def trays_per_trip : ℕ := 7
def total_trips : ℕ := 4
def trays_from_first_table : ℕ := 23

theorem trays_from_second_table :
  trays_per_trip * total_trips - trays_from_first_table = 5 := by
  sorry

end NUMINAMATH_CALUDE_trays_from_second_table_l1038_103851


namespace NUMINAMATH_CALUDE_basketball_teams_l1038_103875

theorem basketball_teams (n : ℕ) (h : n * (n - 1) / 2 = 28) : n = 8 := by
  sorry

end NUMINAMATH_CALUDE_basketball_teams_l1038_103875


namespace NUMINAMATH_CALUDE_principal_determination_l1038_103898

/-- Given a principal amount and an unknown interest rate, if increasing the
    interest rate by 6 percentage points results in Rs. 30 more interest over 1 year,
    then the principal must be Rs. 500. -/
theorem principal_determination (P R : ℝ) (h : P * (R + 6) / 100 - P * R / 100 = 30) :
  P = 500 := by
  sorry

end NUMINAMATH_CALUDE_principal_determination_l1038_103898


namespace NUMINAMATH_CALUDE_x_intercept_of_line_l1038_103895

/-- The x-intercept of the line 3x + 5y = 20 is (20/3, 0) -/
theorem x_intercept_of_line (x y : ℚ) : 
  3 * x + 5 * y = 20 → y = 0 → x = 20 / 3 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_of_line_l1038_103895


namespace NUMINAMATH_CALUDE_factorization_equality_l1038_103860

theorem factorization_equality (a : ℝ) : a * (a - 2) + 1 = (a - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1038_103860


namespace NUMINAMATH_CALUDE_smaller_number_problem_l1038_103887

theorem smaller_number_problem (x y : ℤ) 
  (h1 : x = 2 * y - 3) 
  (h2 : x + y = 51) : 
  min x y = 18 := by sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l1038_103887


namespace NUMINAMATH_CALUDE_purely_imaginary_implies_a_equals_one_l1038_103841

/-- A complex number z is purely imaginary if its real part is zero and its imaginary part is non-zero. -/
def PurelyImaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- The complex number z defined in terms of a real number a. -/
def z (a : ℝ) : ℂ :=
  ⟨a^2 + 2*a - 3, a + 3⟩

theorem purely_imaginary_implies_a_equals_one :
  ∀ a : ℝ, PurelyImaginary (z a) → a = 1 :=
by sorry

end NUMINAMATH_CALUDE_purely_imaginary_implies_a_equals_one_l1038_103841


namespace NUMINAMATH_CALUDE_remainder_of_binary_number_div_8_l1038_103863

def binary_number : ℕ := 0b100101110011

theorem remainder_of_binary_number_div_8 :
  binary_number % 8 = 3 := by
sorry

end NUMINAMATH_CALUDE_remainder_of_binary_number_div_8_l1038_103863


namespace NUMINAMATH_CALUDE_rational_function_value_l1038_103846

/-- A rational function with specific properties -/
structure RationalFunction where
  p : ℝ → ℝ
  q : ℝ → ℝ
  p_linear : ∃ a b : ℝ, ∀ x, p x = a * x + b
  q_quadratic : ∃ a b c : ℝ, ∀ x, q x = a * x^2 + b * x + c
  asymptote_neg_four : q (-4) = 0
  asymptote_one : q 1 = 0
  passes_origin : p 0 = 0 ∧ q 0 ≠ 0
  passes_two_neg_one : p 2 / q 2 = -1

/-- The main theorem -/
theorem rational_function_value (f : RationalFunction) : f.p 1 / f.q 1 = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_rational_function_value_l1038_103846


namespace NUMINAMATH_CALUDE_georges_required_speed_l1038_103883

/-- George's usual walking distance to school in miles -/
def usual_distance : ℝ := 1.5

/-- George's usual walking speed in miles per hour -/
def usual_speed : ℝ := 4

/-- Distance George walks at a slower pace today in miles -/
def slow_distance : ℝ := 1

/-- George's slower walking speed today in miles per hour -/
def slow_speed : ℝ := 3

/-- Remaining distance George needs to run in miles -/
def remaining_distance : ℝ := 0.5

/-- Theorem stating the speed George needs to run to arrive on time -/
theorem georges_required_speed : 
  ∃ (required_speed : ℝ),
    (usual_distance / usual_speed = slow_distance / slow_speed + remaining_distance / required_speed) ∧
    required_speed = 12 := by
  sorry

end NUMINAMATH_CALUDE_georges_required_speed_l1038_103883


namespace NUMINAMATH_CALUDE_circle_intersection_area_l1038_103880

noncomputable def circleIntersection (r : ℝ) (bd ed : ℝ) : ℝ :=
  let ad := 2 * r + bd
  let ea := Real.sqrt (ad^2 + ed^2)
  let ec := ed^2 / ea
  let ac := ea - ec
  let bc := Real.sqrt ((2*r)^2 - ac^2)
  1/2 * bc * ac

theorem circle_intersection_area (r bd ed : ℝ) (hr : r = 4) (hbd : bd = 6) (hed : ed = 5) :
  circleIntersection r bd ed = 11627.6 / 221 :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_area_l1038_103880


namespace NUMINAMATH_CALUDE_total_non_hot_peppers_l1038_103882

/-- Represents the days of the week -/
inductive Day
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents the pepper subtypes -/
inductive PepperSubtype
  | Jalapeno
  | Habanero
  | Bell
  | Banana
  | Poblano
  | Anaheim

/-- Returns the number of peppers picked for a given subtype on a given day -/
def peppers_picked (day : Day) (subtype : PepperSubtype) : Nat :=
  match day, subtype with
  | Day.Sunday,    PepperSubtype.Jalapeno  => 3
  | Day.Sunday,    PepperSubtype.Habanero  => 4
  | Day.Sunday,    PepperSubtype.Bell      => 6
  | Day.Sunday,    PepperSubtype.Banana    => 4
  | Day.Sunday,    PepperSubtype.Poblano   => 7
  | Day.Sunday,    PepperSubtype.Anaheim   => 6
  | Day.Monday,    PepperSubtype.Jalapeno  => 6
  | Day.Monday,    PepperSubtype.Habanero  => 6
  | Day.Monday,    PepperSubtype.Bell      => 4
  | Day.Monday,    PepperSubtype.Banana    => 4
  | Day.Monday,    PepperSubtype.Poblano   => 5
  | Day.Monday,    PepperSubtype.Anaheim   => 5
  | Day.Tuesday,   PepperSubtype.Jalapeno  => 7
  | Day.Tuesday,   PepperSubtype.Habanero  => 7
  | Day.Tuesday,   PepperSubtype.Bell      => 10
  | Day.Tuesday,   PepperSubtype.Banana    => 9
  | Day.Tuesday,   PepperSubtype.Poblano   => 4
  | Day.Tuesday,   PepperSubtype.Anaheim   => 3
  | Day.Wednesday, PepperSubtype.Jalapeno  => 6
  | Day.Wednesday, PepperSubtype.Habanero  => 6
  | Day.Wednesday, PepperSubtype.Bell      => 3
  | Day.Wednesday, PepperSubtype.Banana    => 2
  | Day.Wednesday, PepperSubtype.Poblano   => 12
  | Day.Wednesday, PepperSubtype.Anaheim   => 11
  | Day.Thursday,  PepperSubtype.Jalapeno  => 3
  | Day.Thursday,  PepperSubtype.Habanero  => 2
  | Day.Thursday,  PepperSubtype.Bell      => 10
  | Day.Thursday,  PepperSubtype.Banana    => 10
  | Day.Thursday,  PepperSubtype.Poblano   => 3
  | Day.Thursday,  PepperSubtype.Anaheim   => 2
  | Day.Friday,    PepperSubtype.Jalapeno  => 9
  | Day.Friday,    PepperSubtype.Habanero  => 9
  | Day.Friday,    PepperSubtype.Bell      => 8
  | Day.Friday,    PepperSubtype.Banana    => 7
  | Day.Friday,    PepperSubtype.Poblano   => 6
  | Day.Friday,    PepperSubtype.Anaheim   => 6
  | Day.Saturday,  PepperSubtype.Jalapeno  => 6
  | Day.Saturday,  PepperSubtype.Habanero  => 6
  | Day.Saturday,  PepperSubtype.Bell      => 4
  | Day.Saturday,  PepperSubtype.Banana    => 4
  | Day.Saturday,  PepperSubtype.Poblano   => 15
  | Day.Saturday,  PepperSubtype.Anaheim   => 15

/-- Returns true if the pepper subtype is non-hot (sweet or mild) -/
def is_non_hot (subtype : PepperSubtype) : Bool :=
  match subtype with
  | PepperSubtype.Bell    => true
  | PepperSubtype.Banana  => true
  | PepperSubtype.Poblano => true
  | PepperSubtype.Anaheim => true
  | _                     => false

/-- Theorem: The total number of non-hot peppers picked throughout the week is 185 -/
theorem total_non_hot_peppers :
  (List.sum (List.map
    (fun day =>
      List.sum (List.map
        (fun subtype =>
          if is_non_hot subtype then peppers_picked day subtype else 0)
        [PepperSubtype.Jalapeno, PepperSubtype.Habanero, PepperSubtype.Bell,
         PepperSubtype.Banana, PepperSubtype.Poblano, PepperSubtype.Anaheim]))
    [Day.Sunday, Day.Monday, Day.Tuesday, Day.Wednesday, Day.Thursday, Day.Friday, Day.Saturday]))
  = 185 := by
  sorry

end NUMINAMATH_CALUDE_total_non_hot_peppers_l1038_103882


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l1038_103829

/-- The standard equation of a hyperbola with foci on the y-axis, given a + c = 9 and b = 3 -/
theorem hyperbola_standard_equation (a c b : ℝ) (h1 : a + c = 9) (h2 : b = 3) :
  ∃ (x y : ℝ), y^2 / 16 - x^2 / 9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l1038_103829


namespace NUMINAMATH_CALUDE_two_tangent_lines_l1038_103848

/-- The cubic function f(x) = -x³ + 6x² - 9x + 8 -/
def f (x : ℝ) : ℝ := -x^3 + 6*x^2 - 9*x + 8

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := -3*x^2 + 12*x - 9

/-- Theorem: There are exactly two tangent lines from (0, 0) to the graph of f(x) -/
theorem two_tangent_lines :
  ∃! (s : Finset ℝ), s.card = 2 ∧
    ∀ x₀ ∈ s, f x₀ + f' x₀ * (-x₀) = 0 ∧
    ∀ x ∉ s, f x + f' x * (-x) ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_two_tangent_lines_l1038_103848


namespace NUMINAMATH_CALUDE_total_cost_calculation_l1038_103859

def silverware_cost : ℝ := 20
def plate_cost_ratio : ℝ := 0.5

theorem total_cost_calculation :
  let plate_cost := plate_cost_ratio * silverware_cost
  silverware_cost + plate_cost = 30 :=
by sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l1038_103859


namespace NUMINAMATH_CALUDE_equation_solution_l1038_103824

theorem equation_solution (y : ℝ) : 
  ∃ z : ℝ, 19 * (1 + y) + z = 19 * (-1 + y) - 21 ∧ z = -59 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1038_103824


namespace NUMINAMATH_CALUDE_typists_pages_time_relation_l1038_103836

/-- Given that 10 typists can type 25 pages in 5 minutes, 
    prove that 2 typists can type 2 pages in 2 minutes. -/
theorem typists_pages_time_relation : 
  ∀ (n : ℕ), 
    (10 : ℝ) * (25 : ℝ) / (5 : ℝ) = n * (2 : ℝ) / (2 : ℝ) → 
    n = 2 :=
by sorry

end NUMINAMATH_CALUDE_typists_pages_time_relation_l1038_103836


namespace NUMINAMATH_CALUDE_nonzero_digits_after_decimal_l1038_103866

theorem nonzero_digits_after_decimal (n : ℕ) (d : ℕ) (h : n = 60 ∧ d = 2^3 * 5^8) :
  (Nat.digits 10 (n * 10^7 / d)).length - 7 = 3 :=
sorry

end NUMINAMATH_CALUDE_nonzero_digits_after_decimal_l1038_103866


namespace NUMINAMATH_CALUDE_solution_equality_l1038_103884

-- Define the function F
def F (a b c : ℝ) : ℝ := a * b^3 + c

-- Theorem statement
theorem solution_equality :
  ∃ a : ℝ, F a 2 3 = F a 3 4 ∧ a = -1/19 := by
  sorry

end NUMINAMATH_CALUDE_solution_equality_l1038_103884


namespace NUMINAMATH_CALUDE_exists_number_with_large_square_digit_sum_l1038_103847

/-- Sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem stating the existence of a number whose square's digit sum exceeds 1000 times its own digit sum -/
theorem exists_number_with_large_square_digit_sum :
  ∃ n : ℕ, sumOfDigits (n^2) > 1000 * sumOfDigits n := by
  sorry

end NUMINAMATH_CALUDE_exists_number_with_large_square_digit_sum_l1038_103847


namespace NUMINAMATH_CALUDE_remainder_451951_div_5_l1038_103869

theorem remainder_451951_div_5 : 451951 % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_451951_div_5_l1038_103869


namespace NUMINAMATH_CALUDE_simplify_expression_l1038_103817

theorem simplify_expression (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hab : a ≥ b) :
  (a - b) / (Real.sqrt a + Real.sqrt b) + (a * Real.sqrt a + b * Real.sqrt b) / (a - Real.sqrt (a * b) + b) = 2 * Real.sqrt a :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l1038_103817


namespace NUMINAMATH_CALUDE_mariela_get_well_cards_l1038_103823

theorem mariela_get_well_cards (cards_in_hospital : ℕ) (cards_at_home : ℕ) 
  (h1 : cards_in_hospital = 403)
  (h2 : cards_at_home = 287) :
  cards_in_hospital + cards_at_home = 690 := by
  sorry

end NUMINAMATH_CALUDE_mariela_get_well_cards_l1038_103823


namespace NUMINAMATH_CALUDE_star_operation_proof_l1038_103862

-- Define the ※ operation
def star (a b : ℕ) : ℚ :=
  (b : ℚ) / 2 * (2 * (a : ℚ) / 10 + ((b : ℚ) - 1) / 10)

-- State the theorem
theorem star_operation_proof (a : ℕ) :
  star 1 2 = (3 : ℚ) / 10 ∧
  star 2 3 = (9 : ℚ) / 10 ∧
  star 5 4 = (26 : ℚ) / 10 ∧
  star a 15 = (165 : ℚ) / 10 →
  a = 4 := by
  sorry

end NUMINAMATH_CALUDE_star_operation_proof_l1038_103862


namespace NUMINAMATH_CALUDE_basketball_substitutions_remainder_l1038_103820

/-- The number of ways to make exactly k substitutions in a basketball game -/
def num_substitutions (k : ℕ) : ℕ :=
  match k with
  | 0 => 1
  | k + 1 => 12 * (13 - k) * num_substitutions k

/-- The total number of ways to make substitutions in the basketball game -/
def total_substitutions : ℕ :=
  num_substitutions 0 + num_substitutions 1 + num_substitutions 2 + 
  num_substitutions 3 + num_substitutions 4

theorem basketball_substitutions_remainder :
  total_substitutions % 1000 = 953 := by
  sorry

end NUMINAMATH_CALUDE_basketball_substitutions_remainder_l1038_103820


namespace NUMINAMATH_CALUDE_two_machines_total_copies_l1038_103877

/-- Represents a copy machine with a constant copying rate -/
structure CopyMachine where
  rate : ℕ  -- copies per minute

/-- Calculates the number of copies made by a machine in a given time -/
def copies_made (machine : CopyMachine) (minutes : ℕ) : ℕ :=
  machine.rate * minutes

/-- Represents the problem setup with two copy machines -/
structure TwoMachinesProblem where
  machine1 : CopyMachine
  machine2 : CopyMachine
  time : ℕ  -- in minutes

/-- The main theorem to be proved -/
theorem two_machines_total_copies 
  (problem : TwoMachinesProblem) 
  (h1 : problem.machine1.rate = 25)
  (h2 : problem.machine2.rate = 55)
  (h3 : problem.time = 30) : 
  copies_made problem.machine1 problem.time + copies_made problem.machine2 problem.time = 2400 :=
by sorry

end NUMINAMATH_CALUDE_two_machines_total_copies_l1038_103877


namespace NUMINAMATH_CALUDE_inequality_proof_l1038_103890

theorem inequality_proof (x y : ℤ) : x * (x + 1) ≠ 2 * (5 * y + 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1038_103890


namespace NUMINAMATH_CALUDE_trips_per_month_l1038_103842

/-- Given a person who spends 72 hours driving in a year, with each round trip
    taking 3 hours, prove that the number of trips per month is 2. -/
theorem trips_per_month (hours_per_year : ℕ) (hours_per_trip : ℕ) 
    (months_per_year : ℕ) : ℕ :=
  by
  have h1 : hours_per_year = 72 := by sorry
  have h2 : hours_per_trip = 3 := by sorry
  have h3 : months_per_year = 12 := by sorry
  
  let trips_per_year : ℕ := hours_per_year / hours_per_trip
  
  have h4 : trips_per_year = 24 := by sorry
  
  exact trips_per_year / months_per_year

end NUMINAMATH_CALUDE_trips_per_month_l1038_103842


namespace NUMINAMATH_CALUDE_equation_solutions_l1038_103843

theorem equation_solutions :
  (∃ x : ℝ, 3 * x * (x - 1) = 2 - 2 * x) ∧
  (∃ x : ℝ, 3 * x^2 - 6 * x + 2 = 0) ∧
  (∀ x : ℝ, 3 * x * (x - 1) = 2 - 2 * x ↔ (x = 1 ∨ x = -2/3)) ∧
  (∀ x : ℝ, 3 * x^2 - 6 * x + 2 = 0 ↔ (x = 1 + Real.sqrt 3 / 3 ∨ x = 1 - Real.sqrt 3 / 3)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1038_103843


namespace NUMINAMATH_CALUDE_boat_speed_l1038_103822

/-- The average speed of a boat in still water, given its travel times with and against a current. -/
theorem boat_speed (time_with_current time_against_current current_speed : ℝ) 
  (h1 : time_with_current = 2)
  (h2 : time_against_current = 2.5)
  (h3 : current_speed = 3)
  (h4 : time_with_current * (x + current_speed) = time_against_current * (x - current_speed)) : 
  x = 27 :=
by
  sorry

#check boat_speed

end NUMINAMATH_CALUDE_boat_speed_l1038_103822


namespace NUMINAMATH_CALUDE_mason_tables_theorem_l1038_103833

/-- The number of tables Mason needs settings for -/
def num_tables : ℕ :=
  let silverware_weight : ℕ := 4  -- weight of one piece of silverware in ounces
  let silverware_per_setting : ℕ := 3  -- number of silverware pieces per setting
  let plate_weight : ℕ := 12  -- weight of one plate in ounces
  let plates_per_setting : ℕ := 2  -- number of plates per setting
  let settings_per_table : ℕ := 8  -- number of settings per table
  let backup_settings : ℕ := 20  -- number of backup settings
  let total_weight : ℕ := 5040  -- total weight of all settings in ounces

  -- Calculate the result
  (total_weight / (silverware_weight * silverware_per_setting + plate_weight * plates_per_setting) - backup_settings) / settings_per_table

theorem mason_tables_theorem : num_tables = 15 := by
  sorry

end NUMINAMATH_CALUDE_mason_tables_theorem_l1038_103833


namespace NUMINAMATH_CALUDE_min_coins_for_distribution_l1038_103896

/-- The minimum number of additional coins needed for distribution -/
def min_additional_coins (num_friends : ℕ) (initial_coins : ℕ) : ℕ :=
  (num_friends * (num_friends + 1)) / 2 - initial_coins

/-- Theorem stating the minimum number of additional coins needed -/
theorem min_coins_for_distribution (num_friends : ℕ) (initial_coins : ℕ) 
  (h1 : num_friends = 15) (h2 : initial_coins = 63) :
  min_additional_coins num_friends initial_coins = 57 := by
  sorry

#eval min_additional_coins 15 63

end NUMINAMATH_CALUDE_min_coins_for_distribution_l1038_103896


namespace NUMINAMATH_CALUDE_partnership_gain_l1038_103812

/-- Represents the investment and profit share of a partner in the partnership. -/
structure Partner where
  investment : ℕ  -- Amount invested
  duration : ℕ    -- Duration of investment in months
  share : ℕ       -- Share of profit

/-- Represents the partnership with three partners. -/
structure Partnership where
  a : Partner
  b : Partner
  c : Partner

/-- Calculates the total annual gain of the partnership. -/
def totalAnnualGain (p : Partnership) : ℕ :=
  p.a.share + p.b.share + p.c.share

/-- Theorem stating the total annual gain of the partnership. -/
theorem partnership_gain (p : Partnership) 
  (h1 : p.a.investment > 0)
  (h2 : p.b.investment = 2 * p.a.investment)
  (h3 : p.c.investment = 3 * p.a.investment)
  (h4 : p.a.duration = 12)
  (h5 : p.b.duration = 6)
  (h6 : p.c.duration = 4)
  (h7 : p.a.share = 6100)
  (h8 : p.a.share = p.b.share)
  (h9 : p.b.share = p.c.share) :
  totalAnnualGain p = 18300 := by
  sorry

end NUMINAMATH_CALUDE_partnership_gain_l1038_103812


namespace NUMINAMATH_CALUDE_relay_race_tables_l1038_103825

/-- The number of tables required for a relay race with given conditions -/
def num_tables (race_distance : ℕ) (distance_between_1_and_3 : ℕ) : ℕ :=
  (race_distance / (distance_between_1_and_3 / 2)) + 1

theorem relay_race_tables :
  num_tables 1200 400 = 7 :=
by sorry

end NUMINAMATH_CALUDE_relay_race_tables_l1038_103825


namespace NUMINAMATH_CALUDE_parallel_planes_from_perpendicular_lines_l1038_103856

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (planes_parallel : Plane → Plane → Prop)
variable (non_intersecting_lines : Line → Line → Prop)
variable (non_intersecting_planes : Plane → Plane → Prop)

-- State the theorem
theorem parallel_planes_from_perpendicular_lines 
  (m n : Line) (α β : Plane) : 
  non_intersecting_lines m n →
  non_intersecting_planes α β →
  parallel m n →
  perpendicular m α →
  perpendicular n β →
  planes_parallel α β :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_from_perpendicular_lines_l1038_103856


namespace NUMINAMATH_CALUDE_only_newborn_babies_is_set_l1038_103840

-- Define a type for statements
inductive Statement
| NewbornBabies
| VerySmallNumbers
| HealthyStudents
| CutePandas

-- Define a function to check if a statement satisfies definiteness
def satisfiesDefiniteness (s : Statement) : Prop :=
  match s with
  | Statement.NewbornBabies => true
  | _ => false

-- Theorem: Only NewbornBabies satisfies definiteness
theorem only_newborn_babies_is_set :
  ∀ s : Statement, satisfiesDefiniteness s ↔ s = Statement.NewbornBabies :=
by
  sorry


end NUMINAMATH_CALUDE_only_newborn_babies_is_set_l1038_103840


namespace NUMINAMATH_CALUDE_power_calculation_l1038_103834

theorem power_calculation : 16^4 * 8^2 / 4^10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l1038_103834


namespace NUMINAMATH_CALUDE_reservoir_percentage_before_storm_l1038_103870

-- Define the reservoir capacity in billion gallons
def reservoir_capacity : ℝ := 550

-- Define the original contents in billion gallons
def original_contents : ℝ := 220

-- Define the amount of water added by the storm in billion gallons
def storm_water : ℝ := 110

-- Define the percentage full after the storm
def post_storm_percentage : ℝ := 0.60

-- Theorem to prove
theorem reservoir_percentage_before_storm :
  (original_contents / reservoir_capacity) * 100 = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_reservoir_percentage_before_storm_l1038_103870


namespace NUMINAMATH_CALUDE_shirt_price_calculation_l1038_103857

theorem shirt_price_calculation (total_cost sweater_price shirt_price : ℝ) :
  total_cost = 80.34 →
  shirt_price = sweater_price - 7.43 →
  total_cost = sweater_price + shirt_price →
  shirt_price = 36.455 := by
sorry

end NUMINAMATH_CALUDE_shirt_price_calculation_l1038_103857


namespace NUMINAMATH_CALUDE_arrangements_remainder_l1038_103809

/-- The number of green marbles -/
def green_marbles : ℕ := 8

/-- The maximum number of red marbles that satisfies the equal neighbor condition -/
def max_red_marbles : ℕ := 23

/-- The number of possible arrangements -/
def num_arrangements : ℕ := 490314

/-- The theorem stating the remainder when the number of arrangements is divided by 1000 -/
theorem arrangements_remainder :
  num_arrangements % 1000 = 314 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_remainder_l1038_103809


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1038_103804

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- Define set A
def A : Set ℝ := {x | x^2 - (floor x : ℝ) = 2}

-- Define set B
def B : Set ℝ := {x | -2 < x ∧ x < 2}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {-1, Real.sqrt 3} :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1038_103804


namespace NUMINAMATH_CALUDE_min_questions_to_find_z_l1038_103854

/-- Represents a person in the company -/
structure Person where
  id : Nat

/-- Represents the company with n people -/
structure Company where
  n : Nat
  people : Finset Person
  z : Person
  knows : Person → Person → Prop

/-- Axioms for the company structure -/
axiom company_size (c : Company) : c.people.card = c.n

axiom z_knows_all (c : Company) (p : Person) : 
  p ∈ c.people → p ≠ c.z → c.knows c.z p

axiom z_known_by_none (c : Company) (p : Person) : 
  p ∈ c.people → p ≠ c.z → ¬(c.knows p c.z)

/-- The main theorem to prove -/
theorem min_questions_to_find_z (c : Company) :
  ∃ (strategy : Nat → Person × Person),
    (∀ k, k < c.n - 1 → 
      (strategy k).1 ∈ c.people ∧ (strategy k).2 ∈ c.people) ∧
    (∀ result : Nat → Bool,
      ∃! p, p ∈ c.people ∧ 
        ∀ k, k < c.n - 1 → 
          result k = c.knows (strategy k).1 (strategy k).2 →
          p ≠ (strategy k).1 ∧ p ≠ (strategy k).2) ∧
  ¬∃ (strategy : Nat → Person × Person),
    (∀ k, k < c.n - 2 → 
      (strategy k).1 ∈ c.people ∧ (strategy k).2 ∈ c.people) ∧
    (∀ result : Nat → Bool,
      ∃! p, p ∈ c.people ∧ 
        ∀ k, k < c.n - 2 → 
          result k = c.knows (strategy k).1 (strategy k).2 →
          p ≠ (strategy k).1 ∧ p ≠ (strategy k).2) :=
by
  sorry

end NUMINAMATH_CALUDE_min_questions_to_find_z_l1038_103854


namespace NUMINAMATH_CALUDE_product_of_square_roots_l1038_103849

theorem product_of_square_roots (x : ℝ) (h : x > 0) :
  Real.sqrt (50 * x) * Real.sqrt (18 * x) * Real.sqrt (8 * x) = 60 * x * Real.sqrt x := by
  sorry

end NUMINAMATH_CALUDE_product_of_square_roots_l1038_103849


namespace NUMINAMATH_CALUDE_expression_one_equals_negative_one_expression_two_equals_five_l1038_103876

-- Expression 1
theorem expression_one_equals_negative_one :
  (9/4)^(1/2) - (-8.6)^0 - (8/27)^(-1/3) = -1 := by sorry

-- Expression 2
theorem expression_two_equals_five :
  Real.log 25 / Real.log 10 + Real.log 4 / Real.log 10 + 7^(Real.log 2 / Real.log 7) + 2 * (Real.log 3 / (2 * Real.log 3)) = 5 := by sorry

end NUMINAMATH_CALUDE_expression_one_equals_negative_one_expression_two_equals_five_l1038_103876


namespace NUMINAMATH_CALUDE_range_of_m_l1038_103850

def p (a : ℝ) : Prop := ∀ x : ℝ, 4 * x^2 - 2 * a * x + 2 * a + 5 = 0 → x ∈ ({x | 4 * x^2 - 2 * a * x + 2 * a + 5 = 0} : Set ℝ)

def q (m : ℝ) : Prop := ∀ x : ℝ, 1 - m ≤ x ∧ x ≤ 1 + m

theorem range_of_m :
  (∀ a : ℝ, (¬(p a) → ∃ m : ℝ, m > 0 ∧ ¬(q m)) ∧
   (∃ m : ℝ, m > 0 ∧ ¬(q m) ∧ p a)) →
  {m : ℝ | m ≥ 9} = {m : ℝ | m > 0 ∧ (∀ a : ℝ, p a → q m)} :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1038_103850


namespace NUMINAMATH_CALUDE_prime_fraction_equation_l1038_103814

theorem prime_fraction_equation (p q r : ℕ+) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  (1 : ℚ) / (p + 1) + (1 : ℚ) / (q + 1) - 1 / ((p + 1) * (q + 1)) = 1 / r →
  ((p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2)) ∧ r = 2 := by
  sorry

end NUMINAMATH_CALUDE_prime_fraction_equation_l1038_103814


namespace NUMINAMATH_CALUDE_investment_partnership_profit_share_l1038_103811

/-- Investment partnership problem -/
theorem investment_partnership_profit_share
  (investment_B : ℝ)
  (investment_A : ℝ)
  (investment_C : ℝ)
  (investment_D : ℝ)
  (time_A : ℝ)
  (time_B : ℝ)
  (time_C : ℝ)
  (time_D : ℝ)
  (total_profit : ℝ)
  (h1 : investment_A = 3 * investment_B)
  (h2 : investment_B = 2 / 3 * investment_C)
  (h3 : investment_D = 1 / 2 * (investment_A + investment_B + investment_C))
  (h4 : time_A = 6)
  (h5 : time_B = 9)
  (h6 : time_C = 12)
  (h7 : time_D = 4)
  (h8 : total_profit = 22000) :
  (investment_B * time_B) / (investment_A * time_A + investment_B * time_B + investment_C * time_C + investment_D * time_D) * total_profit = 3666.67 := by
  sorry

end NUMINAMATH_CALUDE_investment_partnership_profit_share_l1038_103811


namespace NUMINAMATH_CALUDE_smallest_positive_k_l1038_103844

theorem smallest_positive_k (m n : ℕ+) (h : m ≤ 2000) : 
  let k := 3 - (m : ℚ) / n
  ∀ k' > 0, k ≥ k' → k' ≥ 1/667 :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_k_l1038_103844


namespace NUMINAMATH_CALUDE_unique_square_divisible_by_three_in_range_l1038_103802

theorem unique_square_divisible_by_three_in_range : ∃! y : ℕ,
  50 < y ∧ y < 120 ∧ ∃ x : ℕ, y = x^2 ∧ y % 3 = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_square_divisible_by_three_in_range_l1038_103802


namespace NUMINAMATH_CALUDE_wire_cutting_l1038_103872

theorem wire_cutting (total_length : ℝ) (ratio : ℝ) (shorter_piece : ℝ) : 
  total_length = 70 →
  ratio = 2 / 3 →
  shorter_piece + ratio * shorter_piece = total_length →
  shorter_piece = 42 := by
sorry

end NUMINAMATH_CALUDE_wire_cutting_l1038_103872


namespace NUMINAMATH_CALUDE_power_of_power_of_three_l1038_103816

theorem power_of_power_of_three : (3 : ℕ) ^ ((3 : ℕ) ^ (3 : ℕ)) = (3 : ℕ) ^ (27 : ℕ) := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_of_three_l1038_103816


namespace NUMINAMATH_CALUDE_halloween_candy_distribution_l1038_103855

theorem halloween_candy_distribution (initial_candy : ℕ) (eaten_candy : ℕ) (num_piles : ℕ) 
  (h1 : initial_candy = 78)
  (h2 : eaten_candy = 30)
  (h3 : num_piles = 6)
  : (initial_candy - eaten_candy) / num_piles = 8 := by
  sorry

end NUMINAMATH_CALUDE_halloween_candy_distribution_l1038_103855


namespace NUMINAMATH_CALUDE_third_number_in_ratio_l1038_103893

theorem third_number_in_ratio (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  (a : ℚ) / 5 = (b : ℚ) / 6 ∧ (b : ℚ) / 6 = (c : ℚ) / 8 →
  a + c = b + 49 →
  b = 42 := by sorry

end NUMINAMATH_CALUDE_third_number_in_ratio_l1038_103893


namespace NUMINAMATH_CALUDE_real_part_of_z_l1038_103879

theorem real_part_of_z (z : ℂ) (h : Complex.I * z = 1 + 2 * Complex.I) : 
  Complex.re z = 2 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_z_l1038_103879


namespace NUMINAMATH_CALUDE_circles_externally_tangent_l1038_103871

/-- Two circles are externally tangent if the distance between their centers
    equals the sum of their radii -/
def externally_tangent (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  Real.sqrt ((c1.1 - c2.1)^2 + (c1.2 - c2.2)^2) = r1 + r2

theorem circles_externally_tangent :
  let c1 : ℝ × ℝ := (0, 0)
  let c2 : ℝ × ℝ := (3, 4)
  let r1 : ℝ := 2
  let r2 : ℝ := 3
  externally_tangent c1 c2 r1 r2 := by
  sorry

#check circles_externally_tangent

end NUMINAMATH_CALUDE_circles_externally_tangent_l1038_103871
