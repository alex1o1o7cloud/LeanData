import Mathlib

namespace NUMINAMATH_CALUDE_expression_equals_sum_l105_10576

theorem expression_equals_sum (a b c : ℚ) (ha : a = 7) (hb : b = 11) (hc : c = 13) :
  let numerator := a^3 * (1/b - 1/c) + b^3 * (1/c - 1/a) + c^3 * (1/a - 1/b)
  let denominator := a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)
  numerator / denominator = a + b + c :=
by sorry

end NUMINAMATH_CALUDE_expression_equals_sum_l105_10576


namespace NUMINAMATH_CALUDE_two_numbers_difference_l105_10570

theorem two_numbers_difference (x y : ℝ) 
  (sum_eq : x + y = 10) 
  (squares_diff_eq : x^2 - y^2 = 20) : 
  |x - y| = 2 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l105_10570


namespace NUMINAMATH_CALUDE_polynomial_division_l105_10594

theorem polynomial_division (x : ℝ) :
  8 * x^3 - 2 * x^2 + 4 * x - 9 = (x - 3) * (8 * x^2 + 22 * x + 70) + 201 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_l105_10594


namespace NUMINAMATH_CALUDE_tickets_to_buy_l105_10505

def ferris_wheel_cost : ℝ := 2.0
def roller_coaster_cost : ℝ := 7.0
def multiple_ride_discount : ℝ := 1.0
def newspaper_coupon : ℝ := 1.0

theorem tickets_to_buy :
  ferris_wheel_cost + roller_coaster_cost - multiple_ride_discount - newspaper_coupon = 7.0 := by
  sorry

end NUMINAMATH_CALUDE_tickets_to_buy_l105_10505


namespace NUMINAMATH_CALUDE_outbound_speed_calculation_l105_10512

theorem outbound_speed_calculation (distance : ℝ) (return_speed : ℝ) (total_time : ℝ) :
  distance = 19.999999999999996 →
  return_speed = 4 →
  total_time = 5.8 →
  ∃ outbound_speed : ℝ, 
    outbound_speed = 25 ∧
    distance / outbound_speed + distance / return_speed = total_time :=
by sorry

end NUMINAMATH_CALUDE_outbound_speed_calculation_l105_10512


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l105_10558

/-- An isosceles triangle with side lengths 1 and 2 has perimeter 5 -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a > 0 ∧ b > 0 ∧ c > 0 →
  (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 1) →
  (a = b ∨ b = c ∨ a = c) →
  a + b + c = 5 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l105_10558


namespace NUMINAMATH_CALUDE_light_ray_equation_l105_10504

/-- A light ray is emitted from point A(-3, 3), hits the x-axis, gets reflected, and is tangent to a circle. This theorem proves that the equation of the line on which the light ray lies is either 3x + 4y - 3 = 0 or 4x + 3y + 3 = 0. -/
theorem light_ray_equation (x y : ℝ) : 
  let A : ℝ × ℝ := (-3, 3)
  let circle (x y : ℝ) := x^2 + y^2 - 4*x - 4*y + 7 = 0
  let ray_hits_x_axis : Prop := ∃ (t : ℝ), t * (A.1 + 3) = -3 ∧ t * (A.2 - 3) = 0
  let is_tangent_to_circle : Prop := ∃ (x₀ y₀ : ℝ), circle x₀ y₀ ∧ 
    ((x - x₀) * (2*x₀ - 4) + (y - y₀) * (2*y₀ - 4) = 0)
  ray_hits_x_axis → is_tangent_to_circle → 
    (3*x + 4*y - 3 = 0) ∨ (4*x + 3*y + 3 = 0) :=
by sorry


end NUMINAMATH_CALUDE_light_ray_equation_l105_10504


namespace NUMINAMATH_CALUDE_steve_email_percentage_l105_10587

/-- Given Steve's email management scenario, prove that the percentage of emails
    moved to the work folder out of the remaining emails after trashing is 40%. -/
theorem steve_email_percentage :
  ∀ (initial_emails : ℕ) (emails_left : ℕ),
    initial_emails = 400 →
    emails_left = 120 →
    let emails_after_trash : ℕ := initial_emails / 2
    let emails_to_work : ℕ := emails_after_trash - emails_left
    (emails_to_work : ℚ) / emails_after_trash * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_steve_email_percentage_l105_10587


namespace NUMINAMATH_CALUDE_function_characterization_l105_10511

theorem function_characterization (a : ℝ) (ha : a > 0) :
  ∀ (f : ℕ → ℝ),
    (∀ (k m : ℕ), k > 0 ∧ m > 0 ∧ a * m ≤ k ∧ k < (a + 1) * m → f (k + m) = f k + f m) ↔
    ∃ (b : ℝ), ∀ (n : ℕ), f n = b * n :=
by sorry

end NUMINAMATH_CALUDE_function_characterization_l105_10511


namespace NUMINAMATH_CALUDE_ellipse_foci_ratio_l105_10542

/-- The equation of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

/-- The foci of the ellipse -/
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

/-- A point on the ellipse -/
def P : ℝ × ℝ := sorry

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Right-angled triangle formed by P, F₁, and F₂ -/
def is_right_triangle (P F₁ F₂ : ℝ × ℝ) : Prop := sorry

theorem ellipse_foci_ratio :
  is_on_ellipse P.1 P.2 →
  is_right_triangle P F₁ F₂ →
  distance P F₁ > distance P F₂ →
  (distance P F₁ / distance P F₂ = 7/2) ∨ (distance P F₁ / distance P F₂ = 2) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_foci_ratio_l105_10542


namespace NUMINAMATH_CALUDE_bicycle_problem_l105_10584

theorem bicycle_problem (total_distance : ℝ) (walking_speed : ℝ) (cycling_speed : ℝ) 
  (h1 : total_distance = 20)
  (h2 : walking_speed = 4)
  (h3 : cycling_speed = 20) :
  ∃ (x : ℝ) (t : ℝ),
    0 < x ∧ x < total_distance ∧
    (x / cycling_speed + (total_distance - x) / walking_speed = 
     x / walking_speed + (total_distance - x) / cycling_speed) ∧
    x = 10 ∧
    t = 3 ∧
    t = x / cycling_speed + (total_distance - x) / walking_speed :=
by sorry

end NUMINAMATH_CALUDE_bicycle_problem_l105_10584


namespace NUMINAMATH_CALUDE_archimedes_segment_theorem_l105_10521

/-- Archimedes' Theorem applied to segments -/
theorem archimedes_segment_theorem 
  (b c : ℝ) 
  (CT AK CK AT AB AC : ℝ) 
  (h1 : CT = AK) 
  (h2 : CK = AK + AB) 
  (h3 : AT = CK) 
  (h4 : AC = b) : 
  AT = (b + c) / 2 ∧ CT = (b - c) / 2 := by
  sorry

#check archimedes_segment_theorem

end NUMINAMATH_CALUDE_archimedes_segment_theorem_l105_10521


namespace NUMINAMATH_CALUDE_cubic_root_difference_l105_10543

/-- The cubic equation x³ - px² + (p² - 1)/4x = 0 has a difference of 1 between its largest and smallest roots -/
theorem cubic_root_difference (p : ℝ) : 
  let f : ℝ → ℝ := λ x => x^3 - p*x^2 + (p^2 - 1)/4*x
  let roots := {x : ℝ | f x = 0}
  ∃ (a b : ℝ), a ∈ roots ∧ b ∈ roots ∧ ∀ c ∈ roots, a ≤ c ∧ c ≤ b ∧ b - a = 1 :=
sorry

end NUMINAMATH_CALUDE_cubic_root_difference_l105_10543


namespace NUMINAMATH_CALUDE_chicken_feathers_l105_10522

theorem chicken_feathers (initial_feathers : ℕ) (cars_dodged : ℕ) : 
  initial_feathers = 5263 →
  cars_dodged = 23 →
  initial_feathers - (cars_dodged * 2) = 5217 := by
  sorry

end NUMINAMATH_CALUDE_chicken_feathers_l105_10522


namespace NUMINAMATH_CALUDE_volume_equality_l105_10593

/-- The region R₁ bounded by x² = 4y, x² = -4y, x = 4, and x = -4 -/
def R₁ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 = 4*p.2 ∨ p.1^2 = -4*p.2 ∨ p.1 = 4 ∨ p.1 = -4}

/-- The region R₂ satisfying x² - y² ≤ 16, x² + (y - 2)² ≥ 4, and x² + (y + 2)² ≥ 4 -/
def R₂ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 - p.2^2 ≤ 16 ∧ p.1^2 + (p.2 - 2)^2 ≥ 4 ∧ p.1^2 + (p.2 + 2)^2 ≥ 4}

/-- The volume V₁ obtained by rotating R₁ about the y-axis -/
noncomputable def V₁ : ℝ := sorry

/-- The volume V₂ obtained by rotating R₂ about the y-axis -/
noncomputable def V₂ : ℝ := sorry

/-- The theorem stating that V₁ equals V₂ -/
theorem volume_equality : V₁ = V₂ := by sorry

end NUMINAMATH_CALUDE_volume_equality_l105_10593


namespace NUMINAMATH_CALUDE_mn_length_in_isosceles_triangle_l105_10507

-- Define the triangle XYZ
structure Triangle :=
  (area : ℝ)
  (altitude : ℝ)
  (isIsosceles : Bool)

-- Define the line MN
structure ParallelLine :=
  (length : ℝ)

-- Define the trapezoid formed by MN
structure Trapezoid :=
  (area : ℝ)

-- Main theorem
theorem mn_length_in_isosceles_triangle 
  (XYZ : Triangle) 
  (MN : ParallelLine) 
  (trap : Trapezoid) : 
  XYZ.area = 144 ∧ 
  XYZ.altitude = 24 ∧ 
  XYZ.isIsosceles = true ∧
  trap.area = 108 →
  MN.length = 6 :=
sorry

end NUMINAMATH_CALUDE_mn_length_in_isosceles_triangle_l105_10507


namespace NUMINAMATH_CALUDE_jacket_price_reduction_l105_10537

theorem jacket_price_reduction (P : ℝ) (x : ℝ) : 
  P > 0 → 
  0 ≤ x → 
  x ≤ 100 → 
  P * (1 - x / 100) * (1 - 20 / 100) * (1 + 2 / 3) = P → 
  x = 25 := by
sorry

end NUMINAMATH_CALUDE_jacket_price_reduction_l105_10537


namespace NUMINAMATH_CALUDE_negative_885_degrees_conversion_l105_10509

theorem negative_885_degrees_conversion :
  ∃ (k : ℤ) (α : ℝ), 
    -885 * (π / 180) = 2 * k * π + α ∧
    0 ≤ α ∧ α ≤ 2 * π ∧
    k = -6 ∧ α = 13 * π / 12 := by
  sorry

end NUMINAMATH_CALUDE_negative_885_degrees_conversion_l105_10509


namespace NUMINAMATH_CALUDE_park_area_l105_10527

/-- Given a rectangular park with length l and width w, where:
    1) l = 3w + 20
    2) The perimeter is 800 feet
    Prove that the area of the park is 28,975 square feet -/
theorem park_area (w l : ℝ) (h1 : l = 3 * w + 20) (h2 : 2 * l + 2 * w = 800) :
  w * l = 28975 := by
  sorry

end NUMINAMATH_CALUDE_park_area_l105_10527


namespace NUMINAMATH_CALUDE_tan_equality_periodic_l105_10564

theorem tan_equality_periodic (n : ℤ) : 
  -90 < n ∧ n < 90 ∧ Real.tan (n * π / 180) = Real.tan (243 * π / 180) → n = 63 := by
  sorry

end NUMINAMATH_CALUDE_tan_equality_periodic_l105_10564


namespace NUMINAMATH_CALUDE_db_length_l105_10563

/-- Triangle ABC with altitudes and median -/
structure TriangleABC where
  -- Points
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  M : ℝ × ℝ
  -- CD is altitude to AB
  cd_altitude : (C.1 - D.1) * (B.1 - A.1) + (C.2 - D.2) * (B.2 - A.2) = 0
  -- AE is altitude to BC
  ae_altitude : (A.1 - E.1) * (C.1 - B.1) + (A.2 - E.2) * (C.2 - B.2) = 0
  -- AM is median to BC
  am_median : M = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  -- Given lengths
  ab_length : Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 12
  cd_length : Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2) = 5
  ae_length : Real.sqrt ((E.1 - A.1)^2 + (E.2 - A.2)^2) = 4
  am_length : Real.sqrt ((M.1 - A.1)^2 + (M.2 - A.2)^2) = 6

/-- The length of DB in the given triangle is 15 -/
theorem db_length (t : TriangleABC) : 
  Real.sqrt ((t.D.1 - t.B.1)^2 + (t.D.2 - t.B.2)^2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_db_length_l105_10563


namespace NUMINAMATH_CALUDE_remainder_problem_l105_10571

theorem remainder_problem : Int.mod (179 + 231 - 359) 37 = 14 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l105_10571


namespace NUMINAMATH_CALUDE_smallest_perfect_cube_l105_10525

/-- Given distinct prime numbers p, q, and r, prove that (p * q * r^2)^3 is the smallest positive
    perfect cube that includes the factor n = p * q^2 * r^4 -/
theorem smallest_perfect_cube (p q r : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r)
  (hpq : p ≠ q) (hpr : p ≠ r) (hqr : q ≠ r) :
  ∃ (k : ℕ), k > 0 ∧ (p * q * r^2)^3 = k^3 ∧
  ∀ (m : ℕ), m > 0 → m^3 ≥ (p * q * r^2)^3 → (p * q^2 * r^4) ∣ m^3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_perfect_cube_l105_10525


namespace NUMINAMATH_CALUDE_simplify_sqrt_sum_l105_10573

theorem simplify_sqrt_sum : 
  Real.sqrt (8 + 6 * Real.sqrt 3) + Real.sqrt (8 - 6 * Real.sqrt 3) = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_sum_l105_10573


namespace NUMINAMATH_CALUDE_leftover_value_is_fifteen_l105_10560

def quarters_per_roll : ℕ := 50
def dimes_per_roll : ℕ := 60
def quarter_value : ℚ := 0.25
def dime_value : ℚ := 0.10

def james_quarters : ℕ := 97
def james_dimes : ℕ := 178
def lindsay_quarters : ℕ := 143
def lindsay_dimes : ℕ := 292

def total_quarters : ℕ := james_quarters + lindsay_quarters
def total_dimes : ℕ := james_dimes + lindsay_dimes

def leftover_quarters : ℕ := total_quarters % quarters_per_roll
def leftover_dimes : ℕ := total_dimes % dimes_per_roll

def leftover_value : ℚ := leftover_quarters * quarter_value + leftover_dimes * dime_value

theorem leftover_value_is_fifteen :
  leftover_value = 15 := by sorry

end NUMINAMATH_CALUDE_leftover_value_is_fifteen_l105_10560


namespace NUMINAMATH_CALUDE_ellipse_equation_1_ellipse_equation_2_x_axis_ellipse_equation_2_y_axis_l105_10554

-- Define the ellipse type
structure Ellipse where
  a : ℝ  -- semi-major axis
  b : ℝ  -- semi-minor axis
  c : ℝ  -- distance from center to focus
  e : ℝ  -- eccentricity

-- Define the standard equation of an ellipse
def standardEquation (E : Ellipse) (x y : ℝ) : Prop :=
  x^2 / E.a^2 + y^2 / E.b^2 = 1

-- Theorem for the first condition
theorem ellipse_equation_1 :
  ∀ E : Ellipse,
  E.c = 6 →
  E.e = 2/3 →
  (∀ x y : ℝ, standardEquation E x y ↔ y^2/81 + x^2/45 = 1) :=
sorry

-- Theorem for the second condition (foci on x-axis)
theorem ellipse_equation_2_x_axis :
  ∀ E : Ellipse,
  E.a = 5 →
  E.c = 3 →
  (∀ x y : ℝ, standardEquation E x y ↔ x^2/25 + y^2/16 = 1) :=
sorry

-- Theorem for the second condition (foci on y-axis)
theorem ellipse_equation_2_y_axis :
  ∀ E : Ellipse,
  E.a = 5 →
  E.c = 3 →
  (∀ x y : ℝ, standardEquation E y x ↔ y^2/25 + x^2/16 = 1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_1_ellipse_equation_2_x_axis_ellipse_equation_2_y_axis_l105_10554


namespace NUMINAMATH_CALUDE_max_non_managers_proof_l105_10526

/-- Represents a department in the company -/
structure Department where
  managers : ℕ
  nonManagers : ℕ

/-- The ratio condition for managers to non-managers -/
def validRatio (d : Department) : Prop :=
  (d.managers : ℚ) / d.nonManagers > 7 / 24

/-- The maximum number of non-managers allowed -/
def maxNonManagers : ℕ := 27

/-- The minimum number of managers required -/
def minManagers : ℕ := 8

theorem max_non_managers_proof (d : Department) 
    (h1 : d.managers ≥ minManagers) 
    (h2 : validRatio d) 
    (h3 : d.nonManagers ≤ maxNonManagers) :
    d.nonManagers = maxNonManagers :=
  sorry

#check max_non_managers_proof

end NUMINAMATH_CALUDE_max_non_managers_proof_l105_10526


namespace NUMINAMATH_CALUDE_max_value_on_unit_circle_l105_10591

theorem max_value_on_unit_circle (x y : ℝ) (h : x^2 + y^2 = 1) :
  ∃ (M : ℝ), M = 7 ∧ ∀ (a b : ℝ), a^2 + b^2 = 1 → a^2 + 4*b + 3 ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_on_unit_circle_l105_10591


namespace NUMINAMATH_CALUDE_prob_at_least_one_woman_l105_10535

/-- The probability of selecting at least one woman when choosing 4 people at random
    from a group of 8 men and 4 women is equal to 85/99. -/
theorem prob_at_least_one_woman (total : ℕ) (men : ℕ) (women : ℕ) (selected : ℕ) :
  total = men + women →
  men = 8 →
  women = 4 →
  selected = 4 →
  (1 : ℚ) - (men.choose selected : ℚ) / (total.choose selected : ℚ) = 85 / 99 := by
  sorry

#check prob_at_least_one_woman

end NUMINAMATH_CALUDE_prob_at_least_one_woman_l105_10535


namespace NUMINAMATH_CALUDE_carl_driving_hours_l105_10561

/-- 
Given Carl's initial daily driving hours and additional weekly hours after promotion, 
prove that the total hours he will drive in two weeks is equal to 40 hours.
-/
theorem carl_driving_hours (initial_daily_hours : ℝ) (additional_weekly_hours : ℝ) : 
  initial_daily_hours = 2 ∧ additional_weekly_hours = 6 → 
  (initial_daily_hours * 7 + additional_weekly_hours) * 2 = 40 := by
sorry

end NUMINAMATH_CALUDE_carl_driving_hours_l105_10561


namespace NUMINAMATH_CALUDE_melted_ice_cream_height_l105_10536

/-- The height of a melted ice cream scoop -/
theorem melted_ice_cream_height (r_sphere r_cylinder : ℝ) (h : ℝ) : 
  r_sphere = 3 →
  r_cylinder = 12 →
  (4 / 3) * π * r_sphere^3 = π * r_cylinder^2 * h →
  h = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_melted_ice_cream_height_l105_10536


namespace NUMINAMATH_CALUDE_haley_flash_drive_files_l105_10513

/-- Calculates the number of files remaining on a flash drive after compression and deletion. -/
def files_remaining (music_files : ℕ) (video_files : ℕ) (document_files : ℕ) 
                    (music_compression : ℕ) (video_compression : ℕ) 
                    (deleted_files : ℕ) : ℕ :=
  music_files * music_compression + video_files * video_compression + document_files - deleted_files

/-- Theorem stating the number of files remaining on Haley's flash drive -/
theorem haley_flash_drive_files : 
  files_remaining 27 42 12 2 3 11 = 181 := by
  sorry

end NUMINAMATH_CALUDE_haley_flash_drive_files_l105_10513


namespace NUMINAMATH_CALUDE_park_visitors_difference_l105_10581

theorem park_visitors_difference (total : ℕ) (hikers : ℕ) (bikers : ℕ) : 
  total = 676 → hikers = 427 → total = hikers + bikers → hikers - bikers = 178 := by
  sorry

end NUMINAMATH_CALUDE_park_visitors_difference_l105_10581


namespace NUMINAMATH_CALUDE_gcd_of_2_powers_l105_10546

theorem gcd_of_2_powers : Nat.gcd (2^2021 - 1) (2^2000 - 1) = 2^21 - 1 := by sorry

end NUMINAMATH_CALUDE_gcd_of_2_powers_l105_10546


namespace NUMINAMATH_CALUDE_smallest_composite_no_small_factors_l105_10545

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def has_no_small_prime_factors (n : ℕ) : Prop := ∀ p, p < 15 → ¬(Nat.Prime p ∧ p ∣ n)

theorem smallest_composite_no_small_factors : 
  (is_composite 289 ∧ has_no_small_prime_factors 289) ∧ 
  (∀ m : ℕ, m < 289 → ¬(is_composite m ∧ has_no_small_prime_factors m)) :=
sorry

end NUMINAMATH_CALUDE_smallest_composite_no_small_factors_l105_10545


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l105_10530

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 36 = 1

-- Define the asymptote equations
def asymptote_equations (x y : ℝ) : Prop :=
  y = 3*x ∨ y = -3*x

-- Theorem statement
theorem hyperbola_asymptotes :
  ∀ x y : ℝ, hyperbola_equation x y → asymptote_equations x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l105_10530


namespace NUMINAMATH_CALUDE_race_finish_distance_l105_10555

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  distance : ℝ

/-- Represents the state of the race -/
structure Race where
  sasha : Runner
  lyosha : Runner
  kolya : Runner
  raceLength : ℝ

/-- The theorem to be proved -/
theorem race_finish_distance (r : Race) : 
  r.raceLength = 100 ∧ 
  r.sasha.distance - r.lyosha.distance = 10 ∧
  r.lyosha.distance - r.kolya.distance = 10 ∧
  r.sasha.distance = r.raceLength →
  r.sasha.distance - r.kolya.distance = 19 := by
  sorry

end NUMINAMATH_CALUDE_race_finish_distance_l105_10555


namespace NUMINAMATH_CALUDE_people_who_got_off_train_l105_10578

theorem people_who_got_off_train (initial_people : ℕ) (people_who_got_on : ℕ) (final_people : ℕ) :
  initial_people = 82 →
  people_who_got_on = 17 →
  final_people = 73 →
  ∃ (people_who_got_off : ℕ), 
    initial_people - people_who_got_off + people_who_got_on = final_people ∧
    people_who_got_off = 26 :=
by
  sorry

end NUMINAMATH_CALUDE_people_who_got_off_train_l105_10578


namespace NUMINAMATH_CALUDE_point_on_line_l105_10534

/-- Given a line with equation x = 2y + 5 and two points (m, n) and (m + 2, n + k) on this line,
    prove that k = 1. -/
theorem point_on_line (m n k : ℝ) : 
  (m = 2*n + 5) ∧ (m + 2 = 2*(n + k) + 5) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l105_10534


namespace NUMINAMATH_CALUDE_chocolate_packs_l105_10520

theorem chocolate_packs (total packs_cookies packs_cake : ℕ) 
  (h_total : total = 42)
  (h_cookies : packs_cookies = 4)
  (h_cake : packs_cake = 22) :
  total - packs_cookies - packs_cake = 16 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_packs_l105_10520


namespace NUMINAMATH_CALUDE_point_distance_to_line_l105_10556

theorem point_distance_to_line (m : ℝ) : 
  let M : ℝ × ℝ := (1, 4)
  let l := {(x, y) : ℝ × ℝ | m * x + y - 1 = 0}
  (abs (m * M.1 + M.2 - 1) / Real.sqrt (m^2 + 1) = 3) → (m = 0 ∨ m = 3/4) := by
sorry

end NUMINAMATH_CALUDE_point_distance_to_line_l105_10556


namespace NUMINAMATH_CALUDE_chip_exits_at_A2_l105_10590

-- Define the grid size
def gridSize : Nat := 4

-- Define the possible directions
inductive Direction
| Up
| Down
| Left
| Right

-- Define a cell position
structure Position where
  row : Nat
  col : Nat

-- Define the state of the game
structure GameState where
  chipPosition : Position
  arrows : Array (Array Direction)

-- Define the initial state
def initialState : GameState := sorry

-- Define a function to get the next position based on current position and direction
def nextPosition (pos : Position) (dir : Direction) : Position := sorry

-- Define a function to flip the direction
def flipDirection (dir : Direction) : Direction := sorry

-- Define a function to make a move
def makeMove (state : GameState) : GameState := sorry

-- Define a function to check if a position is out of bounds
def isOutOfBounds (pos : Position) : Bool := sorry

-- Define a function to simulate the game until the chip exits
def simulateUntilExit (state : GameState) : Position := sorry

-- The main theorem to prove
theorem chip_exits_at_A2 :
  let finalPos := simulateUntilExit initialState
  finalPos = Position.mk 0 1 := sorry

end NUMINAMATH_CALUDE_chip_exits_at_A2_l105_10590


namespace NUMINAMATH_CALUDE_shift_proof_l105_10583

def original_function (x : ℝ) : ℝ := -3 * x + 2

def vertical_shift : ℝ := 3

def shifted_function (x : ℝ) : ℝ := original_function x + vertical_shift

theorem shift_proof : 
  ∀ x : ℝ, shifted_function x = -3 * x + 5 := by
sorry

end NUMINAMATH_CALUDE_shift_proof_l105_10583


namespace NUMINAMATH_CALUDE_probability_between_R_and_S_l105_10508

/-- Given points P, Q, R, and S on a line segment PQ, where PQ = 4PR and PQ = 8RS,
    the probability of a randomly selected point on PQ being between R and S is 1/8. -/
theorem probability_between_R_and_S (P Q R S : ℝ) 
  (h_order : P ≤ R ∧ R ≤ S ∧ S ≤ Q)
  (h_PQ_PR : Q - P = 4 * (R - P))
  (h_PQ_RS : Q - P = 8 * (S - R)) :
  (S - R) / (Q - P) = 1 / 8 := by sorry

end NUMINAMATH_CALUDE_probability_between_R_and_S_l105_10508


namespace NUMINAMATH_CALUDE_profit_difference_l105_10562

def chocolate_cakes_made : ℕ := 40
def vanilla_cakes_made : ℕ := 35
def strawberry_cakes_made : ℕ := 28
def pastries_made : ℕ := 153

def chocolate_cake_price : ℕ := 10
def vanilla_cake_price : ℕ := 12
def strawberry_cake_price : ℕ := 15
def pastry_price : ℕ := 5

def chocolate_cakes_sold : ℕ := 30
def vanilla_cakes_sold : ℕ := 25
def strawberry_cakes_sold : ℕ := 20
def pastries_sold : ℕ := 106

def total_cake_revenue : ℕ := 
  chocolate_cakes_sold * chocolate_cake_price +
  vanilla_cakes_sold * vanilla_cake_price +
  strawberry_cakes_sold * strawberry_cake_price

def total_pastry_revenue : ℕ := pastries_sold * pastry_price

theorem profit_difference : total_cake_revenue - total_pastry_revenue = 370 := by
  sorry

end NUMINAMATH_CALUDE_profit_difference_l105_10562


namespace NUMINAMATH_CALUDE_area_of_triangle_LGH_l105_10519

-- Define the circle and points
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

def P : ℝ × ℝ := (0, 0)
def L : ℝ × ℝ := (-24, 0)
def M : ℝ × ℝ := (-10, 0)
def N : ℝ × ℝ := (10, 0)

-- Define the chords
def EF : Set (ℝ × ℝ) := {p | p.2 = 6}
def GH : Set (ℝ × ℝ) := {p | p.2 = 8}

-- State the theorem
theorem area_of_triangle_LGH : 
  ∀ (G H : ℝ × ℝ),
  G ∈ GH → H ∈ GH →
  G.1 < H.1 →
  H.1 - G.1 = 16 →
  (∀ (E F : ℝ × ℝ), E ∈ EF → F ∈ EF → F.1 - E.1 = 12) →
  (∀ p, p ∈ Circle P 10) →
  (∀ x, (x, 0) ∈ Set.Icc L N → (x, 0) ∈ Circle P 10) →
  let triangle_area := (1 / 2) * 16 * 6
  triangle_area = 48 := by sorry

end NUMINAMATH_CALUDE_area_of_triangle_LGH_l105_10519


namespace NUMINAMATH_CALUDE_percent_of_x_is_y_l105_10557

theorem percent_of_x_is_y (x y : ℝ) (h : 0.25 * (x - y) = 0.15 * (x + y)) : y = 0.25 * x := by
  sorry

end NUMINAMATH_CALUDE_percent_of_x_is_y_l105_10557


namespace NUMINAMATH_CALUDE_line_intersection_plane_intersection_l105_10597

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the properties
variable (lies_in : Line → Plane → Prop)
variable (intersects_line : Line → Line → Prop)
variable (intersects_plane : Plane → Plane → Prop)

-- State the theorem
theorem line_intersection_plane_intersection 
  (a b : Line) (α β : Plane) 
  (ha : lies_in a α) (hb : lies_in b β) :
  (∀ (a b : Line) (α β : Plane), lies_in a α → lies_in b β → 
    intersects_line a b → intersects_plane α β) ∧ 
  (∃ (a b : Line) (α β : Plane), lies_in a α ∧ lies_in b β ∧ 
    intersects_plane α β ∧ ¬intersects_line a b) :=
sorry

end NUMINAMATH_CALUDE_line_intersection_plane_intersection_l105_10597


namespace NUMINAMATH_CALUDE_inequality_equivalence_l105_10575

-- Define the logarithm base 3
noncomputable def log3 (x : ℝ) : ℝ := Real.log x / Real.log 3

-- State the theorem
theorem inequality_equivalence :
  ∀ x : ℝ, (0 < x ∧ |x + log3 x| < |x| + |log3 x|) ↔ (0 < x ∧ x < 1) :=
sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l105_10575


namespace NUMINAMATH_CALUDE_cubic_equation_root_b_value_l105_10532

theorem cubic_equation_root_b_value (a b : ℚ) : 
  (∃ x : ℝ, x = 3 + Real.sqrt 5 ∧ x^3 + a*x^2 + b*x + 12 = 0) → b = -14 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_root_b_value_l105_10532


namespace NUMINAMATH_CALUDE_complex_equation_solution_l105_10585

theorem complex_equation_solution (z : ℂ) : z * (2 - Complex.I) = 3 + Complex.I → z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l105_10585


namespace NUMINAMATH_CALUDE_canada_population_l105_10514

/-- The number of moose in Canada -/
def moose : ℕ := 1000000

/-- The number of beavers in Canada -/
def beavers : ℕ := 2 * moose

/-- The number of humans in Canada -/
def humans : ℕ := 19 * beavers

/-- Theorem: Given the relationship between moose, beavers, and humans in Canada,
    and a moose population of 1 million, the human population is 38 million. -/
theorem canada_population : humans = 38000000 := by
  sorry

end NUMINAMATH_CALUDE_canada_population_l105_10514


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l105_10559

theorem smallest_solution_of_equation (x : ℝ) :
  x > 0 ∧ x / 4 + 2 / (3 * x) = 5 / 6 →
  x ≥ 4 / 3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l105_10559


namespace NUMINAMATH_CALUDE_zebra_population_last_year_l105_10531

/-- The number of zebras in a national park over two consecutive years. -/
structure ZebraPopulation where
  current : ℕ
  born : ℕ
  died : ℕ
  last_year : ℕ

/-- Theorem stating the relationship between the zebra population this year and last year. -/
theorem zebra_population_last_year (zp : ZebraPopulation)
    (h1 : zp.current = 725)
    (h2 : zp.born = 419)
    (h3 : zp.died = 263)
    : zp.last_year = 569 := by
  sorry

end NUMINAMATH_CALUDE_zebra_population_last_year_l105_10531


namespace NUMINAMATH_CALUDE_inequality_proof_l105_10528

theorem inequality_proof (a b c d : ℝ) : 
  (a + b + c + d)^2 ≤ 3 * (a^2 + b^2 + c^2 + d^2) + 6 * a * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l105_10528


namespace NUMINAMATH_CALUDE_pie_distribution_l105_10551

/-- Given a pie with 48 slices, prove that after distributing specific fractions, 2 slices remain -/
theorem pie_distribution (total_slices : ℕ) (joe_fraction darcy_fraction carl_fraction emily_fraction frank_percent : ℚ) : 
  total_slices = 48 →
  joe_fraction = 1/3 →
  darcy_fraction = 1/4 →
  carl_fraction = 1/6 →
  emily_fraction = 1/8 →
  frank_percent = 10/100 →
  total_slices - (total_slices * joe_fraction).floor - (total_slices * darcy_fraction).floor - 
  (total_slices * carl_fraction).floor - (total_slices * emily_fraction).floor - 
  (total_slices * frank_percent).floor = 2 := by
sorry

end NUMINAMATH_CALUDE_pie_distribution_l105_10551


namespace NUMINAMATH_CALUDE_seventy_third_digit_is_zero_l105_10506

/-- The number consisting of 112 ones -/
def number_of_ones : ℕ := 11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111

/-- The square of the number consisting of 112 ones -/
def square_of_ones : ℕ := number_of_ones * number_of_ones

/-- The seventy-third digit from the end of a natural number -/
def seventy_third_digit_from_end (n : ℕ) : ℕ :=
  (n / 10^72) % 10

theorem seventy_third_digit_is_zero :
  seventy_third_digit_from_end square_of_ones = 0 := by
  sorry

end NUMINAMATH_CALUDE_seventy_third_digit_is_zero_l105_10506


namespace NUMINAMATH_CALUDE_three_divides_difference_l105_10577

/-- Represents a three-digit number ABC --/
structure ThreeDigitNumber where
  A : Nat
  B : Nat
  C : Nat
  A_is_digit : A < 10
  B_is_digit : B < 10
  C_is_digit : C < 10

/-- The value of a three-digit number ABC --/
def value (n : ThreeDigitNumber) : Nat :=
  100 * n.A + 10 * n.B + n.C

/-- The reversed value of a three-digit number ABC (i.e., CBA) --/
def reversed_value (n : ThreeDigitNumber) : Nat :=
  100 * n.C + 10 * n.B + n.A

theorem three_divides_difference (n : ThreeDigitNumber) (h : n.A ≠ n.C) :
  3 ∣ (value n - reversed_value n) := by
  sorry

end NUMINAMATH_CALUDE_three_divides_difference_l105_10577


namespace NUMINAMATH_CALUDE_scientific_notation_of_1500000_l105_10548

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_1500000 :
  toScientificNotation 1500000 = ScientificNotation.mk 1.5 6 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_of_1500000_l105_10548


namespace NUMINAMATH_CALUDE_parabola_sum_zero_l105_10516

/-- A parabola passing through two specific points has a + b + c = 0 --/
theorem parabola_sum_zero (a b c : ℝ) : 
  (∀ x y : ℝ, y = a * x^2 + b * x + c) →
  (a * (-2)^2 + b * (-2) + c = -3) →
  (a * 2^2 + b * 2 + c = 5) →
  a + b + c = 0 := by
sorry

end NUMINAMATH_CALUDE_parabola_sum_zero_l105_10516


namespace NUMINAMATH_CALUDE_div_power_eq_power_l105_10515

theorem div_power_eq_power (a : ℝ) : a^4 / (-a)^2 = a^2 := by
  sorry

end NUMINAMATH_CALUDE_div_power_eq_power_l105_10515


namespace NUMINAMATH_CALUDE_min_sum_squares_l105_10579

theorem min_sum_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 8) :
  ∃ (m : ℝ), (∀ a b c : ℝ, a^3 + b^3 + c^3 - 3*a*b*c = 8 → a^2 + b^2 + c^2 ≥ m) ∧
             (x^2 + y^2 + z^2 ≥ m) ∧
             m = 3 :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l105_10579


namespace NUMINAMATH_CALUDE_dinosaur_model_price_reduction_l105_10538

/-- The percentage reduction in dinosaur model prices for a school purchase --/
theorem dinosaur_model_price_reduction :
  -- Original price per model
  ∀ (original_price : ℕ),
  -- Number of models for kindergarten
  ∀ (k : ℕ),
  -- Number of models for elementary
  ∀ (e : ℕ),
  -- Total number of models
  ∀ (total : ℕ),
  -- Total amount paid
  ∀ (total_paid : ℕ),
  -- Conditions
  original_price = 100 →
  k = 2 →
  e = 2 * k →
  total = k + e →
  total > 5 →
  total_paid = 570 →
  -- Conclusion
  (1 - total_paid / (total * original_price : ℚ)) * 100 = 5 := by
sorry

end NUMINAMATH_CALUDE_dinosaur_model_price_reduction_l105_10538


namespace NUMINAMATH_CALUDE_apps_left_l105_10598

/-- 
Given that Dave had 23 apps initially and deleted 18 apps, 
prove that he has 5 apps left.
-/
theorem apps_left (initial_apps : ℕ) (deleted_apps : ℕ) (apps_left : ℕ) : 
  initial_apps = 23 → deleted_apps = 18 → apps_left = initial_apps - deleted_apps → apps_left = 5 := by
  sorry

end NUMINAMATH_CALUDE_apps_left_l105_10598


namespace NUMINAMATH_CALUDE_candy_bar_consumption_l105_10580

/-- Given that a candy bar contains 31 calories and a person consumed 341 calories,
    prove that the number of candy bars eaten is 11. -/
theorem candy_bar_consumption (calories_per_bar : ℕ) (total_calories : ℕ) : 
  calories_per_bar = 31 → total_calories = 341 → total_calories / calories_per_bar = 11 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_consumption_l105_10580


namespace NUMINAMATH_CALUDE_arithmetic_sequence_condition_l105_10586

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_condition 
  (a : ℕ → ℝ) (d : ℝ) (h : is_arithmetic_sequence a) :
  (d > 0 ↔ a 2 > a 1) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_condition_l105_10586


namespace NUMINAMATH_CALUDE_circle_equation_l105_10592

/-- A circle C in the polar coordinate system -/
structure PolarCircle where
  /-- The point through which the circle passes -/
  passingPoint : (ℝ × ℝ)
  /-- The line equation whose intersection with polar axis determines the circle's center -/
  centerLine : ℝ → ℝ → Prop

/-- The polar equation of a circle -/
def polarEquation (c : PolarCircle) (ρ θ : ℝ) : Prop := sorry

theorem circle_equation (c : PolarCircle) (h1 : c.passingPoint = (Real.sqrt 2, π/4)) 
  (h2 : c.centerLine = fun ρ θ ↦ ρ * Real.sin (θ - π/3) = -Real.sqrt 3/2) :
  polarEquation c = fun ρ θ ↦ ρ = 2 * Real.cos θ := by sorry

end NUMINAMATH_CALUDE_circle_equation_l105_10592


namespace NUMINAMATH_CALUDE_talia_drives_16_miles_l105_10568

/-- Represents the total distance Talia drives in a day -/
def total_distance (home_to_park park_to_grocery grocery_to_home : ℝ) : ℝ :=
  home_to_park + park_to_grocery + grocery_to_home

/-- Theorem stating that Talia drives 16 miles given the distances between locations -/
theorem talia_drives_16_miles :
  let home_to_park : ℝ := 5
  let park_to_grocery : ℝ := 3
  let grocery_to_home : ℝ := 8
  total_distance home_to_park park_to_grocery grocery_to_home = 16 := by
  sorry

end NUMINAMATH_CALUDE_talia_drives_16_miles_l105_10568


namespace NUMINAMATH_CALUDE_peanut_butter_jar_servings_l105_10500

/-- The number of servings in a jar of peanut butter -/
def peanut_butter_servings (jar_contents : ℚ) (serving_size : ℚ) : ℚ :=
  jar_contents / serving_size

theorem peanut_butter_jar_servings :
  let jar_contents : ℚ := 35 + 4/5
  let serving_size : ℚ := 2 + 1/3
  peanut_butter_servings jar_contents serving_size = 15 + 17/35 := by
  sorry

end NUMINAMATH_CALUDE_peanut_butter_jar_servings_l105_10500


namespace NUMINAMATH_CALUDE_paula_paint_theorem_l105_10599

/-- Represents the number of rooms that can be painted with a given amount of paint -/
structure PaintCapacity where
  total_rooms : ℕ
  cans : ℕ

/-- Calculates the number of cans needed to paint a given number of rooms -/
def cans_needed (initial : PaintCapacity) (lost_cans : ℕ) (rooms_to_paint : ℕ) : ℕ :=
  let rooms_per_can := initial.total_rooms / initial.cans
  rooms_to_paint / rooms_per_can

theorem paula_paint_theorem (initial : PaintCapacity) (lost_cans : ℕ) :
  initial.total_rooms = 40 →
  initial.cans = initial.cans - lost_cans + lost_cans →
  lost_cans = 6 →
  cans_needed initial lost_cans 30 = 18 := by
  sorry

#check paula_paint_theorem

end NUMINAMATH_CALUDE_paula_paint_theorem_l105_10599


namespace NUMINAMATH_CALUDE_some_number_value_l105_10547

theorem some_number_value (some_number : ℝ) :
  ((3.242 * some_number) / 100) = 0.038903999999999994 →
  ∃ ε > 0, |some_number - 1.2| < ε := by
sorry

end NUMINAMATH_CALUDE_some_number_value_l105_10547


namespace NUMINAMATH_CALUDE_expression_simplification_l105_10588

theorem expression_simplification 
  (a b c : ℝ) 
  (h1 : a + b + c = 0) 
  (h2 : a ≠ 0) 
  (h3 : b ≠ 0) 
  (h4 : c ≠ 0) : 
  a * (1/b + 1/c) + b * (1/a + 1/c) + c * (1/a + 1/b) = -3 := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l105_10588


namespace NUMINAMATH_CALUDE_positive_integer_solutions_of_equation_l105_10550

theorem positive_integer_solutions_of_equation : 
  ∀ x y : ℕ+, 
    (x : ℚ) - (y : ℚ) = (x : ℚ) / (y : ℚ) + (x : ℚ)^2 / (y : ℚ)^2 + (x : ℚ)^3 / (y : ℚ)^3 
    ↔ (x = 28 ∧ y = 14) ∨ (x = 112 ∧ y = 28) :=
by sorry

end NUMINAMATH_CALUDE_positive_integer_solutions_of_equation_l105_10550


namespace NUMINAMATH_CALUDE_translated_line_through_origin_l105_10544

/-- A line in the Cartesian plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translate a line vertically by a given amount -/
def translate_line (l : Line) (dy : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept - dy }

/-- Check if a line passes through a point -/
def passes_through (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.intercept

theorem translated_line_through_origin (b : ℝ) :
  let original_line : Line := { slope := 2, intercept := b }
  let translated_line := translate_line original_line 2
  passes_through translated_line 0 0 → b = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_translated_line_through_origin_l105_10544


namespace NUMINAMATH_CALUDE_a_in_range_l105_10552

/-- Proposition p: for any real number x, ax^2 + ax + 1 > 0 always holds -/
def prop_p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 > 0

/-- Proposition q: the equation x^2 - x + a = 0 has real roots with respect to x -/
def prop_q (a : ℝ) : Prop := ∃ x : ℝ, x^2 - x + a = 0

/-- The set representing the range of a: (-∞, 0) ∪ (1/4, 4) -/
def range_a : Set ℝ := {a | a < 0 ∨ (1/4 < a ∧ a < 4)}

/-- Main theorem: If only one of prop_p and prop_q is true, then a is in range_a -/
theorem a_in_range (a : ℝ) : 
  (prop_p a ∧ ¬prop_q a) ∨ (¬prop_p a ∧ prop_q a) → a ∈ range_a := by sorry

end NUMINAMATH_CALUDE_a_in_range_l105_10552


namespace NUMINAMATH_CALUDE_relationship_between_a_and_b_l105_10518

theorem relationship_between_a_and_b (a b : ℝ) 
  (h1 : (1003 : ℝ) ^ a + (1004 : ℝ) ^ b = (2006 : ℝ) ^ b)
  (h2 : (997 : ℝ) ^ a + (1009 : ℝ) ^ b = (2007 : ℝ) ^ a) : 
  a < b := by
sorry

end NUMINAMATH_CALUDE_relationship_between_a_and_b_l105_10518


namespace NUMINAMATH_CALUDE_even_function_period_2_equivalence_l105_10533

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def has_period_2 (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = f x

theorem even_function_period_2_equivalence (f : ℝ → ℝ) (h : is_even f) :
  (∀ x, f (1 - x) = f (1 + x)) ↔ has_period_2 f :=
sorry

end NUMINAMATH_CALUDE_even_function_period_2_equivalence_l105_10533


namespace NUMINAMATH_CALUDE_subtraction_and_decimal_conversion_l105_10541

theorem subtraction_and_decimal_conversion : 3/4 - 1/16 = 0.6875 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_and_decimal_conversion_l105_10541


namespace NUMINAMATH_CALUDE_total_project_hours_l105_10569

def project_hours (kate_hours : ℝ) : ℝ × ℝ × ℝ := 
  let pat_hours := 2 * kate_hours
  let mark_hours := kate_hours + 65
  (pat_hours, kate_hours, mark_hours)

theorem total_project_hours : 
  ∃ (kate_hours : ℝ), 
    let (pat_hours, _, mark_hours) := project_hours kate_hours
    pat_hours = (1/3) * mark_hours ∧ 
    pat_hours + kate_hours + mark_hours = 117 := by
  sorry

end NUMINAMATH_CALUDE_total_project_hours_l105_10569


namespace NUMINAMATH_CALUDE_sector_area_proof_l105_10589

/-- Given a circle where a central angle of 2 radians corresponds to an arc length of 2 cm,
    prove that the area of the sector formed by this central angle is 1 cm². -/
theorem sector_area_proof (r : ℝ) (θ : ℝ) (l : ℝ) (A : ℝ) : 
  θ = 2 → l = 2 → l = r * θ → A = (1/2) * r^2 * θ → A = 1 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_proof_l105_10589


namespace NUMINAMATH_CALUDE_triangular_array_sum_l105_10565

/-- Represents the sum of numbers in the nth row of the triangular array. -/
def f (n : ℕ) : ℕ :=
  4 * 2^(n-1) - 2*n

/-- The triangular array starts with 1 on top and increases by 1 for each subsequent outer number.
    Interior numbers are obtained by adding two adjacent numbers from the previous row. -/
theorem triangular_array_sum (n : ℕ) (h : n > 0) :
  f n = 4 * 2^(n-1) - 2*n :=
sorry

end NUMINAMATH_CALUDE_triangular_array_sum_l105_10565


namespace NUMINAMATH_CALUDE_problem_odometer_distance_l105_10574

/-- Represents an odometer that skips certain digits -/
structure SkippingOdometer :=
  (skipped_digits : List Nat)
  (displayed_value : Nat)

/-- Calculates the actual distance for a skipping odometer -/
def actualDistance (o : SkippingOdometer) : Nat :=
  sorry

/-- The specific odometer from the problem -/
def problemOdometer : SkippingOdometer :=
  { skipped_digits := [4, 7],
    displayed_value := 3008 }

theorem problem_odometer_distance :
  actualDistance problemOdometer = 1542 :=
sorry

end NUMINAMATH_CALUDE_problem_odometer_distance_l105_10574


namespace NUMINAMATH_CALUDE_distance_to_left_focus_l105_10501

/-- A hyperbola with real axis length m and a point P on it -/
structure Hyperbola (m : ℝ) where
  /-- The distance from P to the right focus is m -/
  dist_right_focus : ℝ
  /-- The distance from P to the right focus equals m -/
  dist_right_focus_eq : dist_right_focus = m

/-- The theorem stating that the distance from P to the left focus is 2m -/
theorem distance_to_left_focus (m : ℝ) (h : Hyperbola m) : 
  ∃ (dist_left_focus : ℝ), dist_left_focus = 2 * m := by
  sorry

end NUMINAMATH_CALUDE_distance_to_left_focus_l105_10501


namespace NUMINAMATH_CALUDE_expression_value_l105_10567

theorem expression_value (x y : ℝ) (h : x - 2*y + 2 = 0) :
  (2*y - x)^2 - 2*x + 4*y - 1 = 7 := by sorry

end NUMINAMATH_CALUDE_expression_value_l105_10567


namespace NUMINAMATH_CALUDE_some_number_equation_l105_10517

theorem some_number_equation : ∃ n : ℤ, (69842^2 - n^2) / (69842 - n) = 100000 ∧ n = 30158 := by
  sorry

end NUMINAMATH_CALUDE_some_number_equation_l105_10517


namespace NUMINAMATH_CALUDE_estimate_fish_population_l105_10549

/-- Estimate the total number of fish in a pond using the mark-recapture method. -/
theorem estimate_fish_population (marked_fish : ℕ) (second_catch : ℕ) (marked_in_second : ℕ) :
  marked_fish = 200 →
  second_catch = 100 →
  marked_in_second = 10 →
  (marked_fish * second_catch) / marked_in_second = 2000 :=
by
  sorry

#check estimate_fish_population

end NUMINAMATH_CALUDE_estimate_fish_population_l105_10549


namespace NUMINAMATH_CALUDE_solve_equation_for_m_l105_10510

theorem solve_equation_for_m : ∃ m : ℤ, 
  62519 * 9999^2 / 314 * (314 - m) = 547864 ∧ m = -547550 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_for_m_l105_10510


namespace NUMINAMATH_CALUDE_solution_set_equality_l105_10524

-- Define a decreasing function on ℝ
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- Define the set of x that satisfy the inequality
def SolutionSet (f : ℝ → ℝ) : Set ℝ :=
  {x | f (|1/x|) < f 1}

-- Theorem statement
theorem solution_set_equality (f : ℝ → ℝ) (h : DecreasingFunction f) :
  SolutionSet f = {x | -1 < x ∧ x < 0} ∪ {x | 0 < x ∧ x < 1} :=
by sorry

end NUMINAMATH_CALUDE_solution_set_equality_l105_10524


namespace NUMINAMATH_CALUDE_initial_weight_proof_l105_10529

theorem initial_weight_proof (W : ℝ) : 
  (W > 0) →
  (0.8 * (0.9 * W) = 36000) →
  (W = 50000) := by
sorry

end NUMINAMATH_CALUDE_initial_weight_proof_l105_10529


namespace NUMINAMATH_CALUDE_sqrt_difference_inequality_l105_10503

theorem sqrt_difference_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  Real.sqrt a - Real.sqrt b < Real.sqrt (a - b) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_inequality_l105_10503


namespace NUMINAMATH_CALUDE_triangle_median_theorem_l105_10596

/-- Triangle XYZ with given side lengths and median --/
structure Triangle where
  XY : ℝ
  XZ : ℝ
  XM : ℝ
  YZ : ℝ

/-- The theorem stating the relationship between sides and median in the given triangle --/
theorem triangle_median_theorem (t : Triangle) (h1 : t.XY = 6) (h2 : t.XZ = 9) (h3 : t.XM = 4) :
  t.YZ = Real.sqrt 170 := by
  sorry

#check triangle_median_theorem

end NUMINAMATH_CALUDE_triangle_median_theorem_l105_10596


namespace NUMINAMATH_CALUDE_range_of_a_l105_10540

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + x + (1/2 : ℝ) > 0) → a > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l105_10540


namespace NUMINAMATH_CALUDE_women_in_luxury_suite_l105_10595

def total_passengers : ℕ := 300
def women_percentage : ℚ := 1/2
def luxury_suite_percentage : ℚ := 3/20

theorem women_in_luxury_suite :
  ⌊(total_passengers : ℚ) * women_percentage * luxury_suite_percentage⌋ = 23 := by
  sorry

end NUMINAMATH_CALUDE_women_in_luxury_suite_l105_10595


namespace NUMINAMATH_CALUDE_distribute_8_3_non_empty_different_l105_10553

/-- The number of ways to distribute n different balls into k different boxes --/
def distribute (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to distribute n different balls into k different boxes,
    where each box contains at least one ball --/
def distributeNonEmpty (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to distribute n different balls into k different boxes,
    where each box contains at least one ball and the number of balls in each box is different --/
def distributeNonEmptyDifferent (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: The number of ways to distribute 8 different balls into 3 different boxes,
    where each box contains at least one ball and the number of balls in each box is different,
    is equal to 2688 --/
theorem distribute_8_3_non_empty_different :
  distributeNonEmptyDifferent 8 3 = 2688 := by sorry

end NUMINAMATH_CALUDE_distribute_8_3_non_empty_different_l105_10553


namespace NUMINAMATH_CALUDE_unique_six_digit_number_l105_10502

def is_valid_number (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000 ∧ (n / 100000 = 2)

def move_first_to_last (n : ℕ) : ℕ :=
  (n % 100000) * 10 + (n / 100000)

theorem unique_six_digit_number : 
  ∀ n : ℕ, is_valid_number n → (move_first_to_last n = 3 * n) → n = 285714 :=
by sorry

end NUMINAMATH_CALUDE_unique_six_digit_number_l105_10502


namespace NUMINAMATH_CALUDE_percentage_of_360_equals_144_l105_10572

theorem percentage_of_360_equals_144 : ∃ (p : ℚ), p * 360 = 144 ∧ p = 40 / 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_360_equals_144_l105_10572


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l105_10566

theorem solve_exponential_equation :
  ∀ x : ℝ, (16 : ℝ)^x * (16 : ℝ)^x * (16 : ℝ)^x * (16 : ℝ)^x = (256 : ℝ)^(10 : ℝ) → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l105_10566


namespace NUMINAMATH_CALUDE_vector_properties_l105_10539

/-- Given vectors a and b, prove that the projection of a onto b is equal to b,
    and that (a - b) is perpendicular to b. -/
theorem vector_properties (a b : ℝ × ℝ) 
    (ha : a = (2, 0)) (hb : b = (1, 1)) : 
    (((a • b) / (b • b)) • b = b) ∧ ((a - b) • b = 0) := by
  sorry

end NUMINAMATH_CALUDE_vector_properties_l105_10539


namespace NUMINAMATH_CALUDE_min_workers_for_painting_job_l105_10582

/-- Represents the painting job scenario -/
structure PaintingJob where
  totalDays : ℕ
  workedDays : ℕ
  initialWorkers : ℕ
  completedFraction : ℚ
  
/-- Calculates the minimum number of workers needed to complete the job on time -/
def minWorkersNeeded (job : PaintingJob) : ℕ :=
  sorry

/-- The theorem stating the minimum number of workers needed for the specific scenario -/
theorem min_workers_for_painting_job :
  let job := PaintingJob.mk 40 8 10 (2/5)
  minWorkersNeeded job = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_workers_for_painting_job_l105_10582


namespace NUMINAMATH_CALUDE_quadratic_sum_l105_10523

/-- A quadratic function f(x) = ax^2 + bx + c passing through (-2,0) and (4,0) with maximum value 54 -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_sum (a b c : ℝ) :
  QuadraticFunction a b c (-2) = 0 →
  QuadraticFunction a b c 4 = 0 →
  (∀ x, QuadraticFunction a b c x ≤ 54) →
  (∃ x, QuadraticFunction a b c x = 54) →
  a + b + c = 54 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l105_10523
