import Mathlib

namespace NUMINAMATH_CALUDE_cube_sum_minus_triple_product_l1846_184677

theorem cube_sum_minus_triple_product (x y z : ℝ) 
  (sum_eq : x + y + z = 13)
  (sum_products_eq : x*y + x*z + y*z = 32) :
  x^3 + y^3 + z^3 - 3*x*y*z = 949 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_minus_triple_product_l1846_184677


namespace NUMINAMATH_CALUDE_shaded_area_rectangle_triangle_l1846_184635

/-- Given a rectangle with width 8 and height 12, and a right triangle with base 6 and height 8,
    prove that the area of the shaded region formed by a segment connecting the top-left vertex
    of the rectangle to the farthest vertex of the triangle is 120 square units. -/
theorem shaded_area_rectangle_triangle (rectangle_width : ℝ) (rectangle_height : ℝ)
    (triangle_base : ℝ) (triangle_height : ℝ) :
  rectangle_width = 8 →
  rectangle_height = 12 →
  triangle_base = 6 →
  triangle_height = 8 →
  let rectangle_area := rectangle_width * rectangle_height
  let triangle_area := (1 / 2) * triangle_base * triangle_height
  let shaded_area := rectangle_area + triangle_area
  shaded_area = 120 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_rectangle_triangle_l1846_184635


namespace NUMINAMATH_CALUDE_investment_time_period_l1846_184633

/-- Proves that given a principal of 2000 invested at simple interest rates of 18% p.a. and 12% p.a.,
    if the difference in interest received is 240, then the time period of investment is 20 years. -/
theorem investment_time_period (principal : ℝ) (rate_high : ℝ) (rate_low : ℝ) (interest_diff : ℝ) :
  principal = 2000 →
  rate_high = 18 →
  rate_low = 12 →
  interest_diff = 240 →
  ∃ time : ℝ,
    principal * (rate_high / 100) * time - principal * (rate_low / 100) * time = interest_diff ∧
    time = 20 := by
  sorry

end NUMINAMATH_CALUDE_investment_time_period_l1846_184633


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l1846_184607

theorem solve_exponential_equation :
  ∃ n : ℕ, 8^n * 8^n * 8^n = 64^3 ∧ n = 2 := by sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l1846_184607


namespace NUMINAMATH_CALUDE_x1_value_l1846_184600

theorem x1_value (x1 x2 x3 : ℝ) 
  (h1 : 0 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1) 
  (h2 : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + x3^2 = 1/2) : 
  x1 = 2/3 := by
sorry

end NUMINAMATH_CALUDE_x1_value_l1846_184600


namespace NUMINAMATH_CALUDE_systematic_sample_theorem_l1846_184654

/-- Represents a systematic sampling scenario -/
structure SystematicSample where
  population : ℕ
  sample_size : ℕ
  interval_start : ℕ
  interval_end : ℕ
  (population_positive : 0 < population)
  (sample_size_positive : 0 < sample_size)
  (sample_size_le_population : sample_size ≤ population)
  (interval_valid : interval_start ≤ interval_end)
  (interval_in_population : interval_end ≤ population)

/-- Calculates the number of selected individuals in the given interval -/
def selected_in_interval (s : SystematicSample) : ℕ :=
  ((s.interval_end - s.interval_start + 1) * s.sample_size + s.population - 1) / s.population

/-- Theorem stating that for the given systematic sample, 11 individuals are selected from the interval -/
theorem systematic_sample_theorem (s : SystematicSample) 
  (h1 : s.population = 640)
  (h2 : s.sample_size = 32)
  (h3 : s.interval_start = 161)
  (h4 : s.interval_end = 380) : 
  selected_in_interval s = 11 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sample_theorem_l1846_184654


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1846_184691

theorem imaginary_part_of_z (z : ℂ) : z = (1 + 2*I) / ((1 - I)^2) → z.im = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1846_184691


namespace NUMINAMATH_CALUDE_sixth_fibonacci_is_eight_l1846_184620

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

theorem sixth_fibonacci_is_eight :
  ∃ x, (fibonacci 0 = 1) ∧ 
       (fibonacci 1 = 1) ∧ 
       (fibonacci 2 = 2) ∧ 
       (fibonacci 3 = 3) ∧ 
       (fibonacci 4 = 5) ∧ 
       (fibonacci 5 = x) ∧ 
       (fibonacci 6 = 13) ∧ 
       (x = 8) := by
  sorry

end NUMINAMATH_CALUDE_sixth_fibonacci_is_eight_l1846_184620


namespace NUMINAMATH_CALUDE_function_passes_through_point_l1846_184618

theorem function_passes_through_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x - 2)
  f 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l1846_184618


namespace NUMINAMATH_CALUDE_greatest_integer_difference_l1846_184697

theorem greatest_integer_difference (x y : ℤ) 
  (hx : 7 < x ∧ x < 9)
  (hy : 9 < y ∧ y < 15) :
  ∃ (d : ℤ), d = y - x ∧ d ≤ 6 ∧ ∀ (d' : ℤ), d' = y - x → d' ≤ d :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_difference_l1846_184697


namespace NUMINAMATH_CALUDE_c_share_is_27_l1846_184608

/-- Represents the rent share calculation for a pasture -/
structure PastureRent where
  a_oxen : ℕ
  a_months : ℕ
  b_oxen : ℕ
  b_months : ℕ
  c_oxen : ℕ
  c_months : ℕ
  total_rent : ℕ

/-- Calculates the share of rent for person C -/
def calculate_c_share (pr : PastureRent) : ℚ :=
  let total_ox_months := pr.a_oxen * pr.a_months + pr.b_oxen * pr.b_months + pr.c_oxen * pr.c_months
  let rent_per_ox_month := pr.total_rent / total_ox_months
  (pr.c_oxen * pr.c_months * rent_per_ox_month : ℚ)

/-- Theorem stating that C's share of rent is 27 Rs -/
theorem c_share_is_27 (pr : PastureRent) 
  (h1 : pr.a_oxen = 10) (h2 : pr.a_months = 7)
  (h3 : pr.b_oxen = 12) (h4 : pr.b_months = 5)
  (h5 : pr.c_oxen = 15) (h6 : pr.c_months = 3)
  (h7 : pr.total_rent = 105) : 
  calculate_c_share pr = 27 := by
  sorry


end NUMINAMATH_CALUDE_c_share_is_27_l1846_184608


namespace NUMINAMATH_CALUDE_jackson_sandwiches_l1846_184602

/-- The number of peanut butter and jelly sandwiches Jackson ate during the school year -/
def sandwiches_eaten (weeks : ℕ) (missed_wednesdays : ℕ) (missed_fridays : ℕ) : ℕ :=
  (weeks - missed_wednesdays) + (weeks - missed_fridays)

/-- Theorem stating that Jackson ate 69 sandwiches during the school year -/
theorem jackson_sandwiches : sandwiches_eaten 36 1 2 = 69 := by
  sorry

end NUMINAMATH_CALUDE_jackson_sandwiches_l1846_184602


namespace NUMINAMATH_CALUDE_twelfth_term_of_sequence_l1846_184605

/-- An arithmetic sequence is defined by its first term and common difference -/
def arithmeticSequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

theorem twelfth_term_of_sequence (a₁ a₂ a₃ : ℚ) (h₁ : a₁ = 1/2) (h₂ : a₂ = 5/6) (h₃ : a₃ = 7/6) :
  arithmeticSequence a₁ (a₂ - a₁) 12 = 25/6 := by
  sorry

end NUMINAMATH_CALUDE_twelfth_term_of_sequence_l1846_184605


namespace NUMINAMATH_CALUDE_pyramid_circumscribed_equivalence_l1846_184687

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a pyramid with n vertices -/
structure Pyramid (n : ℕ) where
  apex : Point3D
  base : Fin n → Point3D

/-- Predicate for the existence of a circumscribed sphere around a pyramid -/
def has_circumscribed_sphere (p : Pyramid n) : Prop := sorry

/-- Predicate for the existence of a circumscribed circle around the base of a pyramid -/
def has_circumscribed_circle_base (p : Pyramid n) : Prop := sorry

/-- Theorem stating the equivalence of circumscribed sphere and circle for a pyramid -/
theorem pyramid_circumscribed_equivalence (n : ℕ) (p : Pyramid n) :
  has_circumscribed_sphere p ↔ has_circumscribed_circle_base p := by sorry

end NUMINAMATH_CALUDE_pyramid_circumscribed_equivalence_l1846_184687


namespace NUMINAMATH_CALUDE_roberts_and_marias_ages_l1846_184628

theorem roberts_and_marias_ages (robert maria : ℕ) : 
  robert = maria + 8 →
  robert + 5 = 3 * (maria - 3) →
  robert + maria = 30 :=
by sorry

end NUMINAMATH_CALUDE_roberts_and_marias_ages_l1846_184628


namespace NUMINAMATH_CALUDE_division_problem_l1846_184636

theorem division_problem (x : ℝ) (h : (120 / x) - 15 = 5) : x = 6 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1846_184636


namespace NUMINAMATH_CALUDE_smallest_next_divisor_after_221_l1846_184659

theorem smallest_next_divisor_after_221 (n : ℕ) :
  (n ≥ 1000 ∧ n ≤ 9999) →  -- n is a 4-digit number
  Even n →                 -- n is even
  221 ∣ n →                -- 221 is a divisor of n
  ∃ (d : ℕ), d ∣ n ∧ d > 221 ∧ d ≤ 238 ∧ ∀ (x : ℕ), x ∣ n → x > 221 → x ≥ d :=
by sorry

end NUMINAMATH_CALUDE_smallest_next_divisor_after_221_l1846_184659


namespace NUMINAMATH_CALUDE_quadratic_inequality_always_negative_l1846_184675

theorem quadratic_inequality_always_negative :
  ∀ x : ℝ, -15 * x^2 + 4 * x - 6 < 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_always_negative_l1846_184675


namespace NUMINAMATH_CALUDE_function_properties_l1846_184648

def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) + f (x - y) = f x * f y

theorem function_properties (f : ℝ → ℝ) 
  (h1 : f 1 = 1) 
  (h2 : functional_equation f) : 
  (f 0 = 2) ∧ 
  (∀ x : ℝ, f x = f (-x)) ∧ 
  (∀ x : ℝ, f (x + 6) = f x) :=
sorry

end NUMINAMATH_CALUDE_function_properties_l1846_184648


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l1846_184674

theorem trigonometric_simplification (θ : ℝ) :
  (Real.sin (π - 2*θ) / (1 - Real.sin (π/2 + 2*θ))) * Real.tan (π + θ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l1846_184674


namespace NUMINAMATH_CALUDE_afternoon_emails_l1846_184640

theorem afternoon_emails (morning evening afternoon : ℕ) : 
  morning = 5 →
  morning = afternoon + 2 →
  afternoon = 7 := by sorry

end NUMINAMATH_CALUDE_afternoon_emails_l1846_184640


namespace NUMINAMATH_CALUDE_former_apartment_size_l1846_184625

-- Define the given constants
def former_rent_per_sqft : ℝ := 2
def new_apartment_rent : ℝ := 2800
def yearly_savings : ℝ := 1200

-- Define the theorem
theorem former_apartment_size :
  ∃ (size : ℝ),
    size * former_rent_per_sqft = new_apartment_rent / 2 + yearly_savings / 12 ∧
    size = 750 :=
by sorry

end NUMINAMATH_CALUDE_former_apartment_size_l1846_184625


namespace NUMINAMATH_CALUDE_specific_cyclic_quadrilateral_radii_l1846_184660

/-- A cyclic quadrilateral with given side lengths --/
structure CyclicQuadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  cyclic : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0

/-- The radius of the circumscribed circle of a cyclic quadrilateral --/
def circumradius (q : CyclicQuadrilateral) : ℝ := sorry

/-- The radius of the inscribed circle of a cyclic quadrilateral --/
def inradius (q : CyclicQuadrilateral) : ℝ := sorry

/-- Theorem about the radii of circumscribed and inscribed circles for a specific cyclic quadrilateral --/
theorem specific_cyclic_quadrilateral_radii :
  ∃ (q : CyclicQuadrilateral),
    q.a = 36 ∧ q.b = 91 ∧ q.c = 315 ∧ q.d = 260 ∧
    circumradius q = 162.5 ∧
    inradius q = 140 / 3 := by sorry

end NUMINAMATH_CALUDE_specific_cyclic_quadrilateral_radii_l1846_184660


namespace NUMINAMATH_CALUDE_gcd_triple_characterization_l1846_184652

theorem gcd_triple_characterization (a b c : ℕ+) :
  (Nat.gcd a.val 20 = b.val ∧ Nat.gcd b.val 15 = c.val ∧ Nat.gcd a.val c.val = 5) ↔
  (∃ t : ℕ+, (a = 20 * t ∧ b = 20 ∧ c = 5) ∨
             (a = 20 * t - 10 ∧ b = 10 ∧ c = 5) ∨
             (a = 10 * t - 5 ∧ b = 5 ∧ c = 5)) :=
by sorry


end NUMINAMATH_CALUDE_gcd_triple_characterization_l1846_184652


namespace NUMINAMATH_CALUDE_semicircle_to_cone_volume_l1846_184679

theorem semicircle_to_cone_volume (R : ℝ) (h : R > 0) :
  let semicircle_radius := R
  let cone_base_radius := R / 2
  let cone_height := (Real.sqrt 3 / 2) * R
  let cone_volume := (1 / 3) * Real.pi * cone_base_radius^2 * cone_height
  cone_volume = (Real.sqrt 3 / 24) * Real.pi * R^3 :=
by sorry

end NUMINAMATH_CALUDE_semicircle_to_cone_volume_l1846_184679


namespace NUMINAMATH_CALUDE_abs_neg_a_eq_three_l1846_184664

theorem abs_neg_a_eq_three (a : ℝ) : |(-a)| = 3 → a = 3 ∨ a = -3 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_a_eq_three_l1846_184664


namespace NUMINAMATH_CALUDE_solve_linear_equation_l1846_184678

theorem solve_linear_equation (x : ℝ) (h : 3 * x + 2 = 11) : 6 * x + 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l1846_184678


namespace NUMINAMATH_CALUDE_james_louise_ages_l1846_184699

/-- James and Louise's ages problem -/
theorem james_louise_ages (j l : ℝ) : 
  j = l + 9 →                   -- James is nine years older than Louise
  j + 8 = 3 * (l - 4) →         -- Eight years from now, James will be three times as old as Louise was four years ago
  j + l = 38 :=                 -- The sum of their current ages is 38
by
  sorry

end NUMINAMATH_CALUDE_james_louise_ages_l1846_184699


namespace NUMINAMATH_CALUDE_unused_sector_angle_l1846_184676

/-- Given a circular piece of cardboard with radius 18 cm, from which a sector is removed
    to form a cone with radius 15 cm and volume 1350π cubic centimeters,
    the measure of the angle of the unused sector is 60°. -/
theorem unused_sector_angle (r_cardboard : ℝ) (r_cone : ℝ) (v_cone : ℝ) :
  r_cardboard = 18 →
  r_cone = 15 →
  v_cone = 1350 * Real.pi →
  ∃ (angle : ℝ),
    angle = 60 ∧
    angle = 360 - (2 * r_cone * Real.pi) / (2 * r_cardboard * Real.pi) * 360 :=
by sorry

end NUMINAMATH_CALUDE_unused_sector_angle_l1846_184676


namespace NUMINAMATH_CALUDE_parallel_line_through_point_line_equation_proof_l1846_184661

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point (x, y) lies on a given line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- Check if two lines are parallel -/
def Line.parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem parallel_line_through_point (given_line : Line) (point : ℝ × ℝ) :
  ∃ (parallel_line : Line),
    parallel_line.contains point.1 point.2 ∧
    Line.parallel parallel_line given_line :=
by
  sorry

/-- The main theorem to prove -/
theorem line_equation_proof :
  let given_line : Line := { a := 1, b := -2, c := -2 }
  let point : ℝ × ℝ := (1, 1)
  let parallel_line : Line := { a := 1, b := -2, c := 1 }
  parallel_line.contains point.1 point.2 ∧
  Line.parallel parallel_line given_line :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_line_equation_proof_l1846_184661


namespace NUMINAMATH_CALUDE_parallelogram_height_l1846_184619

/-- A parallelogram with given area and base has a specific height -/
theorem parallelogram_height (area base height : ℝ) (h_area : area = 375) (h_base : base = 25) :
  area = base * height → height = 15 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_height_l1846_184619


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l1846_184646

theorem consecutive_integers_sum (n : ℤ) : 
  n * (n + 1) * (n + 2) = 504 → n + (n + 1) + (n + 2) = 24 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l1846_184646


namespace NUMINAMATH_CALUDE_book_pages_calculation_l1846_184690

/-- Represents the number of pages read in a book over a week -/
def BookPages : ℕ → ℕ → ℕ → ℕ → ℕ := λ d1 d2 d3 d4 =>
  d1 * 30 + d2 * 50 + d4

theorem book_pages_calculation :
  BookPages 2 4 1 70 = 330 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_calculation_l1846_184690


namespace NUMINAMATH_CALUDE_david_pushups_count_l1846_184630

/-- The number of push-ups Zachary did -/
def zachary_pushups : ℕ := 59

/-- The difference between David's and Zachary's push-ups -/
def david_extra_pushups : ℕ := 19

/-- The number of push-ups David did -/
def david_pushups : ℕ := zachary_pushups + david_extra_pushups

theorem david_pushups_count : david_pushups = 78 := by
  sorry

end NUMINAMATH_CALUDE_david_pushups_count_l1846_184630


namespace NUMINAMATH_CALUDE_commute_time_difference_l1846_184686

def commute_times (x y : ℝ) : List ℝ := [x, y, 8, 11, 9]

theorem commute_time_difference (x y : ℝ) :
  (List.sum (commute_times x y)) / 5 = 8 →
  (List.sum (List.map (λ t => (t - 8)^2) (commute_times x y))) / 5 = 4 →
  |x - y| = 2 := by
sorry

end NUMINAMATH_CALUDE_commute_time_difference_l1846_184686


namespace NUMINAMATH_CALUDE_number_solution_l1846_184634

theorem number_solution : ∃ x : ℝ, 2 * x - 2.6 * 4 = 10 ∧ x = 10.2 := by
  sorry

end NUMINAMATH_CALUDE_number_solution_l1846_184634


namespace NUMINAMATH_CALUDE_independence_day_absences_l1846_184682

theorem independence_day_absences (total_children : ℕ) 
  (h1 : total_children = 780)
  (present_children : ℕ)
  (absent_children : ℕ)
  (h2 : total_children = present_children + absent_children)
  (bananas_distributed : ℕ)
  (h3 : bananas_distributed = 4 * present_children)
  (h4 : bananas_distributed = 2 * total_children) :
  absent_children = 390 := by
sorry

end NUMINAMATH_CALUDE_independence_day_absences_l1846_184682


namespace NUMINAMATH_CALUDE_die_roll_probability_l1846_184685

def is_valid_roll (x : ℕ) : Prop := 1 ≤ x ∧ x ≤ 6

def angle_in_range (m n : ℕ) : Prop :=
  let a : ℝ × ℝ := (m, n)
  let b : ℝ × ℝ := (1, 0)
  let cos_alpha := (m : ℝ) / Real.sqrt ((m^2 : ℝ) + (n^2 : ℝ))
  Real.sqrt 2 / 2 < cos_alpha ∧ cos_alpha < 1

def count_favorable_outcomes : ℕ := 15

def total_outcomes : ℕ := 36

theorem die_roll_probability :
  (count_favorable_outcomes : ℚ) / total_outcomes = 5 / 12 :=
sorry

end NUMINAMATH_CALUDE_die_roll_probability_l1846_184685


namespace NUMINAMATH_CALUDE_proportional_function_decreasing_l1846_184670

theorem proportional_function_decreasing (x₁ x₂ : ℝ) (h : x₁ < x₂) : -2 * x₁ > -2 * x₂ := by
  sorry

end NUMINAMATH_CALUDE_proportional_function_decreasing_l1846_184670


namespace NUMINAMATH_CALUDE_lcm_36_150_l1846_184638

theorem lcm_36_150 : Nat.lcm 36 150 = 900 := by
  sorry

end NUMINAMATH_CALUDE_lcm_36_150_l1846_184638


namespace NUMINAMATH_CALUDE_range_of_a_l1846_184695

/-- Given functions f and g, prove the range of a -/
theorem range_of_a (a : ℝ) : 
  (∀ x₁ : ℝ, ∃ x₂ : ℝ, |2*x₁ - a| + |2*x₁ + 3| = |2*x₂ - 3| + 2) →
  (a ≥ -1 ∨ a ≤ -5) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1846_184695


namespace NUMINAMATH_CALUDE_square_area_ratio_l1846_184667

/-- Given three squares A, B, and C with side lengths x, 3x, and 2x respectively,
    prove that the ratio of the area of Square A to the combined area of Square B and Square C is 1/13 -/
theorem square_area_ratio (x : ℝ) (hx : x > 0) : 
  (x^2) / ((3*x)^2 + (2*x)^2) = 1 / 13 := by
sorry

end NUMINAMATH_CALUDE_square_area_ratio_l1846_184667


namespace NUMINAMATH_CALUDE_range_of_t_l1846_184604

-- Define a monotonically decreasing function
def MonoDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- State the theorem
theorem range_of_t (f : ℝ → ℝ) (h1 : MonoDecreasing f) :
  {t : ℝ | f (t^2) - f t < 0} = {t : ℝ | t < 0 ∨ t > 1} := by
  sorry

end NUMINAMATH_CALUDE_range_of_t_l1846_184604


namespace NUMINAMATH_CALUDE_expression_values_l1846_184696

theorem expression_values (m n : ℕ) (h : m * n ≠ 1) :
  let expr := (m^2 + m*n + n^2) / (m*n - 1)
  expr ∈ ({0, 4, 7} : Set ℕ) :=
sorry

end NUMINAMATH_CALUDE_expression_values_l1846_184696


namespace NUMINAMATH_CALUDE_sequence_properties_l1846_184669

def a : ℕ → ℕ
  | 0 => 1
  | n + 1 => (7 * a n + Nat.sqrt (45 * (a n)^2 - 36)) / 2

theorem sequence_properties :
  (∀ n : ℕ, a n > 0) ∧
  (∀ n : ℕ, ∃ k : ℕ, a n * a (n + 1) - 1 = k^2) := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l1846_184669


namespace NUMINAMATH_CALUDE_two_color_plane_division_l1846_184614

/-- A type representing a line in a plane. -/
structure Line where
  -- We don't need to specify the exact properties of a line for this problem

/-- A type representing a region in a plane. -/
structure Region where
  -- We don't need to specify the exact properties of a region for this problem

/-- A type representing a color (either red or blue). -/
inductive Color
  | Red
  | Blue

/-- A function that determines if two regions are adjacent. -/
def adjacent (r1 r2 : Region) : Prop :=
  sorry  -- The exact definition is not important for the statement

/-- A type representing a coloring of regions. -/
def Coloring := Region → Color

/-- A predicate that checks if a coloring is valid (no adjacent regions have the same color). -/
def valid_coloring (c : Coloring) : Prop :=
  ∀ r1 r2 : Region, adjacent r1 r2 → c r1 ≠ c r2

/-- The main theorem stating that for any set of lines dividing a plane,
    there exists a valid two-coloring of the resulting regions. -/
theorem two_color_plane_division (lines : Set Line) :
  ∃ c : Coloring, valid_coloring c :=
sorry

end NUMINAMATH_CALUDE_two_color_plane_division_l1846_184614


namespace NUMINAMATH_CALUDE_investment_ratio_l1846_184653

/-- Prove that given the conditions, the ratio of B's investment to C's investment is 2:3 -/
theorem investment_ratio (a_invest b_invest c_invest : ℚ) 
  (h1 : a_invest = 3 * b_invest)
  (h2 : ∃ f : ℚ, b_invest = f * c_invest)
  (h3 : b_invest / (a_invest + b_invest + c_invest) * 7700 = 1400) :
  b_invest / c_invest = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_investment_ratio_l1846_184653


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_abs_min_value_is_achievable_l1846_184689

theorem min_value_of_sum_of_abs (x y : ℝ) : 
  |x - 1| + |x| + |y - 1| + |y + 1| ≥ 3 :=
by sorry

theorem min_value_is_achievable : 
  ∃ (x y : ℝ), |x - 1| + |x| + |y - 1| + |y + 1| = 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_abs_min_value_is_achievable_l1846_184689


namespace NUMINAMATH_CALUDE_problem_solution_l1846_184681

-- Define the function f
def f (x : ℝ) : ℝ := 6 * x^2 + x - 1

-- State the theorem
theorem problem_solution (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : f (Real.sin α) = 0) :
  Real.sin α = 1 / 3 ∧
  (Real.tan (π + α) * Real.cos (-α)) / (Real.cos (π / 2 - α) * Real.sin (π - α)) = 3 ∧
  Real.sin (α + π / 6) = (Real.sqrt 3 + 2 * Real.sqrt 2) / 6 :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1846_184681


namespace NUMINAMATH_CALUDE_sochi_price_decrease_in_euros_l1846_184626

/-- Represents the price decrease in Sochi apartments in euros -/
def sochi_price_decrease_euros : ℝ := 32.5

/-- The price decrease of Moscow apartments in rubles -/
def moscow_price_decrease_rubles : ℝ := 20

/-- The price decrease of Moscow apartments in euros -/
def moscow_price_decrease_euros : ℝ := 40

/-- The price decrease of Sochi apartments in rubles -/
def sochi_price_decrease_rubles : ℝ := 10

theorem sochi_price_decrease_in_euros :
  let initial_price_rubles : ℝ := 100  -- Arbitrary initial price
  let initial_price_euros : ℝ := 100   -- Arbitrary initial price
  let moscow_new_price_rubles : ℝ := initial_price_rubles * (1 - moscow_price_decrease_rubles / 100)
  let moscow_new_price_euros : ℝ := initial_price_euros * (1 - moscow_price_decrease_euros / 100)
  let sochi_new_price_rubles : ℝ := initial_price_rubles * (1 - sochi_price_decrease_rubles / 100)
  let exchange_rate : ℝ := moscow_new_price_rubles / moscow_new_price_euros
  let sochi_new_price_euros : ℝ := sochi_new_price_rubles / exchange_rate
  (initial_price_euros - sochi_new_price_euros) / initial_price_euros * 100 = sochi_price_decrease_euros :=
by sorry

end NUMINAMATH_CALUDE_sochi_price_decrease_in_euros_l1846_184626


namespace NUMINAMATH_CALUDE_roof_tiles_needed_l1846_184688

def land_cost_per_sqm : ℕ := 50
def brick_cost_per_thousand : ℕ := 100
def roof_tile_cost : ℕ := 10
def land_area : ℕ := 2000
def brick_count : ℕ := 10000
def total_cost : ℕ := 106000

theorem roof_tiles_needed : ℕ := by
  -- The number of roof tiles needed is 500
  sorry

end NUMINAMATH_CALUDE_roof_tiles_needed_l1846_184688


namespace NUMINAMATH_CALUDE_function_values_unbounded_l1846_184663

/-- A function satisfying the given identity -/
def SatisfiesIdentity (f : ℤ × ℤ → ℤ) : Prop :=
  ∀ n m : ℤ, f (n, m) = (f (n - 1, m) + f (n + 1, m) + f (n, m - 1) + f (n, m + 1)) / 4

/-- The main theorem -/
theorem function_values_unbounded
  (f : ℤ × ℤ → ℤ)
  (h_satisfies : SatisfiesIdentity f)
  (h_nonconstant : ∃ (a b c d : ℤ), f (a, b) ≠ f (c, d)) :
  ∀ k : ℤ, (∃ n m : ℤ, f (n, m) > k) ∧ (∃ n m : ℤ, f (n, m) < k) :=
sorry

end NUMINAMATH_CALUDE_function_values_unbounded_l1846_184663


namespace NUMINAMATH_CALUDE_percentage_qualified_school_B_l1846_184627

/-- Percentage of students qualified from school A -/
def percentage_qualified_A : ℝ := 70

/-- Ratio of students appeared in school B compared to school A -/
def ratio_appeared_B_to_A : ℝ := 1.2

/-- Ratio of students qualified from school B compared to school A -/
def ratio_qualified_B_to_A : ℝ := 1.5

/-- Theorem: The percentage of students qualified from school B is 87.5% -/
theorem percentage_qualified_school_B :
  (ratio_qualified_B_to_A * percentage_qualified_A) / (ratio_appeared_B_to_A * 100) * 100 = 87.5 := by
  sorry

end NUMINAMATH_CALUDE_percentage_qualified_school_B_l1846_184627


namespace NUMINAMATH_CALUDE_ratio_of_squares_difference_l1846_184601

theorem ratio_of_squares_difference : 
  (1523^2 - 1517^2) / (1530^2 - 1510^2) = 3/10 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_squares_difference_l1846_184601


namespace NUMINAMATH_CALUDE_double_inequality_solution_l1846_184673

theorem double_inequality_solution (x : ℝ) : 
  -1 < (x^2 - 16*x + 24) / (x^2 - 4*x + 8) ∧ 
  (x^2 - 16*x + 24) / (x^2 - 4*x + 8) < 1 ↔ 
  (3/2 < x ∧ x < 4) ∨ (8 < x) := by
  sorry

end NUMINAMATH_CALUDE_double_inequality_solution_l1846_184673


namespace NUMINAMATH_CALUDE_james_brothers_cut_sixty_percent_fewer_trees_l1846_184632

/-- The percentage fewer trees James' brothers cut per day compared to James -/
def brothers_percentage_fewer (james_trees_per_day : ℕ) (total_days : ℕ) (james_solo_days : ℕ) (total_trees : ℕ) : ℚ :=
  let total_with_brothers := total_trees - james_trees_per_day * james_solo_days
  let daily_with_brothers := total_with_brothers / (total_days - james_solo_days)
  let brothers_trees_per_day := daily_with_brothers - james_trees_per_day
  (brothers_trees_per_day : ℚ) / james_trees_per_day * 100

theorem james_brothers_cut_sixty_percent_fewer_trees
  (h1 : brothers_percentage_fewer 20 5 2 196 = 60) :
  ∃ (james_trees_per_day : ℕ) (total_days : ℕ) (james_solo_days : ℕ) (total_trees : ℕ),
    james_trees_per_day = 20 ∧
    total_days = 5 ∧
    james_solo_days = 2 ∧
    total_trees = 196 ∧
    brothers_percentage_fewer james_trees_per_day total_days james_solo_days total_trees = 60 :=
by sorry

end NUMINAMATH_CALUDE_james_brothers_cut_sixty_percent_fewer_trees_l1846_184632


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l1846_184647

/-- The value of a for which the line x - y - 1 = 0 is tangent to the parabola y = ax² --/
theorem line_tangent_to_parabola :
  ∃! (a : ℝ), ∀ (x y : ℝ),
    (x - y - 1 = 0 ∧ y = a * x^2) →
    (∃! p : ℝ × ℝ, p.1 - p.2 - 1 = 0 ∧ p.2 = a * p.1^2) ∧
    a = 1/4 :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l1846_184647


namespace NUMINAMATH_CALUDE_largest_59_double_l1846_184643

/-- Converts a number from base 5 to base 10 -/
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a number from base 10 to base 9 -/
def base10ToBase9 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else aux (m / 9) ((m % 9) :: acc)
    aux n []

/-- Checks if a number is a 5-9 double -/
def is59Double (m : Nat) : Prop :=
  let base5Digits := base10ToBase9 m
  let base9Value := base5ToBase10 base5Digits
  base9Value = 2 * m

theorem largest_59_double :
  ∀ n : Nat, n > 20 → ¬(is59Double n) ∧ is59Double 20 :=
sorry

end NUMINAMATH_CALUDE_largest_59_double_l1846_184643


namespace NUMINAMATH_CALUDE_overlap_number_l1846_184612

theorem overlap_number (numbers : List ℝ) : 
  numbers.length = 9 ∧ 
  (numbers.take 5).sum / 5 = 7 ∧ 
  (numbers.drop 4).sum / 5 = 10 ∧ 
  numbers.sum / 9 = 74 / 9 → 
  ∃ x ∈ numbers, x = 11 ∧ x ∈ numbers.take 5 ∧ x ∈ numbers.drop 4 := by
sorry

end NUMINAMATH_CALUDE_overlap_number_l1846_184612


namespace NUMINAMATH_CALUDE_bacteria_population_growth_l1846_184609

def bacteria_count (n : ℕ) : ℕ :=
  if n % 2 = 0 then
    2^(n/2 + 1)
  else
    2^((n-1)/2 + 1)

theorem bacteria_population_growth (n : ℕ) :
  bacteria_count n = if n % 2 = 0 then 2^(n/2 + 1) else 2^((n-1)/2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_bacteria_population_growth_l1846_184609


namespace NUMINAMATH_CALUDE_four_distinct_roots_implies_c_magnitude_l1846_184639

/-- The polynomial Q(x) -/
def Q (c x : ℂ) : ℂ := (x^2 - 2*x + 3) * (x^2 - c*x + 6) * (x^2 - 4*x + 12)

/-- The theorem statement -/
theorem four_distinct_roots_implies_c_magnitude (c : ℂ) :
  (∃ (s : Finset ℂ), s.card = 4 ∧ (∀ x ∈ s, Q c x = 0) ∧
   (∀ x : ℂ, Q c x = 0 → x ∈ s)) →
  Complex.abs c = Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_four_distinct_roots_implies_c_magnitude_l1846_184639


namespace NUMINAMATH_CALUDE_sturgeons_caught_l1846_184624

/-- Given the total number of fishes caught and the number of pikes and herrings,
    prove that the number of sturgeons caught is 40. -/
theorem sturgeons_caught (total_fish : ℕ) (pikes : ℕ) (herrings : ℕ) 
    (h1 : total_fish = 145)
    (h2 : pikes = 30)
    (h3 : herrings = 75) :
    total_fish - (pikes + herrings) = 40 := by
  sorry

end NUMINAMATH_CALUDE_sturgeons_caught_l1846_184624


namespace NUMINAMATH_CALUDE_fraction_saved_is_one_third_l1846_184694

/-- Represents the fraction of take-home pay saved each month -/
def fraction_saved : ℝ := sorry

/-- Represents the monthly take-home pay -/
def monthly_pay : ℝ := sorry

/-- The total amount saved at the end of the year -/
def total_saved : ℝ := 12 * fraction_saved * monthly_pay

/-- The amount not saved in a month -/
def monthly_not_saved : ℝ := (1 - fraction_saved) * monthly_pay

/-- States that the total amount saved is 6 times the monthly amount not saved -/
axiom total_saved_eq_six_times_not_saved : total_saved = 6 * monthly_not_saved

/-- Theorem stating that the fraction saved each month is 1/3 -/
theorem fraction_saved_is_one_third : fraction_saved = 1/3 := by sorry

end NUMINAMATH_CALUDE_fraction_saved_is_one_third_l1846_184694


namespace NUMINAMATH_CALUDE_pentagonal_prism_lateral_angle_l1846_184610

/-- A pentagonal prism is a three-dimensional geometric shape with two congruent pentagonal bases 
    and five rectangular lateral faces. --/
structure PentagonalPrism where
  base : Pentagon
  height : ℝ
  height_pos : height > 0

/-- The angle φ is the angle between adjacent edges in a lateral face of the pentagonal prism. --/
def lateral_face_angle (prism : PentagonalPrism) : ℝ := sorry

/-- Theorem: In a pentagonal prism, the angle φ between adjacent edges in a lateral face must be 90°. --/
theorem pentagonal_prism_lateral_angle (prism : PentagonalPrism) : 
  lateral_face_angle prism = 90 := by sorry

end NUMINAMATH_CALUDE_pentagonal_prism_lateral_angle_l1846_184610


namespace NUMINAMATH_CALUDE_calculation_proof_l1846_184657

theorem calculation_proof : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |3034 - (1002 / 20.04) - 2983.95| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1846_184657


namespace NUMINAMATH_CALUDE_calculate_total_profit_total_profit_is_4600_l1846_184651

/-- Calculates the total profit given the investments, time periods, and Rajan's profit share -/
theorem calculate_total_profit (rajan_investment : ℕ) (rakesh_investment : ℕ) (mukesh_investment : ℕ)
  (rajan_months : ℕ) (rakesh_months : ℕ) (mukesh_months : ℕ) (rajan_profit : ℕ) : ℕ :=
  let rajan_share := rajan_investment * rajan_months
  let rakesh_share := rakesh_investment * rakesh_months
  let mukesh_share := mukesh_investment * mukesh_months
  let total_share := rajan_share + rakesh_share + mukesh_share
  let total_profit := (rajan_profit * total_share) / rajan_share
  total_profit

/-- Proves that the total profit is 4600 given the specific investments and Rajan's profit share -/
theorem total_profit_is_4600 :
  calculate_total_profit 20000 25000 15000 12 4 8 2400 = 4600 := by
  sorry

end NUMINAMATH_CALUDE_calculate_total_profit_total_profit_is_4600_l1846_184651


namespace NUMINAMATH_CALUDE_person_a_number_l1846_184645

theorem person_a_number : ∀ (A B : ℕ), 
  A < 10 → B < 10 →
  A + B = 8 →
  (10 * B + A) - (10 * A + B) = 18 →
  10 * A + B = 35 := by
sorry

end NUMINAMATH_CALUDE_person_a_number_l1846_184645


namespace NUMINAMATH_CALUDE_m_eq_one_necessary_not_sufficient_l1846_184683

/-- A complex number is pure imaginary if its real part is zero -/
def isPureImaginary (z : ℂ) : Prop := z.re = 0

theorem m_eq_one_necessary_not_sufficient :
  ∃ m : ℝ, isPureImaginary (m * (m - 1) + Complex.I) ∧ m ≠ 1 ∧
  ∀ m : ℝ, m = 1 → isPureImaginary (m * (m - 1) + Complex.I) :=
by sorry

end NUMINAMATH_CALUDE_m_eq_one_necessary_not_sufficient_l1846_184683


namespace NUMINAMATH_CALUDE_square_garden_area_l1846_184616

/-- Represents a square garden -/
structure SquareGarden where
  side : ℝ
  area : ℝ
  perimeter : ℝ

/-- Theorem: The area of a square garden is 90.25 square feet given the conditions -/
theorem square_garden_area
  (garden : SquareGarden)
  (h1 : garden.perimeter = 38)
  (h2 : garden.area = 2 * garden.perimeter + 14.25)
  : garden.area = 90.25 := by
  sorry

end NUMINAMATH_CALUDE_square_garden_area_l1846_184616


namespace NUMINAMATH_CALUDE_triangle_properties_l1846_184649

/-- An acute triangle with sides a, b, c -/
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  acute : a > 0 ∧ b > 0 ∧ c > 0
  cosine_law : b^2 = a^2 + c^2 - a*c

/-- The perimeter of a triangle -/
def perimeter (t : AcuteTriangle) : ℝ := t.a + t.b + t.c

/-- The area of a triangle -/
def area (t : AcuteTriangle) : ℝ := sorry

theorem triangle_properties (t : AcuteTriangle) :
  (∃ angleA : ℝ, angleA = 60 * (π / 180) ∧ t.c = 2 → t.a = 2) ∧
  (area t = 2 * Real.sqrt 3 →
    6 * Real.sqrt 2 ≤ perimeter t ∧ perimeter t < 6 + 2 * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l1846_184649


namespace NUMINAMATH_CALUDE_age_ratio_l1846_184665

/-- The ages of John, Mary, and Tonya satisfy certain conditions. -/
def AgeRelations (john mary tonya : ℕ) : Prop :=
  john = tonya / 2 ∧ tonya = 60 ∧ (john + mary + tonya) / 3 = 35

/-- The ratio of John's age to Mary's age is 2:1. -/
theorem age_ratio (john mary tonya : ℕ) 
  (h : AgeRelations john mary tonya) : john = 2 * mary := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_l1846_184665


namespace NUMINAMATH_CALUDE_polynomial_sum_l1846_184629

def f (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

theorem polynomial_sum (x : ℝ) : f x + g x + h x = -4 * x^2 + 12 * x - 12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_l1846_184629


namespace NUMINAMATH_CALUDE_f_satisfies_data_points_l1846_184662

/-- The function f that we want to prove satisfies the given data points -/
def f (x : ℤ) : ℤ := 2 * x^2 + 2 * x - 1

/-- The list of data points given in the table -/
def data_points : List (ℤ × ℤ) := [(1, 3), (2, 11), (3, 23), (4, 39), (5, 59)]

/-- Theorem stating that the function f satisfies all the given data points -/
theorem f_satisfies_data_points : ∀ (point : ℤ × ℤ), point ∈ data_points → f point.1 = point.2 := by
  sorry

#check f_satisfies_data_points

end NUMINAMATH_CALUDE_f_satisfies_data_points_l1846_184662


namespace NUMINAMATH_CALUDE_extremum_implies_a_equals_12_l1846_184603

/-- The function f(x) = a * ln(x) + x^2 - 10x has an extremum at x = 3 -/
def has_extremum_at_3 (a : ℝ) : Prop :=
  let f := fun (x : ℝ) => a * Real.log x + x^2 - 10*x
  ∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), 0 < |x - 3| ∧ |x - 3| < ε → 
    (f x - f 3) * (x - 3) ≤ 0

/-- Given that f(x) = a * ln(x) + x^2 - 10x has an extremum at x = 3, prove that a = 12 -/
theorem extremum_implies_a_equals_12 : 
  has_extremum_at_3 a → a = 12 := by sorry

end NUMINAMATH_CALUDE_extremum_implies_a_equals_12_l1846_184603


namespace NUMINAMATH_CALUDE_butterfly_collection_l1846_184642

theorem butterfly_collection (total : ℕ) (blue : ℕ) : 
  total = 19 → 
  blue = 6 → 
  ∃ (yellow : ℕ), blue = 2 * yellow → 
  ∃ (black : ℕ), black = total - (blue + yellow) ∧ black = 10 := by
  sorry

end NUMINAMATH_CALUDE_butterfly_collection_l1846_184642


namespace NUMINAMATH_CALUDE_ellipse_intersection_product_l1846_184622

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- Definition of the right focus F2 -/
def right_focus : ℝ × ℝ := (1, 0)

/-- Definition of the left vertex A -/
def left_vertex : ℝ × ℝ := (-2, 0)

/-- Definition of a line passing through a point -/
def line_through (m : ℝ) (p : ℝ × ℝ) (x y : ℝ) : Prop :=
  y - p.2 = m * (x - p.1)

/-- Definition of intersection of a line with x = 4 -/
def intersect_x_4 (m : ℝ) (p : ℝ × ℝ) : ℝ × ℝ :=
  (4, m * (4 - p.1) + p.2)

/-- Main theorem -/
theorem ellipse_intersection_product (l m n : ℝ) (P Q : ℝ × ℝ) :
  line_through l right_focus P.1 P.2 →
  line_through l right_focus Q.1 Q.2 →
  ellipse_C P.1 P.2 →
  ellipse_C Q.1 Q.2 →
  let M := intersect_x_4 m (left_vertex.1, left_vertex.2)
  let N := intersect_x_4 n (left_vertex.1, left_vertex.2)
  line_through m left_vertex P.1 P.2 →
  line_through n left_vertex Q.1 Q.2 →
  M.2 * N.2 = -9 :=
sorry

end NUMINAMATH_CALUDE_ellipse_intersection_product_l1846_184622


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_right_triangle_l1846_184641

/-- The radius of the inscribed circle in a right triangle with side lengths 3, 4, and 5 is 1 -/
theorem inscribed_circle_radius_right_triangle :
  let a : ℝ := 3
  let b : ℝ := 4
  let c : ℝ := 5
  let s : ℝ := (a + b + c) / 2
  let area : ℝ := (s * (s - a) * (s - b) * (s - c))^(1/2)
  a^2 + b^2 = c^2 → -- Pythagorean theorem to ensure it's a right triangle
  area / s = 1 := by
sorry


end NUMINAMATH_CALUDE_inscribed_circle_radius_right_triangle_l1846_184641


namespace NUMINAMATH_CALUDE_driver_net_pay_rate_l1846_184698

-- Define the parameters
def travel_time : ℝ := 3
def speed : ℝ := 50
def fuel_efficiency : ℝ := 25
def pay_rate : ℝ := 0.60
def gasoline_cost : ℝ := 2.50

-- Define the theorem
theorem driver_net_pay_rate :
  let total_distance := travel_time * speed
  let gasoline_used := total_distance / fuel_efficiency
  let gross_earnings := pay_rate * total_distance
  let gasoline_expense := gasoline_cost * gasoline_used
  let net_earnings := gross_earnings - gasoline_expense
  net_earnings / travel_time = 25 := by sorry

end NUMINAMATH_CALUDE_driver_net_pay_rate_l1846_184698


namespace NUMINAMATH_CALUDE_rattlesnake_tail_difference_l1846_184655

/-- The number of tail segments in an Eastern rattlesnake -/
def eastern_segments : ℕ := 6

/-- The number of tail segments in a Western rattlesnake -/
def western_segments : ℕ := 8

/-- The percentage difference in tail size between Eastern and Western rattlesnakes,
    expressed as a percentage of the Western rattlesnake's tail size -/
def percentage_difference : ℚ :=
  (western_segments - eastern_segments : ℚ) / western_segments * 100

/-- Theorem stating that the percentage difference in tail size between
    Eastern and Western rattlesnakes is 25% -/
theorem rattlesnake_tail_difference :
  percentage_difference = 25 := by sorry

end NUMINAMATH_CALUDE_rattlesnake_tail_difference_l1846_184655


namespace NUMINAMATH_CALUDE_largest_divisor_of_n_l1846_184671

theorem largest_divisor_of_n (n : ℕ) (hn : n > 0) (h_div : 360 ∣ n^3) :
  ∃ (w : ℕ), w = 30 ∧ w ∣ n ∧ ∀ (k : ℕ), k ∣ n → k ≤ w :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n_l1846_184671


namespace NUMINAMATH_CALUDE_greta_worked_40_hours_l1846_184658

/-- Greta's hourly rate in dollars -/
def greta_rate : ℝ := 12

/-- Lisa's hourly rate in dollars -/
def lisa_rate : ℝ := 15

/-- Number of hours Lisa would need to work to equal Greta's earnings -/
def lisa_hours : ℝ := 32

/-- Theorem stating that Greta worked 40 hours -/
theorem greta_worked_40_hours : 
  ∃ (greta_hours : ℝ), greta_hours * greta_rate = lisa_hours * lisa_rate ∧ greta_hours = 40 := by
  sorry

end NUMINAMATH_CALUDE_greta_worked_40_hours_l1846_184658


namespace NUMINAMATH_CALUDE_f_minus_five_eq_zero_l1846_184668

open Function

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f x = -f (-x)

theorem f_minus_five_eq_zero
  (f : ℝ → ℝ)
  (h1 : is_even (fun x ↦ f (1 - 2*x)))
  (h2 : is_odd (fun x ↦ f (x - 1))) :
  f (-5) = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_minus_five_eq_zero_l1846_184668


namespace NUMINAMATH_CALUDE_systematic_sampling_theorem_l1846_184611

/-- Represents a systematic sampling setup -/
structure SystematicSampling where
  total_students : ℕ
  num_groups : ℕ
  interval : ℕ
  first_number : ℕ

/-- Calculates the number drawn from a given group -/
def number_from_group (s : SystematicSampling) (group : ℕ) : ℕ :=
  s.first_number + s.interval * (group - 1)

theorem systematic_sampling_theorem (s : SystematicSampling) 
  (h1 : s.total_students = 160)
  (h2 : s.num_groups = 20)
  (h3 : s.interval = 8)
  (h4 : number_from_group s 16 = 123) :
  number_from_group s 2 = 11 := by
  sorry

#check systematic_sampling_theorem

end NUMINAMATH_CALUDE_systematic_sampling_theorem_l1846_184611


namespace NUMINAMATH_CALUDE_equation_solution_l1846_184621

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

/-- The fractional part function -/
noncomputable def frac (x : ℝ) : ℝ := x - floor x

/-- The set of solutions to the equation -/
def solution_set : Set ℝ := {29/12, 19/6, 97/24}

/-- The main theorem -/
theorem equation_solution :
  ∀ x : ℝ, (1 / (floor x : ℝ) + 1 / (floor (2*x) : ℝ) = frac x + 1/3) ↔ x ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1846_184621


namespace NUMINAMATH_CALUDE_shopkeeper_decks_l1846_184692

/-- Represents the number of face cards in a standard deck of playing cards. -/
def face_cards_per_deck : ℕ := 12

/-- Represents the total number of face cards the shopkeeper has. -/
def total_face_cards : ℕ := 60

/-- Calculates the number of complete decks given the total number of face cards. -/
def number_of_decks : ℕ := total_face_cards / face_cards_per_deck

theorem shopkeeper_decks : number_of_decks = 5 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_decks_l1846_184692


namespace NUMINAMATH_CALUDE_tiangong_altitude_scientific_notation_l1846_184672

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem tiangong_altitude_scientific_notation :
  toScientificNotation 375000 = ScientificNotation.mk 3.75 5 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_tiangong_altitude_scientific_notation_l1846_184672


namespace NUMINAMATH_CALUDE_rhombus_longer_diagonal_l1846_184623

/-- A rhombus with side length 34 units and shorter diagonal 32 units has a longer diagonal of 60 units. -/
theorem rhombus_longer_diagonal (side : ℝ) (shorter_diagonal : ℝ) (longer_diagonal : ℝ) : 
  side = 34 → shorter_diagonal = 32 → longer_diagonal = 60 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_longer_diagonal_l1846_184623


namespace NUMINAMATH_CALUDE_vector_magnitude_difference_l1846_184656

/-- Given two non-zero vectors in ℝ², if their sum is (-3, 6) and their difference is (-3, 2),
    then the difference of their squared magnitudes is 21. -/
theorem vector_magnitude_difference (a b : ℝ × ℝ) (ha : a ≠ (0, 0)) (hb : b ≠ (0, 0)) 
    (hsum : a.1 + b.1 = -3 ∧ a.2 + b.2 = 6) (hdiff : a.1 - b.1 = -3 ∧ a.2 - b.2 = 2) :
    a.1^2 + a.2^2 - (b.1^2 + b.2^2) = 21 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_difference_l1846_184656


namespace NUMINAMATH_CALUDE_wrong_operation_correction_l1846_184617

theorem wrong_operation_correction (x : ℕ) : 
  x - 46 = 27 → x * 46 = 3358 := by
  sorry

end NUMINAMATH_CALUDE_wrong_operation_correction_l1846_184617


namespace NUMINAMATH_CALUDE_condition_relations_l1846_184684

theorem condition_relations (A B C : Prop) 
  (h1 : B → A)  -- A is necessary for B
  (h2 : C → B)  -- C is sufficient for B
  (h3 : ¬(B → C))  -- C is not necessary for B
  : (C → A) ∧ ¬(A → C) := by sorry

end NUMINAMATH_CALUDE_condition_relations_l1846_184684


namespace NUMINAMATH_CALUDE_scientific_notation_exponent_l1846_184693

theorem scientific_notation_exponent (n : ℤ) :
  0.0000502 = 5.02 * (10 : ℝ) ^ n → n = -4 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_exponent_l1846_184693


namespace NUMINAMATH_CALUDE_hyperbola_property_l1846_184613

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 3 = 1

-- Define the foci
def left_focus : ℝ × ℝ := sorry
def right_focus : ℝ × ℝ := sorry

-- Define a line passing through the left focus
def line_through_left_focus (x y : ℝ) : Prop := sorry

-- Define the left branch of the hyperbola
def left_branch (x y : ℝ) : Prop := hyperbola x y ∧ x < 0

-- Define the intersection points
def point_M : ℝ × ℝ := sorry
def point_N : ℝ × ℝ := sorry

-- State the theorem
theorem hyperbola_property :
  hyperbola point_M.1 point_M.2 ∧
  hyperbola point_N.1 point_N.2 ∧
  left_branch point_M.1 point_M.2 ∧
  left_branch point_N.1 point_N.2 ∧
  line_through_left_focus point_M.1 point_M.2 ∧
  line_through_left_focus point_N.1 point_N.2
  →
  abs (dist point_M right_focus) + abs (dist point_N right_focus) - abs (dist point_M point_N) = 8 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_property_l1846_184613


namespace NUMINAMATH_CALUDE_imaginary_part_of_3_minus_2i_l1846_184606

theorem imaginary_part_of_3_minus_2i :
  Complex.im (3 - 2 * Complex.I) = -2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_3_minus_2i_l1846_184606


namespace NUMINAMATH_CALUDE_coffee_mixture_cost_l1846_184666

/-- Proves that the cost of the second coffee brand is $116.67 per kg -/
theorem coffee_mixture_cost (brand1_cost brand2_cost mixture_price profit_rate : ℝ)
  (h1 : brand1_cost = 200)
  (h2 : mixture_price = 177)
  (h3 : profit_rate = 0.18)
  (h4 : (2 * brand1_cost + 3 * brand2_cost) / 5 * (1 + profit_rate) = mixture_price) :
  brand2_cost = 116.67 := by
  sorry

end NUMINAMATH_CALUDE_coffee_mixture_cost_l1846_184666


namespace NUMINAMATH_CALUDE_percentage_to_pass_l1846_184615

/-- Calculates the percentage of total marks needed to pass an exam -/
theorem percentage_to_pass (marks_obtained : ℕ) (marks_short : ℕ) (total_marks : ℕ) :
  marks_obtained = 125 →
  marks_short = 40 →
  total_marks = 500 →
  (((marks_obtained + marks_short : ℚ) / total_marks) * 100 : ℚ) = 33 := by
  sorry

end NUMINAMATH_CALUDE_percentage_to_pass_l1846_184615


namespace NUMINAMATH_CALUDE_polygon_triangulation_l1846_184631

/-- A color type with three possible values -/
inductive Color
  | one
  | two
  | three

/-- A vertex of a polygon -/
structure Vertex where
  color : Color

/-- A convex polygon -/
structure ConvexPolygon where
  vertices : List Vertex
  convex : Bool
  all_colors_present : Bool
  no_adjacent_same_color : Bool

/-- A triangle with three vertices -/
structure Triangle where
  v1 : Vertex
  v2 : Vertex
  v3 : Vertex

/-- A triangulation of a polygon -/
structure Triangulation where
  triangles : List Triangle

/-- The main theorem -/
theorem polygon_triangulation (p : ConvexPolygon) :
  p.convex ∧ p.all_colors_present ∧ p.no_adjacent_same_color →
  ∃ (t : Triangulation), ∀ (triangle : Triangle), triangle ∈ t.triangles →
    triangle.v1.color ≠ triangle.v2.color ∧
    triangle.v2.color ≠ triangle.v3.color ∧
    triangle.v3.color ≠ triangle.v1.color :=
sorry

end NUMINAMATH_CALUDE_polygon_triangulation_l1846_184631


namespace NUMINAMATH_CALUDE_fraction_simplification_l1846_184637

theorem fraction_simplification (x : ℝ) : (x + 1) / 3 + (2 - 3 * x) / 2 = (8 - 7 * x) / 6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1846_184637


namespace NUMINAMATH_CALUDE_multiplication_equality_l1846_184650

theorem multiplication_equality : 469157 * 9999 = 4691116843 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_equality_l1846_184650


namespace NUMINAMATH_CALUDE_min_colors_correct_key_coloring_distinguishes_min_colors_optimal_l1846_184644

/-- The smallest number of colors needed to distinguish n keys arranged in a circle -/
def min_colors (n : ℕ) : ℕ :=
  if n ≤ 2 then n
  else if n ≤ 5 then 3
  else 2

/-- Theorem stating the minimum number of colors needed to distinguish n keys -/
theorem min_colors_correct (n : ℕ) :
  min_colors n = 
    if n ≤ 2 then n
    else if n ≤ 5 then 3
    else 2 :=
by
  sorry

/-- The coloring function that assigns colors to keys -/
def key_coloring (n : ℕ) : ℕ → Fin (min_colors n) :=
  sorry

/-- Theorem stating that the key_coloring function distinguishes all keys -/
theorem key_coloring_distinguishes (n : ℕ) :
  ∀ i j : Fin n, i ≠ j → 
    ∃ k : ℕ, (key_coloring n ((i + k) % n) ≠ key_coloring n ((j + k) % n)) ∨
            (key_coloring n ((n - i - k - 1) % n) ≠ key_coloring n ((n - j - k - 1) % n)) :=
by
  sorry

/-- Theorem stating that min_colors n is the smallest number that allows a distinguishing coloring -/
theorem min_colors_optimal (n : ℕ) :
  ∀ m : ℕ, m < min_colors n → 
    ¬∃ f : ℕ → Fin m, ∀ i j : Fin n, i ≠ j → 
      ∃ k : ℕ, (f ((i + k) % n) ≠ f ((j + k) % n)) ∨
              (f ((n - i - k - 1) % n) ≠ f ((n - j - k - 1) % n)) :=
by
  sorry

end NUMINAMATH_CALUDE_min_colors_correct_key_coloring_distinguishes_min_colors_optimal_l1846_184644


namespace NUMINAMATH_CALUDE_line_passes_through_point_line_has_slope_line_properties_l1846_184680

/-- A line in the xy-plane defined by the equation y = k(x+1) for some real k -/
structure Line where
  k : ℝ

/-- The point (-1, 0) in the xy-plane -/
def point : ℝ × ℝ := (-1, 0)

/-- Checks if a given point (x, y) lies on the line -/
def Line.contains (l : Line) (p : ℝ × ℝ) : Prop :=
  p.2 = l.k * (p.1 + 1)

/-- States that the line passes through the point (-1, 0) -/
theorem line_passes_through_point (l : Line) : l.contains point := by sorry

/-- States that the line has a defined slope -/
theorem line_has_slope (l : Line) : ∃ m : ℝ, ∀ x y : ℝ, y = m * x + l.k := by sorry

/-- Main theorem combining both properties -/
theorem line_properties (l : Line) : l.contains point ∧ ∃ m : ℝ, ∀ x y : ℝ, y = m * x + l.k := by sorry

end NUMINAMATH_CALUDE_line_passes_through_point_line_has_slope_line_properties_l1846_184680
