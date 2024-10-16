import Mathlib

namespace NUMINAMATH_CALUDE_texasCityGDP2009_scientific_notation_l175_17546

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation with a specified number of significant figures -/
def toScientificNotation (x : ℝ) (sigFigs : ℕ) : ScientificNotation :=
  sorry

/-- The GDP of Texas City in 2009 in billion yuan -/
def texasCityGDP2009 : ℝ := 1545.35

theorem texasCityGDP2009_scientific_notation :
  toScientificNotation (texasCityGDP2009 * 1000000000) 3 =
    ScientificNotation.mk 1.55 11 (by norm_num) :=
  sorry

end NUMINAMATH_CALUDE_texasCityGDP2009_scientific_notation_l175_17546


namespace NUMINAMATH_CALUDE_ellipse_circle_tangent_contained_l175_17531

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b
  h_a_ge_b : a ≥ b

/-- Represents a circle with center (h, k) and radius r -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ
  h_pos_r : 0 < r

/-- Check if a point (x, y) is on the ellipse -/
def Ellipse.contains (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- Check if a point (x, y) is on or inside the circle -/
def Circle.contains (c : Circle) (x y : ℝ) : Prop :=
  (x - c.h)^2 + (y - c.k)^2 ≤ c.r^2

/-- Check if the circle is tangent to the ellipse -/
def is_tangent (e : Ellipse) (c : Circle) : Prop :=
  ∃ x y : ℝ, e.contains x y ∧ (x - c.h)^2 + (y - c.k)^2 = c.r^2 ∧
    ∀ x' y' : ℝ, e.contains x' y' → (x' - c.h)^2 + (y' - c.k)^2 ≥ c.r^2

/-- Check if the circle is entirely contained within the ellipse -/
def is_contained (e : Ellipse) (c : Circle) : Prop :=
  ∀ x y : ℝ, c.contains x y → e.contains x y

/-- Main theorem: The circle with radius 2 centered at a focus of the ellipse
    is tangent to the ellipse and contained within it -/
theorem ellipse_circle_tangent_contained (e : Ellipse) (c : Circle)
    (h_e : e.a = 6 ∧ e.b = 5)
    (h_c : c.h = Real.sqrt 11 ∧ c.k = 0 ∧ c.r = 2) :
    is_tangent e c ∧ is_contained e c := by
  sorry

end NUMINAMATH_CALUDE_ellipse_circle_tangent_contained_l175_17531


namespace NUMINAMATH_CALUDE_M_intersect_N_equals_one_two_open_l175_17595

def M : Set ℝ := {x | ∃ y, y = Real.log (-x^2 - x + 6)}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

theorem M_intersect_N_equals_one_two_open :
  M ∩ N = {x | 1 ≤ x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_equals_one_two_open_l175_17595


namespace NUMINAMATH_CALUDE_data_mode_is_60_l175_17557

def data : List Nat := [65, 60, 75, 60, 80]

def mode (l : List Nat) : Nat :=
  l.foldl (λ acc x => if l.count x > l.count acc then x else acc) 0

theorem data_mode_is_60 : mode data = 60 := by
  sorry

end NUMINAMATH_CALUDE_data_mode_is_60_l175_17557


namespace NUMINAMATH_CALUDE_gcd_1151_3079_l175_17525

theorem gcd_1151_3079 : Nat.gcd 1151 3079 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1151_3079_l175_17525


namespace NUMINAMATH_CALUDE_prism_volume_is_six_times_pyramid_volume_l175_17533

/-- A regular quadrilateral prism with an inscribed pyramid -/
structure PrismWithPyramid where
  /-- Side length of the prism's base -/
  a : ℝ
  /-- Height of the prism -/
  h : ℝ
  /-- Volume of the inscribed pyramid -/
  V : ℝ
  /-- The inscribed pyramid has vertices at the center of the upper base
      and the midpoints of the sides of the lower base -/
  pyramid_vertices : Unit

/-- The volume of the prism is 6 times the volume of the inscribed pyramid -/
theorem prism_volume_is_six_times_pyramid_volume (p : PrismWithPyramid) :
  p.a^2 * p.h = 6 * p.V := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_is_six_times_pyramid_volume_l175_17533


namespace NUMINAMATH_CALUDE_aiden_sleep_fraction_l175_17580

/-- Proves that 15 minutes is equal to 1/4 of an hour, given that an hour has 60 minutes. -/
theorem aiden_sleep_fraction (minutes_in_hour : ℕ) (aiden_sleep_minutes : ℕ) : 
  minutes_in_hour = 60 → aiden_sleep_minutes = 15 → 
  (aiden_sleep_minutes : ℚ) / minutes_in_hour = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_aiden_sleep_fraction_l175_17580


namespace NUMINAMATH_CALUDE_intersection_has_two_elements_l175_17582

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p | p.2 = p.1^2}
def B : Set (ℝ × ℝ) := {p | p.2 = 1 - |p.1|}

-- State the theorem
theorem intersection_has_two_elements :
  ∃ (p₁ p₂ : ℝ × ℝ), p₁ ≠ p₂ ∧ A ∩ B = {p₁, p₂} :=
sorry

end NUMINAMATH_CALUDE_intersection_has_two_elements_l175_17582


namespace NUMINAMATH_CALUDE_mail_in_rebates_difference_l175_17554

/-- The number of additional mail-in rebates compared to bills --/
def additional_rebates : ℕ := 3

/-- The total number of stamps needed --/
def total_stamps : ℕ := 21

/-- The number of thank you cards --/
def thank_you_cards : ℕ := 3

/-- The number of bills --/
def bills : ℕ := 2

theorem mail_in_rebates_difference (rebates : ℕ) (job_applications : ℕ) :
  (thank_you_cards + bills + rebates + job_applications + 1 = total_stamps) →
  (job_applications = 2 * rebates) →
  (rebates = bills + additional_rebates) := by
  sorry

end NUMINAMATH_CALUDE_mail_in_rebates_difference_l175_17554


namespace NUMINAMATH_CALUDE_difference_between_sum_and_average_l175_17543

def numbers : List ℕ := [44, 16, 2, 77, 241]

theorem difference_between_sum_and_average : 
  (numbers.sum : ℚ) - (numbers.sum : ℚ) / numbers.length = 304 := by
  sorry

end NUMINAMATH_CALUDE_difference_between_sum_and_average_l175_17543


namespace NUMINAMATH_CALUDE_total_spent_is_211_20_l175_17587

/-- Calculates the total amount spent on a meal given the food price, sales tax rate, and tip rate. -/
def total_amount_spent (food_price : ℝ) (sales_tax_rate : ℝ) (tip_rate : ℝ) : ℝ :=
  let price_with_tax := food_price * (1 + sales_tax_rate)
  price_with_tax * (1 + tip_rate)

/-- Theorem stating that the total amount spent is $211.20 given the specified conditions. -/
theorem total_spent_is_211_20 :
  total_amount_spent 160 0.1 0.2 = 211.20 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_is_211_20_l175_17587


namespace NUMINAMATH_CALUDE_inequality_proof_l175_17515

theorem inequality_proof (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (sum : a + b + c = Real.sqrt 2) :
  1 / Real.sqrt (1 + a^2) + 1 / Real.sqrt (1 + b^2) + 1 / Real.sqrt (1 + c^2) ≥ 2 + 1 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l175_17515


namespace NUMINAMATH_CALUDE_function_range_l175_17564

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2*x - 4

-- Define the domain
def domain : Set ℝ := Set.Icc (-2) 2

-- Theorem statement
theorem function_range :
  Set.range (fun x => f x) = Set.Icc (-5) 4 := by sorry

end NUMINAMATH_CALUDE_function_range_l175_17564


namespace NUMINAMATH_CALUDE_custom_op_eight_twelve_l175_17517

/-- The custom operation @ for positive integers -/
def custom_op (a b : ℕ+) : ℚ :=
  (a * b : ℚ) / (a + b + 1 : ℚ)

/-- Theorem stating that 8 @ 12 = 96/21 -/
theorem custom_op_eight_twelve : custom_op 8 12 = 96 / 21 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_eight_twelve_l175_17517


namespace NUMINAMATH_CALUDE_distribute_7_4_l175_17503

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes -/
def distribute (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 7 indistinguishable balls into 4 distinguishable boxes is 128 -/
theorem distribute_7_4 : distribute 7 4 = 128 := by sorry

end NUMINAMATH_CALUDE_distribute_7_4_l175_17503


namespace NUMINAMATH_CALUDE_symmetry_sum_l175_17528

/-- Two points are symmetric with respect to the origin if the sum of their coordinates is (0, 0) -/
def symmetric_wrt_origin (p q : ℝ × ℝ) : Prop :=
  p.1 + q.1 = 0 ∧ p.2 + q.2 = 0

theorem symmetry_sum (a b : ℝ) :
  symmetric_wrt_origin (-2022, -1) (a, b) → a + b = 2023 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_sum_l175_17528


namespace NUMINAMATH_CALUDE_specific_trapezoid_area_l175_17591

/-- Represents a circumscribed isosceles trapezoid -/
structure CircumscribedIsoscelesTrapezoid where
  long_base : ℝ
  base_angle : ℝ

/-- Calculates the area of a circumscribed isosceles trapezoid -/
def area (t : CircumscribedIsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem stating that the area of the specific trapezoid is 84 -/
theorem specific_trapezoid_area :
  let t : CircumscribedIsoscelesTrapezoid := {
    long_base := 24,
    base_angle := Real.arcsin 0.6
  }
  area t = 84 := by sorry

end NUMINAMATH_CALUDE_specific_trapezoid_area_l175_17591


namespace NUMINAMATH_CALUDE_binomial_expansion_with_arithmetic_sequence_coefficients_l175_17581

/-- 
Given a binomial expansion (a+b)^n where the coefficients of the first three terms 
form an arithmetic sequence, this theorem proves that n = 8 and identifies 
the rational terms in the expansion when a = x and b = 1/2.
-/
theorem binomial_expansion_with_arithmetic_sequence_coefficients :
  ∀ n : ℕ,
  (∃ d : ℚ, (n.choose 1 : ℚ) = (n.choose 0 : ℚ) + d ∧ (n.choose 2 : ℚ) = (n.choose 1 : ℚ) + d) →
  (n = 8 ∧ 
   ∀ r : ℕ, r ≤ n → 
   (r = 0 ∨ r = 4 ∨ r = 8) ↔ ∃ q : ℚ, (n.choose r : ℚ) * (1 / 2 : ℚ)^r = q) :=
by sorry


end NUMINAMATH_CALUDE_binomial_expansion_with_arithmetic_sequence_coefficients_l175_17581


namespace NUMINAMATH_CALUDE_quadratic_non_real_roots_l175_17548

/-- A quadratic equation x^2 + bx + 16 has two non-real roots if and only if b is in the open interval (-8, 8) -/
theorem quadratic_non_real_roots (b : ℝ) : 
  (∀ x : ℂ, x^2 + b*x + 16 = 0 → x.im ≠ 0) ↔ -8 < b ∧ b < 8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_non_real_roots_l175_17548


namespace NUMINAMATH_CALUDE_spherical_coordinates_negated_z_l175_17597

/-- Given a point with rectangular coordinates (x, y, z) and spherical coordinates
    (5, 3π/4, π/3), prove that the spherical coordinates of (x, y, -z) are (5, 3π/4, 2π/3) -/
theorem spherical_coordinates_negated_z 
  (x y z : ℝ) 
  (h1 : x = 5 * Real.sin (π/3) * Real.cos (3*π/4))
  (h2 : y = 5 * Real.sin (π/3) * Real.sin (3*π/4))
  (h3 : z = 5 * Real.cos (π/3)) :
  ∃ (ρ θ φ : ℝ), 
    ρ = 5 ∧ 
    θ = 3*π/4 ∧ 
    φ = 2*π/3 ∧
    x = ρ * Real.sin φ * Real.cos θ ∧
    y = ρ * Real.sin φ * Real.sin θ ∧
    -z = ρ * Real.cos φ ∧
    ρ > 0 ∧ 
    0 ≤ θ ∧ θ < 2*π ∧
    0 ≤ φ ∧ φ ≤ π := by
  sorry

end NUMINAMATH_CALUDE_spherical_coordinates_negated_z_l175_17597


namespace NUMINAMATH_CALUDE_fixed_point_sum_l175_17500

/-- The function f(x) = a^(x-2) + 2 with a > 0 and a ≠ 1 has a fixed point (m, n) such that m + n = 5 -/
theorem fixed_point_sum (a : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  ∃ (m n : ℝ), (∀ x : ℝ, a^(x - 2) + 2 = a^(m - 2) + 2) ∧ m + n = 5 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_sum_l175_17500


namespace NUMINAMATH_CALUDE_arithmetic_sequence_nth_term_l175_17567

/-- Given an arithmetic sequence {a_n} with a₁ = 1, d = 2, and aₙ = 19, prove that n = 10 -/
theorem arithmetic_sequence_nth_term (a : ℕ → ℝ) (n : ℕ) : 
  (∀ k, a (k + 1) - a k = 2) →  -- common difference is 2
  a 1 = 1 →                     -- first term is 1
  a n = 19 →                    -- n-th term is 19
  n = 10 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_nth_term_l175_17567


namespace NUMINAMATH_CALUDE_sum_of_n_for_perfect_square_l175_17571

theorem sum_of_n_for_perfect_square : ∃ (S : Finset ℕ),
  (∀ n ∈ S, n < 2023 ∧ ∃ k : ℕ, 2 * n^2 + 3 * n = k^2) ∧
  (∀ n : ℕ, n < 2023 → (∃ k : ℕ, 2 * n^2 + 3 * n = k^2) → n ∈ S) ∧
  S.sum id = 444 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_n_for_perfect_square_l175_17571


namespace NUMINAMATH_CALUDE_special_operation_example_l175_17579

def special_operation (a b : ℝ) : ℝ := 2 * a + 5 * b

theorem special_operation_example : special_operation 4 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_special_operation_example_l175_17579


namespace NUMINAMATH_CALUDE_lila_sticker_count_l175_17592

/-- The number of stickers each person has -/
structure StickerCount where
  kristoff : ℕ
  riku : ℕ
  lila : ℕ

/-- The conditions of the problem -/
def sticker_problem (s : StickerCount) : Prop :=
  s.kristoff = 85 ∧
  s.riku = 25 * s.kristoff ∧
  s.lila = 2 * (s.kristoff + s.riku)

/-- The theorem to prove -/
theorem lila_sticker_count (s : StickerCount) :
  sticker_problem s → s.lila = 4420 :=
by
  sorry


end NUMINAMATH_CALUDE_lila_sticker_count_l175_17592


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_seven_satisfies_inequality_seven_is_smallest_l175_17594

theorem smallest_integer_satisfying_inequality :
  ∀ n : ℤ, n^2 - 15*n + 56 ≤ 0 → n ≥ 7 :=
by
  sorry

theorem seven_satisfies_inequality :
  7^2 - 15*7 + 56 ≤ 0 :=
by
  sorry

theorem seven_is_smallest :
  ∀ n : ℤ, n < 7 → n^2 - 15*n + 56 > 0 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_seven_satisfies_inequality_seven_is_smallest_l175_17594


namespace NUMINAMATH_CALUDE_ratio_sum_theorem_l175_17540

theorem ratio_sum_theorem (w x y : ℝ) 
  (h1 : w / x = 1 / 6)
  (h2 : w / y = 1 / 5) :
  (x + y) / y = 11 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_sum_theorem_l175_17540


namespace NUMINAMATH_CALUDE_sqrt_20_plus_sqrt_5_over_sqrt_5_minus_2_l175_17518

theorem sqrt_20_plus_sqrt_5_over_sqrt_5_minus_2 :
  (Real.sqrt 20 + Real.sqrt 5) / Real.sqrt 5 - 2 = 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_20_plus_sqrt_5_over_sqrt_5_minus_2_l175_17518


namespace NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_l175_17555

theorem sqrt_x_div_sqrt_y (x y : ℝ) (h : (1/3)^2 + (1/4)^2 + (1/6)^2 = ((1/5)^2 + (1/7)^2 + (1/8)^2) * (54*x)/(115*y)) : 
  Real.sqrt x / Real.sqrt y = 49/29 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_l175_17555


namespace NUMINAMATH_CALUDE_jungkook_english_score_l175_17574

/-- Jungkook's average score in Korean, math, and science -/
def initial_average : ℝ := 92

/-- The increase in average score after taking the English test -/
def average_increase : ℝ := 2

/-- The number of subjects before taking the English test -/
def initial_subjects : ℕ := 3

/-- The number of subjects after taking the English test -/
def total_subjects : ℕ := 4

/-- Jungkook's English score -/
def english_score : ℝ := total_subjects * (initial_average + average_increase) - initial_subjects * initial_average

theorem jungkook_english_score : english_score = 100 := by
  sorry

end NUMINAMATH_CALUDE_jungkook_english_score_l175_17574


namespace NUMINAMATH_CALUDE_fundamental_property_basis_l175_17584

/-- The fundamental property of fractions states that multiplying or dividing both the numerator
    and denominator of a fraction by the same non-zero number does not change the value of the fraction. -/
def fundamental_property_of_fractions : Prop :=
  ∀ (a b c : ℚ), c ≠ 0 → a / b = (a * c) / (b * c)

/-- Fraction simplification is the process of reducing a fraction to its lowest terms. -/
def fraction_simplification (a b : ℤ) : ℚ :=
  (a : ℚ) / b

/-- Finding a common denominator is the process of converting fractions to equivalent fractions
    with the same denominator. -/
def common_denominator (a b c d : ℤ) : Prop :=
  ∃ (k l : ℤ), (a : ℚ) / b = (a * k : ℚ) / (b * k) ∧
                (c : ℚ) / d = (c * l : ℚ) / (d * l) ∧
                b * k = d * l

/-- The fundamental property of fractions is the basis for fraction simplification
    and finding a common denominator. -/
theorem fundamental_property_basis :
  fundamental_property_of_fractions →
  (∀ a b : ℤ, ∃ k : ℤ, k ≠ 0 ∧ fraction_simplification a b = fraction_simplification (a * k) (b * k)) ∧
  (∀ a b c d : ℤ, common_denominator a b c d) :=
sorry

end NUMINAMATH_CALUDE_fundamental_property_basis_l175_17584


namespace NUMINAMATH_CALUDE_hyperbola_parameters_l175_17523

/-- Represents a hyperbola with equation x²/a² - y²/b² = 1 -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- Theorem: Given a hyperbola with specific asymptote and focus, determine its parameters -/
theorem hyperbola_parameters 
  (h : Hyperbola) 
  (h_asymptote : b / a = 2) 
  (h_focus : Real.sqrt (a^2 + b^2) = Real.sqrt 5) : 
  h.a = 1 ∧ h.b = 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_parameters_l175_17523


namespace NUMINAMATH_CALUDE_james_total_vegetables_l175_17521

/-- The total number of vegetables James ate -/
def total_vegetables (before_carrot before_cucumber after_carrot after_cucumber after_celery : ℕ) : ℕ :=
  before_carrot + before_cucumber + after_carrot + after_cucumber + after_celery

/-- Theorem stating that James ate 77 vegetables in total -/
theorem james_total_vegetables :
  total_vegetables 22 18 15 10 12 = 77 := by
  sorry

end NUMINAMATH_CALUDE_james_total_vegetables_l175_17521


namespace NUMINAMATH_CALUDE_root_in_interval_l175_17558

theorem root_in_interval :
  ∃ x₀ ∈ Set.Ioo (1/2 : ℝ) 1, Real.exp x₀ = 3 - 2 * x₀ := by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l175_17558


namespace NUMINAMATH_CALUDE_least_possible_bananas_l175_17513

/-- Represents the distribution of bananas among three monkeys. -/
structure BananaDistribution where
  b₁ : ℕ  -- bananas taken by first monkey
  b₂ : ℕ  -- bananas taken by second monkey
  b₃ : ℕ  -- bananas taken by third monkey

/-- Checks if the given distribution satisfies all conditions of the problem. -/
def isValidDistribution (d : BananaDistribution) : Prop :=
  let m₁ := (2 * d.b₁) / 3 + d.b₂ / 3 + (7 * d.b₃) / 16
  let m₂ := d.b₁ / 6 + d.b₂ / 3 + (7 * d.b₃) / 16
  let m₃ := d.b₁ / 6 + d.b₂ / 3 + d.b₃ / 8
  (∀ n : ℕ, n ∈ [m₁, m₂, m₃] → n > 0) ∧  -- whole number condition
  5 * m₂ = 3 * m₁ ∧ 5 * m₃ = 2 * m₁       -- ratio condition

/-- The theorem stating the least possible total number of bananas. -/
theorem least_possible_bananas :
  ∃ (d : BananaDistribution),
    isValidDistribution d ∧
    d.b₁ + d.b₂ + d.b₃ = 336 ∧
    (∀ d' : BananaDistribution, isValidDistribution d' → d'.b₁ + d'.b₂ + d'.b₃ ≥ 336) :=
  sorry

end NUMINAMATH_CALUDE_least_possible_bananas_l175_17513


namespace NUMINAMATH_CALUDE_toilet_paper_weeks_l175_17526

/-- The number of bathrooms in the bed and breakfast -/
def num_bathrooms : ℕ := 6

/-- The number of rolls Stella stocks per bathroom per day -/
def rolls_per_bathroom_per_day : ℕ := 1

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of rolls in a pack (1 dozen) -/
def rolls_per_pack : ℕ := 12

/-- The number of packs Stella buys -/
def packs_bought : ℕ := 14

/-- The number of weeks Stella bought toilet paper for -/
def weeks_bought : ℚ :=
  (packs_bought * rolls_per_pack) / (num_bathrooms * rolls_per_bathroom_per_day * days_per_week)

theorem toilet_paper_weeks : weeks_bought = 4 := by sorry

end NUMINAMATH_CALUDE_toilet_paper_weeks_l175_17526


namespace NUMINAMATH_CALUDE_unique_y_value_l175_17585

theorem unique_y_value : ∃! y : ℝ, y > 0 ∧ (y / 100) * y = 9 := by sorry

end NUMINAMATH_CALUDE_unique_y_value_l175_17585


namespace NUMINAMATH_CALUDE_blueberry_bonnie_ratio_l175_17510

/-- Represents the number of fruits eaten by each dog -/
structure DogFruits where
  apples : ℕ
  blueberries : ℕ
  bonnies : ℕ

/-- The problem setup -/
def fruitProblem (dogs : Vector DogFruits 3) : Prop :=
  let d1 := dogs.get 0
  let d2 := dogs.get 1
  let d3 := dogs.get 2
  d1.apples = 3 * d2.blueberries ∧
  d3.bonnies = 60 ∧
  d1.apples + d2.blueberries + d3.bonnies = 240

/-- The theorem to prove -/
theorem blueberry_bonnie_ratio (dogs : Vector DogFruits 3) 
  (h : fruitProblem dogs) : 
  (dogs.get 1).blueberries * 4 = (dogs.get 2).bonnies * 3 := by
  sorry


end NUMINAMATH_CALUDE_blueberry_bonnie_ratio_l175_17510


namespace NUMINAMATH_CALUDE_museum_paintings_ratio_l175_17573

theorem museum_paintings_ratio (total_paintings portraits : ℕ) 
  (h1 : total_paintings = 80)
  (h2 : portraits = 16)
  (h3 : ∃ k : ℕ, total_paintings - portraits = k * portraits) :
  (total_paintings - portraits) / portraits = 4 := by
  sorry

end NUMINAMATH_CALUDE_museum_paintings_ratio_l175_17573


namespace NUMINAMATH_CALUDE_squeak_interval_is_nine_seconds_l175_17586

/-- Represents a gear mechanism with two gears -/
structure GearMechanism where
  small_gear_teeth : ℕ
  large_gear_teeth : ℕ
  large_gear_revolution_time : ℝ

/-- Calculates the time interval between squeaks for a gear mechanism -/
def squeak_interval (gm : GearMechanism) : ℝ :=
  let lcm := Nat.lcm gm.small_gear_teeth gm.large_gear_teeth
  let large_gear_revolutions := lcm / gm.large_gear_teeth
  large_gear_revolutions * gm.large_gear_revolution_time

/-- Theorem stating that for the given gear mechanism, the squeak interval is 9 seconds -/
theorem squeak_interval_is_nine_seconds (gm : GearMechanism) 
  (h1 : gm.small_gear_teeth = 12) 
  (h2 : gm.large_gear_teeth = 32) 
  (h3 : gm.large_gear_revolution_time = 3) : 
  squeak_interval gm = 9 := by
  sorry

#eval squeak_interval { small_gear_teeth := 12, large_gear_teeth := 32, large_gear_revolution_time := 3 }

end NUMINAMATH_CALUDE_squeak_interval_is_nine_seconds_l175_17586


namespace NUMINAMATH_CALUDE_perfect_square_sequence_l175_17520

theorem perfect_square_sequence (a b : ℤ) 
  (h : ∀ n : ℕ, ∃ x : ℤ, 2^n * a + b = x^2) : 
  a = 0 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_sequence_l175_17520


namespace NUMINAMATH_CALUDE_mark_height_in_feet_l175_17576

/-- Represents a height in feet and inches -/
structure Height where
  feet : ℕ
  inches : ℕ
  h : inches < 12

/-- Converts a Height to total inches -/
def Height.toInches (h : Height) : ℕ := h.feet * 12 + h.inches

/-- The height difference between Mike and Mark in inches -/
def heightDifference : ℕ := 10

/-- Mike's height -/
def mikeHeight : Height := ⟨6, 1, by sorry⟩

/-- Mark's height in inches -/
def markHeightInches : ℕ := mikeHeight.toInches - heightDifference

theorem mark_height_in_feet :
  ∃ (h : Height), h.toInches = markHeightInches ∧ h.feet = 5 ∧ h.inches = 3 := by
  sorry

end NUMINAMATH_CALUDE_mark_height_in_feet_l175_17576


namespace NUMINAMATH_CALUDE_chang_e_3_descent_time_l175_17562

/-- Represents the descent phase of Chang'e 3 --/
structure DescentPhase where
  initial_altitude : ℝ  -- in kilometers
  final_altitude : ℝ    -- in meters
  playback_time_initial : ℕ  -- in seconds
  total_video_duration : ℕ   -- in seconds

/-- Calculates the time spent in the descent phase --/
def descent_time (d : DescentPhase) : ℕ :=
  114  -- The actual calculation is omitted and replaced with the known result

/-- Theorem stating that the descent time for the given conditions is 114 seconds --/
theorem chang_e_3_descent_time :
  let d : DescentPhase := {
    initial_altitude := 2.4,
    final_altitude := 100,
    playback_time_initial := 30 * 60 + 28,  -- 30 minutes and 28 seconds
    total_video_duration := 2 * 60 * 60 + 10 * 60 + 48  -- 2 hours, 10 minutes, and 48 seconds
  }
  descent_time d = 114 := by sorry

end NUMINAMATH_CALUDE_chang_e_3_descent_time_l175_17562


namespace NUMINAMATH_CALUDE_roots_of_quadratic_l175_17566

-- Define the variables and conditions
variable (x y : ℝ)
variable (h1 : 2 * x + 3 * y = 18)
variable (h2 : x * y = 8)

-- Define the quadratic polynomial
def f (t : ℝ) := t^2 - 18*t + 8

-- State the theorem
theorem roots_of_quadratic :
  f x = 0 ∧ f y = 0 := by sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_l175_17566


namespace NUMINAMATH_CALUDE_regression_correlation_zero_l175_17505

/-- Regression coefficient -/
def regression_coefficient (X Y : List ℝ) : ℝ := sorry

/-- Correlation coefficient -/
def correlation_coefficient (X Y : List ℝ) : ℝ := sorry

theorem regression_correlation_zero (X Y : List ℝ) :
  regression_coefficient X Y = 0 → correlation_coefficient X Y = 0 := by
  sorry

end NUMINAMATH_CALUDE_regression_correlation_zero_l175_17505


namespace NUMINAMATH_CALUDE_log_50_bounds_sum_l175_17524

theorem log_50_bounds_sum : ∃ c d : ℤ, (1 : ℝ) ≤ c ∧ (c : ℝ) < Real.log 50 / Real.log 10 ∧ Real.log 50 / Real.log 10 < d ∧ (d : ℝ) ≤ 2 ∧ c + d = 3 := by
  sorry

end NUMINAMATH_CALUDE_log_50_bounds_sum_l175_17524


namespace NUMINAMATH_CALUDE_vector_conclusions_l175_17572

-- Define the vector space
variable {V : Type*} [AddCommGroup V]

-- Define the vectors
variable (O D E M : V)

-- Define the given equation
axiom given_equation : D - O + (E - O) = M - O

-- Theorem to prove the three correct conclusions
theorem vector_conclusions :
  (M - O + (D - O) = E - O) ∧
  (M - O - (E - O) = D - O) ∧
  ((O - D) + (O - E) = O - M) := by
  sorry

end NUMINAMATH_CALUDE_vector_conclusions_l175_17572


namespace NUMINAMATH_CALUDE_square_area_ratio_l175_17516

theorem square_area_ratio (side_c side_d : ℝ) (h1 : side_c = 48) (h2 : side_d = 60) :
  (side_c^2) / (side_d^2) = 16 / 25 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l175_17516


namespace NUMINAMATH_CALUDE_square_inequality_l175_17534

theorem square_inequality (x : ℝ) : (x^2 + x + 1)^2 ≤ 3 * (x^4 + x^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_square_inequality_l175_17534


namespace NUMINAMATH_CALUDE_fraction_equivalence_l175_17593

theorem fraction_equivalence (n : ℝ) : (4 + n) / (7 + n) = 7 / 8 → n = 17 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l175_17593


namespace NUMINAMATH_CALUDE_must_divide_p_l175_17514

theorem must_divide_p (p q r s : ℕ+) 
  (h1 : Nat.gcd p q = 30)
  (h2 : Nat.gcd q r = 45)
  (h3 : Nat.gcd r s = 60)
  (h4 : 80 < Nat.gcd s p ∧ Nat.gcd s p < 120) :
  5 ∣ p := by
  sorry

end NUMINAMATH_CALUDE_must_divide_p_l175_17514


namespace NUMINAMATH_CALUDE_social_media_time_theorem_l175_17589

/-- Calculates the weekly time spent on social media given daily phone usage and social media ratio -/
def weekly_social_media_time (daily_phone_time : ℝ) (social_media_ratio : ℝ) : ℝ :=
  daily_phone_time * social_media_ratio * 7

/-- Theorem: Given 6 hours of daily phone usage and half spent on social media, 
    the weekly social media time is 21 hours -/
theorem social_media_time_theorem :
  weekly_social_media_time 6 0.5 = 21 := by
  sorry

end NUMINAMATH_CALUDE_social_media_time_theorem_l175_17589


namespace NUMINAMATH_CALUDE_hardcover_books_purchased_l175_17504

/-- The number of hardcover books purchased -/
def num_hardcover : ℕ := 8

/-- The number of paperback books purchased -/
def num_paperback : ℕ := 12 - num_hardcover

/-- The price of a paperback book -/
def price_paperback : ℕ := 18

/-- The price of a hardcover book -/
def price_hardcover : ℕ := 30

/-- The total amount spent -/
def total_spent : ℕ := 312

/-- Theorem stating that the number of hardcover books purchased is 8 -/
theorem hardcover_books_purchased :
  num_hardcover = 8 ∧
  num_hardcover + num_paperback = 12 ∧
  price_hardcover * num_hardcover + price_paperback * num_paperback = total_spent :=
by sorry

end NUMINAMATH_CALUDE_hardcover_books_purchased_l175_17504


namespace NUMINAMATH_CALUDE_hyperbola_focal_distance_l175_17519

/-- Given a hyperbola with equation x²/64 - y²/36 = 1 and foci F₁ and F₂,
    if P is a point on the hyperbola and |PF₁| = 17, then |PF₂| = 33 -/
theorem hyperbola_focal_distance (P F₁ F₂ : ℝ × ℝ) :
  (∃ x y : ℝ, P = (x, y) ∧ x^2/64 - y^2/36 = 1) →  -- P is on the hyperbola
  (∃ c : ℝ, c > 0 ∧ F₁ = (c, 0) ∧ F₂ = (-c, 0)) →  -- F₁ and F₂ are foci
  abs (P.1 - F₁.1) + abs (P.1 - F₁.2) = 17 →       -- |PF₁| = 17
  abs (P.1 - F₂.1) + abs (P.1 - F₂.2) = 33 :=      -- |PF₂| = 33
by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_distance_l175_17519


namespace NUMINAMATH_CALUDE_min_value_z_l175_17545

theorem min_value_z (x y : ℝ) : x^2 + 2*y^2 + 6*x - 4*y + 22 ≥ 11 := by
  sorry

end NUMINAMATH_CALUDE_min_value_z_l175_17545


namespace NUMINAMATH_CALUDE_partner_c_investment_l175_17536

/-- Represents the investment and profit structure of a business partnership --/
structure BusinessPartnership where
  capital_a : ℝ
  capital_b : ℝ
  capital_c : ℝ
  profit_b : ℝ
  profit_diff_ac : ℝ

/-- Theorem stating that given the conditions of the business partnership,
    the investment of partner c is 40000 --/
theorem partner_c_investment (bp : BusinessPartnership)
  (h1 : bp.capital_a = 8000)
  (h2 : bp.capital_b = 10000)
  (h3 : bp.profit_b = 3500)
  (h4 : bp.profit_diff_ac = 1399.9999999999998)
  : bp.capital_c = 40000 := by
  sorry


end NUMINAMATH_CALUDE_partner_c_investment_l175_17536


namespace NUMINAMATH_CALUDE_smallest_repeating_block_of_nine_elevenths_l175_17529

/-- The number of digits in the smallest repeating block of the decimal expansion of 9/11 -/
def smallest_repeating_block_length : ℕ :=
  2

/-- The fraction we're considering -/
def fraction : ℚ :=
  9 / 11

theorem smallest_repeating_block_of_nine_elevenths :
  smallest_repeating_block_length = 2 ∧
  ∃ (a b : ℕ) (k : ℕ+), fraction = (a * 10^smallest_repeating_block_length + b) / (10^smallest_repeating_block_length - 1) / k :=
by sorry

end NUMINAMATH_CALUDE_smallest_repeating_block_of_nine_elevenths_l175_17529


namespace NUMINAMATH_CALUDE_solution_set_equality_l175_17583

theorem solution_set_equality : 
  {x : ℝ | -x^2 + 3*x - 2 ≥ 0} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equality_l175_17583


namespace NUMINAMATH_CALUDE_digit_sum_inequality_l175_17508

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- Condition for the existence of c_k -/
def HasValidCk (k : ℕ) : Prop :=
  ∃ (c_k : ℝ), c_k > 0 ∧ ∀ (n : ℕ), n > 0 → S (k * n) ≥ c_k * S n

/-- k has no prime divisors other than 2 or 5 -/
def HasOnly2And5Factors (k : ℕ) : Prop :=
  ∀ (p : ℕ), p.Prime → p ∣ k → p = 2 ∨ p = 5

/-- Main theorem -/
theorem digit_sum_inequality (k : ℕ) (h : k > 1) :
  HasValidCk k ↔ HasOnly2And5Factors k := by sorry

end NUMINAMATH_CALUDE_digit_sum_inequality_l175_17508


namespace NUMINAMATH_CALUDE_tetrahedron_subdivision_existence_l175_17537

theorem tetrahedron_subdivision_existence :
  ∃ (n : ℕ), (1 / 2 : ℝ) ^ n < (1 / 100 : ℝ) := by sorry

end NUMINAMATH_CALUDE_tetrahedron_subdivision_existence_l175_17537


namespace NUMINAMATH_CALUDE_subsets_containing_five_and_seven_l175_17547

def S : Finset Nat := {1, 2, 3, 4, 5, 6, 7}

theorem subsets_containing_five_and_seven :
  (Finset.filter (fun s => 5 ∈ s ∧ 7 ∈ s) (Finset.powerset S)).card = 32 := by
  sorry

end NUMINAMATH_CALUDE_subsets_containing_five_and_seven_l175_17547


namespace NUMINAMATH_CALUDE_total_barking_dogs_l175_17538

theorem total_barking_dogs (initial_dogs : ℕ) (additional_dogs : ℕ) :
  initial_dogs = 30 → additional_dogs = 10 → initial_dogs + additional_dogs = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_barking_dogs_l175_17538


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l175_17530

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + a*x + 1 ≥ 0) ↔ a ∈ Set.Icc (-2) 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l175_17530


namespace NUMINAMATH_CALUDE_restaurant_additional_hamburgers_l175_17569

/-- The number of additional hamburgers made by a restaurant -/
def additional_hamburgers (initial : ℝ) (final : ℝ) : ℝ :=
  final - initial

/-- Proof that the restaurant made 3 additional hamburgers -/
theorem restaurant_additional_hamburgers : 
  let initial_hamburgers : ℝ := 9.0
  let final_hamburgers : ℝ := 12.0
  additional_hamburgers initial_hamburgers final_hamburgers = 3 := by
sorry

end NUMINAMATH_CALUDE_restaurant_additional_hamburgers_l175_17569


namespace NUMINAMATH_CALUDE_function_decomposition_l175_17588

theorem function_decomposition (f : ℝ → ℝ) :
  ∃ (f_even f_odd : ℝ → ℝ),
    (∀ x, f x = f_even x + f_odd x) ∧
    (∀ x, f_even (-x) = f_even x) ∧
    (∀ x, f_odd (-x) = -f_odd x) :=
by
  sorry

end NUMINAMATH_CALUDE_function_decomposition_l175_17588


namespace NUMINAMATH_CALUDE_largest_valid_n_l175_17550

def engineers : Nat := 6
def technicians : Nat := 12
def workers : Nat := 18

def total_individuals : Nat := engineers + technicians + workers

def is_valid_n (n : Nat) : Prop :=
  n ∣ total_individuals ∧
  n ≤ Nat.lcm (Nat.lcm engineers technicians) workers ∧
  ¬((n + 1) ∣ total_individuals)

theorem largest_valid_n :
  ∃ (n : Nat), is_valid_n n ∧ ∀ (m : Nat), is_valid_n m → m ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_largest_valid_n_l175_17550


namespace NUMINAMATH_CALUDE_number_relationship_l175_17556

theorem number_relationship (a b c d : ℝ) : 
  a = Real.log (3/2) / Real.log (2/3) →
  b = Real.log 2 / Real.log 3 →
  c = 2 ^ (1/3 : ℝ) →
  d = 3 ^ (1/2 : ℝ) →
  a < b ∧ b < c ∧ c < d := by
  sorry

end NUMINAMATH_CALUDE_number_relationship_l175_17556


namespace NUMINAMATH_CALUDE_decagon_diagonal_intersection_probability_l175_17535

/-- A regular decagon is a 10-sided polygon with all sides and angles equal -/
def RegularDecagon : Type := Unit

/-- The number of vertices in a regular decagon -/
def num_vertices : ℕ := 10

/-- The number of diagonals in a regular decagon -/
def num_diagonals : ℕ := 35

/-- The number of ways to choose 2 diagonals from the total number of diagonals -/
def num_diagonal_pairs : ℕ := 595

/-- The number of ways to choose 4 vertices from the decagon that form a convex quadrilateral -/
def num_convex_quadrilaterals : ℕ := 210

/-- The probability that two randomly chosen diagonals in a regular decagon
    intersect inside the decagon and form a convex quadrilateral -/
theorem decagon_diagonal_intersection_probability (d : RegularDecagon) :
  (num_convex_quadrilaterals : ℚ) / num_diagonal_pairs = 210 / 595 := by sorry

end NUMINAMATH_CALUDE_decagon_diagonal_intersection_probability_l175_17535


namespace NUMINAMATH_CALUDE_tangent_parallel_points_l175_17507

def f (x : ℝ) : ℝ := x^3 + x - 2

theorem tangent_parallel_points :
  ∀ x y : ℝ, (f x = y ∧ (3 * x^2 + 1 = 4)) ↔ ((x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = -4)) := by
  sorry

end NUMINAMATH_CALUDE_tangent_parallel_points_l175_17507


namespace NUMINAMATH_CALUDE_equivalent_operation_l175_17598

theorem equivalent_operation (x : ℝ) : 
  (x * (2/5)) / (3/7) = x * (14/15) := by
  sorry

end NUMINAMATH_CALUDE_equivalent_operation_l175_17598


namespace NUMINAMATH_CALUDE_hexagon_exterior_angles_sum_l175_17596

-- Define a polygon as a type
class Polygon (P : Type)

-- Define a hexagon as a specific type of polygon
class Hexagon (H : Type) extends Polygon H

-- Define the sum of exterior angles for a polygon
def sum_of_exterior_angles (P : Type) [Polygon P] : ℝ := 360

-- Theorem statement
theorem hexagon_exterior_angles_sum (H : Type) [Hexagon H] :
  sum_of_exterior_angles H = 360 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_exterior_angles_sum_l175_17596


namespace NUMINAMATH_CALUDE_acute_angles_subset_first_quadrant_l175_17522

-- Define the sets M, N, and P
def M : Set ℝ := {θ | 0 < θ ∧ θ < 90}
def N : Set ℝ := {θ | θ < 90}
def P : Set ℝ := {θ | ∃ k : ℤ, k * 360 < θ ∧ θ < k * 360 + 90}

-- Theorem to prove
theorem acute_angles_subset_first_quadrant : M ⊆ P := by
  sorry

end NUMINAMATH_CALUDE_acute_angles_subset_first_quadrant_l175_17522


namespace NUMINAMATH_CALUDE_f_t_plus_one_l175_17599

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x - 1

-- State the theorem
theorem f_t_plus_one (t : ℝ) : f (t + 1) = 3 * t + 2 := by
  sorry

end NUMINAMATH_CALUDE_f_t_plus_one_l175_17599


namespace NUMINAMATH_CALUDE_regular_octagon_diagonal_ratio_regular_octagon_diagonal_ratio_proof_l175_17549

/-- The ratio of the shorter diagonal to the longer diagonal in a regular octagon -/
theorem regular_octagon_diagonal_ratio : ℝ :=
  1 / Real.sqrt 2

/-- Proof that the ratio of the shorter diagonal to the longer diagonal in a regular octagon is 1 / √2 -/
theorem regular_octagon_diagonal_ratio_proof :
  regular_octagon_diagonal_ratio = 1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_regular_octagon_diagonal_ratio_regular_octagon_diagonal_ratio_proof_l175_17549


namespace NUMINAMATH_CALUDE_new_person_weight_l175_17542

theorem new_person_weight (initial_count : ℕ) (replaced_weight : ℝ) (avg_increase : ℝ) :
  initial_count = 4 →
  replaced_weight = 70 →
  avg_increase = 3 →
  (initial_count : ℝ) * avg_increase + replaced_weight = 82 := by
  sorry

#check new_person_weight

end NUMINAMATH_CALUDE_new_person_weight_l175_17542


namespace NUMINAMATH_CALUDE_largest_n_satisfying_inequality_l175_17563

theorem largest_n_satisfying_inequality : ∃ (n : ℕ),
  (∃ (x : Fin n → ℝ), (∀ (i j : Fin n), i < j → 
    (1 + x i * x j)^2 ≤ 0.99 * (1 + (x i)^2) * (1 + (x j)^2))) ∧
  (∀ (m : ℕ), m > n → 
    ¬∃ (y : Fin m → ℝ), ∀ (i j : Fin m), i < j → 
      (1 + y i * y j)^2 ≤ 0.99 * (1 + (y i)^2) * (1 + (y j)^2)) ∧
  n = 31 :=
sorry

end NUMINAMATH_CALUDE_largest_n_satisfying_inequality_l175_17563


namespace NUMINAMATH_CALUDE_marble_comparison_l175_17539

theorem marble_comparison (katrina mabel amanda carlos diana : ℕ) : 
  mabel = 5 * katrina →
  amanda + 12 = 2 * katrina →
  carlos = 3 * katrina →
  diana = 2 * katrina + (katrina / 2) →
  mabel = 85 →
  mabel = amanda + carlos + diana - 30 :=
by
  sorry

end NUMINAMATH_CALUDE_marble_comparison_l175_17539


namespace NUMINAMATH_CALUDE_solve_equation_l175_17568

theorem solve_equation (z : ℝ) :
  ∃ (n : ℝ), 14 * (-1 + z) + 18 = -14 * (1 - z) - n :=
by
  use -4
  sorry

#check solve_equation

end NUMINAMATH_CALUDE_solve_equation_l175_17568


namespace NUMINAMATH_CALUDE_binomial_coefficient_two_l175_17559

theorem binomial_coefficient_two (n : ℕ) (h : n ≥ 2) : Nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_two_l175_17559


namespace NUMINAMATH_CALUDE_trash_cans_redistribution_l175_17553

/-- The number of trash cans in Veteran's Park after the redistribution -/
def final_trash_cans_veterans_park (initial_veterans_park : ℕ) : ℕ :=
  let initial_central_park := initial_veterans_park / 2 + 8
  let moved_cans := initial_central_park / 2
  initial_veterans_park + moved_cans

/-- Theorem stating that given 24 initial trash cans in Veteran's Park, 
    the final number of trash cans in Veteran's Park is 34 -/
theorem trash_cans_redistribution :
  final_trash_cans_veterans_park 24 = 34 := by
  sorry

end NUMINAMATH_CALUDE_trash_cans_redistribution_l175_17553


namespace NUMINAMATH_CALUDE_sin_150_degrees_l175_17565

theorem sin_150_degrees : Real.sin (150 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_150_degrees_l175_17565


namespace NUMINAMATH_CALUDE_parallel_and_perpendicular_lines_l175_17502

-- Define a line in a plane
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

-- Define a point in a plane
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define the concept of a point not being on a line
def PointNotOnLine (p : Point) (l : Line) : Prop :=
  p.y ≠ l.slope * p.x + l.intercept

-- Define the concept of parallel lines
def Parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope ∧ l1.intercept ≠ l2.intercept

-- Define the concept of perpendicular lines
def Perpendicular (l1 l2 : Line) : Prop :=
  l1.slope * l2.slope = -1

theorem parallel_and_perpendicular_lines
  (L : Line) (P : Point) (h : PointNotOnLine P L) :
  (∃! l : Line, Parallel l L ∧ l.slope * P.x + l.intercept = P.y) ∧
  (∃ f : ℝ → Line, ∀ t : ℝ, Perpendicular (f t) L ∧ (f t).slope * P.x + (f t).intercept = P.y) :=
sorry

end NUMINAMATH_CALUDE_parallel_and_perpendicular_lines_l175_17502


namespace NUMINAMATH_CALUDE_soccer_ball_donation_l175_17512

theorem soccer_ball_donation :
  let num_schools : ℕ := 2
  let elementary_classes_per_school : ℕ := 4
  let middle_classes_per_school : ℕ := 5
  let balls_per_class : ℕ := 5
  let total_classes : ℕ := num_schools * (elementary_classes_per_school + middle_classes_per_school)
  let total_balls : ℕ := total_classes * balls_per_class
  total_balls = 90 := by
  sorry

end NUMINAMATH_CALUDE_soccer_ball_donation_l175_17512


namespace NUMINAMATH_CALUDE_hotel_rate_problem_l175_17577

theorem hotel_rate_problem (f n : ℝ) 
  (h1 : f + 3 * n = 210)  -- 4-night stay cost
  (h2 : f + 6 * n = 350)  -- 7-night stay cost
  : f = 70 := by
  sorry

end NUMINAMATH_CALUDE_hotel_rate_problem_l175_17577


namespace NUMINAMATH_CALUDE_sum_of_two_valid_numbers_l175_17532

def is_valid_number (n : ℕ) : Prop :=
  ∃ (d1 d2 d3 d4 d5 d6 d7 d8 d9 : ℕ),
    n = d1 * 100000000 + d2 * 10000000 + d3 * 1000000 + d4 * 100000 + 
        d5 * 10000 + d6 * 1000 + d7 * 100 + d8 * 10 + d9 ∧
    d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d1 ≠ d5 ∧ d1 ≠ d6 ∧ d1 ≠ d7 ∧ d1 ≠ d8 ∧ d1 ≠ d9 ∧
    d2 ≠ d3 ∧ d2 ≠ d4 ∧ d2 ≠ d5 ∧ d2 ≠ d6 ∧ d2 ≠ d7 ∧ d2 ≠ d8 ∧ d2 ≠ d9 ∧
    d3 ≠ d4 ∧ d3 ≠ d5 ∧ d3 ≠ d6 ∧ d3 ≠ d7 ∧ d3 ≠ d8 ∧ d3 ≠ d9 ∧
    d4 ≠ d5 ∧ d4 ≠ d6 ∧ d4 ≠ d7 ∧ d4 ≠ d8 ∧ d4 ≠ d9 ∧
    d5 ≠ d6 ∧ d5 ≠ d7 ∧ d5 ≠ d8 ∧ d5 ≠ d9 ∧
    d6 ≠ d7 ∧ d6 ≠ d8 ∧ d6 ≠ d9 ∧
    d7 ≠ d8 ∧ d7 ≠ d9 ∧
    d8 ≠ d9 ∧
    0 < d1 ∧ d1 ≤ 9 ∧ 0 < d2 ∧ d2 ≤ 9 ∧ 0 < d3 ∧ d3 ≤ 9 ∧ 0 < d4 ∧ d4 ≤ 9 ∧
    0 < d5 ∧ d5 ≤ 9 ∧ 0 < d6 ∧ d6 ≤ 9 ∧ 0 < d7 ∧ d7 ≤ 9 ∧ 0 < d8 ∧ d8 ≤ 9 ∧
    0 < d9 ∧ d9 ≤ 9

theorem sum_of_two_valid_numbers :
  ∃ (a b : ℕ), is_valid_number a ∧ is_valid_number b ∧ a + b = 987654321 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_two_valid_numbers_l175_17532


namespace NUMINAMATH_CALUDE_sqrt_two_irrational_and_greater_than_one_l175_17578

theorem sqrt_two_irrational_and_greater_than_one :
  ∃ x : ℝ, Irrational x ∧ x > 1 :=
by
  use Real.sqrt 2
  sorry

end NUMINAMATH_CALUDE_sqrt_two_irrational_and_greater_than_one_l175_17578


namespace NUMINAMATH_CALUDE_kikis_money_l175_17511

theorem kikis_money (num_scarves : ℕ) (scarf_price : ℚ) (hat_ratio : ℚ) (hat_percentage : ℚ) :
  num_scarves = 18 →
  scarf_price = 2 →
  hat_ratio = 2 →
  hat_percentage = 60 / 100 →
  ∃ (total_money : ℚ), 
    total_money = 90 ∧
    (num_scarves : ℚ) * scarf_price = (1 - hat_percentage) * total_money ∧
    (hat_ratio * num_scarves : ℚ) * (hat_percentage * total_money / (hat_ratio * num_scarves)) = hat_percentage * total_money :=
by sorry

end NUMINAMATH_CALUDE_kikis_money_l175_17511


namespace NUMINAMATH_CALUDE_pond_side_length_l175_17560

/-- Represents the dimensions and pond of a rectangular garden -/
structure Garden where
  length : ℝ
  width : ℝ
  pond_side : ℝ

/-- Calculates the area of a rectangular garden -/
def garden_area (g : Garden) : ℝ := g.length * g.width

/-- Calculates the remaining area after building a square pond -/
def remaining_area (g : Garden) : ℝ := garden_area g - g.pond_side ^ 2

/-- Theorem stating the side length of the pond given the conditions -/
theorem pond_side_length (g : Garden) 
  (h1 : g.length = 15)
  (h2 : g.width = 10)
  (h3 : remaining_area g = (garden_area g) / 2) :
  g.pond_side = 5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_pond_side_length_l175_17560


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l175_17590

theorem quadratic_inequality_solution (x : ℝ) :
  (x^2 + 3*x < 10) ↔ (-5 < x ∧ x < 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l175_17590


namespace NUMINAMATH_CALUDE_initial_bedbug_count_l175_17575

/-- The number of bedbugs after n days, given an initial population -/
def bedbug_population (initial : ℕ) (days : ℕ) : ℕ :=
  initial * (3 ^ days)

/-- Theorem: If the number of bedbugs triples every day and there are 810 bedbugs after four days, 
    then the initial number of bedbugs was 30. -/
theorem initial_bedbug_count : bedbug_population 30 4 = 810 := by
  sorry

#check initial_bedbug_count

end NUMINAMATH_CALUDE_initial_bedbug_count_l175_17575


namespace NUMINAMATH_CALUDE_smallest_root_of_quadratic_l175_17541

theorem smallest_root_of_quadratic (x : ℝ) :
  9 * x^2 - 45 * x + 50 = 0 → x ≥ 5/3 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_root_of_quadratic_l175_17541


namespace NUMINAMATH_CALUDE_number_relation_l175_17509

theorem number_relation (A B : ℝ) (h : A = B - (4/5) * B) : B = (A + B) / (4/5) := by
  sorry

end NUMINAMATH_CALUDE_number_relation_l175_17509


namespace NUMINAMATH_CALUDE_max_value_of_g_l175_17544

noncomputable def f (c : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + c * x + 3

def tangent_perpendicular (c : ℝ) : Prop :=
  (deriv (f c) 0) * 1 = -1

noncomputable def g (c : ℝ) (x : ℝ) : ℝ := 4 * Real.log x - deriv (f c) x

theorem max_value_of_g (c : ℝ) :
  tangent_perpendicular c →
  ∃ (x_max : ℝ), x_max > 0 ∧ g c x_max = 2 * Real.log 2 - 1 ∧
  ∀ (x : ℝ), x > 0 → g c x ≤ g c x_max :=
sorry

end NUMINAMATH_CALUDE_max_value_of_g_l175_17544


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l175_17501

/-- Given two vectors a and b in ℝ², where a = (x, x+1) and b = (1, 2),
    if a is perpendicular to b, then x = -2/3 -/
theorem perpendicular_vectors_x_value (x : ℝ) :
  let a : Fin 2 → ℝ := ![x, x + 1]
  let b : Fin 2 → ℝ := ![1, 2]
  (∀ i, i < 2 → a i * b i = 0) → x = -2/3 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l175_17501


namespace NUMINAMATH_CALUDE_storage_tubs_cost_l175_17551

/-- The total cost of storage tubs -/
def total_cost (large_count : ℕ) (small_count : ℕ) (large_price : ℕ) (small_price : ℕ) : ℕ :=
  large_count * large_price + small_count * small_price

/-- Theorem: The total cost of 3 large tubs at $6 each and 6 small tubs at $5 each is $48 -/
theorem storage_tubs_cost :
  total_cost 3 6 6 5 = 48 := by
  sorry

end NUMINAMATH_CALUDE_storage_tubs_cost_l175_17551


namespace NUMINAMATH_CALUDE_trapezoid_area_l175_17552

/-- The area of a trapezoid with sum of bases 36 cm and height 15 cm is 270 square centimeters. -/
theorem trapezoid_area (base_sum : ℝ) (height : ℝ) (h1 : base_sum = 36) (h2 : height = 15) :
  (base_sum * height) / 2 = 270 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_l175_17552


namespace NUMINAMATH_CALUDE_tangent_at_2_minus_6_tangent_through_origin_l175_17527

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + x - 16

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Theorem for the tangent line at (2, -6)
theorem tangent_at_2_minus_6 :
  let x₀ : ℝ := 2
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ y = 13 * x - 32 :=
sorry

-- Theorem for the tangent line passing through the origin
theorem tangent_through_origin :
  ∃ x₀ y₀ : ℝ,
    f x₀ = y₀ ∧
    f' x₀ * (-x₀) + y₀ = 0 ∧
    (∀ x y : ℝ, y = f' x₀ * x ↔ y = 13 * x) ∧
    x₀ = -2 ∧
    y₀ = -26 :=
sorry

end NUMINAMATH_CALUDE_tangent_at_2_minus_6_tangent_through_origin_l175_17527


namespace NUMINAMATH_CALUDE_factorial_quotient_trailing_zeros_l175_17561

def trailing_zeros (n : ℕ) : ℕ := sorry

def factorial (n : ℕ) : ℕ := sorry

theorem factorial_quotient_trailing_zeros :
  trailing_zeros (factorial 2018 / (factorial 30 * factorial 11)) = 493 := by sorry

end NUMINAMATH_CALUDE_factorial_quotient_trailing_zeros_l175_17561


namespace NUMINAMATH_CALUDE_bookshelf_theorem_l175_17570

def bookshelf_problem (yoongi_notebooks jungkook_notebooks hoseok_notebooks : ℕ) : Prop :=
  yoongi_notebooks = 3 ∧ jungkook_notebooks = 3 ∧ hoseok_notebooks = 3 →
  yoongi_notebooks + jungkook_notebooks + hoseok_notebooks = 9

theorem bookshelf_theorem : bookshelf_problem 3 3 3 := by
  sorry

end NUMINAMATH_CALUDE_bookshelf_theorem_l175_17570


namespace NUMINAMATH_CALUDE_colored_regions_bound_l175_17506

/-- A structure representing a plane with n lines and colored regions -/
structure ColoredPlane where
  n : ℕ
  n_ge_2 : n ≥ 2

/-- The number of colored regions in a ColoredPlane -/
def num_colored_regions (p : ColoredPlane) : ℕ := sorry

/-- Theorem stating that the number of colored regions is bounded -/
theorem colored_regions_bound (p : ColoredPlane) :
  num_colored_regions p ≤ (p.n^2 + p.n) / 3 := by sorry

end NUMINAMATH_CALUDE_colored_regions_bound_l175_17506
