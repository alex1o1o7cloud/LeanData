import Mathlib

namespace NUMINAMATH_CALUDE_melissa_driving_hours_l2556_255662

/-- The number of hours Melissa spends driving in a year -/
def driving_hours_per_year (trips_per_month : ℕ) (hours_per_trip : ℕ) : ℕ :=
  trips_per_month * 12 * hours_per_trip

/-- Proof that Melissa spends 72 hours driving in a year -/
theorem melissa_driving_hours :
  driving_hours_per_year 2 3 = 72 := by
  sorry

end NUMINAMATH_CALUDE_melissa_driving_hours_l2556_255662


namespace NUMINAMATH_CALUDE_problem_proof_l2556_255696

theorem problem_proof : Real.sqrt 8 - 4 * Real.sin (π / 4) - (1 / 3)⁻¹ = -3 := by
  sorry

end NUMINAMATH_CALUDE_problem_proof_l2556_255696


namespace NUMINAMATH_CALUDE_integer_roots_of_polynomial_l2556_255632

def polynomial (x : ℤ) : ℤ := x^3 + 2*x^2 - 5*x + 30

def is_root (x : ℤ) : Prop := polynomial x = 0

def divisors_of_30 : Set ℤ := {x : ℤ | x ∣ 30 ∨ x ∣ -30}

theorem integer_roots_of_polynomial :
  {x : ℤ | is_root x} = divisors_of_30 :=
sorry

end NUMINAMATH_CALUDE_integer_roots_of_polynomial_l2556_255632


namespace NUMINAMATH_CALUDE_complex_equation_system_l2556_255648

theorem complex_equation_system (p q r u v w : ℂ) 
  (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hu : u ≠ 0) (hv : v ≠ 0) (hw : w ≠ 0)
  (eq1 : p = (q + r) / (u - 3))
  (eq2 : q = (p + r) / (v - 3))
  (eq3 : r = (p + q) / (w - 3))
  (eq4 : u * v + u * w + v * w = 7)
  (eq5 : u + v + w = 4) :
  u * v * w = 10 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_system_l2556_255648


namespace NUMINAMATH_CALUDE_min_value_theorem_l2556_255673

theorem min_value_theorem (a b : ℝ) (h1 : a + 2*b = 2) (h2 : a > 1) (h3 : b > 0) :
  (∀ x y : ℝ, x > 1 ∧ y > 0 ∧ x + 2*y = 2 → 2/(x-1) + 1/y ≥ 2/(a-1) + 1/b) ∧
  2/(a-1) + 1/b = 8 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2556_255673


namespace NUMINAMATH_CALUDE_fg_properties_l2556_255691

open Real

/-- The function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 3 * x - (a + 1) * log x

/-- The function g(x) -/
def g (a : ℝ) (x : ℝ) : ℝ := x^2 - a * x + 4

/-- The sum of f(x) and g(x) -/
noncomputable def sum_fg (a : ℝ) (x : ℝ) : ℝ := f a x + g a x

/-- The difference of f(x) and g(x) -/
noncomputable def diff_fg (a : ℝ) (x : ℝ) : ℝ := f a x - g a x

/-- Theorem stating the conditions for monotonicity and tangency -/
theorem fg_properties :
  (∀ a : ℝ, a ≤ -1 → ∀ x > 0, Monotone (sum_fg a)) ∧
  (∃ a : ℝ, 1 < a ∧ a < 3 ∧ ∃ x > 0, diff_fg a x = 0 ∧ HasDerivAt (diff_fg a) 0 x) :=
sorry

end NUMINAMATH_CALUDE_fg_properties_l2556_255691


namespace NUMINAMATH_CALUDE_root_quadratic_equation_l2556_255633

theorem root_quadratic_equation (m : ℝ) : 
  (2 * m^2 - 3 * m - 1 = 0) → (4 * m^2 - 6 * m = 2) := by
  sorry

end NUMINAMATH_CALUDE_root_quadratic_equation_l2556_255633


namespace NUMINAMATH_CALUDE_number_problem_l2556_255690

theorem number_problem : ∃ x : ℚ, (x / 6) * 12 = 8 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2556_255690


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2556_255677

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_pos : ∀ n, a n > 0)
  (h_eq : a 6 = a 5 + 2 * a 4) :
  ∃ q : ℝ, q = 2 ∧ ∀ n : ℕ, a (n + 1) = a n * q :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2556_255677


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_0_2_l2556_255637

theorem arithmetic_square_root_of_0_2 : ∃ x : ℝ, x^2 = 0.2 ∧ x ≠ 0.02 := by sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_0_2_l2556_255637


namespace NUMINAMATH_CALUDE_inscribed_cylinder_radius_l2556_255684

/-- A right circular cylinder inscribed in a right circular cone -/
structure InscribedCylinder where
  cone_diameter : ℝ
  cone_altitude : ℝ
  cylinder_radius : ℝ
  cylinder_height : ℝ
  h_cylinder_diameter_height : cylinder_height = 2 * cylinder_radius
  h_axes_coincide : True

/-- The theorem stating the radius of the inscribed cylinder -/
theorem inscribed_cylinder_radius 
  (c : InscribedCylinder) 
  (h_cone_diameter : c.cone_diameter = 16)
  (h_cone_altitude : c.cone_altitude = 20) :
  c.cylinder_radius = 40 / 9 := by
  sorry

#check inscribed_cylinder_radius

end NUMINAMATH_CALUDE_inscribed_cylinder_radius_l2556_255684


namespace NUMINAMATH_CALUDE_jelly_bean_probability_l2556_255656

theorem jelly_bean_probability (red orange blue : ℝ) (h1 : red = 0.25) (h2 : orange = 0.4) (h3 : blue = 0.1) :
  ∃ yellow : ℝ, yellow = 0.25 ∧ red + orange + blue + yellow = 1 := by
  sorry

end NUMINAMATH_CALUDE_jelly_bean_probability_l2556_255656


namespace NUMINAMATH_CALUDE_chairs_count_l2556_255674

/-- The number of chairs bought for the entire house -/
def total_chairs (living_room kitchen dining_room outdoor_patio : ℕ) : ℕ :=
  living_room + kitchen + dining_room + outdoor_patio

/-- Theorem stating that the total number of chairs is 29 -/
theorem chairs_count :
  total_chairs 3 6 8 12 = 29 := by
  sorry

end NUMINAMATH_CALUDE_chairs_count_l2556_255674


namespace NUMINAMATH_CALUDE_vector_parallelism_l2556_255615

/-- Given two 2D vectors a and b, prove that when k*a + b is parallel to a - 3*b, k = -1/3 --/
theorem vector_parallelism (a b : ℝ × ℝ) (k : ℝ) 
    (h1 : a = (1, 2))
    (h2 : b = (-3, 2))
    (h3 : ∃ (t : ℝ), t ≠ 0 ∧ k • a + b = t • (a - 3 • b)) :
  k = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallelism_l2556_255615


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l2556_255634

/-- Given a principal amount and an interest rate, if the simple interest for 2 years
    is $600 and the compound interest for 2 years is $612, then the interest rate is 104%. -/
theorem interest_rate_calculation (P R : ℝ) : 
  P * R * 2 / 100 = 600 →
  P * ((1 + R / 100)^2 - 1) = 612 →
  R = 104 := by
sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l2556_255634


namespace NUMINAMATH_CALUDE_fraction_simplification_l2556_255647

theorem fraction_simplification (x : ℝ) (h : x ≠ 4) :
  (x^2 - 4*x) / (x^2 - 8*x + 16) = x / (x - 4) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2556_255647


namespace NUMINAMATH_CALUDE_betty_gave_forty_percent_l2556_255665

/-- The percentage of marbles Betty gave to Stuart -/
def percentage_given (betty_initial : ℕ) (stuart_initial : ℕ) (stuart_final : ℕ) : ℚ :=
  (stuart_final - stuart_initial : ℚ) / betty_initial * 100

/-- Theorem stating that Betty gave Stuart 40% of her marbles -/
theorem betty_gave_forty_percent :
  let betty_initial : ℕ := 60
  let stuart_initial : ℕ := 56
  let stuart_final : ℕ := 80
  percentage_given betty_initial stuart_initial stuart_final = 40 := by
sorry

end NUMINAMATH_CALUDE_betty_gave_forty_percent_l2556_255665


namespace NUMINAMATH_CALUDE_polynomial_remainder_l2556_255663

theorem polynomial_remainder (x : ℝ) : (x^4 + x + 2) % (x - 3) = 86 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l2556_255663


namespace NUMINAMATH_CALUDE_sequence_problem_l2556_255630

/-- Given a sequence {a_n} with sum S_n = kn^2 + n and a_10 = 39, prove a_100 = 399 -/
theorem sequence_problem (k : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, S n = k * n^2 + n) →
  a 10 = 39 →
  a 100 = 399 := by
sorry

end NUMINAMATH_CALUDE_sequence_problem_l2556_255630


namespace NUMINAMATH_CALUDE_problem_solution_l2556_255659

theorem problem_solution (c d : ℝ) 
  (eq1 : 5 + c = 3 - d)
  (eq2 : 3 + d = 8 + c)
  (eq3 : c - d = 2) : 
  5 - c = 5 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2556_255659


namespace NUMINAMATH_CALUDE_factorial_prime_factorization_l2556_255675

theorem factorial_prime_factorization :
  let x : ℕ := Finset.prod (Finset.range 15) (fun i => i + 1)
  ∀ (i k m p q r : ℕ),
    (i > 0 ∧ k > 0 ∧ m > 0 ∧ p > 0 ∧ q > 0 ∧ r > 0) →
    x = 2^i * 3^k * 5^m * 7^p * 11^q * 13^r →
    i + k + m + p + q + r = 29 := by
  sorry

end NUMINAMATH_CALUDE_factorial_prime_factorization_l2556_255675


namespace NUMINAMATH_CALUDE_least_positive_integer_divisible_by_four_primes_l2556_255610

theorem least_positive_integer_divisible_by_four_primes : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (p₁ p₂ p₃ p₄ : ℕ), Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
    p₁ ∣ n ∧ p₂ ∣ n ∧ p₃ ∣ n ∧ p₄ ∣ n) ∧
  (∀ m : ℕ, m > 0 → m < n → 
    ¬(∃ (q₁ q₂ q₃ q₄ : ℕ), Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ Prime q₄ ∧ 
      q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₃ ≠ q₄ ∧
      q₁ ∣ m ∧ q₂ ∣ m ∧ q₃ ∣ m ∧ q₄ ∣ m)) ∧
  n = 210 := by
sorry

end NUMINAMATH_CALUDE_least_positive_integer_divisible_by_four_primes_l2556_255610


namespace NUMINAMATH_CALUDE_geometric_sequence_12th_term_l2556_255655

/-- A geometric sequence is defined by its first term and common ratio. -/
structure GeometricSequence where
  a : ℝ  -- first term
  r : ℝ  -- common ratio

/-- The nth term of a geometric sequence. -/
def GeometricSequence.nthTerm (seq : GeometricSequence) (n : ℕ) : ℝ :=
  seq.a * seq.r ^ (n - 1)

/-- Theorem: In a geometric sequence where the 5th term is 8 and the 9th term is 128, the 12th term is 1024. -/
theorem geometric_sequence_12th_term
  (seq : GeometricSequence)
  (h5 : seq.nthTerm 5 = 8)
  (h9 : seq.nthTerm 9 = 128) :
  seq.nthTerm 12 = 1024 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_12th_term_l2556_255655


namespace NUMINAMATH_CALUDE_triangle_side_a_triangle_angle_C_l2556_255608

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Part 1
theorem triangle_side_a (t : Triangle) (hb : t.b = 2) (hB : t.B = π/6) (hC : t.C = 3*π/4) :
  t.a = Real.sqrt 6 - Real.sqrt 2 := by
  sorry

-- Part 2
theorem triangle_angle_C (t : Triangle) (hS : t.a * t.b * Real.sin t.C / 2 = (t.a^2 + t.b^2 - t.c^2) / 4) :
  t.C = π/4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_a_triangle_angle_C_l2556_255608


namespace NUMINAMATH_CALUDE_cylinder_cone_dimensions_l2556_255650

theorem cylinder_cone_dimensions (r m : ℝ) : 
  r > 0 ∧ m > 0 →
  (2 * π * r * m) / (π * r * Real.sqrt (m^2 + r^2)) = 8 / 5 →
  r * m = 588 →
  m = 28 ∧ r = 21 :=
by sorry

end NUMINAMATH_CALUDE_cylinder_cone_dimensions_l2556_255650


namespace NUMINAMATH_CALUDE_x_in_terms_of_abc_and_k_l2556_255694

theorem x_in_terms_of_abc_and_k 
  (k a b c x y z : ℝ) 
  (hk : k ≠ 0)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (h1 : x * y / (k * (x + y)) = a)
  (h2 : x * z / (k * (x + z)) = b)
  (h3 : y * z / (k * (y + z)) = c) :
  x = 2 * a * b * c / (k * (a * c + b * c - a * b)) :=
sorry

end NUMINAMATH_CALUDE_x_in_terms_of_abc_and_k_l2556_255694


namespace NUMINAMATH_CALUDE_probability_at_least_one_defective_l2556_255623

theorem probability_at_least_one_defective (total : ℕ) (defective : ℕ) : 
  total = 20 → defective = 4 → 
  (1 - (total - defective) * (total - defective - 1) / (total * (total - 1))) = 7 / 19 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_defective_l2556_255623


namespace NUMINAMATH_CALUDE_greg_initial_amount_l2556_255605

/-- Represents the initial and final monetary states of Earl, Fred, and Greg -/
structure MonetaryState where
  earl_initial : ℕ
  fred_initial : ℕ
  greg_initial : ℕ
  earl_owes_fred : ℕ
  fred_owes_greg : ℕ
  greg_owes_earl : ℕ
  earl_final : ℕ
  fred_final : ℕ
  greg_final : ℕ

/-- The theorem states that given the initial conditions and debt payments,
    Greg's initial amount is 36 dollars -/
theorem greg_initial_amount (state : MonetaryState)
  (h1 : state.earl_initial = 90)
  (h2 : state.fred_initial = 48)
  (h3 : state.earl_owes_fred = 28)
  (h4 : state.fred_owes_greg = 32)
  (h5 : state.greg_owes_earl = 40)
  (h6 : state.earl_final + state.greg_final = 130)
  (h7 : state.earl_final = state.earl_initial - state.earl_owes_fred + state.greg_owes_earl)
  (h8 : state.fred_final = state.fred_initial + state.earl_owes_fred - state.fred_owes_greg)
  (h9 : state.greg_final = state.greg_initial + state.fred_owes_greg - state.greg_owes_earl) :
  state.greg_initial = 36 := by
  sorry


end NUMINAMATH_CALUDE_greg_initial_amount_l2556_255605


namespace NUMINAMATH_CALUDE_parabola_sum_l2556_255652

/-- A parabola with equation y = ax^2 + bx + c, vertex at (-3, 2), and passing through (-1, -2) -/
def Parabola (a b c : ℝ) : Prop :=
  (∀ x y : ℝ, y = a * x^2 + b * x + c ↔ (x + 3)^2 = (y - 2) / a) ∧
  a * (-1)^2 + b * (-1) + c = -2

theorem parabola_sum (a b c : ℝ) (h : Parabola a b c) : a + b + c = -14 := by
  sorry

end NUMINAMATH_CALUDE_parabola_sum_l2556_255652


namespace NUMINAMATH_CALUDE_sandy_average_book_price_l2556_255683

/-- The average price of books bought by Sandy -/
def average_price (shop1_books shop2_books : ℕ) (shop1_cost shop2_cost : ℚ) : ℚ :=
  (shop1_cost + shop2_cost) / (shop1_books + shop2_books)

/-- Theorem: The average price Sandy paid per book is $16 -/
theorem sandy_average_book_price :
  average_price 65 55 1080 840 = 16 := by
  sorry

end NUMINAMATH_CALUDE_sandy_average_book_price_l2556_255683


namespace NUMINAMATH_CALUDE_yellow_marbles_count_l2556_255614

/-- Represents the number of marbles of each color in Jamal's bag -/
structure MarbleCounts where
  blue : ℕ
  green : ℕ
  black : ℕ
  yellow : ℕ

/-- The probability of drawing a black marble from the bag -/
def blackMarbleProbability : ℚ := 1 / 28

/-- The total number of marbles in the bag -/
def totalMarbles (counts : MarbleCounts) : ℕ :=
  counts.blue + counts.green + counts.black + counts.yellow

/-- The theorem stating the number of yellow marbles in Jamal's bag -/
theorem yellow_marbles_count (counts : MarbleCounts) 
  (h_blue : counts.blue = 10)
  (h_green : counts.green = 5)
  (h_black : counts.black = 1)
  (h_prob : (counts.black : ℚ) / (totalMarbles counts) = blackMarbleProbability) :
  counts.yellow = 12 := by
  sorry

end NUMINAMATH_CALUDE_yellow_marbles_count_l2556_255614


namespace NUMINAMATH_CALUDE_unique_non_negative_twelve_quotient_l2556_255627

def pairs : List (Int × Int) := [(24, -2), (-36, 3), (144, 12), (-48, 4), (72, -6)]

theorem unique_non_negative_twelve_quotient :
  ∃! p : Int × Int, p ∈ pairs ∧ p.1 / p.2 ≠ -12 :=
by sorry

end NUMINAMATH_CALUDE_unique_non_negative_twelve_quotient_l2556_255627


namespace NUMINAMATH_CALUDE_survey_result_l2556_255693

theorem survey_result (total : ℕ) (radio_dislike_ratio : ℚ) (music_dislike_ratio : ℚ)
  (h_total : total = 2000)
  (h_radio : radio_dislike_ratio = 1/4)
  (h_music : music_dislike_ratio = 3/20) :
  (total : ℚ) * radio_dislike_ratio * music_dislike_ratio = 75 := by
  sorry

end NUMINAMATH_CALUDE_survey_result_l2556_255693


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2556_255607

theorem inequality_solution_set : 
  {x : ℝ | (x^2 + x^3 - 3*x^4) / (x + x^2 - 3*x^3) ≥ -1} = 
  {x : ℝ | x ∈ Set.Icc (-1) (((-1 - Real.sqrt 13) / 6 : ℝ)) ∪ 
           Set.Ioo (((-1 - Real.sqrt 13) / 6 : ℝ)) (((-1 + Real.sqrt 13) / 6 : ℝ)) ∪
           Set.Ioo (((-1 + Real.sqrt 13) / 6 : ℝ)) 0 ∪
           Set.Ioi 0} :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2556_255607


namespace NUMINAMATH_CALUDE_intersection_of_lines_l2556_255638

/-- The intersection point of two lines in 3D space --/
def intersection_point (A B C D : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  sorry

theorem intersection_of_lines (A B C D : ℝ × ℝ × ℝ) 
  (hA : A = (5, -3, 2)) 
  (hB : B = (15, -13, 7)) 
  (hC : C = (2, 4, -5)) 
  (hD : D = (4, -1, 15)) : 
  intersection_point A B C D = (23/3, -19/3, 7/3) := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l2556_255638


namespace NUMINAMATH_CALUDE_work_completion_time_l2556_255624

theorem work_completion_time (a_days b_days : ℝ) (ha : a_days > 0) (hb : b_days > 0) :
  a_days = 60 → b_days = 20 → (a_days⁻¹ + b_days⁻¹)⁻¹ = 15 := by
  sorry

#check work_completion_time

end NUMINAMATH_CALUDE_work_completion_time_l2556_255624


namespace NUMINAMATH_CALUDE_total_vegetarian_consumers_is_33_l2556_255628

/-- Represents the dietary information of a family -/
structure DietaryInfo where
  only_vegetarian : ℕ
  only_non_vegetarian : ℕ
  both : ℕ
  gluten_free : ℕ
  vegan : ℕ
  non_veg_gluten_free : ℕ
  veg_gluten_free : ℕ
  both_gluten_free : ℕ
  vegan_strict_veg : ℕ
  vegan_non_veg : ℕ

/-- Calculates the total number of people consuming vegetarian dishes -/
def total_vegetarian_consumers (info : DietaryInfo) : ℕ :=
  info.only_vegetarian + info.both + info.vegan_non_veg

/-- The main theorem stating that the total number of vegetarian consumers is 33 -/
theorem total_vegetarian_consumers_is_33 (info : DietaryInfo) 
  (h1 : info.only_vegetarian = 19)
  (h2 : info.only_non_vegetarian = 9)
  (h3 : info.both = 12)
  (h4 : info.gluten_free = 6)
  (h5 : info.vegan = 5)
  (h6 : info.non_veg_gluten_free = 2)
  (h7 : info.veg_gluten_free = 3)
  (h8 : info.both_gluten_free = 1)
  (h9 : info.vegan_strict_veg = 3)
  (h10 : info.vegan_non_veg = 2) :
  total_vegetarian_consumers info = 33 := by
  sorry

end NUMINAMATH_CALUDE_total_vegetarian_consumers_is_33_l2556_255628


namespace NUMINAMATH_CALUDE_percentage_problem_l2556_255625

theorem percentage_problem (P : ℝ) : 
  (0.5 * 640 = P / 100 * 650 + 190) → P = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2556_255625


namespace NUMINAMATH_CALUDE_product_of_roots_l2556_255640

theorem product_of_roots (a b c : ℝ) (h : (5 + 3 * Real.sqrt 5) * a + (3 + Real.sqrt 5) * b + c = 0) :
  a * b = -((15 - 9 * Real.sqrt 5) / 20) := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l2556_255640


namespace NUMINAMATH_CALUDE_min_value_squared_sum_l2556_255620

theorem min_value_squared_sum (x y a : ℝ) : 
  x + y ≥ a →
  x - y ≤ a →
  y ≤ a →
  a > 0 →
  (∀ x' y' : ℝ, x' + y' ≥ a → x' - y' ≤ a → y' ≤ a → x'^2 + y'^2 ≥ 2) →
  (∃ x'' y'' : ℝ, x'' + y'' ≥ a ∧ x'' - y'' ≤ a ∧ y'' ≤ a ∧ x''^2 + y''^2 = 2) →
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_squared_sum_l2556_255620


namespace NUMINAMATH_CALUDE_zero_function_satisfies_equation_l2556_255678

theorem zero_function_satisfies_equation (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x + y) = f x - f y) → (∀ x : ℝ, f x = 0) := by
  sorry

end NUMINAMATH_CALUDE_zero_function_satisfies_equation_l2556_255678


namespace NUMINAMATH_CALUDE_wood_bundles_problem_l2556_255688

/-- The number of wood bundles at the start of the day, given the number of bundles
    burned in the morning and afternoon, and the number left at the end of the day. -/
def initial_bundles (morning_burned : ℕ) (afternoon_burned : ℕ) (end_day_left : ℕ) : ℕ :=
  morning_burned + afternoon_burned + end_day_left

/-- Theorem stating that the initial number of wood bundles is 10, given the
    conditions from the problem. -/
theorem wood_bundles_problem :
  initial_bundles 4 3 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_wood_bundles_problem_l2556_255688


namespace NUMINAMATH_CALUDE_james_glasses_cost_l2556_255622

/-- The total cost of James' new pair of glasses -/
def total_cost (frame_cost lens_cost insurance_coverage frame_discount : ℚ) : ℚ :=
  (frame_cost - frame_discount) + (lens_cost * (1 - insurance_coverage))

/-- Theorem stating the total cost for James' new pair of glasses -/
theorem james_glasses_cost :
  let frame_cost : ℚ := 200
  let lens_cost : ℚ := 500
  let insurance_coverage : ℚ := 0.8
  let frame_discount : ℚ := 50
  total_cost frame_cost lens_cost insurance_coverage frame_discount = 250 := by
sorry

end NUMINAMATH_CALUDE_james_glasses_cost_l2556_255622


namespace NUMINAMATH_CALUDE_bus_route_distance_bounds_l2556_255689

/-- Represents a bus route with n stops -/
structure BusRoute (n : ℕ) where
  distance_between_stops : ℝ
  (distance_positive : distance_between_stops > 0)

/-- Represents a vehicle's journey through all stops -/
def Journey (n : ℕ) := Fin n → Fin n

/-- Calculates the distance traveled in a journey -/
def distance_traveled (r : BusRoute n) (j : Journey n) : ℝ :=
  sorry

/-- Theorem stating the maximum and minimum distances for a 10-stop route -/
theorem bus_route_distance_bounds :
  ∀ (r : BusRoute 10),
    (∃ (j : Journey 10), distance_traveled r j = 50 * r.distance_between_stops) ∧
    (∃ (j : Journey 10), distance_traveled r j = 18 * r.distance_between_stops) ∧
    (∀ (j : Journey 10), 18 * r.distance_between_stops ≤ distance_traveled r j ∧ 
                         distance_traveled r j ≤ 50 * r.distance_between_stops) :=
sorry

end NUMINAMATH_CALUDE_bus_route_distance_bounds_l2556_255689


namespace NUMINAMATH_CALUDE_value_of_a_l2556_255639

-- Define the conversion rate from paise to rupees
def paise_to_rupees (paise : ℚ) : ℚ := paise / 100

-- Define the problem statement
theorem value_of_a (a : ℚ) (h : (a * 0.005) = paise_to_rupees 85) : a = 170 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l2556_255639


namespace NUMINAMATH_CALUDE_perpendicular_transitivity_l2556_255672

/-- In three-dimensional space -/
structure Space3D where

/-- Represent a line in 3D space -/
structure Line (S : Space3D) where

/-- Represent a plane in 3D space -/
structure Plane (S : Space3D) where

/-- Perpendicular relation between two lines -/
def Line.perp (S : Space3D) (l1 l2 : Line S) : Prop :=
  sorry

/-- Perpendicular relation between a line and a plane -/
def Line.perpToPlane (S : Space3D) (l : Line S) (p : Plane S) : Prop :=
  sorry

/-- Perpendicular relation between two planes -/
def Plane.perp (S : Space3D) (p1 p2 : Plane S) : Prop :=
  sorry

/-- The main theorem -/
theorem perpendicular_transitivity (S : Space3D) (a b : Line S) (α β : Plane S) :
  a ≠ b → α ≠ β →
  Line.perp S a b →
  Line.perpToPlane S a α →
  Line.perpToPlane S b β →
  Plane.perp S α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_transitivity_l2556_255672


namespace NUMINAMATH_CALUDE_equation_exists_solution_l2556_255646

theorem equation_exists_solution (x : ℝ) (hx : x = 2407) :
  ∃ (y z : ℝ), x^y + y^x = z :=
sorry

end NUMINAMATH_CALUDE_equation_exists_solution_l2556_255646


namespace NUMINAMATH_CALUDE_loan_repayment_equality_l2556_255654

/-- Represents the loan scenario described in the problem -/
structure LoanScenario where
  M : ℝ  -- Initial loan amount in million yuan
  x : ℝ  -- Monthly repayment amount in million yuan
  r : ℝ  -- Monthly interest rate (as a decimal)
  n : ℕ  -- Number of months for repayment

/-- The theorem representing the loan repayment equality -/
theorem loan_repayment_equality (scenario : LoanScenario) 
  (h_r : scenario.r = 0.05)
  (h_n : scenario.n = 20) : 
  scenario.n * scenario.x = scenario.M * (1 + scenario.r) ^ scenario.n :=
sorry

end NUMINAMATH_CALUDE_loan_repayment_equality_l2556_255654


namespace NUMINAMATH_CALUDE_expense_difference_zero_l2556_255606

def vacation_expenses (anne_paid beth_paid carlos_paid : ℕ) (a b : ℕ) : Prop :=
  let total := anne_paid + beth_paid + carlos_paid
  let share := total / 3
  (anne_paid + b = share + a) ∧
  (beth_paid = share + b) ∧
  (carlos_paid + a = share)

theorem expense_difference_zero 
  (anne_paid beth_paid carlos_paid : ℕ) 
  (a b : ℕ) 
  (h : vacation_expenses anne_paid beth_paid carlos_paid a b) :
  a - b = 0 :=
sorry

end NUMINAMATH_CALUDE_expense_difference_zero_l2556_255606


namespace NUMINAMATH_CALUDE_three_distinct_roots_transformation_l2556_255603

/-- Given an equation a x^5 + b x^4 + c = 0 with three distinct roots,
    prove that c x^5 + b x + a = 0 also has three distinct roots -/
theorem three_distinct_roots_transformation (a b c : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    a * x^5 + b * x^4 + c = 0 ∧
    a * y^5 + b * y^4 + c = 0 ∧
    a * z^5 + b * z^4 + c = 0) →
  (∃ u v w : ℝ, u ≠ v ∧ v ≠ w ∧ u ≠ w ∧
    c * u^5 + b * u + a = 0 ∧
    c * v^5 + b * v + a = 0 ∧
    c * w^5 + b * w + a = 0) :=
by sorry

end NUMINAMATH_CALUDE_three_distinct_roots_transformation_l2556_255603


namespace NUMINAMATH_CALUDE_sqrt_sum_rational_implies_components_rational_l2556_255687

theorem sqrt_sum_rational_implies_components_rational
  (m n p : ℚ)
  (h : ∃ (q : ℚ), Real.sqrt m + Real.sqrt n + Real.sqrt p = q) :
  (∃ (r : ℚ), Real.sqrt m = r) ∧
  (∃ (s : ℚ), Real.sqrt n = s) ∧
  (∃ (t : ℚ), Real.sqrt p = t) :=
sorry

end NUMINAMATH_CALUDE_sqrt_sum_rational_implies_components_rational_l2556_255687


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2556_255635

theorem arithmetic_sequence_property (n : ℕ+) : 
  (∀ S : Finset ℕ, S ⊆ Finset.range 1989 → S.card = n → 
    ∃ (a d : ℕ) (H : Finset ℕ), H ⊆ S ∧ H.card = 29 ∧ 
    ∀ k, k ∈ H → ∃ i, 0 ≤ i ∧ i < 29 ∧ k = a + i * d) → 
  n > 1788 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2556_255635


namespace NUMINAMATH_CALUDE_min_sum_squares_l2556_255621

theorem min_sum_squares (x y z : ℝ) (h : x + 2*y + 2*z = 6) :
  ∃ (m : ℝ), (∀ a b c : ℝ, a + 2*b + 2*c = 6 → a^2 + b^2 + c^2 ≥ m) ∧
             (x^2 + y^2 + z^2 = m) ∧
             m = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_l2556_255621


namespace NUMINAMATH_CALUDE_M_in_second_and_fourth_quadrants_l2556_255699

def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 * p.2 < 0}

theorem M_in_second_and_fourth_quadrants :
  ∀ p ∈ M, (p.1 > 0 ∧ p.2 < 0) ∨ (p.1 < 0 ∧ p.2 > 0) :=
by sorry

end NUMINAMATH_CALUDE_M_in_second_and_fourth_quadrants_l2556_255699


namespace NUMINAMATH_CALUDE_maintenance_interval_after_additive_l2556_255682

/-- Calculates the new maintenance interval after applying an additive -/
def new_maintenance_interval (original_interval : ℕ) (increase_percentage : ℕ) : ℕ :=
  original_interval * (100 + increase_percentage) / 100

/-- Theorem: Given an original maintenance interval of 50 days and a 20% increase,
    the new maintenance interval is 60 days -/
theorem maintenance_interval_after_additive :
  new_maintenance_interval 50 20 = 60 := by
  sorry

end NUMINAMATH_CALUDE_maintenance_interval_after_additive_l2556_255682


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l2556_255618

-- Define the vectors
def a : Fin 2 → ℝ := ![1, 2]
def b (m : ℝ) : Fin 2 → ℝ := ![m, 4]

-- Define the parallelism condition
def are_parallel (u v : Fin 2 → ℝ) : Prop :=
  u 0 * v 1 = u 1 * v 0

-- Theorem statement
theorem parallel_vectors_m_value :
  ∀ m : ℝ, are_parallel a (λ i ↦ 2 * a i + b m i) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l2556_255618


namespace NUMINAMATH_CALUDE_excess_weight_is_51_8_l2556_255692

/-- The weight of the bridge in kilograms -/
def bridge_weight : ℝ := 130

/-- Kelly's weight in kilograms -/
def kelly_weight : ℝ := 34

/-- Sam's weight in kilograms -/
def sam_weight : ℝ := 40

/-- Daisy's weight in kilograms -/
def daisy_weight : ℝ := 28

/-- Megan's weight in kilograms -/
def megan_weight : ℝ := kelly_weight * 1.1

/-- Mike's weight in kilograms -/
def mike_weight : ℝ := megan_weight + 5

/-- The total weight of all five children -/
def total_weight : ℝ := kelly_weight + sam_weight + daisy_weight + megan_weight + mike_weight

/-- Theorem stating that the excess weight is 51.8 kg -/
theorem excess_weight_is_51_8 : total_weight - bridge_weight = 51.8 := by
  sorry

end NUMINAMATH_CALUDE_excess_weight_is_51_8_l2556_255692


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l2556_255642

theorem modulus_of_complex_fraction (z : ℂ) : z = (1 + 2*I) / I → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l2556_255642


namespace NUMINAMATH_CALUDE_area_ratio_quadrilateral_triangle_l2556_255667

-- Define the types for points and shapes
variable (Point : Type) [AddCommGroup Point] [Module ℝ Point]
variable (Quadrilateral : Type)
variable (Triangle : Type)

-- Define functions for area calculation
variable (area : Quadrilateral → ℝ)
variable (area_triangle : Triangle → ℝ)

-- Define a function to create a quadrilateral from four points
variable (make_quadrilateral : Point → Point → Point → Point → Quadrilateral)

-- Define a function to create a triangle from three points
variable (make_triangle : Point → Point → Point → Triangle)

-- Define a function to get the midpoint of two points
variable (midpoint : Point → Point → Point)

-- Define a function to extend two line segments to their intersection
variable (extend_to_intersection : Point → Point → Point → Point → Point)

-- Theorem statement
theorem area_ratio_quadrilateral_triangle 
  (A B C D : Point) 
  (ABCD : Quadrilateral) 
  (E : Point) 
  (H G : Point) 
  (EHG : Triangle) :
  ABCD = make_quadrilateral A B C D →
  E = extend_to_intersection A D B C →
  H = midpoint B D →
  G = midpoint A C →
  EHG = make_triangle E H G →
  (area_triangle EHG) / (area ABCD) = 1/4 := by sorry

end NUMINAMATH_CALUDE_area_ratio_quadrilateral_triangle_l2556_255667


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2556_255697

theorem solution_set_quadratic_inequality :
  {x : ℝ | 6 * x^2 + 5 * x < 4} = {x : ℝ | -4/3 < x ∧ x < 1/2} := by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2556_255697


namespace NUMINAMATH_CALUDE_solution_set_is_open_unit_interval_l2556_255698

-- Define a decreasing function on (-2, 2)
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, -2 < x ∧ x < y ∧ y < 2 → f x > f y

-- Define the solution set
def SolutionSet (f : ℝ → ℝ) : Set ℝ :=
  {x | -2 < x ∧ x < 2 ∧ f x > f (2 - x)}

-- Theorem statement
theorem solution_set_is_open_unit_interval
  (f : ℝ → ℝ) (h : DecreasingFunction f) :
  SolutionSet f = Set.Ioo 0 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_is_open_unit_interval_l2556_255698


namespace NUMINAMATH_CALUDE_solve_colored_paper_problem_l2556_255671

def colored_paper_problem (initial : ℕ) (given_per_friend : ℕ) (num_friends : ℕ) (bought : ℕ) (current : ℕ) : Prop :=
  initial + bought - (given_per_friend * num_friends) = current

theorem solve_colored_paper_problem :
  ∃ initial : ℕ, colored_paper_problem initial 11 2 27 63 ∧ initial = 58 := by
  sorry

end NUMINAMATH_CALUDE_solve_colored_paper_problem_l2556_255671


namespace NUMINAMATH_CALUDE_positive_real_equality_l2556_255626

theorem positive_real_equality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h1 : a^b = b^a) (h2 : b = 4*a) : a = (4 : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_positive_real_equality_l2556_255626


namespace NUMINAMATH_CALUDE_quadratic_inequality_always_holds_l2556_255649

theorem quadratic_inequality_always_holds (m : ℝ) :
  (∀ x : ℝ, m * x^2 - (m + 3) * x - 1 < 0) ↔ -9 < m ∧ m < -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_always_holds_l2556_255649


namespace NUMINAMATH_CALUDE_equation_simplification_l2556_255695

theorem equation_simplification (x : ℝ) : 
  x * (x * (x * (3 - x) - 5) + 12) + 1 = -x^4 + 3*x^3 - 5*x^2 + 12*x + 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_simplification_l2556_255695


namespace NUMINAMATH_CALUDE_smallest_y_for_square_l2556_255645

theorem smallest_y_for_square (y : ℕ+) (M : ℤ) : 
  (∀ k : ℕ+, k < y → ¬∃ N : ℤ, (2310 : ℤ) * k = N^2) →
  (∃ N : ℤ, (2310 : ℤ) * y = N^2) →
  y = 2310 := by sorry

end NUMINAMATH_CALUDE_smallest_y_for_square_l2556_255645


namespace NUMINAMATH_CALUDE_five_letter_words_same_ends_l2556_255661

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The length of the words we're considering -/
def word_length : ℕ := 5

/-- The number of letters that can vary in the word -/
def variable_letters : ℕ := word_length - 2

theorem five_letter_words_same_ends : 
  alphabet_size ^ variable_letters = 456976 := by
  sorry


end NUMINAMATH_CALUDE_five_letter_words_same_ends_l2556_255661


namespace NUMINAMATH_CALUDE_parallelogram_smaller_angle_l2556_255679

theorem parallelogram_smaller_angle (smaller_angle larger_angle : ℝ) : 
  larger_angle = smaller_angle + 120 →
  smaller_angle + larger_angle + smaller_angle + larger_angle = 360 →
  smaller_angle = 30 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_smaller_angle_l2556_255679


namespace NUMINAMATH_CALUDE_student_volunteer_arrangements_l2556_255670

theorem student_volunteer_arrangements :
  let n : ℕ := 5  -- number of students
  let k : ℕ := 2  -- number of communities
  (2^n : ℕ) - k = 30 :=
by sorry

end NUMINAMATH_CALUDE_student_volunteer_arrangements_l2556_255670


namespace NUMINAMATH_CALUDE_min_distance_to_line_l2556_255641

/-- The minimum distance from integral points to the line y = (5/3)x + 4/5 -/
theorem min_distance_to_line : 
  ∃ (d : ℝ), d = Real.sqrt 34 / 85 ∧ 
  ∀ (x y : ℤ), 
    d ≤ (|(5 : ℝ) / 3 * x + 4 / 5 - y| / Real.sqrt (1 + (5 / 3)^2)) ∧
    ∃ (x₀ y₀ : ℤ), (|(5 : ℝ) / 3 * x₀ + 4 / 5 - y₀| / Real.sqrt (1 + (5 / 3)^2)) = d := by
  sorry


end NUMINAMATH_CALUDE_min_distance_to_line_l2556_255641


namespace NUMINAMATH_CALUDE_gcd_720_90_minus_10_l2556_255686

theorem gcd_720_90_minus_10 : Nat.gcd 720 90 - 10 = 80 := by
  sorry

end NUMINAMATH_CALUDE_gcd_720_90_minus_10_l2556_255686


namespace NUMINAMATH_CALUDE_fair_admission_collection_l2556_255629

theorem fair_admission_collection :
  let child_fee : ℚ := 3/2  -- $1.50 as a rational number
  let adult_fee : ℚ := 4    -- $4.00 as a rational number
  let total_people : ℕ := 2200
  let num_children : ℕ := 700
  let num_adults : ℕ := 1500
  
  (num_children : ℚ) * child_fee + (num_adults : ℚ) * adult_fee = 7050
  := by sorry

end NUMINAMATH_CALUDE_fair_admission_collection_l2556_255629


namespace NUMINAMATH_CALUDE_officer_selection_count_correct_l2556_255604

/-- The number of members in the club -/
def club_size : Nat := 12

/-- The number of officer positions to be filled -/
def officer_positions : Nat := 5

/-- The number of ways to choose officers from club members -/
def officer_selection_count : Nat := 95040

/-- Theorem stating that the number of ways to choose officers is correct -/
theorem officer_selection_count_correct :
  (club_size.factorial) / ((club_size - officer_positions).factorial) = officer_selection_count := by
  sorry

end NUMINAMATH_CALUDE_officer_selection_count_correct_l2556_255604


namespace NUMINAMATH_CALUDE_power_zero_equals_one_l2556_255613

theorem power_zero_equals_one (x : ℝ) : x ^ 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_zero_equals_one_l2556_255613


namespace NUMINAMATH_CALUDE_sum_of_roots_l2556_255668

theorem sum_of_roots (k p : ℝ) (x₁ x₂ : ℝ) :
  (4 * x₁^2 - k * x₁ - p = 0) →
  (4 * x₂^2 - k * x₂ - p = 0) →
  (x₁ ≠ x₂) →
  (x₁ + x₂ = k / 4) := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l2556_255668


namespace NUMINAMATH_CALUDE_h_zero_iff_b_eq_seven_fifths_l2556_255660

/-- The function h(x) = 5x - 7 -/
def h (x : ℝ) : ℝ := 5 * x - 7

/-- For the function h(x) = 5x - 7, h(b) = 0 if and only if b = 7/5 -/
theorem h_zero_iff_b_eq_seven_fifths :
  ∀ b : ℝ, h b = 0 ↔ b = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_h_zero_iff_b_eq_seven_fifths_l2556_255660


namespace NUMINAMATH_CALUDE_system_solution_and_arithmetic_progression_l2556_255676

-- Define the system of equations
def system (m a b c x y z : ℝ) : Prop :=
  x + y + m*z = a ∧ x + m*y + z = b ∧ m*x + y + z = c

-- Theorem statement
theorem system_solution_and_arithmetic_progression
  (m a b c : ℝ) :
  (∃! (x y z : ℝ), system m a b c x y z) ↔ 
    (m ≠ -2 ∧ m ≠ 1) ∧
  (∀ (x y z : ℝ), system m a b c x y z → (2*y = x + z) ↔ a + c = b) :=
sorry

end NUMINAMATH_CALUDE_system_solution_and_arithmetic_progression_l2556_255676


namespace NUMINAMATH_CALUDE_boat_license_count_l2556_255658

def boat_license_options : ℕ :=
  let letter_options := 3  -- A, M, or B
  let digit_options := 10  -- 0 to 9
  let digit_positions := 5
  letter_options * digit_options ^ digit_positions

theorem boat_license_count : boat_license_options = 300000 := by
  sorry

end NUMINAMATH_CALUDE_boat_license_count_l2556_255658


namespace NUMINAMATH_CALUDE_initial_crayons_l2556_255681

theorem initial_crayons (initial final added : ℕ) : 
  final = initial + added → 
  added = 6 → 
  final = 13 → 
  initial = 7 := by 
sorry

end NUMINAMATH_CALUDE_initial_crayons_l2556_255681


namespace NUMINAMATH_CALUDE_product_of_four_sqrt_expressions_l2556_255664

theorem product_of_four_sqrt_expressions : 
  let a := Real.sqrt (2 - Real.sqrt 3)
  let b := Real.sqrt (2 - Real.sqrt (2 - Real.sqrt 3))
  let c := Real.sqrt (2 - Real.sqrt (2 - Real.sqrt (2 - Real.sqrt 3)))
  let d := Real.sqrt (2 + Real.sqrt (2 - Real.sqrt (2 - Real.sqrt 3)))
  a * b * c * d = 1 := by sorry

end NUMINAMATH_CALUDE_product_of_four_sqrt_expressions_l2556_255664


namespace NUMINAMATH_CALUDE_cake_pieces_theorem_l2556_255657

/-- The initial number of cake pieces -/
def initial_pieces : ℕ := 240

/-- The percentage of cake pieces eaten -/
def eaten_percentage : ℚ := 60 / 100

/-- The number of people who received the remaining pieces -/
def num_recipients : ℕ := 3

/-- The number of pieces each recipient received -/
def pieces_per_recipient : ℕ := 32

/-- Theorem stating that the initial number of cake pieces is correct -/
theorem cake_pieces_theorem :
  initial_pieces * (1 - eaten_percentage) = num_recipients * pieces_per_recipient := by
  sorry

end NUMINAMATH_CALUDE_cake_pieces_theorem_l2556_255657


namespace NUMINAMATH_CALUDE_sum_of_roots_l2556_255611

theorem sum_of_roots (h b x₁ x₂ : ℝ) 
  (hx : x₁ ≠ x₂) 
  (eq₁ : 3 * x₁^2 - h * x₁ = b) 
  (eq₂ : 3 * x₂^2 - h * x₂ = b) : 
  x₁ + x₂ = h / 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l2556_255611


namespace NUMINAMATH_CALUDE_amy_cupcakes_l2556_255643

theorem amy_cupcakes (todd_ate : ℕ) (packages : ℕ) (cupcakes_per_package : ℕ) 
  (h1 : todd_ate = 5)
  (h2 : packages = 9)
  (h3 : cupcakes_per_package = 5) :
  todd_ate + packages * cupcakes_per_package = 50 := by
  sorry

end NUMINAMATH_CALUDE_amy_cupcakes_l2556_255643


namespace NUMINAMATH_CALUDE_freds_walking_speed_l2556_255636

/-- Proves that Fred's walking speed is 4 miles per hour given the initial conditions -/
theorem freds_walking_speed 
  (initial_distance : ℝ) 
  (sams_speed : ℝ) 
  (sams_distance : ℝ) 
  (h1 : initial_distance = 40) 
  (h2 : sams_speed = 4) 
  (h3 : sams_distance = 20) : 
  (initial_distance - sams_distance) / (sams_distance / sams_speed) = 4 :=
by sorry

end NUMINAMATH_CALUDE_freds_walking_speed_l2556_255636


namespace NUMINAMATH_CALUDE_point_on_x_axis_point_in_second_quadrant_equal_distance_l2556_255609

-- Define the point P as a function of a
def P (a : ℝ) : ℝ × ℝ := (2*a - 3, a + 6)

-- Part 1
theorem point_on_x_axis (a : ℝ) : 
  P a = (-15, 0) ↔ (P a).2 = 0 :=
sorry

-- Part 2
theorem point_in_second_quadrant_equal_distance (a : ℝ) :
  (P a).1 < 0 ∧ (P a).2 > 0 ∧ abs (P a).1 = (P a).2 → a^2003 + 2024 = 2023 :=
sorry

end NUMINAMATH_CALUDE_point_on_x_axis_point_in_second_quadrant_equal_distance_l2556_255609


namespace NUMINAMATH_CALUDE_image_of_negative_two_three_preimage_of_two_negative_three_l2556_255619

-- Define the mapping f
def f (x y : ℝ) : ℝ × ℝ := (x + y, x * y)

-- Theorem for the image of (-2, 3)
theorem image_of_negative_two_three :
  f (-2) 3 = (1, -6) := by sorry

-- Theorem for the preimage of (2, -3)
theorem preimage_of_two_negative_three :
  {p : ℝ × ℝ | f p.1 p.2 = (2, -3)} = {(-1, 3), (3, -1)} := by sorry

end NUMINAMATH_CALUDE_image_of_negative_two_three_preimage_of_two_negative_three_l2556_255619


namespace NUMINAMATH_CALUDE_unique_solution_cubic_system_l2556_255601

theorem unique_solution_cubic_system (x y z : ℝ) :
  x^3 + y = z^2 ∧ y^3 + z = x^2 ∧ z^3 + x = y^2 → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_cubic_system_l2556_255601


namespace NUMINAMATH_CALUDE_cat_shelter_ratio_l2556_255612

theorem cat_shelter_ratio : 
  ∀ (initial_cats replacement_cats adopted_cats dogs : ℕ),
    initial_cats = 15 →
    adopted_cats = initial_cats / 3 →
    initial_cats = initial_cats - adopted_cats + replacement_cats →
    dogs = 2 * initial_cats →
    initial_cats + dogs + replacement_cats = 60 →
    replacement_cats / adopted_cats = 3 ∧ adopted_cats ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_cat_shelter_ratio_l2556_255612


namespace NUMINAMATH_CALUDE_continued_fraction_convergents_l2556_255617

/-- Continued fraction convergents -/
theorem continued_fraction_convergents
  (P Q : ℕ → ℤ)  -- Sequences of numerators and denominators
  (a : ℕ → ℕ)    -- Sequence of continued fraction coefficients
  (h1 : ∀ k, k ≥ 2 → P k = a k * P (k-1) + P (k-2))
  (h2 : ∀ k, k ≥ 2 → Q k = a k * Q (k-1) + Q (k-2))
  (h3 : ∀ k, a k > 0) :
  (∀ k, k ≥ 2 → P k * Q (k-2) - P (k-2) * Q k = (-1)^k * a k) ∧
  (∀ k, k ≥ 1 → (P k : ℚ) / Q k - (P (k-1) : ℚ) / Q (k-1) = (-1)^(k+1) / (Q k * Q (k-1))) ∧
  (∀ n, n ≥ 1 → ∀ k, 1 ≤ k → k < n → Q k < Q (k+1)) ∧
  (∀ n, n ≥ 0 → (P 0 : ℚ) / Q 0 < (P 2 : ℚ) / Q 2 ∧ 
    (P n : ℚ) / Q n < (P (n+1) : ℚ) / Q (n+1) ∧
    (P (n+2) : ℚ) / Q (n+2) < (P (n+1) : ℚ) / Q (n+1)) ∧
  (∀ k l, k ≥ 0 → l ≥ 0 → (P (2*k) : ℚ) / Q (2*k) < (P (2*l+1) : ℚ) / Q (2*l+1)) :=
by sorry

end NUMINAMATH_CALUDE_continued_fraction_convergents_l2556_255617


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2556_255666

theorem inequality_system_solution (m : ℝ) : 
  (∃ x : ℝ, (x - m) / 2 ≥ 2 ∧ x - 4 ≤ 3 * (x - 2)) ∧ 
  (∀ x : ℤ, x < 2 → ¬((x - m) / 2 ≥ 2 ∧ x - 4 ≤ 3 * (x - 2))) →
  -3 < m ∧ m ≤ -2 := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2556_255666


namespace NUMINAMATH_CALUDE_distinct_arrangements_l2556_255644

theorem distinct_arrangements (n : ℕ) (h : n = 8) : Nat.factorial n = 40320 := by
  sorry

end NUMINAMATH_CALUDE_distinct_arrangements_l2556_255644


namespace NUMINAMATH_CALUDE_probability_of_two_specific_stamps_l2556_255600

/-- Represents a set of four distinct stamps -/
def Stamps : Type := Fin 4

/-- The number of ways to choose 2 stamps from 4 stamps -/
def total_combinations : ℕ := Nat.choose 4 2

/-- The number of ways to choose the specific 2 stamps we want -/
def favorable_combinations : ℕ := 1

/-- Theorem: The probability of drawing exactly two specific stamps from a set of four stamps is 1/6 -/
theorem probability_of_two_specific_stamps : 
  (favorable_combinations : ℚ) / total_combinations = 1 / 6 := by
  sorry


end NUMINAMATH_CALUDE_probability_of_two_specific_stamps_l2556_255600


namespace NUMINAMATH_CALUDE_negation_of_existence_quadratic_inequality_negation_l2556_255602

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x, p x) ↔ (∀ x, ¬ p x) := by sorry

theorem quadratic_inequality_negation :
  (¬ ∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_quadratic_inequality_negation_l2556_255602


namespace NUMINAMATH_CALUDE_artist_painting_difference_l2556_255685

/-- Given an artist's painting schedule over three months, prove that the difference
    between the number of pictures painted in July and June is zero. -/
theorem artist_painting_difference (june july august total : ℕ) 
    (h_june : june = 2)
    (h_august : august = 9)
    (h_total : total = 13)
    (h_sum : june + july + august = total) : july - june = 0 := by
  sorry

end NUMINAMATH_CALUDE_artist_painting_difference_l2556_255685


namespace NUMINAMATH_CALUDE_min_x_coeff_for_restricted_poly_with_specific_value_l2556_255653

/-- A polynomial with coefficients from the set {0,1,2,3,4,5} -/
def RestrictedPolynomial (P : Polynomial ℤ) : Prop :=
  ∀ i, (P.coeff i) ∈ ({0, 1, 2, 3, 4, 5} : Set ℤ)

/-- The theorem stating that if P(6) = 2013 for a restricted polynomial P,
    then the coefficient of x in P is at least 5 -/
theorem min_x_coeff_for_restricted_poly_with_specific_value
  (P : Polynomial ℤ) (h : RestrictedPolynomial P) (h2 : P.eval 6 = 2013) :
  P.coeff 1 ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_min_x_coeff_for_restricted_poly_with_specific_value_l2556_255653


namespace NUMINAMATH_CALUDE_all_solutions_are_powers_l2556_255669

-- Define the equation (1) as a predicate
def is_solution (p q : ℤ) : Prop := sorry

-- Define the main theorem
theorem all_solutions_are_powers (p q : ℤ) :
  p ≥ 0 ∧ q ≥ 0 ∧ is_solution p q ↔ ∃ n : ℕ, p + q * Real.sqrt 5 = (9 + 4 * Real.sqrt 5) ^ n := by
  sorry

end NUMINAMATH_CALUDE_all_solutions_are_powers_l2556_255669


namespace NUMINAMATH_CALUDE_park_wheels_count_l2556_255631

/-- The total number of wheels on bikes in a park -/
def total_wheels (regular_bikes children_bikes tandem_4_wheels tandem_6_wheels : ℕ) : ℕ :=
  regular_bikes * 2 + children_bikes * 4 + tandem_4_wheels * 4 + tandem_6_wheels * 6

/-- Theorem: The total number of wheels in the park is 96 -/
theorem park_wheels_count :
  total_wheels 7 11 5 3 = 96 := by
  sorry

end NUMINAMATH_CALUDE_park_wheels_count_l2556_255631


namespace NUMINAMATH_CALUDE_inverse_f_69_l2556_255680

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^3 + 6

-- State the theorem
theorem inverse_f_69 : f⁻¹ 69 = (21 : ℝ)^(1/3) := by sorry

end NUMINAMATH_CALUDE_inverse_f_69_l2556_255680


namespace NUMINAMATH_CALUDE_devin_taught_calculus_four_years_l2556_255616

/-- Represents the number of years Devin taught each subject --/
structure TeachingYears where
  calculus : ℕ
  algebra : ℕ
  statistics : ℕ

/-- Defines the conditions of Devin's teaching career --/
def satisfiesConditions (y : TeachingYears) : Prop :=
  y.algebra = 2 * y.calculus ∧
  y.statistics = 5 * y.algebra ∧
  y.calculus + y.algebra + y.statistics = 52

/-- Theorem stating that given the conditions, Devin taught Calculus for 4 years --/
theorem devin_taught_calculus_four_years :
  ∃ y : TeachingYears, satisfiesConditions y ∧ y.calculus = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_devin_taught_calculus_four_years_l2556_255616


namespace NUMINAMATH_CALUDE_independence_of_phi_l2556_255651

theorem independence_of_phi (α φ : ℝ) : 
  4 * Real.cos α * Real.cos φ * Real.cos (α - φ) + 2 * Real.sin (α - φ)^2 - Real.cos (2 * φ) = Real.cos (2 * α) + 2 := by
  sorry

end NUMINAMATH_CALUDE_independence_of_phi_l2556_255651
