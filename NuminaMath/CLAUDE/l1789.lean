import Mathlib

namespace NUMINAMATH_CALUDE_multiplier_is_five_l1789_178923

/-- Given a number that equals some times the difference between itself and 4,
    prove that when the number is 5, the multiplier is also 5. -/
theorem multiplier_is_five (n m : ℝ) : n = m * (n - 4) → n = 5 → m = 5 := by
  sorry

end NUMINAMATH_CALUDE_multiplier_is_five_l1789_178923


namespace NUMINAMATH_CALUDE_root_property_l1789_178940

theorem root_property (a : ℝ) (h : 2 * a^2 - 3 * a - 5 = 0) : -4 * a^2 + 6 * a = -10 := by
  sorry

end NUMINAMATH_CALUDE_root_property_l1789_178940


namespace NUMINAMATH_CALUDE_extreme_value_implies_parameters_l1789_178951

/-- The function f with parameters a and b -/
def f (a b x : ℝ) : ℝ := x^3 - a*x^2 - b*x + a^2

/-- Theorem stating that if f has an extreme value of 10 at x=1, then (a,b) = (-4,11) -/
theorem extreme_value_implies_parameters
  (a b : ℝ)
  (extreme_value : f a b 1 = 10)
  (is_extreme : ∀ x, f a b x ≤ f a b 1) :
  a = -4 ∧ b = 11 := by
sorry

end NUMINAMATH_CALUDE_extreme_value_implies_parameters_l1789_178951


namespace NUMINAMATH_CALUDE_identity_is_unique_solution_l1789_178911

/-- A function satisfying the given functional equation for all real numbers -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x + f (f y)) = 2 * x + f (f y) - f (f x)

/-- The theorem stating that the identity function is the only solution -/
theorem identity_is_unique_solution :
  ∀ f : ℝ → ℝ, FunctionalEquation f → (∀ x : ℝ, f x = x) :=
sorry

end NUMINAMATH_CALUDE_identity_is_unique_solution_l1789_178911


namespace NUMINAMATH_CALUDE_trigonometric_expression_equals_one_l1789_178900

theorem trigonometric_expression_equals_one : 
  (Real.sin (20 * π / 180) * Real.cos (10 * π / 180) + 
   Real.cos (160 * π / 180) * Real.cos (100 * π / 180)) / 
  (Real.sin (24 * π / 180) * Real.cos (6 * π / 180) + 
   Real.cos (156 * π / 180) * Real.cos (96 * π / 180)) = 1 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equals_one_l1789_178900


namespace NUMINAMATH_CALUDE_shoes_total_price_l1789_178990

/-- Given the conditions of Jeff's purchase, prove the total price of shoes. -/
theorem shoes_total_price (total_cost : ℕ) (shoe_pairs : ℕ) (jerseys : ℕ) 
  (h1 : total_cost = 560)
  (h2 : shoe_pairs = 6)
  (h3 : jerseys = 4)
  (h4 : ∃ (shoe_price : ℚ), total_cost = shoe_pairs * shoe_price + jerseys * (shoe_price / 4)) :
  shoe_pairs * (total_cost / (shoe_pairs + jerseys / 4 : ℚ)) = 480 := by
  sorry

#check shoes_total_price

end NUMINAMATH_CALUDE_shoes_total_price_l1789_178990


namespace NUMINAMATH_CALUDE_clock_equivalent_square_l1789_178989

theorem clock_equivalent_square : 
  ∃ (h : ℕ), h > 10 ∧ h ≤ 12 ∧ (h - h^2) % 12 = 0 ∧ 
  ∀ (k : ℕ), k > 10 ∧ k < h → (k - k^2) % 12 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_clock_equivalent_square_l1789_178989


namespace NUMINAMATH_CALUDE_warehouse_boxes_count_l1789_178964

/-- Given an initial number of boxes and a process of filling some boxes with more boxes,
    calculate the total number of boxes after the process is complete. -/
def total_boxes (initial : ℕ) (fill_amount : ℕ) (final_non_empty : ℕ) : ℕ :=
  initial + (initial * fill_amount) + ((final_non_empty - initial) * fill_amount)

/-- Theorem stating that given the specific conditions of the problem,
    the total number of boxes is 77. -/
theorem warehouse_boxes_count :
  total_boxes 7 7 10 = 77 := by
  sorry

#eval total_boxes 7 7 10

end NUMINAMATH_CALUDE_warehouse_boxes_count_l1789_178964


namespace NUMINAMATH_CALUDE_digit_difference_in_base_d_l1789_178910

/-- Represents a digit in a given base -/
def Digit (d : ℕ) := {n : ℕ // n < d}

/-- Converts a two-digit number in base d to its decimal representation -/
def toDecimal (d : ℕ) (tens : Digit d) (ones : Digit d) : ℕ :=
  d * tens.val + ones.val

theorem digit_difference_in_base_d 
  (d : ℕ) (hd : d > 8) 
  (C : Digit d) (D : Digit d) 
  (h : toDecimal d C D + toDecimal d C C = d * d + 5 * d + 3) :
  C.val - D.val = 1 := by
sorry

end NUMINAMATH_CALUDE_digit_difference_in_base_d_l1789_178910


namespace NUMINAMATH_CALUDE_twice_x_greater_than_five_l1789_178995

theorem twice_x_greater_than_five (x : ℝ) : (2 * x > 5) ↔ (2 * x > 5) := by sorry

end NUMINAMATH_CALUDE_twice_x_greater_than_five_l1789_178995


namespace NUMINAMATH_CALUDE_smallest_n_for_inequality_l1789_178979

theorem smallest_n_for_inequality : ∃ (n : ℕ), n = 2 ∧ 
  (∀ (k : ℕ), k < n → (10 : ℝ) ^ (2 ^ (k + 1)) < 1000) ∧
  (10 : ℝ) ^ (2 ^ (n + 1)) ≥ 1000 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_inequality_l1789_178979


namespace NUMINAMATH_CALUDE_reciprocal_sum_of_roots_l1789_178953

theorem reciprocal_sum_of_roots (γ δ : ℝ) : 
  (∃ r s : ℝ, 6 * r^2 - 11 * r + 7 = 0 ∧ 
              6 * s^2 - 11 * s + 7 = 0 ∧ 
              γ = 1 / r ∧ 
              δ = 1 / s) → 
  γ + δ = 11 / 7 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_sum_of_roots_l1789_178953


namespace NUMINAMATH_CALUDE_restaurant_hiring_l1789_178973

theorem restaurant_hiring (initial_ratio_cooks initial_ratio_waiters new_ratio_cooks new_ratio_waiters num_cooks : ℕ) 
  (h1 : initial_ratio_cooks = 3)
  (h2 : initial_ratio_waiters = 10)
  (h3 : new_ratio_cooks = 3)
  (h4 : new_ratio_waiters = 14)
  (h5 : num_cooks = 9) :
  ∃ (initial_waiters hired_waiters : ℕ),
    initial_ratio_cooks * initial_waiters = initial_ratio_waiters * num_cooks ∧
    new_ratio_cooks * (initial_waiters + hired_waiters) = new_ratio_waiters * num_cooks ∧
    hired_waiters = 12 := by
  sorry


end NUMINAMATH_CALUDE_restaurant_hiring_l1789_178973


namespace NUMINAMATH_CALUDE_line_through_center_line_bisecting_chord_l1789_178960

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 16

-- Define point P
def point_P : ℝ × ℝ := (2, 2)

-- Define the equation of line l passing through P and center of C
def line_l_through_center (x y : ℝ) : Prop := 2*x - y - 2 = 0

-- Define the equation of line l bisecting chord AB
def line_l_bisecting_chord (x y : ℝ) : Prop := x + 2*y - 6 = 0

-- Theorem 1: Line l passing through P and center of C
theorem line_through_center : 
  ∀ x y : ℝ, circle_C x y → line_l_through_center x y → 
  ∃ t : ℝ, x = 2 + t ∧ y = 2 + 2*t :=
sorry

-- Theorem 2: Line l passing through P and bisecting chord AB
theorem line_bisecting_chord :
  ∀ x y : ℝ, circle_C x y → line_l_bisecting_chord x y →
  ∃ t : ℝ, x = 2 + t ∧ y = 2 - t/2 :=
sorry

end NUMINAMATH_CALUDE_line_through_center_line_bisecting_chord_l1789_178960


namespace NUMINAMATH_CALUDE_common_chord_equation_l1789_178994

/-- The equation of the common chord of two circles -/
def common_chord (c1 c2 : ℝ × ℝ → Prop) : ℝ × ℝ → Prop :=
  fun p => c1 p ∧ c2 p

/-- First circle equation -/
def circle1 : ℝ × ℝ → Prop :=
  fun (x, y) => x^2 + y^2 + 2*x = 0

/-- Second circle equation -/
def circle2 : ℝ × ℝ → Prop :=
  fun (x, y) => x^2 + y^2 - 4*y = 0

/-- The proposed common chord equation -/
def proposed_chord : ℝ × ℝ → Prop :=
  fun (x, y) => x + 2*y = 0

theorem common_chord_equation :
  common_chord circle1 circle2 = proposed_chord := by
  sorry

end NUMINAMATH_CALUDE_common_chord_equation_l1789_178994


namespace NUMINAMATH_CALUDE_alicia_sundae_cost_l1789_178949

/-- The cost of Alicia's peanut butter sundae given the prices of other sundaes and the final bill with tip -/
theorem alicia_sundae_cost (yvette_sundae brant_sundae josh_sundae : ℚ)
  (tip_percentage : ℚ) (final_bill : ℚ) :
  yvette_sundae = 9 →
  brant_sundae = 10 →
  josh_sundae = (17/2) →
  tip_percentage = (1/5) →
  final_bill = 42 →
  ∃ (alicia_sundae : ℚ),
    alicia_sundae = (final_bill / (1 + tip_percentage)) - (yvette_sundae + brant_sundae + josh_sundae) ∧
    alicia_sundae = (15/2) := by
  sorry

end NUMINAMATH_CALUDE_alicia_sundae_cost_l1789_178949


namespace NUMINAMATH_CALUDE_smallest_n_for_candy_l1789_178967

theorem smallest_n_for_candy (n : ℕ) : (∀ k : ℕ, k > 0 ∧ k < n → ¬(10 ∣ 25*k ∧ 18 ∣ 25*k ∧ 20 ∣ 25*k)) ∧ 
                                       (10 ∣ 25*n ∧ 18 ∣ 25*n ∧ 20 ∣ 25*n) → 
                                       n = 16 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_candy_l1789_178967


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l1789_178909

def inequality_system (x : ℝ) : Prop :=
  x^2 - 2*x - 3 > 0 ∧ -x^2 - 3*x + 4 ≥ 0

theorem inequality_system_solution_set :
  {x : ℝ | inequality_system x} = {x : ℝ | -4 ≤ x ∧ x < -1} := by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l1789_178909


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1789_178996

theorem polynomial_factorization (m : ℤ) : 
  (∀ x : ℝ, x^2 - 5*x + m = (x - 3) * (x - 2)) → m = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1789_178996


namespace NUMINAMATH_CALUDE_fractional_equation_solution_range_l1789_178998

theorem fractional_equation_solution_range (x a : ℝ) : 
  (1 / (x + 3) - 1 = a / (x + 3)) → -- Given equation
  (x < 0) → -- Solution for x is negative
  (a > -2 ∧ a ≠ 1) -- Range of a
  :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_range_l1789_178998


namespace NUMINAMATH_CALUDE_not_prime_sum_product_l1789_178926

theorem not_prime_sum_product (a b c d : ℕ) 
  (h_pos : 0 < d ∧ d < c ∧ c < b ∧ b < a) 
  (h_eq : a * c + b * d = (b + d - a + c) * (b + d + a - c)) : 
  ¬ Nat.Prime (a * b + c * d) := by
sorry

end NUMINAMATH_CALUDE_not_prime_sum_product_l1789_178926


namespace NUMINAMATH_CALUDE_reflected_ray_equation_l1789_178981

/-- The equation of a reflected ray given an incident ray and a reflecting line. -/
theorem reflected_ray_equation (x y : ℝ) :
  (y = 2 * x + 1) →  -- incident ray
  (y = x) →          -- reflecting line
  (x - 2 * y - 1 = 0) -- reflected ray
  := by sorry

end NUMINAMATH_CALUDE_reflected_ray_equation_l1789_178981


namespace NUMINAMATH_CALUDE_units_digit_47_power_47_l1789_178931

theorem units_digit_47_power_47 : 47^47 % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_47_power_47_l1789_178931


namespace NUMINAMATH_CALUDE_line_relationships_l1789_178919

-- Define the slopes of the lines
def slope1 : ℚ := 2
def slope2 : ℚ := 3
def slope3 : ℚ := 2
def slope4 : ℚ := 3/2
def slope5 : ℚ := 1/2

-- Define a function to check if two slopes are parallel
def are_parallel (m1 m2 : ℚ) : Prop := m1 = m2

-- Define a function to check if two slopes are perpendicular
def are_perpendicular (m1 m2 : ℚ) : Prop := m1 * m2 = -1

-- Define the list of all slopes
def slopes : List ℚ := [slope1, slope2, slope3, slope4, slope5]

-- Theorem statement
theorem line_relationships :
  (∃! (i j : Fin 5), i < j ∧ are_parallel (slopes.get i) (slopes.get j)) ∧
  (∀ (i j : Fin 5), i < j → ¬are_perpendicular (slopes.get i) (slopes.get j)) :=
sorry

end NUMINAMATH_CALUDE_line_relationships_l1789_178919


namespace NUMINAMATH_CALUDE_sum_areas_tangent_circles_l1789_178921

/-- Three mutually externally tangent circles whose centers form a 5-12-13 right triangle -/
structure TangentCircles where
  /-- Radius of the circle centered at the vertex opposite the side of length 5 -/
  a : ℝ
  /-- Radius of the circle centered at the vertex opposite the side of length 12 -/
  b : ℝ
  /-- Radius of the circle centered at the vertex opposite the side of length 13 -/
  c : ℝ
  /-- The circles are mutually externally tangent -/
  tangent_5 : a + b = 5
  tangent_12 : a + c = 12
  tangent_13 : b + c = 13

/-- The sum of the areas of three mutually externally tangent circles 
    whose centers form a 5-12-13 right triangle is 113π -/
theorem sum_areas_tangent_circles (circles : TangentCircles) :
  π * (circles.a^2 + circles.b^2 + circles.c^2) = 113 * π := by
  sorry

end NUMINAMATH_CALUDE_sum_areas_tangent_circles_l1789_178921


namespace NUMINAMATH_CALUDE_f_increasing_on_negative_l1789_178966

def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 - 2 * m * x + 3

theorem f_increasing_on_negative (m : ℝ) :
  (∀ x : ℝ, f m x = f m (-x)) →
  ∀ x y : ℝ, x < y → x < 0 → y ≤ 0 → f m x < f m y :=
sorry

end NUMINAMATH_CALUDE_f_increasing_on_negative_l1789_178966


namespace NUMINAMATH_CALUDE_locus_of_centers_l1789_178950

/-- Circle C₁ with equation x² + y² = 4 -/
def C₁ : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}

/-- Circle C₃ with equation (x-1)² + y² = 25 -/
def C₃ : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + p.2^2 = 25}

/-- A circle is externally tangent to C₁ if the distance between their centers
    is equal to the sum of their radii -/
def externally_tangent_to_C₁ (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  center.1^2 + center.2^2 = (radius + 2)^2

/-- A circle is internally tangent to C₃ if the distance between their centers
    is equal to the difference of their radii -/
def internally_tangent_to_C₃ (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  (center.1 - 1)^2 + center.2^2 = (5 - radius)^2

/-- The locus of centers (a,b) of circles externally tangent to C₁ and
    internally tangent to C₃ satisfies the equation 5a² + 9b² + 80a - 400 = 0 -/
theorem locus_of_centers (a b : ℝ) :
  (∃ r : ℝ, externally_tangent_to_C₁ (a, b) r ∧ internally_tangent_to_C₃ (a, b) r) →
  5 * a^2 + 9 * b^2 + 80 * a - 400 = 0 := by
  sorry

end NUMINAMATH_CALUDE_locus_of_centers_l1789_178950


namespace NUMINAMATH_CALUDE_voice_of_china_sampling_l1789_178955

/-- Systematic sampling function -/
def systematicSample (populationSize : ℕ) (sampleSize : ℕ) (firstSample : ℕ) (n : ℕ) : ℕ :=
  firstSample + (populationSize / sampleSize) * (n - 1)

/-- The Voice of China sampling theorem -/
theorem voice_of_china_sampling :
  let populationSize := 500
  let sampleSize := 20
  let firstSample := 3
  let fifthSample := 5
  systematicSample populationSize sampleSize firstSample fifthSample = 103 := by
sorry

end NUMINAMATH_CALUDE_voice_of_china_sampling_l1789_178955


namespace NUMINAMATH_CALUDE_tan_C_when_a_neg_eight_min_tan_C_l1789_178986

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)

-- Define the condition for tan A and tan B
def roots_condition (t : Triangle) (a : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 + a*x + 4 = 0 ∧ y^2 + a*y + 4 = 0 ∧ 
  x = Real.tan t.A ∧ y = Real.tan t.B

-- Theorem 1: When a = -8, tan C = 8/3
theorem tan_C_when_a_neg_eight (t : Triangle) (a : ℝ) 
  (h : roots_condition t a) (h_a : a = -8) : 
  Real.tan t.C = 8/3 := by sorry

-- Theorem 2: Minimum value of tan C is 4/3, occurring when tan A = tan B = 2
theorem min_tan_C (t : Triangle) (a : ℝ) 
  (h : roots_condition t a) : 
  ∃ (t_min : Triangle), 
    (∀ t' : Triangle, roots_condition t' a → Real.tan t_min.C ≤ Real.tan t'.C) ∧
    Real.tan t_min.C = 4/3 ∧ 
    Real.tan t_min.A = 2 ∧ 
    Real.tan t_min.B = 2 := by sorry

end NUMINAMATH_CALUDE_tan_C_when_a_neg_eight_min_tan_C_l1789_178986


namespace NUMINAMATH_CALUDE_no_real_sqrt_negative_number_l1789_178920

theorem no_real_sqrt_negative_number (x : ℝ) :
  x = -2.5 ∨ x = 0 ∨ x = 2.1 ∨ x = 6 →
  (∃ y : ℝ, y ^ 2 = x) ↔ x ≠ -2.5 :=
by sorry

end NUMINAMATH_CALUDE_no_real_sqrt_negative_number_l1789_178920


namespace NUMINAMATH_CALUDE_second_derivative_of_cosine_at_pi_third_l1789_178952

open Real

theorem second_derivative_of_cosine_at_pi_third :
  let f : ℝ → ℝ := fun x ↦ cos x
  (deriv (deriv f)) (π / 3) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_second_derivative_of_cosine_at_pi_third_l1789_178952


namespace NUMINAMATH_CALUDE_profit_calculation_l1789_178935

/-- The number of pencils purchased -/
def total_pencils : ℕ := 2000

/-- The purchase price per pencil in dollars -/
def purchase_price : ℚ := 1/5

/-- The selling price per pencil in dollars -/
def selling_price : ℚ := 2/5

/-- The desired profit in dollars -/
def desired_profit : ℚ := 160

/-- The number of pencils that must be sold to achieve the desired profit -/
def pencils_to_sell : ℕ := 1400

theorem profit_calculation :
  (pencils_to_sell : ℚ) * selling_price - (total_pencils : ℚ) * purchase_price = desired_profit :=
sorry

end NUMINAMATH_CALUDE_profit_calculation_l1789_178935


namespace NUMINAMATH_CALUDE_equation_solution_l1789_178902

theorem equation_solution :
  ∃ (a b c d : ℚ),
    a^2 + b^2 + c^2 + d^2 - a*b - b*c - c*d - d + 2/5 = 0 ∧
    a = 1/5 ∧ b = 2/5 ∧ c = 3/5 ∧ d = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1789_178902


namespace NUMINAMATH_CALUDE_systematic_sampling_correspondence_l1789_178943

/-- Represents a systematic sampling of students. -/
structure SystematicSampling where
  total_students : Nat
  num_groups : Nat
  students_per_group : Nat
  selected_student : Nat
  selected_group : Nat

/-- Theorem stating the relationship between selected students in different groups. -/
theorem systematic_sampling_correspondence
  (s : SystematicSampling)
  (h1 : s.total_students = 60)
  (h2 : s.num_groups = 5)
  (h3 : s.students_per_group = s.total_students / s.num_groups)
  (h4 : s.selected_student = 16)
  (h5 : s.selected_group = 2)
  : (s.selected_student - (s.selected_group - 1) * s.students_per_group) + 3 * s.students_per_group = 40 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_correspondence_l1789_178943


namespace NUMINAMATH_CALUDE_base5_2202_equals_base10_302_l1789_178906

/-- Converts a base 5 digit to its base 10 equivalent --/
def base5ToBase10 (digit : Nat) (position : Nat) : Nat :=
  digit * (5 ^ position)

/-- Theorem: The base 5 number 2202₅ is equal to the base 10 number 302 --/
theorem base5_2202_equals_base10_302 :
  base5ToBase10 2 3 + base5ToBase10 2 2 + base5ToBase10 0 1 + base5ToBase10 2 0 = 302 := by
  sorry

end NUMINAMATH_CALUDE_base5_2202_equals_base10_302_l1789_178906


namespace NUMINAMATH_CALUDE_exam_results_l1789_178922

/-- Represents the score distribution of students in an examination. -/
structure ScoreDistribution where
  scores : List (Nat × Nat)
  total_students : Nat
  sum_scores : Nat

/-- The given score distribution for the examination. -/
def exam_distribution : ScoreDistribution := {
  scores := [(95, 10), (85, 30), (75, 40), (65, 45), (55, 20), (45, 15)],
  total_students := 160,
  sum_scores := 11200
}

/-- Calculate the average score from a ScoreDistribution. -/
def average_score (d : ScoreDistribution) : Rat :=
  d.sum_scores / d.total_students

/-- Calculate the percentage of students scoring at least 60%. -/
def percentage_passing (d : ScoreDistribution) : Rat :=
  let passing_students := (d.scores.filter (fun p => p.fst ≥ 60)).map (fun p => p.snd) |>.sum
  (passing_students * 100) / d.total_students

theorem exam_results :
  average_score exam_distribution = 70 ∧
  percentage_passing exam_distribution = 78125 / 1000 := by
  sorry

#eval average_score exam_distribution
#eval percentage_passing exam_distribution

end NUMINAMATH_CALUDE_exam_results_l1789_178922


namespace NUMINAMATH_CALUDE_division_rebus_proof_l1789_178901

theorem division_rebus_proof :
  -- Given conditions
  let dividend : ℕ := 1089708
  let divisor : ℕ := 12
  let quotient : ℕ := 90809

  -- Divisor is a two-digit number
  (10 ≤ divisor) ∧ (divisor < 100) →
  
  -- When divisor is multiplied by 8, it results in a two-digit number
  (10 ≤ divisor * 8) ∧ (divisor * 8 < 100) →
  
  -- When divisor is multiplied by the first (or last) digit of quotient, it results in a three-digit number
  (100 ≤ divisor * (quotient / 10000)) ∧ (divisor * (quotient / 10000) < 1000) →
  
  -- Quotient has 5 digits
  (10000 ≤ quotient) ∧ (quotient < 100000) →
  
  -- Second and fourth digits of quotient are 0
  (quotient % 10000 / 1000 = 0) ∧ (quotient % 100 / 10 = 0) →
  
  -- The division problem has a unique solution
  ∃! (d q : ℕ), d * q = dividend ∧ q = quotient →
  
  -- Prove that the division is correct
  dividend / divisor = quotient ∧ dividend % divisor = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_division_rebus_proof_l1789_178901


namespace NUMINAMATH_CALUDE_solution_xy_l1789_178970

theorem solution_xy : ∃ (x y : ℝ), 
  (x + 2*y = (7 - x) + (7 - 2*y)) ∧ 
  (x - y = 2*(x - 2) - (y - 3)) ∧ 
  (x = 1) ∧ (y = 3) := by
  sorry

end NUMINAMATH_CALUDE_solution_xy_l1789_178970


namespace NUMINAMATH_CALUDE_sequence_2009th_term_l1789_178974

theorem sequence_2009th_term :
  let sequence : ℕ → ℕ := fun n => 2^(n - 1)
  sequence 2009 = 2^2008 := by
  sorry

end NUMINAMATH_CALUDE_sequence_2009th_term_l1789_178974


namespace NUMINAMATH_CALUDE_find_T_l1789_178912

theorem find_T (S : ℚ) (T : ℚ) (h1 : (1/4) * (1/6) * T = (1/2) * (1/8) * S) (h2 : S = 64) :
  T = 96 := by
  sorry

end NUMINAMATH_CALUDE_find_T_l1789_178912


namespace NUMINAMATH_CALUDE_prime_sum_square_fourth_power_l1789_178963

theorem prime_sum_square_fourth_power :
  ∀ p q r : ℕ,
  Prime p → Prime q → Prime r →
  p + q^2 = r^4 →
  p = 7 ∧ q = 3 ∧ r = 2 :=
by sorry

end NUMINAMATH_CALUDE_prime_sum_square_fourth_power_l1789_178963


namespace NUMINAMATH_CALUDE_antibiotics_cost_proof_l1789_178948

def antibiotics_problem (doses_per_day : ℕ) (days : ℕ) (total_cost : ℚ) : ℚ :=
  total_cost / (doses_per_day * days)

theorem antibiotics_cost_proof (doses_per_day : ℕ) (days : ℕ) (total_cost : ℚ) 
  (h1 : doses_per_day = 3)
  (h2 : days = 7)
  (h3 : total_cost = 63) :
  antibiotics_problem doses_per_day days total_cost = 3 := by
sorry

end NUMINAMATH_CALUDE_antibiotics_cost_proof_l1789_178948


namespace NUMINAMATH_CALUDE_roots_sum_of_sixth_powers_l1789_178941

theorem roots_sum_of_sixth_powers (r s : ℝ) : 
  r^2 - 2*r*Real.sqrt 3 + 1 = 0 →
  s^2 - 2*s*Real.sqrt 3 + 1 = 0 →
  r ≠ s →
  r^6 + s^6 = 970 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_of_sixth_powers_l1789_178941


namespace NUMINAMATH_CALUDE_square_difference_l1789_178982

theorem square_difference (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l1789_178982


namespace NUMINAMATH_CALUDE_renata_final_amount_l1789_178937

/-- Represents Renata's financial transactions --/
def renataTransactions : List Int :=
  [10, -4, 90, -50, -10, -5, -1, -1, 65]

/-- Calculates the final amount Renata has --/
def finalAmount (transactions : List Int) : Int :=
  transactions.sum

/-- Theorem stating that Renata ends up with $94 --/
theorem renata_final_amount :
  finalAmount renataTransactions = 94 := by
  sorry

#eval finalAmount renataTransactions

end NUMINAMATH_CALUDE_renata_final_amount_l1789_178937


namespace NUMINAMATH_CALUDE_dodge_trucks_count_l1789_178978

/-- Represents the number of vehicles of each type in the parking lot -/
structure ParkingLot where
  dodge : ℚ
  ford : ℚ
  toyota : ℚ
  nissan : ℚ
  volkswagen : ℚ
  honda : ℚ
  mazda : ℚ
  chevrolet : ℚ
  subaru : ℚ
  fiat : ℚ

/-- The conditions of the parking lot -/
def validParkingLot (p : ParkingLot) : Prop :=
  p.ford = (1/3) * p.dodge ∧
  p.ford = 2 * p.toyota ∧
  p.toyota = (7/9) * p.nissan ∧
  p.volkswagen = (1/2) * p.toyota ∧
  p.honda = (3/4) * p.ford ∧
  p.mazda = (2/5) * p.nissan ∧
  p.chevrolet = (2/3) * p.honda ∧
  p.subaru = 4 * p.dodge ∧
  p.fiat = (1/2) * p.mazda ∧
  p.volkswagen = 5

theorem dodge_trucks_count (p : ParkingLot) (h : validParkingLot p) : p.dodge = 60 := by
  sorry

end NUMINAMATH_CALUDE_dodge_trucks_count_l1789_178978


namespace NUMINAMATH_CALUDE_product_remainder_l1789_178976

theorem product_remainder (a b m : ℕ) (ha : a % m = 7) (hb : b % m = 1) (hm : m = 8) :
  (a * b) % m = 7 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_l1789_178976


namespace NUMINAMATH_CALUDE_stormi_car_wash_price_l1789_178905

/-- The amount Stormi charges for washing each car -/
def car_wash_price : ℝ := 10

/-- The number of cars Stormi washes -/
def num_cars : ℕ := 3

/-- The price Stormi charges for mowing a lawn -/
def lawn_mow_price : ℝ := 13

/-- The number of lawns Stormi mows -/
def num_lawns : ℕ := 2

/-- The cost of the bicycle -/
def bicycle_cost : ℝ := 80

/-- The additional amount Stormi needs to afford the bicycle -/
def additional_amount_needed : ℝ := 24

theorem stormi_car_wash_price :
  car_wash_price * num_cars + lawn_mow_price * num_lawns = bicycle_cost - additional_amount_needed :=
sorry

end NUMINAMATH_CALUDE_stormi_car_wash_price_l1789_178905


namespace NUMINAMATH_CALUDE_correct_emu_count_l1789_178961

/-- The number of emus in Farmer Brown's flock -/
def num_emus : ℕ := 20

/-- The number of heads per emu -/
def heads_per_emu : ℕ := 1

/-- The number of legs per emu -/
def legs_per_emu : ℕ := 2

/-- The total count of heads and legs in the flock -/
def total_count : ℕ := 60

/-- Theorem stating that the number of emus is correct given the conditions -/
theorem correct_emu_count : 
  num_emus * (heads_per_emu + legs_per_emu) = total_count :=
by sorry

end NUMINAMATH_CALUDE_correct_emu_count_l1789_178961


namespace NUMINAMATH_CALUDE_total_employees_l1789_178936

/-- Given a corporation with part-time and full-time employees, 
    calculate the total number of employees. -/
theorem total_employees (part_time full_time : ℕ) :
  part_time = 2041 →
  full_time = 63093 →
  part_time + full_time = 65134 := by
  sorry

end NUMINAMATH_CALUDE_total_employees_l1789_178936


namespace NUMINAMATH_CALUDE_five_fourths_of_twelve_fifths_l1789_178983

theorem five_fourths_of_twelve_fifths : (5 / 4 : ℚ) * (12 / 5 : ℚ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_five_fourths_of_twelve_fifths_l1789_178983


namespace NUMINAMATH_CALUDE_smallest_divisible_by_one_to_ten_l1789_178907

theorem smallest_divisible_by_one_to_ten : 
  ∃ n : ℕ, (∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → k ∣ n) ∧ 
    (∀ m : ℕ, m < n → ∃ j : ℕ, 1 ≤ j ∧ j ≤ 10 ∧ ¬(j ∣ m)) ∧ 
    n = 2520 :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_one_to_ten_l1789_178907


namespace NUMINAMATH_CALUDE_cistern_fill_time_l1789_178930

def fill_cistern (problem : ℝ → Prop) : Prop :=
  ∃ t : ℝ,
    -- Tap A fills 1/12 of the cistern per minute
    let rate_A := 1 / 12
    -- Tap B fills 1/t of the cistern per minute
    let rate_B := 1 / t
    -- Both taps run for 4 minutes
    let combined_fill := 4 * (rate_A + rate_B)
    -- Tap B runs for 8 more minutes
    let remaining_fill := 8 * rate_B
    -- The total fill is 1 (complete cistern)
    combined_fill + remaining_fill = 1 ∧
    -- The solution satisfies the original problem
    problem t

theorem cistern_fill_time :
  fill_cistern (λ t ↦ t = 18) :=
sorry

end NUMINAMATH_CALUDE_cistern_fill_time_l1789_178930


namespace NUMINAMATH_CALUDE_simplify_expression_l1789_178999

theorem simplify_expression (a : ℝ) (h1 : a ≠ -1) (h2 : a ≠ 2) : 
  ((3 / (a + 1) - 1) / ((a - 2) / (a^2 + 2*a + 1))) = -a - 1 :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l1789_178999


namespace NUMINAMATH_CALUDE_train_speed_calculation_l1789_178959

/-- Theorem: Train Speed Calculation
Given two trains starting from the same station, traveling along parallel tracks in the same direction,
with one train traveling at 35 mph, and the distance between them after 10 hours being 250 miles,
the speed of the first train is 60 mph. -/
theorem train_speed_calculation (speed_second_train : ℝ) (time : ℝ) (distance : ℝ) :
  speed_second_train = 35 →
  time = 10 →
  distance = 250 →
  ∃ (speed_first_train : ℝ),
    speed_first_train > 0 ∧
    distance = (speed_first_train - speed_second_train) * time ∧
    speed_first_train = 60 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l1789_178959


namespace NUMINAMATH_CALUDE_a_range_l1789_178944

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x, a^x > 1 ↔ x < 0

def q (a : ℝ) : Prop := ∀ x, x^2 - x + a > 0

-- Define the range of a
def range_of_a (a : ℝ) : Prop := (0 < a ∧ a ≤ 1/4) ∨ (a > 1)

-- Theorem statement
theorem a_range (a : ℝ) : 
  ((p a ∨ q a) ∧ ¬(p a ∧ q a)) → range_of_a a :=
sorry

end NUMINAMATH_CALUDE_a_range_l1789_178944


namespace NUMINAMATH_CALUDE_moving_circle_trajectory_l1789_178984

/-- The locus of points satisfying the given conditions is one branch of a hyperbola -/
theorem moving_circle_trajectory (M : ℝ × ℝ) :
  (∃ (x y : ℝ), M = (x, y) ∧ x > 0) →
  (Real.sqrt (M.1^2 + M.2^2) - Real.sqrt ((M.1 - 3)^2 + M.2^2) = 2) →
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ M.1^2 / a^2 - M.2^2 / b^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_moving_circle_trajectory_l1789_178984


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1789_178916

/-- An arithmetic sequence with its sum sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum sequence
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1
  sum_formula : ∀ n, S n = n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1)

/-- Theorem: For an arithmetic sequence with a_2 = 1 and S_4 = 8, a_5 = 7 and S_10 = 80 -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence) 
    (h1 : seq.a 2 = 1) (h2 : seq.S 4 = 8) : 
    seq.a 5 = 7 ∧ seq.S 10 = 80 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1789_178916


namespace NUMINAMATH_CALUDE_total_members_in_math_club_l1789_178915

def math_club (female_members : ℕ) (male_members : ℕ) : Prop :=
  male_members = 2 * female_members

theorem total_members_in_math_club (female_members : ℕ) 
  (h1 : female_members = 6) 
  (h2 : math_club female_members (2 * female_members)) : 
  female_members + 2 * female_members = 18 := by
  sorry

end NUMINAMATH_CALUDE_total_members_in_math_club_l1789_178915


namespace NUMINAMATH_CALUDE_time_taken_by_A_l1789_178997

/-- The time taken by A to reach the destination given the specified conditions -/
theorem time_taken_by_A (distance : ℝ) (speed_A speed_B : ℝ) (time_B : ℝ) : 
  speed_A / speed_B = 3 / 4 →
  time_B * 60 + 30 = speed_B * distance / speed_A →
  speed_A * (time_B * 60 + 30) / 60 = distance →
  speed_A * 2 = distance :=
by sorry

end NUMINAMATH_CALUDE_time_taken_by_A_l1789_178997


namespace NUMINAMATH_CALUDE_unique_number_between_zero_and_two_l1789_178988

theorem unique_number_between_zero_and_two : 
  ∃! (n : ℕ), n ≤ 9 ∧ n > 0 ∧ n < 2 := by sorry

end NUMINAMATH_CALUDE_unique_number_between_zero_and_two_l1789_178988


namespace NUMINAMATH_CALUDE_xyz_inequality_l1789_178971

theorem xyz_inequality (x y z : ℝ) 
  (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) 
  (hsum : x + y + z = 1) : 
  0 ≤ x*y + y*z + x*z - 2*x*y*z ∧ x*y + y*z + x*z - 2*x*y*z ≤ 7/27 := by
  sorry

end NUMINAMATH_CALUDE_xyz_inequality_l1789_178971


namespace NUMINAMATH_CALUDE_opposite_of_negative_third_l1789_178962

theorem opposite_of_negative_third : 
  (fun x : ℚ => -x) (-1/3) = 1/3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_third_l1789_178962


namespace NUMINAMATH_CALUDE_remaining_shoes_l1789_178904

theorem remaining_shoes (large medium small sold : ℕ) 
  (h_large : large = 22)
  (h_medium : medium = 50)
  (h_small : small = 24)
  (h_sold : sold = 83) :
  large + medium + small - sold = 13 := by
  sorry

end NUMINAMATH_CALUDE_remaining_shoes_l1789_178904


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1789_178975

/-- The imaginary part of (2+i)/i is -2 -/
theorem imaginary_part_of_complex_fraction : Complex.im ((2 : Complex) + Complex.I) / Complex.I = -2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1789_178975


namespace NUMINAMATH_CALUDE_dagger_operation_result_l1789_178947

-- Define the † operation
def dagger (m n p q : ℚ) : ℚ := m * p * (q / n)

-- Theorem statement
theorem dagger_operation_result :
  let result := dagger (5/9) (6/4)
  (result + 1/6) = 27/2 := by
  sorry

end NUMINAMATH_CALUDE_dagger_operation_result_l1789_178947


namespace NUMINAMATH_CALUDE_triangle_determinant_zero_l1789_178942

theorem triangle_determinant_zero (A B C : Real) 
  (h_triangle : A + B + C = Real.pi) : 
  let M : Matrix (Fin 3) (Fin 3) Real := 
    ![![Real.cos A ^ 2, Real.tan A, 1],
      ![Real.cos B ^ 2, Real.tan B, 1],
      ![Real.cos C ^ 2, Real.tan C, 1]]
  Matrix.det M = 0 := by
sorry

end NUMINAMATH_CALUDE_triangle_determinant_zero_l1789_178942


namespace NUMINAMATH_CALUDE_trajectory_of_B_l1789_178958

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space of the form ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point lies on a given line -/
def Point.isOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y = l.c

/-- Defines a parallelogram ABCD -/
structure Parallelogram where
  A : Point
  B : Point
  C : Point
  D : Point
  is_parallelogram : (B.x - A.x = D.x - C.x) ∧ (B.y - A.y = D.y - C.y)

/-- Theorem: Trajectory of point B in a parallelogram ABCD -/
theorem trajectory_of_B (ABCD : Parallelogram)
  (hA : ABCD.A = Point.mk (-1) 3)
  (hC : ABCD.C = Point.mk (-3) 2)
  (hD : ABCD.D.isOnLine (Line.mk 1 (-3) 1)) :
  ABCD.B.isOnLine (Line.mk 1 (-3) 20) :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_B_l1789_178958


namespace NUMINAMATH_CALUDE_tank_full_time_l1789_178965

/-- Represents the state of a water tank system with three pipes. -/
structure TankSystem where
  capacity : ℕ
  fill_rate_a : ℕ
  fill_rate_b : ℕ
  drain_rate : ℕ

/-- Calculates the time in minutes required to fill the tank. -/
def time_to_fill (system : TankSystem) : ℕ :=
  let net_fill_per_cycle := system.fill_rate_a + system.fill_rate_b - system.drain_rate
  let cycles := (system.capacity + net_fill_per_cycle - 1) / net_fill_per_cycle
  cycles * 3

/-- Theorem stating that the given tank system will be full after 54 minutes. -/
theorem tank_full_time (system : TankSystem) 
    (h1 : system.capacity = 900)
    (h2 : system.fill_rate_a = 40)
    (h3 : system.fill_rate_b = 30)
    (h4 : system.drain_rate = 20) :
  time_to_fill system = 54 := by
  sorry

#eval time_to_fill { capacity := 900, fill_rate_a := 40, fill_rate_b := 30, drain_rate := 20 }

end NUMINAMATH_CALUDE_tank_full_time_l1789_178965


namespace NUMINAMATH_CALUDE_quadratic_root_sqrt5_minus_2_l1789_178972

theorem quadratic_root_sqrt5_minus_2 :
  ∃ (a b c : ℚ), a = 1 ∧ 
    (∀ x : ℝ, x^2 + b*x + c = 0 ↔ x = Real.sqrt 5 - 2 ∨ x = -(Real.sqrt 5) - 2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_sqrt5_minus_2_l1789_178972


namespace NUMINAMATH_CALUDE_claire_balloons_given_away_l1789_178980

/-- The number of balloons Claire gave away during the fair -/
def balloons_given_away (initial_balloons : ℕ) (floated_away : ℕ) (grabbed_from_coworker : ℕ) (final_balloons : ℕ) : ℕ :=
  initial_balloons - floated_away + grabbed_from_coworker - final_balloons

/-- Theorem stating the number of balloons Claire gave away during the fair -/
theorem claire_balloons_given_away :
  balloons_given_away 50 12 11 39 = 10 := by
  sorry

#eval balloons_given_away 50 12 11 39

end NUMINAMATH_CALUDE_claire_balloons_given_away_l1789_178980


namespace NUMINAMATH_CALUDE_tenth_term_of_arithmetic_sequence_l1789_178933

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem tenth_term_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a1 : a 1 = 1)
  (h_a3 : a 3 = 5) :
  a 10 = 19 := by
sorry

end NUMINAMATH_CALUDE_tenth_term_of_arithmetic_sequence_l1789_178933


namespace NUMINAMATH_CALUDE_triangles_in_pentagon_l1789_178985

/-- The number of triangles formed when all diagonals are drawn in a pentagon -/
def num_triangles_in_pentagon : ℕ := 35

/-- Theorem stating that the number of triangles in a fully connected pentagon is 35 -/
theorem triangles_in_pentagon :
  num_triangles_in_pentagon = 35 := by
  sorry

#check triangles_in_pentagon

end NUMINAMATH_CALUDE_triangles_in_pentagon_l1789_178985


namespace NUMINAMATH_CALUDE_repair_cost_is_13000_l1789_178938

/-- Calculates the repair cost given the purchase price, selling price, and profit percentage --/
def calculate_repair_cost (purchase_price selling_price profit_percent : ℚ) : ℚ :=
  let total_cost := selling_price / (1 + profit_percent / 100)
  total_cost - purchase_price

/-- Theorem stating that the repair cost is 13000 given the problem conditions --/
theorem repair_cost_is_13000 :
  let purchase_price : ℚ := 42000
  let selling_price : ℚ := 60900
  let profit_percent : ℚ := 10.727272727272727
  calculate_repair_cost purchase_price selling_price profit_percent = 13000 := by
  sorry


end NUMINAMATH_CALUDE_repair_cost_is_13000_l1789_178938


namespace NUMINAMATH_CALUDE_solve_system_l1789_178991

theorem solve_system (a b : ℚ) 
  (eq1 : 3 * a + 2 * b = 25)
  (eq2 : 5 * a + b = 20) :
  3 * a + 3 * b = 240 / 7 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l1789_178991


namespace NUMINAMATH_CALUDE_point_on_x_axis_l1789_178929

/-- A point P with coordinates (2m-6, m-1) lies on the x-axis if and only if its coordinates are (-4, 0) -/
theorem point_on_x_axis (m : ℝ) :
  (∃ P : ℝ × ℝ, P = (2*m - 6, m - 1) ∧ P.2 = 0) ↔ (∃ P : ℝ × ℝ, P = (-4, 0)) :=
by sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l1789_178929


namespace NUMINAMATH_CALUDE_chloe_treasures_l1789_178939

theorem chloe_treasures (points_per_treasure : ℕ) (second_level_treasures : ℕ) (total_score : ℕ) 
  (h1 : points_per_treasure = 9)
  (h2 : second_level_treasures = 3)
  (h3 : total_score = 81) :
  total_score = points_per_treasure * (second_level_treasures + 6) :=
by sorry

end NUMINAMATH_CALUDE_chloe_treasures_l1789_178939


namespace NUMINAMATH_CALUDE_earnings_left_over_l1789_178945

/-- Calculates the percentage of earnings left over after spending on rent and dishwasher -/
theorem earnings_left_over (rent_percentage : ℝ) (dishwasher_discount : ℝ) : 
  rent_percentage = 25 →
  dishwasher_discount = 10 →
  100 - (rent_percentage + (rent_percentage - rent_percentage * dishwasher_discount / 100)) = 52.5 := by
  sorry


end NUMINAMATH_CALUDE_earnings_left_over_l1789_178945


namespace NUMINAMATH_CALUDE_distance_A_O_min_distance_O_line_l1789_178924

-- Define the polyline distance function
def polyline_distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  |x₁ - x₂| + |y₁ - y₂|

-- Define point A
def A : ℝ × ℝ := (-1, 3)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define the line equation
def on_line (x y : ℝ) : Prop :=
  2 * x + y - 2 * Real.sqrt 5 = 0

-- Theorem 1: The polyline distance between A and O is 4
theorem distance_A_O :
  polyline_distance A.1 A.2 O.1 O.2 = 4 := by sorry

-- Theorem 2: The minimum polyline distance between O and any point on the line is √5
theorem min_distance_O_line :
  ∃ (x y : ℝ), on_line x y ∧
  ∀ (x' y' : ℝ), on_line x' y' →
  polyline_distance O.1 O.2 x y ≤ polyline_distance O.1 O.2 x' y' ∧
  polyline_distance O.1 O.2 x y = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_distance_A_O_min_distance_O_line_l1789_178924


namespace NUMINAMATH_CALUDE_bess_throws_20_meters_l1789_178914

/-- Represents the Frisbee throwing scenario -/
structure FrisbeeScenario where
  bess_throws : ℕ           -- Number of times Bess throws
  holly_throws : ℕ          -- Number of times Holly throws
  holly_distance : ℕ        -- Distance Holly throws in meters
  total_distance : ℕ        -- Total distance traveled by all Frisbees in meters

/-- Calculates Bess's throwing distance given a FrisbeeScenario -/
def bess_distance (scenario : FrisbeeScenario) : ℕ :=
  (scenario.total_distance - scenario.holly_throws * scenario.holly_distance) / (2 * scenario.bess_throws)

/-- Theorem stating that Bess's throwing distance is 20 meters in the given scenario -/
theorem bess_throws_20_meters (scenario : FrisbeeScenario) 
  (h1 : scenario.bess_throws = 4)
  (h2 : scenario.holly_throws = 5)
  (h3 : scenario.holly_distance = 8)
  (h4 : scenario.total_distance = 200) :
  bess_distance scenario = 20 := by
  sorry

end NUMINAMATH_CALUDE_bess_throws_20_meters_l1789_178914


namespace NUMINAMATH_CALUDE_map_scale_conversion_l1789_178913

/-- Given a map where 15 cm represents 90 km, a 20 cm length represents 120,000 meters -/
theorem map_scale_conversion (map_scale : ℝ) (h : map_scale * 15 = 90) : 
  map_scale * 20 * 1000 = 120000 := by
  sorry

end NUMINAMATH_CALUDE_map_scale_conversion_l1789_178913


namespace NUMINAMATH_CALUDE_cargo_ship_unloading_time_l1789_178987

/-- Cargo ship transportation problem -/
theorem cargo_ship_unloading_time 
  (loading_speed : ℝ) 
  (loading_time : ℝ) 
  (unloading_speed : ℝ) 
  (unloading_time : ℝ) 
  (h1 : loading_speed = 30)
  (h2 : loading_time = 8)
  (h3 : unloading_speed > 0) :
  unloading_time = (loading_speed * loading_time) / unloading_speed :=
by
  sorry

#check cargo_ship_unloading_time

end NUMINAMATH_CALUDE_cargo_ship_unloading_time_l1789_178987


namespace NUMINAMATH_CALUDE_smallest_positive_difference_l1789_178927

theorem smallest_positive_difference (a b : ℤ) (h : 17 * a + 6 * b = 13) :
  ∃ (k : ℤ), k > 0 ∧ a - b = k ∧ ∀ (m : ℤ), m > 0 ∧ (∃ (x y : ℤ), 17 * x + 6 * y = 13 ∧ x - y = m) → k ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_difference_l1789_178927


namespace NUMINAMATH_CALUDE_trapezoid_segment_length_l1789_178932

/-- Represents a trapezoid PQRU -/
structure Trapezoid where
  PQ : ℝ
  RU : ℝ

/-- The theorem stating the length of PQ in the given trapezoid -/
theorem trapezoid_segment_length (PQRU : Trapezoid) 
  (h1 : PQRU.PQ / PQRU.RU = 5 / 2)
  (h2 : PQRU.PQ + PQRU.RU = 180) : 
  PQRU.PQ = 900 / 7 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_segment_length_l1789_178932


namespace NUMINAMATH_CALUDE_mary_next_birthday_age_l1789_178925

theorem mary_next_birthday_age (m s d t : ℝ) : 
  m = 1.25 * s →
  s = 0.7 * d →
  t = 2 * s →
  m + s + d + t = 38 →
  ⌊m⌋ + 1 = 9 :=
sorry

end NUMINAMATH_CALUDE_mary_next_birthday_age_l1789_178925


namespace NUMINAMATH_CALUDE_unique_integer_solution_l1789_178993

theorem unique_integer_solution (a : ℤ) : 
  (∃! x : ℤ, |a*x + a + 2| < 2) ↔ (a = 3 ∨ a = -3) :=
sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l1789_178993


namespace NUMINAMATH_CALUDE_sapling_growth_relation_l1789_178917

/-- Represents the height of a sapling over time -/
def sapling_height (x : ℝ) : ℝ :=
  50 * x + 100

theorem sapling_growth_relation (x : ℝ) (y : ℝ) 
  (h1 : sapling_height 0 = 100) 
  (h2 : ∀ x1 x2, sapling_height x2 - sapling_height x1 = 50 * (x2 - x1)) :
  y = sapling_height x :=
by sorry

end NUMINAMATH_CALUDE_sapling_growth_relation_l1789_178917


namespace NUMINAMATH_CALUDE_remainder_101_pow_37_mod_100_l1789_178918

theorem remainder_101_pow_37_mod_100 : 101^37 % 100 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_101_pow_37_mod_100_l1789_178918


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1789_178957

def A : Set (ℝ × ℝ) := {p | p.2 = p.1 + 1}
def B : Set (ℝ × ℝ) := {p | p.2 = -2 * p.1 + 4}

theorem intersection_of_A_and_B : A ∩ B = {(1, 2)} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1789_178957


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l1789_178992

theorem contrapositive_equivalence (m : ℝ) :
  (¬(∃ x : ℝ, x^2 + x - m = 0) → m ≤ 0) ↔
  (m > 0 → ∃ x : ℝ, x^2 + x - m = 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l1789_178992


namespace NUMINAMATH_CALUDE_smallest_valid_debt_proof_l1789_178968

/-- The value of one sheep in dollars -/
def sheep_value : ℕ := 250

/-- The value of one lamb in dollars -/
def lamb_value : ℕ := 150

/-- A debt resolution is valid if it can be expressed as an integer combination of sheep and lambs -/
def is_valid_debt (d : ℕ) : Prop :=
  ∃ (s l : ℤ), d = sheep_value * s + lamb_value * l

/-- The smallest positive debt that can be resolved -/
def smallest_valid_debt : ℕ := 50

theorem smallest_valid_debt_proof :
  (∀ d : ℕ, d > 0 ∧ d < smallest_valid_debt → ¬is_valid_debt d) ∧
  is_valid_debt smallest_valid_debt :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_debt_proof_l1789_178968


namespace NUMINAMATH_CALUDE_distance_ratio_bound_l1789_178908

/-- Given 5 points on a plane, the ratio of the maximum distance to the minimum distance
    between these points is greater than or equal to 2 sin 54°. -/
theorem distance_ratio_bound (points : Finset (EuclideanSpace ℝ (Fin 2))) 
  (h : points.card = 5) :
  let distances := (Finset.product points points).image (λ (p : EuclideanSpace ℝ (Fin 2) × EuclideanSpace ℝ (Fin 2)) => norm (p.1 - p.2))
  ∃ (max_dist min_dist : ℝ), max_dist ∈ distances ∧ min_dist ∈ distances ∧
    min_dist > 0 ∧ max_dist / min_dist ≥ 2 * Real.sin (54 * π / 180) :=
sorry

end NUMINAMATH_CALUDE_distance_ratio_bound_l1789_178908


namespace NUMINAMATH_CALUDE_power_product_equals_1938400_l1789_178903

theorem power_product_equals_1938400 : 2^4 * 3^2 * 5^2 * 7^2 * 11 = 1938400 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equals_1938400_l1789_178903


namespace NUMINAMATH_CALUDE_total_friends_l1789_178954

theorem total_friends (initial_friends additional_friends : ℕ) 
  (h1 : initial_friends = 4) 
  (h2 : additional_friends = 3) : 
  initial_friends + additional_friends = 7 := by
    sorry

end NUMINAMATH_CALUDE_total_friends_l1789_178954


namespace NUMINAMATH_CALUDE_mike_spent_500_on_self_l1789_178946

def total_rose_bushes : ℕ := 6
def price_per_rose_bush : ℕ := 75
def rose_bushes_for_friend : ℕ := 2
def tiger_tooth_aloes : ℕ := 2
def price_per_aloe : ℕ := 100

def money_spent_on_self : ℕ :=
  (total_rose_bushes - rose_bushes_for_friend) * price_per_rose_bush +
  tiger_tooth_aloes * price_per_aloe

theorem mike_spent_500_on_self :
  money_spent_on_self = 500 := by
  sorry

end NUMINAMATH_CALUDE_mike_spent_500_on_self_l1789_178946


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_log_l1789_178969

theorem arithmetic_geometric_sequence_log (a b : ℝ) : 
  a ≠ b →
  (2 * a = 1 + b) →
  (b ^ 2 = a) →
  7 * a * (Real.log (-b) / Real.log a) = 7/8 := by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_log_l1789_178969


namespace NUMINAMATH_CALUDE_det_special_matrix_is_zero_l1789_178977

open Matrix

theorem det_special_matrix_is_zero (x y z : ℝ) : 
  det ![![1, x + z, y - z],
       ![1, x + y + z, y - z],
       ![1, x + z, x + y]] = 0 := by
  sorry

end NUMINAMATH_CALUDE_det_special_matrix_is_zero_l1789_178977


namespace NUMINAMATH_CALUDE_cos_18_cos_42_minus_cos_72_sin_42_l1789_178928

theorem cos_18_cos_42_minus_cos_72_sin_42 :
  Real.cos (18 * π / 180) * Real.cos (42 * π / 180) - 
  Real.cos (72 * π / 180) * Real.sin (42 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_18_cos_42_minus_cos_72_sin_42_l1789_178928


namespace NUMINAMATH_CALUDE_triple_base_double_exponent_l1789_178934

theorem triple_base_double_exponent (a b x : ℝ) (hb : b ≠ 0) :
  let r := (3 * a) ^ (2 * b)
  r = a ^ b * x ^ b → x = 9 * a := by
sorry

end NUMINAMATH_CALUDE_triple_base_double_exponent_l1789_178934


namespace NUMINAMATH_CALUDE_modified_factor_tree_l1789_178956

theorem modified_factor_tree : 
  ∀ A B C D E : ℕ,
  A = B * C →
  B = 3 * D →
  C = 7 * E →
  D = 5 * 2 →
  E = 7 * 3 →
  A = 4410 := by
  sorry

end NUMINAMATH_CALUDE_modified_factor_tree_l1789_178956
