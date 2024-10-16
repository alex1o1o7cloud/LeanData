import Mathlib

namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l2818_281846

theorem least_subtraction_for_divisibility :
  ∃ (n : ℕ), n ≤ 5 ∧ (∀ m : ℕ, m < n → ¬(37 ∣ (5000 - m))) ∧ (37 ∣ (5000 - n)) := by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l2818_281846


namespace NUMINAMATH_CALUDE_female_officers_count_l2818_281814

/-- The total number of police officers on duty that night -/
def total_on_duty : ℕ := 160

/-- The fraction of officers on duty that were female -/
def female_fraction : ℚ := 1/2

/-- The percentage of all female officers that were on duty -/
def female_on_duty_percentage : ℚ := 16/100

/-- The total number of female officers on the police force -/
def total_female_officers : ℕ := 500

theorem female_officers_count :
  total_female_officers = 
    (total_on_duty * female_fraction) / female_on_duty_percentage := by
  sorry

end NUMINAMATH_CALUDE_female_officers_count_l2818_281814


namespace NUMINAMATH_CALUDE_correct_repetitions_per_bracelet_l2818_281835

/-- The number of pattern repetitions per bracelet -/
def repetitions_per_bracelet : ℕ := 3

/-- The number of green beads in one pattern -/
def green_beads : ℕ := 3

/-- The number of purple beads in one pattern -/
def purple_beads : ℕ := 5

/-- The number of red beads in one pattern -/
def red_beads : ℕ := 6

/-- The number of beads in one pattern -/
def beads_per_pattern : ℕ := green_beads + purple_beads + red_beads

/-- The number of pattern repetitions per necklace -/
def repetitions_per_necklace : ℕ := 5

/-- The number of necklaces -/
def number_of_necklaces : ℕ := 10

/-- The total number of beads for 1 bracelet and 10 necklaces -/
def total_beads : ℕ := 742

theorem correct_repetitions_per_bracelet :
  repetitions_per_bracelet * beads_per_pattern +
  number_of_necklaces * repetitions_per_necklace * beads_per_pattern = total_beads :=
by sorry

end NUMINAMATH_CALUDE_correct_repetitions_per_bracelet_l2818_281835


namespace NUMINAMATH_CALUDE_max_sum_xy_l2818_281833

def associated_numbers (m : ℕ) : List ℕ :=
  sorry

def P (m : ℕ) : ℚ :=
  (associated_numbers m).sum / 22

def x (a b : ℕ) : ℕ := 100 * a + 10 * b + 3

def y (b : ℕ) : ℕ := 400 + 10 * b + 5

theorem max_sum_xy :
  ∀ a b : ℕ,
    1 ≤ a ∧ a ≤ 9 →
    1 ≤ b ∧ b ≤ 9 →
    (∀ d : ℕ, d ∈ associated_numbers (x a b) → d ≠ 0) →
    (∀ d : ℕ, d ∈ associated_numbers (y b) → d ≠ 0) →
    P (x a b) + P (y b) = 20 →
    x a b + y b ≤ 1028 :=
  sorry

end NUMINAMATH_CALUDE_max_sum_xy_l2818_281833


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_by_11_l2818_281866

theorem least_subtraction_for_divisibility_by_11 : 
  ∃ (x : ℕ), x = 7 ∧ 
  (∀ (y : ℕ), y < x → ¬(11 ∣ (427398 - y))) ∧
  (11 ∣ (427398 - x)) := by
sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_by_11_l2818_281866


namespace NUMINAMATH_CALUDE_geometry_propositions_l2818_281840

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (parallelPL : Line → Plane → Prop)
variable (perpendicularPL : Line → Plane → Prop)
variable (parallelPlanes : Plane → Plane → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)
variable (intersection : Plane → Plane → Line)

-- Theorem statement
theorem geometry_propositions 
  (α β : Plane) (m n : Line) : 
  -- Proposition 2
  (∀ m, perpendicularPL m α ∧ perpendicularPL m β → parallelPlanes α β) ∧
  -- Proposition 3
  (intersection α β = n ∧ parallelPL m α ∧ parallelPL m β → parallel m n) ∧
  -- Proposition 4
  (perpendicularPlanes α β ∧ perpendicularPL m α ∧ perpendicularPL n β → perpendicular m n) :=
by sorry

end NUMINAMATH_CALUDE_geometry_propositions_l2818_281840


namespace NUMINAMATH_CALUDE_point_on_extension_line_l2818_281862

/-- Given two points P₁ and P₂ in ℝ², and a point P on the extension line of P₁P₂
    such that the distance between P₁ and P is twice the distance between P and P₂,
    prove that P has the coordinates (-2, 11). -/
theorem point_on_extension_line (P₁ P₂ P : ℝ × ℝ) : 
  P₁ = (2, -1) → 
  P₂ = (0, 5) → 
  (∃ t : ℝ, P = P₁ + t • (P₂ - P₁)) →
  dist P₁ P = 2 * dist P P₂ →
  P = (-2, 11) := by
  sorry

end NUMINAMATH_CALUDE_point_on_extension_line_l2818_281862


namespace NUMINAMATH_CALUDE_amicable_pairs_theorem_l2818_281886

/-- Sum of divisors of a number -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- Two numbers are amicable if the sum of proper divisors of each equals the other number -/
def is_amicable_pair (m n : ℕ) : Prop :=
  sum_of_divisors m = m + n ∧ sum_of_divisors n = m + n

/-- The main theorem stating that the given pairs are amicable -/
theorem amicable_pairs_theorem :
  let pair1_1 := 3^3 * 5 * 7 * 71
  let pair1_2 := 3^3 * 5 * 17 * 31
  let pair2_1 := 3^2 * 5 * 13 * 79 * 29
  let pair2_2 := 3^2 * 5 * 13 * 11 * 199
  is_amicable_pair pair1_1 pair1_2 ∧ is_amicable_pair pair2_1 pair2_2 := by
  sorry

end NUMINAMATH_CALUDE_amicable_pairs_theorem_l2818_281886


namespace NUMINAMATH_CALUDE_intersection_points_with_ellipse_l2818_281824

/-- The line equation mx - ny = 4 and circle x^2 + y^2 = 4 have no intersection points -/
def no_intersection (m n : ℝ) : Prop :=
  ∀ x y : ℝ, (m * x - n * y = 4) → (x^2 + y^2 ≠ 4)

/-- The ellipse equation x^2/9 + y^2/4 = 1 -/
def on_ellipse (x y : ℝ) : Prop :=
  x^2 / 9 + y^2 / 4 = 1

/-- A point (x, y) is on the line passing through (m, n) -/
def on_line_through_P (m n x y : ℝ) : Prop :=
  ∃ t : ℝ, x = m * t ∧ y = n * t

/-- The theorem statement -/
theorem intersection_points_with_ellipse (m n : ℝ) :
  no_intersection m n →
  (∃! (x1 y1 x2 y2 : ℝ), 
    x1 ≠ x2 ∧ 
    on_ellipse x1 y1 ∧ 
    on_ellipse x2 y2 ∧ 
    on_line_through_P m n x1 y1 ∧ 
    on_line_through_P m n x2 y2) :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_with_ellipse_l2818_281824


namespace NUMINAMATH_CALUDE_abs_equation_quadratic_coefficients_l2818_281816

theorem abs_equation_quadratic_coefficients :
  ∀ (b c : ℝ),
  (∀ x : ℝ, |x - 4| = 3 ↔ x^2 + b*x + c = 0) →
  b = -8 ∧ c = 7 := by
sorry

end NUMINAMATH_CALUDE_abs_equation_quadratic_coefficients_l2818_281816


namespace NUMINAMATH_CALUDE_vector_equality_iff_magnitude_and_parallel_l2818_281876

/-- Two plane vectors are equal if and only if their magnitudes are equal and they are parallel. -/
theorem vector_equality_iff_magnitude_and_parallel {a b : ℝ × ℝ} :
  a = b ↔ (‖a‖ = ‖b‖ ∧ ∃ (k : ℝ), a = k • b) :=
by sorry

end NUMINAMATH_CALUDE_vector_equality_iff_magnitude_and_parallel_l2818_281876


namespace NUMINAMATH_CALUDE_andrew_payment_l2818_281860

/-- The amount Andrew paid to the shopkeeper -/
def total_amount (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mango_quantity * mango_rate

/-- Theorem stating that Andrew paid 1376 to the shopkeeper -/
theorem andrew_payment : total_amount 14 54 10 62 = 1376 := by
  sorry

end NUMINAMATH_CALUDE_andrew_payment_l2818_281860


namespace NUMINAMATH_CALUDE_at_least_one_travels_to_beijing_l2818_281832

theorem at_least_one_travels_to_beijing 
  (prob_A : ℝ) 
  (prob_B : ℝ) 
  (h1 : prob_A = 1/3) 
  (h2 : prob_B = 1/4) 
  (h3 : 0 ≤ prob_A ∧ prob_A ≤ 1) 
  (h4 : 0 ≤ prob_B ∧ prob_B ≤ 1) : 
  1 - (1 - prob_A) * (1 - prob_B) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_at_least_one_travels_to_beijing_l2818_281832


namespace NUMINAMATH_CALUDE_problem_solution_l2818_281813

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + 2 * m * x - 1

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := (3 * f m x + 4) / (x - 2)

theorem problem_solution (m : ℝ) :
  m > 0 →
  (∀ x, f m x < 0 ↔ -3 < x ∧ x < 1) →
  (∃ min_g : ℝ, ∀ x > 2, g m x ≥ min_g ∧ ∃ x₀ > 2, g m x₀ = min_g) ∧
  min_g = 12 ∧
  (∃ x₁ x₂ : ℝ, x₁ ∈ [-3, 0] ∧ x₂ ∈ [-3, 0] ∧ |f m x₁ - f m x₂| ≥ 4 → m ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2818_281813


namespace NUMINAMATH_CALUDE_product_equals_zero_l2818_281843

def product_sequence (a : ℤ) : ℤ := (a - 10) * (a - 9) * (a - 8) * (a - 7) * (a - 6) * (a - 5) * (a - 4) * (a - 3) * (a - 2) * (a - 1) * a

theorem product_equals_zero : product_sequence 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_zero_l2818_281843


namespace NUMINAMATH_CALUDE_remainder_two_power_thirty_plus_three_mod_seven_l2818_281867

theorem remainder_two_power_thirty_plus_three_mod_seven :
  (2^30 + 3) % 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_two_power_thirty_plus_three_mod_seven_l2818_281867


namespace NUMINAMATH_CALUDE_graph_shift_l2818_281844

theorem graph_shift (x : ℝ) : (10 : ℝ) ^ (x + 3) = (10 : ℝ) ^ ((x + 4) - 1) := by
  sorry

end NUMINAMATH_CALUDE_graph_shift_l2818_281844


namespace NUMINAMATH_CALUDE_a_more_stable_than_b_l2818_281895

/-- Represents a shooter with their shooting variance -/
structure Shooter where
  name : String
  variance : ℝ
  variance_nonneg : 0 ≤ variance

/-- Definition of more stable shooting performance -/
def more_stable (a b : Shooter) : Prop :=
  a.variance < b.variance

/-- Theorem stating that shooter A has more stable performance than B -/
theorem a_more_stable_than_b :
  let a : Shooter := ⟨"A", 0.12, by norm_num⟩
  let b : Shooter := ⟨"B", 0.6, by norm_num⟩
  more_stable a b := by
  sorry


end NUMINAMATH_CALUDE_a_more_stable_than_b_l2818_281895


namespace NUMINAMATH_CALUDE_monday_rainfall_rate_l2818_281861

/-- Represents the rainfall data for three days -/
structure RainfallData where
  monday_hours : ℝ
  monday_rate : ℝ
  tuesday_hours : ℝ
  tuesday_rate : ℝ
  wednesday_hours : ℝ
  wednesday_rate : ℝ
  total_rainfall : ℝ

/-- Theorem stating that given the rainfall conditions, the rate on Monday was 1 inch per hour -/
theorem monday_rainfall_rate (data : RainfallData)
  (h1 : data.monday_hours = 7)
  (h2 : data.tuesday_hours = 4)
  (h3 : data.tuesday_rate = 2)
  (h4 : data.wednesday_hours = 2)
  (h5 : data.wednesday_rate = 2 * data.tuesday_rate)
  (h6 : data.total_rainfall = 23)
  (h7 : data.total_rainfall = data.monday_hours * data.monday_rate + 
                              data.tuesday_hours * data.tuesday_rate + 
                              data.wednesday_hours * data.wednesday_rate) :
  data.monday_rate = 1 := by
  sorry

end NUMINAMATH_CALUDE_monday_rainfall_rate_l2818_281861


namespace NUMINAMATH_CALUDE_total_sales_l2818_281892

def candy_bar_sales (max_sales seth_sales emma_sales : ℕ) : Prop :=
  (max_sales = 24) ∧
  (seth_sales = 3 * max_sales + 6) ∧
  (emma_sales = seth_sales / 2 + 5)

theorem total_sales (max_sales seth_sales emma_sales : ℕ) :
  candy_bar_sales max_sales seth_sales emma_sales →
  seth_sales + emma_sales = 122 := by
  sorry

end NUMINAMATH_CALUDE_total_sales_l2818_281892


namespace NUMINAMATH_CALUDE_largest_y_coordinate_l2818_281826

theorem largest_y_coordinate (x y : ℝ) : 
  (x^2 / 49) + ((y - 3)^2 / 25) = 0 → y ≤ 3 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_y_coordinate_l2818_281826


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2818_281869

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of specific terms in the sequence equals 70 -/
def sum_condition (a : ℕ → ℝ) : Prop :=
  a 2 + a 7 + a 8 + a 9 + a 14 = 70

theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) 
  (h2 : sum_condition a) : 
  a 8 = 14 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2818_281869


namespace NUMINAMATH_CALUDE_gcd_4288_9277_l2818_281829

theorem gcd_4288_9277 : Int.gcd 4288 9277 = 1 := by sorry

end NUMINAMATH_CALUDE_gcd_4288_9277_l2818_281829


namespace NUMINAMATH_CALUDE_units_digit_of_expression_l2818_281831

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The units digit of 8 * 19 * 1983 - 8^3 is 4 -/
theorem units_digit_of_expression : unitsDigit (8 * 19 * 1983 - 8^3) = 4 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_expression_l2818_281831


namespace NUMINAMATH_CALUDE_binomial_9_5_l2818_281891

theorem binomial_9_5 : Nat.choose 9 5 = 126 := by
  sorry

end NUMINAMATH_CALUDE_binomial_9_5_l2818_281891


namespace NUMINAMATH_CALUDE_cubic_polynomial_factor_property_l2818_281848

/-- Given a cubic polynomial 2x³ - hx + k where x + 2 and x - 1 are factors, 
    prove that |2h-3k| = 0 -/
theorem cubic_polynomial_factor_property (h k : ℝ) : 
  (∀ x, (x + 2) * (x - 1) ∣ (2 * x^3 - h * x + k)) → 
  |2 * h - 3 * k| = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_factor_property_l2818_281848


namespace NUMINAMATH_CALUDE_min_value_theorem_l2818_281806

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 3) :
  (a^2 + b^2 + 22) / (a + b) ≥ 8 ∧ ∃ (a' b' : ℝ), a' > 0 ∧ b' > 0 ∧ a' * b' = 3 ∧ (a'^2 + b'^2 + 22) / (a' + b') = 8 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2818_281806


namespace NUMINAMATH_CALUDE_boxes_in_case_l2818_281841

/-- Proves that the number of boxes in a case is 3 -/
theorem boxes_in_case
  (num_boxes : ℕ)
  (eggs_per_box : ℕ)
  (total_eggs : ℕ)
  (h1 : num_boxes = 3)
  (h2 : eggs_per_box = 7)
  (h3 : total_eggs = 21)
  (h4 : num_boxes * eggs_per_box = total_eggs) :
  num_boxes = 3 := by
  sorry


end NUMINAMATH_CALUDE_boxes_in_case_l2818_281841


namespace NUMINAMATH_CALUDE_range_of_x_plus_y_min_distance_intersection_l2818_281888

-- Define the curve C
def on_curve_C (x y : ℝ) : Prop := x^2 = 4*y

-- Define the line l
def on_line_l (x y t α : ℝ) : Prop := x = t * Real.cos α ∧ y = 1 + t * Real.sin α

-- Theorem 1: Range of x+y
theorem range_of_x_plus_y (x y : ℝ) (h : on_curve_C x y) : x + y ≥ -1 := by
  sorry

-- Theorem 2: Minimum distance between intersection points
theorem min_distance_intersection (α : ℝ) : 
  ∃ (A B : ℝ × ℝ), 
    (on_curve_C A.1 A.2 ∧ ∃ t, on_line_l A.1 A.2 t α) ∧ 
    (on_curve_C B.1 B.2 ∧ ∃ t, on_line_l B.1 B.2 t α) ∧
    ∀ (P Q : ℝ × ℝ), 
      (on_curve_C P.1 P.2 ∧ ∃ t, on_line_l P.1 P.2 t α) →
      (on_curve_C Q.1 Q.2 ∧ ∃ t, on_line_l Q.1 Q.2 t α) →
      Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ≤ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ∧
      Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_plus_y_min_distance_intersection_l2818_281888


namespace NUMINAMATH_CALUDE_inscribed_circle_rectangle_area_l2818_281863

theorem inscribed_circle_rectangle_area :
  ∀ (r : ℝ) (l w : ℝ),
  r = 7 →
  l = 3 * w →
  w = 2 * r →
  l * w = 588 :=
by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_rectangle_area_l2818_281863


namespace NUMINAMATH_CALUDE_intersection_of_perpendicular_lines_l2818_281822

/-- Given two lines in a plane, where one is y = 3x + 4 and the other is perpendicular to it
    passing through the point (3, 2), their intersection point is (3/10, 49/10). -/
theorem intersection_of_perpendicular_lines 
  (line1 : ℝ → ℝ → Prop) 
  (line2 : ℝ → ℝ → Prop) 
  (h1 : ∀ x y, line1 x y ↔ y = 3 * x + 4)
  (h2 : ∀ x y, line2 x y → (y - 2) = -(1/3) * (x - 3))
  (h3 : line2 3 2)
  : ∃ x y, line1 x y ∧ line2 x y ∧ x = 3/10 ∧ y = 49/10 :=
sorry

end NUMINAMATH_CALUDE_intersection_of_perpendicular_lines_l2818_281822


namespace NUMINAMATH_CALUDE_sum_extension_terms_l2818_281819

theorem sum_extension_terms (k : ℕ) (hk : k > 1) : 
  (Finset.range (2^(k+1) - 1)).card - (Finset.range (2^k - 1)).card = 2^k :=
sorry

end NUMINAMATH_CALUDE_sum_extension_terms_l2818_281819


namespace NUMINAMATH_CALUDE_flower_shop_bouquets_l2818_281847

theorem flower_shop_bouquets (roses_per_bouquet : ℕ) 
  (rose_bouquets_sold daisy_bouquets_sold total_flowers : ℕ) :
  roses_per_bouquet = 12 →
  rose_bouquets_sold = 10 →
  daisy_bouquets_sold = 10 →
  total_flowers = 190 →
  total_flowers = roses_per_bouquet * rose_bouquets_sold + 
    (total_flowers - roses_per_bouquet * rose_bouquets_sold) →
  rose_bouquets_sold + daisy_bouquets_sold = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_flower_shop_bouquets_l2818_281847


namespace NUMINAMATH_CALUDE_sin_n_equals_cos_510_l2818_281836

theorem sin_n_equals_cos_510 (n : ℤ) (h1 : -180 ≤ n) (h2 : n ≤ 180) :
  Real.sin (n * π / 180) = Real.cos (510 * π / 180) → n = -60 := by sorry

end NUMINAMATH_CALUDE_sin_n_equals_cos_510_l2818_281836


namespace NUMINAMATH_CALUDE_one_in_linked_triple_l2818_281894

def is_linked (m n : ℕ+) : Prop :=
  (m.val ∣ 3 * n.val + 1) ∧ (n.val ∣ 3 * m.val + 1)

theorem one_in_linked_triple (a b c : ℕ+) :
  a ≠ b → b ≠ c → a ≠ c →
  is_linked a b → is_linked b c →
  1 ∈ ({a.val, b.val, c.val} : Set ℕ) :=
sorry

end NUMINAMATH_CALUDE_one_in_linked_triple_l2818_281894


namespace NUMINAMATH_CALUDE_largest_n_sin_cos_inequality_l2818_281877

theorem largest_n_sin_cos_inequality :
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (x : ℝ), (Real.sin x)^n + (Real.cos x)^n ≥ 1 / (2 * Real.sqrt (n : ℝ))) ∧
  (∀ (m : ℕ), m > n → ∃ (y : ℝ), (Real.sin y)^m + (Real.cos y)^m < 1 / (2 * Real.sqrt (m : ℝ))) ∧
  n = 2 := by
  sorry

end NUMINAMATH_CALUDE_largest_n_sin_cos_inequality_l2818_281877


namespace NUMINAMATH_CALUDE_quadratic_has_minimum_l2818_281883

/-- Given a quadratic function f(x) = ax^2 + bx + c where c = -b^2/(4a) and a > 0,
    prove that the graph of y = f(x) has a minimum. -/
theorem quadratic_has_minimum (a b : ℝ) (ha : a > 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + (-b^2) / (4 * a)
  ∃ x₀, ∀ x, f x₀ ≤ f x :=
by sorry

end NUMINAMATH_CALUDE_quadratic_has_minimum_l2818_281883


namespace NUMINAMATH_CALUDE_polynomial_equality_l2818_281812

theorem polynomial_equality (a : ℝ) : 
  (∀ x : ℝ, x^2 + a*x + 4 = (x + 2)^2) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l2818_281812


namespace NUMINAMATH_CALUDE_hannahs_brothers_l2818_281838

theorem hannahs_brothers (num_brothers : ℕ) : num_brothers = 3 :=
  by
  -- Hannah has some brothers
  have h1 : num_brothers > 0 := by sorry
  
  -- All her brothers are 8 years old
  let brother_age := 8
  
  -- Hannah is 48 years old
  let hannah_age := 48
  
  -- Hannah's age is twice the sum of her brothers' ages
  have h2 : hannah_age = 2 * (num_brothers * brother_age) := by sorry
  
  -- Proof that num_brothers = 3
  sorry

end NUMINAMATH_CALUDE_hannahs_brothers_l2818_281838


namespace NUMINAMATH_CALUDE_woman_lawyer_probability_l2818_281827

/-- Represents a study group with members, women, and lawyers. -/
structure StudyGroup where
  total_members : ℕ
  women_percentage : ℝ
  lawyer_percentage : ℝ
  women_percentage_valid : 0 ≤ women_percentage ∧ women_percentage ≤ 1
  lawyer_percentage_valid : 0 ≤ lawyer_percentage ∧ lawyer_percentage ≤ 1

/-- Calculates the probability of selecting a woman lawyer from the study group. -/
def probability_woman_lawyer (group : StudyGroup) : ℝ :=
  group.women_percentage * group.lawyer_percentage

/-- Theorem stating that the probability of selecting a woman lawyer is 0.08
    given the specified conditions. -/
theorem woman_lawyer_probability (group : StudyGroup) 
  (h1 : group.women_percentage = 0.4)
  (h2 : group.lawyer_percentage = 0.2) : 
  probability_woman_lawyer group = 0.08 := by
  sorry

#check woman_lawyer_probability

end NUMINAMATH_CALUDE_woman_lawyer_probability_l2818_281827


namespace NUMINAMATH_CALUDE_system_solution_l2818_281873

theorem system_solution (a b c x y z : ℝ) 
  (h1 : a^2 + b^2 = c^2) 
  (h2 : z^2 = x^2 + y^2) 
  (h3 : (z + c)^2 = (x + a)^2 + (y + b)^2) : 
  y = (b/a) * x ∧ z = (c/a) * x :=
sorry

end NUMINAMATH_CALUDE_system_solution_l2818_281873


namespace NUMINAMATH_CALUDE_problem_solution_roots_product_l2818_281810

noncomputable section

-- Define the functions f and g
def f (x : ℝ) : ℝ := Real.log x
def g (m : ℝ) (x : ℝ) : ℝ := x + m

-- Define the function F
def F (m : ℝ) (x : ℝ) : ℝ := f x - g m x

theorem problem_solution (m : ℝ) :
  (∀ x > 0, f x ≤ g m x) ↔ m ≥ -1 :=
sorry

theorem roots_product (m : ℝ) (x₁ x₂ : ℝ) :
  x₁ < x₂ →
  F m x₁ = 0 →
  F m x₂ = 0 →
  x₁ * x₂ < 1 :=
sorry

end NUMINAMATH_CALUDE_problem_solution_roots_product_l2818_281810


namespace NUMINAMATH_CALUDE_vacation_pictures_l2818_281868

theorem vacation_pictures (zoo museum beach amusement_park deleted : ℕ) :
  zoo = 802 →
  museum = 526 →
  beach = 391 →
  amusement_park = 868 →
  deleted = 1395 →
  zoo + museum + beach + amusement_park - deleted = 1192 := by
  sorry

end NUMINAMATH_CALUDE_vacation_pictures_l2818_281868


namespace NUMINAMATH_CALUDE_means_inequality_l2818_281870

theorem means_inequality (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a ≠ b) :
  (a^2 + b^2) / 2 > (a + b) / 2 ∧ (a + b) / 2 > Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_means_inequality_l2818_281870


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2818_281898

theorem inequality_solution_set (x : ℝ) : 
  (1 / (x^2 - 4) + 4 / (2*x^2 + 7*x + 6) ≤ 1 / (2*x + 3) + 4 / (2*x^3 + 3*x^2 - 8*x - 12)) ↔ 
  (x ∈ Set.Ioo (-2 : ℝ) (-3/2) ∪ Set.Ico 1 2 ∪ Set.Ici 5) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2818_281898


namespace NUMINAMATH_CALUDE_consecutive_odd_numbers_l2818_281830

theorem consecutive_odd_numbers (n : ℕ) 
  (h_avg : (27 + 27 - 2 * (n - 1)) / 2 = 24) 
  (h_largest : 27 = 27 - 2 * (n - 1) + 2 * (n - 1)) : n = 4 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_numbers_l2818_281830


namespace NUMINAMATH_CALUDE_milk_water_ratio_l2818_281845

/-- 
Given a mixture of milk and water with total volume 145 liters,
if adding 58 liters of water changes the ratio of milk to water to 3:4,
then the initial ratio of milk to water was 3:2.
-/
theorem milk_water_ratio 
  (total_volume : ℝ) 
  (added_water : ℝ) 
  (milk : ℝ) 
  (water : ℝ) : 
  total_volume = 145 →
  added_water = 58 →
  milk + water = total_volume →
  milk / (water + added_water) = 3 / 4 →
  milk / water = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_milk_water_ratio_l2818_281845


namespace NUMINAMATH_CALUDE_ruth_apples_l2818_281887

/-- The number of apples Ruth ends up with after a series of events -/
def final_apples (initial : ℕ) (shared : ℕ) (gift : ℕ) : ℕ :=
  let remaining := initial - shared
  let after_sister := remaining - remaining / 2
  after_sister + gift

/-- Theorem stating that Ruth ends up with 105 apples -/
theorem ruth_apples : final_apples 200 5 7 = 105 := by
  sorry

end NUMINAMATH_CALUDE_ruth_apples_l2818_281887


namespace NUMINAMATH_CALUDE_matrix_multiplication_example_l2818_281839

theorem matrix_multiplication_example :
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; 2, -4]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![7, -3; 0, 2]
  A * B = !![21, -7; 14, -14] := by
  sorry

end NUMINAMATH_CALUDE_matrix_multiplication_example_l2818_281839


namespace NUMINAMATH_CALUDE_max_profit_thermos_l2818_281801

/-- Thermos cup prices and quantities -/
structure ThermosCups where
  price_a : ℝ
  price_b : ℝ
  quantity_a : ℕ
  quantity_b : ℕ

/-- Conditions for thermos cup problem -/
def thermos_conditions (t : ThermosCups) : Prop :=
  t.price_b = t.price_a + 10 ∧
  600 / t.price_b = 480 / t.price_a ∧
  t.quantity_a + t.quantity_b = 120 ∧
  t.quantity_a ≥ t.quantity_b / 2 ∧
  t.quantity_a ≤ t.quantity_b

/-- Profit calculation -/
def profit (t : ThermosCups) : ℝ :=
  (t.price_a - 30) * t.quantity_a + (t.price_b * 0.9 - 30) * t.quantity_b

/-- Theorem: Maximum profit for thermos cup sales -/
theorem max_profit_thermos :
  ∃ t : ThermosCups, thermos_conditions t ∧
    profit t = 1600 ∧
    (∀ t' : ThermosCups, thermos_conditions t' → profit t' ≤ profit t) :=
  sorry

end NUMINAMATH_CALUDE_max_profit_thermos_l2818_281801


namespace NUMINAMATH_CALUDE_average_of_three_l2818_281804

theorem average_of_three (M : ℝ) (h1 : 12 < M) (h2 : M < 25) :
  let avg := (8 + 15 + M) / 3
  (avg = 12 ∨ avg = 15) ∧ avg ≠ 18 ∧ avg ≠ 20 ∧ avg ≠ 23 := by
  sorry

end NUMINAMATH_CALUDE_average_of_three_l2818_281804


namespace NUMINAMATH_CALUDE_y1_greater_than_y2_l2818_281857

/-- Given two points on a linear function, prove that y₁ > y₂ -/
theorem y1_greater_than_y2 (y₁ y₂ : ℝ) : 
  (y₁ = -(-1) + 1) → (y₂ = -(2) + 1) → y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_y1_greater_than_y2_l2818_281857


namespace NUMINAMATH_CALUDE_shaded_area_sum_l2818_281802

/-- The sum of the areas of two pie-shaped regions in a circle with an inscribed square --/
theorem shaded_area_sum (d : ℝ) (h : d = 16) : 
  let r := d / 2
  let sector_area := 2 * (π * r^2 * (45 / 360))
  let triangle_area := 2 * (1 / 2 * r^2)
  sector_area - triangle_area = 32 * π - 64 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_sum_l2818_281802


namespace NUMINAMATH_CALUDE_ariella_daniella_savings_difference_l2818_281856

theorem ariella_daniella_savings_difference :
  ∀ (ariella_initial daniella_savings : ℝ),
    daniella_savings = 400 →
    ariella_initial + ariella_initial * 0.1 * 2 = 720 →
    ariella_initial > daniella_savings →
    ariella_initial - daniella_savings = 200 :=
by
  sorry

end NUMINAMATH_CALUDE_ariella_daniella_savings_difference_l2818_281856


namespace NUMINAMATH_CALUDE_count_squares_in_H_l2818_281825

/-- The set of points (x,y) with integer coordinates satisfying 2 ≤ |x| ≤ 8 and 2 ≤ |y| ≤ 8 -/
def H : Set (ℤ × ℤ) :=
  {p | 2 ≤ |p.1| ∧ |p.1| ≤ 8 ∧ 2 ≤ |p.2| ∧ |p.2| ≤ 8}

/-- A square with vertices in H -/
structure SquareInH where
  vertices : Fin 4 → ℤ × ℤ
  in_H : ∀ i, vertices i ∈ H
  is_square : ∃ (side : ℤ), side ≥ 5 ∧
    (vertices 1).1 - (vertices 0).1 = side ∧
    (vertices 2).1 - (vertices 1).1 = side ∧
    (vertices 3).1 - (vertices 2).1 = -side ∧
    (vertices 0).1 - (vertices 3).1 = -side ∧
    (vertices 1).2 - (vertices 0).2 = side ∧
    (vertices 2).2 - (vertices 1).2 = -side ∧
    (vertices 3).2 - (vertices 2).2 = -side ∧
    (vertices 0).2 - (vertices 3).2 = side

/-- The number of squares with side length at least 5 whose vertices are in H -/
def numSquaresInH : ℕ := sorry

theorem count_squares_in_H : numSquaresInH = 14 := by sorry

end NUMINAMATH_CALUDE_count_squares_in_H_l2818_281825


namespace NUMINAMATH_CALUDE_book_sale_gain_percentage_l2818_281852

def total_cost : ℚ := 420
def cost_loss_book : ℚ := 245
def loss_percentage : ℚ := 15 / 100

theorem book_sale_gain_percentage :
  let cost_gain_book := total_cost - cost_loss_book
  let selling_price := cost_loss_book * (1 - loss_percentage)
  let gain_percentage := (selling_price - cost_gain_book) / cost_gain_book * 100
  gain_percentage = 19 := by sorry

end NUMINAMATH_CALUDE_book_sale_gain_percentage_l2818_281852


namespace NUMINAMATH_CALUDE_students_in_both_clubs_is_40_l2818_281889

/-- The number of students in both photography and science clubs -/
def students_in_both_clubs (total : ℕ) (photo : ℕ) (science : ℕ) (either : ℕ) : ℕ :=
  photo + science - either

/-- Theorem: Given the conditions from the problem, prove that there are 40 students in both clubs -/
theorem students_in_both_clubs_is_40 :
  students_in_both_clubs 300 120 140 220 = 40 := by
  sorry

end NUMINAMATH_CALUDE_students_in_both_clubs_is_40_l2818_281889


namespace NUMINAMATH_CALUDE_distance_sf_to_atlantis_l2818_281817

theorem distance_sf_to_atlantis : 
  let sf : ℂ := 0
  let atlantis : ℂ := 1300 + 3120 * I
  Complex.abs (atlantis - sf) = 3380 := by
sorry

end NUMINAMATH_CALUDE_distance_sf_to_atlantis_l2818_281817


namespace NUMINAMATH_CALUDE_equal_pair_proof_l2818_281808

theorem equal_pair_proof : (-4)^3 = -4^3 := by
  sorry

end NUMINAMATH_CALUDE_equal_pair_proof_l2818_281808


namespace NUMINAMATH_CALUDE_monitor_height_l2818_281890

theorem monitor_height 
  (width : ℝ) 
  (pixel_density : ℝ) 
  (total_pixels : ℝ) 
  (h1 : width = 21)
  (h2 : pixel_density = 100)
  (h3 : total_pixels = 2520000) :
  (total_pixels / (width * pixel_density)) / pixel_density = 12 := by
  sorry

end NUMINAMATH_CALUDE_monitor_height_l2818_281890


namespace NUMINAMATH_CALUDE_quadratic_one_root_from_geometric_sequence_l2818_281853

/-- If a, b, c form a geometric sequence of real numbers, then ax^2 + bx + c has exactly one real root -/
theorem quadratic_one_root_from_geometric_sequence (a b c : ℝ) : 
  (∃ r : ℝ, b = a * r ∧ c = b * r) → 
  ∃! x : ℝ, a * x^2 + b * x + c = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_one_root_from_geometric_sequence_l2818_281853


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_relation_l2818_281875

/-- Represents a parabola with equation y^2 = 8px where p > 0 -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- The focus of a parabola -/
def focus (para : Parabola) : ℝ × ℝ := (2 * para.p, 0)

/-- The x-coordinate of the directrix of a parabola -/
def directrix_x (para : Parabola) : ℝ := -2 * para.p

/-- The distance from the focus to the directrix -/
def focus_directrix_distance (para : Parabola) : ℝ :=
  (focus para).1 - directrix_x para

theorem parabola_focus_directrix_relation (para : Parabola) :
  para.p = (1/4) * focus_directrix_distance para := by sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_relation_l2818_281875


namespace NUMINAMATH_CALUDE_problem_statement_l2818_281884

-- Define the proposition p
def p : Prop := ∃ x₀ : ℝ, x₀ > 0 ∧ 3^x₀ + x₀ = 2016

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x| - a * x

-- Define the proposition q
def q : Prop := ∃ a : ℝ, a > 0 ∧ ∀ x : ℝ, f a x = f a (-x)

-- State the theorem
theorem problem_statement : p ∧ ¬q := by sorry

end NUMINAMATH_CALUDE_problem_statement_l2818_281884


namespace NUMINAMATH_CALUDE_charcoal_drawings_l2818_281859

theorem charcoal_drawings (total : ℕ) (colored_pencil : ℕ) (blending_marker : ℕ)
  (h1 : total = 25)
  (h2 : colored_pencil = 14)
  (h3 : blending_marker = 7)
  (h4 : total = colored_pencil + blending_marker + (total - colored_pencil - blending_marker)) :
  total - colored_pencil - blending_marker = 4 := by
sorry

end NUMINAMATH_CALUDE_charcoal_drawings_l2818_281859


namespace NUMINAMATH_CALUDE_total_cds_l2818_281899

def dawn_cds : ℕ := 10
def kristine_cds : ℕ := dawn_cds + 7

theorem total_cds : dawn_cds + kristine_cds = 27 := by
  sorry

end NUMINAMATH_CALUDE_total_cds_l2818_281899


namespace NUMINAMATH_CALUDE_books_remaining_l2818_281851

/-- Calculates the number of books remaining in Tracy's charity book store -/
theorem books_remaining (initial_books : ℕ) (donors : ℕ) (books_per_donor : ℕ) (borrowed_books : ℕ) : 
  initial_books = 300 → 
  donors = 10 → 
  books_per_donor = 5 → 
  borrowed_books = 140 → 
  initial_books + donors * books_per_donor - borrowed_books = 210 := by
sorry

end NUMINAMATH_CALUDE_books_remaining_l2818_281851


namespace NUMINAMATH_CALUDE_orange_pyramid_count_l2818_281881

def pyramid_oranges (base_length : ℕ) (base_width : ℕ) (top_oranges : ℕ) : ℕ :=
  let layers := min base_length base_width
  (layers * (base_length + base_width - layers + 1) * (2 * base_length + 2 * base_width - 3 * layers + 1)) / 6 + top_oranges

theorem orange_pyramid_count : pyramid_oranges 7 10 3 = 227 := by
  sorry

end NUMINAMATH_CALUDE_orange_pyramid_count_l2818_281881


namespace NUMINAMATH_CALUDE_cos_neg_600_degrees_l2818_281854

theorem cos_neg_600_degrees : Real.cos ((-600 : ℝ) * Real.pi / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_neg_600_degrees_l2818_281854


namespace NUMINAMATH_CALUDE_remaining_money_l2818_281815

def salary : ℝ := 8123.08

theorem remaining_money (food_fraction : ℝ) (rent_fraction : ℝ) (clothes_fraction : ℝ)
  (h_food : food_fraction = 1/3)
  (h_rent : rent_fraction = 1/4)
  (h_clothes : clothes_fraction = 1/5) :
  let total_expenses := salary * (food_fraction + rent_fraction + clothes_fraction)
  ∃ ε > 0, |salary - total_expenses - 1759.00| < ε :=
by sorry

end NUMINAMATH_CALUDE_remaining_money_l2818_281815


namespace NUMINAMATH_CALUDE_dexter_card_boxes_l2818_281803

theorem dexter_card_boxes (x : ℕ) : 
  (15 * x + 20 * (x - 3) = 255) → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_dexter_card_boxes_l2818_281803


namespace NUMINAMATH_CALUDE_average_age_of_ten_students_l2818_281809

theorem average_age_of_ten_students
  (total_students : Nat)
  (average_age_all : ℝ)
  (num_group1 : Nat)
  (average_age_group1 : ℝ)
  (age_last_student : ℝ)
  (h1 : total_students = 15)
  (h2 : average_age_all = 15)
  (h3 : num_group1 = 4)
  (h4 : average_age_group1 = 14)
  (h5 : age_last_student = 9)
  : (total_students * average_age_all - num_group1 * average_age_group1 - age_last_student) / (total_students - num_group1 - 1) = 16 := by
  sorry

#check average_age_of_ten_students

end NUMINAMATH_CALUDE_average_age_of_ten_students_l2818_281809


namespace NUMINAMATH_CALUDE_mixture_cost_ratio_l2818_281893

/-- Given the conditions of the mixture problem, prove that the ratio of nut cost to raisin cost is 3:1 -/
theorem mixture_cost_ratio (R N : ℝ) (h1 : R > 0) (h2 : N > 0) : 
  3 * R = 0.25 * (3 * R + 3 * N) → N / R = 3 := by
  sorry

end NUMINAMATH_CALUDE_mixture_cost_ratio_l2818_281893


namespace NUMINAMATH_CALUDE_divisibility_by_35_l2818_281864

theorem divisibility_by_35 : ∃! n : ℕ, n < 10 ∧ 35 ∣ (80000 + 10000 * n + 975) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_35_l2818_281864


namespace NUMINAMATH_CALUDE_disjoint_sets_imply_t_bounds_l2818_281800

/-- The set M defined by the inequality x^3 + 8y^3 + 6xy ≥ 1 -/
def M : Set (ℝ × ℝ) := {p | p.1^3 + 8*p.2^3 + 6*p.1*p.2 ≥ 1}

/-- The set D defined by the inequality x^2 + y^2 ≤ t^2 -/
def D (t : ℝ) : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 ≤ t^2}

/-- Theorem stating that if D and M are disjoint, then -√5/5 < t < √5/5 -/
theorem disjoint_sets_imply_t_bounds (t : ℝ) (h : t ≠ 0) :
  D t ∩ M = ∅ → -Real.sqrt 5 / 5 < t ∧ t < Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_disjoint_sets_imply_t_bounds_l2818_281800


namespace NUMINAMATH_CALUDE_remaining_trip_time_l2818_281828

/-- Proves the remaining time of a trip given specific conditions -/
theorem remaining_trip_time 
  (total_time : ℝ) 
  (original_speed : ℝ) 
  (first_part_time : ℝ) 
  (first_part_speed : ℝ) 
  (remaining_speed : ℝ) 
  (h1 : total_time = 7.25)
  (h2 : original_speed = 50)
  (h3 : first_part_time = 2)
  (h4 : first_part_speed = 80)
  (h5 : remaining_speed = 40) :
  let total_distance := total_time * original_speed
  let first_part_distance := first_part_time * first_part_speed
  let remaining_distance := total_distance - first_part_distance
  remaining_distance / remaining_speed = 5.0625 := by
  sorry

end NUMINAMATH_CALUDE_remaining_trip_time_l2818_281828


namespace NUMINAMATH_CALUDE_multiple_of_nine_implies_multiple_of_three_l2818_281865

theorem multiple_of_nine_implies_multiple_of_three (n : ℤ) :
  (∀ m : ℤ, 9 ∣ m → 3 ∣ m) →
  (∃ k : ℤ, n = 9 * k ∧ n % 2 = 1) →
  3 ∣ n :=
by sorry

end NUMINAMATH_CALUDE_multiple_of_nine_implies_multiple_of_three_l2818_281865


namespace NUMINAMATH_CALUDE_equal_intercept_line_equation_l2818_281880

/-- A line passing through (1, 2) with equal intercepts on both axes -/
def EqualInterceptLine : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 = 1 ∧ p.2 = 2) ∨ 
               (p.2 = 2 * p.1) ∨ 
               (p.1 + p.2 = 3)}

theorem equal_intercept_line_equation :
  ∀ (x y : ℝ), (x, y) ∈ EqualInterceptLine ↔ (y = 2 * x ∨ x + y = 3) :=
by sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equation_l2818_281880


namespace NUMINAMATH_CALUDE_sum_of_digits_of_seven_to_eleven_l2818_281872

/-- The sum of the tens digit and the ones digit of (3+4)^11 is 7 -/
theorem sum_of_digits_of_seven_to_eleven : 
  let n : ℕ := (3 + 4)^11
  let ones_digit : ℕ := n % 10
  let tens_digit : ℕ := (n / 10) % 10
  ones_digit + tens_digit = 7 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_seven_to_eleven_l2818_281872


namespace NUMINAMATH_CALUDE_megans_earnings_l2818_281878

/-- Calculates the total earnings for a given number of months based on daily work hours, hourly rate, and days worked per month. -/
def total_earnings (hours_per_day : ℕ) (hourly_rate : ℚ) (days_per_month : ℕ) (months : ℕ) : ℚ :=
  hours_per_day * hourly_rate * days_per_month * months

/-- Proves that Megan's total earnings for two months of work is $2400. -/
theorem megans_earnings :
  total_earnings 8 (15/2) 20 2 = 2400 := by
  sorry

#eval total_earnings 8 (15/2) 20 2

end NUMINAMATH_CALUDE_megans_earnings_l2818_281878


namespace NUMINAMATH_CALUDE_prime_extension_l2818_281897

theorem prime_extension (n : ℕ) (h1 : n ≥ 2) 
  (h2 : ∀ k : ℕ, 0 ≤ k ∧ k ≤ Real.sqrt (n / 3) → Nat.Prime (k^2 + k + n)) :
  ∀ k : ℕ, 0 ≤ k ∧ k ≤ n - 2 → Nat.Prime (k^2 + k + n) := by
  sorry

end NUMINAMATH_CALUDE_prime_extension_l2818_281897


namespace NUMINAMATH_CALUDE_segments_covered_by_q_at_most_q_plus_one_l2818_281882

/-- A half-line on the real number line -/
structure HalfLine where
  endpoint : ℝ
  direction : Bool -- true for right-infinite, false for left-infinite

/-- A configuration of half-lines on the real number line -/
def Configuration := List HalfLine

/-- A segment on the real number line -/
structure Segment where
  left : ℝ
  right : ℝ

/-- Count the number of half-lines covering a given segment -/
def coverCount (config : Configuration) (seg : Segment) : ℕ :=
  sorry

/-- The segments formed by the endpoints of the half-lines -/
def segments (config : Configuration) : List Segment :=
  sorry

/-- The segments covered by exactly q half-lines -/
def segmentsCoveredByQ (config : Configuration) (q : ℕ) : List Segment :=
  sorry

/-- The main theorem -/
theorem segments_covered_by_q_at_most_q_plus_one (config : Configuration) (q : ℕ) :
  (segmentsCoveredByQ config q).length ≤ q + 1 :=
  sorry

end NUMINAMATH_CALUDE_segments_covered_by_q_at_most_q_plus_one_l2818_281882


namespace NUMINAMATH_CALUDE_y_intercept_of_parallel_line_l2818_281807

/-- A line in two-dimensional space. -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Returns true if two lines are parallel. -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

/-- Returns true if a point lies on a line. -/
def pointOnLine (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.yIntercept

/-- The given line y = -3x + 6 -/
def givenLine : Line :=
  { slope := -3, yIntercept := 6 }

theorem y_intercept_of_parallel_line :
  ∀ (b : Line),
    parallel b givenLine →
    pointOnLine b 3 (-4) →
    b.yIntercept = 5 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_parallel_line_l2818_281807


namespace NUMINAMATH_CALUDE_distance_AB_l2818_281834

/-- The equation of the line -/
def line_equation (x y : ℝ) : Prop := x - 2*y + 6 = 0

/-- The x-coordinate of point A (x-axis intersection) -/
def point_A : ℝ := -6

/-- The y-coordinate of point B (y-axis intersection) -/
def point_B : ℝ := 3

/-- Theorem stating that the distance between points A and B is 3√5 -/
theorem distance_AB :
  let A : ℝ × ℝ := (point_A, 0)
  let B : ℝ × ℝ := (0, point_B)
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 3 * Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_distance_AB_l2818_281834


namespace NUMINAMATH_CALUDE_total_marbles_l2818_281821

theorem total_marbles (num_boxes : ℕ) (marbles_per_box : ℕ) 
  (h1 : num_boxes = 10) (h2 : marbles_per_box = 100) : 
  num_boxes * marbles_per_box = 1000 := by
  sorry

end NUMINAMATH_CALUDE_total_marbles_l2818_281821


namespace NUMINAMATH_CALUDE_train_length_l2818_281874

/-- The length of a train given relative speeds and passing time -/
theorem train_length (v1 v2 t : ℝ) (h1 : v1 = 36) (h2 : v2 = 45) (h3 : t = 4) :
  (v1 + v2) * (5 / 18) * t = 90 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l2818_281874


namespace NUMINAMATH_CALUDE_division_remainder_problem_l2818_281823

theorem division_remainder_problem (D : ℕ) : 
  D = 12 * 63 + (D % 12) →  -- Incorrect division equation
  D = 21 * 36 + (D % 21) →  -- Correct division equation
  D % 21 = 0 :=             -- Remainder of correct division is 0
by sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l2818_281823


namespace NUMINAMATH_CALUDE_equal_area_line_slope_l2818_281885

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The problem setup -/
def circles : List Circle := [
  { center := (10, 80), radius := 4 },
  { center := (13, 60), radius := 4 },
  { center := (15, 70), radius := 4 }
]

/-- A line passing through a given point -/
structure Line where
  slope : ℝ
  passesThrough : ℝ × ℝ

/-- Checks if a line divides the total area of circles equally -/
def dividesAreaEqually (l : Line) (cs : List Circle) : Prop := sorry

/-- The main theorem -/
theorem equal_area_line_slope :
  ∃ l : Line, l.passesThrough = (13, 60) ∧ 
    dividesAreaEqually l circles ∧ 
    abs l.slope = 5 := by sorry

end NUMINAMATH_CALUDE_equal_area_line_slope_l2818_281885


namespace NUMINAMATH_CALUDE_audiobook_listening_time_l2818_281871

/-- Calculates the average daily listening time for audiobooks -/
def average_daily_listening_time (num_audiobooks : ℕ) (audiobook_length : ℕ) (total_days : ℕ) : ℚ :=
  (num_audiobooks * audiobook_length : ℚ) / total_days

/-- Proves that the average daily listening time is 2 hours given the specific conditions -/
theorem audiobook_listening_time :
  let num_audiobooks : ℕ := 6
  let audiobook_length : ℕ := 30
  let total_days : ℕ := 90
  average_daily_listening_time num_audiobooks audiobook_length total_days = 2 := by
  sorry

end NUMINAMATH_CALUDE_audiobook_listening_time_l2818_281871


namespace NUMINAMATH_CALUDE_least_n_factorial_divisible_by_8_l2818_281842

theorem least_n_factorial_divisible_by_8 : 
  ∃ n : ℕ, n > 0 ∧ 8 ∣ n.factorial ∧ ∀ m : ℕ, m > 0 → m < n → ¬(8 ∣ m.factorial) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_least_n_factorial_divisible_by_8_l2818_281842


namespace NUMINAMATH_CALUDE_michaels_initial_money_l2818_281811

theorem michaels_initial_money (M : ℝ) : 
  (17 + M / 2 - 3 = 35) → M = 42 := by
  sorry

end NUMINAMATH_CALUDE_michaels_initial_money_l2818_281811


namespace NUMINAMATH_CALUDE_skittles_per_friend_l2818_281858

def total_skittles : ℕ := 40
def num_friends : ℕ := 5

theorem skittles_per_friend :
  total_skittles / num_friends = 8 := by sorry

end NUMINAMATH_CALUDE_skittles_per_friend_l2818_281858


namespace NUMINAMATH_CALUDE_det_special_matrix_l2818_281818

theorem det_special_matrix (k a b : ℝ) :
  let M : Matrix (Fin 3) (Fin 3) ℝ := ![![1, k * Real.sin (a - b), Real.sin a],
                                        ![k * Real.sin (a - b), 1, k * Real.sin b],
                                        ![Real.sin a, k * Real.sin b, 1]]
  Matrix.det M = 1 - Real.sin a ^ 2 - k ^ 2 * Real.sin b ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_det_special_matrix_l2818_281818


namespace NUMINAMATH_CALUDE_negative_power_division_l2818_281850

theorem negative_power_division : -2^5 / (-2)^3 = 4 := by sorry

end NUMINAMATH_CALUDE_negative_power_division_l2818_281850


namespace NUMINAMATH_CALUDE_cube_prime_factorization_l2818_281820

theorem cube_prime_factorization (x y : ℕ+) (p : ℕ) :
  (x + y) * (x^2 + 9*y) = p^3 ∧ Nat.Prime p ↔ (x = 2 ∧ y = 5) ∨ (x = 4 ∧ y = 1) := by
  sorry

end NUMINAMATH_CALUDE_cube_prime_factorization_l2818_281820


namespace NUMINAMATH_CALUDE_optimal_strategy_l2818_281837

-- Define the warehouse options
structure Warehouse where
  monthly_rent : ℝ
  repossession_probability : ℝ
  repossession_time : ℕ

-- Define the company's parameters
structure Company where
  planning_horizon : ℕ
  moving_cost : ℝ

-- Define the purchase option
structure PurchaseOption where
  total_price : ℝ
  installment_period : ℕ

def calculate_total_cost (w : Warehouse) (c : Company) (years : ℕ) : ℝ :=
  sorry

def calculate_purchase_cost (p : PurchaseOption) : ℝ :=
  sorry

theorem optimal_strategy (w1 w2 : Warehouse) (c : Company) (p : PurchaseOption) :
  w1.monthly_rent = 80000 ∧
  w2.monthly_rent = 20000 ∧
  w2.repossession_probability = 0.5 ∧
  w2.repossession_time = 5 ∧
  c.planning_horizon = 60 ∧
  c.moving_cost = 150000 ∧
  p.total_price = 3000000 ∧
  p.installment_period = 36 →
  calculate_total_cost w2 c 1 + calculate_purchase_cost p <
  min (calculate_total_cost w1 c 5) (calculate_total_cost w2 c 5) :=
sorry

#check optimal_strategy

end NUMINAMATH_CALUDE_optimal_strategy_l2818_281837


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2818_281896

theorem negation_of_proposition (x₀ : ℝ) : 
  ¬(x₀^2 + 2*x₀ + 2 ≤ 0) ↔ (x₀^2 + 2*x₀ + 2 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2818_281896


namespace NUMINAMATH_CALUDE_max_pens_sold_is_226_l2818_281879

/-- Represents the store's promotional sale --/
structure PromotionalSale where
  penProfit : ℕ            -- Profit per pen in yuan
  teddyBearCost : ℕ        -- Cost of teddy bear in yuan
  pensPerPackage : ℕ       -- Number of pens in a promotional package
  totalProfit : ℕ          -- Total profit from the promotion in yuan

/-- Calculates the maximum number of pens sold during the promotional sale --/
def maxPensSold (sale : PromotionalSale) : ℕ :=
  sorry

/-- Theorem stating that for the given promotional sale conditions, 
    the maximum number of pens sold is 226 --/
theorem max_pens_sold_is_226 :
  let sale : PromotionalSale := {
    penProfit := 9
    teddyBearCost := 2
    pensPerPackage := 4
    totalProfit := 1922
  }
  maxPensSold sale = 226 := by
  sorry

end NUMINAMATH_CALUDE_max_pens_sold_is_226_l2818_281879


namespace NUMINAMATH_CALUDE_square_sum_equals_nineteen_l2818_281849

theorem square_sum_equals_nineteen (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 3) :
  x^2 + y^2 = 19 := by
sorry

end NUMINAMATH_CALUDE_square_sum_equals_nineteen_l2818_281849


namespace NUMINAMATH_CALUDE_interior_angle_sum_difference_l2818_281805

/-- The sum of interior angles of an n-sided polygon -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- Theorem: The difference in sum of interior angles between an (n+1)-sided polygon and an n-sided polygon is 180° -/
theorem interior_angle_sum_difference (n : ℕ) (h : n ≥ 3) :
  sum_interior_angles (n + 1) - sum_interior_angles n = 180 := by
  sorry

end NUMINAMATH_CALUDE_interior_angle_sum_difference_l2818_281805


namespace NUMINAMATH_CALUDE_betty_order_cost_l2818_281855

/-- The total cost of Betty's order -/
def total_cost (slippers_quantity : ℕ) (slippers_price : ℚ) 
               (lipstick_quantity : ℕ) (lipstick_price : ℚ)
               (hair_color_quantity : ℕ) (hair_color_price : ℚ) : ℚ :=
  slippers_quantity * slippers_price + 
  lipstick_quantity * lipstick_price + 
  hair_color_quantity * hair_color_price

/-- Theorem stating that Betty's total order cost is $44 -/
theorem betty_order_cost : 
  total_cost 6 (5/2) 4 (5/4) 8 3 = 44 := by
  sorry

#eval total_cost 6 (5/2) 4 (5/4) 8 3

end NUMINAMATH_CALUDE_betty_order_cost_l2818_281855
