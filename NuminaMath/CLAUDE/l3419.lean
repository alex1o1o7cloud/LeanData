import Mathlib

namespace NUMINAMATH_CALUDE_union_of_A_and_complement_of_B_l3419_341967

open Set

def A : Set ℝ := {x : ℝ | |x - 2| ≤ 1}
def B : Set ℝ := {x : ℝ | Real.exp (x - 1) ≥ 1}

theorem union_of_A_and_complement_of_B :
  A ∪ (univ \ B) = Iic 3 :=
sorry

end NUMINAMATH_CALUDE_union_of_A_and_complement_of_B_l3419_341967


namespace NUMINAMATH_CALUDE_line_slope_l3419_341978

theorem line_slope (x y : ℝ) : x + 2 * y - 6 = 0 → (y - 3) = (-1/2) * x := by
  sorry

end NUMINAMATH_CALUDE_line_slope_l3419_341978


namespace NUMINAMATH_CALUDE_alcohol_dilution_l3419_341905

theorem alcohol_dilution (initial_volume : ℝ) (initial_concentration : ℝ) 
  (final_concentration : ℝ) (water_added : ℝ) : 
  initial_volume = 15 → 
  initial_concentration = 0.6 → 
  final_concentration = 0.4 → 
  water_added = 7.5 → 
  (initial_volume * initial_concentration) / (initial_volume + water_added) = final_concentration :=
by
  sorry

#check alcohol_dilution

end NUMINAMATH_CALUDE_alcohol_dilution_l3419_341905


namespace NUMINAMATH_CALUDE_aarons_brothers_count_l3419_341925

/-- The number of Aaron's brothers -/
def aarons_brothers : ℕ := sorry

/-- The number of Bennett's brothers -/
def bennetts_brothers : ℕ := 6

theorem aarons_brothers_count : aarons_brothers = 4 :=
  by
  have h : bennetts_brothers = 2 * aarons_brothers - 2 := sorry
  sorry

#check aarons_brothers_count

end NUMINAMATH_CALUDE_aarons_brothers_count_l3419_341925


namespace NUMINAMATH_CALUDE_sin_sum_identity_l3419_341999

theorem sin_sum_identity (x : ℝ) (h : Real.sin (x + π / 6) = 1 / 3) :
  Real.sin (x - 5 * π / 6) + Real.sin (π / 3 - x) ^ 2 = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_identity_l3419_341999


namespace NUMINAMATH_CALUDE_fraction_subtraction_simplification_l3419_341942

theorem fraction_subtraction_simplification :
  8 / 21 - 3 / 63 = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_simplification_l3419_341942


namespace NUMINAMATH_CALUDE_math_class_size_l3419_341960

theorem math_class_size (total : ℕ) (both : ℕ) :
  total = 75 →
  both = 10 →
  ∃ (math physics : ℕ),
    total = math + physics - both ∧
    math = 2 * physics →
    math = 56 := by
  sorry

end NUMINAMATH_CALUDE_math_class_size_l3419_341960


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l3419_341912

/-- Given two vectors a and b in ℝ², prove that if a = (1, 2) and b = (m, 1) are perpendicular, then m = -2 -/
theorem perpendicular_vectors_m_value (m : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![m, 1]
  (∀ i, i < 2 → a i * b i = 0) → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l3419_341912


namespace NUMINAMATH_CALUDE_basketball_shot_probability_l3419_341976

theorem basketball_shot_probability (a b c : ℝ) : 
  a ∈ (Set.Ioo 0 1) →
  b ∈ (Set.Ioo 0 1) →
  c ∈ (Set.Ioo 0 1) →
  a + b + c = 1 →
  3*a + 2*b = 2 →
  ∀ x y : ℝ, x ∈ (Set.Ioo 0 1) → y ∈ (Set.Ioo 0 1) → x + y < 1 → x * y ≤ a * b →
  a * b ≤ 1/6 :=
by sorry

end NUMINAMATH_CALUDE_basketball_shot_probability_l3419_341976


namespace NUMINAMATH_CALUDE_pyramid_fifth_face_sum_l3419_341911

/-- Represents a labeling of a square-based pyramid -/
structure PyramidLabeling where
  vertices : Fin 5 → Nat
  sum_to_15 : (vertices 0) + (vertices 1) + (vertices 2) + (vertices 3) + (vertices 4) = 15
  all_different : ∀ i j, i ≠ j → vertices i ≠ vertices j

/-- Represents the sums of faces in the pyramid -/
structure FaceSums (l : PyramidLabeling) where
  sums : Fin 5 → Nat
  four_given_sums : {7, 8, 9, 10} ⊆ (Finset.image sums Finset.univ)

theorem pyramid_fifth_face_sum (l : PyramidLabeling) (s : FaceSums l) :
  ∃ i, s.sums i = 13 :=
sorry

end NUMINAMATH_CALUDE_pyramid_fifth_face_sum_l3419_341911


namespace NUMINAMATH_CALUDE_probability_both_asian_l3419_341954

def asian_countries : ℕ := 3
def european_countries : ℕ := 3
def total_countries : ℕ := asian_countries + european_countries
def countries_to_select : ℕ := 2

def total_outcomes : ℕ := (total_countries.choose countries_to_select)
def favorable_outcomes : ℕ := (asian_countries.choose countries_to_select)

theorem probability_both_asian :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_probability_both_asian_l3419_341954


namespace NUMINAMATH_CALUDE_number_of_smaller_cubes_l3419_341955

theorem number_of_smaller_cubes (surface_area : ℝ) (small_cube_volume : ℝ) : 
  surface_area = 5400 → small_cube_volume = 216 → 
  (surface_area / 6).sqrt ^ 3 / small_cube_volume = 125 := by
  sorry

end NUMINAMATH_CALUDE_number_of_smaller_cubes_l3419_341955


namespace NUMINAMATH_CALUDE_eleventh_term_is_192_l3419_341950

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

/-- The 5th term of the sequence is 3 -/
def FifthTerm (a : ℕ → ℝ) : Prop := a 5 = 3

/-- The 8th term of the sequence is 24 -/
def EighthTerm (a : ℕ → ℝ) : Prop := a 8 = 24

theorem eleventh_term_is_192 (a : ℕ → ℝ) 
  (h_geo : GeometricSequence a) 
  (h_fifth : FifthTerm a) 
  (h_eighth : EighthTerm a) : 
  a 11 = 192 := by
  sorry

end NUMINAMATH_CALUDE_eleventh_term_is_192_l3419_341950


namespace NUMINAMATH_CALUDE_hexagon_pillar_height_l3419_341993

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a regular hexagon with pillars -/
structure HexagonWithPillars where
  sideLength : ℝ
  A : Point3D
  B : Point3D
  C : Point3D
  E : Point3D

/-- The theorem to be proved -/
theorem hexagon_pillar_height 
  (h : HexagonWithPillars) 
  (h_side : h.sideLength > 0)
  (h_A : h.A = ⟨0, 0, 12⟩)
  (h_B : h.B = ⟨h.sideLength, 0, 9⟩)
  (h_C : h.C = ⟨h.sideLength / 2, h.sideLength * Real.sqrt 3 / 2, 10⟩)
  (h_E : h.E = ⟨-h.sideLength, 0, h.E.z⟩) :
  h.E.z = 17 := by
  sorry


end NUMINAMATH_CALUDE_hexagon_pillar_height_l3419_341993


namespace NUMINAMATH_CALUDE_abc_problem_l3419_341986

theorem abc_problem (a b c : ℝ) 
  (h1 : 2011 * (a + b + c) = 1) 
  (h2 : a * b + a * c + b * c = 2011 * a * b * c) : 
  a^2011 * b^2011 + c^2011 = (1 : ℝ) / 2011^2011 := by
  sorry

end NUMINAMATH_CALUDE_abc_problem_l3419_341986


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_sum_l3419_341956

theorem geometric_sequence_ratio_sum (k p r : ℝ) (h1 : k ≠ 0) (h2 : p ≠ r) :
  k * p^2 - k * r^2 = 5 * (k * p - k * r) → p + r = 5 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_sum_l3419_341956


namespace NUMINAMATH_CALUDE_mod_congruence_unique_solution_l3419_341988

theorem mod_congruence_unique_solution : ∃! n : ℕ, n ≤ 19 ∧ n ≡ -5678 [ZMOD 20] ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_mod_congruence_unique_solution_l3419_341988


namespace NUMINAMATH_CALUDE_max_area_rectangle_perimeter_40_l3419_341958

/-- The maximum area of a rectangle with perimeter 40 is 100 -/
theorem max_area_rectangle_perimeter_40 :
  ∀ x y : ℝ,
  x > 0 → y > 0 →
  2 * x + 2 * y = 40 →
  x * y ≤ 100 := by
sorry

end NUMINAMATH_CALUDE_max_area_rectangle_perimeter_40_l3419_341958


namespace NUMINAMATH_CALUDE_billboard_average_is_twenty_l3419_341968

/-- The average number of billboards seen per hour over three hours -/
def average_billboards (hour1 hour2 hour3 : ℕ) : ℚ :=
  (hour1 + hour2 + hour3 : ℚ) / 3

/-- Theorem stating that the average number of billboards seen is 20 -/
theorem billboard_average_is_twenty :
  average_billboards 17 20 23 = 20 := by
  sorry

end NUMINAMATH_CALUDE_billboard_average_is_twenty_l3419_341968


namespace NUMINAMATH_CALUDE_digit_sum_problem_l3419_341935

theorem digit_sum_problem (A B C D E F : ℕ) :
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧
  C ≠ D ∧ C ≠ E ∧ C ≠ F ∧
  D ≠ E ∧ D ≠ F ∧
  E ≠ F →
  A ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) ∧
  B ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) ∧
  C ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) ∧
  D ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) ∧
  E ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) ∧
  F ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) →
  2*A + 3*B + 2*C + 2*D + 2*E + 2*F = 47 →
  B = 5 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_problem_l3419_341935


namespace NUMINAMATH_CALUDE_factorization_of_5_power_1985_minus_1_l3419_341930

theorem factorization_of_5_power_1985_minus_1 :
  ∃ (a b c : ℤ),
    (5^1985 - 1 : ℤ) = a * b * c ∧
    a > 5^100 ∧
    b > 5^100 ∧
    c > 5^100 ∧
    a = 5^397 - 1 ∧
    b = 5^794 - 5^596 + 3*5^397 - 5^199 + 1 ∧
    c = 5^794 + 5^596 + 3*5^397 + 5^199 + 1 :=
by sorry

end NUMINAMATH_CALUDE_factorization_of_5_power_1985_minus_1_l3419_341930


namespace NUMINAMATH_CALUDE_expression_equality_l3419_341980

theorem expression_equality : 
  Real.sqrt 12 - 3 * Real.sqrt (1/3) + Real.sqrt 27 + (Real.pi + 1)^0 = 4 * Real.sqrt 3 + 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3419_341980


namespace NUMINAMATH_CALUDE_one_more_stork_than_birds_l3419_341914

/-- Given the initial conditions of birds and storks on a fence, prove that there is one more stork than birds. -/
theorem one_more_stork_than_birds : 
  let initial_birds : ℕ := 3
  let additional_birds : ℕ := 2
  let storks : ℕ := 6
  let total_birds : ℕ := initial_birds + additional_birds
  storks - total_birds = 1 := by sorry

end NUMINAMATH_CALUDE_one_more_stork_than_birds_l3419_341914


namespace NUMINAMATH_CALUDE_books_from_second_shop_l3419_341998

/- Define the problem parameters -/
def books_shop1 : ℕ := 50
def cost_shop1 : ℕ := 1000
def cost_shop2 : ℕ := 800
def avg_price : ℕ := 20

/- Define the function to calculate the number of books from the second shop -/
def books_shop2 : ℕ :=
  (cost_shop1 + cost_shop2) / avg_price - books_shop1

/- Theorem statement -/
theorem books_from_second_shop :
  books_shop2 = 40 :=
sorry

end NUMINAMATH_CALUDE_books_from_second_shop_l3419_341998


namespace NUMINAMATH_CALUDE_composite_product_division_l3419_341919

def first_eight_composites : List Nat := [4, 6, 8, 9, 10, 12, 14, 15]
def next_eight_composites : List Nat := [16, 18, 20, 21, 22, 24, 25, 26]

def product_of_list (l : List Nat) : Nat :=
  l.foldl (· * ·) 1

theorem composite_product_division :
  (product_of_list first_eight_composites) / 
  (product_of_list next_eight_composites) = 1 / 3120 := by
  sorry

end NUMINAMATH_CALUDE_composite_product_division_l3419_341919


namespace NUMINAMATH_CALUDE_prob_at_most_one_eq_seven_twenty_sevenths_l3419_341915

/-- The probability of making a basket -/
def p : ℚ := 2/3

/-- The number of attempts -/
def n : ℕ := 3

/-- The probability of making exactly k successful shots in n attempts -/
def binomial_prob (k : ℕ) : ℚ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- The probability of making at most 1 successful shot in 3 attempts -/
def prob_at_most_one : ℚ :=
  binomial_prob 0 + binomial_prob 1

theorem prob_at_most_one_eq_seven_twenty_sevenths : 
  prob_at_most_one = 7/27 := by sorry

end NUMINAMATH_CALUDE_prob_at_most_one_eq_seven_twenty_sevenths_l3419_341915


namespace NUMINAMATH_CALUDE_hyperbola_real_axis_length_l3419_341927

/-- Given a hyperbola with equation x²/a² - y²/4 = 1 (a > 0), 
    if a right triangle is formed by its left and right foci and the point (2, 1),
    then the length of its real axis is 2. -/
theorem hyperbola_real_axis_length (a : ℝ) (h1 : a > 0) : 
  let f (x y : ℝ) := x^2 / a^2 - y^2 / 4
  ∃ (c : ℝ), (2 - c) * (2 + c) + 1 * 1 = 0 ∧ 2 * a = 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_real_axis_length_l3419_341927


namespace NUMINAMATH_CALUDE_ice_cream_sundaes_l3419_341940

theorem ice_cream_sundaes (n : ℕ) (k : ℕ) (h1 : n = 8) (h2 : k = 2) :
  Nat.choose n k = 28 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_sundaes_l3419_341940


namespace NUMINAMATH_CALUDE_box_length_with_cubes_l3419_341929

/-- Given a box with dimensions L × 15 × 6 inches that can be filled entirely
    with 90 identical cubes leaving no space unfilled, prove that the length L
    of the box is 27 inches. -/
theorem box_length_with_cubes (L : ℕ) : 
  (∃ (s : ℕ), L * 15 * 6 = 90 * s^3 ∧ s ∣ 15 ∧ s ∣ 6) → L = 27 := by
  sorry

end NUMINAMATH_CALUDE_box_length_with_cubes_l3419_341929


namespace NUMINAMATH_CALUDE_best_marksman_score_l3419_341909

theorem best_marksman_score (team_size : ℕ) (hypothetical_score : ℕ) (hypothetical_average : ℕ) (actual_total : ℕ) : 
  team_size = 6 → 
  hypothetical_score = 92 →
  hypothetical_average = 84 →
  actual_total = 497 →
  ∃ (best_score : ℕ), best_score = 85 ∧ 
    actual_total = (team_size - 1) * hypothetical_average + best_score := by
  sorry

end NUMINAMATH_CALUDE_best_marksman_score_l3419_341909


namespace NUMINAMATH_CALUDE_cookie_distribution_l3419_341904

theorem cookie_distribution (total : ℝ) (blue green red : ℝ) : 
  blue = (1/4) * total ∧ 
  green = (5/9) * (total - blue) → 
  (blue + green) / total = 2/3 := by
sorry

end NUMINAMATH_CALUDE_cookie_distribution_l3419_341904


namespace NUMINAMATH_CALUDE_tire_usage_theorem_l3419_341990

/-- Represents the usage of tires on a car --/
structure TireUsage where
  total_tires : ℕ
  road_tires : ℕ
  total_miles : ℕ

/-- Calculates the miles each tire was used given equal usage --/
def miles_per_tire (usage : TireUsage) : ℕ :=
  (usage.total_miles * usage.road_tires) / usage.total_tires

/-- Theorem stating that for the given car configuration and mileage, each tire was used for 33333 miles --/
theorem tire_usage_theorem (usage : TireUsage) 
  (h1 : usage.total_tires = 6)
  (h2 : usage.road_tires = 4)
  (h3 : usage.total_miles = 50000) :
  miles_per_tire usage = 33333 := by
  sorry

end NUMINAMATH_CALUDE_tire_usage_theorem_l3419_341990


namespace NUMINAMATH_CALUDE_modular_inverse_89_mod_90_l3419_341965

theorem modular_inverse_89_mod_90 : ∃ x : ℕ, 0 ≤ x ∧ x < 90 ∧ (89 * x) % 90 = 1 := by
  sorry

end NUMINAMATH_CALUDE_modular_inverse_89_mod_90_l3419_341965


namespace NUMINAMATH_CALUDE_abs_x_minus_one_lt_two_sufficient_not_necessary_for_x_lt_three_l3419_341943

theorem abs_x_minus_one_lt_two_sufficient_not_necessary_for_x_lt_three :
  (∀ x : ℝ, |x - 1| < 2 → x < 3) ∧
  ¬(∀ x : ℝ, x < 3 → |x - 1| < 2) := by
sorry

end NUMINAMATH_CALUDE_abs_x_minus_one_lt_two_sufficient_not_necessary_for_x_lt_three_l3419_341943


namespace NUMINAMATH_CALUDE_octagon_has_eight_sides_l3419_341945

/-- The number of sides in an octagon -/
def octagon_sides : ℕ := 8

/-- Theorem stating that an octagon has 8 sides -/
theorem octagon_has_eight_sides : octagon_sides = 8 := by
  sorry

end NUMINAMATH_CALUDE_octagon_has_eight_sides_l3419_341945


namespace NUMINAMATH_CALUDE_boys_in_class_l3419_341933

theorem boys_in_class (total : ℕ) (girls_ratio : ℕ) (boys_ratio : ℕ) (boys : ℕ) : 
  total = 56 → 
  girls_ratio = 4 →
  boys_ratio = 3 →
  girls_ratio + boys_ratio = 7 →
  (girls_ratio : ℚ) / (boys_ratio : ℚ) = 4 / 3 →
  boys = (boys_ratio : ℚ) / (girls_ratio + boys_ratio : ℚ) * total →
  boys = 24 := by
sorry

end NUMINAMATH_CALUDE_boys_in_class_l3419_341933


namespace NUMINAMATH_CALUDE_last_digit_of_one_over_two_to_ten_l3419_341951

theorem last_digit_of_one_over_two_to_ten (n : ℕ) : 
  n = 10 → (1 : ℚ) / (2^n : ℚ) * 10^n % 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_one_over_two_to_ten_l3419_341951


namespace NUMINAMATH_CALUDE_full_time_one_year_count_l3419_341991

/-- Represents the number of employees in different categories at company x -/
structure CompanyEmployees where
  total : ℕ
  fullTime : ℕ
  atLeastOneYear : ℕ
  neitherFullTimeNorOneYear : ℕ

/-- The function to calculate the number of full-time employees who have worked at least one year -/
def fullTimeAndOneYear (e : CompanyEmployees) : ℕ :=
  e.total - (e.fullTime + e.atLeastOneYear - e.neitherFullTimeNorOneYear)

/-- Theorem stating the number of full-time employees who have worked at least one year -/
theorem full_time_one_year_count (e : CompanyEmployees) 
  (h1 : e.total = 130)
  (h2 : e.fullTime = 80)
  (h3 : e.atLeastOneYear = 100)
  (h4 : e.neitherFullTimeNorOneYear = 20) :
  fullTimeAndOneYear e = 90 := by
  sorry

end NUMINAMATH_CALUDE_full_time_one_year_count_l3419_341991


namespace NUMINAMATH_CALUDE_correct_students_joined_l3419_341941

/-- The number of students who joined Beth's class -/
def students_joined : ℕ := 30

/-- The initial number of students -/
def initial_students : ℕ := 150

/-- The number of students who left in the final year -/
def students_left : ℕ := 15

/-- The final number of students -/
def final_students : ℕ := 165

/-- Theorem stating that the number of students who joined is correct -/
theorem correct_students_joined :
  initial_students + students_joined - students_left = final_students :=
by sorry

end NUMINAMATH_CALUDE_correct_students_joined_l3419_341941


namespace NUMINAMATH_CALUDE_jeans_savings_theorem_l3419_341961

/-- Calculates the amount saved on a pair of jeans given the original price and discounts -/
def calculate_savings (original_price : ℝ) (sale_discount_percent : ℝ) (coupon_discount : ℝ) (credit_card_discount_percent : ℝ) : ℝ :=
  let price_after_sale := original_price * (1 - sale_discount_percent)
  let price_after_coupon := price_after_sale - coupon_discount
  let final_price := price_after_coupon * (1 - credit_card_discount_percent)
  original_price - final_price

/-- Theorem stating that the savings on the jeans is $44 -/
theorem jeans_savings_theorem :
  calculate_savings 125 0.20 10 0.10 = 44 := by
  sorry

#eval calculate_savings 125 0.20 10 0.10

end NUMINAMATH_CALUDE_jeans_savings_theorem_l3419_341961


namespace NUMINAMATH_CALUDE_alice_probability_after_two_turns_l3419_341973

-- Define the game parameters
def alice_toss_prob : ℚ := 1/2
def alice_keep_prob : ℚ := 1/2
def bob_toss_prob : ℚ := 2/5
def bob_keep_prob : ℚ := 3/5

-- Define the probability that Alice has the ball after two turns
def alice_has_ball_after_two_turns : ℚ := 
  alice_toss_prob * bob_toss_prob + alice_keep_prob * alice_keep_prob

-- Theorem statement
theorem alice_probability_after_two_turns : 
  alice_has_ball_after_two_turns = 9/20 := by sorry

end NUMINAMATH_CALUDE_alice_probability_after_two_turns_l3419_341973


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3419_341921

/-- Given two arithmetic sequences {a_n} and {b_n} with sums S_n and T_n respectively,
    if S_n/T_n = 2n/(3n+1) for all natural numbers n, then a_5/b_5 = 9/14 -/
theorem arithmetic_sequence_ratio (a b : ℕ → ℚ) (S T : ℕ → ℚ) :
  (∀ n : ℕ, S n = (n : ℚ) * (a 1 + a n) / 2) →
  (∀ n : ℕ, T n = (n : ℚ) * (b 1 + b n) / 2) →
  (∀ n : ℕ, S n / T n = (2 * n : ℚ) / (3 * n + 1)) →
  a 5 / b 5 = 9 / 14 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3419_341921


namespace NUMINAMATH_CALUDE_power_of_four_equality_l3419_341918

theorem power_of_four_equality (m : ℕ) : 4^m = 4 * 16^3 * 64^2 → m = 13 := by
  sorry

end NUMINAMATH_CALUDE_power_of_four_equality_l3419_341918


namespace NUMINAMATH_CALUDE_nines_in_hundred_l3419_341987

def count_nines (n : ℕ) : ℕ :=
  (n / 10) + (n / 10)

theorem nines_in_hundred : count_nines 100 = 20 := by sorry

end NUMINAMATH_CALUDE_nines_in_hundred_l3419_341987


namespace NUMINAMATH_CALUDE_class_duty_assignment_l3419_341971

theorem class_duty_assignment (num_boys num_girls : ℕ) 
  (h1 : num_boys = 16) 
  (h2 : num_girls = 14) : 
  num_boys * num_girls = 224 := by
  sorry

end NUMINAMATH_CALUDE_class_duty_assignment_l3419_341971


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3419_341910

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → (x + 1) * Real.exp x > 1) ↔
  (∃ x : ℝ, x > 0 ∧ (x + 1) * Real.exp x ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3419_341910


namespace NUMINAMATH_CALUDE_quadratic_no_real_zeros_l3419_341996

/-- If a, b, and c form a geometric sequence, then ax^2 + bx + c has no real zeros -/
theorem quadratic_no_real_zeros (a b c : ℝ) (h : b^2 = a*c) (ha : a ≠ 0) :
  ∀ x : ℝ, a*x^2 + b*x + c ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_real_zeros_l3419_341996


namespace NUMINAMATH_CALUDE_retailer_profit_percentage_l3419_341936

/-- Represents the problem of calculating the profit percentage for a retailer --/
theorem retailer_profit_percentage 
  (monthly_sales : ℕ)
  (profit_per_item : ℚ)
  (discount_rate : ℚ)
  (break_even_sales : ℚ)
  (h1 : monthly_sales = 100)
  (h2 : profit_per_item = 30)
  (h3 : discount_rate = 0.05)
  (h4 : break_even_sales = 156.86274509803923)
  : ∃ (item_price : ℚ), 
    profit_per_item / item_price = 0.16 :=
by sorry

end NUMINAMATH_CALUDE_retailer_profit_percentage_l3419_341936


namespace NUMINAMATH_CALUDE_no_feasible_distribution_no_feasible_distribution_proof_l3419_341975

/-- Represents a cricket player with their initial average runs and desired increase --/
structure Player where
  initialAvg : ℕ
  desiredIncrease : ℕ

/-- Theorem stating that no feasible distribution exists for the given problem --/
theorem no_feasible_distribution 
  (playerA : Player) 
  (playerB : Player) 
  (playerC : Player) 
  (totalRunsLimit : ℕ) : Prop :=
  playerA.initialAvg = 32 ∧ 
  playerA.desiredIncrease = 4 ∧
  playerB.initialAvg = 45 ∧ 
  playerB.desiredIncrease = 5 ∧
  playerC.initialAvg = 55 ∧ 
  playerC.desiredIncrease = 6 ∧
  totalRunsLimit = 250 →
  ¬∃ (runsA runsB runsC : ℕ),
    (runsA + runsB + runsC ≤ totalRunsLimit) ∧
    ((playerA.initialAvg * 10 + runsA) / 11 ≥ playerA.initialAvg + playerA.desiredIncrease) ∧
    ((playerB.initialAvg * 10 + runsB) / 11 ≥ playerB.initialAvg + playerB.desiredIncrease) ∧
    ((playerC.initialAvg * 10 + runsC) / 11 ≥ playerC.initialAvg + playerC.desiredIncrease)

/-- The proof of the theorem --/
theorem no_feasible_distribution_proof : no_feasible_distribution 
  { initialAvg := 32, desiredIncrease := 4 }
  { initialAvg := 45, desiredIncrease := 5 }
  { initialAvg := 55, desiredIncrease := 6 }
  250 := by
  sorry

end NUMINAMATH_CALUDE_no_feasible_distribution_no_feasible_distribution_proof_l3419_341975


namespace NUMINAMATH_CALUDE_impossible_to_empty_pile_l3419_341917

/-- Represents the state of three piles of stones -/
structure PileState where
  pile1 : ℕ
  pile2 : ℕ
  pile3 : ℕ

/-- Allowed operations on the piles -/
inductive Operation
  | Add : Fin 3 → Operation
  | Remove : Fin 3 → Operation

/-- Applies an operation to a PileState -/
def applyOperation (state : PileState) (op : Operation) : PileState :=
  match op with
  | Operation.Add i => 
      match i with
      | 0 => ⟨state.pile1 + state.pile2 + state.pile3, state.pile2, state.pile3⟩
      | 1 => ⟨state.pile1, state.pile2 + state.pile1 + state.pile3, state.pile3⟩
      | 2 => ⟨state.pile1, state.pile2, state.pile3 + state.pile1 + state.pile2⟩
  | Operation.Remove i =>
      match i with
      | 0 => ⟨state.pile1 - (state.pile2 + state.pile3), state.pile2, state.pile3⟩
      | 1 => ⟨state.pile1, state.pile2 - (state.pile1 + state.pile3), state.pile3⟩
      | 2 => ⟨state.pile1, state.pile2, state.pile3 - (state.pile1 + state.pile2)⟩

/-- Theorem stating that it's impossible to make a pile empty -/
theorem impossible_to_empty_pile (initialState : PileState) 
  (h1 : Odd initialState.pile1) 
  (h2 : Odd initialState.pile2) 
  (h3 : Odd initialState.pile3) :
  ∀ (ops : List Operation), 
    let finalState := ops.foldl applyOperation initialState
    ¬(finalState.pile1 = 0 ∨ finalState.pile2 = 0 ∨ finalState.pile3 = 0) := by
  sorry

end NUMINAMATH_CALUDE_impossible_to_empty_pile_l3419_341917


namespace NUMINAMATH_CALUDE_expand_product_l3419_341995

theorem expand_product (x a : ℝ) : 2 * (x + (a + 2)) * (x + (a - 3)) = 2 * x^2 + (4 * a - 2) * x + 2 * a^2 - 2 * a - 12 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3419_341995


namespace NUMINAMATH_CALUDE_m_range_proof_l3419_341949

-- Define the conditions
def condition_p (x : ℝ) : Prop := x^2 - 3*x - 4 ≤ 0
def condition_q (x m : ℝ) : Prop := x^2 - 6*x + 9 - m^2 ≤ 0

-- Define the range of m
def m_range (m : ℝ) : Prop := m ≤ -4 ∨ m ≥ 4

-- Theorem statement
theorem m_range_proof :
  (∀ x, condition_p x → condition_q x m) ∧
  (∃ x, condition_q x m ∧ ¬condition_p x) →
  m_range m :=
sorry

end NUMINAMATH_CALUDE_m_range_proof_l3419_341949


namespace NUMINAMATH_CALUDE_sqrt_2x_plus_4_real_l3419_341948

theorem sqrt_2x_plus_4_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = 2 * x + 4) ↔ x ≥ -2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2x_plus_4_real_l3419_341948


namespace NUMINAMATH_CALUDE_twentynine_is_perfect_number_pairing_x_squared_minus_6x_plus_13_perfect_number_condition_for_S_l3419_341932

-- Definition of a perfect number
def isPerfectNumber (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = a^2 + b^2 ∧ a > 0 ∧ b > 0

-- Theorem 1
theorem twentynine_is_perfect_number : isPerfectNumber 29 :=
sorry

-- Theorem 2
theorem pairing_x_squared_minus_6x_plus_13 :
  ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧
  (∀ x : ℝ, x^2 - 6*x + 13 = (x - m)^2 + n^2) ∧
  m * n = 6 :=
sorry

-- Theorem 3
theorem perfect_number_condition_for_S (k : ℝ) :
  (∀ x y : ℤ, ∃ a b : ℤ, x^2 + 4*y^2 + 4*x - 12*y + k = a^2 + b^2) ↔ k = 13 :=
sorry

end NUMINAMATH_CALUDE_twentynine_is_perfect_number_pairing_x_squared_minus_6x_plus_13_perfect_number_condition_for_S_l3419_341932


namespace NUMINAMATH_CALUDE_subset_range_l3419_341931

theorem subset_range (a : ℝ) : 
  let A := {x : ℝ | 1 ≤ x ∧ x ≤ a}
  let B := {x : ℝ | 0 < x ∧ x < 5}
  A ⊆ B → (1 ≤ a ∧ a < 5) :=
by
  sorry

end NUMINAMATH_CALUDE_subset_range_l3419_341931


namespace NUMINAMATH_CALUDE_stream_speed_l3419_341992

/-- Given upstream and downstream speeds of a canoe, calculate the speed of the stream. -/
theorem stream_speed (upstream_speed downstream_speed : ℝ) 
  (h_upstream : upstream_speed = 9)
  (h_downstream : downstream_speed = 12) :
  (downstream_speed - upstream_speed) / 2 = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l3419_341992


namespace NUMINAMATH_CALUDE_plot_width_calculation_l3419_341944

/-- Calculates the width of a rectangular plot given tilling parameters -/
theorem plot_width_calculation (plot_length : ℝ) (tilling_time : ℝ) (tiller_width : ℝ) (tilling_rate : ℝ) : 
  plot_length = 120 →
  tilling_time = 220 →
  tiller_width = 2 →
  tilling_rate = 1 / 2 →
  (tilling_time * 60 * tilling_rate * tiller_width) / plot_length = 110 := by
  sorry

end NUMINAMATH_CALUDE_plot_width_calculation_l3419_341944


namespace NUMINAMATH_CALUDE_cubic_root_ratio_l3419_341934

theorem cubic_root_ratio (a b c d : ℝ) (h : ∀ x : ℝ, a * x^3 + b * x^2 + c * x + d = 0 ↔ x = 2 ∨ x = 4 ∨ x = 5) :
  c / d = 19 / 20 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_ratio_l3419_341934


namespace NUMINAMATH_CALUDE_inequality_proof_l3419_341952

theorem inequality_proof (a b x : ℝ) (h1 : a * b > 0) (h2 : 0 < x) (h3 : x < π / 2) :
  (1 + a^2 / Real.sin x) * (1 + b^2 / Real.cos x) ≥ ((1 + Real.sqrt 2 * a * b)^2 * Real.sin (2 * x)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3419_341952


namespace NUMINAMATH_CALUDE_rectangle_area_change_l3419_341922

theorem rectangle_area_change (L B : ℝ) (h₁ : L > 0) (h₂ : B > 0) :
  let A := L * B
  let L' := 1.20 * L
  let B' := 0.95 * B
  let A' := L' * B'
  A' = 1.14 * A := by sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l3419_341922


namespace NUMINAMATH_CALUDE_total_movies_is_nineteen_l3419_341959

/-- The number of movies shown on each screen in a movie theater --/
def movies_per_screen : List Nat := [3, 4, 2, 3, 5, 2]

/-- The total number of movies shown in the theater --/
def total_movies : Nat := movies_per_screen.sum

/-- Theorem stating that the total number of movies shown is 19 --/
theorem total_movies_is_nineteen : total_movies = 19 := by
  sorry

end NUMINAMATH_CALUDE_total_movies_is_nineteen_l3419_341959


namespace NUMINAMATH_CALUDE_horner_v2_equals_10_l3419_341902

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 2x^7 + x^6 + x^4 + x^2 + 1 -/
def f (x : ℝ) : ℝ := 2*x^7 + x^6 + x^4 + x^2 + 1

/-- Coefficients of the polynomial in reverse order -/
def coeffs : List ℝ := [1, 0, 1, 0, 1, 0, 1, 2]

/-- Theorem: V_2 in Horner's method for f(x) when x = 2 is 10 -/
theorem horner_v2_equals_10 : 
  (horner (coeffs.take 3) 2) = 10 := by sorry

end NUMINAMATH_CALUDE_horner_v2_equals_10_l3419_341902


namespace NUMINAMATH_CALUDE_all_statements_equivalent_l3419_341963

-- Define the propositions
variable (P Q : Prop)

-- Define the equivalence of all statements
theorem all_statements_equivalent :
  (P ↔ Q) ↔ (P → Q) ∧ (Q → P) ∧ (¬Q → ¬P) ∧ (¬P ∨ Q) :=
sorry

end NUMINAMATH_CALUDE_all_statements_equivalent_l3419_341963


namespace NUMINAMATH_CALUDE_shirt_cost_l3419_341939

theorem shirt_cost (num_shirts : ℕ) (num_jeans : ℕ) (total_earnings : ℕ) :
  num_shirts = 20 →
  num_jeans = 10 →
  total_earnings = 400 →
  ∃ (shirt_cost : ℕ),
    shirt_cost * num_shirts + (2 * shirt_cost) * num_jeans = total_earnings ∧
    shirt_cost = 10 :=
by sorry

end NUMINAMATH_CALUDE_shirt_cost_l3419_341939


namespace NUMINAMATH_CALUDE_gcd_50404_40303_l3419_341994

theorem gcd_50404_40303 : Nat.gcd 50404 40303 = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_50404_40303_l3419_341994


namespace NUMINAMATH_CALUDE_negation_of_proposition_l3419_341901

theorem negation_of_proposition (p : ℝ → Prop) :
  (¬ ∀ x : ℝ, 2 * x^2 - 1 > 0) ↔ (∃ x : ℝ, 2 * x^2 - 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l3419_341901


namespace NUMINAMATH_CALUDE_original_average_score_l3419_341906

/-- Proves that the original average score of a class is 37, given the conditions. -/
theorem original_average_score (num_students : ℕ) (grace_marks : ℕ) (new_average : ℕ) :
  num_students = 35 →
  grace_marks = 3 →
  new_average = 40 →
  (num_students * new_average - num_students * grace_marks) / num_students = 37 :=
by sorry

end NUMINAMATH_CALUDE_original_average_score_l3419_341906


namespace NUMINAMATH_CALUDE_polygon_contains_circle_l3419_341908

/-- A convex polygon with width 1 -/
structure ConvexPolygon where
  width : ℝ
  width_eq_one : width = 1
  is_convex : Bool  -- This is a simplification, as convexity is more complex to define

/-- A circle with a given radius -/
structure Circle where
  radius : ℝ

/-- Predicate to check if a circle is contained within a polygon -/
def containsCircle (p : ConvexPolygon) (c : Circle) : Prop :=
  sorry  -- The actual implementation would depend on how we represent polygons and circles

theorem polygon_contains_circle (M : ConvexPolygon) : 
  ∃ (c : Circle), c.radius ≥ 1/3 ∧ containsCircle M c := by
  sorry

#check polygon_contains_circle

end NUMINAMATH_CALUDE_polygon_contains_circle_l3419_341908


namespace NUMINAMATH_CALUDE_race_cars_alignment_l3419_341974

theorem race_cars_alignment (a b c : ℕ) (ha : a = 28) (hb : b = 24) (hc : c = 32) :
  Nat.lcm (Nat.lcm a b) c = 672 := by
  sorry

end NUMINAMATH_CALUDE_race_cars_alignment_l3419_341974


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3419_341907

/-- Definition of a hyperbola with given foci and distance property -/
structure Hyperbola where
  f1 : ℝ × ℝ
  f2 : ℝ × ℝ
  dist_diff : ℝ

/-- The standard form of a hyperbola equation -/
def standard_equation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

/-- Theorem: The standard equation of the given hyperbola -/
theorem hyperbola_equation (h : Hyperbola) 
    (h_f1 : h.f1 = (-5, 0))
    (h_f2 : h.f2 = (5, 0))
    (h_dist : h.dist_diff = 8) :
    ∀ x y : ℝ, (x, y) ∈ {p : ℝ × ℝ | ‖p - h.f1‖ - ‖p - h.f2‖ = h.dist_diff} →
    standard_equation 4 3 x y := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3419_341907


namespace NUMINAMATH_CALUDE_smallest_meeting_time_l3419_341972

/-- The number of horses -/
def num_horses : ℕ := 8

/-- The time taken by horse k to complete one lap -/
def lap_time (k : ℕ) : ℕ := k^2

/-- Predicate to check if a time t is when at least 4 horses are at the starting point -/
def at_least_four_horses_meet (t : ℕ) : Prop :=
  ∃ (h1 h2 h3 h4 : ℕ), 
    h1 ≤ num_horses ∧ h2 ≤ num_horses ∧ h3 ≤ num_horses ∧ h4 ≤ num_horses ∧
    h1 ≠ h2 ∧ h1 ≠ h3 ∧ h1 ≠ h4 ∧ h2 ≠ h3 ∧ h2 ≠ h4 ∧ h3 ≠ h4 ∧
    t % (lap_time h1) = 0 ∧ t % (lap_time h2) = 0 ∧ t % (lap_time h3) = 0 ∧ t % (lap_time h4) = 0

/-- The smallest positive time when at least 4 horses meet at the starting point -/
def S : ℕ := 144

theorem smallest_meeting_time : 
  (S > 0) ∧ 
  at_least_four_horses_meet S ∧ 
  ∀ t, 0 < t ∧ t < S → ¬(at_least_four_horses_meet t) :=
by sorry

end NUMINAMATH_CALUDE_smallest_meeting_time_l3419_341972


namespace NUMINAMATH_CALUDE_wages_theorem_l3419_341920

/-- Given a sum of money that can pay B's wages for 12 days and C's wages for 24 days,
    prove that it can pay both B and C's wages together for 8 days -/
theorem wages_theorem (S : ℝ) (W_B W_C : ℝ) (h1 : S = 12 * W_B) (h2 : S = 24 * W_C) :
  S = 8 * (W_B + W_C) := by
  sorry

end NUMINAMATH_CALUDE_wages_theorem_l3419_341920


namespace NUMINAMATH_CALUDE_fifth_match_goals_l3419_341962

/-- Represents the goal-scoring statistics of a football player over 5 matches -/
structure FootballStats where
  total_goals : ℕ
  avg_increase : ℚ

/-- Theorem stating that under given conditions, the player scored 3 goals in the fifth match -/
theorem fifth_match_goals (stats : FootballStats) 
  (h1 : stats.total_goals = 11)
  (h2 : stats.avg_increase = 1/5) : 
  (stats.total_goals : ℚ) - 4 * ((stats.total_goals : ℚ) / 5 - stats.avg_increase) = 3 := by
  sorry

#check fifth_match_goals

end NUMINAMATH_CALUDE_fifth_match_goals_l3419_341962


namespace NUMINAMATH_CALUDE_intersection_equality_l3419_341938

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {1, 3, Real.sqrt m}
def B (m : ℝ) : Set ℝ := {1, m}

-- State the theorem
theorem intersection_equality (m : ℝ) : 
  A m ∩ B m = B m → m = 3 ∨ m = 0 := by
  sorry

end NUMINAMATH_CALUDE_intersection_equality_l3419_341938


namespace NUMINAMATH_CALUDE_final_pen_count_l3419_341979

def pen_collection (initial_pens : ℕ) (mike_gives : ℕ) (sharon_takes : ℕ) : ℕ :=
  let after_mike := initial_pens + mike_gives
  let after_cindy := 2 * after_mike
  after_cindy - sharon_takes

theorem final_pen_count :
  pen_collection 7 22 19 = 39 := by
  sorry

end NUMINAMATH_CALUDE_final_pen_count_l3419_341979


namespace NUMINAMATH_CALUDE_position_2007_l3419_341924

-- Define the possible square positions
inductive SquarePosition
  | ABCD
  | DCBA
  | ADCB
  | BADC
  | CBAD

-- Define the transformation function
def transform (pos : SquarePosition) : SquarePosition :=
  match pos with
  | SquarePosition.ABCD => SquarePosition.DCBA
  | SquarePosition.DCBA => SquarePosition.ADCB
  | SquarePosition.ADCB => SquarePosition.BADC
  | SquarePosition.BADC => SquarePosition.CBAD
  | SquarePosition.CBAD => SquarePosition.ABCD

-- Define the function to get the position after n transformations
def positionAfterN (n : Nat) : SquarePosition :=
  match n % 4 with
  | 0 => SquarePosition.ABCD
  | 1 => SquarePosition.DCBA
  | 2 => SquarePosition.ADCB
  | _ => SquarePosition.BADC

-- Theorem statement
theorem position_2007 : positionAfterN 2007 = SquarePosition.ADCB := by
  sorry


end NUMINAMATH_CALUDE_position_2007_l3419_341924


namespace NUMINAMATH_CALUDE_range_of_m_l3419_341946

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : 2 * x + y = 1)
  (h_ineq : ∀ x y : ℝ, x > 0 → y > 0 → 2 * x + y = 1 → 4 * x^2 + y^2 + Real.sqrt (x * y) - m < 0) :
  m > 17/16 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l3419_341946


namespace NUMINAMATH_CALUDE_product_is_rational_l3419_341937

def primes : List Nat := [3, 5, 7, 11, 13, 17]

def product : ℚ :=
  primes.foldl (fun acc p => acc * (1 - 1 / (p * p : ℚ))) 1

theorem product_is_rational : ∃ (a b : ℕ), product = a / b :=
  sorry

end NUMINAMATH_CALUDE_product_is_rational_l3419_341937


namespace NUMINAMATH_CALUDE_modulus_of_complex_l3419_341903

theorem modulus_of_complex (z : ℂ) (h : z = 4 + 3*I) : Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_l3419_341903


namespace NUMINAMATH_CALUDE_ellipse_equation_l3419_341923

theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  let C : Set (ℝ × ℝ) := {(x, y) | x^2/a^2 + y^2/b^2 = 1}
  let e := (Real.sqrt (a^2 - b^2)) / a
  (0, 4) ∈ C ∧ e = 3/5 → C = {(x, y) | x^2/25 + y^2/16 = 1} := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_l3419_341923


namespace NUMINAMATH_CALUDE_two_digit_numbers_divisibility_l3419_341981

/-- The number of digits in the numbers we're considering -/
def n : ℕ := 2019

/-- The set of possible digits -/
def digits : Set ℕ := {d | 0 ≤ d ∧ d ≤ 9}

/-- A function that counts the number of n-digit numbers made of 2 different digits -/
noncomputable def count_two_digit_numbers (n : ℕ) : ℕ :=
  sorry

/-- The highest power of 3 that divides a natural number -/
noncomputable def highest_power_of_three (m : ℕ) : ℕ :=
  sorry

theorem two_digit_numbers_divisibility :
  highest_power_of_three (count_two_digit_numbers n) = 5 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_numbers_divisibility_l3419_341981


namespace NUMINAMATH_CALUDE_candy_mixture_cost_l3419_341977

/-- The cost per pound of the first candy -/
def first_candy_cost : ℝ := 8

/-- The weight of the first candy in pounds -/
def first_candy_weight : ℝ := 30

/-- The cost per pound of the second candy -/
def second_candy_cost : ℝ := 5

/-- The weight of the second candy in pounds -/
def second_candy_weight : ℝ := 60

/-- The cost per pound of the mixture -/
def mixture_cost : ℝ := 6

/-- The total weight of the mixture in pounds -/
def total_weight : ℝ := first_candy_weight + second_candy_weight

theorem candy_mixture_cost :
  first_candy_cost * first_candy_weight + second_candy_cost * second_candy_weight =
  mixture_cost * total_weight :=
by sorry

end NUMINAMATH_CALUDE_candy_mixture_cost_l3419_341977


namespace NUMINAMATH_CALUDE_cubic_root_sum_l3419_341953

theorem cubic_root_sum (p q r : ℝ) : 
  p^3 - 6*p^2 + 7*p - 1 = 0 ∧ 
  q^3 - 6*q^2 + 7*q - 1 = 0 ∧ 
  r^3 - 6*r^2 + 7*r - 1 = 0 →
  p / (q*r + 1) + q / (p*r + 1) + r / (p*q + 1) = 61/15 :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l3419_341953


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l3419_341913

theorem pure_imaginary_condition (a : ℝ) : 
  (a^2 - 2*a = 0 ∧ a^2 - a - 2 ≠ 0) → a = 0 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l3419_341913


namespace NUMINAMATH_CALUDE_sequence_problem_l3419_341957

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

/-- The main theorem -/
theorem sequence_problem (a b : ℕ → ℝ) 
    (h_arith : arithmetic_sequence a)
    (h_geom : geometric_sequence b)
    (h_eq : 2 * a 3 - (a 8)^2 + 2 * a 13 = 0)
    (h_b8 : b 8 = a 8) :
  b 4 * b 12 = 16 := by
  sorry

end NUMINAMATH_CALUDE_sequence_problem_l3419_341957


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l3419_341900

theorem trigonometric_simplification (x : ℝ) :
  (2 + 3 * Real.sin x - 4 * Real.cos x) / (2 + 3 * Real.sin x + 4 * Real.cos x) = Real.tan (x / 2) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l3419_341900


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_36_l3419_341926

/-- An arithmetic sequence with sum Sₙ of the first n terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

/-- The main theorem -/
theorem arithmetic_sequence_sum_36 (seq : ArithmeticSequence) 
  (h1 : 2 * seq.S 3 = 3 * seq.S 2 + 3)
  (h2 : seq.S 4 = seq.a 10) : 
  seq.S 36 = 666 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_36_l3419_341926


namespace NUMINAMATH_CALUDE_polynomial_sum_simplification_l3419_341970

theorem polynomial_sum_simplification :
  let p₁ : Polynomial ℚ := 2 * X^5 - 3 * X^3 + 5 * X^2 - 4 * X + 6
  let p₂ : Polynomial ℚ := -X^5 + 4 * X^4 - 2 * X^3 - X^2 + 3 * X - 8
  p₁ + p₂ = X^5 + 4 * X^4 - 5 * X^3 + 4 * X^2 - X - 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_simplification_l3419_341970


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3419_341982

/-- A right triangle with perimeter 60 and area 48 has a hypotenuse of length 28.4 -/
theorem right_triangle_hypotenuse : ∃ (a b c : ℝ),
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- sides are positive
  a^2 + b^2 = c^2 ∧  -- right triangle (Pythagorean theorem)
  a + b + c = 60 ∧  -- perimeter is 60
  (1/2) * a * b = 48 ∧  -- area is 48
  c = 28.4 :=  -- hypotenuse is 28.4
by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3419_341982


namespace NUMINAMATH_CALUDE_abs_gt_not_sufficient_nor_necessary_l3419_341984

theorem abs_gt_not_sufficient_nor_necessary (a b : ℝ) : 
  (∃ x y : ℝ, abs x > abs y ∧ x ≤ y) ∧ 
  (∃ u v : ℝ, u > v ∧ abs u ≤ abs v) := by
sorry

end NUMINAMATH_CALUDE_abs_gt_not_sufficient_nor_necessary_l3419_341984


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l3419_341969

theorem cyclic_sum_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x * y) / Real.sqrt ((x^2 + x*z + z^2) * (y^2 + y*z + z^2)) +
  (y * z) / Real.sqrt ((y^2 + y*x + x^2) * (z^2 + z*x + x^2)) +
  (z * x) / Real.sqrt ((z^2 + z*y + y^2) * (x^2 + x*y + y^2)) ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l3419_341969


namespace NUMINAMATH_CALUDE_side_to_perimeter_ratio_l3419_341947

/-- Represents a square garden -/
structure SquareGarden where
  side_length : ℝ

/-- Calculate the perimeter of a square garden -/
def perimeter (g : SquareGarden) : ℝ := 4 * g.side_length

/-- Theorem stating the ratio of side length to perimeter for a 15-foot square garden -/
theorem side_to_perimeter_ratio (g : SquareGarden) (h : g.side_length = 15) :
  g.side_length / perimeter g = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_side_to_perimeter_ratio_l3419_341947


namespace NUMINAMATH_CALUDE_sum_of_odd_periodic_function_l3419_341964

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_periodic_3 (f : ℝ → ℝ) : Prop := ∀ x, f (x + 3) = f x

theorem sum_of_odd_periodic_function 
  (f : ℝ → ℝ) 
  (h_odd : is_odd_function f) 
  (h_periodic : is_periodic_3 f) 
  (h_value : f (-1) = 1) : 
  f 1 + f 2 + f 3 + f 4 + f 5 + f 6 + f 7 + f 8 + f 9 + f 10 = -1 :=
sorry

end NUMINAMATH_CALUDE_sum_of_odd_periodic_function_l3419_341964


namespace NUMINAMATH_CALUDE_increase_by_percentage_increase_800_by_110_percent_l3419_341985

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) : 
  initial * (1 + percentage / 100) = initial + initial * (percentage / 100) := by sorry

theorem increase_800_by_110_percent : 
  800 * (1 + 110 / 100) = 1680 := by sorry

end NUMINAMATH_CALUDE_increase_by_percentage_increase_800_by_110_percent_l3419_341985


namespace NUMINAMATH_CALUDE_hyperbola_perpendicular_line_passes_fixed_point_l3419_341966

/-- The hyperbola equation -/
def is_on_hyperbola (x y : ℝ) : Prop := x^2 - y^2/2 = 1

/-- Point A -/
def point_A : ℝ × ℝ := (-1, 0)

/-- Check if a line passes through a point -/
def line_passes_through (m b x y : ℝ) : Prop := x = m * y + b

/-- Perpendicularity condition -/
def perpendicular (x1 y1 x2 y2 : ℝ) : Prop := 
  (y1 / (x1 + 1)) * (y2 / (x2 + 1)) = -1

theorem hyperbola_perpendicular_line_passes_fixed_point 
  (x1 y1 x2 y2 m b : ℝ) : 
  is_on_hyperbola x1 y1 → 
  is_on_hyperbola x2 y2 → 
  perpendicular x1 y1 x2 y2 → 
  line_passes_through m b x1 y1 → 
  line_passes_through m b x2 y2 → 
  line_passes_through m b 3 0 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_perpendicular_line_passes_fixed_point_l3419_341966


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_roots_l3419_341916

theorem sum_of_reciprocal_roots (a b : ℝ) : 
  a^2 - 6*a + 4 = 0 → b^2 - 6*b + 4 = 0 → a ≠ b → (1/a + 1/b) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_roots_l3419_341916


namespace NUMINAMATH_CALUDE_base_k_conversion_uniqueness_l3419_341997

theorem base_k_conversion_uniqueness :
  ∃! (k : ℕ), k ≥ 4 ∧ 1 * k^2 + 3 * k + 2 = 30 := by sorry

end NUMINAMATH_CALUDE_base_k_conversion_uniqueness_l3419_341997


namespace NUMINAMATH_CALUDE_calculate_principal_amount_l3419_341983

/-- Given simple interest, time period, and interest rate, calculate the principal amount -/
theorem calculate_principal_amount (simple_interest rate : ℚ) (time : ℕ) : 
  simple_interest = 200 → 
  time = 4 → 
  rate = 3125 / 100000 → 
  simple_interest = (1600 : ℚ) * rate * (time : ℚ) := by
  sorry

#check calculate_principal_amount

end NUMINAMATH_CALUDE_calculate_principal_amount_l3419_341983


namespace NUMINAMATH_CALUDE_complementary_of_same_angle_are_equal_l3419_341928

/-- Two angles are complementary if their sum is equal to a right angle (90°) -/
def Complementary (α β : Real) : Prop := α + β = Real.pi / 2

/-- An angle is complementary to itself if it is half of a right angle -/
def SelfComplementary (α : Real) : Prop := α = Real.pi / 4

theorem complementary_of_same_angle_are_equal (α : Real) (h : SelfComplementary α) :
  ∃ β, Complementary α β ∧ α = β := by
  sorry

end NUMINAMATH_CALUDE_complementary_of_same_angle_are_equal_l3419_341928


namespace NUMINAMATH_CALUDE_min_ops_to_500_l3419_341989

def calculator_ops (n : ℕ) : ℕ → ℕ
| 0     => n
| (k+1) => calculator_ops (min (2*n) (n+1)) k

theorem min_ops_to_500 : ∃ k, calculator_ops 1 k = 500 ∧ ∀ j, j < k → calculator_ops 1 j ≠ 500 :=
  sorry

end NUMINAMATH_CALUDE_min_ops_to_500_l3419_341989
