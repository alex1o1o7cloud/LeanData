import Mathlib

namespace NUMINAMATH_CALUDE_polynomial_coefficient_theorem_l3863_386378

theorem polynomial_coefficient_theorem (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, a₀ + a₁ * (2 * x - 1) + a₂ * (2 * x - 1)^2 + a₃ * (2 * x - 1)^3 + 
             a₄ * (2 * x - 1)^4 + a₅ * (2 * x - 1)^5 = x^5) →
  a₂ = 5/16 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_theorem_l3863_386378


namespace NUMINAMATH_CALUDE_meadow_business_revenue_l3863_386327

/-- Represents Meadow's diaper business --/
structure DiaperBusiness where
  boxes_per_week : ℕ
  packs_per_box : ℕ
  diapers_per_pack : ℕ
  price_per_diaper : ℕ

/-- Calculates the total money made from selling all diapers --/
def total_money_made (business : DiaperBusiness) : ℕ :=
  business.boxes_per_week * business.packs_per_box * business.diapers_per_pack * business.price_per_diaper

/-- Theorem stating that Meadow's business makes $960000 from selling all diapers --/
theorem meadow_business_revenue :
  let meadow_business : DiaperBusiness := {
    boxes_per_week := 30,
    packs_per_box := 40,
    diapers_per_pack := 160,
    price_per_diaper := 5
  }
  total_money_made meadow_business = 960000 := by
  sorry

end NUMINAMATH_CALUDE_meadow_business_revenue_l3863_386327


namespace NUMINAMATH_CALUDE_real_part_of_i_times_3_minus_i_l3863_386338

theorem real_part_of_i_times_3_minus_i : ∃ (z : ℂ), z = Complex.I * (3 - Complex.I) ∧ z.re = 1 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_i_times_3_minus_i_l3863_386338


namespace NUMINAMATH_CALUDE_largest_consecutive_sum_45_l3863_386356

/-- The sum of n consecutive positive integers starting from k -/
def consecutiveSum (n k : ℕ) : ℕ := n * (2 * k + n - 1) / 2

/-- The largest number of positive consecutive integers that sum to 45 -/
theorem largest_consecutive_sum_45 :
  (∃ (k : ℕ), k > 0 ∧ consecutiveSum 9 k = 45) ∧
  (∀ (n k : ℕ), n > 9 → k > 0 → consecutiveSum n k ≠ 45) :=
sorry

end NUMINAMATH_CALUDE_largest_consecutive_sum_45_l3863_386356


namespace NUMINAMATH_CALUDE_simple_interest_theorem_l3863_386334

/-- Proves that when the simple interest on a sum of money is 2/5 of the principal amount,
    and the rate is 4% per annum, the time period is 10 years. -/
theorem simple_interest_theorem (P : ℝ) (R : ℝ) (T : ℝ) :
  R = 4 →
  (2 / 5) * P = (P * R * T) / 100 →
  T = 10 := by
sorry


end NUMINAMATH_CALUDE_simple_interest_theorem_l3863_386334


namespace NUMINAMATH_CALUDE_derivative_of_x4_minus_7_l3863_386393

theorem derivative_of_x4_minus_7 (x : ℝ) :
  deriv (fun x => x^4 - 7) x = 4 * x^3 - 7 := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_x4_minus_7_l3863_386393


namespace NUMINAMATH_CALUDE_prob_at_least_one_of_three_l3863_386399

/-- The probability that at least one of three events occurs, given their individual probabilities -/
theorem prob_at_least_one_of_three (pA pB pC : ℝ) 
  (hA : 0 ≤ pA ∧ pA ≤ 1) 
  (hB : 0 ≤ pB ∧ pB ≤ 1) 
  (hC : 0 ≤ pC ∧ pC ≤ 1) 
  (h_pA : pA = 0.8) 
  (h_pB : pB = 0.6) 
  (h_pC : pC = 0.5) : 
  1 - (1 - pA) * (1 - pB) * (1 - pC) = 0.96 := by
sorry

end NUMINAMATH_CALUDE_prob_at_least_one_of_three_l3863_386399


namespace NUMINAMATH_CALUDE_inequality_proof_l3863_386324

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b) * (a^4 + b^4) ≥ (a^2 + b^2) * (a^3 + b^3) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3863_386324


namespace NUMINAMATH_CALUDE_correct_stratified_sampling_l3863_386358

/-- Represents a stratum in the population -/
structure Stratum where
  size : ℕ
  sample_size : ℕ

/-- Represents the population and sample -/
structure Population where
  total_size : ℕ
  sample_size : ℕ
  strata : List Stratum

/-- Checks if the sampling is stratified -/
def is_stratified_sampling (pop : Population) : Prop :=
  pop.strata.all (fun stratum => 
    stratum.sample_size * pop.total_size = pop.sample_size * stratum.size)

/-- The given population data -/
def school_population : Population :=
  { total_size := 1000
  , sample_size := 40
  , strata := 
    [ { size := 400, sample_size := 16 }  -- Blood type O
    , { size := 250, sample_size := 10 }  -- Blood type A
    , { size := 250, sample_size := 10 }  -- Blood type B
    , { size := 100, sample_size := 4 }   -- Blood type AB
    ]
  }

/-- The main theorem to prove -/
theorem correct_stratified_sampling :
  is_stratified_sampling school_population ∧
  school_population.strata.map (fun s => s.sample_size) = [16, 10, 10, 4] :=
sorry

end NUMINAMATH_CALUDE_correct_stratified_sampling_l3863_386358


namespace NUMINAMATH_CALUDE_milk_per_serving_in_cups_l3863_386308

/-- Proof that the amount of milk required per serving is 0.5 cups -/
theorem milk_per_serving_in_cups : 
  let ml_per_cup : ℝ := 250
  let total_people : ℕ := 8
  let servings_per_person : ℕ := 2
  let milk_cartons : ℕ := 2
  let ml_per_carton : ℝ := 1000
  
  let total_milk : ℝ := milk_cartons * ml_per_carton
  let total_servings : ℕ := total_people * servings_per_person
  let ml_per_serving : ℝ := total_milk / total_servings
  let cups_per_serving : ℝ := ml_per_serving / ml_per_cup

  cups_per_serving = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_milk_per_serving_in_cups_l3863_386308


namespace NUMINAMATH_CALUDE_sofia_shopping_cost_l3863_386319

def shirt_cost : ℕ := 7
def shoes_cost : ℕ := shirt_cost + 3
def total_shirts_shoes : ℕ := 2 * shirt_cost + shoes_cost
def bag_cost : ℕ := total_shirts_shoes / 2
def total_cost : ℕ := 2 * shirt_cost + shoes_cost + bag_cost

theorem sofia_shopping_cost : total_cost = 36 := by
  sorry

end NUMINAMATH_CALUDE_sofia_shopping_cost_l3863_386319


namespace NUMINAMATH_CALUDE_student_A_wrong_l3863_386392

-- Define the circle
def circle_center : ℝ × ℝ := (2, -3)
def circle_radius : ℝ := 5

-- Define the points
def point_D : ℝ × ℝ := (5, 1)
def point_A : ℝ × ℝ := (-2, -1)

-- Function to check if a point is on the circle
def is_on_circle (p : ℝ × ℝ) : Prop :=
  (p.1 - circle_center.1)^2 + (p.2 - circle_center.2)^2 = circle_radius^2

-- Theorem statement
theorem student_A_wrong :
  is_on_circle point_D ∧ ¬is_on_circle point_A :=
sorry

end NUMINAMATH_CALUDE_student_A_wrong_l3863_386392


namespace NUMINAMATH_CALUDE_no_linear_term_implies_a_value_l3863_386347

theorem no_linear_term_implies_a_value (a : ℝ) : 
  (∀ x : ℝ, ∃ b c d : ℝ, (x^2 + a*x - 2) * (x - 1) = x^3 + b*x^2 + d) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_no_linear_term_implies_a_value_l3863_386347


namespace NUMINAMATH_CALUDE_semicircle_radius_l3863_386361

/-- The radius of a semi-circle given its perimeter -/
theorem semicircle_radius (perimeter : ℝ) (h : perimeter = 108) :
  ∃ r : ℝ, r = perimeter / (Real.pi + 2) := by
  sorry

end NUMINAMATH_CALUDE_semicircle_radius_l3863_386361


namespace NUMINAMATH_CALUDE_ellipse_intersection_equidistant_point_range_l3863_386317

/-- Ellipse G with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  e : ℝ
  h : a > 0
  i : b > 0
  j : a > b
  k : e = Real.sqrt 3 / 3
  l : a = Real.sqrt 3

/-- Line intersecting the ellipse -/
structure IntersectingLine where
  k : ℝ
  m : ℝ := 1

/-- Point on x-axis equidistant from intersection points -/
structure EquidistantPoint where
  x : ℝ

/-- Main theorem -/
theorem ellipse_intersection_equidistant_point_range
  (G : Ellipse)
  (l : IntersectingLine)
  (M : EquidistantPoint) :
  (∃ A B : ℝ × ℝ,
    (A.1^2 / G.a^2 + A.2^2 / G.b^2 = 1) ∧
    (B.1^2 / G.a^2 + B.2^2 / G.b^2 = 1) ∧
    (A.2 = l.k * A.1 + l.m) ∧
    (B.2 = l.k * B.1 + l.m) ∧
    ((A.1 - M.x)^2 + A.2^2 = (B.1 - M.x)^2 + B.2^2) ∧
    (M.x ≠ A.1) ∧ (M.x ≠ B.1)) →
  -Real.sqrt 6 / 12 ≤ M.x ∧ M.x ≤ Real.sqrt 6 / 12 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_intersection_equidistant_point_range_l3863_386317


namespace NUMINAMATH_CALUDE_fish_tank_water_l3863_386341

theorem fish_tank_water (current : ℝ) : 
  (current + 7 = 14.75) → current = 7.75 := by
  sorry

end NUMINAMATH_CALUDE_fish_tank_water_l3863_386341


namespace NUMINAMATH_CALUDE_max_students_distribution_l3863_386355

theorem max_students_distribution (pens pencils : ℕ) 
  (h_pens : pens = 4860) 
  (h_pencils : pencils = 3645) : 
  (Nat.gcd (2 * pens) (3 * pencils)) / 6 = 202 := by
  sorry

end NUMINAMATH_CALUDE_max_students_distribution_l3863_386355


namespace NUMINAMATH_CALUDE_cosine_inequality_l3863_386348

theorem cosine_inequality (x y : Real) :
  x ∈ Set.Icc 0 (Real.pi / 2) →
  y ∈ Set.Icc 0 (Real.pi / 2) →
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), Real.cos (x + y) ≥ Real.cos x * Real.cos y) ↔
  y = 0 := by
sorry

end NUMINAMATH_CALUDE_cosine_inequality_l3863_386348


namespace NUMINAMATH_CALUDE_corn_preference_result_l3863_386309

/-- The percentage of children preferring corn in Carolyn's daycare -/
def corn_preference_percentage (total_children : ℕ) (corn_preference : ℕ) : ℚ :=
  (corn_preference : ℚ) / (total_children : ℚ) * 100

/-- Theorem stating that the percentage of children preferring corn is 17.5% -/
theorem corn_preference_result : 
  corn_preference_percentage 40 7 = 17.5 := by
  sorry

end NUMINAMATH_CALUDE_corn_preference_result_l3863_386309


namespace NUMINAMATH_CALUDE_equation_solution_l3863_386357

theorem equation_solution (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) :
  (x^3 - 3*x^2) / (x^2 - 4) + 2*x = -16 ↔ x = (23 + Real.sqrt 145) / 6 ∨ x = (23 - Real.sqrt 145) / 6 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3863_386357


namespace NUMINAMATH_CALUDE_area_of_five_presentable_set_l3863_386397

/-- A complex number is five-presentable if it can be expressed as w - 1/w for some w with |w| = 5 -/
def FivePresentable (z : ℂ) : Prop :=
  ∃ w : ℂ, Complex.abs w = 5 ∧ z = w - 1 / w

/-- The set of all five-presentable complex numbers -/
def T : Set ℂ :=
  {z : ℂ | FivePresentable z}

/-- The area of the set T -/
noncomputable def area_T : ℝ := sorry

theorem area_of_five_presentable_set :
  area_T = 624 * Real.pi / 25 := by sorry

end NUMINAMATH_CALUDE_area_of_five_presentable_set_l3863_386397


namespace NUMINAMATH_CALUDE_bagel_cut_theorem_l3863_386316

/-- Number of pieces resulting from cutting a torus-shaped object -/
def torusPieces (cuts : ℕ) : ℕ := cuts + 1

/-- Theorem: Cutting a torus-shaped object (bagel) with 10 cuts results in 11 pieces -/
theorem bagel_cut_theorem :
  torusPieces 10 = 11 := by
  sorry

end NUMINAMATH_CALUDE_bagel_cut_theorem_l3863_386316


namespace NUMINAMATH_CALUDE_rectangle_perimeter_width_ratio_l3863_386368

/-- Given a rectangle with area 150 square centimeters and length 15 centimeters,
    prove that the ratio of its perimeter to its width is 5:1 -/
theorem rectangle_perimeter_width_ratio 
  (area : ℝ) (length : ℝ) (width : ℝ) (perimeter : ℝ) :
  area = 150 →
  length = 15 →
  area = length * width →
  perimeter = 2 * (length + width) →
  perimeter / width = 5 := by
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_width_ratio_l3863_386368


namespace NUMINAMATH_CALUDE_pond_length_l3863_386366

/-- Given a rectangular field with length 24 meters and width 12 meters, 
    containing a square pond whose area is 1/8 of the field's area,
    prove that the length of the pond is 6 meters. -/
theorem pond_length (field_length : ℝ) (field_width : ℝ) (pond_length : ℝ) : 
  field_length = 24 →
  field_width = 12 →
  field_length = 2 * field_width →
  pond_length^2 = (field_length * field_width) / 8 →
  pond_length = 6 :=
by sorry

end NUMINAMATH_CALUDE_pond_length_l3863_386366


namespace NUMINAMATH_CALUDE_total_capacity_l3863_386313

/-- The capacity of a circus tent with five seating sections -/
def circus_tent_capacity (regular_section_capacity : ℕ) (special_section_capacity : ℕ) : ℕ :=
  4 * regular_section_capacity + special_section_capacity

/-- Theorem: The circus tent can accommodate 1298 people -/
theorem total_capacity : circus_tent_capacity 246 314 = 1298 := by
  sorry

end NUMINAMATH_CALUDE_total_capacity_l3863_386313


namespace NUMINAMATH_CALUDE_janelles_blue_marbles_gift_l3863_386380

/-- Calculates the number of blue marbles Janelle gave to her friend --/
def blue_marbles_given (initial_green : ℕ) (blue_bags : ℕ) (marbles_per_bag : ℕ) 
  (green_given : ℕ) (total_remaining : ℕ) : ℕ :=
  let total_blue := blue_bags * marbles_per_bag
  let total_before_gift := initial_green + total_blue
  let total_given := total_before_gift - total_remaining
  total_given - green_given

/-- Proves that Janelle gave 8 blue marbles to her friend --/
theorem janelles_blue_marbles_gift : 
  blue_marbles_given 26 6 10 6 72 = 8 := by
  sorry

end NUMINAMATH_CALUDE_janelles_blue_marbles_gift_l3863_386380


namespace NUMINAMATH_CALUDE_jungkook_item_sum_l3863_386350

theorem jungkook_item_sum : ∀ (a b : ℕ),
  a = 585 →
  a = b + 249 →
  a + b = 921 :=
by
  sorry

end NUMINAMATH_CALUDE_jungkook_item_sum_l3863_386350


namespace NUMINAMATH_CALUDE_sqrt_equality_l3863_386320

theorem sqrt_equality (m n : ℝ) (h1 : m > 0) (h2 : 0 ≤ n) (h3 : n ≤ 3*m) :
  Real.sqrt (6*m + 2*Real.sqrt (9*m^2 - n^2)) - Real.sqrt (6*m - 2*Real.sqrt (9*m^2 - n^2)) = 2 * Real.sqrt (3*m - n) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equality_l3863_386320


namespace NUMINAMATH_CALUDE_compound_proposition_truth_l3863_386349

theorem compound_proposition_truth (p q : Prop) 
  (h : ¬(¬p ∨ ¬q)) : (p ∧ q) ∧ (p ∨ q) := by
  sorry

end NUMINAMATH_CALUDE_compound_proposition_truth_l3863_386349


namespace NUMINAMATH_CALUDE_factorial_multiple_l3863_386398

theorem factorial_multiple (m n : ℕ) : 
  ∃ k : ℕ, (2 * m).factorial * (2 * n).factorial = k * m.factorial * n.factorial * (m + n).factorial := by
sorry

end NUMINAMATH_CALUDE_factorial_multiple_l3863_386398


namespace NUMINAMATH_CALUDE_combination_three_choose_two_l3863_386322

theorem combination_three_choose_two : Finset.card (Finset.powerset {0, 1, 2} |>.filter (fun s => Finset.card s = 2)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_combination_three_choose_two_l3863_386322


namespace NUMINAMATH_CALUDE_B_power_101_l3863_386330

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![0, 1, 0],
    ![0, 0, 1],
    ![1, 0, 0]]

theorem B_power_101 :
  B ^ 101 = ![![0, 0, 1],
              ![1, 0, 0],
              ![0, 1, 0]] := by
  sorry

end NUMINAMATH_CALUDE_B_power_101_l3863_386330


namespace NUMINAMATH_CALUDE_gcd_from_lcm_and_ratio_l3863_386383

theorem gcd_from_lcm_and_ratio (A B : ℕ+) : 
  Nat.lcm A B = 180 → 
  (A : ℚ) / B = 2 / 5 → 
  Nat.gcd A B = 18 := by
sorry

end NUMINAMATH_CALUDE_gcd_from_lcm_and_ratio_l3863_386383


namespace NUMINAMATH_CALUDE_choose_president_vice_president_l3863_386345

/-- The number of boys in the club -/
def num_boys : ℕ := 12

/-- The number of girls in the club -/
def num_girls : ℕ := 12

/-- The total number of members in the club -/
def total_members : ℕ := num_boys + num_girls

/-- The number of ways to choose a president and vice-president of opposite genders -/
def ways_to_choose : ℕ := num_boys * num_girls * 2

theorem choose_president_vice_president :
  ways_to_choose = 288 :=
by sorry

end NUMINAMATH_CALUDE_choose_president_vice_president_l3863_386345


namespace NUMINAMATH_CALUDE_polynomial_irreducibility_l3863_386339

theorem polynomial_irreducibility (n : ℕ) (h : n > 1) :
  Irreducible (Polynomial.X ^ n + 5 * Polynomial.X ^ (n - 1) + 3 : Polynomial ℤ) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_irreducibility_l3863_386339


namespace NUMINAMATH_CALUDE_floor_abs_negative_56_3_l3863_386394

theorem floor_abs_negative_56_3 : ⌊|(-56.3 : ℝ)|⌋ = 56 := by sorry

end NUMINAMATH_CALUDE_floor_abs_negative_56_3_l3863_386394


namespace NUMINAMATH_CALUDE_tiffany_bag_difference_l3863_386306

/-- Calculates the difference in bags between Tuesday and Monday after giving away some bags -/
def bagDifference (mondayBags tuesdayFound givenAway : ℕ) : ℕ :=
  (mondayBags + tuesdayFound - givenAway) - mondayBags

theorem tiffany_bag_difference :
  bagDifference 7 12 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_bag_difference_l3863_386306


namespace NUMINAMATH_CALUDE_smallest_fraction_greater_than_three_fifths_l3863_386312

/-- A fraction with two-digit numerator and denominator -/
structure TwoDigitFraction where
  numerator : ℕ
  denominator : ℕ
  num_two_digit : 10 ≤ numerator ∧ numerator ≤ 99
  den_two_digit : 10 ≤ denominator ∧ denominator ≤ 99

/-- The property of being greater than 3/5 -/
def greater_than_three_fifths (f : TwoDigitFraction) : Prop :=
  (f.numerator : ℚ) / f.denominator > 3 / 5

/-- The theorem stating that 59/98 is the smallest fraction greater than 3/5 with two-digit numerator and denominator -/
theorem smallest_fraction_greater_than_three_fifths :
  ∀ f : TwoDigitFraction, greater_than_three_fifths f →
    (59 : ℚ) / 98 ≤ (f.numerator : ℚ) / f.denominator :=
by sorry

end NUMINAMATH_CALUDE_smallest_fraction_greater_than_three_fifths_l3863_386312


namespace NUMINAMATH_CALUDE_inequality_solution_l3863_386315

theorem inequality_solution (α x : ℝ) : α * x^2 - 2 ≥ 2 * x - α * x ↔
  (α = 0 ∧ x ≤ -1) ∨
  (α > 0 ∧ (x ≥ 2 / α ∨ x ≤ -1)) ∨
  (-2 < α ∧ α < 0 ∧ 2 / α ≤ x ∧ x ≤ -1) ∨
  (α = -2 ∧ x = -1) ∨
  (α < -2 ∧ -1 ≤ x ∧ x ≤ 2 / α) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3863_386315


namespace NUMINAMATH_CALUDE_probability_gpa_at_least_3_5_l3863_386337

/-- Represents the possible grades a student can receive --/
inductive Grade
| A
| B
| C
| D

/-- Converts a grade to its point value --/
def gradeToPoints : Grade → ℕ
| Grade.A => 4
| Grade.B => 3
| Grade.C => 2
| Grade.D => 1

/-- Calculates the GPA given a list of grades --/
def calculateGPA (grades : List Grade) : ℚ :=
  (grades.map gradeToPoints).sum / 4

/-- Represents the probability of getting each grade in a subject --/
structure GradeProbability where
  probA : ℚ
  probB : ℚ
  probC : ℚ
  probD : ℚ

/-- The probability distribution for English grades --/
def englishProb : GradeProbability :=
  { probA := 1/6
  , probB := 1/4
  , probC := 7/12
  , probD := 0 }

/-- The probability distribution for History grades --/
def historyProb : GradeProbability :=
  { probA := 1/4
  , probB := 1/3
  , probC := 5/12
  , probD := 0 }

/-- Theorem stating the probability of getting a GPA of at least 3.5 --/
theorem probability_gpa_at_least_3_5 :
  let mathGrade := Grade.A
  let scienceGrade := Grade.A
  let probAtLeast3_5 := (englishProb.probA * historyProb.probA) +
                        (englishProb.probA * historyProb.probB) +
                        (englishProb.probB * historyProb.probA) +
                        (englishProb.probA * historyProb.probC) +
                        (englishProb.probC * historyProb.probA) +
                        (englishProb.probB * historyProb.probB)
  probAtLeast3_5 = 11/24 := by
  sorry

end NUMINAMATH_CALUDE_probability_gpa_at_least_3_5_l3863_386337


namespace NUMINAMATH_CALUDE_stream_speed_l3863_386321

theorem stream_speed (downstream_distance : ℝ) (upstream_distance : ℝ) (time : ℝ) 
  (h1 : downstream_distance = 78)
  (h2 : upstream_distance = 50)
  (h3 : time = 2) :
  ∃ (boat_speed stream_speed : ℝ),
    downstream_distance = (boat_speed + stream_speed) * time ∧
    upstream_distance = (boat_speed - stream_speed) * time ∧
    stream_speed = 7 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l3863_386321


namespace NUMINAMATH_CALUDE_hypotenuse_length_l3863_386381

/-- A right triangle with given perimeter and difference between median and altitude. -/
structure RightTriangle where
  /-- Side length BC -/
  a : ℝ
  /-- Side length AC -/
  b : ℝ
  /-- Hypotenuse length AB -/
  c : ℝ
  /-- Perimeter of the triangle -/
  perimeter_eq : a + b + c = 72
  /-- Pythagorean theorem -/
  pythagoras : a^2 + b^2 = c^2
  /-- Difference between median and altitude -/
  median_altitude_diff : c / 2 - (a * b) / c = 7

/-- The hypotenuse of a right triangle with the given properties is 32 cm. -/
theorem hypotenuse_length (t : RightTriangle) : t.c = 32 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_length_l3863_386381


namespace NUMINAMATH_CALUDE_complex_number_problem_l3863_386323

theorem complex_number_problem (z₁ z₂ : ℂ) : 
  ((z₁ - 2) * Complex.I = 1 + Complex.I) →
  (z₂.im = 2) →
  ((z₁ * z₂).im = 0) →
  (z₁ = 3 - Complex.I ∧ z₂ = 6 + 2 * Complex.I) := by
sorry

end NUMINAMATH_CALUDE_complex_number_problem_l3863_386323


namespace NUMINAMATH_CALUDE_share_of_B_l3863_386384

theorem share_of_B (total : ℚ) (a b c : ℚ) : 
  total = 595 → 
  a = (2/3) * b → 
  b = (1/4) * c → 
  a + b + c = total → 
  b = 105 := by
sorry

end NUMINAMATH_CALUDE_share_of_B_l3863_386384


namespace NUMINAMATH_CALUDE_deer_leap_distance_proof_l3863_386314

/-- The distance the tiger needs to catch the deer -/
def catch_distance : ℝ := 800

/-- The number of tiger leaps behind the deer initially -/
def initial_leaps_behind : ℕ := 50

/-- The number of leaps the tiger takes per minute -/
def tiger_leaps_per_minute : ℕ := 5

/-- The number of leaps the deer takes per minute -/
def deer_leaps_per_minute : ℕ := 4

/-- The distance the tiger covers per leap in meters -/
def tiger_leap_distance : ℝ := 8

/-- The distance the deer covers per leap in meters -/
def deer_leap_distance : ℝ := 5

theorem deer_leap_distance_proof :
  deer_leap_distance = 5 :=
sorry

end NUMINAMATH_CALUDE_deer_leap_distance_proof_l3863_386314


namespace NUMINAMATH_CALUDE_perfect_squares_existence_l3863_386374

theorem perfect_squares_existence (a : ℕ) (h1 : Odd a) (h2 : a > 17) 
  (h3 : ∃ x : ℕ, 3 * a - 2 = x^2) : 
  ∃ b c : ℕ, b ≠ c ∧ b > 0 ∧ c > 0 ∧ 
    (∃ w x y z : ℕ, a + b = w^2 ∧ a + c = x^2 ∧ b + c = y^2 ∧ a + b + c = z^2) :=
by sorry

end NUMINAMATH_CALUDE_perfect_squares_existence_l3863_386374


namespace NUMINAMATH_CALUDE_stone_volume_l3863_386353

/-- The volume of a stone submerged in a cuboid-shaped container -/
theorem stone_volume (width length initial_height final_height : ℝ) 
  (hw : width = 15) 
  (hl : length = 20) 
  (hi : initial_height = 10) 
  (hf : final_height = 15) : 
  (final_height - initial_height) * width * length = 1500 := by
  sorry

end NUMINAMATH_CALUDE_stone_volume_l3863_386353


namespace NUMINAMATH_CALUDE_mike_initial_cards_l3863_386351

theorem mike_initial_cards (sold : ℕ) (current : ℕ) (h1 : sold = 13) (h2 : current = 74) :
  current + sold = 87 := by
  sorry

end NUMINAMATH_CALUDE_mike_initial_cards_l3863_386351


namespace NUMINAMATH_CALUDE_tens_digit_of_6_pow_18_l3863_386340

/-- The tens digit of a natural number -/
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

/-- The tens digit of 6^18 is 1 -/
theorem tens_digit_of_6_pow_18 : tens_digit (6^18) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_6_pow_18_l3863_386340


namespace NUMINAMATH_CALUDE_unique_polygon_pair_existence_l3863_386325

theorem unique_polygon_pair_existence : 
  ∃! (n₁ n₂ : ℕ), 
    n₁ > 0 ∧ n₂ > 0 ∧
    ∃ x : ℝ, x > 0 ∧
      (180 - 360 / n₁ : ℝ) = x ∧
      (180 - 360 / n₂ : ℝ) = x / 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_polygon_pair_existence_l3863_386325


namespace NUMINAMATH_CALUDE_ariel_age_l3863_386333

/-- Ariel's present age in years -/
def present_age : ℕ := 5

/-- The number of years in the future -/
def years_future : ℕ := 15

/-- Theorem stating that Ariel's present age is 5, given the condition -/
theorem ariel_age : 
  (present_age + years_future = 4 * present_age) → present_age = 5 := by
  sorry

end NUMINAMATH_CALUDE_ariel_age_l3863_386333


namespace NUMINAMATH_CALUDE_sheep_count_l3863_386363

/-- The number of sheep in the meadow -/
def num_sheep : ℕ := 36

/-- The number of cows in the meadow -/
def num_cows : ℕ := 12

/-- The number of ears per cow -/
def ears_per_cow : ℕ := 2

/-- The number of legs per cow -/
def legs_per_cow : ℕ := 4

/-- Theorem stating that the number of sheep is 36 given the conditions -/
theorem sheep_count :
  num_sheep > num_cows * ears_per_cow ∧
  num_sheep < num_cows * legs_per_cow ∧
  num_sheep % 12 = 0 →
  num_sheep = 36 :=
by sorry

end NUMINAMATH_CALUDE_sheep_count_l3863_386363


namespace NUMINAMATH_CALUDE_base_conversion_2450_l3863_386367

/-- Converts a base-10 number to its base-8 representation -/
def toBase8 (n : ℕ) : ℕ := sorry

/-- Converts a base-8 number to its base-10 representation -/
def fromBase8 (n : ℕ) : ℕ := sorry

theorem base_conversion_2450 :
  toBase8 2450 = 4622 ∧ fromBase8 4622 = 2450 := by sorry

end NUMINAMATH_CALUDE_base_conversion_2450_l3863_386367


namespace NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l3863_386304

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def IsArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of the first five terms of the sequence is 20. -/
def SumFirstFiveTerms (a : ℕ → ℝ) : Prop :=
  a 1 + a 2 + a 3 + a 4 + a 5 = 20

theorem arithmetic_sequence_third_term
  (a : ℕ → ℝ)
  (h_arithmetic : IsArithmeticSequence a)
  (h_sum : SumFirstFiveTerms a) :
  a 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l3863_386304


namespace NUMINAMATH_CALUDE_kite_diagonal_length_l3863_386352

/-- A rectangle ABCD with a kite WXYZ inscribed -/
structure RectangleWithKite where
  /-- Length of side AB -/
  ab : ℝ
  /-- Length of side BC -/
  bc : ℝ
  /-- Distance from A to W on AB -/
  aw : ℝ
  /-- Distance from C to Y on CD -/
  cy : ℝ
  /-- AB = CD = 5 -/
  h_ab : ab = 5
  /-- BC = AD = 10 -/
  h_bc : bc = 10
  /-- WX = WZ = √13 -/
  h_wx : aw ^ 2 + cy ^ 2 = 13
  /-- XY = ZY -/
  h_xy_zy : (bc - aw) ^ 2 + cy ^ 2 = (ab - cy) ^ 2 + aw ^ 2

/-- The length of XY in the kite WXYZ is √65 -/
theorem kite_diagonal_length (r : RectangleWithKite) : 
  (r.bc - r.aw) ^ 2 + r.cy ^ 2 = 65 := by
  sorry


end NUMINAMATH_CALUDE_kite_diagonal_length_l3863_386352


namespace NUMINAMATH_CALUDE_circle_radius_is_two_l3863_386370

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 8*x + y^2 - 4*y + 16 = 0

-- Define the center of the circle
def circle_center : ℝ × ℝ := (4, 2)

-- Define the radius of the circle
def circle_radius : ℝ := 2

-- Theorem statement
theorem circle_radius_is_two :
  ∀ x y : ℝ, circle_equation x y →
  (x - circle_center.1)^2 + (y - circle_center.2)^2 = circle_radius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_is_two_l3863_386370


namespace NUMINAMATH_CALUDE_zero_is_monomial_l3863_386343

/-- Definition of a monomial as an algebraic expression with only one term -/
def is_monomial (expr : ℚ) : Prop := true

/-- Theorem: 0 is a monomial -/
theorem zero_is_monomial : is_monomial 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_is_monomial_l3863_386343


namespace NUMINAMATH_CALUDE_at_least_one_not_less_than_one_l3863_386375

theorem at_least_one_not_less_than_one (a b c : ℝ) (h : a + b + c = 3) :
  ¬(a < 1 ∧ b < 1 ∧ c < 1) := by
sorry

end NUMINAMATH_CALUDE_at_least_one_not_less_than_one_l3863_386375


namespace NUMINAMATH_CALUDE_factor_expression_l3863_386332

theorem factor_expression (x : ℝ) : 4*x*(x+2) + 9*(x+2) = (x+2)*(4*x+9) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3863_386332


namespace NUMINAMATH_CALUDE_twice_difference_l3863_386335

/-- Given two real numbers m and n, prove that 2(m-n) is equivalent to twice the difference between m and n -/
theorem twice_difference (m n : ℝ) : 2 * (m - n) = 2 * m - 2 * n := by
  sorry

end NUMINAMATH_CALUDE_twice_difference_l3863_386335


namespace NUMINAMATH_CALUDE_b_fraction_of_a_and_c_l3863_386386

def total_amount : ℕ := 1800

def a_share : ℕ := 600

theorem b_fraction_of_a_and_c (b_share c_share : ℕ) 
  (h1 : a_share = (2 : ℕ) * (b_share + c_share) / 5)
  (h2 : total_amount = a_share + b_share + c_share) :
  b_share * 6 = a_share + c_share :=
by sorry

end NUMINAMATH_CALUDE_b_fraction_of_a_and_c_l3863_386386


namespace NUMINAMATH_CALUDE_meal_price_calculation_l3863_386342

/-- Calculates the total price of a meal including tip -/
theorem meal_price_calculation (appetizer_cost entree_cost dessert_cost : ℚ)
  (num_entrees : ℕ) (tip_percentage : ℚ) :
  appetizer_cost = 9 ∧ 
  entree_cost = 20 ∧ 
  num_entrees = 2 ∧
  dessert_cost = 11 ∧
  tip_percentage = 30 / 100 →
  appetizer_cost + num_entrees * entree_cost + dessert_cost + 
  (appetizer_cost + num_entrees * entree_cost + dessert_cost) * tip_percentage = 78 := by
  sorry

end NUMINAMATH_CALUDE_meal_price_calculation_l3863_386342


namespace NUMINAMATH_CALUDE_product_odd_implies_sum_even_l3863_386360

theorem product_odd_implies_sum_even (a b : ℤ) : 
  Odd (a * b) → Even (a + b) := by
  sorry

end NUMINAMATH_CALUDE_product_odd_implies_sum_even_l3863_386360


namespace NUMINAMATH_CALUDE_smallest_integer_solution_l3863_386307

theorem smallest_integer_solution (x : ℤ) : 
  (∀ y : ℤ, |y^2 - 5*y + 6| = 14 → x ≤ y) ↔ x = -1 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_solution_l3863_386307


namespace NUMINAMATH_CALUDE_cubic_equation_integer_solutions_l3863_386331

theorem cubic_equation_integer_solutions (a b : ℤ) :
  a^3 + b^3 + 3*a*b = 1 ↔ (b = 1 - a) ∨ (a = -1 ∧ b = -1) := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_integer_solutions_l3863_386331


namespace NUMINAMATH_CALUDE_same_color_probability_is_121_450_l3863_386336

/-- Represents a 30-sided die with colored sides -/
structure ColoredDie :=
  (maroon : ℕ)
  (teal : ℕ)
  (cyan : ℕ)
  (sparkly : ℕ)
  (total_sides : ℕ)
  (side_sum : maroon + teal + cyan + sparkly = total_sides)

/-- The probability of rolling the same color or element on two identical colored dice -/
def same_color_probability (d : ColoredDie) : ℚ :=
  (d.maroon^2 + d.teal^2 + d.cyan^2 + d.sparkly^2) / d.total_sides^2

/-- The specific die described in the problem -/
def problem_die : ColoredDie :=
  { maroon := 6
  , teal := 9
  , cyan := 10
  , sparkly := 5
  , total_sides := 30
  , side_sum := by rfl }

theorem same_color_probability_is_121_450 :
  same_color_probability problem_die = 121 / 450 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_is_121_450_l3863_386336


namespace NUMINAMATH_CALUDE_average_score_is_106_l3863_386346

/-- The average bowling score of three people -/
def average_bowling_score (score1 score2 score3 : ℕ) : ℚ :=
  (score1 + score2 + score3 : ℚ) / 3

/-- Theorem: The average bowling score of Gretchen, Mitzi, and Beth is 106 -/
theorem average_score_is_106 :
  average_bowling_score 120 113 85 = 106 := by
  sorry

end NUMINAMATH_CALUDE_average_score_is_106_l3863_386346


namespace NUMINAMATH_CALUDE_factor_implies_b_value_l3863_386328

theorem factor_implies_b_value (a b : ℝ) :
  (∃ c : ℝ, ∀ x : ℝ, a * x^3 + b * x^2 + 1 = (x^2 - x - 1) * (x + c)) →
  b = -2 :=
by sorry

end NUMINAMATH_CALUDE_factor_implies_b_value_l3863_386328


namespace NUMINAMATH_CALUDE_namjoon_rank_l3863_386300

theorem namjoon_rank (total_participants : ℕ) (worse_performers : ℕ) (h1 : total_participants = 13) (h2 : worse_performers = 4) :
  total_participants - worse_performers - 1 = 9 :=
by sorry

end NUMINAMATH_CALUDE_namjoon_rank_l3863_386300


namespace NUMINAMATH_CALUDE_min_nuts_is_480_l3863_386344

/-- Represents the nut-gathering process of three squirrels -/
structure NutGathering where
  n1 : ℕ  -- nuts picked by first squirrel
  n2 : ℕ  -- nuts picked by second squirrel
  n3 : ℕ  -- nuts picked by third squirrel

/-- Checks if the nut distribution satisfies the given conditions -/
def is_valid_distribution (ng : NutGathering) : Prop :=
  let total := ng.n1 + ng.n2 + ng.n3
  let s1_final := (5 * ng.n1) / 6 + ng.n2 / 12 + (3 * ng.n3) / 16
  let s2_final := ng.n1 / 12 + (3 * ng.n2) / 4 + (3 * ng.n3) / 16
  let s3_final := ng.n1 / 12 + ng.n2 / 4 + (5 * ng.n3) / 8
  (s1_final : ℚ) / 5 = (s2_final : ℚ) / 3 ∧
  (s2_final : ℚ) / 3 = (s3_final : ℚ) / 2 ∧
  s1_final * 3 = s2_final * 5 ∧
  s2_final * 2 = s3_final * 3 ∧
  (5 * ng.n1) % 6 = 0 ∧
  ng.n2 % 12 = 0 ∧
  (3 * ng.n3) % 16 = 0 ∧
  ng.n1 % 12 = 0 ∧
  (3 * ng.n2) % 4 = 0 ∧
  ng.n2 % 4 = 0 ∧
  (5 * ng.n3) % 8 = 0

/-- The least possible total number of nuts -/
def min_total_nuts : ℕ := 480

/-- Theorem stating that the minimum total number of nuts is 480 -/
theorem min_nuts_is_480 :
  ∀ ng : NutGathering, is_valid_distribution ng →
    ng.n1 + ng.n2 + ng.n3 ≥ min_total_nuts :=
by sorry

end NUMINAMATH_CALUDE_min_nuts_is_480_l3863_386344


namespace NUMINAMATH_CALUDE_cos_difference_value_l3863_386371

theorem cos_difference_value (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 3/2)
  (h2 : Real.cos A + Real.cos B = 1) : 
  Real.cos (A - B) = 5/8 := by
sorry

end NUMINAMATH_CALUDE_cos_difference_value_l3863_386371


namespace NUMINAMATH_CALUDE_mc_question_time_l3863_386387

-- Define the total number of questions
def total_questions : ℕ := 60

-- Define the number of multiple-choice questions
def mc_questions : ℕ := 30

-- Define the number of fill-in-the-blank questions
def fib_questions : ℕ := 30

-- Define the time to learn each fill-in-the-blank question (in minutes)
def fib_time : ℕ := 25

-- Define the total study time (in minutes)
def total_study_time : ℕ := 20 * 60

-- Define the function to calculate the time for multiple-choice questions
def mc_time (x : ℕ) : ℕ := x * mc_questions

-- Define the function to calculate the time for fill-in-the-blank questions
def fib_total_time : ℕ := fib_questions * fib_time

-- Theorem: The time to learn each multiple-choice question is 15 minutes
theorem mc_question_time : 
  ∃ (x : ℕ), mc_time x + fib_total_time = total_study_time ∧ x = 15 :=
by sorry

end NUMINAMATH_CALUDE_mc_question_time_l3863_386387


namespace NUMINAMATH_CALUDE_prop_p_and_q_false_l3863_386396

theorem prop_p_and_q_false : 
  (¬(∀ a b : ℝ, a > b → a^2 > b^2)) ∧ 
  (¬(∃ x : ℝ, x^2 + 2 > 3*x)) := by
  sorry

end NUMINAMATH_CALUDE_prop_p_and_q_false_l3863_386396


namespace NUMINAMATH_CALUDE_smallest_three_digit_with_digit_product_8_l3863_386310

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_product (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * tens * ones

theorem smallest_three_digit_with_digit_product_8 :
  ∀ n : ℕ, is_three_digit n → digit_product n = 8 → 118 ≤ n :=
sorry

end NUMINAMATH_CALUDE_smallest_three_digit_with_digit_product_8_l3863_386310


namespace NUMINAMATH_CALUDE_total_leaves_count_l3863_386365

/-- The number of pots of basil -/
def basil_pots : ℕ := 3

/-- The number of pots of rosemary -/
def rosemary_pots : ℕ := 9

/-- The number of pots of thyme -/
def thyme_pots : ℕ := 6

/-- The number of leaves per basil plant -/
def basil_leaves : ℕ := 4

/-- The number of leaves per rosemary plant -/
def rosemary_leaves : ℕ := 18

/-- The number of leaves per thyme plant -/
def thyme_leaves : ℕ := 30

/-- The total number of leaves from all plants -/
def total_leaves : ℕ := basil_pots * basil_leaves + rosemary_pots * rosemary_leaves + thyme_pots * thyme_leaves

theorem total_leaves_count : total_leaves = 354 := by
  sorry

end NUMINAMATH_CALUDE_total_leaves_count_l3863_386365


namespace NUMINAMATH_CALUDE_max_value_theorem_l3863_386303

theorem max_value_theorem (x y : ℝ) 
  (h1 : 4 * x + 3 * y ≤ 10) 
  (h2 : 3 * x + 5 * y ≤ 12) : 
  ∀ (a b : ℝ), 4 * a + 3 * b ≤ 10 → 3 * a + 5 * b ≤ 12 → 2 * a + b ≤ 46 / 11 :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3863_386303


namespace NUMINAMATH_CALUDE_max_non_managers_l3863_386318

/-- The maximum number of non-managers in a department with 8 managers,
    given that the ratio of managers to non-managers must be greater than 7:24 -/
theorem max_non_managers (managers : ℕ) (non_managers : ℕ) :
  managers = 8 →
  (managers : ℚ) / non_managers > 7 / 24 →
  non_managers ≤ 27 :=
by sorry

end NUMINAMATH_CALUDE_max_non_managers_l3863_386318


namespace NUMINAMATH_CALUDE_special_right_triangle_angles_l3863_386376

/-- A right triangle with the property that when rotated four times,
    each time aligning the shorter leg with the hypotenuse and
    matching the vertex of the acute angle with the vertex of the right angle,
    results in an isosceles fifth triangle. -/
structure SpecialRightTriangle where
  /-- The measure of one of the acute angles in the triangle -/
  α : Real
  /-- The triangle is a right triangle -/
  is_right_triangle : α + (90 - α) + 90 = 180
  /-- The fifth triangle is isosceles -/
  fifth_triangle_isosceles : 4 * α = 180 - 4 * (90 + α)

/-- Theorem stating that the acute angles in the special right triangle are both 90°/11 -/
theorem special_right_triangle_angles (t : SpecialRightTriangle) : t.α = 90 / 11 := by
  sorry

end NUMINAMATH_CALUDE_special_right_triangle_angles_l3863_386376


namespace NUMINAMATH_CALUDE_speak_both_languages_l3863_386389

theorem speak_both_languages (total : ℕ) (latin : ℕ) (french : ℕ) (neither : ℕ) 
  (h_total : total = 25)
  (h_latin : latin = 13)
  (h_french : french = 15)
  (h_neither : neither = 6) :
  latin + french - (total - neither) = 9 := by
  sorry

end NUMINAMATH_CALUDE_speak_both_languages_l3863_386389


namespace NUMINAMATH_CALUDE_sum_of_digits_n_n_is_greatest_divisor_l3863_386377

/-- The greatest number that divides 1305, 4665, and 6905 leaving the same remainder -/
def n : ℕ := 1120

/-- Function to calculate the sum of digits of a natural number -/
def sum_of_digits (m : ℕ) : ℕ :=
  if m < 10 then m else (m % 10) + sum_of_digits (m / 10)

/-- Theorem stating that the sum of digits of n is 4 -/
theorem sum_of_digits_n : sum_of_digits n = 4 := by
  sorry

/-- Theorem stating that n is the greatest number that divides 1305, 4665, and 6905 
    leaving the same remainder -/
theorem n_is_greatest_divisor : 
  ∀ m : ℕ, m > n → ¬(∃ r : ℕ, 1305 % m = r ∧ 4665 % m = r ∧ 6905 % m = r) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_n_n_is_greatest_divisor_l3863_386377


namespace NUMINAMATH_CALUDE_externally_tangent_circles_l3863_386388

theorem externally_tangent_circles (m : ℝ) : 
  let C₁ := {(x, y) : ℝ × ℝ | (x - m)^2 + (y + 2)^2 = 9}
  let C₂ := {(x, y) : ℝ × ℝ | (x + 1)^2 + (y - m)^2 = 4}
  (∃ (p : ℝ × ℝ), p ∈ C₁ ∧ p ∈ C₂ ∧ 
    (∀ (q : ℝ × ℝ), q ∈ C₁ ∧ q ∈ C₂ → q = p) ∧
    (∀ (r : ℝ × ℝ), r ∈ C₁ → ∃ (s : ℝ × ℝ), s ∈ C₂ ∧ s ≠ r)) →
  m = 2 ∨ m = -5 :=
by sorry

end NUMINAMATH_CALUDE_externally_tangent_circles_l3863_386388


namespace NUMINAMATH_CALUDE_unique_base7_digit_divisible_by_13_l3863_386329

/-- Converts a base-7 number of the form 3dd6_7 to base-10 --/
def base7ToBase10 (d : ℕ) : ℕ := 3 * 7^3 + d * 7^2 + d * 7 + 6

/-- Checks if a number is divisible by 13 --/
def isDivisibleBy13 (n : ℕ) : Prop := n % 13 = 0

/-- Represents a base-7 digit --/
def isBase7Digit (d : ℕ) : Prop := d ≤ 6

theorem unique_base7_digit_divisible_by_13 :
  ∃! d : ℕ, isBase7Digit d ∧ isDivisibleBy13 (base7ToBase10 d) ∧ d = 2 := by sorry

end NUMINAMATH_CALUDE_unique_base7_digit_divisible_by_13_l3863_386329


namespace NUMINAMATH_CALUDE_symmetry_implies_x_equals_one_l3863_386373

/-- A function f: ℝ → ℝ has symmetric graphs for y = f(x-1) and y = f(1-x) with respect to x = 1 -/
def has_symmetric_graphs (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x - 1) = f (1 - x)

/-- If a function has symmetric graphs for y = f(x-1) and y = f(1-x), 
    then they are symmetric with respect to x = 1 -/
theorem symmetry_implies_x_equals_one (f : ℝ → ℝ) 
    (h : has_symmetric_graphs f) : 
    ∃ a : ℝ, a = 1 ∧ ∀ x : ℝ, f (a + (x - a)) = f (a - (x - a)) :=
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_x_equals_one_l3863_386373


namespace NUMINAMATH_CALUDE_min_value_reciprocal_product_l3863_386359

theorem min_value_reciprocal_product (a b : ℝ) : 
  a > 0 → b > 0 → 2*a + b = 4 → (∀ x y : ℝ, x > 0 → y > 0 → 2*x + y = 4 → 1/(a*b) ≤ 1/(x*y)) ∧ (∃ a₀ b₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ 2*a₀ + b₀ = 4 ∧ 1/(a₀*b₀) = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_product_l3863_386359


namespace NUMINAMATH_CALUDE_initial_goldfish_count_l3863_386362

theorem initial_goldfish_count (died : ℕ) (remaining : ℕ) (h1 : died = 32) (h2 : remaining = 57) :
  died + remaining = 89 := by
  sorry

end NUMINAMATH_CALUDE_initial_goldfish_count_l3863_386362


namespace NUMINAMATH_CALUDE_cube_surface_area_increase_l3863_386364

theorem cube_surface_area_increase :
  ∀ s : ℝ, s > 0 →
  let original_surface_area := 6 * s^2
  let new_edge_length := 1.4 * s
  let new_surface_area := 6 * new_edge_length^2
  (new_surface_area - original_surface_area) / original_surface_area = 0.96 :=
by
  sorry

#check cube_surface_area_increase

end NUMINAMATH_CALUDE_cube_surface_area_increase_l3863_386364


namespace NUMINAMATH_CALUDE_chord_length_l3863_386395

/-- The length of the chord formed by the intersection of a circle and a line --/
theorem chord_length (x y : ℝ) : 
  let circle := (fun (x y : ℝ) ↦ (x - 1)^2 + y^2 = 4)
  let line := (fun (x y : ℝ) ↦ x + y + 1 = 0)
  let chord_length := 
    Real.sqrt (8 - 2 * ((1 * 1 + 1 * 0 + 1) / Real.sqrt (1^2 + 1^2))^2)
  (∃ (a b : ℝ × ℝ), circle a.1 a.2 ∧ circle b.1 b.2 ∧ 
                     line a.1 a.2 ∧ line b.1 b.2 ∧ 
                     a ≠ b) →
  chord_length = 2 * Real.sqrt 2 := by
sorry


end NUMINAMATH_CALUDE_chord_length_l3863_386395


namespace NUMINAMATH_CALUDE_yellow_ball_count_l3863_386379

theorem yellow_ball_count (red_count : ℕ) (total_count : ℕ) 
  (h1 : red_count = 10)
  (h2 : (red_count : ℚ) / total_count = 1 / 3) :
  total_count - red_count = 20 := by
  sorry

end NUMINAMATH_CALUDE_yellow_ball_count_l3863_386379


namespace NUMINAMATH_CALUDE_intersection_implies_determinant_one_l3863_386369

/-- Given three lines that intersect at one point, prove that the determinant is 1 -/
theorem intersection_implies_determinant_one 
  (a : ℝ) 
  (h1 : ∃ (x y : ℝ), ax + y + 3 = 0 ∧ x + y + 2 = 0 ∧ 2*x - y + 1 = 0) :
  Matrix.det ![![a, 1], ![1, 1]] = 1 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_determinant_one_l3863_386369


namespace NUMINAMATH_CALUDE_field_trip_students_l3863_386385

/-- The number of people a van can hold -/
def van_capacity : ℕ := 5

/-- The number of adults going on the trip -/
def num_adults : ℕ := 3

/-- The number of vans needed for the trip -/
def num_vans : ℕ := 3

/-- The number of students going on the field trip -/
def num_students : ℕ := van_capacity * num_vans - num_adults

theorem field_trip_students : num_students = 12 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_students_l3863_386385


namespace NUMINAMATH_CALUDE_e_pow_f_neg_two_eq_half_l3863_386305

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem e_pow_f_neg_two_eq_half
  (f : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_log : ∀ x > 0, f x = Real.log x) :
  Real.exp (f (-2)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_e_pow_f_neg_two_eq_half_l3863_386305


namespace NUMINAMATH_CALUDE_root_product_reciprocal_sum_l3863_386390

theorem root_product_reciprocal_sum (p q : ℝ) 
  (h1 : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x^2 + p*x + q = (x - x1) * (x - x2))
  (h2 : ∃ x3 x4 : ℝ, x3 ≠ x4 ∧ x^2 + q*x + p = (x - x3) * (x - x4))
  (h3 : ∀ x1 x2 x3 x4 : ℝ, 
    (x^2 + p*x + q = (x - x1) * (x - x2)) → 
    (x^2 + q*x + p = (x - x3) * (x - x4)) → 
    x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4) :
  ∃ x1 x2 x3 x4 : ℝ, 
    (x^2 + p*x + q = (x - x1) * (x - x2)) ∧
    (x^2 + q*x + p = (x - x3) * (x - x4)) ∧
    1 / (x1 * x3) + 1 / (x1 * x4) + 1 / (x2 * x3) + 1 / (x2 * x4) = 1 :=
by sorry

end NUMINAMATH_CALUDE_root_product_reciprocal_sum_l3863_386390


namespace NUMINAMATH_CALUDE_quartic_equation_minimum_l3863_386326

theorem quartic_equation_minimum (a b : ℝ) : 
  (∃ x : ℝ, x^4 + a*x^3 + b*x^2 + a*x + 1 = 0) → 
  a^2 + b^2 ≥ 4/5 := by
sorry

end NUMINAMATH_CALUDE_quartic_equation_minimum_l3863_386326


namespace NUMINAMATH_CALUDE_savings_calculation_l3863_386382

/-- Calculates the amount left in savings after distributing funds to family members --/
def savings_amount (initial : ℚ) (wife_fraction : ℚ) (son1_fraction : ℚ) (son2_fraction : ℚ) : ℚ :=
  let wife_share := wife_fraction * initial
  let after_wife := initial - wife_share
  let son1_share := son1_fraction * after_wife
  let after_son1 := after_wife - son1_share
  let son2_share := son2_fraction * after_son1
  after_son1 - son2_share

/-- Theorem stating the amount left in savings after distribution --/
theorem savings_calculation :
  savings_amount 2000 (2/5) (2/5) (40/100) = 432 := by
  sorry

end NUMINAMATH_CALUDE_savings_calculation_l3863_386382


namespace NUMINAMATH_CALUDE_octagon_ratio_l3863_386302

/-- Represents an octagon with specific properties -/
structure Octagon where
  total_area : ℝ
  unit_squares : ℕ
  pq_divides_equally : Prop
  below_pq_square : ℝ
  below_pq_triangle_base : ℝ
  xq_plus_qy : ℝ

/-- The theorem to be proved -/
theorem octagon_ratio (o : Octagon) 
  (h1 : o.total_area = 12)
  (h2 : o.unit_squares = 12)
  (h3 : o.pq_divides_equally)
  (h4 : o.below_pq_square = 1)
  (h5 : o.below_pq_triangle_base = 6)
  (h6 : o.xq_plus_qy = 6) :
  ∃ (xq qy : ℝ), xq / qy = 2 ∧ xq + qy = o.xq_plus_qy :=
sorry

end NUMINAMATH_CALUDE_octagon_ratio_l3863_386302


namespace NUMINAMATH_CALUDE_cubic_polynomial_real_root_l3863_386354

/-- Given a cubic polynomial ax³ + 3x² + bx - 125 = 0 where a and b are real numbers,
    and -2 - 3i is a root of this polynomial, the real root of the polynomial is 5/2. -/
theorem cubic_polynomial_real_root (a b : ℝ) : 
  (∃ (z : ℂ), z = -2 - 3*I ∧ a * z^3 + 3 * z^2 + b * z - 125 = 0) →
  (∃ (x : ℝ), a * x^3 + 3 * x^2 + b * x - 125 = 0 ∧ x = 5/2) :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_real_root_l3863_386354


namespace NUMINAMATH_CALUDE_complex_number_equality_l3863_386311

theorem complex_number_equality : (((1 + Complex.I)^4) / (1 - Complex.I)) + 2 = -2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equality_l3863_386311


namespace NUMINAMATH_CALUDE_handshakes_in_specific_tournament_l3863_386301

/-- Represents a tennis tournament with the given conditions -/
structure TennisTournament where
  num_teams : Nat
  players_per_team : Nat
  abstaining_player : Nat
  abstained_team : Nat

/-- Calculates the number of handshakes in the tournament -/
def count_handshakes (t : TennisTournament) : Nat :=
  sorry

/-- Theorem stating the number of handshakes in the specific tournament scenario -/
theorem handshakes_in_specific_tournament :
  ∀ (t : TennisTournament),
    t.num_teams = 4 ∧
    t.players_per_team = 2 ∧
    t.abstaining_player ≥ 1 ∧
    t.abstaining_player ≤ 8 ∧
    t.abstained_team ≥ 1 ∧
    t.abstained_team ≤ 4 ∧
    t.abstained_team ≠ ((t.abstaining_player - 1) / 2 + 1) →
    count_handshakes t = 22 :=
  sorry

end NUMINAMATH_CALUDE_handshakes_in_specific_tournament_l3863_386301


namespace NUMINAMATH_CALUDE_average_of_first_four_l3863_386372

theorem average_of_first_four (numbers : Fin 6 → ℝ) 
  (h1 : (numbers 0 + numbers 1 + numbers 2 + numbers 3 + numbers 4 + numbers 5) / 6 = 30)
  (h2 : (numbers 3 + numbers 4 + numbers 5) / 3 = 35)
  (h3 : numbers 3 = 25) :
  (numbers 0 + numbers 1 + numbers 2 + numbers 3) / 4 = 18.75 := by
sorry

end NUMINAMATH_CALUDE_average_of_first_four_l3863_386372


namespace NUMINAMATH_CALUDE_solution_system1_solution_system2_l3863_386391

-- System 1
def system1 (x y : ℝ) : Prop :=
  3 * (x - 1) = y + 5 ∧ 5 * (y - 1) = 3 * (x + 5)

-- System 2
def system2 (x y a : ℝ) : Prop :=
  2 * x + 4 * y = a ∧ 7 * x - 2 * y = 3 * a

theorem solution_system1 :
  ∃ x y : ℝ, system1 x y ∧ x = 5 ∧ y = 7 := by sorry

theorem solution_system2 :
  ∀ a : ℝ, ∃ x y : ℝ, system2 x y a ∧ x = 7 / 16 * a ∧ y = 1 / 32 * a := by sorry

end NUMINAMATH_CALUDE_solution_system1_solution_system2_l3863_386391
