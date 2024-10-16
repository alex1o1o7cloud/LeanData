import Mathlib

namespace NUMINAMATH_CALUDE_thousand_power_division_l2878_287893

theorem thousand_power_division :
  1000 * (1000^1000) / (500^1000) = 2^1001 * 500 := by
  sorry

end NUMINAMATH_CALUDE_thousand_power_division_l2878_287893


namespace NUMINAMATH_CALUDE_cos_600_degrees_l2878_287884

theorem cos_600_degrees : Real.cos (600 * π / 180) = - (1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_cos_600_degrees_l2878_287884


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l2878_287877

theorem geometric_sequence_third_term
  (a₁ : ℝ)
  (a₅ : ℝ)
  (h₁ : a₁ = 4)
  (h₂ : a₅ = 1296)
  (h₃ : ∀ n : ℕ, 1 ≤ n ∧ n ≤ 5 → ∃ r : ℝ, a₁ * r^(n-1) = a₁ * (a₅ / a₁)^((n-1)/4)) :
  ∃ a₃ : ℝ, a₃ = 36 ∧ a₃ = a₁ * (a₅ / a₁)^(1/2) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l2878_287877


namespace NUMINAMATH_CALUDE_circumcenter_on_side_implies_right_angled_l2878_287872

/-- A triangle is represented by its three vertices in a 2D plane. -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The circumcenter of a triangle is the point where the perpendicular bisectors of the sides intersect. -/
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

/-- A predicate to check if a point lies on a side of a triangle. -/
def point_on_side (p : ℝ × ℝ) (t : Triangle) : Prop := sorry

/-- A predicate to check if a triangle is right-angled. -/
def is_right_angled (t : Triangle) : Prop := sorry

/-- Theorem: If the circumcenter of a triangle lies on one of its sides, then the triangle is right-angled. -/
theorem circumcenter_on_side_implies_right_angled (t : Triangle) :
  point_on_side (circumcenter t) t → is_right_angled t := by
  sorry

end NUMINAMATH_CALUDE_circumcenter_on_side_implies_right_angled_l2878_287872


namespace NUMINAMATH_CALUDE_triangle_ABC_properties_l2878_287883

theorem triangle_ABC_properties (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  A + B + C = π →
  -- Given equation
  2 * Real.sin (2 * A) * Real.cos A - Real.sin (3 * A) + Real.sqrt 3 * Real.cos A = Real.sqrt 3 →
  -- a, b, c are sides opposite to angles A, B, C
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- a = 1
  a = 1 →
  -- Given trigonometric equation
  Real.sin A + Real.sin (B - C) = 2 * Real.sin (2 * C) →
  -- Conclusions
  A = π / 3 ∧ 
  (1/2 * b * c * Real.sin A) = Real.sqrt 3 / 6 :=
by sorry

end NUMINAMATH_CALUDE_triangle_ABC_properties_l2878_287883


namespace NUMINAMATH_CALUDE_number_calculation_l2878_287866

theorem number_calculation (number : ℝ) : 
  (number / 0.3 = 0.03) → number = 0.009 := by
  sorry

end NUMINAMATH_CALUDE_number_calculation_l2878_287866


namespace NUMINAMATH_CALUDE_pet_store_theorem_l2878_287857

/-- The number of ways to choose and assign different pets to four people -/
def pet_store_combinations : ℕ :=
  let puppies : ℕ := 12
  let kittens : ℕ := 10
  let hamsters : ℕ := 9
  let parrots : ℕ := 7
  let people : ℕ := 4
  puppies * kittens * hamsters * parrots * Nat.factorial people

/-- Theorem stating the number of combinations for the pet store problem -/
theorem pet_store_theorem : pet_store_combinations = 181440 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_theorem_l2878_287857


namespace NUMINAMATH_CALUDE_bike_only_households_l2878_287802

theorem bike_only_households (total : ℕ) (neither : ℕ) (both : ℕ) (with_car : ℕ) 
  (h1 : total = 90)
  (h2 : neither = 11)
  (h3 : both = 22)
  (h4 : with_car = 44) :
  total - neither - with_car + both = 35 := by
  sorry

end NUMINAMATH_CALUDE_bike_only_households_l2878_287802


namespace NUMINAMATH_CALUDE_A_divisible_by_8_l2878_287887

def A (n : ℕ) : ℕ := 5^n + 2 * 3^(n-1) + 1

theorem A_divisible_by_8 (n : ℕ) (h : n > 0) : 8 ∣ A n := by
  sorry

end NUMINAMATH_CALUDE_A_divisible_by_8_l2878_287887


namespace NUMINAMATH_CALUDE_janette_beef_jerky_dinner_l2878_287824

/-- Calculates the number of beef jerky pieces eaten for dinner each day during a camping trip. -/
def beef_jerky_for_dinner (days : ℕ) (total_pieces : ℕ) (breakfast_pieces : ℕ) (lunch_pieces : ℕ) (pieces_after_sharing : ℕ) : ℕ :=
  let pieces_before_sharing := 2 * pieces_after_sharing
  let pieces_eaten := total_pieces - pieces_before_sharing
  let pieces_for_breakfast_and_lunch := (breakfast_pieces + lunch_pieces) * days
  (pieces_eaten - pieces_for_breakfast_and_lunch) / days

/-- Theorem stating that Janette ate 2 pieces of beef jerky for dinner each day during her camping trip. -/
theorem janette_beef_jerky_dinner :
  beef_jerky_for_dinner 5 40 1 1 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_janette_beef_jerky_dinner_l2878_287824


namespace NUMINAMATH_CALUDE_exists_quadrilateral_perpendicular_diagonals_not_all_natural_cubed_greater_than_squared_l2878_287808

-- Define a structure for a quadrilateral
structure Quadrilateral where
  vertices : Fin 4 → ℝ × ℝ

-- Define a function to check if diagonals are perpendicular
def diagonalsPerpendicular (q : Quadrilateral) : Prop :=
  sorry

-- Statement 1
theorem exists_quadrilateral_perpendicular_diagonals :
  ∃ q : Quadrilateral, diagonalsPerpendicular q :=
sorry

-- Statement 2
theorem not_all_natural_cubed_greater_than_squared :
  ¬ ∀ x : ℕ, x^3 > x^2 :=
sorry

end NUMINAMATH_CALUDE_exists_quadrilateral_perpendicular_diagonals_not_all_natural_cubed_greater_than_squared_l2878_287808


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2878_287804

/-- An arithmetic sequence -/
def arithmeticSeq (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The property that a₁ + a₇ + a₁₃ = 4 -/
def sumProperty (a : ℕ → ℚ) : Prop :=
  a 1 + a 7 + a 13 = 4

theorem arithmetic_sequence_sum (a : ℕ → ℚ) 
  (h1 : arithmeticSeq a) (h2 : sumProperty a) : 
  a 2 + a 12 = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2878_287804


namespace NUMINAMATH_CALUDE_passing_percentage_l2878_287870

def max_marks : ℕ := 800
def obtained_marks : ℕ := 175
def failed_by : ℕ := 89

theorem passing_percentage :
  (((obtained_marks + failed_by : ℚ) / max_marks) * 100).floor = 33 := by
  sorry

end NUMINAMATH_CALUDE_passing_percentage_l2878_287870


namespace NUMINAMATH_CALUDE_real_solutions_quadratic_l2878_287851

theorem real_solutions_quadratic (x : ℝ) :
  (∃ y : ℝ, 9 * y^2 + 6 * x * y + 2 * x + 1 = 0) ↔ (x < 2 - Real.sqrt 6 ∨ x > 2 + Real.sqrt 6) :=
sorry

end NUMINAMATH_CALUDE_real_solutions_quadratic_l2878_287851


namespace NUMINAMATH_CALUDE_reciprocal_absolute_value_l2878_287888

theorem reciprocal_absolute_value (x : ℝ) : 
  (1 / |x|) = -4 → x = 1/4 ∨ x = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_absolute_value_l2878_287888


namespace NUMINAMATH_CALUDE_french_exam_words_to_learn_l2878_287854

/-- The least number of words to learn for a French exam -/
def least_words_to_learn : ℕ := 569

theorem french_exam_words_to_learn 
  (total_words : ℕ) 
  (recall_rate : ℚ) 
  (target_recall : ℚ) 
  (h1 : total_words = 600)
  (h2 : recall_rate = 95 / 100)
  (h3 : target_recall = 90 / 100) :
  (↑least_words_to_learn : ℚ) ≥ (target_recall * total_words) / recall_rate ∧ 
  (↑(least_words_to_learn - 1) : ℚ) < (target_recall * total_words) / recall_rate :=
sorry

end NUMINAMATH_CALUDE_french_exam_words_to_learn_l2878_287854


namespace NUMINAMATH_CALUDE_average_of_s_and_t_l2878_287864

theorem average_of_s_and_t (s t : ℝ) : 
  (1 + 3 + 7 + s + t) / 5 = 12 → (s + t) / 2 = 24.5 := by
sorry

end NUMINAMATH_CALUDE_average_of_s_and_t_l2878_287864


namespace NUMINAMATH_CALUDE_slope_intercept_form_parallel_lines_a_value_l2878_287874

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The slope-intercept form of a line ax + by + c = 0 is y = (-a/b)x - (c/b) when b ≠ 0 -/
theorem slope_intercept_form {a b c : ℝ} (hb : b ≠ 0) :
  (∀ x y : ℝ, a * x + b * y + c = 0 ↔ y = (-a/b) * x - (c/b)) :=
sorry

theorem parallel_lines_a_value :
  ∀ a : ℝ, (∀ x y : ℝ, a * x - y + a = 0 ↔ (2*a-3) * x + a * y - a = 0) → a = -3 :=
sorry

end NUMINAMATH_CALUDE_slope_intercept_form_parallel_lines_a_value_l2878_287874


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_positive_n_for_unique_solution_l2878_287814

/-- For a quadratic equation ax^2 + bx + c = 0, its discriminant is b^2 - 4ac --/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- A quadratic equation has exactly one solution if and only if its discriminant is zero --/
def has_exactly_one_solution (a b c : ℝ) : Prop :=
  discriminant a b c = 0

theorem unique_solution_quadratic (n : ℝ) :
  has_exactly_one_solution 4 n 16 ↔ n = 16 ∨ n = -16 :=
sorry

theorem positive_n_for_unique_solution :
  ∃ n : ℝ, n > 0 ∧ has_exactly_one_solution 4 n 16 ∧ n = 16 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_positive_n_for_unique_solution_l2878_287814


namespace NUMINAMATH_CALUDE_smallest_cube_factor_l2878_287844

theorem smallest_cube_factor (z : ℕ) (hz : z.Prime ∧ z > 7) :
  let y := 19408850
  (∀ k : ℕ, k > 0 ∧ k < y → ¬∃ n : ℕ, (31360 * z) * k = n^3) ∧
  ∃ n : ℕ, (31360 * z) * y = n^3 :=
sorry

end NUMINAMATH_CALUDE_smallest_cube_factor_l2878_287844


namespace NUMINAMATH_CALUDE_absolute_value_equality_l2878_287800

theorem absolute_value_equality (x : ℝ) (y : ℝ) :
  y > 0 →
  |3 * x - 2 * Real.log y| = 3 * x + 2 * Real.log y →
  x = 0 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l2878_287800


namespace NUMINAMATH_CALUDE_negative_four_squared_equals_sixteen_l2878_287829

theorem negative_four_squared_equals_sixteen :
  (-4 : ℤ) ^ 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_negative_four_squared_equals_sixteen_l2878_287829


namespace NUMINAMATH_CALUDE_smallest_k_with_remainder_one_l2878_287850

theorem smallest_k_with_remainder_one : ∃! k : ℕ, 
  k > 1 ∧ 
  k % 13 = 1 ∧ 
  k % 7 = 1 ∧ 
  k % 3 = 1 ∧ 
  k % 2 = 1 ∧
  ∀ m : ℕ, m > 1 → m % 13 = 1 → m % 7 = 1 → m % 3 = 1 → m % 2 = 1 → k ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_k_with_remainder_one_l2878_287850


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2878_287899

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x + 4 ≤ 4) ↔ (∃ x : ℝ, x^2 - 2*x + 4 > 4) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2878_287899


namespace NUMINAMATH_CALUDE_workshop_2_production_l2878_287886

/-- Represents the production and sampling data for a factory with three workshops -/
structure FactoryData where
  total_production : ℕ
  sample_1 : ℕ
  sample_2 : ℕ
  sample_3 : ℕ

/-- Checks if three numbers form an arithmetic sequence -/
def isArithmeticSequence (a b c : ℕ) : Prop :=
  b - a = c - b

/-- The main theorem about the factory's production -/
theorem workshop_2_production (data : FactoryData) 
    (h_total : data.total_production = 3600)
    (h_arithmetic : isArithmeticSequence data.sample_1 data.sample_2 data.sample_3) :
    data.sample_2 = 1200 := by
  sorry


end NUMINAMATH_CALUDE_workshop_2_production_l2878_287886


namespace NUMINAMATH_CALUDE_smallest_resolvable_debt_l2878_287837

theorem smallest_resolvable_debt (pig_value goat_value : ℕ) 
  (h_pig : pig_value = 350) (h_goat : goat_value = 250) :
  (∃ (d : ℕ), d > 0 ∧ 
   (∃ (p g : ℤ), d = pig_value * p + goat_value * g) ∧
   (∀ (d' : ℕ), d' > 0 → d' < d → 
    ¬(∃ (p' g' : ℤ), d' = pig_value * p' + goat_value * g'))) →
  (∃ (d : ℕ), d = 50 ∧ d > 0 ∧ 
   (∃ (p g : ℤ), d = pig_value * p + goat_value * g) ∧
   (∀ (d' : ℕ), d' > 0 → d' < d → 
    ¬(∃ (p' g' : ℤ), d' = pig_value * p' + goat_value * g'))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_resolvable_debt_l2878_287837


namespace NUMINAMATH_CALUDE_small_pizza_has_four_slices_l2878_287839

/-- The number of slices in a small pizza -/
def small_pizza_slices : ℕ := sorry

/-- The number of slices in a large pizza -/
def large_pizza_slices : ℕ := 8

/-- The number of small pizzas purchased -/
def small_pizzas_bought : ℕ := 3

/-- The number of large pizzas purchased -/
def large_pizzas_bought : ℕ := 2

/-- The number of slices George eats -/
def george_slices : ℕ := 3

/-- The number of slices Bob eats -/
def bob_slices : ℕ := george_slices + 1

/-- The number of slices Susie eats -/
def susie_slices : ℕ := bob_slices / 2

/-- The number of slices Bill eats -/
def bill_slices : ℕ := 3

/-- The number of slices Fred eats -/
def fred_slices : ℕ := 3

/-- The number of slices Mark eats -/
def mark_slices : ℕ := 3

/-- The number of slices left over -/
def leftover_slices : ℕ := 10

theorem small_pizza_has_four_slices : small_pizza_slices = 4 := by
  sorry

end NUMINAMATH_CALUDE_small_pizza_has_four_slices_l2878_287839


namespace NUMINAMATH_CALUDE_store_customer_ratio_l2878_287882

theorem store_customer_ratio : 
  let non_holiday_rate : ℚ := 175  -- customers per hour during non-holiday season
  let holiday_total : ℕ := 2800    -- total customers during holiday season
  let holiday_hours : ℕ := 8       -- number of hours during holiday season
  let holiday_rate : ℚ := holiday_total / holiday_hours  -- customers per hour during holiday season
  holiday_rate / non_holiday_rate = 2 := by
sorry

end NUMINAMATH_CALUDE_store_customer_ratio_l2878_287882


namespace NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l2878_287898

theorem polar_to_rectangular_conversion :
  let r : ℝ := 8
  let θ : ℝ := π / 4
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x = 4 * Real.sqrt 2) ∧ (y = 4 * Real.sqrt 2) := by sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l2878_287898


namespace NUMINAMATH_CALUDE_leftover_floss_amount_l2878_287868

/-- Calculates the amount of leftover floss when distributing to students -/
def leftover_floss (num_students : ℕ) (floss_per_student : ℚ) (floss_per_packet : ℕ) : ℚ :=
  let total_needed : ℚ := num_students * floss_per_student
  let packets_needed : ℕ := (total_needed / floss_per_packet).ceil.toNat
  packets_needed * floss_per_packet - total_needed

/-- Theorem stating the leftover floss amount for the given problem -/
theorem leftover_floss_amount :
  leftover_floss 20 (3/2) 35 = 5 := by
sorry

end NUMINAMATH_CALUDE_leftover_floss_amount_l2878_287868


namespace NUMINAMATH_CALUDE_manicure_cost_calculation_l2878_287855

/-- The cost of a manicure in a nail salon. -/
def manicure_cost (total_revenue : ℚ) (total_fingers : ℕ) (fingers_per_person : ℕ) (non_clients : ℕ) : ℚ :=
  total_revenue / ((total_fingers / fingers_per_person) - non_clients)

/-- Theorem stating the cost of a manicure in the given scenario. -/
theorem manicure_cost_calculation :
  manicure_cost 200 210 10 11 = 952 / 100 := by sorry

end NUMINAMATH_CALUDE_manicure_cost_calculation_l2878_287855


namespace NUMINAMATH_CALUDE_min_distance_to_line_l2878_287845

/-- The minimum distance from the origin to a point on the line x + y - 4 = 0 is 2√2 -/
theorem min_distance_to_line : ∀ (x y : ℝ), 
  x + y - 4 = 0 → 
  ∃ (d : ℝ), d = 2 * Real.sqrt 2 ∧ 
  ∀ (x' y' : ℝ), x' + y' - 4 = 0 → 
  Real.sqrt (x' ^ 2 + y' ^ 2) ≥ d := by
  sorry


end NUMINAMATH_CALUDE_min_distance_to_line_l2878_287845


namespace NUMINAMATH_CALUDE_charity_race_participants_l2878_287801

theorem charity_race_participants (students_20 students_30 : ℕ) : 
  students_20 = 10 →
  students_20 * 20 + students_30 * 30 = 800 →
  students_20 + students_30 = 30 := by
sorry

end NUMINAMATH_CALUDE_charity_race_participants_l2878_287801


namespace NUMINAMATH_CALUDE_school_survey_is_stratified_sampling_l2878_287873

/-- Represents a sampling method -/
inductive SamplingMethod
  | DrawingLots
  | RandomNumber
  | Systematic
  | Stratified

/-- Represents a population divided into groups -/
structure Population where
  totalSize : ℕ
  groups : List (ℕ × ℕ)  -- (group size, sample size) pairs

/-- Checks if a sampling method is stratified -/
def isStratifiedSampling (pop : Population) (method : SamplingMethod) : Prop :=
  method = SamplingMethod.Stratified ∧
  pop.groups.length ≥ 2 ∧
  (∀ (g₁ g₂ : ℕ × ℕ), g₁ ∈ pop.groups → g₂ ∈ pop.groups →
    (g₁.1 : ℚ) / (g₂.1 : ℚ) = (g₁.2 : ℚ) / (g₂.2 : ℚ))

/-- The main theorem to prove -/
theorem school_survey_is_stratified_sampling
  (totalStudents : ℕ)
  (maleStudents femaleStudents : ℕ)
  (maleSample femaleSample : ℕ)
  (h_total : totalStudents = maleStudents + femaleStudents)
  (h_male_ratio : (maleStudents : ℚ) / (totalStudents : ℚ) = 2 / 5)
  (h_female_ratio : (femaleStudents : ℚ) / (totalStudents : ℚ) = 3 / 5)
  (h_sample_ratio : (maleSample : ℚ) / (femaleSample : ℚ) = 2 / 3)
  : isStratifiedSampling
      { totalSize := totalStudents,
        groups := [(maleStudents, maleSample), (femaleStudents, femaleSample)] }
      SamplingMethod.Stratified :=
by sorry

end NUMINAMATH_CALUDE_school_survey_is_stratified_sampling_l2878_287873


namespace NUMINAMATH_CALUDE_jellybean_box_capacity_l2878_287867

theorem jellybean_box_capacity 
  (bert_capacity : ℕ)
  (bert_volume : ℝ)
  (lisa_volume : ℝ)
  (h1 : bert_capacity = 150)
  (h2 : lisa_volume = 24 * bert_volume)
  (h3 : ∀ (c : ℝ) (v : ℝ), c / v = bert_capacity / bert_volume → c = (v / bert_volume) * bert_capacity)
  : (lisa_volume / bert_volume) * bert_capacity = 3600 :=
by sorry

end NUMINAMATH_CALUDE_jellybean_box_capacity_l2878_287867


namespace NUMINAMATH_CALUDE_nested_radical_value_l2878_287815

/-- The value of the infinite nested radical sqrt(16 + sqrt(16 + sqrt(16 + ...))) -/
noncomputable def nestedRadical : ℝ :=
  Real.sqrt (16 + Real.sqrt (16 + Real.sqrt (16 + Real.sqrt (16 + Real.sqrt (16 + Real.sqrt 16)))))

/-- Theorem stating that the nested radical equals (1 + sqrt(65)) / 2 -/
theorem nested_radical_value : nestedRadical = (1 + Real.sqrt 65) / 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_radical_value_l2878_287815


namespace NUMINAMATH_CALUDE_arc_length_formula_l2878_287885

theorem arc_length_formula (r : ℝ) (θ : ℝ) (h : r = 8) (h' : θ = 5 * π / 3) :
  r * θ = 40 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_arc_length_formula_l2878_287885


namespace NUMINAMATH_CALUDE_extreme_value_implies_m_plus_n_11_l2878_287876

/-- A function f with an extreme value of 0 at x = -1 -/
def f (m n : ℝ) (x : ℝ) : ℝ := x^3 + 3*m*x^2 + n*x + m^2

/-- The derivative of f -/
def f' (m n : ℝ) (x : ℝ) : ℝ := 3*x^2 + 6*m*x + n

theorem extreme_value_implies_m_plus_n_11 (m n : ℝ) :
  (f m n (-1) = 0) →
  (f' m n (-1) = 0) →
  (m + n = 11) :=
by sorry

end NUMINAMATH_CALUDE_extreme_value_implies_m_plus_n_11_l2878_287876


namespace NUMINAMATH_CALUDE_parallelepiped_net_theorem_l2878_287810

/-- Represents a parallelepiped with dimensions length, width, and height -/
structure Parallelepiped where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents a net of unfolded parallelepiped -/
structure Net where
  squares : ℕ

/-- Calculates the surface area of a parallelepiped -/
def surfaceArea (p : Parallelepiped) : ℕ :=
  2 * (p.length * p.width + p.length * p.height + p.width * p.height)

/-- Unfolds a parallelepiped into a net -/
def unfold (p : Parallelepiped) : Net :=
  { squares := surfaceArea p }

/-- Removes one square from a net -/
def removeSquare (n : Net) : Net :=
  { squares := n.squares - 1 }

theorem parallelepiped_net_theorem (p : Parallelepiped) 
  (h1 : p.length = 2) (h2 : p.width = 1) (h3 : p.height = 1) :
  ∃ (n : Net), 
    (unfold p).squares = 10 ∧ 
    (removeSquare (unfold p)).squares = 9 ∧
    ∃ (valid : Bool), valid = true :=
  sorry

end NUMINAMATH_CALUDE_parallelepiped_net_theorem_l2878_287810


namespace NUMINAMATH_CALUDE_max_value_implies_m_l2878_287805

-- Define the function f(x) = x^2 - 2x + m
def f (x m : ℝ) : ℝ := x^2 - 2*x + m

-- Define the interval [0, 3]
def interval : Set ℝ := { x | 0 ≤ x ∧ x ≤ 3 }

-- State the theorem
theorem max_value_implies_m (m : ℝ) :
  (∀ x ∈ interval, f x m ≤ 1) ∧
  (∃ x ∈ interval, f x m = 1) →
  m = -2 :=
sorry

end NUMINAMATH_CALUDE_max_value_implies_m_l2878_287805


namespace NUMINAMATH_CALUDE_bill_share_calculation_l2878_287830

/-- Represents the profit share of a partner -/
structure ProfitShare where
  amount : ℕ

/-- Represents the profit-sharing ratio for a partnership -/
structure ProfitRatio where
  bess : ℕ
  bill : ℕ
  bob : ℕ

/-- Calculates a partner's share based on the total profit and their ratio -/
def calculateShare (totalProfit : ℕ) (partnerRatio : ℕ) (totalRatio : ℕ) : ProfitShare :=
  { amount := totalProfit * partnerRatio / totalRatio }

theorem bill_share_calculation (ratio : ProfitRatio) (bobShare : ProfitShare) :
  ratio.bess = 1 → ratio.bill = 2 → ratio.bob = 3 → bobShare.amount = 900 →
  (calculateShare bobShare.amount ratio.bill (ratio.bess + ratio.bill + ratio.bob)).amount = 600 := by
  sorry

end NUMINAMATH_CALUDE_bill_share_calculation_l2878_287830


namespace NUMINAMATH_CALUDE_insect_crawl_properties_l2878_287891

def crawl_distances : List ℤ := [5, -3, 10, -8, -6, 12, -10]

theorem insect_crawl_properties :
  let cumulative_distances := crawl_distances.scanl (· + ·) 0
  (crawl_distances.sum = 0) ∧
  (cumulative_distances.map (Int.natAbs)).maximum? = some 14 ∧
  ((crawl_distances.map Int.natAbs).sum = 54) := by
  sorry

end NUMINAMATH_CALUDE_insect_crawl_properties_l2878_287891


namespace NUMINAMATH_CALUDE_midpoint_trajectory_l2878_287812

/-- The trajectory of the midpoint of a line segment with one end fixed and the other on a circle -/
theorem midpoint_trajectory (x y : ℝ) : 
  (∃ a b : ℝ, (a^2 + b^2 = 16) ∧ 
              (x = (10 + a) / 2) ∧ 
              (y = b / 2)) → 
  (x - 5)^2 + y^2 = 4 := by
sorry

end NUMINAMATH_CALUDE_midpoint_trajectory_l2878_287812


namespace NUMINAMATH_CALUDE_two_card_selection_l2878_287860

theorem two_card_selection (deck_size : ℕ) (h : deck_size = 60) : 
  deck_size * (deck_size - 1) = 3540 :=
by sorry

end NUMINAMATH_CALUDE_two_card_selection_l2878_287860


namespace NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l2878_287863

theorem infinite_geometric_series_first_term 
  (r : ℝ) (S : ℝ) (a : ℝ) 
  (h1 : r = -1/3) 
  (h2 : S = 9) 
  (h3 : S = a / (1 - r)) : a = 12 := by
  sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l2878_287863


namespace NUMINAMATH_CALUDE_product_equals_three_halves_l2878_287862

theorem product_equals_three_halves : 12 * 0.5 * 4 * 0.0625 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_three_halves_l2878_287862


namespace NUMINAMATH_CALUDE_probability_at_least_one_woman_l2878_287875

def total_people : ℕ := 10
def num_men : ℕ := 7
def num_women : ℕ := 3
def selection_size : ℕ := 3

theorem probability_at_least_one_woman :
  let prob_no_women := (num_men.choose selection_size : ℚ) / (total_people.choose selection_size : ℚ)
  (1 : ℚ) - prob_no_women = 17 / 24 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_woman_l2878_287875


namespace NUMINAMATH_CALUDE_cricket_team_average_age_l2878_287865

theorem cricket_team_average_age 
  (n : ℕ) 
  (captain_age : ℕ) 
  (wicket_keeper_age_diff : ℕ) 
  (h1 : n = 11) 
  (h2 : captain_age = 28) 
  (h3 : wicket_keeper_age_diff = 3) : 
  ∃ (team_avg : ℚ), 
    team_avg = 25 ∧ 
    (n : ℚ) * team_avg = 
      (captain_age : ℚ) + 
      ((captain_age : ℚ) + wicket_keeper_age_diff) + 
      ((n - 2 : ℚ) * (team_avg - 1)) := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_average_age_l2878_287865


namespace NUMINAMATH_CALUDE_walter_work_hours_l2878_287806

/-- Walter's work schedule and earnings -/
structure WorkSchedule where
  days_per_week : ℕ
  hourly_rate : ℚ
  allocation_ratio : ℚ
  school_allocation : ℚ

/-- Calculate the daily work hours given a work schedule -/
def daily_work_hours (schedule : WorkSchedule) : ℚ :=
  schedule.school_allocation / (schedule.days_per_week * schedule.hourly_rate * schedule.allocation_ratio)

/-- Theorem: Walter works 4 hours a day -/
theorem walter_work_hours : 
  let walter_schedule : WorkSchedule := {
    days_per_week := 5,
    hourly_rate := 5,
    allocation_ratio := 3/4,
    school_allocation := 75
  }
  daily_work_hours walter_schedule = 4 := by
  sorry

end NUMINAMATH_CALUDE_walter_work_hours_l2878_287806


namespace NUMINAMATH_CALUDE_dream_cost_in_illusions_l2878_287848

/-- Represents the price of an item in the dream market -/
structure DreamPrice where
  illusion : ℚ
  nap : ℚ
  nightmare : ℚ
  dream : ℚ

/-- The dream market pricing system satisfies the given conditions -/
def is_valid_pricing (p : DreamPrice) : Prop :=
  7 * p.illusion + 2 * p.nap + p.nightmare = 4 * p.dream ∧
  4 * p.illusion + 4 * p.nap + 2 * p.nightmare = 7 * p.dream

/-- The cost of one dream is equal to 10 illusions -/
theorem dream_cost_in_illusions (p : DreamPrice) : 
  is_valid_pricing p → p.dream = 10 * p.illusion := by
  sorry

end NUMINAMATH_CALUDE_dream_cost_in_illusions_l2878_287848


namespace NUMINAMATH_CALUDE_polynomial_value_equality_l2878_287840

theorem polynomial_value_equality (x : ℝ) : 
  x^2 - (5/2)*x = 6 → 2*x^2 - 5*x + 6 = 18 := by
sorry

end NUMINAMATH_CALUDE_polynomial_value_equality_l2878_287840


namespace NUMINAMATH_CALUDE_volume_cone_from_right_triangle_l2878_287834

/-- The volume of a cone formed by rotating a right triangle around its hypotenuse -/
theorem volume_cone_from_right_triangle (h : ℝ) (r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let v := (1 / 3) * π * r^2 * h
  h = 2 ∧ r = 1 → v = (2 * π) / 3 := by sorry

end NUMINAMATH_CALUDE_volume_cone_from_right_triangle_l2878_287834


namespace NUMINAMATH_CALUDE_unique_solution_to_equation_l2878_287832

theorem unique_solution_to_equation : 
  ∀ a b : ℝ, 2 * (a^2 + 1) * (b^2 + 1) = (a + 1) * (b + 1) * (a * b + 1) → a = 1 ∧ b = 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_to_equation_l2878_287832


namespace NUMINAMATH_CALUDE_second_number_divisible_by_seven_l2878_287897

theorem second_number_divisible_by_seven (a b c : ℕ+) 
  (ha : a = 105)
  (hc : c = 2436)
  (h_gcd : Nat.gcd a (Nat.gcd b c) = 7) :
  7 ∣ b := by
sorry

end NUMINAMATH_CALUDE_second_number_divisible_by_seven_l2878_287897


namespace NUMINAMATH_CALUDE_books_on_shelves_l2878_287858

theorem books_on_shelves (total : ℕ) (bottom middle top : ℕ) : 
  bottom = (total - bottom) / 2 →
  middle = (total - middle) / 3 →
  top = 30 →
  total = bottom + middle + top →
  total = 72 := by
sorry

end NUMINAMATH_CALUDE_books_on_shelves_l2878_287858


namespace NUMINAMATH_CALUDE_min_value_3a_2b_min_value_3a_2b_achieved_l2878_287842

theorem min_value_3a_2b (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : (a + b)⁻¹ + (a - b)⁻¹ = 1) : 
  ∀ x y : ℝ, x > 0 → y > 0 → (x + y)⁻¹ + (x - y)⁻¹ = 1 → 3*x + 2*y ≥ 3 + Real.sqrt 5 :=
by sorry

theorem min_value_3a_2b_achieved (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : (a + b)⁻¹ + (a - b)⁻¹ = 1) : 
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ (x + y)⁻¹ + (x - y)⁻¹ = 1 ∧ 3*x + 2*y = 3 + Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_3a_2b_min_value_3a_2b_achieved_l2878_287842


namespace NUMINAMATH_CALUDE_median_mode_of_scores_l2878_287818

def scores : List ℕ := [7, 10, 9, 8, 9, 9, 8]

def median (l : List ℕ) : ℚ := sorry

def mode (l : List ℕ) : ℕ := sorry

theorem median_mode_of_scores :
  median scores = 9 ∧ mode scores = 9 := by sorry

end NUMINAMATH_CALUDE_median_mode_of_scores_l2878_287818


namespace NUMINAMATH_CALUDE_investment_proof_l2878_287846

/-- The compound interest function -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

theorem investment_proof :
  let initial_investment : ℝ := 1000
  let interest_rate : ℝ := 0.08
  let time_period : ℕ := 6
  let final_balance : ℝ := 1586.87
  (compound_interest initial_investment interest_rate time_period) = final_balance := by
  sorry

end NUMINAMATH_CALUDE_investment_proof_l2878_287846


namespace NUMINAMATH_CALUDE_rectangle_side_lengths_l2878_287879

theorem rectangle_side_lengths (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a < b) 
  (h4 : a * b = 2 * a + 2 * b) : a < 4 ∧ b > 4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_side_lengths_l2878_287879


namespace NUMINAMATH_CALUDE_integer_count_between_negatives_l2878_287823

theorem integer_count_between_negatives (a : ℚ) : 
  (a > 0) → 
  (∃ n : ℕ, n = (⌊a⌋ - ⌈-a⌉ - 1) ∧ n = 2007) → 
  (1003 < a ∧ a ≤ 1004) :=
by
  sorry

end NUMINAMATH_CALUDE_integer_count_between_negatives_l2878_287823


namespace NUMINAMATH_CALUDE_tripled_base_and_exponent_l2878_287838

theorem tripled_base_and_exponent (a : ℝ) (b : ℤ) (x : ℝ) :
  b ≠ 0 →
  (3 * a) ^ (3 * b) = a ^ b * x ^ (3 * b) →
  x = 3 * a ^ (2/3) :=
by sorry

end NUMINAMATH_CALUDE_tripled_base_and_exponent_l2878_287838


namespace NUMINAMATH_CALUDE_albrecht_equation_solutions_l2878_287827

theorem albrecht_equation_solutions :
  ∀ a b : ℕ+, 
    (a + 2*b - 3)^2 = a^2 + 4*b^2 - 9 ↔ 
    ((a = 2 ∧ b = 15) ∨ (a = 3 ∧ b = 6) ∨ (a = 6 ∧ b = 3) ∨ (a = 15 ∧ b = 2)) :=
by sorry

end NUMINAMATH_CALUDE_albrecht_equation_solutions_l2878_287827


namespace NUMINAMATH_CALUDE_no_solutions_for_equation_l2878_287809

theorem no_solutions_for_equation : ¬∃ (n : ℕ+), (1 + 1 / n.val : ℝ) ^ (n.val + 1) = (1 + 1 / 1998 : ℝ) ^ 1998 := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_for_equation_l2878_287809


namespace NUMINAMATH_CALUDE_hyperbola_sum_l2878_287833

/-- Proves that for a hyperbola with given parameters, the sum of h, k, a, and b equals 6 + 2√10 -/
theorem hyperbola_sum (h k : ℝ) (focus_y vertex_y : ℝ) : 
  h = 1 → 
  k = 2 → 
  focus_y = 9 → 
  vertex_y = -1 → 
  let a := |k - vertex_y|
  let c := |k - focus_y|
  let b := Real.sqrt (c^2 - a^2)
  h + k + a + b = 6 + 2 * Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_sum_l2878_287833


namespace NUMINAMATH_CALUDE_fourth_term_is_27_l2878_287896

-- Define the sequence sum function
def S (n : ℕ) : ℤ := 4 * n^2 - n - 8

-- Define the sequence term function
def a (n : ℕ) : ℤ := S n - S (n - 1)

-- Theorem statement
theorem fourth_term_is_27 : a 4 = 27 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_is_27_l2878_287896


namespace NUMINAMATH_CALUDE_max_value_of_operation_max_value_is_attainable_l2878_287836

theorem max_value_of_operation (n : ℕ) : 
  (10 ≤ n ∧ n ≤ 99) → 3 * (300 - 2 * n) ≤ 840 := by
  sorry

theorem max_value_is_attainable : 
  ∃ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ 3 * (300 - 2 * n) = 840 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_operation_max_value_is_attainable_l2878_287836


namespace NUMINAMATH_CALUDE_problem_1_l2878_287821

theorem problem_1 (a b : ℝ) : a^2 + b^2 - 2*a + 1 = 0 → a = 1 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l2878_287821


namespace NUMINAMATH_CALUDE_expression_value_l2878_287813

theorem expression_value (b : ℚ) (h : b = 1/3) : 
  (3 * b⁻¹ + b⁻¹ / 3) / b = 30 := by sorry

end NUMINAMATH_CALUDE_expression_value_l2878_287813


namespace NUMINAMATH_CALUDE_replaced_person_weight_l2878_287892

/-- Proves that the weight of a replaced person is 65 kg given the conditions of the problem -/
theorem replaced_person_weight (n : ℕ) (old_avg new_avg new_weight : ℝ) : 
  n = 8 ∧ 
  new_avg - old_avg = 3.5 ∧
  new_weight = 93 →
  (n * new_avg - new_weight) / (n - 1) = 65 :=
by sorry

end NUMINAMATH_CALUDE_replaced_person_weight_l2878_287892


namespace NUMINAMATH_CALUDE_pairwise_product_inequality_l2878_287881

theorem pairwise_product_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x * y / z) + (y * z / x) + (z * x / y) > 2 * Real.rpow (x^3 + y^3 + z^3) (1/3) := by
  sorry

end NUMINAMATH_CALUDE_pairwise_product_inequality_l2878_287881


namespace NUMINAMATH_CALUDE_rational_condition_l2878_287835

theorem rational_condition (x : ℝ) : 
  ∃ (q : ℚ), x + Real.sqrt (x^2 + 1) - 1 / (x + Real.sqrt (x^2 + 1)) = ↑q ↔ ∃ (r : ℚ), x = ↑r :=
by sorry

end NUMINAMATH_CALUDE_rational_condition_l2878_287835


namespace NUMINAMATH_CALUDE_function_satisfies_conditions_l2878_287817

/-- The function we're considering -/
def f (x : ℝ) : ℝ := x^2 - 3*x + 5

/-- The derivative of our function -/
def f' (x : ℝ) : ℝ := 2*x - 3

theorem function_satisfies_conditions : 
  (f 1 = 3) ∧ (∀ x, (deriv f) x = f' x) := by sorry

end NUMINAMATH_CALUDE_function_satisfies_conditions_l2878_287817


namespace NUMINAMATH_CALUDE_zero_exponent_l2878_287820

theorem zero_exponent (x : ℝ) (h : x ≠ 0) : x^0 = 1 := by sorry

end NUMINAMATH_CALUDE_zero_exponent_l2878_287820


namespace NUMINAMATH_CALUDE_polynomial_remainder_l2878_287880

theorem polynomial_remainder (x : ℝ) : 
  (x^4 + x^3 + 1) % (x - 2) = 25 := by
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l2878_287880


namespace NUMINAMATH_CALUDE_M_intersect_N_equals_open_interval_l2878_287831

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x, y = 2^x}
def N : Set ℝ := {x | ∃ y, y = Real.log (2*x - x^2)}

-- State the theorem
theorem M_intersect_N_equals_open_interval : M ∩ N = Set.Ioo 0 2 := by
  sorry

end NUMINAMATH_CALUDE_M_intersect_N_equals_open_interval_l2878_287831


namespace NUMINAMATH_CALUDE_football_original_price_l2878_287816

theorem football_original_price : 
  ∀ (original_price : ℝ), 
  (original_price * 0.8 + 25 = original_price) → 
  original_price = 125 := by
sorry

end NUMINAMATH_CALUDE_football_original_price_l2878_287816


namespace NUMINAMATH_CALUDE_cattle_area_calculation_l2878_287826

def farm_length : ℝ := 3.6

theorem cattle_area_calculation (width : ℝ) (total_area : ℝ) (cattle_area : ℝ)
  (h1 : width = 2.5 * farm_length)
  (h2 : total_area = farm_length * width)
  (h3 : cattle_area = total_area / 2) :
  cattle_area = 16.2 := by
  sorry

end NUMINAMATH_CALUDE_cattle_area_calculation_l2878_287826


namespace NUMINAMATH_CALUDE_sum_of_max_min_is_zero_l2878_287849

def f (x : ℝ) := x^2 - 2*x - 1

theorem sum_of_max_min_is_zero :
  let a := 0
  let b := 3
  (∃ (x_max x_min : ℝ), a ≤ x_max ∧ x_max ≤ b ∧ a ≤ x_min ∧ x_min ≤ b ∧
    (∀ x, a ≤ x ∧ x ≤ b → f x ≤ f x_max) ∧
    (∀ x, a ≤ x ∧ x ≤ b → f x_min ≤ f x) ∧
    f x_max + f x_min = 0) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_max_min_is_zero_l2878_287849


namespace NUMINAMATH_CALUDE_nested_diamond_result_l2878_287852

/-- Diamond operation for real numbers -/
def diamond (a b : ℝ) : ℝ := a^2 + b^2 - a * b

/-- Theorem stating the result of the nested diamond operations -/
theorem nested_diamond_result :
  diamond (diamond 3 8) (diamond 8 (-3)) = 7057 := by
  sorry

end NUMINAMATH_CALUDE_nested_diamond_result_l2878_287852


namespace NUMINAMATH_CALUDE_find_y_l2878_287819

theorem find_y (x : ℝ) (y : ℝ) (h1 : 1.5 * x = 0.3 * y) (h2 : x = 20) : y = 100 := by
  sorry

end NUMINAMATH_CALUDE_find_y_l2878_287819


namespace NUMINAMATH_CALUDE_five_roots_sum_l2878_287890

noncomputable def f (x : ℝ) : ℝ :=
  if x = 2 then 1 else Real.log (abs (x - 2))

theorem five_roots_sum (b c : ℝ) 
  (h : ∃ x₁ x₂ x₃ x₄ x₅ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₁ ≠ x₅ ∧ 
                           x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₂ ≠ x₅ ∧
                           x₃ ≠ x₄ ∧ x₃ ≠ x₅ ∧ x₄ ≠ x₅ ∧
                           (f x₁)^2 + b * (f x₁) + c = 0 ∧
                           (f x₂)^2 + b * (f x₂) + c = 0 ∧
                           (f x₃)^2 + b * (f x₃) + c = 0 ∧
                           (f x₄)^2 + b * (f x₄) + c = 0 ∧
                           (f x₅)^2 + b * (f x₅) + c = 0) :
  ∃ x₁ x₂ x₃ x₄ x₅ : ℝ, f (x₁ + x₂ + x₃ + x₄ + x₅) = 3 * Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_five_roots_sum_l2878_287890


namespace NUMINAMATH_CALUDE_parabola_equation_l2878_287825

/-- A parabola with vertex at the origin, opening upward -/
structure Parabola where
  focus : ℝ × ℝ
  directrix : ℝ → ℝ
  pointOnParabola : ℝ × ℝ

/-- The parabola satisfies the given conditions -/
def satisfiesConditions (p : Parabola) : Prop :=
  let (xF, yF) := p.focus
  let (xA, yA) := p.pointOnParabola
  let yM := p.directrix 0
  yF > 0 ∧ 
  Real.sqrt ((xA - 0)^2 + (yA - yM)^2) = Real.sqrt 17 ∧
  Real.sqrt ((xA - xF)^2 + (yA - yF)^2) = 3

/-- The equation of the parabola is x² = 12y -/
def hasEquation (p : Parabola) : Prop :=
  let (x, y) := p.pointOnParabola
  x^2 = 12 * y

theorem parabola_equation (p : Parabola) 
  (h : satisfiesConditions p) : hasEquation p := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l2878_287825


namespace NUMINAMATH_CALUDE_pirate_treasure_distribution_l2878_287843

theorem pirate_treasure_distribution (x : ℕ) : 
  (x * (x + 1)) / 2 = 4 * x → x + 4 * x = 35 := by
  sorry

end NUMINAMATH_CALUDE_pirate_treasure_distribution_l2878_287843


namespace NUMINAMATH_CALUDE_article_cost_price_l2878_287878

/-- Given an article with marked price M and cost price C,
    prove that if 0.95M = 1.25C = 75, then C = 60. -/
theorem article_cost_price (M C : ℝ) (h : 0.95 * M = 1.25 * C ∧ 0.95 * M = 75) : C = 60 := by
  sorry

end NUMINAMATH_CALUDE_article_cost_price_l2878_287878


namespace NUMINAMATH_CALUDE_jim_victory_percentage_l2878_287869

def total_votes : ℕ := 6000
def geoff_percent : ℚ := 1/200

theorem jim_victory_percentage (laura_votes geoff_votes jim_votes : ℕ) :
  geoff_votes = (geoff_percent * total_votes).num ∧
  laura_votes = 2 * geoff_votes ∧
  jim_votes = total_votes - (laura_votes + geoff_votes) ∧
  geoff_votes + 3000 > laura_votes ∧
  geoff_votes + 3000 > jim_votes →
  (jim_votes : ℚ) / total_votes ≥ 5052 / 10000 :=
by sorry

end NUMINAMATH_CALUDE_jim_victory_percentage_l2878_287869


namespace NUMINAMATH_CALUDE_horse_rider_ratio_l2878_287889

theorem horse_rider_ratio (total_horses : ℕ) (total_legs_walking : ℕ) 
  (h1 : total_horses = 10)
  (h2 : total_legs_walking = 50) :
  (total_horses - (total_legs_walking / 6)) / total_horses = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_horse_rider_ratio_l2878_287889


namespace NUMINAMATH_CALUDE_fraction_equality_l2878_287861

theorem fraction_equality : (4 + 14) / (7 + 14) = 6 / 7 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l2878_287861


namespace NUMINAMATH_CALUDE_train_stop_time_l2878_287828

/-- Calculates the time a train stops per hour given its speeds with and without stoppages -/
theorem train_stop_time (speed_without_stoppages speed_with_stoppages : ℝ) : 
  speed_without_stoppages = 45 →
  speed_with_stoppages = 42 →
  (speed_without_stoppages - speed_with_stoppages) / speed_without_stoppages * 60 = 4 := by
  sorry

#check train_stop_time

end NUMINAMATH_CALUDE_train_stop_time_l2878_287828


namespace NUMINAMATH_CALUDE_original_group_size_l2878_287847

theorem original_group_size (initial_avg : ℝ) (new_boy1 new_boy2 new_boy3 : ℝ) (new_avg : ℝ) :
  initial_avg = 35 →
  new_boy1 = 40 →
  new_boy2 = 45 →
  new_boy3 = 50 →
  new_avg = 36 →
  ∃ n : ℕ,
    n * initial_avg + new_boy1 + new_boy2 + new_boy3 = (n + 3) * new_avg ∧
    n = 27 :=
by sorry

end NUMINAMATH_CALUDE_original_group_size_l2878_287847


namespace NUMINAMATH_CALUDE_opposite_of_negative_one_third_l2878_287895

theorem opposite_of_negative_one_third :
  -(-(1/3 : ℚ)) = 1/3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_one_third_l2878_287895


namespace NUMINAMATH_CALUDE_cone_base_radius_l2878_287894

/-- Represents a cone with given properties -/
structure Cone where
  surface_area : ℝ
  lateral_unfolds_semicircle : Prop

/-- Theorem: For a cone with surface area 12π and lateral surface that unfolds into a semicircle, 
    the radius of the base is 2 -/
theorem cone_base_radius 
  (cone : Cone) 
  (h1 : cone.surface_area = 12 * Real.pi) 
  (h2 : cone.lateral_unfolds_semicircle) : 
  ∃ (r : ℝ), r = 2 ∧ r > 0 ∧ 
  cone.surface_area = Real.pi * r^2 + Real.pi * r * (2 * r) := by
  sorry


end NUMINAMATH_CALUDE_cone_base_radius_l2878_287894


namespace NUMINAMATH_CALUDE_curve_equation_and_cosine_value_l2878_287859

-- Define the curves C₁ and C₂
def C₁ (ρ θ : ℝ) : Prop := ρ^2 * (3 + Real.sin θ^2) = 12
def C₂ (x y t α : ℝ) : Prop := x = 1 + t * Real.cos α ∧ y = t * Real.sin α

-- Define the condition for α
def α_condition (α : ℝ) : Prop := 0 < α ∧ α < Real.pi / 2

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) (α : ℝ) : Prop :=
  ∃ (t₁ t₂ : ℝ), C₂ A.1 A.2 t₁ α ∧ C₂ B.1 B.2 t₂ α ∧
  (A.1^2 / 4 + A.2^2 / 3 = 1) ∧ (B.1^2 / 4 + B.2^2 / 3 = 1)

-- Define the distance condition
def distance_condition (A B P : ℝ × ℝ) : Prop :=
  Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2) +
  Real.sqrt ((B.1 - P.1)^2 + (B.2 - P.2)^2) = 7/2

theorem curve_equation_and_cosine_value
  (α : ℝ) (A B P : ℝ × ℝ)
  (h_α : α_condition α)
  (h_int : intersection_points A B α)
  (h_dist : distance_condition A B P) :
  (∀ (x y : ℝ), (x^2 / 4 + y^2 / 3 = 1) ↔ (∃ (ρ θ : ℝ), C₁ ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ)) ∧
  Real.cos α = 2 * Real.sqrt 7 / 7 :=
sorry

end NUMINAMATH_CALUDE_curve_equation_and_cosine_value_l2878_287859


namespace NUMINAMATH_CALUDE_remainder_1234567891_div_98_l2878_287822

theorem remainder_1234567891_div_98 : 1234567891 % 98 = 23 := by
  sorry

end NUMINAMATH_CALUDE_remainder_1234567891_div_98_l2878_287822


namespace NUMINAMATH_CALUDE_tank_fill_time_with_leak_l2878_287871

/-- Given a tank and two processes:
    1. Pipe A that can fill the tank in 6 hours
    2. A leak that can empty the full tank in 15 hours
    This theorem proves that it takes 10 hours for Pipe A to fill the tank with the leak present. -/
theorem tank_fill_time_with_leak (tank : ℝ) (pipe_a_rate : ℝ) (leak_rate : ℝ) : 
  pipe_a_rate = 1 / 6 →
  leak_rate = 1 / 15 →
  (pipe_a_rate - leak_rate)⁻¹ = 10 := by
  sorry

end NUMINAMATH_CALUDE_tank_fill_time_with_leak_l2878_287871


namespace NUMINAMATH_CALUDE_cone_apex_angle_l2878_287811

theorem cone_apex_angle (α β : ℝ) : 
  β = Real.arcsin (1/4) →
  2 * α = Real.arcsin (2 * Real.sin β) + β →
  2 * α = π/6 + Real.arcsin (1/4) :=
by sorry

end NUMINAMATH_CALUDE_cone_apex_angle_l2878_287811


namespace NUMINAMATH_CALUDE_value_of_2x_plus_y_l2878_287803

theorem value_of_2x_plus_y (x y : ℚ) 
  (h : x^2 - 2*y - Real.sqrt 2*y = 17 - 4*Real.sqrt 2) :
  2*x + y = 14 ∨ 2*x + y = -6 := by
sorry

end NUMINAMATH_CALUDE_value_of_2x_plus_y_l2878_287803


namespace NUMINAMATH_CALUDE_base_seventeen_distinct_digits_l2878_287856

/-- The number of three-digit numbers with distinct digits in base b -/
def distinctThreeDigitNumbers (b : ℕ) : ℕ := (b - 1) * (b - 1) * (b - 2)

/-- Theorem stating that there are exactly 256 three-digit numbers with distinct digits in base 17 -/
theorem base_seventeen_distinct_digits : 
  ∃ (b : ℕ), b > 2 ∧ distinctThreeDigitNumbers b = 256 ↔ b = 17 := by sorry

end NUMINAMATH_CALUDE_base_seventeen_distinct_digits_l2878_287856


namespace NUMINAMATH_CALUDE_sum_inequality_l2878_287853

theorem sum_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 2/y = 2) :
  8*x + y ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l2878_287853


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l2878_287841

theorem necessary_not_sufficient_condition (a : ℝ) :
  (∀ a, a^2 > 2*a → (a > 2 ∨ a < 0)) ∧
  (∃ a, a > 2 ∧ a^2 ≤ 2*a) :=
sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l2878_287841


namespace NUMINAMATH_CALUDE_distance_between_points_on_line_l2878_287807

/-- Given a line with equation 2x - 3y + 6 = 0 and two points (p, q) and (r, s) on this line,
    the distance between these points is (√13/3)|r - p| -/
theorem distance_between_points_on_line (p r : ℝ) :
  let q := (2*p + 6)/3
  let s := (2*r + 6)/3
  (2*p - 3*q + 6 = 0) →
  (2*r - 3*s + 6 = 0) →
  Real.sqrt ((r - p)^2 + (s - q)^2) = (Real.sqrt 13 / 3) * |r - p| := by
sorry

end NUMINAMATH_CALUDE_distance_between_points_on_line_l2878_287807
