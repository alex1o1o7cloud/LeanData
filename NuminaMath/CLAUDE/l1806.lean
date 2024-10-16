import Mathlib

namespace NUMINAMATH_CALUDE_correct_page_difference_l1806_180624

/-- Calculates the difference in pages read between yesterday and today -/
def pagesDifference (totalPages yesterday tomorrow : ℕ) : ℕ :=
  yesterday - (totalPages - yesterday - tomorrow)

theorem correct_page_difference :
  pagesDifference 100 35 35 = 5 := by
  sorry

end NUMINAMATH_CALUDE_correct_page_difference_l1806_180624


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1806_180637

theorem polynomial_simplification (x : ℝ) :
  3 - 5*x - 7*x^2 + 9 + 11*x - 13*x^2 + 15 - 17*x + 19*x^2 + 2*x^3 = 2*x^3 - x^2 - 11*x + 27 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1806_180637


namespace NUMINAMATH_CALUDE_special_gp_ratio_equation_ratio_approx_value_l1806_180679

/-- A geometric progression with positive terms where any term is equal to the sum of the next three following terms -/
structure SpecialGeometricProgression where
  a : ℝ  -- first term
  r : ℝ  -- common ratio
  a_pos : a > 0
  r_pos : r > 0
  sum_property : ∀ n : ℕ, a * r^n = a * r^(n+1) + a * r^(n+2) + a * r^(n+3)

/-- The common ratio of a special geometric progression satisfies a specific equation -/
theorem special_gp_ratio_equation (gp : SpecialGeometricProgression) :
  gp.r^3 + gp.r^2 + gp.r - 1 = 0 :=
sorry

/-- The positive real solution to the equation x^3 + x^2 + x - 1 = 0 is approximately 0.543689 -/
theorem ratio_approx_value :
  ∃ x : ℝ, x > 0 ∧ x^3 + x^2 + x - 1 = 0 ∧ abs (x - 0.543689) < 0.000001 :=
sorry

end NUMINAMATH_CALUDE_special_gp_ratio_equation_ratio_approx_value_l1806_180679


namespace NUMINAMATH_CALUDE_sum_of_four_numbers_l1806_180659

theorem sum_of_four_numbers (a b c d : ℤ) 
  (sum_abc : a + b + c = 415)
  (sum_abd : a + b + d = 442)
  (sum_acd : a + c + d = 396)
  (sum_bcd : b + c + d = 325) :
  a + b + c + d = 526 := by
sorry

end NUMINAMATH_CALUDE_sum_of_four_numbers_l1806_180659


namespace NUMINAMATH_CALUDE_light_bulbs_configuration_equals_59_l1806_180670

/-- Converts a list of binary digits to its decimal representation -/
def binary_to_decimal (binary : List Bool) : Nat :=
  binary.enum.foldl (fun acc (i, b) => acc + (if b then 2^i else 0)) 0

/-- The binary representation of the light bulb configuration -/
def light_bulbs : List Bool := [true, true, true, false, true, true]

theorem light_bulbs_configuration_equals_59 :
  binary_to_decimal light_bulbs = 59 := by
  sorry

end NUMINAMATH_CALUDE_light_bulbs_configuration_equals_59_l1806_180670


namespace NUMINAMATH_CALUDE_perfect_square_sum_l1806_180619

theorem perfect_square_sum (a b : ℤ) 
  (h : ∀ (m n : ℕ), ∃ (k : ℕ), a * m^2 + b * n^2 = k^2) : 
  a * b = 0 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_sum_l1806_180619


namespace NUMINAMATH_CALUDE_smaller_rss_better_fit_regression_line_passes_through_center_l1806_180689

/-- Represents a linear regression model -/
structure LinearRegression where
  x : List ℝ  -- Independent variable data
  y : List ℝ  -- Dependent variable data
  β : ℝ       -- Slope of the regression line
  α : ℝ       -- Intercept of the regression line

/-- Calculates the residual sum of squares for a linear regression model -/
def residualSumOfSquares (model : LinearRegression) : ℝ :=
  sorry

/-- Calculates the mean of a list of real numbers -/
def mean (data : List ℝ) : ℝ :=
  sorry

/-- Theorem stating that a smaller residual sum of squares indicates a better fitting effect -/
theorem smaller_rss_better_fit (model1 model2 : LinearRegression) :
  residualSumOfSquares model1 < residualSumOfSquares model2 →
  -- The fitting effect of model1 is better than model2
  sorry :=
sorry

/-- Theorem stating that the linear regression equation passes through the center point (x̄, ȳ) of the sample -/
theorem regression_line_passes_through_center (model : LinearRegression) :
  let x_mean := mean model.x
  let y_mean := mean model.y
  model.α + model.β * x_mean = y_mean :=
sorry

end NUMINAMATH_CALUDE_smaller_rss_better_fit_regression_line_passes_through_center_l1806_180689


namespace NUMINAMATH_CALUDE_gcd_of_quadratic_and_linear_l1806_180623

theorem gcd_of_quadratic_and_linear (a : ℤ) (h : ∃ k : ℤ, a = 2100 * k) :
  Int.gcd (a^2 + 14*a + 49) (a + 7) = 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_quadratic_and_linear_l1806_180623


namespace NUMINAMATH_CALUDE_equation_and_expression_proof_l1806_180606

theorem equation_and_expression_proof :
  (∃ x1 x2 : ℝ, x1 = 2 + Real.sqrt 7 ∧ x2 = 2 - Real.sqrt 7 ∧
    x1^2 - 4*x1 - 3 = 0 ∧ x2^2 - 4*x2 - 3 = 0) ∧
  ((-1)^2 + 2 * Real.sin (π/3) - Real.tan (π/4) = Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_and_expression_proof_l1806_180606


namespace NUMINAMATH_CALUDE_sin_plus_cos_value_l1806_180629

theorem sin_plus_cos_value (θ : Real) 
  (h1 : 0 < θ) (h2 : θ < π) 
  (h3 : Real.tan (θ + π/4) = 1/7) : 
  Real.sin θ + Real.cos θ = -1/5 := by
sorry

end NUMINAMATH_CALUDE_sin_plus_cos_value_l1806_180629


namespace NUMINAMATH_CALUDE_quadratic_form_ratio_l1806_180609

theorem quadratic_form_ratio (x : ℝ) : 
  ∃ (d e : ℝ), x^2 + 2023*x + 2023 = (x + d)^2 + e ∧ e/d = -1009.75 := by
sorry

end NUMINAMATH_CALUDE_quadratic_form_ratio_l1806_180609


namespace NUMINAMATH_CALUDE_cubic_function_extremum_l1806_180618

/-- Given a cubic function f(x) = x³ + ax² + bx + a², 
    if f has an extremum at x = 1 and f(1) = 10, then a + b = -7 -/
theorem cubic_function_extremum (a b : ℝ) : 
  let f := fun (x : ℝ) => x^3 + a*x^2 + b*x + a^2
  (∃ (ε : ℝ), ε > 0 ∧ ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≤ f 1 ∨ f x ≥ f 1) →
  f 1 = 10 →
  a + b = -7 := by
sorry

end NUMINAMATH_CALUDE_cubic_function_extremum_l1806_180618


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1806_180602

/-- A positive geometric sequence -/
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0 ∧ ∃ r > 0, ∀ k, a (k + 1) = r * a k

/-- The theorem statement -/
theorem geometric_sequence_problem (a : ℕ → ℝ) (n : ℕ) 
  (h_seq : is_positive_geometric_sequence a)
  (h_1 : a 1 * a 2 * a 3 = 4)
  (h_2 : a 4 * a 5 * a 6 = 12)
  (h_3 : a (n-1) * a n * a (n+1) = 324) :
  n = 14 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1806_180602


namespace NUMINAMATH_CALUDE_find_n_l1806_180650

theorem find_n : ∃ n : ℕ, 2^n = 2 * 16^2 * 4^3 ∧ n = 15 := by sorry

end NUMINAMATH_CALUDE_find_n_l1806_180650


namespace NUMINAMATH_CALUDE_jason_grew_37_watermelons_l1806_180683

/-- The number of watermelons Jason grew -/
def jason_watermelons : ℕ := 37

/-- The number of watermelons Sandy grew -/
def sandy_watermelons : ℕ := 11

/-- The total number of watermelons grown by Jason and Sandy -/
def total_watermelons : ℕ := 48

/-- Theorem stating that Jason grew 37 watermelons -/
theorem jason_grew_37_watermelons : jason_watermelons = total_watermelons - sandy_watermelons := by
  sorry

end NUMINAMATH_CALUDE_jason_grew_37_watermelons_l1806_180683


namespace NUMINAMATH_CALUDE_babies_age_sum_l1806_180642

def lioness_age : ℕ := 12

theorem babies_age_sum (hyena_age : ℕ) (lioness_baby_age : ℕ) (hyena_baby_age : ℕ)
  (h1 : lioness_age = 2 * hyena_age)
  (h2 : lioness_baby_age = lioness_age / 2)
  (h3 : hyena_baby_age = hyena_age / 2) :
  lioness_baby_age + 5 + hyena_baby_age + 5 = 19 := by
  sorry

end NUMINAMATH_CALUDE_babies_age_sum_l1806_180642


namespace NUMINAMATH_CALUDE_triangle_angles_l1806_180678

/-- Given a triangle with sides 5, 5, and √17 - √5, prove that its angles are θ, φ, φ, where
    θ = arccos((14 + √85) / 25) and φ = (180° - θ) / 2 -/
theorem triangle_angles (a b c : ℝ) (θ φ : ℝ) : 
  a = 5 → b = 5 → c = Real.sqrt 17 - Real.sqrt 5 →
  θ = Real.arccos ((14 + Real.sqrt 85) / 25) →
  φ = (π - θ) / 2 →
  ∃ (α β γ : ℝ), 
    (α = θ ∧ β = φ ∧ γ = φ) ∧
    (α + β + γ = π) ∧
    (Real.cos α = (b^2 + c^2 - a^2) / (2 * b * c)) ∧
    (Real.cos β = (a^2 + c^2 - b^2) / (2 * a * c)) ∧
    (Real.cos γ = (a^2 + b^2 - c^2) / (2 * a * b)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_angles_l1806_180678


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l1806_180639

theorem hyperbola_asymptote_slope (m : ℝ) : m > 0 →
  (∀ x y : ℝ, x^2/16 - y^2/25 = 1 ↔ (y = m*x ∨ y = -m*x)) →
  m = 5/4 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l1806_180639


namespace NUMINAMATH_CALUDE_model_a_better_fit_l1806_180676

/-- Represents a regression model --/
structure RegressionModel where
  rsquare : ℝ
  (rsquare_nonneg : 0 ≤ rsquare)
  (rsquare_le_one : rsquare ≤ 1)

/-- Defines when one model has a better fit than another --/
def better_fit (model1 model2 : RegressionModel) : Prop :=
  model1.rsquare > model2.rsquare

/-- Theorem stating that model A has a better fit than model B --/
theorem model_a_better_fit (model_a model_b : RegressionModel)
  (ha : model_a.rsquare = 0.98)
  (hb : model_b.rsquare = 0.80) :
  better_fit model_a model_b :=
sorry

end NUMINAMATH_CALUDE_model_a_better_fit_l1806_180676


namespace NUMINAMATH_CALUDE_dvd_packs_total_cost_l1806_180622

/-- Calculates the total cost of purchasing two packs of DVDs with given prices, discounts, and an additional discount for buying both. -/
def total_cost (price1 price2 discount1 discount2 additional_discount : ℕ) : ℕ :=
  (price1 - discount1) + (price2 - discount2) - additional_discount

/-- Theorem stating that the total cost of purchasing the two DVD packs is 111 dollars. -/
theorem dvd_packs_total_cost : 
  total_cost 76 85 25 15 10 = 111 := by
  sorry

end NUMINAMATH_CALUDE_dvd_packs_total_cost_l1806_180622


namespace NUMINAMATH_CALUDE_coeff_x4_is_zero_l1806_180674

/-- The coefficient of x^4 in the expansion of (x+2)(x-1)^5 -/
def coeff_x4 (x : ℝ) : ℝ :=
  let expansion := (x + 2) * (x - 1)^5
  sorry

theorem coeff_x4_is_zero :
  coeff_x4 x = 0 := by sorry

end NUMINAMATH_CALUDE_coeff_x4_is_zero_l1806_180674


namespace NUMINAMATH_CALUDE_infinite_x₀_finite_values_l1806_180625

/-- The function f(x) = 3x - x^2 -/
def f (x : ℝ) : ℝ := 3 * x - x^2

/-- The sequence x_n defined by x_n = f(x_{n-1}) -/
def seq (x₀ : ℝ) : ℕ → ℝ
  | 0 => x₀
  | n + 1 => f (seq x₀ n)

/-- A set is finite if it is empty or there exists a bijection with a finite segment of ℕ -/
def IsFiniteSet (S : Set ℝ) : Prop :=
  S = ∅ ∨ ∃ n : ℕ, ∃ h : Fin n → S, Function.Bijective h

/-- The set of values in the sequence starting from x₀ -/
def seqValues (x₀ : ℝ) : Set ℝ :=
  { x | ∃ n : ℕ, seq x₀ n = x }

/-- The theorem stating that infinitely many x₀ in [0, 3] lead to finite value sets -/
theorem infinite_x₀_finite_values :
  ∃ S : Set ℝ, S ⊆ Set.Icc 0 3 ∧ Set.Infinite S ∧ ∀ x₀ ∈ S, IsFiniteSet (seqValues x₀) := by
  sorry


end NUMINAMATH_CALUDE_infinite_x₀_finite_values_l1806_180625


namespace NUMINAMATH_CALUDE_lcm_product_hcf_l1806_180681

theorem lcm_product_hcf (x y : ℕ+) : 
  Nat.lcm x y = 560 → x * y = 42000 → Nat.gcd x y = 75 := by
  sorry

end NUMINAMATH_CALUDE_lcm_product_hcf_l1806_180681


namespace NUMINAMATH_CALUDE_lcm_of_4_9_10_27_l1806_180638

theorem lcm_of_4_9_10_27 : Nat.lcm 4 (Nat.lcm 9 (Nat.lcm 10 27)) = 540 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_4_9_10_27_l1806_180638


namespace NUMINAMATH_CALUDE_expand_and_simplify_l1806_180615

theorem expand_and_simplify (x : ℝ) : (x + 3) * (4 * x - 8) + x^2 = 5 * x^2 + 4 * x - 24 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l1806_180615


namespace NUMINAMATH_CALUDE_sum_quadratic_distinct_roots_l1806_180653

/-- A quadratic function f(x) = x^2 + ax + b -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ

/-- The discriminant of a quadratic function -/
def discriminant (f : QuadraticFunction) : ℝ := f.a^2 - 4*f.b

/-- The sum of two quadratic functions -/
def sum_quadratic (f g : QuadraticFunction) : QuadraticFunction :=
  ⟨f.a + g.a, f.b + g.b⟩

/-- The sum of a list of quadratic functions -/
def sum_quadratic_list (fs : List QuadraticFunction) : QuadraticFunction :=
  fs.foldl sum_quadratic ⟨0, 0⟩

/-- Theorem: Given conditions on quadratic functions, their sum has distinct real roots -/
theorem sum_quadratic_distinct_roots
  (n : ℕ)
  (hn : n ≥ 2)
  (fs : List QuadraticFunction)
  (hfs : fs.length = n)
  (h_same_discriminant : ∀ (f g : QuadraticFunction), f ∈ fs → g ∈ fs → discriminant f = discriminant g)
  (h_distinct_roots : ∀ (f g : QuadraticFunction), f ∈ fs → g ∈ fs → f ≠ g →
    (discriminant (sum_quadratic f g) > 0)) :
  discriminant (sum_quadratic_list fs) > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_quadratic_distinct_roots_l1806_180653


namespace NUMINAMATH_CALUDE_average_height_combined_groups_l1806_180646

theorem average_height_combined_groups
  (group1_count : ℕ)
  (group2_count : ℕ)
  (total_count : ℕ)
  (average_height : ℝ)
  (h1 : group1_count = 20)
  (h2 : group2_count = 11)
  (h3 : total_count = group1_count + group2_count)
  (h4 : average_height = 20) :
  (group1_count * average_height + group2_count * average_height) / total_count = average_height :=
by sorry

end NUMINAMATH_CALUDE_average_height_combined_groups_l1806_180646


namespace NUMINAMATH_CALUDE_power_of_two_plus_one_l1806_180604

theorem power_of_two_plus_one (a : ℤ) (b : ℝ) (h : 2^a = b) : 2^(a+1) = 2*b := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_plus_one_l1806_180604


namespace NUMINAMATH_CALUDE_second_project_breadth_l1806_180611

/-- Represents a digging project with depth, length, and breadth dimensions -/
structure DiggingProject where
  depth : ℝ
  length : ℝ
  breadth : ℝ

/-- Calculates the volume of a digging project -/
def volume (project : DiggingProject) : ℝ :=
  project.depth * project.length * project.breadth

/-- Theorem: Given two digging projects with equal volumes, 
    the breadth of the second project is 50 meters -/
theorem second_project_breadth
  (project1 : DiggingProject)
  (project2 : DiggingProject)
  (h1 : project1.depth = 100)
  (h2 : project1.length = 25)
  (h3 : project1.breadth = 30)
  (h4 : project2.depth = 75)
  (h5 : project2.length = 20)
  (h_equal_volume : volume project1 = volume project2) :
  project2.breadth = 50 := by
  sorry

#check second_project_breadth

end NUMINAMATH_CALUDE_second_project_breadth_l1806_180611


namespace NUMINAMATH_CALUDE_smallest_n_boxes_n_equals_nine_smallest_n_is_nine_l1806_180645

theorem smallest_n_boxes (n : ℕ) : 
  (∃ k : ℕ, 15 * n - 3 = 11 * k) → n ≥ 9 :=
by sorry

theorem n_equals_nine : 
  ∃ k : ℕ, 15 * 9 - 3 = 11 * k :=
by sorry

theorem smallest_n_is_nine : 
  (∀ m : ℕ, m < 9 → ¬∃ k : ℕ, 15 * m - 3 = 11 * k) ∧
  (∃ k : ℕ, 15 * 9 - 3 = 11 * k) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_boxes_n_equals_nine_smallest_n_is_nine_l1806_180645


namespace NUMINAMATH_CALUDE_jellybean_ratio_l1806_180687

/-- Prove that the ratio of Matilda's jellybeans to Matt's jellybeans is 1:2 -/
theorem jellybean_ratio :
  let steve_jellybeans : ℕ := 84
  let matt_jellybeans : ℕ := 10 * steve_jellybeans
  let matilda_jellybeans : ℕ := 420
  (matilda_jellybeans : ℚ) / (matt_jellybeans : ℚ) = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_jellybean_ratio_l1806_180687


namespace NUMINAMATH_CALUDE_potato_slab_length_l1806_180684

/-- The length of the original uncut potato slab given the lengths of its two pieces -/
theorem potato_slab_length 
  (piece1 : ℕ) 
  (piece2 : ℕ) 
  (h1 : piece1 = 275)
  (h2 : piece2 = piece1 + 50) : 
  piece1 + piece2 = 600 := by
  sorry

end NUMINAMATH_CALUDE_potato_slab_length_l1806_180684


namespace NUMINAMATH_CALUDE_rhombus_area_l1806_180691

/-- The area of a rhombus with longer diagonal 30 units and angle 60° between diagonals is 225√3 square units -/
theorem rhombus_area (d₁ : ℝ) (θ : ℝ) (h₁ : d₁ = 30) (h₂ : θ = Real.pi / 3) :
  let d₂ := d₁ * Real.sin θ
  d₁ * d₂ / 2 = 225 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l1806_180691


namespace NUMINAMATH_CALUDE_toms_deck_cost_l1806_180658

/-- Calculates the total cost of a deck of cards given the number of cards of each type and their respective costs. -/
def deck_cost (rare_count : ℕ) (uncommon_count : ℕ) (common_count : ℕ) 
              (rare_cost : ℚ) (uncommon_cost : ℚ) (common_cost : ℚ) : ℚ :=
  rare_count * rare_cost + uncommon_count * uncommon_cost + common_count * common_cost

/-- Proves that the total cost of Tom's deck is $32. -/
theorem toms_deck_cost : 
  deck_cost 19 11 30 1 (1/2) (1/4) = 32 := by
  sorry

end NUMINAMATH_CALUDE_toms_deck_cost_l1806_180658


namespace NUMINAMATH_CALUDE_minimum_balloons_l1806_180692

theorem minimum_balloons (red blue burst_red burst_blue : ℕ) : 
  red = 7 * blue →
  burst_red * 3 = burst_blue →
  burst_red ≥ 1 →
  burst_blue ≥ 1 →
  red + blue ≥ 24 :=
by sorry

end NUMINAMATH_CALUDE_minimum_balloons_l1806_180692


namespace NUMINAMATH_CALUDE_square_sum_given_condition_l1806_180616

theorem square_sum_given_condition (x y : ℝ) :
  (x - 3)^2 + |2 * y + 1| = 0 → x^2 + y^2 = 9 + 1/4 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_condition_l1806_180616


namespace NUMINAMATH_CALUDE_reader_count_l1806_180607

/-- The number of readers who read science fiction -/
def science_fiction_readers : ℕ := 120

/-- The number of readers who read literary works -/
def literary_works_readers : ℕ := 90

/-- The number of readers who read both science fiction and literary works -/
def both_genres_readers : ℕ := 60

/-- The total number of readers in the group -/
def total_readers : ℕ := science_fiction_readers + literary_works_readers - both_genres_readers

theorem reader_count : total_readers = 150 := by
  sorry

end NUMINAMATH_CALUDE_reader_count_l1806_180607


namespace NUMINAMATH_CALUDE_union_complement_problem_l1806_180644

def U : Finset Int := {-2, -1, 0, 1, 2}
def A : Finset Int := {1, 2}
def B : Finset Int := {-2, -1, 2}

theorem union_complement_problem : A ∪ (U \ B) = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_union_complement_problem_l1806_180644


namespace NUMINAMATH_CALUDE_new_year_day_frequency_new_year_day_sunday_more_frequent_l1806_180614

/-- Represents a day of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Represents a date -/
structure Date where
  year : ℕ
  month : ℕ
  day : ℕ

/-- Function to determine if a year is a leap year -/
def isLeapYear (year : ℕ) : Bool :=
  (year % 4 == 0 && year % 100 ≠ 0) || (year % 400 == 0)

/-- Function to get the day of the week for a given date -/
noncomputable def getDayOfWeek (date : Date) : DayOfWeek :=
  sorry

/-- Function to count occurrences of a specific day of the week as New Year's Day over 400 years -/
noncomputable def countNewYearDay (day : DayOfWeek) (startYear : ℕ) : ℕ :=
  sorry

/-- Theorem stating that New Year's Day falls on Sunday more frequently than on Monday over a 400-year cycle -/
theorem new_year_day_frequency (startYear : ℕ) :
  countNewYearDay DayOfWeek.Sunday startYear > countNewYearDay DayOfWeek.Monday startYear :=
by
  sorry

/-- Given condition: 23 October 1948 was a Saturday -/
axiom oct_23_1948_saturday : getDayOfWeek ⟨1948, 10, 23⟩ = DayOfWeek.Saturday

/-- Theorem to prove the frequency of New Year's Day on Sunday vs Monday -/
theorem new_year_day_sunday_more_frequent :
  ∃ startYear, countNewYearDay DayOfWeek.Sunday startYear > countNewYearDay DayOfWeek.Monday startYear :=
by
  sorry

end NUMINAMATH_CALUDE_new_year_day_frequency_new_year_day_sunday_more_frequent_l1806_180614


namespace NUMINAMATH_CALUDE_expression_evaluation_l1806_180697

theorem expression_evaluation :
  let x : ℝ := 1
  let y : ℝ := -1
  (x + y)^2 - 3*x*(x + y) + (x + 2*y)*(x - 2*y) = -3 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1806_180697


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1806_180695

theorem polynomial_factorization :
  ∀ x : ℝ, x^15 + x^10 + x^5 + 1 = (x^2 + x + 1) * (x^13 - x^12 + x^10 - x^9 + x^7 - x^6 + x^4 - x^3 + x - 1) :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1806_180695


namespace NUMINAMATH_CALUDE_real_y_condition_l1806_180688

theorem real_y_condition (x y : ℝ) : 
  (4 * y^2 + 6 * x * y + x + 8 = 0) → 
  (∃ y : ℝ, 4 * y^2 + 6 * x * y + x + 8 = 0) ↔ (x ≤ -8/9 ∨ x ≥ 4) :=
by sorry

end NUMINAMATH_CALUDE_real_y_condition_l1806_180688


namespace NUMINAMATH_CALUDE_smallest_k_inequality_l1806_180671

theorem smallest_k_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a * b + b * c + c * a + 2 * (1 / a + 1 / b + 1 / c) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_inequality_l1806_180671


namespace NUMINAMATH_CALUDE_chocolate_candy_difference_l1806_180669

/-- The cost difference between chocolate and candy bar -/
def cost_difference (chocolate_cost candy_cost : ℕ) : ℕ :=
  chocolate_cost - candy_cost

/-- Theorem stating the cost difference between chocolate and candy bar -/
theorem chocolate_candy_difference :
  cost_difference 3 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_candy_difference_l1806_180669


namespace NUMINAMATH_CALUDE_park_layout_diameter_l1806_180654

/-- The diameter of the outer boundary of a circular park layout -/
def outer_boundary_diameter (statue_diameter bench_width path_width : ℝ) : ℝ :=
  statue_diameter + 2 * (bench_width + path_width)

/-- Theorem: The diameter of the outer boundary of the jogging path is 46 feet -/
theorem park_layout_diameter :
  outer_boundary_diameter 12 10 7 = 46 := by
  sorry

end NUMINAMATH_CALUDE_park_layout_diameter_l1806_180654


namespace NUMINAMATH_CALUDE_ethanol_in_tank_l1806_180608

/-- Calculates the total amount of ethanol in a fuel tank -/
def total_ethanol (tank_capacity : ℝ) (fuel_a_volume : ℝ) (fuel_a_ethanol_percent : ℝ) (fuel_b_ethanol_percent : ℝ) : ℝ :=
  let fuel_b_volume := tank_capacity - fuel_a_volume
  let ethanol_a := fuel_a_volume * fuel_a_ethanol_percent
  let ethanol_b := fuel_b_volume * fuel_b_ethanol_percent
  ethanol_a + ethanol_b

/-- Theorem stating that the total ethanol in the given scenario is 30 gallons -/
theorem ethanol_in_tank : 
  total_ethanol 204 66 0.12 0.16 = 30 := by
  sorry

#eval total_ethanol 204 66 0.12 0.16

end NUMINAMATH_CALUDE_ethanol_in_tank_l1806_180608


namespace NUMINAMATH_CALUDE_arithmetic_equality_l1806_180663

theorem arithmetic_equality : 2021 - 2223 + 2425 = 2223 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l1806_180663


namespace NUMINAMATH_CALUDE_subtraction_of_fractions_l1806_180699

theorem subtraction_of_fractions : (1 : ℚ) / 6 - 5 / 12 = (-1 : ℚ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_fractions_l1806_180699


namespace NUMINAMATH_CALUDE_q_is_false_l1806_180657

theorem q_is_false (p q : Prop) (h1 : ¬(p ∧ q)) (h2 : ¬¬p) : ¬q := by
  sorry

end NUMINAMATH_CALUDE_q_is_false_l1806_180657


namespace NUMINAMATH_CALUDE_triangle_side_length_l1806_180633

theorem triangle_side_length (a b c : ℝ) (A : ℝ) : 
  a = 2 → c = Real.sqrt 2 → Real.cos A = -(Real.sqrt 2) / 4 → b = 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1806_180633


namespace NUMINAMATH_CALUDE_house_elves_do_not_exist_l1806_180660

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (HouseElf : U → Prop)
variable (LovesPranks : U → Prop)
variable (LovesCleanlinessAndOrder : U → Prop)

-- State the theorem
theorem house_elves_do_not_exist :
  (∀ x, HouseElf x → LovesPranks x) →
  (∀ x, HouseElf x → LovesCleanlinessAndOrder x) →
  (∀ x, LovesCleanlinessAndOrder x → ¬LovesPranks x) →
  ¬(∃ x, HouseElf x) :=
by
  sorry

end NUMINAMATH_CALUDE_house_elves_do_not_exist_l1806_180660


namespace NUMINAMATH_CALUDE_min_filters_correct_l1806_180666

/-- The minimum number of filters required to reduce impurities -/
def min_filters : ℕ := 5

/-- The initial impurity concentration -/
def initial_impurity : ℝ := 0.2

/-- The fraction of impurities remaining after each filter -/
def filter_efficiency : ℝ := 0.2

/-- The maximum allowed final impurity concentration -/
def max_final_impurity : ℝ := 0.0001

/-- Theorem stating that min_filters is the minimum number of filters required -/
theorem min_filters_correct :
  (initial_impurity * filter_efficiency ^ min_filters ≤ max_final_impurity) ∧
  (∀ k : ℕ, k < min_filters → initial_impurity * filter_efficiency ^ k > max_final_impurity) :=
sorry

end NUMINAMATH_CALUDE_min_filters_correct_l1806_180666


namespace NUMINAMATH_CALUDE_sum_of_circle_areas_l1806_180661

/-- The sum of the areas of an infinite sequence of circles with decreasing radii -/
theorem sum_of_circle_areas : 
  ∀ (π : ℝ), π > 0 → 
  (∑' n, π * (3 / (3 ^ n : ℝ))^2) = 9 * π / 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_circle_areas_l1806_180661


namespace NUMINAMATH_CALUDE_bernoullis_inequality_l1806_180636

theorem bernoullis_inequality (x : ℝ) (n : ℕ) (h1 : x ≥ -1) (h2 : n ≥ 1) :
  (1 + x)^n ≥ 1 + n*x := by
  sorry

end NUMINAMATH_CALUDE_bernoullis_inequality_l1806_180636


namespace NUMINAMATH_CALUDE_wood_wasted_percentage_l1806_180694

/-- The percentage of wood wasted when carving a cone from a sphere -/
theorem wood_wasted_percentage (sphere_radius cone_height cone_base_diameter : ℝ) :
  sphere_radius = 9 →
  cone_height = 9 →
  cone_base_diameter = 18 →
  let cone_base_radius := cone_base_diameter / 2
  let cone_volume := (1 / 3) * Real.pi * cone_base_radius^2 * cone_height
  let sphere_volume := (4 / 3) * Real.pi * sphere_radius^3
  let percentage_wasted := (cone_volume / sphere_volume) * 100
  percentage_wasted = 25 := by sorry

end NUMINAMATH_CALUDE_wood_wasted_percentage_l1806_180694


namespace NUMINAMATH_CALUDE_weighted_average_girl_scouts_permission_l1806_180685

-- Define the structure for each trip
structure Trip where
  total_scouts : ℕ
  boy_scout_percentage : ℚ
  girl_scout_percentage : ℚ
  boy_scout_permission_percentage : ℚ
  girl_scout_participation_percentage : ℚ
  girl_scout_permission_percentage : ℚ

-- Define the three trips
def trip1 : Trip := {
  total_scouts := 100,
  boy_scout_percentage := 60/100,
  girl_scout_percentage := 40/100,
  boy_scout_permission_percentage := 75/100,
  girl_scout_participation_percentage := 50/100,
  girl_scout_permission_percentage := 50/100
}

def trip2 : Trip := {
  total_scouts := 150,
  boy_scout_percentage := 50/100,
  girl_scout_percentage := 50/100,
  boy_scout_permission_percentage := 80/100,
  girl_scout_participation_percentage := 70/100,
  girl_scout_permission_percentage := 60/100
}

def trip3 : Trip := {
  total_scouts := 200,
  boy_scout_percentage := 40/100,
  girl_scout_percentage := 60/100,
  boy_scout_permission_percentage := 85/100,
  girl_scout_participation_percentage := 100/100,
  girl_scout_permission_percentage := 75/100
}

-- Function to calculate the number of Girl Scouts with permission slips for a trip
def girl_scouts_with_permission (trip : Trip) : ℚ :=
  trip.total_scouts * trip.girl_scout_percentage * trip.girl_scout_participation_percentage * trip.girl_scout_permission_percentage

-- Function to calculate the total number of participating Girl Scouts for a trip
def participating_girl_scouts (trip : Trip) : ℚ :=
  trip.total_scouts * trip.girl_scout_percentage * trip.girl_scout_participation_percentage

-- Theorem statement
theorem weighted_average_girl_scouts_permission (ε : ℚ) (h : ε > 0) :
  let total_with_permission := girl_scouts_with_permission trip1 + girl_scouts_with_permission trip2 + girl_scouts_with_permission trip3
  let total_participating := participating_girl_scouts trip1 + participating_girl_scouts trip2 + participating_girl_scouts trip3
  let weighted_average := total_with_permission / total_participating * 100
  |weighted_average - 68| < ε :=
by sorry

end NUMINAMATH_CALUDE_weighted_average_girl_scouts_permission_l1806_180685


namespace NUMINAMATH_CALUDE_jessica_seashells_count_l1806_180603

/-- The number of seashells Mary found -/
def mary_seashells : ℕ := 18

/-- The total number of seashells Mary and Jessica found together -/
def total_seashells : ℕ := 59

/-- The number of seashells Jessica found -/
def jessica_seashells : ℕ := total_seashells - mary_seashells

theorem jessica_seashells_count : jessica_seashells = 41 := by
  sorry

end NUMINAMATH_CALUDE_jessica_seashells_count_l1806_180603


namespace NUMINAMATH_CALUDE_parabola_complementary_lines_l1806_180690

/-- Represents a point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 2px -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

theorem parabola_complementary_lines (para : Parabola) (P A B : Point)
  (h_P_on_para : P.y^2 = 2 * para.p * P.x)
  (h_P_y_pos : P.y > 0)
  (h_A_on_para : A.y^2 = 2 * para.p * A.x)
  (h_B_on_para : B.y^2 = 2 * para.p * B.x)
  (h_PA_slope_exists : A.x ≠ P.x)
  (h_PB_slope_exists : B.x ≠ P.x)
  (h_complementary : 
    (A.y - P.y) / (A.x - P.x) * (B.y - P.y) / (B.x - P.x) = -1) :
  (A.y + B.y) / P.y = -2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_complementary_lines_l1806_180690


namespace NUMINAMATH_CALUDE_soda_survey_result_l1806_180640

/-- Given a survey of 520 people and a central angle of 220° for the "Soda" sector,
    prove that 317 people chose "Soda". -/
theorem soda_survey_result (total_surveyed : ℕ) (soda_angle : ℝ) :
  total_surveyed = 520 →
  soda_angle = 220 →
  ∃ (soda_count : ℕ),
    soda_count = 317 ∧
    (soda_count : ℝ) / total_surveyed * 360 ≥ soda_angle - 0.5 ∧
    (soda_count : ℝ) / total_surveyed * 360 < soda_angle + 0.5 :=
by sorry


end NUMINAMATH_CALUDE_soda_survey_result_l1806_180640


namespace NUMINAMATH_CALUDE_function_max_min_l1806_180641

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + a^2 - 1

theorem function_max_min (a : ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ 3 → f a x ≤ 24) ∧
  (∃ x, 0 ≤ x ∧ x ≤ 3 ∧ f a x = 24) ∧
  (∀ x, 0 ≤ x ∧ x ≤ 3 → f a x ≥ 3) ∧
  (∃ x, 0 ≤ x ∧ x ≤ 3 ∧ f a x = 3) →
  a = 2 ∨ a = -5 := by sorry

end NUMINAMATH_CALUDE_function_max_min_l1806_180641


namespace NUMINAMATH_CALUDE_expression_equals_8_175_l1806_180680

-- Define the expression
def expression : ℝ := (4.5 - 1.23) * 2.5

-- State the theorem
theorem expression_equals_8_175 : expression = 8.175 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_8_175_l1806_180680


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1806_180647

open Set

-- Define sets A and B
def A : Set ℝ := {x | -3 < x ∧ x < 3}
def B : Set ℝ := {x | x^2 - x - 6 ≤ 0}

-- Theorem statement
theorem union_of_A_and_B : A ∪ B = {x : ℝ | -3 < x ∧ x ≤ 3} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1806_180647


namespace NUMINAMATH_CALUDE_circumscribed_sphere_radius_rectangular_solid_l1806_180631

/-- For a rectangular solid with edges a, b, and c, the radius R of its circumscribed sphere
    satisfies the equation 4R² = a² + b² + c². -/
theorem circumscribed_sphere_radius_rectangular_solid
  (a b c R : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  4 * R^2 = a^2 + b^2 + c^2 :=
sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_radius_rectangular_solid_l1806_180631


namespace NUMINAMATH_CALUDE_three_digit_factorial_sum_l1806_180612

theorem three_digit_factorial_sum : ∃ (x y z : ℕ), 
  (1 ≤ x ∧ x ≤ 9) ∧ 
  (0 ≤ y ∧ y ≤ 9) ∧ 
  (0 ≤ z ∧ z ≤ 9) ∧
  (100 * x + 10 * y + z = Nat.factorial x + Nat.factorial y + Nat.factorial z) ∧
  (x + y + z = 10) := by
  sorry

end NUMINAMATH_CALUDE_three_digit_factorial_sum_l1806_180612


namespace NUMINAMATH_CALUDE_no_natural_squares_l1806_180649

theorem no_natural_squares (x y : ℕ) : ¬(∃ a b : ℕ, x^2 + y = a^2 ∧ y^2 + x = b^2) := by
  sorry

end NUMINAMATH_CALUDE_no_natural_squares_l1806_180649


namespace NUMINAMATH_CALUDE_trapezoid_height_l1806_180656

-- Define the trapezoid properties
structure IsoscelesTrapezoid where
  diagonal : ℝ
  area : ℝ

-- Define the theorem
theorem trapezoid_height (t : IsoscelesTrapezoid) (h_diagonal : t.diagonal = 10) (h_area : t.area = 48) :
  ∃ (height : ℝ), (height = 6 ∨ height = 8) ∧ 
  (∃ (base_avg : ℝ), base_avg * height = t.area ∧ base_avg^2 + height^2 = t.diagonal^2) :=
sorry

end NUMINAMATH_CALUDE_trapezoid_height_l1806_180656


namespace NUMINAMATH_CALUDE_max_cola_bottles_30_yuan_l1806_180673

/-- Calculates the maximum number of cola bottles that can be consumed given an initial amount of money, the cost per bottle, and the exchange rate of empty bottles for full bottles. -/
def max_cola_bottles (initial_money : ℕ) (cost_per_bottle : ℕ) (exchange_rate : ℕ) : ℕ :=
  sorry

/-- Theorem stating that given 30 yuan, a cola cost of 2 yuan per bottle, and the ability to exchange 2 empty bottles for 1 full bottle, the maximum number of cola bottles that can be consumed is 29. -/
theorem max_cola_bottles_30_yuan :
  max_cola_bottles 30 2 2 = 29 :=
sorry

end NUMINAMATH_CALUDE_max_cola_bottles_30_yuan_l1806_180673


namespace NUMINAMATH_CALUDE_sum_product_inequality_l1806_180626

theorem sum_product_inequality (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq_four : a + b + c + d = 4) : 
  a * b + b * c + c * d + d * a ≤ 4 := by
sorry

end NUMINAMATH_CALUDE_sum_product_inequality_l1806_180626


namespace NUMINAMATH_CALUDE_two_digit_number_concatenation_l1806_180628

/-- A two-digit number is an integer between 10 and 99, inclusive. -/
def TwoDigitNumber (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem two_digit_number_concatenation (x y : ℕ) 
  (hx : TwoDigitNumber x) (hy : TwoDigitNumber y) :
  ∃ (n : ℕ), n = 100 * x + y ∧ 1000 ≤ n ∧ n ≤ 9999 :=
by sorry

end NUMINAMATH_CALUDE_two_digit_number_concatenation_l1806_180628


namespace NUMINAMATH_CALUDE_angle_T_measure_l1806_180686

/-- Represents a heptagon with specific angle properties -/
structure Heptagon :=
  (G E O M Y J R T : ℝ)
  (sum_angles : G + E + O + M + Y + J + R + T = 900)
  (equal_angles : G = E ∧ E = T ∧ T = R)
  (supplementary_M_Y : M + Y = 180)
  (supplementary_J_O : J + O = 180)

/-- The measure of angle T in the specified heptagon is 135° -/
theorem angle_T_measure (h : Heptagon) : h.T = 135 := by
  sorry

end NUMINAMATH_CALUDE_angle_T_measure_l1806_180686


namespace NUMINAMATH_CALUDE_line_no_intersection_slope_range_l1806_180634

/-- Given points A(-2,3) and B(3,2), and a line l: y = kx - 2, 
    if l has no intersection with line segment AB, 
    then the slope k of line l is in the range (-5/2, 4/3). -/
theorem line_no_intersection_slope_range (k : ℝ) : 
  let A : ℝ × ℝ := (-2, 3)
  let B : ℝ × ℝ := (3, 2)
  let l (x : ℝ) := k * x - 2
  (∀ x y, (x, y) ∈ Set.Icc A B → y ≠ l x) →
  k ∈ Set.Ioo (-5/2 : ℝ) (4/3 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_line_no_intersection_slope_range_l1806_180634


namespace NUMINAMATH_CALUDE_log_equation_solution_l1806_180655

theorem log_equation_solution (t : ℝ) (h : t > 0) :
  4 * Real.log t / Real.log 3 = Real.log (4 * t) / Real.log 3 + 2 → t = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1806_180655


namespace NUMINAMATH_CALUDE_a_works_friday_50th_week_l1806_180693

/-- Represents the days of the week -/
inductive Day
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents the people working night shifts -/
inductive Person
  | A
  | B
  | C
  | D
  | E
  | F

/-- Returns the next day in the week -/
def nextDay (d : Day) : Day :=
  match d with
  | Day.Sunday => Day.Monday
  | Day.Monday => Day.Tuesday
  | Day.Tuesday => Day.Wednesday
  | Day.Wednesday => Day.Thursday
  | Day.Thursday => Day.Friday
  | Day.Friday => Day.Saturday
  | Day.Saturday => Day.Sunday

/-- Returns the next person in the rotation -/
def nextPerson (p : Person) : Person :=
  match p with
  | Person.A => Person.B
  | Person.B => Person.C
  | Person.C => Person.D
  | Person.D => Person.E
  | Person.E => Person.F
  | Person.F => Person.A

/-- Returns the person working on a given day number -/
def personOnDay (dayNumber : Nat) : Person :=
  match dayNumber % 6 with
  | 0 => Person.F
  | 1 => Person.A
  | 2 => Person.B
  | 3 => Person.C
  | 4 => Person.D
  | 5 => Person.E
  | _ => Person.A  -- This case should never occur

/-- Returns the day of the week for a given day number -/
def dayOfWeek (dayNumber : Nat) : Day :=
  match dayNumber % 7 with
  | 0 => Day.Saturday
  | 1 => Day.Sunday
  | 2 => Day.Monday
  | 3 => Day.Tuesday
  | 4 => Day.Wednesday
  | 5 => Day.Thursday
  | 6 => Day.Friday
  | _ => Day.Sunday  -- This case should never occur

theorem a_works_friday_50th_week :
  personOnDay (50 * 7 - 2) = Person.A ∧ dayOfWeek (50 * 7 - 2) = Day.Friday :=
by sorry

end NUMINAMATH_CALUDE_a_works_friday_50th_week_l1806_180693


namespace NUMINAMATH_CALUDE_smallest_positive_difference_l1806_180630

/-- Vovochka's addition method for three-digit numbers -/
def vovochka_sum (a b c d e f : ℕ) : ℕ :=
  (a + d) * 1000 + (b + e) * 100 + (c + f)

/-- Correct addition method for three-digit numbers -/
def correct_sum (a b c d e f : ℕ) : ℕ :=
  (a + d) * 100 + (b + e) * 10 + (c + f)

/-- The difference between Vovochka's sum and the correct sum -/
def sum_difference (a b c d e f : ℕ) : ℤ :=
  (vovochka_sum a b c d e f : ℤ) - (correct_sum a b c d e f : ℤ)

theorem smallest_positive_difference :
  ∃ (a b c d e f : ℕ),
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10 ∧
    sum_difference a b c d e f > 0 ∧
    ∀ (x y z u v w : ℕ),
      x < 10 → y < 10 → z < 10 → u < 10 → v < 10 → w < 10 →
      sum_difference x y z u v w > 0 →
      sum_difference a b c d e f ≤ sum_difference x y z u v w ∧
    sum_difference a b c d e f = 1800 :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_difference_l1806_180630


namespace NUMINAMATH_CALUDE_muffin_price_theorem_l1806_180696

/-- The price per muffin to raise the required amount -/
def price_per_muffin (total_amount : ℚ) (num_cases : ℕ) (packs_per_case : ℕ) (muffins_per_pack : ℕ) : ℚ :=
  total_amount / (num_cases * packs_per_case * muffins_per_pack)

/-- Theorem: The price per muffin to raise $120 by selling 5 cases of muffins, 
    where each case contains 3 packs and each pack contains 4 muffins, is $2 -/
theorem muffin_price_theorem :
  price_per_muffin 120 5 3 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_muffin_price_theorem_l1806_180696


namespace NUMINAMATH_CALUDE_exactly_two_red_prob_l1806_180675

def total_balls : ℕ := 10
def red_balls : ℕ := 4
def white_balls : ℕ := 6
def num_draws : ℕ := 4
def num_red_draws : ℕ := 2

def prob_red : ℚ := red_balls / total_balls
def prob_white : ℚ := white_balls / total_balls

theorem exactly_two_red_prob : 
  (Nat.choose num_draws num_red_draws : ℚ) * prob_red ^ num_red_draws * prob_white ^ (num_draws - num_red_draws) = 3456 / 10000 := by
  sorry

end NUMINAMATH_CALUDE_exactly_two_red_prob_l1806_180675


namespace NUMINAMATH_CALUDE_curve_tangent_to_line_l1806_180662

/-- The curve y = x^2 - x + a is tangent to the line y = x + 1 if and only if a = 2 -/
theorem curve_tangent_to_line (a : ℝ) : 
  (∃ x y : ℝ, y = x^2 - x + a ∧ y = x + 1 ∧ 2*x - 1 = 1) ↔ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_curve_tangent_to_line_l1806_180662


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_theorem_l1806_180613

-- Define the points
variable (A B C D P Q E : Point)

-- Define the conditions
def is_convex_quadrilateral (A B C D : Point) : Prop := sorry
def is_inside (P Q : Point) (A B C D : Point) : Prop := sorry
def is_cyclic_quadrilateral (P Q D A : Point) : Prop := sorry
def point_on_line (E P Q : Point) : Prop := sorry
def angle_eq (P A E Q D : Point) : Prop := sorry

-- State the theorem
theorem cyclic_quadrilateral_theorem 
  (h1 : is_convex_quadrilateral A B C D)
  (h2 : is_inside P Q A B C D)
  (h3 : is_cyclic_quadrilateral P Q D A)
  (h4 : is_cyclic_quadrilateral Q P B C)
  (h5 : point_on_line E P Q)
  (h6 : angle_eq P A E Q D)
  (h7 : angle_eq P B E Q C) :
  is_cyclic_quadrilateral A B C D :=
sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_theorem_l1806_180613


namespace NUMINAMATH_CALUDE_sqrt_x_plus_5_equals_3_l1806_180677

theorem sqrt_x_plus_5_equals_3 (x : ℝ) : 
  Real.sqrt (x + 5) = 3 → (x + 5)^2 = 81 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_5_equals_3_l1806_180677


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1806_180651

def A : Set ℝ := {-1, 0, 1, 2}
def B : Set ℝ := {x | 0 < x ∧ x < 3}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1806_180651


namespace NUMINAMATH_CALUDE_vector_operations_l1806_180652

/-- Given vectors a and b, prove their sum and dot product -/
theorem vector_operations (a b : ℝ × ℝ × ℝ) 
  (ha : a = (1, 2, 2)) (hb : b = (6, -3, 2)) : 
  (a.1 + b.1, a.2.1 + b.2.1, a.2.2 + b.2.2) = (7, -1, 4) ∧ 
  (a.1 * b.1 + a.2.1 * b.2.1 + a.2.2 * b.2.2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_vector_operations_l1806_180652


namespace NUMINAMATH_CALUDE_zero_in_interval_l1806_180672

def f (x : ℝ) := x^3 + 3*x - 1

theorem zero_in_interval :
  (f 0 < 0) →
  (f 0.5 > 0) →
  (f 0.25 < 0) →
  ∃ x : ℝ, x ∈ Set.Ioo 0.25 0.5 ∧ f x = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_zero_in_interval_l1806_180672


namespace NUMINAMATH_CALUDE_figure_tiling_iff_multiple_of_three_l1806_180635

/-- Represents a figure Φ consisting of three n×n squares. -/
structure Figure (n : ℕ) where
  squares : Fin 3 → Fin n → Fin n → Bool

/-- Represents a 1×3 or 3×1 tile. -/
inductive Tile
  | horizontal : Tile
  | vertical : Tile

/-- A tiling of the figure Φ using 1×3 and 3×1 tiles. -/
def Tiling (n : ℕ) := Set (ℕ × ℕ × Tile)

/-- Predicate to check if a tiling is valid for the given figure. -/
def isValidTiling (n : ℕ) (φ : Figure n) (t : Tiling n) : Prop := sorry

/-- The main theorem stating that a valid tiling exists if and only if n is a multiple of 3. -/
theorem figure_tiling_iff_multiple_of_three (n : ℕ) (φ : Figure n) :
  (n > 1) → (∃ t : Tiling n, isValidTiling n φ t) ↔ ∃ k : ℕ, n = 3 * k :=
sorry

end NUMINAMATH_CALUDE_figure_tiling_iff_multiple_of_three_l1806_180635


namespace NUMINAMATH_CALUDE_average_of_other_results_l1806_180605

theorem average_of_other_results
  (n₁ : ℕ) (n₂ : ℕ) (avg₁ : ℚ) (avg_all : ℚ)
  (h₁ : n₁ = 60)
  (h₂ : n₂ = 40)
  (h₃ : avg₁ = 40)
  (h₄ : avg_all = 48)
  : (n₁ * avg₁ + n₂ * ((n₁ + n₂) * avg_all - n₁ * avg₁) / n₂) / (n₁ + n₂) = avg_all ∧
    ((n₁ + n₂) * avg_all - n₁ * avg₁) / n₂ = 60 :=
by sorry

end NUMINAMATH_CALUDE_average_of_other_results_l1806_180605


namespace NUMINAMATH_CALUDE_cricketer_wickets_after_match_l1806_180617

/-- Represents a cricketer's bowling statistics -/
structure Cricketer where
  initialAverage : ℝ
  initialWickets : ℕ
  matchWickets : ℕ
  matchRuns : ℕ
  averageDecrease : ℝ

/-- Calculates the total number of wickets taken by a cricketer after a match -/
def totalWicketsAfterMatch (c : Cricketer) : ℕ :=
  c.initialWickets + c.matchWickets

/-- Theorem stating that for a cricketer with given statistics, the total wickets after the match is 90 -/
theorem cricketer_wickets_after_match (c : Cricketer) 
  (h1 : c.initialAverage = 12.4)
  (h2 : c.matchWickets = 5)
  (h3 : c.matchRuns = 26)
  (h4 : c.averageDecrease = 0.4) :
  totalWicketsAfterMatch c = 90 := by
  sorry

end NUMINAMATH_CALUDE_cricketer_wickets_after_match_l1806_180617


namespace NUMINAMATH_CALUDE_exists_solution_set_exists_a_range_l1806_180682

-- Define the function f
def f (x a : ℝ) : ℝ := |2*x + 1| + |2*x - a| + a

-- Theorem for part (Ⅰ)
theorem exists_solution_set (a : ℝ) (h : a = 3) :
  ∃ S : Set ℝ, ∀ x ∈ S, f x a > 7 :=
sorry

-- Theorem for part (Ⅱ)
theorem exists_a_range :
  ∃ a_min a_max : ℝ, ∀ a : ℝ, a_min ≤ a ∧ a ≤ a_max →
    ∀ x : ℝ, f x a ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_exists_solution_set_exists_a_range_l1806_180682


namespace NUMINAMATH_CALUDE_baseball_cost_proof_l1806_180664

def football_cost : ℚ := 9.14
def total_payment : ℚ := 20
def change_received : ℚ := 4.05

theorem baseball_cost_proof :
  let total_spent := total_payment - change_received
  let baseball_cost := total_spent - football_cost
  baseball_cost = 6.81 := by sorry

end NUMINAMATH_CALUDE_baseball_cost_proof_l1806_180664


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l1806_180601

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 8/15) (h2 : x - y = 2/15) : x^2 - y^2 = 16/225 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l1806_180601


namespace NUMINAMATH_CALUDE_probability_negative_product_l1806_180665

def set_a : Finset ℤ := {0, 1, -3, 6, -8, -10}
def set_b : Finset ℤ := {-1, 2, -4, 7, 6, -9}

def negative_product_pairs : Finset (ℤ × ℤ) :=
  (set_a.filter (λ x => x > 0) ×ˢ set_b.filter (λ y => y < 0)) ∪
  (set_a.filter (λ x => x < 0) ×ˢ set_b.filter (λ y => y > 0))

theorem probability_negative_product :
  (negative_product_pairs.card : ℚ) / ((set_a.card * set_b.card) : ℚ) = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_probability_negative_product_l1806_180665


namespace NUMINAMATH_CALUDE_happy_valley_theorem_l1806_180643

/-- The number of ways to arrange animals in the Happy Valley Kennel -/
def happy_valley_arrangements : ℕ :=
  let num_chickens : ℕ := 3
  let num_dogs : ℕ := 4
  let num_cats : ℕ := 6
  let total_animals : ℕ := num_chickens + num_dogs + num_cats
  let group_arrangements : ℕ := 2  -- chicken-dog or dog-chicken around cats
  let chicken_arrangements : ℕ := Nat.factorial num_chickens
  let dog_arrangements : ℕ := Nat.factorial num_dogs
  let cat_arrangements : ℕ := Nat.factorial num_cats
  group_arrangements * chicken_arrangements * dog_arrangements * cat_arrangements

/-- Theorem stating the correct number of arrangements for the Happy Valley Kennel problem -/
theorem happy_valley_theorem : happy_valley_arrangements = 69120 := by
  sorry

end NUMINAMATH_CALUDE_happy_valley_theorem_l1806_180643


namespace NUMINAMATH_CALUDE_total_distance_is_twenty_l1806_180600

/-- Represents the travel time per mile for each day -/
def travel_time (day : Nat) : Nat :=
  10 + 6 * (day - 1)

/-- Represents the distance traveled on each day -/
def distance (day : Nat) : Nat :=
  60 / travel_time day

/-- The total distance traveled over 5 days -/
def total_distance : Nat :=
  (List.range 5).map (fun i => distance (i + 1)) |>.sum

/-- Theorem stating that the total distance traveled is 20 miles -/
theorem total_distance_is_twenty : total_distance = 20 := by
  sorry

#eval total_distance

end NUMINAMATH_CALUDE_total_distance_is_twenty_l1806_180600


namespace NUMINAMATH_CALUDE_semiperimeter_radius_sum_eq_legs_sum_l1806_180667

/-- A right triangle with legs a and b, hypotenuse c, semiperimeter p, and inscribed circle radius r -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  p : ℝ
  r : ℝ
  right_angle : c^2 = a^2 + b^2
  semiperimeter : p = (a + b + c) / 2
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

/-- The sum of the semiperimeter and the radius of the inscribed circle is equal to the sum of the legs -/
theorem semiperimeter_radius_sum_eq_legs_sum (t : RightTriangle) : t.p + t.r = t.a + t.b := by
  sorry

end NUMINAMATH_CALUDE_semiperimeter_radius_sum_eq_legs_sum_l1806_180667


namespace NUMINAMATH_CALUDE_largest_n_for_factorization_l1806_180620

/-- 
Theorem: The largest value of n such that 3x^2 + nx + 72 can be factored 
as the product of two linear factors with integer coefficients is 217.
-/
theorem largest_n_for_factorization : 
  ∃ (n : ℤ), n = 217 ∧ 
  (∀ m : ℤ, m > n → 
    ¬∃ (a b c d : ℤ), 3 * X^2 + m * X + 72 = (a * X + b) * (c * X + d)) ∧
  (∃ (a b c d : ℤ), 3 * X^2 + n * X + 72 = (a * X + b) * (c * X + d)) :=
sorry

end NUMINAMATH_CALUDE_largest_n_for_factorization_l1806_180620


namespace NUMINAMATH_CALUDE_transformations_result_l1806_180627

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Rotates a point 180° around the x-axis -/
def rotateX (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := -p.z }

/-- Reflects a point through the xy-plane -/
def reflectXY (p : Point3D) : Point3D :=
  { x := p.x, y := p.y, z := -p.z }

/-- Reflects a point through the yz-plane -/
def reflectYZ (p : Point3D) : Point3D :=
  { x := -p.x, y := p.y, z := p.z }

/-- Applies the sequence of transformations to a point -/
def applyTransformations (p : Point3D) : Point3D :=
  reflectYZ (rotateX (reflectYZ (reflectXY (rotateX p))))

theorem transformations_result :
  applyTransformations { x := 1, y := 1, z := 1 } = { x := 1, y := 1, z := -1 } := by
  sorry


end NUMINAMATH_CALUDE_transformations_result_l1806_180627


namespace NUMINAMATH_CALUDE_a_14_equals_41_l1806_180610

/-- An arithmetic sequence with a_2 = 5 and a_6 = 17 -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  a 2 = 5 ∧ a 6 = 17 ∧ ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In the given arithmetic sequence, a_14 = 41 -/
theorem a_14_equals_41 (a : ℕ → ℝ) (h : arithmetic_sequence a) : a 14 = 41 := by
  sorry

end NUMINAMATH_CALUDE_a_14_equals_41_l1806_180610


namespace NUMINAMATH_CALUDE_shaded_area_regular_octagon_l1806_180698

/-- The area of the shaded region in a regular octagon with side length 8 cm, 
    formed by connecting the midpoints of consecutive sides. -/
theorem shaded_area_regular_octagon (s : ℝ) (h : s = 8) : 
  let outer_area := 2 * (1 + Real.sqrt 2) * s^2
  let inner_side := s * (1 - Real.sqrt 2 / 2)
  let inner_area := 2 * (1 + Real.sqrt 2) * inner_side^2
  outer_area - inner_area = 128 * (1 + Real.sqrt 2) - 2 * (1 + Real.sqrt 2) * (8 - 4 * Real.sqrt 2)^2 :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_regular_octagon_l1806_180698


namespace NUMINAMATH_CALUDE_min_value_expression_l1806_180621

theorem min_value_expression (x : ℝ) (hx : x > 0) :
  4 * Real.sqrt x + 4 / x + 1 / (x^2) ≥ 9 ∧
  ∃ y : ℝ, y > 0 ∧ 4 * Real.sqrt y + 4 / y + 1 / (y^2) = 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l1806_180621


namespace NUMINAMATH_CALUDE_chlorine_used_equals_chloromethane_formed_l1806_180632

/-- Represents the chemical reaction between Methane and Chlorine to form Chloromethane -/
structure ChemicalReaction where
  methane_initial : ℝ
  chloromethane_formed : ℝ

/-- Theorem stating that the moles of Chlorine used equals the moles of Chloromethane formed -/
theorem chlorine_used_equals_chloromethane_formed (reaction : ChemicalReaction)
  (h : reaction.methane_initial = reaction.chloromethane_formed) :
  reaction.chloromethane_formed = reaction.methane_initial :=
by sorry

end NUMINAMATH_CALUDE_chlorine_used_equals_chloromethane_formed_l1806_180632


namespace NUMINAMATH_CALUDE_common_difference_not_three_l1806_180668

def is_valid_sequence (d : ℕ+) : Prop :=
  ∃ (n : ℕ+), 1 + (n - 1) * d = 81

theorem common_difference_not_three :
  ¬(is_valid_sequence 3) := by
  sorry

end NUMINAMATH_CALUDE_common_difference_not_three_l1806_180668


namespace NUMINAMATH_CALUDE_soap_packages_per_box_l1806_180648

theorem soap_packages_per_box (soaps_per_package : ℕ) (num_boxes : ℕ) (total_soaps : ℕ) :
  soaps_per_package = 192 →
  num_boxes = 2 →
  total_soaps = 2304 →
  ∃ (packages_per_box : ℕ), 
    packages_per_box * soaps_per_package * num_boxes = total_soaps ∧
    packages_per_box = 6 :=
by sorry

end NUMINAMATH_CALUDE_soap_packages_per_box_l1806_180648
