import Mathlib

namespace NUMINAMATH_CALUDE_prob_at_least_one_passes_eq_0_995_l350_35093

/-- The probability that at least one candidate passes the test -/
def prob_at_least_one_passes (prob_A prob_B prob_C : ℝ) : ℝ :=
  1 - (1 - prob_A) * (1 - prob_B) * (1 - prob_C)

/-- Theorem stating that the probability of at least one candidate passing is 0.995 -/
theorem prob_at_least_one_passes_eq_0_995 :
  prob_at_least_one_passes 0.9 0.8 0.75 = 0.995 := by
  sorry

#eval prob_at_least_one_passes 0.9 0.8 0.75

end NUMINAMATH_CALUDE_prob_at_least_one_passes_eq_0_995_l350_35093


namespace NUMINAMATH_CALUDE_variance_is_dispersion_measure_mean_is_not_dispersion_measure_median_is_not_dispersion_measure_mode_is_not_dispersion_measure_l350_35026

-- Define a type for data sets
def DataSet := List ℝ

-- Define measures
def mean (data : DataSet) : ℝ := sorry
def variance (data : DataSet) : ℝ := sorry
def median (data : DataSet) : ℝ := sorry
def mode (data : DataSet) : ℝ := sorry

-- Define a predicate for measures of dispersion
def isDispersionMeasure (measure : DataSet → ℝ) : Prop := sorry

-- Theorem stating that variance is a measure of dispersion
theorem variance_is_dispersion_measure : isDispersionMeasure variance := sorry

-- Theorems stating that mean, median, and mode are not measures of dispersion
theorem mean_is_not_dispersion_measure : ¬ isDispersionMeasure mean := sorry
theorem median_is_not_dispersion_measure : ¬ isDispersionMeasure median := sorry
theorem mode_is_not_dispersion_measure : ¬ isDispersionMeasure mode := sorry

end NUMINAMATH_CALUDE_variance_is_dispersion_measure_mean_is_not_dispersion_measure_median_is_not_dispersion_measure_mode_is_not_dispersion_measure_l350_35026


namespace NUMINAMATH_CALUDE_decimal_fraction_equality_l350_35052

theorem decimal_fraction_equality : (0.5^4) / (0.05^3) = 500 := by
  sorry

end NUMINAMATH_CALUDE_decimal_fraction_equality_l350_35052


namespace NUMINAMATH_CALUDE_right_triangle_area_l350_35047

/-- The area of a right triangle with hypotenuse 10 and sum of other sides 14 is 24 -/
theorem right_triangle_area (a b c : ℝ) (h1 : a + b = 14) (h2 : c = 10) (h3 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 24 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l350_35047


namespace NUMINAMATH_CALUDE_unique_divisibility_factor_l350_35057

/-- The polynomial p(x) = 12x^3 + 6x^2 - 54x + 63 -/
def p (x : ℝ) : ℝ := 12 * x^3 + 6 * x^2 - 54 * x + 63

/-- The polynomial is divisible by (x - k)^2 if there exists a polynomial q such that
    p(x) = (x - k)^2 * q(x) for all x -/
def is_divisible_by_squared_factor (k : ℝ) : Prop :=
  ∃ q : ℝ → ℝ, ∀ x, p x = (x - k)^2 * q x

theorem unique_divisibility_factor :
  ∃! k : ℝ, is_divisible_by_squared_factor k ∧ k = -11/9 := by sorry

end NUMINAMATH_CALUDE_unique_divisibility_factor_l350_35057


namespace NUMINAMATH_CALUDE_middle_aged_employees_selected_l350_35085

/-- Represents the age group of an employee -/
inductive AgeGroup
  | Elderly
  | MiddleAged
  | Young

/-- Represents the company with its employee distribution -/
structure Company where
  total_employees : ℕ
  age_ratio : Fin 3 → ℕ
  age_ratio_sum : age_ratio 0 + age_ratio 1 + age_ratio 2 > 0

/-- Calculates the number of employees in a specific age group -/
def employees_in_group (c : Company) (g : Fin 3) : ℕ :=
  c.total_employees * c.age_ratio g / (c.age_ratio 0 + c.age_ratio 1 + c.age_ratio 2)

/-- Theorem: The number of middle-aged employees selected in stratified sampling -/
theorem middle_aged_employees_selected
  (c : Company)
  (h_total : c.total_employees = 1200)
  (h_ratio : c.age_ratio = ![1, 5, 6])
  (sample_size : ℕ)
  (h_sample : sample_size = 36) :
  employees_in_group c 1 * sample_size / c.total_employees = 15 := by
  sorry

end NUMINAMATH_CALUDE_middle_aged_employees_selected_l350_35085


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_solution_set_theorem_l350_35025

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B : Set ℝ := {x | x^2 - 5*x + 6 < 0}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := {x | -1 < x ∧ x < 2}

-- Theorem for part (1)
theorem intersection_of_A_and_B : A ∩ B = A_intersect_B := by sorry

-- Define the solution set of x^2 + ax - b < 0
def solution_set (a b : ℝ) : Set ℝ := {x | x < -1 ∨ x > 2}

-- Theorem for part (2)
theorem solution_set_theorem (a b : ℝ) :
  ({x : ℝ | x^2 + a*x + b < 0} = A_intersect_B) →
  ({x : ℝ | x^2 + a*x - b < 0} = solution_set a b) := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_solution_set_theorem_l350_35025


namespace NUMINAMATH_CALUDE_log_inequality_l350_35059

-- Define the logarithm function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Main theorem
theorem log_inequality (a m n : ℝ) (ha : a > 1) (hm : 0 < m) (hmn : m < 1) (hn : 1 < n) :
  f a m < 0 ∧ 0 < f a n := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l350_35059


namespace NUMINAMATH_CALUDE_box_dimensions_sum_l350_35013

theorem box_dimensions_sum (X Y Z : ℝ) : 
  X > 0 ∧ Y > 0 ∧ Z > 0 →
  X * Y = 24 →
  X * Z = 48 →
  Y * Z = 72 →
  X + Y + Z = 22 := by
sorry

end NUMINAMATH_CALUDE_box_dimensions_sum_l350_35013


namespace NUMINAMATH_CALUDE_problem_statement_l350_35080

/-- Given points P, Q, and O, and a function f, prove properties about f and a related triangle -/
theorem problem_statement 
  (P : ℝ × ℝ) 
  (Q : ℝ → ℝ × ℝ) 
  (f : ℝ → ℝ) 
  (A : ℝ) 
  (BC : ℝ) 
  (h1 : P = (Real.sqrt 3, 1))
  (h2 : ∀ x, Q x = (Real.cos x, Real.sin x))
  (h3 : ∀ x, f x = P.1 * (Q x).1 + P.2 * (Q x).2 - ((Q x).1 * P.1 + (Q x).2 * P.2))
  (h4 : f A = 4)
  (h5 : BC = 3) :
  (∀ x, f x = -2 * Real.sin (x + π/3) + 4) ∧ 
  (∀ T > 0, (∀ x, f (x + T) = f x) → T ≥ π) ∧
  (∀ a b c, a + b + c ≤ 3 + 2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l350_35080


namespace NUMINAMATH_CALUDE_sea_glass_collection_l350_35040

/-- Sea glass collection problem -/
theorem sea_glass_collection (blanche_green blanche_red rose_blue dorothy_total : ℕ) 
  (h1 : blanche_green = 12)
  (h2 : blanche_red = 3)
  (h3 : rose_blue = 11)
  (h4 : dorothy_total = 57) :
  ∃ (rose_red : ℕ),
    dorothy_total = 2 * (blanche_red + rose_red) + 3 * rose_blue ∧ 
    rose_red = 9 := by
  sorry

end NUMINAMATH_CALUDE_sea_glass_collection_l350_35040


namespace NUMINAMATH_CALUDE_weight_sum_l350_35031

theorem weight_sum (m n o p : ℕ) 
  (h1 : m + n = 320)
  (h2 : n + o = 295)
  (h3 : o + p = 310) :
  m + p = 335 := by
  sorry

end NUMINAMATH_CALUDE_weight_sum_l350_35031


namespace NUMINAMATH_CALUDE_simplify_expression_l350_35084

theorem simplify_expression (a : ℝ) (h : 1 < a ∧ a < 2) :
  Real.sqrt ((a - 3)^2) + |1 - a| = 2 := by
sorry

end NUMINAMATH_CALUDE_simplify_expression_l350_35084


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l350_35030

/-- Given that the coefficients of the first three terms in the expansion of (x + 1/(2x))^n form an arithmetic sequence,
    prove that the coefficient of the x^4 term in the expansion is 7. -/
theorem binomial_expansion_coefficient (n : ℕ) : 
  (∃ d : ℚ, (1 : ℚ) = (n.choose 0 : ℚ) ∧ 
             (1/2 : ℚ) * (n.choose 1 : ℚ) = (n.choose 0 : ℚ) + d ∧ 
             (1/4 : ℚ) * (n.choose 2 : ℚ) = (n.choose 0 : ℚ) + 2*d) → 
  (1/4 : ℚ) * (n.choose 4 : ℚ) = 7 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l350_35030


namespace NUMINAMATH_CALUDE_expand_expression_l350_35086

theorem expand_expression (x : ℝ) : (17 * x^2 + 20) * 3 * x^3 = 51 * x^5 + 60 * x^3 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l350_35086


namespace NUMINAMATH_CALUDE_max_value_cosine_fraction_l350_35099

theorem max_value_cosine_fraction :
  ∃ (M : ℝ), M = 3 ∧ ∀ x : ℝ, (2 + Real.cos x) / (2 - Real.cos x) ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_cosine_fraction_l350_35099


namespace NUMINAMATH_CALUDE_complex_multiplication_l350_35087

theorem complex_multiplication : 
  (3 - 4 * Complex.I) * (2 + Complex.I) = 10 - 5 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l350_35087


namespace NUMINAMATH_CALUDE_even_monotone_increasing_neg_implies_f1_gt_fneg2_l350_35024

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def monotone_increasing_on_neg (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y ∧ y ≤ 0 → f x < f y

-- Theorem statement
theorem even_monotone_increasing_neg_implies_f1_gt_fneg2
  (h_even : is_even f)
  (h_mono : monotone_increasing_on_neg f) :
  f 1 > f (-2) :=
sorry

end NUMINAMATH_CALUDE_even_monotone_increasing_neg_implies_f1_gt_fneg2_l350_35024


namespace NUMINAMATH_CALUDE_direct_variation_with_constant_l350_35005

/-- A function that varies directly as x plus a constant -/
def f (k c : ℝ) (x : ℝ) : ℝ := k * x + c

/-- Theorem stating that if f(5) = 10 and f(1) = 6, then f(7) = 12 -/
theorem direct_variation_with_constant 
  (k c : ℝ) 
  (h1 : f k c 5 = 10) 
  (h2 : f k c 1 = 6) : 
  f k c 7 = 12 := by
  sorry

#check direct_variation_with_constant

end NUMINAMATH_CALUDE_direct_variation_with_constant_l350_35005


namespace NUMINAMATH_CALUDE_turnip_zhuchka_weight_ratio_l350_35090

/-- The weight ratio between Zhuchka and a cat -/
def zhuchka_cat_ratio : ℚ := 3

/-- The weight ratio between a cat and a mouse -/
def cat_mouse_ratio : ℚ := 10

/-- The weight ratio between a turnip and a mouse -/
def turnip_mouse_ratio : ℚ := 60

/-- The weight ratio between a turnip and Zhuchka -/
def turnip_zhuchka_ratio : ℚ := 2

theorem turnip_zhuchka_weight_ratio :
  turnip_mouse_ratio / (cat_mouse_ratio * zhuchka_cat_ratio) = turnip_zhuchka_ratio :=
by sorry

end NUMINAMATH_CALUDE_turnip_zhuchka_weight_ratio_l350_35090


namespace NUMINAMATH_CALUDE_largest_five_digit_sum_20_l350_35015

/-- A function that returns the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A function that checks if a number is five-digit -/
def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

theorem largest_five_digit_sum_20 : 
  ∀ n : ℕ, is_five_digit n → sum_of_digits n = 20 → n ≤ 99200 := by sorry

end NUMINAMATH_CALUDE_largest_five_digit_sum_20_l350_35015


namespace NUMINAMATH_CALUDE_company_kw_price_percentage_l350_35069

/-- The price of company KW as a percentage of the combined assets of companies A and B -/
theorem company_kw_price_percentage (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) : 
  let p := 1.2 * a  -- Price of company KW
  let combined_assets := a + b
  p / combined_assets = 0.75 := by sorry

end NUMINAMATH_CALUDE_company_kw_price_percentage_l350_35069


namespace NUMINAMATH_CALUDE_survey_theorem_l350_35019

/-- Represents the response of a student to a subject --/
inductive Response
| Yes
| No
| Unsure

/-- Represents a subject with its response counts --/
structure Subject where
  yes_count : Nat
  no_count : Nat
  unsure_count : Nat

/-- The survey results --/
structure SurveyResults where
  total_students : Nat
  subject_m : Subject
  subject_r : Subject
  yes_only_m : Nat

def SurveyResults.students_not_yes_either (results : SurveyResults) : Nat :=
  results.total_students - (results.subject_m.yes_count + results.subject_r.yes_count - results.yes_only_m)

theorem survey_theorem (results : SurveyResults) 
  (h1 : results.total_students = 800)
  (h2 : results.subject_m.yes_count = 500)
  (h3 : results.subject_m.no_count = 200)
  (h4 : results.subject_m.unsure_count = 100)
  (h5 : results.subject_r.yes_count = 400)
  (h6 : results.subject_r.no_count = 100)
  (h7 : results.subject_r.unsure_count = 300)
  (h8 : results.yes_only_m = 170) :
  results.students_not_yes_either = 230 := by
  sorry

#eval SurveyResults.students_not_yes_either {
  total_students := 800,
  subject_m := { yes_count := 500, no_count := 200, unsure_count := 100 },
  subject_r := { yes_count := 400, no_count := 100, unsure_count := 300 },
  yes_only_m := 170
}

end NUMINAMATH_CALUDE_survey_theorem_l350_35019


namespace NUMINAMATH_CALUDE_probability_all_female_finalists_l350_35027

def total_contestants : ℕ := 7
def female_contestants : ℕ := 4
def male_contestants : ℕ := 3
def finalists : ℕ := 3

theorem probability_all_female_finalists :
  (Nat.choose female_contestants finalists : ℚ) / (Nat.choose total_contestants finalists : ℚ) = 4 / 35 :=
sorry

end NUMINAMATH_CALUDE_probability_all_female_finalists_l350_35027


namespace NUMINAMATH_CALUDE_bernoullis_inequality_l350_35096

theorem bernoullis_inequality (n : ℕ) (a : ℝ) (h : a > -1) :
  (1 + a)^n ≥ n * a + 1 := by
  sorry

end NUMINAMATH_CALUDE_bernoullis_inequality_l350_35096


namespace NUMINAMATH_CALUDE_triangle_property_l350_35011

theorem triangle_property (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Side lengths are positive
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Law of sines
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C →
  -- Given condition
  Real.sin A ^ 2 - Real.sin B ^ 2 - Real.sin C ^ 2 = Real.sin B * Real.sin C →
  -- BC = 3
  a = 3 →
  -- Prove A = 2π/3
  A = 2 * π / 3 ∧
  -- Prove maximum perimeter is 3 + 2√3
  (b + c ≤ 2 * Real.sqrt 3 ∧ a + b + c ≤ 3 + 2 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_property_l350_35011


namespace NUMINAMATH_CALUDE_bronze_medals_count_l350_35079

theorem bronze_medals_count (total : ℕ) (gold : ℕ) (silver : ℕ) (bronze : ℕ)
  (h_total : total = 67)
  (h_gold : gold = 19)
  (h_silver : silver = 32)
  (h_sum : total = gold + silver + bronze) :
  bronze = 16 :=
by sorry

end NUMINAMATH_CALUDE_bronze_medals_count_l350_35079


namespace NUMINAMATH_CALUDE_triangle_longest_side_l350_35009

theorem triangle_longest_side (x : ℕ) (h1 : x > 0) 
  (h2 : 5 * x + 6 * x + 7 * x = 720) 
  (h3 : 5 * x + 6 * x > 7 * x) 
  (h4 : 5 * x + 7 * x > 6 * x) 
  (h5 : 6 * x + 7 * x > 5 * x) :
  7 * x = 280 := by
  sorry

#check triangle_longest_side

end NUMINAMATH_CALUDE_triangle_longest_side_l350_35009


namespace NUMINAMATH_CALUDE_expenditure_ratio_l350_35091

/-- Given two persons P1 and P2 with the following conditions:
    - The ratio of their incomes is 5:4
    - Each saves Rs. 1800
    - The income of P1 is Rs. 4500
    Prove that the ratio of their expenditures is 3:2 -/
theorem expenditure_ratio (income_p1 income_p2 expenditure_p1 expenditure_p2 savings : ℕ) :
  income_p1 = 4500 ∧
  5 * income_p2 = 4 * income_p1 ∧
  savings = 1800 ∧
  income_p1 - expenditure_p1 = savings ∧
  income_p2 - expenditure_p2 = savings →
  3 * expenditure_p2 = 2 * expenditure_p1 := by
  sorry

end NUMINAMATH_CALUDE_expenditure_ratio_l350_35091


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l350_35076

/-- A hyperbola with foci F₁ and F₂, and a point P on the hyperbola. -/
structure Hyperbola where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  P : ℝ × ℝ

/-- The distance between two points in ℝ² -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- The angle between three points in ℝ² -/
def angle (p q r : ℝ × ℝ) : ℝ := sorry

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

theorem hyperbola_eccentricity (h : Hyperbola) 
  (angle_condition : angle h.F₁ h.P h.F₂ = π / 3)
  (distance_condition : distance h.P h.F₁ = 3 * distance h.P h.F₂) :
  eccentricity h = Real.sqrt 7 / 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l350_35076


namespace NUMINAMATH_CALUDE_simplify_sqrt_difference_l350_35032

theorem simplify_sqrt_difference : Real.sqrt 18 - Real.sqrt 8 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_difference_l350_35032


namespace NUMINAMATH_CALUDE_sixth_term_geometric_sequence_l350_35037

/-- The sixth term of a geometric sequence with first term 5 and second term 1.25 is 5/1024 -/
theorem sixth_term_geometric_sequence (a₁ a₂ : ℝ) (h₁ : a₁ = 5) (h₂ : a₂ = 1.25) :
  let r := a₂ / a₁
  let a₆ := a₁ * r^5
  a₆ = 5 / 1024 := by
sorry

end NUMINAMATH_CALUDE_sixth_term_geometric_sequence_l350_35037


namespace NUMINAMATH_CALUDE_function_properties_l350_35049

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^3 + x^2 + b * x

-- Define the function g
def g (a b x : ℝ) : ℝ := f a b x + (3 * a * x^2 + 2 * x + b)

-- Main theorem
theorem function_properties (a b : ℝ) :
  (∀ x, g a b x = -g a b (-x)) →
  (∃ f_simplified : ℝ → ℝ, 
    (∀ x, f a b x = f_simplified x) ∧
    (∀ x, f_simplified x = x^2 - x) ∧
    (∀ x ∈ Set.Icc 1 2, HasDerivAt (g a b) ((2 : ℝ) * x + 1) x) ∧
    (g a b 1 = 1) ∧
    (g a b 2 = 5) ∧
    (∀ x ∈ Set.Icc 1 2, g a b x ≥ 1 ∧ g a b x ≤ 5)) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l350_35049


namespace NUMINAMATH_CALUDE_expression_is_linear_binomial_when_k_is_3_l350_35000

-- Define the algebraic expression
def algebraic_expression (k x : ℝ) : ℝ :=
  (-3*k*x^2 + x - 1) + (9*x^2 - 4*k*x + 3*k)

-- Define what it means for an expression to be a linear binomial
def is_linear_binomial (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b

-- Theorem statement
theorem expression_is_linear_binomial_when_k_is_3 :
  is_linear_binomial (algebraic_expression 3) :=
sorry

end NUMINAMATH_CALUDE_expression_is_linear_binomial_when_k_is_3_l350_35000


namespace NUMINAMATH_CALUDE_expression_equivalence_l350_35038

theorem expression_equivalence : 
  -44 + 1010 + 66 - 55 = (-44) + 1010 + 66 + (-55) := by
  sorry

end NUMINAMATH_CALUDE_expression_equivalence_l350_35038


namespace NUMINAMATH_CALUDE_cosine_sine_relation_l350_35082

open Real

theorem cosine_sine_relation (α β : ℝ) (x y : ℝ) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : 0 < β ∧ β < π/2) 
  (h3 : cos (α + β) = -4/5)
  (h4 : sin β = x)
  (h5 : cos α = y)
  (h6 : 4/5 < x ∧ x < 1) :
  y = -4/5 * sqrt (1 - x^2) + 3/5 * x := by
sorry

end NUMINAMATH_CALUDE_cosine_sine_relation_l350_35082


namespace NUMINAMATH_CALUDE_memory_card_cost_memory_card_cost_is_60_l350_35001

/-- The cost of a single memory card given the following conditions:
  * John takes 10 pictures daily for 3 years
  * Each memory card stores 50 images
  * The total spent on memory cards is $13,140 -/
theorem memory_card_cost (pictures_per_day : ℕ) (years : ℕ) (images_per_card : ℕ) (total_spent : ℕ) : ℕ :=
  let days_per_year : ℕ := 365
  let total_pictures : ℕ := pictures_per_day * years * days_per_year
  let cards_needed : ℕ := total_pictures / images_per_card
  total_spent / cards_needed

/-- Proof that the cost of each memory card is $60 -/
theorem memory_card_cost_is_60 : memory_card_cost 10 3 50 13140 = 60 := by
  sorry

end NUMINAMATH_CALUDE_memory_card_cost_memory_card_cost_is_60_l350_35001


namespace NUMINAMATH_CALUDE_xy_range_l350_35078

theorem xy_range (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y + x*y = 30) :
  12 < x*y ∧ x*y < 870 := by
  sorry

end NUMINAMATH_CALUDE_xy_range_l350_35078


namespace NUMINAMATH_CALUDE_colored_polygons_equality_l350_35046

/-- A regular polygon with n vertices -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  is_regular : sorry

/-- A coloring of the vertices of a regular polygon -/
def Coloring (n : ℕ) := Fin n → ℕ

/-- The set of vertices of a given color -/
def colorVertices (n : ℕ) (p : RegularPolygon n) (c : Coloring n) (color : ℕ) : Set (ℝ × ℝ) :=
  {v | ∃ i, c i = color ∧ p.vertices i = v}

/-- Predicate to check if a set of vertices forms a regular polygon -/
def isRegularPolygon (vertices : Set (ℝ × ℝ)) : Prop := sorry

theorem colored_polygons_equality (n : ℕ) (p : RegularPolygon n) (c : Coloring n) :
  (∀ color, isRegularPolygon (colorVertices n p c color)) →
  ∃ color1 color2, color1 ≠ color2 ∧ 
    colorVertices n p c color1 = colorVertices n p c color2 := by
  sorry

end NUMINAMATH_CALUDE_colored_polygons_equality_l350_35046


namespace NUMINAMATH_CALUDE_square_area_ratio_l350_35017

/-- The ratio of the area of a square with side length x to the area of a square with side length 3x is 1/9 -/
theorem square_area_ratio (x : ℝ) (hx : x > 0) : 
  (x^2) / ((3*x)^2) = 1 / 9 := by
sorry

end NUMINAMATH_CALUDE_square_area_ratio_l350_35017


namespace NUMINAMATH_CALUDE_incorrect_operation_correction_l350_35043

theorem incorrect_operation_correction (x : ℝ) : 
  x - 4.3 = 8.8 → x + 4.3 = 17.4 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_operation_correction_l350_35043


namespace NUMINAMATH_CALUDE_solution_sum_l350_35036

theorem solution_sum (m : ℝ) (x₁ x₂ : ℝ) : 
  (∃ (m : ℝ), 2 * Real.sin (2 * x₁ + π / 6) = m ∧ 
               2 * Real.sin (2 * x₂ + π / 6) = m) →
  x₁ ≠ x₂ →
  x₁ ∈ Set.Icc 0 (π / 2) →
  x₂ ∈ Set.Icc 0 (π / 2) →
  x₁ + x₂ = π / 3 := by
sorry

end NUMINAMATH_CALUDE_solution_sum_l350_35036


namespace NUMINAMATH_CALUDE_original_houses_count_l350_35060

-- Define the given conditions
def houses_built_during_boom : ℕ := 97741
def current_total_houses : ℕ := 118558

-- Define the theorem to prove
theorem original_houses_count : 
  current_total_houses - houses_built_during_boom = 20817 := by
  sorry

end NUMINAMATH_CALUDE_original_houses_count_l350_35060


namespace NUMINAMATH_CALUDE_laborer_income_l350_35002

/-- The monthly income of a laborer given certain expenditure and savings conditions -/
theorem laborer_income (
  average_expenditure : ℝ)
  (reduced_expenditure : ℝ)
  (months_initial : ℕ)
  (months_reduced : ℕ)
  (savings : ℝ)
  (h1 : average_expenditure = 90)
  (h2 : reduced_expenditure = 60)
  (h3 : months_initial = 6)
  (h4 : months_reduced = 4)
  (h5 : savings = 30)
  : ∃ (income : ℝ) (debt : ℝ),
    income * months_initial = average_expenditure * months_initial - debt ∧
    income * months_reduced = reduced_expenditure * months_reduced + debt + savings ∧
    income = 81 := by
  sorry

end NUMINAMATH_CALUDE_laborer_income_l350_35002


namespace NUMINAMATH_CALUDE_ellipse_focus_m_value_l350_35074

/-- Given an ellipse with equation x²/25 + y²/m² = 1 where m > 0,
    if the left focus is at (-4,0), then m = 3 -/
theorem ellipse_focus_m_value (m : ℝ) :
  m > 0 →
  (∀ x y : ℝ, x^2/25 + y^2/m^2 = 1 → (x + 4)^2 + y^2 = (5 + m)^2) →
  m = 3 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_focus_m_value_l350_35074


namespace NUMINAMATH_CALUDE_club_members_count_l350_35035

def sock_cost : ℝ := 6
def tshirt_cost : ℝ := sock_cost + 7
def cap_cost : ℝ := tshirt_cost - 3
def total_cost_per_member : ℝ := 2 * (sock_cost + tshirt_cost + cap_cost)
def total_club_cost : ℝ := 3630

theorem club_members_count : 
  ∃ n : ℕ, n = 63 ∧ (n : ℝ) * total_cost_per_member = total_club_cost :=
sorry

end NUMINAMATH_CALUDE_club_members_count_l350_35035


namespace NUMINAMATH_CALUDE_binary_multiplication_example_l350_35050

/-- Represents a binary number as a list of booleans, where true represents 1 and false represents 0 -/
def BinaryNumber := List Bool

/-- Converts a binary number to its decimal representation -/
def binary_to_decimal (b : BinaryNumber) : Nat :=
  b.foldl (fun acc digit => 2 * acc + if digit then 1 else 0) 0

/-- Multiplies two binary numbers -/
def binary_multiply (a b : BinaryNumber) : BinaryNumber :=
  sorry

theorem binary_multiplication_example :
  let a : BinaryNumber := [true, true, false, true]  -- 1101₂
  let b : BinaryNumber := [true, true, true]         -- 111₂
  let result : BinaryNumber := [true, false, false, true, true, true, true]  -- 1001111₂
  binary_multiply a b = result :=
sorry

end NUMINAMATH_CALUDE_binary_multiplication_example_l350_35050


namespace NUMINAMATH_CALUDE_stan_boxes_count_l350_35061

theorem stan_boxes_count (john jules joseph stan : ℕ) : 
  john = (120 * jules) / 100 →
  jules = joseph + 5 →
  joseph = (20 * stan) / 100 →
  john = 30 →
  stan = 100 := by
sorry

end NUMINAMATH_CALUDE_stan_boxes_count_l350_35061


namespace NUMINAMATH_CALUDE_modulus_of_z_is_two_l350_35051

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the complex number z
def z : ℂ := by sorry

-- State the theorem
theorem modulus_of_z_is_two :
  z * (2 - 3 * i) = 6 + 4 * i → Complex.abs z = 2 := by sorry

end NUMINAMATH_CALUDE_modulus_of_z_is_two_l350_35051


namespace NUMINAMATH_CALUDE_first_liquid_volume_l350_35063

theorem first_liquid_volume (x : ℝ) : 
  (0.75 * x + 63) / (x + 90) = 0.7263157894736842 → x = 100 := by
  sorry

end NUMINAMATH_CALUDE_first_liquid_volume_l350_35063


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_squared_l350_35020

noncomputable def a : ℝ := Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 6
noncomputable def b : ℝ := -Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 6
noncomputable def c : ℝ := Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 6
noncomputable def d : ℝ := -Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 6

theorem sum_of_reciprocals_squared :
  (1/a + 1/b + 1/c + 1/d)^2 = 16*(11 + 2*Real.sqrt 30) / (11 + 2*Real.sqrt 30 - 3*Real.sqrt 6)^2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_squared_l350_35020


namespace NUMINAMATH_CALUDE_eliminate_y_by_addition_l350_35012

/-- Given a system of two linear equations in two variables x and y,
    prove that adding the first equation to twice the second equation
    eliminates the y variable. -/
theorem eliminate_y_by_addition (a b c d e f : ℝ) :
  let eq1 := (a * x + b * y = e)
  let eq2 := (c * x + d * y = f)
  (b = -2 * d) →
  ∃ k, (a * x + b * y) + 2 * (c * x + d * y) = k * x + e + 2 * f :=
by sorry

end NUMINAMATH_CALUDE_eliminate_y_by_addition_l350_35012


namespace NUMINAMATH_CALUDE_unique_solution_system_l350_35071

theorem unique_solution_system (x y z : ℝ) : 
  (x^2 + 25*y + 19*z = -471) ∧
  (y^2 + 23*x + 21*z = -397) ∧
  (z^2 + 21*x + 21*y = -545) ↔
  (x = -22 ∧ y = -23 ∧ z = -20) := by
sorry

end NUMINAMATH_CALUDE_unique_solution_system_l350_35071


namespace NUMINAMATH_CALUDE_minimum_words_for_90_percent_l350_35029

/-- Represents the French exam vocabulary test -/
structure FrenchExam where
  total_words : ℕ
  learned_words : ℕ
  score_threshold : ℚ

/-- Calculate the score for a given exam -/
def calculate_score (exam : FrenchExam) : ℚ :=
  (exam.learned_words + (exam.total_words - exam.learned_words) / 10) / exam.total_words

/-- Theorem stating the minimum number of words to learn for a 90% score -/
theorem minimum_words_for_90_percent (exam : FrenchExam) 
    (h1 : exam.total_words = 800)
    (h2 : exam.score_threshold = 9/10) :
    (∀ n : ℕ, n < 712 → calculate_score ⟨exam.total_words, n, exam.score_threshold⟩ < exam.score_threshold) ∧
    calculate_score ⟨exam.total_words, 712, exam.score_threshold⟩ ≥ exam.score_threshold :=
  sorry


end NUMINAMATH_CALUDE_minimum_words_for_90_percent_l350_35029


namespace NUMINAMATH_CALUDE_min_number_for_triangle_l350_35028

/-- A function that checks if three numbers can form a triangle -/
def can_form_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The property that any 17 numbers chosen from 1 to 2005 always contain a triangle -/
def always_contains_triangle (n : ℕ) : Prop :=
  ∀ (s : Finset ℕ), s.card = n → (∀ x ∈ s, 1 ≤ x ∧ x ≤ 2005) →
    ∃ a b c, a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ can_form_triangle a b c

/-- The theorem stating that 17 is the minimum number for which the property holds -/
theorem min_number_for_triangle :
  always_contains_triangle 17 ∧ ¬(always_contains_triangle 16) :=
sorry

end NUMINAMATH_CALUDE_min_number_for_triangle_l350_35028


namespace NUMINAMATH_CALUDE_faster_river_longer_time_l350_35008

/-- Proves that the total travel time in a faster river is greater than in a slower river -/
theorem faster_river_longer_time
  (v : ℝ) (v₁ v₂ S : ℝ) 
  (h_v : v > 0) 
  (h_v₁ : v₁ > 0) 
  (h_v₂ : v₂ > 0) 
  (h_S : S > 0)
  (h_v₁_gt_v₂ : v₁ > v₂) 
  (h_v_gt_v₁ : v > v₁) 
  (h_v_gt_v₂ : v > v₂) :
  (2 * S * v) / (v^2 - v₁^2) > (2 * S * v) / (v^2 - v₂^2) :=
by sorry

end NUMINAMATH_CALUDE_faster_river_longer_time_l350_35008


namespace NUMINAMATH_CALUDE_isosceles_triangle_area_l350_35016

/-- An isosceles triangle with given altitude and perimeter -/
structure IsoscelesTriangle where
  altitude : ℝ
  perimeter : ℝ

/-- The area of an isosceles triangle -/
def area (t : IsoscelesTriangle) : ℝ :=
  sorry

/-- Theorem: The area of an isosceles triangle with altitude 10 and perimeter 40 is 75 -/
theorem isosceles_triangle_area :
  ∀ t : IsoscelesTriangle, t.altitude = 10 ∧ t.perimeter = 40 → area t = 75 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_area_l350_35016


namespace NUMINAMATH_CALUDE_pascal_row_10_sum_l350_35067

/-- The sum of the numbers in a row of Pascal's Triangle -/
def pascal_row_sum (n : ℕ) : ℕ := 2^n

/-- Theorem: The sum of the numbers in Row 10 of Pascal's Triangle is 1024 -/
theorem pascal_row_10_sum : pascal_row_sum 10 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_pascal_row_10_sum_l350_35067


namespace NUMINAMATH_CALUDE_potion_kit_cost_is_18_silver_l350_35068

/-- Represents the cost of items in Harry's purchase --/
structure PurchaseCost where
  spellbookCost : ℕ
  owlCost : ℕ
  totalSilver : ℕ
  silverToGold : ℕ

/-- Calculates the cost of each potion kit in silver --/
def potionKitCost (p : PurchaseCost) : ℕ :=
  let totalGold := p.totalSilver / p.silverToGold
  let spellbooksTotalCost := 5 * p.spellbookCost
  let remainingGold := totalGold - spellbooksTotalCost - p.owlCost
  let potionKitGold := remainingGold / 3
  potionKitGold * p.silverToGold

/-- Theorem stating that each potion kit costs 18 silvers --/
theorem potion_kit_cost_is_18_silver (p : PurchaseCost) 
  (h1 : p.spellbookCost = 5)
  (h2 : p.owlCost = 28)
  (h3 : p.totalSilver = 537)
  (h4 : p.silverToGold = 9) : 
  potionKitCost p = 18 := by
  sorry


end NUMINAMATH_CALUDE_potion_kit_cost_is_18_silver_l350_35068


namespace NUMINAMATH_CALUDE_fraction_transformation_l350_35089

theorem fraction_transformation (x y : ℝ) (h1 : x / y = 2 / 5) (h2 : x + y = 5.25) :
  (x + 3) / (2 * y) = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_fraction_transformation_l350_35089


namespace NUMINAMATH_CALUDE_function_derivative_problem_l350_35088

/-- Given a function f(x) = x(x+k)(x+2k)(x-3k) where f'(0) = 6, prove that k = -1 -/
theorem function_derivative_problem (k : ℝ) : 
  (∃ f : ℝ → ℝ, (∀ x, f x = x * (x + k) * (x + 2*k) * (x - 3*k)) ∧ 
   (deriv f) 0 = 6) → 
  k = -1 := by
  sorry

end NUMINAMATH_CALUDE_function_derivative_problem_l350_35088


namespace NUMINAMATH_CALUDE_chord_equation_l350_35023

/-- Given an ellipse and a point M, prove the equation of the line containing the chord with midpoint M -/
theorem chord_equation (x y : ℝ) :
  (x^2 / 4 + y^2 = 1) →  -- Ellipse equation
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    (x₁^2 / 4 + y₁^2 = 1) ∧  -- Point (x₁, y₁) is on the ellipse
    (x₂^2 / 4 + y₂^2 = 1) ∧  -- Point (x₂, y₂) is on the ellipse
    ((x₁ + x₂) / 2 = 1) ∧    -- x-coordinate of midpoint M
    ((y₁ + y₂) / 2 = 1/2) ∧  -- y-coordinate of midpoint M
    (y - 1/2 = -(1/2) * (x - 1))) →  -- Equation of the line through M with slope -1/2
  x + 2*y - 2 = 0  -- Resulting equation of the line
:= by sorry

end NUMINAMATH_CALUDE_chord_equation_l350_35023


namespace NUMINAMATH_CALUDE_equation_transformation_l350_35045

variable (x y : ℝ)

theorem equation_transformation (h : y = x + 1/x) :
  x^4 - x^3 - 6*x^2 - x + 1 = 0 ↔ x^2 * (y^2 - y - 6) = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_transformation_l350_35045


namespace NUMINAMATH_CALUDE_all_PQ_pass_through_common_point_l350_35010

-- Define the circle S
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a line
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

-- Define the problem setup
structure Setup where
  S : Circle
  A : ℝ × ℝ
  B : ℝ × ℝ
  L : Line
  c : ℝ

-- Define the condition for X and Y
def satisfiesCondition (setup : Setup) (X Y : ℝ × ℝ) : Prop :=
  X ≠ Y ∧ 
  (X.1 - setup.A.1) * (Y.1 - setup.A.1) + (X.2 - setup.A.2) * (Y.2 - setup.A.2) = setup.c

-- Define the intersection points P and Q
def getIntersectionP (setup : Setup) (X : ℝ × ℝ) : ℝ × ℝ := sorry
def getIntersectionQ (setup : Setup) (Y : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the line PQ
def linePQ (P Q : ℝ × ℝ) : Line := ⟨P, Q⟩

-- Theorem statement
theorem all_PQ_pass_through_common_point (setup : Setup) :
  ∃ (commonPoint : ℝ × ℝ), ∀ (X Y : ℝ × ℝ),
    satisfiesCondition setup X Y →
    let P := getIntersectionP setup X
    let Q := getIntersectionQ setup Y
    let PQ := linePQ P Q
    -- The common point lies on line PQ
    (commonPoint.1 - PQ.point1.1) * (PQ.point2.2 - PQ.point1.2) = 
    (commonPoint.2 - PQ.point1.2) * (PQ.point2.1 - PQ.point1.1) :=
sorry

end NUMINAMATH_CALUDE_all_PQ_pass_through_common_point_l350_35010


namespace NUMINAMATH_CALUDE_range_of_m_l350_35054

def p (x : ℝ) : Prop := |x - 4| ≤ 6

def q (x m : ℝ) : Prop := x^2 - m^2 - 2*x + 1 ≤ 0

theorem range_of_m (m : ℝ) :
  (m > 0) →
  (∀ x, q x m → p x) →
  (∃ x, p x ∧ ¬q x m) →
  m ∈ Set.Ici 9 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l350_35054


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l350_35021

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n : ℕ, a (n + 1) = a n * q) →  -- Geometric sequence condition
  q ≠ 1 →  -- Common ratio not equal to 1
  a 1 * a 2 * a 3 = -1/8 →  -- Product of first three terms
  2 * a 4 = a 2 + a 3 →  -- Arithmetic sequence condition
  a 1 + a 2 + a 3 + a 4 = 5/8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l350_35021


namespace NUMINAMATH_CALUDE_sqrt_81_equals_9_l350_35034

theorem sqrt_81_equals_9 : Real.sqrt 81 = 9 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_81_equals_9_l350_35034


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_arithmetic_progression_l350_35058

/-- A triangle with sides in arithmetic progression has an inscribed circle
    with radius equal to 1/3 of one of its heights. -/
theorem inscribed_circle_radius_arithmetic_progression (a d : ℝ) (h : ℝ) :
  let sides := [a, a + d, a + 2*d]
  let s := (a + (a + d) + (a + 2*d)) / 2
  let area := (a + d) * h / 2
  let r := area / s
  r = h / 3 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_arithmetic_progression_l350_35058


namespace NUMINAMATH_CALUDE_product_calculation_l350_35072

theorem product_calculation : (1/2 : ℚ) * 8 * (1/8 : ℚ) * 32 * (1/32 : ℚ) * 128 * (1/128 : ℚ) * 512 * (1/512 : ℚ) * 2048 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_product_calculation_l350_35072


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l350_35042

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x : ℝ, p x) ↔ (∀ x : ℝ, ¬ p x) := by sorry

theorem negation_of_proposition :
  (¬ ∃ x : ℝ, x > Real.sin x) ↔ (∀ x : ℝ, x ≤ Real.sin x) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l350_35042


namespace NUMINAMATH_CALUDE_building_cleaning_earnings_l350_35039

/-- Calculates the total earnings for cleaning a building -/
def total_earnings (floors : ℕ) (rooms_per_floor : ℕ) (hours_per_room : ℕ) (hourly_rate : ℕ) : ℕ :=
  floors * rooms_per_floor * hours_per_room * hourly_rate

/-- Proves that the total earnings for cleaning the specified building is $3600 -/
theorem building_cleaning_earnings :
  total_earnings 4 10 6 15 = 3600 := by
  sorry

#eval total_earnings 4 10 6 15

end NUMINAMATH_CALUDE_building_cleaning_earnings_l350_35039


namespace NUMINAMATH_CALUDE_journey_time_is_41_hours_l350_35077

-- Define the flight and layover times
def flight_NO_ATL : ℝ := 2
def layover_ATL : ℝ := 4
def flight_ATL_CHI : ℝ := 5
def layover_CHI : ℝ := 3
def flight_CHI_NY : ℝ := 3
def layover_NY : ℝ := 16
def flight_NY_SF : ℝ := 24

-- Define the total time from New Orleans to New York
def time_NO_NY : ℝ := flight_NO_ATL + layover_ATL + flight_ATL_CHI + layover_CHI + flight_CHI_NY

-- Define the total journey time
def total_journey_time : ℝ := time_NO_NY + layover_NY + flight_NY_SF

-- Theorem to prove
theorem journey_time_is_41_hours : total_journey_time = 41 := by
  sorry

end NUMINAMATH_CALUDE_journey_time_is_41_hours_l350_35077


namespace NUMINAMATH_CALUDE_derivative_sin_cos_l350_35094

theorem derivative_sin_cos (x : Real) :
  deriv (fun x => 3 * Real.sin x - 4 * Real.cos x) x = 3 * Real.cos x + 4 * Real.sin x := by
  sorry

end NUMINAMATH_CALUDE_derivative_sin_cos_l350_35094


namespace NUMINAMATH_CALUDE_special_die_probability_sum_l350_35097

/-- Represents a die with special probability distribution -/
structure SpecialDie where
  /-- Probability of rolling an odd number -/
  odd_prob : ℚ
  /-- Probability of rolling an even number -/
  even_prob : ℚ
  /-- Ensure even probability is twice odd probability -/
  even_twice_odd : even_prob = 2 * odd_prob
  /-- Ensure total probability is 1 -/
  total_prob_one : 3 * odd_prob + 3 * even_prob = 1

/-- Calculates the probability of rolling 1, 2, or 3 on the special die -/
def prob_not_exceeding_three (d : SpecialDie) : ℚ :=
  2 * d.odd_prob + d.even_prob

/-- The main theorem stating the sum of numerator and denominator is 13 -/
theorem special_die_probability_sum : 
  ∀ (d : SpecialDie), 
  let p := prob_not_exceeding_three d
  let n := p.den
  let m := p.num
  m + n = 13 := by sorry

end NUMINAMATH_CALUDE_special_die_probability_sum_l350_35097


namespace NUMINAMATH_CALUDE_complex_expression_evaluation_l350_35003

theorem complex_expression_evaluation :
  (7 - 3*Complex.I) - 3*(2 + 4*Complex.I) + (1 + 2*Complex.I) = 2 - 13*Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_expression_evaluation_l350_35003


namespace NUMINAMATH_CALUDE_sufficient_condition_l350_35066

theorem sufficient_condition (x y : ℝ) : x^2 + y^2 < 4 → x*y + 4 > 2*x + 2*y := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_l350_35066


namespace NUMINAMATH_CALUDE_rectangle_area_increase_rectangle_area_percentage_increase_l350_35073

theorem rectangle_area_increase (l w : ℝ) (hl : l > 0) (hw : w > 0) : 
  (1.3 * l) * (1.2 * w) = 1.56 * (l * w) := by
  sorry

theorem rectangle_area_percentage_increase : 
  (1.56 - 1) * 100 = 56 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_increase_rectangle_area_percentage_increase_l350_35073


namespace NUMINAMATH_CALUDE_program_output_l350_35075

theorem program_output (A : ℕ) (h : A = 1) : (((A * 2) * 3) * 4) * 5 = 120 := by
  sorry

end NUMINAMATH_CALUDE_program_output_l350_35075


namespace NUMINAMATH_CALUDE_part1_correct_part2_correct_part3_correct_l350_35007

/-- Represents a coupon type with its discount amount -/
structure CouponType where
  discount : ℕ

/-- The available coupon types -/
def couponTypes : Fin 3 → CouponType
  | 0 => ⟨100⟩  -- A Type
  | 1 => ⟨68⟩   -- B Type
  | 2 => ⟨20⟩   -- C Type

/-- Calculate the total discount from using multiple coupons -/
def totalDiscount (coupons : Fin 3 → ℕ) : ℕ :=
  (coupons 0) * (couponTypes 0).discount +
  (coupons 1) * (couponTypes 1).discount +
  (coupons 2) * (couponTypes 2).discount

/-- Theorem for part 1 -/
theorem part1_correct :
  totalDiscount ![1, 5, 4] = 520 := by sorry

/-- Theorem for part 2 -/
theorem part2_correct :
  totalDiscount ![2, 3, 0] = 404 := by sorry

/-- Helper function to check if a combination is valid -/
def isValidCombination (a b c : ℕ) : Prop :=
  a ≤ 16 ∧ b ≤ 16 ∧ c ≤ 16 ∧
  ((a > 0 ∧ b > 0 ∧ c = 0) ∨
   (a > 0 ∧ b = 0 ∧ c > 0) ∨
   (a = 0 ∧ b > 0 ∧ c > 0))

/-- Theorem for part 3 -/
theorem part3_correct :
  (∀ a b c : ℕ,
    isValidCombination a b c ∧ totalDiscount ![a, b, c] = 708 →
    (a = 3 ∧ b = 6 ∧ c = 0) ∨ (a = 0 ∧ b = 6 ∧ c = 15)) ∧
  isValidCombination 3 6 0 ∧
  totalDiscount ![3, 6, 0] = 708 ∧
  isValidCombination 0 6 15 ∧
  totalDiscount ![0, 6, 15] = 708 := by sorry

end NUMINAMATH_CALUDE_part1_correct_part2_correct_part3_correct_l350_35007


namespace NUMINAMATH_CALUDE_frac_two_x_gt_one_sufficient_not_necessary_for_x_lt_two_l350_35018

theorem frac_two_x_gt_one_sufficient_not_necessary_for_x_lt_two :
  (∃ x : ℝ, 2 / x > 1 ∧ x < 2) ∧
  (∃ x : ℝ, x < 2 ∧ ¬(2 / x > 1)) :=
by
  sorry

end NUMINAMATH_CALUDE_frac_two_x_gt_one_sufficient_not_necessary_for_x_lt_two_l350_35018


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l350_35083

theorem binomial_expansion_coefficient (x : ℝ) (a : Fin 9 → ℝ) :
  (x - 1)^8 = a 0 + a 1 * (1 + x) + a 2 * (1 + x)^2 + a 3 * (1 + x)^3 + 
              a 4 * (1 + x)^4 + a 5 * (1 + x)^5 + a 6 * (1 + x)^6 + 
              a 7 * (1 + x)^7 + a 8 * (1 + x)^8 →
  a 5 = -448 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l350_35083


namespace NUMINAMATH_CALUDE_cosine_inequality_solution_l350_35065

theorem cosine_inequality_solution (y : Real) : 
  (y ∈ Set.Icc 0 Real.pi) → 
  ((∀ x ∈ Set.Icc 0 Real.pi, Real.cos (x + y) ≥ Real.cos x * Real.cos y) ↔ 
   (y = 0 ∨ y = Real.pi)) := by
  sorry

end NUMINAMATH_CALUDE_cosine_inequality_solution_l350_35065


namespace NUMINAMATH_CALUDE_red_jellybean_count_l350_35081

/-- Given a jar of jellybeans with specific counts for different colors, 
    prove that the number of red jellybeans is 120. -/
theorem red_jellybean_count (total : ℕ) (blue purple orange : ℕ) 
  (h_total : total = 200)
  (h_blue : blue = 14)
  (h_purple : purple = 26)
  (h_orange : orange = 40) :
  total - (blue + purple + orange) = 120 := by
  sorry

end NUMINAMATH_CALUDE_red_jellybean_count_l350_35081


namespace NUMINAMATH_CALUDE_gcd_of_powers_of_47_plus_one_l350_35033

theorem gcd_of_powers_of_47_plus_one (h : Prime 47) :
  Nat.gcd (47^5 + 1) (47^5 + 47^3 + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_powers_of_47_plus_one_l350_35033


namespace NUMINAMATH_CALUDE_josh_paid_six_dollars_l350_35095

/-- The amount Josh paid for string cheese -/
def string_cheese_cost (packs : ℕ) (cheeses_per_pack : ℕ) (cents_per_cheese : ℕ) : ℚ :=
  (packs * cheeses_per_pack * cents_per_cheese : ℚ) / 100

/-- Theorem stating that Josh paid 6 dollars for the string cheese -/
theorem josh_paid_six_dollars :
  string_cheese_cost 3 20 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_josh_paid_six_dollars_l350_35095


namespace NUMINAMATH_CALUDE_square_perimeter_l350_35098

theorem square_perimeter (area : ℝ) (side : ℝ) (perimeter : ℝ) : 
  area = 144 → 
  area = side * side →
  perimeter = 4 * side →
  perimeter = 48 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_l350_35098


namespace NUMINAMATH_CALUDE_even_function_sum_l350_35014

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 4

-- Define the property of an even function
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- State the theorem
theorem even_function_sum (a b : ℝ) :
  (∀ x ∈ Set.Icc b 3, f a x = f a (-x)) →
  is_even (f a) →
  a + b = -3 :=
sorry

end NUMINAMATH_CALUDE_even_function_sum_l350_35014


namespace NUMINAMATH_CALUDE_fishing_problem_l350_35044

theorem fishing_problem (jason ryan jeffery : ℕ) : 
  ryan = 3 * jason →
  jason + ryan + jeffery = 100 →
  jeffery = 60 →
  ryan = 30 := by sorry

end NUMINAMATH_CALUDE_fishing_problem_l350_35044


namespace NUMINAMATH_CALUDE_integer_fraction_l350_35004

theorem integer_fraction (a : ℕ+) : 
  (↑(2 * a.val + 8) / ↑(a.val + 1) : ℚ).isInt ↔ a.val = 1 ∨ a.val = 2 ∨ a.val = 5 := by
  sorry

end NUMINAMATH_CALUDE_integer_fraction_l350_35004


namespace NUMINAMATH_CALUDE_miriam_pushups_l350_35092

/-- Miriam's push-up challenge over a week --/
theorem miriam_pushups (monday tuesday wednesday thursday friday : ℕ) : 
  monday = 5 ∧ 
  wednesday = 2 * tuesday ∧
  thursday = (monday + tuesday + wednesday) / 2 ∧
  friday = monday + tuesday + wednesday + thursday ∧
  friday = 39 →
  tuesday = 7 := by
  sorry

end NUMINAMATH_CALUDE_miriam_pushups_l350_35092


namespace NUMINAMATH_CALUDE_seating_arrangements_l350_35055

def number_of_people : ℕ := 10
def table_seats : ℕ := 8

def alice_bob_block : ℕ := 1
def other_individuals : ℕ := table_seats - 2

def ways_to_choose : ℕ := Nat.choose number_of_people table_seats
def ways_to_arrange_units : ℕ := Nat.factorial (other_individuals + alice_bob_block - 1)
def ways_to_arrange_alice_bob : ℕ := 2

theorem seating_arrangements :
  ways_to_choose * ways_to_arrange_units * ways_to_arrange_alice_bob = 64800 :=
sorry

end NUMINAMATH_CALUDE_seating_arrangements_l350_35055


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l350_35070

theorem fractional_equation_solution : 
  ∃ x : ℝ, (1 / x = 2 / (x + 3)) ∧ (x = 3) := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l350_35070


namespace NUMINAMATH_CALUDE_divisible_by_64_l350_35056

theorem divisible_by_64 (n : ℕ) (h : n > 0) : ∃ k : ℤ, 3^(2*n + 2) - 8*n - 9 = 64*k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_64_l350_35056


namespace NUMINAMATH_CALUDE_solution_set_part_I_solution_part_II_l350_35041

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + 3 * x

-- Part I
theorem solution_set_part_I :
  {x : ℝ | f 1 x ≥ 3 * x + 2} = {x : ℝ | x ≥ 3 ∨ x ≤ -1} := by sorry

-- Part II
theorem solution_part_II (a : ℝ) (h : a > 0) :
  ({x : ℝ | f a x ≤ 0} = {x : ℝ | x ≤ -3}) → a = 6 := by sorry

end NUMINAMATH_CALUDE_solution_set_part_I_solution_part_II_l350_35041


namespace NUMINAMATH_CALUDE_binary_10001000_to_octal_l350_35048

def binary_to_octal (b : ℕ) : ℕ := sorry

theorem binary_10001000_to_octal :
  binary_to_octal 0b10001000 = 0o210 := by sorry

end NUMINAMATH_CALUDE_binary_10001000_to_octal_l350_35048


namespace NUMINAMATH_CALUDE_complex_magnitude_thirteen_l350_35064

theorem complex_magnitude_thirteen (x : ℝ) : 
  x > 0 → (Complex.abs (3 + x * Complex.I) = 13 ↔ x = 8 * Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_complex_magnitude_thirteen_l350_35064


namespace NUMINAMATH_CALUDE_oranges_problem_l350_35022

def oranges_left (mary jason tom sarah : ℕ) : ℕ :=
  let initial_total := mary + jason + tom + sarah
  let increased_total := (initial_total * 110 + 50) / 100  -- Rounded up
  (increased_total * 85 + 50) / 100  -- Rounded down

theorem oranges_problem (mary jason tom sarah : ℕ) 
  (h_mary : mary = 122)
  (h_jason : jason = 105)
  (h_tom : tom = 85)
  (h_sarah : sarah = 134) :
  oranges_left mary jason tom sarah = 417 := by
  sorry

end NUMINAMATH_CALUDE_oranges_problem_l350_35022


namespace NUMINAMATH_CALUDE_pirate_treasure_probability_l350_35006

theorem pirate_treasure_probability :
  let n_islands : ℕ := 8
  let n_treasure : ℕ := 4
  let p_treasure : ℚ := 1/3
  let p_traps : ℚ := 1/6
  let p_neither : ℚ := 1/2
  let choose := fun (n k : ℕ) => (Nat.choose n k : ℚ)
  
  (choose n_islands n_treasure) * p_treasure^n_treasure * p_neither^(n_islands - n_treasure) = 35/648 :=
by sorry

end NUMINAMATH_CALUDE_pirate_treasure_probability_l350_35006


namespace NUMINAMATH_CALUDE_koala_fiber_consumption_l350_35053

/-- The absorption rate of fiber for koalas -/
def koala_absorption_rate : ℝ := 0.30

/-- The amount of fiber absorbed by the koala in one day (in ounces) -/
def fiber_absorbed : ℝ := 12

/-- Theorem: If a koala absorbs 30% of the fiber it eats and it absorbed 12 ounces of fiber in one day, 
    then the total amount of fiber it ate that day was 40 ounces. -/
theorem koala_fiber_consumption :
  fiber_absorbed = koala_absorption_rate * 40 := by
  sorry

end NUMINAMATH_CALUDE_koala_fiber_consumption_l350_35053


namespace NUMINAMATH_CALUDE_negation_existence_quadratic_l350_35062

theorem negation_existence_quadratic (m : ℝ) :
  (¬ ∃ x : ℝ, x^2 + 2*x + m ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + m > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_existence_quadratic_l350_35062
