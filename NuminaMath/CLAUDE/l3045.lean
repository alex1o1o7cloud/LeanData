import Mathlib

namespace NUMINAMATH_CALUDE_best_fit_highest_abs_r_model1_best_fit_l3045_304533

/-- Represents a linear regression model with its correlation coefficient -/
structure RegressionModel where
  r : ℝ

/-- Determines if a model is the best fit among a list of models -/
def isBestFit (model : RegressionModel) (models : List RegressionModel) : Prop :=
  ∀ m ∈ models, |model.r| ≥ |m.r|

theorem best_fit_highest_abs_r (models : List RegressionModel) (model : RegressionModel) 
    (h : model ∈ models) :
    isBestFit model models ↔ ∀ m ∈ models, |model.r| ≥ |m.r| := by
  sorry

/-- The four models from the problem -/
def model1 : RegressionModel := ⟨0.98⟩
def model2 : RegressionModel := ⟨0.80⟩
def model3 : RegressionModel := ⟨0.50⟩
def model4 : RegressionModel := ⟨0.25⟩

def allModels : List RegressionModel := [model1, model2, model3, model4]

theorem model1_best_fit : isBestFit model1 allModels := by
  sorry

end NUMINAMATH_CALUDE_best_fit_highest_abs_r_model1_best_fit_l3045_304533


namespace NUMINAMATH_CALUDE_x_convergence_l3045_304570

def x : ℕ → ℚ
  | 0 => 7
  | n + 1 => (x n ^ 2 + 6 * x n + 5) / (x n + 7)

theorem x_convergence :
  ∃ m : ℕ, 130 ≤ m ∧ m ≤ 240 ∧ x m ≤ 5 + 1 / 2^21 ∧
  ∀ k : ℕ, k < m → x k > 5 + 1 / 2^21 :=
sorry

end NUMINAMATH_CALUDE_x_convergence_l3045_304570


namespace NUMINAMATH_CALUDE_zoo_visitors_l3045_304557

theorem zoo_visitors (saturday_visitors : ℕ) (day_visitors : ℕ) : 
  saturday_visitors = 3750 → 
  saturday_visitors = 3 * day_visitors → 
  day_visitors = 1250 := by
sorry

end NUMINAMATH_CALUDE_zoo_visitors_l3045_304557


namespace NUMINAMATH_CALUDE_sequence_sum_theorem_l3045_304554

/-- The sum of the sequence 1+2-3-4+5+6-7-8+...+2017+2018-2019-2020 -/
def sequenceSum : ℤ := -2020

/-- The last term in the sequence -/
def lastTerm : ℕ := 2020

/-- The number of complete groups of four in the sequence -/
def groupCount : ℕ := lastTerm / 4

/-- The sum of each group of four terms in the sequence -/
def groupSum : ℤ := -4

theorem sequence_sum_theorem :
  sequenceSum = groupCount * groupSum :=
by sorry

end NUMINAMATH_CALUDE_sequence_sum_theorem_l3045_304554


namespace NUMINAMATH_CALUDE_age_ratio_proof_l3045_304515

-- Define the present ages of Lewis and Brown
def lewis_age : ℚ := 2
def brown_age : ℚ := 4

-- Define the conditions
theorem age_ratio_proof :
  -- Condition 1: Present ages are in ratio 1:2
  lewis_age / brown_age = 1 / 2 →
  -- Condition 2: Combined present age is 6
  lewis_age + brown_age = 6 →
  -- Prove: Ratio of ages three years from now is 5:7
  (lewis_age + 3) / (brown_age + 3) = 5 / 7 := by
sorry


end NUMINAMATH_CALUDE_age_ratio_proof_l3045_304515


namespace NUMINAMATH_CALUDE_article_selling_prices_l3045_304503

/-- Represents the selling price of an article -/
def selling_price (cost_price : ℚ) (profit_percentage : ℚ) : ℚ :=
  cost_price * (1 + profit_percentage / 100)

/-- Theorem stating the selling prices of three articles given their cost prices and profit/loss percentages -/
theorem article_selling_prices :
  let article1_sp := 500
  let article2_cp := 800
  let article3_cp := 1800
  (selling_price (article1_sp / 1.25) 25 = article1_sp) ∧
  (selling_price article2_cp (-25) = 600) ∧
  (selling_price article3_cp 50 = 2700) := by
  sorry


end NUMINAMATH_CALUDE_article_selling_prices_l3045_304503


namespace NUMINAMATH_CALUDE_only_airplane_survey_comprehensive_l3045_304562

/-- Represents a type of survey --/
inductive SurveyType
  | WaterQuality
  | AirplanePassengers
  | PlasticBags
  | TVViewership

/-- Predicate to determine if a survey type is suitable for comprehensive surveying --/
def is_comprehensive (s : SurveyType) : Prop :=
  match s with
  | SurveyType.AirplanePassengers => true
  | _ => false

/-- Theorem stating that only the airplane passenger survey is comprehensive --/
theorem only_airplane_survey_comprehensive :
  ∀ s : SurveyType, is_comprehensive s ↔ s = SurveyType.AirplanePassengers :=
by
  sorry

#check only_airplane_survey_comprehensive

end NUMINAMATH_CALUDE_only_airplane_survey_comprehensive_l3045_304562


namespace NUMINAMATH_CALUDE_total_jumps_equals_1085_l3045_304530

def ronald_jumps : ℕ := 157

def rupert_jumps : ℕ := 3 * ronald_jumps + 23

def rebecca_initial_jumps : ℕ := 47
def rebecca_common_difference : ℕ := 5
def rebecca_sequences : ℕ := 7

def rebecca_last_jumps : ℕ := rebecca_initial_jumps + (rebecca_sequences - 1) * rebecca_common_difference

def rebecca_total_jumps : ℕ := rebecca_sequences * (rebecca_initial_jumps + rebecca_last_jumps) / 2

def total_jumps : ℕ := ronald_jumps + rupert_jumps + rebecca_total_jumps

theorem total_jumps_equals_1085 : total_jumps = 1085 := by
  sorry

end NUMINAMATH_CALUDE_total_jumps_equals_1085_l3045_304530


namespace NUMINAMATH_CALUDE_decimal_to_fraction_sum_l3045_304501

theorem decimal_to_fraction_sum (c d : ℕ+) : 
  (c : ℚ) / (d : ℚ) = 0.325 ∧ 
  (∀ (k : ℕ+), k ∣ c ∧ k ∣ d → k = 1) →
  (c : ℕ) + d = 53 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_sum_l3045_304501


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l3045_304528

theorem arithmetic_mean_problem (a b c d : ℝ) : 
  (a + b + c + d) / 4 = 17 →
  a = 12 →
  b = 20 →
  c = d →
  c * d = 324 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l3045_304528


namespace NUMINAMATH_CALUDE_inscribed_polygon_sides_l3045_304521

theorem inscribed_polygon_sides (n : ℕ) (s : ℝ) : n ≥ 3 →
  s = 2 * Real.sin (Real.pi / n) →  -- side length formula
  1 < s →
  s < Real.sqrt 2 →
  n = 5 := by sorry

end NUMINAMATH_CALUDE_inscribed_polygon_sides_l3045_304521


namespace NUMINAMATH_CALUDE_triangle_properties_l3045_304509

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  (2 * t.b - t.c) * Real.cos t.A = t.a * Real.cos t.C ∧
  t.a = Real.sqrt 13 ∧
  t.b + t.c = 5

-- Theorem statement
theorem triangle_properties (t : Triangle) 
  (h : satisfies_conditions t) : 
  t.A = π / 3 ∧ 
  (1 / 2 : ℝ) * t.b * t.c * Real.sin t.A = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3045_304509


namespace NUMINAMATH_CALUDE_max_unique_subsets_l3045_304561

theorem max_unique_subsets (n : ℕ) (h : n = 7) : 
  (2 ^ n) - 2 = 128 := by
  sorry

end NUMINAMATH_CALUDE_max_unique_subsets_l3045_304561


namespace NUMINAMATH_CALUDE_debt_payment_average_l3045_304556

theorem debt_payment_average (total_payments : ℕ) (first_payment_count : ℕ) (first_payment_amount : ℚ) (additional_amount : ℚ) : 
  total_payments = 65 ∧ 
  first_payment_count = 20 ∧ 
  first_payment_amount = 410 ∧ 
  additional_amount = 65 → 
  (first_payment_count * first_payment_amount + 
   (total_payments - first_payment_count) * (first_payment_amount + additional_amount)) / total_payments = 455 := by
sorry

end NUMINAMATH_CALUDE_debt_payment_average_l3045_304556


namespace NUMINAMATH_CALUDE_jellybean_bags_l3045_304516

theorem jellybean_bags (initial_average : ℕ) (new_bag_jellybeans : ℕ) (new_average : ℕ) :
  initial_average = 117 →
  new_bag_jellybeans = 362 →
  new_average = 124 →
  ∃ n : ℕ, n * initial_average + new_bag_jellybeans = (n + 1) * new_average ∧ n = 34 :=
by
  sorry

end NUMINAMATH_CALUDE_jellybean_bags_l3045_304516


namespace NUMINAMATH_CALUDE_functional_equation_properties_l3045_304512

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * y) = y^2 * f x + x^2 * f y

theorem functional_equation_properties (f : ℝ → ℝ) (h : FunctionalEquation f) :
  f 0 = 0 ∧ f 1 = 0 ∧ (∀ x : ℝ, f (-x) = f x) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_properties_l3045_304512


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_l3045_304567

theorem rectangle_area_diagonal (l w d : ℝ) (h1 : l / w = 5 / 2) (h2 : l^2 + w^2 = d^2) :
  l * w = (10 / 29) * d^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_l3045_304567


namespace NUMINAMATH_CALUDE_product_of_integers_l3045_304506

theorem product_of_integers (x y : ℕ+) 
  (sum_eq : x + y = 22)
  (diff_squares_eq : x^2 - y^2 = 44) :
  x * y = 120 := by
  sorry

end NUMINAMATH_CALUDE_product_of_integers_l3045_304506


namespace NUMINAMATH_CALUDE_quadratic_expansion_constraint_l3045_304581

theorem quadratic_expansion_constraint (a b m : ℤ) :
  (∀ x, (x + a) * (x + b) = x^2 + m*x + 5) →
  (m = 6 ∨ m = -6) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_expansion_constraint_l3045_304581


namespace NUMINAMATH_CALUDE_warehouse_boxes_l3045_304504

theorem warehouse_boxes : 
  ∀ (warehouse1 warehouse2 : ℕ),
  warehouse1 = 2 * warehouse2 →
  warehouse1 + warehouse2 = 600 →
  warehouse1 = 400 := by
sorry

end NUMINAMATH_CALUDE_warehouse_boxes_l3045_304504


namespace NUMINAMATH_CALUDE_max_crates_third_trip_l3045_304577

/-- Given a trailer with a maximum weight capacity and a minimum weight per crate,
    prove the maximum number of crates for the third trip. -/
theorem max_crates_third_trip
  (max_weight : ℕ)
  (min_crate_weight : ℕ)
  (trip1_crates : ℕ)
  (trip2_crates : ℕ)
  (h_max_weight : max_weight = 750)
  (h_min_crate_weight : min_crate_weight = 150)
  (h_trip1 : trip1_crates = 3)
  (h_trip2 : trip2_crates = 4)
  (h_weight_constraint : ∀ n : ℕ, n * min_crate_weight ≤ max_weight → n ≤ trip1_crates ∨ n ≤ trip2_crates ∨ n ≤ max_weight / min_crate_weight) :
  (max_weight / min_crate_weight : ℕ) = 5 :=
sorry

end NUMINAMATH_CALUDE_max_crates_third_trip_l3045_304577


namespace NUMINAMATH_CALUDE_sams_german_shepherds_l3045_304507

theorem sams_german_shepherds (sam_french_bulldogs peter_total : ℕ) 
  (h1 : sam_french_bulldogs = 4)
  (h2 : peter_total = 17)
  (h3 : ∃ (sam_german_shepherds : ℕ), 
    3 * sam_german_shepherds + 2 * sam_french_bulldogs = peter_total) :
  ∃ (sam_german_shepherds : ℕ), sam_german_shepherds = 3 :=
by sorry

end NUMINAMATH_CALUDE_sams_german_shepherds_l3045_304507


namespace NUMINAMATH_CALUDE_geometric_progression_proof_l3045_304596

theorem geometric_progression_proof (a b c d : ℤ) :
  a = 7 ∧ b = -14 ∧ c = 28 ∧ d = -56 →
  (∃ r : ℚ, b = a * r ∧ c = b * r ∧ d = c * r) ∧
  a + d = -49 ∧
  b + c = 14 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_proof_l3045_304596


namespace NUMINAMATH_CALUDE_joyful_joan_problem_l3045_304583

theorem joyful_joan_problem (a b c d e : ℚ) : 
  a = 2 ∧ b = 3 ∧ c = 4 ∧ d = 5 →
  (a + b + c + d + e = a + (b + (c - (d * e)))) →
  e = -5/6 := by
sorry

end NUMINAMATH_CALUDE_joyful_joan_problem_l3045_304583


namespace NUMINAMATH_CALUDE_green_peppers_weight_l3045_304585

/-- The weight of green peppers bought by Hannah's Vegetarian Restaurant -/
def green_peppers : ℝ := 0.3333333333333333

/-- The weight of red peppers bought by Hannah's Vegetarian Restaurant -/
def red_peppers : ℝ := 0.3333333333333333

/-- The total weight of peppers bought by Hannah's Vegetarian Restaurant -/
def total_peppers : ℝ := 0.6666666666666666

/-- Theorem stating that the weight of green peppers is the difference between
    the total weight of peppers and the weight of red peppers -/
theorem green_peppers_weight :
  green_peppers = total_peppers - red_peppers := by
  sorry

end NUMINAMATH_CALUDE_green_peppers_weight_l3045_304585


namespace NUMINAMATH_CALUDE_cos_symmetry_center_l3045_304564

theorem cos_symmetry_center (x : ℝ) :
  let f : ℝ → ℝ := λ x => Real.cos (2 * x + π / 3)
  let center : ℝ × ℝ := (π / 12, 0)
  ∀ t : ℝ, f (center.1 + t) = f (center.1 - t) :=
by sorry

end NUMINAMATH_CALUDE_cos_symmetry_center_l3045_304564


namespace NUMINAMATH_CALUDE_usual_time_to_school_l3045_304541

/-- Given a boy who walks 7/6 of his usual rate and reaches school 5 minutes early,
    prove that his usual time to reach the school is 35 minutes. -/
theorem usual_time_to_school (R : ℝ) (T : ℝ) : 
  R * T = (7/6 * R) * (T - 5) → T = 35 :=
by sorry

end NUMINAMATH_CALUDE_usual_time_to_school_l3045_304541


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_line_l3045_304568

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def Line.isParallelTo (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem line_through_point_parallel_to_line
  (A : Point)
  (given_line : Line)
  (h_A : A.x = 3 ∧ A.y = 2)
  (h_given : given_line.a = 4 ∧ given_line.b = 1 ∧ given_line.c = -2)
  : ∃ (result_line : Line),
    result_line.a = 4 ∧ 
    result_line.b = 1 ∧ 
    result_line.c = -14 ∧
    A.liesOn result_line ∧
    result_line.isParallelTo given_line :=
  sorry

end NUMINAMATH_CALUDE_line_through_point_parallel_to_line_l3045_304568


namespace NUMINAMATH_CALUDE_cyclist_speed_ratio_l3045_304523

theorem cyclist_speed_ratio (D : ℝ) (v_r v_w : ℝ) (t_r t_w : ℝ) : 
  D > 0 → v_r > 0 → v_w > 0 → t_r > 0 → t_w > 0 →
  (2 / 3 : ℝ) * D = v_r * t_r →
  (1 / 3 : ℝ) * D = v_w * t_w →
  t_w = 2 * t_r →
  v_r = 4 * v_w := by
  sorry

#check cyclist_speed_ratio

end NUMINAMATH_CALUDE_cyclist_speed_ratio_l3045_304523


namespace NUMINAMATH_CALUDE_solve_equation_l3045_304518

theorem solve_equation : 
  ∃ x : ℚ, (27 / 4) * x - 18 = 3 * x + 27 ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3045_304518


namespace NUMINAMATH_CALUDE_f_at_five_halves_l3045_304538

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the properties of f
axiom f_symmetry (x : ℝ) : f ((-2) - x) = f ((-2) + x)
axiom f_period (x : ℝ) : f (x + 2) = f x
axiom f_definition (x : ℝ) (h : x ∈ Set.Icc (-3) (-2)) : f x = (x + 2)^2

-- State the theorem to be proved
theorem f_at_five_halves : f (5/2) = 1/4 := by sorry

end NUMINAMATH_CALUDE_f_at_five_halves_l3045_304538


namespace NUMINAMATH_CALUDE_prime_square_sum_equation_l3045_304590

theorem prime_square_sum_equation (p q : ℕ) (hp : Prime p) (hq : Prime q) :
  (∃ (x y z : ℕ), p^(2*x) + q^(2*y) = z^2) ↔ ((p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2)) :=
by sorry

end NUMINAMATH_CALUDE_prime_square_sum_equation_l3045_304590


namespace NUMINAMATH_CALUDE_expression_simplification_l3045_304588

theorem expression_simplification (a b : ℝ) 
  (ha : a = Real.sqrt 3 + 1) 
  (hb : b = Real.sqrt 3 - 1) : 
  ((a^2 / (a - b) - (2*a*b - b^2) / (a - b)) / ((a - b) / (a * b))) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3045_304588


namespace NUMINAMATH_CALUDE_range_of_m_plus_n_l3045_304552

noncomputable def f (m n x : ℝ) : ℝ := 2^x * m + x^2 + n * x

theorem range_of_m_plus_n (m n : ℝ) :
  (∃ x, f m n x = 0) ∧ 
  (∀ x, f m n x = 0 ↔ f m n (f m n x) = 0) →
  0 ≤ m + n ∧ m + n < 4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_plus_n_l3045_304552


namespace NUMINAMATH_CALUDE_lorelai_jellybeans_l3045_304559

def jellybeans_problem (gigi rory luke lane lorelai : ℕ) : Prop :=
  gigi = 15 ∧
  rory = gigi + 30 ∧
  luke = 2 * rory ∧
  lane = gigi + 10 ∧
  lorelai = 3 * (gigi + luke + lane)

theorem lorelai_jellybeans :
  ∀ gigi rory luke lane lorelai : ℕ,
  jellybeans_problem gigi rory luke lane lorelai →
  lorelai = 390 :=
by
  sorry

end NUMINAMATH_CALUDE_lorelai_jellybeans_l3045_304559


namespace NUMINAMATH_CALUDE_ellipse_range_l3045_304548

-- Define the ellipse
def on_ellipse (x y : ℝ) : Prop := 2 * x^2 + 3 * y^2 = 12

-- Define the function we're interested in
def f (x y : ℝ) : ℝ := x + 2 * y

-- Theorem statement
theorem ellipse_range :
  ∀ x y : ℝ, on_ellipse x y → -Real.sqrt 22 ≤ f x y ∧ f x y ≤ Real.sqrt 22 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_range_l3045_304548


namespace NUMINAMATH_CALUDE_cricketer_average_increase_l3045_304500

theorem cricketer_average_increase (total_innings : ℕ) (last_inning_score : ℕ) (final_average : ℚ) : 
  total_innings = 19 → 
  last_inning_score = 98 → 
  final_average = 26 → 
  (final_average - (total_innings * final_average - last_inning_score) / (total_innings - 1)) = 4 := by
sorry

end NUMINAMATH_CALUDE_cricketer_average_increase_l3045_304500


namespace NUMINAMATH_CALUDE_cherry_bag_cost_l3045_304526

/-- The cost of a four-pound bag of cherries -/
def cherry_cost : ℝ := 13.5

/-- The cost of the pie crust ingredients -/
def crust_cost : ℝ := 4.5

/-- The total cost of the cheapest pie -/
def cheapest_pie_cost : ℝ := 18

/-- The cost of the blueberry pie -/
def blueberry_pie_cost : ℝ := 18

theorem cherry_bag_cost : 
  cherry_cost = cheapest_pie_cost - crust_cost ∧ 
  blueberry_pie_cost = cheapest_pie_cost :=
by sorry

end NUMINAMATH_CALUDE_cherry_bag_cost_l3045_304526


namespace NUMINAMATH_CALUDE_three_prime_divisors_of_eight_power_minus_one_l3045_304555

theorem three_prime_divisors_of_eight_power_minus_one (n : ℕ) :
  let x := 8^n - 1
  (∃ p q r : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
    x = p * q * r) →
  (31 ∣ x) →
  x = 32767 := by
sorry

end NUMINAMATH_CALUDE_three_prime_divisors_of_eight_power_minus_one_l3045_304555


namespace NUMINAMATH_CALUDE_brad_probability_l3045_304514

/-- Represents the outcome of answering a math problem -/
inductive Answer
| correct
| incorrect

/-- Represents a sequence of answers to math problems -/
def AnswerSequence := List Answer

/-- Calculates the probability of a specific answer sequence -/
def probability (seq : AnswerSequence) : Real :=
  sorry

/-- Counts the number of correct answers in a sequence -/
def countCorrect (seq : AnswerSequence) : Nat :=
  sorry

/-- Generates all possible answer sequences for the remaining 8 problems -/
def generateSequences : List AnswerSequence :=
  sorry

theorem brad_probability :
  let allSequences := generateSequences
  let validSequences := allSequences.filter (λ seq => countCorrect (Answer.correct :: Answer.incorrect :: seq) = 5)
  (validSequences.map probability).sum = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_brad_probability_l3045_304514


namespace NUMINAMATH_CALUDE_power_of_two_divisibility_l3045_304508

theorem power_of_two_divisibility (n a b : ℕ) : 
  2^n = 10*a + b → n > 3 → 0 < b → b < 10 → ∃ k, ab = 6*k := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_divisibility_l3045_304508


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l3045_304531

universe u

def U : Set ℕ := {x | x < 6}

def A : Set ℕ := {1, 2, 3}

def B : Set ℕ := {2, 4, 5}

theorem complement_A_intersect_B :
  (U \ A) ∩ B = {4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l3045_304531


namespace NUMINAMATH_CALUDE_no_integer_roots_l3045_304536

theorem no_integer_roots (a b c : ℤ) (h_a : a ≠ 0) 
  (h_f0 : Odd (c)) 
  (h_f1 : Odd (a + b + c)) : 
  ∀ x : ℤ, a * x^2 + b * x + c ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_no_integer_roots_l3045_304536


namespace NUMINAMATH_CALUDE_sunday_only_papers_l3045_304574

/-- The number of papers Kyle delivers in a week -/
def total_papers : ℕ := 720

/-- The number of houses Kyle delivers to from Monday to Saturday -/
def regular_houses : ℕ := 100

/-- The number of regular customers who don't receive the Sunday paper -/
def sunday_opt_out : ℕ := 10

/-- The number of days Kyle delivers from Monday to Saturday -/
def weekdays : ℕ := 6

theorem sunday_only_papers : 
  total_papers - (regular_houses * weekdays) - (regular_houses - sunday_opt_out) = 30 := by
  sorry

end NUMINAMATH_CALUDE_sunday_only_papers_l3045_304574


namespace NUMINAMATH_CALUDE_perry_phil_difference_l3045_304551

/-- The number of games won by each player -/
structure GolfWins where
  perry : ℕ
  dana : ℕ
  charlie : ℕ
  phil : ℕ

/-- The conditions of the golf game -/
def golf_game (g : GolfWins) : Prop :=
  g.perry = g.dana + 5 ∧
  g.charlie = g.dana - 2 ∧
  g.phil = g.charlie + 3 ∧
  g.phil = 12

theorem perry_phil_difference (g : GolfWins) (h : golf_game g) : 
  g.perry - g.phil = 4 := by
  sorry

#check perry_phil_difference

end NUMINAMATH_CALUDE_perry_phil_difference_l3045_304551


namespace NUMINAMATH_CALUDE_computer_rental_rates_l3045_304558

/-- Represents the hourly rental rates and job completion times for three computers -/
structure ComputerRental where
  rateA : ℝ  -- Hourly rate for Computer A
  rateB : ℝ  -- Hourly rate for Computer B
  rateC : ℝ  -- Hourly rate for Computer C
  timeA : ℝ  -- Time for Computer A to complete the job

/-- Conditions for the computer rental problem -/
def rental_conditions (r : ComputerRental) : Prop :=
  r.rateA = 1.4 * r.rateB ∧
  r.rateC = 0.75 * r.rateB ∧
  r.rateA * r.timeA = 550 ∧
  r.rateB * (r.timeA + 20) = 550 ∧
  r.rateC * (r.timeA + 10) = 550

/-- Theorem stating the approximate hourly rates for the computers -/
theorem computer_rental_rates :
  ∃ r : ComputerRental, rental_conditions r ∧
    (abs (r.rateA - 11) < 0.01) ∧
    (abs (r.rateB - 7.86) < 0.01) ∧
    (abs (r.rateC - 5.90) < 0.01) :=
  by sorry

end NUMINAMATH_CALUDE_computer_rental_rates_l3045_304558


namespace NUMINAMATH_CALUDE_difference_between_decimals_and_fractions_l3045_304580

theorem difference_between_decimals_and_fractions : (0.127 : ℝ) - (1/8 : ℝ) = 0.002 := by
  sorry

end NUMINAMATH_CALUDE_difference_between_decimals_and_fractions_l3045_304580


namespace NUMINAMATH_CALUDE_rectangle_ratio_is_2_sqrt_6_l3045_304572

/-- Represents a hexagon with side length s -/
structure Hexagon where
  s : ℝ
  h_positive : s > 0

/-- Represents a rectangle with sides x and y -/
structure Rectangle where
  x : ℝ
  y : ℝ
  h_positive_x : x > 0
  h_positive_y : y > 0

/-- The arrangement of rectangles around hexagons -/
structure HexagonArrangement where
  inner : Hexagon
  outer : Hexagon
  rectangle : Rectangle
  h_area_ratio : outer.s^2 = 6 * inner.s^2
  h_outer_perimeter : 6 * rectangle.x = 6 * outer.s
  h_inner_side : rectangle.y = inner.s / 2

theorem rectangle_ratio_is_2_sqrt_6 (arr : HexagonArrangement) :
  arr.rectangle.x / arr.rectangle.y = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_is_2_sqrt_6_l3045_304572


namespace NUMINAMATH_CALUDE_composite_sum_l3045_304575

theorem composite_sum (a b c d e f : ℕ+) 
  (hS : ∃ (k₁ k₂ : ℕ), 
    (a + b + c + d + e + f) * k₁ = a * b * c + d * e * f ∧ 
    (a + b + c + d + e + f) * k₂ = a * b + b * c + c * a - d * e - e * f - f * d) :
  ∃ (m n : ℕ), m > 1 ∧ n > 1 ∧ a + b + c + d + e + f = m * n := by
  sorry

end NUMINAMATH_CALUDE_composite_sum_l3045_304575


namespace NUMINAMATH_CALUDE_weights_standard_deviation_l3045_304539

def weights (a b : ℝ) : List ℝ := [125, a, 121, b, 127]

def median (l : List ℝ) : ℝ := sorry
def mean (l : List ℝ) : ℝ := sorry
def standardDeviation (l : List ℝ) : ℝ := sorry

theorem weights_standard_deviation (a b : ℝ) :
  median (weights a b) = 124 →
  mean (weights a b) = 124 →
  standardDeviation (weights a b) = 2 := by sorry

end NUMINAMATH_CALUDE_weights_standard_deviation_l3045_304539


namespace NUMINAMATH_CALUDE_inequality_proof_l3045_304544

theorem inequality_proof (a b t : ℝ) (ha : a > 1) (hb : b > 1) (ht : t > 0) :
  (a^2 / (b^t - 1)) + (b^(2*t) / (a^t - 1)) ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3045_304544


namespace NUMINAMATH_CALUDE_percent_to_decimal_twenty_five_percent_value_l3045_304540

theorem percent_to_decimal (x : ℚ) : x / 100 = x * (1 / 100) := by sorry

theorem twenty_five_percent_value : (25 : ℚ) / 100 = (1 : ℚ) / 4 := by sorry

end NUMINAMATH_CALUDE_percent_to_decimal_twenty_five_percent_value_l3045_304540


namespace NUMINAMATH_CALUDE_birthday_money_l3045_304547

def money_spent : ℕ := 3
def money_left : ℕ := 2

theorem birthday_money : 
  ∃ (total : ℕ), total = money_spent + money_left :=
sorry

end NUMINAMATH_CALUDE_birthday_money_l3045_304547


namespace NUMINAMATH_CALUDE_parallel_lines_distance_l3045_304505

def line1 (t : ℝ) : ℝ × ℝ := (3 + 2*t, -2 - 5*t)
def line2 (s : ℝ) : ℝ × ℝ := (4 + 2*s, -6 - 5*s)

def direction : ℝ × ℝ := (2, -5)

theorem parallel_lines_distance :
  let v := (3 - 4, -2 - (-6))
  let proj_v := ((v.1 * direction.1 + v.2 * direction.2) / (direction.1^2 + direction.2^2)) • direction
  let c := (4 + proj_v.1, -6 + proj_v.2)
  Real.sqrt ((3 - c.1)^2 + (-2 - c.2)^2) = 31 / 29 := by sorry

end NUMINAMATH_CALUDE_parallel_lines_distance_l3045_304505


namespace NUMINAMATH_CALUDE_milk_replacement_theorem_l3045_304579

-- Define the percentage of milk replaced by water in each operation
def replacement_percentage : ℝ → Prop := λ x =>
  -- Define the function that calculates the remaining milk percentage after three operations
  let remaining_milk := (1 - x/100)^3
  -- The remaining milk percentage should be 51.2%
  remaining_milk = 0.512

-- Theorem statement
theorem milk_replacement_theorem : 
  ∃ x : ℝ, replacement_percentage x ∧ x = 20 :=
sorry

end NUMINAMATH_CALUDE_milk_replacement_theorem_l3045_304579


namespace NUMINAMATH_CALUDE_third_number_is_seven_l3045_304527

def hcf (a b c : ℕ) : ℕ := Nat.gcd a (Nat.gcd b c)

def lcm (a b c : ℕ) : ℕ := Nat.lcm a (Nat.lcm b c)

theorem third_number_is_seven (x : ℕ) 
  (hcf_condition : hcf 136 144 x = 8)
  (lcm_condition : lcm 136 144 x = 2^4 * 3^2 * 17 * 7) :
  x = 7 := by
  sorry

end NUMINAMATH_CALUDE_third_number_is_seven_l3045_304527


namespace NUMINAMATH_CALUDE_system_solution_range_l3045_304529

theorem system_solution_range (x y k : ℝ) : 
  (2 * x + y = k + 1) →
  (x + 2 * y = 2) →
  (x + y < 0) →
  (k < -3) := by
sorry

end NUMINAMATH_CALUDE_system_solution_range_l3045_304529


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_expression_l3045_304597

theorem greatest_prime_factor_of_expression (p : Nat) :
  (p.Prime ∧ p ∣ (2^8 + 5^4 + 10^3) ∧ ∀ q : Nat, q.Prime → q ∣ (2^8 + 5^4 + 10^3) → q ≤ p) ↔ p = 19 := by
  sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_expression_l3045_304597


namespace NUMINAMATH_CALUDE_parallel_vector_m_values_l3045_304576

def vector_a (m : ℝ) : Fin 2 → ℝ := λ i => if i = 0 then 2 else m

theorem parallel_vector_m_values (m : ℝ) :
  (∃ b : Fin 2 → ℝ, b ≠ 0 ∧ ∃ k : ℝ, k ≠ 0 ∧ vector_a m = λ i => k * b i) →
  m = 2 ∨ m = -2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vector_m_values_l3045_304576


namespace NUMINAMATH_CALUDE_leila_sweater_cost_l3045_304560

/-- Represents Leila's spending on a sweater and jewelry --/
structure LeilasSpending where
  total : ℝ
  sweater : ℝ
  jewelry : ℝ

/-- The conditions of Leila's spending --/
def spending_conditions (s : LeilasSpending) : Prop :=
  s.sweater = (1/4) * s.total ∧
  s.jewelry = (3/4) * s.total - 20 ∧
  s.jewelry = s.sweater + 60

/-- Theorem stating that under the given conditions, Leila spent $40 on the sweater --/
theorem leila_sweater_cost (s : LeilasSpending) 
  (h : spending_conditions s) : s.sweater = 40 := by
  sorry

#check leila_sweater_cost

end NUMINAMATH_CALUDE_leila_sweater_cost_l3045_304560


namespace NUMINAMATH_CALUDE_work_completion_days_l3045_304553

/-- Calculates the number of days needed for the remaining workers to complete a job -/
def daysToComplete (originalWorkers : ℕ) (plannedDays : ℕ) (absentWorkers : ℕ) : ℕ :=
  (originalWorkers * plannedDays) / (originalWorkers - absentWorkers)

/-- Proves that given the original conditions, the remaining workers complete the job in 21 days -/
theorem work_completion_days :
  daysToComplete 42 17 8 = 21 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_days_l3045_304553


namespace NUMINAMATH_CALUDE_expression_value_l3045_304591

theorem expression_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y = 3 * Real.sqrt (x * y)) :
  |(x - y) / (x + y) + (x^2 - y^2) / (x^2 + y^2) + (x^3 - y^3) / (x^3 + y^3)| = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3045_304591


namespace NUMINAMATH_CALUDE_clock_angle_at_3_30_angle_between_clock_hands_at_3_30_l3045_304511

/-- The angle between clock hands at 3:30 -/
theorem clock_angle_at_3_30 : ℝ :=
  let hour_hand_angle : ℝ := 3.5 * 30  -- 3:30 is 3.5 hours from 12 o'clock
  let minute_hand_angle : ℝ := 30 * 6  -- 30 minutes is 6 times 5-minute marks
  let angle_diff : ℝ := |minute_hand_angle - hour_hand_angle|
  75

/-- Theorem: The angle between the hour and minute hands at 3:30 is 75 degrees -/
theorem angle_between_clock_hands_at_3_30 : clock_angle_at_3_30 = 75 := by
  sorry

end NUMINAMATH_CALUDE_clock_angle_at_3_30_angle_between_clock_hands_at_3_30_l3045_304511


namespace NUMINAMATH_CALUDE_total_people_zoo_and_amusement_park_l3045_304595

theorem total_people_zoo_and_amusement_park : 
  let cars_to_zoo : Float := 7.0
  let people_per_car_zoo : Float := 45.0
  let cars_to_amusement_park : Float := 5.0
  let people_per_car_amusement_park : Float := 56.0
  
  cars_to_zoo * people_per_car_zoo + cars_to_amusement_park * people_per_car_amusement_park = 595.0 := by
  sorry

end NUMINAMATH_CALUDE_total_people_zoo_and_amusement_park_l3045_304595


namespace NUMINAMATH_CALUDE_same_terminal_side_angle_l3045_304565

theorem same_terminal_side_angle : ∃ θ : ℝ, 
  0 ≤ θ ∧ θ < 2*π ∧ 
  ∃ k : ℤ, θ = 2*k*π + (-4*π/3) ∧
  θ = 2*π/3 := by
  sorry

end NUMINAMATH_CALUDE_same_terminal_side_angle_l3045_304565


namespace NUMINAMATH_CALUDE_f_six_eq_zero_l3045_304582

def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem f_six_eq_zero
  (f : ℝ → ℝ)
  (hodd : isOddFunction f)
  (hperiod : ∀ x, f (x + 2) = -f x) :
  f 6 = 0 := by
sorry

end NUMINAMATH_CALUDE_f_six_eq_zero_l3045_304582


namespace NUMINAMATH_CALUDE_max_area_of_specific_prism_l3045_304546

/-- A prism with vertical edges parallel to the z-axis and a square cross-section -/
structure Prism where
  side_length : ℝ
  cutting_plane : ℝ → ℝ → ℝ → Prop

/-- The maximum area of the cross-section of the prism cut by a plane -/
def max_cross_section_area (p : Prism) : ℝ := sorry

/-- The theorem stating the maximum area of the cross-section for the given prism -/
theorem max_area_of_specific_prism :
  let p : Prism := {
    side_length := 12,
    cutting_plane := fun x y z ↦ 3 * x - 5 * y + 5 * z = 30
  }
  max_cross_section_area p = 360 := by sorry

end NUMINAMATH_CALUDE_max_area_of_specific_prism_l3045_304546


namespace NUMINAMATH_CALUDE_range_of_m_l3045_304532

-- Define propositions p and q
def p (m : ℝ) : Prop := ∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧
  4 * x₁^2 - 4*(m-2)*x₁ + 1 = 0 ∧ 4 * x₂^2 - 4*(m-2)*x₂ + 1 = 0

def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 3*m*x + 1 ≠ 0

-- Theorem statement
theorem range_of_m :
  ∀ m : ℝ, (p m ∨ q m) ∧ ¬(q m) →
  (m ≤ -2/3) ∨ (2/3 ≤ m ∧ m < 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3045_304532


namespace NUMINAMATH_CALUDE_log_equation_solution_l3045_304537

theorem log_equation_solution :
  ∃! x : ℝ, x > 0 ∧ x + 2 > 0 ∧ 2*x + 3 > 0 ∧
  Real.log x + Real.log (x + 2) = Real.log (2*x + 3) ∧
  x = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3045_304537


namespace NUMINAMATH_CALUDE_chess_game_draw_probability_l3045_304525

/-- Given a chess game between A and B:
    * The game can end in A winning, B winning, or a draw.
    * The probability of A not losing is 0.6.
    * The probability of B not losing is 0.7.
    This theorem proves that the probability of the game ending in a draw is 0.3. -/
theorem chess_game_draw_probability :
  ∀ (p_a_win p_b_win p_draw : ℝ),
    p_a_win + p_b_win + p_draw = 1 →
    p_a_win + p_draw = 0.6 →
    p_b_win + p_draw = 0.7 →
    p_draw = 0.3 :=
by sorry

end NUMINAMATH_CALUDE_chess_game_draw_probability_l3045_304525


namespace NUMINAMATH_CALUDE_factorial_division_l3045_304584

theorem factorial_division (ten_factorial : ℕ) (h : ten_factorial = 3628800) :
  ten_factorial / 24 = 151200 := by
  sorry

end NUMINAMATH_CALUDE_factorial_division_l3045_304584


namespace NUMINAMATH_CALUDE_cos_angle_between_vectors_l3045_304563

def a : ℝ × ℝ := (3, 3)
def b : ℝ × ℝ := (1, 2)

theorem cos_angle_between_vectors :
  let θ := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  (Real.cos θ) = 3 * Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_cos_angle_between_vectors_l3045_304563


namespace NUMINAMATH_CALUDE_H_range_l3045_304519

/-- The function H defined for all real x -/
def H (x : ℝ) : ℝ := 2 * |2*x + 2| - 3 * |2*x - 2|

/-- The theorem stating that the range of H is [8, ∞) -/
theorem H_range : Set.range H = Set.Ici 8 := by sorry

end NUMINAMATH_CALUDE_H_range_l3045_304519


namespace NUMINAMATH_CALUDE_geometric_sum_6_terms_l3045_304587

/-- The sum of the first n terms of a geometric series with first term a and common ratio r -/
def geometricSum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The sum of the first 6 terms of the geometric series with first term 2 and common ratio 1/3 -/
theorem geometric_sum_6_terms :
  geometricSum 2 (1/3) 6 = 728/243 := by
sorry

end NUMINAMATH_CALUDE_geometric_sum_6_terms_l3045_304587


namespace NUMINAMATH_CALUDE_triangle_area_implies_ab_value_l3045_304550

theorem triangle_area_implies_ab_value (a b : ℝ) : 
  a > 0 → b > 0 → 
  (1/2 * (12/a) * (12/b) = 9) → 
  a * b = 8 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_implies_ab_value_l3045_304550


namespace NUMINAMATH_CALUDE_six_by_six_grid_half_shaded_l3045_304502

/-- Represents a square grid -/
structure SquareGrid :=
  (size : ℕ)
  (shaded : ℕ)

/-- Calculates the percentage of shaded area in a square grid -/
def shaded_percentage (grid : SquareGrid) : ℚ :=
  (grid.shaded : ℚ) / (grid.size * grid.size : ℚ) * 100

/-- Theorem: A 6x6 grid with 18 shaded squares is 50% shaded -/
theorem six_by_six_grid_half_shaded :
  let grid : SquareGrid := ⟨6, 18⟩
  shaded_percentage grid = 50 := by sorry

end NUMINAMATH_CALUDE_six_by_six_grid_half_shaded_l3045_304502


namespace NUMINAMATH_CALUDE_candy_distribution_l3045_304522

/-- Represents the number of candies eaten by each person -/
structure CandyCount where
  andrey : ℕ
  boris : ℕ
  denis : ℕ

/-- Calculates the total number of candies eaten by all three people -/
def total_candies (count : CandyCount) : ℕ :=
  count.andrey + count.boris + count.denis

/-- Represents the relative eating rates of the three people -/
structure EatingRates where
  andrey_to_boris : ℚ
  andrey_to_denis : ℚ

/-- Theorem stating the correct number of candies eaten by each person -/
theorem candy_distribution (rates : EatingRates) : 
  ∃ (count : CandyCount), 
    rates.andrey_to_boris = 4 / 3 ∧ 
    rates.andrey_to_denis = 6 / 7 ∧
    total_candies count = 70 ∧
    count.andrey = 24 ∧
    count.boris = 18 ∧
    count.denis = 28 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l3045_304522


namespace NUMINAMATH_CALUDE_flour_already_added_l3045_304593

/-- Given a cake recipe and Mary's current progress, calculate how many cups of flour
    she has already put in. -/
theorem flour_already_added
  (total_required : ℕ)  -- Total cups of flour required by the recipe
  (more_needed : ℕ)     -- Cups of flour Mary still needs to add
  (h1 : total_required = 9)  -- The recipe requires 9 cups of flour
  (h2 : more_needed = 7)     -- Mary needs to add 7 more cups
  : total_required - more_needed = 2 := by
  sorry

end NUMINAMATH_CALUDE_flour_already_added_l3045_304593


namespace NUMINAMATH_CALUDE_slope_of_line_l3045_304594

/-- The slope of a line defined by the equation 4y = 5x + 20 is 5/4. -/
theorem slope_of_line (x y : ℝ) : 4 * y = 5 * x + 20 → (y - 5) / (x - (-5)) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_slope_of_line_l3045_304594


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l3045_304573

theorem simplify_trig_expression (x : ℝ) :
  Real.sqrt 2 * Real.cos x - Real.sqrt 6 * Real.sin x = 2 * Real.sqrt 2 * Real.cos (π / 3 + x) :=
by sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l3045_304573


namespace NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_same_line_l3045_304549

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields

/-- Perpendicularity between a line and a plane -/
def perpendicular (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Parallelism between two planes -/
def parallel (p1 p2 : Plane3D) : Prop :=
  sorry

/-- Theorem: If a line is perpendicular to two planes, then those planes are parallel -/
theorem planes_parallel_if_perpendicular_to_same_line (a : Line3D) (α β : Plane3D) :
  perpendicular a α → perpendicular a β → parallel α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_same_line_l3045_304549


namespace NUMINAMATH_CALUDE_longest_segment_in_cylinder_cylinder_surface_area_l3045_304566

-- Define the cylinder
def cylinder_radius : ℝ := 5
def cylinder_height : ℝ := 10

-- Theorem for the longest segment
theorem longest_segment_in_cylinder :
  Real.sqrt ((2 * cylinder_radius) ^ 2 + cylinder_height ^ 2) = 10 * Real.sqrt 2 := by sorry

-- Theorem for the total surface area
theorem cylinder_surface_area :
  2 * Real.pi * cylinder_radius * (cylinder_height + cylinder_radius) = 150 * Real.pi := by sorry

end NUMINAMATH_CALUDE_longest_segment_in_cylinder_cylinder_surface_area_l3045_304566


namespace NUMINAMATH_CALUDE_rod_length_calculation_l3045_304520

/-- The total length of a rod that can be cut into a given number of pieces of a specific length. -/
def rod_length (num_pieces : ℕ) (piece_length : ℝ) : ℝ :=
  num_pieces * piece_length

/-- Theorem stating that a rod that can be cut into 50 pieces of 0.85 metres each has a total length of 42.5 metres. -/
theorem rod_length_calculation : rod_length 50 0.85 = 42.5 := by
  sorry

end NUMINAMATH_CALUDE_rod_length_calculation_l3045_304520


namespace NUMINAMATH_CALUDE_solution_az_eq_b_l3045_304542

theorem solution_az_eq_b (a b : ℝ) : 
  (∃! x, 4 + 3 * a * x = 2 * a - 7) →
  (¬∃ y, 2 + y = (b + 1) * y) →
  (∀ z, a * z = b ↔ z = 0) :=
by sorry

end NUMINAMATH_CALUDE_solution_az_eq_b_l3045_304542


namespace NUMINAMATH_CALUDE_mall_spending_l3045_304545

def total_spent : ℚ := 347
def movie_cost : ℚ := 24
def num_movies : ℕ := 3
def bean_cost : ℚ := 1.25
def num_bean_bags : ℕ := 20

theorem mall_spending (mall_spent : ℚ) : 
  mall_spent = total_spent - (↑num_movies * movie_cost + ↑num_bean_bags * bean_cost) → 
  mall_spent = 250 := by
  sorry

end NUMINAMATH_CALUDE_mall_spending_l3045_304545


namespace NUMINAMATH_CALUDE_cafeteria_apple_count_l3045_304535

/-- Calculates the final number of apples in the cafeteria after a series of operations -/
def final_apple_count (initial : ℕ) (monday_used monday_bought tuesday_used tuesday_bought wednesday_used : ℕ) : ℕ :=
  initial - monday_used + monday_bought - tuesday_used + tuesday_bought - wednesday_used

/-- Theorem stating that given the initial number of apples and daily changes, the final number of apples is 46 -/
theorem cafeteria_apple_count : 
  final_apple_count 17 2 23 4 15 3 = 46 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_apple_count_l3045_304535


namespace NUMINAMATH_CALUDE_expression_evaluation_l3045_304517

theorem expression_evaluation (a b c : ℝ) 
  (h1 : c = b^2 - 14*b + 45)
  (h2 : b = a^2 + 2*a + 5)
  (h3 : a = 3)
  (h4 : a + 1 ≠ 0)
  (h5 : b - 3 ≠ 0)
  (h6 : c + 7 ≠ 0) :
  (a + 3) / (a + 1) * (b - 1) / (b - 3) * (c + 9) / (c + 7) = 4923 / 2924 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3045_304517


namespace NUMINAMATH_CALUDE_determinant_zero_exists_l3045_304534

def matrix (x a b c : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := 
  ![![x + a, x + b, x + c],
    ![x + b, x + c, x + a],
    ![x + c, x + a, x + b]]

theorem determinant_zero_exists (a b c : ℝ) 
  (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) 
  (h4 : a ≠ 0) (h5 : b ≠ 0) (h6 : c ≠ 0) : 
  ∃ x : ℝ, Matrix.det (matrix x a b c) = 0 := by
  sorry

end NUMINAMATH_CALUDE_determinant_zero_exists_l3045_304534


namespace NUMINAMATH_CALUDE_vector_equality_holds_l3045_304589

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

def vector_equality (e₁ e₂ a b : V) : Prop :=
  (e₁ ≠ 0 ∧ e₂ ≠ 0) ∧  -- non-zero vectors
  (∀ (r : ℝ), r • e₁ ≠ e₂) ∧  -- non-collinear
  (a = 3 • e₁ - 2 • e₂) ∧
  (b = e₂ - 2 • e₁) ∧
  ((1/3) • a + b) + (a - (3/2) • b) + (2 • b - a) = -2 • e₁ + (5/6) • e₂

theorem vector_equality_holds (e₁ e₂ a b : V) :
  vector_equality e₁ e₂ a b := by sorry

end NUMINAMATH_CALUDE_vector_equality_holds_l3045_304589


namespace NUMINAMATH_CALUDE_x_plus_y_value_l3045_304510

theorem x_plus_y_value (x y : ℝ) 
  (h1 : x + Real.sin y = 2010)
  (h2 : x + 2010 * Real.cos y = 2009)
  (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2009 + Real.pi / 2 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l3045_304510


namespace NUMINAMATH_CALUDE_amys_final_money_l3045_304586

def amys_money (initial_amount : ℚ) (chore_payment : ℚ) (num_neighbors : ℕ) 
  (birthday_money : ℚ) (investment_percentage : ℚ) (investment_return : ℚ) 
  (toy_cost : ℚ) (grandparent_multiplier : ℚ) (donation_percentage : ℚ) : ℚ :=
  let total_before_investment := initial_amount + chore_payment * num_neighbors + birthday_money
  let invested_amount := total_before_investment * investment_percentage
  let investment_value := invested_amount * (1 + investment_return)
  let remaining_after_toy := total_before_investment - toy_cost
  let after_grandparent_gift := remaining_after_toy * grandparent_multiplier
  let total_before_donation := after_grandparent_gift + investment_value
  let final_amount := total_before_donation * (1 - donation_percentage)
  final_amount

theorem amys_final_money :
  amys_money 2 13 5 3 (20/100) (10/100) 12 2 (25/100) = 98.55 := by
  sorry

end NUMINAMATH_CALUDE_amys_final_money_l3045_304586


namespace NUMINAMATH_CALUDE_geometric_series_sum_l3045_304578

theorem geometric_series_sum (a r : ℚ) (n : ℕ) (h : r ≠ 1) :
  let series_sum := a * (1 - r^n) / (1 - r)
  (a = 1/4) → (r = -1/4) → (n = 6) → series_sum = 4095/5120 := by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l3045_304578


namespace NUMINAMATH_CALUDE_hexagon_side_length_l3045_304524

/-- A hexagon is a polygon with 6 sides -/
def Hexagon : ℕ := 6

/-- Given a hexagon with perimeter 42 inches, prove that each side length is 7 inches -/
theorem hexagon_side_length (perimeter : ℝ) (h1 : perimeter = 42) :
  perimeter / Hexagon = 7 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_side_length_l3045_304524


namespace NUMINAMATH_CALUDE_probability_r25_to_r35_correct_l3045_304598

def bubble_pass (s : List ℝ) : List ℝ := sorry

def probability_r25_to_r35 (n : ℕ) : ℚ :=
  if n ≥ 50 then 1 / 1260 else 0

theorem probability_r25_to_r35_correct (s : List ℝ) (h : s.length = 50) 
  (h_distinct : s.Nodup) : 
  probability_r25_to_r35 s.length = 1 / 1260 := by sorry

end NUMINAMATH_CALUDE_probability_r25_to_r35_correct_l3045_304598


namespace NUMINAMATH_CALUDE_tangent_slope_at_one_l3045_304592

noncomputable def f (x : ℝ) : ℝ := Real.exp x - 1 / x

theorem tangent_slope_at_one :
  HasDerivAt f (Real.exp 1 + 1) 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_slope_at_one_l3045_304592


namespace NUMINAMATH_CALUDE_function_inequality_solution_set_l3045_304571

/-- Given a function f(x) = (ax-1)/(x+b) where the solution set of f(x) > 0 is (-1, 3),
    prove that the solution set of f(-2x) < 0 is (-∞, -3/2) ∪ (1/2, +∞) -/
theorem function_inequality_solution_set 
  (a b : ℝ) 
  (f : ℝ → ℝ) 
  (h₁ : ∀ x, f x = (a * x - 1) / (x + b))
  (h₂ : Set.Ioo (-1 : ℝ) 3 = {x | f x > 0}) :
  {x : ℝ | f (-2 * x) < 0} = Set.Iic (-3/2) ∪ Set.Ioi (1/2) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_solution_set_l3045_304571


namespace NUMINAMATH_CALUDE_simple_interest_example_l3045_304599

/-- Calculates simple interest given principal, rate, and time -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- Proves that the simple interest on $10000 at 9% per annum for 12 months is $900 -/
theorem simple_interest_example : simple_interest 10000 0.09 1 = 900 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_example_l3045_304599


namespace NUMINAMATH_CALUDE_sum_geq_sqrt_three_l3045_304543

theorem sum_geq_sqrt_three (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum_prod : a * b + b * c + c * a = 1) : 
  a + b + c ≥ Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_sum_geq_sqrt_three_l3045_304543


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3045_304513

theorem polynomial_division_remainder : ∃ q : Polynomial ℚ, 
  2 * X^4 + 9 * X^3 - 38 * X^2 - 50 * X + 35 = 
  (X^2 + 5 * X - 6) * q + (61 * X - 91) := by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3045_304513


namespace NUMINAMATH_CALUDE_parabola_intersection_l3045_304569

theorem parabola_intersection :
  let f (x : ℝ) := 3 * x^2 - 4 * x + 2
  let g (x : ℝ) := 9 * x^2 + 6 * x + 2
  ∀ x y : ℝ, f x = g x ∧ y = f x ↔ (x = 0 ∧ y = 2) ∨ (x = -5/3 ∧ y = 17) :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_l3045_304569
