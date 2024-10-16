import Mathlib

namespace NUMINAMATH_CALUDE_comparison_of_special_points_l2867_286788

theorem comparison_of_special_points (a b c : Real) 
  (ha : 0 < a ∧ a < Real.pi / 2)
  (hb : 0 < b ∧ b < Real.pi / 2)
  (hc : 0 < c ∧ c < Real.pi / 2)
  (eq_a : a = Real.cos a)
  (eq_b : b = Real.sin (Real.cos b))
  (eq_c : c = Real.cos (Real.sin c)) :
  b < a ∧ a < c :=
by sorry

end NUMINAMATH_CALUDE_comparison_of_special_points_l2867_286788


namespace NUMINAMATH_CALUDE_deepak_age_l2867_286781

/-- Given the ratio of Rahul's age to Deepak's age and Rahul's future age, 
    prove Deepak's current age -/
theorem deepak_age (rahul_age deepak_age : ℕ) : 
  (rahul_age : ℚ) / deepak_age = 4 / 3 →
  rahul_age + 6 = 22 →
  deepak_age = 12 := by
sorry

end NUMINAMATH_CALUDE_deepak_age_l2867_286781


namespace NUMINAMATH_CALUDE_min_distance_to_line_l2867_286789

open Real

theorem min_distance_to_line :
  let line := {(x, y) : ℝ × ℝ | 4 * x - 3 * y - 5 * sqrt 2 = 0}
  ∃ (m n : ℝ), (m, n) ∈ line ∧ ∀ (x y : ℝ), (x, y) ∈ line → m^2 + n^2 ≤ x^2 + y^2 ∧ m^2 + n^2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l2867_286789


namespace NUMINAMATH_CALUDE_find_number_l2867_286755

theorem find_number (x : ℝ) : x + 1.35 + 0.123 = 1.794 → x = 0.321 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l2867_286755


namespace NUMINAMATH_CALUDE_locus_of_centers_l2867_286707

-- Define the circles C1 and C2
def C1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def C2 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 25

-- Define external tangency to C1
def externally_tangent_to_C1 (a b r : ℝ) : Prop := a^2 + b^2 = (r + 2)^2

-- Define internal tangency to C2
def internally_tangent_to_C2 (a b r : ℝ) : Prop := (a - 3)^2 + b^2 = (5 - r)^2

-- Define the locus equation
def locus_equation (a b : ℝ) : Prop := 13 * a^2 + 49 * b^2 - 12 * a - 1 = 0

-- Theorem statement
theorem locus_of_centers :
  ∀ a b : ℝ,
  (∃ r : ℝ, externally_tangent_to_C1 a b r ∧ internally_tangent_to_C2 a b r) ↔
  locus_equation a b :=
sorry

end NUMINAMATH_CALUDE_locus_of_centers_l2867_286707


namespace NUMINAMATH_CALUDE_impossible_constant_average_l2867_286770

theorem impossible_constant_average (n : ℕ) (initial_total_age : ℕ) : 
  initial_total_age = n * 19 →
  ¬ ∃ (new_total_age : ℕ), new_total_age = initial_total_age + 1 ∧ 
    new_total_age / (n + 1) = 19 :=
by sorry

end NUMINAMATH_CALUDE_impossible_constant_average_l2867_286770


namespace NUMINAMATH_CALUDE_perpendicular_transitive_perpendicular_parallel_transitive_l2867_286762

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- Theorem 1
theorem perpendicular_transitive 
  (m n : Line) (α : Plane) :
  perpendicular_line_plane m α →
  parallel_line_plane n α →
  perpendicular m n :=
sorry

-- Theorem 2
theorem perpendicular_parallel_transitive 
  (m : Line) (α β γ : Plane) :
  parallel_planes α β →
  parallel_planes β γ →
  perpendicular_line_plane m α →
  perpendicular_line_plane m γ :=
sorry

end NUMINAMATH_CALUDE_perpendicular_transitive_perpendicular_parallel_transitive_l2867_286762


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l2867_286736

theorem smallest_integer_with_remainders : ∃ n : ℕ, 
  (n > 0) ∧
  (n % 2 = 1) ∧
  (n % 3 ≠ 0) ∧
  (n % 4 = 3) ∧
  (n % 10 = 9) ∧
  (∀ m : ℕ, m > 0 → m % 2 = 1 → m % 3 ≠ 0 → m % 4 = 3 → m % 10 = 9 → m ≥ n) ∧
  n = 59 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l2867_286736


namespace NUMINAMATH_CALUDE_oil_leak_total_l2867_286775

theorem oil_leak_total (leaked_before_fixing leaked_while_fixing : ℕ) 
  (h1 : leaked_before_fixing = 2475)
  (h2 : leaked_while_fixing = 3731) :
  leaked_before_fixing + leaked_while_fixing = 6206 :=
by sorry

end NUMINAMATH_CALUDE_oil_leak_total_l2867_286775


namespace NUMINAMATH_CALUDE_grinder_loss_percentage_l2867_286777

theorem grinder_loss_percentage (grinder_cp mobile_cp total_profit mobile_profit_percent : ℝ)
  (h1 : grinder_cp = 15000)
  (h2 : mobile_cp = 10000)
  (h3 : total_profit = 400)
  (h4 : mobile_profit_percent = 10) :
  let mobile_sp := mobile_cp * (1 + mobile_profit_percent / 100)
  let total_sp := grinder_cp + mobile_cp + total_profit
  let grinder_sp := total_sp - mobile_sp
  let loss_amount := grinder_cp - grinder_sp
  loss_amount / grinder_cp * 100 = 4 := by sorry

end NUMINAMATH_CALUDE_grinder_loss_percentage_l2867_286777


namespace NUMINAMATH_CALUDE_store_profit_l2867_286706

/-- The profit made by the store selling New Year cards -/
theorem store_profit (cost_price : ℚ) (total_sales : ℚ) (n : ℕ) (selling_price : ℚ) : 
  cost_price = 21/100 ∧ 
  total_sales = 1457/100 ∧ 
  n * selling_price = total_sales ∧ 
  selling_price ≤ 2 * cost_price →
  n * (selling_price - cost_price) = 47/10 := by
  sorry

#check store_profit

end NUMINAMATH_CALUDE_store_profit_l2867_286706


namespace NUMINAMATH_CALUDE_point_on_graph_l2867_286782

def f (x : ℝ) : ℝ := x + 1

theorem point_on_graph :
  f 0 = 1 ∧ 
  f 1 ≠ 1 ∧
  f 2 ≠ 0 ∧
  f (-1) ≠ 1 := by
sorry

end NUMINAMATH_CALUDE_point_on_graph_l2867_286782


namespace NUMINAMATH_CALUDE_f_positive_range_f_greater_g_range_l2867_286726

-- Define the functions f and g
def f (x : ℝ) := x^2 - x - 6
def g (b x : ℝ) := b*x - 10

-- Theorem for the range of x where f(x) > 0
theorem f_positive_range (x : ℝ) : 
  f x > 0 ↔ x < -2 ∨ x > 3 :=
sorry

-- Theorem for the range of b where f(x) > g(x) for all real x
theorem f_greater_g_range (b : ℝ) : 
  (∀ x : ℝ, f x > g b x) ↔ b < -5 ∨ b > 3 :=
sorry

end NUMINAMATH_CALUDE_f_positive_range_f_greater_g_range_l2867_286726


namespace NUMINAMATH_CALUDE_union_of_sets_l2867_286756

theorem union_of_sets : 
  let P : Set ℕ := {1, 2}
  let Q : Set ℕ := {2, 3}
  P ∪ Q = {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_union_of_sets_l2867_286756


namespace NUMINAMATH_CALUDE_initial_plus_rainfall_equals_final_l2867_286792

/-- Represents the rainfall data for a single day -/
structure RainfallData where
  rate1 : Real  -- rainfall rate from 2pm to 4pm in inches per hour
  duration1 : Real  -- duration of rainfall from 2pm to 4pm in hours
  rate2 : Real  -- rainfall rate from 4pm to 7pm in inches per hour
  duration2 : Real  -- duration of rainfall from 4pm to 7pm in hours
  rate3 : Real  -- rainfall rate from 7pm to 9pm in inches per hour
  duration3 : Real  -- duration of rainfall from 7pm to 9pm in hours
  final_amount : Real  -- amount of water in the gauge at 9pm in inches

/-- Calculates the total rainfall during the day -/
def total_rainfall (data : RainfallData) : Real :=
  data.rate1 * data.duration1 + data.rate2 * data.duration2 + data.rate3 * data.duration3

/-- Theorem stating that the initial amount plus total rainfall equals the final amount -/
theorem initial_plus_rainfall_equals_final (data : RainfallData) 
    (h1 : data.rate1 = 4) (h2 : data.duration1 = 2)
    (h3 : data.rate2 = 3) (h4 : data.duration2 = 3)
    (h5 : data.rate3 = 0.5) (h6 : data.duration3 = 2)
    (h7 : data.final_amount = 20) :
    ∃ initial_amount : Real, initial_amount + total_rainfall data = data.final_amount := by
  sorry

end NUMINAMATH_CALUDE_initial_plus_rainfall_equals_final_l2867_286792


namespace NUMINAMATH_CALUDE_f_seven_equals_163_l2867_286703

theorem f_seven_equals_163 (f : ℝ → ℝ) 
  (h1 : f 1 = 1)
  (h2 : ∀ x y, f (x + y) = f x + f y + 8 * x * y - 2) : 
  f 7 = 163 := by
sorry

end NUMINAMATH_CALUDE_f_seven_equals_163_l2867_286703


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_negation_of_square_not_equal_self_l2867_286721

theorem negation_of_universal_proposition (p : ℝ → Prop) :
  (¬∀ x : ℝ, p x) ↔ (∃ x : ℝ, ¬p x) :=
by sorry

theorem negation_of_square_not_equal_self :
  (¬∀ x : ℝ, x^2 ≠ x) ↔ (∃ x : ℝ, x^2 = x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_negation_of_square_not_equal_self_l2867_286721


namespace NUMINAMATH_CALUDE_circle_passes_fixed_point_circle_tangent_condition_l2867_286727

-- Define the circle equation
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 - 4*a*x + 2*a*y + 20*(a - 1) = 0

-- Define the fixed point
def fixed_point : ℝ × ℝ := (4, -2)

-- Define the second circle
def second_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 4

-- Theorem 1: The circle passes through the fixed point for all real a
theorem circle_passes_fixed_point :
  ∀ a : ℝ, circle_equation fixed_point.1 fixed_point.2 a :=
sorry

-- Theorem 2: The circle is tangent to the second circle iff a = 1 - √5 or a = 1 + √5
theorem circle_tangent_condition :
  ∀ a : ℝ, (∃ x y : ℝ, circle_equation x y a ∧ second_circle x y ∧
    (∀ x' y' : ℝ, circle_equation x' y' a ∧ second_circle x' y' → (x = x' ∧ y = y'))) ↔
    (a = 1 - Real.sqrt 5 ∨ a = 1 + Real.sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_circle_passes_fixed_point_circle_tangent_condition_l2867_286727


namespace NUMINAMATH_CALUDE_existence_of_point_l2867_286784

theorem existence_of_point (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  ∃ x ∈ Set.Icc 0 1, (4 / Real.pi) * (f 1 - f 0) = (1 + x^2) * (deriv f x) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_point_l2867_286784


namespace NUMINAMATH_CALUDE_unique_solution_l2867_286757

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (g x * g y - g (x * y)) / 5 = x + y + 2

/-- The main theorem stating that the function g(x) = x + 3 is the unique solution -/
theorem unique_solution :
  ∀ g : ℝ → ℝ, SatisfiesFunctionalEquation g → ∀ x : ℝ, g x = x + 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l2867_286757


namespace NUMINAMATH_CALUDE_min_value_xy_l2867_286795

theorem min_value_xy (x y : ℝ) (hx : x > 1) (hy : y > 1) 
  (h_seq : (1/4 * Real.log x) * (Real.log y) = 1/16) : 
  x * y ≥ Real.exp 1 ∧ ∃ x y, x > 1 ∧ y > 1 ∧ (1/4 * Real.log x) * (Real.log y) = 1/16 ∧ x * y = Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_xy_l2867_286795


namespace NUMINAMATH_CALUDE_third_sum_third_term_ratio_l2867_286751

/-- An arithmetic progression with the given properties -/
structure ArithmeticProgression where
  a : ℕ → ℝ
  sum : ℕ → ℝ
  first_third_sum : a 1 + a 3 = 5/2
  second_fourth_sum : a 2 + a 4 = 5/4
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1

/-- The theorem stating that S₃/a₃ = 6 for the given arithmetic progression -/
theorem third_sum_third_term_ratio (ap : ArithmeticProgression) :
  ap.sum 3 / ap.a 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_third_sum_third_term_ratio_l2867_286751


namespace NUMINAMATH_CALUDE_complex_equation_result_l2867_286747

theorem complex_equation_result (a b : ℝ) (h : (a + 4 * Complex.I) * Complex.I = b + Complex.I) : a - b = -5 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_result_l2867_286747


namespace NUMINAMATH_CALUDE_simplify_expression_l2867_286722

theorem simplify_expression : (4 + 3) + (8 - 3 - 1) = 11 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2867_286722


namespace NUMINAMATH_CALUDE_exponential_equation_solution_l2867_286730

theorem exponential_equation_solution :
  ∃ m : ℤ, (3 : ℝ)^m * 9^m = 81^(m - 24) ∧ m = 96 :=
by
  sorry

end NUMINAMATH_CALUDE_exponential_equation_solution_l2867_286730


namespace NUMINAMATH_CALUDE_six_hours_prep_score_l2867_286720

/-- Represents the relationship between study time and test score -/
structure TestPreparation where
  actualHours : ℝ
  score : ℝ

/-- Calculates effective hours from actual hours -/
def effectiveHours (ah : ℝ) : ℝ := 0.8 * ah

/-- Theorem: Given the conditions, 6 actual hours of preparation results in a score of 96 points -/
theorem six_hours_prep_score :
  ∀ (test1 test2 : TestPreparation),
  test1.actualHours = 5 ∧
  test1.score = 80 ∧
  test2.actualHours = 6 →
  test2.score = 96 := by sorry

end NUMINAMATH_CALUDE_six_hours_prep_score_l2867_286720


namespace NUMINAMATH_CALUDE_megan_seashell_count_l2867_286713

/-- The number of seashells Megan needs to add to her collection -/
def additional_shells : ℕ := 6

/-- The total number of seashells Megan wants in her collection -/
def target_shells : ℕ := 25

/-- Megan's current number of seashells -/
def current_shells : ℕ := target_shells - additional_shells

theorem megan_seashell_count : current_shells = 19 := by
  sorry

end NUMINAMATH_CALUDE_megan_seashell_count_l2867_286713


namespace NUMINAMATH_CALUDE_sqrt_fifteen_over_two_equals_half_sqrt_thirty_l2867_286790

theorem sqrt_fifteen_over_two_equals_half_sqrt_thirty :
  Real.sqrt (15 / 2) = (1 / 2) * Real.sqrt 30 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fifteen_over_two_equals_half_sqrt_thirty_l2867_286790


namespace NUMINAMATH_CALUDE_quilt_width_l2867_286749

/-- Given a rectangular quilt with the following properties:
  - length is 7 feet
  - cost per square foot is $40
  - total cost is $2240
  Prove that the width of the quilt is 8 feet. -/
theorem quilt_width (length : ℝ) (cost_per_sqft : ℝ) (total_cost : ℝ) :
  length = 7 →
  cost_per_sqft = 40 →
  total_cost = 2240 →
  total_cost / cost_per_sqft / length = 8 :=
by sorry

end NUMINAMATH_CALUDE_quilt_width_l2867_286749


namespace NUMINAMATH_CALUDE_root_product_theorem_l2867_286776

-- Define the polynomial f(x) = x^6 + x^3 + 1
def f (x : ℂ) : ℂ := x^6 + x^3 + 1

-- Define the function g(x) = x^2 - 3
def g (x : ℂ) : ℂ := x^2 - 3

-- State the theorem
theorem root_product_theorem (x₁ x₂ x₃ x₄ x₅ x₆ : ℂ) 
  (hroots : (X - x₁) * (X - x₂) * (X - x₃) * (X - x₄) * (X - x₅) * (X - x₆) = f X) :
  g x₁ * g x₂ * g x₃ * g x₄ * g x₅ * g x₆ = 757 := by
  sorry

end NUMINAMATH_CALUDE_root_product_theorem_l2867_286776


namespace NUMINAMATH_CALUDE_count_measurable_weights_l2867_286705

/-- Represents the available weights in grams -/
def available_weights : List ℕ := [1, 2, 6, 26]

/-- Represents a configuration of weights on the balance scale -/
structure WeightConfiguration :=
  (left : List ℕ)
  (right : List ℕ)

/-- Calculates the measurable weight for a given configuration -/
def measurable_weight (config : WeightConfiguration) : ℤ :=
  (config.left.sum : ℤ) - (config.right.sum : ℤ)

/-- Generates all possible weight configurations -/
def all_configurations : List WeightConfiguration :=
  sorry

/-- Calculates all measurable weights -/
def measurable_weights : List ℕ :=
  sorry

/-- The main theorem to prove -/
theorem count_measurable_weights :
  measurable_weights.length = 28 :=
sorry

end NUMINAMATH_CALUDE_count_measurable_weights_l2867_286705


namespace NUMINAMATH_CALUDE_specific_candidate_prob_l2867_286753

/-- The number of candidates -/
def n : ℕ := 4

/-- The number of candidates to be selected -/
def k : ℕ := 2

/-- Probability of a specific candidate being selected when k out of n are chosen -/
def prob_selected (n k : ℕ) : ℚ := (n - 1).choose (k - 1) / n.choose k

theorem specific_candidate_prob :
  prob_selected n k = 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_specific_candidate_prob_l2867_286753


namespace NUMINAMATH_CALUDE_lillian_sugar_bags_lillian_sugar_bags_proof_l2867_286785

/-- Lillian's cupcake sugar problem -/
theorem lillian_sugar_bags : ℕ :=
  let sugar_at_home : ℕ := 3
  let sugar_per_bag : ℕ := 6
  let sugar_per_dozen_batter : ℕ := 1
  let sugar_per_dozen_frosting : ℕ := 2
  let dozens_of_cupcakes : ℕ := 5

  let total_sugar_needed := dozens_of_cupcakes * (sugar_per_dozen_batter + sugar_per_dozen_frosting)
  let sugar_to_buy := total_sugar_needed - sugar_at_home
  let bags_to_buy := sugar_to_buy / sugar_per_bag

  2

theorem lillian_sugar_bags_proof : lillian_sugar_bags = 2 := by
  sorry

end NUMINAMATH_CALUDE_lillian_sugar_bags_lillian_sugar_bags_proof_l2867_286785


namespace NUMINAMATH_CALUDE_equation_solution_l2867_286791

theorem equation_solution : ∃ y : ℝ, y^4 - 20*y + 1 = 22 ∧ y = -1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2867_286791


namespace NUMINAMATH_CALUDE_square_difference_equals_400_l2867_286708

theorem square_difference_equals_400 : (25 + 8)^2 - (8^2 + 25^2) = 400 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equals_400_l2867_286708


namespace NUMINAMATH_CALUDE_figure_dimension_proof_l2867_286765

theorem figure_dimension_proof (x : ℝ) : 
  let square_area := (3 * x)^2
  let rectangle_area := 2 * x * 6 * x
  let triangle_area := (1 / 2) * (3 * x) * (2 * x)
  let total_area := square_area + rectangle_area + triangle_area
  total_area = 1000 → x = (5 * Real.sqrt 15) / 3 := by
sorry

end NUMINAMATH_CALUDE_figure_dimension_proof_l2867_286765


namespace NUMINAMATH_CALUDE_no_real_d_for_two_distinct_roots_l2867_286723

/-- The function g(x) = x^2 + 4x + d -/
def g (d : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + d

/-- The composition of g with itself -/
def g_comp (d : ℝ) (x : ℝ) : ℝ := g d (g d x)

/-- Theorem stating that there are no real values of d such that g(g(x)) has exactly 2 distinct real roots -/
theorem no_real_d_for_two_distinct_roots :
  ¬ ∃ d : ℝ, ∃! (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ ∀ x : ℝ, g_comp d x = 0 ↔ x = r₁ ∨ x = r₂ :=
sorry

end NUMINAMATH_CALUDE_no_real_d_for_two_distinct_roots_l2867_286723


namespace NUMINAMATH_CALUDE_silver_division_problem_l2867_286701

/-- 
Given:
- m : ℕ is the number of people
- n : ℕ is the total amount of silver in taels
- Adding 7 taels to each person's share and 7 taels in total equals n
- Subtracting 8 taels from each person's share and subtracting 8 taels in total equals n

Prove that the system of equations 7m + 7 = n and 8m - 8 = n correctly represents the situation
-/
theorem silver_division_problem (m n : ℕ) 
  (h1 : 7 * m + 7 = n) 
  (h2 : 8 * m - 8 = n) : 
  (7 * m + 7 = n) ∧ (8 * m - 8 = n) := by
  sorry

end NUMINAMATH_CALUDE_silver_division_problem_l2867_286701


namespace NUMINAMATH_CALUDE_euler_children_mean_age_l2867_286731

def euler_children_ages : List ℕ := [7, 7, 7, 12, 12, 14, 15]

theorem euler_children_mean_age :
  (euler_children_ages.sum : ℚ) / euler_children_ages.length = 74 / 7 := by
  sorry

end NUMINAMATH_CALUDE_euler_children_mean_age_l2867_286731


namespace NUMINAMATH_CALUDE_factor_sum_l2867_286737

theorem factor_sum (P Q : ℝ) : 
  (∃ b c : ℝ, (X^2 + 3*X + 7) * (X^2 + b*X + c) = X^4 + P*X^2 + Q) → 
  P + Q = 54 := by
sorry

end NUMINAMATH_CALUDE_factor_sum_l2867_286737


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_numbers_l2867_286769

def numbers : List ℕ := [18, 24, 42]

theorem arithmetic_mean_of_numbers :
  (numbers.sum / numbers.length : ℚ) = 28 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_numbers_l2867_286769


namespace NUMINAMATH_CALUDE_angle_value_proof_l2867_286732

theorem angle_value_proof (ABC DBC ABD : ℝ) (y : ℝ) : 
  ABC = 90 →
  ABD = 3 * y →
  DBC = 2 * y →
  ABD + DBC = 90 →
  y = 18 := by sorry

end NUMINAMATH_CALUDE_angle_value_proof_l2867_286732


namespace NUMINAMATH_CALUDE_meaningful_expression_l2867_286704

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = (Real.sqrt (x + 3)) / (x - 1)) ↔ x ≥ -3 ∧ x ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_meaningful_expression_l2867_286704


namespace NUMINAMATH_CALUDE_no_positive_integer_solutions_l2867_286779

theorem no_positive_integer_solutions :
  ¬ ∃ (x y : ℕ+), x^2 - 2*y^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_solutions_l2867_286779


namespace NUMINAMATH_CALUDE_oranges_per_box_l2867_286748

theorem oranges_per_box (total_oranges : ℕ) (num_boxes : ℕ) 
  (h1 : total_oranges = 42) (h2 : num_boxes = 7) :
  total_oranges / num_boxes = 6 := by
  sorry

end NUMINAMATH_CALUDE_oranges_per_box_l2867_286748


namespace NUMINAMATH_CALUDE_bob_finishes_24_minutes_after_alice_l2867_286710

/-- Represents the race scenario -/
structure RaceScenario where
  distance : ℕ  -- Race distance in miles
  alice_speed : ℕ  -- Alice's speed in minutes per mile
  bob_speed : ℕ  -- Bob's speed in minutes per mile

/-- Calculates the time difference between Alice and Bob finishing the race -/
def finish_time_difference (race : RaceScenario) : ℕ :=
  race.distance * race.bob_speed - race.distance * race.alice_speed

/-- Theorem stating that in the given race scenario, Bob finishes 24 minutes after Alice -/
theorem bob_finishes_24_minutes_after_alice :
  let race := RaceScenario.mk 12 7 9
  finish_time_difference race = 24 := by
  sorry

end NUMINAMATH_CALUDE_bob_finishes_24_minutes_after_alice_l2867_286710


namespace NUMINAMATH_CALUDE_smallest_positive_d_l2867_286794

theorem smallest_positive_d : ∃ d : ℝ,
  d > 0 ∧
  (2 * Real.sqrt 7)^2 + (d + 5)^2 = (2 * d + 1)^2 ∧
  ∀ d' : ℝ, d' > 0 → (2 * Real.sqrt 7)^2 + (d' + 5)^2 = (2 * d' + 1)^2 → d ≤ d' ∧
  d = 1 + Real.sqrt 660 / 6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_d_l2867_286794


namespace NUMINAMATH_CALUDE_triarc_area_theorem_l2867_286798

/-- Represents a region enclosed by three circular arcs --/
structure TriarcRegion where
  radius : ℝ
  centralAngle : ℝ

/-- Calculates the area of a TriarcRegion --/
def areaOfTriarcRegion (region : TriarcRegion) : ℝ := sorry

/-- Theorem stating the properties of the specific TriarcRegion's area --/
theorem triarc_area_theorem (region : TriarcRegion) 
  (h1 : region.radius = 6)
  (h2 : region.centralAngle = π/2)
  : ∃ (p q r : ℝ), 
    (areaOfTriarcRegion region = p * Real.sqrt q + r * π) ∧ 
    (∀ k : ℝ, k > 1 → ¬(∃ m : ℝ, q = k * k * m)) ∧ 
    (p + q + r = 7.5) := by
  sorry

#check triarc_area_theorem

end NUMINAMATH_CALUDE_triarc_area_theorem_l2867_286798


namespace NUMINAMATH_CALUDE_abs_minus_one_eq_zero_l2867_286729

theorem abs_minus_one_eq_zero (a : ℝ) : |a| - 1 = 0 → a = 1 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_abs_minus_one_eq_zero_l2867_286729


namespace NUMINAMATH_CALUDE_meal_cost_l2867_286734

theorem meal_cost (total_bill : ℝ) (tip_percentage : ℝ) (payment : ℝ) (change : ℝ) :
  total_bill = 2.5 →
  tip_percentage = 0.2 →
  payment = 20 →
  change = 5 →
  ∃ (meal_cost : ℝ), meal_cost = 12.5 ∧ meal_cost + tip_percentage * meal_cost = payment - change :=
by sorry

end NUMINAMATH_CALUDE_meal_cost_l2867_286734


namespace NUMINAMATH_CALUDE_f_le_one_l2867_286759

noncomputable def f (x : ℝ) : ℝ := (1 + Real.log x) / x

theorem f_le_one (x : ℝ) (hx : x > 0) : f x ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_f_le_one_l2867_286759


namespace NUMINAMATH_CALUDE_cubic_factorization_l2867_286741

theorem cubic_factorization (m : ℝ) : m^3 - 4*m = m*(m + 2)*(m - 2) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l2867_286741


namespace NUMINAMATH_CALUDE_total_money_divided_l2867_286739

/-- Proves that the total amount of money divided among three persons is 116000,
    given their share ratios and the amount for one person. -/
theorem total_money_divided (share_a share_b share_c : ℝ) : 
  share_a = 29491.525423728814 →
  share_a / share_b = 3 / 4 →
  share_b / share_c = 5 / 6 →
  share_a + share_b + share_c = 116000 := by
sorry

end NUMINAMATH_CALUDE_total_money_divided_l2867_286739


namespace NUMINAMATH_CALUDE_angle_range_for_skew_lines_l2867_286750

/-- The angle between two lines in 3D space -/
noncomputable def angle_between_lines (l1 l2 : Line3) : ℝ := sorry

/-- Two lines are skew if they are not parallel and do not intersect -/
def are_skew (l1 l2 : Line3) : Prop := sorry

/-- Two lines are perpendicular if their angle is 90° -/
def are_perpendicular (l1 l2 : Line3) : Prop := angle_between_lines l1 l2 = Real.pi / 2

theorem angle_range_for_skew_lines (a b c : Line3) :
  are_skew a b ∧
  angle_between_lines a b = Real.pi / 3 ∧
  are_perpendicular a c →
  Real.pi / 6 ≤ angle_between_lines b c ∧ angle_between_lines b c ≤ Real.pi / 2 := by sorry

end NUMINAMATH_CALUDE_angle_range_for_skew_lines_l2867_286750


namespace NUMINAMATH_CALUDE_celias_budget_weeks_l2867_286766

/-- Celia's budget problem -/
theorem celias_budget_weeks (weekly_food_budget : ℝ) (rent : ℝ) (streaming : ℝ) (cell_phone : ℝ) 
  (savings_percent : ℝ) (savings_amount : ℝ) :
  weekly_food_budget = 100 →
  rent = 1500 →
  streaming = 30 →
  cell_phone = 50 →
  savings_percent = 0.1 →
  savings_amount = 198 →
  ∃ (weeks : ℕ), 
    savings_amount = savings_percent * (weekly_food_budget * ↑weeks + rent + streaming + cell_phone) ∧
    weeks = 4 := by
  sorry

end NUMINAMATH_CALUDE_celias_budget_weeks_l2867_286766


namespace NUMINAMATH_CALUDE_ratio_a_to_c_l2867_286742

theorem ratio_a_to_c (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 4 / 3)
  (hdb : d / b = 1 / 7) :
  a / c = 105 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ratio_a_to_c_l2867_286742


namespace NUMINAMATH_CALUDE_initial_sweets_count_proof_initial_sweets_count_l2867_286796

theorem initial_sweets_count : ℕ → Prop :=
  fun x => 
    (x / 2 + 4 + 7 = x) → 
    x = 22

-- The proof is omitted
theorem proof_initial_sweets_count : initial_sweets_count 22 := by
  sorry

end NUMINAMATH_CALUDE_initial_sweets_count_proof_initial_sweets_count_l2867_286796


namespace NUMINAMATH_CALUDE_percentage_change_equivalence_l2867_286786

theorem percentage_change_equivalence (p q N : ℝ) 
  (hp : p > 0) (hq : q > 0) (hq_bound : q < 50) (hN : N > 0) :
  N * (1 + p / 100) * (1 - q / 100) < N ↔ p < (100 * q) / (100 - q) := by
  sorry

end NUMINAMATH_CALUDE_percentage_change_equivalence_l2867_286786


namespace NUMINAMATH_CALUDE_quartic_polynomial_unique_l2867_286767

/-- A monic quartic polynomial with real coefficients -/
def QuarticPolynomial (a b c d : ℝ) : ℂ → ℂ := fun x ↦ x^4 + a*x^3 + b*x^2 + c*x + d

theorem quartic_polynomial_unique 
  (q : ℂ → ℂ) 
  (h_monic : q = fun x ↦ x^4 + (q 1 - 1)*x^3 + (q 2 - q 1 + 2)*x^2 + (q 3 - q 2 + q 1 - 3)*x + q 0)
  (h_real : ∀ x : ℝ, ∃ y : ℝ, q x = y)
  (h_root : q (2 + I) = 0)
  (h_value : q 0 = -120) :
  q = QuarticPolynomial 1 (-19) (-116) 120 := by
  sorry

end NUMINAMATH_CALUDE_quartic_polynomial_unique_l2867_286767


namespace NUMINAMATH_CALUDE_connected_graphs_lower_bound_l2867_286709

/-- The number of connected labeled graphs on n vertices -/
def g (n : ℕ) : ℕ := sorry

/-- The total number of labeled graphs on n vertices -/
def total_graphs (n : ℕ) : ℕ := 2^(n * (n - 1) / 2)

/-- Theorem: The number of connected labeled graphs on n vertices is at least half of the total number of labeled graphs on n vertices -/
theorem connected_graphs_lower_bound (n : ℕ) : g n ≥ total_graphs n / 2 := by sorry

end NUMINAMATH_CALUDE_connected_graphs_lower_bound_l2867_286709


namespace NUMINAMATH_CALUDE_wx_plus_yz_equals_99_l2867_286712

theorem wx_plus_yz_equals_99 
  (w x y z : ℝ) 
  (h1 : w + x + y = -2)
  (h2 : w + x + z = 4)
  (h3 : w + y + z = 19)
  (h4 : x + y + z = 12) :
  w * x + y * z = 99 := by
sorry

end NUMINAMATH_CALUDE_wx_plus_yz_equals_99_l2867_286712


namespace NUMINAMATH_CALUDE_factorization_equality_l2867_286754

theorem factorization_equality (x y : ℝ) : 
  x^2 - 2*x*y + y^2 - 1 = (x - y + 1) * (x - y - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2867_286754


namespace NUMINAMATH_CALUDE_class_size_count_l2867_286793

def is_valid_class_size (n : ℕ) : Prop :=
  ∃ b g : ℕ, n = b + g ∧ n > 25 ∧ 2 < b ∧ b < 10 ∧ 14 < g ∧ g < 23

theorem class_size_count :
  ∃! (s : Finset ℕ), (∀ n, n ∈ s ↔ is_valid_class_size n) ∧ s.card = 6 := by
  sorry

end NUMINAMATH_CALUDE_class_size_count_l2867_286793


namespace NUMINAMATH_CALUDE_product_mod_23_l2867_286746

theorem product_mod_23 : (2011 * 2012 * 2013 * 2014 * 2015) % 23 = 5 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_23_l2867_286746


namespace NUMINAMATH_CALUDE_no_non_zero_solutions_l2867_286752

theorem no_non_zero_solutions (a b : ℝ) :
  (Real.sqrt (a^2 + b^2) = a^2 - b^2 → a = 0 ∧ b = 0) ∧
  (Real.sqrt (a^2 + b^2) = |a - b| → a = 0 ∧ b = 0) ∧
  (Real.sqrt (a^2 + b^2) = (a + b) / 2 → a = 0 ∧ b = 0) ∧
  (Real.sqrt (a^2 + b^2) = a^3 - b^3 → a = 0 ∧ b = 0) :=
by sorry

end NUMINAMATH_CALUDE_no_non_zero_solutions_l2867_286752


namespace NUMINAMATH_CALUDE_cats_collected_l2867_286763

theorem cats_collected (initial_dogs initial_cats dogs_adopted final_total : ℕ) : 
  initial_dogs = 36 → 
  initial_cats = 29 → 
  dogs_adopted = 20 → 
  final_total = 57 → 
  final_total - (initial_dogs - dogs_adopted) - initial_cats = 12 := by
  sorry

#check cats_collected

end NUMINAMATH_CALUDE_cats_collected_l2867_286763


namespace NUMINAMATH_CALUDE_min_dividing_segment_length_l2867_286744

/-- A trapezoid with midsegment length 4 and a line parallel to the bases dividing its area in half -/
structure DividedTrapezoid where
  /-- The length of the midsegment -/
  midsegment_length : ℝ
  /-- The length of the lower base -/
  lower_base : ℝ
  /-- The length of the upper base -/
  upper_base : ℝ
  /-- The height of the trapezoid -/
  height : ℝ
  /-- The length of the segment created by the dividing line -/
  dividing_segment : ℝ
  /-- The midsegment length is 4 -/
  midsegment_eq : midsegment_length = 4
  /-- The dividing line splits the area in half -/
  area_split : (lower_base + dividing_segment) * (height / 2) = (upper_base + dividing_segment) * (height / 2)
  /-- The sum of the bases is twice the midsegment length -/
  bases_sum : lower_base + upper_base = 2 * midsegment_length

/-- The minimum possible length of the dividing segment is 4 -/
theorem min_dividing_segment_length (t : DividedTrapezoid) : 
  ∃ (min_length : ℝ), min_length = 4 ∧ ∀ (x : ℝ), t.dividing_segment ≥ min_length :=
by sorry

end NUMINAMATH_CALUDE_min_dividing_segment_length_l2867_286744


namespace NUMINAMATH_CALUDE_five_star_three_eq_four_l2867_286728

/-- The star operation for integers -/
def star (a b : ℤ) : ℤ := a^2 - 2*a*b + b^2

/-- Theorem: 5 star 3 equals 4 -/
theorem five_star_three_eq_four : star 5 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_five_star_three_eq_four_l2867_286728


namespace NUMINAMATH_CALUDE_x_fourth_plus_reciprocal_l2867_286724

theorem x_fourth_plus_reciprocal (x : ℝ) (h : x + 1/x = 8) : x^4 + 1/x^4 = 3842 := by
  sorry

end NUMINAMATH_CALUDE_x_fourth_plus_reciprocal_l2867_286724


namespace NUMINAMATH_CALUDE_goose_egg_count_l2867_286768

-- Define the number of goose eggs laid at the pond
def total_eggs : ℕ := 2000

-- Define the fraction of eggs that hatched
def hatch_rate : ℚ := 2/3

-- Define the fraction of hatched geese that survived the first month
def first_month_survival_rate : ℚ := 3/4

-- Define the fraction of geese that survived the first month but did not survive the first year
def first_year_mortality_rate : ℚ := 3/5

-- Define the number of geese that survived the first year
def survived_first_year : ℕ := 100

-- Theorem statement
theorem goose_egg_count :
  (total_eggs : ℚ) * hatch_rate * first_month_survival_rate * (1 - first_year_mortality_rate) = survived_first_year :=
sorry

end NUMINAMATH_CALUDE_goose_egg_count_l2867_286768


namespace NUMINAMATH_CALUDE_initial_guinea_fowls_eq_80_l2867_286719

/-- Represents the initial state and daily losses of birds in a poultry farm --/
structure PoultryFarm :=
  (initial_chickens : ℕ)
  (initial_turkeys : ℕ)
  (daily_chicken_loss : ℕ)
  (daily_turkey_loss : ℕ)
  (daily_guinea_fowl_loss : ℕ)
  (disease_duration : ℕ)
  (total_birds_after_disease : ℕ)

/-- Calculates the initial number of guinea fowls in the farm --/
def initial_guinea_fowls (farm : PoultryFarm) : ℕ :=
  let remaining_chickens := farm.initial_chickens - farm.daily_chicken_loss * farm.disease_duration
  let remaining_turkeys := farm.initial_turkeys - farm.daily_turkey_loss * farm.disease_duration
  let remaining_guinea_fowls := farm.total_birds_after_disease - remaining_chickens - remaining_turkeys
  remaining_guinea_fowls + farm.daily_guinea_fowl_loss * farm.disease_duration

/-- Theorem stating that the initial number of guinea fowls is 80 --/
theorem initial_guinea_fowls_eq_80 (farm : PoultryFarm) 
  (h1 : farm.initial_chickens = 300)
  (h2 : farm.initial_turkeys = 200)
  (h3 : farm.daily_chicken_loss = 20)
  (h4 : farm.daily_turkey_loss = 8)
  (h5 : farm.daily_guinea_fowl_loss = 5)
  (h6 : farm.disease_duration = 7)
  (h7 : farm.total_birds_after_disease = 349) :
  initial_guinea_fowls farm = 80 := by
  sorry

#eval initial_guinea_fowls {
  initial_chickens := 300,
  initial_turkeys := 200,
  daily_chicken_loss := 20,
  daily_turkey_loss := 8,
  daily_guinea_fowl_loss := 5,
  disease_duration := 7,
  total_birds_after_disease := 349
}

end NUMINAMATH_CALUDE_initial_guinea_fowls_eq_80_l2867_286719


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_evaluate_at_two_l2867_286761

theorem simplify_and_evaluate (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 0) :
  (1 - 1 / (x + 1)) / (x / (x^2 + 2*x + 1)) = x + 1 := by
  sorry

-- Evaluation for x = 2
theorem evaluate_at_two :
  (1 - 1 / (2 + 1)) / (2 / (2^2 + 2*2 + 1)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_evaluate_at_two_l2867_286761


namespace NUMINAMATH_CALUDE_circle_area_special_condition_l2867_286718

/-- For a circle where three times the reciprocal of its circumference equals half its diameter, 
    the area of the circle is 3/2. -/
theorem circle_area_special_condition (r : ℝ) (h : 3 * (1 / (2 * π * r)) = 1/2 * (2 * r)) : 
  π * r^2 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_special_condition_l2867_286718


namespace NUMINAMATH_CALUDE_weight_of_B_l2867_286735

def weight_problem (A B C : ℝ) : Prop :=
  (A + B + C) / 3 = 45 ∧
  (A + B) / 2 = 40 ∧
  (B + C) / 2 = 41 ∧
  ∃ (x : ℝ), A = 2*x ∧ B = 3*x ∧ C = 5*x ∧
  A + B + C = 144

theorem weight_of_B (A B C : ℝ) (h : weight_problem A B C) : B = 43.2 :=
sorry

end NUMINAMATH_CALUDE_weight_of_B_l2867_286735


namespace NUMINAMATH_CALUDE_three_digit_number_subtraction_l2867_286758

theorem three_digit_number_subtraction (c : ℕ) 
  (h1 : c < 10) 
  (h2 : 2 * c < 10) 
  (h3 : c + 3 < 10) : 
  (100 * (c + 3) + 10 * (2 * c) + c) - (100 * c + 10 * (2 * c) + (c + 3)) ≡ 7 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_subtraction_l2867_286758


namespace NUMINAMATH_CALUDE_cylinder_height_l2867_286725

/-- A cylinder with base diameter equal to height and volume 16π has height 4 -/
theorem cylinder_height (r h : ℝ) (h_positive : 0 < h) (r_positive : 0 < r) : 
  h = 2 * r → π * r^2 * h = 16 * π → h = 4 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_height_l2867_286725


namespace NUMINAMATH_CALUDE_negation_of_existential_proposition_l2867_286716

theorem negation_of_existential_proposition :
  (¬ ∃ x : ℝ, x^2 + 2*x - 3 ≥ 0) ↔ (∀ x : ℝ, x^2 + 2*x - 3 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existential_proposition_l2867_286716


namespace NUMINAMATH_CALUDE_least_k_for_error_bound_l2867_286711

-- Define the sequence u_k
def u : ℕ → ℚ
  | 0 => 1/3
  | k+1 => 2.5 * u k - 3 * (u k)^2

-- Define the limit L
noncomputable def L : ℚ := 2/5

-- Define the error bound
def error_bound : ℚ := 1 / 2^500

-- Theorem statement
theorem least_k_for_error_bound :
  ∃ k : ℕ, (∀ j : ℕ, j < k → |u j - L| > error_bound) ∧
           |u k - L| ≤ error_bound ∧
           k = 5 := by sorry

end NUMINAMATH_CALUDE_least_k_for_error_bound_l2867_286711


namespace NUMINAMATH_CALUDE_legs_minus_twice_heads_l2867_286799

/-- Represents the number of legs for each animal type -/
def legs_per_animal : Fin 3 → ℕ
  | 0 => 2  -- Ducks
  | 1 => 4  -- Cows
  | 2 => 4  -- Buffaloes

/-- The group of animals -/
structure AnimalGroup where
  ducks : ℕ
  cows : ℕ
  buffaloes : ℕ
  buffalo_count_eq : buffaloes = 24

/-- Total number of heads in the group -/
def total_heads (g : AnimalGroup) : ℕ :=
  g.ducks + g.cows + g.buffaloes

/-- Total number of legs in the group -/
def total_legs (g : AnimalGroup) : ℕ :=
  g.ducks * legs_per_animal 0 + g.cows * legs_per_animal 1 + g.buffaloes * legs_per_animal 2

/-- The statement to be proven -/
theorem legs_minus_twice_heads (g : AnimalGroup) :
  total_legs g > 2 * total_heads g →
  total_legs g - 2 * total_heads g = 2 * g.cows + 48 :=
sorry

end NUMINAMATH_CALUDE_legs_minus_twice_heads_l2867_286799


namespace NUMINAMATH_CALUDE_tangent_secant_theorem_l2867_286702

-- Define the triangle ABC and point X
def Triangle (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

def RelativelyPrime (m n : ℕ) : Prop := Nat.gcd m n = 1

theorem tangent_secant_theorem
  (a b c : ℕ)
  (h_triangle : Triangle a b c)
  (h_coprime : RelativelyPrime b c) :
  ∃ (AX CX : ℚ),
    AX = (a * b * c : ℚ) / ((c * c - b * b) : ℚ) ∧
    CX = (a * b * b : ℚ) / ((c * c - b * b) : ℚ) ∧
    (¬ ∃ (n : ℤ), AX = n) ∧
    (¬ ∃ (n : ℤ), CX = n) :=
by sorry

end NUMINAMATH_CALUDE_tangent_secant_theorem_l2867_286702


namespace NUMINAMATH_CALUDE_village_population_l2867_286714

theorem village_population (partial_population : ℕ) (percentage : ℚ) (total_population : ℕ) :
  percentage = 80 / 100 →
  partial_population = 23040 →
  (percentage : ℚ) * total_population = partial_population →
  total_population = 28800 := by
sorry

end NUMINAMATH_CALUDE_village_population_l2867_286714


namespace NUMINAMATH_CALUDE_australians_in_group_l2867_286745

theorem australians_in_group (total : Nat) (chinese : Nat) (americans : Nat) 
  (h1 : total = 49)
  (h2 : chinese = 22)
  (h3 : americans = 16) :
  total - (chinese + americans) = 11 := by
  sorry

end NUMINAMATH_CALUDE_australians_in_group_l2867_286745


namespace NUMINAMATH_CALUDE_sector_central_angle_l2867_286717

/-- Given a circular sector with radius 8 cm and area 4 cm², 
    prove that its central angle measures 1/8 radians. -/
theorem sector_central_angle (radius : ℝ) (area : ℝ) (angle : ℝ) :
  radius = 8 →
  area = 4 →
  area = 1/2 * angle * radius^2 →
  angle = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l2867_286717


namespace NUMINAMATH_CALUDE_negation_of_forall_positive_negation_of_greater_than_zero_l2867_286760

theorem negation_of_forall_positive (P : ℝ → Prop) :
  (¬ ∀ x : ℝ, P x) ↔ (∃ x : ℝ, ¬ P x) :=
by sorry

theorem negation_of_greater_than_zero :
  (¬ ∀ x : ℝ, x^2 + x + 1 > 0) ↔ (∃ x : ℝ, x^2 + x + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_forall_positive_negation_of_greater_than_zero_l2867_286760


namespace NUMINAMATH_CALUDE_triple_addichiffrer_1998_power_l2867_286700

/-- The addichiffrer function adds all digits of a natural number. -/
def addichiffrer (n : ℕ) : ℕ := sorry

/-- Apply addichiffrer process three times to a given number. -/
def triple_addichiffrer (n : ℕ) : ℕ := 
  addichiffrer (addichiffrer (addichiffrer n))

/-- Theorem stating that applying addichiffrer three times to 1998^1998 results in 9. -/
theorem triple_addichiffrer_1998_power : triple_addichiffrer (1998^1998) = 9 := by sorry

end NUMINAMATH_CALUDE_triple_addichiffrer_1998_power_l2867_286700


namespace NUMINAMATH_CALUDE_interest_rate_problem_l2867_286797

/-- Calculates the amount after simple interest is applied -/
def amountAfterSimpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

theorem interest_rate_problem (originalRate : ℝ) :
  let principal : ℝ := 1000
  let time : ℝ := 5
  let increasedRate : ℝ := originalRate + 0.05
  amountAfterSimpleInterest principal increasedRate time = 1750 →
  amountAfterSimpleInterest principal originalRate time = 1500 :=
by
  sorry

end NUMINAMATH_CALUDE_interest_rate_problem_l2867_286797


namespace NUMINAMATH_CALUDE_tetrahedron_probabilities_l2867_286743

/-- A regular tetrahedron with numbers 1, 2, 3, 4 on its faces -/
structure Tetrahedron :=
  (faces : Fin 4 → Fin 4)
  (bijective : Function.Bijective faces)

/-- The probability space of throwing the tetrahedron twice -/
def TetrahedronThrows := Tetrahedron × Tetrahedron

/-- Event A: 2 or 3 facing down on first throw -/
def event_A (t : TetrahedronThrows) : Prop :=
  t.1.faces 0 = 2 ∨ t.1.faces 0 = 3

/-- Event B: sum of numbers facing down is odd -/
def event_B (t : TetrahedronThrows) : Prop :=
  (t.1.faces 0 + t.2.faces 0) % 2 = 1

/-- Event C: sum of numbers facing up is not less than 15 -/
def event_C (t : TetrahedronThrows) : Prop :=
  (10 - t.1.faces 0 - t.2.faces 0) ≥ 15

/-- The probability measure on TetrahedronThrows -/
noncomputable def P : Set TetrahedronThrows → ℝ := sorry

theorem tetrahedron_probabilities :
  (P {t : TetrahedronThrows | event_A t} * P {t : TetrahedronThrows | event_B t} =
   P {t : TetrahedronThrows | event_A t ∧ event_B t}) ∧
  (P {t : TetrahedronThrows | event_A t ∨ event_B t} = 3/4) ∧
  (P {t : TetrahedronThrows | event_C t} = 5/8) :=
sorry

end NUMINAMATH_CALUDE_tetrahedron_probabilities_l2867_286743


namespace NUMINAMATH_CALUDE_largest_fraction_l2867_286778

theorem largest_fraction (w x y z : ℝ) (h1 : w < x) (h2 : x < y) (h3 : y < z) :
  (z / w > x / w) ∧ (z / w > y / x) ∧ (z / w > y / w) ∧ (z / w > z / x) :=
sorry

end NUMINAMATH_CALUDE_largest_fraction_l2867_286778


namespace NUMINAMATH_CALUDE_sue_made_22_buttons_l2867_286740

def mari_buttons : ℕ := 8

def kendra_buttons (m : ℕ) : ℕ := 5 * m + 4

def sue_buttons (k : ℕ) : ℕ := k / 2

theorem sue_made_22_buttons : 
  sue_buttons (kendra_buttons mari_buttons) = 22 := by
  sorry

end NUMINAMATH_CALUDE_sue_made_22_buttons_l2867_286740


namespace NUMINAMATH_CALUDE_pizza_calories_l2867_286772

theorem pizza_calories (total_slices : ℕ) (eaten_slices_1 : ℕ) (calories_1 : ℕ) 
  (eaten_slices_2 : ℕ) (calories_2 : ℕ) : 
  total_slices = 12 → 
  eaten_slices_1 = 3 →
  calories_1 = 300 →
  eaten_slices_2 = 4 →
  calories_2 = 400 →
  eaten_slices_1 * calories_1 + eaten_slices_2 * calories_2 = 2500 := by
  sorry

end NUMINAMATH_CALUDE_pizza_calories_l2867_286772


namespace NUMINAMATH_CALUDE_total_experienced_monthly_earnings_l2867_286773

def total_sailors : ℕ := 30
def inexperienced_sailors : ℕ := 8
def group_a_sailors : ℕ := 12
def group_b_sailors : ℕ := 10
def inexperienced_hourly_wage : ℚ := 12
def group_a_wage_multiplier : ℚ := 4/3
def group_b_wage_multiplier : ℚ := 5/4
def group_a_weekly_hours : ℕ := 50
def group_b_weekly_hours : ℕ := 60
def weeks_per_month : ℕ := 4

def group_a_hourly_wage : ℚ := inexperienced_hourly_wage * group_a_wage_multiplier
def group_b_hourly_wage : ℚ := inexperienced_hourly_wage * group_b_wage_multiplier

def group_a_monthly_earnings : ℚ := group_a_hourly_wage * group_a_weekly_hours * weeks_per_month * group_a_sailors
def group_b_monthly_earnings : ℚ := group_b_hourly_wage * group_b_weekly_hours * weeks_per_month * group_b_sailors

theorem total_experienced_monthly_earnings :
  group_a_monthly_earnings + group_b_monthly_earnings = 74400 := by
  sorry

end NUMINAMATH_CALUDE_total_experienced_monthly_earnings_l2867_286773


namespace NUMINAMATH_CALUDE_calculation_result_l2867_286738

theorem calculation_result : 8 * 5.4 - 0.6 * 10 / 1.2 = 38.2 := by
  sorry

end NUMINAMATH_CALUDE_calculation_result_l2867_286738


namespace NUMINAMATH_CALUDE_savings_account_relationship_l2867_286774

/-- The function representing the total amount in an education savings account -/
def savings_account (monthly_rate : ℝ) (initial_deposit : ℝ) (months : ℝ) : ℝ :=
  monthly_rate * initial_deposit * months + initial_deposit

/-- Theorem stating the relationship between total amount and number of months -/
theorem savings_account_relationship :
  let monthly_rate : ℝ := 0.0022  -- 0.22%
  let initial_deposit : ℝ := 1000
  ∀ x : ℝ, savings_account monthly_rate initial_deposit x = 2.2 * x + 1000 := by
  sorry

end NUMINAMATH_CALUDE_savings_account_relationship_l2867_286774


namespace NUMINAMATH_CALUDE_product_ab_l2867_286780

theorem product_ab (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 35) : a * b = 13 := by
  sorry

end NUMINAMATH_CALUDE_product_ab_l2867_286780


namespace NUMINAMATH_CALUDE_point_outside_circle_iff_a_in_range_l2867_286715

-- Define the circle equation
def circle_equation (x y a : ℝ) : Prop := x^2 + y^2 - x + y + a = 0

-- Define what it means for a point to be outside the circle
def point_outside_circle (x y a : ℝ) : Prop := x^2 + y^2 - x + y + a > 0

-- Theorem statement
theorem point_outside_circle_iff_a_in_range :
  ∀ a : ℝ, point_outside_circle 2 1 a ↔ -4 < a ∧ a < 1/2 := by sorry

end NUMINAMATH_CALUDE_point_outside_circle_iff_a_in_range_l2867_286715


namespace NUMINAMATH_CALUDE_parabola_transformation_l2867_286733

/-- A parabola in the xy-plane -/
structure Parabola where
  f : ℝ → ℝ

/-- Shift a parabola horizontally and vertically -/
def shift (p : Parabola) (h v : ℝ) : Parabola where
  f x := p.f (x - h) + v

/-- The original parabola y = 2x^2 -/
def original_parabola : Parabola where
  f x := 2 * x^2

/-- The transformed parabola -/
def transformed_parabola : Parabola :=
  shift (shift original_parabola 3 0) 0 (-4)

theorem parabola_transformation :
  ∀ x, transformed_parabola.f x = 2 * (x + 3)^2 - 4 := by sorry

end NUMINAMATH_CALUDE_parabola_transformation_l2867_286733


namespace NUMINAMATH_CALUDE_f_max_min_on_interval_l2867_286764

-- Define the function f(x) = -x^3 + 3x^2
def f (x : ℝ) : ℝ := -x^3 + 3*x^2

-- Define the interval [-2, 2]
def interval : Set ℝ := Set.Icc (-2) 2

-- Theorem statement
theorem f_max_min_on_interval :
  ∃ (max min : ℝ),
    (∀ x ∈ interval, f x ≤ max) ∧
    (∃ x ∈ interval, f x = max) ∧
    (∀ x ∈ interval, min ≤ f x) ∧
    (∃ x ∈ interval, f x = min) ∧
    max = 20 ∧ min = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_max_min_on_interval_l2867_286764


namespace NUMINAMATH_CALUDE_purple_balls_count_l2867_286783

theorem purple_balls_count (total_balls : ℕ) (white_balls : ℕ) (green_balls : ℕ) (yellow_balls : ℕ) (red_balls : ℕ) 
  (h1 : total_balls = 60)
  (h2 : white_balls = 22)
  (h3 : green_balls = 18)
  (h4 : yellow_balls = 2)
  (h5 : red_balls = 15)
  (h6 : (white_balls + green_balls + yellow_balls : ℚ) / total_balls = 7/10) :
  ∃ (purple_balls : ℕ), purple_balls = 3 ∧ total_balls = white_balls + green_balls + yellow_balls + red_balls + purple_balls :=
by
  sorry


end NUMINAMATH_CALUDE_purple_balls_count_l2867_286783


namespace NUMINAMATH_CALUDE_book_selection_theorem_l2867_286771

theorem book_selection_theorem (n m : ℕ) (h1 : n = 8) (h2 : m = 5) :
  (Nat.choose (n - 1) (m - 1)) = 35 := by
  sorry

end NUMINAMATH_CALUDE_book_selection_theorem_l2867_286771


namespace NUMINAMATH_CALUDE_complement_M_equals_expected_l2867_286787

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define the set M
def M : Set Nat := {1, 2}

-- Define the complement of M with respect to U
def complement_M : Set Nat := U \ M

-- Theorem statement
theorem complement_M_equals_expected : complement_M = {3, 4, 5} := by
  sorry

end NUMINAMATH_CALUDE_complement_M_equals_expected_l2867_286787
