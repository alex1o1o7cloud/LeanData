import Mathlib

namespace NUMINAMATH_CALUDE_fires_put_out_l1976_197654

/-- The number of fires Doug put out -/
def doug_fires : ℕ := 20

/-- The number of fires Kai put out -/
def kai_fires : ℕ := 3 * doug_fires

/-- The number of fires Eli put out -/
def eli_fires : ℕ := kai_fires / 2

/-- The total number of fires put out by Doug, Kai, and Eli -/
def total_fires : ℕ := doug_fires + kai_fires + eli_fires

theorem fires_put_out : total_fires = 110 := by
  sorry

end NUMINAMATH_CALUDE_fires_put_out_l1976_197654


namespace NUMINAMATH_CALUDE_water_jar_problem_l1976_197617

theorem water_jar_problem (c1 c2 c3 : ℝ) (h1 : c1 > 0) (h2 : c2 > 0) (h3 : c3 > 0) 
  (h4 : c1 < c2) (h5 : c2 < c3) 
  (h6 : c1 / 6 = c2 / 5) (h7 : c2 / 5 = c3 / 7) : 
  (c1 / 6 + c2 / 5) / c3 = 2 / 7 := by
sorry

end NUMINAMATH_CALUDE_water_jar_problem_l1976_197617


namespace NUMINAMATH_CALUDE_probability_odd_product_sum_div_5_l1976_197662

def is_odd (n : ℕ) : Prop := n % 2 = 1

def is_divisible_by_5 (n : ℕ) : Prop := n % 5 = 0

def valid_pair (a b : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 20 ∧ 1 ≤ b ∧ b ≤ 20 ∧ a ≠ b ∧
  is_odd a ∧ is_odd b ∧ is_divisible_by_5 (a + b)

def total_pairs : ℕ := 190

def valid_pairs : ℕ := 6

theorem probability_odd_product_sum_div_5 :
  (valid_pairs : ℚ) / total_pairs = 3 / 95 := by sorry

end NUMINAMATH_CALUDE_probability_odd_product_sum_div_5_l1976_197662


namespace NUMINAMATH_CALUDE_inverse_function_property_l1976_197613

theorem inverse_function_property (f : ℝ → ℝ) (h_inv : Function.Bijective f) :
  (Function.invFun f) 3 = 1 → f 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_function_property_l1976_197613


namespace NUMINAMATH_CALUDE_garden_tomatoes_count_l1976_197679

/-- Represents the garden layout and vegetable distribution --/
structure Garden where
  tomato_kinds : Nat
  cucumber_kinds : Nat
  cucumbers_per_kind : Nat
  potatoes : Nat
  rows : Nat
  spaces_per_row : Nat
  additional_capacity : Nat

/-- Calculates the number of tomatoes of each kind in the garden --/
def tomatoes_per_kind (g : Garden) : Nat :=
  let total_spaces := g.rows * g.spaces_per_row
  let occupied_spaces := g.cucumber_kinds * g.cucumbers_per_kind + g.potatoes
  let tomato_spaces := total_spaces - occupied_spaces - g.additional_capacity
  tomato_spaces / g.tomato_kinds

/-- Theorem stating that for the given garden configuration, 
    there are 5 tomatoes of each kind --/
theorem garden_tomatoes_count :
  let g : Garden := {
    tomato_kinds := 3,
    cucumber_kinds := 5,
    cucumbers_per_kind := 4,
    potatoes := 30,
    rows := 10,
    spaces_per_row := 15,
    additional_capacity := 85
  }
  tomatoes_per_kind g = 5 := by sorry

end NUMINAMATH_CALUDE_garden_tomatoes_count_l1976_197679


namespace NUMINAMATH_CALUDE_ellipse_axis_endpoint_distance_l1976_197632

/-- Given an ellipse with equation 9(x-1)^2 + y^2 = 36, 
    the distance between an endpoint of its major axis 
    and an endpoint of its minor axis is 2√10 -/
theorem ellipse_axis_endpoint_distance : 
  ∃ (A B : ℝ × ℝ),
    (∀ (x y : ℝ), 9 * (x - 1)^2 + y^2 = 36 → 
      ((x = A.1 ∧ y = A.2) ∨ (x = -A.1 ∧ y = -A.2)) ∨ 
      ((x = B.1 ∧ y = B.2) ∨ (x = -B.1 ∧ y = -B.2))) →
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 40 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_axis_endpoint_distance_l1976_197632


namespace NUMINAMATH_CALUDE_square_eq_product_sum_seven_l1976_197608

theorem square_eq_product_sum_seven (a b : ℕ) : a^2 = b * (b + 7) ↔ (a = 0 ∧ b = 0) ∨ (a = 12 ∧ b = 9) := by
  sorry

end NUMINAMATH_CALUDE_square_eq_product_sum_seven_l1976_197608


namespace NUMINAMATH_CALUDE_union_equals_universal_l1976_197665

def U : Set Nat := {2, 3, 4, 5, 6, 7}
def M : Set Nat := {3, 4, 5, 7}
def N : Set Nat := {2, 4, 5, 6}

theorem union_equals_universal : M ∪ N = U := by
  sorry

end NUMINAMATH_CALUDE_union_equals_universal_l1976_197665


namespace NUMINAMATH_CALUDE_orange_price_problem_l1976_197685

/-- Proof of the orange price problem --/
theorem orange_price_problem 
  (apple_price : ℚ) 
  (total_fruits : ℕ) 
  (initial_avg_price : ℚ) 
  (oranges_removed : ℕ) 
  (final_avg_price : ℚ) 
  (h1 : apple_price = 40/100)
  (h2 : total_fruits = 10)
  (h3 : initial_avg_price = 56/100)
  (h4 : oranges_removed = 6)
  (h5 : final_avg_price = 50/100) :
  ∃ (orange_price : ℚ), orange_price = 60/100 := by
sorry


end NUMINAMATH_CALUDE_orange_price_problem_l1976_197685


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1976_197631

theorem quadratic_inequality (x : ℝ) : -3 * x^2 + 9 * x + 6 > 0 ↔ x < -1 ∨ x > 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1976_197631


namespace NUMINAMATH_CALUDE_fraction_not_going_on_trip_l1976_197699

theorem fraction_not_going_on_trip :
  ∀ (S : ℝ) (J : ℝ),
    S > 0 →
    J = (2/3) * S →
    ((3/4) * J + (1/3) * S) / (J + S) = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_not_going_on_trip_l1976_197699


namespace NUMINAMATH_CALUDE_estimate_above_120_l1976_197606

/-- Represents the score distribution of a class -/
structure ScoreDistribution where
  total_students : ℕ
  mean : ℝ
  std_dev : ℝ
  prob_100_to_110 : ℝ

/-- Estimates the number of students scoring above a given threshold -/
def estimate_students_above (sd : ScoreDistribution) (threshold : ℝ) : ℕ := sorry

/-- The main theorem to prove -/
theorem estimate_above_120 (sd : ScoreDistribution) 
  (h1 : sd.total_students = 50)
  (h2 : sd.mean = 110)
  (h3 : sd.std_dev = 10)
  (h4 : sd.prob_100_to_110 = 0.36) :
  estimate_students_above sd 120 = 7 := by sorry

end NUMINAMATH_CALUDE_estimate_above_120_l1976_197606


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l1976_197642

theorem smallest_integer_with_remainders : ∃ b : ℕ, 
  b > 0 ∧
  b % 5 = 1 ∧
  b % 4 = 2 ∧
  b % 7 = 3 ∧
  ∀ c : ℕ, c > 0 → c % 5 = 1 → c % 4 = 2 → c % 7 = 3 → b ≤ c :=
by
  use 86
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l1976_197642


namespace NUMINAMATH_CALUDE_tan_double_angle_solution_l1976_197646

theorem tan_double_angle_solution (α : ℝ) (h : Real.tan (2 * α) = 4 / 3) :
  Real.tan α = -2 ∨ Real.tan α = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_solution_l1976_197646


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l1976_197605

theorem completing_square_equivalence (x : ℝ) : 
  (x^2 - 8*x + 1 = 0) ↔ ((x - 4)^2 = 15) := by
  sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l1976_197605


namespace NUMINAMATH_CALUDE_equation_solutions_l1976_197601

theorem equation_solutions :
  (∀ x, x^2 - 7*x = 0 ↔ x = 0 ∨ x = 7) ∧
  (∀ x, 2*x^2 - 6*x + 1 = 0 ↔ x = (3 + Real.sqrt 7) / 2 ∨ x = (3 - Real.sqrt 7) / 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1976_197601


namespace NUMINAMATH_CALUDE_pyramid_volume_l1976_197652

theorem pyramid_volume (base_length base_width height : ℝ) 
  (h1 : base_length = 2/3)
  (h2 : base_width = 1/2)
  (h3 : height = 1) : 
  (1/3 : ℝ) * base_length * base_width * height = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_volume_l1976_197652


namespace NUMINAMATH_CALUDE_fisher_min_score_l1976_197607

/-- Represents a student's scores and eligibility requirements -/
structure StudentScores where
  algebra_sem1 : ℝ
  statistics : ℝ
  algebra_requirement : ℝ
  statistics_requirement : ℝ

/-- Calculates the minimum score needed in the second semester of Algebra -/
def min_algebra_sem2_score (s : StudentScores) : ℝ :=
  2 * s.algebra_requirement - s.algebra_sem1

/-- Theorem stating the minimum score Fisher needs in second semester Algebra -/
theorem fisher_min_score (fisher : StudentScores)
  (h1 : fisher.algebra_sem1 = 84)
  (h2 : fisher.statistics = 82)
  (h3 : fisher.algebra_requirement = 85)
  (h4 : fisher.statistics_requirement = 80)
  (h5 : fisher.statistics ≥ fisher.statistics_requirement) :
  min_algebra_sem2_score fisher = 86 := by
  sorry

#eval min_algebra_sem2_score ⟨84, 82, 85, 80⟩

end NUMINAMATH_CALUDE_fisher_min_score_l1976_197607


namespace NUMINAMATH_CALUDE_dollar_composition_60_l1976_197664

-- Define the $ operation
def dollar (x : ℝ) : ℝ := 0.4 * x + 2

-- State the theorem
theorem dollar_composition_60 : dollar (dollar (dollar 60)) = 6.96 := by
  sorry

end NUMINAMATH_CALUDE_dollar_composition_60_l1976_197664


namespace NUMINAMATH_CALUDE_combined_cost_theorem_l1976_197674

/-- Calculate the total cost of a stock given its face value, discount/premium rate, and brokerage rate -/
def stockCost (faceValue : ℚ) (discountPremiumRate : ℚ) (brokerageRate : ℚ) : ℚ :=
  let adjustedValue := faceValue * (1 + discountPremiumRate)
  adjustedValue * (1 + brokerageRate)

/-- The combined cost of stocks A, B, and C -/
def combinedCost : ℚ :=
  stockCost 100 (-0.02) (1/500) +  -- Stock A
  stockCost 150 0.015 (1/600) +    -- Stock B
  stockCost 200 (-0.03) (1/200)    -- Stock C

theorem combined_cost_theorem :
  combinedCost = 445669750/1000000 := by sorry

end NUMINAMATH_CALUDE_combined_cost_theorem_l1976_197674


namespace NUMINAMATH_CALUDE_intersection_nonempty_implies_b_range_l1976_197669

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + 2*p.2^2 = 3}
def N (m b : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = m*p.1 + b}

-- State the theorem
theorem intersection_nonempty_implies_b_range :
  (∀ m : ℝ, (M ∩ N m b).Nonempty) →
  b ∈ Set.Icc (-Real.sqrt 6 / 2) (Real.sqrt 6 / 2) :=
sorry

end NUMINAMATH_CALUDE_intersection_nonempty_implies_b_range_l1976_197669


namespace NUMINAMATH_CALUDE_chairs_per_row_l1976_197637

theorem chairs_per_row (total_chairs : ℕ) (num_rows : ℕ) (h1 : total_chairs = 432) (h2 : num_rows = 27) :
  total_chairs / num_rows = 16 := by
  sorry

end NUMINAMATH_CALUDE_chairs_per_row_l1976_197637


namespace NUMINAMATH_CALUDE_joaos_chocolates_l1976_197672

theorem joaos_chocolates :
  ∃! n : ℕ, 30 < n ∧ n < 100 ∧ n % 7 = 1 ∧ n % 10 = 2 ∧ n = 92 := by
  sorry

end NUMINAMATH_CALUDE_joaos_chocolates_l1976_197672


namespace NUMINAMATH_CALUDE_average_food_expense_percentage_l1976_197658

/-- Calculate the average percentage of income spent on food over two months --/
theorem average_food_expense_percentage (jan_income feb_income : ℚ)
  (jan_petrol feb_petrol : ℚ) : 
  jan_income = 3000 →
  feb_income = 4000 →
  jan_petrol = 300 →
  feb_petrol = 400 →
  let jan_remaining := jan_income - jan_petrol
  let feb_remaining := feb_income - feb_petrol
  let jan_rent := jan_remaining * (14 / 100)
  let feb_rent := feb_remaining * (14 / 100)
  let jan_clothing := jan_income * (10 / 100)
  let feb_clothing := feb_income * (10 / 100)
  let jan_utility := jan_income * (5 / 100)
  let feb_utility := feb_income * (5 / 100)
  let jan_food := jan_remaining - jan_rent - jan_clothing - jan_utility
  let feb_food := feb_remaining - feb_rent - feb_clothing - feb_utility
  let total_food := jan_food + feb_food
  let total_income := jan_income + feb_income
  let avg_food_percentage := (total_food / total_income) * 100
  avg_food_percentage = 62.4 := by
  sorry

end NUMINAMATH_CALUDE_average_food_expense_percentage_l1976_197658


namespace NUMINAMATH_CALUDE_gamblers_initial_win_rate_l1976_197660

theorem gamblers_initial_win_rate 
  (initial_games : ℕ) 
  (additional_games : ℕ) 
  (new_win_rate : ℚ) 
  (final_win_rate : ℚ) :
  initial_games = 30 →
  additional_games = 30 →
  new_win_rate = 4/5 →
  final_win_rate = 3/5 →
  ∃ (initial_win_rate : ℚ),
    initial_win_rate = 2/5 ∧
    (initial_win_rate * initial_games + new_win_rate * additional_games) / (initial_games + additional_games) = final_win_rate :=
by sorry

end NUMINAMATH_CALUDE_gamblers_initial_win_rate_l1976_197660


namespace NUMINAMATH_CALUDE_simplify_fraction_l1976_197640

theorem simplify_fraction (a : ℚ) (h : a = 3) : 10 * a^3 / (55 * a^2) = 6 / 11 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1976_197640


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_in_expansion_l1976_197695

theorem coefficient_x_cubed_in_expansion : 
  let n : ℕ := 5
  let k : ℕ := 3
  let a : ℤ := 1
  let b : ℤ := -2
  (n.choose k) * (b ^ k) * (a ^ (n - k)) = -80 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_in_expansion_l1976_197695


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1976_197611

def geometric_sequence (a : ℕ → ℝ) (r : ℝ) :=
  ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum
  (a : ℕ → ℝ)
  (h1 : geometric_sequence (fun n => a (n + 1) - a n) (1/3))
  (h2 : a 1 - a 0 = 1) :
  ∀ n : ℕ, a n = (3/2) * (1 - (1/3)^n) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1976_197611


namespace NUMINAMATH_CALUDE_factor_expression_l1976_197681

theorem factor_expression (a : ℝ) : 198 * a^2 + 36 * a + 54 = 18 * (11 * a^2 + 2 * a + 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1976_197681


namespace NUMINAMATH_CALUDE_sample_size_proof_l1976_197610

theorem sample_size_proof (f1 f2 f3 : ℝ) (h1 : f1 = 10) (h2 : f2 = 0.35) (h3 : f3 = 0.45) : 
  ∃ M : ℝ, M = f1 / (1 - f2 - f3) ∧ M = 50 := by
sorry

end NUMINAMATH_CALUDE_sample_size_proof_l1976_197610


namespace NUMINAMATH_CALUDE_abc_sum_sqrt_l1976_197687

theorem abc_sum_sqrt (a b c : ℝ) 
  (h1 : b + c = 16)
  (h2 : c + a = 18)
  (h3 : a + b = 20) :
  Real.sqrt (a * b * c * (a + b + c)) = 231 := by
  sorry

end NUMINAMATH_CALUDE_abc_sum_sqrt_l1976_197687


namespace NUMINAMATH_CALUDE_min_value_a_l1976_197627

theorem min_value_a (x y a : ℝ) (hx : x > 0) (hy : y > 0) (ha : a > 0) :
  (∀ x y, x > 0 → y > 0 → Real.sqrt x + Real.sqrt y ≤ a * Real.sqrt (x + y)) →
  a ≥ Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_l1976_197627


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l1976_197625

/-- The quadratic equation (k-1)x^2 + 2x - 2 = 0 has two distinct real roots if and only if k > 1/2 and k ≠ 1 -/
theorem quadratic_two_distinct_roots (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (k - 1) * x₁^2 + 2 * x₁ - 2 = 0 ∧ (k - 1) * x₂^2 + 2 * x₂ - 2 = 0) ↔ 
  (k > 1/2 ∧ k ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l1976_197625


namespace NUMINAMATH_CALUDE_parsnip_box_ratio_l1976_197655

/-- Represents the number of parsnips in a full box -/
def full_box_capacity : ℕ := 20

/-- Represents the total number of boxes in an average harvest -/
def total_boxes : ℕ := 20

/-- Represents the total number of parsnips in an average harvest -/
def total_parsnips : ℕ := 350

/-- Represents the number of full boxes -/
def full_boxes : ℕ := 15

/-- Represents the number of half-full boxes -/
def half_full_boxes : ℕ := total_boxes - full_boxes

theorem parsnip_box_ratio :
  (full_boxes : ℚ) / total_boxes = 3 / 4 ∧
  full_boxes + half_full_boxes = total_boxes ∧
  full_boxes * full_box_capacity + half_full_boxes * (full_box_capacity / 2) = total_parsnips :=
by sorry

end NUMINAMATH_CALUDE_parsnip_box_ratio_l1976_197655


namespace NUMINAMATH_CALUDE_monic_quartic_polynomial_value_l1976_197678

-- Define a monic quartic polynomial
def MonicQuarticPolynomial (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, f x = x^4 + a*x^3 + b*x^2 + c*x + (f 0)

-- State the theorem
theorem monic_quartic_polynomial_value (f : ℝ → ℝ) :
  MonicQuarticPolynomial f →
  f (-1) = -1 →
  f 2 = -4 →
  f (-3) = -9 →
  f 4 = -16 →
  f 1 = 23 := by
  sorry

end NUMINAMATH_CALUDE_monic_quartic_polynomial_value_l1976_197678


namespace NUMINAMATH_CALUDE_current_speed_l1976_197684

/-- Given a boat that moves upstream at 1 km in 40 minutes and downstream at 1 km in 12 minutes,
    prove that the speed of the current is 1.75 km/h. -/
theorem current_speed (upstream_speed : ℝ) (downstream_speed : ℝ)
    (h1 : upstream_speed = 1 / (40 / 60))  -- 1 km in 40 minutes converted to km/h
    (h2 : downstream_speed = 1 / (12 / 60))  -- 1 km in 12 minutes converted to km/h
    : (downstream_speed - upstream_speed) / 2 = 1.75 := by
  sorry

end NUMINAMATH_CALUDE_current_speed_l1976_197684


namespace NUMINAMATH_CALUDE_sons_age_l1976_197649

theorem sons_age (mother_age son_age : ℕ) : 
  mother_age = 4 * son_age →
  mother_age + son_age = 49 + 6 →
  son_age = 11 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l1976_197649


namespace NUMINAMATH_CALUDE_enclosure_blocks_count_l1976_197677

/-- Calculates the number of cubical blocks used to create a cuboidal enclosure --/
def cubicalBlocksCount (length width height thickness : ℕ) : ℕ :=
  length * width * height - (length - 2 * thickness) * (width - 2 * thickness) * (height - thickness)

/-- Theorem stating that the number of cubical blocks for the given dimensions is 372 --/
theorem enclosure_blocks_count :
  cubicalBlocksCount 15 8 7 1 = 372 := by
  sorry

end NUMINAMATH_CALUDE_enclosure_blocks_count_l1976_197677


namespace NUMINAMATH_CALUDE_sine_curve_intersection_l1976_197682

theorem sine_curve_intersection (A a : ℝ) (h1 : A > 0) (h2 : a > 0) :
  (∃ x1 x2 x3 x4 : ℝ, 
    0 ≤ x1 ∧ x1 < x2 ∧ x2 < x3 ∧ x3 < x4 ∧ x4 ≤ 2 * π ∧
    A * Real.sin x1 + a = 2 ∧
    A * Real.sin x2 + a = -1 ∧
    A * Real.sin x3 + a = -1 ∧
    A * Real.sin x4 + a = 2 ∧
    (x2 - x1) = (x4 - x3) ∧
    x2 ≠ x1) →
  a = 1/2 ∧ A > 3/2 := by
sorry

end NUMINAMATH_CALUDE_sine_curve_intersection_l1976_197682


namespace NUMINAMATH_CALUDE_one_fifth_of_eight_point_five_l1976_197686

theorem one_fifth_of_eight_point_five : (8.5 : ℚ) / 5 = 17 / 10 := by
  sorry

end NUMINAMATH_CALUDE_one_fifth_of_eight_point_five_l1976_197686


namespace NUMINAMATH_CALUDE_min_sum_squares_l1976_197689

theorem min_sum_squares (a b : ℝ) : 
  (∃ x : ℝ, x^4 + a*x^3 + b*x^2 + a*x + 1 = 0) → a^2 + b^2 ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1976_197689


namespace NUMINAMATH_CALUDE_job_choice_diploma_percentage_l1976_197667

theorem job_choice_diploma_percentage :
  let total_population : ℝ := 100
  let no_diploma_with_job : ℝ := 18
  let with_job_choice : ℝ := 40
  let with_diploma : ℝ := 37
  let without_job_choice : ℝ := total_population - with_job_choice
  let with_diploma_without_job : ℝ := with_diploma - (with_job_choice - no_diploma_with_job)
  (with_diploma_without_job / without_job_choice) * 100 = 25 := by
sorry

end NUMINAMATH_CALUDE_job_choice_diploma_percentage_l1976_197667


namespace NUMINAMATH_CALUDE_tiling_ways_eq_fib_l1976_197651

/-- The number of ways to tile a 2 × n strip with 1 × 2 or 2 × 1 bricks -/
def tiling_ways (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | k + 2 => tiling_ways (k + 1) + tiling_ways k

/-- The Fibonacci sequence -/
def fib (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | k + 2 => fib (k + 1) + fib k

theorem tiling_ways_eq_fib (n : ℕ) : tiling_ways n = fib n := by
  sorry

end NUMINAMATH_CALUDE_tiling_ways_eq_fib_l1976_197651


namespace NUMINAMATH_CALUDE_propositions_truthfulness_l1976_197697

-- Define the properties
def isPositiveEven (n : ℕ) : Prop := n > 0 ∧ n % 2 = 0
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → n % d ≠ 0

-- Theorem statement
theorem propositions_truthfulness :
  (∃ n : ℕ, isPositiveEven n ∧ isPrime n) ∧
  (∃ n : ℕ, ¬isPrime n ∧ ¬isPositiveEven n) ∧
  (∃ n : ℕ, ¬isPositiveEven n ∧ ¬isPrime n) ∧
  (∀ n : ℕ, isPrime n → ¬isPositiveEven n) :=
sorry

end NUMINAMATH_CALUDE_propositions_truthfulness_l1976_197697


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1976_197620

theorem complex_equation_solution (z : ℂ) : (1 + Complex.I) * z = Complex.I → z = (1/2 : ℂ) + (1/2 : ℂ) * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1976_197620


namespace NUMINAMATH_CALUDE_cubic_root_sum_l1976_197638

theorem cubic_root_sum (p q r : ℝ) : 
  0 < p ∧ p < 1 ∧ 
  0 < q ∧ q < 1 ∧ 
  0 < r ∧ r < 1 ∧ 
  p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
  30 * p^3 - 50 * p^2 + 22 * p - 1 = 0 ∧
  30 * q^3 - 50 * q^2 + 22 * q - 1 = 0 ∧
  30 * r^3 - 50 * r^2 + 22 * r - 1 = 0 →
  1 / (1 - p) + 1 / (1 - q) + 1 / (1 - r) = 12 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l1976_197638


namespace NUMINAMATH_CALUDE_sequence_existence_theorem_l1976_197694

theorem sequence_existence_theorem :
  (¬ ∃ (a : ℕ → ℕ), ∀ n, (a (n + 1))^2 ≥ 2 * (a n) * (a (n + 2))) ∧
  (∃ (a : ℤ → ℝ), (∀ n, Irrational (a n)) ∧ (∀ n, (a (n - 1))^2 ≥ 2 * (a n) * (a (n - 2)))) :=
by sorry

end NUMINAMATH_CALUDE_sequence_existence_theorem_l1976_197694


namespace NUMINAMATH_CALUDE_souvenir_cost_in_usd_l1976_197691

/-- Calculates the cost in USD given the cost in yen and the exchange rate -/
def cost_in_usd (cost_yen : ℚ) (exchange_rate : ℚ) : ℚ :=
  cost_yen / exchange_rate

theorem souvenir_cost_in_usd :
  let cost_yen : ℚ := 500
  let exchange_rate : ℚ := 120
  cost_in_usd cost_yen exchange_rate = 25 / 6 := by sorry

end NUMINAMATH_CALUDE_souvenir_cost_in_usd_l1976_197691


namespace NUMINAMATH_CALUDE_batsman_average_runs_l1976_197671

/-- The average runs scored by a batsman in a series of cricket matches. -/
def AverageRuns (first_10_avg : ℝ) (total_matches : ℕ) (overall_avg : ℝ) : Prop :=
  let first_10_total := first_10_avg * 10
  let total_runs := overall_avg * total_matches
  let next_10_total := total_runs - first_10_total
  let next_10_avg := next_10_total / 10
  next_10_avg = 30

/-- Theorem stating that given the conditions, the average runs scored in the next 10 matches is 30. -/
theorem batsman_average_runs : AverageRuns 40 20 35 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_runs_l1976_197671


namespace NUMINAMATH_CALUDE_unique_integer_solution_l1976_197619

theorem unique_integer_solution (m : ℤ) : 
  (∃! (x : ℤ), |2*x - m| ≤ 1 ∧ x = 2) → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l1976_197619


namespace NUMINAMATH_CALUDE_problem_solution_l1976_197612

theorem problem_solution (a b c : ℝ) 
  (eq : a * b * c = Real.sqrt ((a + 2) * (b + 3)) / (c + 1))
  (h1 : b = 15)
  (h2 : c = 5)
  (h3 : 2 = Real.sqrt ((a + 2) * (15 + 3)) / (5 + 1)) :
  a = 6 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l1976_197612


namespace NUMINAMATH_CALUDE_smallest_difference_l1976_197626

def Digits : Finset ℕ := {2, 3, 4, 5, 6, 7, 8, 9}

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n ≤ 9999 ∧ (Finset.card (Finset.filter (λ d => d ∈ Digits) (Finset.image (λ i => (n / 10^i) % 10) {0,1,2,3})) = 4)

def valid_pair (a b : ℕ) : Prop :=
  is_valid_number a ∧ is_valid_number b ∧
  (Finset.card (Finset.filter (λ d => d ∈ Digits) (Finset.image (λ i => (a / 10^i) % 10) {0,1,2,3} ∪ Finset.image (λ i => (b / 10^i) % 10) {0,1,2,3})) = 8)

theorem smallest_difference :
  ∃ (a b : ℕ), valid_pair a b ∧
    (a > b) ∧
    (a - b = 247) ∧
    (∀ (c d : ℕ), valid_pair c d → c > d → c - d ≥ 247) :=
sorry

end NUMINAMATH_CALUDE_smallest_difference_l1976_197626


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l1976_197633

theorem quadratic_always_positive : ∀ x : ℝ, 15 * x^2 - 8 * x + 3 > 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l1976_197633


namespace NUMINAMATH_CALUDE_inequality_holds_for_even_positive_integers_l1976_197622

theorem inequality_holds_for_even_positive_integers (n : ℕ) (hn : Even n) (hn_pos : 0 < n) :
  ∀ x : ℝ, 3 * x^n + n * (x + 2) - 3 ≥ n * x^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_for_even_positive_integers_l1976_197622


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1976_197641

theorem min_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 9) :
  2/y + 1/x ≥ 1 ∧ (2/y + 1/x = 1 ↔ x = 3 ∧ y = 3) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1976_197641


namespace NUMINAMATH_CALUDE_runners_speed_l1976_197647

/-- The speeds of two runners on a circular track -/
theorem runners_speed (speed_a speed_b : ℝ) (track_length : ℝ) : 
  speed_a > 0 ∧ 
  speed_b > 0 ∧ 
  track_length > 0 ∧ 
  (speed_a + speed_b) * 48 = track_length ∧ 
  (speed_a - speed_b) * 600 = track_length ∧ 
  speed_a = speed_b + 2/3 → 
  speed_a = 9/2 ∧ speed_b = 23/6 := by sorry

end NUMINAMATH_CALUDE_runners_speed_l1976_197647


namespace NUMINAMATH_CALUDE_problem_statement_l1976_197616

open Real

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := log x - 2*a*x + 2*a

theorem problem_statement (a : ℝ) (h1 : 0 < a) (h2 : a ≤ 1/4) :
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < 2 ∧ 0 < x₂ ∧ x₂ < 2 ∧ x₁ ≠ x₂ →
    |g a x₁ - g a x₂| < 2*a*|1/x₁ - 1/x₂|) →
  a = 1/4 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1976_197616


namespace NUMINAMATH_CALUDE_fixed_points_range_l1976_197698

/-- The function f(x) = x^2 + ax + 4 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 4

/-- A fixed point of f is a real number x such that f(x) = x -/
def is_fixed_point (a : ℝ) (x : ℝ) : Prop := f a x = x

/-- The proposition that f has exactly two different fixed points in [1,3] -/
def has_two_fixed_points (a : ℝ) : Prop :=
  ∃ (x y : ℝ), x ≠ y ∧ x ∈ Set.Icc 1 3 ∧ y ∈ Set.Icc 1 3 ∧
  is_fixed_point a x ∧ is_fixed_point a y ∧
  ∀ (z : ℝ), z ∈ Set.Icc 1 3 → is_fixed_point a z → (z = x ∨ z = y)

/-- The main theorem stating the range of a -/
theorem fixed_points_range :
  ∀ a : ℝ, has_two_fixed_points a ↔ a ∈ Set.Icc (-10/3) (-3) :=
sorry

end NUMINAMATH_CALUDE_fixed_points_range_l1976_197698


namespace NUMINAMATH_CALUDE_log_sum_equals_two_l1976_197604

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_sum_equals_two : 2 * log 5 10 + log 5 0.25 = 2 := by sorry

end NUMINAMATH_CALUDE_log_sum_equals_two_l1976_197604


namespace NUMINAMATH_CALUDE_ratio_problem_l1976_197690

theorem ratio_problem (second_part : ℝ) (percent : ℝ) (first_part : ℝ) :
  second_part = 20 →
  percent = 25 →
  first_part / second_part = percent / 100 →
  first_part = 5 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l1976_197690


namespace NUMINAMATH_CALUDE_integral_equals_three_l1976_197663

-- Define the integrand
def f (x : ℝ) : ℝ := 2 - |1 - x|

-- State the theorem
theorem integral_equals_three : ∫ x in (0)..(2), f x = 3 := by
  sorry

end NUMINAMATH_CALUDE_integral_equals_three_l1976_197663


namespace NUMINAMATH_CALUDE_distribution_count_4_3_l1976_197648

/-- The number of ways to distribute n distinct objects into k distinct groups,
    where each group must contain at least one object -/
def distribution_count (n k : ℕ) : ℕ := sorry

/-- Theorem: The number of ways to distribute 4 distinct objects into 3 distinct groups,
    where each group must contain at least one object, is equal to 36 -/
theorem distribution_count_4_3 :
  distribution_count 4 3 = 36 := by sorry

end NUMINAMATH_CALUDE_distribution_count_4_3_l1976_197648


namespace NUMINAMATH_CALUDE_min_value_xyz_l1976_197696

theorem min_value_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 27) :
  x + 3 * y + 6 * z ≥ 27 ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 27 ∧ x₀ + 3 * y₀ + 6 * z₀ = 27 :=
by sorry

end NUMINAMATH_CALUDE_min_value_xyz_l1976_197696


namespace NUMINAMATH_CALUDE_function_defined_range_l1976_197676

open Real

theorem function_defined_range (a : ℝ) :
  (∀ x ∈ Set.Iic 1, (1 + 2^x + 4^x * a) / 3 > 0) ↔ a > -3/4 := by
  sorry

end NUMINAMATH_CALUDE_function_defined_range_l1976_197676


namespace NUMINAMATH_CALUDE_new_students_average_age_l1976_197656

theorem new_students_average_age
  (original_avg : ℝ)
  (original_size : ℕ)
  (new_size : ℕ)
  (avg_decrease : ℝ)
  (h1 : original_avg = 40)
  (h2 : original_size = 18)
  (h3 : new_size = 18)
  (h4 : avg_decrease = 4) :
  let new_avg := original_avg - avg_decrease
  let total_new_size := original_size + new_size
  let original_total_age := original_avg * original_size
  let new_total_age := new_avg * total_new_size
  let new_students_total_age := new_total_age - original_total_age
  new_students_total_age / new_size = 32 :=
by sorry

end NUMINAMATH_CALUDE_new_students_average_age_l1976_197656


namespace NUMINAMATH_CALUDE_zeros_of_g_l1976_197680

-- Define the power function f
def f : ℝ → ℝ := fun x => x^3

-- Define the function g
def g : ℝ → ℝ := fun x => f x - x

-- State the theorem
theorem zeros_of_g :
  (f 2 = 8) →
  (∀ x : ℝ, g x = 0 ↔ x = 0 ∨ x = 1 ∨ x = -1) :=
by sorry

end NUMINAMATH_CALUDE_zeros_of_g_l1976_197680


namespace NUMINAMATH_CALUDE_correct_outfit_count_l1976_197603

/-- The number of outfits that can be made with given clothing items, 
    where shirts and hats cannot be the same color. -/
def number_of_outfits (red_shirts green_shirts pants blue_hats red_hats scarves : ℕ) : ℕ :=
  (red_shirts * pants * blue_hats * scarves) + (green_shirts * pants * red_hats * scarves)

/-- Theorem stating the correct number of outfits given specific quantities of clothing items. -/
theorem correct_outfit_count : 
  number_of_outfits 7 8 10 10 10 5 = 7500 := by
  sorry

end NUMINAMATH_CALUDE_correct_outfit_count_l1976_197603


namespace NUMINAMATH_CALUDE_club_officer_selection_l1976_197670

theorem club_officer_selection (total_members : Nat) (boys : Nat) (girls : Nat)
  (h1 : total_members = boys + girls)
  (h2 : boys = 18)
  (h3 : girls = 12)
  (h4 : boys > 0)
  (h5 : girls > 0) :
  boys * girls = 216 := by
  sorry

end NUMINAMATH_CALUDE_club_officer_selection_l1976_197670


namespace NUMINAMATH_CALUDE_abs_neg_five_l1976_197644

theorem abs_neg_five : |(-5 : ℤ)| = 5 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_five_l1976_197644


namespace NUMINAMATH_CALUDE_olivia_cookies_l1976_197636

/-- The number of chocolate chip cookies Olivia has -/
def chocolate_chip_cookies (cookies_per_bag : ℕ) (oatmeal_cookies : ℕ) (baggies : ℕ) : ℕ :=
  cookies_per_bag * baggies - oatmeal_cookies

/-- Proof that Olivia has 13 chocolate chip cookies -/
theorem olivia_cookies : chocolate_chip_cookies 9 41 6 = 13 := by
  sorry

end NUMINAMATH_CALUDE_olivia_cookies_l1976_197636


namespace NUMINAMATH_CALUDE_inequality_proof_l1976_197645

theorem inequality_proof (a b c : ℝ) : 
  (a + b) * (a + b - 2 * c) + (b + c) * (b + c - 2 * a) + (c + a) * (c + a - 2 * b) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1976_197645


namespace NUMINAMATH_CALUDE_minimum_shoeing_time_l1976_197673

theorem minimum_shoeing_time 
  (num_blacksmiths : ℕ) 
  (num_horses : ℕ) 
  (time_per_horseshoe : ℕ) 
  (horseshoes_per_horse : ℕ) : 
  num_blacksmiths = 48 →
  num_horses = 60 →
  time_per_horseshoe = 5 →
  horseshoes_per_horse = 4 →
  (num_horses * horseshoes_per_horse * time_per_horseshoe) / num_blacksmiths = 25 :=
by sorry

end NUMINAMATH_CALUDE_minimum_shoeing_time_l1976_197673


namespace NUMINAMATH_CALUDE_function_range_l1976_197614

/-- Given a function f(x) = x^2 + ax + 3 - a, if f(x) ≥ 0 for all x in [-2, 2],
    then a is in the range [-7, 2]. -/
theorem function_range (a : ℝ) : 
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, x^2 + a*x + 3 - a ≥ 0) → 
  a ∈ Set.Icc (-7 : ℝ) 2 := by
  sorry

end NUMINAMATH_CALUDE_function_range_l1976_197614


namespace NUMINAMATH_CALUDE_power_and_multiplication_l1976_197635

theorem power_and_multiplication : 2 * (3^2)^4 = 13122 := by
  sorry

end NUMINAMATH_CALUDE_power_and_multiplication_l1976_197635


namespace NUMINAMATH_CALUDE_solution_implies_sum_equals_four_l1976_197692

-- Define the operation ⊗
def otimes (x y : ℝ) := x * (1 - y)

-- Define the theorem
theorem solution_implies_sum_equals_four 
  (h : ∀ x : ℝ, (otimes (x - a) (x - b) > 0) ↔ (2 < x ∧ x < 3)) :
  a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_solution_implies_sum_equals_four_l1976_197692


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_l1976_197630

theorem necessary_not_sufficient :
  (∀ x : ℝ, -1 ≤ x ∧ x < 2 → -1 ≤ x ∧ x < 3) ∧
  ¬(∀ x : ℝ, -1 ≤ x ∧ x < 3 → -1 ≤ x ∧ x < 2) :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_l1976_197630


namespace NUMINAMATH_CALUDE_equation_equals_24_l1976_197688

theorem equation_equals_24 :
  (2 + 2 / 11) * 11 = 24 := by
  sorry

end NUMINAMATH_CALUDE_equation_equals_24_l1976_197688


namespace NUMINAMATH_CALUDE_problem_statement_l1976_197661

def p (m : ℝ) : Prop := ∀ x ∈ Set.Icc 0 1, 2*x - 2 ≥ m^2 - 3*m

def q (m : ℝ) : Prop := ∃ x ∈ Set.Icc (-1) 1, m ≤ x

theorem problem_statement (m : ℝ) :
  (p m ↔ m ∈ Set.Icc 1 2) ∧
  (¬(p m ∧ q m) ∧ (p m ∨ q m) ↔ m < 1 ∨ (1 < m ∧ m ≤ 2)) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l1976_197661


namespace NUMINAMATH_CALUDE_unique_solution_system_l1976_197639

theorem unique_solution_system : ∃! (x y : ℕ+), 
  (x.val : ℝ) ^ (y.val : ℝ) + 1 = (y.val : ℝ) ^ (x.val : ℝ) ∧ 
  2 * (x.val : ℝ) ^ (y.val : ℝ) = (y.val : ℝ) ^ (x.val : ℝ) + 7 ∧
  x.val = 2 ∧ y.val = 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l1976_197639


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l1976_197634

theorem trigonometric_inequality : 
  let a := Real.sin (2 * Real.pi / 5)
  let b := Real.cos (5 * Real.pi / 6)
  let c := Real.tan (7 * Real.pi / 5)
  c > a ∧ a > b := by sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l1976_197634


namespace NUMINAMATH_CALUDE_vessel_base_length_from_cube_immersion_l1976_197657

/-- Calculates the length of a rectangular vessel's base given a cube immersion --/
theorem vessel_base_length_from_cube_immersion 
  (cube_edge : ℝ) 
  (vessel_width : ℝ) 
  (water_rise : ℝ) 
  (h1 : cube_edge = 10) 
  (h2 : vessel_width = 15) 
  (h3 : water_rise = 3.3333333333333335) : 
  (cube_edge^3 / (water_rise * vessel_width)) = 20 := by
sorry


end NUMINAMATH_CALUDE_vessel_base_length_from_cube_immersion_l1976_197657


namespace NUMINAMATH_CALUDE_parabola_comparison_l1976_197623

theorem parabola_comparison :
  ∀ x : ℝ, -x^2 + 2*x + 3 > x^2 - 2*x + 3 :=
by sorry

end NUMINAMATH_CALUDE_parabola_comparison_l1976_197623


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1976_197621

def i : ℂ := Complex.I

theorem imaginary_part_of_z (z : ℂ) : z = (2 + i) / i → z.im = -2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1976_197621


namespace NUMINAMATH_CALUDE_fish_population_estimate_l1976_197615

theorem fish_population_estimate (initial_marked : ℕ) (second_catch : ℕ) (marked_in_second : ℕ) 
  (h1 : initial_marked = 100)
  (h2 : second_catch = 200)
  (h3 : marked_in_second = 25) :
  (initial_marked * second_catch) / marked_in_second = 800 :=
by sorry

end NUMINAMATH_CALUDE_fish_population_estimate_l1976_197615


namespace NUMINAMATH_CALUDE_coin_flip_probability_l1976_197650

-- Define the coin values
def penny_value : ℕ := 1
def nickel_value : ℕ := 5
def dime_value : ℕ := 10
def quarter_value : ℕ := 25
def half_dollar_value : ℕ := 50

-- Define the total number of possible outcomes
def total_outcomes : ℕ := 2^5

-- Define the function to calculate successful outcomes
def successful_outcomes : ℕ := 18

-- Define the target value
def target_value : ℕ := 40

-- Theorem statement
theorem coin_flip_probability :
  (successful_outcomes : ℚ) / total_outcomes = 9 / 16 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l1976_197650


namespace NUMINAMATH_CALUDE_catman_do_whisker_count_l1976_197653

/-- The number of whiskers Catman Do has -/
def catman_do_whiskers (princess_puff_whiskers : ℕ) : ℕ :=
  2 * princess_puff_whiskers - 6

/-- Theorem stating the number of whiskers Catman Do has -/
theorem catman_do_whisker_count :
  catman_do_whiskers 14 = 22 := by
  sorry

end NUMINAMATH_CALUDE_catman_do_whisker_count_l1976_197653


namespace NUMINAMATH_CALUDE_dance_step_ratio_l1976_197693

theorem dance_step_ratio : 
  ∀ (N J : ℕ),
  (∃ (k : ℕ), N = k * J) →  -- Nancy steps k times as often as Jason
  N + J = 32 →              -- Total steps
  J = 8 →                   -- Jason's steps
  N / J = 3 :=              -- Ratio of Nancy's to Jason's steps
by sorry

end NUMINAMATH_CALUDE_dance_step_ratio_l1976_197693


namespace NUMINAMATH_CALUDE_candy_spent_is_10_l1976_197683

-- Define the total amount spent
def total_spent : ℝ := 150

-- Define the fractions spent on each category
def fruits_veg_fraction : ℚ := 1/2
def meat_fraction : ℚ := 1/3
def bakery_fraction : ℚ := 1/10

-- Define the theorem
theorem candy_spent_is_10 :
  let remaining_fraction : ℚ := 1 - (fruits_veg_fraction + meat_fraction + bakery_fraction)
  (remaining_fraction : ℝ) * total_spent = 10 := by sorry

end NUMINAMATH_CALUDE_candy_spent_is_10_l1976_197683


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l1976_197600

theorem geometric_sequence_third_term
  (a : ℕ → ℕ)  -- The sequence
  (h1 : a 1 = 5)  -- First term is 5
  (h2 : a 4 = 320)  -- Fourth term is 320
  (h_geom : ∀ n : ℕ, n > 0 → a (n + 1) = a n * (a 2 / a 1))  -- Geometric sequence property
  : a 3 = 80 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l1976_197600


namespace NUMINAMATH_CALUDE_point_location_l1976_197629

theorem point_location (x y : ℝ) : 
  (4 * x + 7 * y = 28) →  -- Line equation
  (abs x = abs y) →       -- Equidistant from axes
  (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) :=  -- In quadrant I or II
by sorry

end NUMINAMATH_CALUDE_point_location_l1976_197629


namespace NUMINAMATH_CALUDE_barry_average_l1976_197618

def barry_yards : List ℕ := [98, 107, 85, 89, 91]

theorem barry_average : 
  (barry_yards.sum / barry_yards.length : ℚ) = 94 := by sorry

end NUMINAMATH_CALUDE_barry_average_l1976_197618


namespace NUMINAMATH_CALUDE_intersection_on_line_x_eq_4_l1976_197659

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 3) = 1

-- Define the line l
def line_l (m : ℝ) (x y : ℝ) : Prop := x = m * y + 1

-- Define points A and B
def point_A : ℝ × ℝ := (-2, 0)
def point_B : ℝ × ℝ := (2, 0)

-- Define the intersection points M and N
def intersection_points (m : ℝ) : Set (ℝ × ℝ) :=
  {p | ellipse p.1 p.2 ∧ line_l m p.1 p.2}

-- Define the line AM
def line_AM (m : ℝ) (x y : ℝ) : Prop :=
  ∃ (M : ℝ × ℝ), M ∈ intersection_points m ∧
  (y - point_A.2) * (M.1 - point_A.1) = (x - point_A.1) * (M.2 - point_A.2)

-- Define the line BN
def line_BN (m : ℝ) (x y : ℝ) : Prop :=
  ∃ (N : ℝ × ℝ), N ∈ intersection_points m ∧
  (y - point_B.2) * (N.1 - point_B.1) = (x - point_B.1) * (N.2 - point_B.2)

-- Theorem statement
theorem intersection_on_line_x_eq_4 (m : ℝ) :
  ∃ (x y : ℝ), line_AM m x y ∧ line_BN m x y → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_on_line_x_eq_4_l1976_197659


namespace NUMINAMATH_CALUDE_train_delay_l1976_197668

theorem train_delay (car_time train_time : ℝ) : 
  car_time = 4.5 → 
  car_time + train_time = 11 → 
  train_time - car_time = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_train_delay_l1976_197668


namespace NUMINAMATH_CALUDE_gcd_420_135_l1976_197675

theorem gcd_420_135 : Nat.gcd 420 135 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_420_135_l1976_197675


namespace NUMINAMATH_CALUDE_two_tangents_iff_m_gt_two_l1976_197624

/-- The circle equation: x^2 + y^2 + mx + 1 = 0 -/
def circle_equation (x y m : ℝ) : Prop := x^2 + y^2 + m*x + 1 = 0

/-- The point A -/
def point_A : ℝ × ℝ := (1, 0)

/-- Condition for two tangents to be drawn from a point to a circle -/
def two_tangents_condition (m : ℝ) : Prop :=
  let center := (-m/2, 0)
  let radius_squared := m^2/4 - 1
  let distance_squared := (point_A.1 - center.1)^2 + (point_A.2 - center.2)^2
  distance_squared > radius_squared ∧ radius_squared > 0

theorem two_tangents_iff_m_gt_two :
  ∀ m : ℝ, two_tangents_condition m ↔ m > 2 :=
sorry

end NUMINAMATH_CALUDE_two_tangents_iff_m_gt_two_l1976_197624


namespace NUMINAMATH_CALUDE_complement_of_A_l1976_197609

-- Define the universal set U
def U : Set ℝ := {x | 0 ≤ x ∧ x ≤ 1}

-- Define the set A
def A : Set ℝ := {x | 0 < x ∧ x < 1/3}

-- Define the complement of A with respect to U
def complementU (A : Set ℝ) : Set ℝ := {x ∈ U | x ∉ A}

-- Theorem statement
theorem complement_of_A : complementU A = {x | x = 0 ∨ 1/3 ≤ x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_l1976_197609


namespace NUMINAMATH_CALUDE_equation_solution_l1976_197628

theorem equation_solution : ∃! x : ℝ, (x^2 - 6*x + 8)/(x^2 - 7*x + 12) = (x^2 - 3*x - 10)/(x^2 + x - 12) ∧ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1976_197628


namespace NUMINAMATH_CALUDE_sin_2theta_value_l1976_197602

theorem sin_2theta_value (θ : Real) (h : Real.sin (θ + π/4) = 1/3) : 
  Real.sin (2*θ) = -7/9 := by
  sorry

end NUMINAMATH_CALUDE_sin_2theta_value_l1976_197602


namespace NUMINAMATH_CALUDE_sum_components_eq_46_l1976_197643

/-- Represents a trapezoid with four sides --/
structure Trapezoid :=
  (side1 : ℝ) (side2 : ℝ) (side3 : ℝ) (side4 : ℝ)

/-- Represents the sum of areas in the form r₄√n₄ + r₅√n₅ + r₆ --/
structure AreaSum :=
  (r₄ : ℚ) (r₅ : ℚ) (r₆ : ℚ) (n₄ : ℕ) (n₅ : ℕ)

/-- Function to calculate the sum of all possible areas of a trapezoid --/
def sumAreas (t : Trapezoid) : AreaSum :=
  sorry

/-- Theorem stating that the sum of components equals 46 for the given trapezoid --/
theorem sum_components_eq_46 (t : Trapezoid) (a : AreaSum) :
  t.side1 = 4 ∧ t.side2 = 6 ∧ t.side3 = 8 ∧ t.side4 = 10 ∧
  a = sumAreas t →
  a.r₄ + a.r₅ + a.r₆ + a.n₄ + a.n₅ = 46 :=
sorry

end NUMINAMATH_CALUDE_sum_components_eq_46_l1976_197643


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l1976_197666

theorem system_of_equations_solution :
  ∃! (x y : ℝ), (2 * x - y = 6) ∧ (x + 2 * y = -2) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l1976_197666
