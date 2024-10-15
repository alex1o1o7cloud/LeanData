import Mathlib

namespace NUMINAMATH_CALUDE_max_milk_bags_theorem_l1454_145434

/-- Calculates the maximum number of bags of milk that can be purchased given the cost per bag, 
    the promotion rule, and the total available money. -/
def max_milk_bags (cost_per_bag : ℚ) (promotion_rule : ℕ → ℕ) (total_money : ℚ) : ℕ :=
  sorry

/-- The promotion rule: for every 2 bags purchased, 1 additional bag is given for free -/
def buy_two_get_one_free (n : ℕ) : ℕ :=
  n + n / 2

theorem max_milk_bags_theorem :
  max_milk_bags 2.5 buy_two_get_one_free 30 = 18 := by
  sorry

end NUMINAMATH_CALUDE_max_milk_bags_theorem_l1454_145434


namespace NUMINAMATH_CALUDE_six_meter_logs_more_efficient_l1454_145483

/-- Represents the number of pieces obtained from a log of given length -/
def pieces_from_log (log_length : ℕ) : ℕ := log_length

/-- Represents the number of cuts needed to divide a log into 1-meter pieces -/
def cuts_for_log (log_length : ℕ) : ℕ := log_length - 1

/-- Represents the efficiency of cutting a log, measured as pieces per cut -/
def cutting_efficiency (log_length : ℕ) : ℚ :=
  (pieces_from_log log_length : ℚ) / (cuts_for_log log_length : ℚ)

/-- Theorem stating that 6-meter logs are more efficient to cut than 8-meter logs -/
theorem six_meter_logs_more_efficient :
  cutting_efficiency 6 > cutting_efficiency 8 := by
  sorry

end NUMINAMATH_CALUDE_six_meter_logs_more_efficient_l1454_145483


namespace NUMINAMATH_CALUDE_pure_imaginary_m_equals_four_l1454_145447

/-- A complex number z is pure imaginary if its real part is zero and its imaginary part is non-zero. -/
def IsPureImaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- The complex number formed by the given expression. -/
def ComplexExpression (m : ℝ) : ℂ :=
  ⟨m^2 - 3*m - 4, m^2 - 5*m - 6⟩

theorem pure_imaginary_m_equals_four :
  ∃ m : ℝ, IsPureImaginary (ComplexExpression m) → m = 4 :=
sorry

end NUMINAMATH_CALUDE_pure_imaginary_m_equals_four_l1454_145447


namespace NUMINAMATH_CALUDE_linear_equation_condition_l1454_145482

theorem linear_equation_condition (a : ℝ) : 
  (∀ x, ∃ m k, (a - 2) * x^(|a| - 1) + 6 = m * x + k) ∧ (a - 2 ≠ 0) ↔ a = -2 :=
sorry

end NUMINAMATH_CALUDE_linear_equation_condition_l1454_145482


namespace NUMINAMATH_CALUDE_f_properties_l1454_145439

noncomputable def f : ℝ → ℝ := fun x =>
  if x < 0 then x + 2
  else if x = 0 then 0
  else x - 2

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧
  (∀ x, f x < 2 ↔ x < 4) := by sorry

end NUMINAMATH_CALUDE_f_properties_l1454_145439


namespace NUMINAMATH_CALUDE_dataset_mode_l1454_145424

def dataset : List ℕ := [24, 23, 24, 25, 22]

def mode (l : List ℕ) : ℕ :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem dataset_mode : mode dataset = 24 := by
  sorry

end NUMINAMATH_CALUDE_dataset_mode_l1454_145424


namespace NUMINAMATH_CALUDE_det_transformation_l1454_145435

/-- Given a 2x2 matrix with determinant 7, prove that the determinant of a related matrix is also 7 -/
theorem det_transformation (p q r s : ℝ) (h : Matrix.det !![p, q; r, s] = 7) :
  Matrix.det !![p + 2*r, q + 2*s; r, s] = 7 := by
  sorry

end NUMINAMATH_CALUDE_det_transformation_l1454_145435


namespace NUMINAMATH_CALUDE_unique_m_value_l1454_145433

def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}

def B (m : ℝ) : Set ℝ := {x | x^2 - m*x + m - 1 = 0}

theorem unique_m_value : ∃! m : ℝ, A ∪ B m = A := by sorry

end NUMINAMATH_CALUDE_unique_m_value_l1454_145433


namespace NUMINAMATH_CALUDE_exponent_division_l1454_145456

theorem exponent_division (x : ℝ) (h : x ≠ 0) : x^3 / x^2 = x := by sorry

end NUMINAMATH_CALUDE_exponent_division_l1454_145456


namespace NUMINAMATH_CALUDE_initial_weasels_count_l1454_145484

/-- Represents the number of weasels caught by one fox in one week -/
def weasels_per_fox_per_week : ℕ := 4

/-- Represents the number of rabbits caught by one fox in one week -/
def rabbits_per_fox_per_week : ℕ := 2

/-- Represents the number of foxes -/
def num_foxes : ℕ := 3

/-- Represents the number of weeks the foxes hunt -/
def num_weeks : ℕ := 3

/-- Represents the initial number of rabbits -/
def initial_rabbits : ℕ := 50

/-- Represents the number of rabbits and weasels left after hunting -/
def remaining_animals : ℕ := 96

/-- Theorem stating that the initial number of weasels is 100 -/
theorem initial_weasels_count : 
  ∃ (initial_weasels : ℕ), 
    initial_weasels = 100 ∧
    initial_weasels + initial_rabbits = 
      remaining_animals + 
      (weasels_per_fox_per_week * num_foxes * num_weeks) + 
      (rabbits_per_fox_per_week * num_foxes * num_weeks) := by
  sorry

end NUMINAMATH_CALUDE_initial_weasels_count_l1454_145484


namespace NUMINAMATH_CALUDE_sodas_consumed_l1454_145403

def potluck_sodas (brought : ℕ) (taken_back : ℕ) : ℕ :=
  brought - taken_back

theorem sodas_consumed (brought : ℕ) (taken_back : ℕ) 
  (h : brought ≥ taken_back) : 
  potluck_sodas brought taken_back = brought - taken_back :=
by sorry

end NUMINAMATH_CALUDE_sodas_consumed_l1454_145403


namespace NUMINAMATH_CALUDE_area_of_inscribed_rectangle_l1454_145469

/-- The area of a rectangle inscribed in a square, given other inscribed shapes -/
theorem area_of_inscribed_rectangle (s : ℝ) (r1_length r1_width : ℝ) (sq_side : ℝ) :
  s = 4 →
  r1_length = 2 →
  r1_width = 4 →
  sq_side = 1 →
  ∃ (r2_length r2_width : ℝ),
    r2_length * r2_width = s^2 - (r1_length * r1_width + sq_side^2) :=
by sorry

end NUMINAMATH_CALUDE_area_of_inscribed_rectangle_l1454_145469


namespace NUMINAMATH_CALUDE_sin_cos_sum_equals_half_l1454_145436

theorem sin_cos_sum_equals_half : 
  Real.sin (17 * π / 180) * Real.cos (13 * π / 180) + 
  Real.sin (73 * π / 180) * Real.sin (167 * π / 180) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equals_half_l1454_145436


namespace NUMINAMATH_CALUDE_unique_function_satisfying_conditions_l1454_145405

theorem unique_function_satisfying_conditions
  (f : ℕ → ℕ → ℕ)
  (h1 : ∀ a b c : ℕ, f (Nat.gcd a b) c = Nat.gcd a (f c b))
  (h2 : ∀ a : ℕ, f a a ≥ a) :
  ∀ a b : ℕ, f a b = Nat.gcd a b :=
by sorry

end NUMINAMATH_CALUDE_unique_function_satisfying_conditions_l1454_145405


namespace NUMINAMATH_CALUDE_david_average_marks_l1454_145419

def david_marks : List ℕ := [76, 65, 82, 67, 85]

theorem david_average_marks :
  (david_marks.sum / david_marks.length : ℚ) = 75 := by sorry

end NUMINAMATH_CALUDE_david_average_marks_l1454_145419


namespace NUMINAMATH_CALUDE_bolt_defect_probability_l1454_145487

theorem bolt_defect_probability :
  let machine1_production : ℝ := 0.30
  let machine2_production : ℝ := 0.25
  let machine3_production : ℝ := 0.45
  let machine1_defect_rate : ℝ := 0.02
  let machine2_defect_rate : ℝ := 0.01
  let machine3_defect_rate : ℝ := 0.03
  machine1_production + machine2_production + machine3_production = 1 →
  machine1_production * machine1_defect_rate +
  machine2_production * machine2_defect_rate +
  machine3_production * machine3_defect_rate = 0.022 := by
sorry

end NUMINAMATH_CALUDE_bolt_defect_probability_l1454_145487


namespace NUMINAMATH_CALUDE_f_equals_g_l1454_145473

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 - 1
def g (t : ℝ) : ℝ := t^2 - 1

-- Theorem stating that f and g are the same function
theorem f_equals_g : ∀ x : ℝ, f x = g x := by sorry

end NUMINAMATH_CALUDE_f_equals_g_l1454_145473


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_multiplier_l1454_145411

theorem consecutive_odd_integers_multiplier (x : ℤ) (m : ℚ) : 
  x + 4 = 15 →  -- Third integer is 15
  (∀ k : ℤ, x + 2*k ∈ {n : ℤ | n % 2 = 1}) →  -- All three are odd integers
  x * m = 2 * (x + 4) + 3 →  -- First integer times m equals 3 more than twice the third
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_multiplier_l1454_145411


namespace NUMINAMATH_CALUDE_gcf_60_75_l1454_145478

theorem gcf_60_75 : Nat.gcd 60 75 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcf_60_75_l1454_145478


namespace NUMINAMATH_CALUDE_always_possible_scatter_plot_l1454_145499

/-- Represents statistical data for two variables -/
structure TwoVariableData where
  -- We don't need to specify the internal structure of the data
  -- as the problem doesn't provide details about it

/-- Represents a scatter plot -/
structure ScatterPlot where
  -- We don't need to specify the internal structure of the scatter plot
  -- as the problem doesn't provide details about it

/-- States that it's always possible to create a scatter plot from two-variable data -/
theorem always_possible_scatter_plot (data : TwoVariableData) : 
  ∃ (plot : ScatterPlot), true :=
sorry

end NUMINAMATH_CALUDE_always_possible_scatter_plot_l1454_145499


namespace NUMINAMATH_CALUDE_gold_ratio_l1454_145443

theorem gold_ratio (total_gold : ℕ) (greg_gold : ℕ) (h1 : total_gold = 100) (h2 : greg_gold = 20) :
  greg_gold / (total_gold - greg_gold) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_gold_ratio_l1454_145443


namespace NUMINAMATH_CALUDE_characterization_of_special_numbers_l1454_145491

-- Define a structure for numbers of the form p^n - 1
structure PrimeExponentMinusOne where
  p : Nat
  n : Nat
  isPrime : Nat.Prime p

-- Define a predicate for numbers whose all divisors are of the form p^n - 1
def allDivisorsArePrimeExponentMinusOne (m : Nat) : Prop :=
  ∀ d : Nat, d ∣ m → ∃ (p n : Nat), Nat.Prime p ∧ d = p^n - 1

-- Main theorem
theorem characterization_of_special_numbers (m : Nat) 
  (h1 : ∃ (p n : Nat), Nat.Prime p ∧ m = p^n - 1)
  (h2 : allDivisorsArePrimeExponentMinusOne m) :
  (∃ k : Nat, m = 2^k - 1 ∧ Nat.Prime m) ∨ m ∣ 48 :=
sorry

end NUMINAMATH_CALUDE_characterization_of_special_numbers_l1454_145491


namespace NUMINAMATH_CALUDE_odd_function_property_l1454_145471

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def has_period_4 (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 4) = f x

theorem odd_function_property (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_recursive : ∀ x, f (x + 4) = f x + 3 * f 2)
  (h_f_1 : f 1 = 1) :
  f 2015 + f 2016 = -1 := by
sorry

end NUMINAMATH_CALUDE_odd_function_property_l1454_145471


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1454_145448

theorem inequality_system_solution (x : ℝ) : 
  (1 / x < 1 ∧ |4 * x - 1| > 2) ↔ (x < -1/4 ∨ x > 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1454_145448


namespace NUMINAMATH_CALUDE_no_square_free_arithmetic_sequence_l1454_145472

theorem no_square_free_arithmetic_sequence :
  ∀ (a d : ℕ+), d ≠ 1 →
  ∃ (n : ℕ), ∃ (k : ℕ), k > 1 ∧ (k * k ∣ (a + n * d)) :=
by sorry

end NUMINAMATH_CALUDE_no_square_free_arithmetic_sequence_l1454_145472


namespace NUMINAMATH_CALUDE_q_divisibility_q_values_q_cubic_form_q_10_expression_l1454_145492

/-- A cubic polynomial q(x) such that [q(x)]^2 - x is divisible by (x - 2)(x + 2)(x - 5)(x - 7) -/
def q (x : ℝ) : ℝ := sorry

theorem q_divisibility (x : ℝ) : 
  ∃ k : ℝ, q x ^ 2 - x = k * ((x - 2) * (x + 2) * (x - 5) * (x - 7)) := sorry

theorem q_values : 
  q 2 = Real.sqrt 2 ∧ q (-2) = -Real.sqrt 2 ∧ q 5 = Real.sqrt 5 ∧ q 7 = Real.sqrt 7 := sorry

theorem q_cubic_form : 
  ∃ a b c d : ℝ, ∀ x : ℝ, q x = a * x^3 + b * x^2 + c * x + d := sorry

theorem q_10_expression (a b c d : ℝ) 
  (h : ∀ x : ℝ, q x = a * x^3 + b * x^2 + c * x + d) : 
  q 10 = 1000 * a + 100 * b + 10 * c + d := sorry

end NUMINAMATH_CALUDE_q_divisibility_q_values_q_cubic_form_q_10_expression_l1454_145492


namespace NUMINAMATH_CALUDE_expression_bounds_l1454_145495

theorem expression_bounds (x y z w : ℝ) 
  (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) (hw : 0 ≤ w ∧ w ≤ 1) :
  2 * Real.sqrt 2 ≤ Real.sqrt (x^2 + (1 - y)^2) + Real.sqrt (y^2 + (1 - z)^2) + 
    Real.sqrt (z^2 + (1 - w)^2) + Real.sqrt (w^2 + (1 - x)^2) ∧
  Real.sqrt (x^2 + (1 - y)^2) + Real.sqrt (y^2 + (1 - z)^2) + 
    Real.sqrt (z^2 + (1 - w)^2) + Real.sqrt (w^2 + (1 - x)^2) ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_expression_bounds_l1454_145495


namespace NUMINAMATH_CALUDE_triangle_side_range_l1454_145437

theorem triangle_side_range (p : ℝ) : 
  (∃ r s : ℝ, r * s = 4 * 26 ∧ 
              r^2 + p*r + 1 = 0 ∧ 
              s^2 + p*s + 1 = 0 ∧ 
              r > 0 ∧ s > 0 ∧
              r + s > 2 ∧ r + 2 > s ∧ s + 2 > r) →
  -2 * Real.sqrt 2 < p ∧ p < -2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_range_l1454_145437


namespace NUMINAMATH_CALUDE_radical_product_simplification_l1454_145415

theorem radical_product_simplification (q : ℝ) (hq : q ≥ 0) :
  Real.sqrt (50 * q) * Real.sqrt (10 * q) * Real.sqrt (15 * q) = 50 * q * Real.sqrt q :=
by sorry

end NUMINAMATH_CALUDE_radical_product_simplification_l1454_145415


namespace NUMINAMATH_CALUDE_parallel_vectors_t_value_l1454_145445

theorem parallel_vectors_t_value (t : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![-2, t]
  (∃ (k : ℝ), k ≠ 0 ∧ (∀ i, b i = k * a i)) → t = -4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_t_value_l1454_145445


namespace NUMINAMATH_CALUDE_jesse_carpet_problem_l1454_145427

/-- Given a room with length and width, and some carpet already available,
    calculate the additional carpet needed to cover the whole floor. -/
def additional_carpet_needed (length width available_carpet : ℝ) : ℝ :=
  length * width - available_carpet

/-- Theorem: Given a room that is 4 feet long and 20 feet wide, with 18 square feet
    of carpet already available, the additional carpet needed is 62 square feet. -/
theorem jesse_carpet_problem :
  additional_carpet_needed 4 20 18 = 62 := by
  sorry

end NUMINAMATH_CALUDE_jesse_carpet_problem_l1454_145427


namespace NUMINAMATH_CALUDE_min_value_of_fraction_l1454_145426

theorem min_value_of_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x^2 + y^2) / (x + y)^2 ≥ (1 : ℝ) / 2 ∧
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (a^2 + b^2) / (a + b)^2 = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_fraction_l1454_145426


namespace NUMINAMATH_CALUDE_count_numbers_divisible_by_291_l1454_145432

theorem count_numbers_divisible_by_291 :
  let max_k : ℕ := 291000
  let is_valid : ℕ → Prop := λ k => k ≤ max_k ∧ (k^2 - 1) % 291 = 0
  (Finset.filter is_valid (Finset.range (max_k + 1))).card = 4000 := by
  sorry

end NUMINAMATH_CALUDE_count_numbers_divisible_by_291_l1454_145432


namespace NUMINAMATH_CALUDE_number_representation_and_addition_l1454_145428

theorem number_representation_and_addition :
  (4090000 = 409 * 10000) ∧ (800000 + 5000 + 20 + 4 = 805024) := by
  sorry

end NUMINAMATH_CALUDE_number_representation_and_addition_l1454_145428


namespace NUMINAMATH_CALUDE_work_completion_men_difference_l1454_145444

theorem work_completion_men_difference (work : ℕ) : 
  ∀ (m n : ℕ), 
    m = 20 → 
    m * 10 = n * 20 → 
    m - n = 10 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_men_difference_l1454_145444


namespace NUMINAMATH_CALUDE_line_equation_proof_l1454_145474

theorem line_equation_proof (x y : ℝ) :
  let point_A : ℝ × ℝ := (1, 3)
  let slope_reference : ℝ := -4
  let slope_line : ℝ := slope_reference / 3
  (4 * x + 3 * y - 13 = 0) ↔
    (y - point_A.2 = slope_line * (x - point_A.1) ∧
     slope_line = slope_reference / 3) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_proof_l1454_145474


namespace NUMINAMATH_CALUDE_equation_solution_l1454_145422

theorem equation_solution : ∃ x : ℝ, (2 / (x - 4) + 3 = (x - 2) / (4 - x)) ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1454_145422


namespace NUMINAMATH_CALUDE_ellipse_focal_distance_difference_l1454_145470

/-- Given an ellipse with semi-major axis a and semi-minor axis b = √96,
    and a point P on the ellipse such that |PF₁| : |PF₂| : |OF₂| = 8 : 6 : 5,
    prove that |PF₁| - |PF₂| = 4 -/
theorem ellipse_focal_distance_difference 
  (a : ℝ) 
  (h_a : a > 4 * Real.sqrt 6) 
  (P : ℝ × ℝ) 
  (h_P : (P.1 / a)^2 + P.2^2 / 96 = 1) 
  (F₁ F₂ : ℝ × ℝ) 
  (h_foci : ∃ (k : ℝ), k > 0 ∧ dist P F₁ = 8*k ∧ dist P F₂ = 6*k ∧ dist (0, 0) F₂ = 5*k) :
  dist P F₁ - dist P F₂ = 4 := by
  sorry


end NUMINAMATH_CALUDE_ellipse_focal_distance_difference_l1454_145470


namespace NUMINAMATH_CALUDE_star_four_five_l1454_145460

def star (a b : ℤ) : ℤ := (a + 2*b) * (a - 2*b)

theorem star_four_five : star 4 5 = -84 := by
  sorry

end NUMINAMATH_CALUDE_star_four_five_l1454_145460


namespace NUMINAMATH_CALUDE_negation_of_existence_equals_forall_not_equal_l1454_145429

theorem negation_of_existence_equals_forall_not_equal (x : ℝ) :
  ¬(∃ x₀ : ℝ, x₀ > 0 ∧ Real.log x₀ = x₀ - 1) ↔ ∀ x : ℝ, x > 0 → Real.log x ≠ x - 1 :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_equals_forall_not_equal_l1454_145429


namespace NUMINAMATH_CALUDE_emily_meal_combinations_l1454_145485

/-- The number of protein options available --/
def num_proteins : ℕ := 4

/-- The number of side options available --/
def num_sides : ℕ := 5

/-- The number of dessert options available --/
def num_desserts : ℕ := 5

/-- The number of sides Emily must choose --/
def sides_to_choose : ℕ := 3

/-- The total number of different meal combinations Emily can choose --/
def total_meals : ℕ := num_proteins * (num_sides.choose sides_to_choose) * num_desserts

theorem emily_meal_combinations :
  total_meals = 200 :=
sorry

end NUMINAMATH_CALUDE_emily_meal_combinations_l1454_145485


namespace NUMINAMATH_CALUDE_custom_operation_result_l1454_145409

/-- Custom dollar operation -/
def dollar (a b c : ℝ) : ℝ := (a - b - c)^2

/-- Main theorem -/
theorem custom_operation_result (x y z : ℝ) :
  dollar ((x - z)^2) ((y - x)^2) ((y - z)^2) = (-2*x*z + z^2 + 2*y*x - 2*y*z)^2 := by
  sorry

end NUMINAMATH_CALUDE_custom_operation_result_l1454_145409


namespace NUMINAMATH_CALUDE_two_tower_100_gt_3_three_tower_100_gt_three_tower_99_three_tower_100_gt_four_tower_99_l1454_145425

-- Define a function to represent the power tower
def powerTower (base : ℕ) (height : ℕ) : ℕ :=
  match height with
  | 0 => 1
  | n + 1 => base ^ (powerTower base n)

-- Theorem 1
theorem two_tower_100_gt_3 : powerTower 2 100 > 3 := by sorry

-- Theorem 2
theorem three_tower_100_gt_three_tower_99 : powerTower 3 100 > powerTower 3 99 := by sorry

-- Theorem 3
theorem three_tower_100_gt_four_tower_99 : powerTower 3 100 > powerTower 4 99 := by sorry

end NUMINAMATH_CALUDE_two_tower_100_gt_3_three_tower_100_gt_three_tower_99_three_tower_100_gt_four_tower_99_l1454_145425


namespace NUMINAMATH_CALUDE_table_length_is_77_l1454_145438

/-- The length of the rectangular table -/
def table_length : ℕ := 77

/-- The width of the rectangular table -/
def table_width : ℕ := 80

/-- The height of each sheet of paper -/
def sheet_height : ℕ := 5

/-- The width of each sheet of paper -/
def sheet_width : ℕ := 8

/-- The horizontal and vertical increment for each subsequent sheet -/
def increment : ℕ := 1

theorem table_length_is_77 :
  ∃ (n : ℕ), 
    table_length = sheet_height + n * increment ∧
    table_width = sheet_width + n * increment ∧
    table_width - table_length = sheet_width - sheet_height := by
  sorry

end NUMINAMATH_CALUDE_table_length_is_77_l1454_145438


namespace NUMINAMATH_CALUDE_jerry_speed_is_40_l1454_145475

-- Define the given conditions
def jerry_time : ℚ := 1/2  -- 30 minutes in hours
def beth_time : ℚ := 5/6   -- 50 minutes in hours
def beth_speed : ℚ := 30   -- miles per hour
def route_difference : ℚ := 5  -- miles

-- Theorem to prove
theorem jerry_speed_is_40 :
  let beth_distance : ℚ := beth_speed * beth_time
  let jerry_distance : ℚ := beth_distance - route_difference
  jerry_distance / jerry_time = 40 := by
  sorry


end NUMINAMATH_CALUDE_jerry_speed_is_40_l1454_145475


namespace NUMINAMATH_CALUDE_no_three_naturals_sum_power_of_three_l1454_145467

theorem no_three_naturals_sum_power_of_three :
  ¬ ∃ (a b c : ℕ), 
    (∃ k : ℕ, a + b = 3^k) ∧
    (∃ m : ℕ, b + c = 3^m) ∧
    (∃ n : ℕ, c + a = 3^n) :=
sorry

end NUMINAMATH_CALUDE_no_three_naturals_sum_power_of_three_l1454_145467


namespace NUMINAMATH_CALUDE_set_B_determination_l1454_145465

open Set

theorem set_B_determination (A B : Set ℕ) : 
  A = {1, 2} → 
  A ∩ B = {1} → 
  A ∪ B = {0, 1, 2} → 
  B = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_set_B_determination_l1454_145465


namespace NUMINAMATH_CALUDE_remainder_equality_l1454_145404

theorem remainder_equality (A A' D S S' s s' : ℕ) 
  (h1 : A > A')
  (h2 : S = A % D)
  (h3 : S' = A' % D)
  (h4 : s = (A + A') % D)
  (h5 : s' = (S + S') % D) :
  s = s' :=
sorry

end NUMINAMATH_CALUDE_remainder_equality_l1454_145404


namespace NUMINAMATH_CALUDE_last_locker_is_2046_l1454_145431

/-- Represents the state of a locker (open or closed) -/
inductive LockerState
  | Open
  | Closed

/-- Represents the corridor of lockers -/
def Corridor := Fin 2048 → LockerState

/-- Represents the student's locker opening strategy -/
def OpeningStrategy := Corridor → Nat → Nat

/-- The final locker opened by the student -/
def lastOpenedLocker (strategy : OpeningStrategy) : Nat :=
  2046

/-- The theorem stating that the last opened locker is 2046 -/
theorem last_locker_is_2046 (strategy : OpeningStrategy) :
  lastOpenedLocker strategy = 2046 := by
  sorry

#check last_locker_is_2046

end NUMINAMATH_CALUDE_last_locker_is_2046_l1454_145431


namespace NUMINAMATH_CALUDE_second_car_speed_l1454_145453

theorem second_car_speed 
  (highway_length : ℝ) 
  (first_car_speed : ℝ) 
  (meeting_time : ℝ) 
  (h1 : highway_length = 175) 
  (h2 : first_car_speed = 25) 
  (h3 : meeting_time = 2.5) : 
  ∃ second_car_speed : ℝ, 
    first_car_speed * meeting_time + second_car_speed * meeting_time = highway_length ∧ 
    second_car_speed = 45 := by
sorry

end NUMINAMATH_CALUDE_second_car_speed_l1454_145453


namespace NUMINAMATH_CALUDE_problem_solution_l1454_145400

theorem problem_solution (x y : ℤ) (hx : x = 3) (hy : y = 4) : 3 * x - 5 * y = -11 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1454_145400


namespace NUMINAMATH_CALUDE_same_solution_equations_l1454_145414

theorem same_solution_equations (b : ℚ) : 
  (∃ x : ℚ, 3 * x + 9 = 0 ∧ 2 * b * x - 15 = -5) → b = -5/3 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_equations_l1454_145414


namespace NUMINAMATH_CALUDE_base_conversion_problem_l1454_145455

theorem base_conversion_problem (a b : ℕ) : 
  (a < 10 ∧ b < 10) → -- Ensuring a and b are single digits
  (6 * 7^2 + 5 * 7 + 6 = 300 + 10 * a + b) → -- 656₇ = 3ab₁₀
  (a * b) / 15 = 1 := by
sorry

end NUMINAMATH_CALUDE_base_conversion_problem_l1454_145455


namespace NUMINAMATH_CALUDE_largest_expression_l1454_145442

theorem largest_expression : 
  let a := 3 + 1 + 0 + 5
  let b := 3 * 1 + 0 + 5
  let c := 3 + 1 * 0 + 5
  let d := 3 * 1 + 0 * 5
  let e := 3 * 1 + 0 * 5 * 3
  (a > b) ∧ (a > c) ∧ (a > d) ∧ (a > e) := by
  sorry

end NUMINAMATH_CALUDE_largest_expression_l1454_145442


namespace NUMINAMATH_CALUDE_gcd_triple_existence_l1454_145451

theorem gcd_triple_existence (S : Set ℕ+) 
  (h_infinite : Set.Infinite S)
  (h_distinct_gcd : ∃ (a b c d : ℕ+), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
    Nat.gcd a.val b.val ≠ Nat.gcd c.val d.val) :
  ∃ (x y z : ℕ+), x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ 
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    Nat.gcd x.val y.val = Nat.gcd y.val z.val ∧ 
    Nat.gcd y.val z.val ≠ Nat.gcd z.val x.val :=
sorry

end NUMINAMATH_CALUDE_gcd_triple_existence_l1454_145451


namespace NUMINAMATH_CALUDE_lagoon_island_male_alligators_l1454_145402

/-- Represents the population of alligators on Lagoon island -/
structure AlligatorPopulation where
  total : ℕ
  males : ℕ
  females : ℕ
  juvenileFemales : ℕ
  adultFemales : ℕ

/-- Conditions for the Lagoon island alligator population -/
def lagoonIslandConditions (pop : AlligatorPopulation) : Prop :=
  pop.males = pop.females ∧
  pop.females = pop.juvenileFemales + pop.adultFemales ∧
  pop.juvenileFemales = (2 * pop.females) / 5 ∧
  pop.adultFemales = 15

theorem lagoon_island_male_alligators (pop : AlligatorPopulation) 
  (h : lagoonIslandConditions pop) : 
  pop.males = pop.adultFemales / (3 : ℚ) / (10 : ℚ) := by
  sorry

#check lagoon_island_male_alligators

end NUMINAMATH_CALUDE_lagoon_island_male_alligators_l1454_145402


namespace NUMINAMATH_CALUDE_algebraic_simplification_l1454_145408

theorem algebraic_simplification (a b c : ℝ) :
  -32 * a^4 * b^5 * c / (-2 * a * b)^3 * (-3/4 * a * c) = -3 * a^2 * b^2 * c^2 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_simplification_l1454_145408


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1454_145418

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n, a n > 0) →
  a 4 * a 6 + 2 * a 5 * a 7 + a 6 * a 8 = 36 →
  a 5 + a 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1454_145418


namespace NUMINAMATH_CALUDE_expansion_theorem_l1454_145457

-- Define the sum of binomial coefficients
def sum_binomial_coeff (m : ℝ) (n : ℕ) : ℝ := 2^n

-- Define the coefficient of x in the expansion
def coeff_x (m : ℝ) (n : ℕ) : ℝ := (n.choose 2) * m^2

theorem expansion_theorem (m : ℝ) (n : ℕ) (h_m : m > 0) 
  (h_sum : sum_binomial_coeff m n = 256)
  (h_coeff : coeff_x m n = 112) :
  n = 8 ∧ m = 2 ∧ 
  (Nat.choose 8 4 * 2^4 - Nat.choose 8 2 * 2^2 : ℝ) = 1008 :=
sorry

end NUMINAMATH_CALUDE_expansion_theorem_l1454_145457


namespace NUMINAMATH_CALUDE_average_difference_l1454_145466

theorem average_difference (a b c : ℝ) 
  (h1 : (a + b) / 2 = 35) 
  (h2 : (b + c) / 2 = 80) : 
  c - a = 90 := by
sorry

end NUMINAMATH_CALUDE_average_difference_l1454_145466


namespace NUMINAMATH_CALUDE_rain_hours_calculation_l1454_145493

/-- Given a 9-hour period where it rained for 4 hours, prove that it did not rain for 5 hours. -/
theorem rain_hours_calculation (total_hours rain_hours : ℕ) 
  (h1 : total_hours = 9)
  (h2 : rain_hours = 4) : 
  total_hours - rain_hours = 5 := by
  sorry

end NUMINAMATH_CALUDE_rain_hours_calculation_l1454_145493


namespace NUMINAMATH_CALUDE_border_area_l1454_145423

/-- The area of the border around a rectangular painting -/
theorem border_area (height width border_width : ℕ) : 
  height = 12 → width = 16 → border_width = 3 →
  (height + 2 * border_width) * (width + 2 * border_width) - height * width = 204 := by
  sorry

end NUMINAMATH_CALUDE_border_area_l1454_145423


namespace NUMINAMATH_CALUDE_neither_necessary_nor_sufficient_l1454_145462

theorem neither_necessary_nor_sufficient :
  ¬(∀ x : ℝ, -1 < x ∧ x < 2 → |x - 2| < 1) ∧
  ¬(∀ x : ℝ, |x - 2| < 1 → -1 < x ∧ x < 2) :=
by sorry

end NUMINAMATH_CALUDE_neither_necessary_nor_sufficient_l1454_145462


namespace NUMINAMATH_CALUDE_exterior_angle_theorem_l1454_145497

/-- The measure of the exterior angle BAC in a coplanar arrangement 
    where a square and a regular nonagon share a common side AD -/
def exterior_angle_BAC : ℝ := 130

/-- The measure of the interior angle of a regular nonagon -/
def nonagon_interior_angle : ℝ := 140

/-- The measure of the interior angle of a square -/
def square_interior_angle : ℝ := 90

theorem exterior_angle_theorem :
  exterior_angle_BAC = 360 - nonagon_interior_angle - square_interior_angle :=
by sorry

end NUMINAMATH_CALUDE_exterior_angle_theorem_l1454_145497


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1454_145459

theorem sqrt_equation_solution :
  ∃ x : ℝ, x = 6 ∧ Real.sqrt (4 + 9 + x^2) = 7 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1454_145459


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l1454_145498

/-- Given a line with slope 4 passing through (5, -2), prove that m + b = -18 -/
theorem line_slope_intercept_sum (m b : ℝ) : 
  m = 4 ∧ 
  -2 = 4 * 5 + b → 
  m + b = -18 := by
sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l1454_145498


namespace NUMINAMATH_CALUDE_fourth_student_score_l1454_145481

theorem fourth_student_score (s1 s2 s3 s4 : ℕ) : 
  s1 = 70 → s2 = 80 → s3 = 90 → (s1 + s2 + s3 + s4) / 4 = 70 → s4 = 40 := by
  sorry

end NUMINAMATH_CALUDE_fourth_student_score_l1454_145481


namespace NUMINAMATH_CALUDE_even_times_odd_is_even_l1454_145490

/-- An integer is even if it's divisible by 2 -/
def IsEven (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

/-- An integer is odd if it's not divisible by 2 -/
def IsOdd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

/-- The product of an even integer and an odd integer is always even -/
theorem even_times_odd_is_even (a b : ℤ) (ha : IsEven a) (hb : IsOdd b) : IsEven (a * b) := by
  sorry


end NUMINAMATH_CALUDE_even_times_odd_is_even_l1454_145490


namespace NUMINAMATH_CALUDE_student_rank_problem_l1454_145420

/-- Given a total number of students and a student's rank from the right,
    calculates the student's rank from the left. -/
def rank_from_left (total : ℕ) (rank_from_right : ℕ) : ℕ :=
  total - rank_from_right + 1

/-- Proves that for 21 total students and a student ranked 16th from the right,
    the student's rank from the left is 6. -/
theorem student_rank_problem :
  rank_from_left 21 16 = 6 := by
  sorry

#eval rank_from_left 21 16

end NUMINAMATH_CALUDE_student_rank_problem_l1454_145420


namespace NUMINAMATH_CALUDE_usual_time_is_eight_l1454_145449

-- Define the usual speed and time
variable (S : ℝ) -- Usual speed
variable (T : ℝ) -- Usual time

-- Define the theorem
theorem usual_time_is_eight
  (h1 : S > 0) -- Assume speed is positive
  (h2 : T > 0) -- Assume time is positive
  (h3 : S / (0.25 * S) = (T + 24) / T) -- Equation from the problem
  : T = 8 := by
sorry


end NUMINAMATH_CALUDE_usual_time_is_eight_l1454_145449


namespace NUMINAMATH_CALUDE_ratio_to_twelve_l1454_145450

theorem ratio_to_twelve : ∃ x : ℝ, (5 : ℝ) / 1 = x / 12 → x = 60 :=
by sorry

end NUMINAMATH_CALUDE_ratio_to_twelve_l1454_145450


namespace NUMINAMATH_CALUDE_inverse_g_at_43_16_l1454_145454

/-- Given a function g(x) = (x^3 - 5) / 4, prove that g⁻¹(43/16) = 3 * ∛7 / 2 -/
theorem inverse_g_at_43_16 (g : ℝ → ℝ) (h : ∀ x, g x = (x^3 - 5) / 4) :
  g⁻¹ (43/16) = 3 * Real.rpow 7 (1/3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_g_at_43_16_l1454_145454


namespace NUMINAMATH_CALUDE_square_area_ratio_l1454_145476

theorem square_area_ratio (s₁ s₂ : ℝ) (h : s₁ = 2 * s₂ * Real.sqrt 2) :
  s₁^2 / s₂^2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l1454_145476


namespace NUMINAMATH_CALUDE_diplomats_speaking_french_l1454_145479

theorem diplomats_speaking_french (T : ℕ) (F R B : ℕ) : 
  T = 70 →
  R = 38 →
  B = 7 →
  (T - F - R + B : ℤ) = 14 →
  F = 25 :=
by sorry

end NUMINAMATH_CALUDE_diplomats_speaking_french_l1454_145479


namespace NUMINAMATH_CALUDE_quadratic_integer_roots_count_l1454_145496

theorem quadratic_integer_roots_count :
  let f (m : ℤ) := (∃ x₁ x₂ : ℤ, x₁ * x₂ = 30 ∧ x₁ + x₂ = m)
  (∃! s : Finset ℤ, (∀ m : ℤ, m ∈ s ↔ f m) ∧ s.card = 8) :=
sorry

end NUMINAMATH_CALUDE_quadratic_integer_roots_count_l1454_145496


namespace NUMINAMATH_CALUDE_remaining_eggs_l1454_145407

theorem remaining_eggs (initial_eggs : ℕ) (morning_eaten : ℕ) (afternoon_eaten : ℕ) 
  (h1 : initial_eggs = 20)
  (h2 : morning_eaten = 4)
  (h3 : afternoon_eaten = 3) :
  initial_eggs - (morning_eaten + afternoon_eaten) = 13 := by
  sorry

end NUMINAMATH_CALUDE_remaining_eggs_l1454_145407


namespace NUMINAMATH_CALUDE_number_of_boxes_l1454_145477

theorem number_of_boxes (eggs_per_box : ℕ) (total_eggs : ℕ) (h1 : eggs_per_box = 3) (h2 : total_eggs = 6) :
  total_eggs / eggs_per_box = 2 := by
  sorry

end NUMINAMATH_CALUDE_number_of_boxes_l1454_145477


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l1454_145486

theorem triangle_angle_measure (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  b^2 = a^2 - 2*b*c →
  A = 2*π/3 →
  C = π/6 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l1454_145486


namespace NUMINAMATH_CALUDE_divisor_problem_l1454_145401

theorem divisor_problem (x : ℕ) : x > 0 ∧ x ∣ 1058 ∧ ∀ y, 0 < y ∧ y < x → ¬(y ∣ 1058) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l1454_145401


namespace NUMINAMATH_CALUDE_seed_germination_problem_l1454_145416

theorem seed_germination_problem (x : ℝ) : 
  x > 0 ∧ 
  (0.25 * x + 0.35 * 200) / (x + 200) = 0.28999999999999996 → 
  x = 300 := by
  sorry

end NUMINAMATH_CALUDE_seed_germination_problem_l1454_145416


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1454_145494

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (d : ℝ)
  (h1 : d ≠ 0)
  (h2 : a 1 = 9)
  (h3 : ∀ n, a (n + 1) = a n + d)
  (h4 : (a 4) ^ 2 = (a 1) * (a 8)) :
  d = 1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1454_145494


namespace NUMINAMATH_CALUDE_inverse_function_point_and_sum_l1454_145480

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the inverse function of f
noncomputable def f_inv : ℝ → ℝ := sorry

-- State the theorem
theorem inverse_function_point_and_sum :
  (f 2 = 6) →  -- This condition is derived from (2,3) being on y = f(x)/2
  (f_inv 6 = 2) →  -- This is the definition of the inverse function
  (∃ (x y : ℝ), x = 6 ∧ y = 1 ∧ y = (f_inv x) / 2) ∧  -- Point (6,1) is on y = f^(-1)(x)/2
  (6 + 1 = 7)  -- Sum of coordinates
  := by sorry

end NUMINAMATH_CALUDE_inverse_function_point_and_sum_l1454_145480


namespace NUMINAMATH_CALUDE_waitress_income_fraction_from_tips_l1454_145412

/-- Represents the income structure of a waitress -/
structure WaitressIncome where
  salary : ℚ
  tips : ℚ

/-- Theorem: Given a waitress's income where tips are 9/4 of salary,
    the fraction of income from tips is 9/13 -/
theorem waitress_income_fraction_from_tips 
  (income : WaitressIncome) 
  (h : income.tips = (9 : ℚ) / 4 * income.salary) : 
  income.tips / (income.salary + income.tips) = (9 : ℚ) / 13 := by
  sorry


end NUMINAMATH_CALUDE_waitress_income_fraction_from_tips_l1454_145412


namespace NUMINAMATH_CALUDE_sum_of_i_powers_l1454_145468

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Define the property that i^2 = -1
axiom i_squared : i^2 = -1

-- Define the property that powers of i repeat every four powers
axiom i_period (n : ℤ) : i^n = i^(n % 4)

-- State the theorem
theorem sum_of_i_powers : i^23 + i^221 + i^20 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_i_powers_l1454_145468


namespace NUMINAMATH_CALUDE_number_exceeding_percentage_l1454_145458

theorem number_exceeding_percentage (x : ℝ) : x = 0.16 * x + 42 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeding_percentage_l1454_145458


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l1454_145406

theorem fraction_sum_equality (a b c : ℝ) 
  (h : a / (36 - a) + b / (49 - b) + c / (81 - c) = 9) :
  6 / (36 - a) + 7 / (49 - b) + 9 / (81 - c) = 5047 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l1454_145406


namespace NUMINAMATH_CALUDE_eighth_term_is_six_l1454_145446

/-- An arithmetic progression with given conditions -/
structure ArithmeticProgression where
  a : ℕ → ℚ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1
  first_term : a 1 = 2
  sum_condition : a 3 + a 6 = 8

/-- The 8th term of the arithmetic progression is 6 -/
theorem eighth_term_is_six (ap : ArithmeticProgression) : ap.a 8 = 6 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_is_six_l1454_145446


namespace NUMINAMATH_CALUDE_prob_two_females_is_two_fifths_l1454_145461

/-- Represents the survey data for students' preferences on breeding small animal A -/
structure SurveyData where
  male_like : ℕ
  male_dislike : ℕ
  female_like : ℕ
  female_dislike : ℕ

/-- Calculates the probability of selecting two females from a stratified sample -/
def prob_two_females (data : SurveyData) : ℚ :=
  let total_like := data.male_like + data.female_like
  let female_ratio := data.female_like / total_like
  let num_females_selected := 6 * female_ratio
  (num_females_selected * (num_females_selected - 1)) / (6 * 5)

/-- The main theorem to be proved -/
theorem prob_two_females_is_two_fifths (data : SurveyData) 
  (h1 : data.male_like = 20)
  (h2 : data.male_dislike = 30)
  (h3 : data.female_like = 40)
  (h4 : data.female_dislike = 10) :
  prob_two_females data = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_females_is_two_fifths_l1454_145461


namespace NUMINAMATH_CALUDE_nancy_pencils_proof_l1454_145452

/-- The number of pencils Nancy placed in the drawer -/
def pencils_added (initial_pencils total_pencils : ℕ) : ℕ :=
  total_pencils - initial_pencils

theorem nancy_pencils_proof (initial_pencils total_pencils : ℕ) 
  (h1 : initial_pencils = 27)
  (h2 : total_pencils = 72) :
  pencils_added initial_pencils total_pencils = 45 := by
  sorry

#eval pencils_added 27 72

end NUMINAMATH_CALUDE_nancy_pencils_proof_l1454_145452


namespace NUMINAMATH_CALUDE_eleventh_number_with_digit_sum_13_l1454_145489

/-- A function that returns the sum of digits of a positive integer -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- A function that returns the nth positive integer whose digits sum to 13 -/
def nthNumberWithDigitSum13 (n : ℕ) : ℕ := sorry

/-- Theorem stating that the 11th number with digit sum 13 is 175 -/
theorem eleventh_number_with_digit_sum_13 : nthNumberWithDigitSum13 11 = 175 := by sorry

end NUMINAMATH_CALUDE_eleventh_number_with_digit_sum_13_l1454_145489


namespace NUMINAMATH_CALUDE_fractional_inequality_condition_l1454_145464

theorem fractional_inequality_condition (x : ℝ) :
  (∀ x, 1 / x < 1 → x > 1) ∧ (∃ x, x > 1 ∧ ¬(1 / x < 1)) :=
by sorry

end NUMINAMATH_CALUDE_fractional_inequality_condition_l1454_145464


namespace NUMINAMATH_CALUDE_rectangle_area_excluding_hole_l1454_145441

variable (x : ℝ)

def large_rectangle_length : ℝ := 2 * x + 4
def large_rectangle_width : ℝ := x + 7
def hole_length : ℝ := x + 2
def hole_width : ℝ := 3 * x - 5

theorem rectangle_area_excluding_hole (h : x > 5/3) :
  large_rectangle_length x * large_rectangle_width x - hole_length x * hole_width x = -x^2 + 17*x + 38 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_excluding_hole_l1454_145441


namespace NUMINAMATH_CALUDE_sum_specific_arithmetic_progression_l1454_145417

/-- Sum of an arithmetic progression -/
def sum_arithmetic_progression (a : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

/-- Number of terms in an arithmetic progression -/
def num_terms_arithmetic_progression (a : ℤ) (d : ℤ) (l : ℤ) : ℕ :=
  ((l - a) / d).toNat + 1

theorem sum_specific_arithmetic_progression :
  let a : ℤ := -45  -- First term
  let d : ℤ := 2    -- Common difference
  let l : ℤ := 23   -- Last term
  let n : ℕ := num_terms_arithmetic_progression a d l
  sum_arithmetic_progression a d n = -385 := by
sorry

end NUMINAMATH_CALUDE_sum_specific_arithmetic_progression_l1454_145417


namespace NUMINAMATH_CALUDE_sequence_ratio_theorem_l1454_145488

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- Property of the sequence: a_{n+1} - 2a_n = 0 for all n -/
def HasConstantRatio (a : Sequence) : Prop :=
  ∀ n : ℕ, a (n + 1) - 2 * a n = 0

/-- Property of the sequence: a_n ≠ 0 for all n -/
def IsNonZero (a : Sequence) : Prop :=
  ∀ n : ℕ, a n ≠ 0

/-- The main theorem -/
theorem sequence_ratio_theorem (a : Sequence) 
  (h1 : HasConstantRatio a) (h2 : IsNonZero a) : 
  (2 * a 1 + a 2) / (a 3 + a 5) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sequence_ratio_theorem_l1454_145488


namespace NUMINAMATH_CALUDE_sum_of_tenth_powers_l1454_145463

theorem sum_of_tenth_powers (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^10 + b^10 = 123 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_tenth_powers_l1454_145463


namespace NUMINAMATH_CALUDE_hamburgers_count_l1454_145440

/-- The number of hamburgers initially made -/
def initial_hamburgers : ℝ := 9.0

/-- The number of additional hamburgers made -/
def additional_hamburgers : ℝ := 3.0

/-- The total number of hamburgers made -/
def total_hamburgers : ℝ := initial_hamburgers + additional_hamburgers

theorem hamburgers_count : total_hamburgers = 12.0 := by
  sorry

end NUMINAMATH_CALUDE_hamburgers_count_l1454_145440


namespace NUMINAMATH_CALUDE_top_is_multiple_of_four_l1454_145410

/-- Represents a number pyramid with 4 rows -/
structure NumberPyramid where
  bottom_row : Fin 4 → ℤ
  second_row : Fin 3 → ℤ
  third_row : Fin 2 → ℤ
  top : ℤ

/-- Defines a valid number pyramid where each cell above the bottom row
    is the sum of the two cells below it, and the second row contains equal integers -/
def is_valid_pyramid (p : NumberPyramid) : Prop :=
  (∃ n : ℤ, ∀ i : Fin 3, p.second_row i = n) ∧
  (∀ i : Fin 2, p.third_row i = p.second_row i + p.second_row (i + 1)) ∧
  p.top = p.third_row 0 + p.third_row 1

theorem top_is_multiple_of_four (p : NumberPyramid) (h : is_valid_pyramid p) :
  ∃ k : ℤ, p.top = 4 * k :=
sorry

end NUMINAMATH_CALUDE_top_is_multiple_of_four_l1454_145410


namespace NUMINAMATH_CALUDE_magnitude_of_vector_combination_l1454_145430

-- Define the vector type
def Vec2D := ℝ × ℝ

-- Define the angle between vectors a and b
def angle_between (a b : Vec2D) : ℝ := sorry

-- Define the magnitude of a vector
def magnitude (v : Vec2D) : ℝ := sorry

-- Define the dot product of two vectors
def dot_product (a b : Vec2D) : ℝ := sorry

-- Define the vector subtraction
def vec_sub (a b : Vec2D) : Vec2D := sorry

-- Define the vector scalar multiplication
def vec_scalar_mul (r : ℝ) (v : Vec2D) : Vec2D := sorry

theorem magnitude_of_vector_combination (a b : Vec2D) :
  angle_between a b = 2 * Real.pi / 3 →
  a = (3/5, -4/5) →
  magnitude b = 2 →
  magnitude (vec_sub (vec_scalar_mul 2 a) b) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_vector_combination_l1454_145430


namespace NUMINAMATH_CALUDE_no_solution_to_diophantine_equation_l1454_145421

theorem no_solution_to_diophantine_equation :
  ¬ ∃ (x y z t : ℕ), 3 * x^4 + 5 * y^4 + 7 * z^4 = 11 * t^4 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_to_diophantine_equation_l1454_145421


namespace NUMINAMATH_CALUDE_f_range_l1454_145413

noncomputable def f (x : ℝ) : ℝ := 4^x - 2^(x+2) - 3

theorem f_range : Set.range f = Set.Ici (-7) := by sorry

end NUMINAMATH_CALUDE_f_range_l1454_145413
