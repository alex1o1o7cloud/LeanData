import Mathlib

namespace NUMINAMATH_CALUDE_multiple_solutions_exist_l2776_277673

-- Define the system of equations
def system (x y z w : ℝ) : Prop :=
  x = z + w - z*w ∧
  y = w + x - w*x ∧
  z = x + y - x*y ∧
  w = y + z - y*z

-- Theorem statement
theorem multiple_solutions_exist :
  ∃ (x₁ y₁ z₁ w₁ x₂ y₂ z₂ w₂ : ℝ),
    system x₁ y₁ z₁ w₁ ∧
    system x₂ y₂ z₂ w₂ ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂ ∨ z₁ ≠ z₂ ∨ w₁ ≠ w₂) :=
by
  sorry


end NUMINAMATH_CALUDE_multiple_solutions_exist_l2776_277673


namespace NUMINAMATH_CALUDE_percent_relation_l2776_277604

theorem percent_relation (a b c : ℝ) 
  (h1 : c = 0.30 * a) 
  (h2 : c = 0.25 * b) : 
  b = 1.2 * a := by
sorry

end NUMINAMATH_CALUDE_percent_relation_l2776_277604


namespace NUMINAMATH_CALUDE_intersection_of_lines_l2776_277659

theorem intersection_of_lines : ∃! p : ℚ × ℚ, 
  8 * p.1 - 3 * p.2 = 24 ∧ 5 * p.1 + 2 * p.2 = 17 ∧ p = (99/31, 16/31) := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l2776_277659


namespace NUMINAMATH_CALUDE_no_integer_solution_x2_plus_y2_eq_3z2_l2776_277681

theorem no_integer_solution_x2_plus_y2_eq_3z2 :
  ∀ (x y z : ℤ), x^2 + y^2 = 3 * z^2 → x = 0 ∧ y = 0 ∧ z = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_integer_solution_x2_plus_y2_eq_3z2_l2776_277681


namespace NUMINAMATH_CALUDE_julia_tag_tuesday_l2776_277602

/-- The number of kids Julia played tag with on Monday -/
def monday_kids : ℕ := 7

/-- The total number of kids Julia played tag with -/
def total_kids : ℕ := 20

/-- The number of kids Julia played tag with on Tuesday -/
def tuesday_kids : ℕ := total_kids - monday_kids

theorem julia_tag_tuesday : tuesday_kids = 13 := by
  sorry

end NUMINAMATH_CALUDE_julia_tag_tuesday_l2776_277602


namespace NUMINAMATH_CALUDE_claudias_water_problem_l2776_277638

/-- The number of 4-ounce glasses that can be filled with the remaining water --/
def remaining_glasses (total_water : ℕ) (five_ounce_count : ℕ) (eight_ounce_count : ℕ) : ℕ :=
  (total_water - (five_ounce_count * 5 + eight_ounce_count * 8)) / 4

/-- Theorem stating that given the initial conditions, 15 four-ounce glasses can be filled --/
theorem claudias_water_problem :
  remaining_glasses 122 6 4 = 15 := by
  sorry

end NUMINAMATH_CALUDE_claudias_water_problem_l2776_277638


namespace NUMINAMATH_CALUDE_interest_difference_approx_l2776_277688

def principal : ℝ := 147.69
def rate : ℝ := 0.15
def time1 : ℝ := 3.5
def time2 : ℝ := 10

def interest (p r t : ℝ) : ℝ := p * r * t

theorem interest_difference_approx :
  ∃ ε > 0, ε < 0.001 ∧ 
  |interest principal rate time2 - interest principal rate time1 - 143.998| < ε :=
sorry

end NUMINAMATH_CALUDE_interest_difference_approx_l2776_277688


namespace NUMINAMATH_CALUDE_triangle_properties_l2776_277687

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem about the triangle properties -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.c = 2 * t.b) 
  (h2 : 2 * Real.sin t.A = 3 * Real.sin (2 * t.C)) 
  (h3 : (1/2) * t.a * t.b * Real.sin t.C = (3 * Real.sqrt 7) / 2) : 
  t.a = (3 * Real.sqrt 2 / 2) * t.b ∧ 
  (t.c * ((3 * Real.sqrt 7) / 4)) / (2 * ((3 * Real.sqrt 7) / 2)) = (3 * Real.sqrt 7) / 4 := by
  sorry

#check triangle_properties

end NUMINAMATH_CALUDE_triangle_properties_l2776_277687


namespace NUMINAMATH_CALUDE_problem_solution_l2776_277698

theorem problem_solution (m : ℤ) (a : ℝ) : 
  ((-2 : ℝ)^(2*m) = a^(3-m)) → (m = 1) → (a = 2) := by sorry

end NUMINAMATH_CALUDE_problem_solution_l2776_277698


namespace NUMINAMATH_CALUDE_museum_wings_l2776_277612

/-- Represents a museum with paintings and artifacts -/
structure Museum where
  painting_wings : ℕ
  artifact_wings : ℕ
  large_paintings : ℕ
  small_paintings : ℕ
  artifacts_per_wing : ℕ

/-- Calculates the total number of paintings in the museum -/
def total_paintings (m : Museum) : ℕ :=
  m.large_paintings + m.small_paintings

/-- Calculates the total number of artifacts in the museum -/
def total_artifacts (m : Museum) : ℕ :=
  m.artifact_wings * m.artifacts_per_wing

/-- Theorem stating the total number of wings in the museum -/
theorem museum_wings (m : Museum) 
  (h1 : m.painting_wings = 3)
  (h2 : m.large_paintings = 1)
  (h3 : m.small_paintings = 24)
  (h4 : m.artifacts_per_wing = 20)
  (h5 : total_artifacts m = 4 * total_paintings m) :
  m.painting_wings + m.artifact_wings = 8 := by
  sorry

#check museum_wings

end NUMINAMATH_CALUDE_museum_wings_l2776_277612


namespace NUMINAMATH_CALUDE_symmetric_points_sum_power_l2776_277640

/-- Two points P₁ and P₂ are symmetric about the x-axis if their x-coordinates are equal
    and their y-coordinates are opposite in sign but equal in absolute value. -/
def symmetric_about_x_axis (P₁ P₂ : ℝ × ℝ) : Prop :=
  P₁.1 = P₂.1 ∧ P₁.2 = -P₂.2

theorem symmetric_points_sum_power (a b : ℝ) :
  symmetric_about_x_axis (a - 1, 5) (2, b - 1) →
  (a + b)^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_power_l2776_277640


namespace NUMINAMATH_CALUDE_no_infinite_sequence_with_greater_than_neighbors_average_l2776_277634

theorem no_infinite_sequence_with_greater_than_neighbors_average :
  ¬ ∃ (a : ℕ → ℕ+), ∀ n : ℕ, n ≥ 1 → (a n : ℚ) > ((a (n - 1) : ℚ) + (a (n + 1) : ℚ)) / 2 :=
sorry

end NUMINAMATH_CALUDE_no_infinite_sequence_with_greater_than_neighbors_average_l2776_277634


namespace NUMINAMATH_CALUDE_product_of_square_roots_l2776_277603

theorem product_of_square_roots (x y z : ℝ) :
  x = 75 → y = 48 → z = 12 → Real.sqrt x * Real.sqrt y * Real.sqrt z = 120 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_product_of_square_roots_l2776_277603


namespace NUMINAMATH_CALUDE_divisibility_theorem_l2776_277637

theorem divisibility_theorem (a b n : ℕ) (h : a^n ∣ b) : a^(n+1) ∣ ((a+1)^b - 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_theorem_l2776_277637


namespace NUMINAMATH_CALUDE_total_time_outside_class_l2776_277649

def recess_break_1 : ℕ := 15
def recess_break_2 : ℕ := 15
def lunch_break : ℕ := 30
def additional_recess : ℕ := 20

theorem total_time_outside_class : 
  2 * recess_break_1 + lunch_break + additional_recess = 80 := by
  sorry

end NUMINAMATH_CALUDE_total_time_outside_class_l2776_277649


namespace NUMINAMATH_CALUDE_C1_intersects_C2_l2776_277615

-- Define the line C1
def C1 (x : ℝ) : ℝ := 2 * x - 3

-- Define the circle C2
def C2 (x y : ℝ) : Prop := (x - 3)^2 + (y - 5)^2 = 25

-- Theorem stating that C1 and C2 intersect
theorem C1_intersects_C2 : ∃ (x y : ℝ), y = C1 x ∧ C2 x y := by
  sorry

end NUMINAMATH_CALUDE_C1_intersects_C2_l2776_277615


namespace NUMINAMATH_CALUDE_diagonals_perpendicular_l2776_277608

-- Define a cube
structure Cube where
  -- Add necessary properties of a cube
  is_cube : Bool

-- Define the angle between diagonals of adjacent faces
def angle_between_diagonals (c : Cube) : ℝ :=
  sorry

-- Theorem statement
theorem diagonals_perpendicular (c : Cube) :
  angle_between_diagonals c = 90 :=
sorry

end NUMINAMATH_CALUDE_diagonals_perpendicular_l2776_277608


namespace NUMINAMATH_CALUDE_expression_nonnegative_l2776_277679

theorem expression_nonnegative (a b c d e : ℝ) : 
  (a-b)*(a-c)*(a-d)*(a-e) + (b-a)*(b-c)*(b-d)*(b-e) + (c-a)*(c-b)*(c-d)*(c-e) +
  (d-a)*(d-b)*(d-c)*(d-e) + (e-a)*(e-b)*(e-c)*(e-d) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_nonnegative_l2776_277679


namespace NUMINAMATH_CALUDE_intersection_and_subsets_l2776_277697

def A : Set ℤ := {-1, 1, 2}
def B : Set ℤ := {x | x ≥ 0}

theorem intersection_and_subsets :
  (A ∩ B = {1, 2}) ∧
  (Set.powerset (A ∩ B) = {{1}, {2}, ∅, {1, 2}}) := by
  sorry

end NUMINAMATH_CALUDE_intersection_and_subsets_l2776_277697


namespace NUMINAMATH_CALUDE_roof_collapse_time_l2776_277671

/-- Calculates the number of days for a roof to collapse under falling leaves. -/
def days_to_collapse (roof_capacity : ℕ) (leaves_per_day : ℕ) (leaves_per_pound : ℕ) : ℕ :=
  (roof_capacity * leaves_per_pound) / leaves_per_day

/-- Proves that given the specified conditions, it takes 5000 days for the roof to collapse. -/
theorem roof_collapse_time :
  days_to_collapse 500 100 1000 = 5000 := by
  sorry

#eval days_to_collapse 500 100 1000

end NUMINAMATH_CALUDE_roof_collapse_time_l2776_277671


namespace NUMINAMATH_CALUDE_log_sin_cos_theorem_l2776_277606

theorem log_sin_cos_theorem (x n : ℝ) 
  (h1 : Real.log (Real.sin x) + Real.log (Real.cos x) = -2)
  (h2 : Real.log (Real.sin x + Real.cos x) = (Real.log n - 2) / 2) : 
  n = Real.exp 2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_log_sin_cos_theorem_l2776_277606


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l2776_277630

/-- Proves that a parabola and a line intersect at two specific points -/
theorem parabola_line_intersection :
  let parabola (x : ℝ) := 2 * x^2 - 8 * x + 10
  let line (x : ℝ) := x + 1
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    parabola x₁ = line x₁ ∧
    parabola x₂ = line x₂ ∧
    ((x₁ = 3 ∧ parabola x₁ = 4) ∨ (x₁ = 3/2 ∧ parabola x₁ = 5/2)) ∧
    ((x₂ = 3 ∧ parabola x₂ = 4) ∨ (x₂ = 3/2 ∧ parabola x₂ = 5/2)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l2776_277630


namespace NUMINAMATH_CALUDE_max_profit_theorem_l2776_277639

/-- Represents the profit function for a company selling two types of multimedia devices -/
def profit_function (x : ℝ) : ℝ := -0.1 * x + 20

/-- Represents the constraint on the number of devices -/
def constraint (x : ℝ) : Prop := 4 * x ≥ 50 - x

theorem max_profit_theorem :
  ∃ (x : ℝ), 
    constraint x ∧ 
    profit_function x = 19 ∧ 
    ∀ (y : ℝ), constraint y → profit_function y ≤ profit_function x :=
by
  sorry

end NUMINAMATH_CALUDE_max_profit_theorem_l2776_277639


namespace NUMINAMATH_CALUDE_manager_salary_calculation_l2776_277609

/-- The daily salary of a manager in a grocery store -/
def manager_salary : ℝ := 5

/-- The daily salary of a clerk in a grocery store -/
def clerk_salary : ℝ := 2

/-- The number of managers in the grocery store -/
def num_managers : ℕ := 2

/-- The number of clerks in the grocery store -/
def num_clerks : ℕ := 3

/-- The total daily salary of all employees in the grocery store -/
def total_salary : ℝ := 16

theorem manager_salary_calculation :
  manager_salary * num_managers + clerk_salary * num_clerks = total_salary :=
by sorry

end NUMINAMATH_CALUDE_manager_salary_calculation_l2776_277609


namespace NUMINAMATH_CALUDE_inequality_proof_l2776_277685

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_order : a ≥ b ∧ b ≥ c)
  (h_sum : a + b + c ≤ 1) :
  a^2 + 3*b^2 + 5*c^2 ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2776_277685


namespace NUMINAMATH_CALUDE_no_very_convex_function_l2776_277627

theorem no_very_convex_function :
  ∀ f : ℝ → ℝ, ∃ x y : ℝ, (f x + f y) / 2 < f ((x + y) / 2) + |x - y| :=
by sorry

end NUMINAMATH_CALUDE_no_very_convex_function_l2776_277627


namespace NUMINAMATH_CALUDE_cricket_matches_l2776_277684

theorem cricket_matches (score1 score2 overall_avg : ℚ) (matches1 matches2 : ℕ) 
  (h1 : score1 = 60)
  (h2 : score2 = 50)
  (h3 : overall_avg = 54)
  (h4 : matches1 = 2)
  (h5 : matches2 = 3) :
  matches1 + matches2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_cricket_matches_l2776_277684


namespace NUMINAMATH_CALUDE_washer_dryer_total_cost_l2776_277614

/-- The cost of a washer-dryer combination -/
def washer_dryer_cost (dryer_cost washer_cost_difference : ℕ) : ℕ :=
  dryer_cost + (dryer_cost + washer_cost_difference)

/-- Theorem: The washer-dryer combination costs $1200 -/
theorem washer_dryer_total_cost : 
  washer_dryer_cost 490 220 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_washer_dryer_total_cost_l2776_277614


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_seven_twelfths_l2776_277644

theorem sqrt_difference_equals_seven_twelfths :
  Real.sqrt (16 / 9) - Real.sqrt (9 / 16) = 7 / 12 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_seven_twelfths_l2776_277644


namespace NUMINAMATH_CALUDE_equation_solutions_function_property_l2776_277690

-- Part a
theorem equation_solutions (x : ℝ) : 2^x = x + 1 ↔ x = 0 ∨ x = 1 := by sorry

-- Part b
theorem function_property (f : ℝ → ℝ) (h : ∀ x, (f ∘ f) x = 2^x - 1) : f 0 + f 1 = 1 := by sorry

end NUMINAMATH_CALUDE_equation_solutions_function_property_l2776_277690


namespace NUMINAMATH_CALUDE_a_plus_b_value_l2776_277682

-- Define the functions f and h
def f (a b x : ℝ) : ℝ := a * x + b
def h (x : ℝ) : ℝ := 3 * x - 6

-- State the theorem
theorem a_plus_b_value (a b : ℝ) : 
  (∀ x, h (f a b x) = 4 * x + 3) → a + b = 13 / 3 := by
  sorry

end NUMINAMATH_CALUDE_a_plus_b_value_l2776_277682


namespace NUMINAMATH_CALUDE_inequality_solution_l2776_277601

theorem inequality_solution (x : ℝ) : 
  (5 ≤ (x - 1) / (3 * x - 7) ∧ (x - 1) / (3 * x - 7) < 10) ↔ 
  (69 / 29 < x ∧ x ≤ 17 / 7) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2776_277601


namespace NUMINAMATH_CALUDE_isosceles_triangle_angle_bisector_length_l2776_277691

theorem isosceles_triangle_angle_bisector_length 
  (AB BC AC : ℝ) (h_isosceles : AC = BC) (h_base : AB = 5) (h_lateral : AC = 20) :
  let AD := Real.sqrt (AB * AC * (1 - BC / (AB + AC)))
  AD = 6 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_angle_bisector_length_l2776_277691


namespace NUMINAMATH_CALUDE_medicine_types_count_l2776_277658

/-- The number of medical boxes -/
def num_boxes : ℕ := 5

/-- The number of boxes each medicine appears in -/
def boxes_per_medicine : ℕ := 2

/-- Calculates the binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- The number of types of medicine -/
def num_medicine_types : ℕ := binomial num_boxes boxes_per_medicine

theorem medicine_types_count : num_medicine_types = 10 := by
  sorry

end NUMINAMATH_CALUDE_medicine_types_count_l2776_277658


namespace NUMINAMATH_CALUDE_store_holiday_customers_l2776_277652

/-- The number of customers a store sees during holiday season -/
def holiday_customers (regular_rate : ℕ) (hours : ℕ) : ℕ :=
  2 * regular_rate * hours

/-- Theorem: Given the regular customer rate and time period, 
    the store will see 2800 customers during the holiday season -/
theorem store_holiday_customers :
  holiday_customers 175 8 = 2800 := by
  sorry

end NUMINAMATH_CALUDE_store_holiday_customers_l2776_277652


namespace NUMINAMATH_CALUDE_problem_statement_l2776_277666

theorem problem_statement : 5 * 301 + 4 * 301 + 3 * 301 + 300 = 3912 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2776_277666


namespace NUMINAMATH_CALUDE_last_digit_of_nine_power_l2776_277653

theorem last_digit_of_nine_power (n : ℕ) : 9^(9^8) % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_nine_power_l2776_277653


namespace NUMINAMATH_CALUDE_f_of_five_equals_ln_five_l2776_277641

-- Define the function f
noncomputable def f : ℝ → ℝ := fun y ↦ Real.log y

-- State the theorem
theorem f_of_five_equals_ln_five :
  (∀ x : ℝ, f (Real.exp x) = x) → f 5 = Real.log 5 := by
  sorry

end NUMINAMATH_CALUDE_f_of_five_equals_ln_five_l2776_277641


namespace NUMINAMATH_CALUDE_product_divisible_by_1419_l2776_277645

theorem product_divisible_by_1419 : ∃ k : ℕ, 86 * 87 * 88 = 1419 * k := by
  sorry

end NUMINAMATH_CALUDE_product_divisible_by_1419_l2776_277645


namespace NUMINAMATH_CALUDE_parabola_translation_l2776_277665

/-- Original parabola function -/
def f (x : ℝ) : ℝ := x^2 + 1

/-- Translated parabola function -/
def g (x : ℝ) : ℝ := x^2 + 4*x + 5

/-- Translation function -/
def translate (x : ℝ) : ℝ := x + 2

theorem parabola_translation :
  ∀ x : ℝ, g x = f (translate x) :=
by sorry

end NUMINAMATH_CALUDE_parabola_translation_l2776_277665


namespace NUMINAMATH_CALUDE_cos_sum_thirteen_l2776_277672

theorem cos_sum_thirteen : 
  Real.cos (2 * Real.pi / 13) + Real.cos (6 * Real.pi / 13) + Real.cos (8 * Real.pi / 13) = (Real.sqrt 13 - 1) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_thirteen_l2776_277672


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l2776_277655

theorem right_triangle_third_side 
  (x y z : ℝ) 
  (h_right_triangle : x^2 + y^2 = z^2) 
  (h_equation : |x - 3| + Real.sqrt (2 * y - 8) = 0) : 
  z = Real.sqrt 7 ∨ z = 5 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l2776_277655


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2776_277650

/-- The sequence a_n defined by n^2 - c*n for n ∈ ℕ+ is increasing -/
def is_increasing_sequence (c : ℝ) : Prop :=
  ∀ n : ℕ+, (n + 1)^2 - c*(n + 1) > n^2 - c*n

/-- c ≤ 2 is a sufficient but not necessary condition for the sequence to be increasing -/
theorem sufficient_not_necessary_condition :
  (∀ c : ℝ, c ≤ 2 → is_increasing_sequence c) ∧
  (∃ c : ℝ, c > 2 ∧ is_increasing_sequence c) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2776_277650


namespace NUMINAMATH_CALUDE_modular_inverse_of_5_mod_31_l2776_277646

theorem modular_inverse_of_5_mod_31 :
  ∃ x : ℕ, x < 31 ∧ (5 * x) % 31 = 1 :=
by
  use 25
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_5_mod_31_l2776_277646


namespace NUMINAMATH_CALUDE_stock_rise_amount_l2776_277643

/-- Represents the daily change in stock value -/
structure StockChange where
  morning_rise : ℝ
  afternoon_fall : ℝ

/-- Calculates the stock value after n days given initial value and daily change -/
def stock_value_after_days (initial_value : ℝ) (daily_change : StockChange) (n : ℕ) : ℝ :=
  initial_value + n * (daily_change.morning_rise - daily_change.afternoon_fall)

theorem stock_rise_amount (initial_value : ℝ) (daily_change : StockChange) :
  initial_value = 100 →
  daily_change.afternoon_fall = 1 →
  stock_value_after_days initial_value daily_change 100 = 200 →
  daily_change.morning_rise = 2 := by
  sorry

#eval stock_value_after_days 100 ⟨2, 1⟩ 100

end NUMINAMATH_CALUDE_stock_rise_amount_l2776_277643


namespace NUMINAMATH_CALUDE_sin_x_in_terms_of_a_and_b_l2776_277626

theorem sin_x_in_terms_of_a_and_b (a b x : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : 0 < x) (h4 : x < π/2) (h5 : Real.tan x = (3*a*b) / (a^2 - b^2)) : 
  Real.sin x = (3*a*b) / Real.sqrt (a^4 + 7*a^2*b^2 + b^4) := by
  sorry

end NUMINAMATH_CALUDE_sin_x_in_terms_of_a_and_b_l2776_277626


namespace NUMINAMATH_CALUDE_gain_percent_for_equal_cost_and_selling_l2776_277674

/-- Given that the cost price of 50 articles equals the selling price of 30 articles,
    prove that the gain percent is 200/3. -/
theorem gain_percent_for_equal_cost_and_selling (C S : ℝ) 
  (h : 50 * C = 30 * S) : 
  (S - C) / C * 100 = 200 / 3 := by
  sorry

end NUMINAMATH_CALUDE_gain_percent_for_equal_cost_and_selling_l2776_277674


namespace NUMINAMATH_CALUDE_charity_pastries_count_l2776_277696

theorem charity_pastries_count (total_volunteers : ℕ) 
  (group_a_percent group_b_percent group_c_percent : ℚ)
  (group_a_batches group_b_batches group_c_batches : ℕ)
  (group_a_trays group_b_trays group_c_trays : ℕ)
  (group_a_pastries group_b_pastries group_c_pastries : ℕ) :
  total_volunteers = 1500 →
  group_a_percent = 2/5 →
  group_b_percent = 7/20 →
  group_c_percent = 1/4 →
  group_a_batches = 10 →
  group_b_batches = 15 →
  group_c_batches = 8 →
  group_a_trays = 6 →
  group_b_trays = 4 →
  group_c_trays = 5 →
  group_a_pastries = 20 →
  group_b_pastries = 12 →
  group_c_pastries = 15 →
  (↑total_volunteers * group_a_percent).floor * group_a_batches * group_a_trays * group_a_pastries +
  (↑total_volunteers * group_b_percent).floor * group_b_batches * group_b_trays * group_b_pastries +
  (↑total_volunteers * group_c_percent).floor * group_c_batches * group_c_trays * group_c_pastries = 1323000 := by
  sorry


end NUMINAMATH_CALUDE_charity_pastries_count_l2776_277696


namespace NUMINAMATH_CALUDE_football_team_probability_l2776_277620

/-- Given a group of 10 people with 2 from football teams and 8 from basketball teams,
    proves the probability that both randomly selected people are from football teams,
    given that one is from a football team, is 1/9. -/
theorem football_team_probability :
  let total_people : ℕ := 10
  let football_people : ℕ := 2
  let basketball_people : ℕ := 8
  let total_selections : ℕ := 9  -- Total ways to select given one is from football
  let both_football : ℕ := 1     -- Ways to select both from football given one is from football
  football_people + basketball_people = total_people →
  (both_football : ℚ) / total_selections = 1 / 9 :=
by sorry

end NUMINAMATH_CALUDE_football_team_probability_l2776_277620


namespace NUMINAMATH_CALUDE_average_car_selections_l2776_277667

theorem average_car_selections (num_cars : ℕ) (num_clients : ℕ) (selections_per_client : ℕ) : 
  num_cars = 15 → num_clients = 15 → selections_per_client = 3 →
  (num_clients * selections_per_client) / num_cars = 3 := by
  sorry

end NUMINAMATH_CALUDE_average_car_selections_l2776_277667


namespace NUMINAMATH_CALUDE_girls_in_class_l2776_277694

theorem girls_in_class (total_students : ℕ) (girls : ℕ) (boys : ℕ) :
  total_students = 250 →
  girls + boys = total_students →
  girls = 2 * (total_students - (girls + boys - girls)) →
  girls = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_girls_in_class_l2776_277694


namespace NUMINAMATH_CALUDE_tan_non_intersection_l2776_277607

theorem tan_non_intersection :
  ∀ y : ℝ, ∃ k : ℤ, (2 * (π/8) + π/4) = k * π + π/2 :=
by sorry

end NUMINAMATH_CALUDE_tan_non_intersection_l2776_277607


namespace NUMINAMATH_CALUDE_even_function_properties_l2776_277636

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

theorem even_function_properties (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_decreasing : is_decreasing_on f (-5) (-2))
  (h_max : ∀ x, -5 ≤ x ∧ x ≤ -2 → f x ≤ 7) :
  is_increasing_on f 2 5 ∧ ∀ x, 2 ≤ x ∧ x ≤ 5 → f x ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_even_function_properties_l2776_277636


namespace NUMINAMATH_CALUDE_eldest_child_age_l2776_277621

-- Define the number of children
def num_children : ℕ := 8

-- Define the age difference between consecutive children
def age_difference : ℕ := 3

-- Define the total sum of ages
def total_age_sum : ℕ := 100

-- Theorem statement
theorem eldest_child_age :
  ∃ (youngest_age : ℕ),
    (youngest_age + (num_children - 1) * age_difference) +
    (youngest_age + (num_children - 2) * age_difference) +
    (youngest_age + (num_children - 3) * age_difference) +
    (youngest_age + (num_children - 4) * age_difference) +
    (youngest_age + (num_children - 5) * age_difference) +
    (youngest_age + (num_children - 6) * age_difference) +
    (youngest_age + (num_children - 7) * age_difference) +
    youngest_age = total_age_sum →
    youngest_age + (num_children - 1) * age_difference = 23 :=
by
  sorry

end NUMINAMATH_CALUDE_eldest_child_age_l2776_277621


namespace NUMINAMATH_CALUDE_count_satisfying_numbers_l2776_277625

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

def sum_of_digits (n : ℕ) : ℕ :=
  (n % 10) + (n / 10)

def satisfies_conditions (n : ℕ) : Prop :=
  is_two_digit n ∧
  n + reverse_digits n = 110 ∧
  sum_of_digits n % 3 = 0

theorem count_satisfying_numbers :
  ∃! (s : Finset ℕ), (∀ n ∈ s, satisfies_conditions n) ∧ s.card = 3 :=
sorry

end NUMINAMATH_CALUDE_count_satisfying_numbers_l2776_277625


namespace NUMINAMATH_CALUDE_x_value_proof_l2776_277622

theorem x_value_proof (x y : ℤ) (h1 : x + y = 4) (h2 : x - y = 36) : x = 20 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l2776_277622


namespace NUMINAMATH_CALUDE_square_from_l_pieces_l2776_277668

/-- Represents a three-cell L-shaped piece -/
structure LPiece :=
  (cells : Fin 3 → Fin 2 → Fin 2)

/-- Represents a square grid -/
structure Square (n : ℕ) :=
  (grid : Fin n → Fin n → Bool)

/-- Checks if a given square is filled completely -/
def is_filled (s : Square n) : Prop :=
  ∀ i j, s.grid i j = true

/-- Defines the ability to place L-pieces on a square grid -/
def can_place_pieces (n : ℕ) (pieces : List LPiece) (s : Square n) : Prop :=
  sorry

/-- The main theorem stating that it's possible to form a square using L-pieces -/
theorem square_from_l_pieces :
  ∃ (n : ℕ) (pieces : List LPiece) (s : Square n),
    can_place_pieces n pieces s ∧ is_filled s :=
  sorry

end NUMINAMATH_CALUDE_square_from_l_pieces_l2776_277668


namespace NUMINAMATH_CALUDE_table_tennis_team_members_l2776_277680

theorem table_tennis_team_members : ∃ (x : ℕ), x > 0 ∧ x ≤ 33 ∧ 
  (∃ (s r : ℕ), s + r = x ∧ 4 * s + 3 * r + 2 * x = 33) :=
by
  sorry

end NUMINAMATH_CALUDE_table_tennis_team_members_l2776_277680


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2776_277642

def geometric_sequence (n : ℕ) : ℝ := (-3) ^ (n - 1)

theorem geometric_sequence_sum :
  let a := geometric_sequence
  (a 1) + |a 2| + (a 3) + |a 4| + (a 5) = 121 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2776_277642


namespace NUMINAMATH_CALUDE_octagon_placement_l2776_277686

/-- A set of numbers from 1 to 12 -/
def CardSet : Set ℕ := {n | 1 ≤ n ∧ n ≤ 12}

/-- A function representing the placement of numbers on the octagon vertices -/
def Placement := Fin 8 → ℕ

/-- Predicate to check if a placement is valid according to the given conditions -/
def ValidPlacement (p : Placement) : Prop :=
  (∀ i, p i ∈ CardSet) ∧
  (∀ i, (p i + p ((i + 4) % 8)) % 3 = 0)

/-- The set of numbers not placed on the octagon -/
def NotPlaced (p : Placement) : Set ℕ := CardSet \ (Set.range p)

/-- Main theorem -/
theorem octagon_placement :
  ∀ p : Placement, ValidPlacement p → NotPlaced p = {3, 6, 9, 12} := by sorry

end NUMINAMATH_CALUDE_octagon_placement_l2776_277686


namespace NUMINAMATH_CALUDE_intersection_count_l2776_277656

/-- Represents a lattice point in the coordinate plane -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- Represents a circle centered at a lattice point -/
structure Circle where
  center : LatticePoint
  radius : ℚ

/-- Represents a square centered at a lattice point -/
structure Square where
  center : LatticePoint
  sideLength : ℚ

/-- Represents a line segment from (0,0) to (703, 299) -/
def lineSegment : Set (ℚ × ℚ) :=
  {p | ∃ t : ℚ, 0 ≤ t ∧ t ≤ 1 ∧ p = (703 * t, 299 * t)}

/-- Counts the number of intersections with squares and circles -/
def countIntersections (line : Set (ℚ × ℚ)) (squares : Set Square) (circles : Set Circle) : ℕ :=
  sorry

/-- Main theorem statement -/
theorem intersection_count :
  ∀ (squares : Set Square) (circles : Set Circle),
    (∀ p : LatticePoint, ∃ s ∈ squares, s.center = p ∧ s.sideLength = 2/5) →
    (∀ p : LatticePoint, ∃ c ∈ circles, c.center = p ∧ c.radius = 1/5) →
    countIntersections lineSegment squares circles = 2109 := by
  sorry

end NUMINAMATH_CALUDE_intersection_count_l2776_277656


namespace NUMINAMATH_CALUDE_total_amount_proof_l2776_277699

/-- Proves that the total amount of money divided into two parts is Rs. 2600 -/
theorem total_amount_proof (total : ℝ) (part1 : ℝ) (part2 : ℝ) (rate1 : ℝ) (rate2 : ℝ) (income : ℝ) :
  part1 + part2 = total →
  part1 = 1600 →
  rate1 = 0.05 →
  rate2 = 0.06 →
  part1 * rate1 + part2 * rate2 = income →
  income = 140 →
  total = 2600 := by
  sorry

#check total_amount_proof

end NUMINAMATH_CALUDE_total_amount_proof_l2776_277699


namespace NUMINAMATH_CALUDE_tetrahedron_properties_l2776_277664

-- Define the vertices of the tetrahedron
def A1 : ℝ × ℝ × ℝ := (1, 2, 0)
def A2 : ℝ × ℝ × ℝ := (3, 0, -3)
def A3 : ℝ × ℝ × ℝ := (5, 2, 6)
def A4 : ℝ × ℝ × ℝ := (8, 4, -9)

-- Function to calculate the volume of a tetrahedron
def tetrahedron_volume (a b c d : ℝ × ℝ × ℝ) : ℝ := sorry

-- Function to calculate the height of a tetrahedron
def tetrahedron_height (a b c d : ℝ × ℝ × ℝ) : ℝ := sorry

-- Theorem stating the volume and height of the specific tetrahedron
theorem tetrahedron_properties :
  tetrahedron_volume A1 A2 A3 A4 = 34 ∧
  tetrahedron_height A1 A2 A3 A4 = 7 + 2/7 :=
by sorry

end NUMINAMATH_CALUDE_tetrahedron_properties_l2776_277664


namespace NUMINAMATH_CALUDE_power_multiplication_l2776_277677

theorem power_multiplication (m : ℝ) : m^2 * m^3 = m^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l2776_277677


namespace NUMINAMATH_CALUDE_helmet_sales_theorem_l2776_277618

/-- Represents the monthly sales data and pricing information for helmets --/
structure HelmetSalesData where
  april_sales : ℕ
  june_sales : ℕ
  cost_price : ℕ
  reference_price : ℕ
  reference_volume : ℕ
  volume_change_rate : ℕ
  target_profit : ℕ

/-- Calculates the monthly growth rate given the sales data --/
def calculate_growth_rate (data : HelmetSalesData) : ℚ :=
  sorry

/-- Calculates the optimal selling price given the sales data --/
def calculate_optimal_price (data : HelmetSalesData) : ℕ :=
  sorry

/-- Theorem stating the correct growth rate and optimal price --/
theorem helmet_sales_theorem (data : HelmetSalesData) 
  (h1 : data.april_sales = 150)
  (h2 : data.june_sales = 216)
  (h3 : data.cost_price = 30)
  (h4 : data.reference_price = 40)
  (h5 : data.reference_volume = 600)
  (h6 : data.volume_change_rate = 10)
  (h7 : data.target_profit = 10000) :
  calculate_growth_rate data = 1/5 ∧ calculate_optimal_price data = 50 := by
  sorry

end NUMINAMATH_CALUDE_helmet_sales_theorem_l2776_277618


namespace NUMINAMATH_CALUDE_solution_pairs_l2776_277683

-- Define the predicate for the conditions
def satisfies_conditions (x y : ℕ+) : Prop :=
  (y ∣ x^2 + 1) ∧ (x^2 ∣ y^3 + 1)

-- State the theorem
theorem solution_pairs :
  ∀ x y : ℕ+, satisfies_conditions x y →
    ((x = 1 ∧ y = 1) ∨ (x = 1 ∧ y = 2) ∨ (x = 3 ∧ y = 2)) :=
by sorry

end NUMINAMATH_CALUDE_solution_pairs_l2776_277683


namespace NUMINAMATH_CALUDE_luisas_books_l2776_277623

theorem luisas_books (maddie_books amy_books : ℕ) (h1 : maddie_books = 15) (h2 : amy_books = 6)
  (h3 : ∃ luisa_books : ℕ, amy_books + luisa_books = maddie_books + 9) :
  ∃ luisa_books : ℕ, luisa_books = 18 := by
  sorry

end NUMINAMATH_CALUDE_luisas_books_l2776_277623


namespace NUMINAMATH_CALUDE_luck_represents_6789_l2776_277635

/-- Represents a mapping from characters to digits -/
def DigitMapping := Char → Nat

/-- The 12-letter code -/
def code : String := "AMAZING LUCK"

/-- The condition that the code represents digits 0-9 and repeats for two more digits -/
def valid_mapping (m : DigitMapping) : Prop :=
  ∀ i : Fin 12, 
    m (code.get ⟨i⟩) = if i < 10 then i else i - 10

/-- The substring we're interested in -/
def substring : String := "LUCK"

/-- The theorem to prove -/
theorem luck_represents_6789 (m : DigitMapping) (h : valid_mapping m) : 
  (m 'L', m 'U', m 'C', m 'K') = (6, 7, 8, 9) := by
  sorry

end NUMINAMATH_CALUDE_luck_represents_6789_l2776_277635


namespace NUMINAMATH_CALUDE_root_conditions_imply_sum_l2776_277619

/-- Given two polynomial equations with specific root conditions, prove that 100p + q = 502 -/
theorem root_conditions_imply_sum (p q : ℝ) : 
  (∃ (x y : ℝ), x ≠ y ∧ 
    (∀ z : ℝ, (z + p) * (z + q) * (z + 5) / (z + 2)^2 = 0 ↔ (z = x ∨ z = y)) ∧
    (z = -2 → (z + p) * (z + q) * (z + 5) ≠ 0)) →
  (∃ (u v : ℝ), u ≠ v ∧ 
    (∀ w : ℝ, (w + 2*p) * (w + 2) * (w + 3) / ((w + q) * (w + 5)) = 0 ↔ (w = u ∨ w = v)) ∧
    ((w = -q ∨ w = -5) → (w + 2*p) * (w + 2) * (w + 3) ≠ 0)) →
  100 * p + q = 502 := by
sorry

end NUMINAMATH_CALUDE_root_conditions_imply_sum_l2776_277619


namespace NUMINAMATH_CALUDE_work_completion_time_l2776_277632

/-- Represents the work rate of a group of workers -/
def WorkRate (num_workers : ℕ) (days : ℕ) : ℚ :=
  1 / (num_workers * days)

/-- The theorem statement -/
theorem work_completion_time 
  (men_rate : WorkRate 8 20 = WorkRate 12 20) 
  (total_work : ℚ := 1) :
  (6 : ℚ) * WorkRate 8 20 + (11 : ℚ) * WorkRate 12 20 = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l2776_277632


namespace NUMINAMATH_CALUDE_test_probabilities_l2776_277647

/-- Probability of an event occurring -/
def Prob (event : Prop) : ℝ := sorry

/-- The probability that individual A passes the test -/
def probA : ℝ := 0.8

/-- The probability that individual B passes the test -/
def probB : ℝ := 0.6

/-- The probability that individual C passes the test -/
def probC : ℝ := 0.5

/-- A passes the test -/
def A : Prop := sorry

/-- B passes the test -/
def B : Prop := sorry

/-- C passes the test -/
def C : Prop := sorry

theorem test_probabilities :
  (Prob A = probA) ∧
  (Prob B = probB) ∧
  (Prob C = probC) ∧
  (Prob (A ∧ B ∧ C) = 0.24) ∧
  (Prob (A ∨ B ∨ C) = 0.96) := by sorry

end NUMINAMATH_CALUDE_test_probabilities_l2776_277647


namespace NUMINAMATH_CALUDE_apple_bags_theorem_l2776_277689

def is_valid_total (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 6 * a + 12 * b ∧ 70 ≤ n ∧ n ≤ 80

theorem apple_bags_theorem : 
  {n : ℕ | is_valid_total n} = {72, 78} :=
sorry

end NUMINAMATH_CALUDE_apple_bags_theorem_l2776_277689


namespace NUMINAMATH_CALUDE_inequality_proof_l2776_277610

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a / b^2 + b / c^2 + c / a^2 ≥ 1 / a + 1 / b + 1 / c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2776_277610


namespace NUMINAMATH_CALUDE_window_dimensions_correct_l2776_277669

/-- Represents the dimensions of a rectangular window made of glass panes. -/
structure WindowDimensions where
  width : ℝ
  height : ℝ

/-- Calculates the dimensions of a rectangular window given the specifications. -/
def calculateWindowDimensions (paneWidth : ℝ) : WindowDimensions := {
  width := 4 * paneWidth + 10,
  height := 9 * paneWidth + 8
}

/-- Theorem stating that the calculated dimensions are correct for the given specifications. -/
theorem window_dimensions_correct (paneWidth : ℝ) :
  let window := calculateWindowDimensions paneWidth
  window.width = 4 * paneWidth + 10 ∧
  window.height = 9 * paneWidth + 8 :=
by sorry

end NUMINAMATH_CALUDE_window_dimensions_correct_l2776_277669


namespace NUMINAMATH_CALUDE_function_relationship_l2776_277648

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the properties of f
variable (h1 : ∀ x y, 0 < x ∧ x < y ∧ y < 2 → f x < f y)
variable (h2 : ∀ x, f (x + 2) = f (-x + 2))

-- State the theorem
theorem function_relationship :
  f (7/2) < f 1 ∧ f 1 < f (5/2) := by sorry

end NUMINAMATH_CALUDE_function_relationship_l2776_277648


namespace NUMINAMATH_CALUDE_curve_C_left_of_x_equals_2_l2776_277662

/-- The curve C is defined by the equation x³ + 2y² = 8 -/
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^3 + 2 * p.2^2 = 8}

/-- Theorem: All points on curve C have x-coordinate less than or equal to 2 -/
theorem curve_C_left_of_x_equals_2 : ∀ p ∈ C, p.1 ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_curve_C_left_of_x_equals_2_l2776_277662


namespace NUMINAMATH_CALUDE_product_calculation_l2776_277616

theorem product_calculation : 1500 * 2023 * 0.5023 * 50 = 306903675 := by
  sorry

end NUMINAMATH_CALUDE_product_calculation_l2776_277616


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2776_277651

theorem simplify_and_evaluate (x : ℕ) (h1 : x > 0) (h2 : 3 - x ≥ 0) :
  let expr := (1 - 1 / (x - 1)) / ((x^2 - 4*x + 4) / (x - 1))
  x = 3 → expr = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2776_277651


namespace NUMINAMATH_CALUDE_circus_dog_paws_l2776_277675

theorem circus_dog_paws (total_dogs : ℕ) (back_leg_fraction : ℚ) : total_dogs = 24 → back_leg_fraction = 2/3 → (total_dogs : ℚ) * back_leg_fraction * 2 + (total_dogs : ℚ) * (1 - back_leg_fraction) * 4 = 64 := by
  sorry

end NUMINAMATH_CALUDE_circus_dog_paws_l2776_277675


namespace NUMINAMATH_CALUDE_craig_apples_l2776_277654

/-- The number of apples Craig has after receiving more from Eugene -/
def total_apples (initial : Real) (received : Real) : Real :=
  initial + received

/-- Proof that Craig will have 27.0 apples -/
theorem craig_apples : total_apples 20.0 7.0 = 27.0 := by
  sorry

end NUMINAMATH_CALUDE_craig_apples_l2776_277654


namespace NUMINAMATH_CALUDE_sport_formulation_water_amount_l2776_277657

/-- Represents the ratio of ingredients in a drink formulation -/
structure DrinkRatio :=
  (flavoring : ℚ)
  (corn_syrup : ℚ)
  (water : ℚ)

/-- The standard formulation ratio -/
def standard_ratio : DrinkRatio :=
  ⟨1, 12, 30⟩

/-- The sport formulation ratio -/
def sport_ratio : DrinkRatio :=
  ⟨1, 4, 60⟩

/-- Theorem stating the relationship between corn syrup and water in the sport formulation -/
theorem sport_formulation_water_amount
  (corn_syrup_amount : ℚ)
  (h1 : corn_syrup_amount = 5)
  (h2 : sport_ratio.flavoring * sport_ratio.water = 
        2 * (standard_ratio.flavoring * standard_ratio.water))
  (h3 : sport_ratio.flavoring * sport_ratio.corn_syrup = 
        3 * (standard_ratio.flavoring * standard_ratio.corn_syrup)) :
  corn_syrup_amount * (sport_ratio.water / sport_ratio.corn_syrup) = 75 :=
sorry

end NUMINAMATH_CALUDE_sport_formulation_water_amount_l2776_277657


namespace NUMINAMATH_CALUDE_election_win_percentage_l2776_277660

theorem election_win_percentage 
  (total_votes : ℕ) 
  (geoff_percentage : ℚ) 
  (additional_votes_needed : ℕ) 
  (h1 : total_votes = 6000)
  (h2 : geoff_percentage = 1 / 100)
  (h3 : additional_votes_needed = 3000) : 
  ∃ (x : ℚ), x > 51 / 100 ∧ 
    x * total_votes ≤ (geoff_percentage * total_votes + additional_votes_needed) ∧ 
    ∀ (y : ℚ), y < x → y * total_votes < (geoff_percentage * total_votes + additional_votes_needed) :=
by
  sorry

end NUMINAMATH_CALUDE_election_win_percentage_l2776_277660


namespace NUMINAMATH_CALUDE_final_price_theorem_l2776_277676

def mothers_day_discount : ℝ := 0.10
def additional_children_discount : ℝ := 0.04
def vip_discount : ℝ := 0.05
def shoes_cost : ℝ := 125
def handbag_cost : ℝ := 75
def min_purchase : ℝ := 150

def total_cost : ℝ := shoes_cost + handbag_cost

def discounted_price (price : ℝ) : ℝ :=
  let price_after_mothers_day := price * (1 - mothers_day_discount)
  let price_after_children := price_after_mothers_day * (1 - additional_children_discount)
  price_after_children * (1 - vip_discount)

theorem final_price_theorem :
  total_cost ≥ min_purchase →
  discounted_price total_cost = 164.16 :=
by sorry

end NUMINAMATH_CALUDE_final_price_theorem_l2776_277676


namespace NUMINAMATH_CALUDE_workshop_average_salary_l2776_277611

theorem workshop_average_salary 
  (total_workers : ℕ) 
  (num_technicians : ℕ) 
  (technician_salary : ℕ) 
  (other_salary : ℕ) 
  (h1 : total_workers = 18) 
  (h2 : num_technicians = 6) 
  (h3 : technician_salary = 12000) 
  (h4 : other_salary = 6000) :
  (num_technicians * technician_salary + (total_workers - num_technicians) * other_salary) / total_workers = 8000 :=
by sorry

end NUMINAMATH_CALUDE_workshop_average_salary_l2776_277611


namespace NUMINAMATH_CALUDE_cubic_root_sum_product_l2776_277617

theorem cubic_root_sum_product (p q r : ℝ) : 
  (6 * p^3 - 4 * p^2 + 15 * p - 10 = 0) ∧ 
  (6 * q^3 - 4 * q^2 + 15 * q - 10 = 0) ∧ 
  (6 * r^3 - 4 * r^2 + 15 * r - 10 = 0) →
  p * q + q * r + r * p = 5/2 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_product_l2776_277617


namespace NUMINAMATH_CALUDE_house_painting_cost_l2776_277670

/-- The cost of painting a house given its area and price per square foot -/
theorem house_painting_cost (area : ℝ) (price_per_sqft : ℝ) (h1 : area = 484) (h2 : price_per_sqft = 20) :
  area * price_per_sqft = 9680 := by
  sorry

end NUMINAMATH_CALUDE_house_painting_cost_l2776_277670


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2776_277613

/-- An arithmetic sequence with first four terms a, y, b, 3y has a/b = 0 -/
theorem arithmetic_sequence_ratio (a y b : ℝ) : 
  (∃ d : ℝ, y = a + d ∧ b = y + d ∧ 3*y = b + d) → a / b = 0 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2776_277613


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2776_277695

theorem inequality_equivalence (x : ℝ) : 
  (x / 4 ≤ 3 + 2 * x ∧ 3 + 2 * x < -3 * (1 + x)) ↔ 
  (x ≥ -12 / 7 ∧ x < -6 / 5) := by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2776_277695


namespace NUMINAMATH_CALUDE_moving_circle_trajectory_l2776_277629

/-- Two fixed circles in a 2D plane -/
structure FixedCircles where
  C₁ : (x : ℝ) → (y : ℝ) → Prop := λ x y ↦ (x + 4)^2 + y^2 = 2
  C₂ : (x : ℝ) → (y : ℝ) → Prop := λ x y ↦ (x - 4)^2 + y^2 = 2

/-- A moving circle tangent to both fixed circles -/
structure MovingCircle (fc : FixedCircles) where
  center : ℝ × ℝ
  isTangentToC₁ : Prop
  isTangentToC₂ : Prop

/-- The trajectory of the center of the moving circle -/
def trajectory (x y : ℝ) : Prop :=
  x^2 / 2 - y^2 / 14 = 1 ∨ x = 0

/-- Theorem stating that the trajectory of the moving circle's center
    is described by the given equation -/
theorem moving_circle_trajectory (fc : FixedCircles) :
  ∀ (mc : MovingCircle fc), trajectory mc.center.1 mc.center.2 :=
sorry

end NUMINAMATH_CALUDE_moving_circle_trajectory_l2776_277629


namespace NUMINAMATH_CALUDE_non_perfect_power_probability_l2776_277605

/-- A function that determines if a natural number is a perfect power --/
def isPerfectPower (n : ℕ) : Prop :=
  ∃ x y : ℕ, y > 1 ∧ n = x^y

/-- The count of numbers from 1 to 200 that are not perfect powers --/
def nonPerfectPowerCount : ℕ := 178

/-- The total count of numbers from 1 to 200 --/
def totalCount : ℕ := 200

/-- The probability of selecting a non-perfect power from 1 to 200 --/
def probabilityNonPerfectPower : ℚ := 89 / 100

theorem non_perfect_power_probability :
  (nonPerfectPowerCount : ℚ) / (totalCount : ℚ) = probabilityNonPerfectPower :=
sorry

end NUMINAMATH_CALUDE_non_perfect_power_probability_l2776_277605


namespace NUMINAMATH_CALUDE_add_fractions_l2776_277678

theorem add_fractions : (2 : ℚ) / 5 + (3 : ℚ) / 8 = (31 : ℚ) / 40 := by
  sorry

end NUMINAMATH_CALUDE_add_fractions_l2776_277678


namespace NUMINAMATH_CALUDE_correct_calculation_result_proof_correct_calculation_l2776_277693

theorem correct_calculation_result : ℤ → Prop :=
  fun x => (x + 9 = 30) → (x - 7 = 14)

-- The proof is omitted
theorem proof_correct_calculation : correct_calculation_result 21 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_result_proof_correct_calculation_l2776_277693


namespace NUMINAMATH_CALUDE_equation_solution_range_l2776_277628

theorem equation_solution_range : 
  {k : ℝ | ∃ x : ℝ, 2*k*(Real.sin x) = 1 + k^2} = {-1, 1} := by sorry

end NUMINAMATH_CALUDE_equation_solution_range_l2776_277628


namespace NUMINAMATH_CALUDE_variance_linear_transform_l2776_277600

-- Define a random variable X
variable (X : ℝ → ℝ)

-- Define the variance function D
noncomputable def D (Y : ℝ → ℝ) : ℝ := sorry

-- Theorem statement
theorem variance_linear_transform (h : D X = 2) : D (fun ω => 3 * X ω + 2) = 18 := by
  sorry

end NUMINAMATH_CALUDE_variance_linear_transform_l2776_277600


namespace NUMINAMATH_CALUDE_long_jump_distance_difference_long_jump_distance_difference_holds_l2776_277631

/-- Proves that Margarita ran and jumped 1 foot farther than Ricciana -/
theorem long_jump_distance_difference : ℕ → Prop :=
  fun margarita_total =>
    let ricciana_run := 20
    let ricciana_jump := 4
    let ricciana_total := ricciana_run + ricciana_jump
    let margarita_run := 18
    let margarita_jump := 2 * ricciana_jump - 1
    margarita_total = margarita_run + margarita_jump ∧
    margarita_total - ricciana_total = 1

/-- The theorem holds for Margarita's total distance of 25 feet -/
theorem long_jump_distance_difference_holds : long_jump_distance_difference 25 := by
  sorry

end NUMINAMATH_CALUDE_long_jump_distance_difference_long_jump_distance_difference_holds_l2776_277631


namespace NUMINAMATH_CALUDE_albany_syracuse_distance_l2776_277633

/-- The distance between Albany and Syracuse satisfies the equation relating to travel times. -/
theorem albany_syracuse_distance (D : ℝ) : D > 0 → D / 50 + D / 38.71 = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_albany_syracuse_distance_l2776_277633


namespace NUMINAMATH_CALUDE_set_intersection_equals_interval_l2776_277692

-- Define the sets M and N
def M : Set ℝ := {x | 2 * x^2 - 3 * x - 2 ≤ 0}
def N : Set ℝ := {x | x > 0 ∧ x ≠ 1}

-- Define the interval (0,1) ∪ (1,2]
def interval : Set ℝ := {x | (0 < x ∧ x < 1) ∨ (1 < x ∧ x ≤ 2)}

-- State the theorem
theorem set_intersection_equals_interval : M ∩ N = interval := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_equals_interval_l2776_277692


namespace NUMINAMATH_CALUDE_hyperbola_intersection_x_coordinate_l2776_277661

theorem hyperbola_intersection_x_coordinate :
  ∀ x y : ℝ,
  (Real.sqrt ((x - 5)^2 + y^2) - Real.sqrt ((x + 5)^2 + y^2) = 6) →
  (y = 4) →
  (x = -3 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_intersection_x_coordinate_l2776_277661


namespace NUMINAMATH_CALUDE_triangle_construction_l2776_277624

/-- Given a point A, a plane S, and distances ρ, ρₐ, and b-c,
    we can construct a triangle ABC with specific properties. -/
theorem triangle_construction (A : ℝ × ℝ) (S : ℝ × ℝ) (ρ ρₐ : ℝ) (b_minus_c : ℝ) 
  (s a b c : ℝ) :
  -- Side a lies in plane S (represented by the condition that a is real)
  -- One vertex is A (implicit in the construction)
  -- ρ is the inradius
  -- ρₐ is the exradius opposite to side a
  (s = (a + b + c) / 2) →  -- Definition of semiperimeter
  (ρ > 0) →  -- Inradius is positive
  (ρₐ > 0) →  -- Exradius is positive
  (a > 0) ∧ (b > 0) ∧ (c > 0) →  -- Triangle sides are positive
  (b - c = b_minus_c) →  -- Given difference of sides
  -- Then the following relationships hold:
  ((s - b) * (s - c) = ρ * ρₐ) ∧
  ((s - c) - (s - b) = b - c) ∧
  (Real.sqrt ((s - b) * (s - c)) = Real.sqrt (ρ * ρₐ)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_construction_l2776_277624


namespace NUMINAMATH_CALUDE_max_sum_diff_unit_vectors_l2776_277663

theorem max_sum_diff_unit_vectors (a b : EuclideanSpace ℝ (Fin 2)) :
  ‖a‖ = 1 → ‖b‖ = 1 → ‖a + b‖ + ‖a - b‖ ≤ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_diff_unit_vectors_l2776_277663
