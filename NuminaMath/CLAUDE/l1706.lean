import Mathlib

namespace NUMINAMATH_CALUDE_average_running_time_l1706_170616

theorem average_running_time (total_students : ℕ) 
  (sixth_grade_time seventh_grade_time eighth_grade_time : ℕ)
  (sixth_to_seventh_ratio seventh_to_eighth_ratio : ℕ) :
  total_students = 210 →
  sixth_grade_time = 10 →
  seventh_grade_time = 12 →
  eighth_grade_time = 14 →
  sixth_to_seventh_ratio = 3 →
  seventh_to_eighth_ratio = 4 →
  (let eighth_grade_count := total_students / (1 + seventh_to_eighth_ratio + sixth_to_seventh_ratio * seventh_to_eighth_ratio);
   let seventh_grade_count := seventh_to_eighth_ratio * eighth_grade_count;
   let sixth_grade_count := sixth_to_seventh_ratio * seventh_grade_count;
   let total_minutes := sixth_grade_count * sixth_grade_time + 
                        seventh_grade_count * seventh_grade_time + 
                        eighth_grade_count * eighth_grade_time;
   (total_minutes : ℚ) / total_students = 420 / 39) :=
by
  sorry

end NUMINAMATH_CALUDE_average_running_time_l1706_170616


namespace NUMINAMATH_CALUDE_money_division_l1706_170688

theorem money_division (total : ℝ) (p q r : ℝ) : 
  p + q + r = total →
  p / q = 3 / 7 →
  q / r = 7 / 12 →
  q - p = 3600 →
  r - q = 4500 := by
sorry

end NUMINAMATH_CALUDE_money_division_l1706_170688


namespace NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_l1706_170623

theorem min_value_sum_of_reciprocals (x y z : ℝ) 
  (hpos : x > 0 ∧ y > 0 ∧ z > 0) (hsum : x + y + z = 3) :
  (1 / (x + 3*y) + 1 / (y + 3*z) + 1 / (z + 3*x)) ≥ 3/4 :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_l1706_170623


namespace NUMINAMATH_CALUDE_richard_twice_scott_age_david_age_four_years_ago_future_years_correct_l1706_170689

/-- The number of years in the future when Richard will be twice as old as Scott -/
def future_years : ℕ := 8

/-- David's current age -/
def david_age : ℕ := 14

/-- Richard's current age -/
def richard_age : ℕ := david_age + 6

/-- Scott's current age -/
def scott_age : ℕ := david_age - 8

theorem richard_twice_scott_age : 
  richard_age + future_years = 2 * (scott_age + future_years) :=
by sorry

theorem david_age_four_years_ago : 
  david_age = 10 + 4 :=
by sorry

theorem future_years_correct : 
  ∃ (y : ℕ), y = future_years ∧ richard_age + y = 2 * (scott_age + y) :=
by sorry

end NUMINAMATH_CALUDE_richard_twice_scott_age_david_age_four_years_ago_future_years_correct_l1706_170689


namespace NUMINAMATH_CALUDE_sixteen_percent_of_550_is_88_l1706_170617

theorem sixteen_percent_of_550_is_88 : 
  (16 : ℚ) / 100 * 550 = 88 := by
  sorry

end NUMINAMATH_CALUDE_sixteen_percent_of_550_is_88_l1706_170617


namespace NUMINAMATH_CALUDE_sam_eats_280_apples_in_week_l1706_170659

/-- Calculates the number of apples Sam eats in a week -/
def apples_eaten_in_week (apples_per_sandwich : ℕ) (sandwiches_per_day : ℕ) (days_in_week : ℕ) : ℕ :=
  apples_per_sandwich * sandwiches_per_day * days_in_week

/-- Proves that Sam eats 280 apples in a week -/
theorem sam_eats_280_apples_in_week :
  apples_eaten_in_week 4 10 7 = 280 := by
  sorry

end NUMINAMATH_CALUDE_sam_eats_280_apples_in_week_l1706_170659


namespace NUMINAMATH_CALUDE_planted_fraction_specific_case_l1706_170601

/-- Represents a right triangle field with an unplanted area -/
structure FieldWithUnplantedArea where
  /-- Length of the first leg of the right triangle field -/
  leg1 : ℝ
  /-- Length of the second leg of the right triangle field -/
  leg2 : ℝ
  /-- Shortest distance from the base of the unplanted triangle to the hypotenuse -/
  unplanted_distance : ℝ

/-- Calculates the fraction of the planted area in the field -/
def planted_fraction (field : FieldWithUnplantedArea) : ℝ :=
  -- Implementation details omitted
  sorry

theorem planted_fraction_specific_case :
  let field := FieldWithUnplantedArea.mk 5 12 3
  planted_fraction field = 2665 / 2890 := by
  sorry

end NUMINAMATH_CALUDE_planted_fraction_specific_case_l1706_170601


namespace NUMINAMATH_CALUDE_solve_equation_l1706_170682

theorem solve_equation : ∃ x : ℚ, (2 * x + 3 * x = 500 - (4 * x + 5 * x - 20)) ∧ x = 520 / 14 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1706_170682


namespace NUMINAMATH_CALUDE_bus_speed_problem_l1706_170673

theorem bus_speed_problem (distance : ℝ) (speed_ratio : ℝ) (time_difference : ℝ) :
  distance = 72 →
  speed_ratio = 1.2 →
  time_difference = 1/5 →
  ∀ (speed_large : ℝ),
    (distance / speed_large - distance / (speed_ratio * speed_large) = time_difference) →
    speed_large = 60 := by
  sorry

end NUMINAMATH_CALUDE_bus_speed_problem_l1706_170673


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1706_170626

/-- An isosceles triangle with sides of length 2 and 5 has a perimeter of 12 -/
theorem isosceles_triangle_perimeter (a b c : ℝ) : 
  a = 5 → b = 5 → c = 2 → a + b + c = 12 := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1706_170626


namespace NUMINAMATH_CALUDE_total_rope_length_l1706_170618

/-- The original length of each rope -/
def rope_length : ℝ := 52

/-- The length used from the first rope -/
def used_first : ℝ := 42

/-- The length used from the second rope -/
def used_second : ℝ := 12

theorem total_rope_length :
  (rope_length - used_first) * 4 = rope_length - used_second →
  2 * rope_length = 104 := by
  sorry

end NUMINAMATH_CALUDE_total_rope_length_l1706_170618


namespace NUMINAMATH_CALUDE_monotone_quadratic_function_m_range_l1706_170672

/-- A function f is monotonically increasing on an interval (a, b) if for any x, y in (a, b) with x < y, we have f(x) < f(y) -/
def MonotonicallyIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

/-- The function f(x) = mx^2 + x - 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + x - 1

theorem monotone_quadratic_function_m_range :
  (∀ m : ℝ, MonotonicallyIncreasing (f m) (-1) Real.pi) ↔ 
  (∀ m : ℝ, 0 ≤ m ∧ m ≤ 1/2) :=
sorry

end NUMINAMATH_CALUDE_monotone_quadratic_function_m_range_l1706_170672


namespace NUMINAMATH_CALUDE_equation_proof_l1706_170680

theorem equation_proof (a b : ℚ) (h : 3 * a = 2 * b) : (a + b) / b = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l1706_170680


namespace NUMINAMATH_CALUDE_two_true_statements_l1706_170604

theorem two_true_statements : 
  let original := ∀ a : ℝ, a > -5 → a > -8
  let converse := ∀ a : ℝ, a > -8 → a > -5
  let inverse := ∀ a : ℝ, a ≤ -5 → a ≤ -8
  let contrapositive := ∀ a : ℝ, a ≤ -8 → a ≤ -5
  (original ∧ ¬converse ∧ ¬inverse ∧ contrapositive) :=
by
  sorry

end NUMINAMATH_CALUDE_two_true_statements_l1706_170604


namespace NUMINAMATH_CALUDE_purple_car_count_l1706_170640

theorem purple_car_count (total : ℕ) (purple blue red orange yellow green : ℕ)
  (h_total : total = 987)
  (h_blue : blue = 2 * red)
  (h_red : red = 3 * orange)
  (h_yellow1 : yellow = orange / 2)
  (h_yellow2 : yellow = 3 * purple)
  (h_green : green = 5 * purple)
  (h_sum : purple + yellow + orange + red + blue + green = total) :
  purple = 14 := by
  sorry

end NUMINAMATH_CALUDE_purple_car_count_l1706_170640


namespace NUMINAMATH_CALUDE_negate_difference_l1706_170696

theorem negate_difference (a b : ℝ) : -(a - b) = -a + b := by
  sorry

end NUMINAMATH_CALUDE_negate_difference_l1706_170696


namespace NUMINAMATH_CALUDE_dandan_age_problem_l1706_170625

theorem dandan_age_problem (dandan_age : ℕ) (father_age : ℕ) (a : ℕ) :
  dandan_age = 4 →
  father_age = 28 →
  father_age + a = 3 * (dandan_age + a) →
  a = 8 :=
by sorry

end NUMINAMATH_CALUDE_dandan_age_problem_l1706_170625


namespace NUMINAMATH_CALUDE_equation1_solutions_equation2_solutions_l1706_170631

-- Define the quadratic equations
def equation1 (x : ℝ) : Prop := x^2 - 4*x + 1 = 0
def equation2 (x : ℝ) : Prop := 2*x^2 - 3*x + 1 = 0

-- Theorem for equation 1
theorem equation1_solutions :
  ∃ x₁ x₂ : ℝ, x₁ = 2 + Real.sqrt 3 ∧ x₂ = 2 - Real.sqrt 3 ∧
  equation1 x₁ ∧ equation1 x₂ ∧
  ∀ x : ℝ, equation1 x → x = x₁ ∨ x = x₂ :=
sorry

-- Theorem for equation 2
theorem equation2_solutions :
  ∃ x₁ x₂ : ℝ, x₁ = 1 ∧ x₂ = 1/2 ∧
  equation2 x₁ ∧ equation2 x₂ ∧
  ∀ x : ℝ, equation2 x → x = x₁ ∨ x = x₂ :=
sorry

end NUMINAMATH_CALUDE_equation1_solutions_equation2_solutions_l1706_170631


namespace NUMINAMATH_CALUDE_razorback_tshirt_revenue_l1706_170671

/-- The total money made by selling a given number of t-shirts at a fixed price -/
def total_money_made (num_shirts : ℕ) (price_per_shirt : ℕ) : ℕ :=
  num_shirts * price_per_shirt

/-- Theorem stating that selling 45 t-shirts at $16 each results in $720 total -/
theorem razorback_tshirt_revenue : total_money_made 45 16 = 720 := by
  sorry

end NUMINAMATH_CALUDE_razorback_tshirt_revenue_l1706_170671


namespace NUMINAMATH_CALUDE_circle_condition_l1706_170662

theorem circle_condition (m : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 2*m*x - 2*m*y + 2*m^2 + m - 1 = 0 → 
    ∃ r : ℝ, r > 0 ∧ ∃ a b : ℝ, (x - a)^2 + (y - b)^2 = r^2) → 
  m < 1 := by
sorry

end NUMINAMATH_CALUDE_circle_condition_l1706_170662


namespace NUMINAMATH_CALUDE_safety_rent_a_truck_cost_per_mile_l1706_170669

/-- The cost per mile for Safety Rent A Truck -/
def safety_cost_per_mile : ℝ := sorry

/-- The base cost for Safety Rent A Truck -/
def safety_base_cost : ℝ := 41.95

/-- The base cost for City Rentals -/
def city_base_cost : ℝ := 38.95

/-- The cost per mile for City Rentals -/
def city_cost_per_mile : ℝ := 0.31

/-- The number of miles for which the total costs are equal -/
def equal_cost_miles : ℝ := 150.0

theorem safety_rent_a_truck_cost_per_mile :
  safety_base_cost + equal_cost_miles * safety_cost_per_mile =
  city_base_cost + equal_cost_miles * city_cost_per_mile ∧
  safety_cost_per_mile = 0.29 := by sorry

end NUMINAMATH_CALUDE_safety_rent_a_truck_cost_per_mile_l1706_170669


namespace NUMINAMATH_CALUDE_equation_solution_l1706_170650

theorem equation_solution : 
  ∃ x : ℚ, (2 * x + 1) / 3 - (x - 1) / 6 = 2 ∧ x = 3 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l1706_170650


namespace NUMINAMATH_CALUDE_jacket_cost_ratio_l1706_170646

theorem jacket_cost_ratio (marked_price : ℝ) (h1 : marked_price > 0) : 
  let discount_ratio : ℝ := 1/4
  let selling_price : ℝ := marked_price * (1 - discount_ratio)
  let cost_ratio : ℝ := 2/3
  let cost : ℝ := selling_price * cost_ratio
  cost / marked_price = 1/2 := by
sorry

end NUMINAMATH_CALUDE_jacket_cost_ratio_l1706_170646


namespace NUMINAMATH_CALUDE_derivative_zero_at_negative_one_l1706_170611

theorem derivative_zero_at_negative_one (t : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ (x^2 - 4) * (x - t)
  (deriv f) (-1) = 0 → t = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_zero_at_negative_one_l1706_170611


namespace NUMINAMATH_CALUDE_remainder_371073_div_6_l1706_170697

theorem remainder_371073_div_6 : 371073 % 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_371073_div_6_l1706_170697


namespace NUMINAMATH_CALUDE_joan_remaining_apples_l1706_170649

/-- Given that Joan picked 43 apples and gave 27 to Melanie, prove that she now has 16 apples. -/
theorem joan_remaining_apples : 
  ∀ (initial_apples given_apples remaining_apples : ℕ), 
    initial_apples = 43 → 
    given_apples = 27 → 
    remaining_apples = initial_apples - given_apples → 
    remaining_apples = 16 := by
  sorry

end NUMINAMATH_CALUDE_joan_remaining_apples_l1706_170649


namespace NUMINAMATH_CALUDE_circle_and_line_properties_l1706_170655

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 8 = 0

-- Define the line l
def line_l (k x y : ℝ) : Prop := y = k*(x + 1) + 1

-- Theorem statement
theorem circle_and_line_properties :
  -- 1. The center of circle C is (1, 0)
  (∃ r : ℝ, ∀ x y : ℝ, circle_C x y ↔ (x - 1)^2 + y^2 = r^2) ∧
  -- 2. The point (-1, 1) lies on line l for any real k
  (∀ k : ℝ, line_l k (-1) 1) ∧
  -- 3. Line l intersects circle C for any real k
  (∀ k : ℝ, ∃ x y : ℝ, circle_C x y ∧ line_l k x y) :=
by sorry

end NUMINAMATH_CALUDE_circle_and_line_properties_l1706_170655


namespace NUMINAMATH_CALUDE_no_triangle_with_special_sides_l1706_170684

-- Define a triangle structure
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : a > 0 ∧ b > 0 ∧ c > 0
  h_inequality : a + b > c ∧ b + c > a ∧ c + a > b

-- Define functions for altitude, angle bisector, and median
def altitude (t : Triangle) : ℝ := sorry
def angleBisector (t : Triangle) : ℝ := sorry
def median (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem no_triangle_with_special_sides :
  ¬ ∃ (t : Triangle),
    (t.a = altitude t ∧ t.b = angleBisector t ∧ t.c = median t) ∨
    (t.a = altitude t ∧ t.b = median t ∧ t.c = angleBisector t) ∨
    (t.a = angleBisector t ∧ t.b = altitude t ∧ t.c = median t) ∨
    (t.a = angleBisector t ∧ t.b = median t ∧ t.c = altitude t) ∨
    (t.a = median t ∧ t.b = altitude t ∧ t.c = angleBisector t) ∨
    (t.a = median t ∧ t.b = angleBisector t ∧ t.c = altitude t) := by
  sorry

end NUMINAMATH_CALUDE_no_triangle_with_special_sides_l1706_170684


namespace NUMINAMATH_CALUDE_main_theorem_l1706_170608

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (x * y) = y * f x + x * f y

/-- The main theorem capturing the problem statements -/
theorem main_theorem (f : ℝ → ℝ) (hf : FunctionalEquation f)
    (a b c d : ℝ) (F : ℝ → ℝ) (hF : ∀ x, F x = a * f x + b * x^5 + c * x^3 + 2 * x^2 + d * x + 3)
    (hF_neg5 : F (-5) = 7) :
    f 0 = 0 ∧ (∀ x, f (-x) = -f x) ∧ F 5 = 99 := by
  sorry


end NUMINAMATH_CALUDE_main_theorem_l1706_170608


namespace NUMINAMATH_CALUDE_kelly_time_indeterminate_but_longest_l1706_170609

/-- Represents the breath-holding contest results -/
structure BreathHoldingContest where
  kelly_time : ℝ
  brittany_time : ℝ
  buffy_time : ℝ
  brittany_kelly_diff : kelly_time - brittany_time = 20
  buffy_time_exact : buffy_time = 120

/-- Kelly's time is indeterminate but greater than Buffy's if she won -/
theorem kelly_time_indeterminate_but_longest (contest : BreathHoldingContest) :
  (∀ t : ℝ, contest.kelly_time ≠ t) ∧
  (contest.kelly_time > contest.buffy_time) :=
sorry

end NUMINAMATH_CALUDE_kelly_time_indeterminate_but_longest_l1706_170609


namespace NUMINAMATH_CALUDE_simplify_expression_l1706_170660

theorem simplify_expression (x : ℝ) : (3*x - 10) + (7*x + 20) - (2*x - 5) = 8*x + 15 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1706_170660


namespace NUMINAMATH_CALUDE_calculation_proof_l1706_170693

theorem calculation_proof : (-2)^3 - |2 - 5| / (-3) = -7 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1706_170693


namespace NUMINAMATH_CALUDE_divisible_by_eight_probability_l1706_170651

theorem divisible_by_eight_probability (n : ℕ) : 
  (Finset.filter (λ k => (k * (k + 1)) % 8 = 0) (Finset.range 100)).card / 100 = 1 / 8 := by
sorry

end NUMINAMATH_CALUDE_divisible_by_eight_probability_l1706_170651


namespace NUMINAMATH_CALUDE_width_to_perimeter_ratio_l1706_170694

/-- The ratio of width to perimeter for a rectangular room -/
theorem width_to_perimeter_ratio (length width : ℝ) (h1 : length = 15) (h2 : width = 13) :
  width / (2 * (length + width)) = 13 / 56 := by
  sorry

end NUMINAMATH_CALUDE_width_to_perimeter_ratio_l1706_170694


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l1706_170687

theorem complex_number_in_first_quadrant : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (Complex.I / (1 + Complex.I) = ↑a + ↑b * Complex.I) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l1706_170687


namespace NUMINAMATH_CALUDE_common_number_in_overlapping_lists_l1706_170612

theorem common_number_in_overlapping_lists (nums : List ℝ) : 
  nums.length = 9 ∧ 
  (nums.take 5).sum / 5 = 7 ∧ 
  (nums.drop 4).sum / 5 = 9 ∧ 
  nums.sum / 9 = 73 / 9 →
  ∃ x ∈ nums.take 5 ∩ nums.drop 4, x = 7 :=
by sorry

end NUMINAMATH_CALUDE_common_number_in_overlapping_lists_l1706_170612


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_condition_l1706_170637

/-- The probability that no two of three independently chosen real numbers 
    from [0, n] are within 2 units of each other is greater than 1/2 -/
def probability_condition (n : ℕ) : Prop :=
  (n - 4)^3 / n^3 > 1/2

/-- 12 is the smallest positive integer satisfying the probability condition -/
theorem smallest_n_satisfying_condition : 
  (∀ k < 12, ¬ probability_condition k) ∧ probability_condition 12 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_condition_l1706_170637


namespace NUMINAMATH_CALUDE_circle_symmetry_l1706_170602

/-- Given a circle symmetrical to circle C with respect to the line x-y+1=0,
    prove that the equation of circle C is x^2 + (y-2)^2 = 1 -/
theorem circle_symmetry (x y : ℝ) :
  ((x - 1)^2 + (y - 1)^2 = 1) →  -- Equation of the symmetrical circle
  (x - y + 1 = 0 →               -- Equation of the line of symmetry
   (∃ (a b : ℝ), (x - a)^2 + (y - b)^2 = 1 ∧  -- Existence of circle C
    (a^2 + (b - 2)^2 = 1)))      -- Equation of circle C
:= by sorry

end NUMINAMATH_CALUDE_circle_symmetry_l1706_170602


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l1706_170679

def P : Set ℕ := {x : ℕ | x * (x - 3) ≤ 0}
def Q : Set ℕ := {x : ℕ | x ≥ 2}

theorem intersection_of_P_and_Q : P ∩ Q = {2, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l1706_170679


namespace NUMINAMATH_CALUDE_susan_age_l1706_170607

theorem susan_age (susan arthur tom bob : ℕ) 
  (h1 : arthur = susan + 2)
  (h2 : tom = bob - 3)
  (h3 : bob = 11)
  (h4 : susan + arthur + tom + bob = 51) :
  susan = 15 := by
sorry

end NUMINAMATH_CALUDE_susan_age_l1706_170607


namespace NUMINAMATH_CALUDE_additional_bottles_needed_l1706_170676

/-- Represents the number of bottles in a case of water -/
def bottles_per_case : ℕ := 24

/-- Represents the number of cases purchased -/
def cases_purchased : ℕ := 13

/-- Represents the duration of the camp in days -/
def camp_duration : ℕ := 3

/-- Represents the number of children in the first group -/
def group1_children : ℕ := 14

/-- Represents the number of children in the second group -/
def group2_children : ℕ := 16

/-- Represents the number of children in the third group -/
def group3_children : ℕ := 12

/-- Represents the number of bottles consumed by each child per day -/
def bottles_per_child_per_day : ℕ := 3

/-- Calculates the total number of children in the camp -/
def total_children : ℕ :=
  let first_three := group1_children + group2_children + group3_children
  first_three + first_three / 2

/-- Calculates the total number of bottles needed for the entire camp -/
def total_bottles_needed : ℕ :=
  total_children * bottles_per_child_per_day * camp_duration

/-- Calculates the number of bottles already purchased -/
def bottles_purchased : ℕ :=
  cases_purchased * bottles_per_case

/-- Theorem stating that 255 additional bottles are needed -/
theorem additional_bottles_needed : 
  total_bottles_needed - bottles_purchased = 255 := by
  sorry

end NUMINAMATH_CALUDE_additional_bottles_needed_l1706_170676


namespace NUMINAMATH_CALUDE_x_range_theorem_l1706_170610

-- Define the propositions p and q
def p (x : ℝ) : Prop := Real.log (x^2 - 2*x - 2) ≥ 0
def q (x : ℝ) : Prop := 0 < x ∧ x < 4

-- Define the range of x
def range_of_x (x : ℝ) : Prop := x ≤ -1 ∨ (0 < x ∧ x < 3) ∨ x ≥ 4

-- Theorem statement
theorem x_range_theorem (x : ℝ) : 
  (¬(p x) ∧ ¬(q x)) ∧ (p x ∨ q x) → range_of_x x :=
by sorry

end NUMINAMATH_CALUDE_x_range_theorem_l1706_170610


namespace NUMINAMATH_CALUDE_area_square_with_semicircles_l1706_170639

/-- The area of a shape formed by a square with semicircles on each side -/
theorem area_square_with_semicircles (π : ℝ) : 
  let square_side : ℝ := 2 * π
  let square_area : ℝ := square_side ^ 2
  let semicircle_radius : ℝ := square_side / 2
  let semicircle_area : ℝ := 1 / 2 * π * semicircle_radius ^ 2
  let total_semicircle_area : ℝ := 4 * semicircle_area
  let total_area : ℝ := square_area + total_semicircle_area
  total_area = 2 * π^2 * (π + 2) :=
by sorry

end NUMINAMATH_CALUDE_area_square_with_semicircles_l1706_170639


namespace NUMINAMATH_CALUDE_intersection_is_ellipse_l1706_170665

-- Define the plane
def plane (z : ℝ) : Prop := z = 2

-- Define the ellipsoid
def ellipsoid (x y z : ℝ) : Prop := x^2/12 + y^2/4 + z^2/16 = 1

-- Define the intersection curve
def intersection_curve (x y : ℝ) : Prop := x^2/9 + y^2/3 = 1

-- Theorem statement
theorem intersection_is_ellipse :
  ∀ x y z : ℝ,
  plane z ∧ ellipsoid x y z →
  intersection_curve x y ∧
  ∃ a b : ℝ, a = 3 ∧ b = Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_intersection_is_ellipse_l1706_170665


namespace NUMINAMATH_CALUDE_joans_marbles_l1706_170627

/-- Given that Mary has 9 yellow marbles and the total number of yellow marbles
    between Mary and Joan is 12, prove that Joan has 3 yellow marbles. -/
theorem joans_marbles (mary_marbles : ℕ) (total_marbles : ℕ) (joan_marbles : ℕ) 
    (h1 : mary_marbles = 9)
    (h2 : total_marbles = 12)
    (h3 : mary_marbles + joan_marbles = total_marbles) :
  joan_marbles = 3 := by
  sorry

end NUMINAMATH_CALUDE_joans_marbles_l1706_170627


namespace NUMINAMATH_CALUDE_right_triangle_seven_units_contains_28_triangles_l1706_170622

/-- Represents a right-angled triangle on a grid -/
structure GridTriangle where
  leg_length : ℕ
  is_right_angled : Bool

/-- Calculates the maximum number of triangles that can be formed within a GridTriangle -/
def max_triangles (t : GridTriangle) : ℕ :=
  if t.is_right_angled && t.leg_length > 0 then
    (t.leg_length + 1).choose 2
  else
    0

/-- Theorem stating that a right-angled triangle with legs of 7 units on a grid contains 28 triangles -/
theorem right_triangle_seven_units_contains_28_triangles :
  let t : GridTriangle := { leg_length := 7, is_right_angled := true }
  max_triangles t = 28 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_seven_units_contains_28_triangles_l1706_170622


namespace NUMINAMATH_CALUDE_all_propositions_incorrect_l1706_170619

/-- Represents a proposition with potential flaws in statistical reasoning --/
structure Proposition where
  hasTemporalityIgnorance : Bool
  hasSpeciesCharacteristicsIgnorance : Bool
  hasCausalityMisinterpretation : Bool
  hasIncorrectUsageRange : Bool

/-- Determines if a proposition is incorrect based on its flaws --/
def isIncorrect (p : Proposition) : Bool :=
  p.hasTemporalityIgnorance ∨ 
  p.hasSpeciesCharacteristicsIgnorance ∨ 
  p.hasCausalityMisinterpretation ∨ 
  p.hasIncorrectUsageRange

/-- Counts the number of incorrect propositions in a list --/
def countIncorrectPropositions (props : List Proposition) : Nat :=
  props.filter isIncorrect |>.length

/-- The main theorem stating that all given propositions are incorrect --/
theorem all_propositions_incorrect (props : List Proposition) 
  (h1 : props.length = 4)
  (h2 : ∀ p ∈ props, isIncorrect p = true) : 
  countIncorrectPropositions props = 4 := by
  sorry

#check all_propositions_incorrect

end NUMINAMATH_CALUDE_all_propositions_incorrect_l1706_170619


namespace NUMINAMATH_CALUDE_fish_population_estimate_l1706_170670

/-- Estimates the number of fish in a lake using the capture-recapture technique -/
theorem fish_population_estimate (tagged_april : ℕ) (captured_august : ℕ) (tagged_recaptured : ℕ)
  (tagged_survival_rate : ℝ) (original_fish_rate : ℝ) :
  tagged_april = 100 →
  captured_august = 100 →
  tagged_recaptured = 5 →
  tagged_survival_rate = 0.7 →
  original_fish_rate = 0.8 →
  ∃ (estimated_population : ℕ), estimated_population = 1120 :=
by sorry

end NUMINAMATH_CALUDE_fish_population_estimate_l1706_170670


namespace NUMINAMATH_CALUDE_workshop_assignment_l1706_170633

theorem workshop_assignment (total_workers : ℕ) 
  (type_a_rate type_b_rate : ℕ) (ratio_a ratio_b : ℕ) 
  (type_a_workers : ℕ) (type_b_workers : ℕ) : 
  total_workers = 90 →
  type_a_rate = 15 →
  type_b_rate = 8 →
  ratio_a = 3 →
  ratio_b = 2 →
  type_a_workers = 40 →
  type_b_workers = 50 →
  total_workers = type_a_workers + type_b_workers →
  ratio_a * (type_b_rate * type_b_workers) = ratio_b * (type_a_rate * type_a_workers) := by
  sorry

#check workshop_assignment

end NUMINAMATH_CALUDE_workshop_assignment_l1706_170633


namespace NUMINAMATH_CALUDE_sqrt_2450_minus_2_theorem_l1706_170686

theorem sqrt_2450_minus_2_theorem (a b : ℕ+) :
  (Real.sqrt 2450 - 2 : ℝ) = ((Real.sqrt a.val : ℝ) - b.val)^2 →
  a.val + b.val = 2451 := by
sorry

end NUMINAMATH_CALUDE_sqrt_2450_minus_2_theorem_l1706_170686


namespace NUMINAMATH_CALUDE_triangle_inequality_l1706_170658

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  Real.sqrt (a + b - c) + Real.sqrt (c + a - b) + Real.sqrt (b + c - a) ≤ Real.sqrt a + Real.sqrt b + Real.sqrt c ∧
  (Real.sqrt (a + b - c) + Real.sqrt (c + a - b) + Real.sqrt (b + c - a) = Real.sqrt a + Real.sqrt b + Real.sqrt c ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1706_170658


namespace NUMINAMATH_CALUDE_sum_real_imag_parts_of_z_l1706_170683

theorem sum_real_imag_parts_of_z (z : ℂ) (h : z * (2 + Complex.I) = 2 * Complex.I - 1) :
  z.re + z.im = 1 := by sorry

end NUMINAMATH_CALUDE_sum_real_imag_parts_of_z_l1706_170683


namespace NUMINAMATH_CALUDE_fraction_equality_l1706_170600

theorem fraction_equality : (18 : ℚ) / (5 * 107 + 3) = 18 / 538 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l1706_170600


namespace NUMINAMATH_CALUDE_triangle_perimeter_l1706_170666

-- Define the triangle
def triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Define the equation for the third side
def third_side_equation (x : ℝ) : Prop :=
  x^2 - 8*x + 12 = 0

-- Theorem statement
theorem triangle_perimeter : 
  ∃ (x : ℝ), 
    third_side_equation x ∧ 
    triangle 4 7 x ∧ 
    4 + 7 + x = 17 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l1706_170666


namespace NUMINAMATH_CALUDE_nancy_carrots_l1706_170685

/-- The number of carrots Nancy threw out -/
def carrots_thrown_out : ℕ := 2

/-- The number of carrots Nancy initially picked -/
def initial_carrots : ℕ := 12

/-- The number of carrots Nancy picked the next day -/
def next_day_carrots : ℕ := 21

/-- The total number of carrots Nancy ended up with -/
def total_carrots : ℕ := 31

theorem nancy_carrots :
  initial_carrots - carrots_thrown_out + next_day_carrots = total_carrots :=
by sorry

end NUMINAMATH_CALUDE_nancy_carrots_l1706_170685


namespace NUMINAMATH_CALUDE_solution_satisfies_system_l1706_170628

theorem solution_satisfies_system :
  ∃ (x y z w : ℝ), 
    (x = 2 ∧ y = 2 ∧ z = 0 ∧ w = 0) ∧
    (x + y + Real.sqrt z = 4) ∧
    (Real.sqrt x * Real.sqrt y - Real.sqrt w = 2) :=
by sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_l1706_170628


namespace NUMINAMATH_CALUDE_least_common_multiple_first_ten_l1706_170624

theorem least_common_multiple_first_ten : ∃ n : ℕ, 
  (∀ k : ℕ, k ≤ 10 → k > 0 → n % k = 0) ∧ 
  (∀ m : ℕ, m < n → ∃ k : ℕ, k ≤ 10 ∧ k > 0 ∧ m % k ≠ 0) ∧
  n = 2520 := by
sorry

end NUMINAMATH_CALUDE_least_common_multiple_first_ten_l1706_170624


namespace NUMINAMATH_CALUDE_range_of_a_l1706_170645

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - 1| < 1 → x ≥ a) ∧ 
  (∃ y : ℝ, y ≥ a ∧ ¬(|y - 1| < 1)) → 
  a ≤ 0 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l1706_170645


namespace NUMINAMATH_CALUDE_parabola_intersection_ratio_l1706_170695

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 4x -/
def Parabola := {p : Point | p.y^2 = 4 * p.x}

/-- The focus of the parabola -/
def focus : Point := ⟨1, 0⟩

/-- The fixed point A -/
def A : Point := ⟨0, -2⟩

/-- Point M on the parabola -/
def M : Point := sorry

/-- Point N on the directrix -/
def N : Point := sorry

/-- Distance between two points -/
def distance (p q : Point) : ℝ := sorry

/-- Theorem: The ratio |MN| : |FN| = √5 : (1 + √5) -/
theorem parabola_intersection_ratio :
  (distance M N) / (distance focus N) = Real.sqrt 5 / (1 + Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_ratio_l1706_170695


namespace NUMINAMATH_CALUDE_arrangements_starting_with_vowel_l1706_170657

def word : String := "basics"

def is_vowel (c : Char) : Bool :=
  c = 'a' || c = 'e' || c = 'i' || c = 'o' || c = 'u'

def count_vowels (s : String) : Nat :=
  s.toList.filter is_vowel |>.length

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def permutations_with_repetition (total : Nat) (repeated : List Nat) : Nat :=
  factorial total / (repeated.map factorial).prod

theorem arrangements_starting_with_vowel :
  let total_letters := word.length
  let vowels := count_vowels word
  let consonants := total_letters - vowels
  let arrangements := 
    vowels * permutations_with_repetition (total_letters - 1) [consonants, vowels - 1, 1]
  arrangements = 120 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_starting_with_vowel_l1706_170657


namespace NUMINAMATH_CALUDE_factor_of_polynomial_l1706_170629

theorem factor_of_polynomial (x : ℝ) : 
  ∃ (y : ℝ), x^4 + 16 = (x^2 - 4*x + 4) * y :=
sorry

end NUMINAMATH_CALUDE_factor_of_polynomial_l1706_170629


namespace NUMINAMATH_CALUDE_parallel_transitivity_l1706_170636

-- Define a type for lines in a plane
def Line : Type := ℝ × ℝ → Prop

-- Define parallel relationship between lines
def parallel (l1 l2 : Line) : Prop := sorry

-- State the theorem
theorem parallel_transitivity (a b c : Line) :
  parallel a b → parallel b c → parallel a c := by sorry

end NUMINAMATH_CALUDE_parallel_transitivity_l1706_170636


namespace NUMINAMATH_CALUDE_new_members_average_weight_l1706_170667

theorem new_members_average_weight 
  (initial_count : ℕ) 
  (initial_average : ℝ) 
  (new_count : ℕ) 
  (new_average : ℝ) 
  (double_counted_weight : ℝ) :
  initial_count = 10 →
  initial_average = 75 →
  new_count = 3 →
  new_average = 77 →
  double_counted_weight = 65 →
  let corrected_total := initial_count * initial_average - double_counted_weight
  let new_total := (initial_count + new_count - 1) * new_average
  let new_members_total := new_total - corrected_total
  (new_members_total / new_count) = 79.67 := by
sorry

end NUMINAMATH_CALUDE_new_members_average_weight_l1706_170667


namespace NUMINAMATH_CALUDE_circus_ticket_sales_l1706_170674

/-- Calculates the total number of tickets sold at a circus given the prices, revenue, and number of lower seat tickets sold. -/
def total_tickets (lower_price upper_price : ℕ) (total_revenue : ℕ) (lower_tickets : ℕ) : ℕ :=
  lower_tickets + (total_revenue - lower_price * lower_tickets) / upper_price

/-- Theorem stating that given the specific conditions of the circus problem, the total number of tickets sold is 80. -/
theorem circus_ticket_sales :
  total_tickets 30 20 2100 50 = 80 := by
  sorry

end NUMINAMATH_CALUDE_circus_ticket_sales_l1706_170674


namespace NUMINAMATH_CALUDE_polynomial_equality_sum_l1706_170613

theorem polynomial_equality_sum (a b c d : ℤ) : 
  (∀ x : ℝ, (x^2 + a*x + b) * (x^2 + c*x + d) = x^4 + 2*x^3 - x^2 + 18*x + 24) →
  a + b + c + d = 12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_sum_l1706_170613


namespace NUMINAMATH_CALUDE_fifteen_segments_two_monochromatic_triangles_fourteen_segments_no_monochromatic_triangle_possible_l1706_170647

/-- Represents a segment (edge or diagonal) in a regular hexagon --/
inductive Segment
| Edge : Fin 6 → Fin 6 → Segment
| Diagonal : Fin 6 → Fin 6 → Segment

/-- Represents the color of a segment --/
inductive Color
| Red
| Blue

/-- Represents a coloring of segments in a regular hexagon --/
def Coloring := Segment → Option Color

/-- Checks if a triangle is monochromatic --/
def isMonochromatic (c : Coloring) (v1 v2 v3 : Fin 6) : Bool :=
  sorry

/-- Counts the number of colored segments in a coloring --/
def countColoredSegments (c : Coloring) : Nat :=
  sorry

/-- Counts the number of monochromatic triangles in a coloring --/
def countMonochromaticTriangles (c : Coloring) : Nat :=
  sorry

/-- Theorem: If 15 segments are colored, there are at least two monochromatic triangles --/
theorem fifteen_segments_two_monochromatic_triangles (c : Coloring) :
  countColoredSegments c = 15 → countMonochromaticTriangles c ≥ 2 :=
  sorry

/-- Theorem: It's possible to color 14 segments without forming a monochromatic triangle --/
theorem fourteen_segments_no_monochromatic_triangle_possible :
  ∃ c : Coloring, countColoredSegments c = 14 ∧ countMonochromaticTriangles c = 0 :=
  sorry

end NUMINAMATH_CALUDE_fifteen_segments_two_monochromatic_triangles_fourteen_segments_no_monochromatic_triangle_possible_l1706_170647


namespace NUMINAMATH_CALUDE_equivalent_percentage_increase_l1706_170681

/-- Theorem: Equivalent percentage increase after multiple increases -/
theorem equivalent_percentage_increase (P : ℝ) (X : ℝ) :
  let first_increase := P * (1 + 0.06)
  let second_increase := first_increase * (1 + 0.06)
  let final_increase := second_increase * (1 + X / 100)
  let equivalent_increase := P * (1 + (12.36 + 1.1236 * X) / 100)
  final_increase = equivalent_increase :=
by sorry

end NUMINAMATH_CALUDE_equivalent_percentage_increase_l1706_170681


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1706_170632

-- Define the sets M and N
def M : Set ℝ := {x | |x - 1| < 1}
def N : Set ℝ := {x | x * (x - 3) < 0}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | 0 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1706_170632


namespace NUMINAMATH_CALUDE_lines_planes_perpendicular_l1706_170642

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- State the theorem
theorem lines_planes_perpendicular
  (m n : Line) (α β : Plane)
  (h1 : parallel_lines m n)
  (h2 : parallel_line_plane m α)
  (h3 : perpendicular_line_plane n β) :
  perpendicular_planes α β :=
sorry

end NUMINAMATH_CALUDE_lines_planes_perpendicular_l1706_170642


namespace NUMINAMATH_CALUDE_eggs_from_martha_l1706_170698

/-- The number of chickens Trevor collects eggs from -/
def num_chickens : ℕ := 4

/-- The number of eggs Trevor collected from Gertrude -/
def eggs_from_gertrude : ℕ := 4

/-- The number of eggs Trevor collected from Blanche -/
def eggs_from_blanche : ℕ := 3

/-- The number of eggs Trevor collected from Nancy -/
def eggs_from_nancy : ℕ := 2

/-- The number of eggs Trevor dropped -/
def eggs_dropped : ℕ := 2

/-- The number of eggs Trevor had left after dropping some -/
def eggs_left : ℕ := 9

/-- Theorem stating that Trevor got 2 eggs from Martha -/
theorem eggs_from_martha : 
  eggs_from_gertrude + eggs_from_blanche + eggs_from_nancy + 2 = eggs_left + eggs_dropped :=
by sorry

end NUMINAMATH_CALUDE_eggs_from_martha_l1706_170698


namespace NUMINAMATH_CALUDE_cards_given_away_l1706_170614

theorem cards_given_away (brother_sets sister_sets friend_sets : ℕ) 
  (cards_per_set : ℕ) (h1 : brother_sets = 15) (h2 : sister_sets = 8) 
  (h3 : friend_sets = 4) (h4 : cards_per_set = 25) : 
  (brother_sets + sister_sets + friend_sets) * cards_per_set = 675 := by
  sorry

end NUMINAMATH_CALUDE_cards_given_away_l1706_170614


namespace NUMINAMATH_CALUDE_dividend_calculation_l1706_170692

theorem dividend_calculation (divisor quotient remainder : ℕ) : 
  divisor = 18 → quotient = 9 → remainder = 5 → 
  divisor * quotient + remainder = 167 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l1706_170692


namespace NUMINAMATH_CALUDE_solve_for_m_l1706_170661

theorem solve_for_m (x y m : ℝ) 
  (h1 : x = 3 * m + 1)
  (h2 : y = 2 * m - 2)
  (h3 : 4 * x - 3 * y = 10) : 
  m = 0 := by
sorry

end NUMINAMATH_CALUDE_solve_for_m_l1706_170661


namespace NUMINAMATH_CALUDE_line_parametrization_l1706_170603

/-- The line equation --/
def line_equation (x y : ℝ) : Prop := y = (2/3) * x + 3

/-- The parametric equation of the line --/
def parametric_equation (x y s l t : ℝ) : Prop :=
  x = -9 + t * l ∧ y = s + t * (-7)

/-- The theorem stating the values of s and l --/
theorem line_parametrization :
  ∃ (s l : ℝ), (∀ (x y t : ℝ), line_equation x y ↔ parametric_equation x y s l t) ∧ s = -3 ∧ l = -10.5 := by
  sorry

end NUMINAMATH_CALUDE_line_parametrization_l1706_170603


namespace NUMINAMATH_CALUDE_winning_strategy_l1706_170638

/-- Represents the winner of the game -/
inductive Winner
  | FirstPlayer
  | SecondPlayer

/-- Determines the winner of the game based on board dimensions -/
def gameWinner (n k : ℕ) : Winner :=
  if (n + k) % 2 = 0 then Winner.SecondPlayer else Winner.FirstPlayer

/-- Theorem stating the winning condition for the game -/
theorem winning_strategy (n k : ℕ) (h1 : n > 0) (h2 : k > 1) :
  gameWinner n k = if (n + k) % 2 = 0 then Winner.SecondPlayer else Winner.FirstPlayer :=
by sorry

end NUMINAMATH_CALUDE_winning_strategy_l1706_170638


namespace NUMINAMATH_CALUDE_uniform_count_l1706_170690

theorem uniform_count (pants_cost shirt_cost tie_cost socks_cost total_spend : ℚ) 
  (h1 : pants_cost = 20)
  (h2 : shirt_cost = 2 * pants_cost)
  (h3 : tie_cost = shirt_cost / 5)
  (h4 : socks_cost = 3)
  (h5 : total_spend = 355) :
  (total_spend / (pants_cost + shirt_cost + tie_cost + socks_cost) : ℚ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_uniform_count_l1706_170690


namespace NUMINAMATH_CALUDE_team_selection_with_quadruplets_l1706_170635

/-- The number of ways to choose a team with restrictions on quadruplets -/
def choose_team (total_players : ℕ) (team_size : ℕ) (quadruplets : ℕ) : ℕ :=
  Nat.choose total_players team_size - Nat.choose (total_players - quadruplets) (team_size - quadruplets)

/-- Theorem stating the number of ways to choose the team under given conditions -/
theorem team_selection_with_quadruplets :
  choose_team 16 11 4 = 3576 := by
  sorry

end NUMINAMATH_CALUDE_team_selection_with_quadruplets_l1706_170635


namespace NUMINAMATH_CALUDE_distance_polar_point_to_circle_center_distance_specific_point_to_specific_circle_l1706_170606

/-- The distance between a point in polar coordinates and the center of a circle defined by a polar equation --/
theorem distance_polar_point_to_circle_center 
  (r : ℝ) (θ : ℝ) (circle_eq : ℝ → ℝ → Prop) : Prop :=
  let p_rect := (r * Real.cos θ, r * Real.sin θ)
  let circle_center := (1, 0)
  Real.sqrt ((p_rect.1 - circle_center.1)^2 + (p_rect.2 - circle_center.2)^2) = Real.sqrt 3

/-- The main theorem to be proved --/
theorem distance_specific_point_to_specific_circle : 
  distance_polar_point_to_circle_center 2 (Real.pi / 3) (fun ρ θ ↦ ρ = 2 * Real.cos θ) :=
sorry

end NUMINAMATH_CALUDE_distance_polar_point_to_circle_center_distance_specific_point_to_specific_circle_l1706_170606


namespace NUMINAMATH_CALUDE_divide_algebraic_expression_l1706_170663

theorem divide_algebraic_expression (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  12 * a^4 * b^3 * c / (-4 * a^3 * b^2) = -3 * a * b * c :=
by sorry

end NUMINAMATH_CALUDE_divide_algebraic_expression_l1706_170663


namespace NUMINAMATH_CALUDE_divisibility_of_power_minus_odd_l1706_170652

theorem divisibility_of_power_minus_odd (k m : ℕ) (hk : k > 0) (hm : Odd m) :
  ∃ n : ℕ, n > 0 ∧ (2^k : ℕ) ∣ (n^n - m) :=
sorry

end NUMINAMATH_CALUDE_divisibility_of_power_minus_odd_l1706_170652


namespace NUMINAMATH_CALUDE_triangle_perimeter_from_medians_l1706_170634

/-- If a triangle has medians of lengths 3, 4, and 6, then its perimeter is 26. -/
theorem triangle_perimeter_from_medians (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (med1 : ∃ (m : ℝ), m = 3 ∧ m^2 = (b^2 + c^2) / 4 - a^2 / 16)
  (med2 : ∃ (m : ℝ), m = 4 ∧ m^2 = (a^2 + c^2) / 4 - b^2 / 16)
  (med3 : ∃ (m : ℝ), m = 6 ∧ m^2 = (a^2 + b^2) / 4 - c^2 / 16) :
  a + b + c = 26 := by
  sorry


end NUMINAMATH_CALUDE_triangle_perimeter_from_medians_l1706_170634


namespace NUMINAMATH_CALUDE_special_number_in_list_l1706_170678

theorem special_number_in_list (numbers : List ℝ) (n : ℝ) : 
  numbers.length = 21 ∧ 
  n ∈ numbers ∧
  n = 4 * ((numbers.sum - n) / 20) →
  n = (1 / 6) * numbers.sum :=
by sorry

end NUMINAMATH_CALUDE_special_number_in_list_l1706_170678


namespace NUMINAMATH_CALUDE_babysitting_hours_l1706_170615

/-- Represents the babysitting scenario -/
structure BabysittingScenario where
  hourly_rate : ℚ
  makeup_fraction : ℚ
  skincare_fraction : ℚ
  remaining_amount : ℚ

/-- Calculates the number of hours babysitted per day -/
def hours_per_day (scenario : BabysittingScenario) : ℚ :=
  ((1 - scenario.makeup_fraction - scenario.skincare_fraction) * scenario.remaining_amount) /
  (7 * scenario.hourly_rate)

/-- Theorem stating that given the specific scenario, the person babysits for 3 hours each day -/
theorem babysitting_hours (scenario : BabysittingScenario) 
  (h1 : scenario.hourly_rate = 10)
  (h2 : scenario.makeup_fraction = 3/10)
  (h3 : scenario.skincare_fraction = 2/5)
  (h4 : scenario.remaining_amount = 63) :
  hours_per_day scenario = 3 := by
  sorry

#eval hours_per_day { hourly_rate := 10, makeup_fraction := 3/10, skincare_fraction := 2/5, remaining_amount := 63 }

end NUMINAMATH_CALUDE_babysitting_hours_l1706_170615


namespace NUMINAMATH_CALUDE_fraction_inequality_l1706_170677

theorem fraction_inequality (a b : ℝ) (h : b < a ∧ a < 0) : 1 / a < 1 / b := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l1706_170677


namespace NUMINAMATH_CALUDE_bbq_attendance_l1706_170653

def ice_per_person : ℕ := 2
def bags_per_pack : ℕ := 10
def price_per_pack : ℚ := 3
def total_spent : ℚ := 9

theorem bbq_attendance : ℕ := by
  sorry

end NUMINAMATH_CALUDE_bbq_attendance_l1706_170653


namespace NUMINAMATH_CALUDE_physics_marks_calculation_l1706_170605

def english_marks : ℕ := 76
def math_marks : ℕ := 65
def chemistry_marks : ℕ := 67
def biology_marks : ℕ := 85
def average_marks : ℕ := 75
def total_subjects : ℕ := 5

theorem physics_marks_calculation :
  let known_subjects_total : ℕ := english_marks + math_marks + chemistry_marks + biology_marks
  let total_marks : ℕ := average_marks * total_subjects
  total_marks - known_subjects_total = 82 := by sorry

end NUMINAMATH_CALUDE_physics_marks_calculation_l1706_170605


namespace NUMINAMATH_CALUDE_sin_2005_equals_neg_sin_25_l1706_170654

theorem sin_2005_equals_neg_sin_25 :
  Real.sin (2005 * π / 180) = -Real.sin (25 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_sin_2005_equals_neg_sin_25_l1706_170654


namespace NUMINAMATH_CALUDE_chocolate_fraction_is_11_24_l1706_170630

/-- The fraction of students who chose chocolate ice cream -/
def chocolate_fraction (chocolate strawberry vanilla : ℕ) : ℚ :=
  chocolate / (chocolate + strawberry + vanilla)

/-- Theorem stating that the fraction of students who chose chocolate ice cream is 11/24 -/
theorem chocolate_fraction_is_11_24 :
  chocolate_fraction 11 5 8 = 11 / 24 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_fraction_is_11_24_l1706_170630


namespace NUMINAMATH_CALUDE_book_sale_result_l1706_170644

theorem book_sale_result (selling_price : ℝ) (profit_percent : ℝ) (loss_percent : ℝ) :
  selling_price = 4.5 ∧ 
  profit_percent = 25 ∧ 
  loss_percent = 25 →
  (selling_price * 2) - (selling_price / (1 + profit_percent / 100) + selling_price / (1 - loss_percent / 100)) = -0.6 := by
  sorry

end NUMINAMATH_CALUDE_book_sale_result_l1706_170644


namespace NUMINAMATH_CALUDE_pentagonal_prism_lateral_angle_l1706_170675

/-- A pentagonal prism is a three-dimensional geometric shape with a pentagonal base
    and rectangular lateral faces. -/
structure PentagonalPrism where
  base : Pentagon
  height : ℝ
  height_pos : height > 0

/-- The angle between a lateral edge and the base of a pentagonal prism. -/
def lateral_angle (p : PentagonalPrism) : ℝ := sorry

/-- Theorem: The angle between any lateral edge and the base of a pentagonal prism is 90°. -/
theorem pentagonal_prism_lateral_angle (p : PentagonalPrism) :
  lateral_angle p = Real.pi / 2 := by sorry

end NUMINAMATH_CALUDE_pentagonal_prism_lateral_angle_l1706_170675


namespace NUMINAMATH_CALUDE_evenness_of_k_l1706_170620

theorem evenness_of_k (a b n k : ℕ) (ha : 0 < a) (hb : 0 < b) (hn : 0 < n) (hk : 0 < k)
  (h1 : 2^n - 1 = a * b)
  (h2 : (a * b + a - b - 1) % 2^k = 0)
  (h3 : (a * b + a - b - 1) % 2^(k+1) ≠ 0) :
  Even k := by sorry

end NUMINAMATH_CALUDE_evenness_of_k_l1706_170620


namespace NUMINAMATH_CALUDE_angle_line_plane_l1706_170664

-- Define the line and plane
def line_eq1 (x z : ℝ) : Prop := x - 2*z + 3 = 0
def line_eq2 (y z : ℝ) : Prop := y + 3*z - 1 = 0
def plane_eq (x y z : ℝ) : Prop := 2*x - y + z + 3 = 0

-- Define the angle between the line and plane
def angle_between_line_and_plane : ℝ := sorry

-- State the theorem
theorem angle_line_plane :
  Real.sin angle_between_line_and_plane = 4 * Real.sqrt 21 / 21 :=
sorry

end NUMINAMATH_CALUDE_angle_line_plane_l1706_170664


namespace NUMINAMATH_CALUDE_probability_two_heads_two_tails_l1706_170641

theorem probability_two_heads_two_tails : 
  let n : ℕ := 4  -- total number of coins
  let k : ℕ := 2  -- number of heads (or tails) we want
  let p : ℚ := 1/2  -- probability of getting heads (or tails) on a single toss
  Nat.choose n k * p^n = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_heads_two_tails_l1706_170641


namespace NUMINAMATH_CALUDE_nes_sale_price_l1706_170621

/-- The sale price of an NES given trade-in and cash transactions -/
theorem nes_sale_price 
  (snes_value : ℝ) 
  (trade_in_percentage : ℝ) 
  (cash_given : ℝ) 
  (change_received : ℝ) 
  (game_value : ℝ) 
  (h1 : snes_value = 150)
  (h2 : trade_in_percentage = 0.8)
  (h3 : cash_given = 80)
  (h4 : change_received = 10)
  (h5 : game_value = 30) :
  snes_value * trade_in_percentage + cash_given - change_received - game_value = 160 :=
by sorry

end NUMINAMATH_CALUDE_nes_sale_price_l1706_170621


namespace NUMINAMATH_CALUDE_elmer_milton_ratio_l1706_170668

-- Define the daily food intake for each animal
def penelope_intake : ℚ := 20
def greta_intake : ℚ := penelope_intake / 10
def milton_intake : ℚ := greta_intake / 100
def elmer_intake : ℚ := penelope_intake + 60

-- Theorem statement
theorem elmer_milton_ratio : 
  elmer_intake / milton_intake = 4000 := by sorry

end NUMINAMATH_CALUDE_elmer_milton_ratio_l1706_170668


namespace NUMINAMATH_CALUDE_max_sum_given_constraints_l1706_170648

theorem max_sum_given_constraints (x y : ℝ) 
  (h1 : x^2 + y^2 = 98) 
  (h2 : x * y = 40) : 
  x + y ≤ Real.sqrt 178 := by
sorry

end NUMINAMATH_CALUDE_max_sum_given_constraints_l1706_170648


namespace NUMINAMATH_CALUDE_distinct_polygons_from_circle_points_l1706_170699

theorem distinct_polygons_from_circle_points (n : ℕ) (h : n = 12) : 
  (2^n : ℕ) - (1 + n + n*(n-1)/2) = 4017 := by
  sorry

end NUMINAMATH_CALUDE_distinct_polygons_from_circle_points_l1706_170699


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1706_170656

/-- The number of terms in a geometric sequence with first term 1 and common ratio 1/4 
    that sum to 85/64 -/
theorem geometric_sequence_sum (n : ℕ) : 
  (1 - (1/4)^n) / (1 - 1/4) = 85/64 → n = 4 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1706_170656


namespace NUMINAMATH_CALUDE_x_range_for_P_in_fourth_quadrant_l1706_170691

-- Define the point P
def P (x : ℝ) : ℝ × ℝ := (2*x - 6, x - 5)

-- Define the condition for a point to be in the fourth quadrant
def in_fourth_quadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 < 0

-- Theorem statement
theorem x_range_for_P_in_fourth_quadrant :
  ∀ x : ℝ, in_fourth_quadrant (P x) ↔ 3 < x ∧ x < 5 := by sorry

end NUMINAMATH_CALUDE_x_range_for_P_in_fourth_quadrant_l1706_170691


namespace NUMINAMATH_CALUDE_cubic_equation_one_solution_l1706_170643

/-- The cubic equation in x with parameter b -/
def cubic_equation (x b : ℝ) : ℝ := x^3 - b*x^2 - 3*b*x + b^2 - 4

/-- The condition for the equation to have exactly one real solution -/
def has_one_real_solution (b : ℝ) : Prop :=
  ∃! x : ℝ, cubic_equation x b = 0

theorem cubic_equation_one_solution :
  ∀ b : ℝ, has_one_real_solution b ↔ b > 3 := by sorry

end NUMINAMATH_CALUDE_cubic_equation_one_solution_l1706_170643
