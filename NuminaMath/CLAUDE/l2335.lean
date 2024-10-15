import Mathlib

namespace NUMINAMATH_CALUDE_sqrt_two_sqrt_two_power_l2335_233552

theorem sqrt_two_sqrt_two_power : (((2 * Real.sqrt 2) ^ 4).sqrt) ^ 3 = 512 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_sqrt_two_power_l2335_233552


namespace NUMINAMATH_CALUDE_salary_restoration_l2335_233575

theorem salary_restoration (original_salary : ℝ) (original_salary_positive : 0 < original_salary) :
  let reduced_salary := original_salary * (1 - 0.25)
  let increase_factor := 1 + (1 / 3)
  reduced_salary * increase_factor = original_salary :=
by sorry

end NUMINAMATH_CALUDE_salary_restoration_l2335_233575


namespace NUMINAMATH_CALUDE_variation_relationship_l2335_233528

theorem variation_relationship (x y z : ℝ) (k j : ℝ) (h1 : x = k * y^2) (h2 : y = j * z^(1/3)) :
  ∃ m : ℝ, x = m * z^(2/3) :=
by sorry

end NUMINAMATH_CALUDE_variation_relationship_l2335_233528


namespace NUMINAMATH_CALUDE_triangle_inequality_theorem_l2335_233566

/-- Checks if three lengths can form a triangle according to the triangle inequality theorem -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_inequality_theorem :
  can_form_triangle 3 4 5 ∧
  ¬can_form_triangle 2 4 7 ∧
  ¬can_form_triangle 3 6 9 ∧
  ¬can_form_triangle 4 4 9 :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_theorem_l2335_233566


namespace NUMINAMATH_CALUDE_max_length_sum_xy_l2335_233595

/-- The length of an integer is the number of positive prime factors (not necessarily distinct) whose product equals the integer. -/
def length (n : ℕ) : ℕ := sorry

/-- Given the constraints, the maximum sum of lengths of x and y is 15. -/
theorem max_length_sum_xy : 
  ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ x + 3*y < 920 ∧ 
  ∀ (a b : ℕ), a > 1 → b > 1 → a + 3*b < 920 → 
  length x + length y ≥ length a + length b ∧
  length x + length y = 15 := by
sorry

end NUMINAMATH_CALUDE_max_length_sum_xy_l2335_233595


namespace NUMINAMATH_CALUDE_shobhas_current_age_l2335_233583

/-- Given the ratio of Shekhar's age to Shobha's age and Shekhar's future age, 
    prove Shobha's current age -/
theorem shobhas_current_age 
  (shekhar_age shobha_age : ℕ) 
  (ratio : shekhar_age / shobha_age = 4 / 3)
  (future_age : shekhar_age + 6 = 26) : 
  shobha_age = 15 := by
  sorry

end NUMINAMATH_CALUDE_shobhas_current_age_l2335_233583


namespace NUMINAMATH_CALUDE_profit_maximization_profit_function_correct_sales_at_price_l2335_233537

/-- Represents the daily profit function for a product -/
def profit_function (x : ℝ) : ℝ := (200 - x) * (x - 120)

/-- The cost price of the product -/
def cost_price : ℝ := 120

/-- The reference price point -/
def reference_price : ℝ := 130

/-- The daily sales at the reference price -/
def reference_sales : ℝ := 70

/-- The rate of change in sales with respect to price -/
def sales_price_ratio : ℝ := -1

theorem profit_maximization :
  ∃ (max_price max_profit : ℝ),
    (∀ x, profit_function x ≤ max_profit) ∧
    profit_function max_price = max_profit ∧
    max_price = 160 ∧
    max_profit = 1600 := by sorry

theorem profit_function_correct :
  ∀ x, profit_function x = (200 - x) * (x - cost_price) := by sorry

theorem sales_at_price (x : ℝ) :
  x ≥ reference_price →
  profit_function x = (reference_sales + sales_price_ratio * (x - reference_price)) * (x - cost_price) := by sorry

end NUMINAMATH_CALUDE_profit_maximization_profit_function_correct_sales_at_price_l2335_233537


namespace NUMINAMATH_CALUDE_solution_of_equation_l2335_233530

theorem solution_of_equation (x : ℚ) : -2 * x + 11 = 0 ↔ x = 11 / 2 := by sorry

end NUMINAMATH_CALUDE_solution_of_equation_l2335_233530


namespace NUMINAMATH_CALUDE_fuse_probability_l2335_233573

/-- The probability of the union of two events -/
def prob_union (prob_A prob_B prob_A_and_B : ℝ) : ℝ :=
  prob_A + prob_B - prob_A_and_B

theorem fuse_probability (prob_A prob_B prob_A_and_B : ℝ) 
  (h1 : prob_A = 0.085)
  (h2 : prob_B = 0.074)
  (h3 : prob_A_and_B = 0.063) :
  prob_union prob_A prob_B prob_A_and_B = 0.096 := by
sorry

end NUMINAMATH_CALUDE_fuse_probability_l2335_233573


namespace NUMINAMATH_CALUDE_work_days_calculation_l2335_233506

/-- Proves that A and B worked together for 10 days given the conditions -/
theorem work_days_calculation (a_rate : ℚ) (b_rate : ℚ) (remaining_work : ℚ) : 
  a_rate = 1 / 30 →
  b_rate = 1 / 40 →
  remaining_work = 5 / 12 →
  ∃ d : ℚ, d = 10 ∧ (a_rate + b_rate) * d = 1 - remaining_work :=
by sorry

end NUMINAMATH_CALUDE_work_days_calculation_l2335_233506


namespace NUMINAMATH_CALUDE_pink_cookies_l2335_233594

theorem pink_cookies (total : ℕ) (red : ℕ) (h1 : total = 86) (h2 : red = 36) :
  total - red = 50 := by
  sorry

end NUMINAMATH_CALUDE_pink_cookies_l2335_233594


namespace NUMINAMATH_CALUDE_diaries_count_l2335_233518

/-- The number of diaries Natalie's sister has after buying and losing some -/
def final_diaries : ℕ :=
  let initial : ℕ := 23
  let bought : ℕ := 5 * initial
  let total : ℕ := initial + bought
  let lost : ℕ := (7 * total) / 9
  total - lost

theorem diaries_count : final_diaries = 31 := by
  sorry

end NUMINAMATH_CALUDE_diaries_count_l2335_233518


namespace NUMINAMATH_CALUDE_mobile_plan_comparison_l2335_233508

/-- Represents the monthly cost in yuan for a mobile phone plan -/
def monthly_cost (rental : ℝ) (rate : ℝ) (duration : ℝ) : ℝ :=
  rental + rate * duration

/-- The monthly rental fee for Global Call in yuan -/
def global_call_rental : ℝ := 50

/-- The per-minute call rate for Global Call in yuan -/
def global_call_rate : ℝ := 0.4

/-- The monthly rental fee for Shenzhouxing in yuan -/
def shenzhouxing_rental : ℝ := 0

/-- The per-minute call rate for Shenzhouxing in yuan -/
def shenzhouxing_rate : ℝ := 0.6

/-- The breakeven point in minutes where both plans cost the same -/
def breakeven_point : ℝ := 250

theorem mobile_plan_comparison :
  ∀ duration : ℝ,
    duration > breakeven_point →
      monthly_cost global_call_rental global_call_rate duration <
      monthly_cost shenzhouxing_rental shenzhouxing_rate duration ∧
    duration < breakeven_point →
      monthly_cost global_call_rental global_call_rate duration >
      monthly_cost shenzhouxing_rental shenzhouxing_rate duration ∧
    duration = breakeven_point →
      monthly_cost global_call_rental global_call_rate duration =
      monthly_cost shenzhouxing_rental shenzhouxing_rate duration :=
by
  sorry

end NUMINAMATH_CALUDE_mobile_plan_comparison_l2335_233508


namespace NUMINAMATH_CALUDE_prob_roll_less_than_4_l2335_233582

/-- A fair 8-sided die -/
def fair_8_sided_die : Finset (Fin 8) := Finset.univ

/-- The event of rolling a number less than 4 -/
def roll_less_than_4 : Finset (Fin 8) := Finset.filter (λ x => x.val < 4) fair_8_sided_die

/-- The probability of an event occurring when rolling a fair 8-sided die -/
def prob (event : Finset (Fin 8)) : ℚ :=
  (event.card : ℚ) / (fair_8_sided_die.card : ℚ)

theorem prob_roll_less_than_4 : 
  prob roll_less_than_4 = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_prob_roll_less_than_4_l2335_233582


namespace NUMINAMATH_CALUDE_expression_evaluation_l2335_233510

theorem expression_evaluation : 200 * (200 - 3) + (200^2 - 8^2) = 79336 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2335_233510


namespace NUMINAMATH_CALUDE_triangle_area_l2335_233558

/-- Given a triangle ABC with side lengths b and c, and angle C, prove that its area is √3/4 -/
theorem triangle_area (b c : ℝ) (C : ℝ) (h1 : b = 1) (h2 : c = Real.sqrt 3) (h3 : C = 2 * Real.pi / 3) :
  (1 / 2) * b * c * Real.sin (Real.pi / 6) = Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2335_233558


namespace NUMINAMATH_CALUDE_hyperbola_other_asymptote_l2335_233555

/-- A hyperbola with given properties -/
structure Hyperbola where
  /-- One asymptote of the hyperbola -/
  asymptote1 : ℝ → ℝ
  /-- The x-coordinate of the foci -/
  foci_x : ℝ
  /-- The hyperbola has a horizontal axis -/
  horizontal_axis : Prop

/-- The other asymptote of the hyperbola -/
def other_asymptote (h : Hyperbola) : ℝ → ℝ := 
  fun x ↦ -2 * x + 16

theorem hyperbola_other_asymptote (h : Hyperbola) 
  (h1 : h.asymptote1 = fun x ↦ 2 * x) 
  (h2 : h.foci_x = 4)
  (h3 : h.horizontal_axis) :
  other_asymptote h = fun x ↦ -2 * x + 16 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_other_asymptote_l2335_233555


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2335_233580

theorem imaginary_part_of_complex_fraction (z : ℂ) : z = (1 + 2*I) / (2 - I) → z.im = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2335_233580


namespace NUMINAMATH_CALUDE_infinite_binary_decimal_divisible_by_2017_l2335_233564

/-- A number composed only of digits 0 and 1 in decimal representation -/
def is_binary_decimal (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 0 ∨ d = 1

/-- The set of numbers composed only of digits 0 and 1 in decimal representation -/
def binary_decimal_set : Set ℕ :=
  {n : ℕ | is_binary_decimal n}

/-- The theorem statement -/
theorem infinite_binary_decimal_divisible_by_2017 :
  ∃ S : Set ℕ, (∀ n ∈ S, is_binary_decimal n ∧ 2017 ∣ n) ∧ Set.Infinite S :=
sorry

end NUMINAMATH_CALUDE_infinite_binary_decimal_divisible_by_2017_l2335_233564


namespace NUMINAMATH_CALUDE_total_notes_count_l2335_233577

def total_amount : ℕ := 10350
def note_50_value : ℕ := 50
def note_500_value : ℕ := 500
def note_50_count : ℕ := 37

theorem total_notes_count : 
  ∃ (note_500_count : ℕ), 
    note_50_count * note_50_value + note_500_count * note_500_value = total_amount ∧
    note_50_count + note_500_count = 54 :=
by sorry

end NUMINAMATH_CALUDE_total_notes_count_l2335_233577


namespace NUMINAMATH_CALUDE_student_number_problem_l2335_233533

theorem student_number_problem (x : ℝ) : 2 * x - 200 = 110 → x = 155 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l2335_233533


namespace NUMINAMATH_CALUDE_new_year_fireworks_display_l2335_233539

def fireworks_per_number : ℕ := 6
def fireworks_per_letter : ℕ := 5
def additional_boxes : ℕ := 50
def fireworks_per_box : ℕ := 8

def year_numbers : ℕ := 4
def phrase_letters : ℕ := 12

theorem new_year_fireworks_display :
  let year_fireworks := year_numbers * fireworks_per_number
  let phrase_fireworks := phrase_letters * fireworks_per_letter
  let additional_fireworks := additional_boxes * fireworks_per_box
  year_fireworks + phrase_fireworks + additional_fireworks = 476 := by
sorry

end NUMINAMATH_CALUDE_new_year_fireworks_display_l2335_233539


namespace NUMINAMATH_CALUDE_tank_filling_proof_l2335_233550

/-- The number of buckets required to fill a tank with the original bucket size -/
def original_buckets : ℕ := 10

/-- The number of buckets required to fill the tank with reduced bucket capacity -/
def reduced_buckets : ℕ := 25

/-- The ratio of reduced bucket capacity to original bucket capacity -/
def capacity_ratio : ℚ := 2 / 5

theorem tank_filling_proof :
  original_buckets * 1 = reduced_buckets * capacity_ratio :=
by sorry

end NUMINAMATH_CALUDE_tank_filling_proof_l2335_233550


namespace NUMINAMATH_CALUDE_equation_solutions_l2335_233589

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, (x₁^2 - 3*x₁ - 4 = 0 ∧ x₂^2 - 3*x₂ - 4 = 0) ∧ x₁ = 4 ∧ x₂ = -1) ∧
  (∃ y₁ y₂ : ℝ, (y₁*(y₁ - 2) = 1 ∧ y₂*(y₂ - 2) = 1) ∧ y₁ = 1 + Real.sqrt 2 ∧ y₂ = 1 - Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2335_233589


namespace NUMINAMATH_CALUDE_abs_three_minus_a_l2335_233543

theorem abs_three_minus_a (a : ℝ) (h : |1 - a| = 1 + |a|) : |3 - a| = 3 - a := by
  sorry

end NUMINAMATH_CALUDE_abs_three_minus_a_l2335_233543


namespace NUMINAMATH_CALUDE_optimal_purchase_l2335_233541

def budget : ℕ := 100
def basic_calc_cost : ℕ := 8
def battery_cost : ℕ := 2
def scientific_calc_cost : ℕ := 2 * basic_calc_cost
def graphing_calc_cost : ℕ := 3 * scientific_calc_cost

def total_basic_cost : ℕ := basic_calc_cost + battery_cost
def total_scientific_cost : ℕ := scientific_calc_cost + battery_cost
def total_graphing_cost : ℕ := graphing_calc_cost + battery_cost

def one_of_each_cost : ℕ := total_basic_cost + total_scientific_cost + total_graphing_cost

theorem optimal_purchase :
  ∀ (b s g : ℕ),
    b ≥ 1 → s ≥ 1 → g ≥ 1 →
    (b + s + g) % 3 = 0 →
    b * total_basic_cost + s * total_scientific_cost + g * total_graphing_cost ≤ budget →
    b + s + g ≤ 3 ∧
    budget - (b * total_basic_cost + s * total_scientific_cost + g * total_graphing_cost) ≤ budget - one_of_each_cost :=
by sorry

end NUMINAMATH_CALUDE_optimal_purchase_l2335_233541


namespace NUMINAMATH_CALUDE_sparklers_to_crackers_value_comparison_l2335_233523

-- Define the exchange rates
def ornament_to_cracker : ℚ := 2
def sparkler_to_garland : ℚ := 2/5
def ornament_to_garland : ℚ := 1/4

-- Define the conversion function
def convert (item : String) (quantity : ℚ) : ℚ :=
  match item with
  | "sparkler" => quantity * sparkler_to_garland * (1 / ornament_to_garland) * ornament_to_cracker
  | "ornament" => quantity * ornament_to_cracker
  | _ => 0

-- Theorem 1: 10 sparklers are equivalent to 32 crackers
theorem sparklers_to_crackers :
  convert "sparkler" 10 = 32 :=
sorry

-- Theorem 2: 5 Christmas ornaments and 1 cracker are more valuable than 2 sparklers
theorem value_comparison :
  convert "ornament" 5 + 1 > convert "sparkler" 2 :=
sorry

end NUMINAMATH_CALUDE_sparklers_to_crackers_value_comparison_l2335_233523


namespace NUMINAMATH_CALUDE_cuboid_area_example_l2335_233526

/-- The surface area of a cuboid -/
def cuboid_surface_area (width length height : ℝ) : ℝ :=
  2 * (width * length + width * height + length * height)

/-- Theorem: The surface area of a cuboid with width 3 cm, length 4 cm, and height 5 cm is 94 cm² -/
theorem cuboid_area_example : cuboid_surface_area 3 4 5 = 94 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_area_example_l2335_233526


namespace NUMINAMATH_CALUDE_circle_equation_l2335_233522

/-- The standard equation of a circle with center (-1, 2) passing through (2, -2) -/
theorem circle_equation : ∀ x y : ℝ, (x + 1)^2 + (y - 2)^2 = 25 ↔ 
  ((x + 1)^2 + (y - 2)^2 = ((2 + 1)^2 + (-2 - 2)^2) ∧ 
   (x, y) ≠ (-1, 2)) := by sorry

end NUMINAMATH_CALUDE_circle_equation_l2335_233522


namespace NUMINAMATH_CALUDE_negative_root_condition_l2335_233560

theorem negative_root_condition (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ 7^(x+1) - 7^x * a - a - 5 = 0) ↔ -5 < a ∧ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_negative_root_condition_l2335_233560


namespace NUMINAMATH_CALUDE_triangle_perimeter_is_five_l2335_233502

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem: If in a triangle ABC, b*cos(A) + a*cos(B) = c^2 and a = b = 2, 
    then the perimeter of the triangle is 5 -/
theorem triangle_perimeter_is_five (t : Triangle) 
  (h1 : t.b * Real.cos t.A + t.a * Real.cos t.B = t.c^2)
  (h2 : t.a = 2)
  (h3 : t.b = 2) : 
  t.a + t.b + t.c = 5 := by
  sorry


end NUMINAMATH_CALUDE_triangle_perimeter_is_five_l2335_233502


namespace NUMINAMATH_CALUDE_intersection_implies_m_value_subset_implies_m_range_l2335_233553

-- Define sets A and B
def A : Set ℝ := {x : ℝ | 6 / (x + 1) ≥ 1}
def B (m : ℝ) : Set ℝ := {x : ℝ | x^2 - 2*x + 2*m < 0}

-- Theorem 1
theorem intersection_implies_m_value :
  ∀ m : ℝ, (A ∩ B m = {x : ℝ | -1 < x ∧ x < 4}) → m = -4 := by sorry

-- Theorem 2
theorem subset_implies_m_range :
  ∀ m : ℝ, (B m ⊆ A) → m ≥ -3/2 := by sorry

end NUMINAMATH_CALUDE_intersection_implies_m_value_subset_implies_m_range_l2335_233553


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l2335_233572

-- Define the line equation
def line_equation (x y a b : ℝ) : Prop := x / (a^2) - y / (b^2) = 1

-- Define y-intercept
def y_intercept (f : ℝ → ℝ) : ℝ := f 0

-- Theorem statement
theorem y_intercept_of_line (a b : ℝ) (h : b ≠ 0) :
  ∃ f : ℝ → ℝ, (∀ x, line_equation x (f x) a b) ∧ y_intercept f = -b^2 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l2335_233572


namespace NUMINAMATH_CALUDE_largest_value_l2335_233590

theorem largest_value (x y z w : ℝ) (h : x + 3 = y - 1 ∧ y - 1 = z + 5 ∧ z + 5 = w - 2) :
  w ≥ x ∧ w ≥ y ∧ w ≥ z := by sorry

end NUMINAMATH_CALUDE_largest_value_l2335_233590


namespace NUMINAMATH_CALUDE_fourth_intersection_point_l2335_233532

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The curve xy = 4 -/
def curve (p : Point) : Prop := p.x * p.y = 4

/-- A circle in the 2D plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- A point lies on a circle -/
def onCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

theorem fourth_intersection_point (c : Circle) 
    (h1 : curve (Point.mk 4 1) ∧ onCircle (Point.mk 4 1) c)
    (h2 : curve (Point.mk (-2) (-2)) ∧ onCircle (Point.mk (-2) (-2)) c)
    (h3 : curve (Point.mk 8 (1/2)) ∧ onCircle (Point.mk 8 (1/2)) c)
    (h4 : ∃ p : Point, curve p ∧ onCircle p c ∧ p ≠ Point.mk 4 1 ∧ p ≠ Point.mk (-2) (-2) ∧ p ≠ Point.mk 8 (1/2)) :
    ∃ p : Point, p = Point.mk (-1/4) (-16) ∧ curve p ∧ onCircle p c := by
  sorry

end NUMINAMATH_CALUDE_fourth_intersection_point_l2335_233532


namespace NUMINAMATH_CALUDE_sum_of_powers_of_i_l2335_233546

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_powers_of_i :
  1 + i + i^2 + i^3 + i^4 + i^5 + i^6 = i :=
by sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_i_l2335_233546


namespace NUMINAMATH_CALUDE_detergent_in_altered_solution_l2335_233559

/-- Represents the ratio of bleach : detergent : water in a solution -/
structure SolutionRatio :=
  (bleach : ℕ)
  (detergent : ℕ)
  (water : ℕ)

/-- Calculates the new ratio after tripling bleach to detergent and halving detergent to water -/
def alter_ratio (r : SolutionRatio) : SolutionRatio :=
  { bleach := 3 * r.bleach,
    detergent := 2 * r.detergent,
    water := 4 * r.water }

/-- Calculates the amount of detergent in the altered solution -/
def detergent_amount (r : SolutionRatio) (water_amount : ℕ) : ℕ :=
  (r.detergent * water_amount) / r.water

theorem detergent_in_altered_solution :
  let original_ratio : SolutionRatio := { bleach := 4, detergent := 40, water := 100 }
  let altered_ratio := alter_ratio original_ratio
  detergent_amount altered_ratio 300 = 60 := by
  sorry

end NUMINAMATH_CALUDE_detergent_in_altered_solution_l2335_233559


namespace NUMINAMATH_CALUDE_special_triangle_cosine_l2335_233598

/-- A triangle with consecutive integer side lengths where the middle angle is 1.5 times the smallest angle -/
structure SpecialTriangle where
  n : ℕ
  side1 : ℕ := n
  side2 : ℕ := n + 1
  side3 : ℕ := n + 2
  smallest_angle : ℝ
  middle_angle : ℝ
  largest_angle : ℝ
  angle_sum : middle_angle = 1.5 * smallest_angle
  angle_total : smallest_angle + middle_angle + largest_angle = Real.pi

/-- The cosine of the smallest angle in a SpecialTriangle is 53/60 -/
theorem special_triangle_cosine (t : SpecialTriangle) : 
  Real.cos t.smallest_angle = 53 / 60 := by sorry

end NUMINAMATH_CALUDE_special_triangle_cosine_l2335_233598


namespace NUMINAMATH_CALUDE_distribution_methods_eq_240_l2335_233513

/-- The number of ways to distribute 5 volunteers into 4 groups and assign them to intersections -/
def distributionMethods : ℕ := 
  (Nat.choose 5 2) * (Nat.factorial 4)

/-- Theorem stating that the number of distribution methods is 240 -/
theorem distribution_methods_eq_240 : distributionMethods = 240 := by
  sorry

end NUMINAMATH_CALUDE_distribution_methods_eq_240_l2335_233513


namespace NUMINAMATH_CALUDE_bob_spending_is_26_l2335_233509

-- Define the prices and quantities
def bread_price : ℚ := 2
def bread_quantity : ℕ := 4
def cheese_price : ℚ := 6
def cheese_quantity : ℕ := 2
def chocolate_price : ℚ := 3
def chocolate_quantity : ℕ := 3
def oil_price : ℚ := 10
def oil_quantity : ℕ := 1

-- Define the discount and coupon
def cheese_discount : ℚ := 0.25
def coupon_value : ℚ := 10
def coupon_threshold : ℚ := 30

-- Define Bob's spending function
def bob_spending : ℚ :=
  let bread_total := bread_price * bread_quantity
  let cheese_total := cheese_price * cheese_quantity * (1 - cheese_discount)
  let chocolate_total := chocolate_price * chocolate_quantity
  let oil_total := oil_price * oil_quantity
  let subtotal := bread_total + cheese_total + chocolate_total + oil_total
  if subtotal ≥ coupon_threshold then subtotal - coupon_value else subtotal

-- Theorem to prove
theorem bob_spending_is_26 : bob_spending = 26 := by sorry

end NUMINAMATH_CALUDE_bob_spending_is_26_l2335_233509


namespace NUMINAMATH_CALUDE_parallel_vectors_condition_l2335_233571

/-- Given plane vectors a and b, if a + b is parallel to a - b, then the second component of b is -2√3. -/
theorem parallel_vectors_condition (a b : ℝ × ℝ) :
  a = (1, -Real.sqrt 3) →
  b.1 = 2 →
  (∃ (k : ℝ), k ≠ 0 ∧ (a + b) = k • (a - b)) →
  b.2 = -2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_condition_l2335_233571


namespace NUMINAMATH_CALUDE_minutes_after_midnight_theorem_l2335_233504

/-- Represents a date and time -/
structure DateTime where
  year : Nat
  month : Nat
  day : Nat
  hour : Nat
  minute : Nat

/-- Adds minutes to a DateTime -/
def addMinutes (dt : DateTime) (minutes : Nat) : DateTime :=
  sorry

/-- The initial DateTime (midnight on February 1, 2022) -/
def initialDateTime : DateTime :=
  { year := 2022, month := 2, day := 1, hour := 0, minute := 0 }

/-- The final DateTime after adding 1553 minutes -/
def finalDateTime : DateTime :=
  addMinutes initialDateTime 1553

/-- Theorem stating that 1553 minutes after midnight on February 1, 2022 is February 2 at 1:53 AM -/
theorem minutes_after_midnight_theorem :
  finalDateTime = { year := 2022, month := 2, day := 2, hour := 1, minute := 53 } :=
  sorry

end NUMINAMATH_CALUDE_minutes_after_midnight_theorem_l2335_233504


namespace NUMINAMATH_CALUDE_trigonometric_expression_equality_l2335_233542

theorem trigonometric_expression_equality : 
  1 / Real.cos (70 * π / 180) - Real.sqrt 2 / Real.sin (70 * π / 180) = 4 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equality_l2335_233542


namespace NUMINAMATH_CALUDE_problem_solution_l2335_233586

theorem problem_solution (m n : ℝ) (h1 : m - n = 6) (h2 : m * n = 4) : 
  (m^2 + n^2 = 44) ∧ ((m + 2) * (n - 2) = -12) := by sorry

end NUMINAMATH_CALUDE_problem_solution_l2335_233586


namespace NUMINAMATH_CALUDE_rational_roots_count_l2335_233587

/-- The set of factors of a natural number -/
def factors (n : ℕ) : Finset ℤ :=
  sorry

/-- The set of possible rational roots for a polynomial with given leading coefficient and constant term -/
def possibleRationalRoots (leadingCoeff constTerm : ℤ) : Finset ℚ :=
  sorry

/-- Theorem stating that the number of different possible rational roots for the given polynomial form is 20 -/
theorem rational_roots_count :
  let leadingCoeff := 4
  let constTerm := 18
  (possibleRationalRoots leadingCoeff constTerm).card = 20 := by
  sorry

end NUMINAMATH_CALUDE_rational_roots_count_l2335_233587


namespace NUMINAMATH_CALUDE_b_minus_c_equals_one_l2335_233536

theorem b_minus_c_equals_one (A B C : ℤ) 
  (h1 : A = 9 - 4)
  (h2 : B = A + 5)
  (h3 : C - 8 = 1)
  (h4 : A ≠ B ∧ B ≠ C ∧ A ≠ C) :
  B - C = 1 := by
  sorry

end NUMINAMATH_CALUDE_b_minus_c_equals_one_l2335_233536


namespace NUMINAMATH_CALUDE_divisibility_property_l2335_233548

theorem divisibility_property (a b c : ℤ) (h : 13 ∣ (a + b + c)) :
  13 ∣ (a^2007 + b^2007 + c^2007 + 2 * 2007 * a * b * c) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_property_l2335_233548


namespace NUMINAMATH_CALUDE_marathon_first_hour_distance_l2335_233505

/-- Represents a marathon runner's performance -/
structure MarathonRunner where
  initialPace : ℝ  -- Initial pace in miles per hour
  totalDistance : ℝ -- Total marathon distance in miles
  totalTime : ℝ     -- Total race time in hours
  remainingPaceFactor : ℝ -- Factor for remaining pace (e.g., 0.8 for 80%)

/-- Calculates the distance covered in the first hour -/
def distanceInFirstHour (runner : MarathonRunner) : ℝ :=
  runner.initialPace

/-- Calculates the remaining distance after the first hour -/
def remainingDistance (runner : MarathonRunner) : ℝ :=
  runner.totalDistance - distanceInFirstHour runner

/-- Calculates the time spent running the remaining distance -/
def remainingTime (runner : MarathonRunner) : ℝ :=
  runner.totalTime - 1

/-- Theorem: The distance covered in the first hour of a 26-mile marathon is 10 miles -/
theorem marathon_first_hour_distance
  (runner : MarathonRunner)
  (h1 : runner.totalDistance = 26)
  (h2 : runner.totalTime = 3)
  (h3 : runner.remainingPaceFactor = 0.8)
  (h4 : remainingTime runner = 2)
  (h5 : remainingDistance runner / (runner.initialPace * runner.remainingPaceFactor) = remainingTime runner) :
  distanceInFirstHour runner = 10 := by
  sorry


end NUMINAMATH_CALUDE_marathon_first_hour_distance_l2335_233505


namespace NUMINAMATH_CALUDE_at_most_two_match_count_l2335_233554

/-- The number of ways to arrange 5 balls in 5 boxes -/
def total_arrangements : ℕ := 120

/-- The number of ways to arrange 5 balls in 5 boxes where exactly 3 balls match their box number -/
def three_match_arrangements : ℕ := 10

/-- The number of ways to arrange 5 balls in 5 boxes where all 5 balls match their box number -/
def all_match_arrangement : ℕ := 1

/-- The number of ways to arrange 5 balls in 5 boxes such that at most two balls have the same number as their respective boxes -/
def at_most_two_match : ℕ := total_arrangements - three_match_arrangements - all_match_arrangement

theorem at_most_two_match_count : at_most_two_match = 109 := by
  sorry

end NUMINAMATH_CALUDE_at_most_two_match_count_l2335_233554


namespace NUMINAMATH_CALUDE_unique_sequence_existence_l2335_233574

theorem unique_sequence_existence : ∃! a : ℕ → ℤ,
  (a 1 = 1) ∧
  (a 2 = 2) ∧
  (∀ n : ℕ, n ≥ 1 → (a (n + 1))^3 + 1 = (a n) * (a (n + 2))) :=
by sorry

end NUMINAMATH_CALUDE_unique_sequence_existence_l2335_233574


namespace NUMINAMATH_CALUDE_Q_subset_complement_P_l2335_233568

-- Define the sets P and Q
def P : Set ℝ := {x | x < 1}
def Q : Set ℝ := {x | x > 1}

-- Define the complement of P in the real numbers
def CₘP : Set ℝ := {x | x ≥ 1}

-- Theorem statement
theorem Q_subset_complement_P : Q ⊆ CₘP := by sorry

end NUMINAMATH_CALUDE_Q_subset_complement_P_l2335_233568


namespace NUMINAMATH_CALUDE_harry_pencils_left_l2335_233565

/-- Calculates the number of pencils left with Harry given Anna's pencils and Harry's lost pencils. -/
def pencils_left_with_harry (anna_pencils : ℕ) (harry_lost_pencils : ℕ) : ℕ :=
  2 * anna_pencils - harry_lost_pencils

/-- Proves that Harry has 81 pencils left given the problem conditions. -/
theorem harry_pencils_left : pencils_left_with_harry 50 19 = 81 := by
  sorry

end NUMINAMATH_CALUDE_harry_pencils_left_l2335_233565


namespace NUMINAMATH_CALUDE_math_department_candidates_l2335_233563

theorem math_department_candidates :
  ∀ (m : ℕ),
    (∃ (cs_candidates : ℕ),
      cs_candidates = 7 ∧
      (Nat.choose cs_candidates 2) * m = 84) →
    m = 4 :=
by sorry

end NUMINAMATH_CALUDE_math_department_candidates_l2335_233563


namespace NUMINAMATH_CALUDE_arithmetic_geometric_progression_y_value_l2335_233531

theorem arithmetic_geometric_progression_y_value
  (x y z : ℝ)
  (nonzero_x : x ≠ 0)
  (nonzero_y : y ≠ 0)
  (nonzero_z : z ≠ 0)
  (arithmetic_prog : 2 * y = x + z)
  (geometric_prog1 : ∃ r : ℝ, r ≠ 0 ∧ -y = r * (x + 1) ∧ z = r * (-y))
  (geometric_prog2 : ∃ s : ℝ, s ≠ 0 ∧ y = s * x ∧ z + 2 = s * y) :
  y = 12 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_progression_y_value_l2335_233531


namespace NUMINAMATH_CALUDE_min_value_when_a_is_one_a_range_when_f_2_gt_5_l2335_233540

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + a| + |x - a|

-- Part I
theorem min_value_when_a_is_one :
  ∀ x : ℝ, f 1 x ≥ 2 :=
sorry

-- Part II
theorem a_range_when_f_2_gt_5 :
  ∀ a : ℝ, f a 2 > 5 → a < -5/2 ∨ a > 5/2 :=
sorry

end NUMINAMATH_CALUDE_min_value_when_a_is_one_a_range_when_f_2_gt_5_l2335_233540


namespace NUMINAMATH_CALUDE_certain_number_operations_l2335_233503

theorem certain_number_operations (x : ℝ) : 
  ∃ (p q : ℕ), p < q ∧ ((x + 20) * 2 / 2 - 2 = x + 18) ∧ (x + 18 = (p : ℝ) / q * 88) := by
  sorry

end NUMINAMATH_CALUDE_certain_number_operations_l2335_233503


namespace NUMINAMATH_CALUDE_solve_system_with_equal_xy_l2335_233520

theorem solve_system_with_equal_xy (x y n : ℝ) 
  (eq1 : 5 * x - 4 * y = n)
  (eq2 : 3 * x + 5 * y = 8)
  (eq3 : x = y) :
  n = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_with_equal_xy_l2335_233520


namespace NUMINAMATH_CALUDE_journey_distance_l2335_233585

/-- Proves that a journey with given conditions results in a total distance of 560 km -/
theorem journey_distance (total_time : ℝ) (speed_first_half : ℝ) (speed_second_half : ℝ) 
  (h1 : total_time = 25)
  (h2 : speed_first_half = 21)
  (h3 : speed_second_half = 24) :
  let total_distance := total_time * (speed_first_half + speed_second_half) / 2
  total_distance = 560 := by
  sorry

end NUMINAMATH_CALUDE_journey_distance_l2335_233585


namespace NUMINAMATH_CALUDE_log_problem_trig_problem_l2335_233517

theorem log_problem (k : ℝ) (p : ℝ) 
  (h : Real.log 210 + Real.log k - Real.log 56 + Real.log 40 - Real.log 120 + Real.log 25 = p) : 
  p = 3 := by sorry

theorem trig_problem (A : ℝ) (q : ℝ) 
  (h1 : Real.sin A = 3 / 5) 
  (h2 : Real.cos A / Real.tan A = q / 15) : 
  q = 16 := by sorry

end NUMINAMATH_CALUDE_log_problem_trig_problem_l2335_233517


namespace NUMINAMATH_CALUDE_janet_snowball_percentage_l2335_233549

/-- Given that Janet makes 50 snowballs and her brother makes 150 snowballs,
    prove that Janet made 25% of the total snowballs. -/
theorem janet_snowball_percentage
  (janet_snowballs : ℕ)
  (brother_snowballs : ℕ)
  (h1 : janet_snowballs = 50)
  (h2 : brother_snowballs = 150) :
  (janet_snowballs : ℚ) / (janet_snowballs + brother_snowballs) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_janet_snowball_percentage_l2335_233549


namespace NUMINAMATH_CALUDE_largest_sum_proof_l2335_233556

theorem largest_sum_proof : 
  let sums : List ℚ := [1/4 + 1/9, 1/4 + 1/10, 1/4 + 1/11, 1/4 + 1/12, 1/4 + 1/13]
  (∀ x ∈ sums, x ≤ (1/4 + 1/9)) ∧ (1/4 + 1/9 = 13/36) :=
by sorry

end NUMINAMATH_CALUDE_largest_sum_proof_l2335_233556


namespace NUMINAMATH_CALUDE_cubic_function_values_l2335_233581

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^3 - 6 * a * x^2 + b

-- State the theorem
theorem cubic_function_values (a b : ℝ) (ha : a ≠ 0) :
  (∀ x ∈ Set.Icc (-1) 2, f a b x ≤ 3) ∧
  (∃ x ∈ Set.Icc (-1) 2, f a b x = 3) ∧
  (∀ x ∈ Set.Icc (-1) 2, f a b x ≥ -29) ∧
  (∃ x ∈ Set.Icc (-1) 2, f a b x = -29) →
  ((a = 2 ∧ b = 3) ∨ (a = -2 ∧ b = -29)) :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_values_l2335_233581


namespace NUMINAMATH_CALUDE_F_odd_and_increasing_l2335_233551

-- Define f(x) implicitly using the given condition
noncomputable def f : ℝ → ℝ := fun x => Real.exp (x * Real.log 2)

-- Define F(x) using f(x)
noncomputable def F : ℝ → ℝ := fun x => f x - 1 / f x

-- Theorem stating that F is odd and increasing
theorem F_odd_and_increasing :
  (∀ x : ℝ, F (-x) = -F x) ∧
  (∀ x y : ℝ, x < y → F x < F y) :=
by sorry

end NUMINAMATH_CALUDE_F_odd_and_increasing_l2335_233551


namespace NUMINAMATH_CALUDE_fruit_cost_proof_l2335_233578

/-- Given the cost of fruits, prove the cost of a different combination -/
theorem fruit_cost_proof (cost_six_apples_three_oranges : ℝ) 
                         (cost_one_apple : ℝ) : 
  cost_six_apples_three_oranges = 1.77 →
  cost_one_apple = 0.21 →
  2 * cost_one_apple + 5 * ((cost_six_apples_three_oranges - 6 * cost_one_apple) / 3) = 1.27 := by
  sorry

end NUMINAMATH_CALUDE_fruit_cost_proof_l2335_233578


namespace NUMINAMATH_CALUDE_hill_climb_speed_l2335_233561

/-- Proves that given a journey with an uphill climb taking 4 hours and a downhill descent
    taking 2 hours, if the average speed for the entire journey is 1.5 km/h,
    then the average speed for the uphill climb is 1.125 km/h. -/
theorem hill_climb_speed (distance : ℝ) (climb_time : ℝ) (descent_time : ℝ) 
    (average_speed : ℝ) (h1 : climb_time = 4) (h2 : descent_time = 2) 
    (h3 : average_speed = 1.5) :
  distance / climb_time = 1.125 := by
  sorry

end NUMINAMATH_CALUDE_hill_climb_speed_l2335_233561


namespace NUMINAMATH_CALUDE_time_saved_by_bike_l2335_233579

/-- Given that it takes Mike 98 minutes to walk to school and riding a bicycle saves him 64 minutes,
    prove that the time saved by Mike when riding a bicycle is 64 minutes. -/
theorem time_saved_by_bike (walking_time : ℕ) (time_saved : ℕ) 
  (h1 : walking_time = 98) 
  (h2 : time_saved = 64) : 
  time_saved = 64 := by
  sorry

end NUMINAMATH_CALUDE_time_saved_by_bike_l2335_233579


namespace NUMINAMATH_CALUDE_touching_circle_exists_l2335_233547

-- Define the rectangle
structure Rectangle where
  width : ℝ
  height : ℝ

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the configuration of circles
structure CircleConfiguration where
  rect : Rectangle
  circle1 : Circle
  circle2 : Circle
  circle3 : Circle

-- Define the property that circles touch each other and the rectangle sides
def circlesValidConfiguration (config : CircleConfiguration) : Prop :=
  -- Circles touch each other
  (config.circle1.center.1 + config.circle1.radius = config.circle2.center.1 - config.circle2.radius) ∧
  (config.circle2.center.1 + config.circle2.radius = config.circle3.center.1 - config.circle3.radius) ∧
  -- Circles touch the rectangle sides
  (config.circle1.center.2 = config.circle1.radius) ∧
  (config.circle2.center.2 = config.rect.height - config.circle2.radius) ∧
  (config.circle3.center.2 = config.circle3.radius)

-- Define the existence of a circle touching all three circles and one side of the rectangle
def existsTouchingCircle (config : CircleConfiguration) : Prop :=
  ∃ (x : ℝ), x > 0 ∧
    -- The new circle touches circle1 and circle2
    (x + config.circle1.radius)^2 + config.circle1.radius^2 = (x + config.circle2.radius)^2 + (config.circle2.center.2 - config.circle1.center.2)^2 ∧
    -- The new circle touches circle2 and circle3
    (x + config.circle2.radius)^2 + (config.rect.height - config.circle2.center.2 - x)^2 = (x + config.circle3.radius)^2 + (config.circle3.center.2 - x)^2

-- The theorem to be proved
theorem touching_circle_exists (config : CircleConfiguration) 
  (h1 : config.circle1.radius = 1)
  (h2 : config.circle2.radius = 3)
  (h3 : config.circle3.radius = 4)
  (h4 : circlesValidConfiguration config) :
  existsTouchingCircle config :=
sorry

end NUMINAMATH_CALUDE_touching_circle_exists_l2335_233547


namespace NUMINAMATH_CALUDE_solve_widgets_problem_l2335_233569

def widgets_problem (initial_widgets : ℕ) (total_money : ℕ) (price_reduction : ℕ) : Prop :=
  let initial_price := total_money / initial_widgets
  let new_price := initial_price - price_reduction
  let new_widgets := total_money / new_price
  new_widgets = 8

theorem solve_widgets_problem :
  widgets_problem 6 48 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_widgets_problem_l2335_233569


namespace NUMINAMATH_CALUDE_solve_for_y_l2335_233593

theorem solve_for_y (x y : ℤ) (h1 : x^2 + 5 = y - 8) (h2 : x = -7) : y = 62 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l2335_233593


namespace NUMINAMATH_CALUDE_number_of_cows_l2335_233524

/-- The number of cows in a field with a total of 200 animals, 56 sheep, and 104 goats. -/
theorem number_of_cows (total : ℕ) (sheep : ℕ) (goats : ℕ) (h1 : total = 200) (h2 : sheep = 56) (h3 : goats = 104) :
  total - sheep - goats = 40 := by
  sorry

end NUMINAMATH_CALUDE_number_of_cows_l2335_233524


namespace NUMINAMATH_CALUDE_min_value_of_x_plus_y_l2335_233588

theorem min_value_of_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : y + 9 * x = x * y) :
  x + y ≥ 16 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ y₀ + 9 * x₀ = x₀ * y₀ ∧ x₀ + y₀ = 16 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_x_plus_y_l2335_233588


namespace NUMINAMATH_CALUDE_five_by_five_uncoverable_l2335_233534

/-- A checkerboard that can be completely covered by dominoes. -/
structure CoverableCheckerboard where
  rows : ℕ
  cols : ℕ
  even_rows : Even rows
  even_cols : Even cols
  even_total : Even (rows * cols)

/-- A domino covers exactly two squares. -/
def domino_covers : ℕ := 2

/-- Theorem stating that a 5x5 checkerboard cannot be completely covered by dominoes. -/
theorem five_by_five_uncoverable :
  ¬ ∃ (c : CoverableCheckerboard), c.rows = 5 ∧ c.cols = 5 :=
sorry

end NUMINAMATH_CALUDE_five_by_five_uncoverable_l2335_233534


namespace NUMINAMATH_CALUDE_solution_to_equation_l2335_233514

theorem solution_to_equation : ∃ x y : ℤ, x - 3 * y = 1 ∧ x = -2 ∧ y = -1 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_equation_l2335_233514


namespace NUMINAMATH_CALUDE_min_value_of_sum_l2335_233516

theorem min_value_of_sum (x y : ℝ) : 
  x > 1 → 
  y > 1 → 
  2 * Real.log 2 = Real.log x + Real.log y → 
  x + y ≥ 200 ∧ ∃ x y, x > 1 ∧ y > 1 ∧ 2 * Real.log 2 = Real.log x + Real.log y ∧ x + y = 200 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_l2335_233516


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l2335_233544

theorem quadratic_inequality_solution_range (c : ℝ) : 
  (c > 0 ∧ ∃ x : ℝ, x^2 - 8*x + c < 0) ↔ (c > 0 ∧ c < 16) := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l2335_233544


namespace NUMINAMATH_CALUDE_prime_sum_10_product_21_l2335_233507

theorem prime_sum_10_product_21 (p q : ℕ) : 
  Prime p → Prime q → p ≠ q → p + q = 10 → p * q = 21 := by sorry

end NUMINAMATH_CALUDE_prime_sum_10_product_21_l2335_233507


namespace NUMINAMATH_CALUDE_tank_capacity_l2335_233500

theorem tank_capacity (C : ℚ) 
  (h1 : (3/4 : ℚ) * C + 4 = (7/8 : ℚ) * C) : C = 32 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l2335_233500


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2335_233512

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of specific terms in the sequence equals 120 -/
def SpecificSum (a : ℕ → ℝ) : Prop :=
  a 4 + a 6 + a 8 + a 10 + a 12 = 120

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h1 : ArithmeticSequence a) (h2 : SpecificSum a) : 
  a 9 - (1/2) * a 10 = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2335_233512


namespace NUMINAMATH_CALUDE_composition_fraction_l2335_233591

def f (x : ℝ) : ℝ := 3 * x + 2

def g (x : ℝ) : ℝ := 2 * x - 3

theorem composition_fraction : f (g (f 3)) / g (f (g 3)) = 59 / 35 := by
  sorry

end NUMINAMATH_CALUDE_composition_fraction_l2335_233591


namespace NUMINAMATH_CALUDE_quarter_circles_sum_limit_l2335_233545

theorem quarter_circles_sum_limit (D : ℝ) (h : D > 0) :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N,
    |2 * n * (π * D / (8 * n)) - (π * D / 4)| < ε :=
sorry

end NUMINAMATH_CALUDE_quarter_circles_sum_limit_l2335_233545


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_sum_l2335_233521

theorem geometric_sequence_common_ratio_sum 
  (k p q : ℝ) 
  (h1 : p ≠ q) 
  (h2 : k ≠ 0) 
  (h3 : k * p^2 - k * q^2 = 5 * (k * p - k * q)) : 
  p + q = 5 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_sum_l2335_233521


namespace NUMINAMATH_CALUDE_perfect_rectangle_theorem_l2335_233538

/-- Represents a perfect rectangle divided into squares -/
structure PerfectRectangle where
  squares : List ℕ
  is_perfect : squares.length > 0

/-- The specific perfect rectangle from the problem -/
def given_rectangle : PerfectRectangle where
  squares := [9, 16, 2, 5, 7, 25, 28, 33]
  is_perfect := by simp

/-- Checks if the list is sorted in ascending order -/
def is_sorted (l : List ℕ) : Prop :=
  ∀ i j, i < j → j < l.length → l[i]! ≤ l[j]!

/-- The main theorem to prove -/
theorem perfect_rectangle_theorem (rect : PerfectRectangle) :
  rect = given_rectangle →
  is_sorted (rect.squares.filter (λ x => x ≠ 9 ∧ x ≠ 16)) ∧
  (rect.squares.filter (λ x => x ≠ 9 ∧ x ≠ 16)).length = 6 :=
by sorry

end NUMINAMATH_CALUDE_perfect_rectangle_theorem_l2335_233538


namespace NUMINAMATH_CALUDE_candy_difference_l2335_233576

def frankie_candy : ℕ := 74
def max_candy : ℕ := 92

theorem candy_difference : max_candy - frankie_candy = 18 := by
  sorry

end NUMINAMATH_CALUDE_candy_difference_l2335_233576


namespace NUMINAMATH_CALUDE_arithmetic_computation_l2335_233519

theorem arithmetic_computation : 5 + 4 * (2 - 7)^2 = 105 := by sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l2335_233519


namespace NUMINAMATH_CALUDE_expected_value_of_coins_l2335_233529

/-- The expected value of coins coming up heads when flipping four coins simultaneously -/
theorem expected_value_of_coins (nickel quarter half_dollar dollar : ℕ) 
  (h_nickel : nickel = 5)
  (h_quarter : quarter = 25)
  (h_half_dollar : half_dollar = 50)
  (h_dollar : dollar = 100)
  (p_heads : ℚ)
  (h_p_heads : p_heads = 1 / 2) : 
  p_heads * (nickel + quarter + half_dollar + dollar : ℚ) = 90 := by
sorry

end NUMINAMATH_CALUDE_expected_value_of_coins_l2335_233529


namespace NUMINAMATH_CALUDE_monotonicity_intervals_l2335_233597

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - (3/2) * a * x^2 + (2*a^2 + a - 1) * x + 3

theorem monotonicity_intervals (a : ℝ) :
  (a = 2 → ∀ x y, x < y → f a x < f a y) ∧
  (a < 2 → (∀ x y, x < y ∧ y < 2*a - 1 → f a x < f a y) ∧
           (∀ x y, 2*a - 1 < x ∧ x < y ∧ y < a + 1 → f a x > f a y) ∧
           (∀ x y, a + 1 < x ∧ x < y → f a x < f a y)) ∧
  (a > 2 → (∀ x y, x < y ∧ y < a + 1 → f a x < f a y) ∧
           (∀ x y, a + 1 < x ∧ x < y ∧ y < 2*a - 1 → f a x > f a y) ∧
           (∀ x y, 2*a - 1 < x ∧ x < y → f a x < f a y)) :=
by sorry

end NUMINAMATH_CALUDE_monotonicity_intervals_l2335_233597


namespace NUMINAMATH_CALUDE_rectangle_length_ratio_l2335_233501

theorem rectangle_length_ratio (L B : ℝ) (L' : ℝ) (h1 : L > 0) (h2 : B > 0) : 
  (L' * (3 * B) = (3/2) * (L * B)) → L' / L = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_ratio_l2335_233501


namespace NUMINAMATH_CALUDE_tailor_buttons_l2335_233567

/-- The number of buttons purchased by a tailor -/
def total_buttons (green : ℕ) : ℕ :=
  let yellow := green + 10
  let blue := green - 5
  let red := 2 * (yellow + blue)
  let white := red + green
  let black := red - green
  green + yellow + blue + red + white + black

/-- Theorem: The tailor purchased 1385 buttons -/
theorem tailor_buttons : total_buttons 90 = 1385 := by
  sorry

end NUMINAMATH_CALUDE_tailor_buttons_l2335_233567


namespace NUMINAMATH_CALUDE_existence_of_many_prime_factors_l2335_233562

theorem existence_of_many_prime_factors (N : ℕ+) :
  ∃ n : ℕ+, ∃ p : Finset ℕ,
    (∀ q ∈ p, Nat.Prime q) ∧
    (Finset.card p ≥ N) ∧
    (∀ q ∈ p, q ∣ (n^2013 - n^20 + n^13 - 2013)) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_many_prime_factors_l2335_233562


namespace NUMINAMATH_CALUDE_square_sum_from_means_l2335_233570

theorem square_sum_from_means (x y : ℝ) 
  (h_arithmetic : (x + y) / 2 = 20) 
  (h_geometric : Real.sqrt (x * y) = Real.sqrt 110) : 
  x^2 + y^2 = 1380 := by
sorry

end NUMINAMATH_CALUDE_square_sum_from_means_l2335_233570


namespace NUMINAMATH_CALUDE_odd_functions_sufficient_not_necessary_l2335_233596

-- Define the real-valued functions
variable (f g h : ℝ → ℝ)

-- Define odd and even functions
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define the relationship between f, g, and h
def FunctionsRelated (f g h : ℝ → ℝ) : Prop := ∀ x, h x = f x * g x

-- Theorem statement
theorem odd_functions_sufficient_not_necessary :
  (∀ f g h, FunctionsRelated f g h → (IsOdd f ∧ IsOdd g → IsEven h)) ∧
  (∃ f g h, FunctionsRelated f g h ∧ IsEven h ∧ ¬(IsOdd f ∧ IsOdd g)) :=
sorry

end NUMINAMATH_CALUDE_odd_functions_sufficient_not_necessary_l2335_233596


namespace NUMINAMATH_CALUDE_gcd_lcm_pairs_l2335_233592

theorem gcd_lcm_pairs : 
  ∀ a b : ℕ, 
    a > 0 ∧ b > 0 →
    Nat.gcd a b = 24 ∧ Nat.lcm a b = 360 → 
    ((a = 24 ∧ b = 360) ∨ (a = 360 ∧ b = 24) ∨ (a = 72 ∧ b = 120) ∨ (a = 120 ∧ b = 72)) :=
by sorry

end NUMINAMATH_CALUDE_gcd_lcm_pairs_l2335_233592


namespace NUMINAMATH_CALUDE_pizza_toppings_theorem_l2335_233599

/-- Given a number of pizza flavors and total pizza varieties (including pizzas with and without additional toppings), 
    calculate the number of possible additional toppings. -/
def calculate_toppings (flavors : ℕ) (total_varieties : ℕ) : ℕ :=
  (total_varieties / flavors) - 1

/-- Theorem stating that with 4 pizza flavors and 16 total pizza varieties, 
    there are 3 possible additional toppings. -/
theorem pizza_toppings_theorem :
  calculate_toppings 4 16 = 3 := by
  sorry

#eval calculate_toppings 4 16

end NUMINAMATH_CALUDE_pizza_toppings_theorem_l2335_233599


namespace NUMINAMATH_CALUDE_accommodation_theorem_l2335_233535

/-- The number of ways to accommodate 6 people in 5 rooms --/
def accommodationWays : ℕ := 39600

/-- The number of ways to accommodate 6 people in 5 rooms with each room having at least one person --/
def waysWithAllRoomsOccupied : ℕ := 3600

/-- The number of ways to accommodate 6 people in 5 rooms with exactly one room left empty --/
def waysWithOneRoomEmpty : ℕ := 36000

/-- The number of people --/
def numPeople : ℕ := 6

/-- The number of rooms --/
def numRooms : ℕ := 5

theorem accommodation_theorem :
  accommodationWays = waysWithAllRoomsOccupied + waysWithOneRoomEmpty ∧
  numPeople = 6 ∧
  numRooms = 5 := by
  sorry

end NUMINAMATH_CALUDE_accommodation_theorem_l2335_233535


namespace NUMINAMATH_CALUDE_sin_cos_equivalence_l2335_233557

theorem sin_cos_equivalence (x : ℝ) : 
  Real.sin (2 * x + π / 3) = Real.cos (2 * (x - π / 12)) := by sorry

end NUMINAMATH_CALUDE_sin_cos_equivalence_l2335_233557


namespace NUMINAMATH_CALUDE_vector_subtraction_l2335_233584

theorem vector_subtraction (u v : Fin 3 → ℝ) 
  (hu : u = ![-3, 5, 2]) 
  (hv : v = ![1, -1, 3]) : 
  u - 2 • v = ![-5, 7, -4] := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_l2335_233584


namespace NUMINAMATH_CALUDE_k_range_for_not_in_second_quadrant_l2335_233527

/-- A linear function that does not pass through the second quadrant -/
structure LinearFunctionNotInSecondQuadrant where
  k : ℝ
  not_in_second_quadrant : ∀ x y : ℝ, y = k * x - k + 3 → ¬(x < 0 ∧ y > 0)

/-- The range of k for a linear function not passing through the second quadrant -/
theorem k_range_for_not_in_second_quadrant (f : LinearFunctionNotInSecondQuadrant) : f.k ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_k_range_for_not_in_second_quadrant_l2335_233527


namespace NUMINAMATH_CALUDE_dave_candy_pieces_l2335_233525

/-- Calculates the number of candy pieces Dave has left after giving some boxes away. -/
def candyPiecesLeft (initialBoxes : ℕ) (boxesGivenAway : ℕ) (piecesPerBox : ℕ) : ℕ :=
  (initialBoxes - boxesGivenAway) * piecesPerBox

/-- Proves that Dave has 21 pieces of candy left. -/
theorem dave_candy_pieces : 
  candyPiecesLeft 12 5 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_dave_candy_pieces_l2335_233525


namespace NUMINAMATH_CALUDE_difference_of_squares_640_360_l2335_233515

theorem difference_of_squares_640_360 : 640^2 - 360^2 = 280000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_640_360_l2335_233515


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l2335_233511

theorem cylinder_surface_area (r h : ℝ) (hr : r = 1) (hh : h = 1) :
  2 * Real.pi * r * (r + h) = 4 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l2335_233511
