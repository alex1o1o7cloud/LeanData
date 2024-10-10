import Mathlib

namespace pure_imaginary_complex_number_l1976_197641

theorem pure_imaginary_complex_number (m : ℝ) : 
  let z : ℂ := (m^2 - m - 2) + (m + 1) * Complex.I
  (z.re = 0 ∧ z.im ≠ 0) → m = 2 := by sorry

end pure_imaginary_complex_number_l1976_197641


namespace range_of_a_l1976_197662

open Set

def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := x^2 - 6*x + 8 > 0

def sufficient_not_necessary (P Q : Set ℝ) : Prop :=
  P ⊂ Q ∧ P ≠ Q

theorem range_of_a (a : ℝ) :
  (a > 0) →
  (sufficient_not_necessary {x | p x a} {x | q x}) →
  (a ≥ 4 ∨ (0 < a ∧ a ≤ 2/3)) :=
by sorry

end range_of_a_l1976_197662


namespace division_result_l1976_197689

theorem division_result : (4.036 : ℝ) / 0.04 = 100.9 := by
  sorry

end division_result_l1976_197689


namespace hockey_league_games_l1976_197617

theorem hockey_league_games (n : ℕ) (k : ℕ) (h1 : n = 18) (h2 : k = 10) :
  (n * (n - 1) / 2) * k = 1530 :=
by sorry

end hockey_league_games_l1976_197617


namespace sum_of_repeating_decimals_five_seven_l1976_197629

/-- Represents a repeating decimal with a single digit repeating infinitely after the decimal point. -/
def RepeatingDecimal (n : ℕ) : ℚ := n / 9

theorem sum_of_repeating_decimals_five_seven :
  RepeatingDecimal 5 + RepeatingDecimal 7 = 4 / 3 := by sorry

end sum_of_repeating_decimals_five_seven_l1976_197629


namespace triangle_trigonometric_identity_l1976_197670

theorem triangle_trigonometric_identity (A B C : Real) 
  (h : A + B + C = Real.pi) : 
  Real.sin A ^ 2 + Real.sin B ^ 2 + Real.sin C ^ 2 - 
  2 * Real.cos A * Real.cos B * Real.cos C = 2 := by
  sorry

end triangle_trigonometric_identity_l1976_197670


namespace rational_number_conditions_l1976_197657

theorem rational_number_conditions (a b : ℚ) : 
  a ≠ 0 → b ≠ 0 → abs a = a → abs b = -b → a + b < 0 → 
  ∃ (a b : ℚ), a = 1 ∧ b = -2 := by
  sorry

end rational_number_conditions_l1976_197657


namespace problem_statement_l1976_197683

theorem problem_statement (m n : ℝ) (h : 3 * m - n = 1) : 
  9 * m^2 - n^2 - 2 * n = 1 := by
sorry

end problem_statement_l1976_197683


namespace min_value_at_one_third_l1976_197676

def y (x : ℝ) : ℝ := |x - 1| + |2*x - 1| + |3*x - 1| + |4*x - 1| + |5*x - 1|

theorem min_value_at_one_third :
  ∀ x : ℝ, y (1/3 : ℝ) ≤ y x := by
  sorry

end min_value_at_one_third_l1976_197676


namespace function_inequality_l1976_197614

open Real

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x ∈ Set.Ioo (-π/2) (π/2), (deriv f x) * cos x + f x * sin x > 0) :
  Real.sqrt 2 * f (-π/3) < f (-π/4) := by
sorry

end function_inequality_l1976_197614


namespace geometric_sequence_cosine_l1976_197678

/-- 
Given a geometric sequence {an} with common ratio √2,
prove that if sn(a7a8) = 3/5, then cos(2a5) = 7/25
-/
theorem geometric_sequence_cosine (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n * Real.sqrt 2) →  -- Common ratio is √2
  (∃ sn : ℝ, sn * (a 7 * a 8) = 3/5) →    -- sn(a7a8) = 3/5
  Real.cos (2 * a 5) = 7/25 :=             -- cos(2a5) = 7/25
by sorry

end geometric_sequence_cosine_l1976_197678


namespace maggie_earnings_l1976_197611

def subscriptions_to_parents : ℕ := 4
def subscriptions_to_grandfather : ℕ := 1
def subscriptions_to_next_door : ℕ := 2
def subscriptions_to_another : ℕ := 4

def base_pay : ℚ := 5
def family_bonus : ℚ := 2
def neighbor_bonus : ℚ := 1
def additional_bonus_base : ℚ := 10
def additional_bonus_per_extra : ℚ := 0.5

def total_subscriptions : ℕ := subscriptions_to_parents + subscriptions_to_grandfather + subscriptions_to_next_door + subscriptions_to_another

def family_subscriptions : ℕ := subscriptions_to_parents + subscriptions_to_grandfather
def neighbor_subscriptions : ℕ := subscriptions_to_next_door + subscriptions_to_another

def base_earnings : ℚ := base_pay * total_subscriptions
def family_bonus_earnings : ℚ := family_bonus * family_subscriptions
def neighbor_bonus_earnings : ℚ := neighbor_bonus * neighbor_subscriptions

def additional_bonus : ℚ :=
  if total_subscriptions > 10
  then additional_bonus_base + additional_bonus_per_extra * (total_subscriptions - 10)
  else 0

def total_earnings : ℚ := base_earnings + family_bonus_earnings + neighbor_bonus_earnings + additional_bonus

theorem maggie_earnings : total_earnings = 81.5 := by sorry

end maggie_earnings_l1976_197611


namespace angle_A_is_60_degrees_b_plus_c_range_l1976_197673

/- Define a triangle with sides a, b, c and angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/- Define the condition √3a*sin(C) + a*cos(C) = c + b -/
def condition (t : Triangle) : Prop :=
  Real.sqrt 3 * t.a * Real.sin t.C + t.a * Real.cos t.C = t.c + t.b

/- Theorem 1: If the condition holds, then angle A = 60° -/
theorem angle_A_is_60_degrees (t : Triangle) (h : condition t) : t.A = 60 * π / 180 := by
  sorry

/- Theorem 2: If a = √3 and the condition holds, then √3 < b + c ≤ 2√3 -/
theorem b_plus_c_range (t : Triangle) (h1 : t.a = Real.sqrt 3) (h2 : condition t) :
  Real.sqrt 3 < t.b + t.c ∧ t.b + t.c ≤ 2 * Real.sqrt 3 := by
  sorry

end angle_A_is_60_degrees_b_plus_c_range_l1976_197673


namespace ratio_of_a_to_c_l1976_197604

theorem ratio_of_a_to_c (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 4 / 3)
  (hdb : d / b = 1 / 8) :
  a / c = 7.5 := by
sorry

end ratio_of_a_to_c_l1976_197604


namespace invalid_formula_l1976_197602

def sequence_formula_a (n : ℕ) : ℚ := n
def sequence_formula_b (n : ℕ) : ℚ := n^3 - 6*n^2 + 12*n - 6
def sequence_formula_c (n : ℕ) : ℚ := (1/2)*n^2 - (1/2)*n + 1
def sequence_formula_d (n : ℕ) : ℚ := 6 / (n^2 - 6*n + 11)

theorem invalid_formula :
  (sequence_formula_a 1 = 1 ∧ sequence_formula_a 2 = 2 ∧ sequence_formula_a 3 = 3) ∧
  (sequence_formula_b 1 = 1 ∧ sequence_formula_b 2 = 2 ∧ sequence_formula_b 3 = 3) ∧
  (sequence_formula_d 1 = 1 ∧ sequence_formula_d 2 = 2 ∧ sequence_formula_d 3 = 3) ∧
  ¬(sequence_formula_c 1 = 1 ∧ sequence_formula_c 2 = 2 ∧ sequence_formula_c 3 = 3) :=
by sorry

end invalid_formula_l1976_197602


namespace unique_positive_solution_l1976_197669

theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ x * ↑(⌊x⌋) = 162 ∧ x = 13.5 := by sorry

end unique_positive_solution_l1976_197669


namespace geometric_sequence_property_l1976_197652

/-- A geometric sequence with positive first term and a_2 * a_4 = 25 has a_3 = 5 -/
theorem geometric_sequence_property (a : ℕ → ℝ) (h_geom : ∃ q : ℝ, ∀ n, a (n + 1) = a n * q)
  (h_pos : a 1 > 0) (h_prod : a 2 * a 4 = 25) : a 3 = 5 := by
  sorry

end geometric_sequence_property_l1976_197652


namespace triangle_special_area_implies_30_degree_angle_l1976_197633

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if the area of the triangle is (a² + b² - c²) / (4√3),
    then angle C equals 30°. -/
theorem triangle_special_area_implies_30_degree_angle
  (a b c : ℝ)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_area : (a^2 + b^2 - c^2) / (4 * Real.sqrt 3) = 1/2 * a * b * Real.sin (Real.pi / 6)) :
  ∃ A B C : ℝ,
    0 < A ∧ A < Real.pi ∧
    0 < B ∧ B < Real.pi ∧
    0 < C ∧ C < Real.pi ∧
    A + B + C = Real.pi ∧
    C = Real.pi / 6 := by
  sorry


end triangle_special_area_implies_30_degree_angle_l1976_197633


namespace homework_is_duration_l1976_197679

-- Define a type for time expressions
inductive TimeExpression
  | PointInTime (description : String)
  | Duration (description : String)

-- Define the given options
def option_a : TimeExpression := TimeExpression.PointInTime "Get up at 6:30"
def option_b : TimeExpression := TimeExpression.PointInTime "School ends at 3:40"
def option_c : TimeExpression := TimeExpression.Duration "It took 30 minutes to do the homework"

-- Define a function to check if a TimeExpression represents a duration
def is_duration (expr : TimeExpression) : Prop :=
  match expr with
  | TimeExpression.Duration _ => True
  | _ => False

-- Theorem to prove
theorem homework_is_duration :
  is_duration option_c ∧ ¬is_duration option_a ∧ ¬is_duration option_b :=
sorry

end homework_is_duration_l1976_197679


namespace belle_weekly_treat_cost_l1976_197688

def cost_brand_a : ℚ := 0.25
def cost_brand_b : ℚ := 0.35
def cost_small_rawhide : ℚ := 1
def cost_large_rawhide : ℚ := 1.5

def odd_day_cost : ℚ :=
  3 * cost_brand_a + 2 * cost_brand_b + cost_small_rawhide + cost_large_rawhide

def even_day_cost : ℚ :=
  4 * cost_brand_a + 2 * cost_small_rawhide

def days_in_week : ℕ := 7
def odd_days_in_week : ℕ := 4
def even_days_in_week : ℕ := 3

theorem belle_weekly_treat_cost :
  odd_days_in_week * odd_day_cost + even_days_in_week * even_day_cost = 24.8 := by
  sorry

end belle_weekly_treat_cost_l1976_197688


namespace sin_product_equals_one_sixteenth_sin_cos_sum_equals_three_fourths_plus_quarter_sin_seventy_l1976_197620

-- Part 1
theorem sin_product_equals_one_sixteenth :
  Real.sin (6 * π / 180) * Real.sin (42 * π / 180) * Real.sin (66 * π / 180) * Real.sin (78 * π / 180) = 1 / 16 := by
  sorry

-- Part 2
theorem sin_cos_sum_equals_three_fourths_plus_quarter_sin_seventy :
  Real.sin (20 * π / 180) ^ 2 + Real.cos (50 * π / 180) ^ 2 + Real.sin (20 * π / 180) * Real.cos (50 * π / 180) =
  3 / 4 + (1 / 4) * Real.sin (70 * π / 180) := by
  sorry

end sin_product_equals_one_sixteenth_sin_cos_sum_equals_three_fourths_plus_quarter_sin_seventy_l1976_197620


namespace ratio_problem_l1976_197664

theorem ratio_problem : ∀ x : ℚ, (20 : ℚ) / 1 = x / 10 → x = 200 := by
  sorry

end ratio_problem_l1976_197664


namespace sandy_sums_theorem_l1976_197631

theorem sandy_sums_theorem (correct_marks : ℕ → ℕ) (incorrect_marks : ℕ → ℕ) 
  (total_marks : ℕ) (correct_sums : ℕ) :
  (correct_marks = λ n => 3 * n) →
  (incorrect_marks = λ n => 2 * n) →
  (total_marks = 50) →
  (correct_sums = 22) →
  (∃ (total_sums : ℕ), 
    total_sums = correct_sums + (total_sums - correct_sums) ∧
    total_marks = correct_marks correct_sums - incorrect_marks (total_sums - correct_sums) ∧
    total_sums = 30) :=
by sorry

end sandy_sums_theorem_l1976_197631


namespace f_satisfies_equation_l1976_197663

/-- The function f satisfying the given functional equation -/
noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ -0.5 then 1 / (x + 0.5) else 0.5

/-- Theorem stating that f satisfies the functional equation for all real x -/
theorem f_satisfies_equation : ∀ x : ℝ, f x - (x - 0.5) * f (-x - 1) = 1 := by
  sorry

end f_satisfies_equation_l1976_197663


namespace rationalization_sum_l1976_197672

/-- Represents a cube root expression of the form (a * ∛b) / c --/
structure CubeRootExpression where
  a : ℤ
  b : ℕ
  c : ℕ
  c_pos : c > 0
  b_not_perfect_cube : ∀ (p : ℕ), Prime p → ¬(p^3 ∣ b)

/-- Rationalizes the denominator of 5 / (3 * ∛7) --/
def rationalize_denominator : CubeRootExpression :=
  { a := 5
    b := 49
    c := 21
    c_pos := by sorry
    b_not_perfect_cube := by sorry }

/-- The sum of a, b, and c in the rationalized expression --/
def sum_of_parts (expr : CubeRootExpression) : ℤ :=
  expr.a + expr.b + expr.c

theorem rationalization_sum :
  sum_of_parts rationalize_denominator = 75 := by sorry

end rationalization_sum_l1976_197672


namespace shaded_region_perimeter_l1976_197687

theorem shaded_region_perimeter (r : ℝ) (h : r = 7) : 
  2 * r + 3 * π * r / 2 = 14 + 10.5 * π := by sorry

end shaded_region_perimeter_l1976_197687


namespace min_value_expression_l1976_197660

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (m : ℝ), m = Real.sqrt 3 ∧ 
  (∀ (x y : ℝ) (hx : x > 0) (hy : y > 0), 
    x^2 + x*y + y^2 + 1/(x+y)^2 ≥ m) ∧
  (∃ (u v : ℝ) (hu : u > 0) (hv : v > 0), 
    u^2 + u*v + v^2 + 1/(u+v)^2 = m) :=
sorry

end min_value_expression_l1976_197660


namespace inequalities_truth_l1976_197638

theorem inequalities_truth (a b c d : ℝ) : 
  (a^2 + b^2 + c^2 ≥ a*b + b*c + a*c) ∧ 
  (a*(1 - a) ≤ (1/4 : ℝ)) ∧ 
  ((a^2 + b^2)*(c^2 + d^2) ≥ (a*c + b*d)^2) ∧
  ¬(∀ (a b : ℝ), a/b + b/a ≥ 2) := by
  sorry

end inequalities_truth_l1976_197638


namespace judgment_not_basic_structure_l1976_197665

/-- Represents the basic structures of flowcharts in algorithms -/
inductive FlowchartStructure
  | Sequential
  | Selection
  | Loop
  | Judgment

/-- The set of basic flowchart structures -/
def BasicStructures : Set FlowchartStructure :=
  {FlowchartStructure.Sequential, FlowchartStructure.Selection, FlowchartStructure.Loop}

/-- Theorem: The judgment structure is not one of the three basic structures of flowcharts -/
theorem judgment_not_basic_structure :
  FlowchartStructure.Judgment ∉ BasicStructures :=
by sorry

end judgment_not_basic_structure_l1976_197665


namespace max_value_expression_l1976_197647

theorem max_value_expression (x y z : ℝ) 
  (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) :
  (1 / ((2 - x) * (2 - y) * (2 - z)) + 1 / ((2 + x) * (2 + y) * (2 + z))) ≤ 12 / 27 ∧
  (1 / ((2 - 1) * (2 - 1) * (2 - 1)) + 1 / ((2 + 1) * (2 + 1) * (2 + 1))) = 12 / 27 :=
by sorry

end max_value_expression_l1976_197647


namespace quadratic_roots_relations_l1976_197682

theorem quadratic_roots_relations (a : ℝ) :
  let x₁ : ℝ := (1 + Real.sqrt (5 - 4*a)) / 2
  let x₂ : ℝ := (1 - Real.sqrt (5 - 4*a)) / 2
  (x₁*x₂ + x₁ + x₂ - a = 0) ∧ (x₁*x₂ - a*(x₁ + x₂) + 1 = 0) :=
by sorry

end quadratic_roots_relations_l1976_197682


namespace solution_numbers_l1976_197690

/-- Sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- The theorem statement -/
theorem solution_numbers : 
  {n : ℕ | n + sumOfDigits n = 2021} = {2014, 1996} := by sorry

end solution_numbers_l1976_197690


namespace cone_slant_height_l1976_197608

/-- The slant height of a cone with base radius 6 cm and lateral surface sector angle 240° is 9 cm. -/
theorem cone_slant_height (base_radius : ℝ) (sector_angle : ℝ) (slant_height : ℝ) : 
  base_radius = 6 →
  sector_angle = 240 →
  slant_height = (360 / sector_angle) * base_radius →
  slant_height = 9 := by
sorry

end cone_slant_height_l1976_197608


namespace hyperbola_eccentricity_l1976_197696

/-- The eccentricity of a hyperbola with equation x²/5 - y²/4 = 1 is 3√5/5 -/
theorem hyperbola_eccentricity : 
  let a : ℝ := Real.sqrt 5
  let b : ℝ := 2
  let c : ℝ := 3
  let e : ℝ := c / a
  (∀ x y : ℝ, x^2 / 5 - y^2 / 4 = 1) →
  e = 3 * Real.sqrt 5 / 5 := by
sorry

end hyperbola_eccentricity_l1976_197696


namespace inequality_proof_l1976_197654

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : a + b + c + a*b + b*c + c*a + a*b*c = 7) :
  Real.sqrt (a^2 + b^2 + 2) + Real.sqrt (b^2 + c^2 + 2) + Real.sqrt (c^2 + a^2 + 2) ≥ 6 := by
  sorry

end inequality_proof_l1976_197654


namespace journey_distance_is_20km_l1976_197600

/-- Represents a round trip journey with a horizontal section and a hill. -/
structure Journey where
  total_time : ℝ
  horizontal_speed : ℝ
  uphill_speed : ℝ
  downhill_speed : ℝ

/-- Calculates the total distance covered in a journey. -/
def total_distance (j : Journey) : ℝ :=
  j.total_time * j.horizontal_speed

/-- Theorem stating that for the given journey parameters, the total distance is 20 km. -/
theorem journey_distance_is_20km (j : Journey)
  (h_time : j.total_time = 5)
  (h_horizontal : j.horizontal_speed = 4)
  (h_uphill : j.uphill_speed = 3)
  (h_downhill : j.downhill_speed = 6) :
  total_distance j = 20 := by
  sorry

#eval total_distance { total_time := 5, horizontal_speed := 4, uphill_speed := 3, downhill_speed := 6 }

end journey_distance_is_20km_l1976_197600


namespace simplify_expressions_l1976_197642

theorem simplify_expressions :
  (∃ x y : ℝ, x^2 = 8 ∧ y^2 = 3 ∧ 
    x + 2*y - (3*y - Real.sqrt 2) = 3*Real.sqrt 2 - y) ∧
  (∃ a b : ℝ, a^2 = 2 ∧ b^2 = 3 ∧ 
    (a - b)^2 + 2*Real.sqrt (1/3) * 3*a = 5) := by
  sorry

end simplify_expressions_l1976_197642


namespace total_eggs_theorem_l1976_197699

/-- Represents the number of eggs used for a family member's breakfast on a given day type --/
structure EggUsage where
  children : ℕ  -- eggs per child
  husband : ℕ   -- eggs for husband
  lisa : ℕ      -- eggs for Lisa

/-- Represents the egg usage patterns for different days of the week --/
structure WeeklyEggUsage where
  monday_tuesday : EggUsage
  wednesday : EggUsage
  thursday : EggUsage
  friday : EggUsage

/-- Calculates the total eggs used in a year based on the given parameters --/
def total_eggs_per_year (
  num_children : ℕ
  ) (weekly_usage : WeeklyEggUsage
  ) (num_holidays : ℕ
  ) (holiday_usage : EggUsage
  ) : ℕ :=
  sorry

/-- The main theorem stating the total number of eggs used in a year --/
theorem total_eggs_theorem : 
  total_eggs_per_year 
    4  -- number of children
    {  -- weekly egg usage
      monday_tuesday := { children := 2, husband := 3, lisa := 2 },
      wednesday := { children := 3, husband := 4, lisa := 3 },
      thursday := { children := 1, husband := 2, lisa := 1 },
      friday := { children := 2, husband := 3, lisa := 2 }
    }
    8  -- number of holidays
    { children := 2, husband := 2, lisa := 2 }  -- holiday egg usage
  = 3476 := by
  sorry

end total_eggs_theorem_l1976_197699


namespace sum_of_variables_l1976_197675

theorem sum_of_variables (x y z : ℚ) 
  (eq1 : y + z = 20 - 5*x)
  (eq2 : x + z = -18 - 5*y)
  (eq3 : x + y = 10 - 5*z) :
  3*x + 3*y + 3*z = 36/7 := by sorry

end sum_of_variables_l1976_197675


namespace triangle_height_proof_l1976_197691

/-- Triangle ABC with vertices A(-2,10), B(2,0), and C(10,0) -/
structure Triangle :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)

/-- A vertical line intersecting AC at R and BC at S -/
structure IntersectingLine :=
  (R : ℝ × ℝ)
  (S : ℝ × ℝ)

/-- The problem statement -/
theorem triangle_height_proof 
  (ABC : Triangle)
  (RS : IntersectingLine)
  (h1 : ABC.A = (-2, 10))
  (h2 : ABC.B = (2, 0))
  (h3 : ABC.C = (10, 0))
  (h4 : RS.R.1 = RS.S.1)  -- R and S have the same x-coordinate (vertical line)
  (h5 : RS.S.2 = 0)  -- S lies on BC (y-coordinate is 0)
  (h6 : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ RS.R = (1 - t) • ABC.A + t • ABC.C)  -- R lies on AC
  (h7 : (1/2) * |RS.R.2| * 8 = 24)  -- Area of RSC is 24
  : RS.R.2 = 6 := by sorry

end triangle_height_proof_l1976_197691


namespace baseball_tickets_sold_l1976_197628

theorem baseball_tickets_sold (fair_tickets : ℕ) (baseball_tickets : ℕ) : 
  fair_tickets = 25 →
  2 * fair_tickets + 6 = baseball_tickets →
  baseball_tickets = 56 := by
  sorry

end baseball_tickets_sold_l1976_197628


namespace science_marks_calculation_l1976_197668

/-- Calculates the marks scored in science given the total marks and marks in other subjects. -/
def science_marks (total : ℕ) (music : ℕ) (social_studies : ℕ) : ℕ :=
  total - (music + social_studies + music / 2)

/-- Theorem stating that given the specific marks, the science marks must be 70. -/
theorem science_marks_calculation :
  science_marks 275 80 85 = 70 := by
  sorry

end science_marks_calculation_l1976_197668


namespace correct_change_calculation_l1976_197685

/-- The change to be returned when mailing items with given costs and payment -/
def change_to_return (cost1 cost2 payment : ℚ) : ℚ :=
  payment - (cost1 + cost2)

/-- Theorem stating that the change to be returned is 1.2 yuan given the specific costs and payment -/
theorem correct_change_calculation :
  change_to_return (1.6) (12.2) (15) = (1.2) := by
  sorry

end correct_change_calculation_l1976_197685


namespace opposite_of_negative_one_half_l1976_197622

theorem opposite_of_negative_one_half : 
  -((-1 : ℚ) / 2) = 1 / 2 := by sorry

end opposite_of_negative_one_half_l1976_197622


namespace binomial_coefficient_25_7_l1976_197632

theorem binomial_coefficient_25_7 
  (h1 : Nat.choose 23 5 = 33649)
  (h2 : Nat.choose 23 6 = 42504)
  (h3 : Nat.choose 23 7 = 33649) : 
  Nat.choose 25 7 = 152306 := by
  sorry

end binomial_coefficient_25_7_l1976_197632


namespace square_sum_given_conditions_l1976_197623

theorem square_sum_given_conditions (x y : ℝ) 
  (h1 : (x + y)^2 = 4) 
  (h2 : x * y = -1) : 
  x^2 + y^2 = 6 := by
sorry

end square_sum_given_conditions_l1976_197623


namespace cubic_equation_roots_l1976_197637

theorem cubic_equation_roots (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h : a * b^2 + 1 = 0) :
  let f := fun x : ℝ => x / a + x^2 / b + x^3 / c - b * c
  (c > 0 → (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ f x₃ = 0)) ∧
  (c < 0 → (∃! x : ℝ, f x = 0)) :=
by sorry

end cubic_equation_roots_l1976_197637


namespace smallest_positive_integer_satisfying_congruences_l1976_197612

theorem smallest_positive_integer_satisfying_congruences : 
  ∃ (x : ℕ), x > 0 ∧ 
  (42 * x + 14) % 26 = 4 ∧ 
  x % 5 = 3 ∧
  (∀ (y : ℕ), y > 0 ∧ (42 * y + 14) % 26 = 4 ∧ y % 5 = 3 → x ≤ y) ∧
  x = 38 := by
  sorry

end smallest_positive_integer_satisfying_congruences_l1976_197612


namespace swim_meet_transport_theorem_l1976_197619

/-- Represents the transportation setup for the swimming club's trip --/
structure SwimMeetTransport where
  num_cars : Nat
  num_vans : Nat
  people_per_car : Nat
  max_people_per_car : Nat
  max_people_per_van : Nat
  additional_capacity : Nat

/-- Calculates the number of people in each van --/
def people_per_van (t : SwimMeetTransport) : Nat :=
  let total_capacity := t.num_cars * t.max_people_per_car + t.num_vans * t.max_people_per_van
  let actual_people := total_capacity - t.additional_capacity
  let people_in_cars := t.num_cars * t.people_per_car
  let people_in_vans := actual_people - people_in_cars
  people_in_vans / t.num_vans

/-- Theorem stating that the number of people in each van is 3 --/
theorem swim_meet_transport_theorem (t : SwimMeetTransport) 
  (h1 : t.num_cars = 2)
  (h2 : t.num_vans = 3)
  (h3 : t.people_per_car = 5)
  (h4 : t.max_people_per_car = 6)
  (h5 : t.max_people_per_van = 8)
  (h6 : t.additional_capacity = 17) :
  people_per_van t = 3 := by
  sorry

#eval people_per_van { 
  num_cars := 2, 
  num_vans := 3, 
  people_per_car := 5, 
  max_people_per_car := 6, 
  max_people_per_van := 8, 
  additional_capacity := 17 
}

end swim_meet_transport_theorem_l1976_197619


namespace largest_three_digit_multiple_of_8_with_digit_sum_24_l1976_197605

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem largest_three_digit_multiple_of_8_with_digit_sum_24 :
  ∀ n : ℕ, is_three_digit n → n % 8 = 0 → sum_of_digits n = 24 → n ≤ 888 :=
by sorry

end largest_three_digit_multiple_of_8_with_digit_sum_24_l1976_197605


namespace value_of_expression_l1976_197610

theorem value_of_expression (x : ℝ) (h : x^2 + 3*x + 5 = 11) : 
  3*x^2 + 9*x + 12 = 30 := by
sorry

end value_of_expression_l1976_197610


namespace max_number_bound_l1976_197618

/-- Represents an arc on the circle with two natural numbers -/
structure Arc where
  a : ℕ
  b : ℕ

/-- Represents the circle with 1000 arcs -/
def Circle := Fin 1000 → Arc

/-- The condition that the sum of numbers on each arc is divisible by the product of numbers on the next arc -/
def valid_circle (c : Circle) : Prop :=
  ∀ i : Fin 1000, (c i).a + (c i).b ∣ (c (i + 1)).a * (c (i + 1)).b

/-- The theorem stating that the maximum number on any arc is at most 2001 -/
theorem max_number_bound (c : Circle) (h : valid_circle c) :
  ∀ i : Fin 1000, (c i).a ≤ 2001 ∧ (c i).b ≤ 2001 :=
sorry

end max_number_bound_l1976_197618


namespace chess_club_election_l1976_197666

theorem chess_club_election (total_candidates : ℕ) (officer_positions : ℕ) (past_officers : ℕ) :
  total_candidates = 20 →
  officer_positions = 6 →
  past_officers = 8 →
  (Nat.choose total_candidates officer_positions - 
   Nat.choose (total_candidates - past_officers) officer_positions) = 37836 :=
by sorry

end chess_club_election_l1976_197666


namespace intersection_equals_open_interval_l1976_197606

-- Define the sets A and B
def A : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^(1/3)}
def B : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.log (x - 1)}

-- Define the open interval (1, +∞)
def open_interval : Set ℝ := {x : ℝ | x > 1}

-- Theorem statement
theorem intersection_equals_open_interval : A ∩ B = open_interval := by
  sorry

end intersection_equals_open_interval_l1976_197606


namespace jessica_bank_balance_l1976_197639

theorem jessica_bank_balance (B : ℝ) : 
  B - 400 = (3/5) * B → 
  B - 400 + (1/4) * (B - 400) = 750 := by
sorry

end jessica_bank_balance_l1976_197639


namespace greatest_x_quadratic_inequality_l1976_197607

theorem greatest_x_quadratic_inequality :
  ∃ (x : ℝ), x^2 - 6*x + 8 ≤ 0 ∧
  ∀ (y : ℝ), y^2 - 6*y + 8 ≤ 0 → y ≤ x :=
by
  -- The proof goes here
  sorry

end greatest_x_quadratic_inequality_l1976_197607


namespace julias_initial_money_l1976_197681

theorem julias_initial_money (initial_money : ℝ) : 
  initial_money / 2 - (initial_money / 2) / 4 = 15 → initial_money = 40 := by
  sorry

end julias_initial_money_l1976_197681


namespace salesperson_earnings_theorem_l1976_197650

/-- Represents the earnings of a salesperson based on their sales. -/
structure SalespersonEarnings where
  sales : ℕ
  earnings : ℝ

/-- Represents the direct proportionality between sales and earnings. -/
def directlyProportional (e1 e2 : SalespersonEarnings) : Prop :=
  e1.sales * e2.earnings = e2.sales * e1.earnings

/-- Theorem: If earnings are directly proportional to sales, and a salesperson
    earns $180 for 15 sales, then they will earn $240 for 20 sales. -/
theorem salesperson_earnings_theorem
  (e1 e2 : SalespersonEarnings)
  (h1 : directlyProportional e1 e2)
  (h2 : e1.sales = 15)
  (h3 : e1.earnings = 180)
  (h4 : e2.sales = 20) :
  e2.earnings = 240 := by
  sorry


end salesperson_earnings_theorem_l1976_197650


namespace find_natural_number_A_l1976_197648

theorem find_natural_number_A : ∃ A : ℕ, 
  A > 0 ∧ 
  312 % A = 2 * (270 % A) ∧ 
  270 % A = 2 * (211 % A) ∧ 
  A = 19 := by
  sorry

end find_natural_number_A_l1976_197648


namespace team_captain_selection_l1976_197658

def total_team_size : ℕ := 15
def shortlisted_size : ℕ := 5
def captains_to_choose : ℕ := 4

theorem team_captain_selection :
  (Nat.choose total_team_size captains_to_choose) -
  (Nat.choose (total_team_size - shortlisted_size) captains_to_choose) = 1155 :=
by sorry

end team_captain_selection_l1976_197658


namespace bowl_capacity_sum_l1976_197626

theorem bowl_capacity_sum : 
  let second_bowl : ℕ := 600
  let first_bowl : ℕ := (3 * second_bowl) / 4
  let third_bowl : ℕ := first_bowl / 2
  let fourth_bowl : ℕ := second_bowl / 3
  second_bowl + first_bowl + third_bowl + fourth_bowl = 1475 :=
by sorry

end bowl_capacity_sum_l1976_197626


namespace parametric_to_standard_equation_l1976_197634

theorem parametric_to_standard_equation :
  ∀ (x y : ℝ), (∃ t : ℝ, x = 4 * t + 1 ∧ y = -2 * t - 5) → x + 2 * y + 9 = 0 := by
  sorry

end parametric_to_standard_equation_l1976_197634


namespace circumcircle_equation_l1976_197686

-- Define the vertices of the triangle
def A : ℝ × ℝ := (2, 2)
def B : ℝ × ℝ := (5, 3)
def C : ℝ × ℝ := (3, -1)

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop :=
  (x - 4)^2 + (y - 1)^2 = 5

-- Theorem statement
theorem circumcircle_equation :
  (circle_equation A.1 A.2) ∧
  (circle_equation B.1 B.2) ∧
  (circle_equation C.1 C.2) ∧
  (∀ (a b r : ℝ), (
    ((A.1 - a)^2 + (A.2 - b)^2 = r^2) ∧
    ((B.1 - a)^2 + (B.2 - b)^2 = r^2) ∧
    ((C.1 - a)^2 + (C.2 - b)^2 = r^2)
  ) → a = 4 ∧ b = 1 ∧ r^2 = 5) :=
sorry

end circumcircle_equation_l1976_197686


namespace paint_O_circles_l1976_197644

theorem paint_O_circles (num_circles : Nat) (num_colors : Nat) : 
  num_circles = 4 → num_colors = 3 → num_colors ^ num_circles = 81 := by
  sorry

#check paint_O_circles

end paint_O_circles_l1976_197644


namespace total_wax_sticks_is_42_l1976_197659

/-- Calculates the total number of wax sticks used for animal sculptures --/
def total_wax_sticks (large_animal_wax : ℕ) (small_animal_wax : ℕ) (small_animal_total_wax : ℕ) : ℕ :=
  let small_animals := small_animal_total_wax / small_animal_wax
  let large_animals := small_animals / 5
  let large_animal_total_wax := large_animals * large_animal_wax
  small_animal_total_wax + large_animal_total_wax

/-- Theorem stating that the total number of wax sticks used is 42 --/
theorem total_wax_sticks_is_42 :
  total_wax_sticks 6 3 30 = 42 :=
by
  sorry

#eval total_wax_sticks 6 3 30

end total_wax_sticks_is_42_l1976_197659


namespace remainder_9876543210_mod_101_l1976_197684

theorem remainder_9876543210_mod_101 : 9876543210 % 101 = 1 := by
  sorry

end remainder_9876543210_mod_101_l1976_197684


namespace number_of_zeros_l1976_197613

-- Define the piecewise function
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 2 else Real.log x - 1

-- Define what it means for x to be a zero of f
def isZero (x : ℝ) : Prop := f x = 0

-- State the theorem
theorem number_of_zeros : ∃ (a b : ℝ), a ≠ b ∧ isZero a ∧ isZero b ∧ ∀ c, isZero c → c = a ∨ c = b := by
  sorry

end number_of_zeros_l1976_197613


namespace point_in_fourth_quadrant_y_negative_l1976_197680

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The fourth quadrant of the Cartesian plane -/
def fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

theorem point_in_fourth_quadrant_y_negative (p : Point) 
  (h : p.x = 5) (h₂ : fourth_quadrant p) : p.y < 0 := by
  sorry

end point_in_fourth_quadrant_y_negative_l1976_197680


namespace stock_percentage_return_l1976_197625

/-- Calculate the percentage return on a stock given the income and investment. -/
theorem stock_percentage_return (income : ℝ) (investment : ℝ) :
  income = 650 →
  investment = 6240 →
  abs ((income / investment * 100) - 10.42) < 0.01 := by
  sorry

end stock_percentage_return_l1976_197625


namespace extreme_value_implies_a_eq_one_l1976_197646

/-- The function f(x) = ae^x - sin(x) has an extreme value at x = 0 -/
def has_extreme_value_at_zero (a : ℝ) : Prop :=
  let f := fun x => a * Real.exp x - Real.sin x
  ∃ ε > 0, ∀ x ∈ Set.Ioo (-ε) ε, f x ≤ f 0 ∨ f x ≥ f 0

/-- If f(x) = ae^x - sin(x) has an extreme value at x = 0, then a = 1 -/
theorem extreme_value_implies_a_eq_one :
  ∀ a : ℝ, has_extreme_value_at_zero a → a = 1 := by
  sorry

end extreme_value_implies_a_eq_one_l1976_197646


namespace merry_go_round_diameter_l1976_197693

/-- The diameter of a circular platform with area 3.14 square yards is 2 yards. -/
theorem merry_go_round_diameter : 
  ∀ (r : ℝ), r > 0 → π * r^2 = 3.14 → 2 * r = 2 := by sorry

end merry_go_round_diameter_l1976_197693


namespace fraction_power_seven_l1976_197635

theorem fraction_power_seven : (5 / 3 : ℚ) ^ 7 = 78125 / 2187 := by
  sorry

end fraction_power_seven_l1976_197635


namespace balloon_division_l1976_197624

theorem balloon_division (n : ℕ) : 
  (∃ k : ℕ, n = 7 * k + 4) ↔ (∃ m : ℕ, n = 7 * m + 4) :=
by sorry

end balloon_division_l1976_197624


namespace unique_factorization_l1976_197651

/-- The set of all positive integers that cannot be written as a sum of an arithmetic progression with difference d, having at least two terms and consisting of positive integers. -/
def M (d : ℕ) : Set ℕ :=
  {n : ℕ | ∀ (a k : ℕ), k ≥ 2 → n ≠ (k * (2 * a + (k - 1) * d)) / 2}

/-- A is the set M₁ -/
def A : Set ℕ := M 1

/-- B is the set M₂ without the element 2 -/
def B : Set ℕ := M 2 \ {2}

/-- C is the set M₃ -/
def C : Set ℕ := M 3

/-- Every element in C can be uniquely expressed as a product of an element from A and an element from B -/
theorem unique_factorization (c : ℕ) (hc : c ∈ C) :
  ∃! (a b : ℕ), a ∈ A ∧ b ∈ B ∧ c = a * b :=
sorry

end unique_factorization_l1976_197651


namespace cube_monotonicity_l1976_197643

theorem cube_monotonicity (a b : ℝ) : a^3 > b^3 → a > b := by
  sorry

end cube_monotonicity_l1976_197643


namespace paving_cost_calculation_l1976_197649

/-- The cost of paving a rectangular floor -/
def paving_cost (length width rate : ℝ) : ℝ :=
  length * width * rate

/-- Theorem: The cost of paving a rectangular floor with given dimensions and rate -/
theorem paving_cost_calculation :
  paving_cost 5.5 3.75 800 = 16500 := by
  sorry

end paving_cost_calculation_l1976_197649


namespace f_is_odd_l1976_197621

noncomputable def f (x : ℝ) : ℝ := 2 * x - 1 / x

theorem f_is_odd :
  ∀ x ∈ {x : ℝ | x < 0 ∨ x > 0}, f (-x) = -f x :=
by
  sorry

end f_is_odd_l1976_197621


namespace jill_final_llama_count_l1976_197661

/-- Represents the number of llamas Jill has after all operations -/
def final_llama_count (single_calf_llamas twin_calf_llamas traded_calves new_adults : ℕ) : ℕ :=
  let initial_llamas := single_calf_llamas + twin_calf_llamas
  let total_calves := single_calf_llamas + 2 * twin_calf_llamas
  let remaining_calves := total_calves - traded_calves
  let total_before_sale := initial_llamas + remaining_calves + new_adults
  total_before_sale - (total_before_sale / 3)

/-- Theorem stating that Jill ends up with 18 llamas given the initial conditions -/
theorem jill_final_llama_count :
  final_llama_count 9 5 8 2 = 18 := by
  sorry

end jill_final_llama_count_l1976_197661


namespace number_scientific_notation_equality_l1976_197694

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  one_le_coeff : 1 ≤ coefficient
  coeff_lt_ten : coefficient < 10

/-- The number to be represented in scientific notation -/
def number : ℕ := 11090000

/-- The scientific notation representation of the number -/
def scientific_rep : ScientificNotation :=
  { coefficient := 1.109
    exponent := 7
    one_le_coeff := by sorry
    coeff_lt_ten := by sorry }

theorem number_scientific_notation_equality :
  (number : ℝ) = scientific_rep.coefficient * (10 : ℝ) ^ scientific_rep.exponent := by
  sorry

end number_scientific_notation_equality_l1976_197694


namespace picture_distribution_l1976_197609

theorem picture_distribution (total : ℕ) (transfer : ℕ) 
  (h_total : total = 74) (h_transfer : transfer = 6) : 
  ∃ (wang_original fang_original : ℕ),
    wang_original + fang_original = total ∧
    wang_original - transfer = fang_original + transfer ∧
    wang_original = 43 ∧ 
    fang_original = 31 := by
sorry

end picture_distribution_l1976_197609


namespace milk_distribution_l1976_197630

/-- The milk distribution problem -/
theorem milk_distribution (container_a capacity_a quantity_b quantity_c : ℚ) : 
  capacity_a = 1264 →
  quantity_b + quantity_c = capacity_a →
  quantity_b + 158 = quantity_c - 158 →
  (capacity_a - (quantity_b + 158)) / capacity_a * 100 = 50 := by
  sorry

#check milk_distribution

end milk_distribution_l1976_197630


namespace arithmetic_sequence_sum_l1976_197653

/-- Given an arithmetic sequence {a_n} where a_3 = 20 - a_6, prove that S_8 = 80 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  (∀ n, S n = (n * (a 1 + a n)) / 2) →               -- sum formula
  a 3 = 20 - a 6 →                                   -- given condition
  S 8 = 80 := by
sorry

end arithmetic_sequence_sum_l1976_197653


namespace opposite_of_negative_two_l1976_197697

theorem opposite_of_negative_two :
  ∀ x : ℝ, (x + (-2) = 0) → x = 2 := by
  sorry

end opposite_of_negative_two_l1976_197697


namespace coin_balance_problem_l1976_197601

theorem coin_balance_problem :
  ∃ (a b c : ℕ),
    a + b + c = 99 ∧
    2 * a + 3 * b + c = 297 ∧
    3 * a + b + 2 * c = 297 :=
by
  sorry

end coin_balance_problem_l1976_197601


namespace certain_number_equation_l1976_197603

theorem certain_number_equation : ∃! x : ℝ, 16 * x + 17 * x + 20 * x + 11 = 170 ∧ x = 3 := by
  sorry

end certain_number_equation_l1976_197603


namespace number_calculation_l1976_197636

theorem number_calculation (x : ℝ) (h : 0.45 * x = 162) : x = 360 := by
  sorry

end number_calculation_l1976_197636


namespace rectangle_area_with_hole_l1976_197671

theorem rectangle_area_with_hole (x : ℝ) : 
  (x + 7) * (x + 5) - (x + 1) * (x + 4) = 7 * x + 31 := by
  sorry

end rectangle_area_with_hole_l1976_197671


namespace samples_per_box_l1976_197667

theorem samples_per_box (boxes_opened : ℕ) (samples_leftover : ℕ) (customers : ℕ) : 
  boxes_opened = 12 → samples_leftover = 5 → customers = 235 → 
  ∃ (samples_per_box : ℕ), samples_per_box * boxes_opened - samples_leftover = customers ∧ samples_per_box = 20 := by
  sorry

end samples_per_box_l1976_197667


namespace fir_trees_not_adjacent_probability_l1976_197627

/-- The number of pine trees -/
def pine_trees : ℕ := 4

/-- The number of cedar trees -/
def cedar_trees : ℕ := 5

/-- The number of fir trees -/
def fir_trees : ℕ := 6

/-- The total number of trees -/
def total_trees : ℕ := pine_trees + cedar_trees + fir_trees

/-- The probability that no two fir trees are next to one another when planted in a random order -/
theorem fir_trees_not_adjacent_probability : 
  (Nat.choose (pine_trees + cedar_trees + 1) fir_trees : ℚ) / 
  (Nat.choose total_trees fir_trees) = 6 / 143 := by sorry

end fir_trees_not_adjacent_probability_l1976_197627


namespace chicken_pot_pie_customers_l1976_197656

/-- The number of pieces in a shepherd's pie -/
def shepherds_pie_pieces : ℕ := 4

/-- The number of pieces in a chicken pot pie -/
def chicken_pot_pie_pieces : ℕ := 5

/-- The number of customers who ordered slices of shepherd's pie -/
def shepherds_pie_customers : ℕ := 52

/-- The total number of pies sold -/
def total_pies_sold : ℕ := 29

/-- Theorem stating the number of customers who ordered slices of chicken pot pie -/
theorem chicken_pot_pie_customers : ℕ := by
  sorry

end chicken_pot_pie_customers_l1976_197656


namespace R_equals_triangle_interior_l1976_197692

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The polynomial z^2 + az + b -/
def polynomial (a b : ℝ) (z : ℂ) : ℂ := z^2 + a*z + b

/-- The region R -/
def R : Set (ℝ × ℝ) :=
  {p | ∀ z, polynomial p.1 p.2 z = 0 → Complex.abs z < 1}

/-- The triangle ABC -/
def triangle_ABC : Set (ℝ × ℝ) :=
  {p | p.1 > -2 ∧ p.1 < 2 ∧ p.2 > -1 ∧ p.2 < 1 ∧ p.2 < (1 - p.1/2)}

/-- The theorem stating that R is equivalent to the interior of triangle ABC -/
theorem R_equals_triangle_interior : R = triangle_ABC := by sorry

end R_equals_triangle_interior_l1976_197692


namespace min_value_expression_l1976_197616

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (((a^2 + b^2) * (4*a^2 + b^2)).sqrt) / (a * b) ≥ 3 :=
sorry

end min_value_expression_l1976_197616


namespace intersection_M_N_l1976_197677

-- Define the sets M and N
def M : Set ℝ := {x | x > 1}
def N : Set ℝ := {x | x^2 - 2*x ≥ 0}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x | x ≥ 2} := by sorry

end intersection_M_N_l1976_197677


namespace disease_mortality_percentage_l1976_197695

theorem disease_mortality_percentage (population : ℝ) (affected_percentage : ℝ) (mortality_rate : ℝ) 
  (h1 : affected_percentage = 15)
  (h2 : mortality_rate = 8) :
  (affected_percentage / 100) * (mortality_rate / 100) * 100 = 1.2 := by
  sorry

end disease_mortality_percentage_l1976_197695


namespace derivative_f_at_pi_half_l1976_197645

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x

theorem derivative_f_at_pi_half : 
  (deriv f) (π / 2) = 1 := by sorry

end derivative_f_at_pi_half_l1976_197645


namespace intersection_of_M_and_N_l1976_197655

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_of_M_and_N : M ∩ N = {2, 4} := by
  sorry

end intersection_of_M_and_N_l1976_197655


namespace max_cars_ac_no_stripes_l1976_197698

theorem max_cars_ac_no_stripes 
  (total_cars : ℕ) 
  (cars_without_ac : ℕ) 
  (cars_with_stripes : ℕ) 
  (h1 : total_cars = 200)
  (h2 : cars_without_ac = 85)
  (h3 : cars_with_stripes ≥ 110) :
  ∃ (max_ac_no_stripes : ℕ),
    max_ac_no_stripes = 5 ∧
    max_ac_no_stripes ≤ total_cars - cars_without_ac ∧
    max_ac_no_stripes ≤ total_cars - cars_with_stripes :=
by
  sorry

end max_cars_ac_no_stripes_l1976_197698


namespace money_distribution_solution_l1976_197615

/-- Represents the money distribution problem --/
structure MoneyDistribution where
  ann_initial : ℕ
  bill_initial : ℕ
  charlie_initial : ℕ
  bill_to_ann : ℕ
  charlie_to_bill : ℕ

/-- Checks if the money distribution results in equal amounts --/
def isEqualDistribution (md : MoneyDistribution) : Prop :=
  let ann_final := md.ann_initial + md.bill_to_ann
  let bill_final := md.bill_initial - md.bill_to_ann + md.charlie_to_bill
  let charlie_final := md.charlie_initial - md.charlie_to_bill
  ann_final = bill_final ∧ bill_final = charlie_final

/-- Theorem stating the solution to the money distribution problem --/
theorem money_distribution_solution :
  let md : MoneyDistribution := {
    ann_initial := 777,
    bill_initial := 1111,
    charlie_initial := 1555,
    bill_to_ann := 371,
    charlie_to_bill := 408
  }
  isEqualDistribution md ∧ 
  (md.ann_initial + md.bill_to_ann = 1148) ∧
  (md.bill_initial - md.bill_to_ann + md.charlie_to_bill = 1148) ∧
  (md.charlie_initial - md.charlie_to_bill = 1148) :=
by
  sorry


end money_distribution_solution_l1976_197615


namespace max_sum_after_adding_pyramid_l1976_197674

/-- A rectangular prism -/
structure RectangularPrism :=
  (faces : ℕ)
  (edges : ℕ)
  (vertices : ℕ)

/-- Properties of the resulting solid after adding a square pyramid -/
structure ResultingSolid :=
  (faces : ℕ)
  (edges : ℕ)
  (vertices : ℕ)

/-- Function to calculate the resulting solid properties -/
def add_pyramid (prism : RectangularPrism) : ResultingSolid :=
  { faces := prism.faces - 1 + 4,
    edges := prism.edges + 4,
    vertices := prism.vertices + 1 }

/-- Theorem stating the maximum sum of faces, edges, and vertices -/
theorem max_sum_after_adding_pyramid (prism : RectangularPrism)
  (h1 : prism.faces = 6)
  (h2 : prism.edges = 12)
  (h3 : prism.vertices = 8) :
  let resulting := add_pyramid prism
  resulting.faces + resulting.edges + resulting.vertices = 34 :=
sorry

end max_sum_after_adding_pyramid_l1976_197674


namespace garden_area_increase_l1976_197640

/-- Given a rectangular garden with length 60 feet and width 20 feet,
    prove that reshaping it into a square garden with the same perimeter
    increases the area by 400 square feet. -/
theorem garden_area_increase :
  let rect_length : ℝ := 60
  let rect_width : ℝ := 20
  let rect_perimeter : ℝ := 2 * (rect_length + rect_width)
  let square_side : ℝ := rect_perimeter / 4
  let rect_area : ℝ := rect_length * rect_width
  let square_area : ℝ := square_side * square_side
  square_area - rect_area = 400 := by
sorry


end garden_area_increase_l1976_197640
