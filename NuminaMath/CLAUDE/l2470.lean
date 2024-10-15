import Mathlib

namespace NUMINAMATH_CALUDE_students_on_sports_teams_l2470_247038

theorem students_on_sports_teams 
  (total_students : ℕ) 
  (band_students : ℕ) 
  (both_activities : ℕ) 
  (either_activity : ℕ) 
  (h1 : total_students = 320)
  (h2 : band_students = 85)
  (h3 : both_activities = 60)
  (h4 : either_activity = 225)
  (h5 : either_activity = band_students + sports_students - both_activities) :
  sports_students = 200 :=
by
  sorry

end NUMINAMATH_CALUDE_students_on_sports_teams_l2470_247038


namespace NUMINAMATH_CALUDE_magnitude_relationship_l2470_247031

theorem magnitude_relationship (a b : ℝ) (ha : a < 0) (hb : -1 < b ∧ b < 0) :
  a < a * b ∧ a * b < a * b^2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_relationship_l2470_247031


namespace NUMINAMATH_CALUDE_diamond_digit_equality_l2470_247023

theorem diamond_digit_equality (diamond : ℕ) : 
  diamond < 10 →  -- diamond is a digit
  (9 * diamond + 6 = 10 * diamond + 3) →  -- diamond6₉ = diamond3₁₀
  diamond = 3 :=
by sorry

end NUMINAMATH_CALUDE_diamond_digit_equality_l2470_247023


namespace NUMINAMATH_CALUDE_students_not_eating_lunch_l2470_247058

theorem students_not_eating_lunch (total_students : ℕ) 
  (cafeteria_students : ℕ) (h1 : total_students = 60) 
  (h2 : cafeteria_students = 10) : 
  total_students - (3 * cafeteria_students + cafeteria_students) = 20 := by
  sorry

end NUMINAMATH_CALUDE_students_not_eating_lunch_l2470_247058


namespace NUMINAMATH_CALUDE_eight_people_lineup_l2470_247012

theorem eight_people_lineup : Nat.factorial 8 = 40320 := by
  sorry

end NUMINAMATH_CALUDE_eight_people_lineup_l2470_247012


namespace NUMINAMATH_CALUDE_sample_size_l2470_247090

theorem sample_size (n : ℕ) (f₁ f₂ f₃ f₄ f₅ f₆ : ℕ) : 
  f₁ + f₂ + f₃ + f₄ + f₅ + f₆ = n →
  f₁ + f₂ + f₃ = 27 →
  2 * f₆ = f₁ →
  3 * f₆ = f₂ →
  4 * f₆ = f₃ →
  6 * f₆ = f₄ →
  4 * f₆ = f₅ →
  n = 60 := by
sorry

end NUMINAMATH_CALUDE_sample_size_l2470_247090


namespace NUMINAMATH_CALUDE_underlined_numbers_are_correct_l2470_247029

def sequence_term (n : ℕ) : ℕ := 3 * n - 2

def has_same_digits (n : ℕ) : Prop :=
  ∀ d₁ d₂, d₁ ∈ n.digits 10 ∧ d₂ ∈ n.digits 10 → d₁ = d₂

def underlined_numbers : Set ℕ :=
  {n | ∃ k, sequence_term k = n ∧ 10 < n ∧ n < 100000 ∧ has_same_digits n}

theorem underlined_numbers_are_correct : underlined_numbers = 
  {22, 55, 88, 1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999, 
   11111, 22222, 33333, 44444, 55555, 66666, 77777, 88888, 99999} := by
  sorry

end NUMINAMATH_CALUDE_underlined_numbers_are_correct_l2470_247029


namespace NUMINAMATH_CALUDE_heartsuit_calculation_l2470_247087

-- Define the heartsuit operation
def heartsuit (x y : ℝ) : ℝ := x^2 - y^2

-- Theorem statement
theorem heartsuit_calculation :
  heartsuit 3 (heartsuit 4 5) = -72 := by
  sorry

end NUMINAMATH_CALUDE_heartsuit_calculation_l2470_247087


namespace NUMINAMATH_CALUDE_new_shipment_bears_l2470_247088

def initial_stock : ℕ := 4
def bears_per_shelf : ℕ := 7
def shelves_used : ℕ := 2

theorem new_shipment_bears :
  initial_stock + (bears_per_shelf * shelves_used) - initial_stock = 10 := by
  sorry

end NUMINAMATH_CALUDE_new_shipment_bears_l2470_247088


namespace NUMINAMATH_CALUDE_continued_fraction_equality_l2470_247095

theorem continued_fraction_equality : 
  2 + (3 / (2 + (5 / (4 + (7 / 3))))) = 91 / 19 := by sorry

end NUMINAMATH_CALUDE_continued_fraction_equality_l2470_247095


namespace NUMINAMATH_CALUDE_colored_cube_covers_plane_l2470_247028

/-- A cube with colored middle squares on each face -/
structure ColoredCube where
  a : ℕ
  b : ℕ
  c : ℕ

/-- An infinite plane with unit squares -/
def Plane := ℕ × ℕ

/-- A point on the plane is colorable if the cube can land on it with its colored face -/
def isColorable (cube : ColoredCube) (point : Plane) : Prop := sorry

/-- The main theorem: If any two sides of the cube are relatively prime, 
    then every point on the plane is colorable -/
theorem colored_cube_covers_plane (cube : ColoredCube) :
  (Nat.gcd (2 * cube.a + 1) (2 * cube.b + 1) = 1 ∨
   Nat.gcd (2 * cube.b + 1) (2 * cube.c + 1) = 1 ∨
   Nat.gcd (2 * cube.a + 1) (2 * cube.c + 1) = 1) →
  ∀ (point : Plane), isColorable cube point := by
  sorry

end NUMINAMATH_CALUDE_colored_cube_covers_plane_l2470_247028


namespace NUMINAMATH_CALUDE_car_owners_without_motorcycle_l2470_247078

theorem car_owners_without_motorcycle (total : ℕ) (car_owners : ℕ) (motorcycle_owners : ℕ)
  (h1 : total = 351)
  (h2 : car_owners = 331)
  (h3 : motorcycle_owners = 45)
  (h4 : car_owners + motorcycle_owners - total ≥ 0) :
  car_owners - (car_owners + motorcycle_owners - total) = 306 := by
  sorry

end NUMINAMATH_CALUDE_car_owners_without_motorcycle_l2470_247078


namespace NUMINAMATH_CALUDE_orange_count_l2470_247048

theorem orange_count (b t o : ℕ) : 
  (b + t) / 2 = 89 →
  (b + t + o) / 3 = 91 →
  o = 95 := by
sorry

end NUMINAMATH_CALUDE_orange_count_l2470_247048


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2470_247000

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 1)*x + 1 ≥ 0) → -1 ≤ a ∧ a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2470_247000


namespace NUMINAMATH_CALUDE_multiply_squared_terms_l2470_247081

theorem multiply_squared_terms (a : ℝ) : 3 * a^2 * (2 * a^2) = 6 * a^4 := by
  sorry

end NUMINAMATH_CALUDE_multiply_squared_terms_l2470_247081


namespace NUMINAMATH_CALUDE_complex_percentage_calculation_l2470_247094

theorem complex_percentage_calculation : 
  let a := 0.15 * 50
  let b := 0.25 * 75
  let c := -0.10 * 120
  let sum := a + b + c
  let d := -0.05 * 150
  2.5 * d - (1/3) * sum = -23.5 := by
sorry

end NUMINAMATH_CALUDE_complex_percentage_calculation_l2470_247094


namespace NUMINAMATH_CALUDE_bakery_storage_l2470_247057

theorem bakery_storage (sugar flour baking_soda : ℕ) : 
  (sugar : ℚ) / flour = 5 / 2 →
  (flour : ℚ) / baking_soda = 10 / 1 →
  (flour : ℚ) / (baking_soda + 60) = 8 / 1 →
  sugar = 6000 := by
sorry


end NUMINAMATH_CALUDE_bakery_storage_l2470_247057


namespace NUMINAMATH_CALUDE_solve_missed_questions_l2470_247008

def missed_questions_problem (your_missed : ℕ) (friend_ratio : ℕ) : Prop :=
  let friend_missed := (your_missed / friend_ratio : ℕ)
  your_missed = 36 ∧ friend_ratio = 5 →
  your_missed + friend_missed = 43

theorem solve_missed_questions : missed_questions_problem 36 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_missed_questions_l2470_247008


namespace NUMINAMATH_CALUDE_polygon_exterior_angles_l2470_247041

theorem polygon_exterior_angles (n : ℕ) (exterior_angle : ℝ) : 
  n > 2 → exterior_angle = 24 → n * exterior_angle = 360 → n = 15 := by
  sorry

end NUMINAMATH_CALUDE_polygon_exterior_angles_l2470_247041


namespace NUMINAMATH_CALUDE_paycheck_calculation_l2470_247021

def biweekly_gross_pay : ℝ := 1120
def retirement_rate : ℝ := 0.25
def tax_deduction : ℝ := 100

theorem paycheck_calculation :
  biweekly_gross_pay * (1 - retirement_rate) - tax_deduction = 740 := by
  sorry

end NUMINAMATH_CALUDE_paycheck_calculation_l2470_247021


namespace NUMINAMATH_CALUDE_a_used_car_for_seven_hours_l2470_247051

/-- Represents the car hire scenario -/
structure CarHire where
  totalCost : ℕ
  bHours : ℕ
  bCost : ℕ
  cHours : ℕ

/-- Calculates the number of hours A used the car -/
def aHours (hire : CarHire) : ℕ :=
  (hire.totalCost - hire.bCost - (hire.cHours * hire.bCost / hire.bHours)) / (hire.bCost / hire.bHours)

/-- Theorem stating that A used the car for 7 hours given the conditions -/
theorem a_used_car_for_seven_hours :
  let hire := CarHire.mk 520 8 160 11
  aHours hire = 7 := by
  sorry


end NUMINAMATH_CALUDE_a_used_car_for_seven_hours_l2470_247051


namespace NUMINAMATH_CALUDE_symmetry_point_x_axis_l2470_247091

-- Define a 3D point
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define symmetry with respect to x-axis
def symmetricToXAxis (p q : Point3D) : Prop :=
  q.x = p.x ∧ q.y = -p.y ∧ q.z = -p.z

theorem symmetry_point_x_axis :
  let M : Point3D := ⟨1, 2, 3⟩
  let N : Point3D := ⟨1, -2, -3⟩
  symmetricToXAxis M N := by sorry

end NUMINAMATH_CALUDE_symmetry_point_x_axis_l2470_247091


namespace NUMINAMATH_CALUDE_simplify_expression_l2470_247007

theorem simplify_expression (w : ℝ) : 2*w + 3 - 4*w - 5 + 6*w + 7 - 8*w - 9 = -4*w - 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2470_247007


namespace NUMINAMATH_CALUDE_toffee_cost_l2470_247052

/-- The cost of 1 kg of toffees in rubles -/
def cost_per_kg : ℝ := 1.11

/-- The cost of 9 kg of toffees is less than 10 rubles -/
axiom nine_kg_cost : cost_per_kg * 9 < 10

/-- The cost of 10 kg of toffees is more than 11 rubles -/
axiom ten_kg_cost : cost_per_kg * 10 > 11

/-- Theorem: The cost of 1 kg of toffees is 1.11 rubles -/
theorem toffee_cost : cost_per_kg = 1.11 := by
  sorry

end NUMINAMATH_CALUDE_toffee_cost_l2470_247052


namespace NUMINAMATH_CALUDE_number_of_males_l2470_247055

def town_population : ℕ := 500
def male_percentage : ℚ := 2/5

theorem number_of_males :
  (town_population : ℚ) * male_percentage = 200 := by sorry

end NUMINAMATH_CALUDE_number_of_males_l2470_247055


namespace NUMINAMATH_CALUDE_major_axis_coincide_condition_l2470_247065

/-- Represents the coefficients of a general ellipse equation -/
structure EllipseCoefficients where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  E : ℝ
  F : ℝ

/-- Predicate to check if the major axis coincides with a conjugate diameter -/
def majorAxisCoincideWithConjugateDiameter (c : EllipseCoefficients) : Prop :=
  c.A * c.E - c.B * c.D = 0 ∧ 2 * c.B^2 + (c.A - c.C) * c.A = 0

/-- Theorem stating the conditions for the major axis to coincide with a conjugate diameter -/
theorem major_axis_coincide_condition (c : EllipseCoefficients) :
  majorAxisCoincideWithConjugateDiameter c ↔
  (c.A * c.E - c.B * c.D = 0 ∧ 2 * c.B^2 + (c.A - c.C) * c.A = 0) :=
by sorry

end NUMINAMATH_CALUDE_major_axis_coincide_condition_l2470_247065


namespace NUMINAMATH_CALUDE_product_repeating_decimal_and_fraction_l2470_247003

theorem product_repeating_decimal_and_fraction :
  ∃ (x : ℚ), (∀ (n : ℕ), (x * 10^n - x.floor) * 10 ≥ 6 ∧ (x * 10^n - x.floor) * 10 < 7) →
  x * (7/3) = 14/9 := by
sorry

end NUMINAMATH_CALUDE_product_repeating_decimal_and_fraction_l2470_247003


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_range_l2470_247019

theorem geometric_sequence_sum_range (m : ℝ) (hm : m > 0) :
  ∃ (a b c : ℝ), (a ≠ 0 ∧ b / a = c / b) ∧ (a + b + c = m) →
  b ∈ Set.Icc (-m) 0 ∪ Set.Ioc 0 (m / 3) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_range_l2470_247019


namespace NUMINAMATH_CALUDE_hundred_with_fewer_threes_l2470_247096

/-- An arithmetic expression using threes and basic operations -/
inductive Expr
  | three : Expr
  | add : Expr → Expr → Expr
  | sub : Expr → Expr → Expr
  | mul : Expr → Expr → Expr
  | div : Expr → Expr → Expr

/-- Count the number of threes in an expression -/
def countThrees : Expr → Nat
  | Expr.three => 1
  | Expr.add e1 e2 => countThrees e1 + countThrees e2
  | Expr.sub e1 e2 => countThrees e1 + countThrees e2
  | Expr.mul e1 e2 => countThrees e1 + countThrees e2
  | Expr.div e1 e2 => countThrees e1 + countThrees e2

/-- Evaluate an expression to a rational number -/
def eval : Expr → ℚ
  | Expr.three => 3
  | Expr.add e1 e2 => eval e1 + eval e2
  | Expr.sub e1 e2 => eval e1 - eval e2
  | Expr.mul e1 e2 => eval e1 * eval e2
  | Expr.div e1 e2 => eval e1 / eval e2

/-- There exists an expression using fewer than ten threes that evaluates to 100 -/
theorem hundred_with_fewer_threes : ∃ e : Expr, countThrees e < 10 ∧ eval e = 100 := by
  sorry

end NUMINAMATH_CALUDE_hundred_with_fewer_threes_l2470_247096


namespace NUMINAMATH_CALUDE_little_john_money_distribution_l2470_247009

theorem little_john_money_distribution (initial_amount spent_on_sweets final_amount : ℚ) 
  (num_friends : ℕ) (h1 : initial_amount = 20.10) (h2 : spent_on_sweets = 1.05) 
  (h3 : final_amount = 17.05) (h4 : num_friends = 2) : 
  (initial_amount - final_amount - spent_on_sweets) / num_friends = 1 := by
  sorry

end NUMINAMATH_CALUDE_little_john_money_distribution_l2470_247009


namespace NUMINAMATH_CALUDE_root_sum_squares_equality_l2470_247083

theorem root_sum_squares_equality (a b : ℝ) : 
  (∃ x y : ℝ, x^2 + a*x + b = 0 ∧ y^2 + b*y + a = 0) →  -- both equations have real roots
  (∃ p q r s : ℝ, p^2 + q^2 = r^2 + s^2 ∧               -- sum of squares of roots are equal
                  p^2 + a*p + b = 0 ∧ q^2 + a*q + b = 0 ∧ 
                  r^2 + b*r + a = 0 ∧ s^2 + b*s + a = 0) →
  a ≠ b →                                               -- a is not equal to b
  a + b = -2 :=                                         -- conclusion
by sorry

end NUMINAMATH_CALUDE_root_sum_squares_equality_l2470_247083


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2470_247076

theorem inequality_equivalence (x : ℝ) :
  (x - 4) / (x^2 + 4*x + 13) ≥ 0 ↔ x ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2470_247076


namespace NUMINAMATH_CALUDE_line_equation_proof_l2470_247049

/-- Given a line passing through the point (√3, -3) with an inclination angle of 30°,
    prove that its equation is y = (√3/3)x - 4 -/
theorem line_equation_proof (x y : ℝ) :
  let point : ℝ × ℝ := (Real.sqrt 3, -3)
  let angle : ℝ := 30 * π / 180  -- Convert 30° to radians
  let slope : ℝ := Real.tan angle
  slope * (x - point.1) = y - point.2 →
  y = (Real.sqrt 3 / 3) * x - 4 := by
sorry

end NUMINAMATH_CALUDE_line_equation_proof_l2470_247049


namespace NUMINAMATH_CALUDE_lemon_head_problem_l2470_247061

/-- Given a package size and a total number of Lemon Heads eaten, 
    calculate the number of whole boxes eaten and Lemon Heads left over. -/
def lemonHeadBoxes (packageSize : ℕ) (totalEaten : ℕ) : ℕ × ℕ :=
  (totalEaten / packageSize, totalEaten % packageSize)

/-- Theorem: Given a package size of 6 Lemon Heads and 54 Lemon Heads eaten,
    prove that 9 whole boxes were eaten with 0 Lemon Heads left over. -/
theorem lemon_head_problem : lemonHeadBoxes 6 54 = (9, 0) := by
  sorry

end NUMINAMATH_CALUDE_lemon_head_problem_l2470_247061


namespace NUMINAMATH_CALUDE_triangle_perimeter_not_72_l2470_247068

theorem triangle_perimeter_not_72 (a b c : ℝ) : 
  a = 20 → b = 15 → a + b > c → a + c > b → b + c > a → a + b + c ≠ 72 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_not_72_l2470_247068


namespace NUMINAMATH_CALUDE_vector_problem_l2470_247042

/-- Given vectors a and b, if |a| = 6 and a ∥ b, then x = 4 and x + y = 8 -/
theorem vector_problem (x y : ℝ) : 
  let a : ℝ × ℝ × ℝ := (2, 4, x)
  let b : ℝ × ℝ × ℝ := (2, y, 2)
  (‖a‖ = 6 ∧ ∃ (k : ℝ), k ≠ 0 ∧ a = k • b) → x = 4 ∧ x + y = 8 := by
  sorry


end NUMINAMATH_CALUDE_vector_problem_l2470_247042


namespace NUMINAMATH_CALUDE_trigonometric_function_property_l2470_247089

theorem trigonometric_function_property (f : ℝ → ℝ) :
  (∀ α, f (Real.cos α) = Real.sin α) → f 1 = 0 ∧ f (-1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_function_property_l2470_247089


namespace NUMINAMATH_CALUDE_quadratic_form_equivalence_l2470_247092

theorem quadratic_form_equivalence (b : ℝ) (m : ℝ) : 
  b > 0 → 
  (∀ x, x^2 + b*x + 44 = (x + m)^2 + 8) → 
  b = 12 := by
sorry

end NUMINAMATH_CALUDE_quadratic_form_equivalence_l2470_247092


namespace NUMINAMATH_CALUDE_pencils_given_equals_nine_l2470_247010

/-- The number of pencils in one stroke -/
def pencils_per_stroke : ℕ := 12

/-- The number of strokes Namjoon had -/
def namjoon_strokes : ℕ := 2

/-- The number of pencils Namjoon had left after giving some to Yoongi -/
def pencils_left : ℕ := 15

/-- The number of pencils Namjoon gave to Yoongi -/
def pencils_given_to_yoongi : ℕ := namjoon_strokes * pencils_per_stroke - pencils_left

theorem pencils_given_equals_nine : pencils_given_to_yoongi = 9 := by
  sorry

end NUMINAMATH_CALUDE_pencils_given_equals_nine_l2470_247010


namespace NUMINAMATH_CALUDE_shirt_price_proof_l2470_247006

theorem shirt_price_proof (final_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) 
  (h1 : final_price = 105)
  (h2 : discount1 = 19.954259576901087)
  (h3 : discount2 = 12.55) :
  ∃ (list_price : ℝ), 
    list_price * (1 - discount1 / 100) * (1 - discount2 / 100) = final_price ∧ 
    list_price = 150 := by
  sorry

end NUMINAMATH_CALUDE_shirt_price_proof_l2470_247006


namespace NUMINAMATH_CALUDE_vector_equation_solution_l2470_247053

theorem vector_equation_solution :
  let a : ℚ := -491/342
  let b : ℚ := 233/342
  let c : ℚ := 49/38
  let v₁ : Fin 3 → ℚ := ![1, -2, 3]
  let v₂ : Fin 3 → ℚ := ![4, 1, -1]
  let v₃ : Fin 3 → ℚ := ![-3, 2, 1]
  let result : Fin 3 → ℚ := ![0, 1, 4]
  (a • v₁) + (b • v₂) + (c • v₃) = result := by
  sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l2470_247053


namespace NUMINAMATH_CALUDE_certain_number_problem_l2470_247035

theorem certain_number_problem (x : ℤ) : x + 3 = 226 → 3 * x = 669 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l2470_247035


namespace NUMINAMATH_CALUDE_min_value_x2_plus_y2_l2470_247043

theorem min_value_x2_plus_y2 (x y : ℝ) (h : (x + 1)^2 + y^2 = 1/4) :
  ∃ (min : ℝ), min = 1/4 ∧ ∀ (a b : ℝ), (a + 1)^2 + b^2 = 1/4 → a^2 + b^2 ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_x2_plus_y2_l2470_247043


namespace NUMINAMATH_CALUDE_system_of_equations_solutions_l2470_247011

theorem system_of_equations_solutions :
  (∃ x y : ℝ, x - 2*y = 0 ∧ 3*x - y = 5 ∧ x = 2 ∧ y = 1) ∧
  (∃ x y : ℝ, 3*(x - 1) - 4*(y + 1) = -1 ∧ x/2 + y/3 = -2 ∧ x = -2 ∧ y = -3) :=
by sorry

end NUMINAMATH_CALUDE_system_of_equations_solutions_l2470_247011


namespace NUMINAMATH_CALUDE_root_form_sum_l2470_247070

/-- The cubic polynomial 2x^3 + 3x^2 - 5x - 2 = 0 has a real root of the form (∛p + ∛q + 2)/r 
    where p, q, and r are positive integers. -/
def has_root_of_form (p q r : ℕ+) : Prop :=
  ∃ x : ℝ, 2 * x^3 + 3 * x^2 - 5 * x - 2 = 0 ∧
           x = (Real.rpow p (1/3 : ℝ) + Real.rpow q (1/3 : ℝ) + 2) / r

/-- If the cubic polynomial has a root of the specified form, then p + q + r = 10. -/
theorem root_form_sum (p q r : ℕ+) : has_root_of_form p q r → p + q + r = 10 := by
  sorry

end NUMINAMATH_CALUDE_root_form_sum_l2470_247070


namespace NUMINAMATH_CALUDE_class_size_calculation_l2470_247060

theorem class_size_calculation (female_students : ℕ) (male_students : ℕ) : 
  female_students = 13 → 
  male_students = 3 * female_students → 
  female_students + male_students = 52 := by
sorry

end NUMINAMATH_CALUDE_class_size_calculation_l2470_247060


namespace NUMINAMATH_CALUDE_savings_after_four_weeks_l2470_247066

/-- Calculates the total savings after a given number of weeks, 
    with an initial saving amount and a fixed weekly increase. -/
def totalSavings (initialSaving : ℕ) (weeklyIncrease : ℕ) (weeks : ℕ) : ℕ :=
  initialSaving + weeklyIncrease * (weeks - 1)

/-- Theorem: Given an initial saving of $20 and a weekly increase of $10,
    the total savings after 4 weeks is $60. -/
theorem savings_after_four_weeks :
  totalSavings 20 10 4 = 60 := by
  sorry

end NUMINAMATH_CALUDE_savings_after_four_weeks_l2470_247066


namespace NUMINAMATH_CALUDE_solution_set_min_value_min_value_ab_equality_condition_l2470_247032

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 2| + 2 * |x - 1|

-- Part 1: Solution set of f(x) ≤ 4
theorem solution_set (x : ℝ) : f x ≤ 4 ↔ 0 ≤ x ∧ x ≤ 4/3 := by sorry

-- Part 2: Minimum value of f(x)
theorem min_value : ∃ (x : ℝ), ∀ (y : ℝ), f x ≤ f y ∧ f x = 3 := by sorry

-- Part 3: Minimum value of 1/(a-1) + 2/b given conditions
theorem min_value_ab (a b : ℝ) (ha : a > 1) (hb : b > 0) (hab : a + 2*b = 3) :
  1/(a-1) + 2/b ≥ 9/2 := by sorry

-- Part 4: Equality condition for the minimum value
theorem equality_condition (a b : ℝ) (ha : a > 1) (hb : b > 0) (hab : a + 2*b = 3) :
  1/(a-1) + 2/b = 9/2 ↔ a = 5/3 ∧ b = 2/3 := by sorry

end NUMINAMATH_CALUDE_solution_set_min_value_min_value_ab_equality_condition_l2470_247032


namespace NUMINAMATH_CALUDE_no_valid_x_l2470_247062

theorem no_valid_x : ¬ ∃ (x : ℕ+), 
  (x : ℝ) - 7 > 0 ∧ 
  (x + 5) * (x - 7) * (x^2 + x + 30) < 800 :=
sorry

end NUMINAMATH_CALUDE_no_valid_x_l2470_247062


namespace NUMINAMATH_CALUDE_displacement_increment_l2470_247024

/-- Given an object with equation of motion s = 2t^2, 
    prove that the increment of displacement from time t = 2 to t = 2 + d 
    is equal to 8d + 2d^2 -/
theorem displacement_increment (d : ℝ) : 
  let s (t : ℝ) := 2 * t^2
  (s (2 + d) - s 2) = 8*d + 2*d^2 := by
sorry

end NUMINAMATH_CALUDE_displacement_increment_l2470_247024


namespace NUMINAMATH_CALUDE_daisy_monday_toys_l2470_247005

/-- The number of dog toys Daisy had on Monday -/
def monday_toys : ℕ := sorry

/-- The number of dog toys Daisy had on Tuesday after losing some -/
def tuesday_toys : ℕ := 3

/-- The number of new toys Daisy's owner bought on Tuesday -/
def tuesday_new_toys : ℕ := 3

/-- The number of new toys Daisy's owner bought on Wednesday -/
def wednesday_new_toys : ℕ := 5

/-- The total number of dog toys Daisy would have if all lost toys were found -/
def total_toys : ℕ := 13

theorem daisy_monday_toys : 
  monday_toys = 5 :=
by sorry

end NUMINAMATH_CALUDE_daisy_monday_toys_l2470_247005


namespace NUMINAMATH_CALUDE_binomial_coefficient_problem_l2470_247025

theorem binomial_coefficient_problem (m : ℕ) :
  (1 : ℚ) / (Nat.choose 5 m) - (1 : ℚ) / (Nat.choose 6 m) = 7 / (10 * Nat.choose 7 m) →
  Nat.choose 8 m = 28 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_problem_l2470_247025


namespace NUMINAMATH_CALUDE_partial_fraction_sum_zero_l2470_247040

theorem partial_fraction_sum_zero (A B C D E F : ℝ) :
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ -1 ∧ x ≠ -2 ∧ x ≠ -3 ∧ x ≠ -4 ∧ x ≠ -5 →
    1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5)) =
    A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5)) →
  A + B + C + D + E + F = 0 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_sum_zero_l2470_247040


namespace NUMINAMATH_CALUDE_symmetry_implies_values_l2470_247069

/-- Two lines are symmetric about y = x if and only if they are inverse functions of each other -/
axiom symmetry_iff_inverse (f g : ℝ → ℝ) : 
  (∀ x y, f y = x ↔ g x = y) ↔ (∀ x, f (g x) = x ∧ g (f x) = x)

/-- The line ax - y + 2 = 0 -/
def line1 (a : ℝ) (x : ℝ) : ℝ := a * x + 2

/-- The line 3x - y - b = 0 -/
def line2 (b : ℝ) (x : ℝ) : ℝ := 3 * x - b

theorem symmetry_implies_values (a b : ℝ) : 
  (∀ x y, line1 a y = x ↔ line2 b x = y) → a = 1/3 ∧ b = 6 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_values_l2470_247069


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l2470_247030

theorem cubic_sum_theorem (a b c : ℝ) (ha : a ≠ b) (hb : b ≠ c) (hc : c ≠ a)
  (h : (a^3 + 12) / a = (b^3 + 12) / b ∧ (b^3 + 12) / b = (c^3 + 12) / c) :
  a^3 + b^3 + c^3 = -36 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l2470_247030


namespace NUMINAMATH_CALUDE_greatest_integer_radius_l2470_247044

theorem greatest_integer_radius (A : ℝ) (h : A < 60 * Real.pi) :
  ∃ (r : ℕ), r * r * Real.pi ≤ A ∧ ∀ (s : ℕ), s * s * Real.pi ≤ A → s ≤ r ∧ r = 7 :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_radius_l2470_247044


namespace NUMINAMATH_CALUDE_paris_cafe_contribution_l2470_247046

/-- Represents the currency exchange problem in the Paris cafe --/
theorem paris_cafe_contribution
  (pastry_cost : ℝ)
  (emily_dollars : ℝ)
  (exchange_rate : ℝ)
  (h_pastry_cost : pastry_cost = 8)
  (h_emily_dollars : emily_dollars = 10)
  (h_exchange_rate : exchange_rate = 1.20)
  : ∃ (berengere_contribution : ℝ),
    berengere_contribution = 0 ∧
    emily_dollars / exchange_rate + berengere_contribution ≥ pastry_cost :=
by sorry

end NUMINAMATH_CALUDE_paris_cafe_contribution_l2470_247046


namespace NUMINAMATH_CALUDE_test_questions_count_l2470_247036

/-- Calculates the total number of questions on a test given the time spent answering,
    time per question, and number of unanswered questions. -/
def totalQuestions (hoursSpent : ℕ) (minutesPerQuestion : ℕ) (unansweredQuestions : ℕ) : ℕ :=
  (hoursSpent * 60 / minutesPerQuestion) + unansweredQuestions

/-- Proves that the total number of questions on the test is 100 -/
theorem test_questions_count :
  totalQuestions 2 2 40 = 100 := by sorry

end NUMINAMATH_CALUDE_test_questions_count_l2470_247036


namespace NUMINAMATH_CALUDE_additional_distance_with_speed_increase_l2470_247097

/-- Calculates the additional distance traveled when increasing speed for a given initial distance and speeds. -/
theorem additional_distance_with_speed_increase 
  (actual_speed : ℝ) 
  (faster_speed : ℝ) 
  (actual_distance : ℝ) 
  (h1 : actual_speed > 0)
  (h2 : faster_speed > actual_speed)
  (h3 : actual_distance > 0)
  : let time := actual_distance / actual_speed
    let faster_distance := faster_speed * time
    faster_distance - actual_distance = 20 :=
by sorry

end NUMINAMATH_CALUDE_additional_distance_with_speed_increase_l2470_247097


namespace NUMINAMATH_CALUDE_marathon_remainder_yards_l2470_247050

/-- Represents the length of a marathon in miles and yards -/
structure Marathon where
  miles : ℕ
  yards : ℕ

/-- Represents the total distance run in miles and yards -/
structure TotalDistance where
  miles : ℕ
  yards : ℕ

def yardsPerMile : ℕ := 1760

def marathonLength : Marathon := { miles := 25, yards := 500 }

def numberOfMarathons : ℕ := 12

theorem marathon_remainder_yards :
  ∃ (m : ℕ) (y : ℕ), 
    y < yardsPerMile ∧
    TotalDistance.yards (
      { miles := m
      , yards := y
      } : TotalDistance
    ) = 720 ∧
    numberOfMarathons * (marathonLength.miles * yardsPerMile + marathonLength.yards) =
      m * yardsPerMile + y :=
by sorry

end NUMINAMATH_CALUDE_marathon_remainder_yards_l2470_247050


namespace NUMINAMATH_CALUDE_zoes_bottles_l2470_247045

/-- Given the initial number of bottles, the number of bottles drunk, and the number of bottles bought,
    calculate the final number of bottles. -/
def finalBottles (initial drunk bought : ℕ) : ℕ :=
  initial - drunk + bought

/-- Prove that for Zoe's specific case, the final number of bottles is 47. -/
theorem zoes_bottles : finalBottles 42 25 30 = 47 := by
  sorry

end NUMINAMATH_CALUDE_zoes_bottles_l2470_247045


namespace NUMINAMATH_CALUDE_product_digit_sum_base_8_l2470_247018

def base_8_to_decimal (n : ℕ) : ℕ := sorry

def decimal_to_base_8 (n : ℕ) : ℕ := sorry

def sum_of_digits_base_8 (n : ℕ) : ℕ := sorry

theorem product_digit_sum_base_8 :
  let a := 35
  let b := 21
  let product := (base_8_to_decimal a) * (base_8_to_decimal b)
  sum_of_digits_base_8 (decimal_to_base_8 product) = 21
  := by sorry

end NUMINAMATH_CALUDE_product_digit_sum_base_8_l2470_247018


namespace NUMINAMATH_CALUDE_martha_total_savings_l2470_247039

/-- Represents Martha's savings plan for a month --/
structure SavingsPlan where
  daily_allowance : ℝ
  week1_savings : List ℝ
  week2_savings : List ℝ
  week3_savings : List ℝ
  week4_savings : List ℝ
  week1_expense : ℝ
  week2_expense : ℝ
  week3_expense : ℝ
  week4_expense : ℝ

/-- Calculates the total savings for a given week --/
def weekly_savings (savings : List ℝ) (expense : ℝ) : ℝ :=
  savings.sum - expense

/-- Calculates the total savings for the month --/
def total_monthly_savings (plan : SavingsPlan) : ℝ :=
  weekly_savings plan.week1_savings plan.week1_expense +
  weekly_savings plan.week2_savings plan.week2_expense +
  weekly_savings plan.week3_savings plan.week3_expense +
  weekly_savings plan.week4_savings plan.week4_expense

/-- Martha's specific savings plan --/
def martha_plan : SavingsPlan :=
  { daily_allowance := 15
  , week1_savings := [6, 6, 6, 6, 6, 6, 4.5]
  , week2_savings := [7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 6]
  , week3_savings := [9, 9, 9, 9, 7.5, 9, 9]
  , week4_savings := [10.5, 10.5, 10.5, 10.5, 10.5, 10.5, 10.5, 10.5, 9]
  , week1_expense := 20
  , week2_expense := 30
  , week3_expense := 40
  , week4_expense := 50
  }

/-- Theorem stating that Martha's total savings at the end of the month is $106 --/
theorem martha_total_savings :
  total_monthly_savings martha_plan = 106 := by
  sorry

end NUMINAMATH_CALUDE_martha_total_savings_l2470_247039


namespace NUMINAMATH_CALUDE_equation_solution_l2470_247013

theorem equation_solution (y : ℚ) : 
  (∃ x : ℚ, 19 * (x + y) + 17 = 19 * (-x + y) - 21) → 
  (∀ x : ℚ, 19 * (x + y) + 17 = 19 * (-x + y) - 21 → x = -1) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2470_247013


namespace NUMINAMATH_CALUDE_perfect_square_identification_l2470_247077

theorem perfect_square_identification :
  ¬ ∃ (x : ℕ), 7^2051 = x^2 ∧
  ∃ (a b c d : ℕ), 6^2048 = a^2 ∧ 8^2050 = b^2 ∧ 9^2052 = c^2 ∧ 10^2040 = d^2 :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_identification_l2470_247077


namespace NUMINAMATH_CALUDE_charity_donation_l2470_247099

def cassandra_pennies : ℕ := 5000
def james_difference : ℕ := 276

def total_donation : ℕ := cassandra_pennies + (cassandra_pennies - james_difference)

theorem charity_donation :
  total_donation = 9724 :=
sorry

end NUMINAMATH_CALUDE_charity_donation_l2470_247099


namespace NUMINAMATH_CALUDE_vector_b_solution_l2470_247071

def vector_a : ℝ × ℝ := (1, -2)

theorem vector_b_solution (b : ℝ × ℝ) :
  (b.1 * vector_a.2 = b.2 * vector_a.1) →  -- parallel condition
  (b.1^2 + b.2^2 = 20) →                   -- magnitude condition
  (b = (2, -4) ∨ b = (-2, 4)) :=
by sorry

end NUMINAMATH_CALUDE_vector_b_solution_l2470_247071


namespace NUMINAMATH_CALUDE_min_value_at_seven_l2470_247033

/-- The quadratic function f(x) = x^2 - 14x + 45 -/
def f (x : ℝ) : ℝ := x^2 - 14*x + 45

theorem min_value_at_seven :
  ∀ x : ℝ, f 7 ≤ f x :=
by sorry

end NUMINAMATH_CALUDE_min_value_at_seven_l2470_247033


namespace NUMINAMATH_CALUDE_y₁_less_than_y₂_l2470_247002

/-- Linear function f(x) = -2x + 1 -/
def f (x : ℝ) : ℝ := -2 * x + 1

/-- Point A on the graph of f -/
def A : ℝ × ℝ := (2, f 2)

/-- Point B on the graph of f -/
def B : ℝ × ℝ := (-1, f (-1))

/-- y₁ coordinate of point A -/
def y₁ : ℝ := A.2

/-- y₂ coordinate of point B -/
def y₂ : ℝ := B.2

theorem y₁_less_than_y₂ : y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_y₁_less_than_y₂_l2470_247002


namespace NUMINAMATH_CALUDE_triplet_sum_to_two_l2470_247064

theorem triplet_sum_to_two :
  -- Triplet A
  (1/4 : ℚ) + (1/4 : ℚ) + (3/2 : ℚ) = 2 ∧
  -- Triplet B
  (3 : ℤ) + (-1 : ℤ) + (0 : ℤ) = 2 ∧
  -- Triplet C
  (0.2 : ℝ) + (0.7 : ℝ) + (1.1 : ℝ) = 2 ∧
  -- Triplet D
  (2.2 : ℝ) + (-0.5 : ℝ) + (0.5 : ℝ) ≠ 2 ∧
  -- Triplet E
  (3/5 : ℚ) + (4/5 : ℚ) + (1/5 : ℚ) ≠ 2 := by
  sorry

end NUMINAMATH_CALUDE_triplet_sum_to_two_l2470_247064


namespace NUMINAMATH_CALUDE_largest_n_for_quadratic_equation_l2470_247098

theorem largest_n_for_quadratic_equation : ∃ (x y z : ℕ+),
  13^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 6*x + 6*y + 6*z - 12 ∧
  ∀ (n : ℕ+), n > 13 →
    ¬∃ (a b c : ℕ+), n^2 = a^2 + b^2 + c^2 + 2*a*b + 2*b*c + 2*c*a + 6*a + 6*b + 6*c - 12 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_quadratic_equation_l2470_247098


namespace NUMINAMATH_CALUDE_blue_pencil_length_l2470_247020

/-- Given a pencil with a total length of 6 cm, a purple part of 3 cm, and a black part of 2 cm,
    prove that the length of the blue part is 1 cm. -/
theorem blue_pencil_length (total : ℝ) (purple : ℝ) (black : ℝ) (blue : ℝ)
    (h_total : total = 6)
    (h_purple : purple = 3)
    (h_black : black = 2)
    (h_sum : total = purple + black + blue) :
    blue = 1 := by
  sorry

end NUMINAMATH_CALUDE_blue_pencil_length_l2470_247020


namespace NUMINAMATH_CALUDE_polynomial_equality_l2470_247037

-- Define the polynomials P and Q
def P (x y z w : ℝ) : ℝ := x * y + x^2 - z + w
def Q (x y z w : ℝ) : ℝ := x + y

-- State the theorem
theorem polynomial_equality (x y z w : ℝ) :
  (x * y + z + w)^2 - (x^2 - 2*z)*(y^2 - 2*w) = 
  (P x y z w)^2 - (x^2 - 2*z)*(Q x y z w)^2 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_equality_l2470_247037


namespace NUMINAMATH_CALUDE_triangle_area_l2470_247017

/-- A triangle with integral sides and perimeter 12 has an area of 6 -/
theorem triangle_area (a b c : ℕ) : 
  a + b + c = 12 → 
  a + b > c → b + c > a → c + a > b → 
  (a * b : ℚ) / 2 = 6 :=
sorry

end NUMINAMATH_CALUDE_triangle_area_l2470_247017


namespace NUMINAMATH_CALUDE_income_for_given_tax_l2470_247080

/-- Proves that given the tax conditions, an income of $56,000 results in a total tax of $8,000 --/
theorem income_for_given_tax : ∀ (I : ℝ),
  (min I 40000 * 0.12 + max (I - 40000) 0 * 0.20 = 8000) → I = 56000 := by
  sorry

end NUMINAMATH_CALUDE_income_for_given_tax_l2470_247080


namespace NUMINAMATH_CALUDE_car_speed_problem_l2470_247015

/-- Proves that given a 15-hour trip where a car travels at 30 mph for the first 5 hours
    and the overall average speed is 38 mph, the average speed for the remaining 10 hours is 42 mph. -/
theorem car_speed_problem (v : ℝ) : 
  (5 * 30 + 10 * v) / 15 = 38 → v = 42 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l2470_247015


namespace NUMINAMATH_CALUDE_average_bicycling_speed_l2470_247074

/-- Calculates the average bicycling speed given the conditions of the problem -/
theorem average_bicycling_speed (total_distance : ℝ) (bicycle_time : ℝ) (run_speed : ℝ) (total_time : ℝ) :
  total_distance = 20 →
  bicycle_time = 12 / 60 →
  run_speed = 8 →
  total_time = 117 / 60 →
  let run_time := total_time - bicycle_time
  let run_distance := run_speed * run_time
  let bicycle_distance := total_distance - run_distance
  bicycle_distance / bicycle_time = 30 := by
  sorry

#check average_bicycling_speed

end NUMINAMATH_CALUDE_average_bicycling_speed_l2470_247074


namespace NUMINAMATH_CALUDE_helen_raisin_cookies_l2470_247085

/-- Represents the number of cookies baked --/
structure CookieCount where
  yesterday_chocolate : ℕ
  yesterday_raisin : ℕ
  today_chocolate : ℕ
  today_raisin : ℕ
  total_chocolate : ℕ

/-- Helen's cookie baking scenario --/
def helen_cookies : CookieCount where
  yesterday_chocolate := 527
  yesterday_raisin := 527  -- This is what we want to prove
  today_chocolate := 554
  today_raisin := 554
  total_chocolate := 1081

/-- Theorem stating that Helen baked 527 raisin cookies yesterday --/
theorem helen_raisin_cookies : 
  helen_cookies.yesterday_raisin = 527 := by
  sorry

#check helen_raisin_cookies

end NUMINAMATH_CALUDE_helen_raisin_cookies_l2470_247085


namespace NUMINAMATH_CALUDE_equal_pay_implies_hours_constraint_l2470_247026

/-- Represents the payment structure and hours worked for Harry and James -/
structure WorkData where
  x : ℝ  -- hourly rate
  h : ℝ  -- Harry's normal hours
  y : ℝ  -- Harry's overtime hours

/-- The theorem states that if Harry and James were paid the same amount,
    and James worked 41 hours, then h + 2y = 42 -/
theorem equal_pay_implies_hours_constraint (data : WorkData) :
  data.x * data.h + 2 * data.x * data.y = data.x * 40 + 2 * data.x * 1 →
  data.h + 2 * data.y = 42 := by
  sorry

#check equal_pay_implies_hours_constraint

end NUMINAMATH_CALUDE_equal_pay_implies_hours_constraint_l2470_247026


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l2470_247014

/-- The speed of a boat in still water, given its downstream travel information and current rate. -/
theorem boat_speed_in_still_water 
  (current_rate : ℝ) 
  (distance_downstream : ℝ) 
  (time_minutes : ℝ) 
  (h1 : current_rate = 5)
  (h2 : distance_downstream = 11.25)
  (h3 : time_minutes = 27) :
  ∃ (speed_still_water : ℝ), 
    speed_still_water = 20 ∧ 
    distance_downstream = (speed_still_water + current_rate) * (time_minutes / 60) :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l2470_247014


namespace NUMINAMATH_CALUDE_extra_boxes_calculation_l2470_247004

/-- Proves that given an order of 3 dozen boxes with extra free boxes equivalent to a 25% discount, the number of extra boxes received is 9 -/
theorem extra_boxes_calculation (dozen : ℕ) (order_size : ℕ) (discount_percent : ℚ) : 
  dozen = 12 →
  order_size = 3 →
  discount_percent = 25 / 100 →
  (dozen * order_size : ℚ) * (1 - discount_percent) = dozen * order_size - 9 :=
by sorry

end NUMINAMATH_CALUDE_extra_boxes_calculation_l2470_247004


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_relations_l2470_247022

theorem sum_of_reciprocal_relations (x y : ℚ) 
  (h1 : x⁻¹ + y⁻¹ = 4)
  (h2 : x⁻¹ - y⁻¹ = -5) : 
  x + y = -16/9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_relations_l2470_247022


namespace NUMINAMATH_CALUDE_park_area_is_3750_l2470_247047

/-- Represents a rectangular park with sides in ratio 3:2 -/
structure RectangularPark where
  x : ℝ
  length : ℝ := 3 * x
  width : ℝ := 2 * x

/-- The perimeter of the park in meters -/
def perimeter (park : RectangularPark) : ℝ :=
  2 * (park.length + park.width)

/-- The area of the park in square meters -/
def area (park : RectangularPark) : ℝ :=
  park.length * park.width

/-- The cost of fencing per meter in dollars -/
def fencingCostPerMeter : ℝ := 0.80

/-- The total cost of fencing the park in dollars -/
def totalFencingCost (park : RectangularPark) : ℝ :=
  perimeter park * fencingCostPerMeter

theorem park_area_is_3750 (park : RectangularPark) 
    (h : totalFencingCost park = 200) : area park = 3750 := by
  sorry

end NUMINAMATH_CALUDE_park_area_is_3750_l2470_247047


namespace NUMINAMATH_CALUDE_min_students_solved_both_l2470_247082

theorem min_students_solved_both (total : ℕ) (first : ℕ) (second : ℕ) :
  total = 30 →
  first = 21 →
  second = 18 →
  ∃ (both : ℕ), both ≥ 9 ∧
    both ≤ first ∧
    both ≤ second ∧
    (∀ (x : ℕ), x < both → x + (first - x) + (second - x) > total) :=
by sorry

end NUMINAMATH_CALUDE_min_students_solved_both_l2470_247082


namespace NUMINAMATH_CALUDE_total_pet_time_is_108_minutes_l2470_247034

-- Define the time spent on each activity
def dog_walk_play_time : ℚ := 1/2
def dog_feed_time : ℚ := 1/5
def cat_play_time : ℚ := 1/4
def cat_feed_time : ℚ := 1/10

-- Define the number of times each activity is performed daily
def dog_walk_play_frequency : ℕ := 2
def dog_feed_frequency : ℕ := 1
def cat_play_frequency : ℕ := 2
def cat_feed_frequency : ℕ := 1

-- Define the number of minutes in an hour
def minutes_per_hour : ℕ := 60

-- Theorem to prove
theorem total_pet_time_is_108_minutes :
  (dog_walk_play_time * dog_walk_play_frequency +
   dog_feed_time * dog_feed_frequency +
   cat_play_time * cat_play_frequency +
   cat_feed_time * cat_feed_frequency) * minutes_per_hour = 108 := by
  sorry

end NUMINAMATH_CALUDE_total_pet_time_is_108_minutes_l2470_247034


namespace NUMINAMATH_CALUDE_units_digit_of_product_l2470_247093

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- A number is composite if it has a factor other than 1 and itself -/
def isComposite (n : ℕ) : Prop := ∃ k m : ℕ, 1 < k ∧ k < n ∧ n = k * m

theorem units_digit_of_product :
  isComposite 9 ∧ isComposite 10 ∧ isComposite 12 →
  unitsDigit (9 * 10 * 12) = 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_product_l2470_247093


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l2470_247072

/-- Given a cone with base radius 2 and lateral surface forming a semicircle,
    prove that its lateral surface area is 8π. -/
theorem cone_lateral_surface_area (r : ℝ) (h : r = 2) :
  let l := 2 * r  -- slant height is twice the base radius for a semicircle lateral surface
  π * r * l = 8 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l2470_247072


namespace NUMINAMATH_CALUDE_two_bishops_placement_l2470_247086

/-- Represents a chessboard with 8 rows and 8 columns -/
structure Chessboard :=
  (rows : Nat)
  (columns : Nat)
  (total_squares : Nat)
  (white_squares : Nat)
  (black_squares : Nat)

/-- Represents the number of ways to place two bishops on a chessboard -/
def bishop_placements (board : Chessboard) : Nat :=
  board.white_squares * (board.black_squares - board.rows)

/-- Theorem stating the number of ways to place two bishops on a chessboard -/
theorem two_bishops_placement (board : Chessboard) 
  (h1 : board.rows = 8)
  (h2 : board.columns = 8)
  (h3 : board.total_squares = board.rows * board.columns)
  (h4 : board.white_squares = board.total_squares / 2)
  (h5 : board.black_squares = board.total_squares / 2) :
  bishop_placements board = 768 := by
  sorry

#eval bishop_placements {rows := 8, columns := 8, total_squares := 64, white_squares := 32, black_squares := 32}

end NUMINAMATH_CALUDE_two_bishops_placement_l2470_247086


namespace NUMINAMATH_CALUDE_systems_equivalence_l2470_247079

-- Define the systems of equations
def system1 (x y a b : ℝ) : Prop :=
  2 * (x + 1) - y = 7 ∧ x + b * y = a

def system2 (x y a b : ℝ) : Prop :=
  a * x + y = b ∧ 3 * x + 2 * (y - 1) = 9

-- Theorem statement
theorem systems_equivalence :
  ∀ (a b : ℝ),
  (∃ (x y : ℝ), system1 x y a b ∧ system2 x y a b) →
  (∃! (x y : ℝ), x = 3 ∧ y = 1 ∧ system1 x y a b ∧ system2 x y a b) ∧
  (3 * a - b)^2023 = -1 :=
sorry

end NUMINAMATH_CALUDE_systems_equivalence_l2470_247079


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l2470_247056

theorem modulus_of_complex_fraction (i : ℂ) (h : i * i = -1) :
  Complex.abs (2 * i / (i - 1)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l2470_247056


namespace NUMINAMATH_CALUDE_total_amount_is_175_l2470_247001

/-- Represents the share distribution among x, y, and z -/
structure ShareDistribution where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the total amount from a given share distribution -/
def totalAmount (s : ShareDistribution) : ℝ :=
  s.x + s.y + s.z

/-- Theorem stating that given the conditions, the total amount is 175 -/
theorem total_amount_is_175 :
  ∀ (s : ShareDistribution),
    s.y = 45 →                -- y's share is 45
    s.y = 0.45 * s.x →        -- y gets 45 paisa for each rupee x gets
    s.z = 0.30 * s.x →        -- z gets 30 paisa for each rupee x gets
    totalAmount s = 175 :=
by
  sorry

#check total_amount_is_175

end NUMINAMATH_CALUDE_total_amount_is_175_l2470_247001


namespace NUMINAMATH_CALUDE_boys_count_proof_l2470_247063

/-- Given a total number of eyes and the number of eyes per boy, 
    calculate the number of boys. -/
def number_of_boys (total_eyes : ℕ) (eyes_per_boy : ℕ) : ℕ :=
  total_eyes / eyes_per_boy

theorem boys_count_proof (total_eyes : ℕ) (eyes_per_boy : ℕ) 
  (h1 : total_eyes = 46) (h2 : eyes_per_boy = 2) : 
  number_of_boys total_eyes eyes_per_boy = 23 := by
  sorry

#eval number_of_boys 46 2

end NUMINAMATH_CALUDE_boys_count_proof_l2470_247063


namespace NUMINAMATH_CALUDE_peach_crate_pigeonhole_l2470_247084

/-- The number of crates of peaches -/
def total_crates : ℕ := 154

/-- The minimum number of peaches in a crate -/
def min_peaches : ℕ := 130

/-- The maximum number of peaches in a crate -/
def max_peaches : ℕ := 160

/-- The number of possible peach counts per crate -/
def possible_counts : ℕ := max_peaches - min_peaches + 1

theorem peach_crate_pigeonhole :
  ∃ (n : ℕ), n = 4 ∧
  (∀ (m : ℕ), m > n →
    ∃ (distribution : Fin total_crates → ℕ),
      (∀ i, min_peaches ≤ distribution i ∧ distribution i ≤ max_peaches) ∧
      (∀ k, ¬(∃ (S : Finset (Fin total_crates)), S.card = m ∧ (∀ i ∈ S, distribution i = k)))) ∧
  (∃ (distribution : Fin total_crates → ℕ),
    (∀ i, min_peaches ≤ distribution i ∧ distribution i ≤ max_peaches) →
    ∃ (k : ℕ) (S : Finset (Fin total_crates)), S.card = n ∧ (∀ i ∈ S, distribution i = k)) := by
  sorry


end NUMINAMATH_CALUDE_peach_crate_pigeonhole_l2470_247084


namespace NUMINAMATH_CALUDE_one_intersection_point_l2470_247016

-- Define the three lines
def line1 (x y : ℝ) : Prop := 2 * y - 3 * x = 4
def line2 (x y : ℝ) : Prop := x + 3 * y = 6
def line3 (x y : ℝ) : Prop := 6 * x - 9 * y = 12

-- Define a point of intersection
def is_intersection (x y : ℝ) : Prop :=
  (line1 x y ∧ line2 x y) ∨ (line1 x y ∧ line3 x y) ∨ (line2 x y ∧ line3 x y)

-- Theorem statement
theorem one_intersection_point :
  ∃! p : ℝ × ℝ, is_intersection p.1 p.2 :=
sorry

end NUMINAMATH_CALUDE_one_intersection_point_l2470_247016


namespace NUMINAMATH_CALUDE_half_coverage_days_l2470_247075

/-- Represents the number of days it takes for the lily pad patch to cover the entire lake -/
def full_coverage_days : ℕ := 58

/-- Represents the daily growth factor of the lily pad patch -/
def daily_growth_factor : ℕ := 2

/-- Theorem stating that the number of days to cover half the lake is one less than the full coverage days -/
theorem half_coverage_days : 
  ∃ (days : ℕ), days = full_coverage_days - 1 ∧ 
  (daily_growth_factor : ℚ) * ((1 : ℚ) / 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_half_coverage_days_l2470_247075


namespace NUMINAMATH_CALUDE_symmetric_line_equation_l2470_247067

/-- Given two lines l₁ and l in the plane, this theorem states that the line l₂ 
    which is symmetric to l₁ with respect to l has a specific equation. -/
theorem symmetric_line_equation (x y : ℝ) : 
  let l₁ : ℝ → ℝ := λ x => 2 * x
  let l : ℝ → ℝ := λ x => 3 * x + 3
  let l₂ : ℝ → ℝ := λ x => (11 * x - 21) / 2
  (∀ x, l₂ x = y ↔ 11 * x - 2 * y + 21 = 0) ∧
  (∀ p : ℝ × ℝ, 
    let p₁ := (p.1, l₁ p.1)
    let m := ((p.1 + p₁.1) / 2, (p.2 + p₁.2) / 2)
    m.2 = l m.1 → p.2 = l₂ p.1) :=
by sorry


end NUMINAMATH_CALUDE_symmetric_line_equation_l2470_247067


namespace NUMINAMATH_CALUDE_fourth_and_fifth_hexagons_sum_l2470_247054

/-- Represents the number of dots in a hexagonal layer -/
def hexDots : ℕ → ℕ
| 0 => 1  -- central dot
| 1 => 6  -- first layer around central dot
| 2 => 12 -- second layer
| n + 3 => 6 * (n + 1) + 2 * hexDots n  -- new pattern from 4th hexagon onwards

/-- Total dots up to and including the nth hexagon -/
def totalDots : ℕ → ℕ
| 0 => 1
| n + 1 => totalDots n + hexDots (n + 1)

theorem fourth_and_fifth_hexagons_sum :
  totalDots 5 - totalDots 3 = 138 := by
  sorry

end NUMINAMATH_CALUDE_fourth_and_fifth_hexagons_sum_l2470_247054


namespace NUMINAMATH_CALUDE_emily_candies_l2470_247073

theorem emily_candies (bob_candies : ℕ) (jennifer_candies : ℕ) (emily_candies : ℕ)
  (h1 : jennifer_candies = 2 * emily_candies)
  (h2 : jennifer_candies = 3 * bob_candies)
  (h3 : bob_candies = 4) :
  emily_candies = 6 := by
sorry

end NUMINAMATH_CALUDE_emily_candies_l2470_247073


namespace NUMINAMATH_CALUDE_point_on_line_l2470_247059

/-- Given a line passing through points (0,2) and (-4,-1), prove that if (t,7) lies on this line, then t = 20/3 -/
theorem point_on_line (t : ℝ) : 
  (∀ x y : ℝ, (y - 2) / x = (-1 - 2) / (-4 - 0)) → -- Line through (0,2) and (-4,-1)
  ((7 - 2) / t = (-1 - 2) / (-4 - 0)) →             -- (t,7) lies on the line
  t = 20 / 3 := by
sorry

end NUMINAMATH_CALUDE_point_on_line_l2470_247059


namespace NUMINAMATH_CALUDE_increasing_function_unique_root_l2470_247027

/-- An increasing function on ℝ has exactly one root -/
theorem increasing_function_unique_root (f : ℝ → ℝ) 
  (h_increasing : ∀ x y, x < y → f x < f y) :
  ∃! x, f x = 0 :=
sorry

end NUMINAMATH_CALUDE_increasing_function_unique_root_l2470_247027
