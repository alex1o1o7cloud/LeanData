import Mathlib

namespace NUMINAMATH_CALUDE_curve_transformation_l213_21380

theorem curve_transformation (x y : ℝ) :
  x^2 + y^2 = 1 → 4 * (x/2)^2 + (2*y)^2 / 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_curve_transformation_l213_21380


namespace NUMINAMATH_CALUDE_rectangle_problem_l213_21354

theorem rectangle_problem (a b k l : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < k) (h4 : 0 < l) :
  (13 * (a + b) = a * k) →
  (13 * (a + b) = b * l) →
  (k > l) →
  (k = 182) ∧ (l = 14) := by
sorry

end NUMINAMATH_CALUDE_rectangle_problem_l213_21354


namespace NUMINAMATH_CALUDE_solution_for_E_l213_21379

/-- Definition of the function E --/
def E (a b c : ℚ) : ℚ := a * b^2 + c

/-- Theorem stating that -5/8 is the solution to E(a,3,1) = E(a,5,11) --/
theorem solution_for_E : ∃ a : ℚ, E a 3 1 = E a 5 11 ∧ a = -5/8 := by
  sorry

end NUMINAMATH_CALUDE_solution_for_E_l213_21379


namespace NUMINAMATH_CALUDE_max_distance_to_line_l213_21360

theorem max_distance_to_line (a b c : ℝ) (h : a - b - c = 0) :
  ∃ (x y : ℝ), a * x + b * y + c = 0 ∧
  ∀ (x' y' : ℝ), a * x' + b * y' + c = 0 →
  (x' ^ 2 + y' ^ 2 : ℝ) ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_max_distance_to_line_l213_21360


namespace NUMINAMATH_CALUDE_oranges_per_box_l213_21363

theorem oranges_per_box (total_oranges : ℕ) (num_boxes : ℕ) (h1 : total_oranges = 35) (h2 : num_boxes = 7) :
  total_oranges / num_boxes = 5 := by
  sorry

end NUMINAMATH_CALUDE_oranges_per_box_l213_21363


namespace NUMINAMATH_CALUDE_star_ratio_equals_two_thirds_l213_21334

-- Define the ⋆ operation
def star (n m : ℕ) : ℕ := n^2 * m^3

-- Theorem statement
theorem star_ratio_equals_two_thirds :
  (star 3 2 : ℚ) / (star 2 3 : ℚ) = 2/3 := by sorry

end NUMINAMATH_CALUDE_star_ratio_equals_two_thirds_l213_21334


namespace NUMINAMATH_CALUDE_container_capacity_problem_l213_21340

/-- Represents a rectangular container with dimensions and capacity -/
structure Container where
  height : ℝ
  width : ℝ
  length : ℝ
  capacity : ℝ

/-- The problem statement -/
theorem container_capacity_problem (c1 c2 : Container) :
  c1.height = 4 →
  c1.width = 2 →
  c1.length = 8 →
  c1.capacity = 64 →
  c2.height = 3 * c1.height →
  c2.width = 2 * c1.width →
  c2.length = c1.length →
  c2.capacity = 384 := by
  sorry

end NUMINAMATH_CALUDE_container_capacity_problem_l213_21340


namespace NUMINAMATH_CALUDE_direct_proportion_m_value_l213_21308

/-- A linear function y = mx + b is a direct proportion if and only if b = 0 -/
def is_direct_proportion (m b : ℝ) : Prop := b = 0

/-- Given that y = mx + (m - 2) is a direct proportion function, prove that m = 2 -/
theorem direct_proportion_m_value (m : ℝ) 
  (h : is_direct_proportion m (m - 2)) : m = 2 := by
  sorry

end NUMINAMATH_CALUDE_direct_proportion_m_value_l213_21308


namespace NUMINAMATH_CALUDE_magician_works_two_weeks_l213_21374

/-- Calculates the number of weeks a magician works given their hourly rate, daily hours, and total payment. -/
def magician_weeks_worked (hourly_rate : ℚ) (daily_hours : ℚ) (total_payment : ℚ) : ℚ :=
  total_payment / (hourly_rate * daily_hours * 7)

/-- Theorem stating that a magician charging $60 per hour, working 3 hours per day, and receiving $2520 in total works for 2 weeks. -/
theorem magician_works_two_weeks :
  magician_weeks_worked 60 3 2520 = 2 := by
  sorry

end NUMINAMATH_CALUDE_magician_works_two_weeks_l213_21374


namespace NUMINAMATH_CALUDE_image_fixed_point_l213_21371

variable {S : Type*} [Finite S]

-- Define the set of all functions from S to S
def AllFunctions (S : Type*) := S → S

-- Define the image of a set under a function
def Image (f : S → S) (A : Set S) : Set S := {y | ∃ x ∈ A, f x = y}

-- Main theorem
theorem image_fixed_point
  (f : S → S)
  (h : ∀ g : S → S, g ≠ f → (f ∘ g ∘ f) ≠ (g ∘ f ∘ g)) :
  Image f (Image f (Set.univ : Set S)) = Image f (Set.univ : Set S) :=
sorry

end NUMINAMATH_CALUDE_image_fixed_point_l213_21371


namespace NUMINAMATH_CALUDE_oil_leak_calculation_l213_21335

theorem oil_leak_calculation (total_leak : ℕ) (leak_during_fix : ℕ) 
  (h1 : total_leak = 6206)
  (h2 : leak_during_fix = 3731) :
  total_leak - leak_during_fix = 2475 := by
  sorry

end NUMINAMATH_CALUDE_oil_leak_calculation_l213_21335


namespace NUMINAMATH_CALUDE_sector_radius_l213_21384

/-- Given a circular sector with area 11.25 cm² and arc length 4.5 cm, 
    the radius of the circle is 5 cm. -/
theorem sector_radius (area : ℝ) (arc_length : ℝ) (radius : ℝ) : 
  area = 11.25 → arc_length = 4.5 → area = (1/2) * radius * arc_length → radius = 5 := by
  sorry

end NUMINAMATH_CALUDE_sector_radius_l213_21384


namespace NUMINAMATH_CALUDE_equal_side_length_l213_21323

/-- An isosceles right-angled triangle with side lengths a, a, and c, where the sum of squares of sides is 725 --/
structure IsoscelesRightTriangle where
  a : ℝ
  c : ℝ
  isosceles : c^2 = 2 * a^2
  sum_of_squares : a^2 + a^2 + c^2 = 725

/-- The length of each equal side in the isosceles right-angled triangle is 13.5 --/
theorem equal_side_length (t : IsoscelesRightTriangle) : t.a = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_equal_side_length_l213_21323


namespace NUMINAMATH_CALUDE_marble_probability_l213_21338

theorem marble_probability (total_marbles : ℕ) 
  (prob_both_black : ℚ) (box1 box2 : ℕ) :
  total_marbles = 30 →
  box1 + box2 = total_marbles →
  prob_both_black = 3/5 →
  box1 > 0 ∧ box2 > 0 →
  ∃ (black1 black2 : ℕ),
    black1 ≤ box1 ∧ black2 ≤ box2 ∧
    (black1 : ℚ) / box1 * (black2 : ℚ) / box2 = prob_both_black →
    ((box1 - black1 : ℚ) / box1 * (box2 - black2 : ℚ) / box2 = 4/25) :=
by sorry

#check marble_probability

end NUMINAMATH_CALUDE_marble_probability_l213_21338


namespace NUMINAMATH_CALUDE_davids_english_marks_l213_21311

/-- Represents a student's marks in various subjects -/
structure StudentMarks where
  mathematics : ℕ
  physics : ℕ
  chemistry : ℕ
  biology : ℕ
  english : ℕ
  average : ℚ

/-- Theorem stating that given David's marks in other subjects and his average,
    his English marks must be 90 -/
theorem davids_english_marks (david : StudentMarks) 
  (math_marks : david.mathematics = 92)
  (physics_marks : david.physics = 85)
  (chemistry_marks : david.chemistry = 87)
  (biology_marks : david.biology = 85)
  (avg_marks : david.average = 87.8)
  : david.english = 90 := by
  sorry

#check davids_english_marks

end NUMINAMATH_CALUDE_davids_english_marks_l213_21311


namespace NUMINAMATH_CALUDE_horse_sale_problem_l213_21368

theorem horse_sale_problem (x : ℝ) : 
  (x - x^2 / 100 = 24) → (x = 40 ∨ x = 60) :=
by
  sorry

end NUMINAMATH_CALUDE_horse_sale_problem_l213_21368


namespace NUMINAMATH_CALUDE_distance_between_points_l213_21372

/-- The distance between points (0, 8) and (6, 0) is 10. -/
theorem distance_between_points : Real.sqrt ((6 - 0)^2 + (0 - 8)^2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l213_21372


namespace NUMINAMATH_CALUDE_second_derivative_f_l213_21331

noncomputable def f (x : ℝ) : ℝ := Real.exp x / x + x * Real.sin x

theorem second_derivative_f (x : ℝ) (hx : x ≠ 0) :
  (deriv^[2] f) x = (2 * x * Real.exp x * (1 - x)) / x^4 + Real.cos x - x * Real.sin x :=
sorry

end NUMINAMATH_CALUDE_second_derivative_f_l213_21331


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l213_21357

def A (a : ℝ) : Set ℝ := {a^2, a+1, -3}
def B (a : ℝ) : Set ℝ := {a-3, 2*a-1, a^2+1}

theorem intersection_implies_a_value (a : ℝ) :
  A a ∩ B a = {-3} → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l213_21357


namespace NUMINAMATH_CALUDE_product_of_divisors_equals_3_30_5_40_l213_21301

/-- The product of divisors function -/
def productOfDivisors (n : ℕ) : ℕ := sorry

/-- Theorem: If the product of all divisors of N equals 3^30 * 5^40, then N = 3^3 * 5^4 -/
theorem product_of_divisors_equals_3_30_5_40 (N : ℕ) :
  productOfDivisors N = 3^30 * 5^40 → N = 3^3 * 5^4 := by sorry

end NUMINAMATH_CALUDE_product_of_divisors_equals_3_30_5_40_l213_21301


namespace NUMINAMATH_CALUDE_range_of_a_l213_21326

/-- A function that is decreasing on R and defined piecewise --/
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.exp (-x) else -x^2 - 2*x + 1

/-- f is decreasing on R --/
axiom f_decreasing : ∀ x y : ℝ, x < y → f y < f x

/-- The theorem to prove --/
theorem range_of_a (a : ℝ) : f (a - 1) ≥ f (-a^2 + 1) ↔ -2 ≤ a ∧ a ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l213_21326


namespace NUMINAMATH_CALUDE_milk_box_width_l213_21378

/-- Represents a rectangular milk box -/
structure MilkBox where
  length : Real
  width : Real

/-- Calculates the volume of milk removed when lowering the level by a certain height -/
def volumeRemoved (box : MilkBox) (height : Real) : Real :=
  box.length * box.width * height

theorem milk_box_width (box : MilkBox) 
  (h1 : box.length = 50)
  (h2 : volumeRemoved box 0.5 = 4687.5 / 7.5) : 
  box.width = 25 := by
  sorry

#check milk_box_width

end NUMINAMATH_CALUDE_milk_box_width_l213_21378


namespace NUMINAMATH_CALUDE_tony_school_years_l213_21365

/-- The number of years Tony spent getting his initial science degree -/
def initial_degree_years : ℕ := 4

/-- The number of additional degrees Tony obtained -/
def additional_degrees : ℕ := 2

/-- The number of years each additional degree took -/
def years_per_additional_degree : ℕ := 4

/-- The number of years Tony spent getting his graduate degree in physics -/
def graduate_degree_years : ℕ := 2

/-- The total number of years Tony spent in school to become an astronaut -/
def total_years : ℕ := initial_degree_years + additional_degrees * years_per_additional_degree + graduate_degree_years

theorem tony_school_years : total_years = 14 := by
  sorry

end NUMINAMATH_CALUDE_tony_school_years_l213_21365


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l213_21392

theorem expression_simplification_and_evaluation :
  let a : ℝ := Real.sqrt 3
  let b : ℝ := Real.sqrt 3 - 1
  ((3 * a) / (2 * a - b) - 1) / ((a + b) / (4 * a^2 - b^2)) = 3 * Real.sqrt 3 - 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l213_21392


namespace NUMINAMATH_CALUDE_min_value_theorem_l213_21375

theorem min_value_theorem (a : ℝ) (h : a > 1) :
  a + 1 / (a - 1) ≥ 3 ∧ (a + 1 / (a - 1) = 3 ↔ a = 2) := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l213_21375


namespace NUMINAMATH_CALUDE_arc_length_120_degrees_l213_21387

/-- The arc length of a sector in a circle with radius π and central angle 120° -/
theorem arc_length_120_degrees (r : Real) (θ : Real) : 
  r = π → θ = 2 * π / 3 → 2 * π * r * (θ / (2 * π)) = 2 * π^2 / 3 := by
  sorry

#check arc_length_120_degrees

end NUMINAMATH_CALUDE_arc_length_120_degrees_l213_21387


namespace NUMINAMATH_CALUDE_probability_odd_sum_rows_l213_21317

/-- Represents a 4x3 grid filled with numbers 1 to 12 --/
def Grid := Fin 4 → Fin 3 → Fin 12

/-- Checks if a list of numbers has an odd sum --/
def has_odd_sum (row : List (Fin 12)) : Prop :=
  (row.map (fun n => n.val + 1)).sum % 2 = 1

/-- Represents a valid grid configuration --/
def valid_grid (g : Grid) : Prop :=
  ∀ i : Fin 4, has_odd_sum [g i 0, g i 1, g i 2]

/-- The total number of ways to arrange 12 numbers in a 4x3 grid --/
def total_arrangements : ℕ := 479001600

/-- The number of valid arrangements (where each row has an odd sum) --/
def valid_arrangements : ℕ := 21600

theorem probability_odd_sum_rows :
  (valid_arrangements : ℚ) / total_arrangements = 1 / 22176 :=
sorry

end NUMINAMATH_CALUDE_probability_odd_sum_rows_l213_21317


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l213_21353

theorem diophantine_equation_solutions :
  ∀ x y : ℤ, 9*x*y - x^2 - 8*y^2 = 2005 ↔ 
  (x = 63 ∧ y = 58) ∨ (x = 459 ∧ y = 58) ∨ (x = -63 ∧ y = -58) ∨ (x = -459 ∧ y = -58) := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l213_21353


namespace NUMINAMATH_CALUDE_cube_preserves_order_l213_21361

theorem cube_preserves_order (a b : ℝ) : a > b → a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_preserves_order_l213_21361


namespace NUMINAMATH_CALUDE_arithmetic_progression_five_digit_term_l213_21312

theorem arithmetic_progression_five_digit_term (n : ℕ) (k : ℕ) : 
  let a : ℕ → ℤ := λ i => -1 + (i - 1) * 19
  let is_all_fives : ℤ → Prop := λ x => ∃ m : ℕ, x = 5 * ((10^m - 1) / 9)
  (∃ n, is_all_fives (a n)) ↔ k = 3 ∧ 19 * n - 20 = 5 * ((10^k - 1) / 9) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_five_digit_term_l213_21312


namespace NUMINAMATH_CALUDE_inequality_equivalence_l213_21315

theorem inequality_equivalence (x : ℝ) : x + 1 < (4 + 3 * x) / 2 ↔ x > -2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l213_21315


namespace NUMINAMATH_CALUDE_solution_set_correct_l213_21394

-- Define the solution set
def solution_set : Set ℝ := {x | x < 0 ∨ x ≥ 1}

-- Define the inequality
def inequality (x : ℝ) : Prop := (1 - x) / x ≤ 0

-- Theorem stating that the solution set is correct
theorem solution_set_correct :
  ∀ x : ℝ, x ∈ solution_set ↔ inequality x :=
by sorry

end NUMINAMATH_CALUDE_solution_set_correct_l213_21394


namespace NUMINAMATH_CALUDE_biased_coin_probability_l213_21376

theorem biased_coin_probability (p : ℝ) : 
  p < (1 : ℝ) / 2 →
  (20 : ℝ) * p^3 * (1 - p)^3 = (5 : ℝ) / 32 →
  p = (1 - Real.sqrt ((32 - 4 * Real.rpow 5 (1/3)) / 8)) / 2 := by
sorry

end NUMINAMATH_CALUDE_biased_coin_probability_l213_21376


namespace NUMINAMATH_CALUDE_range_of_squared_sum_l213_21351

theorem range_of_squared_sum (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hsum : x + y = 1) :
  1/2 ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_squared_sum_l213_21351


namespace NUMINAMATH_CALUDE_trip_time_difference_l213_21355

/-- Proves that the difference in time between a 600-mile trip and a 540-mile trip,
    when traveling at a constant speed of 60 miles per hour, is 60 minutes. -/
theorem trip_time_difference (speed : ℝ) (distance1 : ℝ) (distance2 : ℝ) 
    (h1 : speed = 60) 
    (h2 : distance1 = 600) 
    (h3 : distance2 = 540) : 
  (distance1 / speed - distance2 / speed) * 60 = 60 := by
  sorry

#check trip_time_difference

end NUMINAMATH_CALUDE_trip_time_difference_l213_21355


namespace NUMINAMATH_CALUDE_equation_proof_l213_21321

theorem equation_proof : 361 + 2 * 19 * 6 + 36 = 625 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l213_21321


namespace NUMINAMATH_CALUDE_complex_number_proof_l213_21307

def i : ℂ := Complex.I

def is_real (z : ℂ) : Prop := z.im = 0

theorem complex_number_proof (z : ℂ) 
  (h1 : is_real (z + 2*i)) 
  (h2 : is_real (z / (2 - i))) : 
  z = 4 - 2*i ∧ Complex.abs (z / (1 + i)) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_proof_l213_21307


namespace NUMINAMATH_CALUDE_one_third_point_coordinates_l213_21316

/-- 
Given two points (x₁, y₁) and (x₂, y₂) in a 2D plane, and a rational number t between 0 and 1,
this function returns the coordinates of a point that is t of the way from (x₁, y₁) to (x₂, y₂).
-/
def pointOnLine (x₁ y₁ x₂ y₂ t : ℚ) : ℚ × ℚ :=
  ((1 - t) * x₁ + t * x₂, (1 - t) * y₁ + t * y₂)

theorem one_third_point_coordinates :
  let p := pointOnLine 2 6 8 (-2) (1/3)
  p.1 = 4 ∧ p.2 = 10/3 := by sorry

end NUMINAMATH_CALUDE_one_third_point_coordinates_l213_21316


namespace NUMINAMATH_CALUDE_balls_per_bag_l213_21397

theorem balls_per_bag (total_balls : ℕ) (num_bags : ℕ) (balls_per_bag : ℕ) : 
  total_balls = 36 → num_bags = 9 → total_balls = num_bags * balls_per_bag → balls_per_bag = 4 := by
  sorry

end NUMINAMATH_CALUDE_balls_per_bag_l213_21397


namespace NUMINAMATH_CALUDE_annual_mischief_convention_handshakes_l213_21358

/-- The number of handshakes at the Annual Mischief Convention -/
def total_handshakes (num_gremlins num_imps num_friendly_imps : ℕ) : ℕ :=
  let gremlin_handshakes := num_gremlins * (num_gremlins - 1) / 2
  let imp_gremlin_handshakes := num_imps * num_gremlins
  let friendly_imp_handshakes := num_friendly_imps * (num_friendly_imps - 1) / 2
  gremlin_handshakes + imp_gremlin_handshakes + friendly_imp_handshakes

/-- Theorem stating the total number of handshakes at the Annual Mischief Convention -/
theorem annual_mischief_convention_handshakes :
  total_handshakes 30 20 5 = 1045 := by
  sorry

end NUMINAMATH_CALUDE_annual_mischief_convention_handshakes_l213_21358


namespace NUMINAMATH_CALUDE_transaction_fraction_proof_l213_21342

theorem transaction_fraction_proof (mabel_transactions anthony_transactions cal_transactions jade_transactions : ℕ) :
  mabel_transactions = 90 →
  anthony_transactions = mabel_transactions + mabel_transactions / 10 →
  jade_transactions = 83 →
  jade_transactions = cal_transactions + 17 →
  3 * cal_transactions = 2 * anthony_transactions :=
by
  sorry

#check transaction_fraction_proof

end NUMINAMATH_CALUDE_transaction_fraction_proof_l213_21342


namespace NUMINAMATH_CALUDE_f_is_quadratic_l213_21377

/-- Definition of a quadratic equation in one variable -/
def is_quadratic_equation_in_one_variable (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x^2 + 15x - 7 = 0 -/
def f (x : ℝ) : ℝ := x^2 + 15*x - 7

/-- Theorem: f is a quadratic equation in one variable -/
theorem f_is_quadratic : is_quadratic_equation_in_one_variable f := by
  sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l213_21377


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l213_21382

theorem simplify_and_evaluate (a : ℚ) (h : a = -1/3) :
  (a + 1) * (a - 1) - a * (a + 3) = 0 := by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l213_21382


namespace NUMINAMATH_CALUDE_slices_per_pizza_l213_21309

theorem slices_per_pizza (total_pizzas : ℕ) (total_slices : ℕ) 
  (h1 : total_pizzas = 17) 
  (h2 : total_slices = 68) : 
  total_slices / total_pizzas = 4 := by
  sorry

end NUMINAMATH_CALUDE_slices_per_pizza_l213_21309


namespace NUMINAMATH_CALUDE_intersection_chord_length_l213_21341

noncomputable def line_l (x y : ℝ) : Prop := x + 2*y = 0

noncomputable def circle_C (x y : ℝ) : Prop := (x - Real.sqrt 2 / 2)^2 + (y - Real.sqrt 2 / 2)^2 = 2

theorem intersection_chord_length :
  ∀ (A B : ℝ × ℝ),
    line_l A.1 A.2 ∧ line_l B.1 B.2 ∧
    circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧
    A ≠ B →
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 1435 / 35 := by
  sorry

end NUMINAMATH_CALUDE_intersection_chord_length_l213_21341


namespace NUMINAMATH_CALUDE_divisible_by_1998_digit_sum_l213_21322

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: For all natural numbers n, if n is divisible by 1998, 
    then the sum of its digits is greater than or equal to 27 -/
theorem divisible_by_1998_digit_sum (n : ℕ) : 
  n % 1998 = 0 → sum_of_digits n ≥ 27 := by sorry

end NUMINAMATH_CALUDE_divisible_by_1998_digit_sum_l213_21322


namespace NUMINAMATH_CALUDE_eight_bead_bracelet_arrangements_l213_21364

/-- The number of distinct arrangements of n distinct beads on a bracelet,
    considering rotational and reflectional symmetries --/
def braceletArrangements (n : ℕ) : ℕ :=
  Nat.factorial n / (n * 2)

/-- Theorem stating that the number of distinct arrangements of 8 distinct beads
    on a bracelet, considering rotational and reflectional symmetries, is 2520 --/
theorem eight_bead_bracelet_arrangements :
  braceletArrangements 8 = 2520 := by
  sorry

end NUMINAMATH_CALUDE_eight_bead_bracelet_arrangements_l213_21364


namespace NUMINAMATH_CALUDE_liam_juice_consumption_l213_21318

/-- The number of glasses of juice Liam drinks in a given time period -/
def glasses_of_juice (time_minutes : ℕ) : ℕ :=
  time_minutes / 20

theorem liam_juice_consumption : glasses_of_juice 340 = 17 := by
  sorry

end NUMINAMATH_CALUDE_liam_juice_consumption_l213_21318


namespace NUMINAMATH_CALUDE_inequality_proof_l213_21395

-- Define the set M
def M : Set ℝ := {x | 0 < |x + 2| - |1 - x| ∧ |x + 2| - |1 - x| < 2}

-- State the theorem
theorem inequality_proof (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) :
  (|a + 1/2 * b| < 3/4) ∧ (|4 * a * b - 1| > 2 * |b - a|) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l213_21395


namespace NUMINAMATH_CALUDE_multiple_with_binary_digits_l213_21319

theorem multiple_with_binary_digits (n : ℕ) : ∃ m : ℕ, 
  (m % n = 0) ∧ 
  (Nat.digits 2 m).length = n ∧ 
  (∀ d ∈ Nat.digits 2 m, d = 0 ∨ d = 1) :=
sorry

end NUMINAMATH_CALUDE_multiple_with_binary_digits_l213_21319


namespace NUMINAMATH_CALUDE_linda_savings_l213_21352

theorem linda_savings (tv_cost : ℝ) (tv_fraction : ℝ) : 
  tv_cost = 300 → tv_fraction = 1/2 → tv_cost / tv_fraction = 600 := by
  sorry

end NUMINAMATH_CALUDE_linda_savings_l213_21352


namespace NUMINAMATH_CALUDE_arithmetic_simplification_l213_21393

theorem arithmetic_simplification : (4 + 4 + 6) / 2 - 2 / 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_simplification_l213_21393


namespace NUMINAMATH_CALUDE_physics_marks_l213_21391

theorem physics_marks (P C M : ℝ) 
  (total_avg : (P + C + M) / 3 = 85)
  (phys_math_avg : (P + M) / 2 = 90)
  (phys_chem_avg : (P + C) / 2 = 70) :
  P = 65 := by
sorry

end NUMINAMATH_CALUDE_physics_marks_l213_21391


namespace NUMINAMATH_CALUDE_ashutosh_completion_time_l213_21329

/-- The time it takes Suresh to complete the job alone -/
def suresh_time : ℝ := 15

/-- The time Suresh works on the job -/
def suresh_work_time : ℝ := 9

/-- The time it takes Ashutosh to complete the remaining job -/
def ashutosh_remaining_time : ℝ := 14

/-- The time it takes Ashutosh to complete the job alone -/
def ashutosh_time : ℝ := 35

theorem ashutosh_completion_time :
  (suresh_work_time / suresh_time) + 
  ((1 - suresh_work_time / suresh_time) / ashutosh_time) = 
  (1 / ashutosh_remaining_time) := by
  sorry

#check ashutosh_completion_time

end NUMINAMATH_CALUDE_ashutosh_completion_time_l213_21329


namespace NUMINAMATH_CALUDE_p_sufficient_but_not_necessary_for_q_l213_21350

-- Define the propositions
variable (p q r s : Prop)

-- Define the relationships between the propositions
variable (h1 : p → r)  -- p is sufficient for r
variable (h2 : r → s)  -- s is necessary for r
variable (h3 : s → q)  -- q is necessary for s

-- State the theorem
theorem p_sufficient_but_not_necessary_for_q :
  (p → q) ∧ ¬(q → p) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_but_not_necessary_for_q_l213_21350


namespace NUMINAMATH_CALUDE_miss_grayson_class_size_l213_21369

/-- The number of students in Miss Grayson's class -/
def num_students : ℕ := sorry

/-- The amount raised by the students -/
def amount_raised : ℕ := sorry

/-- The cost of the trip -/
def trip_cost : ℕ := sorry

/-- The remaining fund after the trip -/
def remaining_fund : ℕ := sorry

theorem miss_grayson_class_size :
  (amount_raised = num_students * 5) →
  (trip_cost = num_students * 7) →
  (amount_raised - trip_cost = remaining_fund) →
  (remaining_fund = 10) →
  (num_students = 5) := by sorry

end NUMINAMATH_CALUDE_miss_grayson_class_size_l213_21369


namespace NUMINAMATH_CALUDE_activities_equally_popular_l213_21328

def dodgeball : Rat := 10 / 25
def artWorkshop : Rat := 12 / 30
def movieScreening : Rat := 18 / 45
def quizBowl : Rat := 16 / 40

theorem activities_equally_popular :
  dodgeball = artWorkshop ∧
  artWorkshop = movieScreening ∧
  movieScreening = quizBowl := by
  sorry

end NUMINAMATH_CALUDE_activities_equally_popular_l213_21328


namespace NUMINAMATH_CALUDE_alice_wins_coin_game_l213_21344

def coin_game (initial_coins : ℕ) : Prop :=
  ∃ (k : ℕ),
    k^2 ≤ initial_coins ∧
    initial_coins < k * (k + 1) - 1

theorem alice_wins_coin_game :
  coin_game 1331 :=
sorry

end NUMINAMATH_CALUDE_alice_wins_coin_game_l213_21344


namespace NUMINAMATH_CALUDE_fencing_cost_rectangle_l213_21332

/-- Proves that for a rectangular field with sides in the ratio 3:4 and an area of 8748 sq. m, 
    the cost of fencing at 25 paise per metre is 94.5 rupees. -/
theorem fencing_cost_rectangle (length width : ℝ) (area perimeter cost_per_meter total_cost : ℝ) : 
  length / width = 4 / 3 →
  area = 8748 →
  area = length * width →
  perimeter = 2 * (length + width) →
  cost_per_meter = 0.25 →
  total_cost = perimeter * cost_per_meter →
  total_cost = 94.5 := by
sorry

end NUMINAMATH_CALUDE_fencing_cost_rectangle_l213_21332


namespace NUMINAMATH_CALUDE_functional_equation_solution_l213_21346

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x^2 - y^2) = (x - y) * (f x + f y)) :
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l213_21346


namespace NUMINAMATH_CALUDE_ammonium_chloride_formed_l213_21399

-- Define the reaction components
variable (NH3 : ℝ) -- Moles of Ammonia
variable (HCl : ℝ) -- Moles of Hydrochloric acid
variable (NH4Cl : ℝ) -- Moles of Ammonium chloride

-- Define the conditions
axiom ammonia_moles : NH3 = 3
axiom total_product : NH4Cl = 3

-- Theorem to prove
theorem ammonium_chloride_formed : NH4Cl = 3 :=
by sorry

end NUMINAMATH_CALUDE_ammonium_chloride_formed_l213_21399


namespace NUMINAMATH_CALUDE_number_puzzle_l213_21330

theorem number_puzzle (N : ℚ) : (5/4) * N = (4/5) * N + 45 → N = 100 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l213_21330


namespace NUMINAMATH_CALUDE_chairs_arrangement_l213_21300

/-- Given a total number of chairs and chairs per row, calculates the number of rows -/
def calculate_rows (total_chairs : ℕ) (chairs_per_row : ℕ) : ℕ :=
  total_chairs / chairs_per_row

/-- Theorem: For 432 chairs arranged in rows of 16, there are 27 rows -/
theorem chairs_arrangement :
  calculate_rows 432 16 = 27 := by
  sorry

end NUMINAMATH_CALUDE_chairs_arrangement_l213_21300


namespace NUMINAMATH_CALUDE_b_is_positive_l213_21337

theorem b_is_positive (x a : ℤ) (h1 : x < a) (h2 : a < 0) (b : ℤ) (h3 : b = x^2 - a^2) : b > 0 := by
  sorry

end NUMINAMATH_CALUDE_b_is_positive_l213_21337


namespace NUMINAMATH_CALUDE_exists_monomial_with_conditions_l213_21339

/-- A monomial is a product of a coefficient and variables raised to non-negative integer powers. -/
structure Monomial (α : Type*) [Semiring α] where
  coeff : α
  powers : List (Nat × Nat)

/-- The degree of a monomial is the sum of the exponents of its variables. -/
def Monomial.degree {α : Type*} [Semiring α] (m : Monomial α) : Nat :=
  m.powers.foldl (fun acc (_, pow) => acc + pow) 0

/-- A monomial contains specific variables if they appear in its power list. -/
def Monomial.containsVariables {α : Type*} [Semiring α] (m : Monomial α) (vars : List Nat) : Prop :=
  ∀ v ∈ vars, ∃ (pow : Nat), (v, pow) ∈ m.powers

/-- There exists a monomial with coefficient 3, containing variables x and y, and having a total degree of 3. -/
theorem exists_monomial_with_conditions :
  ∃ (m : Monomial ℕ),
    m.coeff = 3 ∧
    m.containsVariables [1, 2] ∧  -- Let 1 represent x and 2 represent y
    m.degree = 3 := by
  sorry

end NUMINAMATH_CALUDE_exists_monomial_with_conditions_l213_21339


namespace NUMINAMATH_CALUDE_solve_linear_equation_l213_21396

theorem solve_linear_equation (x : ℝ) (h : 3*x - 4*x + 7*x = 120) : x = 20 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l213_21396


namespace NUMINAMATH_CALUDE_indeterminate_157th_digit_l213_21370

/-- The decimal representation of a rational number -/
def decimal_representation (q : ℚ) : ℕ → ℕ :=
  sorry

/-- The nth digit after the decimal point in the decimal representation of q -/
def nth_digit_after_decimal (q : ℚ) (n : ℕ) : ℕ :=
  sorry

theorem indeterminate_157th_digit :
  ∀ (d : ℕ),
  (∃ (q : ℚ), q = 525 / 2027 ∧ 
    (∀ (i : ℕ), i < 6 → nth_digit_after_decimal q (i + 1) = nth_digit_after_decimal (0.258973 : ℚ) (i + 1))) →
  (∃ (r : ℚ), r ≠ q ∧ 
    (∀ (i : ℕ), i < 6 → nth_digit_after_decimal r (i + 1) = nth_digit_after_decimal (0.258973 : ℚ) (i + 1)) ∧
    nth_digit_after_decimal r 157 ≠ d) :=
by
  sorry


end NUMINAMATH_CALUDE_indeterminate_157th_digit_l213_21370


namespace NUMINAMATH_CALUDE_orthocenter_locus_l213_21359

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse (a b : ℝ) where
  (ha : a > 0)
  (hb : b > 0)
  (hba : b ≤ a)

/-- A triangle inscribed in an ellipse -/
structure InscribedTriangle (e : Ellipse a b) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  hA : A = (-a, 0)
  hB : B = (a, 0)
  hC : (C.1^2 / a^2) + (C.2^2 / b^2) = 1

/-- The orthocenter of a triangle -/
def orthocenter (t : InscribedTriangle e) : ℝ × ℝ :=
  sorry

/-- The locus of the orthocenter is an ellipse -/
theorem orthocenter_locus (e : Ellipse a b) :
  ∀ t : InscribedTriangle e,
  let M := orthocenter t
  ((M.1^2 / a^2) + (M.2^2 / (a^2/b)^2) = 1) :=
sorry

end NUMINAMATH_CALUDE_orthocenter_locus_l213_21359


namespace NUMINAMATH_CALUDE_multiple_of_9_digit_sum_possible_digits_for_multiple_of_9_l213_21362

def is_multiple_of_9 (n : ℕ) : Prop := ∃ k : ℕ, n = 9 * k

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.repr.data.map (λ c => c.toNat - '0'.toNat)
  digits.sum

theorem multiple_of_9_digit_sum (n : ℕ) : is_multiple_of_9 n ↔ is_multiple_of_9 (digit_sum n) := by sorry

theorem possible_digits_for_multiple_of_9 :
  ∀ d : ℕ, d < 10 →
    (is_multiple_of_9 (86300 + d * 10 + 7) ↔ d = 3 ∨ d = 9) := by sorry

end NUMINAMATH_CALUDE_multiple_of_9_digit_sum_possible_digits_for_multiple_of_9_l213_21362


namespace NUMINAMATH_CALUDE_solve_equation_l213_21324

theorem solve_equation : ∃ x : ℝ, (2 * x + 7) / 5 = 17 ∧ x = 39 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l213_21324


namespace NUMINAMATH_CALUDE_water_needed_for_lemonade_l213_21310

-- Define the ratio of water to lemon juice
def water_ratio : ℚ := 4
def lemon_juice_ratio : ℚ := 1

-- Define the total volume in gallons
def total_volume : ℚ := 3

-- Define the conversion factor from gallons to quarts
def quarts_per_gallon : ℚ := 4

-- Theorem statement
theorem water_needed_for_lemonade :
  let total_ratio : ℚ := water_ratio + lemon_juice_ratio
  let total_quarts : ℚ := total_volume * quarts_per_gallon
  let quarts_per_part : ℚ := total_quarts / total_ratio
  let water_quarts : ℚ := water_ratio * quarts_per_part
  water_quarts = 9.6 := by
  sorry

end NUMINAMATH_CALUDE_water_needed_for_lemonade_l213_21310


namespace NUMINAMATH_CALUDE_total_molecular_weight_l213_21343

-- Define atomic weights
def carbon_weight : ℝ := 12.01
def hydrogen_weight : ℝ := 1.008
def oxygen_weight : ℝ := 16.00

-- Define molecular formulas
def ascorbic_acid_carbon : ℕ := 6
def ascorbic_acid_hydrogen : ℕ := 8
def ascorbic_acid_oxygen : ℕ := 6

def citric_acid_carbon : ℕ := 6
def citric_acid_hydrogen : ℕ := 8
def citric_acid_oxygen : ℕ := 7

-- Define number of moles
def ascorbic_acid_moles : ℕ := 7
def citric_acid_moles : ℕ := 5

-- Calculate molecular weights
def ascorbic_acid_weight : ℝ :=
  (ascorbic_acid_carbon * carbon_weight) +
  (ascorbic_acid_hydrogen * hydrogen_weight) +
  (ascorbic_acid_oxygen * oxygen_weight)

def citric_acid_weight : ℝ :=
  (citric_acid_carbon * carbon_weight) +
  (citric_acid_hydrogen * hydrogen_weight) +
  (citric_acid_oxygen * oxygen_weight)

-- Theorem statement
theorem total_molecular_weight :
  (ascorbic_acid_moles * ascorbic_acid_weight) +
  (citric_acid_moles * citric_acid_weight) = 2193.488 :=
by sorry

end NUMINAMATH_CALUDE_total_molecular_weight_l213_21343


namespace NUMINAMATH_CALUDE_log_sum_equation_l213_21381

theorem log_sum_equation (x y z : ℝ) (hx : x = 625) (hy : y = 5) (hz : z = 1/25) :
  Real.log x / Real.log 5 + Real.log y / Real.log 5 - Real.log z / Real.log 5 = 7 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equation_l213_21381


namespace NUMINAMATH_CALUDE_eve_gift_cost_is_135_l213_21366

/-- The cost of Eve's gifts for her nieces --/
def eve_gift_cost : ℝ :=
  let hand_mitts : ℝ := 14
  let apron : ℝ := 16
  let utensils : ℝ := 10
  let knife : ℝ := 2 * utensils
  let cost_per_niece : ℝ := hand_mitts + apron + utensils + knife
  let total_cost : ℝ := 3 * cost_per_niece
  let discount_rate : ℝ := 0.25
  let discounted_cost : ℝ := total_cost * (1 - discount_rate)
  discounted_cost

theorem eve_gift_cost_is_135 : eve_gift_cost = 135 := by
  sorry

end NUMINAMATH_CALUDE_eve_gift_cost_is_135_l213_21366


namespace NUMINAMATH_CALUDE_parabola_translation_l213_21389

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h + p.b
    c := p.a * h^2 - p.b * h + p.c - v }

theorem parabola_translation :
  let p : Parabola := { a := 1, b := 2, c := -1 }
  let translated_p := translate p 2 1
  translated_p = { a := 1, b := -2, c := -3 } :=
by sorry

end NUMINAMATH_CALUDE_parabola_translation_l213_21389


namespace NUMINAMATH_CALUDE_cuboid_surface_area_l213_21383

/-- The surface area of a rectangular parallelepiped with given dimensions -/
theorem cuboid_surface_area (w : ℝ) (h l : ℝ) : 
  w = 4 →
  l = w + 6 →
  h = l + 5 →
  2 * l * w + 2 * l * h + 2 * w * h = 500 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_surface_area_l213_21383


namespace NUMINAMATH_CALUDE_fraction_simplification_l213_21385

theorem fraction_simplification :
  (1 / 5 + 1 / 7) / ((2 / 3 - 1 / 4) * 2 / 5) = 72 / 35 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l213_21385


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l213_21347

/-- Given two vectors a and b in ℝ², and a real number k, 
    we define vector c as a sum of a and k * b. 
    If b is perpendicular to c, then k equals -3. -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (k : ℝ) :
  a = (10, 20) →
  b = (5, 5) →
  let c := (a.1 + k * b.1, a.2 + k * b.2)
  (b.1 * c.1 + b.2 * c.2 = 0) →
  k = -3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l213_21347


namespace NUMINAMATH_CALUDE_cars_without_ac_l213_21320

theorem cars_without_ac (total : ℕ) (min_racing : ℕ) (max_ac_no_racing : ℕ)
  (h_total : total = 100)
  (h_min_racing : min_racing = 51)
  (h_max_ac_no_racing : max_ac_no_racing = 49) :
  total - (max_ac_no_racing + (min_racing - max_ac_no_racing)) = 49 := by
  sorry

end NUMINAMATH_CALUDE_cars_without_ac_l213_21320


namespace NUMINAMATH_CALUDE_intersection_points_vary_l213_21313

theorem intersection_points_vary (A B C : ℝ) (hA : A > 0) (hC : C > 0) (hB : B ≥ 0) :
  ∃ x y : ℝ, y = A * x^2 + B * x + C ∧ y^2 + 2 * x = x^2 + 4 * y + C ∧
  ∃ A' B' C' : ℝ, A' > 0 ∧ C' > 0 ∧ B' ≥ 0 ∧
    (∃ x1 y1 x2 y2 : ℝ, x1 ≠ x2 ∧
      y1 = A' * x1^2 + B' * x1 + C' ∧ y1^2 + 2 * x1 = x1^2 + 4 * y1 + C' ∧
      y2 = A' * x2^2 + B' * x2 + C' ∧ y2^2 + 2 * x2 = x2^2 + 4 * y2 + C') :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_vary_l213_21313


namespace NUMINAMATH_CALUDE_unique_digit_product_l213_21398

theorem unique_digit_product (n : ℕ) : n ≤ 9 → (n * (10 * n + n) = 176) ↔ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_digit_product_l213_21398


namespace NUMINAMATH_CALUDE_common_chord_of_circles_l213_21388

/-- Represents a circle in the 2D plane -/
structure Circle where
  a : ℝ  -- x^2 coefficient
  b : ℝ  -- y^2 coefficient
  c : ℝ  -- x coefficient
  d : ℝ  -- y coefficient
  e : ℝ  -- constant term

/-- Represents a line in the 2D plane -/
structure Line where
  a : ℝ  -- x coefficient
  b : ℝ  -- y coefficient
  c : ℝ  -- constant term

/-- Definition of the first circle -/
def circle1 : Circle := { a := 1, b := 1, c := -2, d := 0, e := -4 }

/-- Definition of the second circle -/
def circle2 : Circle := { a := 1, b := 1, c := 0, d := 2, e := -6 }

/-- The common chord line -/
def commonChord : Line := { a := 1, b := 1, c := -1 }

/-- Theorem: The given line is the common chord of the two circles -/
theorem common_chord_of_circles :
  commonChord = Line.mk 1 1 (-1) ∧
  (∀ x y : ℝ, x + y - 1 = 0 →
    (x^2 + y^2 - 2*x - 4 = 0 ↔ x^2 + y^2 + 2*y - 6 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_common_chord_of_circles_l213_21388


namespace NUMINAMATH_CALUDE_rancher_animals_count_l213_21373

theorem rancher_animals_count : ∀ (horses cows total : ℕ),
  cows = 5 * horses →
  cows = 140 →
  total = cows + horses →
  total = 168 :=
by
  sorry

end NUMINAMATH_CALUDE_rancher_animals_count_l213_21373


namespace NUMINAMATH_CALUDE_square_rectangle_area_relation_l213_21336

theorem square_rectangle_area_relation (x : ℝ) :
  let square_side := x - 4
  let rect_length := x - 5
  let rect_width := x + 6
  let square_area := square_side ^ 2
  let rect_area := rect_length * rect_width
  rect_area = 3 * square_area →
  ∃ x₁ x₂ : ℝ, (x = x₁ ∨ x = x₂) ∧ x₁ + x₂ = 12.5 :=
by sorry

end NUMINAMATH_CALUDE_square_rectangle_area_relation_l213_21336


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l213_21314

/-- Given a hyperbola with equation mx^2 + y^2 = 1, if one of its asymptotes has a slope of 2, then m = -4 -/
theorem hyperbola_asymptote_slope (m : ℝ) : 
  (∃ (x y : ℝ), m * x^2 + y^2 = 1) →  -- Hyperbola equation exists
  (∃ (k : ℝ), k = 2 ∧ k^2 = -m) →    -- One asymptote has slope 2
  m = -4 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l213_21314


namespace NUMINAMATH_CALUDE_geometric_sequence_alternating_l213_21348

def is_alternating_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n * a (n + 1) < 0

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_alternating
  (a : ℕ → ℝ)
  (h_geometric : is_geometric_sequence a)
  (h_sum1 : a 1 + a 2 = -3/2)
  (h_sum2 : a 4 + a 5 = 12) :
  is_alternating_sequence a :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_alternating_l213_21348


namespace NUMINAMATH_CALUDE_net_population_change_l213_21386

def population_change (initial : ℝ) : ℝ :=
  initial * (1.2 * 0.9 * 1.3 * 0.85)

theorem net_population_change :
  ∀ initial : ℝ, initial > 0 →
  let final := population_change initial
  let percent_change := (final - initial) / initial * 100
  round percent_change = 51 := by
  sorry

#check net_population_change

end NUMINAMATH_CALUDE_net_population_change_l213_21386


namespace NUMINAMATH_CALUDE_unique_group_size_l213_21327

theorem unique_group_size (n : ℕ) (k : ℕ) : 
  (∀ (i j : Fin n), i ≠ j → ∃! (call : Bool), call) →
  (∀ (subset : Finset (Fin n)), subset.card = n - 2 → 
    (subset.sum (λ i => (subset.filter (λ j => j ≠ i)).card) / 2) = 3^k) →
  n = 5 :=
sorry

end NUMINAMATH_CALUDE_unique_group_size_l213_21327


namespace NUMINAMATH_CALUDE_betty_savings_ratio_l213_21333

theorem betty_savings_ratio (wallet_cost parents_gift grandparents_gift needed_more initial_savings : ℚ) :
  wallet_cost = 100 →
  parents_gift = 15 →
  grandparents_gift = 2 * parents_gift →
  needed_more = 5 →
  initial_savings + parents_gift + grandparents_gift = wallet_cost - needed_more →
  initial_savings / wallet_cost = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_betty_savings_ratio_l213_21333


namespace NUMINAMATH_CALUDE_nina_weekend_sales_l213_21304

/-- Calculates the total money Nina made from jewelry sales over the weekend -/
def weekend_sales (necklace_price bracelet_price earring_price ensemble_price : ℚ)
                  (necklaces_sold bracelets_sold earrings_sold ensembles_sold : ℕ) : ℚ :=
  necklace_price * necklaces_sold +
  bracelet_price * bracelets_sold +
  earring_price * earrings_sold +
  ensemble_price * ensembles_sold

/-- Proves that Nina's weekend sales totaled $565.00 -/
theorem nina_weekend_sales :
  weekend_sales 25 15 10 45 5 10 20 2 = 565 := by
  sorry

end NUMINAMATH_CALUDE_nina_weekend_sales_l213_21304


namespace NUMINAMATH_CALUDE_total_oranges_picked_l213_21303

theorem total_oranges_picked (mary_oranges jason_oranges amanda_oranges : ℕ)
  (h1 : mary_oranges = 14)
  (h2 : jason_oranges = 41)
  (h3 : amanda_oranges = 56) :
  mary_oranges + jason_oranges + amanda_oranges = 111 := by
  sorry

end NUMINAMATH_CALUDE_total_oranges_picked_l213_21303


namespace NUMINAMATH_CALUDE_complex_multiplication_l213_21345

def i : ℂ := Complex.I

theorem complex_multiplication :
  (1 + i) * (3 - i) = 4 + 2*i := by sorry

end NUMINAMATH_CALUDE_complex_multiplication_l213_21345


namespace NUMINAMATH_CALUDE_new_sequence_69th_is_original_18th_l213_21305

/-- Given a sequence, insert_between n seq inserts n elements between each pair of adjacent elements in seq -/
def insert_between (n : ℕ) (seq : ℕ → ℕ) : ℕ → ℕ := sorry

/-- The original sequence -/
def original_sequence : ℕ → ℕ := sorry

/-- The new sequence with 3 elements inserted between each pair of adjacent elements -/
def new_sequence : ℕ → ℕ := insert_between 3 original_sequence

theorem new_sequence_69th_is_original_18th :
  new_sequence 69 = original_sequence 18 := by sorry

end NUMINAMATH_CALUDE_new_sequence_69th_is_original_18th_l213_21305


namespace NUMINAMATH_CALUDE_equidistant_point_l213_21349

theorem equidistant_point (x y : ℝ) : 
  let d_y_axis := |x|
  let d_line1 := |x + y - 1| / Real.sqrt 2
  let d_line2 := |y - 3*x| / Real.sqrt 10
  (d_y_axis = d_line1 ∧ d_y_axis = d_line2) → 
  x = 1 / (4 + Real.sqrt 10 - Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_equidistant_point_l213_21349


namespace NUMINAMATH_CALUDE_print_shop_charge_l213_21390

/-- The charge per color copy at print shop X -/
def charge_X : ℝ := 1.25

/-- The charge per color copy at print shop Y -/
def charge_Y : ℝ := 2.75

/-- The number of color copies -/
def num_copies : ℕ := 60

/-- The difference in total charge between print shops Y and X -/
def charge_difference : ℝ := 90

theorem print_shop_charge : 
  charge_X * num_copies + charge_difference = charge_Y * num_copies := by
  sorry

end NUMINAMATH_CALUDE_print_shop_charge_l213_21390


namespace NUMINAMATH_CALUDE_ipads_sold_l213_21356

/-- Proves that the number of iPads sold is 20 given the conditions of the problem -/
theorem ipads_sold (iphones : ℕ) (ipads : ℕ) (apple_tvs : ℕ) 
  (iphone_cost : ℝ) (ipad_cost : ℝ) (apple_tv_cost : ℝ) (average_cost : ℝ) :
  iphones = 100 →
  apple_tvs = 80 →
  iphone_cost = 1000 →
  ipad_cost = 900 →
  apple_tv_cost = 200 →
  average_cost = 670 →
  (iphones * iphone_cost + ipads * ipad_cost + apple_tvs * apple_tv_cost) / 
    (iphones + ipads + apple_tvs : ℝ) = average_cost →
  ipads = 20 := by
  sorry

#check ipads_sold

end NUMINAMATH_CALUDE_ipads_sold_l213_21356


namespace NUMINAMATH_CALUDE_blue_faces_cube_l213_21306

theorem blue_faces_cube (n : ℕ) (h : n > 0) : 
  (6 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1 / 3 → n = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_blue_faces_cube_l213_21306


namespace NUMINAMATH_CALUDE_circle_tripled_radius_l213_21325

theorem circle_tripled_radius (r : ℝ) (h : r > 0) :
  let new_r := 3 * r
  let original_area := π * r^2
  let new_area := π * new_r^2
  let original_circumference := 2 * π * r
  let new_circumference := 2 * π * new_r
  (new_area = 9 * original_area) ∧ (new_circumference = 3 * original_circumference) := by
  sorry

end NUMINAMATH_CALUDE_circle_tripled_radius_l213_21325


namespace NUMINAMATH_CALUDE_sequence_minus_two_is_geometric_l213_21367

/-- Given a sequence a and its partial sums s, prove {a n - 2} is geometric -/
theorem sequence_minus_two_is_geometric
  (a : ℕ+ → ℝ)  -- The sequence a_n
  (s : ℕ+ → ℝ)  -- The sequence of partial sums s_n
  (h : ∀ n : ℕ+, s n + a n = 2 * n)  -- The given condition
  : ∃ r : ℝ, ∀ n : ℕ+, a (n + 1) - 2 = r * (a n - 2) :=
sorry

end NUMINAMATH_CALUDE_sequence_minus_two_is_geometric_l213_21367


namespace NUMINAMATH_CALUDE_find_a_empty_solution_set_l213_21302

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Theorem for part (1)
theorem find_a : 
  ∀ a : ℝ, (∀ x : ℝ, f a (2*x) ≤ 4 ↔ 0 ≤ x ∧ x ≤ 4) → a = 4 := by sorry

-- Theorem for part (2)
theorem empty_solution_set (m : ℝ) : 
  (∀ x : ℝ, ¬(f 4 x + f 4 (x + m) < 2)) ↔ (m ≥ 2 ∨ m ≤ -2) := by sorry

end NUMINAMATH_CALUDE_find_a_empty_solution_set_l213_21302
