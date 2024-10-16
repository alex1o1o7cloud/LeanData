import Mathlib

namespace NUMINAMATH_CALUDE_number_of_type_C_is_16_l3403_340318

/-- Represents the types of people in the problem -/
inductive PersonType
| A
| B
| C

/-- The total number of people -/
def total_people : ℕ := 25

/-- The number of people who answered "yes" to "Are you a Type A person?" -/
def yes_to_A : ℕ := 17

/-- The number of people who answered "yes" to "Are you a Type C person?" -/
def yes_to_C : ℕ := 12

/-- The number of people who answered "yes" to "Are you a Type B person?" -/
def yes_to_B : ℕ := 8

/-- Theorem stating that the number of Type C people is 16 -/
theorem number_of_type_C_is_16 :
  ∃ (a b c : ℕ),
    a + b + c = total_people ∧
    a + b + (c / 2) = yes_to_A ∧
    b + (c / 2) = yes_to_C ∧
    c / 2 = yes_to_B ∧
    c = 16 := by
  sorry

end NUMINAMATH_CALUDE_number_of_type_C_is_16_l3403_340318


namespace NUMINAMATH_CALUDE_sundae_price_l3403_340334

/-- Given the following conditions:
  * The caterer ordered 125 ice-cream bars
  * The caterer ordered 125 sundaes
  * The total price was $225.00
  * The price of each ice-cream bar was $0.60
Prove that the price of each sundae was $1.20 -/
theorem sundae_price 
  (num_ice_cream : ℕ) 
  (num_sundae : ℕ) 
  (total_price : ℚ) 
  (ice_cream_price : ℚ) 
  (h1 : num_ice_cream = 125)
  (h2 : num_sundae = 125)
  (h3 : total_price = 225)
  (h4 : ice_cream_price = 6/10) : 
  (total_price - num_ice_cream * ice_cream_price) / num_sundae = 12/10 := by
  sorry


end NUMINAMATH_CALUDE_sundae_price_l3403_340334


namespace NUMINAMATH_CALUDE_acute_angles_equal_l3403_340362

/-- A circle with a rhombus and an isosceles trapezoid inscribed around it -/
structure InscribedFigures where
  /-- Radius of the inscribed circle -/
  r : ℝ
  /-- Acute angle of the rhombus -/
  α : ℝ
  /-- Acute angle of the isosceles trapezoid -/
  β : ℝ
  /-- The rhombus and trapezoid are inscribed around the same circle -/
  inscribed : r > 0
  /-- The areas of the rhombus and trapezoid are equal -/
  equal_areas : (4 * r^2) / Real.sin α = (4 * r^2) / Real.sin β

/-- 
Given a rhombus and an isosceles trapezoid inscribed around the same circle with equal areas,
their acute angles are equal.
-/
theorem acute_angles_equal (fig : InscribedFigures) : fig.α = fig.β :=
  sorry

end NUMINAMATH_CALUDE_acute_angles_equal_l3403_340362


namespace NUMINAMATH_CALUDE_all_boys_are_brothers_l3403_340333

/-- A type representing the group of boys -/
def Boys := Fin 7

/-- A relation indicating whether two boys are brothers -/
def is_brother (a b : Boys) : Prop := sorry

/-- Axiom: Each boy has at least 3 brothers among the others -/
axiom at_least_three_brothers (b : Boys) : 
  ∃ (s : Finset Boys), s.card ≥ 3 ∧ ∀ x ∈ s, x ≠ b ∧ is_brother x b

/-- Theorem: All seven boys are brothers -/
theorem all_boys_are_brothers : ∀ (a b : Boys), is_brother a b :=
sorry

end NUMINAMATH_CALUDE_all_boys_are_brothers_l3403_340333


namespace NUMINAMATH_CALUDE_daltons_uncle_gift_l3403_340374

/-- The amount of money Dalton's uncle gave him -/
def uncles_gift (jump_rope_cost board_game_cost playground_ball_cost savings needed : ℕ) : ℕ :=
  jump_rope_cost + board_game_cost + playground_ball_cost - savings - needed

theorem daltons_uncle_gift :
  uncles_gift 7 12 4 6 4 = 13 := by
  sorry

end NUMINAMATH_CALUDE_daltons_uncle_gift_l3403_340374


namespace NUMINAMATH_CALUDE_equation_solution_l3403_340361

theorem equation_solution (x : ℝ) (hx : x ≠ 0) :
  3 - 5 / x + 2 / (x^2) = 0 → (3 / x = 9 / 2 ∨ 3 / x = 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3403_340361


namespace NUMINAMATH_CALUDE_visit_neither_country_l3403_340388

theorem visit_neither_country (total : ℕ) (iceland : ℕ) (norway : ℕ) (both : ℕ) :
  total = 50 →
  iceland = 25 →
  norway = 23 →
  both = 21 →
  total - (iceland + norway - both) = 23 := by
  sorry

end NUMINAMATH_CALUDE_visit_neither_country_l3403_340388


namespace NUMINAMATH_CALUDE_interior_angles_sum_l3403_340358

theorem interior_angles_sum (n : ℕ) : 
  (180 * (n - 2) = 3240) → (180 * ((n + 3) - 2) = 3780) := by
  sorry

end NUMINAMATH_CALUDE_interior_angles_sum_l3403_340358


namespace NUMINAMATH_CALUDE_combined_work_theorem_l3403_340314

/-- The time taken for three workers to complete a task together, given their individual completion times -/
def combined_completion_time (time_A time_B time_C : ℚ) : ℚ :=
  1 / (1 / time_A + 1 / time_B + 1 / time_C)

/-- Theorem: Given the individual completion times, the combined completion time is 72/13 days -/
theorem combined_work_theorem :
  combined_completion_time 12 18 24 = 72 / 13 := by
  sorry

end NUMINAMATH_CALUDE_combined_work_theorem_l3403_340314


namespace NUMINAMATH_CALUDE_exam_max_marks_calculation_l3403_340373

/-- Represents the maximum marks and passing criteria for a subject -/
structure Subject where
  max_marks : ℕ
  passing_percentage : ℚ

/-- Represents a student's performance in a subject -/
structure Performance where
  score : ℕ
  failed_by : ℕ

/-- Calculates the maximum marks for a subject given the performance and passing criteria -/
def calculate_max_marks (perf : Performance) (pass_percentage : ℚ) : ℕ :=
  ((perf.score + perf.failed_by : ℚ) / pass_percentage).ceil.toNat

theorem exam_max_marks_calculation (math science english : Subject) 
    (math_perf science_perf english_perf : Performance) : 
    math.max_marks = 275 ∧ science.max_marks = 414 ∧ english.max_marks = 300 :=
  by
    have h_math : math.passing_percentage = 2/5 := by sorry
    have h_science : science.passing_percentage = 7/20 := by sorry
    have h_english : english.passing_percentage = 3/10 := by sorry
    
    have h_math_perf : math_perf = ⟨90, 20⟩ := by sorry
    have h_science_perf : science_perf = ⟨110, 35⟩ := by sorry
    have h_english_perf : english_perf = ⟨80, 10⟩ := by sorry
    
    have h_math_max : math.max_marks = calculate_max_marks math_perf math.passing_percentage := by sorry
    have h_science_max : science.max_marks = calculate_max_marks science_perf science.passing_percentage := by sorry
    have h_english_max : english.max_marks = calculate_max_marks english_perf english.passing_percentage := by sorry
    
    sorry

end NUMINAMATH_CALUDE_exam_max_marks_calculation_l3403_340373


namespace NUMINAMATH_CALUDE_number_puzzle_l3403_340365

theorem number_puzzle (x y : ℕ) : x = 20 → 3 * (2 * x + y) = 135 → y = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l3403_340365


namespace NUMINAMATH_CALUDE_monomial_degree_6_l3403_340389

def monomial_degree (x_exp y_exp : ℕ) : ℕ := x_exp + y_exp

theorem monomial_degree_6 (a : ℕ) : 
  monomial_degree 2 a = 6 → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_monomial_degree_6_l3403_340389


namespace NUMINAMATH_CALUDE_mean_problem_l3403_340386

theorem mean_problem (x : ℝ) : 
  (12 + x + 42 + 78 + 104) / 5 = 62 → 
  (128 + 255 + 511 + 1023 + x) / 5 = 413 := by
sorry

end NUMINAMATH_CALUDE_mean_problem_l3403_340386


namespace NUMINAMATH_CALUDE_integer_decimal_parts_sqrt10_l3403_340347

theorem integer_decimal_parts_sqrt10 (a b : ℝ) : 
  (a = ⌊6 - Real.sqrt 10⌋) → 
  (b = 6 - Real.sqrt 10 - a) → 
  (2 * a + Real.sqrt 10) * b = 6 := by
sorry

end NUMINAMATH_CALUDE_integer_decimal_parts_sqrt10_l3403_340347


namespace NUMINAMATH_CALUDE_total_serving_time_is_44_minutes_l3403_340383

/-- Represents the properties of a soup pot -/
structure SoupPot where
  gallons : Float
  servingRate : Float  -- bowls per minute
  bowlSize : Float     -- ounces per bowl

/-- Calculates the time to serve a pot of soup -/
def timeToServe (pot : SoupPot) : Float :=
  let ouncesInPot := pot.gallons * 128
  let bowls := (ouncesInPot / pot.bowlSize).floor
  bowls / pot.servingRate

/-- Proves that the total serving time for all soups is 44 minutes when rounded -/
theorem total_serving_time_is_44_minutes (pot1 pot2 pot3 : SoupPot)
  (h1 : pot1 = { gallons := 8, servingRate := 5, bowlSize := 10 })
  (h2 : pot2 = { gallons := 5.5, servingRate := 4, bowlSize := 12 })
  (h3 : pot3 = { gallons := 3.25, servingRate := 6, bowlSize := 8 }) :
  (timeToServe pot1 + timeToServe pot2 + timeToServe pot3).round = 44 := by
  sorry

#eval (timeToServe { gallons := 8, servingRate := 5, bowlSize := 10 } +
       timeToServe { gallons := 5.5, servingRate := 4, bowlSize := 12 } +
       timeToServe { gallons := 3.25, servingRate := 6, bowlSize := 8 }).round

end NUMINAMATH_CALUDE_total_serving_time_is_44_minutes_l3403_340383


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3403_340377

theorem sqrt_equation_solution (y : ℝ) : 
  Real.sqrt (2 + Real.sqrt (3 * y - 4)) = Real.sqrt 7 → y = 29 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3403_340377


namespace NUMINAMATH_CALUDE_rational_function_value_l3403_340385

-- Define f as a function from ℚ to ℚ (rational numbers)
variable (f : ℚ → ℚ)

-- State the main theorem
theorem rational_function_value : 
  (∀ x : ℚ, x ≠ 0 → 4 * f (1 / x) + 3 * f x / x = 2 * x^2) →
  f (-3) = 494 / 117 := by
  sorry

end NUMINAMATH_CALUDE_rational_function_value_l3403_340385


namespace NUMINAMATH_CALUDE_project_hours_difference_l3403_340313

theorem project_hours_difference (total_hours : ℕ) 
  (h_total : total_hours = 135) 
  (pat kate mark : ℕ) 
  (h_pat_kate : pat = 2 * kate) 
  (h_pat_mark : pat * 3 = mark) 
  (h_sum : pat + kate + mark = total_hours) : 
  mark - kate = 75 := by
sorry

end NUMINAMATH_CALUDE_project_hours_difference_l3403_340313


namespace NUMINAMATH_CALUDE_circle_equation_l3403_340325

/-- The equation (x - 3)^2 + (y + 4)^2 = 9 represents a circle centered at (3, -4) with radius 3 -/
theorem circle_equation (x y : ℝ) : 
  (x - 3)^2 + (y + 4)^2 = 9 ↔ 
  ∃ (center : ℝ × ℝ) (radius : ℝ), 
    center = (3, -4) ∧ 
    radius = 3 ∧ 
    (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l3403_340325


namespace NUMINAMATH_CALUDE_negative_three_to_zero_power_l3403_340319

theorem negative_three_to_zero_power : (-3 : ℤ) ^ (0 : ℕ) = 1 := by sorry

end NUMINAMATH_CALUDE_negative_three_to_zero_power_l3403_340319


namespace NUMINAMATH_CALUDE_gcd_squared_plus_one_l3403_340344

theorem gcd_squared_plus_one (n : ℕ+) : 
  (Nat.gcd (n.val^2 + 1) ((n.val + 1)^2 + 1) = 1) ∨ 
  (Nat.gcd (n.val^2 + 1) ((n.val + 1)^2 + 1) = 5) := by
sorry

end NUMINAMATH_CALUDE_gcd_squared_plus_one_l3403_340344


namespace NUMINAMATH_CALUDE_no_error_in_calculation_l3403_340353

theorem no_error_in_calculation : 
  (7 * 4) / (5/3) = (7 * 4) * (3/5) := by
  sorry

#eval (7 * 4) / (5/3) -- To verify the result
#eval (7 * 4) * (3/5) -- To verify the result

end NUMINAMATH_CALUDE_no_error_in_calculation_l3403_340353


namespace NUMINAMATH_CALUDE_pie_cost_l3403_340300

def mary_initial_amount : ℕ := 58
def mary_remaining_amount : ℕ := 52

theorem pie_cost : mary_initial_amount - mary_remaining_amount = 6 := by
  sorry

end NUMINAMATH_CALUDE_pie_cost_l3403_340300


namespace NUMINAMATH_CALUDE_total_age_in_10_years_l3403_340366

def jackson_age : ℕ := 20
def mandy_age : ℕ := jackson_age + 10
def adele_age : ℕ := (3 * jackson_age) / 4

theorem total_age_in_10_years : 
  (jackson_age + 10) + (mandy_age + 10) + (adele_age + 10) = 95 := by
  sorry

end NUMINAMATH_CALUDE_total_age_in_10_years_l3403_340366


namespace NUMINAMATH_CALUDE_monomial_properties_l3403_340337

/-- Represents a monomial of the form ax²y -/
structure Monomial where
  a : ℝ
  x : ℝ
  y : ℝ

/-- Checks if two monomials are of the same type -/
def same_type (m1 m2 : Monomial) : Prop :=
  (m1.x ^ 2 * m1.y = m2.x ^ 2 * m2.y)

/-- Returns the coefficient of a monomial -/
def coefficient (m : Monomial) : ℝ := m.a

/-- Calculates the degree of a monomial -/
def degree (m : Monomial) : ℕ := 3

theorem monomial_properties (m : Monomial) (h : m.a ≠ 0) :
  same_type m { a := -2, x := m.x, y := m.y } ∧
  coefficient m = m.a ∧
  degree m = 3 := by
  sorry

end NUMINAMATH_CALUDE_monomial_properties_l3403_340337


namespace NUMINAMATH_CALUDE_initial_men_is_100_l3403_340380

/-- Represents the road construction project -/
structure RoadProject where
  totalLength : ℝ
  totalDays : ℝ
  completedLength : ℝ
  completedDays : ℝ
  extraMen : ℕ

/-- Calculates the initial number of men employed in the road project -/
def initialMenEmployed (project : RoadProject) : ℕ :=
  sorry

/-- Theorem stating that the initial number of men employed is 100 -/
theorem initial_men_is_100 (project : RoadProject) 
  (h1 : project.totalLength = 15)
  (h2 : project.totalDays = 300)
  (h3 : project.completedLength = 2.5)
  (h4 : project.completedDays = 100)
  (h5 : project.extraMen = 60) :
  initialMenEmployed project = 100 := by
  sorry

#check initial_men_is_100

end NUMINAMATH_CALUDE_initial_men_is_100_l3403_340380


namespace NUMINAMATH_CALUDE_house_glass_panels_l3403_340324

/-- The number of glass panels per window -/
def panels_per_window : ℕ := 4

/-- The number of double windows downstairs -/
def double_windows_downstairs : ℕ := 6

/-- The number of single windows upstairs -/
def single_windows_upstairs : ℕ := 8

/-- The total number of glass panels in the house -/
def total_panels : ℕ := panels_per_window * (2 * double_windows_downstairs + single_windows_upstairs)

theorem house_glass_panels :
  total_panels = 80 :=
by sorry

end NUMINAMATH_CALUDE_house_glass_panels_l3403_340324


namespace NUMINAMATH_CALUDE_walter_exceptional_days_l3403_340307

/-- Represents Walter's chore earnings over a period of days -/
structure ChoreEarnings where
  regularPay : ℕ
  exceptionalPay : ℕ
  bonusThreshold : ℕ
  bonusAmount : ℕ
  totalDays : ℕ
  totalEarnings : ℕ

/-- Calculates the number of exceptional days given ChoreEarnings -/
def exceptionalDays (ce : ChoreEarnings) : ℕ :=
  sorry

/-- Theorem stating that Walter did chores exceptionally well for 5 days -/
theorem walter_exceptional_days (ce : ChoreEarnings) 
  (h1 : ce.regularPay = 4)
  (h2 : ce.exceptionalPay = 6)
  (h3 : ce.bonusThreshold = 5)
  (h4 : ce.bonusAmount = 10)
  (h5 : ce.totalDays = 12)
  (h6 : ce.totalEarnings = 58) :
  exceptionalDays ce = 5 :=
sorry

end NUMINAMATH_CALUDE_walter_exceptional_days_l3403_340307


namespace NUMINAMATH_CALUDE_y_minimum_value_l3403_340376

/-- The function y in terms of x, a, b, and k -/
def y (x a b k : ℝ) : ℝ := 3 * (x - a)^2 + (x - b)^2 + k * x

/-- The derivative of y with respect to x -/
def y_deriv (x a b k : ℝ) : ℝ := 8 * x - 6 * a - 2 * b + k

/-- The second derivative of y with respect to x -/
def y_second_deriv : ℝ := 8

theorem y_minimum_value (a b k : ℝ) :
  ∃ x : ℝ, y_deriv x a b k = 0 ∧
           y_second_deriv > 0 ∧
           x = (6 * a + 2 * b - k) / 8 :=
sorry

end NUMINAMATH_CALUDE_y_minimum_value_l3403_340376


namespace NUMINAMATH_CALUDE_least_number_for_divisibility_l3403_340322

theorem least_number_for_divisibility (n m : ℕ) (h : n = 1056 ∧ m = 26) :
  ∃ x : ℕ, (x = 10 ∧ (n + x) % m = 0 ∧ ∀ y : ℕ, y < x → (n + y) % m ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_least_number_for_divisibility_l3403_340322


namespace NUMINAMATH_CALUDE_set_operations_and_subset_condition_l3403_340355

-- Define the sets A and B
def A : Set ℝ := {x | x < -4 ∨ x > 1}
def B : Set ℝ := {x | -3 ≤ x - 1 ∧ x - 1 ≤ 2}

-- Define the set M parameterized by k
def M (k : ℝ) : Set ℝ := {x | 2*k - 1 ≤ x ∧ x ≤ 2*k + 1}

-- Theorem statement
theorem set_operations_and_subset_condition :
  (A ∩ B = {x | 1 < x ∧ x ≤ 3}) ∧
  ((Aᶜ ∪ Bᶜ) = {x | x ≤ 1 ∨ x > 3}) ∧
  (∀ k, M k ⊆ A ↔ k < -5/2 ∨ k > 1) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_and_subset_condition_l3403_340355


namespace NUMINAMATH_CALUDE_coffee_stock_problem_l3403_340392

/-- Represents the coffee stock problem --/
theorem coffee_stock_problem 
  (initial_stock : ℝ) 
  (additional_purchase : ℝ) 
  (decaf_percent_additional : ℝ) 
  (total_decaf_percent : ℝ) : 
  initial_stock = 400 ∧ 
  additional_purchase = 100 ∧ 
  decaf_percent_additional = 60 ∧ 
  total_decaf_percent = 32 → 
  (initial_stock * (25 / 100) + additional_purchase * (decaf_percent_additional / 100)) / 
  (initial_stock + additional_purchase) = total_decaf_percent / 100 := by
  sorry

#check coffee_stock_problem

end NUMINAMATH_CALUDE_coffee_stock_problem_l3403_340392


namespace NUMINAMATH_CALUDE_sector_area_l3403_340357

/-- The area of a circular sector with radius 2 cm and central angle 120° is 4π/3 cm² -/
theorem sector_area (r : ℝ) (θ_deg : ℝ) (A : ℝ) : 
  r = 2 → θ_deg = 120 → A = (1/2) * r^2 * (θ_deg * π / 180) → A = (4/3) * π := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l3403_340357


namespace NUMINAMATH_CALUDE_roses_to_mother_l3403_340330

def roses_problem (total_roses grandmother_roses sister_roses kept_roses : ℕ) : ℕ :=
  total_roses - (grandmother_roses + sister_roses + kept_roses)

theorem roses_to_mother :
  roses_problem 20 9 4 1 = 6 := by
  sorry

end NUMINAMATH_CALUDE_roses_to_mother_l3403_340330


namespace NUMINAMATH_CALUDE_ab_equals_op_l3403_340335

noncomputable section

/-- Line l with parametric equations x = -1/2 * t, y = a + (√3/2) * t -/
def line_l (a t : ℝ) : ℝ × ℝ := (-1/2 * t, a + (Real.sqrt 3 / 2) * t)

/-- Curve C with rectangular equation x² + y² - 4x = 0 -/
def curve_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x = 0

/-- Length of AB, where A and B are intersection points of line l and curve C -/
def length_AB (a : ℝ) : ℝ := Real.sqrt (4 + 4 * Real.sqrt 3 * a - a^2)

/-- Theorem stating that |AB| = 2 if and only if a = 0 or a = 4√3 -/
theorem ab_equals_op (a : ℝ) : length_AB a = 2 ↔ a = 0 ∨ a = 4 * Real.sqrt 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ab_equals_op_l3403_340335


namespace NUMINAMATH_CALUDE_inequality_proof_l3403_340329

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z ≥ 1) : 
  (x * Real.sqrt x) / (y + z) + (y * Real.sqrt y) / (z + x) + (z * Real.sqrt z) / (x + y) ≥ Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3403_340329


namespace NUMINAMATH_CALUDE_uniform_scores_smaller_variance_l3403_340352

/-- Class scores data -/
structure ClassScores where
  mean : ℝ
  variance : ℝ

/-- Uniformity of scores -/
def more_uniform (a b : ClassScores) : Prop :=
  a.variance < b.variance

/-- Theorem: Class with smaller variance has more uniform scores -/
theorem uniform_scores_smaller_variance 
  (class_a class_b : ClassScores) 
  (h_mean : class_a.mean = class_b.mean) 
  (h_var : class_a.variance > class_b.variance) : 
  more_uniform class_b class_a :=
by sorry

end NUMINAMATH_CALUDE_uniform_scores_smaller_variance_l3403_340352


namespace NUMINAMATH_CALUDE_carpet_coverage_percentage_l3403_340317

/-- The percentage of a living room floor covered by a rectangular carpet -/
theorem carpet_coverage_percentage 
  (carpet_length : ℝ) 
  (carpet_width : ℝ) 
  (room_area : ℝ) 
  (h1 : carpet_length = 4) 
  (h2 : carpet_width = 9) 
  (h3 : room_area = 120) : 
  (carpet_length * carpet_width) / room_area * 100 = 30 := by
sorry

end NUMINAMATH_CALUDE_carpet_coverage_percentage_l3403_340317


namespace NUMINAMATH_CALUDE_money_problem_l3403_340327

theorem money_problem (a b : ℚ) : 
  (4 * a + 2 * b = 92) ∧ (6 * a - 4 * b = 60) → 
  (a = 122 / 7) ∧ (b = 78 / 7) := by
  sorry

end NUMINAMATH_CALUDE_money_problem_l3403_340327


namespace NUMINAMATH_CALUDE_exponential_decreasing_range_l3403_340326

/-- A function f: ℝ → ℝ is strictly decreasing -/
def StrictlyDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

theorem exponential_decreasing_range (a : ℝ) :
  StrictlyDecreasing (fun x ↦ (a - 1) ^ x) → 1 < a ∧ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_exponential_decreasing_range_l3403_340326


namespace NUMINAMATH_CALUDE_polyhedron_with_specific_projections_l3403_340348

/-- A polyhedron is a three-dimensional geometric object with flat polygonal faces, straight edges and sharp corners or vertices. -/
def Polyhedron : Type := sorry

/-- A plane in three-dimensional space. -/
def Plane : Type := sorry

/-- A projection of a polyhedron onto a plane. -/
def projection (p : Polyhedron) (plane : Plane) : Set (ℝ × ℝ) := sorry

/-- A triangle is a polygon with three sides. -/
def isTriangle (s : Set (ℝ × ℝ)) : Prop := sorry

/-- A quadrilateral is a polygon with four sides. -/
def isQuadrilateral (s : Set (ℝ × ℝ)) : Prop := sorry

/-- A pentagon is a polygon with five sides. -/
def isPentagon (s : Set (ℝ × ℝ)) : Prop := sorry

/-- Two planes are perpendicular if they intersect at a right angle. -/
def arePerpendicular (p1 p2 : Plane) : Prop := sorry

theorem polyhedron_with_specific_projections :
  ∃ (p : Polyhedron) (p1 p2 p3 : Plane),
    arePerpendicular p1 p2 ∧
    arePerpendicular p2 p3 ∧
    arePerpendicular p3 p1 ∧
    isTriangle (projection p p1) ∧
    isQuadrilateral (projection p p2) ∧
    isPentagon (projection p p3) := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_with_specific_projections_l3403_340348


namespace NUMINAMATH_CALUDE_quadratic_root_equivalence_l3403_340349

theorem quadratic_root_equivalence (a b c : ℝ) (ha : a ≠ 0) :
  (a + b + c = 0) ↔ (a * 1^2 + b * 1 + c = 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_equivalence_l3403_340349


namespace NUMINAMATH_CALUDE_parallelogram_area_l3403_340343

structure Vector2D where
  x : ℝ
  y : ℝ

def angle (v w : Vector2D) : ℝ := sorry

def norm (v : Vector2D) : ℝ := sorry

def cross (v w : Vector2D) : ℝ := sorry

theorem parallelogram_area (p q : Vector2D) : 
  let a := Vector2D.mk (6 * p.x - q.x) (6 * p.y - q.y)
  let b := Vector2D.mk (5 * q.x + p.x) (5 * q.y + p.y)
  norm p = 1/2 →
  norm q = 4 →
  angle p q = 5 * π / 6 →
  abs (cross a b) = 31 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l3403_340343


namespace NUMINAMATH_CALUDE_subtracted_value_l3403_340378

theorem subtracted_value (x y : ℤ) (h1 : x = 120) (h2 : 2 * x - y = 102) : y = 138 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_value_l3403_340378


namespace NUMINAMATH_CALUDE_characterize_special_function_l3403_340368

/-- A strictly increasing function from ℕ to ℕ satisfying nf(f(n)) = f(n)² for all positive integers n -/
def StrictlyIncreasingSpecialFunction (f : ℕ → ℕ) : Prop :=
  (∀ m n, m < n → f m < f n) ∧ 
  (∀ n : ℕ, n > 0 → n * f (f n) = (f n) ^ 2)

/-- The characterization of all functions satisfying the given conditions -/
theorem characterize_special_function (f : ℕ → ℕ) :
  StrictlyIncreasingSpecialFunction f →
  (∀ x, f x = x) ∨
  (∃ c d : ℕ, c > 1 ∧ 
    (∀ x, x < d → f x = x) ∧
    (∀ x, x ≥ d → f x = c * x)) :=
by sorry

end NUMINAMATH_CALUDE_characterize_special_function_l3403_340368


namespace NUMINAMATH_CALUDE_selection_theorem_l3403_340394

/-- The number of ways to select k items from n items -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The number of books on the shelf -/
def total_books : ℕ := 10

/-- The number of books to be selected -/
def books_to_select : ℕ := 5

/-- The number of specific books that must be included -/
def specific_books : ℕ := 2

/-- The number of ways to select 5 books from 10 books, given that 2 specific books must always be included -/
def selection_ways : ℕ := binomial (total_books - specific_books) (books_to_select - specific_books)

theorem selection_theorem : selection_ways = 56 := by sorry

end NUMINAMATH_CALUDE_selection_theorem_l3403_340394


namespace NUMINAMATH_CALUDE_star_equation_solution_l3403_340338

-- Define the ☆ operation
def star (a b : ℝ) : ℝ := a + b - 1

-- Theorem statement
theorem star_equation_solution :
  ∃ x : ℝ, star 2 x = 1 ∧ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_star_equation_solution_l3403_340338


namespace NUMINAMATH_CALUDE_gumballs_remaining_is_sixty_l3403_340341

/-- The number of gumballs remaining in the bowl after Pedro takes out 40% -/
def remaining_gumballs (alicia_gumballs : ℕ) : ℕ :=
  let pedro_gumballs := alicia_gumballs + 3 * alicia_gumballs
  let total_gumballs := alicia_gumballs + pedro_gumballs
  let taken_out := (40 * total_gumballs) / 100
  total_gumballs - taken_out

/-- Theorem stating that given Alicia has 20 gumballs, the number of gumballs
    remaining in the bowl after Pedro takes out 40% is 60 -/
theorem gumballs_remaining_is_sixty :
  remaining_gumballs 20 = 60 := by
  sorry

end NUMINAMATH_CALUDE_gumballs_remaining_is_sixty_l3403_340341


namespace NUMINAMATH_CALUDE_rice_distribution_l3403_340381

theorem rice_distribution (total_weight : ℚ) (num_containers : ℕ) (pound_to_ounce : ℕ) : 
  total_weight = 35 / 2 →
  num_containers = 4 →
  pound_to_ounce = 16 →
  (total_weight * pound_to_ounce) / num_containers = 70 := by
  sorry

end NUMINAMATH_CALUDE_rice_distribution_l3403_340381


namespace NUMINAMATH_CALUDE_cards_in_unfilled_box_l3403_340351

theorem cards_in_unfilled_box (total_cards : Nat) (cards_per_box : Nat) (h1 : total_cards = 94) (h2 : cards_per_box = 8) :
  total_cards % cards_per_box = 6 := by
sorry

end NUMINAMATH_CALUDE_cards_in_unfilled_box_l3403_340351


namespace NUMINAMATH_CALUDE_roots_sum_fraction_eq_neg_two_l3403_340336

theorem roots_sum_fraction_eq_neg_two (z₁ z₂ : ℂ) 
  (h₁ : z₁^2 + z₁ + 1 = 0) 
  (h₂ : z₂^2 + z₂ + 1 = 0) 
  (h₃ : z₁ ≠ z₂) : 
  z₂ / (z₁ + 1) + z₁ / (z₂ + 1) = -2 := by sorry

end NUMINAMATH_CALUDE_roots_sum_fraction_eq_neg_two_l3403_340336


namespace NUMINAMATH_CALUDE_hugo_first_roll_7_given_win_l3403_340354

-- Define the number of players
def num_players : ℕ := 5

-- Define the number of sides on the first die
def first_die_sides : ℕ := 8

-- Define the number of sides on the subsequent die
def subsequent_die_sides : ℕ := 10

-- Define Hugo's winning probability
def hugo_win_prob : ℚ := 1 / num_players

-- Define the probability of rolling a 7 on the first die
def prob_roll_7 : ℚ := 1 / first_die_sides

-- Define the event that Hugo wins given his first roll was 7
def hugo_win_given_7 : ℚ := 961 / 2560

-- Theorem to prove
theorem hugo_first_roll_7_given_win (num_players : ℕ) (first_die_sides : ℕ) 
  (subsequent_die_sides : ℕ) (hugo_win_prob : ℚ) (prob_roll_7 : ℚ) 
  (hugo_win_given_7 : ℚ) :
  num_players = 5 → 
  first_die_sides = 8 → 
  subsequent_die_sides = 10 → 
  hugo_win_prob = 1 / 5 → 
  prob_roll_7 = 1 / 8 → 
  hugo_win_given_7 = 961 / 2560 → 
  (prob_roll_7 * hugo_win_given_7) / hugo_win_prob = 961 / 2048 := by
  sorry


end NUMINAMATH_CALUDE_hugo_first_roll_7_given_win_l3403_340354


namespace NUMINAMATH_CALUDE_absolute_value_sum_l3403_340384

theorem absolute_value_sum (m n p : ℤ) 
  (h : |m - n|^3 + |p - m|^5 = 1) : 
  |p - m| + |m - n| + 2 * |n - p| = 3 := by sorry

end NUMINAMATH_CALUDE_absolute_value_sum_l3403_340384


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l3403_340328

theorem cyclic_sum_inequality (x y z : ℝ) (hpos_x : x > 0) (hpos_y : y > 0) (hpos_z : z > 0)
  (h_sum_squares : x^2 + y^2 + z^2 = 1) :
  (x*y/z) + (y*z/x) + (z*x/y) ≥ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l3403_340328


namespace NUMINAMATH_CALUDE_not_divisible_by_101_l3403_340370

theorem not_divisible_by_101 (k : ℤ) : ¬(101 ∣ k^2 + k + 1) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_101_l3403_340370


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l3403_340339

theorem inscribed_squares_ratio (r : ℝ) (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (2 * a)^2 + (2 * b)^2 = r^2 ∧ 
  (a + 2*b)^2 + b^2 = r^2 → 
  a / b = 5 := by
sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l3403_340339


namespace NUMINAMATH_CALUDE_circle_from_diameter_l3403_340306

-- Define the points A and B
def A : ℝ × ℝ := (2, 0)
def B : ℝ × ℝ := (0, 4)

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 2)^2 = 5

-- Theorem statement
theorem circle_from_diameter :
  ∀ (x y : ℝ),
  circle_equation x y ↔
  ∃ (t : ℝ), 
    0 ≤ t ∧ t ≤ 1 ∧
    x = A.1 * (1 - t) + B.1 * t ∧
    y = A.2 * (1 - t) + B.2 * t :=
by sorry


end NUMINAMATH_CALUDE_circle_from_diameter_l3403_340306


namespace NUMINAMATH_CALUDE_remaining_marbles_l3403_340332

def initial_marbles : ℕ := 64
def marbles_given : ℕ := 14

theorem remaining_marbles :
  initial_marbles - marbles_given = 50 :=
by sorry

end NUMINAMATH_CALUDE_remaining_marbles_l3403_340332


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l3403_340301

theorem sqrt_product_equality : Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l3403_340301


namespace NUMINAMATH_CALUDE_reading_ratio_l3403_340320

/-- Given a book with a certain number of pages, prove the ratio of pages read on two consecutive days --/
theorem reading_ratio (total_pages : ℕ) (pages_day1 : ℕ) (pages_left : ℕ) : 
  total_pages = 360 →
  pages_day1 = 50 →
  pages_left = 210 →
  (total_pages - pages_left - pages_day1) / pages_day1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_reading_ratio_l3403_340320


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3403_340345

theorem inequality_solution_set 
  (a b c : ℝ) 
  (h1 : ∀ x, ax^2 + b*x + c > 0 ↔ -1 < x ∧ x < 2) 
  (h2 : a < 0) : 
  ∀ x, a*(x^2 + 1) + b*(x - 1) + c > 2*a*x ↔ 0 < x ∧ x < 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3403_340345


namespace NUMINAMATH_CALUDE_sum_of_squares_of_cubic_roots_l3403_340310

/-- Given a cubic equation dx³ - ex² + fx - g = 0 with real coefficients,
    the sum of squares of its roots is (e² - 2df) / d². -/
theorem sum_of_squares_of_cubic_roots
  (d e f g : ℝ) (a b c : ℝ)
  (hroots : d * (X - a) * (X - b) * (X - c) = d * X^3 - e * X^2 + f * X - g) :
  a^2 + b^2 + c^2 = (e^2 - 2*d*f) / d^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_cubic_roots_l3403_340310


namespace NUMINAMATH_CALUDE_linear_function_not_in_third_quadrant_l3403_340308

-- Define the quadratic equation
def quadratic_eq (x : ℝ) : Prop := x^2 - 2*x - 3 = 0

-- Define the roots of the quadratic equation
def roots (a b : ℝ) : Prop := quadratic_eq a ∧ quadratic_eq b ∧ a ≠ b

-- Define the linear function
def linear_function (x : ℝ) (a b : ℝ) : ℝ := (a*b - 1)*x + a + b

-- Theorem: The linear function does not pass through the third quadrant
theorem linear_function_not_in_third_quadrant (a b : ℝ) (h : roots a b) :
  ∀ x y : ℝ, y = linear_function x a b → ¬(x < 0 ∧ y < 0) :=
sorry

end NUMINAMATH_CALUDE_linear_function_not_in_third_quadrant_l3403_340308


namespace NUMINAMATH_CALUDE_largest_five_digit_congruent_16_mod_25_l3403_340360

theorem largest_five_digit_congruent_16_mod_25 : ∃ n : ℕ,
  n = 99991 ∧
  10000 ≤ n ∧ n < 100000 ∧
  n % 25 = 16 ∧
  ∀ m : ℕ, 10000 ≤ m ∧ m < 100000 ∧ m % 25 = 16 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_five_digit_congruent_16_mod_25_l3403_340360


namespace NUMINAMATH_CALUDE_roses_in_vase_l3403_340391

theorem roses_in_vase (total_flowers : ℕ) (carnations : ℕ) (roses : ℕ) : 
  total_flowers = 10 → carnations = 5 → total_flowers = roses + carnations → roses = 5 := by
  sorry

end NUMINAMATH_CALUDE_roses_in_vase_l3403_340391


namespace NUMINAMATH_CALUDE_simplify_expression_l3403_340369

theorem simplify_expression (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 2) :
  (x - 3*x/(x+1)) / ((x-2)/(x^2 + 2*x + 1)) = x^2 + x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3403_340369


namespace NUMINAMATH_CALUDE_watch_profit_percentage_l3403_340382

/-- Calculate the percentage of profit given the cost price and selling price -/
def percentage_profit (cost_price selling_price : ℚ) : ℚ :=
  ((selling_price - cost_price) / cost_price) * 100

/-- Theorem: The percentage of profit for a watch with cost price 90 and selling price 144 is 60% -/
theorem watch_profit_percentage :
  percentage_profit 90 144 = 60 := by
  sorry

end NUMINAMATH_CALUDE_watch_profit_percentage_l3403_340382


namespace NUMINAMATH_CALUDE_rotation_result_l3403_340311

-- Define a type for the shapes
inductive Shape
  | Square
  | Pentagon
  | Ellipse

-- Define a type for the positions
inductive Position
  | X
  | Y
  | Z

-- Define a function to represent the initial configuration
def initial_config : Shape → Position
  | Shape.Square => Position.X
  | Shape.Pentagon => Position.Y
  | Shape.Ellipse => Position.Z

-- Define a function to represent the rotation
def rotate_180 (p : Position) : Position :=
  match p with
  | Position.X => Position.Y
  | Position.Y => Position.X
  | Position.Z => Position.Z

-- Theorem statement
theorem rotation_result :
  ∀ (s : Shape),
    rotate_180 (initial_config s) =
      match s with
      | Shape.Square => Position.Y
      | Shape.Pentagon => Position.X
      | Shape.Ellipse => Position.Z
  := by sorry

end NUMINAMATH_CALUDE_rotation_result_l3403_340311


namespace NUMINAMATH_CALUDE_probability_nine_matches_zero_l3403_340302

/-- A matching problem with n pairs -/
structure MatchingProblem (n : ℕ) where
  /-- The number of pairs to match -/
  pairs : ℕ
  /-- Assertion that the number of pairs is n -/
  pairs_eq : pairs = n

/-- The probability of correctly matching exactly k pairs in a matching problem with n pairs by random selection -/
noncomputable def probability_exact_matches (n k : ℕ) (problem : MatchingProblem n) : ℝ :=
  sorry

/-- Theorem: In a matching problem with 10 pairs, the probability of correctly matching exactly 9 pairs by random selection is 0 -/
theorem probability_nine_matches_zero :
  ∀ (problem : MatchingProblem 10), probability_exact_matches 10 9 problem = 0 :=
sorry

end NUMINAMATH_CALUDE_probability_nine_matches_zero_l3403_340302


namespace NUMINAMATH_CALUDE_marys_income_percentage_l3403_340331

theorem marys_income_percentage (juan tim mary : ℝ) 
  (h1 : tim = juan * 0.9) 
  (h2 : mary = tim * 1.6) : 
  mary = juan * 1.44 := by
sorry

end NUMINAMATH_CALUDE_marys_income_percentage_l3403_340331


namespace NUMINAMATH_CALUDE_salary_change_l3403_340364

theorem salary_change (initial_salary : ℝ) (h : initial_salary > 0) :
  let after_first_decrease := initial_salary * 0.5
  let after_increase := after_first_decrease * 1.3
  let final_salary := after_increase * 0.8
  (final_salary - initial_salary) / initial_salary = -0.48 :=
by sorry

end NUMINAMATH_CALUDE_salary_change_l3403_340364


namespace NUMINAMATH_CALUDE_smallest_a_equals_36_l3403_340350

/-- A function satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x > 0, f (2 * x) = 2 * f x) ∧
  (∀ x, 1 < x ∧ x ≤ 2 → f x = 2 - x)

/-- The theorem statement -/
theorem smallest_a_equals_36 (f : ℝ → ℝ) (hf : special_function f) :
  (∃ a : ℝ, a > 0 ∧ f a = f 2020 ∧ ∀ b, b > 0 ∧ f b = f 2020 → a ≤ b) →
  (∃ a : ℝ, a > 0 ∧ f a = f 2020 ∧ ∀ b, b > 0 ∧ f b = f 2020 → a ≤ b) ∧ a = 36 :=
sorry

end NUMINAMATH_CALUDE_smallest_a_equals_36_l3403_340350


namespace NUMINAMATH_CALUDE_union_and_intersection_when_m_2_intersection_empty_iff_l3403_340398

def A : Set ℝ := {x | -3 < x ∧ x < 4}
def B (m : ℝ) : Set ℝ := {x | m - 1 < x ∧ x < 3 * m + 3}

theorem union_and_intersection_when_m_2 :
  (A ∪ B 2 = {x | -3 < x ∧ x < 9}) ∧
  (A ∩ (Set.univ \ B 2) = {x | -3 < x ∧ x ≤ 1}) := by sorry

theorem intersection_empty_iff :
  ∀ m : ℝ, A ∩ B m = ∅ ↔ m ≥ 5 ∨ m ≤ -2 := by sorry

end NUMINAMATH_CALUDE_union_and_intersection_when_m_2_intersection_empty_iff_l3403_340398


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l3403_340342

-- Define the function f
def f (t : ℝ) (x : ℝ) : ℝ := |x - 1| + |x - t|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 2 x > 2} = {x : ℝ | x < 1/2 ∨ x > 5/2} :=
by sorry

-- Part 2
theorem range_of_a_part2 :
  ∀ t ∈ Set.Icc 1 2,
    (∀ x ∈ Set.Icc (-1) 3, ∃ a : ℝ, f t x ≥ a + x) →
    ∃ a : ℝ, a ≤ -1 ∧ ∀ x ∈ Set.Icc (-1) 3, f t x ≥ a + x :=
by sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l3403_340342


namespace NUMINAMATH_CALUDE_first_shift_members_l3403_340399

-- Define the number of shifts
def num_shifts : ℕ := 3

-- Define the total number of workers in the company
def total_workers (shift1 : ℕ) : ℕ := shift1 + 50 + 40

-- Define the participation rate for each shift
def participation_rate1 : ℚ := 1/5
def participation_rate2 : ℚ := 2/5
def participation_rate3 : ℚ := 1/10

-- Define the total number of participants in the pension program
def total_participants (shift1 : ℕ) : ℚ :=
  participation_rate1 * shift1 + participation_rate2 * 50 + participation_rate3 * 40

-- State the theorem
theorem first_shift_members :
  ∃ (shift1 : ℕ), 
    shift1 > 0 ∧
    (total_participants shift1) / (total_workers shift1) = 6/25 ∧
    shift1 = 60 :=
by sorry

end NUMINAMATH_CALUDE_first_shift_members_l3403_340399


namespace NUMINAMATH_CALUDE_ninety_mile_fare_l3403_340356

/-- Represents the fare structure for a taxi ride -/
structure TaxiFare where
  baseFare : ℝ
  ratePerMile : ℝ

/-- Calculates the total fare for a given distance -/
def totalFare (tf : TaxiFare) (distance : ℝ) : ℝ :=
  tf.baseFare + tf.ratePerMile * distance

theorem ninety_mile_fare :
  ∃ (tf : TaxiFare),
    tf.baseFare = 30 ∧
    totalFare tf 60 = 150 ∧
    totalFare tf 90 = 210 := by
  sorry

end NUMINAMATH_CALUDE_ninety_mile_fare_l3403_340356


namespace NUMINAMATH_CALUDE_union_when_m_neg_one_subset_condition_l3403_340395

-- Define sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 3}
def B (m : ℝ) : Set ℝ := {x | 2*m < x ∧ x < 1-m}

-- Theorem for part 1
theorem union_when_m_neg_one :
  A ∪ B (-1) = {x : ℝ | -2 < x ∧ x < 3} := by sorry

-- Theorem for part 2
theorem subset_condition (m : ℝ) :
  A ⊆ B m ↔ m ≤ -2 := by sorry

end NUMINAMATH_CALUDE_union_when_m_neg_one_subset_condition_l3403_340395


namespace NUMINAMATH_CALUDE_email_sample_not_representative_l3403_340312

/-- Represents the urban population -/
def UrbanPopulation : Type := Unit

/-- Represents a person in the urban population -/
def Person : Type := Unit

/-- Predicate for whether a person owns an email address -/
def has_email_address (p : Person) : Prop := sorry

/-- Predicate for whether a person uses the internet -/
def uses_internet (p : Person) : Prop := sorry

/-- Predicate for whether a person gets news from the internet -/
def gets_news_from_internet (p : Person) : Prop := sorry

/-- The sample of email address owners -/
def email_sample (n : ℕ) : Set Person := sorry

/-- A sample is representative if it accurately reflects the population characteristics -/
def is_representative (s : Set Person) : Prop := sorry

/-- Theorem stating that the email sample is not representative -/
theorem email_sample_not_representative (n : ℕ) : 
  ¬(is_representative (email_sample n)) := by sorry

end NUMINAMATH_CALUDE_email_sample_not_representative_l3403_340312


namespace NUMINAMATH_CALUDE_sum_of_composite_functions_l3403_340363

def p (x : ℝ) : ℝ := |x + 1| - 3

def q (x : ℝ) : ℝ := -|x|

def x_values : List ℝ := [-4, -3, -2, -1, 0, 1, 2, 3, 4]

theorem sum_of_composite_functions :
  (x_values.map (λ x => q (p x))).sum = -12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_composite_functions_l3403_340363


namespace NUMINAMATH_CALUDE_shirt_cost_l3403_340323

/-- Given the cost of jeans and shirts in two scenarios, prove the cost of one shirt. -/
theorem shirt_cost (j s : ℚ) 
  (scenario1 : 3 * j + 2 * s = 69)
  (scenario2 : 2 * j + 3 * s = 66) :
  s = 12 := by
  sorry

end NUMINAMATH_CALUDE_shirt_cost_l3403_340323


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l3403_340390

/-- The y-intercept of the line 2x - 3y = 6 is -2 -/
theorem y_intercept_of_line (x y : ℝ) : 2 * x - 3 * y = 6 → x = 0 → y = -2 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l3403_340390


namespace NUMINAMATH_CALUDE_cos_4alpha_minus_9pi_over_2_l3403_340309

theorem cos_4alpha_minus_9pi_over_2 (α : ℝ) : 
  4.53 * (1 + Real.cos (2 * α - 2 * Real.pi) + Real.cos (4 * α + 2 * Real.pi) - Real.cos (6 * α - Real.pi)) / 
  (Real.cos (2 * Real.pi - 2 * α) + 2 * (Real.cos (2 * α + Real.pi))^2 - 1) = 2 * Real.cos (2 * α) →
  Real.cos (4 * α - 9 * Real.pi / 2) = Real.cos (4 * α - Real.pi / 2) := by
sorry

end NUMINAMATH_CALUDE_cos_4alpha_minus_9pi_over_2_l3403_340309


namespace NUMINAMATH_CALUDE_complex_product_magnitude_l3403_340379

theorem complex_product_magnitude (a b : ℂ) (t : ℝ) :
  Complex.abs a = 3 →
  Complex.abs b = 5 →
  a * b = t - 3 * Complex.I →
  t = 6 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_complex_product_magnitude_l3403_340379


namespace NUMINAMATH_CALUDE_total_nuts_eq_3200_l3403_340387

/-- The number of busy squirrels -/
def busy_squirrels : ℕ := 2

/-- The number of nuts stockpiled by each busy squirrel per day -/
def busy_squirrel_nuts_per_day : ℕ := 30

/-- The number of sleepy squirrels -/
def sleepy_squirrels : ℕ := 1

/-- The number of nuts stockpiled by each sleepy squirrel per day -/
def sleepy_squirrel_nuts_per_day : ℕ := 20

/-- The number of days the squirrels have been stockpiling -/
def stockpiling_days : ℕ := 40

/-- The total number of nuts stockpiled by all squirrels -/
def total_nuts : ℕ := 
  (busy_squirrels * busy_squirrel_nuts_per_day + 
   sleepy_squirrels * sleepy_squirrel_nuts_per_day) * 
  stockpiling_days

theorem total_nuts_eq_3200 : total_nuts = 3200 :=
by sorry

end NUMINAMATH_CALUDE_total_nuts_eq_3200_l3403_340387


namespace NUMINAMATH_CALUDE_x_minus_y_values_l3403_340396

theorem x_minus_y_values (x y : ℝ) (hx : |x| = 5) (hy : |y| = 3) (hxy : y > x) :
  x - y = -8 ∨ x - y = -2 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_values_l3403_340396


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l3403_340305

theorem rectangle_perimeter (l w : ℕ+) : 
  l * w = 24 → 2 * (l + w) ≠ 36 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l3403_340305


namespace NUMINAMATH_CALUDE_path_construction_cost_l3403_340359

/-- Given a rectangular grass field with surrounding path, calculate the total cost of constructing the path -/
theorem path_construction_cost 
  (field_length : ℝ) 
  (field_width : ℝ) 
  (long_side_path_width : ℝ) 
  (short_side1_path_width : ℝ) 
  (short_side2_path_width : ℝ) 
  (long_side_cost_per_sqm : ℝ) 
  (short_side1_cost_per_sqm : ℝ) 
  (short_side2_cost_per_sqm : ℝ) 
  (h1 : field_length = 75) 
  (h2 : field_width = 55) 
  (h3 : long_side_path_width = 2.5) 
  (h4 : short_side1_path_width = 3) 
  (h5 : short_side2_path_width = 4) 
  (h6 : long_side_cost_per_sqm = 7) 
  (h7 : short_side1_cost_per_sqm = 9) 
  (h8 : short_side2_cost_per_sqm = 12) :
  let long_sides_area := 2 * field_length * long_side_path_width
  let short_side1_area := field_width * short_side1_path_width
  let short_side2_area := field_width * short_side2_path_width
  let long_sides_cost := long_sides_area * long_side_cost_per_sqm
  let short_side1_cost := short_side1_area * short_side1_cost_per_sqm
  let short_side2_cost := short_side2_area * short_side2_cost_per_sqm
  let total_cost := long_sides_cost + short_side1_cost + short_side2_cost
  total_cost = 6750 := by sorry


end NUMINAMATH_CALUDE_path_construction_cost_l3403_340359


namespace NUMINAMATH_CALUDE_probability_of_target_urn_l3403_340367

/-- Represents the contents of the urn -/
structure UrnContents where
  red : ℕ
  blue : ℕ

/-- Represents one operation of drawing and adding a ball -/
inductive Operation
  | DrawRed
  | DrawBlue

/-- The probability of a specific sequence of operations resulting in 4 red and 3 blue balls -/
def probability_of_sequence (seq : List Operation) : ℚ :=
  sorry

/-- The number of possible sequences resulting in 4 red and 3 blue balls -/
def number_of_favorable_sequences : ℕ :=
  sorry

/-- The initial contents of the urn -/
def initial_urn : UrnContents :=
  { red := 2, blue := 1 }

/-- The final contents of the urn we're interested in -/
def target_urn : UrnContents :=
  { red := 4, blue := 3 }

/-- The number of operations performed -/
def num_operations : ℕ := 5

/-- The total number of balls in the urn after all operations -/
def final_total_balls : ℕ := 10

theorem probability_of_target_urn :
  probability_of_sequence (List.replicate num_operations Operation.DrawRed) *
    number_of_favorable_sequences = 4 / 7 :=
  sorry

end NUMINAMATH_CALUDE_probability_of_target_urn_l3403_340367


namespace NUMINAMATH_CALUDE_total_leaves_l3403_340340

theorem total_leaves (initial_leaves additional_leaves : ℝ) :
  initial_leaves + additional_leaves = initial_leaves + additional_leaves :=
by sorry

end NUMINAMATH_CALUDE_total_leaves_l3403_340340


namespace NUMINAMATH_CALUDE_extremum_implies_f_2_l3403_340372

/-- A function f(x) with an extremum at x = 1 and f(1) = 10 -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x

/-- The derivative of f(x) -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extremum_implies_f_2 (a b : ℝ) :
  f' a b 1 = 0 → f a b 1 = 10 → f a b 2 = 2 := by
  sorry

#check extremum_implies_f_2

end NUMINAMATH_CALUDE_extremum_implies_f_2_l3403_340372


namespace NUMINAMATH_CALUDE_plant_equation_correct_l3403_340321

/-- Represents the structure of a plant with branches and small branches. -/
structure Plant where
  branches : ℕ
  smallBranches : ℕ

/-- The total number of parts in a plant, including the main stem. -/
def totalParts (p : Plant) : ℕ := 1 + p.branches + p.smallBranches

/-- Constructs a plant where each branch grows a specific number of small branches. -/
def makePlant (x : ℕ) : Plant :=
  { branches := x, smallBranches := x * x }

/-- Theorem stating that for some natural number x, the plant structure
    results in a total of 91 parts. -/
theorem plant_equation_correct :
  ∃ x : ℕ, totalParts (makePlant x) = 91 := by
  sorry

end NUMINAMATH_CALUDE_plant_equation_correct_l3403_340321


namespace NUMINAMATH_CALUDE_sequence_a_property_l3403_340304

def sequence_a : ℕ → ℚ
  | 0 => 1/2
  | (n+1) => sequence_a n + (sequence_a n)^2 / 2023

theorem sequence_a_property : sequence_a 2023 < 1 ∧ 1 < sequence_a 2024 := by
  sorry

end NUMINAMATH_CALUDE_sequence_a_property_l3403_340304


namespace NUMINAMATH_CALUDE_older_brother_running_distance_l3403_340315

/-- The running speed of the older brother in meters per minute -/
def older_brother_speed : ℝ := 110

/-- The running speed of the younger brother in meters per minute -/
def younger_brother_speed : ℝ := 80

/-- The additional time the younger brother runs in minutes -/
def additional_time : ℝ := 30

/-- The additional distance the younger brother runs in meters -/
def additional_distance : ℝ := 900

/-- The distance run by the older brother in meters -/
def older_brother_distance : ℝ := 5500

theorem older_brother_running_distance :
  ∃ (t : ℝ), 
    t > 0 ∧
    (t + additional_time) * younger_brother_speed = t * older_brother_speed + additional_distance ∧
    t * older_brother_speed = older_brother_distance :=
by sorry

end NUMINAMATH_CALUDE_older_brother_running_distance_l3403_340315


namespace NUMINAMATH_CALUDE_third_team_pies_l3403_340397

/-- Given a catering job requiring 750 mini meat pies to be made by 3 teams,
    where the first team made 235 pies and the second team made 275 pies,
    prove that the third team should make 240 pies. -/
theorem third_team_pies (total : ℕ) (teams : ℕ) (first : ℕ) (second : ℕ) 
    (h1 : total = 750)
    (h2 : teams = 3)
    (h3 : first = 235)
    (h4 : second = 275) :
  total - first - second = 240 := by
  sorry

end NUMINAMATH_CALUDE_third_team_pies_l3403_340397


namespace NUMINAMATH_CALUDE_initial_mixture_volume_l3403_340393

/-- Proves that the initial volume of a milk-water mixture is 165 liters
    given the initial and final ratios, and the amount of water added. -/
theorem initial_mixture_volume
  (initial_milk : ℝ)
  (initial_water : ℝ)
  (added_water : ℝ)
  (h1 : initial_milk / initial_water = 3 / 2)
  (h2 : added_water = 66)
  (h3 : initial_milk / (initial_water + added_water) = 3 / 4) :
  initial_milk + initial_water = 165 :=
by sorry


end NUMINAMATH_CALUDE_initial_mixture_volume_l3403_340393


namespace NUMINAMATH_CALUDE_polar_to_cartesian_circle_l3403_340371

theorem polar_to_cartesian_circle (ρ θ x y : ℝ) :
  (ρ = 4 * Real.cos θ) ∧ (x = ρ * Real.cos θ) ∧ (y = ρ * Real.sin θ) →
  (x - 2)^2 + y^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_circle_l3403_340371


namespace NUMINAMATH_CALUDE_parabola_chord_length_l3403_340346

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2, 0)

-- Define the line passing through the focus with slope 45°
def line (x y : ℝ) : Prop := y = x - 2

-- Define the chord length
def chord_length : ℝ := 16

-- Theorem statement
theorem parabola_chord_length :
  ∀ (A B : ℝ × ℝ),
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  parabola x₁ y₁ ∧ parabola x₂ y₂ ∧
  line x₁ y₁ ∧ line x₂ y₂ ∧
  A ≠ B →
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = chord_length :=
by sorry

end NUMINAMATH_CALUDE_parabola_chord_length_l3403_340346


namespace NUMINAMATH_CALUDE_prob_at_least_one_multiple_of_three_l3403_340316

/-- The number of integers from 1 to 50 inclusive -/
def total_numbers : ℕ := 50

/-- The number of multiples of 3 from 1 to 50 inclusive -/
def multiples_of_three : ℕ := 16

/-- The probability of choosing a number that is not a multiple of 3 -/
def prob_not_multiple : ℚ := (total_numbers - multiples_of_three) / total_numbers

/-- The probability of choosing at least one multiple of 3 in two selections -/
def prob_at_least_one_multiple : ℚ := 1 - prob_not_multiple ^ 2

theorem prob_at_least_one_multiple_of_three :
  prob_at_least_one_multiple = 336 / 625 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_multiple_of_three_l3403_340316


namespace NUMINAMATH_CALUDE_hanks_route_length_l3403_340303

theorem hanks_route_length :
  ∀ (route_length : ℝ) (monday_speed tuesday_speed : ℝ) (time_diff : ℝ),
    monday_speed = 70 →
    tuesday_speed = 75 →
    time_diff = 1/30 →
    route_length / monday_speed - route_length / tuesday_speed = time_diff →
    route_length = 35 := by
  sorry

end NUMINAMATH_CALUDE_hanks_route_length_l3403_340303


namespace NUMINAMATH_CALUDE_dannys_chickens_l3403_340375

/-- Calculates the number of chickens on Dany's farm -/
theorem dannys_chickens (cows sheep : ℕ) (cow_sheep_bushels chicken_bushels total_bushels : ℕ) : 
  cows = 4 →
  sheep = 3 →
  cow_sheep_bushels = 2 →
  chicken_bushels = 3 →
  total_bushels = 35 →
  (cows + sheep) * cow_sheep_bushels + (total_bushels - (cows + sheep) * cow_sheep_bushels) / chicken_bushels = 7 := by
  sorry

end NUMINAMATH_CALUDE_dannys_chickens_l3403_340375
