import Mathlib

namespace NUMINAMATH_CALUDE_age_problem_l2132_213207

theorem age_problem (mehki_age jordyn_age certain_age : ℕ) : 
  mehki_age = jordyn_age + 10 →
  jordyn_age = 2 * certain_age →
  mehki_age = 22 →
  certain_age = 6 := by
  sorry

end NUMINAMATH_CALUDE_age_problem_l2132_213207


namespace NUMINAMATH_CALUDE_longest_segment_in_cylinder_l2132_213202

theorem longest_segment_in_cylinder (r h : ℝ) (hr : r = 4) (hh : h = 10) :
  Real.sqrt ((2 * r)^2 + h^2) = Real.sqrt 164 := by
  sorry

end NUMINAMATH_CALUDE_longest_segment_in_cylinder_l2132_213202


namespace NUMINAMATH_CALUDE_skew_lines_and_tetrahedron_l2132_213263

-- Define the types for points and lines
variable (Point Line : Type)

-- Define the relation for a point lying on a line
variable (lies_on : Point → Line → Prop)

-- Define the property of lines being skew
variable (skew : Line → Line → Prop)

-- Define the property of lines being perpendicular
variable (perpendicular : Line → Line → Prop)

-- Define the property of points forming a regular tetrahedron
variable (form_regular_tetrahedron : Point → Point → Point → Point → Prop)

-- State the theorem
theorem skew_lines_and_tetrahedron 
  (A B C D : Point) (a b : Line) :
  lies_on A a → lies_on B a → lies_on C b → lies_on D b →
  skew a b →
  ¬perpendicular a b →
  (∃ (AC BD : Line), lies_on A AC ∧ lies_on C AC ∧ lies_on B BD ∧ lies_on D BD ∧ skew AC BD) ∧
  ¬form_regular_tetrahedron A B C D :=
sorry

end NUMINAMATH_CALUDE_skew_lines_and_tetrahedron_l2132_213263


namespace NUMINAMATH_CALUDE_boat_journey_time_l2132_213290

/-- Calculates the total journey time for a boat traveling upstream and downstream in a river -/
theorem boat_journey_time (river_speed : ℝ) (boat_speed : ℝ) (distance : ℝ) : 
  river_speed = 2 →
  boat_speed = 6 →
  distance = 56 →
  (distance / (boat_speed - river_speed) + distance / (boat_speed + river_speed)) = 21 := by
  sorry

#check boat_journey_time

end NUMINAMATH_CALUDE_boat_journey_time_l2132_213290


namespace NUMINAMATH_CALUDE_equation_solutions_l2132_213222

theorem equation_solutions :
  (∃ x : ℝ, x - 4 = -5 ∧ x = -1) ∧
  (∃ x : ℝ, (1/2) * x + 2 = 6 ∧ x = 8) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2132_213222


namespace NUMINAMATH_CALUDE_repaired_shoes_duration_is_one_year_l2132_213273

/-- The duration for which the repaired shoes last, in years -/
def repaired_shoes_duration : ℝ := 1

/-- The cost to repair the used shoes, in dollars -/
def repair_cost : ℝ := 10.50

/-- The cost of new shoes, in dollars -/
def new_shoes_cost : ℝ := 30.00

/-- The duration for which new shoes last, in years -/
def new_shoes_duration : ℝ := 2

/-- The percentage by which the average cost per year of new shoes 
    is greater than the cost of repairing used shoes -/
def cost_difference_percentage : ℝ := 42.857142857142854

theorem repaired_shoes_duration_is_one_year :
  repaired_shoes_duration = 
    (repair_cost * (1 + cost_difference_percentage / 100)) / 
    (new_shoes_cost / new_shoes_duration) :=
by sorry

end NUMINAMATH_CALUDE_repaired_shoes_duration_is_one_year_l2132_213273


namespace NUMINAMATH_CALUDE_trig_identity_proof_l2132_213288

theorem trig_identity_proof (θ : Real) (h : Real.tan θ = 2) : 
  Real.sin θ ^ 2 + Real.sin θ * Real.cos θ - 2 * Real.cos θ ^ 2 = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l2132_213288


namespace NUMINAMATH_CALUDE_converse_of_zero_product_is_false_l2132_213282

theorem converse_of_zero_product_is_false :
  ¬ (∀ a b : ℝ, a * b = 0 → a = 0) :=
sorry

end NUMINAMATH_CALUDE_converse_of_zero_product_is_false_l2132_213282


namespace NUMINAMATH_CALUDE_triangle_side_range_l2132_213208

/-- Given a triangle ABC with side lengths a, b, and c, prove that if |a+b-6|+(a-b+4)^2=0, then 4 < c < 6 -/
theorem triangle_side_range (a b c : ℝ) (h : |a+b-6|+(a-b+4)^2=0) : 4 < c ∧ c < 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_range_l2132_213208


namespace NUMINAMATH_CALUDE_complementary_event_is_both_red_l2132_213251

/-- Represents the color of a ball -/
inductive Color
| Red
| White

/-- Represents the outcome of drawing two balls -/
structure TwoBallDraw where
  first : Color
  second : Color

/-- The set of all possible outcomes when drawing two balls -/
def allOutcomes : Set TwoBallDraw :=
  {⟨Color.Red, Color.Red⟩, ⟨Color.Red, Color.White⟩, 
   ⟨Color.White, Color.Red⟩, ⟨Color.White, Color.White⟩}

/-- Event A: At least one white ball -/
def eventA : Set TwoBallDraw :=
  {draw ∈ allOutcomes | draw.first = Color.White ∨ draw.second = Color.White}

/-- The complementary event of A -/
def complementA : Set TwoBallDraw :=
  allOutcomes \ eventA

theorem complementary_event_is_both_red :
  complementA = {⟨Color.Red, Color.Red⟩} :=
by sorry

end NUMINAMATH_CALUDE_complementary_event_is_both_red_l2132_213251


namespace NUMINAMATH_CALUDE_function_zero_implies_a_bound_l2132_213221

/-- If the function f(x) = e^x - 2x + a has a zero, then a ≤ 2ln2 - 2 -/
theorem function_zero_implies_a_bound (a : ℝ) : 
  (∃ x : ℝ, Real.exp x - 2 * x + a = 0) → a ≤ 2 * Real.log 2 - 2 := by
  sorry

end NUMINAMATH_CALUDE_function_zero_implies_a_bound_l2132_213221


namespace NUMINAMATH_CALUDE_compare_x_y_l2132_213253

theorem compare_x_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (hx4 : x^4 = 2) (hy3 : y^3 = 3) : x < y := by
  sorry

end NUMINAMATH_CALUDE_compare_x_y_l2132_213253


namespace NUMINAMATH_CALUDE_problem_solution_l2132_213224

theorem problem_solution (x : ℝ) :
  x + Real.sqrt (x^2 - 1) + 1 / (x - Real.sqrt (x^2 - 1)) = 20 →
  x^2 + Real.sqrt (x^4 - 1) + 1 / (x^2 + Real.sqrt (x^4 - 1)) = 10201 / 200 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2132_213224


namespace NUMINAMATH_CALUDE_four_digit_numbers_with_sum_counts_l2132_213293

/-- A function that returns the number of four-digit natural numbers with a given digit sum -/
def countFourDigitNumbersWithSum (sum : Nat) : Nat :=
  sorry

/-- The theorem stating the correct counts for digit sums 5, 6, and 7 -/
theorem four_digit_numbers_with_sum_counts :
  (countFourDigitNumbersWithSum 5 = 35) ∧
  (countFourDigitNumbersWithSum 6 = 56) ∧
  (countFourDigitNumbersWithSum 7 = 84) := by
  sorry

end NUMINAMATH_CALUDE_four_digit_numbers_with_sum_counts_l2132_213293


namespace NUMINAMATH_CALUDE_sum_mod_twelve_l2132_213232

theorem sum_mod_twelve : (2101 + 2103 + 2105 + 2107 + 2109) % 12 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_twelve_l2132_213232


namespace NUMINAMATH_CALUDE_max_value_constraint_l2132_213268

theorem max_value_constraint (x y z : ℝ) (h : 9*x^2 + 4*y^2 + 25*z^2 = 1) :
  ∃ (max : ℝ), max = Real.sqrt 173 ∧ 
  (∀ a b c : ℝ, 9*a^2 + 4*b^2 + 25*c^2 = 1 → 8*a + 3*b + 10*c ≤ max) ∧
  (8*x + 3*y + 10*z = max) :=
by sorry

end NUMINAMATH_CALUDE_max_value_constraint_l2132_213268


namespace NUMINAMATH_CALUDE_sequence_formula_l2132_213270

theorem sequence_formula (a : ℕ → ℝ) :
  a 1 = 1 ∧
  (∀ n : ℕ, n ≥ 2 → a n - a (n-1) = (1/3)^(n-1)) →
  ∀ n : ℕ, n ≥ 1 → a n = (3/2) * (1 - (1/3)^n) :=
by sorry

end NUMINAMATH_CALUDE_sequence_formula_l2132_213270


namespace NUMINAMATH_CALUDE_wallpaper_overlap_theorem_l2132_213283

/-- The combined area of three walls with overlapping wallpaper -/
def combined_area (two_layer_area : ℝ) (three_layer_area : ℝ) (total_covered_area : ℝ) : ℝ :=
  total_covered_area + two_layer_area + 2 * three_layer_area

/-- Theorem stating the combined area of three walls with given overlapping conditions -/
theorem wallpaper_overlap_theorem (two_layer_area : ℝ) (three_layer_area : ℝ) (total_covered_area : ℝ)
    (h1 : two_layer_area = 40)
    (h2 : three_layer_area = 40)
    (h3 : total_covered_area = 180) :
    combined_area two_layer_area three_layer_area total_covered_area = 300 := by
  sorry

end NUMINAMATH_CALUDE_wallpaper_overlap_theorem_l2132_213283


namespace NUMINAMATH_CALUDE_train_length_l2132_213235

/-- Calculates the length of a train given its speed, tunnel length, and time to pass through -/
theorem train_length (train_speed : ℝ) (tunnel_length : ℝ) (time_to_pass : ℝ) :
  train_speed = 72 →
  tunnel_length = 3.5 →
  time_to_pass = 3 / 60 →
  (train_speed * time_to_pass - tunnel_length) * 1000 = 100 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2132_213235


namespace NUMINAMATH_CALUDE_number_equation_solution_l2132_213212

theorem number_equation_solution : ∃ x : ℝ, (3 * x - 6 = 2 * x) ∧ (x = 6) := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l2132_213212


namespace NUMINAMATH_CALUDE_cookie_distribution_probability_l2132_213281

/-- Represents the number of cookies of each type -/
def num_cookies_per_type : ℕ := 4

/-- Represents the total number of cookies -/
def total_cookies : ℕ := 3 * num_cookies_per_type

/-- Represents the number of students -/
def num_students : ℕ := 4

/-- Represents the number of cookies each student receives -/
def cookies_per_student : ℕ := 3

/-- Calculates the probability of a specific distribution of cookies -/
def probability_distribution (n : ℕ) : ℚ :=
  (num_cookies_per_type * (num_cookies_per_type - 1) * (num_cookies_per_type - 2)) /
  ((total_cookies - n * cookies_per_student + 2) *
   (total_cookies - n * cookies_per_student + 1) *
   (total_cookies - n * cookies_per_student))

/-- The main theorem stating the probability of each student getting one cookie of each type -/
theorem cookie_distribution_probability :
  (probability_distribution 0 * probability_distribution 1 * probability_distribution 2) = 81 / 3850 := by
  sorry

end NUMINAMATH_CALUDE_cookie_distribution_probability_l2132_213281


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2132_213298

theorem quadratic_inequality_range (a : ℝ) :
  (∃ x : ℝ, x^2 - a*x + 5 < 0) ↔ (a < -2 * Real.sqrt 5 ∨ a > 2 * Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2132_213298


namespace NUMINAMATH_CALUDE_mrs_sheridan_fish_count_l2132_213242

theorem mrs_sheridan_fish_count :
  let initial_fish : ℕ := 22
  let fish_from_sister : ℕ := 47
  initial_fish + fish_from_sister = 69 :=
by sorry

end NUMINAMATH_CALUDE_mrs_sheridan_fish_count_l2132_213242


namespace NUMINAMATH_CALUDE_second_grade_students_selected_l2132_213200

/-- Given a school with 3300 students and a ratio of 12:10:11 for first, second, and third grades,
    prove that when 66 students are randomly selected using stratified sampling,
    the number of second-grade students selected is 20. -/
theorem second_grade_students_selected
  (total_students : ℕ)
  (first_grade_ratio second_grade_ratio third_grade_ratio : ℕ)
  (selected_students : ℕ)
  (h1 : total_students = 3300)
  (h2 : first_grade_ratio = 12)
  (h3 : second_grade_ratio = 10)
  (h4 : third_grade_ratio = 11)
  (h5 : selected_students = 66) :
  (second_grade_ratio : ℚ) / (first_grade_ratio + second_grade_ratio + third_grade_ratio) * selected_students = 20 := by
  sorry

end NUMINAMATH_CALUDE_second_grade_students_selected_l2132_213200


namespace NUMINAMATH_CALUDE_g_of_4_l2132_213214

/-- Given a function g: ℝ → ℝ satisfying g(x) + 3*g(2 - x) = 2*x^2 + x - 1 for all real x,
    prove that g(4) = -5/2 -/
theorem g_of_4 (g : ℝ → ℝ) (h : ∀ x : ℝ, g x + 3 * g (2 - x) = 2 * x^2 + x - 1) : 
  g 4 = -5/2 := by
sorry

end NUMINAMATH_CALUDE_g_of_4_l2132_213214


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l2132_213296

/-- The equation (2kx^2 + 4kx + 2) = 0 has equal roots when k = 1 -/
theorem equal_roots_quadratic (k : ℝ) : 
  (∀ x : ℝ, 2 * k * x^2 + 4 * k * x + 2 = 0) → 
  (∃! r : ℝ, 2 * x^2 + 4 * x + 2 = 0) := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l2132_213296


namespace NUMINAMATH_CALUDE_sunset_time_calculation_l2132_213266

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Represents the length of a time period in hours and minutes -/
structure Duration where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Calculates the sunset time given the sunrise time and daylight duration -/
def calculate_sunset (sunrise : Time) (daylight : Duration) : Time :=
  sorry

theorem sunset_time_calculation :
  let sunrise : Time := ⟨6, 22⟩
  let daylight : Duration := ⟨11, 36⟩
  let calculated_sunset : Time := calculate_sunset sunrise daylight
  calculated_sunset = ⟨18, 58⟩ := by sorry

end NUMINAMATH_CALUDE_sunset_time_calculation_l2132_213266


namespace NUMINAMATH_CALUDE_parallel_line_equation_perpendicular_line_coefficient_l2132_213248

-- Define the lines
def l₁ (x y : ℝ) : Prop := 3 * x + 4 * y - 2 = 0
def l₂ (x y : ℝ) : Prop := 2 * x + y + 2 = 0
def l₃ (x y : ℝ) : Prop := x - 2 * y - 1 = 0
def l₅ (a x y : ℝ) : Prop := a * x - 2 * y + 1 = 0

-- Define the intersection point P
def P : ℝ × ℝ := (-2, 2)

-- Theorem 1: The equation of line l₄
theorem parallel_line_equation : 
  ∃ (m b : ℝ), (∀ x y : ℝ, y = m * x + b ↔ (P.1 = x ∧ P.2 = y ∨ l₃ x y)) ∧ m = 1/2 ∧ b = 3 := by
  sorry

-- Theorem 2: The value of a for perpendicular lines
theorem perpendicular_line_coefficient :
  ∃! a : ℝ, ∀ x y : ℝ, (l₅ a x y ∧ l₂ x y) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_equation_perpendicular_line_coefficient_l2132_213248


namespace NUMINAMATH_CALUDE_cube_edge_increase_l2132_213289

theorem cube_edge_increase (e : ℝ) (f : ℝ) (h : e > 0) : (f * e)^3 = 8 * e^3 → f = 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_increase_l2132_213289


namespace NUMINAMATH_CALUDE_james_units_per_semester_l2132_213216

/-- Given that James pays $2000 for 2 semesters and each unit costs $50,
    prove that he takes 20 units per semester. -/
theorem james_units_per_semester 
  (total_cost : ℕ) 
  (num_semesters : ℕ) 
  (unit_cost : ℕ) 
  (h1 : total_cost = 2000) 
  (h2 : num_semesters = 2) 
  (h3 : unit_cost = 50) : 
  (total_cost / num_semesters) / unit_cost = 20 := by
  sorry

end NUMINAMATH_CALUDE_james_units_per_semester_l2132_213216


namespace NUMINAMATH_CALUDE_committee_formation_with_previous_member_l2132_213276

def total_members : ℕ := 18
def committee_size : ℕ := 6
def previous_members : ℕ := 5

theorem committee_formation_with_previous_member :
  (Nat.choose total_members committee_size) - 
  (Nat.choose (total_members - previous_members) committee_size) = 16848 := by
  sorry

end NUMINAMATH_CALUDE_committee_formation_with_previous_member_l2132_213276


namespace NUMINAMATH_CALUDE_rotation_of_D_around_E_l2132_213243

-- Define the points
def D : ℝ × ℝ := (3, 2)
def E : ℝ × ℝ := (6, 5)
def F : ℝ × ℝ := (6, 2)

-- Define the rotation function
def rotate180AroundPoint (center : ℝ × ℝ) (point : ℝ × ℝ) : ℝ × ℝ :=
  (2 * center.1 - point.1, 2 * center.2 - point.2)

-- Theorem statement
theorem rotation_of_D_around_E :
  rotate180AroundPoint E D = (9, 8) := by sorry

end NUMINAMATH_CALUDE_rotation_of_D_around_E_l2132_213243


namespace NUMINAMATH_CALUDE_combined_final_selling_price_is_630_45_l2132_213255

/-- Calculate the final selling price for an item given its cost price, profit percentage, and tax or discount percentage -/
def finalSellingPrice (costPrice : ℝ) (profitPercentage : ℝ) (taxOrDiscountPercentage : ℝ) (isTax : Bool) : ℝ :=
  let sellingPriceBeforeTaxOrDiscount := costPrice * (1 + profitPercentage)
  if isTax then
    sellingPriceBeforeTaxOrDiscount * (1 + taxOrDiscountPercentage)
  else
    sellingPriceBeforeTaxOrDiscount * (1 - taxOrDiscountPercentage)

/-- The combined final selling price for all three items -/
def combinedFinalSellingPrice : ℝ :=
  finalSellingPrice 180 0.15 0.05 true +
  finalSellingPrice 220 0.20 0.10 false +
  finalSellingPrice 130 0.25 0.08 true

theorem combined_final_selling_price_is_630_45 :
  combinedFinalSellingPrice = 630.45 := by sorry

end NUMINAMATH_CALUDE_combined_final_selling_price_is_630_45_l2132_213255


namespace NUMINAMATH_CALUDE_cos_120_degrees_l2132_213233

theorem cos_120_degrees : Real.cos (2 * Real.pi / 3) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_120_degrees_l2132_213233


namespace NUMINAMATH_CALUDE_wednesday_bags_theorem_l2132_213262

/-- Represents the leaf raking business of Bob and Johnny --/
structure LeafRakingBusiness where
  price_per_bag : ℕ
  monday_bags : ℕ
  tuesday_bags : ℕ
  total_money : ℕ

/-- Calculates the number of bags raked on Wednesday --/
def bags_raked_wednesday (business : LeafRakingBusiness) : ℕ :=
  (business.total_money - business.price_per_bag * (business.monday_bags + business.tuesday_bags)) / business.price_per_bag

/-- Theorem stating that given the conditions, the number of bags raked on Wednesday is 9 --/
theorem wednesday_bags_theorem (business : LeafRakingBusiness) 
  (h1 : business.price_per_bag = 4)
  (h2 : business.monday_bags = 5)
  (h3 : business.tuesday_bags = 3)
  (h4 : business.total_money = 68) :
  bags_raked_wednesday business = 9 := by
  sorry

#eval bags_raked_wednesday { price_per_bag := 4, monday_bags := 5, tuesday_bags := 3, total_money := 68 }

end NUMINAMATH_CALUDE_wednesday_bags_theorem_l2132_213262


namespace NUMINAMATH_CALUDE_megans_hourly_wage_l2132_213275

/-- Megan's hourly wage problem -/
theorem megans_hourly_wage (hours_per_day : ℕ) (days_per_month : ℕ) (earnings_two_months : ℕ) 
  (h1 : hours_per_day = 8)
  (h2 : days_per_month = 20)
  (h3 : earnings_two_months = 2400) :
  (earnings_two_months : ℚ) / (2 * days_per_month * hours_per_day) = 15/2 := by
  sorry

#eval (2400 : ℚ) / (2 * 20 * 8) -- This should evaluate to 7.5

end NUMINAMATH_CALUDE_megans_hourly_wage_l2132_213275


namespace NUMINAMATH_CALUDE_malfunctioning_odometer_theorem_l2132_213210

/-- Converts a digit in the malfunctioning odometer system to its actual value -/
def convert_digit (d : Nat) : Nat :=
  if d < 4 then d else d + 2

/-- Converts an odometer reading to actual miles -/
def odometer_to_miles (reading : List Nat) : Nat :=
  reading.foldr (fun d acc => convert_digit d + 8 * acc) 0

/-- Theorem: The malfunctioning odometer reading 000306 corresponds to 134 miles -/
theorem malfunctioning_odometer_theorem :
  odometer_to_miles [0, 0, 0, 3, 0, 6] = 134 := by
  sorry

#eval odometer_to_miles [0, 0, 0, 3, 0, 6]

end NUMINAMATH_CALUDE_malfunctioning_odometer_theorem_l2132_213210


namespace NUMINAMATH_CALUDE_sin_plus_one_quasi_odd_quasi_odd_to_odd_cubic_quasi_odd_l2132_213260

/-- Definition of a quasi-odd function -/
def QuasiOdd (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ ∀ x, f (a + x) + f (a - x) = 2 * b

/-- Central point of a quasi-odd function -/
def CentralPoint (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (a ≠ 0 ∨ b ≠ 0) ∧ ∀ x, f (a + x) + f (a - x) = 2 * b

/-- Theorem stating that sin(x) + 1 is a quasi-odd function with central point (0, 1) -/
theorem sin_plus_one_quasi_odd :
  QuasiOdd (fun x ↦ Real.sin x + 1) ∧ CentralPoint (fun x ↦ Real.sin x + 1) 0 1 := by
  sorry

/-- Theorem stating that if f is quasi-odd with central point (a, f(a)),
    then F(x) = f(x+a) - f(a) is odd -/
theorem quasi_odd_to_odd (f : ℝ → ℝ) (a : ℝ) :
  QuasiOdd f ∧ CentralPoint f a (f a) →
  ∀ x, f ((x + a) + a) - f a = -(f ((-x + a) + a) - f a) := by
  sorry

/-- Theorem stating that x^3 - 3x^2 + 6x - 2 is a quasi-odd function with central point (1, 2) -/
theorem cubic_quasi_odd :
  QuasiOdd (fun x ↦ x^3 - 3*x^2 + 6*x - 2) ∧
  CentralPoint (fun x ↦ x^3 - 3*x^2 + 6*x - 2) 1 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_plus_one_quasi_odd_quasi_odd_to_odd_cubic_quasi_odd_l2132_213260


namespace NUMINAMATH_CALUDE_fort_blocks_count_l2132_213249

/-- Represents the dimensions of a rectangular fort --/
structure FortDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of blocks needed to build a fort with given dimensions and wall thickness --/
def blocksNeeded (d : FortDimensions) (wallThickness : ℕ) : ℕ :=
  d.length * d.width * d.height - (d.length - 2 * wallThickness) * (d.width - 2 * wallThickness) * (d.height - 2 * wallThickness)

/-- Theorem stating that a fort with dimensions 15x12x6 and wall thickness 1 requires 560 blocks --/
theorem fort_blocks_count : 
  let fortDims : FortDimensions := { length := 15, width := 12, height := 6 }
  blocksNeeded fortDims 1 = 560 := by
  sorry

end NUMINAMATH_CALUDE_fort_blocks_count_l2132_213249


namespace NUMINAMATH_CALUDE_annie_pays_36_for_12kg_l2132_213267

-- Define the price function for oranges
def price (mass : ℝ) : ℝ := sorry

-- Define the given conditions
axiom price_proportional : ∃ k : ℝ, ∀ m : ℝ, price m = k * m
axiom paid_36_for_12kg : price 12 = 36

-- Theorem to prove
theorem annie_pays_36_for_12kg : price 12 = 36 := by
  sorry

end NUMINAMATH_CALUDE_annie_pays_36_for_12kg_l2132_213267


namespace NUMINAMATH_CALUDE_train_speed_l2132_213258

/-- Given a train of length 125 metres that takes 7.5 seconds to pass a pole, 
    its speed is 60 km/hr. -/
theorem train_speed (train_length : Real) (time_to_pass : Real) 
  (h1 : train_length = 125) 
  (h2 : time_to_pass = 7.5) : 
  (train_length / time_to_pass) * 3.6 = 60 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2132_213258


namespace NUMINAMATH_CALUDE_circle_equation_tangent_to_line_l2132_213236

/-- The equation of a circle with center (0, 1) tangent to the line y = 2 is x^2 + (y-1)^2 = 1 -/
theorem circle_equation_tangent_to_line (x y : ℝ) : 
  (∃ (r : ℝ), r > 0 ∧ x^2 + (y - 1)^2 = r^2 ∧ |2 - 1| = r) → 
  x^2 + (y - 1)^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_tangent_to_line_l2132_213236


namespace NUMINAMATH_CALUDE_lcm_hcf_problem_l2132_213259

theorem lcm_hcf_problem (A B : ℕ) (h1 : A = 330) (h2 : Nat.lcm A B = 2310) (h3 : Nat.gcd A B = 30) :
  B = 210 := by
  sorry

end NUMINAMATH_CALUDE_lcm_hcf_problem_l2132_213259


namespace NUMINAMATH_CALUDE_divisibility_implies_equality_l2132_213203

theorem divisibility_implies_equality (a b : ℕ) :
  (4 * a * b - 1) ∣ (4 * a^2 - 1)^2 → a = b := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implies_equality_l2132_213203


namespace NUMINAMATH_CALUDE_gcd_72_168_l2132_213230

theorem gcd_72_168 : Nat.gcd 72 168 = 24 := by
  sorry

end NUMINAMATH_CALUDE_gcd_72_168_l2132_213230


namespace NUMINAMATH_CALUDE_amalie_remaining_coins_l2132_213206

/-- Given the ratio of Elsa's coins to Amalie's coins and their total coins,
    calculate how many coins Amalie remains with after spending 3/4 of her coins. -/
theorem amalie_remaining_coins
  (ratio_elsa : ℚ)
  (ratio_amalie : ℚ)
  (total_coins : ℕ)
  (h_ratio : ratio_elsa / ratio_amalie = 10 / 45)
  (h_total : ratio_elsa + ratio_amalie = 1)
  (h_coins : (ratio_amalie * total_coins : ℚ).num * (1 / (ratio_amalie * total_coins : ℚ).den : ℚ) * total_coins = 360) :
  (1 / 4 : ℚ) * ((ratio_amalie * total_coins : ℚ).num * (1 / (ratio_amalie * total_coins : ℚ).den : ℚ) * total_coins) = 90 :=
sorry

end NUMINAMATH_CALUDE_amalie_remaining_coins_l2132_213206


namespace NUMINAMATH_CALUDE_bookshelf_problem_l2132_213234

theorem bookshelf_problem (shelf_length : ℕ) (total_books : ℕ) (thin_book_thickness : ℕ) (thick_book_thickness : ℕ) :
  shelf_length = 200 →
  total_books = 46 →
  thin_book_thickness = 3 →
  thick_book_thickness = 5 →
  ∃ (thin_books thick_books : ℕ),
    thin_books + thick_books = total_books ∧
    thin_books * thin_book_thickness + thick_books * thick_book_thickness = shelf_length ∧
    thin_books = 15 :=
by sorry

end NUMINAMATH_CALUDE_bookshelf_problem_l2132_213234


namespace NUMINAMATH_CALUDE_tv_price_change_l2132_213279

theorem tv_price_change (initial_price : ℝ) (h : initial_price > 0) : 
  let price_after_decrease : ℝ := 0.8 * initial_price
  let final_price : ℝ := 1.24 * initial_price
  ∃ x : ℝ, price_after_decrease * (1 + x / 100) = final_price ∧ x = 55 := by
sorry

end NUMINAMATH_CALUDE_tv_price_change_l2132_213279


namespace NUMINAMATH_CALUDE_sum_of_squared_roots_l2132_213294

theorem sum_of_squared_roots (p q r : ℝ) : 
  (3 * p^3 + 2 * p^2 - 3 * p - 8 = 0) →
  (3 * q^3 + 2 * q^2 - 3 * q - 8 = 0) →
  (3 * r^3 + 2 * r^2 - 3 * r - 8 = 0) →
  p^2 + q^2 + r^2 = -14/9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squared_roots_l2132_213294


namespace NUMINAMATH_CALUDE_quadratic_integer_solution_l2132_213252

theorem quadratic_integer_solution (a : ℕ+) :
  (∃ x : ℤ, a * x^2 + 2*(2*a - 1)*x + 4*a - 7 = 0) ↔ (a = 1 ∨ a = 5) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_integer_solution_l2132_213252


namespace NUMINAMATH_CALUDE_ninth_grade_students_count_l2132_213227

def total_payment : ℝ := 1936
def additional_sets : ℕ := 88
def discount_rate : ℝ := 0.2

theorem ninth_grade_students_count :
  ∃ x : ℕ, 
    (total_payment / x) * (1 - discount_rate) = total_payment / (x + additional_sets) ∧
    x = 352 := by
  sorry

end NUMINAMATH_CALUDE_ninth_grade_students_count_l2132_213227


namespace NUMINAMATH_CALUDE_gear_system_teeth_count_l2132_213278

theorem gear_system_teeth_count (teeth1 teeth2 rotations3 : ℕ) 
  (h1 : teeth1 = 32)
  (h2 : teeth2 = 24)
  (h3 : rotations3 = 8)
  (h4 : ∃ total_teeth : ℕ, 
    total_teeth % 8 = 0 ∧ 
    total_teeth > teeth1 * 4 ∧ 
    total_teeth < teeth2 * 6 ∧
    total_teeth % teeth1 = 0 ∧
    total_teeth % teeth2 = 0 ∧
    total_teeth % rotations3 = 0) :
  total_teeth / rotations3 = 17 := by
  sorry

end NUMINAMATH_CALUDE_gear_system_teeth_count_l2132_213278


namespace NUMINAMATH_CALUDE_total_squares_5x6_l2132_213244

/-- The number of squares of a given size in a grid --/
def count_squares (rows : ℕ) (cols : ℕ) (size : ℕ) : ℕ :=
  (rows - size) * (cols - size)

/-- The total number of squares in a 5x6 grid --/
def total_squares : ℕ :=
  count_squares 5 6 1 + count_squares 5 6 2 + count_squares 5 6 3 + count_squares 5 6 4

/-- Theorem: The total number of squares in a 5x6 grid is 40 --/
theorem total_squares_5x6 : total_squares = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_squares_5x6_l2132_213244


namespace NUMINAMATH_CALUDE_air_quality_consecutive_good_days_l2132_213229

/-- Represents the air quality index for a given day -/
def AirQualityIndex := ℕ → ℝ

/-- Determines if the air quality is good for a given index -/
def is_good (index : ℝ) : Prop := index < 100

/-- Determines if two consecutive days have good air quality -/
def consecutive_good_days (aqi : AirQualityIndex) (day : ℕ) : Prop :=
  is_good (aqi day) ∧ is_good (aqi (day + 1))

/-- The air quality index for the 10 days -/
axiom aqi : AirQualityIndex

/-- The theorem to prove -/
theorem air_quality_consecutive_good_days :
  (consecutive_good_days aqi 1 ∧ consecutive_good_days aqi 5) ∧
  (∀ d : ℕ, d ≠ 1 ∧ d ≠ 5 → ¬consecutive_good_days aqi d) :=
sorry

end NUMINAMATH_CALUDE_air_quality_consecutive_good_days_l2132_213229


namespace NUMINAMATH_CALUDE_worker_speed_ratio_l2132_213272

/-- Given two workers a and b, where b can complete a work in 60 days,
    and a and b together can complete the work in 12 days,
    prove that the ratio of a's speed to b's speed is 4:1 -/
theorem worker_speed_ratio (a b : ℝ) 
    (h₁ : b = 1 / 60)  -- b's speed (work per day)
    (h₂ : a + b = 1 / 12) -- combined speed of a and b (work per day)
    : a / b = 4 := by
  sorry

end NUMINAMATH_CALUDE_worker_speed_ratio_l2132_213272


namespace NUMINAMATH_CALUDE_petrol_price_equation_l2132_213218

/-- The original price of petrol satisfies the equation relating to a 15% price reduction and additional 7 gallons for $300 -/
theorem petrol_price_equation (P : ℝ) : P > 0 → 300 / (0.85 * P) = 300 / P + 7 := by
  sorry

end NUMINAMATH_CALUDE_petrol_price_equation_l2132_213218


namespace NUMINAMATH_CALUDE_parabola_point_relationship_l2132_213280

/-- Parabola function -/
def f (x m : ℝ) : ℝ := x^2 - 4*x - m

theorem parabola_point_relationship (m : ℝ) :
  let y₁ := f 2 m
  let y₂ := f (-3) m
  let y₃ := f (-1) m
  y₁ < y₃ ∧ y₃ < y₂ := by sorry

end NUMINAMATH_CALUDE_parabola_point_relationship_l2132_213280


namespace NUMINAMATH_CALUDE_f_one_equals_four_l2132_213241

/-- A function f(x) that is always non-negative for real x -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x - 3*a - 9

/-- The theorem stating that f(1) = 4 given the conditions -/
theorem f_one_equals_four (a : ℝ) (h : ∀ x : ℝ, f a x ≥ 0) : f a 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_f_one_equals_four_l2132_213241


namespace NUMINAMATH_CALUDE_probability_sum_seven_is_one_sixth_l2132_213205

def dice_faces : ℕ := 6

def favorable_outcomes : ℕ := 6

def total_outcomes : ℕ := dice_faces * dice_faces

def probability_sum_seven : ℚ := favorable_outcomes / total_outcomes

theorem probability_sum_seven_is_one_sixth : 
  probability_sum_seven = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_probability_sum_seven_is_one_sixth_l2132_213205


namespace NUMINAMATH_CALUDE_min_races_for_top_five_l2132_213215

/-- Represents a horse in the race. -/
structure Horse :=
  (id : Nat)

/-- Represents a race with up to 4 horses. -/
structure Race :=
  (participants : Finset Horse)
  (hLimited : participants.card ≤ 4)

/-- Represents the outcome of a series of races. -/
structure RaceOutcome :=
  (races : List Race)
  (topFive : Finset Horse)
  (hTopFiveSize : topFive.card = 5)

/-- The main theorem stating the minimum number of races required. -/
theorem min_races_for_top_five (horses : Finset Horse) 
  (hSize : horses.card = 30) :
  ∃ (outcome : RaceOutcome), 
    outcome.topFive ⊆ horses ∧ 
    outcome.races.length = 8 ∧ 
    (∀ (alt_outcome : RaceOutcome), 
      alt_outcome.topFive ⊆ horses → 
      alt_outcome.races.length ≥ 8) :=
sorry

end NUMINAMATH_CALUDE_min_races_for_top_five_l2132_213215


namespace NUMINAMATH_CALUDE_no_perfect_square_19xx99_l2132_213292

theorem no_perfect_square_19xx99 : ¬ ∃ (n : ℕ), 
  (n * n ≥ 1900000) ∧ 
  (n * n < 2000000) ∧ 
  (n * n % 100 = 99) := by
sorry

end NUMINAMATH_CALUDE_no_perfect_square_19xx99_l2132_213292


namespace NUMINAMATH_CALUDE_binomial_expansion_example_l2132_213231

theorem binomial_expansion_example : 
  57^4 + 4*(57^3 * 2) + 6*(57^2 * 2^2) + 4*(57 * 2^3) + 2^4 = 12117361 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_example_l2132_213231


namespace NUMINAMATH_CALUDE_vertex_coordinates_l2132_213201

-- Define the parabola equation
def parabola (x : ℝ) : ℝ := -(x - 1)^2 - 2

-- State the theorem
theorem vertex_coordinates :
  ∃ (x y : ℝ), x = 1 ∧ y = -2 ∧ ∀ (t : ℝ), parabola t ≤ parabola x :=
sorry

end NUMINAMATH_CALUDE_vertex_coordinates_l2132_213201


namespace NUMINAMATH_CALUDE_min_cosine_sqrt3_sine_l2132_213240

theorem min_cosine_sqrt3_sine (A : Real) :
  let f := λ A : Real => Real.cos (A / 2) + Real.sqrt 3 * Real.sin (A / 2)
  ∃ (min : Real), f min ≤ f A ∧ min = 840 * Real.pi / 180 :=
sorry

end NUMINAMATH_CALUDE_min_cosine_sqrt3_sine_l2132_213240


namespace NUMINAMATH_CALUDE_correct_subtraction_result_l2132_213284

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  units : Nat
  is_valid : tens ≥ 1 ∧ tens ≤ 9 ∧ units ≤ 9

/-- Calculates the numeric value of a two-digit number -/
def TwoDigitNumber.value (n : TwoDigitNumber) : Nat :=
  10 * n.tens + n.units

theorem correct_subtraction_result : ∀ (minuend subtrahend : TwoDigitNumber),
  minuend.units = 3 →
  (minuend.value - 3 + 5) - 25 = 60 →
  subtrahend.value = 52 →
  minuend.value - subtrahend.value = 31 := by
  sorry

end NUMINAMATH_CALUDE_correct_subtraction_result_l2132_213284


namespace NUMINAMATH_CALUDE_g_equals_zero_l2132_213238

/-- The function g(x) = 5x - 7 -/
def g (x : ℝ) : ℝ := 5 * x - 7

/-- Theorem: g(7/5) = 0 -/
theorem g_equals_zero : g (7 / 5) = 0 := by
  sorry

end NUMINAMATH_CALUDE_g_equals_zero_l2132_213238


namespace NUMINAMATH_CALUDE_zero_subset_X_l2132_213265

def X : Set ℝ := {x | x > -1}

theorem zero_subset_X : {0} ⊆ X := by sorry

end NUMINAMATH_CALUDE_zero_subset_X_l2132_213265


namespace NUMINAMATH_CALUDE_rectangle_area_l2132_213286

/-- Represents a square in the rectangle --/
structure Square where
  sideLength : ℝ
  area : ℝ
  area_def : area = sideLength ^ 2

/-- The rectangle XYZW with its squares --/
structure Rectangle where
  smallSquares : Fin 3 → Square
  largeSquare : Square
  smallSquaresEqual : ∀ i j : Fin 3, (smallSquares i).sideLength = (smallSquares j).sideLength
  smallSquareArea : ∀ i : Fin 3, (smallSquares i).area = 4
  largeSquareSideLength : largeSquare.sideLength = 2 * (smallSquares 0).sideLength
  noOverlap : True  -- This condition is simplified as it's hard to represent geometrically

/-- The theorem to prove --/
theorem rectangle_area (rect : Rectangle) : 
  (3 * (rect.smallSquares 0).area + rect.largeSquare.area : ℝ) = 28 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2132_213286


namespace NUMINAMATH_CALUDE_baseball_team_groups_l2132_213257

theorem baseball_team_groups (new_players returning_players players_per_group : ℕ) 
  (h1 : new_players = 4)
  (h2 : returning_players = 6)
  (h3 : players_per_group = 5) :
  (new_players + returning_players) / players_per_group = 2 := by
  sorry

end NUMINAMATH_CALUDE_baseball_team_groups_l2132_213257


namespace NUMINAMATH_CALUDE_number_equation_l2132_213254

theorem number_equation (x : ℝ) : 3 * x - 4 = 5 ↔ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l2132_213254


namespace NUMINAMATH_CALUDE_volume_sphere_minus_cylinder_l2132_213277

/-- The volume of space inside a sphere and outside an inscribed right cylinder -/
theorem volume_sphere_minus_cylinder (r_sphere : ℝ) (r_cylinder : ℝ) : 
  r_sphere = 6 → r_cylinder = 4 → 
  ∃ V : ℝ, V = (288 - 64 * Real.sqrt 5) * Real.pi ∧
    V = (4 / 3 * Real.pi * r_sphere^3) - (Real.pi * r_cylinder^2 * Real.sqrt (r_sphere^2 - r_cylinder^2)) :=
by sorry

end NUMINAMATH_CALUDE_volume_sphere_minus_cylinder_l2132_213277


namespace NUMINAMATH_CALUDE_tangent_line_at_origin_l2132_213245

/-- Given a real number a, if f(x) = x^3 + ax^2 + (a - 3)x and its derivative f'(x) is an even function,
    then the equation of the tangent line to the curve y = f(x) at the origin is 2x + y = 0. -/
theorem tangent_line_at_origin (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^3 + a*x^2 + (a - 3)*x
  let f' : ℝ → ℝ := λ x ↦ 3*x^2 + 2*a*x + (a - 3)
  (∀ x, f' x = f' (-x)) →  -- f' is an even function
  (λ x y ↦ 2*x + y = 0) = (λ x y ↦ y = f' 0 * x) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_at_origin_l2132_213245


namespace NUMINAMATH_CALUDE_problem_solution_l2132_213247

theorem problem_solution (x z : ℝ) (hx : x ≠ 0) (h1 : x/3 = z^2 + 1) (h2 : x/5 = 5*z + 2) :
  x = (685 + 25 * Real.sqrt 541) / 6 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2132_213247


namespace NUMINAMATH_CALUDE_cost_of_second_set_l2132_213287

/-- The cost of a set of pencils and pens -/
def cost_set (pencil_count : ℕ) (pen_count : ℕ) (pencil_cost : ℚ) (pen_cost : ℚ) : ℚ :=
  pencil_count * pencil_cost + pen_count * pen_cost

/-- The theorem stating that the cost of 4 pencils and 5 pens is 2.00 dollars -/
theorem cost_of_second_set :
  ∃ (pen_cost : ℚ),
    cost_set 4 5 0.1 pen_cost = 2 ∧
    cost_set 4 5 0.1 pen_cost = cost_set 4 5 0.1 ((2 - 4 * 0.1) / 5) :=
by
  sorry

end NUMINAMATH_CALUDE_cost_of_second_set_l2132_213287


namespace NUMINAMATH_CALUDE_exact_five_blue_probability_l2132_213297

def total_marbles : ℕ := 12
def blue_marbles : ℕ := 8
def red_marbles : ℕ := 4
def total_draws : ℕ := 8
def blue_draws : ℕ := 5

def probability_blue : ℚ := blue_marbles / total_marbles
def probability_red : ℚ := red_marbles / total_marbles

theorem exact_five_blue_probability :
  (Nat.choose total_draws blue_draws : ℚ) *
  (probability_blue ^ blue_draws) *
  (probability_red ^ (total_draws - blue_draws)) =
  (56 : ℚ) * 32 / 6561 :=
sorry

end NUMINAMATH_CALUDE_exact_five_blue_probability_l2132_213297


namespace NUMINAMATH_CALUDE_inequality_solution_equivalence_l2132_213213

-- Define the types for our variables
variable (x : ℝ)
variable (a b : ℝ)

-- Define the original inequality and its solution set
def original_inequality (x a b : ℝ) : Prop := a * (x + b) * (x + 5 / a) > 0
def original_solution_set (x : ℝ) : Prop := x < -1 ∨ x > 3

-- Define the new inequality we want to solve
def new_inequality (x a b : ℝ) : Prop := x^2 + b*x - 2*a < 0

-- Define the solution set we want to prove
def target_solution_set (x : ℝ) : Prop := x > -2 ∧ x < 5

-- State the theorem
theorem inequality_solution_equivalence :
  (∀ x, original_inequality x a b ↔ original_solution_set x) →
  (∀ x, new_inequality x a b ↔ target_solution_set x) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_equivalence_l2132_213213


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l2132_213250

theorem quadratic_roots_relation (k n p : ℝ) (hk : k ≠ 0) (hn : n ≠ 0) (hp : p ≠ 0) :
  (∃ s₁ s₂ : ℝ, (s₁ + s₂ = -p ∧ s₁ * s₂ = k) ∧
               (3*s₁ + 3*s₂ = -k ∧ 9*s₁*s₂ = n)) →
  n / p = 27 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l2132_213250


namespace NUMINAMATH_CALUDE_three_numbers_sum_l2132_213228

theorem three_numbers_sum (a b c : ℝ) : 
  a ≤ b → b ≤ c → 
  b = 7 → 
  (a + b + c) / 3 = a + 12 → 
  (a + b + c) / 3 = c - 18 → 
  a + b + c = 39 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l2132_213228


namespace NUMINAMATH_CALUDE_min_value_product_l2132_213271

theorem min_value_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 8) :
  (3 * a + b) * (a + 3 * c) * (2 * b * c + 4) ≥ 384 ∧
  ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ a₀ * b₀ * c₀ = 8 ∧
    (3 * a₀ + b₀) * (a₀ + 3 * c₀) * (2 * b₀ * c₀ + 4) = 384 :=
by sorry

end NUMINAMATH_CALUDE_min_value_product_l2132_213271


namespace NUMINAMATH_CALUDE_lcm_from_product_and_hcf_l2132_213261

theorem lcm_from_product_and_hcf (A B : ℕ+) :
  A * B = 62216 →
  Nat.gcd A B = 22 →
  Nat.lcm A B = 2828 := by
sorry

end NUMINAMATH_CALUDE_lcm_from_product_and_hcf_l2132_213261


namespace NUMINAMATH_CALUDE_sum_of_smallest_multiples_l2132_213204

/-- The smallest positive two-digit multiple of 5 -/
def c : ℕ := sorry

/-- The smallest positive three-digit multiple of 7 -/
def d : ℕ := sorry

/-- c is a two-digit number -/
axiom c_two_digit : 10 ≤ c ∧ c ≤ 99

/-- d is a three-digit number -/
axiom d_three_digit : 100 ≤ d ∧ d ≤ 999

/-- c is a multiple of 5 -/
axiom c_multiple_of_5 : ∃ k : ℕ, c = 5 * k

/-- d is a multiple of 7 -/
axiom d_multiple_of_7 : ∃ k : ℕ, d = 7 * k

/-- c is the smallest two-digit multiple of 5 -/
axiom c_smallest : ∀ x : ℕ, (10 ≤ x ∧ x ≤ 99 ∧ ∃ k : ℕ, x = 5 * k) → c ≤ x

/-- d is the smallest three-digit multiple of 7 -/
axiom d_smallest : ∀ x : ℕ, (100 ≤ x ∧ x ≤ 999 ∧ ∃ k : ℕ, x = 7 * k) → d ≤ x

theorem sum_of_smallest_multiples : c + d = 115 := by sorry

end NUMINAMATH_CALUDE_sum_of_smallest_multiples_l2132_213204


namespace NUMINAMATH_CALUDE_smallest_factorizable_b_l2132_213239

/-- Represents a factorization of x^2 + bx + 2016 into (x + r)(x + s) -/
structure Factorization where
  r : ℤ
  s : ℤ
  sum_eq : r + s = b
  product_eq : r * s = 2016

/-- Returns true if the quadratic x^2 + bx + 2016 can be factored with integer coefficients -/
def has_integer_factorization (b : ℤ) : Prop :=
  ∃ f : Factorization, f.r + f.s = b ∧ f.r * f.s = 2016

theorem smallest_factorizable_b :
  (has_integer_factorization 90) ∧
  (∀ b : ℤ, 0 < b → b < 90 → ¬(has_integer_factorization b)) :=
sorry

end NUMINAMATH_CALUDE_smallest_factorizable_b_l2132_213239


namespace NUMINAMATH_CALUDE_car_speed_problem_l2132_213209

/-- Proves that given a car traveling 75% of a trip at 50 mph and the remaining 25% at speed s,
    if the average speed for the entire trip is 50 mph, then s must equal 50 mph. -/
theorem car_speed_problem (D : ℝ) (s : ℝ) (h1 : D > 0) (h2 : s > 0) : 
  (0.75 * D / 50 + 0.25 * D / s) = D / 50 → s = 50 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l2132_213209


namespace NUMINAMATH_CALUDE_complex_number_problem_l2132_213299

-- Define the complex number Z₁
def Z₁ (a : ℝ) : ℂ := 2 + a * Complex.I

-- Main theorem
theorem complex_number_problem (a : ℝ) (ha : a > 0) 
  (h_pure_imag : ∃ b : ℝ, Z₁ a ^ 2 = b * Complex.I) :
  a = 2 ∧ Complex.abs (Z₁ a / (1 - Complex.I)) = 2 := by
  sorry


end NUMINAMATH_CALUDE_complex_number_problem_l2132_213299


namespace NUMINAMATH_CALUDE_average_speed_of_trip_l2132_213211

/-- Proves that the average speed of a 100-mile trip is 40 mph, given specific conditions -/
theorem average_speed_of_trip (total_distance : ℝ) (first_part_distance : ℝ) (second_part_distance : ℝ)
  (first_part_speed : ℝ) (second_part_speed : ℝ) (h1 : total_distance = 100)
  (h2 : first_part_distance = 30) (h3 : second_part_distance = 70)
  (h4 : first_part_speed = 60) (h5 : second_part_speed = 35)
  (h6 : total_distance = first_part_distance + second_part_distance) :
  (total_distance) / ((first_part_distance / first_part_speed) + (second_part_distance / second_part_speed)) = 40 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_of_trip_l2132_213211


namespace NUMINAMATH_CALUDE_traffic_light_probability_l2132_213285

theorem traffic_light_probability (m : ℕ) : 
  (35 : ℝ) / (38 + m) > (m : ℝ) / (38 + m) → m = 30 :=
by sorry

end NUMINAMATH_CALUDE_traffic_light_probability_l2132_213285


namespace NUMINAMATH_CALUDE_f_odd_and_decreasing_l2132_213225

-- Define the function f(x) = -x|x|
def f (x : ℝ) : ℝ := -x * abs x

-- Theorem stating that f is both odd and decreasing
theorem f_odd_and_decreasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ > f x₂) :=
by
  sorry

end NUMINAMATH_CALUDE_f_odd_and_decreasing_l2132_213225


namespace NUMINAMATH_CALUDE_prob_heart_then_king_is_one_fiftytwo_l2132_213220

-- Define a standard deck of cards
def standard_deck : ℕ := 52

-- Define the number of hearts in a standard deck
def hearts_in_deck : ℕ := 13

-- Define the number of kings in a standard deck
def kings_in_deck : ℕ := 4

-- Define the probability of drawing a heart first and a king second
def prob_heart_then_king : ℚ := hearts_in_deck / standard_deck * kings_in_deck / (standard_deck - 1)

-- Theorem statement
theorem prob_heart_then_king_is_one_fiftytwo :
  prob_heart_then_king = 1 / standard_deck :=
by sorry

end NUMINAMATH_CALUDE_prob_heart_then_king_is_one_fiftytwo_l2132_213220


namespace NUMINAMATH_CALUDE_arcade_spending_fraction_l2132_213291

def allowance : ℚ := 4.5

theorem arcade_spending_fraction (x : ℚ) 
  (h1 : (2/3) * (1 - x) * allowance = 1.2) : x = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_arcade_spending_fraction_l2132_213291


namespace NUMINAMATH_CALUDE_log_product_theorem_l2132_213274

-- Define the exponent rule
axiom exponent_rule {a : ℝ} (m n : ℝ) : a^m * a^n = a^(m + n)

-- Define the logarithm function
noncomputable def log (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

-- State the theorem
theorem log_product_theorem (b x y : ℝ) (hb : b > 0) (hb1 : b ≠ 1) (hx : x > 0) (hy : y > 0) :
  log b (x * y) = log b x + log b y :=
sorry

end NUMINAMATH_CALUDE_log_product_theorem_l2132_213274


namespace NUMINAMATH_CALUDE_kids_difference_l2132_213226

theorem kids_difference (monday_kids tuesday_kids : ℕ) 
  (h1 : monday_kids = 18) 
  (h2 : tuesday_kids = 10) : 
  monday_kids - tuesday_kids = 8 := by
sorry

end NUMINAMATH_CALUDE_kids_difference_l2132_213226


namespace NUMINAMATH_CALUDE_ze_and_triplet_ages_l2132_213246

/-- Represents the ages of Zé Roberto and his children -/
structure FamilyAges where
  ze : ℕ
  twin : ℕ
  triplet : ℕ

/-- Conditions for Zé Roberto's family ages -/
def valid_family_ages (f : FamilyAges) : Prop :=
  -- Zé's current age equals the sum of his children's ages
  f.ze = 2 * f.twin + 3 * f.triplet ∧
  -- In 15 years, the sum of the children's ages will be twice Zé's age
  2 * (f.ze + 15) = 2 * (f.twin + 15) + 3 * (f.triplet + 15) ∧
  -- In 15 years, the sum of the twins' ages will equal the sum of the triplets' ages
  2 * (f.twin + 15) = 3 * (f.triplet + 15)

/-- Theorem stating Zé's current age and the age of each triplet -/
theorem ze_and_triplet_ages (f : FamilyAges) (h : valid_family_ages f) :
  f.ze = 45 ∧ f.triplet = 5 := by
  sorry


end NUMINAMATH_CALUDE_ze_and_triplet_ages_l2132_213246


namespace NUMINAMATH_CALUDE_clock_hands_angle_at_3_15_clock_hands_angle_at_3_15_is_7_5_l2132_213295

/-- The angle between clock hands at 3:15 -/
theorem clock_hands_angle_at_3_15 : ℝ :=
  let hours_on_clock : ℕ := 12
  let degrees_per_hour : ℝ := 360 / hours_on_clock
  let minutes_per_hour : ℕ := 60
  let degrees_per_minute : ℝ := 360 / minutes_per_hour
  let minutes_past_3 : ℕ := 15
  let minute_hand_angle : ℝ := degrees_per_minute * minutes_past_3
  let hour_hand_angle : ℝ := 3 * degrees_per_hour + (degrees_per_hour / 4)
  let angle_difference : ℝ := hour_hand_angle - minute_hand_angle
  angle_difference

theorem clock_hands_angle_at_3_15_is_7_5 :
  clock_hands_angle_at_3_15 = 7.5 := by sorry

end NUMINAMATH_CALUDE_clock_hands_angle_at_3_15_clock_hands_angle_at_3_15_is_7_5_l2132_213295


namespace NUMINAMATH_CALUDE_arc_length_cardioid_l2132_213219

/-- The arc length of the curve ρ = 1 - sin φ from -π/2 to -π/6 is 2 -/
theorem arc_length_cardioid (φ : ℝ) : 
  let ρ : ℝ → ℝ := λ φ => 1 - Real.sin φ
  let L : ℝ := ∫ φ in Set.Icc (-Real.pi/2) (-Real.pi/6), 
    Real.sqrt ((ρ φ)^2 + (- Real.cos φ)^2)
  L = 2 := by sorry

end NUMINAMATH_CALUDE_arc_length_cardioid_l2132_213219


namespace NUMINAMATH_CALUDE_right_triangle_division_area_ratio_l2132_213269

/-- Given a right triangle divided into a rectangle and two smaller right triangles,
    where one smaller triangle has area n times the rectangle's area,
    and the rectangle's length is twice its width,
    prove that the ratio of the other small triangle's area to the rectangle's area is 1/(4n). -/
theorem right_triangle_division_area_ratio (n : ℝ) (h : n > 0) :
  ∃ (x : ℝ) (t s : ℝ),
    x > 0 ∧ 
    2 * x^2 > 0 ∧  -- Area of rectangle
    (1/2) * t * x = n * (2 * x^2) ∧  -- Area of one small triangle
    s / x = x / (2 * n * x) ∧  -- Similar triangles ratio
    ((1/2) * (2*x) * s) / (2 * x^2) = 1 / (4*n) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_division_area_ratio_l2132_213269


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l2132_213223

theorem smallest_prime_divisor_of_sum (n : ℕ) (h : n = 3^15 + 5^21) :
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ n ∧ ∀ q : ℕ, Nat.Prime q → q ∣ n → p ≤ q :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l2132_213223


namespace NUMINAMATH_CALUDE_jims_remaining_distance_l2132_213217

theorem jims_remaining_distance 
  (total_distance : ℕ) 
  (driven_distance : ℕ) 
  (b_to_c : ℕ) 
  (c_to_d : ℕ) 
  (d_to_e : ℕ) 
  (h1 : total_distance = 2500) 
  (h2 : driven_distance = 642) 
  (h3 : b_to_c = 400) 
  (h4 : c_to_d = 550) 
  (h5 : d_to_e = 200) : 
  total_distance - driven_distance = b_to_c + c_to_d + d_to_e :=
by sorry

end NUMINAMATH_CALUDE_jims_remaining_distance_l2132_213217


namespace NUMINAMATH_CALUDE_target_compound_has_one_iodine_l2132_213237

/-- Represents the number of atoms of each element in the compound -/
structure Compound where
  nitrogen : ℕ
  hydrogen : ℕ
  iodine : ℕ

/-- Atomic weights of elements -/
def atomic_weight : Fin 3 → ℝ
| 0 => 14.01  -- Nitrogen
| 1 => 1.01   -- Hydrogen
| 2 => 126.90 -- Iodine

/-- Calculate the molecular weight of a compound -/
def molecular_weight (c : Compound) : ℝ :=
  c.nitrogen * atomic_weight 0 + c.hydrogen * atomic_weight 1 + c.iodine * atomic_weight 2

/-- The compound in question -/
def target_compound : Compound := { nitrogen := 1, hydrogen := 4, iodine := 1 }

/-- Theorem stating that the target compound has exactly one iodine atom -/
theorem target_compound_has_one_iodine :
  molecular_weight target_compound = 145 ∧ target_compound.iodine = 1 := by
  sorry

end NUMINAMATH_CALUDE_target_compound_has_one_iodine_l2132_213237


namespace NUMINAMATH_CALUDE_notebooks_divisible_by_three_l2132_213256

/-- A family is preparing backpacks with school supplies. -/
structure SchoolSupplies where
  pencils : ℕ
  notebooks : ℕ
  backpacks : ℕ

/-- The conditions of the problem -/
def problem_conditions (s : SchoolSupplies) : Prop :=
  s.pencils = 9 ∧
  s.backpacks = 3 ∧
  s.pencils % s.backpacks = 0 ∧
  s.notebooks % s.backpacks = 0

/-- Theorem stating that the number of notebooks must be divisible by 3 -/
theorem notebooks_divisible_by_three (s : SchoolSupplies) 
  (h : problem_conditions s) : 
  s.notebooks % 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_notebooks_divisible_by_three_l2132_213256


namespace NUMINAMATH_CALUDE_quadratic_form_equivalence_l2132_213264

theorem quadratic_form_equivalence :
  ∀ x y : ℝ, y = x^2 - 4*x + 5 ↔ y = (x - 2)^2 + 1 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_equivalence_l2132_213264
