import Mathlib

namespace steven_peach_apple_difference_l1042_104203

-- Define the number of peaches and apples Steven has
def steven_peaches : ℕ := 18
def steven_apples : ℕ := 11

-- Theorem to prove the difference between peaches and apples
theorem steven_peach_apple_difference :
  steven_peaches - steven_apples = 7 := by
  sorry

end steven_peach_apple_difference_l1042_104203


namespace sqrt_sum_inequality_l1042_104296

theorem sqrt_sum_inequality : Real.sqrt 2 + Real.sqrt 3 ≠ Real.sqrt 5 := by
  sorry

end sqrt_sum_inequality_l1042_104296


namespace crypto_puzzle_l1042_104268

theorem crypto_puzzle (A B C D : Nat) : 
  A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧  -- Digits are 0-9
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧  -- Unique digits
  A + B + C = D ∧
  B + C = 7 ∧
  A - B = 1 →
  D = 9 := by
sorry

end crypto_puzzle_l1042_104268


namespace subtract_negative_negative_two_minus_five_l1042_104289

theorem subtract_negative (a b : ℤ) : a - b = a + (-b) := by sorry

theorem negative_two_minus_five : (-2 : ℤ) - 5 = -7 := by sorry

end subtract_negative_negative_two_minus_five_l1042_104289


namespace probability_at_least_two_green_is_one_third_l1042_104246

/-- The probability of selecting at least two green apples when randomly choosing
    3 apples from a set of 10 apples, where 4 are green and 6 are red. -/
def probability_at_least_two_green : ℚ :=
  let total_apples : ℕ := 10
  let green_apples : ℕ := 4
  let red_apples : ℕ := 6
  let chosen_apples : ℕ := 3
  let total_ways := Nat.choose total_apples chosen_apples
  let ways_two_green := Nat.choose green_apples 2 * Nat.choose red_apples 1
  let ways_three_green := Nat.choose green_apples 3
  (ways_two_green + ways_three_green : ℚ) / total_ways

theorem probability_at_least_two_green_is_one_third :
  probability_at_least_two_green = 1 / 3 := by
  sorry

end probability_at_least_two_green_is_one_third_l1042_104246


namespace remaining_volume_cube_with_hole_l1042_104287

/-- The remaining volume of a cube after drilling a cylindrical hole -/
theorem remaining_volume_cube_with_hole (cube_side : Real) (hole_radius : Real) (hole_height : Real) :
  cube_side = 6 →
  hole_radius = 3 →
  hole_height = 4 →
  cube_side ^ 3 - π * hole_radius ^ 2 * hole_height = 216 - 36 * π := by
  sorry

#check remaining_volume_cube_with_hole

end remaining_volume_cube_with_hole_l1042_104287


namespace triangle_rectangle_perimeter_l1042_104232

theorem triangle_rectangle_perimeter (d : ℕ) : 
  ∀ (t w : ℝ),
  t > 0 ∧ w > 0 →  -- positive sides
  3 * t - (6 * w) = 2016 →  -- perimeter difference
  t = 2 * w + d →  -- side length difference
  d = 672 ∧ ∀ (x : ℕ), x ≠ 672 → x ≠ d :=
by sorry

end triangle_rectangle_perimeter_l1042_104232


namespace EF_is_one_eighth_of_GH_l1042_104221

-- Define the line segment GH and points E and F on it
variable (G H E F : Real)

-- Define the condition that E and F lie on GH
axiom E_on_GH : G ≤ E ∧ E ≤ H
axiom F_on_GH : G ≤ F ∧ F ≤ H

-- Define the length ratios
axiom GE_ratio : E - G = 3 * (H - E)
axiom GF_ratio : F - G = 7 * (H - F)

-- State the theorem to be proved
theorem EF_is_one_eighth_of_GH : (F - E) = (1/8) * (H - G) := by
  sorry

end EF_is_one_eighth_of_GH_l1042_104221


namespace book_length_is_4556_l1042_104299

/-- Represents the properties of a book and a reader's progress. -/
structure BookReading where
  total_hours : Nat
  pages_read : Nat
  speed_increase : Nat
  extra_pages : Nat

/-- Calculates the total number of pages in the book based on the given reading information. -/
def calculate_total_pages (reading : BookReading) : Nat :=
  reading.pages_read + (reading.pages_read - reading.extra_pages)

/-- Theorem stating that given the specific reading conditions, the total number of pages in the book is 4556. -/
theorem book_length_is_4556 (reading : BookReading)
  (h1 : reading.total_hours = 5)
  (h2 : reading.pages_read = 2323)
  (h3 : reading.speed_increase = 10)
  (h4 : reading.extra_pages = 90) :
  calculate_total_pages reading = 4556 := by
  sorry

#eval calculate_total_pages { total_hours := 5, pages_read := 2323, speed_increase := 10, extra_pages := 90 }

end book_length_is_4556_l1042_104299


namespace wayne_shrimp_appetizer_cost_l1042_104237

/-- Calculates the total cost of shrimp for an appetizer given the number of shrimp per guest,
    number of guests, cost per pound of shrimp, and number of shrimp per pound. -/
def shrimp_appetizer_cost (shrimp_per_guest : ℕ) (num_guests : ℕ) (cost_per_pound : ℚ) (shrimp_per_pound : ℕ) : ℚ :=
  (shrimp_per_guest * num_guests : ℚ) / shrimp_per_pound * cost_per_pound

/-- Proves that Wayne's shrimp appetizer will cost $170 given the specified conditions. -/
theorem wayne_shrimp_appetizer_cost :
  shrimp_appetizer_cost 5 40 17 20 = 170 := by
  sorry

end wayne_shrimp_appetizer_cost_l1042_104237


namespace isosceles_triangle_proof_l1042_104218

/-- A triangle is acute-angled if all its angles are less than 90 degrees -/
def IsAcuteAngled (triangle : Set Point) : Prop :=
  sorry

/-- An altitude of a triangle is a line segment from a vertex perpendicular to the opposite side -/
def IsAltitude (segment : Set Point) (triangle : Set Point) : Prop :=
  sorry

/-- The area of a triangle -/
noncomputable def TriangleArea (triangle : Set Point) : ℝ :=
  sorry

/-- A triangle is isosceles if it has at least two equal sides -/
def IsIsosceles (triangle : Set Point) : Prop :=
  sorry

theorem isosceles_triangle_proof (A B C D E : Point) :
  let triangle := {A, B, C}
  IsAcuteAngled triangle →
  IsAltitude {A, D} triangle →
  IsAltitude {B, E} triangle →
  TriangleArea {B, D, E} ≤ TriangleArea {D, E, A} ∧
  TriangleArea {D, E, A} ≤ TriangleArea {E, A, B} ∧
  TriangleArea {E, A, B} ≤ TriangleArea {A, B, D} →
  IsIsosceles triangle :=
by sorry

end isosceles_triangle_proof_l1042_104218


namespace perpendicular_necessary_and_sufficient_l1042_104270

/-- A plane -/
structure Plane where
  dummy : Unit

/-- A line in a plane -/
structure Line (α : Plane) where
  dummy : Unit

/-- Predicate for a line being straight -/
def isStraight (α : Plane) (l : Line α) : Prop :=
  sorry

/-- Predicate for a line being oblique -/
def isOblique (α : Plane) (m : Line α) : Prop :=
  sorry

/-- Predicate for two lines being perpendicular -/
def isPerpendicular (α : Plane) (l m : Line α) : Prop :=
  sorry

/-- Theorem stating that for a straight line l and an oblique line m on plane α,
    l being perpendicular to m is both necessary and sufficient -/
theorem perpendicular_necessary_and_sufficient (α : Plane) (l m : Line α)
    (h1 : isStraight α l) (h2 : isOblique α m) :
    isPerpendicular α l m ↔ True :=
  sorry

end perpendicular_necessary_and_sufficient_l1042_104270


namespace cone_height_ratio_l1042_104234

/-- Proves the ratio of new height to original height for a cone with reduced height -/
theorem cone_height_ratio (base_circumference : ℝ) (original_height : ℝ) (new_volume : ℝ) :
  base_circumference = 20 * Real.pi →
  original_height = 40 →
  new_volume = 400 * Real.pi →
  ∃ (new_height : ℝ),
    (1 / 3) * Real.pi * ((base_circumference / (2 * Real.pi)) ^ 2) * new_height = new_volume ∧
    new_height / original_height = 3 / 10 := by
  sorry


end cone_height_ratio_l1042_104234


namespace square_field_area_l1042_104251

/-- Given a square field where a horse takes 10 hours to run around it at a speed of 16 km/h, 
    the area of the field is 1600 square kilometers. -/
theorem square_field_area (s : ℝ) : 
  s > 0 → -- s is positive (side length of square)
  (4 * s = 16 * 10) → -- perimeter equals distance traveled by horse
  s^2 = 1600 := by sorry

end square_field_area_l1042_104251


namespace largest_solution_logarithm_equation_l1042_104206

theorem largest_solution_logarithm_equation (x : ℝ) : 
  (x > 0) → 
  (∀ y, y > 0 → (Real.log 2 / Real.log (2*y) + Real.log 2 / Real.log (4*y^2) = -1) → x ≥ y) →
  (Real.log 2 / Real.log (2*x) + Real.log 2 / Real.log (4*x^2) = -1) →
  1 / x^12 = 4096 := by
sorry

end largest_solution_logarithm_equation_l1042_104206


namespace complex_inequality_complex_inequality_equality_condition_l1042_104219

theorem complex_inequality (z : ℂ) (h : Complex.abs z ≥ 1) :
  (Complex.abs (2 * z - 1))^5 / (25 * Real.sqrt 5) ≥ (Complex.abs (z - 1))^4 / 4 :=
by sorry

theorem complex_inequality_equality_condition (z : ℂ) :
  (Complex.abs (2 * z - 1))^5 / (25 * Real.sqrt 5) = (Complex.abs (z - 1))^4 / 4 ↔
  z = Complex.I ∨ z = -Complex.I :=
by sorry

end complex_inequality_complex_inequality_equality_condition_l1042_104219


namespace deduce_day_from_statements_l1042_104297

structure Animal where
  name : String
  lying_days : Finset Nat

def day_of_week (d : Nat) : Nat :=
  d % 7

theorem deduce_day_from_statements
  (lion unicorn : Animal)
  (today yesterday : Nat)
  (h_lion_statement : day_of_week yesterday ∈ lion.lying_days)
  (h_unicorn_statement : day_of_week yesterday ∈ unicorn.lying_days)
  (h_common_lying_day : ∃! d, d ∈ lion.lying_days ∧ d ∈ unicorn.lying_days)
  (h_today_yesterday : day_of_week today = (day_of_week yesterday + 1) % 7) :
  ∃ (common_day : Nat),
    day_of_week yesterday = common_day ∧
    common_day ∈ lion.lying_days ∧
    common_day ∈ unicorn.lying_days ∧
    day_of_week today = (common_day + 1) % 7 :=
by sorry

end deduce_day_from_statements_l1042_104297


namespace vacation_expense_sharing_l1042_104222

/-- The vacation expense sharing problem -/
theorem vacation_expense_sharing 
  (alex kim lee nina : ℝ)
  (h_alex : alex = 130)
  (h_kim : kim = 150)
  (h_lee : lee = 170)
  (h_nina : nina = 200)
  (h_total : alex + kim + lee + nina = 650)
  (h_equal_share : (alex + kim + lee + nina) / 4 = 162.5)
  (a k : ℝ)
  (h_a : a = 162.5 - alex)
  (h_k : k = 162.5 - kim) :
  a - k = 20 := by
sorry

end vacation_expense_sharing_l1042_104222


namespace triangle_side_expression_zero_l1042_104236

theorem triangle_side_expression_zero (a b c : ℝ) 
  (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) : 
  |a - b - c| - |c - a + b| = 0 := by
  sorry

end triangle_side_expression_zero_l1042_104236


namespace other_communities_students_l1042_104277

theorem other_communities_students (total : ℕ) (muslim_percent hindu_percent sikh_percent christian_percent buddhist_percent : ℚ) :
  total = 1500 →
  muslim_percent = 38/100 →
  hindu_percent = 26/100 →
  sikh_percent = 12/100 →
  christian_percent = 6/100 →
  buddhist_percent = 4/100 →
  ↑(total * (1 - (muslim_percent + hindu_percent + sikh_percent + christian_percent + buddhist_percent))) = 210 :=
by
  sorry

end other_communities_students_l1042_104277


namespace root_equation_q_value_l1042_104286

theorem root_equation_q_value (a b m p q : ℝ) : 
  (a^2 - m*a + (3/2) = 0) →
  (b^2 - m*b + (3/2) = 0) →
  ((a + 1/b)^2 - p*(a + 1/b) + q = 0) →
  ((b + 1/a)^2 - p*(b + 1/a) + q = 0) →
  q = 19/6 := by
sorry

end root_equation_q_value_l1042_104286


namespace pictures_per_album_l1042_104212

theorem pictures_per_album 
  (phone_pics : ℕ) 
  (camera_pics : ℕ) 
  (num_albums : ℕ) 
  (h1 : phone_pics = 35) 
  (h2 : camera_pics = 5) 
  (h3 : num_albums = 5) : 
  (phone_pics + camera_pics) / num_albums = 8 := by
  sorry

end pictures_per_album_l1042_104212


namespace value_of_a_l1042_104267

theorem value_of_a (a : ℝ) (h : a = -a) : a = 0 := by
  sorry

end value_of_a_l1042_104267


namespace store_dvds_count_l1042_104275

def total_dvds : ℕ := 10
def online_dvds : ℕ := 2

theorem store_dvds_count : total_dvds - online_dvds = 8 := by
  sorry

end store_dvds_count_l1042_104275


namespace cereal_box_servings_l1042_104243

theorem cereal_box_servings (total_cereal : ℝ) (serving_size : ℝ) (h1 : total_cereal = 24.5) (h2 : serving_size = 1.75) : 
  ⌊total_cereal / serving_size⌋ = 14 := by
sorry

end cereal_box_servings_l1042_104243


namespace ice_cream_volume_l1042_104230

/-- The volume of ice cream in a cone with hemisphere and cylindrical topping -/
theorem ice_cream_volume (h_cone r_cone h_cylinder : ℝ) 
  (h_cone_pos : 0 < h_cone)
  (r_cone_pos : 0 < r_cone)
  (h_cylinder_pos : 0 < h_cylinder)
  (h_cone_val : h_cone = 12)
  (r_cone_val : r_cone = 3)
  (h_cylinder_val : h_cylinder = 2) :
  (1/3 * π * r_cone^2 * h_cone) +  -- Volume of cone
  (2/3 * π * r_cone^3) +           -- Volume of hemisphere
  (π * r_cone^2 * h_cylinder) =    -- Volume of cylinder
  72 * π := by
sorry


end ice_cream_volume_l1042_104230


namespace hall_ratio_l1042_104288

/-- Given a rectangular hall with area 578 sq. m and difference between length and width 17 m,
    prove that the ratio of width to length is 1:2 -/
theorem hall_ratio (w l : ℝ) (hw : w > 0) (hl : l > 0) : 
  w * l = 578 → l - w = 17 → w / l = 1 / 2 := by
  sorry

end hall_ratio_l1042_104288


namespace f_composition_value_l1042_104272

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.sin (5 * Real.pi * x / 2)
  else 1/6 - Real.log x / Real.log 3

theorem f_composition_value : f (f (3 * Real.sqrt 3)) = Real.sqrt 3 / 2 := by
  sorry

end f_composition_value_l1042_104272


namespace no_roots_in_larger_interval_l1042_104252

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Define the property of having exactly one root
def has_unique_root (f : ℝ → ℝ) : Prop :=
  ∃! x, f x = 0

-- Define the property of a root being within an open interval
def root_in_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x, a < x ∧ x < b ∧ f x = 0

-- Theorem statement
theorem no_roots_in_larger_interval
  (h_unique : has_unique_root f)
  (h_16 : root_in_interval f 0 16)
  (h_8 : root_in_interval f 0 8)
  (h_4 : root_in_interval f 0 4)
  (h_2 : root_in_interval f 0 2) :
  ∀ x ∈ Set.Icc 2 16, f x ≠ 0 :=
sorry

end no_roots_in_larger_interval_l1042_104252


namespace model_y_completion_time_l1042_104280

/-- The time (in minutes) it takes for a Model Y computer to complete the task -/
def model_y_time : ℝ := 30

/-- The time (in minutes) it takes for a Model X computer to complete the task -/
def model_x_time : ℝ := 60

/-- The number of Model X computers used -/
def num_model_x : ℝ := 20

/-- The time (in minutes) it takes for both models working together to complete the task -/
def total_time : ℝ := 1

theorem model_y_completion_time :
  (num_model_x / model_x_time + num_model_x / model_y_time) * total_time = 1 :=
by sorry

end model_y_completion_time_l1042_104280


namespace john_total_spend_l1042_104241

def calculate_total_spend (tshirt_price : ℝ) (tshirt_count : ℕ) (pants_price : ℝ) (pants_count : ℕ)
  (jacket_price : ℝ) (jacket_discount : ℝ) (hat_price : ℝ) (shoes_price : ℝ) (shoes_discount : ℝ)
  (clothes_tax_rate : ℝ) (shoes_tax_rate : ℝ) : ℝ :=
  let tshirt_total := tshirt_price * 2 + tshirt_price * 0.5
  let pants_total := pants_price * pants_count
  let jacket_total := jacket_price * (1 - jacket_discount)
  let shoes_total := shoes_price * (1 - shoes_discount)
  let clothes_subtotal := tshirt_total + pants_total + jacket_total + hat_price
  let total_before_tax := clothes_subtotal + shoes_total
  let clothes_tax := clothes_subtotal * clothes_tax_rate
  let shoes_tax := shoes_total * shoes_tax_rate
  total_before_tax + clothes_tax + shoes_tax

theorem john_total_spend :
  calculate_total_spend 20 3 50 2 80 0.25 15 60 0.1 0.05 0.08 = 294.57 := by
  sorry

end john_total_spend_l1042_104241


namespace polynomial_simplification_l1042_104282

theorem polynomial_simplification (x : ℝ) : 
  (7 * x^12 + 2 * x^10 + x^9) + (3 * x^11 + x^10 + 6 * x^9 + 5 * x^7) + (x^12 + 4 * x^10 + 2 * x^9 + x^3) = 
  8 * x^12 + 3 * x^11 + 7 * x^10 + 9 * x^9 + 5 * x^7 + x^3 :=
by sorry

end polynomial_simplification_l1042_104282


namespace rational_function_characterization_l1042_104207

theorem rational_function_characterization (f : ℚ → ℚ) 
  (h1 : f 1 = 2)
  (h2 : ∀ x y : ℚ, f (x * y) = f x * f y - f (x + y) + 1) :
  ∀ x : ℚ, f x = x + 1 := by sorry

end rational_function_characterization_l1042_104207


namespace deck_size_proof_l1042_104231

theorem deck_size_proof (r b : ℕ) : 
  (r : ℚ) / (r + b) = 1/4 →
  (r : ℚ) / (r + b + 6) = 1/5 →
  r + b = 24 := by
sorry

end deck_size_proof_l1042_104231


namespace cubic_geometric_roots_property_l1042_104283

/-- A cubic equation with coefficients a, b, c has three nonzero real roots in geometric progression -/
structure CubicWithGeometricRoots (a b c : ℝ) : Prop where
  roots_exist : ∃ (d q : ℝ), d ≠ 0 ∧ q ≠ 0 ∧ q ≠ 1
  root_equation : ∀ (d q : ℝ), d ≠ 0 → q ≠ 0 → q ≠ 1 →
    d^3 + a*d^2 + b*d + c = 0 ∧
    (d*q)^3 + a*(d*q)^2 + b*(d*q) + c = 0 ∧
    (d*q^2)^3 + a*(d*q^2)^2 + b*(d*q^2) + c = 0

/-- The main theorem -/
theorem cubic_geometric_roots_property {a b c : ℝ} (h : CubicWithGeometricRoots a b c) :
  a^3 * c - b^3 = 0 := by
  sorry

end cubic_geometric_roots_property_l1042_104283


namespace romeo_chocolate_profit_l1042_104273

/-- Calculates the profit for Romeo's chocolate business -/
theorem romeo_chocolate_profit :
  let total_revenue : ℕ := 340
  let chocolate_cost : ℕ := 175
  let packaging_cost : ℕ := 60
  let advertising_cost : ℕ := 20
  let total_cost : ℕ := chocolate_cost + packaging_cost + advertising_cost
  let profit : ℕ := total_revenue - total_cost
  profit = 85 := by sorry

end romeo_chocolate_profit_l1042_104273


namespace hyperbola_equation_l1042_104233

/-- Given two hyperbolas with the same asymptotes and a specific focus, prove the equation of one hyperbola -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1) →  -- Equation of C₁
  (∀ x y : ℝ, x^2/4 - y^2/16 = 1) →    -- Equation of C₂
  (b/a = 2) →                          -- Same asymptotes condition
  (a^2 + b^2 = 5) →                    -- Right focus condition
  (∀ x y : ℝ, x^2 - y^2/4 = 1) :=      -- Conclusion: Equation of C₁
by sorry


end hyperbola_equation_l1042_104233


namespace sqrt_product_and_difference_of_squares_l1042_104201

theorem sqrt_product_and_difference_of_squares :
  (∀ x y : ℝ, x > 0 → y > 0 → Real.sqrt (x * y) = Real.sqrt x * Real.sqrt y) ∧
  (∀ a b : ℝ, (a + b) * (a - b) = a^2 - b^2) ∧
  (Real.sqrt 3 * Real.sqrt 27 = 9) ∧
  ((Real.sqrt 2 + 1) * (Real.sqrt 2 - 1) - (Real.sqrt 3 - 2)^2 = 4 * Real.sqrt 3 - 6) := by
  sorry

end sqrt_product_and_difference_of_squares_l1042_104201


namespace series_sum_l1042_104290

theorem series_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  let series_term (n : ℕ) := 1 / (((2 * n - 3) * a - (n - 2) * b) * (2 * n * a - (2 * n - 1) * b))
  ∑' n, series_term n = 1 / ((a - b) * b) := by
  sorry

end series_sum_l1042_104290


namespace snow_probability_l1042_104261

theorem snow_probability (p1 p2 : ℝ) (h1 : p1 = 1/4) (h2 : p2 = 1/3) :
  let prob_no_snow := (1 - p1)^4 * (1 - p2)^3
  1 - prob_no_snow = 29/32 := by
  sorry

end snow_probability_l1042_104261


namespace sphere_volume_from_inscribed_box_l1042_104276

/-- The volume of a sphere given an inscribed rectangular box --/
theorem sphere_volume_from_inscribed_box (AB BC AA₁ : ℝ) (h1 : AB = 2) (h2 : BC = 2) (h3 : AA₁ = 2 * Real.sqrt 2) :
  let box_diagonal := Real.sqrt (AB^2 + BC^2 + AA₁^2)
  let sphere_radius := box_diagonal / 2
  let sphere_volume := (4 / 3) * Real.pi * sphere_radius^3
  sphere_volume = (32 * Real.pi) / 3 := by
  sorry

end sphere_volume_from_inscribed_box_l1042_104276


namespace cell_growth_10_days_l1042_104205

/-- Calculates the number of cells after a given number of days, 
    starting with an initial population that doubles every two days. -/
def cell_population (initial_cells : ℕ) (days : ℕ) : ℕ :=
  initial_cells * 2^(days / 2)

/-- Theorem stating that given 4 initial cells that double every two days for 10 days, 
    the final number of cells is 64. -/
theorem cell_growth_10_days : cell_population 4 10 = 64 := by
  sorry

end cell_growth_10_days_l1042_104205


namespace train_speed_l1042_104264

/-- Proves that a train of given length crossing a bridge of given length in a given time has a specific speed in km/hr -/
theorem train_speed (train_length bridge_length : Real) (crossing_time : Real) : 
  train_length = 150 ∧ 
  bridge_length = 225 ∧ 
  crossing_time = 30 → 
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

end train_speed_l1042_104264


namespace subway_passenger_decrease_l1042_104210

theorem subway_passenger_decrease (initial : ℕ) (got_off : ℕ) (got_on : ℕ)
  (h1 : initial = 35)
  (h2 : got_off = 18)
  (h3 : got_on = 15) :
  initial - (initial - got_off + got_on) = 3 :=
by sorry

end subway_passenger_decrease_l1042_104210


namespace y_value_proof_l1042_104291

theorem y_value_proof (y : ℝ) (h : (40 : ℝ) / 80 = Real.sqrt (y / 80)) : y = 20 := by
  sorry

end y_value_proof_l1042_104291


namespace power_mod_remainder_l1042_104242

theorem power_mod_remainder : 6^50 % 215 = 36 := by sorry

end power_mod_remainder_l1042_104242


namespace good_couples_parity_l1042_104238

/-- Represents the color of a grid on the chess board -/
inductive Color
| Red
| Blue

/-- Converts a Color to an integer label -/
def color_to_label (c : Color) : Int :=
  match c with
  | Color.Red => 1
  | Color.Blue => -1

/-- Represents a chess board with m rows and n columns -/
structure ChessBoard (m n : Nat) where
  grid : Fin m → Fin n → Color
  m_ge_three : m ≥ 3
  n_ge_three : n ≥ 3

/-- Counts the number of "good couples" on the chess board -/
def count_good_couples (board : ChessBoard m n) : Nat :=
  sorry

/-- Calculates the product of labels for border grids (excluding corners) -/
def border_product (board : ChessBoard m n) : Int :=
  sorry

/-- Main theorem: The parity of good couples is determined by the border product -/
theorem good_couples_parity (m n : Nat) (board : ChessBoard m n) :
  Even (count_good_couples board) ↔ border_product board = 1 :=
  sorry

end good_couples_parity_l1042_104238


namespace min_value_of_function_l1042_104269

theorem min_value_of_function (x : ℝ) (h1 : 0 < x) (h2 : x < 1/3) :
  (3 / (2 * x) + 2 / (1 - 3 * x)) ≥ 25/2 :=
by sorry

end min_value_of_function_l1042_104269


namespace thousand_pow_ten_zeros_l1042_104215

/-- The number of trailing zeros in a natural number -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- 1000 is equal to 10^3 -/
axiom thousand_eq_ten_cubed : (1000 : ℕ) = 10^3

/-- The number of trailing zeros in 1000^10 is 30 -/
theorem thousand_pow_ten_zeros : trailingZeros (1000^10) = 30 := by sorry

end thousand_pow_ten_zeros_l1042_104215


namespace no_valid_m_l1042_104224

theorem no_valid_m : ¬ ∃ (m : ℕ+), (∃ (a b : ℕ+), (1806 : ℤ) = a * (m.val ^ 2 - 2) ∧ (1806 : ℤ) = b * (m.val ^ 2 + 2)) := by
  sorry

end no_valid_m_l1042_104224


namespace storks_on_fence_l1042_104248

/-- The number of storks that joined the birds on the fence -/
def storks_joined : ℕ := 6

/-- The initial number of birds on the fence -/
def initial_birds : ℕ := 3

/-- The number of additional birds that joined -/
def additional_birds : ℕ := 2

theorem storks_on_fence :
  storks_joined = initial_birds + additional_birds + 1 :=
by sorry

end storks_on_fence_l1042_104248


namespace julian_frederick_age_difference_l1042_104262

/-- Given the ages of Kyle, Julian, Frederick, and Tyson, prove that Julian is 20 years younger than Frederick. -/
theorem julian_frederick_age_difference :
  ∀ (kyle_age julian_age frederick_age tyson_age : ℕ),
  kyle_age = julian_age + 5 →
  frederick_age > julian_age →
  frederick_age = 2 * tyson_age →
  tyson_age = 20 →
  kyle_age = 25 →
  frederick_age - julian_age = 20 :=
by sorry

end julian_frederick_age_difference_l1042_104262


namespace quadratic_equations_solutions_l1042_104278

theorem quadratic_equations_solutions :
  (∃ x1 x2 : ℝ, (x1^2 - 4*x1 = 5 ∧ x2^2 - 4*x2 = 5) ∧ (x1 = 5 ∧ x2 = -1)) ∧
  (∃ y1 y2 : ℝ, (y1^2 + 7*y1 - 18 = 0 ∧ y2^2 + 7*y2 - 18 = 0) ∧ (y1 = -9 ∧ y2 = 2)) :=
by sorry

end quadratic_equations_solutions_l1042_104278


namespace g_difference_l1042_104244

theorem g_difference (x h : ℝ) : 
  let g := λ (t : ℝ) => 3 * t^3 - 4 * t + 5
  g (x + h) - g x = h * (9 * x^2 + 9 * x * h + 3 * h^2 - 4) := by
  sorry

end g_difference_l1042_104244


namespace number_of_lineups_is_4290_l1042_104298

/-- Represents the total number of players in the team -/
def total_players : ℕ := 15

/-- Represents the number of players in a starting lineup -/
def lineup_size : ℕ := 6

/-- Represents the number of players who refuse to play together -/
def incompatible_players : ℕ := 2

/-- Calculates the number of possible starting lineups -/
def number_of_lineups : ℕ :=
  let remaining_players := total_players - incompatible_players
  Nat.choose remaining_players (lineup_size - 1) * 2 +
  Nat.choose remaining_players lineup_size

/-- Theorem stating that the number of possible lineups is 4290 -/
theorem number_of_lineups_is_4290 :
  number_of_lineups = 4290 := by sorry

end number_of_lineups_is_4290_l1042_104298


namespace value_of_y_l1042_104295

theorem value_of_y (x y : ℝ) (h1 : x - y = 16) (h2 : x + y = 4) : y = -6 := by
  sorry

end value_of_y_l1042_104295


namespace leaves_blown_away_l1042_104260

theorem leaves_blown_away (initial_leaves left_leaves : ℕ) 
  (h1 : initial_leaves = 5678)
  (h2 : left_leaves = 1432) :
  initial_leaves - left_leaves = 4246 := by
  sorry

end leaves_blown_away_l1042_104260


namespace batsman_average_after_12th_innings_l1042_104249

/-- Calculates the new average of a batsman after 12 innings -/
def new_average (previous_total : ℕ) (new_score : ℕ) : ℚ :=
  (previous_total + new_score) / 12

/-- Represents the increase in average after the 12th innings -/
def average_increase (previous_average : ℚ) (new_average : ℚ) : ℚ :=
  new_average - previous_average

theorem batsman_average_after_12th_innings 
  (previous_total : ℕ) 
  (previous_average : ℚ) 
  (new_score : ℕ) :
  previous_total = previous_average * 11 →
  new_score = 115 →
  average_increase previous_average (new_average previous_total new_score) = 3 →
  new_average previous_total new_score = 82 :=
by sorry

end batsman_average_after_12th_innings_l1042_104249


namespace base7_subtraction_theorem_l1042_104213

/-- Represents a number in base 7 --/
def Base7 : Type := ℕ

/-- Converts a base 7 number to its decimal representation --/
def to_decimal (n : Base7) : ℕ := sorry

/-- Converts a decimal number to its base 7 representation --/
def from_decimal (n : ℕ) : Base7 := sorry

/-- Subtracts two base 7 numbers --/
def base7_subtract (a b : Base7) : Base7 := sorry

theorem base7_subtraction_theorem :
  base7_subtract (from_decimal 4321) (from_decimal 1234) = from_decimal 3054 := by
  sorry

end base7_subtraction_theorem_l1042_104213


namespace room_width_proof_l1042_104247

/-- Given a rectangular room with known length, paving cost, and paving rate per square meter,
    prove that the width of the room is 2.75 meters. -/
theorem room_width_proof (length : ℝ) (paving_cost : ℝ) (paving_rate : ℝ) :
  length = 6.5 →
  paving_cost = 10725 →
  paving_rate = 600 →
  paving_cost / paving_rate / length = 2.75 := by
  sorry


end room_width_proof_l1042_104247


namespace expression_evaluation_l1042_104254

theorem expression_evaluation (a : ℝ) (h : a = 3) : 
  (3 * a⁻¹ + (2 * a⁻¹) / 3) / (2 * a) = 11 / 54 := by
  sorry

end expression_evaluation_l1042_104254


namespace monotonic_increasing_iff_m_ge_four_thirds_l1042_104208

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^3 + 2*x^2 + m*x + 1

-- State the theorem
theorem monotonic_increasing_iff_m_ge_four_thirds (m : ℝ) :
  (∀ x : ℝ, Monotone (f m)) ↔ m ≥ 4/3 := by sorry

end monotonic_increasing_iff_m_ge_four_thirds_l1042_104208


namespace geometric_series_first_term_l1042_104245

theorem geometric_series_first_term (a r : ℝ) (h1 : r ≠ 1) (h2 : |r| < 1) : 
  (a / (1 - r) = 12) → ((a^2) / (1 - r^2) = 36) → a = 4.8 := by
  sorry

end geometric_series_first_term_l1042_104245


namespace students_registered_l1042_104253

theorem students_registered (students_yesterday : ℕ) (students_today : ℕ) : ℕ :=
  let students_registered := 156
  let students_absent := 30
  have h1 : students_today + students_absent = students_registered := by sorry
  have h2 : students_today = (2 * students_yesterday * 9) / 10 := by sorry
  students_registered

#check students_registered

end students_registered_l1042_104253


namespace regular_polygon_sides_l1042_104266

theorem regular_polygon_sides (n : ℕ) (exterior_angle : ℝ) : 
  n > 0 ∧ exterior_angle = 15 → n * exterior_angle = 360 → n = 24 :=
by sorry

end regular_polygon_sides_l1042_104266


namespace increase_in_average_commission_l1042_104214

/-- Calculates the increase in average commission after a big sale -/
theorem increase_in_average_commission 
  (big_sale_commission : ℕ) 
  (new_average_commission : ℕ) 
  (total_sales : ℕ) 
  (h1 : big_sale_commission = 1000)
  (h2 : new_average_commission = 250)
  (h3 : total_sales = 6) :
  new_average_commission - (new_average_commission * total_sales - big_sale_commission) / (total_sales - 1) = 150 := by
  sorry

end increase_in_average_commission_l1042_104214


namespace g_is_even_l1042_104216

-- Define a function f from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define g as f(x) + f(-x)
def g (x : ℝ) : ℝ := f x + f (-x)

-- Theorem: g is an even function
theorem g_is_even : ∀ x : ℝ, g f x = g f (-x) := by sorry

end g_is_even_l1042_104216


namespace ellipse_focus_k_value_l1042_104239

/-- Given an ellipse 3kx^2 + y^2 = 1 with focus F(2,0), prove that k = 1/15 -/
theorem ellipse_focus_k_value (k : ℝ) : 
  (∃ (x y : ℝ), 3 * k * x^2 + y^2 = 1) →  -- Ellipse equation
  (2 : ℝ)^2 = (1 / (3 * k)) - 1 →  -- Focus condition (c^2 = a^2 - b^2)
  k = 1 / 15 := by
sorry

end ellipse_focus_k_value_l1042_104239


namespace hyperbola_eccentricity_l1042_104226

theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let ellipse_eccentricity := Real.sqrt (1 - (b / a) ^ 2)
  let hyperbola_eccentricity := Real.sqrt (1 + (b / a) ^ 2)
  ellipse_eccentricity = Real.sqrt 3 / 2 →
  hyperbola_eccentricity = Real.sqrt 5 / 2 :=
by sorry

end hyperbola_eccentricity_l1042_104226


namespace tennis_racket_packaging_l1042_104202

/-- Given information about tennis racket packaging, prove the number of rackets in the other carton type. -/
theorem tennis_racket_packaging (total_cartons : ℕ) (total_rackets : ℕ) (three_racket_cartons : ℕ) 
  (h1 : total_cartons = 38)
  (h2 : total_rackets = 100)
  (h3 : three_racket_cartons = 24)
  : ∃ (other_carton_size : ℕ), 
    other_carton_size * (total_cartons - three_racket_cartons) + 3 * three_racket_cartons = total_rackets ∧ 
    other_carton_size = 2 :=
by sorry

end tennis_racket_packaging_l1042_104202


namespace trigonometric_identity_l1042_104274

theorem trigonometric_identity (c : ℝ) (h : c = 2 * Real.pi / 9) :
  (Real.sin (2 * c) * Real.sin (5 * c) * Real.sin (8 * c) * Real.sin (11 * c) * Real.sin (14 * c)) /
  (Real.sin c * Real.sin (3 * c) * Real.sin (4 * c) * Real.sin (7 * c) * Real.sin (8 * c)) =
  Real.sin (80 * Real.pi / 180) :=
by sorry

end trigonometric_identity_l1042_104274


namespace arithmetic_progression_1980_l1042_104292

/-- An arithmetic progression of natural numbers. -/
structure ArithProgression where
  first : ℕ
  diff : ℕ

/-- Check if a natural number belongs to an arithmetic progression. -/
def belongsTo (n : ℕ) (ap : ArithProgression) : Prop :=
  ∃ k : ℕ, n = ap.first + k * ap.diff

/-- The main theorem statement. -/
theorem arithmetic_progression_1980 (P₁ P₂ P₃ : ArithProgression) :
  (∀ n : ℕ, n ≤ 8 → belongsTo n P₁ ∨ belongsTo n P₂ ∨ belongsTo n P₃) →
  belongsTo 1980 P₁ ∨ belongsTo 1980 P₂ ∨ belongsTo 1980 P₃ := by
  sorry

end arithmetic_progression_1980_l1042_104292


namespace union_of_M_and_N_l1042_104209

-- Define sets M and N
def M : Set ℝ := {x | -3 < x ∧ x < 1}
def N : Set ℝ := {x | x ≤ -3}

-- State the theorem
theorem union_of_M_and_N : M ∪ N = {x : ℝ | x < 1} := by sorry

end union_of_M_and_N_l1042_104209


namespace complex_fraction_evaluation_l1042_104281

theorem complex_fraction_evaluation (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a^4 + a^2 * b^2 + b^4 = 0) : 
  (a^15 + b^15) / (a + b)^15 = -2 := by
  sorry

end complex_fraction_evaluation_l1042_104281


namespace augmented_matrix_solution_l1042_104265

theorem augmented_matrix_solution (c₁ c₂ : ℝ) : 
  (∃ (x y : ℝ), 2 * x + 3 * y = c₁ ∧ y = c₂ ∧ x = 3 ∧ y = 5) → c₁ - c₂ = 16 := by
  sorry

end augmented_matrix_solution_l1042_104265


namespace box_dimensions_l1042_104240

/-- A box with a square base and specific ribbon tying properties has dimensions 22 cm × 22 cm × 11 cm -/
theorem box_dimensions (s : ℝ) (b : ℝ) :
  s > 0 →
  6 * s + b = 156 →
  7 * s + b = 178 →
  s = 22 ∧ s / 2 = 11 :=
by sorry

end box_dimensions_l1042_104240


namespace max_regions_formula_l1042_104256

/-- The maximum number of regions formed by n lines in a plane -/
def max_regions (n : ℕ) : ℚ :=
  (n^2 + n + 2) / 2

/-- Conditions for the lines in the plane -/
structure PlaneLines where
  n : ℕ
  n_ge_3 : n ≥ 3
  no_parallel : True  -- represents the condition that no two lines are parallel
  no_triple_intersection : True  -- represents the condition that no three lines intersect at the same point

theorem max_regions_formula (p : PlaneLines) :
  max_regions p.n = (p.n^2 + p.n + 2) / 2 :=
sorry

end max_regions_formula_l1042_104256


namespace sin_double_angle_with_tan_two_l1042_104220

theorem sin_double_angle_with_tan_two (θ : Real) (h : Real.tan θ = 2) : 
  Real.sin (2 * θ) = 4 / 5 := by
  sorry

end sin_double_angle_with_tan_two_l1042_104220


namespace square_field_diagonal_l1042_104284

theorem square_field_diagonal (area : ℝ) (diagonal : ℝ) : 
  area = 128 → diagonal = 16 := by
  sorry

end square_field_diagonal_l1042_104284


namespace cylinder_height_in_hemisphere_l1042_104255

theorem cylinder_height_in_hemisphere (r_cylinder r_hemisphere : ℝ) (h_cylinder_radius : r_cylinder = 3)
  (h_hemisphere_radius : r_hemisphere = 7) (h_inscribed : r_cylinder ≤ r_hemisphere) :
  let height := Real.sqrt (r_hemisphere^2 - r_cylinder^2)
  height = 2 * Real.sqrt 10 :=
by sorry

end cylinder_height_in_hemisphere_l1042_104255


namespace seating_arrangements_eq_144_l1042_104250

/-- The number of ways to arrange n distinct objects. -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- The number of ways to choose k objects from n objects. -/
def choose (n k : ℕ) : ℕ := 
  if k > n then 0
  else factorial n / (factorial k * factorial (n - k))

/-- The number of seating arrangements for 4 students and 2 teachers,
    where teachers cannot sit at either end and must not sit next to each other. -/
def seating_arrangements : ℕ := 
  let student_arrangements := factorial 4
  let teacher_positions := choose 3 2
  let teacher_arrangements := factorial 2
  student_arrangements * teacher_positions * teacher_arrangements

theorem seating_arrangements_eq_144 : seating_arrangements = 144 := by
  sorry

#eval seating_arrangements

end seating_arrangements_eq_144_l1042_104250


namespace four_digit_number_proof_l1042_104279

theorem four_digit_number_proof :
  ∀ (a b : ℕ),
    (2^a * 9^b ≥ 1000) ∧ 
    (2^a * 9^b < 10000) ∧
    (2^a * 9^b = 2000 + 100*a + 90 + b) →
    a = 5 ∧ b = 2 := by
  sorry

end four_digit_number_proof_l1042_104279


namespace union_of_M_and_N_l1042_104271

def M : Set ℝ := {x : ℝ | x^2 - x - 12 = 0}
def N : Set ℝ := {x : ℝ | x^2 + 3*x = 0}

theorem union_of_M_and_N : M ∪ N = {-3, 0, 4} := by
  sorry

end union_of_M_and_N_l1042_104271


namespace tip_difference_calculation_l1042_104293

/-- Calculates the difference in euro cents between a good tip and a bad tip -/
def tip_difference (initial_bill : ℝ) (bad_tip_percent : ℝ) (good_tip_percent : ℝ) 
  (discount_percent : ℝ) (tax_percent : ℝ) (usd_to_eur : ℝ) : ℝ :=
  let discounted_bill := initial_bill * (1 - discount_percent)
  let final_bill := discounted_bill * (1 + tax_percent)
  let bad_tip := final_bill * bad_tip_percent
  let good_tip := final_bill * good_tip_percent
  let difference_usd := good_tip - bad_tip
  let difference_eur := difference_usd * usd_to_eur
  difference_eur * 100  -- Convert to cents

theorem tip_difference_calculation :
  tip_difference 26 0.05 0.20 0.08 0.07 0.85 = 326.33 := by
  sorry

end tip_difference_calculation_l1042_104293


namespace distance_between_vertices_l1042_104217

-- Define the hyperbola equation
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

-- Define the vertices of the hyperbola
def vertices : Set (ℝ × ℝ) := {(-4, 0), (4, 0)}

-- Theorem: The distance between the vertices of the hyperbola is 8
theorem distance_between_vertices :
  ∀ (v1 v2 : ℝ × ℝ), v1 ∈ vertices → v2 ∈ vertices → v1 ≠ v2 →
  Real.sqrt ((v1.1 - v2.1)^2 + (v1.2 - v2.2)^2) = 8 :=
sorry

end distance_between_vertices_l1042_104217


namespace samples_are_stratified_l1042_104294

/-- Represents a sample of 10 student numbers -/
structure Sample :=
  (numbers : List Nat)
  (h_size : numbers.length = 10)
  (h_range : ∀ n ∈ numbers, 1 ≤ n ∧ n ≤ 270)

/-- Represents the distribution of students across grades -/
structure SchoolDistribution :=
  (total : Nat)
  (first_grade : Nat)
  (second_grade : Nat)
  (third_grade : Nat)
  (h_total : total = first_grade + second_grade + third_grade)

/-- Checks if a sample can represent stratified sampling for a given school distribution -/
def is_stratified_sampling (s : Sample) (sd : SchoolDistribution) : Prop :=
  ∃ (n1 n2 n3 : Nat),
    n1 + n2 + n3 = 10 ∧
    n1 ≤ sd.first_grade ∧
    n2 ≤ sd.second_grade ∧
    n3 ≤ sd.third_grade ∧
    (∀ n ∈ s.numbers, 
      (n ≤ sd.first_grade) ∨ 
      (sd.first_grade < n ∧ n ≤ sd.first_grade + sd.second_grade) ∨
      (sd.first_grade + sd.second_grade < n))

def sample1 : Sample := {
  numbers := [7, 34, 61, 88, 115, 142, 169, 196, 223, 250],
  h_size := by rfl,
  h_range := sorry
}

def sample3 : Sample := {
  numbers := [11, 38, 65, 92, 119, 146, 173, 200, 227, 254],
  h_size := by rfl,
  h_range := sorry
}

def school : SchoolDistribution := {
  total := 270,
  first_grade := 108,
  second_grade := 81,
  third_grade := 81,
  h_total := by rfl
}

theorem samples_are_stratified : 
  is_stratified_sampling sample1 school ∧ is_stratified_sampling sample3 school :=
sorry

end samples_are_stratified_l1042_104294


namespace binomial_expansion_properties_l1042_104235

theorem binomial_expansion_properties :
  ∀ (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ),
  (∀ x : ℝ, (2 * x - Real.sqrt 3) ^ 10 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5 + 
                                         a₆ * x^6 + a₇ * x^7 + a₈ * x^8 + a₉ * x^9 + a₁₀ * x^10) →
  (a₀ = 243 ∧ 
   (a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀) * 
   (a₀ - a₁ + a₂ - a₃ + a₄ - a₅ + a₆ - a₇ + a₈ - a₉ + a₁₀) = 1) := by
  sorry

end binomial_expansion_properties_l1042_104235


namespace unique_records_count_l1042_104285

/-- The number of records in either Samantha's or Lily's collection, but not both -/
def unique_records (samantha_total : ℕ) (shared : ℕ) (lily_unique : ℕ) : ℕ :=
  (samantha_total - shared) + lily_unique

/-- Proof that the number of unique records is 18 -/
theorem unique_records_count :
  unique_records 24 15 9 = 18 := by
  sorry

end unique_records_count_l1042_104285


namespace forest_to_street_ratio_l1042_104229

/-- The ratio of forest area to street area is 3:1 -/
theorem forest_to_street_ratio : 
  ∀ (street_side_length : ℝ) (trees_per_sq_meter : ℝ) (total_trees : ℝ),
  street_side_length = 100 →
  trees_per_sq_meter = 4 →
  total_trees = 120000 →
  (total_trees / trees_per_sq_meter) / (street_side_length ^ 2) = 3 := by
sorry

end forest_to_street_ratio_l1042_104229


namespace cyclist_distance_l1042_104211

/-- The distance traveled by a cyclist given the conditions of the problem -/
theorem cyclist_distance (distance_AB : ℝ) (pedestrian_speed : ℝ) 
  (h1 : distance_AB = 5)
  (h2 : pedestrian_speed > 0) : 
  let cyclist_speed := 2 * pedestrian_speed
  let time := distance_AB / pedestrian_speed
  cyclist_speed * time = 10 := by sorry

end cyclist_distance_l1042_104211


namespace monotonicity_of_F_intersection_property_l1042_104200

noncomputable section

variable (x : ℝ) (a : ℝ)

def f (x : ℝ) : ℝ := x * (Real.log x + 1)

def f' (x : ℝ) : ℝ := Real.log x + 2

def F (x : ℝ) (a : ℝ) : ℝ := a * x^2 + f' x

theorem monotonicity_of_F (x : ℝ) (a : ℝ) (h : x > 0) :
  (a ≥ 0 → StrictMono (F · a)) ∧
  (a < 0 → StrictMonoOn (F · a) (Set.Ioo 0 (Real.sqrt (-1 / (2 * a)))) ∧
           StrictAntiOn (F · a) (Set.Ioi (Real.sqrt (-1 / (2 * a))))) :=
sorry

theorem intersection_property (x₁ x₂ k : ℝ) (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₁ < x₂) :
  k = (f' x₂ - f' x₁) / (x₂ - x₁) → x₁ < 1 / k ∧ 1 / k < x₂ :=
sorry

end monotonicity_of_F_intersection_property_l1042_104200


namespace winston_remaining_cents_l1042_104263

/-- The number of cents in a quarter -/
def cents_per_quarter : ℕ := 25

/-- The number of cents in half a dollar -/
def half_dollar_cents : ℕ := 50

/-- The number of quarters Winston has -/
def winston_quarters : ℕ := 14

theorem winston_remaining_cents : 
  winston_quarters * cents_per_quarter - half_dollar_cents = 300 := by
  sorry

end winston_remaining_cents_l1042_104263


namespace smallest_prime_perimeter_scalene_triangle_l1042_104257

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_scalene_triangle (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

def triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ a + c > b

theorem smallest_prime_perimeter_scalene_triangle :
  ∀ a b c : ℕ,
    is_scalene_triangle a b c →
    is_prime a ∧ is_prime b ∧ is_prime c →
    is_prime (a + b + c) →
    a ≥ 5 ∧ b ≥ 5 ∧ c ≥ 5 →
    triangle_inequality a b c →
    (a + b + c ≥ 23) ∧ (∃ x y z : ℕ, x + y + z = 23 ∧ 
      is_scalene_triangle x y z ∧
      is_prime x ∧ is_prime y ∧ is_prime z ∧
      is_prime (x + y + z) ∧
      x ≥ 5 ∧ y ≥ 5 ∧ z ≥ 5 ∧
      triangle_inequality x y z) :=
by sorry

end smallest_prime_perimeter_scalene_triangle_l1042_104257


namespace solution_set_a_2_range_of_a_l1042_104228

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a^2| + |x - 2*a + 1|

-- Part 1: Solution set when a = 2
theorem solution_set_a_2 :
  {x : ℝ | f x 2 ≥ 4} = {x : ℝ | x ≤ 3/2 ∨ x ≥ 11/2} :=
sorry

-- Part 2: Range of a
theorem range_of_a :
  {a : ℝ | ∃ x, f x a ≥ 4} = {a : ℝ | a ≤ -1 ∨ a ≥ 3} :=
sorry

end solution_set_a_2_range_of_a_l1042_104228


namespace ramola_rank_from_last_l1042_104223

theorem ramola_rank_from_last (total_students : ℕ) (rank_from_start : ℕ) :
  total_students = 26 →
  rank_from_start = 14 →
  total_students - rank_from_start + 1 = 14 :=
by sorry

end ramola_rank_from_last_l1042_104223


namespace mean_equality_implies_sum_l1042_104227

theorem mean_equality_implies_sum (y z : ℝ) : 
  (8 + 15 + 21) / 3 = (14 + y + z) / 3 → y + z = 30 := by
  sorry

end mean_equality_implies_sum_l1042_104227


namespace floss_per_student_l1042_104225

/-- Proves that each student needs 5 yards of floss given the problem conditions -/
theorem floss_per_student 
  (num_students : ℕ) 
  (floss_per_packet : ℕ) 
  (leftover_floss : ℕ) 
  (total_floss : ℕ) :
  num_students = 20 →
  floss_per_packet = 35 →
  leftover_floss = 5 →
  total_floss = num_students * (total_floss / num_students) →
  total_floss % floss_per_packet = 0 →
  total_floss / num_students = 5 := by
sorry

end floss_per_student_l1042_104225


namespace club_members_count_l1042_104204

theorem club_members_count : ∃! n : ℕ, 
  200 ≤ n ∧ n ≤ 300 ∧ 
  (n + 4) % 10 = 0 ∧ 
  (n + 5) % 11 = 0 ∧ 
  n = 226 := by
  sorry

end club_members_count_l1042_104204


namespace geometric_sequence_ratio_l1042_104258

/-- Given a geometric sequence with first term a₁ and common ratio q,
    S₃ represents the sum of the first 3 terms -/
def S₃ (a₁ q : ℝ) : ℝ := a₁ + a₁ * q + a₁ * q^2

/-- Theorem: For a geometric sequence with common ratio q,
    if S₃ = 7a₁, then q = 2 or q = -3 -/
theorem geometric_sequence_ratio (a₁ q : ℝ) (h : a₁ ≠ 0) :
  S₃ a₁ q = 7 * a₁ → q = 2 ∨ q = -3 := by
  sorry


end geometric_sequence_ratio_l1042_104258


namespace negation_of_existence_cubic_inequality_negation_l1042_104259

theorem negation_of_existence (P : ℝ → Prop) : 
  (¬ ∃ x, P x) ↔ (∀ x, ¬ P x) :=
by sorry

theorem cubic_inequality_negation : 
  (¬ ∃ x : ℝ, x^3 - x^2 + 1 > 0) ↔ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) :=
by sorry

end negation_of_existence_cubic_inequality_negation_l1042_104259
