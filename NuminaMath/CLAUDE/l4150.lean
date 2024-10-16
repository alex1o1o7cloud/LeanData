import Mathlib

namespace NUMINAMATH_CALUDE_fred_has_ten_balloons_l4150_415096

/-- The number of red balloons Fred has -/
def fred_balloons : ℝ := sorry

/-- The number of red balloons Sam has -/
def sam_balloons : ℝ := 46

/-- The number of red balloons Dan destroyed -/
def destroyed_balloons : ℝ := 16

/-- The total number of remaining red balloons -/
def total_balloons : ℝ := 40

/-- Theorem stating that Fred has 10 red balloons -/
theorem fred_has_ten_balloons : fred_balloons = 10 := by
  sorry

end NUMINAMATH_CALUDE_fred_has_ten_balloons_l4150_415096


namespace NUMINAMATH_CALUDE_oplus_composition_l4150_415093

/-- Definition of the ⊕ operation -/
def oplus (x y : ℝ) : ℝ := x^2 + y

/-- Theorem stating that h ⊕ (h ⊕ h) = 2h^2 + h -/
theorem oplus_composition (h : ℝ) : oplus h (oplus h h) = 2 * h^2 + h := by
  sorry

end NUMINAMATH_CALUDE_oplus_composition_l4150_415093


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l4150_415017

theorem sqrt_product_equality : Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l4150_415017


namespace NUMINAMATH_CALUDE_crayons_lost_or_given_away_l4150_415011

theorem crayons_lost_or_given_away 
  (initial_crayons : ℕ) 
  (crayons_given_away : ℕ) 
  (crayons_lost : ℕ) : 
  crayons_given_away + crayons_lost = 
    (initial_crayons - (initial_crayons - (crayons_given_away + crayons_lost))) :=
by sorry

end NUMINAMATH_CALUDE_crayons_lost_or_given_away_l4150_415011


namespace NUMINAMATH_CALUDE_tamara_height_l4150_415014

/-- Given that Tamara's height is 3 times Kim's height minus 4 inches,
    and their combined height is 92 inches, prove that Tamara is 68 inches tall. -/
theorem tamara_height (kim : ℝ) (tamara : ℝ) : 
  tamara = 3 * kim - 4 → 
  tamara + kim = 92 → 
  tamara = 68 := by
sorry

end NUMINAMATH_CALUDE_tamara_height_l4150_415014


namespace NUMINAMATH_CALUDE_price_reduction_percentage_l4150_415061

theorem price_reduction_percentage (original_price reduction_amount : ℝ) : 
  original_price = 500 → 
  reduction_amount = 250 → 
  (reduction_amount / original_price) * 100 = 50 :=
by sorry

end NUMINAMATH_CALUDE_price_reduction_percentage_l4150_415061


namespace NUMINAMATH_CALUDE_owen_sleep_time_l4150_415054

/-- Owen's daily schedule and sleep time calculation -/
theorem owen_sleep_time :
  let hours_in_day : ℝ := 24
  let work_hours : ℝ := 6
  let commute_hours : ℝ := 2
  let exercise_hours : ℝ := 3
  let cooking_hours : ℝ := 1
  let leisure_hours : ℝ := 3
  let grooming_hours : ℝ := 1.5
  let total_activity_hours := work_hours + commute_hours + exercise_hours + 
                              cooking_hours + leisure_hours + grooming_hours
  let sleep_hours := hours_in_day - total_activity_hours
  sleep_hours = 7.5 := by sorry

end NUMINAMATH_CALUDE_owen_sleep_time_l4150_415054


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l4150_415019

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  focus1 : Point
  focus2 : Point

/-- Checks if an ellipse is tangent to both x and y axes -/
def isTangentToAxes (e : Ellipse) : Prop := sorry

/-- Calculates the length of the major axis of an ellipse -/
def majorAxisLength (e : Ellipse) : ℝ := sorry

/-- Main theorem: The length of the major axis of the given ellipse is 10 -/
theorem ellipse_major_axis_length :
  ∀ (e : Ellipse),
    e.focus1 = ⟨3, -5 + 2 * Real.sqrt 2⟩ ∧
    e.focus2 = ⟨3, -5 - 2 * Real.sqrt 2⟩ ∧
    isTangentToAxes e →
    majorAxisLength e = 10 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l4150_415019


namespace NUMINAMATH_CALUDE_complex_modulus_range_l4150_415085

theorem complex_modulus_range (z k : ℂ) (h1 : Complex.abs z = Complex.abs (1 + k * z)) (h2 : Complex.abs k < 1) :
  1 / (Complex.abs k + 1) ≤ Complex.abs z ∧ Complex.abs z ≤ 1 / (1 - Complex.abs k) :=
by sorry

end NUMINAMATH_CALUDE_complex_modulus_range_l4150_415085


namespace NUMINAMATH_CALUDE_circle_center_is_zero_one_l4150_415053

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the condition of circle passing through a point
def passes_through (c : Circle) (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

-- Define the condition of circle being tangent to parabola at a point
def tangent_to_parabola (c : Circle) (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  y = parabola x ∧ passes_through c p ∧
  ∀ q : ℝ × ℝ, q ≠ p → parabola q.1 = q.2 → ¬passes_through c q

theorem circle_center_is_zero_one :
  ∃ c : Circle,
    passes_through c (0, 2) ∧
    tangent_to_parabola c (1, 1) ∧
    c.center = (0, 1) := by
  sorry

end NUMINAMATH_CALUDE_circle_center_is_zero_one_l4150_415053


namespace NUMINAMATH_CALUDE_geometric_series_sum_l4150_415007

theorem geometric_series_sum (a b : ℝ) (h : ∑' n, a / b^n = 7) :
  ∑' n, a / (a + 2*b)^n = (7*(b-1)) / (9*b-8) := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l4150_415007


namespace NUMINAMATH_CALUDE_equation_represents_three_non_concurrent_lines_l4150_415060

/-- The equation represents three lines that do not all pass through a common point -/
theorem equation_represents_three_non_concurrent_lines :
  ∃ (l₁ l₂ l₃ : ℝ → ℝ → Prop),
    (∀ x y, (x^2 - 3*y)*(x - y + 1) = (y^2 - 3*x)*(x - y + 1) ↔ l₁ x y ∨ l₂ x y ∨ l₃ x y) ∧
    (∃ x₁ y₁, l₁ x₁ y₁ ∧ l₂ x₁ y₁ ∧ ¬l₃ x₁ y₁) ∧
    (∃ x₂ y₂, l₁ x₂ y₂ ∧ ¬l₂ x₂ y₂ ∧ l₃ x₂ y₂) ∧
    (∃ x₃ y₃, ¬l₁ x₃ y₃ ∧ l₂ x₃ y₃ ∧ l₃ x₃ y₃) ∧
    (∀ x y, ¬(l₁ x y ∧ l₂ x y ∧ l₃ x y)) :=
by
  sorry


end NUMINAMATH_CALUDE_equation_represents_three_non_concurrent_lines_l4150_415060


namespace NUMINAMATH_CALUDE_distinct_colorings_l4150_415075

/-- The number of disks in the circle -/
def n : ℕ := 7

/-- The number of blue disks -/
def blue : ℕ := 3

/-- The number of red disks -/
def red : ℕ := 3

/-- The number of green disks -/
def green : ℕ := 1

/-- The total number of colorings without considering symmetries -/
def total_colorings : ℕ := (n.choose blue) * ((n - blue).choose red)

/-- The number of rotational symmetries of the circle -/
def symmetries : ℕ := n

/-- The theorem stating the number of distinct colorings -/
theorem distinct_colorings : 
  (total_colorings / symmetries : ℚ) = 20 := by sorry

end NUMINAMATH_CALUDE_distinct_colorings_l4150_415075


namespace NUMINAMATH_CALUDE_triple_digit_sum_of_2012_pow_2012_l4150_415099

/-- The sum of the digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- The function that applies digit_sum three times -/
def triple_digit_sum (n : ℕ) : ℕ := digit_sum (digit_sum (digit_sum n))

theorem triple_digit_sum_of_2012_pow_2012 :
  triple_digit_sum (2012^2012) = 7 := by sorry

end NUMINAMATH_CALUDE_triple_digit_sum_of_2012_pow_2012_l4150_415099


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l4150_415027

theorem quadratic_roots_condition (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + 2 * x - 3 = 0 ∧ a * y^2 + 2 * y - 3 = 0) → a > -1/3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l4150_415027


namespace NUMINAMATH_CALUDE_cafe_customers_l4150_415039

/-- The number of sandwiches ordered by offices -/
def office_sandwiches : ℕ := 30

/-- The number of sandwiches each ordering customer in the group ordered -/
def sandwiches_per_customer : ℕ := 4

/-- The total number of sandwiches made by the café -/
def total_sandwiches : ℕ := 54

/-- The fraction of the group that ordered sandwiches -/
def ordering_fraction : ℚ := 1/2

theorem cafe_customers : ℕ :=
  let group_sandwiches := total_sandwiches - office_sandwiches
  let ordering_customers := group_sandwiches / sandwiches_per_customer
  let total_customers := ordering_customers / ordering_fraction
  12

#check cafe_customers

end NUMINAMATH_CALUDE_cafe_customers_l4150_415039


namespace NUMINAMATH_CALUDE_cookies_left_l4150_415012

/-- The number of cookies Paco had initially -/
def initial_cookies : ℕ := 28

/-- The number of cookies Paco ate -/
def eaten_cookies : ℕ := 21

/-- The number of cookies left after Paco ate some -/
def remaining_cookies : ℕ := initial_cookies - eaten_cookies

theorem cookies_left : remaining_cookies = 7 := by
  sorry

end NUMINAMATH_CALUDE_cookies_left_l4150_415012


namespace NUMINAMATH_CALUDE_oliver_candy_boxes_l4150_415021

/-- Given that Oliver initially bought 8 boxes of candy and ended up with a total of 14 boxes,
    prove that he bought 6 boxes later. -/
theorem oliver_candy_boxes (initial_boxes : ℕ) (total_boxes : ℕ) (h1 : initial_boxes = 8) (h2 : total_boxes = 14) :
  total_boxes - initial_boxes = 6 := by
  sorry

end NUMINAMATH_CALUDE_oliver_candy_boxes_l4150_415021


namespace NUMINAMATH_CALUDE_geese_survival_theorem_l4150_415062

/-- Represents the fraction of geese that did not survive the first year out of those that survived the first month -/
def fraction_not_survived_first_year (
  total_eggs : ℕ
  ) (
  hatch_rate : ℚ
  ) (
  first_month_survival_rate : ℚ
  ) (
  first_year_survivors : ℕ
  ) : ℚ :=
  1 - (first_year_survivors : ℚ) / (total_eggs * hatch_rate * first_month_survival_rate)

/-- Theorem stating that the fraction of geese that did not survive the first year is 0 -/
theorem geese_survival_theorem (
  total_eggs : ℕ
  ) (
  hatch_rate : ℚ
  ) (
  first_month_survival_rate : ℚ
  ) (
  first_year_survivors : ℕ
  ) (
  h1 : hatch_rate = 1/3
  ) (
  h2 : first_month_survival_rate = 4/5
  ) (
  h3 : first_year_survivors = 120
  ) (
  h4 : total_eggs * hatch_rate * first_month_survival_rate = first_year_survivors
  ) : fraction_not_survived_first_year total_eggs hatch_rate first_month_survival_rate first_year_survivors = 0 :=
by
  sorry

#check geese_survival_theorem

end NUMINAMATH_CALUDE_geese_survival_theorem_l4150_415062


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l4150_415073

/-- The discriminant of a quadratic equation ax^2 + bx + c -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- The coefficients of the quadratic equation 5x^2 - 9x + 4 -/
def a : ℝ := 5
def b : ℝ := -9
def c : ℝ := 4

theorem quadratic_discriminant :
  discriminant a b c = 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l4150_415073


namespace NUMINAMATH_CALUDE_intersection_point_unique_l4150_415066

/-- The line equation (x+3)/2 = (y-1)/3 = (z-1)/5 -/
def line_eq (x y z : ℝ) : Prop :=
  (x + 3) / 2 = (y - 1) / 3 ∧ (y - 1) / 3 = (z - 1) / 5

/-- The plane equation 2x + 3y + 7z - 52 = 0 -/
def plane_eq (x y z : ℝ) : Prop :=
  2 * x + 3 * y + 7 * z - 52 = 0

/-- The intersection point (-1, 4, 6) -/
def intersection_point : ℝ × ℝ × ℝ := (-1, 4, 6)

theorem intersection_point_unique :
  ∀ x y z : ℝ, line_eq x y z ∧ plane_eq x y z ↔ (x, y, z) = intersection_point :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_unique_l4150_415066


namespace NUMINAMATH_CALUDE_reciprocal_equation_solution_l4150_415009

theorem reciprocal_equation_solution (x : ℝ) :
  (2 - (1 / (1 - x)) = 1 / (1 - x)) → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_equation_solution_l4150_415009


namespace NUMINAMATH_CALUDE_inequality_proof_l4150_415072

theorem inequality_proof (a b c : ℝ) 
  (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) 
  (h4 : a * b + b * c + c * a = 1/3) : 
  1 / (a^2 - b*c + 1) + 1 / (b^2 - c*a + 1) + 1 / (c^2 - a*b + 1) ≤ 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l4150_415072


namespace NUMINAMATH_CALUDE_circle_division_theorem_l4150_415023

/-- The number of regions formed when drawing radii and concentric circles inside a circle -/
def num_regions (num_radii : ℕ) (num_concentric_circles : ℕ) : ℕ :=
  (num_concentric_circles + 1) * num_radii

/-- Theorem stating that 16 radii and 10 concentric circles divide a circle into 176 regions -/
theorem circle_division_theorem :
  num_regions 16 10 = 176 := by
  sorry

end NUMINAMATH_CALUDE_circle_division_theorem_l4150_415023


namespace NUMINAMATH_CALUDE_andrews_cat_catch_l4150_415025

theorem andrews_cat_catch (martha_cat cara_cat T : ℕ) : 
  martha_cat = 10 →
  cara_cat = 47 →
  T = martha_cat + cara_cat →
  T^2 + 2 = 3251 :=
by
  sorry

end NUMINAMATH_CALUDE_andrews_cat_catch_l4150_415025


namespace NUMINAMATH_CALUDE_tiles_per_square_foot_l4150_415078

def wall1_length : ℝ := 5
def wall1_width : ℝ := 8
def wall2_length : ℝ := 7
def wall2_width : ℝ := 8
def turquoise_cost : ℝ := 13
def purple_cost : ℝ := 11
def total_savings : ℝ := 768

theorem tiles_per_square_foot :
  let total_area := wall1_length * wall1_width + wall2_length * wall2_width
  let cost_difference := turquoise_cost - purple_cost
  let total_tiles := total_savings / cost_difference
  total_tiles / total_area = 4 := by sorry

end NUMINAMATH_CALUDE_tiles_per_square_foot_l4150_415078


namespace NUMINAMATH_CALUDE_unique_integer_solution_range_l4150_415086

open Real

theorem unique_integer_solution_range (a : ℝ) : 
  (∃! (x : ℤ), (log (20 - 5 * (x : ℝ)^2) > log (a - (x : ℝ)) + 1)) ↔ 
  (2 ≤ a ∧ a < 5/2) :=
by sorry

end NUMINAMATH_CALUDE_unique_integer_solution_range_l4150_415086


namespace NUMINAMATH_CALUDE_specific_prism_volume_max_prism_volume_max_volume_achievable_l4150_415038

/-- Regular quadrangular pyramid with inscribed regular triangular prism -/
structure PyramidWithPrism where
  /-- Volume of the pyramid -/
  V : ℝ
  /-- Angle between lateral edge and base plane (in radians) -/
  angle : ℝ
  /-- Ratio of the division of the pyramid's height by the prism's face -/
  ratio : ℝ × ℝ
  /-- Volume of the inscribed prism -/
  prismVolume : ℝ
  /-- Constraint: angle is 30 degrees (π/6 radians) -/
  angle_is_30_deg : angle = Real.pi / 6
  /-- Constraint: ratio is valid (both parts positive, sum > 0) -/
  ratio_valid : ratio.1 > 0 ∧ ratio.2 > 0 ∧ ratio.1 + ratio.2 > 0
  /-- Constraint: prism volume is positive and less than pyramid volume -/
  volume_valid : 0 < prismVolume ∧ prismVolume < V

/-- Theorem for the volume of the specific prism -/
theorem specific_prism_volume (p : PyramidWithPrism) (h : p.ratio = (2, 3)) :
  p.prismVolume = 9/250 * p.V := by sorry

/-- Theorem for the maximum volume of any such prism -/
theorem max_prism_volume (p : PyramidWithPrism) :
  p.prismVolume ≤ 1/12 * p.V := by sorry

/-- Theorem that 1/12 is achievable -/
theorem max_volume_achievable (V : ℝ) (h : V > 0) :
  ∃ p : PyramidWithPrism, p.V = V ∧ p.prismVolume = 1/12 * V := by sorry

end NUMINAMATH_CALUDE_specific_prism_volume_max_prism_volume_max_volume_achievable_l4150_415038


namespace NUMINAMATH_CALUDE_line_parabola_single_intersection_l4150_415071

theorem line_parabola_single_intersection (a : ℝ) :
  (∃! x : ℝ, a * x - 6 = x^2 + 4*x + 3) ↔ (a = -2 ∨ a = 10) :=
sorry

end NUMINAMATH_CALUDE_line_parabola_single_intersection_l4150_415071


namespace NUMINAMATH_CALUDE_jeff_shelter_cats_l4150_415098

/-- The number of cats in Jeff's shelter after a week of changes --/
def final_cat_count (initial : ℕ) (monday_added : ℕ) (tuesday_added : ℕ) (people_adopting : ℕ) (cats_per_adoption : ℕ) : ℕ :=
  initial + monday_added + tuesday_added - people_adopting * cats_per_adoption

/-- Theorem stating that Jeff's shelter has 17 cats after the week's changes --/
theorem jeff_shelter_cats : 
  final_cat_count 20 2 1 3 2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_jeff_shelter_cats_l4150_415098


namespace NUMINAMATH_CALUDE_sequence_difference_l4150_415088

theorem sequence_difference (a : ℕ → ℕ) (S : ℕ → ℕ) : 
  (∀ n : ℕ, n > 0 → S n = n^2 + 2*n) →
  (∀ n : ℕ, n ≥ 2 → a n = S n - S (n-1)) →
  a 4 - a 2 = 4 := by
sorry

end NUMINAMATH_CALUDE_sequence_difference_l4150_415088


namespace NUMINAMATH_CALUDE_a_greater_than_b_l4150_415043

theorem a_greater_than_b (n : ℕ) (a b : ℝ) 
  (h_n : n ≥ 2) 
  (h_a_pos : a > 0) 
  (h_b_pos : b > 0)
  (h_a_eq : a^n = a + 1) 
  (h_b_eq : b^(2*n) = b + 3*a) : 
  a > b := by
  sorry

end NUMINAMATH_CALUDE_a_greater_than_b_l4150_415043


namespace NUMINAMATH_CALUDE_number_of_hens_l4150_415097

theorem number_of_hens (total_animals : ℕ) (total_feet : ℕ) (hens : ℕ) (cows : ℕ) : 
  total_animals = 60 →
  total_feet = 200 →
  total_animals = hens + cows →
  total_feet = 2 * hens + 4 * cows →
  hens = 20 := by
sorry

end NUMINAMATH_CALUDE_number_of_hens_l4150_415097


namespace NUMINAMATH_CALUDE_jeans_price_calculation_l4150_415068

/-- The price of jeans after discount and tax -/
def jeans_final_price (socks_price t_shirt_price jeans_price : ℝ)
  (jeans_discount t_shirt_discount tax_rate : ℝ) : ℝ :=
  let jeans_discounted := jeans_price * (1 - jeans_discount)
  let taxable_amount := jeans_discounted + t_shirt_price * (1 - t_shirt_discount)
  jeans_discounted * (1 + tax_rate)

/-- The problem statement -/
theorem jeans_price_calculation :
  let socks_price := 5
  let t_shirt_price := socks_price + 10
  let jeans_price := 2 * t_shirt_price
  let jeans_discount := 0.15
  let t_shirt_discount := 0.10
  let tax_rate := 0.08
  jeans_final_price socks_price t_shirt_price jeans_price
    jeans_discount t_shirt_discount tax_rate = 27.54 := by
  sorry

end NUMINAMATH_CALUDE_jeans_price_calculation_l4150_415068


namespace NUMINAMATH_CALUDE_money_spent_on_baseball_gear_l4150_415040

def initial_amount : ℕ := 67
def amount_left : ℕ := 34

theorem money_spent_on_baseball_gear :
  initial_amount - amount_left = 33 := by sorry

end NUMINAMATH_CALUDE_money_spent_on_baseball_gear_l4150_415040


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l4150_415049

/-- The y-intercept of a line is the y-coordinate of the point where the line intersects the y-axis. -/
def y_intercept (m a : ℝ) : ℝ := a

/-- A line in slope-intercept form is defined by y = mx + b, where m is the slope and b is the y-intercept. -/
def line_equation (x : ℝ) (m b : ℝ) : ℝ := m * x + b

theorem y_intercept_of_line :
  y_intercept 2 1 = 1 :=
sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l4150_415049


namespace NUMINAMATH_CALUDE_zeros_after_one_in_500_to_150_l4150_415036

-- Define 500 as 5 * 10^2
def five_hundred : ℕ := 5 * 10^2

-- Theorem statement
theorem zeros_after_one_in_500_to_150 :
  (∃ n : ℕ, five_hundred^150 = 10^n * (1 + 10 * m) ∧ m < 10) ∧
  (∀ k : ℕ, five_hundred^150 = 10^k * (1 + 10 * m) ∧ m < 10 → k = 300) :=
sorry

end NUMINAMATH_CALUDE_zeros_after_one_in_500_to_150_l4150_415036


namespace NUMINAMATH_CALUDE_simplify_radical_product_l4150_415003

theorem simplify_radical_product (x : ℝ) (hx : x ≥ 0) :
  Real.sqrt (12 * x) * Real.sqrt (18 * x) * Real.sqrt (27 * x) = 54 * x * Real.sqrt x :=
by sorry

end NUMINAMATH_CALUDE_simplify_radical_product_l4150_415003


namespace NUMINAMATH_CALUDE_medical_team_selection_l4150_415028

theorem medical_team_selection (male_doctors female_doctors : ℕ) 
  (h1 : male_doctors = 6) (h2 : female_doctors = 5) :
  (Nat.choose male_doctors 2) * (Nat.choose female_doctors 1) = 75 := by
  sorry

end NUMINAMATH_CALUDE_medical_team_selection_l4150_415028


namespace NUMINAMATH_CALUDE_intersection_A_B_complement_union_A_B_l4150_415015

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B : Set ℝ := {x | 0 < x ∧ x ≤ 3}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x : ℝ | 0 < x ∧ x < 2} := by sorry

-- Theorem for the complement of the union of A and B
theorem complement_union_A_B : (A ∪ B)ᶜ = {x : ℝ | x ≤ -1 ∨ x > 3} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_complement_union_A_B_l4150_415015


namespace NUMINAMATH_CALUDE_sixth_graders_count_l4150_415035

/-- The number of fifth graders -/
def fifth_graders : ℕ := 109

/-- The number of seventh graders -/
def seventh_graders : ℕ := 118

/-- The number of teachers -/
def teachers : ℕ := 4

/-- The number of parents per grade -/
def parents_per_grade : ℕ := 2

/-- The number of buses -/
def buses : ℕ := 5

/-- The number of seats per bus -/
def seats_per_bus : ℕ := 72

/-- The total number of seats available -/
def total_seats : ℕ := buses * seats_per_bus

/-- The total number of chaperones -/
def total_chaperones : ℕ := (teachers + parents_per_grade) * 3

/-- The number of students and chaperones excluding sixth graders -/
def non_sixth_grade_total : ℕ := fifth_graders + seventh_graders + total_chaperones

theorem sixth_graders_count : total_seats - non_sixth_grade_total = 115 := by
  sorry

end NUMINAMATH_CALUDE_sixth_graders_count_l4150_415035


namespace NUMINAMATH_CALUDE_largest_triangle_perimeter_l4150_415045

theorem largest_triangle_perimeter :
  ∀ x : ℤ,
  (8 : ℝ) + 11 > (x : ℝ) →
  (8 : ℝ) + (x : ℝ) > 11 →
  (11 : ℝ) + (x : ℝ) > 8 →
  (8 : ℝ) + 11 + (x : ℝ) ≤ 37 :=
by sorry

end NUMINAMATH_CALUDE_largest_triangle_perimeter_l4150_415045


namespace NUMINAMATH_CALUDE_exhibition_spacing_l4150_415076

theorem exhibition_spacing (wall_width : ℕ) (painting_width : ℕ) (num_paintings : ℕ) :
  wall_width = 320 ∧ painting_width = 30 ∧ num_paintings = 6 →
  (wall_width - num_paintings * painting_width) / (num_paintings + 1) = 20 :=
by sorry

end NUMINAMATH_CALUDE_exhibition_spacing_l4150_415076


namespace NUMINAMATH_CALUDE_no_solution_exists_l4150_415030

theorem no_solution_exists : ¬∃ (s c : ℕ), 
  15 ≤ s ∧ s ≤ 35 ∧ c > 0 ∧ 30 * s + 31 * c = 1200 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_exists_l4150_415030


namespace NUMINAMATH_CALUDE_last_segment_speed_l4150_415059

/-- Represents the average speed during a journey segment -/
structure JourneySegment where
  duration : ℚ  -- Duration in hours
  speed : ℚ     -- Average speed in mph
  distance : ℚ  -- Distance traveled in miles

/-- Represents a complete journey -/
structure Journey where
  totalDistance : ℚ
  totalTime : ℚ
  segments : List JourneySegment

/-- Calculates the average speed for a given distance and time -/
def averageSpeed (distance : ℚ) (time : ℚ) : ℚ :=
  distance / time

theorem last_segment_speed (j : Journey) 
  (h1 : j.totalDistance = 120)
  (h2 : j.totalTime = 2)
  (h3 : j.segments.length = 3)
  (h4 : j.segments[0].duration = 2/3)
  (h5 : j.segments[0].speed = 50)
  (h6 : j.segments[1].duration = 5/6)
  (h7 : j.segments[1].speed = 60)
  (h8 : j.segments[2].duration = 1/2) :
  averageSpeed j.segments[2].distance j.segments[2].duration = 220/3 := by
  sorry

#eval (220 : ℚ) / 3  -- To verify the result is approximately 73.33

end NUMINAMATH_CALUDE_last_segment_speed_l4150_415059


namespace NUMINAMATH_CALUDE_total_snakes_in_neighborhood_l4150_415033

theorem total_snakes_in_neighborhood (total_people : ℕ) 
  (only_dogs only_cats only_snakes only_rabbits only_birds : ℕ)
  (dogs_and_cats dogs_and_snakes dogs_and_rabbits dogs_and_birds : ℕ)
  (cats_and_snakes cats_and_rabbits cats_and_birds : ℕ)
  (snakes_and_rabbits snakes_and_birds rabbits_and_birds : ℕ)
  (dogs_cats_snakes dogs_cats_rabbits dogs_cats_birds : ℕ)
  (dogs_snakes_rabbits cats_snakes_rabbits : ℕ)
  (all_five : ℕ) :
  total_people = 125 →
  only_dogs = 20 →
  only_cats = 15 →
  only_snakes = 8 →
  only_rabbits = 10 →
  only_birds = 5 →
  dogs_and_cats = 12 →
  dogs_and_snakes = 7 →
  dogs_and_rabbits = 4 →
  dogs_and_birds = 3 →
  cats_and_snakes = 9 →
  cats_and_rabbits = 6 →
  cats_and_birds = 2 →
  snakes_and_rabbits = 5 →
  snakes_and_birds = 3 →
  rabbits_and_birds = 1 →
  dogs_cats_snakes = 4 →
  dogs_cats_rabbits = 2 →
  dogs_cats_birds = 1 →
  dogs_snakes_rabbits = 3 →
  cats_snakes_rabbits = 2 →
  all_five = 1 →
  only_snakes + dogs_and_snakes + cats_and_snakes + snakes_and_rabbits + 
  snakes_and_birds + dogs_cats_snakes + dogs_snakes_rabbits + 
  cats_snakes_rabbits + all_five = 42 :=
by sorry


end NUMINAMATH_CALUDE_total_snakes_in_neighborhood_l4150_415033


namespace NUMINAMATH_CALUDE_shortest_path_on_specific_floor_l4150_415052

/-- Represents a rectangular floor with a missing tile -/
structure RectangularFloor :=
  (width : Nat)
  (length : Nat)
  (missingTileX : Nat)
  (missingTileY : Nat)

/-- Calculates the shortest path length for a bug traversing the floor -/
def shortestPathLength (floor : RectangularFloor) : Nat :=
  floor.width + floor.length - Nat.gcd floor.width floor.length + 1

/-- Theorem stating the shortest path length for the given floor configuration -/
theorem shortest_path_on_specific_floor :
  let floor : RectangularFloor := {
    width := 12,
    length := 20,
    missingTileX := 6,
    missingTileY := 10
  }
  shortestPathLength floor = 29 := by
  sorry


end NUMINAMATH_CALUDE_shortest_path_on_specific_floor_l4150_415052


namespace NUMINAMATH_CALUDE_decimal_addition_subtraction_l4150_415037

theorem decimal_addition_subtraction :
  (0.45 : ℚ) - 0.03 + 0.008 = 0.428 := by
  sorry

end NUMINAMATH_CALUDE_decimal_addition_subtraction_l4150_415037


namespace NUMINAMATH_CALUDE_recurring_decimal_fraction_l4150_415042

theorem recurring_decimal_fraction (a b : ℚ) :
  a = 36 * (1 / 99) ∧ b = 12 * (1 / 99) → a / b = 3 := by
  sorry

end NUMINAMATH_CALUDE_recurring_decimal_fraction_l4150_415042


namespace NUMINAMATH_CALUDE_binary_101101_is_45_l4150_415057

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_101101_is_45 :
  binary_to_decimal [true, false, true, true, false, true] = 45 := by
  sorry

end NUMINAMATH_CALUDE_binary_101101_is_45_l4150_415057


namespace NUMINAMATH_CALUDE_megan_finished_problems_l4150_415087

theorem megan_finished_problems (total_problems : ℕ) (remaining_pages : ℕ) (problems_per_page : ℕ) 
  (h1 : total_problems = 40)
  (h2 : remaining_pages = 2)
  (h3 : problems_per_page = 7) :
  total_problems - (remaining_pages * problems_per_page) = 26 := by
sorry

end NUMINAMATH_CALUDE_megan_finished_problems_l4150_415087


namespace NUMINAMATH_CALUDE_x_equals_two_l4150_415051

/-- The sum of digits for all four-digit numbers formed by 1, 4, 5, and x -/
def sumOfDigits (x : ℕ) : ℕ :=
  if x = 0 then
    24 * (1 + 4 + 5)
  else
    24 * (1 + 4 + 5 + x)

/-- Theorem stating that x must be 2 given the conditions -/
theorem x_equals_two :
  ∃! x : ℕ, x ≤ 9 ∧ sumOfDigits x = 288 :=
sorry

end NUMINAMATH_CALUDE_x_equals_two_l4150_415051


namespace NUMINAMATH_CALUDE_exponential_function_fixed_point_l4150_415026

theorem exponential_function_fixed_point (a b : ℝ) (ha : a > 0) :
  (∀ x, (a^(x - b) + 1 = 2) ↔ (x = 1)) → b = 1 := by
  sorry

end NUMINAMATH_CALUDE_exponential_function_fixed_point_l4150_415026


namespace NUMINAMATH_CALUDE_fraction_equality_l4150_415055

theorem fraction_equality (a b : ℝ) (h : a / b = 5 / 4) :
  (4 * a + 3 * b) / (4 * a - 3 * b) = 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l4150_415055


namespace NUMINAMATH_CALUDE_rhombus_always_symmetrical_triangle_not_always_symmetrical_parallelogram_not_always_symmetrical_trapezoid_not_always_symmetrical_l4150_415013

-- Define the basic shapes
inductive Shape
  | Triangle
  | Parallelogram
  | Rhombus
  | Trapezoid

-- Define a property for symmetry
def isSymmetrical (s : Shape) : Prop :=
  match s with
  | Shape.Rhombus => True
  | _ => false

-- Theorem stating that only Rhombus is always symmetrical
theorem rhombus_always_symmetrical :
  ∀ (s : Shape), isSymmetrical s ↔ s = Shape.Rhombus :=
by sorry

-- Additional theorems to show that other shapes are not always symmetrical
theorem triangle_not_always_symmetrical :
  ∃ (t : Shape), t = Shape.Triangle ∧ ¬(isSymmetrical t) :=
by sorry

theorem parallelogram_not_always_symmetrical :
  ∃ (p : Shape), p = Shape.Parallelogram ∧ ¬(isSymmetrical p) :=
by sorry

theorem trapezoid_not_always_symmetrical :
  ∃ (t : Shape), t = Shape.Trapezoid ∧ ¬(isSymmetrical t) :=
by sorry

end NUMINAMATH_CALUDE_rhombus_always_symmetrical_triangle_not_always_symmetrical_parallelogram_not_always_symmetrical_trapezoid_not_always_symmetrical_l4150_415013


namespace NUMINAMATH_CALUDE_solution_range_l4150_415067

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem solution_range (a b c : ℝ) :
  (f a b c 3 = 0.5) →
  (f a b c 4 = -0.5) →
  (f a b c 5 = -1) →
  ∃ x : ℝ, (ax^2 + b*x + c = 0) ∧ (3 < x) ∧ (x < 4) :=
by sorry

end NUMINAMATH_CALUDE_solution_range_l4150_415067


namespace NUMINAMATH_CALUDE_fraction_equality_l4150_415069

theorem fraction_equality (n : ℝ) (h : n ≥ 2) :
  1 / (n^2 - 1) = (1/2) * (1 / (n - 1) - 1 / (n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l4150_415069


namespace NUMINAMATH_CALUDE_min_value_of_function_l4150_415070

theorem min_value_of_function (t : ℝ) (h : t > 0) :
  (t^2 - 4*t + 1) / t ≥ -2 ∧ ∃ t > 0, (t^2 - 4*t + 1) / t = -2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_function_l4150_415070


namespace NUMINAMATH_CALUDE_inequality_proof_largest_constant_l4150_415010

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x / Real.sqrt (y + z) + y / Real.sqrt (z + x) + z / Real.sqrt (x + y) ≤ (Real.sqrt 6 / 2) * Real.sqrt (x + y + z) :=
sorry

theorem largest_constant :
  ∀ k, (∀ (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0),
    x / Real.sqrt (y + z) + y / Real.sqrt (z + x) + z / Real.sqrt (x + y) ≤ k * Real.sqrt (x + y + z)) →
  k ≤ Real.sqrt 6 / 2 :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_largest_constant_l4150_415010


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l4150_415065

theorem arithmetic_mean_of_fractions :
  (1 / 2 : ℚ) * ((3 / 8 : ℚ) + (5 / 9 : ℚ)) = 67 / 144 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l4150_415065


namespace NUMINAMATH_CALUDE_base_7_representation_l4150_415016

/-- Converts a list of digits in base 7 to a natural number in base 10 -/
def base7ToNat (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The base-10 representation of 624 -/
def base10Num : Nat := 624

/-- The base-7 representation of 624 as a list of digits -/
def base7Digits : List Nat := [1, 5, 5, 1]

theorem base_7_representation :
  base7ToNat base7Digits = base10Num := by sorry

end NUMINAMATH_CALUDE_base_7_representation_l4150_415016


namespace NUMINAMATH_CALUDE_school_contribution_l4150_415082

def book_cost : ℕ := 12
def num_students : ℕ := 30
def sally_paid : ℕ := 40

theorem school_contribution : 
  ∃ (school_amount : ℕ), 
    school_amount = book_cost * num_students - sally_paid ∧ 
    school_amount = 320 := by
  sorry

end NUMINAMATH_CALUDE_school_contribution_l4150_415082


namespace NUMINAMATH_CALUDE_cars_distance_theorem_l4150_415005

/-- The distance between two cars on a straight road -/
def distance_between_cars (initial_distance : ℝ) (car1_distance : ℝ) (car2_distance : ℝ) : ℝ :=
  initial_distance - (car1_distance + car2_distance)

/-- Theorem: The distance between two cars is 28 km -/
theorem cars_distance_theorem (initial_distance car1_distance car2_distance : ℝ) 
  (h1 : initial_distance = 113)
  (h2 : car1_distance = 50)
  (h3 : car2_distance = 35) :
  distance_between_cars initial_distance car1_distance car2_distance = 28 := by
  sorry

#eval distance_between_cars 113 50 35

end NUMINAMATH_CALUDE_cars_distance_theorem_l4150_415005


namespace NUMINAMATH_CALUDE_foreign_trade_income_equation_l4150_415034

/-- The foreign trade income equation over two years with a constant growth rate -/
theorem foreign_trade_income_equation
  (m : ℝ) -- foreign trade income in 2001 (billion yuan)
  (x : ℝ) -- annual growth rate
  (n : ℝ) -- foreign trade income in 2003 (billion yuan)
  : m * (1 + x)^2 = n :=
by sorry

end NUMINAMATH_CALUDE_foreign_trade_income_equation_l4150_415034


namespace NUMINAMATH_CALUDE_bisection_method_max_experiments_l4150_415080

theorem bisection_method_max_experiments (n : ℕ) (h : n = 33) :
  ∃ k : ℕ, k = 6 ∧ ∀ m : ℕ, 2^m < n → m < k :=
sorry

end NUMINAMATH_CALUDE_bisection_method_max_experiments_l4150_415080


namespace NUMINAMATH_CALUDE_central_cell_value_l4150_415006

/-- A 3x3 table of real numbers -/
structure Table :=
  (a b c d e f g h i : ℝ)

/-- The conditions for the table -/
def satisfies_conditions (t : Table) : Prop :=
  t.a * t.b * t.c = 10 ∧
  t.d * t.e * t.f = 10 ∧
  t.g * t.h * t.i = 10 ∧
  t.a * t.d * t.g = 10 ∧
  t.b * t.e * t.h = 10 ∧
  t.c * t.f * t.i = 10 ∧
  t.a * t.b * t.d * t.e = 3 ∧
  t.b * t.c * t.e * t.f = 3 ∧
  t.d * t.e * t.g * t.h = 3 ∧
  t.e * t.f * t.h * t.i = 3

/-- The theorem statement -/
theorem central_cell_value (t : Table) (h : satisfies_conditions t) : t.e = 0.00081 := by
  sorry

end NUMINAMATH_CALUDE_central_cell_value_l4150_415006


namespace NUMINAMATH_CALUDE_average_of_three_numbers_l4150_415032

theorem average_of_three_numbers (y : ℝ) : (15 + 24 + y) / 3 = 23 → y = 30 := by
  sorry

end NUMINAMATH_CALUDE_average_of_three_numbers_l4150_415032


namespace NUMINAMATH_CALUDE_boat_rental_cost_l4150_415079

theorem boat_rental_cost (students : ℕ) (boat_capacity : ℕ) (rental_fee : ℕ) 
  (h1 : students = 42)
  (h2 : boat_capacity = 6)
  (h3 : rental_fee = 125) :
  (((students + boat_capacity - 1) / boat_capacity) * rental_fee) = 875 :=
by
  sorry

#check boat_rental_cost

end NUMINAMATH_CALUDE_boat_rental_cost_l4150_415079


namespace NUMINAMATH_CALUDE_parabola_point_order_l4150_415091

/-- A parabola with equation y = -(x-2)^2 + k -/
structure Parabola where
  k : ℝ

/-- A point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point lies on the parabola -/
def lies_on (p : Point) (para : Parabola) : Prop :=
  p.y = -(p.x - 2)^2 + para.k

theorem parabola_point_order (para : Parabola) 
  (A B C : Point)
  (hA : A.x = -2) (hB : B.x = -1) (hC : C.x = 3)
  (liesA : lies_on A para) (liesB : lies_on B para) (liesC : lies_on C para) :
  A.y < B.y ∧ B.y < C.y := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_order_l4150_415091


namespace NUMINAMATH_CALUDE_school_population_l4150_415063

theorem school_population (total_students : ℕ) : 
  (128 : ℕ) = (total_students / 2) →
  total_students = 256 := by
sorry

end NUMINAMATH_CALUDE_school_population_l4150_415063


namespace NUMINAMATH_CALUDE_tom_read_six_books_in_june_l4150_415058

/-- The number of books Tom read in May -/
def books_may : ℕ := 2

/-- The number of books Tom read in July -/
def books_july : ℕ := 10

/-- The total number of books Tom read -/
def total_books : ℕ := 18

/-- The number of books Tom read in June -/
def books_june : ℕ := total_books - (books_may + books_july)

theorem tom_read_six_books_in_june : books_june = 6 := by
  sorry

end NUMINAMATH_CALUDE_tom_read_six_books_in_june_l4150_415058


namespace NUMINAMATH_CALUDE_squares_below_line_l4150_415024

/-- Represents a line in 2D space --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Counts the number of integer points strictly below a line in the first quadrant --/
def countPointsBelowLine (l : Line) : ℕ :=
  sorry

/-- The specific line from the problem --/
def problemLine : Line :=
  { a := 12, b := 180, c := 2160 }

/-- The theorem statement --/
theorem squares_below_line :
  countPointsBelowLine problemLine = 984 := by
  sorry

end NUMINAMATH_CALUDE_squares_below_line_l4150_415024


namespace NUMINAMATH_CALUDE_circles_are_separate_l4150_415004

-- Define the circles
def C₁ (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1
def C₂ (x y : ℝ) : Prop := (x + 3)^2 + (y - 2)^2 = 4

-- Define the centers and radii
def center₁ : ℝ × ℝ := (1, 0)
def center₂ : ℝ × ℝ := (-3, 2)
def radius₁ : ℝ := 1
def radius₂ : ℝ := 2

-- Theorem statement
theorem circles_are_separate :
  let d := Real.sqrt ((center₁.1 - center₂.1)^2 + (center₁.2 - center₂.2)^2)
  d > radius₁ + radius₂ :=
by sorry

end NUMINAMATH_CALUDE_circles_are_separate_l4150_415004


namespace NUMINAMATH_CALUDE_equal_profit_percentage_l4150_415047

def shopkeeper_profit (total_quantity : ℝ) (portion1_percentage : ℝ) (portion2_percentage : ℝ) (profit_percentage : ℝ) : Prop :=
  portion1_percentage + portion2_percentage = 100 ∧
  portion1_percentage > 0 ∧
  portion2_percentage > 0 ∧
  profit_percentage ≥ 0

theorem equal_profit_percentage 
  (total_quantity : ℝ) 
  (portion1_percentage : ℝ) 
  (portion2_percentage : ℝ) 
  (total_profit_percentage : ℝ) 
  (h : shopkeeper_profit total_quantity portion1_percentage portion2_percentage total_profit_percentage) :
  ∃ (individual_profit_percentage : ℝ),
    individual_profit_percentage = total_profit_percentage ∧
    individual_profit_percentage * portion1_percentage / 100 + 
    individual_profit_percentage * portion2_percentage / 100 = 
    total_profit_percentage := by
  sorry

end NUMINAMATH_CALUDE_equal_profit_percentage_l4150_415047


namespace NUMINAMATH_CALUDE_stratified_sampling_size_l4150_415089

theorem stratified_sampling_size (high_school_students junior_high_students : ℕ) 
  (high_school_sample : ℕ) (total_sample : ℕ) : 
  high_school_students = 3500 →
  junior_high_students = 1500 →
  high_school_sample = 70 →
  (high_school_sample : ℚ) / high_school_students = 
    (total_sample : ℚ) / (high_school_students + junior_high_students) →
  total_sample = 100 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_size_l4150_415089


namespace NUMINAMATH_CALUDE_rectangle_area_l4150_415050

theorem rectangle_area : 
  ∀ (square_side : ℝ) (circle_radius : ℝ) (rectangle_length : ℝ) (rectangle_breadth : ℝ),
    square_side^2 = 625 →
    circle_radius = square_side →
    rectangle_length = (2/5) * circle_radius →
    rectangle_breadth = 10 →
    rectangle_length * rectangle_breadth = 100 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l4150_415050


namespace NUMINAMATH_CALUDE_negation_of_p_l4150_415095

-- Define the proposition p
def p : Prop := ∃ m : ℝ, m > 0 ∧ ∃ x : ℝ, m * x^2 + x - 2*m = 0

-- State the theorem
theorem negation_of_p : ¬p ↔ ∀ m : ℝ, m > 0 → ∀ x : ℝ, m * x^2 + x - 2*m ≠ 0 := by sorry

end NUMINAMATH_CALUDE_negation_of_p_l4150_415095


namespace NUMINAMATH_CALUDE_lowest_temperature_record_l4150_415031

/-- The lowest temperature ever recorded in the world -/
def lowest_temperature : ℝ := -89.2

/-- The location where the lowest temperature was recorded -/
def record_location : String := "Vostok Station, Antarctica"

/-- How the temperature is written -/
def temperature_written : String := "-89.2 °C"

/-- How the temperature is read -/
def temperature_read : String := "negative eighty-nine point two degrees Celsius"

/-- Theorem stating the lowest recorded temperature and its representation -/
theorem lowest_temperature_record :
  lowest_temperature = -89.2 ∧
  record_location = "Vostok Station, Antarctica" ∧
  temperature_written = "-89.2 °C" ∧
  temperature_read = "negative eighty-nine point two degrees Celsius" :=
by sorry

end NUMINAMATH_CALUDE_lowest_temperature_record_l4150_415031


namespace NUMINAMATH_CALUDE_no_integer_solutions_for_perfect_square_l4150_415000

theorem no_integer_solutions_for_perfect_square : 
  ¬ ∃ (x : ℤ), ∃ (y : ℤ), x^4 + 4*x^3 + 10*x^2 + 4*x + 29 = y^2 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_for_perfect_square_l4150_415000


namespace NUMINAMATH_CALUDE_gcd_power_minus_one_l4150_415081

theorem gcd_power_minus_one (m n : ℕ+) :
  Nat.gcd (2^(m : ℕ) - 1) (2^(n : ℕ) - 1) = 2^(Nat.gcd m n) - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_power_minus_one_l4150_415081


namespace NUMINAMATH_CALUDE_fraction_inequality_l4150_415064

theorem fraction_inequality (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0) (hab : a > b) :
  b / a < (b + x) / (a + x) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l4150_415064


namespace NUMINAMATH_CALUDE_geometric_decreasing_condition_l4150_415090

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

def is_decreasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) ≤ a n

theorem geometric_decreasing_condition (a : ℕ → ℝ) 
  (h_geom : is_geometric_sequence a) (h_pos : a 1 > 0) :
  (is_decreasing_sequence a → a 1 > a 2) ∧
  ¬(a 1 > a 2 → is_decreasing_sequence a) :=
sorry

end NUMINAMATH_CALUDE_geometric_decreasing_condition_l4150_415090


namespace NUMINAMATH_CALUDE_cubic_sum_l4150_415048

theorem cubic_sum (x y : ℝ) (h1 : x + y = 8) (h2 : x * y = 12) : x^3 + y^3 = 224 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_l4150_415048


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l4150_415008

/-- Given a quadratic function f(x) = ax² + bx + c with specific properties,
    prove statements about its coefficients and roots. -/
theorem quadratic_function_properties (a b c : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = a * x^2 + b * x + c)
    (h2 : f 1 = -a / 2)
    (h3 : 3 * a > 2 * c)
    (h4 : 2 * c > 2 * b) : 
  (a > 0 ∧ -3 < b / a ∧ b / a < -3 / 4) ∧ 
  (∃ x : ℝ, 0 < x ∧ x < 2 ∧ f x = 0) ∧
  (∀ x₁ x₂ : ℝ, f x₁ = 0 → f x₂ = 0 → x₁ ≠ x₂ → 
    Real.sqrt 2 ≤ |x₁ - x₂| ∧ |x₁ - x₂| < Real.sqrt 57 / 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l4150_415008


namespace NUMINAMATH_CALUDE_abc_product_l4150_415020

theorem abc_product (a b c : ℝ) 
  (eq1 : a + b = 23)
  (eq2 : b + c = 25)
  (eq3 : c + a = 30) :
  a * b * c = 2016 := by
  sorry

end NUMINAMATH_CALUDE_abc_product_l4150_415020


namespace NUMINAMATH_CALUDE_mrs_hilt_reading_l4150_415092

/-- The number of books Mrs. Hilt read -/
def num_books : ℕ := 4

/-- The number of chapters in each book -/
def chapters_per_book : ℕ := 17

/-- The total number of chapters Mrs. Hilt read -/
def total_chapters : ℕ := num_books * chapters_per_book

theorem mrs_hilt_reading :
  total_chapters = 68 := by sorry

end NUMINAMATH_CALUDE_mrs_hilt_reading_l4150_415092


namespace NUMINAMATH_CALUDE_total_spent_is_14_l4150_415029

/-- The cost of one set of barrettes in dollars -/
def barrette_cost : ℕ := 3

/-- The cost of one comb in dollars -/
def comb_cost : ℕ := 1

/-- The number of barrette sets Kristine buys -/
def kristine_barrettes : ℕ := 1

/-- The number of combs Kristine buys -/
def kristine_combs : ℕ := 1

/-- The number of barrette sets Crystal buys -/
def crystal_barrettes : ℕ := 3

/-- The number of combs Crystal buys -/
def crystal_combs : ℕ := 1

/-- The total amount spent by both Kristine and Crystal -/
def total_spent : ℕ := 
  (kristine_barrettes * barrette_cost + kristine_combs * comb_cost) +
  (crystal_barrettes * barrette_cost + crystal_combs * comb_cost)

theorem total_spent_is_14 : total_spent = 14 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_is_14_l4150_415029


namespace NUMINAMATH_CALUDE_smallest_n_l4150_415041

def is_factor (a b : ℕ) : Prop := b % a = 0

theorem smallest_n : ∃ (n : ℕ), n > 0 ∧ 
  is_factor 25 (n * 2^5 * 6^2 * 7^3) ∧ 
  is_factor 27 (n * 2^5 * 6^2 * 7^3) ∧
  (∀ (m : ℕ), m > 0 → 
    is_factor 25 (m * 2^5 * 6^2 * 7^3) → 
    is_factor 27 (m * 2^5 * 6^2 * 7^3) → 
    m ≥ n) ∧
  n = 75 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_l4150_415041


namespace NUMINAMATH_CALUDE_orange_juice_division_l4150_415077

theorem orange_juice_division (total_pints : ℚ) (num_glasses : ℕ) 
  (h1 : total_pints = 153)
  (h2 : num_glasses = 5) :
  total_pints / num_glasses = 30.6 := by
  sorry

end NUMINAMATH_CALUDE_orange_juice_division_l4150_415077


namespace NUMINAMATH_CALUDE_fruit_arrangement_l4150_415018

theorem fruit_arrangement (n a o b p : ℕ) 
  (total : n = a + o + b + p)
  (apple : a = 4)
  (orange : o = 2)
  (banana : b = 2)
  (pear : p = 1) :
  Nat.factorial n / (Nat.factorial a * Nat.factorial o * Nat.factorial b * Nat.factorial p) = 3780 := by
  sorry

end NUMINAMATH_CALUDE_fruit_arrangement_l4150_415018


namespace NUMINAMATH_CALUDE_absolute_value_square_l4150_415002

theorem absolute_value_square (a b : ℝ) : a > |b| → a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_square_l4150_415002


namespace NUMINAMATH_CALUDE_queue_adjustment_ways_l4150_415094

theorem queue_adjustment_ways (n m k : ℕ) (hn : n = 10) (hm : m = 3) (hk : k = 2) :
  (Nat.choose (n - m) k) * (m + 1) * (m + 2) = 420 := by
  sorry

end NUMINAMATH_CALUDE_queue_adjustment_ways_l4150_415094


namespace NUMINAMATH_CALUDE_towel_rate_calculation_l4150_415084

/-- Given the prices and quantities of towels, calculates the unknown rate. -/
def unknown_towel_rate (qty1 qty2 qty3 : ℕ) (price1 price2 avg_price : ℚ) : ℚ :=
  ((qty1 + qty2 + qty3 : ℚ) * avg_price - qty1 * price1 - qty2 * price2) / qty3

/-- Theorem stating that under the given conditions, the unknown rate is 325. -/
theorem towel_rate_calculation :
  let qty1 := 3
  let qty2 := 5
  let qty3 := 2
  let price1 := 100
  let price2 := 150
  let avg_price := 170
  unknown_towel_rate qty1 qty2 qty3 price1 price2 avg_price = 325 := by
sorry

end NUMINAMATH_CALUDE_towel_rate_calculation_l4150_415084


namespace NUMINAMATH_CALUDE_tree_planting_impossibility_l4150_415074

theorem tree_planting_impossibility :
  ∀ (arrangement : List ℕ),
    (arrangement.length = 50) →
    (∀ n : ℕ, n ∈ arrangement → 1 ≤ n ∧ n ≤ 25) →
    (∀ n : ℕ, 1 ≤ n ∧ n ≤ 25 → (arrangement.count n = 2)) →
    ¬(∀ n : ℕ, 1 ≤ n ∧ n ≤ 25 →
      ∃ (i j : ℕ), i < j ∧ 
        arrangement.nthLe i sorry = n ∧
        arrangement.nthLe j sorry = n ∧
        (j - i = 2 ∨ j - i = 4)) :=
by sorry

end NUMINAMATH_CALUDE_tree_planting_impossibility_l4150_415074


namespace NUMINAMATH_CALUDE_online_price_calculation_l4150_415056

/-- Calculates the price a buyer observes online for a product sold by a distributor through an online store, given various costs and desired profit margin. -/
theorem online_price_calculation 
  (producer_price : ℝ) 
  (shipping_cost : ℝ) 
  (commission_rate : ℝ) 
  (tax_rate : ℝ) 
  (profit_margin : ℝ) 
  (h1 : producer_price = 19) 
  (h2 : shipping_cost = 5) 
  (h3 : commission_rate = 0.2) 
  (h4 : tax_rate = 0.1) 
  (h5 : profit_margin = 0.2) : 
  ∃ (online_price : ℝ), online_price = 39.6 ∧ 
  online_price * (1 - commission_rate) = 
    (producer_price + shipping_cost) * (1 + profit_margin) * (1 + tax_rate) := by
  sorry

end NUMINAMATH_CALUDE_online_price_calculation_l4150_415056


namespace NUMINAMATH_CALUDE_unused_streetlights_l4150_415083

theorem unused_streetlights (total_streetlights : ℕ) (num_squares : ℕ) (lights_per_square : ℕ) :
  total_streetlights = 200 →
  num_squares = 15 →
  lights_per_square = 12 →
  total_streetlights - (num_squares * lights_per_square) = 20 := by
  sorry

#check unused_streetlights

end NUMINAMATH_CALUDE_unused_streetlights_l4150_415083


namespace NUMINAMATH_CALUDE_vector_parallel_sum_l4150_415001

/-- Given vectors a and b in ℝ², if a is parallel to (a + b), then the y-coordinate of b is -1/2. -/
theorem vector_parallel_sum (a b : ℝ × ℝ) (h : a = (4, -1)) (h' : b.1 = 2) :
  (∃ (k : ℝ), k • a = a + b) → b.2 = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_sum_l4150_415001


namespace NUMINAMATH_CALUDE_fifth_month_sale_l4150_415046

def sales_problem (sales1 sales2 sales3 sales4 sales6 average : ℕ) : Prop :=
  let total_sales := average * 6
  let known_sales := sales1 + sales2 + sales3 + sales4 + sales6
  total_sales - known_sales = 3560

theorem fifth_month_sale :
  sales_problem 3435 3920 3855 4230 2000 3500 := by
  sorry

end NUMINAMATH_CALUDE_fifth_month_sale_l4150_415046


namespace NUMINAMATH_CALUDE_not_unique_perpendicular_l4150_415044

/-- A line in a plane --/
structure Line where
  -- We don't need to define the internals of a line for this statement
  mk :: 

/-- A plane --/
structure Plane where
  -- We don't need to define the internals of a plane for this statement
  mk ::

/-- Perpendicularity relation between two lines --/
def perpendicular (l1 l2 : Line) : Prop :=
  sorry

/-- The statement to be proven false --/
def unique_perpendicular (p : Plane) : Prop :=
  ∃! (l : Line), ∀ (m : Line), perpendicular l m

/-- The theorem stating that the unique perpendicular line statement is false --/
theorem not_unique_perpendicular :
  ∃ (p : Plane), ¬(unique_perpendicular p) :=
sorry

end NUMINAMATH_CALUDE_not_unique_perpendicular_l4150_415044


namespace NUMINAMATH_CALUDE_events_B_C_complementary_l4150_415022

-- Define the sample space (faces of the die)
def Die : Type := Fin 6

-- Define event B
def eventB (x : Die) : Prop := x.val + 1 ≤ 3

-- Define event C
def eventC (x : Die) : Prop := x.val + 1 ≥ 4

-- Theorem statement
theorem events_B_C_complementary :
  ∀ (x : Die), (eventB x ∧ ¬eventC x) ∨ (¬eventB x ∧ eventC x) :=
by sorry

end NUMINAMATH_CALUDE_events_B_C_complementary_l4150_415022
