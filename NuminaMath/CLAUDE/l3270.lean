import Mathlib

namespace NUMINAMATH_CALUDE_triangular_pyramid_angles_l3270_327061

/-- 
Given a triangular pyramid with lateral surface area S and lateral edge length l,
if the plane angles at the apex form an arithmetic progression with common difference π/6,
then the angles are as specified.
-/
theorem triangular_pyramid_angles (S l : ℝ) (h_positive_S : S > 0) (h_positive_l : l > 0) :
  let α := Real.arcsin ((S * (Real.sqrt 3 - 1)) / l^2)
  ∃ (θ₁ θ₂ θ₃ : ℝ),
    (θ₁ = α - π/6 ∧ θ₂ = α ∧ θ₃ = α + π/6) ∧
    (θ₁ + θ₂ + θ₃ = π/2) ∧
    (θ₃ - θ₂ = θ₂ - θ₁) ∧
    (θ₃ - θ₂ = π/6) ∧
    (S = (l^2 / 2) * (Real.sin θ₁ + Real.sin θ₂ + Real.sin θ₃)) :=
by sorry

end NUMINAMATH_CALUDE_triangular_pyramid_angles_l3270_327061


namespace NUMINAMATH_CALUDE_product_of_roots_l3270_327027

theorem product_of_roots (x : ℝ) : 
  (x^2 - 4*x - 42 = 0) → 
  ∃ y : ℝ, (y^2 - 4*y - 42 = 0) ∧ (x * y = -42) :=
by sorry

end NUMINAMATH_CALUDE_product_of_roots_l3270_327027


namespace NUMINAMATH_CALUDE_tan_195_in_terms_of_cos_165_l3270_327052

theorem tan_195_in_terms_of_cos_165 (a : ℝ) (h : Real.cos (165 * π / 180) = a) :
  Real.tan (195 * π / 180) = -Real.sqrt (1 - a^2) / a := by
  sorry

end NUMINAMATH_CALUDE_tan_195_in_terms_of_cos_165_l3270_327052


namespace NUMINAMATH_CALUDE_bridge_toll_fee_calculation_l3270_327004

/-- Represents the taxi fare structure -/
structure TaxiFare where
  start_fee : ℝ
  per_mile_rate : ℝ

/-- Calculates the total fare for a given distance -/
def calculate_fare (fare : TaxiFare) (distance : ℝ) : ℝ :=
  fare.start_fee + fare.per_mile_rate * distance

theorem bridge_toll_fee_calculation :
  let mike_fare : TaxiFare := { start_fee := 2.50, per_mile_rate := 0.25 }
  let annie_fare : TaxiFare := { start_fee := 2.50, per_mile_rate := 0.25 }
  let mike_distance : ℝ := 36
  let annie_distance : ℝ := 16
  let mike_total : ℝ := calculate_fare mike_fare mike_distance
  let annie_base : ℝ := calculate_fare annie_fare annie_distance
  let bridge_toll : ℝ := mike_total - annie_base
  bridge_toll = 5 := by sorry

end NUMINAMATH_CALUDE_bridge_toll_fee_calculation_l3270_327004


namespace NUMINAMATH_CALUDE_ellipse_focus_l3270_327016

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  center : Point
  majorAxisEnd1 : Point
  majorAxisEnd2 : Point
  minorAxisEnd1 : Point
  minorAxisEnd2 : Point

/-- Theorem: The focus with greater x-coordinate of the given ellipse -/
theorem ellipse_focus (e : Ellipse) 
  (h1 : e.center = ⟨4, -1⟩)
  (h2 : e.majorAxisEnd1 = ⟨0, -1⟩)
  (h3 : e.majorAxisEnd2 = ⟨8, -1⟩)
  (h4 : e.minorAxisEnd1 = ⟨4, 2⟩)
  (h5 : e.minorAxisEnd2 = ⟨4, -4⟩) :
  ∃ (focus : Point), focus.x = 4 + Real.sqrt 7 ∧ focus.y = -1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_focus_l3270_327016


namespace NUMINAMATH_CALUDE_add_twice_equals_thrice_l3270_327011

theorem add_twice_equals_thrice (a : ℝ) : a + 2 * a = 3 * a := by
  sorry

end NUMINAMATH_CALUDE_add_twice_equals_thrice_l3270_327011


namespace NUMINAMATH_CALUDE_smallest_square_side_exists_valid_division_5_l3270_327057

/-- Represents a square with integer side length -/
structure Square where
  side : ℕ

/-- Represents a division of a square into smaller squares -/
structure SquareDivision where
  original : Square
  parts : List Square
  sum_areas : (parts.map (λ s => s.side ^ 2)).sum = original.side ^ 2

/-- The property we want to prove -/
def is_valid_division (d : SquareDivision) : Prop :=
  d.parts.length = 15 ∧
  (d.parts.filter (λ s => s.side = 1)).length ≥ 12

/-- The main theorem -/
theorem smallest_square_side :
  ∀ d : SquareDivision, is_valid_division d → d.original.side ≥ 5 :=
by sorry

/-- The existence of a valid division with side 5 -/
theorem exists_valid_division_5 :
  ∃ d : SquareDivision, d.original.side = 5 ∧ is_valid_division d :=
by sorry

end NUMINAMATH_CALUDE_smallest_square_side_exists_valid_division_5_l3270_327057


namespace NUMINAMATH_CALUDE_cubic_roots_relation_l3270_327064

theorem cubic_roots_relation (a b c r s t : ℝ) : 
  (∀ x, x^3 + 5*x^2 + 6*x - 13 = 0 ↔ x = a ∨ x = b ∨ x = c) →
  (∀ x, x^3 + r*x^2 + s*x + t = 0 ↔ x = a+1 ∨ x = b+1 ∨ x = c+1) →
  t = -15 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_relation_l3270_327064


namespace NUMINAMATH_CALUDE_set_equality_l3270_327086

theorem set_equality : 
  let M : Set ℝ := {3, 2}
  let N : Set ℝ := {x | x^2 - 5*x + 6 = 0}
  M = N := by sorry

end NUMINAMATH_CALUDE_set_equality_l3270_327086


namespace NUMINAMATH_CALUDE_subtract_multiply_real_l3270_327038

theorem subtract_multiply_real : 3.56 - 2.1 * 1.5 = 0.41 := by
  sorry

end NUMINAMATH_CALUDE_subtract_multiply_real_l3270_327038


namespace NUMINAMATH_CALUDE_expression_evaluation_l3270_327066

theorem expression_evaluation : 10 - 9 + 8 * 7^2 + 6 - 5 * 4 + 3 - 2 = 380 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3270_327066


namespace NUMINAMATH_CALUDE_real_part_of_z_l3270_327072

theorem real_part_of_z (z : ℂ) (h : (3 + 4*I)*z = 5*(1 - I)) : 
  z.re = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_z_l3270_327072


namespace NUMINAMATH_CALUDE_sugar_solution_replacement_l3270_327032

/-- Calculates the sugar percentage of a final solution after replacing part of an original solution with a new solution. -/
def final_sugar_percentage (original_percentage : ℚ) (replaced_fraction : ℚ) (new_percentage : ℚ) : ℚ :=
  (1 - replaced_fraction) * original_percentage + replaced_fraction * new_percentage

/-- Theorem stating that replacing 1/4 of a 10% sugar solution with a 38% sugar solution results in a 17% sugar solution. -/
theorem sugar_solution_replacement :
  final_sugar_percentage (10 / 100) (1 / 4) (38 / 100) = 17 / 100 := by
  sorry

#eval final_sugar_percentage (10 / 100) (1 / 4) (38 / 100)

end NUMINAMATH_CALUDE_sugar_solution_replacement_l3270_327032


namespace NUMINAMATH_CALUDE_students_per_grade_l3270_327094

theorem students_per_grade (total_students : ℕ) (total_grades : ℕ) 
  (h1 : total_students = 22800) 
  (h2 : total_grades = 304) : 
  total_students / total_grades = 75 := by
  sorry

end NUMINAMATH_CALUDE_students_per_grade_l3270_327094


namespace NUMINAMATH_CALUDE_coupon_a_best_discount_correct_prices_l3270_327017

def coupon_a_discount (x : ℝ) : ℝ := 0.15 * x

def coupon_b_discount : ℝ := 30

def coupon_c_discount (x : ℝ) : ℝ := 0.22 * (x - 150)

theorem coupon_a_best_discount (x : ℝ) 
  (h1 : 200 < x) (h2 : x < 471.43) : 
  coupon_a_discount x > coupon_b_discount ∧ 
  coupon_a_discount x > coupon_c_discount x := by
  sorry

def price_list : List ℝ := [179.95, 199.95, 249.95, 299.95, 349.95]

theorem correct_prices (p : ℝ) (h : p ∈ price_list) :
  (200 < p ∧ p < 471.43) ↔ (p = 249.95 ∨ p = 299.95 ∨ p = 349.95) := by
  sorry

end NUMINAMATH_CALUDE_coupon_a_best_discount_correct_prices_l3270_327017


namespace NUMINAMATH_CALUDE_two_fifths_in_twice_one_tenth_l3270_327009

theorem two_fifths_in_twice_one_tenth : (2 * (1 / 10)) / (2 / 5) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_two_fifths_in_twice_one_tenth_l3270_327009


namespace NUMINAMATH_CALUDE_hannah_shopping_cost_hannah_spent_65_dollars_l3270_327075

theorem hannah_shopping_cost : ℕ → ℕ → ℕ → ℕ → ℕ → Prop :=
  fun num_sweatshirts sweatshirt_cost num_tshirts tshirt_cost total_cost =>
    num_sweatshirts * sweatshirt_cost + num_tshirts * tshirt_cost = total_cost

theorem hannah_spent_65_dollars :
  hannah_shopping_cost 3 15 2 10 65 := by
  sorry

end NUMINAMATH_CALUDE_hannah_shopping_cost_hannah_spent_65_dollars_l3270_327075


namespace NUMINAMATH_CALUDE_balloon_difference_l3270_327022

/-- The number of balloons Allan brought to the park -/
def allan_initial : ℕ := 2

/-- The number of balloons Allan bought at the park -/
def allan_bought : ℕ := 3

/-- The number of balloons Jake brought to the park -/
def jake_balloons : ℕ := 6

/-- The total number of balloons Allan had in the park -/
def allan_total : ℕ := allan_initial + allan_bought

theorem balloon_difference : jake_balloons - allan_total = 1 := by
  sorry

end NUMINAMATH_CALUDE_balloon_difference_l3270_327022


namespace NUMINAMATH_CALUDE_circle_touches_angle_sides_l3270_327073

-- Define the angle
def Angle : Type := sorry

-- Define a circle
structure Circle (α : Type) where
  center : α
  radius : ℝ

-- Define a point
structure Point where
  x : ℝ
  y : ℝ

-- Define the property of a circle touching a line
def touches_line (c : Circle Point) (l : Set Point) : Prop := sorry

-- Define the property of two circles touching each other
def circles_touch (c1 c2 : Circle Point) : Prop := sorry

-- Define the circle with diameter AB
def circle_with_diameter (A B : Point) : Circle Point := sorry

-- Define the sides of an angle
def sides_of_angle (a : Angle) : Set (Set Point) := sorry

theorem circle_touches_angle_sides 
  (θ : Angle) 
  (A B : Point) 
  (c1 c2 : Circle Point) 
  (h1 : c1.center = A)
  (h2 : c2.center = B)
  (h3 : ∀ s ∈ sides_of_angle θ, touches_line c1 s ∧ touches_line c2 s)
  (h4 : circles_touch c1 c2) :
  ∀ s ∈ sides_of_angle θ, touches_line (circle_with_diameter A B) s := by
  sorry

end NUMINAMATH_CALUDE_circle_touches_angle_sides_l3270_327073


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l3270_327006

/-- The decimal representation of a repeating decimal ending in 6 -/
def S : ℚ := 0.666666

/-- Theorem stating that the decimal 0.666... is equal to 2/3 -/
theorem decimal_to_fraction : S = 2/3 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l3270_327006


namespace NUMINAMATH_CALUDE_inverse_g_sum_l3270_327026

-- Define the function g
def g (x : ℝ) : ℝ := x^3 * |x|

-- State the theorem
theorem inverse_g_sum : 
  ∃ (a b : ℝ), g a = 8 ∧ g b = -64 ∧ a + b = Real.sqrt 2 - 2 :=
sorry

end NUMINAMATH_CALUDE_inverse_g_sum_l3270_327026


namespace NUMINAMATH_CALUDE_coffee_mixture_price_l3270_327012

/-- The price of the second type of coffee bean -/
def second_coffee_price : ℝ := 36

/-- The total weight of the mixture in pounds -/
def total_mixture_weight : ℝ := 100

/-- The selling price of the mixture per pound -/
def mixture_price : ℝ := 11.25

/-- The price of the first type of coffee bean per pound -/
def first_coffee_price : ℝ := 9

/-- The weight of each type of coffee bean in the mixture -/
def each_coffee_weight : ℝ := 25

theorem coffee_mixture_price :
  second_coffee_price * each_coffee_weight +
  first_coffee_price * each_coffee_weight =
  mixture_price * total_mixture_weight :=
by sorry

end NUMINAMATH_CALUDE_coffee_mixture_price_l3270_327012


namespace NUMINAMATH_CALUDE_cube_root_of_one_sixty_fourth_l3270_327088

theorem cube_root_of_one_sixty_fourth (x : ℝ) : x^3 = 1/64 → x = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_one_sixty_fourth_l3270_327088


namespace NUMINAMATH_CALUDE_cone_surface_area_l3270_327019

/-- The surface area of a cone with base radius 1 and height √3 is 3π. -/
theorem cone_surface_area : 
  let r : ℝ := 1
  let h : ℝ := Real.sqrt 3
  let l : ℝ := Real.sqrt (r^2 + h^2)
  let surface_area : ℝ := π * r^2 + π * r * l
  surface_area = 3 * π := by sorry

end NUMINAMATH_CALUDE_cone_surface_area_l3270_327019


namespace NUMINAMATH_CALUDE_ellipse_condition_l3270_327099

-- Define the condition for an ellipse with foci on the x-axis
def is_ellipse_x_axis (m n : ℝ) : Prop :=
  ∃ (a b : ℝ), a > b ∧ a > 0 ∧ b > 0 ∧ m = 1 / (a^2) ∧ n = 1 / (b^2)

-- State the theorem
theorem ellipse_condition (m n : ℝ) :
  is_ellipse_x_axis m n ↔ n > m ∧ m > 0 :=
sorry

end NUMINAMATH_CALUDE_ellipse_condition_l3270_327099


namespace NUMINAMATH_CALUDE_recurring_decimal_to_fraction_l3270_327013

theorem recurring_decimal_to_fraction :
  ∃ (n d : ℕ), n > 0 ∧ d > 0 ∧ (n.gcd d = 1) ∧
  (7 + 318 / 999 : ℚ) = n / d :=
sorry

end NUMINAMATH_CALUDE_recurring_decimal_to_fraction_l3270_327013


namespace NUMINAMATH_CALUDE_max_value_ab_l3270_327069

theorem max_value_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (1 / ((2 * a + b) * b)) + (2 / ((2 * b + a) * a)) = 1) :
  ab ≤ 2 - (2 * Real.sqrt 2) / 3 ∧ 
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 
    (1 / ((2 * a₀ + b₀) * b₀)) + (2 / ((2 * b₀ + a₀) * a₀)) = 1 ∧
    a₀ * b₀ = 2 - (2 * Real.sqrt 2) / 3 :=
sorry

end NUMINAMATH_CALUDE_max_value_ab_l3270_327069


namespace NUMINAMATH_CALUDE_convex_ngon_regions_l3270_327003

/-- The number of regions in a convex n-gon divided by its diagonals -/
def num_regions (n : ℕ) : ℚ :=
  (n^4 - 6*n^3 + 23*n^2 - 36*n + 24) / 24

/-- Theorem: For a convex n-gon (n ≥ 4) with all its diagonals drawn and 
    no three diagonals intersecting at the same point, the number of regions 
    into which the n-gon is divided is (n^4 - 6n^3 + 23n^2 - 36n + 24) / 24 -/
theorem convex_ngon_regions (n : ℕ) (h : n ≥ 4) :
  num_regions n = (n^4 - 6*n^3 + 23*n^2 - 36*n + 24) / 24 :=
by sorry

end NUMINAMATH_CALUDE_convex_ngon_regions_l3270_327003


namespace NUMINAMATH_CALUDE_zoo_animals_l3270_327097

theorem zoo_animals (lions : ℕ) (penguins : ℕ) : 
  lions = 30 →
  11 * lions = 3 * penguins →
  penguins - lions = 80 := by
  sorry

end NUMINAMATH_CALUDE_zoo_animals_l3270_327097


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3270_327002

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 6 * a 10 + a 3 * a 5 = 41 →
  a 4 * a 8 = 5 →
  a 4 + a 8 = Real.sqrt 51 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3270_327002


namespace NUMINAMATH_CALUDE_division_problem_l3270_327091

theorem division_problem (a b q : ℕ) (h1 : a - b = 1370) (h2 : a = 1626) (h3 : a = b * q + 15) : q = 6 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3270_327091


namespace NUMINAMATH_CALUDE_platform_length_l3270_327084

/-- The length of a platform given train parameters --/
theorem platform_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 250 →
  train_speed_kmh = 55 →
  crossing_time = 35.99712023038157 →
  ∃ (platform_length : ℝ), platform_length = 300 := by
  sorry

#check platform_length

end NUMINAMATH_CALUDE_platform_length_l3270_327084


namespace NUMINAMATH_CALUDE_bank_account_withdrawal_l3270_327025

theorem bank_account_withdrawal (initial_balance : ℚ) : 
  initial_balance > 0 →
  let remaining_balance := initial_balance - 400
  let deposit := (1 / 4) * remaining_balance
  let final_balance := remaining_balance + deposit
  final_balance = 750 →
  400 / initial_balance = 2 / 5 := by
sorry

end NUMINAMATH_CALUDE_bank_account_withdrawal_l3270_327025


namespace NUMINAMATH_CALUDE_soccer_league_female_fraction_l3270_327033

theorem soccer_league_female_fraction :
  ∀ (male_last_year female_last_year : ℕ)
    (total_this_year : ℚ)
    (male_this_year female_this_year : ℚ),
  male_last_year = 15 →
  total_this_year = 1.15 * (male_last_year + female_last_year) →
  male_this_year = 1.1 * male_last_year →
  female_this_year = 2 * female_last_year →
  female_this_year / total_this_year = 5 / 51 :=
by sorry

end NUMINAMATH_CALUDE_soccer_league_female_fraction_l3270_327033


namespace NUMINAMATH_CALUDE_system_always_solvable_l3270_327036

/-- Given a system of linear equations:
    ax + by = c - 1
    (a+5)x + (b+3)y = c + 1
    This theorem states that for the system to always have a solution
    for any real a and b, c must equal (2a + 5) / 5. -/
theorem system_always_solvable (a b c : ℝ) :
  (∀ x y : ℝ, a * x + b * y = c - 1 ∧ (a + 5) * x + (b + 3) * y = c + 1) ↔
  c = (2 * a + 5) / 5 := by
  sorry


end NUMINAMATH_CALUDE_system_always_solvable_l3270_327036


namespace NUMINAMATH_CALUDE_perfect_square_binomial_l3270_327044

theorem perfect_square_binomial (x : ℝ) :
  ∃! a : ℝ, ∃ r s : ℝ, 
    a * x^2 + 18 * x + 16 = (r * x + s)^2 ∧ 
    a = (81 : ℝ) / 16 :=
sorry

end NUMINAMATH_CALUDE_perfect_square_binomial_l3270_327044


namespace NUMINAMATH_CALUDE_complex_power_difference_l3270_327095

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_difference (h : i^2 = -1) : (1 + i)^10 - (1 - i)^10 = 64 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_power_difference_l3270_327095


namespace NUMINAMATH_CALUDE_race_theorem_l3270_327034

/-- A race with 40 kids where some finish under 6 minutes, some under 8 minutes, and the rest take longer. -/
structure Race where
  total_kids : ℕ
  under_6_min : ℕ
  under_8_min : ℕ
  over_certain_min : ℕ

/-- The race satisfies the given conditions. -/
def race_conditions (r : Race) : Prop :=
  r.total_kids = 40 ∧
  r.under_6_min = (10 : ℕ) * r.total_kids / 100 ∧
  r.under_8_min = 3 * r.under_6_min ∧
  r.over_certain_min = 4 ∧
  r.over_certain_min = (r.total_kids - (r.under_6_min + r.under_8_min)) / 6

/-- The theorem stating that the number of kids who take more than a certain number of minutes is 4. -/
theorem race_theorem (r : Race) (h : race_conditions r) : r.over_certain_min = 4 := by
  sorry


end NUMINAMATH_CALUDE_race_theorem_l3270_327034


namespace NUMINAMATH_CALUDE_object_speed_l3270_327029

/-- An object traveling 10800 feet in one hour has a speed of 3 feet per second. -/
theorem object_speed (distance : ℝ) (time_in_seconds : ℝ) (h1 : distance = 10800) (h2 : time_in_seconds = 3600) :
  distance / time_in_seconds = 3 := by
  sorry

end NUMINAMATH_CALUDE_object_speed_l3270_327029


namespace NUMINAMATH_CALUDE_ellen_painted_twenty_vines_l3270_327079

/-- Represents the time in minutes required to paint different types of flowers and vines. -/
structure PaintingTimes where
  lily : ℕ
  rose : ℕ
  orchid : ℕ
  vine : ℕ

/-- Represents the number of each type of flower and vine painted. -/
structure FlowerCounts where
  lilies : ℕ
  roses : ℕ
  orchids : ℕ
  vines : ℕ

/-- Calculates the total time spent painting given the painting times and flower counts. -/
def totalPaintingTime (times : PaintingTimes) (counts : FlowerCounts) : ℕ :=
  times.lily * counts.lilies + times.rose * counts.roses + 
  times.orchid * counts.orchids + times.vine * counts.vines

/-- Theorem stating that Ellen painted 20 vines given the problem conditions. -/
theorem ellen_painted_twenty_vines 
  (times : PaintingTimes)
  (counts : FlowerCounts)
  (h1 : times.lily = 5)
  (h2 : times.rose = 7)
  (h3 : times.orchid = 3)
  (h4 : times.vine = 2)
  (h5 : counts.lilies = 17)
  (h6 : counts.roses = 10)
  (h7 : counts.orchids = 6)
  (h8 : totalPaintingTime times counts = 213) :
  counts.vines = 20 := by
  sorry

end NUMINAMATH_CALUDE_ellen_painted_twenty_vines_l3270_327079


namespace NUMINAMATH_CALUDE_lunch_cost_with_tip_l3270_327037

theorem lunch_cost_with_tip (total_cost : ℝ) (tip_percentage : ℝ) (original_cost : ℝ) :
  total_cost = 58.075 ∧
  tip_percentage = 0.15 ∧
  total_cost = original_cost * (1 + tip_percentage) →
  original_cost = 50.5 := by
sorry

end NUMINAMATH_CALUDE_lunch_cost_with_tip_l3270_327037


namespace NUMINAMATH_CALUDE_binomial_sum_identity_l3270_327007

theorem binomial_sum_identity (p q n : ℕ+) :
  (∑' k : ℕ, (Nat.choose (p.val + k) p.val) * (Nat.choose (q.val + n.val - k) q.val)) =
  Nat.choose (p.val + q.val + n.val + 1) (p.val + q.val + 1) :=
sorry

end NUMINAMATH_CALUDE_binomial_sum_identity_l3270_327007


namespace NUMINAMATH_CALUDE_greatest_k_value_l3270_327042

theorem greatest_k_value (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
    x^2 + k*x + 7 = 0 ∧ 
    y^2 + k*y + 7 = 0 ∧ 
    |x - y| = Real.sqrt 85) →
  k ≤ Real.sqrt 113 :=
by sorry

end NUMINAMATH_CALUDE_greatest_k_value_l3270_327042


namespace NUMINAMATH_CALUDE_roots_of_Q_are_fifth_powers_of_roots_of_P_l3270_327067

-- Define the polynomial P
def P (x : ℂ) : ℂ := x^3 - 3*x + 1

-- Define the polynomial Q
def Q (y : ℂ) : ℂ := y^3 + 15*y^2 - 198*y + 1

-- Theorem statement
theorem roots_of_Q_are_fifth_powers_of_roots_of_P :
  ∀ (α : ℂ), P α = 0 → ∃ (β : ℂ), Q (β^5) = 0 ∧ P β = 0 :=
by sorry

end NUMINAMATH_CALUDE_roots_of_Q_are_fifth_powers_of_roots_of_P_l3270_327067


namespace NUMINAMATH_CALUDE_sin_negative_three_pi_halves_l3270_327005

theorem sin_negative_three_pi_halves : Real.sin (-3 * π / 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_three_pi_halves_l3270_327005


namespace NUMINAMATH_CALUDE_balance_proof_l3270_327076

/-- The weight of a single diamond -/
def diamond_weight : ℝ := sorry

/-- The weight of a single emerald -/
def emerald_weight : ℝ := sorry

/-- The number of diamonds that balance one emerald -/
def diamonds_per_emerald : ℕ := sorry

theorem balance_proof :
  -- Condition 1 and 2: 9 diamonds balance 4 emeralds
  9 * diamond_weight = 4 * emerald_weight →
  -- Condition 3: 9 diamonds + 1 emerald balance 4 emeralds
  9 * diamond_weight + emerald_weight = 4 * emerald_weight →
  -- Conclusion: 3 diamonds balance 1 emerald
  diamonds_per_emerald = 3 := by sorry

end NUMINAMATH_CALUDE_balance_proof_l3270_327076


namespace NUMINAMATH_CALUDE_floor_times_x_eq_152_l3270_327087

theorem floor_times_x_eq_152 : ∃ x : ℝ, (⌊x⌋ : ℝ) * x = 152 ∧ x = 38 / 3 := by
  sorry

end NUMINAMATH_CALUDE_floor_times_x_eq_152_l3270_327087


namespace NUMINAMATH_CALUDE_pqr_product_l3270_327063

theorem pqr_product (p q r : ℝ) (h_distinct : p ≠ q ∧ q ≠ r ∧ r ≠ p)
  (h_nonzero : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0)
  (h_eq : p^2 + 2/q = q^2 + 2/r ∧ q^2 + 2/r = r^2 + 2/p) :
  |p * q * r| = 2 := by
  sorry

end NUMINAMATH_CALUDE_pqr_product_l3270_327063


namespace NUMINAMATH_CALUDE_expression_value_at_negative_three_l3270_327093

theorem expression_value_at_negative_three :
  let x : ℤ := -3
  let expr := 5 * x - (3 * x - 2 * (2 * x - 3))
  expr = -24 := by sorry

end NUMINAMATH_CALUDE_expression_value_at_negative_three_l3270_327093


namespace NUMINAMATH_CALUDE_derivative_at_one_l3270_327020

theorem derivative_at_one (f : ℝ → ℝ) (h : ∀ x, f x = x^3 - 2 * (deriv f 1) * x) :
  deriv f 1 = 1 :=
sorry

end NUMINAMATH_CALUDE_derivative_at_one_l3270_327020


namespace NUMINAMATH_CALUDE_fraction_equality_l3270_327010

theorem fraction_equality (a b c x : ℝ) 
  (hx : x = a / b) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0) 
  (hab : a ≠ b) 
  (hbc : b ≠ c) 
  (hac : a ≠ c) : 
  (a + 2*b + 3*c) / (a - b - 3*c) = (b*(x + 2) + 3*c) / (b*(x - 1) - 3*c) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3270_327010


namespace NUMINAMATH_CALUDE_fraction_equality_l3270_327098

theorem fraction_equality (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) 
  (h3 : (4*x + 2*y) / (2*x - 4*y) = 3) : 
  (2*x + 4*y) / (4*x - 2*y) = 9/13 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3270_327098


namespace NUMINAMATH_CALUDE_work_completion_time_l3270_327031

/-- 
Given:
- A person B can do a work in 20 days
- Persons A and B together can do the work in 15 days

Prove that A can do the work alone in 60 days
-/
theorem work_completion_time (b_time : ℝ) (ab_time : ℝ) (a_time : ℝ) : 
  b_time = 20 → ab_time = 15 → a_time = 60 → 
  1 / a_time + 1 / b_time = 1 / ab_time := by
  sorry

#check work_completion_time

end NUMINAMATH_CALUDE_work_completion_time_l3270_327031


namespace NUMINAMATH_CALUDE_price_reduction_theorem_l3270_327083

/-- Given three consecutive price reductions, calculates the overall percentage reduction -/
def overall_reduction (r1 r2 r3 : ℝ) : ℝ :=
  (1 - (1 - r1) * (1 - r2) * (1 - r3)) * 100

/-- Theorem stating that the overall reduction after 25%, 20%, and 15% reductions is 49% -/
theorem price_reduction_theorem : 
  overall_reduction 0.25 0.20 0.15 = 49 := by
  sorry

#eval overall_reduction 0.25 0.20 0.15

end NUMINAMATH_CALUDE_price_reduction_theorem_l3270_327083


namespace NUMINAMATH_CALUDE_lcm_ratio_sum_l3270_327001

theorem lcm_ratio_sum (a b : ℕ+) : 
  Nat.lcm a b = 30 → 
  a.val * 3 = b.val * 2 → 
  a + b = 50 := by
sorry

end NUMINAMATH_CALUDE_lcm_ratio_sum_l3270_327001


namespace NUMINAMATH_CALUDE_function_value_at_two_l3270_327023

theorem function_value_at_two (f : ℝ → ℝ) (h : ∀ x, f (x + 1) = x^2 + 1) : f 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_two_l3270_327023


namespace NUMINAMATH_CALUDE_vector_norm_equation_solutions_l3270_327070

theorem vector_norm_equation_solutions :
  let v : ℝ × ℝ := (3, -4)
  let w : ℝ × ℝ := (5, 8)
  let norm_eq : ℝ → Prop := λ k => ‖k • v - w‖ = 5 * Real.sqrt 13
  ∀ k : ℝ, norm_eq k ↔ (k = 123 / 50 ∨ k = -191 / 50) :=
by sorry

end NUMINAMATH_CALUDE_vector_norm_equation_solutions_l3270_327070


namespace NUMINAMATH_CALUDE_volleyball_count_l3270_327008

theorem volleyball_count : ∃ (x y z : ℕ),
  x + y + z = 20 ∧
  60 * x + 30 * y + 10 * z = 330 ∧
  z = 15 := by
sorry

end NUMINAMATH_CALUDE_volleyball_count_l3270_327008


namespace NUMINAMATH_CALUDE_penguin_arrangements_l3270_327060

def word_length : ℕ := 7
def repeated_letter_count : ℕ := 2

theorem penguin_arrangements :
  (word_length.factorial / repeated_letter_count.factorial) = 2520 := by
  sorry

end NUMINAMATH_CALUDE_penguin_arrangements_l3270_327060


namespace NUMINAMATH_CALUDE_min_value_f_l3270_327054

theorem min_value_f (x y : ℝ) (hx : 0 ≤ x ∧ x ≤ π) (hy : 0 ≤ y ∧ y ≤ 1) :
  (2 * y - 1) * Real.sin x + (1 - y) * Real.sin ((1 - y) * x) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_min_value_f_l3270_327054


namespace NUMINAMATH_CALUDE_triangle_law_of_sines_l3270_327068

theorem triangle_law_of_sines (A B C : Real) (a b c : Real) :
  A = π / 6 →
  a = Real.sqrt 2 →
  b / Real.sin B = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_law_of_sines_l3270_327068


namespace NUMINAMATH_CALUDE_abc_product_l3270_327080

theorem abc_product (a b c : ℝ) (h1 : b + c = 3) (h2 : c + a = 6) (h3 : a + b = 7) : a * b * c = 10 := by
  sorry

end NUMINAMATH_CALUDE_abc_product_l3270_327080


namespace NUMINAMATH_CALUDE_metallic_sheet_length_l3270_327046

/-- Proves that a rectangular sheet with one side 36 m, when cut to form a box of volume 3780 m³, has an original length of 48 m -/
theorem metallic_sheet_length (L : ℝ) : 
  L > 0 → 
  (L - 6) * (36 - 6) * 3 = 3780 → 
  L = 48 :=
by sorry

end NUMINAMATH_CALUDE_metallic_sheet_length_l3270_327046


namespace NUMINAMATH_CALUDE_new_oranges_added_l3270_327051

theorem new_oranges_added (initial : ℕ) (thrown_away : ℕ) (final : ℕ) : 
  initial = 31 → thrown_away = 9 → final = 60 → final - (initial - thrown_away) = 38 := by
  sorry

end NUMINAMATH_CALUDE_new_oranges_added_l3270_327051


namespace NUMINAMATH_CALUDE_triangle_inequality_l3270_327018

/-- For any triangle with sides a, b, c and area S, 
    the inequality a^2 + b^2 + c^2 - 1/2(|a-b| + |b-c| + |c-a|)^2 ≥ 4√3 S holds. -/
theorem triangle_inequality (a b c S : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_area : S = Real.sqrt ((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c)) / 4) :
  a^2 + b^2 + c^2 - 1/2 * (|a - b| + |b - c| + |c - a|)^2 ≥ 4 * Real.sqrt 3 * S := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3270_327018


namespace NUMINAMATH_CALUDE_exam_score_97_impossible_l3270_327062

theorem exam_score_97_impossible :
  ¬ ∃ (correct unanswered : ℕ),
    correct + unanswered ≤ 20 ∧
    5 * correct + unanswered = 97 :=
by sorry

end NUMINAMATH_CALUDE_exam_score_97_impossible_l3270_327062


namespace NUMINAMATH_CALUDE_inverse_proportionality_l3270_327074

/-- Given that x is inversely proportional to y, prove that if x = 5 when y = -4, then x = -2 when y = 10 -/
theorem inverse_proportionality (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : 5 * (-4) = k) :
  10 * x = k → x = -2 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportionality_l3270_327074


namespace NUMINAMATH_CALUDE_target_hit_probability_l3270_327041

theorem target_hit_probability (prob_A prob_B : ℝ) : 
  prob_A = 1/2 → 
  prob_B = 1/3 → 
  1 - (1 - prob_A) * (1 - prob_B) = 2/3 := by
sorry

end NUMINAMATH_CALUDE_target_hit_probability_l3270_327041


namespace NUMINAMATH_CALUDE_fraction_value_at_2017_l3270_327053

theorem fraction_value_at_2017 :
  let x : ℤ := 2017
  (x^2 + 6*x + 9) / (x + 3) = 2020 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_at_2017_l3270_327053


namespace NUMINAMATH_CALUDE_complex_arithmetic_equality_l3270_327092

theorem complex_arithmetic_equality : (18 * 23 - 24 * 17) / 3 + 5 = 7 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equality_l3270_327092


namespace NUMINAMATH_CALUDE_nested_sqrt_evaluation_l3270_327059

theorem nested_sqrt_evaluation (x : ℝ) (h : x ≥ 0) :
  Real.sqrt (x^2 * Real.sqrt (x^2 * Real.sqrt (x^2))) = x^(7/4) := by
  sorry

end NUMINAMATH_CALUDE_nested_sqrt_evaluation_l3270_327059


namespace NUMINAMATH_CALUDE_function_not_monotonic_iff_a_in_range_l3270_327030

/-- The function f(x) is not monotonic on the interval (0, 4) if and only if a is in the open interval (-4, 9/4) -/
theorem function_not_monotonic_iff_a_in_range (a : ℝ) :
  (∃ x y, x ∈ (Set.Ioo 0 4) ∧ y ∈ (Set.Ioo 0 4) ∧ x < y ∧
    ((1/3 * x^3 - 3/2 * x^2 + a*x + 4) > (1/3 * y^3 - 3/2 * y^2 + a*y + 4) ∨
     (1/3 * x^3 - 3/2 * x^2 + a*x + 4) < (1/3 * y^3 - 3/2 * y^2 + a*y + 4)))
  ↔ a ∈ Set.Ioo (-4) (9/4) :=
by sorry

end NUMINAMATH_CALUDE_function_not_monotonic_iff_a_in_range_l3270_327030


namespace NUMINAMATH_CALUDE_certain_number_proof_l3270_327049

theorem certain_number_proof (x : ℝ) : (15 * x) / 100 = 0.04863 → x = 0.3242 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3270_327049


namespace NUMINAMATH_CALUDE_line_segment_param_sum_squares_l3270_327055

/-- 
Given a line segment from (-3,5) to (4,15) parameterized by x = at + b and y = ct + d,
where -1 ≤ t ≤ 2 and t = -1 corresponds to (-3,5), prove that a² + b² + c² + d² = 790/9
-/
theorem line_segment_param_sum_squares (a b c d : ℚ) : 
  (∀ t, -1 ≤ t → t ≤ 2 → ∃ x y, x = a * t + b ∧ y = c * t + d) →
  a * (-1) + b = -3 →
  c * (-1) + d = 5 →
  a * 2 + b = 4 →
  c * 2 + d = 15 →
  a^2 + b^2 + c^2 + d^2 = 790/9 := by
sorry

end NUMINAMATH_CALUDE_line_segment_param_sum_squares_l3270_327055


namespace NUMINAMATH_CALUDE_is_circle_center_l3270_327077

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x + 8*y + 9 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (3, -4)

/-- Theorem stating that the given point is the center of the circle -/
theorem is_circle_center :
  ∀ x y : ℝ, circle_equation x y ↔ (x - circle_center.1)^2 + (y - circle_center.2)^2 = 16 :=
by sorry

end NUMINAMATH_CALUDE_is_circle_center_l3270_327077


namespace NUMINAMATH_CALUDE_intersection_when_a_neg_two_subset_condition_l3270_327047

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | 2*a - 1 ≤ x ∧ x ≤ a + 3}
def B : Set ℝ := {x | x < -1 ∨ x > 5}

-- Theorem 1: Intersection of A and B when a = -2
theorem intersection_when_a_neg_two :
  A (-2) ∩ B = {x | -5 ≤ x ∧ x < -1} := by sorry

-- Theorem 2: Condition for A to be a subset of B
theorem subset_condition (a : ℝ) :
  A a ⊆ B ↔ a ≤ -4 ∨ a ≥ 3 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_neg_two_subset_condition_l3270_327047


namespace NUMINAMATH_CALUDE_meeting_point_coordinates_l3270_327090

/-- The point two-thirds of the way from one point to another -/
def two_thirds_point (x₁ y₁ x₂ y₂ : ℚ) : ℚ × ℚ :=
  (x₁ + 2/3 * (x₂ - x₁), y₁ + 2/3 * (y₂ - y₁))

/-- Prove that the meeting point is at (14/3, 11/3) -/
theorem meeting_point_coordinates :
  two_thirds_point 10 (-3) 2 7 = (14/3, 11/3) := by
  sorry

#check meeting_point_coordinates

end NUMINAMATH_CALUDE_meeting_point_coordinates_l3270_327090


namespace NUMINAMATH_CALUDE_remainder_problem_l3270_327078

theorem remainder_problem (k : Nat) (h : k > 0) :
  (90 % (k^2) = 6) → (130 % k = 4) := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3270_327078


namespace NUMINAMATH_CALUDE_violets_to_carnations_ratio_l3270_327081

/-- Represents the number of each type of flower in the shop -/
structure FlowerShop where
  violets : ℕ
  carnations : ℕ
  tulips : ℕ
  roses : ℕ

/-- The conditions of the flower shop -/
def FlowerShopConditions (shop : FlowerShop) : Prop :=
  shop.tulips = shop.violets / 4 ∧
  shop.roses = shop.tulips ∧
  shop.carnations = (2 * (shop.violets + shop.carnations + shop.tulips + shop.roses)) / 3

/-- The theorem stating the ratio of violets to carnations -/
theorem violets_to_carnations_ratio (shop : FlowerShop) 
  (h : FlowerShopConditions shop) : 
  shop.violets = shop.carnations / 3 := by
  sorry

#check violets_to_carnations_ratio

end NUMINAMATH_CALUDE_violets_to_carnations_ratio_l3270_327081


namespace NUMINAMATH_CALUDE_equidistant_point_on_z_axis_l3270_327043

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculate the squared distance between two points -/
def squaredDistance (p1 p2 : Point3D) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2

/-- The theorem stating that C(0, 0, 1) is equidistant from A(1, 0, 2) and B(1, 1, 1) -/
theorem equidistant_point_on_z_axis : 
  let A : Point3D := ⟨1, 0, 2⟩
  let B : Point3D := ⟨1, 1, 1⟩
  let C : Point3D := ⟨0, 0, 1⟩
  squaredDistance A C = squaredDistance B C := by
  sorry

end NUMINAMATH_CALUDE_equidistant_point_on_z_axis_l3270_327043


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3270_327024

theorem quadratic_equation_roots (m : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, m * x^2 + 2*(m+1)*x + (m-1) = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ ≠ x₂ →
  x₁^2 + x₂^2 = 8 →
  m > -1/2 →
  m ≠ 0 →
  m = (6 + 2*Real.sqrt 33) / 8 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3270_327024


namespace NUMINAMATH_CALUDE_quadratic_range_l3270_327085

theorem quadratic_range (x : ℝ) (h : x^2 - 4*x + 3 < 0) :
  8 < x^2 + 4*x + 3 ∧ x^2 + 4*x + 3 < 24 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_range_l3270_327085


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l3270_327040

theorem rectangle_diagonal (perimeter : ℝ) (length_ratio width_ratio : ℕ) 
  (h_perimeter : perimeter = 72) 
  (h_ratio : length_ratio = 5 ∧ width_ratio = 4) : 
  ∃ (diagonal : ℝ), diagonal = 4 * Real.sqrt 41 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l3270_327040


namespace NUMINAMATH_CALUDE_symmetric_functions_properties_l3270_327048

/-- Given a > 1, f(x) is symmetric to g(x) = 4 - a^|x-2| - 2*a^(x-2) w.r.t (1, 2) -/
def SymmetricFunctions (a : ℝ) (f : ℝ → ℝ) : Prop :=
  a > 1 ∧ ∀ x y, f x = y ↔ 4 - a^|2-x| - 2*a^(2-x) = 4 - y

theorem symmetric_functions_properties {a : ℝ} {f : ℝ → ℝ} 
  (h : SymmetricFunctions a f) :
  (∀ x, f x = a^|x| + 2*a^(-x)) ∧ 
  (∀ m, (∃ x₁ x₂, 0 < x₁ ∧ 0 < x₂ ∧ x₁ ≠ x₂ ∧ f x₁ = m ∧ f x₂ = m) ↔ 
    2*(2:ℝ)^(1/2) < m ∧ m < 3) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_functions_properties_l3270_327048


namespace NUMINAMATH_CALUDE_selection_ways_10_people_l3270_327058

/-- The number of ways to choose a president, vice-president, and 2-person committee from n people -/
def selection_ways (n : ℕ) : ℕ :=
  n * (n - 1) * (Nat.choose (n - 2) 2)

/-- Theorem stating that there are 2520 ways to make the selection from 10 people -/
theorem selection_ways_10_people :
  selection_ways 10 = 2520 := by
  sorry

end NUMINAMATH_CALUDE_selection_ways_10_people_l3270_327058


namespace NUMINAMATH_CALUDE_expression_evaluation_l3270_327082

theorem expression_evaluation : 
  2 * (7 ^ (1/3 : ℝ)) + 16 ^ (3/4 : ℝ) + (4 / (Real.sqrt 3 - 1)) ^ (0 : ℝ) + (-3) ^ (-1 : ℝ) = 44/3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3270_327082


namespace NUMINAMATH_CALUDE_parabola_solutions_l3270_327000

/-- The parabola y = ax² + bx + c passes through points (-1, 3) and (2, 3).
    The solutions of a(x-2)² - 3 = 2b - bx - c are 1 and 4. -/
theorem parabola_solutions (a b c : ℝ) : 
  (∀ x : ℝ, a * x^2 + b * x + c = 3 ↔ x = -1 ∨ x = 2) →
  (∀ x : ℝ, a * (x - 2)^2 - 3 = 2 * b - b * x - c ↔ x = 1 ∨ x = 4) :=
by sorry

end NUMINAMATH_CALUDE_parabola_solutions_l3270_327000


namespace NUMINAMATH_CALUDE_excess_students_equals_35_l3270_327015

/-- Represents a kindergarten classroom at Maple Ridge School -/
structure Classroom where
  students : Nat
  rabbits : Nat
  guinea_pigs : Nat

/-- The number of classrooms at Maple Ridge School -/
def num_classrooms : Nat := 5

/-- A standard classroom at Maple Ridge School -/
def standard_classroom : Classroom := {
  students := 15,
  rabbits := 3,
  guinea_pigs := 5
}

/-- The total number of students in all classrooms -/
def total_students : Nat := num_classrooms * standard_classroom.students

/-- The total number of rabbits in all classrooms -/
def total_rabbits : Nat := num_classrooms * standard_classroom.rabbits

/-- The total number of guinea pigs in all classrooms -/
def total_guinea_pigs : Nat := num_classrooms * standard_classroom.guinea_pigs

/-- 
Theorem: The sum of the number of students in excess of the number of pet rabbits 
and the number of guinea pigs in all 5 classrooms is equal to 35.
-/
theorem excess_students_equals_35 : 
  total_students - (total_rabbits + total_guinea_pigs) = 35 := by
  sorry

end NUMINAMATH_CALUDE_excess_students_equals_35_l3270_327015


namespace NUMINAMATH_CALUDE_product_of_cosines_l3270_327021

theorem product_of_cosines : 
  (1 + Real.cos (π / 6)) * (1 + Real.cos (π / 3)) * 
  (1 + Real.cos ((2 * π) / 3)) * (1 + Real.cos ((5 * π) / 6)) = 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_product_of_cosines_l3270_327021


namespace NUMINAMATH_CALUDE_division_problem_l3270_327014

theorem division_problem (n : ℕ) (h1 : n % 11 = 1) (h2 : n / 11 = 13) : n = 144 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3270_327014


namespace NUMINAMATH_CALUDE_modular_inverse_28_mod_29_l3270_327096

theorem modular_inverse_28_mod_29 : ∃ x : ℤ, (28 * x) % 29 = 1 :=
by
  use 28
  sorry

end NUMINAMATH_CALUDE_modular_inverse_28_mod_29_l3270_327096


namespace NUMINAMATH_CALUDE_sum_of_f_values_l3270_327045

noncomputable def f (x : ℝ) : ℝ := 2 / (2^x + 1) + Real.sin x

theorem sum_of_f_values : 
  f (-3) + f (-2) + f (-1) + f 0 + f 1 + f 2 + f 3 = 7 := by sorry

end NUMINAMATH_CALUDE_sum_of_f_values_l3270_327045


namespace NUMINAMATH_CALUDE_triangle_angle_A_l3270_327028

theorem triangle_angle_A (a b : ℝ) (B : ℝ) (h1 : a = Real.sqrt 3) (h2 : b = Real.sqrt 2) (h3 : B = π / 4) :
  let A := Real.arcsin ((a * Real.sin B) / b)
  A = π / 3 ∨ A = 2 * π / 3 :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_A_l3270_327028


namespace NUMINAMATH_CALUDE_camille_bird_counting_l3270_327035

theorem camille_bird_counting (cardinals : ℕ) 
  (h1 : cardinals > 0)
  (h2 : cardinals + 4 * cardinals + 2 * cardinals + (3 * cardinals + 1) = 31) :
  cardinals = 3 := by
sorry

end NUMINAMATH_CALUDE_camille_bird_counting_l3270_327035


namespace NUMINAMATH_CALUDE_divisor_problem_l3270_327039

def divisor_count (n : ℕ) : ℕ := (Nat.divisors n).card

theorem divisor_problem :
  (∃! k : ℕ, 2 ∣ k ∧ 9 ∣ k ∧ divisor_count k = 14) ∧
  (∃ k₁ k₂ : ℕ, k₁ ≠ k₂ ∧ 2 ∣ k₁ ∧ 9 ∣ k₁ ∧ divisor_count k₁ = 15 ∧
                2 ∣ k₂ ∧ 9 ∣ k₂ ∧ divisor_count k₂ = 15) ∧
  (¬ ∃ k : ℕ, 2 ∣ k ∧ 9 ∣ k ∧ divisor_count k = 17) :=
by sorry

end NUMINAMATH_CALUDE_divisor_problem_l3270_327039


namespace NUMINAMATH_CALUDE_system_solution_l3270_327056

-- Define the system of linear equations
def system (k : ℝ) (x y : ℝ) : Prop :=
  x - y = 9 * k ∧ x + y = 5 * k

-- Define the additional equation
def additional_eq (x y : ℝ) : Prop :=
  2 * x + 3 * y = 8

-- Theorem statement
theorem system_solution :
  ∀ k x y, system k x y → additional_eq x y → k = 1 ∧ x = 7 ∧ y = -2 :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3270_327056


namespace NUMINAMATH_CALUDE_power_two_plus_one_div_by_three_l3270_327089

theorem power_two_plus_one_div_by_three (n : ℕ) : 
  3 ∣ (2^n + 1) ↔ n % 2 = 1 := by
sorry

end NUMINAMATH_CALUDE_power_two_plus_one_div_by_three_l3270_327089


namespace NUMINAMATH_CALUDE_employee_pay_percentage_l3270_327050

/-- Given two employees A and B who are paid a total of 580 per week, 
    with B being paid 232 per week, prove that the percentage of A's pay 
    compared to B's pay is 150%. -/
theorem employee_pay_percentage (total_pay b_pay a_pay : ℚ) : 
  total_pay = 580 → 
  b_pay = 232 → 
  a_pay = total_pay - b_pay →
  (a_pay / b_pay) * 100 = 150 := by
  sorry

end NUMINAMATH_CALUDE_employee_pay_percentage_l3270_327050


namespace NUMINAMATH_CALUDE_min_value_fraction_l3270_327071

theorem min_value_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (4 / a + 9 / b) ≥ 25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_l3270_327071


namespace NUMINAMATH_CALUDE_binary_division_and_double_l3270_327065

def binary_number : ℕ := 3666 -- 111011010010₂ in decimal

theorem binary_division_and_double :
  (binary_number % 4) * 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_binary_division_and_double_l3270_327065
