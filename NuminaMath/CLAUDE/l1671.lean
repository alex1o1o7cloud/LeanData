import Mathlib

namespace NUMINAMATH_CALUDE_fixed_points_for_specific_values_range_for_two_distinct_fixed_points_l1671_167144

/-- The function f(x) = ax^2 + (b+1)x + b - 2 -/
def f (a b x : ℝ) : ℝ := a * x^2 + (b + 1) * x + b - 2

/-- A point x is a fixed point of f if f(x) = x -/
def is_fixed_point (a b x : ℝ) : Prop := f a b x = x

theorem fixed_points_for_specific_values :
  ∀ x : ℝ, is_fixed_point 2 (-2) x ↔ x = -1 ∨ x = 2 := by sorry

theorem range_for_two_distinct_fixed_points :
  ∀ a : ℝ, (∃ x y : ℝ, x ≠ y ∧ is_fixed_point a b x ∧ is_fixed_point a b y) →
  (0 < a ∧ a < 2) := by sorry

end NUMINAMATH_CALUDE_fixed_points_for_specific_values_range_for_two_distinct_fixed_points_l1671_167144


namespace NUMINAMATH_CALUDE_shopping_cost_calculation_l1671_167135

/-- Calculates the total cost of a shopping trip, including discounts and sales tax -/
theorem shopping_cost_calculation 
  (tshirt_price sweater_price jacket_price : ℚ)
  (jacket_discount sales_tax : ℚ)
  (tshirt_quantity sweater_quantity jacket_quantity : ℕ)
  (h1 : tshirt_price = 8)
  (h2 : sweater_price = 18)
  (h3 : jacket_price = 80)
  (h4 : jacket_discount = 1/10)
  (h5 : sales_tax = 1/20)
  (h6 : tshirt_quantity = 6)
  (h7 : sweater_quantity = 4)
  (h8 : jacket_quantity = 5) :
  let tshirt_cost := tshirt_quantity * tshirt_price
  let sweater_cost := sweater_quantity * sweater_price
  let jacket_cost := jacket_quantity * jacket_price * (1 - jacket_discount)
  let subtotal := tshirt_cost + sweater_cost + jacket_cost
  let total := subtotal * (1 + sales_tax)
  total = 504 := by sorry

end NUMINAMATH_CALUDE_shopping_cost_calculation_l1671_167135


namespace NUMINAMATH_CALUDE_work_completion_time_l1671_167109

/-- Given two workers A and B, where:
    - A and B together can complete a job in 6 days
    - A alone can complete the job in 14 days
    This theorem proves that B alone can complete the job in 10.5 days -/
theorem work_completion_time (work_rate_A : ℝ) (work_rate_B : ℝ) : 
  work_rate_A + work_rate_B = 1 / 6 →
  work_rate_A = 1 / 14 →
  1 / work_rate_B = 10.5 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l1671_167109


namespace NUMINAMATH_CALUDE_special_function_zero_location_l1671_167193

/-- A function f satisfying the given conditions -/
structure SpecialFunction (f : ℝ → ℝ) : Prop :=
  (decreasing : ∀ x₁ x₂ : ℝ, (x₁ - x₂) * (f x₁ - f x₂) < 0)

/-- The theorem statement -/
theorem special_function_zero_location
  (f : ℝ → ℝ) (hf : SpecialFunction f) (a b c d : ℝ)
  (h_order : c < b ∧ b < a)
  (h_product : f a * f b * f c < 0)
  (h_zero : f d = 0) :
  (d < c) ∨ (b < d ∧ d < a) :=
sorry

end NUMINAMATH_CALUDE_special_function_zero_location_l1671_167193


namespace NUMINAMATH_CALUDE_right_focus_coordinates_l1671_167198

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 36 - y^2 / 64 = 1

/-- The right focus of the hyperbola -/
def right_focus : ℝ × ℝ := (10, 0)

/-- Theorem: The right focus of the given hyperbola is (10, 0) -/
theorem right_focus_coordinates :
  ∀ (x y : ℝ), hyperbola_equation x y → right_focus = (10, 0) := by
  sorry

end NUMINAMATH_CALUDE_right_focus_coordinates_l1671_167198


namespace NUMINAMATH_CALUDE_average_speed_round_trip_l1671_167162

/-- Calculates the average speed for a round trip journey between two points -/
theorem average_speed_round_trip (d : ℝ) (uphill_speed downhill_speed : ℝ) 
  (h1 : uphill_speed > 0)
  (h2 : downhill_speed > 0)
  (h3 : uphill_speed = 60)
  (h4 : downhill_speed = 36) :
  (2 * d) / (d / uphill_speed + d / downhill_speed) = 45 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_round_trip_l1671_167162


namespace NUMINAMATH_CALUDE_possible_values_of_a_l1671_167167

def M : Set ℝ := {x | x^2 + x - 6 = 0}
def N (a : ℝ) : Set ℝ := {x | a * x + 2 = 0}

theorem possible_values_of_a :
  ∀ a : ℝ, (N a ⊆ M) ↔ (a = -1 ∨ a = 0 ∨ a = 2/3) :=
sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l1671_167167


namespace NUMINAMATH_CALUDE_john_needs_29_planks_l1671_167126

/-- The number of large planks John uses for the house wall. -/
def large_planks : ℕ := 12

/-- The number of small planks John uses for the house wall. -/
def small_planks : ℕ := 17

/-- The total number of planks John needs for the house wall. -/
def total_planks : ℕ := large_planks + small_planks

/-- Theorem stating that the total number of planks John needs is 29. -/
theorem john_needs_29_planks : total_planks = 29 := by
  sorry

end NUMINAMATH_CALUDE_john_needs_29_planks_l1671_167126


namespace NUMINAMATH_CALUDE_surface_area_of_revolution_l1671_167164

theorem surface_area_of_revolution (S α : ℝ) (h1 : S > 0) (h2 : 0 < α ∧ α < 2 * π) :
  let surface_area := (8 * π * S * Real.sin (α / 4)^2 * (1 + Real.cos (α / 4)^2)) / (α - Real.sin α)
  ∃ (R : ℝ), R > 0 ∧
    S = R^2 / 2 * (α - Real.sin α) ∧
    surface_area = 2 * π * R * (R * (1 - Real.cos (α / 2))) + π * (R * Real.sin (α / 2))^2 :=
by sorry

end NUMINAMATH_CALUDE_surface_area_of_revolution_l1671_167164


namespace NUMINAMATH_CALUDE_fraction_equality_l1671_167171

theorem fraction_equality : (2018 + 2018 + 2018) / (2018 + 2018 + 2018 + 2018) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1671_167171


namespace NUMINAMATH_CALUDE_largest_number_l1671_167159

def hcf (a b c : ℕ) : ℕ := Nat.gcd a (Nat.gcd b c)

def lcm (a b c : ℕ) : ℕ := Nat.lcm a (Nat.lcm b c)

theorem largest_number (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (hcf_cond : hcf a b c = 23)
  (lcm_cond : lcm a b c = 23 * 13 * 19 * 17) :
  max a (max b c) = 437 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l1671_167159


namespace NUMINAMATH_CALUDE_maria_juan_mm_l1671_167186

theorem maria_juan_mm (j : ℕ) (k : ℕ) (h1 : j > 0) : 
  (k * j - 3 = 2 * (j + 3)) → k = 11 := by
  sorry

end NUMINAMATH_CALUDE_maria_juan_mm_l1671_167186


namespace NUMINAMATH_CALUDE_simplify_expression_l1671_167114

theorem simplify_expression (a b : ℝ) : 
  (50*a + 130*b) + (21*a + 64*b) - (30*a + 115*b) - 2*(10*a - 25*b) = 21*a + 129*b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1671_167114


namespace NUMINAMATH_CALUDE_greatest_integer_for_integer_fraction_l1671_167148

theorem greatest_integer_for_integer_fraction : 
  (∀ y : ℤ, y > 35 → ¬(∃ n : ℤ, (y^2 + 2*y + 7) / (y - 4) = n)) ∧ 
  (∃ n : ℤ, (35^2 + 2*35 + 7) / (35 - 4) = n) := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_for_integer_fraction_l1671_167148


namespace NUMINAMATH_CALUDE_line_through_point_intersecting_circle_l1671_167136

/-- A line passing through a point and intersecting a circle -/
theorem line_through_point_intersecting_circle 
  (M : ℝ × ℝ) 
  (A B : ℝ × ℝ) 
  (h_M : M = (1, 0))
  (h_circle : ∀ P : ℝ × ℝ, P ∈ {P | P.1^2 + P.2^2 = 5} ↔ A ∈ {P | P.1^2 + P.2^2 = 5} ∧ B ∈ {P | P.1^2 + P.2^2 = 5})
  (h_first_quadrant : A.1 > 0 ∧ A.2 > 0)
  (h_vector_relation : B - M = 2 • (A - M))
  : ∃ (m c : ℝ), m = 1 ∧ c = -1 ∧ ∀ P : ℝ × ℝ, P ∈ {P | P.1 = m * P.2 + c} ↔ (A ∈ {P | P.1 = m * P.2 + c} ∧ B ∈ {P | P.1 = m * P.2 + c} ∧ M ∈ {P | P.1 = m * P.2 + c}) :=
sorry

end NUMINAMATH_CALUDE_line_through_point_intersecting_circle_l1671_167136


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_least_addition_for_51234_div_9_least_addition_is_3_l1671_167157

theorem least_addition_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (x : ℕ), x < d ∧ (n + x) % d = 0 ∧ ∀ (y : ℕ), y < x → (n + y) % d ≠ 0 :=
by sorry

theorem least_addition_for_51234_div_9 :
  ∃ (x : ℕ), x < 9 ∧ (51234 + x) % 9 = 0 ∧ ∀ (y : ℕ), y < x → (51234 + y) % 9 ≠ 0 :=
by
  apply least_addition_for_divisibility 51234 9
  norm_num

theorem least_addition_is_3 :
  ∃! (x : ℕ), x < 9 ∧ (51234 + x) % 9 = 0 ∧ ∀ (y : ℕ), y < x → (51234 + y) % 9 ≠ 0 ∧ x = 3 :=
by sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_least_addition_for_51234_div_9_least_addition_is_3_l1671_167157


namespace NUMINAMATH_CALUDE_sum_of_first_five_terms_l1671_167100

/-- Coordinate of point P_n on y-axis -/
def a (n : ℕ+) : ℚ := 2 / n

/-- Area of triangle formed by line through P_n and P_{n+1} and coordinate axes -/
def b (n : ℕ+) : ℚ := 4 + 1 / n - 1 / (n + 1)

/-- Sum of first n terms of sequence {b_n} -/
def S (n : ℕ+) : ℚ := 4 * n + n / (n + 1)

/-- Theorem: The sum of the first 5 terms of sequence {b_n} is 125/6 -/
theorem sum_of_first_five_terms : S 5 = 125 / 6 := by sorry

end NUMINAMATH_CALUDE_sum_of_first_five_terms_l1671_167100


namespace NUMINAMATH_CALUDE_frank_candy_weight_l1671_167141

/-- Frank's candy weight in pounds -/
def frank_candy : ℕ := 10

/-- Gwen's candy weight in pounds -/
def gwen_candy : ℕ := 7

/-- Total candy weight in pounds -/
def total_candy : ℕ := 17

theorem frank_candy_weight : 
  frank_candy + gwen_candy = total_candy :=
by sorry

end NUMINAMATH_CALUDE_frank_candy_weight_l1671_167141


namespace NUMINAMATH_CALUDE_ourSystem_is_valid_l1671_167123

-- Define a structure for a linear equation
structure LinearEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  -- ax + by = c

-- Define a system of two linear equations
structure SystemOfTwoLinearEquations where
  eq1 : LinearEquation
  eq2 : LinearEquation

-- Define the specific system we want to prove is valid
def ourSystem : SystemOfTwoLinearEquations := {
  eq1 := { a := 1, b := 1, c := 5 },  -- x + y = 5
  eq2 := { a := 0, b := 1, c := 2 }   -- y = 2
}

-- Theorem stating that our system is a valid system of two linear equations
theorem ourSystem_is_valid : 
  (ourSystem.eq1.a ≠ 0 ∨ ourSystem.eq1.b ≠ 0) ∧ 
  (ourSystem.eq2.a ≠ 0 ∨ ourSystem.eq2.b ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_ourSystem_is_valid_l1671_167123


namespace NUMINAMATH_CALUDE_martin_correct_is_40_l1671_167147

/-- The number of questions Campbell answered correctly -/
def campbell_correct : ℕ := 35

/-- The number of additional questions Kelsey answered correctly compared to Campbell -/
def kelsey_additional : ℕ := 8

/-- The number of fewer questions Martin answered correctly compared to Kelsey -/
def martin_fewer : ℕ := 3

/-- The number of questions Martin answered correctly -/
def martin_correct : ℕ := campbell_correct + kelsey_additional - martin_fewer

theorem martin_correct_is_40 : martin_correct = 40 := by
  sorry

end NUMINAMATH_CALUDE_martin_correct_is_40_l1671_167147


namespace NUMINAMATH_CALUDE_max_product_of_functions_l1671_167146

-- Define the functions h and k
def h : ℝ → ℝ := sorry
def k : ℝ → ℝ := sorry

-- State the theorem
theorem max_product_of_functions (h k : ℝ → ℝ) 
  (h_range : ∀ x, h x ∈ Set.Icc (-3) 5) 
  (k_range : ∀ x, k x ∈ Set.Icc (-1) 4) : 
  (∃ x y, h x * k y = 20) ∧ (∀ x y, h x * k y ≤ 20) := by
  sorry

end NUMINAMATH_CALUDE_max_product_of_functions_l1671_167146


namespace NUMINAMATH_CALUDE_marigold_sale_problem_l1671_167156

theorem marigold_sale_problem (day1 day2 day3 total : ℕ) : 
  day1 = 14 →
  day3 = 2 * day2 →
  total = day1 + day2 + day3 →
  total = 89 →
  day2 = 25 := by
sorry

end NUMINAMATH_CALUDE_marigold_sale_problem_l1671_167156


namespace NUMINAMATH_CALUDE_integer_count_inequality_l1671_167161

theorem integer_count_inequality (x : ℤ) : 
  (Finset.filter (fun i => (i - 1)^2 ≤ 9) (Finset.range 7)).card = 7 := by
  sorry

end NUMINAMATH_CALUDE_integer_count_inequality_l1671_167161


namespace NUMINAMATH_CALUDE_student_excess_is_105_l1671_167190

/-- Represents the composition of a fourth-grade classroom -/
structure Classroom where
  students : Nat
  guinea_pigs : Nat
  teachers : Nat

/-- The number of fourth-grade classrooms -/
def num_classrooms : Nat := 5

/-- A fourth-grade classroom in Big Valley School -/
def big_valley_classroom : Classroom :=
  { students := 25, guinea_pigs := 3, teachers := 1 }

/-- Theorem: The number of students exceeds the total number of guinea pigs and teachers by 105 in all fourth-grade classrooms -/
theorem student_excess_is_105 : 
  (num_classrooms * big_valley_classroom.students) - 
  (num_classrooms * (big_valley_classroom.guinea_pigs + big_valley_classroom.teachers)) = 105 := by
  sorry

end NUMINAMATH_CALUDE_student_excess_is_105_l1671_167190


namespace NUMINAMATH_CALUDE_hyperbola_parabola_property_l1671_167149

/-- Given a hyperbola and a parabola with specific properties, prove that 2e - b² = 4 -/
theorem hyperbola_parabola_property (a b : ℝ) (e : ℝ) :
  a > 0 →
  b > 0 →
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 ∧ y^2 = 4*x) →  -- Common point exists
  (∃ (x₀ y₀ : ℝ), x₀^2 / a^2 - y₀^2 / b^2 = 1 ∧ y₀^2 = 4*x₀ ∧ x₀ + 1 = 2) →  -- Distance to directrix is 2
  e = Real.sqrt (1 + b^2 / a^2) →  -- Definition of hyperbola eccentricity
  2*e - b^2 = 4 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_parabola_property_l1671_167149


namespace NUMINAMATH_CALUDE_parrot_seed_consumption_l1671_167105

/-- Calculates the weekly seed consumption of a parrot given the total birdseed supply,
    number of weeks, and the cockatiel's weekly consumption. --/
theorem parrot_seed_consumption
  (total_boxes : ℕ)
  (seeds_per_box : ℕ)
  (weeks : ℕ)
  (cockatiel_weekly : ℕ)
  (h1 : total_boxes = 8)
  (h2 : seeds_per_box = 225)
  (h3 : weeks = 12)
  (h4 : cockatiel_weekly = 50) :
  (total_boxes * seeds_per_box - weeks * cockatiel_weekly) / weeks = 100 := by
  sorry

#check parrot_seed_consumption

end NUMINAMATH_CALUDE_parrot_seed_consumption_l1671_167105


namespace NUMINAMATH_CALUDE_minimum_pigs_on_farm_l1671_167189

theorem minimum_pigs_on_farm (total : ℕ) (pigs : ℕ) : 
  (pigs : ℝ) / total ≥ 0.54 ∧ (pigs : ℝ) / total ≤ 0.57 → pigs ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_minimum_pigs_on_farm_l1671_167189


namespace NUMINAMATH_CALUDE_deepak_age_l1671_167108

/-- Given the ratio of Rahul's age to Deepak's age and Rahul's future age, 
    prove Deepak's current age. -/
theorem deepak_age (rahul_age deepak_age : ℕ) : 
  (rahul_age : ℚ) / deepak_age = 4 / 3 →
  rahul_age + 10 = 26 →
  deepak_age = 12 := by
sorry

end NUMINAMATH_CALUDE_deepak_age_l1671_167108


namespace NUMINAMATH_CALUDE_sqrt_equality_condition_l1671_167172

theorem sqrt_equality_condition (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  Real.sqrt (a - b + c) = Real.sqrt a - Real.sqrt b + Real.sqrt c ↔ a = b ∨ b = c :=
sorry

end NUMINAMATH_CALUDE_sqrt_equality_condition_l1671_167172


namespace NUMINAMATH_CALUDE_correct_equation_l1671_167122

theorem correct_equation : (-3)^2 * |-(1/3)| = 3 := by
  sorry

end NUMINAMATH_CALUDE_correct_equation_l1671_167122


namespace NUMINAMATH_CALUDE_matrix_sum_equality_l1671_167180

def A : Matrix (Fin 3) (Fin 3) ℤ := !![4, 1, -3; 0, -2, 5; 7, 0, 1]
def B : Matrix (Fin 3) (Fin 3) ℤ := !![-6, 9, 2; 3, -4, -8; 0, 5, -3]
def C : Matrix (Fin 3) (Fin 3) ℤ := !![-2, 10, -1; 3, -6, -3; 7, 5, -2]

theorem matrix_sum_equality : A + B = C := by sorry

end NUMINAMATH_CALUDE_matrix_sum_equality_l1671_167180


namespace NUMINAMATH_CALUDE_floor_times_self_72_l1671_167154

theorem floor_times_self_72 :
  ∃ (x : ℝ), x > 0 ∧ (Int.floor x : ℝ) * x = 72 ∧ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_floor_times_self_72_l1671_167154


namespace NUMINAMATH_CALUDE_composite_numbers_l1671_167130

theorem composite_numbers (N₁ N₂ : ℕ) : 
  N₁ = 2011 * 2012 * 2013 * 2014 + 1 →
  N₂ = 2012 * 2013 * 2014 * 2015 + 1 →
  ¬(Nat.Prime N₁) ∧ ¬(Nat.Prime N₂) :=
by sorry

end NUMINAMATH_CALUDE_composite_numbers_l1671_167130


namespace NUMINAMATH_CALUDE_modulo_graph_intercepts_sum_l1671_167117

theorem modulo_graph_intercepts_sum (x₀ y₀ : ℕ) : 
  x₀ < 37 → y₀ < 37 →
  (2 * x₀) % 37 = 1 →
  (3 * y₀ + 1) % 37 = 0 →
  x₀ + y₀ = 31 := by
sorry

end NUMINAMATH_CALUDE_modulo_graph_intercepts_sum_l1671_167117


namespace NUMINAMATH_CALUDE_probability_cap_given_sunglasses_l1671_167169

/-- The number of people wearing sunglasses -/
def sunglasses_wearers : ℕ := 60

/-- The number of people wearing caps -/
def cap_wearers : ℕ := 40

/-- The number of people wearing both sunglasses and caps and hats -/
def triple_wearers : ℕ := 8

/-- The probability that a person wearing a cap is also wearing sunglasses -/
def prob_sunglasses_given_cap : ℚ := 1/2

theorem probability_cap_given_sunglasses :
  let both_wearers := cap_wearers * prob_sunglasses_given_cap
  (both_wearers : ℚ) / sunglasses_wearers = 1/3 := by sorry

end NUMINAMATH_CALUDE_probability_cap_given_sunglasses_l1671_167169


namespace NUMINAMATH_CALUDE_round_0_689_to_two_places_l1671_167184

/-- Rounds a real number to the specified number of decimal places. -/
def round_to_decimal_places (x : ℝ) (places : ℕ) : ℝ := 
  sorry

/-- The given number to be rounded -/
def given_number : ℝ := 0.689

/-- Theorem stating that rounding 0.689 to two decimal places results in 0.69 -/
theorem round_0_689_to_two_places :
  round_to_decimal_places given_number 2 = 0.69 := by
  sorry

end NUMINAMATH_CALUDE_round_0_689_to_two_places_l1671_167184


namespace NUMINAMATH_CALUDE_function_inequality_l1671_167132

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : f 1 = 1)
variable (h2 : ∀ x, HasDerivAt f (f' x) x ∧ f' x < 1/3)

-- State the theorem
theorem function_inequality (x : ℝ) :
  f x < x/3 + 2/3 ↔ x > 1 := by sorry

end NUMINAMATH_CALUDE_function_inequality_l1671_167132


namespace NUMINAMATH_CALUDE_fraction_simplification_l1671_167192

theorem fraction_simplification : 
  (4 : ℝ) / (Real.sqrt 108 + 2 * Real.sqrt 12 + 2 * Real.sqrt 27) = Real.sqrt 3 / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1671_167192


namespace NUMINAMATH_CALUDE_shot_radius_l1671_167197

/-- Given a sphere of radius 4 cm from which 64 equal-sized spherical shots can be made,
    the radius of each shot is 1 cm. -/
theorem shot_radius (R : ℝ) (N : ℕ) (r : ℝ) : R = 4 → N = 64 → (R / r)^3 = N → r = 1 := by
  sorry

end NUMINAMATH_CALUDE_shot_radius_l1671_167197


namespace NUMINAMATH_CALUDE_smallest_winning_number_for_bernardo_l1671_167179

theorem smallest_winning_number_for_bernardo :
  ∃ (N : ℕ), N = 22 ∧
  (∀ k : ℕ, k < N →
    (3*k ≤ 999 ∧
     3*k + 30 ≤ 999 ∧
     9*k + 90 ≤ 999 ∧
     9*k + 120 ≤ 999 ∧
     27*k + 360 ≤ 999)) ∧
  (3*N ≤ 999 ∧
   3*N + 30 ≤ 999 ∧
   9*N + 90 ≤ 999 ∧
   9*N + 120 ≤ 999 ∧
   27*N + 360 ≤ 999 ∧
   27*N + 390 > 999) :=
by sorry

end NUMINAMATH_CALUDE_smallest_winning_number_for_bernardo_l1671_167179


namespace NUMINAMATH_CALUDE_subset_complement_of_intersection_eq_l1671_167194

universe u

theorem subset_complement_of_intersection_eq {U : Type u} [TopologicalSpace U] (M N : Set U) 
  (h : M ∩ N = N) : (Mᶜ : Set U) ⊆ Nᶜ := by
  sorry

end NUMINAMATH_CALUDE_subset_complement_of_intersection_eq_l1671_167194


namespace NUMINAMATH_CALUDE_complex_equation_modulus_l1671_167102

theorem complex_equation_modulus (z : ℂ) (h : z * (2 - 3*Complex.I) = 6 + 4*Complex.I) : 
  Complex.abs z = 2 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_modulus_l1671_167102


namespace NUMINAMATH_CALUDE_mixed_doubles_selections_l1671_167182

/-- The number of male players in the table tennis team -/
def num_male_players : ℕ := 5

/-- The number of female players in the table tennis team -/
def num_female_players : ℕ := 4

/-- The total number of ways to select a mixed doubles team -/
def total_selections : ℕ := num_male_players * num_female_players

theorem mixed_doubles_selections :
  total_selections = 20 :=
sorry

end NUMINAMATH_CALUDE_mixed_doubles_selections_l1671_167182


namespace NUMINAMATH_CALUDE_vitamin_d_pack_size_l1671_167118

/-- The number of Vitamin A supplements in each pack -/
def vitamin_a_pack_size : ℕ := 7

/-- The smallest number of each type of vitamin sold -/
def smallest_quantity_sold : ℕ := 119

/-- Theorem stating that the number of Vitamin D supplements in each pack is 17 -/
theorem vitamin_d_pack_size :
  ∃ (n m x : ℕ),
    n * vitamin_a_pack_size = m * x ∧
    n * vitamin_a_pack_size = smallest_quantity_sold ∧
    x > 1 ∧
    x < vitamin_a_pack_size ∧
    x = 17 := by
  sorry

end NUMINAMATH_CALUDE_vitamin_d_pack_size_l1671_167118


namespace NUMINAMATH_CALUDE_f_decreasing_inequality_solution_set_l1671_167134

noncomputable section

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
axiom f_prop1 : ∀ (x y : ℝ), x > 0 → y > 0 → f (x * y) = f x + f y
axiom f_prop2 : ∀ (x : ℝ), 0 < x → x < 1 → f x > 0
axiom f_prop3 : f (1/2) = 1

-- Theorem 1: f is decreasing on its domain
theorem f_decreasing : ∀ (x₁ x₂ : ℝ), 0 < x₁ → 0 < x₂ → x₁ < x₂ → f x₁ > f x₂ := by
  sorry

-- Theorem 2: Solution set of the inequality
theorem inequality_solution_set : 
  {x : ℝ | f (x - 3) > f (1/x) - 2} = Set.Ioo 3 4 := by
  sorry

end

end NUMINAMATH_CALUDE_f_decreasing_inequality_solution_set_l1671_167134


namespace NUMINAMATH_CALUDE_solutions_of_quartic_equation_l1671_167139

theorem solutions_of_quartic_equation :
  ∀ x : ℂ, x^4 - 16 = 0 ↔ x ∈ ({2, -2, 2*I, -2*I} : Set ℂ) :=
by sorry

end NUMINAMATH_CALUDE_solutions_of_quartic_equation_l1671_167139


namespace NUMINAMATH_CALUDE_vowels_on_board_l1671_167115

/-- The number of vowels in the English alphabet -/
def num_vowels : ℕ := 5

/-- The number of times each vowel is written -/
def times_written : ℕ := 2

/-- The total number of vowels written on the board -/
def total_vowels : ℕ := num_vowels * times_written

theorem vowels_on_board : total_vowels = 10 := by
  sorry

end NUMINAMATH_CALUDE_vowels_on_board_l1671_167115


namespace NUMINAMATH_CALUDE_common_measure_proof_l1671_167176

theorem common_measure_proof (a b : ℚ) (ha : a = 4/15) (hb : b = 8/21) :
  ∃ (m : ℚ), m > 0 ∧ ∃ (k₁ k₂ : ℕ), a = k₁ * m ∧ b = k₂ * m :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_common_measure_proof_l1671_167176


namespace NUMINAMATH_CALUDE_prob_at_least_one_woman_l1671_167125

/-- The probability of selecting at least one woman when choosing 3 people at random from a group of 5 men and 5 women -/
theorem prob_at_least_one_woman (total_people : ℕ) (men : ℕ) (women : ℕ) (selected : ℕ) : 
  total_people = men + women → 
  men = 5 → 
  women = 5 → 
  selected = 3 → 
  (1 : ℚ) - (men.choose selected : ℚ) / (total_people.choose selected : ℚ) = 11 / 12 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_woman_l1671_167125


namespace NUMINAMATH_CALUDE_tangent_line_slope_at_zero_l1671_167178

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (x^2 + 1)

theorem tangent_line_slope_at_zero :
  let f' := deriv f
  f' 0 = 1 := by sorry

end NUMINAMATH_CALUDE_tangent_line_slope_at_zero_l1671_167178


namespace NUMINAMATH_CALUDE_factor_calculation_l1671_167111

theorem factor_calculation (initial_number : ℕ) (factor : ℚ) : 
  initial_number = 9 →
  factor * (2 * initial_number + 13) = 93 →
  factor = 3 := by sorry

end NUMINAMATH_CALUDE_factor_calculation_l1671_167111


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l1671_167168

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 8 / x + 1 / y = 1) : x + 2 * y ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l1671_167168


namespace NUMINAMATH_CALUDE_walkway_diameter_l1671_167188

theorem walkway_diameter (water_diameter : Real) (tile_width : Real) (walkway_width : Real) :
  water_diameter = 16 →
  tile_width = 12 →
  walkway_width = 10 →
  2 * (water_diameter / 2 + tile_width + walkway_width) = 60 := by
  sorry

end NUMINAMATH_CALUDE_walkway_diameter_l1671_167188


namespace NUMINAMATH_CALUDE_beavers_help_l1671_167199

theorem beavers_help (initial_beavers : Real) (current_beavers : Nat) 
  (h1 : initial_beavers = 2.0) 
  (h2 : current_beavers = 3) : 
  (current_beavers : Real) - initial_beavers = 1 := by
  sorry

end NUMINAMATH_CALUDE_beavers_help_l1671_167199


namespace NUMINAMATH_CALUDE_basketball_lineup_combinations_l1671_167181

theorem basketball_lineup_combinations : 
  ∀ (total_players : ℕ) (fixed_players : ℕ) (lineup_size : ℕ),
    total_players = 15 →
    fixed_players = 2 →
    lineup_size = 6 →
    Nat.choose (total_players - fixed_players) (lineup_size - fixed_players) = 715 := by
  sorry

end NUMINAMATH_CALUDE_basketball_lineup_combinations_l1671_167181


namespace NUMINAMATH_CALUDE_m_range_for_inequality_l1671_167104

theorem m_range_for_inequality (m : ℝ) : 
  (∀ x : ℝ, x ≤ -1 → (m^2 - m) * 4^x - 2^x < 0) ↔ -1 < m ∧ m < 2 := by
  sorry

end NUMINAMATH_CALUDE_m_range_for_inequality_l1671_167104


namespace NUMINAMATH_CALUDE_sqrt_x_plus_reciprocal_l1671_167195

theorem sqrt_x_plus_reciprocal (x : ℝ) (hx : x > 0) (h : x + 1/x = 50) :
  Real.sqrt x + 1 / Real.sqrt x = Real.sqrt 52 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_reciprocal_l1671_167195


namespace NUMINAMATH_CALUDE_rice_left_calculation_l1671_167142

/-- Calculates the amount of rice left in grams after cooking -/
def rice_left (initial : ℚ) (morning_cooked : ℚ) (evening_fraction : ℚ) : ℚ :=
  let remaining_after_morning := initial - morning_cooked
  let evening_cooked := remaining_after_morning * evening_fraction
  let final_remaining := remaining_after_morning - evening_cooked
  final_remaining * 1000  -- Convert to grams

/-- Theorem stating the amount of rice left after cooking -/
theorem rice_left_calculation :
  rice_left 10 (9/10 * 10) (1/4) = 750 := by
  sorry

#eval rice_left 10 (9/10 * 10) (1/4)

end NUMINAMATH_CALUDE_rice_left_calculation_l1671_167142


namespace NUMINAMATH_CALUDE_valid_pairs_l1671_167124

def is_valid_pair (m n : ℕ+) : Prop :=
  let d := Nat.gcd m n
  m + n^2 + d^3 = m * n * d

theorem valid_pairs :
  ∀ m n : ℕ+, is_valid_pair m n ↔ (m = 4 ∧ n = 2) ∨ (m = 4 ∧ n = 6) ∨ (m = 5 ∧ n = 2) ∨ (m = 5 ∧ n = 3) :=
sorry

end NUMINAMATH_CALUDE_valid_pairs_l1671_167124


namespace NUMINAMATH_CALUDE_sum_of_cubes_of_roots_l1671_167173

theorem sum_of_cubes_of_roots (x₁ x₂ x₃ : ℝ) : 
  (2 * x₁^3 + 3 * x₁^2 - 11 * x₁ + 6 = 0) →
  (2 * x₂^3 + 3 * x₂^2 - 11 * x₂ + 6 = 0) →
  (2 * x₃^3 + 3 * x₃^2 - 11 * x₃ + 6 = 0) →
  (x₁ + x₂ + x₃ = -3/2) →
  (x₁*x₂ + x₂*x₃ + x₃*x₁ = -11/2) →
  (x₁*x₂*x₃ = -3) →
  x₁^3 + x₂^3 + x₃^3 = -99/8 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_cubes_of_roots_l1671_167173


namespace NUMINAMATH_CALUDE_codecracker_combinations_l1671_167165

/-- The number of different colors of pegs available in CodeCracker -/
def num_colors : ℕ := 6

/-- The number of slots in a CodeCracker code -/
def num_slots : ℕ := 5

/-- The number of possible secret codes in CodeCracker -/
def num_codes : ℕ := num_colors ^ num_slots

theorem codecracker_combinations : num_codes = 7776 := by
  sorry

end NUMINAMATH_CALUDE_codecracker_combinations_l1671_167165


namespace NUMINAMATH_CALUDE_simple_interest_from_sum_and_true_discount_l1671_167151

/-- Simple interest calculation given sum and true discount -/
theorem simple_interest_from_sum_and_true_discount
  (sum : ℝ) (true_discount : ℝ) (h1 : sum = 947.1428571428571)
  (h2 : true_discount = 78) :
  sum - (sum - true_discount) = true_discount :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_from_sum_and_true_discount_l1671_167151


namespace NUMINAMATH_CALUDE_equal_expressions_l1671_167196

theorem equal_expressions : 2007 * 2011 - 2008 * 2010 = 2008 * 2012 - 2009 * 2011 := by
  sorry

end NUMINAMATH_CALUDE_equal_expressions_l1671_167196


namespace NUMINAMATH_CALUDE_alternating_sum_of_coefficients_l1671_167150

theorem alternating_sum_of_coefficients : 
  ∀ (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ), 
  (∀ x : ℝ, (1 + 3*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₀ - a₁ + a₂ - a₃ + a₄ - a₅ = -32 := by
sorry

end NUMINAMATH_CALUDE_alternating_sum_of_coefficients_l1671_167150


namespace NUMINAMATH_CALUDE_u_v_sum_of_squares_l1671_167129

theorem u_v_sum_of_squares (u v : ℝ) (hu : u > 1) (hv : v > 1)
  (h : (Real.log u / Real.log 3)^4 + (Real.log v / Real.log 7)^4 = 10 * (Real.log u / Real.log 3) * (Real.log v / Real.log 7)) :
  u^2 + v^2 = 3^Real.sqrt 5 + 7^Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_u_v_sum_of_squares_l1671_167129


namespace NUMINAMATH_CALUDE_gcd_g_y_l1671_167187

def g (y : ℤ) : ℤ := (3*y + 5)*(8*y + 1)*(11*y + 3)*(y + 15)

theorem gcd_g_y (y : ℤ) (h : ∃ k : ℤ, y = 4060 * k) : 
  Int.gcd (g y) y = 5 := by sorry

end NUMINAMATH_CALUDE_gcd_g_y_l1671_167187


namespace NUMINAMATH_CALUDE_reciprocal_roots_l1671_167140

theorem reciprocal_roots (p q : ℝ) (r₁ r₂ : ℂ) : 
  (r₁^2 + p*r₁ + q = 0 ∧ r₂^2 + p*r₂ + q = 0) → 
  ((1/r₁)^2 * q + (1/r₁) * p + 1 = 0 ∧ (1/r₂)^2 * q + (1/r₂) * p + 1 = 0) :=
sorry

end NUMINAMATH_CALUDE_reciprocal_roots_l1671_167140


namespace NUMINAMATH_CALUDE_f_composition_negative_two_l1671_167113

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ -2 then x + 2
  else if x < 3 then 2^x
  else Real.log x

theorem f_composition_negative_two : f (f (-2)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_negative_two_l1671_167113


namespace NUMINAMATH_CALUDE_sandy_marks_per_correct_sum_l1671_167183

theorem sandy_marks_per_correct_sum :
  ∀ (total_sums : ℕ) (total_marks : ℤ) (correct_sums : ℕ) (marks_lost_per_incorrect : ℤ) (marks_per_correct : ℤ),
    total_sums = 30 →
    total_marks = 60 →
    correct_sums = 24 →
    marks_lost_per_incorrect = 2 →
    (marks_per_correct * correct_sums : ℤ) - (marks_lost_per_incorrect * (total_sums - correct_sums) : ℤ) = total_marks →
    marks_per_correct = 3 :=
by sorry

end NUMINAMATH_CALUDE_sandy_marks_per_correct_sum_l1671_167183


namespace NUMINAMATH_CALUDE_cos_equality_theorem_l1671_167174

theorem cos_equality_theorem :
  ∃ (n : ℤ), 0 ≤ n ∧ n ≤ 180 ∧ Real.cos (n * π / 180) = Real.cos (942 * π / 180) ∧ n = 138 := by
  sorry

end NUMINAMATH_CALUDE_cos_equality_theorem_l1671_167174


namespace NUMINAMATH_CALUDE_f_of_2_equals_1_l1671_167137

-- Define the function f
def f (x : ℝ) : ℝ := (x - 1)^3 - (x - 1) + 1

-- State the theorem
theorem f_of_2_equals_1 : f 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_of_2_equals_1_l1671_167137


namespace NUMINAMATH_CALUDE_steves_speed_back_l1671_167120

/-- Proves that Steve's speed on the way back from work is 14 km/h given the conditions --/
theorem steves_speed_back (distance : ℝ) (total_time : ℝ) (speed_ratio : ℝ) : 
  distance = 28 → 
  total_time = 6 → 
  speed_ratio = 2 → 
  (distance / (distance / (2 * (distance / (total_time - distance / (2 * (distance / total_time)))))) = 14) := by
  sorry

end NUMINAMATH_CALUDE_steves_speed_back_l1671_167120


namespace NUMINAMATH_CALUDE_expansion_coefficient_l1671_167133

/-- The binomial coefficient function -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The sum of binomial coefficients for a given n -/
def sum_binomial_coefficients (n : ℕ) : ℕ := 2^n

/-- The sum of all coefficients in the expansion of (x + 3/√x)^n when x = 1 -/
def sum_all_coefficients (n : ℕ) : ℕ := 4^n

/-- The coefficient of x^3 in the expansion of (x + 3/√x)^n -/
def coefficient_x3 (n : ℕ) : ℕ := binomial n 2 * 3^2

theorem expansion_coefficient :
  ∃ n : ℕ,
    sum_all_coefficients n / sum_binomial_coefficients n = 64 ∧
    coefficient_x3 n = 135 := by sorry

end NUMINAMATH_CALUDE_expansion_coefficient_l1671_167133


namespace NUMINAMATH_CALUDE_second_hand_revolution_time_l1671_167155

/-- The time in seconds for a second hand to complete one revolution -/
def revolution_time_seconds : ℕ := 60

/-- The number of seconds in a minute -/
def seconds_per_minute : ℕ := 60

/-- The time in minutes for a second hand to complete one revolution -/
def revolution_time_minutes : ℚ := revolution_time_seconds / seconds_per_minute

theorem second_hand_revolution_time :
  revolution_time_seconds = 60 ∧ revolution_time_minutes = 1 := by sorry

end NUMINAMATH_CALUDE_second_hand_revolution_time_l1671_167155


namespace NUMINAMATH_CALUDE_gcd_2100_2091_l1671_167152

theorem gcd_2100_2091 : Nat.gcd (2^2100 - 1) (2^2091 - 1) = 2^9 - 1 := by sorry

end NUMINAMATH_CALUDE_gcd_2100_2091_l1671_167152


namespace NUMINAMATH_CALUDE_helen_to_betsy_win_ratio_l1671_167106

/-- The ratio of Helen's wins to Betsy's wins in a Monopoly game scenario -/
theorem helen_to_betsy_win_ratio :
  ∀ (helen_wins : ℕ),
  let betsy_wins : ℕ := 5
  let susan_wins : ℕ := 3 * betsy_wins
  let total_wins : ℕ := 30
  (betsy_wins + helen_wins + susan_wins = total_wins) →
  (helen_wins : ℚ) / betsy_wins = 2 := by
    sorry

end NUMINAMATH_CALUDE_helen_to_betsy_win_ratio_l1671_167106


namespace NUMINAMATH_CALUDE_smallest_n_for_eq1_smallest_n_for_eq2_l1671_167191

-- Define the properties for the equations
def satisfies_eq1 (n : ℕ) : Prop :=
  ∃ x y : ℕ, x * (x + n) = y^2

def satisfies_eq2 (n : ℕ) : Prop :=
  ∃ x y : ℕ, x * (x + n) = y^3

-- Define the smallest n for each equation
def smallest_n1 : ℕ := 3
def smallest_n2 : ℕ := 2

-- Theorem for the first equation
theorem smallest_n_for_eq1 :
  satisfies_eq1 smallest_n1 ∧
  ∀ m : ℕ, m < smallest_n1 → ¬(satisfies_eq1 m) :=
by sorry

-- Theorem for the second equation
theorem smallest_n_for_eq2 :
  satisfies_eq2 smallest_n2 ∧
  ∀ m : ℕ, m < smallest_n2 → ¬(satisfies_eq2 m) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_eq1_smallest_n_for_eq2_l1671_167191


namespace NUMINAMATH_CALUDE_distance_city_A_to_B_distance_city_A_to_B_value_l1671_167127

/-- Proves that the distance between city A and city B is 450 km given the problem conditions -/
theorem distance_city_A_to_B : ℝ → ℝ → ℝ → ℝ → Prop :=
  fun (time_eddy : ℝ) (time_freddy : ℝ) (speed_ratio : ℝ) (known_distance : ℝ) =>
    time_eddy = 3 ∧ 
    time_freddy = 4 ∧ 
    speed_ratio = 2 ∧ 
    known_distance = 300 →
    ∃ (distance_AB distance_AC : ℝ),
      distance_AB / time_eddy = speed_ratio * (distance_AC / time_freddy) ∧
      (distance_AB = known_distance ∨ distance_AC = known_distance) ∧
      distance_AB = 450

theorem distance_city_A_to_B_value : distance_city_A_to_B 3 4 2 300 := by
  sorry

end NUMINAMATH_CALUDE_distance_city_A_to_B_distance_city_A_to_B_value_l1671_167127


namespace NUMINAMATH_CALUDE_max_sum_distance_to_line_l1671_167163

theorem max_sum_distance_to_line (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : x₁^2 + y₁^2 = 1)
  (h2 : x₂^2 + y₂^2 = 1)
  (h3 : x₁*x₂ + y₁*y₂ = 1/2) :
  (|x₁ + y₁ - 1| / Real.sqrt 2) + (|x₂ + y₂ - 1| / Real.sqrt 2) ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_max_sum_distance_to_line_l1671_167163


namespace NUMINAMATH_CALUDE_infinite_sum_of_square_and_prime_infinite_not_sum_of_square_and_prime_l1671_167177

-- Define a function to check if a number is a perfect square
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m ^ 2

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

-- Define a function to check if a perfect square can be expressed as the sum of a perfect square and a prime
def is_sum_of_square_and_prime (n : ℕ) : Prop :=
  is_perfect_square n ∧ ∃ a b : ℕ, is_perfect_square a ∧ is_prime b ∧ n = a + b

-- Statement 1: The set of perfect squares that can be expressed as the sum of a perfect square and a prime number is infinite
theorem infinite_sum_of_square_and_prime :
  ∀ N : ℕ, ∃ n : ℕ, n > N ∧ is_sum_of_square_and_prime n :=
sorry

-- Statement 2: The set of perfect squares that cannot be expressed as the sum of a perfect square and a prime number is infinite
theorem infinite_not_sum_of_square_and_prime :
  ∀ N : ℕ, ∃ n : ℕ, n > N ∧ is_perfect_square n ∧ ¬is_sum_of_square_and_prime n :=
sorry

end NUMINAMATH_CALUDE_infinite_sum_of_square_and_prime_infinite_not_sum_of_square_and_prime_l1671_167177


namespace NUMINAMATH_CALUDE_total_spent_equals_sum_l1671_167160

/-- The total amount Jason spent on clothing -/
def total_spent : ℚ := 19.02

/-- The amount Jason spent on shorts -/
def shorts_cost : ℚ := 14.28

/-- The amount Jason spent on a jacket -/
def jacket_cost : ℚ := 4.74

/-- Theorem stating that the total amount spent is the sum of the costs of shorts and jacket -/
theorem total_spent_equals_sum : total_spent = shorts_cost + jacket_cost := by
  sorry

end NUMINAMATH_CALUDE_total_spent_equals_sum_l1671_167160


namespace NUMINAMATH_CALUDE_sock_selection_combinations_l1671_167101

theorem sock_selection_combinations : Nat.choose 7 4 = 35 := by
  sorry

end NUMINAMATH_CALUDE_sock_selection_combinations_l1671_167101


namespace NUMINAMATH_CALUDE_batsman_average_increase_l1671_167185

/-- Represents a batsman's performance -/
structure Batsman where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an additional inning -/
def newAverage (b : Batsman) (additionalRuns : ℕ) : ℚ :=
  (b.totalRuns + additionalRuns) / (b.innings + 1)

/-- Theorem: If a batsman's average increases by 5 after scoring 110 runs in the 11th inning, 
    then his new average is 60 runs -/
theorem batsman_average_increase 
  (b : Batsman) 
  (h1 : b.innings = 10) 
  (h2 : newAverage b 110 = b.average + 5) : 
  newAverage b 110 = 60 := by
  sorry

#check batsman_average_increase

end NUMINAMATH_CALUDE_batsman_average_increase_l1671_167185


namespace NUMINAMATH_CALUDE_sin_330_degrees_l1671_167131

theorem sin_330_degrees : Real.sin (330 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l1671_167131


namespace NUMINAMATH_CALUDE_tina_career_result_l1671_167107

def boxer_career (initial_wins : ℕ) (additional_wins1 : ℕ) (triple_factor : ℕ) (additional_wins2 : ℕ) (double_factor : ℕ) : ℕ × ℕ :=
  let wins1 := initial_wins + additional_wins1
  let wins2 := wins1 * triple_factor
  let wins3 := wins2 + additional_wins2
  let final_wins := wins3 * double_factor
  let losses := 3
  (final_wins, losses)

theorem tina_career_result :
  let (wins, losses) := boxer_career 10 5 3 7 2
  wins - losses = 131 := by sorry

end NUMINAMATH_CALUDE_tina_career_result_l1671_167107


namespace NUMINAMATH_CALUDE_x_value_proof_l1671_167170

theorem x_value_proof (x : ℝ) (h1 : x > 0) (h2 : Real.sqrt ((10 * x) / 3) = x) : x = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l1671_167170


namespace NUMINAMATH_CALUDE_max_min_product_l1671_167145

theorem max_min_product (A B : ℕ) (sum_constraint : A + B = 100) :
  (∃ (X Y : ℕ), X + Y = 100 ∧ X * Y ≤ A * B) ∧
  (∃ (X Y : ℕ), X + Y = 100 ∧ A * B ≤ X * Y) ∧
  (∃ (X Y : ℕ), X + Y = 100 ∧ X * Y = 2500) ∧
  (∃ (X Y : ℕ), X + Y = 100 ∧ X * Y = 0) :=
by sorry

end NUMINAMATH_CALUDE_max_min_product_l1671_167145


namespace NUMINAMATH_CALUDE_unique_perpendicular_line_l1671_167128

/-- A line in a plane -/
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

/-- A point in a plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Predicate to check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- Two lines are perpendicular if their slopes are negative reciprocals of each other -/
def Line.isPerpendicular (l1 l2 : Line) : Prop :=
  l1.slope * l2.slope = -1

/-- Main theorem: There exists exactly one perpendicular line through a point on a given line -/
theorem unique_perpendicular_line (l : Line) (p : Point) (h : p.liesOn l) :
  ∃! l_perp : Line, l_perp.isPerpendicular l ∧ p.liesOn l_perp :=
sorry

end NUMINAMATH_CALUDE_unique_perpendicular_line_l1671_167128


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1671_167158

/-- Given a hyperbola and a parabola with specific properties, prove the equation of the hyperbola. -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (k : ℝ), k * a = b ∧ k * 2 = Real.sqrt 3) →  -- asymptote condition
  (∃ (c : ℝ), c^2 = a^2 + b^2 ∧ c = Real.sqrt 7) →  -- focus and directrix condition
  a^2 = 4 ∧ b^2 = 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1671_167158


namespace NUMINAMATH_CALUDE_min_value_of_function_l1671_167143

theorem min_value_of_function (x : ℝ) (h : x > -1) :
  x + (x + 1)⁻¹ ≥ 1 ∧ (x + (x + 1)⁻¹ = 1 ↔ x = 0) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_l1671_167143


namespace NUMINAMATH_CALUDE_remaining_pool_area_l1671_167119

/-- The area of the remaining pool space given a circular pool with diameter 13 meters
    and a rectangular obstacle with dimensions 2.5 meters by 4 meters. -/
theorem remaining_pool_area :
  let pool_diameter : ℝ := 13
  let obstacle_length : ℝ := 2.5
  let obstacle_width : ℝ := 4
  let pool_area := π * (pool_diameter / 2) ^ 2
  let obstacle_area := obstacle_length * obstacle_width
  pool_area - obstacle_area = 132.7325 * π - 10 := by sorry

end NUMINAMATH_CALUDE_remaining_pool_area_l1671_167119


namespace NUMINAMATH_CALUDE_days_to_fulfill_orders_l1671_167116

/-- Represents the production details of Wallace's beef jerky company -/
structure JerkyProduction where
  small_batch_time : ℕ
  small_batch_output : ℕ
  large_batch_time : ℕ
  large_batch_output : ℕ
  total_small_bags_ordered : ℕ
  total_large_bags_ordered : ℕ
  small_bags_in_stock : ℕ
  large_bags_in_stock : ℕ
  max_daily_production_hours : ℕ

/-- Calculates the minimum number of days required to fulfill all orders -/
def min_days_to_fulfill_orders (prod : JerkyProduction) : ℕ :=
  let small_bags_to_produce := prod.total_small_bags_ordered - prod.small_bags_in_stock
  let large_bags_to_produce := prod.total_large_bags_ordered - prod.large_bags_in_stock
  let small_batches_needed := (small_bags_to_produce + prod.small_batch_output - 1) / prod.small_batch_output
  let large_batches_needed := (large_bags_to_produce + prod.large_batch_output - 1) / prod.large_batch_output
  let total_hours_needed := small_batches_needed * prod.small_batch_time + large_batches_needed * prod.large_batch_time
  (total_hours_needed + prod.max_daily_production_hours - 1) / prod.max_daily_production_hours

/-- Theorem stating that given the specific conditions, 13 days are required to fulfill all orders -/
theorem days_to_fulfill_orders :
  let prod := JerkyProduction.mk 8 12 12 8 157 97 18 10 18
  min_days_to_fulfill_orders prod = 13 := by
  sorry


end NUMINAMATH_CALUDE_days_to_fulfill_orders_l1671_167116


namespace NUMINAMATH_CALUDE_derivative_f_derivative_g_l1671_167112

noncomputable section

open Real

-- Function 1
def f (x : ℝ) : ℝ := (1 / Real.sqrt x) * cos x

-- Function 2
def g (x : ℝ) : ℝ := 5 * x^10 * sin x - 2 * Real.sqrt x * cos x - 9

-- Theorem for the derivative of function 1
theorem derivative_f (x : ℝ) (hx : x > 0) :
  deriv f x = -(cos x + 2 * x * sin x) / (2 * x * Real.sqrt x) :=
sorry

-- Theorem for the derivative of function 2
theorem derivative_g (x : ℝ) (hx : x > 0) :
  deriv g x = 50 * x^9 * sin x + 5 * x^10 * cos x - (Real.sqrt x * cos x) / x + 2 * Real.sqrt x * sin x :=
sorry

end NUMINAMATH_CALUDE_derivative_f_derivative_g_l1671_167112


namespace NUMINAMATH_CALUDE_decreasing_iff_a_in_range_l1671_167138

/-- A piecewise function f defined on ℝ -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then a / x else (2 - 3 * a) * x + 1

/-- The property that f is a decreasing function on ℝ -/
def is_decreasing (a : ℝ) : Prop :=
  ∀ x y, x < y → f a x > f a y

/-- The main theorem stating the equivalence between f being decreasing and a being in (2/3, 3/4] -/
theorem decreasing_iff_a_in_range (a : ℝ) :
  is_decreasing a ↔ 2/3 < a ∧ a ≤ 3/4 := by sorry

end NUMINAMATH_CALUDE_decreasing_iff_a_in_range_l1671_167138


namespace NUMINAMATH_CALUDE_union_determines_a_l1671_167103

def A : Set ℝ := {1, 2}
def B (a : ℝ) : Set ℝ := {a, a^2 + 1}

theorem union_determines_a (a : ℝ) (h : A ∪ B a = {0, 1, 2}) : a = 0 := by
  sorry

end NUMINAMATH_CALUDE_union_determines_a_l1671_167103


namespace NUMINAMATH_CALUDE_vector_on_line_and_parallel_l1671_167175

/-- A line parameterized by x = 5t + 3 and y = 2t + 3 -/
def parameterized_line (t : ℝ) : ℝ × ℝ := (5 * t + 3, 2 * t + 3)

/-- The vector we want to prove is on the line and parallel to (5, 2) -/
def vector : ℝ × ℝ := (-1.5, -0.6)

/-- The direction vector we want our vector to be parallel to -/
def direction : ℝ × ℝ := (5, 2)

theorem vector_on_line_and_parallel :
  ∃ (t : ℝ), parameterized_line t = vector ∧
  ∃ (k : ℝ), vector.1 = k * direction.1 ∧ vector.2 = k * direction.2 := by
  sorry

end NUMINAMATH_CALUDE_vector_on_line_and_parallel_l1671_167175


namespace NUMINAMATH_CALUDE_george_money_left_l1671_167121

def monthly_income : ℕ := 240

def donation : ℕ := monthly_income / 2

def remaining_after_donation : ℕ := monthly_income - donation

def groceries_cost : ℕ := 20

def amount_left : ℕ := remaining_after_donation - groceries_cost

theorem george_money_left : amount_left = 100 := by
  sorry

end NUMINAMATH_CALUDE_george_money_left_l1671_167121


namespace NUMINAMATH_CALUDE_problem1_problem2_l1671_167153

-- Problem 1
theorem problem1 : Real.sqrt 3 ^ 2 - (2023 + π / 2) ^ 0 - (-1) ^ (-1 : ℤ) = 3 := by sorry

-- Problem 2
theorem problem2 : ¬∃ x : ℝ, 5 * x - 4 > 3 * x ∧ (2 * x - 1) / 3 < x / 2 := by sorry

end NUMINAMATH_CALUDE_problem1_problem2_l1671_167153


namespace NUMINAMATH_CALUDE_example_monomial_properties_l1671_167110

/-- Represents a monomial with variables x, y, and z -/
structure Monomial where
  coeff : ℤ
  x_exp : ℕ
  y_exp : ℕ
  z_exp : ℕ

/-- Calculates the degree of a monomial -/
def degree (m : Monomial) : ℕ :=
  m.x_exp + m.y_exp + m.z_exp

/-- The monomial -xy^2z^3 -/
def example_monomial : Monomial :=
  { coeff := -1, x_exp := 1, y_exp := 2, z_exp := 3 }

theorem example_monomial_properties :
  example_monomial.coeff = -1 ∧ degree example_monomial = 6 := by
  sorry

end NUMINAMATH_CALUDE_example_monomial_properties_l1671_167110


namespace NUMINAMATH_CALUDE_set_inclusion_l1671_167166

-- Define the sets M, N, and P
def M : Set (ℝ × ℝ) := {p | abs p.1 + abs p.2 < 1}

def N : Set (ℝ × ℝ) := {p | Real.sqrt ((p.1 - 1/2)^2 + (p.2 + 1/2)^2) + 
                             Real.sqrt ((p.1 + 1/2)^2 + (p.2 - 1/2)^2) < 2 * Real.sqrt 2}

def P : Set (ℝ × ℝ) := {p | abs (p.1 + p.2) < 1 ∧ abs p.1 < 1 ∧ abs p.2 < 1}

-- State the theorem
theorem set_inclusion : M ⊆ P ∧ P ⊆ N := by sorry

end NUMINAMATH_CALUDE_set_inclusion_l1671_167166
