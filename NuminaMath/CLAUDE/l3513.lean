import Mathlib

namespace NUMINAMATH_CALUDE_fraction_inequality_l3513_351314

theorem fraction_inequality (a b m : ℝ) (h1 : b > a) (h2 : m > 0) :
  b / a > (b + m) / (a + m) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l3513_351314


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l3513_351386

theorem cubic_equation_roots (k m : ℝ) : 
  (∃ a b c : ℕ+, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    (∀ x : ℝ, x^3 - 9*x^2 + k*x - m = 0 ↔ (x = a ∨ x = b ∨ x = c))) →
  k + m = 38 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l3513_351386


namespace NUMINAMATH_CALUDE_multiple_of_seven_l3513_351315

def is_single_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def six_digit_number (d : ℕ) : ℕ := 567800 + d * 10 + 2

theorem multiple_of_seven (d : ℕ) (h : is_single_digit d) : 
  (six_digit_number d) % 7 = 0 ↔ d = 0 ∨ d = 7 := by
sorry

end NUMINAMATH_CALUDE_multiple_of_seven_l3513_351315


namespace NUMINAMATH_CALUDE_product_of_powers_equals_square_l3513_351367

theorem product_of_powers_equals_square : (1889568 : ℕ)^2 = 3^8 * 3^12 * 2^5 * 2^10 := by
  sorry

end NUMINAMATH_CALUDE_product_of_powers_equals_square_l3513_351367


namespace NUMINAMATH_CALUDE_cake_price_calculation_l3513_351319

theorem cake_price_calculation (smoothie_price : ℝ) (smoothie_count : ℕ) (cake_count : ℕ) (total_revenue : ℝ) :
  smoothie_price = 3 →
  smoothie_count = 40 →
  cake_count = 18 →
  total_revenue = 156 →
  ∃ (cake_price : ℝ), cake_price = 2 ∧ smoothie_price * smoothie_count + cake_price * cake_count = total_revenue :=
by
  sorry

#check cake_price_calculation

end NUMINAMATH_CALUDE_cake_price_calculation_l3513_351319


namespace NUMINAMATH_CALUDE_a_range_l3513_351325

/-- Piecewise function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (2 - a) * x + 1 else a^x

/-- The condition for the function -/
def strictly_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ≠ x₂ → (x₁ - x₂) * (f x₁ - f x₂) > 0

/-- The theorem stating the range of a -/
theorem a_range (a : ℝ) :
  (strictly_increasing (f a)) ↔ (3/2 ≤ a ∧ a < 2) :=
sorry

end NUMINAMATH_CALUDE_a_range_l3513_351325


namespace NUMINAMATH_CALUDE_line_properties_l3513_351350

/-- A line with slope -3 and x-intercept (8, 0) has y-intercept (0, 24) and the point 4 units
    to the left of the x-intercept has coordinates (4, 12) -/
theorem line_properties (f : ℝ → ℝ) (h_slope : ∀ x y, f y - f x = -3 * (y - x))
    (h_x_intercept : f 8 = 0) :
  f 0 = 24 ∧ f 4 = 12 := by
  sorry

end NUMINAMATH_CALUDE_line_properties_l3513_351350


namespace NUMINAMATH_CALUDE_claire_gerbils_l3513_351371

/-- Represents the number of gerbils Claire has -/
def num_gerbils : ℕ := 60

/-- Represents the number of hamsters Claire has -/
def num_hamsters : ℕ := 30

/-- The total number of pets Claire has -/
def total_pets : ℕ := 90

/-- The total number of male pets Claire has -/
def total_male_pets : ℕ := 25

theorem claire_gerbils :
  (num_gerbils + num_hamsters = total_pets) ∧
  (num_gerbils / 4 + num_hamsters / 3 = total_male_pets) →
  num_gerbils = 60 := by
  sorry

end NUMINAMATH_CALUDE_claire_gerbils_l3513_351371


namespace NUMINAMATH_CALUDE_total_toys_count_l3513_351323

/-- The number of toys Mandy has -/
def mandy_toys : ℕ := 20

/-- The number of toys Anna has -/
def anna_toys : ℕ := 3 * mandy_toys

/-- The number of toys Amanda has -/
def amanda_toys : ℕ := anna_toys + 2

/-- The total number of toys -/
def total_toys : ℕ := mandy_toys + anna_toys + amanda_toys

theorem total_toys_count : total_toys = 142 := by sorry

end NUMINAMATH_CALUDE_total_toys_count_l3513_351323


namespace NUMINAMATH_CALUDE_problem_solution_l3513_351305

theorem problem_solution :
  (∃ n : ℕ, n = 4 * 7 + 5 ∧ n = 33) ∧
  (∃ m : ℕ, m * 6 = 300 ∧ m = 50) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3513_351305


namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l3513_351387

/-- Given a circle D with equation x^2 - 4y - 34 = -y^2 + 12x + 74,
    prove that its center (c,d) and radius r satisfy c + d + r = 4 + 2√17 -/
theorem circle_center_radius_sum (x y c d r : ℝ) : 
  (∀ x y, x^2 - 4*y - 34 = -y^2 + 12*x + 74) →
  ((x - c)^2 + (y - d)^2 = r^2) →
  (c + d + r = 4 + 2 * Real.sqrt 17) := by
  sorry


end NUMINAMATH_CALUDE_circle_center_radius_sum_l3513_351387


namespace NUMINAMATH_CALUDE_remainder_divisibility_l3513_351322

theorem remainder_divisibility (N : ℤ) : 
  (∃ k : ℤ, N = 39 * k + 17) → 
  (∃ m : ℤ, N = 13 * m + 4) :=
by sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l3513_351322


namespace NUMINAMATH_CALUDE_hypotenuse_length_l3513_351321

/-- A rectangle with an inscribed circle -/
structure RectangleWithInscribedCircle where
  -- The length of side AB
  ab : ℝ
  -- The length of side BC
  bc : ℝ
  -- The point where the circle touches AB
  p : ℝ × ℝ
  -- The point where the circle touches BC
  q : ℝ × ℝ
  -- The point where the circle touches CD
  r : ℝ × ℝ
  -- The point where the circle touches DA
  s : ℝ × ℝ

/-- The theorem stating the length of the hypotenuse of triangle APD -/
theorem hypotenuse_length (rect : RectangleWithInscribedCircle)
  (h_ab : rect.ab = 20)
  (h_bc : rect.bc = 10) :
  Real.sqrt ((rect.ab - 2 * (rect.ab * rect.bc) / (2 * (rect.ab + rect.bc)))^2 + rect.bc^2) = 50 / 3 :=
by sorry

end NUMINAMATH_CALUDE_hypotenuse_length_l3513_351321


namespace NUMINAMATH_CALUDE_expression_evaluation_l3513_351358

theorem expression_evaluation : (96 / 6) * 3 / 2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3513_351358


namespace NUMINAMATH_CALUDE_fraction_addition_l3513_351320

theorem fraction_addition (d : ℝ) : (5 + 4 * d) / 8 + 3 = (29 + 4 * d) / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l3513_351320


namespace NUMINAMATH_CALUDE_dogs_count_l3513_351347

/-- Represents the number of animals in a pet shop -/
structure PetShop where
  dogs : ℕ
  cats : ℕ
  bunnies : ℕ

/-- The ratio of dogs to cats to bunnies is 4 : 7 : 9 -/
def ratio_condition (shop : PetShop) : Prop :=
  ∃ (x : ℕ), shop.dogs = 4 * x ∧ shop.cats = 7 * x ∧ shop.bunnies = 9 * x

/-- The total number of dogs and bunnies is 364 -/
def total_condition (shop : PetShop) : Prop :=
  shop.dogs + shop.bunnies = 364

/-- Theorem stating that under the given conditions, there are 112 dogs -/
theorem dogs_count (shop : PetShop) 
  (h_ratio : ratio_condition shop) 
  (h_total : total_condition shop) : 
  shop.dogs = 112 := by
  sorry

end NUMINAMATH_CALUDE_dogs_count_l3513_351347


namespace NUMINAMATH_CALUDE_min_value_of_function_l3513_351317

theorem min_value_of_function (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : x + y = 2) :
  (2 / x + 1 / y) ≥ 3 / 2 + Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_function_l3513_351317


namespace NUMINAMATH_CALUDE_matrix_equation_properties_l3513_351353

open Matrix ComplexConjugate

variable {n : ℕ}

def conjugate_transpose (A : Matrix (Fin n) (Fin n) ℂ) : Matrix (Fin n) (Fin n) ℂ :=
  star A

theorem matrix_equation_properties
  (α : ℂ)
  (A : Matrix (Fin n) (Fin n) ℂ)
  (h_alpha : α ≠ 0)
  (h_A : A ≠ 0)
  (h_eq : A ^ 2 + (conjugate_transpose A) ^ 2 = α • (A * conjugate_transpose A)) :
  α.im = 0 ∧ Complex.abs α ≤ 2 ∧ A * conjugate_transpose A = conjugate_transpose A * A := by
  sorry

end NUMINAMATH_CALUDE_matrix_equation_properties_l3513_351353


namespace NUMINAMATH_CALUDE_boat_speed_is_48_l3513_351348

/-- The speed of the stream in kilometers per hour -/
def stream_speed : ℝ := 16

/-- The speed of the boat in still water in kilometers per hour -/
def boat_speed : ℝ := 48

/-- The time taken to row downstream -/
def time_downstream : ℝ := 1

/-- The time taken to row upstream -/
def time_upstream : ℝ := 2 * time_downstream

/-- The theorem stating that the boat's speed in still water is 48 kmph -/
theorem boat_speed_is_48 : 
  (boat_speed + stream_speed) * time_downstream = 
  (boat_speed - stream_speed) * time_upstream :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_is_48_l3513_351348


namespace NUMINAMATH_CALUDE_birthday_box_crayons_l3513_351365

/-- The number of crayons Paul gave away -/
def crayons_given : ℕ := 571

/-- The number of crayons Paul lost -/
def crayons_lost : ℕ := 161

/-- The difference between crayons given away and lost -/
def crayons_difference : ℕ := 410

/-- Theorem: The number of crayons in Paul's birthday box is 732 -/
theorem birthday_box_crayons :
  crayons_given + crayons_lost = 732 ∧
  crayons_given - crayons_lost = crayons_difference :=
by sorry

end NUMINAMATH_CALUDE_birthday_box_crayons_l3513_351365


namespace NUMINAMATH_CALUDE_bridge_building_time_l3513_351376

theorem bridge_building_time
  (workers₁ workers₂ : ℕ)
  (days₁ : ℕ)
  (h_workers₁ : workers₁ = 60)
  (h_workers₂ : workers₂ = 30)
  (h_days₁ : days₁ = 6)
  (h_positive : workers₁ > 0 ∧ workers₂ > 0 ∧ days₁ > 0)
  (h_same_rate : ∀ w : ℕ, w > 0 → ∃ r : ℚ, r > 0 ∧ w * r * days₁ = 1) :
  ∃ days₂ : ℕ, days₂ = 12 ∧ workers₂ * days₂ = workers₁ * days₁ :=
by sorry


end NUMINAMATH_CALUDE_bridge_building_time_l3513_351376


namespace NUMINAMATH_CALUDE_cross_in_square_l3513_351372

theorem cross_in_square (s : ℝ) (h : s > 0) : 
  (2 * (s/2)^2 + 2 * (s/4)^2 = 810) → s = 36 := by
  sorry

end NUMINAMATH_CALUDE_cross_in_square_l3513_351372


namespace NUMINAMATH_CALUDE_distinct_arrangements_count_l3513_351339

/-- A regular six-pointed star -/
structure SixPointedStar :=
  (points : Fin 12)

/-- The number of distinct arrangements of 12 different objects on a regular six-pointed star -/
def num_arrangements (star : SixPointedStar) : ℕ :=
  Nat.factorial 12 / 12

theorem distinct_arrangements_count : 
  ∀ (star : SixPointedStar), num_arrangements star = 39916800 := by
  sorry

end NUMINAMATH_CALUDE_distinct_arrangements_count_l3513_351339


namespace NUMINAMATH_CALUDE_polar_to_cartesian_circle_l3513_351364

/-- The polar equation ρ = 5 sin θ represents a circle in Cartesian coordinates. -/
theorem polar_to_cartesian_circle :
  ∀ (x y : ℝ), (∃ (ρ θ : ℝ), ρ = 5 * Real.sin θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
  ∃ (h k r : ℝ), (x - h)^2 + (y - k)^2 = r^2 :=
by sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_circle_l3513_351364


namespace NUMINAMATH_CALUDE_a_equals_two_l3513_351397

theorem a_equals_two (a : ℝ) : 
  (∀ x : ℝ, (a - 2) * x + 1 > 0) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_a_equals_two_l3513_351397


namespace NUMINAMATH_CALUDE_smallest_cube_multiple_l3513_351328

theorem smallest_cube_multiple : ∃ (x : ℕ), x > 0 ∧ (∃ (M : ℤ), 2520 * x = M^3) ∧ 
  (∀ (y : ℕ), y > 0 → (∃ (N : ℤ), 2520 * y = N^3) → x ≤ y) ∧ x = 3675 := by
  sorry

end NUMINAMATH_CALUDE_smallest_cube_multiple_l3513_351328


namespace NUMINAMATH_CALUDE_salary_increase_percentage_l3513_351370

def current_salary : ℝ := 300

theorem salary_increase_percentage : 
  (∃ (increase_percent : ℝ), 
    current_salary * (1 + 0.16) = 348 ∧ 
    current_salary * (1 + increase_percent / 100) = 330 ∧ 
    increase_percent = 10) :=
by sorry

end NUMINAMATH_CALUDE_salary_increase_percentage_l3513_351370


namespace NUMINAMATH_CALUDE_salt_solution_volume_l3513_351378

/-- Proves that for a salt solution with given conditions, the initial volume is 56 gallons -/
theorem salt_solution_volume 
  (initial_concentration : ℝ) 
  (final_concentration : ℝ) 
  (added_water : ℝ) 
  (h1 : initial_concentration = 0.10)
  (h2 : final_concentration = 0.08)
  (h3 : added_water = 14) :
  ∃ (initial_volume : ℝ), 
    initial_volume * initial_concentration = 
    (initial_volume + added_water) * final_concentration ∧ 
    initial_volume = 56 := by
  sorry

end NUMINAMATH_CALUDE_salt_solution_volume_l3513_351378


namespace NUMINAMATH_CALUDE_min_abs_sum_with_constraints_l3513_351307

theorem min_abs_sum_with_constraints (α β γ : ℝ) 
  (sum_constraint : α + β + γ = 2)
  (product_constraint : α * β * γ = 4) :
  ∃ v : ℝ, v = 6 ∧ ∀ α' β' γ' : ℝ, 
    α' + β' + γ' = 2 → α' * β' * γ' = 4 → 
    |α'| + |β'| + |γ'| ≥ v :=
sorry

end NUMINAMATH_CALUDE_min_abs_sum_with_constraints_l3513_351307


namespace NUMINAMATH_CALUDE_fifth_week_consumption_l3513_351394

/-- Represents the vegetable consumption for a day -/
structure VegetableConsumption where
  asparagus : Float
  broccoli : Float
  cauliflower : Float
  spinach : Float
  kale : Float
  zucchini : Float
  carrots : Float

/-- Calculates the total vegetable consumption for a day -/
def totalConsumption (vc : VegetableConsumption) : Float :=
  vc.asparagus + vc.broccoli + vc.cauliflower + vc.spinach + vc.kale + vc.zucchini + vc.carrots

/-- Initial weekday consumption -/
def initialWeekday : VegetableConsumption := {
  asparagus := 0.25, broccoli := 0.25, cauliflower := 0.5,
  spinach := 0, kale := 0, zucchini := 0, carrots := 0
}

/-- Initial weekend consumption -/
def initialWeekend : VegetableConsumption := {
  asparagus := 0.3, broccoli := 0.4, cauliflower := 0.6,
  spinach := 0, kale := 0, zucchini := 0, carrots := 0
}

/-- Updated weekday consumption -/
def updatedWeekday : VegetableConsumption := {
  asparagus := initialWeekday.asparagus * 2,
  broccoli := initialWeekday.broccoli * 3,
  cauliflower := initialWeekday.cauliflower * 1.75,
  spinach := 0.5,
  kale := 0, zucchini := 0, carrots := 0
}

/-- Updated Saturday consumption -/
def updatedSaturday : VegetableConsumption := {
  asparagus := initialWeekend.asparagus,
  broccoli := initialWeekend.broccoli,
  cauliflower := initialWeekend.cauliflower,
  spinach := 0,
  kale := 1,
  zucchini := 0.3,
  carrots := 0
}

/-- Updated Sunday consumption -/
def updatedSunday : VegetableConsumption := {
  asparagus := initialWeekend.asparagus,
  broccoli := initialWeekend.broccoli,
  cauliflower := initialWeekend.cauliflower,
  spinach := 0,
  kale := 0,
  zucchini := 0,
  carrots := 0.5
}

/-- Theorem: The total vegetable consumption in the fifth week is 17.225 pounds -/
theorem fifth_week_consumption :
  5 * totalConsumption updatedWeekday +
  totalConsumption updatedSaturday +
  totalConsumption updatedSunday = 17.225 := by
  sorry

end NUMINAMATH_CALUDE_fifth_week_consumption_l3513_351394


namespace NUMINAMATH_CALUDE_perpendicular_line_to_cosine_tangent_l3513_351310

open Real

/-- The equation of a line perpendicular to the tangent of y = cos x at (π/3, 1/2) --/
theorem perpendicular_line_to_cosine_tangent :
  let f : ℝ → ℝ := fun x ↦ cos x
  let p : ℝ × ℝ := (π / 3, 1 / 2)
  let tangent_slope : ℝ := -sin (π / 3)
  let perpendicular_slope : ℝ := -1 / tangent_slope
  let line_equation : ℝ → ℝ → ℝ := fun x y ↦ 2 * x - sqrt 3 * y - 2 * π / 3 + sqrt 3 / 2
  (f (π / 3) = 1 / 2) →
  (perpendicular_slope = 2 / sqrt 3) →
  (∀ x y, line_equation x y = 0 ↔ y - p.2 = perpendicular_slope * (x - p.1)) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_to_cosine_tangent_l3513_351310


namespace NUMINAMATH_CALUDE_min_value_expression_l3513_351388

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1/2) :
  a^2 + 4*a*b + 9*b^2 + 8*b*c + 3*c^2 ≥ 13.5 ∧
  ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ a₀ * b₀ * c₀ = 1/2 ∧
    a₀^2 + 4*a₀*b₀ + 9*b₀^2 + 8*b₀*c₀ + 3*c₀^2 = 13.5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3513_351388


namespace NUMINAMATH_CALUDE_appetizer_cost_per_person_l3513_351338

def potato_chip_cost : ℝ := 1.00
def creme_fraiche_cost : ℝ := 5.00
def caviar_cost : ℝ := 73.00
def num_people : ℕ := 3
def num_potato_chip_bags : ℕ := 3

theorem appetizer_cost_per_person :
  (num_potato_chip_bags * potato_chip_cost + creme_fraiche_cost + caviar_cost) / num_people = 27.00 := by
  sorry

end NUMINAMATH_CALUDE_appetizer_cost_per_person_l3513_351338


namespace NUMINAMATH_CALUDE_S_max_l3513_351360

/-- The general term of the sequence -/
def a (n : ℕ) : ℤ := 26 - 2 * n

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ) : ℤ := n * (26 - n)

/-- The theorem stating that S is maximized when n is 12 or 13 -/
theorem S_max : ∀ k : ℕ, S k ≤ max (S 12) (S 13) :=
sorry

end NUMINAMATH_CALUDE_S_max_l3513_351360


namespace NUMINAMATH_CALUDE_forty_ab_value_l3513_351329

theorem forty_ab_value (a b : ℝ) (h : 4 * a = 5 * b ∧ 5 * b = 30) : 40 * a * b = 1800 := by
  sorry

end NUMINAMATH_CALUDE_forty_ab_value_l3513_351329


namespace NUMINAMATH_CALUDE_subset_M_N_l3513_351337

theorem subset_M_N : ∀ (x y : ℝ), (|x| + |y| ≤ 1) → (x^2 + y^2 ≤ |x| + |y|) := by
  sorry

end NUMINAMATH_CALUDE_subset_M_N_l3513_351337


namespace NUMINAMATH_CALUDE_negation_equivalence_l3513_351334

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x > 1 ∧ x^2 + 2 < 0) ↔ (∀ x : ℝ, x > 1 → x^2 + 2 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3513_351334


namespace NUMINAMATH_CALUDE_calculate_expression_l3513_351330

theorem calculate_expression : -Real.sqrt 4 + |Real.sqrt 2 - 2| - 202 * 3^0 = -Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3513_351330


namespace NUMINAMATH_CALUDE_simplify_expression_l3513_351303

theorem simplify_expression : (27 * 10^9) / (9 * 10^2) = 3000000 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3513_351303


namespace NUMINAMATH_CALUDE_triangle_perimeter_not_85_l3513_351335

theorem triangle_perimeter_not_85 (a b c : ℝ) : 
  a = 24 → b = 18 → a + b + c > a + b → a + c > b → b + c > a → a + b + c ≠ 85 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_not_85_l3513_351335


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l3513_351374

theorem quadratic_root_problem (p d c : ℝ) : 
  c = 1 / 216 →
  (∀ x, p * x^2 + d * x = 1 ↔ x = -2 ∨ x = 216 * c) →
  d = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l3513_351374


namespace NUMINAMATH_CALUDE_children_who_got_on_bus_stop_l3513_351375

/-- The number of children who got on the bus at a stop -/
def children_who_got_on (initial : ℕ) (final : ℕ) : ℕ :=
  final - initial

/-- Proof that 38 children got on the bus at the stop -/
theorem children_who_got_on_bus_stop : children_who_got_on 26 64 = 38 := by
  sorry

end NUMINAMATH_CALUDE_children_who_got_on_bus_stop_l3513_351375


namespace NUMINAMATH_CALUDE_bus_ride_difference_l3513_351344

theorem bus_ride_difference (vince_ride zachary_ride : ℝ) 
  (h1 : vince_ride = 0.62)
  (h2 : zachary_ride = 0.5) :
  vince_ride - zachary_ride = 0.12 := by
sorry

end NUMINAMATH_CALUDE_bus_ride_difference_l3513_351344


namespace NUMINAMATH_CALUDE_quadratic_shift_sum_l3513_351316

/-- Given a quadratic function f(x) = 3x^2 + 5x - 2, prove that when it's shifted 5 units to the left,
    resulting in a new quadratic function g(x) = ax^2 + bx + c, then a + b + c = 136. -/
theorem quadratic_shift_sum (f g : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f x = 3 * x^2 + 5 * x - 2) →
  (∀ x, g x = f (x + 5)) →
  (∀ x, g x = a * x^2 + b * x + c) →
  a + b + c = 136 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_shift_sum_l3513_351316


namespace NUMINAMATH_CALUDE_part1_part2_l3513_351354

/-- Definition of a golden equation -/
def is_golden_equation (a b c : ℝ) : Prop := a ≠ 0 ∧ a - b + c = 0

/-- Part 1: Prove that 2x^2 + 5x + 3 = 0 is a golden equation -/
theorem part1 : is_golden_equation 2 5 3 := by sorry

/-- Part 2: Prove that if 3x^2 - ax + b = 0 is a golden equation and a is a root, then a = -1 or a = 3/2 -/
theorem part2 (a b : ℝ) (h1 : is_golden_equation 3 (-a) b) (h2 : 3 * a^2 - a * a + b = 0) :
  a = -1 ∨ a = 3/2 := by sorry

end NUMINAMATH_CALUDE_part1_part2_l3513_351354


namespace NUMINAMATH_CALUDE_candy_bar_sales_proof_l3513_351390

/-- The number of candy bars sold on the first day -/
def first_day_sales : ℕ := 190

/-- The number of days Sol sells candy bars in a week -/
def days_per_week : ℕ := 6

/-- The cost of each candy bar in cents -/
def candy_bar_cost : ℕ := 10

/-- The increase in candy bar sales each day after the first day -/
def daily_increase : ℕ := 4

/-- The total earnings in cents for the week -/
def total_earnings : ℕ := 1200

theorem candy_bar_sales_proof :
  (first_day_sales * days_per_week + 
   (daily_increase * (days_per_week - 1) * days_per_week) / 2) * 
  candy_bar_cost = total_earnings :=
sorry

end NUMINAMATH_CALUDE_candy_bar_sales_proof_l3513_351390


namespace NUMINAMATH_CALUDE_sin_plus_cos_zero_equiv_cos_2x_over_sin_minus_cos_zero_l3513_351327

open Real

theorem sin_plus_cos_zero_equiv_cos_2x_over_sin_minus_cos_zero :
  ∀ x : ℝ, (sin x + cos x = 0) ↔ ((cos (2 * x)) / (sin x - cos x) = 0) :=
by sorry

end NUMINAMATH_CALUDE_sin_plus_cos_zero_equiv_cos_2x_over_sin_minus_cos_zero_l3513_351327


namespace NUMINAMATH_CALUDE_least_integer_with_leading_six_and_fraction_l3513_351341

theorem least_integer_with_leading_six_and_fraction (x : ℕ) : x ≥ 625 →
  (∃ n : ℕ, ∃ y : ℕ, 
    x = 6 * 10^n + y ∧ 
    y < 10^n ∧ 
    y = x / 25) →
  x = 625 :=
sorry

end NUMINAMATH_CALUDE_least_integer_with_leading_six_and_fraction_l3513_351341


namespace NUMINAMATH_CALUDE_changhyeok_snacks_l3513_351368

theorem changhyeok_snacks :
  ∀ (s d : ℕ),
  s + d = 12 →
  1000 * s + 1300 * d = 15000 →
  s = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_changhyeok_snacks_l3513_351368


namespace NUMINAMATH_CALUDE_largest_negative_root_l3513_351332

noncomputable def α : ℝ := Real.arctan (4 / 13)
noncomputable def β : ℝ := Real.arctan (8 / 11)

def equation (x : ℝ) : Prop :=
  4 * Real.sin (3 * x) + 13 * Real.cos (3 * x) = 8 * Real.sin x + 11 * Real.cos x

theorem largest_negative_root :
  ∃ (x : ℝ), x < 0 ∧ equation x ∧ 
  ∀ (y : ℝ), y < 0 → equation y → y ≤ x ∧
  x = (α - β) / 2 :=
sorry

end NUMINAMATH_CALUDE_largest_negative_root_l3513_351332


namespace NUMINAMATH_CALUDE_unique_valid_prism_l3513_351351

/-- A right rectangular prism with integer side lengths -/
structure RectPrism where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  h1 : a ≤ b
  h2 : b ≤ c

/-- Predicate for a valid division of a prism -/
def validDivision (p : RectPrism) : Prop :=
  ∃ (k : ℚ), 0 < k ∧ k < 1 ∧
  ((k * p.a.val = p.a.val ∧ k * p.b.val = p.a.val) ∨
   (k * p.b.val = p.b.val ∧ k * p.c.val = p.b.val) ∨
   (k * p.c.val = p.c.val ∧ k * p.a.val = p.c.val))

theorem unique_valid_prism :
  ∃! (p : RectPrism), p.b = 101 ∧ validDivision p :=
sorry

end NUMINAMATH_CALUDE_unique_valid_prism_l3513_351351


namespace NUMINAMATH_CALUDE_tv_sales_effect_l3513_351324

theorem tv_sales_effect (price_reduction : ℝ) (sales_increase : ℝ) : 
  price_reduction = 0.22 → sales_increase = 0.86 → 
  (1 - price_reduction) * (1 + sales_increase) - 1 = 0.4518 := by
  sorry

end NUMINAMATH_CALUDE_tv_sales_effect_l3513_351324


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3513_351383

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 15 → b = 36 → c^2 = a^2 + b^2 → c = 39 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3513_351383


namespace NUMINAMATH_CALUDE_same_color_probability_l3513_351377

/-- The probability of drawing balls of the same color from two bags -/
theorem same_color_probability (bagA_white bagA_red bagB_white bagB_red : ℕ) :
  bagA_white = 8 →
  bagA_red = 4 →
  bagB_white = 6 →
  bagB_red = 6 →
  (bagA_white / (bagA_white + bagA_red : ℚ)) * (bagB_white / (bagB_white + bagB_red : ℚ)) +
  (bagA_red / (bagA_white + bagA_red : ℚ)) * (bagB_red / (bagB_white + bagB_red : ℚ)) = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_same_color_probability_l3513_351377


namespace NUMINAMATH_CALUDE_non_coincident_terminal_sides_l3513_351340

-- Define a function to check if two angles have coincident terminal sides
def coincident_terminal_sides (α β : ℝ) : Prop :=
  ∃ k : ℤ, β = α + k * 360

-- Define the sets of angles
def angle_sets : List (ℝ × ℝ) :=
  [(60, -300), (230, 950), (1050, -300), (-1000, 80)]

-- State the theorem
theorem non_coincident_terminal_sides :
  ∃! pair : ℝ × ℝ, pair ∈ angle_sets ∧ ¬ coincident_terminal_sides pair.1 pair.2 :=
by sorry

end NUMINAMATH_CALUDE_non_coincident_terminal_sides_l3513_351340


namespace NUMINAMATH_CALUDE_problem_solution_l3513_351304

theorem problem_solution : 
  (2 * (Real.sqrt 3 - Real.sqrt 5) + 3 * (Real.sqrt 3 + Real.sqrt 5) = 5 * Real.sqrt 3 + Real.sqrt 5) ∧
  (-1^2 - |1 - Real.sqrt 3| + (8 : Real)^(1/3) - (-3) * Real.sqrt 9 = 11 - Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3513_351304


namespace NUMINAMATH_CALUDE_factor_of_expression_l3513_351357

theorem factor_of_expression (x y z : ℝ) :
  ∃ (k : ℝ), x^2 - y^2 - z^2 + 2*y*z + 3*x + 2*y - 4*z = (x + y - z) * k := by
  sorry

end NUMINAMATH_CALUDE_factor_of_expression_l3513_351357


namespace NUMINAMATH_CALUDE_polynomial_inequality_l3513_351336

theorem polynomial_inequality (m : ℚ) : 5 * m^2 - 8 * m + 1 > 4 * m^2 - 8 * m - 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_inequality_l3513_351336


namespace NUMINAMATH_CALUDE_georginas_parrots_l3513_351346

/-- Represents a parrot with its phrases and learning rate -/
structure Parrot where
  name : String
  current_phrases : ℕ
  phrases_per_week : ℕ
  initial_phrases : ℕ

/-- Calculates the number of weekdays since a parrot was bought -/
def weekdays_since_bought (p : Parrot) : ℕ :=
  ((p.current_phrases - p.initial_phrases + p.phrases_per_week - 1) / p.phrases_per_week) * 5

/-- The main theorem about Georgina's parrots -/
theorem georginas_parrots :
  let polly : Parrot := { name := "Polly", current_phrases := 17, phrases_per_week := 2, initial_phrases := 3 }
  let pedro : Parrot := { name := "Pedro", current_phrases := 12, phrases_per_week := 3, initial_phrases := 0 }
  let penelope : Parrot := { name := "Penelope", current_phrases := 8, phrases_per_week := 1, initial_phrases := 0 }
  let pascal : Parrot := { name := "Pascal", current_phrases := 20, phrases_per_week := 4, initial_phrases := 1 }
  weekdays_since_bought polly = 35 ∧
  weekdays_since_bought pedro = 20 ∧
  weekdays_since_bought penelope = 40 ∧
  weekdays_since_bought pascal = 25 := by
  sorry

end NUMINAMATH_CALUDE_georginas_parrots_l3513_351346


namespace NUMINAMATH_CALUDE_divisibility_proof_l3513_351342

theorem divisibility_proof (x y a b S : ℤ) : 
  x + y = S → 
  S ∣ (a * x + b * y) → 
  S ∣ (b * x + a * y) := by
sorry

end NUMINAMATH_CALUDE_divisibility_proof_l3513_351342


namespace NUMINAMATH_CALUDE_line_slope_proof_l3513_351326

theorem line_slope_proof (x y : ℝ) : 
  (((Real.sqrt 3) / 3) * x + y - 7 = 0) → 
  (∃ m : ℝ, m = -(Real.sqrt 3) / 3 ∧ y = m * x + 7) := by
  sorry

end NUMINAMATH_CALUDE_line_slope_proof_l3513_351326


namespace NUMINAMATH_CALUDE_monomial_properties_l3513_351359

/-- Represents a monomial with coefficient and exponents for variables a, b, and c -/
structure Monomial where
  coeff : ℤ
  a_exp : ℕ
  b_exp : ℕ
  c_exp : ℕ

/-- Calculates the degree of a monomial -/
def degree (m : Monomial) : ℕ :=
  m.a_exp + m.b_exp + m.c_exp

/-- The given monomial -7a³b⁴c -/
def given_monomial : Monomial :=
  { coeff := -7
    a_exp := 3
    b_exp := 4
    c_exp := 1 }

theorem monomial_properties :
  given_monomial.coeff = -7 ∧ degree given_monomial = 8 := by
  sorry

end NUMINAMATH_CALUDE_monomial_properties_l3513_351359


namespace NUMINAMATH_CALUDE_perfect_square_condition_l3513_351309

theorem perfect_square_condition (k : ℤ) : 
  (∀ x : ℤ, ∃ y : ℤ, x^2 - 2*(k+1)*x + 4 = y^2) → (k = -3 ∨ k = 1) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l3513_351309


namespace NUMINAMATH_CALUDE_negative_difference_l3513_351392

theorem negative_difference (a b : ℝ) : -(a - b) = -a + b := by
  sorry

end NUMINAMATH_CALUDE_negative_difference_l3513_351392


namespace NUMINAMATH_CALUDE_cos_pi_minus_alpha_l3513_351356

theorem cos_pi_minus_alpha (α : Real) (h : Real.sin (π / 2 + α) = 1 / 3) :
  Real.cos (π - α) = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_minus_alpha_l3513_351356


namespace NUMINAMATH_CALUDE_fish_population_estimate_l3513_351373

/-- Represents the number of fish in a pond given certain sampling conditions -/
def fish_in_pond (initial_caught : ℕ) (second_caught : ℕ) (marked_in_second : ℕ) : ℕ :=
  (initial_caught * second_caught) / marked_in_second

/-- Theorem stating that under given conditions, there are approximately 1200 fish in the pond -/
theorem fish_population_estimate :
  let initial_caught := 120
  let second_caught := 100
  let marked_in_second := 10
  fish_in_pond initial_caught second_caught marked_in_second = 1200 := by
  sorry

#eval fish_in_pond 120 100 10

end NUMINAMATH_CALUDE_fish_population_estimate_l3513_351373


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3513_351302

/-- Given a hyperbola E with equation x²/a² - y²/b² = 1 (where a > 0 and b > 0),
    if one of its asymptotes has a slope of 30°, then its eccentricity is 2√3/3. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (b / a = Real.tan (π / 6)) →
  Real.sqrt (1 + (b / a)^2) = 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3513_351302


namespace NUMINAMATH_CALUDE_initial_persons_count_l3513_351379

/-- The number of persons initially in the group. -/
def initial_persons : ℕ := sorry

/-- The average weight increase when a new person joins the group. -/
def avg_weight_increase : ℚ := 7/2

/-- The weight difference between the new person and the replaced person. -/
def weight_difference : ℚ := 28

theorem initial_persons_count : initial_persons = 8 := by
  have h1 : (initial_persons : ℚ) * avg_weight_increase = weight_difference := by sorry
  sorry

end NUMINAMATH_CALUDE_initial_persons_count_l3513_351379


namespace NUMINAMATH_CALUDE_double_counted_page_l3513_351355

theorem double_counted_page (n : ℕ) : 
  (n * (n + 1)) / 2 + 80 = 2550 → 
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → 
  (n * (n + 1)) / 2 + k = 2550 → 
  k = 80 := by
sorry

end NUMINAMATH_CALUDE_double_counted_page_l3513_351355


namespace NUMINAMATH_CALUDE_commute_days_calculation_l3513_351391

theorem commute_days_calculation (x : ℕ) 
  (h1 : x > 0)  -- Ensure x is positive
  (h2 : ∃ a b c : ℕ, 
    a + b + c = x ∧  -- Total days
    b + c = 6 ∧      -- Bus to work
    a + c = 18 ∧     -- Bus from work
    a + b = 14) :    -- Train commutes
  x = 19 := by
sorry

end NUMINAMATH_CALUDE_commute_days_calculation_l3513_351391


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3513_351306

/-- An arithmetic sequence with its sum satisfying a specific quadratic equation -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum of first n terms
  A : ℝ      -- Coefficient of n^2
  B : ℝ      -- Coefficient of n
  h1 : A ≠ 0
  h2 : ∀ n : ℕ, a n + S n = A * n^2 + B * n + 1

/-- The main theorem: if an arithmetic sequence satisfies the given condition, then (B-1)/A = 3 -/
theorem arithmetic_sequence_property (seq : ArithmeticSequence) : (seq.B - 1) / seq.A = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3513_351306


namespace NUMINAMATH_CALUDE_license_plate_count_l3513_351398

/-- The number of possible digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of possible letters (A-Z) -/
def num_letters : ℕ := 26

/-- The number of digits in a license plate -/
def digits_count : ℕ := 5

/-- The number of letters in a license plate -/
def letters_count : ℕ := 3

/-- The number of positions where the letter block can be placed -/
def letter_block_positions : ℕ := digits_count + 1

/-- The total number of distinct license plates -/
def total_license_plates : ℕ := 
  letter_block_positions * (num_digits ^ digits_count) * (num_letters ^ letters_count)

theorem license_plate_count : total_license_plates = 10584576000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l3513_351398


namespace NUMINAMATH_CALUDE_jimin_tangerines_l3513_351349

/-- Given an initial number of tangerines and a number of eaten tangerines,
    calculate the number of tangerines left. -/
def tangerines_left (initial : ℕ) (eaten : ℕ) : ℕ :=
  initial - eaten

/-- Theorem stating that given 12 initial tangerines and 7 eaten tangerines,
    the number of tangerines left is 5. -/
theorem jimin_tangerines :
  tangerines_left 12 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_jimin_tangerines_l3513_351349


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3513_351331

def A : Set ℕ := {x | (x + 1) * (x - 2) = 0}
def B : Set ℕ := {2, 4, 5}

theorem union_of_A_and_B : A ∪ B = {2, 4, 5} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3513_351331


namespace NUMINAMATH_CALUDE_divisibility_implies_sum_ten_l3513_351318

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def number (C F : ℕ) : ℕ := C * 1000000 + 854000 + F * 100 + 72

theorem divisibility_implies_sum_ten (C F : ℕ) 
  (h_C : is_digit C) (h_F : is_digit F) 
  (h_div_8 : number C F % 8 = 0) 
  (h_div_9 : number C F % 9 = 0) : 
  C + F = 10 := by
sorry

end NUMINAMATH_CALUDE_divisibility_implies_sum_ten_l3513_351318


namespace NUMINAMATH_CALUDE_min_steps_even_correct_min_steps_odd_correct_l3513_351300

-- Define the stone arrangement
structure StoneArrangement where
  k : Nat
  n : Nat
  stones : List Nat

-- Define a step
def step (arrangement : StoneArrangement) : StoneArrangement := sorry

-- Define the minimum number of steps for even n
def min_steps_even (k : Nat) (n : Nat) : Nat :=
  (n^2 * k * (k-1)) / 4

-- Define the minimum number of steps for odd n and k = 3
def min_steps_odd (n : Nat) : Nat :=
  let q := (n - 1) / 2
  n^2 + 2 * q * (q + 1)

-- Theorem for even n
theorem min_steps_even_correct (k n : Nat) (h1 : k ≥ 2) (h2 : n % 2 = 0) :
  ∀ (arrangement : StoneArrangement),
    arrangement.k = k ∧ arrangement.n = n →
    ∃ (m : Nat), m ≤ min_steps_even k n ∧
      ∃ (final_arrangement : StoneArrangement),
        final_arrangement = (step^[m] arrangement) ∧
        -- The n stones of the same color are together in final_arrangement
        sorry := by sorry

-- Theorem for odd n and k = 3
theorem min_steps_odd_correct (n : Nat) (h1 : n % 2 = 1) :
  ∀ (arrangement : StoneArrangement),
    arrangement.k = 3 ∧ arrangement.n = n →
    ∃ (m : Nat), m ≤ min_steps_odd n ∧
      ∃ (final_arrangement : StoneArrangement),
        final_arrangement = (step^[m] arrangement) ∧
        -- The n stones of the same color are together in final_arrangement
        sorry := by sorry

end NUMINAMATH_CALUDE_min_steps_even_correct_min_steps_odd_correct_l3513_351300


namespace NUMINAMATH_CALUDE_nested_a_value_l3513_351396

-- Define the function a
def a (k : ℕ) : ℕ := (k + 1)^2

-- State the theorem
theorem nested_a_value :
  let k : ℕ := 1
  a (a (a (a k))) = 458329 := by
  sorry

end NUMINAMATH_CALUDE_nested_a_value_l3513_351396


namespace NUMINAMATH_CALUDE_not_valid_prism_diagonals_5_6_9_not_valid_prism_diagonals_7_8_11_l3513_351384

/-- A function to check if three positive real numbers can represent the lengths of external diagonals of a right regular prism -/
def is_valid_prism_diagonals (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a^2 + b^2 > c^2 ∧
  b^2 + c^2 > a^2 ∧
  c^2 + a^2 > b^2

/-- Theorem stating that {5,6,9} cannot be the lengths of external diagonals of a right regular prism -/
theorem not_valid_prism_diagonals_5_6_9 :
  ¬ is_valid_prism_diagonals 5 6 9 :=
sorry

/-- Theorem stating that {7,8,11} cannot be the lengths of external diagonals of a right regular prism -/
theorem not_valid_prism_diagonals_7_8_11 :
  ¬ is_valid_prism_diagonals 7 8 11 :=
sorry

end NUMINAMATH_CALUDE_not_valid_prism_diagonals_5_6_9_not_valid_prism_diagonals_7_8_11_l3513_351384


namespace NUMINAMATH_CALUDE_train_length_l3513_351301

/-- The length of a train given its speed, a man's walking speed, and the time taken to pass the man. -/
theorem train_length (train_speed : Real) (man_speed : Real) (passing_time : Real) :
  train_speed = 63 →
  man_speed = 3 →
  passing_time = 44.99640028797696 →
  (train_speed - man_speed) * passing_time * (1000 / 3600) = 750 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3513_351301


namespace NUMINAMATH_CALUDE_even_periodic_function_monotonicity_l3513_351361

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def monotone_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

theorem even_periodic_function_monotonicity
  (f : ℝ → ℝ) (h_even : is_even f) (h_period : has_period f 2) :
  (monotone_on f (Set.Icc 0 1)) ↔ (∀ x ∈ Set.Icc 3 4, ∀ y ∈ Set.Icc 3 4, x ≤ y → f y ≤ f x) :=
sorry

end NUMINAMATH_CALUDE_even_periodic_function_monotonicity_l3513_351361


namespace NUMINAMATH_CALUDE_rectangle_to_square_l3513_351313

theorem rectangle_to_square (k : ℕ) (h1 : k > 7) :
  (∃ (n : ℕ), k * (k - 7) = n^2) → (∃ (n : ℕ), k * (k - 7) = n^2 ∧ n = 24) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_to_square_l3513_351313


namespace NUMINAMATH_CALUDE_product_sum_of_three_numbers_l3513_351311

theorem product_sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 62) 
  (h2 : a + b + c = 18) : 
  a*b + b*c + a*c = 131 := by
sorry

end NUMINAMATH_CALUDE_product_sum_of_three_numbers_l3513_351311


namespace NUMINAMATH_CALUDE_fred_baseball_cards_l3513_351382

def final_baseball_cards (initial : ℕ) (sold : ℕ) (traded : ℕ) (bought : ℕ) : ℕ :=
  initial - sold - traded + bought

theorem fred_baseball_cards : final_baseball_cards 25 7 3 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_fred_baseball_cards_l3513_351382


namespace NUMINAMATH_CALUDE_ann_blocks_proof_l3513_351343

def ann_initial_blocks (found_blocks final_blocks : ℕ) : ℕ :=
  final_blocks - found_blocks

theorem ann_blocks_proof (found_blocks final_blocks : ℕ) 
  (h1 : found_blocks = 44) 
  (h2 : final_blocks = 53) : 
  ann_initial_blocks found_blocks final_blocks = 9 := by
  sorry

end NUMINAMATH_CALUDE_ann_blocks_proof_l3513_351343


namespace NUMINAMATH_CALUDE_sum_of_digits_square_999999999_l3513_351389

def digit_sum (n : ℕ) : ℕ := 
  if n < 10 then n else n % 10 + digit_sum (n / 10)

def repeated_nines (n : ℕ) : ℕ :=
  (10^n - 1) / 9

theorem sum_of_digits_square_999999999 :
  digit_sum ((repeated_nines 9)^2) = 81 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_square_999999999_l3513_351389


namespace NUMINAMATH_CALUDE_cricket_innings_count_l3513_351369

/-- 
Proves that a cricket player has played 10 innings, given:
1. The current average is 32 runs.
2. Scoring 76 runs in the next innings will increase the average by 4 runs.
-/
theorem cricket_innings_count : 
  ∀ (n : ℕ) (current_average : ℚ) (next_innings_score : ℕ) (average_increase : ℚ),
    current_average = 32 →
    next_innings_score = 76 →
    average_increase = 4 →
    (n * current_average + next_innings_score : ℚ) / (n + 1) = current_average + average_increase →
    n = 10 := by
  sorry

end NUMINAMATH_CALUDE_cricket_innings_count_l3513_351369


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3513_351333

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B : Set ℝ := {x | x > 0}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 < x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3513_351333


namespace NUMINAMATH_CALUDE_notebook_ratio_l3513_351352

theorem notebook_ratio (total_students : ℕ) (total_notebooks : ℕ) 
  (h1 : total_students = 28)
  (h2 : total_notebooks = 112)
  (h3 : ∃ (x y : ℕ), x + y = total_students ∧ y = total_students / 2 ∧ 5 * x + 3 * y = total_notebooks) :
  ∃ (x y : ℕ), x = y ∧ x + y = total_students ∧ 5 * x + 3 * y = total_notebooks := by
  sorry

end NUMINAMATH_CALUDE_notebook_ratio_l3513_351352


namespace NUMINAMATH_CALUDE_price_adjustment_l3513_351366

-- Define the original price
variable (P : ℝ)
-- Define the percentage x
variable (x : ℝ)

-- Theorem statement
theorem price_adjustment (h : P * (1 + x/100) * (1 - x/100) = 0.75 * P) : x = 50 := by
  sorry

end NUMINAMATH_CALUDE_price_adjustment_l3513_351366


namespace NUMINAMATH_CALUDE_yarn_length_problem_l3513_351345

theorem yarn_length_problem (green_length red_length total_length : ℕ) : 
  green_length = 156 →
  red_length = 3 * green_length + 8 →
  total_length = green_length + red_length →
  total_length = 632 := by
  sorry

end NUMINAMATH_CALUDE_yarn_length_problem_l3513_351345


namespace NUMINAMATH_CALUDE_no_integer_n_makes_complex_fifth_power_real_l3513_351312

theorem no_integer_n_makes_complex_fifth_power_real : 
  ¬∃ (n : ℤ), (Complex.I : ℂ).im * ((n + 2 * Complex.I)^5).im = 0 := by sorry

end NUMINAMATH_CALUDE_no_integer_n_makes_complex_fifth_power_real_l3513_351312


namespace NUMINAMATH_CALUDE_abs_one_plus_i_over_i_l3513_351308

def i : ℂ := Complex.I

theorem abs_one_plus_i_over_i : Complex.abs ((1 + i) / i) = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_abs_one_plus_i_over_i_l3513_351308


namespace NUMINAMATH_CALUDE_gcd_m_n_equals_two_l3513_351362

def m : ℕ := 22222222
def n : ℕ := 444444444

theorem gcd_m_n_equals_two : Nat.gcd m n = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_m_n_equals_two_l3513_351362


namespace NUMINAMATH_CALUDE_average_of_abc_is_three_l3513_351395

theorem average_of_abc_is_three (A B C : ℝ) 
  (eq1 : 2003 * C - 4004 * A = 8008)
  (eq2 : 2003 * B + 6006 * A = 10010)
  (eq3 : B = 2 * A - 6) :
  (A + B + C) / 3 = 3 := by
sorry

end NUMINAMATH_CALUDE_average_of_abc_is_three_l3513_351395


namespace NUMINAMATH_CALUDE_quadratic_roots_interlace_l3513_351399

theorem quadratic_roots_interlace (p1 p2 q1 q2 : ℝ) 
  (h : (q1 - q2)^2 + (p1 - p2)*(p1*q2 - p2*q1) < 0) :
  ∃ (α1 β1 α2 β2 : ℝ),
    (∀ x, x^2 + p1*x + q1 = (x - α1) * (x - β1)) ∧
    (∀ x, x^2 + p2*x + q2 = (x - α2) * (x - β2)) ∧
    ((α1 < α2 ∧ α2 < β1 ∧ β1 < β2) ∨ (α2 < α1 ∧ α1 < β2 ∧ β2 < β1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_interlace_l3513_351399


namespace NUMINAMATH_CALUDE_pizza_eating_l3513_351393

theorem pizza_eating (n : ℕ) (initial_pizza : ℚ) : 
  initial_pizza = 1 →
  (let eat_fraction := 1/3
   let remaining_fraction := 1 - eat_fraction
   let total_eaten := (1 - remaining_fraction^n) / (1 - remaining_fraction)
   n = 6 →
   total_eaten = 665/729) := by
sorry

end NUMINAMATH_CALUDE_pizza_eating_l3513_351393


namespace NUMINAMATH_CALUDE_min_positive_period_sin_cos_squared_l3513_351363

theorem min_positive_period_sin_cos_squared (x : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ (Real.sin x + Real.cos x)^2 + 1
  ∃ T : ℝ, T > 0 ∧ (∀ t : ℝ, f (t + T) = f t) ∧
    (∀ S : ℝ, S > 0 ∧ (∀ t : ℝ, f (t + S) = f t) → T ≤ S) ∧
    T = π :=
by sorry

end NUMINAMATH_CALUDE_min_positive_period_sin_cos_squared_l3513_351363


namespace NUMINAMATH_CALUDE_geometric_series_sum_l3513_351380

theorem geometric_series_sum : ∀ (a r : ℚ), 
  a = 1 → r = 1/3 → abs r < 1 → 
  (∑' n, a * r^n) = 3/2 := by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l3513_351380


namespace NUMINAMATH_CALUDE_total_boys_in_camp_l3513_351385

theorem total_boys_in_camp (total : ℕ) 
  (h1 : (total : ℚ) * (1 / 5) = (total : ℚ) * (20 / 100))
  (h2 : (total : ℚ) * (1 / 5) * (3 / 10) = (total : ℚ) * (1 / 5) * (30 / 100))
  (h3 : (total : ℚ) * (1 / 5) * (7 / 10) = 77) :
  total = 550 := by
sorry

end NUMINAMATH_CALUDE_total_boys_in_camp_l3513_351385


namespace NUMINAMATH_CALUDE_total_word_count_180_to_220_l3513_351381

/-- Represents the word count for a number in the range [180, 220] -/
def word_count (n : ℕ) : ℕ :=
  if n = 180 then 3
  else if n ≥ 190 ∧ n ≤ 220 then 2
  else 3

/-- The sum of word counts for numbers in the range [a, b] -/
def sum_word_counts (a b : ℕ) : ℕ :=
  (Finset.range (b - a + 1)).sum (λ i => word_count (a + i))

theorem total_word_count_180_to_220 :
  sum_word_counts 180 220 = 99 := by
  sorry

end NUMINAMATH_CALUDE_total_word_count_180_to_220_l3513_351381
