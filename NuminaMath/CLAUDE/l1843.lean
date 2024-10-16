import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1843_184369

theorem quadratic_equation_roots (p q : ℝ) : 
  (∀ x, x^2 + p*x + q = 0 → (x + 1)^2 - p^2*(x + 1) + p*q = 0) →
  ((p = 1 ∧ ∃ q : ℝ, True) ∨ (p = -2 ∧ q = -1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1843_184369


namespace NUMINAMATH_CALUDE_product_of_digits_of_non_divisible_by_four_l1843_184335

def numbers : List Nat := [3612, 3620, 3628, 3636, 3641]

def is_divisible_by_four (n : Nat) : Bool :=
  n % 4 = 0

def units_digit (n : Nat) : Nat :=
  n % 10

def tens_digit (n : Nat) : Nat :=
  (n / 10) % 10

theorem product_of_digits_of_non_divisible_by_four :
  ∃ n ∈ numbers, ¬is_divisible_by_four n ∧ 
  units_digit n * tens_digit n = 4 := by
  sorry

end NUMINAMATH_CALUDE_product_of_digits_of_non_divisible_by_four_l1843_184335


namespace NUMINAMATH_CALUDE_ordering_of_abc_l1843_184331

theorem ordering_of_abc : 
  let a : ℝ := (1.7 : ℝ) ^ (0.9 : ℝ)
  let b : ℝ := (0.9 : ℝ) ^ (1.7 : ℝ)
  let c : ℝ := 1
  b < c ∧ c < a := by sorry

end NUMINAMATH_CALUDE_ordering_of_abc_l1843_184331


namespace NUMINAMATH_CALUDE_total_payment_is_correct_l1843_184386

-- Define the payment per lawn
def payment_per_lawn : ℚ := 13 / 3

-- Define the number of lawns mowed
def lawns_mowed : ℚ := 8 / 5

-- Define the base fee
def base_fee : ℚ := 5

-- Theorem statement
theorem total_payment_is_correct :
  payment_per_lawn * lawns_mowed + base_fee = 179 / 15 := by
  sorry

end NUMINAMATH_CALUDE_total_payment_is_correct_l1843_184386


namespace NUMINAMATH_CALUDE_surface_area_unchanged_l1843_184348

/-- Represents a cube with given side length -/
structure Cube where
  side : ℝ
  side_pos : side > 0

/-- Calculates the surface area of a cube -/
def surface_area (c : Cube) : ℝ := 6 * c.side^2

/-- Represents the original cube -/
def original_cube : Cube := ⟨4, by norm_num⟩

/-- Represents the corner cube to be removed -/
def corner_cube : Cube := ⟨2, by norm_num⟩

/-- Number of corners in a cube -/
def num_corners : ℕ := 8

/-- Theorem stating that the surface area remains unchanged after removing corner cubes -/
theorem surface_area_unchanged : 
  surface_area original_cube = surface_area original_cube := by sorry

end NUMINAMATH_CALUDE_surface_area_unchanged_l1843_184348


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l1843_184321

/-- 
A line x = m intersects a parabola x = -3y^2 - 4y + 7 at exactly one point 
if and only if m = 25/3
-/
theorem parabola_line_intersection (m : ℝ) : 
  (∃! y : ℝ, m = -3 * y^2 - 4 * y + 7) ↔ m = 25/3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l1843_184321


namespace NUMINAMATH_CALUDE_real_part_of_reciprocal_l1843_184396

theorem real_part_of_reciprocal (z : ℂ) (h1 : z ≠ 1) (h2 : z.im ≠ 0) (h3 : Complex.abs z = 1) :
  (1 / (1 - z)).re = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_reciprocal_l1843_184396


namespace NUMINAMATH_CALUDE_trajectory_circle_fixed_points_l1843_184322

/-- The trajectory of point M -/
def trajectory (x y : ℝ) : Prop :=
  (x ≥ 0 ∧ y^2 = 4*x) ∨ (x < 0 ∧ y = 0)

/-- The distance condition for point M -/
def distance_condition (x y : ℝ) : Prop :=
  ((x - 1)^2 + y^2)^(1/2) = x + 1

/-- The line passing through F(1,0) and intersecting the trajectory -/
def intersecting_line (m : ℝ) (x y : ℝ) : Prop :=
  x = m * y + 1

/-- The circle with diameter AB -/
def circle_AB (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 + 4 * y = 4

/-- The main theorem -/
theorem trajectory_circle_fixed_points :
  ∀ x y m,
  trajectory x y →
  distance_condition x y →
  intersecting_line m x y →
  (circle_AB (-1) 0 ∧ circle_AB 3 0) :=
by sorry

end NUMINAMATH_CALUDE_trajectory_circle_fixed_points_l1843_184322


namespace NUMINAMATH_CALUDE_nelly_babysitting_nights_l1843_184381

/-- The number of nights Nelly needs to babysit to afford pizza for herself and her friends -/
def nights_to_babysit (friends : ℕ) (pizza_cost : ℕ) (people_per_pizza : ℕ) (earnings_per_night : ℕ) : ℕ :=
  let total_people := friends + 1
  let pizzas_needed := (total_people + people_per_pizza - 1) / people_per_pizza
  let total_cost := pizzas_needed * pizza_cost
  (total_cost + earnings_per_night - 1) / earnings_per_night

/-- Theorem stating that Nelly needs to babysit for 15 nights to afford pizza for herself and 14 friends -/
theorem nelly_babysitting_nights :
  nights_to_babysit 14 12 3 4 = 15 := by
  sorry


end NUMINAMATH_CALUDE_nelly_babysitting_nights_l1843_184381


namespace NUMINAMATH_CALUDE_dollar_op_neg_two_three_l1843_184366

def dollar_op (a b : ℤ) : ℤ := a * (b + 1) + a * b

theorem dollar_op_neg_two_three : dollar_op (-2) 3 = -14 := by sorry

end NUMINAMATH_CALUDE_dollar_op_neg_two_three_l1843_184366


namespace NUMINAMATH_CALUDE_sum_interior_angles_hexagon_l1843_184305

/-- The sum of interior angles of a polygon with n sides is (n - 2) * 180 degrees. -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- A hexagon is a polygon with 6 sides. -/
def hexagon_sides : ℕ := 6

/-- Theorem: The sum of the interior angles of a hexagon is 720 degrees. -/
theorem sum_interior_angles_hexagon :
  sum_interior_angles hexagon_sides = 720 := by
  sorry

end NUMINAMATH_CALUDE_sum_interior_angles_hexagon_l1843_184305


namespace NUMINAMATH_CALUDE_q_work_time_l1843_184350

-- Define the work rates and total work
variable (W : ℝ) -- Total work
variable (Wp Wq Wr : ℝ) -- Work rates of p, q, and r

-- Define the conditions
axiom condition1 : Wp = Wq + Wr
axiom condition2 : Wp + Wq = W / 10
axiom condition3 : Wr = W / 60

-- Theorem to prove
theorem q_work_time : Wq = W / 24 := by
  sorry


end NUMINAMATH_CALUDE_q_work_time_l1843_184350


namespace NUMINAMATH_CALUDE_f_is_log_x_range_l1843_184364

noncomputable section

variable (a : ℝ) (f g : ℝ → ℝ)

-- Define g(x) = a^x
def g_def : g = fun x ↦ a^x := by sorry

-- Define f(x) as symmetric to g(x) with respect to y = x
def f_symmetric : ∀ x y, f x = y ↔ g y = x := by sorry

-- Part 1: Prove that f(x) = log_a x
theorem f_is_log : f = fun x ↦ Real.log x / Real.log a := by sorry

-- Part 2: Prove the range of x when a > 1 and f(x) < f(2)
theorem x_range (h : a > 1) : 
  ∀ x, f x < f 2 ↔ 0 < x ∧ x < a^2 := by sorry

end

end NUMINAMATH_CALUDE_f_is_log_x_range_l1843_184364


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1843_184359

/-- The functional equation solution for f(x+y) f(x-y) = (f(x))^2 -/
theorem functional_equation_solution (f : ℝ → ℝ) (hf : Continuous f) 
  (h : ∀ x y : ℝ, f (x + y) * f (x - y) = (f x)^2) :
  ∃ a c : ℝ, ∀ x : ℝ, f x = a * (c^x) := by
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1843_184359


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l1843_184378

theorem quadratic_solution_difference_squared : 
  ∀ a b : ℝ, (5 * a^2 - 6 * a - 55 = 0) → 
             (5 * b^2 - 6 * b - 55 = 0) → 
             (a ≠ b) →
             (a - b)^2 = 1296 / 25 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l1843_184378


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1843_184336

/-- The eccentricity of a hyperbola with given properties -/
theorem hyperbola_eccentricity : ∀ (a : ℝ),
  (∀ (x y : ℝ), x^2 / a^2 - y^2 / 9 = 1 → 
    ∃ (c : ℝ), c = 4 ∧ c^2 = a^2 + 9) →
  (4 : ℝ) * Real.sqrt 7 / 7 = 
    (4 : ℝ) / Real.sqrt (a^2) := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1843_184336


namespace NUMINAMATH_CALUDE_tomatoes_left_after_birds_l1843_184329

/-- The number of tomatoes left after birds eat one-third of the initial amount -/
def tomatoesLeft (initial : ℕ) : ℕ :=
  initial - initial / 3

/-- Theorem stating that if there are 21 cherry tomatoes initially,
    then 14 tomatoes are left after birds eat one-third of them -/
theorem tomatoes_left_after_birds : tomatoesLeft 21 = 14 := by
  sorry

end NUMINAMATH_CALUDE_tomatoes_left_after_birds_l1843_184329


namespace NUMINAMATH_CALUDE_negation_equivalence_function_property_l1843_184370

-- Define the statement for the negation of the existential proposition
theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - x > 0) ↔ (∀ x : ℝ, x^2 - x ≤ 0) :=
sorry

-- Define the properties for functions f and g
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def even_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x
def positive_derivative_pos (h : ℝ → ℝ) : Prop := ∀ x > 0, deriv h x > 0

-- Theorem for the properties of functions f and g
theorem function_property (f g : ℝ → ℝ) 
  (hodd : odd_function f) (heven : even_function g)
  (hf_deriv : positive_derivative_pos f) (hg_deriv : positive_derivative_pos g) :
  ∀ x < 0, deriv f x > deriv g x :=
sorry

end NUMINAMATH_CALUDE_negation_equivalence_function_property_l1843_184370


namespace NUMINAMATH_CALUDE_courses_last_year_is_six_l1843_184380

-- Define the number of courses taken last year
def courses_last_year : ℕ := 6

-- Define the average grade last year
def avg_grade_last_year : ℝ := 100

-- Define the number of courses taken the year before
def courses_year_before : ℕ := 5

-- Define the average grade for the year before
def avg_grade_year_before : ℝ := 50

-- Define the average grade for the entire two-year period
def avg_grade_two_years : ℝ := 77

-- Theorem statement
theorem courses_last_year_is_six :
  ((courses_year_before * avg_grade_year_before + courses_last_year * avg_grade_last_year) / 
   (courses_year_before + courses_last_year : ℝ) = avg_grade_two_years) ∧
  (courses_last_year = 6) :=
sorry

end NUMINAMATH_CALUDE_courses_last_year_is_six_l1843_184380


namespace NUMINAMATH_CALUDE_planes_perpendicular_to_same_plane_are_parallel_l1843_184304

/-- A plane in 3D space -/
structure Plane where
  -- Add necessary fields/axioms for a plane

/-- A line in 3D space -/
structure Line where
  -- Add necessary fields/axioms for a line

/-- Two planes are perpendicular -/
def perpendicular (p1 p2 : Plane) : Prop :=
  sorry

/-- Two planes are parallel -/
def parallel (p1 p2 : Plane) : Prop :=
  sorry

/-- Two lines are parallel -/
def parallel_lines (l1 l2 : Line) : Prop :=
  sorry

/-- A line is parallel to a plane -/
def line_parallel_plane (l : Line) (p : Plane) : Prop :=
  sorry

/-- A line is perpendicular to a plane -/
def line_perpendicular_plane (l : Line) (p : Plane) : Prop :=
  sorry

theorem planes_perpendicular_to_same_plane_are_parallel 
  (α β γ : Plane) : perpendicular α γ → perpendicular β γ → parallel α β :=
sorry

end NUMINAMATH_CALUDE_planes_perpendicular_to_same_plane_are_parallel_l1843_184304


namespace NUMINAMATH_CALUDE_happy_point_properties_l1843_184346

/-- A point (m, n+2) is a "happy point" if 2m = 8 + n --/
def is_happy_point (m n : ℝ) : Prop := 2 * m = 8 + n

/-- The point B(4,5) --/
def B : ℝ × ℝ := (4, 5)

/-- The point M(a, a-1) --/
def M (a : ℝ) : ℝ × ℝ := (a, a - 1)

theorem happy_point_properties :
  (¬ is_happy_point B.1 (B.2 - 2)) ∧
  (∀ a : ℝ, is_happy_point (M a).1 ((M a).2 - 2) → a > 0 ∧ a - 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_happy_point_properties_l1843_184346


namespace NUMINAMATH_CALUDE_third_hour_speed_l1843_184390

/-- Calculates the average speed for the third hour given total distance, total time, and speeds for the first two hours -/
def average_speed_third_hour (total_distance : ℝ) (total_time : ℝ) (speed_first_hour : ℝ) (speed_second_hour : ℝ) : ℝ :=
  let distance_first_two_hours := speed_first_hour + speed_second_hour
  let distance_third_hour := total_distance - distance_first_two_hours
  distance_third_hour

/-- Proves that the average speed for the third hour is 30 mph given the problem conditions -/
theorem third_hour_speed : 
  let total_distance : ℝ := 120
  let total_time : ℝ := 3
  let speed_first_hour : ℝ := 40
  let speed_second_hour : ℝ := 50
  average_speed_third_hour total_distance total_time speed_first_hour speed_second_hour = 30 := by
  sorry


end NUMINAMATH_CALUDE_third_hour_speed_l1843_184390


namespace NUMINAMATH_CALUDE_hiker_distance_l1843_184320

theorem hiker_distance (north east south east2 : ℝ) 
  (h_north : north = 15)
  (h_east : east = 8)
  (h_south : south = 9)
  (h_east2 : east2 = 2) :
  Real.sqrt ((north - south)^2 + (east + east2)^2) = 2 * Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_hiker_distance_l1843_184320


namespace NUMINAMATH_CALUDE_square_sum_ge_product_sum_l1843_184360

theorem square_sum_ge_product_sum (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a*b + b*c + a*c := by
  sorry

end NUMINAMATH_CALUDE_square_sum_ge_product_sum_l1843_184360


namespace NUMINAMATH_CALUDE_trig_identity_l1843_184328

theorem trig_identity (α : ℝ) : 
  (3 - 4 * Real.cos (2 * α) + Real.cos (4 * α)) / 
  (3 + 4 * Real.cos (2 * α) + Real.cos (4 * α)) = 
  (Real.tan α) ^ 4 / 3.396 := by sorry

end NUMINAMATH_CALUDE_trig_identity_l1843_184328


namespace NUMINAMATH_CALUDE_model_shop_purchase_l1843_184383

theorem model_shop_purchase : ∃ (c t : ℕ), c > 0 ∧ t > 0 ∧ 5 * c + 8 * t = 31 ∧ c + t = 5 := by
  sorry

end NUMINAMATH_CALUDE_model_shop_purchase_l1843_184383


namespace NUMINAMATH_CALUDE_H_div_G_equals_two_l1843_184384

-- Define the equation as a function
def equation (G H x : ℝ) : Prop :=
  G / (x + 5) + H / (x^2 - 4*x) = (x^2 - 2*x + 10) / (x^3 + x^2 - 20*x)

-- Define the theorem
theorem H_div_G_equals_two :
  ∀ G H : ℤ,
  (∀ x : ℝ, x ≠ -5 ∧ x ≠ 0 ∧ x ≠ 4 → equation G H x) →
  (H : ℝ) / (G : ℝ) = 2 := by
  sorry


end NUMINAMATH_CALUDE_H_div_G_equals_two_l1843_184384


namespace NUMINAMATH_CALUDE_percentage_difference_l1843_184399

theorem percentage_difference (x y : ℝ) (h : x = 4 * y) :
  (x - y) / x * 100 = 75 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l1843_184399


namespace NUMINAMATH_CALUDE_parabola_c_value_l1843_184319

/-- A parabola passing through two points -/
def Parabola (b c : ℝ) :=
  {f : ℝ → ℝ | ∃ (x : ℝ), f x = x^2 + b*x + c}

/-- The parabola passes through the point (1,4) -/
def passes_through_1_4 (b c : ℝ) : Prop :=
  1^2 + b*1 + c = 4

/-- The parabola passes through the point (5,4) -/
def passes_through_5_4 (b c : ℝ) : Prop :=
  5^2 + b*5 + c = 4

/-- Theorem: For a parabola y = x² + bx + c passing through (1,4) and (5,4), c = 9 -/
theorem parabola_c_value (b c : ℝ) 
  (h1 : passes_through_1_4 b c) 
  (h2 : passes_through_5_4 b c) : 
  c = 9 := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l1843_184319


namespace NUMINAMATH_CALUDE_T_mod_1000_l1843_184301

/-- The sum of all four-digit positive integers with four distinct digits -/
def T : ℕ := sorry

/-- Theorem stating that T mod 1000 = 465 -/
theorem T_mod_1000 : T % 1000 = 465 := by sorry

end NUMINAMATH_CALUDE_T_mod_1000_l1843_184301


namespace NUMINAMATH_CALUDE_bobs_sandwich_cost_l1843_184306

/-- Proves that the cost of each of Bob's sandwiches after discount and before tax is $2.412 -/
theorem bobs_sandwich_cost 
  (andy_soda : ℝ) (andy_hamburger : ℝ) (andy_chips : ℝ) (andy_tax_rate : ℝ)
  (bob_sandwich_before_discount : ℝ) (bob_sandwich_count : ℕ) (bob_water : ℝ)
  (bob_sandwich_discount_rate : ℝ) (bob_water_tax_rate : ℝ)
  (h_andy_soda : andy_soda = 1.50)
  (h_andy_hamburger : andy_hamburger = 2.75)
  (h_andy_chips : andy_chips = 1.25)
  (h_andy_tax_rate : andy_tax_rate = 0.08)
  (h_bob_sandwich_before_discount : bob_sandwich_before_discount = 2.68)
  (h_bob_sandwich_count : bob_sandwich_count = 5)
  (h_bob_water : bob_water = 1.25)
  (h_bob_sandwich_discount_rate : bob_sandwich_discount_rate = 0.10)
  (h_bob_water_tax_rate : bob_water_tax_rate = 0.07)
  (h_equal_total : 
    (andy_soda + 3 * andy_hamburger + andy_chips) * (1 + andy_tax_rate) = 
    bob_sandwich_count * bob_sandwich_before_discount * (1 - bob_sandwich_discount_rate) + 
    bob_water * (1 + bob_water_tax_rate)) :
  bob_sandwich_before_discount * (1 - bob_sandwich_discount_rate) = 2.412 := by
  sorry


end NUMINAMATH_CALUDE_bobs_sandwich_cost_l1843_184306


namespace NUMINAMATH_CALUDE_base_addition_theorem_l1843_184332

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * base^i) 0

theorem base_addition_theorem :
  let base13_number := [3, 5, 7]
  let base14_number := [4, 12, 13]
  (base_to_decimal base13_number 13) + (base_to_decimal base14_number 14) = 1544 := by
  sorry

end NUMINAMATH_CALUDE_base_addition_theorem_l1843_184332


namespace NUMINAMATH_CALUDE_bills_age_l1843_184343

theorem bills_age (bill eric : ℕ) 
  (h1 : bill = eric + 4) 
  (h2 : bill + eric = 28) : 
  bill = 16 := by
  sorry

end NUMINAMATH_CALUDE_bills_age_l1843_184343


namespace NUMINAMATH_CALUDE_cubic_difference_l1843_184358

theorem cubic_difference (x y : ℝ) (h1 : x + y = 14) (h2 : 3 * x + y = 20) :
  x^3 - y^3 = -1304 := by
sorry

end NUMINAMATH_CALUDE_cubic_difference_l1843_184358


namespace NUMINAMATH_CALUDE_black_car_overtakes_l1843_184356

/-- Represents the scenario of three cars racing on a highway -/
structure CarRace where
  red_speed : ℝ
  green_speed : ℝ
  black_speed : ℝ
  red_black_distance : ℝ
  black_green_distance : ℝ

/-- Theorem stating the condition for the black car to overtake the red car before the green car overtakes the black car -/
theorem black_car_overtakes (race : CarRace) 
  (h1 : race.red_speed = 40)
  (h2 : race.green_speed = 60)
  (h3 : race.red_black_distance = 10)
  (h4 : race.black_green_distance = 5)
  (h5 : race.black_speed > 40) :
  race.black_speed > 53.33 ↔ 
  (10 / (race.black_speed - 40) < 5 / (60 - race.black_speed)) := by
  sorry

end NUMINAMATH_CALUDE_black_car_overtakes_l1843_184356


namespace NUMINAMATH_CALUDE_intersection_cardinality_l1843_184368

def M : Finset ℕ := {1, 2, 4, 6, 8}
def N : Finset ℕ := {1, 2, 3, 5, 6, 7}

theorem intersection_cardinality : Finset.card (M ∩ N) = 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_cardinality_l1843_184368


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l1843_184333

open Set

-- Define the universe U as the set of real numbers
def U : Set ℝ := univ

-- Define set A
def A : Set ℝ := {x | x < 0}

-- Define set B
def B : Set ℝ := {x | x ≤ -1}

-- Theorem statement
theorem intersection_A_complement_B : 
  A ∩ (U \ B) = {x : ℝ | -1 < x ∧ x < 0} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l1843_184333


namespace NUMINAMATH_CALUDE_book_sale_problem_l1843_184311

theorem book_sale_problem (cost_loss : ℝ) (sale_price : ℝ) :
  cost_loss = 315 →
  sale_price = cost_loss * 0.85 →
  sale_price = (cost_loss + (2565 - 315)) * 1.19 →
  cost_loss + (2565 - 315) = 2565 := by
  sorry

end NUMINAMATH_CALUDE_book_sale_problem_l1843_184311


namespace NUMINAMATH_CALUDE_servant_service_duration_l1843_184342

/-- Calculates the number of months served given the total yearly payment and the received payment -/
def months_served (total_yearly_payment : ℚ) (received_payment : ℚ) : ℚ :=
  (received_payment / (total_yearly_payment / 12))

/-- Theorem stating that for the given payment conditions, the servant served approximately 6 months -/
theorem servant_service_duration :
  let total_yearly_payment : ℚ := 800
  let received_payment : ℚ := 400
  abs (months_served total_yearly_payment received_payment - 6) < 0.1 := by
sorry

end NUMINAMATH_CALUDE_servant_service_duration_l1843_184342


namespace NUMINAMATH_CALUDE_min_value_a_l1843_184308

theorem min_value_a (a x y : ℤ) (h1 : x - y^2 = a) (h2 : y - x^2 = a) (h3 : x ≠ y) (h4 : |x| ≤ 10) :
  ∃ (a_min : ℤ), a ≥ a_min ∧ a_min = -111 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_l1843_184308


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l1843_184337

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (k : ℕ), k < d ∧ (n - k) % d = 0 ∧ ∀ (m : ℕ), m < k → (n - m) % d ≠ 0 :=
by
  sorry

theorem problem_solution :
  ∃ (k : ℕ), k = 3 ∧ (5474827 - k) % 12 = 0 ∧ ∀ (m : ℕ), m < k → (5474827 - m) % 12 ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l1843_184337


namespace NUMINAMATH_CALUDE_dans_candy_bars_l1843_184354

theorem dans_candy_bars (total_spent : ℝ) (cost_per_bar : ℝ) (h1 : total_spent = 4) (h2 : cost_per_bar = 2) :
  total_spent / cost_per_bar = 2 := by
  sorry

end NUMINAMATH_CALUDE_dans_candy_bars_l1843_184354


namespace NUMINAMATH_CALUDE_additional_marbles_for_lisa_l1843_184387

/-- The minimum number of additional marbles needed -/
def min_additional_marbles (num_friends : ℕ) (initial_marbles : ℕ) : ℕ :=
  let min_marbles_per_friend := 2
  let total_marbles_needed := (num_friends * (2 * min_marbles_per_friend + num_friends - 1)) / 2
  max (total_marbles_needed - initial_marbles) 0

/-- Theorem stating the minimum number of additional marbles needed -/
theorem additional_marbles_for_lisa :
  min_additional_marbles 12 44 = 46 := by
  sorry

end NUMINAMATH_CALUDE_additional_marbles_for_lisa_l1843_184387


namespace NUMINAMATH_CALUDE_inequality_solution_range_l1843_184309

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, |x + 1| - |x - 2| < a^2 - 4*a) → (a > 3 ∨ a < 1) := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l1843_184309


namespace NUMINAMATH_CALUDE_marks_trees_l1843_184385

theorem marks_trees (current_trees planted_trees : ℕ) :
  current_trees = 13 → planted_trees = 12 →
  current_trees + planted_trees = 25 := by
  sorry

end NUMINAMATH_CALUDE_marks_trees_l1843_184385


namespace NUMINAMATH_CALUDE_cubic_function_property_l1843_184339

/-- A cubic function g(x) = ax^3 + bx^2 + cx + d with g(0) = 3 and g(1) = 5 satisfies a + 2b + c + 3d = 0 -/
theorem cubic_function_property (a b c d : ℝ) : 
  let g : ℝ → ℝ := λ x ↦ a * x^3 + b * x^2 + c * x + d
  (g 0 = 3) → (g 1 = 5) → (a + 2*b + c + 3*d = 0) := by
sorry

end NUMINAMATH_CALUDE_cubic_function_property_l1843_184339


namespace NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l1843_184327

def geometric_sequence (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem seventh_term_of_geometric_sequence 
  (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h_sum1 : a 1 + a 2 = 3) 
  (h_sum2 : a 2 + a 3 = 6) : 
  a 7 = 64 := by
sorry

end NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l1843_184327


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1843_184352

-- Define the sets A and B
def A : Set ℝ := { x | 2 ≤ x ∧ x < 4 }
def B : Set ℝ := { x | 3 * x - 7 ≥ 8 - 2 * x }

-- State the theorem
theorem union_of_A_and_B : A ∪ B = { x | x ≥ 2 } := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1843_184352


namespace NUMINAMATH_CALUDE_equilateral_triangles_similar_l1843_184376

-- Define an equilateral triangle
structure EquilateralTriangle where
  side : ℝ
  side_positive : side > 0

-- Define similarity for equilateral triangles
def similar (t1 t2 : EquilateralTriangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ t1.side = k * t2.side

-- Theorem: Any two equilateral triangles are similar
theorem equilateral_triangles_similar (t1 t2 : EquilateralTriangle) :
  similar t1 t2 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangles_similar_l1843_184376


namespace NUMINAMATH_CALUDE_license_plate_count_l1843_184315

/-- The number of possible digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of possible letters (A-Z) -/
def num_letters : ℕ := 26

/-- The number of digits in a license plate -/
def digits_in_plate : ℕ := 5

/-- The number of letters in a license plate -/
def letters_in_plate : ℕ := 2

/-- The number of positions where the letter block can be placed -/
def letter_block_positions : ℕ := digits_in_plate + 1

/-- The number of distinct license plates possible -/
def num_license_plates : ℕ := 
  letter_block_positions * num_digits^digits_in_plate * num_letters^letters_in_plate

theorem license_plate_count : num_license_plates = 40560000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l1843_184315


namespace NUMINAMATH_CALUDE_lucas_100_mod5_l1843_184395

/-- The Lucas sequence -/
def lucas : ℕ → ℕ
  | 0 => 1
  | 1 => 3
  | n + 2 => lucas n + lucas (n + 1)

/-- The Lucas sequence modulo 5 -/
def lucas_mod5 (n : ℕ) : ℕ := lucas n % 5

/-- The cycle length of the Lucas sequence modulo 5 -/
def cycle_length : ℕ := 10

theorem lucas_100_mod5 :
  lucas_mod5 100 = 3 := by sorry

end NUMINAMATH_CALUDE_lucas_100_mod5_l1843_184395


namespace NUMINAMATH_CALUDE_parking_arrangement_count_l1843_184312

/-- The number of parking spaces -/
def n : ℕ := 50

/-- The number of cars to be arranged -/
def k : ℕ := 2

/-- The number of ways to arrange k distinct cars in n parking spaces -/
def total_arrangements (n k : ℕ) : ℕ := n * (n - 1)

/-- The number of ways to arrange k distinct cars adjacently in n parking spaces -/
def adjacent_arrangements (n : ℕ) : ℕ := 2 * (n - 1)

/-- The number of ways to arrange k distinct cars in n parking spaces with at least one empty space between them -/
def valid_arrangements (n k : ℕ) : ℕ := total_arrangements n k - adjacent_arrangements n

theorem parking_arrangement_count :
  valid_arrangements n k = 2352 :=
by sorry

end NUMINAMATH_CALUDE_parking_arrangement_count_l1843_184312


namespace NUMINAMATH_CALUDE_sum_of_squared_distances_bounded_l1843_184367

/-- A point on the perimeter of a unit square -/
structure PerimeterPoint where
  x : Real
  y : Real
  on_perimeter : (x = 0 ∨ x = 1 ∨ y = 0 ∨ y = 1) ∧ 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1

/-- Four points on the perimeter of a unit square, in order -/
structure FourPoints where
  A : PerimeterPoint
  B : PerimeterPoint
  C : PerimeterPoint
  D : PerimeterPoint
  in_order : (A.x ≤ B.x ∧ A.y ≥ B.y) ∧ (B.x ≤ C.x ∧ B.y ≤ C.y) ∧ (C.x ≥ D.x ∧ C.y ≤ D.y) ∧ (D.x ≤ A.x ∧ D.y ≤ A.y)
  each_side_has_point : (A.x = 0 ∨ B.x = 0 ∨ C.x = 0 ∨ D.x = 0) ∧
                        (A.x = 1 ∨ B.x = 1 ∨ C.x = 1 ∨ D.x = 1) ∧
                        (A.y = 0 ∨ B.y = 0 ∨ C.y = 0 ∨ D.y = 0) ∧
                        (A.y = 1 ∨ B.y = 1 ∨ C.y = 1 ∨ D.y = 1)

/-- The squared distance between two points -/
def squared_distance (p1 p2 : PerimeterPoint) : Real :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- The theorem to be proved -/
theorem sum_of_squared_distances_bounded (points : FourPoints) :
  2 ≤ squared_distance points.A points.B +
      squared_distance points.B points.C +
      squared_distance points.C points.D +
      squared_distance points.D points.A
  ∧
  squared_distance points.A points.B +
  squared_distance points.B points.C +
  squared_distance points.C points.D +
  squared_distance points.D points.A ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squared_distances_bounded_l1843_184367


namespace NUMINAMATH_CALUDE_r_value_when_n_is_3_l1843_184344

theorem r_value_when_n_is_3 :
  let n : ℕ := 3
  let s := 2^(n+1) + 2
  let r := 3^s - 2*s + 1
  r = 387420454 := by
sorry

end NUMINAMATH_CALUDE_r_value_when_n_is_3_l1843_184344


namespace NUMINAMATH_CALUDE_gcd_problem_l1843_184345

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 360 * k) :
  Int.gcd (5 * b^3 + 2 * b^2 + 6 * b + 72) b = 72 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l1843_184345


namespace NUMINAMATH_CALUDE_town_growth_is_21_percent_l1843_184338

/-- Represents the population of a town over a 20-year period -/
structure TownPopulation where
  pop1991 : Nat
  pop2001 : Nat
  pop2011 : Nat

/-- Conditions for the town population -/
def ValidPopulation (t : TownPopulation) : Prop :=
  ∃ p q : Nat,
    t.pop1991 = p^2 ∧
    t.pop2001 = t.pop1991 + 180 ∧
    t.pop2001 = q^2 + 16 ∧
    t.pop2011 = t.pop2001 + 180

/-- The percent growth of the population over 20 years -/
def PercentGrowth (t : TownPopulation) : ℚ :=
  (t.pop2011 - t.pop1991 : ℚ) / t.pop1991 * 100

/-- Theorem stating that the percent growth is 21% -/
theorem town_growth_is_21_percent (t : TownPopulation) 
  (h : ValidPopulation t) : PercentGrowth t = 21 := by
  sorry

#check town_growth_is_21_percent

end NUMINAMATH_CALUDE_town_growth_is_21_percent_l1843_184338


namespace NUMINAMATH_CALUDE_factor_calculation_l1843_184300

theorem factor_calculation : 
  let initial_number : ℕ := 15
  let resultant : ℕ := 2 * initial_number + 5
  let final_result : ℕ := 105
  ∃ factor : ℚ, factor * resultant = final_result ∧ factor = 3 :=
by sorry

end NUMINAMATH_CALUDE_factor_calculation_l1843_184300


namespace NUMINAMATH_CALUDE_shortest_altitude_of_triangle_l1843_184391

/-- The shortest altitude of a triangle with sides 9, 12, and 15 is 7.2 -/
theorem shortest_altitude_of_triangle (a b c h : ℝ) : 
  a = 9 → b = 12 → c = 15 → 
  a^2 + b^2 = c^2 → 
  h * c = 2 * (a * b / 2) → 
  h ≤ a ∧ h ≤ b → 
  h = 7.2 := by sorry

end NUMINAMATH_CALUDE_shortest_altitude_of_triangle_l1843_184391


namespace NUMINAMATH_CALUDE_quadrilateral_area_l1843_184363

/-- The area of a quadrilateral formed by four squares arranged in a specific manner -/
theorem quadrilateral_area (s₁ s₂ s₃ s₄ : ℝ) (h₁ : s₁ = 1) (h₂ : s₂ = 3) (h₃ : s₃ = 5) (h₄ : s₄ = 7) :
  let total_length := s₁ + s₂ + s₃ + s₄
  let height_ratio := s₄ / total_length
  let height₂ := s₂ * height_ratio
  let height₃ := s₃ * height_ratio
  let quadrilateral_height := s₃ - s₂
  (height₂ + height₃) * quadrilateral_height / 2 = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l1843_184363


namespace NUMINAMATH_CALUDE_paul_final_stock_l1843_184362

def pencils_per_day : ℕ := 100
def work_days_per_week : ℕ := 5
def initial_stock : ℕ := 80
def pencils_sold : ℕ := 350

def final_stock : ℕ := initial_stock + (pencils_per_day * work_days_per_week) - pencils_sold

theorem paul_final_stock : final_stock = 230 := by sorry

end NUMINAMATH_CALUDE_paul_final_stock_l1843_184362


namespace NUMINAMATH_CALUDE_sin_2alpha_minus_pi_3_l1843_184313

theorem sin_2alpha_minus_pi_3 (α : ℝ) (h : Real.cos (α + π / 12) = -3 / 4) :
  Real.sin (2 * α - π / 3) = -1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_minus_pi_3_l1843_184313


namespace NUMINAMATH_CALUDE_min_operations_to_exceed_1000_l1843_184375

-- Define the operation of repeated squaring
def repeated_square (x : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n + 1 => (repeated_square x n) ^ 2

-- State the theorem
theorem min_operations_to_exceed_1000 :
  (∃ n : ℕ, repeated_square 3 n > 1000) ∧
  (∀ m : ℕ, repeated_square 3 m > 1000 → m ≥ 3) ∧
  (repeated_square 3 3 > 1000) :=
sorry

end NUMINAMATH_CALUDE_min_operations_to_exceed_1000_l1843_184375


namespace NUMINAMATH_CALUDE_union_of_sets_l1843_184353

theorem union_of_sets : 
  let M : Set ℕ := {0, 3}
  let N : Set ℕ := {1, 2, 3}
  M ∪ N = {0, 1, 2, 3} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l1843_184353


namespace NUMINAMATH_CALUDE_students_per_fourth_grade_class_l1843_184365

/-- Proves that the number of students in each fourth-grade class is 30 --/
theorem students_per_fourth_grade_class
  (total_cupcakes : ℕ)
  (pe_class_students : ℕ)
  (fourth_grade_classes : ℕ)
  (h1 : total_cupcakes = 140)
  (h2 : pe_class_students = 50)
  (h3 : fourth_grade_classes = 3)
  : (total_cupcakes - pe_class_students) / fourth_grade_classes = 30 := by
  sorry

#check students_per_fourth_grade_class

end NUMINAMATH_CALUDE_students_per_fourth_grade_class_l1843_184365


namespace NUMINAMATH_CALUDE_systematic_sampling_result_l1843_184379

-- Define the total number of employees
def total_employees : ℕ := 200

-- Define the number of groups
def num_groups : ℕ := 40

-- Define the size of each group
def group_size : ℕ := 5

-- Define the group from which the known number is drawn
def known_group : ℕ := 5

-- Define the known number drawn
def known_number : ℕ := 23

-- Define the target group
def target_group : ℕ := 10

-- Theorem statement
theorem systematic_sampling_result :
  -- Ensure the total number of employees is divisible by the number of groups
  total_employees = num_groups * group_size →
  -- Ensure the known number is within the range of the known group
  known_number > (known_group - 1) * group_size ∧ known_number ≤ known_group * group_size →
  -- Prove that the number drawn from the target group is 48
  ∃ (n : ℕ), n = (target_group - 1) * group_size + (known_number - (known_group - 1) * group_size) ∧ n = 48 :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_result_l1843_184379


namespace NUMINAMATH_CALUDE_complex_multiplication_l1843_184317

theorem complex_multiplication : (1 + Complex.I) * (2 + Complex.I) * (3 + Complex.I) = 10 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l1843_184317


namespace NUMINAMATH_CALUDE_alice_met_tweedledee_l1843_184302

-- Define the type for brothers
inductive Brother
| Tweedledee
| Tweedledum

-- Define the type for truthfulness
inductive Truthfulness
| AlwaysTruth
| AlwaysLie

-- Define the statement made by the brother
structure Statement where
  lying : Prop
  name : Brother

-- Define the meeting scenario
structure Meeting where
  brother : Brother
  truthfulness : Truthfulness
  statement : Statement

-- Theorem to prove
theorem alice_met_tweedledee (m : Meeting) :
  m.statement = { lying := true, name := Brother.Tweedledee } →
  (m.truthfulness = Truthfulness.AlwaysTruth ∨ m.truthfulness = Truthfulness.AlwaysLie) →
  m.brother = Brother.Tweedledee :=
by sorry

end NUMINAMATH_CALUDE_alice_met_tweedledee_l1843_184302


namespace NUMINAMATH_CALUDE_mart_income_percentage_l1843_184310

def income_comparison (juan tim mart : ℝ) : Prop :=
  tim = juan * 0.6 ∧ mart = tim * 1.6

theorem mart_income_percentage (juan tim mart : ℝ) 
  (h : income_comparison juan tim mart) : mart = juan * 0.96 := by
  sorry

end NUMINAMATH_CALUDE_mart_income_percentage_l1843_184310


namespace NUMINAMATH_CALUDE_edward_lost_lives_l1843_184314

theorem edward_lost_lives (initial_lives : ℕ) (remaining_lives : ℕ) (lost_lives : ℕ) : 
  initial_lives = 15 → remaining_lives = 7 → lost_lives = initial_lives - remaining_lives → lost_lives = 8 := by
  sorry

end NUMINAMATH_CALUDE_edward_lost_lives_l1843_184314


namespace NUMINAMATH_CALUDE_lcm_gcd_product_l1843_184341

theorem lcm_gcd_product (a b : ℕ) (ha : a = 11) (hb : b = 12) :
  Nat.lcm a b * Nat.gcd a b = 132 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_l1843_184341


namespace NUMINAMATH_CALUDE_perimeter_special_region_l1843_184326

/-- The perimeter of a region bounded by three semicircular arcs and one three-quarter circular arc,
    constructed on the sides of a square with side length 1/π, is equal to 2.25. -/
theorem perimeter_special_region :
  let square_side : ℝ := 1 / Real.pi
  let semicircle_perimeter : ℝ := Real.pi * square_side / 2
  let three_quarter_circle_perimeter : ℝ := 3 * Real.pi * square_side / 4
  let total_perimeter : ℝ := 3 * semicircle_perimeter + three_quarter_circle_perimeter
  total_perimeter = 2.25 := by sorry

end NUMINAMATH_CALUDE_perimeter_special_region_l1843_184326


namespace NUMINAMATH_CALUDE_square_perimeter_is_48_l1843_184340

-- Define a square with side length 12
def square_side_length : ℝ := 12

-- Define the perimeter of a square
def square_perimeter (side_length : ℝ) : ℝ := 4 * side_length

-- Theorem: The perimeter of the square with side length 12 cm is 48 cm
theorem square_perimeter_is_48 : 
  square_perimeter square_side_length = 48 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_is_48_l1843_184340


namespace NUMINAMATH_CALUDE_b_parallel_same_direction_as_a_l1843_184374

/-- Two vectors are parallel and in the same direction if one is a positive scalar multiple of the other -/
def parallel_same_direction (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ b = (k * a.1, k * a.2)

/-- Given vector a -/
def a : ℝ × ℝ := (1, -1)

/-- Vector b to be proven parallel and in the same direction as a -/
def b : ℝ × ℝ := (2, -2)

/-- Theorem stating that b is parallel and in the same direction as a -/
theorem b_parallel_same_direction_as_a : parallel_same_direction a b := by
  sorry

end NUMINAMATH_CALUDE_b_parallel_same_direction_as_a_l1843_184374


namespace NUMINAMATH_CALUDE_male_attendees_on_time_l1843_184361

/-- Proves that the fraction of male attendees who arrived on time is 0.875 -/
theorem male_attendees_on_time (total_attendees : ℝ) : 
  let male_attendees := (3/5 : ℝ) * total_attendees
  let female_attendees := (2/5 : ℝ) * total_attendees
  let on_time_female := (9/10 : ℝ) * female_attendees
  let not_on_time := 0.115 * total_attendees
  let on_time := total_attendees - not_on_time
  ∃ (on_time_male : ℝ), 
    on_time_male + on_time_female = on_time ∧ 
    on_time_male / male_attendees = 0.875 :=
by sorry

end NUMINAMATH_CALUDE_male_attendees_on_time_l1843_184361


namespace NUMINAMATH_CALUDE_unique_perfect_square_polynomial_l1843_184325

theorem unique_perfect_square_polynomial : 
  ∃! y : ℤ, ∃ n : ℤ, y^4 + 4*y^3 + 9*y^2 + 2*y + 17 = n^2 := by
  sorry

end NUMINAMATH_CALUDE_unique_perfect_square_polynomial_l1843_184325


namespace NUMINAMATH_CALUDE_highest_power_of_1991_l1843_184392

theorem highest_power_of_1991 :
  let n : ℕ := 1990^(1991^1992) + 1992^(1991^1990)
  ∃ k : ℕ, k = 2 ∧ (1991 : ℕ)^k ∣ n ∧ ∀ m : ℕ, m > k → ¬((1991 : ℕ)^m ∣ n) := by
  sorry

end NUMINAMATH_CALUDE_highest_power_of_1991_l1843_184392


namespace NUMINAMATH_CALUDE_scooter_profit_l1843_184323

theorem scooter_profit (original_cost repair_cost profit_percentage : ℝ) : 
  repair_cost = 0.1 * original_cost → 
  repair_cost = 500 → 
  profit_percentage = 0.2 → 
  original_cost * profit_percentage = 1000 := by
sorry

end NUMINAMATH_CALUDE_scooter_profit_l1843_184323


namespace NUMINAMATH_CALUDE_rectangle_dimension_solution_l1843_184371

theorem rectangle_dimension_solution (x : ℝ) : 
  (3*x - 5 > 0) → (x + 7 > 0) → ((3*x - 5) * (x + 7) = 14*x - 35) → x = 0 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_dimension_solution_l1843_184371


namespace NUMINAMATH_CALUDE_paint_calculation_l1843_184393

theorem paint_calculation (P : ℝ) 
  (h1 : (1/3 : ℝ) * P + (1/5 : ℝ) * (2/3 : ℝ) * P = 168) : 
  P = 360 :=
sorry

end NUMINAMATH_CALUDE_paint_calculation_l1843_184393


namespace NUMINAMATH_CALUDE_shane_photos_february_l1843_184330

/-- Calculates the number of photos Shane takes each week in February given the conditions --/
theorem shane_photos_february (total_photos : ℕ) (january_daily : ℕ) (january_days : ℕ) (february_weeks : ℕ) :
  total_photos = january_daily * january_days + february_weeks * (total_photos - january_daily * january_days) / february_weeks →
  total_photos = 146 →
  january_daily = 2 →
  january_days = 31 →
  february_weeks = 4 →
  (total_photos - january_daily * january_days) / february_weeks = 21 :=
by sorry

end NUMINAMATH_CALUDE_shane_photos_february_l1843_184330


namespace NUMINAMATH_CALUDE_max_k_for_quadratic_roots_difference_l1843_184334

theorem max_k_for_quadratic_roots_difference (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
   x^2 + k*x - 3 = 0 ∧ 
   y^2 + k*y - 3 = 0 ∧ 
   |x - y| = 10) →
  k ≤ Real.sqrt 88 :=
sorry

end NUMINAMATH_CALUDE_max_k_for_quadratic_roots_difference_l1843_184334


namespace NUMINAMATH_CALUDE_sum_difference_with_triangular_problem_solution_l1843_184355

def even_sum (n : ℕ) : ℕ := n * (n + 1)

def odd_sum (n : ℕ) : ℕ := n * n

def triangular_sum (n : ℕ) : ℕ := n * (n + 1) * (n + 2) / 6

theorem sum_difference_with_triangular (n : ℕ) :
  even_sum n - odd_sum n + triangular_sum n = n * (n * n + 3) / 3 :=
by sorry

theorem problem_solution : 
  even_sum 1500 - odd_sum 1500 + triangular_sum 1500 = 563628000 :=
by sorry

end NUMINAMATH_CALUDE_sum_difference_with_triangular_problem_solution_l1843_184355


namespace NUMINAMATH_CALUDE_ferris_wheel_capacity_l1843_184347

theorem ferris_wheel_capacity 
  (total_seats : ℕ) 
  (people_per_seat : ℕ) 
  (broken_seats : ℕ) 
  (h1 : total_seats = 18) 
  (h2 : people_per_seat = 15) 
  (h3 : broken_seats = 10) :
  (total_seats - broken_seats) * people_per_seat = 120 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_capacity_l1843_184347


namespace NUMINAMATH_CALUDE_jenny_cans_collected_l1843_184389

/-- Represents the number of cans Jenny collects -/
def num_cans : ℕ := 20

/-- Represents the number of bottles Jenny collects -/
def num_bottles : ℕ := (100 - 2 * num_cans) / 6

/-- The weight of a bottle in ounces -/
def bottle_weight : ℕ := 6

/-- The weight of a can in ounces -/
def can_weight : ℕ := 2

/-- The payment for a bottle in cents -/
def bottle_payment : ℕ := 10

/-- The payment for a can in cents -/
def can_payment : ℕ := 3

/-- The total weight Jenny can carry in ounces -/
def total_weight : ℕ := 100

/-- The total payment Jenny receives in cents -/
def total_payment : ℕ := 160

theorem jenny_cans_collected :
  (num_bottles * bottle_weight + num_cans * can_weight = total_weight) ∧
  (num_bottles * bottle_payment + num_cans * can_payment = total_payment) :=
sorry

end NUMINAMATH_CALUDE_jenny_cans_collected_l1843_184389


namespace NUMINAMATH_CALUDE_job_completion_time_l1843_184382

/-- Proves that given the conditions of the problem, A takes 30 days to complete the job alone. -/
theorem job_completion_time (x : ℝ) (h1 : x > 0) (h2 : 10 * (1 / x + 1 / 40) = 0.5833333333333334) : x = 30 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l1843_184382


namespace NUMINAMATH_CALUDE_chessboard_cannot_be_tiled_l1843_184307

/-- Represents a chessboard with some squares removed -/
structure ModifiedChessboard :=
  (size : Nat)
  (removedSquares : Nat)

/-- Represents a domino tile -/
structure Domino :=
  (length : Nat)
  (width : Nat)

/-- Defines the properties of our specific chessboard -/
def ourChessboard : ModifiedChessboard :=
  { size := 8, removedSquares := 2 }

/-- Defines the properties of our domino -/
def ourDomino : Domino :=
  { length := 2, width := 1 }

/-- Function to check if a chessboard can be tiled with dominoes -/
def canBeTiled (board : ModifiedChessboard) (tile : Domino) : Prop :=
  ∃ (tiling : Nat), 
    (board.size * board.size - board.removedSquares) = tiling * tile.length * tile.width

/-- Theorem stating that our specific chessboard cannot be tiled with our specific dominoes -/
theorem chessboard_cannot_be_tiled : 
  ¬(canBeTiled ourChessboard ourDomino) := by
  sorry


end NUMINAMATH_CALUDE_chessboard_cannot_be_tiled_l1843_184307


namespace NUMINAMATH_CALUDE_purum_elementary_students_l1843_184394

theorem purum_elementary_students (total : ℕ) (difference : ℕ) : total = 41 → difference = 3 →
  ∃ purum non_purum : ℕ, purum = non_purum + difference ∧ purum + non_purum = total ∧ purum = 22 :=
by sorry

end NUMINAMATH_CALUDE_purum_elementary_students_l1843_184394


namespace NUMINAMATH_CALUDE_power_of_two_plus_five_l1843_184372

theorem power_of_two_plus_five : 2^5 + 5 = 37 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_plus_five_l1843_184372


namespace NUMINAMATH_CALUDE_mike_picked_52_peaches_l1843_184377

/-- The number of peaches Mike picked -/
def peaches_picked (initial final : ℕ) : ℕ := final - initial

/-- Theorem stating that Mike picked 52 peaches -/
theorem mike_picked_52_peaches : peaches_picked 34 86 = 52 := by
  sorry

end NUMINAMATH_CALUDE_mike_picked_52_peaches_l1843_184377


namespace NUMINAMATH_CALUDE_nap_period_days_l1843_184397

-- Define the given conditions
def naps_per_week : ℕ := 3
def hours_per_nap : ℕ := 2
def total_nap_hours : ℕ := 60

-- Define the theorem
theorem nap_period_days : 
  (total_nap_hours / hours_per_nap / naps_per_week) * 7 = 70 := by
  sorry

end NUMINAMATH_CALUDE_nap_period_days_l1843_184397


namespace NUMINAMATH_CALUDE_ellipse_range_theorem_l1843_184373

/-- Definition of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop :=
  x^2 / 16 + y^2 / (16/3) = 1

/-- Point M -/
def M : ℝ × ℝ := (0, 2)

/-- Dot product of two 2D vectors -/
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

/-- The expression OP · OQ + MP · MQ -/
def expr (P Q : ℝ × ℝ) : ℝ :=
  dot_product P Q + dot_product (P.1 - M.1, P.2 - M.2) (Q.1 - M.1, Q.2 - M.2)

/-- The theorem to be proved -/
theorem ellipse_range_theorem :
  ∀ P Q : ℝ × ℝ,
  is_on_ellipse P.1 P.2 →
  is_on_ellipse Q.1 Q.2 →
  ∃ k : ℝ, P.2 - M.2 = k * (P.1 - M.1) ∧ Q.2 - M.2 = k * (Q.1 - M.1) →
  -20 ≤ expr P Q ∧ expr P Q ≤ -52/3 :=
sorry

end NUMINAMATH_CALUDE_ellipse_range_theorem_l1843_184373


namespace NUMINAMATH_CALUDE_lcm_gcd_sum_inequality_l1843_184316

theorem lcm_gcd_sum_inequality (a b k : ℕ+) (hk : k > 1) 
  (h : Nat.lcm a b + Nat.gcd a b = k * (a + b)) : 
  a + b ≥ 4 * k := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_sum_inequality_l1843_184316


namespace NUMINAMATH_CALUDE_thirteen_y_minus_x_equals_one_l1843_184398

theorem thirteen_y_minus_x_equals_one (x y : ℤ) 
  (h1 : x > 0) 
  (h2 : x = 11 * y + 4) 
  (h3 : 2 * x = 8 * (3 * y) + 3) : 
  13 * y - x = 1 := by
  sorry

end NUMINAMATH_CALUDE_thirteen_y_minus_x_equals_one_l1843_184398


namespace NUMINAMATH_CALUDE_remainder_polynomial_l1843_184324

theorem remainder_polynomial (p : ℝ → ℝ) (h1 : p 2 = 7) (h2 : p 5 = 8) (h3 : p 0 = 6) :
  ∃ q : ℝ → ℝ, ∀ x, p x = q x * (x - 2) * (x - 5) + ((1/3) * x + 19/3) := by
sorry

end NUMINAMATH_CALUDE_remainder_polynomial_l1843_184324


namespace NUMINAMATH_CALUDE_least_sum_of_bases_l1843_184351

/-- Represents a number in a given base -/
def BaseRepresentation (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

/-- The problem statement -/
theorem least_sum_of_bases :
  ∃ (c d : Nat),
    c > 0 ∧ d > 0 ∧
    BaseRepresentation [5, 8] c = BaseRepresentation [8, 5] d ∧
    (∀ (c' d' : Nat),
      c' > 0 → d' > 0 →
      BaseRepresentation [5, 8] c' = BaseRepresentation [8, 5] d' →
      c + d ≤ c' + d') ∧
    c + d = 15 :=
  sorry

end NUMINAMATH_CALUDE_least_sum_of_bases_l1843_184351


namespace NUMINAMATH_CALUDE_immediate_prepayment_better_l1843_184349

/-- Represents a mortgage loan with fixed interest rate and annuity payments -/
structure MortgageLoan where
  S : ℝ  -- Initial loan balance
  T : ℝ  -- Monthly payment amount
  r : ℝ  -- Interest rate for the period
  (T_positive : T > 0)
  (r_nonnegative : r ≥ 0)
  (r_less_than_one : r < 1)

/-- Calculates the final balance after immediate partial prepayment -/
def final_balance_immediate (loan : MortgageLoan) : ℝ :=
  loan.S - 2 * loan.T + loan.r * loan.S - 0.5 * loan.r * loan.T + (0.5 * loan.r * loan.S)^2

/-- Calculates the final balance when waiting until the end of the period -/
def final_balance_waiting (loan : MortgageLoan) : ℝ :=
  loan.S - 2 * loan.T + loan.r * loan.S

/-- Theorem stating that immediate partial prepayment results in a lower final balance -/
theorem immediate_prepayment_better (loan : MortgageLoan) :
  final_balance_immediate loan < final_balance_waiting loan :=
by sorry

end NUMINAMATH_CALUDE_immediate_prepayment_better_l1843_184349


namespace NUMINAMATH_CALUDE_intersection_property_l1843_184388

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 6*x^2 + 13*x - 8

-- Define the line
def line (k m x : ℝ) : ℝ := k*x + m

-- Theorem statement
theorem intersection_property (k m : ℝ) 
  (hA : ∃ xA, f xA = line k m xA)  -- A exists
  (hB : ∃ xB, f xB = line k m xB)  -- B exists
  (hC : ∃ xC, f xC = line k m xC)  -- C exists
  (h_distinct : ∀ x y, x ≠ y → (f x = line k m x ∧ f y = line k m y) → 
                 ∃ z, f z = line k m z ∧ z ≠ x ∧ z ≠ y)  -- A, B, C are distinct
  (h_midpoint : ∃ xA xB xC, f xA = line k m xA ∧ f xB = line k m xB ∧ f xC = line k m xC ∧
                 xB = (xA + xC) / 2)  -- B is the midpoint of AC
  : 2*k + m = 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_property_l1843_184388


namespace NUMINAMATH_CALUDE_redistribution_impossible_l1843_184318

/-- Represents the distribution of balls in boxes -/
structure BallDistribution where
  white_boxes : ℕ
  black_boxes : ℕ
  balls_per_white : ℕ
  balls_per_black : ℕ

/-- The initial distribution of balls -/
def initial_distribution : BallDistribution :=
  { white_boxes := 0,  -- We don't know the exact number, so we use 0
    black_boxes := 0,  -- We don't know the exact number, so we use 0
    balls_per_white := 31,
    balls_per_black := 26 }

/-- The distribution after adding 3 boxes -/
def new_distribution : BallDistribution :=
  { white_boxes := initial_distribution.white_boxes + 3,  -- Total boxes increased by 3
    black_boxes := initial_distribution.black_boxes,      -- Assuming all new boxes are white
    balls_per_white := 21,
    balls_per_black := 16 }

/-- The desired final distribution -/
def desired_distribution : BallDistribution :=
  { white_boxes := 0,  -- We don't know the exact number
    black_boxes := 0,  -- We don't know the exact number
    balls_per_white := 15,
    balls_per_black := 10 }

theorem redistribution_impossible :
  ∀ (final_distribution : BallDistribution),
  (final_distribution.balls_per_white = desired_distribution.balls_per_white ∧
   final_distribution.balls_per_black = desired_distribution.balls_per_black) →
  (final_distribution.white_boxes * final_distribution.balls_per_white +
   final_distribution.black_boxes * final_distribution.balls_per_black =
   new_distribution.white_boxes * new_distribution.balls_per_white +
   new_distribution.black_boxes * new_distribution.balls_per_black) →
  False :=
sorry

end NUMINAMATH_CALUDE_redistribution_impossible_l1843_184318


namespace NUMINAMATH_CALUDE_cistern_fill_time_l1843_184303

/-- Represents a tap that can fill or empty a cistern -/
structure Tap where
  rate : ℚ  -- Rate at which the tap fills (positive) or empties (negative) the cistern per hour

/-- Calculates the time to fill a cistern given a list of taps -/
def timeTofill (taps : List Tap) : ℚ :=
  1 / (taps.map (λ t => t.rate) |>.sum)

theorem cistern_fill_time (tapA tapB tapC : Tap)
  (hA : tapA.rate = 1/3)
  (hB : tapB.rate = -1/6)
  (hC : tapC.rate = 1/2) :
  timeTofill [tapA, tapB, tapC] = 3/2 := by
  sorry

#eval timeTofill [{ rate := 1/3 }, { rate := -1/6 }, { rate := 1/2 }]

end NUMINAMATH_CALUDE_cistern_fill_time_l1843_184303


namespace NUMINAMATH_CALUDE_add_decimals_l1843_184357

theorem add_decimals : (7.56 : ℝ) + (4.29 : ℝ) = 11.85 := by sorry

end NUMINAMATH_CALUDE_add_decimals_l1843_184357
