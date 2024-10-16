import Mathlib

namespace NUMINAMATH_CALUDE_factorization_1_factorization_2_l3163_316371

variable (a b x y : ℝ)

/-- Factorization of 3ax^2 - 6ax + 3a --/
theorem factorization_1 : 3*a*x^2 - 6*a*x + 3*a = 3*a*(x-1)^2 := by sorry

/-- Factorization of 9x^2(a-b) + 4y^3(b-a) --/
theorem factorization_2 : 9*x^2*(a-b) + 4*y^3*(b-a) = (a-b)*(9*x^2 - 4*y^3) := by sorry

end NUMINAMATH_CALUDE_factorization_1_factorization_2_l3163_316371


namespace NUMINAMATH_CALUDE_xy_negative_implies_abs_sum_less_abs_diff_l3163_316334

theorem xy_negative_implies_abs_sum_less_abs_diff (x y : ℝ) 
  (h1 : x * y < 0) : 
  |x + y| < |x - y| := by sorry

end NUMINAMATH_CALUDE_xy_negative_implies_abs_sum_less_abs_diff_l3163_316334


namespace NUMINAMATH_CALUDE_largest_reciprocal_l3163_316356

theorem largest_reciprocal (a b c d e : ℚ) : 
  a = 1/4 ∧ b = 3/7 ∧ c = 2 ∧ d = 10 ∧ e = 2023 →
  (1/a > 1/b) ∧ (1/a > 1/c) ∧ (1/a > 1/d) ∧ (1/a > 1/e) :=
by sorry

end NUMINAMATH_CALUDE_largest_reciprocal_l3163_316356


namespace NUMINAMATH_CALUDE_cube_surface_area_l3163_316331

/-- Given a cube with volume 729 cubic centimeters, its surface area is 486 square centimeters. -/
theorem cube_surface_area (volume : ℝ) (side : ℝ) : 
  volume = 729 → 
  volume = side ^ 3 → 
  6 * side ^ 2 = 486 := by
sorry

end NUMINAMATH_CALUDE_cube_surface_area_l3163_316331


namespace NUMINAMATH_CALUDE_rental_fee_calculation_l3163_316347

/-- The rental fee for a truck, given the total cost, per-mile charge, and miles driven. -/
def rental_fee (total_cost per_mile_charge miles_driven : ℚ) : ℚ :=
  total_cost - per_mile_charge * miles_driven

/-- Theorem stating that the rental fee is $20.99 under the given conditions. -/
theorem rental_fee_calculation :
  rental_fee 95.74 0.25 299 = 20.99 := by
  sorry

end NUMINAMATH_CALUDE_rental_fee_calculation_l3163_316347


namespace NUMINAMATH_CALUDE_smallest_t_value_l3163_316340

theorem smallest_t_value (u v w t : ℤ) : 
  (u^3 + v^3 + w^3 = t^3) →
  (u^3 < v^3) →
  (v^3 < w^3) →
  (w^3 < t^3) →
  (u^3 < 0) →
  (v^3 < 0) →
  (w^3 < 0) →
  (t^3 < 0) →
  (∃ k : ℤ, u = k - 1 ∧ v = k ∧ w = k + 1 ∧ t = k + 2) →
  (∀ s : ℤ, s < 0 ∧ (∃ x y z : ℤ, x^3 + y^3 + z^3 = s^3 ∧ 
    x^3 < y^3 ∧ y^3 < z^3 ∧ z^3 < s^3 ∧ 
    x^3 < 0 ∧ y^3 < 0 ∧ z^3 < 0 ∧ s^3 < 0 ∧
    (∃ j : ℤ, x = j - 1 ∧ y = j ∧ z = j + 1 ∧ s = j + 2)) → 
    8 ≤ |s|) →
  8 = |t| :=
sorry

end NUMINAMATH_CALUDE_smallest_t_value_l3163_316340


namespace NUMINAMATH_CALUDE_recruitment_probability_one_pass_reinspection_probability_l3163_316396

/-- Probabilities of passing re-inspection for students A, B, and C -/
def p_reinspect_A : ℝ := 0.5
def p_reinspect_B : ℝ := 0.6
def p_reinspect_C : ℝ := 0.75

/-- Probabilities of passing cultural examination for students A, B, and C -/
def p_cultural_A : ℝ := 0.6
def p_cultural_B : ℝ := 0.5
def p_cultural_C : ℝ := 0.4

/-- All students pass political review -/
def p_political : ℝ := 1

/-- Assumption: Outcomes of the last three stages are independent -/
axiom independence : True

theorem recruitment_probability :
  p_reinspect_A * p_cultural_A * p_political = 0.3 :=
sorry

theorem one_pass_reinspection_probability :
  p_reinspect_A * (1 - p_reinspect_B) * (1 - p_reinspect_C) +
  (1 - p_reinspect_A) * p_reinspect_B * (1 - p_reinspect_C) +
  (1 - p_reinspect_A) * (1 - p_reinspect_B) * p_reinspect_C = 0.275 :=
sorry

end NUMINAMATH_CALUDE_recruitment_probability_one_pass_reinspection_probability_l3163_316396


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3163_316376

/-- An arithmetic sequence -/
def arithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence where a₂ = 2 and a₆ = 10, a₁₀ = 18 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arith : arithmeticSequence a) 
    (h_a2 : a 2 = 2) 
    (h_a6 : a 6 = 10) : 
  a 10 = 18 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3163_316376


namespace NUMINAMATH_CALUDE_sqrt_36_equals_6_l3163_316374

theorem sqrt_36_equals_6 : Real.sqrt 36 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_36_equals_6_l3163_316374


namespace NUMINAMATH_CALUDE_sara_quarters_count_l3163_316325

theorem sara_quarters_count (initial_quarters final_quarters dad_quarters : ℕ) : 
  initial_quarters = 21 → dad_quarters = 49 → final_quarters = initial_quarters + dad_quarters → 
  final_quarters = 70 := by
sorry

end NUMINAMATH_CALUDE_sara_quarters_count_l3163_316325


namespace NUMINAMATH_CALUDE_product_xy_equals_one_l3163_316333

theorem product_xy_equals_one (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : 1 / (1 + x + x^2) + 1 / (1 + y + y^2) + 1 / (1 + x + y) = 1) :
  x * y = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_xy_equals_one_l3163_316333


namespace NUMINAMATH_CALUDE_fraction_equality_l3163_316337

theorem fraction_equality (a b : ℝ) (h : a / b = 3 / 4) : (a - b) / (a + b) = -1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3163_316337


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l3163_316380

def point : ℝ × ℝ := (2, -3)

def is_in_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

theorem point_in_fourth_quadrant : is_in_fourth_quadrant point := by
  sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l3163_316380


namespace NUMINAMATH_CALUDE_min_overlap_social_media_l3163_316338

/-- The minimum percentage of adults using both Facebook and Instagram -/
theorem min_overlap_social_media (facebook_users instagram_users : ℝ) 
  (h1 : facebook_users = 85)
  (h2 : instagram_users = 75) :
  (facebook_users + instagram_users) - 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_min_overlap_social_media_l3163_316338


namespace NUMINAMATH_CALUDE_perpendicular_to_oblique_implies_perpendicular_to_projection_l3163_316335

/-- A plane in which we consider lines and their projections. -/
structure Plane where
  -- Add necessary fields here

/-- Represents a line in the plane. -/
structure Line (P : Plane) where
  -- Add necessary fields here

/-- Indicates that a line is oblique (not parallel or perpendicular to some reference). -/
def isOblique (P : Plane) (l : Line P) : Prop :=
  sorry

/-- The projection of a line onto the plane. -/
def projection (P : Plane) (l : Line P) : Line P :=
  sorry

/-- Indicates that two lines are perpendicular. -/
def isPerpendicular (P : Plane) (l1 l2 : Line P) : Prop :=
  sorry

/-- 
The main theorem: If a line is perpendicular to an oblique line in a plane,
then it is also perpendicular to the projection of the oblique line in this plane.
-/
theorem perpendicular_to_oblique_implies_perpendicular_to_projection
  (P : Plane) (l1 l2 : Line P) (h1 : isOblique P l1) (h2 : isPerpendicular P l1 l2) :
  isPerpendicular P (projection P l1) l2 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_to_oblique_implies_perpendicular_to_projection_l3163_316335


namespace NUMINAMATH_CALUDE_f_passes_through_2_8_f_neg_one_eq_neg_one_l3163_316388

/-- A power function passing through (2, 8) -/
def f (x : ℝ) : ℝ := x^3

/-- The function f passes through (2, 8) -/
theorem f_passes_through_2_8 : f 2 = 8 := by sorry

/-- The value of f(-1) is -1 -/
theorem f_neg_one_eq_neg_one : f (-1) = -1 := by sorry

end NUMINAMATH_CALUDE_f_passes_through_2_8_f_neg_one_eq_neg_one_l3163_316388


namespace NUMINAMATH_CALUDE_parallel_condition_neither_sufficient_nor_necessary_l3163_316363

-- Define the types for lines and planes
variable (Line Plane : Type*)

-- Define the subset relation
variable (subset : Line → Plane → Prop)

-- Define the parallel relation for lines
variable (parallel_lines : Line → Line → Prop)

-- Define the parallel relation between a line and a plane
variable (parallel_line_plane : Line → Plane → Prop)

-- State the theorem
theorem parallel_condition_neither_sufficient_nor_necessary
  (l m : Line) (α : Plane) (h : subset m α) :
  ¬(∀ l m α, subset m α → (parallel_lines l m → parallel_line_plane l α)) ∧
  ¬(∀ l m α, subset m α → (parallel_line_plane l α → parallel_lines l m)) :=
sorry

end NUMINAMATH_CALUDE_parallel_condition_neither_sufficient_nor_necessary_l3163_316363


namespace NUMINAMATH_CALUDE_problem_solution_l3163_316307

theorem problem_solution (a b m : ℚ) 
  (h1 : 2 * a = m) 
  (h2 : 5 * b = m) 
  (h3 : a + b = 2) : 
  m = 20 / 7 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3163_316307


namespace NUMINAMATH_CALUDE_intersection_points_form_hyperbola_l3163_316393

theorem intersection_points_form_hyperbola :
  ∀ (t x y : ℝ), 
    (2 * t * x - 3 * y - 4 * t = 0) → 
    (x - 3 * t * y + 4 = 0) → 
    (x^2 / 16 - y^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_form_hyperbola_l3163_316393


namespace NUMINAMATH_CALUDE_student_tape_cost_problem_l3163_316378

theorem student_tape_cost_problem :
  ∃ (n : ℕ) (x : ℕ) (price : ℕ),
    Even n ∧
    10 < n ∧ n < 20 ∧
    100 ≤ price ∧ price ≤ 120 ∧
    n * x = price ∧
    (n - 2) * (x + 1) = price ∧
    n = 14 := by
  sorry

end NUMINAMATH_CALUDE_student_tape_cost_problem_l3163_316378


namespace NUMINAMATH_CALUDE_first_statue_weight_l3163_316360

/-- Given the weights of a marble block and its carved statues, prove the weight of the first statue -/
theorem first_statue_weight
  (total_weight : ℝ)
  (second_statue : ℝ)
  (third_statue : ℝ)
  (fourth_statue : ℝ)
  (discarded : ℝ)
  (h1 : total_weight = 80)
  (h2 : second_statue = 18)
  (h3 : third_statue = 15)
  (h4 : fourth_statue = 15)
  (h5 : discarded = 22)
  : ∃ (first_statue : ℝ),
    first_statue + second_statue + third_statue + fourth_statue + discarded = total_weight ∧
    first_statue = 10 := by
  sorry

end NUMINAMATH_CALUDE_first_statue_weight_l3163_316360


namespace NUMINAMATH_CALUDE_g_composition_of_3_l3163_316365

def g (n : ℕ) : ℕ :=
  if n ≤ 5 then n^2 + 2*n + 1 else 2*n + 4

theorem g_composition_of_3 : g (g (g 3)) = 76 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_of_3_l3163_316365


namespace NUMINAMATH_CALUDE_problem_solution_l3163_316353

def f (n : ℕ) : ℚ := (n^2 - 5*n + 4) / (n - 4)

theorem problem_solution :
  (f 1 = 0) ∧
  (∀ n : ℕ, n ≠ 4 → (f n = 5 ↔ n = 6)) ∧
  (∀ n : ℕ, n ≠ 4 → f n ≠ 3) := by sorry

end NUMINAMATH_CALUDE_problem_solution_l3163_316353


namespace NUMINAMATH_CALUDE_brendan_taxes_l3163_316309

/-- Calculates the taxes paid by a waiter named Brendan based on his work schedule and income. -/
theorem brendan_taxes : 
  let hourly_wage : ℚ := 6
  let shifts_8hour : ℕ := 2
  let shifts_12hour : ℕ := 1
  let hourly_tips : ℚ := 12
  let tax_rate : ℚ := 1/5
  let reported_tips_fraction : ℚ := 1/3
  
  let total_hours : ℕ := shifts_8hour * 8 + shifts_12hour * 12
  let wage_income : ℚ := hourly_wage * total_hours
  let total_tips : ℚ := hourly_tips * total_hours
  let reported_tips : ℚ := total_tips * reported_tips_fraction
  let reported_income : ℚ := wage_income + reported_tips
  let taxes_paid : ℚ := reported_income * tax_rate

  taxes_paid = 56 := by sorry

end NUMINAMATH_CALUDE_brendan_taxes_l3163_316309


namespace NUMINAMATH_CALUDE_sophie_total_spend_l3163_316324

-- Define the quantities and prices
def cupcakes : ℕ := 5
def cupcake_price : ℚ := 2

def doughnuts : ℕ := 6
def doughnut_price : ℚ := 1

def apple_pie_slices : ℕ := 4
def apple_pie_price : ℚ := 2

def cookies : ℕ := 15
def cookie_price : ℚ := 0.6

-- Define the total cost function
def total_cost : ℚ :=
  cupcakes * cupcake_price +
  doughnuts * doughnut_price +
  apple_pie_slices * apple_pie_price +
  cookies * cookie_price

-- Theorem statement
theorem sophie_total_spend : total_cost = 33 := by
  sorry

end NUMINAMATH_CALUDE_sophie_total_spend_l3163_316324


namespace NUMINAMATH_CALUDE_oplus_comm_l3163_316343

def oplus (a b : ℕ+) : ℕ+ := a ^ b.val + b ^ a.val

theorem oplus_comm (a b : ℕ+) : oplus a b = oplus b a := by
  sorry

end NUMINAMATH_CALUDE_oplus_comm_l3163_316343


namespace NUMINAMATH_CALUDE_no_rational_roots_for_all_quadratics_l3163_316398

/-- The largest known prime number -/
def p : ℕ := 2^24036583 - 1

/-- Theorem stating that there are no positive integers c such that
    both p^2 - 4c and p^2 + 4c are perfect squares -/
theorem no_rational_roots_for_all_quadratics :
  ¬∃ c : ℕ+, ∃ a b : ℕ, (p^2 - 4*c.val = a^2) ∧ (p^2 + 4*c.val = b^2) :=
sorry

end NUMINAMATH_CALUDE_no_rational_roots_for_all_quadratics_l3163_316398


namespace NUMINAMATH_CALUDE_lcm_five_equals_lcm_three_l3163_316330

def is_subset_prime_factorization (a b : Nat) : Prop :=
  ∀ p : Nat, Prime p → (p^(a.factorization p) ∣ b)

theorem lcm_five_equals_lcm_three
  (a b c d e : Nat)
  (h : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0)
  (h_lcm : Nat.lcm a (Nat.lcm b (Nat.lcm c (Nat.lcm d e))) = Nat.lcm a (Nat.lcm b c)) :
  (is_subset_prime_factorization d a ∨ is_subset_prime_factorization d b ∨ is_subset_prime_factorization d c) ∧
  (is_subset_prime_factorization e a ∨ is_subset_prime_factorization e b ∨ is_subset_prime_factorization e c) :=
sorry

end NUMINAMATH_CALUDE_lcm_five_equals_lcm_three_l3163_316330


namespace NUMINAMATH_CALUDE_sum_xyz_l3163_316358

theorem sum_xyz (x y z : ℝ) 
  (eq1 : y + z = 20 - 4*x)
  (eq2 : x + z = 10 - 5*y)
  (eq3 : x + y = 15 - 2*z) :
  3*x + 3*y + 3*z = 22.5 := by
sorry

end NUMINAMATH_CALUDE_sum_xyz_l3163_316358


namespace NUMINAMATH_CALUDE_quadratic_circle_properties_l3163_316386

/-- A quadratic function that intersects both coordinate axes at three points -/
structure QuadraticFunction where
  b : ℝ
  intersects_axes : ∃ (x₁ x₂ y : ℝ), x₁ ≠ x₂ ∧ 
    x₁^2 + 2*x₁ + b = 0 ∧ 
    x₂^2 + 2*x₂ + b = 0 ∧ 
    b = y

/-- The circle passing through the three intersection points -/
def circle_equation (f : QuadraticFunction) (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - (f.b + 1)*y + f.b = 0

theorem quadratic_circle_properties (f : QuadraticFunction) :
  (f.b < 1 ∧ f.b ≠ 0) ∧
  (∀ x y, circle_equation f x y ↔ x^2 + y^2 + 2*x - (f.b + 1)*y + f.b = 0) ∧
  circle_equation f (-2) 1 ∧
  circle_equation f 0 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_circle_properties_l3163_316386


namespace NUMINAMATH_CALUDE_polynomial_roots_l3163_316346

def polynomial (x : ℝ) : ℝ := x^3 - 5*x^2 + 3*x + 9

theorem polynomial_roots : 
  (polynomial (-1) = 0) ∧ 
  (polynomial 3 = 0) ∧ 
  (∃ (f : ℝ → ℝ), ∀ x, polynomial x = (x + 1) * (x - 3)^2 * f x) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_roots_l3163_316346


namespace NUMINAMATH_CALUDE_line_passes_through_first_and_fourth_quadrants_l3163_316305

-- Define the line y = kx + b
def line (k b x : ℝ) : ℝ := k * x + b

-- Define the condition bk < 0
def condition (b k : ℝ) : Prop := b * k < 0

-- Define the first quadrant
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

-- Define the fourth quadrant
def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

-- Theorem statement
theorem line_passes_through_first_and_fourth_quadrants (b k : ℝ) :
  condition b k →
  (∃ x y : ℝ, y = line k b x ∧ first_quadrant x y) ∧
  (∃ x y : ℝ, y = line k b x ∧ fourth_quadrant x y) :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_first_and_fourth_quadrants_l3163_316305


namespace NUMINAMATH_CALUDE_new_consumption_per_soldier_l3163_316354

/-- Calculates the new daily consumption per soldier after additional soldiers join a fort, given the initial conditions and the number of new soldiers. -/
theorem new_consumption_per_soldier
  (initial_soldiers : ℕ)
  (initial_consumption : ℚ)
  (initial_duration : ℕ)
  (new_duration : ℕ)
  (new_soldiers : ℕ)
  (h_initial_soldiers : initial_soldiers = 1200)
  (h_initial_consumption : initial_consumption = 3)
  (h_initial_duration : initial_duration = 30)
  (h_new_duration : new_duration = 25)
  (h_new_soldiers : new_soldiers = 528) :
  let total_provisions := initial_soldiers * initial_consumption * initial_duration
  let total_soldiers := initial_soldiers + new_soldiers
  total_provisions / (total_soldiers * new_duration) = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_new_consumption_per_soldier_l3163_316354


namespace NUMINAMATH_CALUDE_correspondence_C_is_mapping_l3163_316301

def is_mapping (A B : Type) (f : A → B) : Prop :=
  ∀ x : A, ∃! y : B, f x = y

theorem correspondence_C_is_mapping :
  let A := Nat
  let B := { x : Int // x = -1 ∨ x = 0 ∨ x = 1 }
  let f : A → B := λ x => ⟨(-1)^x, by sorry⟩
  is_mapping A B f := by sorry

end NUMINAMATH_CALUDE_correspondence_C_is_mapping_l3163_316301


namespace NUMINAMATH_CALUDE_coin_flip_problem_l3163_316308

theorem coin_flip_problem (total_coins : ℕ) (two_ruble : ℕ) (five_ruble : ℕ) :
  total_coins = 14 →
  two_ruble > 0 →
  five_ruble > 0 →
  two_ruble + five_ruble = total_coins →
  ∃ (k : ℕ), k > 0 ∧ 3 * k = 2 * two_ruble + 5 * five_ruble →
  five_ruble = 4 ∨ five_ruble = 8 ∨ five_ruble = 12 :=
by sorry

end NUMINAMATH_CALUDE_coin_flip_problem_l3163_316308


namespace NUMINAMATH_CALUDE_double_burgers_count_l3163_316352

/-- Represents the purchase of hamburgers for the marching band. -/
structure HamburgerPurchase where
  total_cost : ℚ
  total_burgers : ℕ
  single_burger_price : ℚ
  double_burger_price : ℚ

/-- Calculates the number of double burgers purchased. -/
def number_of_double_burgers (purchase : HamburgerPurchase) : ℕ := 
  sorry

/-- Theorem stating that the number of double burgers purchased is 41. -/
theorem double_burgers_count (purchase : HamburgerPurchase) 
  (h1 : purchase.total_cost = 70.5)
  (h2 : purchase.total_burgers = 50)
  (h3 : purchase.single_burger_price = 1)
  (h4 : purchase.double_burger_price = 1.5) :
  number_of_double_burgers purchase = 41 := by
  sorry

end NUMINAMATH_CALUDE_double_burgers_count_l3163_316352


namespace NUMINAMATH_CALUDE_area_of_rectangle_with_squares_l3163_316399

/-- A rectangle divided into four identical squares with a given perimeter -/
structure RectangleWithSquares where
  side_length : ℝ
  perimeter : ℝ
  perimeter_eq : perimeter = 8 * side_length

/-- The area of a rectangle divided into four identical squares -/
def area (r : RectangleWithSquares) : ℝ :=
  4 * r.side_length^2

/-- Theorem: A rectangle divided into four identical squares with a perimeter of 160 has an area of 1600 -/
theorem area_of_rectangle_with_squares (r : RectangleWithSquares) (h : r.perimeter = 160) :
  area r = 1600 := by
  sorry

#check area_of_rectangle_with_squares

end NUMINAMATH_CALUDE_area_of_rectangle_with_squares_l3163_316399


namespace NUMINAMATH_CALUDE_u_plus_v_value_l3163_316390

theorem u_plus_v_value (u v : ℚ) 
  (eq1 : 3 * u + 7 * v = 17)
  (eq2 : 5 * u - 3 * v = 9) :
  u + v = 43 / 11 := by
sorry

end NUMINAMATH_CALUDE_u_plus_v_value_l3163_316390


namespace NUMINAMATH_CALUDE_alternating_arrangements_count_alternating_arrangements_proof_l3163_316311

/-- The number of ways to arrange 2 men and 2 women in a row,
    such that no two men or two women are adjacent. -/
def alternating_arrangements : ℕ := 8

/-- The number of men in the arrangement. -/
def num_men : ℕ := 2

/-- The number of women in the arrangement. -/
def num_women : ℕ := 2

/-- Theorem stating that the number of alternating arrangements
    of 2 men and 2 women is 8. -/
theorem alternating_arrangements_count :
  alternating_arrangements = 8 ∧
  num_men = 2 ∧
  num_women = 2 := by
  sorry

/-- Proof that the number of alternating arrangements is correct. -/
theorem alternating_arrangements_proof :
  alternating_arrangements = 2 * (Nat.factorial num_men) * (Nat.factorial num_women) := by
  sorry

end NUMINAMATH_CALUDE_alternating_arrangements_count_alternating_arrangements_proof_l3163_316311


namespace NUMINAMATH_CALUDE_monomial_exponents_l3163_316328

/-- Two monomials are like terms if they have the same variables with the same exponents -/
def are_like_terms (m1 m2 : ℕ → ℕ) : Prop :=
  ∀ i, m1 i = m2 i

theorem monomial_exponents (a b : ℕ) :
  are_like_terms (fun i => if i = 0 then a + 1 else if i = 1 then 3 else 0)
                 (fun i => if i = 0 then 2 else if i = 1 then b else 0) →
  a = 1 ∧ b = 3 := by
  sorry

end NUMINAMATH_CALUDE_monomial_exponents_l3163_316328


namespace NUMINAMATH_CALUDE_find_m_value_l3163_316387

theorem find_m_value : ∃ m : ℝ, 
  (∀ x : ℝ, (m - 1) * x^2 + 5 * x + m^2 - 3 * m + 2 = 0) ∧ 
  (m^2 - 3 * m + 2 = 0) ∧ 
  (m - 1 ≠ 0) ∧ 
  (m = 2) := by
sorry

end NUMINAMATH_CALUDE_find_m_value_l3163_316387


namespace NUMINAMATH_CALUDE_vector_parallel_condition_l3163_316368

/-- Given vectors a and b, if a + 3b is parallel to b, then the first component of a is 6. -/
theorem vector_parallel_condition (a b : ℝ × ℝ) (m : ℝ) :
  a = (m, 2) →
  b = (3, 1) →
  ∃ k : ℝ, a + 3 • b = k • b →
  m = 6 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_condition_l3163_316368


namespace NUMINAMATH_CALUDE_min_games_for_20_teams_l3163_316391

/-- Represents a football tournament --/
structure Tournament where
  num_teams : ℕ
  num_games : ℕ

/-- Checks if a tournament satisfies the condition that among any three teams, 
    two have played against each other --/
def satisfies_condition (t : Tournament) : Prop :=
  ∀ (a b c : Fin t.num_teams), a ≠ b ∧ b ≠ c ∧ a ≠ c → 
    (∃ (x y : Fin t.num_teams), x ≠ y ∧ ((x = a ∧ y = b) ∨ (x = b ∧ y = c) ∨ (x = a ∧ y = c)))

/-- The main theorem stating the minimum number of games required --/
theorem min_games_for_20_teams : 
  ∃ (t : Tournament), t.num_teams = 20 ∧ t.num_games = 90 ∧ 
    satisfies_condition t ∧ 
    (∀ (t' : Tournament), t'.num_teams = 20 ∧ satisfies_condition t' → t'.num_games ≥ 90) :=
sorry

end NUMINAMATH_CALUDE_min_games_for_20_teams_l3163_316391


namespace NUMINAMATH_CALUDE_function_value_2023_l3163_316384

/-- A function f is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The main theorem -/
theorem function_value_2023 (f : ℝ → ℝ) 
    (h_even : IsEven f)
    (h_not_zero : ∃ x, f x ≠ 0)
    (h_equation : ∀ x, x * f (x + 2) = (x + 2) * f x + 2) :
  f 2023 = -1 := by
sorry

end NUMINAMATH_CALUDE_function_value_2023_l3163_316384


namespace NUMINAMATH_CALUDE_expression_evaluation_l3163_316367

theorem expression_evaluation (x y : ℝ) (hx : x = 1) (hy : y = 2) :
  (2 * x - y) * (y + 2 * x) - (2 * y + x) * (2 * y - x) = -15 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3163_316367


namespace NUMINAMATH_CALUDE_sum_of_numbers_greater_than_point_four_l3163_316341

theorem sum_of_numbers_greater_than_point_four : 
  let numbers : List ℚ := [0.8, 1/2, 0.9]
  let sum_of_greater : ℚ := (numbers.filter (λ x => x > 0.4)).sum
  sum_of_greater = 2.2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_greater_than_point_four_l3163_316341


namespace NUMINAMATH_CALUDE_division_base4_correct_l3163_316379

/-- Converts a number from base 4 to base 10 --/
def base4ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- Converts a number from base 10 to base 4 --/
def base10ToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

/-- Performs division in base 4 --/
def divBase4 (a b : List Nat) : (List Nat × List Nat) :=
  let a10 := base4ToBase10 a
  let b10 := base4ToBase10 b
  let q := a10 / b10
  let r := a10 % b10
  (base10ToBase4 q, base10ToBase4 r)

theorem division_base4_correct (a b : List Nat) :
  a = [2, 3, 0, 2] ∧ b = [2, 1] →
  divBase4 a b = ([3, 1, 1], [0, 1]) := by
  sorry

end NUMINAMATH_CALUDE_division_base4_correct_l3163_316379


namespace NUMINAMATH_CALUDE_dog_age_difference_l3163_316383

theorem dog_age_difference (
  avg_age_1_5 : ℝ)
  (age_1 : ℝ)
  (age_2 : ℝ)
  (age_3 : ℝ)
  (age_4 : ℝ)
  (age_5 : ℝ)
  (h1 : avg_age_1_5 = 18)
  (h2 : age_1 = 10)
  (h3 : age_2 = age_1 - 2)
  (h4 : age_3 = age_2 + 4)
  (h5 : age_4 = age_3 / 2)
  (h6 : age_5 = age_4 + 20)
  (h7 : avg_age_1_5 = (age_1 + age_5) / 2) :
  age_3 - age_2 = 4 := by
sorry

end NUMINAMATH_CALUDE_dog_age_difference_l3163_316383


namespace NUMINAMATH_CALUDE_satisfying_numbers_are_741_234_975_468_l3163_316322

-- Define a structure for three-digit numbers
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  h_hundreds : hundreds ≥ 1 ∧ hundreds ≤ 9
  h_tens : tens ≥ 0 ∧ tens ≤ 9
  h_ones : ones ≥ 0 ∧ ones ≤ 9

-- Define the property of middle digit being the arithmetic mean
def isMiddleDigitMean (n : ThreeDigitNumber) : Prop :=
  2 * n.tens = n.hundreds + n.ones

-- Define divisibility by 13
def isDivisibleBy13 (n : ThreeDigitNumber) : Prop :=
  (100 * n.hundreds + 10 * n.tens + n.ones) % 13 = 0

-- Define the set of numbers satisfying both conditions
def satisfyingNumbers : Set ThreeDigitNumber :=
  {n | isMiddleDigitMean n ∧ isDivisibleBy13 n}

-- The theorem to prove
theorem satisfying_numbers_are_741_234_975_468 :
  satisfyingNumbers = {
    ⟨7, 4, 1, by norm_num, by norm_num, by norm_num⟩,
    ⟨2, 3, 4, by norm_num, by norm_num, by norm_num⟩,
    ⟨9, 7, 5, by norm_num, by norm_num, by norm_num⟩,
    ⟨4, 6, 8, by norm_num, by norm_num, by norm_num⟩
  } := by sorry


end NUMINAMATH_CALUDE_satisfying_numbers_are_741_234_975_468_l3163_316322


namespace NUMINAMATH_CALUDE_cost_price_per_meter_correct_cost_price_fabric_C_is_120_l3163_316373

/-- Calculates the cost price per meter of fabric given the selling price, number of meters, and profit per meter. -/
def costPricePerMeter (sellingPrice : ℚ) (meters : ℚ) (profitPerMeter : ℚ) : ℚ :=
  (sellingPrice - meters * profitPerMeter) / meters

/-- Represents the fabric types and their properties -/
structure FabricType where
  name : String
  sellingPrice : ℚ
  meters : ℚ
  profitPerMeter : ℚ

/-- Theorem stating that the cost price per meter calculation is correct for all fabric types -/
theorem cost_price_per_meter_correct (fabric : FabricType) :
  costPricePerMeter fabric.sellingPrice fabric.meters fabric.profitPerMeter =
  (fabric.sellingPrice - fabric.meters * fabric.profitPerMeter) / fabric.meters :=
by
  sorry

/-- The three fabric types given in the problem -/
def fabricA : FabricType := ⟨"A", 6000, 45, 12⟩
def fabricB : FabricType := ⟨"B", 10800, 60, 15⟩
def fabricC : FabricType := ⟨"C", 3900, 30, 10⟩

/-- Theorem stating that the cost price per meter for fabric C is 120 -/
theorem cost_price_fabric_C_is_120 :
  costPricePerMeter fabricC.sellingPrice fabricC.meters fabricC.profitPerMeter = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_cost_price_per_meter_correct_cost_price_fabric_C_is_120_l3163_316373


namespace NUMINAMATH_CALUDE_inequality_iff_solution_set_l3163_316366

def inequality (x : ℝ) : Prop :=
  (3 / (x + 2)) + (4 / (x + 6)) > 1

def solution_set (x : ℝ) : Prop :=
  x < -7 ∨ (-6 < x ∧ x < -2) ∨ x > 2

theorem inequality_iff_solution_set :
  ∀ x : ℝ, inequality x ↔ solution_set x :=
by sorry

end NUMINAMATH_CALUDE_inequality_iff_solution_set_l3163_316366


namespace NUMINAMATH_CALUDE_number_of_children_l3163_316381

theorem number_of_children (B C : ℕ) : 
  B = 2 * C →
  B = 4 * (C - 160) →
  C = 320 := by
sorry

end NUMINAMATH_CALUDE_number_of_children_l3163_316381


namespace NUMINAMATH_CALUDE_winning_strategy_works_l3163_316359

/-- Represents a player in the coin game -/
inductive Player : Type
| One : Player
| Two : Player

/-- The game state -/
structure GameState :=
  (coins : ℕ)
  (currentPlayer : Player)

/-- Valid moves for each player -/
def validMove (player : Player) (n : ℕ) : Prop :=
  match player with
  | Player.One => n % 2 = 1 ∧ 1 ≤ n ∧ n ≤ 99
  | Player.Two => n % 2 = 0 ∧ 2 ≤ n ∧ n ≤ 100

/-- The winning strategy function -/
def winningStrategy (state : GameState) : Option ℕ :=
  match state.currentPlayer with
  | Player.One => 
    if state.coins > 95 then some 95
    else if state.coins % 101 ≠ 0 then some (state.coins % 101)
    else none
  | Player.Two => none

/-- The main theorem -/
theorem winning_strategy_works : 
  ∀ (state : GameState), 
    state.coins = 2015 → 
    state.currentPlayer = Player.One → 
    ∃ (move : ℕ), 
      validMove Player.One move ∧ 
      move = 95 ∧
      ∀ (opponentMove : ℕ), 
        validMove Player.Two opponentMove → 
        ∃ (nextMove : ℕ), 
          validMove Player.One nextMove ∧ 
          state.coins - move - opponentMove - nextMove ≡ 0 [MOD 101] :=
sorry

#check winning_strategy_works

end NUMINAMATH_CALUDE_winning_strategy_works_l3163_316359


namespace NUMINAMATH_CALUDE_min_value_of_angle_sum_l3163_316350

theorem min_value_of_angle_sum (α β : Real) : 
  α > 0 → β > 0 → α + β = π / 2 → (4 / α + 1 / β ≥ 18 / π) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_angle_sum_l3163_316350


namespace NUMINAMATH_CALUDE_distinct_prime_factors_of_84_l3163_316361

theorem distinct_prime_factors_of_84 : ∃ (p q r : Nat), 
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ 
  p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
  84 = p * q * r := by
  sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_of_84_l3163_316361


namespace NUMINAMATH_CALUDE_binder_problem_l3163_316303

/-- Given that 18 binders can bind 900 books in 10 days, prove that 11 binders can bind 660 books in 12 days. -/
theorem binder_problem (binders_initial : ℕ) (books_initial : ℕ) (days_initial : ℕ)
  (binders_final : ℕ) (days_final : ℕ) 
  (h1 : binders_initial = 18) (h2 : books_initial = 900) (h3 : days_initial = 10)
  (h4 : binders_final = 11) (h5 : days_final = 12) :
  (books_initial * binders_final * days_final) / (binders_initial * days_initial) = 660 := by
sorry

end NUMINAMATH_CALUDE_binder_problem_l3163_316303


namespace NUMINAMATH_CALUDE_perpendicular_transitivity_perpendicular_parallel_planes_l3163_316392

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_parallel : Plane → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)

-- Theorem 1
theorem perpendicular_transitivity 
  (m n : Line) (α : Plane) 
  (h1 : parallel m n) (h2 : perpendicular m α) : 
  perpendicular n α :=
sorry

-- Theorem 2
theorem perpendicular_parallel_planes 
  (m n : Line) (α β : Plane)
  (h1 : plane_parallel α β) (h2 : parallel m n) (h3 : perpendicular m α) :
  perpendicular n β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_transitivity_perpendicular_parallel_planes_l3163_316392


namespace NUMINAMATH_CALUDE_vampire_population_growth_l3163_316306

/-- Represents the vampire population growth in Willowton over two nights -/
theorem vampire_population_growth 
  (initial_population : ℕ) 
  (initial_vampires : ℕ) 
  (first_night_converts : ℕ) 
  (subsequent_night_increase : ℕ) : 
  initial_population ≥ 300 → 
  initial_vampires = 3 → 
  first_night_converts = 7 → 
  subsequent_night_increase = 1 → 
  (initial_vampires * first_night_converts + initial_vampires) * 
    (first_night_converts + subsequent_night_increase) + 
    (initial_vampires * first_night_converts + initial_vampires) = 216 := by
  sorry

end NUMINAMATH_CALUDE_vampire_population_growth_l3163_316306


namespace NUMINAMATH_CALUDE_triangle_proof_l3163_316375

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the properties of the triangle
def scalene_triangle (t : Triangle) : Prop :=
  t.a ≠ t.b ∧ t.b ≠ t.c ∧ t.c ≠ t.a

-- Define the given conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a = 3 ∧ t.c = 4 ∧ t.C = 2 * t.A

-- Theorem statement
theorem triangle_proof (t : Triangle) 
  (h1 : scalene_triangle t) 
  (h2 : triangle_conditions t) : 
  Real.cos t.A = 2/3 ∧ 
  t.b = 7/3 ∧ 
  Real.cos (2 * t.A + Real.pi/6) = -(Real.sqrt 3 + 4 * Real.sqrt 5)/18 := by
  sorry

end NUMINAMATH_CALUDE_triangle_proof_l3163_316375


namespace NUMINAMATH_CALUDE_power_subtraction_division_l3163_316349

theorem power_subtraction_division (n : ℕ) : 1^567 - 3^8 / 3^5 = -26 := by
  sorry

end NUMINAMATH_CALUDE_power_subtraction_division_l3163_316349


namespace NUMINAMATH_CALUDE_betty_oranges_l3163_316345

/-- Given 3 boxes and 8 oranges per box, the total number of oranges is 24. -/
theorem betty_oranges (num_boxes : ℕ) (oranges_per_box : ℕ) 
  (h1 : num_boxes = 3) 
  (h2 : oranges_per_box = 8) : 
  num_boxes * oranges_per_box = 24 := by
  sorry

end NUMINAMATH_CALUDE_betty_oranges_l3163_316345


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l3163_316370

/-- Two vectors are parallel if their components are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  ∀ x : ℝ, parallel (6, 2) (x, 3) → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l3163_316370


namespace NUMINAMATH_CALUDE_motion_equation_l3163_316339

theorem motion_equation (g a V V₀ S t : ℝ) 
  (hV : V = (g + a) * t + V₀)
  (hS : S = (1/2) * (g + a) * t^2 + V₀ * t) :
  t = 2 * S / (V + V₀) := by
  sorry

end NUMINAMATH_CALUDE_motion_equation_l3163_316339


namespace NUMINAMATH_CALUDE_min_coins_for_change_l3163_316315

/-- Represents the available denominations in cents -/
def denominations : List ℕ := [200, 100, 25, 10, 5, 1]

/-- Calculates the minimum number of bills and coins needed for change -/
def minCoins (amount : ℕ) : ℕ :=
  sorry

/-- The change amount in cents -/
def changeAmount : ℕ := 456

/-- Theorem stating that the minimum number of bills and coins for $4.56 change is 6 -/
theorem min_coins_for_change : minCoins changeAmount = 6 := by
  sorry

end NUMINAMATH_CALUDE_min_coins_for_change_l3163_316315


namespace NUMINAMATH_CALUDE_five_digit_divisibility_l3163_316304

def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999

def remove_middle_digit (n : ℕ) : ℕ :=
  (n / 10000) * 1000 + (n / 100 % 10) * 10 + (n % 10)

def satisfies_condition (n : ℕ) : Prop :=
  is_five_digit n ∧ (n % (remove_middle_digit n) = 0)

theorem five_digit_divisibility :
  ∀ n : ℕ, satisfies_condition n ↔ ∃ N : ℕ, 10 ≤ N ∧ N ≤ 99 ∧ n = N * 1000 := by
  sorry

end NUMINAMATH_CALUDE_five_digit_divisibility_l3163_316304


namespace NUMINAMATH_CALUDE_parabola_smallest_a_l3163_316394

/-- Given a parabola with vertex (1/2, -5/4), equation y = ax^2 + bx + c,
    a > 0, and directrix y = -2, prove that the smallest possible value of a is 2/3 -/
theorem parabola_smallest_a (a b c : ℝ) :
  (∀ x y : ℝ, y = a * x^2 + b * x + c) →  -- Equation of parabola
  (a > 0) →                               -- a is positive
  (∀ x : ℝ, a * (x - 1/2)^2 - 5/4 = a * x^2 + b * x + c) →  -- Vertex form
  (∀ x : ℝ, -2 = a * x^2 + b * x + c - 3/4 * (1/a)) →       -- Directrix equation
  a = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_parabola_smallest_a_l3163_316394


namespace NUMINAMATH_CALUDE_expand_and_simplify_1_simplify_division_2_l3163_316364

-- Define variables
variable (a b : ℝ)

-- Theorem 1
theorem expand_and_simplify_1 : 3 * a * (5 * a - 2 * b) = 15 * a^2 - 6 * a * b := by
  sorry

-- Theorem 2
theorem simplify_division_2 : (12 * a^3 - 6 * a^2 + 3 * a) / (3 * a) = 4 * a^2 - 2 * a + 1 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_1_simplify_division_2_l3163_316364


namespace NUMINAMATH_CALUDE_typists_letters_time_relation_typists_letters_theorem_l3163_316357

/-- The number of letters a single typist can type in one minute -/
def typing_rate (typists : ℕ) (letters : ℕ) (minutes : ℕ) : ℚ :=
  (letters : ℚ) / (typists * minutes)

/-- The theorem stating the relationship between typists, letters, and time -/
theorem typists_letters_time_relation 
  (initial_typists : ℕ) (initial_letters : ℕ) (initial_minutes : ℕ)
  (final_typists : ℕ) (final_minutes : ℕ) :
  initial_typists > 0 → initial_minutes > 0 → final_typists > 0 → final_minutes > 0 →
  (typing_rate initial_typists initial_letters initial_minutes) * 
    (final_typists * final_minutes) = 
  (final_typists * final_minutes * initial_letters : ℚ) / (initial_typists * initial_minutes) :=
by sorry

/-- The main theorem to prove -/
theorem typists_letters_theorem :
  typing_rate 20 42 20 * (30 * 60) = 189 :=
by sorry

end NUMINAMATH_CALUDE_typists_letters_time_relation_typists_letters_theorem_l3163_316357


namespace NUMINAMATH_CALUDE_polynomial_expansion_theorem_l3163_316377

theorem polynomial_expansion_theorem (a b : ℝ) : 
  a > 0 → b > 0 → a * b = 1/2 → 
  (28 : ℝ) * a^6 * b^2 = (56 : ℝ) * a^5 * b^3 → 
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_expansion_theorem_l3163_316377


namespace NUMINAMATH_CALUDE_estimate_fish_population_l3163_316355

/-- Estimates the total number of fish in a pond using the mark-recapture method. -/
theorem estimate_fish_population (tagged_fish : ℕ) (second_sample : ℕ) (tagged_in_sample : ℕ) :
  tagged_fish = 100 →
  second_sample = 200 →
  tagged_in_sample = 10 →
  (tagged_fish * second_sample) / tagged_in_sample = 2000 :=
by
  sorry

#check estimate_fish_population

end NUMINAMATH_CALUDE_estimate_fish_population_l3163_316355


namespace NUMINAMATH_CALUDE_base4_division_theorem_l3163_316362

/-- Converts a number from base 4 to base 10 -/
def base4ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 4 -/
def base10ToBase4 (n : ℕ) : ℕ := sorry

theorem base4_division_theorem :
  let dividend := 2313
  let divisor := 13
  let quotient := 122
  base10ToBase4 (base4ToBase10 dividend / base4ToBase10 divisor) = quotient := by
  sorry

end NUMINAMATH_CALUDE_base4_division_theorem_l3163_316362


namespace NUMINAMATH_CALUDE_sam_has_46_balloons_l3163_316369

/-- Given the number of red balloons Fred and Dan have, and the total number of red balloons,
    calculate the number of red balloons Sam has. -/
def sams_balloons (fred_balloons dan_balloons total_balloons : ℕ) : ℕ :=
  total_balloons - (fred_balloons + dan_balloons)

/-- Theorem stating that given the specific numbers of balloons in the problem,
    Sam must have 46 red balloons. -/
theorem sam_has_46_balloons :
  sams_balloons 10 16 72 = 46 := by
  sorry

end NUMINAMATH_CALUDE_sam_has_46_balloons_l3163_316369


namespace NUMINAMATH_CALUDE_arithmetic_sequence_30th_term_l3163_316348

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The 30th term of the given arithmetic sequence is 351. -/
theorem arithmetic_sequence_30th_term
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_a1 : a 1 = 3)
  (h_a2 : a 2 = 15)
  (h_a3 : a 3 = 27) :
  a 30 = 351 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_30th_term_l3163_316348


namespace NUMINAMATH_CALUDE_ab_value_l3163_316317

theorem ab_value (a b : ℝ) (h1 : a - b = 10) (h2 : a^2 + b^2 = 150) : a * b = 25 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l3163_316317


namespace NUMINAMATH_CALUDE_meaningful_fraction_l3163_316310

theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 3)) ↔ x ≠ 3 := by sorry

end NUMINAMATH_CALUDE_meaningful_fraction_l3163_316310


namespace NUMINAMATH_CALUDE_mona_unique_players_l3163_316336

/-- The number of unique players Mona grouped with in a video game --/
def unique_players (groups : ℕ) (players_per_group : ℕ) (repeated_players : ℕ) : ℕ :=
  groups * players_per_group - repeated_players

/-- Theorem stating the number of unique players Mona grouped with --/
theorem mona_unique_players :
  let groups : ℕ := 9
  let players_per_group : ℕ := 4
  let repeated_players : ℕ := 3
  unique_players groups players_per_group repeated_players = 33 := by
  sorry

#eval unique_players 9 4 3

end NUMINAMATH_CALUDE_mona_unique_players_l3163_316336


namespace NUMINAMATH_CALUDE_circle_properties_l3163_316316

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 2

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := x + y - 2 = 0

/-- Point of tangency -/
def point_of_tangency : ℝ × ℝ := (2, 0)

theorem circle_properties :
  (∃ (x y : ℝ), circle_equation x y ∧ x = 0 ∧ y = 0) ∧  -- Passes through origin
  (∀ (x y : ℝ), circle_equation x y → line_equation x y → (x, y) = point_of_tangency) ∧  -- Tangent at (2, 0)
  circle_equation (point_of_tangency.1) (point_of_tangency.2) :=  -- Point (2, 0) is on the circle
by sorry

end NUMINAMATH_CALUDE_circle_properties_l3163_316316


namespace NUMINAMATH_CALUDE_maximum_marks_l3163_316319

theorem maximum_marks : ∃ M : ℕ, 
  (M ≥ 434) ∧ 
  (M < 435) ∧ 
  (⌈(0.45 : ℝ) * (M : ℝ)⌉ = 130 + 65) := by
  sorry

end NUMINAMATH_CALUDE_maximum_marks_l3163_316319


namespace NUMINAMATH_CALUDE_min_odd_integers_l3163_316382

theorem min_odd_integers (a b c d e f : ℤ) : 
  a + b = 28 → 
  a + b + c + d = 45 → 
  a + b + c + d + e + f = 60 → 
  ∃ (odds : Finset ℤ), odds ⊆ {a, b, c, d, e, f} ∧ 
    (∀ x ∈ odds, Odd x) ∧ 
    odds.card = 2 ∧
    (∀ (other_odds : Finset ℤ), other_odds ⊆ {a, b, c, d, e, f} ∧ 
      (∀ x ∈ other_odds, Odd x) → 
      other_odds.card ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_min_odd_integers_l3163_316382


namespace NUMINAMATH_CALUDE_triangle_base_measurement_l3163_316351

/-- Given a triangular shape with height 20 cm, if the total area of three similar such shapes is 1200 cm², then the base of each triangle is 40 cm. -/
theorem triangle_base_measurement (height : ℝ) (total_area : ℝ) : 
  height = 20 → total_area = 1200 → ∃ (base : ℝ), base = 40 ∧ 3 * (base * height / 2) = total_area := by
  sorry

end NUMINAMATH_CALUDE_triangle_base_measurement_l3163_316351


namespace NUMINAMATH_CALUDE_value_of_expression_l3163_316389

theorem value_of_expression (x y z : ℝ) 
  (h1 : 3 * x - 4 * y - 2 * z = 0)
  (h2 : x - 2 * y - 8 * z = 0)
  (h3 : z ≠ 0) :
  (x^2 + 3*x*y) / (y^2 + z^2) = 329/61 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l3163_316389


namespace NUMINAMATH_CALUDE_minimum_value_implies_a_l3163_316312

/-- The function f(x) = x^3 + 3ax^2 - 6ax + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 - 6*a*x + 2

/-- The derivative of f(x) with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 6*a*x - 6*a

theorem minimum_value_implies_a (a : ℝ) :
  ∃ x₀ : ℝ, x₀ > 1 ∧ x₀ < 3 ∧
  (∀ x : ℝ, f a x ≥ f a x₀) ∧
  (f_derivative a x₀ = 0) →
  a = -2 := by sorry

end NUMINAMATH_CALUDE_minimum_value_implies_a_l3163_316312


namespace NUMINAMATH_CALUDE_chessboard_color_swap_theorem_l3163_316314

/-- A color is represented by a natural number -/
def Color := ℕ

/-- A chessboard is represented by a function from coordinates to colors -/
def Chessboard (n : ℕ) := Fin (2*n) → Fin (2*n) → Color

/-- A rectangle on the chessboard is defined by its corner coordinates -/
structure Rectangle (n : ℕ) where
  i1 : Fin (2*n)
  j1 : Fin (2*n)
  i2 : Fin (2*n)
  j2 : Fin (2*n)

/-- Predicate to check if all corners of a rectangle have the same color -/
def same_color_corners (board : Chessboard n) (rect : Rectangle n) : Prop :=
  board rect.i1 rect.j1 = board rect.i1 rect.j2 ∧
  board rect.i1 rect.j1 = board rect.i2 rect.j1 ∧
  board rect.i1 rect.j1 = board rect.i2 rect.j2

/-- Main theorem: There exist two tiles in the same column such that swapping
    their colors creates a rectangle with all four corners of the same color -/
theorem chessboard_color_swap_theorem (n : ℕ) (board : Chessboard n) :
  ∃ (i1 i2 j : Fin (2*n)) (rect : Rectangle n),
    i1 ≠ i2 ∧
    (∀ (i : Fin (2*n)), board i j ≠ board i1 j → board i j = board i2 j) →
    same_color_corners board rect :=
  sorry

end NUMINAMATH_CALUDE_chessboard_color_swap_theorem_l3163_316314


namespace NUMINAMATH_CALUDE_cow_chicken_problem_l3163_316318

theorem cow_chicken_problem (c h : ℕ) : 
  4 * c + 2 * h = 2 * (c + h) + 16 → c = 8 := by
  sorry

end NUMINAMATH_CALUDE_cow_chicken_problem_l3163_316318


namespace NUMINAMATH_CALUDE_fraction_sum_to_decimal_l3163_316344

theorem fraction_sum_to_decimal : (9 : ℚ) / 10 + (8 : ℚ) / 100 = (98 : ℚ) / 100 := by sorry

end NUMINAMATH_CALUDE_fraction_sum_to_decimal_l3163_316344


namespace NUMINAMATH_CALUDE_sum_of_solutions_abs_equation_l3163_316327

theorem sum_of_solutions_abs_equation : 
  ∃ (x₁ x₂ : ℝ), 
    (|3 * x₁ - 5| = 8) ∧ 
    (|3 * x₂ - 5| = 8) ∧ 
    (x₁ + x₂ = 10 / 3) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_abs_equation_l3163_316327


namespace NUMINAMATH_CALUDE_right_triangle_adjacent_side_l3163_316321

theorem right_triangle_adjacent_side (h a o : ℝ) (h_positive : h > 0) (a_positive : a > 0) (o_positive : o > 0) 
  (hypotenuse : h = 8) (opposite : o = 5) (pythagorean : h^2 = a^2 + o^2) : a = Real.sqrt 39 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_adjacent_side_l3163_316321


namespace NUMINAMATH_CALUDE_simplify_expression_l3163_316332

theorem simplify_expression : 
  (Real.sqrt (Real.sqrt 81) - Real.sqrt (8 + 1/2))^2 = 35/2 - 3 * Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3163_316332


namespace NUMINAMATH_CALUDE_circle_probability_theorem_l3163_316323

theorem circle_probability_theorem (R : ℝ) (h : R = 4) :
  let outer_circle_area := π * R^2
  let inner_circle_radius := R - 3
  let inner_circle_area := π * inner_circle_radius^2
  (inner_circle_area / outer_circle_area) = 1/16 := by
sorry

end NUMINAMATH_CALUDE_circle_probability_theorem_l3163_316323


namespace NUMINAMATH_CALUDE_tylenol_interval_l3163_316302

/-- Represents the duration of Jeremy's Tylenol regimen in weeks -/
def duration : ℕ := 2

/-- Represents the total number of pills Jeremy takes -/
def total_pills : ℕ := 112

/-- Represents the amount of Tylenol in each pill in milligrams -/
def mg_per_pill : ℕ := 500

/-- Represents the amount of Tylenol Jeremy takes per dose in milligrams -/
def mg_per_dose : ℕ := 1000

/-- Theorem stating that the time interval between doses is 6 hours -/
theorem tylenol_interval : 
  (duration * 7 * 24) / ((total_pills * mg_per_pill) / mg_per_dose) = 6 := by
  sorry


end NUMINAMATH_CALUDE_tylenol_interval_l3163_316302


namespace NUMINAMATH_CALUDE_complex_equation_proof_l3163_316320

theorem complex_equation_proof (a b : ℝ) : 
  (a + b * Complex.I) / (2 - Complex.I) = (3 : ℂ) + Complex.I → a - b = 8 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_proof_l3163_316320


namespace NUMINAMATH_CALUDE_cone_spheres_radius_theorem_l3163_316329

/-- A right circular cone with four congruent spheres inside --/
structure ConeWithSpheres where
  base_radius : ℝ
  height : ℝ
  sphere_radius : ℝ
  is_right_circular : Bool
  spheres_count : Nat
  spheres_congruent : Bool
  spheres_tangent_to_each_other : Bool
  spheres_tangent_to_base : Bool
  spheres_tangent_to_side : Bool

/-- The theorem stating the relationship between cone dimensions and sphere radius --/
theorem cone_spheres_radius_theorem (c : ConeWithSpheres) :
  c.base_radius = 6 ∧
  c.height = 15 ∧
  c.is_right_circular = true ∧
  c.spheres_count = 4 ∧
  c.spheres_congruent = true ∧
  c.spheres_tangent_to_each_other = true ∧
  c.spheres_tangent_to_base = true ∧
  c.spheres_tangent_to_side = true →
  c.sphere_radius = 45 / 7 := by
sorry

end NUMINAMATH_CALUDE_cone_spheres_radius_theorem_l3163_316329


namespace NUMINAMATH_CALUDE_matrix_product_l3163_316372

def A : Matrix (Fin 3) (Fin 3) ℤ := ![![3, 1, 1], ![2, 1, 2], ![1, 2, 3]]
def B : Matrix (Fin 3) (Fin 3) ℤ := ![![1, 1, -1], ![2, -1, 1], ![1, 0, 1]]
def C : Matrix (Fin 3) (Fin 3) ℤ := ![![6, 2, -1], ![6, 1, 1], ![8, -1, 4]]

theorem matrix_product : A * B = C := by sorry

end NUMINAMATH_CALUDE_matrix_product_l3163_316372


namespace NUMINAMATH_CALUDE_birdseed_mix_problem_l3163_316395

/-- Proves that Brand A contains 60% sunflower given the conditions of the birdseed mix problem -/
theorem birdseed_mix_problem (brand_a_millet : ℝ) (brand_b_millet : ℝ) (brand_b_safflower : ℝ)
  (mix_millet : ℝ) (mix_brand_a : ℝ) :
  brand_a_millet = 0.4 →
  brand_b_millet = 0.65 →
  brand_b_safflower = 0.35 →
  mix_millet = 0.5 →
  mix_brand_a = 0.6 →
  ∃ (brand_a_sunflower : ℝ),
    brand_a_sunflower = 0.6 ∧
    brand_a_millet + brand_a_sunflower = 1 ∧
    mix_brand_a * brand_a_millet + (1 - mix_brand_a) * brand_b_millet = mix_millet :=
by sorry

end NUMINAMATH_CALUDE_birdseed_mix_problem_l3163_316395


namespace NUMINAMATH_CALUDE_stating_inscribed_triangle_area_bound_l3163_316326

/-- A parallelogram in a 2D plane. -/
structure Parallelogram where
  vertices : Fin 4 → ℝ × ℝ

/-- A triangle in a 2D plane. -/
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

/-- Checks if a point is inside or on the perimeter of a parallelogram. -/
def isInOrOnParallelogram (p : ℝ × ℝ) (pgram : Parallelogram) : Prop :=
  sorry

/-- Checks if a triangle is inscribed in a parallelogram. -/
def isInscribed (t : Triangle) (pgram : Parallelogram) : Prop :=
  ∀ i, isInOrOnParallelogram (t.vertices i) pgram

/-- Calculates the area of a parallelogram. -/
noncomputable def areaParallelogram (pgram : Parallelogram) : ℝ :=
  sorry

/-- Calculates the area of a triangle. -/
noncomputable def areaTriangle (t : Triangle) : ℝ :=
  sorry

/-- 
Theorem stating that the area of any triangle inscribed in a parallelogram
is less than or equal to half the area of the parallelogram.
-/
theorem inscribed_triangle_area_bound
  (pgram : Parallelogram) (t : Triangle) (h : isInscribed t pgram) :
  areaTriangle t ≤ (1/2) * areaParallelogram pgram :=
by sorry

end NUMINAMATH_CALUDE_stating_inscribed_triangle_area_bound_l3163_316326


namespace NUMINAMATH_CALUDE_simplify_radical_expression_l3163_316385

theorem simplify_radical_expression : 
  Real.sqrt 7 - Real.sqrt 28 + Real.sqrt 63 = 2 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_radical_expression_l3163_316385


namespace NUMINAMATH_CALUDE_evaluate_expression_l3163_316300

theorem evaluate_expression : 6 - 5 * (9 - 2^3) * 3 = -9 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3163_316300


namespace NUMINAMATH_CALUDE_fraction_product_l3163_316397

theorem fraction_product : (5/8 : ℚ) * (7/9 : ℚ) * (11/13 : ℚ) * (3/5 : ℚ) * (17/19 : ℚ) * (8/15 : ℚ) = 14280/1107000 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_l3163_316397


namespace NUMINAMATH_CALUDE_complex_magnitude_example_l3163_316342

theorem complex_magnitude_example : Complex.abs (12 - 5*Complex.I) = 13 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_example_l3163_316342


namespace NUMINAMATH_CALUDE_evaluate_expression_l3163_316313

theorem evaluate_expression : (8^6 / 8^4) * 3^10 = 3783136 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3163_316313
