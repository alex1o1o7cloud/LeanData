import Mathlib

namespace NUMINAMATH_CALUDE_lucas_running_speed_l1123_112310

theorem lucas_running_speed :
  let eugene_speed : ℚ := 5
  let brianna_speed : ℚ := (3 / 4) * eugene_speed
  let katie_speed : ℚ := (4 / 3) * brianna_speed
  let lucas_speed : ℚ := (5 / 6) * katie_speed
  lucas_speed = 25 / 6 := by sorry

end NUMINAMATH_CALUDE_lucas_running_speed_l1123_112310


namespace NUMINAMATH_CALUDE_intersection_A_B_complement_B_union_A_complementB_l1123_112362

-- Define the sets A and B
def A : Set ℝ := {x | -2 < x ∧ x < 2}
def B : Set ℝ := {x | x < -1 ∨ x > 4}

-- Define the complement of B
def complementB : Set ℝ := {x | ¬ (x ∈ B)}

-- Theorem statements
theorem intersection_A_B : A ∩ B = {x | -2 < x ∧ x < -1} := by sorry

theorem complement_B : complementB = {x | -1 ≤ x ∧ x ≤ 4} := by sorry

theorem union_A_complementB : A ∪ complementB = {x | -2 < x ∧ x ≤ 4} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_complement_B_union_A_complementB_l1123_112362


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l1123_112332

theorem unique_solution_quadratic (m : ℝ) : 
  (∃! x : ℝ, (x + 5) * (x + 2) = m + 3 * x) ↔ m = 6 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l1123_112332


namespace NUMINAMATH_CALUDE_victors_stickers_l1123_112373

theorem victors_stickers (flower_stickers : ℕ) (animal_stickers : ℕ) : 
  flower_stickers = 8 →
  animal_stickers = flower_stickers - 2 →
  flower_stickers + animal_stickers = 14 :=
by sorry

end NUMINAMATH_CALUDE_victors_stickers_l1123_112373


namespace NUMINAMATH_CALUDE_father_daughter_age_inconsistency_l1123_112354

theorem father_daughter_age_inconsistency :
  ¬ ∃ (x : ℕ), 
    (40 : ℝ) = 2 * 40 ∧ 
    (40 : ℝ) - x = 3 * (40 - x) :=
by sorry

end NUMINAMATH_CALUDE_father_daughter_age_inconsistency_l1123_112354


namespace NUMINAMATH_CALUDE_symmetric_point_xoy_plane_l1123_112316

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The xoy plane in 3D space -/
def xoy_plane : Set Point3D := {p : Point3D | p.z = 0}

/-- Symmetry with respect to the xoy plane -/
def symmetric_wrt_xoy (p q : Point3D) : Prop :=
  p.x = q.x ∧ p.y = q.y ∧ p.z = -q.z

theorem symmetric_point_xoy_plane :
  let M : Point3D := ⟨2, 5, 8⟩
  let N : Point3D := ⟨2, 5, -8⟩
  symmetric_wrt_xoy M N := by sorry

end NUMINAMATH_CALUDE_symmetric_point_xoy_plane_l1123_112316


namespace NUMINAMATH_CALUDE_jessica_expense_increase_l1123_112340

-- Define Jessica's monthly expenses last year
def last_year_rent : ℝ := 1000
def last_year_food : ℝ := 200
def last_year_car_insurance : ℝ := 100
def last_year_utilities : ℝ := 50
def last_year_healthcare : ℝ := 150

-- Define the increase rates
def rent_increase_rate : ℝ := 0.3
def food_increase_rate : ℝ := 0.5
def car_insurance_increase_rate : ℝ := 2
def utilities_increase_rate : ℝ := 0.2
def healthcare_increase_rate : ℝ := 1

-- Define the theorem
theorem jessica_expense_increase :
  let this_year_rent := last_year_rent * (1 + rent_increase_rate)
  let this_year_food := last_year_food * (1 + food_increase_rate)
  let this_year_car_insurance := last_year_car_insurance * (1 + car_insurance_increase_rate)
  let this_year_utilities := last_year_utilities * (1 + utilities_increase_rate)
  let this_year_healthcare := last_year_healthcare * (1 + healthcare_increase_rate)
  let last_year_total := last_year_rent + last_year_food + last_year_car_insurance + last_year_utilities + last_year_healthcare
  let this_year_total := this_year_rent + this_year_food + this_year_car_insurance + this_year_utilities + this_year_healthcare
  (this_year_total - last_year_total) * 12 = 9120 :=
by sorry


end NUMINAMATH_CALUDE_jessica_expense_increase_l1123_112340


namespace NUMINAMATH_CALUDE_sqrt_inequality_l1123_112324

theorem sqrt_inequality (a b c : ℝ) 
  (h1 : a > b) 
  (h2 : b > c) 
  (h3 : a + b + c = 0) : 
  Real.sqrt (b^2 - a*c) < Real.sqrt (3 * a^2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l1123_112324


namespace NUMINAMATH_CALUDE_triangle_area_from_medians_l1123_112398

theorem triangle_area_from_medians (a b : ℝ) (cos_angle : ℝ) (h1 : a = 3) (h2 : b = 2 * Real.sqrt 7) (h3 : cos_angle = -3/4) :
  let sin_angle := Real.sqrt (1 - cos_angle^2)
  let sub_triangle_area := 1/2 * (2/3 * a) * (1/3 * b) * sin_angle
  6 * sub_triangle_area = 7 :=
sorry

end NUMINAMATH_CALUDE_triangle_area_from_medians_l1123_112398


namespace NUMINAMATH_CALUDE_exponent_equation_solution_l1123_112304

theorem exponent_equation_solution :
  ∀ x : ℝ, (5 : ℝ) ^ (2 * x + 3) = 125 ^ (x + 1) → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_exponent_equation_solution_l1123_112304


namespace NUMINAMATH_CALUDE_triangle_isosceles_l1123_112377

/-- A triangle with side lengths satisfying a specific equation is isosceles. -/
theorem triangle_isosceles (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_triangle : (a - c)^2 + (a - c) * b = 0) : a = c := by
  sorry

end NUMINAMATH_CALUDE_triangle_isosceles_l1123_112377


namespace NUMINAMATH_CALUDE_ron_multiplication_mistake_l1123_112392

theorem ron_multiplication_mistake (a b : ℕ) : 
  10 ≤ a ∧ a < 100 →  -- a is a two-digit number
  b < 10 →            -- b is a single-digit number
  a * (b + 10) = 190 →
  a * b = 0 := by
sorry

end NUMINAMATH_CALUDE_ron_multiplication_mistake_l1123_112392


namespace NUMINAMATH_CALUDE_power_of_integer_for_3150_l1123_112338

theorem power_of_integer_for_3150 (a : ℕ) (h1 : a = 14) 
  (h2 : ∀ k < a, ∃ n : ℕ, 3150 * k = n ^ 2 → False) : 
  ∃ n : ℕ, 3150 * a = n ^ 2 :=
sorry

end NUMINAMATH_CALUDE_power_of_integer_for_3150_l1123_112338


namespace NUMINAMATH_CALUDE_dance_partners_exist_l1123_112368

-- Define the types for boys and girls
variable {Boy Girl : Type}

-- Define the dance relation
variable (danced : Boy → Girl → Prop)

-- Define the conditions
variable (h1 : ∀ b : Boy, ∃ g : Girl, ¬danced b g)
variable (h2 : ∀ g : Girl, ∃ b : Boy, danced b g)

-- State the theorem
theorem dance_partners_exist :
  ∃ (f f' : Boy) (g g' : Girl), f ≠ f' ∧ g ≠ g' ∧ danced f g ∧ danced f' g' :=
sorry

end NUMINAMATH_CALUDE_dance_partners_exist_l1123_112368


namespace NUMINAMATH_CALUDE_simplify_expression_l1123_112350

theorem simplify_expression (x : ℝ) : 3 * (4 * x^2)^4 = 768 * x^8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1123_112350


namespace NUMINAMATH_CALUDE_restaurant_gratuities_l1123_112326

/-- Calculates the gratuities charged by a restaurant --/
theorem restaurant_gratuities
  (total_bill : ℝ)
  (sales_tax_rate : ℝ)
  (striploin_cost : ℝ)
  (wine_cost : ℝ)
  (h_total_bill : total_bill = 140)
  (h_sales_tax_rate : sales_tax_rate = 0.1)
  (h_striploin_cost : striploin_cost = 80)
  (h_wine_cost : wine_cost = 10) :
  total_bill - (striploin_cost + wine_cost) * (1 + sales_tax_rate) = 41 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_gratuities_l1123_112326


namespace NUMINAMATH_CALUDE_coin_flip_difference_l1123_112322

theorem coin_flip_difference (total_flips : ℕ) (heads : ℕ) (h1 : total_flips = 211) (h2 : heads = 65) :
  total_flips - heads - heads = 81 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_difference_l1123_112322


namespace NUMINAMATH_CALUDE_save_fraction_is_one_seventh_l1123_112369

/-- Represents the worker's financial situation over a year --/
structure WorkerFinances where
  monthly_pay : ℝ
  save_fraction : ℝ
  months : ℕ := 12

/-- The conditions of the problem as described --/
def valid_finances (w : WorkerFinances) : Prop :=
  w.monthly_pay > 0 ∧
  w.save_fraction > 0 ∧
  w.save_fraction < 1 ∧
  w.months * w.save_fraction * w.monthly_pay = 2 * (1 - w.save_fraction) * w.monthly_pay

/-- The main theorem stating that the save fraction is 1/7 --/
theorem save_fraction_is_one_seventh (w : WorkerFinances) 
  (h : valid_finances w) : w.save_fraction = 1 / 7 := by
  sorry

#check save_fraction_is_one_seventh

end NUMINAMATH_CALUDE_save_fraction_is_one_seventh_l1123_112369


namespace NUMINAMATH_CALUDE_range_of_x_l1123_112394

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 - 2*x - 3 < 0
def q (x : ℝ) : Prop := 1/(x-2) < 0

-- State the theorem
theorem range_of_x (x : ℝ) : p x ∧ q x ↔ -1 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l1123_112394


namespace NUMINAMATH_CALUDE_fishing_problem_l1123_112371

/-- The number of fish Xiaohua caught -/
def xiaohua_fish : ℕ := 26

/-- The number of fish Xiaobai caught -/
def xiaobai_fish : ℕ := 4

/-- The condition when Xiaohua gives 2 fish to Xiaobai -/
def condition1 (x y : ℕ) : Prop :=
  y - 2 = 4 * (x + 2)

/-- The condition when Xiaohua gives 6 fish to Xiaobai -/
def condition2 (x y : ℕ) : Prop :=
  y - 6 = 2 * (x + 6)

theorem fishing_problem :
  condition1 xiaobai_fish xiaohua_fish ∧
  condition2 xiaobai_fish xiaohua_fish := by
  sorry


end NUMINAMATH_CALUDE_fishing_problem_l1123_112371


namespace NUMINAMATH_CALUDE_largest_solution_and_ratio_l1123_112323

theorem largest_solution_and_ratio (x a b c d : ℤ) : 
  (7 * x / 5 + 3 = 4 / x) →
  (x = (a + b * Real.sqrt c) / d) →
  (a = -15 ∧ b = 1 ∧ c = 785 ∧ d = 14) →
  (x = (-15 + Real.sqrt 785) / 14 ∧ a * c * d / b = -164850) := by
  sorry

end NUMINAMATH_CALUDE_largest_solution_and_ratio_l1123_112323


namespace NUMINAMATH_CALUDE_largest_angle_convex_pentagon_l1123_112301

theorem largest_angle_convex_pentagon (x : ℝ) : 
  (x + 2) + (2 * x + 3) + (3 * x + 4) + (4 * x + 5) + (5 * x + 6) = 540 →
  max (x + 2) (max (2 * x + 3) (max (3 * x + 4) (max (4 * x + 5) (5 * x + 6)))) = 538 / 3 :=
by sorry

end NUMINAMATH_CALUDE_largest_angle_convex_pentagon_l1123_112301


namespace NUMINAMATH_CALUDE_cube_root_54880000_l1123_112342

theorem cube_root_54880000 : 
  (Real.rpow 54880000 (1/3 : ℝ)) = 140 * Real.rpow 2 (1/3 : ℝ) := by sorry

end NUMINAMATH_CALUDE_cube_root_54880000_l1123_112342


namespace NUMINAMATH_CALUDE_factorization_problem1_l1123_112379

theorem factorization_problem1 (x y : ℚ) : x^2 * y - 4 * x * y = x * y * (x - 4) := by sorry

end NUMINAMATH_CALUDE_factorization_problem1_l1123_112379


namespace NUMINAMATH_CALUDE_max_sum_of_distances_l1123_112351

/-- Triangle ABC is a right triangle with ∠ABC = 90°, AC = 10, AB = 8, BC = 6 -/
structure RightTriangle where
  AC : ℝ
  AB : ℝ
  BC : ℝ
  right_angle : AC^2 = AB^2 + BC^2
  AC_eq : AC = 10
  AB_eq : AB = 8
  BC_eq : BC = 6

/-- Point on triangle A'B'C'' -/
structure PointOnTriangleABC' where
  x : ℝ
  y : ℝ
  on_triangle : 0 ≤ x ∧ 0 ≤ y ∧ x + y ≤ 1

/-- Sum of distances from a point to the sides of triangle ABC -/
def sum_of_distances (t : RightTriangle) (p : PointOnTriangleABC') : ℝ :=
  p.x * t.AB + p.y * t.BC + 1

/-- Maximum sum of distances theorem -/
theorem max_sum_of_distances (t : RightTriangle) :
  ∃ (max : ℝ), max = 7 ∧ ∀ (p : PointOnTriangleABC'), sum_of_distances t p ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_distances_l1123_112351


namespace NUMINAMATH_CALUDE_ball_bounce_theorem_l1123_112347

theorem ball_bounce_theorem (h : Real) (r : Real) (target : Real) :
  h = 700 ∧ r = 1/3 ∧ target = 2 →
  (∀ k : ℕ, h * r^k < target ↔ k ≥ 6) :=
by sorry

end NUMINAMATH_CALUDE_ball_bounce_theorem_l1123_112347


namespace NUMINAMATH_CALUDE_expression_value_l1123_112311

theorem expression_value : 
  (2 * 6) / (12 * 14) * (3 * 12 * 14) / (2 * 6 * 3) * 2 = 2 := by sorry

end NUMINAMATH_CALUDE_expression_value_l1123_112311


namespace NUMINAMATH_CALUDE_quadratic_solution_l1123_112327

theorem quadratic_solution (m : ℝ) : 
  (2^2 - m*2 + 8 = 0) → m = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l1123_112327


namespace NUMINAMATH_CALUDE_inequality_counterexample_l1123_112385

theorem inequality_counterexample (a b : ℝ) (h : a < b) :
  ∃ c : ℝ, ¬(a * c < b * c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_counterexample_l1123_112385


namespace NUMINAMATH_CALUDE_sqrt_x_plus_sqrt_y_equals_two_l1123_112399

theorem sqrt_x_plus_sqrt_y_equals_two (θ : ℝ) (x y : ℝ) 
  (h1 : x + y = 3 - Real.cos (4 * θ)) 
  (h2 : x - y = 4 * Real.sin (2 * θ)) : 
  Real.sqrt x + Real.sqrt y = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_sqrt_y_equals_two_l1123_112399


namespace NUMINAMATH_CALUDE_alien_martian_limb_difference_l1123_112302

/-- The number of arms an alien has -/
def alien_arms : ℕ := 3

/-- The number of legs an alien has -/
def alien_legs : ℕ := 8

/-- The number of arms a Martian has -/
def martian_arms : ℕ := 2 * alien_arms

/-- The number of legs a Martian has -/
def martian_legs : ℕ := alien_legs / 2

/-- The total number of limbs an alien has -/
def alien_limbs : ℕ := alien_arms + alien_legs

/-- The total number of limbs a Martian has -/
def martian_limbs : ℕ := martian_arms + martian_legs

/-- The number of aliens and Martians being compared -/
def group_size : ℕ := 5

theorem alien_martian_limb_difference :
  group_size * alien_limbs - group_size * martian_limbs = 5 := by
  sorry

end NUMINAMATH_CALUDE_alien_martian_limb_difference_l1123_112302


namespace NUMINAMATH_CALUDE_kyles_presents_l1123_112307

theorem kyles_presents (cost1 cost2 cost3 : ℝ) : 
  cost2 = cost1 + 7 →
  cost3 = cost1 - 11 →
  cost1 + cost2 + cost3 = 50 →
  cost1 = 18 := by
sorry

end NUMINAMATH_CALUDE_kyles_presents_l1123_112307


namespace NUMINAMATH_CALUDE_largest_number_game_l1123_112390

theorem largest_number_game (a b c d : ℤ) 
  (eq1 : (a + b + c) / 3 + d = 17)
  (eq2 : (a + b + d) / 3 + c = 21)
  (eq3 : (a + c + d) / 3 + b = 23)
  (eq4 : (b + c + d) / 3 + a = 29) :
  max a (max b (max c d)) = 21 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_game_l1123_112390


namespace NUMINAMATH_CALUDE_franks_filled_boxes_l1123_112366

/-- Given that Frank had 13 boxes initially and 5 boxes are left unfilled,
    prove that the number of boxes he filled with toys is 8. -/
theorem franks_filled_boxes (total : ℕ) (unfilled : ℕ) (filled : ℕ) : 
  total = 13 → unfilled = 5 → filled = total - unfilled → filled = 8 := by sorry

end NUMINAMATH_CALUDE_franks_filled_boxes_l1123_112366


namespace NUMINAMATH_CALUDE_product_inequality_l1123_112348

theorem product_inequality (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_one : x + y + z = 1) : 
  (1 - x) * (1 - y) * (1 - z) ≥ 8 * x * y * z := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l1123_112348


namespace NUMINAMATH_CALUDE_valid_pairs_count_l1123_112345

/-- Represents a person's age --/
structure Age :=
  (value : ℕ)

/-- Represents the current ages of Jane and Dick --/
structure CurrentAges :=
  (jane : Age)
  (dick : Age)

/-- Represents the ages of Jane and Dick after n years --/
structure FutureAges :=
  (jane : Age)
  (dick : Age)

/-- Checks if an age is a two-digit number --/
def is_two_digit (age : Age) : Prop :=
  10 ≤ age.value ∧ age.value ≤ 99

/-- Checks if two ages have interchanged digits --/
def has_interchanged_digits (age1 age2 : Age) : Prop :=
  ∃ (a b : ℕ), a ≠ b ∧ age1.value = 10 * a + b ∧ age2.value = 10 * b + a

/-- Calculates the number of valid (d, n) pairs --/
def count_valid_pairs (current : CurrentAges) : ℕ :=
  sorry

/-- The main theorem to be proved --/
theorem valid_pairs_count (current : CurrentAges) :
  current.jane.value = 30 ∧
  current.dick.value > current.jane.value →
  (∀ n : ℕ, n > 0 →
    let future : FutureAges := ⟨⟨current.jane.value + n⟩, ⟨current.dick.value + n⟩⟩
    is_two_digit future.jane ∧
    is_two_digit future.dick ∧
    has_interchanged_digits future.jane future.dick) →
  count_valid_pairs current = 26 :=
by sorry

end NUMINAMATH_CALUDE_valid_pairs_count_l1123_112345


namespace NUMINAMATH_CALUDE_min_value_expression_l1123_112303

theorem min_value_expression (x y : ℝ) (h : x ≥ 4) :
  x^2 + y^2 - 8*x + 6*y + 26 ≥ 1 := by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1123_112303


namespace NUMINAMATH_CALUDE_cafeteria_bill_calculation_l1123_112389

/-- Calculates the total cost for Mell and her friends at the cafeteria --/
def cafeteria_bill (coffee_price ice_cream_price cake_price : ℚ) 
  (discount_rate tax_rate : ℚ) : ℚ :=
  let mell_order := 2 * coffee_price + cake_price
  let friend_order := 2 * coffee_price + cake_price + ice_cream_price
  let total_before_discount := mell_order + 2 * friend_order
  let discounted_total := total_before_discount * (1 - discount_rate)
  let final_total := discounted_total * (1 + tax_rate)
  final_total

/-- Theorem stating that the total bill for Mell and her friends is $47.69 --/
theorem cafeteria_bill_calculation : 
  cafeteria_bill 4 3 7 (15/100) (10/100) = 47.69 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_bill_calculation_l1123_112389


namespace NUMINAMATH_CALUDE_pen_purchase_theorem_l1123_112382

def budget : ℕ := 31
def price1 : ℕ := 2
def price2 : ℕ := 3
def price3 : ℕ := 4

def max_pens (b p1 p2 p3 : ℕ) : ℕ :=
  (b - p2 - p3) / p1 + 2

def min_pens (b p1 p2 p3 : ℕ) : ℕ :=
  (b - p1 - p2) / p3 + 3

theorem pen_purchase_theorem :
  max_pens budget price1 price2 price3 = 14 ∧
  min_pens budget price1 price2 price3 = 9 :=
by sorry

end NUMINAMATH_CALUDE_pen_purchase_theorem_l1123_112382


namespace NUMINAMATH_CALUDE_function_b_increasing_on_negative_reals_l1123_112314

/-- The function f(x) = 1 - 1/x is increasing on the interval (-∞,0) -/
theorem function_b_increasing_on_negative_reals :
  ∀ x y : ℝ, x < y → x < 0 → y < 0 → (1 - 1/x) < (1 - 1/y) := by
sorry

end NUMINAMATH_CALUDE_function_b_increasing_on_negative_reals_l1123_112314


namespace NUMINAMATH_CALUDE_function_inequality_l1123_112361

-- Define a differentiable function f
variable (f : ℝ → ℝ)

-- Assume f is differentiable
variable (hf : Differentiable ℝ f)

-- Assume f'(x) < f(x) for all x in ℝ
variable (h : ∀ x : ℝ, deriv f x < f x)

-- Theorem statement
theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x : ℝ, deriv f x < f x) : 
  f 1 < Real.exp 1 * f 0 ∧ f 2014 < Real.exp 2014 * f 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l1123_112361


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l1123_112391

theorem complex_fraction_equality : 1 + 1 / (2 + 1 / (2 + 2)) = 13 / 9 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l1123_112391


namespace NUMINAMATH_CALUDE_max_value_trig_expression_l1123_112315

theorem max_value_trig_expression (a b c : ℝ) :
  (∃ (θ : ℝ), ∀ (φ : ℝ), a * Real.cos θ + b * Real.sin θ + c * Real.sin (2 * θ) ≥ 
                         a * Real.cos φ + b * Real.sin φ + c * Real.sin (2 * φ)) →
  (∃ (θ : ℝ), a * Real.cos θ + b * Real.sin θ + c * Real.sin (2 * θ) = Real.sqrt (2 * (a^2 + b^2 + c^2))) :=
by sorry

end NUMINAMATH_CALUDE_max_value_trig_expression_l1123_112315


namespace NUMINAMATH_CALUDE_intersection_M_N_l1123_112349

def M : Set ℝ := {x : ℝ | -2 < x ∧ x < 3}
def N : Set ℝ := {-2, -1, 0, 1}

theorem intersection_M_N : M ∩ N = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1123_112349


namespace NUMINAMATH_CALUDE_simplify_expression1_simplify_expression2_l1123_112343

-- Define the expressions
def expression1 (a : ℝ) : ℝ := 5 * a^2 - 7 + 4 * a - 2 * a^2 - 9 * a + 3
def expression2 (x : ℝ) : ℝ := (5 * x^2 - 6 * x) - 3 * (2 * x^2 - 3 * x)

-- State the theorems
theorem simplify_expression1 : ∀ a : ℝ, expression1 a = 3 * a^2 - 5 * a - 4 := by sorry

theorem simplify_expression2 : ∀ x : ℝ, expression2 x = -x^2 + 3 * x := by sorry

end NUMINAMATH_CALUDE_simplify_expression1_simplify_expression2_l1123_112343


namespace NUMINAMATH_CALUDE_units_digit_of_product_l1123_112321

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_product :
  units_digit (17 * 28) = 6 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_product_l1123_112321


namespace NUMINAMATH_CALUDE_rectangle_area_change_l1123_112344

theorem rectangle_area_change (L B : ℝ) (h1 : L > 0) (h2 : B > 0) :
  let new_length := 1.1 * L
  let new_breadth := 0.9 * B
  let original_area := L * B
  let new_area := new_length * new_breadth
  new_area = 0.99 * original_area := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l1123_112344


namespace NUMINAMATH_CALUDE_sum_in_base8_l1123_112309

def base8_to_decimal (n : Nat) : Nat :=
  let digits := n.digits 10
  digits.foldr (fun d acc => acc * 8 + d) 0

theorem sum_in_base8 :
  let a := base8_to_decimal 245
  let b := base8_to_decimal 174
  let c := base8_to_decimal 354
  let sum := a + b + c
  base8_to_decimal 1015 = sum := by
sorry

end NUMINAMATH_CALUDE_sum_in_base8_l1123_112309


namespace NUMINAMATH_CALUDE_snow_probability_l1123_112308

theorem snow_probability (p1 p2 : ℚ) : 
  p1 = 1/4 → p2 = 1/3 → 
  (1 - (1 - p1)^4 * (1 - p2)^3 : ℚ) = 29/32 := by
  sorry

end NUMINAMATH_CALUDE_snow_probability_l1123_112308


namespace NUMINAMATH_CALUDE_simplified_expression_equals_zero_l1123_112320

theorem simplified_expression_equals_zero (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^3 + y^3 = x + y) : x/y + y/x - 2/(x*y) = 0 := by
  sorry

end NUMINAMATH_CALUDE_simplified_expression_equals_zero_l1123_112320


namespace NUMINAMATH_CALUDE_inequality_solution_l1123_112396

theorem inequality_solution (x : ℝ) : 
  (x + 3) / (x + 4) > (2 * x + 7) / (3 * x + 12) ↔ x < -4 ∨ x > -4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1123_112396


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l1123_112335

theorem quadratic_roots_sum (α β : ℝ) : 
  (α^2 + 3*α - 7 = 0) → 
  (β^2 + 3*β - 7 = 0) → 
  α^2 + 4*α + β = 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l1123_112335


namespace NUMINAMATH_CALUDE_smallest_number_is_10011_binary_l1123_112356

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

theorem smallest_number_is_10011_binary :
  let a := 25
  let b := 111
  let c := binary_to_decimal [false, true, true, false, true]
  let d := binary_to_decimal [true, true, false, false, true]
  d = min a (min b (min c d)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_is_10011_binary_l1123_112356


namespace NUMINAMATH_CALUDE_quadratic_constant_term_l1123_112372

theorem quadratic_constant_term (a : ℝ) : 
  ((∀ x, (a + 2) * x^2 - 3 * a * x + a - 6 = 0 → (a + 2) ≠ 0) ∧ 
   a - 6 = 0) → 
  a = 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_constant_term_l1123_112372


namespace NUMINAMATH_CALUDE_solve_for_b_l1123_112378

theorem solve_for_b (a b c d m : ℝ) (h1 : a ≠ b) (h2 : m = (c * a * d * b) / (a - b)) :
  b = (m * a) / (c * a * d + m) := by
sorry

end NUMINAMATH_CALUDE_solve_for_b_l1123_112378


namespace NUMINAMATH_CALUDE_remaining_problems_l1123_112352

theorem remaining_problems (total : ℕ) (first_20min : ℕ) (second_20min : ℕ) 
  (h1 : total = 75)
  (h2 : first_20min = 10)
  (h3 : second_20min = 2 * first_20min) :
  total - (first_20min + second_20min) = 45 := by
  sorry

end NUMINAMATH_CALUDE_remaining_problems_l1123_112352


namespace NUMINAMATH_CALUDE_smallest_base_for_fourth_power_l1123_112375

theorem smallest_base_for_fourth_power (b : ℕ) (N : ℕ) : b = 18 ↔ 
  (∃ (x : ℕ), N = x^4) ∧ 
  (11 * 30 * N).digits b = [7, 7, 7] ∧ 
  ∀ (b' : ℕ), b' < b → 
    ¬(∃ (N' : ℕ) (x' : ℕ), 
      N' = x'^4 ∧ 
      (11 * 30 * N').digits b' = [7, 7, 7]) :=
sorry

end NUMINAMATH_CALUDE_smallest_base_for_fourth_power_l1123_112375


namespace NUMINAMATH_CALUDE_marble_selection_ways_l1123_112376

theorem marble_selection_ways (n m : ℕ) (h1 : n = 9) (h2 : m = 4) :
  Nat.choose n m = 126 := by
  sorry

end NUMINAMATH_CALUDE_marble_selection_ways_l1123_112376


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1123_112319

/-- An arithmetic sequence. -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The main theorem. -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_sum : a 1 + 3 * a 8 + a 15 = 120) : 
  2 * a 9 - a 10 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1123_112319


namespace NUMINAMATH_CALUDE_sum_of_distinct_words_l1123_112312

/-- Calculates the number of distinct permutations of a word with repeated letters -/
def distinctPermutations (totalLetters : ℕ) (repeatedLetters : List ℕ) : ℕ :=
  Nat.factorial totalLetters / (repeatedLetters.map Nat.factorial).prod

/-- The word "САМСА" has 5 total letters and 2 letters that repeat twice each -/
def samsa : ℕ := distinctPermutations 5 [2, 2]

/-- The word "ПАСТА" has 5 total letters and 1 letter that repeats twice -/
def pasta : ℕ := distinctPermutations 5 [2]

theorem sum_of_distinct_words : samsa + pasta = 90 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_distinct_words_l1123_112312


namespace NUMINAMATH_CALUDE_subsets_with_even_l1123_112383

def S : Finset Nat := {1, 2, 3, 4}

theorem subsets_with_even (A : Finset (Finset Nat)) : 
  A = {s : Finset Nat | s ⊆ S ∧ ∃ n ∈ s, Even n} → Finset.card A = 12 := by
  sorry

end NUMINAMATH_CALUDE_subsets_with_even_l1123_112383


namespace NUMINAMATH_CALUDE_mothers_age_five_times_daughters_l1123_112329

/-- 
Given:
- The mother's current age is 43 years.
- The daughter's current age is 11 years.

Prove that 3 years ago, the mother's age was five times her daughter's age.
-/
theorem mothers_age_five_times_daughters (mother_age : ℕ) (daughter_age : ℕ) :
  mother_age = 43 → daughter_age = 11 → 
  ∃ (x : ℕ), x = 3 ∧ (mother_age - x) = 5 * (daughter_age - x) :=
by sorry

end NUMINAMATH_CALUDE_mothers_age_five_times_daughters_l1123_112329


namespace NUMINAMATH_CALUDE_direct_variation_with_constant_l1123_112313

/-- A function that varies directly as x with an additional constant term -/
def f (k c : ℝ) (x : ℝ) : ℝ := k * x + c

/-- Theorem stating that if f(3) = 9 and f(4) = 12, then f(-5) = -15 -/
theorem direct_variation_with_constant 
  (k c : ℝ) 
  (h1 : f k c 3 = 9) 
  (h2 : f k c 4 = 12) : 
  f k c (-5) = -15 := by
  sorry

#check direct_variation_with_constant

end NUMINAMATH_CALUDE_direct_variation_with_constant_l1123_112313


namespace NUMINAMATH_CALUDE_f_is_quadratic_l1123_112384

-- Define a quadratic function
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

-- Define the function y = x(2x - 1)
def f (x : ℝ) : ℝ := x * (2 * x - 1)

-- Theorem statement
theorem f_is_quadratic : is_quadratic f := by
  sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l1123_112384


namespace NUMINAMATH_CALUDE_quadratic_properties_l1123_112339

/-- Quadratic function defined by k -/
def f (k : ℝ) (x : ℝ) : ℝ := (k + 2) * x^2 - 2 * k * x + 3 * k

/-- Condition for vertex on x-axis -/
def vertex_on_x_axis (k : ℝ) : Prop :=
  ∃ x, f k x = 0 ∧ ∀ y, f k y ≥ f k x

/-- Condition for segment cut on x-axis equals 4 -/
def segment_cut_4 (k : ℝ) : Prop :=
  ∃ a b, a > b ∧ f k a = 0 ∧ f k b = 0 ∧ a - b = 4

/-- Main theorem -/
theorem quadratic_properties :
  (∀ k, vertex_on_x_axis k ↔ (k = 0 ∨ k = -3)) ∧
  (∀ k, segment_cut_4 k ↔ (k = -8/3 ∨ k = -1)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_properties_l1123_112339


namespace NUMINAMATH_CALUDE_min_value_expression_l1123_112358

theorem min_value_expression (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hxy : x + y = 2) :
  ((x + 1)^2 + 3) / (x + 2) + y^2 / (y + 1) ≥ 14/5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1123_112358


namespace NUMINAMATH_CALUDE_remaining_students_count_l1123_112317

/-- The number of groups with 15 students -/
def groups_15 : ℕ := 4

/-- The number of groups with 18 students -/
def groups_18 : ℕ := 2

/-- The number of students in each of the first 4 groups -/
def students_per_group_15 : ℕ := 15

/-- The number of students in each of the last 2 groups -/
def students_per_group_18 : ℕ := 18

/-- The number of students who left early from the first 4 groups -/
def left_early_15 : ℕ := 8

/-- The number of students who left early from the last 2 groups -/
def left_early_18 : ℕ := 5

/-- The total number of remaining students -/
def remaining_students : ℕ := 
  (groups_15 * students_per_group_15 - left_early_15) + 
  (groups_18 * students_per_group_18 - left_early_18)

theorem remaining_students_count : remaining_students = 83 := by
  sorry

end NUMINAMATH_CALUDE_remaining_students_count_l1123_112317


namespace NUMINAMATH_CALUDE_max_profit_increase_2008_l1123_112370

def profit_growth : Fin 10 → ℝ
  | ⟨0, _⟩ => 20
  | ⟨1, _⟩ => 40
  | ⟨2, _⟩ => 60
  | ⟨3, _⟩ => 65
  | ⟨4, _⟩ => 80
  | ⟨5, _⟩ => 85
  | ⟨6, _⟩ => 90
  | ⟨7, _⟩ => 95
  | ⟨8, _⟩ => 100
  | ⟨9, _⟩ => 80

def year_from_index (i : Fin 10) : ℕ := 2000 + 2 * i.val

def profit_increase (i : Fin 9) : ℝ := profit_growth (i.succ) - profit_growth i

theorem max_profit_increase_2008 :
  ∃ (i : Fin 9), year_from_index i.succ = 2008 ∧
  ∀ (j : Fin 9), profit_increase i ≥ profit_increase j :=
by sorry

end NUMINAMATH_CALUDE_max_profit_increase_2008_l1123_112370


namespace NUMINAMATH_CALUDE_opposite_solutions_l1123_112395

theorem opposite_solutions (k : ℝ) : k = 7 →
  ∃ (x y : ℝ), x = -y ∧ 
  (3 * (2 * x - 1) = 1 - 2 * x) ∧
  (8 - k = 2 * (y + 1)) := by
sorry

end NUMINAMATH_CALUDE_opposite_solutions_l1123_112395


namespace NUMINAMATH_CALUDE_triangle_similarity_problem_l1123_112364

theorem triangle_similarity_problem (DC CB : ℝ) (AB ED AD : ℝ) (FC : ℝ) :
  DC = 9 →
  CB = 9 →
  AB = (1 / 3) * AD →
  ED = (2 / 3) * AD →
  -- Assuming triangle similarity
  (∃ (k₁ k₂ : ℝ), k₁ > 0 ∧ k₂ > 0 ∧
    AB / AD = k₁ ∧
    FC / (CB + AB) = k₁ ∧
    ED / AD = k₂ ∧
    FC / (CB + AB) = k₂) →
  FC = 12 :=
by sorry

end NUMINAMATH_CALUDE_triangle_similarity_problem_l1123_112364


namespace NUMINAMATH_CALUDE_three_digit_number_divisible_by_6_l1123_112341

def is_divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

theorem three_digit_number_divisible_by_6 (n : ℕ) (h1 : n ≥ 500 ∧ n < 600) 
  (h2 : n % 10 = 2) (h3 : is_divisible_by_6 n) : 
  n ≥ 100 ∧ n < 1000 :=
sorry

end NUMINAMATH_CALUDE_three_digit_number_divisible_by_6_l1123_112341


namespace NUMINAMATH_CALUDE_schlaf_flachs_divisibility_l1123_112380

def SCHLAF (S C H L A F : ℕ) : ℕ := S * 10^5 + C * 10^4 + H * 10^3 + L * 10^2 + A * 10 + F

def FLACHS (F L A C H S : ℕ) : ℕ := F * 10^5 + L * 10^4 + A * 10^3 + C * 10^2 + H * 10 + S

theorem schlaf_flachs_divisibility 
  (S C H L A F : ℕ) 
  (hS : S ∈ Finset.range 10) 
  (hC : C ∈ Finset.range 10) 
  (hH : H ∈ Finset.range 10) 
  (hL : L ∈ Finset.range 10) 
  (hA : A ∈ Finset.range 10) 
  (hF : F ∈ Finset.range 10) 
  (hSnonzero : S ≠ 0) 
  (hFnonzero : F ≠ 0) : 
  (271 ∣ (SCHLAF S C H L A F - FLACHS F L A C H S)) ↔ (C = L ∧ H = A) :=
sorry

end NUMINAMATH_CALUDE_schlaf_flachs_divisibility_l1123_112380


namespace NUMINAMATH_CALUDE_smallest_sum_arithmetic_geometric_l1123_112355

theorem smallest_sum_arithmetic_geometric (A B C D : ℤ) : 
  (∃ r : ℚ, B - A = C - B ∧ C = r * B ∧ D = r * C) →  -- Arithmetic and geometric sequence conditions
  C = (3/2) * B →                                    -- Given ratio
  A + B + C + D ≥ 21 :=                              -- Smallest possible sum
by sorry

end NUMINAMATH_CALUDE_smallest_sum_arithmetic_geometric_l1123_112355


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1123_112337

theorem geometric_sequence_ratio (a : ℝ) (r : ℝ) (h1 : a > 0) (h2 : r > 0) :
  a + a * r + a * r^2 + a * r^3 = 5 * (a + a * r) →
  r = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1123_112337


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1123_112346

def f (x : ℝ) : ℝ := x^2 - 4*x + 4

theorem quadratic_inequality (x₁ x₂ y₁ y₂ : ℝ) 
  (h₁ : 1 < x₁) (h₂ : x₁ < 2) (h₃ : 3 < x₂) (h₄ : x₂ < 4)
  (hy₁ : y₁ = f x₁) (hy₂ : y₂ = f x₂) : y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1123_112346


namespace NUMINAMATH_CALUDE_money_duration_l1123_112318

def mowing_earnings : ℕ := 9
def weed_eating_earnings : ℕ := 18
def weekly_spending : ℕ := 3

theorem money_duration : 
  (mowing_earnings + weed_eating_earnings) / weekly_spending = 9 := by
  sorry

end NUMINAMATH_CALUDE_money_duration_l1123_112318


namespace NUMINAMATH_CALUDE_dot_product_of_perpendicular_vectors_l1123_112325

/-- Given two planar vectors a and b, where a = (1, √3) and a is perpendicular to (a - b),
    prove that the dot product of a and b is 4. -/
theorem dot_product_of_perpendicular_vectors (a b : ℝ × ℝ) :
  a = (1, Real.sqrt 3) →
  a.1 * (a.1 - b.1) + a.2 * (a.2 - b.2) = 0 →
  a.1 * b.1 + a.2 * b.2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_of_perpendicular_vectors_l1123_112325


namespace NUMINAMATH_CALUDE_triple_solution_l1123_112357

def is_integer (x : ℚ) : Prop := ∃ n : ℤ, x = n

def is_solution (a b c : ℕ+) : Prop :=
  (a.val > 0 ∧ b.val > 0 ∧ c.val > 0) ∧
  is_integer ((a + b : ℚ)^4 / c + (b + c : ℚ)^4 / a + (c + a : ℚ)^4 / b) ∧
  Nat.Prime (a + b + c)

theorem triple_solution :
  ∀ a b c : ℕ+, is_solution a b c ↔
    ((a, b, c) = (1, 1, 1) ∨ (a, b, c) = (2, 2, 1) ∨ (a, b, c) = (6, 3, 2)) ∨
    ((a, b, c) = (1, 2, 2) ∨ (a, b, c) = (2, 1, 2) ∨ (a, b, c) = (2, 2, 1)) ∨
    ((a, b, c) = (6, 3, 2) ∨ (a, b, c) = (6, 2, 3) ∨ (a, b, c) = (3, 6, 2) ∨
     (a, b, c) = (3, 2, 6) ∨ (a, b, c) = (2, 6, 3) ∨ (a, b, c) = (2, 3, 6)) :=
by sorry

end NUMINAMATH_CALUDE_triple_solution_l1123_112357


namespace NUMINAMATH_CALUDE_inequality_always_true_l1123_112360

theorem inequality_always_true : ∀ x : ℝ, 3 * x - 5 ≤ 12 - 2 * x + x^2 := by sorry

end NUMINAMATH_CALUDE_inequality_always_true_l1123_112360


namespace NUMINAMATH_CALUDE_family_milk_consumption_l1123_112397

/-- Represents the milk consumption of a family member -/
structure MilkConsumption where
  regular : ℝ
  soy : ℝ
  almond : ℝ
  cashew : ℝ
  oat : ℝ
  coconut : ℝ
  lactoseFree : ℝ

/-- Calculates the total milk consumption excluding lactose-free milk -/
def totalConsumption (c : MilkConsumption) : ℝ :=
  c.regular + c.soy + c.almond + c.cashew + c.oat + c.coconut

/-- Represents the family's milk consumption -/
structure FamilyConsumption where
  mitch : MilkConsumption
  sister : MilkConsumption
  mother : MilkConsumption
  father : MilkConsumption
  extraSoyMilk : ℝ

theorem family_milk_consumption (family : FamilyConsumption)
    (h_mitch : family.mitch = ⟨3, 2, 1, 0, 0, 0, 0⟩)
    (h_sister : family.sister = ⟨1.5, 3, 1.5, 1, 0, 0, 0⟩)
    (h_mother : family.mother = ⟨0.5, 2.5, 0, 0, 1, 0, 0.5⟩)
    (h_father : family.father = ⟨2, 1, 3, 0, 0, 1, 0⟩)
    (h_extra_soy : family.extraSoyMilk = 7.5) :
    totalConsumption family.mitch +
    totalConsumption family.sister +
    totalConsumption family.mother +
    totalConsumption family.father +
    family.extraSoyMilk = 31.5 := by
  sorry


end NUMINAMATH_CALUDE_family_milk_consumption_l1123_112397


namespace NUMINAMATH_CALUDE_thirteen_in_binary_l1123_112387

theorem thirteen_in_binary : 
  (13 : ℕ).digits 2 = [1, 0, 1, 1] :=
sorry

end NUMINAMATH_CALUDE_thirteen_in_binary_l1123_112387


namespace NUMINAMATH_CALUDE_negation_of_universal_quantifier_l1123_112381

theorem negation_of_universal_quantifier :
  (¬ (∀ x : ℝ, x^2 - x + 1/4 ≥ 0)) ↔ (∃ x : ℝ, x^2 - x + 1/4 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_quantifier_l1123_112381


namespace NUMINAMATH_CALUDE_y_intercept_of_f_l1123_112306

/-- A linear function f(x) = x + 1 -/
def f (x : ℝ) : ℝ := x + 1

/-- The y-intercept of f is the point (0, 1) -/
theorem y_intercept_of_f :
  f 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_f_l1123_112306


namespace NUMINAMATH_CALUDE_louisa_travel_time_l1123_112300

theorem louisa_travel_time 
  (distance_day1 : ℝ) 
  (distance_day2 : ℝ) 
  (time_difference : ℝ) 
  (h1 : distance_day1 = 200)
  (h2 : distance_day2 = 360)
  (h3 : time_difference = 4)
  (h4 : ∃ v : ℝ, v > 0 ∧ distance_day1 / v + time_difference = distance_day2 / v) :
  ∃ total_time : ℝ, total_time = 14 ∧ total_time = distance_day1 / (distance_day2 - distance_day1) * time_difference + distance_day2 / (distance_day2 - distance_day1) * time_difference :=
by sorry

end NUMINAMATH_CALUDE_louisa_travel_time_l1123_112300


namespace NUMINAMATH_CALUDE_sum_of_derived_geometric_progression_l1123_112393

/-- An arithmetic progression with the given properties -/
structure ArithmeticProgression where
  a : ℕ  -- First term
  d : ℕ  -- Common difference
  sum_first_three : a + (a + d) + (a + 2 * d) = 21
  increasing : d > 0

/-- A geometric progression derived from the arithmetic progression -/
def geometric_from_arithmetic (ap : ArithmeticProgression) : Fin 3 → ℕ
  | 0 => ap.a - 1
  | 1 => ap.a + ap.d - 1
  | 2 => ap.a + 2 * ap.d + 2

/-- The theorem to be proved -/
theorem sum_of_derived_geometric_progression (ap : ArithmeticProgression) :
  let gp := geometric_from_arithmetic ap
  let q := gp 1 / gp 0  -- Common ratio of the geometric progression
  gp 0 * (q^8 - 1) / (q - 1) = 765 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_derived_geometric_progression_l1123_112393


namespace NUMINAMATH_CALUDE_reinforcement_size_correct_l1123_112359

/-- Calculates the size of reinforcement given initial garrison size, initial provision duration,
    days before reinforcement, and remaining provision duration after reinforcement. -/
def reinforcement_size (initial_garrison : ℕ) (initial_duration : ℕ) (days_before_reinforcement : ℕ) (remaining_duration : ℕ) : ℕ :=
  let remaining_provisions := initial_garrison * (initial_duration - days_before_reinforcement)
  let total_men_days := remaining_provisions
  (total_men_days / remaining_duration) - initial_garrison

theorem reinforcement_size_correct (initial_garrison : ℕ) (initial_duration : ℕ) (days_before_reinforcement : ℕ) (remaining_duration : ℕ) 
    (h1 : initial_garrison = 2000)
    (h2 : initial_duration = 54)
    (h3 : days_before_reinforcement = 15)
    (h4 : remaining_duration = 20) :
  reinforcement_size initial_garrison initial_duration days_before_reinforcement remaining_duration = 1900 := by
  sorry

#eval reinforcement_size 2000 54 15 20

end NUMINAMATH_CALUDE_reinforcement_size_correct_l1123_112359


namespace NUMINAMATH_CALUDE_basketball_game_result_l1123_112333

/-- Calculates the final score difference after the last quarter of a basketball game -/
def final_score_difference (initial_deficit : ℤ) (liz_free_throws : ℕ) (liz_three_pointers : ℕ) (liz_jump_shots : ℕ) (opponent_points : ℕ) : ℤ :=
  initial_deficit - (liz_free_throws + 3 * liz_three_pointers + 2 * liz_jump_shots - opponent_points)

theorem basketball_game_result :
  final_score_difference 20 5 3 4 10 = 8 := by sorry

end NUMINAMATH_CALUDE_basketball_game_result_l1123_112333


namespace NUMINAMATH_CALUDE_solve_manuscript_typing_l1123_112365

def manuscript_typing_problem (total_pages : ℕ) (twice_revised : ℕ) (first_typing_cost : ℕ) (revision_cost : ℕ) (total_cost : ℕ) : Prop :=
  ∃ (once_revised : ℕ),
    once_revised + twice_revised ≤ total_pages ∧
    first_typing_cost * total_pages + revision_cost * once_revised + 2 * revision_cost * twice_revised = total_cost ∧
    once_revised = 30

theorem solve_manuscript_typing :
  manuscript_typing_problem 100 20 5 4 780 :=
sorry

end NUMINAMATH_CALUDE_solve_manuscript_typing_l1123_112365


namespace NUMINAMATH_CALUDE_loop_contains_conditional_l1123_112388

/-- Represents a flowchart structure -/
inductive FlowchartStructure
  | Sequential
  | Conditional
  | Loop

/-- Represents the containment relationship between flowchart structures -/
def contains : FlowchartStructure → FlowchartStructure → Prop := sorry

/-- A loop structure must contain a conditional structure -/
theorem loop_contains_conditional :
  ∀ (loop : FlowchartStructure), loop = FlowchartStructure.Loop →
    ∃ (cond : FlowchartStructure), cond = FlowchartStructure.Conditional ∧ contains loop cond :=
  sorry

end NUMINAMATH_CALUDE_loop_contains_conditional_l1123_112388


namespace NUMINAMATH_CALUDE_function_difference_constant_l1123_112305

open Function Real

theorem function_difference_constant 
  (f g : ℝ → ℝ) 
  (hf : Differentiable ℝ f) 
  (hg : Differentiable ℝ g)
  (h_second_deriv : ∀ x, deriv (deriv f) x = deriv (deriv g) x) :
  ∃ C, ∀ x, f x - g x = C :=
sorry

end NUMINAMATH_CALUDE_function_difference_constant_l1123_112305


namespace NUMINAMATH_CALUDE_base_for_256_with_4_digits_l1123_112367

theorem base_for_256_with_4_digits : ∃ (b : ℕ), b = 5 ∧ b^3 ≤ 256 ∧ 256 < b^4 ∧ ∀ (x : ℕ), x < b → (x^3 ≤ 256 → 256 ≥ x^4) := by
  sorry

end NUMINAMATH_CALUDE_base_for_256_with_4_digits_l1123_112367


namespace NUMINAMATH_CALUDE_opposite_sides_line_range_l1123_112374

/-- Given that the points (1, 1) and (0, 1) are on opposite sides of the line 3x - 2y + a = 0,
    the range of values for a is (-1, 2). -/
theorem opposite_sides_line_range (a : ℝ) : 
  (((3 * 1 - 2 * 1 + a) * (3 * 0 - 2 * 1 + a) < 0) ↔ -1 < a ∧ a < 2) := by
  sorry

end NUMINAMATH_CALUDE_opposite_sides_line_range_l1123_112374


namespace NUMINAMATH_CALUDE_token_passing_game_termination_l1123_112334

/-- Represents the state of the token-passing game -/
structure GameState where
  tokens : Fin 1994 → ℕ
  total_tokens : ℕ

/-- Defines a single move in the game -/
def make_move (state : GameState) (i : Fin 1994) : GameState :=
  sorry

/-- Predicate to check if the game has terminated -/
def is_terminated (state : GameState) : Prop :=
  ∀ i : Fin 1994, state.tokens i ≤ 1

/-- The main theorem about the token-passing game -/
theorem token_passing_game_termination 
  (n : ℕ) (initial_state : GameState) 
  (h_initial : ∃ i : Fin 1994, initial_state.tokens i = n ∧ 
               ∀ j : Fin 1994, j ≠ i → initial_state.tokens j = 0) 
  (h_total : initial_state.total_tokens = n) :
  (n < 1994 → ∃ (final_state : GameState), is_terminated final_state) ∧
  (n = 1994 → ∀ (state : GameState), ¬is_terminated state) :=
sorry

end NUMINAMATH_CALUDE_token_passing_game_termination_l1123_112334


namespace NUMINAMATH_CALUDE_number_difference_l1123_112353

theorem number_difference (x y : ℝ) 
  (sum_eq : x + y = 10) 
  (diff_eq : x - y = 19) 
  (square_diff_eq : x^2 - y^2 = 190) : 
  x - y = 19 := by
sorry

end NUMINAMATH_CALUDE_number_difference_l1123_112353


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1123_112328

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, 2 * x^2 - 1 > 0)) ↔ (∃ x₀ : ℝ, 2 * x₀^2 - 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1123_112328


namespace NUMINAMATH_CALUDE_solve_chocolate_problem_l1123_112330

def chocolate_problem (initial_bars : ℕ) : Prop :=
  let cost_per_bar : ℕ := 4
  let unsold_bars : ℕ := 7
  let revenue : ℕ := 16
  (initial_bars - unsold_bars) * cost_per_bar = revenue

theorem solve_chocolate_problem :
  ∃ (x : ℕ), chocolate_problem x ∧ x = 11 := by
  sorry

end NUMINAMATH_CALUDE_solve_chocolate_problem_l1123_112330


namespace NUMINAMATH_CALUDE_remainder_of_sum_of_fourth_powers_l1123_112386

theorem remainder_of_sum_of_fourth_powers (x y : ℕ+) (P Q : ℕ) :
  x^4 + y^4 = (x + y) * (P + 13) + Q ∧ Q < x + y →
  Q = 8 :=
by sorry

end NUMINAMATH_CALUDE_remainder_of_sum_of_fourth_powers_l1123_112386


namespace NUMINAMATH_CALUDE_p_range_q_range_p_or_q_range_l1123_112331

-- Define propositions p and q
def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*m - 3 > 0
def q (m : ℝ) : Prop := ∃ x : ℝ, x^2 - 2*m*x + m + 2 < 0

-- Theorem for the range of m when p is true
theorem p_range (m : ℝ) : p m ↔ m > 3/2 := by sorry

-- Theorem for the range of m when q is true
theorem q_range (m : ℝ) : q m ↔ m < -1 ∨ m > 2 := by sorry

-- Theorem for the range of m when at least one of p or q is true
theorem p_or_q_range (m : ℝ) : p m ∨ q m ↔ m < -1 ∨ m > 3/2 := by sorry

end NUMINAMATH_CALUDE_p_range_q_range_p_or_q_range_l1123_112331


namespace NUMINAMATH_CALUDE_maximal_value_S_l1123_112336

theorem maximal_value_S (a b c d : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (hd : d ≥ 0)
  (hsum : a + b + c + d = 100) :
  let S := (a / (b + 7))^(1/3) + (b / (c + 7))^(1/3) + (c / (d + 7))^(1/3) + (d / (a + 7))^(1/3)
  S ≤ 8 / 7^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_maximal_value_S_l1123_112336


namespace NUMINAMATH_CALUDE_even_quadratic_iff_b_eq_zero_l1123_112363

/-- A quadratic function f(x) = ax^2 + bx + c is even if and only if b = 0, given a ≠ 0 -/
theorem even_quadratic_iff_b_eq_zero (a b c : ℝ) (h : a ≠ 0) :
  (∀ x, a * x^2 + b * x + c = a * (-x)^2 + b * (-x) + c) ↔ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_even_quadratic_iff_b_eq_zero_l1123_112363
