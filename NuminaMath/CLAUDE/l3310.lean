import Mathlib

namespace NUMINAMATH_CALUDE_expression_evaluation_l3310_331093

theorem expression_evaluation : (1 + (3 * 5)) / 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3310_331093


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l3310_331035

open Set

def U : Finset Nat := {1, 2, 3, 4, 5}
def A : Finset Nat := {1, 2, 3}
def B : Finset Nat := {1, 4}

theorem intersection_complement_equality : A ∩ (U \ B) = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l3310_331035


namespace NUMINAMATH_CALUDE_yadav_yearly_savings_yadav_savings_l3310_331056

/-- Mr. Yadav's monthly salary savings calculation --/
theorem yadav_yearly_savings (monthly_salary : ℝ) 
  (h1 : monthly_salary * 0.2 = 4038) : 
  monthly_salary * 0.2 * 12 = 48456 := by
  sorry

/-- Main theorem: Mr. Yadav's yearly savings --/
theorem yadav_savings : ∃ (monthly_salary : ℝ), 
  monthly_salary * 0.2 = 4038 ∧ 
  monthly_salary * 0.2 * 12 = 48456 := by
  sorry

end NUMINAMATH_CALUDE_yadav_yearly_savings_yadav_savings_l3310_331056


namespace NUMINAMATH_CALUDE_english_class_grouping_l3310_331018

/-- The maximum number of groups that can be formed with equal composition -/
def maxGroups (boys girls : ℕ) : ℕ := Nat.gcd boys girls

/-- The problem statement -/
theorem english_class_grouping (boys girls : ℕ) 
  (h_boys : boys = 10) 
  (h_girls : girls = 15) : 
  maxGroups boys girls = 5 := by
  sorry

end NUMINAMATH_CALUDE_english_class_grouping_l3310_331018


namespace NUMINAMATH_CALUDE_can_capacity_is_eight_litres_l3310_331045

/-- Represents the contents and capacity of a can containing a mixture of milk and water. -/
structure Can where
  initial_milk : ℝ
  initial_water : ℝ
  capacity : ℝ

/-- Proves that the capacity of the can is 8 litres given the specified conditions. -/
theorem can_capacity_is_eight_litres (can : Can)
  (h1 : can.initial_milk / can.initial_water = 1 / 5)
  (h2 : (can.initial_milk + 2) / can.initial_water = 3 / 5)
  (h3 : can.capacity = can.initial_milk + can.initial_water + 2) :
  can.capacity = 8 := by
  sorry

#check can_capacity_is_eight_litres

end NUMINAMATH_CALUDE_can_capacity_is_eight_litres_l3310_331045


namespace NUMINAMATH_CALUDE_function_passes_through_point_l3310_331080

/-- The function f(x) = a^(x-2) + 1 passes through the point (2, 2) when a > 0 and a ≠ 1 -/
theorem function_passes_through_point (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := λ x => a^(x - 2) + 1
  f 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l3310_331080


namespace NUMINAMATH_CALUDE_expand_product_l3310_331097

theorem expand_product (x : ℝ) : (3 * x + 4) * (2 * x - 6) = 6 * x^2 - 10 * x - 24 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3310_331097


namespace NUMINAMATH_CALUDE_cube_root_problem_l3310_331094

theorem cube_root_problem (a : ℕ) : a^3 = 21 * 25 * 45 * 49 → a = 105 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_problem_l3310_331094


namespace NUMINAMATH_CALUDE_shirt_cost_calculation_l3310_331033

theorem shirt_cost_calculation (C : ℝ) : 
  (C * (1 + 0.3) * 0.5 = 13) → C = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_shirt_cost_calculation_l3310_331033


namespace NUMINAMATH_CALUDE_inequality_proof_l3310_331062

theorem inequality_proof (a b c d : ℝ) (h : a + b + c + d = 8) :
  a / (8 + b - d)^(1/3) + b / (8 + c - a)^(1/3) + c / (8 + d - b)^(1/3) + d / (8 + a - c)^(1/3) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3310_331062


namespace NUMINAMATH_CALUDE_yoyo_cost_l3310_331054

/-- Given that Mrs. Hilt bought a yoyo and a whistle for a total of 38 cents,
    and the whistle costs 14 cents, prove that the yoyo costs 24 cents. -/
theorem yoyo_cost (total : ℕ) (whistle : ℕ) (yoyo : ℕ)
    (h1 : total = 38)
    (h2 : whistle = 14)
    (h3 : total = whistle + yoyo) :
  yoyo = 24 := by
  sorry

end NUMINAMATH_CALUDE_yoyo_cost_l3310_331054


namespace NUMINAMATH_CALUDE_juggling_improvement_l3310_331079

/-- 
Given:
- start_objects: The number of objects Jeanette starts juggling with
- weeks: The number of weeks Jeanette practices
- end_objects: The number of objects Jeanette can juggle at the end
- weekly_improvement: The number of additional objects Jeanette can juggle each week

Prove that with the given conditions, the weekly improvement is 2.
-/
theorem juggling_improvement 
  (start_objects : ℕ) 
  (weeks : ℕ) 
  (end_objects : ℕ) 
  (weekly_improvement : ℕ) 
  (h1 : start_objects = 3)
  (h2 : weeks = 5)
  (h3 : end_objects = 13)
  (h4 : end_objects = start_objects + weeks * weekly_improvement) : 
  weekly_improvement = 2 := by
  sorry


end NUMINAMATH_CALUDE_juggling_improvement_l3310_331079


namespace NUMINAMATH_CALUDE_denis_neighbors_l3310_331047

-- Define the set of children
inductive Child : Type
  | Anya : Child
  | Borya : Child
  | Vera : Child
  | Gena : Child
  | Denis : Child

-- Define the line as a function from position (1 to 5) to Child
def Line : Type := Fin 5 → Child

-- Define what it means for two children to be next to each other
def NextTo (line : Line) (c1 c2 : Child) : Prop :=
  ∃ i : Fin 4, (line i = c1 ∧ line (i.succ) = c2) ∨ (line i = c2 ∧ line (i.succ) = c1)

-- Define the conditions
def LineConditions (line : Line) : Prop :=
  (line 0 = Child.Borya) ∧ 
  (NextTo line Child.Vera Child.Anya) ∧
  (¬ NextTo line Child.Vera Child.Gena) ∧
  (¬ NextTo line Child.Anya Child.Borya) ∧
  (¬ NextTo line Child.Anya Child.Gena) ∧
  (¬ NextTo line Child.Borya Child.Gena)

-- Theorem statement
theorem denis_neighbors 
  (line : Line) 
  (h : LineConditions line) : 
  (NextTo line Child.Denis Child.Anya) ∧ (NextTo line Child.Denis Child.Gena) :=
sorry

end NUMINAMATH_CALUDE_denis_neighbors_l3310_331047


namespace NUMINAMATH_CALUDE_student_average_greater_than_actual_average_l3310_331068

theorem student_average_greater_than_actual_average (x y z : ℝ) (h : x < y ∧ y < z) :
  (x + y + 2 * z) / 4 > (x + y + z) / 3 := by
  sorry

end NUMINAMATH_CALUDE_student_average_greater_than_actual_average_l3310_331068


namespace NUMINAMATH_CALUDE_max_N_value_l3310_331088

def N (a b c : ℕ) : ℕ := a * b * c + a * b + b * c + a - b - c

theorem max_N_value :
  ∃ (a b c : ℕ),
    a ∈ ({2, 3, 4, 5, 6} : Set ℕ) ∧
    b ∈ ({2, 3, 4, 5, 6} : Set ℕ) ∧
    c ∈ ({2, 3, 4, 5, 6} : Set ℕ) ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    ∀ (x y z : ℕ),
      x ∈ ({2, 3, 4, 5, 6} : Set ℕ) →
      y ∈ ({2, 3, 4, 5, 6} : Set ℕ) →
      z ∈ ({2, 3, 4, 5, 6} : Set ℕ) →
      x ≠ y → y ≠ z → x ≠ z →
      N a b c ≥ N x y z ∧
    N a b c = 167 :=
  sorry

end NUMINAMATH_CALUDE_max_N_value_l3310_331088


namespace NUMINAMATH_CALUDE_percentage_of_girls_taking_lunch_l3310_331007

theorem percentage_of_girls_taking_lunch (total : ℕ) (boys girls : ℕ) 
  (h_ratio : boys = 3 * girls / 2)
  (h_total : total = boys + girls)
  (boys_lunch : ℕ) (total_lunch : ℕ)
  (h_boys_lunch : boys_lunch = 3 * boys / 5)
  (h_total_lunch : total_lunch = 13 * total / 25) :
  (total_lunch - boys_lunch) * 5 = 2 * girls := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_girls_taking_lunch_l3310_331007


namespace NUMINAMATH_CALUDE_marbles_taken_correct_l3310_331010

/-- The number of green marbles Mike took from Dan -/
def marbles_taken (initial_green : ℕ) (remaining_green : ℕ) : ℕ :=
  initial_green - remaining_green

/-- Theorem stating that the number of marbles Mike took is the difference between
    Dan's initial and remaining green marbles -/
theorem marbles_taken_correct (initial_green : ℕ) (remaining_green : ℕ) 
    (h : initial_green ≥ remaining_green) :
  marbles_taken initial_green remaining_green = initial_green - remaining_green :=
by
  sorry

#eval marbles_taken 32 9  -- Should output 23

end NUMINAMATH_CALUDE_marbles_taken_correct_l3310_331010


namespace NUMINAMATH_CALUDE_chair_table_cost_fraction_l3310_331030

theorem chair_table_cost_fraction :
  let table_cost : ℚ := 140
  let total_cost : ℚ := 220
  let chair_cost : ℚ := (total_cost - table_cost) / 4
  chair_cost / table_cost = 1 / 7 := by
sorry

end NUMINAMATH_CALUDE_chair_table_cost_fraction_l3310_331030


namespace NUMINAMATH_CALUDE_usual_time_to_catch_bus_l3310_331064

/-- The usual time to catch the bus, given that walking with 4/5 of the usual speed
    results in missing the bus by 4 minutes, is 16 minutes. -/
theorem usual_time_to_catch_bus (T : ℝ) (S : ℝ) : T > 0 → S > 0 → S * T = (4/5 * S) * (T + 4) → T = 16 := by
  sorry

end NUMINAMATH_CALUDE_usual_time_to_catch_bus_l3310_331064


namespace NUMINAMATH_CALUDE_smallest_number_of_eggs_l3310_331021

theorem smallest_number_of_eggs : ∃ (n : ℕ), 
  (n > 150) ∧ 
  (∃ (c : ℕ), n = 18 * c - 7) ∧ 
  (∀ m : ℕ, (m > 150) ∧ (∃ (d : ℕ), m = 18 * d - 7) → m ≥ n) ∧ 
  n = 155 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_of_eggs_l3310_331021


namespace NUMINAMATH_CALUDE_y_intercept_for_specific_line_l3310_331073

/-- A line in the 2D plane. -/
structure Line where
  slope : ℝ
  x_intercept : ℝ

/-- The y-intercept of a line. -/
def y_intercept (l : Line) : ℝ × ℝ :=
  (0, l.slope * (-l.x_intercept) + 0)

/-- Theorem: For a line with slope -3 and x-intercept (7,0), the y-intercept is (0,21). -/
theorem y_intercept_for_specific_line :
  let l : Line := { slope := -3, x_intercept := 7 }
  y_intercept l = (0, 21) := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_for_specific_line_l3310_331073


namespace NUMINAMATH_CALUDE_polynomial_nonnegative_l3310_331014

theorem polynomial_nonnegative (x : ℝ) : x^8 + x^6 - 4*x^4 + x^2 + 1 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_nonnegative_l3310_331014


namespace NUMINAMATH_CALUDE_expand_product_l3310_331078

theorem expand_product (x : ℝ) : (x - 4) * (x^2 + 2*x + 1) = x^3 - 2*x^2 - 7*x - 4 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3310_331078


namespace NUMINAMATH_CALUDE_root_difference_quadratic_nonnegative_difference_roots_l3310_331016

theorem root_difference_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a * 1^2 + b * 1 + c = 0 → |r₁ - r₂| = Real.sqrt (b^2 - 4*a*c) / a :=
by sorry

theorem nonnegative_difference_roots :
  let f (x : ℝ) := x^2 + 34*x + 274
  ∃ (r₁ r₂ : ℝ), f r₁ = 0 ∧ f r₂ = 0 ∧ |r₁ - r₂| = 6 :=
by sorry

end NUMINAMATH_CALUDE_root_difference_quadratic_nonnegative_difference_roots_l3310_331016


namespace NUMINAMATH_CALUDE_e_pi_third_in_first_quadrant_l3310_331004

-- Define Euler's formula
axiom euler_formula (x : ℝ) : Complex.exp (Complex.I * x) = Complex.cos x + Complex.I * Complex.sin x

-- Define the first quadrant
def first_quadrant (z : ℂ) : Prop := 0 < z.re ∧ 0 < z.im

-- Theorem statement
theorem e_pi_third_in_first_quadrant :
  first_quadrant (Complex.exp (Complex.I * (π / 3))) :=
sorry

end NUMINAMATH_CALUDE_e_pi_third_in_first_quadrant_l3310_331004


namespace NUMINAMATH_CALUDE_octagon_area_equal_perimeter_l3310_331057

theorem octagon_area_equal_perimeter (s : Real) (o : Real) : 
  s > 0 → o > 0 →
  s^2 = 16 →
  4 * s = 8 * o →
  2 * (1 + Real.sqrt 2) * o^2 = 8 + 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_octagon_area_equal_perimeter_l3310_331057


namespace NUMINAMATH_CALUDE_louis_lemon_heads_l3310_331070

/-- The number of Lemon Heads in a package -/
def lemon_heads_per_package : ℕ := 6

/-- The number of whole boxes Louis ate -/
def boxes_eaten : ℕ := 9

/-- The total number of Lemon Heads Louis ate -/
def total_lemon_heads : ℕ := lemon_heads_per_package * boxes_eaten

theorem louis_lemon_heads : total_lemon_heads = 54 := by
  sorry

end NUMINAMATH_CALUDE_louis_lemon_heads_l3310_331070


namespace NUMINAMATH_CALUDE_cat_addition_l3310_331015

/-- Proves that buying more cats results in the correct total number of cats. -/
theorem cat_addition (initial_cats bought_cats : ℕ) :
  initial_cats = 11 →
  bought_cats = 43 →
  initial_cats + bought_cats = 54 := by
  sorry

end NUMINAMATH_CALUDE_cat_addition_l3310_331015


namespace NUMINAMATH_CALUDE_complement_is_acute_l3310_331040

-- Define an angle as a real number between 0 and 180 degrees
def Angle := {x : ℝ // 0 ≤ x ∧ x ≤ 180}

-- Define an acute angle
def isAcute (a : Angle) : Prop := a.val < 90

-- Define the complement of an angle
def complement (a : Angle) : Angle :=
  ⟨90 - a.val, by sorry⟩  -- The proof that this is a valid angle is omitted

-- Theorem statement
theorem complement_is_acute (a : Angle) (h : a.val < 90) : isAcute (complement a) := by
  sorry


end NUMINAMATH_CALUDE_complement_is_acute_l3310_331040


namespace NUMINAMATH_CALUDE_system_of_equations_sum_l3310_331049

theorem system_of_equations_sum (x y : ℝ) :
  3 * x + 2 * y = 10 →
  2 * x + 3 * y = 5 →
  x + y = 3 := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_sum_l3310_331049


namespace NUMINAMATH_CALUDE_paper_products_pallets_l3310_331052

theorem paper_products_pallets : ∃ P : ℚ,
  P / 2 + P / 4 + P / 5 + 1 = P ∧ P = 20 := by
  sorry

end NUMINAMATH_CALUDE_paper_products_pallets_l3310_331052


namespace NUMINAMATH_CALUDE_scalene_triangles_count_l3310_331081

def is_valid_scalene_triangle (a b c : ℕ) : Prop :=
  a < b ∧ b < c ∧ a + b + c < 20 ∧ a + b > c ∧ a + c > b ∧ b + c > a

theorem scalene_triangles_count :
  ∃ (S : Finset (ℕ × ℕ × ℕ)), 
    (∀ (t : ℕ × ℕ × ℕ), t ∈ S → is_valid_scalene_triangle t.1 t.2.1 t.2.2) ∧
    S.card > 7 :=
sorry

end NUMINAMATH_CALUDE_scalene_triangles_count_l3310_331081


namespace NUMINAMATH_CALUDE_unique_function_on_rationals_l3310_331053

theorem unique_function_on_rationals
  (f : ℚ → ℝ)
  (h1 : ∀ x y : ℚ, f (x + y) - y * f x - x * f y = f x * f y - x - y + x * y)
  (h2 : ∀ x : ℚ, f x = 2 * f (x + 1) + 2 + x)
  (h3 : f 1 + 1 > 0) :
  ∀ x : ℚ, f x = -x / 2 := by sorry

end NUMINAMATH_CALUDE_unique_function_on_rationals_l3310_331053


namespace NUMINAMATH_CALUDE_ava_apple_trees_l3310_331087

theorem ava_apple_trees (lily_trees : ℕ) : 
  (lily_trees + 3) + lily_trees = 15 → (lily_trees + 3) = 9 := by
  sorry

end NUMINAMATH_CALUDE_ava_apple_trees_l3310_331087


namespace NUMINAMATH_CALUDE_right_triangle_area_l3310_331005

theorem right_triangle_area (a b : ℝ) (h1 : a = 30) (h2 : b = 34) : 
  (1/2) * a * b = 510 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3310_331005


namespace NUMINAMATH_CALUDE_corn_yield_theorem_l3310_331083

/-- Calculates the total corn yield for Johnson and his neighbor after 6 months -/
def total_corn_yield (johnson_yield : ℕ) (johnson_area : ℕ) (neighbor_area : ℕ) (months : ℕ) : ℕ :=
  let johnson_total := johnson_yield * (months / 2)
  let neighbor_yield := 2 * johnson_yield
  let neighbor_total := neighbor_yield * neighbor_area * (months / 2)
  johnson_total + neighbor_total

/-- Theorem stating that the total corn yield is 1200 under given conditions -/
theorem corn_yield_theorem :
  total_corn_yield 80 1 2 6 = 1200 :=
by
  sorry

end NUMINAMATH_CALUDE_corn_yield_theorem_l3310_331083


namespace NUMINAMATH_CALUDE_parallelogram_angle_measure_l3310_331038

theorem parallelogram_angle_measure (α β : ℝ) : 
  (α + β = π) →  -- Adjacent angles in a parallelogram sum to π
  (β = α + π/9) →  -- One angle exceeds the other by π/9
  (α = 4*π/9) :=  -- The smaller angle is 4π/9
by sorry

end NUMINAMATH_CALUDE_parallelogram_angle_measure_l3310_331038


namespace NUMINAMATH_CALUDE_total_goals_scored_l3310_331086

def soccer_match (team_a_first_half : ℕ) (team_b_second_half : ℕ) : Prop :=
  let team_b_first_half := team_a_first_half / 2
  let team_a_second_half := team_b_second_half - 2
  let team_a_total := team_a_first_half + team_a_second_half
  let team_b_total := team_b_first_half + team_b_second_half
  team_a_total + team_b_total = 26

theorem total_goals_scored :
  soccer_match 8 8 := by sorry

end NUMINAMATH_CALUDE_total_goals_scored_l3310_331086


namespace NUMINAMATH_CALUDE_second_prime_range_l3310_331046

theorem second_prime_range (p q : ℕ) (hp : Prime p) (hq : Prime q) : 
  15 < p * q ∧ p * q ≤ 36 → 2 < p ∧ p < 6 → p * q = 33 → q = 11 := by
  sorry

end NUMINAMATH_CALUDE_second_prime_range_l3310_331046


namespace NUMINAMATH_CALUDE_logarithm_identity_l3310_331034

theorem logarithm_identity (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (ha_ne_one : a ≠ 1) (hb_ne_one : b ≠ 1) : 
  Real.log c / Real.log (a * b) = (Real.log c / Real.log a * Real.log c / Real.log b) / 
    (Real.log c / Real.log a + Real.log c / Real.log b) := by
  sorry

end NUMINAMATH_CALUDE_logarithm_identity_l3310_331034


namespace NUMINAMATH_CALUDE_chocolate_gain_percent_l3310_331036

/-- Calculates the gain percent given the cost price and selling price ratio -/
theorem chocolate_gain_percent 
  (cost_price selling_price : ℝ) 
  (h : 81 * cost_price = 45 * selling_price) :
  (selling_price - cost_price) / cost_price * 100 = 80 := by
sorry

end NUMINAMATH_CALUDE_chocolate_gain_percent_l3310_331036


namespace NUMINAMATH_CALUDE_vector_angle_problem_l3310_331092

theorem vector_angle_problem (α β : ℝ) (a b : ℝ × ℝ) 
  (h1 : a = (Real.cos α, Real.sin α))
  (h2 : b = (Real.cos β, Real.sin β))
  (h3 : ‖a - b‖ = (2 / 5) * Real.sqrt 5)
  (h4 : -π/2 < β ∧ β < 0 ∧ 0 < α ∧ α < π/2)
  (h5 : Real.sin β = -5/13) :
  Real.cos (α - β) = 3/5 ∧ Real.sin α = 33/65 := by
  sorry

end NUMINAMATH_CALUDE_vector_angle_problem_l3310_331092


namespace NUMINAMATH_CALUDE_root_conditions_imply_a_range_l3310_331017

/-- Given a quadratic function f(x) = x² + (a² - 1)x + (a - 2) where 'a' is a real number,
    if one root of f(x) is greater than 1 and the other root is less than 1,
    then 'a' is in the open interval (-2, 1). -/
theorem root_conditions_imply_a_range (a : ℝ) :
  let f := fun x : ℝ => x^2 + (a^2 - 1)*x + (a - 2)
  (∃ r₁ r₂ : ℝ, f r₁ = 0 ∧ f r₂ = 0 ∧ r₁ > 1 ∧ r₂ < 1) →
  -2 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_root_conditions_imply_a_range_l3310_331017


namespace NUMINAMATH_CALUDE_irrational_between_neg_one_and_two_l3310_331099

theorem irrational_between_neg_one_and_two :
  (-1 < Real.sqrt 3 ∧ Real.sqrt 3 < 2) ∧
  ¬(-1 < -Real.sqrt 3 ∧ -Real.sqrt 3 < 2) ∧
  ¬(-1 < -Real.sqrt 5 ∧ -Real.sqrt 5 < 2) ∧
  ¬(-1 < Real.sqrt 5 ∧ Real.sqrt 5 < 2) :=
by
  sorry

end NUMINAMATH_CALUDE_irrational_between_neg_one_and_two_l3310_331099


namespace NUMINAMATH_CALUDE_smith_family_seating_arrangements_l3310_331090

/-- Represents a family with parents and children -/
structure Family :=
  (num_parents : Nat)
  (num_children : Nat)

/-- Represents a car with front and back seats -/
structure Car :=
  (front_seats : Nat)
  (back_seats : Nat)

/-- Calculates the number of seating arrangements for a family in a car -/
def seating_arrangements (f : Family) (c : Car) (parent_driver : Bool) : Nat :=
  sorry

/-- The Smith family with 2 parents and 3 children -/
def smith_family : Family :=
  { num_parents := 2, num_children := 3 }

/-- The Smith family car with 2 front seats and 3 back seats -/
def smith_car : Car :=
  { front_seats := 2, back_seats := 3 }

theorem smith_family_seating_arrangements :
  seating_arrangements smith_family smith_car true = 48 := by
  sorry

end NUMINAMATH_CALUDE_smith_family_seating_arrangements_l3310_331090


namespace NUMINAMATH_CALUDE_cubic_equation_root_sum_l3310_331065

/-- Given a cubic equation ax³ + bx² + cx + d = 0 with a ≠ 0 and roots 4 and -1, prove b+c = -13a -/
theorem cubic_equation_root_sum (a b c d : ℝ) (ha : a ≠ 0) : 
  (∀ x, a * x^3 + b * x^2 + c * x + d = 0 ↔ x = 4 ∨ x = -1 ∨ x = -(b + c + 13 * a) / a) →
  b + c = -13 * a :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_root_sum_l3310_331065


namespace NUMINAMATH_CALUDE_x_range_l3310_331055

theorem x_range (x y : ℝ) (h : x - 4 * Real.sqrt y = 2 * Real.sqrt (x - y)) :
  x ∈ Set.Icc 0 20 := by
  sorry

end NUMINAMATH_CALUDE_x_range_l3310_331055


namespace NUMINAMATH_CALUDE_no_functions_satisfying_condition_l3310_331043

theorem no_functions_satisfying_condition :
  ¬ (∃ (f g : ℝ → ℝ), ∀ (x y : ℝ), x ≠ y → |f x - f y| + |g x - g y| > 1) :=
by sorry

end NUMINAMATH_CALUDE_no_functions_satisfying_condition_l3310_331043


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l3310_331096

def vector_a : ℝ × ℝ := (4, -5)
def vector_b (b : ℝ) : ℝ × ℝ := (b, 3)

def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem perpendicular_vectors (b : ℝ) :
  dot_product vector_a (vector_b b) = 0 → b = 15/4 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l3310_331096


namespace NUMINAMATH_CALUDE_spinner_probability_l3310_331058

theorem spinner_probability (pA pB pC pD : ℚ) : 
  pA = 1/4 → pB = 1/3 → pD = 1/6 → pA + pB + pC + pD = 1 → pC = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_spinner_probability_l3310_331058


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3310_331095

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_a2 : a 2 = 3)
  (h_a5 : a 5 = 12) :
  a 8 = 21 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3310_331095


namespace NUMINAMATH_CALUDE_geometric_sequence_11th_term_l3310_331061

/-- 
Given a geometric sequence where:
  a₅ = 5 (5th term is 5)
  a₈ = 40 (8th term is 40)
Prove that a₁₁ = 320 (11th term is 320)
-/
theorem geometric_sequence_11th_term 
  (a : ℕ → ℝ) -- The geometric sequence
  (h₁ : a 5 = 5) -- 5th term is 5
  (h₂ : a 8 = 40) -- 8th term is 40
  (h₃ : ∀ n m : ℕ, a (n + m) = a n * (a 6 / a 5) ^ m) -- Geometric sequence property
  : a 11 = 320 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_11th_term_l3310_331061


namespace NUMINAMATH_CALUDE_lucy_crayons_l3310_331032

/-- Given that Willy has 1400 crayons and 1110 more crayons than Lucy, 
    prove that Lucy has 290 crayons. -/
theorem lucy_crayons (willy_crayons : ℕ) (difference : ℕ) (lucy_crayons : ℕ) 
  (h1 : willy_crayons = 1400) 
  (h2 : difference = 1110) 
  (h3 : willy_crayons = lucy_crayons + difference) : 
  lucy_crayons = 290 := by
  sorry

end NUMINAMATH_CALUDE_lucy_crayons_l3310_331032


namespace NUMINAMATH_CALUDE_sum_of_integers_l3310_331077

theorem sum_of_integers (x y : ℕ+) 
  (h1 : x.val - y.val = 8) 
  (h2 : x.val * y.val = 135) : 
  x.val + y.val = 26 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_l3310_331077


namespace NUMINAMATH_CALUDE_sqrt_sum_difference_equality_l3310_331085

theorem sqrt_sum_difference_equality : 
  Real.sqrt 27 + Real.sqrt (1/3) - Real.sqrt 2 * Real.sqrt 6 = (4 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_difference_equality_l3310_331085


namespace NUMINAMATH_CALUDE_irrational_sqrt_sin_cos_l3310_331000

theorem irrational_sqrt_sin_cos (θ : Real) (h : 0 < θ ∧ θ < π / 2) :
  ¬(∃ (a b c d : ℤ), b ≠ 0 ∧ d ≠ 0 ∧ 
    Real.sqrt (Real.sin θ) = a / b ∧ 
    Real.sqrt (Real.cos θ) = c / d) :=
by sorry

end NUMINAMATH_CALUDE_irrational_sqrt_sin_cos_l3310_331000


namespace NUMINAMATH_CALUDE_max_two_digit_decimals_l3310_331029

def digits : List Nat := [2, 0, 5]

def is_valid_two_digit_decimal (n : Nat) (d : Nat) : Bool :=
  n ∈ digits ∧ d ∈ digits ∧ (n ≠ 0 ∨ d ≠ 0)

def count_valid_decimals : Nat :=
  (List.filter (fun (pair : Nat × Nat) => is_valid_two_digit_decimal pair.1 pair.2)
    (List.product digits digits)).length

theorem max_two_digit_decimals : count_valid_decimals = 6 := by
  sorry

end NUMINAMATH_CALUDE_max_two_digit_decimals_l3310_331029


namespace NUMINAMATH_CALUDE_unique_base6_divisible_by_13_l3310_331048

/-- Converts a base-6 number of the form 2dd3₆ to base 10 --/
def base6_to_base10 (d : Nat) : Nat :=
  2 * 6^3 + d * 6^2 + d * 6 + 3

/-- Checks if a number is divisible by 13 --/
def divisible_by_13 (n : Nat) : Prop :=
  n % 13 = 0

/-- Theorem stating that 2553₆ is divisible by 13 and is the only number of the form 2dd3₆ with this property --/
theorem unique_base6_divisible_by_13 :
  divisible_by_13 (base6_to_base10 5) ∧
  ∀ d : Nat, d < 6 → d ≠ 5 → ¬(divisible_by_13 (base6_to_base10 d)) :=
by sorry

end NUMINAMATH_CALUDE_unique_base6_divisible_by_13_l3310_331048


namespace NUMINAMATH_CALUDE_x_minus_y_value_l3310_331066

theorem x_minus_y_value (x y : ℝ) 
  (h1 : |x| = 3)
  (h2 : y^2 = 1/4)
  (h3 : x + y < 0) :
  x - y = -7/2 ∨ x - y = -5/2 :=
sorry

end NUMINAMATH_CALUDE_x_minus_y_value_l3310_331066


namespace NUMINAMATH_CALUDE_complement_of_union_l3310_331076

open Set

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 2, 3}
def B : Set Nat := {3, 4}

theorem complement_of_union :
  (U \ (A ∪ B)) = {5} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l3310_331076


namespace NUMINAMATH_CALUDE_spending_solution_l3310_331067

def spending_problem (n : ℚ) : Prop :=
  let after_hardware := (3/4) * n
  let after_cleaners := after_hardware - 9
  let after_grocery := (1/2) * after_cleaners
  after_grocery = 12

theorem spending_solution : 
  ∃ (n : ℚ), spending_problem n ∧ n = 44 :=
sorry

end NUMINAMATH_CALUDE_spending_solution_l3310_331067


namespace NUMINAMATH_CALUDE_proportionality_problem_l3310_331071

/-- Given that x is directly proportional to y², y is inversely proportional to z²,
    and x = 5 when z = 8, prove that x = 5/256 when z = 32 -/
theorem proportionality_problem (x y z : ℝ) (k₁ k₂ : ℝ) 
    (h₁ : x = k₁ * y^2)
    (h₂ : y * z^2 = k₂)
    (h₃ : x = 5 ∧ z = 8) :
  x = 5/256 ∧ z = 32 := by
  sorry

end NUMINAMATH_CALUDE_proportionality_problem_l3310_331071


namespace NUMINAMATH_CALUDE_second_stop_off_count_l3310_331019

/-- Represents the number of passengers on the bus after each stop -/
def passengers : List ℕ := [0, 7, 0, 11]

/-- Represents the number of people getting on at each stop -/
def people_on : List ℕ := [7, 5, 4]

/-- Represents the number of people getting off at each stop -/
def people_off : List ℕ := [0, 0, 2]

/-- The unknown number of people who got off at the second stop -/
def x : ℕ := sorry

theorem second_stop_off_count :
  x = 3 ∧
  passengers[3] = passengers[1] + people_on[1] - x + people_on[2] - people_off[2] :=
by sorry

end NUMINAMATH_CALUDE_second_stop_off_count_l3310_331019


namespace NUMINAMATH_CALUDE_expression_value_l3310_331013

theorem expression_value : 
  let a : ℝ := 1.69
  let b : ℝ := 1.73
  let c : ℝ := 0.48
  1 / (a^2 - a*c - a*b + b*c) + 2 / (b^2 - a*b - b*c + a*c) + 1 / (c^2 - a*c - b*c + a*b) = 20 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l3310_331013


namespace NUMINAMATH_CALUDE_worker_daily_hours_l3310_331008

/-- Represents the number of work hours per day for a worker -/
def daily_hours (total_hours : ℕ) (days_per_week : ℕ) (weeks_per_month : ℕ) : ℚ :=
  total_hours / (days_per_week * weeks_per_month)

/-- Theorem stating that under given conditions, a worker's daily work hours are 10 -/
theorem worker_daily_hours :
  let total_hours : ℕ := 200
  let days_per_week : ℕ := 5
  let weeks_per_month : ℕ := 4
  daily_hours total_hours days_per_week weeks_per_month = 10 := by
  sorry

end NUMINAMATH_CALUDE_worker_daily_hours_l3310_331008


namespace NUMINAMATH_CALUDE_grade_average_condition_l3310_331089

theorem grade_average_condition (grades : List ℤ) (n : ℕ) :
  n > 0 →
  n = grades.length →
  (grades.sum : ℚ) / n = 46 / 10 →
  ∃ k : ℕ, n = 5 * k :=
by sorry

end NUMINAMATH_CALUDE_grade_average_condition_l3310_331089


namespace NUMINAMATH_CALUDE_problem_solution_l3310_331028

theorem problem_solution (x y : ℚ) : 
  x = 103 → x^3 * y - 4 * x^2 * y + 4 * x * y = 1106600 → y = 1085/1030 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3310_331028


namespace NUMINAMATH_CALUDE_max_xy_given_constraint_l3310_331020

theorem max_xy_given_constraint (x y : ℝ) : 
  x > 0 → y > 0 → x + 4 * y = 1 → x * y ≤ 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_max_xy_given_constraint_l3310_331020


namespace NUMINAMATH_CALUDE_coefficient_theorem_l3310_331059

theorem coefficient_theorem (a : ℝ) : 
  (∃ c : ℝ, c = 6 * a^2 - 15 * a + 20 ∧ c = 56) → (a = 6 ∨ a = -1) := by sorry

end NUMINAMATH_CALUDE_coefficient_theorem_l3310_331059


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3310_331060

theorem sum_of_roots_quadratic (x : ℝ) : 
  (x^2 = 16*x - 4) → (∃ y : ℝ, y^2 = 16*y - 4 ∧ x + y = 16) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3310_331060


namespace NUMINAMATH_CALUDE_solve_for_a_l3310_331075

theorem solve_for_a : ∃ (a : ℝ), 
  let A : Set ℝ := {2, 3, a^2 + 2*a - 3}
  let B : Set ℝ := {|a + 3|, 2}
  5 ∈ A ∧ 5 ∉ B ∧ a = -4 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l3310_331075


namespace NUMINAMATH_CALUDE_white_balls_count_l3310_331009

/-- Represents a box of balls with white and black colors. -/
structure BallBox where
  total : ℕ
  white : ℕ
  black : ℕ
  sum_correct : white + black = total
  white_condition : ∀ (n : ℕ), n ≥ 12 → n.choose white > 0
  black_condition : ∀ (n : ℕ), n ≥ 20 → n.choose black > 0

/-- Theorem stating that a box with 30 balls satisfying the given conditions has 19 white balls. -/
theorem white_balls_count (box : BallBox) (h_total : box.total = 30) : box.white = 19 := by
  sorry

end NUMINAMATH_CALUDE_white_balls_count_l3310_331009


namespace NUMINAMATH_CALUDE_water_addition_proof_l3310_331003

/-- Proves that adding 3 litres of water to 11 litres of 42% alcohol solution results in 33% alcohol mixture -/
theorem water_addition_proof (initial_volume : ℝ) (initial_alcohol_percent : ℝ) 
  (final_alcohol_percent : ℝ) (water_added : ℝ) : 
  initial_volume = 11 →
  initial_alcohol_percent = 0.42 →
  final_alcohol_percent = 0.33 →
  water_added = 3 →
  initial_volume * initial_alcohol_percent = 
    (initial_volume + water_added) * final_alcohol_percent := by
  sorry

#check water_addition_proof

end NUMINAMATH_CALUDE_water_addition_proof_l3310_331003


namespace NUMINAMATH_CALUDE_smallest_fraction_l3310_331006

theorem smallest_fraction (x : ℝ) (h : x = 9) : 
  min (8/x) (min (8/(x+2)) (min (8/(x-2)) (min (x/8) (x^2/64)))) = 8/(x+2) := by
  sorry

end NUMINAMATH_CALUDE_smallest_fraction_l3310_331006


namespace NUMINAMATH_CALUDE_triangle_inequality_l3310_331084

theorem triangle_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b) :
  Real.sqrt (a^2 + a*b + b^2) + Real.sqrt (b^2 + b*c + c^2) + Real.sqrt (c^2 + c*a + a^2)
  ≤ Real.sqrt (5*a^2 + 5*b^2 + 5*c^2 + 4*a*b + 4*b*c + 4*c*a) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3310_331084


namespace NUMINAMATH_CALUDE_sector_area_l3310_331051

theorem sector_area (r : ℝ) (θ : ℝ) (h : θ = 2 * Real.pi / 3) :
  let area := (1 / 2) * r^2 * θ
  area = 3 * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_sector_area_l3310_331051


namespace NUMINAMATH_CALUDE_theresa_video_games_l3310_331091

/-- The number of video games each person has -/
structure VideoGames where
  theresa : ℕ
  julia : ℕ
  tory : ℕ
  alex : ℕ

/-- The conditions of the problem -/
def satisfies_conditions (vg : VideoGames) : Prop :=
  vg.theresa = 3 * vg.julia + 5 ∧
  vg.julia = vg.tory / 3 ∧
  vg.tory = 2 * vg.alex ∧
  vg.tory = 6

/-- The theorem to prove -/
theorem theresa_video_games (vg : VideoGames) (h : satisfies_conditions vg) : vg.theresa = 11 := by
  sorry

#check theresa_video_games

end NUMINAMATH_CALUDE_theresa_video_games_l3310_331091


namespace NUMINAMATH_CALUDE_square_of_102_l3310_331069

theorem square_of_102 : 102 * 102 = 10404 := by
  sorry

end NUMINAMATH_CALUDE_square_of_102_l3310_331069


namespace NUMINAMATH_CALUDE_points_one_unit_from_negative_two_l3310_331074

theorem points_one_unit_from_negative_two : 
  ∀ x : ℝ, abs (x - (-2)) = 1 ↔ x = -3 ∨ x = -1 := by sorry

end NUMINAMATH_CALUDE_points_one_unit_from_negative_two_l3310_331074


namespace NUMINAMATH_CALUDE_consecutive_sum_100_l3310_331050

theorem consecutive_sum_100 (n : ℕ) : 
  (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 100) → n = 18 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_sum_100_l3310_331050


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l3310_331011

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_fourth_term
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h1 : a 1 + a 2 = -1)
  (h2 : a 1 - a 3 = -3) :
  a 4 = -8 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l3310_331011


namespace NUMINAMATH_CALUDE_unique_common_root_value_l3310_331039

theorem unique_common_root_value (m : ℝ) : 
  m > 5 →
  (∃! x : ℝ, x^2 - 5*x + 6 = 0 ∧ x^2 + 2*x - 2*m + 1 = 0) →
  m = 8 := by
sorry

end NUMINAMATH_CALUDE_unique_common_root_value_l3310_331039


namespace NUMINAMATH_CALUDE_vegetable_ghee_mixture_ratio_l3310_331098

/-- The ratio of volumes of two brands of vegetable ghee in a mixture -/
theorem vegetable_ghee_mixture_ratio :
  ∀ (Va Vb : ℝ),
  Va + Vb = 4 →
  900 * Va + 850 * Vb = 3520 →
  Va / Vb = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_vegetable_ghee_mixture_ratio_l3310_331098


namespace NUMINAMATH_CALUDE_logarithm_sum_equality_l3310_331063

theorem logarithm_sum_equality : 2 * Real.log 10 / Real.log 5 + Real.log 0.25 / Real.log 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_sum_equality_l3310_331063


namespace NUMINAMATH_CALUDE_dolls_distribution_l3310_331042

theorem dolls_distribution (total_dolls : ℕ) (defective_dolls : ℕ) (num_stores : ℕ) : 
  total_dolls = 40 → defective_dolls = 4 → num_stores = 4 →
  (total_dolls - defective_dolls) / num_stores = 9 := by
  sorry

end NUMINAMATH_CALUDE_dolls_distribution_l3310_331042


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3310_331023

/-- A quadratic function with specific properties -/
def f (x : ℝ) : ℝ := x^2 - 4*x + 3

/-- Another function defined in terms of f -/
def g (k : ℝ) (x : ℝ) : ℝ := f x - k*x

/-- Theorem stating the properties of f and g -/
theorem quadratic_function_properties :
  (∀ x, f x ≥ -1) ∧ 
  (f 2 = -1) ∧ 
  (f 1 + f 4 = 3) ∧
  (∀ k, (∀ x ∈ Set.Ioo 1 4, ∃ y ∈ Set.Ioo 1 4, g k y < g k x) ↔ 
    k ∈ Set.Iic (-2) ∪ Set.Ici 4) := by sorry


end NUMINAMATH_CALUDE_quadratic_function_properties_l3310_331023


namespace NUMINAMATH_CALUDE_sum_of_roots_cubic_equation_l3310_331037

theorem sum_of_roots_cubic_equation :
  let f : ℝ → ℝ := λ x => 6 * x^3 - 3 * x^2 - 18 * x + 9
  ∃ (r₁ r₂ r₃ : ℝ), (∀ x, f x = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) ∧ r₁ + r₂ + r₃ = 0.5 :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_cubic_equation_l3310_331037


namespace NUMINAMATH_CALUDE_rectangle_midpoint_angle_equality_l3310_331025

-- Define the rectangle ABCD
variable (A B C D : Point)

-- Define the property of being a rectangle
def is_rectangle (A B C D : Point) : Prop := sorry

-- Define the midpoint property
def is_midpoint (M A D : Point) : Prop := sorry

-- Define a point on the extension of a line segment
def on_extension (P D C : Point) : Prop := sorry

-- Define the intersection of two lines
def intersection (Q P M A C : Point) : Prop := sorry

-- Define the angle equality
def angle_eq (Q N M P : Point) : Prop := sorry

-- State the theorem
theorem rectangle_midpoint_angle_equality 
  (h_rect : is_rectangle A B C D)
  (h_midpoint_M : is_midpoint M A D)
  (h_midpoint_N : is_midpoint N B C)
  (h_extension_P : on_extension P D C)
  (h_intersection_Q : intersection Q P M A C) :
  angle_eq Q N M P :=
sorry

end NUMINAMATH_CALUDE_rectangle_midpoint_angle_equality_l3310_331025


namespace NUMINAMATH_CALUDE_triangle_special_x_values_l3310_331082

/-- A triangle with side lengths a, b, c and circumradius R -/
structure Triangle (a b c R : ℝ) : Prop where
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b
  circumradius : R = (a * b * c) / (4 * area)
  area_positive : 0 < area

/-- The main theorem -/
theorem triangle_special_x_values
  (a b c : ℝ)
  (h_triangle : Triangle a b c 2)
  (h_angle : a^2 + c^2 ≤ b^2)
  (h_polynomial : ∃ x : ℝ, x^4 + a*x^3 + b*x^2 + c*x + 1 = 0) :
  ∃ x : ℝ, x = -1/2 * (Real.sqrt 6 + Real.sqrt 2) ∨ x = -1/2 * (Real.sqrt 6 - Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_triangle_special_x_values_l3310_331082


namespace NUMINAMATH_CALUDE_no_prime_solution_for_equation_l3310_331002

theorem no_prime_solution_for_equation : 
  ¬∃ (x y z : ℕ), Prime x ∧ Prime y ∧ Prime z ∧ x^2 + y^3 = z^4 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_solution_for_equation_l3310_331002


namespace NUMINAMATH_CALUDE_carla_project_days_l3310_331044

/-- The number of days needed to complete a project given the number of items to collect and items collected per day. -/
def daysNeeded (leaves : ℕ) (bugs : ℕ) (itemsPerDay : ℕ) : ℕ :=
  (leaves + bugs) / itemsPerDay

/-- Theorem: Carla needs 10 days to complete the project. -/
theorem carla_project_days : daysNeeded 30 20 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_carla_project_days_l3310_331044


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3310_331012

theorem inequality_solution_set (x : ℝ) : 6 + 5*x - x^2 > 0 ↔ -1 < x ∧ x < 6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3310_331012


namespace NUMINAMATH_CALUDE_smallest_base_for_100_l3310_331026

theorem smallest_base_for_100 :
  ∃ (b : ℕ), b = 5 ∧ b^2 ≤ 100 ∧ 100 < b^3 ∧ ∀ (x : ℕ), x < b → (x^2 ≤ 100 → 100 ≥ x^3) :=
by sorry

end NUMINAMATH_CALUDE_smallest_base_for_100_l3310_331026


namespace NUMINAMATH_CALUDE_sum_a_b_equals_six_l3310_331031

theorem sum_a_b_equals_six (a b : ℝ) 
  (eq1 : 3 * a + 5 * b = 22) 
  (eq2 : 4 * a + 2 * b = 20) : 
  a + b = 6 := by
sorry

end NUMINAMATH_CALUDE_sum_a_b_equals_six_l3310_331031


namespace NUMINAMATH_CALUDE_smallest_factor_of_4814_l3310_331072

theorem smallest_factor_of_4814 (a b : ℕ) : 
  10 ≤ a ∧ a ≤ 99 ∧
  10 ≤ b ∧ b ≤ 99 ∧
  a * b = 4814 ∧
  a ≤ b →
  a = 53 := by sorry

end NUMINAMATH_CALUDE_smallest_factor_of_4814_l3310_331072


namespace NUMINAMATH_CALUDE_rectangle_area_is_six_l3310_331024

/-- The quadratic equation representing the sides of the rectangle -/
def quadratic_equation (x : ℝ) : Prop := x^2 - 5*x + 6 = 0

/-- The roots of the quadratic equation -/
def roots : Set ℝ := {x : ℝ | quadratic_equation x}

/-- The rectangle with sides equal to the roots of the quadratic equation -/
structure Rectangle where
  side1 : ℝ
  side2 : ℝ
  side1_root : quadratic_equation side1
  side2_root : quadratic_equation side2
  different_sides : side1 ≠ side2

/-- The area of the rectangle -/
def area (rect : Rectangle) : ℝ := rect.side1 * rect.side2

/-- Theorem: The area of the rectangle is 6 -/
theorem rectangle_area_is_six (rect : Rectangle) : area rect = 6 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_is_six_l3310_331024


namespace NUMINAMATH_CALUDE_team_formation_count_l3310_331001

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of female teachers -/
def num_female : ℕ := 4

/-- The number of male teachers -/
def num_male : ℕ := 5

/-- The total number of teachers to be selected -/
def team_size : ℕ := 3

theorem team_formation_count : 
  choose num_female 1 * choose num_male 2 + choose num_female 2 * choose num_male 1 = 70 := by
  sorry

end NUMINAMATH_CALUDE_team_formation_count_l3310_331001


namespace NUMINAMATH_CALUDE_f_strictly_increasing_l3310_331027

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 2*x^2 - 4*x + 2

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := 3*x^2 - 4*x - 4

-- Theorem statement
theorem f_strictly_increasing :
  (∀ x y, x < y ∧ x < -2/3 → f x < f y) ∧
  (∀ x y, x < y ∧ 2 < x → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_f_strictly_increasing_l3310_331027


namespace NUMINAMATH_CALUDE_y_value_l3310_331022

theorem y_value (y : ℕ) (h1 : ∃ k : ℕ, y = 9 * k) (h2 : y^2 > 200) (h3 : y < 30) : y = 18 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l3310_331022


namespace NUMINAMATH_CALUDE_smallest_integer_980_divisors_l3310_331041

theorem smallest_integer_980_divisors (n m k : ℕ) : 
  (∀ i < n, (Nat.divisors i).card ≠ 980) →
  (Nat.divisors n).card = 980 →
  n = m * 4^k →
  ¬(4 ∣ m) →
  (∀ j, j < n → (Nat.divisors j).card = 980 → j = n) →
  m + k = 649 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_980_divisors_l3310_331041
