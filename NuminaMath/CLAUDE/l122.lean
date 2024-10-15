import Mathlib

namespace NUMINAMATH_CALUDE_speeding_ticket_percentage_l122_12285

/-- The percentage of motorists who exceed the speed limit -/
def exceed_limit : ℝ := 16.666666666666664

/-- The percentage of speeding motorists who do not receive tickets -/
def no_ticket_rate : ℝ := 40

/-- The percentage of motorists who receive speeding tickets -/
def receive_ticket : ℝ := 10

theorem speeding_ticket_percentage :
  receive_ticket = exceed_limit * (1 - no_ticket_rate / 100) := by sorry

end NUMINAMATH_CALUDE_speeding_ticket_percentage_l122_12285


namespace NUMINAMATH_CALUDE_phone_contract_cost_l122_12240

/-- The total cost of buying a phone with a contract -/
def total_cost (phone_price : ℕ) (monthly_fee : ℕ) (contract_months : ℕ) : ℕ :=
  phone_price + monthly_fee * contract_months

/-- Theorem: The total cost of buying 1 phone with a 4-month contract is $30 -/
theorem phone_contract_cost :
  total_cost 2 7 4 = 30 := by
  sorry

end NUMINAMATH_CALUDE_phone_contract_cost_l122_12240


namespace NUMINAMATH_CALUDE_imaginary_unit_power_l122_12270

theorem imaginary_unit_power (i : ℂ) : i^2 = -1 → i^2015 = -i := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_power_l122_12270


namespace NUMINAMATH_CALUDE_multiply_powers_of_same_base_l122_12227

theorem multiply_powers_of_same_base (a b : ℝ) : 2 * a * b * b^2 = 2 * a * b^3 := by
  sorry

end NUMINAMATH_CALUDE_multiply_powers_of_same_base_l122_12227


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l122_12225

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 3) ↔ x ≥ 3 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l122_12225


namespace NUMINAMATH_CALUDE_apartment_tax_calculation_l122_12217

/-- Calculates the tax amount for an apartment --/
def calculate_tax (cadastral_value : ℝ) (tax_rate : ℝ) : ℝ :=
  cadastral_value * tax_rate

/-- Theorem: The tax amount for an apartment with a cadastral value of 3 million rubles
    and a tax rate of 0.1% is equal to 3000 rubles --/
theorem apartment_tax_calculation :
  let cadastral_value : ℝ := 3000000
  let tax_rate : ℝ := 0.001
  calculate_tax cadastral_value tax_rate = 3000 := by
sorry

/-- Additional information about the apartment (not used in the main calculation) --/
def apartment_area : ℝ := 70
def is_only_property : Prop := true

end NUMINAMATH_CALUDE_apartment_tax_calculation_l122_12217


namespace NUMINAMATH_CALUDE_range_of_x_no_solution_exists_l122_12269

-- Define the conditions
def conditions (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a + b = a * b

-- Theorem for the first part of the problem
theorem range_of_x (a b : ℝ) (h : conditions a b) :
  (∀ x : ℝ, |x| + |x - 2| ≤ a + b) ↔ ∀ x : ℝ, -1 ≤ x ∧ x ≤ 3 :=
sorry

-- Theorem for the second part of the problem
theorem no_solution_exists :
  ¬∃ a b : ℝ, conditions a b ∧ 4 * a + b = 8 :=
sorry

end NUMINAMATH_CALUDE_range_of_x_no_solution_exists_l122_12269


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l122_12271

theorem cube_volume_from_surface_area :
  ∀ (s : ℝ), s > 0 → 6 * s^2 = 54 → s^3 = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l122_12271


namespace NUMINAMATH_CALUDE_y_derivative_l122_12204

noncomputable def y (x : ℝ) : ℝ := (Real.sin x) / x + Real.sqrt x + 2

theorem y_derivative (x : ℝ) (h : x ≠ 0) : 
  deriv y x = (x * Real.cos x - Real.sin x) / x^2 + 1 / (2 * Real.sqrt x) :=
by sorry

end NUMINAMATH_CALUDE_y_derivative_l122_12204


namespace NUMINAMATH_CALUDE_ceiling_floor_square_zero_l122_12222

theorem ceiling_floor_square_zero : 
  (Int.ceil (7 / 3 : ℚ) + Int.floor (-7 / 3 : ℚ))^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_square_zero_l122_12222


namespace NUMINAMATH_CALUDE_max_profit_at_84_l122_12224

/-- The defect rate as a function of daily output --/
def defect_rate (x : ℕ) : ℚ :=
  if x ≤ 94 then 1 / (96 - x) else 2/3

/-- The daily profit as a function of daily output and profit per qualified instrument --/
def daily_profit (x : ℕ) (A : ℚ) : ℚ :=
  if x ≤ 94 
  then (x - 3*x / (2*(96 - x))) * A
  else 0

/-- Theorem: The daily profit is maximized when the daily output is 84 --/
theorem max_profit_at_84 (A : ℚ) (h : A > 0) :
  ∀ x : ℕ, x ≥ 1 → daily_profit 84 A ≥ daily_profit x A :=
by sorry

end NUMINAMATH_CALUDE_max_profit_at_84_l122_12224


namespace NUMINAMATH_CALUDE_largest_three_digit_multiple_of_7_with_digit_sum_21_l122_12279

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem largest_three_digit_multiple_of_7_with_digit_sum_21 :
  ∃ (n : ℕ), is_three_digit n ∧ n % 7 = 0 ∧ digit_sum n = 21 ∧
  ∀ (m : ℕ), is_three_digit m ∧ m % 7 = 0 ∧ digit_sum m = 21 → m ≤ n :=
by
  use 966
  sorry

end NUMINAMATH_CALUDE_largest_three_digit_multiple_of_7_with_digit_sum_21_l122_12279


namespace NUMINAMATH_CALUDE_sum_of_selected_numbers_l122_12253

def numbers : List ℚ := [14/10, 9/10, 12/10, 5/10, 13/10]
def threshold : ℚ := 11/10

theorem sum_of_selected_numbers :
  (numbers.filter (λ x => x ≥ threshold)).sum = 39/10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_selected_numbers_l122_12253


namespace NUMINAMATH_CALUDE_nell_initial_cards_l122_12201

/-- The number of cards Nell gave to John -/
def cards_to_john : ℕ := 195

/-- The number of cards Nell gave to Jeff -/
def cards_to_jeff : ℕ := 168

/-- The number of cards Nell has left -/
def cards_left : ℕ := 210

/-- The initial number of cards Nell had -/
def initial_cards : ℕ := cards_to_john + cards_to_jeff + cards_left

theorem nell_initial_cards : initial_cards = 573 := by
  sorry

end NUMINAMATH_CALUDE_nell_initial_cards_l122_12201


namespace NUMINAMATH_CALUDE_inequality_is_linear_one_var_l122_12262

/-- A linear inequality with one variable is an inequality of the form ax + b ≤ c or ax + b ≥ c,
    where a, b, and c are constants and x is a variable. -/
def is_linear_inequality_one_var (f : ℝ → Prop) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ 
  ((∀ x, f x ↔ a * x + b ≤ c) ∨ (∀ x, f x ↔ a * x + b ≥ c))

/-- The inequality 2 - x ≤ 4 -/
def inequality (x : ℝ) : Prop := 2 - x ≤ 4

theorem inequality_is_linear_one_var : is_linear_inequality_one_var inequality := by
  sorry

end NUMINAMATH_CALUDE_inequality_is_linear_one_var_l122_12262


namespace NUMINAMATH_CALUDE_remainder_3012_div_96_l122_12212

theorem remainder_3012_div_96 : 3012 % 96 = 36 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3012_div_96_l122_12212


namespace NUMINAMATH_CALUDE_train_length_proof_l122_12248

/-- Proves that the length of each train is 150 meters given the specified conditions -/
theorem train_length_proof (faster_speed slower_speed : ℝ) (passing_time : ℝ) : 
  faster_speed = 46 →
  slower_speed = 36 →
  passing_time = 108 →
  let relative_speed := (faster_speed - slower_speed) * (5 / 18)
  let train_length := relative_speed * passing_time / 2
  train_length = 150 := by
  sorry

#check train_length_proof

end NUMINAMATH_CALUDE_train_length_proof_l122_12248


namespace NUMINAMATH_CALUDE_f_properties_l122_12226

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x)

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧
  (∀ x y, x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l122_12226


namespace NUMINAMATH_CALUDE_reflection_theorem_l122_12282

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The line of symmetry for the fold -/
def lineOfSymmetry : ℝ := 2

/-- Function to reflect a point across the line of symmetry -/
def reflect (p : Point) : Point :=
  { x := p.x, y := 2 * lineOfSymmetry - p.y }

/-- The original point before folding -/
def originalPoint : Point := { x := -4, y := 1 }

/-- The expected point after folding -/
def expectedPoint : Point := { x := -4, y := 3 }

/-- Theorem stating that reflecting the original point results in the expected point -/
theorem reflection_theorem : reflect originalPoint = expectedPoint := by
  sorry

end NUMINAMATH_CALUDE_reflection_theorem_l122_12282


namespace NUMINAMATH_CALUDE_new_quadratic_equation_l122_12214

theorem new_quadratic_equation (α β : ℝ) : 
  (3 * α^2 + 7 * α + 4 = 0) → 
  (3 * β^2 + 7 * β + 4 = 0) → 
  (21 * (α / (β - 1))^2 - 23 * (α / (β - 1)) + 6 = 0) ∧
  (21 * (β / (α - 1))^2 - 23 * (β / (α - 1)) + 6 = 0) := by
sorry

end NUMINAMATH_CALUDE_new_quadratic_equation_l122_12214


namespace NUMINAMATH_CALUDE_kolya_purchase_options_l122_12206

-- Define the store's pricing rule
def item_price (rubles : ℕ) : ℕ := 100 * rubles + 99

-- Define Kolya's total purchase amount in kopecks
def total_purchase : ℕ := 20083

-- Define the possible number of items
def possible_items : Set ℕ := {17, 117}

-- Theorem statement
theorem kolya_purchase_options :
  ∀ n : ℕ, (∃ r : ℕ, n * item_price r = total_purchase) ↔ n ∈ possible_items :=
by sorry

end NUMINAMATH_CALUDE_kolya_purchase_options_l122_12206


namespace NUMINAMATH_CALUDE_unity_community_club_ratio_l122_12216

theorem unity_community_club_ratio :
  ∀ (f m c : ℕ),
  f > 0 → m > 0 → c > 0 →
  (35 * f + 30 * m + 10 * c) / (f + m + c) = 25 →
  ∃ (k : ℕ), k > 0 ∧ f = k ∧ m = k ∧ c = k :=
by sorry

end NUMINAMATH_CALUDE_unity_community_club_ratio_l122_12216


namespace NUMINAMATH_CALUDE_inequality_solution_set_l122_12256

theorem inequality_solution_set 
  (h : ∀ x : ℝ, x^2 - 2*a*x + a > 0) :
  {t : ℝ | a^(t^2 + 2*t - 3) < 1} = {t : ℝ | t < -3 ∨ t > 1} :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l122_12256


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l122_12243

/-- An isosceles triangle with side lengths 2 and 4 has perimeter 10 -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ), 
  a = 4 → b = 4 → c = 2 →  -- Two sides are 4, one side is 2
  a = b →  -- It's an isosceles triangle
  a + b + c = 10  -- The perimeter is 10
  := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l122_12243


namespace NUMINAMATH_CALUDE_complex_modulus_product_l122_12249

theorem complex_modulus_product : 
  Complex.abs ((7 - 4 * Complex.I) * (5 + 12 * Complex.I)) = 13 * Real.sqrt 65 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_product_l122_12249


namespace NUMINAMATH_CALUDE_triangle_inequality_from_sum_product_l122_12286

theorem triangle_inequality_from_sum_product (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  6 * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2) → 
  c < a + b ∧ a < b + c ∧ b < c + a := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_from_sum_product_l122_12286


namespace NUMINAMATH_CALUDE_corner_removed_cube_edges_l122_12281

/-- Represents a solid formed by removing smaller cubes from corners of a larger cube. -/
structure CornerRemovedCube where
  originalSideLength : ℝ
  removedSideLength : ℝ

/-- Calculates the number of edges in the resulting solid after corner removal. -/
def edgeCount (cube : CornerRemovedCube) : ℕ :=
  12 + 24  -- This is a placeholder. The actual calculation would be more complex.

/-- Theorem stating that a cube of side length 4 with corners of side length 2 removed has 36 edges. -/
theorem corner_removed_cube_edges :
  ∀ (cube : CornerRemovedCube),
    cube.originalSideLength = 4 →
    cube.removedSideLength = 2 →
    edgeCount cube = 36 := by
  sorry

#check corner_removed_cube_edges

end NUMINAMATH_CALUDE_corner_removed_cube_edges_l122_12281


namespace NUMINAMATH_CALUDE_cloth_trimming_l122_12265

theorem cloth_trimming (x : ℝ) :
  (x > 0) →
  (x - 4 > 0) →
  (x - 3 > 0) →
  ((x - 4) * (x - 3) = 120) →
  (x = 12) :=
by sorry

end NUMINAMATH_CALUDE_cloth_trimming_l122_12265


namespace NUMINAMATH_CALUDE_class_average_proof_l122_12203

theorem class_average_proof (group1_percent : Real) (group1_avg : Real)
                            (group2_percent : Real) (group2_avg : Real)
                            (group3_percent : Real) (group3_avg : Real)
                            (group4_percent : Real) (group4_avg : Real)
                            (group5_percent : Real) (group5_avg : Real)
                            (h1 : group1_percent = 0.25)
                            (h2 : group1_avg = 80)
                            (h3 : group2_percent = 0.35)
                            (h4 : group2_avg = 65)
                            (h5 : group3_percent = 0.20)
                            (h6 : group3_avg = 90)
                            (h7 : group4_percent = 0.10)
                            (h8 : group4_avg = 75)
                            (h9 : group5_percent = 0.10)
                            (h10 : group5_avg = 85)
                            (h11 : group1_percent + group2_percent + group3_percent + group4_percent + group5_percent = 1) :
  group1_percent * group1_avg + group2_percent * group2_avg + group3_percent * group3_avg +
  group4_percent * group4_avg + group5_percent * group5_avg = 76.75 := by
  sorry

#check class_average_proof

end NUMINAMATH_CALUDE_class_average_proof_l122_12203


namespace NUMINAMATH_CALUDE_staircase_cube_construction_l122_12205

/-- A staircase-brick with 3 steps of width 2, made of 12 unit cubes -/
structure StaircaseBrick where
  steps : Nat
  width : Nat
  volume : Nat
  steps_eq : steps = 3
  width_eq : width = 2
  volume_eq : volume = 12

/-- Predicate to check if a cube of side n can be built using staircase-bricks -/
def canBuildCube (n : Nat) : Prop :=
  ∃ (k : Nat), n^3 = k * 12

/-- Theorem stating that a cube of side n can be built using staircase-bricks
    if and only if n is a multiple of 12 -/
theorem staircase_cube_construction (n : Nat) :
  canBuildCube n ↔ ∃ (m : Nat), n = 12 * m :=
by sorry

end NUMINAMATH_CALUDE_staircase_cube_construction_l122_12205


namespace NUMINAMATH_CALUDE_man_walking_distance_l122_12219

theorem man_walking_distance (x t d : ℝ) : 
  (d = x * t) →                           -- distance = rate * time
  (d = (x + 1) * (3/4 * t)) →             -- faster speed condition
  (d = (x - 1) * (t + 3)) →               -- slower speed condition
  (d = 18) :=                             -- distance is 18 miles
by sorry

end NUMINAMATH_CALUDE_man_walking_distance_l122_12219


namespace NUMINAMATH_CALUDE_tangent_points_collinearity_l122_12228

-- Define the structure for a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the structure for a point
structure Point where
  coords : ℝ × ℝ

-- Define the property of three circles being pairwise non-intersecting
def pairwise_non_intersecting (c1 c2 c3 : Circle) : Prop :=
  sorry

-- Define the property of a point being on the internal tangent of two circles
def on_internal_tangent (p : Point) (c1 c2 : Circle) : Prop :=
  sorry

-- Define the property of a point being on the external tangent of two circles
def on_external_tangent (p : Point) (c1 c2 : Circle) : Prop :=
  sorry

-- Define the property of three points being collinear
def collinear (p1 p2 p3 : Point) : Prop :=
  sorry

-- Main theorem
theorem tangent_points_collinearity 
  (c1 c2 c3 : Circle)
  (A1 A2 A3 B1 B2 B3 : Point)
  (h_non_intersecting : pairwise_non_intersecting c1 c2 c3)
  (h_A1 : on_internal_tangent A1 c2 c3)
  (h_A2 : on_internal_tangent A2 c1 c3)
  (h_A3 : on_internal_tangent A3 c1 c2)
  (h_B1 : on_external_tangent B1 c2 c3)
  (h_B2 : on_external_tangent B2 c1 c3)
  (h_B3 : on_external_tangent B3 c1 c2) :
  (collinear A1 A2 B3) ∧ 
  (collinear A1 B2 A3) ∧ 
  (collinear B1 A2 A3) ∧ 
  (collinear B1 B2 B3) :=
sorry

end NUMINAMATH_CALUDE_tangent_points_collinearity_l122_12228


namespace NUMINAMATH_CALUDE_sally_found_thirteen_l122_12268

/-- The number of seashells Tim found -/
def tim_seashells : ℕ := 37

/-- The total number of seashells Tim and Sally found together -/
def total_seashells : ℕ := 50

/-- The number of seashells Sally found -/
def sally_seashells : ℕ := total_seashells - tim_seashells

theorem sally_found_thirteen : sally_seashells = 13 := by
  sorry

end NUMINAMATH_CALUDE_sally_found_thirteen_l122_12268


namespace NUMINAMATH_CALUDE_second_half_speed_l122_12242

/-- Proves that given a journey of 300 km completed in 11 hours, where the first half of the distance
    is traveled at 30 kmph, the speed for the second half of the journey is 25 kmph. -/
theorem second_half_speed 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (first_half_speed : ℝ) 
  (h1 : total_distance = 300)
  (h2 : total_time = 11)
  (h3 : first_half_speed = 30)
  : ∃ second_half_speed : ℝ, 
    second_half_speed = 25 ∧ 
    total_distance / 2 / first_half_speed + total_distance / 2 / second_half_speed = total_time :=
by sorry

end NUMINAMATH_CALUDE_second_half_speed_l122_12242


namespace NUMINAMATH_CALUDE_num_intersection_values_correct_l122_12293

/-- The number of different possible values for the count of intersection points
    formed by 5 distinct lines on a plane. -/
def num_intersection_values : ℕ := 9

/-- The set of possible values for the count of intersection points
    formed by 5 distinct lines on a plane. -/
def possible_intersection_values : Finset ℕ :=
  {0, 1, 4, 5, 6, 7, 8, 9, 10}

/-- Theorem stating that the number of different possible values for the count
    of intersection points formed by 5 distinct lines on a plane is correct. -/
theorem num_intersection_values_correct :
    num_intersection_values = Finset.card possible_intersection_values := by
  sorry

end NUMINAMATH_CALUDE_num_intersection_values_correct_l122_12293


namespace NUMINAMATH_CALUDE_f_range_implies_a_values_l122_12289

/-- The function f(x) defined as x^2 - 2ax + 2a + 4 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 2*a + 4

/-- The property that the range of f is [1, +∞) -/
def range_property (a : ℝ) : Prop :=
  ∀ x, f a x ≥ 1 ∧ ∀ y ≥ 1, ∃ x, f a x = y

/-- Theorem stating that the only values of a satisfying the conditions are -1 and 3 -/
theorem f_range_implies_a_values :
  ∀ a : ℝ, range_property a ↔ (a = -1 ∨ a = 3) :=
sorry

end NUMINAMATH_CALUDE_f_range_implies_a_values_l122_12289


namespace NUMINAMATH_CALUDE_find_a_l122_12294

theorem find_a (a x y : ℝ) (h1 : a^(3*x - 1) * 3^(4*y - 3) = 49^x * 27^y) (h2 : x + y = 4) : a = 7 := by
  sorry

end NUMINAMATH_CALUDE_find_a_l122_12294


namespace NUMINAMATH_CALUDE_second_divisor_l122_12210

theorem second_divisor (n : ℕ) : 
  (n ≠ 12 ∧ n ≠ 18 ∧ n ≠ 21 ∧ n ≠ 28) →
  (1008 % n = 0) →
  (∀ m : ℕ, m < n → m ≠ 12 → m ≠ 18 → m ≠ 21 → m ≠ 28 → 1008 % m ≠ 0) →
  n = 14 :=
by sorry

end NUMINAMATH_CALUDE_second_divisor_l122_12210


namespace NUMINAMATH_CALUDE_family_members_count_l122_12252

/-- Represents the number of family members -/
def n : ℕ := sorry

/-- The average age of family members in years -/
def average_age : ℕ := 29

/-- The present age of the youngest member in years -/
def youngest_age : ℕ := 5

/-- The average age of the remaining members at the time of birth of the youngest member in years -/
def average_age_at_birth : ℕ := 28

/-- The sum of ages of all family members -/
def sum_of_ages : ℕ := n * average_age

/-- The sum of ages of the remaining members at present -/
def sum_of_remaining_ages : ℕ := (n - 1) * (average_age_at_birth + youngest_age)

theorem family_members_count :
  sum_of_ages = sum_of_remaining_ages + youngest_age → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_family_members_count_l122_12252


namespace NUMINAMATH_CALUDE_line_l_and_symmetrical_line_l122_12236

-- Define the lines
def line1 (x y : ℝ) : Prop := 3 * x + 4 * y - 2 = 0
def line2 (x y : ℝ) : Prop := 2 * x + y + 2 = 0
def line3 (x y : ℝ) : Prop := x - 2 * y - 1 = 0

-- Define the intersection point P
def P : ℝ × ℝ := (-2, 2)

-- Define line l
def l (x y : ℝ) : Prop := 2 * x + y + 2 = 0

-- Define the symmetrical line
def symmetrical_l (x y : ℝ) : Prop := 2 * x + y - 2 = 0

theorem line_l_and_symmetrical_line : 
  (∀ x y : ℝ, line1 x y ∧ line2 x y → (x, y) = P) → 
  (∀ x y : ℝ, l x y → line3 (y + 2) (-x - 1)) →
  (∀ x y : ℝ, l x y ↔ 2 * x + y + 2 = 0) ∧
  (∀ x y : ℝ, symmetrical_l x y ↔ 2 * x + y - 2 = 0) := by sorry

end NUMINAMATH_CALUDE_line_l_and_symmetrical_line_l122_12236


namespace NUMINAMATH_CALUDE_product_of_proper_fractions_sum_of_proper_and_improper_l122_12211

-- Define a fraction as a pair of integers where the denominator is non-zero
def Fraction := { p : ℚ // p > 0 }

-- Define a proper fraction
def isProper (f : Fraction) : Prop := f.val < 1

-- Define an improper fraction
def isImproper (f : Fraction) : Prop := f.val ≥ 1

-- Statement 2
theorem product_of_proper_fractions (f g : Fraction) 
  (hf : isProper f) (hg : isProper g) : 
  isProper ⟨f.val * g.val, by sorry⟩ := by sorry

-- Statement 3
theorem sum_of_proper_and_improper (f g : Fraction) 
  (hf : isProper f) (hg : isImproper g) : 
  isImproper ⟨f.val + g.val, by sorry⟩ := by sorry

end NUMINAMATH_CALUDE_product_of_proper_fractions_sum_of_proper_and_improper_l122_12211


namespace NUMINAMATH_CALUDE_leaves_collected_first_day_l122_12296

/-- Represents the number of leaves collected by Bronson -/
def total_leaves : ℕ := 25

/-- Represents the number of leaves collected on the second day -/
def second_day_leaves : ℕ := 13

/-- Represents the percentage of brown leaves -/
def brown_percent : ℚ := 1/5

/-- Represents the percentage of green leaves -/
def green_percent : ℚ := 1/5

/-- Represents the number of yellow leaves -/
def yellow_leaves : ℕ := 15

/-- Theorem stating the number of leaves collected on the first day -/
theorem leaves_collected_first_day : 
  total_leaves - second_day_leaves = 12 :=
sorry

end NUMINAMATH_CALUDE_leaves_collected_first_day_l122_12296


namespace NUMINAMATH_CALUDE_order_total_parts_l122_12231

theorem order_total_parts (total_cost : ℕ) (cost_cheap : ℕ) (cost_expensive : ℕ) (num_expensive : ℕ) :
  total_cost = 2380 →
  cost_cheap = 20 →
  cost_expensive = 50 →
  num_expensive = 40 →
  ∃ (num_cheap : ℕ), num_cheap * cost_cheap + num_expensive * cost_expensive = total_cost ∧
                      num_cheap + num_expensive = 59 :=
by sorry

end NUMINAMATH_CALUDE_order_total_parts_l122_12231


namespace NUMINAMATH_CALUDE_tetris_single_ratio_is_eight_to_one_l122_12267

/-- The ratio of points for a tetris to points for a single line -/
def tetris_to_single_ratio (single_points tetris_points : ℕ) : ℚ :=
  tetris_points / single_points

/-- The total score given the number of singles, number of tetrises, points for a single, and points for a tetris -/
def total_score (num_singles num_tetrises single_points tetris_points : ℕ) : ℕ :=
  num_singles * single_points + num_tetrises * tetris_points

theorem tetris_single_ratio_is_eight_to_one :
  ∃ (tetris_points : ℕ),
    single_points = 1000 ∧
    num_singles = 6 ∧
    num_tetrises = 4 ∧
    total_score num_singles num_tetrises single_points tetris_points = 38000 ∧
    tetris_to_single_ratio single_points tetris_points = 8 := by
  sorry

end NUMINAMATH_CALUDE_tetris_single_ratio_is_eight_to_one_l122_12267


namespace NUMINAMATH_CALUDE_custom_op_two_five_l122_12288

/-- Custom binary operation on real numbers -/
def custom_op (a b : ℝ) : ℝ := 4 * a + 3 * b

theorem custom_op_two_five : custom_op 2 5 = 23 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_two_five_l122_12288


namespace NUMINAMATH_CALUDE_min_omega_for_overlapping_sine_graphs_l122_12255

/-- Given a function f(x) = sin(ωx + π/3) where ω > 0, if the graph of y = f(x) is shifted
    to the right by 2π/3 units and overlaps with the original graph, then the minimum
    value of ω is 3. -/
theorem min_omega_for_overlapping_sine_graphs (ω : ℝ) (f : ℝ → ℝ) :
  ω > 0 →
  (∀ x, f x = Real.sin (ω * x + π / 3)) →
  (∀ x, f (x + 2 * π / 3) = f x) →
  3 ≤ ω ∧ ∀ ω', (ω' > 0 ∧ ∀ x, f (x + 2 * π / 3) = f x) → ω ≤ ω' :=
by sorry

end NUMINAMATH_CALUDE_min_omega_for_overlapping_sine_graphs_l122_12255


namespace NUMINAMATH_CALUDE_sum_of_proportions_l122_12276

theorem sum_of_proportions (a b c d e f : ℝ) 
  (h1 : a / b = 2) 
  (h2 : c / d = 2) 
  (h3 : e / f = 2) 
  (h4 : b + d + f = 4) : 
  a + c + e = 8 := by
sorry

end NUMINAMATH_CALUDE_sum_of_proportions_l122_12276


namespace NUMINAMATH_CALUDE_omega_range_l122_12237

theorem omega_range (f : ℝ → ℝ) (ω : ℝ) :
  (∀ x, f x = Real.cos (ω * x + π / 6)) →
  ω > 0 →
  (∀ x ∈ Set.Icc 0 π, f x ∈ Set.Icc (-1) (Real.sqrt 3 / 2)) →
  ω ∈ Set.Icc (5 / 6) (5 / 3) :=
sorry

end NUMINAMATH_CALUDE_omega_range_l122_12237


namespace NUMINAMATH_CALUDE_min_value_expression_l122_12284

theorem min_value_expression (a b : ℝ) (h1 : b = 1 + a) (h2 : 0 < b) (h3 : b < 1) :
  ∀ x y : ℝ, x = 1 + y → 0 < x → x < 1 → 
    (2023 / b - (a + 1) / (2023 * a)) ≤ (2023 / x - (y + 1) / (2023 * y)) →
    2023 / b - (a + 1) / (2023 * a) ≥ 2025 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l122_12284


namespace NUMINAMATH_CALUDE_largest_factor_of_consecutive_product_l122_12261

theorem largest_factor_of_consecutive_product (n : ℕ) : 
  n % 10 = 4 → 120 ∣ n * (n + 1) * (n + 2) ∧ 
  ∀ m : ℕ, m > 120 → ∃ k : ℕ, k % 10 = 4 ∧ ¬(m ∣ k * (k + 1) * (k + 2)) := by
  sorry

end NUMINAMATH_CALUDE_largest_factor_of_consecutive_product_l122_12261


namespace NUMINAMATH_CALUDE_residue_11_1201_mod_19_l122_12251

theorem residue_11_1201_mod_19 :
  (11 : ℤ) ^ 1201 ≡ 1 [ZMOD 19] := by sorry

end NUMINAMATH_CALUDE_residue_11_1201_mod_19_l122_12251


namespace NUMINAMATH_CALUDE_functional_equation_solution_l122_12235

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x + y) + f (x - y) = x^2 + y^2) →
  (∀ x : ℝ, f x = x^2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l122_12235


namespace NUMINAMATH_CALUDE_winnie_keeps_lollipops_l122_12245

/-- The number of cherry lollipops Winnie has -/
def cherry : ℕ := 45

/-- The number of wintergreen lollipops Winnie has -/
def wintergreen : ℕ := 116

/-- The number of grape lollipops Winnie has -/
def grape : ℕ := 4

/-- The number of shrimp cocktail lollipops Winnie has -/
def shrimp : ℕ := 229

/-- The number of Winnie's friends -/
def friends : ℕ := 11

/-- The total number of lollipops Winnie has -/
def total : ℕ := cherry + wintergreen + grape + shrimp

/-- The theorem stating how many lollipops Winnie keeps for herself -/
theorem winnie_keeps_lollipops : total % friends = 9 := by
  sorry

end NUMINAMATH_CALUDE_winnie_keeps_lollipops_l122_12245


namespace NUMINAMATH_CALUDE_basketball_score_proof_l122_12254

/-- 
Given a basketball team's scoring pattern:
- 4 games with 10t points each
- g games with 20 points each
- Average score of 28 points per game
Prove that g = 16
-/
theorem basketball_score_proof (t : ℕ) (g : ℕ) : 
  (40 * t + 20 * g) / (4 + g) = 28 → g = 16 := by
  sorry

end NUMINAMATH_CALUDE_basketball_score_proof_l122_12254


namespace NUMINAMATH_CALUDE_bedroom_paint_area_l122_12200

/-- Calculates the total paintable area in multiple identical bedrooms -/
def total_paintable_area (
  num_bedrooms : ℕ
  ) (length width height : ℝ
  ) (unpaintable_area : ℝ
  ) : ℝ :=
  num_bedrooms * (2 * (length * height + width * height) - unpaintable_area)

/-- Proves that the total paintable area in the given conditions is 1288 square feet -/
theorem bedroom_paint_area :
  total_paintable_area 4 10 12 9 74 = 1288 := by
  sorry

end NUMINAMATH_CALUDE_bedroom_paint_area_l122_12200


namespace NUMINAMATH_CALUDE_sam_initial_yellow_marbles_l122_12291

/-- The number of yellow marbles Sam had initially -/
def initial_yellow_marbles : ℕ := sorry

/-- The number of yellow marbles Joan took -/
def marbles_taken : ℕ := 25

/-- The number of yellow marbles Sam has now -/
def current_yellow_marbles : ℕ := 61

theorem sam_initial_yellow_marbles :
  initial_yellow_marbles = current_yellow_marbles + marbles_taken :=
by sorry

end NUMINAMATH_CALUDE_sam_initial_yellow_marbles_l122_12291


namespace NUMINAMATH_CALUDE_annas_size_l122_12272

theorem annas_size (anna_size : ℕ) 
  (becky_size : ℕ) (ginger_size : ℕ) 
  (h1 : becky_size = 3 * anna_size)
  (h2 : ginger_size = 2 * becky_size - 4)
  (h3 : ginger_size = 8) : 
  anna_size = 2 := by
  sorry

end NUMINAMATH_CALUDE_annas_size_l122_12272


namespace NUMINAMATH_CALUDE_car_speed_problem_l122_12295

theorem car_speed_problem (highway_length : ℝ) (meeting_time : ℝ) (car2_speed : ℝ) : 
  highway_length = 333 ∧ 
  meeting_time = 3 ∧ 
  car2_speed = 57 →
  ∃ car1_speed : ℝ, 
    car1_speed * meeting_time + car2_speed * meeting_time = highway_length ∧ 
    car1_speed = 54 :=
by sorry

end NUMINAMATH_CALUDE_car_speed_problem_l122_12295


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l122_12221

theorem absolute_value_equation_solution (x z : ℝ) :
  |5 * x - Real.log z| = 5 * x + 3 * Real.log z →
  x = 0 ∧ z = 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l122_12221


namespace NUMINAMATH_CALUDE_max_even_differences_l122_12263

/-- A permutation of numbers from 1 to 25 -/
def Arrangement := Fin 25 → Fin 25

/-- The sequence 1, 2, 3, ..., 25 -/
def OriginalSequence : Fin 25 → ℕ := fun i => i.val + 1

/-- The difference function, always subtracting the smaller from the larger -/
def Difference (arr : Arrangement) (i : Fin 25) : ℕ :=
  max (OriginalSequence i) (arr i).val + 1 - min (OriginalSequence i) (arr i).val + 1

/-- Predicate to check if a number is even -/
def IsEven (n : ℕ) : Prop := ∃ k, n = 2 * k

theorem max_even_differences :
  ∃ (arr : Arrangement), ∀ (i : Fin 25), IsEven (Difference arr i) :=
sorry

end NUMINAMATH_CALUDE_max_even_differences_l122_12263


namespace NUMINAMATH_CALUDE_sqrt_x_minus_2_real_l122_12223

theorem sqrt_x_minus_2_real (x : ℝ) : (∃ y : ℝ, y^2 = x - 2) ↔ x ≥ 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_2_real_l122_12223


namespace NUMINAMATH_CALUDE_triangle_area_l122_12298

theorem triangle_area (a b : ℝ) (C : ℝ) (h1 : a = 3 * Real.sqrt 2) (h2 : b = 2 * Real.sqrt 3) (h3 : Real.cos C = 1/3) :
  (1/2) * a * b * Real.sin C = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l122_12298


namespace NUMINAMATH_CALUDE_right_triangle_altitude_l122_12266

theorem right_triangle_altitude (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^2 + b^2 = c^2) (h5 : 1/a + 1/b = 3/c) :
  ∃ m_c : ℝ, m_c = c * (1 + Real.sqrt 10) / 9 ∧ m_c^2 * c = a * b := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_altitude_l122_12266


namespace NUMINAMATH_CALUDE_orange_delivery_problem_l122_12250

def bag_weights : List ℕ := [22, 25, 28, 31, 34, 36, 38, 40, 45]

def total_weight : ℕ := bag_weights.sum

theorem orange_delivery_problem (weights_A B : ℕ) (weight_C : ℕ) :
  weights_A = 2 * weights_B →
  weights_A + weights_B + weight_C = total_weight →
  weight_C ∈ bag_weights →
  weight_C % 3 = 2 →
  weight_C = 38 := by
  sorry

end NUMINAMATH_CALUDE_orange_delivery_problem_l122_12250


namespace NUMINAMATH_CALUDE_vertical_strips_count_l122_12241

/-- Represents a grid rectangle with a hole -/
structure GridRectangleWithHole where
  outer_perimeter : ℕ
  hole_perimeter : ℕ
  horizontal_strips : ℕ

/-- Theorem: Given a grid rectangle with a hole, if cutting horizontally yields 20 strips,
    then cutting vertically yields 21 strips -/
theorem vertical_strips_count
  (rect : GridRectangleWithHole)
  (h_outer : rect.outer_perimeter = 50)
  (h_hole : rect.hole_perimeter = 32)
  (h_horizontal : rect.horizontal_strips = 20) :
  ∃ (vertical_strips : ℕ), vertical_strips = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_vertical_strips_count_l122_12241


namespace NUMINAMATH_CALUDE_number_line_mark_distance_not_always_1cm_l122_12277

/-- Represents a number line -/
structure NumberLine where
  origin : ℝ
  positive_direction : Bool
  unit_length : ℝ

/-- Properties of a number line -/
def valid_number_line (nl : NumberLine) : Prop :=
  nl.unit_length > 0

theorem number_line_mark_distance_not_always_1cm 
  (h1 : ∀ (x : ℝ), ∃ (nl : NumberLine), nl.origin = x ∧ valid_number_line nl)
  (h2 : ∀ (nl : NumberLine), valid_number_line nl → nl.positive_direction = true)
  (h3 : ∀ (l : ℝ), l > 0 → ∃ (nl : NumberLine), nl.unit_length = l ∧ valid_number_line nl) :
  ¬(∀ (nl : NumberLine), valid_number_line nl → nl.unit_length = 1) :=
sorry

end NUMINAMATH_CALUDE_number_line_mark_distance_not_always_1cm_l122_12277


namespace NUMINAMATH_CALUDE_perpendicular_vector_l122_12213

theorem perpendicular_vector (a b : ℝ × ℝ) : 
  a = (Real.sqrt 3, Real.sqrt 5) →
  (a.1 * b.1 + a.2 * b.2 = 0) →
  (b.1^2 + b.2^2 = 4) →
  (b = (-Real.sqrt (10) / 2, Real.sqrt 6 / 2) ∨ 
   b = (Real.sqrt (10) / 2, -Real.sqrt 6 / 2)) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_vector_l122_12213


namespace NUMINAMATH_CALUDE_least_eight_binary_digits_l122_12260

/-- The number of binary digits required to represent a positive integer -/
def binaryDigits (n : ℕ+) : ℕ :=
  (Nat.log2 n.val) + 1

/-- Theorem: 128 is the least positive integer that requires 8 binary digits -/
theorem least_eight_binary_digits :
  (∀ m : ℕ+, m < 128 → binaryDigits m < 8) ∧ binaryDigits 128 = 8 := by
  sorry

end NUMINAMATH_CALUDE_least_eight_binary_digits_l122_12260


namespace NUMINAMATH_CALUDE_quadrilateral_theorem_l122_12220

structure Quadrilateral :=
  (C D X W P : ℝ × ℝ)
  (CD_parallel_WX : (D.1 - C.1) * (X.2 - W.2) = (D.2 - C.2) * (X.1 - W.1))
  (P_on_CW : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (C.1 + t * (W.1 - C.1), C.2 + t * (W.2 - C.2)))
  (CW_length : Real.sqrt ((W.1 - C.1)^2 + (W.2 - C.2)^2) = 56)
  (DP_length : Real.sqrt ((P.1 - D.1)^2 + (P.2 - D.2)^2) = 16)
  (PX_length : Real.sqrt ((X.1 - P.1)^2 + (X.2 - P.2)^2) = 32)

theorem quadrilateral_theorem (q : Quadrilateral) :
  Real.sqrt ((q.W.1 - q.P.1)^2 + (q.W.2 - q.P.2)^2) = 112/3 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_theorem_l122_12220


namespace NUMINAMATH_CALUDE_caterpillar_length_difference_l122_12280

/-- The length difference between two caterpillars -/
theorem caterpillar_length_difference : 
  let green_length : ℝ := 3
  let orange_length : ℝ := 1.17
  green_length - orange_length = 1.83 := by sorry

end NUMINAMATH_CALUDE_caterpillar_length_difference_l122_12280


namespace NUMINAMATH_CALUDE_dave_initial_apps_l122_12247

/-- Represents the number of apps on Dave's phone at different stages -/
structure AppCount where
  initial : ℕ
  afterAdding : ℕ
  afterDeleting : ℕ
  final : ℕ

/-- Represents the number of apps added and deleted -/
structure AppChanges where
  added : ℕ
  deleted : ℕ

/-- The theorem stating Dave's initial app count based on the given conditions -/
theorem dave_initial_apps (ac : AppCount) (ch : AppChanges) : 
  ch.added = 89 ∧ 
  ac.afterDeleting = 24 ∧ 
  ch.added = ch.deleted + 3 ∧
  ac.afterAdding = ac.initial + ch.added ∧
  ac.afterDeleting = ac.afterAdding - ch.deleted ∧
  ac.final = ac.afterDeleting + (ch.added - ch.deleted) →
  ac.initial = 21 := by
  sorry


end NUMINAMATH_CALUDE_dave_initial_apps_l122_12247


namespace NUMINAMATH_CALUDE_prob_10_or_7_prob_below_7_l122_12274

/-- Probability of hitting the 10 ring -/
def P10 : ℝ := 0.21

/-- Probability of hitting the 9 ring -/
def P9 : ℝ := 0.23

/-- Probability of hitting the 8 ring -/
def P8 : ℝ := 0.25

/-- Probability of hitting the 7 ring -/
def P7 : ℝ := 0.28

/-- The probability of hitting either the 10 or 7 ring is 0.49 -/
theorem prob_10_or_7 : P10 + P7 = 0.49 := by sorry

/-- The probability of scoring below 7 rings is 0.03 -/
theorem prob_below_7 : 1 - (P10 + P9 + P8 + P7) = 0.03 := by sorry

end NUMINAMATH_CALUDE_prob_10_or_7_prob_below_7_l122_12274


namespace NUMINAMATH_CALUDE_zeros_not_adjacent_probability_l122_12292

/-- The number of ones in the arrangement -/
def num_ones : ℕ := 4

/-- The number of zeros in the arrangement -/
def num_zeros : ℕ := 2

/-- The total number of elements to be arranged -/
def total_elements : ℕ := num_ones + num_zeros

/-- The number of spaces where zeros can be placed without being adjacent -/
def num_spaces : ℕ := num_ones + 1

/-- The probability that the zeros are not adjacent when randomly arranged -/
theorem zeros_not_adjacent_probability :
  (Nat.choose num_spaces num_zeros : ℚ) / (Nat.choose total_elements num_zeros : ℚ) = 2/3 :=
sorry

end NUMINAMATH_CALUDE_zeros_not_adjacent_probability_l122_12292


namespace NUMINAMATH_CALUDE_constant_sum_l122_12239

theorem constant_sum (x y : ℝ) (h : x + y = 4) : 5 * x + 5 * y = 20 := by
  sorry

end NUMINAMATH_CALUDE_constant_sum_l122_12239


namespace NUMINAMATH_CALUDE_prime_divisibility_problem_l122_12209

theorem prime_divisibility_problem (p : ℕ) (x : ℕ) (hp : Prime p) :
  (1 ≤ x ∧ x ≤ 2 * p) →
  (x^(p - 1) ∣ (p - 1)^x + 1) →
  ((p = 2 ∧ (x = 1 ∨ x = 2)) ∨ (p = 3 ∧ (x = 1 ∨ x = 3)) ∨ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_prime_divisibility_problem_l122_12209


namespace NUMINAMATH_CALUDE_cricketer_average_score_l122_12208

theorem cricketer_average_score 
  (total_matches : ℕ) 
  (first_set_matches : ℕ) 
  (second_set_matches : ℕ) 
  (first_set_average : ℝ) 
  (second_set_average : ℝ) 
  (h1 : total_matches = first_set_matches + second_set_matches)
  (h2 : total_matches = 5)
  (h3 : first_set_matches = 2)
  (h4 : second_set_matches = 3)
  (h5 : first_set_average = 60)
  (h6 : second_set_average = 50) :
  (first_set_matches * first_set_average + second_set_matches * second_set_average) / total_matches = 54 := by
  sorry

end NUMINAMATH_CALUDE_cricketer_average_score_l122_12208


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l122_12244

theorem quadratic_equation_roots (p q : ℝ) : 
  (∃ α β : ℝ, α ≠ β ∧ 
   α^2 + p*α + q = 0 ∧ 
   β^2 + p*β + q = 0 ∧ 
   ({α, β} : Set ℝ) ⊆ {1, 2, 3, 4} ∧ 
   ({α, β} : Set ℝ) ∩ {2, 4, 5, 6} = ∅) →
  p = -4 ∧ q = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l122_12244


namespace NUMINAMATH_CALUDE_combination_permutation_problem_l122_12287

-- Define the combination function
def C (n k : ℕ) : ℕ := n.choose k

-- Define the permutation function
def A (n k : ℕ) : ℕ := n.factorial / (n - k).factorial

-- State the theorem
theorem combination_permutation_problem (n : ℕ) :
  C n 2 * A 2 2 = 42 → n.factorial / (3 * (n - 3).factorial) = 35 := by
  sorry

end NUMINAMATH_CALUDE_combination_permutation_problem_l122_12287


namespace NUMINAMATH_CALUDE_problem_solution_l122_12299

theorem problem_solution (a b : ℝ) (ha : a = 2 + Real.sqrt 3) (hb : b = 2 - Real.sqrt 3) :
  (a - b = 2 * Real.sqrt 3) ∧ (a * b = 1) ∧ (a^2 + b^2 - 5*a*b = 9) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l122_12299


namespace NUMINAMATH_CALUDE_complement_union_theorem_l122_12218

-- Define the universal set U
def U : Set Nat := {0, 1, 2, 3, 4}

-- Define set A
def A : Set Nat := {0, 1, 2}

-- Define set B
def B : Set Nat := {2, 3}

-- Theorem statement
theorem complement_union_theorem :
  (U \ A) ∪ B = {2, 3, 4} := by
  sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l122_12218


namespace NUMINAMATH_CALUDE_inequality_proof_l122_12257

theorem inequality_proof (a b c : ℝ) : 
  1 < (a / Real.sqrt (a^2 + b^2)) + (b / Real.sqrt (b^2 + c^2)) + (c / Real.sqrt (c^2 + a^2)) ∧
  (a / Real.sqrt (a^2 + b^2)) + (b / Real.sqrt (b^2 + c^2)) + (c / Real.sqrt (c^2 + a^2)) ≤ (3 * Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l122_12257


namespace NUMINAMATH_CALUDE_coefficient_x5y2_is_90_l122_12259

/-- The coefficient of x^5y^2 in the expansion of (x^2 + 3x - y)^5 -/
def coefficient_x5y2 : ℕ :=
  let n : ℕ := 5
  let k : ℕ := 3
  let binomial_coeff : ℕ := n.choose k
  let x_coeff : ℕ := 9  -- Coefficient of x^5 in (x^2 + 3x)^3
  binomial_coeff * x_coeff

/-- The coefficient of x^5y^2 in the expansion of (x^2 + 3x - y)^5 is 90 -/
theorem coefficient_x5y2_is_90 : coefficient_x5y2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x5y2_is_90_l122_12259


namespace NUMINAMATH_CALUDE_banana_arrangements_l122_12290

theorem banana_arrangements : 
  let total_letters : ℕ := 6
  let repeated_letter1_count : ℕ := 3
  let repeated_letter2_count : ℕ := 2
  let unique_letter_count : ℕ := 1
  (total_letters = repeated_letter1_count + repeated_letter2_count + unique_letter_count) →
  (Nat.factorial total_letters / (Nat.factorial repeated_letter1_count * Nat.factorial repeated_letter2_count) = 60) := by
  sorry

end NUMINAMATH_CALUDE_banana_arrangements_l122_12290


namespace NUMINAMATH_CALUDE_calculation_proof_l122_12278

theorem calculation_proof : (((2207 - 2024) ^ 2 * 4) : ℚ) / 144 = 930.25 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l122_12278


namespace NUMINAMATH_CALUDE_equalize_guppies_l122_12207

/-- Represents a fish tank -/
structure Tank where
  guppies : ℕ
  swordtails : ℕ
  angelfish : ℕ

/-- The problem setup -/
def problem : Prop :=
  let tankA : Tank := { guppies := 180, swordtails := 32, angelfish := 0 }
  let tankB : Tank := { guppies := 120, swordtails := 45, angelfish := 15 }
  let tankC : Tank := { guppies := 80, swordtails := 15, angelfish := 33 }
  let totalFish : ℕ := tankA.guppies + tankA.swordtails + tankA.angelfish +
                       tankB.guppies + tankB.swordtails + tankB.angelfish +
                       tankC.guppies + tankC.swordtails + tankC.angelfish

  totalFish = 520 ∧
  (tankA.guppies + tankB.guppies - (tankA.guppies + tankB.guppies) / 2) = 30

theorem equalize_guppies (tankA tankB : Tank) :
  tankA.guppies = 180 →
  tankB.guppies = 120 →
  (tankA.guppies + tankB.guppies - (tankA.guppies + tankB.guppies) / 2) = 30 :=
by sorry

end NUMINAMATH_CALUDE_equalize_guppies_l122_12207


namespace NUMINAMATH_CALUDE_divisibility_in_ones_sequence_l122_12238

theorem divisibility_in_ones_sequence (k : ℕ) (hprime : Nat.Prime k) (h2 : k ≠ 2) (h5 : k ≠ 5) :
  ∃ n : ℕ, n ≤ k ∧ k ∣ ((10^n - 1) / 9) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_in_ones_sequence_l122_12238


namespace NUMINAMATH_CALUDE_mark_and_carolyn_money_sum_l122_12232

theorem mark_and_carolyn_money_sum : (5 : ℚ) / 8 + (7 : ℚ) / 20 = 0.975 := by
  sorry

end NUMINAMATH_CALUDE_mark_and_carolyn_money_sum_l122_12232


namespace NUMINAMATH_CALUDE_skew_parameter_calculation_l122_12233

/-- Dilation matrix -/
def D (k : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![k, 0; 0, k]

/-- Skew transformation matrix -/
def S (a : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![1, a; 0, 1]

/-- The problem statement -/
theorem skew_parameter_calculation (k : ℝ) (a : ℝ) (h1 : k > 0) :
  S a * D k = !![10, 5; 0, 10] →
  a = 1/2 := by
sorry

end NUMINAMATH_CALUDE_skew_parameter_calculation_l122_12233


namespace NUMINAMATH_CALUDE_compare_negative_fractions_l122_12273

theorem compare_negative_fractions : -2/3 > -3/4 := by sorry

end NUMINAMATH_CALUDE_compare_negative_fractions_l122_12273


namespace NUMINAMATH_CALUDE_car_speed_second_hour_l122_12264

/-- Given a car's speed over two hours, prove its speed in the second hour. -/
theorem car_speed_second_hour
  (speed_first_hour : ℝ)
  (average_speed : ℝ)
  (total_time : ℝ)
  (h1 : speed_first_hour = 50)
  (h2 : average_speed = 55)
  (h3 : total_time = 2)
  : ∃ (speed_second_hour : ℝ), speed_second_hour = 60 :=
by
  sorry

#check car_speed_second_hour

end NUMINAMATH_CALUDE_car_speed_second_hour_l122_12264


namespace NUMINAMATH_CALUDE_arithmetic_sequence_1001st_term_l122_12229

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  p : ℚ
  q : ℚ
  first_term : ℚ
  second_term : ℚ
  third_term : ℚ
  fourth_term : ℚ
  is_arithmetic : ∃ (d : ℚ), second_term = first_term + d ∧
                              third_term = second_term + d ∧
                              fourth_term = third_term + d
  first_is_p : first_term = p
  second_is_12 : second_term = 12
  third_is_3p_minus_q : third_term = 3 * p - q
  fourth_is_3p_plus_2q : fourth_term = 3 * p + 2 * q

/-- The 1001st term of the sequence is 5545 -/
theorem arithmetic_sequence_1001st_term (seq : ArithmeticSequence) : 
  seq.first_term + 1000 * (seq.second_term - seq.first_term) = 5545 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_1001st_term_l122_12229


namespace NUMINAMATH_CALUDE_committee_selection_count_club_committee_count_l122_12234

theorem committee_selection_count : Nat → Nat → Nat
  | n, k => (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem club_committee_count :
  committee_selection_count 30 5 = 142506 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_count_club_committee_count_l122_12234


namespace NUMINAMATH_CALUDE_mike_five_dollar_bills_l122_12202

theorem mike_five_dollar_bills (total_amount : ℕ) (bill_denomination : ℕ) (h1 : total_amount = 45) (h2 : bill_denomination = 5) :
  total_amount / bill_denomination = 9 := by
  sorry

end NUMINAMATH_CALUDE_mike_five_dollar_bills_l122_12202


namespace NUMINAMATH_CALUDE_tangent_line_at_x_1_l122_12283

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - x + 3

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

-- Theorem stating that the tangent line equation is correct
theorem tangent_line_at_x_1 : 
  ∀ x y : ℝ, (y = f 1 + f' 1 * (x - 1)) ↔ (2*x - y + 1 = 0) :=
by sorry

-- Note: The proof is omitted as per the instructions

end NUMINAMATH_CALUDE_tangent_line_at_x_1_l122_12283


namespace NUMINAMATH_CALUDE_input_is_input_statement_l122_12258

-- Define an enumeration for different types of statements
inductive StatementType
  | Print
  | Input
  | If
  | End

-- Define a function to classify statements
def classifyStatement (s : StatementType) : String :=
  match s with
  | StatementType.Print => "output"
  | StatementType.Input => "input"
  | StatementType.If => "conditional"
  | StatementType.End => "end"

-- Theorem to prove
theorem input_is_input_statement :
  classifyStatement StatementType.Input = "input" := by
  sorry

end NUMINAMATH_CALUDE_input_is_input_statement_l122_12258


namespace NUMINAMATH_CALUDE_system_solution_l122_12230

theorem system_solution (x y z : ℝ) 
  (eq1 : y + z = 17 - 2*x)
  (eq2 : x + z = 1 - 2*y)
  (eq3 : x + y = 8 - 2*z) :
  x + y + z = 6.5 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l122_12230


namespace NUMINAMATH_CALUDE_removing_2013th_digit_increases_one_seventh_l122_12297

-- Define the decimal representation of 1/7
def one_seventh_decimal : ℚ := 1 / 7

-- Define the period of the repeating decimal
def period : ℕ := 6

-- Define the position of the digit to be removed
def removed_digit_position : ℕ := 2013

-- Define the function that removes the nth digit after the decimal point
def remove_nth_digit (q : ℚ) (n : ℕ) : ℚ := sorry

-- Theorem statement
theorem removing_2013th_digit_increases_one_seventh :
  remove_nth_digit one_seventh_decimal removed_digit_position > one_seventh_decimal := by
  sorry

end NUMINAMATH_CALUDE_removing_2013th_digit_increases_one_seventh_l122_12297


namespace NUMINAMATH_CALUDE_total_fish_caught_l122_12246

def blaine_fish : ℕ := 5

def keith_fish (blaine : ℕ) : ℕ := 2 * blaine

theorem total_fish_caught : blaine_fish + keith_fish blaine_fish = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_fish_caught_l122_12246


namespace NUMINAMATH_CALUDE_chocolate_count_correct_l122_12275

/-- The number of small boxes in the large box -/
def total_small_boxes : ℕ := 17

/-- The number of small boxes containing medium boxes -/
def boxes_with_medium : ℕ := 10

/-- The number of medium boxes in each of the first 10 small boxes -/
def medium_boxes_per_small : ℕ := 4

/-- The number of chocolate bars in each medium box -/
def chocolates_per_medium : ℕ := 26

/-- The number of chocolate bars in each of the first two of the remaining small boxes -/
def chocolates_in_first_two : ℕ := 18

/-- The number of chocolate bars in each of the next three of the remaining small boxes -/
def chocolates_in_next_three : ℕ := 22

/-- The number of chocolate bars in each of the last two of the remaining small boxes -/
def chocolates_in_last_two : ℕ := 30

/-- The total number of chocolate bars in the large box -/
def total_chocolates : ℕ := 1202

theorem chocolate_count_correct : 
  (boxes_with_medium * medium_boxes_per_small * chocolates_per_medium) +
  (2 * chocolates_in_first_two) +
  (3 * chocolates_in_next_three) +
  (2 * chocolates_in_last_two) = total_chocolates :=
by sorry

end NUMINAMATH_CALUDE_chocolate_count_correct_l122_12275


namespace NUMINAMATH_CALUDE_prob_same_color_is_17_25_l122_12215

def num_green_balls : ℕ := 8
def num_red_balls : ℕ := 2
def total_balls : ℕ := num_green_balls + num_red_balls

def prob_same_color : ℚ := (num_green_balls / total_balls)^2 + (num_red_balls / total_balls)^2

theorem prob_same_color_is_17_25 : prob_same_color = 17 / 25 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_color_is_17_25_l122_12215
