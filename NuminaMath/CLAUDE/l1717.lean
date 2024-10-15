import Mathlib

namespace NUMINAMATH_CALUDE_committee_arrangement_l1717_171781

theorem committee_arrangement (n m : ℕ) (hn : n = 6) (hm : m = 4) : 
  Nat.choose (n + m) m = 210 := by
  sorry

end NUMINAMATH_CALUDE_committee_arrangement_l1717_171781


namespace NUMINAMATH_CALUDE_equation_solution_l1717_171789

theorem equation_solution (n k l m : ℕ) :
  l > 1 →
  (1 + n^k)^l = 1 + n^m →
  n = 2 ∧ k = 1 ∧ l = 2 ∧ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1717_171789


namespace NUMINAMATH_CALUDE_correct_total_items_l1717_171734

/-- Represents the requirements for a packed lunch --/
structure LunchRequirements where
  sandwiches_per_student : ℕ
  bread_slices_per_sandwich : ℕ
  chips_per_student : ℕ
  apples_per_student : ℕ
  granola_bars_per_student : ℕ

/-- Represents the number of students in each group --/
structure StudentGroups where
  group_a : ℕ
  group_b : ℕ
  group_c : ℕ

/-- Calculates the total number of items needed for packed lunches --/
def calculate_total_items (req : LunchRequirements) (groups : StudentGroups) :
  (ℕ × ℕ × ℕ × ℕ) :=
  let total_students := groups.group_a + groups.group_b + groups.group_c
  let total_bread_slices := total_students * req.sandwiches_per_student * req.bread_slices_per_sandwich
  let total_chips := total_students * req.chips_per_student
  let total_apples := total_students * req.apples_per_student
  let total_granola_bars := total_students * req.granola_bars_per_student
  (total_bread_slices, total_chips, total_apples, total_granola_bars)

/-- Theorem stating the correct calculation of total items needed --/
theorem correct_total_items :
  let req : LunchRequirements := {
    sandwiches_per_student := 2,
    bread_slices_per_sandwich := 4,
    chips_per_student := 1,
    apples_per_student := 3,
    granola_bars_per_student := 1
  }
  let groups : StudentGroups := {
    group_a := 10,
    group_b := 15,
    group_c := 20
  }
  calculate_total_items req groups = (360, 45, 135, 45) :=
by
  sorry


end NUMINAMATH_CALUDE_correct_total_items_l1717_171734


namespace NUMINAMATH_CALUDE_cube_root_function_l1717_171730

theorem cube_root_function (k : ℝ) :
  (∃ y : ℝ, y = k * 64^(1/3) ∧ y = 4 * Real.sqrt 3) →
  k * 8^(1/3) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_function_l1717_171730


namespace NUMINAMATH_CALUDE_sophie_donuts_to_sister_l1717_171792

/-- The number of donuts Sophie gave to her sister --/
def donuts_to_sister (total_boxes : ℕ) (donuts_per_box : ℕ) (boxes_to_mom : ℕ) (donuts_for_self : ℕ) : ℕ :=
  total_boxes * donuts_per_box - boxes_to_mom * donuts_per_box - donuts_for_self

theorem sophie_donuts_to_sister :
  donuts_to_sister 4 12 1 30 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sophie_donuts_to_sister_l1717_171792


namespace NUMINAMATH_CALUDE_last_boat_passengers_l1717_171780

/-- The number of people on a boat trip -/
def boat_trip (m : ℕ) : Prop :=
  ∃ (total : ℕ),
    -- Condition 1: m boats with 10 seats each leaves 8 people without seats
    total = 10 * m + 8 ∧
    -- Condition 2 & 3: Using boats with 16 seats each, 1 fewer boat is rented, and last boat is not full
    ∃ (last_boat : ℕ), last_boat > 0 ∧ last_boat < 16 ∧
      total = 16 * (m - 1) + last_boat

/-- The number of people on the last boat with 16 seats -/
theorem last_boat_passengers (m : ℕ) (h : boat_trip m) :
  ∃ (last_boat : ℕ), last_boat = 40 - 6 * m :=
by sorry

end NUMINAMATH_CALUDE_last_boat_passengers_l1717_171780


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l1717_171714

theorem arithmetic_mean_problem (a₁ a₂ a₃ a₄ a₅ a₆ A : ℝ) 
  (h1 : (a₁ + a₂ + a₃ + a₄ + a₅ + a₆) / 6 = A)
  (h2 : (a₁ + a₂ + a₃ + a₄) / 4 = A + 10)
  (h3 : (a₃ + a₄ + a₅ + a₆) / 4 = A - 7) :
  (a₁ + a₂ + a₅ + a₆) / 4 = A - 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l1717_171714


namespace NUMINAMATH_CALUDE_range_of_a_l1717_171790

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x > 0 → a < x + 1/x) → a < 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1717_171790


namespace NUMINAMATH_CALUDE_log_simplification_l1717_171706

theorem log_simplification :
  (Real.log 3 / Real.log 4 + Real.log 3 / Real.log 8) *
  (Real.log 2 / Real.log 3 + Real.log 2 / Real.log 9) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_log_simplification_l1717_171706


namespace NUMINAMATH_CALUDE_complex_calculation_l1717_171775

theorem complex_calculation : 
  let z : ℂ := 1 + I
  z^2 - 2/z = -1 + 3*I := by
  sorry

end NUMINAMATH_CALUDE_complex_calculation_l1717_171775


namespace NUMINAMATH_CALUDE_cos_three_pi_halves_l1717_171751

theorem cos_three_pi_halves : Real.cos (3 * π / 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_three_pi_halves_l1717_171751


namespace NUMINAMATH_CALUDE_shopkeeper_profit_l1717_171705

/-- Calculates the actual percent profit when a shopkeeper labels an item's price
    to earn a specified profit percentage and then offers a discount. -/
def actualPercentProfit (labeledProfitPercent : ℝ) (discountPercent : ℝ) : ℝ :=
  let labeledPrice := 1 + labeledProfitPercent
  let sellingPrice := labeledPrice * (1 - discountPercent)
  (sellingPrice - 1) * 100

/-- Proves that when a shopkeeper labels an item's price to earn a 30% profit
    on the cost price and then offers a 10% discount on the labeled price,
    the actual percent profit earned is 17%. -/
theorem shopkeeper_profit :
  actualPercentProfit 0.3 0.1 = 17 :=
by sorry

end NUMINAMATH_CALUDE_shopkeeper_profit_l1717_171705


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1717_171747

open Set

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x : ℝ | x^2 - 2*x - 3 > 0}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = Iic 2 ∪ Ioi 3 := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1717_171747


namespace NUMINAMATH_CALUDE_root_implies_h_value_l1717_171774

theorem root_implies_h_value (h : ℝ) : 
  ((-1 : ℝ)^3 + h * (-1) - 20 = 0) → h = -21 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_h_value_l1717_171774


namespace NUMINAMATH_CALUDE_cost_price_calculation_l1717_171738

theorem cost_price_calculation (selling_price : ℚ) (profit_percentage : ℚ) : 
  selling_price = 600 → profit_percentage = 60 → 
  ∃ (cost_price : ℚ), cost_price = 375 ∧ selling_price = cost_price * (1 + profit_percentage / 100) :=
by sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l1717_171738


namespace NUMINAMATH_CALUDE_wine_sales_regression_l1717_171707

/-- Linear regression problem for white wine sales and unit cost -/
theorem wine_sales_regression 
  (x_mean : ℝ) 
  (y_mean : ℝ) 
  (sum_x_squared : ℝ) 
  (sum_xy : ℝ) 
  (n : ℕ) 
  (h_x_mean : x_mean = 7/2)
  (h_y_mean : y_mean = 71)
  (h_sum_x_squared : sum_x_squared = 79)
  (h_sum_xy : sum_xy = 1481)
  (h_n : n = 6) :
  let b := (sum_xy - n * x_mean * y_mean) / (sum_x_squared - n * x_mean^2)
  ∃ ε > 0, |b + 1.8182| < ε :=
sorry

end NUMINAMATH_CALUDE_wine_sales_regression_l1717_171707


namespace NUMINAMATH_CALUDE_exponent_problem_l1717_171712

theorem exponent_problem (a : ℝ) (x y : ℝ) (h1 : a^x = 2) (h2 : a^y = 3) : a^(2*x + y) = 12 := by
  sorry

end NUMINAMATH_CALUDE_exponent_problem_l1717_171712


namespace NUMINAMATH_CALUDE_fraction_difference_l1717_171795

theorem fraction_difference (m n : ℝ) (h1 : m^2 - n^2 = m*n) (h2 : m*n ≠ 0) :
  n/m - m/n = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_difference_l1717_171795


namespace NUMINAMATH_CALUDE_birthday_age_problem_l1717_171748

theorem birthday_age_problem (current_age : ℕ) : 
  (current_age = 3 * (current_age - 6)) → current_age = 9 := by
  sorry

end NUMINAMATH_CALUDE_birthday_age_problem_l1717_171748


namespace NUMINAMATH_CALUDE_total_fruits_count_l1717_171717

-- Define the given conditions
def gerald_apple_bags : ℕ := 5
def gerald_apples_per_bag : ℕ := 30
def gerald_orange_bags : ℕ := 4
def gerald_oranges_per_bag : ℕ := 25

def pam_apple_bags : ℕ := 6
def pam_orange_bags : ℕ := 4

def sue_apple_bags : ℕ := 2 * gerald_apple_bags
def sue_orange_bags : ℕ := gerald_orange_bags / 2

def pam_apples_per_bag : ℕ := 3 * gerald_apples_per_bag
def pam_oranges_per_bag : ℕ := 2 * gerald_oranges_per_bag

def sue_apples_per_bag : ℕ := gerald_apples_per_bag - 10
def sue_oranges_per_bag : ℕ := gerald_oranges_per_bag + 5

-- Theorem statement
theorem total_fruits_count : 
  (gerald_apple_bags * gerald_apples_per_bag + 
   gerald_orange_bags * gerald_oranges_per_bag +
   pam_apple_bags * pam_apples_per_bag + 
   pam_orange_bags * pam_oranges_per_bag +
   sue_apple_bags * sue_apples_per_bag + 
   sue_orange_bags * sue_oranges_per_bag) = 1250 := by
  sorry

end NUMINAMATH_CALUDE_total_fruits_count_l1717_171717


namespace NUMINAMATH_CALUDE_fraction_cube_theorem_l1717_171750

theorem fraction_cube_theorem :
  (2 : ℚ) / 5 ^ 3 = 8 / 125 :=
by sorry

end NUMINAMATH_CALUDE_fraction_cube_theorem_l1717_171750


namespace NUMINAMATH_CALUDE_smallest_n_for_f_greater_than_15_l1717_171797

-- Define the function f
def f (n : ℕ+) : ℕ := sorry

-- Theorem statement
theorem smallest_n_for_f_greater_than_15 :
  (∀ k : ℕ+, k < 4 → f k ≤ 15) ∧ f 4 > 15 := by sorry

end NUMINAMATH_CALUDE_smallest_n_for_f_greater_than_15_l1717_171797


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l1717_171701

theorem circle_diameter_from_area :
  ∀ (r d : ℝ),
  r > 0 →
  d = 2 * r →
  π * r^2 = 225 * π →
  d = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l1717_171701


namespace NUMINAMATH_CALUDE_sum_of_interior_angles_formula_l1717_171782

-- Define a non-crossed polygon
def NonCrossedPolygon (n : ℕ) : Type := sorry

-- Define the sum of interior angles of a polygon
def SumOfInteriorAngles (p : NonCrossedPolygon n) : ℝ := sorry

-- Theorem statement
theorem sum_of_interior_angles_formula {n : ℕ} (h : n ≥ 3) (p : NonCrossedPolygon n) :
  SumOfInteriorAngles p = (n - 2) * 180 := by sorry

end NUMINAMATH_CALUDE_sum_of_interior_angles_formula_l1717_171782


namespace NUMINAMATH_CALUDE_triangle_inequality_l1717_171728

theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt (a / (b + c)) + Real.sqrt (b / (a + c)) + Real.sqrt (c / (a + b)) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1717_171728


namespace NUMINAMATH_CALUDE_all_five_digit_sum_30_div_9_l1717_171720

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

def digit_sum (n : ℕ) : ℕ :=
  (n / 10000) + ((n / 1000) % 10) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

theorem all_five_digit_sum_30_div_9 :
  ∀ n : ℕ, is_five_digit n → digit_sum n = 30 → n % 9 = 0 :=
sorry

end NUMINAMATH_CALUDE_all_five_digit_sum_30_div_9_l1717_171720


namespace NUMINAMATH_CALUDE_circle_ellipse_tangent_l1717_171725

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 5

-- Define the ellipse E
def ellipse_E (x y : ℝ) : Prop := x^2 / 18 + y^2 / 2 = 1

-- Define the line PF₁
def line_PF1 (x y : ℝ) : Prop := x - 2*y + 4 = 0

-- Define the point A
def point_A : ℝ × ℝ := (3, 1)

-- Define the point P
def point_P : ℝ × ℝ := (4, 4)

-- State the theorem
theorem circle_ellipse_tangent :
  ∃ (m : ℝ),
    m < 3 ∧
    (∃ (a b : ℝ), a > b ∧ b > 0 ∧
      (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ ellipse_E x y)) ∧
    (∃ e : ℝ, e > 1/2 ∧
      (∀ x y : ℝ, ((x - m)^2 + y^2 = 5) ↔ circle_C x y) ∧
      circle_C point_A.1 point_A.2 ∧
      ellipse_E point_A.1 point_A.2 ∧
      line_PF1 point_P.1 point_P.2) :=
sorry

end NUMINAMATH_CALUDE_circle_ellipse_tangent_l1717_171725


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l1717_171723

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 10) :
  (1 / x + 1 / y) ≥ 2 / 5 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ x + y = 10 ∧ 1 / x + 1 / y = 2 / 5 :=
sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l1717_171723


namespace NUMINAMATH_CALUDE_base_five_to_decimal_l1717_171740

/-- Converts a list of digits in a given base to its decimal representation. -/
def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base ^ (digits.length - 1 - i)) 0

/-- The decimal representation of 3412 in base 5 is 482. -/
theorem base_five_to_decimal : to_decimal [3, 4, 1, 2] 5 = 482 := by sorry

end NUMINAMATH_CALUDE_base_five_to_decimal_l1717_171740


namespace NUMINAMATH_CALUDE_dianes_honey_harvest_l1717_171763

/-- Diane's honey harvest calculation -/
theorem dianes_honey_harvest (last_year_harvest : ℕ) (increase : ℕ) : 
  last_year_harvest = 2479 → increase = 6085 → last_year_harvest + increase = 8564 := by
  sorry

end NUMINAMATH_CALUDE_dianes_honey_harvest_l1717_171763


namespace NUMINAMATH_CALUDE_sphere_surface_area_l1717_171737

/-- The surface area of a sphere, given specific conditions for a hemisphere --/
theorem sphere_surface_area (r : ℝ) 
  (h1 : π * r^2 = 3)  -- area of the base of the hemisphere
  (h2 : 3 * π * r^2 = 9)  -- total surface area of the hemisphere
  : 4 * π * r^2 = 12 := by
  sorry

#check sphere_surface_area

end NUMINAMATH_CALUDE_sphere_surface_area_l1717_171737


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l1717_171700

theorem least_subtraction_for_divisibility :
  ∃ (n : ℕ), n ≤ 7 ∧
  12 ∣ (652543 - n) ∧
  ∀ (m : ℕ), m < n → ¬(12 ∣ (652543 - m)) :=
sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l1717_171700


namespace NUMINAMATH_CALUDE_track_length_is_600_l1717_171770

/-- Represents a circular running track with two runners -/
structure CircularTrack where
  length : ℝ
  brenda_speed : ℝ
  sally_speed : ℝ

/-- The conditions of the problem -/
def problem_conditions (track : CircularTrack) : Prop :=
  ∃ (first_meeting_time second_meeting_time : ℝ),
    -- Brenda runs 120 meters before first meeting
    track.brenda_speed * first_meeting_time = 120 ∧
    -- Sally runs (length/2 - 120) meters before first meeting
    track.sally_speed * first_meeting_time = track.length / 2 - 120 ∧
    -- Sally runs an additional 180 meters between meetings
    track.sally_speed * (second_meeting_time - first_meeting_time) = 180 ∧
    -- Brenda's position at second meeting
    track.brenda_speed * second_meeting_time =
      track.length - (track.length / 2 - 120 + 180)

/-- The theorem to be proven -/
theorem track_length_is_600 (track : CircularTrack) :
  problem_conditions track → track.length = 600 := by
  sorry


end NUMINAMATH_CALUDE_track_length_is_600_l1717_171770


namespace NUMINAMATH_CALUDE_power_product_rule_l1717_171741

theorem power_product_rule (a : ℝ) : a^2 * a^4 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_product_rule_l1717_171741


namespace NUMINAMATH_CALUDE_parallel_lines_problem_l1717_171762

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m1 m2 b1 b2 : ℝ} :
  (∀ x y : ℝ, m1 * x + y = b1 ↔ m2 * x + y = b2) ↔ m1 = m2

/-- The problem statement -/
theorem parallel_lines_problem (a : ℝ) :
  (∀ x y : ℝ, (a - 1) * x + y - 1 = 0 ↔ 6 * x + a * y + 2 = 0) →
  a = 3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_problem_l1717_171762


namespace NUMINAMATH_CALUDE_solution_set_f_range_of_a_l1717_171798

-- Part 1
def f (x : ℝ) : ℝ := |2 * x + 1| + |2 * x - 3|

theorem solution_set_f (x : ℝ) : f x = 4 ↔ -1/2 ≤ x ∧ x ≤ 3/2 := by sorry

-- Part 2
def f' (a x : ℝ) : ℝ := |x + a| + |x - 1|
def g (x : ℝ) : ℝ := |x - 2| + 1

theorem range_of_a (a : ℝ) :
  (∀ x₁, ∃ x₂, g x₂ = f' a x₁) → a ≤ -2 ∨ a ≥ 0 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_range_of_a_l1717_171798


namespace NUMINAMATH_CALUDE_investment_value_l1717_171711

theorem investment_value (x : ℝ) : 
  (0.07 * 500 + 0.23 * x = 0.19 * (500 + x)) → x = 1500 := by
  sorry

end NUMINAMATH_CALUDE_investment_value_l1717_171711


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_and_discriminant_l1717_171757

theorem quadratic_equation_roots_and_discriminant :
  let a : ℝ := 1
  let b : ℝ := 5
  let c : ℝ := 0
  let f : ℝ → ℝ := λ x => x^2 + b*x + c
  let discriminant := b^2 - 4*a*c
  ∃ x₁ x₂ : ℝ, (f x₁ = 0 ∧ f x₂ = 0) ∧ 
              (x₁ = 0 ∧ x₂ = -5) ∧
              discriminant = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_and_discriminant_l1717_171757


namespace NUMINAMATH_CALUDE_triangle_area_with_given_conditions_l1717_171791

/-- Given a triangle PQR with inradius r, circumradius R, and angles P, Q, R,
    prove that if r = 8, R = 25, and 2 * cos Q = cos P + cos R, then the area of the triangle is 96. -/
theorem triangle_area_with_given_conditions (P Q R : Real) (r R : ℝ) : 
  r = 8 → R = 25 → 2 * Real.cos Q = Real.cos P + Real.cos R → 
  ∃ (area : ℝ), area = 96 ∧ area = r * (R * Real.sin Q) := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_with_given_conditions_l1717_171791


namespace NUMINAMATH_CALUDE_suit_price_calculation_l1717_171733

theorem suit_price_calculation (original_price : ℝ) : 
  original_price * 1.25 * 0.75 = 187.5 → original_price = 200 := by
  sorry

end NUMINAMATH_CALUDE_suit_price_calculation_l1717_171733


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l1717_171727

theorem quadratic_inequality_solution_range (a : ℝ) : 
  (a < -4) ↔ 
  (∃ x : ℝ, 1 < x ∧ x < 4 ∧ 2 * x^2 - 8 * x - 4 - a > 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l1717_171727


namespace NUMINAMATH_CALUDE_triangle_game_probability_l1717_171777

/-- A game board constructed from an equilateral triangle -/
structure GameBoard :=
  (total_sections : ℕ)
  (shaded_sections : ℕ)
  (h_positive : 0 < total_sections)
  (h_shaded_le_total : shaded_sections ≤ total_sections)

/-- The probability of the spinner landing in a shaded region -/
def landing_probability (board : GameBoard) : ℚ :=
  board.shaded_sections / board.total_sections

/-- Theorem stating that for a game board with 6 total sections and 2 shaded sections,
    the probability of landing in a shaded region is 1/3 -/
theorem triangle_game_probability :
  ∀ (board : GameBoard),
    board.total_sections = 6 →
    board.shaded_sections = 2 →
    landing_probability board = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_game_probability_l1717_171777


namespace NUMINAMATH_CALUDE_arccos_one_half_equals_pi_third_l1717_171722

theorem arccos_one_half_equals_pi_third : Real.arccos (1/2) = π/3 := by
  sorry

end NUMINAMATH_CALUDE_arccos_one_half_equals_pi_third_l1717_171722


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l1717_171744

/-- The repeating decimal 0.363636... as a real number -/
def repeating_decimal : ℚ := 36 / 99

/-- Theorem stating that the repeating decimal 0.363636... is equal to 4/11 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = 4 / 11 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l1717_171744


namespace NUMINAMATH_CALUDE_jupiter_properties_l1717_171794

/-- Given orbital parameters of a moon, calculate properties of Jupiter -/
theorem jupiter_properties 
  (T : ℝ) -- Orbital period of the moon
  (R : ℝ) -- Orbital distance of the moon
  (f : ℝ) -- Gravitational constant
  (ρ : ℝ) -- Radius of Jupiter
  (V : ℝ) -- Volume of Jupiter
  (T_rot : ℝ) -- Rotational period of Jupiter
  (h₁ : T > 0)
  (h₂ : R > 0)
  (h₃ : f > 0)
  (h₄ : ρ > 0)
  (h₅ : V > 0)
  (h₆ : T_rot > 0) :
  ∃ (M σ g₁ Cf : ℝ),
    M = 4 * Real.pi^2 * R^3 / (f * T^2) ∧
    σ = M / V ∧
    g₁ = f * M / ρ^2 ∧
    Cf = 4 * Real.pi^2 * ρ / T_rot^2 :=
by
  sorry


end NUMINAMATH_CALUDE_jupiter_properties_l1717_171794


namespace NUMINAMATH_CALUDE_remainder_3_304_mod_11_l1717_171787

theorem remainder_3_304_mod_11 : 3^304 % 11 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3_304_mod_11_l1717_171787


namespace NUMINAMATH_CALUDE_binary_decimal_octal_conversion_l1717_171779

/-- Converts a binary number represented as a list of bits to a decimal number -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a decimal number to an octal number represented as a list of digits -/
def decimal_to_octal (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec go (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc
      else go (m / 8) ((m % 8) :: acc)
    go n []

/-- The binary representation of 11011100₂ -/
def binary_num : List Bool := [false, false, true, true, true, false, true, true]

theorem binary_decimal_octal_conversion :
  (binary_to_decimal binary_num = 110) ∧
  (decimal_to_octal 110 = [1, 5, 6]) := by
  sorry


end NUMINAMATH_CALUDE_binary_decimal_octal_conversion_l1717_171779


namespace NUMINAMATH_CALUDE_evaluate_expression_l1717_171764

theorem evaluate_expression : (4^4 - 4*(4-2)^4)^4 = 1358954496 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1717_171764


namespace NUMINAMATH_CALUDE_first_day_of_month_is_sunday_l1717_171754

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define a function to get the day of the week for a given day of the month
def dayOfWeek (dayOfMonth : Nat) : DayOfWeek := sorry

-- Theorem statement
theorem first_day_of_month_is_sunday 
  (h : dayOfWeek 18 = DayOfWeek.Wednesday) : 
  dayOfWeek 1 = DayOfWeek.Sunday := by
  sorry

end NUMINAMATH_CALUDE_first_day_of_month_is_sunday_l1717_171754


namespace NUMINAMATH_CALUDE_total_supervisors_is_25_l1717_171784

/-- The total number of supervisors on 5 buses -/
def total_supervisors : ℕ := 4 + 5 + 3 + 6 + 7

/-- Theorem stating that the total number of supervisors is 25 -/
theorem total_supervisors_is_25 : total_supervisors = 25 := by
  sorry

end NUMINAMATH_CALUDE_total_supervisors_is_25_l1717_171784


namespace NUMINAMATH_CALUDE_num_triangles_in_circle_l1717_171710

/-- The number of points on the circle -/
def n : ℕ := 9

/-- The number of chords -/
def num_chords : ℕ := n.choose 2

/-- The number of intersection points inside the circle -/
def num_intersections : ℕ := n.choose 4

/-- Theorem: The number of triangles formed by intersection points of chords inside a circle -/
theorem num_triangles_in_circle (n : ℕ) (h : n = 9) : 
  (num_intersections.choose 3) = 315500 :=
sorry

end NUMINAMATH_CALUDE_num_triangles_in_circle_l1717_171710


namespace NUMINAMATH_CALUDE_find_x_l1717_171721

theorem find_x : ∃ x : ℝ,
  (24 + 35 + 58) / 3 = ((19 + 51 + x) / 3) + 6 → x = 29 :=
by sorry

end NUMINAMATH_CALUDE_find_x_l1717_171721


namespace NUMINAMATH_CALUDE_digits_of_2_pow_100_l1717_171773

theorem digits_of_2_pow_100 (h : ∃ n : ℕ, 10^(n-1) ≤ 2^200 ∧ 2^200 < 10^n ∧ n = 61) :
  ∃ m : ℕ, 10^(m-1) ≤ 2^100 ∧ 2^100 < 10^m ∧ m = 31 :=
by sorry

end NUMINAMATH_CALUDE_digits_of_2_pow_100_l1717_171773


namespace NUMINAMATH_CALUDE_parallelogram_area_l1717_171702

/-- Represents a parallelogram with given base, height, and one angle -/
structure Parallelogram where
  base : ℝ
  height : ℝ
  angle : ℝ

/-- Calculates the area of a parallelogram -/
def area (p : Parallelogram) : ℝ := p.base * p.height

/-- Theorem: The area of a parallelogram with base 20 and height 4 is 80 -/
theorem parallelogram_area :
  ∀ (p : Parallelogram), p.base = 20 ∧ p.height = 4 ∧ p.angle = 60 → area p = 80 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l1717_171702


namespace NUMINAMATH_CALUDE_dinos_third_gig_rate_l1717_171786

/-- Dino's monthly income calculation -/
def monthly_income (hours1 hours2 hours3 : ℕ) (rate1 rate2 rate3 : ℚ) : ℚ :=
  hours1 * rate1 + hours2 * rate2 + hours3 * rate3

/-- Theorem: Dino's hourly rate for the third gig is $40/hour -/
theorem dinos_third_gig_rate :
  ∀ (rate3 : ℚ),
  monthly_income 20 30 5 10 20 rate3 = 1000 →
  rate3 = 40 := by
sorry

end NUMINAMATH_CALUDE_dinos_third_gig_rate_l1717_171786


namespace NUMINAMATH_CALUDE_percentage_failed_english_l1717_171766

theorem percentage_failed_english (failed_hindi : ℝ) (failed_both : ℝ) (passed_both : ℝ)
  (h1 : failed_hindi = 34)
  (h2 : failed_both = 22)
  (h3 : passed_both = 44) :
  ∃ failed_english : ℝ,
    failed_english = 44 ∧
    failed_hindi + failed_english - failed_both = 100 - passed_both :=
by sorry

end NUMINAMATH_CALUDE_percentage_failed_english_l1717_171766


namespace NUMINAMATH_CALUDE_trapezoid_segment_length_squared_l1717_171755

/-- Represents a trapezoid with the given properties -/
structure Trapezoid where
  b : ℝ  -- length of the shorter base
  h : ℝ  -- height of the trapezoid
  midline_ratio : (b + 75) / (b + 25) = 3 / 2  -- ratio condition for the midline
  x : ℝ  -- length of the segment dividing the trapezoid into two equal areas
  equal_area_condition : x = 125 * (100 / (x - 75)) - 75

/-- The main theorem about the trapezoid -/
theorem trapezoid_segment_length_squared (t : Trapezoid) :
  ⌊(t.x^2) / 100⌋ = 181 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_segment_length_squared_l1717_171755


namespace NUMINAMATH_CALUDE_quadratic_real_solutions_l1717_171743

theorem quadratic_real_solutions (p : ℝ) :
  (∃ x : ℝ, x^2 + p = 0) ↔ p ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_solutions_l1717_171743


namespace NUMINAMATH_CALUDE_no_prime_sum_10003_l1717_171745

theorem no_prime_sum_10003 : ¬∃ (p q : ℕ), Prime p ∧ Prime q ∧ p + q = 10003 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_sum_10003_l1717_171745


namespace NUMINAMATH_CALUDE_max_CP_value_l1717_171771

-- Define the equilateral triangle ABC
def EquilateralTriangle (A B C : ℝ × ℝ) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A

-- Define the point P
def P : ℝ × ℝ := sorry

-- Define the distances
def AP : ℝ := 2
def BP : ℝ := 3

-- Theorem statement
theorem max_CP_value 
  (A B C : ℝ × ℝ) 
  (h_equilateral : EquilateralTriangle A B C) 
  (h_AP : dist A P = AP) 
  (h_BP : dist B P = BP) :
  ∀ P', dist C P' ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_max_CP_value_l1717_171771


namespace NUMINAMATH_CALUDE_path_length_proof_l1717_171726

theorem path_length_proof :
  let rectangle_width : ℝ := 3
  let rectangle_height : ℝ := 4
  let diagonal_length : ℝ := (rectangle_width^2 + rectangle_height^2).sqrt
  let vertical_segments : ℝ := 2 * rectangle_height
  let horizontal_segments : ℝ := 3 * rectangle_width
  diagonal_length + vertical_segments + horizontal_segments = 22 := by
sorry

end NUMINAMATH_CALUDE_path_length_proof_l1717_171726


namespace NUMINAMATH_CALUDE_jesse_banana_sharing_l1717_171732

theorem jesse_banana_sharing (total_bananas : ℕ) (bananas_per_friend : ℕ) (h1 : total_bananas = 21) (h2 : bananas_per_friend = 7) :
  total_bananas / bananas_per_friend = 3 :=
by sorry

end NUMINAMATH_CALUDE_jesse_banana_sharing_l1717_171732


namespace NUMINAMATH_CALUDE_angle_A_is_60_degrees_range_of_b_plus_c_l1717_171749

/-- Triangle ABC with sides a, b, c corresponding to angles A, B, C respectively --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition for the triangle --/
def triangle_condition (t : Triangle) : Prop :=
  Real.sqrt 3 * t.a * Real.sin t.C + t.a * Real.cos t.C = t.c + t.b

/-- Theorem 1: Angle A is 60° --/
theorem angle_A_is_60_degrees (t : Triangle) (h : triangle_condition t) : t.A = π / 3 := by
  sorry

/-- Theorem 2: Range of b + c when a = √3 --/
theorem range_of_b_plus_c (t : Triangle) (h1 : triangle_condition t) (h2 : t.a = Real.sqrt 3) :
  Real.sqrt 3 < t.b + t.c ∧ t.b + t.c ≤ 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_A_is_60_degrees_range_of_b_plus_c_l1717_171749


namespace NUMINAMATH_CALUDE_problem_solution_l1717_171719

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

-- Theorem statement
theorem problem_solution :
  (∀ x : ℝ, f x ≤ 4 ↔ -2 ≤ x ∧ x ≤ 2) ∧
  (∀ a : ℝ, (∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 → |x + 3| + |x + a| < x + 6) ↔ -1 < a ∧ a < 1) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1717_171719


namespace NUMINAMATH_CALUDE_right_to_left_equiv_ordinary_l1717_171752

-- Define a function to represent the right-to-left evaluation
def rightToLeftEval (a b c d : ℝ) : ℝ := a * (b + (c - d))

-- Define a function to represent the ordinary algebraic notation
def ordinaryNotation (a b c d : ℝ) : ℝ := a * (b + c - d)

-- Theorem statement
theorem right_to_left_equiv_ordinary (a b c d : ℝ) :
  rightToLeftEval a b c d = ordinaryNotation a b c d := by
  sorry

end NUMINAMATH_CALUDE_right_to_left_equiv_ordinary_l1717_171752


namespace NUMINAMATH_CALUDE_prob_first_odd_given_two_odd_one_even_l1717_171735

/-- Represents the outcome of picking a ball -/
inductive BallType
| Odd
| Even

/-- Represents the result of picking 3 balls -/
structure ThreePickResult :=
  (first second third : BallType)

def is_valid_pick (result : ThreePickResult) : Prop :=
  (result.first = BallType.Odd ∧ result.second = BallType.Odd ∧ result.third = BallType.Even) ∨
  (result.first = BallType.Odd ∧ result.second = BallType.Even ∧ result.third = BallType.Odd) ∨
  (result.first = BallType.Even ∧ result.second = BallType.Odd ∧ result.third = BallType.Odd)

def total_balls : ℕ := 100
def odd_balls : ℕ := 50
def even_balls : ℕ := 50

theorem prob_first_odd_given_two_odd_one_even :
  ∀ (sample_space : Set ThreePickResult) (prob : Set ThreePickResult → ℝ),
  (∀ result ∈ sample_space, is_valid_pick result) →
  (∀ A ⊆ sample_space, 0 ≤ prob A ∧ prob A ≤ 1) →
  prob sample_space = 1 →
  prob {result ∈ sample_space | result.first = BallType.Odd} = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_prob_first_odd_given_two_odd_one_even_l1717_171735


namespace NUMINAMATH_CALUDE_point_b_not_on_curve_l1717_171760

/-- The equation of curve C -/
def curve_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 + 6*a*x - 8*a*y = 0

/-- Point B does not lie on curve C -/
theorem point_b_not_on_curve (a : ℝ) : ¬ curve_equation (2*a) (4*a) a := by
  sorry

end NUMINAMATH_CALUDE_point_b_not_on_curve_l1717_171760


namespace NUMINAMATH_CALUDE_initial_gummy_worms_l1717_171778

def gummy_worms (n : ℕ) : ℕ → ℕ
  | 0 => n  -- Initial number of gummy worms
  | d + 1 => (gummy_worms n d) / 2  -- Number of gummy worms after d + 1 days

theorem initial_gummy_worms :
  ∀ n : ℕ, gummy_worms n 4 = 4 → n = 64 := by
  sorry

end NUMINAMATH_CALUDE_initial_gummy_worms_l1717_171778


namespace NUMINAMATH_CALUDE_inequalities_theorem_l1717_171718

theorem inequalities_theorem (a b : ℝ) (m n : ℕ) 
    (ha : 0 < a) (hb : 0 < b) (hm : 0 < m) (hn : 0 < n) : 
  (a^n + b^n) * (a^m + b^m) ≤ 2 * (a^(m+n) + b^(m+n)) ∧ 
  (a + b) / 2 * (a^2 + b^2) / 2 * (a^3 + b^3) / 2 ≤ (a^6 + b^6) / 2 :=
by sorry

end NUMINAMATH_CALUDE_inequalities_theorem_l1717_171718


namespace NUMINAMATH_CALUDE_johns_starting_elevation_l1717_171704

def starting_elevation (rate : ℝ) (time : ℝ) (final_elevation : ℝ) : ℝ :=
  final_elevation + rate * time

theorem johns_starting_elevation :
  starting_elevation 10 5 350 = 400 := by sorry

end NUMINAMATH_CALUDE_johns_starting_elevation_l1717_171704


namespace NUMINAMATH_CALUDE_roberta_shopping_trip_l1717_171724

def shopping_trip (initial_amount bag_price_difference lunch_price_fraction : ℚ) : ℚ :=
  let shoe_price := 45
  let bag_price := shoe_price - bag_price_difference
  let lunch_price := bag_price * lunch_price_fraction
  initial_amount - (shoe_price + bag_price + lunch_price)

theorem roberta_shopping_trip :
  shopping_trip 158 17 (1/4) = 78 := by
  sorry

end NUMINAMATH_CALUDE_roberta_shopping_trip_l1717_171724


namespace NUMINAMATH_CALUDE_point_quadrant_relation_l1717_171788

/-- If point M(1+a, 2b-1) is in the third quadrant, then point N(a-1, 1-2b) is in the second quadrant. -/
theorem point_quadrant_relation (a b : ℝ) : 
  (1 + a < 0 ∧ 2*b - 1 < 0) → (a - 1 < 0 ∧ 1 - 2*b > 0) :=
by sorry

end NUMINAMATH_CALUDE_point_quadrant_relation_l1717_171788


namespace NUMINAMATH_CALUDE_digit_properties_l1717_171715

def Digits : Set Nat := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

theorem digit_properties :
  ∀ (a b : Nat), a ∈ Digits → b ∈ Digits → a ≠ b →
    (∀ (x y : Nat), x ∈ Digits → y ∈ Digits → x ≠ y → (a + b) * (a * b) ≥ (x + y) * (x * y)) ∧
    (∀ (x y : Nat), x ∈ Digits → y ∈ Digits → x ≠ y → (0 + 1) * (0 * 1) ≤ (x + y) * (x * y)) ∧
    (∀ (x y : Nat), x ∈ Digits → y ∈ Digits → x + y = 10 ↔ 
      ((x = 1 ∧ y = 9) ∨ (x = 2 ∧ y = 8) ∨ (x = 3 ∧ y = 7) ∨ (x = 4 ∧ y = 6) ∨
       (x = 9 ∧ y = 1) ∨ (x = 8 ∧ y = 2) ∨ (x = 7 ∧ y = 3) ∨ (x = 6 ∧ y = 4))) :=
by
  sorry

end NUMINAMATH_CALUDE_digit_properties_l1717_171715


namespace NUMINAMATH_CALUDE_tenth_term_is_123_a_plus_b_power_10_is_123_l1717_171799

-- Define the sequence
def seq : ℕ → ℕ
| 0 => 1  -- a + b
| 1 => 3  -- a² + b²
| 2 => 4  -- a³ + b³
| n + 3 => seq (n + 1) + seq (n + 2)

-- State the theorem
theorem tenth_term_is_123 : seq 9 = 123 := by
  sorry

-- Define a and b
noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

-- State the given conditions
axiom sum_1 : a + b = 1
axiom sum_2 : a^2 + b^2 = 3
axiom sum_3 : a^3 + b^3 = 4
axiom sum_4 : a^4 + b^4 = 7
axiom sum_5 : a^5 + b^5 = 11

-- State the main theorem
theorem a_plus_b_power_10_is_123 : a^10 + b^10 = 123 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_is_123_a_plus_b_power_10_is_123_l1717_171799


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1717_171793

theorem quadratic_inequality_solution (a : ℝ) (h : 1 < a ∧ a < 2) :
  ∃ (x : ℝ), x^2 - (a^2 + 3*a + 2)*x + 3*a*(a^2 + 2) < 0 ↔ a^2 + 2 < x ∧ x < 3*a :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1717_171793


namespace NUMINAMATH_CALUDE_determinant_of_specific_matrix_l1717_171742

theorem determinant_of_specific_matrix : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![4, 1; 2, 5]
  Matrix.det A = 18 := by sorry

end NUMINAMATH_CALUDE_determinant_of_specific_matrix_l1717_171742


namespace NUMINAMATH_CALUDE_jacob_walking_distance_l1717_171731

/-- Calculates the distance traveled given a constant rate and time --/
def distance (rate : ℝ) (time : ℝ) : ℝ := rate * time

/-- Theorem: Jacob walks 8 miles in 2 hours at a rate of 4 miles per hour --/
theorem jacob_walking_distance :
  let rate : ℝ := 4
  let time : ℝ := 2
  distance rate time = 8 := by
  sorry

end NUMINAMATH_CALUDE_jacob_walking_distance_l1717_171731


namespace NUMINAMATH_CALUDE_fourth_root_squared_l1717_171703

theorem fourth_root_squared (y : ℝ) : (y^(1/4))^2 = 81 → y = 81 := by sorry

end NUMINAMATH_CALUDE_fourth_root_squared_l1717_171703


namespace NUMINAMATH_CALUDE_trigonometric_identities_l1717_171713

theorem trigonometric_identities (α : Real) (h : Real.tan α = 2) :
  (Real.sin α - 3 * Real.cos α) / (Real.sin α + Real.cos α) = -1/3 ∧
  2 * Real.sin α ^ 2 - Real.sin α * Real.cos α + Real.cos α ^ 2 = 7/5 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l1717_171713


namespace NUMINAMATH_CALUDE_jebb_take_home_pay_l1717_171739

/-- Calculates the take-home pay after tax deduction -/
def takeHomePay (totalPay : ℝ) (taxRate : ℝ) : ℝ :=
  totalPay * (1 - taxRate)

/-- Proves that Jebb's take-home pay is $585 -/
theorem jebb_take_home_pay :
  let totalPay : ℝ := 650
  let taxRate : ℝ := 0.1
  takeHomePay totalPay taxRate = 585 := by
  sorry

end NUMINAMATH_CALUDE_jebb_take_home_pay_l1717_171739


namespace NUMINAMATH_CALUDE_trillion_equals_ten_to_sixteen_l1717_171746

theorem trillion_equals_ten_to_sixteen :
  let ten_thousand : ℕ := 10^4
  let hundred_million : ℕ := 10^8
  let trillion : ℕ := ten_thousand * ten_thousand * hundred_million
  trillion = 10^16 := by
  sorry

end NUMINAMATH_CALUDE_trillion_equals_ten_to_sixteen_l1717_171746


namespace NUMINAMATH_CALUDE_sum_of_first_20_odd_integers_greater_than_10_l1717_171758

/-- The sum of the first n terms of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- The 20th term of the arithmetic sequence starting at 11 with common difference 2 -/
def a₂₀ : ℕ := 11 + 19 * 2

theorem sum_of_first_20_odd_integers_greater_than_10 :
  arithmetic_sum 11 2 20 = 600 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_20_odd_integers_greater_than_10_l1717_171758


namespace NUMINAMATH_CALUDE_externally_tangent_case_intersecting_case_l1717_171785

-- Define the circles
def circle_O₁ (x y : ℝ) : Prop := x^2 + (y + 1)^2 = 4
def center_O₂ : ℝ × ℝ := (2, 1)

-- Define the equations for O₂
def equation_O₂_tangent (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 12 - 8 * Real.sqrt 2
def equation_O₂_intersect_1 (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 4
def equation_O₂_intersect_2 (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 20

-- Theorem for externally tangent case
theorem externally_tangent_case :
  (∀ x y, circle_O₁ x y → ¬equation_O₂_tangent x y) ∧
  (∃ x y, circle_O₁ x y ∧ equation_O₂_tangent x y) →
  ∀ x y, equation_O₂_tangent x y :=
sorry

-- Theorem for intersecting case
theorem intersecting_case (A B : ℝ × ℝ) :
  (A ≠ B) ∧
  (∀ x y, circle_O₁ x y ↔ ((x - A.1)^2 + (y - A.2)^2 = 0 ∨ (x - B.1)^2 + (y - B.2)^2 = 0)) ∧
  ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 8) →
  (∀ x y, equation_O₂_intersect_1 x y ∨ equation_O₂_intersect_2 x y) :=
sorry

end NUMINAMATH_CALUDE_externally_tangent_case_intersecting_case_l1717_171785


namespace NUMINAMATH_CALUDE_yuan_david_age_difference_l1717_171796

theorem yuan_david_age_difference : 
  ∀ (yuan_age david_age : ℕ),
    david_age = 7 →
    yuan_age = 2 * david_age →
    yuan_age - david_age = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_yuan_david_age_difference_l1717_171796


namespace NUMINAMATH_CALUDE_area_of_overlap_l1717_171776

/-- Represents a 30-60-90 triangle -/
structure Triangle30_60_90 where
  hypotenuse : ℝ
  shortLeg : ℝ
  longLeg : ℝ
  hypotenuse_eq : hypotenuse = 10
  shortLeg_eq : shortLeg = 5
  longLeg_eq : longLeg = 5 * Real.sqrt 3

/-- Represents the configuration of two overlapping 30-60-90 triangles -/
structure OverlappingTriangles where
  triangle1 : Triangle30_60_90
  triangle2 : Triangle30_60_90
  overlap_angle : ℝ
  overlap_angle_eq : overlap_angle = 60

/-- The theorem to be proved -/
theorem area_of_overlap (ot : OverlappingTriangles) :
  let base := 2 * ot.triangle1.shortLeg
  let height := ot.triangle1.longLeg
  base * height = 50 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_area_of_overlap_l1717_171776


namespace NUMINAMATH_CALUDE_equation_solutions_l1717_171768

theorem equation_solutions : ∀ x : ℝ,
  (x^2 - 3*x = 4 ↔ x = 4 ∨ x = -1) ∧
  (x*(x-2) + x - 2 = 0 ↔ x = 2 ∨ x = -1) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1717_171768


namespace NUMINAMATH_CALUDE_difference_of_squares_l1717_171753

theorem difference_of_squares : (535 : ℕ)^2 - (465 : ℕ)^2 = 70000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1717_171753


namespace NUMINAMATH_CALUDE_decimal_existence_l1717_171736

theorem decimal_existence :
  (∃ (a b : ℚ), 3.5 < a ∧ a < 3.6 ∧ 3.5 < b ∧ b < 3.6 ∧ a ≠ b) ∧
  (∃ (x y z : ℚ), 0 < x ∧ x < 0.1 ∧ 0 < y ∧ y < 0.1 ∧ 0 < z ∧ z < 0.1 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z) :=
by sorry

end NUMINAMATH_CALUDE_decimal_existence_l1717_171736


namespace NUMINAMATH_CALUDE_asymptotic_stability_l1717_171765

noncomputable section

/-- The system of differential equations -/
def system (x y : ℝ) : ℝ × ℝ :=
  (y - x/2 - x*y^3/2, -y - 2*x + x^2*y^2)

/-- The Lyapunov function candidate -/
def V (x y : ℝ) : ℝ :=
  2*x^2 + y^2

/-- The time derivative of V along the system trajectories -/
def dVdt (x y : ℝ) : ℝ :=
  let (dx, dy) := system x y
  4*x*dx + 2*y*dy

theorem asymptotic_stability :
  ∃ δ > 0, ∀ x y : ℝ, x^2 + y^2 < δ^2 →
    (∀ t : ℝ, t ≥ 0 → 
      let (xt, yt) := system x y
      V xt yt ≤ V x y ∧ (x ≠ 0 ∨ y ≠ 0 → V xt yt < V x y)) ∧
    (∀ ε > 0, ∃ T : ℝ, T > 0 → 
      let (xT, yT) := system x y
      xT^2 + yT^2 < ε^2) :=
sorry

end

end NUMINAMATH_CALUDE_asymptotic_stability_l1717_171765


namespace NUMINAMATH_CALUDE_smallest_prime_sum_of_three_composites_l1717_171769

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

theorem smallest_prime_sum_of_three_composites : 
  ∀ p : ℕ, Prime p → 
    (∃ a b c : ℕ, is_composite a ∧ is_composite b ∧ is_composite c ∧ 
      a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ p = a + b + c) → 
    p ≥ 19 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_sum_of_three_composites_l1717_171769


namespace NUMINAMATH_CALUDE_complement_of_at_least_two_defective_l1717_171709

/-- The number of products inspected --/
def n : ℕ := 10

/-- Event A: at least two defective products --/
def event_A (x : ℕ) : Prop := x ≥ 2

/-- The complementary event of A --/
def complement_A (x : ℕ) : Prop := x ≤ 1

/-- Theorem stating that the complement of "at least two defective products" 
    is "at most one defective product" --/
theorem complement_of_at_least_two_defective :
  ∀ x : ℕ, x ≤ n → (¬ event_A x ↔ complement_A x) := by sorry

end NUMINAMATH_CALUDE_complement_of_at_least_two_defective_l1717_171709


namespace NUMINAMATH_CALUDE_binomial_seven_four_l1717_171772

theorem binomial_seven_four : Nat.choose 7 4 = 35 := by
  sorry

end NUMINAMATH_CALUDE_binomial_seven_four_l1717_171772


namespace NUMINAMATH_CALUDE_max_value_f_l1717_171767

theorem max_value_f (x : ℝ) (h : x < 3) : 
  (x^2 - 3*x + 4) / (x - 3) ≤ -1 := by sorry

end NUMINAMATH_CALUDE_max_value_f_l1717_171767


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l1717_171759

noncomputable def f (x : ℝ) := (2 * x + 1) * Real.exp x

theorem f_derivative_at_zero : 
  deriv f 0 = 3 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l1717_171759


namespace NUMINAMATH_CALUDE_tank_volume_in_cubic_yards_l1717_171756

/-- Conversion factor from cubic feet to cubic yards -/
def cubicFeetToCubicYards : ℚ := 1 / 27

/-- Volume of the tank in cubic feet -/
def tankVolumeCubicFeet : ℚ := 216

/-- Theorem: The volume of the tank in cubic yards is 8 -/
theorem tank_volume_in_cubic_yards :
  tankVolumeCubicFeet * cubicFeetToCubicYards = 8 := by
  sorry

end NUMINAMATH_CALUDE_tank_volume_in_cubic_yards_l1717_171756


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l1717_171729

theorem geometric_sequence_first_term 
  (a : ℕ → ℝ) 
  (h1 : ∀ n ≥ 1, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h2 : a 4 = 16) 
  (h3 : a 5 = 32) 
  (h4 : a 6 = 64) : 
  a 1 = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l1717_171729


namespace NUMINAMATH_CALUDE_third_flip_expected_value_l1717_171783

/-- The expected value of a biased coin flip -/
def expected_value (p_heads : ℚ) (win_amount : ℚ) (loss_amount : ℚ) : ℚ :=
  p_heads * win_amount + (1 - p_heads) * (-loss_amount)

theorem third_flip_expected_value :
  let p_heads : ℚ := 2/5
  let win_amount : ℚ := 4
  let loss_amount : ℚ := 6  -- doubled loss amount due to previous two tails
  expected_value p_heads win_amount loss_amount = -2 := by
sorry

end NUMINAMATH_CALUDE_third_flip_expected_value_l1717_171783


namespace NUMINAMATH_CALUDE_max_k_value_l1717_171708

theorem max_k_value (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0)
  (heq : 6 = k^2 * (x^2 / y^2 + y^2 / x^2) + k * (x / y + y / x)) :
  k ≤ 3/2 ∧ ∃ (x' y' : ℝ), x' > 0 ∧ y' > 0 ∧ 
    6 = (3/2)^2 * (x'^2 / y'^2 + y'^2 / x'^2) + (3/2) * (x' / y' + y' / x') :=
sorry

end NUMINAMATH_CALUDE_max_k_value_l1717_171708


namespace NUMINAMATH_CALUDE_range_of_m_l1717_171716

theorem range_of_m (x y m : ℝ) : 
  (x + 2*y = 4*m) → 
  (2*x + y = 2*m + 1) → 
  (-1 < x - y) → 
  (x - y < 0) → 
  (1/2 < m ∧ m < 1) := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l1717_171716


namespace NUMINAMATH_CALUDE_branch_A_more_profitable_l1717_171761

/-- Represents a branch of the factory -/
inductive Branch
| A
| B

/-- Represents the grade of a product -/
inductive Grade
| A
| B
| C
| D

/-- Processing fee for each grade -/
def processingFee (g : Grade) : ℝ :=
  match g with
  | Grade.A => 90
  | Grade.B => 50
  | Grade.C => 20
  | Grade.D => -50

/-- Processing cost for each branch -/
def processingCost (b : Branch) : ℝ :=
  match b with
  | Branch.A => 25
  | Branch.B => 20

/-- Frequency distribution for each branch and grade -/
def frequency (b : Branch) (g : Grade) : ℝ :=
  match b, g with
  | Branch.A, Grade.A => 0.4
  | Branch.A, Grade.B => 0.2
  | Branch.A, Grade.C => 0.2
  | Branch.A, Grade.D => 0.2
  | Branch.B, Grade.A => 0.28
  | Branch.B, Grade.B => 0.17
  | Branch.B, Grade.C => 0.34
  | Branch.B, Grade.D => 0.21

/-- Average profit for a branch -/
def averageProfit (b : Branch) : ℝ :=
  (processingFee Grade.A - processingCost b) * frequency b Grade.A +
  (processingFee Grade.B - processingCost b) * frequency b Grade.B +
  (processingFee Grade.C - processingCost b) * frequency b Grade.C +
  (processingFee Grade.D - processingCost b) * frequency b Grade.D

/-- Theorem stating that Branch A has higher average profit than Branch B -/
theorem branch_A_more_profitable :
  averageProfit Branch.A > averageProfit Branch.B :=
by sorry


end NUMINAMATH_CALUDE_branch_A_more_profitable_l1717_171761
