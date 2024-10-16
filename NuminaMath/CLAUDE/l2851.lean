import Mathlib

namespace NUMINAMATH_CALUDE_even_power_minus_one_factorization_l2851_285116

theorem even_power_minus_one_factorization (n : ℕ) (h1 : Even n) (h2 : n > 4) :
  ∃ (a b c : ℕ), (a > 1 ∧ b > 1 ∧ c > 1) ∧ (2^n - 1 = a * b * c) :=
sorry

end NUMINAMATH_CALUDE_even_power_minus_one_factorization_l2851_285116


namespace NUMINAMATH_CALUDE_total_removed_volume_is_one_forty_eighth_l2851_285190

/-- A unit cube with corners cut off such that each face forms a regular hexagon -/
structure ModifiedCube where
  /-- The original cube is a unit cube -/
  is_unit_cube : Bool
  /-- Each face of the modified cube forms a regular hexagon -/
  faces_are_hexagons : Bool

/-- The volume of a single removed triangular pyramid -/
def single_pyramid_volume (cube : ModifiedCube) : ℝ :=
  sorry

/-- The total number of removed triangular pyramids -/
def num_pyramids : ℕ := 8

/-- The total volume of all removed triangular pyramids -/
def total_removed_volume (cube : ModifiedCube) : ℝ :=
  (single_pyramid_volume cube) * (num_pyramids : ℝ)

/-- Theorem: The total volume of removed triangular pyramids is 1/48 -/
theorem total_removed_volume_is_one_forty_eighth (cube : ModifiedCube) :
  cube.is_unit_cube ∧ cube.faces_are_hexagons →
  total_removed_volume cube = 1 / 48 :=
sorry

end NUMINAMATH_CALUDE_total_removed_volume_is_one_forty_eighth_l2851_285190


namespace NUMINAMATH_CALUDE_inverse_f_sum_l2851_285102

-- Define the function f(x) = x|x|
def f (x : ℝ) : ℝ := x * abs x

-- State the theorem
theorem inverse_f_sum : ∃ y z : ℝ, f y = 9 ∧ f z = -81 ∧ y + z = -6 := by sorry

end NUMINAMATH_CALUDE_inverse_f_sum_l2851_285102


namespace NUMINAMATH_CALUDE_circplus_comm_circplus_not_scalar_mult_circplus_zero_circplus_self_circplus_pos_l2851_285167

-- Define the ⊕ operation
def circplus (x y : ℝ) : ℝ := |x - y|^2

-- Theorem statements
theorem circplus_comm (x y : ℝ) : circplus x y = circplus y x := by sorry

theorem circplus_not_scalar_mult (x y : ℝ) : 
  2 * (circplus x y) ≠ circplus (2 * x) (2 * y) := by sorry

theorem circplus_zero (x : ℝ) : circplus x 0 = x^2 := by sorry

theorem circplus_self (x : ℝ) : circplus x x = 0 := by sorry

theorem circplus_pos (x y : ℝ) : x ≠ y → circplus x y > 0 := by sorry

end NUMINAMATH_CALUDE_circplus_comm_circplus_not_scalar_mult_circplus_zero_circplus_self_circplus_pos_l2851_285167


namespace NUMINAMATH_CALUDE_g_difference_l2851_285106

/-- Given g(x) = 3x^3 - 4x + 5, prove that g(x + h) - g(x) = h(9x^2 + 9xh + 3h^2 - 4) for all real x and h -/
theorem g_difference (x h : ℝ) : 
  let g : ℝ → ℝ := fun x ↦ 3 * x^3 - 4 * x + 5
  g (x + h) - g x = h * (9 * x^2 + 9 * x * h + 3 * h^2 - 4) := by
  sorry

end NUMINAMATH_CALUDE_g_difference_l2851_285106


namespace NUMINAMATH_CALUDE_lcm_12_15_18_l2851_285101

theorem lcm_12_15_18 : Nat.lcm (Nat.lcm 12 15) 18 = 180 := by
  sorry

end NUMINAMATH_CALUDE_lcm_12_15_18_l2851_285101


namespace NUMINAMATH_CALUDE_pupusa_minimum_l2851_285141

theorem pupusa_minimum (a b : ℕ+) (h1 : a < 391) (h2 : Nat.lcm a b > Nat.lcm a 391) : 
  ∀ b' : ℕ+, (∃ a' : ℕ+, a' < 391 ∧ Nat.lcm a' b' > Nat.lcm a' 391) → b' ≥ 18 := by
sorry

end NUMINAMATH_CALUDE_pupusa_minimum_l2851_285141


namespace NUMINAMATH_CALUDE_cricket_bat_price_l2851_285118

/-- Calculates the final price of an item after two consecutive sales with given profit percentages -/
def finalPrice (initialCost : ℚ) (profit1 : ℚ) (profit2 : ℚ) : ℚ :=
  initialCost * (1 + profit1) * (1 + profit2)

/-- Theorem stating that a cricket bat initially costing $154, sold twice with 20% and 25% profit, results in a final price of $231 -/
theorem cricket_bat_price : 
  finalPrice 154 (20/100) (25/100) = 231 := by sorry

end NUMINAMATH_CALUDE_cricket_bat_price_l2851_285118


namespace NUMINAMATH_CALUDE_least_common_denominator_l2851_285149

theorem least_common_denominator : Nat.lcm 2 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 8 (Nat.lcm 9 11))))) = 3960 := by
  sorry

end NUMINAMATH_CALUDE_least_common_denominator_l2851_285149


namespace NUMINAMATH_CALUDE_eve_age_proof_l2851_285166

/-- Adam's current age -/
def adam_age : ℕ := 9

/-- Eve's current age -/
def eve_age : ℕ := 14

/-- Theorem stating Eve's age based on the given conditions -/
theorem eve_age_proof :
  (adam_age < eve_age) ∧
  (eve_age + 1 = 3 * (adam_age - 4)) ∧
  (adam_age = 9) →
  eve_age = 14 := by
sorry

end NUMINAMATH_CALUDE_eve_age_proof_l2851_285166


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l2851_285151

theorem sum_of_two_numbers : ∃ (x y : ℝ), 
  3 * x - y = 20 ∧ 
  y = 17 ∧ 
  x + y = 29.333333333333332 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l2851_285151


namespace NUMINAMATH_CALUDE_triangle_area_with_consecutive_integer_sides_and_height_l2851_285180

theorem triangle_area_with_consecutive_integer_sides_and_height :
  ∀ (a b c h : ℕ),
    a + 1 = b →
    b + 1 = c →
    c + 1 = h →
    (1 / 2 : ℚ) * b * h = 84 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_with_consecutive_integer_sides_and_height_l2851_285180


namespace NUMINAMATH_CALUDE_intersection_point_is_solution_l2851_285123

/-- The intersection point of two lines is the solution to a system of equations -/
theorem intersection_point_is_solution (x y : ℝ) :
  (y = 2*x + 1) ∧ (y = -x + 4) →  -- Given intersection point equations
  (x = 1 ∧ y = 3) →               -- Given intersection point
  (2*x - y = -1) ∧ (x + y = 4)    -- System of equations to prove
  := by sorry

end NUMINAMATH_CALUDE_intersection_point_is_solution_l2851_285123


namespace NUMINAMATH_CALUDE_junk_mail_total_l2851_285155

/-- Calculates the total number of junk mail pieces a mailman should give --/
theorem junk_mail_total (houses_per_block : ℕ) (num_blocks : ℕ) (mail_per_house : ℕ) : 
  houses_per_block = 50 → num_blocks = 3 → mail_per_house = 45 → 
  houses_per_block * num_blocks * mail_per_house = 6750 := by
  sorry

#check junk_mail_total

end NUMINAMATH_CALUDE_junk_mail_total_l2851_285155


namespace NUMINAMATH_CALUDE_milk_ratio_l2851_285107

def weekday_boxes : ℕ := 3
def saturday_boxes : ℕ := 2 * weekday_boxes
def total_boxes : ℕ := 30

def weekdays : ℕ := 5
def saturdays : ℕ := 1

def sunday_boxes : ℕ := total_boxes - (weekday_boxes * weekdays + saturday_boxes * saturdays)

theorem milk_ratio :
  (sunday_boxes : ℚ) / (weekday_boxes * weekdays : ℚ) = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_milk_ratio_l2851_285107


namespace NUMINAMATH_CALUDE_first_number_equation_l2851_285115

theorem first_number_equation (x : ℤ) : x + 7314 = 3362 + 13500 → x = 9548 := by
  sorry

end NUMINAMATH_CALUDE_first_number_equation_l2851_285115


namespace NUMINAMATH_CALUDE_k_range_l2851_285179

-- Define the function h
def h (x : ℝ) : ℝ := 5 * x - 3

-- Define the function k as a composition of h
def k (x : ℝ) : ℝ := h (h (h x))

-- State the theorem
theorem k_range :
  ∀ x ∈ Set.Icc (-1 : ℝ) 3,
  ∃ y ∈ Set.Icc (-218 : ℝ) 282,
  k x = y ∧
  ∀ z, k x = z → z ∈ Set.Icc (-218 : ℝ) 282 :=
sorry

end NUMINAMATH_CALUDE_k_range_l2851_285179


namespace NUMINAMATH_CALUDE_roots_sum_and_product_l2851_285154

theorem roots_sum_and_product : ∃ (r₁ r₂ : ℚ),
  (∀ x, (3 * x + 2) * (x - 5) + (3 * x + 2) * (x - 8) = 0 ↔ x = r₁ ∨ x = r₂) ∧
  r₁ + r₂ = 35 / 6 ∧
  r₁ * r₂ = -13 / 3 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_and_product_l2851_285154


namespace NUMINAMATH_CALUDE_system_solution_l2851_285163

theorem system_solution :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    (x₁^2 - 9*y₁^2 = 0 ∧ 2*x₁ - 3*y₁ = 6) ∧
    (x₂^2 - 9*y₂^2 = 0 ∧ 2*x₂ - 3*y₂ = 6) ∧
    x₁ = 6 ∧ y₁ = 2 ∧ x₂ = 2 ∧ y₂ = -2/3 ∧
    ∀ (x y : ℝ), (x^2 - 9*y^2 = 0 ∧ 2*x - 3*y = 6) →
      ((x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂)) :=
by sorry


end NUMINAMATH_CALUDE_system_solution_l2851_285163


namespace NUMINAMATH_CALUDE_marcus_percentage_of_team_points_l2851_285160

/-- Represents the number of points for each type of goal -/
def threePointValue : ℕ := 3
def twoPointValue : ℕ := 2

/-- Represents the number of goals Marcus scored -/
def marcusThreePointers : ℕ := 5
def marcusTwoPointers : ℕ := 10

/-- Represents the total points scored by the team -/
def teamTotalPoints : ℕ := 70

/-- Calculates the total points scored by Marcus -/
def marcusTotalPoints : ℕ :=
  marcusThreePointers * threePointValue + marcusTwoPointers * twoPointValue

/-- Theorem: Marcus scored 50% of the team's total points -/
theorem marcus_percentage_of_team_points :
  (marcusTotalPoints : ℚ) / teamTotalPoints = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_marcus_percentage_of_team_points_l2851_285160


namespace NUMINAMATH_CALUDE_convention_center_tables_l2851_285161

/-- Converts a number from base 7 to base 10 -/
def base7ToBase10 (n : Nat) : Nat :=
  (n / 100) * 7^2 + ((n / 10) % 10) * 7^1 + (n % 10) * 7^0

/-- Calculates the number of tables needed given the total number of people and people per table -/
def calculateTables (totalPeople : Nat) (peoplePerTable : Nat) : Nat :=
  totalPeople / peoplePerTable

theorem convention_center_tables :
  let seatingCapacityBase7 : Nat := 315
  let peoplePerTable : Nat := 3
  let totalPeopleBase10 : Nat := base7ToBase10 seatingCapacityBase7
  calculateTables totalPeopleBase10 peoplePerTable = 53 := by
  sorry

end NUMINAMATH_CALUDE_convention_center_tables_l2851_285161


namespace NUMINAMATH_CALUDE_experts_win_probability_l2851_285196

/-- The probability of Experts winning a single round -/
def p : ℝ := 0.6

/-- The probability of Viewers winning a single round -/
def q : ℝ := 1 - p

/-- The current score of Experts -/
def expert_score : ℕ := 3

/-- The current score of Viewers -/
def viewer_score : ℕ := 4

/-- The number of rounds needed to win the game -/
def winning_score : ℕ := 6

/-- The probability that the Experts will eventually win the game -/
theorem experts_win_probability : 
  p^4 + 4 * p^3 * q = 0.4752 := by sorry

end NUMINAMATH_CALUDE_experts_win_probability_l2851_285196


namespace NUMINAMATH_CALUDE_existence_of_complex_root_l2851_285171

theorem existence_of_complex_root (n : ℕ) (A : Finset ℕ) (hn : n ≥ 2) (hA : A.card = n) :
  ∃ z : ℂ, Complex.abs z = 1 ∧ Complex.abs (A.sum (λ a => z^a)) = Real.sqrt (n - 2) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_complex_root_l2851_285171


namespace NUMINAMATH_CALUDE_square_areas_sum_l2851_285100

theorem square_areas_sum (a : ℝ) (h1 : a > 0) (h2 : (a + 4)^2 - a^2 = 80) : 
  a^2 + (a + 4)^2 = 208 := by
  sorry

end NUMINAMATH_CALUDE_square_areas_sum_l2851_285100


namespace NUMINAMATH_CALUDE_roots_sum_of_powers_l2851_285117

theorem roots_sum_of_powers (γ δ : ℝ) : 
  γ^2 - 3*γ - 2 = 0 → δ^2 - 3*δ - 2 = 0 → 3*γ^4 + 7*δ^3 = -135 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_of_powers_l2851_285117


namespace NUMINAMATH_CALUDE_square_perimeter_from_area_l2851_285181

-- Define a square with a given area
def Square (area : ℝ) := {side : ℝ // side^2 = area}

-- Define the perimeter of a square
def perimeter (s : Square area) : ℝ := 4 * s.val

-- Theorem statement
theorem square_perimeter_from_area (s : Square 225) : 
  perimeter s = 60 := by sorry

end NUMINAMATH_CALUDE_square_perimeter_from_area_l2851_285181


namespace NUMINAMATH_CALUDE_base_conversion_equality_l2851_285114

theorem base_conversion_equality (b : ℕ) : 
  (1 * 5^2 + 3 * 5^1 + 2 * 5^0 = 2 * b^2 + 1 * b^1 + 0 * b^0) → 
  (b > 0) → 
  (b = 4) := by
sorry

end NUMINAMATH_CALUDE_base_conversion_equality_l2851_285114


namespace NUMINAMATH_CALUDE_total_placards_taken_l2851_285112

/-- The number of placards taken by people entering a stadium -/
def placards_taken (people : ℕ) (placards_per_person : ℕ) : ℕ :=
  people * placards_per_person

/-- Theorem stating the total number of placards taken by 2841 people -/
theorem total_placards_taken :
  placards_taken 2841 2 = 5682 := by
  sorry

end NUMINAMATH_CALUDE_total_placards_taken_l2851_285112


namespace NUMINAMATH_CALUDE_savings_calculation_l2851_285195

theorem savings_calculation (savings : ℚ) (tv_cost : ℚ) 
  (h1 : tv_cost = 150)
  (h2 : (1 : ℚ) / 4 * savings = tv_cost) : 
  savings = 600 := by
  sorry

end NUMINAMATH_CALUDE_savings_calculation_l2851_285195


namespace NUMINAMATH_CALUDE_sum_in_base7_l2851_285105

/-- Converts a base 7 number to its decimal (base 10) representation -/
def toDecimal (n : List Nat) : Nat :=
  n.enum.foldr (fun (i, d) acc => acc + d * (7 ^ i)) 0

/-- Converts a decimal (base 10) number to its base 7 representation -/
def toBase7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
    aux n []

theorem sum_in_base7 :
  let a := [2, 3, 4]  -- 432₇
  let b := [4, 5]     -- 54₇
  let c := [6]        -- 6₇
  let sum := toBase7 (toDecimal a + toDecimal b + toDecimal c)
  sum = [5, 2, 5] := by sorry

end NUMINAMATH_CALUDE_sum_in_base7_l2851_285105


namespace NUMINAMATH_CALUDE_boys_who_bought_balloons_l2851_285110

/-- The number of balloons in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of balloons the clown initially had -/
def initial_dozens : ℕ := 3

/-- The number of girls who bought a balloon -/
def girls_bought : ℕ := 12

/-- The number of balloons the clown has left after sales -/
def remaining_balloons : ℕ := 21

/-- The number of boys who bought a balloon -/
def boys_bought : ℕ := initial_dozens * dozen - remaining_balloons - girls_bought

theorem boys_who_bought_balloons :
  boys_bought = 3 := by sorry

end NUMINAMATH_CALUDE_boys_who_bought_balloons_l2851_285110


namespace NUMINAMATH_CALUDE_negation_of_absolute_value_less_than_zero_is_true_l2851_285176

theorem negation_of_absolute_value_less_than_zero_is_true : 
  ¬(∃ x : ℝ, |x - 1| < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_absolute_value_less_than_zero_is_true_l2851_285176


namespace NUMINAMATH_CALUDE_oil_mixture_price_l2851_285121

theorem oil_mixture_price (x y z : ℝ) 
  (volume_constraint : x + y + z = 23.5)
  (cost_constraint : 55 * x + 70 * y + 82 * z = 65 * 23.5) :
  (55 * x + 70 * y + 82 * z) / (x + y + z) = 65 := by
  sorry

end NUMINAMATH_CALUDE_oil_mixture_price_l2851_285121


namespace NUMINAMATH_CALUDE_impossible_30_cents_with_5_coins_l2851_285173

def coin_values : List ℕ := [1, 5, 10, 25, 50]

theorem impossible_30_cents_with_5_coins :
  ¬ ∃ (coins : List ℕ), 
    coins.length = 5 ∧ 
    (∀ c ∈ coins, c ∈ coin_values) ∧ 
    coins.sum = 30 :=
by sorry

end NUMINAMATH_CALUDE_impossible_30_cents_with_5_coins_l2851_285173


namespace NUMINAMATH_CALUDE_complex_magnitude_l2851_285136

theorem complex_magnitude (z : ℂ) (h : (1 - 2*I)*z = 3 + I) : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l2851_285136


namespace NUMINAMATH_CALUDE_circle_area_from_polar_equation_l2851_285147

/-- The area of the circle described by the polar equation r = 3 cos θ - 4 sin θ is 25π/4 -/
theorem circle_area_from_polar_equation (θ : ℝ) :
  let r := 3 * Real.cos θ - 4 * Real.sin θ
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ (x y : ℝ), (x - center.1)^2 + (y - center.2)^2 = radius^2 ↔ 
      ∃ θ, x = r * Real.cos θ ∧ y = r * Real.sin θ) ∧
    π * radius^2 = 25 * π / 4 :=
by sorry

end NUMINAMATH_CALUDE_circle_area_from_polar_equation_l2851_285147


namespace NUMINAMATH_CALUDE_luke_sticker_problem_l2851_285144

/-- The number of stickers Luke used to decorate the greeting card -/
def stickers_used_for_card (initial : ℕ) (bought : ℕ) (birthday : ℕ) (given_to_sister : ℕ) (final : ℕ) : ℕ :=
  initial + bought + birthday - given_to_sister - final

/-- Theorem stating the number of stickers Luke used for the greeting card -/
theorem luke_sticker_problem :
  stickers_used_for_card 20 12 20 5 39 = 8 := by
  sorry

end NUMINAMATH_CALUDE_luke_sticker_problem_l2851_285144


namespace NUMINAMATH_CALUDE_utility_value_sets_l2851_285153

theorem utility_value_sets (A B : Set α) (h : B ⊆ A) : A ∪ B = A := by
  sorry

end NUMINAMATH_CALUDE_utility_value_sets_l2851_285153


namespace NUMINAMATH_CALUDE_parallelogram_area_gt_one_l2851_285145

/-- A lattice point in 2D space -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A parallelogram defined by four lattice points -/
structure Parallelogram where
  v1 : LatticePoint
  v2 : LatticePoint
  v3 : LatticePoint
  v4 : LatticePoint

/-- Checks if a point is inside or on the sides of a parallelogram -/
def isInsideOrOnSides (p : LatticePoint) (pg : Parallelogram) : Prop :=
  sorry

/-- Calculates the area of a parallelogram -/
def area (pg : Parallelogram) : ℝ :=
  sorry

/-- Theorem: The area of a parallelogram with vertices at lattice points 
    and at least one additional lattice point inside or on its sides is greater than 1 -/
theorem parallelogram_area_gt_one (pg : Parallelogram) 
  (h : ∃ p : LatticePoint, p ≠ pg.v1 ∧ p ≠ pg.v2 ∧ p ≠ pg.v3 ∧ p ≠ pg.v4 ∧ isInsideOrOnSides p pg) : 
  area pg > 1 :=
sorry

end NUMINAMATH_CALUDE_parallelogram_area_gt_one_l2851_285145


namespace NUMINAMATH_CALUDE_zebra_stripes_l2851_285109

theorem zebra_stripes (w n b : ℕ) : 
  w + n = b + 1 →  -- Total black stripes (wide + narrow) is one more than white stripes
  b = w + 7 →      -- Number of white stripes is 7 more than wide black stripes
  n = 8 :=         -- Number of narrow black stripes is 8
by sorry

end NUMINAMATH_CALUDE_zebra_stripes_l2851_285109


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l2851_285158

theorem quadratic_inequality_condition (a : ℝ) :
  ((∀ x : ℝ, x^2 - 2*a*x + a > 0) → (0 ≤ a ∧ a ≤ 1)) ∧
  ¬((0 ≤ a ∧ a ≤ 1) → (∀ x : ℝ, x^2 - 2*a*x + a > 0)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l2851_285158


namespace NUMINAMATH_CALUDE_fourth_grade_students_l2851_285165

/-- The number of students in fourth grade at the end of the year -/
def final_student_count (initial : ℕ) (left : ℕ) (new : ℕ) : ℕ :=
  initial - left + new

/-- Theorem stating that the final number of students is 48 -/
theorem fourth_grade_students :
  final_student_count 10 4 42 = 48 := by
  sorry

end NUMINAMATH_CALUDE_fourth_grade_students_l2851_285165


namespace NUMINAMATH_CALUDE_additional_class_choices_l2851_285142

def total_classes : ℕ := 10
def compulsory_classes : ℕ := 1
def total_classes_to_take : ℕ := 4

theorem additional_class_choices : 
  Nat.choose (total_classes - compulsory_classes) (total_classes_to_take - compulsory_classes) = 84 := by
  sorry

end NUMINAMATH_CALUDE_additional_class_choices_l2851_285142


namespace NUMINAMATH_CALUDE_completing_square_transform_l2851_285192

theorem completing_square_transform (x : ℝ) :
  (2 * x^2 - 4 * x - 3 = 0) ↔ ((x - 1)^2 - 5/2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_completing_square_transform_l2851_285192


namespace NUMINAMATH_CALUDE_no_real_solutions_l2851_285157

theorem no_real_solutions : ¬∃ x : ℝ, (2*x - 10*x + 24)^2 + 4 = -2*|x| := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l2851_285157


namespace NUMINAMATH_CALUDE_parabola_focus_equation_l2851_285138

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a parabola in standard form -/
inductive Parabola
  | VertexAtOrigin (p : ℝ) : Parabola
  | FocusOnXAxis (p : ℝ) : Parabola

/-- Function to check if a point is on a line -/
def isPointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Function to check if a point is on the x-axis -/
def isPointOnXAxis (p : Point) : Prop :=
  p.y = 0

/-- Function to check if a point is on the y-axis -/
def isPointOnYAxis (p : Point) : Prop :=
  p.x = 0

/-- Theorem stating the relationship between the focus of a parabola and its equation -/
theorem parabola_focus_equation (l : Line) (f : Point) :
  (l.a = 3 ∧ l.b = -4 ∧ l.c = -12) →
  isPointOnLine f l →
  (isPointOnXAxis f ∨ isPointOnYAxis f) →
  (∃ p : Parabola, p = Parabola.VertexAtOrigin (-12) ∨ p = Parabola.FocusOnXAxis 8) :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_equation_l2851_285138


namespace NUMINAMATH_CALUDE_min_negations_for_zero_sum_l2851_285129

def clock_numbers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

def sum_list (l : List ℤ) : ℤ := l.foldl (· + ·) 0

def negate_elements (l : List ℕ) (indices : List ℕ) : List ℤ :=
  l.enum.map (fun (i, x) => if i + 1 ∈ indices then -x else x)

theorem min_negations_for_zero_sum :
  ∃ (indices : List ℕ),
    (indices.length = 4) ∧
    (sum_list (negate_elements clock_numbers indices) = 0) ∧
    (∀ (other_indices : List ℕ),
      sum_list (negate_elements clock_numbers other_indices) = 0 →
      other_indices.length ≥ 4) :=
sorry

end NUMINAMATH_CALUDE_min_negations_for_zero_sum_l2851_285129


namespace NUMINAMATH_CALUDE_cookies_per_person_l2851_285124

/-- The number of cookies in a dozen --/
def dozen : ℕ := 12

/-- The number of batches Beth bakes --/
def batches : ℕ := 4

/-- The number of dozens per batch --/
def dozens_per_batch : ℕ := 2

/-- The number of people sharing the cookies --/
def people : ℕ := 16

/-- Theorem: Each person consumes 6 cookies when 4 batches of 2 dozen cookies are shared equally among 16 people --/
theorem cookies_per_person :
  (batches * dozens_per_batch * dozen) / people = 6 := by
  sorry

end NUMINAMATH_CALUDE_cookies_per_person_l2851_285124


namespace NUMINAMATH_CALUDE_complex_number_modulus_l2851_285113

theorem complex_number_modulus (i : ℂ) (h : i * i = -1) :
  Complex.abs (2 * i / (1 - i)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_modulus_l2851_285113


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2851_285135

theorem imaginary_part_of_z (z : ℂ) (h : (z + 1) / (1 - Complex.I) = Complex.I) : 
  Complex.im z = 1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2851_285135


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_negative_three_l2851_285150

theorem sqrt_expression_equals_negative_three :
  Real.sqrt 12 - Real.sqrt 3 * (2 + Real.sqrt 3) = -3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_negative_three_l2851_285150


namespace NUMINAMATH_CALUDE_turtle_problem_l2851_285185

theorem turtle_problem (initial_turtles : ℕ) : initial_turtles = 9 →
  let additional_turtles := 3 * initial_turtles - 2
  let total_turtles := initial_turtles + additional_turtles
  let remaining_turtles := total_turtles / 2
  remaining_turtles = 17 := by
sorry

end NUMINAMATH_CALUDE_turtle_problem_l2851_285185


namespace NUMINAMATH_CALUDE_max_students_l2851_285197

theorem max_students (n : ℕ) : n < 100 ∧ n % 9 = 4 ∧ n % 7 = 3 → n ≤ 94 := by
  sorry

end NUMINAMATH_CALUDE_max_students_l2851_285197


namespace NUMINAMATH_CALUDE_average_children_in_families_with_children_l2851_285199

theorem average_children_in_families_with_children 
  (total_families : ℕ) 
  (average_all : ℚ) 
  (childless_families : ℕ) 
  (h1 : total_families = 12)
  (h2 : average_all = 5/2)
  (h3 : childless_families = 2) :
  (total_families * average_all) / (total_families - childless_families) = 3 := by
sorry

end NUMINAMATH_CALUDE_average_children_in_families_with_children_l2851_285199


namespace NUMINAMATH_CALUDE_proposition_relationship_l2851_285104

theorem proposition_relationship (a b : ℝ) : 
  ¬(((a + b ≠ 4) → (a ≠ 1 ∧ b ≠ 3)) ∧ ((a ≠ 1 ∧ b ≠ 3) → (a + b ≠ 4))) :=
by sorry

end NUMINAMATH_CALUDE_proposition_relationship_l2851_285104


namespace NUMINAMATH_CALUDE_complex_magnitude_product_l2851_285194

theorem complex_magnitude_product : Complex.abs ((7 - 4*I) * (3 + 11*I)) = Real.sqrt 8450 := by sorry

end NUMINAMATH_CALUDE_complex_magnitude_product_l2851_285194


namespace NUMINAMATH_CALUDE_sum_of_squares_l2851_285131

theorem sum_of_squares (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (eq_a : a + a^2 = 1) (eq_b : b^2 + b^4 = 1) : a^2 + b^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l2851_285131


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l2851_285148

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 156) 
  (h2 : a*b + b*c + a*c = 50) : 
  a + b + c = 16 := by sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l2851_285148


namespace NUMINAMATH_CALUDE_distance_to_external_point_specific_distance_to_external_point_l2851_285143

/-- Given a circle with radius r and two tangents drawn from a common external point P
    with a sum length of s, the distance from the center O to P is sqrt(r^2 + (s/2)^2). -/
theorem distance_to_external_point (r s : ℝ) (hr : r > 0) (hs : s > 0) :
  let d := Real.sqrt (r^2 + (s/2)^2)
  d = Real.sqrt (r^2 + (s/2)^2) := by
  sorry

/-- For a circle with radius 11 and two tangents with sum length 120,
    the distance from the center to the external point is 61. -/
theorem specific_distance_to_external_point :
  let r : ℝ := 11
  let s : ℝ := 120
  let d := Real.sqrt (r^2 + (s/2)^2)
  d = 61 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_external_point_specific_distance_to_external_point_l2851_285143


namespace NUMINAMATH_CALUDE_triangle_area_approx_l2851_285108

/-- The area of a triangle with sides 30, 26, and 10 is approximately 126.72 -/
theorem triangle_area_approx : ∃ (area : ℝ), 
  let a : ℝ := 30
  let b : ℝ := 26
  let c : ℝ := 10
  let s : ℝ := (a + b + c) / 2
  area = Real.sqrt (s * (s - a) * (s - b) * (s - c)) ∧ 
  126.71 < area ∧ area < 126.73 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_approx_l2851_285108


namespace NUMINAMATH_CALUDE_log_inequality_equiv_solution_set_l2851_285198

def log_inequality (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ x ≠ 1 ∧ x ≠ y ∧ Real.log y / Real.log x ≥ (Real.log x + Real.log y) / (Real.log x - Real.log y)

def solution_set (x y : ℝ) : Prop :=
  (0 < x ∧ x < 1 ∧ 0 < y ∧ y < x) ∨ (x > 1 ∧ y > x)

theorem log_inequality_equiv_solution_set :
  ∀ x y : ℝ, log_inequality x y ↔ solution_set x y :=
sorry

end NUMINAMATH_CALUDE_log_inequality_equiv_solution_set_l2851_285198


namespace NUMINAMATH_CALUDE_square_construction_theorem_l2851_285188

/-- A line in a plane -/
structure Line :=
  (point : ℝ × ℝ)
  (direction : ℝ × ℝ)

/-- A square in a plane -/
structure Square :=
  (center : ℝ × ℝ)
  (side_length : ℝ)
  (rotation : ℝ)

/-- Check if a point lies on a line -/
def point_on_line (p : ℝ × ℝ) (l : Line) : Prop := sorry

/-- Check if a line intersects a square (including its extensions) -/
def line_intersects_square (l : Line) (s : Square) : Prop := sorry

/-- The main theorem -/
theorem square_construction_theorem 
  (L : Line) 
  (A B C D : ℝ × ℝ) 
  (h_distinct : A ≠ B ∧ B ≠ C ∧ C ≠ D)
  (h_order : point_on_line A L ∧ point_on_line B L ∧ point_on_line C L ∧ point_on_line D L) :
  ∃ (S : Square), 
    (∃ (p q : ℝ × ℝ), line_intersects_square L S ∧ p ≠ q ∧ 
      ((p = A ∧ q = B) ∨ (p = B ∧ q = A))) ∧
    (∃ (r s : ℝ × ℝ), line_intersects_square L S ∧ r ≠ s ∧ 
      ((r = C ∧ s = D) ∨ (r = D ∧ s = C))) :=
sorry

end NUMINAMATH_CALUDE_square_construction_theorem_l2851_285188


namespace NUMINAMATH_CALUDE_h_function_iff_strictly_increasing_l2851_285128

/-- Definition of an H function -/
def is_h_function (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → x₁ * f x₁ + x₂ * f x₂ > x₁ * f x₂ + x₂ * f x₁

/-- A function is strictly increasing -/
def strictly_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

/-- Theorem: A function is an H function if and only if it is strictly increasing -/
theorem h_function_iff_strictly_increasing (f : ℝ → ℝ) :
  is_h_function f ↔ strictly_increasing f :=
sorry

end NUMINAMATH_CALUDE_h_function_iff_strictly_increasing_l2851_285128


namespace NUMINAMATH_CALUDE_naH_required_for_h2O_l2851_285156

-- Define the molecules and their molar ratios in the reactions
structure Reaction :=
  (naH : ℚ) (h2O : ℚ) (naOH : ℚ) (h2 : ℚ)

-- Define the first step reaction
def firstStepReaction : Reaction :=
  { naH := 1, h2O := 1, naOH := 1, h2 := 1 }

-- Theorem stating that 1 mole of NaH is required to react with 1 mole of H2O
theorem naH_required_for_h2O :
  firstStepReaction.naH = firstStepReaction.h2O := by sorry

end NUMINAMATH_CALUDE_naH_required_for_h2O_l2851_285156


namespace NUMINAMATH_CALUDE_negation_of_universal_conditional_l2851_285127

theorem negation_of_universal_conditional (P : ℝ → Prop) :
  (¬∀ x : ℝ, x ≥ 2 → P x) ↔ (∃ x : ℝ, x < 2 ∧ ¬P x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_conditional_l2851_285127


namespace NUMINAMATH_CALUDE_video_game_sales_l2851_285184

/-- Calculates the money earned from selling working video games. -/
def money_earned (total_games : ℕ) (non_working_games : ℕ) (price_per_game : ℕ) : ℕ :=
  (total_games - non_working_games) * price_per_game

/-- Proves that given 10 total games, 8 non-working games, and a price of $6 per working game,
    the total money earned is $12. -/
theorem video_game_sales : money_earned 10 8 6 = 12 := by
  sorry

end NUMINAMATH_CALUDE_video_game_sales_l2851_285184


namespace NUMINAMATH_CALUDE_motel_flat_fee_calculation_l2851_285177

/-- A motel charging system with a flat fee for the first night and a fixed amount for additional nights. -/
structure MotelCharge where
  flatFee : ℕ  -- Flat fee for the first night
  nightlyRate : ℕ  -- Fixed amount for each additional night

/-- Calculates the total cost for a given number of nights -/
def totalCost (charge : MotelCharge) (nights : ℕ) : ℕ :=
  charge.flatFee + (nights - 1) * charge.nightlyRate

theorem motel_flat_fee_calculation (charge : MotelCharge) :
  totalCost charge 3 = 155 → totalCost charge 6 = 290 → charge.flatFee = 65 := by
  sorry

#check motel_flat_fee_calculation

end NUMINAMATH_CALUDE_motel_flat_fee_calculation_l2851_285177


namespace NUMINAMATH_CALUDE_point_relationship_l2851_285139

/-- Prove that for points A(-1/2, m) and B(2, n) lying on the line y = 3x + b, m < n. -/
theorem point_relationship (m n b : ℝ) : 
  ((-1/2 : ℝ), m) ∈ {(x, y) | y = 3*x + b} →
  ((2 : ℝ), n) ∈ {(x, y) | y = 3*x + b} →
  m < n := by sorry

end NUMINAMATH_CALUDE_point_relationship_l2851_285139


namespace NUMINAMATH_CALUDE_max_stable_angle_l2851_285159

/-- A sign consisting of two uniform legs attached by a frictionless hinge -/
structure Sign where
  μ : ℝ  -- coefficient of friction between the ground and the legs
  θ : ℝ  -- angle between the legs

/-- The condition for the sign to be in equilibrium -/
def is_stable (s : Sign) : Prop :=
  Real.tan (s.θ / 2) = 2 * s.μ

/-- Theorem stating the maximum angle for stability -/
theorem max_stable_angle (s : Sign) :
  is_stable s ↔ s.θ = Real.arctan (2 * s.μ) * 2 :=
sorry

end NUMINAMATH_CALUDE_max_stable_angle_l2851_285159


namespace NUMINAMATH_CALUDE_f_increasing_iff_a_in_range_l2851_285134

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 + 4*a*x else (2*a + 3)*x - 4*a + 5

theorem f_increasing_iff_a_in_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ a ∈ Set.Icc (1/2) (3/2) :=
sorry

end NUMINAMATH_CALUDE_f_increasing_iff_a_in_range_l2851_285134


namespace NUMINAMATH_CALUDE_increasing_quadratic_function_l2851_285137

/-- The function f(x) = x^2 - 2ax is increasing on [1, +∞) if and only if a ≤ 1 -/
theorem increasing_quadratic_function (a : ℝ) :
  (∀ x ≥ 1, Monotone (fun x => x^2 - 2*a*x)) ↔ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_increasing_quadratic_function_l2851_285137


namespace NUMINAMATH_CALUDE_monkey_nuts_problem_l2851_285125

theorem monkey_nuts_problem (n : ℕ) (x : ℕ) : 
  n > 1 → 
  x > 1 → 
  n * x - n * (n - 1) = 35 → 
  x = 11 :=
by sorry

end NUMINAMATH_CALUDE_monkey_nuts_problem_l2851_285125


namespace NUMINAMATH_CALUDE_total_books_total_books_specific_l2851_285178

theorem total_books (stu_books : ℕ) (albert_multiplier : ℕ) : ℕ :=
  let albert_books := albert_multiplier * stu_books
  stu_books + albert_books

theorem total_books_specific : total_books 9 4 = 45 := by
  sorry

end NUMINAMATH_CALUDE_total_books_total_books_specific_l2851_285178


namespace NUMINAMATH_CALUDE_range_of_sqrt_function_l2851_285189

theorem range_of_sqrt_function (x : ℝ) :
  (∃ y : ℝ, y = Real.sqrt (2 - x)) ↔ x ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_sqrt_function_l2851_285189


namespace NUMINAMATH_CALUDE_max_product_division_l2851_285164

theorem max_product_division (N : ℝ) (h : N > 0) :
  ∀ x : ℝ, 0 < x ∧ x < N → x * (N - x) ≤ (N / 2) * (N / 2) := by
  sorry

end NUMINAMATH_CALUDE_max_product_division_l2851_285164


namespace NUMINAMATH_CALUDE_triangle_angle_determinant_l2851_285119

/-- Given angles α, β, γ of a triangle, the determinant of the matrix
    | tan α   sin α cos α   1 |
    | tan β   sin β cos β   1 |
    | tan γ   sin γ cos γ   1 |
    is equal to 0. -/
theorem triangle_angle_determinant (α β γ : Real) 
  (h : α + β + γ = Real.pi) : 
  let M : Matrix (Fin 3) (Fin 3) ℝ := ![
    ![Real.tan α, Real.sin α * Real.cos α, 1],
    ![Real.tan β, Real.sin β * Real.cos β, 1],
    ![Real.tan γ, Real.sin γ * Real.cos γ, 1]
  ]
  Matrix.det M = 0 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_determinant_l2851_285119


namespace NUMINAMATH_CALUDE_cos_330_degrees_l2851_285168

theorem cos_330_degrees : Real.cos (330 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_330_degrees_l2851_285168


namespace NUMINAMATH_CALUDE_even_sum_probability_l2851_285103

-- Define the properties of the wheels
def first_wheel_sections : ℕ := 5
def first_wheel_even_sections : ℕ := 2
def first_wheel_odd_sections : ℕ := 3

def second_wheel_sections : ℕ := 4
def second_wheel_even_sections : ℕ := 1
def second_wheel_odd_sections : ℕ := 2
def second_wheel_special_sections : ℕ := 1

-- Define the probability of getting an even sum
def prob_even_sum : ℚ := 1/2

-- Theorem statement
theorem even_sum_probability :
  let p_even_first : ℚ := first_wheel_even_sections / first_wheel_sections
  let p_odd_first : ℚ := first_wheel_odd_sections / first_wheel_sections
  let p_even_second : ℚ := second_wheel_even_sections / second_wheel_sections
  let p_odd_second : ℚ := second_wheel_odd_sections / second_wheel_sections
  let p_special_second : ℚ := second_wheel_special_sections / second_wheel_sections
  
  -- Probability of both numbers being even (including special section effect)
  let p_both_even : ℚ := p_even_first * p_even_second + p_even_first * p_special_second
  
  -- Probability of both numbers being odd
  let p_both_odd : ℚ := p_odd_first * p_odd_second
  
  -- Total probability of an even sum
  p_both_even + p_both_odd = prob_even_sum :=
by sorry

end NUMINAMATH_CALUDE_even_sum_probability_l2851_285103


namespace NUMINAMATH_CALUDE_positive_difference_problem_l2851_285175

theorem positive_difference_problem : 
  ∀ x : ℝ, (33 + x) / 2 = 37 → |x - 33| = 8 := by
sorry

end NUMINAMATH_CALUDE_positive_difference_problem_l2851_285175


namespace NUMINAMATH_CALUDE_complex_equation_real_solution_l2851_285111

theorem complex_equation_real_solution (a : ℝ) : 
  (((a : ℂ) / (1 + Complex.I)) + ((1 + Complex.I) / 2)).im = 0 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_real_solution_l2851_285111


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2851_285183

-- Define the function type
def RealFunction := ℝ → ℝ

-- State the theorem
theorem functional_equation_solution (f : RealFunction) :
  (∀ x y : ℝ, |x| * (f y) + y * (f x) = f (x * y) + f (x^2) + f (f y)) →
  ∃ c : ℝ, c ≥ 0 ∧ ∀ x : ℝ, f x = c * (|x| - x) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2851_285183


namespace NUMINAMATH_CALUDE_polynomial_root_implies_k_l2851_285132

theorem polynomial_root_implies_k (k : ℝ) : 
  (3 : ℝ)^3 + k * 3 - 18 = 0 → k = -3 := by sorry

end NUMINAMATH_CALUDE_polynomial_root_implies_k_l2851_285132


namespace NUMINAMATH_CALUDE_average_increase_is_five_l2851_285193

/-- Represents a batsman's performance -/
structure BatsmanPerformance where
  innings : ℕ
  totalRuns : ℕ
  lastInningRuns : ℕ

/-- Calculates the average runs per inning -/
def average (bp : BatsmanPerformance) : ℚ :=
  bp.totalRuns / bp.innings

/-- Theorem: The increase in average is 5 runs -/
theorem average_increase_is_five (bp : BatsmanPerformance) 
  (h1 : bp.innings = 11)
  (h2 : bp.lastInningRuns = 85)
  (h3 : average bp = 35) :
  average bp - average { bp with 
    innings := bp.innings - 1,
    totalRuns := bp.totalRuns - bp.lastInningRuns
  } = 5 := by
  sorry

#check average_increase_is_five

end NUMINAMATH_CALUDE_average_increase_is_five_l2851_285193


namespace NUMINAMATH_CALUDE_calculate_savings_savings_calculation_l2851_285152

/-- Given a person's income and expenditure ratio, and their income, calculate their savings. -/
theorem calculate_savings (income_ratio : ℕ) (expenditure_ratio : ℕ) (income : ℕ) : ℕ :=
  let total_ratio := income_ratio + expenditure_ratio
  let expenditure := (expenditure_ratio * income) / income_ratio
  income - expenditure

/-- Prove that given a person's income and expenditure ratio of 10:7 and an income of Rs. 10000, the person's savings are Rs. 3000. -/
theorem savings_calculation : calculate_savings 10 7 10000 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_calculate_savings_savings_calculation_l2851_285152


namespace NUMINAMATH_CALUDE_remainder_problem_l2851_285140

theorem remainder_problem (x : ℤ) : x % 61 = 24 → x % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2851_285140


namespace NUMINAMATH_CALUDE_distribute_five_balls_three_boxes_l2851_285191

/-- The number of ways to distribute n distinguishable objects into k distinguishable containers -/
def distribute (n k : ℕ) : ℕ := k^n

/-- Theorem: There are 243 ways to distribute 5 distinguishable balls into 3 distinguishable boxes -/
theorem distribute_five_balls_three_boxes : distribute 5 3 = 243 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_balls_three_boxes_l2851_285191


namespace NUMINAMATH_CALUDE_side_c_length_l2851_285174

/-- Given a triangle ABC with side lengths a, b, and c, and angle C opposite side c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  C : ℝ

/-- The Law of Cosines for a triangle -/
def lawOfCosines (t : Triangle) : Prop :=
  t.c^2 = t.a^2 + t.b^2 - 2 * t.a * t.b * Real.cos t.C

theorem side_c_length (t : Triangle) 
  (ha : t.a = 2) 
  (hb : t.b = 1) 
  (hC : t.C = π / 3) -- 60° in radians
  (hlawCosines : lawOfCosines t) :
  t.c = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_side_c_length_l2851_285174


namespace NUMINAMATH_CALUDE_equation_solution_l2851_285172

theorem equation_solution : ∃ x : ℝ, (24 - 6 = 3 * x + 3) ∧ (x = 5) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2851_285172


namespace NUMINAMATH_CALUDE_satisfying_function_is_identity_l2851_285120

/-- A function satisfying the given conditions -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x > 0, f x > 0) ∧ 
  (f 1 = 1) ∧
  (∀ a b : ℝ, f (a + b) * (f a + f b) = 2 * f a * f b + a^2 + b^2)

/-- Theorem stating that any function satisfying the conditions is the identity function -/
theorem satisfying_function_is_identity (f : ℝ → ℝ) (hf : SatisfyingFunction f) : 
  ∀ x : ℝ, f x = x := by
  sorry

end NUMINAMATH_CALUDE_satisfying_function_is_identity_l2851_285120


namespace NUMINAMATH_CALUDE_inequality_of_positive_numbers_l2851_285122

theorem inequality_of_positive_numbers (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  a^2 * b + a * b^2 ≤ a^3 + b^3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_positive_numbers_l2851_285122


namespace NUMINAMATH_CALUDE_max_garden_area_l2851_285126

/-- Represents the dimensions of a rectangular garden. -/
structure GardenDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular garden given its dimensions. -/
def area (d : GardenDimensions) : ℝ := d.length * d.width

/-- Calculates the perimeter of a rectangular garden given its dimensions. -/
def perimeter (d : GardenDimensions) : ℝ := 2 * (d.length + d.width)

/-- Theorem: The maximum area of a rectangular garden with 320 feet of fencing
    and length no less than 100 feet is 6000 square feet, achieved when
    the length is 100 feet and the width is 60 feet. -/
theorem max_garden_area :
  ∃ (d : GardenDimensions),
    perimeter d = 320 ∧
    d.length ≥ 100 ∧
    area d = 6000 ∧
    (∀ (d' : GardenDimensions), perimeter d' = 320 ∧ d'.length ≥ 100 → area d' ≤ area d) :=
by sorry

end NUMINAMATH_CALUDE_max_garden_area_l2851_285126


namespace NUMINAMATH_CALUDE_counterexample_squared_inequality_l2851_285186

theorem counterexample_squared_inequality :
  ∃ (m n : ℝ), m > n ∧ m^2 ≤ n^2 := by sorry

end NUMINAMATH_CALUDE_counterexample_squared_inequality_l2851_285186


namespace NUMINAMATH_CALUDE_f_monotonicity_and_minimum_l2851_285146

def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + 2

theorem f_monotonicity_and_minimum (m : ℝ) (h : m > -1) :
  (∀ x y, x < y ∧ ((x < -1 ∧ y < -1) ∨ (x > 3 ∧ y > 3)) → f x < f y) ∧
  (∀ x y, -1 < x ∧ x < y ∧ y < 3 → f x > f y) ∧
  (∀ x ∈ Set.Icc (-1) m, 
    f x ≥ (if m ≤ 3 then f m else -25)) ∧
  (if m ≤ 3 
   then ∀ x ∈ Set.Icc (-1) m, f x ≥ m^3 - 3*m^2 - 9*m + 2
   else ∀ x ∈ Set.Icc (-1) m, f x ≥ -25) :=
sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_minimum_l2851_285146


namespace NUMINAMATH_CALUDE_power_division_simplification_l2851_285187

theorem power_division_simplification : (1000 : ℕ)^7 / (10 : ℕ)^17 = 10000 := by
  sorry

end NUMINAMATH_CALUDE_power_division_simplification_l2851_285187


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l2851_285133

theorem min_reciprocal_sum (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_sum : m + n = 2) :
  (1 / m + 1 / n) ≥ 2 ∧ ∃ (m₀ n₀ : ℝ), m₀ > 0 ∧ n₀ > 0 ∧ m₀ + n₀ = 2 ∧ 1 / m₀ + 1 / n₀ = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l2851_285133


namespace NUMINAMATH_CALUDE_arithmetic_seq_ratio_theorem_l2851_285182

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  seq_def : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

theorem arithmetic_seq_ratio_theorem (a b : ArithmeticSequence) :
  (∀ n : ℕ, sum_n a n / sum_n b n = (2 * n + 1) / (3 * n + 2)) →
  (a.a 2 + a.a 5 + a.a 17 + a.a 22) / (b.a 8 + b.a 10 + b.a 12 + b.a 16) = 45 / 68 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_seq_ratio_theorem_l2851_285182


namespace NUMINAMATH_CALUDE_M_on_y_axis_coordinates_l2851_285130

/-- A point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of a point on the y-axis -/
def on_y_axis (p : Point) : Prop := p.x = 0

/-- The point M with coordinates (m+1, m+3) -/
def M (m : ℝ) : Point :=
  { x := m + 1
    y := m + 3 }

/-- Theorem: If M(m+1, m+3) is on the y-axis, then its coordinates are (0, 2) -/
theorem M_on_y_axis_coordinates :
  ∀ m : ℝ, on_y_axis (M m) → M m = { x := 0, y := 2 } := by
  sorry

end NUMINAMATH_CALUDE_M_on_y_axis_coordinates_l2851_285130


namespace NUMINAMATH_CALUDE_number_difference_l2851_285170

theorem number_difference (x y : ℝ) 
  (sum_eq : x + y = 15) 
  (diff_eq : x - y = 10) 
  (square_diff_eq : x^2 - y^2 = 150) : 
  x - y = 10 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l2851_285170


namespace NUMINAMATH_CALUDE_alex_problem_count_l2851_285162

/-- Given that Alex has written 61 problems out of 187 total problems,
    this theorem proves that he needs to write 65 more problems
    to have written half of the total problems. -/
theorem alex_problem_count (alex_initial : ℕ) (total_initial : ℕ)
    (h1 : alex_initial = 61)
    (h2 : total_initial = 187) :
    ∃ x : ℕ, 2 * (alex_initial + x) = total_initial + x ∧ x = 65 := by
  sorry

end NUMINAMATH_CALUDE_alex_problem_count_l2851_285162


namespace NUMINAMATH_CALUDE_milkshake_cost_calculation_l2851_285169

/-- The cost of a milkshake given initial money, cupcake spending fraction, and remaining money --/
def milkshake_cost (initial : ℚ) (cupcake_fraction : ℚ) (remaining : ℚ) : ℚ :=
  initial - initial * cupcake_fraction - remaining

theorem milkshake_cost_calculation (initial : ℚ) (cupcake_fraction : ℚ) (remaining : ℚ) 
  (h1 : initial = 10)
  (h2 : cupcake_fraction = 1/5)
  (h3 : remaining = 3) :
  milkshake_cost initial cupcake_fraction remaining = 5 := by
  sorry

#eval milkshake_cost 10 (1/5) 3

end NUMINAMATH_CALUDE_milkshake_cost_calculation_l2851_285169
