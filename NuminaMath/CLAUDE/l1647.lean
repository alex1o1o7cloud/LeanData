import Mathlib

namespace f_2010_equals_zero_l1647_164747

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem f_2010_equals_zero
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_period : ∀ x, f (x + 3) = -f x) :
  f 2010 = 0 := by
  sorry

end f_2010_equals_zero_l1647_164747


namespace max_luggage_length_l1647_164791

theorem max_luggage_length : 
  ∀ (length width : ℝ),
  length > 0 ∧ width > 0 ∧
  length / width = 3 / 2 ∧
  length + width + 30 ≤ 160 →
  length ≤ 78 :=
by
  sorry

end max_luggage_length_l1647_164791


namespace largest_integer_less_than_100_remainder_5_mod_7_l1647_164725

theorem largest_integer_less_than_100_remainder_5_mod_7 : 
  ∀ n : ℤ, n < 100 ∧ n % 7 = 5 → n ≤ 96 :=
by
  sorry

end largest_integer_less_than_100_remainder_5_mod_7_l1647_164725


namespace cos_sum_inequality_l1647_164709

theorem cos_sum_inequality (x y : Real) :
  x ∈ Set.Icc 0 (Real.pi / 2) →
  y ∈ Set.Icc 0 Real.pi →
  Real.cos (x + y) ≤ Real.cos x * Real.cos y :=
by sorry

end cos_sum_inequality_l1647_164709


namespace flat_tax_calculation_l1647_164741

/-- Calculate the flat tax on a property with given characteristics -/
def calculate_flat_tax (condo_price condo_size barn_price barn_size detached_price detached_size
                        townhouse_price townhouse_size garage_price garage_size pool_price pool_size
                        tax_rate : ℝ) : ℝ :=
  let condo_value := condo_price * condo_size
  let barn_value := barn_price * barn_size
  let detached_value := detached_price * detached_size
  let townhouse_value := townhouse_price * townhouse_size
  let garage_value := garage_price * garage_size
  let pool_value := pool_price * pool_size
  let total_value := condo_value + barn_value + detached_value + townhouse_value + garage_value + pool_value
  total_value * tax_rate

theorem flat_tax_calculation :
  calculate_flat_tax 98 2400 84 1200 102 3500 96 2750 60 480 50 600 0.0125 = 12697.50 := by
  sorry

end flat_tax_calculation_l1647_164741


namespace line_moved_up_two_units_l1647_164782

/-- Represents a line in the 2D plane --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Moves a line vertically by a given amount --/
def moveLine (l : Line) (amount : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + amount }

/-- The theorem stating that moving y = 4x - 1 up by 2 units results in y = 4x + 1 --/
theorem line_moved_up_two_units :
  let original_line : Line := { slope := 4, intercept := -1 }
  let moved_line := moveLine original_line 2
  moved_line = { slope := 4, intercept := 1 } := by
  sorry

end line_moved_up_two_units_l1647_164782


namespace license_plate_theorem_l1647_164799

/-- The number of possible letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of possible digits -/
def num_digits : ℕ := 10

/-- The length of the license plate -/
def plate_length : ℕ := 5

/-- The number of letters at the start of the plate -/
def num_start_letters : ℕ := 2

/-- The number of digits at the end of the plate -/
def num_end_digits : ℕ := 3

/-- The number of ways to design a license plate with the given conditions -/
def license_plate_designs : ℕ :=
  num_letters * num_digits * (num_digits - 1)

theorem license_plate_theorem :
  license_plate_designs = 2340 :=
by sorry

end license_plate_theorem_l1647_164799


namespace absolute_value_sum_l1647_164756

theorem absolute_value_sum (a : ℝ) (h : 3 < a ∧ a < 4) : |a - 3| + |a - 4| = 1 := by
  sorry

end absolute_value_sum_l1647_164756


namespace smallest_x_squared_l1647_164775

/-- Represents a trapezoid ABCD with specific properties -/
structure Trapezoid where
  AB : ℝ
  CD : ℝ
  x : ℝ
  h : AB = 120 ∧ CD = 25

/-- A circle is tangent to AD if its center is on AB and touches AD -/
def is_tangent_circle (t : Trapezoid) (center : ℝ) : Prop :=
  0 ≤ center ∧ center ≤ t.AB ∧ ∃ (point : ℝ), 0 ≤ point ∧ point ≤ t.x

/-- The theorem stating the smallest possible value of x^2 -/
theorem smallest_x_squared (t : Trapezoid) : 
  (∃ center, is_tangent_circle t center) → 
  (∀ y, (∃ center, is_tangent_circle { AB := t.AB, CD := t.CD, x := y, h := t.h } center) → 
    t.x^2 ≤ y^2) → 
  t.x^2 = 3443.75 := by
  sorry

end smallest_x_squared_l1647_164775


namespace geometric_progression_naturals_l1647_164777

theorem geometric_progression_naturals (a₁ : ℕ) (q : ℚ) :
  (∃ (a₁₀ a₃₀ : ℕ), a₁₀ = a₁ * q^9 ∧ a₃₀ = a₁ * q^29) →
  ∃ (a₂₀ : ℕ), a₂₀ = a₁ * q^19 := by
sorry

end geometric_progression_naturals_l1647_164777


namespace circle_properties_l1647_164760

-- Define the circle's circumference
def circumference : ℝ := 36

-- Theorem statement
theorem circle_properties :
  let radius := circumference / (2 * Real.pi)
  let diameter := 2 * radius
  let area := Real.pi * radius^2
  (radius = 18 / Real.pi) ∧
  (diameter = 36 / Real.pi) ∧
  (area = 324 / Real.pi) := by
  sorry


end circle_properties_l1647_164760


namespace union_of_A_and_B_l1647_164765

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {|a + 1|, 3, 5}
def B (a : ℝ) : Set ℝ := {2*a + 1, a^2 + 2*a, a^2 + 2*a - 1}

-- Theorem statement
theorem union_of_A_and_B :
  ∃ a : ℝ, (A a ∩ B a = {2, 3}) → (A a ∪ B a = {-5, 2, 3, 5}) :=
by
  sorry

end union_of_A_and_B_l1647_164765


namespace blue_pens_count_l1647_164745

theorem blue_pens_count (total : ℕ) (difference : ℕ) (blue : ℕ) (red : ℕ) : 
  total = 82 → 
  difference = 6 → 
  total = blue + red → 
  blue = red + difference → 
  blue = 44 := by
sorry

end blue_pens_count_l1647_164745


namespace power_function_through_point_l1647_164737

theorem power_function_through_point (α : ℝ) : 
  (∀ x : ℝ, x > 0 → (fun x => x^α) x = x^α) → 
  (2 : ℝ)^α = 4 → 
  α = 2 := by sorry

end power_function_through_point_l1647_164737


namespace jellybean_probability_l1647_164755

def total_jellybeans : ℕ := 15
def red_jellybeans : ℕ := 6
def blue_jellybeans : ℕ := 3
def green_jellybeans : ℕ := 6
def picked_jellybeans : ℕ := 4

theorem jellybean_probability :
  let total_combinations := Nat.choose total_jellybeans picked_jellybeans
  let successful_combinations := Nat.choose red_jellybeans 2 * Nat.choose green_jellybeans 2
  (successful_combinations : ℚ) / total_combinations = 5 / 9 := by
  sorry

end jellybean_probability_l1647_164755


namespace tan_x_minus_pi_fourth_l1647_164708

theorem tan_x_minus_pi_fourth (x : ℝ) 
  (h1 : x ∈ Set.Ioo 0 π) 
  (h2 : Real.cos (2 * x - π / 2) = Real.sin x ^ 2) : 
  Real.tan (x - π / 4) = 1 / 3 := by
  sorry

end tan_x_minus_pi_fourth_l1647_164708


namespace poem_lines_proof_l1647_164738

/-- The number of lines added to the poem each month -/
def lines_per_month : ℕ := 3

/-- The number of months after which the poem will have 90 lines -/
def months : ℕ := 22

/-- The total number of lines in the poem after 22 months -/
def total_lines : ℕ := 90

/-- The current number of lines in the poem -/
def current_lines : ℕ := total_lines - (lines_per_month * months)

theorem poem_lines_proof : current_lines = 24 := by
  sorry

end poem_lines_proof_l1647_164738


namespace periodic_even_symmetric_function_l1647_164722

/-- A function f is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- A function f is symmetric about the line x = a if f(a - x) = f(a + x) for all x -/
def IsSymmetricAbout (f : ℝ → ℝ) (a : ℝ) : Prop := ∀ x, f (a - x) = f (a + x)

/-- A function f is periodic with period p if f(x + p) = f(x) for all x -/
def IsPeriodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem periodic_even_symmetric_function (f : ℝ → ℝ) 
  (h_nonconstant : ∃ x y, f x ≠ f y)
  (h_even : IsEven f)
  (h_symmetric : IsSymmetricAbout f (Real.sqrt 2 / 2)) :
  IsPeriodic f (Real.sqrt 2) := by
  sorry

end periodic_even_symmetric_function_l1647_164722


namespace hyperbola_asymptotes_l1647_164748

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0, b > 0,
    and eccentricity 2, its asymptotes have the equation y = ± √3 x -/
theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := 2  -- eccentricity
  let c := e * a  -- focal distance
  let hyperbola := fun (x y : ℝ) => x^2 / a^2 - y^2 / b^2 = 1
  let asymptote := fun (x y : ℝ) => y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x
  (∀ x y, hyperbola x y → b^2 = c^2 - a^2) →
  (∀ x y, asymptote x y ↔ (x / a - y / b = 0 ∨ x / a + y / b = 0)) :=
by sorry

end hyperbola_asymptotes_l1647_164748


namespace odd_sequence_concat_theorem_l1647_164718

def odd_sequence (n : ℕ) : List ℕ :=
  List.filter (λ x => x % 2 = 1) (List.range (n + 1))

def concat_digits (lst : List ℕ) : ℕ := sorry

def digit_sum (n : ℕ) : ℕ := sorry

theorem odd_sequence_concat_theorem :
  let seq := odd_sequence 103
  let A := concat_digits seq
  (Nat.digits 10 A).length = 101 ∧ A % 9 = 4 := by
  sorry

end odd_sequence_concat_theorem_l1647_164718


namespace remainder_base12_2543_div_9_l1647_164772

-- Define a function to convert base-12 to decimal
def base12ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (12 ^ (digits.length - 1 - i))) 0

-- Define the base-12 number 2543
def base12_2543 : List Nat := [2, 5, 4, 3]

-- Theorem statement
theorem remainder_base12_2543_div_9 :
  (base12ToDecimal base12_2543) % 9 = 8 := by
  sorry


end remainder_base12_2543_div_9_l1647_164772


namespace equation_solution_l1647_164703

theorem equation_solution (a b : ℝ) :
  (∀ x : ℝ, (a*x^2 + b*x - 5)*(a*x^2 + b*x + 25) + c = (a*x^2 + b*x + 10)^2) →
  c = 225 :=
by
  sorry

end equation_solution_l1647_164703


namespace sean_bought_two_soups_l1647_164734

/-- The cost of a single soda in dollars -/
def soda_cost : ℚ := 1

/-- The number of sodas Sean bought -/
def num_sodas : ℕ := 3

/-- The cost of a single soup in dollars -/
def soup_cost : ℚ := soda_cost * num_sodas

/-- The cost of the sandwich in dollars -/
def sandwich_cost : ℚ := 3 * soup_cost

/-- The total cost of all items in dollars -/
def total_cost : ℚ := 18

/-- The number of soups Sean bought -/
def num_soups : ℕ := 2

theorem sean_bought_two_soups :
  soda_cost * num_sodas + sandwich_cost + soup_cost * num_soups = total_cost :=
sorry

end sean_bought_two_soups_l1647_164734


namespace calculate_upstream_speed_l1647_164702

/-- Represents the speed of a man rowing in different water conditions -/
structure RowingSpeed where
  still : ℝ  -- Speed in still water
  downstream : ℝ  -- Speed downstream
  upstream : ℝ  -- Speed upstream

/-- Theorem: Given a man's speed in still water and downstream, calculate his upstream speed -/
theorem calculate_upstream_speed (speed : RowingSpeed) 
  (h1 : speed.still = 35)
  (h2 : speed.downstream = 40) : 
  speed.upstream = 30 := by
  sorry

#check calculate_upstream_speed

end calculate_upstream_speed_l1647_164702


namespace isosceles_triangle_23_perimeter_l1647_164752

-- Define an isosceles triangle with side lengths 2 and 3
structure IsoscelesTriangle23 where
  base : ℝ
  side : ℝ
  is_isosceles : (base = 2 ∧ side = 3) ∨ (base = 3 ∧ side = 2)

-- Define the perimeter of the triangle
def perimeter (t : IsoscelesTriangle23) : ℝ := t.base + 2 * t.side

-- Theorem statement
theorem isosceles_triangle_23_perimeter :
  ∀ t : IsoscelesTriangle23, perimeter t = 7 ∨ perimeter t = 8 := by
  sorry

end isosceles_triangle_23_perimeter_l1647_164752


namespace geometric_sequence_ratio_l1647_164761

/-- A geometric sequence with specific properties -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > a (n + 1)) ∧
  (∃ r : ℝ, 0 < r ∧ r < 1 ∧ ∀ n, a (n + 1) = r * a n) ∧
  (a 7 * a 14 = 6) ∧
  (a 4 + a 17 = 5)

/-- The main theorem stating the ratio of a_5 to a_18 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (h : GeometricSequence a) :
  a 5 / a 18 = 3 / 2 := by
  sorry

end geometric_sequence_ratio_l1647_164761


namespace fertilizing_to_mowing_ratio_l1647_164759

def mowing_time : ℕ := 40
def total_time : ℕ := 120

def fertilizing_time : ℕ := total_time - mowing_time

theorem fertilizing_to_mowing_ratio :
  (fertilizing_time : ℚ) / mowing_time = 2 := by sorry

end fertilizing_to_mowing_ratio_l1647_164759


namespace remove_seven_improves_mean_l1647_164773

def scores : List ℕ := [6, 7, 7, 8, 8, 8, 9, 10]

def mode (l : List ℕ) : ℕ := sorry

def range (l : List ℕ) : ℕ := sorry

def mean (l : List ℕ) : ℚ := sorry

def remove_score (s : List ℕ) (n : ℕ) : List ℕ := sorry

theorem remove_seven_improves_mean :
  let original_scores := scores
  let new_scores := remove_score original_scores 7
  mode new_scores = mode original_scores ∧
  range new_scores = range original_scores ∧
  mean new_scores > mean original_scores :=
sorry

end remove_seven_improves_mean_l1647_164773


namespace count_numbers_with_three_is_180_l1647_164711

/-- The count of natural numbers from 1 to 1000 that contain the digit 3 at least once -/
def count_numbers_with_three : ℕ :=
  let total_numbers := 1000
  let numbers_without_three := 820
  total_numbers - numbers_without_three

/-- Theorem stating that the count of natural numbers from 1 to 1000 
    containing the digit 3 at least once is equal to 180 -/
theorem count_numbers_with_three_is_180 :
  count_numbers_with_three = 180 := by
  sorry

#eval count_numbers_with_three

end count_numbers_with_three_is_180_l1647_164711


namespace at_least_one_negative_l1647_164700

theorem at_least_one_negative (a b c d : ℝ) 
  (sum_ab : a + b = 1) 
  (sum_cd : c + d = 1) 
  (prod_sum : a * c + b * d > 1) : 
  ¬(0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d) := by
sorry

end at_least_one_negative_l1647_164700


namespace pencils_per_row_l1647_164764

theorem pencils_per_row (total_pencils : ℕ) (num_rows : ℕ) (pencils_per_row : ℕ) : 
  total_pencils = 12 → num_rows = 3 → total_pencils = num_rows * pencils_per_row → pencils_per_row = 4 := by
  sorry

end pencils_per_row_l1647_164764


namespace olivias_dad_spending_l1647_164744

theorem olivias_dad_spending (cost_per_meal : ℕ) (number_of_meals : ℕ) (total_cost : ℕ) : 
  cost_per_meal = 7 → number_of_meals = 3 → total_cost = cost_per_meal * number_of_meals → total_cost = 21 :=
by
  sorry

end olivias_dad_spending_l1647_164744


namespace hiking_team_participants_l1647_164762

theorem hiking_team_participants (min_gloves : ℕ) (gloves_per_participant : ℕ) : 
  min_gloves = 86 → gloves_per_participant = 2 → min_gloves / gloves_per_participant = 43 :=
by
  sorry

end hiking_team_participants_l1647_164762


namespace parabola_tangent_to_line_l1647_164707

/-- A parabola y = ax^2 + 4x + 3 is tangent to the line y = 2x + 1 if and only if a = 1/2 -/
theorem parabola_tangent_to_line (a : ℝ) : 
  (∃ x : ℝ, ax^2 + 4*x + 3 = 2*x + 1 ∧ 
   ∀ y : ℝ, y ≠ x → ax^2 + 4*x + 3 ≠ 2*y + 1) ↔ 
  a = (1/2 : ℝ) := by
sorry

end parabola_tangent_to_line_l1647_164707


namespace unique_solution_xyz_l1647_164701

theorem unique_solution_xyz (x y z : ℝ) 
  (hx : x > 4) (hy : y > 4) (hz : z > 4)
  (h : (x + 3)^2 / (y + z - 3) + (y + 5)^2 / (z + x - 5) + (z + 7)^2 / (x + y - 7) = 45) :
  x = 12 ∧ y = 10 ∧ z = 8 := by
  sorry

end unique_solution_xyz_l1647_164701


namespace absolute_value_ratio_l1647_164795

theorem absolute_value_ratio (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a^2 + b^2 = 8*a*b) :
  |a + b| / |a - b| = Real.sqrt 15 / 3 := by
  sorry

end absolute_value_ratio_l1647_164795


namespace max_sum_of_factors_l1647_164712

theorem max_sum_of_factors (X Y Z : ℕ) : 
  X > 0 ∧ Y > 0 ∧ Z > 0 →  -- Positive integers
  X ≠ Y ∧ Y ≠ Z ∧ X ≠ Z →  -- Distinct integers
  X * Y * Z = 399 →        -- Product constraint
  X + Y + Z ≤ 29           -- Maximum sum
  := by sorry

end max_sum_of_factors_l1647_164712


namespace yogurt_combinations_l1647_164789

theorem yogurt_combinations (flavors : ℕ) (toppings : ℕ) : 
  flavors = 5 → toppings = 7 → flavors * (toppings.choose 3) = 175 := by
  sorry

end yogurt_combinations_l1647_164789


namespace largest_divisible_n_ten_is_divisible_largest_n_is_ten_l1647_164798

theorem largest_divisible_n : ∀ n : ℕ, n > 10 → ¬(n + 15 ∣ n^3 + 250) := by
  sorry

theorem ten_is_divisible : (10 + 15 ∣ 10^3 + 250) := by
  sorry

theorem largest_n_is_ten : 
  ∀ n : ℕ, n > 0 → (n + 15 ∣ n^3 + 250) → n ≤ 10 := by
  sorry

end largest_divisible_n_ten_is_divisible_largest_n_is_ten_l1647_164798


namespace difference_before_l1647_164717

/-- The number of battle cards Sang-cheol had originally -/
def S : ℕ := sorry

/-- The number of battle cards Byeong-ji had originally -/
def B : ℕ := sorry

/-- Sang-cheol gave Byeong-ji 2 battle cards -/
axiom exchange : S ≥ 2

/-- After the exchange, the difference between Byeong-ji and Sang-cheol was 6 -/
axiom difference_after : B + 2 - (S - 2) = 6

/-- Byeong-ji has more cards than Sang-cheol -/
axiom byeongji_has_more : B > S

/-- The difference between Byeong-ji and Sang-cheol before the exchange was 2 -/
theorem difference_before : B - S = 2 := by sorry

end difference_before_l1647_164717


namespace sum_of_greater_than_l1647_164757

theorem sum_of_greater_than (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : a + c > b + d := by
  sorry

end sum_of_greater_than_l1647_164757


namespace geometric_sequence_sum_l1647_164780

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (∀ n, a n > 0) →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25 →
  a 3 + a 5 = 5 := by
  sorry

end geometric_sequence_sum_l1647_164780


namespace fraction_equality_l1647_164784

theorem fraction_equality (a b : ℝ) (h : a ≠ b) (h1 : a ≠ 0) (h2 : b ≠ 0) :
  (1 / a - 1 / b = 1 / 3) → (a * b / (a - b) = -3) := by
  sorry

end fraction_equality_l1647_164784


namespace ball_probability_l1647_164793

theorem ball_probability (x : ℕ) : 
  (6 : ℝ) / ((6 : ℝ) + x) = (3 : ℝ) / 10 → x = 14 := by
  sorry

end ball_probability_l1647_164793


namespace sandwich_combinations_l1647_164758

theorem sandwich_combinations (meat_types cheese_types : ℕ) 
  (h1 : meat_types = 12) 
  (h2 : cheese_types = 8) : 
  meat_types * (cheese_types.choose 3) = 672 := by
  sorry

end sandwich_combinations_l1647_164758


namespace unique_solution_sqrt_equation_l1647_164753

theorem unique_solution_sqrt_equation :
  ∃! x : ℝ, x ≥ 4 ∧ Real.sqrt (x + 2 - 2 * Real.sqrt (x - 4)) + Real.sqrt (x + 12 - 8 * Real.sqrt (x - 4)) = 4 :=
by
  sorry

end unique_solution_sqrt_equation_l1647_164753


namespace parallelogram_acute_angle_cosine_l1647_164785

/-- Given a parallelogram with sides a and b where a ≠ b, if perpendicular lines drawn from
    vertices of obtuse angles form a similar parallelogram, then the cosine of the acute angle α
    is (2ab) / (a² + b²) -/
theorem parallelogram_acute_angle_cosine (a b : ℝ) (h : a ≠ b) :
  ∃ α : ℝ, 0 < α ∧ α < π / 2 ∧
  (∃ (similar : Bool), similar = true →
    Real.cos α = (2 * a * b) / (a^2 + b^2)) :=
sorry

end parallelogram_acute_angle_cosine_l1647_164785


namespace consecutive_odd_sum_divisible_by_four_l1647_164763

theorem consecutive_odd_sum_divisible_by_four (n : ℤ) : 
  4 ∣ ((2*n + 1) + (2*n + 3)) := by
sorry

end consecutive_odd_sum_divisible_by_four_l1647_164763


namespace no_integer_satisfies_conditions_l1647_164739

theorem no_integer_satisfies_conditions : ¬ ∃ m : ℤ, m % 9 = 2 ∧ m % 6 = 1 := by
  sorry

end no_integer_satisfies_conditions_l1647_164739


namespace no_real_b_for_single_solution_l1647_164743

theorem no_real_b_for_single_solution :
  ¬ ∃ b : ℝ, ∃! x : ℝ, |x^2 + 3*b*x + 5*b| ≤ 5 :=
by sorry

end no_real_b_for_single_solution_l1647_164743


namespace combined_fuel_efficiency_l1647_164726

/-- Calculates the combined fuel efficiency of two cars -/
theorem combined_fuel_efficiency
  (efficiency1 : ℝ) -- Fuel efficiency of the first car in miles per gallon
  (efficiency2 : ℝ) -- Fuel efficiency of the second car in miles per gallon
  (h1 : efficiency1 = 40) -- Given: Ray's car averages 40 miles per gallon
  (h2 : efficiency2 = 10) -- Given: Tom's car averages 10 miles per gallon
  (distance : ℝ) -- Distance driven by each car
  (h3 : distance > 0) -- Assumption: Distance driven is positive
  : (2 * distance) / ((distance / efficiency1) + (distance / efficiency2)) = 16 := by
  sorry

end combined_fuel_efficiency_l1647_164726


namespace g_of_2_eq_11_l1647_164751

/-- The function g(x) = x^3 + x^2 - 1 -/
def g (x : ℝ) : ℝ := x^3 + x^2 - 1

/-- Theorem: g(2) = 11 -/
theorem g_of_2_eq_11 : g 2 = 11 := by
  sorry

end g_of_2_eq_11_l1647_164751


namespace cost_price_is_seven_l1647_164788

/-- The cost price of an article satisfying the given condition -/
def cost_price : ℕ := sorry

/-- The selling price that results in a profit -/
def profit_price : ℕ := 54

/-- The selling price that results in a loss -/
def loss_price : ℕ := 40

/-- The profit is equal to the loss -/
axiom profit_equals_loss : profit_price - cost_price = cost_price - loss_price

theorem cost_price_is_seven : cost_price = 7 := by sorry

end cost_price_is_seven_l1647_164788


namespace greatest_common_measure_l1647_164754

theorem greatest_common_measure (a b c : ℕ) (ha : a = 700) (hb : b = 385) (hc : c = 1295) :
  Nat.gcd a (Nat.gcd b c) = 35 := by
  sorry

end greatest_common_measure_l1647_164754


namespace break_even_components_min_profitable_components_l1647_164714

/-- The number of components produced and sold monthly -/
def components : ℕ := 150

/-- Production cost per component -/
def production_cost : ℚ := 80

/-- Shipping cost per component -/
def shipping_cost : ℚ := 5

/-- Fixed monthly costs -/
def fixed_costs : ℚ := 16500

/-- Minimum selling price per component -/
def selling_price : ℚ := 195

/-- Theorem stating that the number of components produced and sold monthly
    is the break-even point where costs equal revenues -/
theorem break_even_components :
  (selling_price * components : ℚ) = 
  fixed_costs + (production_cost + shipping_cost) * components := by
  sorry

/-- Theorem stating that the number of components is the minimum
    where revenues are not less than costs -/
theorem min_profitable_components :
  ∀ n : ℕ, n < components → 
  (selling_price * n : ℚ) < fixed_costs + (production_cost + shipping_cost) * n := by
  sorry

end break_even_components_min_profitable_components_l1647_164714


namespace square_circle_ratio_l1647_164771

theorem square_circle_ratio : 
  let square_area : ℝ := 784
  let small_circle_circumference : ℝ := 8
  let larger_radius_ratio : ℝ := 7/3

  let square_side : ℝ := Real.sqrt square_area
  let small_circle_radius : ℝ := small_circle_circumference / (2 * Real.pi)
  let large_circle_radius : ℝ := larger_radius_ratio * small_circle_radius

  square_side / large_circle_radius = 3 * Real.pi :=
by sorry

end square_circle_ratio_l1647_164771


namespace division_remainder_l1647_164706

theorem division_remainder (k : ℕ) : 
  k > 0 ∧ k < 38 ∧ 
  k % 5 = 2 ∧ 
  (∃ n : ℕ, n > 5 ∧ k % n = 5) →
  k % 7 = 5 :=
by sorry

end division_remainder_l1647_164706


namespace fixed_points_existence_l1647_164779

-- Define the fixed point F and line l
def F : ℝ × ℝ := (1, 0)
def l : ℝ → Prop := λ x => x = 4

-- Define the trajectory E
def E : ℝ × ℝ → Prop := λ p => (p.1^2 / 4) + (p.2^2 / 3) = 1

-- Define the distance ratio condition
def distance_ratio (P : ℝ × ℝ) : Prop :=
  Real.sqrt ((P.1 - F.1)^2 + (P.2 - F.2)^2) / |P.1 - 4| = 1/2

-- Define point A
def A : ℝ × ℝ := (-2, 0)

-- Define the theorem
theorem fixed_points_existence :
  ∃ Q₁ Q₂ : ℝ × ℝ,
    Q₁.2 = 0 ∧ Q₂.2 = 0 ∧
    Q₁ ≠ Q₂ ∧
    (∀ B C M N : ℝ × ℝ,
      E B ∧ E C ∧
      (∃ m : ℝ, B.1 = m * B.2 + 1 ∧ C.1 = m * C.2 + 1) ∧
      (M.1 = 4 ∧ N.1 = 4) ∧
      (∃ t : ℝ, M.2 = t * (B.1 + 2) ∧ N.2 = t * (C.1 + 2)) →
      ((Q₁.1 - M.1) * (Q₁.1 - N.1) + (Q₁.2 - M.2) * (Q₁.2 - N.2) = 0 ∧
       (Q₂.1 - M.1) * (Q₂.1 - N.1) + (Q₂.2 - M.2) * (Q₂.2 - N.2) = 0)) ∧
    Q₁ = (1, 0) ∧ Q₂ = (7, 0) :=
by sorry

end fixed_points_existence_l1647_164779


namespace trajectory_is_ellipse_l1647_164716

/-- The set of complex numbers z satisfying |z-i|+|z+i|=3 forms an ellipse in the complex plane -/
theorem trajectory_is_ellipse (z : ℂ) : 
  (Set.range fun (z : ℂ) => Complex.abs (z - Complex.I) + Complex.abs (z + Complex.I) = 3) 
  IsEllipse :=
sorry

end trajectory_is_ellipse_l1647_164716


namespace k_value_l1647_164774

theorem k_value (k : ℝ) (h1 : k ≠ 0) :
  (∀ x : ℝ, (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 6)) → k = 6 := by
  sorry

end k_value_l1647_164774


namespace large_rectangle_perimeter_l1647_164719

/-- The perimeter of a large rectangle composed of nine identical smaller rectangles -/
theorem large_rectangle_perimeter (small_length : ℝ) (h1 : small_length = 10) :
  let large_length := 2 * small_length
  let large_height := 4 * small_length / 5
  let perimeter := 2 * (large_length + large_height)
  perimeter = 76 := by sorry

end large_rectangle_perimeter_l1647_164719


namespace sandwich_combinations_theorem_l1647_164736

def num_meat_types : ℕ := 8
def num_cheese_types : ℕ := 7

def num_meat_combinations : ℕ := (num_meat_types * (num_meat_types - 1)) / 2
def num_cheese_combinations : ℕ := num_cheese_types

def total_sandwich_combinations : ℕ := num_meat_combinations * num_cheese_combinations

theorem sandwich_combinations_theorem : total_sandwich_combinations = 196 := by
  sorry

end sandwich_combinations_theorem_l1647_164736


namespace larger_number_of_product_35_sum_12_l1647_164787

theorem larger_number_of_product_35_sum_12 :
  ∀ x y : ℕ,
  x * y = 35 →
  x + y = 12 →
  max x y = 7 :=
by
  sorry

end larger_number_of_product_35_sum_12_l1647_164787


namespace expression_evaluation_l1647_164786

theorem expression_evaluation : (3^3 + 2)^2 - (3^3 - 2)^2 = 216 := by
  sorry

end expression_evaluation_l1647_164786


namespace triangle_tan_b_l1647_164796

/-- Given a triangle ABC with sides a, b, c opposite angles A, B, C respectively -/
theorem triangle_tan_b (a b c : ℝ) (A B C : ℝ) :
  /- a², b², c² form an arithmetic sequence -/
  (a^2 + c^2 = 2*b^2) →
  /- Area of triangle ABC is b²/3 -/
  (1/2 * a * c * Real.sin B = b^2/3) →
  /- Law of cosines -/
  (b^2 = a^2 + c^2 - 2*a*c*Real.cos B) →
  /- Then tan B = 4/3 -/
  Real.tan B = 4/3 := by
sorry

end triangle_tan_b_l1647_164796


namespace julian_needs_1100_more_legos_l1647_164705

/-- The number of legos Julian has -/
def julian_legos : ℕ := 400

/-- The number of airplane models Julian wants to make -/
def num_models : ℕ := 4

/-- The number of legos required for each airplane model -/
def legos_per_model : ℕ := 375

/-- The number of additional legos Julian needs -/
def additional_legos_needed : ℕ := 1100

/-- Theorem stating that Julian needs 1100 more legos to make 4 identical airplane models -/
theorem julian_needs_1100_more_legos :
  (num_models * legos_per_model) - julian_legos = additional_legos_needed := by
  sorry

end julian_needs_1100_more_legos_l1647_164705


namespace tangent_line_equation_l1647_164724

/-- The curve function -/
def f (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 - 18 * x + 7

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 6 * x^2 - 12 * x - 18

/-- The point of tangency -/
def p : ℝ × ℝ := (-2, 3)

/-- The slope of the tangent line at the point of tangency -/
def m : ℝ := f' p.1

theorem tangent_line_equation :
  ∀ x y : ℝ, y = f p.1 → (y - p.2 = m * (x - p.1)) ↔ (30 * x - y + 63 = 0) :=
by sorry

end tangent_line_equation_l1647_164724


namespace length_ratio_theorem_l1647_164770

/-- Represents a three-stage rocket with cylindrical stages -/
structure ThreeStageRocket where
  l₁ : ℝ  -- Length of the first stage
  l₂ : ℝ  -- Length of the second stage
  l₃ : ℝ  -- Length of the third stage

/-- The conditions for the three-stage rocket -/
def RocketConditions (r : ThreeStageRocket) : Prop :=
  r.l₂ = (r.l₁ + r.l₃) / 2 ∧
  r.l₂^3 = (6 / 13) * (r.l₁^3 + r.l₃^3)

/-- The theorem stating the ratio of lengths of the first and third stages -/
theorem length_ratio_theorem (r : ThreeStageRocket) (h : RocketConditions r) :
  r.l₁ / r.l₃ = 7 / 5 := by
  sorry


end length_ratio_theorem_l1647_164770


namespace only_prime_of_form_l1647_164713

theorem only_prime_of_form (p : ℕ) : 
  (∃ x : ℤ, p = 4 * x^4 + 1) ∧ Nat.Prime p ↔ p = 5 := by
  sorry

end only_prime_of_form_l1647_164713


namespace circle_y_axis_intersection_sum_l1647_164723

/-- The sum of y-coordinates of points where a circle intersects the y-axis -/
theorem circle_y_axis_intersection_sum (c : ℝ × ℝ) (r : ℝ) : 
  c.1 = -8 → c.2 = 3 → r = 15 → 
  ∃ y₁ y₂ : ℝ, 
    (0 - c.1)^2 + (y₁ - c.2)^2 = r^2 ∧
    (0 - c.1)^2 + (y₂ - c.2)^2 = r^2 ∧
    y₁ + y₂ = 6 :=
by sorry

end circle_y_axis_intersection_sum_l1647_164723


namespace max_groups_equals_gcd_l1647_164727

theorem max_groups_equals_gcd (boys girls : ℕ) (h1 : boys = 120) (h2 : girls = 140) :
  let max_groups := Nat.gcd boys girls
  ∀ k : ℕ, k ∣ boys ∧ k ∣ girls → k ≤ max_groups :=
by sorry

end max_groups_equals_gcd_l1647_164727


namespace set_equality_implies_sum_l1647_164704

theorem set_equality_implies_sum (a b : ℝ) (ha : a ≠ 0) :
  ({a, b / a, 1} : Set ℝ) = {a^2, a + b, 0} →
  a^2015 + b^2016 = -1 := by sorry

end set_equality_implies_sum_l1647_164704


namespace complex_number_in_first_quadrant_l1647_164729

theorem complex_number_in_first_quadrant : 
  let z : ℂ := (2 - I) / (1 - 3*I)
  (z.re > 0) ∧ (z.im > 0) :=
by sorry

end complex_number_in_first_quadrant_l1647_164729


namespace point_on_y_axis_l1647_164776

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define what it means for a point to be on the y-axis
def onYAxis (p : Point2D) : Prop := p.x = 0

-- Theorem statement
theorem point_on_y_axis (p : Point2D) : onYAxis p ↔ p.x = 0 := by
  sorry

end point_on_y_axis_l1647_164776


namespace inequality_and_equality_condition_l1647_164740

theorem inequality_and_equality_condition (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (h : x^2 + y^2 + z^2 + 3 = 2*(x*y + y*z + z*x)) :
  Real.sqrt (x*y) + Real.sqrt (y*z) + Real.sqrt (z*x) ≥ 3 ∧
  (Real.sqrt (x*y) + Real.sqrt (y*z) + Real.sqrt (z*x) = 3 ↔ x = 1 ∧ y = 1 ∧ z = 1) :=
by sorry

end inequality_and_equality_condition_l1647_164740


namespace intersection_M_N_l1647_164715

def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | Real.log (2 * x + 1) > 0}

theorem intersection_M_N : M ∩ N = {x | 0 < x ∧ x < 1} := by sorry

end intersection_M_N_l1647_164715


namespace gcd_lcm_360_possibilities_l1647_164783

theorem gcd_lcm_360_possibilities (a b : ℕ+) : 
  (Nat.gcd a b * Nat.lcm a b = 360) → 
  (∃ (S : Finset ℕ), S.card = 23 ∧ (∀ x, x ∈ S ↔ ∃ (a b : ℕ+), Nat.gcd a b = x ∧ Nat.gcd a b * Nat.lcm a b = 360)) := by
  sorry

end gcd_lcm_360_possibilities_l1647_164783


namespace total_choices_is_81_l1647_164778

/-- The number of bases available for students to choose from. -/
def num_bases : ℕ := 3

/-- The number of students choosing bases. -/
def num_students : ℕ := 4

/-- The total number of ways students can choose bases. -/
def total_choices : ℕ := num_bases ^ num_students

/-- Theorem stating that the total number of choices is 81. -/
theorem total_choices_is_81 : total_choices = 81 := by
  sorry

end total_choices_is_81_l1647_164778


namespace max_value_of_sequence_l1647_164749

def a (n : ℕ) : ℚ := n / (n^2 + 90)

theorem max_value_of_sequence :
  ∃ (M : ℚ), M = 1/19 ∧ ∀ (n : ℕ), a n ≤ M ∧ ∃ (k : ℕ), a k = M :=
sorry

end max_value_of_sequence_l1647_164749


namespace alexas_vacation_time_l1647_164710

/-- Proves that Alexa's vacation time is 9 days given the conditions of the problem. -/
theorem alexas_vacation_time (E : ℝ) 
  (ethan_time : E > 0)
  (alexa_time : ℝ)
  (joey_time : ℝ)
  (alexa_vacation : alexa_time = 3/4 * E)
  (joey_swimming : joey_time = 1/2 * E)
  (joey_days : joey_time = 6) : 
  alexa_time = 9 := by
  sorry

end alexas_vacation_time_l1647_164710


namespace largest_among_abcd_l1647_164768

theorem largest_among_abcd (a b c d : ℝ) 
  (h : a - 1 = b + 2 ∧ a - 1 = c - 3 ∧ a - 1 = d + 4) : 
  c ≥ a ∧ c ≥ b ∧ c ≥ d := by
sorry

end largest_among_abcd_l1647_164768


namespace sampling_probability_l1647_164750

theorem sampling_probability (m : ℕ) (h_m : m ≥ 2017) :
  let systematic_prob := (3 : ℚ) / 2017
  let stratified_prob := (3 : ℚ) / 2017
  systematic_prob = stratified_prob := by sorry

end sampling_probability_l1647_164750


namespace seed_purchase_calculation_l1647_164792

/-- Given the cost of seeds and the amount spent by a farmer, 
    calculate the number of pounds of seeds purchased. -/
theorem seed_purchase_calculation 
  (seed_cost : ℝ) 
  (seed_amount : ℝ) 
  (farmer_spent : ℝ) 
  (h1 : seed_cost = 44.68)
  (h2 : seed_amount = 2)
  (h3 : farmer_spent = 134.04) :
  farmer_spent / (seed_cost / seed_amount) = 6 :=
by sorry

end seed_purchase_calculation_l1647_164792


namespace marcus_second_goal_value_l1647_164794

def team_total_points : ℕ := 70
def marcus_3point_goals : ℕ := 5
def marcus_unknown_goals : ℕ := 10
def marcus_percentage : ℚ := 1/2

theorem marcus_second_goal_value :
  ∃ (second_goal_value : ℕ),
    (marcus_3point_goals * 3 + marcus_unknown_goals * second_goal_value : ℚ) = 
      (marcus_percentage * team_total_points) ∧
    second_goal_value = 2 := by
  sorry

end marcus_second_goal_value_l1647_164794


namespace inscribed_sphere_radius_regular_tetrahedron_inscribed_sphere_radius_regular_tetrahedron_is_correct_l1647_164742

/-- Given a regular tetrahedron with face area S and volume V, 
    the radius of its inscribed sphere is 3V/(4S) -/
theorem inscribed_sphere_radius_regular_tetrahedron 
  (S V : ℝ) (S_pos : S > 0) (V_pos : V > 0) : ℝ :=
  3 * V / (4 * S)

/-- The calculated radius is indeed the radius of the inscribed sphere -/
theorem inscribed_sphere_radius_regular_tetrahedron_is_correct 
  (S V : ℝ) (S_pos : S > 0) (V_pos : V > 0) :
  inscribed_sphere_radius_regular_tetrahedron S V S_pos V_pos = 
    3 * V / (4 * S) := by sorry

end inscribed_sphere_radius_regular_tetrahedron_inscribed_sphere_radius_regular_tetrahedron_is_correct_l1647_164742


namespace min_value_a_plus_9b_l1647_164731

theorem min_value_a_plus_9b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + b = 10 * a * b) :
  8/5 ≤ a + 9*b ∧ ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ a₀ + b₀ = 10 * a₀ * b₀ ∧ a₀ + 9*b₀ = 8/5 :=
by sorry

end min_value_a_plus_9b_l1647_164731


namespace max_value_f_on_interval_range_of_a_for_inequality_l1647_164721

-- Define the quadratic function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (a - 1) * x + a

-- Part 1: Maximum value of f(x) when a = 2 on [-1, 1]
theorem max_value_f_on_interval :
  ∃ (M : ℝ), M = 5 ∧ ∀ x ∈ Set.Icc (-1 : ℝ) 1, f 2 x ≤ M :=
sorry

-- Part 2: Range of a for f(x)/x ≥ 2 when x ∈ [1, 2]
theorem range_of_a_for_inequality :
  ∀ a : ℝ, (∀ x ∈ Set.Icc (1 : ℝ) 2, f a x / x ≥ 2) ↔ a ≥ 1 :=
sorry

end max_value_f_on_interval_range_of_a_for_inequality_l1647_164721


namespace amy_biking_week_l1647_164746

def total_miles_biked (monday_miles : ℕ) : ℕ :=
  let tuesday_miles := 2 * monday_miles - 3
  let wednesday_miles := tuesday_miles + 2
  let thursday_miles := wednesday_miles + 2
  let friday_miles := thursday_miles + 2
  let saturday_miles := friday_miles + 2
  let sunday_miles := saturday_miles + 2
  monday_miles + tuesday_miles + wednesday_miles + thursday_miles + friday_miles + saturday_miles + sunday_miles

theorem amy_biking_week (monday_miles : ℕ) (h : monday_miles = 12) : 
  total_miles_biked monday_miles = 168 := by
  sorry

end amy_biking_week_l1647_164746


namespace base_of_term_l1647_164797

theorem base_of_term (x : ℝ) (k : ℝ) : 
  (1/2)^23 * (1/x)^k = 1/18^23 ∧ k = 11.5 → x = 9 := by
  sorry

end base_of_term_l1647_164797


namespace unique_prime_pair_solution_l1647_164733

theorem unique_prime_pair_solution : 
  ∃! (p q : ℕ), 
    Prime p ∧ Prime q ∧ 
    (7 * p * q^2 + p = q^3 + 43 * p^3 + 1) ∧ 
    p = 2 ∧ q = 7 := by
  sorry

end unique_prime_pair_solution_l1647_164733


namespace orange_probability_l1647_164730

theorem orange_probability (total : ℕ) (large : ℕ) (small : ℕ) (choose : ℕ) :
  total = 8 →
  large = 5 →
  small = 3 →
  choose = 3 →
  (Nat.choose small choose : ℚ) / (Nat.choose total choose : ℚ) = 1 / 56 :=
by sorry

end orange_probability_l1647_164730


namespace green_team_score_l1647_164766

/-- Given a winning team's score and their lead over the opponent,
    calculate the opponent's (losing team's) score. -/
def opponent_score (winning_score lead : ℕ) : ℕ :=
  winning_score - lead

/-- Theorem stating that given a winning score of 68 and a lead of 29,
    the opponent's score is 39. -/
theorem green_team_score :
  opponent_score 68 29 = 39 := by
  sorry

end green_team_score_l1647_164766


namespace min_sum_of_squares_l1647_164732

theorem min_sum_of_squares (x y : ℝ) (h : (x + 4) * (y - 4) = 0) :
  ∃ (min : ℝ), min = 32 ∧ ∀ (a b : ℝ), (a + 4) * (b - 4) = 0 → a^2 + b^2 ≥ min :=
sorry

end min_sum_of_squares_l1647_164732


namespace sufficient_not_necessary_condition_l1647_164728

theorem sufficient_not_necessary_condition (a b : ℝ) : 
  (∀ a b, a > b + 1 → a > b) ∧ 
  (∃ a b, a > b ∧ ¬(a > b + 1)) := by
  sorry

end sufficient_not_necessary_condition_l1647_164728


namespace arithmetic_progression_theorem_l1647_164720

/-- An arithmetic progression with n terms, first term a, and common difference d. -/
structure ArithmeticProgression where
  n : ℕ
  a : ℚ
  d : ℚ

/-- Sum of the first k terms of an arithmetic progression -/
def sum_first_k (ap : ArithmeticProgression) (k : ℕ) : ℚ :=
  k / 2 * (2 * ap.a + (k - 1) * ap.d)

/-- Sum of the last k terms of an arithmetic progression -/
def sum_last_k (ap : ArithmeticProgression) (k : ℕ) : ℚ :=
  k * (2 * ap.a + (ap.n - k + 1 + ap.n - 1) * ap.d / 2)

/-- Sum of all terms except the first k terms -/
def sum_without_first_k (ap : ArithmeticProgression) (k : ℕ) : ℚ :=
  (ap.n - k) / 2 * (2 * ap.a + (2 * k - 1 + ap.n - 1) * ap.d)

/-- Sum of all terms except the last k terms -/
def sum_without_last_k (ap : ArithmeticProgression) (k : ℕ) : ℚ :=
  (ap.n - k) / 2 * (2 * ap.a + (ap.n - k - 1) * ap.d)

/-- Theorem: If the sum of the first 13 terms is 50% of the sum of the last 13 terms,
    and the sum of all terms without the first 3 terms is 3/2 times the sum of all terms
    without the last 3 terms, then the number of terms in the progression is 18. -/
theorem arithmetic_progression_theorem (ap : ArithmeticProgression) :
  sum_first_k ap 13 = (1/2) * sum_last_k ap 13 ∧
  sum_without_first_k ap 3 = (3/2) * sum_without_last_k ap 3 →
  ap.n = 18 := by
  sorry

end arithmetic_progression_theorem_l1647_164720


namespace number_divided_by_002_l1647_164769

theorem number_divided_by_002 :
  ∃ x : ℝ, x / 0.02 = 201.79999999999998 ∧ x = 4.0359999999999996 := by
  sorry

end number_divided_by_002_l1647_164769


namespace leading_coefficient_of_p_l1647_164790

/-- The polynomial expression -/
def p (x : ℝ) : ℝ := 5*(x^5 - 2*x^3 + x) - 8*(x^5 + x^3 + 3*x) + 6*(3*x^5 - x^2 + 4)

/-- The leading coefficient of a polynomial -/
def leading_coefficient (f : ℝ → ℝ) : ℝ :=
  sorry

theorem leading_coefficient_of_p :
  leading_coefficient p = 15 := by
  sorry

end leading_coefficient_of_p_l1647_164790


namespace festival_attendance_l1647_164781

theorem festival_attendance (total_students : ℕ) (total_attendees : ℕ) 
  (h_total : total_students = 1500)
  (h_attendees : total_attendees = 975)
  (girls : ℕ) (boys : ℕ)
  (h_students : girls + boys = total_students)
  (h_attendance : (3 * girls / 4 : ℚ) + (2 * boys / 5 : ℚ) = total_attendees) :
  (3 * girls / 4 : ℕ) = 803 :=
sorry

end festival_attendance_l1647_164781


namespace cube_volume_from_body_diagonal_l1647_164735

theorem cube_volume_from_body_diagonal (diagonal : ℝ) (h : diagonal = 15) :
  ∃ (side : ℝ), side * Real.sqrt 3 = diagonal ∧ side^3 = 375 * Real.sqrt 3 := by
  sorry

end cube_volume_from_body_diagonal_l1647_164735


namespace correct_articles_for_categories_l1647_164767

-- Define a type for grammatical articles
inductive Article
  | Indefinite -- represents "a/an"
  | Definite   -- represents "the"
  | None       -- represents no article (used for plural nouns)

-- Define a function to determine the correct article for a category
def correctArticle (isFirstCategory : Bool) (isPlural : Bool) : Article :=
  if isFirstCategory then
    Article.Indefinite
  else if isPlural then
    Article.None
  else
    Article.Definite

-- Theorem statement
theorem correct_articles_for_categories :
  ∀ (isFirstCategory : Bool) (isPlural : Bool),
    (isFirstCategory ∧ ¬isPlural) →
    (¬isFirstCategory ∧ isPlural) →
    (correctArticle isFirstCategory isPlural = Article.Indefinite ∧
     correctArticle (¬isFirstCategory) isPlural = Article.None) :=
by
  sorry


end correct_articles_for_categories_l1647_164767
