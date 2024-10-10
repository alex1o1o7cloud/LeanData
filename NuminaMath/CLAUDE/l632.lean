import Mathlib

namespace jelly_bean_problem_l632_63237

theorem jelly_bean_problem (initial_red : ℚ) (initial_green : ℚ) (initial_blue : ℚ)
  (removed : ℚ) (final_blue_percentage : ℚ) (final_red_percentage : ℚ) :
  initial_red = 54 / 100 →
  initial_green = 30 / 100 →
  initial_blue = 16 / 100 →
  initial_red + initial_green + initial_blue = 1 →
  removed ≥ 0 →
  removed ≤ min initial_red initial_green →
  final_blue_percentage = 20 / 100 →
  final_blue_percentage = initial_blue / (1 - 2 * removed) →
  final_red_percentage = (initial_red - removed) / (1 - 2 * removed) →
  final_red_percentage = 55 / 100 :=
by sorry

end jelly_bean_problem_l632_63237


namespace intersection_M_N_l632_63202

def M : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def N : Set ℝ := {x | ∃ y, y = Real.sqrt (x - 1)}

theorem intersection_M_N : M ∩ N = {x | 1 ≤ x ∧ x < 3} := by sorry

end intersection_M_N_l632_63202


namespace min_abs_z_l632_63261

theorem min_abs_z (z : ℂ) (h : Complex.abs (z - 2*I) + Complex.abs (z - 5) = 7) :
  Complex.abs z ≥ 10 / Real.sqrt 29 ∧ ∃ w : ℂ, Complex.abs (w - 2*I) + Complex.abs (w - 5) = 7 ∧ Complex.abs w = 10 / Real.sqrt 29 :=
by sorry

end min_abs_z_l632_63261


namespace water_addition_theorem_l632_63277

/-- Represents the amount of water that can be added to the 6-liter bucket --/
def water_to_add (bucket3 bucket5 bucket6 : ℕ) : ℕ :=
  bucket6 - (bucket5 - bucket3)

/-- Theorem stating the amount of water that can be added to the 6-liter bucket --/
theorem water_addition_theorem :
  water_to_add 3 5 6 = 4 :=
by
  sorry

end water_addition_theorem_l632_63277


namespace direction_vector_of_bisecting_line_l632_63266

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 2) + 2

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y = 0

-- Define what it means for a line to bisect a circle
def bisects (k : ℝ) : Prop := ∃ (x₀ y₀ : ℝ), circle_C x₀ y₀ ∧ line_l k x₀ y₀

-- Theorem statement
theorem direction_vector_of_bisecting_line :
  ∃ (k : ℝ), bisects k → ∃ (t : ℝ), t ≠ 0 ∧ (2 = t * 2 ∧ 2 = t * 2) :=
sorry

end direction_vector_of_bisecting_line_l632_63266


namespace ratio_proof_l632_63223

theorem ratio_proof (x y : ℝ) (h : (3 * x^2 - 2 * y^2) / (x^2 + 4 * y^2) = 5 / 7) :
  x / y = Real.sqrt (17 / 8) :=
by sorry

end ratio_proof_l632_63223


namespace aras_height_is_55_l632_63258

/-- Calculates Ara's current height given the conditions of the problem -/
def aras_current_height (original_height : ℝ) (sheas_growth_rate : ℝ) (sheas_current_height : ℝ) (aras_growth_fraction : ℝ) : ℝ :=
  let sheas_growth := sheas_current_height - original_height
  let aras_growth := aras_growth_fraction * sheas_growth
  original_height + aras_growth

/-- The theorem stating that Ara's current height is 55 inches -/
theorem aras_height_is_55 :
  let original_height := 50
  let sheas_growth_rate := 0.3
  let sheas_current_height := 65
  let aras_growth_fraction := 1/3
  aras_current_height original_height sheas_growth_rate sheas_current_height aras_growth_fraction = 55 := by
  sorry


end aras_height_is_55_l632_63258


namespace mens_wages_l632_63204

/-- Given that 5 men are equal to W women, W women are equal to 8 boys,
    and the total earnings of all (5 men + W women + 8 boys) is 180 Rs,
    prove that each man's wage is 36 Rs. -/
theorem mens_wages (W : ℕ) : 
  (5 : ℕ) = W → -- 5 men are equal to W women
  W = 8 → -- W women are equal to 8 boys
  (5 : ℕ) * x + W * x + 8 * x = 180 → -- total earnings equation
  x = 36 := by sorry

end mens_wages_l632_63204


namespace adams_friends_strawberries_l632_63288

/-- The number of strawberries Adam's friends ate -/
def friends_strawberries (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

/-- Proof that Adam's friends ate 2 strawberries -/
theorem adams_friends_strawberries :
  friends_strawberries 35 33 = 2 := by
  sorry

end adams_friends_strawberries_l632_63288


namespace triangle_area_special_conditions_l632_63272

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem stating the area of the triangle under given conditions -/
theorem triangle_area_special_conditions (t : Triangle) 
  (h1 : (t.a - t.c)^2 = t.b^2 - 3/4 * t.a * t.c)
  (h2 : t.b = Real.sqrt 13)
  (h3 : ∃ (d : ℝ), Real.sin t.A + Real.sin t.C = 2 * Real.sin t.B) :
  (1/2 * t.a * t.c * Real.sin t.B) = (3 * Real.sqrt 39) / 4 :=
sorry

end triangle_area_special_conditions_l632_63272


namespace exists_valid_coloring_l632_63252

/-- A coloring function for points in ℚ × ℚ -/
def Coloring := ℚ × ℚ → Fin 2

/-- The distance between two points in ℚ × ℚ -/
def distance (p q : ℚ × ℚ) : ℚ :=
  ((p.1 - q.1)^2 + (p.2 - q.2)^2).sqrt

/-- A valid coloring function assigns different colors to points with distance 1 -/
def is_valid_coloring (f : Coloring) : Prop :=
  ∀ p q : ℚ × ℚ, distance p q = 1 → f p ≠ f q

theorem exists_valid_coloring : ∃ f : Coloring, is_valid_coloring f := by
  sorry

end exists_valid_coloring_l632_63252


namespace choir_theorem_l632_63225

def choir_problem (original_size absent first_fraction second_fraction third_fraction fourth_fraction late_arrivals : ℕ) : Prop :=
  let present := original_size - absent
  let first_verse := present / 2
  let second_verse := (present - first_verse) / 3
  let third_verse := (present - first_verse - second_verse) / 4
  let fourth_verse := (present - first_verse - second_verse - third_verse) / 5
  let total_before_fifth := first_verse + second_verse + third_verse + fourth_verse + late_arrivals
  total_before_fifth + (present - total_before_fifth) = present

theorem choir_theorem :
  choir_problem 70 10 2 3 4 5 5 :=
sorry

end choir_theorem_l632_63225


namespace line_passes_through_fixed_point_l632_63249

/-- The line y = kx + 2k + 1 always passes through the point (-2, 1) for all real k. -/
theorem line_passes_through_fixed_point :
  ∀ (k : ℝ), ((-2 : ℝ) * k + 2 * k + 1 = 1) := by sorry

end line_passes_through_fixed_point_l632_63249


namespace remainder_99_36_mod_100_l632_63251

theorem remainder_99_36_mod_100 : 99^36 % 100 = 1 := by
  sorry

end remainder_99_36_mod_100_l632_63251


namespace unique_intersection_values_l632_63222

-- Define the set of complex numbers that satisfy |z - 2| = 3|z + 2|
def S : Set ℂ := {z : ℂ | Complex.abs (z - 2) = 3 * Complex.abs (z + 2)}

-- Define a function that returns the set of intersection points between S and |z| = k
def intersection (k : ℝ) : Set ℂ := S ∩ {z : ℂ | Complex.abs z = k}

-- State the theorem
theorem unique_intersection_values :
  ∀ k : ℝ, (∃! z : ℂ, z ∈ intersection k) ↔ (k = 1 ∨ k = 4) :=
by sorry

end unique_intersection_values_l632_63222


namespace second_train_length_second_train_length_solution_l632_63292

/-- Calculates the length of the second train given the speeds of two trains, 
    the time they take to cross each other, and the length of the first train. -/
theorem second_train_length 
  (speed1 : ℝ) 
  (speed2 : ℝ) 
  (crossing_time : ℝ) 
  (length1 : ℝ) : ℝ :=
  let relative_speed := speed1 + speed2
  let relative_speed_mps := relative_speed * (5/18)
  let length2 := relative_speed_mps * crossing_time - length1
  length2

/-- The length of the second train is approximately 159.97 meters. -/
theorem second_train_length_solution :
  ∃ ε > 0, |second_train_length 60 40 11.879049676025918 170 - 159.97| < ε :=
by
  sorry

end second_train_length_second_train_length_solution_l632_63292


namespace symmetry_sum_l632_63228

/-- Two points are symmetric with respect to the x-axis if their x-coordinates are equal
    and their y-coordinates are negatives of each other. -/
def symmetric_wrt_x_axis (A B : ℝ × ℝ) : Prop :=
  A.1 = B.1 ∧ A.2 = -B.2

/-- If point A(m, 3) is symmetric to point B(2, n) with respect to the x-axis,
    then m + n = -1. -/
theorem symmetry_sum (m n : ℝ) :
  symmetric_wrt_x_axis (m, 3) (2, n) → m + n = -1 := by
  sorry

end symmetry_sum_l632_63228


namespace alex_coin_distribution_l632_63263

/-- The minimum number of additional coins needed to distribute distinct, positive numbers of coins to a given number of friends, starting with a given number of coins. -/
def min_additional_coins (num_friends : ℕ) (initial_coins : ℕ) : ℕ :=
  let required_coins := (num_friends * (num_friends + 1)) / 2
  if required_coins > initial_coins then
    required_coins - initial_coins
  else
    0

/-- The problem statement as a theorem -/
theorem alex_coin_distribution :
  min_additional_coins 15 97 = 23 := by
  sorry

end alex_coin_distribution_l632_63263


namespace triangle_existence_theorem_l632_63232

/-- The sum of angles in a triangle is 180 degrees -/
axiom triangle_angle_sum : ℝ → ℝ → ℝ → Prop

/-- A right angle is 90 degrees -/
def is_right_angle (angle : ℝ) : Prop := angle = 90

/-- An acute angle is less than 90 degrees -/
def is_acute_angle (angle : ℝ) : Prop := angle < 90

/-- An equilateral triangle has three equal angles -/
def is_equilateral_triangle (a b c : ℝ) : Prop := a = b ∧ b = c

theorem triangle_existence_theorem :
  (∀ a b c : ℝ, triangle_angle_sum a b c → is_right_angle a → b = 60 → c = 60 → False) ∧
  (∃ a b c : ℝ, triangle_angle_sum a b c ∧ is_equilateral_triangle a b c ∧ a = 60) ∧
  (∀ a b c : ℝ, triangle_angle_sum a b c → is_right_angle a → is_right_angle b → is_right_angle c → False) ∧
  (∃ a b c : ℝ, triangle_angle_sum a b c ∧ is_equilateral_triangle a b c ∧ is_acute_angle a) ∧
  (∀ a b c : ℝ, triangle_angle_sum a b c → is_right_angle a → b = 45 → c = 15 → False) :=
by sorry

end triangle_existence_theorem_l632_63232


namespace three_greater_than_negative_four_l632_63220

theorem three_greater_than_negative_four : 3 > -4 := by
  sorry

end three_greater_than_negative_four_l632_63220


namespace circle_tangent_line_l632_63265

theorem circle_tangent_line (a : ℝ) : 
  (∃ (x y : ℝ), x - y + 1 = 0 ∧ x^2 + y^2 - 2*x + 1 - a = 0 ∧ 
  ∀ (x' y' : ℝ), x' - y' + 1 = 0 → x'^2 + y'^2 - 2*x' + 1 - a ≥ 0) → 
  a = 2 := by
sorry

end circle_tangent_line_l632_63265


namespace solve_race_problem_l632_63283

def race_problem (patrick_time manu_extra_time : ℕ) (amy_speed_ratio : ℚ) : Prop :=
  let manu_time := patrick_time + manu_extra_time
  let amy_time := manu_time / amy_speed_ratio
  amy_time = 36

theorem solve_race_problem :
  race_problem 60 12 2 := by sorry

end solve_race_problem_l632_63283


namespace stack_map_front_view_l632_63221

/-- Represents a column of stacked cubes -/
def Column := List Nat

/-- Represents the top view of a stack map -/
structure StackMap :=
  (column1 : Column)
  (column2 : Column)
  (column3 : Column)

/-- Returns the maximum height of a column -/
def maxHeight (c : Column) : Nat :=
  c.foldl max 0

/-- Returns the front view of a stack map -/
def frontView (s : StackMap) : List Nat :=
  [maxHeight s.column1, maxHeight s.column2, maxHeight s.column3]

/-- The given stack map -/
def givenStackMap : StackMap :=
  { column1 := [3, 2]
  , column2 := [2, 4, 2]
  , column3 := [5, 2] }

theorem stack_map_front_view :
  frontView givenStackMap = [3, 4, 5] := by
  sorry

end stack_map_front_view_l632_63221


namespace solution_set_part_i_range_of_a_part_ii_l632_63281

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Theorem for part (I)
theorem solution_set_part_i :
  let a : ℝ := -2
  let S := {x : ℝ | f a x + f a (2 * x) > 2}
  S = {x : ℝ | x < -2 ∨ x > -2/3} :=
sorry

-- Theorem for part (II)
theorem range_of_a_part_ii :
  ∀ a : ℝ, a < 0 →
  (∃ x : ℝ, f a x + f a (2 * x) < 1/2) →
  -1 < a ∧ a < 0 :=
sorry

end solution_set_part_i_range_of_a_part_ii_l632_63281


namespace jason_has_four_balloons_l632_63296

/-- The number of violet balloons Jason has now, given his initial count and the number he lost. -/
def remaining_balloons (initial : ℕ) (lost : ℕ) : ℕ :=
  initial - lost

/-- Theorem stating that Jason has 4 violet balloons now. -/
theorem jason_has_four_balloons : remaining_balloons 7 3 = 4 := by
  sorry

end jason_has_four_balloons_l632_63296


namespace johns_spending_l632_63291

theorem johns_spending (initial_amount : ℚ) (snack_fraction : ℚ) (final_amount : ℚ)
  (h1 : initial_amount = 20)
  (h2 : snack_fraction = 1/5)
  (h3 : final_amount = 4) :
  let remaining_after_snacks := initial_amount - snack_fraction * initial_amount
  (remaining_after_snacks - final_amount) / remaining_after_snacks = 3/4 := by
sorry

end johns_spending_l632_63291


namespace correct_hand_in_amount_l632_63242

/-- Calculates the amount of money Jack will hand in given the number of bills of each denomination and the amount to be left in the till -/
def money_to_hand_in (hundreds twos fifties twenties tens fives ones leave_in_till : ℕ) : ℕ :=
  let total_in_notes := 100 * hundreds + 50 * fifties + 20 * twenties + 10 * tens + 5 * fives + ones
  total_in_notes - leave_in_till

/-- Theorem stating that the amount Jack will hand in is correct given the problem conditions -/
theorem correct_hand_in_amount :
  money_to_hand_in 2 0 1 5 3 7 27 300 = 142 := by
  sorry

#eval money_to_hand_in 2 0 1 5 3 7 27 300

end correct_hand_in_amount_l632_63242


namespace jujube_sales_theorem_l632_63217

/-- Represents the daily sales deviation from the planned amount -/
def daily_deviations : List Int := [4, -3, -5, 14, -8, 21, -6]

/-- The planned daily sales amount in pounds -/
def planned_daily_sales : Nat := 100

/-- The selling price per pound in yuan -/
def selling_price : Nat := 8

/-- The freight cost per pound in yuan -/
def freight_cost : Nat := 3

theorem jujube_sales_theorem :
  /- Total amount sold in first three days -/
  (List.take 3 daily_deviations).sum + 3 * planned_daily_sales = 296 ∧
  /- Total earnings for the week -/
  (daily_deviations.sum + 7 * planned_daily_sales) * (selling_price - freight_cost) = 3585 := by
  sorry

end jujube_sales_theorem_l632_63217


namespace donut_distribution_l632_63227

/-- The number of ways to distribute n items among k categories,
    with at least one item in each category. -/
def distribute_with_minimum (n k : ℕ) : ℕ :=
  Nat.choose (n - k + (k - 1)) (k - 1)

/-- Theorem: There are 35 ways to distribute 8 donuts among 4 types,
    with at least one donut of each type. -/
theorem donut_distribution : distribute_with_minimum 8 4 = 35 := by
  sorry

#eval distribute_with_minimum 8 4

end donut_distribution_l632_63227


namespace point_inside_circle_l632_63219

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Determines if a point is inside a circle -/
def isInside (p : ℝ × ℝ) (c : Circle) : Prop :=
  Real.sqrt ((p.1 - c.center.1)^2 + (p.2 - c.center.2)^2) < c.radius

theorem point_inside_circle (O : Circle) (P : ℝ × ℝ) 
    (h1 : O.radius = 5)
    (h2 : Real.sqrt ((P.1 - O.center.1)^2 + (P.2 - O.center.2)^2) = 4) :
  isInside P O := by
  sorry

end point_inside_circle_l632_63219


namespace value_of_a_l632_63224

theorem value_of_a (a b : ℚ) (h1 : b/a = 4) (h2 : b = 15 - 4*a) : a = 15/8 := by
  sorry

end value_of_a_l632_63224


namespace exchange_rates_problem_l632_63243

theorem exchange_rates_problem (drum wife leopard_skin : ℕ) : 
  (2 * drum + 3 * wife + leopard_skin = 111) →
  (3 * drum + 4 * wife = 2 * leopard_skin + 8) →
  (leopard_skin % 2 = 0) →
  (drum = 20 ∧ wife = 9 ∧ leopard_skin = 44) := by
  sorry

end exchange_rates_problem_l632_63243


namespace unique_number_property_l632_63298

theorem unique_number_property : ∃! x : ℝ, x / 3 = x - 5 := by sorry

end unique_number_property_l632_63298


namespace z_in_second_quadrant_l632_63245

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the complex number z
def z : ℂ := sorry

-- State the theorem
theorem z_in_second_quadrant :
  (1 + i) * z = -1 →
  z.re < 0 ∧ z.im > 0 := by
  sorry

end z_in_second_quadrant_l632_63245


namespace volume_increase_when_quadrupled_l632_63212

/-- Given a cylindrical container, when all its dimensions are quadrupled, 
    its volume increases by a factor of 64. -/
theorem volume_increase_when_quadrupled (r h V : ℝ) :
  V = π * r^2 * h →
  (π * (4*r)^2 * (4*h)) = 64 * V :=
by sorry

end volume_increase_when_quadrupled_l632_63212


namespace solve_for_m_l632_63299

theorem solve_for_m : ∃ m : ℝ, (-1 : ℝ) - 2 * m = 9 → m = -5 := by
  sorry

end solve_for_m_l632_63299


namespace product_as_sum_of_tens_l632_63239

theorem product_as_sum_of_tens :
  ∃ n : ℕ, n * 10 = 100 * 100 ∧ n = 1000 := by
  sorry

end product_as_sum_of_tens_l632_63239


namespace leftover_value_is_correct_l632_63284

/-- Calculate the value of leftover coins after combining and rolling --/
def leftover_value (james_quarters james_dimes emily_quarters emily_dimes : ℕ)
  (quarters_per_roll dimes_per_roll : ℕ) : ℚ :=
  let total_quarters := james_quarters + emily_quarters
  let total_dimes := james_dimes + emily_dimes
  let leftover_quarters := total_quarters % quarters_per_roll
  let leftover_dimes := total_dimes % dimes_per_roll
  (leftover_quarters : ℚ) * (1 / 4) + (leftover_dimes : ℚ) * (1 / 10)

/-- The main theorem --/
theorem leftover_value_is_correct :
  leftover_value 65 134 103 229 40 50 = 33 / 10 :=
by sorry

end leftover_value_is_correct_l632_63284


namespace sector_area_l632_63271

/-- The area of a sector with central angle 2π/3 and radius √3 is π -/
theorem sector_area (θ : Real) (r : Real) (h1 : θ = 2 * Real.pi / 3) (h2 : r = Real.sqrt 3) :
  1/2 * r^2 * θ = Real.pi := by
  sorry

end sector_area_l632_63271


namespace difference_of_squares_l632_63253

theorem difference_of_squares (x : ℝ) : x^2 - 9 = (x + 3) * (x - 3) := by
  sorry

end difference_of_squares_l632_63253


namespace cost_per_box_l632_63287

-- Define the box dimensions
def box_length : ℝ := 20
def box_width : ℝ := 20
def box_height : ℝ := 12

-- Define the total volume of the collection
def total_volume : ℝ := 1920000

-- Define the minimum total cost for boxes
def min_total_cost : ℝ := 200

-- Theorem to prove
theorem cost_per_box :
  let box_volume := box_length * box_width * box_height
  let num_boxes := total_volume / box_volume
  let cost_per_box := min_total_cost / num_boxes
  cost_per_box = 0.5 := by sorry

end cost_per_box_l632_63287


namespace adjacent_diff_at_least_16_l632_63203

/-- Represents a 6x6 grid with integers from 1 to 36 -/
def Grid := Fin 6 → Fin 6 → Fin 36

/-- Checks if two positions in the grid are adjacent -/
def adjacent (p1 p2 : Fin 6 × Fin 6) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2 = p2.2 + 1 ∨ p2.2 = p1.2 + 1)) ∨
  (p1.2 = p2.2 ∧ (p1.1 = p2.1 + 1 ∨ p2.1 = p1.1 + 1))

/-- A valid grid satisfies the problem conditions -/
def valid_grid (g : Grid) : Prop :=
  (∀ i j, g i j ≤ 36) ∧
  (∃ i1 j1 i2 j2 i3 j3 i4 j4,
    g i1 j1 = 1 ∧ g i2 j2 = 2 ∧ g i3 j3 = 3 ∧ g i4 j4 = 4) ∧
  (∀ i j, g i j ≤ 4 → g i j ≥ 1) ∧
  (∀ i j, g i j > 4 → g i j ≤ 36)

/-- The main theorem -/
theorem adjacent_diff_at_least_16 (g : Grid) (h : valid_grid g) :
  ∃ p1 p2 : Fin 6 × Fin 6, adjacent p1 p2 ∧ |g p1.1 p1.2 - g p2.1 p2.2| ≥ 16 := by
  sorry

end adjacent_diff_at_least_16_l632_63203


namespace probability_even_product_l632_63211

def range_start : ℕ := 6
def range_end : ℕ := 18

def is_in_range (n : ℕ) : Prop := range_start ≤ n ∧ n ≤ range_end

def total_integers : ℕ := range_end - range_start + 1

def total_combinations : ℕ := (total_integers * (total_integers - 1)) / 2

def count_even_in_range : ℕ := (range_end - range_start) / 2 + 1

def count_odd_in_range : ℕ := total_integers - count_even_in_range

def combinations_with_odd_product : ℕ := (count_odd_in_range * (count_odd_in_range - 1)) / 2

def combinations_with_even_product : ℕ := total_combinations - combinations_with_odd_product

theorem probability_even_product : 
  (combinations_with_even_product : ℚ) / total_combinations = 9 / 13 := by sorry

end probability_even_product_l632_63211


namespace percentage_50_59_is_100_over_9_l632_63218

/-- Represents the frequency distribution of test scores -/
structure ScoreDistribution :=
  (score_90_100 : ℕ)
  (score_80_89 : ℕ)
  (score_70_79 : ℕ)
  (score_60_69 : ℕ)
  (score_50_59 : ℕ)
  (score_below_50 : ℕ)

/-- The actual score distribution from Ms. Garcia's geometry class -/
def garcia_distribution : ScoreDistribution :=
  { score_90_100 := 5,
    score_80_89 := 7,
    score_70_79 := 9,
    score_60_69 := 8,
    score_50_59 := 4,
    score_below_50 := 3 }

/-- Calculate the total number of students -/
def total_students (d : ScoreDistribution) : ℕ :=
  d.score_90_100 + d.score_80_89 + d.score_70_79 + d.score_60_69 + d.score_50_59 + d.score_below_50

/-- Calculate the percentage of students in the 50%-59% range -/
def percentage_50_59 (d : ScoreDistribution) : ℚ :=
  (d.score_50_59 : ℚ) / (total_students d : ℚ) * 100

/-- Theorem stating that the percentage of students who scored in the 50%-59% range is 100/9 -/
theorem percentage_50_59_is_100_over_9 :
  percentage_50_59 garcia_distribution = 100 / 9 := by
  sorry


end percentage_50_59_is_100_over_9_l632_63218


namespace abc_fraction_value_l632_63238

theorem abc_fraction_value (a b c : ℝ) 
  (h1 : a * b / (a + b) = 1 / 3)
  (h2 : b * c / (b + c) = 1 / 4)
  (h3 : a * c / (c + a) = 1 / 5) :
  24 * a * b * c / (a * b + b * c + c * a) = 4 := by
sorry

end abc_fraction_value_l632_63238


namespace intersection_of_S_and_T_l632_63254

-- Define the sets S and T
def S : Set ℝ := {x : ℝ | x ≥ 2}
def T : Set ℝ := {x : ℝ | x ≤ 5}

-- State the theorem
theorem intersection_of_S_and_T : S ∩ T = Set.Icc 2 5 := by
  sorry

end intersection_of_S_and_T_l632_63254


namespace ratio_a_to_c_l632_63267

theorem ratio_a_to_c (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 2 / 3)
  (hdb : d / b = 1 / 5) : 
  a / c = 75 / 8 := by
  sorry

end ratio_a_to_c_l632_63267


namespace f_sum_equals_two_l632_63209

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + 9 * x^2) - 3 * x) + 1

theorem f_sum_equals_two : f (Real.log 2) + f (Real.log (1/2)) = 2 := by
  sorry

end f_sum_equals_two_l632_63209


namespace library_visitors_l632_63241

/-- Proves that the average number of visitors on Sundays is 540 given the specified conditions --/
theorem library_visitors (total_days : Nat) (non_sunday_visitors : Nat) (avg_visitors : Nat) :
  total_days = 30 ∧
  non_sunday_visitors = 240 ∧
  avg_visitors = 290 →
  (5 * (((avg_visitors * total_days) - (25 * non_sunday_visitors)) / 5) + 25 * non_sunday_visitors) / total_days = avg_visitors :=
by sorry

end library_visitors_l632_63241


namespace orchid_rose_difference_is_nine_l632_63268

/-- Flower quantities and ratios in a vase --/
structure FlowerVase where
  initial_roses : ℕ
  initial_orchids : ℕ
  initial_tulips : ℕ
  final_roses : ℕ
  final_orchids : ℕ
  final_tulips : ℕ
  rose_orchid_ratio : ℚ
  rose_tulip_ratio : ℚ

/-- The difference between orchids and roses after adding new flowers --/
def orchid_rose_difference (v : FlowerVase) : ℕ :=
  v.final_orchids - v.final_roses

/-- Theorem stating the difference between orchids and roses is 9 --/
theorem orchid_rose_difference_is_nine (v : FlowerVase)
  (h1 : v.initial_roses = 7)
  (h2 : v.initial_orchids = 12)
  (h3 : v.initial_tulips = 5)
  (h4 : v.final_roses = 11)
  (h5 : v.final_orchids = 20)
  (h6 : v.final_tulips = 10)
  (h7 : v.rose_orchid_ratio = 2/5)
  (h8 : v.rose_tulip_ratio = 3/5) :
  orchid_rose_difference v = 9 := by
  sorry

#eval orchid_rose_difference {
  initial_roses := 7,
  initial_orchids := 12,
  initial_tulips := 5,
  final_roses := 11,
  final_orchids := 20,
  final_tulips := 10,
  rose_orchid_ratio := 2/5,
  rose_tulip_ratio := 3/5
}

end orchid_rose_difference_is_nine_l632_63268


namespace product_plus_number_equals_93_l632_63259

theorem product_plus_number_equals_93 : ∃ x : ℤ, (-11 * -8) + x = 93 ∧ x = 5 := by sorry

end product_plus_number_equals_93_l632_63259


namespace pet_store_bird_count_l632_63276

theorem pet_store_bird_count :
  let num_cages : ℕ := 6
  let parrots_per_cage : ℕ := 6
  let parakeets_per_cage : ℕ := 2
  let total_birds : ℕ := num_cages * (parrots_per_cage + parakeets_per_cage)
  total_birds = 48 := by
sorry

end pet_store_bird_count_l632_63276


namespace subtraction_theorem_l632_63244

/-- Represents a four-digit number --/
structure FourDigitNumber where
  thousands : Nat
  hundreds : Nat
  tens : Nat
  ones : Nat
  h_digits : thousands < 10 ∧ hundreds < 10 ∧ tens < 10 ∧ ones < 10

/-- The result of subtracting two four-digit numbers --/
structure SubtractionResult where
  thousands : Int
  hundreds : Int
  tens : Int
  ones : Int

def subtract (minuend subtrahend : FourDigitNumber) : SubtractionResult :=
  sorry

theorem subtraction_theorem (a b c d : Nat) 
  (h_digits : a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10) :
  let minuend : FourDigitNumber := ⟨a, b, c, d, h_digits⟩
  let subtrahend : FourDigitNumber := ⟨d, b, a, c, sorry⟩
  let result := subtract minuend subtrahend
  (result.hundreds = 7 ∧ minuend.thousands ≥ subtrahend.thousands) →
  result.thousands = 9 := by
  sorry

end subtraction_theorem_l632_63244


namespace fruit_selling_results_l632_63274

/-- Represents the farmer's fruit selling scenario -/
structure FruitSelling where
  investment : ℝ
  total_yield : ℝ
  orchard_price : ℝ
  market_price : ℝ
  daily_market_sales : ℝ
  orchard_sales : ℝ

/-- The main theorem about the fruit selling scenario -/
theorem fruit_selling_results (s : FruitSelling)
  (h1 : s.investment = 13500)
  (h2 : s.total_yield = 19000)
  (h3 : s.orchard_price = 4)
  (h4 : s.market_price > 4)
  (h5 : s.daily_market_sales = 1000)
  (h6 : s.orchard_sales = 6000) :
  (s.total_yield / s.daily_market_sales = 19) ∧
  (s.total_yield * s.market_price - s.total_yield * s.orchard_price = 19000 * s.market_price - 76000) ∧
  (s.orchard_sales * s.orchard_price + (s.total_yield - s.orchard_sales) * s.market_price - s.investment = 13000 * s.market_price + 10500) := by
  sorry


end fruit_selling_results_l632_63274


namespace triangle_angle_matrix_det_zero_l632_63205

/-- The determinant of a specific matrix formed by angles of a triangle is zero -/
theorem triangle_angle_matrix_det_zero (A B C : Real) 
  (h : A + B + C = Real.pi) : -- Condition that A, B, C are angles of a triangle
  let M : Matrix (Fin 3) (Fin 3) ℝ := ![
    ![Real.exp A, Real.exp (-A), 1],
    ![Real.exp B, Real.exp (-B), 1],
    ![Real.exp C, Real.exp (-C), 1]
  ]
  Matrix.det M = 0 := by
  sorry


end triangle_angle_matrix_det_zero_l632_63205


namespace spring_decrease_percentage_l632_63215

theorem spring_decrease_percentage 
  (fall_increase : Real) 
  (total_change : Real) 
  (h1 : fall_increase = 0.08) 
  (h2 : total_change = -0.1252) : 
  let initial := 100
  let after_fall := initial * (1 + fall_increase)
  let after_spring := initial * (1 + total_change)
  (after_fall - after_spring) / after_fall = 0.19 := by
sorry

end spring_decrease_percentage_l632_63215


namespace min_tangent_length_l632_63208

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 + 4*x - 2*y + 3 = 0

-- Define point A
def point_A : ℝ × ℝ := (-1, 1)

-- Define the property that P is outside C
def outside_circle (x y : ℝ) : Prop :=
  x^2 + y^2 + 4*x - 2*y + 3 > 0

-- Define the tangent condition (|PM| = |PA|)
def tangent_condition (x y : ℝ) : Prop :=
  ∃ (mx my : ℝ), circle_C mx my ∧
  (x - mx)^2 + (y - my)^2 = (x + 1)^2 + (y - 1)^2

-- Theorem statement
theorem min_tangent_length :
  ∃ (min_length : ℝ),
    (∀ (x y : ℝ), outside_circle x y → tangent_condition x y →
      (x + 1)^2 + (y - 1)^2 ≥ min_length^2) ∧
    (∃ (x y : ℝ), outside_circle x y ∧ tangent_condition x y ∧
      (x + 1)^2 + (y - 1)^2 = min_length^2) ∧
    min_length = 1/2 :=
sorry

end min_tangent_length_l632_63208


namespace units_digit_of_p_plus_two_l632_63262

def is_positive_even (n : ℕ) : Prop := n > 0 ∧ n % 2 = 0

def has_positive_units_digit (n : ℕ) : Prop := n % 10 > 0

def units_digit (n : ℕ) : ℕ := n % 10

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem units_digit_of_p_plus_two (p q : ℕ) (x : ℕ+) :
  is_positive_even p →
  is_positive_even q →
  has_positive_units_digit p →
  has_positive_units_digit q →
  units_digit (p^3) - units_digit (p^2) = 0 →
  sum_of_digits p % q = 0 →
  p^(x : ℕ) = q →
  units_digit (p + 2) = 8 :=
by sorry

end units_digit_of_p_plus_two_l632_63262


namespace inclined_plane_friction_l632_63234

/-- The coefficient of friction between a block and an inclined plane -/
theorem inclined_plane_friction (P F_up F_down : ℝ) (α : ℝ) (μ : ℝ) :
  F_up = 3 * F_down →
  F_up + F_down = P →
  F_up = P * Real.sin α + μ * P * Real.cos α →
  F_down = P * Real.sin α - μ * P * Real.cos α →
  μ = Real.sqrt 3 / 6 := by
sorry

end inclined_plane_friction_l632_63234


namespace sum_product_inequality_l632_63240

theorem sum_product_inequality (a b c : ℝ) (h : a + b + c = 0) : a * b + b * c + c * a ≤ 0 := by
  sorry

end sum_product_inequality_l632_63240


namespace exam_percentage_l632_63214

theorem exam_percentage (total_students : ℕ) (assigned_avg makeup_avg overall_avg : ℚ) 
  (h1 : total_students = 100)
  (h2 : assigned_avg = 55 / 100)
  (h3 : makeup_avg = 95 / 100)
  (h4 : overall_avg = 67 / 100) :
  ∃ (x : ℚ), 
    0 ≤ x ∧ x ≤ 1 ∧
    x * assigned_avg + (1 - x) * makeup_avg = overall_avg ∧
    x = 70 / 100 := by
  sorry

end exam_percentage_l632_63214


namespace basketball_team_enrollment_l632_63255

theorem basketball_team_enrollment (total_players : ℕ) 
  (physics_enrollment : ℕ) (both_enrollment : ℕ) :
  total_players = 15 →
  physics_enrollment = 9 →
  both_enrollment = 3 →
  physics_enrollment + (total_players - physics_enrollment) ≥ total_players →
  total_players - physics_enrollment + both_enrollment = 9 :=
by sorry

end basketball_team_enrollment_l632_63255


namespace initial_number_of_persons_l632_63278

theorem initial_number_of_persons (N : ℕ) 
  (h1 : ∃ (avg : ℝ), N * (avg + 5) - N * avg = 105 - 65) : N = 8 := by
  sorry

end initial_number_of_persons_l632_63278


namespace induction_principle_l632_63260

theorem induction_principle (P : ℕ → Prop) :
  (∀ k, P k → P (k + 1)) →
  ¬ P 4 →
  ∀ n, n ≤ 4 → ¬ P n :=
sorry

end induction_principle_l632_63260


namespace parabola_vertex_l632_63282

/-- The parabola defined by y = -x^2 + cx + d -/
noncomputable def parabola (c d : ℝ) (x : ℝ) : ℝ := -x^2 + c*x + d

/-- The set of x values satisfying the inequality -x^2 + cx + d ≤ 0 -/
def inequality_solution (c d : ℝ) : Set ℝ := {x | x ∈ Set.Icc (-7) 3 ∪ Set.Ici 9}

/-- The vertex of a parabola -/
structure Vertex where
  x : ℝ
  y : ℝ

theorem parabola_vertex (c d : ℝ) :
  (inequality_solution c d = {x | x ∈ Set.Icc (-7) 3 ∪ Set.Ici 9}) →
  (∃ (vertex : Vertex), vertex.x = 1 ∧ vertex.y = -62 ∧
    ∀ (x : ℝ), parabola c d x ≤ parabola c d vertex.x) :=
sorry

end parabola_vertex_l632_63282


namespace square_comparison_l632_63269

theorem square_comparison (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a * b * Real.sqrt (a^2 + b^2)) / (a^2 + a * b + b^2) < (a * b) / (a + b) := by
  sorry

end square_comparison_l632_63269


namespace leak_emptying_time_l632_63280

/-- Given a pipe that fills a tank in 12 hours and a leak that causes the tank to take 20 hours to fill when both are active, prove that the leak alone will empty the full tank in 30 hours. -/
theorem leak_emptying_time (pipe_fill_rate : ℝ) (combined_fill_rate : ℝ) (leak_empty_rate : ℝ) :
  pipe_fill_rate = 1 / 12 →
  combined_fill_rate = 1 / 20 →
  pipe_fill_rate - leak_empty_rate = combined_fill_rate →
  1 / leak_empty_rate = 30 := by
  sorry

#check leak_emptying_time

end leak_emptying_time_l632_63280


namespace pascal_triangle_row20_element7_l632_63248

theorem pascal_triangle_row20_element7 : Nat.choose 20 6 = 38760 := by
  sorry

end pascal_triangle_row20_element7_l632_63248


namespace sqrt_sum_of_powers_l632_63285

theorem sqrt_sum_of_powers : Real.sqrt (5^4 + 5^4 + 5^4 + 2^4) = Real.sqrt 1891 := by
  sorry

end sqrt_sum_of_powers_l632_63285


namespace inequality_1_inequality_2_l632_63250

-- Define the solution set for the first inequality
def solution_set_1 : Set ℝ := {x | x < -1 ∨ x > 5}

-- Theorem for the first inequality
theorem inequality_1 : ∀ x : ℝ, -x^2 + 4*x + 5 < 0 ↔ x ∈ solution_set_1 := by sorry

-- Define the solution set for the second inequality
def solution_set_2 (a : ℝ) : Set ℝ :=
  if a = -1 then ∅ 
  else if a > -1 then {x | -1 < x ∧ x < a}
  else {x | a < x ∧ x < -1}

-- Theorem for the second inequality
theorem inequality_2 : ∀ a x : ℝ, x^2 + (1-a)*x - a < 0 ↔ x ∈ solution_set_2 a := by sorry

end inequality_1_inequality_2_l632_63250


namespace typing_task_correct_characters_l632_63216

/-- The total number of characters in the typing task -/
def total_characters : ℕ := 10000

/-- Xiaoyuan's error rate: 1 mistake per 10 characters -/
def xiaoyuan_error_rate : ℚ := 1 / 10

/-- Xiaofang's error rate: 2 mistakes per 10 characters -/
def xiaofang_error_rate : ℚ := 2 / 10

/-- The ratio of correct characters typed by Xiaoyuan to Xiaofang -/
def correct_ratio : ℕ := 2

theorem typing_task_correct_characters :
  ∃ (xiaoyuan_correct xiaofang_correct : ℕ),
    xiaoyuan_correct + xiaofang_correct = 8640 ∧
    xiaoyuan_correct = 2 * xiaofang_correct ∧
    xiaoyuan_correct = total_characters * (1 - xiaoyuan_error_rate) ∧
    xiaofang_correct = total_characters * (1 - xiaofang_error_rate) :=
sorry

end typing_task_correct_characters_l632_63216


namespace givenSampleIsValidSystematic_l632_63270

/-- Checks if a list of integers represents a valid systematic sample -/
def isValidSystematicSample (sample : List Nat) (populationSize : Nat) : Prop :=
  let n := sample.length
  ∃ k : Nat,
    k > 0 ∧
    (∀ i : Fin n, sample[i] = k * (i + 1)) ∧
    sample.all (· ≤ populationSize)

/-- The given sample -/
def givenSample : List Nat := [3, 13, 23, 33, 43]

/-- The theorem stating that the given sample is a valid systematic sample -/
theorem givenSampleIsValidSystematic :
  isValidSystematicSample givenSample 50 := by
  sorry


end givenSampleIsValidSystematic_l632_63270


namespace geometric_sequence_product_l632_63297

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = r * b n

/-- The 9th term is the arithmetic mean of 1 and 3 -/
def ninth_term_is_mean (b : ℕ → ℝ) : Prop :=
  b 9 = (1 + 3) / 2

theorem geometric_sequence_product (b : ℕ → ℝ) 
  (h1 : geometric_sequence b) 
  (h2 : ninth_term_is_mean b) : 
  b 2 * b 16 = 4 := by
sorry

end geometric_sequence_product_l632_63297


namespace movie_outing_cost_is_36_l632_63273

/-- Represents the cost of a movie outing for a family -/
def MovieOutingCost (ticket_price : ℚ) (popcorn_ratio : ℚ) (soda_ratio : ℚ) 
  (num_tickets : ℕ) (num_popcorn : ℕ) (num_soda : ℕ) : ℚ :=
  let popcorn_price := ticket_price * popcorn_ratio
  let soda_price := popcorn_price * soda_ratio
  (ticket_price * num_tickets) + (popcorn_price * num_popcorn) + (soda_price * num_soda)

/-- Theorem stating that the total cost for the family's movie outing is $36 -/
theorem movie_outing_cost_is_36 : 
  MovieOutingCost 5 (80/100) (50/100) 4 2 4 = 36 := by
  sorry

end movie_outing_cost_is_36_l632_63273


namespace f_neg_nine_equals_neg_three_l632_63275

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_neg_nine_equals_neg_three
  (f : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_pos : ∀ x > 0, f x = Real.sqrt x) :
  f (-9) = -3 := by
  sorry

end f_neg_nine_equals_neg_three_l632_63275


namespace haley_initial_lives_l632_63286

theorem haley_initial_lives : 
  ∀ (initial_lives : ℕ), 
    (initial_lives - 4 + 36 = 46) → 
    initial_lives = 14 := by
  sorry

end haley_initial_lives_l632_63286


namespace pear_sales_l632_63226

theorem pear_sales (morning_sales afternoon_sales total_sales : ℕ) : 
  afternoon_sales = 2 * morning_sales →
  total_sales = morning_sales + afternoon_sales →
  total_sales = 390 →
  afternoon_sales = 260 := by
sorry

end pear_sales_l632_63226


namespace estimate_sqrt_expression_l632_63210

theorem estimate_sqrt_expression :
  ∀ (x : ℝ), (1.4 < Real.sqrt 2 ∧ Real.sqrt 2 < 1.5) →
  (6 < (3 * Real.sqrt 6 + Real.sqrt 12) * Real.sqrt (1/3) ∧
   (3 * Real.sqrt 6 + Real.sqrt 12) * Real.sqrt (1/3) < 7) := by
  sorry

end estimate_sqrt_expression_l632_63210


namespace smaller_angle_at_4_oclock_l632_63207

/-- The number of hours on a clock face -/
def clock_hours : ℕ := 12

/-- The number of degrees in a full circle -/
def full_circle_degrees : ℕ := 360

/-- The hour we're considering -/
def target_hour : ℕ := 4

/-- Calculates the angle between clock hands at a given hour -/
def clock_angle (hour : ℕ) : ℕ := 
  (hour * full_circle_degrees) / clock_hours

theorem smaller_angle_at_4_oclock : 
  min (clock_angle target_hour) (full_circle_degrees - clock_angle target_hour) = 120 := by
  sorry

end smaller_angle_at_4_oclock_l632_63207


namespace intersection_chord_length_l632_63206

theorem intersection_chord_length :
  let line : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1}
  let circle : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = 8}
  let intersection := line ∩ circle
  ∃ (A B : ℝ × ℝ), A ∈ intersection ∧ B ∈ intersection ∧ A ≠ B ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 30 :=
by sorry

end intersection_chord_length_l632_63206


namespace not_all_prime_l632_63233

theorem not_all_prime (a₁ a₂ a₃ : ℕ) : 
  a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₂ ≠ a₃ →
  a₁ > 0 ∧ a₂ > 0 ∧ a₃ > 0 →
  a₁ ∣ (a₂ + a₃ + a₂ * a₃) →
  a₂ ∣ (a₃ + a₁ + a₃ * a₁) →
  a₃ ∣ (a₁ + a₂ + a₁ * a₂) →
  ¬(Prime a₁ ∧ Prime a₂ ∧ Prime a₃) :=
by sorry

end not_all_prime_l632_63233


namespace opposite_of_seven_l632_63295

/-- The opposite of a number is the number that, when added to the original number, results in zero. -/
def opposite (a : ℝ) : ℝ := -a

/-- Theorem: The opposite of 7 is -7. -/
theorem opposite_of_seven : opposite 7 = -7 := by
  sorry

end opposite_of_seven_l632_63295


namespace congruence_solution_l632_63213

theorem congruence_solution : ∃! n : ℤ, 0 ≤ n ∧ n < 17 ∧ 38574 ≡ n [ZMOD 17] ∧ n = 1 := by
  sorry

end congruence_solution_l632_63213


namespace neon_signs_blink_together_l632_63229

theorem neon_signs_blink_together (a b : ℕ) (ha : a = 9) (hb : b = 15) : 
  Nat.lcm a b = 45 := by
  sorry

end neon_signs_blink_together_l632_63229


namespace certain_number_equation_l632_63264

theorem certain_number_equation (x : ℝ) : x = 25 ↔ 0.8 * 45 = 4/5 * x + 16 := by sorry

end certain_number_equation_l632_63264


namespace rectangle_triangle_perimeter_l632_63236

/-- A rectangle ABCD with an equilateral triangle CMN where M is on AB -/
structure RectangleWithTriangle where
  /-- Length of rectangle ABCD -/
  length : ℝ
  /-- Width of rectangle ABCD -/
  width : ℝ
  /-- Distance AM, where M is the point on AB where the triangle meets the rectangle -/
  x : ℝ

/-- The perimeter of the equilateral triangle CMN in the RectangleWithTriangle -/
def triangle_perimeter (r : RectangleWithTriangle) : ℝ :=
  3 * (r.x^2 + 1)

theorem rectangle_triangle_perimeter 
  (r : RectangleWithTriangle) 
  (h1 : r.length = 2) 
  (h2 : r.width = 1) 
  (h3 : 0 ≤ r.x) 
  (h4 : r.x ≤ 2) : 
  ∃ (p : ℝ), triangle_perimeter r = p := by
  sorry

end rectangle_triangle_perimeter_l632_63236


namespace square_of_negative_sqrt_two_l632_63279

theorem square_of_negative_sqrt_two : (-Real.sqrt 2)^2 = 2 := by
  sorry

end square_of_negative_sqrt_two_l632_63279


namespace inscribed_triangle_theorem_l632_63246

-- Define the parabola C: y^2 = 4x
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the point M
def M : ℝ × ℝ := (1, 2)

-- Define the fixed point P
def P : ℝ × ℝ := (5, -2)

-- Define a line passing through two points
def line_through (p1 p2 : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, p = (1 - t) • p1 + t • p2}

-- Define a right triangle
def right_triangle (A B M : ℝ × ℝ) : Prop :=
  (A.1 - M.1) * (B.1 - M.1) + (A.2 - M.2) * (B.2 - M.2) = 0

-- Define the theorem
theorem inscribed_triangle_theorem (A B : ℝ × ℝ) :
  parabola A.1 A.2 →
  parabola B.1 B.2 →
  right_triangle A B M →
  (∃ (N : ℝ × ℝ), N ∈ line_through A B ∧ 
    (N.1 - M.1) * (B.1 - A.1) + (N.2 - M.2) * (B.2 - A.2) = 0) →
  (P ∈ line_through A B) ∧
  (∀ (x y : ℝ), x ≠ 1 → ((x - 3)^2 + y^2 = 8 ↔ 
    (∃ (A' B' : ℝ × ℝ), parabola A'.1 A'.2 ∧ parabola B'.1 B'.2 ∧
      right_triangle A' B' M ∧ (x, y) ∈ line_through A' B' ∧
      (x - M.1) * (B'.1 - A'.1) + (y - M.2) * (B'.2 - A'.2) = 0))) := by
  sorry

end inscribed_triangle_theorem_l632_63246


namespace book_reading_time_l632_63200

theorem book_reading_time (chapters : ℕ) (total_pages : ℕ) (pages_per_day : ℕ) : 
  chapters = 41 → total_pages = 450 → pages_per_day = 15 → 
  (total_pages / pages_per_day : ℕ) = 30 := by
sorry

end book_reading_time_l632_63200


namespace clown_mobile_distribution_l632_63235

theorem clown_mobile_distribution (total_clowns : ℕ) (num_mobiles : ℕ) (clowns_per_mobile : ℕ) :
  total_clowns = 140 →
  num_mobiles = 5 →
  total_clowns = num_mobiles * clowns_per_mobile →
  clowns_per_mobile = 28 := by
  sorry

end clown_mobile_distribution_l632_63235


namespace prob_6_or_less_l632_63247

/-- The probability of an archer hitting 9 rings or more in one shot. -/
def p_9_or_more : ℝ := 0.5

/-- The probability of an archer hitting exactly 8 rings in one shot. -/
def p_8 : ℝ := 0.2

/-- The probability of an archer hitting exactly 7 rings in one shot. -/
def p_7 : ℝ := 0.1

/-- Theorem: The probability of an archer hitting 6 rings or less in one shot is 0.2. -/
theorem prob_6_or_less : 1 - (p_9_or_more + p_8 + p_7) = 0.2 := by
  sorry

end prob_6_or_less_l632_63247


namespace garden_fencing_theorem_l632_63290

/-- Represents a rectangular garden -/
structure RectangularGarden where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangular garden -/
def perimeter (garden : RectangularGarden) : ℝ :=
  2 * (garden.length + garden.width)

theorem garden_fencing_theorem :
  ∀ (garden : RectangularGarden),
    garden.length = 50 →
    garden.length = 2 * garden.width →
    perimeter garden = 150 := by
  sorry

end garden_fencing_theorem_l632_63290


namespace modulus_of_z_l632_63256

theorem modulus_of_z (z : ℂ) (h : (1 + Complex.I) * z = 1 - Complex.I) : Complex.abs z = Real.sqrt 2 := by
  sorry

end modulus_of_z_l632_63256


namespace simplify_expression_l632_63230

theorem simplify_expression (x : ℝ) : 3 - (2 - (1 + (2 * (1 - (3 - 2*x))))) = 8 - 4*x := by
  sorry

end simplify_expression_l632_63230


namespace two_hats_on_first_maximizes_sum_optimal_distribution_l632_63293

/-- The number of hats in the hat box -/
def total_hats : ℕ := 21

/-- The number of caps in the hat box -/
def total_caps : ℕ := 18

/-- The capacity of the first shelf -/
def first_shelf_capacity : ℕ := 20

/-- The capacity of the second shelf -/
def second_shelf_capacity : ℕ := 19

/-- The percentage of hats on a shelf given the number of hats and total items -/
def hat_percentage (hats : ℕ) (total : ℕ) : ℚ :=
  (hats : ℚ) / (total : ℚ) * 100

/-- The sum of hat percentages for a given distribution -/
def sum_of_percentages (hats_on_first : ℕ) : ℚ :=
  hat_percentage hats_on_first first_shelf_capacity +
  hat_percentage (total_hats - hats_on_first) second_shelf_capacity

/-- Theorem stating that 2 hats on the first shelf maximizes the sum of percentages -/
theorem two_hats_on_first_maximizes_sum :
  ∀ x : ℕ, x ≤ total_hats → sum_of_percentages 2 ≥ sum_of_percentages x :=
sorry

/-- Corollary stating the optimal distribution of hats -/
theorem optimal_distribution :
  sum_of_percentages 2 = hat_percentage 2 first_shelf_capacity +
                         hat_percentage 19 second_shelf_capacity :=
sorry

end two_hats_on_first_maximizes_sum_optimal_distribution_l632_63293


namespace optimal_newspaper_sales_l632_63289

/-- Represents the newsstand's daily newspaper sales and profit calculation. -/
structure NewspaperSales where
  buyPrice : ℚ
  sellPrice : ℚ
  returnPrice : ℚ
  highDemandDays : ℕ
  lowDemandDays : ℕ
  highDemandAmount : ℕ
  lowDemandAmount : ℕ

/-- Calculates the monthly profit for a given number of daily purchases. -/
def monthlyProfit (sales : NewspaperSales) (dailyPurchase : ℕ) : ℚ :=
  sorry

/-- Theorem stating the optimal daily purchase and maximum monthly profit. -/
theorem optimal_newspaper_sales :
  ∃ (sales : NewspaperSales),
    sales.buyPrice = 24/100 ∧
    sales.sellPrice = 40/100 ∧
    sales.returnPrice = 8/100 ∧
    sales.highDemandDays = 20 ∧
    sales.lowDemandDays = 10 ∧
    sales.highDemandAmount = 300 ∧
    sales.lowDemandAmount = 200 ∧
    (∀ x : ℕ, monthlyProfit sales x ≤ monthlyProfit sales 300) ∧
    monthlyProfit sales 300 = 1120 := by
  sorry

end optimal_newspaper_sales_l632_63289


namespace quadratic_root_implies_a_l632_63201

theorem quadratic_root_implies_a (a : ℝ) : 
  (∃ x : ℝ, x^2 - 3*x - a = 0) ∧ ((-1)^2 - 3*(-1) - a = 0) → a = 4 := by
  sorry

end quadratic_root_implies_a_l632_63201


namespace sequence_property_l632_63294

theorem sequence_property (m : ℤ) (a : ℕ → ℤ) (r s : ℕ) :
  (|m| ≥ 2) →
  (a 1 ≠ 0 ∨ a 2 ≠ 0) →
  (∀ n : ℕ, a (n + 2) = a (n + 1) - m * a n) →
  (r > s) →
  (s ≥ 2) →
  (a r = a s) →
  (a r = a 1) →
  (r - s : ℤ) ≥ |m| :=
by sorry

end sequence_property_l632_63294


namespace min_value_sum_of_reciprocals_l632_63257

theorem min_value_sum_of_reciprocals (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_6 : a + b + c = 6) : 
  (9 / a) + (16 / b) + (25 / c) ≥ 24 := by
  sorry

end min_value_sum_of_reciprocals_l632_63257


namespace inequality_proof_l632_63231

theorem inequality_proof (x y z : ℝ) (h_nonneg_x : x ≥ 0) (h_nonneg_y : y ≥ 0) (h_nonneg_z : z ≥ 0)
  (h_condition : x^2 + y^2 + z^2 = x + y + z) :
  (x + 1) / Real.sqrt (x^5 + x + 1) + (y + 1) / Real.sqrt (y^5 + y + 1) + 
  (z + 1) / Real.sqrt (z^5 + z + 1) ≥ 3 ∧
  ((x + 1) / Real.sqrt (x^5 + x + 1) + (y + 1) / Real.sqrt (y^5 + y + 1) + 
   (z + 1) / Real.sqrt (z^5 + z + 1) = 3 ↔ x = 0 ∧ y = 0 ∧ z = 0) :=
by sorry

end inequality_proof_l632_63231
