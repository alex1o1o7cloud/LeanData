import Mathlib

namespace NUMINAMATH_CALUDE_hexagon_area_l1932_193280

/-- A regular hexagon divided by three diagonals -/
structure RegularHexagon where
  /-- The area of one small triangle formed by the diagonals -/
  small_triangle_area : ℝ
  /-- The total number of small triangles in the hexagon -/
  total_triangles : ℕ
  /-- The number of shaded triangles -/
  shaded_triangles : ℕ
  /-- The total shaded area -/
  shaded_area : ℝ
  /-- The hexagon is divided into 12 congruent triangles -/
  triangle_count : total_triangles = 12
  /-- Two regions (equivalent to 5 small triangles) are shaded -/
  shaded_count : shaded_triangles = 5
  /-- The total shaded area is 20 cm² -/
  shaded_area_value : shaded_area = 20

/-- The theorem stating the area of the hexagon -/
theorem hexagon_area (h : RegularHexagon) : h.total_triangles * h.small_triangle_area = 48 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_area_l1932_193280


namespace NUMINAMATH_CALUDE_greatest_number_l1932_193202

theorem greatest_number (A B C : ℤ) : 
  A = 95 - 35 →
  B = A + 12 →
  C = B - 19 →
  A ≠ B ∧ B ≠ C ∧ A ≠ C →
  B > A ∧ B > C :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_number_l1932_193202


namespace NUMINAMATH_CALUDE_career_preference_theorem_l1932_193278

/-- Represents the degrees in a circle graph for a career preference -/
def career_preference_degrees (male_ratio female_ratio : ℚ) 
  (male_preference female_preference : ℚ) : ℚ :=
  ((male_ratio * male_preference + female_ratio * female_preference) / 
   (male_ratio + female_ratio)) * 360

/-- Theorem: The degrees for the given career preference -/
theorem career_preference_theorem : 
  career_preference_degrees 2 3 (1/4) (3/4) = 198 := by
  sorry

end NUMINAMATH_CALUDE_career_preference_theorem_l1932_193278


namespace NUMINAMATH_CALUDE_chandler_can_buy_bike_l1932_193274

/-- The cost of the mountain bike in dollars -/
def bike_cost : ℕ := 800

/-- The total amount of gift money Chandler received in dollars -/
def gift_money : ℕ := 100 + 50 + 20 + 30

/-- The amount Chandler earns per week from his paper route in dollars -/
def weekly_earnings : ℕ := 20

/-- The number of weeks Chandler needs to save to buy the mountain bike -/
def weeks_to_save : ℕ := 30

/-- Theorem stating that Chandler can buy the bike after saving for the calculated number of weeks -/
theorem chandler_can_buy_bike :
  gift_money + weekly_earnings * weeks_to_save = bike_cost :=
by sorry

end NUMINAMATH_CALUDE_chandler_can_buy_bike_l1932_193274


namespace NUMINAMATH_CALUDE_females_who_chose_malt_l1932_193257

/-- Represents the number of cheerleaders who chose each drink -/
structure CheerleaderChoices where
  coke : ℕ
  malt : ℕ

/-- Represents the gender distribution of cheerleaders -/
structure CheerleaderGenders where
  males : ℕ
  females : ℕ

theorem females_who_chose_malt 
  (choices : CheerleaderChoices)
  (genders : CheerleaderGenders)
  (h1 : genders.males = 10)
  (h2 : genders.females = 16)
  (h3 : choices.malt = 2 * choices.coke)
  (h4 : choices.coke + choices.malt = genders.males + genders.females)
  (h5 : choices.malt ≥ genders.males)
  (h6 : genders.males = 6) :
  choices.malt - genders.males = 10 := by
sorry

end NUMINAMATH_CALUDE_females_who_chose_malt_l1932_193257


namespace NUMINAMATH_CALUDE_stratified_sampling_survey_size_l1932_193298

/-- Proves that the total number of surveyed students is 10 given the conditions of the problem -/
theorem stratified_sampling_survey_size 
  (total_students : ℕ) 
  (female_students : ℕ) 
  (sampled_females : ℕ) 
  (h1 : total_students = 50)
  (h2 : female_students = 20)
  (h3 : sampled_females = 4)
  (h4 : female_students < total_students) :
  ∃ (surveyed_students : ℕ), 
    surveyed_students * female_students = sampled_females * total_students ∧ 
    surveyed_students = 10 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_survey_size_l1932_193298


namespace NUMINAMATH_CALUDE_valid_draw_count_l1932_193297

def total_cards : ℕ := 16
def cards_per_color : ℕ := 4
def cards_drawn : ℕ := 3

def valid_draw (total : ℕ) (per_color : ℕ) (drawn : ℕ) : ℕ :=
  Nat.choose total drawn - 
  4 * Nat.choose per_color drawn - 
  Nat.choose per_color 2 * Nat.choose (total - per_color) 1

theorem valid_draw_count :
  valid_draw total_cards cards_per_color cards_drawn = 472 := by
  sorry

end NUMINAMATH_CALUDE_valid_draw_count_l1932_193297


namespace NUMINAMATH_CALUDE_sphere_cylinder_volume_difference_l1932_193273

/-- The volume of the space inside a sphere and outside an inscribed right cylinder -/
theorem sphere_cylinder_volume_difference (r_sphere r_cylinder : ℝ) (h_sphere : r_sphere = 7) (h_cylinder : r_cylinder = 4) :
  let h_cylinder := 2 * Real.sqrt 33
  let v_sphere := (4 / 3) * π * r_sphere ^ 3
  let v_cylinder := π * r_cylinder ^ 2 * h_cylinder
  v_sphere - v_cylinder = (1372 / 3 - 32 * Real.sqrt 33) * π :=
by sorry

end NUMINAMATH_CALUDE_sphere_cylinder_volume_difference_l1932_193273


namespace NUMINAMATH_CALUDE_white_coinciding_pairs_l1932_193255

/-- Represents the number of triangles of each color in each half of the figure -/
structure TriangleCount where
  red : ℕ
  blue : ℕ
  green : ℕ
  white : ℕ

/-- Represents the number of coinciding pairs of triangles when the figure is folded -/
structure CoincidingPairs where
  red_red : ℕ
  blue_blue : ℕ
  green_green : ℕ
  red_white : ℕ
  green_blue : ℕ

/-- Theorem stating that the number of coinciding white triangle pairs is 4 -/
theorem white_coinciding_pairs
  (half_count : TriangleCount)
  (coinciding : CoincidingPairs)
  (h1 : half_count.red = 4)
  (h2 : half_count.blue = 4)
  (h3 : half_count.green = 2)
  (h4 : half_count.white = 6)
  (h5 : coinciding.red_red = 3)
  (h6 : coinciding.blue_blue = 2)
  (h7 : coinciding.green_green = 1)
  (h8 : coinciding.red_white = 2)
  (h9 : coinciding.green_blue = 1) :
  ∃ (white_pairs : ℕ), white_pairs = 4 ∧ 
  white_pairs = half_count.white - coinciding.red_white := by
  sorry

end NUMINAMATH_CALUDE_white_coinciding_pairs_l1932_193255


namespace NUMINAMATH_CALUDE_rectangle_y_value_l1932_193218

/-- Given a rectangle with vertices at (-2, y), (6, y), (-2, 2), and (6, 2),
    if the area is 80 square units, then y = 12 -/
theorem rectangle_y_value (y : ℝ) : 
  let vertices : List (ℝ × ℝ) := [(-2, y), (6, y), (-2, 2), (6, 2)]
  let width : ℝ := 6 - (-2)
  let height : ℝ := y - 2
  let area : ℝ := width * height
  (∀ v ∈ vertices, v.1 = -2 ∨ v.1 = 6) ∧
  (∀ v ∈ vertices, v.2 = y ∨ v.2 = 2) ∧
  area = 80 →
  y = 12 := by
sorry

end NUMINAMATH_CALUDE_rectangle_y_value_l1932_193218


namespace NUMINAMATH_CALUDE_bicycle_wheels_count_l1932_193287

theorem bicycle_wheels_count (num_bicycles num_tricycles tricycle_wheels total_wheels : ℕ) 
  (h1 : num_bicycles = 24)
  (h2 : num_tricycles = 14)
  (h3 : tricycle_wheels = 3)
  (h4 : total_wheels = 90)
  (h5 : total_wheels = num_bicycles * bicycle_wheels + num_tricycles * tricycle_wheels) :
  bicycle_wheels = 2 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_wheels_count_l1932_193287


namespace NUMINAMATH_CALUDE_yardley_snowfall_l1932_193216

/-- The total snowfall in Yardley is the sum of morning and afternoon snowfall -/
theorem yardley_snowfall (morning_snowfall afternoon_snowfall : ℚ) 
  (h1 : morning_snowfall = 0.125)
  (h2 : afternoon_snowfall = 0.5) :
  morning_snowfall + afternoon_snowfall = 0.625 := by
  sorry

end NUMINAMATH_CALUDE_yardley_snowfall_l1932_193216


namespace NUMINAMATH_CALUDE_rajs_house_bathrooms_l1932_193295

/-- Represents the floor plan of Raj's house -/
structure HouseFloorPlan where
  total_area : ℕ
  bedroom_count : ℕ
  bedroom_side : ℕ
  bathroom_length : ℕ
  bathroom_width : ℕ
  kitchen_area : ℕ

/-- Calculates the number of bathrooms in Raj's house -/
def calculate_bathrooms (house : HouseFloorPlan) : ℕ :=
  let bedroom_area := house.bedroom_count * house.bedroom_side * house.bedroom_side
  let living_area := house.kitchen_area
  let remaining_area := house.total_area - (bedroom_area + house.kitchen_area + living_area)
  let bathroom_area := house.bathroom_length * house.bathroom_width
  remaining_area / bathroom_area

/-- Theorem stating that Raj's house has exactly 2 bathrooms -/
theorem rajs_house_bathrooms :
  let house : HouseFloorPlan := {
    total_area := 1110,
    bedroom_count := 4,
    bedroom_side := 11,
    bathroom_length := 6,
    bathroom_width := 8,
    kitchen_area := 265
  }
  calculate_bathrooms house = 2 := by
  sorry


end NUMINAMATH_CALUDE_rajs_house_bathrooms_l1932_193295


namespace NUMINAMATH_CALUDE_square_root_of_4096_l1932_193220

theorem square_root_of_4096 (x : ℝ) (h1 : x > 0) (h2 : x^2 = 4096) : x = 64 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_4096_l1932_193220


namespace NUMINAMATH_CALUDE_nested_abs_ratio_values_l1932_193245

/-- Recursive function representing nested absolute value operations -/
def nestedAbs (n : ℕ) (x y : ℝ) : ℝ :=
  match n with
  | 0 => x
  | n + 1 => |nestedAbs n x y - y|

/-- The equation condition from the problem -/
def equationCondition (x y : ℝ) : Prop :=
  nestedAbs 2019 x y = nestedAbs 2019 y x

/-- The theorem statement -/
theorem nested_abs_ratio_values (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_eq : equationCondition x y) :
  x / y = 1/3 ∨ x / y = 1 ∨ x / y = 3 := by
  sorry

end NUMINAMATH_CALUDE_nested_abs_ratio_values_l1932_193245


namespace NUMINAMATH_CALUDE_range_of_a_l1932_193256

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x - a < 0) ∧ 
  (∃ x : ℝ, x^2 + x + 2*a - 1 ≤ 0) → 
  -1 < a ∧ a ≤ 5/8 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1932_193256


namespace NUMINAMATH_CALUDE_range_of_m_range_of_a_l1932_193228

-- Part I
theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, 1/3 < x ∧ x < 1/2 → |x - m| < 1) → 
  -1/2 ≤ m ∧ m ≤ 4/3 := by
sorry

-- Part II
theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, |x - 3| + |x - 5| < a) →
  a > 2 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_range_of_a_l1932_193228


namespace NUMINAMATH_CALUDE_expression_value_l1932_193214

theorem expression_value (a b c d m : ℚ) 
  (h1 : a = -b) 
  (h2 : c * d = 1) 
  (h3 : |m| = 4) : 
  2*a - 5*c*d - m + 2*b = -9 ∨ 2*a - 5*c*d - m + 2*b = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1932_193214


namespace NUMINAMATH_CALUDE_faulty_balance_inequality_l1932_193212

/-- A faulty balance with unequal arm lengths -/
structure FaultyBalance where
  m : ℝ  -- Length of one arm
  n : ℝ  -- Length of the other arm
  h_positive_m : m > 0
  h_positive_n : n > 0
  h_unequal : m ≠ n

/-- Measurements obtained from weighing an object on a faulty balance -/
structure Measurements (fb : FaultyBalance) where
  a : ℝ  -- Measurement on one side
  b : ℝ  -- Measurement on the other side
  G : ℝ  -- True weight of the object
  h_positive_a : a > 0
  h_positive_b : b > 0
  h_positive_G : G > 0
  h_relation_a : fb.m * a = fb.n * G
  h_relation_b : fb.n * b = fb.m * G

/-- The arithmetic mean of the measurements is greater than the true weight -/
theorem faulty_balance_inequality (fb : FaultyBalance) (m : Measurements fb) :
  (m.a + m.b) / 2 > m.G := by
  sorry

end NUMINAMATH_CALUDE_faulty_balance_inequality_l1932_193212


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l1932_193263

theorem rectangle_diagonal (side : ℝ) (area : ℝ) (diagonal : ℝ) : 
  side = 15 → area = 120 → diagonal = 17 → 
  ∃ other_side : ℝ, 
    area = side * other_side ∧ 
    diagonal^2 = side^2 + other_side^2 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l1932_193263


namespace NUMINAMATH_CALUDE_specific_l_shape_area_l1932_193215

/-- The area of an "L" shape formed by removing a smaller rectangle from a larger rectangle --/
def l_shape_area (length width subtract_length subtract_width : ℕ) : ℕ :=
  length * width - (length - subtract_length) * (width - subtract_width)

/-- Theorem: The area of the specific "L" shape is 42 square units --/
theorem specific_l_shape_area : l_shape_area 10 7 3 3 = 42 := by
  sorry

end NUMINAMATH_CALUDE_specific_l_shape_area_l1932_193215


namespace NUMINAMATH_CALUDE_factorial_sum_power_of_two_solutions_l1932_193286

def is_solution (a b c n : ℕ) : Prop :=
  Nat.factorial a + Nat.factorial b + Nat.factorial c = 2^n

theorem factorial_sum_power_of_two_solutions :
  ∀ a b c n : ℕ,
    is_solution a b c n ↔
      ((a, b, c) = (1, 1, 2) ∧ n = 2) ∨
      ((a, b, c) = (1, 1, 3) ∧ n = 3) ∨
      ((a, b, c) = (2, 3, 4) ∧ n = 5) ∨
      ((a, b, c) = (2, 3, 5) ∧ n = 7) :=
by sorry

end NUMINAMATH_CALUDE_factorial_sum_power_of_two_solutions_l1932_193286


namespace NUMINAMATH_CALUDE_paco_cookies_theorem_l1932_193261

/-- The number of cookies Paco ate -/
def cookies_eaten (initial : ℕ) (remaining : ℕ) : ℕ := initial - remaining

theorem paco_cookies_theorem (initial : ℕ) (remaining : ℕ) 
  (h1 : initial = 28) 
  (h2 : remaining = 7) : 
  cookies_eaten initial remaining = 21 := by
  sorry

end NUMINAMATH_CALUDE_paco_cookies_theorem_l1932_193261


namespace NUMINAMATH_CALUDE_percentage_problem_l1932_193239

theorem percentage_problem :
  let percentage := 6.620000000000001
  let value := 66.2
  let x := value / (percentage / 100)
  x = 1000 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1932_193239


namespace NUMINAMATH_CALUDE_largest_n_with_special_divisor_property_l1932_193226

theorem largest_n_with_special_divisor_property : ∃ N : ℕ, 
  (∀ m : ℕ, m > N → ¬(∃ d₁ d₂ d₃ : ℕ, 
    d₁ ∣ m ∧ d₂ ∣ m ∧ d₃ ∣ m ∧
    (∀ x : ℕ, x ∣ m → x = 1 ∨ x ≥ d₁) ∧
    (∀ x : ℕ, x ∣ m → x = 1 ∨ x = d₁ ∨ x ≥ d₂) ∧
    (∃ y z : ℕ, y ∣ m ∧ z ∣ m ∧ y > d₃ ∧ z > y) ∧
    d₃ = 21 * d₁)) ∧
  (∃ d₁ d₂ d₃ : ℕ, 
    d₁ ∣ N ∧ d₂ ∣ N ∧ d₃ ∣ N ∧
    (∀ x : ℕ, x ∣ N → x = 1 ∨ x ≥ d₁) ∧
    (∀ x : ℕ, x ∣ N → x = 1 ∨ x = d₁ ∨ x ≥ d₂) ∧
    (∃ y z : ℕ, y ∣ N ∧ z ∣ N ∧ y > d₃ ∧ z > y) ∧
    d₃ = 21 * d₁) ∧
  N = 441 := by
  sorry

end NUMINAMATH_CALUDE_largest_n_with_special_divisor_property_l1932_193226


namespace NUMINAMATH_CALUDE_weeks_to_buy_bike_l1932_193266

def mountain_bike_cost : ℕ := 600
def birthday_money : ℕ := 60 + 40 + 20 + 30
def weekly_earnings : ℕ := 18

theorem weeks_to_buy_bike : 
  ∃ (weeks : ℕ), birthday_money + weeks * weekly_earnings = mountain_bike_cost ∧ weeks = 25 :=
by sorry

end NUMINAMATH_CALUDE_weeks_to_buy_bike_l1932_193266


namespace NUMINAMATH_CALUDE_fraction_of_special_number_in_list_l1932_193260

theorem fraction_of_special_number_in_list (l : List ℝ) (n : ℝ) :
  l.length = 21 →
  l.Nodup →
  n ∈ l →
  n = 4 * ((l.sum - n) / 20) →
  n = (1 / 6) * l.sum := by
sorry

end NUMINAMATH_CALUDE_fraction_of_special_number_in_list_l1932_193260


namespace NUMINAMATH_CALUDE_opposite_of_one_over_23_l1932_193283

theorem opposite_of_one_over_23 : 
  -(1 / 23) = -1 / 23 := by sorry

end NUMINAMATH_CALUDE_opposite_of_one_over_23_l1932_193283


namespace NUMINAMATH_CALUDE_negation_of_all_x_squared_positive_negation_is_true_l1932_193267

theorem negation_of_all_x_squared_positive :
  (¬ (∀ x : ℝ, x^2 > 0)) ↔ (∃ x : ℝ, x^2 ≤ 0) :=
by sorry

theorem negation_is_true : ∃ x : ℝ, x^2 ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_negation_of_all_x_squared_positive_negation_is_true_l1932_193267


namespace NUMINAMATH_CALUDE_range_of_g_l1932_193264

def f (x : ℝ) : ℝ := 2 * x + 3

def g (x : ℝ) : ℝ := f (f (f (f x)))

theorem range_of_g :
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 3 → 29 ≤ g x ∧ g x ≤ 93 :=
by sorry

end NUMINAMATH_CALUDE_range_of_g_l1932_193264


namespace NUMINAMATH_CALUDE_cos_48_degrees_l1932_193224

theorem cos_48_degrees : Real.cos (48 * π / 180) = (Real.sqrt 3 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_48_degrees_l1932_193224


namespace NUMINAMATH_CALUDE_three_lines_intersection_l1932_193207

/-- Three lines intersect at a single point if and only if m = 9 -/
theorem three_lines_intersection (m : ℝ) : 
  (∃ (x y : ℝ), y = 2*x ∧ x + y = 3 ∧ m*x - 2*y - 5 = 0) ↔ m = 9 :=
by sorry

end NUMINAMATH_CALUDE_three_lines_intersection_l1932_193207


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l1932_193258

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ -2}
def B : Set ℝ := {x | x < 1}

-- State the theorem
theorem complement_A_intersect_B :
  (Aᶜ ∩ B) = {x | -2 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l1932_193258


namespace NUMINAMATH_CALUDE_last_two_digits_2018_power_2018_base_7_l1932_193271

theorem last_two_digits_2018_power_2018_base_7 : 
  (2018^2018 : ℕ) % 49 = 32 :=
sorry

end NUMINAMATH_CALUDE_last_two_digits_2018_power_2018_base_7_l1932_193271


namespace NUMINAMATH_CALUDE_unique_a_value_l1932_193230

def A (a : ℝ) : Set ℝ := {2, 3, a^2 - 3*a, a + 2/a + 7}
def B (a : ℝ) : Set ℝ := {|a - 2|, 3}

theorem unique_a_value : ∃! a : ℝ, (4 ∈ A a) ∧ (4 ∉ B a) := by
  sorry

end NUMINAMATH_CALUDE_unique_a_value_l1932_193230


namespace NUMINAMATH_CALUDE_cone_lateral_surface_angle_l1932_193262

/-- Given a cone where the total surface area is three times its base area,
    the central angle of the sector in the lateral surface development diagram is 180 degrees. -/
theorem cone_lateral_surface_angle (r l : ℝ) (h : r > 0) (h' : l > 0) : 
  (π * r^2 + π * r * l = 3 * π * r^2) → 
  (2 * π * r / l) * (180 / π) = 180 := by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_angle_l1932_193262


namespace NUMINAMATH_CALUDE_ratio_of_squares_to_products_l1932_193244

theorem ratio_of_squares_to_products (x y z : ℝ) 
  (h_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z) 
  (h_sum : x + 2*y + 3*z = 0) : 
  (x^2 + y^2 + z^2) / (x*y + y*z + z*x) = -4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_squares_to_products_l1932_193244


namespace NUMINAMATH_CALUDE_cookie_brownie_difference_l1932_193221

/-- Represents the daily consumption of cookies and brownies --/
structure DailyConsumption where
  cookies : Nat
  brownies : Nat

/-- Calculates the remaining items after a week of consumption --/
def remainingItems (initial : Nat) (daily : List Nat) : Nat :=
  max (initial - daily.sum) 0

theorem cookie_brownie_difference :
  let initialCookies : Nat := 60
  let initialBrownies : Nat := 10
  let weeklyConsumption : List DailyConsumption := [
    ⟨2, 1⟩, ⟨4, 2⟩, ⟨3, 1⟩, ⟨5, 1⟩, ⟨4, 3⟩, ⟨3, 2⟩, ⟨2, 1⟩
  ]
  let cookiesLeft := remainingItems initialCookies (weeklyConsumption.map DailyConsumption.cookies)
  let browniesLeft := remainingItems initialBrownies (weeklyConsumption.map DailyConsumption.brownies)
  cookiesLeft - browniesLeft = 37 := by
  sorry

end NUMINAMATH_CALUDE_cookie_brownie_difference_l1932_193221


namespace NUMINAMATH_CALUDE_swallow_pests_calculation_l1932_193213

/-- The number of pests a frog can catch per day -/
def frog_pests : ℕ := 145

/-- The multiplier for how many times more pests a swallow can eliminate compared to a frog -/
def swallow_multiplier : ℕ := 12

/-- The number of pests a swallow can eliminate per day -/
def swallow_pests : ℕ := frog_pests * swallow_multiplier

theorem swallow_pests_calculation : swallow_pests = 1740 := by
  sorry

end NUMINAMATH_CALUDE_swallow_pests_calculation_l1932_193213


namespace NUMINAMATH_CALUDE_smallest_x_quadratic_equation_l1932_193276

theorem smallest_x_quadratic_equation :
  let f : ℝ → ℝ := λ x => 4*x^2 + 6*x + 1
  ∃ x : ℝ, (f x = 5) ∧ (∀ y : ℝ, f y = 5 → x ≤ y) ∧ x = -2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_quadratic_equation_l1932_193276


namespace NUMINAMATH_CALUDE_average_increase_l1932_193200

theorem average_increase (s : Finset ℕ) (f : ℕ → ℝ) :
  s.card = 15 →
  (s.sum f) / s.card = 30 →
  (s.sum (λ x => f x + 15)) / s.card = 45 := by
sorry

end NUMINAMATH_CALUDE_average_increase_l1932_193200


namespace NUMINAMATH_CALUDE_days_at_grandparents_l1932_193238

def vacation_duration : ℕ := 21  -- 3 weeks * 7 days

def travel_to_grandparents : ℕ := 1
def travel_to_brother : ℕ := 1
def stay_at_brother : ℕ := 5
def travel_to_sister : ℕ := 2
def stay_at_sister : ℕ := 5
def travel_home : ℕ := 2

def known_days : ℕ := travel_to_grandparents + travel_to_brother + stay_at_brother + travel_to_sister + stay_at_sister + travel_home

theorem days_at_grandparents :
  vacation_duration - known_days = 5 :=
by sorry

end NUMINAMATH_CALUDE_days_at_grandparents_l1932_193238


namespace NUMINAMATH_CALUDE_crayons_left_correct_l1932_193204

/-- Represents the number of crayons and erasers Paul has -/
structure PaulsCrayonsAndErasers where
  initial_crayons : ℕ
  initial_erasers : ℕ
  remaining_difference : ℕ

/-- Calculates the number of crayons Paul has left -/
def crayons_left (p : PaulsCrayonsAndErasers) : ℕ :=
  p.initial_erasers + p.remaining_difference

theorem crayons_left_correct (p : PaulsCrayonsAndErasers) 
  (h : p.initial_crayons = 531 ∧ p.initial_erasers = 38 ∧ p.remaining_difference = 353) : 
  crayons_left p = 391 := by
  sorry

end NUMINAMATH_CALUDE_crayons_left_correct_l1932_193204


namespace NUMINAMATH_CALUDE_six_digit_divisibility_l1932_193222

theorem six_digit_divisibility (a b : Nat) : 
  (a < 10 ∧ b < 10) →  -- Ensure a and b are single digits
  (201000 + 100 * a + 10 * b + 7) % 11 = 0 →  -- Divisible by 11
  (201000 + 100 * a + 10 * b + 7) % 13 = 0 →  -- Divisible by 13
  10 * a + b = 48 := by
sorry

end NUMINAMATH_CALUDE_six_digit_divisibility_l1932_193222


namespace NUMINAMATH_CALUDE_shoveling_time_l1932_193227

theorem shoveling_time (kevin dave john allison : ℝ)
  (h_kevin : kevin = 12)
  (h_dave : dave = 8)
  (h_john : john = 6)
  (h_allison : allison = 4) :
  (1 / kevin + 1 / dave + 1 / john + 1 / allison)⁻¹ * 60 = 96 := by
  sorry

end NUMINAMATH_CALUDE_shoveling_time_l1932_193227


namespace NUMINAMATH_CALUDE_waiter_customer_count_l1932_193277

/-- Calculates the final number of customers after a series of arrivals and departures. -/
def finalCustomerCount (initial : ℕ) (left1 left2 : ℕ) (arrived1 arrived2 : ℕ) : ℕ :=
  initial - left1 + arrived1 + arrived2 - left2

/-- Theorem stating that given the specific customer movements, the final count is 14. -/
theorem waiter_customer_count : 
  finalCustomerCount 13 5 6 4 8 = 14 := by
  sorry

end NUMINAMATH_CALUDE_waiter_customer_count_l1932_193277


namespace NUMINAMATH_CALUDE_tens_digit_of_3_power_205_l1932_193259

theorem tens_digit_of_3_power_205 : ∃ n : ℕ, 3^205 ≡ 40 + n [ZMOD 100] ∧ n < 10 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_3_power_205_l1932_193259


namespace NUMINAMATH_CALUDE_min_value_of_fraction_lower_bound_achievable_l1932_193289

theorem min_value_of_fraction (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 3) :
  (a + b) / (a * b * c) ≥ 16 / 9 :=
sorry

theorem lower_bound_achievable :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 3 ∧ (a + b) / (a * b * c) = 16 / 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_fraction_lower_bound_achievable_l1932_193289


namespace NUMINAMATH_CALUDE_general_term_max_sum_l1932_193209

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- General term of the sequence
  S : ℕ → ℤ  -- Sum function of the sequence
  sum_3 : S 3 = 42
  sum_6 : S 6 = 57

/-- The general term of the sequence is 20 - 3n -/
theorem general_term (seq : ArithmeticSequence) : 
  ∀ n : ℕ, seq.a n = 20 - 3 * n := by sorry

/-- The sum S_n is maximized when n = 6 -/
theorem max_sum (seq : ArithmeticSequence) : 
  ∃ n : ℕ, ∀ m : ℕ, seq.S n ≥ seq.S m ∧ n = 6 := by sorry

end NUMINAMATH_CALUDE_general_term_max_sum_l1932_193209


namespace NUMINAMATH_CALUDE_slope_of_line_l_l1932_193291

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/16 + y^2/4 = 1

-- Define the point M
def M : ℝ × ℝ := (2, 1)

-- Define the line l passing through M with slope m
def line_l (m : ℝ) (x y : ℝ) : Prop :=
  y - M.2 = m * (x - M.1)

-- Define the intersection points A and B
def intersection_points (m : ℝ) :=
  ∃ (xa ya xb yb : ℝ),
    ellipse xa ya ∧ ellipse xb yb ∧
    line_l m xa ya ∧ line_l m xb yb ∧
    (xa, ya) ≠ (xb, yb)

-- Define M as the trisection point of AB
def M_is_trisection (m : ℝ) :=
  ∃ (xa ya xb yb : ℝ),
    ellipse xa ya ∧ ellipse xb yb ∧
    line_l m xa ya ∧ line_l m xb yb ∧
    2 * M.1 = xa + xb ∧ 2 * M.2 = ya + yb

-- The main theorem
theorem slope_of_line_l :
  ∃ (m : ℝ), intersection_points m ∧ M_is_trisection m ∧
  (m = (-4 + Real.sqrt 7) / 6 ∨ m = (-4 - Real.sqrt 7) / 6) :=
sorry

end NUMINAMATH_CALUDE_slope_of_line_l_l1932_193291


namespace NUMINAMATH_CALUDE_trapezium_other_side_length_l1932_193268

theorem trapezium_other_side_length 
  (a : ℝ) -- Area of the trapezium
  (b : ℝ) -- Length of one parallel side
  (h : ℝ) -- Distance between parallel sides
  (x : ℝ) -- Length of the other parallel side
  (h1 : a = 380) -- Area is 380 square centimeters
  (h2 : b = 18)  -- One parallel side is 18 cm
  (h3 : h = 20)  -- Distance between parallel sides is 20 cm
  (h4 : a = (1/2) * (x + b) * h) -- Area formula for trapezium
  : x = 20 := by
  sorry

end NUMINAMATH_CALUDE_trapezium_other_side_length_l1932_193268


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1932_193246

theorem quadratic_equation_solution (m n : ℝ) : 
  (∀ x, x^2 + m*x - 15 = (x + 5)*(x + n)) → m = 2 ∧ n = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1932_193246


namespace NUMINAMATH_CALUDE_circumcircle_diameter_l1932_193279

/-- Given a triangle ABC with side length a = 2 and angle A = 60°,
    prove that the diameter of its circumcircle is 4√3/3 -/
theorem circumcircle_diameter (a : ℝ) (A : ℝ) (h1 : a = 2) (h2 : A = π/3) :
  (2 * a) / Real.sin A = 4 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_circumcircle_diameter_l1932_193279


namespace NUMINAMATH_CALUDE_octal_subtraction_l1932_193243

/-- Represents a number in base 8 --/
def OctalNum := ℕ

/-- Converts a natural number to its octal representation --/
def toOctal (n : ℕ) : OctalNum :=
  sorry

/-- Performs subtraction in base 8 --/
def octalSub (a b : OctalNum) : OctalNum :=
  sorry

theorem octal_subtraction :
  octalSub (toOctal 126) (toOctal 57) = toOctal 47 :=
sorry

end NUMINAMATH_CALUDE_octal_subtraction_l1932_193243


namespace NUMINAMATH_CALUDE_invest_in_good_B_l1932_193235

def expected_profit (p1 p2 p3 : ℝ) (v1 v2 v3 : ℝ) : ℝ :=
  p1 * v1 + p2 * v2 + p3 * v3

theorem invest_in_good_B (capital : ℝ) 
  (a_p1 a_p2 a_p3 : ℝ) (a_v1 a_v2 a_v3 : ℝ)
  (b_p1 b_p2 b_p3 : ℝ) (b_v1 b_v2 b_v3 : ℝ)
  (ha1 : a_p1 = 0.4) (ha2 : a_p2 = 0.3) (ha3 : a_p3 = 0.3)
  (ha4 : a_v1 = 20000) (ha5 : a_v2 = 30000) (ha6 : a_v3 = -10000)
  (hb1 : b_p1 = 0.6) (hb2 : b_p2 = 0.2) (hb3 : b_p3 = 0.2)
  (hb4 : b_v1 = 20000) (hb5 : b_v2 = 40000) (hb6 : b_v3 = -20000)
  (hcap : capital = 100000) :
  expected_profit b_p1 b_p2 b_p3 b_v1 b_v2 b_v3 > 
  expected_profit a_p1 a_p2 a_p3 a_v1 a_v2 a_v3 := by
  sorry

#check invest_in_good_B

end NUMINAMATH_CALUDE_invest_in_good_B_l1932_193235


namespace NUMINAMATH_CALUDE_intersection_M_N_l1932_193290

def M : Set ℝ := {-1, 0, 1, 2}
def N : Set ℝ := {x | x^2 - x - 2 < 0}

theorem intersection_M_N : M ∩ N = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1932_193290


namespace NUMINAMATH_CALUDE_scissors_in_drawer_l1932_193284

theorem scissors_in_drawer (initial_scissors : ℕ) : initial_scissors = 54 →
  initial_scissors + 22 = 76 := by
  sorry

end NUMINAMATH_CALUDE_scissors_in_drawer_l1932_193284


namespace NUMINAMATH_CALUDE_sum_of_roots_l1932_193249

theorem sum_of_roots (a b : ℝ) : 
  a * (a - 4) = 21 → 
  b * (b - 4) = 21 → 
  a ≠ b → 
  a + b = 4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l1932_193249


namespace NUMINAMATH_CALUDE_tangent_product_equals_two_l1932_193211

theorem tangent_product_equals_two :
  let tan17 := Real.tan (17 * π / 180)
  let tan28 := Real.tan (28 * π / 180)
  (∀ a b, Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b)) →
  17 + 28 = 45 →
  Real.tan (45 * π / 180) = 1 →
  (1 + tan17) * (1 + tan28) = 2 := by
sorry

end NUMINAMATH_CALUDE_tangent_product_equals_two_l1932_193211


namespace NUMINAMATH_CALUDE_element_in_set_l1932_193219

def U : Set Nat := {1, 2, 3, 4, 5}

theorem element_in_set (M : Set Nat) (h : Set.compl M = {1, 3}) : 2 ∈ M := by
  sorry

end NUMINAMATH_CALUDE_element_in_set_l1932_193219


namespace NUMINAMATH_CALUDE_upper_limit_y_l1932_193281

theorem upper_limit_y (x y : ℤ) (h1 : 5 < x) (h2 : x < 8) (h3 : 8 < y) 
  (h4 : ∀ (a b : ℤ), 5 < a → a < 8 → 8 < b → b - a ≤ 7) : y ≤ 14 := by
  sorry

end NUMINAMATH_CALUDE_upper_limit_y_l1932_193281


namespace NUMINAMATH_CALUDE_large_paintings_sold_is_five_l1932_193242

/-- Represents the sale of paintings at an art show -/
structure PaintingSale where
  large_price : ℕ
  small_price : ℕ
  small_count : ℕ
  total_earnings : ℕ

/-- Calculates the number of large paintings sold -/
def large_paintings_sold (sale : PaintingSale) : ℕ :=
  (sale.total_earnings - sale.small_price * sale.small_count) / sale.large_price

/-- Theorem stating that the number of large paintings sold is 5 -/
theorem large_paintings_sold_is_five (sale : PaintingSale)
  (h1 : sale.large_price = 100)
  (h2 : sale.small_price = 80)
  (h3 : sale.small_count = 8)
  (h4 : sale.total_earnings = 1140) :
  large_paintings_sold sale = 5 := by
  sorry

end NUMINAMATH_CALUDE_large_paintings_sold_is_five_l1932_193242


namespace NUMINAMATH_CALUDE_min_value_expression_l1932_193265

/-- Given positive real numbers a and b satisfying a + 3b = 7, 
    the expression 1/(1+a) + 4/(2+b) has a minimum value of (13 + 4√3)/14 -/
theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (heq : a + 3*b = 7) :
  (1 / (1 + a) + 4 / (2 + b)) ≥ (13 + 4 * Real.sqrt 3) / 14 ∧
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 3*b₀ = 7 ∧
    1 / (1 + a₀) + 4 / (2 + b₀) = (13 + 4 * Real.sqrt 3) / 14 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1932_193265


namespace NUMINAMATH_CALUDE_water_bottle_pricing_l1932_193236

theorem water_bottle_pricing (x : ℝ) 
  (h1 : x > 0) -- Ensure x is positive (price can't be negative or zero)
  (h2 : x > 10) -- Ensure x-10 is positive (price of type B can't be negative)
  : 700 / x = 500 / (x - 10) := by
  sorry

end NUMINAMATH_CALUDE_water_bottle_pricing_l1932_193236


namespace NUMINAMATH_CALUDE_cone_lateral_area_l1932_193232

/-- The lateral area of a cone with base radius 1 and height √3 is 2π. -/
theorem cone_lateral_area : 
  let r : ℝ := 1
  let h : ℝ := Real.sqrt 3
  let l : ℝ := Real.sqrt (r^2 + h^2)
  let S : ℝ := π * r * l
  S = 2 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_lateral_area_l1932_193232


namespace NUMINAMATH_CALUDE_real_part_of_inverse_one_minus_z_squared_l1932_193253

theorem real_part_of_inverse_one_minus_z_squared (z : ℂ) 
  (h1 : z ≠ (z.re : ℂ)) -- z is nonreal
  (h2 : Complex.abs z = 1) :
  Complex.re (1 / (1 - z^2)) = (1 - z.re^2) / 2 := by sorry

end NUMINAMATH_CALUDE_real_part_of_inverse_one_minus_z_squared_l1932_193253


namespace NUMINAMATH_CALUDE_intersection_P_T_l1932_193229

def P : Set ℝ := {x | x^2 - x - 2 = 0}
def T : Set ℝ := {x | -1 < x ∧ x ≤ 2}

theorem intersection_P_T : P ∩ T = {2} := by sorry

end NUMINAMATH_CALUDE_intersection_P_T_l1932_193229


namespace NUMINAMATH_CALUDE_max_quotient_value_l1932_193254

theorem max_quotient_value (x y : ℝ) (hx : 100 ≤ x ∧ x ≤ 300) (hy : 900 ≤ y ∧ y ≤ 1800) :
  (∀ x' y', 100 ≤ x' ∧ x' ≤ 300 → 900 ≤ y' ∧ y' ≤ 1800 → y' / x' ≤ 18) ∧
  (∃ x' y', 100 ≤ x' ∧ x' ≤ 300 ∧ 900 ≤ y' ∧ y' ≤ 1800 ∧ y' / x' = 18) :=
by sorry

end NUMINAMATH_CALUDE_max_quotient_value_l1932_193254


namespace NUMINAMATH_CALUDE_closest_fraction_to_37_57_l1932_193201

theorem closest_fraction_to_37_57 :
  ∀ n : ℤ, n ≠ 15 → |37/57 - 15/23| < |37/57 - n/23| := by
sorry

end NUMINAMATH_CALUDE_closest_fraction_to_37_57_l1932_193201


namespace NUMINAMATH_CALUDE_lcm_1540_2310_l1932_193296

theorem lcm_1540_2310 : Nat.lcm 1540 2310 = 4620 := by
  sorry

end NUMINAMATH_CALUDE_lcm_1540_2310_l1932_193296


namespace NUMINAMATH_CALUDE_not_parallel_to_skew_line_l1932_193233

-- Define the concept of lines in 3D space
variable (Line : Type)

-- Define the relationships between lines
variable (skew : Line → Line → Prop)
variable (parallel : Line → Line → Prop)

-- Theorem statement
theorem not_parallel_to_skew_line 
  (a b c : Line) 
  (h1 : skew a b) 
  (h2 : parallel c a) : 
  ¬ parallel c b :=
sorry

end NUMINAMATH_CALUDE_not_parallel_to_skew_line_l1932_193233


namespace NUMINAMATH_CALUDE_custom_mult_theorem_l1932_193208

/-- Custom multiplication operation -/
def custom_mult (m n : ℝ) : ℝ := 2 * m - 3 * n

/-- Theorem stating that if x satisfies the given condition, then x = 7 -/
theorem custom_mult_theorem (x : ℝ) : 
  (∀ m n : ℝ, custom_mult m n = 2 * m - 3 * n) → 
  custom_mult x 7 = custom_mult 7 x → 
  x = 7 := by
  sorry

end NUMINAMATH_CALUDE_custom_mult_theorem_l1932_193208


namespace NUMINAMATH_CALUDE_distinct_roots_condition_l1932_193247

-- Define the quadratic equation
def quadratic_equation (k x : ℝ) : ℝ := k * x^2 - 2 * x - 1

-- Define the discriminant of the quadratic equation
def discriminant (k : ℝ) : ℝ := 4 + 4 * k

-- Theorem statement
theorem distinct_roots_condition (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ quadratic_equation k x = 0 ∧ quadratic_equation k y = 0) ↔
  (k > -1 ∧ k ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_distinct_roots_condition_l1932_193247


namespace NUMINAMATH_CALUDE_division_of_fractions_l1932_193210

theorem division_of_fractions : (5 : ℚ) / 6 / ((9 : ℚ) / 10) = 25 / 27 := by
  sorry

end NUMINAMATH_CALUDE_division_of_fractions_l1932_193210


namespace NUMINAMATH_CALUDE_disjoint_iff_valid_range_l1932_193250

/-- Set M represents a unit circle centered at the origin -/
def M : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}

/-- Set N represents a diamond centered at (1, 1) with side length a/√2 -/
def N (a : ℝ) : Set (ℝ × ℝ) := {p | |p.1 - 1| + |p.2 - 1| = a}

/-- The range of a for which M and N are disjoint -/
def valid_range : Set ℝ := {a | a < 2 - Real.sqrt 2 ∨ a > 2 + Real.sqrt 2}

theorem disjoint_iff_valid_range (a : ℝ) : 
  Disjoint (M : Set (ℝ × ℝ)) (N a) ↔ a ∈ valid_range := by sorry

end NUMINAMATH_CALUDE_disjoint_iff_valid_range_l1932_193250


namespace NUMINAMATH_CALUDE_water_evaporation_rate_l1932_193275

theorem water_evaporation_rate 
  (initial_water : ℝ) 
  (days : ℕ) 
  (evaporation_percentage : ℝ) 
  (h1 : initial_water = 10)
  (h2 : days = 50)
  (h3 : evaporation_percentage = 40) : 
  (initial_water * evaporation_percentage / 100) / days = 0.08 := by
  sorry

end NUMINAMATH_CALUDE_water_evaporation_rate_l1932_193275


namespace NUMINAMATH_CALUDE_apples_in_basket_proof_l1932_193282

/-- Given a total number of apples and the capacity of each box,
    calculate the number of apples left for the basket. -/
def applesInBasket (totalApples : ℕ) (applesPerBox : ℕ) : ℕ :=
  totalApples - (totalApples / applesPerBox) * applesPerBox

/-- Prove that with 138 total apples and boxes of 18 apples each,
    there will be 12 apples left for the basket. -/
theorem apples_in_basket_proof :
  applesInBasket 138 18 = 12 := by
  sorry

end NUMINAMATH_CALUDE_apples_in_basket_proof_l1932_193282


namespace NUMINAMATH_CALUDE_books_in_series_l1932_193270

/-- The number of books in the 'crazy silly school' series -/
def num_books : ℕ := 59

/-- The number of movies in the 'crazy silly school' series -/
def num_movies : ℕ := 61

/-- There are 2 more movies than books in the series -/
axiom movie_book_difference : num_movies = num_books + 2

theorem books_in_series : num_books = 59 := by sorry

end NUMINAMATH_CALUDE_books_in_series_l1932_193270


namespace NUMINAMATH_CALUDE_rational_sum_and_square_sum_integer_implies_integer_l1932_193241

theorem rational_sum_and_square_sum_integer_implies_integer (a b : ℚ) 
  (h1 : ∃ n : ℤ, (a + b : ℚ) = n)
  (h2 : ∃ m : ℤ, (a^2 + b^2 : ℚ) = m) :
  ∃ (x y : ℤ), (a = x ∧ b = y) :=
sorry

end NUMINAMATH_CALUDE_rational_sum_and_square_sum_integer_implies_integer_l1932_193241


namespace NUMINAMATH_CALUDE_sum_reciprocals_l1932_193223

theorem sum_reciprocals (a b c d : ℝ) (ω : ℂ) : 
  a ≠ -1 → b ≠ -1 → c ≠ -1 → d ≠ -1 →
  ω^4 = 1 → ω ≠ 1 →
  (1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω) = 3 / ω) →
  (1 / (a + 2) + 1 / (b + 2) + 1 / (c + 2) + 1 / (d + 2) = 1 / 4) :=
by sorry

end NUMINAMATH_CALUDE_sum_reciprocals_l1932_193223


namespace NUMINAMATH_CALUDE_f_minimum_value_l1932_193294

def f (x y : ℝ) : ℝ := (1 - y)^2 + (x + y - 3)^2 + (2*x + y - 6)^2

theorem f_minimum_value :
  ∀ x y : ℝ, f x y ≥ 1/6 ∧
  (f x y = 1/6 ↔ x = 17/4 ∧ y = 1/4) := by sorry

end NUMINAMATH_CALUDE_f_minimum_value_l1932_193294


namespace NUMINAMATH_CALUDE_toy_production_on_time_l1932_193206

/-- Proves that the toy production can be completed on time --/
theorem toy_production_on_time (total_toys : ℕ) (first_three_days_avg : ℕ) (remaining_days_avg : ℕ) 
  (available_days : ℕ) (h1 : total_toys = 3000) (h2 : first_three_days_avg = 250) 
  (h3 : remaining_days_avg = 375) (h4 : available_days = 11) : 
  (3 + ((total_toys - 3 * first_three_days_avg) / remaining_days_avg : ℕ)) ≤ available_days := by
  sorry

#check toy_production_on_time

end NUMINAMATH_CALUDE_toy_production_on_time_l1932_193206


namespace NUMINAMATH_CALUDE_triangle_max_sin_sum_l1932_193237

theorem triangle_max_sin_sum (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  c * Real.sin A = Real.sqrt 3 * a * Real.cos C →
  (∃ (max : ℝ), max = Real.sqrt 3 ∧ 
    ∀ A' B' : ℝ, 0 < A' ∧ 0 < B' ∧ A' + B' = 2*π/3 →
      Real.sin A' + Real.sin B' ≤ max) :=
sorry

end NUMINAMATH_CALUDE_triangle_max_sin_sum_l1932_193237


namespace NUMINAMATH_CALUDE_snail_climb_problem_l1932_193203

/-- The number of days required for a snail to climb out of a well -/
def days_to_climb (well_height : ℕ) (day_climb : ℕ) (night_slip : ℕ) : ℕ :=
  sorry

theorem snail_climb_problem :
  let well_height : ℕ := 12
  let day_climb : ℕ := 3
  let night_slip : ℕ := 2
  days_to_climb well_height day_climb night_slip = 10 := by sorry

end NUMINAMATH_CALUDE_snail_climb_problem_l1932_193203


namespace NUMINAMATH_CALUDE_contractor_absent_days_l1932_193288

/-- Represents the contractor's work scenario -/
structure ContractorScenario where
  totalDays : ℕ
  payPerWorkDay : ℚ
  finePerAbsentDay : ℚ
  totalPay : ℚ

/-- Calculates the number of absent days for a given contractor scenario -/
def absentDays (scenario : ContractorScenario) : ℚ :=
  (scenario.totalDays * scenario.payPerWorkDay - scenario.totalPay) / (scenario.payPerWorkDay + scenario.finePerAbsentDay)

/-- Theorem stating that for the given scenario, the number of absent days is 2 -/
theorem contractor_absent_days :
  let scenario : ContractorScenario := {
    totalDays := 30,
    payPerWorkDay := 25,
    finePerAbsentDay := 7.5,
    totalPay := 685
  }
  absentDays scenario = 2 := by sorry

end NUMINAMATH_CALUDE_contractor_absent_days_l1932_193288


namespace NUMINAMATH_CALUDE_rebus_puzzle_solution_l1932_193272

theorem rebus_puzzle_solution :
  ∃! (A B C : Nat),
    A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧
    A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
    A < 10 ∧ B < 10 ∧ C < 10 ∧
    (100 * A + 10 * B + A) + (100 * A + 10 * B + C) + (100 * A + 10 * C + C) = 1416 ∧
    A = 4 ∧ B = 7 ∧ C = 6 := by
  sorry

end NUMINAMATH_CALUDE_rebus_puzzle_solution_l1932_193272


namespace NUMINAMATH_CALUDE_negation_equivalence_l1932_193299

theorem negation_equivalence (m : ℤ) :
  (¬ ∃ x : ℤ, x^2 + 2*x + m < 0) ↔ (∀ x : ℤ, x^2 + 2*x + m ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1932_193299


namespace NUMINAMATH_CALUDE_maria_reading_capacity_l1932_193205

/-- The number of books Maria can read given her reading speed, book length, and available time -/
def books_read (reading_speed : ℕ) (pages_per_book : ℕ) (available_hours : ℕ) : ℕ :=
  (reading_speed * available_hours) / pages_per_book

/-- Theorem: Maria can read 3 books of 360 pages each in 9 hours at a speed of 120 pages per hour -/
theorem maria_reading_capacity :
  books_read 120 360 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_maria_reading_capacity_l1932_193205


namespace NUMINAMATH_CALUDE_opposite_of_negative_seven_l1932_193234

theorem opposite_of_negative_seven :
  ∃ x : ℤ, ((-7 : ℤ) + x = 0) ∧ x = 7 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_seven_l1932_193234


namespace NUMINAMATH_CALUDE_largest_box_size_l1932_193240

theorem largest_box_size (olivia noah liam : ℕ) 
  (h_olivia : olivia = 48)
  (h_noah : noah = 60)
  (h_liam : liam = 72) :
  Nat.gcd olivia (Nat.gcd noah liam) = 12 := by
  sorry

end NUMINAMATH_CALUDE_largest_box_size_l1932_193240


namespace NUMINAMATH_CALUDE_second_subject_grade_l1932_193248

theorem second_subject_grade (grade1 grade2 grade3 average : ℚ) : 
  grade1 = 50 →
  grade3 = 90 →
  average = 70 →
  (grade1 + grade2 + grade3) / 3 = average →
  grade2 = 70 := by
sorry

end NUMINAMATH_CALUDE_second_subject_grade_l1932_193248


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1932_193292

theorem rationalize_denominator :
  (Real.sqrt 12 + Real.sqrt 5) / (Real.sqrt 3 + Real.sqrt 5) = (Real.sqrt 15 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1932_193292


namespace NUMINAMATH_CALUDE_problem_solution_l1932_193217

theorem problem_solution (y : ℝ) (d e f : ℕ+) :
  y = Real.sqrt ((Real.sqrt 75) / 2 + 5 / 2) →
  y^100 = 3*y^98 + 18*y^96 + 15*y^94 - y^50 + (d : ℝ)*y^46 + (e : ℝ)*y^44 + (f : ℝ)*y^40 →
  (d : ℝ) + (e : ℝ) + (f : ℝ) = 556.5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1932_193217


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l1932_193293

theorem geometric_series_common_ratio : 
  let a₁ := 7 / 8
  let a₂ := -5 / 12
  let a₃ := 25 / 144
  let r := a₂ / a₁
  r = -10 / 21 := by sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l1932_193293


namespace NUMINAMATH_CALUDE_cake_and_bread_weight_l1932_193225

/-- Given the weight of 4 cakes and the weight difference between a cake and a piece of bread,
    calculate the total weight of 3 cakes and 5 pieces of bread. -/
theorem cake_and_bread_weight (cake_weight : ℕ) (bread_weight : ℕ) : 
  (4 * cake_weight = 800) →
  (cake_weight = bread_weight + 100) →
  (3 * cake_weight + 5 * bread_weight = 1100) :=
by sorry

end NUMINAMATH_CALUDE_cake_and_bread_weight_l1932_193225


namespace NUMINAMATH_CALUDE_product_digit_sum_l1932_193251

/-- The number of digits in each factor -/
def n : ℕ := 2012

/-- The first factor in the multiplication -/
def first_factor : ℕ := (10^n - 1) * 4 / 9

/-- The second factor in the multiplication -/
def second_factor : ℕ := 10^n - 1

/-- Function to calculate the sum of digits of a natural number -/
def sum_of_digits (k : ℕ) : ℕ :=
  if k < 10 then k else k % 10 + sum_of_digits (k / 10)

/-- The main theorem to be proved -/
theorem product_digit_sum :
  sum_of_digits (first_factor * second_factor) = 18108 := by
  sorry

end NUMINAMATH_CALUDE_product_digit_sum_l1932_193251


namespace NUMINAMATH_CALUDE_intersection_A_B_l1932_193269

def A : Set ℝ := {-1, 0, 1, 2, 3}
def B : Set ℝ := {x | x^2 - 3*x < 0}

theorem intersection_A_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1932_193269


namespace NUMINAMATH_CALUDE_marketValueTheorem_l1932_193231

/-- Calculates the market value of a machine after two years, given initial value,
    depreciation rates, and inflation rate. -/
def marketValueAfterTwoYears (initialValue : ℝ) (depreciation1 : ℝ) (depreciation2 : ℝ) (inflation : ℝ) : ℝ :=
  let value1 := initialValue * (1 - depreciation1) * (1 + inflation)
  let value2 := value1 * (1 - depreciation2) * (1 + inflation)
  value2

/-- Theorem stating that the market value of a machine with given parameters
    after two years is approximately 4939.20. -/
theorem marketValueTheorem :
  ∃ ε > 0, ε < 0.01 ∧ 
  |marketValueAfterTwoYears 8000 0.3 0.2 0.05 - 4939.20| < ε :=
sorry

end NUMINAMATH_CALUDE_marketValueTheorem_l1932_193231


namespace NUMINAMATH_CALUDE_right_handed_players_count_l1932_193252

theorem right_handed_players_count (total_players throwers : ℕ) 
  (h1 : total_players = 70)
  (h2 : throwers = 46)
  (h3 : throwers ≤ total_players)
  (h4 : (total_players - throwers) % 3 = 0) -- Ensure non-throwers are divisible by 3
  : (throwers + (2 * (total_players - throwers) / 3)) = 62 := by
  sorry

end NUMINAMATH_CALUDE_right_handed_players_count_l1932_193252


namespace NUMINAMATH_CALUDE_glass_bowl_problem_l1932_193285

/-- The initial rate per bowl given the conditions of the glass bowl sales problem -/
def initial_rate_per_bowl (total_bowls : ℕ) (sold_bowls : ℕ) (selling_price : ℚ) (percentage_gain : ℚ) : ℚ :=
  (sold_bowls * selling_price) / (total_bowls * (1 + percentage_gain / 100))

theorem glass_bowl_problem :
  let total_bowls : ℕ := 114
  let sold_bowls : ℕ := 108
  let selling_price : ℚ := 17
  let percentage_gain : ℚ := 23.88663967611336
  abs (initial_rate_per_bowl total_bowls sold_bowls selling_price percentage_gain - 13) < 0.01 := by
  sorry

#eval initial_rate_per_bowl 114 108 17 23.88663967611336

end NUMINAMATH_CALUDE_glass_bowl_problem_l1932_193285
