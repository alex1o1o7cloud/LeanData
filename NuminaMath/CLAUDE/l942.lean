import Mathlib

namespace circle_focus_at_center_l942_94201

/-- An ellipse with equal major and minor axes is a circle -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The focus of a circle is at its center -/
def Circle.focus (c : Circle) : ℝ × ℝ := c.center

theorem circle_focus_at_center (h_center : ℝ × ℝ) (h_radius : ℝ) :
  let c : Circle := { center := h_center, radius := h_radius }
  c.focus = c.center := by sorry

end circle_focus_at_center_l942_94201


namespace trader_weighted_avg_gain_percentage_l942_94258

/-- Calculates the weighted average gain percentage for a trader selling three types of pens -/
theorem trader_weighted_avg_gain_percentage
  (quantity_A quantity_B quantity_C : ℕ)
  (cost_A cost_B cost_C : ℚ)
  (gain_quantity_A gain_quantity_B gain_quantity_C : ℕ)
  (h_quantity_A : quantity_A = 60)
  (h_quantity_B : quantity_B = 40)
  (h_quantity_C : quantity_C = 50)
  (h_cost_A : cost_A = 2)
  (h_cost_B : cost_B = 3)
  (h_cost_C : cost_C = 4)
  (h_gain_quantity_A : gain_quantity_A = 20)
  (h_gain_quantity_B : gain_quantity_B = 15)
  (h_gain_quantity_C : gain_quantity_C = 10) :
  let total_cost := quantity_A * cost_A + quantity_B * cost_B + quantity_C * cost_C
  let total_gain := gain_quantity_A * cost_A + gain_quantity_B * cost_B + gain_quantity_C * cost_C
  let weighted_avg_gain_percentage := (total_gain / total_cost) * 100
  weighted_avg_gain_percentage = 28.41 := by
  sorry

end trader_weighted_avg_gain_percentage_l942_94258


namespace trapezoid_division_areas_l942_94289

/-- Given a trapezoid with base length a, parallel side length b, and height m,
    when divided into three equal parts, prove that the areas of the resulting
    trapezoids are as stated. -/
theorem trapezoid_division_areas (a b m : ℝ) (ha : a > 0) (hb : b > 0) (hm : m > 0) :
  let s := m / 3
  let x := (2 * a + b) / 3
  let y := (a + 2 * b) / 3
  let t₁ := ((a + x) / 2) * s
  let t₂ := ((x + y) / 2) * s
  let t₃ := ((y + b) / 2) * s
  (t₁ = (5 * a + b) * m / 18) ∧
  (t₂ = (a + b) * m / 6) ∧
  (t₃ = (a + 5 * b) * m / 18) :=
by sorry

end trapezoid_division_areas_l942_94289


namespace train_length_l942_94229

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmph : ℝ) (time_sec : ℝ) : 
  speed_kmph = 18 → time_sec = 5 → (speed_kmph * 1000 / 3600) * time_sec = 25 := by
  sorry

end train_length_l942_94229


namespace conference_attendance_theorem_l942_94276

/-- The percentage of attendees who paid their conference fee in full but did not register at least two weeks in advance -/
def late_payment_percentage : ℝ := 10

/-- The percentage of conference attendees who registered at least two weeks in advance -/
def early_registration_percentage : ℝ := 86.67

/-- The percentage of attendees who registered at least two weeks in advance and paid their conference fee in full -/
def early_registration_full_payment_percentage : ℝ := 96.3

theorem conference_attendance_theorem :
  (100 - late_payment_percentage) / 100 * early_registration_full_payment_percentage = early_registration_percentage :=
by sorry

end conference_attendance_theorem_l942_94276


namespace hyperbola_symmetric_intersection_l942_94287

/-- The hyperbola and its symmetric curve with respect to a line have common points for all real k -/
theorem hyperbola_symmetric_intersection (k : ℝ) : ∃ (x y : ℝ), 
  (x^2 - y^2 = 1) ∧ 
  (∃ (x' y' : ℝ), (x'^2 - y'^2 = 1) ∧ 
    ((x + x') / 2 = (y + y') / (2*k) - 1/k) ∧
    ((y + y') / 2 = k * ((x + x') / 2) - 1)) :=
by sorry

end hyperbola_symmetric_intersection_l942_94287


namespace donut_distribution_count_l942_94263

/-- The number of ways to distribute items into bins -/
def distribute_items (total_items : ℕ) (num_bins : ℕ) (items_to_distribute : ℕ) : ℕ :=
  Nat.choose (items_to_distribute + num_bins - 1) (num_bins - 1)

/-- Theorem stating the number of ways to distribute donuts -/
theorem donut_distribution_count :
  let total_donuts : ℕ := 10
  let donut_types : ℕ := 5
  let donuts_to_distribute : ℕ := total_donuts - donut_types
  distribute_items total_donuts donut_types donuts_to_distribute = 126 := by
  sorry

#eval distribute_items 10 5 5

end donut_distribution_count_l942_94263


namespace point_on_line_proof_l942_94295

def point_on_line (x₁ y₁ x₂ y₂ x y : ℝ) : Prop :=
  (y - y₁) * (x₂ - x₁) = (y₂ - y₁) * (x - x₁)

theorem point_on_line_proof : point_on_line 2 1 10 5 14 7 := by
  sorry

end point_on_line_proof_l942_94295


namespace sum_of_sequences_l942_94206

def sequence1 : List ℕ := [1, 11, 21, 31, 41, 51, 61, 71, 81, 91]
def sequence2 : List ℕ := [9, 19, 29, 39, 49, 59, 69, 79, 89, 99]

theorem sum_of_sequences : (sequence1.sum + sequence2.sum = 1000) := by
  sorry

end sum_of_sequences_l942_94206


namespace jacks_remaining_money_l942_94213

def remaining_money (initial_amount snack_cost ride_multiplier game_multiplier : ℝ) : ℝ :=
  initial_amount - (snack_cost + ride_multiplier * snack_cost + game_multiplier * snack_cost)

theorem jacks_remaining_money :
  remaining_money 100 15 3 1.5 = 17.5 := by
  sorry

end jacks_remaining_money_l942_94213


namespace simplify_square_roots_l942_94274

theorem simplify_square_roots :
  Real.sqrt 8 - Real.sqrt 32 + Real.sqrt 72 - Real.sqrt 50 = -Real.sqrt 2 := by
  sorry

end simplify_square_roots_l942_94274


namespace may_greatest_drop_l942_94249

/-- Represents the months of the year --/
inductive Month
| january
| february
| march
| april
| may
| june

/-- Price change for a given month --/
def price_change : Month → ℝ
| Month.january  => -1.00
| Month.february => 3.50
| Month.march    => -3.00
| Month.april    => 4.00
| Month.may      => -5.00
| Month.june     => 2.00

/-- Returns true if the price change is negative (a drop) --/
def is_price_drop (m : Month) : Prop :=
  price_change m < 0

/-- The month with the greatest price drop --/
def greatest_drop : Month :=
  Month.may

theorem may_greatest_drop :
  ∀ m : Month, is_price_drop m → price_change greatest_drop ≤ price_change m :=
by sorry

end may_greatest_drop_l942_94249


namespace complement_intersection_theorem_l942_94292

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define set M
def M : Set Nat := {1, 2, 3}

-- Define set N
def N : Set Nat := {2, 3, 5}

-- Theorem statement
theorem complement_intersection_theorem :
  (U \ M) ∩ N = {5} := by
  sorry

end complement_intersection_theorem_l942_94292


namespace sandal_pairs_bought_l942_94203

def shirt_price : ℕ := 5
def sandal_price : ℕ := 3
def num_shirts : ℕ := 10
def total_paid : ℕ := 100
def change_received : ℕ := 41

theorem sandal_pairs_bought : ℕ := by
  sorry

end sandal_pairs_bought_l942_94203


namespace constraint_extrema_l942_94209

def constraint (x y : ℝ) : Prop :=
  Real.sqrt (x - 3) + Real.sqrt (y - 4) = 4

def objective (x y : ℝ) : ℝ :=
  2 * x + 3 * y

theorem constraint_extrema :
  ∃ (x_min y_min x_max y_max : ℝ),
    constraint x_min y_min ∧
    constraint x_max y_max ∧
    (∀ x y, constraint x y → objective x y ≥ objective x_min y_min) ∧
    (∀ x y, constraint x y → objective x y ≤ objective x_max y_max) ∧
    x_min = 219 / 25 ∧
    y_min = 264 / 25 ∧
    x_max = 3 ∧
    y_max = 20 ∧
    objective x_min y_min = 37.2 ∧
    objective x_max y_max = 66 :=
  sorry

#check constraint_extrema

end constraint_extrema_l942_94209


namespace no_triples_satisfying_lcm_conditions_l942_94238

theorem no_triples_satisfying_lcm_conditions :
  ¬∃ (x y z : ℕ+), 
    (Nat.lcm x.val y.val = 48) ∧ 
    (Nat.lcm x.val z.val = 900) ∧ 
    (Nat.lcm y.val z.val = 180) :=
by sorry

end no_triples_satisfying_lcm_conditions_l942_94238


namespace jackson_holidays_l942_94296

/-- The number of holidays taken in a year given the number of days off per month and the number of months in a year -/
def holidays_in_year (days_off_per_month : ℕ) (months_in_year : ℕ) : ℕ :=
  days_off_per_month * months_in_year

/-- Theorem stating that taking 3 days off every month for 12 months results in 36 holidays in a year -/
theorem jackson_holidays :
  holidays_in_year 3 12 = 36 := by
  sorry

end jackson_holidays_l942_94296


namespace calculation_proof_l942_94261

theorem calculation_proof : (π - 2019)^0 + |Real.sqrt 3 - 1| + (-1/2)⁻¹ - 2 * Real.tan (30 * π / 180) = -2 + Real.sqrt 3 / 3 := by
  sorry

end calculation_proof_l942_94261


namespace fraction_simplification_l942_94278

theorem fraction_simplification : 
  (1/4 - 1/5) / (1/3 - 1/4) = 3/5 := by sorry

end fraction_simplification_l942_94278


namespace cylinder_volume_l942_94298

/-- The volume of a cylinder with equal base diameter and height, and lateral area π. -/
theorem cylinder_volume (r h : ℝ) (h1 : h = 2 * r) (h2 : 2 * π * r * h = π) : π * r^2 * h = π / 4 := by
  sorry

end cylinder_volume_l942_94298


namespace all_points_collinear_l942_94271

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a line -/
def Point.onLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if three points are collinear -/
def collinear (p q r : Point) : Prop :=
  ∃ l : Line, p.onLine l ∧ q.onLine l ∧ r.onLine l

/-- The main theorem -/
theorem all_points_collinear (S : Set Point) (h_finite : Set.Finite S)
    (h_three_point : ∀ p q r : Point, p ∈ S → q ∈ S → r ∈ S → p ≠ q → 
      (∃ l : Line, p.onLine l ∧ q.onLine l) → r.onLine l) :
    ∀ p q r : Point, p ∈ S → q ∈ S → r ∈ S → collinear p q r :=
  sorry

end all_points_collinear_l942_94271


namespace problem_solution_l942_94233

def is_arithmetic_sequence (s : Fin 5 → ℝ) : Prop :=
  ∃ d : ℝ, ∀ i : Fin 4, s (i + 1) - s i = d

def is_geometric_sequence (s : Fin 5 → ℝ) : Prop :=
  ∃ r : ℝ, ∀ i : Fin 4, s (i + 1) / s i = r

theorem problem_solution (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) :
  is_arithmetic_sequence (λ i => match i with
    | 0 => 1
    | 1 => a₁
    | 2 => a₂
    | 3 => a₃
    | 4 => 9) →
  is_geometric_sequence (λ i => match i with
    | 0 => -9
    | 1 => b₁
    | 2 => b₂
    | 3 => b₃
    | 4 => -1) →
  b₂ / (a₁ + a₃) = -3/10 := by
  sorry

end problem_solution_l942_94233


namespace pet_shop_dogs_l942_94247

/-- Given a pet shop with dogs, cats, and bunnies, where the ratio of dogs to cats to bunnies
    is 3:7:12 and the total number of dogs and bunnies is 375, prove that there are 75 dogs. -/
theorem pet_shop_dogs (dogs cats bunnies : ℕ) : 
  dogs + cats + bunnies > 0 →
  dogs * 7 = cats * 3 →
  dogs * 12 = bunnies * 3 →
  dogs + bunnies = 375 →
  dogs = 75 := by
sorry

end pet_shop_dogs_l942_94247


namespace rectangle_area_l942_94218

theorem rectangle_area (x : ℝ) (w : ℝ) (h : w > 0) : 
  (3 * w)^2 + w^2 = x^2 → 3 * w^2 = 3 * x^2 / 10 :=
by sorry

end rectangle_area_l942_94218


namespace sum_of_coefficients_factorization_l942_94204

theorem sum_of_coefficients_factorization (x y : ℝ) : 
  ∃ (a b c d e f g h j k : ℤ),
    27 * x^6 - 512 * y^6 = (a*x + b*y) * (c*x^2 + d*x*y + e*y^2) * (f*x + g*y) * (h*x^2 + j*x*y + k*y^2) ∧
    a + b + c + d + e + f + g + h + j + k = 55 :=
by sorry

end sum_of_coefficients_factorization_l942_94204


namespace two_color_theorem_l942_94255

/-- A line in a plane --/
structure Line where
  -- We don't need to define the specifics of a line for this problem

/-- A region in a plane --/
structure Region where
  -- We don't need to define the specifics of a region for this problem

/-- A color (we only need two colors) --/
inductive Color
  | A
  | B

/-- A function that determines if two regions are adjacent --/
def adjacent (r1 r2 : Region) : Prop :=
  sorry -- The specific implementation is not important for the statement

/-- A coloring of regions --/
def Coloring := Region → Color

/-- A valid coloring ensures adjacent regions have different colors --/
def valid_coloring (c : Coloring) : Prop :=
  ∀ r1 r2, adjacent r1 r2 → c r1 ≠ c r2

/-- The main theorem --/
theorem two_color_theorem (lines : List Line) :
  ∃ (regions : List Region) (c : Coloring), valid_coloring c :=
sorry

end two_color_theorem_l942_94255


namespace det2_specific_values_det2_quadratic_relation_l942_94266

def det2 (a b c d : ℝ) : ℝ := a * d - b * c

theorem det2_specific_values :
  det2 5 6 7 8 = -2 :=
sorry

theorem det2_quadratic_relation (x : ℝ) (h : x^2 - 3*x + 1 = 0) :
  det2 (x + 1) (3*x) (x - 2) (x - 1) = 6*x + 1 :=
sorry

end det2_specific_values_det2_quadratic_relation_l942_94266


namespace letitia_order_l942_94215

theorem letitia_order (julie_order anton_order individual_tip tip_percentage : ℚ) 
  (h1 : julie_order = 10)
  (h2 : anton_order = 30)
  (h3 : individual_tip = 4)
  (h4 : tip_percentage = 1/5)
  : ∃ letitia_order : ℚ, 
    tip_percentage * (julie_order + letitia_order + anton_order) = 3 * individual_tip ∧ 
    letitia_order = 20 := by
  sorry

end letitia_order_l942_94215


namespace centroid_sum_l942_94219

def vertex1 : Fin 3 → ℚ := ![9, 2, -1]
def vertex2 : Fin 3 → ℚ := ![5, -2, 3]
def vertex3 : Fin 3 → ℚ := ![1, 6, 5]

def centroid (v1 v2 v3 : Fin 3 → ℚ) : Fin 3 → ℚ :=
  fun i => (v1 i + v2 i + v3 i) / 3

theorem centroid_sum :
  (centroid vertex1 vertex2 vertex3 0 +
   centroid vertex1 vertex2 vertex3 1 +
   centroid vertex1 vertex2 vertex3 2) = 28 / 3 := by
  sorry

end centroid_sum_l942_94219


namespace speed_of_light_scientific_notation_l942_94290

def speed_of_light : ℝ := 300000000

theorem speed_of_light_scientific_notation : 
  speed_of_light = 3 * (10 : ℝ) ^ 8 := by sorry

end speed_of_light_scientific_notation_l942_94290


namespace price_restoration_l942_94288

theorem price_restoration (original_price : ℝ) (h : original_price > 0) :
  let reduced_price := 0.8 * original_price
  (reduced_price * 1.25 = original_price) := by
sorry

end price_restoration_l942_94288


namespace bicycle_inventory_problem_l942_94268

/-- Represents the bicycle inventory problem for Hank's store over three days --/
theorem bicycle_inventory_problem 
  (B : ℤ) -- Initial number of bicycles
  (S : ℤ) -- Number of bicycles sold on Friday
  (h1 : S ≥ 0) -- Number of bicycles sold is non-negative
  (h2 : B - S + 15 - 12 + 8 - 9 + 11 = B + 3) -- Net increase equation
  : S = 10 := by
  sorry


end bicycle_inventory_problem_l942_94268


namespace negation_of_existence_implication_l942_94239

theorem negation_of_existence_implication :
  ¬(∃ n : ℤ, ∀ m : ℤ, n^2 = m^2 → n = m) ↔
  (∀ n : ℤ, ∃ m : ℤ, n^2 = m^2 ∧ n ≠ m) :=
by sorry

end negation_of_existence_implication_l942_94239


namespace max_value_of_function_l942_94250

theorem max_value_of_function (x : ℝ) (h : x^2 - x + 1 ≠ 0) :
  (2*x^2 - 2*x + 3) / (x^2 - x + 1) ≤ 10/3 ∧
  ∃ y : ℝ, (2*y^2 - 2*y + 3) / (y^2 - y + 1) = 10/3 :=
by sorry

end max_value_of_function_l942_94250


namespace inequality_system_solution_l942_94260

theorem inequality_system_solution (a b : ℝ) : 
  (∀ x : ℝ, (x - a ≥ b ∧ 2*x - a - 1 < 2*b) ↔ (3 ≤ x ∧ x < 5)) →
  a = -3 ∧ b = 6 := by
sorry

end inequality_system_solution_l942_94260


namespace equation_solution_l942_94275

theorem equation_solution : 
  ∃! x : ℝ, Real.sqrt x + 3 * Real.sqrt (x^2 + 8*x) + Real.sqrt (x + 8) = 42 - 3*x :=
by
  -- The proof goes here
  sorry

end equation_solution_l942_94275


namespace tile_border_ratio_l942_94226

/-- Proves that for a square tiled surface with n^2 tiles, each tile of side length s,
    surrounded by a border of width d, if n = 30 and the tiles cover 81% of the total area,
    then d/s = 1/18. -/
theorem tile_border_ratio (n s d : ℝ) (h1 : n = 30) 
    (h2 : (n^2 * s^2) / ((n*s + 2*n*d)^2) = 0.81) : d/s = 1/18 := by
  sorry

end tile_border_ratio_l942_94226


namespace empty_set_implies_a_range_l942_94283

theorem empty_set_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - |x + 1| + 2 * a ≥ 0) → 
  a > (Real.sqrt 3 + 1) / 4 := by
sorry

end empty_set_implies_a_range_l942_94283


namespace tank_initial_water_l942_94240

def tank_capacity : ℚ := 100
def day1_collection : ℚ := 15
def day2_collection : ℚ := 20
def day3_overflow : ℚ := 25

theorem tank_initial_water (initial_water : ℚ) :
  initial_water + day1_collection + day2_collection = tank_capacity ∧
  (initial_water / tank_capacity = 13 / 20) := by
  sorry

end tank_initial_water_l942_94240


namespace triangle_projection_similarity_l942_94286

/-- For any triangle, there exist perpendicular distances that make the projected triangle similar to the original -/
theorem triangle_projection_similarity (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ (x y : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ 
    x^2 + b^2 = y^2 + a^2 ∧
    (x - y)^2 + c^2 = y^2 + a^2 :=
by sorry

end triangle_projection_similarity_l942_94286


namespace music_exam_songs_l942_94257

/-- Represents a girl participating in the music exam -/
inductive Girl
| Anna
| Bea
| Cili
| Dora

/-- The number of times each girl sang -/
def timesSang (g : Girl) : ℕ :=
  match g with
  | Girl.Anna => 8
  | Girl.Bea => 7  -- We assume 7 as it satisfies the conditions
  | Girl.Cili => 7 -- We assume 7 as it satisfies the conditions
  | Girl.Dora => 5

/-- The total number of individual singing assignments -/
def totalSingingAssignments : ℕ := 
  (timesSang Girl.Anna) + (timesSang Girl.Bea) + (timesSang Girl.Cili) + (timesSang Girl.Dora)

theorem music_exam_songs :
  (∀ g : Girl, timesSang g ≤ timesSang Girl.Anna) ∧ 
  (∀ g : Girl, g ≠ Girl.Anna → timesSang g < timesSang Girl.Anna) ∧
  (∀ g : Girl, g ≠ Girl.Dora → timesSang Girl.Dora < timesSang g) ∧
  (totalSingingAssignments % 3 = 0) →
  totalSingingAssignments / 3 = 9 := by
  sorry

#eval totalSingingAssignments / 3

end music_exam_songs_l942_94257


namespace hexagon_angle_problem_l942_94230

/-- Given a hexagon with specific angle conditions, prove that the unknown angle is 25 degrees. -/
theorem hexagon_angle_problem (a b c d e x : ℝ) : 
  -- Sum of interior angles of a hexagon
  a + b + c + d + e + x = (6 - 2) * 180 →
  -- Sum of five known angles
  a + b + c + d + e = 100 →
  -- Two adjacent angles are 75° each
  75 + x + 75 = 360 →
  -- Conclusion: x is 25°
  x = 25 := by
  sorry

end hexagon_angle_problem_l942_94230


namespace pitcher_problem_l942_94281

theorem pitcher_problem (C : ℝ) (h : C > 0) : 
  let juice_volume : ℝ := (3/4) * C
  let num_cups : ℕ := 5
  let juice_per_cup : ℝ := juice_volume / num_cups
  (juice_per_cup / C) * 100 = 15 := by
sorry

end pitcher_problem_l942_94281


namespace diamond_two_three_l942_94222

-- Define the diamond operation
def diamond (a b : ℤ) : ℤ := a * b^2 - b + a^2 + 1

-- Theorem statement
theorem diamond_two_three : diamond 2 3 = 20 := by sorry

end diamond_two_three_l942_94222


namespace lillian_candy_count_l942_94251

def initial_candies : ℕ := 88
def additional_candies : ℕ := 5

theorem lillian_candy_count :
  initial_candies + additional_candies = 93 := by sorry

end lillian_candy_count_l942_94251


namespace plough_time_for_A_l942_94202

/-- Given two workers A and B who can plough a field together in 10 hours,
    and B alone takes 30 hours, prove that A alone would take 15 hours. -/
theorem plough_time_for_A (time_together time_B : ℝ) (time_together_pos : time_together > 0)
    (time_B_pos : time_B > 0) (h1 : time_together = 10) (h2 : time_B = 30) :
    ∃ time_A : ℝ, time_A > 0 ∧ 1 / time_A + 1 / time_B = 1 / time_together ∧ time_A = 15 := by
  sorry

end plough_time_for_A_l942_94202


namespace smallest_number_l942_94207

theorem smallest_number (a b c d e : ℝ) 
  (ha : a = 0.997) 
  (hb : b = 0.979) 
  (hc : c = 0.999) 
  (hd : d = 0.9797) 
  (he : e = 0.9709) : 
  e ≤ a ∧ e ≤ b ∧ e ≤ c ∧ e ≤ d := by
  sorry

end smallest_number_l942_94207


namespace class_size_problem_l942_94224

/-- Given classes A, B, and C with the following properties:
  * Class A is twice as big as Class B
  * Class A is a third the size of Class C
  * Class B has 20 people
  Prove that Class C has 120 people -/
theorem class_size_problem (class_A class_B class_C : ℕ) : 
  class_A = 2 * class_B →
  class_A = class_C / 3 →
  class_B = 20 →
  class_C = 120 := by
  sorry

end class_size_problem_l942_94224


namespace coin_stack_arrangements_l942_94294

/-- The number of indistinguishable gold coins -/
def num_gold_coins : ℕ := 5

/-- The number of indistinguishable silver coins -/
def num_silver_coins : ℕ := 3

/-- The total number of coins -/
def total_coins : ℕ := num_gold_coins + num_silver_coins

/-- The number of ways to arrange gold and silver coins -/
def gold_silver_arrangements : ℕ := Nat.choose total_coins num_gold_coins

/-- The number of valid head-tail sequences -/
def valid_head_tail_sequences : ℕ := total_coins + 1

/-- The total number of distinguishable arrangements -/
def total_arrangements : ℕ := gold_silver_arrangements * valid_head_tail_sequences

theorem coin_stack_arrangements :
  total_arrangements = 504 :=
sorry

end coin_stack_arrangements_l942_94294


namespace election_result_l942_94208

/-- Represents the total number of valid votes cast in the election -/
def total_votes : ℕ := sorry

/-- Represents the number of votes received by the winning candidate -/
def winning_votes : ℕ := 7320

/-- Represents the percentage of votes received by the winning candidate after redistribution -/
def winning_percentage : ℚ := 43 / 100

theorem election_result :
  total_votes * winning_percentage = winning_votes ∧
  total_votes ≥ 17023 ∧
  total_votes < 17024 :=
sorry

end election_result_l942_94208


namespace six_digit_multiple_of_nine_l942_94242

theorem six_digit_multiple_of_nine :
  ∃! d : Nat, d < 10 ∧ (456780 + d) % 9 = 0 :=
by
  sorry

end six_digit_multiple_of_nine_l942_94242


namespace inscribed_circumscribed_quadrilateral_relation_l942_94254

/-- A quadrilateral inscribed in one circle and circumscribed about another -/
structure InscribedCircumscribedQuadrilateral where
  R : ℝ  -- radius of the circumscribed circle
  r : ℝ  -- radius of the inscribed circle
  d : ℝ  -- distance between the centers of the circles
  R_pos : 0 < R
  r_pos : 0 < r
  d_pos : 0 < d
  d_lt_R : d < R

/-- The relationship between R, r, and d for an inscribed-circumscribed quadrilateral -/
theorem inscribed_circumscribed_quadrilateral_relation 
  (q : InscribedCircumscribedQuadrilateral) : 
  1 / (q.R + q.d)^2 + 1 / (q.R - q.d)^2 = 1 / q.r^2 := by
  sorry

end inscribed_circumscribed_quadrilateral_relation_l942_94254


namespace complex_equation_solution_l942_94241

theorem complex_equation_solution (c d : ℂ) (x : ℝ) :
  Complex.abs c = 3 →
  Complex.abs d = 5 →
  c * d = x - 3 * Complex.I →
  x = 6 * Real.sqrt 6 :=
by sorry

end complex_equation_solution_l942_94241


namespace quirkyville_reading_paradox_l942_94284

/-- Represents the student population at Quirkyville College -/
structure StudentPopulation where
  total : ℕ
  enjoy_reading : ℕ
  claim_enjoy : ℕ
  claim_not_enjoy : ℕ

/-- The fraction of students who say they don't enjoy reading but actually do -/
def fraction_false_negative (pop : StudentPopulation) : ℚ :=
  (pop.enjoy_reading - pop.claim_enjoy) / pop.claim_not_enjoy

/-- Theorem stating the fraction of students who say they don't enjoy reading but actually do -/
theorem quirkyville_reading_paradox (pop : StudentPopulation) : 
  pop.total > 0 ∧ 
  pop.enjoy_reading = (70 * pop.total) / 100 ∧
  pop.claim_enjoy = (75 * pop.enjoy_reading) / 100 ∧
  pop.claim_not_enjoy = pop.total - pop.claim_enjoy →
  fraction_false_negative pop = 35 / 83 := by
  sorry

#eval (35 : ℚ) / 83

end quirkyville_reading_paradox_l942_94284


namespace max_triangle_sum_l942_94236

/-- Represents the arrangement of numbers on the vertices of the triangles -/
def TriangleArrangement := Fin 6 → Fin 6

/-- The sum of three numbers on a side of a triangle -/
def sideSum (arr : TriangleArrangement) (i j k : Fin 6) : ℕ :=
  (arr i).val + 12 + (arr j).val + 12 + (arr k).val + 12

/-- Predicate to check if an arrangement is valid -/
def isValidArrangement (arr : TriangleArrangement) : Prop :=
  (∀ i j, i ≠ j → arr i ≠ arr j) ∧
  (∀ i, arr i < 6)

/-- Predicate to check if all sides have the same sum -/
def allSidesEqual (arr : TriangleArrangement) (S : ℕ) : Prop :=
  sideSum arr 0 1 2 = S ∧
  sideSum arr 2 3 4 = S ∧
  sideSum arr 4 5 0 = S

theorem max_triangle_sum :
  ∃ (S : ℕ) (arr : TriangleArrangement),
    isValidArrangement arr ∧
    allSidesEqual arr S ∧
    (∀ (S' : ℕ) (arr' : TriangleArrangement),
      isValidArrangement arr' → allSidesEqual arr' S' → S' ≤ S) ∧
    S = 45 := by
  sorry

end max_triangle_sum_l942_94236


namespace absolute_quadratic_inequality_l942_94228

/-- The set of real numbers x satisfying |x^2 - 4x + 3| ≤ 3 is equal to the closed interval [0, 4]. -/
theorem absolute_quadratic_inequality (x : ℝ) :
  |x^2 - 4*x + 3| ≤ 3 ↔ x ∈ Set.Icc 0 4 := by
  sorry

end absolute_quadratic_inequality_l942_94228


namespace factor_expression_1_l942_94262

theorem factor_expression_1 (m n : ℝ) :
  4/9 * m^2 + 4/3 * m * n + n^2 = (2/3 * m + n)^2 := by
  sorry

end factor_expression_1_l942_94262


namespace factorization_proof_l942_94273

theorem factorization_proof (x y : ℝ) : x^2 + y^2 + 2*x*y - 1 = (x + y + 1) * (x + y - 1) := by
  sorry

end factorization_proof_l942_94273


namespace amount_difference_l942_94244

theorem amount_difference (p q r : ℝ) : 
  p = 47.99999999999999 →
  q = p / 6 →
  r = p / 6 →
  p - (q + r) = 32 :=
by
  sorry

end amount_difference_l942_94244


namespace area_of_smaller_circle_l942_94256

/-- Two circles are externally tangent with common tangent lines -/
structure TangentCircles where
  center_small : ℝ × ℝ
  center_large : ℝ × ℝ
  radius_small : ℝ
  radius_large : ℝ
  tangent_point : ℝ × ℝ
  externally_tangent : (center_small.1 - center_large.1)^2 + (center_small.2 - center_large.2)^2 = (radius_small + radius_large)^2
  radius_ratio : radius_large = 3 * radius_small

/-- Common tangent line -/
structure CommonTangent where
  P : ℝ × ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  PA_length : ℝ
  AB_length : ℝ
  PA_eq_AB : PA_length = AB_length
  PA_eq_8 : PA_length = 8

/-- The main theorem -/
theorem area_of_smaller_circle (tc : TangentCircles) (ct : CommonTangent) : 
  π * tc.radius_small^2 = 16 * π :=
sorry

end area_of_smaller_circle_l942_94256


namespace magazine_subscription_cost_l942_94217

theorem magazine_subscription_cost (reduced_cost : ℝ) (reduction_percentage : ℝ) (original_cost : ℝ) : 
  reduced_cost = 752 ∧ 
  reduction_percentage = 0.20 ∧ 
  reduced_cost = original_cost * (1 - reduction_percentage) →
  original_cost = 940 := by
sorry

end magazine_subscription_cost_l942_94217


namespace polynomial_equality_l942_94279

/-- Given that 4x^5 + 3x^3 + 2x^2 + p(x) = 6x^3 - 5x^2 + 4x - 2 for all x,
    prove that p(x) = -4x^5 + 3x^3 - 7x^2 + 4x - 2 -/
theorem polynomial_equality (p : ℝ → ℝ) :
  (∀ x, 4 * x^5 + 3 * x^3 + 2 * x^2 + p x = 6 * x^3 - 5 * x^2 + 4 * x - 2) →
  (∀ x, p x = -4 * x^5 + 3 * x^3 - 7 * x^2 + 4 * x - 2) :=
by
  sorry

end polynomial_equality_l942_94279


namespace polynomial_product_expansion_l942_94252

-- Define the polynomials
def p (x : ℝ) : ℝ := 7 * x^2 + 5 * x + 3
def q (x : ℝ) : ℝ := 3 * x^3 + 2 * x^2 + 1

-- State the theorem
theorem polynomial_product_expansion :
  ∀ x : ℝ, p x * q x = 21 * x^5 + 29 * x^4 + 19 * x^3 + 13 * x^2 + 5 * x + 3 :=
by sorry

end polynomial_product_expansion_l942_94252


namespace circle_equation_l942_94259

theorem circle_equation (x y : ℝ) : 
  (∃ (r : ℝ), r > 0 ∧ (x - 2)^2 + y^2 = r^2) ∧ 
  ((-2)^2 + 0^2 = 4) := by
  sorry

end circle_equation_l942_94259


namespace bucket_fill_time_l942_94280

/-- Given that two-thirds of a bucket is filled in 90 seconds,
    prove that the time taken to fill the bucket completely is 135 seconds. -/
theorem bucket_fill_time (fill_time : ℝ) (h : fill_time = 90) :
  (3 / 2) * fill_time = 135 := by
  sorry

end bucket_fill_time_l942_94280


namespace problem_solution_l942_94216

theorem problem_solution : 
  (2/3 - 3/4 + 1/6) / (-1/24) = -2 ∧ 
  -2^3 + 3 * (-1)^2023 - |3-7| = -15 :=
by sorry

end problem_solution_l942_94216


namespace ellipse_tangent_line_l942_94285

/-- The equation of the tangent line to an ellipse -/
theorem ellipse_tangent_line 
  (a b x₀ y₀ : ℝ) 
  (h_pos_a : a > 0) 
  (h_pos_b : b > 0) 
  (h_on_ellipse : x₀^2 / a^2 + y₀^2 / b^2 = 1) :
  ∀ x y : ℝ, (x₀ * x / a^2 + y₀ * y / b^2 = 1) ↔ 
    (∃ t : ℝ, x = x₀ + t * (-b^2 * x₀) ∧ y = y₀ + t * (a^2 * y₀) ∧ 
    ∀ u : ℝ, (x₀ + u * (-b^2 * x₀))^2 / a^2 + (y₀ + u * (a^2 * y₀))^2 / b^2 ≥ 1) :=
by sorry

end ellipse_tangent_line_l942_94285


namespace train_crossing_length_train_B_length_l942_94299

/-- The length of a train crossing another train in opposite direction --/
theorem train_crossing_length (length_A : ℝ) (speed_A speed_B : ℝ) (time : ℝ) : ℝ :=
  let relative_speed := (speed_A + speed_B) * (1000 / 3600)
  let total_distance := relative_speed * time
  total_distance - length_A

/-- Proof of the length of Train B --/
theorem train_B_length : 
  train_crossing_length 360 120 150 6 = 90 := by
  sorry

end train_crossing_length_train_B_length_l942_94299


namespace ryans_initial_funds_l942_94277

/-- Proves that Ryan's initial funds equal the total cost minus crowdfunding amount -/
theorem ryans_initial_funds 
  (average_funding : ℕ) 
  (people_to_recruit : ℕ) 
  (total_cost : ℕ) 
  (h1 : average_funding = 10)
  (h2 : people_to_recruit = 80)
  (h3 : total_cost = 1000) :
  total_cost - (average_funding * people_to_recruit) = 200 := by
  sorry

#check ryans_initial_funds

end ryans_initial_funds_l942_94277


namespace seaweed_livestock_amount_l942_94227

-- Define the total amount of seaweed harvested
def total_seaweed : ℝ := 400

-- Define the percentage of seaweed used for fires
def fire_percentage : ℝ := 0.5

-- Define the percentage of remaining seaweed for human consumption
def human_percentage : ℝ := 0.25

-- Function to calculate the amount of seaweed fed to livestock
def seaweed_for_livestock : ℝ :=
  let remaining_after_fire := total_seaweed * (1 - fire_percentage)
  let for_humans := remaining_after_fire * human_percentage
  remaining_after_fire - for_humans

-- Theorem stating the amount of seaweed fed to livestock
theorem seaweed_livestock_amount : seaweed_for_livestock = 150 := by
  sorry

end seaweed_livestock_amount_l942_94227


namespace cos_D_is_zero_l942_94272

-- Define the triangle DEF
structure Triangle (DE EF : ℝ) where
  -- Ensure DE and EF are positive
  de_pos : DE > 0
  ef_pos : EF > 0

-- Define the right triangle DEF with given side lengths
def rightTriangleDEF : Triangle 9 40 where
  de_pos := by norm_num
  ef_pos := by norm_num

-- Theorem: In the right triangle DEF where angle D is 90°, cos D = 0
theorem cos_D_is_zero (t : Triangle 9 40) : Real.cos (π / 2) = 0 := by
  sorry

end cos_D_is_zero_l942_94272


namespace intersection_of_lines_l942_94248

theorem intersection_of_lines :
  ∃! (x y : ℚ), (3 * y = -2 * x + 6) ∧ (-2 * y = 4 * x - 3) ∧ (x = 3/8) ∧ (y = 7/4) := by
  sorry

end intersection_of_lines_l942_94248


namespace library_visits_total_l942_94225

/-- The number of times William goes to the library per week -/
def william_freq : ℕ := 2

/-- The number of times Jason goes to the library per week -/
def jason_freq : ℕ := 4 * william_freq

/-- The number of times Emma goes to the library per week -/
def emma_freq : ℕ := 3 * jason_freq

/-- The number of times Zoe goes to the library per week -/
def zoe_freq : ℕ := william_freq / 2

/-- The number of times Chloe goes to the library per week -/
def chloe_freq : ℕ := emma_freq / 3

/-- The number of weeks -/
def weeks : ℕ := 8

/-- The total number of times Jason, Emma, Zoe, and Chloe go to the library over 8 weeks -/
def total_visits : ℕ := (jason_freq + emma_freq + zoe_freq + chloe_freq) * weeks

theorem library_visits_total : total_visits = 328 := by
  sorry

end library_visits_total_l942_94225


namespace basketball_tournament_handshakes_l942_94212

/-- Calculates the total number of handshakes in a basketball tournament --/
def total_handshakes (num_teams : ℕ) (players_per_team : ℕ) (num_referees : ℕ) : ℕ :=
  let total_players := num_teams * players_per_team
  let player_handshakes := (total_players * (total_players - players_per_team)) / 2
  let referee_handshakes := total_players * num_referees
  player_handshakes + referee_handshakes

/-- Theorem stating the total number of handshakes in the specific basketball tournament scenario --/
theorem basketball_tournament_handshakes :
  total_handshakes 3 5 2 = 105 := by
  sorry

end basketball_tournament_handshakes_l942_94212


namespace infinitely_many_primary_triplets_l942_94210

/-- A primary triplet is a triplet of positive integers (x, y, z) satisfying
    x, y, z > 1 and x^3 - yz^3 = 2021, where at least two of x, y, z are prime numbers. -/
def PrimaryTriplet (x y z : ℕ) : Prop :=
  x > 1 ∧ y > 1 ∧ z > 1 ∧
  x^3 - y*z^3 = 2021 ∧
  (Nat.Prime x ∧ Nat.Prime y) ∨ (Nat.Prime x ∧ Nat.Prime z) ∨ (Nat.Prime y ∧ Nat.Prime z)

/-- There exist infinitely many primary triplets. -/
theorem infinitely_many_primary_triplets :
  ∀ n : ℕ, ∃ m : ℕ, m > n ∧ ∃ x y z : ℕ, PrimaryTriplet x y z :=
sorry

end infinitely_many_primary_triplets_l942_94210


namespace a_is_editor_l942_94220

-- Define the professions
inductive Profession
| Doctor
| Teacher
| Editor

-- Define the volunteers
structure Volunteer where
  name : String
  profession : Profession
  age : Nat

-- Define the fair
structure Fair where
  volunteers : List Volunteer

-- Define the proposition
theorem a_is_editor (f : Fair) : 
  (∃ a b c : Volunteer, 
    a ∈ f.volunteers ∧ b ∈ f.volunteers ∧ c ∈ f.volunteers ∧
    a.name = "A" ∧ b.name = "B" ∧ c.name = "C" ∧
    a.profession ≠ b.profession ∧ b.profession ≠ c.profession ∧ c.profession ≠ a.profession ∧
    (∃ d : Volunteer, d ∈ f.volunteers ∧ d.profession = Profession.Doctor ∧ d.age ≠ a.age) ∧
    (∃ e : Volunteer, e ∈ f.volunteers ∧ e.profession = Profession.Editor ∧ e.age > c.age) ∧
    (∃ d : Volunteer, d ∈ f.volunteers ∧ d.profession = Profession.Doctor ∧ d.age > b.age)) →
  (∃ a : Volunteer, a ∈ f.volunteers ∧ a.name = "A" ∧ a.profession = Profession.Editor) :=
by sorry

end a_is_editor_l942_94220


namespace problem_solution_l942_94293

theorem problem_solution (x y : ℝ) (hx : x = Real.sqrt 3 + 1) (hy : y = Real.sqrt 3 - 1) :
  (x^2 + 2*x*y + y^2 = 12) ∧ (x^2 - y^2 = 4 * Real.sqrt 3) := by
  sorry

end problem_solution_l942_94293


namespace fixed_point_sets_l942_94211

-- Define the function f
def f (a b x : ℝ) : ℝ := x^2 + a*x + b

-- Define set A
def A (a b : ℝ) : Set ℝ := {x | f a b x = x}

-- Define set B
def B (a b : ℝ) : Set ℝ := {x | f a b (f a b x) = x}

-- Theorem statement
theorem fixed_point_sets (a b : ℝ) :
  A a b = {-1, 3} →
  B a b = {-Real.sqrt 3, -1, Real.sqrt 3, 3} ∧ A a b ⊆ B a b := by
  sorry

end fixed_point_sets_l942_94211


namespace range_of_a_l942_94231

/-- Given functions f and g, prove the range of a -/
theorem range_of_a (a : ℝ) : 
  (∀ x₁ : ℝ, ∃ x₂ : ℝ, |2 * x₁ - a| + |2 * x₁ + 3| = |x₂ - 1| + 2) →
  (a ≥ -1 ∨ a ≤ -5) := by
  sorry

end range_of_a_l942_94231


namespace largest_multiple_of_9_under_100_l942_94264

theorem largest_multiple_of_9_under_100 : ∃ n : ℕ, n = 99 ∧ 9 ∣ n ∧ n < 100 ∧ ∀ m : ℕ, 9 ∣ m → m < 100 → m ≤ n :=
by sorry

end largest_multiple_of_9_under_100_l942_94264


namespace roots_of_equation_l942_94253

theorem roots_of_equation : ∃ x₁ x₂ : ℝ,
  (88 * (x₁ - 2)^2 = 95) ∧
  (88 * (x₂ - 2)^2 = 95) ∧
  (x₁ < 1) ∧
  (x₂ > 3) :=
by sorry

end roots_of_equation_l942_94253


namespace digit_B_value_l942_94267

theorem digit_B_value (A B : ℕ) (h : 100 * A + 10 * B + 2 - 41 = 591) : B = 3 := by
  sorry

end digit_B_value_l942_94267


namespace elderly_workers_in_sample_l942_94205

/-- Represents the composition of workers in a company --/
structure WorkforceComposition where
  total : ℕ
  young : ℕ
  middleAged : ℕ
  elderly : ℕ

/-- Represents a stratified sample from the workforce --/
structure StratifiedSample where
  youngInSample : ℕ
  elderlyInSample : ℕ

/-- Theorem stating the number of elderly workers in the stratified sample --/
theorem elderly_workers_in_sample 
  (wc : WorkforceComposition) 
  (sample : StratifiedSample) : 
  wc.total = 430 →
  wc.young = 160 →
  wc.middleAged = 2 * wc.elderly →
  sample.youngInSample = 32 →
  sample.elderlyInSample = 18 := by
  sorry

end elderly_workers_in_sample_l942_94205


namespace find_other_number_l942_94235

-- Define the given conditions
def n : ℕ := 48
def lcm_nm : ℕ := 56
def gcf_nm : ℕ := 12

-- Define the theorem
theorem find_other_number (m : ℕ) : 
  (Nat.lcm n m = lcm_nm) → 
  (Nat.gcd n m = gcf_nm) → 
  m = 14 := by
  sorry

end find_other_number_l942_94235


namespace install_remaining_windows_time_l942_94237

/-- Calculates the time needed to install remaining windows -/
def timeToInstallRemaining (totalWindows installedWindows timePerWindow : ℕ) : ℕ :=
  (totalWindows - installedWindows) * timePerWindow

/-- Proves that the time to install the remaining windows is 18 hours -/
theorem install_remaining_windows_time :
  timeToInstallRemaining 9 6 6 = 18 := by
  sorry

end install_remaining_windows_time_l942_94237


namespace satellite_altitude_scientific_notation_l942_94270

/-- The altitude of a Beidou satellite in meters -/
def satellite_altitude : ℝ := 21500000

/-- Scientific notation representation of the satellite altitude -/
def scientific_notation : ℝ := 2.15 * (10 ^ 7)

/-- Theorem stating that the satellite altitude is equal to its scientific notation representation -/
theorem satellite_altitude_scientific_notation : 
  satellite_altitude = scientific_notation := by sorry

end satellite_altitude_scientific_notation_l942_94270


namespace power_four_inequality_l942_94269

theorem power_four_inequality (a b : ℝ) : (a^4 + b^4) / 2 ≥ ((a + b) / 2)^4 := by
  sorry

end power_four_inequality_l942_94269


namespace intersection_minimum_distance_l942_94234

/-- Given a line y = b intersecting f(x) = 2x + 3 and g(x) = ax + ln x at points A and B respectively,
    if the minimum value of |AB| is 2, then a + b = 2 -/
theorem intersection_minimum_distance (a b : ℝ) : 
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    2 * x₁ + 3 = b ∧ 
    a * x₂ + Real.log x₂ = b ∧
    (∀ (y₁ y₂ : ℝ), 2 * y₁ + 3 = b → a * y₂ + Real.log y₂ = b → |y₂ - y₁| ≥ 2) ∧
    |x₂ - x₁| = 2) →
  a + b = 2 := by
sorry

end intersection_minimum_distance_l942_94234


namespace correct_mean_calculation_l942_94221

theorem correct_mean_calculation (n : ℕ) (initial_mean : ℝ) 
  (error1 error2 error3 error4 error5 : ℝ) : 
  n = 70 →
  initial_mean = 350 →
  error1 = 215.5 - 195.5 →
  error2 = -30 - 30 →
  error3 = 720.8 - 670.8 →
  error4 = -95.4 - (-45.4) →
  error5 = 124.2 - 114.2 →
  (n : ℝ) * initial_mean + (error1 + error2 + error3 + error4 + error5) = n * 349.57 := by
  sorry

end correct_mean_calculation_l942_94221


namespace simplify_and_evaluate_l942_94200

theorem simplify_and_evaluate : 
  let a : ℚ := -2
  let b : ℚ := 1/5
  2 * a * b^2 - (6 * a^3 * b + 2 * (a * b^2 - 1/2 * a^3 * b)) = 8 := by
  sorry

end simplify_and_evaluate_l942_94200


namespace specific_sampling_problem_l942_94282

/-- Systematic sampling function -/
def systematicSample (totalPopulation sampleSize firstDrawn nthGroup : ℕ) : ℕ :=
  let interval := totalPopulation / sampleSize
  firstDrawn + interval * (nthGroup - 1)

/-- Theorem for the specific sampling problem -/
theorem specific_sampling_problem :
  systematicSample 1000 50 15 21 = 415 := by
  sorry

end specific_sampling_problem_l942_94282


namespace ellipse_and_distance_l942_94291

/-- An ellipse with the given conditions -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- The configuration of points and lines for the problem -/
structure Configuration (C : Ellipse) where
  right_focus : ℝ × ℝ
  passing_point : ℝ × ℝ
  M : ℝ
  l : ℝ → ℝ → Prop
  A : ℝ × ℝ
  B : ℝ × ℝ
  N : ℝ × ℝ
  h₁ : right_focus = (Real.sqrt 3, 0)
  h₂ : passing_point = (-1, Real.sqrt 3 / 2)
  h₃ : C.a^2 * (passing_point.1^2 / C.a^2 + passing_point.2^2 / C.b^2) = C.a^2
  h₄ : l M A.2 ∧ l M B.2
  h₅ : A.2 > 0 ∧ B.2 < 0
  h₆ : (A.1 - M)^2 + A.2^2 = 4 * ((B.1 - M)^2 + B.2^2)
  h₇ : N.1^2 + N.2^2 = 4/7
  h₈ : ∀ x y, l x y → (x - N.1)^2 + (y - N.2)^2 ≥ 4/7

/-- The main theorem to be proved -/
theorem ellipse_and_distance (C : Ellipse) (cfg : Configuration C) :
  C.a^2 = 4 ∧ C.b^2 = 1 ∧ 
  (cfg.M - cfg.N.1)^2 + cfg.N.2^2 = (4 * Real.sqrt 21 / 21)^2 := by
  sorry

end ellipse_and_distance_l942_94291


namespace uncle_zhang_revenue_l942_94243

/-- Uncle Zhang's newspaper selling problem -/
theorem uncle_zhang_revenue
  (a b : ℕ)  -- a and b are natural numbers representing the number of newspapers
  (purchase_price sell_price return_price : ℚ)  -- prices are rational numbers
  (h1 : purchase_price = 0.4)  -- purchase price is 0.4 yuan
  (h2 : sell_price = 0.5)  -- selling price is 0.5 yuan
  (h3 : return_price = 0.2)  -- return price is 0.2 yuan
  (h4 : b ≤ a)  -- number of sold newspapers cannot exceed purchased newspapers
  : (sell_price * b + return_price * (a - b) - purchase_price * a : ℚ) = 0.3 * b - 0.2 * a :=
by sorry

end uncle_zhang_revenue_l942_94243


namespace sum_of_extremes_l942_94297

def is_valid_number (n : ℕ) : Prop :=
  n > 100 ∧ n < 1000 ∧ ∀ d : ℕ, d ∈ n.digits 10 → d ∈ [1, 2, 5]

def smallest_valid_number : ℕ := sorry

def largest_valid_number : ℕ := sorry

theorem sum_of_extremes :
  smallest_valid_number + largest_valid_number = 646 ∧
  is_valid_number smallest_valid_number ∧
  is_valid_number largest_valid_number ∧
  ∀ n : ℕ, is_valid_number n →
    smallest_valid_number ≤ n ∧ n ≤ largest_valid_number :=
sorry

end sum_of_extremes_l942_94297


namespace line_equation_length_BC_l942_94245

-- Problem 1
def projection_point : ℝ × ℝ := (2, -1)

theorem line_equation (l : Set (ℝ × ℝ)) (h : projection_point ∈ l) :
  l = {(x, y) | 2*x - y - 5 = 0} := by sorry

-- Problem 2
def point_A : ℝ × ℝ := (4, -1)
def midpoint_AB : ℝ × ℝ := (3, 2)
def centroid : ℝ × ℝ := (4, 2)

theorem length_BC :
  let B : ℝ × ℝ := (2*midpoint_AB.1 - point_A.1, 2*midpoint_AB.2 - point_A.2)
  let C : ℝ × ℝ := (3*centroid.1 - point_A.1 - B.1, 3*centroid.2 - point_A.2 - B.2)
  Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 5 := by sorry

end line_equation_length_BC_l942_94245


namespace inequality_condition_max_value_l942_94246

-- Define the functions
def f (x : ℝ) : ℝ := x^2 - 1
def g (a x : ℝ) : ℝ := a * |x - 1|
def h (a x : ℝ) : ℝ := |f x| + g a x

-- Statement 1
theorem inequality_condition (a : ℝ) : 
  (∀ x : ℝ, f x ≥ g a x) ↔ a ≤ -2 :=
sorry

-- Statement 2
theorem max_value (a : ℝ) :
  (∃ x : ℝ, x ∈ Set.Icc 0 2 ∧ 
    ∀ y ∈ Set.Icc 0 2, h a x ≥ h a y) ∧
  (∃ m : ℝ, (∀ x ∈ Set.Icc 0 2, h a x ≤ m) ∧
    m = if a ≥ -3 then a + 3 else 0) :=
sorry

end inequality_condition_max_value_l942_94246


namespace harkamal_payment_l942_94223

/-- The total amount Harkamal paid to the shopkeeper for grapes and mangoes. -/
def total_amount_paid (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mango_quantity * mango_rate

/-- Theorem stating that Harkamal paid 1125 to the shopkeeper. -/
theorem harkamal_payment : total_amount_paid 9 70 9 55 = 1125 := by
  sorry

#eval total_amount_paid 9 70 9 55

end harkamal_payment_l942_94223


namespace min_value_of_function_l942_94214

theorem min_value_of_function (x : ℝ) (h : x ≥ 0) :
  (4 * x^2 + 12 * x + 25) / (6 * (1 + x)) ≥ 8/3 := by
  sorry

end min_value_of_function_l942_94214


namespace deepak_present_age_l942_94232

/-- Given the ratio of Rahul's age to Deepak's age and Rahul's future age, 
    prove Deepak's present age -/
theorem deepak_present_age 
  (ratio_rahul : ℕ) 
  (ratio_deepak : ℕ) 
  (rahul_future_age : ℕ) 
  (years_difference : ℕ) : 
  ratio_rahul = 4 → 
  ratio_deepak = 3 → 
  rahul_future_age = 22 → 
  years_difference = 6 → 
  (ratio_deepak * (rahul_future_age - years_difference)) / ratio_rahul = 12 := by
  sorry

end deepak_present_age_l942_94232


namespace orange_to_apple_ratio_l942_94265

/-- Given the total weight of fruits and the weight of oranges, proves the ratio of oranges to apples -/
theorem orange_to_apple_ratio
  (total_weight : ℕ)
  (orange_weight : ℕ)
  (h1 : total_weight = 12)
  (h2 : orange_weight = 10) :
  orange_weight / (total_weight - orange_weight) = 5 := by
  sorry

end orange_to_apple_ratio_l942_94265
