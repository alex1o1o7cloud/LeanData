import Mathlib

namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2332_233242

theorem imaginary_part_of_z (z : ℂ) : z = (1 - I) / (1 + 3*I) → z.im = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2332_233242


namespace NUMINAMATH_CALUDE_expected_malfunctioning_computers_correct_l2332_233236

/-- The expected number of malfunctioning computers -/
def expected_malfunctioning_computers (a b : ℝ) : ℝ := a + b

/-- Theorem: The expected number of malfunctioning computers is a + b -/
theorem expected_malfunctioning_computers_correct (a b : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) : 
  expected_malfunctioning_computers a b = a + b := by
  sorry

end NUMINAMATH_CALUDE_expected_malfunctioning_computers_correct_l2332_233236


namespace NUMINAMATH_CALUDE_at_operation_example_l2332_233231

def at_operation (x y : ℤ) : ℤ := x * y - 2 * x + 3 * y

theorem at_operation_example : (at_operation 8 5) - (at_operation 5 8) = -15 := by
  sorry

end NUMINAMATH_CALUDE_at_operation_example_l2332_233231


namespace NUMINAMATH_CALUDE_no_positive_integer_solution_l2332_233240

theorem no_positive_integer_solution :
  ¬ ∃ (a b c d : ℕ+), (a^2 + b^2 = c^2 - d^2) ∧ (a * b = c * d) := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_solution_l2332_233240


namespace NUMINAMATH_CALUDE_reflection_line_sum_l2332_233212

/-- Given a line y = mx + b, if the reflection of point (2, 3) across this line is (10, -1), then m + b = -9 -/
theorem reflection_line_sum (m b : ℝ) : 
  (∃ (x y : ℝ), 
    -- The point (x, y) is on the line y = mx + b
    y = m * x + b ∧ 
    -- The point (x, y) is equidistant from (2, 3) and (10, -1)
    (x - 2)^2 + (y - 3)^2 = (x - 10)^2 + (y + 1)^2 ∧
    -- The line through (2, 3) and (10, -1) is perpendicular to y = mx + b
    m * ((10 - 2) / ((-1) - 3)) = -1) →
  m + b = -9 := by sorry


end NUMINAMATH_CALUDE_reflection_line_sum_l2332_233212


namespace NUMINAMATH_CALUDE_sequence_sum_l2332_233200

theorem sequence_sum (a : ℕ → ℕ) (h : ∀ k : ℕ, k > 0 → a k + a (k + 1) = 2 * k + 1) :
  a 1 + a 100 = 101 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l2332_233200


namespace NUMINAMATH_CALUDE_cole_fence_cost_is_225_l2332_233254

/-- Calculates the total cost for Cole's fence installation given the backyard dimensions,
    fencing costs, and neighbor contributions. -/
def cole_fence_cost (side_length : ℝ) (back_length : ℝ) (side_cost : ℝ) (back_cost : ℝ)
                    (back_neighbor_contribution : ℝ) (left_neighbor_contribution : ℝ)
                    (installation_fee : ℝ) : ℝ :=
  let total_fencing_cost := 2 * side_length * side_cost + back_length * back_cost
  let neighbor_contributions := back_neighbor_contribution + left_neighbor_contribution
  total_fencing_cost - neighbor_contributions + installation_fee

theorem cole_fence_cost_is_225 :
  cole_fence_cost 15 30 4 5 75 20 50 = 225 := by
  sorry

end NUMINAMATH_CALUDE_cole_fence_cost_is_225_l2332_233254


namespace NUMINAMATH_CALUDE_comparison_of_expressions_l2332_233250

theorem comparison_of_expressions (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^2 / b + b^2 / a ≥ a + b := by
  sorry

end NUMINAMATH_CALUDE_comparison_of_expressions_l2332_233250


namespace NUMINAMATH_CALUDE_pet_store_cats_l2332_233262

theorem pet_store_cats (initial_siamese : ℕ) (cats_sold : ℕ) (cats_left : ℕ) :
  initial_siamese = 12 →
  cats_sold = 20 →
  cats_left = 12 →
  ∃ initial_house : ℕ, initial_house = 20 ∧ initial_siamese + initial_house = cats_sold + cats_left :=
by sorry

end NUMINAMATH_CALUDE_pet_store_cats_l2332_233262


namespace NUMINAMATH_CALUDE_remainder_928927_div_6_l2332_233234

theorem remainder_928927_div_6 : 928927 % 6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_928927_div_6_l2332_233234


namespace NUMINAMATH_CALUDE_class_savings_l2332_233222

/-- Calculates the total amount saved by a class for a field trip over a given period. -/
theorem class_savings (num_students : ℕ) (contribution : ℕ) (num_weeks : ℕ) :
  num_students = 30 →
  contribution = 2 →
  num_weeks = 8 →
  num_students * contribution * num_weeks = 480 := by
  sorry

#check class_savings

end NUMINAMATH_CALUDE_class_savings_l2332_233222


namespace NUMINAMATH_CALUDE_remaining_volume_cube_with_cylindrical_hole_l2332_233203

/-- The remaining volume of a cube after drilling a cylindrical hole -/
theorem remaining_volume_cube_with_cylindrical_hole :
  let cube_side : ℝ := 6
  let hole_radius : ℝ := 3
  let hole_height : ℝ := 6
  let cube_volume : ℝ := cube_side ^ 3
  let cylinder_volume : ℝ := π * hole_radius ^ 2 * hole_height
  let remaining_volume : ℝ := cube_volume - cylinder_volume
  remaining_volume = 216 - 54 * π := by
  sorry


end NUMINAMATH_CALUDE_remaining_volume_cube_with_cylindrical_hole_l2332_233203


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2332_233264

theorem complex_fraction_simplification :
  1 + 2 / (1 + 2 / (2 * 2)) = 7 / 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2332_233264


namespace NUMINAMATH_CALUDE_cos_2x_value_l2332_233249

theorem cos_2x_value (x : Real) (h : Real.sin (π / 2 + x) = 3 / 5) : 
  Real.cos (2 * x) = -7 / 25 := by
  sorry

end NUMINAMATH_CALUDE_cos_2x_value_l2332_233249


namespace NUMINAMATH_CALUDE_circle_locus_l2332_233253

/-- The locus of the center of a circle passing through (-2, 0) and tangent to x = 2 -/
theorem circle_locus (x₀ y₀ : ℝ) : 
  (∃ (r : ℝ), r > 0 ∧ 
    (x₀ + 2)^2 + y₀^2 = r^2 ∧ 
    |x₀ - 2| = r) →
  y₀^2 = -8 * x₀ :=
by sorry

end NUMINAMATH_CALUDE_circle_locus_l2332_233253


namespace NUMINAMATH_CALUDE_horse_track_distance_l2332_233293

/-- The distance covered by a horse running one turn around a square-shaped track -/
def track_distance (side_length : ℝ) : ℝ := 4 * side_length

/-- Theorem: The distance covered by a horse running one turn around a square-shaped track
    with sides of length 40 meters is equal to 160 meters -/
theorem horse_track_distance :
  track_distance 40 = 160 := by
  sorry

end NUMINAMATH_CALUDE_horse_track_distance_l2332_233293


namespace NUMINAMATH_CALUDE_power_product_equality_l2332_233232

theorem power_product_equality : (-0.125)^2021 * 8^2022 = -8 := by sorry

end NUMINAMATH_CALUDE_power_product_equality_l2332_233232


namespace NUMINAMATH_CALUDE_raghu_investment_l2332_233245

/-- Represents the investment amounts of Raghu, Trishul, and Vishal -/
structure Investments where
  raghu : ℝ
  trishul : ℝ
  vishal : ℝ

/-- The conditions of the investment problem -/
def investment_conditions (inv : Investments) : Prop :=
  inv.trishul = 0.9 * inv.raghu ∧
  inv.vishal = 1.1 * inv.trishul ∧
  inv.raghu + inv.trishul + inv.vishal = 6069

/-- Theorem stating that under the given conditions, Raghu's investment is 2100 -/
theorem raghu_investment (inv : Investments) 
  (h : investment_conditions inv) : inv.raghu = 2100 := by
  sorry

end NUMINAMATH_CALUDE_raghu_investment_l2332_233245


namespace NUMINAMATH_CALUDE_min_max_quadratic_form_l2332_233230

theorem min_max_quadratic_form (x y : ℝ) (h : 9*x^2 + 12*x*y + 4*y^2 = 1) :
  let f := fun (x y : ℝ) => 3*x^2 + 4*x*y + 2*y^2
  ∃ (m M : ℝ), (∀ a b : ℝ, m ≤ f a b ∧ f a b ≤ M) ∧ m = 1 ∧ M = 1 := by
  sorry

end NUMINAMATH_CALUDE_min_max_quadratic_form_l2332_233230


namespace NUMINAMATH_CALUDE_john_emu_pens_l2332_233221

/-- The number of pens for emus that John has -/
def num_pens : ℕ := sorry

/-- The number of emus per pen -/
def emus_per_pen : ℕ := 6

/-- The ratio of female emus to total emus -/
def female_ratio : ℚ := 1/2

/-- The number of eggs laid by each female emu per day -/
def eggs_per_female_per_day : ℕ := 1

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The total number of eggs collected in a week -/
def total_eggs_per_week : ℕ := 84

theorem john_emu_pens : 
  num_pens = 4 := by sorry

end NUMINAMATH_CALUDE_john_emu_pens_l2332_233221


namespace NUMINAMATH_CALUDE_music_tool_cost_l2332_233270

/-- Calculates the cost of a music tool given the total spent and costs of other items --/
theorem music_tool_cost (total_spent flute_cost songbook_cost : ℚ) :
  total_spent = 158.35 ∧ flute_cost = 142.46 ∧ songbook_cost = 7 →
  total_spent - (flute_cost + songbook_cost) = 8.89 := by
sorry

end NUMINAMATH_CALUDE_music_tool_cost_l2332_233270


namespace NUMINAMATH_CALUDE_weight_of_b_l2332_233273

theorem weight_of_b (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 43) 
  (h2 : (a + b) / 2 = 40) 
  (h3 : (b + c) / 2 = 43) : 
  b = 37 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_b_l2332_233273


namespace NUMINAMATH_CALUDE_room_length_proof_l2332_233260

/-- Given a rectangular room with known width, total paving cost, and paving rate per square meter,
    prove that the length of the room is 5.5 meters. -/
theorem room_length_proof (width : ℝ) (total_cost : ℝ) (paving_rate : ℝ) :
  width = 3.75 ∧ total_cost = 20625 ∧ paving_rate = 1000 →
  total_cost / paving_rate / width = 5.5 :=
by sorry

end NUMINAMATH_CALUDE_room_length_proof_l2332_233260


namespace NUMINAMATH_CALUDE_quadratic_maximum_l2332_233278

theorem quadratic_maximum (r : ℝ) : 
  -7 * r^2 + 50 * r - 20 ≤ 5 ∧ ∃ r, -7 * r^2 + 50 * r - 20 = 5 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_maximum_l2332_233278


namespace NUMINAMATH_CALUDE_orange_harvest_after_six_days_l2332_233238

/-- The number of sacks of oranges harvested after a given number of days. -/
def oranges_harvested (daily_rate : ℕ) (days : ℕ) : ℕ :=
  daily_rate * days

/-- Theorem stating that given a daily harvest rate of 83 sacks per day,
    the total number of sacks harvested after 6 days is equal to 498. -/
theorem orange_harvest_after_six_days :
  oranges_harvested 83 6 = 498 := by
  sorry

end NUMINAMATH_CALUDE_orange_harvest_after_six_days_l2332_233238


namespace NUMINAMATH_CALUDE_boats_left_l2332_233256

def total_boats : ℕ := 30
def fish_eaten_percentage : ℚ := 1/5
def boats_shot : ℕ := 2

theorem boats_left : 
  total_boats - (total_boats * fish_eaten_percentage).floor - boats_shot = 22 := by
sorry

end NUMINAMATH_CALUDE_boats_left_l2332_233256


namespace NUMINAMATH_CALUDE_expected_imbalance_six_teams_l2332_233265

/-- Represents a baseball league with n teams -/
structure BaseballLeague (n : ℕ) where
  teams : Fin n → Unit

/-- Represents the schedule of games in the league -/
def Schedule (n : ℕ) := Fin n → Fin n → Bool

/-- Calculates the imbalance (minimum number of undefeated teams) for a given schedule -/
def imbalance (n : ℕ) (schedule : Schedule n) : ℕ := sorry

/-- The expected value of the imbalance for a league with n teams -/
def expectedImbalance (n : ℕ) : ℚ := sorry

/-- Theorem: The expected imbalance for a 6-team league is 5055 / 2^15 -/
theorem expected_imbalance_six_teams :
  expectedImbalance 6 = 5055 / 2^15 := by sorry

end NUMINAMATH_CALUDE_expected_imbalance_six_teams_l2332_233265


namespace NUMINAMATH_CALUDE_right_triangle_with_60_degree_angle_l2332_233266

theorem right_triangle_with_60_degree_angle (α β : ℝ) : 
  α = 60 → -- One acute angle is 60°
  α + β + 90 = 180 → -- Sum of angles in a triangle is 180°
  β = 30 := by -- The other acute angle is 30°
sorry

end NUMINAMATH_CALUDE_right_triangle_with_60_degree_angle_l2332_233266


namespace NUMINAMATH_CALUDE_male_workers_percentage_l2332_233218

theorem male_workers_percentage (female_workers : ℝ) (male_workers : ℝ) :
  male_workers = 0.6 * female_workers →
  (female_workers - male_workers) / female_workers = 0.4 :=
by
  sorry

end NUMINAMATH_CALUDE_male_workers_percentage_l2332_233218


namespace NUMINAMATH_CALUDE_red_hair_count_example_l2332_233228

/-- Given a class with a hair color ratio and total number of students,
    calculate the number of students with red hair. -/
def red_hair_count (red blonde black total : ℕ) : ℕ :=
  (red * total) / (red + blonde + black)

/-- Theorem: In a class of 48 students with a hair color ratio of 3 : 6 : 7
    (red : blonde : black), the number of students with red hair is 9. -/
theorem red_hair_count_example : red_hair_count 3 6 7 48 = 9 := by
  sorry

end NUMINAMATH_CALUDE_red_hair_count_example_l2332_233228


namespace NUMINAMATH_CALUDE_equation_one_integral_root_l2332_233220

theorem equation_one_integral_root :
  ∃! x : ℤ, x - 9 / (x - 2) = 5 - 9 / (x - 2) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_one_integral_root_l2332_233220


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2332_233227

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 2) :
  1/x + 2/y ≥ 9/2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2332_233227


namespace NUMINAMATH_CALUDE_prob_select_one_from_2_7_l2332_233208

/-- The decimal representation of 2/7 -/
def decimal_rep_2_7 : List Nat := [2, 8, 5, 7, 1, 4]

/-- The probability of selecting a specific digit from the decimal representation of 2/7 -/
def prob_select_digit (d : Nat) : Rat :=
  (decimal_rep_2_7.count d) / (decimal_rep_2_7.length)

theorem prob_select_one_from_2_7 :
  prob_select_digit 1 = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_prob_select_one_from_2_7_l2332_233208


namespace NUMINAMATH_CALUDE_min_value_exponential_sum_l2332_233298

theorem min_value_exponential_sum (x y : ℝ) (h : x + y = 5) :
  3^x + 3^y ≥ 18 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_exponential_sum_l2332_233298


namespace NUMINAMATH_CALUDE_dave_book_cost_l2332_233246

/-- Calculates the total cost of Dave's books including discounts and taxes -/
def total_cost (animal_books_count : ℕ) (animal_book_price : ℚ)
                (space_books_count : ℕ) (space_book_price : ℚ)
                (train_books_count : ℕ) (train_book_price : ℚ)
                (history_books_count : ℕ) (history_book_price : ℚ)
                (science_books_count : ℕ) (science_book_price : ℚ)
                (animal_discount_rate : ℚ) (science_tax_rate : ℚ) : ℚ :=
  let animal_cost := animal_books_count * animal_book_price * (1 - animal_discount_rate)
  let space_cost := space_books_count * space_book_price
  let train_cost := train_books_count * train_book_price
  let history_cost := history_books_count * history_book_price
  let science_cost := science_books_count * science_book_price * (1 + science_tax_rate)
  animal_cost + space_cost + train_cost + history_cost + science_cost

/-- Theorem stating that the total cost of Dave's books is $379.5 -/
theorem dave_book_cost :
  total_cost 8 10 6 12 9 8 4 15 5 18 (1/10) (15/100) = 379.5 := by
  sorry

end NUMINAMATH_CALUDE_dave_book_cost_l2332_233246


namespace NUMINAMATH_CALUDE_angle_difference_range_l2332_233241

theorem angle_difference_range (α β : Real) (h1 : -π < α) (h2 : α < β) (h3 : β < π) :
  -2*π < α - β ∧ α - β < 0 :=
by sorry

end NUMINAMATH_CALUDE_angle_difference_range_l2332_233241


namespace NUMINAMATH_CALUDE_xy_inequality_l2332_233214

theorem xy_inequality (x y : ℝ) (h : (x + 1) * (y + 2) = 8) :
  (x * y - 10)^2 ≥ 64 ∧
  ((x * y - 10)^2 = 64 ↔ (x = 1 ∧ y = 2) ∨ (x = -3 ∧ y = -6)) :=
by sorry

end NUMINAMATH_CALUDE_xy_inequality_l2332_233214


namespace NUMINAMATH_CALUDE_no_very_convex_function_l2332_233269

theorem no_very_convex_function :
  ∀ f : ℝ → ℝ, ∃ x y : ℝ, (f x + f y) / 2 < f ((x + y) / 2) + |x - y| :=
by sorry

end NUMINAMATH_CALUDE_no_very_convex_function_l2332_233269


namespace NUMINAMATH_CALUDE_range_of_a_range_of_m_l2332_233202

-- Define the sets
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | (x - 5) / x ≤ 0}
def C (a : ℝ) : Set ℝ := {x | 3 * a ≤ x ∧ x ≤ 2 * a + 1}
def D (m : ℝ) : Set ℝ := {x | m ≤ x ∧ x ≤ m + 1/2}

-- Part 1
theorem range_of_a : 
  ∀ a : ℝ, (C a ⊆ (A ∩ B)) ↔ (a ∈ Set.Ioo 0 (1/2) ∪ Set.Ioi 1) :=
sorry

-- Part 2
theorem range_of_m :
  ∀ m : ℝ, (∀ x : ℝ, x ∈ D m → x ∈ (A ∪ B)) ∧ 
           (∃ y : ℝ, y ∈ (A ∪ B) ∧ y ∉ D m) ↔
           m ∈ Set.Icc (-2) (9/2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_range_of_m_l2332_233202


namespace NUMINAMATH_CALUDE_at_least_one_quadratic_has_root_l2332_233292

theorem at_least_one_quadratic_has_root (a b c d : ℝ) (h : a * c = 2 * b + 2 * d) :
  (a^2 - 4*b ≥ 0) ∨ (c^2 - 4*d ≥ 0) := by sorry

end NUMINAMATH_CALUDE_at_least_one_quadratic_has_root_l2332_233292


namespace NUMINAMATH_CALUDE_floor_equation_solution_l2332_233288

theorem floor_equation_solution (x : ℝ) : 
  (⌊⌊3 * x⌋ + 1/2⌋ = ⌊x + 4⌋) ↔ (5/3 ≤ x ∧ x < 7/3) :=
sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l2332_233288


namespace NUMINAMATH_CALUDE_min_value_xyz_l2332_233287

theorem min_value_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 8) :
  x + 2 * y + 4 * z ≥ 12 := by
  sorry

end NUMINAMATH_CALUDE_min_value_xyz_l2332_233287


namespace NUMINAMATH_CALUDE_product_divisible_by_1419_l2332_233248

theorem product_divisible_by_1419 : ∃ k : ℕ, 86 * 87 * 88 = 1419 * k := by
  sorry

end NUMINAMATH_CALUDE_product_divisible_by_1419_l2332_233248


namespace NUMINAMATH_CALUDE_clock_centers_distance_l2332_233289

/-- Two identically accurate clocks with hour hands -/
structure Clock where
  center : ℝ × ℝ
  hand_length : ℝ

/-- The configuration of two clocks -/
structure ClockPair where
  clock1 : Clock
  clock2 : Clock
  m : ℝ  -- Minimum distance between hour hand ends
  M : ℝ  -- Maximum distance between hour hand ends

/-- The theorem stating the distance between clock centers -/
theorem clock_centers_distance (cp : ClockPair) :
  let (x1, y1) := cp.clock1.center
  let (x2, y2) := cp.clock2.center
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = (cp.M + cp.m) / 2 := by
  sorry

end NUMINAMATH_CALUDE_clock_centers_distance_l2332_233289


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_seven_twelfths_l2332_233247

theorem sqrt_difference_equals_seven_twelfths :
  Real.sqrt (16 / 9) - Real.sqrt (9 / 16) = 7 / 12 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_seven_twelfths_l2332_233247


namespace NUMINAMATH_CALUDE_apple_bags_problem_l2332_233204

theorem apple_bags_problem (A B C : ℕ) 
  (h1 : A + B + C = 24)
  (h2 : B + C = 18)
  (h3 : A + C = 19) :
  A + B = 11 := by
sorry

end NUMINAMATH_CALUDE_apple_bags_problem_l2332_233204


namespace NUMINAMATH_CALUDE_inequality_solution_and_sqrt2_l2332_233267

-- Define the inequality
def inequality (x : ℝ) : Prop := (5/2 * x - 1) > 3 * x

-- Define the solution set
def solution_set : Set ℝ := {x | x < -2}

-- Theorem statement
theorem inequality_solution_and_sqrt2 :
  (∀ x, inequality x ↔ x ∈ solution_set) ∧
  ¬ inequality (-Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_and_sqrt2_l2332_233267


namespace NUMINAMATH_CALUDE_alex_has_largest_final_answer_l2332_233286

def maria_operation (x : ℕ) : ℕ := ((x - 2) * 3) + 4

def alex_operation (x : ℕ) : ℕ := ((x * 3) - 3) + 4

def lee_operation (x : ℕ) : ℕ := ((x - 2) + 4) * 3

theorem alex_has_largest_final_answer :
  let maria_start := 12
  let alex_start := 15
  let lee_start := 13
  let maria_final := maria_operation maria_start
  let alex_final := alex_operation alex_start
  let lee_final := lee_operation lee_start
  alex_final > maria_final ∧ alex_final > lee_final :=
by sorry

end NUMINAMATH_CALUDE_alex_has_largest_final_answer_l2332_233286


namespace NUMINAMATH_CALUDE_sum_of_last_two_digits_of_8_pow_2003_l2332_233276

theorem sum_of_last_two_digits_of_8_pow_2003 : 
  ∃ (n : ℕ), 8^2003 ≡ n [ZMOD 100] ∧ (n / 10 % 10 + n % 10 = 5) :=
sorry

end NUMINAMATH_CALUDE_sum_of_last_two_digits_of_8_pow_2003_l2332_233276


namespace NUMINAMATH_CALUDE_real_roots_of_polynomial_l2332_233213

def p (x : ℝ) : ℝ := x^5 - 3*x^4 + 3*x^3 - x^2 - 4*x + 4

theorem real_roots_of_polynomial :
  ∀ x : ℝ, p x = 0 ↔ x = -1 ∨ x = 1 ∨ x = 2 :=
by sorry

end NUMINAMATH_CALUDE_real_roots_of_polynomial_l2332_233213


namespace NUMINAMATH_CALUDE_quadrilateral_area_l2332_233205

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_right_angled_at_B_and_D (q : Quadrilateral) : Prop := sorry
def diagonal_length (q : Quadrilateral) (p1 p2 : ℝ × ℝ) : ℝ := sorry
def side_length (q : Quadrilateral) (p1 p2 : ℝ × ℝ) : ℝ := sorry
def area (q : Quadrilateral) : ℝ := sorry

-- Theorem statement
theorem quadrilateral_area 
  (q : Quadrilateral)
  (h1 : is_right_angled_at_B_and_D q)
  (h2 : diagonal_length q q.A q.C = 5)
  (h3 : side_length q q.B q.C = 4)
  (h4 : side_length q q.A q.D = 3) :
  area q = 12 := by sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l2332_233205


namespace NUMINAMATH_CALUDE_fraction_equality_l2332_233209

theorem fraction_equality (a b : ℝ) (h : b ≠ 0) : (2 * a) / (2 * b) = a / b := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2332_233209


namespace NUMINAMATH_CALUDE_cube_root_of_110592_l2332_233285

theorem cube_root_of_110592 :
  ∃ (n : ℕ), n^3 = 110592 ∧ n = 48 :=
by
  -- Define the number
  let number : ℕ := 110592

  -- Define the conditions
  have h1 : 10^3 = 1000 := by sorry
  have h2 : 100^3 = 1000000 := by sorry
  have h3 : 1000 < number ∧ number < 1000000 := by sorry
  have h4 : number % 10 = 2 := by sorry
  have h5 : ∀ (m : ℕ), m % 10 = 8 → (m^3) % 10 = 2 := by sorry
  have h6 : 4^3 = 64 := by sorry
  have h7 : 5^3 = 125 := by sorry
  have h8 : 64 < 110 ∧ 110 < 125 := by sorry

  -- Prove the theorem
  sorry

end NUMINAMATH_CALUDE_cube_root_of_110592_l2332_233285


namespace NUMINAMATH_CALUDE_taxi_charge_correct_l2332_233219

/-- Calculates the total charge for a taxi trip given the initial fee, per-increment charge, increment distance, and total trip distance. -/
def total_charge (initial_fee : ℚ) (per_increment_charge : ℚ) (increment_distance : ℚ) (trip_distance : ℚ) : ℚ :=
  initial_fee + (trip_distance / increment_distance).floor * per_increment_charge

/-- Proves that the total charge for a specific taxi trip is correct. -/
theorem taxi_charge_correct :
  let initial_fee : ℚ := 41/20  -- $2.05
  let per_increment_charge : ℚ := 7/20  -- $0.35
  let increment_distance : ℚ := 2/5  -- 2/5 mile
  let trip_distance : ℚ := 18/5  -- 3.6 miles
  total_charge initial_fee per_increment_charge increment_distance trip_distance = 26/5  -- $5.20
  := by sorry

end NUMINAMATH_CALUDE_taxi_charge_correct_l2332_233219


namespace NUMINAMATH_CALUDE_geometric_sum_first_10_terms_l2332_233297

def geometric_sequence (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ := a₁ * r^(n - 1)

def geometric_sum (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ := a₁ * (1 - r^n) / (1 - r)

theorem geometric_sum_first_10_terms :
  let a₁ : ℚ := 12
  let r : ℚ := 1/3
  let n : ℕ := 10
  geometric_sum a₁ r n = 1062864/59049 := by sorry

end NUMINAMATH_CALUDE_geometric_sum_first_10_terms_l2332_233297


namespace NUMINAMATH_CALUDE_ice_cube_distribution_l2332_233282

/-- Given a total of 30 ice cubes and 6 cups, prove that each cup should contain 5 ice cubes when divided equally. -/
theorem ice_cube_distribution (total_ice_cubes : ℕ) (num_cups : ℕ) (ice_per_cup : ℕ) :
  total_ice_cubes = 30 →
  num_cups = 6 →
  ice_per_cup = total_ice_cubes / num_cups →
  ice_per_cup = 5 := by
  sorry

end NUMINAMATH_CALUDE_ice_cube_distribution_l2332_233282


namespace NUMINAMATH_CALUDE_backpacks_sold_to_dept_store_l2332_233271

def total_backpacks : ℕ := 48
def total_cost : ℕ := 576
def swap_meet_sold : ℕ := 17
def swap_meet_price : ℕ := 18
def dept_store_price : ℕ := 25
def remainder_price : ℕ := 22
def total_profit : ℕ := 442

theorem backpacks_sold_to_dept_store :
  ∃ x : ℕ, 
    x * dept_store_price + 
    swap_meet_sold * swap_meet_price + 
    (total_backpacks - swap_meet_sold - x) * remainder_price - 
    total_cost = total_profit ∧
    x = 10 := by
  sorry

end NUMINAMATH_CALUDE_backpacks_sold_to_dept_store_l2332_233271


namespace NUMINAMATH_CALUDE_expression_simplification_appropriate_integer_value_l2332_233210

theorem expression_simplification (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 2) (h3 : x ≠ -2) :
  (x / (x - 2) - 4 / (x^2 - 2*x)) / ((x + 2) / x^2) = x :=
by sorry

theorem appropriate_integer_value :
  ∃ (x : ℤ), -2 ≤ x ∧ x < Real.sqrt 7 ∧ x ≠ 0 ∧ x ≠ 2 ∧ x = 1 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_appropriate_integer_value_l2332_233210


namespace NUMINAMATH_CALUDE_range_of_a_l2332_233259

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x > 0 → a < x + 4 / x) → 
  a < 4 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l2332_233259


namespace NUMINAMATH_CALUDE_ellipse_equation_1_ellipse_equation_2_l2332_233284

-- Define the ellipse type
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the standard equation of an ellipse
def standard_equation (e : Ellipse) (x y : ℝ) : Prop :=
  y^2 / e.a^2 + x^2 / e.b^2 = 1

-- Define the focal length
def focal_length (e : Ellipse) : ℝ := 2 * e.c

-- Define the sum of distances from a point on the ellipse to the two focal points
def sum_of_distances (e : Ellipse) : ℝ := 2 * e.a

-- Theorem 1
theorem ellipse_equation_1 :
  ∃ (e : Ellipse),
    focal_length e = 4 ∧
    standard_equation e 3 2 ∧
    e.a = 4 ∧ e.b^2 = 12 :=
sorry

-- Theorem 2
theorem ellipse_equation_2 :
  ∃ (e : Ellipse),
    focal_length e = 10 ∧
    sum_of_distances e = 26 ∧
    ((e.a = 13 ∧ e.b = 12) ∨ (e.a = 12 ∧ e.b = 13)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_1_ellipse_equation_2_l2332_233284


namespace NUMINAMATH_CALUDE_triangle_sine_inequality_l2332_233229

theorem triangle_sine_inequality (A B C : Real) : 
  A > 0 → B > 0 → C > 0 → A + B + C = π →
  1 / Real.sin (A / 2) + 1 / Real.sin (B / 2) + 1 / Real.sin (C / 2) ≥ 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_sine_inequality_l2332_233229


namespace NUMINAMATH_CALUDE_tobys_money_sharing_l2332_233291

theorem tobys_money_sharing (initial_amount : ℚ) (brothers : ℕ) (share_fraction : ℚ) :
  initial_amount = 343 →
  brothers = 2 →
  share_fraction = 1/7 →
  initial_amount - (brothers * (share_fraction * initial_amount)) = 245 := by
  sorry

end NUMINAMATH_CALUDE_tobys_money_sharing_l2332_233291


namespace NUMINAMATH_CALUDE_bake_sale_profit_split_l2332_233272

/-- The number of dozens of cookies John makes -/
def dozens : ℕ := 6

/-- The number of cookies in a dozen -/
def cookies_per_dozen : ℕ := 12

/-- The selling price of each cookie in dollars -/
def selling_price : ℚ := 3/2

/-- The cost to make each cookie in dollars -/
def cost_per_cookie : ℚ := 1/4

/-- The amount each charity receives in dollars -/
def charity_amount : ℚ := 45

/-- The number of charities John splits the profit between -/
def num_charities : ℕ := 2

theorem bake_sale_profit_split :
  (dozens * cookies_per_dozen * selling_price - dozens * cookies_per_dozen * cost_per_cookie) / charity_amount = num_charities := by
  sorry

end NUMINAMATH_CALUDE_bake_sale_profit_split_l2332_233272


namespace NUMINAMATH_CALUDE_seated_students_count_l2332_233252

/-- Given a school meeting with teachers and students, calculate the number of seated students. -/
theorem seated_students_count 
  (total_attendees : ℕ) 
  (seated_teachers : ℕ) 
  (standing_students : ℕ) 
  (h1 : total_attendees = 355) 
  (h2 : seated_teachers = 30) 
  (h3 : standing_students = 25) : 
  total_attendees = seated_teachers + standing_students + 300 :=
by sorry

end NUMINAMATH_CALUDE_seated_students_count_l2332_233252


namespace NUMINAMATH_CALUDE_min_value_trigonometric_expression_l2332_233223

theorem min_value_trigonometric_expression (x₁ x₂ x₃ x₄ : ℝ) 
  (h_positive : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0) 
  (h_sum : x₁ + x₂ + x₃ + x₄ = π) : 
  (2 * Real.sin x₁ ^ 2 + 1 / Real.sin x₁ ^ 2) * 
  (2 * Real.sin x₂ ^ 2 + 1 / Real.sin x₂ ^ 2) * 
  (2 * Real.sin x₃ ^ 2 + 1 / Real.sin x₃ ^ 2) * 
  (2 * Real.sin x₄ ^ 2 + 1 / Real.sin x₄ ^ 2) ≥ 81 :=
by sorry

end NUMINAMATH_CALUDE_min_value_trigonometric_expression_l2332_233223


namespace NUMINAMATH_CALUDE_third_largest_number_l2332_233226

/-- Given five numbers in a specific ratio with a known product, 
    this theorem proves the value of the third largest number. -/
theorem third_largest_number 
  (a b c d e : ℝ) 
  (ratio : a / 2.3 = b / 3.7 ∧ a / 2.3 = c / 5.5 ∧ a / 2.3 = d / 7.1 ∧ a / 2.3 = e / 8.9) 
  (product : a * b * c * d * e = 900000) : 
  ∃ (ε : ℝ), abs (c - 14.85) < ε ∧ ε > 0 := by
  sorry

end NUMINAMATH_CALUDE_third_largest_number_l2332_233226


namespace NUMINAMATH_CALUDE_tangent_sum_l2332_233255

theorem tangent_sum (x y : ℝ) 
  (h1 : (Real.sin x / Real.cos y) + (Real.sin y / Real.cos x) = 1)
  (h2 : (Real.cos x / Real.sin y) + (Real.cos y / Real.sin x) = 6) :
  (Real.tan x / Real.tan y) + (Real.tan y / Real.tan x) = 124/13 := by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_l2332_233255


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l2332_233243

theorem cubic_equation_solution (x : ℝ) (h : x^3 + 1/x^3 = -52) : x + 1/x = -4 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l2332_233243


namespace NUMINAMATH_CALUDE_exam_percentage_l2332_233244

theorem exam_percentage (total_items : ℕ) (correct_A correct_B incorrect_B : ℕ) :
  total_items = 60 →
  correct_B = correct_A + 2 →
  incorrect_B = 4 →
  correct_B + incorrect_B = total_items →
  (correct_A : ℚ) / total_items * 100 = 90 := by
  sorry

end NUMINAMATH_CALUDE_exam_percentage_l2332_233244


namespace NUMINAMATH_CALUDE_solve_equation_l2332_233251

theorem solve_equation (x : ℝ) : (x^3 * 6^2) / 432 = 144 → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2332_233251


namespace NUMINAMATH_CALUDE_shara_shells_after_vacation_l2332_233206

/-- Calculates the total number of shells after a vacation -/
def total_shells_after_vacation (initial_shells : ℕ) (shells_per_day : ℕ) (days : ℕ) (fourth_day_shells : ℕ) : ℕ :=
  initial_shells + shells_per_day * days + fourth_day_shells

/-- Proves that Shara has 41 shells after her vacation -/
theorem shara_shells_after_vacation :
  total_shells_after_vacation 20 5 3 6 = 41 := by
  sorry

end NUMINAMATH_CALUDE_shara_shells_after_vacation_l2332_233206


namespace NUMINAMATH_CALUDE_prime_power_divisibility_l2332_233268

theorem prime_power_divisibility : 
  (∃ p : ℕ, p ≥ 7 ∧ Nat.Prime p ∧ (p^4 - 1) % 48 = 0) ∧ 
  (∃ q : ℕ, q ≥ 7 ∧ Nat.Prime q ∧ (q^4 - 1) % 48 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_prime_power_divisibility_l2332_233268


namespace NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l2332_233294

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- Define the theorem
theorem lines_perpendicular_to_plane_are_parallel
  (m n : Line) (α : Plane)
  (h1 : m ≠ n)
  (h2 : perpendicular m α)
  (h3 : perpendicular n α) :
  parallel m n :=
sorry

end NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l2332_233294


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l2332_233224

theorem triangle_angle_proof (A B C : ℝ) (m n : ℝ × ℝ) :
  A + B + C = π →
  m = (Real.sqrt 3 * Real.sin A, Real.sin B) →
  n = (Real.cos B, Real.sqrt 3 * Real.cos A) →
  m.1 * n.1 + m.2 * n.2 = 1 + Real.cos (A + B) →
  C = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_proof_l2332_233224


namespace NUMINAMATH_CALUDE_total_time_equals_sum_l2332_233281

/-- The total time Porche initially had for homework -/
def total_time : ℕ := 180

/-- Time required for math homework -/
def math_time : ℕ := 45

/-- Time required for English homework -/
def english_time : ℕ := 30

/-- Time required for science homework -/
def science_time : ℕ := 50

/-- Time required for history homework -/
def history_time : ℕ := 25

/-- Time left for the special project -/
def project_time : ℕ := 30

/-- Theorem stating that the total time is the sum of all homework times -/
theorem total_time_equals_sum :
  total_time = math_time + english_time + science_time + history_time + project_time := by
  sorry

end NUMINAMATH_CALUDE_total_time_equals_sum_l2332_233281


namespace NUMINAMATH_CALUDE_ratio_problem_l2332_233237

theorem ratio_problem (a b c : ℝ) (h1 : b / a = 3) (h2 : c / b = 2) : 
  (a + b) / (b + c) = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l2332_233237


namespace NUMINAMATH_CALUDE_x_minus_y_equals_three_l2332_233274

theorem x_minus_y_equals_three (x y : ℝ) (h : |x - 2| + (y + 1)^2 = 0) : x - y = 3 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_three_l2332_233274


namespace NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_l2332_233215

theorem sqrt_x_div_sqrt_y (x y : ℝ) :
  (1/3)^2 + (1/4)^2 = (17*x/60) * ((1/5)^2 + (1/6)^2) →
  Real.sqrt x / Real.sqrt y = (25/2) * Real.sqrt (60/1037) := by
sorry

end NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_l2332_233215


namespace NUMINAMATH_CALUDE_roberto_outfits_l2332_233258

/-- The number of different outfits Roberto can create -/
def number_of_outfits (trousers shirts jackets hats : ℕ) : ℕ :=
  trousers * shirts * jackets * hats

/-- Theorem stating the number of outfits Roberto can create -/
theorem roberto_outfits :
  number_of_outfits 5 8 4 2 = 320 := by
  sorry

end NUMINAMATH_CALUDE_roberto_outfits_l2332_233258


namespace NUMINAMATH_CALUDE_two_digit_product_less_than_five_digits_l2332_233217

theorem two_digit_product_less_than_five_digits : ∀ a b : ℕ, 
  10 ≤ a ∧ a ≤ 99 → 10 ≤ b ∧ b ≤ 99 → a * b < 10000 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_product_less_than_five_digits_l2332_233217


namespace NUMINAMATH_CALUDE_motorboat_travel_theorem_l2332_233225

noncomputable def motorboat_travel_fraction (S : ℝ) (v : ℝ) : Set ℝ :=
  let u₁ := (2 / 3) * v
  let u₂ := (1 / 3) * v
  let V_m₁ := 2 * v + u₁
  let V_m₂ := 2 * v + u₂
  let V_b₁ := 3 * v - u₁
  let V_b₂ := 3 * v - u₂
  let t₁ := S / (5 * v)
  let d := (56 / 225) * S
  { x | x = (V_m₁ * t₁ + d) / S ∨ x = (V_m₂ * t₁ + d) / S }

theorem motorboat_travel_theorem (S : ℝ) (v : ℝ) (h_S : S > 0) (h_v : v > 0) :
  motorboat_travel_fraction S v = {161 / 225, 176 / 225} := by
  sorry

end NUMINAMATH_CALUDE_motorboat_travel_theorem_l2332_233225


namespace NUMINAMATH_CALUDE_phi_value_l2332_233201

open Real

noncomputable def f (x φ : ℝ) : ℝ := sin (Real.sqrt 3 * x + φ)

noncomputable def f_deriv (x φ : ℝ) : ℝ := Real.sqrt 3 * cos (Real.sqrt 3 * x + φ)

theorem phi_value (φ : ℝ) (h1 : 0 < φ) (h2 : φ < π) 
  (h3 : ∀ x, f x φ + f_deriv x φ = -(f (-x) φ + f_deriv (-x) φ)) : 
  φ = 2 * π / 3 := by
sorry

end NUMINAMATH_CALUDE_phi_value_l2332_233201


namespace NUMINAMATH_CALUDE_one_sixth_of_x_l2332_233290

theorem one_sixth_of_x (x : ℝ) (h : x / 3 = 4) : x / 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_one_sixth_of_x_l2332_233290


namespace NUMINAMATH_CALUDE_train_passing_platform_l2332_233257

/-- A train passes a platform -/
theorem train_passing_platform 
  (train_length : ℝ) 
  (tree_crossing_time : ℝ) 
  (platform_length : ℝ) 
  (h1 : train_length = 600) 
  (h2 : tree_crossing_time = 60) 
  (h3 : platform_length = 450) : 
  (train_length + platform_length) / (train_length / tree_crossing_time) = 105 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_platform_l2332_233257


namespace NUMINAMATH_CALUDE_sqrt_difference_equality_l2332_233277

theorem sqrt_difference_equality : Real.sqrt (49 + 121) - Real.sqrt (36 - 9) = Real.sqrt 170 - 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equality_l2332_233277


namespace NUMINAMATH_CALUDE_subtracted_number_l2332_233233

theorem subtracted_number (x n : ℚ) : 
  x / 4 - (x - n) / 6 = 1 → x = 6 → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_number_l2332_233233


namespace NUMINAMATH_CALUDE_expression_simplification_l2332_233279

theorem expression_simplification (a b : ℤ) (h : b = a + 1) (ha : a = 2015) :
  (a^4 - 3*a^3*b + 3*a^2*b^2 - b^4 + a) / (a*b) = -(a-1)^2 / a^3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2332_233279


namespace NUMINAMATH_CALUDE_remainder_97_103_mod_9_l2332_233261

theorem remainder_97_103_mod_9 : (97 * 103) % 9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_97_103_mod_9_l2332_233261


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2332_233295

theorem geometric_sequence_property (a : ℕ → ℝ) (h_geom : ∀ n m : ℕ, a (n + m) = a n * a m) :
  a 6 = 6 → a 9 = 9 → a 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2332_233295


namespace NUMINAMATH_CALUDE_one_distinct_computable_value_l2332_233275

/-- Represents a valid parenthesization of the expression 3^3^3^3 --/
inductive Parenthesization
| Original : Parenthesization
| Left : Parenthesization
| Middle : Parenthesization
| Right : Parenthesization
| DoubleLeft : Parenthesization
| DoubleRight : Parenthesization

/-- Evaluates a given parenthesization to a natural number if computable --/
def evaluate : Parenthesization → Option ℕ
| Parenthesization.Original => none
| Parenthesization.Left => none
| Parenthesization.Middle => some 19683
| Parenthesization.Right => none
| Parenthesization.DoubleLeft => none
| Parenthesization.DoubleRight => none

/-- The number of distinct, computable values when changing the order of exponentiation in 3^3^3^3 --/
def distinctComputableValues : ℕ :=
  (List.map evaluate [Parenthesization.Left, Parenthesization.Middle, Parenthesization.Right,
                      Parenthesization.DoubleLeft, Parenthesization.DoubleRight]).filterMap id |>.eraseDups |>.length

theorem one_distinct_computable_value : distinctComputableValues = 1 := by
  sorry

end NUMINAMATH_CALUDE_one_distinct_computable_value_l2332_233275


namespace NUMINAMATH_CALUDE_time_per_toy_l2332_233211

/-- Given a worker who makes 50 toys in 150 hours, prove that the time taken to make one toy is 3 hours. -/
theorem time_per_toy (total_hours : ℝ) (total_toys : ℝ) (h1 : total_hours = 150) (h2 : total_toys = 50) :
  total_hours / total_toys = 3 := by
  sorry

end NUMINAMATH_CALUDE_time_per_toy_l2332_233211


namespace NUMINAMATH_CALUDE_collinear_vectors_m_value_l2332_233280

/-- Two vectors are collinear in opposite directions if one is a negative scalar multiple of the other -/
def collinear_opposite (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k < 0 ∧ a.1 = k * b.1 ∧ a.2 = k * b.2

theorem collinear_vectors_m_value (m : ℝ) :
  let a : ℝ × ℝ := (m, -4)
  let b : ℝ × ℝ := (-1, m + 3)
  collinear_opposite a b → m = 1 := by
sorry

end NUMINAMATH_CALUDE_collinear_vectors_m_value_l2332_233280


namespace NUMINAMATH_CALUDE_sport_popularity_order_l2332_233216

/-- Represents a sport with its popularity fraction -/
structure Sport where
  name : String
  popularity : Rat

/-- Determines if one sport is more popular than another -/
def morePopularThan (s1 s2 : Sport) : Prop :=
  s1.popularity > s2.popularity

theorem sport_popularity_order (basketball tennis volleyball : Sport)
  (h_basketball : basketball.name = "Basketball" ∧ basketball.popularity = 9/24)
  (h_tennis : tennis.name = "Tennis" ∧ tennis.popularity = 8/24)
  (h_volleyball : volleyball.name = "Volleyball" ∧ volleyball.popularity = 7/24) :
  morePopularThan basketball tennis ∧ 
  morePopularThan tennis volleyball ∧
  [basketball.name, tennis.name, volleyball.name] = ["Basketball", "Tennis", "Volleyball"] :=
by sorry

end NUMINAMATH_CALUDE_sport_popularity_order_l2332_233216


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_when_area_equals_perimeter_l2332_233283

/-- A triangle with an inscribed circle -/
structure Triangle :=
  (area : ℝ)
  (perimeter : ℝ)
  (inradius : ℝ)

/-- The theorem stating that if a triangle's area equals its perimeter, 
    then the radius of its inscribed circle is 2 -/
theorem inscribed_circle_radius_when_area_equals_perimeter 
  (t : Triangle) 
  (h : t.area = t.perimeter) : 
  t.inradius = 2 :=
sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_when_area_equals_perimeter_l2332_233283


namespace NUMINAMATH_CALUDE_triangle_perimeter_l2332_233239

theorem triangle_perimeter (a b c A B C : ℝ) : 
  (c * Real.cos B + b * Real.cos C = 2 * a * Real.cos A) →
  (a = 2) →
  (1/2 * b * c * Real.sin A = Real.sqrt 3) →
  (a + b + c = 6) := by
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l2332_233239


namespace NUMINAMATH_CALUDE_protective_clothing_equation_l2332_233263

/-- Represents the equation for the protective clothing production problem -/
theorem protective_clothing_equation (x : ℝ) (h : x > 0) :
  let total_sets := 1000
  let increase_rate := 0.2
  let days_ahead := 2
  let original_days := total_sets / x
  let actual_days := total_sets / (x * (1 + increase_rate))
  original_days - actual_days = days_ahead :=
by sorry

end NUMINAMATH_CALUDE_protective_clothing_equation_l2332_233263


namespace NUMINAMATH_CALUDE_product_of_square_roots_l2332_233296

theorem product_of_square_roots (x y z : ℝ) :
  x = 75 → y = 48 → z = 12 → Real.sqrt x * Real.sqrt y * Real.sqrt z = 120 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_product_of_square_roots_l2332_233296


namespace NUMINAMATH_CALUDE_cone_unfolded_side_view_is_sector_l2332_233299

/-- A shape with one curved side and two straight sides -/
structure ConeUnfoldedSideView where
  curved_side : ℕ
  straight_sides : ℕ
  h_curved : curved_side = 1
  h_straight : straight_sides = 2

/-- Definition of a sector -/
def is_sector (shape : ConeUnfoldedSideView) : Prop :=
  shape.curved_side = 1 ∧ shape.straight_sides = 2

/-- Theorem: The unfolded side view of a cone is a sector -/
theorem cone_unfolded_side_view_is_sector (shape : ConeUnfoldedSideView) :
  is_sector shape :=
by sorry

end NUMINAMATH_CALUDE_cone_unfolded_side_view_is_sector_l2332_233299


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2332_233207

def is_geometric_sequence (α : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, α (n + 1) = α n * r

theorem geometric_sequence_property 
  (α : ℕ → ℝ) 
  (h_geometric : is_geometric_sequence α) 
  (h_product : α 4 * α 5 * α 6 = 27) : 
  α 5 = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2332_233207


namespace NUMINAMATH_CALUDE_intersection_range_l2332_233235

-- Define the semicircle
def semicircle (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 2)^2 = 4 ∧ y ≥ 2

-- Define the line
def line (x y k : ℝ) : Prop :=
  y = k * (x - 1) + 5

-- Define the condition for two distinct intersection points
def has_two_intersections (k : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ 
    semicircle x₁ y₁ ∧ semicircle x₂ y₂ ∧
    line x₁ y₁ k ∧ line x₂ y₂ k

-- Theorem statement
theorem intersection_range :
  ∀ k : ℝ, has_two_intersections k ↔ 
    (k ∈ Set.Icc (-3/2) (-Real.sqrt 5/2) ∨ 
     k ∈ Set.Ioc (Real.sqrt 5/2) (3/2)) :=
sorry

end NUMINAMATH_CALUDE_intersection_range_l2332_233235
