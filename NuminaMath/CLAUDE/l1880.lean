import Mathlib

namespace tan_945_degrees_l1880_188031

theorem tan_945_degrees : Real.tan (945 * π / 180) = 1 := by sorry

end tan_945_degrees_l1880_188031


namespace collinear_points_values_coplanar_points_value_l1880_188066

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point3D) : Prop :=
  ∃ t : ℝ, (p3.x - p1.x, p3.y - p1.y, p3.z - p1.z) = 
    t • (p2.x - p1.x, p2.y - p1.y, p2.z - p1.z)

/-- Check if four points are coplanar -/
def coplanar (p1 p2 p3 p4 : Point3D) : Prop :=
  ∃ a b c : ℝ, 
    (p4.x - p1.x, p4.y - p1.y, p4.z - p1.z) = 
    a • (p2.x - p1.x, p2.y - p1.y, p2.z - p1.z) +
    b • (p3.x - p1.x, p3.y - p1.y, p3.z - p1.z)

theorem collinear_points_values (a b : ℝ) :
  let A : Point3D := ⟨2, a, -1⟩
  let B : Point3D := ⟨-2, 3, b⟩
  let C : Point3D := ⟨1, 2, -2⟩
  collinear A B C → a = 5/3 ∧ b = -5 := by
  sorry

theorem coplanar_points_value (a : ℝ) :
  let A : Point3D := ⟨2, a, -1⟩
  let B : Point3D := ⟨-2, 3, -3⟩
  let C : Point3D := ⟨1, 2, -2⟩
  let D : Point3D := ⟨-1, 3, -3⟩
  coplanar A B C D → a = 1 := by
  sorry

end collinear_points_values_coplanar_points_value_l1880_188066


namespace tangent_parallel_to_x_axis_l1880_188019

/-- The function f(x) defined in the problem -/
def f (t : ℝ) (x : ℝ) : ℝ := x^3 + (2*t - 1)*x + 3

/-- The derivative of f(x) with respect to x -/
def f' (t : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*t - 1

/-- Theorem stating that t = -1 given the conditions -/
theorem tangent_parallel_to_x_axis (t : ℝ) : 
  (f' t (-1) = 0) → t = -1 := by
  sorry

#check tangent_parallel_to_x_axis

end tangent_parallel_to_x_axis_l1880_188019


namespace angle_between_vectors_l1880_188094

theorem angle_between_vectors (a b : ℝ × ℝ) :
  a = (2, 0) →
  ‖b‖ = 1 →
  ‖a + b‖ = Real.sqrt 7 →
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (‖a‖ * ‖b‖)) = π / 3 := by
  sorry

end angle_between_vectors_l1880_188094


namespace envelope_counting_time_l1880_188067

/-- Represents the time in seconds to count a given number of envelopes -/
def count_time (envelopes : ℕ) : ℕ :=
  if envelopes ≤ 100 then
    min ((100 - envelopes) / 10 * 10) (envelopes / 10 * 10)
  else
    envelopes / 10 * 10

theorem envelope_counting_time :
  (count_time 60 = 40) ∧ (count_time 90 = 10) := by
  sorry

end envelope_counting_time_l1880_188067


namespace line_y_coordinate_proof_l1880_188054

/-- Given a line passing through points (-1, y) and (5, 0.8) with slope 0.8,
    prove that the y-coordinate of the first point is 4. -/
theorem line_y_coordinate_proof (y : ℝ) : 
  (0.8 - y) / (5 - (-1)) = 0.8 → y = 4 := by
  sorry

end line_y_coordinate_proof_l1880_188054


namespace lcm_gcd_product_l1880_188092

theorem lcm_gcd_product (a b : ℕ) (ha : a = 15) (hb : b = 10) :
  Nat.lcm a b * Nat.gcd a b = 150 := by
  sorry

end lcm_gcd_product_l1880_188092


namespace A_contribution_is_500_l1880_188074

def total : ℕ := 820
def ratio_A_to_B : Rat := 5 / 2
def ratio_B_to_C : Rat := 5 / 3

theorem A_contribution_is_500 : 
  ∃ (a b c : ℕ), 
    a + b + c = total ∧ 
    (a : ℚ) / b = ratio_A_to_B ∧ 
    (b : ℚ) / c = ratio_B_to_C ∧ 
    a = 500 := by
  sorry

end A_contribution_is_500_l1880_188074


namespace concession_stand_theorem_l1880_188008

def concession_stand_revenue (hot_dog_price soda_price : ℚ) 
                             (total_items hot_dogs_sold : ℕ) : ℚ :=
  let sodas_sold := total_items - hot_dogs_sold
  hot_dog_price * hot_dogs_sold + soda_price * sodas_sold

theorem concession_stand_theorem :
  concession_stand_revenue (3/2) (1/2) 87 35 = 157/2 := by
  sorry

end concession_stand_theorem_l1880_188008


namespace binomial_n_minus_two_l1880_188040

theorem binomial_n_minus_two (n : ℕ+) : Nat.choose n (n - 2) = n * (n - 1) / 2 := by
  sorry

end binomial_n_minus_two_l1880_188040


namespace dream_star_results_l1880_188005

/-- Represents the results of a football team in a league --/
structure TeamResults where
  games_played : ℕ
  games_won : ℕ
  games_drawn : ℕ
  games_lost : ℕ
  points : ℕ

/-- Calculates the points earned by a team based on their results --/
def calculate_points (r : TeamResults) : ℕ :=
  3 * r.games_won + r.games_drawn

/-- Theorem stating the unique solution for the given problem --/
theorem dream_star_results :
  ∃! r : TeamResults,
    r.games_played = 9 ∧
    r.games_lost = 2 ∧
    r.points = 17 ∧
    r.games_played = r.games_won + r.games_drawn + r.games_lost ∧
    r.points = calculate_points r ∧
    r.games_won = 5 ∧
    r.games_drawn = 2 := by
  sorry

#check dream_star_results

end dream_star_results_l1880_188005


namespace complex_multiplication_l1880_188072

theorem complex_multiplication : 
  (3 - 4 * Complex.I) * (2 + Complex.I) = 10 - 5 * Complex.I := by
  sorry

end complex_multiplication_l1880_188072


namespace cucumber_weight_after_evaporation_l1880_188018

theorem cucumber_weight_after_evaporation 
  (initial_weight : ℝ) 
  (initial_water_percent : ℝ) 
  (final_water_percent : ℝ) 
  (h1 : initial_weight = 100) 
  (h2 : initial_water_percent = 0.99) 
  (h3 : final_water_percent = 0.98) : 
  ∃ (final_weight : ℝ), final_weight = 50 ∧ 
    (1 - initial_water_percent) * initial_weight = 
    (1 - final_water_percent) * final_weight := by
  sorry

end cucumber_weight_after_evaporation_l1880_188018


namespace tangent_line_at_zero_l1880_188078

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 2 * Real.cos x

theorem tangent_line_at_zero (x y : ℝ) :
  (x - y + 3 = 0) ↔ 
  (∃ (m : ℝ), y - f 0 = m * (x - 0) ∧ 
               m = (deriv f) 0 ∧
               y = f 0 + m * x) :=
by sorry

end tangent_line_at_zero_l1880_188078


namespace tino_jellybean_count_l1880_188016

/-- The number of jellybeans each person has. -/
structure JellybeanCount where
  tino : ℕ
  lee : ℕ
  arnold : ℕ

/-- The conditions of the jellybean problem. -/
def jellybean_conditions (j : JellybeanCount) : Prop :=
  j.tino = j.lee + 24 ∧ 
  j.arnold * 2 = j.lee ∧ 
  j.arnold = 5

/-- The theorem stating that Tino has 34 jellybeans under the given conditions. -/
theorem tino_jellybean_count (j : JellybeanCount) 
  (h : jellybean_conditions j) : j.tino = 34 := by
  sorry

end tino_jellybean_count_l1880_188016


namespace parabola_focus_distance_l1880_188080

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8 * (x + 2)

-- Define the focus of the parabola
def focus : ℝ × ℝ := (0, 0)

-- Define the line with 60° inclination
def line (x y : ℝ) : Prop := y = Real.sqrt 3 * x

-- Define the intersection points A and B (we don't need to calculate them explicitly)
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Define the perpendicular bisector of AB
def perp_bisector (x y : ℝ) : Prop := sorry

-- Define point P as the intersection of the perpendicular bisector with the x-axis
def P : ℝ × ℝ := sorry

-- State the theorem
theorem parabola_focus_distance :
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧
  line A.1 A.2 ∧ line B.1 B.2 ∧
  perp_bisector P.1 P.2 ∧ P.2 = 0 →
  Real.sqrt ((P.1 - focus.1)^2 + (P.2 - focus.2)^2) = 16/3 := by
  sorry

end parabola_focus_distance_l1880_188080


namespace sum_of_squares_and_products_l1880_188012

theorem sum_of_squares_and_products (a b c : ℝ) 
  (h_nonneg_a : a ≥ 0) (h_nonneg_b : b ≥ 0) (h_nonneg_c : c ≥ 0)
  (h_sum_squares : a^2 + b^2 + c^2 = 48)
  (h_sum_products : a*b + b*c + c*a = 18) :
  a + b + c = 2 * Real.sqrt 21 := by
sorry

end sum_of_squares_and_products_l1880_188012


namespace average_production_l1880_188025

/-- Given a production of 4000 items/month for 3 months and 4500 items/month for 9 months,
    the average production for 12 months is 4375 items/month. -/
theorem average_production (first_3_months : ℕ) (next_9_months : ℕ) (total_months : ℕ) :
  first_3_months = 3 →
  next_9_months = 9 →
  total_months = first_3_months + next_9_months →
  (first_3_months * 4000 + next_9_months * 4500) / total_months = 4375 :=
by sorry

end average_production_l1880_188025


namespace lego_count_l1880_188037

theorem lego_count (initial_legos : ℝ) (won_legos : ℝ) :
  initial_legos + won_legos = initial_legos + won_legos :=
by sorry

end lego_count_l1880_188037


namespace mike_height_l1880_188026

/-- Converts feet and inches to total inches -/
def to_inches (feet : ℕ) (inches : ℕ) : ℕ := feet * 12 + inches

/-- Converts total inches to feet and inches -/
def to_feet_and_inches (total_inches : ℕ) : ℕ × ℕ :=
  (total_inches / 12, total_inches % 12)

theorem mike_height (mark_feet : ℕ) (mark_inches : ℕ) (mike_inches : ℕ) :
  mark_feet = 5 →
  mark_inches = 3 →
  mike_inches = 1 →
  to_inches mark_feet mark_inches + 10 = to_inches 6 mike_inches :=
by sorry

end mike_height_l1880_188026


namespace schedule_arrangements_proof_l1880_188057

/-- Represents the number of periods in the morning -/
def morning_periods : Nat := 4

/-- Represents the number of periods in the afternoon -/
def afternoon_periods : Nat := 3

/-- Represents the total number of periods in a day -/
def total_periods : Nat := morning_periods + afternoon_periods

/-- Represents the number of subjects that require two consecutive periods -/
def two_period_subjects : Nat := 2

/-- Represents the number of subjects that require one period -/
def one_period_subjects : Nat := 4

/-- Represents the total number of subjects to be scheduled -/
def total_subjects : Nat := two_period_subjects + one_period_subjects

/-- Calculates the number of possible arrangements for the schedule -/
def schedule_arrangements : Nat := 336

theorem schedule_arrangements_proof :
  schedule_arrangements = 336 := by sorry

end schedule_arrangements_proof_l1880_188057


namespace inscribed_circle_radius_in_60_degree_sector_l1880_188021

/-- The radius of a circle inscribed in a sector with a central angle of 60° and radius R is R/3 -/
theorem inscribed_circle_radius_in_60_degree_sector (R : ℝ) (R_pos : R > 0) :
  let sector_angle : ℝ := 60 * π / 180
  let inscribed_radius : ℝ := R / 3
  inscribed_radius = R / 3 := by sorry

end inscribed_circle_radius_in_60_degree_sector_l1880_188021


namespace taehyung_candy_distribution_l1880_188085

/-- Given a total number of candies and the number of candies to be given to each friend,
    calculate the maximum number of friends who can receive candies. -/
def max_friends_with_candies (total_candies : ℕ) (candies_per_friend : ℕ) : ℕ :=
  total_candies / candies_per_friend

/-- Theorem: Given 45 candies and distributing 5 candies per friend,
    the maximum number of friends who can receive candies is 9. -/
theorem taehyung_candy_distribution :
  max_friends_with_candies 45 5 = 9 := by
  sorry

end taehyung_candy_distribution_l1880_188085


namespace smallest_crate_dimension_l1880_188048

/-- Represents the dimensions of a rectangular crate -/
structure CrateDimensions where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a right circular cylinder -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Checks if a cylinder can fit upright in a crate -/
def cylinderFitsInCrate (crate : CrateDimensions) (cylinder : Cylinder) : Prop :=
  2 * cylinder.radius ≤ min crate.x (min crate.y crate.z)

/-- The theorem stating the smallest dimension of the crate -/
theorem smallest_crate_dimension
  (crate : CrateDimensions)
  (cylinder : Cylinder)
  (h1 : crate.y = 8)
  (h2 : crate.z = 12)
  (h3 : cylinder.radius = 3)
  (h4 : cylinderFitsInCrate crate cylinder) :
  crate.x ≥ 6 := by
  sorry

end smallest_crate_dimension_l1880_188048


namespace neo_tokropolis_population_change_is_40_l1880_188039

/-- Represents the population change in Neo-Tokropolis over a month -/
def neo_tokropolis_population_change : ℚ :=
  let births_per_day : ℚ := 24 / 12
  let deaths_per_day : ℚ := 24 / 36
  let net_change_per_day : ℚ := births_per_day - deaths_per_day
  let days_in_month : ℚ := 30
  net_change_per_day * days_in_month

/-- Theorem stating that the population change in Neo-Tokropolis over a month is 40 -/
theorem neo_tokropolis_population_change_is_40 :
  neo_tokropolis_population_change = 40 := by
  sorry

#eval neo_tokropolis_population_change

end neo_tokropolis_population_change_is_40_l1880_188039


namespace intersection_is_canonical_line_intersection_is_canonical_line_proof_l1880_188022

/-- Represents a plane in 3D space defined by ax + by + cz + d = 0 --/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Represents a line in 3D space in canonical form (x-x₀)/a = (y-y₀)/b = (z-z₀)/c --/
structure CanonicalLine where
  x₀ : ℝ
  y₀ : ℝ
  z₀ : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- Returns true if the given point (x, y, z) lies on the plane --/
def Plane.contains (p : Plane) (x y z : ℝ) : Prop :=
  p.a * x + p.b * y + p.c * z + p.d = 0

/-- Returns true if the given point (x, y, z) lies on the canonical line --/
def CanonicalLine.contains (l : CanonicalLine) (x y z : ℝ) : Prop :=
  (x - l.x₀) / l.a = (y - l.y₀) / l.b ∧ (y - l.y₀) / l.b = (z - l.z₀) / l.c

/-- The main theorem stating that the intersection of the two given planes
    is exactly the line defined by the given canonical equations --/
theorem intersection_is_canonical_line (p₁ p₂ : Plane) (l : CanonicalLine) : Prop :=
  (p₁.a = 4 ∧ p₁.b = 1 ∧ p₁.c = 1 ∧ p₁.d = 2) →
  (p₂.a = 2 ∧ p₂.b = -1 ∧ p₂.c = -3 ∧ p₂.d = 8) →
  (l.x₀ = 1 ∧ l.y₀ = -6 ∧ l.z₀ = 0 ∧ l.a = -2 ∧ l.b = 14 ∧ l.c = -6) →
  ∀ x y z : ℝ, (p₁.contains x y z ∧ p₂.contains x y z) ↔ l.contains x y z

theorem intersection_is_canonical_line_proof : intersection_is_canonical_line 
  { a := 4, b := 1, c := 1, d := 2 }
  { a := 2, b := -1, c := -3, d := 8 }
  { x₀ := 1, y₀ := -6, z₀ := 0, a := -2, b := 14, c := -6 } := by
  sorry

end intersection_is_canonical_line_intersection_is_canonical_line_proof_l1880_188022


namespace bac_hex_to_decimal_l1880_188024

/-- Converts a hexadecimal digit to its decimal value -/
def hex_to_decimal (digit : Char) : ℕ :=
  match digit with
  | 'B' => 11
  | 'A' => 10
  | 'C' => 12
  | _ => 0  -- Default case, should not be reached for our specific problem

/-- Converts a three-digit hexadecimal number to decimal -/
def hex_to_decimal_3digit (h1 h2 h3 : Char) : ℕ :=
  (hex_to_decimal h1) * 256 + (hex_to_decimal h2) * 16 + (hex_to_decimal h3)

theorem bac_hex_to_decimal :
  hex_to_decimal_3digit 'B' 'A' 'C' = 2988 := by
  sorry

end bac_hex_to_decimal_l1880_188024


namespace sqrt_equation_solution_l1880_188032

theorem sqrt_equation_solution :
  ∃ x : ℝ, (Real.sqrt (3 * x + 15) = 12) ∧ (x = 43) := by
  sorry

end sqrt_equation_solution_l1880_188032


namespace zongzi_pricing_solution_l1880_188079

/-- Represents the purchase and sales scenario of zongzi -/
structure ZongziScenario where
  egg_yolk_price : ℝ
  red_bean_price : ℝ
  first_purchase_egg : ℕ
  first_purchase_red : ℕ
  first_purchase_total : ℝ
  second_purchase_egg : ℕ
  second_purchase_red : ℕ
  second_purchase_total : ℝ
  initial_selling_price : ℝ
  initial_daily_sales : ℕ
  price_reduction : ℝ
  sales_increase : ℕ
  target_daily_profit : ℝ

/-- Theorem stating the solution to the zongzi pricing problem -/
theorem zongzi_pricing_solution (s : ZongziScenario)
  (h1 : s.first_purchase_egg * s.egg_yolk_price + s.first_purchase_red * s.red_bean_price = s.first_purchase_total)
  (h2 : s.second_purchase_egg * s.egg_yolk_price + s.second_purchase_red * s.red_bean_price = s.second_purchase_total)
  (h3 : s.first_purchase_egg = 60 ∧ s.first_purchase_red = 90 ∧ s.first_purchase_total = 4800)
  (h4 : s.second_purchase_egg = 40 ∧ s.second_purchase_red = 80 ∧ s.second_purchase_total = 3600)
  (h5 : s.initial_selling_price = 70 ∧ s.initial_daily_sales = 20)
  (h6 : s.price_reduction = 1 ∧ s.sales_increase = 5)
  (h7 : s.target_daily_profit = 220) :
  s.egg_yolk_price = 50 ∧ s.red_bean_price = 20 ∧
  ∃ (selling_price : ℝ),
    selling_price = 52 ∧
    (selling_price - s.egg_yolk_price) * (s.initial_daily_sales + s.sales_increase * (s.initial_selling_price - selling_price)) = s.target_daily_profit :=
by sorry

end zongzi_pricing_solution_l1880_188079


namespace imaginary_power_sum_l1880_188086

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_sum : i^45 + i^345 = 2 * i := by sorry

end imaginary_power_sum_l1880_188086


namespace furniture_production_l1880_188007

theorem furniture_production (x : ℝ) (h : x > 0) :
  (540 / x - 540 / (x + 2) = 3) ↔ 
  (∃ (original_days actual_days : ℝ),
    original_days > 0 ∧
    actual_days > 0 ∧
    original_days - actual_days = 3 ∧
    original_days * x = 540 ∧
    actual_days * (x + 2) = 540) :=
  sorry

end furniture_production_l1880_188007


namespace hannahs_measuring_spoons_price_l1880_188042

/-- The price of each measuring spoon set given the conditions of Hannah's sales and purchases -/
theorem hannahs_measuring_spoons_price 
  (cookie_count : ℕ) 
  (cookie_price : ℚ) 
  (cupcake_count : ℕ) 
  (cupcake_price : ℚ) 
  (spoon_set_count : ℕ) 
  (money_left : ℚ)
  (h1 : cookie_count = 40)
  (h2 : cookie_price = 4/5)
  (h3 : cupcake_count = 30)
  (h4 : cupcake_price = 2)
  (h5 : spoon_set_count = 2)
  (h6 : money_left = 79) :
  let total_earnings := cookie_count * cookie_price + cupcake_count * cupcake_price
  let spoon_sets_cost := total_earnings - money_left
  let price_per_spoon_set := spoon_sets_cost / spoon_set_count
  price_per_spoon_set = 13/2 := by sorry

end hannahs_measuring_spoons_price_l1880_188042


namespace garden_width_l1880_188059

theorem garden_width (garden_perimeter : ℝ) (playground_length playground_width : ℝ) 
  (h1 : garden_perimeter = 64)
  (h2 : playground_length = 16)
  (h3 : playground_width = 12) :
  ∃ (garden_width : ℝ),
    garden_width > 0 ∧
    garden_width < garden_perimeter / 2 ∧
    (garden_perimeter / 2 - garden_width) * garden_width = playground_length * playground_width ∧
    garden_width = 12 := by
  sorry

#check garden_width

end garden_width_l1880_188059


namespace special_choose_result_l1880_188045

/-- The number of ways to choose k elements from a set of n elements -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose 6 players from 16, with at most one from a special group of 4 -/
def specialChoose : ℕ :=
  choose 16 6 - (choose 4 2 * choose 12 4 + choose 4 3 * choose 12 3 + choose 4 4 * choose 12 2)

theorem special_choose_result : specialChoose = 4092 := by sorry

end special_choose_result_l1880_188045


namespace sum_of_reciprocal_relations_l1880_188076

theorem sum_of_reciprocal_relations (x y : ℚ) 
  (h1 : x ≠ 0) (h2 : y ≠ 0)
  (h3 : 1 / x + 1 / y = 4) 
  (h4 : 1 / x - 1 / y = -6) : 
  x + y = -4/5 := by
  sorry

end sum_of_reciprocal_relations_l1880_188076


namespace third_question_points_l1880_188095

/-- Represents a quiz with a sequence of questions and their point values. -/
structure Quiz where
  num_questions : Nat
  first_question_points : Nat
  point_increase : Nat
  total_points : Nat

/-- Calculates the points for a specific question in the quiz. -/
def question_points (q : Quiz) (n : Nat) : Nat :=
  q.first_question_points + (n - 1) * q.point_increase

/-- Calculates the sum of points for all questions in the quiz. -/
def sum_points (q : Quiz) : Nat :=
  Finset.sum (Finset.range q.num_questions) (λ i => question_points q (i + 1))

/-- The main theorem stating that the third question is worth 39 points. -/
theorem third_question_points (q : Quiz) 
  (h1 : q.num_questions = 8)
  (h2 : q.point_increase = 4)
  (h3 : q.total_points = 360)
  (h4 : sum_points q = q.total_points) :
  question_points q 3 = 39 := by
  sorry

#eval question_points { num_questions := 8, first_question_points := 31, point_increase := 4, total_points := 360 } 3

end third_question_points_l1880_188095


namespace special_sequence_theorem_l1880_188010

/-- A sequence of 2017 integers satisfying specific conditions -/
def SpecialSequence : Type := Fin 2017 → Int

/-- The sum of squares of any 7 numbers in the sequence is 7 -/
def SumSquaresSevenIsSeven (seq : SpecialSequence) : Prop :=
  ∀ (s : Finset (Fin 2017)), s.card = 7 → (s.sum (λ i => (seq i)^2) = 7)

/-- The sum of any 11 numbers in the sequence is positive -/
def SumElevenIsPositive (seq : SpecialSequence) : Prop :=
  ∀ (s : Finset (Fin 2017)), s.card = 11 → (s.sum (λ i => seq i) > 0)

/-- The sum of all 2017 numbers is divisible by 9 -/
def SumAllDivisibleByNine (seq : SpecialSequence) : Prop :=
  (Finset.univ.sum seq) % 9 = 0

/-- The sequence consists of five -1's and 2012 1's -/
def IsFiveNegativeOnesRestOnes (seq : SpecialSequence) : Prop :=
  (Finset.filter (λ i => seq i = -1) Finset.univ).card = 5 ∧
  (Finset.filter (λ i => seq i = 1) Finset.univ).card = 2012

theorem special_sequence_theorem (seq : SpecialSequence) :
  SumSquaresSevenIsSeven seq →
  SumElevenIsPositive seq →
  SumAllDivisibleByNine seq →
  IsFiveNegativeOnesRestOnes seq :=
by sorry

end special_sequence_theorem_l1880_188010


namespace largest_s_value_l1880_188077

/-- The largest possible value of s for regular polygons P₁ (r-gon) and P₂ (s-gon) 
    satisfying the given conditions -/
theorem largest_s_value : ℕ := by
  /- Define the interior angle of a regular n-gon -/
  let interior_angle (n : ℕ) : ℚ := (n - 2 : ℚ) * 180 / n

  /- Define the relationship between r and s based on the interior angle ratio -/
  let r_s_relation (r s : ℕ) : Prop :=
    interior_angle r / interior_angle s = 29 / 28

  /- Define the conditions on r and s -/
  let valid_r_s (r s : ℕ) : Prop :=
    r ≥ s ∧ s ≥ 3 ∧ r_s_relation r s

  /- The theorem states that 114 is the largest value of s satisfying all conditions -/
  have h1 : ∃ (r : ℕ), valid_r_s r 114 := sorry
  have h2 : ∀ (s : ℕ), s > 114 → ¬∃ (r : ℕ), valid_r_s r s := sorry

  exact 114

end largest_s_value_l1880_188077


namespace remaining_sweet_cookies_l1880_188083

def initial_sweet_cookies : ℕ := 22
def eaten_sweet_cookies : ℕ := 15

theorem remaining_sweet_cookies :
  initial_sweet_cookies - eaten_sweet_cookies = 7 :=
by sorry

end remaining_sweet_cookies_l1880_188083


namespace ellipse_properties_l1880_188060

noncomputable def ellipse_C (x y : ℝ) (a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

def right_focus : ℝ × ℝ := (1, 0)

theorem ellipse_properties (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  ellipse_C (-1) (Real.sqrt 2 / 2) a b →
  (∀ x y : ℝ, ellipse_C x y a b ↔ x^2 / 2 + y^2 = 1) ∧
  (∃ Q : ℝ × ℝ, Q.1 = 5/4 ∧ Q.2 = 0 ∧
    ∀ A B : ℝ × ℝ, 
      ellipse_C A.1 A.2 a b →
      ellipse_C B.1 B.2 a b →
      (∃ t : ℝ, A.1 = t * A.2 + 1 ∧ B.1 = t * B.2 + 1) →
      ((A.1 - Q.1) * (B.1 - Q.1) + (A.2 - Q.2) * (B.2 - Q.2) = -7/16)) :=
by sorry

end ellipse_properties_l1880_188060


namespace zero_subset_A_l1880_188065

def A : Set ℝ := {x | x > -3}

theorem zero_subset_A : {0} ⊆ A := by
  sorry

end zero_subset_A_l1880_188065


namespace largest_digit_sum_l1880_188081

theorem largest_digit_sum (a b c : ℕ) (y : ℕ) : 
  (a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9) →
  (0 < y ∧ y ≤ 7) →
  (100 * a + 10 * b + c : ℚ) / 900 = 1 / y →
  (∀ a' b' c' : ℕ, 
    a' ≤ 9 ∧ b' ≤ 9 ∧ c' ≤ 9 →
    (∃ y' : ℕ, 0 < y' ∧ y' ≤ 7 ∧ (100 * a' + 10 * b' + c' : ℚ) / 900 = 1 / y') →
    a + b + c ≥ a' + b' + c') →
  a + b + c = 9 := by
sorry

end largest_digit_sum_l1880_188081


namespace exp_13pi_div_2_l1880_188033

/-- Euler's formula -/
axiom euler_formula (θ : ℝ) : Complex.exp (θ * Complex.I) = Complex.cos θ + Complex.I * Complex.sin θ

/-- The main theorem -/
theorem exp_13pi_div_2 : Complex.exp ((13 * Real.pi / 2) * Complex.I) = Complex.I := by
  sorry

end exp_13pi_div_2_l1880_188033


namespace quadratic_equation_properties_sum_of_roots_when_k_is_nine_l1880_188090

theorem quadratic_equation_properties (k : ℝ) :
  let equation := fun x => k * x^2 - 6 * x + 1
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ equation x₁ = 0 ∧ equation x₂ = 0) ↔ (0 < k ∧ k ≤ 9) :=
by sorry

theorem sum_of_roots_when_k_is_nine :
  let equation := fun x => 9 * x^2 - 6 * x + 1
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ equation x₁ = 0 ∧ equation x₂ = 0 ∧ x₁ + x₂ = 2/3 :=
by sorry

end quadratic_equation_properties_sum_of_roots_when_k_is_nine_l1880_188090


namespace log_sum_simplification_l1880_188069

theorem log_sum_simplification :
  1 / (Real.log 3 / Real.log 12 + 1) +
  1 / (Real.log 2 / Real.log 8 + 1) +
  1 / (Real.log 3 / Real.log 9 + 1) = 2 := by
  sorry

end log_sum_simplification_l1880_188069


namespace unique_solution_for_all_real_b_l1880_188046

/-- The equation has exactly one real solution in x for all real b, 
    unless b forces other roots or violates the unimodality of solution. -/
theorem unique_solution_for_all_real_b : 
  ∀ b : ℝ, ∃! x : ℝ, x^3 - b*x^2 - 3*b*x + b^2 - 2 = 0 :=
sorry

end unique_solution_for_all_real_b_l1880_188046


namespace expand_expression_l1880_188071

theorem expand_expression (x : ℝ) : (17 * x^2 + 20) * 3 * x^3 = 51 * x^5 + 60 * x^3 := by
  sorry

end expand_expression_l1880_188071


namespace number_of_true_propositions_l1880_188002

-- Define a type for propositions about polygons
inductive PolygonProposition
  | equalSidesRegular
  | regularCentrallySymmetric
  | hexagonRadiusEqualsSide
  | regularNgonNAxes

-- Function to evaluate the truth of each proposition
def isTrue (p : PolygonProposition) : Bool :=
  match p with
  | PolygonProposition.equalSidesRegular => false
  | PolygonProposition.regularCentrallySymmetric => false
  | PolygonProposition.hexagonRadiusEqualsSide => true
  | PolygonProposition.regularNgonNAxes => true

-- List of all propositions
def allPropositions : List PolygonProposition :=
  [PolygonProposition.equalSidesRegular,
   PolygonProposition.regularCentrallySymmetric,
   PolygonProposition.hexagonRadiusEqualsSide,
   PolygonProposition.regularNgonNAxes]

-- Theorem stating that the number of true propositions is 2
theorem number_of_true_propositions :
  (allPropositions.filter isTrue).length = 2 := by
  sorry

end number_of_true_propositions_l1880_188002


namespace ellipse_k_values_l1880_188038

-- Define the eccentricity
def eccentricity : ℝ := 6

-- Define the ellipse equation
def is_on_ellipse (x y k : ℝ) : Prop :=
  x^2 / 20 + y^2 / k = 1

-- Define the relationship between eccentricity, semi-major axis, and semi-minor axis
def eccentricity_relation (a b : ℝ) : Prop :=
  eccentricity^2 = 1 - (b^2 / a^2)

-- Theorem statement
theorem ellipse_k_values :
  ∃ k : ℝ, (k = 11 ∨ k = 29) ∧
  (∀ x y : ℝ, is_on_ellipse x y k) ∧
  ((eccentricity_relation 20 k ∧ 20 > k) ∨ (eccentricity_relation k 20 ∧ k > 20)) :=
sorry

end ellipse_k_values_l1880_188038


namespace sales_second_month_l1880_188011

def sales_problem (X : ℕ) : Prop :=
  let sales : List ℕ := [2435, X, 2855, 3230, 2560, 1000]
  (sales.sum / sales.length = 2500) ∧ 
  (sales.length = 6)

theorem sales_second_month : 
  ∃ (X : ℕ), sales_problem X ∧ X = 2920 := by
sorry

end sales_second_month_l1880_188011


namespace point_on_modified_graph_l1880_188047

/-- Given a function g : ℝ → ℝ where (3, 9) is on its graph,
    prove that (1, 10) is on the graph of 3y = 4g(3x) - 6 -/
theorem point_on_modified_graph (g : ℝ → ℝ) (h : g 3 = 9) :
  3 * 10 = 4 * g (3 * 1) - 6 := by
  sorry

end point_on_modified_graph_l1880_188047


namespace aquafaba_needed_is_32_l1880_188036

/-- The number of tablespoons of aquafaba equivalent to one egg white -/
def aquafaba_per_egg : ℕ := 2

/-- The number of egg whites required for one angel food cake -/
def egg_whites_per_cake : ℕ := 8

/-- The number of angel food cakes Christine is making -/
def number_of_cakes : ℕ := 2

/-- The total number of tablespoons of aquafaba needed for the cakes -/
def total_aquafaba_needed : ℕ := aquafaba_per_egg * egg_whites_per_cake * number_of_cakes

theorem aquafaba_needed_is_32 : total_aquafaba_needed = 32 := by
  sorry

end aquafaba_needed_is_32_l1880_188036


namespace ratio_equality_l1880_188028

theorem ratio_equality (a b c : ℝ) (h : a / 2 = b / 3 ∧ b / 3 = c / 4 ∧ c ≠ 0) :
  (a + b) / c = 5 / 4 := by
  sorry

end ratio_equality_l1880_188028


namespace taco_castle_parking_lot_l1880_188030

/-- The number of Dodge trucks in the Taco Castle parking lot -/
def dodge_trucks : ℕ := 60

/-- The number of Ford trucks in the Taco Castle parking lot -/
def ford_trucks : ℕ := dodge_trucks / 3

/-- The number of Toyota trucks in the Taco Castle parking lot -/
def toyota_trucks : ℕ := ford_trucks / 2

/-- The number of Volkswagen Bugs in the Taco Castle parking lot -/
def volkswagen_bugs : ℕ := 5

theorem taco_castle_parking_lot :
  dodge_trucks = 60 ∧
  ford_trucks = dodge_trucks / 3 ∧
  ford_trucks = toyota_trucks * 2 ∧
  volkswagen_bugs = toyota_trucks / 2 ∧
  volkswagen_bugs = 5 :=
by sorry

end taco_castle_parking_lot_l1880_188030


namespace f_max_min_on_interval_l1880_188014

def f (x : ℝ) : ℝ := x^3 - 3*x

theorem f_max_min_on_interval :
  let a := -3
  let b := 3
  ∃ (x_max x_min : ℝ), a ≤ x_max ∧ x_max ≤ b ∧ a ≤ x_min ∧ x_min ≤ b ∧
    (∀ x, a ≤ x ∧ x ≤ b → f x ≤ f x_max) ∧
    (∀ x, a ≤ x ∧ x ≤ b → f x_min ≤ f x) ∧
    f x_max = 18 ∧ f x_min = -18 :=
by sorry

end f_max_min_on_interval_l1880_188014


namespace stock_price_calculation_l1880_188056

def initial_price : ℝ := 120
def first_year_increase : ℝ := 0.8
def second_year_decrease : ℝ := 0.3

theorem stock_price_calculation :
  let price_after_first_year := initial_price * (1 + first_year_increase)
  let final_price := price_after_first_year * (1 - second_year_decrease)
  final_price = 151.2 := by
  sorry

end stock_price_calculation_l1880_188056


namespace smallest_n_value_l1880_188063

def count_factors_of_five (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625)

theorem smallest_n_value (a b c m n : ℕ) : 
  a > 0 → b > 0 → c > 0 →
  a + b + c = 2023 →
  11 ∣ a →
  a * b * c ≠ 0 →
  ∃ (m : ℕ), m % 10 ≠ 0 ∧ a.factorial * b.factorial * c.factorial = m * (10 ^ n) →
  (∀ k, k < n → ¬∃ (l : ℕ), l % 10 ≠ 0 ∧ a.factorial * b.factorial * c.factorial = l * (10 ^ k)) →
  n = 497 :=
sorry

end smallest_n_value_l1880_188063


namespace equal_area_point_sum_l1880_188003

def P : ℝ × ℝ := (-4, 3)
def Q : ℝ × ℝ := (7, -5)
def R : ℝ × ℝ := (0, 6)

def triangle_area (A B C : ℝ × ℝ) : ℝ := sorry

theorem equal_area_point_sum (S : ℝ × ℝ) :
  S.1 > (min P.1 (min Q.1 R.1)) ∧ 
  S.1 < (max P.1 (max Q.1 R.1)) ∧ 
  S.2 > (min P.2 (min Q.2 R.2)) ∧ 
  S.2 < (max P.2 (max Q.2 R.2)) ∧
  triangle_area P Q S = triangle_area Q R S ∧ 
  triangle_area Q R S = triangle_area R P S →
  10 * S.1 + S.2 = 34/3 := by sorry

end equal_area_point_sum_l1880_188003


namespace european_passenger_fraction_l1880_188073

theorem european_passenger_fraction (total : ℕ) 
  (north_america : ℚ) (africa : ℚ) (asia : ℚ) (other : ℕ) :
  total = 108 →
  north_america = 1 / 12 →
  africa = 1 / 9 →
  asia = 1 / 6 →
  other = 42 →
  (total : ℚ) - (north_america * total + africa * total + asia * total + other) = 1 / 4 * total :=
by sorry

end european_passenger_fraction_l1880_188073


namespace min_value_sum_l1880_188020

theorem min_value_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 1 → 
    (a + 1/a) + (b + 1/b) ≤ (x + 1/x) + (y + 1/y) ∧
    (a + 1/a) + (b + 1/b) = 5 :=
by sorry

end min_value_sum_l1880_188020


namespace equation_solutions_l1880_188068

theorem equation_solutions : 
  ∀ x y : ℤ, (3 : ℚ) / (x - 1) = (5 : ℚ) / (y - 2) ↔ (x = 1 ∧ y = 2) ∨ (x = 4 ∧ y = 7) :=
by sorry

end equation_solutions_l1880_188068


namespace equal_area_division_ratio_l1880_188098

/-- Represents a T-shaped figure composed of unit squares -/
structure TShape :=
  (squares : ℕ)
  (is_t_shaped : squares = 22)

/-- Represents a line passing through a point in the T-shaped figure -/
structure DividingLine :=
  (t_shape : TShape)
  (divides_equally : Bool)

/-- Represents a segment in the T-shaped figure -/
structure Segment :=
  (t_shape : TShape)
  (length : ℚ)

/-- Theorem stating that if a line divides a T-shaped figure into equal areas,
    it divides a certain segment in the ratio 3:1 -/
theorem equal_area_division_ratio 
  (t : TShape) 
  (l : DividingLine) 
  (s : Segment) 
  (h1 : l.t_shape = t) 
  (h2 : s.t_shape = t) 
  (h3 : l.divides_equally = true) :
  s.length * (1/4) = 1 ∧ s.length * (3/4) = 3 :=
sorry

end equal_area_division_ratio_l1880_188098


namespace prime_square_mod_30_l1880_188043

theorem prime_square_mod_30 (p : ℕ) (hp : Nat.Prime p) (hp_gt_5 : p > 5) :
  p^2 % 30 = 1 ∨ p^2 % 30 = 19 := by
  sorry

end prime_square_mod_30_l1880_188043


namespace min_product_under_constraints_l1880_188055

theorem min_product_under_constraints (x y w : ℝ) : 
  x > 0 → y > 0 → w > 0 →
  x + y + w = 1 →
  x ≤ 2*y ∧ x ≤ 2*w ∧ y ≤ 2*x ∧ y ≤ 2*w ∧ w ≤ 2*x ∧ w ≤ 2*y →
  x*y*w ≥ 2*(2*Real.sqrt 3 - 3)/27 := by
sorry

end min_product_under_constraints_l1880_188055


namespace equality_check_l1880_188000

theorem equality_check : 
  (-3^2 ≠ -2^3) ∧ 
  (-6^3 = (-6)^3) ∧ 
  (-6^2 ≠ (-6)^2) ∧ 
  ((-3 * 2)^2 ≠ (-3) * 2^2) :=
by
  sorry

end equality_check_l1880_188000


namespace sin_double_angle_sum_l1880_188013

open Real

theorem sin_double_angle_sum (θ : ℝ) (h : ∑' n, sin θ ^ (2 * n) = 3) : 
  sin (2 * θ) = 2 * sqrt 2 / 3 := by
  sorry

end sin_double_angle_sum_l1880_188013


namespace power_of_five_division_l1880_188049

theorem power_of_five_division : (5 ^ 12) / (25 ^ 3) = 15625 := by sorry

end power_of_five_division_l1880_188049


namespace train_stop_time_l1880_188001

/-- Proves that a train with given speeds stops for 20 minutes per hour -/
theorem train_stop_time (speed_without_stops speed_with_stops : ℝ) 
  (h1 : speed_without_stops = 45)
  (h2 : speed_with_stops = 30) : 
  (speed_without_stops - speed_with_stops) / speed_without_stops * 60 = 20 := by
  sorry

#check train_stop_time

end train_stop_time_l1880_188001


namespace two_successive_discounts_l1880_188064

theorem two_successive_discounts (list_price : ℝ) (final_price : ℝ) (first_discount : ℝ) (second_discount : ℝ) :
  list_price = 70 →
  final_price = 59.22 →
  first_discount = 10 →
  (list_price - (first_discount / 100) * list_price) * (1 - second_discount / 100) = final_price →
  second_discount = 6 := by
sorry

end two_successive_discounts_l1880_188064


namespace system_of_equations_sum_l1880_188093

theorem system_of_equations_sum (x y z : ℝ) 
  (eq1 : y + z = 20 - 4*x)
  (eq2 : x + z = -18 - 4*y)
  (eq3 : x + y = 10 - 4*z) :
  2*x + 2*y + 2*z = 4 := by
sorry

end system_of_equations_sum_l1880_188093


namespace tangent_line_b_value_l1880_188058

-- Define the curve
def f (x a : ℝ) : ℝ := x^3 + a*x + 1

-- Define the tangent line
def tangent_line (k b x : ℝ) : ℝ := k*x + b

-- Theorem statement
theorem tangent_line_b_value :
  ∀ (a k b : ℝ),
  (f 2 a = 3) →  -- The curve passes through (2, 3)
  (tangent_line k b 2 = 3) →  -- The tangent line passes through (2, 3)
  (∀ x, tangent_line k b x = k*x + b) →  -- Definition of tangent line
  (∀ x, f x a = x^3 + a*x + 1) →  -- Definition of the curve
  (k = (3*2^2 + a)) →  -- Slope of tangent is derivative at x = 2
  (b = -15) :=
by
  sorry

end tangent_line_b_value_l1880_188058


namespace unique_intersection_l1880_188053

/-- The value of m for which the line x = m intersects the parabola x = -3y^2 - 4y + 7 at exactly one point -/
def m : ℚ := 25/3

/-- The parabola equation -/
def parabola (y : ℝ) : ℝ := -3 * y^2 - 4 * y + 7

theorem unique_intersection :
  ∃! y, parabola y = m :=
sorry

end unique_intersection_l1880_188053


namespace kellys_initial_games_prove_kellys_initial_games_l1880_188004

/-- Theorem: Kelly's initial number of Nintendo games
Given that Kelly gives away 250 Nintendo games and has 300 games left,
prove that she initially had 550 games. -/
theorem kellys_initial_games : ℕ → Prop :=
  fun initial_games =>
    let games_given_away : ℕ := 250
    let games_left : ℕ := 300
    initial_games = games_given_away + games_left ∧ initial_games = 550

/-- Proof of Kelly's initial number of Nintendo games -/
theorem prove_kellys_initial_games : ∃ (initial_games : ℕ), kellys_initial_games initial_games :=
  sorry

end kellys_initial_games_prove_kellys_initial_games_l1880_188004


namespace square_less_than_triple_l1880_188041

theorem square_less_than_triple (x : ℤ) : x^2 < 3*x ↔ x = 1 ∨ x = 2 := by
  sorry

end square_less_than_triple_l1880_188041


namespace line_intersects_circle_l1880_188089

/-- Theorem: A line intersects a circle if the point defining the line is outside the circle -/
theorem line_intersects_circle (x₀ y₀ R : ℝ) (h : x₀^2 + y₀^2 > R^2) :
  ∃ (x y : ℝ), x^2 + y^2 = R^2 ∧ x₀*x + y₀*y = R^2 := by
  sorry

#check line_intersects_circle

end line_intersects_circle_l1880_188089


namespace box_packing_problem_l1880_188096

theorem box_packing_problem (n : ℕ) (h : n = 301) :
  (n % 3 = 1 ∧ n % 4 = 1 ∧ n % 5 = 1 ∧ n % 6 = 1) ∧
  (∃ k : ℕ, k ≠ 1 ∧ k ≠ n ∧ n % k = 0) :=
by sorry

end box_packing_problem_l1880_188096


namespace condition_t_necessary_not_sufficient_l1880_188050

theorem condition_t_necessary_not_sufficient (x y : ℝ) :
  (∀ x y, (x + y ≤ 28 ∨ x * y ≤ 192) → (x ≤ 12 ∨ y ≤ 16)) ∧
  (∃ x y, (x ≤ 12 ∨ y ≤ 16) ∧ ¬(x + y ≤ 28 ∨ x * y ≤ 192)) :=
by sorry

end condition_t_necessary_not_sufficient_l1880_188050


namespace inequality_system_solution_l1880_188084

theorem inequality_system_solution (x : ℝ) (hx : x ≠ 0) :
  (abs (2 * x - 3) ≤ 3 ∧ 1 / x < 1) ↔ (1 < x ∧ x ≤ 3) :=
by sorry

end inequality_system_solution_l1880_188084


namespace middle_aged_employees_selected_l1880_188070

/-- Represents the age group of an employee -/
inductive AgeGroup
  | Elderly
  | MiddleAged
  | Young

/-- Represents the company with its employee distribution -/
structure Company where
  total_employees : ℕ
  age_ratio : Fin 3 → ℕ
  age_ratio_sum : age_ratio 0 + age_ratio 1 + age_ratio 2 > 0

/-- Calculates the number of employees in a specific age group -/
def employees_in_group (c : Company) (g : Fin 3) : ℕ :=
  c.total_employees * c.age_ratio g / (c.age_ratio 0 + c.age_ratio 1 + c.age_ratio 2)

/-- Theorem: The number of middle-aged employees selected in stratified sampling -/
theorem middle_aged_employees_selected
  (c : Company)
  (h_total : c.total_employees = 1200)
  (h_ratio : c.age_ratio = ![1, 5, 6])
  (sample_size : ℕ)
  (h_sample : sample_size = 36) :
  employees_in_group c 1 * sample_size / c.total_employees = 15 := by
  sorry

end middle_aged_employees_selected_l1880_188070


namespace ripe_orange_harvest_l1880_188006

/-- The number of days of harvest -/
def harvest_days : ℕ := 73

/-- The number of sacks of ripe oranges harvested per day -/
def daily_ripe_harvest : ℕ := 5

/-- The total number of sacks of ripe oranges harvested over the entire period -/
def total_ripe_harvest : ℕ := harvest_days * daily_ripe_harvest

theorem ripe_orange_harvest :
  total_ripe_harvest = 365 := by sorry

end ripe_orange_harvest_l1880_188006


namespace algebraic_expressions_equality_l1880_188009

theorem algebraic_expressions_equality (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) :
  (((a^2 * b) / (-c))^3 * ((c^2) / (-a * b))^2) / ((b * c / a)^4) = -a^10 / (b^3 * c^7) ∧
  ((2 / (a^2 - b^2) - 1 / (a^2 - a * b)) / (a / (a + b))) = 1 / a^2 := by
sorry

end algebraic_expressions_equality_l1880_188009


namespace sunday_temp_is_98_1_l1880_188099

/-- Given a list of 6 temperatures and a weekly average for 7 days, 
    calculate the 7th temperature (Sunday) -/
def calculate_sunday_temp (temps : List Float) (weekly_avg : Float) : Float :=
  7 * weekly_avg - temps.sum

/-- Theorem stating that given the specific temperatures and weekly average,
    the Sunday temperature is 98.1 -/
theorem sunday_temp_is_98_1 : 
  let temps := [98.2, 98.7, 99.3, 99.8, 99, 98.9]
  let weekly_avg := 99
  calculate_sunday_temp temps weekly_avg = 98.1 := by
  sorry

#eval calculate_sunday_temp [98.2, 98.7, 99.3, 99.8, 99, 98.9] 99

end sunday_temp_is_98_1_l1880_188099


namespace cos_180_degrees_l1880_188087

theorem cos_180_degrees : Real.cos (π) = -1 := by
  sorry

end cos_180_degrees_l1880_188087


namespace pears_theorem_l1880_188035

def pears_problem (keith_picked mike_picked sarah_picked keith_gave mike_gave sarah_gave : ℕ) : Prop :=
  let keith_left := keith_picked - keith_gave
  let mike_left := mike_picked - mike_gave
  let sarah_left := sarah_picked - sarah_gave
  keith_left + mike_left + sarah_left = 15

theorem pears_theorem :
  pears_problem 47 12 22 46 5 15 := by sorry

end pears_theorem_l1880_188035


namespace constant_sum_list_difference_l1880_188023

/-- A list of four real numbers where the sum of any two adjacent numbers is constant -/
structure ConstantSumList (a b c d : ℝ) : Prop where
  first_pair : a + b = b + c
  second_pair : b + c = c + d

/-- Theorem: In a list [2, x, y, 5] where the sum of any two adjacent numbers is constant, x - y = 3 -/
theorem constant_sum_list_difference (x y : ℝ) (h : ConstantSumList 2 x y 5) : x - y = 3 := by
  sorry

end constant_sum_list_difference_l1880_188023


namespace jeans_cost_l1880_188091

theorem jeans_cost (shirt_cost hat_cost total_cost : ℚ) 
  (h1 : shirt_cost = 5)
  (h2 : hat_cost = 4)
  (h3 : total_cost = 51)
  (h4 : 3 * shirt_cost + 2 * (total_cost - 3 * shirt_cost - 4 * hat_cost) / 2 + 4 * hat_cost = total_cost) :
  (total_cost - 3 * shirt_cost - 4 * hat_cost) / 2 = 10 := by
  sorry

end jeans_cost_l1880_188091


namespace bunnies_given_away_l1880_188034

theorem bunnies_given_away (initial_bunnies : ℕ) (kittens_per_bunny : ℕ) (final_total : ℕ) :
  initial_bunnies = 30 →
  kittens_per_bunny = 2 →
  final_total = 54 →
  (initial_bunnies - (final_total - initial_bunnies) / kittens_per_bunny) / initial_bunnies = 3 / 5 :=
by sorry

end bunnies_given_away_l1880_188034


namespace power_of_three_plus_five_mod_five_l1880_188029

theorem power_of_three_plus_five_mod_five : 
  (3^100 + 5) % 5 = 1 := by sorry

end power_of_three_plus_five_mod_five_l1880_188029


namespace complex_equation_solution_l1880_188082

theorem complex_equation_solution (z : ℂ) : z * (2 - Complex.I) = 11 + 7 * Complex.I → z = 3 + 5 * Complex.I := by
  sorry

end complex_equation_solution_l1880_188082


namespace sequence_and_sum_formula_l1880_188062

def sequence_a (n : ℕ) : ℚ := (3^n - 1) / 2

def S (n : ℕ) : ℚ := (3^(n+2) - 9) / 8 - n * (n+4) / 4

theorem sequence_and_sum_formula :
  (∀ n : ℕ, ∃ q : ℚ, sequence_a (n+1) + 1/2 = q * (sequence_a n + 1/2)) ∧ 
  (sequence_a 1 + 1/2 = 3/2) ∧
  (sequence_a 4 - sequence_a 1 = 39) →
  (∀ n : ℕ, sequence_a n = (3^n - 1) / 2) ∧
  (∀ n : ℕ, S n = (3^(n+2) - 9) / 8 - n * (n+4) / 4) :=
by sorry

end sequence_and_sum_formula_l1880_188062


namespace modulo_equivalence_l1880_188051

theorem modulo_equivalence (n : ℤ) : 0 ≤ n ∧ n ≤ 8 ∧ n ≡ -4981 [ZMOD 9] → n = 5 := by
  sorry

end modulo_equivalence_l1880_188051


namespace remainder_of_29_times_182_power_1000_mod_13_l1880_188017

theorem remainder_of_29_times_182_power_1000_mod_13 : 
  (29 * 182^1000) % 13 = 0 := by
sorry

end remainder_of_29_times_182_power_1000_mod_13_l1880_188017


namespace inequality_of_squares_l1880_188075

theorem inequality_of_squares (x : ℝ) : (x - 1)^2 ≠ x^2 - 1 := by
  sorry

end inequality_of_squares_l1880_188075


namespace election_win_margin_l1880_188027

theorem election_win_margin :
  ∀ (total_votes : ℕ) (winner_votes loser_votes : ℕ),
    (winner_votes : ℚ) / total_votes = 3/5 →
    winner_votes = 720 →
    total_votes = winner_votes + loser_votes →
    winner_votes - loser_votes = 240 := by
  sorry

end election_win_margin_l1880_188027


namespace min_squares_cover_l1880_188044

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Represents a square with integer side length -/
structure Square where
  side : ℕ

/-- Function to calculate the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℕ :=
  r.length * r.width

/-- Function to calculate the area of a square -/
def squareArea (s : Square) : ℕ :=
  s.side * s.side

/-- Function to check if a list of squares can cover a rectangle -/
def canCover (r : Rectangle) (squares : List Square) : Prop :=
  rectangleArea r = (squares.map squareArea).sum

/-- The main theorem to be proved -/
theorem min_squares_cover (r : Rectangle) (squares : List Square) :
  r.length = 10 ∧ r.width = 9 ∧ 
  (∀ s ∈ squares, s.side > 0) ∧
  canCover r squares →
  squares.length ≥ 10 :=
sorry

end min_squares_cover_l1880_188044


namespace equation_solution_l1880_188097

theorem equation_solution :
  ∃ x : ℝ, x > 0 ∧ 6 * x^(1/3) - 3 * (x / x^(2/3)) = 10 + 2 * x^(1/3) ∧ x = 1000 :=
by
  sorry

end equation_solution_l1880_188097


namespace polygon_properties_l1880_188015

theorem polygon_properties (n : ℕ) (internal_angle : ℝ) (external_angle : ℝ) :
  (internal_angle + external_angle = 180) →
  (external_angle = (2/3) * internal_angle) →
  (360 / external_angle = n) →
  (n > 2) →
  (n = 5 ∧ (n - 2) * 180 = 540) :=
by sorry

end polygon_properties_l1880_188015


namespace range_of_a_l1880_188052

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 0 1 ∧ (2:ℝ)^x * (3*x + a) < 1) → a < 1 := by
  sorry

end range_of_a_l1880_188052


namespace two_digit_number_condition_l1880_188088

/-- A two-digit number satisfies the given condition if and only if its ones digit is 9 -/
theorem two_digit_number_condition (a b : ℕ) (h1 : a > 0) (h2 : a < 10) (h3 : b < 10) :
  (10 * a + b) - (a * b) = a + b ↔ b = 9 :=
by sorry

end two_digit_number_condition_l1880_188088


namespace f_satisfies_conditions_l1880_188061

def P : Set (ℕ × ℕ) := Set.univ

def f : ℕ → ℕ → ℝ
  | p, q => p * q

theorem f_satisfies_conditions :
  (∀ p q : ℕ, p * q = 0 → f p q = 0) ∧
  (∀ p q : ℕ, p * q ≠ 0 → f p q = 1 + 1/2 * f (p+1) (q-1) + 1/2 * f (p-1) (q+1)) :=
by sorry

end f_satisfies_conditions_l1880_188061
