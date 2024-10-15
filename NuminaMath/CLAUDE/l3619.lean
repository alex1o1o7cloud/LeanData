import Mathlib

namespace NUMINAMATH_CALUDE_completing_square_result_l3619_361926

theorem completing_square_result (x : ℝ) :
  x^2 - 6*x + 4 = 0 ↔ (x - 3)^2 = 5 :=
sorry

end NUMINAMATH_CALUDE_completing_square_result_l3619_361926


namespace NUMINAMATH_CALUDE_same_color_eyes_percentage_l3619_361943

/-- Represents the proportion of students with a specific eye color combination -/
structure EyeColorProportion where
  eggCream : ℝ    -- proportion of students with eggshell and cream eyes
  eggCorn : ℝ     -- proportion of students with eggshell and cornsilk eyes
  eggEgg : ℝ      -- proportion of students with both eggshell eyes
  creamCorn : ℝ   -- proportion of students with cream and cornsilk eyes
  creamCream : ℝ  -- proportion of students with both cream eyes
  cornCorn : ℝ    -- proportion of students with both cornsilk eyes

/-- The conditions given in the problem -/
def eyeColorConditions (p : EyeColorProportion) : Prop :=
  p.eggCream + p.eggCorn + p.eggEgg = 0.3 ∧
  p.eggCream + p.creamCorn + p.creamCream = 0.4 ∧
  p.eggCorn + p.creamCorn + p.cornCorn = 0.5 ∧
  p.eggCream + p.eggCorn + p.eggEgg + p.creamCorn + p.creamCream + p.cornCorn = 1

/-- The theorem to be proved -/
theorem same_color_eyes_percentage (p : EyeColorProportion) 
  (h : eyeColorConditions p) : 
  p.eggEgg + p.creamCream + p.cornCorn = 0.8 := by
  sorry


end NUMINAMATH_CALUDE_same_color_eyes_percentage_l3619_361943


namespace NUMINAMATH_CALUDE_power_five_addition_l3619_361911

theorem power_five_addition (a : ℝ) : a^5 + a^5 = 2*a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_five_addition_l3619_361911


namespace NUMINAMATH_CALUDE_chalkboard_area_l3619_361983

/-- A rectangular chalkboard with a width of 3 feet and a length that is 2 times its width has an area of 18 square feet. -/
theorem chalkboard_area : 
  ∀ (width length area : ℝ), 
  width = 3 → 
  length = 2 * width → 
  area = width * length → 
  area = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_chalkboard_area_l3619_361983


namespace NUMINAMATH_CALUDE_roots_properties_l3619_361959

-- Define the coefficients of the quadratic equation
def a : ℝ := 24
def b : ℝ := 60
def c : ℝ := -600

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- Theorem statement
theorem roots_properties :
  (∃ x y : ℝ, x ≠ y ∧ quadratic_equation x ∧ quadratic_equation y) →
  (∃ x y : ℝ, x ≠ y ∧ quadratic_equation x ∧ quadratic_equation y ∧ x * y = -25 ∧ x + y = -2.5) :=
sorry

end NUMINAMATH_CALUDE_roots_properties_l3619_361959


namespace NUMINAMATH_CALUDE_new_person_weight_l3619_361955

theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) : 
  initial_count = 8 → 
  weight_increase = 2.5 → 
  replaced_weight = 55 → 
  (initial_count : ℝ) * weight_increase + replaced_weight = 75 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l3619_361955


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l3619_361995

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

/-- Given a geometric sequence {aₙ} where a₂ = 4 and a₆a₇ = 16a₉, prove that a₅ = ±32. -/
theorem geometric_sequence_problem (a : ℕ → ℝ) 
    (h_geo : IsGeometricSequence a) 
    (h_a2 : a 2 = 4)
    (h_a6a7 : a 6 * a 7 = 16 * a 9) : 
  a 5 = 32 ∨ a 5 = -32 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l3619_361995


namespace NUMINAMATH_CALUDE_flower_shop_sales_ratio_l3619_361915

/-- Proves that the ratio of Tuesday's sales to Monday's sales is 3:1 given the conditions of the flower shop's three-day sale. -/
theorem flower_shop_sales_ratio : 
  ∀ (tuesday_sales : ℕ),
  12 + tuesday_sales + tuesday_sales / 3 = 60 →
  tuesday_sales / 12 = 3 := by
  sorry

end NUMINAMATH_CALUDE_flower_shop_sales_ratio_l3619_361915


namespace NUMINAMATH_CALUDE_optimal_transportation_plan_l3619_361920

structure VehicleType where
  capacity : ℕ
  cost : ℕ

def total_supply : ℕ := 120

def vehicle_a : VehicleType := ⟨5, 300⟩
def vehicle_b : VehicleType := ⟨8, 400⟩
def vehicle_c : VehicleType := ⟨10, 500⟩

def total_vehicles : ℕ := 18

def transportation_plan (a b c : ℕ) : Prop :=
  a + b + c = total_vehicles ∧
  a * vehicle_a.capacity + b * vehicle_b.capacity + c * vehicle_c.capacity ≥ total_supply

def transportation_cost (a b c : ℕ) : ℕ :=
  a * vehicle_a.cost + b * vehicle_b.cost + c * vehicle_c.cost

theorem optimal_transportation_plan :
  ∀ (a b c : ℕ),
    transportation_plan a b c →
    transportation_cost a b c ≥ transportation_cost 8 10 0 :=
by sorry

end NUMINAMATH_CALUDE_optimal_transportation_plan_l3619_361920


namespace NUMINAMATH_CALUDE_total_troll_count_l3619_361971

/-- The number of trolls Erin counted in different locations -/
structure TrollCount where
  forest : ℕ
  bridge : ℕ
  plains : ℕ

/-- The conditions given in the problem -/
def troll_conditions (t : TrollCount) : Prop :=
  t.forest = 6 ∧
  t.bridge = 4 * t.forest - 6 ∧
  t.plains = t.bridge / 2

/-- The theorem stating the total number of trolls Erin counted -/
theorem total_troll_count (t : TrollCount) (h : troll_conditions t) : 
  t.forest + t.bridge + t.plains = 33 := by
  sorry


end NUMINAMATH_CALUDE_total_troll_count_l3619_361971


namespace NUMINAMATH_CALUDE_fraction_equality_l3619_361916

theorem fraction_equality (a b c d : ℝ) 
  (h : (a - b) * (c - d) / ((b - c) * (d - a)) = 3 / 4) : 
  (a - c) * (b - d) / ((a - b) * (c - d)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3619_361916


namespace NUMINAMATH_CALUDE_sale_total_cost_l3619_361998

/-- Calculates the total cost of ice cream and juice during a sale. -/
def calculate_total_cost (original_ice_cream_price : ℚ) 
                         (ice_cream_discount : ℚ) 
                         (juice_price : ℚ) 
                         (juice_cans_per_price : ℕ) 
                         (ice_cream_tubs : ℕ) 
                         (juice_cans : ℕ) : ℚ :=
  let sale_ice_cream_price := original_ice_cream_price - ice_cream_discount
  let ice_cream_cost := sale_ice_cream_price * ice_cream_tubs
  let juice_cost := (juice_price / juice_cans_per_price) * juice_cans
  ice_cream_cost + juice_cost

/-- Theorem stating that the total cost is $24 for the given conditions. -/
theorem sale_total_cost : 
  calculate_total_cost 12 2 2 5 2 10 = 24 := by
  sorry

end NUMINAMATH_CALUDE_sale_total_cost_l3619_361998


namespace NUMINAMATH_CALUDE_mans_rowing_speed_l3619_361969

/-- Given a man's speed with and against a stream, calculate his rowing speed in still water. -/
theorem mans_rowing_speed (speed_with_stream speed_against_stream : ℝ) 
  (h1 : speed_with_stream = 26)
  (h2 : speed_against_stream = 4) :
  (speed_with_stream + speed_against_stream) / 2 = 15 := by
  sorry

#check mans_rowing_speed

end NUMINAMATH_CALUDE_mans_rowing_speed_l3619_361969


namespace NUMINAMATH_CALUDE_basketball_baseball_volume_ratio_l3619_361934

theorem basketball_baseball_volume_ratio : 
  ∀ (r R : ℝ), r > 0 → R = 4 * r → 
  (4 / 3 * Real.pi * R^3) / (4 / 3 * Real.pi * r^3) = 64 := by
sorry

end NUMINAMATH_CALUDE_basketball_baseball_volume_ratio_l3619_361934


namespace NUMINAMATH_CALUDE_age_equality_time_l3619_361906

/-- Given two people a and b, where a is 5 years older than b and their present ages sum to 13,
    this theorem proves that it will take 11 years for thrice a's age to equal 4 times b's age. -/
theorem age_equality_time (a b : ℕ) : 
  a = b + 5 → 
  a + b = 13 → 
  ∃ x : ℕ, x = 11 ∧ 3 * (a + x) = 4 * (b + x) :=
by sorry

end NUMINAMATH_CALUDE_age_equality_time_l3619_361906


namespace NUMINAMATH_CALUDE_exam_sections_percentage_l3619_361950

theorem exam_sections_percentage :
  let total_candidates : ℕ := 1200
  let all_sections_percent : ℚ := 5 / 100
  let no_sections_percent : ℚ := 5 / 100
  let one_section_percent : ℚ := 25 / 100
  let four_sections_percent : ℚ := 20 / 100
  let three_sections_count : ℕ := 300
  
  ∃ (two_sections_percent : ℚ),
    two_sections_percent = 20 / 100 ∧
    (all_sections_percent + no_sections_percent + one_section_percent + 
     four_sections_percent + two_sections_percent + 
     (three_sections_count : ℚ) / total_candidates) = 1 :=
by sorry

end NUMINAMATH_CALUDE_exam_sections_percentage_l3619_361950


namespace NUMINAMATH_CALUDE_solution_product_log_l3619_361941

-- Define the system of equations
def system_of_equations (x y : ℝ) : Prop :=
  (Real.log x / Real.log 225 + Real.log y / Real.log 64 = 4) ∧
  (Real.log 225 / Real.log x - Real.log 64 / Real.log y = 1)

-- State the theorem
theorem solution_product_log (x₁ y₁ x₂ y₂ : ℝ) :
  system_of_equations x₁ y₁ ∧ system_of_equations x₂ y₂ →
  Real.log (x₁ * y₁ * x₂ * y₂) / Real.log 30 = 12 :=
sorry

end NUMINAMATH_CALUDE_solution_product_log_l3619_361941


namespace NUMINAMATH_CALUDE_largest_number_given_hcf_and_lcm_factors_l3619_361993

theorem largest_number_given_hcf_and_lcm_factors (a b : ℕ+) : 
  (Nat.gcd a b = 40) → 
  (∃ (k : ℕ+), Nat.lcm a b = 40 * 11 * 12 * k) → 
  (max a b = 480) := by
sorry

end NUMINAMATH_CALUDE_largest_number_given_hcf_and_lcm_factors_l3619_361993


namespace NUMINAMATH_CALUDE_fish_difference_l3619_361961

/-- Proves that Matthias has 15 fewer fish than Kenneth given the conditions in the problem -/
theorem fish_difference (micah_fish : ℕ) (total_fish : ℕ) : 
  micah_fish = 7 →
  total_fish = 34 →
  let kenneth_fish := 3 * micah_fish
  let matthias_fish := total_fish - micah_fish - kenneth_fish
  kenneth_fish - matthias_fish = 15 := by
sorry


end NUMINAMATH_CALUDE_fish_difference_l3619_361961


namespace NUMINAMATH_CALUDE_fraction_chain_l3619_361918

theorem fraction_chain (a b c d e : ℝ) 
  (h1 : a/b = 5)
  (h2 : b/c = 1/2)
  (h3 : c/d = 4)
  (h4 : d/e = 1/3)
  : e/a = 3/10 := by
  sorry

end NUMINAMATH_CALUDE_fraction_chain_l3619_361918


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3619_361951

theorem min_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y ≤ 1) :
  x^4 + y^4 - x^2*y - x*y^2 ≥ -1/8 ∧ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b ≤ 1 ∧ a^4 + b^4 - a^2*b - a*b^2 = -1/8 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3619_361951


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l3619_361909

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y - 6 = 0

-- State the theorem
theorem circle_center_and_radius :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ (x y : ℝ), circle_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) ∧
    center = (-1, 2) ∧
    radius = Real.sqrt 11 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l3619_361909


namespace NUMINAMATH_CALUDE_job_completion_days_l3619_361904

/-- The number of days initially planned for a job to be completed, given:
  * 6 workers start the job
  * After 3 days, 4 more workers join
  * With 10 workers, the job is finished in 3 more days
  * Each worker has the same efficiency -/
def initial_days : ℕ := 6

/-- The total amount of work to be done -/
def total_work : ℝ := 1

/-- The number of workers that start the job -/
def initial_workers : ℕ := 6

/-- The number of days worked before additional workers join -/
def days_before_join : ℕ := 3

/-- The number of additional workers that join -/
def additional_workers : ℕ := 4

/-- The number of days needed to finish the job after additional workers join -/
def days_after_join : ℕ := 3

theorem job_completion_days :
  let work_rate := total_work / initial_days
  let work_done_before_join := days_before_join * work_rate
  let remaining_work := total_work - work_done_before_join
  let total_workers := initial_workers + additional_workers
  remaining_work / days_after_join = total_workers * work_rate
  → initial_days = 6 := by sorry

end NUMINAMATH_CALUDE_job_completion_days_l3619_361904


namespace NUMINAMATH_CALUDE_hyperbola_distance_property_l3619_361999

-- Define the hyperbola
def is_on_hyperbola (x y : ℝ) : Prop := x^2 / 25 - y^2 / 9 = 1

-- Define the foci
def F1 : ℝ × ℝ := sorry
def F2 : ℝ × ℝ := sorry

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem hyperbola_distance_property (P : ℝ × ℝ) :
  is_on_hyperbola P.1 P.2 →
  distance P F1 = 12 →
  (distance P F2 = 2 ∨ distance P F2 = 22) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_distance_property_l3619_361999


namespace NUMINAMATH_CALUDE_stripe_ratio_l3619_361919

/-- Given the conditions about stripes on tennis shoes, prove the ratio of Hortense's to Olga's stripes -/
theorem stripe_ratio (olga_stripes_per_shoe : ℕ) (rick_stripes_per_shoe : ℕ) (total_stripes : ℕ)
  (h1 : olga_stripes_per_shoe = 3)
  (h2 : rick_stripes_per_shoe = olga_stripes_per_shoe - 1)
  (h3 : total_stripes = 22)
  (h4 : total_stripes = 2 * olga_stripes_per_shoe + 2 * rick_stripes_per_shoe + hortense_stripes) :
  hortense_stripes / (2 * olga_stripes_per_shoe) = 2 :=
by sorry

end NUMINAMATH_CALUDE_stripe_ratio_l3619_361919


namespace NUMINAMATH_CALUDE_vins_school_distance_l3619_361980

/-- The distance Vins rides to school -/
def distance_to_school : ℝ := sorry

/-- The distance Vins rides back home -/
def distance_back_home : ℝ := 7

/-- The number of round trips Vins made this week -/
def number_of_trips : ℕ := 5

/-- The total distance Vins rode this week -/
def total_distance : ℝ := 65

theorem vins_school_distance :
  distance_to_school = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_vins_school_distance_l3619_361980


namespace NUMINAMATH_CALUDE_expression_equality_l3619_361987

theorem expression_equality (x : ℝ) (h : x > 0) : 
  (∃! e : ℕ, e = 1) ∧ 
  (6^x * x^3 = 3^x * x^3 + 3^x * x^3) ∧ 
  ((3*x)^(3*x) ≠ 3^x * x^3 + 3^x * x^3) ∧ 
  (3^x * x^6 ≠ 3^x * x^3 + 3^x * x^3) ∧ 
  ((6*x)^x ≠ 3^x * x^3 + 3^x * x^3) :=
sorry

end NUMINAMATH_CALUDE_expression_equality_l3619_361987


namespace NUMINAMATH_CALUDE_cubic_roots_inequality_l3619_361935

theorem cubic_roots_inequality (a b c d : ℝ) (h1 : a ≠ 0) 
  (h2 : ∃ (x₁ x₂ x₃ : ℝ), x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    a * x₁^3 + b * x₁^2 + c * x₁ + d = 0 ∧
    a * x₂^3 + b * x₂^2 + c * x₂ + d = 0 ∧
    a * x₃^3 + b * x₃^2 + c * x₃ + d = 0) :
  b * c < 3 * a * d := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_inequality_l3619_361935


namespace NUMINAMATH_CALUDE_original_triangle_area_l3619_361994

theorem original_triangle_area (original_area new_area : ℝ) : 
  (∀ side, new_side = 5 * side) → 
  new_area = 125 → 
  original_area = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_original_triangle_area_l3619_361994


namespace NUMINAMATH_CALUDE_quadratic_no_roots_implies_line_not_in_third_quadrant_l3619_361975

theorem quadratic_no_roots_implies_line_not_in_third_quadrant 
  (m : ℝ) (h : ∀ x : ℝ, m * x^2 - 2*x - 1 ≠ 0) :
  ∀ x y : ℝ, y = m*x - m → ¬(x < 0 ∧ y < 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_no_roots_implies_line_not_in_third_quadrant_l3619_361975


namespace NUMINAMATH_CALUDE_cube_of_negative_half_x_y_squared_l3619_361968

theorem cube_of_negative_half_x_y_squared (x y : ℝ) :
  (-1/2 * x * y^2)^3 = -1/8 * x^3 * y^6 := by sorry

end NUMINAMATH_CALUDE_cube_of_negative_half_x_y_squared_l3619_361968


namespace NUMINAMATH_CALUDE_min_sum_distances_to_four_points_l3619_361996

/-- The minimum sum of distances from a point to four fixed points -/
theorem min_sum_distances_to_four_points :
  let A : ℝ × ℝ := (-3, 0)
  let B : ℝ × ℝ := (1, -1)
  let C : ℝ × ℝ := (0, 3)
  let D : ℝ × ℝ := (-1, 3)
  ∀ P : ℝ × ℝ,
    Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) +
    Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) +
    Real.sqrt ((P.1 - C.1)^2 + (P.2 - C.2)^2) +
    Real.sqrt ((P.1 - D.1)^2 + (P.2 - D.2)^2) ≥
    3 * Real.sqrt 2 + 2 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_CALUDE_min_sum_distances_to_four_points_l3619_361996


namespace NUMINAMATH_CALUDE_change_amount_l3619_361964

-- Define the given conditions
def pants_price : ℚ := 60
def shirt_price : ℚ := 45
def tie_price : ℚ := 20
def discount_rate : ℚ := 0.1
def tax_rate : ℚ := 0.075
def paid_amount : ℚ := 500

-- Define the calculation steps
def pants_total : ℚ := 3 * pants_price
def shirts_total : ℚ := 2 * shirt_price
def discount_amount : ℚ := discount_rate * shirt_price
def discounted_shirts_total : ℚ := shirts_total - discount_amount
def subtotal : ℚ := pants_total + discounted_shirts_total + tie_price
def tax_amount : ℚ := tax_rate * subtotal
def total_purchase : ℚ := subtotal + tax_amount
def change : ℚ := paid_amount - total_purchase

-- Theorem to prove
theorem change_amount : change = 193.09 := by
  sorry

end NUMINAMATH_CALUDE_change_amount_l3619_361964


namespace NUMINAMATH_CALUDE_remainder_of_3_pow_2000_mod_13_l3619_361958

theorem remainder_of_3_pow_2000_mod_13 : (3^2000 : ℕ) % 13 = 9 := by sorry

end NUMINAMATH_CALUDE_remainder_of_3_pow_2000_mod_13_l3619_361958


namespace NUMINAMATH_CALUDE_disk_count_l3619_361925

/-- Represents the colors of disks in the bag -/
inductive DiskColor
  | Blue
  | Yellow
  | Green

/-- Represents the bag of disks -/
structure DiskBag where
  blue : ℕ
  yellow : ℕ
  green : ℕ

/-- The ratio of blue:yellow:green disks is 3:7:8 -/
def ratio_condition (bag : DiskBag) : Prop :=
  ∃ (x : ℕ), bag.blue = 3 * x ∧ bag.yellow = 7 * x ∧ bag.green = 8 * x

/-- There are 20 more green disks than blue disks -/
def green_blue_difference (bag : DiskBag) : Prop :=
  bag.green = bag.blue + 20

/-- The total number of disks in the bag -/
def total_disks (bag : DiskBag) : ℕ :=
  bag.blue + bag.yellow + bag.green

/-- Theorem: The total number of disks in the bag is 72 -/
theorem disk_count (bag : DiskBag) 
  (h1 : ratio_condition bag) 
  (h2 : green_blue_difference bag) : 
  total_disks bag = 72 := by
  sorry


end NUMINAMATH_CALUDE_disk_count_l3619_361925


namespace NUMINAMATH_CALUDE_art_gallery_total_pieces_l3619_361978

theorem art_gallery_total_pieces : 
  ∀ (D S : ℕ),
  (2 : ℚ) / 5 * D + (3 : ℚ) / 7 * S = (D + S) * (2 : ℚ) / 5 →
  (1 : ℚ) / 5 * D + (2 : ℚ) / 7 * S = 1500 →
  (2 : ℚ) / 5 * D = 600 →
  D + S = 5700 :=
by
  sorry

end NUMINAMATH_CALUDE_art_gallery_total_pieces_l3619_361978


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l3619_361945

theorem right_triangle_side_length : ∀ (a b c : ℝ),
  a > 0 → b > 0 → c > 0 →
  c = 17 → a = 15 →
  c^2 = a^2 + b^2 →
  b = 8 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l3619_361945


namespace NUMINAMATH_CALUDE_greatest_non_sum_of_composites_l3619_361939

def isComposite (n : ℕ) : Prop := n > 1 ∧ ¬ Nat.Prime n

def isSumOfTwoComposites (n : ℕ) : Prop :=
  ∃ a b : ℕ, isComposite a ∧ isComposite b ∧ a + b = n

theorem greatest_non_sum_of_composites :
  (∀ n : ℕ, n > 11 → isSumOfTwoComposites n) ∧
  ¬ isSumOfTwoComposites 11 := by sorry

end NUMINAMATH_CALUDE_greatest_non_sum_of_composites_l3619_361939


namespace NUMINAMATH_CALUDE_power_of_product_l3619_361977

theorem power_of_product (a b : ℝ) : (a * b) ^ 3 = a ^ 3 * b ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l3619_361977


namespace NUMINAMATH_CALUDE_binomial_coefficient_7_choose_5_l3619_361967

theorem binomial_coefficient_7_choose_5 : Nat.choose 7 5 = 21 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_7_choose_5_l3619_361967


namespace NUMINAMATH_CALUDE_P_on_x_axis_P_distance_to_y_axis_l3619_361948

-- Define the point P as a function of a
def P (a : ℝ) : ℝ × ℝ := (2 * a - 1, a + 3)

-- Condition 1: P lies on the x-axis
theorem P_on_x_axis (a : ℝ) :
  P a = (-7, 0) ↔ (P a).2 = 0 :=
sorry

-- Condition 2: Distance from P to y-axis is 5
theorem P_distance_to_y_axis (a : ℝ) :
  (abs (P a).1 = 5) ↔ (P a = (-5, 1) ∨ P a = (5, 6)) :=
sorry

end NUMINAMATH_CALUDE_P_on_x_axis_P_distance_to_y_axis_l3619_361948


namespace NUMINAMATH_CALUDE_sufficient_condition_for_existence_necessary_condition_for_existence_not_necessary_condition_l3619_361957

theorem sufficient_condition_for_existence (m : ℝ) :
  (m ≤ 4) → (∃ x : ℝ, x ∈ Set.Icc (-1) 2 ∧ x^2 - m ≥ 0) :=
by sorry

theorem necessary_condition_for_existence (m : ℝ) :
  (∃ x : ℝ, x ∈ Set.Icc (-1) 2 ∧ x^2 - m ≥ 0) → (m ≤ 4) :=
by sorry

theorem not_necessary_condition (m : ℝ) :
  ∃ m₀ : ℝ, m₀ ≤ 4 ∧ m₀ ≠ 4 ∧ (∃ x : ℝ, x ∈ Set.Icc (-1) 2 ∧ x^2 - m₀ ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_existence_necessary_condition_for_existence_not_necessary_condition_l3619_361957


namespace NUMINAMATH_CALUDE_set_operations_l3619_361914

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B : Set ℝ := {x | 0 < x ∧ x ≤ 4}

-- Define the intervals for the expected results
def open_interval (a b : ℝ) : Set ℝ := {x | a < x ∧ x < b}
def closed_open_interval (a b : ℝ) : Set ℝ := {x | a ≤ x ∧ x < b}
def open_closed_interval (a b : ℝ) : Set ℝ := {x | a < x ∧ x ≤ b}
def left_ray (a : ℝ) : Set ℝ := {x | x ≤ a}
def right_ray (a : ℝ) : Set ℝ := {x | a < x}

-- State the theorem
theorem set_operations :
  (A ∩ B = open_interval 0 3) ∧
  (A ∪ B = open_interval (-1) 4) ∧
  ((Aᶜ ∩ Bᶜ) = left_ray (-1) ∪ right_ray 4) :=
by sorry

end NUMINAMATH_CALUDE_set_operations_l3619_361914


namespace NUMINAMATH_CALUDE_ski_price_after_discounts_l3619_361902

def original_price : ℝ := 200
def first_discount : ℝ := 0.4
def second_discount : ℝ := 0.2

theorem ski_price_after_discounts :
  let price_after_first := original_price * (1 - first_discount)
  let final_price := price_after_first * (1 - second_discount)
  final_price = 96 := by sorry

end NUMINAMATH_CALUDE_ski_price_after_discounts_l3619_361902


namespace NUMINAMATH_CALUDE_bus_seat_difference_l3619_361913

/-- Represents a bus with seats on both sides and a special seat at the back. -/
structure Bus where
  leftSeats : Nat
  rightSeats : Nat
  backSeatCapacity : Nat
  regularSeatCapacity : Nat
  totalCapacity : Nat

/-- The difference in the number of seats between the left and right sides of the bus. -/
def seatDifference (bus : Bus) : Nat :=
  bus.leftSeats - bus.rightSeats

/-- Theorem stating the difference in seats for a specific bus configuration. -/
theorem bus_seat_difference :
  ∃ (bus : Bus),
    bus.leftSeats = 15 ∧
    bus.backSeatCapacity = 11 ∧
    bus.regularSeatCapacity = 3 ∧
    bus.totalCapacity = 92 ∧
    seatDifference bus = 3 := by
  sorry

#check bus_seat_difference

end NUMINAMATH_CALUDE_bus_seat_difference_l3619_361913


namespace NUMINAMATH_CALUDE_condition_relationship_l3619_361940

theorem condition_relationship (x : ℝ) :
  (∀ x, (x + 2) * (x - 1) < 0 → x < 1) ∧
  (∃ x, x < 1 ∧ ¬((x + 2) * (x - 1) < 0)) :=
by sorry

end NUMINAMATH_CALUDE_condition_relationship_l3619_361940


namespace NUMINAMATH_CALUDE_inequality_proof_l3619_361954

theorem inequality_proof (a b c d : ℝ) 
  (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (hd : d ≥ 0)
  (hca : c + d ≤ a) (hcb : c + d ≤ b) : 
  a * d + b * c ≤ a * b := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3619_361954


namespace NUMINAMATH_CALUDE_win_sector_area_l3619_361900

/-- The area of the WIN sector on a circular spinner with given radius and winning probability -/
theorem win_sector_area (r : ℝ) (p : ℝ) (h_r : r = 10) (h_p : p = 3/7) :
  p * π * r^2 = 300 * π / 7 := by
  sorry

end NUMINAMATH_CALUDE_win_sector_area_l3619_361900


namespace NUMINAMATH_CALUDE_identity_proof_l3619_361944

theorem identity_proof (a b c x y z : ℝ) : 
  (a*x + b*y + c*z)^2 + (b*x + c*y + a*z)^2 + (c*x + a*y + b*z)^2 = 
  (c*x + b*y + a*z)^2 + (b*x + a*y + c*z)^2 + (a*x + c*y + b*z)^2 := by
  sorry

end NUMINAMATH_CALUDE_identity_proof_l3619_361944


namespace NUMINAMATH_CALUDE_percentage_difference_l3619_361928

theorem percentage_difference (a b : ℝ) (h : b ≠ 0) :
  (a - b) / b * 100 = 25 → a = 100 ∧ b = 80 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l3619_361928


namespace NUMINAMATH_CALUDE_subtract_negative_three_l3619_361956

theorem subtract_negative_three : 0 - (-3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_subtract_negative_three_l3619_361956


namespace NUMINAMATH_CALUDE_smallest_number_divisible_l3619_361905

def is_divisible_by_all (n : ℕ) : Prop :=
  (n + 7) % 24 = 0 ∧
  (n + 7) % 36 = 0 ∧
  (n + 7) % 50 = 0 ∧
  (n + 7) % 56 = 0 ∧
  (n + 7) % 81 = 0

theorem smallest_number_divisible : 
  is_divisible_by_all 113393 ∧ 
  ∀ m : ℕ, m < 113393 → ¬is_divisible_by_all m :=
sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_l3619_361905


namespace NUMINAMATH_CALUDE_range_of_a_l3619_361942

theorem range_of_a (a : ℝ) : 
  (¬ ∃ t : ℝ, t^2 - 2*t - a < 0) → 
  (∀ x : ℝ, x ≤ a → x ≤ -1) ∧ a ≤ -1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3619_361942


namespace NUMINAMATH_CALUDE_area_of_specific_quadrilateral_l3619_361946

/-- Represents a convex quadrilateral ABCD with specific side lengths and a right angle -/
structure ConvexQuadrilateral where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DA : ℝ
  angle_CDA : ℝ
  convex : AB > 0 ∧ BC > 0 ∧ CD > 0 ∧ DA > 0
  right_angle : angle_CDA = 90

/-- The area of the specific convex quadrilateral ABCD is 62 -/
theorem area_of_specific_quadrilateral (ABCD : ConvexQuadrilateral)
    (h1 : ABCD.AB = 8)
    (h2 : ABCD.BC = 4)
    (h3 : ABCD.CD = 10)
    (h4 : ABCD.DA = 10) :
    Real.sqrt 0 + 62 * Real.sqrt 1 = 62 := by
  sorry

end NUMINAMATH_CALUDE_area_of_specific_quadrilateral_l3619_361946


namespace NUMINAMATH_CALUDE_f_values_l3619_361984

/-- 
Represents the number of permutations a₁, ..., aₙ of the set {1, 2, ..., n} 
such that |aᵢ - aᵢ₊₁| ≠ 1 for all i = 1, 2, ..., n-1.
-/
def f (n : ℕ) : ℕ := sorry

/-- The main theorem stating the values of f for n from 2 to 6 -/
theorem f_values : 
  f 2 = 0 ∧ f 3 = 0 ∧ f 4 = 2 ∧ f 5 = 14 ∧ f 6 = 90 := by sorry

end NUMINAMATH_CALUDE_f_values_l3619_361984


namespace NUMINAMATH_CALUDE_dot_product_om_on_l3619_361982

/-- Given two points M and N on the line x + y - 2 = 0, where M(1,1) and |MN| = √2,
    prove that the dot product of OM and ON equals 2 -/
theorem dot_product_om_on (N : ℝ × ℝ) : 
  N.1 + N.2 = 2 →  -- N is on the line x + y - 2 = 0
  (N.1 - 1)^2 + (N.2 - 1)^2 = 2 →  -- |MN| = √2
  (1 * N.1 + 1 * N.2 : ℝ) = 2 := by  -- OM · ON = 2
sorry

end NUMINAMATH_CALUDE_dot_product_om_on_l3619_361982


namespace NUMINAMATH_CALUDE_discount_savings_l3619_361949

theorem discount_savings (original_price : ℝ) (discount_rate : ℝ) (num_contributors : ℕ) 
  (discounted_price : ℝ) (individual_savings : ℝ) : 
  original_price > 0 → 
  discount_rate = 0.2 → 
  num_contributors = 3 → 
  discounted_price = 48 → 
  discounted_price = original_price * (1 - discount_rate) → 
  individual_savings = (original_price - discounted_price) / num_contributors → 
  individual_savings = 4 := by
sorry

end NUMINAMATH_CALUDE_discount_savings_l3619_361949


namespace NUMINAMATH_CALUDE_total_trees_after_planting_l3619_361962

/-- The number of walnut trees initially in the park -/
def initial_trees : ℕ := 4

/-- The number of new walnut trees to be planted -/
def new_trees : ℕ := 6

/-- Theorem: The total number of walnut trees after planting is 10 -/
theorem total_trees_after_planting : 
  initial_trees + new_trees = 10 := by sorry

end NUMINAMATH_CALUDE_total_trees_after_planting_l3619_361962


namespace NUMINAMATH_CALUDE_only_set_C_forms_triangle_l3619_361912

/-- Triangle inequality theorem: The sum of the lengths of any two sides of a triangle
    must be greater than the length of the remaining side. -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Check if a set of three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

/-- The given sets of line segments -/
def set_A : (ℝ × ℝ × ℝ) := (3, 5, 8)
def set_B : (ℝ × ℝ × ℝ) := (8, 8, 18)
def set_C : (ℝ × ℝ × ℝ) := (1, 1, 1)
def set_D : (ℝ × ℝ × ℝ) := (3, 4, 8)

/-- Theorem: Among the given sets, only set C can form a triangle -/
theorem only_set_C_forms_triangle :
  ¬(can_form_triangle set_A.1 set_A.2.1 set_A.2.2) ∧
  ¬(can_form_triangle set_B.1 set_B.2.1 set_B.2.2) ∧
  can_form_triangle set_C.1 set_C.2.1 set_C.2.2 ∧
  ¬(can_form_triangle set_D.1 set_D.2.1 set_D.2.2) :=
by sorry

end NUMINAMATH_CALUDE_only_set_C_forms_triangle_l3619_361912


namespace NUMINAMATH_CALUDE_min_ceiling_height_for_illumination_l3619_361901

/-- The minimum ceiling height for complete illumination of a rectangular field. -/
theorem min_ceiling_height_for_illumination (length width : ℝ) 
  (h : ℝ) (multiple : ℝ) : 
  length = 100 →
  width = 80 →
  multiple = 0.1 →
  (∃ (n : ℕ), h = n * multiple) →
  (2 * h ≥ Real.sqrt (length^2 + width^2)) →
  (∀ (h' : ℝ), (∃ (n : ℕ), h' = n * multiple) → 
    (2 * h' ≥ Real.sqrt (length^2 + width^2)) → h' ≥ h) →
  h = 32.1 :=
by sorry

end NUMINAMATH_CALUDE_min_ceiling_height_for_illumination_l3619_361901


namespace NUMINAMATH_CALUDE_scientific_notation_of_20160_l3619_361903

theorem scientific_notation_of_20160 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 20160 = a * (10 : ℝ) ^ n ∧ a = 2.016 ∧ n = 4 :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_20160_l3619_361903


namespace NUMINAMATH_CALUDE_add_8035_seconds_to_8am_l3619_361930

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat
  deriving Repr

/-- Adds seconds to a given time -/
def addSeconds (t : Time) (s : Nat) : Time :=
  sorry

/-- The starting time (8:00:00 AM) -/
def startTime : Time :=
  { hours := 8, minutes := 0, seconds := 0 }

/-- The number of seconds to add -/
def secondsToAdd : Nat := 8035

/-- The expected end time (10:13:35) -/
def endTime : Time :=
  { hours := 10, minutes := 13, seconds := 35 }

theorem add_8035_seconds_to_8am :
  addSeconds startTime secondsToAdd = endTime := by
  sorry

end NUMINAMATH_CALUDE_add_8035_seconds_to_8am_l3619_361930


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_sum_of_squares_l3619_361953

theorem arithmetic_geometric_mean_sum_of_squares (a b : ℝ) :
  (a + b) / 2 = 20 → Real.sqrt (a * b) = Real.sqrt 110 → a^2 + b^2 = 1380 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_sum_of_squares_l3619_361953


namespace NUMINAMATH_CALUDE_ninth_term_is_zero_l3619_361923

/-- An arithmetic sequence with a₄ = 5 and a₅ = 4 -/
def arithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d ∧ a 4 = 5 ∧ a 5 = 4

theorem ninth_term_is_zero (a : ℕ → ℤ) (h : arithmeticSequence a) : a 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ninth_term_is_zero_l3619_361923


namespace NUMINAMATH_CALUDE_big_boxes_count_l3619_361985

/-- The number of big boxes given the conditions of the problem -/
def number_of_big_boxes (small_boxes_per_big_box : ℕ) 
                        (candles_per_small_box : ℕ) 
                        (total_candles : ℕ) : ℕ :=
  total_candles / (small_boxes_per_big_box * candles_per_small_box)

theorem big_boxes_count :
  number_of_big_boxes 4 40 8000 = 50 := by
  sorry

#eval number_of_big_boxes 4 40 8000

end NUMINAMATH_CALUDE_big_boxes_count_l3619_361985


namespace NUMINAMATH_CALUDE_integer_power_sum_l3619_361986

theorem integer_power_sum (a : ℝ) (h1 : a ≠ 0) (h2 : ∃ k : ℤ, a + 1/a = k) :
  ∀ n : ℕ, 2 ≤ n ∧ n ≤ 7 → ∃ m : ℤ, a^n + 1/(a^n) = m :=
sorry

end NUMINAMATH_CALUDE_integer_power_sum_l3619_361986


namespace NUMINAMATH_CALUDE_max_parts_is_ten_l3619_361963

/-- Represents a Viennese pretzel lying on a table -/
structure ViennesePretzel where
  loops : ℕ
  intersections : ℕ

/-- Represents a straight cut through the pretzel -/
structure StraightCut where
  intersectionsCut : ℕ

/-- The number of parts resulting from a straight cut -/
def numParts (p : ViennesePretzel) (c : StraightCut) : ℕ :=
  c.intersectionsCut + 1

/-- The maximum number of intersections that can be cut by a single straight line -/
def maxIntersectionsCut (p : ViennesePretzel) : ℕ := 9

/-- Theorem stating that the maximum number of parts is 10 -/
theorem max_parts_is_ten (p : ViennesePretzel) :
  ∃ c : StraightCut, numParts p c = 10 ∧
  ∀ c' : StraightCut, numParts p c' ≤ 10 :=
sorry

end NUMINAMATH_CALUDE_max_parts_is_ten_l3619_361963


namespace NUMINAMATH_CALUDE_increase_by_percentage_l3619_361997

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (result : ℝ) :
  initial = 240 →
  percentage = 20 →
  result = initial * (1 + percentage / 100) →
  result = 288 := by
  sorry

end NUMINAMATH_CALUDE_increase_by_percentage_l3619_361997


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l3619_361921

theorem quadratic_one_solution (n : ℝ) : 
  (∃! x : ℝ, 4 * x^2 + n * x + 4 = 0) ↔ (n = 8 ∧ n > 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l3619_361921


namespace NUMINAMATH_CALUDE_reflection_sum_theorem_l3619_361907

def point (x y : ℝ) := (x, y)

def reflect_over_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

def sum_coordinates (p1 p2 : ℝ × ℝ) : ℝ :=
  p1.1 + p1.2 + p2.1 + p2.2

theorem reflection_sum_theorem :
  let C : ℝ × ℝ := point 5 (-3)
  let D : ℝ × ℝ := reflect_over_x_axis C
  sum_coordinates C D = 10 := by sorry

end NUMINAMATH_CALUDE_reflection_sum_theorem_l3619_361907


namespace NUMINAMATH_CALUDE_chicken_cost_problem_l3619_361965

/-- A problem about calculating the cost of chickens given various expenses --/
theorem chicken_cost_problem (land_acres : ℕ) (land_cost_per_acre : ℕ) 
  (house_cost : ℕ) (cow_count : ℕ) (cow_cost : ℕ) (chicken_count : ℕ) 
  (solar_install_hours : ℕ) (solar_install_rate : ℕ) (solar_equipment_cost : ℕ) 
  (total_cost : ℕ) : 
  land_acres = 30 →
  land_cost_per_acre = 20 →
  house_cost = 120000 →
  cow_count = 20 →
  cow_cost = 1000 →
  chicken_count = 100 →
  solar_install_hours = 6 →
  solar_install_rate = 100 →
  solar_equipment_cost = 6000 →
  total_cost = 147700 →
  (total_cost - (land_acres * land_cost_per_acre + house_cost + cow_count * cow_cost + 
    solar_install_hours * solar_install_rate + solar_equipment_cost)) / chicken_count = 5 :=
by sorry

end NUMINAMATH_CALUDE_chicken_cost_problem_l3619_361965


namespace NUMINAMATH_CALUDE_parentheses_placement_l3619_361937

theorem parentheses_placement :
  let original := 0.5 + 0.5 / 0.5 + 0.5 / 0.5
  let with_parentheses := ((0.5 + 0.5) / 0.5 + 0.5) / 0.5
  with_parentheses = 5 ∧ with_parentheses ≠ original :=
by sorry

end NUMINAMATH_CALUDE_parentheses_placement_l3619_361937


namespace NUMINAMATH_CALUDE_first_number_problem_l3619_361952

theorem first_number_problem (x : ℤ) : x + 7314 = 3362 + 13500 → x = 9548 := by
  sorry

end NUMINAMATH_CALUDE_first_number_problem_l3619_361952


namespace NUMINAMATH_CALUDE_division_problem_l3619_361988

theorem division_problem (L S q : ℕ) : 
  L - S = 2415 → 
  L = 2520 → 
  L = S * q + 15 → 
  q = 23 := by sorry

end NUMINAMATH_CALUDE_division_problem_l3619_361988


namespace NUMINAMATH_CALUDE_two_plus_three_eq_eight_is_proposition_l3619_361973

-- Define what a proposition is
def is_proposition (s : String) : Prop := ∃ (b : Bool), (s = "true" ∨ s = "false")

-- State the theorem
theorem two_plus_three_eq_eight_is_proposition :
  is_proposition "2+3=8" :=
sorry

end NUMINAMATH_CALUDE_two_plus_three_eq_eight_is_proposition_l3619_361973


namespace NUMINAMATH_CALUDE_orange_harvest_per_day_l3619_361929

/-- Given a consistent daily harvest of oranges over 6 days resulting in 498 sacks,
    prove that the daily harvest is 83 sacks. -/
theorem orange_harvest_per_day :
  ∀ (daily_harvest : ℕ),
  daily_harvest * 6 = 498 →
  daily_harvest = 83 := by
  sorry

end NUMINAMATH_CALUDE_orange_harvest_per_day_l3619_361929


namespace NUMINAMATH_CALUDE_nested_fraction_value_l3619_361974

theorem nested_fraction_value : 
  (1 : ℚ) / (1 + 1 / (1 + 1 / 2)) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_value_l3619_361974


namespace NUMINAMATH_CALUDE_willowton_vampires_l3619_361970

def vampire_growth (initial_population : ℕ) (initial_vampires : ℕ) (turns_per_night : ℕ) (nights : ℕ) : ℕ :=
  sorry

theorem willowton_vampires :
  vampire_growth 300 2 5 2 = 72 :=
sorry

end NUMINAMATH_CALUDE_willowton_vampires_l3619_361970


namespace NUMINAMATH_CALUDE_root_property_l3619_361991

theorem root_property (a : ℝ) (h : a^2 + a - 2009 = 0) : a^2 + a - 1 = 2008 := by
  sorry

end NUMINAMATH_CALUDE_root_property_l3619_361991


namespace NUMINAMATH_CALUDE_estate_distribution_l3619_361933

/-- Represents the estate distribution problem --/
theorem estate_distribution (E : ℕ) : 
  -- Daughter and son together receive half the estate
  (∃ x : ℕ, 5 * x = E / 2) →
  -- Wife receives three times as much as the son
  (∃ y : ℕ, y = 6 * x) →
  -- First cook receives $800
  (∃ z₁ : ℕ, z₁ = 800) →
  -- Second cook receives $1200
  (∃ z₂ : ℕ, z₂ = 1200) →
  -- Total estate equals sum of all shares
  (E = 11 * x + 2000) →
  -- The estate value is $20000
  E = 20000 := by
sorry

end NUMINAMATH_CALUDE_estate_distribution_l3619_361933


namespace NUMINAMATH_CALUDE_intersection_points_theorem_l3619_361932

-- Define the functions
def f (x : ℝ) : ℝ := (x - 2) * (x - 5)
def g (x : ℝ) : ℝ := 2 * f x
def h (x : ℝ) : ℝ := f (-x) + 2

-- Define the number of intersection points
def a : ℕ := 2  -- number of intersection points between y=f(x) and y=g(x)
def b : ℕ := 1  -- number of intersection points between y=f(x) and y=h(x)

-- Theorem statement
theorem intersection_points_theorem : 10 * a + b = 21 := by sorry

end NUMINAMATH_CALUDE_intersection_points_theorem_l3619_361932


namespace NUMINAMATH_CALUDE_folded_area_ratio_l3619_361960

/-- Represents a rectangular paper with specific folding properties -/
structure FoldedPaper where
  width : ℝ
  length : ℝ
  area : ℝ
  foldedArea : ℝ
  lengthWidthRatio : length = Real.sqrt 3 * width
  areaDefinition : area = length * width
  foldedAreaDefinition : foldedArea = area - (Real.sqrt 3 * width^2) / 6

/-- The ratio of the folded area to the original area is 5/6 -/
theorem folded_area_ratio (paper : FoldedPaper) : 
  paper.foldedArea / paper.area = 5 / 6 := by
  sorry


end NUMINAMATH_CALUDE_folded_area_ratio_l3619_361960


namespace NUMINAMATH_CALUDE_solve_score_problem_l3619_361981

def score_problem (s1 s3 s4 : ℕ) (avg : ℚ) : Prop :=
  s1 ≤ 100 ∧ s3 ≤ 100 ∧ s4 ≤ 100 ∧
  s1 = 65 ∧ s3 = 82 ∧ s4 = 85 ∧
  avg = 75 ∧
  ∃ (s2 : ℕ), s2 ≤ 100 ∧ (s1 + s2 + s3 + s4 : ℚ) / 4 = avg ∧ s2 = 68

theorem solve_score_problem (s1 s3 s4 : ℕ) (avg : ℚ) 
  (h : score_problem s1 s3 s4 avg) : 
  ∃ (s2 : ℕ), s2 = 68 ∧ (s1 + s2 + s3 + s4 : ℚ) / 4 = avg :=
by sorry

end NUMINAMATH_CALUDE_solve_score_problem_l3619_361981


namespace NUMINAMATH_CALUDE_greatest_integer_less_than_negative_nineteen_fifths_l3619_361936

theorem greatest_integer_less_than_negative_nineteen_fifths :
  Int.floor (-19 / 5 : ℚ) = -4 := by sorry

end NUMINAMATH_CALUDE_greatest_integer_less_than_negative_nineteen_fifths_l3619_361936


namespace NUMINAMATH_CALUDE_f_inequality_solution_set_f_inequality_a_range_l3619_361990

def f (x : ℝ) : ℝ := |2*x - 2| - |x + 1|

theorem f_inequality_solution_set :
  {x : ℝ | f x ≤ 3} = {x : ℝ | -2/3 ≤ x ∧ x ≤ 6} := by sorry

theorem f_inequality_a_range :
  {a : ℝ | ∀ x, f x ≤ |x + 1| + a^2} = {a : ℝ | a ≤ -2 ∨ 2 ≤ a} := by sorry

end NUMINAMATH_CALUDE_f_inequality_solution_set_f_inequality_a_range_l3619_361990


namespace NUMINAMATH_CALUDE_vector_BC_l3619_361927

/-- Given points A and B, and vector AC, prove that vector BC is (-3, 2) -/
theorem vector_BC (A B C : ℝ × ℝ) : 
  A = (-1, 1) → B = (0, 2) → (C.1 - A.1, C.2 - A.2) = (-2, 3) → 
  (C.1 - B.1, C.2 - B.2) = (-3, 2) := by sorry

end NUMINAMATH_CALUDE_vector_BC_l3619_361927


namespace NUMINAMATH_CALUDE_unique_solution_implies_a_equals_10_l3619_361910

-- Define the equation
def equation (a : ℝ) (x : ℝ) : Prop :=
  (x * Real.log a ^ 2 - 1) / (x + Real.log a) = x

-- Theorem statement
theorem unique_solution_implies_a_equals_10 :
  (∃! x : ℝ, equation a x) → a = 10 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_implies_a_equals_10_l3619_361910


namespace NUMINAMATH_CALUDE_marked_price_calculation_l3619_361979

theorem marked_price_calculation (total_cost : ℚ) (discount_rate : ℚ) : 
  total_cost = 50 → discount_rate = 1/10 → 
  ∃ (marked_price : ℚ), marked_price = 250/9 ∧ 
  2 * (marked_price * (1 - discount_rate)) = total_cost :=
by sorry

end NUMINAMATH_CALUDE_marked_price_calculation_l3619_361979


namespace NUMINAMATH_CALUDE_one_nonnegative_solution_l3619_361924

theorem one_nonnegative_solution :
  ∃! (x : ℝ), x ≥ 0 ∧ x^2 + 6*x = 0 :=
by sorry

end NUMINAMATH_CALUDE_one_nonnegative_solution_l3619_361924


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3619_361938

theorem polynomial_factorization :
  (∀ x : ℝ, 3 * x^2 - 7 * x - 6 = (x - 3) * (3 * x + 2)) ∧
  (∀ x : ℝ, 6 * x^2 - 7 * x - 5 = (2 * x + 1) * (3 * x - 5)) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3619_361938


namespace NUMINAMATH_CALUDE_marble_remainder_l3619_361966

theorem marble_remainder (r p : ℤ) 
  (hr : r % 8 = 5) 
  (hp : p % 8 = 6) : 
  (r + p) % 8 = 3 := by sorry

end NUMINAMATH_CALUDE_marble_remainder_l3619_361966


namespace NUMINAMATH_CALUDE_f_zero_at_negative_one_f_negative_iff_a_greater_than_negative_one_l3619_361976

-- Define the function f
def f (a b x : ℝ) : ℝ := (x - a) * abs x + b

-- Part 1: Prove that when a = 2 and b = 3, the only zero of f is x = -1
theorem f_zero_at_negative_one :
  ∃! x : ℝ, f 2 3 x = 0 ∧ x = -1 := by sorry

-- Part 2: Prove that when b = -2, f(x) < 0 for all x ∈ [-1, 1] if and only if a > -1
theorem f_negative_iff_a_greater_than_negative_one :
  ∀ a : ℝ, (∀ x : ℝ, x ∈ [-1, 1] → f a (-2) x < 0) ↔ a > -1 := by sorry

end NUMINAMATH_CALUDE_f_zero_at_negative_one_f_negative_iff_a_greater_than_negative_one_l3619_361976


namespace NUMINAMATH_CALUDE_mobile_bus_uses_gps_and_gis_mobile_bus_not_uses_remote_sensing_l3619_361992

/-- Represents the technologies used in the "Mobile Bus" app --/
inductive MobileBusTechnology
  | GPS        : MobileBusTechnology
  | GIS        : MobileBusTechnology
  | RemoteSensing : MobileBusTechnology
  | DigitalEarth  : MobileBusTechnology

/-- The set of technologies used in the "Mobile Bus" app --/
def mobileBusTechnologies : Set MobileBusTechnology :=
  {MobileBusTechnology.GPS, MobileBusTechnology.GIS}

/-- Theorem stating that the "Mobile Bus" app uses GPS and GIS --/
theorem mobile_bus_uses_gps_and_gis :
  MobileBusTechnology.GPS ∈ mobileBusTechnologies ∧
  MobileBusTechnology.GIS ∈ mobileBusTechnologies :=
by sorry

/-- Theorem stating that the "Mobile Bus" app does not use Remote Sensing --/
theorem mobile_bus_not_uses_remote_sensing :
  MobileBusTechnology.RemoteSensing ∉ mobileBusTechnologies :=
by sorry

end NUMINAMATH_CALUDE_mobile_bus_uses_gps_and_gis_mobile_bus_not_uses_remote_sensing_l3619_361992


namespace NUMINAMATH_CALUDE_total_payment_for_bikes_l3619_361947

/-- The payment for painting a bike -/
def paint_fee : ℕ := 5

/-- The additional payment for selling a bike compared to painting it -/
def sell_bonus : ℕ := 8

/-- The number of bikes Henry sells and paints -/
def num_bikes : ℕ := 8

/-- The total payment for selling and painting one bike -/
def payment_per_bike : ℕ := paint_fee + (paint_fee + sell_bonus)

/-- Theorem stating the total payment for selling and painting 8 bikes -/
theorem total_payment_for_bikes : num_bikes * payment_per_bike = 144 := by
  sorry

end NUMINAMATH_CALUDE_total_payment_for_bikes_l3619_361947


namespace NUMINAMATH_CALUDE_cube_root_equation_l3619_361989

theorem cube_root_equation : ∃ A : ℝ, 32 * A * A * A = 42592 ∧ A = 11 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_l3619_361989


namespace NUMINAMATH_CALUDE_sculpture_height_proof_l3619_361972

/-- The height of the sculpture in inches -/
def sculpture_height : ℝ := 34

/-- The height of the base in inches -/
def base_height : ℝ := 8

/-- The total height of the sculpture and base in feet -/
def total_height_feet : ℝ := 3.5

/-- Conversion factor from feet to inches -/
def feet_to_inches : ℝ := 12

theorem sculpture_height_proof :
  sculpture_height = total_height_feet * feet_to_inches - base_height := by
  sorry

end NUMINAMATH_CALUDE_sculpture_height_proof_l3619_361972


namespace NUMINAMATH_CALUDE_division_ratio_l3619_361922

theorem division_ratio (divisor quotient remainder : ℕ) 
  (h1 : divisor = 10 * quotient)
  (h2 : ∃ n : ℕ, divisor = n * remainder)
  (h3 : remainder = 46)
  (h4 : 5290 = divisor * quotient + remainder) :
  divisor / remainder = 5 :=
sorry

end NUMINAMATH_CALUDE_division_ratio_l3619_361922


namespace NUMINAMATH_CALUDE_average_speed_to_first_summit_l3619_361931

/-- Proves that the average speed to the first summit is equal to the overall average speed
    given the total journey time, overall average speed, and time to first summit. -/
theorem average_speed_to_first_summit
  (total_time : ℝ)
  (overall_avg_speed : ℝ)
  (time_to_first_summit : ℝ)
  (h_total_time : total_time = 8)
  (h_overall_avg_speed : overall_avg_speed = 3)
  (h_time_to_first_summit : time_to_first_summit = 3) :
  (overall_avg_speed * time_to_first_summit) / time_to_first_summit = overall_avg_speed :=
by sorry

#check average_speed_to_first_summit

end NUMINAMATH_CALUDE_average_speed_to_first_summit_l3619_361931


namespace NUMINAMATH_CALUDE_remainder_101_37_mod_100_l3619_361908

theorem remainder_101_37_mod_100 : (101^37) % 100 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_101_37_mod_100_l3619_361908


namespace NUMINAMATH_CALUDE_jeans_sold_proof_l3619_361917

/-- The number of pairs of jeans sold by a clothing store -/
def num_jeans : ℕ := 10

theorem jeans_sold_proof (shirts : ℕ) (shirt_price : ℕ) (jeans_price : ℕ) (total_revenue : ℕ) :
  shirts = 20 →
  shirt_price = 10 →
  jeans_price = 2 * shirt_price →
  total_revenue = 400 →
  shirts * shirt_price + num_jeans * jeans_price = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_jeans_sold_proof_l3619_361917
