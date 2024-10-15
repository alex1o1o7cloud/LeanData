import Mathlib

namespace NUMINAMATH_CALUDE_fractional_equation_root_l3614_361467

theorem fractional_equation_root (x m : ℝ) : 
  (∃ x, x / (x - 2) - 2 = m / (x - 2)) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_root_l3614_361467


namespace NUMINAMATH_CALUDE_john_volunteer_hours_per_year_l3614_361476

/-- 
Given that John volunteers twice a month for 3 hours each time, 
this theorem proves that he volunteers for 72 hours per year.
-/
theorem john_volunteer_hours_per_year 
  (times_per_month : ℕ) 
  (hours_per_time : ℕ) 
  (h1 : times_per_month = 2) 
  (h2 : hours_per_time = 3) : 
  times_per_month * 12 * hours_per_time = 72 := by
  sorry

end NUMINAMATH_CALUDE_john_volunteer_hours_per_year_l3614_361476


namespace NUMINAMATH_CALUDE_max_median_soda_sales_l3614_361460

/-- Represents the soda sales data for a weekend -/
structure SodaSales where
  totalCans : ℕ
  totalCustomers : ℕ
  minCansPerCustomer : ℕ

/-- Calculates the maximum possible median number of cans bought per customer -/
def maxPossibleMedian (sales : SodaSales) : ℚ :=
  sorry

/-- Theorem stating the maximum possible median for the given scenario -/
theorem max_median_soda_sales (sales : SodaSales)
  (h1 : sales.totalCans = 300)
  (h2 : sales.totalCustomers = 120)
  (h3 : sales.minCansPerCustomer = 2) :
  maxPossibleMedian sales = 3 :=
  sorry

end NUMINAMATH_CALUDE_max_median_soda_sales_l3614_361460


namespace NUMINAMATH_CALUDE_two_visits_count_l3614_361475

/-- Represents the visiting schedule of friends -/
structure VisitSchedule where
  alice : Nat
  beatrix : Nat
  claire : Nat

/-- Calculates the number of days when exactly two friends visit -/
def exactlyTwoVisits (schedule : VisitSchedule) (totalDays : Nat) : Nat :=
  sorry

/-- The main theorem to prove -/
theorem two_visits_count (schedule : VisitSchedule) (totalDays : Nat) :
  schedule.alice = 5 →
  schedule.beatrix = 6 →
  schedule.claire = 8 →
  totalDays = 400 →
  exactlyTwoVisits schedule totalDays = 39 :=
sorry

end NUMINAMATH_CALUDE_two_visits_count_l3614_361475


namespace NUMINAMATH_CALUDE_intersection_point_l3614_361414

def line1 (x y : ℚ) : Prop := 8 * x - 5 * y = 4
def line2 (x y : ℚ) : Prop := 6 * x + 2 * y = 18

theorem intersection_point :
  ∃! p : ℚ × ℚ, line1 p.1 p.2 ∧ line2 p.1 p.2 ∧ p = (49/23, 60/23) := by sorry

end NUMINAMATH_CALUDE_intersection_point_l3614_361414


namespace NUMINAMATH_CALUDE_angle_order_l3614_361431

-- Define the angles of inclination
variable (α₁ α₂ α₃ : Real)

-- Define the slopes of the lines
def m₁ : Real := 1
def m₂ : Real := -1
def m₃ : Real := -2

-- Define the relationship between angles and slopes
axiom tan_α₁ : Real.tan α₁ = m₁
axiom tan_α₂ : Real.tan α₂ = m₂
axiom tan_α₃ : Real.tan α₃ = m₃

-- Theorem to prove
theorem angle_order : α₁ < α₃ ∧ α₃ < α₂ := by
  sorry

end NUMINAMATH_CALUDE_angle_order_l3614_361431


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l3614_361443

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 32) : 
  1 / x + 1 / y = 3 / 8 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l3614_361443


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_range_l3614_361486

/-- The range of m for which a line y = kx + 1 and an ellipse (x^2)/5 + (y^2)/m = 1 always have common points -/
theorem line_ellipse_intersection_range (k : ℝ) :
  (∀ x y : ℝ, y = k * x + 1 → x^2 / 5 + y^2 / m = 1 → (∃ x' y' : ℝ, y' = k * x' + 1 ∧ x'^2 / 5 + y'^2 / m = 1)) →
  m ≥ 1 ∧ m ≠ 5 :=
sorry

end NUMINAMATH_CALUDE_line_ellipse_intersection_range_l3614_361486


namespace NUMINAMATH_CALUDE_work_completion_time_l3614_361412

/-- Given workers A and B, where:
  * A can complete the entire work in 15 days
  * A works for 5 days and then leaves
  * B completes the remaining work in 6 days
  This theorem proves that B alone can complete the entire work in 9 days -/
theorem work_completion_time (a_total_days b_completion_days : ℕ) 
  (a_worked_days : ℕ) (h1 : a_total_days = 15) (h2 : a_worked_days = 5) 
  (h3 : b_completion_days = 6) : 
  (b_completion_days : ℚ) / ((a_total_days - a_worked_days : ℚ) / a_total_days) = 9 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3614_361412


namespace NUMINAMATH_CALUDE_range_of_z_l3614_361407

theorem range_of_z (x y : ℝ) (h1 : x + 2 ≥ y) (h2 : x + 2*y ≥ 4) (h3 : y ≤ 5 - 2*x) :
  let z := (2*x + y - 1) / (x + 1)
  ∃ (z_min z_max : ℝ), z_min = 1 ∧ z_max = 2 ∧ ∀ z', z' = z → z_min ≤ z' ∧ z' ≤ z_max :=
by sorry

end NUMINAMATH_CALUDE_range_of_z_l3614_361407


namespace NUMINAMATH_CALUDE_last_four_digits_pow_5_2017_l3614_361491

/-- The last four digits of a natural number -/
def lastFourDigits (n : ℕ) : ℕ := n % 10000

/-- The cycle length of the last four digits of powers of 5 -/
def cycleLengthPowersOf5 : ℕ := 4

theorem last_four_digits_pow_5_2017 :
  lastFourDigits (5^2017) = lastFourDigits (5^5) :=
sorry

end NUMINAMATH_CALUDE_last_four_digits_pow_5_2017_l3614_361491


namespace NUMINAMATH_CALUDE_correct_inequalities_count_proof_correct_inequalities_count_l3614_361468

theorem correct_inequalities_count : ℕ :=
  let inequality1 := ∀ a : ℝ, a^2 + 1 ≥ 2*a
  let inequality2 := ∀ x : ℝ, x ≥ 2
  let inequality3 := ∀ x : ℝ, x^2 + x ≥ 1
  2

theorem proof_correct_inequalities_count : 
  (inequality1 → True) ∧ (inequality2 → False) ∧ (inequality3 → True) →
  correct_inequalities_count = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_correct_inequalities_count_proof_correct_inequalities_count_l3614_361468


namespace NUMINAMATH_CALUDE_square_plus_double_eq_one_implies_double_square_plus_quad_minus_one_eq_one_l3614_361419

/-- Given that a^2 + 2a = 1, prove that 2a^2 + 4a - 1 = 1 -/
theorem square_plus_double_eq_one_implies_double_square_plus_quad_minus_one_eq_one
  (a : ℝ) (h : a^2 + 2*a = 1) : 2*a^2 + 4*a - 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_double_eq_one_implies_double_square_plus_quad_minus_one_eq_one_l3614_361419


namespace NUMINAMATH_CALUDE_exists_polygon_with_n_axes_of_symmetry_l3614_361403

/-- A convex polygon. -/
structure ConvexPolygon where
  -- Add necessary fields here
  -- This is a placeholder definition

/-- The number of axes of symmetry of a convex polygon. -/
def axesOfSymmetry (p : ConvexPolygon) : ℕ :=
  sorry -- Placeholder definition

/-- For any natural number n, there exists a convex polygon with exactly n axes of symmetry. -/
theorem exists_polygon_with_n_axes_of_symmetry :
  ∀ n : ℕ, ∃ p : ConvexPolygon, axesOfSymmetry p = n :=
sorry

end NUMINAMATH_CALUDE_exists_polygon_with_n_axes_of_symmetry_l3614_361403


namespace NUMINAMATH_CALUDE_machine_selling_price_l3614_361423

/-- Calculates the selling price of a machine given its costs and desired profit percentage --/
def selling_price (purchase_price repair_cost transportation_charges profit_percentage : ℕ) : ℕ :=
  let total_cost := purchase_price + repair_cost + transportation_charges
  let profit := total_cost * profit_percentage / 100
  total_cost + profit

/-- Theorem stating that the selling price of the machine is 27000 --/
theorem machine_selling_price :
  selling_price 12000 5000 1000 50 = 27000 := by
  sorry

end NUMINAMATH_CALUDE_machine_selling_price_l3614_361423


namespace NUMINAMATH_CALUDE_johnny_pencil_packs_l3614_361463

theorem johnny_pencil_packs :
  ∀ (total_red_pencils : ℕ) (extra_red_packs : ℕ) (extra_red_per_pack : ℕ),
    total_red_pencils = 21 →
    extra_red_packs = 3 →
    extra_red_per_pack = 2 →
    ∃ (total_packs : ℕ),
      total_packs = (total_red_pencils - extra_red_packs * extra_red_per_pack) + extra_red_packs ∧
      total_packs = 18 :=
by sorry

end NUMINAMATH_CALUDE_johnny_pencil_packs_l3614_361463


namespace NUMINAMATH_CALUDE_dagger_example_l3614_361410

-- Define the † operation
def dagger (m n p q : ℚ) : ℚ := m * p * (n / q)

-- Theorem statement
theorem dagger_example : dagger (5/9) (12/4) = 135 := by
  sorry

end NUMINAMATH_CALUDE_dagger_example_l3614_361410


namespace NUMINAMATH_CALUDE_max_intersections_circles_lines_l3614_361445

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in a 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Function to count the number of intersection points -/
def count_intersections (circles : List Circle) (lines : List Line) : ℕ :=
  sorry

/-- Main theorem statement -/
theorem max_intersections_circles_lines :
  ∀ (circles : List Circle) (lines : List Line),
    circles.length = 2 →
    lines.length = 3 →
    (∃ (l : Line) (c : Circle), l ∈ lines ∧ c ∈ circles ∧
      (∀ (c' : Circle) (l' : Line), c' ∈ circles → l' ∈ lines →
        c' ≠ c → l' ≠ l → ¬ (count_intersections [c'] [l'] > 0))) →
    count_intersections circles lines ≤ 12 :=
  sorry

end NUMINAMATH_CALUDE_max_intersections_circles_lines_l3614_361445


namespace NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l3614_361413

theorem square_sum_zero_implies_both_zero (a b : ℝ) : a^2 + b^2 = 0 → a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l3614_361413


namespace NUMINAMATH_CALUDE_rectangle_triangle_max_area_and_hypotenuse_l3614_361469

theorem rectangle_triangle_max_area_and_hypotenuse (x y : ℝ) :
  x > 0 → y > 0 →  -- rectangle has positive dimensions
  x + y = 30 →     -- half the perimeter is 30
  (∃ h : ℝ, h^2 = x^2 + y^2) →  -- it's a right triangle
  x * y ≤ 225 ∧    -- max area is 225
  (x * y = 225 → ∃ h : ℝ, h^2 = x^2 + y^2 ∧ h = 15 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_triangle_max_area_and_hypotenuse_l3614_361469


namespace NUMINAMATH_CALUDE_max_product_sum_constant_l3614_361438

theorem max_product_sum_constant (a b M : ℝ) : 
  a > 0 → b > 0 → a + b = M → (∀ x y : ℝ, x > 0 → y > 0 → x + y = M → x * y ≤ 2) → M = 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_constant_l3614_361438


namespace NUMINAMATH_CALUDE_right_triangle_check_triangle_sets_check_l3614_361455

theorem right_triangle_check (a b c : ℝ) : Prop :=
  (a * a + b * b = c * c) ∨ (a * a + c * c = b * b) ∨ (b * b + c * c = a * a)

theorem triangle_sets_check : 
  right_triangle_check 1 (Real.sqrt 2) (Real.sqrt 3) ∧
  right_triangle_check 6 8 10 ∧
  right_triangle_check 5 12 13 ∧
  ¬(right_triangle_check (Real.sqrt 3) 2 (Real.sqrt 5)) := by
sorry

end NUMINAMATH_CALUDE_right_triangle_check_triangle_sets_check_l3614_361455


namespace NUMINAMATH_CALUDE_circle_intersection_theorem_l3614_361480

-- Define the circle
def Circle (r : ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = r^2}

-- Define the line
def Line := {p : ℝ × ℝ | p.2 = -p.1 + 2}

-- Define the origin
def O : ℝ × ℝ := (0, 0)

-- Theorem statement
theorem circle_intersection_theorem (r : ℝ) 
  (A B C : ℝ × ℝ) 
  (hA : A ∈ Circle r ∩ Line) 
  (hB : B ∈ Circle r ∩ Line) 
  (hC : C ∈ Circle r) 
  (hOC : C.1 * C.1 + C.2 * C.2 = (5/4 * A.1 + 3/4 * B.1)^2 + (5/4 * A.2 + 3/4 * B.2)^2) :
  r = Real.sqrt 10 := by
  sorry


end NUMINAMATH_CALUDE_circle_intersection_theorem_l3614_361480


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3614_361446

theorem quadratic_inequality (x : ℝ) : x^2 - 3*x - 18 < 0 ↔ -3 < x ∧ x < 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3614_361446


namespace NUMINAMATH_CALUDE_max_pencil_length_in_square_hallway_l3614_361464

/-- Represents the length of a pencil that can navigate a square turn in a hallway -/
def max_pencil_length (L : ℝ) : ℝ := 3 * L

/-- Theorem stating that the maximum length of a pencil that can navigate a square turn
    in a hallway of width and height L is 3L -/
theorem max_pencil_length_in_square_hallway (L : ℝ) (h : L > 0) :
  max_pencil_length L = 3 * L :=
by sorry

end NUMINAMATH_CALUDE_max_pencil_length_in_square_hallway_l3614_361464


namespace NUMINAMATH_CALUDE_product_of_solutions_abs_y_eq_3_abs_y_minus_2_l3614_361429

theorem product_of_solutions_abs_y_eq_3_abs_y_minus_2 :
  ∃ (y₁ y₂ : ℝ), (|y₁| = 3 * (|y₁| - 2)) ∧ (|y₂| = 3 * (|y₂| - 2)) ∧ y₁ ≠ y₂ ∧ y₁ * y₂ = -9 :=
sorry

end NUMINAMATH_CALUDE_product_of_solutions_abs_y_eq_3_abs_y_minus_2_l3614_361429


namespace NUMINAMATH_CALUDE_largest_angle_in_special_quadrilateral_l3614_361400

/-- A quadrilateral with angles in the ratio 3:4:5:6 has its largest angle equal to 120°. -/
theorem largest_angle_in_special_quadrilateral : 
  ∀ (a b c d : ℝ), 
  a > 0 → b > 0 → c > 0 → d > 0 →
  (a + b + c + d = 360) →
  (b = 4/3 * a) → (c = 5/3 * a) → (d = 2 * a) →
  d = 120 := by
sorry

end NUMINAMATH_CALUDE_largest_angle_in_special_quadrilateral_l3614_361400


namespace NUMINAMATH_CALUDE_six_digit_number_divisibility_l3614_361488

theorem six_digit_number_divisibility (W : ℕ) :
  (100000 ≤ W) ∧ (W < 1000000) ∧
  (∃ a b c : ℕ, a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧
    W = 100000*a + 10000*b + 1000*c + 200*a + 20*b + 2*c) →
  2 ∣ W :=
by sorry

end NUMINAMATH_CALUDE_six_digit_number_divisibility_l3614_361488


namespace NUMINAMATH_CALUDE_harper_mineral_water_cost_l3614_361493

/-- Harper's mineral water purchase problem -/
theorem harper_mineral_water_cost 
  (daily_consumption : ℚ) 
  (bottles_per_case : ℕ) 
  (cost_per_case : ℚ) 
  (days : ℕ) : 
  daily_consumption = 1/2 → 
  bottles_per_case = 24 → 
  cost_per_case = 12 → 
  days = 240 → 
  (days * daily_consumption / bottles_per_case).ceil * cost_per_case = 60 := by
  sorry

end NUMINAMATH_CALUDE_harper_mineral_water_cost_l3614_361493


namespace NUMINAMATH_CALUDE_base6_addition_l3614_361452

/-- Converts a base 6 number to base 10 --/
def base6_to_base10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- Converts a base 10 number to base 6 --/
def base10_to_base6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc
    else aux (m / 6) ((m % 6) :: acc)
  aux n []

/-- Addition in base 6 --/
def add_base6 (a b : List Nat) : List Nat :=
  base10_to_base6 (base6_to_base10 a + base6_to_base10 b)

theorem base6_addition :
  add_base6 [2, 5, 4, 1] [4, 5, 3, 2] = [0, 5, 2, 4] := by sorry

end NUMINAMATH_CALUDE_base6_addition_l3614_361452


namespace NUMINAMATH_CALUDE_impossible_tiling_l3614_361472

/-- Represents a tile type -/
inductive TileType
| TwoByTwo
| OneByFour

/-- Represents a set of tiles -/
structure TileSet where
  twoByTwo : Nat
  oneByFour : Nat

/-- Represents a rectangular box -/
structure Box where
  length : Nat
  width : Nat

/-- Checks if a box can be tiled with a given tile set -/
def canTile (box : Box) (tiles : TileSet) : Prop :=
  sorry

/-- The main theorem -/
theorem impossible_tiling (box : Box) (initialTiles : TileSet) :
  canTile box initialTiles →
  ¬canTile box { twoByTwo := initialTiles.twoByTwo - 1, oneByFour := initialTiles.oneByFour + 1 } :=
sorry

end NUMINAMATH_CALUDE_impossible_tiling_l3614_361472


namespace NUMINAMATH_CALUDE_parallel_angles_theorem_l3614_361436

/-- Two angles in space with parallel sides --/
structure ParallelAngles where
  α : Real
  β : Real
  sides_parallel : Bool

/-- The theorem stating that if two angles have parallel sides and one is 30°, the other is either 30° or 150° --/
theorem parallel_angles_theorem (angles : ParallelAngles) 
  (h1 : angles.sides_parallel = true) 
  (h2 : angles.α = 30) : 
  angles.β = 30 ∨ angles.β = 150 := by
  sorry

end NUMINAMATH_CALUDE_parallel_angles_theorem_l3614_361436


namespace NUMINAMATH_CALUDE_small_boxes_count_l3614_361473

theorem small_boxes_count (total_chocolates : ℕ) (chocolates_per_box : ℕ) 
  (h1 : total_chocolates = 504) 
  (h2 : chocolates_per_box = 28) : 
  total_chocolates / chocolates_per_box = 18 := by
  sorry

#check small_boxes_count

end NUMINAMATH_CALUDE_small_boxes_count_l3614_361473


namespace NUMINAMATH_CALUDE_single_point_equation_l3614_361451

/-- 
Theorem: If the equation 3x^2 + 4y^2 + 12x - 16y + d = 0 represents a single point, then d = 28.
-/
theorem single_point_equation (d : ℝ) : 
  (∃! p : ℝ × ℝ, 3 * p.1^2 + 4 * p.2^2 + 12 * p.1 - 16 * p.2 + d = 0) → 
  d = 28 := by
  sorry

end NUMINAMATH_CALUDE_single_point_equation_l3614_361451


namespace NUMINAMATH_CALUDE_third_angle_relationship_l3614_361479

theorem third_angle_relationship (a b c : ℝ) : 
  a = b → a = 36 → a + b + c = 180 → c = 3 * a := by sorry

end NUMINAMATH_CALUDE_third_angle_relationship_l3614_361479


namespace NUMINAMATH_CALUDE_bark_ratio_is_two_to_one_l3614_361411

/-- The number of times the terrier's owner says "hush" -/
def hush_count : ℕ := 6

/-- The number of times the poodle barks -/
def poodle_barks : ℕ := 24

/-- The number of times the terrier barks before being hushed -/
def terrier_barks_per_hush : ℕ := 2

/-- Calculates the total number of times the terrier barks -/
def total_terrier_barks : ℕ := hush_count * terrier_barks_per_hush

/-- The ratio of poodle barks to terrier barks -/
def bark_ratio : ℚ := poodle_barks / total_terrier_barks

theorem bark_ratio_is_two_to_one : bark_ratio = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_bark_ratio_is_two_to_one_l3614_361411


namespace NUMINAMATH_CALUDE_circles_intersect_example_l3614_361418

/-- Two circles are intersecting if the distance between their centers is less than the sum of their radii
    and greater than the absolute difference of their radii. -/
def circles_intersect (r₁ r₂ d : ℝ) : Prop :=
  d < r₁ + r₂ ∧ d > |r₁ - r₂|

/-- Theorem: Two circles with radii 4 and 5, whose centers are 7 units apart, are intersecting. -/
theorem circles_intersect_example : circles_intersect 4 5 7 := by
  sorry


end NUMINAMATH_CALUDE_circles_intersect_example_l3614_361418


namespace NUMINAMATH_CALUDE_power_fraction_simplification_l3614_361466

theorem power_fraction_simplification : (4 : ℝ)^800 / (8 : ℝ)^400 = (2 : ℝ)^400 := by sorry

end NUMINAMATH_CALUDE_power_fraction_simplification_l3614_361466


namespace NUMINAMATH_CALUDE_ice_cream_parlor_distance_l3614_361458

/-- The distance to the ice cream parlor -/
def D : ℝ := sorry

/-- Rita's upstream paddling speed -/
def upstream_speed : ℝ := 3

/-- Rita's downstream paddling speed -/
def downstream_speed : ℝ := 9

/-- Upstream wind speed -/
def upstream_wind : ℝ := 2

/-- Downstream wind speed -/
def downstream_wind : ℝ := 4

/-- Total trip time -/
def total_time : ℝ := 8

/-- Effective upstream speed -/
def effective_upstream_speed : ℝ := upstream_speed - upstream_wind

/-- Effective downstream speed -/
def effective_downstream_speed : ℝ := downstream_speed + downstream_wind

theorem ice_cream_parlor_distance : 
  D / effective_upstream_speed + D / effective_downstream_speed = total_time := by sorry

end NUMINAMATH_CALUDE_ice_cream_parlor_distance_l3614_361458


namespace NUMINAMATH_CALUDE_shirt_cost_l3614_361461

theorem shirt_cost (initial_amount : ℕ) (change : ℕ) (shirt_cost : ℕ) : 
  initial_amount = 50 → change = 23 → shirt_cost = initial_amount - change → shirt_cost = 27 := by
sorry

end NUMINAMATH_CALUDE_shirt_cost_l3614_361461


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l3614_361428

theorem consecutive_integers_sum (x : ℤ) : x * (x + 1) = 440 → x + (x + 1) = 43 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l3614_361428


namespace NUMINAMATH_CALUDE_heart_op_calculation_l3614_361456

def heart_op (a b : ℤ) : ℤ := Int.natAbs (a^2 - b^2)

theorem heart_op_calculation : heart_op 3 (heart_op 2 5) = 432 := by
  sorry

end NUMINAMATH_CALUDE_heart_op_calculation_l3614_361456


namespace NUMINAMATH_CALUDE_only_class_math_scores_comprehensive_l3614_361465

/-- Represents a survey scenario -/
inductive SurveyScenario
  | NationwideVision
  | LightBulbLifespan
  | ClassMathScores
  | DistrictIncome

/-- Determines if a survey scenario is suitable for a comprehensive survey -/
def isSuitableForComprehensiveSurvey (scenario : SurveyScenario) : Prop :=
  match scenario with
  | .ClassMathScores => true
  | _ => false

/-- The main theorem stating that only ClassMathScores is suitable for a comprehensive survey -/
theorem only_class_math_scores_comprehensive :
  ∀ (scenario : SurveyScenario),
    isSuitableForComprehensiveSurvey scenario ↔ scenario = SurveyScenario.ClassMathScores :=
by
  sorry

/-- Helper lemma: NationwideVision is not suitable for a comprehensive survey -/
lemma nationwide_vision_not_comprehensive :
  ¬ isSuitableForComprehensiveSurvey SurveyScenario.NationwideVision :=
by
  sorry

/-- Helper lemma: LightBulbLifespan is not suitable for a comprehensive survey -/
lemma light_bulb_lifespan_not_comprehensive :
  ¬ isSuitableForComprehensiveSurvey SurveyScenario.LightBulbLifespan :=
by
  sorry

/-- Helper lemma: DistrictIncome is not suitable for a comprehensive survey -/
lemma district_income_not_comprehensive :
  ¬ isSuitableForComprehensiveSurvey SurveyScenario.DistrictIncome :=
by
  sorry

/-- Helper lemma: ClassMathScores is suitable for a comprehensive survey -/
lemma class_math_scores_comprehensive :
  isSuitableForComprehensiveSurvey SurveyScenario.ClassMathScores :=
by
  sorry

end NUMINAMATH_CALUDE_only_class_math_scores_comprehensive_l3614_361465


namespace NUMINAMATH_CALUDE_simple_interest_calculation_l3614_361447

/-- Given a principal amount P, prove that if the compound interest at 4% for 2 years is $612,
    then the simple interest at 4% for 2 years is $600. -/
theorem simple_interest_calculation (P : ℝ) : 
  P * (1 + 0.04)^2 - P = 612 → P * 0.04 * 2 = 600 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_calculation_l3614_361447


namespace NUMINAMATH_CALUDE_parabola_vertex_x_coordinate_l3614_361470

/-- Proves that for a quadratic function y = ax^2 + bx + c passing through points 
(-2,8), (4,8), and (7,15), the x-coordinate of the vertex is 1. -/
theorem parabola_vertex_x_coordinate (a b c : ℝ) : 
  (8 = a * (-2)^2 + b * (-2) + c) →
  (8 = a * 4^2 + b * 4 + c) →
  (15 = a * 7^2 + b * 7 + c) →
  (∃ (x : ℝ), x = 1 ∧ ∀ (t : ℝ), a * t^2 + b * t + c ≥ a * x^2 + b * x + c) :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_x_coordinate_l3614_361470


namespace NUMINAMATH_CALUDE_mary_sheep_count_l3614_361478

theorem mary_sheep_count : ∃ (m : ℕ), 
  (∀ (b : ℕ), b = 2 * m + 35 →
    m + 266 = b - 69) → m = 300 := by
  sorry

end NUMINAMATH_CALUDE_mary_sheep_count_l3614_361478


namespace NUMINAMATH_CALUDE_inequality_proof_l3614_361430

theorem inequality_proof (S a b c x y z : ℝ) 
  (hS : S > 0)
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : a + x = S) (eq2 : b + y = S) (eq3 : c + z = S) : 
  a * y + b * z + c * x < S^2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3614_361430


namespace NUMINAMATH_CALUDE_sin_75_degrees_l3614_361409

theorem sin_75_degrees : Real.sin (75 * π / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_75_degrees_l3614_361409


namespace NUMINAMATH_CALUDE_complementary_angle_of_37_38_l3614_361496

/-- Represents an angle in degrees and minutes -/
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)

/-- The sum of two angles in degrees and minutes -/
def Angle.add (a b : Angle) : Angle :=
  let totalMinutes := a.minutes + b.minutes
  let carryDegrees := totalMinutes / 60
  { degrees := a.degrees + b.degrees + carryDegrees
  , minutes := totalMinutes % 60 }

/-- Checks if two angles are complementary -/
def are_complementary (a b : Angle) : Prop :=
  Angle.add a b = ⟨90, 0⟩

/-- The main theorem statement -/
theorem complementary_angle_of_37_38 :
  let angle : Angle := ⟨37, 38⟩
  let complement : Angle := ⟨52, 22⟩
  are_complementary angle complement :=
by sorry

end NUMINAMATH_CALUDE_complementary_angle_of_37_38_l3614_361496


namespace NUMINAMATH_CALUDE_good_carrots_count_l3614_361497

theorem good_carrots_count (nancy_carrots : ℕ) (mom_carrots : ℕ) (bad_carrots : ℕ) :
  nancy_carrots = 38 →
  mom_carrots = 47 →
  bad_carrots = 14 →
  nancy_carrots + mom_carrots - bad_carrots = 71 := by
sorry

end NUMINAMATH_CALUDE_good_carrots_count_l3614_361497


namespace NUMINAMATH_CALUDE_mango_crates_sold_l3614_361462

-- Define the types of fruit
inductive Fruit
  | Grapes
  | Mangoes
  | PassionFruits

-- Define the total number of crates sold
def total_crates : ℕ := 50

-- Define the number of grape crates sold
def grape_crates : ℕ := 13

-- Define the number of passion fruit crates sold
def passion_fruit_crates : ℕ := 17

-- Define the function to calculate the number of mango crates
def mango_crates : ℕ := total_crates - (grape_crates + passion_fruit_crates)

-- Theorem statement
theorem mango_crates_sold : mango_crates = 20 := by
  sorry

end NUMINAMATH_CALUDE_mango_crates_sold_l3614_361462


namespace NUMINAMATH_CALUDE_maintenance_check_interval_l3614_361449

theorem maintenance_check_interval (original : ℝ) (new : ℝ) : 
  new = 1.5 * original → new = 45 → original = 30 := by
  sorry

end NUMINAMATH_CALUDE_maintenance_check_interval_l3614_361449


namespace NUMINAMATH_CALUDE_partnership_problem_l3614_361471

/-- Partnership problem -/
theorem partnership_problem (a_months b_months : ℕ) (b_contribution total_profit a_share : ℝ) 
  (h1 : a_months = 8)
  (h2 : b_months = 5)
  (h3 : b_contribution = 6000)
  (h4 : total_profit = 8400)
  (h5 : a_share = 4800) :
  ∃ (a_contribution : ℝ),
    a_contribution * a_months * (total_profit - a_share) = 
    b_contribution * b_months * a_share ∧ 
    a_contribution = 5000 := by
  sorry

end NUMINAMATH_CALUDE_partnership_problem_l3614_361471


namespace NUMINAMATH_CALUDE_tyler_puppies_l3614_361487

/-- The number of puppies Tyler has, given the number of dogs and puppies per dog. -/
def total_puppies (num_dogs : ℕ) (puppies_per_dog : ℕ) : ℕ :=
  num_dogs * puppies_per_dog

/-- Theorem stating that Tyler has 75 puppies given 15 dogs with 5 puppies each. -/
theorem tyler_puppies : total_puppies 15 5 = 75 := by
  sorry

end NUMINAMATH_CALUDE_tyler_puppies_l3614_361487


namespace NUMINAMATH_CALUDE_seating_arrangements_seven_people_l3614_361422

/-- The number of ways to arrange n people around a circular table, considering rotations as identical -/
def circularArrangements (n : ℕ) : ℕ := (n - 1).factorial

/-- The number of ways to arrange n blocks around a circular table, with one block containing 3 fixed people -/
def arrangementsWithFixedBlock (n : ℕ) : ℕ := 
  circularArrangements (n - 2) * 2

theorem seating_arrangements_seven_people : 
  arrangementsWithFixedBlock 7 = 240 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_seven_people_l3614_361422


namespace NUMINAMATH_CALUDE_distance_proof_l3614_361420

/-- The distance between points A and B in kilometers -/
def distance : ℝ := 180

/-- The total travel time in hours -/
def total_time : ℝ := 19

/-- The velocity of the stream in km/h -/
def stream_velocity : ℝ := 4

/-- The speed of the boat in still water in km/h -/
def boat_speed : ℝ := 14

/-- The downstream speed of the boat in km/h -/
def downstream_speed : ℝ := boat_speed + stream_velocity

/-- The upstream speed of the boat in km/h -/
def upstream_speed : ℝ := boat_speed - stream_velocity

theorem distance_proof :
  distance / downstream_speed + (distance / 2) / upstream_speed = total_time :=
sorry

end NUMINAMATH_CALUDE_distance_proof_l3614_361420


namespace NUMINAMATH_CALUDE_parabola_c_value_l3614_361404

/-- Represents a parabola of the form x = ay² + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.a * y^2 + p.b * y + p.c

theorem parabola_c_value (p : Parabola) :
  p.x_coord 1 = -3 →  -- vertex at (-3, 1)
  p.x_coord 3 = -1 →  -- passes through (-1, 3)
  p.c = -5/2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l3614_361404


namespace NUMINAMATH_CALUDE_math_competition_problem_l3614_361417

theorem math_competition_problem (p_a p_either : ℝ) (h1 : p_a = 0.6) (h2 : p_either = 0.92) :
  ∃ p_b : ℝ, p_b = 0.8 ∧ 1 - p_either = (1 - p_a) * (1 - p_b) :=
by sorry

end NUMINAMATH_CALUDE_math_competition_problem_l3614_361417


namespace NUMINAMATH_CALUDE_triangle_exists_for_all_x_l3614_361482

/-- Represents an equilateral triangle with points D, E, F on its sides -/
structure TriangleWithPoints where
  -- Side length of the equilateral triangle
  side : ℝ
  -- Position of point D on side BC
  d : ℝ
  -- Position of point E on side CA
  e : ℝ
  -- Position of point F on side AB
  f : ℝ
  -- Ensure D is on side BC
  h_d : d ≥ 0 ∧ d ≤ side
  -- Ensure E is on side CA
  h_e : e ≥ 0 ∧ e ≤ side
  -- Ensure F is on side AB
  h_f : f ≥ 0 ∧ f ≤ side
  -- Ensure D, E, F form a straight line
  h_straight : d + e + f = side

/-- The main theorem stating that for any real x, there exists a valid triangle configuration -/
theorem triangle_exists_for_all_x (x : ℝ) : 
  ∃ t : TriangleWithPoints, 
    t.d = 4 ∧ 
    t.side - t.d = 2*x ∧ 
    t.e = x + 5 ∧ 
    t.side - t.e - t.f = 3 ∧ 
    t.f = 7 + x :=
  sorry

end NUMINAMATH_CALUDE_triangle_exists_for_all_x_l3614_361482


namespace NUMINAMATH_CALUDE_find_m_find_k_l3614_361481

-- Define the vectors
def a : ℝ × ℝ := (1, -3)
def b (m : ℝ) : ℝ × ℝ := (-2, m)

-- Define dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define vector subtraction
def vec_sub (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)

-- Define vector addition
def vec_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)

-- Define scalar multiplication
def scalar_mul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)

-- Define parallel vectors
def parallel (v w : ℝ × ℝ) : Prop := ∃ (k : ℝ), v = scalar_mul k w

-- Theorem 1: Find the value of m
theorem find_m :
  ∃ (m : ℝ), dot_product a (vec_sub a (b m)) = 0 ∧ m = -4 :=
sorry

-- Theorem 2: Find the value of k
theorem find_k :
  ∃ (k : ℝ), parallel (vec_add (scalar_mul k a) (b (-4))) (vec_sub a (b (-4))) ∧ k = -1 :=
sorry

end NUMINAMATH_CALUDE_find_m_find_k_l3614_361481


namespace NUMINAMATH_CALUDE_impossible_coin_probabilities_l3614_361490

theorem impossible_coin_probabilities :
  ¬ ∃ (p₁ p₂ : ℝ), 0 ≤ p₁ ∧ p₁ ≤ 1 ∧ 0 ≤ p₂ ∧ p₂ ≤ 1 ∧
    (1 - p₁) * (1 - p₂) = p₁ * p₂ ∧
    p₁ * p₂ = p₁ * (1 - p₂) + p₂ * (1 - p₁) :=
by sorry

end NUMINAMATH_CALUDE_impossible_coin_probabilities_l3614_361490


namespace NUMINAMATH_CALUDE_like_terms_imply_exponent_one_l3614_361444

theorem like_terms_imply_exponent_one (a b : ℝ) (m n x : ℕ) :
  (∃ (k : ℝ), 2 * a^x * b^(n+1) = k * (-3 * a * b^(2*m))) →
  (2*m - n)^x = 1 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_imply_exponent_one_l3614_361444


namespace NUMINAMATH_CALUDE_quadratic_roots_d_value_l3614_361459

theorem quadratic_roots_d_value (d : ℝ) : 
  (∀ x : ℝ, x^2 + 7*x + d = 0 ↔ x = (-7 + Real.sqrt d) / 2 ∨ x = (-7 - Real.sqrt d) / 2) →
  d = 9.8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_d_value_l3614_361459


namespace NUMINAMATH_CALUDE_circle_center_radius_l3614_361498

theorem circle_center_radius (x y : ℝ) :
  x^2 + y^2 - 4*x = 0 → (∃ (center : ℝ × ℝ) (radius : ℝ), 
    center = (2, 0) ∧ radius = 2 ∧ 
    (x - center.1)^2 + (y - center.2)^2 = radius^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_radius_l3614_361498


namespace NUMINAMATH_CALUDE_largest_three_digit_congruent_to_12_mod_15_l3614_361499

theorem largest_three_digit_congruent_to_12_mod_15 : ∃ n : ℕ,
  n = 987 ∧
  n ≥ 100 ∧ n < 1000 ∧
  n % 15 = 12 ∧
  ∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ m % 15 = 12 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_congruent_to_12_mod_15_l3614_361499


namespace NUMINAMATH_CALUDE_price_reduction_proof_optimal_price_increase_proof_l3614_361421

/-- Initial price in yuan per kilogram -/
def initial_price : ℝ := 50

/-- Final price after two reductions in yuan per kilogram -/
def final_price : ℝ := 32

/-- Initial profit in yuan per kilogram -/
def initial_profit : ℝ := 10

/-- Initial daily sales in kilograms -/
def initial_sales : ℝ := 500

/-- Maximum allowed price increase in yuan per kilogram -/
def max_price_increase : ℝ := 8

/-- Sales decrease per yuan of price increase in kilograms -/
def sales_decrease_rate : ℝ := 20

/-- Target daily profit in yuan -/
def target_profit : ℝ := 6000

/-- Percentage reduction after each price cut -/
def reduction_percentage : ℝ := 0.2

/-- Price increase to achieve target profit -/
def optimal_price_increase : ℝ := 5

theorem price_reduction_proof :
  initial_price * (1 - reduction_percentage)^2 = final_price :=
sorry

theorem optimal_price_increase_proof :
  (initial_profit + optimal_price_increase) * 
  (initial_sales - sales_decrease_rate * optimal_price_increase) = target_profit ∧
  0 < optimal_price_increase ∧
  optimal_price_increase ≤ max_price_increase :=
sorry

end NUMINAMATH_CALUDE_price_reduction_proof_optimal_price_increase_proof_l3614_361421


namespace NUMINAMATH_CALUDE_matthew_cakes_l3614_361405

theorem matthew_cakes (initial_crackers : ℕ) (friends : ℕ) (total_eaten_per_friend : ℕ)
  (h1 : initial_crackers = 14)
  (h2 : friends = 7)
  (h3 : total_eaten_per_friend = 5)
  (h4 : initial_crackers / friends = initial_crackers % friends) :
  ∃ initial_cakes : ℕ, initial_cakes = 21 := by
  sorry

end NUMINAMATH_CALUDE_matthew_cakes_l3614_361405


namespace NUMINAMATH_CALUDE_geordie_commute_cost_l3614_361454

/-- Represents the cost calculation for Geordie's weekly commute -/
def weekly_commute_cost (car_toll : ℚ) (motorcycle_toll : ℚ) (mpg : ℚ) (distance : ℚ) (gas_price : ℚ) (car_trips : ℕ) (motorcycle_trips : ℕ) : ℚ :=
  let total_toll := car_toll * car_trips + motorcycle_toll * motorcycle_trips
  let total_miles := (distance * 2) * (car_trips + motorcycle_trips)
  let total_gas_cost := (total_miles / mpg) * gas_price
  total_toll + total_gas_cost

/-- Theorem stating that Geordie's weekly commute cost is $66.50 -/
theorem geordie_commute_cost :
  weekly_commute_cost 12.5 7 35 14 3.75 3 2 = 66.5 := by
  sorry

end NUMINAMATH_CALUDE_geordie_commute_cost_l3614_361454


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3614_361433

theorem complex_fraction_simplification :
  (5 + 6 * Complex.I) / (2 - 3 * Complex.I) = (-8 : ℚ) / 13 + (27 : ℚ) / 13 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3614_361433


namespace NUMINAMATH_CALUDE_shirley_eggs_left_shirley_eggs_problem_l3614_361492

theorem shirley_eggs_left (initial_eggs : ℕ) (bought_eggs : ℕ) 
  (eggs_per_cupcake_batch1 : ℕ) (eggs_per_cupcake_batch2 : ℕ)
  (cupcakes_batch1 : ℕ) (cupcakes_batch2 : ℕ) : ℕ :=
  let total_eggs := initial_eggs + bought_eggs
  let eggs_used_batch1 := eggs_per_cupcake_batch1 * cupcakes_batch1
  let eggs_used_batch2 := eggs_per_cupcake_batch2 * cupcakes_batch2
  let total_eggs_used := eggs_used_batch1 + eggs_used_batch2
  total_eggs - total_eggs_used

theorem shirley_eggs_problem :
  shirley_eggs_left 98 8 5 7 6 4 = 48 := by
  sorry

end NUMINAMATH_CALUDE_shirley_eggs_left_shirley_eggs_problem_l3614_361492


namespace NUMINAMATH_CALUDE_cat_head_start_l3614_361494

/-- Proves that given a rabbit with speed 25 mph and a cat with speed 20 mph,
    if the rabbit catches up to the cat in 1 hour, then the cat's head start is 15 minutes. -/
theorem cat_head_start (rabbit_speed cat_speed : ℝ) (catch_up_time : ℝ) (head_start : ℝ) :
  rabbit_speed = 25 →
  cat_speed = 20 →
  catch_up_time = 1 →
  rabbit_speed * catch_up_time = cat_speed * (catch_up_time + head_start / 60) →
  head_start = 15 := by
  sorry

#check cat_head_start

end NUMINAMATH_CALUDE_cat_head_start_l3614_361494


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3614_361440

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) (h1 : d ≠ 0) 
  (h2 : ∀ n, a (n + 1) = a n + d) 
  (h3 : ∃ r, (a 3) / (a 2) = r ∧ (a 6) / (a 3) = r) : 
  (a 3) / (a 2) = 3 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3614_361440


namespace NUMINAMATH_CALUDE_alcohol_percentage_first_vessel_l3614_361416

theorem alcohol_percentage_first_vessel
  (vessel1_capacity : ℝ)
  (vessel2_capacity : ℝ)
  (vessel2_alcohol_percentage : ℝ)
  (total_liquid : ℝ)
  (final_vessel_capacity : ℝ)
  (final_mixture_concentration : ℝ)
  (h1 : vessel1_capacity = 2)
  (h2 : vessel2_capacity = 6)
  (h3 : vessel2_alcohol_percentage = 50)
  (h4 : total_liquid = 8)
  (h5 : final_vessel_capacity = 10)
  (h6 : final_mixture_concentration = 37)
  : ∃ (vessel1_alcohol_percentage : ℝ),
    vessel1_alcohol_percentage = 35 ∧
    (vessel1_alcohol_percentage / 100 * vessel1_capacity +
     vessel2_alcohol_percentage / 100 * vessel2_capacity =
     final_mixture_concentration / 100 * final_vessel_capacity) :=
by sorry

end NUMINAMATH_CALUDE_alcohol_percentage_first_vessel_l3614_361416


namespace NUMINAMATH_CALUDE_negation_of_p_l3614_361437

-- Define the proposition p
def p (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) ≥ 0

-- State the theorem
theorem negation_of_p (f : ℝ → ℝ) : ¬(p f) ↔ ∃ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) < 0 :=
  sorry

end NUMINAMATH_CALUDE_negation_of_p_l3614_361437


namespace NUMINAMATH_CALUDE_janice_starting_sentences_l3614_361442

/-- Represents the typing scenario for Janice --/
structure TypingScenario where
  typing_speed : ℕ
  first_session : ℕ
  second_session : ℕ
  third_session : ℕ
  erased_sentences : ℕ
  final_total : ℕ

/-- Calculates the number of sentences Janice started with today --/
def sentences_at_start (scenario : TypingScenario) : ℕ :=
  scenario.final_total - (scenario.typing_speed * (scenario.first_session + scenario.second_session + scenario.third_session) - scenario.erased_sentences)

/-- Theorem stating that Janice started with 258 sentences --/
theorem janice_starting_sentences (scenario : TypingScenario) 
  (h1 : scenario.typing_speed = 6)
  (h2 : scenario.first_session = 20)
  (h3 : scenario.second_session = 15)
  (h4 : scenario.third_session = 18)
  (h5 : scenario.erased_sentences = 40)
  (h6 : scenario.final_total = 536) :
  sentences_at_start scenario = 258 := by
  sorry

#eval sentences_at_start {
  typing_speed := 6,
  first_session := 20,
  second_session := 15,
  third_session := 18,
  erased_sentences := 40,
  final_total := 536
}

end NUMINAMATH_CALUDE_janice_starting_sentences_l3614_361442


namespace NUMINAMATH_CALUDE_product_of_numbers_l3614_361457

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 18) (h2 : x^2 + y^2 = 220) : x * y = 56 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l3614_361457


namespace NUMINAMATH_CALUDE_smallest_b_for_factorization_l3614_361495

theorem smallest_b_for_factorization : ∃ (b : ℕ), 
  (∀ (r s : ℤ), x^2 + b*x + 2016 = (x + r) * (x + s) → b ≥ 90) ∧
  (∃ (r s : ℤ), x^2 + 90*x + 2016 = (x + r) * (x + s)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_for_factorization_l3614_361495


namespace NUMINAMATH_CALUDE_min_value_theorem_l3614_361434

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x - 1|

-- Define the function g
def g (x : ℝ) : ℝ := f x + f (x - 1)

-- State the theorem
theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_sum : m + n = 2) :
  (m^2 + 2) / m + (n^2 + 1) / n ≥ (7 + 2 * Real.sqrt 2) / 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3614_361434


namespace NUMINAMATH_CALUDE_distinct_roots_condition_l3614_361448

-- Define the quadratic equation
def quadratic_equation (x k : ℝ) : Prop :=
  x^2 - 2*(k-1)*x + k^2 - 1 = 0

-- Define the discriminant of the quadratic equation
def discriminant (k : ℝ) : ℝ :=
  (-2*(k-1))^2 - 4*(k^2 - 1)

-- Theorem statement
theorem distinct_roots_condition (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ quadratic_equation x k ∧ quadratic_equation y k) ↔ k < 1 := by
  sorry

end NUMINAMATH_CALUDE_distinct_roots_condition_l3614_361448


namespace NUMINAMATH_CALUDE_count_less_than_one_l3614_361483

def number_list : List ℝ := [0.03, 1.5, -0.2, 0.76]

theorem count_less_than_one : 
  (number_list.filter (λ x => x < 1)).length = 3 := by
  sorry

end NUMINAMATH_CALUDE_count_less_than_one_l3614_361483


namespace NUMINAMATH_CALUDE_range_of_m_l3614_361424

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - x - 2 > 0}
def B : Set ℝ := {x | 3 - |x| ≥ 0}

-- Define set C parameterized by m
def C (m : ℝ) : Set ℝ := {x | (x - m + 1) * (x - 2*m - 1) < 0}

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, (C m ⊆ B) ↔ (m ∈ Set.Icc (-2) 1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l3614_361424


namespace NUMINAMATH_CALUDE_average_marks_combined_l3614_361401

theorem average_marks_combined (n1 n2 : ℕ) (avg1 avg2 : ℝ) :
  n1 = 20 →
  n2 = 50 →
  avg1 = 40 →
  avg2 = 60 →
  let total_marks := n1 * avg1 + n2 * avg2
  let total_students := n1 + n2
  abs ((total_marks / total_students) - 54.29) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_combined_l3614_361401


namespace NUMINAMATH_CALUDE_correct_ranking_l3614_361474

-- Define the colleagues
inductive Colleague
| David
| Emily
| Frank

-- Define the years of service comparison
def has_more_years (a b : Colleague) : Prop := sorry

-- Define the statements
def statement_I : Prop := has_more_years Colleague.Emily Colleague.David ∧ has_more_years Colleague.Emily Colleague.Frank
def statement_II : Prop := ¬(has_more_years Colleague.David Colleague.Emily) ∨ ¬(has_more_years Colleague.David Colleague.Frank)
def statement_III : Prop := has_more_years Colleague.Frank Colleague.David ∨ has_more_years Colleague.Frank Colleague.Emily

-- Theorem to prove
theorem correct_ranking :
  (statement_I ∨ statement_II ∨ statement_III) ∧
  ¬(statement_I ∧ statement_II) ∧
  ¬(statement_I ∧ statement_III) ∧
  ¬(statement_II ∧ statement_III) →
  has_more_years Colleague.David Colleague.Frank ∧
  has_more_years Colleague.Frank Colleague.Emily :=
by sorry

end NUMINAMATH_CALUDE_correct_ranking_l3614_361474


namespace NUMINAMATH_CALUDE_triangle_circle_area_relation_l3614_361489

theorem triangle_circle_area_relation (A B C : ℝ) : 
  -- The triangle is inscribed in a circle
  -- The triangle has side lengths of 20, 21, and 29
  -- A, B, and C are the areas of the three parts outside the triangle
  -- C is the largest area among A, B, and C
  (20 : ℝ)^2 + 21^2 = 29^2 →  -- This ensures it's a right triangle
  A ≥ 0 → B ≥ 0 → C ≥ 0 →
  C ≥ A → C ≥ B →
  -- Prove the relation
  A + B + 210 = C := by
  sorry

end NUMINAMATH_CALUDE_triangle_circle_area_relation_l3614_361489


namespace NUMINAMATH_CALUDE_lottery_winnings_calculation_l3614_361402

/-- Calculates the amount taken home from lottery winnings after tax and processing fee --/
def amountTakenHome (winnings : ℝ) (taxRate : ℝ) (processingFee : ℝ) : ℝ :=
  winnings - (winnings * taxRate) - processingFee

/-- Theorem stating that given specific lottery winnings, tax rate, and processing fee, 
    the amount taken home is $35 --/
theorem lottery_winnings_calculation :
  amountTakenHome 50 0.2 5 = 35 := by
  sorry

end NUMINAMATH_CALUDE_lottery_winnings_calculation_l3614_361402


namespace NUMINAMATH_CALUDE_total_dress_designs_l3614_361435

/-- The number of fabric colors available -/
def num_colors : ℕ := 5

/-- The number of patterns available -/
def num_patterns : ℕ := 4

/-- The number of sleeve types available -/
def num_sleeve_types : ℕ := 3

/-- Theorem stating the total number of possible dress designs -/
theorem total_dress_designs :
  num_colors * num_patterns * num_sleeve_types = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_dress_designs_l3614_361435


namespace NUMINAMATH_CALUDE_sequence_integer_value_l3614_361425

def u (M : ℤ) : ℕ → ℚ
  | 0 => M + 1/2
  | n + 1 => u M n * ⌊u M n⌋

theorem sequence_integer_value (M : ℤ) (h : M ≥ 1) :
  (∃ n : ℕ, ∃ k : ℤ, u M n = k) ↔ M > 1 :=
sorry

end NUMINAMATH_CALUDE_sequence_integer_value_l3614_361425


namespace NUMINAMATH_CALUDE_odd_function_value_l3614_361408

/-- Given a function f(x) = sin(x + φ) + √3 cos(x + φ) where 0 ≤ φ ≤ π,
    if f(x) is an odd function, then f(π/6) = -1 -/
theorem odd_function_value (φ : Real) (h1 : 0 ≤ φ) (h2 : φ ≤ π) :
  let f : Real → Real := λ x => Real.sin (x + φ) + Real.sqrt 3 * Real.cos (x + φ)
  (∀ x, f (-x) = -f x) →
  f (π / 6) = -1 := by
sorry

end NUMINAMATH_CALUDE_odd_function_value_l3614_361408


namespace NUMINAMATH_CALUDE_solve_system_l3614_361415

theorem solve_system (a b c d : ℚ)
  (eq1 : a = 2 * b + c)
  (eq2 : b = 2 * c + d)
  (eq3 : 2 * c = d + a - 1)
  (eq4 : d = a - c) :
  b = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l3614_361415


namespace NUMINAMATH_CALUDE_product_of_fractions_and_root_l3614_361427

theorem product_of_fractions_and_root : 
  (2 : ℝ) / 3 * (3 : ℝ) / 5 * ((4 : ℝ) / 7) ^ (1 / 2) = 4 * Real.sqrt 7 / 35 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_and_root_l3614_361427


namespace NUMINAMATH_CALUDE_range_of_b_l3614_361432

def f (b c x : ℝ) : ℝ := x^2 + b*x + c

theorem range_of_b (b c : ℝ) :
  (∃ x₀ : ℝ, f (f b c x₀) b c = 0 ∧ f b c x₀ ≠ 0) →
  b < 0 ∨ b ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_b_l3614_361432


namespace NUMINAMATH_CALUDE_find_n_l3614_361453

theorem find_n : ∃ n : ℕ, (1/5 : ℝ)^n * (1/4 : ℝ)^18 = 1 / (2 * (10 : ℝ)^35) → n = 35 := by
  sorry

end NUMINAMATH_CALUDE_find_n_l3614_361453


namespace NUMINAMATH_CALUDE_cookies_sold_first_village_l3614_361439

/-- Given the total number of packs sold and the number sold in the second village,
    calculate the number of packs sold in the first village. -/
theorem cookies_sold_first_village 
  (total_packs : ℕ) 
  (second_village_packs : ℕ) 
  (h1 : total_packs = 51) 
  (h2 : second_village_packs = 28) : 
  total_packs - second_village_packs = 23 := by
  sorry

end NUMINAMATH_CALUDE_cookies_sold_first_village_l3614_361439


namespace NUMINAMATH_CALUDE_total_distance_walked_and_run_l3614_361450

/-- Calculates the total distance traveled when walking and running at different rates for different durations. -/
theorem total_distance_walked_and_run 
  (walking_time : ℝ) (walking_rate : ℝ) (running_time : ℝ) (running_rate : ℝ) :
  walking_time = 45 →
  walking_rate = 4 →
  running_time = 30 →
  running_rate = 10 →
  (walking_time / 60) * walking_rate + (running_time / 60) * running_rate = 8 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_walked_and_run_l3614_361450


namespace NUMINAMATH_CALUDE_smallest_odd_divisible_by_three_l3614_361426

theorem smallest_odd_divisible_by_three :
  ∀ n : ℕ, n % 2 = 1 → n % 3 = 0 → n ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_odd_divisible_by_three_l3614_361426


namespace NUMINAMATH_CALUDE_regular_tetrahedron_edges_l3614_361485

/-- A regular tetrahedron is a tetrahedron in which all faces are congruent equilateral triangles. -/
def RegularTetrahedron : Type := sorry

/-- The number of edges in a geometric shape. -/
def num_edges (shape : Type) : ℕ := sorry

/-- Theorem: A regular tetrahedron has 6 edges. -/
theorem regular_tetrahedron_edges : num_edges RegularTetrahedron = 6 := by
  sorry

end NUMINAMATH_CALUDE_regular_tetrahedron_edges_l3614_361485


namespace NUMINAMATH_CALUDE_set_partition_l3614_361406

def S : Set ℝ := {-5/6, 0, -3.5, 1.2, 6}

def N : Set ℝ := {x ∈ S | x < 0}

def NN : Set ℝ := {x ∈ S | x ≥ 0}

theorem set_partition :
  N = {-5/6, -3.5} ∧ NN = {0, 1.2, 6} := by sorry

end NUMINAMATH_CALUDE_set_partition_l3614_361406


namespace NUMINAMATH_CALUDE_min_value_of_a_l3614_361477

theorem min_value_of_a (p : ∃ x₀ : ℝ, |x₀ + 1| + |x₀ - 2| ≤ a) : 
  ∀ b : ℝ, b < 3 → ¬(∃ x₀ : ℝ, |x₀ + 1| + |x₀ - 2| ≤ b) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_a_l3614_361477


namespace NUMINAMATH_CALUDE_angle_from_point_l3614_361441

theorem angle_from_point (a : Real) (h1 : 0 < a ∧ a < π/2) : 
  (∃ (x y : Real), x = 4 * Real.sin 3 ∧ y = -4 * Real.cos 3 ∧ 
   x = 4 * Real.sin a ∧ y = -4 * Real.cos a) → 
  a = 3 - π/2 := by
sorry

end NUMINAMATH_CALUDE_angle_from_point_l3614_361441


namespace NUMINAMATH_CALUDE_quadratic_function_range_l3614_361484

theorem quadratic_function_range (a : ℝ) : 
  (∃ x₀ : ℝ, |x₀^2 + a*x₀ + 1| ≤ 1/4 ∧ |(x₀+1)^2 + a*(x₀+1) + 1| ≤ 1/4) → 
  a ∈ Set.Icc (-Real.sqrt 6) (-2) ∪ Set.Icc 2 (Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l3614_361484
