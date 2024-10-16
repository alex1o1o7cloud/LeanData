import Mathlib

namespace NUMINAMATH_CALUDE_power_expression_l1088_108845

theorem power_expression (m n : ℕ+) (a b : ℝ) 
  (h1 : 9^(m : ℕ) = a) 
  (h2 : 3^(n : ℕ) = b) : 
  3^((2*m + 4*n) : ℕ) = a * b^4 := by
  sorry

end NUMINAMATH_CALUDE_power_expression_l1088_108845


namespace NUMINAMATH_CALUDE_sin_cos_15_deg_l1088_108823

theorem sin_cos_15_deg : 4 * Real.sin (15 * π / 180) * Real.cos (15 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_15_deg_l1088_108823


namespace NUMINAMATH_CALUDE_work_left_after_collaboration_l1088_108898

/-- Represents the fraction of work left after two workers collaborate for a given number of days. -/
def work_left (a_days b_days collab_days : ℕ) : ℚ :=
  1 - (collab_days : ℚ) * (1 / a_days + 1 / b_days)

/-- Theorem stating that if A can complete the work in 15 days and B in 20 days,
    then after working together for 4 days, 8/15 of the work is left. -/
theorem work_left_after_collaboration :
  work_left 15 20 4 = 8/15 := by
  sorry

end NUMINAMATH_CALUDE_work_left_after_collaboration_l1088_108898


namespace NUMINAMATH_CALUDE_simplify_expression_l1088_108887

theorem simplify_expression : (1 : ℝ) / (1 + Real.sqrt 3) * (1 / (1 + Real.sqrt 3)) = 1 - Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1088_108887


namespace NUMINAMATH_CALUDE_simplify_fraction_l1088_108858

theorem simplify_fraction : 
  (1 / ((1 / (Real.sqrt 3 + 1)) + (3 / (Real.sqrt 5 - 2)))) = 
  (2 / (Real.sqrt 3 + 6 * Real.sqrt 5 + 11)) := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1088_108858


namespace NUMINAMATH_CALUDE_fraction_ordering_l1088_108844

theorem fraction_ordering : (12 : ℚ) / 35 < 10 / 29 ∧ 10 / 29 < 6 / 17 := by
  sorry

end NUMINAMATH_CALUDE_fraction_ordering_l1088_108844


namespace NUMINAMATH_CALUDE_cans_per_bag_l1088_108862

theorem cans_per_bag (total_cans : ℕ) (num_bags : ℕ) (h1 : total_cans = 20) (h2 : num_bags = 4) :
  total_cans / num_bags = 5 := by
  sorry

end NUMINAMATH_CALUDE_cans_per_bag_l1088_108862


namespace NUMINAMATH_CALUDE_max_value_expression_l1088_108878

theorem max_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x : ℝ, 3 * (a - x) * (2 * x + Real.sqrt (x^2 + 4 * b^2)) ≤ 3 * a^2 + 12 * b^2) ∧
  (∃ x : ℝ, 3 * (a - x) * (2 * x + Real.sqrt (x^2 + 4 * b^2)) = 3 * a^2 + 12 * b^2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l1088_108878


namespace NUMINAMATH_CALUDE_tea_consumption_discrepancy_l1088_108840

theorem tea_consumption_discrepancy 
  (box_size : ℕ) 
  (cups_per_bag_min cups_per_bag_max : ℕ) 
  (darya_cups marya_cups : ℕ) :
  cups_per_bag_min = 3 →
  cups_per_bag_max = 4 →
  darya_cups = 74 →
  marya_cups = 105 →
  (∃ n : ℕ, n * cups_per_bag_min ≤ darya_cups ∧ darya_cups < (n + 1) * cups_per_bag_min ∧
            n * cups_per_bag_min ≤ marya_cups ∧ marya_cups < (n + 1) * cups_per_bag_min) →
  (∃ m : ℕ, m * cups_per_bag_max ≤ darya_cups ∧ darya_cups < (m + 1) * cups_per_bag_max ∧
            m * cups_per_bag_max ≤ marya_cups ∧ marya_cups < (m + 1) * cups_per_bag_max) →
  False :=
by sorry

end NUMINAMATH_CALUDE_tea_consumption_discrepancy_l1088_108840


namespace NUMINAMATH_CALUDE_yellow_shirts_calculation_l1088_108885

/-- The number of yellow shirts in each pack -/
def yellow_shirts_per_pack : ℕ :=
  let black_packs : ℕ := 3
  let yellow_packs : ℕ := 3
  let black_shirts_per_pack : ℕ := 5
  let total_shirts : ℕ := 21
  (total_shirts - black_packs * black_shirts_per_pack) / yellow_packs

theorem yellow_shirts_calculation :
  yellow_shirts_per_pack = 2 := by
  sorry

end NUMINAMATH_CALUDE_yellow_shirts_calculation_l1088_108885


namespace NUMINAMATH_CALUDE_reciprocal_of_proper_fraction_greater_l1088_108837

theorem reciprocal_of_proper_fraction_greater {a b : ℚ} (h1 : 0 < a) (h2 : a < b) :
  b / a > a / b :=
sorry

end NUMINAMATH_CALUDE_reciprocal_of_proper_fraction_greater_l1088_108837


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1088_108881

/-- A right triangle with perimeter 40 and area 24 has a hypotenuse of length 18.8 -/
theorem right_triangle_hypotenuse : ∃ (a b c : ℝ),
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- positive sides
  a + b + c = 40 ∧  -- perimeter condition
  (1/2) * a * b = 24 ∧  -- area condition
  a^2 + b^2 = c^2 ∧  -- right triangle (Pythagorean theorem)
  c = 18.8 := by
  sorry


end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1088_108881


namespace NUMINAMATH_CALUDE_clapping_theorem_l1088_108802

/-- Represents the clapping pattern of a person -/
structure ClappingPattern where
  interval : ℕ
  start_time : ℕ

/-- Checks if a clapping pattern results in a clap at the given time -/
def claps_at (pattern : ClappingPattern) (time : ℕ) : Prop :=
  ∃ k : ℕ, time = pattern.start_time + k * pattern.interval

theorem clapping_theorem (jirka_start petr_start : ℕ) :
  jirka_start ≤ 15 ∧ petr_start ≤ 15 ∧
  claps_at { interval := 7, start_time := jirka_start } 90 ∧
  claps_at { interval := 13, start_time := petr_start } 90 →
  (jirka_start = 6 ∨ jirka_start = 13) ∧ petr_start = 12 := by
  sorry

#check clapping_theorem

end NUMINAMATH_CALUDE_clapping_theorem_l1088_108802


namespace NUMINAMATH_CALUDE_two_sqrt_six_lt_five_l1088_108820

theorem two_sqrt_six_lt_five : 2 * Real.sqrt 6 < 5 := by
  sorry

end NUMINAMATH_CALUDE_two_sqrt_six_lt_five_l1088_108820


namespace NUMINAMATH_CALUDE_range_of_m_l1088_108874

-- Define propositions p and q as functions of m
def p (m : ℝ) : Prop := (m - 2) / (m - 3) ≤ 2/3

def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 - 4*x + m^2 > 0

-- Define the set of m values that satisfy the conditions
def M : Set ℝ := {m : ℝ | (p m ∨ q m) ∧ ¬(p m ∧ q m)}

-- State the theorem
theorem range_of_m : M = {m : ℝ | m < -2 ∨ (0 ≤ m ∧ m ≤ 2) ∨ m ≥ 3} := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1088_108874


namespace NUMINAMATH_CALUDE_point_on_line_implies_tan_2theta_l1088_108827

theorem point_on_line_implies_tan_2theta (θ : ℝ) : 
  2 * Real.sin θ + Real.cos θ = 0 → Real.tan (2 * θ) = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_implies_tan_2theta_l1088_108827


namespace NUMINAMATH_CALUDE_maria_earnings_l1088_108895

def brush_cost_1 : ℕ := 20
def brush_cost_2 : ℕ := 25
def brush_cost_3 : ℕ := 30
def acrylic_paint_cost : ℕ := 8
def oil_paint_cost : ℕ := 12
def acrylic_paint_amount : ℕ := 5
def oil_paint_amount : ℕ := 3
def selling_price : ℕ := 200

def total_brush_cost : ℕ := brush_cost_1 + brush_cost_2 + brush_cost_3

def canvas_cost_1 : ℕ := 3 * total_brush_cost
def canvas_cost_2 : ℕ := 2 * total_brush_cost

def total_paint_cost : ℕ := acrylic_paint_cost * acrylic_paint_amount + oil_paint_cost * oil_paint_amount

def total_cost : ℕ := total_brush_cost + canvas_cost_1 + canvas_cost_2 + total_paint_cost

theorem maria_earnings : (selling_price : ℤ) - total_cost = -326 := by sorry

end NUMINAMATH_CALUDE_maria_earnings_l1088_108895


namespace NUMINAMATH_CALUDE_angle_terminal_side_value_l1088_108896

/-- Given that the terminal side of angle α passes through the point P(-4a,3a) where a ≠ 0,
    the value of 2sin α + cos α is either 2/5 or -2/5 -/
theorem angle_terminal_side_value (a : ℝ) (α : ℝ) (h : a ≠ 0) :
  let P : ℝ × ℝ := (-4 * a, 3 * a)
  (∃ t : ℝ, t > 0 ∧ P = (t * Real.cos α, t * Real.sin α)) →
  2 * Real.sin α + Real.cos α = 2/5 ∨ 2 * Real.sin α + Real.cos α = -2/5 :=
by sorry

end NUMINAMATH_CALUDE_angle_terminal_side_value_l1088_108896


namespace NUMINAMATH_CALUDE_fisherman_tuna_count_l1088_108836

/-- The number of Red snappers the fisherman gets every day -/
def red_snappers : ℕ := 8

/-- The cost of a Red snapper in dollars -/
def red_snapper_cost : ℕ := 3

/-- The cost of a Tuna in dollars -/
def tuna_cost : ℕ := 2

/-- The total earnings of the fisherman in dollars per day -/
def total_earnings : ℕ := 52

/-- The number of Tunas the fisherman gets every day -/
def tuna_count : ℕ := (total_earnings - red_snappers * red_snapper_cost) / tuna_cost

theorem fisherman_tuna_count : tuna_count = 14 := by
  sorry

end NUMINAMATH_CALUDE_fisherman_tuna_count_l1088_108836


namespace NUMINAMATH_CALUDE_fourth_team_odd_l1088_108835

/-- Calculates the odd for the fourth team in a soccer bet -/
theorem fourth_team_odd (odd1 odd2 odd3 : ℝ) (bet_amount expected_winnings : ℝ) :
  odd1 = 1.28 →
  odd2 = 5.23 →
  odd3 = 3.25 →
  bet_amount = 5.00 →
  expected_winnings = 223.0072 →
  ∃ (odd4 : ℝ), abs (odd4 - 2.061) < 0.001 ∧ 
    odd1 * odd2 * odd3 * odd4 = expected_winnings / bet_amount :=
by
  sorry

#check fourth_team_odd

end NUMINAMATH_CALUDE_fourth_team_odd_l1088_108835


namespace NUMINAMATH_CALUDE_crate_tower_probability_l1088_108841

def crate_dimensions := (3, 4, 6)
def num_crates := 11
def target_height := 50

def valid_arrangements (a b c : ℕ) : ℕ :=
  if a + b + c = num_crates ∧ 3 * a + 4 * b + 6 * c = target_height
  then Nat.factorial num_crates / (Nat.factorial a * Nat.factorial b * Nat.factorial c)
  else 0

def total_valid_arrangements : ℕ :=
  valid_arrangements 4 2 5 + valid_arrangements 2 5 4 + valid_arrangements 0 8 3

def total_possible_arrangements : ℕ := 3^num_crates

theorem crate_tower_probability : 
  (total_valid_arrangements : ℚ) / total_possible_arrangements = 72 / 115 := by
  sorry

end NUMINAMATH_CALUDE_crate_tower_probability_l1088_108841


namespace NUMINAMATH_CALUDE_triangle_area_squared_l1088_108884

theorem triangle_area_squared (A B C : ℝ × ℝ) : 
  let AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let BC := Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)
  let CA := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let s := (AB + BC + CA) / 2
  let area := Real.sqrt (s * (s - AB) * (s - BC) * (s - CA))
  AB = 7 ∧ BC = 9 ∧ CA = 4 → area^2 = 180 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_squared_l1088_108884


namespace NUMINAMATH_CALUDE_total_dolls_count_l1088_108891

/-- The number of dolls in a big box -/
def dolls_per_big_box : ℕ := 7

/-- The number of dolls in a small box -/
def dolls_per_small_box : ℕ := 4

/-- The number of big boxes -/
def num_big_boxes : ℕ := 5

/-- The number of small boxes -/
def num_small_boxes : ℕ := 9

/-- The total number of dolls in all boxes -/
def total_dolls : ℕ := dolls_per_big_box * num_big_boxes + dolls_per_small_box * num_small_boxes

theorem total_dolls_count : total_dolls = 71 := by
  sorry

end NUMINAMATH_CALUDE_total_dolls_count_l1088_108891


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_evaluate_at_four_l1088_108828

theorem simplify_and_evaluate (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) : 
  (((2 * x + 2) / (x^2 - 1) + 1) / ((x + 1) / (x^2 - 2*x + 1))) = x - 1 :=
by sorry

theorem evaluate_at_four : 
  (((2 * 4 + 2) / (4^2 - 1) + 1) / ((4 + 1) / (4^2 - 2*4 + 1))) = 3 :=
by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_evaluate_at_four_l1088_108828


namespace NUMINAMATH_CALUDE_cube_opposite_face_l1088_108893

-- Define the faces of the cube
inductive Face : Type
| A | B | C | D | E | F

-- Define the adjacency relation between faces
def adjacent : Face → Face → Prop := sorry

-- Define the opposite relation between faces
def opposite : Face → Face → Prop := sorry

-- Define the shares_vertex relation between faces
def shares_vertex : Face → Face → Prop := sorry

-- Define the shares_edge relation between faces
def shares_edge : Face → Face → Prop := sorry

theorem cube_opposite_face :
  -- Condition 2: Face B is adjacent to Face A
  adjacent Face.B Face.A →
  -- Condition 3: Face C and Face D are adjacent to each other
  adjacent Face.C Face.D →
  -- Condition 3: Face C shares a vertex with Face A
  shares_vertex Face.C Face.A →
  -- Condition 3: Face D shares a vertex with Face A
  shares_vertex Face.D Face.A →
  -- Condition 4: Face E and Face F share an edge with each other
  shares_edge Face.E Face.F →
  -- Condition 4: Face E does not share an edge with Face A
  ¬ shares_edge Face.E Face.A →
  -- Condition 4: Face F does not share an edge with Face A
  ¬ shares_edge Face.F Face.A →
  -- Conclusion: Face F is opposite to Face A
  opposite Face.F Face.A := by sorry

end NUMINAMATH_CALUDE_cube_opposite_face_l1088_108893


namespace NUMINAMATH_CALUDE_circle_position_l1088_108807

def circle_center : ℝ × ℝ := (-3, 4)
def circle_radius : ℝ := 3

theorem circle_position :
  let (x, y) := circle_center
  let r := circle_radius
  (abs y > r) ∧ (abs x = r) := by sorry

end NUMINAMATH_CALUDE_circle_position_l1088_108807


namespace NUMINAMATH_CALUDE_angle_bak_is_right_angle_l1088_108808

-- Define the tetrahedron and its points
variable (A B C D K : EuclideanSpace ℝ (Fin 3))

-- Define the angles
def angle (p q r : EuclideanSpace ℝ (Fin 3)) : ℝ := sorry

-- State the conditions
variable (h1 : angle B A C + angle B A D = Real.pi)
variable (h2 : angle C A K = angle K A D)

-- State the theorem
theorem angle_bak_is_right_angle : angle B A K = Real.pi / 2 := by sorry

end NUMINAMATH_CALUDE_angle_bak_is_right_angle_l1088_108808


namespace NUMINAMATH_CALUDE_sum_of_roots_cubic_equation_l1088_108817

theorem sum_of_roots_cubic_equation : 
  let f (x : ℝ) := (x^3 - 3*x^2 - 12*x) / (x + 3)
  ∃ (a b : ℝ), (∀ x ≠ -3, f x = 3 ↔ x = a ∨ x = b) ∧ a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_cubic_equation_l1088_108817


namespace NUMINAMATH_CALUDE_committee_probability_l1088_108826

def total_members : ℕ := 30
def boys : ℕ := 12
def girls : ℕ := 18
def committee_size : ℕ := 5

theorem committee_probability :
  (Nat.choose total_members committee_size - 
   (Nat.choose boys committee_size + Nat.choose girls committee_size)) / 
   Nat.choose total_members committee_size = 133146 / 142506 := by
sorry

end NUMINAMATH_CALUDE_committee_probability_l1088_108826


namespace NUMINAMATH_CALUDE_polar_to_cartesian_x_plus_y_bounds_l1088_108872

-- Define the circle in polar coordinates
def polar_circle (ρ θ : ℝ) : Prop :=
  ρ^2 - 4 * Real.sqrt 2 * ρ * Real.cos (θ - Real.pi / 4) + 6 = 0

-- Define the circle in Cartesian coordinates
def cartesian_circle (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 2)^2 = 2

-- Theorem stating the equivalence of polar and Cartesian equations
theorem polar_to_cartesian :
  ∀ (x y ρ θ : ℝ), 
    x = ρ * Real.cos θ → 
    y = ρ * Real.sin θ → 
    polar_circle ρ θ ↔ cartesian_circle x y :=
sorry

-- Theorem for the bounds of x + y
theorem x_plus_y_bounds :
  ∀ (x y : ℝ), cartesian_circle x y → 2 ≤ x + y ∧ x + y ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_x_plus_y_bounds_l1088_108872


namespace NUMINAMATH_CALUDE_recipe_total_cups_l1088_108879

/-- Represents the ratio of ingredients in the recipe -/
structure RecipeRatio where
  butter : ℕ
  flour : ℕ
  sugar : ℕ

/-- Calculates the total cups of ingredients given a ratio and the amount of flour -/
def totalCups (ratio : RecipeRatio) (flourAmount : ℕ) : ℕ :=
  let unitAmount := flourAmount / ratio.flour
  unitAmount * (ratio.butter + ratio.flour + ratio.sugar)

/-- Theorem: Given the specified ratio and flour amount, the total cups of ingredients is 30 -/
theorem recipe_total_cups :
  let ratio : RecipeRatio := { butter := 2, flour := 5, sugar := 3 }
  let flourAmount : ℕ := 15
  totalCups ratio flourAmount = 30 := by
  sorry


end NUMINAMATH_CALUDE_recipe_total_cups_l1088_108879


namespace NUMINAMATH_CALUDE_simultaneous_integer_fractions_l1088_108815

theorem simultaneous_integer_fractions (x : ℤ) :
  (∃ y z : ℤ, (x - 3) / 7 = y ∧ (x - 2) / 5 = z) ↔ 
  (∃ t : ℤ, x = 35 * t + 17) :=
by sorry

end NUMINAMATH_CALUDE_simultaneous_integer_fractions_l1088_108815


namespace NUMINAMATH_CALUDE_sum_of_factors_24_l1088_108804

def factors (n : ℕ) : Finset ℕ :=
  Finset.filter (λ x => n % x = 0) (Finset.range (n + 1))

theorem sum_of_factors_24 :
  (factors 24).sum id = 60 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_factors_24_l1088_108804


namespace NUMINAMATH_CALUDE_fraction_simplification_l1088_108831

theorem fraction_simplification :
  201920192019 / 191719171917 = 673 / 639 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1088_108831


namespace NUMINAMATH_CALUDE_correct_fraction_l1088_108849

theorem correct_fraction (number : ℕ) (x y : ℕ) (h1 : number = 192) 
  (h2 : (5 : ℚ) / 6 * number = x / y * number + 100) : x / y = (5 : ℚ) / 16 := by
  sorry

end NUMINAMATH_CALUDE_correct_fraction_l1088_108849


namespace NUMINAMATH_CALUDE_sum_of_powers_l1088_108821

theorem sum_of_powers (k n : ℕ) : 
  (∀ x y : ℝ, 2 * x^k * y^(k+2) + 3 * x^2 * y^n = 5 * x^2 * y^n) → 
  k + n = 6 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_powers_l1088_108821


namespace NUMINAMATH_CALUDE_halloween_candy_distribution_l1088_108865

theorem halloween_candy_distribution (initial_candy : ℕ) (eaten_candy : ℕ) (num_piles : ℕ) : 
  initial_candy = 32 → 
  eaten_candy = 12 → 
  num_piles = 4 → 
  (initial_candy - eaten_candy) / num_piles = 5 := by
  sorry

end NUMINAMATH_CALUDE_halloween_candy_distribution_l1088_108865


namespace NUMINAMATH_CALUDE_clock_equal_angles_l1088_108889

/-- The time in minutes when the hour and minute hands form equal angles with their positions at 12 o'clock -/
def equal_angle_time : ℚ := 55 + 5/13

/-- The angular speed of the minute hand in degrees per minute -/
def minute_hand_speed : ℚ := 6

/-- The angular speed of the hour hand in degrees per hour -/
def hour_hand_speed : ℚ := 30

theorem clock_equal_angles :
  let t : ℚ := equal_angle_time / 60  -- Convert minutes to hours
  minute_hand_speed * 60 * t = 360 - hour_hand_speed * t := by sorry

#eval equal_angle_time

end NUMINAMATH_CALUDE_clock_equal_angles_l1088_108889


namespace NUMINAMATH_CALUDE_circle_tangent_to_line_l1088_108846

theorem circle_tangent_to_line (m : ℝ) (h : m ≥ 0) :
  ∃ (x y : ℝ), x^2 + y^2 = m ∧ x + y = Real.sqrt (2 * m) ∧
  ∀ (x' y' : ℝ), x'^2 + y'^2 = m → x' + y' ≤ Real.sqrt (2 * m) :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_line_l1088_108846


namespace NUMINAMATH_CALUDE_number_ratio_l1088_108818

theorem number_ratio (f s t : ℝ) : 
  t = 2 * f →
  (f + s + t) / 3 = 77 →
  f = 33 →
  s / f = 4 := by
sorry

end NUMINAMATH_CALUDE_number_ratio_l1088_108818


namespace NUMINAMATH_CALUDE_reciprocal_equation_solution_l1088_108855

theorem reciprocal_equation_solution (x : ℝ) : 
  (2 - 1 / (2 - x) = 1 / (2 - x)) → x = 1 := by sorry

end NUMINAMATH_CALUDE_reciprocal_equation_solution_l1088_108855


namespace NUMINAMATH_CALUDE_gcd_2720_1530_l1088_108830

theorem gcd_2720_1530 : Nat.gcd 2720 1530 = 170 := by
  sorry

end NUMINAMATH_CALUDE_gcd_2720_1530_l1088_108830


namespace NUMINAMATH_CALUDE_shekar_average_proof_l1088_108814

def shekar_average_marks (math science social_studies english biology : ℕ) : ℚ :=
  (math + science + social_studies + english + biology : ℚ) / 5

theorem shekar_average_proof :
  shekar_average_marks 76 65 82 67 75 = 73 := by
  sorry

end NUMINAMATH_CALUDE_shekar_average_proof_l1088_108814


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_one_l1088_108813

theorem fraction_zero_implies_x_negative_one (x : ℝ) :
  (abs x - 1) / (x - 1) = 0 → x = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_one_l1088_108813


namespace NUMINAMATH_CALUDE_all_terms_are_integers_l1088_108882

def sequence_a : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | (n + 2) => (1 + (sequence_a (n + 1))^2) / (sequence_a n)

theorem all_terms_are_integers :
  ∀ n : ℕ, ∃ k : ℤ, sequence_a n = k :=
by sorry

end NUMINAMATH_CALUDE_all_terms_are_integers_l1088_108882


namespace NUMINAMATH_CALUDE_alternating_squares_sum_l1088_108861

theorem alternating_squares_sum : 
  23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2 = 288 := by
  sorry

end NUMINAMATH_CALUDE_alternating_squares_sum_l1088_108861


namespace NUMINAMATH_CALUDE_jessy_jokes_count_l1088_108810

theorem jessy_jokes_count (jessy_jokes alan_jokes : ℕ) : 
  alan_jokes = 7 →
  2 * (jessy_jokes + alan_jokes) = 54 →
  jessy_jokes = 20 := by
sorry

end NUMINAMATH_CALUDE_jessy_jokes_count_l1088_108810


namespace NUMINAMATH_CALUDE_remainder_82460_div_8_l1088_108833

theorem remainder_82460_div_8 : 82460 % 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_82460_div_8_l1088_108833


namespace NUMINAMATH_CALUDE_jackie_eligible_for_free_shipping_l1088_108894

def shampoo_price : ℝ := 12.50
def conditioner_price : ℝ := 15.00
def face_cream_price : ℝ := 20.00
def discount_rate : ℝ := 0.10
def free_shipping_threshold : ℝ := 75.00

def total_cost : ℝ := 2 * shampoo_price + 3 * conditioner_price + face_cream_price

def discounted_cost : ℝ := total_cost * (1 - discount_rate)

theorem jackie_eligible_for_free_shipping :
  discounted_cost ≥ free_shipping_threshold := by
  sorry

end NUMINAMATH_CALUDE_jackie_eligible_for_free_shipping_l1088_108894


namespace NUMINAMATH_CALUDE_olivia_math_problem_l1088_108847

theorem olivia_math_problem (x : ℝ) 
  (h1 : 7 * x + 3 = 31) : 
  x = 4 := by
sorry

end NUMINAMATH_CALUDE_olivia_math_problem_l1088_108847


namespace NUMINAMATH_CALUDE_athlete_speed_l1088_108838

/-- Given an athlete who runs 200 meters in 40 seconds, prove that their speed is 5 meters per second. -/
theorem athlete_speed (distance : Real) (time : Real) (speed : Real) 
  (h1 : distance = 200) 
  (h2 : time = 40) 
  (h3 : speed = distance / time) : speed = 5 := by
  sorry

end NUMINAMATH_CALUDE_athlete_speed_l1088_108838


namespace NUMINAMATH_CALUDE_function_inequality_l1088_108852

theorem function_inequality (f : ℝ → ℝ) (h_diff : Differentiable ℝ f) 
  (h_ineq : ∀ x, deriv f x < f x) : 
  f 1 < Real.exp 1 * f 0 ∧ f 2013 < Real.exp 2013 * f 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l1088_108852


namespace NUMINAMATH_CALUDE_smoothie_servings_l1088_108811

/-- The number of servings that can be made from a given volume of smoothie mix -/
def number_of_servings (watermelon_puree : ℕ) (cream : ℕ) (serving_size : ℕ) : ℕ :=
  (watermelon_puree + cream) / serving_size

/-- Theorem: Given 500 ml of watermelon puree and 100 ml of cream, 
    the number of 150 ml servings that can be made is equal to 4 -/
theorem smoothie_servings : 
  number_of_servings 500 100 150 = 4 := by
  sorry

end NUMINAMATH_CALUDE_smoothie_servings_l1088_108811


namespace NUMINAMATH_CALUDE_optimal_characterization_l1088_108801

def Ω : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 ≤ 2008}

def better (p q : ℝ × ℝ) : Prop := p.1 ≤ q.1 ∧ p.2 ≥ q.2

def optimal (q : ℝ × ℝ) : Prop :=
  q ∈ Ω ∧ ∀ p ∈ Ω, ¬(better p q ∧ p ≠ q)

theorem optimal_characterization (q : ℝ × ℝ) :
  optimal q ↔ q.1^2 + q.2^2 = 2008 ∧ q.1 ≤ 0 ∧ q.2 ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_optimal_characterization_l1088_108801


namespace NUMINAMATH_CALUDE_fourth_sampled_number_l1088_108897

/-- Represents a random number table -/
def RandomNumberTable := List (List Nat)

/-- Represents a position in the random number table -/
structure TablePosition where
  row : Nat
  column : Nat

/-- Checks if a number is valid for sampling (between 1 and 40) -/
def isValidNumber (n : Nat) : Bool :=
  1 ≤ n ∧ n ≤ 40

/-- Gets the next position in the table -/
def nextPosition (pos : TablePosition) (tableWidth : Nat) : TablePosition :=
  if pos.column < tableWidth then
    { row := pos.row, column := pos.column + 1 }
  else
    { row := pos.row + 1, column := 1 }

/-- Samples the next valid number from the table -/
def sampleNextNumber (table : RandomNumberTable) (startPos : TablePosition) : Option Nat :=
  sorry

/-- Samples n valid numbers from the table -/
def sampleNumbers (table : RandomNumberTable) (startPos : TablePosition) (n : Nat) : List Nat :=
  sorry

/-- The main theorem to prove -/
theorem fourth_sampled_number
  (table : RandomNumberTable)
  (startPos : TablePosition)
  (h_table : table = [
    [84, 42, 17, 56, 31, 07, 23, 55, 06, 82, 77, 04, 74, 43, 59, 76, 30, 63, 50, 25, 83, 92, 12, 06],
    [63, 01, 63, 78, 59, 16, 95, 56, 67, 19, 98, 10, 50, 71, 75, 12, 86, 73, 58, 07, 44, 39, 52, 38]
  ])
  (h_startPos : startPos = { row := 0, column := 7 })
  : (sampleNumbers table startPos 4).get! 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_fourth_sampled_number_l1088_108897


namespace NUMINAMATH_CALUDE_tangent_line_implies_m_eq_two_l1088_108819

/-- A circle defined by parametric equations with parameter m > 0 -/
structure ParametricCircle (m : ℝ) where
  x : ℝ → ℝ
  y : ℝ → ℝ
  h_x : ∀ φ, x φ = Real.sqrt m * Real.cos φ
  h_y : ∀ φ, y φ = Real.sqrt m * Real.sin φ
  h_m : m > 0

/-- The line x + y = m is tangent to the circle -/
def isTangent (m : ℝ) (circle : ParametricCircle m) : Prop :=
  ∃ φ, circle.x φ + circle.y φ = m ∧
    ∀ ψ, circle.x ψ + circle.y ψ ≤ m

theorem tangent_line_implies_m_eq_two (m : ℝ) (circle : ParametricCircle m)
    (h_tangent : isTangent m circle) : m = 2 := by
  sorry

#check tangent_line_implies_m_eq_two

end NUMINAMATH_CALUDE_tangent_line_implies_m_eq_two_l1088_108819


namespace NUMINAMATH_CALUDE_problem_solution_l1088_108822

theorem problem_solution (p q r s : ℕ+) 
  (h1 : p^3 = q^2) 
  (h2 : r^4 = s^3) 
  (h3 : r - p = 25) : 
  s - q = 73 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1088_108822


namespace NUMINAMATH_CALUDE_range_of_a_l1088_108853

/-- An odd function with period 3 -/
def OddPeriodic3 (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (x + 3) = f x)

/-- The main theorem -/
theorem range_of_a (a : ℝ) :
  (∃ f : ℝ → ℝ, OddPeriodic3 f ∧ f 2 > 1 ∧ f 2014 = (2 * a - 3) / (a + 1)) →
  -1 < a ∧ a < 2/3 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l1088_108853


namespace NUMINAMATH_CALUDE_intersection_complement_subset_condition_l1088_108864

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x ≤ a + 3}
def B : Set ℝ := {x | x < -1 ∨ x > 5}

-- Theorem 1: When a = -2, A ∩ (ℝ \ B) = {x | -1 ≤ x ≤ 1}
theorem intersection_complement (a : ℝ) (h : a = -2) :
  A a ∩ (Set.univ \ B) = {x : ℝ | -1 ≤ x ∧ x ≤ 1} := by sorry

-- Theorem 2: A ⊆ B if and only if a ≥ 2
theorem subset_condition (a : ℝ) :
  A a ⊆ B ↔ a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_intersection_complement_subset_condition_l1088_108864


namespace NUMINAMATH_CALUDE_cos_alpha_value_l1088_108873

theorem cos_alpha_value (α : Real) (h : Real.sin (α - Real.pi/2) = 3/5) :
  Real.cos α = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l1088_108873


namespace NUMINAMATH_CALUDE_family_meeting_impossible_l1088_108883

theorem family_meeting_impossible (n : ℕ) (h : n = 9) :
  ¬ ∃ (handshakes : ℕ), 2 * handshakes = n * 3 :=
by
  sorry

end NUMINAMATH_CALUDE_family_meeting_impossible_l1088_108883


namespace NUMINAMATH_CALUDE_no_solution_equation_l1088_108860

theorem no_solution_equation : ∀ x : ℝ, 
  4 * x * (10 * x - (-10 - (3 * x - 8 * (x + 1)))) + 
  5 * (12 - (4 * (x + 1) - 3 * x)) ≠ 
  18 * x^2 - (6 * x^2 - (7 * x + 4 * (2 * x^2 - x + 11))) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_equation_l1088_108860


namespace NUMINAMATH_CALUDE_square_perimeter_sum_l1088_108800

theorem square_perimeter_sum (x y : ℝ) (h1 : x^2 + y^2 = 85) (h2 : x^2 - y^2 = 45) :
  4 * (x + y) = 4 * (Real.sqrt 65 + 2 * Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_sum_l1088_108800


namespace NUMINAMATH_CALUDE_rectangle_relationships_l1088_108832

-- Define the rectangle
def rectangle (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ 2*x + 2*y = 10

-- Define the area function
def area (x y : ℝ) : ℝ := x * y

-- Theorem statement
theorem rectangle_relationships (x y : ℝ) (h : rectangle x y) :
  ∃ (a b : ℝ), y = a*x + b ∧    -- Linear relationship between y and x
  ∃ (p q r : ℝ), area x y = p*x^2 + q*x + r :=  -- Quadratic relationship between S and x
by sorry

end NUMINAMATH_CALUDE_rectangle_relationships_l1088_108832


namespace NUMINAMATH_CALUDE_odd_function_sum_l1088_108839

-- Define an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- State the theorem
theorem odd_function_sum (f : ℝ → ℝ) 
  (h_odd : odd_function f) 
  (h_eq : f (-2) + f (-1) - 3 = f 1 + f 2 + 3) : 
  f 1 + f 2 = -3 :=
by sorry

end NUMINAMATH_CALUDE_odd_function_sum_l1088_108839


namespace NUMINAMATH_CALUDE_train_length_l1088_108870

/-- The length of a train given its speed, the speed of a person moving in the opposite direction, and the time it takes for the train to pass the person. -/
theorem train_length (train_speed : ℝ) (person_speed : ℝ) (passing_time : ℝ) :
  train_speed = 82 →
  person_speed = 6 →
  passing_time = 4.499640028797696 →
  ∃ (length : ℝ), abs (length - 110) < 0.5 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l1088_108870


namespace NUMINAMATH_CALUDE_shaded_region_value_l1088_108877

/-- Rectangle PQRS with PS = 2 and PQ = 4 -/
structure Rectangle where
  ps : ℝ
  pq : ℝ
  h_ps : ps = 2
  h_pq : pq = 4

/-- Points T, U, V, W positioned so that RT = RU = PW = PV = a -/
def points_position (rect : Rectangle) (a : ℝ) : Prop :=
  ∃ (t u v w : ℝ × ℝ), 
    (rect.pq - a = t.1) ∧ (rect.pq - a = u.1) ∧ (a = v.1) ∧ (a = w.1) ∧
    (rect.ps = t.2) ∧ (0 = u.2) ∧ (rect.ps = v.2) ∧ (0 = w.2)

/-- VU and WT pass through the center of the rectangle -/
def lines_through_center (rect : Rectangle) (a : ℝ) : Prop :=
  ∃ (center : ℝ × ℝ), center = (rect.pq / 2, rect.ps / 2)

/-- The shaded region is 1/8 the area of PQRS -/
def shaded_region_ratio (rect : Rectangle) (a : ℝ) : Prop :=
  3 * a = 1/8 * (rect.ps * rect.pq)

/-- Main theorem -/
theorem shaded_region_value (rect : Rectangle) :
  points_position rect (1/3) ∧ 
  lines_through_center rect (1/3) ∧ 
  shaded_region_ratio rect (1/3) := by
  sorry

end NUMINAMATH_CALUDE_shaded_region_value_l1088_108877


namespace NUMINAMATH_CALUDE_max_pies_36_l1088_108842

/-- Calculates the maximum number of pies that can be made given a certain number of apples,
    where every two pies require 12 apples and every third pie needs an extra apple. -/
def max_pies (total_apples : ℕ) : ℕ :=
  let basic_pies := 2 * (total_apples / 12)
  let extra_apples := basic_pies / 3
  let adjusted_apples := total_apples - extra_apples
  let full_sets := adjusted_apples / 12
  let remaining_apples := adjusted_apples % 12
  2 * full_sets + if remaining_apples ≥ 6 then 1 else 0

/-- Theorem stating that given 36 apples, the maximum number of pies that can be made is 9. -/
theorem max_pies_36 : max_pies 36 = 9 := by
  sorry

end NUMINAMATH_CALUDE_max_pies_36_l1088_108842


namespace NUMINAMATH_CALUDE_factorization_proof_l1088_108848

theorem factorization_proof (a b x y : ℝ) : 
  (4 * a^2 * b - 6 * a * b^2 = 2 * a * b * (2 * a - 3 * b)) ∧ 
  (25 * x^2 - 9 * y^2 = (5 * x + 3 * y) * (5 * x - 3 * y)) ∧ 
  (2 * a^2 * b - 8 * a * b^2 + 8 * b^3 = 2 * b * (a - 2 * b)^2) ∧ 
  ((x + 2) * (x - 8) + 25 = (x - 3)^2) :=
by sorry


end NUMINAMATH_CALUDE_factorization_proof_l1088_108848


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l1088_108892

theorem sum_of_two_numbers (x y : ℝ) (h1 : y = 4 * x) (h2 : x + y = 45) : x + y = 45 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l1088_108892


namespace NUMINAMATH_CALUDE_complex_number_equality_l1088_108857

theorem complex_number_equality : ((-1 + Complex.I * Real.sqrt 3) ^ 5) / (1 + Complex.I * Real.sqrt 3) = -16 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equality_l1088_108857


namespace NUMINAMATH_CALUDE_mans_walking_speed_l1088_108850

/-- Proves that given a man who walks a certain distance in 5 hours and runs the same distance at 15 kmph in 36 minutes, his walking speed is 1.8 kmph. -/
theorem mans_walking_speed 
  (walking_time : ℝ) 
  (running_speed : ℝ) 
  (running_time_minutes : ℝ) :
  walking_time = 5 →
  running_speed = 15 →
  running_time_minutes = 36 →
  (walking_time * (running_speed * (running_time_minutes / 60))) / walking_time = 1.8 :=
by sorry

end NUMINAMATH_CALUDE_mans_walking_speed_l1088_108850


namespace NUMINAMATH_CALUDE_favorite_numbers_sum_of_squares_l1088_108863

/-- The sum of the squares of Misty's, Glory's, and Dawn's favorite numbers -/
def sumOfSquares (gloryFavorite : ℕ) : ℕ :=
  let mistyFavorite := gloryFavorite / 3
  let dawnFavorite := gloryFavorite * 2
  mistyFavorite ^ 2 + gloryFavorite ^ 2 + dawnFavorite ^ 2

/-- Theorem stating that the sum of squares of the favorite numbers is 1,035,000 -/
theorem favorite_numbers_sum_of_squares :
  sumOfSquares 450 = 1035000 := by
  sorry

#eval sumOfSquares 450

end NUMINAMATH_CALUDE_favorite_numbers_sum_of_squares_l1088_108863


namespace NUMINAMATH_CALUDE_ratio_average_problem_l1088_108875

theorem ratio_average_problem (a b c : ℕ+) (h_ratio : a.val / 2 = b.val / 3 ∧ b.val / 3 = c.val / 4) (h_a : a = 28) :
  (a.val + b.val + c.val) / 3 = 42 := by
sorry

end NUMINAMATH_CALUDE_ratio_average_problem_l1088_108875


namespace NUMINAMATH_CALUDE_quadratic_equation_solvability_l1088_108867

theorem quadratic_equation_solvability (m : ℝ) :
  (∃ x : ℝ, x^2 + 2*x - m - 1 = 0) ↔ m ≥ -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solvability_l1088_108867


namespace NUMINAMATH_CALUDE_conference_arrangements_l1088_108871

/-- The number of ways to arrange n distinct elements --/
def arrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n distinct elements with k pairs having a specific order requirement --/
def arrangementsWithOrderRequirements (n : ℕ) (k : ℕ) : ℕ :=
  arrangements n / (2^k)

/-- Theorem stating that arranging 7 lecturers with 2 order requirements results in 1260 possible arrangements --/
theorem conference_arrangements : arrangementsWithOrderRequirements 7 2 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_conference_arrangements_l1088_108871


namespace NUMINAMATH_CALUDE_obrien_current_hats_l1088_108880

/-- The number of hats Policeman O'Brien has after theft -/
def obrien_hats_after_theft (simpson_hats : ℕ) (stolen_hats : ℕ) : ℕ :=
  2 * simpson_hats + 5 - stolen_hats

/-- Theorem stating the number of hats Policeman O'Brien has after theft -/
theorem obrien_current_hats (simpson_hats stolen_hats : ℕ) 
  (h1 : simpson_hats = 15) :
  obrien_hats_after_theft simpson_hats stolen_hats = 35 - stolen_hats := by
  sorry

#check obrien_current_hats

end NUMINAMATH_CALUDE_obrien_current_hats_l1088_108880


namespace NUMINAMATH_CALUDE_fraction_sum_squared_l1088_108803

theorem fraction_sum_squared (a b c : ℝ) (h : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a / (b - c)^2 + b / (c - a)^2 + c / (a - b)^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_squared_l1088_108803


namespace NUMINAMATH_CALUDE_one_correct_judgment_l1088_108890

theorem one_correct_judgment :
  let judgment1 := ∀ a b : ℝ, a + b ≠ 6 → a ≠ 3 ∨ b ≠ 3
  let judgment2 := ∀ p q : Prop, (p ∨ q) → (p ∧ q)
  let judgment3 := (¬ ∀ a b : ℝ, a^2 + b^2 ≥ 2*(a - b - 1)) ↔ (∃ a b : ℝ, a^2 + b^2 ≤ 2*(a - b - 1))
  let judgment4 := (judgment1 ∧ ¬judgment2 ∧ ¬judgment3)
  judgment4 := by sorry

end NUMINAMATH_CALUDE_one_correct_judgment_l1088_108890


namespace NUMINAMATH_CALUDE_unique_solution_cubic_system_l1088_108899

theorem unique_solution_cubic_system :
  ∃! (x y z : ℝ), x^3 = 2*y - 1 ∧ y^3 = 2*z - 1 ∧ z^3 = 2*x - 1 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_cubic_system_l1088_108899


namespace NUMINAMATH_CALUDE_exists_valid_strategy_365_l1088_108824

/-- A strategy for sorting n elements using 3-way comparisons -/
def SortingStrategy (n : ℕ) := ℕ

/-- The number of 3-way comparisons needed to sort n elements using a given strategy -/
def comparisons (n : ℕ) (s : SortingStrategy n) : ℕ := sorry

/-- A strategy is valid if it correctly sorts n elements -/
def is_valid_strategy (n : ℕ) (s : SortingStrategy n) : Prop := sorry

/-- The main theorem: there exists a valid strategy for 365 elements using at most 1691 comparisons -/
theorem exists_valid_strategy_365 :
  ∃ (s : SortingStrategy 365), is_valid_strategy 365 s ∧ comparisons 365 s ≤ 1691 := by sorry

end NUMINAMATH_CALUDE_exists_valid_strategy_365_l1088_108824


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l1088_108816

theorem greatest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 17 = 0 → n ≤ 986 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l1088_108816


namespace NUMINAMATH_CALUDE_james_remaining_money_l1088_108829

def weekly_allowance : ℕ := 10
def saving_weeks : ℕ := 4
def video_game_fraction : ℚ := 1/2
def book_fraction : ℚ := 1/4

theorem james_remaining_money :
  let total_savings := weekly_allowance * saving_weeks
  let after_video_game := total_savings * (1 - video_game_fraction)
  let book_cost := after_video_game * book_fraction
  let remaining := after_video_game - book_cost
  remaining = 15 := by sorry

end NUMINAMATH_CALUDE_james_remaining_money_l1088_108829


namespace NUMINAMATH_CALUDE_dog_food_bag_weight_l1088_108812

/-- Proves that the weight of each bag of dog food is 20 pounds -/
theorem dog_food_bag_weight :
  let cup_weight : ℚ := 1/4  -- Weight of a cup of dog food in pounds
  let num_dogs : ℕ := 2  -- Number of dogs
  let cups_per_meal : ℕ := 6  -- Cups of food per meal per dog
  let meals_per_day : ℕ := 2  -- Number of meals per day
  let bags_per_month : ℕ := 9  -- Number of bags bought per month
  let days_per_month : ℕ := 30  -- Number of days in a month
  
  let daily_consumption : ℚ := num_dogs * cups_per_meal * meals_per_day * cup_weight
  let monthly_consumption : ℚ := daily_consumption * days_per_month
  let bag_weight : ℚ := monthly_consumption / bags_per_month

  bag_weight = 20 := by
    sorry

end NUMINAMATH_CALUDE_dog_food_bag_weight_l1088_108812


namespace NUMINAMATH_CALUDE_loot_box_solution_l1088_108806

/-- Represents the loot box problem -/
def LootBoxProblem (cost_per_box : ℝ) (avg_value_per_box : ℝ) (total_spent : ℝ) : Prop :=
  let num_boxes : ℝ := total_spent / cost_per_box
  let total_avg_value : ℝ := avg_value_per_box * num_boxes
  let total_lost : ℝ := total_spent - total_avg_value
  let avg_lost_per_box : ℝ := total_lost / num_boxes
  avg_lost_per_box = 1.5

/-- Theorem stating the solution to the loot box problem -/
theorem loot_box_solution :
  LootBoxProblem 5 3.5 40 := by
  sorry

#check loot_box_solution

end NUMINAMATH_CALUDE_loot_box_solution_l1088_108806


namespace NUMINAMATH_CALUDE_beanie_baby_ratio_l1088_108868

theorem beanie_baby_ratio : 
  ∀ (lori_bb sydney_bb : ℕ),
    lori_bb = 300 →
    lori_bb + sydney_bb = 320 →
    lori_bb = 15 * sydney_bb :=
by
  sorry

end NUMINAMATH_CALUDE_beanie_baby_ratio_l1088_108868


namespace NUMINAMATH_CALUDE_qiannan_establishment_year_l1088_108805

/-- Represents the Heavenly Stems -/
inductive HeavenlyStem
| Jia | Yi | Bing | Ding | Wu | Ji | Geng | Xin | Ren | Gui

/-- Represents the Earthly Branches -/
inductive EarthlyBranch
| Zi | Chou | Yin | Mao | Chen | Si | Wu | Wei | Shen | You | Xu | Hai

/-- Represents a year in the Heavenly Stems and Earthly Branches system -/
structure StemBranchYear :=
  (stem : HeavenlyStem)
  (branch : EarthlyBranch)

/-- Function to get the previous stem -/
def prevStem (s : HeavenlyStem) : HeavenlyStem :=
  match s with
  | HeavenlyStem.Jia => HeavenlyStem.Gui
  | HeavenlyStem.Yi => HeavenlyStem.Jia
  | HeavenlyStem.Bing => HeavenlyStem.Yi
  | HeavenlyStem.Ding => HeavenlyStem.Bing
  | HeavenlyStem.Wu => HeavenlyStem.Ding
  | HeavenlyStem.Ji => HeavenlyStem.Wu
  | HeavenlyStem.Geng => HeavenlyStem.Ji
  | HeavenlyStem.Xin => HeavenlyStem.Geng
  | HeavenlyStem.Ren => HeavenlyStem.Xin
  | HeavenlyStem.Gui => HeavenlyStem.Ren

/-- Function to get the previous branch -/
def prevBranch (b : EarthlyBranch) : EarthlyBranch :=
  match b with
  | EarthlyBranch.Zi => EarthlyBranch.Hai
  | EarthlyBranch.Chou => EarthlyBranch.Zi
  | EarthlyBranch.Yin => EarthlyBranch.Chou
  | EarthlyBranch.Mao => EarthlyBranch.Yin
  | EarthlyBranch.Chen => EarthlyBranch.Mao
  | EarthlyBranch.Si => EarthlyBranch.Chen
  | EarthlyBranch.Wu => EarthlyBranch.Si
  | EarthlyBranch.Wei => EarthlyBranch.Wu
  | EarthlyBranch.Shen => EarthlyBranch.Wei
  | EarthlyBranch.You => EarthlyBranch.Shen
  | EarthlyBranch.Xu => EarthlyBranch.You
  | EarthlyBranch.Hai => EarthlyBranch.Xu

/-- Function to get the year n years before a given year -/
def yearsBefore (n : Nat) (year : StemBranchYear) : StemBranchYear :=
  if n = 0 then year
  else yearsBefore (n - 1) { stem := prevStem year.stem, branch := prevBranch year.branch }

theorem qiannan_establishment_year :
  let year2023 : StemBranchYear := { stem := HeavenlyStem.Gui, branch := EarthlyBranch.Mao }
  let establishmentYear := yearsBefore 67 year2023
  establishmentYear.stem = HeavenlyStem.Bing ∧ establishmentYear.branch = EarthlyBranch.Shen :=
by sorry

end NUMINAMATH_CALUDE_qiannan_establishment_year_l1088_108805


namespace NUMINAMATH_CALUDE_ratio_problem_l1088_108809

theorem ratio_problem (A B C : ℚ) (h : A / B = 3 ∧ B / C = 1/6) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 5/8 := by sorry

end NUMINAMATH_CALUDE_ratio_problem_l1088_108809


namespace NUMINAMATH_CALUDE_equilateral_hyperbola_through_point_l1088_108888

/-- An equilateral hyperbola is a hyperbola with perpendicular asymptotes -/
def is_equilateral_hyperbola (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ f = λ x y => a * (x^2 - y^2) + b * x + c * y + 1

/-- A point (x, y) lies on a curve defined by function f if f x y = 0 -/
def point_on_curve (f : ℝ → ℝ → ℝ) (x y : ℝ) : Prop :=
  f x y = 0

/-- A curve is symmetric about the x-axis if for every point (x, y) on the curve,
    the point (x, -y) is also on the curve -/
def symmetric_about_x_axis (f : ℝ → ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x y = 0 → f x (-y) = 0

/-- A curve is symmetric about the y-axis if for every point (x, y) on the curve,
    the point (-x, y) is also on the curve -/
def symmetric_about_y_axis (f : ℝ → ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x y = 0 → f (-x) y = 0

theorem equilateral_hyperbola_through_point :
  ∃ f : ℝ → ℝ → ℝ,
    is_equilateral_hyperbola f ∧
    point_on_curve f 3 (-1) ∧
    symmetric_about_x_axis f ∧
    symmetric_about_y_axis f ∧
    f = λ x y => x^2 - y^2 - 8 := by sorry

end NUMINAMATH_CALUDE_equilateral_hyperbola_through_point_l1088_108888


namespace NUMINAMATH_CALUDE_final_value_less_than_original_l1088_108886

theorem final_value_less_than_original (p q M : ℝ) 
  (hp : p > 0) (hq : q > 0) (hq_upper : q < 50) (hM : M > 0) :
  M * (1 - p / 100) * (1 + q / 100) < M ↔ p > (100 * q - q^2) / 100 := by
  sorry

end NUMINAMATH_CALUDE_final_value_less_than_original_l1088_108886


namespace NUMINAMATH_CALUDE_no_simultaneous_overtake_l1088_108851

/-- Proves that there is no time when Teena is simultaneously 25 miles ahead of Yoe and 10 miles ahead of Lona -/
theorem no_simultaneous_overtake :
  ¬ ∃ t : ℝ, t > 0 ∧ 
  (85 * t - 60 * t = 25 + 17.5) ∧ 
  (85 * t - 70 * t = 10 + 20) :=
sorry

end NUMINAMATH_CALUDE_no_simultaneous_overtake_l1088_108851


namespace NUMINAMATH_CALUDE_f_properties_l1088_108834

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6) + 2 * (Real.cos x) ^ 2

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T')) ∧
  (∃ (M : ℝ), M = 2 ∧ (∀ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 → f x ≤ M) ∧
    (∃ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x = M)) ∧
  (∃ (m : ℝ), m = 1 / 2 ∧ (∀ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 → m ≤ f x) ∧
    (∃ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x = m)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1088_108834


namespace NUMINAMATH_CALUDE_right_triangle_squares_area_l1088_108876

theorem right_triangle_squares_area (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →
  c^2 = 1009 →
  4*a^2 + 4*b^2 + 4*c^2 = 8072 := by sorry

end NUMINAMATH_CALUDE_right_triangle_squares_area_l1088_108876


namespace NUMINAMATH_CALUDE_daily_food_cost_l1088_108843

theorem daily_food_cost (purchase_price : ℕ) (vaccination_cost : ℕ) (selling_price : ℕ) (num_days : ℕ) (profit : ℕ) :
  purchase_price = 600 →
  vaccination_cost = 500 →
  selling_price = 2500 →
  num_days = 40 →
  profit = 600 →
  (selling_price - (purchase_price + vaccination_cost) - profit) / num_days = 20 := by
  sorry

end NUMINAMATH_CALUDE_daily_food_cost_l1088_108843


namespace NUMINAMATH_CALUDE_min_original_tables_l1088_108825

/-- Given a restaurant scenario with customers and tables, prove that the minimum number of original tables is 3. -/
theorem min_original_tables (X Y Z A B C : ℕ) : 
  X = Z + A + B + C →  -- Total customers equals those who left plus those who remained
  Y ≥ 3 :=             -- The original number of tables is at least 3
by sorry

end NUMINAMATH_CALUDE_min_original_tables_l1088_108825


namespace NUMINAMATH_CALUDE_set_operations_and_subset_l1088_108856

open Set

def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x > 2}
def C (a : ℝ) : Set ℝ := {x | 1 < x ∧ x < a}

theorem set_operations_and_subset :
  (A ∪ B = {x | 2 < x ∧ x ≤ 3}) ∧
  ((Bᶜ) ∩ A = {x | 1 ≤ x ∧ x ≤ 2}) ∧
  (∀ a : ℝ, C a ⊆ A → a ≤ 3) := by sorry

end NUMINAMATH_CALUDE_set_operations_and_subset_l1088_108856


namespace NUMINAMATH_CALUDE_fifth_power_sum_l1088_108859

theorem fifth_power_sum (a b x y : ℝ) 
  (eq1 : a * x + b * y = 3)
  (eq2 : a * x^2 + b * y^2 = 7)
  (eq3 : a * x^3 + b * y^3 = 16)
  (eq4 : a * x^4 + b * y^4 = 42) :
  a * x^5 + b * y^5 = 99 := by
sorry

end NUMINAMATH_CALUDE_fifth_power_sum_l1088_108859


namespace NUMINAMATH_CALUDE_distance_to_market_is_40_l1088_108866

/-- The distance between Andy's house and the market -/
def distance_to_market (distance_to_school : ℕ) (total_distance : ℕ) : ℕ :=
  total_distance - 2 * distance_to_school

/-- Theorem: Given the conditions, the distance to the market is 40 meters -/
theorem distance_to_market_is_40 :
  let distance_to_school : ℕ := 50
  let total_distance : ℕ := 140
  distance_to_market distance_to_school total_distance = 40 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_market_is_40_l1088_108866


namespace NUMINAMATH_CALUDE_tangent_line_implies_a_equals_two_l1088_108854

-- Define the curve and tangent line
def curve (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x + 1
def tangent_line (x : ℝ) : ℝ := 2*x + 1

-- Theorem statement
theorem tangent_line_implies_a_equals_two (a : ℝ) :
  (∃ x₀ : ℝ, curve a x₀ = tangent_line x₀ ∧ 
    ∀ x : ℝ, x ≠ x₀ → curve a x ≠ tangent_line x) →
  a = 2 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_implies_a_equals_two_l1088_108854


namespace NUMINAMATH_CALUDE_ursula_annual_salary_l1088_108869

/-- Calculates the annual salary given hourly wage, hours per day, and days per month -/
def annual_salary (hourly_wage : ℝ) (hours_per_day : ℝ) (days_per_month : ℝ) : ℝ :=
  hourly_wage * hours_per_day * days_per_month * 12

/-- Proves that Ursula's annual salary is $16,320 given her work conditions -/
theorem ursula_annual_salary :
  annual_salary 8.50 8 20 = 16320 := by
  sorry

end NUMINAMATH_CALUDE_ursula_annual_salary_l1088_108869
