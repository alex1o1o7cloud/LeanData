import Mathlib

namespace NUMINAMATH_CALUDE_meeting_equation_correct_l2913_291327

/-- Represents the scenario of two people meeting on a straight road -/
def meeting_equation (x : ℝ) : Prop :=
  x / 6 + (x - 1) / 4 = 1

/-- The time it takes for A to travel the entire distance -/
def time_A : ℝ := 4

/-- The time it takes for B to travel the entire distance -/
def time_B : ℝ := 6

/-- The time difference between A and B starting their journey -/
def time_difference : ℝ := 1

/-- Theorem stating that the meeting equation correctly represents the scenario -/
theorem meeting_equation_correct (x : ℝ) :
  (x ≥ time_difference) →
  (x / time_B + (x - time_difference) / time_A = 1) ↔ meeting_equation x :=
by sorry

end NUMINAMATH_CALUDE_meeting_equation_correct_l2913_291327


namespace NUMINAMATH_CALUDE_fast_site_selection_probability_l2913_291318

theorem fast_site_selection_probability (total : ℕ) (guizhou : ℕ) (selected : ℕ)
  (h1 : total = 8)
  (h2 : guizhou = 3)
  (h3 : selected = 2)
  (h4 : guizhou ≤ total) :
  (Nat.choose guizhou 1 * Nat.choose (total - guizhou) 1 + Nat.choose guizhou 2) / Nat.choose total selected = 9 / 14 :=
by sorry

end NUMINAMATH_CALUDE_fast_site_selection_probability_l2913_291318


namespace NUMINAMATH_CALUDE_expression_evaluation_l2913_291393

theorem expression_evaluation : (10^9) / ((2 * 10^6) * 3) = 500/3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2913_291393


namespace NUMINAMATH_CALUDE_alicia_wages_l2913_291399

/- Define the hourly wage in dollars -/
def hourly_wage : ℚ := 25

/- Define the local tax rate as a percentage -/
def tax_rate : ℚ := 2.5

/- Define the conversion rate from dollars to cents -/
def cents_per_dollar : ℕ := 100

/- Theorem statement -/
theorem alicia_wages :
  let wage_in_cents := hourly_wage * cents_per_dollar
  let tax_amount := (tax_rate / 100) * wage_in_cents
  let after_tax_earnings := wage_in_cents - tax_amount
  (tax_amount = 62.5 ∧ after_tax_earnings = 2437.5) := by
  sorry

end NUMINAMATH_CALUDE_alicia_wages_l2913_291399


namespace NUMINAMATH_CALUDE_random_placement_probability_l2913_291314

-- Define the number of bins and items
def num_bins : ℕ := 4
def num_items : ℕ := 4

-- Define the probability of correct placement
def correct_placement_probability : ℚ := 1 / (num_bins.factorial)

-- Theorem statement
theorem random_placement_probability :
  correct_placement_probability = 1 / 24 := by
  sorry

end NUMINAMATH_CALUDE_random_placement_probability_l2913_291314


namespace NUMINAMATH_CALUDE_alyssa_pears_l2913_291377

theorem alyssa_pears (total_pears nancy_pears : ℕ) 
  (h1 : total_pears = 59) 
  (h2 : nancy_pears = 17) : 
  total_pears - nancy_pears = 42 := by
  sorry

end NUMINAMATH_CALUDE_alyssa_pears_l2913_291377


namespace NUMINAMATH_CALUDE_bella_earrings_l2913_291391

/-- Given three friends Bella, Monica, and Rachel, with the following conditions:
    1. Bella has 25% of Monica's earrings
    2. Monica has twice as many earrings as Rachel
    3. The total number of earrings among the three friends is 70
    Prove that Bella has 10 earrings. -/
theorem bella_earrings (bella monica rachel : ℕ) : 
  bella = (25 : ℕ) * monica / 100 →
  monica = 2 * rachel →
  bella + monica + rachel = 70 →
  bella = 10 := by
sorry

end NUMINAMATH_CALUDE_bella_earrings_l2913_291391


namespace NUMINAMATH_CALUDE_quotient_problem_l2913_291317

theorem quotient_problem (L S Q : ℕ) : 
  L - S = 1365 → 
  L = 1620 → 
  L = S * Q + 15 → 
  Q = 6 := by
sorry

end NUMINAMATH_CALUDE_quotient_problem_l2913_291317


namespace NUMINAMATH_CALUDE_our_number_not_perfect_square_l2913_291367

-- Define a perfect square
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

-- Define the number we want to prove is not a perfect square
def our_number : ℕ := 4^2021

-- Theorem statement
theorem our_number_not_perfect_square : ¬ (is_perfect_square our_number) := by
  sorry

end NUMINAMATH_CALUDE_our_number_not_perfect_square_l2913_291367


namespace NUMINAMATH_CALUDE_distance_from_origin_l2913_291341

theorem distance_from_origin (a : ℝ) : |a| = 4 → a = 4 ∨ a = -4 := by sorry

end NUMINAMATH_CALUDE_distance_from_origin_l2913_291341


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l2913_291383

theorem smallest_n_satisfying_conditions : 
  ∃ n : ℕ, n > 20 ∧ n % 6 = 5 ∧ n % 7 = 3 ∧ 
  ∀ m : ℕ, m > 20 ∧ m % 6 = 5 ∧ m % 7 = 3 → n ≤ m :=
by
  use 59
  sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l2913_291383


namespace NUMINAMATH_CALUDE_polygon_intersection_points_l2913_291354

/-- The number of intersection points between two regular polygons inscribed in a circle -/
def intersectionPoints (n m : ℕ) : ℕ := 2 * min n m

/-- The total number of intersection points for four regular polygons -/
def totalIntersectionPoints (a b c d : ℕ) : ℕ :=
  intersectionPoints a b + intersectionPoints a c + intersectionPoints a d +
  intersectionPoints b c + intersectionPoints b d + intersectionPoints c d

theorem polygon_intersection_points :
  totalIntersectionPoints 6 7 8 9 = 80 := by
  sorry

#eval totalIntersectionPoints 6 7 8 9

end NUMINAMATH_CALUDE_polygon_intersection_points_l2913_291354


namespace NUMINAMATH_CALUDE_distance_to_nearest_city_l2913_291350

theorem distance_to_nearest_city (d : ℝ) : 
  (¬(d ≥ 8)) ∧ (¬(d ≤ 7)) ∧ (¬(d ≤ 6)) ∧ (d ≠ 10) → 7 < d ∧ d < 8 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_nearest_city_l2913_291350


namespace NUMINAMATH_CALUDE_greatest_multiple_of_8_remainder_l2913_291326

def is_valid_number (n : ℕ) : Prop :=
  ∀ d₁ d₂, d₁ ∈ n.digits 10 → d₂ ∈ n.digits 10 → d₁ ≠ d₂ → d₁ ≠ 0 ∧ d₂ ≠ 0

theorem greatest_multiple_of_8_remainder (M : ℕ) : 
  (∀ n, n > M → ¬(is_valid_number n ∧ 8 ∣ n)) →
  is_valid_number M →
  8 ∣ M →
  M % 1000 = 984 :=
sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_8_remainder_l2913_291326


namespace NUMINAMATH_CALUDE_megan_snacks_l2913_291300

/-- The number of snacks Megan has in a given time period -/
def num_snacks (snack_interval : ℕ) (total_minutes : ℕ) : ℕ :=
  total_minutes / snack_interval

theorem megan_snacks : num_snacks 20 220 = 11 := by
  sorry

end NUMINAMATH_CALUDE_megan_snacks_l2913_291300


namespace NUMINAMATH_CALUDE_f_of_2_equals_12_l2913_291337

-- Define the function f
def f (x : ℝ) : ℝ := 5 * x + 2

-- State the theorem
theorem f_of_2_equals_12 : f 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_f_of_2_equals_12_l2913_291337


namespace NUMINAMATH_CALUDE_domain_of_k_l2913_291331

def h (x : ℝ) : ℝ := sorry

def k (x : ℝ) : ℝ := h (-3 * x + 1)

theorem domain_of_k :
  ∀ x : ℝ, x ∈ Set.Icc (11/3) (5/3) ↔ (-3 * x + 1) ∈ Set.Icc (-10) 6 := by
  sorry

end NUMINAMATH_CALUDE_domain_of_k_l2913_291331


namespace NUMINAMATH_CALUDE_total_games_won_l2913_291348

def bulls_games : ℕ := 70
def heat_games : ℕ := bulls_games + 5

theorem total_games_won : bulls_games + heat_games = 145 := by
  sorry

end NUMINAMATH_CALUDE_total_games_won_l2913_291348


namespace NUMINAMATH_CALUDE_count_triples_eq_200_l2913_291372

/-- Counts the number of ways to partition a positive integer into two positive integers -/
def partitionCount (n : ℕ) : ℕ := if n ≤ 1 then 0 else n - 1

/-- Counts the number of ordered triples (a,b,c) satisfying the given conditions -/
def countTriples : ℕ :=
  (partitionCount 3) + (partitionCount 4) + (partitionCount 9) +
  (partitionCount 19) + (partitionCount 24) + (partitionCount 49) +
  (partitionCount 99)

theorem count_triples_eq_200 :
  countTriples = 200 :=
sorry

end NUMINAMATH_CALUDE_count_triples_eq_200_l2913_291372


namespace NUMINAMATH_CALUDE_max_area_rectangle_l2913_291378

/-- The maximum area of a rectangle with a perimeter of 40 inches is 100 square inches. -/
theorem max_area_rectangle (x y : ℝ) (h_perimeter : x + y = 20) :
  x * y ≤ 100 ∧ ∃ (a b : ℝ), a + b = 20 ∧ a * b = 100 :=
by sorry

end NUMINAMATH_CALUDE_max_area_rectangle_l2913_291378


namespace NUMINAMATH_CALUDE_right_quadrilateral_area_l2913_291312

/-- A quadrilateral with right angles at B and D, diagonal AC = 5, and two sides with distinct integer lengths -/
structure RightQuadrilateral where
  AB : ℕ
  BC : ℕ
  CD : ℕ
  DA : ℕ
  AC_eq_five : AB^2 + BC^2 = 25
  distinct_sides : (AB ≠ BC) ∧ (AB ≠ CD) ∧ (AB ≠ DA) ∧ (BC ≠ CD) ∧ (BC ≠ DA) ∧ (CD ≠ DA)

/-- The area of a RightQuadrilateral is 12 -/
theorem right_quadrilateral_area (q : RightQuadrilateral) : (q.AB * q.BC + q.CD * q.DA) / 2 = 12 := by
  sorry


end NUMINAMATH_CALUDE_right_quadrilateral_area_l2913_291312


namespace NUMINAMATH_CALUDE_cyclic_n_gon_characterization_l2913_291320

/-- A convex n-gon is cyclic if and only if there exist real numbers a_i and b_i
    for each vertex P_i such that for any i < j, the distance P_i P_j = |a_i b_j - a_j b_i|. -/
theorem cyclic_n_gon_characterization {n : ℕ} (P : Fin n → ℝ × ℝ) :
  (∃ (center : ℝ × ℝ) (radius : ℝ), ∀ i : Fin n, dist center (P i) = radius) ↔
  (∃ (a b : Fin n → ℝ), ∀ (i j : Fin n), i < j →
    dist (P i) (P j) = |a i * b j - a j * b i|) :=
by sorry

end NUMINAMATH_CALUDE_cyclic_n_gon_characterization_l2913_291320


namespace NUMINAMATH_CALUDE_sector_area_l2913_291343

theorem sector_area (θ : ℝ) (p : ℝ) (h1 : θ = 2) (h2 : p = 4) :
  let r := (p - θ) / 2
  let area := r^2 * θ / 2
  area = 1 := by sorry

end NUMINAMATH_CALUDE_sector_area_l2913_291343


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2913_291386

theorem inequality_system_solution (a b : ℝ) : 
  (∀ x, (-1 < x ∧ x < 1) ↔ (2*x - a < 1 ∧ x - 2*b > 3)) → 
  (a + 1) * (b - 1) = -6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2913_291386


namespace NUMINAMATH_CALUDE_isosceles_triangle_area_l2913_291398

/-- Represents an isosceles triangle with vertex angle 80°, leg length a, and base length b -/
structure IsoscelesTriangle where
  a : ℝ  -- length of the legs
  b : ℝ  -- length of the base
  h₁ : a > 0
  h₂ : b > 0

/-- Calculates the area of an isosceles triangle -/
noncomputable def triangleArea (t : IsoscelesTriangle) : ℝ :=
  (t.a^3 * t.b) / (4 * (t.b^2 - t.a^2))

/-- Theorem stating that the area of the isosceles triangle with vertex angle 80° is (a^3 * b) / (4 * (b^2 - a^2)) -/
theorem isosceles_triangle_area (t : IsoscelesTriangle) :
  triangleArea t = (t.a^3 * t.b) / (4 * (t.b^2 - t.a^2)) := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_area_l2913_291398


namespace NUMINAMATH_CALUDE_f_8_equals_60_l2913_291301

-- Define the function f
def f (n : ℤ) : ℤ := n^2 - 3*n + 20

-- Theorem statement
theorem f_8_equals_60 : f 8 = 60 := by
  sorry

end NUMINAMATH_CALUDE_f_8_equals_60_l2913_291301


namespace NUMINAMATH_CALUDE_triangles_drawn_l2913_291345

theorem triangles_drawn (squares pentagons total_lines : ℕ) 
  (h_squares : squares = 8)
  (h_pentagons : pentagons = 4)
  (h_total_lines : total_lines = 88) :
  ∃ (triangles : ℕ), 
    3 * triangles + 4 * squares + 5 * pentagons = total_lines ∧ 
    triangles = 12 := by
  sorry

end NUMINAMATH_CALUDE_triangles_drawn_l2913_291345


namespace NUMINAMATH_CALUDE_mask_probability_l2913_291390

theorem mask_probability (regular_ratio surgical_ratio regular_ear_loop_ratio surgical_ear_loop_ratio : Real) 
  (h1 : regular_ratio = 0.8)
  (h2 : surgical_ratio = 0.2)
  (h3 : regular_ear_loop_ratio = 0.1)
  (h4 : surgical_ear_loop_ratio = 0.2)
  (h5 : regular_ratio + surgical_ratio = 1) :
  regular_ratio * regular_ear_loop_ratio + surgical_ratio * surgical_ear_loop_ratio = 0.12 := by
sorry

end NUMINAMATH_CALUDE_mask_probability_l2913_291390


namespace NUMINAMATH_CALUDE_tile_arrangement_probability_l2913_291360

theorem tile_arrangement_probability : 
  let total_tiles : ℕ := 7
  let x_tiles : ℕ := 4
  let o_tiles : ℕ := 3
  let favorable_arrangements : ℕ := Nat.choose 4 2
  let total_arrangements : ℕ := Nat.choose total_tiles x_tiles
  (favorable_arrangements : ℚ) / total_arrangements = 6 / 35 := by
sorry

end NUMINAMATH_CALUDE_tile_arrangement_probability_l2913_291360


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l2913_291379

theorem complex_number_in_fourth_quadrant :
  let i : ℂ := Complex.I
  let z : ℂ := (2 * i^3) / (1 - i)
  (z.re > 0 ∧ z.im < 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l2913_291379


namespace NUMINAMATH_CALUDE_child_b_share_child_b_share_is_552_l2913_291362

/-- Calculates the share of child B given the total amount, tax rate, interest rate, and distribution ratio. -/
theorem child_b_share (total_amount : ℝ) (tax_rate : ℝ) (interest_rate : ℝ) (ratio_a ratio_b ratio_c : ℕ) : ℝ :=
  let tax := total_amount * tax_rate
  let interest := total_amount * interest_rate
  let remaining_amount := total_amount - (tax + interest)
  let total_parts := ratio_a + ratio_b + ratio_c
  let part_value := remaining_amount / total_parts
  ratio_b * part_value

/-- Proves that given the specific conditions, B's share is $552. -/
theorem child_b_share_is_552 : 
  child_b_share 1800 0.05 0.03 2 3 4 = 552 := by
  sorry

end NUMINAMATH_CALUDE_child_b_share_child_b_share_is_552_l2913_291362


namespace NUMINAMATH_CALUDE_sqrt_500_simplification_l2913_291334

theorem sqrt_500_simplification : Real.sqrt 500 = 10 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_sqrt_500_simplification_l2913_291334


namespace NUMINAMATH_CALUDE_tom_pennies_l2913_291375

/-- Represents the number of coins of each type --/
structure CoinCounts where
  quarters : ℕ
  dimes : ℕ
  nickels : ℕ
  pennies : ℕ

/-- Calculates the total value in cents given a CoinCounts --/
def totalValueInCents (coins : CoinCounts) : ℕ :=
  coins.quarters * 25 + coins.dimes * 10 + coins.nickels * 5 + coins.pennies

/-- The main theorem --/
theorem tom_pennies (coins : CoinCounts) 
    (h1 : coins.quarters = 10)
    (h2 : coins.dimes = 3)
    (h3 : coins.nickels = 4)
    (h4 : totalValueInCents coins = 500) :
    coins.pennies = 200 := by
  sorry


end NUMINAMATH_CALUDE_tom_pennies_l2913_291375


namespace NUMINAMATH_CALUDE_simplified_expression_equals_one_l2913_291357

theorem simplified_expression_equals_one (a : ℚ) (h : a = 1/2) :
  (1 / (a + 2) + 1 / (a - 2)) / (1 / (a^2 - 4)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplified_expression_equals_one_l2913_291357


namespace NUMINAMATH_CALUDE_bicycle_count_l2913_291315

theorem bicycle_count (tricycles : ℕ) (bicycle_wheels : ℕ) (tricycle_wheels : ℕ) (total_wheels : ℕ) :
  tricycles = 7 →
  bicycle_wheels = 2 →
  tricycle_wheels = 3 →
  total_wheels = 53 →
  ∃ bicycles : ℕ, bicycles * bicycle_wheels + tricycles * tricycle_wheels = total_wheels ∧ bicycles = 16 :=
by sorry

end NUMINAMATH_CALUDE_bicycle_count_l2913_291315


namespace NUMINAMATH_CALUDE_band_tryouts_l2913_291347

theorem band_tryouts (flutes clarinets trumpets pianists : ℕ) : 
  flutes = 20 →
  clarinets = 30 →
  pianists = 20 →
  (80 : ℚ) / 100 * flutes + 1 / 2 * clarinets + 1 / 3 * trumpets + 1 / 10 * pianists = 53 →
  trumpets = 60 :=
by sorry

end NUMINAMATH_CALUDE_band_tryouts_l2913_291347


namespace NUMINAMATH_CALUDE_right_triangle_area_l2913_291303

/-- A right triangle with vertices at (0, 0), (0, 10), and (-10, 0), 
    and two points (-3, 7) and (-7, 3) on its hypotenuse. -/
structure RightTriangle where
  -- Define the vertices
  v1 : ℝ × ℝ := (0, 0)
  v2 : ℝ × ℝ := (0, 10)
  v3 : ℝ × ℝ := (-10, 0)
  -- Define the points on the hypotenuse
  p1 : ℝ × ℝ := (-3, 7)
  p2 : ℝ × ℝ := (-7, 3)
  -- Ensure the triangle is right-angled
  is_right_angle : (v2.1 - v1.1) * (v3.1 - v1.1) + (v2.2 - v1.2) * (v3.2 - v1.2) = 0
  -- Ensure the points lie on the hypotenuse
  p1_on_hypotenuse : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p1 = (t * v2.1 + (1 - t) * v3.1, t * v2.2 + (1 - t) * v3.2)
  p2_on_hypotenuse : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ p2 = (s * v2.1 + (1 - s) * v3.1, s * v2.2 + (1 - s) * v3.2)

/-- The area of the right triangle is 50 square units. -/
theorem right_triangle_area (t : RightTriangle) : 
  (1/2) * abs (t.v2.1 * t.v3.2 - t.v3.1 * t.v2.2) = 50 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2913_291303


namespace NUMINAMATH_CALUDE_divisibility_by_eleven_l2913_291336

theorem divisibility_by_eleven (n : ℤ) : 
  (11 : ℤ) ∣ ((n + 11)^2 - n^2) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_eleven_l2913_291336


namespace NUMINAMATH_CALUDE_jakes_peaches_l2913_291385

theorem jakes_peaches (steven jill jake : ℕ) : 
  steven = 16 →
  jake < steven →
  jake = jill + 9 →
  ∃ (l : ℕ), jake = l + 9 :=
by sorry

end NUMINAMATH_CALUDE_jakes_peaches_l2913_291385


namespace NUMINAMATH_CALUDE_division_equality_l2913_291321

theorem division_equality : (49 : ℝ) / 0.07 = 700 := by
  sorry

end NUMINAMATH_CALUDE_division_equality_l2913_291321


namespace NUMINAMATH_CALUDE_sandbox_cost_l2913_291376

/-- Calculates the cost of filling an L-shaped sandbox with sand -/
theorem sandbox_cost (short_length short_width short_depth long_length long_width long_depth sand_cost discount_threshold discount_rate : ℝ) :
  let short_volume := short_length * short_width * short_depth
  let long_volume := long_length * long_width * long_depth
  let total_volume := short_volume + long_volume
  let base_cost := total_volume * sand_cost
  let discounted_cost := if total_volume > discount_threshold then base_cost * (1 - discount_rate) else base_cost
  short_length = 3 ∧ 
  short_width = 2 ∧ 
  short_depth = 2 ∧ 
  long_length = 5 ∧ 
  long_width = 2 ∧ 
  long_depth = 2 ∧ 
  sand_cost = 3 ∧ 
  discount_threshold = 20 ∧ 
  discount_rate = 0.1 →
  discounted_cost = 86.4 := by
  sorry

end NUMINAMATH_CALUDE_sandbox_cost_l2913_291376


namespace NUMINAMATH_CALUDE_simplify_trig_expression_simplify_trig_product_l2913_291302

-- Part 1
theorem simplify_trig_expression (α : Real) :
  (Real.sin (α - π/2) + Real.cos (3*π/2 + α)) / (Real.sin (π - α) + Real.cos (3*π + α)) =
  1 / 0 := by sorry

-- Part 2
theorem simplify_trig_product :
  Real.sin (40 * π/180) * (Real.tan (10 * π/180) - Real.sqrt 3) =
  -Real.sin (80 * π/180) / Real.cos (10 * π/180) := by sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_simplify_trig_product_l2913_291302


namespace NUMINAMATH_CALUDE_right_angle_complementary_angle_l2913_291369

theorem right_angle_complementary_angle (x : ℝ) : 
  x + 23 = 90 → x = 67 := by
  sorry

end NUMINAMATH_CALUDE_right_angle_complementary_angle_l2913_291369


namespace NUMINAMATH_CALUDE_weekend_rain_probability_l2913_291394

theorem weekend_rain_probability
  (p_rain_saturday : ℝ)
  (p_rain_sunday : ℝ)
  (p_rain_sunday_given_no_saturday : ℝ)
  (h1 : p_rain_saturday = 0.6)
  (h2 : p_rain_sunday = 0.4)
  (h3 : p_rain_sunday_given_no_saturday = 0.7)
  : ℝ :=
by
  -- Probability of rain over the weekend
  sorry

#check weekend_rain_probability

end NUMINAMATH_CALUDE_weekend_rain_probability_l2913_291394


namespace NUMINAMATH_CALUDE_lawrence_county_houses_l2913_291332

/-- The number of houses in Lawrence County before the housing boom -/
def houses_before : ℕ := 1426

/-- The number of houses built during the housing boom -/
def houses_built : ℕ := 574

/-- The total number of houses in Lawrence County after the housing boom -/
def total_houses : ℕ := houses_before + houses_built

theorem lawrence_county_houses : total_houses = 2000 := by
  sorry

end NUMINAMATH_CALUDE_lawrence_county_houses_l2913_291332


namespace NUMINAMATH_CALUDE_increase_by_percentage_l2913_291311

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (result : ℝ) : 
  initial = 80 → percentage = 150 → result = initial * (1 + percentage / 100) → result = 200 := by
  sorry

end NUMINAMATH_CALUDE_increase_by_percentage_l2913_291311


namespace NUMINAMATH_CALUDE_cement_calculation_l2913_291313

/-- The amount of cement originally owned -/
def original_cement : ℕ := sorry

/-- The amount of cement bought -/
def bought_cement : ℕ := 215

/-- The amount of cement brought by the son -/
def son_brought_cement : ℕ := 137

/-- The current total amount of cement -/
def current_cement : ℕ := 450

/-- Theorem stating the relationship between the amounts of cement -/
theorem cement_calculation : 
  original_cement = current_cement - (bought_cement + son_brought_cement) :=
by sorry

end NUMINAMATH_CALUDE_cement_calculation_l2913_291313


namespace NUMINAMATH_CALUDE_city_visit_selection_schemes_l2913_291397

theorem city_visit_selection_schemes :
  let total_people : ℕ := 6
  let selected_people : ℕ := 4
  let total_cities : ℕ := 4
  let restricted_people : ℕ := 2
  let restricted_cities : ℕ := 1

  (total_people - restricted_people) *
  (total_people - 1) *
  (total_people - 2) *
  (total_people - 3) = 240 :=
by
  sorry

end NUMINAMATH_CALUDE_city_visit_selection_schemes_l2913_291397


namespace NUMINAMATH_CALUDE_supplement_of_half_angle_l2913_291351

-- Define the angle α
def α : ℝ := 90 - 50

-- Theorem statement
theorem supplement_of_half_angle (h : α = 90 - 50) : 
  180 - (α / 2) = 160 := by sorry

end NUMINAMATH_CALUDE_supplement_of_half_angle_l2913_291351


namespace NUMINAMATH_CALUDE_john_ate_three_slices_l2913_291339

/-- Represents the number of slices in a pizza -/
def total_slices : ℕ := 12

/-- Represents the number of slices left -/
def slices_left : ℕ := 3

/-- Represents the number of slices John ate -/
def john_slices : ℕ := 3

/-- Represents the number of slices Sam ate -/
def sam_slices : ℕ := 2 * john_slices

theorem john_ate_three_slices :
  john_slices = 3 ∧
  sam_slices = 2 * john_slices ∧
  total_slices = john_slices + sam_slices + slices_left :=
by sorry

end NUMINAMATH_CALUDE_john_ate_three_slices_l2913_291339


namespace NUMINAMATH_CALUDE_triangle_circle_tangent_l2913_291388

theorem triangle_circle_tangent (a b c : ℝ) (x : ℝ) :
  -- Triangle ABC is a right triangle
  a^2 = b^2 + c^2 →
  -- Perimeter of triangle ABC is 190
  a + b + c = 190 →
  -- Circle with radius 23 centered at O on AB is tangent to BC
  (b - x) / b = 23 / a →
  -- AO = x (where O is the center of the circle)
  x^2 + (b - x)^2 = c^2 →
  -- The length of AO is 67
  x = 67 := by
    sorry

#eval 67 + 1  -- x + y = 68

end NUMINAMATH_CALUDE_triangle_circle_tangent_l2913_291388


namespace NUMINAMATH_CALUDE_max_servings_is_twelve_l2913_291368

/-- Represents the number of servings that can be made from a given ingredient --/
def ServingsFromIngredient (available : ℕ) (required : ℕ) : ℕ :=
  (available * 4) / required

/-- Represents the recipe and available ingredients --/
structure SmoothieRecipe where
  bananas_required : ℕ
  yogurt_required : ℕ
  strawberries_required : ℕ
  bananas_available : ℕ
  yogurt_available : ℕ
  strawberries_available : ℕ

/-- Calculates the maximum number of servings that can be made --/
def MaxServings (recipe : SmoothieRecipe) : ℕ :=
  min (ServingsFromIngredient recipe.bananas_available recipe.bananas_required)
    (min (ServingsFromIngredient recipe.yogurt_available recipe.yogurt_required)
      (ServingsFromIngredient recipe.strawberries_available recipe.strawberries_required))

theorem max_servings_is_twelve :
  ∀ (recipe : SmoothieRecipe),
    recipe.bananas_required = 3 →
    recipe.yogurt_required = 2 →
    recipe.strawberries_required = 1 →
    recipe.bananas_available = 9 →
    recipe.yogurt_available = 10 →
    recipe.strawberries_available = 3 →
    MaxServings recipe = 12 := by
  sorry

end NUMINAMATH_CALUDE_max_servings_is_twelve_l2913_291368


namespace NUMINAMATH_CALUDE_motion_analysis_l2913_291392

-- Define the motion function
def s (t : ℝ) : ℝ := t^2 + 2*t - 3

-- Define velocity as the derivative of s
def v (t : ℝ) : ℝ := 2*t + 2

-- Define acceleration as the derivative of v
def a : ℝ := 2

theorem motion_analysis :
  v 2 = 6 ∧ a = 2 :=
sorry

end NUMINAMATH_CALUDE_motion_analysis_l2913_291392


namespace NUMINAMATH_CALUDE_seven_eighths_of_sixteen_thirds_l2913_291329

theorem seven_eighths_of_sixteen_thirds :
  (7 / 8 : ℚ) * (16 / 3 : ℚ) = 14 / 3 := by
  sorry

end NUMINAMATH_CALUDE_seven_eighths_of_sixteen_thirds_l2913_291329


namespace NUMINAMATH_CALUDE_lily_of_valley_cost_price_l2913_291373

/-- The cost price of a pot of lily of the valley -/
def cost_price : ℝ := 2.4

/-- The selling price of a pot of lily of the valley -/
def selling_price : ℝ := cost_price * 1.25

/-- The number of pots sold -/
def num_pots : ℕ := 150

/-- The total revenue from selling the pots -/
def total_revenue : ℝ := 450

theorem lily_of_valley_cost_price :
  cost_price = 2.4 ∧
  selling_price = cost_price * 1.25 ∧
  (num_pots : ℝ) * selling_price = total_revenue :=
sorry

end NUMINAMATH_CALUDE_lily_of_valley_cost_price_l2913_291373


namespace NUMINAMATH_CALUDE_greatest_n_value_exists_n_value_l2913_291340

theorem greatest_n_value (n : ℤ) (h : 101 * n^2 ≤ 12100) : n ≤ 10 := by
  sorry

theorem exists_n_value : ∃ (n : ℤ), 101 * n^2 ≤ 12100 ∧ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_greatest_n_value_exists_n_value_l2913_291340


namespace NUMINAMATH_CALUDE_wire_length_ratio_l2913_291307

theorem wire_length_ratio : 
  let large_cube_edge : ℝ := 8
  let large_cube_edges : ℕ := 12
  let unit_cube_edge : ℝ := 1
  let unit_cube_edges : ℕ := 12

  let large_cube_volume := large_cube_edge ^ 3
  let num_unit_cubes := large_cube_volume

  let large_cube_wire_length := large_cube_edge * large_cube_edges
  let unit_cubes_wire_length := num_unit_cubes * unit_cube_edge * unit_cube_edges

  large_cube_wire_length / unit_cubes_wire_length = 1 / 64 :=
by
  sorry

end NUMINAMATH_CALUDE_wire_length_ratio_l2913_291307


namespace NUMINAMATH_CALUDE_paint_mixture_ratio_l2913_291371

/-- Given a paint mixture with a ratio of 5:3:7 for red:yellow:white paint,
    if 21 quarts of white paint is used, then 9 quarts of yellow paint should be used. -/
theorem paint_mixture_ratio (red yellow white : ℚ) :
  red / yellow = 5 / 3 →
  yellow / white = 3 / 7 →
  white = 21 →
  yellow = 9 := by
  sorry

end NUMINAMATH_CALUDE_paint_mixture_ratio_l2913_291371


namespace NUMINAMATH_CALUDE_polyhedron_relations_l2913_291305

structure Polyhedron where
  E : ℕ  -- number of edges
  F : ℕ  -- number of faces
  V : ℕ  -- number of vertices
  n : ℕ  -- number of sides in each face
  m : ℕ  -- number of edges meeting at each vertex

theorem polyhedron_relations (P : Polyhedron) : 
  (P.n * P.F = 2 * P.E) ∧ 
  (P.m * P.V = 2 * P.E) ∧ 
  (P.V + P.F = P.E + 2) ∧ 
  ¬(P.m * P.F = 2 * P.E) := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_relations_l2913_291305


namespace NUMINAMATH_CALUDE_existence_of_alpha_for_tan_l2913_291316

open Real

theorem existence_of_alpha_for_tan : ∃ α : ℝ, 
  (∃ α₀ : ℝ, tan (π / 2 - α₀) = 1) ∧ 
  (¬∀ α₁ : ℝ, tan (π / 2 - α₁) = 1) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_alpha_for_tan_l2913_291316


namespace NUMINAMATH_CALUDE_cos_six_arccos_one_fourth_l2913_291330

theorem cos_six_arccos_one_fourth : 
  Real.cos (6 * Real.arccos (1/4)) = -7/128 := by
  sorry

end NUMINAMATH_CALUDE_cos_six_arccos_one_fourth_l2913_291330


namespace NUMINAMATH_CALUDE_spinner_probability_l2913_291359

theorem spinner_probability (pA pB pC pD pE : ℚ) : 
  pA = 3/8 →
  pB = 1/8 →
  pC = pD →
  pC = pE →
  pA + pB + pC + pD + pE = 1 →
  pC = 1/6 := by
sorry

end NUMINAMATH_CALUDE_spinner_probability_l2913_291359


namespace NUMINAMATH_CALUDE_john_remaining_money_l2913_291352

def trip_finances (initial_amount spent_amount remaining_amount : ℕ) : Prop :=
  (initial_amount = 1600) ∧
  (remaining_amount = spent_amount - 600) ∧
  (remaining_amount = initial_amount - spent_amount)

theorem john_remaining_money :
  ∃ (spent_amount remaining_amount : ℕ),
    trip_finances 1600 spent_amount remaining_amount ∧
    remaining_amount = 500 :=
by
  sorry

end NUMINAMATH_CALUDE_john_remaining_money_l2913_291352


namespace NUMINAMATH_CALUDE_degrees_90_to_radians_l2913_291374

/-- Conversion of 90 degrees to radians -/
theorem degrees_90_to_radians : 
  (90 : ℝ) * (Real.pi / 180) = Real.pi / 2 := by sorry

end NUMINAMATH_CALUDE_degrees_90_to_radians_l2913_291374


namespace NUMINAMATH_CALUDE_greatest_prime_factor_factorial_sum_l2913_291358

theorem greatest_prime_factor_factorial_sum : 
  (Nat.factors (Nat.factorial 15 + Nat.factorial 18)).maximum? = some 17 := by
  sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_factorial_sum_l2913_291358


namespace NUMINAMATH_CALUDE_factorial_500_properties_l2913_291355

/-- The number of trailing zeroes in n! -/
def trailingZeroes (n : ℕ) : ℕ := sorry

/-- The highest power of 3 that divides n! -/
def highestPowerOfThree (n : ℕ) : ℕ := sorry

/-- Theorem about 500! -/
theorem factorial_500_properties :
  (trailingZeroes 500 = 124) ∧ (highestPowerOfThree 500 = 247) := by sorry

end NUMINAMATH_CALUDE_factorial_500_properties_l2913_291355


namespace NUMINAMATH_CALUDE_line_tangent_to_ellipse_l2913_291319

/-- Theorem: If a line y = mx + 3 is tangent to the ellipse x² + 9y² = 9, then m² = 8/9 -/
theorem line_tangent_to_ellipse (m : ℝ) : 
  (∃! x y : ℝ, y = m * x + 3 ∧ x^2 + 9 * y^2 = 9) → m^2 = 8/9 := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_ellipse_l2913_291319


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sum_l2913_291361

/-- Given an arithmetic sequence {a_n} with a₁ = 1, common difference d ≠ 0,
    and a₁, a₂, and a₅ forming a geometric sequence, 
    prove that the sum of the first 8 terms (S₈) is equal to 64. -/
theorem arithmetic_geometric_sum (d : ℝ) (h1 : d ≠ 0) : 
  let a : ℕ → ℝ := fun n => 1 + (n - 1) * d
  let S : ℕ → ℝ := fun n => (n * (2 + (n - 1) * d)) / 2
  (a 2)^2 = (a 1) * (a 5) → S 8 = 64 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sum_l2913_291361


namespace NUMINAMATH_CALUDE_vinnie_saturday_words_l2913_291323

/-- The number of words Vinnie wrote on Saturday -/
def saturday_words : ℕ := sorry

/-- The word limit -/
def word_limit : ℕ := 1000

/-- The number of words Vinnie wrote on Sunday -/
def sunday_words : ℕ := 650

/-- The number of words Vinnie exceeded the limit by -/
def excess_words : ℕ := 100

/-- Theorem stating that Vinnie wrote 450 words on Saturday -/
theorem vinnie_saturday_words :
  saturday_words = 450 ∧
  saturday_words + sunday_words = word_limit + excess_words :=
sorry

end NUMINAMATH_CALUDE_vinnie_saturday_words_l2913_291323


namespace NUMINAMATH_CALUDE_price_difference_in_cents_l2913_291356

-- Define the list price and discounts
def list_price : ℚ := 5999 / 100  -- $59.99 represented as a rational number
def tech_bargains_discount : ℚ := 15  -- $15 off
def budget_bytes_discount_rate : ℚ := 30 / 100  -- 30% off

-- Calculate the sale prices
def tech_bargains_price : ℚ := list_price - tech_bargains_discount
def budget_bytes_price : ℚ := list_price * (1 - budget_bytes_discount_rate)

-- Find the cheaper price
def cheaper_price : ℚ := min tech_bargains_price budget_bytes_price
def more_expensive_price : ℚ := max tech_bargains_price budget_bytes_price

-- Define the theorem
theorem price_difference_in_cents : 
  (more_expensive_price - cheaper_price) * 100 = 300 := by
  sorry

end NUMINAMATH_CALUDE_price_difference_in_cents_l2913_291356


namespace NUMINAMATH_CALUDE_frog_jump_distance_l2913_291364

/-- The jumping contest problem -/
theorem frog_jump_distance 
  (grasshopper_jump : ℕ) 
  (frog_grasshopper_diff : ℕ) 
  (h1 : grasshopper_jump = 36)
  (h2 : frog_grasshopper_diff = 17) :
  grasshopper_jump + frog_grasshopper_diff = 53 :=
by sorry

end NUMINAMATH_CALUDE_frog_jump_distance_l2913_291364


namespace NUMINAMATH_CALUDE_hyperbola_vertices_distance_l2913_291346

theorem hyperbola_vertices_distance (x y : ℝ) :
  (((x - 1)^2 / 16) - (y^2 / 25) = 1) →
  (∃ v₁ v₂ : ℝ, v₁ ≠ v₂ ∧ 
    (((v₁ - 1)^2 / 16) - (0^2 / 25) = 1) ∧
    (((v₂ - 1)^2 / 16) - (0^2 / 25) = 1) ∧
    |v₁ - v₂| = 8) :=
by
  sorry

end NUMINAMATH_CALUDE_hyperbola_vertices_distance_l2913_291346


namespace NUMINAMATH_CALUDE_min_value_constraint_l2913_291344

theorem min_value_constraint (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_constraint : x^3 * y^2 * z = 1) : 
  x + 2*y + 3*z ≥ 2 ∧ ∃ (x₀ y₀ z₀ : ℝ), 
    x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ 
    x₀^3 * y₀^2 * z₀ = 1 ∧ 
    x₀ + 2*y₀ + 3*z₀ = 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_constraint_l2913_291344


namespace NUMINAMATH_CALUDE_water_bottles_problem_l2913_291338

theorem water_bottles_problem (initial_bottles : ℕ) : 
  (initial_bottles : ℚ) * (2/3) * (1/2) = 8 → initial_bottles = 24 := by
  sorry

end NUMINAMATH_CALUDE_water_bottles_problem_l2913_291338


namespace NUMINAMATH_CALUDE_mascot_costs_l2913_291395

/-- The cost of an Auspicious Mascot Plush Toy -/
def plush_toy_cost : ℝ := 80

/-- The cost of an Auspicious Mascot Metal Ornament -/
def metal_ornament_cost : ℝ := 100

/-- The total cost of Plush Toys purchased -/
def plush_toy_total : ℝ := 6400

/-- The total cost of Metal Ornaments purchased -/
def metal_ornament_total : ℝ := 4000

theorem mascot_costs :
  (metal_ornament_cost = plush_toy_cost + 20) ∧
  (plush_toy_total / plush_toy_cost = 2 * (metal_ornament_total / metal_ornament_cost)) ∧
  (plush_toy_cost = 80) ∧
  (metal_ornament_cost = 100) := by
  sorry

end NUMINAMATH_CALUDE_mascot_costs_l2913_291395


namespace NUMINAMATH_CALUDE_students_present_l2913_291349

theorem students_present (total : ℕ) (absent_percent : ℚ) : 
  total = 50 → absent_percent = 14/100 → 
  (total : ℚ) * (1 - absent_percent) = 43 := by
  sorry

end NUMINAMATH_CALUDE_students_present_l2913_291349


namespace NUMINAMATH_CALUDE_seventh_term_is_29_3_l2913_291308

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- First term
  a : ℚ
  -- Common difference
  d : ℚ
  -- Sum of first five terms is 15
  sum_first_five : a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) = 15
  -- Sixth term is 8
  sixth_term : a + 5*d = 8

/-- The seventh term of the arithmetic sequence is 29/3 -/
theorem seventh_term_is_29_3 (seq : ArithmeticSequence) : seq.a + 6*seq.d = 29/3 := by
  sorry


end NUMINAMATH_CALUDE_seventh_term_is_29_3_l2913_291308


namespace NUMINAMATH_CALUDE_impossibleConstruction_l2913_291370

-- Define the basic structures
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a triangle in 3D space
structure Triangle3D where
  A : Point3D
  B : Point3D
  C : Point3D

-- Define a function to check if a triangle is acute-angled
def isAcuteAngled (t : Triangle3D) : Prop := sorry

-- Define a function to check if a triangle is equilateral
def isEquilateral (t : Triangle3D) : Prop := sorry

-- Define a function to check if three lines intersect at a point
def linesIntersectAtPoint (A A' B B' C C' O : Point3D) : Prop := sorry

-- Main theorem
theorem impossibleConstruction (ABC : Triangle3D) (O : Point3D) :
  isAcuteAngled ABC →
  ¬∃ (A'B'C' : Triangle3D),
    isEquilateral A'B'C' ∧
    linesIntersectAtPoint ABC.A A'B'C'.A ABC.B A'B'C'.B ABC.C A'B'C'.C O :=
by sorry

end NUMINAMATH_CALUDE_impossibleConstruction_l2913_291370


namespace NUMINAMATH_CALUDE_investment_duration_l2913_291328

/-- Represents a partner in the investment scenario -/
structure Partner where
  investment : ℚ
  profit : ℚ
  duration : ℚ

/-- The investment scenario with two partners -/
def InvestmentScenario (p q : Partner) : Prop :=
  p.investment / q.investment = 7 / 5 ∧
  p.profit / q.profit = 7 / 10 ∧
  p.duration = 8

theorem investment_duration (p q : Partner) 
  (h : InvestmentScenario p q) : q.duration = 16 := by
  sorry

end NUMINAMATH_CALUDE_investment_duration_l2913_291328


namespace NUMINAMATH_CALUDE_right_handed_players_count_l2913_291353

theorem right_handed_players_count (total_players throwers : ℕ) 
  (h1 : total_players = 70)
  (h2 : throwers = 37)
  (h3 : throwers ≤ total_players)
  (h4 : (total_players - throwers) % 3 = 0) :
  throwers + 2 * ((total_players - throwers) / 3) = 59 := by
sorry

end NUMINAMATH_CALUDE_right_handed_players_count_l2913_291353


namespace NUMINAMATH_CALUDE_necessary_sufficient_condition_l2913_291310

theorem necessary_sufficient_condition (a b : ℝ) :
  (a > 1 ∧ b > 1) ↔ (a + b > 2 ∧ a * b - a - b + 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_necessary_sufficient_condition_l2913_291310


namespace NUMINAMATH_CALUDE_new_xanadu_license_plates_l2913_291365

/-- The number of possible letters in each letter position of a license plate. -/
def num_letters : ℕ := 26

/-- The number of possible digits in each digit position of a license plate. -/
def num_digits : ℕ := 10

/-- The total number of valid license plates in New Xanadu. -/
def total_license_plates : ℕ := num_letters ^ 3 * num_digits ^ 3

/-- Theorem stating the total number of valid license plates in New Xanadu. -/
theorem new_xanadu_license_plates : total_license_plates = 17576000 := by
  sorry

end NUMINAMATH_CALUDE_new_xanadu_license_plates_l2913_291365


namespace NUMINAMATH_CALUDE_binomial_expansion_coeff_l2913_291306

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The coefficient of x^3 in the expansion of (x^2 - m/x)^6 -/
def coeff_x3 (m : ℝ) : ℝ := (-1)^3 * binomial 6 3 * m^3

theorem binomial_expansion_coeff (m : ℝ) :
  coeff_x3 m = -160 → m = 2 := by sorry

end NUMINAMATH_CALUDE_binomial_expansion_coeff_l2913_291306


namespace NUMINAMATH_CALUDE_triangle_shape_l2913_291381

/-- A triangle with side lengths a, b, and c is either isosceles or right-angled if a^4 - b^4 + (b^2c^2 - a^2c^2) = 0 -/
theorem triangle_shape (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (eq : a^4 - b^4 + (b^2 * c^2 - a^2 * c^2) = 0) : 
  (a = b) ∨ (a^2 + b^2 = c^2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_shape_l2913_291381


namespace NUMINAMATH_CALUDE_markus_bags_l2913_291322

theorem markus_bags (mara_bags : ℕ) (mara_marbles_per_bag : ℕ) (markus_marbles_per_bag : ℕ) (markus_extra_marbles : ℕ) :
  mara_bags = 12 →
  mara_marbles_per_bag = 2 →
  markus_marbles_per_bag = 13 →
  markus_extra_marbles = 2 →
  (mara_bags * mara_marbles_per_bag + markus_extra_marbles) / markus_marbles_per_bag = 2 :=
by sorry

end NUMINAMATH_CALUDE_markus_bags_l2913_291322


namespace NUMINAMATH_CALUDE_no_natural_square_difference_2018_l2913_291387

theorem no_natural_square_difference_2018 : ¬ ∃ (m n : ℕ), m^2 = n^2 + 2018 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_square_difference_2018_l2913_291387


namespace NUMINAMATH_CALUDE_function_monotonic_decreasing_l2913_291396

/-- The function f(x) = 3x^2 - 2ln(x) is monotonically decreasing on the interval (0, √3/3) -/
theorem function_monotonic_decreasing (x : ℝ) :
  let f : ℝ → ℝ := λ x => 3 * x^2 - 2 * Real.log x
  0 < x → x < Real.sqrt 3 / 3 → StrictMonoOn f (Set.Ioo 0 (Real.sqrt 3 / 3)) := by
  sorry

end NUMINAMATH_CALUDE_function_monotonic_decreasing_l2913_291396


namespace NUMINAMATH_CALUDE_phone_number_combinations_l2913_291342

def first_four_digits : ℕ := 12

def fifth_digit_options : ℕ := 2

def sixth_digit_options : ℕ := 10

theorem phone_number_combinations : 
  first_four_digits * fifth_digit_options * sixth_digit_options = 240 := by
  sorry

end NUMINAMATH_CALUDE_phone_number_combinations_l2913_291342


namespace NUMINAMATH_CALUDE_sqrt_nine_factorial_over_126_l2913_291382

theorem sqrt_nine_factorial_over_126 :
  let nine_factorial : ℕ := 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1
  let one_twenty_six : ℕ := 2 * 7 * 9
  (nine_factorial / one_twenty_six : ℚ).sqrt = 12 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_nine_factorial_over_126_l2913_291382


namespace NUMINAMATH_CALUDE_smallest_equal_packs_l2913_291324

theorem smallest_equal_packs (pencil_pack : Nat) (eraser_pack : Nat) : 
  pencil_pack = 5 → eraser_pack = 7 → 
  (∃ n : Nat, n > 0 ∧ ∃ m : Nat, n * eraser_pack = m * pencil_pack ∧ 
  ∀ k : Nat, k > 0 → k * eraser_pack = m * pencil_pack → n ≤ k) → n = 5 := by
sorry

end NUMINAMATH_CALUDE_smallest_equal_packs_l2913_291324


namespace NUMINAMATH_CALUDE_least_coins_seventeen_coins_least_possible_coins_l2913_291333

theorem least_coins (a : ℕ) : (a % 7 = 3 ∧ a % 4 = 1) → a ≥ 17 := by
  sorry

theorem seventeen_coins : 17 % 7 = 3 ∧ 17 % 4 = 1 := by
  sorry

theorem least_possible_coins : ∃ (a : ℕ), a % 7 = 3 ∧ a % 4 = 1 ∧ ∀ (b : ℕ), (b % 7 = 3 ∧ b % 4 = 1) → a ≤ b := by
  sorry

end NUMINAMATH_CALUDE_least_coins_seventeen_coins_least_possible_coins_l2913_291333


namespace NUMINAMATH_CALUDE_min_dot_product_in_triangle_l2913_291366

theorem min_dot_product_in_triangle (A B C : ℝ × ℝ) : 
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let angle_A := Real.arccos ((B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2)) / 
                  (Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) * Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2))
  BC = 2 → angle_A = 2 * Real.pi / 3 → 
  (∀ A' B' C' : ℝ × ℝ, 
    let BC' := Real.sqrt ((B'.1 - C'.1)^2 + (B'.2 - C'.2)^2)
    let angle_A' := Real.arccos ((B'.1 - A'.1) * (C'.1 - A'.1) + (B'.2 - A'.2) * (C'.2 - A'.2)) / 
                    (Real.sqrt ((B'.1 - A'.1)^2 + (B'.2 - A'.2)^2) * Real.sqrt ((C'.1 - A'.1)^2 + (C'.2 - A'.2)^2))
    BC' = 2 → angle_A' = 2 * Real.pi / 3 → 
    ((B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2)) ≤ 
    ((B'.1 - A'.1) * (C'.1 - A'.1) + (B'.2 - A'.2) * (C'.2 - A'.2))) →
  ((B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2)) = -2/3 := by
sorry

end NUMINAMATH_CALUDE_min_dot_product_in_triangle_l2913_291366


namespace NUMINAMATH_CALUDE_rectangle_area_l2913_291384

-- Define the points
variable (P Q R S T U : Point)

-- Define the rectangle PQRS
def is_rectangle (P Q R S : Point) : Prop := sorry

-- Define the trisection of angle S
def trisects_angle (S T U : Point) : Prop := sorry

-- Define that T is on PQ
def point_on_line (T P Q : Point) : Prop := sorry

-- Define that U is on PS
def point_on_line_2 (U P S : Point) : Prop := sorry

-- Define the length of QT
def length_QT (Q T : Point) : ℝ := sorry

-- Define the length of PU
def length_PU (P U : Point) : ℝ := sorry

-- Define the area of a rectangle
def area_rectangle (P Q R S : Point) : ℝ := sorry

-- Theorem statement
theorem rectangle_area (P Q R S T U : Point) 
  (h1 : is_rectangle P Q R S)
  (h2 : trisects_angle S T U)
  (h3 : point_on_line T P Q)
  (h4 : point_on_line_2 U P S)
  (h5 : length_QT Q T = 8)
  (h6 : length_PU P U = 4) :
  area_rectangle P Q R S = 64 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2913_291384


namespace NUMINAMATH_CALUDE_fraction_ordering_l2913_291380

theorem fraction_ordering : 6 / 22 < 5 / 17 ∧ 5 / 17 < 8 / 24 := by
  sorry

end NUMINAMATH_CALUDE_fraction_ordering_l2913_291380


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2913_291389

theorem geometric_sequence_common_ratio 
  (b₁ : ℕ+) 
  (q : ℕ+) 
  (seq : ℕ → ℕ+) 
  (h_geometric : ∀ n, seq n = b₁ * q ^ (n - 1)) 
  (h_sum : seq 3 + seq 5 + seq 7 = 819 * 6^2016) :
  q = 1 ∨ q = 2 ∨ q = 3 ∨ q = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2913_291389


namespace NUMINAMATH_CALUDE_seating_arrangement_count_l2913_291363

/-- Represents a circular table with chairs -/
structure CircularTable :=
  (num_chairs : ℕ)

/-- Represents a group of married couples -/
structure MarriedCouples :=
  (num_couples : ℕ)

/-- Represents the constraints for seating arrangements -/
structure SeatingConstraints :=
  (alternate_gender : Bool)
  (no_adjacent_spouses : Bool)
  (no_opposite_spouses : Bool)

/-- Calculates the number of valid seating arrangements -/
noncomputable def count_seating_arrangements (table : CircularTable) (couples : MarriedCouples) (constraints : SeatingConstraints) : ℕ :=
  sorry

/-- The main theorem to be proved -/
theorem seating_arrangement_count :
  ∀ (table : CircularTable) (couples : MarriedCouples) (constraints : SeatingConstraints),
    table.num_chairs = 10 →
    couples.num_couples = 5 →
    constraints.alternate_gender = true →
    constraints.no_adjacent_spouses = true →
    constraints.no_opposite_spouses = true →
    count_seating_arrangements table couples constraints = 480 :=
by
  sorry

end NUMINAMATH_CALUDE_seating_arrangement_count_l2913_291363


namespace NUMINAMATH_CALUDE_hyperbola_center_l2913_291309

/-- The center of a hyperbola is the midpoint of its foci -/
theorem hyperbola_center (f1 f2 : ℝ × ℝ) :
  let center := ((f1.1 + f2.1) / 2, (f1.2 + f2.2) / 2)
  f1 = (2, 0) ∧ f2 = (8, 6) →
  center = (5, 3) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_center_l2913_291309


namespace NUMINAMATH_CALUDE_snow_probability_l2913_291335

theorem snow_probability (p : ℝ) (h : p = 3/4) : 
  1 - (1 - p)^5 = 1023/1024 := by
sorry

end NUMINAMATH_CALUDE_snow_probability_l2913_291335


namespace NUMINAMATH_CALUDE_tower_of_threes_greater_than_tower_of_twos_l2913_291304

-- Define a function to represent the tower of exponents
def tower (base : ℕ) (height : ℕ) : ℕ :=
  match height with
  | 0 => 1
  | n + 1 => base ^ (tower base n)

-- State the theorem
theorem tower_of_threes_greater_than_tower_of_twos :
  tower 3 99 > tower 2 100 :=
sorry

end NUMINAMATH_CALUDE_tower_of_threes_greater_than_tower_of_twos_l2913_291304


namespace NUMINAMATH_CALUDE_jerry_logs_count_l2913_291325

/-- The number of logs Jerry gets from cutting trees -/
def total_logs : ℕ :=
  let pine_logs_per_tree : ℕ := 80
  let maple_logs_per_tree : ℕ := 60
  let walnut_logs_per_tree : ℕ := 100
  let pine_trees_cut : ℕ := 8
  let maple_trees_cut : ℕ := 3
  let walnut_trees_cut : ℕ := 4
  pine_logs_per_tree * pine_trees_cut +
  maple_logs_per_tree * maple_trees_cut +
  walnut_logs_per_tree * walnut_trees_cut

theorem jerry_logs_count : total_logs = 1220 := by
  sorry

end NUMINAMATH_CALUDE_jerry_logs_count_l2913_291325
