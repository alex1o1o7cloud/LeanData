import Mathlib

namespace lune_area_zero_l528_52817

/-- The area of a lune formed by a semicircle of diameter 2 sitting on top of a semicircle of diameter 4 is 0 -/
theorem lune_area_zero (π : ℝ) (h : π > 0) : 
  let small_semicircle_area := (1/2) * π * (2/2)^2
  let large_semicircle_sector_area := (1/8) * π * (4/2)^2
  let lune_area := small_semicircle_area - large_semicircle_sector_area
  lune_area = 0 := by sorry

end lune_area_zero_l528_52817


namespace trapezoid_perimeter_l528_52824

/-- Represents a trapezoid EFGH with specific properties -/
structure Trapezoid where
  EF : ℝ
  FG : ℝ
  GH : ℝ
  EH : ℝ
  ef_eq_gh : EF = GH
  fg_eq_10 : FG = 10
  eh_eq_20 : EH = 20
  right_triangle : EF^2 = 5^2 + 5^2

/-- The perimeter of the trapezoid EFGH is 30 + 10√2 -/
theorem trapezoid_perimeter (t : Trapezoid) : 
  t.EF + t.FG + t.GH + t.EH = 30 + 10 * Real.sqrt 2 := by
  sorry

#check trapezoid_perimeter

end trapezoid_perimeter_l528_52824


namespace john_average_change_l528_52840

def john_scores : List ℝ := [84, 88, 95, 92]

theorem john_average_change : 
  let initial_average := (john_scores.take 3).sum / 3
  let new_average := john_scores.sum / 4
  new_average - initial_average = 0.75 := by sorry

end john_average_change_l528_52840


namespace handshake_count_l528_52877

/-- The number of players in each team -/
def team_size : ℕ := 6

/-- The number of teams -/
def num_teams : ℕ := 2

/-- The number of referees -/
def num_referees : ℕ := 2

/-- The total number of handshakes -/
def total_handshakes : ℕ :=
  -- Handshakes between teams
  team_size * team_size +
  -- Handshakes within each team
  num_teams * (team_size.choose 2) +
  -- Handshakes with referees
  (num_teams * team_size) * num_referees

theorem handshake_count : total_handshakes = 90 := by
  sorry

end handshake_count_l528_52877


namespace square_inequality_l528_52889

theorem square_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^2 > a*b ∧ a*b > b^2 := by
  sorry

end square_inequality_l528_52889


namespace range_of_a_l528_52800

def A : Set ℝ := {x : ℝ | x^2 + 4*x = 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 + 2*(a+1)*x + a^2 - 1 = 0}

theorem range_of_a (a : ℝ) : 
  (B a ⊆ A) ↔ (a ≤ -1 ∨ a = 1) := by sorry

end range_of_a_l528_52800


namespace ratio_a_to_c_l528_52822

-- Define the variables
variable (a b c d x y z : ℝ)

-- Define the given ratios
def ratio_a_to_b : a / b = 2 * x / (3 * y) := by sorry
def ratio_b_to_c : b / c = z / (5 * z) := by sorry
def ratio_a_to_d : a / d = 4 * x / (7 * y) := by sorry
def ratio_d_to_c : d / c = 7 * y / (3 * z) := by sorry

-- State the theorem
theorem ratio_a_to_c (ha : a > 0) (hc : c > 0) : a / c = 2 * x / (15 * y) := by sorry

end ratio_a_to_c_l528_52822


namespace time_after_2023_hours_l528_52835

def clock_add (current_time : ℕ) (hours_passed : ℕ) : ℕ :=
  (current_time + hours_passed) % 12

theorem time_after_2023_hours :
  clock_add 7 2023 = 2 := by
  sorry

end time_after_2023_hours_l528_52835


namespace hospital_worker_count_l528_52884

theorem hospital_worker_count 
  (total_workers : ℕ) 
  (chosen_workers : ℕ) 
  (specific_pair_prob : ℚ) : 
  total_workers = 8 → 
  chosen_workers = 2 → 
  specific_pair_prob = 1 / 28 → 
  total_workers - 2 = 6 := by
sorry

end hospital_worker_count_l528_52884


namespace sin_cos_equation_solutions_l528_52821

theorem sin_cos_equation_solutions (x : Real) :
  (Real.sin x + Real.cos x + Real.sin x * Real.cos x = 1) ↔
  (∃ k : ℤ, x = k * (2 * Real.pi) ∨ x = Real.pi / 2 + k * (2 * Real.pi)) :=
by sorry

end sin_cos_equation_solutions_l528_52821


namespace quadratic_inequality_always_positive_l528_52857

theorem quadratic_inequality_always_positive (k : ℝ) : 
  (∀ x : ℝ, 2 * x^2 - (k - 1) * x - 2 * k + 8 > 0) ↔ -9 < k ∧ k < 7 := by
  sorry

end quadratic_inequality_always_positive_l528_52857


namespace milk_remaining_l528_52832

theorem milk_remaining (initial : ℚ) (given_away : ℚ) (remaining : ℚ) :
  initial = 5.5 ∧ given_away = 17/4 → remaining = initial - given_away → remaining = 5/4 := by
  sorry

end milk_remaining_l528_52832


namespace divide_n_plus_one_l528_52849

theorem divide_n_plus_one (n : ℕ+) : (n^2 + 1) ∣ (n + 1) ↔ n = 1 := by
  sorry

end divide_n_plus_one_l528_52849


namespace hoseok_persimmons_l528_52864

theorem hoseok_persimmons :
  ∀ (jungkook_persimmons hoseok_persimmons : ℕ),
    jungkook_persimmons = 25 →
    jungkook_persimmons = 3 * hoseok_persimmons + 4 →
    hoseok_persimmons = 7 := by
  sorry

end hoseok_persimmons_l528_52864


namespace ripe_fruits_weight_l528_52861

/-- Given the following conditions:
    - Total fruits: 14 apples, 10 pears, 5 lemons
    - Average weights of ripe fruits: apples 150g, pears 200g, lemons 100g
    - Average weights of unripe fruits: apples 120g, pears 180g, lemons 80g
    - Unripe fruits: 6 apples, 4 pears, 2 lemons
    Prove that the total weight of ripe fruits is 2700 grams -/
theorem ripe_fruits_weight (
  total_apples : ℕ) (total_pears : ℕ) (total_lemons : ℕ)
  (ripe_apple_weight : ℕ) (ripe_pear_weight : ℕ) (ripe_lemon_weight : ℕ)
  (unripe_apple_weight : ℕ) (unripe_pear_weight : ℕ) (unripe_lemon_weight : ℕ)
  (unripe_apples : ℕ) (unripe_pears : ℕ) (unripe_lemons : ℕ)
  (h1 : total_apples = 14)
  (h2 : total_pears = 10)
  (h3 : total_lemons = 5)
  (h4 : ripe_apple_weight = 150)
  (h5 : ripe_pear_weight = 200)
  (h6 : ripe_lemon_weight = 100)
  (h7 : unripe_apple_weight = 120)
  (h8 : unripe_pear_weight = 180)
  (h9 : unripe_lemon_weight = 80)
  (h10 : unripe_apples = 6)
  (h11 : unripe_pears = 4)
  (h12 : unripe_lemons = 2) :
  (total_apples - unripe_apples) * ripe_apple_weight +
  (total_pears - unripe_pears) * ripe_pear_weight +
  (total_lemons - unripe_lemons) * ripe_lemon_weight = 2700 := by
  sorry

end ripe_fruits_weight_l528_52861


namespace layla_nahima_score_difference_l528_52858

theorem layla_nahima_score_difference :
  ∀ (layla_score nahima_score : ℕ),
    layla_score = 70 →
    layla_score + nahima_score = 112 →
    layla_score - nahima_score = 28 :=
by
  sorry

end layla_nahima_score_difference_l528_52858


namespace equator_arc_length_equals_radius_l528_52867

/-- The radius of the Earth's equator in kilometers -/
def earth_radius : ℝ := 6370

/-- The length of an arc on a circle, given its radius and angle in radians -/
def arc_length (radius : ℝ) (angle : ℝ) : ℝ := radius * angle

/-- Theorem: The length of an arc on the Earth's equator corresponding to 1 radian 
    is equal to the radius of the Earth's equator -/
theorem equator_arc_length_equals_radius : 
  arc_length earth_radius 1 = earth_radius := by sorry

end equator_arc_length_equals_radius_l528_52867


namespace opposite_hands_theorem_l528_52872

/-- The time in minutes past 10:00 when the minute hand will be exactly opposite
    the place where the hour hand was four minutes ago, eight minutes from now. -/
def opposite_hands_time : ℝ :=
  let t : ℝ := 29.09090909090909  -- Approximate value of 29 1/11
  t

/-- Theorem stating that the calculated time satisfies the given conditions -/
theorem opposite_hands_theorem :
  let t := opposite_hands_time
  -- Time is between 10:00 and 11:00
  0 < t ∧ t < 60 ∧
  -- Minute hand position 8 minutes from now
  let minute_pos := 6 * (t + 8)
  -- Hour hand position 4 minutes ago
  let hour_pos := 30 + 0.5 * (t - 4)
  -- Hands are opposite (180 degrees apart)
  |minute_pos - hour_pos| = 180 := by
  sorry

#eval opposite_hands_time

end opposite_hands_theorem_l528_52872


namespace parabola_c_value_l528_52865

/-- A parabola in the xy-plane with equation x = ay² + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.a * y^2 + p.b * y + p.c

theorem parabola_c_value (p : Parabola) :
  p.x_coord 1 = 4 →  -- vertex at (4, 1)
  p.x_coord 3 = -2 →  -- passes through (-2, 3)
  p.c = 1/2 := by
  sorry

end parabola_c_value_l528_52865


namespace min_sin4_plus_2cos4_l528_52818

theorem min_sin4_plus_2cos4 (x : ℝ) : 
  (Real.sin x)^4 + 2 * (Real.cos x)^4 ≥ (1/2 : ℝ) :=
by
  sorry

#check min_sin4_plus_2cos4

end min_sin4_plus_2cos4_l528_52818


namespace sphere_radius_touching_cones_l528_52804

/-- The radius of a sphere touching three cones and a table -/
theorem sphere_radius_touching_cones (r₁ r₂ r₃ : ℝ) (α β γ : ℝ) : 
  r₁ = 1 → 
  r₂ = 12 → 
  r₃ = 12 → 
  α = -4 * Real.arctan (1/3) → 
  β = 4 * Real.arctan (2/3) → 
  γ = 4 * Real.arctan (2/3) → 
  ∃ R : ℝ, R = 40/21 ∧ 
    (∀ x y z : ℝ, 
      x^2 + y^2 + z^2 = R^2 → 
      (∃ t : ℝ, t ≥ 0 ∧ 
        ((x - r₁)^2 + y^2 = (t * Real.tan (α/2))^2 ∧ z = t) ∨
        ((x - (r₁ + r₂))^2 + y^2 = (t * Real.tan (β/2))^2 ∧ z = t) ∨
        (x^2 + (y - (r₂ + r₃))^2 = (t * Real.tan (γ/2))^2 ∧ z = t) ∨
        z = 0)) :=
by
  sorry

end sphere_radius_touching_cones_l528_52804


namespace polynomial_simplification_l528_52808

theorem polynomial_simplification (x : ℝ) : 
  5 - 3*x - 7*x^2 + 3 + 12*x - 9*x^2 - 8 + 15*x + 21*x^2 = 5*x^2 + 24*x := by
  sorry

end polynomial_simplification_l528_52808


namespace cyclic_inequality_l528_52879

theorem cyclic_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 1) :
  (1 / Real.sqrt (x + 2 * y + 6) + 1 / Real.sqrt (y + 2 * z + 6) + 1 / Real.sqrt (z + 2 * x + 6)) ≤
  (x / Real.sqrt (x^2 + 4 * Real.sqrt y + 4 * Real.sqrt z) +
   y / Real.sqrt (y^2 + 4 * Real.sqrt z + 4 * Real.sqrt x) +
   z / Real.sqrt (z^2 + 4 * Real.sqrt x + 4 * Real.sqrt y)) :=
by sorry

end cyclic_inequality_l528_52879


namespace even_square_sum_perfect_square_l528_52847

theorem even_square_sum_perfect_square (x y z : ℤ) 
  (h_even : Even x)
  (h_odd : Odd y)
  (h_sum : x^2 + y^2 = z^2) :
  4 ∣ x :=
sorry

end even_square_sum_perfect_square_l528_52847


namespace snakes_not_hiding_l528_52883

/-- Given a total of 95 snakes and 64 hiding snakes, prove that the number of snakes not hiding is 31. -/
theorem snakes_not_hiding (total_snakes : ℕ) (hiding_snakes : ℕ) 
  (h1 : total_snakes = 95) (h2 : hiding_snakes = 64) : 
  total_snakes - hiding_snakes = 31 := by
  sorry

end snakes_not_hiding_l528_52883


namespace basketball_practice_average_l528_52893

/-- The average practice time per day for a basketball player over a week -/
theorem basketball_practice_average (weekday_hours : ℝ) (weekend_hours : ℝ) (weekdays : ℕ) (weekend_days : ℕ) :
  weekday_hours = 2 →
  weekend_hours = 11 →
  weekdays = 5 →
  weekend_days = 2 →
  (weekday_hours * weekdays + weekend_hours) / (weekdays + weekend_days) = 3 := by
  sorry

end basketball_practice_average_l528_52893


namespace cereal_box_initial_price_l528_52871

/-- The initial price of a cereal box given a price reduction and total purchase amount -/
theorem cereal_box_initial_price 
  (price_reduction : ℝ) 
  (num_boxes : ℕ) 
  (total_paid : ℝ) 
  (h1 : price_reduction = 24)
  (h2 : num_boxes = 20)
  (h3 : total_paid = 1600) : 
  ∃ (initial_price : ℝ), 
    num_boxes * (initial_price - price_reduction) = total_paid ∧ 
    initial_price = 104 := by
  sorry

end cereal_box_initial_price_l528_52871


namespace equation_solution_l528_52823

theorem equation_solution : ∀ x : ℝ, x + 36 / (x - 3) = -9 ↔ x = -3 := by sorry

end equation_solution_l528_52823


namespace ribbon_fraction_per_box_l528_52816

theorem ribbon_fraction_per_box 
  (total_fraction : ℚ) 
  (num_boxes : ℕ) 
  (h1 : total_fraction = 5/12) 
  (h2 : num_boxes = 5) : 
  total_fraction / num_boxes = 1/12 := by
sorry

end ribbon_fraction_per_box_l528_52816


namespace find_other_number_l528_52891

theorem find_other_number (a b : ℤ) : 
  (a = 17 ∨ b = 17) → 
  (3 * a + 4 * b = 131) → 
  (a = 21 ∨ b = 21) :=
by sorry

end find_other_number_l528_52891


namespace compound_molecular_weight_l528_52810

/-- Calculates the molecular weight of a compound given the number of atoms and their atomic weights -/
def molecular_weight (ca_count : ℕ) (o_count : ℕ) (h_count : ℕ) 
                     (ca_weight : ℝ) (o_weight : ℝ) (h_weight : ℝ) : ℝ :=
  ca_count * ca_weight + o_count * o_weight + h_count * h_weight

/-- Theorem stating that the molecular weight of the given compound is 74.094 g/mol -/
theorem compound_molecular_weight : 
  molecular_weight 1 2 2 40.08 15.999 1.008 = 74.094 := by
  sorry

end compound_molecular_weight_l528_52810


namespace latoya_phone_card_initial_amount_l528_52815

/-- The initial amount paid for a prepaid phone card -/
def initial_amount (call_cost_per_minute : ℚ) (call_duration : ℕ) (remaining_credit : ℚ) : ℚ :=
  call_cost_per_minute * call_duration + remaining_credit

/-- Theorem: The initial amount paid for Latoya's phone card is $30.00 -/
theorem latoya_phone_card_initial_amount :
  initial_amount (16 / 100) 22 26.48 = 30 :=
by sorry

end latoya_phone_card_initial_amount_l528_52815


namespace bamboo_tube_rice_problem_l528_52895

theorem bamboo_tube_rice_problem (a : ℕ → ℚ) :
  (∀ n : ℕ, n < 8 → a (n + 1) - a n = a 1 - a 0) →  -- arithmetic sequence
  a 0 + a 1 + a 2 = 39/10 →                         -- bottom three joints
  a 5 + a 6 + a 7 + a 8 = 3 →                       -- top four joints
  a 4 = 1 :=                                        -- middle joint
by sorry

end bamboo_tube_rice_problem_l528_52895


namespace line_circle_intersection_radius_l528_52860

/-- Given a line and a circle that intersect at two points forming a 120° angle at the origin, 
    prove that the radius of the circle is 2. -/
theorem line_circle_intersection_radius (r : ℝ) (h_r_pos : r > 0) : 
  (∃ (A B : ℝ × ℝ), 
    (3 * A.1 - 4 * A.2 + 5 = 0) ∧ 
    (3 * B.1 - 4 * B.2 + 5 = 0) ∧
    (A.1^2 + A.2^2 = r^2) ∧ 
    (B.1^2 + B.2^2 = r^2) ∧
    (Real.cos (Real.pi / 3) * r^2 = A.1 * B.1 + A.2 * B.2)) →
  r = 2 :=
by sorry

end line_circle_intersection_radius_l528_52860


namespace noah_garden_larger_by_75_l528_52803

/-- Represents the dimensions of a rectangular garden -/
structure GardenDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular garden -/
def gardenArea (d : GardenDimensions) : ℝ := d.length * d.width

theorem noah_garden_larger_by_75 (liam_garden noah_garden : GardenDimensions) 
  (h1 : liam_garden.length = 30 ∧ liam_garden.width = 50)
  (h2 : noah_garden.length = 35 ∧ noah_garden.width = 45) : 
  gardenArea noah_garden - gardenArea liam_garden = 75 := by
  sorry

#check noah_garden_larger_by_75

end noah_garden_larger_by_75_l528_52803


namespace fraction_simplification_l528_52827

/-- Given a, b, c, x, y, z are real numbers, prove that the given complex fraction 
    is equal to the simplified form. -/
theorem fraction_simplification (a b c x y z : ℝ) :
  (c * z * (a^3 * x^3 + 3 * a^3 * y^3 + c^3 * z^3) + 
   b * z * (a^3 * x^3 + 3 * c^3 * x^3 + c^3 * z^3)) / (c * z + b * z) = 
  a^3 * x^3 + c^3 * z^3 + (3 * c * z * a^3 * y^3 + 3 * b * z * c^3 * x^3) / (c * z + b * z) :=
by
  sorry

end fraction_simplification_l528_52827


namespace expand_and_simplify_l528_52844

theorem expand_and_simplify (x : ℝ) : (3 * x - 4) * (2 * x + 6) = 6 * x^2 + 10 * x - 24 := by
  sorry

end expand_and_simplify_l528_52844


namespace problem_solution_l528_52828

theorem problem_solution : ∃ x : ℚ, (70 / 100) * x - (1 / 3) * x = 110 ∧ x = 300 := by
  sorry

end problem_solution_l528_52828


namespace solution_problem_l528_52846

theorem solution_problem (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 143) : x = 17 := by
  sorry

end solution_problem_l528_52846


namespace ellipse_k_range_l528_52897

/-- The ellipse equation -/
def ellipse (k x y : ℝ) : Prop :=
  k^2 * x^2 + y^2 - 4*k*x + 2*k*y + k^2 - 1 = 0

/-- The origin is inside the ellipse -/
def origin_inside (k : ℝ) : Prop :=
  ∃ ε > 0, ∀ x y : ℝ, x^2 + y^2 < ε^2 → ellipse k x y

/-- The theorem stating the range of k -/
theorem ellipse_k_range :
  ∀ k : ℝ, origin_inside k → 0 < |k| ∧ |k| < 1 :=
sorry

end ellipse_k_range_l528_52897


namespace solve_for_x_l528_52870

theorem solve_for_x (x y : ℝ) (h1 : x + 2*y = 20) (h2 : y = 5) : x = 10 := by
  sorry

end solve_for_x_l528_52870


namespace triangle_area_and_angle_B_l528_52807

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
def Triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0

theorem triangle_area_and_angle_B 
  (a b c : ℝ) 
  (h_triangle : Triangle a b c)
  (h_eq : b^2 = c^2 + a^2 - Real.sqrt 2 * a * c)
  (h_a : a = Real.sqrt 2)
  (h_cos_A : Real.cos A = 4/5)
  : Real.cos B = Real.sqrt 2 / 2 ∧ 
    ∃ (S : ℝ), S = 7/6 ∧ S = 1/2 * a * b * Real.sin C :=
by sorry

end triangle_area_and_angle_B_l528_52807


namespace binomial_60_3_l528_52888

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end binomial_60_3_l528_52888


namespace distance_calculation_l528_52848

/-- Proves that the distance run by A and B is 2250 meters given their running times and the difference in distance covered. -/
theorem distance_calculation (D : ℝ) 
  (h1 : D / 90 * 180 = D + 2250) : D = 2250 := by
  sorry

#check distance_calculation

end distance_calculation_l528_52848


namespace function_derivative_value_l528_52836

theorem function_derivative_value (f : ℝ → ℝ) (hf : ∀ x, f x = x^2 * (deriv f (π/3)) + Real.sin x) :
  deriv f (π/3) = 3 / (6 - 4*π) := by sorry

end function_derivative_value_l528_52836


namespace tangent_lines_and_intersection_points_l528_52878

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 5

-- Define the point M
def M : ℝ × ℝ := (3, 2)

-- Define the line l
def l (m x y : ℝ) : Prop := m * x - y - m + 1 = 0

theorem tangent_lines_and_intersection_points :
  -- Part 1: Tangent lines
  (∀ x y : ℝ, (x + 2*y - 7 = 0 → C x y) ∧ (2*x - y - 4 = 0 → C x y)) ∧
  (x + 2*y - 7 = 0 → x = M.1 ∧ y = M.2) ∧
  (2*x - y - 4 = 0 → x = M.1 ∧ y = M.2) ∧
  -- Part 2: Intersection points
  (∀ m : ℝ, (∃ A B : ℝ × ℝ, 
    l m A.1 A.2 ∧ l m B.1 B.2 ∧ 
    C A.1 A.2 ∧ C B.1 B.2 ∧ 
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 17) → 
  m = Real.sqrt 3 ∨ m = -Real.sqrt 3) :=
sorry


end tangent_lines_and_intersection_points_l528_52878


namespace max_value_x_plus_reciprocal_l528_52855

def sum_and_reciprocal_sum_equal (s : Finset ℝ) (n : ℕ) (v : ℝ) : Prop :=
  s.card = n ∧ (∀ x ∈ s, x > 0) ∧ (s.sum id = v) ∧ (s.sum (λ x => 1 / x) = v)

theorem max_value_x_plus_reciprocal 
  (s : Finset ℝ) (h : sum_and_reciprocal_sum_equal s 1001 1002) :
  ∀ x ∈ s, x + 1/x ≤ 4007/1002 :=
sorry

end max_value_x_plus_reciprocal_l528_52855


namespace coincident_foci_and_vertices_m_range_l528_52862

-- Define the ellipse equation
def is_ellipse (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (9 - m) + y^2 / (2 * m) = 1 ∧ 9 - m > 2 * m ∧ 2 * m > 0

-- Define the hyperbola equation and its eccentricity condition
def hyperbola_eccentricity_condition (m : ℝ) : Prop :=
  ∃ (e : ℝ), e > Real.sqrt 6 / 2 ∧ e < Real.sqrt 2 ∧
  e^2 = (5 + m) / 5

-- Theorem for part (I)
theorem coincident_foci_and_vertices (m : ℝ) 
  (h1 : is_ellipse m) (h2 : hyperbola_eccentricity_condition m) :
  (∃ (x : ℝ), x^2 / (9 - m) + 0^2 / (2 * m) = 1 ∧ 
              x^2 / 5 - 0^2 / m = 1) → m = 4 / 3 :=
sorry

-- Theorem for part (II)
theorem m_range (m : ℝ) 
  (h1 : is_ellipse m) (h2 : hyperbola_eccentricity_condition m) :
  5 / 2 < m ∧ m < 3 :=
sorry

end coincident_foci_and_vertices_m_range_l528_52862


namespace det_A_eq_zero_iff_x_eq_52_19_l528_52845

def A (x : ℚ) : Matrix (Fin 3) (Fin 3) ℚ :=
  !![3, 1, -1;
     4, x, 2;
     1, 3, 6]

theorem det_A_eq_zero_iff_x_eq_52_19 :
  ∀ x : ℚ, Matrix.det (A x) = 0 ↔ x = 52 / 19 := by sorry

end det_A_eq_zero_iff_x_eq_52_19_l528_52845


namespace sugar_consumption_reduction_l528_52811

theorem sugar_consumption_reduction (initial_price new_price : ℝ) 
  (h1 : initial_price = 3)
  (h2 : new_price = 5) :
  let reduction_percentage := (1 - initial_price / new_price) * 100
  reduction_percentage = 40 := by
sorry

end sugar_consumption_reduction_l528_52811


namespace intersection_single_element_l528_52805

/-- The value of k when the intersection of sets A and B has only one element -/
theorem intersection_single_element (x y : ℝ) :
  let A := {p : ℝ × ℝ | p.1^2 - 3*p.1*p.2 + 4*p.2^2 = 7/2}
  let B := {p : ℝ × ℝ | ∃ (k : ℝ), k > 0 ∧ k*p.1 + p.2 = 2}
  (∃! p, p ∈ A ∩ B) → (∃ (k : ℝ), k = 1/4 ∧ k > 0 ∧ ∀ p, p ∈ A ∩ B → k*p.1 + p.2 = 2) :=
by sorry

end intersection_single_element_l528_52805


namespace min_editors_conference_l528_52820

theorem min_editors_conference (total : ℕ) (writers : ℕ) (max_both : ℕ) :
  total = 100 →
  writers = 35 →
  max_both = 26 →
  ∃ (editors : ℕ) (both : ℕ),
    both ≤ max_both ∧
    editors ≥ 39 ∧
    total = writers + editors - both + 2 * both :=
by
  sorry

end min_editors_conference_l528_52820


namespace smallest_a_for_integer_roots_and_product_condition_l528_52899

theorem smallest_a_for_integer_roots_and_product_condition : 
  (∃ (a : ℕ+), 
    (∀ (x : ℤ), x^2 + a*x = 30 → ∃ (y z : ℤ), y * z > 30 ∧ x = y ∧ x = z) ∧ 
    (∀ (b : ℕ+), b < a → 
      ¬(∀ (x : ℤ), x^2 + b*x = 30 → ∃ (y z : ℤ), y * z > 30 ∧ x = y ∧ x = z))) ∧
  (∀ (a : ℕ+), 
    (∀ (x : ℤ), x^2 + a*x = 30 → ∃ (y z : ℤ), y * z > 30 ∧ x = y ∧ x = z) ∧ 
    (∀ (b : ℕ+), b < a → 
      ¬(∀ (x : ℤ), x^2 + b*x = 30 → ∃ (y z : ℤ), y * z > 30 ∧ x = y ∧ x = z)) → 
    a = 11) :=
sorry

end smallest_a_for_integer_roots_and_product_condition_l528_52899


namespace remainder_h_x10_div_h_x_l528_52809

-- Define the polynomial h(x)
def h (x : ℝ) : ℝ := x^5 + x^4 + x^3 + x^2 + x + 1

-- Define the theorem
theorem remainder_h_x10_div_h_x :
  ∃ (q : ℝ → ℝ), h (x^10) = h x * q x + 6 :=
sorry

end remainder_h_x10_div_h_x_l528_52809


namespace smallest_y_for_perfect_cube_l528_52887

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

theorem smallest_y_for_perfect_cube (x : ℕ) (h : x = 11 * 36 * 54) :
  ∃! y : ℕ, y > 0 ∧ is_perfect_cube (x * y) ∧ ∀ z : ℕ, z > 0 → is_perfect_cube (x * z) → y ≤ z :=
by sorry

end smallest_y_for_perfect_cube_l528_52887


namespace line_intercepts_sum_l528_52898

/-- Given a line with equation x - 2y + 1 = 0, prove that the sum of its x-intercept and y-intercept is -1/2 -/
theorem line_intercepts_sum (x y : ℝ) : 
  x - 2*y + 1 = 0 → 
  ∃ (x_int y_int : ℝ), x_int - 2*0 + 1 = 0 ∧ 0 - 2*y_int + 1 = 0 ∧ x_int + y_int = -1/2 :=
by sorry

end line_intercepts_sum_l528_52898


namespace positive_integer_solutions_for_equation_l528_52826

theorem positive_integer_solutions_for_equation :
  ∀ m n : ℕ+,
  m^2 = n^2 + m + n + 2018 ↔ (m = 1010 ∧ n = 1008) ∨ (m = 506 ∧ n = 503) :=
by sorry

end positive_integer_solutions_for_equation_l528_52826


namespace base_representation_of_500_l528_52876

theorem base_representation_of_500 :
  ∃! b : ℕ, b > 1 ∧ b^5 ≤ 500 ∧ 500 < b^6 :=
sorry

end base_representation_of_500_l528_52876


namespace x_plus_y_value_l528_52892

theorem x_plus_y_value (x y : ℝ) (h1 : x - y = 4) (h2 : |x| + |y| = 7) :
  x + y = 7 ∨ x + y = -7 := by
  sorry

end x_plus_y_value_l528_52892


namespace integer_roots_count_l528_52834

theorem integer_roots_count : 
  let lower_bound := -5 - Real.sqrt 42
  let upper_bound := -5 + Real.sqrt 42
  let is_valid_root (x : ℤ) := 
    (Real.cos (2 * π * ↑x) + Real.cos (π * ↑x) = Real.sin (3 * π * ↑x) + Real.sin (π * ↑x)) ∧
    (lower_bound < x) ∧ (x < upper_bound)
  ∃! (roots : Finset ℤ), (Finset.card roots = 7) ∧ (∀ x, x ∈ roots ↔ is_valid_root x) :=
by sorry

end integer_roots_count_l528_52834


namespace inequality_solution_l528_52812

/-- Theorem: Solutions to the inequality ax^2 - 2 ≥ 2x - ax for a < 0 -/
theorem inequality_solution (a : ℝ) (h : a < 0) :
  (∀ x : ℝ, ¬(a * x^2 - 2 ≥ 2 * x - a * x) ∨ (a = -2 ∧ x = -1)) :=
by sorry

end inequality_solution_l528_52812


namespace log_relation_difference_l528_52854

theorem log_relation_difference (a b c d : ℤ) 
  (h1 : (Real.log b) / (Real.log a) = 3/2)
  (h2 : (Real.log d) / (Real.log c) = 5/4)
  (h3 : a - c = 9) : 
  b - d = 93 := by
sorry

end log_relation_difference_l528_52854


namespace power_of_64_l528_52894

theorem power_of_64 : (64 : ℝ) ^ (5/6 : ℝ) = 32 :=
by
  have h : (64 : ℝ) = 2^6 := by sorry
  sorry

end power_of_64_l528_52894


namespace line_through_point_parallel_to_polar_axis_l528_52863

/-- A point in polar coordinates -/
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

/-- Represents a line in polar coordinates -/
structure PolarLine where
  -- The equation of the line in the form ρ sin θ = k
  k : ℝ

/-- Checks if a point lies on a given polar line -/
def isOnLine (p : PolarPoint) (l : PolarLine) : Prop :=
  p.ρ * Real.sin p.θ = l.k

theorem line_through_point_parallel_to_polar_axis 
  (p : PolarPoint) (l : PolarLine) 
  (h1 : p.ρ = 1) 
  (h2 : p.θ = Real.pi / 2) 
  (h3 : l.k = 1) : 
  ∀ q : PolarPoint, isOnLine q l ↔ q.ρ * Real.sin q.θ = 1 := by
  sorry

#check line_through_point_parallel_to_polar_axis

end line_through_point_parallel_to_polar_axis_l528_52863


namespace high_school_total_students_l528_52885

/-- Represents a high school with three grades and a stratified sampling method. -/
structure HighSchool where
  total_sample : ℕ
  grade1_sample : ℕ
  grade2_sample : ℕ
  grade3_total : ℕ
  sample_sum : total_sample = grade1_sample + grade2_sample + (total_sample - grade1_sample - grade2_sample)
  grade3_prob : (total_sample - grade1_sample - grade2_sample) / grade3_total = 1 / 20

/-- The total number of students in the high school is 3600. -/
theorem high_school_total_students (h : HighSchool) 
  (h_total_sample : h.total_sample = 180)
  (h_grade1_sample : h.grade1_sample = 70)
  (h_grade2_sample : h.grade2_sample = 60)
  (h_grade3_total : h.grade3_total = 1000) : 
  h.total_sample * 20 = 3600 := by
  sorry

#check high_school_total_students

end high_school_total_students_l528_52885


namespace angle_E_measure_l528_52859

/-- A quadrilateral with specific angle relationships -/
structure SpecialQuadrilateral where
  E : ℝ  -- Angle E in degrees
  F : ℝ  -- Angle F in degrees
  G : ℝ  -- Angle G in degrees
  H : ℝ  -- Angle H in degrees
  angle_sum : E + F + G + H = 360  -- Sum of angles in a quadrilateral
  E_eq_5H : E = 5 * H  -- Relationship between E and H
  E_eq_4G : E = 4 * G  -- Relationship between E and G
  E_eq_5div3F : E = 5 / 3 * F  -- Relationship between E and F

/-- The measure of angle E in the special quadrilateral -/
theorem angle_E_measure (q : SpecialQuadrilateral) : q.E = 1440 / 11 := by
  sorry


end angle_E_measure_l528_52859


namespace time_per_video_l528_52838

-- Define the parameters
def setup_time : ℝ := 1
def cleanup_time : ℝ := 1
def painting_time_per_video : ℝ := 1
def editing_time_per_video : ℝ := 1.5
def num_videos : ℕ := 4

-- Define the theorem
theorem time_per_video : 
  (setup_time + cleanup_time + num_videos * painting_time_per_video + num_videos * editing_time_per_video) / num_videos = 3 := by
  sorry

end time_per_video_l528_52838


namespace ring_arrangement_count_l528_52843

/-- The number of ways to choose k items from n items -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The factorial of a natural number -/
def factorial (n : ℕ) : ℕ := sorry

/-- The number of ring arrangements on four fingers -/
def ring_arrangements : ℕ :=
  let total_rings := 10
  let chosen_rings := 7
  let fingers := 4
  binomial total_rings chosen_rings * 
  factorial chosen_rings * 
  chosen_rings * 
  binomial (chosen_rings + fingers - 2) (fingers - 1)

theorem ring_arrangement_count : ring_arrangements = 264537600 := by
  sorry

end ring_arrangement_count_l528_52843


namespace polynomial_coefficient_equality_l528_52869

theorem polynomial_coefficient_equality 
  (a b c d : ℚ) :
  (∀ x : ℚ, (6 * x^3 - 4 * x + 2) * (a * x^3 + b * x^2 + c * x + d) = 
    18 * x^6 - 2 * x^5 + 16 * x^4 - 28/3 * x^3 + 8/3 * x^2 - 4 * x + 2) →
  b = -1/3 ∧ c = 14/9 := by
sorry

end polynomial_coefficient_equality_l528_52869


namespace parallel_lines_k_value_l528_52852

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_slopes_equal {m₁ m₂ b₁ b₂ : ℝ} : 
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The value of k for which the lines y = 5x + 3 and y = (3k)x + 1 are parallel -/
theorem parallel_lines_k_value :
  (∀ x y : ℝ, y = 5 * x + 3 ↔ y = (3 * k) * x + 1) → k = 5 / 3 :=
by sorry

end parallel_lines_k_value_l528_52852


namespace tangent_from_cosine_central_angle_from_perimeter_and_area_l528_52881

-- Part 1
theorem tangent_from_cosine (m : ℝ) (α : ℝ) :
  (m : ℝ) = -Real.sqrt 2 / 4 →
  Real.cos α = -1/3 →
  Real.tan α = -2 * Real.sqrt 2 :=
by sorry

-- Part 2
theorem central_angle_from_perimeter_and_area (r l : ℝ) :
  2 * r + l = 8 →
  1/2 * l * r = 3 →
  (l / r = 2/3 ∨ l / r = 6) :=
by sorry

end tangent_from_cosine_central_angle_from_perimeter_and_area_l528_52881


namespace orange_juice_problem_l528_52842

/-- Calculates the number of servings of orange juice prepared from concentrate -/
def orange_juice_servings (concentrate_cans : ℕ) (concentrate_oz_per_can : ℕ) 
  (water_cans_per_concentrate : ℕ) (oz_per_serving : ℕ) : ℕ :=
  let total_oz := concentrate_cans * concentrate_oz_per_can * (water_cans_per_concentrate + 1)
  total_oz / oz_per_serving

/-- Theorem stating that 60 cans of 5-oz concentrate mixed with 3 cans of water per
    1 can of concentrate yields 200 servings of 6-oz orange juice -/
theorem orange_juice_problem : 
  orange_juice_servings 60 5 3 6 = 200 := by
  sorry

end orange_juice_problem_l528_52842


namespace probability_yellow_second_is_67_135_l528_52875

/-- Represents the contents of a bag of marbles -/
structure Bag where
  white : ℕ := 0
  black : ℕ := 0
  yellow : ℕ := 0
  blue : ℕ := 0

/-- Calculates the total number of marbles in a bag -/
def Bag.total (b : Bag) : ℕ := b.white + b.black + b.yellow + b.blue

/-- Bag X contents -/
def bagX : Bag := { white := 4, black := 5 }

/-- Bag Y contents -/
def bagY : Bag := { yellow := 7, blue := 3 }

/-- Bag Z contents -/
def bagZ : Bag := { yellow := 3, blue := 6 }

/-- Probability of drawing a yellow marble as the second marble -/
def probabilityYellowSecond : ℚ :=
  (bagX.white * bagY.yellow) / (bagX.total * bagY.total) +
  (bagX.black * bagZ.yellow) / (bagX.total * bagZ.total)

theorem probability_yellow_second_is_67_135 : probabilityYellowSecond = 67 / 135 := by
  sorry

end probability_yellow_second_is_67_135_l528_52875


namespace item_a_price_correct_l528_52868

/-- The price of item (a) in won -/
def item_a_price : ℕ := 7 * 1000 + 4 * 100 + 5 * 10

/-- The number of 1000 won coins used -/
def coins_1000 : ℕ := 7

/-- The number of 100 won coins used -/
def coins_100 : ℕ := 4

/-- The number of 10 won coins used -/
def coins_10 : ℕ := 5

theorem item_a_price_correct : item_a_price = 7450 := by
  sorry

end item_a_price_correct_l528_52868


namespace rationalize_denominator_l528_52806

theorem rationalize_denominator :
  (1 : ℝ) / (Real.rpow 3 (1/3) + Real.rpow 27 (1/3)) = Real.rpow 9 (1/3) / 12 := by
  sorry

end rationalize_denominator_l528_52806


namespace rich_walk_ratio_l528_52801

theorem rich_walk_ratio : 
  ∀ (x : ℝ), 
    (20 : ℝ) + 200 + 220 * x + ((20 + 200 + 220 * x) / 2) = 990 → 
    (220 * x) / (20 + 200) = 2 := by
  sorry

end rich_walk_ratio_l528_52801


namespace sin_240_degrees_l528_52856

theorem sin_240_degrees : Real.sin (240 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_240_degrees_l528_52856


namespace set_operations_l528_52802

def A : Set ℝ := {x | -5 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x < -2 ∨ x > 4}

theorem set_operations (x : ℝ) :
  (x ∈ A ∩ B ↔ -5 ≤ x ∧ x < -2) ∧
  (x ∈ (Set.univ \ A) ∩ B ↔ x < -5 ∨ x > 4) ∧
  (x ∈ (Set.univ \ A) ∩ (Set.univ \ B) ↔ 3 < x ∧ x ≤ 4) := by
  sorry

end set_operations_l528_52802


namespace smallest_palindromic_prime_l528_52819

/-- A function that checks if a number is a three-digit palindrome with hundreds digit 2 -/
def isValidPalindrome (n : ℕ) : Prop :=
  n ≥ 200 ∧ n ≤ 299 ∧ (n / 100 = 2) ∧ (n % 10 = n / 100)

/-- The theorem stating that 232 is the smallest three-digit palindromic prime with hundreds digit 2 -/
theorem smallest_palindromic_prime :
  isValidPalindrome 232 ∧ Nat.Prime 232 ∧
  ∀ n < 232, isValidPalindrome n → ¬Nat.Prime n :=
sorry

end smallest_palindromic_prime_l528_52819


namespace negation_of_implication_l528_52813

theorem negation_of_implication (a b : ℝ) :
  ¬(∀ x : ℝ, x < a → x < b) ↔ (∃ x : ℝ, x ≥ a ∧ x ≥ b) :=
by sorry

end negation_of_implication_l528_52813


namespace rice_restocking_solution_l528_52886

def rice_restocking_problem (initial_stock sold final_stock : ℕ) : ℕ :=
  final_stock - (initial_stock - sold)

theorem rice_restocking_solution :
  rice_restocking_problem 55 23 164 = 132 := by
  sorry

end rice_restocking_solution_l528_52886


namespace complex_distance_theorem_l528_52829

theorem complex_distance_theorem (z₁ z₂ : ℂ) 
  (h₁ : Complex.abs z₁ = 3)
  (h₂ : Complex.abs z₂ = 5)
  (h₃ : Complex.abs (z₁ + z₂) = 6) :
  Complex.abs (z₁ - z₂) = 4 * Real.sqrt 2 :=
by sorry

end complex_distance_theorem_l528_52829


namespace leo_has_largest_answer_l528_52830

def starting_number : ℕ := 12

def rodrigo_process (n : ℕ) : ℕ := ((n - 3)^2 + 4)

def samantha_process (n : ℕ) : ℕ := (n^2 - 5 + 4)

def leo_process (n : ℕ) : ℕ := ((n - 3 + 4)^2)

theorem leo_has_largest_answer :
  leo_process starting_number > rodrigo_process starting_number ∧
  leo_process starting_number > samantha_process starting_number :=
sorry

end leo_has_largest_answer_l528_52830


namespace total_students_l528_52833

theorem total_students (n : ℕ) (male_students : ℕ) (selected_students : ℕ) (male_participants : ℕ) (female_participants : ℕ) : 
  male_students = 550 →
  selected_students = n / 10 →
  male_participants = female_participants + 10 →
  male_participants = male_students / 10 →
  selected_students = male_participants + female_participants →
  n = 1000 := by
sorry

end total_students_l528_52833


namespace c_share_l528_52866

/-- Represents the share of money for each person -/
structure Share where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- The theorem stating C's share given the conditions -/
theorem c_share (s : Share) 
  (h1 : s.a / s.b = 5 / 3)
  (h2 : s.b / s.c = 3 / 2)
  (h3 : s.c / s.d = 2 / 3)
  (h4 : s.a = s.b + 1000) :
  s.c = 1000 := by
sorry

end c_share_l528_52866


namespace annie_thyme_pots_l528_52850

/-- The number of pots of thyme Annie planted -/
def thyme_pots : ℕ := sorry

/-- The total number of leaves -/
def total_leaves : ℕ := 354

/-- The number of pots of basil -/
def basil_pots : ℕ := 3

/-- The number of pots of rosemary -/
def rosemary_pots : ℕ := 9

/-- The number of leaves per basil plant -/
def leaves_per_basil : ℕ := 4

/-- The number of leaves per rosemary plant -/
def leaves_per_rosemary : ℕ := 18

/-- The number of leaves per thyme plant -/
def leaves_per_thyme : ℕ := 30

theorem annie_thyme_pots : 
  thyme_pots = 6 :=
by sorry

end annie_thyme_pots_l528_52850


namespace product_remainder_mod_25_l528_52896

theorem product_remainder_mod_25 : (1523 * 1857 * 1919 * 2012) % 25 = 8 := by
  sorry

end product_remainder_mod_25_l528_52896


namespace cube_root_scaling_l528_52831

theorem cube_root_scaling (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h1 : a^(1/3) = 2.938) (h2 : b^(1/3) = 6.329) (h3 : c = 253600) 
  (h4 : b = 10 * a) (h5 : c = 1000 * b) : 
  c^(1/3) = 63.29 := by sorry

end cube_root_scaling_l528_52831


namespace point_on_y_axis_l528_52890

/-- If point M (a+3, 2a-2) is on the y-axis, then its coordinates are (0, -8) -/
theorem point_on_y_axis (a : ℝ) : 
  (a + 3 = 0) → ((a + 3, 2*a - 2) : ℝ × ℝ) = (0, -8) := by
  sorry

end point_on_y_axis_l528_52890


namespace minimal_fraction_sum_l528_52825

theorem minimal_fraction_sum (a b : ℕ+) (h : (45 : ℚ) / 11 < (a : ℚ) / b ∧ (a : ℚ) / b < 5 / 11) :
  (∀ c d : ℕ+, (45 : ℚ) / 11 < (c : ℚ) / d ∧ (c : ℚ) / d < 5 / 11 → c + d ≥ a + b) →
  a = 3 ∧ b = 7 :=
sorry

end minimal_fraction_sum_l528_52825


namespace product_signs_l528_52853

theorem product_signs (a b c d e : ℝ) : 
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 →
  (∃ (pos neg : Finset (ℝ × ℝ)), 
    pos.card = 5 ∧ 
    neg.card = 5 ∧ 
    (∀ p ∈ pos, p.1 + p.2 > 0) ∧
    (∀ p ∈ neg, p.1 + p.2 < 0) ∧
    pos ∩ neg = ∅ ∧
    pos ∪ neg = {(a,b), (a,c), (a,d), (a,e), (b,c), (b,d), (b,e), (c,d), (c,e), (d,e)}) →
  (∃ (pos_prod neg_prod : Finset (ℝ × ℝ)),
    pos_prod.card = 4 ∧
    neg_prod.card = 6 ∧
    (∀ p ∈ pos_prod, p.1 * p.2 > 0) ∧
    (∀ p ∈ neg_prod, p.1 * p.2 < 0) ∧
    pos_prod ∩ neg_prod = ∅ ∧
    pos_prod ∪ neg_prod = {(a,b), (a,c), (a,d), (a,e), (b,c), (b,d), (b,e), (c,d), (c,e), (d,e)}) :=
by sorry

end product_signs_l528_52853


namespace max_distance_complex_l528_52882

/-- Given a complex number z₁ = i(1-i)³ and any complex number z such that |z| = 1,
    the maximum value of |z - z₁| is 1 + 2√2. -/
theorem max_distance_complex (z : ℂ) : 
  let z₁ : ℂ := Complex.I * (1 - Complex.I)^3
  Complex.abs z = 1 →
  (⨆ (z : ℂ), Complex.abs (z - z₁)) = 1 + 2 * Real.sqrt 2 :=
by sorry

end max_distance_complex_l528_52882


namespace division_multiplication_result_l528_52837

theorem division_multiplication_result : (-1 : ℚ) / (-5 : ℚ) * (-1/5 : ℚ) = -1/25 := by
  sorry

end division_multiplication_result_l528_52837


namespace convex_polygon_not_divisible_into_nonconvex_quadrilaterals_l528_52841

-- Define a polygon as a set of points in the plane
def Polygon : Type := Set (ℝ × ℝ)

-- Define convexity for a polygon
def IsConvex (P : Polygon) : Prop := sorry

-- Define a quadrilateral as a polygon with exactly 4 vertices
def Quadrilateral (Q : Polygon) : Prop := sorry

-- Define nonconvexity for a quadrilateral
def IsNonConvex (Q : Polygon) : Prop := Quadrilateral Q ∧ ¬IsConvex Q

-- Main theorem
theorem convex_polygon_not_divisible_into_nonconvex_quadrilaterals 
  (M : Polygon) (n : ℕ) (M_i : Fin n → Polygon) :
  IsConvex M →
  (∀ i, IsNonConvex (M_i i)) →
  M ≠ ⋃ i, M_i i :=
sorry

end convex_polygon_not_divisible_into_nonconvex_quadrilaterals_l528_52841


namespace ellipse_slope_l528_52873

theorem ellipse_slope (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  let e : ℝ := 1/3
  let c : ℝ := a * e
  let k : ℝ := (b^2/a) / (c - (-a))
  (∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 → 
    (x = -a ∧ y = 0) ∨ (x = c ∧ y = b^2/a)) →
  k = 2/3 := by sorry

end ellipse_slope_l528_52873


namespace three_plane_division_l528_52814

/-- The number of regions that n planes can divide 3-dimensional space into -/
def regions (n : ℕ) : ℕ := sorry

/-- The minimum number of regions that 3 planes can divide 3-dimensional space into -/
def min_regions : ℕ := regions 3

/-- The maximum number of regions that 3 planes can divide 3-dimensional space into -/
def max_regions : ℕ := regions 3

theorem three_plane_division :
  min_regions = 4 ∧ max_regions = 8 := by sorry

end three_plane_division_l528_52814


namespace units_digit_of_sum_of_squares_of_first_1503_odd_integers_l528_52880

def first_n_odd_integers (n : ℕ) : List ℕ :=
  List.range n |> List.map (fun i => 2 * i + 1)

def sum_of_squares (list : List ℕ) : ℕ :=
  list.map (fun x => x * x) |> List.sum

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_of_sum_of_squares_of_first_1503_odd_integers :
  units_digit (sum_of_squares (first_n_odd_integers 1503)) = 5 := by
  sorry

end units_digit_of_sum_of_squares_of_first_1503_odd_integers_l528_52880


namespace difference_of_squares_81_49_l528_52874

theorem difference_of_squares_81_49 : 81^2 - 49^2 = 4160 := by
  sorry

end difference_of_squares_81_49_l528_52874


namespace cube_root_simplification_and_rationalization_l528_52851

theorem cube_root_simplification_and_rationalization :
  let x := (Real.rpow 6 (1/3)) / (Real.rpow 7 (1/3))
  let y := (Real.rpow 8 (1/3)) / (Real.rpow 9 (1/3))
  let z := (Real.rpow 10 (1/3)) / (Real.rpow 11 (1/3))
  x * y * z = (Real.rpow 223948320 (1/3)) / 693 := by
  sorry

end cube_root_simplification_and_rationalization_l528_52851


namespace random_events_identification_l528_52839

structure Event where
  description : String
  is_random : Bool

def event1 : Event := { 
  description := "An object will fall freely under the influence of gravity alone",
  is_random := false 
}

def event2 : Event := { 
  description := "The equation x^2 + 2x + 8 = 0 has two real roots",
  is_random := false 
}

def event3 : Event := { 
  description := "A certain information desk receives more than 10 requests for information consultation during a certain period of the day",
  is_random := true 
}

def event4 : Event := { 
  description := "It will rain next Saturday",
  is_random := true 
}

def events : List Event := [event1, event2, event3, event4]

theorem random_events_identification : 
  (events.filter (λ e => e.is_random)).map (λ e => e.description) = 
  [event3.description, event4.description] := by sorry

end random_events_identification_l528_52839
