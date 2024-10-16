import Mathlib

namespace NUMINAMATH_CALUDE_find_number_l495_49537

theorem find_number : ∃ x : ℝ, ((55 + x) / 7 + 40) * 5 = 555 ∧ x = 442 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l495_49537


namespace NUMINAMATH_CALUDE_shaded_area_ratio_l495_49598

/-- Given a rectangle divided into a 4x5 grid of 1cm x 1cm smaller rectangles,
    with a shaded area consisting of 3 full small rectangles and 4 half small rectangles,
    prove that the ratio of the shaded area to the total area is 1/4. -/
theorem shaded_area_ratio (total_width : ℝ) (total_height : ℝ) 
  (full_rectangles : ℕ) (half_rectangles : ℕ) :
  total_width = 4 →
  total_height = 5 →
  full_rectangles = 3 →
  half_rectangles = 4 →
  (full_rectangles + half_rectangles / 2) / (total_width * total_height) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_ratio_l495_49598


namespace NUMINAMATH_CALUDE_suit_tie_discount_cost_l495_49512

/-- Represents the cost calculation for two discount options in a suit and tie sale. -/
theorem suit_tie_discount_cost 
  (suit_price : ℕ) 
  (tie_price : ℕ) 
  (num_suits : ℕ) 
  (num_ties : ℕ) 
  (h1 : suit_price = 500)
  (h2 : tie_price = 100)
  (h3 : num_suits = 20)
  (h4 : num_ties > 20) :
  (num_suits * suit_price + (num_ties - num_suits) * tie_price = 100 * num_ties + 8000) ∧ 
  (((num_suits * suit_price + num_ties * tie_price) * 90) / 100 = 90 * num_ties + 9000) := by
  sorry

end NUMINAMATH_CALUDE_suit_tie_discount_cost_l495_49512


namespace NUMINAMATH_CALUDE_tangent_slope_at_one_l495_49547

-- Define the function
def f (x : ℝ) : ℝ := 2 * x^2 - 2 * x

-- State the theorem
theorem tangent_slope_at_one : 
  (deriv f) 1 = 2 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_one_l495_49547


namespace NUMINAMATH_CALUDE_cubical_box_edge_length_cubical_box_edge_length_proof_l495_49526

/-- The edge length of a cubical box that can hold 64 cubes with edge length 25 cm is 1 meter. -/
theorem cubical_box_edge_length : Real → Prop := fun edge_length =>
  let small_cube_volume := (25 / 100) ^ 3
  let box_volume := 64 * small_cube_volume
  edge_length ^ 3 = box_volume → edge_length = 1

/-- Proof of the cubical box edge length theorem -/
theorem cubical_box_edge_length_proof : cubical_box_edge_length 1 := by
  sorry


end NUMINAMATH_CALUDE_cubical_box_edge_length_cubical_box_edge_length_proof_l495_49526


namespace NUMINAMATH_CALUDE_log_problem_l495_49506

theorem log_problem (m : ℝ) : 
  (Real.log 4 / Real.log 3) * (Real.log 8 / Real.log 4) * (Real.log m / Real.log 8) = Real.log 16 / Real.log 4 → 
  m = 9 := by
sorry

end NUMINAMATH_CALUDE_log_problem_l495_49506


namespace NUMINAMATH_CALUDE_fraction_equality_l495_49557

theorem fraction_equality (x y : ℝ) (h : x ≠ y) : (x - y) / (x^2 - y^2) = 1 / (x + y) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l495_49557


namespace NUMINAMATH_CALUDE_segment_length_product_l495_49511

theorem segment_length_product (b₁ b₂ : ℝ) : 
  (((3 * b₁ - 5)^2 + (b₁ + 3)^2 = 45) ∧ 
   ((3 * b₂ - 5)^2 + (b₂ + 3)^2 = 45) ∧ 
   b₁ ≠ b₂) → 
  b₁ * b₂ = -11/10 := by
sorry

end NUMINAMATH_CALUDE_segment_length_product_l495_49511


namespace NUMINAMATH_CALUDE_reflected_ray_equation_l495_49550

/-- Given that:
  - Point A is at (-1/2, 0)
  - Point B is at (0, 1)
  - A' is the reflection of A across the y-axis
Prove that the line passing through A' and B has the equation 2x + y - 1 = 0 -/
theorem reflected_ray_equation (A : ℝ × ℝ) (B : ℝ × ℝ) (A' : ℝ × ℝ) :
  A = (-1/2, 0) →
  B = (0, 1) →
  A'.1 = -A.1 →  -- A' is reflection of A across y-axis
  A'.2 = A.2 →   -- A' is reflection of A across y-axis
  ∀ (x y : ℝ), (x = A'.1 ∧ y = A'.2) ∨ (x = B.1 ∧ y = B.2) →
    2 * x + y - 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_reflected_ray_equation_l495_49550


namespace NUMINAMATH_CALUDE_andrew_stamps_hundred_permits_l495_49531

/-- Calculates the number of permits Andrew stamps in a day -/
def permits_stamped (num_appointments : ℕ) (appointment_duration : ℕ) (workday_hours : ℕ) (stamps_per_hour : ℕ) : ℕ :=
  let appointment_time := num_appointments * appointment_duration
  let stamping_time := workday_hours - appointment_time
  stamping_time * stamps_per_hour

/-- Proves that Andrew stamps 100 permits given the specified conditions -/
theorem andrew_stamps_hundred_permits :
  permits_stamped 2 3 8 50 = 100 := by
  sorry

end NUMINAMATH_CALUDE_andrew_stamps_hundred_permits_l495_49531


namespace NUMINAMATH_CALUDE_select_five_from_ten_l495_49572

theorem select_five_from_ten : Nat.choose 10 5 = 2520 := by
  sorry

end NUMINAMATH_CALUDE_select_five_from_ten_l495_49572


namespace NUMINAMATH_CALUDE_stating_optimal_swap_distance_maximizes_total_distance_l495_49571

/-- Front tire lifespan in kilometers -/
def front_lifespan : ℝ := 11000

/-- Rear tire lifespan in kilometers -/
def rear_lifespan : ℝ := 9000

/-- The optimal swap distance in kilometers -/
def optimal_swap_distance : ℝ := 4950

/-- 
Theorem stating that the optimal swap distance maximizes total distance traveled
while ensuring both tires wear out simultaneously.
-/
theorem optimal_swap_distance_maximizes_total_distance :
  let total_distance := front_lifespan + rear_lifespan
  let front_remaining := 1 - (optimal_swap_distance / front_lifespan)
  let rear_remaining := 1 - (optimal_swap_distance / rear_lifespan)
  let distance_after_swap := front_remaining * rear_lifespan
  (front_remaining * rear_lifespan = rear_remaining * front_lifespan) ∧
  (optimal_swap_distance + distance_after_swap = total_distance) ∧
  (∀ x : ℝ, x ≠ optimal_swap_distance →
    let front_remaining' := 1 - (x / front_lifespan)
    let rear_remaining' := 1 - (x / rear_lifespan)
    let distance_after_swap' := min (front_remaining' * rear_lifespan) (rear_remaining' * front_lifespan)
    x + distance_after_swap' ≤ total_distance) :=
by
  sorry

end NUMINAMATH_CALUDE_stating_optimal_swap_distance_maximizes_total_distance_l495_49571


namespace NUMINAMATH_CALUDE_intersection_theorem_l495_49513

/-- A line passing through two points -/
structure Line1 where
  x1 : ℝ
  y1 : ℝ
  x2 : ℝ
  y2 : ℝ

/-- A line described by y = mx + b -/
structure Line2 where
  m : ℝ
  b : ℝ

/-- The intersection point of two lines -/
def intersection_point (l1 : Line1) (l2 : Line2) : ℝ × ℝ :=
  sorry

theorem intersection_theorem :
  let l1 : Line1 := { x1 := 0, y1 := 3, x2 := 4, y2 := 11 }
  let l2 : Line2 := { m := -1, b := 15 }
  intersection_point l1 l2 = (4, 11) := by
  sorry

end NUMINAMATH_CALUDE_intersection_theorem_l495_49513


namespace NUMINAMATH_CALUDE_ellipse_inequality_l495_49519

noncomputable section

-- Define the ellipse
def Ellipse (x y : ℝ) : Prop := x^2 / 8 + y^2 / 2 = 1

-- Define the right vertex C
def C : ℝ × ℝ := (2 * Real.sqrt 2, 0)

-- Define a point A on the ellipse in the first quadrant
def A (α : ℝ) : ℝ × ℝ := (2 * Real.sqrt 2 * Real.cos α, Real.sqrt 2 * Real.sin α)

-- Define point B symmetric to A with respect to the origin
def B (α : ℝ) : ℝ × ℝ := (-2 * Real.sqrt 2 * Real.cos α, -Real.sqrt 2 * Real.sin α)

-- Define point D
def D (α : ℝ) : ℝ × ℝ := (2 * Real.sqrt 2 * Real.cos α, 
  (Real.sqrt 2 * Real.sin α * (1 - Real.cos α)) / (1 + Real.cos α))

-- State the theorem
theorem ellipse_inequality (α : ℝ) 
  (h1 : 0 < α ∧ α < π/2)  -- Ensure A is in the first quadrant
  (h2 : Ellipse (A α).1 (A α).2)  -- Ensure A is on the ellipse
  : ‖A α - C‖^2 < ‖C - D α‖ * ‖D α - B α‖ := by
  sorry

end

end NUMINAMATH_CALUDE_ellipse_inequality_l495_49519


namespace NUMINAMATH_CALUDE_isabel_candy_count_l495_49523

/-- Calculates the total number of candy pieces Isabel has -/
def total_candy (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem: Given Isabel's initial candy count and the additional pieces she received,
    prove that her total candy count is 93 -/
theorem isabel_candy_count :
  let initial := 68
  let additional := 25
  total_candy initial additional = 93 := by
  sorry

end NUMINAMATH_CALUDE_isabel_candy_count_l495_49523


namespace NUMINAMATH_CALUDE_max_value_trig_product_max_value_trig_product_achievable_l495_49510

theorem max_value_trig_product (x y z : ℝ) :
  (Real.sin x + Real.sin (2 * y) + Real.sin (3 * z)) *
  (Real.cos x + Real.cos (2 * y) + Real.cos (3 * z)) ≤ 4.5 :=
by sorry

theorem max_value_trig_product_achievable :
  ∃ x y z : ℝ,
    (Real.sin x + Real.sin (2 * y) + Real.sin (3 * z)) *
    (Real.cos x + Real.cos (2 * y) + Real.cos (3 * z)) = 4.5 :=
by sorry

end NUMINAMATH_CALUDE_max_value_trig_product_max_value_trig_product_achievable_l495_49510


namespace NUMINAMATH_CALUDE_bakers_sales_l495_49532

/-- Baker's cake and pastry sales problem -/
theorem bakers_sales (cakes_made pastries_made cakes_sold pastries_sold : ℕ) 
  (h1 : cakes_made = 157)
  (h2 : pastries_made = 169)
  (h3 : cakes_sold = 158)
  (h4 : pastries_sold = 147) :
  cakes_sold - pastries_sold = 11 := by
  sorry

end NUMINAMATH_CALUDE_bakers_sales_l495_49532


namespace NUMINAMATH_CALUDE_gcd_840_1764_l495_49540

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end NUMINAMATH_CALUDE_gcd_840_1764_l495_49540


namespace NUMINAMATH_CALUDE_ab_plus_cd_value_l495_49567

theorem ab_plus_cd_value (a b c d : ℝ) 
  (eq1 : a + b + c = 3)
  (eq2 : a + b + d = -5)
  (eq3 : a + c + d = 10)
  (eq4 : b + c + d = -1) :
  a * b + c * d = -274/9 := by
sorry

end NUMINAMATH_CALUDE_ab_plus_cd_value_l495_49567


namespace NUMINAMATH_CALUDE_complex_modulus_equation_l495_49529

theorem complex_modulus_equation (t : ℝ) : 
  t > 0 → Complex.abs (8 + 3 * t * Complex.I) = 13 → t = Real.sqrt 105 / 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_equation_l495_49529


namespace NUMINAMATH_CALUDE_half_angle_quadrant_l495_49590

/-- Given an angle θ in the second quadrant, this theorem states that θ/2 lies in either the first or third quadrant. -/
theorem half_angle_quadrant (θ : Real) : 
  (π / 2 < θ ∧ θ < π) → 
  (0 < θ / 2 ∧ θ / 2 < π / 2) ∨ (π < θ / 2 ∧ θ / 2 < 3 * π / 2) := by
  sorry

end NUMINAMATH_CALUDE_half_angle_quadrant_l495_49590


namespace NUMINAMATH_CALUDE_eggplant_seed_distribution_l495_49530

theorem eggplant_seed_distribution (total_seeds : ℕ) (num_pots : ℕ) (seeds_in_last_pot : ℕ) :
  total_seeds = 10 →
  num_pots = 4 →
  seeds_in_last_pot = 1 →
  ∃ (seeds_per_pot : ℕ),
    seeds_per_pot * (num_pots - 1) + seeds_in_last_pot = total_seeds ∧
    seeds_per_pot = 3 :=
by sorry

end NUMINAMATH_CALUDE_eggplant_seed_distribution_l495_49530


namespace NUMINAMATH_CALUDE_line_through_intersection_parallel_to_l₃_l495_49544

-- Define the lines l₁, l₂, and l₃
def l₁ (x y : ℝ) : Prop := 3 * x + 4 * y - 2 = 0
def l₂ (x y : ℝ) : Prop := 2 * x + y + 2 = 0
def l₃ (x y : ℝ) : Prop := 4 * x + 3 * y - 2 = 0

-- Define the intersection point of l₁ and l₂
def intersection_point (x y : ℝ) : Prop := l₁ x y ∧ l₂ x y

-- Define parallel lines
def parallel (f g : ℝ → ℝ → Prop) : Prop :=
  ∃ (k : ℝ), ∀ (x y : ℝ), f x y ↔ g (x + k) y

-- Theorem statement
theorem line_through_intersection_parallel_to_l₃ :
  ∃ (a b c : ℝ), 
    (∀ (x y : ℝ), intersection_point x y → a * x + b * y + c = 0) ∧
    parallel (fun x y => a * x + b * y + c = 0) l₃ ∧
    (a = 4 ∧ b = 3 ∧ c = 2) :=
sorry

end NUMINAMATH_CALUDE_line_through_intersection_parallel_to_l₃_l495_49544


namespace NUMINAMATH_CALUDE_expression_value_l495_49564

theorem expression_value : 4^3 - 2 * 4^2 + 2 * 4 - 1 = 39 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l495_49564


namespace NUMINAMATH_CALUDE_max_single_player_salary_is_454000_l495_49543

/-- Represents a basketball team in the semi-professional league --/
structure BasketballTeam where
  players : Nat
  minSalary : Nat
  maxTotalSalary : Nat

/-- Calculates the maximum possible salary for a single player on the team --/
def maxSinglePlayerSalary (team : BasketballTeam) : Nat :=
  team.maxTotalSalary - (team.players - 1) * team.minSalary

/-- Theorem stating the maximum possible salary for a single player --/
theorem max_single_player_salary_is_454000 :
  let team := BasketballTeam.mk 23 18000 850000
  maxSinglePlayerSalary team = 454000 := by
  sorry

#eval maxSinglePlayerSalary (BasketballTeam.mk 23 18000 850000)

end NUMINAMATH_CALUDE_max_single_player_salary_is_454000_l495_49543


namespace NUMINAMATH_CALUDE_arcsin_equation_solution_l495_49556

theorem arcsin_equation_solution : 
  ∃ x : ℝ, x = Real.sqrt 102 / 51 ∧ Real.arcsin x + Real.arcsin (3 * x) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_equation_solution_l495_49556


namespace NUMINAMATH_CALUDE_chewing_gum_revenue_projection_l495_49517

theorem chewing_gum_revenue_projection (R : ℝ) (h : R > 0) :
  let projected_revenue := 1.40 * R
  let actual_revenue := 0.70 * R
  actual_revenue / projected_revenue = 0.50 := by
sorry

end NUMINAMATH_CALUDE_chewing_gum_revenue_projection_l495_49517


namespace NUMINAMATH_CALUDE_equation_a_solution_equation_b_no_solution_l495_49520

-- Part (a)
theorem equation_a_solution (x : ℚ) : 
  1 + 1 / (2 + 1 / ((4*x + 1) / (2*x + 1) - 1 / (2 + 1/x))) = 19/14 ↔ x = 1/2 :=
sorry

-- Part (b)
theorem equation_b_no_solution :
  ¬∃ (x : ℚ), ((2*x - 1)/2 + 4/3) / ((x - 1)/3 - 1/2 * (1 - 1/3)) - 
  (x + 4) / ((2*x + 1)/2 + 1/5 - 2 - 1/(1 + 1/(2 + 1/3))) = (9 - 2*x) / (2*x - 4) :=
sorry

end NUMINAMATH_CALUDE_equation_a_solution_equation_b_no_solution_l495_49520


namespace NUMINAMATH_CALUDE_f_increasing_iff_a_range_l495_49551

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 - a*x - 5 else a/x

theorem f_increasing_iff_a_range (a : ℝ) :
  (∀ x y, x < y → f a x < f a y) ↔ -3 ≤ a ∧ a ≤ -2 :=
by sorry

end NUMINAMATH_CALUDE_f_increasing_iff_a_range_l495_49551


namespace NUMINAMATH_CALUDE_green_peaches_count_l495_49528

/-- Given a basket of peaches, prove the number of green peaches. -/
theorem green_peaches_count (red : ℕ) (green : ℕ) : 
  red = 7 → green = red + 1 → green = 8 := by
  sorry

end NUMINAMATH_CALUDE_green_peaches_count_l495_49528


namespace NUMINAMATH_CALUDE_no_real_roots_quadratic_l495_49568

theorem no_real_roots_quadratic (k : ℝ) : 
  (∀ x : ℝ, 2 * x^2 + x - k ≠ 0) ↔ k < -1/8 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_quadratic_l495_49568


namespace NUMINAMATH_CALUDE_problem_statement_l495_49535

theorem problem_statement (a b c p q : ℕ) (hp : p > q) 
  (h_sum : a + b + c = 2 * p * q * (p^30 + q^30)) : 
  let k := a^3 + b^3 + c^3
  ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ k = x * y ∧ 
  (∀ (a' b' c' : ℕ), a' + b' + c' = 2 * p * q * (p^30 + q^30) → 
    a' * b' * c' ≤ a * b * c → 1984 ∣ k) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l495_49535


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_properties_l495_49522

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 3 + y^2 / 2 = 1

-- Define the line
def line (x y k : ℝ) : Prop := y = k * x + 1

-- Define the foci
def F1 : ℝ × ℝ := (-1, 0)
def F2 : ℝ × ℝ := (1, 0)

-- Define the intersection points
variable (A B : ℝ × ℝ)

-- Define the parallel condition
def parallel (F1 A F2 B : ℝ × ℝ) : Prop :=
  (A.1 - F1.1) * (B.2 - F2.2) = (A.2 - F1.2) * (B.1 - F2.1)

-- Define the perpendicular condition
def perpendicular (A F1 F2 : ℝ × ℝ) : Prop :=
  (A.1 - F1.1) * (A.1 - F2.1) + (A.2 - F1.2) * (A.2 - F2.2) = 0

-- Theorem statement
theorem ellipse_line_intersection_properties
  (k : ℝ)
  (hA : ellipse A.1 A.2 ∧ line A.1 A.2 k)
  (hB : ellipse B.1 B.2 ∧ line B.1 B.2 k) :
  ¬(parallel F1 A F2 B) ∧ ¬(perpendicular A F1 F2) := by sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_properties_l495_49522


namespace NUMINAMATH_CALUDE_kenneth_to_micah_ratio_l495_49548

/-- The number of fish Micah has -/
def micah_fish : ℕ := 7

/-- The number of fish Kenneth has -/
def kenneth_fish : ℕ := 21

/-- The number of fish Matthias has -/
def matthias_fish : ℕ := kenneth_fish - 15

/-- The total number of fish the boys have -/
def total_fish : ℕ := 34

/-- Theorem stating that the ratio of Kenneth's fish to Micah's fish is 3:1 -/
theorem kenneth_to_micah_ratio :
  micah_fish + kenneth_fish + matthias_fish = total_fish →
  kenneth_fish / micah_fish = 3 := by
  sorry

end NUMINAMATH_CALUDE_kenneth_to_micah_ratio_l495_49548


namespace NUMINAMATH_CALUDE_square_perimeter_when_area_equals_side_l495_49500

/-- A square with area numerically equal to its side length has a perimeter of 4 units. -/
theorem square_perimeter_when_area_equals_side : ∀ s : ℝ,
  s > 0 → s^2 = s → 4 * s = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_when_area_equals_side_l495_49500


namespace NUMINAMATH_CALUDE_calculation_proof_l495_49518

theorem calculation_proof : Real.sqrt 4 + |3 - π| + (1/3)⁻¹ = 2 + π := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l495_49518


namespace NUMINAMATH_CALUDE_a_range_l495_49563

theorem a_range (a : ℝ) (h1 : a < 9 * a^3 - 11 * a) (h2 : 9 * a^3 - 11 * a < |a|) (h3 : a < 0) :
  -2 * Real.sqrt 3 / 3 < a ∧ a < -Real.sqrt 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_a_range_l495_49563


namespace NUMINAMATH_CALUDE_fifth_month_sale_l495_49555

def average_sale : ℕ := 5600
def num_months : ℕ := 6
def sale_month1 : ℕ := 5400
def sale_month2 : ℕ := 9000
def sale_month3 : ℕ := 6300
def sale_month4 : ℕ := 7200
def sale_month6 : ℕ := 1200

theorem fifth_month_sale :
  ∃ (sale_month5 : ℕ),
    sale_month5 = average_sale * num_months - (sale_month1 + sale_month2 + sale_month3 + sale_month4 + sale_month6) ∧
    sale_month5 = 4500 := by
  sorry

end NUMINAMATH_CALUDE_fifth_month_sale_l495_49555


namespace NUMINAMATH_CALUDE_ladder_wood_length_50ft_l495_49538

/-- Calculates the total length of wood needed for ladder rungs -/
def ladder_wood_length (rung_length inches_between_rungs total_height_feet : ℚ) : ℚ :=
  let inches_per_foot : ℚ := 12
  let total_height_inches : ℚ := total_height_feet * inches_per_foot
  let space_per_rung : ℚ := rung_length + inches_between_rungs
  let num_rungs : ℚ := total_height_inches / space_per_rung
  (num_rungs * rung_length) / inches_per_foot

/-- The total length of wood needed for rungs to climb 50 feet is 37.5 feet -/
theorem ladder_wood_length_50ft :
  ladder_wood_length 18 6 50 = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_ladder_wood_length_50ft_l495_49538


namespace NUMINAMATH_CALUDE_solve_equation_l495_49579

theorem solve_equation : ∀ x : ℝ, 2 * 3 * 4 = 6 * x → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l495_49579


namespace NUMINAMATH_CALUDE_min_value_theorem_l495_49595

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (1 / x) + (4 / (y + 1)) ≥ 9 / 2 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + y₀ = 1 ∧ (1 / x₀) + (4 / (y₀ + 1)) = 9 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l495_49595


namespace NUMINAMATH_CALUDE_hannahs_peppers_l495_49587

theorem hannahs_peppers (green_peppers red_peppers : ℚ) 
  (h1 : green_peppers = 0.3333333333333333)
  (h2 : red_peppers = 0.3333333333333333) :
  green_peppers + red_peppers = 0.6666666666666666 := by
  sorry

end NUMINAMATH_CALUDE_hannahs_peppers_l495_49587


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l495_49589

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift_parabola (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
  , b := -2 * p.a * h + p.b
  , c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_shift_theorem :
  let original := Parabola.mk 1 0 0  -- y = x^2
  let shifted := shift_parabola original 1 2
  shifted = Parabola.mk 1 (-2) 3  -- y = (x-1)^2 + 2
  := by sorry

end NUMINAMATH_CALUDE_parabola_shift_theorem_l495_49589


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l495_49594

theorem arithmetic_calculations :
  (156 - 135 / 9 = 141) ∧
  ((124 - 56) / 4 = 17) ∧
  (55 * 6 + 45 * 6 = 600) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l495_49594


namespace NUMINAMATH_CALUDE_brenda_mice_fraction_l495_49586

/-- The fraction of baby mice Brenda gave to Robbie -/
def f : ℚ := sorry

/-- The total number of baby mice -/
def total_mice : ℕ := 3 * 8

theorem brenda_mice_fraction :
  (f * total_mice : ℚ) +                        -- Mice given to Robbie
  (3 * f * total_mice : ℚ) +                    -- Mice sold to pet store
  ((1 - 4 * f) * total_mice / 2 : ℚ) +          -- Mice sold to snake owners
  4 = total_mice ∧                              -- Remaining mice
  f = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_brenda_mice_fraction_l495_49586


namespace NUMINAMATH_CALUDE_good_student_count_l495_49505

/-- Represents a student in the class -/
inductive Student
| Good
| Troublemaker

/-- The total number of students in the class -/
def totalStudents : Nat := 25

/-- The number of students who made the first statement -/
def firstStatementCount : Nat := 5

/-- The number of students who made the second statement -/
def secondStatementCount : Nat := 20

/-- Checks if the first statement is true for a given number of good students -/
def firstStatementTrue (goodCount : Nat) : Prop :=
  totalStudents - goodCount > (totalStudents - 1) / 2

/-- Checks if the second statement is true for a given number of good students -/
def secondStatementTrue (goodCount : Nat) : Prop :=
  totalStudents - goodCount = 3 * (goodCount - 1)

/-- Theorem stating that the number of good students is either 5 or 7 -/
theorem good_student_count :
  ∃ (goodCount : Nat), (goodCount = 5 ∨ goodCount = 7) ∧
    (firstStatementTrue goodCount ∨ ¬firstStatementTrue goodCount) ∧
    (secondStatementTrue goodCount ∨ ¬secondStatementTrue goodCount) ∧
    goodCount ≤ totalStudents :=
  sorry

end NUMINAMATH_CALUDE_good_student_count_l495_49505


namespace NUMINAMATH_CALUDE_apple_bags_l495_49515

theorem apple_bags (A B C : ℕ) 
  (h1 : A + B + C = 24) 
  (h2 : A + B = 11) 
  (h3 : B + C = 18) : 
  A + C = 19 := by
  sorry

end NUMINAMATH_CALUDE_apple_bags_l495_49515


namespace NUMINAMATH_CALUDE_instructors_reunion_l495_49504

/-- The number of weeks between Rita's teaching sessions -/
def rita_weeks : ℕ := 5

/-- The number of weeks between Pedro's teaching sessions -/
def pedro_weeks : ℕ := 8

/-- The number of weeks between Elaine's teaching sessions -/
def elaine_weeks : ℕ := 10

/-- The number of weeks between Moe's teaching sessions -/
def moe_weeks : ℕ := 9

/-- The number of weeks until all instructors teach together again -/
def weeks_until_reunion : ℕ := 360

theorem instructors_reunion :
  Nat.lcm rita_weeks (Nat.lcm pedro_weeks (Nat.lcm elaine_weeks moe_weeks)) = weeks_until_reunion :=
sorry

end NUMINAMATH_CALUDE_instructors_reunion_l495_49504


namespace NUMINAMATH_CALUDE_don_rum_limit_l495_49527

/-- The amount of rum Sally gave Don on his pancakes (in oz) -/
def sally_rum : ℝ := 10

/-- The multiplier for the maximum amount of rum Don can consume for a healthy diet -/
def max_multiplier : ℝ := 3

/-- The amount of rum Don had earlier that day (in oz) -/
def earlier_rum : ℝ := 12

/-- The amount of rum Don can have after eating all of the rum and pancakes (in oz) -/
def remaining_rum : ℝ := max_multiplier * sally_rum - earlier_rum

theorem don_rum_limit : remaining_rum = 18 := by sorry

end NUMINAMATH_CALUDE_don_rum_limit_l495_49527


namespace NUMINAMATH_CALUDE_circle_equation_with_radius_5_l495_49581

/-- Given a circle with equation x^2 - 2x + y^2 + 6y + c = 0 and radius 5, prove c = -15 -/
theorem circle_equation_with_radius_5 (c : ℝ) :
  (∀ x y : ℝ, x^2 - 2*x + y^2 + 6*y + c = 0 ↔ (x - 1)^2 + (y + 3)^2 = 5^2) →
  c = -15 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_with_radius_5_l495_49581


namespace NUMINAMATH_CALUDE_range_m_when_p_necessary_for_q_range_m_when_not_p_necessary_not_sufficient_for_not_q_l495_49533

-- Define the conditions p and q
def p (x : ℝ) : Prop := x^2 - 8*x - 20 ≤ 0

def q (x m : ℝ) : Prop := 1 - m^2 ≤ x ∧ x ≤ 1 + m^2

-- Theorem 1: If p is a necessary condition for q, then the range of m is [-√3, √3]
theorem range_m_when_p_necessary_for_q :
  (∀ x m : ℝ, q x m → p x) →
  ∀ m : ℝ, -Real.sqrt 3 ≤ m ∧ m ≤ Real.sqrt 3 :=
sorry

-- Theorem 2: If ¬p is a necessary but not sufficient condition for ¬q, 
-- then the range of m is (-∞, -3] ∪ [3, +∞)
theorem range_m_when_not_p_necessary_not_sufficient_for_not_q :
  (∀ x m : ℝ, ¬(q x m) → ¬(p x)) ∧ 
  (∃ x m : ℝ, ¬(p x) ∧ q x m) →
  ∀ m : ℝ, m ≤ -3 ∨ m ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_range_m_when_p_necessary_for_q_range_m_when_not_p_necessary_not_sufficient_for_not_q_l495_49533


namespace NUMINAMATH_CALUDE_weeks_to_afford_bike_l495_49534

/-- The cost of the bike in dollars -/
def bike_cost : ℕ := 600

/-- The amount of birthday money Chandler received in dollars -/
def birthday_money : ℕ := 150

/-- Chandler's weekly earnings from tutoring in dollars -/
def weekly_earnings : ℕ := 14

/-- The function that calculates the total money Chandler has after working for a given number of weeks -/
def total_money (weeks : ℕ) : ℕ := birthday_money + weekly_earnings * weeks

/-- The theorem stating that 33 is the smallest number of weeks Chandler needs to work to afford the bike -/
theorem weeks_to_afford_bike : 
  (∀ w : ℕ, w < 33 → total_money w < bike_cost) ∧ 
  total_money 33 ≥ bike_cost := by
sorry

end NUMINAMATH_CALUDE_weeks_to_afford_bike_l495_49534


namespace NUMINAMATH_CALUDE_donna_additional_flyers_eq_five_l495_49585

/-- The number of flyers Maisie dropped off -/
def maisie_flyers : ℕ := 33

/-- The total number of flyers Donna dropped off -/
def donna_total_flyers : ℕ := 71

/-- The number of additional flyers Donna dropped off -/
def donna_additional_flyers : ℕ := donna_total_flyers - 2 * maisie_flyers

theorem donna_additional_flyers_eq_five : donna_additional_flyers = 5 := by
  sorry

end NUMINAMATH_CALUDE_donna_additional_flyers_eq_five_l495_49585


namespace NUMINAMATH_CALUDE_regular_polygon_150_degrees_has_12_sides_l495_49524

/-- Proves that a regular polygon with interior angles of 150 degrees has 12 sides -/
theorem regular_polygon_150_degrees_has_12_sides :
  ∀ n : ℕ, 
    n > 2 →
    (∀ angle : ℝ, angle = 150 → n * angle = (n - 2) * 180) →
    n = 12 :=
by
  sorry

#check regular_polygon_150_degrees_has_12_sides

end NUMINAMATH_CALUDE_regular_polygon_150_degrees_has_12_sides_l495_49524


namespace NUMINAMATH_CALUDE_product_of_odot_l495_49509

def A : Finset Int := {-2, 1}
def B : Finset Int := {-1, 2}

def odot (A B : Finset Int) : Finset Int :=
  (A.product B).image (fun (x : Int × Int) => x.1 * x.2)

theorem product_of_odot :
  (odot A B).prod id = 8 := by sorry

end NUMINAMATH_CALUDE_product_of_odot_l495_49509


namespace NUMINAMATH_CALUDE_max_value_m_plus_n_l495_49596

theorem max_value_m_plus_n (a b m n : ℝ) : 
  (a < 0 ∧ b < 0) →  -- a and b have the same sign (negative)
  (∀ x, ax^2 + 2*x + b < 0 ↔ x ≠ -1/a) →  -- solution set condition
  m = b + 1/a →  -- definition of m
  n = a + 1/b →  -- definition of n
  (∀ k, m + n ≤ k) → k = -4 :=
by sorry

end NUMINAMATH_CALUDE_max_value_m_plus_n_l495_49596


namespace NUMINAMATH_CALUDE_smaller_number_in_ratio_l495_49542

theorem smaller_number_in_ratio (a b c d x y : ℝ) : 
  0 < a → a < b → 0 < d → d < c →
  x > 0 → y > 0 →
  x / y = a / b →
  x + y = c - d →
  d = 2 * x - y →
  min x y = (2 * a * c - b * c) / (3 * (2 * a - b)) := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_in_ratio_l495_49542


namespace NUMINAMATH_CALUDE_second_neighbor_brought_fewer_hotdog_difference_l495_49599

/-- The number of hotdogs brought by the first neighbor -/
def first_neighbor_hotdogs : ℕ := 75

/-- The total number of hotdogs brought by both neighbors -/
def total_hotdogs : ℕ := 125

/-- The number of hotdogs brought by the second neighbor -/
def second_neighbor_hotdogs : ℕ := total_hotdogs - first_neighbor_hotdogs

/-- The second neighbor brought fewer hotdogs than the first -/
theorem second_neighbor_brought_fewer :
  second_neighbor_hotdogs < first_neighbor_hotdogs := by sorry

/-- The difference in hotdogs between the first and second neighbor is 25 -/
theorem hotdog_difference :
  first_neighbor_hotdogs - second_neighbor_hotdogs = 25 := by sorry

end NUMINAMATH_CALUDE_second_neighbor_brought_fewer_hotdog_difference_l495_49599


namespace NUMINAMATH_CALUDE_largest_multiple_of_8_less_than_neg_80_l495_49536

theorem largest_multiple_of_8_less_than_neg_80 :
  ∀ n : ℤ, n % 8 = 0 ∧ n < -80 → n ≤ -88 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_8_less_than_neg_80_l495_49536


namespace NUMINAMATH_CALUDE_tensor_A_equals_result_l495_49541

def A : Set ℕ := {0, 2, 3}

def tensorOp (S : Set ℕ) : Set ℕ :=
  {x | ∃ a b, a ∈ S ∧ b ∈ S ∧ x = a + b}

theorem tensor_A_equals_result : tensorOp A = {0, 2, 3, 4, 5, 6} := by
  sorry

end NUMINAMATH_CALUDE_tensor_A_equals_result_l495_49541


namespace NUMINAMATH_CALUDE_expression_value_l495_49552

theorem expression_value (x y : ℚ) (hx : x = -5/4) (hy : y = -3/2) :
  -2 * x - y^2 = 1/4 := by sorry

end NUMINAMATH_CALUDE_expression_value_l495_49552


namespace NUMINAMATH_CALUDE_eight_chickens_ten_eggs_l495_49593

/-- Given that 5 chickens lay 7 eggs in 4 days, this function calculates
    the number of days it takes for 8 chickens to lay 10 eggs. -/
def days_to_lay_eggs (initial_chickens : ℕ) (initial_eggs : ℕ) (initial_days : ℕ)
                     (target_chickens : ℕ) (target_eggs : ℕ) : ℚ :=
  (initial_chickens * initial_days * target_eggs : ℚ) /
  (initial_eggs * target_chickens : ℚ)

/-- Theorem stating that 8 chickens will take 50/7 days to lay 10 eggs,
    given that 5 chickens lay 7 eggs in 4 days. -/
theorem eight_chickens_ten_eggs :
  days_to_lay_eggs 5 7 4 8 10 = 50 / 7 := by
  sorry

#eval days_to_lay_eggs 5 7 4 8 10

end NUMINAMATH_CALUDE_eight_chickens_ten_eggs_l495_49593


namespace NUMINAMATH_CALUDE_product_mod_seven_l495_49573

theorem product_mod_seven : (2007 * 2008 * 2009 * 2010) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_seven_l495_49573


namespace NUMINAMATH_CALUDE_power_multiplication_result_l495_49545

theorem power_multiplication_result : 0.25^2023 * 4^2024 = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_result_l495_49545


namespace NUMINAMATH_CALUDE_square_tiles_problem_l495_49559

/-- 
Given a square area tiled with congruent square tiles,
if the total number of tiles on the two diagonals is 25,
then the total number of tiles covering the entire square area is 169.
-/
theorem square_tiles_problem (n : ℕ) : 
  n > 0 → 
  2 * n - 1 = 25 → 
  n ^ 2 = 169 := by
  sorry

end NUMINAMATH_CALUDE_square_tiles_problem_l495_49559


namespace NUMINAMATH_CALUDE_doctor_team_formations_l495_49584

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

theorem doctor_team_formations :
  let total_doctors : ℕ := 9
  let male_doctors : ℕ := 5
  let female_doctors : ℕ := 4
  let team_size : ℕ := 3
  let one_male_two_female : ℕ := choose male_doctors 1 * choose female_doctors 2
  let two_male_one_female : ℕ := choose male_doctors 2 * choose female_doctors 1
  one_male_two_female + two_male_one_female = 70 :=
sorry

end NUMINAMATH_CALUDE_doctor_team_formations_l495_49584


namespace NUMINAMATH_CALUDE_rational_nonzero_l495_49592

theorem rational_nonzero (a b : ℚ) (h1 : a * b > a) (h2 : a - b > b) : a ≠ 0 ∧ b ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_rational_nonzero_l495_49592


namespace NUMINAMATH_CALUDE_tiffany_score_l495_49578

/-- Tiffany's video game scoring system -/
structure GameScore where
  pointsPerTreasure : ℕ
  treasuresFirstLevel : ℕ
  treasuresSecondLevel : ℕ

/-- Calculate the total score based on the game rules -/
def totalScore (game : GameScore) : ℕ :=
  game.pointsPerTreasure * (game.treasuresFirstLevel + game.treasuresSecondLevel)

/-- Theorem: Tiffany's total score is 48 points -/
theorem tiffany_score :
  ∀ (game : GameScore),
  game.pointsPerTreasure = 6 →
  game.treasuresFirstLevel = 3 →
  game.treasuresSecondLevel = 5 →
  totalScore game = 48 :=
by
  sorry

end NUMINAMATH_CALUDE_tiffany_score_l495_49578


namespace NUMINAMATH_CALUDE_no_90_cents_possible_l495_49569

/-- Represents the types of coins available --/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter

/-- Returns the value of a coin in cents --/
def coinValue : Coin → Nat
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Quarter => 25

/-- Represents a selection of coins --/
structure CoinSelection :=
  (pennies : Nat)
  (nickels : Nat)
  (dimes : Nat)
  (quarters : Nat)

/-- Checks if a coin selection is valid according to the problem constraints --/
def isValidSelection (s : CoinSelection) : Prop :=
  s.pennies + s.nickels + s.dimes + s.quarters = 6 ∧
  s.pennies ≤ 4 ∧ s.nickels ≤ 4 ∧ s.dimes ≤ 4 ∧ s.quarters ≤ 4

/-- Calculates the total value of a coin selection in cents --/
def totalValue (s : CoinSelection) : Nat :=
  s.pennies * coinValue Coin.Penny +
  s.nickels * coinValue Coin.Nickel +
  s.dimes * coinValue Coin.Dime +
  s.quarters * coinValue Coin.Quarter

/-- Theorem stating that it's impossible to make 90 cents with a valid coin selection --/
theorem no_90_cents_possible :
  ¬∃ (s : CoinSelection), isValidSelection s ∧ totalValue s = 90 := by
  sorry


end NUMINAMATH_CALUDE_no_90_cents_possible_l495_49569


namespace NUMINAMATH_CALUDE_tangent_angle_at_one_l495_49501

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 2*x + 4

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 2

-- Theorem statement
theorem tangent_angle_at_one (x : ℝ) :
  let slope := f' 1
  let angle := Real.arctan slope
  angle = π/4 := by sorry

end NUMINAMATH_CALUDE_tangent_angle_at_one_l495_49501


namespace NUMINAMATH_CALUDE_barium_atoms_in_compound_l495_49574

/-- The number of Barium atoms in the compound -/
def num_barium_atoms : ℕ := 1

/-- The number of Oxygen atoms in the compound -/
def num_oxygen_atoms : ℕ := 2

/-- The number of Hydrogen atoms in the compound -/
def num_hydrogen_atoms : ℕ := 2

/-- The molecular weight of the compound in g/mol -/
def molecular_weight : ℝ := 171

/-- The atomic weight of Barium in g/mol -/
def atomic_weight_barium : ℝ := 137.33

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_oxygen : ℝ := 16.00

/-- The atomic weight of Hydrogen in g/mol -/
def atomic_weight_hydrogen : ℝ := 1.01

theorem barium_atoms_in_compound :
  num_barium_atoms = 1 :=
sorry

end NUMINAMATH_CALUDE_barium_atoms_in_compound_l495_49574


namespace NUMINAMATH_CALUDE_complex_fraction_equals_i_l495_49566

theorem complex_fraction_equals_i (m n : ℝ) (h : m + Complex.I = 1 + n * Complex.I) :
  (m + n * Complex.I) / (m - n * Complex.I) = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_i_l495_49566


namespace NUMINAMATH_CALUDE_perpendicular_length_is_five_l495_49539

/-- Properties of a right triangle DEF with given side lengths -/
structure RightTriangle where
  DE : ℝ
  EF : ℝ
  is_right : DE = 5 ∧ EF = 12

/-- The length of the perpendicular from the hypotenuse to the midpoint of the angle bisector -/
def perpendicular_length (t : RightTriangle) : ℝ :=
  sorry

/-- Theorem: The perpendicular length is 5 -/
theorem perpendicular_length_is_five (t : RightTriangle) :
  perpendicular_length t = 5 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_length_is_five_l495_49539


namespace NUMINAMATH_CALUDE_total_cost_theorem_l495_49591

def cost_of_meat (pork_price chicken_price pork_weight chicken_weight : ℝ) : ℝ :=
  pork_price * pork_weight + chicken_price * chicken_weight

theorem total_cost_theorem (pork_price : ℝ) (h1 : pork_price = 6) :
  let chicken_price := pork_price - 2
  cost_of_meat pork_price chicken_price 1 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_theorem_l495_49591


namespace NUMINAMATH_CALUDE_range_of_a_for_always_nonnegative_quadratic_l495_49597

theorem range_of_a_for_always_nonnegative_quadratic :
  {a : ℝ | ∀ x : ℝ, x^2 + a*x + a ≥ 0} = Set.Icc 0 4 := by sorry

end NUMINAMATH_CALUDE_range_of_a_for_always_nonnegative_quadratic_l495_49597


namespace NUMINAMATH_CALUDE_system_solution_l495_49508

/-- Prove that the given system of linear equations has the specified solution -/
theorem system_solution (x y : ℝ) : 
  (x = 2 ∧ y = -3) → (3 * x + y = 3 ∧ 4 * x - y = 11) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l495_49508


namespace NUMINAMATH_CALUDE_parabola_properties_l495_49576

/-- Parabola represented by its parameter p -/
structure Parabola where
  p : ℝ
  p_pos : p > 0

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given a parabola and a point on it, prove the standard form of the parabola
    and the ratio of distances from two points to the focus -/
theorem parabola_properties (E : Parabola) (A : Point)
    (h_on_parabola : A.y^2 = 2 * E.p * A.x)
    (h_y_pos : A.y > 0)
    (h_A_coords : A.x = 9 ∧ A.y = 6)
    (h_AF_length : 5 = |A.x - E.p| + |A.y|) : 
  (∀ (x y : ℝ), y^2 = 4*x ↔ y^2 = 2*E.p*x) ∧ 
  ∃ (B : Point), B ≠ A ∧ 
    (∃ (t : ℝ), 0 < t ∧ t < 1 ∧ 
      B.x = t * A.x + (1 - t) * E.p ∧
      B.y = t * A.y) ∧
    5 / (|B.x - E.p| + |B.y|) = 4 :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l495_49576


namespace NUMINAMATH_CALUDE_perimeter_plus_area_equals_9_sqrt_41_l495_49575

/-- A parallelogram with integer coordinates -/
structure Parallelogram where
  a : ℤ × ℤ
  b : ℤ × ℤ
  c : ℤ × ℤ
  d : ℤ × ℤ

/-- The specific parallelogram from the problem -/
def specificParallelogram : Parallelogram :=
  { a := (0, 0),
    b := (4, 5),
    c := (11, 5),
    d := (7, 0) }

/-- Calculate the perimeter of a parallelogram -/
def perimeter (p : Parallelogram) : ℝ :=
  sorry

/-- Calculate the area of a parallelogram -/
def area (p : Parallelogram) : ℝ :=
  sorry

/-- The main theorem to prove -/
theorem perimeter_plus_area_equals_9_sqrt_41 :
  perimeter specificParallelogram + area specificParallelogram = 9 * Real.sqrt 41 :=
sorry

end NUMINAMATH_CALUDE_perimeter_plus_area_equals_9_sqrt_41_l495_49575


namespace NUMINAMATH_CALUDE_least_possible_average_of_four_integers_l495_49503

theorem least_possible_average_of_four_integers (a b c d : ℤ) : 
  a < b ∧ b < c ∧ c < d ∧  -- Four different integers
  d = 90 ∧                 -- Largest integer is 90
  a ≥ 21 →                 -- Smallest integer is at least 21
  (a + b + c + d) / 4 ≥ 39 ∧ 
  ∃ (x y z w : ℤ), x < y ∧ y < z ∧ z < w ∧ w = 90 ∧ x ≥ 21 ∧ (x + y + z + w) / 4 = 39 :=
by
  sorry

#check least_possible_average_of_four_integers

end NUMINAMATH_CALUDE_least_possible_average_of_four_integers_l495_49503


namespace NUMINAMATH_CALUDE_eight_times_seven_divided_by_three_l495_49525

theorem eight_times_seven_divided_by_three :
  (∃ (a b c : ℕ), a = 5 ∧ b = 6 ∧ c = 7 ∧ a * b = 30 ∧ b * c = 42 ∧ c * 8 = 56) →
  (8 * 7) / 3 = 18 ∧ (8 * 7) % 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_eight_times_seven_divided_by_three_l495_49525


namespace NUMINAMATH_CALUDE_isosceles_triangle_angles_l495_49582

/-- Represents an isosceles triangle with one angle of 50 degrees -/
structure IsoscelesTriangle where
  /-- The measure of the first angle in degrees -/
  angle1 : ℝ
  /-- The measure of the second angle in degrees -/
  angle2 : ℝ
  /-- The measure of the third angle in degrees -/
  angle3 : ℝ
  /-- The sum of all angles is 180 degrees -/
  sum_of_angles : angle1 + angle2 + angle3 = 180
  /-- One angle is 50 degrees -/
  has_50_degree_angle : angle1 = 50 ∨ angle2 = 50 ∨ angle3 = 50
  /-- The triangle is isosceles (two angles are equal) -/
  is_isosceles : (angle1 = angle2) ∨ (angle2 = angle3) ∨ (angle1 = angle3)

/-- Theorem: In an isosceles triangle with one angle of 50°, the other two angles are 50° and 80° -/
theorem isosceles_triangle_angles (t : IsoscelesTriangle) :
  (t.angle1 = 50 ∧ t.angle2 = 50 ∧ t.angle3 = 80) ∨
  (t.angle1 = 50 ∧ t.angle2 = 80 ∧ t.angle3 = 50) ∨
  (t.angle1 = 80 ∧ t.angle2 = 50 ∧ t.angle3 = 50) :=
by sorry


end NUMINAMATH_CALUDE_isosceles_triangle_angles_l495_49582


namespace NUMINAMATH_CALUDE_siblings_height_l495_49502

/-- The total height of 5 siblings -/
def total_height (h1 h2 h3 h4 h5 : ℕ) : ℕ := h1 + h2 + h3 + h4 + h5

/-- Theorem stating the total height of the 5 siblings is 330 inches -/
theorem siblings_height :
  ∃ (h5 : ℕ), 
    total_height 66 66 60 68 h5 = 330 ∧ h5 = 68 + 2 := by
  sorry

end NUMINAMATH_CALUDE_siblings_height_l495_49502


namespace NUMINAMATH_CALUDE_triangle_angle_measures_l495_49565

theorem triangle_angle_measures (A B C : ℝ) 
  (h1 : B - A = 5)
  (h2 : C - B = 20)
  (h3 : A + B + C = 180) : 
  A = 50 ∧ B = 55 ∧ C = 75 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measures_l495_49565


namespace NUMINAMATH_CALUDE_john_and_sarah_money_l495_49561

theorem john_and_sarah_money (john_money : ℚ) (sarah_money : ℚ)
  (h1 : john_money = 5 / 8)
  (h2 : sarah_money = 7 / 16) :
  john_money + sarah_money = 1.0625 := by
sorry

end NUMINAMATH_CALUDE_john_and_sarah_money_l495_49561


namespace NUMINAMATH_CALUDE_similar_triangle_longest_side_l495_49562

theorem similar_triangle_longest_side 
  (a b c : ℝ) 
  (h_triangle : a = 5 ∧ b = 12 ∧ c = 13) 
  (h_perimeter : ∃ k : ℝ, k > 0 ∧ k * (a + b + c) = 150) :
  ∃ s : ℝ, s = 65 ∧ s = max (k * a) (max (k * b) (k * c)) :=
sorry

end NUMINAMATH_CALUDE_similar_triangle_longest_side_l495_49562


namespace NUMINAMATH_CALUDE_concert_drive_distance_l495_49588

/-- Calculates the remaining distance to drive given the total distance and the distance already driven. -/
def remaining_distance (total : ℕ) (driven : ℕ) : ℕ :=
  total - driven

/-- Theorem stating that given a total distance of 78 miles and a distance already driven of 32 miles, 
    the remaining distance to drive is 46 miles. -/
theorem concert_drive_distance : remaining_distance 78 32 = 46 := by
  sorry

end NUMINAMATH_CALUDE_concert_drive_distance_l495_49588


namespace NUMINAMATH_CALUDE_exponentiation_equality_l495_49516

theorem exponentiation_equality : 
  (-2 : ℤ)^3 = -2^3 ∧ 
  (-4 : ℤ)^2 ≠ -4^2 ∧ 
  (-1 : ℤ)^2020 ≠ (-1 : ℤ)^2021 ∧ 
  (2/3 : ℚ)^3 = (2/3 : ℚ)^3 := by sorry

end NUMINAMATH_CALUDE_exponentiation_equality_l495_49516


namespace NUMINAMATH_CALUDE_congruence_solution_l495_49554

theorem congruence_solution : ∃! n : ℤ, 0 ≤ n ∧ n < 23 ∧ -250 ≡ n [ZMOD 23] ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l495_49554


namespace NUMINAMATH_CALUDE_smallest_even_sum_fourteen_is_achievable_l495_49580

def S : Finset Int := {8, -4, 3, 27, 10}

def isValidSum (x y z : Int) : Prop :=
  x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ Even (x + y + z)

theorem smallest_even_sum :
  ∀ x y z, isValidSum x y z → x + y + z ≥ 14 :=
by sorry

theorem fourteen_is_achievable :
  ∃ x y z, isValidSum x y z ∧ x + y + z = 14 :=
by sorry

end NUMINAMATH_CALUDE_smallest_even_sum_fourteen_is_achievable_l495_49580


namespace NUMINAMATH_CALUDE_apple_distribution_l495_49549

/-- The number of apples Adam and Jackie have together -/
def total_adam_jackie : ℕ := 12

/-- The number of additional apples He has compared to Adam and Jackie together -/
def he_additional : ℕ := 9

/-- The number of additional apples Adam has compared to Jackie -/
def adam_additional : ℕ := 8

/-- The number of apples Adam has -/
def adam : ℕ := sorry

/-- The number of apples Jackie has -/
def jackie : ℕ := sorry

/-- The number of apples He has -/
def he : ℕ := sorry

theorem apple_distribution :
  (adam + jackie = total_adam_jackie) ∧
  (he = adam + jackie + he_additional) ∧
  (adam = jackie + adam_additional) →
  he = 21 := by sorry

end NUMINAMATH_CALUDE_apple_distribution_l495_49549


namespace NUMINAMATH_CALUDE_original_number_proof_l495_49521

theorem original_number_proof (x : ℚ) : 
  2 + (1 / x) = 10 / 3 → x = 3 / 4 := by
sorry

end NUMINAMATH_CALUDE_original_number_proof_l495_49521


namespace NUMINAMATH_CALUDE_base_nine_ones_triangular_l495_49583

theorem base_nine_ones_triangular (k : ℕ+) : ∃ n : ℕ, (9^k.val - 1) / 8 = n * (n + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_base_nine_ones_triangular_l495_49583


namespace NUMINAMATH_CALUDE_timmy_candies_l495_49507

theorem timmy_candies : ∃ x : ℕ, 
  (x / 2 - 3) / 2 - 5 = 10 ∧ x = 66 := by
  sorry

end NUMINAMATH_CALUDE_timmy_candies_l495_49507


namespace NUMINAMATH_CALUDE_floor_equation_solution_l495_49514

theorem floor_equation_solution (a b : ℝ) : 
  (∀ x y : ℝ, ⌊a*x + b*y⌋ + ⌊b*x + a*y⌋ = (a + b)*⌊x + y⌋) ↔ 
  ((a = 0 ∧ b = 0) ∨ (a = 1 ∧ b = 1)) :=
sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l495_49514


namespace NUMINAMATH_CALUDE_circular_field_diameter_circular_field_diameter_proof_l495_49553

/-- The diameter of a circular field given the cost of fencing per meter and the total cost -/
theorem circular_field_diameter (cost_per_meter : ℝ) (total_cost : ℝ) : ℝ :=
  let circumference := total_cost / cost_per_meter
  circumference / Real.pi

/-- Proof that the diameter of the circular field is approximately 28 meters -/
theorem circular_field_diameter_proof :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  |circular_field_diameter 1.50 131.95 - 28| < ε :=
sorry

end NUMINAMATH_CALUDE_circular_field_diameter_circular_field_diameter_proof_l495_49553


namespace NUMINAMATH_CALUDE_female_officers_count_l495_49570

/-- Proves the total number of female officers on a police force given certain conditions -/
theorem female_officers_count (total_on_duty : ℕ) (female_on_duty_percent : ℚ) :
  total_on_duty = 152 →
  female_on_duty_percent = 19 / 100 →
  ∃ (total_female : ℕ),
    total_female = 400 ∧
    (total_female : ℚ) * female_on_duty_percent = total_on_duty / 2 :=
by sorry

end NUMINAMATH_CALUDE_female_officers_count_l495_49570


namespace NUMINAMATH_CALUDE_arrow_balance_l495_49546

/-- A polygon with arrows on its sides. -/
structure ArrowPolygon where
  n : ℕ  -- number of sides/vertices
  arrows : Fin n → Bool  -- True if arrow points clockwise, False if counterclockwise

/-- The number of vertices with two incoming arrows. -/
def incoming_two (p : ArrowPolygon) : ℕ := sorry

/-- The number of vertices with two outgoing arrows. -/
def outgoing_two (p : ArrowPolygon) : ℕ := sorry

/-- Theorem: The number of vertices with two incoming arrows equals the number of vertices with two outgoing arrows. -/
theorem arrow_balance (p : ArrowPolygon) : incoming_two p = outgoing_two p := by sorry

end NUMINAMATH_CALUDE_arrow_balance_l495_49546


namespace NUMINAMATH_CALUDE_chord_intersection_diameter_segments_l495_49560

theorem chord_intersection_diameter_segments (r : ℝ) (chord_length : ℝ) : 
  r = 6 → chord_length = 10 → ∃ (s₁ s₂ : ℝ), s₁ = 6 - Real.sqrt 11 ∧ s₂ = 6 + Real.sqrt 11 ∧ s₁ + s₂ = 2 * r :=
by sorry

end NUMINAMATH_CALUDE_chord_intersection_diameter_segments_l495_49560


namespace NUMINAMATH_CALUDE_sum_of_fractions_bounds_l495_49558

theorem sum_of_fractions_bounds (v w x y z : ℝ) (hv : v > 0) (hw : w > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  1 < (v / (v + w)) + (w / (w + x)) + (x / (x + y)) + (y / (y + z)) + (z / (z + v)) ∧
  (v / (v + w)) + (w / (w + x)) + (x / (x + y)) + (y / (y + z)) + (z / (z + v)) < 4 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_fractions_bounds_l495_49558


namespace NUMINAMATH_CALUDE_school_population_l495_49577

theorem school_population (x : ℝ) : 
  (242 = (x / 100) * (50 / 100 * x)) → x = 220 := by
  sorry

end NUMINAMATH_CALUDE_school_population_l495_49577
