import Mathlib

namespace NUMINAMATH_CALUDE_driver_net_pay_rate_driver_net_pay_is_24_l2312_231225

/-- Calculates the net rate of pay per hour for a driver given specific conditions --/
theorem driver_net_pay_rate (travel_time : ℝ) (travel_speed : ℝ) (car_efficiency : ℝ) 
  (pay_rate : ℝ) (gasoline_cost : ℝ) : ℝ :=
  let total_distance := travel_time * travel_speed
  let gasoline_used := total_distance / car_efficiency
  let earnings := pay_rate * total_distance
  let gasoline_expense := gasoline_cost * gasoline_used
  let net_earnings := earnings - gasoline_expense
  let net_pay_rate := net_earnings / travel_time
  net_pay_rate

/-- Proves that the driver's net rate of pay is $24 per hour under given conditions --/
theorem driver_net_pay_is_24 :
  driver_net_pay_rate 3 50 25 0.60 3.00 = 24 := by
  sorry

end NUMINAMATH_CALUDE_driver_net_pay_rate_driver_net_pay_is_24_l2312_231225


namespace NUMINAMATH_CALUDE_parabola_focus_l2312_231286

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop := y = 2 * x^2

/-- The focus of a parabola -/
def focus (p : ℝ × ℝ) (parabola : ℝ → ℝ → Prop) : Prop :=
  ∃ (a : ℝ), a ≠ 0 ∧ 
  ∀ (x y : ℝ), parabola x y ↔ (x - p.1)^2 = 4 * a * (y - p.2)

/-- Theorem: The focus of the parabola y = 2x² is at the point (0, 1/8) -/
theorem parabola_focus :
  focus (0, 1/8) parabola_equation :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_l2312_231286


namespace NUMINAMATH_CALUDE_baker_cakes_problem_l2312_231276

theorem baker_cakes_problem (initial_cakes : ℕ) 
  (h1 : initial_cakes - 78 + 31 = initial_cakes) 
  (h2 : 78 = 31 + 47) : 
  initial_cakes = 109 := by
  sorry

end NUMINAMATH_CALUDE_baker_cakes_problem_l2312_231276


namespace NUMINAMATH_CALUDE_robes_savings_l2312_231229

/-- Calculates Robe's initial savings given the repair costs and remaining savings --/
def initial_savings (repair_fee : ℕ) (corner_light_cost : ℕ) (brake_disk_cost : ℕ) (remaining_savings : ℕ) : ℕ :=
  remaining_savings + repair_fee + corner_light_cost + 2 * brake_disk_cost

theorem robes_savings :
  let repair_fee : ℕ := 10
  let corner_light_cost : ℕ := 2 * repair_fee
  let brake_disk_cost : ℕ := 3 * corner_light_cost
  let remaining_savings : ℕ := 480
  initial_savings repair_fee corner_light_cost brake_disk_cost remaining_savings = 630 := by
  sorry

#eval initial_savings 10 20 60 480

end NUMINAMATH_CALUDE_robes_savings_l2312_231229


namespace NUMINAMATH_CALUDE_radical_product_equals_27_l2312_231258

theorem radical_product_equals_27 :
  let a := 81
  let b := 27
  let c := 9
  (a = 3^4) → (b = 3^3) → (c = 3^2) →
  (a^(1/4) * b^(1/3) * c^(1/2) : ℝ) = 27 := by
  sorry

end NUMINAMATH_CALUDE_radical_product_equals_27_l2312_231258


namespace NUMINAMATH_CALUDE_inverse_sum_theorem_l2312_231213

noncomputable def g (x : ℝ) : ℝ :=
  if x < 15 then x + 5 else 3 * x - 1

theorem inverse_sum_theorem : 
  (Function.invFun g) 10 + (Function.invFun g) 50 = 22 := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_theorem_l2312_231213


namespace NUMINAMATH_CALUDE_sqrt_product_sqrt_three_times_sqrt_two_l2312_231260

theorem sqrt_product (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) :
  Real.sqrt (a * b) = Real.sqrt a * Real.sqrt b := by sorry

theorem sqrt_three_times_sqrt_two :
  Real.sqrt 3 * Real.sqrt 2 = Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_sqrt_product_sqrt_three_times_sqrt_two_l2312_231260


namespace NUMINAMATH_CALUDE_apollonian_circle_l2312_231220

/-- The Apollonian circle theorem -/
theorem apollonian_circle (r : ℝ) (h_r_pos : r > 0) : 
  (∃! P : ℝ × ℝ, (P.1 - 2)^2 + P.2^2 = r^2 ∧ 
    ((P.1 - 3)^2 + P.2^2) = 4 * (P.1^2 + P.2^2)) → r = 1 := by
  sorry

end NUMINAMATH_CALUDE_apollonian_circle_l2312_231220


namespace NUMINAMATH_CALUDE_balls_in_boxes_with_empty_l2312_231290

/-- The number of ways to put n distinguishable balls in k distinguishable boxes -/
def ways_to_put_balls_in_boxes (n : ℕ) (k : ℕ) : ℕ := k^n

/-- The number of ways to put n distinguishable balls in k distinguishable boxes
    with at least one box empty -/
def ways_with_empty_box (n : ℕ) (k : ℕ) : ℕ :=
  ways_to_put_balls_in_boxes n k -
  (Nat.choose k 1) * ways_to_put_balls_in_boxes n (k-1) +
  (Nat.choose k 2) * ways_to_put_balls_in_boxes n (k-2) -
  (Nat.choose k 3) * ways_to_put_balls_in_boxes n (k-3)

/-- Theorem: There are 240 ways to put 5 distinguishable balls in 4 distinguishable boxes
    with at least one box remaining empty -/
theorem balls_in_boxes_with_empty : ways_with_empty_box 5 4 = 240 := by
  sorry

end NUMINAMATH_CALUDE_balls_in_boxes_with_empty_l2312_231290


namespace NUMINAMATH_CALUDE_unique_two_digit_number_with_remainder_one_l2312_231266

theorem unique_two_digit_number_with_remainder_one : 
  ∃! n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ n % 4 = 1 ∧ n % 17 = 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_two_digit_number_with_remainder_one_l2312_231266


namespace NUMINAMATH_CALUDE_joan_driving_speed_l2312_231237

theorem joan_driving_speed 
  (total_distance : ℝ) 
  (total_trip_time : ℝ) 
  (lunch_break : ℝ) 
  (bathroom_break : ℝ) 
  (num_bathroom_breaks : ℕ) :
  total_distance = 480 →
  total_trip_time = 9 →
  lunch_break = 0.5 →
  bathroom_break = 0.25 →
  num_bathroom_breaks = 2 →
  let total_break_time := lunch_break + num_bathroom_breaks * bathroom_break
  let driving_time := total_trip_time - total_break_time
  let speed := total_distance / driving_time
  speed = 60 := by sorry

end NUMINAMATH_CALUDE_joan_driving_speed_l2312_231237


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2312_231257

theorem necessary_but_not_sufficient :
  (∃ a : ℝ, (a < 1 → a ≤ 1) ∧ ¬(a ≤ 1 → a < 1)) ∧
  (∃ x y : ℝ, (x = 1 ∧ y = 0 → x^2 + y^2 = 1) ∧ ¬(x^2 + y^2 = 1 → x = 1 ∧ y = 0)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2312_231257


namespace NUMINAMATH_CALUDE_ping_pong_balls_sold_l2312_231281

/-- Calculates the number of ping pong balls sold in a shop -/
theorem ping_pong_balls_sold
  (initial_baseballs : ℕ)
  (initial_ping_pong : ℕ)
  (baseballs_sold : ℕ)
  (total_left : ℕ)
  (h1 : initial_baseballs = 2754)
  (h2 : initial_ping_pong = 1938)
  (h3 : baseballs_sold = 1095)
  (h4 : total_left = 3021)
  (h5 : total_left = initial_baseballs + initial_ping_pong - baseballs_sold - (initial_ping_pong - ping_pong_sold))
  : ping_pong_sold = 576 :=
by
  sorry

#check ping_pong_balls_sold

end NUMINAMATH_CALUDE_ping_pong_balls_sold_l2312_231281


namespace NUMINAMATH_CALUDE_product_xyz_l2312_231228

theorem product_xyz (x y z : ℝ) (h1 : x + 2 / y = 2) (h2 : y + 2 / z = 2) (h3 : z ≠ 0) : x * y * z = -4 := by
  sorry

end NUMINAMATH_CALUDE_product_xyz_l2312_231228


namespace NUMINAMATH_CALUDE_max_candy_leftover_l2312_231221

theorem max_candy_leftover (x : ℕ) : ∃ (q r : ℕ), x = 7 * q + r ∧ r < 7 ∧ r ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_max_candy_leftover_l2312_231221


namespace NUMINAMATH_CALUDE_triangle_ABC_properties_l2312_231209

-- Define the triangle ABC
def A : ℝ × ℝ := (4, 0)
def B : ℝ × ℝ := (8, 10)
def C : ℝ × ℝ := (0, 6)

-- Define the altitude from C to AB
def altitude_equation (x y : ℝ) : Prop :=
  2 * x + 5 * y - 30 = 0

-- Define the midline parallel to AC
def midline_equation (x : ℝ) : Prop :=
  x = 4

-- Theorem statement
theorem triangle_ABC_properties :
  -- The altitude from C to AB satisfies the equation
  (∀ x y : ℝ, altitude_equation x y ↔ 
    (x - C.1) * (B.2 - A.2) = (y - C.2) * (B.1 - A.1)) ∧
  -- The midline parallel to AC satisfies the equation
  (∀ x : ℝ, midline_equation x ↔ 
    x = (B.1 + C.1) / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_ABC_properties_l2312_231209


namespace NUMINAMATH_CALUDE_inequality_addition_l2312_231210

theorem inequality_addition {a b c d : ℝ} (h1 : a > b) (h2 : c > d) : a + c > b + d := by
  sorry

end NUMINAMATH_CALUDE_inequality_addition_l2312_231210


namespace NUMINAMATH_CALUDE_box_surface_area_l2312_231275

/-- Represents the dimensions of a rectangular sheet. -/
structure SheetDimensions where
  length : ℕ
  width : ℕ

/-- Represents the size of the square cut from each corner. -/
def CornerCutSize : ℕ := 4

/-- Calculates the surface area of the interior of the box formed by folding a rectangular sheet
    with squares cut from each corner. -/
def interiorSurfaceArea (sheet : SheetDimensions) : ℕ :=
  sheet.length * sheet.width - 4 * (CornerCutSize * CornerCutSize)

/-- Theorem stating that the surface area of the interior of the box is 936 square units. -/
theorem box_surface_area :
  interiorSurfaceArea ⟨25, 40⟩ = 936 := by sorry

end NUMINAMATH_CALUDE_box_surface_area_l2312_231275


namespace NUMINAMATH_CALUDE_mans_walking_rate_l2312_231293

/-- The problem of finding a man's initial walking rate given certain conditions. -/
theorem mans_walking_rate (distance : ℝ) (early_speed : ℝ) (early_time : ℝ) (late_time : ℝ) :
  distance = 6.000000000000001 →
  early_speed = 6 →
  early_time = 5 / 60 →
  late_time = 7 / 60 →
  ∃ (initial_speed : ℝ),
    initial_speed = distance / (distance / early_speed + early_time + late_time) ∧
    initial_speed = 5 := by
  sorry

#eval 6.000000000000001 / (6.000000000000001 / 6 + 5 / 60 + 7 / 60)

end NUMINAMATH_CALUDE_mans_walking_rate_l2312_231293


namespace NUMINAMATH_CALUDE_litter_patrol_problem_l2312_231224

/-- The Litter Patrol Problem -/
theorem litter_patrol_problem (total_litter : ℕ) (aluminum_cans : ℕ) (glass_bottles : ℕ) :
  total_litter = 18 →
  aluminum_cans = 8 →
  total_litter = aluminum_cans + glass_bottles →
  glass_bottles = 10 := by
sorry

end NUMINAMATH_CALUDE_litter_patrol_problem_l2312_231224


namespace NUMINAMATH_CALUDE_tangency_distance_value_l2312_231271

/-- Configuration of four circles where three small circles of radius 2 are externally
    tangent to each other and internally tangent to a larger circle -/
structure CircleConfiguration where
  -- Radius of each small circle
  small_radius : ℝ
  -- Center of the large circle
  large_center : ℝ × ℝ
  -- Centers of the three small circles
  small_centers : Fin 3 → ℝ × ℝ
  -- The three small circles are externally tangent to each other
  small_circles_tangent : ∀ (i j : Fin 3), i ≠ j →
    ‖small_centers i - small_centers j‖ = 2 * small_radius
  -- The three small circles are internally tangent to the large circle
  large_circle_tangent : ∀ (i : Fin 3),
    ‖large_center - small_centers i‖ = ‖large_center - small_centers 0‖

/-- The distance from the center of the large circle to the point of tangency
    on one of the small circles in the given configuration -/
def tangency_distance (config : CircleConfiguration) : ℝ :=
  ‖config.large_center - config.small_centers 0‖ - config.small_radius

/-- Theorem stating that the tangency distance is equal to 2√3 - 2 -/
theorem tangency_distance_value (config : CircleConfiguration) 
    (h : config.small_radius = 2) : 
    tangency_distance config = 2 * Real.sqrt 3 - 2 := by
  sorry

end NUMINAMATH_CALUDE_tangency_distance_value_l2312_231271


namespace NUMINAMATH_CALUDE_negation_of_forall_positive_l2312_231297

theorem negation_of_forall_positive (R : Type) [OrderedRing R] :
  (¬ (∀ x : R, x^2 + x + 1 > 0)) ↔ (∃ x : R, x^2 + x + 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_forall_positive_l2312_231297


namespace NUMINAMATH_CALUDE_erdos_theorem_l2312_231295

/-- For any integer k, there exists a graph H with girth greater than k and chromatic number greater than k. -/
theorem erdos_theorem (k : ℕ) : ∃ H : SimpleGraph ℕ, SimpleGraph.girth H > k ∧ SimpleGraph.chromaticNumber H > k := by
  sorry

end NUMINAMATH_CALUDE_erdos_theorem_l2312_231295


namespace NUMINAMATH_CALUDE_abs_neg_five_eq_five_l2312_231216

theorem abs_neg_five_eq_five : |(-5 : ℤ)| = 5 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_five_eq_five_l2312_231216


namespace NUMINAMATH_CALUDE_orthogonal_vectors_k_value_l2312_231269

/-- Given vector a = (3, -4), a + 2b = (k+1, k-4), and a is orthogonal to b, prove that k = -6 -/
theorem orthogonal_vectors_k_value (k : ℝ) (b : ℝ × ℝ) : 
  let a : ℝ × ℝ := (3, -4)
  (a.1 + 2 * b.1 = k + 1 ∧ a.2 + 2 * b.2 = k - 4) → 
  (a.1 * b.1 + a.2 * b.2 = 0) → 
  k = -6 := by
  sorry

end NUMINAMATH_CALUDE_orthogonal_vectors_k_value_l2312_231269


namespace NUMINAMATH_CALUDE_congruent_mod_divisor_congruent_mod_polynomial_l2312_231279

/-- Definition of congruence modulo m -/
def congruent_mod (a b m : ℤ) : Prop :=
  ∃ k : ℤ, a - b = m * k

/-- Statement 1 -/
theorem congruent_mod_divisor (a b m d : ℤ) (hm : 0 < m) (hd : 0 < d) (hdiv : d ∣ m) 
    (h : congruent_mod a b m) : congruent_mod a b d := by
  sorry

/-- Definition of the polynomial f(x) = x³ - 2x + 5 -/
def f (x : ℤ) : ℤ := x^3 - 2*x + 5

/-- Statement 4 -/
theorem congruent_mod_polynomial (a b m : ℤ) (hm : 0 < m) 
    (h : congruent_mod a b m) : congruent_mod (f a) (f b) m := by
  sorry

end NUMINAMATH_CALUDE_congruent_mod_divisor_congruent_mod_polynomial_l2312_231279


namespace NUMINAMATH_CALUDE_angle4_value_l2312_231244

-- Define the angles
variable (angle1 angle2 angle3 angle4 angle5 angle6 : ℝ)

-- Define the given conditions
axiom sum_angles_1_2 : angle1 + angle2 = 180
axiom equal_angles_3_4 : angle3 = angle4
axiom new_angle1 : angle1 = 85
axiom new_angle5 : angle5 = 45
axiom triangle_sum : angle1 + angle5 + angle6 = 180

-- Define the theorem to prove
theorem angle4_value : angle4 = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_angle4_value_l2312_231244


namespace NUMINAMATH_CALUDE_sachins_age_l2312_231280

/-- Proves that Sachin's age is 38.5 years given the conditions -/
theorem sachins_age (sachin rahul : ℝ) 
  (h1 : sachin = rahul + 7)
  (h2 : sachin / rahul = 11 / 9) : 
  sachin = 38.5 := by
  sorry

end NUMINAMATH_CALUDE_sachins_age_l2312_231280


namespace NUMINAMATH_CALUDE_zero_score_probability_l2312_231246

def num_balls : ℕ := 6
def num_red : ℕ := 1
def num_yellow : ℕ := 2
def num_blue : ℕ := 3
def num_draws : ℕ := 3

def score_red : ℤ := 1
def score_yellow : ℤ := 0
def score_blue : ℤ := -1

def prob_zero_score : ℚ := 11 / 54

theorem zero_score_probability :
  (num_balls = num_red + num_yellow + num_blue) →
  (prob_zero_score = (num_yellow^num_draws + num_red * num_yellow * num_blue * 6) / num_balls^num_draws) :=
by sorry

end NUMINAMATH_CALUDE_zero_score_probability_l2312_231246


namespace NUMINAMATH_CALUDE_quadratic_solution_implies_value_l2312_231239

theorem quadratic_solution_implies_value (a : ℝ) :
  (2^2 - 3*2 + a = 0) → (2*a - 1 = 3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_implies_value_l2312_231239


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2312_231296

-- 1. Prove that 522 - 112 ÷ 4 = 494
theorem problem_1 : 522 - 112 / 4 = 494 := by
  sorry

-- 2. Prove that (603 - 587) × 80 = 1280
theorem problem_2 : (603 - 587) * 80 = 1280 := by
  sorry

-- 3. Prove that 26 × 18 + 463 = 931
theorem problem_3 : 26 * 18 + 463 = 931 := by
  sorry

-- 4. Prove that 400 × (45 ÷ 9) = 2000
theorem problem_4 : 400 * (45 / 9) = 2000 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2312_231296


namespace NUMINAMATH_CALUDE_rectangle_max_area_l2312_231200

theorem rectangle_max_area (d : ℝ) (h : d > 0) :
  ∀ (w h : ℝ), w > 0 → h > 0 → w^2 + h^2 = d^2 →
  w * h ≤ (d^2) / 2 ∧ (w * h = (d^2) / 2 ↔ w = h) :=
sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l2312_231200


namespace NUMINAMATH_CALUDE_parabola_distance_theorem_l2312_231202

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus
def focus : ℝ × ℝ := (2, 0)

-- Define the directrix
def directrix (x : ℝ) : Prop := x = -2

-- Define the condition for Q being on the parabola
def Q_on_parabola (Q : ℝ × ℝ) : Prop := parabola Q.1 Q.2

-- Define the vector relation
def vector_relation (P Q : ℝ × ℝ) : Prop :=
  (Q.1 - focus.1, Q.2 - focus.2) = (-4 * (focus.1 - P.1), -4 * (focus.2 - P.2))

-- The theorem to prove
theorem parabola_distance_theorem (P Q : ℝ × ℝ) :
  directrix P.1 →
  Q_on_parabola Q →
  vector_relation P Q →
  Real.sqrt ((Q.1 - focus.1)^2 + (Q.2 - focus.2)^2) = 20 :=
by sorry

end NUMINAMATH_CALUDE_parabola_distance_theorem_l2312_231202


namespace NUMINAMATH_CALUDE_cricket_team_average_age_l2312_231233

/-- Represents the average age of a cricket team --/
def average_age : ℝ := 23

/-- Represents the number of team members --/
def team_size : ℕ := 11

/-- Represents the age of the captain --/
def captain_age : ℕ := 25

/-- Represents the age of the wicket keeper --/
def wicket_keeper_age : ℕ := captain_age + 3

/-- Represents the age of the vice-captain --/
def vice_captain_age : ℕ := wicket_keeper_age - 4

/-- Theorem stating that the average age of the cricket team is 23 years --/
theorem cricket_team_average_age :
  average_age * team_size =
    captain_age + wicket_keeper_age + vice_captain_age +
    (team_size - 3) * (average_age - 1) := by
  sorry

#check cricket_team_average_age

end NUMINAMATH_CALUDE_cricket_team_average_age_l2312_231233


namespace NUMINAMATH_CALUDE_min_draw_count_correct_l2312_231222

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  white : Nat
  black : Nat

/-- The minimum number of balls to draw to guarantee the condition -/
def minDrawCount : Nat := 82

/-- The theorem stating the minimum number of balls to draw -/
theorem min_draw_count_correct (box : BallCounts) 
  (h1 : box.red = 30)
  (h2 : box.green = 22)
  (h3 : box.yellow = 20)
  (h4 : box.blue = 15)
  (h5 : box.white = 12)
  (h6 : box.black = 10) :
  minDrawCount = 82 ∧ 
  (∀ n : Nat, n < 82 → 
    ∃ draw : BallCounts, 
      draw.red + draw.green + draw.yellow + draw.blue + draw.white + draw.black = n ∧
      draw.red ≤ box.red ∧ draw.green ≤ box.green ∧ draw.yellow ≤ box.yellow ∧ 
      draw.blue ≤ box.blue ∧ draw.white ≤ box.white ∧ draw.black ≤ box.black ∧
      draw.white ≤ 12 ∧
      draw.red < 16 ∧ draw.green < 16 ∧ draw.yellow < 16 ∧ draw.blue < 16 ∧ draw.black < 16) ∧
  (∃ draw : BallCounts,
    draw.red + draw.green + draw.yellow + draw.blue + draw.white + draw.black = 82 ∧
    draw.red ≤ box.red ∧ draw.green ≤ box.green ∧ draw.yellow ≤ box.yellow ∧ 
    draw.blue ≤ box.blue ∧ draw.white ≤ box.white ∧ draw.black ≤ box.black ∧
    draw.white ≤ 12 ∧
    (draw.red ≥ 16 ∨ draw.green ≥ 16 ∨ draw.yellow ≥ 16 ∨ draw.blue ≥ 16 ∨ draw.black ≥ 16)) :=
by sorry

end NUMINAMATH_CALUDE_min_draw_count_correct_l2312_231222


namespace NUMINAMATH_CALUDE_gavins_dreams_l2312_231245

/-- The number of dreams Gavin has every day this year -/
def dreams_per_day : ℕ := sorry

/-- The number of days in a year -/
def days_in_year : ℕ := 365

/-- The total number of dreams in two years -/
def total_dreams : ℕ := 4380

theorem gavins_dreams : 
  dreams_per_day * days_in_year + 2 * (dreams_per_day * days_in_year) = total_dreams ∧ 
  dreams_per_day = 4 := by sorry

end NUMINAMATH_CALUDE_gavins_dreams_l2312_231245


namespace NUMINAMATH_CALUDE_parallel_equal_sides_is_parallelogram_l2312_231223

/-- A quadrilateral in 2D space --/
structure Quadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- Definition of parallel sides in a quadrilateral --/
def has_parallel_sides (q : Quadrilateral) : Prop :=
  (q.A.1 - q.B.1) / (q.A.2 - q.B.2) = (q.D.1 - q.C.1) / (q.D.2 - q.C.2) ∧
  (q.A.1 - q.D.1) / (q.A.2 - q.D.2) = (q.B.1 - q.C.1) / (q.B.2 - q.C.2)

/-- Definition of equal sides in a quadrilateral --/
def has_equal_sides (q : Quadrilateral) : Prop :=
  (q.A.1 - q.B.1)^2 + (q.A.2 - q.B.2)^2 = (q.B.1 - q.C.1)^2 + (q.B.2 - q.C.2)^2 ∧
  (q.B.1 - q.C.1)^2 + (q.B.2 - q.C.2)^2 = (q.C.1 - q.D.1)^2 + (q.C.2 - q.D.2)^2 ∧
  (q.C.1 - q.D.1)^2 + (q.C.2 - q.D.2)^2 = (q.D.1 - q.A.1)^2 + (q.D.2 - q.A.2)^2

/-- Definition of a parallelogram --/
def is_parallelogram (q : Quadrilateral) : Prop :=
  has_parallel_sides q

/-- Theorem: A quadrilateral with parallel and equal sides is a parallelogram --/
theorem parallel_equal_sides_is_parallelogram (q : Quadrilateral) :
  has_parallel_sides q → has_equal_sides q → is_parallelogram q :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_equal_sides_is_parallelogram_l2312_231223


namespace NUMINAMATH_CALUDE_cid_earnings_l2312_231299

def oil_change_price : ℕ := 20
def repair_price : ℕ := 30
def car_wash_price : ℕ := 5

def oil_changes_performed : ℕ := 5
def repairs_performed : ℕ := 10
def car_washes_performed : ℕ := 15

def total_earnings : ℕ := 
  oil_change_price * oil_changes_performed + 
  repair_price * repairs_performed + 
  car_wash_price * car_washes_performed

theorem cid_earnings : total_earnings = 475 := by
  sorry

end NUMINAMATH_CALUDE_cid_earnings_l2312_231299


namespace NUMINAMATH_CALUDE_integer_root_implies_specific_m_l2312_231274

/-- Defines a quadratic equation with coefficient m -/
def quadratic_equation (m : ℤ) (x : ℤ) : ℤ := m * x^2 + 2*(m-5)*x + (m-4)

/-- Checks if the equation has an integer root -/
def has_integer_root (m : ℤ) : Prop := ∃ x : ℤ, quadratic_equation m x = 0

/-- The main theorem to be proved -/
theorem integer_root_implies_specific_m :
  ∀ m : ℤ, has_integer_root m → m = -4 ∨ m = 4 ∨ m = -16 := by sorry

end NUMINAMATH_CALUDE_integer_root_implies_specific_m_l2312_231274


namespace NUMINAMATH_CALUDE_painted_cells_theorem_l2312_231203

theorem painted_cells_theorem (k l : ℕ) : 
  k * l = 74 →
  ((2 * k + 1) * (2 * l + 1) - k * l = 373) ∨
  ((2 * k + 1) * (2 * l + 1) - k * l = 301) := by
sorry

end NUMINAMATH_CALUDE_painted_cells_theorem_l2312_231203


namespace NUMINAMATH_CALUDE_area_of_ABHFGD_l2312_231235

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square -/
structure Square where
  a : Point
  b : Point
  c : Point
  d : Point

/-- Calculates the area of a square -/
def squareArea (s : Square) : ℝ := sorry

/-- Checks if a point is the midpoint of two other points -/
def isMidpoint (m : Point) (p1 : Point) (p2 : Point) : Prop := sorry

/-- Checks if a point divides a line segment in a given ratio -/
def divideSegment (p : Point) (a : Point) (b : Point) (ratio : ℝ) : Prop := sorry

/-- Calculates the area of a polygon given its vertices -/
def polygonArea (vertices : List Point) : ℝ := sorry

theorem area_of_ABHFGD (a b c d e f g h : Point) :
  let abcd := Square.mk a b c d
  let efgd := Square.mk e f g d
  squareArea abcd = 25 ∧
  squareArea efgd = 25 ∧
  isMidpoint h e f ∧
  divideSegment h b c (1/3) →
  abs (polygonArea [a, b, h, f, g, d] - 27.09) < 0.01 := by sorry

end NUMINAMATH_CALUDE_area_of_ABHFGD_l2312_231235


namespace NUMINAMATH_CALUDE_quadratic_maximum_l2312_231249

/-- The quadratic function we're analyzing -/
def f (x : ℝ) : ℝ := -2 * x^2 - 8 * x + 10

/-- The point where the maximum occurs -/
def x_max : ℝ := -2

theorem quadratic_maximum :
  ∀ x : ℝ, f x ≤ f x_max :=
sorry

end NUMINAMATH_CALUDE_quadratic_maximum_l2312_231249


namespace NUMINAMATH_CALUDE_watermelon_juice_percentage_l2312_231252

def total_volume : ℝ := 120
def orange_juice_percentage : ℝ := 15
def grape_juice_volume : ℝ := 30

theorem watermelon_juice_percentage :
  let orange_juice_volume := total_volume * (orange_juice_percentage / 100)
  let watermelon_juice_volume := total_volume - orange_juice_volume - grape_juice_volume
  (watermelon_juice_volume / total_volume) * 100 = 60 := by
sorry

end NUMINAMATH_CALUDE_watermelon_juice_percentage_l2312_231252


namespace NUMINAMATH_CALUDE_dave_breaks_two_strings_per_night_l2312_231247

def shows_per_week : ℕ := 6
def total_weeks : ℕ := 12
def total_strings : ℕ := 144

theorem dave_breaks_two_strings_per_night :
  (total_strings : ℚ) / (shows_per_week * total_weeks) = 2 := by
  sorry

end NUMINAMATH_CALUDE_dave_breaks_two_strings_per_night_l2312_231247


namespace NUMINAMATH_CALUDE_root_product_theorem_l2312_231243

theorem root_product_theorem (y₁ y₂ y₃ y₄ y₅ : ℂ) : 
  (y₁^5 - y₁^3 + 1 = 0) → 
  (y₂^5 - y₂^3 + 1 = 0) → 
  (y₃^5 - y₃^3 + 1 = 0) → 
  (y₄^5 - y₄^3 + 1 = 0) → 
  (y₅^5 - y₅^3 + 1 = 0) → 
  ((y₁^2 - 3) * (y₂^2 - 3) * (y₃^2 - 3) * (y₄^2 - 3) * (y₅^2 - 3) = 22) := by
sorry

end NUMINAMATH_CALUDE_root_product_theorem_l2312_231243


namespace NUMINAMATH_CALUDE_certain_number_problem_l2312_231212

theorem certain_number_problem (x : ℝ) : 4 * (3 * x / 5 - 220) = 320 → x = 500 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l2312_231212


namespace NUMINAMATH_CALUDE_john_age_proof_l2312_231207

/-- John's current age -/
def john_age : ℕ := 18

/-- Proposition: John's current age satisfies the given condition -/
theorem john_age_proof : 
  (john_age - 5 : ℤ) = (john_age + 8 : ℤ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_john_age_proof_l2312_231207


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l2312_231270

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem geometric_sequence_third_term
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_a1 : a 1 = 1)
  (h_a5 : a 5 = 16) :
  a 3 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l2312_231270


namespace NUMINAMATH_CALUDE_point_in_first_quadrant_l2312_231250

/-- A proportional function where y increases as x increases -/
structure IncreasingProportionalFunction where
  k : ℝ
  increasing : ∀ x₁ x₂, x₁ < x₂ → k * x₁ < k * x₂

/-- The point (√3, k) is in the first quadrant for an increasing proportional function -/
theorem point_in_first_quadrant (f : IncreasingProportionalFunction) :
  f.k > 0 ∧ Real.sqrt 3 > 0 :=
by sorry

end NUMINAMATH_CALUDE_point_in_first_quadrant_l2312_231250


namespace NUMINAMATH_CALUDE_isosceles_obtuse_triangle_smallest_angle_l2312_231205

/-- 
Given an isosceles, obtuse triangle where the largest angle is 20% larger than 60 degrees,
prove that the measure of each of the two smallest angles is 54 degrees.
-/
theorem isosceles_obtuse_triangle_smallest_angle 
  (α β γ : ℝ) 
  (isosceles : α = β)
  (obtuse : γ > 90)
  (largest_angle : γ = 60 * (1 + 0.2))
  (angle_sum : α + β + γ = 180) :
  α = 54 := by
sorry

end NUMINAMATH_CALUDE_isosceles_obtuse_triangle_smallest_angle_l2312_231205


namespace NUMINAMATH_CALUDE_valleyball_league_members_l2312_231288

/-- The cost of a pair of socks in dollars -/
def sock_cost : ℕ := 6

/-- The cost of a T-shirt in dollars -/
def tshirt_cost : ℕ := sock_cost + 7

/-- The cost of a cap in dollars -/
def cap_cost : ℕ := tshirt_cost

/-- The cost of equipment for home games per member in dollars -/
def home_cost : ℕ := sock_cost + tshirt_cost

/-- The cost of equipment for away games per member in dollars -/
def away_cost : ℕ := sock_cost + tshirt_cost + cap_cost

/-- The total cost of equipment per member in dollars -/
def member_cost : ℕ := home_cost + away_cost

/-- The total cost of equipment for all members in dollars -/
def total_cost : ℕ := 4324

theorem valleyball_league_members : 
  ∃ n : ℕ, n * member_cost ≤ total_cost ∧ total_cost < (n + 1) * member_cost ∧ n = 85 := by
  sorry

end NUMINAMATH_CALUDE_valleyball_league_members_l2312_231288


namespace NUMINAMATH_CALUDE_unique_fraction_representation_l2312_231254

theorem unique_fraction_representation (n : ℕ+) :
  ∃! (a b : ℚ), a > 0 ∧ b > 0 ∧ 
  ∃ (k m : ℤ), a = k / n ∧ b = m / (n + 1) ∧
  (2 * n + 1 : ℚ) / (n * (n + 1)) = a + b :=
by sorry

end NUMINAMATH_CALUDE_unique_fraction_representation_l2312_231254


namespace NUMINAMATH_CALUDE_ellipse_equation_l2312_231256

theorem ellipse_equation (A B : ℝ × ℝ) (h1 : A = (0, 5/3)) (h2 : B = (1, 1)) :
  ∃ (m n : ℝ), m > 0 ∧ n > 0 ∧ m ≠ n ∧
  (∀ (x y : ℝ), m * x^2 + n * y^2 = 1 ↔ 16 * x^2 + 9 * y^2 = 225) :=
sorry


end NUMINAMATH_CALUDE_ellipse_equation_l2312_231256


namespace NUMINAMATH_CALUDE_sin_squared_plus_sin_minus_two_range_l2312_231204

theorem sin_squared_plus_sin_minus_two_range :
  ∀ x : ℝ, -9/4 ≤ Real.sin x ^ 2 + Real.sin x - 2 ∧
  (∃ x : ℝ, Real.sin x ^ 2 + Real.sin x - 2 = -9/4) ∧
  (∃ x : ℝ, Real.sin x ^ 2 + Real.sin x - 2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_sin_squared_plus_sin_minus_two_range_l2312_231204


namespace NUMINAMATH_CALUDE_fewer_men_than_women_l2312_231272

theorem fewer_men_than_women (total : ℕ) (men : ℕ) (h1 : total = 180) (h2 : men = 80) (h3 : men < total - men) :
  total - men - men = 20 := by
  sorry

end NUMINAMATH_CALUDE_fewer_men_than_women_l2312_231272


namespace NUMINAMATH_CALUDE_business_class_seats_count_l2312_231218

/-- A small airplane with first, business, and economy class seating. -/
structure Airplane where
  first_class_seats : ℕ
  business_class_seats : ℕ
  economy_class_seats : ℕ

/-- Theorem stating the number of business class seats in the airplane. -/
theorem business_class_seats_count (a : Airplane) 
  (h1 : a.first_class_seats = 10)
  (h2 : a.economy_class_seats = 50)
  (h3 : a.economy_class_seats / 2 = a.first_class_seats + (a.business_class_seats - 8))
  (h4 : a.first_class_seats - 7 = 3) : 
  a.business_class_seats = 30 := by
  sorry


end NUMINAMATH_CALUDE_business_class_seats_count_l2312_231218


namespace NUMINAMATH_CALUDE_gcd_f_x_l2312_231287

def f (x : ℤ) : ℤ := (3*x+4)*(7*x+1)*(13*x+6)*(2*x+9)

theorem gcd_f_x (x : ℤ) (h : ∃ k : ℤ, x = 15336 * k) : 
  Nat.gcd (Int.natAbs (f x)) (Int.natAbs x) = 216 := by
  sorry

end NUMINAMATH_CALUDE_gcd_f_x_l2312_231287


namespace NUMINAMATH_CALUDE_subcommittee_formation_count_l2312_231268

/-- The number of ways to choose k items from n items -/
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The total number of Republicans in the committee -/
def total_republicans : ℕ := 10

/-- The total number of Democrats in the committee -/
def total_democrats : ℕ := 8

/-- The number of Republicans in the subcommittee -/
def subcommittee_republicans : ℕ := 4

/-- The number of Democrats in the subcommittee -/
def subcommittee_democrats : ℕ := 3

/-- The number of ways to form the subcommittee -/
def subcommittee_combinations : ℕ := 
  (binomial total_republicans subcommittee_republicans) * 
  (binomial total_democrats subcommittee_democrats)

theorem subcommittee_formation_count : subcommittee_combinations = 11760 := by
  sorry

end NUMINAMATH_CALUDE_subcommittee_formation_count_l2312_231268


namespace NUMINAMATH_CALUDE_q_gt_one_neither_sufficient_nor_necessary_l2312_231283

/-- A geometric sequence with common ratio q -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

/-- An increasing sequence -/
def IncreasingSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

/-- Theorem stating that "q > 1" is neither sufficient nor necessary for a geometric sequence to be increasing -/
theorem q_gt_one_neither_sufficient_nor_necessary :
  (∃ (a : ℕ → ℝ) (q : ℝ), q > 1 ∧ GeometricSequence a q ∧ ¬IncreasingSequence a) ∧
  (∃ (a : ℕ → ℝ) (q : ℝ), q ≤ 1 ∧ GeometricSequence a q ∧ IncreasingSequence a) :=
sorry

end NUMINAMATH_CALUDE_q_gt_one_neither_sufficient_nor_necessary_l2312_231283


namespace NUMINAMATH_CALUDE_largest_coin_distribution_exists_largest_distribution_l2312_231201

theorem largest_coin_distribution (n : ℕ) : n < 150 → n % 15 = 3 → n ≤ 138 := by
  sorry

theorem exists_largest_distribution : ∃ n : ℕ, n < 150 ∧ n % 15 = 3 ∧ n = 138 := by
  sorry

end NUMINAMATH_CALUDE_largest_coin_distribution_exists_largest_distribution_l2312_231201


namespace NUMINAMATH_CALUDE_port_vessel_count_port_vessel_count_proof_l2312_231255

theorem port_vessel_count : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun cruise_ships cargo_ships sailboats fishing_boats =>
    cruise_ships = 4 ∧
    cargo_ships = 2 * cruise_ships ∧
    sailboats = cargo_ships + 6 ∧
    sailboats = 7 * fishing_boats →
    cruise_ships + cargo_ships + sailboats + fishing_boats = 28

/-- Proof of the theorem -/
theorem port_vessel_count_proof : port_vessel_count 4 8 14 2 := by
  sorry

end NUMINAMATH_CALUDE_port_vessel_count_port_vessel_count_proof_l2312_231255


namespace NUMINAMATH_CALUDE_unique_polynomial_composition_l2312_231230

-- Define the polynomial P(x) = a x^2 + b x + c
def P (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define a general n-degree polynomial
def NPolynomial (n : ℕ) := ℝ → ℝ

theorem unique_polynomial_composition (a b c : ℝ) (ha : a ≠ 0) (n : ℕ) :
  ∃! Q : NPolynomial n, ∀ x : ℝ, Q (P a b c x) = P a b c (Q x) :=
sorry

end NUMINAMATH_CALUDE_unique_polynomial_composition_l2312_231230


namespace NUMINAMATH_CALUDE_candidate_vote_percentage_l2312_231291

theorem candidate_vote_percentage 
  (total_votes : ℕ) 
  (invalid_percentage : ℚ) 
  (candidate_valid_votes : ℕ) 
  (h1 : total_votes = 560000) 
  (h2 : invalid_percentage = 15 / 100) 
  (h3 : candidate_valid_votes = 357000) : 
  (candidate_valid_votes : ℚ) / ((1 - invalid_percentage) * total_votes) = 75 / 100 := by
sorry

end NUMINAMATH_CALUDE_candidate_vote_percentage_l2312_231291


namespace NUMINAMATH_CALUDE_lowest_score_within_two_std_dev_l2312_231278

/-- Given a mean score and standard deviation, calculate the lowest score within a certain number of standard deviations from the mean. -/
def lowest_score (mean : ℝ) (std_dev : ℝ) (num_std_dev : ℝ) : ℝ :=
  mean - num_std_dev * std_dev

/-- Theorem stating that for a mean of 60 and standard deviation of 10, the lowest score within 2 standard deviations is 40. -/
theorem lowest_score_within_two_std_dev :
  lowest_score 60 10 2 = 40 := by
  sorry

#eval lowest_score 60 10 2

end NUMINAMATH_CALUDE_lowest_score_within_two_std_dev_l2312_231278


namespace NUMINAMATH_CALUDE_binomial_coefficient_geometric_mean_l2312_231273

theorem binomial_coefficient_geometric_mean (a : ℚ) : 
  (∃ (k : ℕ), k = 7 ∧ 
    (Nat.choose k 4 * a^3)^2 = (Nat.choose k 5 * a^2) * (Nat.choose k 2 * a^5)) ↔ 
  a = 25 / 9 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_geometric_mean_l2312_231273


namespace NUMINAMATH_CALUDE_football_players_l2312_231240

theorem football_players (total : ℕ) (cricket : ℕ) (neither : ℕ) (both : ℕ)
  (h1 : total = 460)
  (h2 : cricket = 175)
  (h3 : neither = 50)
  (h4 : both = 90) :
  total - neither - cricket + both = 325 :=
by sorry

end NUMINAMATH_CALUDE_football_players_l2312_231240


namespace NUMINAMATH_CALUDE_pairwise_sums_distinct_digits_impossible_l2312_231298

theorem pairwise_sums_distinct_digits_impossible :
  ¬ ∃ (a b c d e : ℕ),
    let sums := [a + b, a + c, a + d, a + e, b + c, b + d, b + e, c + d, c + e, d + e]
    ∀ (i j : Fin 10), i ≠ j → sums[i] % 10 ≠ sums[j] % 10 := by
  sorry

#check pairwise_sums_distinct_digits_impossible

end NUMINAMATH_CALUDE_pairwise_sums_distinct_digits_impossible_l2312_231298


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l2312_231262

-- Define the universal set U
def U : Set ℤ := {x : ℤ | x^2 - 5*x - 6 < 0}

-- Define set A
def A : Set ℤ := {x : ℤ | -1 < x ∧ x ≤ 2}

-- Define set B
def B : Set ℤ := {2, 3, 5}

-- Theorem statement
theorem complement_A_intersect_B : (U \ A) ∩ B = {3, 5} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l2312_231262


namespace NUMINAMATH_CALUDE_unique_angle_with_same_tangent_l2312_231208

theorem unique_angle_with_same_tangent :
  ∃! (n : ℤ), -180 < n ∧ n < 180 ∧ Real.tan (n * π / 180) = Real.tan (1230 * π / 180) ∧ n = 150 := by
  sorry

end NUMINAMATH_CALUDE_unique_angle_with_same_tangent_l2312_231208


namespace NUMINAMATH_CALUDE_angie_necessities_contribution_l2312_231248

def salary : ℕ := 80
def taxes : ℕ := 20
def leftover : ℕ := 18

theorem angie_necessities_contribution :
  salary - taxes - leftover = 42 := by sorry

end NUMINAMATH_CALUDE_angie_necessities_contribution_l2312_231248


namespace NUMINAMATH_CALUDE_not_necessarily_right_triangle_l2312_231263

theorem not_necessarily_right_triangle (A B C : ℝ) (h : A / B = 3 / 4 ∧ B / C = 4 / 5) :
  ¬ (A = 90 ∨ B = 90 ∨ C = 90) :=
sorry

end NUMINAMATH_CALUDE_not_necessarily_right_triangle_l2312_231263


namespace NUMINAMATH_CALUDE_seating_arrangement_count_l2312_231242

-- Define the number of people
def num_people : ℕ := 10

-- Define the number of seats in each row
def seats_per_row : ℕ := 5

-- Define a function to calculate the number of valid seating arrangements
def valid_seating_arrangements (n : ℕ) (s : ℕ) : ℕ :=
  -- The actual implementation is not provided here
  sorry

-- State the theorem
theorem seating_arrangement_count :
  valid_seating_arrangements num_people seats_per_row = 518400 :=
sorry

end NUMINAMATH_CALUDE_seating_arrangement_count_l2312_231242


namespace NUMINAMATH_CALUDE_current_short_trees_count_l2312_231206

/-- The number of short trees currently in the park -/
def current_short_trees : ℕ := 41

/-- The number of short trees to be planted today -/
def trees_to_plant : ℕ := 57

/-- The total number of short trees after planting -/
def total_short_trees : ℕ := 98

/-- Theorem stating that the number of short trees currently in the park is 41 -/
theorem current_short_trees_count :
  current_short_trees + trees_to_plant = total_short_trees :=
by sorry

end NUMINAMATH_CALUDE_current_short_trees_count_l2312_231206


namespace NUMINAMATH_CALUDE_probability_is_two_thirds_l2312_231217

structure Diagram where
  total_triangles : ℕ
  triangles_with_G : ℕ
  equal_probability : Bool

def probability_including_G (d : Diagram) : ℚ :=
  d.triangles_with_G / d.total_triangles

theorem probability_is_two_thirds (d : Diagram) 
  (h1 : d.total_triangles = 6)
  (h2 : d.triangles_with_G = 4)
  (h3 : d.equal_probability = true) :
  probability_including_G d = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_two_thirds_l2312_231217


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l2312_231231

theorem point_in_fourth_quadrant (x : ℝ) : 
  let P : ℝ × ℝ := (x^2 + 1, -2)
  P.1 > 0 ∧ P.2 < 0 :=
by sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l2312_231231


namespace NUMINAMATH_CALUDE_intersection_A_B_l2312_231219

def A : Set ℝ := {-3, -1, 0, 1, 2, 3, 4}
def B : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 3}

theorem intersection_A_B : A ∩ B = {0, 1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2312_231219


namespace NUMINAMATH_CALUDE_no_extreme_points_iff_m_range_l2312_231292

/-- The function f(x) = x ln x + mx² - m has no extreme points in its domain
    if and only if m ∈ (-∞, -1/2] --/
theorem no_extreme_points_iff_m_range (m : ℝ) :
  (∀ x : ℝ, x > 0 → ∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε),
    (y * Real.log y + m * y^2 - m ≠ x * Real.log x + m * x^2 - m)) ↔
  m ≤ -1/2 :=
sorry

end NUMINAMATH_CALUDE_no_extreme_points_iff_m_range_l2312_231292


namespace NUMINAMATH_CALUDE_min_value_theorem_l2312_231236

theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_constraint : 3 * m + n = 1) :
  1 / m + 3 / n ≥ 12 ∧ ∃ m₀ n₀ : ℝ, m₀ > 0 ∧ n₀ > 0 ∧ 3 * m₀ + n₀ = 1 ∧ 1 / m₀ + 3 / n₀ = 12 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2312_231236


namespace NUMINAMATH_CALUDE_polar_to_cartesian_line_l2312_231277

/-- The polar equation r = 2 / (2sin θ - cos θ) represents a line in Cartesian coordinates. -/
theorem polar_to_cartesian_line :
  ∃ (a b c : ℝ), a ≠ 0 ∨ b ≠ 0 ∧
  ∀ (x y : ℝ), (∃ (r θ : ℝ), r > 0 ∧
    r = 2 / (2 * Real.sin θ - Real.cos θ) ∧
    x = r * Real.cos θ ∧
    y = r * Real.sin θ) →
  a * x + b * y = c :=
sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_line_l2312_231277


namespace NUMINAMATH_CALUDE_orange_apple_difference_l2312_231261

def apples : ℕ := 14
def dozen : ℕ := 12
def oranges : ℕ := 2 * dozen

theorem orange_apple_difference : oranges - apples = 10 := by
  sorry

end NUMINAMATH_CALUDE_orange_apple_difference_l2312_231261


namespace NUMINAMATH_CALUDE_ellipse_chord_ratio_range_l2312_231285

/-- Define an ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0

/-- Define a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Define a line with slope k passing through a point -/
structure Line where
  k : ℝ
  p : Point
  h_k : k ≠ 0

/-- Theorem statement -/
theorem ellipse_chord_ratio_range (C : Ellipse) (F : Point) (l : Line) :
  F.x = 1 ∧ F.y = 0 ∧ l.p = F →
  ∃ (B₁ B₂ M N D P : Point),
    -- B₁ and B₂ are endpoints of minor axis
    B₁.x = 0 ∧ B₂.x = 0 ∧ B₁.y = -C.b ∧ B₂.y = C.b ∧
    -- Condition on FB₁ · FB₂
    (F.x - B₁.x) * (F.x - B₂.x) + (F.y - B₁.y) * (F.y - B₂.y) = -C.a ∧
    -- M and N are intersections of l and C
    (M.y - F.y = l.k * (M.x - F.x) ∧ M.x^2 / C.a^2 + M.y^2 / C.b^2 = 1) ∧
    (N.y - F.y = l.k * (N.x - F.x) ∧ N.x^2 / C.a^2 + N.y^2 / C.b^2 = 1) ∧
    -- P is midpoint of MN
    P.x = (M.x + N.x) / 2 ∧ P.y = (M.y + N.y) / 2 ∧
    -- D is on x-axis and PD is perpendicular to MN
    D.y = 0 ∧ (P.y - D.y) * (N.x - M.x) = -(P.x - D.x) * (N.y - M.y) →
    -- Conclusion: range of DP/MN
    ∀ r : ℝ, (r = (Real.sqrt ((P.x - D.x)^2 + (P.y - D.y)^2)) /
               (Real.sqrt ((M.x - N.x)^2 + (M.y - N.y)^2))) →
      0 < r ∧ r < 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_chord_ratio_range_l2312_231285


namespace NUMINAMATH_CALUDE_reading_percentage_third_night_l2312_231241

/-- Theorem: Reading percentage on the third night
Given:
- A book with 500 pages
- 20% read on the first night
- 20% read on the second night
- 150 pages left after three nights of reading
Prove: The percentage read on the third night is 30%
-/
theorem reading_percentage_third_night
  (total_pages : ℕ)
  (first_night_percentage : ℚ)
  (second_night_percentage : ℚ)
  (pages_left : ℕ)
  (h1 : total_pages = 500)
  (h2 : first_night_percentage = 20 / 100)
  (h3 : second_night_percentage = 20 / 100)
  (h4 : pages_left = 150) :
  let pages_read_first_two_nights := (first_night_percentage + second_night_percentage) * total_pages
  let total_pages_read := total_pages - pages_left
  let pages_read_third_night := total_pages_read - pages_read_first_two_nights
  pages_read_third_night / total_pages = 30 / 100 := by
  sorry

end NUMINAMATH_CALUDE_reading_percentage_third_night_l2312_231241


namespace NUMINAMATH_CALUDE_max_red_squares_is_twelve_l2312_231265

/-- A configuration of colored squares on a 5x5 grid -/
def ColorConfiguration := Fin 5 → Fin 5 → Bool

/-- Checks if four points form an axis-parallel rectangle -/
def isAxisParallelRectangle (p1 p2 p3 p4 : Fin 5 × Fin 5) : Bool :=
  sorry

/-- Checks if a configuration contains an axis-parallel rectangle formed by red squares -/
def containsAxisParallelRectangle (config : ColorConfiguration) : Bool :=
  sorry

/-- Counts the number of red squares in a configuration -/
def countRedSquares (config : ColorConfiguration) : Nat :=
  sorry

/-- The maximum number of red squares possible without forming an axis-parallel rectangle -/
def maxRedSquares : Nat :=
  sorry

theorem max_red_squares_is_twelve :
  maxRedSquares = 12 :=
sorry

end NUMINAMATH_CALUDE_max_red_squares_is_twelve_l2312_231265


namespace NUMINAMATH_CALUDE_square_sum_inequality_l2312_231284

theorem square_sum_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a^2 + b^2 + c^2 + d^2)^2 ≥ (a + b) * (b + c) * (c + d) * (d + a) ∧
  ((a^2 + b^2 + c^2 + d^2)^2 = (a + b) * (b + c) * (c + d) * (d + a) ↔ a = b ∧ b = c ∧ c = d) :=
by sorry

end NUMINAMATH_CALUDE_square_sum_inequality_l2312_231284


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l2312_231238

theorem regular_polygon_sides (interior_angle : ℝ) : 
  interior_angle = 150 → (360 / (180 - interior_angle) : ℝ) = 12 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l2312_231238


namespace NUMINAMATH_CALUDE_cube_sum_problem_l2312_231294

theorem cube_sum_problem (x y z : ℝ) 
  (sum_eq : x + y + z = 8)
  (sum_prod_eq : x*y + x*z + y*z = 17)
  (prod_eq : x*y*z = -14) :
  x^3 + y^3 + z^3 = 62 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_problem_l2312_231294


namespace NUMINAMATH_CALUDE_existence_of_special_odd_numbers_l2312_231251

theorem existence_of_special_odd_numbers : ∃ m n : ℕ, 
  Odd m ∧ Odd n ∧ 
  m > 2009 ∧ n > 2009 ∧ 
  (n^2 + 8) % m = 0 ∧ 
  (m^2 + 8) % n = 0 := by
sorry

end NUMINAMATH_CALUDE_existence_of_special_odd_numbers_l2312_231251


namespace NUMINAMATH_CALUDE_smallest_integer_l2312_231289

theorem smallest_integer (x : ℕ+) (a b : ℕ+) : 
  (Nat.gcd a b = x + 3) →
  (Nat.lcm a b = x * (x + 3)) →
  (b = 36) →
  (∀ y : ℕ+, y < x → ¬(∃ c : ℕ+, 
    Nat.gcd c 36 = y + 3 ∧ 
    Nat.lcm c 36 = y * (y + 3))) →
  (a = 108) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_l2312_231289


namespace NUMINAMATH_CALUDE_cubic_roots_collinear_k_l2312_231267

/-- A cubic polynomial with coefficient k -/
def cubic_polynomial (k : ℤ) (x : ℂ) : ℂ :=
  x^3 - 15*x^2 + k*x - 1105

/-- Predicate for three complex numbers being distinct and collinear -/
def distinct_collinear (z₁ z₂ z₃ : ℂ) : Prop :=
  z₁ ≠ z₂ ∧ z₁ ≠ z₃ ∧ z₂ ≠ z₃ ∧
  ∃ (a b : ℝ), (z₁.im - a * z₁.re = b) ∧ 
               (z₂.im - a * z₂.re = b) ∧ 
               (z₃.im - a * z₃.re = b)

theorem cubic_roots_collinear_k (k : ℤ) :
  (∃ (z₁ z₂ z₃ : ℂ), 
    distinct_collinear z₁ z₂ z₃ ∧
    (cubic_polynomial k z₁ = 0) ∧
    (cubic_polynomial k z₂ = 0) ∧
    (cubic_polynomial k z₃ = 0)) →
  k = 271 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_collinear_k_l2312_231267


namespace NUMINAMATH_CALUDE_max_sunny_days_thursday_l2312_231264

/-- Represents the days of the week -/
inductive Day : Type
  | monday
  | tuesday
  | wednesday
  | thursday
  | friday
  | saturday
  | sunday

/-- Represents the weather conditions -/
inductive Weather : Type
  | sunny
  | rainy
  | foggy

/-- The weather pattern for each day of the week -/
def weatherPattern (d : Day) : Weather :=
  match d with
  | Day.monday => Weather.rainy
  | Day.friday => Weather.rainy
  | Day.saturday => Weather.foggy
  | _ => Weather.sunny

/-- Calculates the number of sunny days in a 30-day period starting from a given day -/
def sunnyDaysCount (startDay : Day) : Nat :=
  sorry

/-- Theorem: Starting on Thursday maximizes the number of sunny days in a 30-day period -/
theorem max_sunny_days_thursday :
  ∀ d : Day, sunnyDaysCount Day.thursday ≥ sunnyDaysCount d :=
  sorry

end NUMINAMATH_CALUDE_max_sunny_days_thursday_l2312_231264


namespace NUMINAMATH_CALUDE_fault_line_movement_l2312_231215

/-- The movement of a fault line over two years -/
theorem fault_line_movement 
  (movement_past_year : ℝ) 
  (movement_year_before : ℝ) 
  (h1 : movement_past_year = 1.25)
  (h2 : movement_year_before = 5.25) : 
  movement_past_year + movement_year_before = 6.50 := by
sorry

end NUMINAMATH_CALUDE_fault_line_movement_l2312_231215


namespace NUMINAMATH_CALUDE_inequality_solution_range_l2312_231232

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, x^2 - a*x - a ≤ -3) → 
  (a ≤ -6 ∨ a ≥ 2) := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l2312_231232


namespace NUMINAMATH_CALUDE_unknown_cube_edge_length_l2312_231226

theorem unknown_cube_edge_length 
  (edge1 : ℝ) (edge2 : ℝ) (edge_unknown : ℝ) (edge_new : ℝ)
  (h1 : edge1 = 6)
  (h2 : edge2 = 8)
  (h3 : edge_new = 12)
  (h4 : edge1^3 + edge2^3 + edge_unknown^3 = edge_new^3) :
  edge_unknown = 10 := by sorry

end NUMINAMATH_CALUDE_unknown_cube_edge_length_l2312_231226


namespace NUMINAMATH_CALUDE_a_less_equal_two_l2312_231253

-- Define the sets A and B
def A : Set ℝ := {x | x ≥ 3}
def B (a : ℝ) : Set ℝ := {x | 2 * a - x > 1}

-- State the theorem
theorem a_less_equal_two (a : ℝ) : A ∩ (Set.univ \ B a) = A → a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_a_less_equal_two_l2312_231253


namespace NUMINAMATH_CALUDE_combination_permutation_equation_l2312_231211

/-- Binomial coefficient -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- Falling factorial -/
def fallingFactorial (n k : ℕ) : ℕ := (n - k + 1).factorial / (n - k).factorial

theorem combination_permutation_equation : 
  ∃ x : ℕ, binomial (x + 5) x = binomial (x + 3) (x - 1) + binomial (x + 3) (x - 2) + 
    (3 * fallingFactorial (x + 3) 3) / 4 ∧ x = 14 := by
  sorry

end NUMINAMATH_CALUDE_combination_permutation_equation_l2312_231211


namespace NUMINAMATH_CALUDE_intersection_points_on_parabola_l2312_231234

theorem intersection_points_on_parabola :
  ∀ (x y : ℝ),
  (x = y^2 ∧ (x - 11)^2 + (y - 1)^2 = 25) →
  y = (1/2) * x^2 - (21/2) * x + 97/2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_on_parabola_l2312_231234


namespace NUMINAMATH_CALUDE_sequence_sum_l2312_231227

theorem sequence_sum (S : ℝ) (a b : ℝ) : 
  (S - a) / 100 = 2022 →
  (S - b) / 100 = 2023 →
  (a + b) / 2 = 51 →
  S = 202301 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_l2312_231227


namespace NUMINAMATH_CALUDE_simplify_product_of_square_roots_l2312_231259

theorem simplify_product_of_square_roots (x : ℝ) :
  Real.sqrt (x^2 - 4*x + 4) * Real.sqrt (x^2 + 4*x + 4) = |x - 2| * |x + 2| := by
  sorry

end NUMINAMATH_CALUDE_simplify_product_of_square_roots_l2312_231259


namespace NUMINAMATH_CALUDE_arithmetic_sequence_proof_l2312_231214

-- Define an arithmetic sequence
def arithmeticSequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1) * d

theorem arithmetic_sequence_proof :
  (arithmeticSequence 8 (-3) 20 = -49) ∧
  (arithmeticSequence (-5) (-4) 100 = -401) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_proof_l2312_231214


namespace NUMINAMATH_CALUDE_min_value_expression_l2312_231282

theorem min_value_expression (x y : ℝ) (hx : x > 1) (hy : y > 1) :
  (x^3 / (y - 1)) + (y^3 / (x - 1)) ≥ 8 ∧
  ((x^3 / (y - 1)) + (y^3 / (x - 1)) = 8 ↔ x = 2 ∧ y = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2312_231282
