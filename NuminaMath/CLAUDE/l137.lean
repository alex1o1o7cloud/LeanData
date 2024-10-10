import Mathlib

namespace roof_collapse_days_l137_13717

theorem roof_collapse_days (roof_weight_limit : ℕ) (leaves_per_day : ℕ) (leaves_per_pound : ℕ) : 
  roof_weight_limit = 500 → 
  leaves_per_day = 100 → 
  leaves_per_pound = 1000 → 
  (roof_weight_limit * leaves_per_pound) / leaves_per_day = 5000 := by
  sorry

#check roof_collapse_days

end roof_collapse_days_l137_13717


namespace cone_sphere_ratio_l137_13777

/-- A cone with three spheres inside it satisfying specific conditions -/
structure ConeWithSpheres where
  R : ℝ  -- Radius of the base of the cone
  r : ℝ  -- Radius of each sphere
  h : ℝ  -- Height of the cone
  -- The diameter of the base of the cone is equal to the slant height
  diam_eq_slant : R * 2 = Real.sqrt (R^2 + h^2)
  -- The spheres touch each other externally
  spheres_touch : True
  -- Two spheres touch the lateral surface and the base of the cone
  two_spheres_touch_base : True
  -- The third sphere touches the lateral surface at a point lying in the same plane with the centers of the spheres
  third_sphere_touch : True

/-- The ratio of the radius of the base of the cone to the radius of a sphere is (5/4 + √3) -/
theorem cone_sphere_ratio (c : ConeWithSpheres) : c.R / c.r = 5/4 + Real.sqrt 3 := by
  sorry

end cone_sphere_ratio_l137_13777


namespace kittens_given_to_jessica_l137_13754

theorem kittens_given_to_jessica (initial_kittens : ℕ) (kittens_to_sara : ℕ) (kittens_left : ℕ) :
  initial_kittens = 18 → kittens_to_sara = 6 → kittens_left = 9 →
  initial_kittens - kittens_to_sara - kittens_left = 3 :=
by sorry

end kittens_given_to_jessica_l137_13754


namespace sum_x_coordinates_invariant_l137_13705

/-- Represents a polygon in the Cartesian plane -/
structure Polygon where
  vertices : List (ℝ × ℝ)

/-- Creates a new polygon by finding midpoints of edges of the given polygon -/
def midpointTransform (p : Polygon) : Polygon :=
  sorry

/-- Calculates the sum of x-coordinates of a polygon's vertices -/
def sumXCoordinates (p : Polygon) : ℝ :=
  sorry

/-- Theorem: The sum of x-coordinates remains constant after two midpoint transformations -/
theorem sum_x_coordinates_invariant (Q₁ : Polygon) 
  (h : sumXCoordinates Q₁ = 132) 
  (h_vertices : Q₁.vertices.length = 44) : 
  sumXCoordinates (midpointTransform (midpointTransform Q₁)) = 132 := by
  sorry

end sum_x_coordinates_invariant_l137_13705


namespace operation_b_correct_operation_c_correct_l137_13715

-- Operation B
theorem operation_b_correct (t : ℝ) : (-2 * t) * (3 * t + t^2 - 1) = -6 * t^2 - 2 * t^3 + 2 * t := by
  sorry

-- Operation C
theorem operation_c_correct (x y : ℝ) : (-2 * x * y^3)^2 = 4 * x^2 * y^6 := by
  sorry

end operation_b_correct_operation_c_correct_l137_13715


namespace sin_shift_l137_13722

theorem sin_shift (x : ℝ) : Real.sin (5 * π / 6 - x) = Real.sin (x + π / 6) := by
  sorry

end sin_shift_l137_13722


namespace point_transformation_l137_13773

/-- Reflect a point (x, y) across the line y = x -/
def reflect_across_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

/-- Rotate a point (x, y) by 180° around a center (h, k) -/
def rotate_180_around (p : ℝ × ℝ) (center : ℝ × ℝ) : ℝ × ℝ :=
  (2 * center.1 - p.1, 2 * center.2 - p.2)

/-- The main theorem -/
theorem point_transformation (a b : ℝ) :
  let Q : ℝ × ℝ := (a, b)
  let reflected := reflect_across_y_eq_x Q
  let rotated := rotate_180_around reflected (1, 5)
  rotated = (-8, 2) → a - b = -2 := by
  sorry

end point_transformation_l137_13773


namespace solution_for_equation_l137_13757

noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

theorem solution_for_equation (x : ℝ) (h1 : 0 < x) (h2 : x ≠ 1) :
  x^2 * log x 27 * log 9 x = x + 4 ↔ x = 2 :=
sorry

end solution_for_equation_l137_13757


namespace three_planes_seven_parts_intersection_lines_l137_13740

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields for a plane

/-- Represents the division of space by planes -/
structure SpaceDivision where
  planes : List Plane3D
  num_parts : Nat

/-- Counts the number of intersection lines between planes -/
def count_intersection_lines (division : SpaceDivision) : Nat :=
  sorry

theorem three_planes_seven_parts_intersection_lines 
  (division : SpaceDivision) 
  (h_planes : division.planes.length = 3)
  (h_parts : division.num_parts = 7) :
  count_intersection_lines division = 3 := by
  sorry

end three_planes_seven_parts_intersection_lines_l137_13740


namespace divisors_of_eight_n_cubed_l137_13776

theorem divisors_of_eight_n_cubed (n : ℕ) (h_odd : Odd n) (h_divisors : (Nat.divisors n).card = 17) :
  (Nat.divisors (8 * n^3)).card = 196 := by
  sorry

end divisors_of_eight_n_cubed_l137_13776


namespace range_of_a_l137_13793

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, Real.exp x - a * x ≥ -x + Real.log (a * x)) ↔ (0 < a ∧ a ≤ Real.exp 1) :=
sorry

end range_of_a_l137_13793


namespace light_glow_start_time_l137_13728

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Calculates the difference between two times in seconds -/
def timeDiffInSeconds (t1 t2 : Time) : Nat :=
  (t1.hours * 3600 + t1.minutes * 60 + t1.seconds) -
  (t2.hours * 3600 + t2.minutes * 60 + t2.seconds)

theorem light_glow_start_time 
  (glow_interval : Nat) 
  (glow_count : Nat) 
  (end_time : Time) 
  (start_time : Time) : 
  glow_interval = 17 →
  glow_count = 292 →
  end_time = { hours := 3, minutes := 20, seconds := 47 } →
  start_time = { hours := 1, minutes := 58, seconds := 3 } →
  timeDiffInSeconds end_time start_time = glow_interval * glow_count :=
by sorry

end light_glow_start_time_l137_13728


namespace jamie_water_bottle_limit_l137_13799

/-- The maximum amount of liquid Jamie can consume before needing the bathroom -/
def bathroom_limit : ℕ := 32

/-- The amount of milk Jamie consumed -/
def milk_consumed : ℕ := 8

/-- The amount of grape juice Jamie consumed -/
def grape_juice_consumed : ℕ := 16

/-- The amount Jamie can drink from her water bottle during the test -/
def water_bottle_limit : ℕ := bathroom_limit - (milk_consumed + grape_juice_consumed)

theorem jamie_water_bottle_limit :
  water_bottle_limit = 8 :=
by sorry

end jamie_water_bottle_limit_l137_13799


namespace arithmetic_sequence_special_case_l137_13782

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_special_case (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 4)^2 - 4*(a 4) - 1 = 0 →
  (a 8)^2 - 4*(a 8) - 1 = 0 →
  a 6 = 2 := by
sorry

end arithmetic_sequence_special_case_l137_13782


namespace mobile_phone_cost_mobile_phone_cost_is_8000_l137_13784

/-- Proves that the cost of the mobile phone is 8000, given the conditions of the problem -/
theorem mobile_phone_cost : ℕ → Prop :=
  fun cost_mobile =>
    let cost_refrigerator : ℕ := 15000
    let loss_rate_refrigerator : ℚ := 2 / 100
    let profit_rate_mobile : ℚ := 10 / 100
    let overall_profit : ℕ := 500
    let selling_price_refrigerator : ℚ := cost_refrigerator * (1 - loss_rate_refrigerator)
    let selling_price_mobile : ℚ := cost_mobile * (1 + profit_rate_mobile)
    selling_price_refrigerator + selling_price_mobile - (cost_refrigerator + cost_mobile) = overall_profit →
    cost_mobile = 8000

/-- The cost of the mobile phone is 8000 -/
theorem mobile_phone_cost_is_8000 : mobile_phone_cost 8000 := by
  sorry

end mobile_phone_cost_mobile_phone_cost_is_8000_l137_13784


namespace train_speed_l137_13730

/-- The speed of a train given its length and time to cross a fixed point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 1500) (h2 : time = 50) :
  length / time = 30 := by
  sorry

end train_speed_l137_13730


namespace sqrt_81_equals_9_l137_13732

theorem sqrt_81_equals_9 : Real.sqrt 81 = 9 := by
  sorry

end sqrt_81_equals_9_l137_13732


namespace plane_sphere_intersection_l137_13797

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Represents a sphere in 3D space -/
structure Sphere3D where
  center : Point3D
  radius : ℝ

/-- The theorem to be proved -/
theorem plane_sphere_intersection (a b c p q r : ℝ) 
  (plane : Plane3D) 
  (sphere : Sphere3D) : 
  (plane.a = a ∧ plane.b = b ∧ plane.c = c) →  -- Plane passes through (a,b,c)
  (∃ (α β γ : ℝ), 
    plane.a * α + plane.b * 0 + plane.c * 0 + plane.d = 0 ∧  -- Plane intersects x-axis at (α,0,0)
    plane.a * 0 + plane.b * β + plane.c * 0 + plane.d = 0 ∧  -- Plane intersects y-axis at (0,β,0)
    plane.a * 0 + plane.b * 0 + plane.c * γ + plane.d = 0) →  -- Plane intersects z-axis at (0,0,γ)
  (sphere.center = Point3D.mk (p+1) (q+1) (r+1)) →  -- Sphere center is shifted by (1,1,1)
  (∃ (α β γ : ℝ), 
    sphere.radius^2 = (p+1)^2 + (q+1)^2 + (r+1)^2 ∧  -- Sphere passes through origin
    sphere.radius^2 = ((p+1) - α)^2 + (q+1)^2 + (r+1)^2 ∧  -- Sphere passes through A
    sphere.radius^2 = (p+1)^2 + ((q+1) - β)^2 + (r+1)^2 ∧  -- Sphere passes through B
    sphere.radius^2 = (p+1)^2 + (q+1)^2 + ((r+1) - γ)^2) →  -- Sphere passes through C
  a/p + b/q + c/r = 2 := by
  sorry


end plane_sphere_intersection_l137_13797


namespace isosceles_triangle_special_angles_l137_13783

/-- An isosceles triangle with vertex angle twice the base angle has a 90° vertex angle and 45° base angles. -/
theorem isosceles_triangle_special_angles :
  ∀ (vertex_angle base_angle : ℝ),
    vertex_angle > 0 →
    base_angle > 0 →
    vertex_angle = 2 * base_angle →
    vertex_angle + 2 * base_angle = 180 →
    vertex_angle = 90 ∧ base_angle = 45 :=
by
  sorry

end isosceles_triangle_special_angles_l137_13783


namespace line_plane_perpendicularity_l137_13786

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (m n : Line) (α β : Plane) 
  (h_different_lines : m ≠ n)
  (h_different_planes : α ≠ β)
  (h_n_subset_β : subset n β)
  (h_m_parallel_n : parallel m n)
  (h_m_perp_α : perpendicular m α) :
  perpendicular_planes α β := by
  sorry

end line_plane_perpendicularity_l137_13786


namespace problem_statement_l137_13701

theorem problem_statement (a b : ℝ) (ha : a > 0) (heq : Real.exp a + Real.log b = 1) :
  (a + Real.log b < 0) ∧ (Real.exp a + b > 2) ∧ (a + b > 1) := by
  sorry

end problem_statement_l137_13701


namespace certain_amount_calculation_l137_13724

theorem certain_amount_calculation (amount : ℝ) : 
  (5 / 100) * ((25 / 100) * amount) = 20 → amount = 1600 := by
  sorry

end certain_amount_calculation_l137_13724


namespace smallest_brownie_pan_dimension_l137_13742

def is_valid_brownie_pan (m n : ℕ) : Prop :=
  m > 0 ∧ n > 0 ∧ (m - 2) * (n - 2) = 2 * (2 * m + 2 * n - 4)

def smallest_dimension : ℕ := 12

theorem smallest_brownie_pan_dimension :
  (is_valid_brownie_pan smallest_dimension smallest_dimension) ∧
  (∀ k : ℕ, k < smallest_dimension → ¬(is_valid_brownie_pan k k) ∧ ¬(∃ l : ℕ, is_valid_brownie_pan k l ∨ is_valid_brownie_pan l k)) :=
by sorry

end smallest_brownie_pan_dimension_l137_13742


namespace jordan_running_time_l137_13765

/-- Given that Jordan ran 3 miles in 1/3 of the time it took Steve to run 4 miles,
    and Steve ran 4 miles in 32 minutes, prove that Jordan would take 224/9 minutes
    to run 7 miles. -/
theorem jordan_running_time (steve_time : ℝ) (jordan_distance : ℝ) :
  steve_time = 32 →
  jordan_distance = 7 →
  (3 / (1/3 * steve_time)) * jordan_distance = 224/9 := by
  sorry

end jordan_running_time_l137_13765


namespace sausage_distance_ratio_l137_13781

/-- Represents the scenario of a dog and cat running towards sausages --/
structure SausageScenario where
  dog_speed : ℝ
  cat_speed : ℝ
  dog_eat_rate : ℝ
  cat_eat_rate : ℝ
  total_sausages : ℝ
  total_distance : ℝ

/-- The theorem to be proved --/
theorem sausage_distance_ratio 
  (scenario : SausageScenario)
  (h1 : scenario.cat_speed = 2 * scenario.dog_speed)
  (h2 : scenario.dog_eat_rate = scenario.cat_eat_rate / 2)
  (h3 : scenario.cat_eat_rate * 1 = scenario.total_sausages)
  (h4 : scenario.cat_speed * 1 = scenario.total_distance)
  (h5 : scenario.total_sausages > 0)
  (h6 : scenario.total_distance > 0) :
  ∃ (cat_distance dog_distance : ℝ),
    cat_distance + dog_distance = scenario.total_distance ∧
    cat_distance / dog_distance = 7 / 5 := by
  sorry


end sausage_distance_ratio_l137_13781


namespace shooting_test_probability_l137_13729

/-- The number of shots in the test -/
def num_shots : ℕ := 3

/-- The minimum number of successful shots required to pass -/
def min_success : ℕ := 2

/-- The probability of making a single shot -/
def shot_probability : ℝ := 0.6

/-- The probability of passing the test -/
def pass_probability : ℝ := 0.648

/-- Theorem stating that the calculated probability of passing the test is correct -/
theorem shooting_test_probability : 
  (Finset.sum (Finset.range (num_shots - min_success + 1))
    (λ k => Nat.choose num_shots (num_shots - k) * 
      shot_probability ^ (num_shots - k) * 
      (1 - shot_probability) ^ k)) = pass_probability := by
  sorry

end shooting_test_probability_l137_13729


namespace discontinuous_function_l137_13747

def M (f : ℝ → ℝ) (x : Fin n → ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  Matrix.of (λ i j => if i = j then 1 + f (x i) else f (x j))

theorem discontinuous_function
  (f : ℝ → ℝ)
  (f_nonzero : ∀ x, f x ≠ 0)
  (f_condition : f 2014 = 1 - f 2013)
  (det_zero : ∀ (n : ℕ) (x : Fin n → ℝ), Function.Injective x → Matrix.det (M f x) = 0) :
  ¬Continuous f :=
sorry

end discontinuous_function_l137_13747


namespace mike_buys_36_games_l137_13746

/-- Represents the number of days Mike worked --/
def total_days : ℕ := 20

/-- Represents the earnings per lawn in dollars --/
def earnings_per_lawn : ℕ := 5

/-- Represents the number of lawns mowed on a weekday --/
def lawns_per_weekday : ℕ := 2

/-- Represents the number of lawns mowed on a weekend day --/
def lawns_per_weekend : ℕ := 3

/-- Represents the cost of new mower blades in dollars --/
def cost_of_blades : ℕ := 24

/-- Represents the cost of gasoline in dollars --/
def cost_of_gas : ℕ := 15

/-- Represents the cost of each game in dollars --/
def cost_per_game : ℕ := 5

/-- Calculates the number of games Mike can buy --/
def games_mike_can_buy : ℕ :=
  let weekdays := 16
  let weekend_days := 4
  let total_lawns := weekdays * lawns_per_weekday + weekend_days * lawns_per_weekend
  let total_earnings := total_lawns * earnings_per_lawn
  let total_expenses := cost_of_blades + cost_of_gas
  let money_left := total_earnings - total_expenses
  money_left / cost_per_game

/-- Theorem stating that Mike can buy 36 games --/
theorem mike_buys_36_games : games_mike_can_buy = 36 := by
  sorry

end mike_buys_36_games_l137_13746


namespace fourth_power_trinomial_coefficients_l137_13779

/-- A trinomial that is an exact fourth power for all integers -/
def is_fourth_power (a b c : ℝ) : Prop :=
  ∀ x : ℤ, ∃ y : ℝ, a * x^2 + b * x + c = y^4

/-- If a trinomial is an exact fourth power for all integers, then its quadratic and linear coefficients are zero -/
theorem fourth_power_trinomial_coefficients (a b c : ℝ) :
  is_fourth_power a b c → a = 0 ∧ b = 0 :=
by sorry

end fourth_power_trinomial_coefficients_l137_13779


namespace units_digit_of_150_factorial_l137_13711

theorem units_digit_of_150_factorial (n : ℕ) : n = 150 → n.factorial % 10 = 0 := by
  sorry

end units_digit_of_150_factorial_l137_13711


namespace binomial_square_condition_l137_13700

/-- If ax^2 + 24x + 9 is the square of a binomial, then a = 16 -/
theorem binomial_square_condition (a : ℝ) : 
  (∃ r s : ℝ, ∀ x : ℝ, a * x^2 + 24 * x + 9 = (r * x + s)^2) → a = 16 := by
  sorry

end binomial_square_condition_l137_13700


namespace fg_squared_value_l137_13748

-- Define the functions g and f
def g (x : ℝ) : ℝ := 4 * x + 5
def f (x : ℝ) : ℝ := 6 * x - 11

-- State the theorem
theorem fg_squared_value : (f (g 6))^2 = 26569 := by sorry

end fg_squared_value_l137_13748


namespace min_abs_sum_l137_13774

theorem min_abs_sum (x₁ x₂ : ℝ) 
  (h : (2 + Real.sin x₁) * (2 + Real.sin (2 * x₂)) = 1) : 
  ∃ (k m : ℤ), |x₁ + x₂| ≥ π / 4 ∧ 
  |x₁ + x₂| = π / 4 ↔ x₁ = 3 * π / 2 + 2 * π * k ∧ x₂ = 3 * π / 4 + π * m := by
  sorry

end min_abs_sum_l137_13774


namespace revenue_change_l137_13792

/-- Proves that given a 75% price decrease and a specific ratio between percent increase in units sold
    and percent decrease in price, the new revenue is 50% of the original revenue -/
theorem revenue_change (P Q : ℝ) (P' Q' : ℝ) (h1 : P' = 0.25 * P) 
    (h2 : (Q' / Q - 1) / 0.75 = 1.3333333333333333) : P' * Q' = 0.5 * P * Q := by
  sorry

end revenue_change_l137_13792


namespace max_cables_theorem_l137_13750

/-- Represents a computer network with two brands of computers. -/
structure ComputerNetwork where
  brandA : ℕ  -- Number of brand A computers
  brandB : ℕ  -- Number of brand B computers

/-- Calculates the maximum number of cables that can be used in the network. -/
def maxCables (network : ComputerNetwork) : ℕ :=
  network.brandA * network.brandB

/-- Theorem: The maximum number of cables in a network with 25 brand A and 15 brand B computers is 361. -/
theorem max_cables_theorem (network : ComputerNetwork) 
  (h1 : network.brandA = 25) 
  (h2 : network.brandB = 15) : 
  maxCables network = 361 := by
  sorry

#eval maxCables { brandA := 25, brandB := 15 }

end max_cables_theorem_l137_13750


namespace lucille_house_height_difference_l137_13753

theorem lucille_house_height_difference (h1 h2 h3 : ℝ) :
  h1 = 80 ∧ h2 = 70 ∧ h3 = 99 →
  ((h1 + h2 + h3) / 3) - h1 = 3 :=
by sorry

end lucille_house_height_difference_l137_13753


namespace f_composition_negative_four_l137_13787

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 + x - 2 else -Real.log x

-- State the theorem
theorem f_composition_negative_four (x : ℝ) : f (f (-4)) = -1 := by
  sorry

end f_composition_negative_four_l137_13787


namespace complex_average_calculation_l137_13741

def avg2 (a b : ℚ) : ℚ := (a + b) / 2

def avg4 (a b c d : ℚ) : ℚ := (a + b + c + d) / 4

theorem complex_average_calculation :
  avg4 (avg4 2 2 (-1) (avg2 1 3)) 7 (avg2 4 (5 - 2)) = 27 / 8 := by
  sorry

end complex_average_calculation_l137_13741


namespace volunteer_arrangement_count_l137_13770

theorem volunteer_arrangement_count (n : ℕ) (k : ℕ) (h1 : n = 7) (h2 : k = 3) :
  Nat.choose n k * Nat.choose (n - k) k = 140 := by
  sorry

end volunteer_arrangement_count_l137_13770


namespace parallelogram_is_rhombus_l137_13702

/-- A parallelogram ABCD in a 2D Euclidean space. -/
structure Parallelogram where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- Vector addition -/
def vecAdd (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)

/-- Vector subtraction -/
def vecSub (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)

/-- Dot product of two vectors -/
def dotProduct (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- The zero vector -/
def zeroVec : ℝ × ℝ := (0, 0)

/-- Theorem: A parallelogram is a rhombus if it satisfies certain vector conditions -/
theorem parallelogram_is_rhombus (ABCD : Parallelogram)
  (h1 : vecAdd (vecSub ABCD.B ABCD.A) (vecSub ABCD.D ABCD.C) = zeroVec)
  (h2 : dotProduct (vecSub (vecSub ABCD.B ABCD.A) (vecSub ABCD.D ABCD.A)) (vecSub ABCD.C ABCD.A) = 0) :
  ABCD.A = ABCD.B ∧ ABCD.B = ABCD.C ∧ ABCD.C = ABCD.D ∧ ABCD.D = ABCD.A := by
  sorry

end parallelogram_is_rhombus_l137_13702


namespace star_pattern_identifiable_and_separable_l137_13707

/-- Represents a patch in the tablecloth -/
structure Patch :=
  (shape : Type)
  (material : Type)

/-- Represents the tablecloth -/
structure Tablecloth :=
  (patches : Set Patch)
  (isTriangular : ∀ p ∈ patches, p.shape = Triangle)
  (isSilk : ∀ p ∈ patches, p.material = Silk)

/-- Represents a star pattern -/
structure StarPattern :=
  (patches : Set Patch)
  (isSymmetrical : Bool)
  (fitsWithRest : Tablecloth → Bool)

/-- Theorem: If a symmetrical star pattern exists in the tablecloth, it can be identified and separated -/
theorem star_pattern_identifiable_and_separable 
  (tc : Tablecloth) 
  (sp : StarPattern) 
  (h1 : sp.patches ⊆ tc.patches) 
  (h2 : sp.isSymmetrical = true) 
  (h3 : sp.fitsWithRest tc = true) : 
  ∃ (identified_sp : StarPattern), identified_sp = sp ∧ 
  ∃ (separated_tc : Tablecloth), separated_tc.patches = tc.patches \ sp.patches :=
sorry


end star_pattern_identifiable_and_separable_l137_13707


namespace max_brownies_is_100_l137_13716

/-- Represents the dimensions of a rectangular pan of brownies -/
structure BrowniePan where
  m : ℕ+  -- length
  n : ℕ+  -- width

/-- The total number of brownies in the pan -/
def totalBrownies (pan : BrowniePan) : ℕ := pan.m.val * pan.n.val

/-- The number of brownies along the perimeter of the pan -/
def perimeterBrownies (pan : BrowniePan) : ℕ := 2 * (pan.m.val + pan.n.val) - 4

/-- The condition that the total number of brownies is twice the perimeter brownies -/
def validCut (pan : BrowniePan) : Prop :=
  totalBrownies pan = 2 * perimeterBrownies pan

theorem max_brownies_is_100 :
  ∃ (pan : BrowniePan), validCut pan ∧
    (∀ (other : BrowniePan), validCut other → totalBrownies other ≤ totalBrownies pan) ∧
    totalBrownies pan = 100 :=
sorry

end max_brownies_is_100_l137_13716


namespace remainder_three_power_twenty_mod_five_l137_13743

theorem remainder_three_power_twenty_mod_five : 3^20 ≡ 1 [ZMOD 5] := by
  sorry

end remainder_three_power_twenty_mod_five_l137_13743


namespace monthly_parking_rate_l137_13752

/-- Proves that the monthly parking rate is $24 given the specified conditions -/
theorem monthly_parking_rate (weekly_rate : ℕ) (yearly_savings : ℕ) (weeks_per_year : ℕ) (months_per_year : ℕ) :
  weekly_rate = 10 →
  yearly_savings = 232 →
  weeks_per_year = 52 →
  months_per_year = 12 →
  ∃ (monthly_rate : ℕ), monthly_rate = 24 ∧ weeks_per_year * weekly_rate - months_per_year * monthly_rate = yearly_savings :=
by sorry

end monthly_parking_rate_l137_13752


namespace rectangular_field_perimeter_l137_13785

/-- Represents a rectangular field -/
structure RectangularField where
  width : ℝ
  length : ℝ

/-- Calculates the area of a rectangular field -/
def area (field : RectangularField) : ℝ :=
  field.width * field.length

/-- Calculates the perimeter of a rectangular field -/
def perimeter (field : RectangularField) : ℝ :=
  2 * (field.width + field.length)

theorem rectangular_field_perimeter 
  (field : RectangularField) 
  (h_area : area field = 50) 
  (h_width : field.width = 5) : 
  perimeter field = 30 := by
  sorry

#check rectangular_field_perimeter

end rectangular_field_perimeter_l137_13785


namespace sequence_product_l137_13767

/-- An arithmetic sequence with first term -9 and last term -1 -/
def arithmetic_sequence (a₁ a₂ : ℝ) : Prop :=
  ∃ d : ℝ, a₁ = -9 + d ∧ a₂ = a₁ + d ∧ -1 = a₂ + d

/-- A geometric sequence with first term -9 and last term -1 -/
def geometric_sequence (b₁ b₂ b₃ : ℝ) : Prop :=
  ∃ r : ℝ, b₁ = -9 * r ∧ b₂ = b₁ * r ∧ b₃ = b₂ * r ∧ -1 = b₃ * r

theorem sequence_product (a₁ a₂ b₁ b₂ b₃ : ℝ) 
  (h₁ : arithmetic_sequence a₁ a₂)
  (h₂ : geometric_sequence b₁ b₂ b₃) :
  b₂ * (a₂ - a₁) = -8 := by
  sorry

end sequence_product_l137_13767


namespace total_chickens_and_ducks_l137_13795

theorem total_chickens_and_ducks (num_chickens : ℕ) (duck_difference : ℕ) : 
  num_chickens = 45 → 
  duck_difference = 8 → 
  num_chickens + (num_chickens - duck_difference) = 82 :=
by sorry

end total_chickens_and_ducks_l137_13795


namespace greatest_t_value_l137_13723

theorem greatest_t_value (t : ℝ) : 
  (t^2 - t - 56) / (t - 8) = 3 / (t + 5) → t ≤ -4 :=
by sorry

end greatest_t_value_l137_13723


namespace prism_volume_l137_13725

/-- A right rectangular prism with given face areas has the specified volume -/
theorem prism_volume (l w h : ℝ) (h1 : l * w = 15) (h2 : w * h = 10) (h3 : l * h = 6) :
  l * w * h = 30 := by
  sorry

end prism_volume_l137_13725


namespace odd_function_property_l137_13778

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem odd_function_property (f : ℝ → ℝ) (h : is_odd_function f) :
  f 2016 = 2 → f (-2016) = -2 := by
  sorry

end odd_function_property_l137_13778


namespace supplementary_angles_problem_l137_13708

theorem supplementary_angles_problem (x y : ℝ) :
  x + y = 180 ∧ y = x + 18 → x = 81 := by
  sorry

end supplementary_angles_problem_l137_13708


namespace circle_radius_squared_l137_13744

-- Define the circle and points
variable (r : ℝ) -- radius of the circle
variable (A B C D P : ℝ × ℝ) -- points in 2D plane

-- Define the conditions
def AB : ℝ := 12 -- length of chord AB
def CD : ℝ := 8 -- length of chord CD
def BP : ℝ := 9 -- distance from B to P

-- Define the angle condition
def angle_APD_is_right : Prop := sorry

-- Define that P is outside the circle
def P_outside_circle : Prop := sorry

-- Define that AB and CD extended intersect at P
def chords_intersect_at_P : Prop := sorry

-- Theorem statement
theorem circle_radius_squared 
  (h1 : AB = 12)
  (h2 : CD = 8)
  (h3 : BP = 9)
  (h4 : angle_APD_is_right)
  (h5 : P_outside_circle)
  (h6 : chords_intersect_at_P) :
  r^2 = 97.361 := by sorry

end circle_radius_squared_l137_13744


namespace cos_angle_between_vectors_l137_13789

def a : ℝ × ℝ := (3, -1)
def b : ℝ × ℝ := (2, 0)

theorem cos_angle_between_vectors :
  let θ := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  (θ.cos : ℝ) = 3 * Real.sqrt 10 / 10 := by
  sorry

end cos_angle_between_vectors_l137_13789


namespace parking_lot_wheel_count_l137_13735

def parking_lot_wheels (num_cars : ℕ) (num_bikes : ℕ) (wheels_per_car : ℕ) (wheels_per_bike : ℕ) : ℕ :=
  num_cars * wheels_per_car + num_bikes * wheels_per_bike

theorem parking_lot_wheel_count : parking_lot_wheels 14 10 4 2 = 76 := by
  sorry

end parking_lot_wheel_count_l137_13735


namespace angle_sum_is_420_l137_13768

/-- A geometric configuration with six angles A, B, C, D, E, and F -/
structure GeometricConfig where
  A : Real
  B : Real
  C : Real
  D : Real
  E : Real
  F : Real

/-- The theorem stating that if angle E is 30 degrees, then the sum of all angles is 420 degrees -/
theorem angle_sum_is_420 (config : GeometricConfig) (h : config.E = 30) :
  config.A + config.B + config.C + config.D + config.E + config.F = 420 := by
  sorry

#check angle_sum_is_420

end angle_sum_is_420_l137_13768


namespace least_common_denominator_l137_13733

theorem least_common_denominator : 
  Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 8))))) = 840 := by
  sorry

end least_common_denominator_l137_13733


namespace bobby_final_paycheck_l137_13766

/-- Represents Bobby's weekly paycheck calculation -/
def bobby_paycheck (salary : ℝ) (federal_tax_rate : ℝ) (state_tax_rate : ℝ) 
                   (health_insurance : ℝ) (life_insurance : ℝ) (parking_fee : ℝ) : ℝ :=
  salary - (federal_tax_rate * salary) - (state_tax_rate * salary) - 
  health_insurance - life_insurance - parking_fee

/-- Theorem stating that Bobby's final paycheck amount is $184 -/
theorem bobby_final_paycheck : 
  bobby_paycheck 450 (1/3) 0.08 50 20 10 = 184 := by
  sorry

end bobby_final_paycheck_l137_13766


namespace prism_volume_l137_13749

/-- A right rectangular prism with face areas 45, 49, and 56 square units has a volume of 1470 cubic units. -/
theorem prism_volume (a b c : ℝ) (h1 : a * b = 45) (h2 : b * c = 49) (h3 : a * c = 56) :
  a * b * c = 1470 := by
  sorry

end prism_volume_l137_13749


namespace partner_a_share_l137_13756

/-- Calculates the share of profit for a partner in a partnership --/
def calculate_share (investment_a investment_b investment_c profit_b : ℚ) : ℚ :=
  let total_investment := investment_a + investment_b + investment_c
  let total_profit := (profit_b * total_investment) / investment_b
  (investment_a * total_profit) / total_investment

theorem partner_a_share :
  let investment_a : ℚ := 7000
  let investment_b : ℚ := 11000
  let investment_c : ℚ := 18000
  let profit_b : ℚ := 2200
  calculate_share investment_a investment_b investment_c profit_b = 1400 := by
  sorry

#eval calculate_share 7000 11000 18000 2200

end partner_a_share_l137_13756


namespace interior_alternate_angles_equal_implies_parallel_l137_13726

/-- Two lines in a plane -/
structure Line

/-- A transversal line cutting two other lines -/
structure Transversal

/-- An angle formed by the intersection of lines -/
structure Angle

/-- Defines the concept of interior alternate angles -/
def interior_alternate_angles (l1 l2 : Line) (t : Transversal) (a1 a2 : Angle) : Prop :=
  sorry

/-- Defines parallel lines -/
def parallel (l1 l2 : Line) : Prop :=
  sorry

/-- The main theorem: if interior alternate angles are equal, then the lines are parallel -/
theorem interior_alternate_angles_equal_implies_parallel 
  (l1 l2 : Line) (t : Transversal) (a1 a2 : Angle) :
  interior_alternate_angles l1 l2 t a1 a2 → a1 = a2 → parallel l1 l2 :=
sorry

end interior_alternate_angles_equal_implies_parallel_l137_13726


namespace ticket_sales_l137_13709

theorem ticket_sales (total : ℕ) (reduced_first_week : ℕ) (full_price : ℕ) :
  total = 25200 →
  reduced_first_week = 5400 →
  full_price = 5 * reduced_first_week →
  total = reduced_first_week + full_price →
  full_price = 27000 := by
  sorry

end ticket_sales_l137_13709


namespace min_value_expression_l137_13731

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (x^2 / (x + 2)) + (y^2 / (y + 1)) ≥ (1/4 : ℝ) := by
  sorry

end min_value_expression_l137_13731


namespace tory_cookie_sales_l137_13751

theorem tory_cookie_sales (grandmother_packs uncle_packs neighbor_packs more_packs : ℕ) 
  (h1 : grandmother_packs = 12)
  (h2 : uncle_packs = 7)
  (h3 : neighbor_packs = 5)
  (h4 : more_packs = 26) :
  grandmother_packs + uncle_packs + neighbor_packs + more_packs = 50 := by
  sorry

end tory_cookie_sales_l137_13751


namespace triangle_not_right_angle_l137_13745

theorem triangle_not_right_angle (A B C : ℝ) (h1 : A + B + C = 180) 
  (h2 : A = 2 * B) (h3 : A = 3 * C) : A ≠ 90 ∧ B ≠ 90 ∧ C ≠ 90 := by
  sorry

end triangle_not_right_angle_l137_13745


namespace spanish_test_score_difference_l137_13704

theorem spanish_test_score_difference (average_score : ℝ) (marco_percentage : ℝ) (margaret_score : ℝ) :
  average_score = 90 ∧
  marco_percentage = 10 ∧
  margaret_score = 86 →
  margaret_score - (average_score * (1 - marco_percentage / 100)) = 5 := by
  sorry

end spanish_test_score_difference_l137_13704


namespace power_division_equality_l137_13791

theorem power_division_equality (m : ℕ) (h : m = 32^500) : m / 8 = 2^2497 := by
  sorry

end power_division_equality_l137_13791


namespace complex_set_forms_line_l137_13712

/-- The set of complex numbers z such that (2-3i)z is real forms a line in the complex plane -/
theorem complex_set_forms_line : 
  ∃ (m : ℝ) (b : ℝ), 
    {z : ℂ | ∃ (r : ℝ), (2 - 3*I) * z = r} = 
    {z : ℂ | z.im = m * z.re + b} :=
by sorry

end complex_set_forms_line_l137_13712


namespace dart_target_probability_l137_13727

theorem dart_target_probability (n : ℕ) : 
  (n : ℝ) * π / (n : ℝ)^2 ≥ (1 : ℝ) / 2 → n ≤ 6 :=
by
  sorry

end dart_target_probability_l137_13727


namespace least_three_digit_product_6_l137_13788

/-- A function that returns the product of the digits of a natural number -/
def digit_product (n : ℕ) : ℕ := sorry

/-- A function that checks if a number is three-digit -/
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem least_three_digit_product_6 :
  ∀ n : ℕ, is_three_digit n → digit_product n = 6 → 116 ≤ n := by sorry

end least_three_digit_product_6_l137_13788


namespace infinitely_many_disconnected_pLandia_l137_13714

/-- A function that determines if two islands are connected in p-Landia -/
def isConnected (p n m : ℕ) : Prop :=
  p ∣ (n^2 - m + 1) * (m^2 - n + 1)

/-- The graph representation of p-Landia -/
def pLandiaGraph (p : ℕ) : SimpleGraph ℕ :=
  SimpleGraph.fromRel (λ n m ↦ n ≠ m ∧ isConnected p n m)

/-- The theorem stating that infinitely many p-Landia graphs are disconnected -/
theorem infinitely_many_disconnected_pLandia :
  ∃ (S : Set ℕ), (∀ p ∈ S, Nat.Prime p) ∧ Set.Infinite S ∧
    ∀ p ∈ S, ¬(pLandiaGraph p).Connected :=
  sorry

end infinitely_many_disconnected_pLandia_l137_13714


namespace complex_magnitude_l137_13759

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Define the complex number z
def z : ℂ := sorry

-- State the theorem
theorem complex_magnitude (h : z * i^2023 = 1 + i) : Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_magnitude_l137_13759


namespace figure_area_l137_13737

/-- The total area of a figure composed of five rectangles -/
def total_area (a b c d e f g h i j : ℕ) : ℕ :=
  a * b + c * d + e * f + g * h + i * j

theorem figure_area : 
  total_area 7 4 5 4 7 3 5 2 3 1 = 82 := by
  sorry

end figure_area_l137_13737


namespace square_area_possibilities_l137_13738

/-- Represents a square in a 2D plane -/
structure Square where
  side_length : ℝ
  area : ℝ := side_length ^ 2

/-- Represents a parallelogram in a 2D plane -/
structure Parallelogram where
  side1 : ℝ
  side2 : ℝ
  area : ℝ

/-- Represents an oblique projection from a square to a parallelogram -/
def oblique_projection (s : Square) (p : Parallelogram) : Prop :=
  (s.side_length = p.side1 ∨ s.side_length = p.side2) ∧ p.area = s.area

theorem square_area_possibilities (s : Square) (p : Parallelogram) :
  oblique_projection s p → p.side1 = 4 → s.area = 16 ∨ s.area = 64 := by
  sorry

end square_area_possibilities_l137_13738


namespace minimize_area_between_curves_l137_13713

/-- The cubic function C(x) = x^3 - 3x^2 + 2x -/
def C (x : ℝ) : ℝ := x^3 - 3*x^2 + 2*x

/-- The linear function L(x, a) = ax -/
def L (x a : ℝ) : ℝ := a * x

/-- The area S(a) bounded by C and L -/
def S (a : ℝ) : ℝ := sorry

/-- The theorem stating that the value of a minimizing S(a) is 38 - 27√2 -/
theorem minimize_area_between_curves :
  ∃ (a : ℝ), a > -1/4 ∧ ∀ (b : ℝ), b > -1/4 → S a ≤ S b ∧ a = 38 - 27 * Real.sqrt 2 := by
  sorry

end minimize_area_between_curves_l137_13713


namespace complex_fourth_power_integer_count_l137_13764

theorem complex_fourth_power_integer_count : 
  ∃! (n : ℤ), ∃ (m : ℤ), (n + 2 * Complex.I) ^ 4 = m := by sorry

end complex_fourth_power_integer_count_l137_13764


namespace walking_time_l137_13761

/-- Given a walking speed of 10 km/hr and a distance of 4 km, the time taken is 24 minutes. -/
theorem walking_time (speed : ℝ) (distance : ℝ) : 
  speed = 10 → distance = 4 → (distance / speed) * 60 = 24 := by
  sorry

end walking_time_l137_13761


namespace unique_integral_solution_l137_13719

theorem unique_integral_solution :
  ∃! (x y z : ℕ), 
    (z^x = y^(3*x)) ∧ 
    (2^z = 4 * 8^x) ∧ 
    (x + y + z = 20) ∧
    x = 2 ∧ y = 2 ∧ z = 8 := by
  sorry

end unique_integral_solution_l137_13719


namespace integral_equals_sqrt3_over_2_minus_ln2_l137_13721

noncomputable def integral_function (x : ℝ) : ℝ := (Real.cos x)^2 / (1 + Real.cos x - Real.sin x)^2

theorem integral_equals_sqrt3_over_2_minus_ln2 :
  ∫ x in -((2 * Real.pi) / 3)..0, integral_function x = Real.sqrt 3 / 2 - Real.log 2 := by
  sorry

end integral_equals_sqrt3_over_2_minus_ln2_l137_13721


namespace a_most_stable_l137_13755

/-- Represents a participant in the shooting test -/
inductive Participant
  | A
  | B
  | C
  | D

/-- The variance of a participant's scores -/
def variance : Participant → ℝ
  | Participant.A => 0.12
  | Participant.B => 0.25
  | Participant.C => 0.35
  | Participant.D => 0.46

/-- A participant has the most stable performance if their variance is the lowest -/
def hasMostStablePerformance (p : Participant) : Prop :=
  ∀ q : Participant, variance p ≤ variance q

/-- Theorem: Participant A has the most stable performance -/
theorem a_most_stable : hasMostStablePerformance Participant.A := by
  sorry

end a_most_stable_l137_13755


namespace area_bounds_l137_13718

/-- An acute triangle with sides a, b, c and area t, satisfying abc = a + b + c -/
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  t : ℝ
  acute : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b
  area_condition : t > 0
  side_condition : a * b * c = a + b + c

/-- The area of an acute triangle satisfying the given conditions is bounded -/
theorem area_bounds (triangle : AcuteTriangle) : 1 < triangle.t ∧ triangle.t ≤ (3 * Real.sqrt 3) / 4 := by
  sorry

end area_bounds_l137_13718


namespace arithmetic_sequence_sum_l137_13794

/-- An arithmetic sequence with first term a₁ and common difference d -/
def arithmeticSequence (a₁ d : ℚ) : ℕ → ℚ := λ n => a₁ + (n - 1) * d

/-- Sum of the first n terms of an arithmetic sequence -/
def arithmeticSum (a₁ d : ℚ) (n : ℕ) : ℚ := n * a₁ + n * (n - 1) / 2 * d

theorem arithmetic_sequence_sum 
  (a : ℕ → ℚ) 
  (h_arith : ∃ (d : ℚ), ∀ (n : ℕ), a (n + 1) = a n + d) 
  (h_a₁ : a 1 = 1/2) 
  (h_S₂ : arithmeticSum (a 1) (a 2 - a 1) 2 = a 3) :
  ∀ (n : ℕ), arithmeticSum (a 1) (a 2 - a 1) n = 1/4 * n^2 + 1/4 * n :=
sorry

end arithmetic_sequence_sum_l137_13794


namespace principal_amount_l137_13771

/-- Proves that given the specified conditions, the principal amount is 1300 --/
theorem principal_amount (P : ℝ) : 
  P * ((1 + 0.1)^2 - 1) - P * (0.1 * 2) = 13 → P = 1300 := by
  sorry

end principal_amount_l137_13771


namespace S_congruence_l137_13760

def is_valid_N (N : ℕ) : Prop :=
  300 ≤ N ∧ N ≤ 600

def base_4_repr (N : ℕ) : ℕ × ℕ × ℕ :=
  (N / 16, (N / 4) % 4, N % 4)

def base_7_repr (N : ℕ) : ℕ × ℕ × ℕ :=
  (N / 49, (N / 7) % 7, N % 7)

def S (N : ℕ) : ℕ :=
  let (a₁, a₂, a₃) := base_4_repr N
  let (b₁, b₂, b₃) := base_7_repr N
  16 * a₁ + 4 * a₂ + a₃ + 49 * b₁ + 7 * b₂ + b₃

theorem S_congruence (N : ℕ) (h : is_valid_N N) :
  S N % 100 = (3 * N) % 100 ↔ (base_4_repr N).2.2 + (base_7_repr N).2.2 ≡ 3 * N [ZMOD 100] :=
sorry

end S_congruence_l137_13760


namespace smallest_d_for_inverse_l137_13780

def g (x : ℝ) : ℝ := (x - 3)^2 + 4

theorem smallest_d_for_inverse (d : ℝ) : 
  (∀ x y, x ∈ Set.Ici d → y ∈ Set.Ici d → g x = g y → x = y) ∧ 
  (∀ d' < d, ∃ x y, x ∈ Set.Ici d' → y ∈ Set.Ici d' → g x = g y ∧ x ≠ y) ↔ 
  d = 3 :=
sorry

end smallest_d_for_inverse_l137_13780


namespace chris_money_left_l137_13762

/-- Calculates the money left over after purchases given the following conditions:
  * Video game cost: $60
  * Candy cost: $5
  * Babysitting pay rate: $8 per hour
  * Hours worked: 9
-/
def money_left_over (video_game_cost : ℕ) (candy_cost : ℕ) (pay_rate : ℕ) (hours_worked : ℕ) : ℕ :=
  pay_rate * hours_worked - (video_game_cost + candy_cost)

theorem chris_money_left : money_left_over 60 5 8 9 = 7 := by
  sorry

end chris_money_left_l137_13762


namespace discount_calculation_l137_13739

/-- Proves that given a list price of 70, a final price of 59.85, and two successive discounts
    where one is 10%, the other discount percentage is 5%. -/
theorem discount_calculation (list_price : ℝ) (final_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) :
  list_price = 70 →
  final_price = 59.85 →
  discount1 = 10 →
  final_price = list_price * (1 - discount1 / 100) * (1 - discount2 / 100) →
  discount2 = 5 := by
  sorry

end discount_calculation_l137_13739


namespace beta_max_success_ratio_l137_13734

theorem beta_max_success_ratio 
  (alpha_day1_score alpha_day1_total : ℕ)
  (alpha_day2_score alpha_day2_total : ℕ)
  (beta_day1_score beta_day1_total : ℕ)
  (beta_day2_score beta_day2_total : ℕ)
  (h1 : alpha_day1_score = 160)
  (h2 : alpha_day1_total = 300)
  (h3 : alpha_day2_score = 140)
  (h4 : alpha_day2_total = 200)
  (h5 : beta_day1_total + beta_day2_total = 500)
  (h6 : beta_day1_total ≠ 300)
  (h7 : beta_day1_score > 0)
  (h8 : beta_day2_score > 0)
  (h9 : (beta_day1_score : ℚ) / beta_day1_total < (alpha_day1_score : ℚ) / alpha_day1_total)
  (h10 : (beta_day2_score : ℚ) / beta_day2_total < (alpha_day2_score : ℚ) / alpha_day2_total)
  (h11 : (alpha_day1_score + alpha_day2_score : ℚ) / (alpha_day1_total + alpha_day2_total) = 3/5) :
  (beta_day1_score + beta_day2_score : ℚ) / (beta_day1_total + beta_day2_total) ≤ 349/500 :=
by sorry

end beta_max_success_ratio_l137_13734


namespace qr_length_l137_13758

/-- Triangle DEF with given side lengths -/
structure Triangle where
  DE : ℝ
  EF : ℝ
  DF : ℝ

/-- Circle with center and two points it passes through -/
structure Circle where
  center : ℝ × ℝ
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- The problem setup -/
def ProblemSetup (t : Triangle) (c1 c2 : Circle) : Prop :=
  t.DE = 7 ∧ t.EF = 24 ∧ t.DF = 25 ∧
  c1.center.1 = c1.point1.1 ∧ -- Q is on the same vertical line as D
  c1.point2 = (t.DF, 0) ∧ -- F is at (25, 0)
  c2.center.2 = c2.point1.2 ∧ -- R is on the same horizontal line as E
  c2.point2 = (0, 0) -- D is at (0, 0)

theorem qr_length (t : Triangle) (c1 c2 : Circle) 
  (h : ProblemSetup t c1 c2) : 
  ‖c1.center - c2.center‖ = 8075 / 84 := by
  sorry

end qr_length_l137_13758


namespace inequality_proof_l137_13763

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (a + c))) + (1 / (c^3 * (a + b))) ≥ 3/2 := by
  sorry

end inequality_proof_l137_13763


namespace area_of_second_square_l137_13790

-- Define the circle
def Circle : Type := Unit

-- Define squares
structure Square where
  area : ℝ

-- Define the inscribed square
def inscribed_square (c : Circle) (s : Square) : Prop :=
  s.area = 16

-- Define the second square
def second_square (c : Circle) (s1 s2 : Square) : Prop :=
  -- Vertices E and F are on sides of s1, G and H are on the circle
  True

-- Theorem statement
theorem area_of_second_square 
  (c : Circle) 
  (s1 s2 : Square) 
  (h1 : inscribed_square c s1) 
  (h2 : second_square c s1 s2) : 
  s2.area = 8 := by
  sorry

end area_of_second_square_l137_13790


namespace liquid_x_percentage_in_mixed_solution_l137_13720

/-- Proves that mixing solutions A and B results in a solution with approximately 1.44% liquid X -/
theorem liquid_x_percentage_in_mixed_solution :
  let solution_a_weight : ℝ := 400
  let solution_b_weight : ℝ := 700
  let liquid_x_percent_a : ℝ := 0.8
  let liquid_x_percent_b : ℝ := 1.8
  let total_weight := solution_a_weight + solution_b_weight
  let liquid_x_weight_a := solution_a_weight * (liquid_x_percent_a / 100)
  let liquid_x_weight_b := solution_b_weight * (liquid_x_percent_b / 100)
  let total_liquid_x_weight := liquid_x_weight_a + liquid_x_weight_b
  let result_percent := (total_liquid_x_weight / total_weight) * 100
  ∃ ε > 0, |result_percent - 1.44| < ε :=
by
  sorry

end liquid_x_percentage_in_mixed_solution_l137_13720


namespace exactly_one_project_not_selected_l137_13772

/-- The number of employees and projects -/
def n : ℕ := 4

/-- The probability of exactly one project not being selected -/
def probability : ℚ := 9/16

/-- Theorem stating the probability of exactly one project not being selected -/
theorem exactly_one_project_not_selected :
  (n : ℚ)^n * probability = (n.choose 2) * n! :=
sorry

end exactly_one_project_not_selected_l137_13772


namespace inverse_variation_problem_l137_13703

/-- Two quantities vary inversely if their product is constant -/
def vary_inversely (a b : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, a x * b x = k

theorem inverse_variation_problem (a b : ℝ → ℝ) 
  (h_inverse : vary_inversely a b)
  (h_initial : b 800 = 0.5) :
  b 3200 = 0.125 := by
  sorry

end inverse_variation_problem_l137_13703


namespace count_twos_in_hotel_l137_13736

/-- Represents a hotel room number -/
structure RoomNumber where
  floor : Nat
  room : Nat
  h1 : 1 ≤ floor ∧ floor ≤ 5
  h2 : 1 ≤ room ∧ room ≤ 35

/-- Counts occurrences of a digit in a natural number -/
def countDigit (digit : Nat) (n : Nat) : Nat :=
  sorry

/-- All room numbers in the hotel -/
def allRoomNumbers : List RoomNumber :=
  sorry

/-- Counts occurrences of digit 2 in all room numbers -/
def countTwos : Nat :=
  sorry

theorem count_twos_in_hotel : countTwos = 105 := by
  sorry

end count_twos_in_hotel_l137_13736


namespace fraction_simplification_l137_13798

theorem fraction_simplification (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) :
  (x - 2) / (x^2 - 2*x + 1) / (x / (x - 1)) + 1 / (x^2 - x) = 1 / x := by
  sorry

end fraction_simplification_l137_13798


namespace diamond_club_evaluation_l137_13769

-- Define the diamond operation
def diamond (a b : ℚ) : ℚ := (3 * a + b) / (a - b)

-- Define the club operation
def club (a b : ℚ) : ℚ := 2

-- Theorem statement
theorem diamond_club_evaluation :
  club (diamond 4 6) (diamond 7 5) = 2 := by sorry

end diamond_club_evaluation_l137_13769


namespace initial_sets_count_l137_13706

/-- The number of letters available (A through J) -/
def n : ℕ := 10

/-- The number of letters in each set of initials -/
def k : ℕ := 3

/-- The number of different three-letter sets of initials possible -/
def num_initial_sets : ℕ := n * (n - 1) * (n - 2)

/-- Theorem stating that the number of different three-letter sets of initials
    using letters A through J, with no repeated letters, is equal to 720 -/
theorem initial_sets_count : num_initial_sets = 720 := by
  sorry

end initial_sets_count_l137_13706


namespace tan_ratio_equals_two_l137_13775

theorem tan_ratio_equals_two (a β : ℝ) (h : 3 * Real.sin β = Real.sin (2 * a + β)) :
  Real.tan (a + β) / Real.tan a = 2 := by sorry

end tan_ratio_equals_two_l137_13775


namespace jellybean_ratio_l137_13796

/-- Proves that the ratio of Sophie's jellybeans to Caleb's jellybeans is 1:2 -/
theorem jellybean_ratio (caleb_dozens : ℕ) (total : ℕ) : 
  caleb_dozens = 3 → total = 54 → 
  (total - caleb_dozens * 12) / (caleb_dozens * 12) = 1 / 2 := by
sorry

end jellybean_ratio_l137_13796


namespace sqrt_square_abs_l137_13710

theorem sqrt_square_abs (x : ℝ) : Real.sqrt (x^2) = |x| := by
  sorry

end sqrt_square_abs_l137_13710
