import Mathlib

namespace number_manipulation_l2129_212922

theorem number_manipulation (x : ℝ) : (x - 5) / 7 = 7 → (x - 14) / 10 = 4 := by
  sorry

end number_manipulation_l2129_212922


namespace jerrys_action_figures_l2129_212991

/-- The problem of Jerry's action figures --/
theorem jerrys_action_figures 
  (final_count : ℕ) 
  (added_count : ℕ) 
  (h1 : final_count = 10) 
  (h2 : added_count = 2) :
  final_count - added_count = 8 := by
  sorry

end jerrys_action_figures_l2129_212991


namespace sqrt_sum_equality_l2129_212943

theorem sqrt_sum_equality : Real.sqrt 50 + Real.sqrt 72 = 11 * Real.sqrt 2 := by
  sorry

end sqrt_sum_equality_l2129_212943


namespace base_conversion_equality_l2129_212998

/-- Given that 10b1₍₂₎ = a02₍₃₎, b ∈ {0, 1}, and a ∈ {0, 1, 2}, prove that a = 1 and b = 1 -/
theorem base_conversion_equality (a b : ℕ) : 
  (1 + 2 * b + 8 = 2 + 9 * a) → 
  (b = 0 ∨ b = 1) → 
  (a = 0 ∨ a = 1 ∨ a = 2) → 
  (a = 1 ∧ b = 1) := by
sorry

end base_conversion_equality_l2129_212998


namespace regular_polygon_not_unique_by_circumradius_triangle_not_unique_by_circumradius_l2129_212915

/-- A regular polygon -/
structure RegularPolygon where
  /-- The number of sides in the polygon -/
  sides : ℕ
  /-- The radius of the circumscribed circle -/
  circumRadius : ℝ
  /-- Assertion that the number of sides is at least 3 -/
  sidesGe3 : sides ≥ 3

/-- Theorem stating that a regular polygon is not uniquely determined by its circumradius -/
theorem regular_polygon_not_unique_by_circumradius :
  ∃ (p q : RegularPolygon), p.circumRadius = q.circumRadius ∧ p.sides ≠ q.sides :=
sorry

/-- Corollary specifically for triangles -/
theorem triangle_not_unique_by_circumradius :
  ∃ (t : RegularPolygon) (p : RegularPolygon), 
    t.sides = 3 ∧ p.sides ≠ 3 ∧ t.circumRadius = p.circumRadius :=
sorry

end regular_polygon_not_unique_by_circumradius_triangle_not_unique_by_circumradius_l2129_212915


namespace min_cubes_for_given_box_l2129_212937

/-- Calculates the minimum number of cubes required to build a box -/
def min_cubes_for_box (length width height cube_volume : ℕ) : ℕ :=
  (length * width * height + cube_volume - 1) / cube_volume

/-- Theorem stating the minimum number of cubes required for the given box -/
theorem min_cubes_for_given_box :
  min_cubes_for_box 8 15 5 10 = 60 := by
  sorry

end min_cubes_for_given_box_l2129_212937


namespace triangle_angle_measure_l2129_212923

theorem triangle_angle_measure (a b : ℝ) (A B : ℝ) :
  a > 0 → b > 0 → 0 < A → A < π → 0 < B → B < π →
  Real.sqrt 3 * a = 2 * b * Real.sin A →
  B = π / 3 := by
  sorry

end triangle_angle_measure_l2129_212923


namespace algebraic_expression_value_l2129_212957

theorem algebraic_expression_value (p q : ℝ) :
  (2^3 * p + 2 * q + 1 = -2022) → ((-2)^3 * p + (-2) * q + 1 = 2024) := by
  sorry

end algebraic_expression_value_l2129_212957


namespace distance_AB_is_360_l2129_212910

/-- The distance between two points A and B --/
def distance_AB : ℝ := sorry

/-- The initial speed of the passenger train --/
def v_pass : ℝ := sorry

/-- The initial speed of the freight train --/
def v_freight : ℝ := sorry

/-- The time taken by the freight train to travel from A to B --/
def t_freight : ℝ := sorry

/-- The time difference between the passenger and freight trains --/
def time_diff : ℝ := 3.2

/-- The additional distance traveled by the passenger train --/
def additional_distance : ℝ := 288

/-- The speed increase for both trains --/
def speed_increase : ℝ := 10

/-- The new time difference after speed increase --/
def new_time_diff : ℝ := 2.4

theorem distance_AB_is_360 :
  v_pass * (t_freight - time_diff) = v_freight * t_freight + additional_distance ∧
  distance_AB / (v_freight + speed_increase) - distance_AB / (v_pass + speed_increase) = new_time_diff ∧
  distance_AB = v_freight * t_freight →
  distance_AB = 360 := by sorry

end distance_AB_is_360_l2129_212910


namespace tan_45_degrees_equals_one_l2129_212949

theorem tan_45_degrees_equals_one : Real.tan (π / 4) = 1 := by
  sorry

end tan_45_degrees_equals_one_l2129_212949


namespace system_solution_1_l2129_212918

theorem system_solution_1 (x y : ℝ) :
  x + y = 10^20 ∧ x - y = 10^19 → x = 55 * 10^18 ∧ y = 45 * 10^18 := by
  sorry

end system_solution_1_l2129_212918


namespace inverse_proportion_l2129_212909

theorem inverse_proportion (x y : ℝ → ℝ) (k : ℝ) :
  (∀ t, x t * y t = k) →  -- x is inversely proportional to y
  x 2 = 4 →               -- x = 4 when y = 2
  y 2 = 2 →               -- y = 2 when x = 4
  y (-5) = -5 →           -- y = -5
  x (-5) = -8/5 :=        -- x = -8/5 when y = -5
by
  sorry


end inverse_proportion_l2129_212909


namespace tech_club_theorem_l2129_212960

/-- The number of students in the tech club who take neither coding nor robotics -/
def students_taking_neither (total : ℕ) (coding : ℕ) (robotics : ℕ) (both : ℕ) : ℕ :=
  total - (coding + robotics - both)

/-- Theorem: Given the conditions from the problem, 20 students take neither coding nor robotics -/
theorem tech_club_theorem :
  students_taking_neither 150 80 70 20 = 20 := by
  sorry

end tech_club_theorem_l2129_212960


namespace box_fill_rate_l2129_212913

-- Define the box dimensions
def box_length : ℝ := 7
def box_width : ℝ := 6
def box_height : ℝ := 2

-- Define the time to fill the box
def fill_time : ℝ := 21

-- Calculate the volume of the box
def box_volume : ℝ := box_length * box_width * box_height

-- Define the theorem
theorem box_fill_rate :
  box_volume / fill_time = 4 := by sorry

end box_fill_rate_l2129_212913


namespace squirrel_walnut_theorem_l2129_212985

/-- Calculates the final number of walnuts after squirrel activities -/
def final_walnut_count (initial : ℕ) (boy_gathered : ℕ) (boy_dropped : ℕ) (girl_brought : ℕ) (girl_ate : ℕ) : ℕ :=
  initial + boy_gathered - boy_dropped + girl_brought - girl_ate

/-- Theorem stating that given the squirrel activities, the final walnut count is 20 -/
theorem squirrel_walnut_theorem : 
  final_walnut_count 12 6 1 5 2 = 20 := by
  sorry

#eval final_walnut_count 12 6 1 5 2

end squirrel_walnut_theorem_l2129_212985


namespace percentage_problem_l2129_212925

theorem percentage_problem (x : ℝ) (h1 : x > 0) (h2 : (x / 100) * x = 2 * x) : x = 200 := by
  sorry

end percentage_problem_l2129_212925


namespace max_candy_leftover_l2129_212990

theorem max_candy_leftover (x : ℕ) : ∃ (q : ℕ), x = 11 * q + 10 ∧ ∀ (r : ℕ), r < 11 → x ≠ 11 * q + r + 1 :=
sorry

end max_candy_leftover_l2129_212990


namespace num_monomials_for_problem_exponent_l2129_212958

/-- The number of monomials with nonzero coefficients in the expansion of (x+y+z)^n + (x-y-z)^n -/
def num_monomials (n : ℕ) : ℕ :=
  (n / 2 + 1) ^ 2

/-- The given exponent in the problem -/
def problem_exponent : ℕ := 2032

theorem num_monomials_for_problem_exponent :
  num_monomials problem_exponent = 1034289 := by
  sorry

end num_monomials_for_problem_exponent_l2129_212958


namespace student_weight_is_90_l2129_212993

/-- The student's weight in kilograms -/
def student_weight : ℝ := sorry

/-- The sister's weight in kilograms -/
def sister_weight : ℝ := sorry

/-- The combined weight of the student and his sister in kilograms -/
def combined_weight : ℝ := 132

/-- If the student loses 6 kilograms, he will weigh twice as much as his sister -/
axiom weight_relation : student_weight - 6 = 2 * sister_weight

/-- The combined weight of the student and his sister is 132 kilograms -/
axiom total_weight : student_weight + sister_weight = combined_weight

/-- Theorem: The student's present weight is 90 kilograms -/
theorem student_weight_is_90 : student_weight = 90 := by sorry

end student_weight_is_90_l2129_212993


namespace train_passing_jogger_time_l2129_212941

/-- The time taken for a train to pass a jogger under specific conditions -/
theorem train_passing_jogger_time : 
  let jogger_speed : ℝ := 9 -- km/hr
  let train_speed : ℝ := 45 -- km/hr
  let initial_distance : ℝ := 240 -- meters
  let train_length : ℝ := 120 -- meters

  let jogger_speed_ms : ℝ := jogger_speed * 1000 / 3600 -- Convert to m/s
  let train_speed_ms : ℝ := train_speed * 1000 / 3600 -- Convert to m/s
  let relative_speed : ℝ := train_speed_ms - jogger_speed_ms
  let total_distance : ℝ := initial_distance + train_length
  let time : ℝ := total_distance / relative_speed

  time = 36 := by sorry

end train_passing_jogger_time_l2129_212941


namespace number_of_students_l2129_212927

/-- Proves the number of students in a class given average ages and teacher's age -/
theorem number_of_students (avg_age : ℝ) (teacher_age : ℝ) (new_avg_age : ℝ) :
  avg_age = 22 →
  teacher_age = 46 →
  new_avg_age = 23 →
  (avg_age * n + teacher_age) / (n + 1) = new_avg_age →
  n = 23 :=
by
  sorry

end number_of_students_l2129_212927


namespace friday_fries_ratio_l2129_212980

/-- Represents the number of fries sold -/
structure FriesSold where
  total : ℕ
  small : ℕ

/-- Calculates the ratio of large fries to small fries -/
def largeToSmallRatio (fs : FriesSold) : ℚ :=
  (fs.total - fs.small : ℚ) / fs.small

theorem friday_fries_ratio :
  let fs : FriesSold := { total := 24, small := 4 }
  largeToSmallRatio fs = 5 := by
  sorry

end friday_fries_ratio_l2129_212980


namespace problem_1_problem_2_problem_3_problem_4_problem_5_l2129_212965

-- (1) Prove that x = 3 or x = -1 is a solution to 4(x-1)^2 - 16 = 0
theorem problem_1 : ∃ x : ℝ, (x = 3 ∨ x = -1) ∧ 4 * (x - 1)^2 - 16 = 0 := by sorry

-- (2) Prove that ∛(-64) + √16 * √(9/4) + (-√2)^2 = 4
theorem problem_2 : ((-64 : ℝ)^(1/3)) + Real.sqrt 16 * Real.sqrt (9/4) + (-Real.sqrt 2)^2 = 4 := by sorry

-- (3) Prove that if a is the integer part and b is the decimal part of 9 - √13, then 2a + b = 14 - √13
theorem problem_3 (a b : ℝ) (h : a = ⌊9 - Real.sqrt 13⌋ ∧ b = 9 - Real.sqrt 13 - a) :
  2 * a + b = 14 - Real.sqrt 13 := by sorry

-- (4) Define an operation ⊕ and prove that x = 5 or x = -5 is a solution to (4 ⊕ 3) ⊕ x = 24
def circle_plus (a b : ℝ) : ℝ := a^2 - b^2

theorem problem_4 : ∃ x : ℝ, (x = 5 ∨ x = -5) ∧ circle_plus (circle_plus 4 3) x = 24 := by sorry

-- (5) Prove that if ∠1 and ∠2 are parallel, and ∠1 is 36° less than three times ∠2, then ∠1 = 18° or ∠1 = 126°
theorem problem_5 (angle1 angle2 : ℝ) 
  (h1 : angle1 = 3 * angle2 - 36)
  (h2 : angle1 = angle2 ∨ angle1 + angle2 = 180) :
  angle1 = 18 ∨ angle1 = 126 := by sorry

end problem_1_problem_2_problem_3_problem_4_problem_5_l2129_212965


namespace min_value_m_min_value_m_tight_l2129_212931

theorem min_value_m (m : ℝ) : 
  (∀ x ∈ Set.Icc 0 (π/3), m ≥ 2 * Real.tan x) → m ≥ 2 * Real.sqrt 3 :=
by sorry

theorem min_value_m_tight : 
  ∃ m : ℝ, (∀ x ∈ Set.Icc 0 (π/3), m ≥ 2 * Real.tan x) ∧ m = 2 * Real.sqrt 3 :=
by sorry

end min_value_m_min_value_m_tight_l2129_212931


namespace smallest_n_for_roots_of_unity_l2129_212970

theorem smallest_n_for_roots_of_unity : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (z : ℂ), z^6 - z^3 + 1 = 0 → z^n = 1) ∧
  (∀ (m : ℕ), m > 0 → (∀ (z : ℂ), z^6 - z^3 + 1 = 0 → z^m = 1) → m ≥ n) ∧
  n = 18 := by
  sorry

end smallest_n_for_roots_of_unity_l2129_212970


namespace marc_watching_friends_l2129_212902

theorem marc_watching_friends (total_episodes : ℕ) (watch_fraction : ℚ) (days : ℕ) : 
  total_episodes = 50 → 
  watch_fraction = 1 / 10 → 
  (total_episodes : ℚ) * watch_fraction = (days : ℚ) → 
  days = 10 := by
sorry

end marc_watching_friends_l2129_212902


namespace log_relationship_l2129_212921

theorem log_relationship (a b : ℝ) : 
  a = Real.log 256 / Real.log 8 → b = Real.log 16 / Real.log 2 → a = (2 * b) / 3 := by
sorry

end log_relationship_l2129_212921


namespace commercial_time_l2129_212947

theorem commercial_time (p : ℝ) (h : p = 0.9) : (1 - p) * 60 = 6 := by
  sorry

end commercial_time_l2129_212947


namespace geometric_sequence_20th_term_l2129_212939

/-- A geometric sequence is defined by its first term and common ratio -/
def geometric_sequence (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r^(n - 1)

/-- Theorem: In a geometric sequence where the 5th term is 5 and the 12th term is 1280, the 20th term is 2621440 -/
theorem geometric_sequence_20th_term 
  (a : ℝ) (r : ℝ) 
  (h1 : geometric_sequence a r 5 = 5)
  (h2 : geometric_sequence a r 12 = 1280) :
  geometric_sequence a r 20 = 2621440 := by
  sorry


end geometric_sequence_20th_term_l2129_212939


namespace common_divisors_9240_10800_l2129_212916

theorem common_divisors_9240_10800 : Nat.card {d : ℕ | d ∣ 9240 ∧ d ∣ 10800} = 16 := by
  sorry

end common_divisors_9240_10800_l2129_212916


namespace three_planes_division_l2129_212982

theorem three_planes_division (x y : ℕ) : 
  (x = 4 ∧ y = 8) → y - x = 4 := by
  sorry

end three_planes_division_l2129_212982


namespace units_digit_of_power_product_l2129_212900

theorem units_digit_of_power_product : 2^1201 * 4^1302 * 6^1403 ≡ 2 [ZMOD 10] := by sorry

end units_digit_of_power_product_l2129_212900


namespace triarc_area_sum_l2129_212994

/-- A region bounded by three circular arcs -/
structure TriarcRegion where
  radius : ℝ
  central_angle : ℝ

/-- The area of a TriarcRegion in the form a√b + cπ -/
structure TriarcArea where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Compute the area of a TriarcRegion -/
noncomputable def compute_triarc_area (region : TriarcRegion) : TriarcArea :=
  sorry

theorem triarc_area_sum (region : TriarcRegion) 
  (h1 : region.radius = 5)
  (h2 : region.central_angle = 2 * π / 3) : 
  let area := compute_triarc_area region
  area.a + area.b + area.c = -28.25 := by
  sorry

end triarc_area_sum_l2129_212994


namespace least_n_factorial_divisible_by_1029_l2129_212955

theorem least_n_factorial_divisible_by_1029 : 
  ∃ n : ℕ, n = 21 ∧ 
  (∀ k : ℕ, k < n → ¬(1029 ∣ k!)) ∧ 
  (1029 ∣ n!) := by
  sorry

end least_n_factorial_divisible_by_1029_l2129_212955


namespace scientific_notation_1300000_l2129_212973

theorem scientific_notation_1300000 : 
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 1300000 = a * (10 : ℝ) ^ n ∧ a = 1.3 ∧ n = 6 := by
  sorry

end scientific_notation_1300000_l2129_212973


namespace beef_purchase_l2129_212924

theorem beef_purchase (initial_budget : ℕ) (chicken_cost : ℕ) (beef_cost_per_pound : ℕ) (remaining_budget : ℕ)
  (h1 : initial_budget = 80)
  (h2 : chicken_cost = 12)
  (h3 : beef_cost_per_pound = 3)
  (h4 : remaining_budget = 53) :
  (initial_budget - remaining_budget - chicken_cost) / beef_cost_per_pound = 5 := by
  sorry

end beef_purchase_l2129_212924


namespace sum_of_x_and_y_l2129_212907

theorem sum_of_x_and_y (a x y : ℝ) (hx : a / x = 1 / 3) (hy : a / y = 1 / 4) :
  x + y = 7 * a := by
  sorry

end sum_of_x_and_y_l2129_212907


namespace expression_simplification_l2129_212961

theorem expression_simplification (m : ℝ) (h : m = Real.sqrt 3 - 2) :
  (m^2 - 4*m + 4) / (m - 1) / ((3 / (m - 1)) - m - 1) = (-3 + 4 * Real.sqrt 3) / 3 :=
by sorry

end expression_simplification_l2129_212961


namespace infinite_power_tower_four_implies_sqrt_two_l2129_212999

/-- The limit of the infinite power tower x^(x^(x^...)) -/
noncomputable def infinitePowerTower (x : ℝ) : ℝ :=
  Real.log x / Real.log (Real.log x)

/-- Theorem: If the infinite power tower of x equals 4, then x equals √2 -/
theorem infinite_power_tower_four_implies_sqrt_two (x : ℝ) 
  (h_pos : x > 0) 
  (h_converge : infinitePowerTower x = 4) : 
  x = Real.sqrt 2 :=
sorry

end infinite_power_tower_four_implies_sqrt_two_l2129_212999


namespace rectangle_area_l2129_212983

theorem rectangle_area (width length : ℝ) (h1 : length = width + 6) (h2 : 2 * (width + length) = 68) :
  width * length = 280 :=
by sorry

end rectangle_area_l2129_212983


namespace smallest_two_digit_with_digit_product_12_l2129_212972

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ := (n / 10) * (n % 10)

theorem smallest_two_digit_with_digit_product_12 :
  ∃ (n : ℕ), is_two_digit n ∧ digit_product n = 12 ∧
  ∀ (m : ℕ), is_two_digit m → digit_product m = 12 → n ≤ m :=
by
  sorry

end smallest_two_digit_with_digit_product_12_l2129_212972


namespace root_sum_squares_l2129_212979

theorem root_sum_squares (r s : ℝ) (α β γ δ : ℂ) : 
  (α^2 - r*α - 2 = 0) → 
  (β^2 - r*β - 2 = 0) → 
  (γ^2 + s*γ - 2 = 0) → 
  (δ^2 + s*δ - 2 = 0) → 
  (α - γ)^2 + (β - γ)^2 + (α + δ)^2 + (β + δ)^2 = 4*s*(r - s) + 8 := by
  sorry

end root_sum_squares_l2129_212979


namespace ice_cream_volume_l2129_212940

/-- The volume of ice cream in a cone and hemisphere -/
theorem ice_cream_volume (h : ℝ) (r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let cone_volume := (1 / 3) * π * r^2 * h
  let hemisphere_volume := (2 / 3) * π * r^3
  h = 10 ∧ r = 3 →
  cone_volume + hemisphere_volume = 48 * π := by
sorry

end ice_cream_volume_l2129_212940


namespace galaxy_planets_l2129_212997

theorem galaxy_planets (total : ℕ) (ratio : ℕ) (h1 : total = 200) (h2 : ratio = 8) : 
  ∃ (planets : ℕ), planets * (ratio + 1) = total ∧ planets = 22 := by
  sorry

end galaxy_planets_l2129_212997


namespace circle_line_intersection_l2129_212930

theorem circle_line_intersection (α β : ℝ) (n k : ℤ) : 
  (∃ A B : ℝ × ℝ, 
    A = (Real.cos (2 * α), Real.cos (2 * β)) ∧ 
    B = (Real.cos (2 * β), Real.cos α) ∧
    (A = (-1/2, 0) ∧ B = (0, -1/2) ∨ A = (0, -1/2) ∧ B = (-1/2, 0))) →
  (α = 2 * Real.pi / 3 + 2 * Real.pi * ↑n ∨ 
   α = -2 * Real.pi / 3 + 2 * Real.pi * ↑n) ∧
  β = Real.pi / 4 + Real.pi / 2 * ↑k :=
by sorry

end circle_line_intersection_l2129_212930


namespace sine_value_from_tangent_cosine_relation_l2129_212988

theorem sine_value_from_tangent_cosine_relation (θ : Real) 
  (h1 : 8 * Real.tan θ = 3 * Real.cos θ) 
  (h2 : 0 < θ) (h3 : θ < Real.pi) : 
  Real.sin θ = 1/3 := by
  sorry

end sine_value_from_tangent_cosine_relation_l2129_212988


namespace simon_change_calculation_l2129_212987

def pansy_price : ℝ := 2.50
def pansy_quantity : ℕ := 5
def hydrangea_price : ℝ := 12.50
def hydrangea_quantity : ℕ := 1
def petunia_price : ℝ := 1.00
def petunia_quantity : ℕ := 5
def discount_rate : ℝ := 0.10
def paid_amount : ℝ := 50.00

theorem simon_change_calculation :
  let total_before_discount := pansy_price * pansy_quantity + hydrangea_price * hydrangea_quantity + petunia_price * petunia_quantity
  let discount := total_before_discount * discount_rate
  let total_after_discount := total_before_discount - discount
  let change := paid_amount - total_after_discount
  change = 23.00 := by sorry

end simon_change_calculation_l2129_212987


namespace club_member_selection_l2129_212981

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of members in the club -/
def totalMembers : ℕ := 15

/-- The number of members to be chosen -/
def chosenMembers : ℕ := 4

/-- The number of remaining members after excluding the two specific members -/
def remainingMembers : ℕ := totalMembers - 2

theorem club_member_selection :
  choose totalMembers chosenMembers - choose remainingMembers (chosenMembers - 2) = 1287 := by
  sorry

end club_member_selection_l2129_212981


namespace line_segment_endpoint_l2129_212954

theorem line_segment_endpoint (x : ℝ) : x > 0 ∧ 
  Real.sqrt ((x - 2)^2 + (5 - 2)^2) = 8 → x = 2 + Real.sqrt 55 := by
  sorry

end line_segment_endpoint_l2129_212954


namespace quadratic_real_roots_condition_l2129_212933

/-- 
For a quadratic equation x^2 + x + m = 0 with m ∈ ℝ, 
the condition "m > 1/4" is neither sufficient nor necessary for real roots.
-/
theorem quadratic_real_roots_condition (m : ℝ) : 
  ¬(∀ x : ℝ, x^2 + x + m = 0 → m > 1/4) ∧ 
  ¬(m > 1/4 → ∃ x : ℝ, x^2 + x + m = 0) := by
  sorry

end quadratic_real_roots_condition_l2129_212933


namespace complex_magnitude_l2129_212934

theorem complex_magnitude (z : ℂ) (h : Complex.I * z = 4 - 2 * Complex.I) : 
  Complex.abs z = 2 * Real.sqrt 5 := by
  sorry

end complex_magnitude_l2129_212934


namespace complex_equation_sum_l2129_212914

theorem complex_equation_sum (x y : ℝ) :
  (x / (1 - Complex.I)) + (y / (1 - 2 * Complex.I)) = 5 / (1 - 3 * Complex.I) →
  x + y = 4 := by
  sorry

end complex_equation_sum_l2129_212914


namespace gcd_1989_1547_l2129_212929

theorem gcd_1989_1547 : Nat.gcd 1989 1547 = 221 := by
  sorry

end gcd_1989_1547_l2129_212929


namespace unique_solution_is_zero_l2129_212908

theorem unique_solution_is_zero : 
  ∃! x : ℝ, (3 : ℝ) / (x - 3) = (5 : ℝ) / (x - 5) :=
by
  -- Proof goes here
  sorry

end unique_solution_is_zero_l2129_212908


namespace angle_triple_complement_l2129_212995

theorem angle_triple_complement (x : ℝ) : 
  (x = 3 * (90 - x)) → x = 67.5 := by
  sorry

end angle_triple_complement_l2129_212995


namespace max_value_of_g_l2129_212936

/-- The function g(x) = 4x - x^4 -/
def g (x : ℝ) : ℝ := 4 * x - x^4

/-- The interval [0, 2] -/
def I : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}

theorem max_value_of_g :
  ∃ (m : ℝ), m = 3 ∧ ∀ (x : ℝ), x ∈ I → g x ≤ m :=
sorry

end max_value_of_g_l2129_212936


namespace point_not_above_curve_l2129_212906

theorem point_not_above_curve :
  ∀ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 →
  ¬(b > a * b^3 - b * b^2) := by
  sorry

end point_not_above_curve_l2129_212906


namespace traffic_light_probability_l2129_212996

theorem traffic_light_probability : 
  let p_A : ℚ := 25 / 60
  let p_B : ℚ := 35 / 60
  let p_C : ℚ := 45 / 60
  p_A * p_B * p_C = 35 / 192 := by
sorry

end traffic_light_probability_l2129_212996


namespace basketball_activity_results_l2129_212959

/-- Represents the outcome of a shot -/
inductive ShotResult
| Hit
| Miss

/-- Represents the game state -/
inductive GameState
| InProgress
| Cleared
| Failed

/-- Represents the possible coupon amounts -/
inductive CouponAmount
| Three
| Six
| Nine

/-- The shooting accuracy of Xiao Ming -/
def accuracy : ℚ := 2/3

/-- Updates the game state based on the current state and the new shot result -/
def updateGameState (state : GameState) (shot : ShotResult) : GameState :=
  sorry

/-- Simulates the game for a given number of shots -/
def simulateGame (n : ℕ) : GameState :=
  sorry

/-- Calculates the probability of ending the game after exactly 5 shots -/
def probEndAfterFiveShots : ℚ :=
  sorry

/-- Represents the distribution of the coupon amount -/
def couponDistribution : CouponAmount → ℚ :=
  sorry

/-- Calculates the expectation of the coupon amount -/
def expectedCouponAmount : ℚ :=
  sorry

theorem basketball_activity_results :
  probEndAfterFiveShots = 8/81 ∧
  couponDistribution CouponAmount.Three = 233/729 ∧
  couponDistribution CouponAmount.Six = 112/729 ∧
  couponDistribution CouponAmount.Nine = 128/243 ∧
  expectedCouponAmount = 1609/243 :=
sorry

end basketball_activity_results_l2129_212959


namespace total_calories_l2129_212932

/-- The number of calories in a single candy bar -/
def calories_per_bar : ℕ := 3

/-- The number of candy bars -/
def num_bars : ℕ := 5

/-- Theorem: The total calories in 5 candy bars is 15 -/
theorem total_calories : calories_per_bar * num_bars = 15 := by
  sorry

end total_calories_l2129_212932


namespace age_ratio_proof_l2129_212901

theorem age_ratio_proof (a b c : ℕ) : 
  a = b + 2 →
  a + b + c = 12 →
  b = 4 →
  b / c = 2 := by
sorry

end age_ratio_proof_l2129_212901


namespace police_force_ratio_l2129_212966

/-- Given a police force with the following properties:
  * 20% of female officers were on duty
  * 100 officers were on duty that night
  * The police force has 250 female officers
  Prove that the ratio of female officers to total officers on duty is 1:2 -/
theorem police_force_ratio : 
  ∀ (total_female : ℕ) (on_duty : ℕ) (female_percent : ℚ),
  total_female = 250 →
  on_duty = 100 →
  female_percent = 1/5 →
  (female_percent * total_female) / on_duty = 1/2 := by
sorry

end police_force_ratio_l2129_212966


namespace vector_dot_product_l2129_212920

def vector_a : ℝ × ℝ := (-1, 3)
def vector_b (m : ℝ) : ℝ × ℝ := (m, m - 4)
def vector_c (m : ℝ) : ℝ × ℝ := (2*m, 3)

def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem vector_dot_product (m : ℝ) :
  parallel vector_a (vector_b m) →
  dot_product (vector_b m) (vector_c m) = -7 := by
  sorry

end vector_dot_product_l2129_212920


namespace remainder_problem_l2129_212950

theorem remainder_problem (x : ℤ) (h : x % 61 = 24) : x % 5 = 4 := by
  sorry

end remainder_problem_l2129_212950


namespace shaded_area_semicircle_arrangement_l2129_212946

/-- The area of the shaded region in a semicircle arrangement -/
theorem shaded_area_semicircle_arrangement (n : ℕ) (d : ℝ) (h : n = 8 ∧ d = 5) :
  let large_diameter := n * d
  let large_semicircle_area := π * (large_diameter / 2)^2 / 2
  let small_semicircle_area := n * (π * d^2 / 8)
  large_semicircle_area - small_semicircle_area = 175 * π :=
by sorry

end shaded_area_semicircle_arrangement_l2129_212946


namespace first_cat_brown_eyed_kittens_l2129_212905

theorem first_cat_brown_eyed_kittens :
  ∀ (brown_eyed_first : ℕ),
  let blue_eyed_first : ℕ := 3
  let blue_eyed_second : ℕ := 4
  let brown_eyed_second : ℕ := 6
  let total_kittens : ℕ := blue_eyed_first + brown_eyed_first + blue_eyed_second + brown_eyed_second
  let total_blue_eyed : ℕ := blue_eyed_first + blue_eyed_second
  (total_blue_eyed : ℚ) / total_kittens = 35 / 100 →
  brown_eyed_first = 7 :=
by sorry

end first_cat_brown_eyed_kittens_l2129_212905


namespace solution_comparison_l2129_212951

theorem solution_comparison (c c' d d' : ℝ) (hc : c ≠ 0) (hc' : c' ≠ 0) :
  (-d / c > -d' / c') ↔ (d' / c' < d / c) := by sorry

end solution_comparison_l2129_212951


namespace specific_hexagon_area_l2129_212938

/-- A hexagon formed by cutting a triangular corner from a square -/
structure CornerCutHexagon where
  sides : Fin 6 → ℕ
  is_valid_sides : (sides 0) + (sides 1) + (sides 2) + (sides 3) + (sides 4) + (sides 5) = 11 + 17 + 14 + 23 + 17 + 20

/-- The area of the hexagon -/
def hexagon_area (h : CornerCutHexagon) : ℕ :=
  sorry

/-- Theorem stating that the area of the specific hexagon is 1096 -/
theorem specific_hexagon_area : ∃ h : CornerCutHexagon, hexagon_area h = 1096 := by
  sorry

end specific_hexagon_area_l2129_212938


namespace painter_problem_solution_l2129_212986

/-- Given a painting job with a total number of rooms, time per room, and some rooms already painted,
    calculate the time needed to paint the remaining rooms. -/
def time_to_paint_remaining (total_rooms : ℕ) (time_per_room : ℕ) (painted_rooms : ℕ) : ℕ :=
  (total_rooms - painted_rooms) * time_per_room

/-- Theorem stating that for the specific problem, the time to paint the remaining rooms is 63 hours. -/
theorem painter_problem_solution :
  time_to_paint_remaining 11 7 2 = 63 := by
  sorry

#eval time_to_paint_remaining 11 7 2

end painter_problem_solution_l2129_212986


namespace cube_surface_area_difference_l2129_212968

theorem cube_surface_area_difference (large_cube_volume : ℕ) (num_small_cubes : ℕ) (small_cube_volume : ℕ) : 
  large_cube_volume = 6859 →
  num_small_cubes = 6859 →
  small_cube_volume = 1 →
  (num_small_cubes * 6 * small_cube_volume^(2/3) : ℕ) - (6 * large_cube_volume^(2/3) : ℕ) = 38988 := by
  sorry

#eval (6859 * 6 * 1^(2/3) : ℕ) - (6 * 6859^(2/3) : ℕ)

end cube_surface_area_difference_l2129_212968


namespace expression_evaluation_l2129_212967

theorem expression_evaluation : 3^(1^(0^8)) + ((3^1)^0)^8 = 4 := by
  sorry

end expression_evaluation_l2129_212967


namespace palindrome_pairs_exist_l2129_212971

/-- A function that checks if a positive integer is a palindrome -/
def is_palindrome (n : ℕ) : Prop :=
  ∃ (digits : List ℕ), n = digits.foldl (λ acc d => 10 * acc + d) 0 ∧ digits = digits.reverse

/-- A function that generates a palindrome given three digits -/
def generate_palindrome (a b k : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that there are at least 2005 palindrome pairs -/
theorem palindrome_pairs_exist : 
  ∃ (pairs : List (ℕ × ℕ)), pairs.length ≥ 2005 ∧ 
    ∀ (pair : ℕ × ℕ), pair ∈ pairs → 
      is_palindrome pair.1 ∧ is_palindrome pair.2 ∧ pair.2 = pair.1 + 110 :=
sorry

end palindrome_pairs_exist_l2129_212971


namespace book_pages_l2129_212903

/-- Given a book where the total number of digits used in numbering its pages is 930,
    prove that the book has 346 pages. -/
theorem book_pages (total_digits : ℕ) (h : total_digits = 930) : ∃ (pages : ℕ), pages = 346 := by
  sorry

end book_pages_l2129_212903


namespace energy_increase_with_center_charge_l2129_212928

/-- Represents the energy stored between two charges -/
structure EnergyBetweenCharges where
  charge1 : ℝ
  charge2 : ℝ
  distance : ℝ
  energy : ℝ
  proportionality : energy = (charge1 * charge2) / distance

/-- Configuration of charges on a square -/
structure SquareChargeConfiguration where
  sideLength : ℝ
  chargeValue : ℝ
  totalEnergy : ℝ

/-- Configuration with one charge moved to the center -/
structure CenterChargeConfiguration where
  sideLength : ℝ
  chargeValue : ℝ
  totalEnergy : ℝ

theorem energy_increase_with_center_charge 
  (initial : SquareChargeConfiguration)
  (final : CenterChargeConfiguration)
  (h1 : initial.totalEnergy = 20)
  (h2 : initial.sideLength = final.sideLength)
  (h3 : initial.chargeValue = final.chargeValue)
  : final.totalEnergy - initial.totalEnergy = 40 := by
  sorry

end energy_increase_with_center_charge_l2129_212928


namespace blackjack_bet_l2129_212975

theorem blackjack_bet (payout_ratio : Rat) (received_amount : ℚ) (original_bet : ℚ) : 
  payout_ratio = 3/2 →
  received_amount = 60 →
  received_amount = payout_ratio * original_bet →
  original_bet = 40 := by
sorry

end blackjack_bet_l2129_212975


namespace probability_of_four_ones_in_twelve_dice_l2129_212962

def number_of_dice : ℕ := 12
def sides_per_die : ℕ := 6
def desired_ones : ℕ := 4

def probability_of_one : ℚ := 1 / sides_per_die
def probability_of_not_one : ℚ := 1 - probability_of_one

def binomial_coefficient (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

def probability_exact_ones : ℚ :=
  (binomial_coefficient number_of_dice desired_ones : ℚ) *
  (probability_of_one ^ desired_ones) *
  (probability_of_not_one ^ (number_of_dice - desired_ones))

theorem probability_of_four_ones_in_twelve_dice :
  probability_exact_ones = 495 * 390625 / 2176782336 :=
sorry

-- The following line is to show the approximate decimal value
#eval (495 * 390625 : ℚ) / 2176782336

end probability_of_four_ones_in_twelve_dice_l2129_212962


namespace triangle_theorem_cosine_rule_sine_rule_l2129_212963

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the main theorem
theorem triangle_theorem (t : Triangle) 
  (h1 : 3 * t.a * Real.cos t.A = t.c * Real.cos t.B + t.b * Real.cos t.C) :
  Real.cos t.A = 1/3 ∧ 
  (t.a = 1 ∧ Real.cos t.B + Real.cos t.C = 2 * Real.sqrt 3 / 3 → t.c = Real.sqrt 3 / 2) := by
  sorry

-- Define helper theorems for cosine and sine rules
theorem cosine_rule (t : Triangle) :
  2 * t.a * t.c * Real.cos t.B = t.a^2 + t.c^2 - t.b^2 := by
  sorry

theorem sine_rule (t : Triangle) :
  t.a / Real.sin t.A = t.b / Real.sin t.B := by
  sorry

end triangle_theorem_cosine_rule_sine_rule_l2129_212963


namespace geometric_sequence_property_l2129_212952

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_property 
  (a : ℕ → ℝ) 
  (h_geometric : is_geometric_sequence a)
  (h_eq1 : a 2 * a 4 * a 5 = a 3 * a 6)
  (h_eq2 : a 9 * a 10 = -8) :
  a 7 = -2 := by
  sorry

end geometric_sequence_property_l2129_212952


namespace married_employees_percentage_l2129_212904

-- Define the company
structure Company where
  total_employees : ℕ
  women_percentage : ℚ
  men_single_ratio : ℚ
  women_married_percentage : ℚ

-- Define the conditions
def company_conditions (c : Company) : Prop :=
  c.women_percentage = 64 / 100 ∧
  c.men_single_ratio = 2 / 3 ∧
  c.women_married_percentage = 75 / 100

-- Define the function to calculate the percentage of married employees
def married_percentage (c : Company) : ℚ :=
  let men_percentage := 1 - c.women_percentage
  let married_men := (1 - c.men_single_ratio) * men_percentage
  let married_women := c.women_married_percentage * c.women_percentage
  married_men + married_women

-- Theorem statement
theorem married_employees_percentage (c : Company) :
  company_conditions c → married_percentage c = 60 / 100 := by
  sorry


end married_employees_percentage_l2129_212904


namespace square_root_three_expansion_l2129_212976

theorem square_root_three_expansion {a b m n : ℕ+} :
  a + b * Real.sqrt 3 = (m + n * Real.sqrt 3) ^ 2 →
  a = m ^ 2 + 3 * n ^ 2 ∧ b = 2 * m * n :=
by sorry

end square_root_three_expansion_l2129_212976


namespace OPSQ_configurations_l2129_212944

structure Point where
  x : ℝ
  y : ℝ

def O : Point := ⟨0, 0⟩

def isCollinear (p q r : Point) : Prop :=
  (q.x - p.x) * (r.y - p.y) = (r.x - p.x) * (q.y - p.y)

def isParallelogram (p q r s : Point) : Prop :=
  (q.x - p.x = s.x - r.x) ∧ (q.y - p.y = s.y - r.y)

theorem OPSQ_configurations (x₁ y₁ x₂ y₂ : ℝ) :
  let P : Point := ⟨x₁, y₁⟩
  let Q : Point := ⟨x₂, y₂⟩
  let S : Point := ⟨2*x₁, 2*y₁⟩
  (isCollinear O P Q ∨ 
   ¬(isCollinear O P Q) ∨ 
   isParallelogram O P S Q) := by sorry

end OPSQ_configurations_l2129_212944


namespace y_value_l2129_212974

theorem y_value (x y : ℤ) (h1 : x^2 = y - 3) (h2 : x = -5) : y = 28 := by
  sorry

end y_value_l2129_212974


namespace edges_after_cutting_l2129_212948

/-- A convex polyhedron. -/
structure ConvexPolyhedron where
  edges : ℕ

/-- Result of cutting off pyramids from a convex polyhedron. -/
def cutOffPyramids (P : ConvexPolyhedron) : ConvexPolyhedron :=
  ConvexPolyhedron.mk (3 * P.edges)

/-- Theorem stating the number of edges in the new polyhedron after cutting off pyramids. -/
theorem edges_after_cutting (P : ConvexPolyhedron) 
  (h : P.edges = 2021) : 
  (cutOffPyramids P).edges = 6063 := by
  sorry

end edges_after_cutting_l2129_212948


namespace trig_identity_l2129_212912

theorem trig_identity (α : Real) (h : Real.cos α ^ 2 = Real.sin α) :
  1 / Real.sin α + Real.cos α ^ 4 = 2 := by
  sorry

end trig_identity_l2129_212912


namespace subset_iff_range_l2129_212969

def A : Set ℝ := {x | x^2 - 4*x + 3 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | x^2 - a*x < x - a}

theorem subset_iff_range (a : ℝ) : A ⊇ B a ↔ 1 ≤ a ∧ a ≤ 3 := by
  sorry

end subset_iff_range_l2129_212969


namespace q4_value_l2129_212953

def sequence_a : ℕ → ℝ
| 0 => 1  -- We define a₁ = 1 based on the solution
| n + 1 => 2 * sequence_a n + 4

def sequence_q : ℕ → ℝ
| 0 => 17  -- We define q₁ = 17 to satisfy q₄ = 76
| n + 1 => 4 * sequence_q n + 8

theorem q4_value :
  sequence_a 4 = sequence_q 3 ∧ 
  sequence_a 6 = 316 → 
  sequence_q 3 = 76 := by
sorry

end q4_value_l2129_212953


namespace min_value_of_a_l2129_212964

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x + 3 - x / (Real.exp x)

theorem min_value_of_a (a : ℝ) :
  (∃ x : ℝ, x ≥ -2 ∧ f x ≤ a) ↔ a ≥ 1 - 1 / Real.exp 1 :=
sorry

end min_value_of_a_l2129_212964


namespace sourball_theorem_l2129_212935

/-- The number of sourball candies Nellie can eat before crying -/
def nellies_candies : ℕ := 12

/-- The initial number of candies in the bucket -/
def initial_candies : ℕ := 30

/-- The number of candies each person gets after dividing the remaining candies -/
def remaining_candies_per_person : ℕ := 3

/-- The number of sourball candies Jacob can eat before crying -/
def jacobs_candies (n : ℕ) : ℕ := n / 2

/-- The number of sourball candies Lana can eat before crying -/
def lanas_candies (n : ℕ) : ℕ := jacobs_candies n - 3

theorem sourball_theorem : 
  nellies_candies + jacobs_candies nellies_candies + lanas_candies nellies_candies = 
  initial_candies - 3 * remaining_candies_per_person := by
  sorry

end sourball_theorem_l2129_212935


namespace systematic_sampling_l2129_212917

/-- Represents a systematic sampling scheme -/
structure SystematicSample where
  total_students : Nat
  sample_size : Nat
  groups : Nat
  first_group_number : Nat
  sixteenth_group_number : Nat

/-- The systematic sampling theorem -/
theorem systematic_sampling
  (s : SystematicSample)
  (h1 : s.total_students = 160)
  (h2 : s.sample_size = 20)
  (h3 : s.groups = 20)
  (h4 : s.sixteenth_group_number = 126) :
  s.first_group_number = 6 := by
  sorry


end systematic_sampling_l2129_212917


namespace sets_equality_l2129_212989

def M : Set ℤ := {u : ℤ | ∃ m n l : ℤ, u = 12*m + 8*n + 4*l}
def N : Set ℤ := {u : ℤ | ∃ p q r : ℤ, u = 20*p + 16*q + 12*r}

theorem sets_equality : M = N :=
sorry

end sets_equality_l2129_212989


namespace inequality_solution_set_l2129_212911

theorem inequality_solution_set (x : ℝ) : 
  (1 - 2*x) / (x + 3) ≥ 1 ↔ -3 < x ∧ x ≤ -2/3 :=
by sorry

end inequality_solution_set_l2129_212911


namespace football_games_this_year_l2129_212977

theorem football_games_this_year 
  (total_games : ℕ) 
  (last_year_games : ℕ) 
  (h1 : total_games = 9)
  (h2 : last_year_games = 5) :
  total_games - last_year_games = 4 :=
by sorry

end football_games_this_year_l2129_212977


namespace john_total_skateboard_distance_l2129_212956

/-- The total distance John skateboarded, given his trip to and from the park -/
def total_skateboard_distance (distance_to_park : ℕ) : ℕ :=
  2 * distance_to_park

/-- Theorem: John skateboarded a total of 32 miles -/
theorem john_total_skateboard_distance :
  total_skateboard_distance 16 = 32 :=
by sorry

end john_total_skateboard_distance_l2129_212956


namespace min_value_function_l2129_212984

theorem min_value_function (x : ℝ) (h : x > -1) :
  (x^2 + 3*x + 4) / (x + 1) ≥ 2*Real.sqrt 2 + 1 := by
  sorry

end min_value_function_l2129_212984


namespace sum_of_max_min_is_zero_l2129_212926

def f (x : ℝ) : ℝ := x^2 - 2*x - 1

theorem sum_of_max_min_is_zero :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc 0 3, f x ≤ max) ∧
    (∃ x ∈ Set.Icc 0 3, f x = max) ∧
    (∀ x ∈ Set.Icc 0 3, min ≤ f x) ∧
    (∃ x ∈ Set.Icc 0 3, f x = min) ∧
    max + min = 0 := by
  sorry

end sum_of_max_min_is_zero_l2129_212926


namespace fraction_sum_equals_one_eighth_l2129_212978

theorem fraction_sum_equals_one_eighth :
  (1 : ℚ) / 6 - 5 / 12 + 3 / 8 = 1 / 8 := by
  sorry

end fraction_sum_equals_one_eighth_l2129_212978


namespace mean_median_difference_l2129_212919

-- Define the score distribution
def score_60_percent : ℝ := 0.20
def score_75_percent : ℝ := 0.40
def score_85_percent : ℝ := 0.25
def score_95_percent : ℝ := 1 - (score_60_percent + score_75_percent + score_85_percent)

-- Define the scores
def score_60 : ℝ := 60
def score_75 : ℝ := 75
def score_85 : ℝ := 85
def score_95 : ℝ := 95

-- Calculate the mean score
def mean_score : ℝ :=
  score_60_percent * score_60 +
  score_75_percent * score_75 +
  score_85_percent * score_85 +
  score_95_percent * score_95

-- Define the median score
def median_score : ℝ := score_75

-- Theorem stating the difference between mean and median
theorem mean_median_difference :
  |mean_score - median_score| = 2.5 := by
  sorry

end mean_median_difference_l2129_212919


namespace construct_angle_l2129_212942

-- Define the given angle
def given_angle : ℝ := 70

-- Define the target angle
def target_angle : ℝ := 40

-- Theorem statement
theorem construct_angle (straight_angle : ℝ) (right_angle : ℝ) 
  (h1 : straight_angle = 180) 
  (h2 : right_angle = 90) : 
  ∃ (constructed_angle : ℝ), constructed_angle = target_angle :=
sorry

end construct_angle_l2129_212942


namespace root_implies_k_value_l2129_212992

theorem root_implies_k_value (k : ℝ) : 
  (2 * 7^2 + 3 * 7 - k = 0) → k = 119 := by
  sorry

end root_implies_k_value_l2129_212992


namespace triangle_side_sum_max_l2129_212945

/-- In a triangle ABC, prove that given certain conditions, b + c has a maximum value of 6 --/
theorem triangle_side_sum_max (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ A < π ∧ 
  0 < B ∧ B < π ∧ 
  0 < C ∧ C < π ∧ 
  A + B + C = π ∧ 
  a = 3 ∧ 
  1 + (Real.tan A / Real.tan B) = (2 * c / b) ∧ 
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A → 
  b + c ≤ 6 ∧ ∃ b c, b + c = 6 :=
by sorry

end triangle_side_sum_max_l2129_212945
