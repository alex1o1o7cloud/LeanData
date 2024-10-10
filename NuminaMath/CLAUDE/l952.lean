import Mathlib

namespace prime_divisibility_l952_95298

theorem prime_divisibility (n : ℕ) (h1 : n ≥ 3) (h2 : Nat.Prime (4 * n + 1)) :
  (4 * n + 1) ∣ (n^(2*n) - 1) := by
  sorry

end prime_divisibility_l952_95298


namespace vector_projection_l952_95260

/-- Given vectors m and n, prove that the projection of m onto n is 8√13/13 -/
theorem vector_projection (m n : ℝ × ℝ) : m = (1, 2) → n = (2, 3) → 
  (m.1 * n.1 + m.2 * n.2) / Real.sqrt (n.1^2 + n.2^2) = 8 * Real.sqrt 13 / 13 := by
  sorry

end vector_projection_l952_95260


namespace car_trip_local_road_distance_l952_95216

theorem car_trip_local_road_distance 
  (local_speed highway_speed avg_speed : ℝ)
  (highway_distance : ℝ)
  (local_speed_pos : local_speed > 0)
  (highway_speed_pos : highway_speed > 0)
  (avg_speed_pos : avg_speed > 0)
  (highway_distance_pos : highway_distance > 0)
  (h_local_speed : local_speed = 20)
  (h_highway_speed : highway_speed = 60)
  (h_highway_distance : highway_distance = 120)
  (h_avg_speed : avg_speed = 36) :
  ∃ (local_distance : ℝ),
    local_distance > 0 ∧
    (local_distance + highway_distance) / ((local_distance / local_speed) + (highway_distance / highway_speed)) = avg_speed ∧
    local_distance = 60 := by
  sorry

end car_trip_local_road_distance_l952_95216


namespace equation_solution_l952_95262

theorem equation_solution : 
  ∃ x : ℝ, (7 + 3.5 * x = 2.1 * x - 25) ∧ (x = -32 / 1.4) :=
by sorry

end equation_solution_l952_95262


namespace closest_point_l952_95223

def v (s : ℝ) : Fin 3 → ℝ := fun i => 
  match i with
  | 0 => 3 + 5*s
  | 1 => -2 + 3*s
  | 2 => -4 - 2*s

def a : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 5
  | 1 => 5
  | 2 => 6

def direction : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 5
  | 1 => 3
  | 2 => -2

theorem closest_point (s : ℝ) : 
  (∀ t : ℝ, ‖v t - a‖ ≥ ‖v s - a‖) ↔ s = 11/38 :=
sorry

end closest_point_l952_95223


namespace shop_distance_is_500_l952_95284

/-- Represents the configuration of camps and shop -/
structure CampConfig where
  girls_distance : ℝ  -- perpendicular distance from girls' camp to road
  boys_distance : ℝ   -- distance along road from perpendicular to boys' camp
  shop_distance : ℝ   -- distance from shop to each camp

/-- The shop is equidistant from both camps -/
def is_equidistant (config : CampConfig) : Prop :=
  config.shop_distance^2 = config.girls_distance^2 + (config.shop_distance - config.boys_distance)^2

/-- The theorem stating that given the conditions, the shop is 500 rods from each camp -/
theorem shop_distance_is_500 (config : CampConfig) 
    (h1 : config.girls_distance = 400)
    (h2 : config.boys_distance = 800)
    (h3 : is_equidistant config) : 
  config.shop_distance = 500 := by
  sorry

#check shop_distance_is_500

end shop_distance_is_500_l952_95284


namespace distance_CD_l952_95214

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := 16 * (x - 3)^2 + 4 * (y + 2)^2 = 64

-- Define the center of the ellipse
def center : ℝ × ℝ := (3, -2)

-- Define the length of the semi-major axis
def a : ℝ := 4

-- Define the length of the semi-minor axis
def b : ℝ := 2

-- Define an endpoint of the major axis
def C : ℝ × ℝ := (center.1 + a, center.2)

-- Define an endpoint of the minor axis
def D : ℝ × ℝ := (center.1, center.2 + b)

-- Theorem statement
theorem distance_CD : Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 2 * Real.sqrt 5 := by
  sorry

end distance_CD_l952_95214


namespace expense_difference_l952_95296

def road_trip_expenses (alex_paid bob_paid carol_paid : ℚ) 
                       (a b : ℚ) : Prop :=
  let total := alex_paid + bob_paid + carol_paid
  let share := total / 3
  let alex_owes := share - alex_paid
  let bob_receives := bob_paid - share
  let carol_receives := carol_paid - share
  (alex_owes = a) ∧ (bob_receives + b = carol_receives) ∧ (a - b = 30)

theorem expense_difference :
  road_trip_expenses 120 150 210 40 10 := by sorry

end expense_difference_l952_95296


namespace rhombus_diagonal_l952_95266

/-- 
Given a rhombus with area 90 cm² and one diagonal of length 12 cm,
prove that the length of the other diagonal is 15 cm.
-/
theorem rhombus_diagonal (area : ℝ) (d1 : ℝ) (d2 : ℝ) : 
  area = 90 → d2 = 12 → area = (d1 * d2) / 2 → d1 = 15 := by
  sorry

end rhombus_diagonal_l952_95266


namespace rectangular_hall_area_l952_95211

theorem rectangular_hall_area (length width : ℝ) : 
  width = (1/2) * length →
  length - width = 12 →
  length * width = 288 := by
sorry

end rectangular_hall_area_l952_95211


namespace scooter_cost_recovery_l952_95200

/-- The minimum number of deliveries required to recover the initial cost of a scooter -/
def min_deliveries (initial_cost earnings fuel_cost parking_fee : ℕ) : ℕ :=
  (initial_cost + (earnings - fuel_cost - parking_fee) - 1) / (earnings - fuel_cost - parking_fee)

/-- Theorem stating the minimum number of deliveries required to recover the scooter cost -/
theorem scooter_cost_recovery :
  min_deliveries 3000 12 4 1 = 429 := by
  sorry

end scooter_cost_recovery_l952_95200


namespace number_puzzle_l952_95202

theorem number_puzzle : ∃ x : ℝ, 3 * (2 * x + 5) = 129 ∧ x = 19 := by
  sorry

end number_puzzle_l952_95202


namespace napoleon_has_17_beans_l952_95282

/-- The number of jelly beans Napoleon has -/
def napoleon_beans : ℕ := sorry

/-- The number of jelly beans Sedrich has -/
def sedrich_beans : ℕ := napoleon_beans + 4

/-- The number of jelly beans Mikey has -/
def mikey_beans : ℕ := 19

theorem napoleon_has_17_beans : napoleon_beans = 17 := by
  have h1 : sedrich_beans = napoleon_beans + 4 := rfl
  have h2 : 2 * (napoleon_beans + sedrich_beans) = 4 * mikey_beans := sorry
  have h3 : mikey_beans = 19 := rfl
  sorry

end napoleon_has_17_beans_l952_95282


namespace beth_graphic_novels_l952_95283

theorem beth_graphic_novels (total : ℕ) (novel_percent : ℚ) (comic_percent : ℚ) 
  (h_total : total = 120)
  (h_novel : novel_percent = 65 / 100)
  (h_comic : comic_percent = 20 / 100) :
  total - (novel_percent * total).floor - (comic_percent * total).floor = 18 := by
  sorry

end beth_graphic_novels_l952_95283


namespace total_spending_is_638_l952_95268

/-- The total spending of Elizabeth, Emma, and Elsa -/
def total_spending (emma_spending : ℕ) : ℕ :=
  let elsa_spending := 2 * emma_spending
  let elizabeth_spending := 4 * elsa_spending
  emma_spending + elsa_spending + elizabeth_spending

/-- Theorem: The total spending is $638 given the conditions -/
theorem total_spending_is_638 : total_spending 58 = 638 := by
  sorry

end total_spending_is_638_l952_95268


namespace min_value_f_range_of_m_l952_95248

noncomputable section

def f (x : ℝ) : ℝ := x * Real.log x + 2

def g (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x

theorem min_value_f (t : ℝ) (ht : t > 0) :
  (if t ≥ 1/Real.exp 1 then
    IsMinOn f (Set.Icc t (t + 2)) (f t)
   else
    IsMinOn f (Set.Icc t (t + 2)) (f (1/Real.exp 1))) ∧
  (if t ≥ 1/Real.exp 1 then
    ∀ x ∈ Set.Icc t (t + 2), f x ≥ t * Real.log t + 2
   else
    ∀ x ∈ Set.Icc t (t + 2), f x ≥ -1/Real.exp 1 + 2) :=
sorry

theorem range_of_m :
  {m : ℝ | ∃ x₀ ∈ Set.Icc (1/Real.exp 1) (Real.exp 1),
    m * (Real.log x₀ + 1) + g m x₀ ≥ 2*x₀ + m} = Set.Iic (-1) :=
sorry

end min_value_f_range_of_m_l952_95248


namespace current_speed_l952_95239

/-- Given a boat's upstream and downstream speeds, calculate the speed of the current --/
theorem current_speed (v_upstream v_downstream : ℝ) (h1 : v_upstream = 2) (h2 : v_downstream = 5) :
  (v_downstream - v_upstream) / 2 = 1.5 := by
  sorry

end current_speed_l952_95239


namespace lines_coincide_by_rotation_l952_95204

/-- Given two lines l₁ and l₂ in the plane, prove that they can coincide by rotation -/
theorem lines_coincide_by_rotation (α c : ℝ) :
  ∃ (x₀ y₀ θ : ℝ), 
    (y₀ = x₀ * Real.sin α) ∧  -- Point (x₀, y₀) is on l₁
    (∀ x y : ℝ, 
      y = x * Real.sin α →  -- Original line l₁
      ∃ x' y' : ℝ, 
        x' = (x - x₀) * Real.cos θ - (y - y₀) * Real.sin θ + x₀ ∧
        y' = (x - x₀) * Real.sin θ + (y - y₀) * Real.cos θ + y₀ ∧
        y' = 2 * x' + c)  -- Rotated line coincides with l₂
  := by sorry

end lines_coincide_by_rotation_l952_95204


namespace least_number_to_add_or_subtract_l952_95277

def original_number : ℕ := 856324

def is_three_digit_prime (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ Nat.Prime n

def divisible_by_three_digit_prime (n : ℕ) : Prop :=
  ∃ p : ℕ, is_three_digit_prime p ∧ p ∣ n

theorem least_number_to_add_or_subtract :
  ∀ k : ℕ, k < 46 →
    ¬(divisible_by_three_digit_prime (original_number + k) ∨
      divisible_by_three_digit_prime (original_number - k)) ∧
    (divisible_by_three_digit_prime (original_number - 46)) :=
by sorry

end least_number_to_add_or_subtract_l952_95277


namespace circle_max_area_center_l952_95231

/-- Given a circle with equation x^2 + y^2 + kx + 2y + k^2 = 0,
    prove that its center is (0, -1) when the area is maximum. -/
theorem circle_max_area_center (k : ℝ) :
  let circle_eq := λ (x y : ℝ) => x^2 + y^2 + k*x + 2*y + k^2 = 0
  let center := (-(k/2), -1)
  let radius_squared := 1 - (3/4) * k^2
  (∀ x y, circle_eq x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius_squared) →
  (radius_squared ≤ 1) →
  (radius_squared = 1 ↔ k = 0) →
  (k = 0 → center = (0, -1)) :=
by sorry

end circle_max_area_center_l952_95231


namespace two_variable_data_representable_by_scatter_plot_l952_95276

/-- Represents statistical data for two variables -/
structure TwoVariableData where
  -- Define the structure of two-variable data
  -- (We don't need to specify the exact structure for this problem)

/-- Represents a scatter plot -/
structure ScatterPlot where
  -- Define the structure of a scatter plot
  -- (We don't need to specify the exact structure for this problem)

/-- Creates a scatter plot from two-variable data -/
def create_scatter_plot (data : TwoVariableData) : ScatterPlot :=
  sorry -- The actual implementation is not important for this statement

/-- Theorem: Any two-variable statistical data can be represented by a scatter plot -/
theorem two_variable_data_representable_by_scatter_plot (data : TwoVariableData) :
  ∃ (plot : ScatterPlot), plot = create_scatter_plot data :=
sorry

end two_variable_data_representable_by_scatter_plot_l952_95276


namespace real_part_of_one_plus_i_squared_l952_95217

theorem real_part_of_one_plus_i_squared (i : ℂ) : 
  Complex.re ((1 + i)^2) = 0 := by sorry

end real_part_of_one_plus_i_squared_l952_95217


namespace reduced_journey_time_l952_95278

/-- Calculates the reduced time of a journey when speed is increased -/
theorem reduced_journey_time 
  (original_time : ℝ) 
  (original_speed : ℝ) 
  (new_speed : ℝ) 
  (h1 : original_time = 50) 
  (h2 : original_speed = 48) 
  (h3 : new_speed = 60) : 
  (original_time * original_speed) / new_speed = 40 := by
  sorry

end reduced_journey_time_l952_95278


namespace nadine_chairs_purchase_l952_95201

/-- Proves that Nadine bought 2 chairs given the conditions of her purchases -/
theorem nadine_chairs_purchase :
  ∀ (total_spent table_cost chair_cost : ℕ),
    total_spent = 56 →
    table_cost = 34 →
    chair_cost = 11 →
    ∃ (num_chairs : ℕ),
      num_chairs * chair_cost = total_spent - table_cost ∧
      num_chairs = 2 := by
  sorry

end nadine_chairs_purchase_l952_95201


namespace cos_105_degrees_l952_95240

theorem cos_105_degrees : Real.cos (105 * Real.pi / 180) = (Real.sqrt 2 - Real.sqrt 6) / 4 := by
  sorry

end cos_105_degrees_l952_95240


namespace work_rate_problem_l952_95255

/-- Given three workers with work rates satisfying certain conditions,
    prove that two of them together have a specific work rate. -/
theorem work_rate_problem (A B C : ℚ) 
  (h1 : A + B = 1/8)
  (h2 : A + B + C = 1/6)
  (h3 : A + C = 1/8) :
  B + C = 1/12 := by
  sorry

end work_rate_problem_l952_95255


namespace union_M_N_equals_interval_l952_95203

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x ≥ 0}
def N : Set ℝ := {x : ℝ | (x + 1) * (x - 3) < 0}

-- Define the interval (-1, +∞)
def openIntervalFromNegativeOneToInfinity : Set ℝ := {x : ℝ | x > -1}

-- State the theorem
theorem union_M_N_equals_interval : M ∪ N = openIntervalFromNegativeOneToInfinity := by
  sorry

end union_M_N_equals_interval_l952_95203


namespace gold_coin_count_l952_95292

theorem gold_coin_count (c n : ℕ) (h1 : n = 8 * (c - 3))
  (h2 : n = 5 * c + 4) (h3 : c ≥ 10) : n = 54 := by
  sorry

end gold_coin_count_l952_95292


namespace motorcycle_price_l952_95285

theorem motorcycle_price (upfront_payment : ℝ) (upfront_percentage : ℝ) (total_price : ℝ) : 
  upfront_payment = 400 → 
  upfront_percentage = 20 → 
  upfront_payment = (upfront_percentage / 100) * total_price →
  total_price = 2000 := by
sorry

end motorcycle_price_l952_95285


namespace special_triangle_properties_l952_95254

open Real

-- Define an acute triangle ABC
structure AcuteTriangle where
  A : ℝ
  B : ℝ
  C : ℝ
  acute_A : 0 < A ∧ A < π/2
  acute_B : 0 < B ∧ B < π/2
  acute_C : 0 < C ∧ C < π/2
  angle_sum : A + B + C = π

-- Define the specific conditions of the triangle
def SpecialTriangle (t : AcuteTriangle) : Prop :=
  t.B = 2 * t.A ∧ sin t.A ≠ 0 ∧ cos t.A ≠ 0

-- State the theorems to be proved
theorem special_triangle_properties (t : AcuteTriangle) (h : SpecialTriangle t) :
  ∃ (AC : ℝ), 
    AC / cos t.A = 2 ∧ 
    sqrt 2 < AC ∧ 
    AC < sqrt 3 := by sorry

end special_triangle_properties_l952_95254


namespace d_range_l952_95212

-- Define the circle C
def C (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 1

-- Define points A and B
def A : ℝ × ℝ := (0, -1)
def B : ℝ × ℝ := (0, 1)

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

-- Define the function d
def d (P : ℝ × ℝ) : ℝ := distance P A + distance P B

-- Theorem statement
theorem d_range :
  ∀ P : ℝ × ℝ, C P.1 P.2 → 32 ≤ d P ∧ d P ≤ 72 :=
sorry

end d_range_l952_95212


namespace cone_with_hole_volume_l952_95243

/-- The volume of a cone with a cylindrical hole -/
theorem cone_with_hole_volume
  (cone_diameter : ℝ)
  (cone_height : ℝ)
  (hole_diameter : ℝ)
  (h_cone_diameter : cone_diameter = 12)
  (h_cone_height : cone_height = 12)
  (h_hole_diameter : hole_diameter = 4) :
  (1/3 * π * (cone_diameter/2)^2 * cone_height) - (π * (hole_diameter/2)^2 * cone_height) = 96 * π :=
by sorry

end cone_with_hole_volume_l952_95243


namespace banana_permutations_proof_l952_95271

def banana_permutations : ℕ := 60

theorem banana_permutations_proof :
  let total_letters : ℕ := 6
  let b_count : ℕ := 1
  let a_count : ℕ := 3
  let n_count : ℕ := 2
  banana_permutations = (Nat.factorial total_letters) / (Nat.factorial b_count * Nat.factorial a_count * Nat.factorial n_count) :=
by sorry

end banana_permutations_proof_l952_95271


namespace ball_drawing_probabilities_l952_95209

/-- Represents a bag of colored balls -/
structure ColoredBalls where
  total : ℕ
  red : ℕ
  black : ℕ

/-- Calculates the probability of drawing two red balls -/
def prob_two_red (bag : ColoredBalls) : ℚ :=
  (bag.red.choose 2 : ℚ) / (bag.total.choose 2)

/-- Calculates the probability of drawing two balls of different colors -/
def prob_different_colors (bag : ColoredBalls) : ℚ :=
  (bag.red * bag.black : ℚ) / (bag.total.choose 2)

/-- The main theorem about probabilities in the ball drawing scenario -/
theorem ball_drawing_probabilities (bag : ColoredBalls) 
    (h_total : bag.total = 6)
    (h_red : bag.red = 4)
    (h_black : bag.black = 2) :
    prob_two_red bag = 2/5 ∧ prob_different_colors bag = 8/15 := by
  sorry


end ball_drawing_probabilities_l952_95209


namespace smallest_number_l952_95245

theorem smallest_number (a b c d : ℝ) : 
  a = -2 → b = 4 → c = -5 → d = 1 → 
  (c < -3 ∧ a > -3 ∧ b > -3 ∧ d > -3) :=
by sorry

end smallest_number_l952_95245


namespace min_value_expression_l952_95261

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (6 * z) / (3 * x + y) + (6 * x) / (y + 3 * z) + (2 * y) / (x + 2 * z) ≥ 3 ∧
  ((6 * z) / (3 * x + y) + (6 * x) / (y + 3 * z) + (2 * y) / (x + 2 * z) = 3 ↔ 3 * x = y ∧ y = 3 * z) :=
by sorry

end min_value_expression_l952_95261


namespace product_of_complex_magnitudes_l952_95286

theorem product_of_complex_magnitudes : Complex.abs (4 - 3*I) * Complex.abs (4 + 3*I) = 25 := by
  sorry

end product_of_complex_magnitudes_l952_95286


namespace base_conversion_537_8_to_7_l952_95293

def base_8_to_10 (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * 8^2 + tens * 8^1 + ones * 8^0

def base_10_to_7 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 7) ((m % 7) :: acc)
    aux n []

theorem base_conversion_537_8_to_7 :
  base_10_to_7 (base_8_to_10 537) = [1, 1, 0, 1, 1] := by
  sorry

end base_conversion_537_8_to_7_l952_95293


namespace reduced_rate_end_time_l952_95215

/-- Represents the fraction of a week with reduced rates -/
def reduced_rate_fraction : ℚ := 0.6428571428571429

/-- Represents the number of hours in a week -/
def hours_in_week : ℕ := 7 * 24

/-- Represents the number of hours with reduced rates on weekends -/
def weekend_reduced_hours : ℕ := 2 * 24

/-- Represents the hour when reduced rates start on weekdays (24-hour format) -/
def weekday_start_hour : ℕ := 20

/-- Represents the hour when reduced rates end on weekdays (24-hour format) -/
def weekday_end_hour : ℕ := 8

theorem reduced_rate_end_time :
  (reduced_rate_fraction * hours_in_week).floor - weekend_reduced_hours = 
  5 * (24 - weekday_start_hour + weekday_end_hour) :=
sorry

end reduced_rate_end_time_l952_95215


namespace pythagorean_triple_9_12_15_l952_95250

theorem pythagorean_triple_9_12_15 : 9^2 + 12^2 = 15^2 := by
  sorry

end pythagorean_triple_9_12_15_l952_95250


namespace jerry_debt_payment_l952_95291

/-- Jerry's debt payment problem -/
theorem jerry_debt_payment (total_debt : ℝ) (remaining_debt : ℝ) (extra_payment : ℝ) :
  total_debt = 50 ∧ 
  remaining_debt = 23 ∧ 
  extra_payment = 3 →
  ∃ (payment_two_months_ago : ℝ),
    payment_two_months_ago = 12 ∧
    total_debt = remaining_debt + payment_two_months_ago + (payment_two_months_ago + extra_payment) :=
by sorry

end jerry_debt_payment_l952_95291


namespace rectangle_area_change_l952_95229

theorem rectangle_area_change (l w : ℝ) (h : l * w = 1100) :
  (1.1 * l) * (0.9 * w) = 1089 := by
  sorry

end rectangle_area_change_l952_95229


namespace fish_ratio_problem_l952_95238

/-- The ratio of tagged fish to total fish in a second catch -/
def fish_ratio (tagged_initial : ℕ) (second_catch : ℕ) (tagged_in_catch : ℕ) (total_fish : ℕ) : ℚ :=
  tagged_in_catch / second_catch

/-- Theorem stating the ratio of tagged fish to total fish in the second catch -/
theorem fish_ratio_problem :
  let tagged_initial : ℕ := 30
  let second_catch : ℕ := 50
  let tagged_in_catch : ℕ := 2
  let total_fish : ℕ := 750
  fish_ratio tagged_initial second_catch tagged_in_catch total_fish = 1 / 25 := by
  sorry


end fish_ratio_problem_l952_95238


namespace jerrys_age_l952_95252

/-- Given that Mickey's age is 18 and Mickey's age is 4 years less than 400% of Jerry's age,
    prove that Jerry's age is 5.5 years. -/
theorem jerrys_age (mickey_age jerry_age : ℝ) : 
  mickey_age = 18 ∧ 
  mickey_age = 4 * jerry_age - 4 → 
  jerry_age = 5.5 := by
sorry

end jerrys_age_l952_95252


namespace smallest_side_of_triangle_l952_95269

theorem smallest_side_of_triangle (x : ℝ) : 
  10 + (3*x + 6) + (x + 5) = 60 →
  10 ≤ 3*x + 6 ∧ 10 ≤ x + 5 →
  10 = min 10 (min (3*x + 6) (x + 5)) :=
by sorry

end smallest_side_of_triangle_l952_95269


namespace min_attendees_with_both_l952_95219

theorem min_attendees_with_both (n : ℕ) (h1 : n > 0) : ∃ x : ℕ,
  x ≥ 1 ∧
  x ≤ n ∧
  x ≤ n / 3 ∧
  x ≤ n / 2 ∧
  ∀ y : ℕ, (y < x → ¬(y ≤ n / 3 ∧ y ≤ n / 2)) :=
by
  sorry

#check min_attendees_with_both

end min_attendees_with_both_l952_95219


namespace square_sum_geq_root3_product_l952_95247

theorem square_sum_geq_root3_product (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_product_leq_sum : a * b * c ≤ a + b + c) : 
  a^2 + b^2 + c^2 ≥ Real.sqrt 3 * (a * b * c) := by
  sorry

end square_sum_geq_root3_product_l952_95247


namespace parallel_vectors_m_zero_l952_95258

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem parallel_vectors_m_zero :
  let a : ℝ × ℝ := (-2, 3)
  let b : ℝ × ℝ := (1, m - 3/2)
  parallel a b → m = 0 := by
  sorry

end parallel_vectors_m_zero_l952_95258


namespace simplify_expression_1_simplify_expression_2_l952_95279

-- Problem 1
theorem simplify_expression_1 (m n : ℝ) : 
  (2*m + n)^2 - (4*m + 3*n)*(m - n) = 8*m*n + 4*n^2 := by sorry

-- Problem 2
theorem simplify_expression_2 (x : ℝ) 
  (h1 : x ≠ 3) (h2 : 2*x^2 - 5*x - 3 ≠ 0) : 
  ((2*x + 1)*(3*x - 4) / (2*x^2 - 5*x - 3) - 1) / ((4*x^2 - 1) / (x - 3)) = 1 / (2*x + 1) := by sorry

end simplify_expression_1_simplify_expression_2_l952_95279


namespace total_wheels_is_102_l952_95242

/-- The number of wheels Dimitri saw at the park -/
def total_wheels : ℕ :=
  let bicycle_wheels := 2
  let tricycle_wheels := 3
  let unicycle_wheels := 1
  let scooter_wheels := 4
  let men_on_bicycles := 6
  let women_on_bicycles := 5
  let boys_on_tricycles := 8
  let girls_on_tricycles := 7
  let boys_on_unicycles := 2
  let girls_on_unicycles := 1
  let boys_on_scooters := 5
  let girls_on_scooters := 3
  (men_on_bicycles + women_on_bicycles) * bicycle_wheels +
  (boys_on_tricycles + girls_on_tricycles) * tricycle_wheels +
  (boys_on_unicycles + girls_on_unicycles) * unicycle_wheels +
  (boys_on_scooters + girls_on_scooters) * scooter_wheels

theorem total_wheels_is_102 : total_wheels = 102 := by
  sorry

end total_wheels_is_102_l952_95242


namespace quadratic_function_k_value_l952_95259

/-- Definition of the quadratic function g(x) -/
def g (a b c : ℤ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- Theorem stating that under the given conditions, k = 0 -/
theorem quadratic_function_k_value
  (a b c : ℤ)
  (h1 : g a b c 2 = 0)
  (h2 : 60 < g a b c 6 ∧ g a b c 6 < 70)
  (h3 : 90 < g a b c 9 ∧ g a b c 9 < 100)
  (k : ℤ)
  (h4 : 10000 * ↑k < g a b c 50 ∧ g a b c 50 < 10000 * ↑(k + 1)) :
  k = 0 := by
  sorry

end quadratic_function_k_value_l952_95259


namespace gcd_6051_10085_l952_95288

theorem gcd_6051_10085 : Nat.gcd 6051 10085 = 2017 := by
  sorry

end gcd_6051_10085_l952_95288


namespace quadratic_inequality_solution_range_l952_95274

theorem quadratic_inequality_solution_range (t : ℝ) :
  (∃ c : ℝ, c ≤ 1 ∧ c^2 - 3*c + t ≤ 0) → t ≤ 2 := by
  sorry

end quadratic_inequality_solution_range_l952_95274


namespace expression_evaluation_l952_95257

theorem expression_evaluation :
  ∀ x : ℝ, x = -2 → x * (x^2 - 4) = 0 →
  (x - 3) / (3 * x^2 - 6 * x) * (x + 2 - 5 / (x - 2)) = -1/6 := by
  sorry

end expression_evaluation_l952_95257


namespace central_high_school_ratio_l952_95235

theorem central_high_school_ratio (f s : ℚ) 
  (h1 : f > 0) (h2 : s > 0)
  (h3 : (3/7) * f = (2/3) * s) : f / s = 14/9 := by
  sorry

end central_high_school_ratio_l952_95235


namespace line_plane_intersection_l952_95205

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the intersection operation for planes
variable (intersect : Plane → Plane → Set Line)

-- Define the subset relation for lines and planes
variable (subset : Line → Plane → Prop)

-- Define parallel relation for lines
variable (parallel : Line → Line → Prop)

-- Define intersection relation for lines
variable (intersects : Line → Line → Prop)

-- Theorem statement
theorem line_plane_intersection 
  (m n : Line) (α β : Plane) :
  (intersect α β = {m} ∧ subset n α) →
  (parallel m n ∨ intersects m n) :=
sorry

end line_plane_intersection_l952_95205


namespace inverse_of_A_l952_95210

def A : Matrix (Fin 2) (Fin 2) ℚ := !![3, 4; -2, 9]

theorem inverse_of_A :
  A⁻¹ = !![9/35, -4/35; 2/35, 3/35] := by
  sorry

end inverse_of_A_l952_95210


namespace alloy_mixture_percentage_l952_95227

/-- Proves that mixing 66 ounces of 10% alloy with 55 ounces of 21% alloy
    results in 121 ounces of an alloy with 15% copper content. -/
theorem alloy_mixture_percentage :
  let alloy_10_amount : ℝ := 66
  let alloy_10_percentage : ℝ := 10
  let alloy_21_amount : ℝ := 55
  let alloy_21_percentage : ℝ := 21
  let total_amount : ℝ := alloy_10_amount + alloy_21_amount
  let total_copper : ℝ := (alloy_10_amount * alloy_10_percentage / 100) +
                          (alloy_21_amount * alloy_21_percentage / 100)
  let final_percentage : ℝ := total_copper / total_amount * 100
  total_amount = 121 ∧ final_percentage = 15 := by sorry

end alloy_mixture_percentage_l952_95227


namespace coin_flip_probability_difference_l952_95249

/-- The probability of getting exactly k heads in n flips of a fair coin -/
def prob_k_heads (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * (1 / 2) ^ n

/-- The statement of the problem -/
theorem coin_flip_probability_difference : 
  prob_k_heads 4 3 - prob_k_heads 4 4 = 3 / 16 := by
  sorry

end coin_flip_probability_difference_l952_95249


namespace inequality_abc_l952_95287

theorem inequality_abc (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (ha : Real.sqrt a = x * (y - z)^2)
  (hb : Real.sqrt b = y * (z - x)^2)
  (hc : Real.sqrt c = z * (x - y)^2) :
  a^2 + b^2 + c^2 ≥ 2*(a*b + b*c + c*a) := by
sorry

end inequality_abc_l952_95287


namespace complex_equation_solution_l952_95263

theorem complex_equation_solution :
  ∀ (a b : ℝ), (Complex.I * 2 + 1) * a + b = Complex.I * 2 → a = 1 ∧ b = -1 :=
by
  sorry

end complex_equation_solution_l952_95263


namespace Q_value_at_8_l952_95230

-- Define the polynomial Q(x)
def Q (x : ℂ) (g h i j k l m : ℝ) : ℂ :=
  (3 * x^4 - 54 * x^3 + g * x^2 + h * x + i) *
  (4 * x^5 - 100 * x^4 + j * x^3 + k * x^2 + l * x + m)

-- Define the set of roots
def roots : Set ℂ := {2, 3, 4, 6, 7}

-- Theorem statement
theorem Q_value_at_8 (g h i j k l m : ℝ) :
  (∀ z : ℂ, Q z g h i j k l m = 0 → z ∈ roots) →
  Q 8 g h i j k l m = 14400 := by
  sorry


end Q_value_at_8_l952_95230


namespace product_of_six_consecutive_numbers_l952_95246

theorem product_of_six_consecutive_numbers (n : ℕ) (h : n = 3) :
  (n - 1) * n * (n + 1) * (n + 2) * (n + 3) * (n + 4) = 5040 := by
  sorry

end product_of_six_consecutive_numbers_l952_95246


namespace total_students_is_fifteen_l952_95256

/-- The number of students originally in Class 1 -/
def n1 : ℕ := 8

/-- The number of students originally in Class 2 -/
def n2 : ℕ := 5

/-- Lei Lei's height in cm -/
def lei_lei_height : ℕ := 158

/-- Rong Rong's height in cm -/
def rong_rong_height : ℕ := 140

/-- The change in average height of Class 1 after the swap (in cm) -/
def class1_avg_change : ℚ := 2

/-- The change in average height of Class 2 after the swap (in cm) -/
def class2_avg_change : ℚ := 3

/-- The total number of students in both classes -/
def total_students : ℕ := n1 + n2 + 2

theorem total_students_is_fifteen :
  (lei_lei_height - rong_rong_height : ℚ) / (n1 + 1) = class1_avg_change ∧
  (lei_lei_height - rong_rong_height : ℚ) / (n2 + 1) = class2_avg_change →
  total_students = 15 := by sorry

end total_students_is_fifteen_l952_95256


namespace mode_median_determinable_l952_95251

/-- Represents the age distribution of students in the model aviation interest group --/
structure AgeDistribution where
  total : Nat
  age13 : Nat
  age14 : Nat
  age15 : Nat
  age16 : Nat

/-- Conditions of the problem --/
def aviation_group : AgeDistribution where
  total := 50
  age13 := 5
  age14 := 23
  age15 := 0  -- Unknown, represented as 0
  age16 := 0  -- Unknown, represented as 0

/-- Definition of mode --/
def mode (ad : AgeDistribution) : Nat :=
  max (max ad.age13 ad.age14) (max ad.age15 ad.age16)

/-- Definition of median for even number of students --/
def median (ad : AgeDistribution) : Nat :=
  if ad.age13 + ad.age14 ≥ ad.total / 2 then 14 else 15

/-- Main theorem --/
theorem mode_median_determinable (ad : AgeDistribution) 
  (h1 : ad.total = 50)
  (h2 : ad.age13 = 5)
  (h3 : ad.age14 = 23)
  (h4 : ad.age15 + ad.age16 = ad.total - ad.age13 - ad.age14) :
  (∃ (m : Nat), mode ad = m) ∧ 
  (∃ (n : Nat), median ad = n) ∧
  (¬ ∃ (mean : ℚ), true) ∧  -- Mean cannot be determined
  (¬ ∃ (variance : ℚ), true) :=  -- Variance cannot be determined
sorry


end mode_median_determinable_l952_95251


namespace parabola_equation_l952_95244

/-- The equation of a parabola with focus at the center of x^2 + y^2 = 4x and vertex at origin -/
theorem parabola_equation (x y : ℝ) :
  (∃ (c : ℝ × ℝ), c.1^2 + c.2^2 = 4*c.1 ∧ 
   (x - c.1)^2 + (y - c.2)^2 = (x - 0)^2 + (y - 0)^2) →
  y^2 = 8*x :=
by sorry

end parabola_equation_l952_95244


namespace specific_hexahedron_volume_l952_95264

/-- A regular hexahedron with specific dimensions -/
structure RegularHexahedron where
  -- Base edge length
  ab : ℝ
  -- Top edge length
  a₁b₁ : ℝ
  -- Height
  aa₁ : ℝ
  -- Regularity conditions
  ab_positive : 0 < ab
  a₁b₁_positive : 0 < a₁b₁
  aa₁_positive : 0 < aa₁

/-- The volume of a regular hexahedron -/
def volume (h : RegularHexahedron) : ℝ :=
  -- Definition of volume calculation
  sorry

/-- Theorem stating the volume of the specific hexahedron -/
theorem specific_hexahedron_volume :
  ∃ (h : RegularHexahedron),
    h.ab = 2 ∧
    h.a₁b₁ = 3 ∧
    h.aa₁ = Real.sqrt 10 ∧
    volume h = (57 * Real.sqrt 3) / 2 := by
  sorry

end specific_hexahedron_volume_l952_95264


namespace triangle_count_in_square_with_inscribed_circle_l952_95281

structure SquareWithInscribedCircle where
  square : Set (ℝ × ℝ)
  circle : Set (ℝ × ℝ)
  midpoints : Set (ℝ × ℝ)
  diagonals : Set (Set (ℝ × ℝ))
  midpoint_segments : Set (Set (ℝ × ℝ))

/-- Given a square with an inscribed circle touching the midpoints of each side,
    with diagonals and segments joining midpoints of opposite sides drawn,
    the total number of triangles formed is 16. -/
theorem triangle_count_in_square_with_inscribed_circle
  (config : SquareWithInscribedCircle) : Nat :=
  16

#check triangle_count_in_square_with_inscribed_circle

end triangle_count_in_square_with_inscribed_circle_l952_95281


namespace target_is_largest_in_column_and_smallest_in_row_l952_95290

/-- The matrix represented as a 4x4 array of integers -/
def matrix : Matrix (Fin 4) (Fin 4) ℤ :=
  ![![5, -2, 3, 7],
    ![8, 0, 2, -1],
    ![1, -3, 6, 0],
    ![9, 1, 4, 2]]

/-- The element we're proving to be both largest in column and smallest in row -/
def target_element : ℤ := 1

/-- The position of the target element in the matrix -/
def target_position : Fin 4 × Fin 4 := (3, 1)

theorem target_is_largest_in_column_and_smallest_in_row :
  (∀ i : Fin 4, matrix i (target_position.2) ≤ target_element) ∧
  (∀ j : Fin 4, target_element ≤ matrix (target_position.1) j) := by
  sorry

#check target_is_largest_in_column_and_smallest_in_row

end target_is_largest_in_column_and_smallest_in_row_l952_95290


namespace new_person_weight_l952_95270

/-- The weight of the new person given the conditions of the problem -/
def weight_of_new_person (initial_count : ℕ) (average_increase : ℚ) (replaced_weight : ℚ) : ℚ :=
  replaced_weight + (initial_count : ℚ) * average_increase

/-- Theorem stating that the weight of the new person is 87 kg -/
theorem new_person_weight :
  weight_of_new_person 8 (5/2) 67 = 87 := by
  sorry

end new_person_weight_l952_95270


namespace actual_car_mass_is_1331_l952_95225

/-- The mass of a scaled model car -/
def model_mass : ℝ := 1

/-- The scale factor between the model and the actual car -/
def scale_factor : ℝ := 11

/-- Calculates the mass of the actual car given the model mass and scale factor -/
def actual_car_mass (model_mass : ℝ) (scale_factor : ℝ) : ℝ :=
  model_mass * (scale_factor ^ 3)

/-- Theorem stating that the mass of the actual car is 1331 kg -/
theorem actual_car_mass_is_1331 :
  actual_car_mass model_mass scale_factor = 1331 := by
  sorry

end actual_car_mass_is_1331_l952_95225


namespace cost_price_equals_selling_price_l952_95275

/-- The number of articles whose selling price equals the cost price of 20 articles -/
def x : ℚ :=
  16

/-- The profit percentage -/
def profit_percentage : ℚ :=
  25 / 100

theorem cost_price_equals_selling_price (C : ℚ) (h : C > 0) :
  20 * C = x * C * (1 + profit_percentage) :=
by sorry

end cost_price_equals_selling_price_l952_95275


namespace complete_sets_l952_95280

def is_complete (A : Set ℕ) : Prop :=
  ∀ a b : ℕ, (a + b) ∈ A → (a * b) ∈ A

theorem complete_sets :
  ∀ A : Set ℕ, A.Nonempty →
    (is_complete A ↔ 
      A = {1} ∨ 
      A = {1, 2} ∨ 
      A = {1, 2, 3, 4} ∨ 
      A = Set.univ) :=
sorry

end complete_sets_l952_95280


namespace boat_speed_in_still_water_l952_95236

/-- The speed of a boat in still water, given its speed with and against a stream. -/
theorem boat_speed_in_still_water 
  (speed_with_stream : ℝ) 
  (speed_against_stream : ℝ) 
  (h1 : speed_with_stream = 36) 
  (h2 : speed_against_stream = 8) : 
  (speed_with_stream + speed_against_stream) / 2 = 22 := by
  sorry

end boat_speed_in_still_water_l952_95236


namespace jills_shopping_tax_percentage_l952_95213

/-- Calculates the total tax percentage given spending percentages and tax rates -/
def totalTaxPercentage (clothingPercent foodPercent otherPercent : ℝ)
                       (clothingTaxRate foodTaxRate otherTaxRate : ℝ) : ℝ :=
  (clothingPercent * clothingTaxRate + foodPercent * foodTaxRate + otherPercent * otherTaxRate) * 100

/-- Theorem stating that the total tax percentage for Jill's shopping trip is 5.20% -/
theorem jills_shopping_tax_percentage :
  totalTaxPercentage 0.50 0.10 0.40 0.04 0 0.08 = 5.20 := by
  sorry

end jills_shopping_tax_percentage_l952_95213


namespace outside_point_distance_l952_95253

/-- A circle with center O and radius 5 -/
structure Circle :=
  (O : ℝ × ℝ)
  (radius : ℝ)
  (h_radius : radius = 5)

/-- A point P outside the circle -/
structure OutsidePoint (c : Circle) :=
  (P : ℝ × ℝ)
  (h_outside : dist P c.O > c.radius)

/-- The statement to prove -/
theorem outside_point_distance {c : Circle} (p : OutsidePoint c) :
  dist p.P c.O > 5 := by sorry

end outside_point_distance_l952_95253


namespace consecutive_integers_product_mod_three_l952_95299

theorem consecutive_integers_product_mod_three (n : ℤ) : 
  (n * (n + 1) / 2) % 3 = 0 ∨ (n * (n + 1) / 2) % 3 = 1 :=
by sorry

end consecutive_integers_product_mod_three_l952_95299


namespace factorization_problem_1_factorization_problem_2_l952_95297

-- Problem 1
theorem factorization_problem_1 (m n : ℝ) :
  2 * (m - n)^2 - m * (n - m) = (n - m) * (2 * n - 3 * m) := by
  sorry

-- Problem 2
theorem factorization_problem_2 (x y : ℝ) :
  -4 * x * y^2 + 4 * x^2 * y + y^3 = y * (2 * x - y)^2 := by
  sorry

end factorization_problem_1_factorization_problem_2_l952_95297


namespace de_morgan_laws_l952_95267

theorem de_morgan_laws (A B : Prop) : 
  (¬(A ∧ B) ↔ ¬A ∨ ¬B) ∧ (¬(A ∨ B) ↔ ¬A ∧ ¬B) := by
  sorry

end de_morgan_laws_l952_95267


namespace f_extrema_l952_95224

noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x + 3) + x^2

theorem f_extrema :
  let a : ℝ := -1
  let b : ℝ := (Real.exp 2 - 3) / 2
  (∀ x ∈ Set.Icc a b, f (-1/2) ≤ f x) ∧
  (∀ x ∈ Set.Icc a b, f x ≤ f ((Real.exp 2 - 3) / 2)) ∧
  f (-1/2) = Real.log 2 + 1/4 ∧
  f ((Real.exp 2 - 3) / 2) = 2 + (Real.exp 2 - 3)^2 / 4 :=
by sorry

end f_extrema_l952_95224


namespace range_of_f_l952_95228

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

-- Define the domain
def domain : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem range_of_f :
  {y : ℝ | ∃ x ∈ domain, f x = y} = {y : ℝ | 2 ≤ y ∧ y ≤ 6} :=
sorry

end range_of_f_l952_95228


namespace regular_tetrahedron_height_l952_95221

/-- Given a regular tetrahedron with an inscribed sphere, 
    prove that its height is 4 times the radius of the inscribed sphere -/
theorem regular_tetrahedron_height (h r : ℝ) : h = 4 * r :=
  sorry

end regular_tetrahedron_height_l952_95221


namespace proposition_and_equivalents_l952_95232

def IsDecreasing (a : ℕ+ → ℝ) : Prop :=
  ∀ n : ℕ+, a n ≥ a (n + 1)

theorem proposition_and_equivalents (a : ℕ+ → ℝ) :
  (∀ n : ℕ+, (a n + a (n + 1)) / 2 < a n ↔ IsDecreasing a) ∧
  (IsDecreasing a ↔ ∀ n : ℕ+, (a n + a (n + 1)) / 2 < a n) ∧
  (∀ n : ℕ+, (a n + a (n + 1)) / 2 ≥ a n ↔ ¬IsDecreasing a) ∧
  (¬IsDecreasing a ↔ ∀ n : ℕ+, (a n + a (n + 1)) / 2 ≥ a n) :=
by sorry

end proposition_and_equivalents_l952_95232


namespace apples_handed_out_to_students_l952_95222

theorem apples_handed_out_to_students 
  (initial_apples : ℕ) 
  (pies_made : ℕ) 
  (apples_per_pie : ℕ) 
  (h1 : initial_apples = 62)
  (h2 : pies_made = 6)
  (h3 : apples_per_pie = 9) :
  initial_apples - (pies_made * apples_per_pie) = 8 := by
sorry

end apples_handed_out_to_students_l952_95222


namespace absolute_value_equation_solutions_l952_95207

theorem absolute_value_equation_solutions :
  ∀ x : ℚ, (|2 * x - 3| = x + 1) ↔ (x = 4 ∨ x = 2/3) := by
  sorry

end absolute_value_equation_solutions_l952_95207


namespace quadratic_complex_roots_l952_95208

theorem quadratic_complex_roots
  (a b c : ℝ) (x : ℂ)
  (h_a : a ≠ 0)
  (h_root : a * (1 + Complex.I)^2 + b * (1 + Complex.I) + c = 0) :
  a * (1 - Complex.I)^2 + b * (1 - Complex.I) + c = 0 :=
sorry

end quadratic_complex_roots_l952_95208


namespace trigonometric_identity_l952_95272

theorem trigonometric_identity : 
  1 / Real.cos (70 * π / 180) - Real.sqrt 3 / Real.sin (70 * π / 180) = 4 := by
  sorry

end trigonometric_identity_l952_95272


namespace three_correct_letters_probability_l952_95294

/-- The number of people and letters --/
def n : ℕ := 5

/-- The number of people who receive the correct letter --/
def k : ℕ := 3

/-- The probability of exactly k people receiving their correct letter when n letters are randomly distributed to n people --/
def prob_correct_letters (n k : ℕ) : ℚ :=
  (Nat.choose n k * Nat.factorial (n - k)) / Nat.factorial n

theorem three_correct_letters_probability :
  prob_correct_letters n k = 1 / 12 := by
  sorry

end three_correct_letters_probability_l952_95294


namespace biking_difference_l952_95220

/-- Calculates the difference in miles biked between two cyclists given their speeds, 
    total time, and break times. -/
def miles_difference (alberto_speed bjorn_speed total_time alberto_break bjorn_break : ℝ) : ℝ :=
  let alberto_distance := alberto_speed * (total_time - alberto_break)
  let bjorn_distance := bjorn_speed * (total_time - bjorn_break)
  alberto_distance - bjorn_distance

/-- The difference in miles biked between Alberto and Bjorn is 17.625 miles. -/
theorem biking_difference : 
  miles_difference 15 10.5 5 0.5 0.25 = 17.625 := by
  sorry

end biking_difference_l952_95220


namespace flip_colors_iff_even_l952_95233

/-- Represents the color of a square on the board -/
inductive Color
| White
| Black
| Orange

/-- Represents a 3n × 3n board -/
def Board (n : ℕ) := Fin (3*n) → Fin (3*n) → Color

/-- Initial coloring of the board -/
def initialBoard (n : ℕ) : Board n :=
  λ i j => if (i.val + j.val) % 3 = 2 then Color.Black else Color.White

/-- A move on the board -/
def move (b : Board n) (i j : Fin (3*n)) : Board n :=
  λ x y => if x.val ∈ [i.val, i.val+1] ∧ y.val ∈ [j.val, j.val+1]
           then match b x y with
                | Color.White => Color.Orange
                | Color.Orange => Color.Black
                | Color.Black => Color.White
           else b x y

/-- The goal state of the board -/
def goalBoard (n : ℕ) : Board n :=
  λ i j => if (i.val + j.val) % 3 = 2 then Color.White else Color.Black

/-- A sequence of moves -/
def MoveSequence (n : ℕ) := List (Fin (3*n) × Fin (3*n))

/-- Apply a sequence of moves to a board -/
def applyMoves (b : Board n) (moves : MoveSequence n) : Board n :=
  moves.foldl (λ board (i, j) => move board i j) b

theorem flip_colors_iff_even (n : ℕ) (h : n > 0) :
  (∃ (moves : MoveSequence n), applyMoves (initialBoard n) moves = goalBoard n) ↔ Even n :=
sorry

end flip_colors_iff_even_l952_95233


namespace inequality_system_solutions_l952_95237

theorem inequality_system_solutions :
  let S := {x : ℤ | (3 * x + 1 < x - 3) ∧ ((1 + x) / 2 ≤ (1 + 2 * x) / 3 + 1)}
  S = {-5, -4, -3} := by
sorry

end inequality_system_solutions_l952_95237


namespace equation_solutions_l952_95241

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = -2 ∧ x₂ = 1 ∧ 
    (∀ x : ℝ, x * (x + 2) = (x + 2) ↔ x = x₁ ∨ x = x₂)) ∧
  (∃ y₁ y₂ : ℝ, y₁ = (3 - Real.sqrt 7) / 2 ∧ y₂ = (3 + Real.sqrt 7) / 2 ∧ 
    (∀ x : ℝ, 2 * x^2 - 6 * x + 1 = 0 ↔ x = y₁ ∨ x = y₂)) :=
by sorry

end equation_solutions_l952_95241


namespace dataset_mode_is_five_l952_95273

def dataset : List ℕ := [0, 1, 2, 3, 3, 5, 5, 5]

def mode (l : List ℕ) : ℕ :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem dataset_mode_is_five : mode dataset = 5 := by
  sorry

end dataset_mode_is_five_l952_95273


namespace largest_angle_in_specific_pentagon_l952_95265

/-- The measure of the largest angle in a pentagon with specific angle conditions -/
theorem largest_angle_in_specific_pentagon : 
  ∀ (A B C D E x : ℝ),
  -- Pentagon conditions
  A + B + C + D + E = 540 →
  -- Specific angle conditions
  A = 70 →
  B = 90 →
  C = D →
  E = 3 * x - 10 →
  C = x →
  -- Conclusion
  max A (max B (max C (max D E))) = 224 :=
by
  sorry

end largest_angle_in_specific_pentagon_l952_95265


namespace probability_kings_or_aces_value_l952_95234

/-- A standard deck of cards. -/
structure Deck :=
  (total_cards : ℕ)
  (num_aces : ℕ)
  (num_kings : ℕ)

/-- The probability of drawing either three kings or at least 2 aces
    when 3 cards are selected randomly from a standard deck. -/
def probability_kings_or_aces (d : Deck) : ℚ :=
  sorry

/-- The theorem stating the probability of drawing either three kings or at least 2 aces
    when 3 cards are selected randomly from a standard deck. -/
theorem probability_kings_or_aces_value (d : Deck) 
  (h1 : d.total_cards = 52)
  (h2 : d.num_aces = 4)
  (h3 : d.num_kings = 4) :
  probability_kings_or_aces d = 74 / 5525 :=
sorry

end probability_kings_or_aces_value_l952_95234


namespace circle_M_equation_l952_95289

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 2 * x + y - 1 = 0

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 5

-- Define the point M
def point_M : ℝ × ℝ := (1, -1)

-- Theorem statement
theorem circle_M_equation :
  (∃ (x y : ℝ), line_equation x y ∧ (x, y) = point_M) ∧
  circle_equation 3 0 ∧
  circle_equation 0 1 →
  ∀ (x y : ℝ), circle_equation x y ↔ (x - 1)^2 + (y + 1)^2 = 5 :=
sorry

end circle_M_equation_l952_95289


namespace third_grade_class_size_l952_95295

/-- Represents the number of students in each third grade class -/
def third_grade_students : ℕ := sorry

/-- Represents the total number of classes -/
def total_classes : ℕ := 5 + 4 + 4

/-- Represents the total number of students in fourth and fifth grades -/
def fourth_fifth_students : ℕ := 4 * 28 + 4 * 27

/-- Represents the cost of lunch per student in cents -/
def lunch_cost_per_student : ℕ := 210 + 50 + 20

/-- Represents the total cost of all lunches in cents -/
def total_lunch_cost : ℕ := 103600

theorem third_grade_class_size :
  third_grade_students = 30 ∧
  third_grade_students * 5 * lunch_cost_per_student +
  fourth_fifth_students * lunch_cost_per_student = total_lunch_cost :=
sorry

end third_grade_class_size_l952_95295


namespace gas_volume_at_10_degrees_l952_95206

-- Define the relationship between temperature change and volume change
def volume_change (temp_change : ℤ) : ℤ := (3 * temp_change) / 5

-- Define the initial conditions
def initial_temp : ℤ := 25
def initial_volume : ℤ := 40
def final_temp : ℤ := 10

-- Define the theorem
theorem gas_volume_at_10_degrees : 
  initial_volume + volume_change (final_temp - initial_temp) = 31 := by
  sorry

end gas_volume_at_10_degrees_l952_95206


namespace calculate_expression_l952_95218

theorem calculate_expression : 10 + 7 * (3 + 8)^2 = 857 := by
  sorry

end calculate_expression_l952_95218


namespace divisibility_condition_l952_95226

theorem divisibility_condition (n : ℕ+) :
  (5^(n.val - 1) + 3^(n.val - 1)) ∣ (5^n.val + 3^n.val) ↔ n = 1 := by
  sorry

end divisibility_condition_l952_95226
