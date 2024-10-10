import Mathlib

namespace chandler_wrapping_paper_sales_l1049_104936

/-- Chandler's wrapping paper sales problem -/
theorem chandler_wrapping_paper_sales 
  (total_goal : ℕ) 
  (sold_to_grandmother : ℕ) 
  (sold_to_uncle : ℕ) 
  (sold_to_neighbor : ℕ) 
  (h1 : total_goal = 12)
  (h2 : sold_to_grandmother = 3)
  (h3 : sold_to_uncle = 4)
  (h4 : sold_to_neighbor = 3) :
  total_goal - (sold_to_grandmother + sold_to_uncle + sold_to_neighbor) = 2 :=
by sorry

end chandler_wrapping_paper_sales_l1049_104936


namespace quadratic_inequality_solution_set_l1049_104966

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 + x - 12 < 0} = Set.Ioo (-4) 3 := by sorry

end quadratic_inequality_solution_set_l1049_104966


namespace mans_speed_against_current_l1049_104922

/-- Given the conditions of a man's speed in various situations, prove that his speed against the current with wind, waves, and raft is 4 km/hr. -/
theorem mans_speed_against_current (speed_with_current speed_of_current wind_effect wave_effect raft_effect : ℝ)
  (h1 : speed_with_current = 20)
  (h2 : speed_of_current = 5)
  (h3 : wind_effect = 2)
  (h4 : wave_effect = 1)
  (h5 : raft_effect = 3) :
  speed_with_current - speed_of_current - wind_effect - speed_of_current - wave_effect - raft_effect = 4 := by
  sorry

end mans_speed_against_current_l1049_104922


namespace barber_loss_l1049_104931

/-- Represents the monetary transactions in the barbershop scenario -/
structure BarbershopScenario where
  haircut_price : ℕ
  counterfeit_bill : ℕ
  change_given : ℕ
  replacement_bill : ℕ

/-- Calculates the total loss for the barber in the given scenario -/
def calculate_loss (scenario : BarbershopScenario) : ℕ :=
  scenario.haircut_price + scenario.change_given + scenario.replacement_bill - scenario.counterfeit_bill

/-- Theorem stating that the barber's loss in the given scenario is $25 -/
theorem barber_loss (scenario : BarbershopScenario) 
  (h1 : scenario.haircut_price = 15)
  (h2 : scenario.counterfeit_bill = 20)
  (h3 : scenario.change_given = 5)
  (h4 : scenario.replacement_bill = 20) :
  calculate_loss scenario = 25 := by
  sorry

end barber_loss_l1049_104931


namespace estate_value_l1049_104904

def estate_problem (total_estate : ℚ) : Prop :=
  let daughters_son_share := (3 : ℚ) / 5 * total_estate
  let first_daughter := (5 : ℚ) / 10 * daughters_son_share
  let second_daughter := (3 : ℚ) / 10 * daughters_son_share
  let son := (2 : ℚ) / 10 * daughters_son_share
  let husband := 2 * son
  let gardener := 600
  let charity := 800
  total_estate = first_daughter + second_daughter + son + husband + gardener + charity

theorem estate_value : 
  ∃ (total_estate : ℚ), estate_problem total_estate ∧ total_estate = 35000 := by
  sorry

end estate_value_l1049_104904


namespace arithmetic_mean_of_divisors_greater_than_sqrt_l1049_104974

theorem arithmetic_mean_of_divisors_greater_than_sqrt (n : ℕ) (hn : n > 1) :
  let divisors := (Finset.filter (· ∣ n) (Finset.range (n + 1))).toList
  (divisors.sum / divisors.length : ℝ) > Real.sqrt n := by
  sorry

end arithmetic_mean_of_divisors_greater_than_sqrt_l1049_104974


namespace parabola_focus_l1049_104972

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = -2 * x^2 + 4 * x + 1

/-- The focus of a parabola -/
def focus (f : ℝ × ℝ) (p : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (a b c : ℝ), 
    (∀ x y, p x y ↔ y = a * x^2 + b * x + c) ∧
    f = (- b / (2 * a), c - b^2 / (4 * a) - 1 / (4 * a))

/-- Theorem: The focus of the parabola y = -2x^2 + 4x + 1 is (1, 23/8) -/
theorem parabola_focus : focus (1, 23/8) parabola := by
  sorry

end parabola_focus_l1049_104972


namespace square_root_equation_l1049_104961

theorem square_root_equation (x : ℝ) : 
  (Real.sqrt x / Real.sqrt 0.64) + (Real.sqrt 1.44 / Real.sqrt 0.49) = 3.0892857142857144 → x = 1.21 := by
  sorry

end square_root_equation_l1049_104961


namespace intersection_of_M_and_N_l1049_104987

-- Define the sets M and N
def M : Set ℝ := {x | Real.sqrt x < 4}
def N : Set ℝ := {x | 3 * x ≥ 1}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {x | 1/3 ≤ x ∧ x < 16} := by
  sorry

end intersection_of_M_and_N_l1049_104987


namespace commentator_mistake_l1049_104916

def round_robin_games (n : ℕ) : ℕ := n * (n - 1) / 2

theorem commentator_mistake (n : ℕ) (x y : ℚ) (h1 : n = 15) :
  ¬(∃ (x y : ℚ),
    (x > 0) ∧
    (y > x) ∧
    (y < 2 * x) ∧
    (3 * x + 13 * y = round_robin_games n)) :=
  sorry

end commentator_mistake_l1049_104916


namespace photo_arrangements_l1049_104964

/-- The number of male students -/
def num_male : Nat := 4

/-- The number of female students -/
def num_female : Nat := 3

/-- The total number of students -/
def total_students : Nat := num_male + num_female

/-- Calculates the number of arrangements with male student A at one end -/
def arrangements_A_at_end : Nat :=
  2 * Nat.factorial (total_students - 1)

/-- Calculates the number of arrangements where female student B is not to the left of female student C -/
def arrangements_B_not_left_of_C : Nat :=
  Nat.factorial total_students / 2

/-- Calculates the number of arrangements where female student B is not at the ends and female student C is not in the middle -/
def arrangements_B_not_ends_C_not_middle : Nat :=
  Nat.factorial (total_students - 1) + 4 * 5 * Nat.factorial (total_students - 2)

theorem photo_arrangements :
  arrangements_A_at_end = 1440 ∧
  arrangements_B_not_left_of_C = 2520 ∧
  arrangements_B_not_ends_C_not_middle = 3120 := by
  sorry

end photo_arrangements_l1049_104964


namespace train_speed_calculation_l1049_104900

/-- Represents the speed of a train in various conditions -/
structure TrainSpeed where
  /-- Speed of the train including stoppages (in kmph) -/
  average_speed : ℝ
  /-- Time the train stops per hour (in minutes) -/
  stop_time : ℝ
  /-- Speed of the train when not stopping (in kmph) -/
  actual_speed : ℝ

/-- Theorem stating the relationship between average speed, stop time, and actual speed -/
theorem train_speed_calculation (t : TrainSpeed) (h1 : t.average_speed = 36) 
    (h2 : t.stop_time = 24) : t.actual_speed = 60 := by
  sorry

end train_speed_calculation_l1049_104900


namespace ninth_power_five_and_eleventh_power_five_l1049_104909

theorem ninth_power_five_and_eleventh_power_five :
  9^5 = 59149 ∧ 11^5 = 161051 := by
  sorry

#check ninth_power_five_and_eleventh_power_five

end ninth_power_five_and_eleventh_power_five_l1049_104909


namespace abc_inequality_l1049_104924

theorem abc_inequality : 
  let a : ℝ := -(0.3^2)
  let b : ℝ := 3⁻¹
  let c : ℝ := (-1/3)^0
  a < b ∧ b < c := by sorry

end abc_inequality_l1049_104924


namespace smallest_angle_at_vertices_l1049_104967

/-- A cube in 3D space -/
structure Cube where
  side : ℝ
  center : ℝ × ℝ × ℝ

/-- A point in 3D space -/
def Point3D := ℝ × ℝ × ℝ

/-- The angle at which a point sees the space diagonal of a cube -/
def angle_at_point (c : Cube) (p : Point3D) : ℝ := sorry

/-- The vertices of a cube -/
def cube_vertices (c : Cube) : Set Point3D := sorry

/-- The surface of a cube -/
def cube_surface (c : Cube) : Set Point3D := sorry

/-- Theorem: The vertices of a cube are the only points on its surface where 
    the space diagonal is seen at a 90-degree angle, which is the smallest possible angle -/
theorem smallest_angle_at_vertices (c : Cube) : 
  ∀ p ∈ cube_surface c, 
    angle_at_point c p = Real.pi / 2 ↔ p ∈ cube_vertices c :=
by sorry

end smallest_angle_at_vertices_l1049_104967


namespace buses_passed_count_l1049_104948

/-- Represents the frequency of bus departures in minutes -/
def dallas_departure_frequency : ℕ := 60
def houston_departure_frequency : ℕ := 60

/-- Represents the offset of Houston departures from the hour in minutes -/
def houston_departure_offset : ℕ := 45

/-- Represents the trip duration in hours -/
def trip_duration : ℕ := 6

/-- Represents the number of Dallas-bound buses passed by a Houston-bound bus -/
def buses_passed : ℕ := 11

theorem buses_passed_count :
  buses_passed = 11 := by sorry

end buses_passed_count_l1049_104948


namespace ellipse_eccentricity_half_l1049_104906

/-- Given an ellipse and a hyperbola with shared foci, prove the eccentricity of the ellipse is 1/2 -/
theorem ellipse_eccentricity_half 
  (a b m n c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hm : m > 0) 
  (hn : n > 0) 
  (hab : a > b)
  (ellipse_eq : ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 → (x, y) ∈ {p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1})
  (hyperbola_eq : ∀ (x y : ℝ), x^2 / m^2 - y^2 / n^2 = 1 → (x, y) ∈ {p : ℝ × ℝ | p.1^2 / m^2 - p.2^2 / n^2 = 1})
  (foci : c > 0 ∧ {(-c, 0), (c, 0)} ⊆ {p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1} ∩ {p : ℝ × ℝ | p.1^2 / m^2 - p.2^2 / n^2 = 1})
  (geom_mean : c^2 = a * m)
  (arith_mean : n^2 = m^2 + c^2 / 2) :
  c / a = 1 / 2 := by
sorry

end ellipse_eccentricity_half_l1049_104906


namespace max_value_sum_of_fractions_l1049_104995

theorem max_value_sum_of_fractions (a b c : ℝ) 
  (nonneg_a : 0 ≤ a) (nonneg_b : 0 ≤ b) (nonneg_c : 0 ≤ c)
  (sum_eq_three : a + b + c = 3) :
  (a * b) / (a + b) + (a * c) / (a + c) + (b * c) / (b + c) ≤ 3 / 2 := by
sorry

end max_value_sum_of_fractions_l1049_104995


namespace swap_digits_theorem_l1049_104951

/-- Represents a two-digit number with digits a and b -/
structure TwoDigitNumber where
  a : ℕ
  b : ℕ
  a_less_than_ten : a < 10
  b_less_than_ten : b < 10

/-- The value of a two-digit number -/
def TwoDigitNumber.value (n : TwoDigitNumber) : ℕ := 10 * n.a + n.b

/-- The value of a two-digit number with swapped digits -/
def TwoDigitNumber.swapped_value (n : TwoDigitNumber) : ℕ := 10 * n.b + n.a

/-- Theorem stating that swapping digits in a two-digit number results in 10b + a -/
theorem swap_digits_theorem (n : TwoDigitNumber) : 
  n.swapped_value = 10 * n.b + n.a := by sorry

end swap_digits_theorem_l1049_104951


namespace keith_card_spend_l1049_104962

/-- The amount Keith spent on cards -/
def total_spent (digimon_packs : ℕ) (digimon_price : ℚ) (baseball_price : ℚ) : ℚ :=
  digimon_packs * digimon_price + baseball_price

/-- Proof that Keith spent $23.86 on cards -/
theorem keith_card_spend :
  total_spent 4 (4.45 : ℚ) (6.06 : ℚ) = (23.86 : ℚ) := by
  sorry

end keith_card_spend_l1049_104962


namespace train_crossing_time_l1049_104956

/-- Calculates the time for a train to cross a platform -/
theorem train_crossing_time
  (train_speed : Real)
  (man_crossing_time : Real)
  (platform_length : Real)
  (h1 : train_speed = 72 * (1000 / 3600)) -- 72 kmph converted to m/s
  (h2 : man_crossing_time = 20)
  (h3 : platform_length = 200) :
  let train_length := train_speed * man_crossing_time
  let total_length := train_length + platform_length
  total_length / train_speed = 30 := by sorry

end train_crossing_time_l1049_104956


namespace complex_number_quadrant_l1049_104980

theorem complex_number_quadrant : ∃ (z : ℂ), z = Complex.I * (1 - 2 * Complex.I) ∧ 
  Complex.re z > 0 ∧ Complex.im z > 0 := by
  sorry

end complex_number_quadrant_l1049_104980


namespace complex_sum_problem_l1049_104910

theorem complex_sum_problem (a b c d e f : ℝ) :
  b = 5 →
  e = -2 * (a + c) →
  (a + b * Complex.I) + (c + d * Complex.I) + (e + f * Complex.I) = 4 * Complex.I →
  d + f = -1 := by
  sorry

end complex_sum_problem_l1049_104910


namespace min_value_of_expression_l1049_104954

theorem min_value_of_expression (a : ℝ) (h1 : 1 < a) (h2 : a < 3/2) :
  ∃ (m : ℝ), m = 16/9 ∧ ∀ x, 1 < x ∧ x < 3/2 → (1/(3-2*x) + 2/(x-1)) ≥ m :=
sorry

end min_value_of_expression_l1049_104954


namespace percent_of_a_is_4b_l1049_104999

theorem percent_of_a_is_4b (a b : ℝ) (h : a = 1.8 * b) : 
  (4 * b) / a * 100 = 222.22 := by
  sorry

end percent_of_a_is_4b_l1049_104999


namespace sequence_property_l1049_104986

def a (n : ℕ) (x : ℝ) : ℝ := 1 + x^(n+1) + x^(n+2)

theorem sequence_property (x : ℝ) :
  (a 2 x)^2 = (a 1 x) * (a 3 x) →
  ∀ n ≥ 3, (a n x)^2 = a n x :=
by sorry

end sequence_property_l1049_104986


namespace system_solution_l1049_104913

theorem system_solution : ∃! (x y : ℚ), 7 * x - 50 * y = 3 ∧ 3 * y - x = 5 ∧ x = -241/29 ∧ y = -32/29 := by
  sorry

end system_solution_l1049_104913


namespace trigonometric_values_l1049_104990

theorem trigonometric_values : 
  (Real.sin (30 * π / 180) = 1 / 2) ∧ 
  (Real.cos (11 * π / 4) = -Real.sqrt 2 / 2) := by
  sorry

end trigonometric_values_l1049_104990


namespace range_of_a_l1049_104976

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - a| < 1 ↔ (1/2 : ℝ) < x ∧ x < (3/2 : ℝ)) → 
  ((1/2 : ℝ) ≤ a ∧ a ≤ (3/2 : ℝ)) :=
by sorry

end range_of_a_l1049_104976


namespace machine_x_production_rate_l1049_104927

/-- The number of sprockets produced by both machines -/
def total_sprockets : ℕ := 660

/-- The additional time taken by Machine X compared to Machine B -/
def time_difference : ℕ := 10

/-- The production rate of Machine B relative to Machine X -/
def rate_ratio : ℚ := 11/10

/-- The production rate of Machine X in sprockets per hour -/
def machine_x_rate : ℚ := 6

theorem machine_x_production_rate :
  ∃ (machine_b_rate : ℚ) (time_x time_b : ℚ),
    machine_b_rate = rate_ratio * machine_x_rate ∧
    time_x = time_b + time_difference ∧
    machine_x_rate * time_x = total_sprockets ∧
    machine_b_rate * time_b = total_sprockets :=
by sorry

end machine_x_production_rate_l1049_104927


namespace ratio_x_sqrt_w_l1049_104981

theorem ratio_x_sqrt_w (x y z w v : ℝ) 
  (hx : x = 1.20 * y)
  (hy : y = 0.30 * z)
  (hz : z = 1.35 * w)
  (hw : w = v^2)
  (hv : v = 0.50 * x) :
  x / Real.sqrt w = 2 := by
  sorry

end ratio_x_sqrt_w_l1049_104981


namespace wall_space_to_paint_is_560_l1049_104917

/-- Represents the dimensions of a rectangular room -/
structure RoomDimensions where
  width : ℝ
  length : ℝ
  height : ℝ

/-- Represents a rectangular feature on a wall (e.g., door or window) -/
structure WallFeature where
  width : ℝ
  height : ℝ

/-- Calculates the total area of wall space to paint in a room -/
def wallSpaceToPaint (room : RoomDimensions) (doorway1 : WallFeature) (window : WallFeature) (doorway2 : WallFeature) : ℝ :=
  let totalWallArea := 2 * (room.width * room.height + room.length * room.height)
  let featureArea := doorway1.width * doorway1.height + window.width * window.height + doorway2.width * doorway2.height
  totalWallArea - featureArea

/-- The main theorem stating that the wall space to paint is 560 square feet -/
theorem wall_space_to_paint_is_560 (room : RoomDimensions) (doorway1 : WallFeature) (window : WallFeature) (doorway2 : WallFeature) :
  room.width = 20 ∧ room.length = 20 ∧ room.height = 8 ∧
  doorway1.width = 3 ∧ doorway1.height = 7 ∧
  window.width = 6 ∧ window.height = 4 ∧
  doorway2.width = 5 ∧ doorway2.height = 7 →
  wallSpaceToPaint room doorway1 window doorway2 = 560 :=
by
  sorry

end wall_space_to_paint_is_560_l1049_104917


namespace money_left_over_l1049_104932

theorem money_left_over (hourly_rate : ℕ) (hours_worked : ℕ) (game_cost : ℕ) (candy_cost : ℕ) : 
  hourly_rate = 8 → 
  hours_worked = 9 → 
  game_cost = 60 → 
  candy_cost = 5 → 
  hourly_rate * hours_worked - (game_cost + candy_cost) = 7 := by
  sorry

end money_left_over_l1049_104932


namespace savings_calculation_l1049_104975

theorem savings_calculation (savings : ℚ) : 
  (1 / 2 : ℚ) * savings = 300 → savings = 600 := by
  sorry

end savings_calculation_l1049_104975


namespace cube_surface_area_from_volume_l1049_104918

theorem cube_surface_area_from_volume (volume : ℝ) (side_length : ℝ) (surface_area : ℝ) : 
  volume = 343 →
  volume = side_length ^ 3 →
  surface_area = 6 * side_length ^ 2 →
  surface_area = 294 := by
sorry

end cube_surface_area_from_volume_l1049_104918


namespace second_task_end_time_l1049_104988

-- Define the start and end times in minutes since midnight
def start_time : Nat := 8 * 60  -- 8:00 AM
def end_time : Nat := 12 * 60 + 20  -- 12:20 PM

-- Define the number of tasks
def num_tasks : Nat := 4

-- Theorem to prove
theorem second_task_end_time :
  let total_duration : Nat := end_time - start_time
  let task_duration : Nat := total_duration / num_tasks
  let second_task_end : Nat := start_time + 2 * task_duration
  second_task_end = 10 * 60 + 10  -- 10:10 AM
  := by sorry

end second_task_end_time_l1049_104988


namespace sexual_reproduction_genetic_diversity_l1049_104952

/-- Represents a set of genes -/
def GeneticMaterial : Type := Set Nat

/-- Represents an organism with genetic material -/
structure Organism :=
  (genes : GeneticMaterial)

/-- Represents the process of meiosis -/
def meiosis (parent : Organism) : GeneticMaterial :=
  sorry

/-- Represents the process of fertilization -/
def fertilization (gamete1 gamete2 : GeneticMaterial) : Organism :=
  sorry

/-- Theorem stating that sexual reproduction produces offspring with different genetic combinations -/
theorem sexual_reproduction_genetic_diversity 
  (parent1 parent2 : Organism) : 
  ∃ (offspring : Organism), 
    offspring = fertilization (meiosis parent1) (meiosis parent2) ∧
    offspring.genes ≠ parent1.genes ∧
    offspring.genes ≠ parent2.genes :=
  sorry

end sexual_reproduction_genetic_diversity_l1049_104952


namespace probability_at_least_one_heart_or_king_l1049_104958

def standard_deck_size : ℕ := 52
def heart_or_king_count : ℕ := 16

theorem probability_at_least_one_heart_or_king :
  let p : ℚ := 1 - (1 - heart_or_king_count / standard_deck_size) ^ 2
  p = 88 / 169 := by
  sorry

end probability_at_least_one_heart_or_king_l1049_104958


namespace scientific_notation_of_1206_million_l1049_104982

theorem scientific_notation_of_1206_million : 
  ∃ (a : ℝ) (n : ℤ), 1206000000 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.206 ∧ n = 7 := by
  sorry

end scientific_notation_of_1206_million_l1049_104982


namespace vicente_meat_purchase_l1049_104934

theorem vicente_meat_purchase
  (rice_kg : ℕ)
  (rice_price : ℚ)
  (meat_price : ℚ)
  (total_spent : ℚ)
  (h1 : rice_kg = 5)
  (h2 : rice_price = 2)
  (h3 : meat_price = 5)
  (h4 : total_spent = 25)
  : (total_spent - rice_kg * rice_price) / meat_price = 3 := by
  sorry

end vicente_meat_purchase_l1049_104934


namespace unique_solution_exponential_equation_l1049_104942

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (2 : ℝ)^(4*x+2) * (4 : ℝ)^(2*x+8) = (8 : ℝ)^(3*x+7) :=
by sorry

end unique_solution_exponential_equation_l1049_104942


namespace rectangular_parallelepiped_surface_area_l1049_104914

/-- A rectangular parallelepiped with length and width twice the height and sum of edge lengths 100 cm has surface area 400 cm² -/
theorem rectangular_parallelepiped_surface_area 
  (h : ℝ) 
  (sum_edges : 4 * (2 * h) + 4 * (2 * h) + 4 * h = 100) : 
  2 * (2 * h) * (2 * h) + 2 * (2 * h) * h + 2 * (2 * h) * h = 400 := by
  sorry

end rectangular_parallelepiped_surface_area_l1049_104914


namespace max_concert_tickets_l1049_104919

theorem max_concert_tickets (ticket_cost : ℚ) (available_money : ℚ) : 
  ticket_cost = 15 → available_money = 120 → 
  (∃ (n : ℕ), n * ticket_cost ≤ available_money ∧ 
    ∀ (m : ℕ), m * ticket_cost ≤ available_money → m ≤ n) → 
  (∃ (max_tickets : ℕ), max_tickets = 8) :=
by sorry

end max_concert_tickets_l1049_104919


namespace minimal_fence_posts_l1049_104955

/-- Calculates the number of fence posts required for a rectangular park --/
def fence_posts (length width post_spacing : ℕ) : ℕ :=
  let long_side_posts := length / post_spacing + 1
  let short_side_posts := width / post_spacing
  long_side_posts + 2 * short_side_posts

/-- Theorem stating the minimal number of fence posts required for the given park --/
theorem minimal_fence_posts :
  fence_posts 90 45 15 = 13 := by
  sorry

end minimal_fence_posts_l1049_104955


namespace proposition_truth_values_l1049_104984

-- Define proposition p
def p : Prop := ∀ a : ℝ, (∀ x : ℝ, (x^2 + |x - a| = (-x)^2 + |(-x) - a|)) → a = 0

-- Define proposition q
def q : Prop := ∀ m : ℝ, m > 0 → ∃ x : ℝ, m * x^2 - 2 * x + 1 = 0

-- Theorem stating the truth values of the propositions
theorem proposition_truth_values : 
  p ∧ 
  ¬q ∧ 
  (p ∨ q) ∧ 
  ¬(p ∧ q) ∧ 
  ¬((¬p) ∧ q) ∧ 
  ((¬p) ∨ (¬q)) := by
  sorry

end proposition_truth_values_l1049_104984


namespace camila_garden_walkway_area_camila_garden_walkway_area_proof_l1049_104908

/-- The total area of walkways in Camila's garden -/
theorem camila_garden_walkway_area : ℕ :=
  let num_rows : ℕ := 4
  let num_cols : ℕ := 3
  let bed_width : ℕ := 8
  let bed_height : ℕ := 3
  let walkway_width : ℕ := 2
  let total_width : ℕ := num_cols * bed_width + (num_cols + 1) * walkway_width
  let total_height : ℕ := num_rows * bed_height + (num_rows + 1) * walkway_width
  let total_area : ℕ := total_width * total_height
  let total_bed_area : ℕ := num_rows * num_cols * bed_width * bed_height
  let walkway_area : ℕ := total_area - total_bed_area
  416

theorem camila_garden_walkway_area_proof : camila_garden_walkway_area = 416 := by
  sorry

end camila_garden_walkway_area_camila_garden_walkway_area_proof_l1049_104908


namespace angle_terminal_side_point_l1049_104991

theorem angle_terminal_side_point (α : Real) (m : Real) :
  m > 0 →
  (2 : Real) / Real.sqrt (4 + m^2) = 2 * Real.sqrt 5 / 5 →
  m = 1 := by
sorry

end angle_terminal_side_point_l1049_104991


namespace cyclist_pedestrian_meeting_point_l1049_104953

/-- Given three points A, B, C on a line, with AB = 3 km and BC = 4 km,
    a cyclist starting from A towards C, and a pedestrian starting from B towards A,
    prove that they meet at a point 2.1 km from A if they arrive at their
    destinations simultaneously. -/
theorem cyclist_pedestrian_meeting_point
  (A B C : ℝ) -- Points represented as real numbers
  (h_order : A < B ∧ B < C) -- Points are in order
  (h_AB : B - A = 3) -- Distance AB is 3 km
  (h_BC : C - B = 4) -- Distance BC is 4 km
  (cyclist_speed pedestrian_speed : ℝ) -- Speeds of cyclist and pedestrian
  (h_speeds_positive : cyclist_speed > 0 ∧ pedestrian_speed > 0) -- Speeds are positive
  (h_simultaneous_arrival : (C - A) / cyclist_speed = (B - A) / pedestrian_speed) -- Simultaneous arrival
  : ∃ (D : ℝ), D - A = 21/10 ∧ A < D ∧ D < B :=
sorry

end cyclist_pedestrian_meeting_point_l1049_104953


namespace alexis_dresses_l1049_104940

theorem alexis_dresses (isabella_total : ℕ) (alexis_pants : ℕ) 
  (h1 : isabella_total = 13)
  (h2 : alexis_pants = 21) : 
  3 * isabella_total - alexis_pants = 18 := by
  sorry

end alexis_dresses_l1049_104940


namespace inradius_eq_centroid_height_l1049_104963

/-- A non-equilateral triangle with sides a, b, and c, where a + b = 2c -/
structure NonEquilateralTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  non_equilateral : a ≠ b ∨ b ≠ c ∨ a ≠ c
  side_relation : a + b = 2 * c

/-- The inradius of a triangle -/
def inradius (t : NonEquilateralTriangle) : ℝ :=
  sorry

/-- The vertical distance from the base c to the centroid -/
def centroid_height (t : NonEquilateralTriangle) : ℝ :=
  sorry

/-- Theorem stating that the inradius is equal to the vertical distance from the base to the centroid -/
theorem inradius_eq_centroid_height (t : NonEquilateralTriangle) :
  inradius t = centroid_height t :=
sorry

end inradius_eq_centroid_height_l1049_104963


namespace total_seeds_l1049_104977

/-- The number of seeds Emily planted in the big garden. -/
def big_garden_seeds : ℕ := 36

/-- The number of small gardens Emily had. -/
def small_gardens : ℕ := 3

/-- The number of seeds Emily planted in each small garden. -/
def seeds_per_small_garden : ℕ := 2

/-- Theorem stating the total number of seeds Emily started with. -/
theorem total_seeds : 
  big_garden_seeds + small_gardens * seeds_per_small_garden = 42 := by
  sorry

end total_seeds_l1049_104977


namespace smallest_integers_satisfying_equation_l1049_104939

theorem smallest_integers_satisfying_equation :
  ∃ (a b : ℕ+),
    (7 * a^3 = 11 * b^5) ∧
    (∀ (a' b' : ℕ+), 7 * a'^3 = 11 * b'^5 → a ≤ a' ∧ b ≤ b') ∧
    a = 41503 ∧
    b = 539 := by
  sorry

end smallest_integers_satisfying_equation_l1049_104939


namespace ice_cream_distribution_l1049_104985

theorem ice_cream_distribution (total_sandwiches : ℕ) (num_nieces : ℕ) 
  (h1 : total_sandwiches = 1857)
  (h2 : num_nieces = 37) :
  (total_sandwiches / num_nieces : ℕ) = 50 :=
by
  sorry

end ice_cream_distribution_l1049_104985


namespace arithmetic_sequence_count_l1049_104989

theorem arithmetic_sequence_count (a₁ aₙ d : ℤ) (n : ℕ) :
  a₁ = 165 ∧ aₙ = 35 ∧ d = -5 →
  aₙ = a₁ + (n - 1) * d →
  n = 27 := by
  sorry

end arithmetic_sequence_count_l1049_104989


namespace deposit_calculation_l1049_104933

theorem deposit_calculation (total_cost : ℝ) (deposit : ℝ) : 
  deposit = 0.1 * total_cost ∧ 
  total_cost - deposit = 1080 → 
  deposit = 120 := by
sorry

end deposit_calculation_l1049_104933


namespace toy_selection_proof_l1049_104938

def factorial (n : ℕ) : ℕ := sorry

def combinations (n r : ℕ) : ℕ := 
  factorial n / (factorial r * factorial (n - r))

theorem toy_selection_proof : 
  combinations 10 3 = 120 := by sorry

end toy_selection_proof_l1049_104938


namespace geometric_sequence_property_l1049_104901

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The theorem statement -/
theorem geometric_sequence_property 
  (a : ℕ → ℝ) 
  (h_geo : geometric_sequence a) 
  (h_sum : a 6 + a 8 = 4) : 
  a 8 * (a 4 + 2 * a 6 + a 8) = 16 := by
  sorry

end geometric_sequence_property_l1049_104901


namespace height_survey_groups_l1049_104947

theorem height_survey_groups (max_height min_height class_interval : ℝ) 
  (h1 : max_height = 173)
  (h2 : min_height = 140)
  (h3 : class_interval = 5) : 
  Int.ceil ((max_height - min_height) / class_interval) = 7 := by
  sorry

end height_survey_groups_l1049_104947


namespace hannahs_purchase_cost_l1049_104950

/-- The total cost of purchasing sweatshirts and T-shirts -/
def total_cost (num_sweatshirts num_tshirts sweatshirt_price tshirt_price : ℕ) : ℕ :=
  num_sweatshirts * sweatshirt_price + num_tshirts * tshirt_price

/-- Theorem stating that the total cost of 3 sweatshirts at $15 each and 2 T-shirts at $10 each is $65 -/
theorem hannahs_purchase_cost :
  total_cost 3 2 15 10 = 65 := by
  sorry

end hannahs_purchase_cost_l1049_104950


namespace lawn_mowing_earnings_l1049_104930

theorem lawn_mowing_earnings 
  (total_lawns : ℕ) 
  (unmowed_lawns : ℕ) 
  (total_earnings : ℕ) 
  (h1 : total_lawns = 17) 
  (h2 : unmowed_lawns = 9) 
  (h3 : total_earnings = 32) : 
  (total_earnings : ℚ) / ((total_lawns - unmowed_lawns) : ℚ) = 4 := by
sorry

end lawn_mowing_earnings_l1049_104930


namespace revenue_growth_exists_l1049_104915

/-- Represents the revenue growth rate in a supermarket over three months -/
def revenue_growth_equation (x : ℝ) : Prop :=
  let january_revenue : ℝ := 90
  let total_revenue : ℝ := 144
  january_revenue + january_revenue * (1 + x) + january_revenue * (1 + x)^2 = total_revenue

/-- Theorem stating that the revenue growth equation holds for some growth rate x -/
theorem revenue_growth_exists : ∃ x : ℝ, revenue_growth_equation x := by
  sorry

end revenue_growth_exists_l1049_104915


namespace equal_bills_at_20_minutes_l1049_104970

/-- Represents a telephone company with a base rate and per-minute charge. -/
structure TelephoneCompany where
  base_rate : ℝ
  per_minute_charge : ℝ

/-- Calculates the total cost for a given number of minutes. -/
def total_cost (company : TelephoneCompany) (minutes : ℝ) : ℝ :=
  company.base_rate + company.per_minute_charge * minutes

/-- The three telephone companies with their respective rates. -/
def united_telephone : TelephoneCompany := ⟨11, 0.25⟩
def atlantic_call : TelephoneCompany := ⟨12, 0.20⟩
def global_connect : TelephoneCompany := ⟨13, 0.15⟩

theorem equal_bills_at_20_minutes :
  ∃ (m : ℝ),
    m = 20 ∧
    total_cost united_telephone m = total_cost atlantic_call m ∧
    total_cost atlantic_call m = total_cost global_connect m :=
  sorry

end equal_bills_at_20_minutes_l1049_104970


namespace banana_arrangements_l1049_104937

def word := "BANANA"
def total_letters : ℕ := 6
def freq_B : ℕ := 1
def freq_A : ℕ := 3
def freq_N : ℕ := 2

theorem banana_arrangements : 
  (Nat.factorial total_letters) / 
  (Nat.factorial freq_B * Nat.factorial freq_A * Nat.factorial freq_N) = 60 := by
  sorry

end banana_arrangements_l1049_104937


namespace interest_equality_implies_second_sum_l1049_104905

theorem interest_equality_implies_second_sum (total : ℚ) 
  (h1 : total = 2665) 
  (h2 : ∃ x : ℚ, x * (3/100) * 8 = (total - x) * (5/100) * 3) : 
  ∃ second : ℚ, second = total - 2460 :=
sorry

end interest_equality_implies_second_sum_l1049_104905


namespace intersection_of_ranges_equality_of_ranges_equality_of_functions_l1049_104903

def f (x : ℝ) : ℝ := x^2 + 1
def g (x : ℝ) : ℝ := 4*x + 1

def A₁ : Set ℝ := Set.Icc 1 2
def S₁ : Set ℝ := Set.image f A₁
def T₁ : Set ℝ := Set.image g A₁

def A₂ (m : ℝ) : Set ℝ := Set.Icc 0 m
def S₂ (m : ℝ) : Set ℝ := Set.image f (A₂ m)
def T₂ (m : ℝ) : Set ℝ := Set.image g (A₂ m)

theorem intersection_of_ranges : S₁ ∩ T₁ = {5} := by sorry

theorem equality_of_ranges (m : ℝ) : S₂ m = T₂ m → m = 4 := by sorry

theorem equality_of_functions : 
  {A : Set ℝ | ∀ x ∈ A, f x = g x} ⊆ {{0}, {4}, {0, 4}} := by sorry

end intersection_of_ranges_equality_of_ranges_equality_of_functions_l1049_104903


namespace fraction_zero_implies_x_one_l1049_104996

theorem fraction_zero_implies_x_one :
  ∀ x : ℝ, (x - 1) / (2 * x - 4) = 0 → x = 1 := by
  sorry

end fraction_zero_implies_x_one_l1049_104996


namespace min_value_expression_l1049_104926

theorem min_value_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  a^3 + b^3 + 1/a^3 + b/a ≥ 53/27 := by
  sorry

end min_value_expression_l1049_104926


namespace bakery_batches_l1049_104946

/-- The number of baguettes in each batch -/
def baguettes_per_batch : ℕ := 48

/-- The number of baguettes sold after each batch -/
def baguettes_sold : List ℕ := [37, 52, 49]

/-- The number of baguettes left unsold -/
def baguettes_left : ℕ := 6

/-- The number of batches of baguettes the bakery makes a day -/
def num_batches : ℕ := 3

theorem bakery_batches :
  (baguettes_per_batch * num_batches) = (baguettes_sold.sum + baguettes_left) :=
by sorry

end bakery_batches_l1049_104946


namespace similarity_coefficient_bounds_l1049_104941

/-- Two triangles are similar if their corresponding sides are proportional -/
def similar_triangles (x y z p : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ x = k * y ∧ y = k * z ∧ z = k * p

/-- The interval for the similarity coefficient -/
def similarity_coefficient_interval (k : ℝ) : Prop :=
  k > Real.sqrt 5 / 2 - 1 / 2 ∧ k < Real.sqrt 5 / 2 + 1 / 2

/-- Theorem: The similarity coefficient of two similar triangles lies within a specific interval -/
theorem similarity_coefficient_bounds (x y z p : ℝ) (h : x > 0 ∧ y > 0 ∧ z > 0 ∧ p > 0) 
  (h_similar : similar_triangles x y z p) : 
  ∃ k : ℝ, similarity_coefficient_interval k ∧ x = k * y ∧ y = k * z ∧ z = k * p :=
by
  sorry

end similarity_coefficient_bounds_l1049_104941


namespace problem_solution_l1049_104902

noncomputable section

def f (x : ℝ) : ℝ := Real.log ((2 / (x + 1)) - 1) / Real.log 10

def g (a x : ℝ) : ℝ := Real.sqrt (1 - a^2 - 2*a*x - x^2)

def A : Set ℝ := {x : ℝ | (2 / (x + 1)) - 1 > 0}

def B (a : ℝ) : Set ℝ := {x : ℝ | 1 - a^2 - 2*a*x - x^2 ≥ 0}

theorem problem_solution (a : ℝ) :
  (f (1/2013) + f (-1/2013) = 0) ∧
  (∀ a, a ≥ 2 → A ∩ B a = ∅) ∧
  (∃ a, a < 2 ∧ A ∩ B a = ∅) :=
sorry

end

end problem_solution_l1049_104902


namespace foot_of_perpendicular_to_yOz_plane_l1049_104928

/-- The foot of a perpendicular from a point to a plane -/
def foot_of_perpendicular (P : ℝ × ℝ × ℝ) (plane : Set (ℝ × ℝ × ℝ)) : ℝ × ℝ × ℝ :=
  sorry

/-- The yOz plane in ℝ³ -/
def yOz_plane : Set (ℝ × ℝ × ℝ) :=
  {p | p.1 = 0}

theorem foot_of_perpendicular_to_yOz_plane :
  let P : ℝ × ℝ × ℝ := (1, Real.sqrt 2, Real.sqrt 3)
  let Q := foot_of_perpendicular P yOz_plane
  Q = (0, Real.sqrt 2, Real.sqrt 3) :=
sorry

end foot_of_perpendicular_to_yOz_plane_l1049_104928


namespace increase_both_averages_l1049_104959

def group1 : List ℕ := [5, 3, 5, 3, 5, 4, 3, 4, 3, 4, 5, 5]
def group2 : List ℕ := [3, 4, 5, 2, 3, 2, 5, 4, 5, 3]

def average (l : List ℕ) : ℚ := (l.sum : ℚ) / l.length

theorem increase_both_averages :
  ∃ x ∈ group1,
    average (group1.filter (· ≠ x)) > average group1 ∧
    average (x :: group2) > average group2 :=
by sorry

end increase_both_averages_l1049_104959


namespace manager_selection_problem_l1049_104935

theorem manager_selection_problem (n m k : ℕ) (h1 : n = 7) (h2 : m = 4) (h3 : k = 2) :
  (Nat.choose n m) - (Nat.choose (n - k) (m - k)) = 25 := by
  sorry

end manager_selection_problem_l1049_104935


namespace division_remainder_l1049_104971

theorem division_remainder : ∃ q : ℤ, 1346584 = 137 * q + 5 ∧ 0 ≤ 5 ∧ 5 < 137 := by
  sorry

end division_remainder_l1049_104971


namespace dog_food_cans_per_package_l1049_104912

/-- Proves that the number of cans in each package of dog food is 5 -/
theorem dog_food_cans_per_package : 
  ∀ (cat_packages dog_packages cat_cans_per_package : ℕ),
    cat_packages = 9 →
    dog_packages = 7 →
    cat_cans_per_package = 10 →
    cat_packages * cat_cans_per_package = dog_packages * 5 + 55 →
    5 = (cat_packages * cat_cans_per_package - 55) / dog_packages := by
  sorry

end dog_food_cans_per_package_l1049_104912


namespace hyperbola_eccentricity_l1049_104944

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if it forms an acute angle of 60° with the y-axis,
    then its eccentricity is 2√3/3 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_angle : b / a = Real.sqrt 3 / 3) : 
  let e := Real.sqrt (1 + (b / a)^2)
  e = 2 * Real.sqrt 3 / 3 := by sorry

end hyperbola_eccentricity_l1049_104944


namespace kristin_income_l1049_104979

/-- Represents the tax structure and Kristin's income --/
structure TaxSystem where
  p : ℝ  -- base tax rate in decimal form
  income : ℝ  -- Kristin's annual income

/-- Calculates the total tax paid based on the given tax structure --/
def totalTax (ts : TaxSystem) : ℝ :=
  ts.p * 28000 + (ts.p + 0.02) * (ts.income - 28000)

/-- Theorem stating that Kristin's income is $32000 given the tax conditions --/
theorem kristin_income (ts : TaxSystem) :
  (totalTax ts = (ts.p + 0.0025) * ts.income) → ts.income = 32000 := by
  sorry


end kristin_income_l1049_104979


namespace solution_is_negative_two_l1049_104969

-- Define the equation
def fractional_equation (x : ℝ) : Prop :=
  x ≠ 0 ∧ x ≠ 2 ∧ (4 / (x - 2) = 2 / x)

-- Theorem statement
theorem solution_is_negative_two :
  ∃ (x : ℝ), fractional_equation x ∧ x = -2 :=
by sorry

end solution_is_negative_two_l1049_104969


namespace tangent_circle_right_triangle_l1049_104957

/-- Given a right triangle DEF with right angle at E, DF = √85, DE = 7, and a circle with center 
    on DE tangent to DF and EF, prove that FQ = 6 where Q is the point where the circle meets DF. -/
theorem tangent_circle_right_triangle (D E F Q : ℝ × ℝ) 
  (h_right_angle : (E.1 - D.1) * (F.1 - D.1) + (E.2 - D.2) * (F.2 - D.2) = 0)
  (h_df : Real.sqrt ((F.1 - D.1)^2 + (F.2 - D.2)^2) = Real.sqrt 85)
  (h_de : Real.sqrt ((E.1 - D.1)^2 + (E.2 - D.2)^2) = 7)
  (h_circle : ∃ (C : ℝ × ℝ), C ∈ Set.Icc D E ∧ 
    dist C D = dist C Q ∧ dist C E = dist C F ∧ dist C Q = dist C F)
  (h_q_on_df : (Q.1 - D.1) * (F.2 - D.2) = (Q.2 - D.2) * (F.1 - D.1)) :
  dist F Q = 6 := by sorry


end tangent_circle_right_triangle_l1049_104957


namespace arrangement_count_l1049_104993

-- Define the number of red and blue balls
def red_balls : ℕ := 8
def blue_balls : ℕ := 9
def total_balls : ℕ := red_balls + blue_balls

-- Define the number of jars
def num_jars : ℕ := 2

-- Define a function to calculate the number of distinguishable arrangements
def count_arrangements (red : ℕ) (blue : ℕ) (jars : ℕ) : ℕ :=
  sorry -- The actual implementation would go here

-- State the theorem
theorem arrangement_count :
  count_arrangements red_balls blue_balls num_jars = 7 :=
by sorry

end arrangement_count_l1049_104993


namespace race_results_l1049_104994

/-- Represents a runner in the race -/
structure Runner where
  pace : ℕ  -- pace in minutes per mile
  breakTime : ℕ  -- break time in minutes
  breakStart : ℕ  -- time at which the break starts in minutes

/-- Calculates the total time taken by a runner to complete the race -/
def totalTime (r : Runner) (raceDistance : ℕ) : ℕ :=
  let distanceBeforeBreak := r.breakStart / r.pace
  let distanceAfterBreak := raceDistance - distanceBeforeBreak
  r.breakStart + r.breakTime + distanceAfterBreak * r.pace

/-- The main theorem stating the total time for each runner -/
theorem race_results (raceDistance : ℕ) (runner1 runner2 runner3 : Runner) : 
  raceDistance = 15 ∧ 
  runner1.pace = 6 ∧ runner1.breakTime = 3 ∧ runner1.breakStart = 42 ∧
  runner2.pace = 7 ∧ runner2.breakTime = 5 ∧ runner2.breakStart = 49 ∧
  runner3.pace = 8 ∧ runner3.breakTime = 7 ∧ runner3.breakStart = 56 →
  totalTime runner1 raceDistance = 93 ∧
  totalTime runner2 raceDistance = 110 ∧
  totalTime runner3 raceDistance = 127 := by
  sorry

end race_results_l1049_104994


namespace specific_trip_mpg_l1049_104921

/-- Represents a car trip with odometer readings and fuel consumption --/
structure CarTrip where
  initial_odometer : ℕ
  initial_fuel : ℕ
  first_refill : ℕ
  second_refill_odometer : ℕ
  second_refill_amount : ℕ
  final_odometer : ℕ
  final_refill : ℕ

/-- Calculates the average miles per gallon for a car trip --/
def averageMPG (trip : CarTrip) : ℚ :=
  let total_distance := trip.final_odometer - trip.initial_odometer
  let total_fuel := trip.initial_fuel + trip.first_refill + trip.second_refill_amount + trip.final_refill
  (total_distance : ℚ) / total_fuel

/-- The specific car trip from the problem --/
def specificTrip : CarTrip := {
  initial_odometer := 58000
  initial_fuel := 2
  first_refill := 8
  second_refill_odometer := 58400
  second_refill_amount := 15
  final_odometer := 59000
  final_refill := 25
}

/-- Theorem stating that the average MPG for the specific trip is 20.0 --/
theorem specific_trip_mpg : averageMPG specificTrip = 20 := by
  sorry

end specific_trip_mpg_l1049_104921


namespace nabla_example_l1049_104997

-- Define the nabla operation
def nabla (a b : ℕ) : ℕ := 2 + b^a

-- Theorem statement
theorem nabla_example : nabla (nabla 1 2) 3 = 83 := by
  sorry

end nabla_example_l1049_104997


namespace quadratic_equation_coefficients_l1049_104923

/-- 
Given the equation 2x(x+5) = 10, this theorem states that when converted to 
general form ax² + bx + c = 0, the coefficients a, b, and c are 2, 10, and -10 respectively.
-/
theorem quadratic_equation_coefficients : 
  ∃ (a b c : ℝ), (∀ x, 2*x*(x+5) = 10 ↔ a*x^2 + b*x + c = 0) ∧ a = 2 ∧ b = 10 ∧ c = -10 :=
sorry

end quadratic_equation_coefficients_l1049_104923


namespace recipe_total_ingredients_l1049_104983

/-- Represents the ratio of ingredients in the recipe -/
structure RecipeRatio :=
  (butter : ℕ)
  (flour : ℕ)
  (sugar : ℕ)

/-- Calculates the total cups of ingredients given a recipe ratio and cups of sugar -/
def totalIngredients (ratio : RecipeRatio) (sugarCups : ℕ) : ℕ :=
  let partSize := sugarCups / ratio.sugar
  partSize * (ratio.butter + ratio.flour + ratio.sugar)

/-- Theorem stating that for the given recipe ratio and sugar amount, the total ingredients is 28 cups -/
theorem recipe_total_ingredients :
  let ratio : RecipeRatio := ⟨1, 8, 5⟩
  totalIngredients ratio 10 = 28 := by
  sorry

end recipe_total_ingredients_l1049_104983


namespace bruce_grapes_purchase_l1049_104998

theorem bruce_grapes_purchase (grape_price : ℝ) (mango_price : ℝ) (mango_quantity : ℝ) (total_paid : ℝ) :
  grape_price = 70 →
  mango_price = 55 →
  mango_quantity = 10 →
  total_paid = 1110 →
  (total_paid - mango_price * mango_quantity) / grape_price = 8 := by
  sorry

end bruce_grapes_purchase_l1049_104998


namespace floor_ceil_sum_l1049_104960

theorem floor_ceil_sum : ⌊(-3.67 : ℝ)⌋ + ⌈(34.2 : ℝ)⌉ = 31 := by
  sorry

end floor_ceil_sum_l1049_104960


namespace line_through_quadrants_l1049_104992

/-- A line y = kx + b passes through the second, third, and fourth quadrants if and only if
    k and b satisfy the conditions: k + b = -5 and kb = 6 -/
theorem line_through_quadrants (k b : ℝ) : 
  (∀ x y : ℝ, y = k * x + b → 
    ((x < 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0) ∨ (x > 0 ∧ y < 0))) ↔ 
  (k + b = -5 ∧ k * b = 6) := by
  sorry


end line_through_quadrants_l1049_104992


namespace simplify_expression_l1049_104965

theorem simplify_expression (x : ℝ) : (2*x + 20) + (150*x + 20) = 152*x + 40 := by
  sorry

end simplify_expression_l1049_104965


namespace billy_video_count_l1049_104949

/-- The number of videos suggested in each round -/
def suggestions_per_round : ℕ := 15

/-- The number of rounds Billy goes through without liking any videos -/
def unsuccessful_rounds : ℕ := 5

/-- The position of the video Billy watches in the final round -/
def final_video_position : ℕ := 5

/-- The total number of videos Billy watches -/
def total_videos_watched : ℕ := suggestions_per_round * unsuccessful_rounds + 1

theorem billy_video_count :
  total_videos_watched = 76 :=
sorry

end billy_video_count_l1049_104949


namespace dormitory_to_city_distance_prove_dormitory_to_city_distance_l1049_104943

theorem dormitory_to_city_distance : ℝ → Prop :=
  fun D : ℝ =>
    (1/4 : ℝ) * D + (1/2 : ℝ) * D + 10 = D → D = 40

-- The proof is omitted
theorem prove_dormitory_to_city_distance :
  ∃ D : ℝ, dormitory_to_city_distance D :=
by
  sorry

end dormitory_to_city_distance_prove_dormitory_to_city_distance_l1049_104943


namespace integer_pair_problem_l1049_104920

theorem integer_pair_problem (a b q r : ℕ) (h1 : a > b) (h2 : b > 0)
  (h3 : a^2 + b^2 = q * (a + b) + r) (h4 : q^2 + 2*r = 2020) :
  ((a = 53 ∧ b = 29) ∨ (a = 53 ∧ b = 15)) := by
  sorry

end integer_pair_problem_l1049_104920


namespace dennis_initial_money_l1049_104978

def shirt_cost : ℕ := 27
def ten_dollar_bills : ℕ := 2
def loose_coins : ℕ := 3

theorem dennis_initial_money : 
  shirt_cost + ten_dollar_bills * 10 + loose_coins = 50 := by
  sorry

end dennis_initial_money_l1049_104978


namespace max_sections_school_l1049_104973

theorem max_sections_school (num_boys : ℕ) (num_girls : ℕ) (min_boys_per_section : ℕ) (min_girls_per_section : ℕ) 
  (h1 : num_boys = 2016) 
  (h2 : num_girls = 1284) 
  (h3 : min_boys_per_section = 80) 
  (h4 : min_girls_per_section = 60) : 
  (num_boys / min_boys_per_section + num_girls / min_girls_per_section : ℕ) = 46 :=
by
  sorry

end max_sections_school_l1049_104973


namespace coefficient_x2y2_l1049_104925

/-- The coefficient of x²y² in the expansion of (x+y)⁵(c+1/c)⁸ is 700 -/
theorem coefficient_x2y2 : 
  (Finset.sum Finset.univ (fun (k : Fin 6) => 
    Nat.choose 5 k.val * Nat.choose 8 4 * k.val.choose 2)) = 700 := by
  sorry

end coefficient_x2y2_l1049_104925


namespace five_digit_divisible_by_11_l1049_104968

/-- Represents a five-digit number in the form 53A47 -/
def number (A : ℕ) : ℕ := 53000 + A * 100 + 47

/-- Checks if a number is divisible by 11 -/
def divisible_by_11 (n : ℕ) : Prop := n % 11 = 0

theorem five_digit_divisible_by_11 :
  ∃ (A : ℕ), A < 10 ∧ divisible_by_11 (number A) ∧
  ∀ (B : ℕ), B < A → ¬divisible_by_11 (number B) :=
by sorry

end five_digit_divisible_by_11_l1049_104968


namespace average_cost_before_gratuity_l1049_104911

theorem average_cost_before_gratuity 
  (total_people : ℕ) 
  (total_bill : ℚ) 
  (gratuity_rate : ℚ) 
  (h1 : total_people = 9)
  (h2 : total_bill = 756)
  (h3 : gratuity_rate = 1/5) : 
  (total_bill / (1 + gratuity_rate)) / total_people = 70 :=
by sorry

end average_cost_before_gratuity_l1049_104911


namespace remainder_theorem_l1049_104929

theorem remainder_theorem (x : ℤ) : x % 66 = 14 → x % 11 = 3 := by
  sorry

end remainder_theorem_l1049_104929


namespace system_solution_l1049_104907

theorem system_solution : ∃ (x y z : ℝ), 
  (x + y = -1) ∧ (x + z = 0) ∧ (y + z = 1) ∧ (x = -1) ∧ (y = 0) ∧ (z = 1) := by
  sorry

end system_solution_l1049_104907


namespace shelly_money_proof_l1049_104945

/-- Calculates the total amount of money Shelly has given the number of $10 and $5 bills -/
def total_money (ten_dollar_bills : ℕ) (five_dollar_bills : ℕ) : ℕ :=
  10 * ten_dollar_bills + 5 * five_dollar_bills

/-- Proves that Shelly has $390 in total -/
theorem shelly_money_proof :
  let ten_dollar_bills : ℕ := 30
  let five_dollar_bills : ℕ := ten_dollar_bills - 12
  total_money ten_dollar_bills five_dollar_bills = 390 := by
sorry

end shelly_money_proof_l1049_104945
