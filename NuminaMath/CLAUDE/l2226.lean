import Mathlib

namespace storage_unit_capacity_l2226_222634

/-- A storage unit with three shelves for storing CDs. -/
structure StorageUnit where
  shelf1_racks : ℕ
  shelf1_cds_per_rack : ℕ
  shelf2_racks : ℕ
  shelf2_cds_per_rack : ℕ
  shelf3_racks : ℕ
  shelf3_cds_per_rack : ℕ

/-- Calculate the total number of CDs that can fit in a storage unit. -/
def totalCDs (unit : StorageUnit) : ℕ :=
  unit.shelf1_racks * unit.shelf1_cds_per_rack +
  unit.shelf2_racks * unit.shelf2_cds_per_rack +
  unit.shelf3_racks * unit.shelf3_cds_per_rack

/-- Theorem stating that the specific storage unit can hold 116 CDs. -/
theorem storage_unit_capacity :
  let unit : StorageUnit := {
    shelf1_racks := 5,
    shelf1_cds_per_rack := 8,
    shelf2_racks := 4,
    shelf2_cds_per_rack := 10,
    shelf3_racks := 3,
    shelf3_cds_per_rack := 12
  }
  totalCDs unit = 116 := by
  sorry

end storage_unit_capacity_l2226_222634


namespace reservoir_water_supply_l2226_222641

/-- Reservoir water supply problem -/
theorem reservoir_water_supply
  (reservoir_volume : ℝ)
  (initial_population : ℝ)
  (initial_sustainability : ℝ)
  (new_population : ℝ)
  (new_sustainability : ℝ)
  (h_reservoir : reservoir_volume = 120)
  (h_initial_pop : initial_population = 160000)
  (h_initial_sus : initial_sustainability = 20)
  (h_new_pop : new_population = 200000)
  (h_new_sus : new_sustainability = 15) :
  ∃ (annual_precipitation : ℝ) (annual_consumption_pp : ℝ),
    annual_precipitation = 200 ∧
    annual_consumption_pp = 50 ∧
    reservoir_volume + initial_sustainability * annual_precipitation = initial_population * initial_sustainability * annual_consumption_pp / 1000000 ∧
    reservoir_volume + new_sustainability * annual_precipitation = new_population * new_sustainability * annual_consumption_pp / 1000000 :=
by sorry


end reservoir_water_supply_l2226_222641


namespace mans_rowing_speed_l2226_222664

theorem mans_rowing_speed 
  (v : ℝ) -- Man's rowing speed in still water
  (c : ℝ) -- Speed of the current
  (h1 : c = 1.5) -- The current speed is 1.5 km/hr
  (h2 : (v + c) * 1 = (v - c) * 2) -- It takes twice as long to row upstream as downstream
  : v = 4.5 := by
sorry

end mans_rowing_speed_l2226_222664


namespace negation_of_implication_l2226_222661

theorem negation_of_implication (a b : ℝ) :
  ¬(a = 0 ∧ b = 0 → a^2 + b^2 = 0) ↔ (a ≠ 0 ∨ b ≠ 0 → a^2 + b^2 ≠ 0) :=
by sorry

end negation_of_implication_l2226_222661


namespace manolo_face_mask_production_l2226_222643

/-- Represents the face-mask production scenario for Manolo -/
structure FaceMaskProduction where
  initial_rate : ℕ  -- Rate of production in the first hour (minutes per mask)
  total_masks : ℕ   -- Total masks produced in a 4-hour shift
  shift_duration : ℕ -- Total duration of the shift in hours

/-- Calculates the time required to make one face-mask after the first hour -/
def time_per_mask_after_first_hour (p : FaceMaskProduction) : ℕ :=
  let masks_in_first_hour := 60 / p.initial_rate
  let remaining_masks := p.total_masks - masks_in_first_hour
  let remaining_time := (p.shift_duration - 1) * 60
  remaining_time / remaining_masks

/-- Theorem stating that given the initial conditions, the time per mask after the first hour is 6 minutes -/
theorem manolo_face_mask_production :
  ∀ (p : FaceMaskProduction),
    p.initial_rate = 4 ∧
    p.total_masks = 45 ∧
    p.shift_duration = 4 →
    time_per_mask_after_first_hour p = 6 := by
  sorry

end manolo_face_mask_production_l2226_222643


namespace small_circle_radius_l2226_222626

theorem small_circle_radius (R : ℝ) (r : ℝ) : 
  R = 10 →  -- radius of the large circle is 10 meters
  4 * (2 * r) = 2 * R →  -- four diameters of small circles equal the diameter of the large circle
  r = 2.5 :=  -- radius of each small circle is 2.5 meters
by sorry

end small_circle_radius_l2226_222626


namespace optimal_large_trucks_for_fruit_loading_l2226_222695

/-- Represents the problem of loading fruits onto trucks -/
structure FruitLoading where
  total_fruits : ℕ
  large_truck_capacity : ℕ
  small_truck_capacity : ℕ

/-- Checks if a given number of large trucks is optimal for the fruit loading problem -/
def is_optimal_large_trucks (problem : FruitLoading) (num_large_trucks : ℕ) : Prop :=
  let remaining_fruits := problem.total_fruits - num_large_trucks * problem.large_truck_capacity
  -- The remaining fruits can be loaded onto small trucks without leftovers
  remaining_fruits % problem.small_truck_capacity = 0 ∧
  -- Using one more large truck would exceed the total fruits
  (num_large_trucks + 1) * problem.large_truck_capacity > problem.total_fruits

/-- Theorem stating that 8 large trucks is the optimal solution for the given problem -/
theorem optimal_large_trucks_for_fruit_loading :
  let problem : FruitLoading := ⟨134, 15, 7⟩
  is_optimal_large_trucks problem 8 :=
by sorry

end optimal_large_trucks_for_fruit_loading_l2226_222695


namespace polynomial_factorization_l2226_222650

theorem polynomial_factorization 
  (P Q R : Polynomial ℝ) 
  (h : P^4 + Q^4 = R^2) : 
  ∃ (p q r : ℝ) (S : Polynomial ℝ), 
    P = p • S ∧ Q = q • S ∧ R = r • S^2 :=
sorry

end polynomial_factorization_l2226_222650


namespace glass_volume_l2226_222673

theorem glass_volume (V : ℝ) 
  (h1 : 0.4 * V = volume_pessimist)
  (h2 : 0.6 * V = volume_optimist)
  (h3 : volume_optimist - volume_pessimist = 46) :
  V = 230 :=
by
  sorry

end glass_volume_l2226_222673


namespace kindergarten_tissues_l2226_222648

/-- The total number of tissues brought by three kindergartner groups -/
def total_tissues (group1 group2 group3 tissues_per_box : ℕ) : ℕ :=
  (group1 + group2 + group3) * tissues_per_box

/-- Theorem: The total number of tissues brought by the kindergartner groups is 1200 -/
theorem kindergarten_tissues :
  total_tissues 9 10 11 40 = 1200 := by
  sorry

end kindergarten_tissues_l2226_222648


namespace regular_hexagon_area_l2226_222633

/-- The area of a regular hexagon with side length 8 inches is 96√3 square inches. -/
theorem regular_hexagon_area :
  let side_length : ℝ := 8
  let area : ℝ := (3 * Real.sqrt 3 / 2) * side_length ^ 2
  area = 96 * Real.sqrt 3 := by sorry

end regular_hexagon_area_l2226_222633


namespace johns_total_cost_l2226_222622

/-- Calculates the total cost of a cell phone plan --/
def calculate_total_cost (base_cost : ℝ) (text_cost : ℝ) (extra_minute_cost : ℝ) 
  (texts_sent : ℕ) (hours_talked : ℕ) : ℝ :=
  let text_charge := text_cost * texts_sent
  let extra_hours := max (hours_talked - 50) 0
  let extra_minutes := extra_hours * 60
  let extra_minute_charge := extra_minute_cost * extra_minutes
  base_cost + text_charge + extra_minute_charge

/-- Theorem stating that John's total cost is $69.00 --/
theorem johns_total_cost : 
  calculate_total_cost 30 0.10 0.20 150 52 = 69 := by
  sorry

end johns_total_cost_l2226_222622


namespace distinct_prime_factors_of_30_factorial_l2226_222685

theorem distinct_prime_factors_of_30_factorial :
  (Finset.filter Nat.Prime (Finset.range 31)).card = 10 := by sorry

end distinct_prime_factors_of_30_factorial_l2226_222685


namespace only_one_divisible_l2226_222628

theorem only_one_divisible (n : ℕ+) : (3^(n : ℕ) + 1) % (n : ℕ)^2 = 0 → n = 1 := by
  sorry

end only_one_divisible_l2226_222628


namespace fish_offspring_conversion_l2226_222662

/-- Converts a number from base 7 to base 10 --/
def base7ToBase10 (n : ℕ) : ℕ :=
  (n / 100) * 7^2 + ((n / 10) % 10) * 7^1 + (n % 10) * 7^0

/-- The fish offspring count in base 7 --/
def fishOffspringBase7 : ℕ := 265

theorem fish_offspring_conversion :
  base7ToBase10 fishOffspringBase7 = 145 := by
  sorry

end fish_offspring_conversion_l2226_222662


namespace matrix_cube_equals_negative_identity_l2226_222632

def A : Matrix (Fin 2) (Fin 2) ℤ := !![1, -1; 1, 0]

theorem matrix_cube_equals_negative_identity :
  A ^ 3 = !![(-1 : ℤ), 0; 0, -1] := by sorry

end matrix_cube_equals_negative_identity_l2226_222632


namespace min_value_sum_ratios_l2226_222618

theorem min_value_sum_ratios (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / b) + (b / c) + (c / a) + (b / a) ≥ 4 ∧
  ((a / b) + (b / c) + (c / a) + (b / a) = 4 ↔ a = b ∧ b = c) :=
by sorry

end min_value_sum_ratios_l2226_222618


namespace cookie_markup_is_twenty_percent_l2226_222619

/-- The percentage markup on cookies sold by Joe -/
def percentage_markup (num_cookies : ℕ) (total_earned : ℚ) (cost_per_cookie : ℚ) : ℚ :=
  ((total_earned / num_cookies.cast) / cost_per_cookie - 1) * 100

/-- Theorem stating that the percentage markup is 20% given the problem conditions -/
theorem cookie_markup_is_twenty_percent :
  let num_cookies : ℕ := 50
  let total_earned : ℚ := 60
  let cost_per_cookie : ℚ := 1
  percentage_markup num_cookies total_earned cost_per_cookie = 20 := by
sorry

end cookie_markup_is_twenty_percent_l2226_222619


namespace x_squared_plus_reciprocal_l2226_222601

theorem x_squared_plus_reciprocal (x : ℝ) (h : 15 = x^4 + 1/x^4) : x^2 + 1/x^2 = Real.sqrt 17 := by
  sorry

end x_squared_plus_reciprocal_l2226_222601


namespace xy_squared_equals_one_l2226_222681

theorem xy_squared_equals_one (x y : ℝ) (h : |x - 2| + (3 + y)^2 = 0) : (x + y)^2 = 1 := by
  sorry

end xy_squared_equals_one_l2226_222681


namespace solution_set_part1_solution_set_part2_l2226_222607

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 2| + |x + a|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x ≥ 4} = {x : ℝ | x ≤ -3/2 ∨ x ≥ 5/2} :=
sorry

-- Part 2
theorem solution_set_part2 (a : ℝ) :
  ({x : ℝ | f a x ≤ 2*x} = {x : ℝ | x ≥ 1}) → (a = 0 ∨ a = -2) :=
sorry

end solution_set_part1_solution_set_part2_l2226_222607


namespace complex_subtraction_and_multiplication_l2226_222647

theorem complex_subtraction_and_multiplication :
  (7 - 3*I) - 3*(2 + 4*I) = 1 - 15*I :=
by sorry

end complex_subtraction_and_multiplication_l2226_222647


namespace q_value_l2226_222672

theorem q_value (p q : ℝ) 
  (h1 : 1 < p) 
  (h2 : p < q) 
  (h3 : 1/p + 1/q = 1) 
  (h4 : p * q = 8) : 
  q = 4 + 2 * Real.sqrt 2 := by
  sorry

end q_value_l2226_222672


namespace derivative_of_f_l2226_222630

noncomputable def f (x : ℝ) : ℝ := (1 / Real.sqrt 2) * Real.log (Real.sqrt 2 * Real.tan x + Real.sqrt (1 + 2 * Real.tan x ^ 2))

theorem derivative_of_f (x : ℝ) : 
  deriv f x = 1 / (Real.cos x ^ 2 * Real.sqrt (1 + 2 * Real.tan x ^ 2)) :=
by sorry

end derivative_of_f_l2226_222630


namespace prob_rain_weekend_l2226_222684

/-- Probability of rain on Saturday -/
def prob_rain_saturday : ℝ := 0.6

/-- Probability of rain on Sunday given it rained on Saturday -/
def prob_rain_sunday_given_rain_saturday : ℝ := 0.7

/-- Probability of rain on Sunday given it didn't rain on Saturday -/
def prob_rain_sunday_given_no_rain_saturday : ℝ := 0.4

/-- Theorem: The probability of rain over the weekend (at least one day) is 76% -/
theorem prob_rain_weekend : 
  1 - (1 - prob_rain_saturday) * (1 - prob_rain_sunday_given_no_rain_saturday) = 0.76 := by
  sorry

end prob_rain_weekend_l2226_222684


namespace product_digits_sum_l2226_222659

/-- Converts a base-7 number to decimal --/
def base7ToDecimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal number to base-7 --/
def decimalToBase7 (n : ℕ) : ℕ := sorry

/-- Computes the sum of digits of a base-7 number --/
def sumOfDigitsBase7 (n : ℕ) : ℕ := sorry

/-- The product of 24₇ and 35₇ in base-7 --/
def productBase7 : ℕ := decimalToBase7 (base7ToDecimal 24 * base7ToDecimal 35)

theorem product_digits_sum :
  sumOfDigitsBase7 productBase7 = 15 :=
sorry

end product_digits_sum_l2226_222659


namespace fraction_meaningful_iff_l2226_222646

theorem fraction_meaningful_iff (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x + 3)) ↔ x ≠ -3 := by
  sorry

end fraction_meaningful_iff_l2226_222646


namespace mashas_balls_l2226_222629

theorem mashas_balls (r w n p : ℕ) : 
  r + n * w = 101 →
  p * r + w = 103 →
  (r + w = 51 ∨ r + w = 68) :=
by sorry

end mashas_balls_l2226_222629


namespace definite_integral_x_squared_plus_sqrt_one_minus_x_squared_l2226_222683

theorem definite_integral_x_squared_plus_sqrt_one_minus_x_squared :
  ∫ x in (-1)..1, (x^2 + Real.sqrt (1 - x^2)) = 2/3 + π/2 := by sorry

end definite_integral_x_squared_plus_sqrt_one_minus_x_squared_l2226_222683


namespace min_points_dodecahedron_correct_min_points_icosahedron_correct_l2226_222676

/-- A dodecahedron is a polyhedron with 12 faces, where each face is a regular pentagon and each vertex belongs to 3 faces. -/
structure Dodecahedron where
  faces : ℕ
  faces_are_pentagons : Bool
  vertex_face_count : ℕ
  h_faces : faces = 12
  h_pentagons : faces_are_pentagons = true
  h_vertex : vertex_face_count = 3

/-- An icosahedron is a polyhedron with 20 faces and 12 vertices, where each face is an equilateral triangle. -/
structure Icosahedron where
  faces : ℕ
  vertices : ℕ
  faces_are_triangles : Bool
  h_faces : faces = 20
  h_vertices : vertices = 12
  h_triangles : faces_are_triangles = true

/-- The minimum number of points that must be marked on the surface of a dodecahedron
    so that there is at least one marked point on each face. -/
def min_points_dodecahedron (d : Dodecahedron) : ℕ := 4

/-- The minimum number of points that must be marked on the surface of an icosahedron
    so that there is at least one marked point on each face. -/
def min_points_icosahedron (i : Icosahedron) : ℕ := 6

/-- Theorem stating the minimum number of points for a dodecahedron. -/
theorem min_points_dodecahedron_correct (d : Dodecahedron) :
  min_points_dodecahedron d = 4 := by sorry

/-- Theorem stating the minimum number of points for an icosahedron. -/
theorem min_points_icosahedron_correct (i : Icosahedron) :
  min_points_icosahedron i = 6 := by sorry

end min_points_dodecahedron_correct_min_points_icosahedron_correct_l2226_222676


namespace input_statement_incorrect_l2226_222658

-- Define a type for program statements
inductive ProgramStatement
| Input (prompt : String) (value : String)
| Print (prompt : String) (value : String)
| Assignment (left : String) (right : String)

-- Define a function to check if an input statement is valid
def isValidInputStatement (stmt : ProgramStatement) : Prop :=
  match stmt with
  | ProgramStatement.Input _ value => ¬ (value.contains '+' ∨ value.contains '-' ∨ value.contains '*' ∨ value.contains '/')
  | _ => True

-- Theorem to prove
theorem input_statement_incorrect :
  let stmt := ProgramStatement.Input "MATH=" "a+b+c"
  ¬ (isValidInputStatement stmt) := by
sorry

end input_statement_incorrect_l2226_222658


namespace decagon_diagonal_intersections_l2226_222689

/-- The number of distinct interior intersection points of diagonals in a regular decagon -/
def interior_intersection_points (n : ℕ) : ℕ :=
  Nat.choose n 4

/-- Theorem: The number of distinct interior points where two or more diagonals 
    intersect in a regular decagon is equal to C(10,4) -/
theorem decagon_diagonal_intersections : 
  interior_intersection_points 10 = 210 := by
  sorry

#eval interior_intersection_points 10

end decagon_diagonal_intersections_l2226_222689


namespace trajectory_is_straight_line_l2226_222602

/-- The set of points P(x,y) satisfying the given equation forms a straight line -/
theorem trajectory_is_straight_line :
  ∃ (a b c : ℝ), a ≠ 0 ∨ b ≠ 0 ∧
  ∀ (x y : ℝ), Real.sqrt ((x - 1)^2 + (y - 1)^2) = |x + y - 2| / Real.sqrt 2 →
  a * x + b * y + c = 0 := by
  sorry

end trajectory_is_straight_line_l2226_222602


namespace rachel_money_left_l2226_222698

theorem rachel_money_left (earnings : ℝ) (lunch_fraction : ℝ) (dvd_fraction : ℝ) : 
  earnings = 200 →
  lunch_fraction = 1/4 →
  dvd_fraction = 1/2 →
  earnings - (lunch_fraction * earnings + dvd_fraction * earnings) = 50 := by
sorry

end rachel_money_left_l2226_222698


namespace lemonade_proportion_lemons_for_lemonade_l2226_222688

theorem lemonade_proportion (lemons_initial : ℝ) (gallons_initial : ℝ) (gallons_target : ℝ) :
  lemons_initial > 0 ∧ gallons_initial > 0 ∧ gallons_target > 0 →
  let lemons_target := (lemons_initial * gallons_target) / gallons_initial
  lemons_initial / gallons_initial = lemons_target / gallons_target :=
by
  sorry

theorem lemons_for_lemonade :
  let lemons_initial : ℝ := 36
  let gallons_initial : ℝ := 48
  let gallons_target : ℝ := 10
  (lemons_initial * gallons_target) / gallons_initial = 7.5 :=
by
  sorry

end lemonade_proportion_lemons_for_lemonade_l2226_222688


namespace number_plus_thrice_value_l2226_222697

theorem number_plus_thrice_value (x : ℕ) (value : ℕ) : x = 5 → x + 3 * x = value → value = 20 := by
  sorry

end number_plus_thrice_value_l2226_222697


namespace min_value_expression_l2226_222655

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + 1/b) * (b + 4/a) ≥ 9 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ (a₀ + 1/b₀) * (b₀ + 4/a₀) = 9 := by
  sorry

end min_value_expression_l2226_222655


namespace car_distance_l2226_222637

theorem car_distance (total_distance : ℝ) (foot_fraction : ℝ) (bus_fraction : ℝ) :
  total_distance = 90 →
  foot_fraction = 1/5 →
  bus_fraction = 2/3 →
  total_distance * (1 - foot_fraction - bus_fraction) = 12 := by
  sorry

end car_distance_l2226_222637


namespace complex_quotient_real_l2226_222678

theorem complex_quotient_real (t : ℝ) : 
  let z₁ : ℂ := 2*t + Complex.I
  let z₂ : ℂ := 1 - 2*Complex.I
  (∃ (r : ℝ), z₁ / z₂ = r) → t = -1/4 := by
sorry

end complex_quotient_real_l2226_222678


namespace flu_infection_spread_l2226_222657

/-- The average number of people infected by one person in each round of infection -/
def average_infections : ℕ := 13

/-- The number of rounds of infection -/
def num_rounds : ℕ := 2

/-- The total number of people infected after two rounds -/
def total_infected : ℕ := 196

/-- The number of initially infected people -/
def initial_infected : ℕ := 1

theorem flu_infection_spread :
  (initial_infected + average_infections * initial_infected + 
   average_infections * (initial_infected + average_infections * initial_infected) = total_infected) ∧
  (average_infections > 0) := by
  sorry

end flu_infection_spread_l2226_222657


namespace line_through_points_l2226_222642

/-- The equation of a line passing through two points (x₁, y₁) and (x₂, y₂) -/
def line_equation (x₁ y₁ x₂ y₂ : ℝ) (x y : ℝ) : Prop :=
  (y - y₁) * (x₂ - x₁) = (x - x₁) * (y₂ - y₁)

/-- Theorem: The equation of the line passing through (1, 0) and (0, 1) is x + y - 1 = 0 -/
theorem line_through_points : 
  ∀ x y : ℝ, line_equation 1 0 0 1 x y ↔ x + y - 1 = 0 := by
  sorry

#check line_through_points

end line_through_points_l2226_222642


namespace sale_recording_l2226_222652

/-- Represents the inventory change for a given number of items. -/
def inventoryChange (items : ℤ) : ℤ := items

/-- The bookkeeping convention for recording purchases. -/
axiom purchase_convention (items : ℕ) : inventoryChange items = items

/-- Theorem: The sale of 5 items should be recorded as -5. -/
theorem sale_recording : inventoryChange (-5) = -5 := by
  sorry

end sale_recording_l2226_222652


namespace adams_total_school_time_l2226_222671

/-- The time Adam spent at school on each day of the week --/
structure SchoolWeek where
  monday : Float
  tuesday : Float
  wednesday : Float
  thursday : Float
  friday : Float

/-- Calculate the total time Adam spent at school during the week --/
def totalSchoolTime (week : SchoolWeek) : Float :=
  week.monday + week.tuesday + week.wednesday + week.thursday + week.friday

/-- Adam's actual school week --/
def adamsWeek : SchoolWeek := {
  monday := 7.75,
  tuesday := 5.75,
  wednesday := 13.5,
  thursday := 8,
  friday := 6.75
}

/-- Theorem stating that Adam's total school time for the week is 41.75 hours --/
theorem adams_total_school_time :
  totalSchoolTime adamsWeek = 41.75 := by
  sorry


end adams_total_school_time_l2226_222671


namespace area_code_combinations_l2226_222621

/-- The number of digits in the area code -/
def n : ℕ := 4

/-- The set of digits used in the area code -/
def digits : Finset ℕ := {9, 8, 7, 6}

/-- The number of possible combinations for the area code -/
def num_combinations : ℕ := n.factorial

theorem area_code_combinations :
  Finset.card (Finset.powerset digits) = n ∧ num_combinations = 24 := by sorry

end area_code_combinations_l2226_222621


namespace car_speed_l2226_222612

/-- Given a car that travels 375 km in 3 hours, its speed is 125 km/h -/
theorem car_speed (distance : ℝ) (time : ℝ) (speed : ℝ) : 
  distance = 375 ∧ time = 3 → speed = distance / time → speed = 125 := by
sorry

end car_speed_l2226_222612


namespace complex_modulus_problem_l2226_222645

theorem complex_modulus_problem (a b : ℝ) :
  (a + Complex.I) * (1 - Complex.I) = 3 + b * Complex.I →
  Complex.abs (a + b * Complex.I) = Real.sqrt 5 := by
  sorry

end complex_modulus_problem_l2226_222645


namespace rectangle_area_constant_l2226_222600

theorem rectangle_area_constant (d : ℝ) (h : d > 0) : 
  ∃ (w l : ℝ), w > 0 ∧ l > 0 ∧ w / l = 3 / 5 ∧ w ^ 2 + l ^ 2 = (10 * d) ^ 2 ∧ w * l = (750 / 17) * d ^ 2 := by
  sorry

end rectangle_area_constant_l2226_222600


namespace equal_area_rectangles_width_l2226_222620

/-- Given two rectangles of equal area, where one rectangle measures 12 inches by 15 inches
    and the other has a length of 9 inches, prove that the width of the second rectangle is 20 inches. -/
theorem equal_area_rectangles_width (area carol_length carol_width jordan_length jordan_width : ℝ) :
  area = carol_length * carol_width →
  area = jordan_length * jordan_width →
  carol_length = 12 →
  carol_width = 15 →
  jordan_length = 9 →
  jordan_width = 20 := by
sorry

end equal_area_rectangles_width_l2226_222620


namespace salary_increase_l2226_222614

-- Define the salary function
def salary (x : ℝ) : ℝ := 60 + 90 * x

-- State the theorem
theorem salary_increase (x : ℝ) :
  salary (x + 1) - salary x = 90 := by
  sorry

end salary_increase_l2226_222614


namespace identity_is_unique_solution_l2226_222640

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 + f y) = y + (f x)^2

/-- The theorem stating that the identity function is the only solution -/
theorem identity_is_unique_solution :
  ∀ f : ℝ → ℝ, SatisfiesFunctionalEquation f → (∀ x : ℝ, f x = x) :=
by sorry

end identity_is_unique_solution_l2226_222640


namespace rebus_solution_l2226_222679

theorem rebus_solution : ∃! (A B C : ℕ), 
  (A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0) ∧ 
  (A ≠ B ∧ B ≠ C ∧ A ≠ C) ∧
  (100 * A + 10 * B + A) + (100 * A + 10 * B + C) + (100 * A + 10 * C + C) = 1416 ∧
  A = 4 ∧ B = 7 ∧ C = 6 := by
sorry

end rebus_solution_l2226_222679


namespace john_has_14_burritos_left_l2226_222639

/-- The number of burritos John has left after buying, receiving a free box, giving away some, and eating for 10 days. -/
def burritos_left : ℕ :=
  let total_burritos : ℕ := 15 + 20 + 25 + 5
  let given_away : ℕ := (total_burritos / 3 : ℕ)
  let after_giving : ℕ := total_burritos - given_away
  let eaten : ℕ := 3 * 10
  after_giving - eaten

/-- Theorem stating that John has 14 burritos left -/
theorem john_has_14_burritos_left : burritos_left = 14 := by
  sorry

end john_has_14_burritos_left_l2226_222639


namespace total_hot_dog_cost_l2226_222674

def hot_dog_cost (group : ℕ) (quantity : ℕ) (price : ℚ) : ℚ :=
  quantity * price

theorem total_hot_dog_cost : 
  let group1_cost := hot_dog_cost 1 4 0.60
  let group2_cost := hot_dog_cost 2 5 0.75
  let group3_cost := hot_dog_cost 3 3 0.90
  group1_cost + group2_cost + group3_cost = 8.85 := by
  sorry

end total_hot_dog_cost_l2226_222674


namespace ribbon_count_l2226_222660

theorem ribbon_count (morning_given afternoon_given remaining : ℕ) 
  (h1 : morning_given = 14)
  (h2 : afternoon_given = 16)
  (h3 : remaining = 8) :
  morning_given + afternoon_given + remaining = 38 := by
  sorry

end ribbon_count_l2226_222660


namespace inequality_proof_l2226_222605

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ((x + y + z) / 3) ^ (x + y + z) ≤ x^x * y^y * z^z ∧
  x^x * y^y * z^z ≤ ((x^2 + y^2 + z^2) / (x + y + z)) ^ (x + y + z) := by
sorry

end inequality_proof_l2226_222605


namespace amy_spelling_problems_l2226_222631

/-- The number of spelling problems Amy had to solve -/
def spelling_problems (total_problems math_problems : ℕ) : ℕ :=
  total_problems - math_problems

/-- Proof that Amy had 6 spelling problems -/
theorem amy_spelling_problems :
  spelling_problems 24 18 = 6 := by
  sorry

end amy_spelling_problems_l2226_222631


namespace horner_V₃_eq_9_l2226_222613

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℚ) (x : ℚ) : ℚ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 2x^5 - 3x^4 + 7x^3 - 9x^2 + 4x - 10 -/
def f : List ℚ := [2, -3, 7, -9, 4, -10]

/-- V₃ in Horner's method for f(x) at x = 2 -/
def V₃ : ℚ := horner [2, -3, 7] 2

theorem horner_V₃_eq_9 : V₃ = 9 := by
  sorry

end horner_V₃_eq_9_l2226_222613


namespace tan_45_degrees_l2226_222665

theorem tan_45_degrees : Real.tan (π / 4) = 1 := by
  sorry

end tan_45_degrees_l2226_222665


namespace log_half_inequality_condition_l2226_222615

-- Define the logarithm function with base 1/2
noncomputable def log_half (x : ℝ) := Real.log x / Real.log (1/2)

theorem log_half_inequality_condition (x : ℝ) (hx : x ∈ Set.Ioo 0 (1/2)) :
  (∀ a : ℝ, a < 0 → log_half x > x + a) ∧
  ∃ a : ℝ, a ≥ 0 ∧ log_half x > x + a :=
by
  sorry

#check log_half_inequality_condition

end log_half_inequality_condition_l2226_222615


namespace consecutive_decreasing_difference_l2226_222611

/-- Represents a three-digit number with consecutive decreasing digits -/
structure ConsecutiveDecreasingNumber where
  x : ℕ
  h1 : x ≥ 1
  h2 : x ≤ 7

/-- Calculates the value of a three-digit number given its digits -/
def number_value (n : ConsecutiveDecreasingNumber) : ℕ :=
  100 * (n.x + 2) + 10 * (n.x + 1) + n.x

/-- Calculates the value of the reversed three-digit number given its digits -/
def reversed_value (n : ConsecutiveDecreasingNumber) : ℕ :=
  100 * n.x + 10 * (n.x + 1) + (n.x + 2)

/-- Theorem stating that the difference between a three-digit number with consecutive 
    decreasing digits and its reverse is always 198 -/
theorem consecutive_decreasing_difference 
  (n : ConsecutiveDecreasingNumber) : 
  number_value n - reversed_value n = 198 := by
  sorry

end consecutive_decreasing_difference_l2226_222611


namespace expression_is_factored_l2226_222651

/-- Represents a quadratic expression of the form ax^2 + bx + c -/
structure QuadraticExpression (α : Type*) [Ring α] where
  a : α
  b : α
  c : α

/-- Represents a factored quadratic expression of the form (x - r)^2 -/
structure FactoredQuadratic (α : Type*) [Ring α] where
  r : α

/-- Checks if a quadratic expression is factored from left to right -/
def is_factored_left_to_right {α : Type*} [Ring α] (q : QuadraticExpression α) (f : FactoredQuadratic α) : Prop :=
  q.a = 1 ∧ q.b = -2 * f.r ∧ q.c = f.r^2

/-- The given quadratic expression x^2 - 6x + 9 -/
def given_expression : QuadraticExpression ℤ := ⟨1, -6, 9⟩

/-- The factored form (x - 3)^2 -/
def factored_form : FactoredQuadratic ℤ := ⟨3⟩

/-- Theorem stating that the given expression represents factorization from left to right -/
theorem expression_is_factored : is_factored_left_to_right given_expression factored_form := by
  sorry

end expression_is_factored_l2226_222651


namespace investment_dividend_theorem_l2226_222627

/-- Calculates the dividend received from an investment in shares with premium and dividend rate --/
def calculate_dividend (investment : ℚ) (share_value : ℚ) (premium_rate : ℚ) (dividend_rate : ℚ) : ℚ :=
  let share_cost := share_value * (1 + premium_rate)
  let num_shares := investment / share_cost
  let dividend_per_share := share_value * dividend_rate
  num_shares * dividend_per_share

/-- Theorem: Given the specified investment conditions, the dividend received is 600 --/
theorem investment_dividend_theorem (investment : ℚ) (share_value : ℚ) (premium_rate : ℚ) (dividend_rate : ℚ)
  (h1 : investment = 14400)
  (h2 : share_value = 100)
  (h3 : premium_rate = 1/5)
  (h4 : dividend_rate = 1/20) :
  calculate_dividend investment share_value premium_rate dividend_rate = 600 := by
  sorry

#eval calculate_dividend 14400 100 (1/5) (1/20)

end investment_dividend_theorem_l2226_222627


namespace min_value_of_sum_l2226_222691

theorem min_value_of_sum (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∀ x y : ℝ, a * x + 2 * b * y - 2 = 0) → 
  (∀ x y : ℝ, x^2 + y^2 - 4*x - 2*y - 8 = 0) → 
  (∀ x y : ℝ, a * x + 2 * b * y - 2 = 0 → 
    (x - 2)^2 + (y - 1)^2 = 9) → 
  (1 / (2 * a) + 1 / b) ≥ (3 + 2 * Real.sqrt 2) / 2 :=
by sorry

end min_value_of_sum_l2226_222691


namespace equation_solution_l2226_222606

theorem equation_solution :
  ∀ x : ℝ, (Real.sqrt (9 * x - 2) + 18 / Real.sqrt (9 * x - 2) = 11) ↔ (x = 83 / 9 ∨ x = 2 / 3) :=
by sorry

end equation_solution_l2226_222606


namespace oil_truck_tank_radius_l2226_222675

/-- Represents a right circular cylinder -/
structure RightCircularCylinder where
  radius : ℝ
  height : ℝ

/-- The problem statement -/
theorem oil_truck_tank_radius 
  (stationary_tank : RightCircularCylinder)
  (oil_truck_tank : RightCircularCylinder)
  (oil_level_drop : ℝ)
  (h_stationary_radius : stationary_tank.radius = 100)
  (h_stationary_height : stationary_tank.height = 25)
  (h_truck_height : oil_truck_tank.height = 10)
  (h_oil_drop : oil_level_drop = 0.025)
  (h_volume_equality : π * stationary_tank.radius^2 * oil_level_drop = 
                       π * oil_truck_tank.radius^2 * oil_truck_tank.height) :
  oil_truck_tank.radius = 5 := by
  sorry

#check oil_truck_tank_radius

end oil_truck_tank_radius_l2226_222675


namespace lcm_gcf_problem_l2226_222699

theorem lcm_gcf_problem (n : ℕ) :
  Nat.lcm n 12 = 54 ∧ Nat.gcd n 12 = 8 → n = 36 := by sorry

end lcm_gcf_problem_l2226_222699


namespace ceiling_floor_difference_l2226_222649

theorem ceiling_floor_difference : ⌈(15 : ℝ) / 8 * (-34 : ℝ) / 4⌉ - ⌊(15 : ℝ) / 8 * ⌊(-34 : ℝ) / 4⌋⌋ = 1 := by
  sorry

end ceiling_floor_difference_l2226_222649


namespace greatest_power_of_two_factor_l2226_222692

theorem greatest_power_of_two_factor (n : ℕ) : 
  ∃ (k : ℕ), (2^k : ℤ) ∣ (12^500 - 6^500) ∧ 
  ∀ (m : ℕ), (2^m : ℤ) ∣ (12^500 - 6^500) → m ≤ k :=
by
  use 501
  sorry

end greatest_power_of_two_factor_l2226_222692


namespace brown_dogs_count_l2226_222690

/-- Represents the number of dogs in a kennel with specific characteristics. -/
structure DogKennel where
  total : ℕ
  longFur : ℕ
  neitherLongFurNorBrown : ℕ
  longFurAndBrown : ℕ

/-- Theorem stating the number of brown dogs in the kennel. -/
theorem brown_dogs_count (k : DogKennel)
    (h1 : k.total = 45)
    (h2 : k.longFur = 29)
    (h3 : k.neitherLongFurNorBrown = 8)
    (h4 : k.longFurAndBrown = 9) :
    k.total - k.neitherLongFurNorBrown - (k.longFur - k.longFurAndBrown) = 17 := by
  sorry

#check brown_dogs_count

end brown_dogs_count_l2226_222690


namespace cubes_with_le_four_neighbors_eq_144_l2226_222693

/-- Represents a parallelepiped constructed from unit cubes. -/
structure Parallelepiped where
  a : ℕ
  b : ℕ
  c : ℕ
  sides_gt_four : min a (min b c) > 4
  internal_cubes : (a - 2) * (b - 2) * (c - 2) = 836

/-- The number of cubes with no more than four neighbors in the parallelepiped. -/
def cubes_with_le_four_neighbors (p : Parallelepiped) : ℕ :=
  4 * (p.a - 2 + p.b - 2 + p.c - 2) + 8

/-- Theorem stating that the number of cubes with no more than four neighbors is 144. -/
theorem cubes_with_le_four_neighbors_eq_144 (p : Parallelepiped) :
  cubes_with_le_four_neighbors p = 144 := by
  sorry

end cubes_with_le_four_neighbors_eq_144_l2226_222693


namespace box_ratio_proof_l2226_222682

def box_problem (total_balls white_balls : ℕ) (blue_white_diff : ℕ) : Prop :=
  let blue_balls : ℕ := white_balls + blue_white_diff
  let red_balls : ℕ := total_balls - (white_balls + blue_balls)
  (red_balls : ℚ) / blue_balls = 2 / 1

theorem box_ratio_proof :
  box_problem 100 16 12 := by
  sorry

end box_ratio_proof_l2226_222682


namespace congruence_problem_l2226_222617

theorem congruence_problem : ∃! n : ℤ, 0 ≤ n ∧ n < 31 ∧ -527 ≡ n [ZMOD 31] ∧ n = 0 := by
  sorry

end congruence_problem_l2226_222617


namespace sum_digits_888_base_8_l2226_222644

/-- Represents a number in base 8 as a list of digits (least significant digit first) -/
def BaseEightRepresentation := List Nat

/-- Converts a natural number from base 10 to base 8 -/
def toBaseEight (n : Nat) : BaseEightRepresentation :=
  sorry

/-- Calculates the sum of digits in a base 8 representation -/
def sumDigits (repr : BaseEightRepresentation) : Nat :=
  sorry

theorem sum_digits_888_base_8 :
  sumDigits (toBaseEight 888) = 13 := by
  sorry

end sum_digits_888_base_8_l2226_222644


namespace library_books_count_l2226_222635

theorem library_books_count (num_bookshelves : ℕ) (floors_per_bookshelf : ℕ) (books_per_floor : ℕ) :
  num_bookshelves = 28 →
  floors_per_bookshelf = 6 →
  books_per_floor = 19 →
  num_bookshelves * floors_per_bookshelf * books_per_floor = 3192 :=
by
  sorry

end library_books_count_l2226_222635


namespace girls_in_sample_l2226_222666

/-- Calculates the number of girls in a stratified sample -/
def stratified_sample_girls (total_students : ℕ) (total_girls : ℕ) (sample_size : ℕ) : ℕ :=
  (sample_size * total_girls) / total_students

/-- Proves that the number of girls in the stratified sample is 2 -/
theorem girls_in_sample (total_boys : ℕ) (total_girls : ℕ) (sample_size : ℕ) 
  (h1 : total_boys = 36)
  (h2 : total_girls = 18)
  (h3 : sample_size = 6) :
  stratified_sample_girls (total_boys + total_girls) total_girls sample_size = 2 := by
  sorry

#eval stratified_sample_girls 54 18 6

end girls_in_sample_l2226_222666


namespace larger_number_problem_l2226_222668

theorem larger_number_problem (x y : ℝ) : 
  x + y = 84 → y = 3 * x → max x y = 63 := by
  sorry

end larger_number_problem_l2226_222668


namespace jellybean_average_proof_l2226_222616

/-- Proves that the initial average number of jellybeans per bag was 117,
    given the conditions of the problem. -/
theorem jellybean_average_proof 
  (initial_bags : ℕ) 
  (new_bag_jellybeans : ℕ) 
  (average_increase : ℕ) 
  (h1 : initial_bags = 34)
  (h2 : new_bag_jellybeans = 362)
  (h3 : average_increase = 7) :
  ∃ (initial_average : ℕ),
    (initial_average * initial_bags + new_bag_jellybeans) / (initial_bags + 1) = 
    initial_average + average_increase ∧ 
    initial_average = 117 := by
  sorry

end jellybean_average_proof_l2226_222616


namespace systematic_sample_validity_l2226_222680

def isValidSystematicSample (sample : List Nat) (populationSize : Nat) (sampleSize : Nat) : Prop :=
  sample.length = sampleSize ∧
  sample.all (· ≤ populationSize) ∧
  sample.all (· > 0) ∧
  ∃ k : Nat, k > 0 ∧ List.zipWith (·-·) (sample.tail) sample = List.replicate (sampleSize - 1) k

theorem systematic_sample_validity :
  isValidSystematicSample [3, 13, 23, 33, 43] 50 5 :=
sorry

end systematic_sample_validity_l2226_222680


namespace positive_reals_inequalities_l2226_222623

theorem positive_reals_inequalities (x y : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_sum : x + y = 1) : 
  x + y - 4*x*y ≥ 0 ∧ 1/x + 4/(1+y) ≥ 9/2 := by
  sorry

end positive_reals_inequalities_l2226_222623


namespace quadratic_equation_solutions_cubic_equation_solutions_l2226_222696

theorem quadratic_equation_solutions (x : ℝ) :
  (x^2 + 2*x - 4 = 0) ↔ (x = Real.sqrt 5 - 1 ∨ x = -Real.sqrt 5 - 1) :=
sorry

theorem cubic_equation_solutions (x : ℝ) :
  (3*x*(x-5) = 5-x) ↔ (x = 5 ∨ x = -1/3) :=
sorry

end quadratic_equation_solutions_cubic_equation_solutions_l2226_222696


namespace race_distance_theorem_l2226_222663

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  (speed_pos : speed > 0)

/-- Calculates the distance covered by a runner in a given time -/
def distance (r : Runner) (t : ℝ) : ℝ := r.speed * t

theorem race_distance_theorem 
  (A B C : Runner) 
  (race_length : ℝ)
  (AB_difference : ℝ)
  (BC_difference : ℝ)
  (h1 : race_length = 100)
  (h2 : AB_difference = 10)
  (h3 : BC_difference = 10)
  (h4 : distance A (race_length / A.speed) = race_length)
  (h5 : distance B (race_length / A.speed) = race_length - AB_difference)
  (h6 : distance C (race_length / B.speed) = race_length - BC_difference) :
  distance C (race_length / A.speed) = race_length - 19 := by
  sorry


end race_distance_theorem_l2226_222663


namespace square_difference_fourth_power_l2226_222609

theorem square_difference_fourth_power : (7^2 - 6^2)^4 = 28561 := by
  sorry

end square_difference_fourth_power_l2226_222609


namespace same_name_pair_exists_l2226_222654

theorem same_name_pair_exists (n : ℕ) (h_n : n = 33) :
  ∀ (first_name_groups last_name_groups : Fin n → Fin 11),
    (∀ i : Fin 11, ∃ j : Fin n, first_name_groups j = i) →
    (∀ i : Fin 11, ∃ j : Fin n, last_name_groups j = i) →
    ∃ x y : Fin n, x ≠ y ∧ first_name_groups x = first_name_groups y ∧ last_name_groups x = last_name_groups y :=
by
  sorry

#check same_name_pair_exists

end same_name_pair_exists_l2226_222654


namespace min_selling_price_A_is_190_l2226_222604

/-- Represents the number of units and prices of water purifiers --/
structure WaterPurifiers where
  units_A : ℕ
  units_B : ℕ
  cost_A : ℕ
  cost_B : ℕ
  total_cost : ℕ

/-- Calculates the minimum selling price of model A --/
def min_selling_price_A (w : WaterPurifiers) : ℕ :=
  w.cost_A + (w.total_cost - w.units_A * w.cost_A - w.units_B * w.cost_B) / w.units_A

/-- Theorem stating the minimum selling price of model A --/
theorem min_selling_price_A_is_190 (w : WaterPurifiers) 
  (h1 : w.units_A + w.units_B = 100)
  (h2 : w.units_A * w.cost_A + w.units_B * w.cost_B = w.total_cost)
  (h3 : w.cost_A = 150)
  (h4 : w.cost_B = 250)
  (h5 : w.total_cost = 19000)
  (h6 : ∀ (sell_A : ℕ), 
    (sell_A - w.cost_A) * w.units_A + 2 * (sell_A - w.cost_A) * w.units_B ≥ 5600 → 
    min_selling_price_A w ≤ sell_A) :
  min_selling_price_A w = 190 := by
  sorry

#eval min_selling_price_A ⟨60, 40, 150, 250, 19000⟩

end min_selling_price_A_is_190_l2226_222604


namespace sqrt_simplification_l2226_222603

theorem sqrt_simplification : (5 - 3 * Real.sqrt 2) ^ 2 = 45 - 28 * Real.sqrt 2 := by
  sorry

end sqrt_simplification_l2226_222603


namespace annie_ride_distance_l2226_222670

/-- Taxi fare calculation --/
def taxi_fare (start_fee : ℚ) (toll : ℚ) (per_mile : ℚ) (miles : ℚ) : ℚ :=
  start_fee + toll + per_mile * miles

theorem annie_ride_distance :
  let mike_start_fee : ℚ := 25/10
  let annie_start_fee : ℚ := 25/10
  let mike_toll : ℚ := 0
  let annie_toll : ℚ := 5
  let per_mile : ℚ := 1/4
  let mike_miles : ℚ := 34
  let annie_miles : ℚ := 14

  taxi_fare mike_start_fee mike_toll per_mile mike_miles =
  taxi_fare annie_start_fee annie_toll per_mile annie_miles :=
by
  sorry


end annie_ride_distance_l2226_222670


namespace boat_speed_in_still_water_l2226_222667

/-- The speed of a boat in still water, given its downstream and upstream distances in one hour -/
theorem boat_speed_in_still_water (downstream upstream : ℝ) 
  (h_downstream : downstream = 11) 
  (h_upstream : upstream = 5) : 
  (downstream + upstream) / 2 = 8 := by
  sorry

end boat_speed_in_still_water_l2226_222667


namespace num_acceptance_configs_prove_num_acceptance_configs_l2226_222608

/-- Represents the number of students -/
def num_students : ℕ := 4

/-- Represents the minimum number of companies -/
def min_companies : ℕ := 3

/-- Represents the acceptance configuration -/
structure AcceptanceConfig where
  student_acceptances : Fin num_students → ℕ
  company_acceptances : ℕ → Fin num_students → Bool
  each_student_diff : ∀ (i j : Fin num_students), i ≠ j → student_acceptances i ≠ student_acceptances j
  student_order : ∀ (i j : Fin num_students), i < j → student_acceptances i < student_acceptances j
  company_nonempty : ∀ (c : ℕ), c < min_companies → ∃ (s : Fin num_students), company_acceptances c s = true

/-- The main theorem stating the number of valid acceptance configurations -/
theorem num_acceptance_configs : (AcceptanceConfig → Prop) → ℕ := 60

/-- Proof of the theorem -/
theorem prove_num_acceptance_configs : num_acceptance_configs = 60 := by sorry

end num_acceptance_configs_prove_num_acceptance_configs_l2226_222608


namespace fraction_equals_zero_l2226_222638

theorem fraction_equals_zero (x : ℝ) (h : 6 * x ≠ 0) :
  (x - 5) / (6 * x) = 0 ↔ x = 5 := by
  sorry

end fraction_equals_zero_l2226_222638


namespace lcm_equality_implies_equal_no_lcm_equality_with_shift_l2226_222686

theorem lcm_equality_implies_equal (a b : ℕ+) :
  Nat.lcm a (a + 5) = Nat.lcm b (b + 5) → a = b := by sorry

theorem no_lcm_equality_with_shift :
  ¬ ∃ (a b c : ℕ+), Nat.lcm a b = Nat.lcm (a + c) (b + c) := by sorry

end lcm_equality_implies_equal_no_lcm_equality_with_shift_l2226_222686


namespace three_digit_reverse_subtraction_l2226_222656

theorem three_digit_reverse_subtraction (b c : ℕ) : 
  (0 < c) ∧ (c < 10) ∧ (b < 10) → 
  (101*c + 10*b + 300) - (101*c + 10*b + 3) = 297 := by
  sorry

#check three_digit_reverse_subtraction

end three_digit_reverse_subtraction_l2226_222656


namespace problem_statement_l2226_222624

theorem problem_statement (a : ℝ) : 
  (∀ x : ℝ, x > 0 → (x - a + 2) * (x^2 - a*x - 2) ≥ 0) → a = 1 := by
  sorry

end problem_statement_l2226_222624


namespace remaining_questions_to_write_l2226_222677

theorem remaining_questions_to_write
  (total_multiple_choice : ℕ)
  (total_problem_solving : ℕ)
  (total_true_false : ℕ)
  (fraction_multiple_choice_written : ℚ)
  (fraction_problem_solving_written : ℚ)
  (fraction_true_false_written : ℚ)
  (h1 : total_multiple_choice = 35)
  (h2 : total_problem_solving = 15)
  (h3 : total_true_false = 20)
  (h4 : fraction_multiple_choice_written = 3/7)
  (h5 : fraction_problem_solving_written = 1/5)
  (h6 : fraction_true_false_written = 1/4) :
  (total_multiple_choice - (fraction_multiple_choice_written * total_multiple_choice).num) +
  (total_problem_solving - (fraction_problem_solving_written * total_problem_solving).num) +
  (total_true_false - (fraction_true_false_written * total_true_false).num) = 47 :=
by sorry

end remaining_questions_to_write_l2226_222677


namespace min_value_a_plus_2b_min_value_is_2_sqrt_2_equality_condition_l2226_222610

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 1) :
  ∀ x y : ℝ, x > 0 → y > 0 → x * y = 1 → a + 2 * b ≤ x + 2 * y :=
by sorry

theorem min_value_is_2_sqrt_2 (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 1) :
  a + 2 * b ≥ 2 * Real.sqrt 2 :=
by sorry

theorem equality_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 1) :
  a + 2 * b = 2 * Real.sqrt 2 ↔ a = Real.sqrt 2 ∧ b = Real.sqrt 2 / 2 :=
by sorry

end min_value_a_plus_2b_min_value_is_2_sqrt_2_equality_condition_l2226_222610


namespace inequality_solution_set_l2226_222625

theorem inequality_solution_set (x : ℝ) : 
  (((x + 1) / (x - 2) + (x + 3) / (3 * x) ≥ 4) ↔ 
  (x > -1/4 ∧ x < 0) ∨ (x ≥ 3/2 ∧ x < 2)) :=
by sorry

end inequality_solution_set_l2226_222625


namespace odd_monotonous_unique_zero_implies_k_is_quarter_l2226_222653

/-- A function is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function is monotonous if it's either increasing or decreasing -/
def IsMonotonous (f : ℝ → ℝ) : Prop :=
  (∀ x y, x < y → f x < f y) ∨ (∀ x y, x < y → f x > f y)

/-- A function has only one zero point if there exists exactly one x such that f(x) = 0 -/
def HasUniqueZero (f : ℝ → ℝ) : Prop :=
  ∃! x, f x = 0

theorem odd_monotonous_unique_zero_implies_k_is_quarter
    (f : ℝ → ℝ) (k : ℝ)
    (h_odd : IsOdd f)
    (h_monotonous : IsMonotonous f)
    (h_unique_zero : HasUniqueZero (fun x ↦ f (x^2) + f (k - x))) :
    k = 1/4 := by
  sorry

end odd_monotonous_unique_zero_implies_k_is_quarter_l2226_222653


namespace three_digit_numbers_with_one_or_six_l2226_222636

/-- The number of three-digit whole numbers -/
def total_three_digit_numbers : ℕ := 900

/-- The number of choices for the first digit (excluding 1 and 6) -/
def first_digit_choices : ℕ := 7

/-- The number of choices for the second and third digits (excluding 1 and 6) -/
def other_digit_choices : ℕ := 8

/-- The number of three-digit numbers without 1 or 6 -/
def numbers_without_one_or_six : ℕ := first_digit_choices * other_digit_choices * other_digit_choices

theorem three_digit_numbers_with_one_or_six : 
  total_three_digit_numbers - numbers_without_one_or_six = 452 := by
  sorry

end three_digit_numbers_with_one_or_six_l2226_222636


namespace kates_retirement_fund_l2226_222694

/-- 
Given an initial retirement fund value and a decrease amount, 
calculate the current value of the retirement fund.
-/
def current_fund_value (initial_value decrease : ℕ) : ℕ :=
  initial_value - decrease

/-- 
Theorem: Given Kate's initial retirement fund value of $1472 and a decrease of $12, 
the current value of her retirement fund is $1460.
-/
theorem kates_retirement_fund : 
  current_fund_value 1472 12 = 1460 := by
  sorry

end kates_retirement_fund_l2226_222694


namespace cos_sum_specific_values_l2226_222687

theorem cos_sum_specific_values (α β : ℝ) :
  Complex.exp (α * Complex.I) = (8 : ℝ) / 17 + (15 : ℝ) / 17 * Complex.I →
  Complex.exp (β * Complex.I) = -(5 : ℝ) / 13 + (12 : ℝ) / 13 * Complex.I →
  Real.cos (α + β) = -(220 : ℝ) / 221 := by
  sorry

end cos_sum_specific_values_l2226_222687


namespace hyperbola_circle_max_radius_l2226_222669

/-- Given a hyperbola and a circle with specific properties, prove that the maximum radius of the circle is √3 -/
theorem hyperbola_circle_max_radius (a b r : ℝ) (e : ℝ) :
  a > 0 →
  b > 0 →
  r > 0 →
  e ≤ 2 →
  (∃ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ x y : ℝ, (x - 2)^2 + y^2 = r^2) →
  (∃ x y : ℝ, b * x + a * y = 0 ∨ b * x - a * y = 0) →
  (∀ x y : ℝ, (b * x + a * y = 0 ∨ b * x - a * y = 0) → 
    ((x - 2)^2 + y^2 = r^2 → (x - 2)^2 + y^2 ≥ r^2)) →
  r ≤ Real.sqrt 3 :=
sorry

end hyperbola_circle_max_radius_l2226_222669
