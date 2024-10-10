import Mathlib

namespace distance_time_relationship_l1080_108052

/-- The relationship between distance and time for a car traveling at 60 km/h -/
theorem distance_time_relationship (s t : ℝ) (h : s = 60 * t) :
  s = 60 * t :=
by sorry

end distance_time_relationship_l1080_108052


namespace translated_line_equation_l1080_108041

/-- Given a line with slope 2 passing through the point (5, 1), prove that its equation is y = 2x - 9 -/
theorem translated_line_equation (x y : ℝ) :
  (y = 2 * x + 3) →  -- Original line equation
  (∃ b, y = 2 * x + b) →  -- Translated line has the same slope but different y-intercept
  (1 = 2 * 5 + b) →  -- The translated line passes through (5, 1)
  (y = 2 * x - 9)  -- The equation of the translated line
  := by sorry

end translated_line_equation_l1080_108041


namespace square_area_from_diagonal_l1080_108075

/-- The area of a square field with a given diagonal length -/
theorem square_area_from_diagonal (d : ℝ) (h : d = 98.00000000000001) :
  d^2 / 2 = 4802.000000000001 := by
  sorry

end square_area_from_diagonal_l1080_108075


namespace third_person_weight_l1080_108013

/-- The weight of the third person entering an elevator given specific average weight changes --/
theorem third_person_weight (initial_people : ℕ) (initial_avg : ℝ) 
  (avg_after_first : ℝ) (avg_after_second : ℝ) (avg_after_third : ℝ) :
  initial_people = 6 →
  initial_avg = 156 →
  avg_after_first = 159 →
  avg_after_second = 162 →
  avg_after_third = 161 →
  ∃ (w1 w2 w3 : ℝ),
    w1 = (initial_people + 1) * avg_after_first - initial_people * initial_avg ∧
    w2 = (initial_people + 2) * avg_after_second - (initial_people + 1) * avg_after_first ∧
    w3 = (initial_people + 3) * avg_after_third - (initial_people + 2) * avg_after_second ∧
    w3 = 163 :=
by sorry

end third_person_weight_l1080_108013


namespace largest_four_digit_number_l1080_108032

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Calculates the result of the operation (AAA + BA) * C -/
def calculate (A B C : Digit) : ℕ :=
  (111 * A.val + 10 * B.val + A.val) * C.val

/-- Checks if three digits are all different -/
def allDifferent (A B C : Digit) : Prop :=
  A ≠ B ∧ B ≠ C ∧ A ≠ C

theorem largest_four_digit_number :
  ∃ (A B C : Digit), allDifferent A B C ∧ 
    calculate A B C = 8624 ∧
    (∀ (X Y Z : Digit), allDifferent X Y Z → calculate X Y Z ≤ 8624) :=
sorry

end largest_four_digit_number_l1080_108032


namespace move_right_coords_specific_point_move_l1080_108043

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Move a point horizontally -/
def moveRight (p : Point) (h : ℝ) : Point :=
  { x := p.x + h, y := p.y }

theorem move_right_coords (p : Point) (h : ℝ) :
  moveRight p h = { x := p.x + h, y := p.y } := by sorry

theorem specific_point_move :
  let p : Point := { x := -1, y := 3 }
  moveRight p 2 = { x := 1, y := 3 } := by sorry

end move_right_coords_specific_point_move_l1080_108043


namespace german_students_l1080_108066

theorem german_students (total : ℕ) (both : ℕ) (german : ℕ) (spanish : ℕ) :
  total = 30 ∧ 
  both = 2 ∧ 
  german + spanish + both = total ∧ 
  german = 3 * spanish →
  german - both = 20 := by
  sorry

end german_students_l1080_108066


namespace abs_diff_eq_diff_implies_le_l1080_108096

theorem abs_diff_eq_diff_implies_le (x y : ℝ) :
  |x - y| = y - x → x ≤ y := by
  sorry

end abs_diff_eq_diff_implies_le_l1080_108096


namespace burgerCaloriesTheorem_l1080_108078

/-- Calculates the total calories consumed over a number of days, given the number of burgers eaten per day and calories per burger. -/
def totalCalories (burgersPerDay : ℕ) (caloriesPerBurger : ℕ) (days : ℕ) : ℕ :=
  burgersPerDay * caloriesPerBurger * days

/-- Theorem stating that eating 3 burgers per day, with 20 calories per burger, results in 120 calories consumed after two days. -/
theorem burgerCaloriesTheorem : totalCalories 3 20 2 = 120 := by
  sorry


end burgerCaloriesTheorem_l1080_108078


namespace pentagon_largest_angle_l1080_108069

/-- The sum of interior angles of a pentagon in degrees -/
def pentagon_angle_sum : ℕ := 540

/-- Represents the five consecutive integer angles of a pentagon -/
structure PentagonAngles where
  middle : ℕ
  valid : middle - 2 > 0 -- Ensures all angles are positive

/-- The sum of the five consecutive integer angles -/
def angle_sum (p : PentagonAngles) : ℕ :=
  (p.middle - 2) + (p.middle - 1) + p.middle + (p.middle + 1) + (p.middle + 2)

/-- The largest angle in the pentagon -/
def largest_angle (p : PentagonAngles) : ℕ := p.middle + 2

theorem pentagon_largest_angle :
  ∃ p : PentagonAngles, angle_sum p = pentagon_angle_sum ∧ largest_angle p = 110 := by
  sorry

end pentagon_largest_angle_l1080_108069


namespace sqrt_17_irrational_l1080_108093

theorem sqrt_17_irrational : ¬ ∃ (p q : ℤ), q ≠ 0 ∧ (p : ℚ) / q = Real.sqrt 17 := by
  sorry

end sqrt_17_irrational_l1080_108093


namespace B_power_15_minus_3_power_14_l1080_108098

def B : Matrix (Fin 2) (Fin 2) ℝ := !![4, 1; 0, 2]

theorem B_power_15_minus_3_power_14 :
  B^15 - 3 • B^14 = !![4, 3; 0, -2] := by sorry

end B_power_15_minus_3_power_14_l1080_108098


namespace cylinder_dimensions_l1080_108008

/-- Proves that a cylinder with equal surface area to a sphere of radius 4 cm
    and with equal height and diameter has height and diameter of 8 cm. -/
theorem cylinder_dimensions (r : ℝ) (h : ℝ) :
  r = 4 →  -- radius of the sphere is 4 cm
  (4 * π * r^2 : ℝ) = 2 * π * r * h →  -- surface areas are equal
  h = 2 * r →  -- height equals diameter
  h = 8 ∧ (2 * r) = 8 :=  -- height and diameter are both 8 cm
by sorry

end cylinder_dimensions_l1080_108008


namespace jill_second_bus_ride_time_l1080_108009

/-- The time Jill spends waiting for her first bus, in minutes -/
def first_bus_wait : ℕ := 12

/-- The time Jill spends riding on her first bus, in minutes -/
def first_bus_ride : ℕ := 30

/-- The total time Jill spends on her first bus (waiting and riding), in minutes -/
def first_bus_total : ℕ := first_bus_wait + first_bus_ride

/-- The time Jill spends on her second bus ride, in minutes -/
def second_bus_ride : ℕ := first_bus_total / 2

theorem jill_second_bus_ride_time :
  second_bus_ride = 21 := by sorry

end jill_second_bus_ride_time_l1080_108009


namespace intersection_integer_point_l1080_108090

/-- A point with integer coordinates -/
structure IntegerPoint where
  x : ℤ
  y : ℤ

/-- The intersection point of two lines -/
def intersection (m : ℤ) : ℚ × ℚ :=
  let x := (4 + 2*m) / (1 - m)
  let y := x - 4
  (x, y)

/-- Predicate to check if a point has integer coordinates -/
def isIntegerPoint (p : ℚ × ℚ) : Prop :=
  ∃ (ip : IntegerPoint), (ip.x : ℚ) = p.1 ∧ (ip.y : ℚ) = p.2

theorem intersection_integer_point :
  ∃ (m : ℤ), isIntegerPoint (intersection m) ∧ m = 8 :=
sorry

end intersection_integer_point_l1080_108090


namespace honey_harvest_increase_l1080_108018

def last_year_harvest : ℕ := 2479
def this_year_harvest : ℕ := 8564

theorem honey_harvest_increase :
  this_year_harvest - last_year_harvest = 6085 :=
by sorry

end honey_harvest_increase_l1080_108018


namespace max_stamps_proof_l1080_108087

/-- The price of a stamp in cents -/
def stamp_price : ℕ := 37

/-- The amount of money available in cents -/
def available_money : ℕ := 4000

/-- The maximum number of stamps that can be purchased -/
def max_stamps : ℕ := 108

theorem max_stamps_proof :
  (stamp_price * max_stamps ≤ available_money) ∧
  (∀ n : ℕ, stamp_price * n ≤ available_money → n ≤ max_stamps) := by
  sorry

end max_stamps_proof_l1080_108087


namespace largest_interior_angle_of_triangle_l1080_108051

theorem largest_interior_angle_of_triangle (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a + b + c = 360 →
  a / 5 = b / 4 →
  a / 5 = c / 3 →
  max (180 - a) (max (180 - b) (180 - c)) = 90 :=
by sorry

end largest_interior_angle_of_triangle_l1080_108051


namespace derivative_f_at_one_l1080_108095

-- Define the function f(x) = 2x^2
def f (x : ℝ) : ℝ := 2 * x^2

-- State the theorem
theorem derivative_f_at_one :
  deriv f 1 = 4 := by sorry

end derivative_f_at_one_l1080_108095


namespace power_mod_seventeen_l1080_108006

theorem power_mod_seventeen : 7^2048 % 17 = 1 := by
  sorry

end power_mod_seventeen_l1080_108006


namespace table_area_proof_l1080_108040

theorem table_area_proof (total_runner_area : ℝ) 
                         (coverage_percentage : ℝ)
                         (two_layer_area : ℝ)
                         (three_layer_area : ℝ) 
                         (h1 : total_runner_area = 220)
                         (h2 : coverage_percentage = 0.80)
                         (h3 : two_layer_area = 24)
                         (h4 : three_layer_area = 28) :
  ∃ (table_area : ℝ), table_area = 275 ∧ 
    coverage_percentage * table_area = total_runner_area := by
  sorry


end table_area_proof_l1080_108040


namespace buses_meet_time_l1080_108034

/-- Represents a time of day in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Represents a bus journey -/
structure BusJourney where
  startTime : Time
  endTime : Time
  distance : Nat
  deriving Repr

/-- The problem setup -/
def busProbe : Prop :=
  let totalDistance : Nat := 189
  let lishanToCounty : Nat := 54
  let busAToCounty : BusJourney := { startTime := { hours := 8, minutes := 30 },
                                     endTime := { hours := 9, minutes := 15 },
                                     distance := lishanToCounty }
  let busAToProvincial : BusJourney := { startTime := { hours := 9, minutes := 30 },
                                         endTime := { hours := 11, minutes := 0 },
                                         distance := totalDistance - lishanToCounty }
  let busBSpeed : Nat := 60
  let busBStartTime : Time := { hours := 8, minutes := 50 }
  
  ∃ (meetingTime : Time),
    meetingTime.hours = 10 ∧ meetingTime.minutes = 8

theorem buses_meet_time : busProbe := by
  sorry

end buses_meet_time_l1080_108034


namespace concatenated_evens_not_divisible_by_24_l1080_108031

def concatenated_evens : ℕ := 121416182022242628303234

theorem concatenated_evens_not_divisible_by_24 : ¬ (concatenated_evens % 24 = 0) := by
  sorry

end concatenated_evens_not_divisible_by_24_l1080_108031


namespace max_prob_second_game_C_l1080_108081

variable (p₁ p₂ p₃ : ℝ)

-- Define the probabilities of winning against each player
def prob_A := p₁
def prob_B := p₂
def prob_C := p₃

-- Define the conditions
axiom prob_order : 0 < p₁ ∧ p₁ < p₂ ∧ p₂ < p₃

-- Define the probability of winning two consecutive games for each scenario
def P_A := 2 * (p₁ * (p₂ + p₃) - 2 * p₁ * p₂ * p₃)
def P_B := 2 * (p₂ * (p₁ + p₃) - 2 * p₁ * p₂ * p₃)
def P_C := 2 * (p₁ * p₃ + p₂ * p₃ - 2 * p₁ * p₂ * p₃)

-- Theorem statement
theorem max_prob_second_game_C :
  P_C > P_A ∧ P_C > P_B :=
sorry

end max_prob_second_game_C_l1080_108081


namespace student_handshake_problem_l1080_108029

theorem student_handshake_problem (m n : ℕ) (hm : m ≥ 3) (hn : n ≥ 3) :
  let total_handshakes := (12 + 10 * (m + n - 4) + 8 * (m - 2) * (n - 2)) / 2
  total_handshakes = 1020 → m * n = 280 := by
  sorry

end student_handshake_problem_l1080_108029


namespace salary_increase_with_manager_l1080_108094

/-- Calculates the increase in average salary when a manager's salary is added to a group of employees. -/
theorem salary_increase_with_manager 
  (num_employees : ℕ) 
  (avg_salary : ℚ) 
  (manager_salary : ℚ) : 
  num_employees = 20 →
  avg_salary = 1500 →
  manager_salary = 22500 →
  (((num_employees : ℚ) * avg_salary + manager_salary) / ((num_employees : ℚ) + 1)) - avg_salary = 1000 := by
  sorry

end salary_increase_with_manager_l1080_108094


namespace two_faces_same_sides_l1080_108092

/-- A face of a polyhedron -/
structure Face where
  sides : ℕ
  sides_ge_3 : sides ≥ 3

/-- A convex polyhedron -/
structure ConvexPolyhedron where
  faces : Finset Face
  nonempty : faces.Nonempty

theorem two_faces_same_sides (P : ConvexPolyhedron) : 
  ∃ f₁ f₂ : Face, f₁ ∈ P.faces ∧ f₂ ∈ P.faces ∧ f₁ ≠ f₂ ∧ f₁.sides = f₂.sides :=
sorry

end two_faces_same_sides_l1080_108092


namespace expansion_sum_zero_l1080_108012

theorem expansion_sum_zero (n k : ℕ) (a b : ℝ) (h1 : n ≥ 2) (h2 : a * b ≠ 0) (h3 : a = k^2 * b) (h4 : k > 0) :
  (n * (a - b)^(n-1) * (-b) + n * (n-1) / 2 * (a - b)^(n-2) * (-b)^2 = 0) →
  n = 2 * k + 1 :=
by sorry

end expansion_sum_zero_l1080_108012


namespace inscribed_cube_surface_area_l1080_108059

/-- The surface area of a cube inscribed in a sphere of radius 2 -/
theorem inscribed_cube_surface_area (r : ℝ) (a : ℝ) : 
  r = 2 →  -- The radius of the sphere is 2
  3 * a^2 = (2*r)^2 →  -- The cube's diagonal equals the sphere's diameter
  6 * a^2 = 32 :=  -- The surface area of the cube is 32
by
  sorry

end inscribed_cube_surface_area_l1080_108059


namespace average_temperature_of_three_cities_l1080_108060

/-- Proves that the average temperature of three cities is 95 degrees given specific temperature relationships --/
theorem average_temperature_of_three_cities
  (temp_new_york : ℝ)
  (h1 : temp_new_york = 80)
  (temp_miami : ℝ)
  (h2 : temp_miami = temp_new_york + 10)
  (temp_san_diego : ℝ)
  (h3 : temp_san_diego = temp_miami + 25) :
  (temp_new_york + temp_miami + temp_san_diego) / 3 = 95 := by
  sorry

end average_temperature_of_three_cities_l1080_108060


namespace constant_t_equation_l1080_108079

theorem constant_t_equation (t : ℝ) : 
  (∀ x : ℝ, (3*x^2 - 4*x + 5)*(2*x^2 + t*x + 8) = 6*x^4 - 26*x^3 + 58*x^2 - 76*x + 40) ↔ 
  t = -6 := by
sorry

end constant_t_equation_l1080_108079


namespace smallest_four_digit_mod_8_l1080_108025

theorem smallest_four_digit_mod_8 :
  ∀ n : ℕ, n ≥ 1000 ∧ n ≡ 5 [MOD 8] → n ≥ 1005 :=
by
  sorry

end smallest_four_digit_mod_8_l1080_108025


namespace unique_zip_code_l1080_108050

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_palindrome (a b c : ℕ) : Prop := a = c

def is_consecutive (a b : ℕ) : Prop := b = a + 1

theorem unique_zip_code (a b c d e : ℕ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e ∧
  a + b + c + d + e = 20 ∧
  is_consecutive a b ∧
  c ≠ 0 ∧ c ≠ a ∧ c ≠ b ∧
  is_palindrome a b c ∧
  d = 2 * a ∧
  d + e = 13 ∧
  is_prime (a * 10000 + b * 1000 + c * 100 + d * 10 + e) →
  a * 10000 + b * 1000 + c * 100 + d * 10 + e = 34367 :=
by sorry

end unique_zip_code_l1080_108050


namespace three_digit_integers_with_remainders_l1080_108061

theorem three_digit_integers_with_remainders : 
  ∃! (S : Finset ℕ), 
    (∀ n ∈ S, 100 ≤ n ∧ n < 1000 ∧ 
              n % 7 = 3 ∧ 
              n % 10 = 6 ∧ 
              n % 12 = 9) ∧
    S.card = 2 := by
  sorry

end three_digit_integers_with_remainders_l1080_108061


namespace yang_final_floor_l1080_108017

/-- The number of floors in the building -/
def total_floors : ℕ := 36

/-- The floor Xiao Wu reaches in the initial observation -/
def wu_initial : ℕ := 6

/-- The floor Xiao Yang reaches in the initial observation -/
def yang_initial : ℕ := 5

/-- The starting floor for both climbers -/
def start_floor : ℕ := 1

/-- The floor Xiao Yang reaches when Xiao Wu reaches the top floor -/
def yang_final : ℕ := 29

theorem yang_final_floor :
  (wu_initial - start_floor) / (yang_initial - start_floor) =
  (total_floors - start_floor) / (yang_final - start_floor) :=
sorry

end yang_final_floor_l1080_108017


namespace nabla_calculation_l1080_108080

def nabla (a b : ℚ) : ℚ := (a + b) / (1 + a * b)

theorem nabla_calculation : nabla (nabla 4 3) 2 = 11 / 9 := by
  sorry

end nabla_calculation_l1080_108080


namespace tree_planting_around_lake_l1080_108016

theorem tree_planting_around_lake (circumference : ℕ) (willow_interval : ℕ) : 
  circumference = 1200 → willow_interval = 10 → 
  (circumference / willow_interval + circumference / willow_interval = 240) := by
  sorry

end tree_planting_around_lake_l1080_108016


namespace soccer_balls_in_bag_l1080_108065

theorem soccer_balls_in_bag (initial_balls : ℕ) (additional_balls : ℕ) : 
  initial_balls = 6 → additional_balls = 18 → initial_balls + additional_balls = 24 :=
by sorry

end soccer_balls_in_bag_l1080_108065


namespace crane_among_chickens_is_random_l1080_108015

-- Define the type for events
inductive Event
| CoveringSky
| FumingOrifices
| StridingMeteor
| CraneAmongChickens

-- Define the property of being a random event
def isRandomEvent (e : Event) : Prop :=
  ∃ (outcome : Bool), (outcome = true ∨ outcome = false)

-- State the theorem
theorem crane_among_chickens_is_random :
  isRandomEvent Event.CraneAmongChickens :=
sorry

end crane_among_chickens_is_random_l1080_108015


namespace rectangle_properties_l1080_108028

/-- Rectangle with adjacent sides x and 4, and perimeter y -/
structure Rectangle where
  x : ℝ
  y : ℝ

/-- The perimeter of the rectangle is related to x -/
axiom perimeter_relation (rect : Rectangle) : rect.y = 2 * rect.x + 8

theorem rectangle_properties :
  ∀ (rect : Rectangle),
  (rect.x = 10 → rect.y = 28) ∧
  (rect.y = 30 → rect.x = 11) := by
  sorry

end rectangle_properties_l1080_108028


namespace m_neg_one_necessary_not_sufficient_l1080_108000

-- Define the lines l₁ and l₂
def l₁ (m : ℝ) (x y : ℝ) : Prop := m * x + (2 * m - 1) * y + 1 = 0
def l₂ (m : ℝ) (x y : ℝ) : Prop := 3 * x + m * y + 3 = 0

-- Define perpendicularity of two lines
def perpendicular (m : ℝ) : Prop := ∃ (k₁ k₂ : ℝ), k₁ * k₂ = -1 ∧
  (∀ (x y : ℝ), l₁ m x y → m * x + (2 * m - 1) * y = k₁) ∧
  (∀ (x y : ℝ), l₂ m x y → 3 * x + m * y = k₂)

-- State the theorem
theorem m_neg_one_necessary_not_sufficient :
  (∀ m : ℝ, m = -1 → perpendicular m) ∧
  ¬(∀ m : ℝ, perpendicular m → m = -1) :=
sorry

end m_neg_one_necessary_not_sufficient_l1080_108000


namespace probability_divisible_by_four_probability_calculation_l1080_108072

def fair_12_sided_die := Finset.range 12

theorem probability_divisible_by_four (a b : ℕ) : 
  a ∈ fair_12_sided_die → b ∈ fair_12_sided_die →
  (a % 4 = 0 ∧ b % 4 = 0) ↔ (10 * a + b) % 4 = 0 ∧ a % 4 = 0 ∧ b % 4 = 0 :=
by sorry

theorem probability_calculation :
  (Finset.filter (λ x : ℕ × ℕ => x.1 % 4 = 0 ∧ x.2 % 4 = 0) (fair_12_sided_die.product fair_12_sided_die)).card /
  (fair_12_sided_die.card * fair_12_sided_die.card : ℚ) = 1 / 16 :=
by sorry

end probability_divisible_by_four_probability_calculation_l1080_108072


namespace observed_price_in_local_currency_l1080_108084

-- Define constants for the given conditions
def producer_cost : ℝ := 19
def shipping_cost : ℝ := 5
def tax_rate : ℝ := 0.10
def commission_rate : ℝ := 0.20
def exchange_rate : ℝ := 0.90
def profit_rate : ℝ := 0.20

-- Define the theorem
theorem observed_price_in_local_currency :
  let base_cost := producer_cost + shipping_cost
  let total_cost := base_cost + tax_rate * base_cost
  let profit := profit_rate * total_cost
  let price_before_commission := total_cost + profit
  let distributor_price := price_before_commission / (1 - commission_rate)
  let local_price := distributor_price * exchange_rate
  local_price = 35.64 := by sorry

end observed_price_in_local_currency_l1080_108084


namespace quadratic_equation_solution_fractional_equation_solution_l1080_108001

-- Equation 1
theorem quadratic_equation_solution (x : ℝ) :
  2 * x^2 + 3 * x = 1 ↔ x = (-3 + Real.sqrt 17) / 4 ∨ x = (-3 - Real.sqrt 17) / 4 :=
sorry

-- Equation 2
theorem fractional_equation_solution (x : ℝ) (h : x ≠ 2) :
  3 / (x - 2) = 5 / (2 - x) - 1 ↔ x = -6 :=
sorry

end quadratic_equation_solution_fractional_equation_solution_l1080_108001


namespace max_square_area_with_perimeter_34_l1080_108091

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def rectangle_area (l w : ℕ) : ℕ := l * w

def rectangle_perimeter (l w : ℕ) : ℕ := 2 * (l + w)

theorem max_square_area_with_perimeter_34 :
  ∀ l w : ℕ,
    rectangle_perimeter l w = 34 →
    is_perfect_square (rectangle_area l w) →
    rectangle_area l w ≤ 16 :=
by sorry

end max_square_area_with_perimeter_34_l1080_108091


namespace tree_height_increase_l1080_108053

/-- Proves that the annual increase in tree height is 2 feet given the initial conditions --/
theorem tree_height_increase (initial_height : ℝ) (annual_increase : ℝ) : 
  initial_height = 4 →
  initial_height + 6 * annual_increase = (initial_height + 4 * annual_increase) * (4/3) →
  annual_increase = 2 := by
sorry

end tree_height_increase_l1080_108053


namespace sum_of_logarithmic_equation_l1080_108035

theorem sum_of_logarithmic_equation : 
  ∃ (k m n : ℕ+), 
    (Nat.gcd k.val (Nat.gcd m.val n.val) = 1) ∧ 
    (k.val * Real.log 5 / Real.log 400 + m.val * Real.log 2 / Real.log 400 = n.val) ∧
    (k.val + m.val + n.val = 7) := by
  sorry

end sum_of_logarithmic_equation_l1080_108035


namespace polynomial_value_theorem_l1080_108021

/-- Given a polynomial ax³ + bx - 3 where a and b are constants,
    if the value of the polynomial is 15 when x = 2,
    then the value of the polynomial is -21 when x = -2. -/
theorem polynomial_value_theorem (a b : ℝ) : 
  (8 * a + 2 * b - 3 = 15) → (-8 * a - 2 * b - 3 = -21) := by
  sorry

end polynomial_value_theorem_l1080_108021


namespace odd_decreasing_sum_negative_l1080_108039

/-- A function f is odd if f(-x) = -f(x) for all x -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function f is decreasing on [0, ∞) if f(x) ≥ f(y) whenever 0 ≤ x < y -/
def DecreasingOnNonnegative (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → x < y → f x ≥ f y

theorem odd_decreasing_sum_negative
  (f : ℝ → ℝ)
  (hodd : OddFunction f)
  (hdec : DecreasingOnNonnegative f)
  (a b : ℝ)
  (hsum : a + b > 0) :
  f a + f b < 0 :=
sorry

end odd_decreasing_sum_negative_l1080_108039


namespace rectangle_area_l1080_108071

-- Define the rectangle ABCD
structure Rectangle :=
  (A B C D : ℝ × ℝ)

-- Define the properties of the rectangle
def isRectangle (rect : Rectangle) : Prop :=
  -- Add properties that define a rectangle
  sorry

-- Define the length of a side
def sideLength (p1 p2 : ℝ × ℝ) : ℝ :=
  sorry

-- Define the area of a rectangle
def area (rect : Rectangle) : ℝ :=
  sorry

-- Theorem statement
theorem rectangle_area (rect : Rectangle) :
  isRectangle rect →
  sideLength rect.A rect.B = 15 →
  sideLength rect.A rect.C = 17 →
  area rect = 120 := by
  sorry

end rectangle_area_l1080_108071


namespace symmetric_point_simplification_l1080_108044

theorem symmetric_point_simplification (x : ℝ) :
  (∃ P : ℝ × ℝ, P = (x + 1, 2 * x - 1) ∧ 
   (∃ P' : ℝ × ℝ, P' = (-x - 1, -2 * x + 1) ∧ 
    P'.1 > 0 ∧ P'.2 > 0)) →
  |x - 3| - |1 - x| = 2 := by
sorry

end symmetric_point_simplification_l1080_108044


namespace no_equivalent_expressions_l1080_108058

theorem no_equivalent_expressions (x : ℝ) (h : x > 0) : 
  (∀ y : ℝ, y > 0 → 2*(y+1)^(y+1) ≠ 2*(y+1)^y) ∧
  (∀ y : ℝ, y > 0 → 2*(y+1)^(y+1) ≠ (y+1)^(2*y+2)) ∧
  (∀ y : ℝ, y > 0 → 2*(y+1)^(y+1) ≠ 2*(y+0.5*y)^y) ∧
  (∀ y : ℝ, y > 0 → 2*(y+1)^(y+1) ≠ (2*y+2)^(2*y+2)) :=
by sorry

end no_equivalent_expressions_l1080_108058


namespace simple_interest_calculation_l1080_108020

/-- Simple interest calculation -/
theorem simple_interest_calculation (interest : ℚ) (rate : ℚ) (time : ℚ) :
  interest = 4016.25 →
  rate = 1 / 100 →
  time = 3 →
  ∃ principal : ℚ, principal = 133875 ∧ interest = principal * rate * time :=
by sorry

end simple_interest_calculation_l1080_108020


namespace collinear_points_k_value_l1080_108027

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the basis vectors
variable (e₁ e₂ : V)

-- Define the points and vectors
variable (A B C D : V)
variable (k : ℝ)

-- State the theorem
theorem collinear_points_k_value
  (hAB : B - A = e₁ - e₂)
  (hBC : C - B = 3 • e₁ + 2 • e₂)
  (hCD : D - C = k • e₁ + 2 • e₂)
  (hCollinear : ∃ (t : ℝ), D - A = t • (C - A)) :
  k = 8 := by sorry

end collinear_points_k_value_l1080_108027


namespace cannot_cut_squares_l1080_108070

theorem cannot_cut_squares (paper_length paper_width : ℝ) 
  (square1_side square2_side : ℝ) (total_area : ℝ) : 
  paper_length = 10 →
  paper_width = 8 →
  square1_side / square2_side = 4 / 3 →
  square1_side^2 + square2_side^2 = total_area →
  total_area = 75 →
  square1_side + square2_side > paper_length :=
by sorry

end cannot_cut_squares_l1080_108070


namespace rabbit_escape_theorem_l1080_108063

/-- The number of additional jumps a rabbit can make before a dog catches it. -/
def rabbit_jumps_before_catch (head_start : ℕ) (dog_jumps : ℕ) (rabbit_jumps : ℕ)
  (dog_distance : ℕ) (rabbit_distance : ℕ) : ℕ :=
  14 * head_start

/-- Theorem stating the number of jumps a rabbit can make before being caught by a dog
    under specific conditions. -/
theorem rabbit_escape_theorem :
  rabbit_jumps_before_catch 50 5 6 7 9 = 700 := by
  sorry

#eval rabbit_jumps_before_catch 50 5 6 7 9

end rabbit_escape_theorem_l1080_108063


namespace total_card_cost_l1080_108097

def christmas_cards : ℕ := 20
def birthday_cards : ℕ := 15
def card_cost : ℕ := 2

theorem total_card_cost : christmas_cards * card_cost + birthday_cards * card_cost = 70 := by
  sorry

end total_card_cost_l1080_108097


namespace race_outcomes_l1080_108019

theorem race_outcomes (n : ℕ) (k : ℕ) (h1 : n = 6) (h2 : k = 4) :
  (n.factorial / (n - k).factorial) = 360 := by
  sorry

end race_outcomes_l1080_108019


namespace basket_average_price_l1080_108089

/-- Given 4 baskets with an average cost of $4 and a fifth basket costing $8,
    the average price of all 5 baskets is $4.80. -/
theorem basket_average_price (num_initial_baskets : ℕ) (initial_avg_cost : ℚ) (fifth_basket_cost : ℚ) :
  num_initial_baskets = 4 →
  initial_avg_cost = 4 →
  fifth_basket_cost = 8 →
  (num_initial_baskets * initial_avg_cost + fifth_basket_cost) / (num_initial_baskets + 1) = 4.8 := by
  sorry

end basket_average_price_l1080_108089


namespace two_zeros_implies_a_range_l1080_108088

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then a + 2^x else (1/2) * x + a

-- Theorem statement
theorem two_zeros_implies_a_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0 ∧
    ∀ x : ℝ, x ≠ x₁ ∧ x ≠ x₂ → f a x ≠ 0) →
  a ∈ Set.Icc (-2) (-1/2) :=
sorry

end two_zeros_implies_a_range_l1080_108088


namespace inequality_solution_l1080_108002

theorem inequality_solution (x : ℝ) :
  (x * (x + 1)) / ((x - 5)^2) ≥ 15 ↔ (x > Real.sqrt (151 - Real.sqrt 1801) / 2 ∧ x < 5) ∨
                                    (x > 5 ∧ x < Real.sqrt (151 + Real.sqrt 1801) / 2) :=
by sorry

end inequality_solution_l1080_108002


namespace events_mutually_exclusive_not_complementary_l1080_108026

-- Define the set of numbers
def S : Set Nat := {n | 1 ≤ n ∧ n ≤ 9}

-- Define a type for a selection of three numbers
structure Selection :=
  (a b c : Nat)
  (distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (inS : a ∈ S ∧ b ∈ S ∧ c ∈ S)

-- Define events
def allEven (s : Selection) : Prop := s.a % 2 = 0 ∧ s.b % 2 = 0 ∧ s.c % 2 = 0
def allOdd (s : Selection) : Prop := s.a % 2 = 1 ∧ s.b % 2 = 1 ∧ s.c % 2 = 1
def oneEvenTwoOdd (s : Selection) : Prop :=
  (s.a % 2 = 0 ∧ s.b % 2 = 1 ∧ s.c % 2 = 1) ∨
  (s.a % 2 = 1 ∧ s.b % 2 = 0 ∧ s.c % 2 = 1) ∨
  (s.a % 2 = 1 ∧ s.b % 2 = 1 ∧ s.c % 2 = 0)
def twoEvenOneOdd (s : Selection) : Prop :=
  (s.a % 2 = 0 ∧ s.b % 2 = 0 ∧ s.c % 2 = 1) ∨
  (s.a % 2 = 0 ∧ s.b % 2 = 1 ∧ s.c % 2 = 0) ∨
  (s.a % 2 = 1 ∧ s.b % 2 = 0 ∧ s.c % 2 = 0)

-- Define mutual exclusivity and complementarity
def mutuallyExclusive (e1 e2 : Selection → Prop) : Prop :=
  ∀ s : Selection, ¬(e1 s ∧ e2 s)

def complementary (e1 e2 : Selection → Prop) : Prop :=
  ∀ s : Selection, e1 s ∨ e2 s

-- Theorem statement
theorem events_mutually_exclusive_not_complementary :
  (mutuallyExclusive allEven allOdd ∧ ¬complementary allEven allOdd) ∧
  (mutuallyExclusive oneEvenTwoOdd twoEvenOneOdd ∧ ¬complementary oneEvenTwoOdd twoEvenOneOdd) :=
sorry

end events_mutually_exclusive_not_complementary_l1080_108026


namespace exactlyOneAndTwoBlackMutuallyExclusiveAndNonOpposite_l1080_108033

/-- A pocket containing balls of two colors -/
structure Pocket where
  red : ℕ
  black : ℕ

/-- An event that can occur when selecting balls from a pocket -/
inductive Event
  | ExactlyOneBlack
  | ExactlyTwoBlack

/-- The pocket we're considering in this problem -/
def problemPocket : Pocket := { red := 2, black := 2 }

/-- Two events are mutually exclusive if they cannot occur simultaneously -/
def mutuallyExclusive (e1 e2 : Event) : Prop :=
  ¬(e1 = Event.ExactlyOneBlack ∧ e2 = Event.ExactlyTwoBlack)

/-- Two events are non-opposite if it's possible for neither to occur -/
def nonOpposite (e1 e2 : Event) : Prop :=
  ∃ (outcome : Pocket → Bool), ¬outcome problemPocket

/-- The main theorem stating that ExactlyOneBlack and ExactlyTwoBlack are mutually exclusive and non-opposite -/
theorem exactlyOneAndTwoBlackMutuallyExclusiveAndNonOpposite :
  mutuallyExclusive Event.ExactlyOneBlack Event.ExactlyTwoBlack ∧
  nonOpposite Event.ExactlyOneBlack Event.ExactlyTwoBlack :=
sorry

end exactlyOneAndTwoBlackMutuallyExclusiveAndNonOpposite_l1080_108033


namespace base_n_representation_of_b_l1080_108038

/-- Represents a number in base n -/
def BaseNRepr (n : ℕ) (x : ℕ) : Prop :=
  ∃ (d₀ d₁ : ℕ), x = d₁ * n + d₀ ∧ d₁ < n ∧ d₀ < n

theorem base_n_representation_of_b
  (n : ℕ)
  (hn : n > 9)
  (a b : ℕ)
  (heq : n^2 - a*n + b = 0)
  (ha : BaseNRepr n a ∧ a = 19) :
  BaseNRepr n b ∧ b = 90 := by
  sorry

end base_n_representation_of_b_l1080_108038


namespace circles_intersect_l1080_108036

/-- Circle C₁ with equation x² + y² + 2x + 8y - 8 = 0 -/
def C₁ (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x + 8*y - 8 = 0

/-- Circle C₂ with equation x² + y² - 4x - 5 = 0 -/
def C₂ (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 5 = 0

/-- The circles C₁ and C₂ intersect -/
theorem circles_intersect : ∃ (x y : ℝ), C₁ x y ∧ C₂ x y := by
  sorry

end circles_intersect_l1080_108036


namespace power_product_equality_l1080_108042

theorem power_product_equality : (-0.25)^2022 * 4^2022 = 1 := by
  sorry

end power_product_equality_l1080_108042


namespace problem_solution_l1080_108083

theorem problem_solution (a b c d : ℕ+) 
  (h1 : a ^ 5 = b ^ 4)
  (h2 : c ^ 3 = d ^ 2)
  (h3 : c - a = 19) :
  d - b = 757 := by
  sorry

end problem_solution_l1080_108083


namespace line_intercepts_sum_l1080_108062

/-- Given a line with equation y - 3 = -3(x - 6), prove that the sum of its x-intercept and y-intercept is 28. -/
theorem line_intercepts_sum (x y : ℝ) :
  (y - 3 = -3 * (x - 6)) →
  (∃ x_int y_int : ℝ,
    (y_int - 3 = -3 * (x_int - 6) ∧ y_int = 0) ∧
    (0 - 3 = -3 * (0 - 6) ∧ y = y_int) ∧
    x_int + y_int = 28) :=
by sorry

end line_intercepts_sum_l1080_108062


namespace james_score_l1080_108022

/-- Quiz bowl scoring system at Highridge High -/
structure QuizBowl where
  pointsPerCorrect : ℕ := 2
  bonusPoints : ℕ := 4
  numRounds : ℕ := 5
  questionsPerRound : ℕ := 5

/-- Calculate the total points scored by a student in the quiz bowl -/
def calculatePoints (qb : QuizBowl) (missedQuestions : ℕ) : ℕ :=
  let totalQuestions := qb.numRounds * qb.questionsPerRound
  let correctAnswers := totalQuestions - missedQuestions
  let pointsFromCorrect := correctAnswers * qb.pointsPerCorrect
  let fullRounds := qb.numRounds - (if missedQuestions > 0 then 1 else 0)
  let bonusPointsTotal := fullRounds * qb.bonusPoints
  pointsFromCorrect + bonusPointsTotal

/-- Theorem: James scored 64 points in the quiz bowl -/
theorem james_score (qb : QuizBowl) : calculatePoints qb 1 = 64 := by
  sorry

end james_score_l1080_108022


namespace smallest_with_eight_factors_l1080_108068

/-- A function that returns the number of distinct positive factors of a positive integer -/
def number_of_factors (n : ℕ+) : ℕ := sorry

/-- A function that checks if a given number has exactly eight distinct positive factors -/
def has_eight_factors (n : ℕ+) : Prop := number_of_factors n = 8

/-- Theorem stating that 24 is the smallest positive integer with exactly eight distinct positive factors -/
theorem smallest_with_eight_factors :
  has_eight_factors 24 ∧ ∀ m : ℕ+, m < 24 → ¬(has_eight_factors m) :=
sorry

end smallest_with_eight_factors_l1080_108068


namespace not_always_divisible_l1080_108023

theorem not_always_divisible : ¬ ∀ n : ℕ, (5^n - 1) % (4^n - 1) = 0 := by
  sorry

end not_always_divisible_l1080_108023


namespace tanα_tanβ_value_l1080_108073

theorem tanα_tanβ_value (α β : ℝ) 
  (h1 : Real.cos (α + β) = 1/5)
  (h2 : Real.cos (α - β) = 3/5) :
  Real.tan α * Real.tan β = 1/2 := by
  sorry

end tanα_tanβ_value_l1080_108073


namespace ron_chocolate_cost_l1080_108037

/-- Calculates the cost of chocolate bars for a boy scout camp out -/
def chocolate_cost (chocolate_bar_price : ℚ) (sections_per_bar : ℕ) (num_scouts : ℕ) (smores_per_scout : ℕ) : ℚ :=
  let total_smores := num_scouts * smores_per_scout
  let bars_needed := (total_smores + sections_per_bar - 1) / sections_per_bar
  bars_needed * chocolate_bar_price

/-- Theorem: The cost of chocolate bars for Ron's boy scout camp out is $15.00 -/
theorem ron_chocolate_cost :
  chocolate_cost (3/2) 3 15 2 = 15 := by
  sorry

end ron_chocolate_cost_l1080_108037


namespace total_miles_walked_l1080_108077

/-- The number of ladies in the walking group -/
def num_ladies : ℕ := 5

/-- The number of miles walked together by the group each day -/
def group_miles_per_day : ℕ := 3

/-- The number of additional miles Jamie walks per day -/
def jamie_additional_miles_per_day : ℕ := 2

/-- The number of days they walk per week -/
def days_per_week : ℕ := 6

/-- The total miles walked by the ladies in 6 days -/
def total_miles : ℕ := num_ladies * group_miles_per_day * days_per_week + jamie_additional_miles_per_day * days_per_week

theorem total_miles_walked :
  total_miles = 120 := by sorry

end total_miles_walked_l1080_108077


namespace point_movement_to_y_axis_l1080_108049

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The y-axis -/
def yAxis : Set Point := {p : Point | p.x = 0}

theorem point_movement_to_y_axis (m : ℝ) :
  let P : Point := ⟨m + 2, 3⟩
  let P' : Point := ⟨P.x + 3, P.y⟩
  P' ∈ yAxis → m = -5 := by
  sorry

end point_movement_to_y_axis_l1080_108049


namespace logarithm_comparison_l1080_108056

theorem logarithm_comparison : 
  (Real.log 3.4 / Real.log 3 < Real.log 8.5 / Real.log 3) ∧ 
  ¬(π^(-0.7) < π^(-0.9)) ∧ 
  ¬(Real.log 1.8 / Real.log 0.3 < Real.log 2.7 / Real.log 0.3) ∧ 
  ¬(0.99^2.7 < 0.99^3.5) := by
  sorry

end logarithm_comparison_l1080_108056


namespace rationalize_denominator_l1080_108099

theorem rationalize_denominator :
  ∃ (A B C : ℤ),
    (2 + Real.sqrt 5) / (2 - Real.sqrt 5) = A + B * Real.sqrt C ∧
    A = -9 ∧
    B = -4 ∧
    C = 5 := by
  sorry

end rationalize_denominator_l1080_108099


namespace smallest_k_for_equalization_l1080_108024

/-- Represents the state of gas cylinders -/
def CylinderState := List ℝ

/-- Represents a connection operation on cylinders -/
def Connection := List Nat

/-- Applies a single connection operation to a cylinder state -/
def applyConnection (state : CylinderState) (conn : Connection) : CylinderState :=
  sorry

/-- Checks if all pressures in a state are equal -/
def isEqualized (state : CylinderState) : Prop :=
  sorry

/-- Checks if a connection is valid (size ≤ k) -/
def isValidConnection (conn : Connection) (k : ℕ) : Prop :=
  sorry

/-- Represents a sequence of connection operations -/
def EqualizationProcess := List Connection

/-- Checks if an equalization process is valid for a given k -/
def isValidProcess (process : EqualizationProcess) (k : ℕ) : Prop :=
  sorry

/-- Applies an equalization process to a cylinder state -/
def applyProcess (state : CylinderState) (process : EqualizationProcess) : CylinderState :=
  sorry

/-- Main theorem: 5 is the smallest k that allows equalization -/
theorem smallest_k_for_equalization :
  (∀ (initial : CylinderState), initial.length = 40 →
    ∃ (process : EqualizationProcess), 
      isValidProcess process 5 ∧ 
      isEqualized (applyProcess initial process)) ∧
  (∀ k < 5, ∃ (initial : CylinderState), initial.length = 40 ∧
    ∀ (process : EqualizationProcess), 
      isValidProcess process k → 
      ¬isEqualized (applyProcess initial process)) :=
  sorry

end smallest_k_for_equalization_l1080_108024


namespace income_comparison_l1080_108074

theorem income_comparison (juan tim mary : ℝ) 
  (h1 : tim = 0.4 * juan)
  (h2 : mary = 0.6400000000000001 * juan) :
  (mary - tim) / tim = 0.6 := by
sorry

end income_comparison_l1080_108074


namespace correct_converses_l1080_108082

-- Proposition 1
def prop1 (x : ℝ) : Prop := x^2 - 3*x + 2 = 0 → x = 1 ∨ x = 2

-- Proposition 2
def prop2 (x : ℝ) : Prop := -2 ≤ x ∧ x < 3 → (x + 2) * (x - 3) ≤ 0

-- Proposition 3
def prop3 (x y : ℝ) : Prop := x = 0 ∧ y = 0 → x^2 + y^2 = 0

-- Proposition 4
def prop4 (x y : ℕ) : Prop := x ≠ 0 ∧ y ≠ 0 ∧ Even x ∧ Even y → Even (x + y)

-- Converses of the propositions
def conv1 (x : ℝ) : Prop := x = 1 ∨ x = 2 → x^2 - 3*x + 2 = 0

def conv2 (x : ℝ) : Prop := (x + 2) * (x - 3) ≤ 0 → -2 ≤ x ∧ x < 3

def conv3 (x y : ℝ) : Prop := x^2 + y^2 = 0 → x = 0 ∧ y = 0

def conv4 (x y : ℕ) : Prop := x ≠ 0 ∧ y ≠ 0 ∧ Even (x + y) → Even x ∧ Even y

theorem correct_converses :
  (∀ x, conv1 x) ∧
  (∀ x y, conv3 x y) ∧
  ¬(∀ x, conv2 x) ∧
  ¬(∀ x y, conv4 x y) :=
by sorry

end correct_converses_l1080_108082


namespace caseys_water_ratio_l1080_108086

/-- Proves that the ratio of water needed by each duck to water needed by each pig is 1:16 given the conditions of Casey's water pumping scenario. -/
theorem caseys_water_ratio :
  let pump_rate : ℚ := 3  -- gallons per minute
  let pump_time : ℚ := 25  -- minutes
  let corn_rows : ℕ := 4
  let corn_plants_per_row : ℕ := 15
  let water_per_corn_plant : ℚ := 1/2  -- gallons
  let num_pigs : ℕ := 10
  let water_per_pig : ℚ := 4  -- gallons
  let num_ducks : ℕ := 20

  let total_water : ℚ := pump_rate * pump_time
  let corn_water : ℚ := (corn_rows * corn_plants_per_row : ℚ) * water_per_corn_plant
  let pig_water : ℚ := (num_pigs : ℚ) * water_per_pig
  let duck_water : ℚ := total_water - corn_water - pig_water
  let water_per_duck : ℚ := duck_water / num_ducks

  water_per_duck / water_per_pig = 1 / 16 :=
by
  sorry

end caseys_water_ratio_l1080_108086


namespace complex_equation_solution_l1080_108076

theorem complex_equation_solution :
  ∃ (x y : ℂ), (3 + 5*I)*x + (2 - I)*y = 17 - 2*I ∧ x = 1 ∧ y = 7 :=
by
  sorry

end complex_equation_solution_l1080_108076


namespace sum_of_angles_l1080_108085

-- Define the angles
variable (A B C D E F : ℝ)

-- Define the triangles and quadrilateral
def is_triangle (x y z : ℝ) : Prop := x + y + z = 180

-- Axioms based on the problem conditions
axiom triangle_ABC : is_triangle A B C
axiom triangle_DEF : is_triangle D E F
axiom quadrilateral_BEFC : B + E + F + C = 360

-- Theorem to prove
theorem sum_of_angles : A + B + C + D + E + F = 360 := by
  sorry

end sum_of_angles_l1080_108085


namespace closed_polygonal_line_links_divisible_by_four_l1080_108055

/-- Represents a link in the polygonal line -/
structure Link where
  direction : Bool  -- True for horizontal, False for vertical
  length : Nat
  is_odd : Odd length

/-- Represents a closed polygonal line on a square grid -/
structure PolygonalLine where
  links : List Link
  is_closed : links.length > 0

/-- The main theorem to prove -/
theorem closed_polygonal_line_links_divisible_by_four (p : PolygonalLine) :
  4 ∣ p.links.length :=
sorry

end closed_polygonal_line_links_divisible_by_four_l1080_108055


namespace lcm_of_4_6_9_l1080_108046

theorem lcm_of_4_6_9 : Nat.lcm (Nat.lcm 4 6) 9 = 36 := by
  sorry

end lcm_of_4_6_9_l1080_108046


namespace max_days_for_88_alligators_l1080_108054

/-- Represents the number of days in a week -/
def days_per_week : ℕ := 7

/-- Represents the eating rate of the snake in alligators per week -/
def eating_rate : ℕ := 1

/-- Represents the total number of alligators eaten -/
def total_alligators : ℕ := 88

/-- Calculates the maximum number of days to eat a given number of alligators -/
def max_days_to_eat (alligators : ℕ) (rate : ℕ) (days_in_week : ℕ) : ℕ :=
  alligators * days_in_week / rate

/-- Theorem stating that the maximum number of days to eat 88 alligators is 616 -/
theorem max_days_for_88_alligators :
  max_days_to_eat total_alligators eating_rate days_per_week = 616 := by
  sorry

end max_days_for_88_alligators_l1080_108054


namespace four_digit_sum_reverse_equals_4983_l1080_108048

def reverse_number (n : ℕ) : ℕ :=
  let d1 := n / 1000
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  let d4 := n % 10
  d4 * 1000 + d3 * 100 + d2 * 10 + d1

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem four_digit_sum_reverse_equals_4983 :
  ∃ (n : ℕ), is_four_digit n ∧ n + reverse_number n = 4983 :=
sorry

end four_digit_sum_reverse_equals_4983_l1080_108048


namespace marble_distribution_l1080_108004

def is_valid_combination (a b c d e : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e ∧
  a > 1 ∧ b > 1 ∧ c > 1 ∧ d > 1 ∧ e > 1 ∧
  a + e ≥ 11 ∧
  c + a < 11 ∧
  b + c ≥ 11 ∧
  c + d ≥ 11 ∧
  a + b + c + d + e = 26

theorem marble_distribution :
  ∀ a b c d e : ℕ,
  is_valid_combination a b c d e ↔
  ((a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 9 ∧ e = 11) ∨
   (a = 1 ∧ b = 2 ∧ c = 4 ∧ d = 9 ∧ e = 10) ∨
   (a = 1 ∧ b = 3 ∧ c = 4 ∧ d = 8 ∧ e = 10) ∨
   (a = 2 ∧ b = 3 ∧ c = 4 ∧ d = 8 ∧ e = 9)) :=
by sorry

end marble_distribution_l1080_108004


namespace polynomial_divisibility_by_five_l1080_108045

theorem polynomial_divisibility_by_five (a b c d : ℤ) :
  (∀ x : ℤ, (5 : ℤ) ∣ (a * x^3 + b * x^2 + c * x + d)) →
  (5 : ℤ) ∣ a ∧ (5 : ℤ) ∣ b ∧ (5 : ℤ) ∣ c ∧ (5 : ℤ) ∣ d := by
sorry


end polynomial_divisibility_by_five_l1080_108045


namespace enid_sweaters_count_l1080_108067

/-- The number of scarves Aaron makes -/
def aaron_scarves : ℕ := 10

/-- The number of sweaters Aaron makes -/
def aaron_sweaters : ℕ := 5

/-- The number of balls of wool used for one scarf -/
def wool_per_scarf : ℕ := 3

/-- The number of balls of wool used for one sweater -/
def wool_per_sweater : ℕ := 4

/-- The total number of balls of wool used by both Enid and Aaron -/
def total_wool : ℕ := 82

/-- The number of sweaters Enid makes -/
def enid_sweaters : ℕ := (total_wool - (aaron_scarves * wool_per_scarf + aaron_sweaters * wool_per_sweater)) / wool_per_sweater

theorem enid_sweaters_count :
  enid_sweaters = 8 := by sorry

end enid_sweaters_count_l1080_108067


namespace ice_cream_combinations_l1080_108010

/-- The number of different types of ice cream cones available. -/
def num_cone_types : ℕ := 2

/-- The number of different ice cream flavors available. -/
def num_flavors : ℕ := 4

/-- The total number of different ways to order ice cream. -/
def total_combinations : ℕ := num_cone_types * num_flavors

/-- Theorem stating that the total number of different ways to order ice cream is 8. -/
theorem ice_cream_combinations : total_combinations = 8 := by
  sorry

end ice_cream_combinations_l1080_108010


namespace condo_rented_units_l1080_108014

/-- Represents the number of units of each bedroom type in a condominium -/
structure CondoUnits where
  one_bedroom : ℕ
  two_bedroom : ℕ
  three_bedroom : ℕ

/-- Represents the number of rented units of each bedroom type in a condominium -/
structure RentedUnits where
  one_bedroom : ℕ
  two_bedroom : ℕ
  three_bedroom : ℕ

def total_units (c : CondoUnits) : ℕ :=
  c.one_bedroom + c.two_bedroom + c.three_bedroom

def total_rented (r : RentedUnits) : ℕ :=
  r.one_bedroom + r.two_bedroom + r.three_bedroom

theorem condo_rented_units 
  (c : CondoUnits)
  (r : RentedUnits)
  (h1 : total_units c = 1200)
  (h2 : total_rented r = 700)
  (h3 : r.one_bedroom * 3 = r.two_bedroom * 2)
  (h4 : r.one_bedroom * 2 = r.three_bedroom)
  (h5 : r.two_bedroom * 2 = c.two_bedroom)
  : c.two_bedroom - r.two_bedroom = 231 := by
  sorry

end condo_rented_units_l1080_108014


namespace log_sum_equals_two_l1080_108007

theorem log_sum_equals_two : Real.log 0.01 / Real.log 10 + Real.log 16 / Real.log 2 = 2 := by
  sorry

end log_sum_equals_two_l1080_108007


namespace sum_g_h_equals_negative_eight_l1080_108057

theorem sum_g_h_equals_negative_eight (g h : ℝ) :
  (∀ d : ℝ, (8*d^2 - 4*d + g) * (4*d^2 + h*d + 7) = 32*d^4 + (4*h-16)*d^3 - (14*d^2 - 28*d - 56)) →
  g + h = -8 := by sorry

end sum_g_h_equals_negative_eight_l1080_108057


namespace sum_of_digits_product_53_nines_53_fours_l1080_108064

def repeat_digit (d : ℕ) (n : ℕ) : ℕ :=
  d * (10^n - 1) / 9

def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_product_53_nines_53_fours :
  sum_of_digits (repeat_digit 9 53 * repeat_digit 4 53) = 477 := by
  sorry

end sum_of_digits_product_53_nines_53_fours_l1080_108064


namespace geometric_sequence_sum_l1080_108030

/-- A geometric sequence {a_n} -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The conditions of the problem -/
structure ProblemConditions (a : ℕ → ℝ) : Prop :=
  (geom_seq : geometric_sequence a)
  (sum_cond : a 4 + a 7 = 2)
  (prod_cond : a 2 * a 9 = -8)

/-- The theorem to prove -/
theorem geometric_sequence_sum 
  (a : ℕ → ℝ) 
  (h : ProblemConditions a) : 
  a 1 + a 10 = -7 := by
  sorry

end geometric_sequence_sum_l1080_108030


namespace A_intersect_B_l1080_108011

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {x | x > 2}

theorem A_intersect_B : A ∩ B = {3, 4} := by sorry

end A_intersect_B_l1080_108011


namespace sqrt_245_simplification_l1080_108005

theorem sqrt_245_simplification : Real.sqrt 245 = 7 * Real.sqrt 5 := by
  sorry

end sqrt_245_simplification_l1080_108005


namespace triangle_inequalities_l1080_108003

/-- Given four collinear points E, F, G, H in order, with EF = a, EG = b, EH = c,
    if EF and GH are rotated to form a triangle with positive area,
    then a < c/3 and b < a + c/3 must be true, while b < c/3 is not necessarily true. -/
theorem triangle_inequalities (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hab : a < b) (hbc : b < c) 
  (h_triangle : a + (c - b) > b - a) : 
  (a < c / 3 ∧ b < a + c / 3) ∧ ¬(b < c / 3 → True) := by
  sorry

end triangle_inequalities_l1080_108003


namespace stream_current_rate_l1080_108047

/-- Represents the man's usual rowing speed in still water -/
def r : ℝ := sorry

/-- Represents the speed of the stream's current -/
def w : ℝ := sorry

/-- The distance traveled downstream and upstream -/
def distance : ℝ := 24

/-- Theorem stating the conditions and the conclusion about the stream's current -/
theorem stream_current_rate :
  (distance / (r + w) + 6 = distance / (r - w)) ∧
  (distance / (3*r + w) + 2 = distance / (3*r - w)) →
  w = 2 := by sorry

end stream_current_rate_l1080_108047
