import Mathlib

namespace journey_equations_l977_97719

/-- Represents a journey between two points with an uphill and a flat section -/
structure Journey where
  uphill_length : ℝ  -- Length of uphill section in km
  flat_length : ℝ    -- Length of flat section in km
  uphill_speed : ℝ   -- Speed on uphill section in km/h
  flat_speed : ℝ     -- Speed on flat section in km/h
  downhill_speed : ℝ -- Speed on downhill section in km/h
  time_ab : ℝ        -- Time from A to B in minutes
  time_ba : ℝ        -- Time from B to A in minutes

/-- The correct system of equations for the journey -/
def correct_equations (j : Journey) : Prop :=
  (j.uphill_length / j.uphill_speed + j.flat_length / j.flat_speed = j.time_ab / 60) ∧
  (j.uphill_length / j.downhill_speed + j.flat_length / j.flat_speed = j.time_ba / 60)

/-- Theorem stating that the given journey satisfies the correct system of equations -/
theorem journey_equations (j : Journey) 
  (h1 : j.uphill_speed = 3)
  (h2 : j.flat_speed = 4)
  (h3 : j.downhill_speed = 5)
  (h4 : j.time_ab = 54)
  (h5 : j.time_ba = 42) :
  correct_equations j := by
  sorry

end journey_equations_l977_97719


namespace even_sin_function_phi_l977_97768

theorem even_sin_function_phi (f : ℝ → ℝ) (φ : ℝ) : 
  (∀ x, f x = Real.sin ((x + φ) / 3)) →
  (0 ≤ φ ∧ φ ≤ 2 * Real.pi) →
  (∀ x, f x = f (-x)) →
  φ = 3 * Real.pi / 2 := by
sorry

end even_sin_function_phi_l977_97768


namespace star_example_l977_97714

-- Define the star operation
def star (m n p q : ℚ) : ℚ := m * p * (n / q)

-- Theorem statement
theorem star_example : star (5/9) (10/6) = 75 := by
  sorry

end star_example_l977_97714


namespace power_equation_l977_97720

theorem power_equation (y : ℝ) : (12 : ℝ)^2 * 6^y / 432 = 72 → y = 3 := by
  sorry

end power_equation_l977_97720


namespace simplify_expression_l977_97705

theorem simplify_expression (x : ℝ) : (3*x)^5 - (4*x)*(x^4) = 239*(x^5) := by
  sorry

end simplify_expression_l977_97705


namespace gcd_minus_twelve_equals_thirtysix_l977_97709

theorem gcd_minus_twelve_equals_thirtysix :
  Nat.gcd 7344 48 - 12 = 36 := by
  sorry

end gcd_minus_twelve_equals_thirtysix_l977_97709


namespace triangle_area_l977_97776

-- Define the lines
def line1 (x : ℝ) : ℝ := x
def line2 (x : ℝ) : ℝ := -x
def line3 : ℝ := 8

-- Define the theorem
theorem triangle_area : 
  let A : ℝ × ℝ := (8, 8)
  let B : ℝ × ℝ := (-8, 8)
  let O : ℝ × ℝ := (0, 0)
  let base := |A.1 - B.1|
  let height := |O.2 - line3|
  (1 / 2 : ℝ) * base * height = 64 := by
  sorry

end triangle_area_l977_97776


namespace sum_due_proof_l977_97772

/-- Represents the relationship between banker's discount, true discount, and face value. -/
def bankers_discount_relation (bd td fv : ℚ) : Prop :=
  bd = td + (td * bd / fv)

/-- Proves that given a banker's discount of 80 and a true discount of 70,
    the face value (sum due) is 560. -/
theorem sum_due_proof :
  ∃ (fv : ℚ), bankers_discount_relation 80 70 fv ∧ fv = 560 :=
by sorry

end sum_due_proof_l977_97772


namespace triangle_area_l977_97750

/-- The area of a triangle with base 8 and height 4 is 16 -/
theorem triangle_area : 
  ∀ (base height area : ℝ), 
  base = 8 → 
  height = 4 → 
  area = (base * height) / 2 → 
  area = 16 := by
sorry

end triangle_area_l977_97750


namespace systematic_sampling_proof_l977_97731

/-- Represents a sequence of 5 integers -/
def Sequence := Fin 5 → ℕ

/-- Checks if a sequence is valid for systematic sampling -/
def isValidSample (s : Sequence) (totalBags : ℕ) (sampleSize : ℕ) : Prop :=
  ∃ (start : ℕ) (interval : ℕ),
    (∀ i : Fin 5, s i = start + i.val * interval) ∧
    (∀ i : Fin 5, 1 ≤ s i ∧ s i ≤ totalBags) ∧
    interval = totalBags / sampleSize

theorem systematic_sampling_proof :
  let s : Sequence := fun i => [7, 17, 27, 37, 47][i]
  let totalBags := 50
  let sampleSize := 5
  isValidSample s totalBags sampleSize :=
by sorry

end systematic_sampling_proof_l977_97731


namespace ham_and_cake_probability_l977_97789

/-- The probability of packing a ham sandwich and cake on the same day -/
def prob_ham_and_cake (total_days : ℕ) (ham_days : ℕ) (cake_days : ℕ) : ℚ :=
  (ham_days : ℚ) / total_days * (cake_days : ℚ) / total_days

theorem ham_and_cake_probability :
  let total_days : ℕ := 5
  let ham_days : ℕ := 3
  let cake_days : ℕ := 1
  prob_ham_and_cake total_days ham_days cake_days = 12 / 100 := by
sorry

end ham_and_cake_probability_l977_97789


namespace fruit_cost_theorem_l977_97783

def calculate_fruit_cost (quantity : ℝ) (price : ℝ) (discount : ℝ) (tax : ℝ) : ℝ :=
  let cost_before_discount := quantity * price
  let discounted_cost := cost_before_discount * (1 - discount)
  let tax_amount := discounted_cost * tax
  discounted_cost + tax_amount

def grapes_cost := calculate_fruit_cost 8 70 0.1 0.05
def mangoes_cost := calculate_fruit_cost 9 65 0.05 0.06
def oranges_cost := calculate_fruit_cost 6 60 0 0.03
def apples_cost := calculate_fruit_cost 4 80 0.12 0.07

def total_cost := grapes_cost + mangoes_cost + oranges_cost + apples_cost

theorem fruit_cost_theorem : total_cost = 1790.407 := by
  sorry

end fruit_cost_theorem_l977_97783


namespace problem_solution_l977_97797

theorem problem_solution (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y * z = 1) (h2 : x + 1 / z = 7) (h3 : y + 1 / x = 31) :
  z + 1 / y = 5 / 27 := by
sorry

end problem_solution_l977_97797


namespace distinct_arrangements_count_l977_97717

/-- A regular six-pointed star -/
structure SixPointedStar :=
  (points : Fin 12)

/-- The symmetry group of a regular six-pointed star -/
def star_symmetry_group_order : ℕ := 12

/-- The number of distinct arrangements of 12 unique objects on a regular six-pointed star,
    considering reflections and rotations as equivalent -/
def distinct_arrangements (star : SixPointedStar) : ℕ :=
  Nat.factorial 12 / star_symmetry_group_order

theorem distinct_arrangements_count :
  ∀ (star : SixPointedStar), distinct_arrangements star = 39916800 := by
  sorry

end distinct_arrangements_count_l977_97717


namespace point_on_x_axis_l977_97762

/-- A point P with coordinates (m+3, m-1) lies on the x-axis if and only if its coordinates are (4, 0) -/
theorem point_on_x_axis (m : ℝ) : 
  (m - 1 = 0 ∧ (m + 3, m - 1) = (m + 3, 0)) ↔ (m + 3, m - 1) = (4, 0) :=
by sorry

end point_on_x_axis_l977_97762


namespace car_average_speed_l977_97739

/-- Given a car that travels 65 km in the first hour and 45 km in the second hour,
    prove that its average speed is 55 km/h. -/
theorem car_average_speed (distance1 : ℝ) (distance2 : ℝ) (time : ℝ) 
  (h1 : distance1 = 65)
  (h2 : distance2 = 45)
  (h3 : time = 2) :
  (distance1 + distance2) / time = 55 := by
  sorry

end car_average_speed_l977_97739


namespace quarters_spent_at_arcade_l977_97795

theorem quarters_spent_at_arcade (initial_quarters : ℕ) (remaining_quarters : ℕ) 
  (h1 : initial_quarters = 88) 
  (h2 : remaining_quarters = 79) : 
  initial_quarters - remaining_quarters = 9 := by
  sorry

end quarters_spent_at_arcade_l977_97795


namespace log_eight_x_three_halves_l977_97774

theorem log_eight_x_three_halves (x : ℝ) :
  Real.log x / Real.log 8 = 3/2 → x = 16 * Real.sqrt 2 := by
  sorry

end log_eight_x_three_halves_l977_97774


namespace school_selection_probability_l977_97704

theorem school_selection_probability :
  let total_schools : ℕ := 4
  let schools_to_select : ℕ := 2
  let total_combinations : ℕ := (total_schools.choose schools_to_select)
  let favorable_outcomes : ℕ := ((total_schools - 1).choose (schools_to_select - 1))
  favorable_outcomes / total_combinations = 1 / 2 :=
by sorry

end school_selection_probability_l977_97704


namespace derivative_x_squared_cos_x_l977_97791

theorem derivative_x_squared_cos_x (x : ℝ) :
  deriv (fun x => x^2 * Real.cos x) x = 2 * x * Real.cos x - x^2 * Real.sin x :=
by sorry

end derivative_x_squared_cos_x_l977_97791


namespace intersection_of_M_and_N_l977_97754

def M : Set ℤ := {-2, 1, 2}
def N : Set ℤ := {1, 2, 4}

theorem intersection_of_M_and_N :
  M ∩ N = {1, 2} := by sorry

end intersection_of_M_and_N_l977_97754


namespace rectangle_y_value_l977_97790

/-- A rectangle with vertices at (0, 0), (0, 5), (y, 5), and (y, 0) has an area of 35 square units. -/
def rectangle_area (y : ℝ) : Prop :=
  y > 0 ∧ y * 5 = 35

/-- The value of y for which the rectangle has an area of 35 square units is 7. -/
theorem rectangle_y_value : ∃ y : ℝ, rectangle_area y ∧ y = 7 := by
  sorry

end rectangle_y_value_l977_97790


namespace correct_observation_value_l977_97729

theorem correct_observation_value (n : ℕ) (initial_mean corrected_mean wrong_value : ℝ) 
  (h1 : n = 50)
  (h2 : initial_mean = 36)
  (h3 : corrected_mean = 36.02)
  (h4 : wrong_value = 47) :
  let total_sum := n * initial_mean
  let remaining_sum := total_sum - wrong_value
  let corrected_total := n * corrected_mean
  corrected_total - remaining_sum = 48 := by
  sorry

end correct_observation_value_l977_97729


namespace simplify_sqrt_expression_l977_97716

theorem simplify_sqrt_expression :
  2 * Real.sqrt 5 - 3 * Real.sqrt 25 + 4 * Real.sqrt 80 = 18 * Real.sqrt 5 - 15 := by
  sorry

end simplify_sqrt_expression_l977_97716


namespace bounded_function_periodic_l977_97751

/-- A bounded real function satisfying a specific functional equation is periodic with period 1. -/
theorem bounded_function_periodic (f : ℝ → ℝ) 
  (hbounded : ∃ M, ∀ x, |f x| ≤ M) 
  (hcond : ∀ x, f (x + 1/3) + f (x + 1/2) = f x + f (x + 5/6)) : 
  ∀ x, f (x + 1) = f x := by
  sorry

end bounded_function_periodic_l977_97751


namespace subtraction_with_division_l977_97712

theorem subtraction_with_division : 6000 - (105 / 21.0) = 5995 := by sorry

end subtraction_with_division_l977_97712


namespace tire_circumference_l977_97763

-- Define the given conditions
def car_speed : Real := 168 -- km/h
def tire_revolutions : Real := 400 -- revolutions per minute

-- Define the conversion factors
def km_to_m : Real := 1000 -- 1 km = 1000 m
def hour_to_minute : Real := 60 -- 1 hour = 60 minutes

-- Theorem statement
theorem tire_circumference :
  let speed_m_per_minute : Real := car_speed * km_to_m / hour_to_minute
  let circumference : Real := speed_m_per_minute / tire_revolutions
  circumference = 7 := by sorry

end tire_circumference_l977_97763


namespace cube_of_integer_l977_97794

theorem cube_of_integer (n p : ℕ+) (h_prime : Nat.Prime p) (h_p_gt_3 : p > 3)
  (h_div1 : n ∣ (p - 3)) (h_div2 : p ∣ ((n + 1)^3 - 1)) :
  p * n + 1 = (n + 1)^3 := by
  sorry

end cube_of_integer_l977_97794


namespace officers_selection_count_l977_97708

/-- Represents the number of ways to choose officers in a club -/
def choose_officers (total_members boys girls : ℕ) : ℕ :=
  total_members * girls * (boys - 1)

/-- Theorem: The number of ways to choose officers under given conditions is 6300 -/
theorem officers_selection_count :
  let total_members : ℕ := 30
  let boys : ℕ := 15
  let girls : ℕ := 15
  choose_officers total_members boys girls = 6300 := by
  sorry

#eval choose_officers 30 15 15

end officers_selection_count_l977_97708


namespace fraction_problem_l977_97700

theorem fraction_problem (F : ℚ) (m : ℕ) : 
  F = 1/5 ∧ m = 4 → (F^m) * (1/4)^2 = 1/((10:ℚ)^4) :=
by
  sorry

end fraction_problem_l977_97700


namespace yangbajing_largest_in_1975_l977_97734

/-- Represents a geothermal power station -/
structure GeothermalStation where
  name : String
  capacity : ℕ  -- capacity in kilowatts
  country : String
  year_established : ℕ

/-- The set of all geothermal power stations in China in 1975 -/
def china_geothermal_stations_1975 : Set GeothermalStation :=
  sorry

/-- The Yangbajing Geothermal Power Station -/
def yangbajing : GeothermalStation :=
  { name := "Yangbajing Geothermal Power Station"
  , capacity := 50  -- 50 kilowatts in 1975
  , country := "China"
  , year_established := 1975 }

/-- Theorem: Yangbajing was the largest geothermal power station in China in 1975 -/
theorem yangbajing_largest_in_1975 :
  yangbajing ∈ china_geothermal_stations_1975 ∧
  ∀ s ∈ china_geothermal_stations_1975, s.capacity ≤ yangbajing.capacity :=
by
  sorry

end yangbajing_largest_in_1975_l977_97734


namespace cos_alpha_minus_pi_sixth_l977_97799

theorem cos_alpha_minus_pi_sixth (α : ℝ) 
  (h : Real.sin (α + π / 6) + Real.cos α = 4 * Real.sqrt 3 / 5) : 
  Real.cos (α - π / 6) = 4 / 5 := by
  sorry

end cos_alpha_minus_pi_sixth_l977_97799


namespace string_cutting_problem_l977_97796

theorem string_cutting_problem (s l : ℝ) (h1 : s > 0) (h2 : l > 0) 
  (h3 : l - s = 48) (h4 : l + s = 64) : l / s = 7 := by
  sorry

end string_cutting_problem_l977_97796


namespace construction_worker_wage_l977_97747

/-- Represents the daily wage structure for a construction project -/
structure WageStructure where
  worker_wage : ℝ
  electrician_wage : ℝ
  plumber_wage : ℝ
  total_cost : ℝ

/-- Defines the wage structure based on the given conditions -/
def project_wage_structure (w : ℝ) : WageStructure :=
  { worker_wage := w
  , electrician_wage := 2 * w
  , plumber_wage := 2.5 * w
  , total_cost := 2 * w + 2 * w + 2.5 * w }

/-- Theorem stating that the daily wage of a construction worker is $100 -/
theorem construction_worker_wage :
  ∃ w : ℝ, (project_wage_structure w).total_cost = 650 ∧ w = 100 :=
by
  sorry


end construction_worker_wage_l977_97747


namespace total_consumption_is_7700_l977_97701

/-- Fuel consumption rates --/
def highway_rate : ℝ := 3
def city_rate : ℝ := 5

/-- Miles driven each day --/
def day1_highway : ℝ := 200
def day1_city : ℝ := 300
def day2_highway : ℝ := 300
def day2_city : ℝ := 500
def day3_highway : ℝ := 150
def day3_city : ℝ := 350

/-- Total gas consumption calculation --/
def total_consumption : ℝ :=
  (day1_highway * highway_rate + day1_city * city_rate) +
  (day2_highway * highway_rate + day2_city * city_rate) +
  (day3_highway * highway_rate + day3_city * city_rate)

/-- Theorem stating that the total gas consumption is 7700 gallons --/
theorem total_consumption_is_7700 : total_consumption = 7700 := by
  sorry

end total_consumption_is_7700_l977_97701


namespace fraction_sum_inequality_l977_97726

theorem fraction_sum_inequality (b x y z : ℝ) 
  (hb : b > 0) 
  (hx : 0 < x ∧ x < b) 
  (hy : 0 < y ∧ y < b) 
  (hz : 0 < z ∧ z < b) : 
  (x / (b^2 + b*y + z*x)) + (y / (b^2 + b*z + x*y)) + (z / (b^2 + b*x + y*z)) < 1/b := by
  sorry

end fraction_sum_inequality_l977_97726


namespace barbara_initial_candies_l977_97728

/-- The number of candies Barbara used -/
def candies_used : ℝ := 9.0

/-- The number of candies Barbara has left -/
def candies_left : ℕ := 9

/-- The initial number of candies Barbara had -/
def initial_candies : ℝ := candies_used + candies_left

/-- Theorem stating that Barbara initially had 18 candies -/
theorem barbara_initial_candies : initial_candies = 18 := by
  sorry

end barbara_initial_candies_l977_97728


namespace inequality_solution_set_l977_97761

-- Define the solution set based on the value of a
def solutionSet (a : ℝ) : Set ℝ :=
  if a > 0 then {x | x < -a/4 ∨ x > a/3}
  else if a = 0 then {x | x ≠ 0}
  else {x | x > -a/4 ∨ x < a/3}

-- Theorem statement
theorem inequality_solution_set (a : ℝ) :
  {x : ℝ | 12 * x^2 - a * x > a^2} = solutionSet a := by
  sorry

end inequality_solution_set_l977_97761


namespace consecutive_numbers_sum_l977_97702

theorem consecutive_numbers_sum (a : ℕ) (h : a + (a + 1) + (a + 2) + (a + 3) + (a + 4) = 60) :
  a + 4 = 14 := by sorry

end consecutive_numbers_sum_l977_97702


namespace verandah_flooring_rate_l977_97756

def hall_length : ℝ := 20
def hall_width : ℝ := 15
def verandah_width : ℝ := 2.5
def total_cost : ℝ := 700

def total_length : ℝ := hall_length + 2 * verandah_width
def total_width : ℝ := hall_width + 2 * verandah_width

def hall_area : ℝ := hall_length * hall_width
def total_area : ℝ := total_length * total_width
def verandah_area : ℝ := total_area - hall_area

theorem verandah_flooring_rate :
  total_cost / verandah_area = 3.5 := by sorry

end verandah_flooring_rate_l977_97756


namespace remaining_honey_l977_97742

/-- Theorem: Remaining honey after bear consumption --/
theorem remaining_honey (total_honey : ℝ) (eaten_honey : ℝ) 
  (h1 : total_honey = 0.36)
  (h2 : eaten_honey = 0.05) : 
  total_honey - eaten_honey = 0.31 := by
sorry

end remaining_honey_l977_97742


namespace station_distance_l977_97766

theorem station_distance (d : ℝ) : 
  (d > 0) → 
  (∃ (x_speed y_speed : ℝ), x_speed > 0 ∧ y_speed > 0 ∧ 
    (d + 100) / x_speed = (d - 100) / y_speed ∧
    (2 * d + 300) / x_speed = (d + 400) / y_speed) →
  (2 * d = 600) := by sorry

end station_distance_l977_97766


namespace min_perimeter_isosceles_triangles_l977_97755

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  leg : ℕ
  base : ℕ

/-- Calculates the perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℕ := 2 * t.leg + t.base

/-- Calculates the area of an isosceles triangle -/
noncomputable def area (t : IsoscelesTriangle) : ℝ :=
  (t.base : ℝ) * (((t.leg : ℝ) ^ 2 - ((t.base : ℝ) / 2) ^ 2).sqrt) / 2

theorem min_perimeter_isosceles_triangles :
  ∃ (t1 t2 : IsoscelesTriangle),
    t1 ≠ t2 ∧
    perimeter t1 = perimeter t2 ∧
    area t1 = area t2 ∧
    5 * t1.base = 4 * t2.base ∧
    perimeter t1 = 740 ∧
    (∀ (s1 s2 : IsoscelesTriangle),
      s1 ≠ s2 →
      perimeter s1 = perimeter s2 →
      area s1 = area s2 →
      5 * s1.base = 4 * s2.base →
      perimeter s1 ≥ 740) :=
by sorry

end min_perimeter_isosceles_triangles_l977_97755


namespace teaching_years_difference_l977_97706

/-- The combined total of teaching years for Virginia, Adrienne, and Dennis -/
def total_years : ℕ := 102

/-- The number of years Dennis has taught -/
def dennis_years : ℕ := 43

/-- The number of years Virginia has taught -/
def virginia_years : ℕ := 34

/-- The number of years Adrienne has taught -/
def adrienne_years : ℕ := 25

theorem teaching_years_difference :
  total_years = virginia_years + adrienne_years + dennis_years ∧
  virginia_years = adrienne_years + 9 ∧
  virginia_years < dennis_years →
  dennis_years - virginia_years = 9 := by
sorry

end teaching_years_difference_l977_97706


namespace domain_of_f_l977_97715

noncomputable def f (x : ℝ) := (2 * x - 3) / (x^2 - 5 * x + 6)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < 2 ∨ (2 < x ∧ x < 3) ∨ 3 < x} := by
  sorry

end domain_of_f_l977_97715


namespace equilateral_triangle_area_l977_97780

/-- The area of an equilateral triangle with altitude √8 is 32√3/3 square units. -/
theorem equilateral_triangle_area (h : ℝ) (altitude_eq : h = Real.sqrt 8) :
  let side := (4 * Real.sqrt 6) / 3
  let area := (Real.sqrt 3 / 4) * side^2
  area = 32 * Real.sqrt 3 / 3 := by
  sorry


end equilateral_triangle_area_l977_97780


namespace unique_age_group_split_l977_97745

theorem unique_age_group_split (total_students : ℕ) 
  (under_10_fraction : ℚ) (between_10_12_fraction : ℚ) (between_12_14_fraction : ℚ) :
  total_students = 60 →
  under_10_fraction = 1/4 →
  between_10_12_fraction = 1/2 →
  between_12_14_fraction = 1/6 →
  ∃! (under_10 between_10_12 between_12_14 above_14 : ℕ),
    under_10 + between_10_12 + between_12_14 + above_14 = total_students ∧
    under_10 = (under_10_fraction * total_students).num ∧
    between_10_12 = (between_10_12_fraction * total_students).num ∧
    between_12_14 = (between_12_14_fraction * total_students).num ∧
    above_14 = total_students - (under_10 + between_10_12 + between_12_14) :=
by
  sorry

end unique_age_group_split_l977_97745


namespace megan_fourth_game_score_l977_97784

/-- Represents Megan's basketball scores --/
structure MeganScores where
  threeGameAverage : ℝ
  fourGameAverage : ℝ

/-- Calculates Megan's score in the fourth game --/
def fourthGameScore (scores : MeganScores) : ℝ :=
  4 * scores.fourGameAverage - 3 * scores.threeGameAverage

/-- Theorem stating Megan's score in the fourth game --/
theorem megan_fourth_game_score :
  ∀ (scores : MeganScores),
    scores.threeGameAverage = 18 →
    scores.fourGameAverage = 17 →
    fourthGameScore scores = 14 := by
  sorry

#eval fourthGameScore { threeGameAverage := 18, fourGameAverage := 17 }

end megan_fourth_game_score_l977_97784


namespace gloria_leftover_money_l977_97798

/-- Calculates the amount of money Gloria has left after selling her trees and buying a cabin -/
def gloria_money_left (initial_cash : ℕ) (cypress_count : ℕ) (pine_count : ℕ) (maple_count : ℕ)
  (cypress_price : ℕ) (pine_price : ℕ) (maple_price : ℕ) (cabin_price : ℕ) : ℕ :=
  let total_earned := initial_cash + cypress_count * cypress_price + pine_count * pine_price + maple_count * maple_price
  total_earned - cabin_price

/-- Theorem stating that Gloria will have $350 left after buying the cabin -/
theorem gloria_leftover_money :
  gloria_money_left 150 20 600 24 100 200 300 129000 = 350 := by
  sorry

end gloria_leftover_money_l977_97798


namespace trigonometric_equation_solution_l977_97725

theorem trigonometric_equation_solution (x : ℝ) : 
  (2 * Real.sin x - Real.sin (2 * x)) / (2 * Real.sin x + Real.sin (2 * x)) + 
  (Real.cos (x / 2) / Real.sin (x / 2))^2 = 10 / 3 →
  ∃ k : ℤ, x = π / 3 * (3 * ↑k + 1) ∨ x = π / 3 * (3 * ↑k - 1) :=
by sorry

end trigonometric_equation_solution_l977_97725


namespace number_equality_l977_97724

theorem number_equality : ∃ x : ℝ, (0.4 * x = 0.3 * 50) ∧ (x = 37.5) := by
  sorry

end number_equality_l977_97724


namespace distance_from_circle_center_to_point_l977_97727

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 = 4*x + 6*y + 3

-- Define the center of the circle
def circle_center : ℝ × ℝ :=
  let center_x := 2
  let center_y := 3
  (center_x, center_y)

-- Define the given point
def given_point : ℝ × ℝ := (10, 5)

-- State the theorem
theorem distance_from_circle_center_to_point :
  let (cx, cy) := circle_center
  let (px, py) := given_point
  Real.sqrt ((px - cx)^2 + (py - cy)^2) = 2 * Real.sqrt 17 :=
by sorry

end distance_from_circle_center_to_point_l977_97727


namespace fraction_addition_simplest_form_l977_97777

theorem fraction_addition : (13 : ℚ) / 15 + (7 : ℚ) / 9 = (74 : ℚ) / 45 := by
  sorry

theorem simplest_form : Int.gcd 74 45 = 1 := by
  sorry

end fraction_addition_simplest_form_l977_97777


namespace arithmetic_mean_problem_l977_97760

theorem arithmetic_mean_problem (x : ℚ) : 
  (((x + 10) + 18 + 3*x + 16 + (x + 5) + (3*x + 6)) / 6 = 25) → x = 95/8 := by
  sorry

end arithmetic_mean_problem_l977_97760


namespace orange_bin_problem_l977_97732

theorem orange_bin_problem (initial : ℕ) (thrown_away : ℕ) (final : ℕ) 
  (h1 : initial = 40)
  (h2 : thrown_away = 25)
  (h3 : final = 36) :
  final - (initial - thrown_away) = 21 := by
  sorry

end orange_bin_problem_l977_97732


namespace abs_neg_five_l977_97746

theorem abs_neg_five : abs (-5 : ℤ) = 5 := by sorry

end abs_neg_five_l977_97746


namespace no_savings_on_joint_purchase_l977_97785

/-- The price of a single window -/
def window_price : ℕ := 100

/-- The number of windows needed to get one free -/
def windows_for_free : ℕ := 3

/-- The number of windows Dave needs -/
def dave_windows : ℕ := 11

/-- The number of windows Doug needs -/
def doug_windows : ℕ := 9

/-- Calculate the cost of windows given the number needed -/
def calculate_cost (windows_needed : ℕ) : ℕ :=
  let free_windows := windows_needed / windows_for_free
  let paid_windows := windows_needed - free_windows
  paid_windows * window_price

/-- The theorem stating that there's no savings when purchasing together -/
theorem no_savings_on_joint_purchase :
  calculate_cost dave_windows + calculate_cost doug_windows =
  calculate_cost (dave_windows + doug_windows) :=
sorry

end no_savings_on_joint_purchase_l977_97785


namespace value_of_expression_l977_97744

theorem value_of_expression (m n : ℤ) (h : m - n = -2) : 2 - 5*m + 5*n = 12 := by
  sorry

end value_of_expression_l977_97744


namespace correct_calculation_l977_97782

theorem correct_calculation (x : ℝ) : x + 10 = 21 → x * 10 = 110 := by
  sorry

end correct_calculation_l977_97782


namespace homework_problem_count_l977_97753

theorem homework_problem_count (math_pages reading_pages problems_per_page : ℕ) : 
  math_pages = 4 → reading_pages = 6 → problems_per_page = 4 →
  (math_pages + reading_pages) * problems_per_page = 40 := by
sorry

end homework_problem_count_l977_97753


namespace pig_price_l977_97764

/-- Given 5 pigs and 15 hens with a total cost of 2100 currency units,
    and an average price of 30 currency units per hen,
    prove that the average price of a pig is 330 currency units. -/
theorem pig_price (num_pigs : ℕ) (num_hens : ℕ) (total_cost : ℕ) (hen_price : ℕ) :
  num_pigs = 5 →
  num_hens = 15 →
  total_cost = 2100 →
  hen_price = 30 →
  (total_cost - num_hens * hen_price) / num_pigs = 330 := by
  sorry

end pig_price_l977_97764


namespace three_weighings_sufficient_and_necessary_l977_97786

/-- Represents the outcome of a weighing on a balance scale -/
inductive WeighResult
  | Equal : WeighResult
  | LeftHeavier : WeighResult
  | RightHeavier : WeighResult

/-- A type representing a weighing strategy -/
def WeighStrategy := List (List Nat × List Nat)

/-- Represents the state of knowledge about which coin might be fake -/
structure FakeCoinInfo where
  possibleFakes : List Nat
  isHeavy : Option Bool

/-- The total number of coins -/
def totalCoins : Nat := 13

/-- A theorem stating that 3 weighings are sufficient and necessary to identify the fake coin -/
theorem three_weighings_sufficient_and_necessary :
  ∃ (strategy : WeighStrategy),
    (strategy.length ≤ 3) ∧
    (∀ (fakeCoin : Nat) (isHeavy : Bool),
      fakeCoin < totalCoins →
      ∃ (finalInfo : FakeCoinInfo),
        finalInfo.possibleFakes = [fakeCoin] ∧
        finalInfo.isHeavy = some isHeavy) ∧
    (∀ (strategy' : WeighStrategy),
      strategy'.length < 3 →
      ∃ (fakeCoin1 fakeCoin2 : Nat) (isHeavy1 isHeavy2 : Bool),
        fakeCoin1 ≠ fakeCoin2 ∧
        fakeCoin1 < totalCoins ∧
        fakeCoin2 < totalCoins ∧
        ¬∃ (finalInfo : FakeCoinInfo),
          (finalInfo.possibleFakes = [fakeCoin1] ∧ finalInfo.isHeavy = some isHeavy1) ∨
          (finalInfo.possibleFakes = [fakeCoin2] ∧ finalInfo.isHeavy = some isHeavy2)) :=
by sorry

end three_weighings_sufficient_and_necessary_l977_97786


namespace cubic_factorization_sum_of_squares_l977_97779

theorem cubic_factorization_sum_of_squares (a b c d e f : ℤ) :
  (∀ x : ℝ, 1728 * x^3 + 64 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) →
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 23456 := by
  sorry

end cubic_factorization_sum_of_squares_l977_97779


namespace solution_characterization_l977_97773

def valid_solution (a b c x y z : ℕ) : Prop :=
  a + b + c = x * y * z ∧
  x + y + z = a * b * c ∧
  a ≥ b ∧ b ≥ c ∧ c ≥ 1 ∧
  x ≥ y ∧ y ≥ z ∧ z ≥ 1

def solution_set : Set (ℕ × ℕ × ℕ × ℕ × ℕ × ℕ) :=
  {(2, 2, 2, 6, 1, 1), (5, 2, 1, 8, 1, 1), (3, 3, 1, 7, 1, 1), (3, 2, 1, 6, 2, 1)}

theorem solution_characterization :
  ∀ a b c x y z : ℕ, valid_solution a b c x y z ↔ (a, b, c, x, y, z) ∈ solution_set :=
sorry

end solution_characterization_l977_97773


namespace x_value_l977_97703

theorem x_value : ∃ x : ℝ, x = 12 * (1 + 0.2) ∧ x = 14.4 := by
  sorry

end x_value_l977_97703


namespace product_of_distinct_prime_factors_of_B_l977_97752

def divisors_of_60 : List ℕ := [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60]

def B : ℕ := (List.prod divisors_of_60)

theorem product_of_distinct_prime_factors_of_B :
  (Finset.prod (Finset.filter Nat.Prime (Finset.range (B + 1))) id) = 30 := by
  sorry

end product_of_distinct_prime_factors_of_B_l977_97752


namespace shortest_player_height_l977_97757

theorem shortest_player_height (tallest_height shortest_height height_difference : ℝ) :
  tallest_height = 77.75 →
  height_difference = 9.5 →
  tallest_height = shortest_height + height_difference →
  shortest_height = 68.25 := by
sorry

end shortest_player_height_l977_97757


namespace value_of_M_l977_97741

theorem value_of_M : ∃ M : ℝ, (0.25 * M = 0.35 * 1200) ∧ (M = 1680) := by
  sorry

end value_of_M_l977_97741


namespace arithmetic_progression_formula_l977_97767

/-- An arithmetic progression with specific conditions -/
def ArithmeticProgression (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧
  a 3 + a 11 = 24 ∧
  a 4 = 3

/-- The general term formula for the arithmetic progression -/
def GeneralTermFormula (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n = 3 * n - 9

/-- Theorem stating that the given arithmetic progression has the specified general term formula -/
theorem arithmetic_progression_formula (a : ℕ → ℝ) :
  ArithmeticProgression a → GeneralTermFormula a := by sorry

end arithmetic_progression_formula_l977_97767


namespace f_always_negative_l977_97748

/-- The function f(x) = ax^2 + ax - 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + a * x - 1

/-- Theorem stating that f(x) < 0 for all x ∈ ℝ if and only if -4 < a ≤ 0 -/
theorem f_always_negative (a : ℝ) : 
  (∀ x : ℝ, f a x < 0) ↔ (-4 < a ∧ a ≤ 0) := by sorry

end f_always_negative_l977_97748


namespace six_balls_three_boxes_l977_97707

/-- Represents a distribution of balls into boxes -/
def Distribution := List Nat

/-- Counts the number of ways to distribute indistinguishable balls into indistinguishable boxes -/
def countDistributions (balls : Nat) (boxes : Nat) : Nat :=
  sorry

/-- Theorem: There are 7 ways to distribute 6 indistinguishable balls into 3 indistinguishable boxes -/
theorem six_balls_three_boxes : countDistributions 6 3 = 7 := by
  sorry

end six_balls_three_boxes_l977_97707


namespace diana_bottle_caps_l977_97723

/-- The number of bottle caps Diana starts with -/
def initial_caps : ℕ := 65

/-- The number of bottle caps eaten by the hippopotamus -/
def eaten_caps : ℕ := 4

/-- The number of bottle caps Diana ends with -/
def final_caps : ℕ := initial_caps - eaten_caps

theorem diana_bottle_caps : final_caps = 61 := by
  sorry

end diana_bottle_caps_l977_97723


namespace multiply_polynomials_l977_97788

theorem multiply_polynomials (x : ℝ) : (x^4 + 8*x^2 + 64) * (x^2 - 8) = x^4 + 16*x^2 := by
  sorry

end multiply_polynomials_l977_97788


namespace decreasing_cubic_function_l977_97738

-- Define the function f(x) = ax³ - 2x
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 2 * x

-- State the theorem
theorem decreasing_cubic_function (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x ≥ f a y) → a ≤ 0 := by
  sorry

end decreasing_cubic_function_l977_97738


namespace unit_fraction_decomposition_l977_97781

theorem unit_fraction_decomposition (n : ℕ+) : 
  (1 : ℚ) / n = 1 / (2 * n) + 1 / (3 * n) + 1 / (6 * n) := by
  sorry

end unit_fraction_decomposition_l977_97781


namespace corner_cut_pentagon_area_l977_97735

/-- A pentagon formed by cutting a triangular corner from a rectangular sheet. -/
structure CornerCutPentagon where
  sides : Finset ℝ
  is_valid : sides = {14, 21, 22, 28, 35}

/-- The area of a CornerCutPentagon is 759.5 -/
theorem corner_cut_pentagon_area (p : CornerCutPentagon) : ∃ (area : ℝ), area = 759.5 := by
  sorry

#check corner_cut_pentagon_area

end corner_cut_pentagon_area_l977_97735


namespace two_wizards_theorem_l977_97787

/-- Represents a student in the wizardry school -/
structure Student where
  id : Fin 13
  hasDiploma : Bool

/-- The configuration of students around the table -/
def StudentConfiguration := Fin 13 → Student

/-- Check if a student's prediction is correct -/
def isPredictionCorrect (config : StudentConfiguration) (s : Student) : Bool :=
  let otherStudents := (List.range 13).filter (fun i => 
    i ≠ s.id.val ∧ 
    i ≠ (s.id.val + 1) % 13 ∧ 
    i ≠ (s.id.val + 12) % 13)
  otherStudents.all (fun i => ¬(config i).hasDiploma)

/-- The main theorem to prove -/
theorem two_wizards_theorem :
  ∃ (config : StudentConfiguration),
    (∀ s, (config s.id = s)) ∧
    (∃! (s1 s2 : Student), s1.hasDiploma ∧ s2.hasDiploma ∧ s1 ≠ s2) ∧
    (∀ s, s.hasDiploma ↔ isPredictionCorrect config s) := by
  sorry


end two_wizards_theorem_l977_97787


namespace parallel_lines_distance_l977_97775

def line1 (t : ℝ) : ℝ × ℝ := (3 + 2*t, -4 - 5*t)
def line2 (s : ℝ) : ℝ × ℝ := (2 + 2*s, -6 - 5*s)

def direction : ℝ × ℝ := (2, -5)

theorem parallel_lines_distance :
  let v := (3 - 2, -4 - (-6))
  let projection := ((v.1 * direction.1 + v.2 * direction.2) / (direction.1^2 + direction.2^2)) • direction
  let perpendicular := (v.1 - projection.1, v.2 - projection.2)
  Real.sqrt (perpendicular.1^2 + perpendicular.2^2) = Real.sqrt 2349 / 29 := by
  sorry

end parallel_lines_distance_l977_97775


namespace triangle_abc_right_angled_l977_97733

theorem triangle_abc_right_angled (A B C : ℝ) 
  (h1 : A = (1/2) * B) 
  (h2 : A = (1/3) * C) 
  (h3 : A + B + C = 180) : 
  C = 90 := by
  sorry

end triangle_abc_right_angled_l977_97733


namespace lcm_ratio_sum_l977_97778

theorem lcm_ratio_sum (a b : ℕ+) : 
  Nat.lcm a b = 42 → 
  a * 3 = b * 2 → 
  a + b = 70 := by
sorry

end lcm_ratio_sum_l977_97778


namespace geometric_series_common_ratio_l977_97711

theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 4/7
  let a₂ : ℚ := -8/21
  let a₃ : ℚ := 16/63
  let r : ℚ := a₂ / a₁
  (r = -2/3) ∧ (a₃ / a₂ = r) := by sorry

end geometric_series_common_ratio_l977_97711


namespace egyptian_fraction_equation_solutions_l977_97769

theorem egyptian_fraction_equation_solutions :
  ∀ x y z : ℕ+,
  (1 : ℚ) / x + (1 : ℚ) / y + (1 : ℚ) / z = 4 / 5 →
  ((x = 2 ∧ y = 4 ∧ z = 20) ∨ (x = 2 ∧ y = 5 ∧ z = 10)) :=
by sorry

end egyptian_fraction_equation_solutions_l977_97769


namespace daves_shirts_l977_97759

theorem daves_shirts (short_sleeve : ℕ) (long_sleeve : ℕ) (washed : ℕ) (not_washed : ℕ) : 
  long_sleeve = 27 →
  washed = 20 →
  not_washed = 16 →
  short_sleeve + long_sleeve = washed + not_washed →
  short_sleeve = 9 := by
sorry

end daves_shirts_l977_97759


namespace boat_speed_in_still_water_l977_97718

/-- Given a boat that travels 11 km along a stream and 5 km against the stream in one hour,
    prove that its speed in still water is 8 km/hr. -/
theorem boat_speed_in_still_water : 
  ∀ (boat_speed stream_speed : ℝ),
    boat_speed + stream_speed = 11 →
    boat_speed - stream_speed = 5 →
    boat_speed = 8 := by
  sorry

end boat_speed_in_still_water_l977_97718


namespace simplify_expression_l977_97792

theorem simplify_expression (x : ℝ) : 4 * (x^2 - 5*x) - 5 * (2*x^2 + 3*x) = -6*x^2 - 35*x := by
  sorry

end simplify_expression_l977_97792


namespace marbles_fraction_taken_l977_97713

theorem marbles_fraction_taken (chris_marbles ryan_marbles remaining_marbles : ℕ) 
  (h1 : chris_marbles = 12)
  (h2 : ryan_marbles = 28)
  (h3 : remaining_marbles = 20) :
  (chris_marbles + ryan_marbles - remaining_marbles) / (chris_marbles + ryan_marbles) = 1 / 2 := by
  sorry

end marbles_fraction_taken_l977_97713


namespace amount_distribution_l977_97730

theorem amount_distribution (A : ℝ) : 
  (A / 14 = A / 18 + 80) → A = 5040 := by
  sorry

end amount_distribution_l977_97730


namespace horse_food_per_day_l977_97737

/-- Given the ratio of sheep to horses, number of sheep, and total horse food,
    prove the amount of food each horse gets per day. -/
theorem horse_food_per_day
  (sheep_horse_ratio : ℚ) -- Ratio of sheep to horses
  (num_sheep : ℕ) -- Number of sheep
  (total_horse_food : ℕ) -- Total amount of horse food in ounces
  (h1 : sheep_horse_ratio = 2 / 7) -- The ratio of sheep to horses is 2:7
  (h2 : num_sheep = 16) -- There are 16 sheep on the farm
  (h3 : total_horse_food = 12880) -- The farm needs 12,880 ounces of horse food per day
  : ℕ :=
by
  sorry

#check horse_food_per_day

end horse_food_per_day_l977_97737


namespace red_cars_count_l977_97771

theorem red_cars_count (black_cars : ℕ) (ratio_red : ℕ) (ratio_black : ℕ) : 
  black_cars = 70 → ratio_red = 3 → ratio_black = 8 → 
  (ratio_red : ℚ) / (ratio_black : ℚ) * black_cars = 26 := by
  sorry

end red_cars_count_l977_97771


namespace intersection_and_parallel_line_equation_l977_97758

/-- Given two lines in the plane and their intersection point, prove that a third line
    passing through the intersection point and parallel to a fourth line has a specific equation. -/
theorem intersection_and_parallel_line_equation :
  -- Define the first line: 2x - 3y - 3 = 0
  let l₁ : Set (ℝ × ℝ) := {p | 2 * p.1 - 3 * p.2 - 3 = 0}
  -- Define the second line: x + y + 2 = 0
  let l₂ : Set (ℝ × ℝ) := {p | p.1 + p.2 + 2 = 0}
  -- Define the parallel line: 3x + y - 1 = 0
  let l_parallel : Set (ℝ × ℝ) := {p | 3 * p.1 + p.2 - 1 = 0}
  -- Define the intersection point of l₁ and l₂
  let intersection : ℝ × ℝ := (-3/5, -7/5)
  -- Assume the intersection point lies on both l₁ and l₂
  (intersection ∈ l₁) ∧ (intersection ∈ l₂) →
  -- Define the line we want to prove
  let l : Set (ℝ × ℝ) := {p | 15 * p.1 + 5 * p.2 + 16 = 0}
  -- The line l passes through the intersection point
  (intersection ∈ l) ∧
  -- The line l is parallel to l_parallel
  (∀ (p q : ℝ × ℝ), p ∈ l → q ∈ l → p ≠ q →
    ∃ (r s : ℝ × ℝ), r ∈ l_parallel ∧ s ∈ l_parallel ∧ r ≠ s ∧
      (s.2 - r.2) / (s.1 - r.1) = (q.2 - p.2) / (q.1 - p.1)) :=
by
  sorry


end intersection_and_parallel_line_equation_l977_97758


namespace complex_modulus_l977_97765

theorem complex_modulus (i : ℂ) (h : i * i = -1) : 
  Complex.abs (5 * i / (2 - i)) = Real.sqrt 5 := by sorry

end complex_modulus_l977_97765


namespace banana_permutations_eq_60_l977_97722

/-- The number of distinct permutations of the word BANANA -/
def banana_permutations : ℕ :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem banana_permutations_eq_60 : banana_permutations = 60 := by
  sorry

end banana_permutations_eq_60_l977_97722


namespace sum_of_special_numbers_l977_97721

/-- A natural number that ends with 7 zeros and has exactly 72 divisors -/
def SeventyTwoDivisorNumber (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 10^7 * k ∧ (Nat.divisors n).card = 72

theorem sum_of_special_numbers :
  ∃ (a b : ℕ), a ≠ b ∧
    SeventyTwoDivisorNumber a ∧
    SeventyTwoDivisorNumber b ∧
    a + b = 70000000 :=
sorry

end sum_of_special_numbers_l977_97721


namespace quadratic_function_range_l977_97793

/-- Given a quadratic function f(x) = ax^2 + bx, prove that if f(-1) is between -1 and 2,
    and f(1) is between 2 and 4, then f(-2) is between -1 and 10. -/
theorem quadratic_function_range (a b : ℝ) :
  let f := fun x : ℝ => a * x^2 + b * x
  ((-1 : ℝ) ≤ f (-1) ∧ f (-1) ≤ 2) →
  (2 ≤ f 1 ∧ f 1 ≤ 4) →
  ((-1 : ℝ) ≤ f (-2) ∧ f (-2) ≤ 10) :=
by sorry

end quadratic_function_range_l977_97793


namespace range_of_a_l977_97710

theorem range_of_a (a : ℝ) : 
  (a > 0 ∧ ∀ x : ℝ, x > 0 → (Real.exp x / x + a * Real.log x - a * x + Real.exp 2) ≥ 0) →
  (0 < a ∧ a ≤ Real.exp 2) :=
by sorry

end range_of_a_l977_97710


namespace driving_time_ratio_l977_97736

theorem driving_time_ratio : 
  ∀ (t_28 t_60 : ℝ),
  t_28 + t_60 = 30 →
  t_28 * 28 + t_60 * 60 = 11 * 120 →
  t_28 = 15 ∧ t_28 / (t_28 + t_60) = 1 / 2 :=
by sorry

end driving_time_ratio_l977_97736


namespace t_shape_perimeter_specific_l977_97770

/-- The perimeter of a T-shape formed by two rectangles --/
def t_shape_perimeter (length width overlap : ℝ) : ℝ :=
  2 * (2 * (length + width)) - 2 * overlap

/-- Theorem: The perimeter of a T-shape formed by two 3x5 inch rectangles with a 1.5 inch overlap is 29 inches --/
theorem t_shape_perimeter_specific : t_shape_perimeter 5 3 1.5 = 29 := by
  sorry

end t_shape_perimeter_specific_l977_97770


namespace adam_ella_equation_l977_97740

theorem adam_ella_equation (d e : ℝ) : 
  (∀ x, |x - 8| = 3 ↔ x^2 + d*x + e = 0) → 
  d = -16 ∧ e = 55 := by
sorry

end adam_ella_equation_l977_97740


namespace ratio_of_negatives_l977_97743

theorem ratio_of_negatives (x y : ℝ) (hx : x < 0) (hy : y < 0) (h : 3 * x - 2 * y = Real.sqrt (x * y)) : 
  y / x = 9 / 4 := by
sorry

end ratio_of_negatives_l977_97743


namespace aunt_uncle_gift_amount_l977_97749

/-- The amount of money Chris had before his birthday -/
def initial_amount : ℕ := 159

/-- The amount Chris received from his grandmother -/
def grandmother_gift : ℕ := 25

/-- The amount Chris received from his parents -/
def parents_gift : ℕ := 75

/-- The total amount Chris has after his birthday -/
def final_amount : ℕ := 279

/-- The amount Chris received from his aunt and uncle -/
def aunt_uncle_gift : ℕ := final_amount - initial_amount - grandmother_gift - parents_gift

theorem aunt_uncle_gift_amount : aunt_uncle_gift = 20 := by
  sorry

end aunt_uncle_gift_amount_l977_97749
