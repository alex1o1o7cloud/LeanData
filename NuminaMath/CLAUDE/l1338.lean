import Mathlib

namespace sum_of_solutions_equation_l1338_133895

theorem sum_of_solutions_equation (x : ℝ) :
  (x ≠ 1 ∧ x ≠ -1) →
  ((-12 * x) / (x^2 - 1) = (3 * x) / (x + 1) - 8 / (x - 1)) →
  ∃ (y : ℝ), (y ≠ 1 ∧ y ≠ -1) ∧
    ((-12 * y) / (y^2 - 1) = (3 * y) / (y + 1) - 8 / (y - 1)) ∧
    (x + y = 10 / 3) :=
by sorry

end sum_of_solutions_equation_l1338_133895


namespace inequality_proof_l1338_133836

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (h_prod : a * b * c * d = 1) :
  ((1 + a * b) / (1 + a)) ^ 2008 + 
  ((1 + b * c) / (1 + b)) ^ 2008 + 
  ((1 + c * d) / (1 + c)) ^ 2008 + 
  ((1 + d * a) / (1 + d)) ^ 2008 ≥ 4 := by
  sorry

end inequality_proof_l1338_133836


namespace zach_lawn_mowing_pay_l1338_133883

/-- Represents the financial situation for Zach's bike savings --/
structure BikeSavings where
  bikeCost : ℕ
  weeklyAllowance : ℕ
  currentSavings : ℕ
  babysittingPayRate : ℕ
  babysittingHours : ℕ
  additionalNeeded : ℕ

/-- Calculates the amount Zach's parent should pay him to mow the lawn --/
def lawnMowingPay (s : BikeSavings) : ℕ :=
  s.bikeCost - s.currentSavings - s.weeklyAllowance - s.babysittingPayRate * s.babysittingHours - s.additionalNeeded

/-- Theorem stating that the amount Zach's parent will pay him to mow the lawn is 10 --/
theorem zach_lawn_mowing_pay :
  let s : BikeSavings := {
    bikeCost := 100
    weeklyAllowance := 5
    currentSavings := 65
    babysittingPayRate := 7
    babysittingHours := 2
    additionalNeeded := 6
  }
  lawnMowingPay s = 10 := by sorry

end zach_lawn_mowing_pay_l1338_133883


namespace sod_area_calculation_sod_area_is_9474_l1338_133861

/-- Calculates the area of sod needed for Jill's front yard -/
theorem sod_area_calculation (front_yard_width front_yard_length sidewalk_width sidewalk_length
                              flowerbed1_depth flowerbed1_length flowerbed2_width flowerbed2_length
                              flowerbed3_width flowerbed3_length : ℕ) : ℕ :=
  let front_yard_area := front_yard_width * front_yard_length
  let sidewalk_area := sidewalk_width * sidewalk_length
  let flowerbed1_area := 2 * (flowerbed1_depth * flowerbed1_length)
  let flowerbed2_area := flowerbed2_width * flowerbed2_length
  let flowerbed3_area := flowerbed3_width * flowerbed3_length
  let total_subtract_area := sidewalk_area + flowerbed1_area + flowerbed2_area + flowerbed3_area
  front_yard_area - total_subtract_area

/-- Proves that the area of sod needed for Jill's front yard is 9,474 square feet -/
theorem sod_area_is_9474 :
  sod_area_calculation 200 50 3 50 4 25 10 12 7 8 = 9474 := by
  sorry

end sod_area_calculation_sod_area_is_9474_l1338_133861


namespace room_with_193_black_tiles_has_1089_total_tiles_l1338_133858

/-- Represents a square room with tiled floor -/
structure TiledRoom where
  side_length : ℕ
  black_tile_count : ℕ

/-- Calculates the number of black tiles in a square room with given side length -/
def black_tiles (s : ℕ) : ℕ := 6 * s - 5

/-- Calculates the total number of tiles in a square room with given side length -/
def total_tiles (s : ℕ) : ℕ := s * s

/-- Theorem stating that a square room with 193 black tiles has 1089 total tiles -/
theorem room_with_193_black_tiles_has_1089_total_tiles :
  ∃ (room : TiledRoom), room.black_tile_count = 193 ∧ total_tiles room.side_length = 1089 :=
by sorry

end room_with_193_black_tiles_has_1089_total_tiles_l1338_133858


namespace first_prime_in_special_product_l1338_133877

theorem first_prime_in_special_product (x y z : Nat) : 
  Nat.Prime x ∧ Nat.Prime y ∧ Nat.Prime z ∧  -- x, y, z are prime
  x ≠ y ∧ x ≠ z ∧ y ≠ z ∧  -- x, y, z are different
  (∃ (divisors : Finset Nat), divisors.card = 12 ∧ 
    ∀ d ∈ divisors, (x^2 * y * z) % d = 0) →  -- x^2 * y * z has 12 divisors
  x = 2 :=
by sorry

end first_prime_in_special_product_l1338_133877


namespace average_speed_calculation_l1338_133813

-- Define the cycling parameters
def cycling_speed : ℝ := 20
def cycling_time : ℝ := 1

-- Define the walking parameters
def walking_speed : ℝ := 3
def walking_time : ℝ := 2

-- Define the total distance and time
def total_distance : ℝ := cycling_speed * cycling_time + walking_speed * walking_time
def total_time : ℝ := cycling_time + walking_time

-- Theorem statement
theorem average_speed_calculation :
  total_distance / total_time = 26 / 3 := by sorry

end average_speed_calculation_l1338_133813


namespace perfect_square_quotient_l1338_133864

theorem perfect_square_quotient (a b : ℕ+) (h : (a * b + 1) ∣ (a ^ 2 + b ^ 2)) :
  ∃ k : ℕ, (a ^ 2 + b ^ 2) / (a * b + 1) = k ^ 2 := by
  sorry

end perfect_square_quotient_l1338_133864


namespace framed_photo_ratio_l1338_133811

/-- Represents the dimensions of a framed photograph -/
structure FramedPhoto where
  original_width : ℝ
  original_height : ℝ
  frame_width : ℝ

/-- Calculates the area of the original photograph -/
def original_area (photo : FramedPhoto) : ℝ :=
  photo.original_width * photo.original_height

/-- Calculates the area of the framed photograph -/
def framed_area (photo : FramedPhoto) : ℝ :=
  (photo.original_width + 2 * photo.frame_width) * (photo.original_height + 6 * photo.frame_width)

/-- Theorem: The ratio of the shorter to the longer dimension of the framed photograph is 1:2 -/
theorem framed_photo_ratio (photo : FramedPhoto) 
  (h1 : photo.original_width = 20)
  (h2 : photo.original_height = 30)
  (h3 : framed_area photo = 2 * original_area photo) :
  (photo.original_width + 2 * photo.frame_width) / (photo.original_height + 6 * photo.frame_width) = 1 / 2 := by
  sorry

end framed_photo_ratio_l1338_133811


namespace family_ages_l1338_133841

/-- Family ages problem -/
theorem family_ages :
  ∀ (son_age man_age daughter_age wife_age : ℝ),
  (man_age = son_age + 29) →
  (man_age + 2 = 2 * (son_age + 2)) →
  (daughter_age = son_age - 3.5) →
  (wife_age = 1.5 * daughter_age) →
  (son_age = 27 ∧ man_age = 56 ∧ daughter_age = 23.5 ∧ wife_age = 35.25) :=
by
  sorry

#check family_ages

end family_ages_l1338_133841


namespace joe_initial_cars_l1338_133868

/-- Given that Joe will have 62 cars after getting 12 more, prove that he initially had 50 cars. -/
theorem joe_initial_cars : 
  ∀ (initial_cars : ℕ), 
  (initial_cars + 12 = 62) → 
  initial_cars = 50 := by
sorry

end joe_initial_cars_l1338_133868


namespace range_of_g_l1338_133824

theorem range_of_g (x : ℝ) (h : x ∈ Set.Icc (-1) 1) :
  -Real.pi^2 / 2 ≤ (Real.arccos x)^2 - (Real.arcsin x)^2 ∧ 
  (Real.arccos x)^2 - (Real.arcsin x)^2 ≤ Real.pi^2 / 2 := by
  sorry

end range_of_g_l1338_133824


namespace natashas_distance_l1338_133821

/-- The distance to Natasha's destination given her speed and travel time -/
theorem natashas_distance (speed_limit : ℝ) (over_limit : ℝ) (travel_time : ℝ) 
  (h1 : speed_limit = 50)
  (h2 : over_limit = 10)
  (h3 : travel_time = 1) :
  speed_limit + over_limit * travel_time = 60 := by
  sorry

end natashas_distance_l1338_133821


namespace first_group_size_l1338_133816

/-- The number of students in the first group -/
def first_group_count : ℕ := sorry

/-- The number of students in the second group -/
def second_group_count : ℕ := 11

/-- The total number of students in both groups -/
def total_students : ℕ := 31

/-- The average height of students in centimeters -/
def average_height : ℝ := 20

theorem first_group_size :
  (first_group_count : ℝ) * average_height +
  (second_group_count : ℝ) * average_height =
  (total_students : ℝ) * average_height ∧
  first_group_count + second_group_count = total_students →
  first_group_count = 20 := by
  sorry

end first_group_size_l1338_133816


namespace sum_of_odds_15_to_51_l1338_133874

def arithmetic_sum (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) : ℕ := 
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem sum_of_odds_15_to_51 : 
  arithmetic_sum 15 51 2 = 627 := by sorry

end sum_of_odds_15_to_51_l1338_133874


namespace special_polynomial_q_count_l1338_133849

/-- A polynomial of degree 4 with specific properties -/
structure SpecialPolynomial where
  o : ℤ
  p : ℤ
  q : ℤ
  roots_distinct : True  -- represents that the roots are distinct
  roots_positive : True  -- represents that the roots are positive
  one_integer_root : True  -- represents that exactly one root is an integer
  integer_root_sum : True  -- represents that the integer root is the sum of two other roots

/-- The number of possible values for q in the special polynomial -/
def count_q_values : ℕ := 1003001

/-- Theorem stating the number of possible q values -/
theorem special_polynomial_q_count :
  ∀ (poly : SpecialPolynomial), count_q_values = 1003001 := by
  sorry

end special_polynomial_q_count_l1338_133849


namespace solution_is_correct_l1338_133870

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- Define the equation
def equation (y : ℝ) : Prop :=
  log 3 ((4*y + 16) / (6*y - 9)) + log 3 ((6*y - 9) / (2*y - 5)) = 3

-- Theorem statement
theorem solution_is_correct :
  equation (151/50) := by sorry

end solution_is_correct_l1338_133870


namespace inequality_chain_l1338_133817

theorem inequality_chain (m n : ℝ) 
  (hm : m < 0) 
  (hn : n > 0) 
  (hmn : m + n < 0) : 
  m < -n ∧ -n < n ∧ n < -m :=
by sorry

end inequality_chain_l1338_133817


namespace circle1_satisfies_conditions_circle2_passes_through_points_l1338_133806

-- Define the circle equations
def circle1 (x y : ℝ) : Prop := (x + 1)^2 + (y + 2)^2 = 10
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y + 1 = 0

-- Define the line equation
def line (x y : ℝ) : Prop := x - 2*y - 3 = 0

-- Theorem for the first circle
theorem circle1_satisfies_conditions :
  (∃ x y : ℝ, line x y ∧ circle1 x y) ∧
  circle1 2 (-3) ∧
  circle1 (-2) (-5) := by sorry

-- Theorem for the second circle
theorem circle2_passes_through_points :
  circle2 1 0 ∧
  circle2 (-1) (-2) ∧
  circle2 3 (-2) := by sorry

end circle1_satisfies_conditions_circle2_passes_through_points_l1338_133806


namespace log_arithmetic_progression_implies_power_relation_l1338_133814

theorem log_arithmetic_progression_implies_power_relation
  (k m n x : ℝ)
  (hk : k > 0)
  (hm : m > 0)
  (hn : n > 0)
  (hx_pos : x > 0)
  (hx_neq_one : x ≠ 1)
  (h_arith_prog : 2 * (Real.log x / Real.log m) = 
                  (Real.log x / Real.log k) + (Real.log x / Real.log n)) :
  n^2 = (n*k)^(Real.log m / Real.log k) :=
by sorry

end log_arithmetic_progression_implies_power_relation_l1338_133814


namespace f_symmetry_l1338_133857

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

-- State the theorem
theorem f_symmetry (a b : ℝ) : f a b (-2) = 10 → f a b 2 = -26 := by
  sorry

end f_symmetry_l1338_133857


namespace least_sum_of_bases_l1338_133891

theorem least_sum_of_bases (a b : ℕ+) : 
  (7 * a.val + 8 = 8 * b.val + 7) → 
  (∀ c d : ℕ+, (7 * c.val + 8 = 8 * d.val + 7) → (c.val + d.val ≥ a.val + b.val)) →
  a.val + b.val = 17 := by
sorry

end least_sum_of_bases_l1338_133891


namespace exists_zero_sum_subset_l1338_133844

/-- Represents a row in the table -/
def Row (n : ℕ) := Fin n → Int

/-- The table with all possible rows of 1 and -1 -/
def OriginalTable (n : ℕ) : Finset (Row n) :=
  sorry

/-- A function that potentially replaces some elements with zero -/
def Corrupt (n : ℕ) : Row n → Row n :=
  sorry

/-- The corrupted table after replacing some elements with zero -/
def CorruptedTable (n : ℕ) : Finset (Row n) :=
  sorry

/-- Sum of a set of rows -/
def RowSum (n : ℕ) (rows : Finset (Row n)) : Row n :=
  sorry

/-- A row of all zeros -/
def ZeroRow (n : ℕ) : Row n :=
  sorry

/-- The main theorem -/
theorem exists_zero_sum_subset (n : ℕ) :
  ∃ (subset : Finset (Row n)), subset ⊆ CorruptedTable n ∧ RowSum n subset = ZeroRow n :=
sorry

end exists_zero_sum_subset_l1338_133844


namespace max_sequence_length_l1338_133879

theorem max_sequence_length (a : ℕ → ℤ) (n : ℕ) : 
  (∀ i : ℕ, i + 6 < n → (a i + a (i+1) + a (i+2) + a (i+3) + a (i+4) + a (i+5) + a (i+6) > 0)) →
  (∀ i : ℕ, i + 10 < n → (a i + a (i+1) + a (i+2) + a (i+3) + a (i+4) + a (i+5) + a (i+6) + a (i+7) + a (i+8) + a (i+9) + a (i+10) < 0)) →
  n ≤ 18 :=
by sorry

end max_sequence_length_l1338_133879


namespace jury_stabilization_jury_stabilization_30_l1338_133860

/-- Represents a jury member -/
structure JuryMember where
  id : Nat

/-- Represents the state of the jury after a voting session -/
structure JuryState where
  members : List JuryMember
  sessionCount : Nat

/-- Represents a voting process -/
def votingProcess (state : JuryState) : JuryState :=
  sorry

/-- Theorem: For a jury with 2n members (n ≥ 2), the jury stabilizes after at most n sessions -/
theorem jury_stabilization (n : Nat) (h : n ≥ 2) :
  ∀ (initialState : JuryState),
    initialState.members.length = 2 * n →
    ∃ (finalState : JuryState),
      finalState = (votingProcess^[n]) initialState ∧
      finalState.members = ((votingProcess^[n + 1]) initialState).members :=
by
  sorry

/-- Corollary: A jury with 30 members stabilizes after at most 15 sessions -/
theorem jury_stabilization_30 :
  ∀ (initialState : JuryState),
    initialState.members.length = 30 →
    ∃ (finalState : JuryState),
      finalState = (votingProcess^[15]) initialState ∧
      finalState.members = ((votingProcess^[16]) initialState).members :=
by
  sorry

end jury_stabilization_jury_stabilization_30_l1338_133860


namespace fractional_equation_solution_exists_l1338_133852

theorem fractional_equation_solution_exists : ∃ m : ℝ, ∃ x : ℝ, x ≠ 1 ∧ (x + 2) / (x - 1) = m / (1 - x) := by
  sorry

end fractional_equation_solution_exists_l1338_133852


namespace traveler_water_consumption_l1338_133822

/-- The amount of water drunk by the traveler and camel -/
theorem traveler_water_consumption (traveler_ounces : ℝ) : 
  traveler_ounces > 0 →  -- Assume the traveler drinks a positive amount
  (∃ (camel_ounces : ℝ), 
    camel_ounces = 7 * traveler_ounces ∧  -- Camel drinks 7 times as much
    128 * 2 = traveler_ounces + camel_ounces) →  -- Total consumption is 2 gallons
  traveler_ounces = 32 := by
sorry

end traveler_water_consumption_l1338_133822


namespace trigonometric_simplification_l1338_133897

open Real

theorem trigonometric_simplification (α x : ℝ) :
  ((sin (π - α) * cos (3*π - α) * tan (-α - π) * tan (α - 2*π)) / 
   (tan (4*π - α) * sin (5*π + α)) = sin α) ∧
  ((sin (3*π - x) / tan (5*π - x)) * 
   (1 / (tan (5*π/2 - x) * tan (4.5*π - x))) * 
   (cos (2*π - x) / sin (-x)) = sin x) := by
  sorry

end trigonometric_simplification_l1338_133897


namespace complex_number_equality_l1338_133820

theorem complex_number_equality (a : ℝ) : 
  let z : ℂ := (a + Complex.I) / Complex.I
  (z.re = z.im) → a = -1 := by
sorry

end complex_number_equality_l1338_133820


namespace right_triangle_base_length_l1338_133866

theorem right_triangle_base_length 
  (area : ℝ) 
  (hypotenuse : ℝ) 
  (side : ℝ) 
  (h_area : area = 24) 
  (h_hypotenuse : hypotenuse = 10) 
  (h_side : side = 8) : 
  ∃ (base height : ℝ), 
    area = (1/2) * base * height ∧ 
    hypotenuse^2 = base^2 + height^2 ∧ 
    (base = side ∨ height = side) ∧ 
    base = 8 := by
  sorry

#check right_triangle_base_length

end right_triangle_base_length_l1338_133866


namespace even_function_implies_a_plus_minus_one_l1338_133834

/-- A function f : ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def EvenFunction (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

theorem even_function_implies_a_plus_minus_one (a : ℝ) :
  EvenFunction (fun x => x^2 + (a^2 - 1)*x + (a - 1)) →
  a = 1 ∨ a = -1 :=
by sorry

end even_function_implies_a_plus_minus_one_l1338_133834


namespace arithmetic_sequence_common_difference_l1338_133862

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_a2 : a 2 = 2) 
  (h_a4 : a 4 = 8) : 
  ∃ d : ℝ, (∀ n : ℕ, a (n + 1) - a n = d) ∧ d = 3 := by
sorry

end arithmetic_sequence_common_difference_l1338_133862


namespace yellow_marbles_count_l1338_133819

def total_marbles : ℕ := 240

theorem yellow_marbles_count (y b : ℕ) 
  (h1 : y + b = total_marbles) 
  (h2 : b = y - 2) : 
  y = 121 := by
  sorry

end yellow_marbles_count_l1338_133819


namespace vector_subtraction_magnitude_l1338_133888

def angle_between (a b : ℝ × ℝ) : ℝ := sorry

theorem vector_subtraction_magnitude (a b : ℝ × ℝ) 
  (h1 : angle_between a b = π / 3)
  (h2 : a = (2, 0))
  (h3 : Real.sqrt ((Prod.fst b)^2 + (Prod.snd b)^2) = 1) :
  Real.sqrt ((Prod.fst (a - 2 • b))^2 + (Prod.snd (a - 2 • b))^2) = 2 := by
  sorry

end vector_subtraction_magnitude_l1338_133888


namespace suzanna_bike_ride_l1338_133815

/-- Suzanna's bike ride problem -/
theorem suzanna_bike_ride (distance_per_interval : ℝ) (interval_duration : ℝ) 
  (initial_ride_duration : ℝ) (break_duration : ℝ) (final_ride_duration : ℝ) 
  (h1 : distance_per_interval = 1.5)
  (h2 : interval_duration = 7)
  (h3 : initial_ride_duration = 21)
  (h4 : break_duration = 5)
  (h5 : final_ride_duration = 14) :
  (initial_ride_duration / interval_duration) * distance_per_interval + 
  (final_ride_duration / interval_duration) * distance_per_interval = 7.5 := by
  sorry

#check suzanna_bike_ride

end suzanna_bike_ride_l1338_133815


namespace rabbit_calories_l1338_133828

/-- Brandon's hunting scenario -/
structure HuntingScenario where
  squirrels_per_hour : ℕ := 6
  rabbits_per_hour : ℕ := 2
  calories_per_squirrel : ℕ := 300
  calorie_difference : ℕ := 200

/-- Calculates the calories per rabbit in Brandon's hunting scenario -/
def calories_per_rabbit (scenario : HuntingScenario) : ℕ :=
  (scenario.squirrels_per_hour * scenario.calories_per_squirrel - scenario.calorie_difference) / scenario.rabbits_per_hour

/-- Theorem stating that each rabbit has 800 calories in Brandon's scenario -/
theorem rabbit_calories (scenario : HuntingScenario) :
  calories_per_rabbit scenario = 800 := by
  sorry

end rabbit_calories_l1338_133828


namespace missing_number_proof_l1338_133882

def known_numbers : List ℤ := [744, 745, 747, 748, 752, 752, 753, 755, 755]

theorem missing_number_proof (total_count : ℕ) (average : ℤ) (missing_number : ℤ) :
  total_count = 10 →
  average = 750 →
  missing_number = 1549 →
  (List.sum known_numbers + missing_number) / total_count = average :=
by sorry

end missing_number_proof_l1338_133882


namespace expression_is_perfect_square_l1338_133887

/-- The expression is a perfect square when x = 0.04 -/
theorem expression_is_perfect_square : 
  ∃ y : ℝ, (11.98 * 11.98 + 11.98 * 0.04 + 0.02 * 0.02) = y * y := by
  sorry

end expression_is_perfect_square_l1338_133887


namespace box_filling_proof_l1338_133886

theorem box_filling_proof (length width depth : ℕ) (num_cubes : ℕ) : 
  length = 49 → 
  width = 42 → 
  depth = 14 → 
  num_cubes = 84 → 
  ∃ (cube_side : ℕ), 
    cube_side > 0 ∧ 
    length % cube_side = 0 ∧ 
    width % cube_side = 0 ∧ 
    depth % cube_side = 0 ∧ 
    (length / cube_side) * (width / cube_side) * (depth / cube_side) = num_cubes :=
by
  sorry

#check box_filling_proof

end box_filling_proof_l1338_133886


namespace unique_monic_polynomial_l1338_133894

/-- A monic polynomial of degree 3 satisfying specific conditions -/
def f (x : ℝ) : ℝ := x^3 + 2*x^2 + 3*x + 4

/-- Theorem stating that f is the unique monic polynomial of degree 3 satisfying given conditions -/
theorem unique_monic_polynomial :
  (∀ x, f x = x^3 + 2*x^2 + 3*x + 4) ∧
  f 0 = 4 ∧ f 1 = 10 ∧ f (-1) = 2 ∧
  (∀ g : ℝ → ℝ, (∃ a b c : ℝ, ∀ x, g x = x^3 + a*x^2 + b*x + c) →
    g 0 = 4 → g 1 = 10 → g (-1) = 2 → g = f) :=
by sorry

end unique_monic_polynomial_l1338_133894


namespace cheese_arrangement_count_l1338_133898

/-- Represents a cheese flavor -/
inductive Flavor
| Paprika
| BearsGarlic

/-- Represents a cheese slice -/
structure CheeseSlice :=
  (flavor : Flavor)

/-- Represents a box of cheese slices -/
structure CheeseBox :=
  (slices : List CheeseSlice)

/-- Represents an arrangement of cheese slices in two boxes -/
structure CheeseArrangement :=
  (box1 : CheeseBox)
  (box2 : CheeseBox)

/-- Checks if two arrangements are equivalent under rotation -/
def areEquivalentUnderRotation (arr1 arr2 : CheeseArrangement) : Prop :=
  sorry

/-- Counts the number of distinct arrangements -/
def countDistinctArrangements (arrangements : List CheeseArrangement) : Nat :=
  sorry

/-- The main theorem to prove -/
theorem cheese_arrangement_count :
  let totalSlices := 16
  let paprikaSlices := 8
  let bearsGarlicSlices := 8
  let allArrangements := sorry -- List of all possible arrangements
  countDistinctArrangements allArrangements = 234 :=
sorry

end cheese_arrangement_count_l1338_133898


namespace pipe_filling_time_l1338_133859

theorem pipe_filling_time (pipe_a_rate pipe_b_rate total_time : ℚ) 
  (h1 : pipe_a_rate = 1 / 12)
  (h2 : pipe_b_rate = 1 / 20)
  (h3 : total_time = 10) :
  ∃ (x : ℚ), 
    x * (pipe_a_rate + pipe_b_rate) + (total_time - x) * pipe_b_rate = 1 ∧ 
    x = 6 := by
  sorry

end pipe_filling_time_l1338_133859


namespace all_numbers_multiple_of_three_l1338_133808

def is_multiple_of_three (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 3 * k

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

def numbers_to_check : List ℕ := [123, 234, 345, 456, 567]

theorem all_numbers_multiple_of_three 
  (h : ∀ n : ℕ, is_multiple_of_three n ↔ is_multiple_of_three (sum_of_digits n)) :
  ∀ n ∈ numbers_to_check, is_multiple_of_three n :=
by sorry

end all_numbers_multiple_of_three_l1338_133808


namespace some_fast_animals_are_pets_l1338_133899

-- Define our universe
variable (U : Type)

-- Define our predicates
variable (Wolf FastAnimal Pet : U → Prop)

-- State the theorem
theorem some_fast_animals_are_pets
  (h1 : ∀ x, Wolf x → FastAnimal x)
  (h2 : ∃ x, Pet x ∧ Wolf x) :
  ∃ x, FastAnimal x ∧ Pet x :=
sorry

end some_fast_animals_are_pets_l1338_133899


namespace circle_center_polar_coordinates_l1338_133871

/-- Given a circle with polar coordinate equation ρ = 2(cosθ + sinθ), 
    the polar coordinates of its center are (√2, π/4) -/
theorem circle_center_polar_coordinates :
  ∀ ρ θ : ℝ, 
  ρ = 2 * (Real.cos θ + Real.sin θ) →
  ∃ r α : ℝ, 
    r = Real.sqrt 2 ∧ 
    α = π / 4 ∧ 
    (r * Real.cos α - 1)^2 + (r * Real.sin α - 1)^2 = 2 := by
  sorry


end circle_center_polar_coordinates_l1338_133871


namespace rosie_pie_making_l1338_133839

/-- Represents the number of pies that can be made from a given number of apples -/
def pies_from_apples (apples : ℚ) : ℚ :=
  (2 / 9) * apples

/-- Represents the number of apples left after making pies -/
def apples_left (total_apples : ℚ) (pies_made : ℚ) : ℚ :=
  total_apples - (pies_made * (9 / 2))

theorem rosie_pie_making (total_apples : ℚ) 
  (h1 : total_apples = 36) : 
  pies_from_apples total_apples = 8 ∧ 
  apples_left total_apples (pies_from_apples total_apples) = 0 := by
  sorry

end rosie_pie_making_l1338_133839


namespace units_digit_sum_l1338_133873

theorem units_digit_sum (n m : ℕ) : (35^87 + 3^45) % 10 = 8 := by
  sorry

end units_digit_sum_l1338_133873


namespace inequality_problem_l1338_133851

theorem inequality_problem (a b : ℝ) (h : a < b ∧ b < 0) :
  (1/a > 1/b) ∧ (abs a > -b) ∧ (Real.sqrt (-a) > Real.sqrt (-b)) ∧ ¬(1/(a-b) > 1/a) := by
  sorry

end inequality_problem_l1338_133851


namespace units_digit_of_7_power_2023_l1338_133890

theorem units_digit_of_7_power_2023 : (7^2023 : ℕ) % 10 = 3 := by
  sorry

end units_digit_of_7_power_2023_l1338_133890


namespace numeric_methods_students_count_second_year_students_count_l1338_133827

/-- The number of second-year students studying numeric methods -/
def numeric_methods_students : ℕ := 241

/-- The number of second-year students studying automatic control of airborne vehicles -/
def acav_students : ℕ := 423

/-- The number of second-year students studying both numeric methods and ACAV -/
def both_subjects_students : ℕ := 134

/-- The total number of students in the faculty -/
def total_students : ℕ := 663

/-- The proportion of second-year students in the faculty -/
def second_year_proportion : ℚ := 4/5

/-- The total number of second-year students -/
def total_second_year_students : ℕ := 530

theorem numeric_methods_students_count :
  numeric_methods_students + acav_students - both_subjects_students = total_second_year_students :=
by sorry

theorem second_year_students_count :
  total_second_year_students = (total_students : ℚ) * second_year_proportion :=
by sorry

end numeric_methods_students_count_second_year_students_count_l1338_133827


namespace square_root_pattern_square_root_ten_squared_minus_one_l1338_133838

theorem square_root_pattern (n : ℕ) (hn : n ≥ 3) :
  ∀ m : ℕ, m ≥ 3 → m ≤ 5 →
  Real.sqrt (m^2 - 1) = Real.sqrt (m - 1) * Real.sqrt (m + 1) :=
  sorry

theorem square_root_ten_squared_minus_one :
  Real.sqrt (10^2 - 1) = 3 * Real.sqrt 11 :=
  sorry

end square_root_pattern_square_root_ten_squared_minus_one_l1338_133838


namespace binomial_sum_abs_coefficients_l1338_133881

theorem binomial_sum_abs_coefficients :
  ∀ (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℤ),
  (∀ x : ℝ, (1 - 2*x)^7 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  |a₀| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| + |a₇| = 2187 :=
by sorry

end binomial_sum_abs_coefficients_l1338_133881


namespace vector_problem_solution_l1338_133848

def vector_problem (a b : ℝ × ℝ) (m : ℝ) : Prop :=
  let norm_a : ℝ := Real.sqrt ((a.1 ^ 2) + (a.2 ^ 2))
  let norm_b : ℝ := Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2))
  let dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2
  norm_a = 3 ∧
  norm_b = 2 ∧
  dot_product a b = norm_a * norm_b * (-1/2) ∧
  dot_product (a.1 + m * b.1, a.2 + m * b.2) a = 0

theorem vector_problem_solution (a b : ℝ × ℝ) (m : ℝ) 
  (h : vector_problem a b m) : m = 3 := by
  sorry

end vector_problem_solution_l1338_133848


namespace jessica_quarters_l1338_133837

/-- Calculates the number of quarters Jessica has after her sister borrows some. -/
def quarters_remaining (initial : ℕ) (borrowed : ℕ) : ℕ :=
  initial - borrowed

/-- Theorem stating that if Jessica had 8 quarters initially and her sister borrowed 3,
    then Jessica now has 5 quarters. -/
theorem jessica_quarters :
  quarters_remaining 8 3 = 5 := by
  sorry

end jessica_quarters_l1338_133837


namespace min_cos_plus_sin_l1338_133853

theorem min_cos_plus_sin (A : Real) :
  let f := λ A : Real => Real.cos (A / 2) + Real.sin (A / 2)
  ∃ (min_value : Real), 
    (∀ A, f A ≥ min_value) ∧ 
    (min_value = -Real.sqrt 2) ∧
    (f (π / 2) = min_value) :=
by sorry

end min_cos_plus_sin_l1338_133853


namespace waiter_new_customers_l1338_133892

theorem waiter_new_customers 
  (initial_customers : ℕ) 
  (customers_left : ℕ) 
  (final_customers : ℕ) 
  (h1 : initial_customers = 33) 
  (h2 : customers_left = 31) 
  (h3 : final_customers = 28) : 
  final_customers - (initial_customers - customers_left) = 26 := by
  sorry

end waiter_new_customers_l1338_133892


namespace student_rabbit_difference_l1338_133823

/-- The number of third-grade classrooms -/
def num_classrooms : ℕ := 4

/-- The number of students in each third-grade classroom -/
def students_per_classroom : ℕ := 18

/-- The number of pet rabbits in each third-grade classroom -/
def rabbits_per_classroom : ℕ := 2

/-- The difference between the total number of students and the total number of rabbits -/
theorem student_rabbit_difference : 
  num_classrooms * students_per_classroom - num_classrooms * rabbits_per_classroom = 64 := by
  sorry

end student_rabbit_difference_l1338_133823


namespace solution_set_of_inequality_l1338_133802

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_decreasing_on_nonneg (f : ℝ → ℝ) : Prop := 
  ∀ x y, 0 ≤ x → 0 ≤ y → x < y → f y < f x

-- State the theorem
theorem solution_set_of_inequality 
  (h_even : is_even f) 
  (h_decreasing : is_decreasing_on_nonneg f) :
  {x : ℝ | f (2*x + 5) > f (x^2 + 2)} = {x : ℝ | x < -1 ∨ x > 3} := by
  sorry

end solution_set_of_inequality_l1338_133802


namespace isosceles_right_triangle_ratio_l1338_133801

theorem isosceles_right_triangle_ratio (a c : ℝ) (h1 : a > 0) (h2 : c > 0) : 
  (a^2 + a^2 = c^2) → (2 * a / c = Real.sqrt 2) := by
  sorry

end isosceles_right_triangle_ratio_l1338_133801


namespace polynomial_degree_problem_l1338_133850

theorem polynomial_degree_problem (m n : ℤ) : 
  (m + 1 + 2 = 6) →  -- Degree of the polynomial term x^(m+1)y^2 is 6
  (2*n + (5 - m) = 6) →  -- Degree of the monomial x^(2n)y^(5-m) is 6
  (-m)^3 + 2*n = -23 := by
sorry

end polynomial_degree_problem_l1338_133850


namespace jessica_withdrawal_l1338_133845

theorem jessica_withdrawal (initial_balance : ℝ) (withdrawal : ℝ) : 
  withdrawal = (2 / 5) * initial_balance ∧
  (3 / 5) * initial_balance + (1 / 2) * ((3 / 5) * initial_balance) = 450 →
  withdrawal = 200 := by
sorry

end jessica_withdrawal_l1338_133845


namespace prime_power_divisors_l1338_133893

theorem prime_power_divisors (p q : ℕ) (n : ℕ) : 
  Prime p → Prime q → (Nat.divisors (p^n * q^6)).card = 28 → n = 3 := by
  sorry

end prime_power_divisors_l1338_133893


namespace probability_three_teachers_same_gate_l1338_133885

-- Define the number of teachers and gates
def num_teachers : ℕ := 12
def num_gates : ℕ := 3
def teachers_per_gate : ℕ := 4

-- Define the probability function
noncomputable def probability_same_gate : ℚ :=
  3 / 55

-- Theorem statement
theorem probability_three_teachers_same_gate :
  probability_same_gate = 3 / 55 :=
by
  sorry

end probability_three_teachers_same_gate_l1338_133885


namespace sector_central_angle_l1338_133878

theorem sector_central_angle (arc_length : Real) (area : Real) :
  arc_length = 2 * Real.pi ∧ area = 5 * Real.pi →
  ∃ (central_angle : Real),
    central_angle = 72 ∧
    central_angle * Real.pi / 180 = 2 * Real.pi * Real.pi / (5 * Real.pi) :=
by sorry

end sector_central_angle_l1338_133878


namespace confectioner_pastries_l1338_133830

theorem confectioner_pastries :
  ∀ (P : ℕ) (x : ℕ),
    (P = 28 * (10 + x)) →
    (P = 49 * (4 + x)) →
    P = 392 :=
by
  sorry

end confectioner_pastries_l1338_133830


namespace dress_cost_equals_total_savings_l1338_133865

/-- Calculates the cost of the dress based on initial savings, weekly allowance, weekly spending, and waiting period. -/
def dress_cost (initial_savings : ℕ) (weekly_allowance : ℕ) (weekly_spending : ℕ) (waiting_weeks : ℕ) : ℕ :=
  initial_savings + (weekly_allowance - weekly_spending) * waiting_weeks

/-- Proves that the dress cost is equal to Vanessa's total savings after the waiting period. -/
theorem dress_cost_equals_total_savings :
  dress_cost 20 30 10 3 = 80 := by
  sorry

end dress_cost_equals_total_savings_l1338_133865


namespace acute_triangle_theorem_l1338_133869

theorem acute_triangle_theorem (A B C : Real) (a b c : Real) :
  0 < A → A < π / 2 →
  0 < B → B < π / 2 →
  0 < C → C < π / 2 →
  A + B + C = π →
  a > 0 → b > 0 → c > 0 →
  Real.sqrt 3 * b * Real.sin A - a * Real.cos B - a = 0 →
  (B = π / 3) ∧ 
  (3 * Real.sqrt 3 / 2 < Real.sin A + Real.sin C ∧ Real.sin A + Real.sin C ≤ Real.sqrt 3) :=
by sorry

end acute_triangle_theorem_l1338_133869


namespace inequality_proof_l1338_133843

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^4 - y^4 = x - y) :
  (x - y) / (x^6 - y^6) ≤ (4/3) * (x + y) := by
  sorry

end inequality_proof_l1338_133843


namespace root_sum_reciprocals_l1338_133812

theorem root_sum_reciprocals (p q r s : ℂ) : 
  p^4 + 6*p^3 + 11*p^2 + 6*p + 3 = 0 →
  q^4 + 6*q^3 + 11*q^2 + 6*q + 3 = 0 →
  r^4 + 6*r^3 + 11*r^2 + 6*r + 3 = 0 →
  s^4 + 6*s^3 + 11*s^2 + 6*s + 3 = 0 →
  p ≠ q → p ≠ r → p ≠ s → q ≠ r → q ≠ s → r ≠ s →
  1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(q*r) + 1/(q*s) + 1/(r*s) = 11/3 := by
sorry

end root_sum_reciprocals_l1338_133812


namespace at_least_one_geq_two_l1338_133829

theorem at_least_one_geq_two (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + 1/b ≥ 2) ∨ (b + 1/c ≥ 2) ∨ (c + 1/a ≥ 2) := by
  sorry

end at_least_one_geq_two_l1338_133829


namespace emmalyn_fence_count_l1338_133840

/-- The number of fences Emmalyn painted -/
def number_of_fences : ℕ := 50

/-- The price per meter in dollars -/
def price_per_meter : ℚ := 0.20

/-- The length of each fence in meters -/
def fence_length : ℕ := 500

/-- The total earnings in dollars -/
def total_earnings : ℕ := 5000

/-- Theorem stating that the number of fences Emmalyn painted is correct -/
theorem emmalyn_fence_count :
  number_of_fences = total_earnings / (price_per_meter * fence_length) := by
  sorry

end emmalyn_fence_count_l1338_133840


namespace misha_money_total_l1338_133832

theorem misha_money_total (initial_money earned_money : ℕ) : 
  initial_money = 34 → earned_money = 13 → initial_money + earned_money = 47 := by
  sorry

end misha_money_total_l1338_133832


namespace regular_octagon_diagonal_l1338_133896

theorem regular_octagon_diagonal (s : ℝ) (h : s = 12) : 
  let diagonal := s * Real.sqrt 2
  diagonal = 12 * Real.sqrt 2 := by
sorry

end regular_octagon_diagonal_l1338_133896


namespace geometric_sequence_sum_l1338_133833

def geometric_sequence (a₁ q : ℝ) : ℕ → ℝ
  | n => a₁ * q ^ (n - 1)

theorem geometric_sequence_sum (a₁ q : ℝ) :
  ∃ (a₁ q : ℝ),
    (geometric_sequence a₁ q 2 + geometric_sequence a₁ q 4 = 20) ∧
    (geometric_sequence a₁ q 3 + geometric_sequence a₁ q 5 = 40) →
    (geometric_sequence a₁ q 5 + geometric_sequence a₁ q 7 = 160) := by
  sorry

end geometric_sequence_sum_l1338_133833


namespace sqrt_product_equality_l1338_133805

theorem sqrt_product_equality : 2 * Real.sqrt 3 * (5 * Real.sqrt 6) = 30 * Real.sqrt 2 := by
  sorry

end sqrt_product_equality_l1338_133805


namespace a_in_M_necessary_not_sufficient_for_a_in_N_l1338_133825

-- Define the sets M and N
def M : Set ℝ := {x | x > 0}
def N : Set ℝ := {x | x > 1}

-- Theorem statement
theorem a_in_M_necessary_not_sufficient_for_a_in_N :
  (∀ a : ℝ, a ∈ N → a ∈ M) ∧ 
  (∃ a : ℝ, a ∈ M ∧ a ∉ N) :=
by sorry

end a_in_M_necessary_not_sufficient_for_a_in_N_l1338_133825


namespace used_cd_price_l1338_133880

theorem used_cd_price (n u : ℝ) 
  (eq1 : 6 * n + 2 * u = 127.92)
  (eq2 : 3 * n + 8 * u = 133.89) :
  u = 9.99 := by
sorry

end used_cd_price_l1338_133880


namespace division_of_decimals_l1338_133884

theorem division_of_decimals : (0.05 : ℝ) / 0.01 = 5 := by
  sorry

end division_of_decimals_l1338_133884


namespace pizza_payment_difference_l1338_133835

theorem pizza_payment_difference :
  -- Define the total number of slices
  let total_slices : ℕ := 12
  -- Define the cost of the plain pizza
  let plain_cost : ℚ := 12
  -- Define the additional cost for mushrooms
  let mushroom_cost : ℚ := 3
  -- Define the number of slices with mushrooms (one-third of the pizza)
  let mushroom_slices : ℕ := total_slices / 3
  -- Define the number of slices Laura ate
  let laura_slices : ℕ := mushroom_slices + 2
  -- Define the number of slices Jessica ate
  let jessica_slices : ℕ := total_slices - laura_slices
  -- Calculate the total cost of the pizza
  let total_cost : ℚ := plain_cost + mushroom_cost
  -- Calculate the cost per slice
  let cost_per_slice : ℚ := total_cost / total_slices
  -- Calculate Laura's payment
  let laura_payment : ℚ := laura_slices * cost_per_slice
  -- Calculate Jessica's payment (only plain slices)
  let jessica_payment : ℚ := jessica_slices * (plain_cost / total_slices)
  -- The difference in payment
  laura_payment - jessica_payment = 1.5 := by
  sorry

end pizza_payment_difference_l1338_133835


namespace pipe_stack_height_l1338_133863

/-- The height of a stack of three pipes in an isosceles triangular configuration -/
theorem pipe_stack_height (d : ℝ) (h : d = 12) : 
  let r := d / 2
  let base_center_distance := 2 * r
  let triangle_height := Real.sqrt (base_center_distance ^ 2 - (base_center_distance / 2) ^ 2)
  let total_height := triangle_height + 2 * r
  total_height = 12 + 6 * Real.sqrt 3 :=
by sorry

end pipe_stack_height_l1338_133863


namespace sqrt_sum_fractions_l1338_133810

theorem sqrt_sum_fractions : 
  Real.sqrt ((25 : ℝ) / 36 + 16 / 9) = Real.sqrt 89 / 6 := by
  sorry

end sqrt_sum_fractions_l1338_133810


namespace circle_point_x_value_l1338_133876

/-- Given a circle with diameter endpoints (-8, 0) and (32, 0), 
    if the point (x, 20) lies on this circle, then x = 12. -/
theorem circle_point_x_value 
  (x : ℝ) 
  (h : (x - 12)^2 + 20^2 = ((32 - (-8)) / 2)^2) : 
  x = 12 := by
sorry

end circle_point_x_value_l1338_133876


namespace geometric_sequence_sum_l1338_133800

/-- A geometric sequence with common ratio 2 and specific sum condition -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) = 2 * a n) ∧ (a 1 + a 2 + a 3 = 21)

/-- The sum of the 3rd, 4th, and 5th terms equals 84 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h : GeometricSequence a) :
    a 3 + a 4 + a 5 = 84 := by
  sorry

end geometric_sequence_sum_l1338_133800


namespace number_equation_solution_l1338_133818

theorem number_equation_solution : ∃ x : ℝ, 
  (0.6667 * x + 1 = 0.75 * x) ∧ 
  (abs (x - 12) < 0.01) := by
  sorry

end number_equation_solution_l1338_133818


namespace ceiling_of_3_7_l1338_133847

theorem ceiling_of_3_7 : ⌈(3.7 : ℝ)⌉ = 4 := by sorry

end ceiling_of_3_7_l1338_133847


namespace cubic_properties_l1338_133854

theorem cubic_properties :
  (∀ x : ℝ, x^3 > 0 → x > 0) ∧
  (∀ x : ℝ, x < 1 → x^3 < x) :=
by sorry

end cubic_properties_l1338_133854


namespace tracy_art_fair_sales_l1338_133846

theorem tracy_art_fair_sales (total_customers : ℕ) (first_group : ℕ) (second_group : ℕ) (last_group : ℕ)
  (first_group_purchases : ℕ) (second_group_purchases : ℕ) (total_sales : ℕ) :
  total_customers = first_group + second_group + last_group →
  first_group = 4 →
  second_group = 12 →
  last_group = 4 →
  first_group_purchases = 2 →
  second_group_purchases = 1 →
  total_sales = 36 →
  ∃ (last_group_purchases : ℕ),
    total_sales = first_group * first_group_purchases +
                  second_group * second_group_purchases +
                  last_group * last_group_purchases ∧
    last_group_purchases = 4 :=
by sorry

end tracy_art_fair_sales_l1338_133846


namespace max_M_value_l1338_133809

theorem max_M_value (x y z u : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hu : u > 0)
  (eq1 : x - 2*y = z - 2*u) (eq2 : 2*y*z = u*x) (h_zy : z ≥ y) :
  ∃ M : ℝ, M > 0 ∧ M ≤ z/y ∧ ∀ N : ℝ, (N > 0 ∧ N ≤ z/y → N ≤ M) ∧ M = 6 + 4*Real.sqrt 2 :=
sorry

end max_M_value_l1338_133809


namespace bd_length_is_twelve_l1338_133875

-- Define the triangle ABC
def triangle_ABC : Type := Unit

-- Define point D
def point_D : Type := Unit

-- Define that B is a right angle
def B_is_right_angle (t : triangle_ABC) : Prop := sorry

-- Define that a circle with diameter BC intersects AC at D
def circle_intersects_AC (t : triangle_ABC) (d : point_D) : Prop := sorry

-- Define the area of triangle ABC
def area_ABC (t : triangle_ABC) : ℝ := 120

-- Define the length of AC
def length_AC (t : triangle_ABC) : ℝ := 20

-- Define the length of BD
def length_BD (t : triangle_ABC) (d : point_D) : ℝ := sorry

-- Theorem statement
theorem bd_length_is_twelve (t : triangle_ABC) (d : point_D) :
  B_is_right_angle t →
  circle_intersects_AC t d →
  length_BD t d = 12 :=
sorry

end bd_length_is_twelve_l1338_133875


namespace consecutive_even_numbers_square_sum_l1338_133804

theorem consecutive_even_numbers_square_sum (a b c d : ℕ) : 
  (∃ x : ℕ, a = 2*x ∧ b = 2*x + 2 ∧ c = 2*x + 4 ∧ d = 2*x + 6) →  -- Consecutive even numbers
  a + b + c + d = 36 →                                           -- Sum is 36
  a^2 + b^2 + c^2 + d^2 = 344 :=                                 -- Sum of squares is 344
by sorry

end consecutive_even_numbers_square_sum_l1338_133804


namespace bicycle_wheels_l1338_133842

theorem bicycle_wheels (num_bicycles : ℕ) (num_tricycles : ℕ) (total_wheels : ℕ) (tricycle_wheels : ℕ) :
  num_bicycles = 6 →
  num_tricycles = 15 →
  total_wheels = 57 →
  tricycle_wheels = 3 →
  ∃ (bicycle_wheels : ℕ), 
    bicycle_wheels = 2 ∧ 
    num_bicycles * bicycle_wheels + num_tricycles * tricycle_wheels = total_wheels :=
by
  sorry

end bicycle_wheels_l1338_133842


namespace clay_cost_calculation_l1338_133856

/-- The price of clay in won per gram -/
def clay_price : ℝ := 17.25

/-- The weight of the first clay piece in grams -/
def clay_weight_1 : ℝ := 1000

/-- The weight of the second clay piece in grams -/
def clay_weight_2 : ℝ := 10

/-- The total cost of clay for Seungjun -/
def total_cost : ℝ := clay_price * (clay_weight_1 + clay_weight_2)

theorem clay_cost_calculation :
  total_cost = 17422.5 := by sorry

end clay_cost_calculation_l1338_133856


namespace reciprocal_sum_theorem_l1338_133803

theorem reciprocal_sum_theorem (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x + y = 5 * x * y) : 1 / x + 1 / y = 5 := by
  sorry

end reciprocal_sum_theorem_l1338_133803


namespace square_area_error_l1338_133872

theorem square_area_error (s : ℝ) (h : s > 0) : 
  let measured_side := s * (1 + 0.02)
  let actual_area := s^2
  let calculated_area := measured_side^2
  (calculated_area - actual_area) / actual_area * 100 = 4.04 := by
sorry

end square_area_error_l1338_133872


namespace andrews_age_l1338_133889

/-- Andrew's age problem -/
theorem andrews_age :
  ∀ (a g : ℚ),
  g = 10 * a →
  g - a = 60 →
  a = 20 / 3 := by
sorry

end andrews_age_l1338_133889


namespace tank_base_diameter_calculation_l1338_133826

/-- The volume of a cylindrical tank in cubic meters. -/
def tank_volume : ℝ := 1848

/-- The depth of the cylindrical tank in meters. -/
def tank_depth : ℝ := 12.00482999321725

/-- The diameter of the base of the cylindrical tank in meters. -/
def tank_base_diameter : ℝ := 24.838

/-- Theorem stating that the diameter of the base of a cylindrical tank with given volume and depth is approximately equal to the calculated value. -/
theorem tank_base_diameter_calculation :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ 
  |2 * Real.sqrt (tank_volume / (Real.pi * tank_depth)) - tank_base_diameter| < ε :=
sorry

end tank_base_diameter_calculation_l1338_133826


namespace garden_perimeter_is_72_l1338_133855

/-- A rectangular garden with specific properties -/
structure Garden where
  /-- The shorter side of the garden -/
  short_side : ℝ
  /-- The longer side of the garden -/
  long_side : ℝ
  /-- The diagonal of the garden is 34 meters -/
  diagonal_eq : short_side ^ 2 + long_side ^ 2 = 34 ^ 2
  /-- The area of the garden is 240 square meters -/
  area_eq : short_side * long_side = 240
  /-- The longer side is three times the shorter side -/
  side_ratio : long_side = 3 * short_side

/-- The perimeter of a rectangular garden -/
def perimeter (g : Garden) : ℝ :=
  2 * (g.short_side + g.long_side)

/-- Theorem stating that the perimeter of the garden is 72 meters -/
theorem garden_perimeter_is_72 (g : Garden) : perimeter g = 72 := by
  sorry

end garden_perimeter_is_72_l1338_133855


namespace exists_n_plus_S_n_eq_1980_l1338_133831

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- Theorem stating the existence of n such that n + S(n) = 1980 -/
theorem exists_n_plus_S_n_eq_1980 : ∃ n : ℕ, n + S n = 1980 := by sorry

end exists_n_plus_S_n_eq_1980_l1338_133831


namespace pairing_probability_l1338_133807

/-- The probability of one student being paired with another specific student
    in a class where some students are absent. -/
theorem pairing_probability
  (total_students : ℕ)
  (absent_students : ℕ)
  (h1 : total_students = 40)
  (h2 : absent_students = 5)
  (h3 : absent_students < total_students) :
  (1 : ℚ) / (total_students - absent_students - 1) = 1 / 34 :=
by sorry

end pairing_probability_l1338_133807


namespace total_students_present_l1338_133867

/-- Calculates the total number of students present across four kindergarten sessions -/
theorem total_students_present
  (morning_registered : ℕ) (morning_absent : ℕ)
  (early_afternoon_registered : ℕ) (early_afternoon_absent : ℕ)
  (late_afternoon_registered : ℕ) (late_afternoon_absent : ℕ)
  (evening_registered : ℕ) (evening_absent : ℕ)
  (h1 : morning_registered = 25) (h2 : morning_absent = 3)
  (h3 : early_afternoon_registered = 24) (h4 : early_afternoon_absent = 4)
  (h5 : late_afternoon_registered = 30) (h6 : late_afternoon_absent = 5)
  (h7 : evening_registered = 35) (h8 : evening_absent = 7) :
  (morning_registered - morning_absent) +
  (early_afternoon_registered - early_afternoon_absent) +
  (late_afternoon_registered - late_afternoon_absent) +
  (evening_registered - evening_absent) = 95 :=
by sorry

end total_students_present_l1338_133867
