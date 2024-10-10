import Mathlib

namespace missy_capacity_l1589_158908

/-- The number of claims each agent can handle -/
structure AgentCapacity where
  jan : ℕ
  john : ℕ
  missy : ℕ

/-- Calculate the capacity of insurance agents based on given conditions -/
def calculate_capacity : AgentCapacity :=
  let jan_capacity := 20
  let john_capacity := jan_capacity + (jan_capacity * 30 / 100)
  let missy_capacity := john_capacity + 15
  { jan := jan_capacity,
    john := john_capacity,
    missy := missy_capacity }

/-- Theorem stating that Missy can handle 41 claims -/
theorem missy_capacity : (calculate_capacity).missy = 41 := by
  sorry

end missy_capacity_l1589_158908


namespace power_of_three_decomposition_l1589_158930

theorem power_of_three_decomposition : 3^25 = 27^7 * 81 := by
  sorry

end power_of_three_decomposition_l1589_158930


namespace product_of_five_terms_l1589_158985

/-- A line passing through the origin with normal vector (3,1) -/
def line_l (x y : ℝ) : Prop := 3 * x + y = 0

/-- Sequence a_n where (a_{n+1}, a_n) lies on the line for all positive integers n -/
def sequence_property (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, line_l (a (n + 1)) (a n)

theorem product_of_five_terms (a : ℕ → ℝ) :
  sequence_property a → a 2 = 6 → a 1 * a 2 * a 3 * a 4 * a 5 = -32 := by
  sorry

end product_of_five_terms_l1589_158985


namespace at_least_one_not_less_than_two_l1589_158996

theorem at_least_one_not_less_than_two 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_three : a + b + c = 3) : 
  (a + 1/b ≥ 2) ∨ (b + 1/c ≥ 2) ∨ (c + 1/a ≥ 2) := by
  sorry

end at_least_one_not_less_than_two_l1589_158996


namespace veggie_patty_percentage_l1589_158909

/-- Proves that the percentage of a veggie patty that is not made up of spices and additives is 70% -/
theorem veggie_patty_percentage (total_weight spice_weight : ℝ) 
  (h1 : total_weight = 150)
  (h2 : spice_weight = 45) :
  (total_weight - spice_weight) / total_weight * 100 = 70 := by
  sorry

end veggie_patty_percentage_l1589_158909


namespace total_swordfish_caught_l1589_158918

def fishing_trips : ℕ := 5

def shelly_catch : ℕ := 5 - 2

def sam_catch : ℕ := shelly_catch - 1

theorem total_swordfish_caught : shelly_catch * fishing_trips + sam_catch * fishing_trips = 25 := by
  sorry

end total_swordfish_caught_l1589_158918


namespace digital_earth_not_equal_gis_l1589_158991

-- Define the concept of Digital Earth
def DigitalEarth : Type := Unit

-- Define Geographic Information Technology
def GeographicInformationTechnology : Type := Unit

-- Define other related technologies
def RemoteSensing : Type := Unit
def GPS : Type := Unit
def VirtualTechnology : Type := Unit
def NetworkTechnology : Type := Unit

-- Define the correct properties of Digital Earth
axiom digital_earth_properties : 
  DigitalEarth → 
  (GeographicInformationTechnology × VirtualTechnology × NetworkTechnology)

-- Define the incorrect statement
def incorrect_statement : Prop :=
  DigitalEarth = GeographicInformationTechnology

-- Theorem to prove
theorem digital_earth_not_equal_gis : ¬incorrect_statement :=
sorry

end digital_earth_not_equal_gis_l1589_158991


namespace earnings_difference_theorem_l1589_158998

/-- Represents the investment and return ratios for three investors -/
structure InvestmentData where
  inv_ratio_a : ℕ
  inv_ratio_b : ℕ
  inv_ratio_c : ℕ
  ret_ratio_a : ℕ
  ret_ratio_b : ℕ
  ret_ratio_c : ℕ

/-- Calculates the earnings difference between investors b and a -/
def earnings_difference (data : InvestmentData) (total_earnings : ℕ) : ℕ :=
  let total_ratio := data.inv_ratio_a * data.ret_ratio_a + 
                     data.inv_ratio_b * data.ret_ratio_b + 
                     data.inv_ratio_c * data.ret_ratio_c
  let unit_earning := total_earnings / total_ratio
  (data.inv_ratio_b * data.ret_ratio_b - data.inv_ratio_a * data.ret_ratio_a) * unit_earning

/-- Theorem: Given the investment ratios 3:4:5, return ratios 6:5:4, and total earnings 10150,
    the earnings difference between b and a is 350 -/
theorem earnings_difference_theorem : 
  let data : InvestmentData := {
    inv_ratio_a := 3, inv_ratio_b := 4, inv_ratio_c := 5,
    ret_ratio_a := 6, ret_ratio_b := 5, ret_ratio_c := 4
  }
  earnings_difference data 10150 = 350 := by
  sorry


end earnings_difference_theorem_l1589_158998


namespace arithmetic_equalities_l1589_158979

theorem arithmetic_equalities : 
  (-(2^3) / 8 - 1/4 * (-2)^2 = -2) ∧ 
  ((-1/12 - 1/16 + 3/4 - 1/6) * (-48) = -21) := by
  sorry

end arithmetic_equalities_l1589_158979


namespace abab_baba_divisible_by_three_l1589_158911

theorem abab_baba_divisible_by_three (A B : ℕ) :
  A ≠ B →
  A ∈ Finset.range 10 →
  B ∈ Finset.range 10 →
  A ≠ 0 →
  B ≠ 0 →
  ∃ k : ℤ, (1010 * A + 101 * B) - (101 * A + 1010 * B) = 3 * k :=
by sorry

end abab_baba_divisible_by_three_l1589_158911


namespace general_term_formula_l1589_158952

-- Define the sequence
def a (n : ℕ) : ℚ :=
  if n = 1 then 3/2
  else if n = 2 then 8/3
  else if n = 3 then 15/4
  else if n = 4 then 24/5
  else if n = 5 then 35/6
  else if n = 6 then 48/7
  else (n^2 + 2*n) / (n + 1)

-- State the theorem
theorem general_term_formula (n : ℕ) (h : n > 0) :
  a n = (n^2 + 2*n) / (n + 1) := by
  sorry

end general_term_formula_l1589_158952


namespace line_relationships_l1589_158949

-- Define a type for lines in a plane
structure Line2D where
  -- You might represent a line by its slope and y-intercept, or by two points, etc.
  -- For this abstract representation, we'll leave the internal structure unspecified

-- Define a type for planes
structure Plane where
  -- Again, we'll leave the internal structure unspecified for this abstract representation

-- Define what it means for two lines to be non-overlapping
def non_overlapping (l1 l2 : Line2D) : Prop :=
  l1 ≠ l2

-- Define what it means for two lines to be in the same plane
def same_plane (p : Plane) (l1 l2 : Line2D) : Prop :=
  -- This would typically involve some geometric condition
  True  -- placeholder

-- Define parallel relationship
def parallel (l1 l2 : Line2D) : Prop :=
  -- This would typically involve some geometric condition
  sorry

-- Define intersecting relationship
def intersecting (l1 l2 : Line2D) : Prop :=
  -- This would typically involve some geometric condition
  sorry

-- The main theorem
theorem line_relationships (p : Plane) (l1 l2 : Line2D) 
  (h1 : non_overlapping l1 l2) (h2 : same_plane p l1 l2) :
  parallel l1 l2 ∨ intersecting l1 l2 := by
  sorry

end line_relationships_l1589_158949


namespace rounded_number_accuracy_l1589_158932

/-- 
Given an approximate number obtained by rounding, represented as 6.18 × 10^4,
prove that it is accurate to the hundred place.
-/
theorem rounded_number_accuracy : 
  let rounded_number : ℝ := 6.18 * 10^4
  ∃ (exact_number : ℝ), 
    (abs (exact_number - rounded_number) ≤ 50) ∧ 
    (∀ (place : ℕ), place > 2 → 
      ∃ (n : ℤ), rounded_number = (n * 10^place : ℝ)) :=
by sorry

end rounded_number_accuracy_l1589_158932


namespace sequence_formula_l1589_158923

/-- For a sequence {a_n} where a_1 = 1 and a_{n+1} = 2^n * a_n for n ≥ 1,
    a_n = 2^(n*(n-1)/2) for all n ≥ 1 -/
theorem sequence_formula (a : ℕ → ℕ) (h1 : a 1 = 1) 
    (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 1) = 2^n * a n) :
  ∀ n : ℕ, n ≥ 1 → a n = 2^(n*(n-1)/2) := by
  sorry

end sequence_formula_l1589_158923


namespace magnitude_of_z_l1589_158997

theorem magnitude_of_z (i : ℂ) (h : i^2 = -1) : Complex.abs ((1 + i) / i) = Real.sqrt 2 := by
  sorry

end magnitude_of_z_l1589_158997


namespace binomial_sum_of_squares_l1589_158910

theorem binomial_sum_of_squares (a : ℝ) : 
  3 * a^4 + 1 = (a^2 + a)^2 + (a^2 - a)^2 + (a^2 - 1)^2 := by
  sorry

end binomial_sum_of_squares_l1589_158910


namespace lawrence_county_kids_l1589_158987

theorem lawrence_county_kids (home_percentage : Real) (kids_at_home : ℕ) : 
  home_percentage = 0.607 →
  kids_at_home = 907611 →
  ∃ total_kids : ℕ, total_kids = (kids_at_home : Real) / home_percentage := by
    sorry

end lawrence_county_kids_l1589_158987


namespace different_color_probability_l1589_158938

def totalChips : ℕ := 18

def blueChips : ℕ := 4
def greenChips : ℕ := 5
def redChips : ℕ := 6
def yellowChips : ℕ := 3

def probBlue : ℚ := blueChips / totalChips
def probGreen : ℚ := greenChips / totalChips
def probRed : ℚ := redChips / totalChips
def probYellow : ℚ := yellowChips / totalChips

theorem different_color_probability : 
  (probBlue * probGreen * probRed + 
   probBlue * probGreen * probYellow + 
   probBlue * probRed * probYellow + 
   probGreen * probRed * probYellow) * 6 = 141 / 162 := by sorry

end different_color_probability_l1589_158938


namespace win_sector_area_l1589_158961

theorem win_sector_area (r : ℝ) (p : ℝ) (h1 : r = 8) (h2 : p = 3/8) :
  p * (π * r^2) = 24 * π := by
sorry

end win_sector_area_l1589_158961


namespace inequality_system_solution_set_l1589_158970

theorem inequality_system_solution_set :
  let S := {x : ℝ | 3 * x - 1 ≥ x + 1 ∧ x + 4 > 4 * x - 2}
  S = {x : ℝ | 1 ≤ x ∧ x < 2} := by
  sorry

end inequality_system_solution_set_l1589_158970


namespace intersection_of_M_and_N_l1589_158993

def M : Set ℝ := {x | Real.sqrt x < 2}
def N : Set ℝ := {x | 3 * x ≥ 1}

theorem intersection_of_M_and_N :
  M ∩ N = {x | 1/3 ≤ x ∧ x < 4} := by sorry

end intersection_of_M_and_N_l1589_158993


namespace auction_bids_per_person_l1589_158914

theorem auction_bids_per_person 
  (starting_price : ℕ) 
  (final_price : ℕ) 
  (price_increase : ℕ) 
  (num_bidders : ℕ) 
  (h1 : starting_price = 15)
  (h2 : final_price = 65)
  (h3 : price_increase = 5)
  (h4 : num_bidders = 2) :
  (final_price - starting_price) / price_increase / num_bidders = 5 :=
by sorry

end auction_bids_per_person_l1589_158914


namespace school_arrival_time_l1589_158948

/-- Calculates how early (in minutes) a boy arrives at school on the second day given the following conditions:
  * The distance between home and school is 2.5 km
  * On the first day, he travels at 5 km/hr and arrives 5 minutes late
  * On the second day, he travels at 10 km/hr and arrives early
-/
theorem school_arrival_time (distance : ℝ) (speed1 speed2 : ℝ) (late_time : ℝ) : 
  distance = 2.5 ∧ 
  speed1 = 5 ∧ 
  speed2 = 10 ∧ 
  late_time = 5 → 
  (distance / speed1 * 60 - late_time) - (distance / speed2 * 60) = 10 := by
  sorry

#check school_arrival_time

end school_arrival_time_l1589_158948


namespace imaginary_part_of_i_squared_times_one_plus_i_l1589_158928

/-- The imaginary part of i²(1+i) is -1 -/
theorem imaginary_part_of_i_squared_times_one_plus_i :
  Complex.im (Complex.I^2 * (1 + Complex.I)) = -1 := by
  sorry

end imaginary_part_of_i_squared_times_one_plus_i_l1589_158928


namespace percentage_problem_l1589_158912

theorem percentage_problem (x : ℝ) (h : 0.2 * x = 100) : 1.2 * x = 600 := by
  sorry

end percentage_problem_l1589_158912


namespace salesman_commission_problem_l1589_158944

/-- A problem about a salesman's commission schemes -/
theorem salesman_commission_problem 
  (old_commission_rate : ℝ)
  (fixed_salary : ℝ)
  (sales_threshold : ℝ)
  (total_sales : ℝ)
  (remuneration_difference : ℝ)
  (h1 : old_commission_rate = 0.05)
  (h2 : fixed_salary = 1000)
  (h3 : sales_threshold = 4000)
  (h4 : total_sales = 12000)
  (h5 : remuneration_difference = 600) :
  ∃ new_commission_rate : ℝ,
    new_commission_rate * (total_sales - sales_threshold) + fixed_salary = 
    old_commission_rate * total_sales + remuneration_difference ∧
    new_commission_rate = 0.025 := by
  sorry

end salesman_commission_problem_l1589_158944


namespace book_pages_proof_l1589_158982

/-- Proves that a book has 500 pages given specific writing and damage conditions -/
theorem book_pages_proof (total_pages : ℕ) : 
  (150 : ℕ) < total_pages →
  (0.8 * 0.7 * (total_pages - 150 : ℕ) : ℝ) = 196 →
  total_pages = 500 := by
sorry

end book_pages_proof_l1589_158982


namespace marble_217_is_red_l1589_158953

/-- Represents the color of a marble -/
inductive MarbleColor
| Red
| Blue
| Green

/-- Returns the color of the nth marble in the sequence -/
def marbleColor (n : ℕ) : MarbleColor :=
  let cycleLength := 15
  let position := n % cycleLength
  if position ≤ 6 then MarbleColor.Red
  else if position ≤ 11 then MarbleColor.Blue
  else MarbleColor.Green

/-- Theorem stating that the 217th marble is red -/
theorem marble_217_is_red : marbleColor 217 = MarbleColor.Red := by
  sorry


end marble_217_is_red_l1589_158953


namespace math_competition_score_l1589_158900

theorem math_competition_score (x : ℕ) : 
  let total_problems := 8 * x + x
  let missed_problems := 2 * x
  let bonus_problems := x
  let standard_points := (total_problems - missed_problems - bonus_problems)
  let bonus_points := 2 * bonus_problems
  let total_available_points := total_problems + bonus_problems
  let scored_points := standard_points + bonus_points
  (scored_points : ℚ) / total_available_points = 9 / 10 :=
by sorry

end math_competition_score_l1589_158900


namespace monthly_income_calculation_l1589_158940

/-- Calculates the monthly income given the percentage saved and the amount saved -/
def calculate_income (percent_saved : ℚ) (amount_saved : ℚ) : ℚ :=
  amount_saved / percent_saved

/-- The percentage of income spent on various categories -/
def total_expenses : ℚ := 35 + 18 + 6 + 11 + 12 + 5 + 7

/-- The percentage of income saved -/
def percent_saved : ℚ := 100 - total_expenses

/-- The amount saved in Rupees -/
def amount_saved : ℚ := 12500

theorem monthly_income_calculation :
  calculate_income percent_saved amount_saved = 208333.33 := by
  sorry

end monthly_income_calculation_l1589_158940


namespace max_red_points_l1589_158994

/-- Represents a point on the circle -/
structure Point where
  color : Bool  -- True for red, False for blue
  connections : Nat

/-- Represents the circle with its points -/
structure Circle where
  points : Finset Point
  total_points : Nat
  red_points : Nat
  blue_points : Nat
  valid_connections : Bool

/-- The main theorem statement -/
theorem max_red_points (c : Circle) : 
  c.total_points = 25 ∧ 
  c.red_points + c.blue_points = c.total_points ∧
  c.valid_connections ∧
  (∀ p q : Point, p ∈ c.points → q ∈ c.points → 
    p.color = true → q.color = true → p ≠ q → p.connections ≠ q.connections) →
  c.red_points ≤ 13 :=
sorry

end max_red_points_l1589_158994


namespace badminton_medals_count_l1589_158906

theorem badminton_medals_count (total_medals : ℕ) (track_medals : ℕ) : 
  total_medals = 20 →
  track_medals = 5 →
  total_medals = track_medals + 2 * track_medals + (total_medals - track_medals - 2 * track_medals) →
  (total_medals - track_medals - 2 * track_medals) = 5 :=
by
  sorry

end badminton_medals_count_l1589_158906


namespace adjusted_work_schedule_earnings_l1589_158954

/-- Proves that the adjusted work schedule results in the same total earnings --/
theorem adjusted_work_schedule_earnings (initial_hours_per_week : ℝ) 
  (initial_weeks : ℕ) (missed_weeks : ℕ) (total_earnings : ℝ) 
  (adjusted_hours_per_week : ℝ) :
  initial_hours_per_week = 25 →
  initial_weeks = 15 →
  missed_weeks = 3 →
  total_earnings = 3750 →
  adjusted_hours_per_week = 31.25 →
  (initial_weeks - missed_weeks : ℝ) * adjusted_hours_per_week = initial_weeks * initial_hours_per_week :=
by sorry

end adjusted_work_schedule_earnings_l1589_158954


namespace multiplication_formula_l1589_158917

theorem multiplication_formula (x y z : ℝ) :
  (2*x + y + z) * (2*x - y - z) = 4*x^2 - y^2 - 2*y*z - z^2 := by
  sorry

end multiplication_formula_l1589_158917


namespace range_of_x_l1589_158935

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the theorem
theorem range_of_x (h1 : ∀ x ∈ [-1, 1], Monotone f) 
  (h2 : ∀ x, f (x - 1) < f (1 - 3*x)) :
  ∃ S : Set ℝ, S = {x | 0 ≤ x ∧ x < 1/2} ∧ 
  (∀ x, x ∈ S ↔ (x - 1 ∈ [-1, 1] ∧ 1 - 3*x ∈ [-1, 1] ∧ f (x - 1) < f (1 - 3*x))) :=
sorry

end range_of_x_l1589_158935


namespace sin_two_a_value_l1589_158999

theorem sin_two_a_value (a : ℝ) (h : Real.sin a - Real.cos a = 4/3) : 
  Real.sin (2 * a) = -7/9 := by
  sorry

end sin_two_a_value_l1589_158999


namespace hyperbola_asymptotes_l1589_158951

/-- The equation of a hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop := x^2 / 4 - y^2 / 9 = 1

/-- The equation of the asymptotes -/
def asymptote_equation (x y : ℝ) : Prop := y = (3/2) * x ∨ y = -(3/2) * x

/-- Theorem: The asymptotes of the given hyperbola are y = ±(3/2)x -/
theorem hyperbola_asymptotes :
  ∀ x y : ℝ, hyperbola_equation x y → asymptote_equation x y :=
sorry

end hyperbola_asymptotes_l1589_158951


namespace baseball_earnings_l1589_158920

/-- The total earnings from two baseball games -/
def total_earnings (saturday_earnings wednesday_earnings : ℚ) : ℚ :=
  saturday_earnings + wednesday_earnings

/-- Theorem stating the total earnings from two baseball games -/
theorem baseball_earnings : 
  ∃ (saturday_earnings wednesday_earnings : ℚ),
    saturday_earnings = 2662.50 ∧
    wednesday_earnings = saturday_earnings - 142.50 ∧
    total_earnings saturday_earnings wednesday_earnings = 5182.50 :=
by sorry

end baseball_earnings_l1589_158920


namespace geometric_sequence_product_l1589_158924

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a →
  a 6 = 3 →
  a 3 * a 4 * a 5 * a 6 * a 7 * a 8 * a 9 = 2187 := by
  sorry

end geometric_sequence_product_l1589_158924


namespace complement_of_union_is_four_l1589_158927

open Set

def U : Finset ℕ := {0, 1, 2, 3, 4}
def A : Finset ℕ := {0, 1, 3}
def B : Finset ℕ := {2, 3}

theorem complement_of_union_is_four :
  (U \ (A ∪ B)) = {4} := by sorry

end complement_of_union_is_four_l1589_158927


namespace fifteen_factorial_base_twelve_zeroes_l1589_158980

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem fifteen_factorial_base_twelve_zeroes :
  ∃ k : ℕ, k = 5 ∧ 12^k ∣ factorial 15 ∧ ¬(12^(k+1) ∣ factorial 15) :=
by sorry

end fifteen_factorial_base_twelve_zeroes_l1589_158980


namespace largest_valid_number_l1589_158971

def is_valid_number (n : ℕ) : Prop :=
  (Nat.digits 10 n).length = 85 ∧
  (Nat.digits 10 n).sum = (Nat.digits 10 n).prod

def target_number : ℕ := 8322 * 10^81 + (10^81 - 1)

theorem largest_valid_number :
  is_valid_number target_number ∧
  ∀ m : ℕ, is_valid_number m → m ≤ target_number := by
  sorry

end largest_valid_number_l1589_158971


namespace largest_x_for_equation_l1589_158956

theorem largest_x_for_equation : 
  (∀ x y : ℤ, x > 3 → x^2 - x*y - 2*y^2 ≠ 9) ∧ 
  (∃ y : ℤ, 3^2 - 3*y - 2*y^2 = 9) :=
sorry

end largest_x_for_equation_l1589_158956


namespace spinner_probability_l1589_158934

theorem spinner_probability : ∀ (p_A p_B p_C p_D p_E : ℝ),
  p_A = 1/5 →
  p_B = 1/5 →
  p_C = p_D →
  p_E = 2 * p_C →
  p_A + p_B + p_C + p_D + p_E = 1 →
  p_C = 3/20 :=
by
  sorry

end spinner_probability_l1589_158934


namespace intersection_distance_l1589_158959

-- Define the line C₁
def C₁ (x y : ℝ) : Prop := y - 2*x + 1 = 0

-- Define the circle C₂
def C₂ (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 5

-- Theorem statement
theorem intersection_distance :
  ∃ (A B : ℝ × ℝ),
    C₁ A.1 A.2 ∧ C₂ A.1 A.2 ∧
    C₁ B.1 B.2 ∧ C₂ B.1 B.2 ∧
    A ≠ B ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8 * Real.sqrt 5 / 5 :=
sorry

end intersection_distance_l1589_158959


namespace secret_room_number_l1589_158966

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

def is_odd (n : ℕ) : Prop := n % 2 = 1

def sum_of_digits (n : ℕ) : ℕ := (n / 10) + (n % 10)

def has_digit_8 (n : ℕ) : Prop := (n / 10 = 8) ∨ (n % 10 = 8)

def exactly_three_true (p q r s : Prop) : Prop :=
  (p ∧ q ∧ r ∧ ¬s) ∨ (p ∧ q ∧ ¬r ∧ s) ∨ (p ∧ ¬q ∧ r ∧ s) ∨ (¬p ∧ q ∧ r ∧ s)

theorem secret_room_number (n : ℕ) 
  (h1 : is_two_digit n)
  (h2 : exactly_three_true (divisible_by_4 n) (is_odd n) (sum_of_digits n = 12) (has_digit_8 n)) :
  n % 10 = 4 := by
sorry

end secret_room_number_l1589_158966


namespace projection_matrix_values_l1589_158933

/-- A projection matrix Q satisfies Q^2 = Q -/
def is_projection_matrix (Q : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  Q * Q = Q

/-- The given matrix form -/
def projection_matrix (x y : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![x, 12/25],
    ![y, 13/25]]

/-- Theorem stating that the projection matrix has x = 0 and y = 12/25 -/
theorem projection_matrix_values :
  ∀ x y : ℚ, is_projection_matrix (projection_matrix x y) → x = 0 ∧ y = 12/25 := by
  sorry


end projection_matrix_values_l1589_158933


namespace sum_abc_equals_33_l1589_158977

theorem sum_abc_equals_33 
  (a b c N : ℕ+) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ b ≠ c)
  (h_eq1 : N = 5*a + 3*b + 5*c)
  (h_eq2 : N = 4*a + 5*b + 4*c)
  (h_range : 131 < N ∧ N < 150) :
  a + b + c = 33 := by
sorry

end sum_abc_equals_33_l1589_158977


namespace shelbys_drive_l1589_158945

/-- Represents the weather conditions during Shelby's drive --/
inductive Weather
  | Sunny
  | Rainy
  | Foggy

/-- Shelby's driving scenario --/
structure DrivingScenario where
  speed : Weather → ℝ
  total_distance : ℝ
  total_time : ℝ
  time_in_weather : Weather → ℝ

/-- The theorem statement for Shelby's driving problem --/
theorem shelbys_drive (scenario : DrivingScenario) : 
  scenario.speed Weather.Sunny = 35 ∧ 
  scenario.speed Weather.Rainy = 25 ∧ 
  scenario.speed Weather.Foggy = 15 ∧ 
  scenario.total_distance = 19.5 ∧ 
  scenario.total_time = 45 ∧ 
  (scenario.time_in_weather Weather.Sunny + 
   scenario.time_in_weather Weather.Rainy + 
   scenario.time_in_weather Weather.Foggy = scenario.total_time) ∧
  (scenario.speed Weather.Sunny * scenario.time_in_weather Weather.Sunny / 60 +
   scenario.speed Weather.Rainy * scenario.time_in_weather Weather.Rainy / 60 +
   scenario.speed Weather.Foggy * scenario.time_in_weather Weather.Foggy / 60 = 
   scenario.total_distance) →
  scenario.time_in_weather Weather.Foggy = 10.25 := by
  sorry

end shelbys_drive_l1589_158945


namespace stratified_selection_count_l1589_158947

def female_students : ℕ := 8
def male_students : ℕ := 4
def total_selected : ℕ := 3

theorem stratified_selection_count :
  (Nat.choose female_students 2 * Nat.choose male_students 1) +
  (Nat.choose female_students 1 * Nat.choose male_students 2) = 112 :=
by sorry

end stratified_selection_count_l1589_158947


namespace total_cats_is_thirteen_l1589_158919

/-- The number of cats owned by Jamie, Gordon, and Hawkeye --/
def total_cats : ℕ :=
  let jamie_persians : ℕ := 4
  let jamie_maine_coons : ℕ := 2
  let gordon_persians : ℕ := jamie_persians / 2
  let gordon_maine_coons : ℕ := jamie_maine_coons + 1
  let hawkeye_persians : ℕ := 0
  let hawkeye_maine_coons : ℕ := gordon_maine_coons - 1
  jamie_persians + jamie_maine_coons +
  gordon_persians + gordon_maine_coons +
  hawkeye_persians + hawkeye_maine_coons

theorem total_cats_is_thirteen : total_cats = 13 := by
  sorry

end total_cats_is_thirteen_l1589_158919


namespace f_two_plus_f_five_l1589_158958

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_one : f 1 = 4

axiom f_z (z : ℝ) : z ≠ 1 → f z = 3 * z + 6

axiom f_sum (x y : ℝ) : ∃ (a b : ℝ), f (x + y) = f x + f y + a * x * y + b

theorem f_two_plus_f_five : f 2 + f 5 = 33 := by sorry

end f_two_plus_f_five_l1589_158958


namespace vector_magnitude_problem_l1589_158972

def a : ℝ × ℝ := (1, 2)
def b (m : ℝ) : ℝ × ℝ := (-2, m)

theorem vector_magnitude_problem (m : ℝ) :
  (‖a + b m‖ = ‖a - b m‖) → ‖a + 2 • (b m)‖ = 5 :=
by sorry

end vector_magnitude_problem_l1589_158972


namespace max_distance_le_150cm_l1589_158907

/-- Represents the extended table with two semicircles and a rectangular section -/
structure ExtendedTable where
  semicircle_diameter : ℝ
  rectangle_length : ℝ
  rectangle_width : ℝ

/-- The maximum distance between any two points on the extended table -/
def max_distance (table : ExtendedTable) : ℝ :=
  sorry

/-- Theorem stating that the maximum distance between any two points on the extended table
    is less than or equal to 150 cm -/
theorem max_distance_le_150cm (table : ExtendedTable)
  (h1 : table.semicircle_diameter = 1)
  (h2 : table.rectangle_length = 1)
  (h3 : table.rectangle_width = 0.5) :
  max_distance table ≤ 1.5 := by
  sorry

end max_distance_le_150cm_l1589_158907


namespace prime_sum_squares_l1589_158975

theorem prime_sum_squares (p q : ℕ) : 
  Prime p → Prime q → 
  ∃ (x y : ℕ), x^2 = p + q ∧ y^2 = p + 7*q → 
  p = 2 := by
sorry

end prime_sum_squares_l1589_158975


namespace stacy_paper_completion_time_l1589_158963

/-- The number of days Stacy has to complete her history paper -/
def days_to_complete : ℕ := 
  63 / 9

/-- The total number of pages in Stacy's history paper -/
def total_pages : ℕ := 63

/-- The number of pages Stacy has to write per day -/
def pages_per_day : ℕ := 9

/-- Theorem stating that Stacy has 7 days to complete her paper -/
theorem stacy_paper_completion_time : days_to_complete = 7 := by
  sorry

end stacy_paper_completion_time_l1589_158963


namespace range_of_a_l1589_158915

theorem range_of_a (P : Set ℝ) (M : Set ℝ) (a : ℝ) 
  (h1 : P = {x : ℝ | x^2 ≤ 1})
  (h2 : M = {a})
  (h3 : P ∪ M = P) : 
  -1 ≤ a ∧ a ≤ 1 := by
  sorry

end range_of_a_l1589_158915


namespace absolute_value_inequality_solution_set_l1589_158960

theorem absolute_value_inequality_solution_set :
  ∀ x : ℝ, |4 - 3*x| - 5 ≤ 0 ↔ -1/3 ≤ x ∧ x ≤ 3 := by sorry

end absolute_value_inequality_solution_set_l1589_158960


namespace vidyas_age_l1589_158981

theorem vidyas_age (vidya_age : ℕ) (mother_age : ℕ) : 
  mother_age = 3 * vidya_age + 5 →
  mother_age = 44 →
  vidya_age = 13 :=
by
  sorry

end vidyas_age_l1589_158981


namespace range_of_a_l1589_158931

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - 3*a*x + 9 ≥ 0) → a ∈ Set.Icc (-2) 2 := by
  sorry

end range_of_a_l1589_158931


namespace sin_neg_360_degrees_l1589_158929

theorem sin_neg_360_degrees : Real.sin (-(360 * π / 180)) = 0 := by
  sorry

end sin_neg_360_degrees_l1589_158929


namespace right_isosceles_triangle_projection_l1589_158901

/-- Represents a right isosceles triangle -/
structure RightIsoscelesTriangle where
  side : ℝ
  right_angle : Bool
  isosceles : Bool

/-- Represents the projection of a triangle -/
def project (t : RightIsoscelesTriangle) (parallel : Bool) : RightIsoscelesTriangle :=
  if parallel then t else sorry

theorem right_isosceles_triangle_projection
  (t : RightIsoscelesTriangle)
  (h_side : t.side = 6)
  (h_right : t.right_angle = true)
  (h_isosceles : t.isosceles = true)
  (h_parallel : parallel = true) :
  let projected := project t parallel
  projected.side = 6 ∧
  projected.right_angle = true ∧
  projected.isosceles = true ∧
  Real.sqrt (2 * projected.side ^ 2) = 6 * Real.sqrt 2 :=
by sorry

end right_isosceles_triangle_projection_l1589_158901


namespace ferris_wheel_seats_l1589_158969

/-- The Ferris wheel problem -/
theorem ferris_wheel_seats (people_per_seat : ℕ) (total_people : ℕ) (h1 : people_per_seat = 9) (h2 : total_people = 18) :
  total_people / people_per_seat = 2 := by
  sorry

end ferris_wheel_seats_l1589_158969


namespace complex_fraction_simplification_l1589_158921

theorem complex_fraction_simplification (x y z : ℚ) 
  (hx : x = 4)
  (hy : y = 5)
  (hz : z = 2) :
  (1 / z / y) / (1 / x) = 2 / 5 := by
  sorry

end complex_fraction_simplification_l1589_158921


namespace intersection_line_equation_l1589_158913

/-- Definition of line l1 -/
def l1 (x y : ℝ) : Prop := x - y + 3 = 0

/-- Definition of line l2 -/
def l2 (x y : ℝ) : Prop := 2*x + y = 0

/-- Definition of the intersection point of l1 and l2 -/
def intersection_point (x y : ℝ) : Prop := l1 x y ∧ l2 x y

/-- Definition of a line with inclination angle π/3 passing through a point -/
def line_with_inclination (x₀ y₀ x y : ℝ) : Prop :=
  y - y₀ = Real.sqrt 3 * (x - x₀)

/-- The main theorem -/
theorem intersection_line_equation :
  ∃ x₀ y₀ : ℝ, intersection_point x₀ y₀ ∧
  ∀ x y : ℝ, line_with_inclination x₀ y₀ x y ↔ Real.sqrt 3 * x - y + Real.sqrt 3 + 2 = 0 :=
sorry

end intersection_line_equation_l1589_158913


namespace special_function_characterization_l1589_158941

/-- A monotonic and invertible function from ℝ to ℝ satisfying f(x) + f⁻¹(x) = 2x for all x ∈ ℝ -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  Monotone f ∧ Function.Bijective f ∧ ∀ x, f x + (Function.invFun f) x = 2 * x

/-- The theorem stating that any function satisfying SpecialFunction is of the form f(x) = x + c -/
theorem special_function_characterization (f : ℝ → ℝ) (h : SpecialFunction f) :
  ∃ c : ℝ, ∀ x, f x = x + c :=
sorry

end special_function_characterization_l1589_158941


namespace djibo_age_proof_l1589_158989

/-- Djibo's current age -/
def djibo_age : ℕ := 17

/-- Djibo's sister's current age -/
def sister_age : ℕ := 28

/-- Sum of Djibo's and his sister's ages 5 years ago -/
def sum_ages_5_years_ago : ℕ := 35

theorem djibo_age_proof :
  djibo_age = 17 ∧
  sister_age = 28 ∧
  (djibo_age - 5) + (sister_age - 5) = sum_ages_5_years_ago :=
by sorry

end djibo_age_proof_l1589_158989


namespace thirteen_travel_methods_l1589_158939

/-- The number of different methods to travel from Place A to Place B -/
def travel_methods (bus_services train_services ship_services : ℕ) : ℕ :=
  bus_services + train_services + ship_services

/-- Theorem: There are 13 different methods to travel from Place A to Place B -/
theorem thirteen_travel_methods :
  travel_methods 8 3 2 = 13 := by
  sorry

end thirteen_travel_methods_l1589_158939


namespace min_stool_height_l1589_158942

/-- The minimum height of the stool for Alice to reach the ceiling fan switch -/
theorem min_stool_height (ceiling_height : ℝ) (switch_below_ceiling : ℝ) 
  (alice_height : ℝ) (alice_reach : ℝ) (books_height : ℝ) 
  (h1 : ceiling_height = 300) 
  (h2 : switch_below_ceiling = 15)
  (h3 : alice_height = 160)
  (h4 : alice_reach = 50)
  (h5 : books_height = 12) : 
  ∃ (s : ℝ), s ≥ 63 ∧ 
  ∀ (x : ℝ), x < 63 → alice_height + alice_reach + books_height + x < ceiling_height - switch_below_ceiling :=
sorry

end min_stool_height_l1589_158942


namespace hyperbola_equation_l1589_158922

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if its focal length is 10 and the point (1, 2) lies on its asymptote,
    then its equation is x²/5 - y²/20 = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a^2 + b^2 = 25) →
  (b - 2*a = 0) →
  ∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1 ↔ x^2/5 - y^2/20 = 1 :=
by sorry

end hyperbola_equation_l1589_158922


namespace bryans_precious_stones_l1589_158926

theorem bryans_precious_stones (price_per_stone total_amount : ℕ) 
  (h1 : price_per_stone = 1785)
  (h2 : total_amount = 14280) :
  total_amount / price_per_stone = 8 := by
  sorry

end bryans_precious_stones_l1589_158926


namespace express_u_in_terms_of_f_and_g_l1589_158988

/-- Given functions u, f, and g satisfying certain conditions, 
    prove that u can be expressed in terms of f and g. -/
theorem express_u_in_terms_of_f_and_g 
  (u f g : ℝ → ℝ)
  (h1 : ∀ x, u (x + 1) + u (x - 1) = 2 * f x)
  (h2 : ∀ x, u (x + 4) + u (x - 4) = 2 * g x) :
  ∀ x, u x = g (x + 4) - f (x + 7) + f (x + 5) - f (x + 3) + f (x + 1) := by
  sorry

end express_u_in_terms_of_f_and_g_l1589_158988


namespace sin_difference_inequality_l1589_158973

theorem sin_difference_inequality (a b : ℝ) :
  ((0 ≤ a ∧ a < b ∧ b ≤ π / 2) ∨ (π ≤ a ∧ a < b ∧ b ≤ 3 * π / 2)) →
  a - Real.sin a < b - Real.sin b :=
by sorry

end sin_difference_inequality_l1589_158973


namespace common_tangent_l1589_158974

-- Define the circles
def C1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def C2 (x y : ℝ) : Prop := (x-3)^2 + (y-4)^2 = 16

-- Define the line x = -1
def tangent_line (x : ℝ) : Prop := x = -1

-- Theorem statement
theorem common_tangent :
  (∀ x y : ℝ, C1 x y → tangent_line x → (x^2 + y^2 = 1 ∧ x = -1)) ∧
  (∀ x y : ℝ, C2 x y → tangent_line x → ((x-3)^2 + (y-4)^2 = 16 ∧ x = -1)) :=
sorry

end common_tangent_l1589_158974


namespace right_triangle_hypotenuse_l1589_158984

theorem right_triangle_hypotenuse : ∀ (a b c : ℝ), 
  a = 30 → b = 40 → c^2 = a^2 + b^2 → c = 50 := by
  sorry

end right_triangle_hypotenuse_l1589_158984


namespace third_year_students_l1589_158968

theorem third_year_students (total_first_year : ℕ) (total_selected : ℕ) (second_year_selected : ℕ) :
  total_first_year = 720 →
  total_selected = 180 →
  second_year_selected = 40 →
  ∃ (first_year_selected third_year_selected : ℕ),
    first_year_selected = (second_year_selected + third_year_selected) / 2 ∧
    first_year_selected + second_year_selected + third_year_selected = total_selected ∧
    (total_first_year * third_year_selected : ℚ) / first_year_selected = 960 :=
by sorry

end third_year_students_l1589_158968


namespace number_pattern_l1589_158903

theorem number_pattern (A : ℕ) : 10 * A + 9 = A * 9 + (A + 9) := by
  sorry

end number_pattern_l1589_158903


namespace rachel_piggy_bank_l1589_158937

/-- The amount of money Rachel originally had in her piggy bank -/
def original_amount : ℕ := 5

/-- The amount of money Rachel now has in her piggy bank -/
def current_amount : ℕ := 3

/-- The amount of money Rachel took from her piggy bank -/
def amount_taken : ℕ := original_amount - current_amount

theorem rachel_piggy_bank :
  amount_taken = 2 :=
sorry

end rachel_piggy_bank_l1589_158937


namespace stirring_evenly_key_to_representativeness_l1589_158957

/-- Represents a sampling method -/
inductive SamplingMethod
| Lottery
| Other

/-- Represents actions in the lottery method -/
inductive LotteryAction
| MakeTickets
| StirEvenly
| DrawOneByOne
| DrawWithoutReplacement

/-- Represents the property of being representative -/
def IsRepresentative (sample : Set α) : Prop := sorry

/-- The lottery method -/
def lotteryMethod : SamplingMethod := SamplingMethod.Lottery

/-- Function to determine if an action is key to representativeness -/
def isKeyToRepresentativeness (action : LotteryAction) (method : SamplingMethod) : Prop := sorry

/-- Theorem stating that stirring evenly is key to representativeness in the lottery method -/
theorem stirring_evenly_key_to_representativeness :
  isKeyToRepresentativeness LotteryAction.StirEvenly lotteryMethod := by sorry

end stirring_evenly_key_to_representativeness_l1589_158957


namespace wanda_walking_distance_l1589_158950

/-- The distance in miles Wanda walks to school one way -/
def distance_to_school : ℝ := 0.5

/-- The number of round trips Wanda makes per day -/
def round_trips_per_day : ℕ := 2

/-- The number of school days per week -/
def school_days_per_week : ℕ := 5

/-- The number of weeks -/
def num_weeks : ℕ := 4

/-- The total distance Wanda walks after the given number of weeks -/
def total_distance : ℝ :=
  distance_to_school * 2 * round_trips_per_day * school_days_per_week * num_weeks

theorem wanda_walking_distance :
  total_distance = 40 := by sorry

end wanda_walking_distance_l1589_158950


namespace license_plate_difference_l1589_158936

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of digits available -/
def num_digits : ℕ := 10

/-- The number of possible license plates for Alpha state -/
def alpha_plates : ℕ := num_letters^3 * num_digits^4

/-- The number of possible license plates for Beta state -/
def beta_plates : ℕ := num_letters^4 * num_digits^3

/-- The difference in the number of possible license plates between Beta and Alpha -/
def plate_difference : ℕ := beta_plates - alpha_plates

theorem license_plate_difference : plate_difference = 281216000 := by
  sorry

end license_plate_difference_l1589_158936


namespace polynomial_expansion_l1589_158983

theorem polynomial_expansion (x : ℝ) :
  (3*x^2 + 4*x + 8)*(x - 2) - (x - 2)*(x^2 + 5*x - 72) + (4*x - 15)*(x - 2)*(x + 6) =
  6*x^3 - 4*x^2 - 26*x + 20 := by
  sorry

end polynomial_expansion_l1589_158983


namespace multiplication_digits_sum_l1589_158965

theorem multiplication_digits_sum (x y : Nat) : 
  x < 10 → y < 10 → (30 + x) * (10 * y + 4) = 136 → x + y = 7 := by
  sorry

end multiplication_digits_sum_l1589_158965


namespace jellybean_problem_l1589_158995

theorem jellybean_problem (initial_count : ℕ) : 
  (initial_count : ℝ) * (3/4)^3 = 27 → initial_count = 64 :=
by
  sorry

end jellybean_problem_l1589_158995


namespace largest_non_expressible_l1589_158905

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def is_expressible (n : ℕ) : Prop :=
  ∃ (a : ℕ) (b : ℕ), n = 48 * a + b ∧ is_composite b ∧ 0 < b

theorem largest_non_expressible :
  (∀ n > 95, is_expressible n) ∧
  ¬is_expressible 95 :=
sorry

end largest_non_expressible_l1589_158905


namespace remainder_theorem_l1589_158992

/-- The dividend polynomial -/
def P (x : ℝ) : ℝ := x^100 - 2*x^51 + 1

/-- The divisor polynomial -/
def D (x : ℝ) : ℝ := x^2 - 1

/-- The proposed remainder -/
def R (x : ℝ) : ℝ := -2*x + 2

/-- Theorem stating that R is the remainder of P divided by D -/
theorem remainder_theorem : 
  ∃ Q : ℝ → ℝ, ∀ x : ℝ, P x = D x * Q x + R x :=
sorry

end remainder_theorem_l1589_158992


namespace age_problem_l1589_158967

theorem age_problem :
  ∀ (a b c : ℕ),
  a + b + c = 29 →
  a = b →
  c = 11 →
  a = 9 :=
by
  sorry

end age_problem_l1589_158967


namespace parabola_x_intercepts_distance_l1589_158943

-- Define the parabola function
def f (x : ℝ) : ℝ := -x^2 + 2*x + 3

-- Theorem statement
theorem parabola_x_intercepts_distance : 
  ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ x₂ - x₁ = 4 := by
  sorry

end parabola_x_intercepts_distance_l1589_158943


namespace point_in_fourth_quadrant_l1589_158986

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant -/
def fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- Theorem: If a - b > 0 and ab < 0, then the point P(a,b) lies in the fourth quadrant -/
theorem point_in_fourth_quadrant (a b : ℝ) (h1 : a - b > 0) (h2 : a * b < 0) :
  fourth_quadrant (Point.mk a b) := by
  sorry


end point_in_fourth_quadrant_l1589_158986


namespace system_solution_unique_l1589_158925

theorem system_solution_unique :
  ∃! (x y z : ℝ), 
    x^2 - y*z = 1 ∧
    y^2 - x*z = 2 ∧
    z^2 - x*y = 3 ∧
    (x = 5*Real.sqrt 2/6 ∨ x = -5*Real.sqrt 2/6) ∧
    (y = -Real.sqrt 2/6 ∨ y = Real.sqrt 2/6) ∧
    (z = -7*Real.sqrt 2/6 ∨ z = 7*Real.sqrt 2/6) :=
by sorry

end system_solution_unique_l1589_158925


namespace experts_win_probability_l1589_158978

/-- The probability of Experts winning a single round -/
def p : ℝ := 0.6

/-- The probability of Viewers winning a single round -/
def q : ℝ := 1 - p

/-- The current score of Experts -/
def expertsScore : ℕ := 3

/-- The current score of Viewers -/
def viewersScore : ℕ := 4

/-- The number of rounds needed to win the game -/
def winningScore : ℕ := 6

/-- The probability that the Experts will eventually win the game -/
def expertsWinProbability : ℝ := p^4 + 4 * p^3 * q

theorem experts_win_probability : 
  expertsWinProbability = 0.4752 := by sorry

end experts_win_probability_l1589_158978


namespace ellipse_foci_distance_l1589_158946

-- Define the ellipse Γ
def Γ : Set (ℝ × ℝ) := sorry

-- Define points A, B, and C
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry
def C : ℝ × ℝ := sorry

-- State the theorem
theorem ellipse_foci_distance :
  -- AB is the major axis of ellipse Γ
  (∀ p ∈ Γ, (p.1 - A.1)^2 + (p.2 - A.2)^2 ≤ (B.1 - A.1)^2 + (B.2 - A.2)^2) →
  -- Point C is on Γ
  C ∈ Γ →
  -- Angle CBA = π/4
  Real.arccos ((C.1 - B.1) / Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)) = π/4 →
  -- AB = 4
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 4 →
  -- BC = √2
  Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = Real.sqrt 2 →
  -- The distance between the two foci is 4√6/3
  ∃ F₁ F₂ : ℝ × ℝ, F₁ ∈ Γ ∧ F₂ ∈ Γ ∧
    Real.sqrt ((F₂.1 - F₁.1)^2 + (F₂.2 - F₁.2)^2) = 4 * Real.sqrt 6 / 3 :=
by sorry

end ellipse_foci_distance_l1589_158946


namespace square_count_theorem_l1589_158964

/-- Represents a family of parallel lines -/
structure LineFamily :=
  (count : ℕ)

/-- Represents the configuration of two perpendicular families of lines -/
structure LineConfiguration :=
  (family1 : LineFamily)
  (family2 : LineFamily)

/-- Represents the set of intersection points -/
def IntersectionPoints (config : LineConfiguration) : ℕ :=
  config.family1.count * config.family2.count

/-- Counts the number of squares with sides parallel to the coordinate axes -/
def countParallelSquares (config : LineConfiguration) : ℕ :=
  sorry

/-- Counts the number of slanted squares -/
def countSlantedSquares (config : LineConfiguration) : ℕ :=
  sorry

/-- The main theorem -/
theorem square_count_theorem (config : LineConfiguration) 
  (h1 : config.family1.count = 15)
  (h2 : config.family2.count = 11)
  (h3 : IntersectionPoints config = 165) :
  countParallelSquares config + countSlantedSquares config ≥ 1986 :=
sorry

end square_count_theorem_l1589_158964


namespace wednesday_temperature_l1589_158976

/-- Given the high temperatures for three consecutive days (Monday, Tuesday, Wednesday),
    prove that Wednesday's temperature is 12°C. -/
theorem wednesday_temperature
  (monday tuesday wednesday : ℝ)
  (h1 : tuesday = monday + 4)
  (h2 : wednesday = monday - 6)
  (h3 : tuesday = 22) :
  wednesday = 12 := by
  sorry

end wednesday_temperature_l1589_158976


namespace inequality_proof_l1589_158990

theorem inequality_proof (a b c : ℝ) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : c > 0) :
  a + b ≤ 2 * c ∧ 2 * c ≤ 3 * c := by
  sorry

end inequality_proof_l1589_158990


namespace line_intercept_sum_minimum_equality_condition_l1589_158955

theorem line_intercept_sum_minimum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : a + b = a * b) : a + b ≥ 4 := by
  sorry

theorem equality_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : a + b = a * b) : a + b = 4 ↔ a = 2 ∧ b = 2 := by
  sorry

end line_intercept_sum_minimum_equality_condition_l1589_158955


namespace train_crossing_bridge_time_l1589_158904

/-- The time it takes for a train to cross a bridge -/
theorem train_crossing_bridge_time 
  (train_length : ℝ) 
  (bridge_length : ℝ) 
  (train_speed_kmph : ℝ) 
  (h1 : train_length = 130)
  (h2 : bridge_length = 150)
  (h3 : train_speed_kmph = 36) : 
  (train_length + bridge_length) / (train_speed_kmph * (5/18)) = 28 := by
sorry

end train_crossing_bridge_time_l1589_158904


namespace complex_sum_problem_l1589_158962

theorem complex_sum_problem (b d e f : ℝ) : 
  b = 2 →
  e = -5 →
  (2 : ℂ) + b * I + (3 : ℂ) + d * I + e + f * I = (1 : ℂ) - 3 * I →
  d + f = -5 :=
by
  sorry

end complex_sum_problem_l1589_158962


namespace base_eight_4372_equals_2298_l1589_158916

def base_eight_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

theorem base_eight_4372_equals_2298 :
  base_eight_to_decimal [2, 7, 3, 4] = 2298 := by
  sorry

end base_eight_4372_equals_2298_l1589_158916


namespace inequality_proof_l1589_158902

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x + y + z = 1) :
  (x^2 + y^2) / z + (y^2 + z^2) / x + (z^2 + x^2) / y ≥ 2 :=
by sorry

end inequality_proof_l1589_158902
