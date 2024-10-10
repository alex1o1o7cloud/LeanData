import Mathlib

namespace cuboid_volume_doubled_l1209_120911

/-- The volume of a cuboid after doubling its dimensions -/
theorem cuboid_volume_doubled (original_volume : ℝ) : 
  original_volume = 36 → 8 * original_volume = 288 := by
  sorry

end cuboid_volume_doubled_l1209_120911


namespace equation_solution_l1209_120927

theorem equation_solution (z : ℚ) : 
  Real.sqrt (5 - 4 * z) = 10 → z = -95/4 := by sorry

end equation_solution_l1209_120927


namespace equation_solution_l1209_120988

theorem equation_solution : 
  ∃! x : ℚ, (x^2 + 3*x + 5) / (x + 6) = x + 7 ∧ x = -37/10 := by
  sorry

end equation_solution_l1209_120988


namespace stella_toilet_paper_stocking_l1209_120996

/-- Proves that Stella stocks 1 roll per day in each bathroom given the conditions --/
theorem stella_toilet_paper_stocking :
  let num_bathrooms : ℕ := 6
  let days_per_week : ℕ := 7
  let rolls_per_pack : ℕ := 12
  let weeks : ℕ := 4
  let packs_bought : ℕ := 14
  
  let total_rolls : ℕ := packs_bought * rolls_per_pack
  let rolls_per_week : ℕ := total_rolls / weeks
  let rolls_per_day : ℕ := rolls_per_week / days_per_week
  let rolls_per_bathroom_per_day : ℕ := rolls_per_day / num_bathrooms

  rolls_per_bathroom_per_day = 1 :=
by sorry

end stella_toilet_paper_stocking_l1209_120996


namespace probability_of_pair_after_removal_l1209_120971

/-- Represents a deck of cards -/
structure Deck :=
  (cards : Finset (Fin 10 × Fin 4))
  (card_count : cards.card = 40)

/-- Represents the deck after removing a matching pair -/
def RemainingDeck (d : Deck) : Finset (Fin 10 × Fin 4) :=
  d.cards.filter (λ x ↦ x.2 ≠ 3)

/-- The probability of selecting a matching pair from the remaining deck -/
def ProbabilityOfPair (d : Deck) : ℚ :=
  55 / 703

theorem probability_of_pair_after_removal (d : Deck) :
  ProbabilityOfPair d = 55 / 703 :=
sorry

end probability_of_pair_after_removal_l1209_120971


namespace min_value_trig_expression_l1209_120924

theorem min_value_trig_expression (x : Real) (h : 0 < x ∧ x < π / 2) :
  (8 / Real.sin x) + (1 / Real.cos x) ≥ 10 := by
  sorry

end min_value_trig_expression_l1209_120924


namespace vector_addition_and_scalar_multiplication_l1209_120990

/-- Given vectors a and b, prove that c = 2a + 5b has the specified coordinates -/
theorem vector_addition_and_scalar_multiplication (a b : ℝ × ℝ × ℝ) 
  (h1 : a = (3, -4, 5)) 
  (h2 : b = (-1, 0, -2)) : 
  (2 : ℝ) • a + (5 : ℝ) • b = (1, -8, 0) := by
  sorry

#check vector_addition_and_scalar_multiplication

end vector_addition_and_scalar_multiplication_l1209_120990


namespace solution_set_reciprocal_inequality_l1209_120942

theorem solution_set_reciprocal_inequality (x : ℝ) :
  (∃ y ∈ Set.Ioo (0 : ℝ) (1/3 : ℝ), x = y) ↔ 1/x > 3 := by
  sorry

end solution_set_reciprocal_inequality_l1209_120942


namespace problem_solution_l1209_120954

theorem problem_solution (x : ℝ) (h : x^2 - 5*x = 14) :
  (x - 1) * (2*x - 1) - (x + 1)^2 + 1 = 15 := by
  sorry

end problem_solution_l1209_120954


namespace smallest_number_with_sum_l1209_120914

/-- Calculates the sum of all unique permutations of digits in a number -/
def sumOfPermutations (n : ℕ) : ℕ := sorry

/-- Checks if a number is the smallest with a given sum of permutations -/
def isSmallestWithSum (n : ℕ) (sum : ℕ) : Prop :=
  (sumOfPermutations n = sum) ∧ 
  (∀ m : ℕ, m < n → sumOfPermutations m ≠ sum)

/-- The main theorem stating that 47899 is the smallest number 
    whose sum of digit permutations is 4,933,284 -/
theorem smallest_number_with_sum :
  isSmallestWithSum 47899 4933284 := by sorry

end smallest_number_with_sum_l1209_120914


namespace moving_circle_trajectory_l1209_120982

/-- The trajectory of the center of a moving circle that is externally tangent to two fixed circles -/
theorem moving_circle_trajectory (x y : ℝ) : 
  (∃ (r : ℝ), 
    -- First fixed circle
    (∃ (x₁ y₁ : ℝ), x₁^2 + y₁^2 + 4*x₁ + 3 = 0 ∧ 
      -- Moving circle is externally tangent to the first fixed circle
      (x - x₁)^2 + (y - y₁)^2 = (r + 1)^2) ∧ 
    -- Second fixed circle
    (∃ (x₂ y₂ : ℝ), x₂^2 + y₂^2 - 4*x₂ - 5 = 0 ∧ 
      -- Moving circle is externally tangent to the second fixed circle
      (x - x₂)^2 + (y - y₂)^2 = (r + 3)^2)) →
  -- The trajectory of the center of the moving circle
  x^2 - 3*y^2 = 1 := by
sorry

end moving_circle_trajectory_l1209_120982


namespace two_intersecting_circles_common_tangents_l1209_120966

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The number of common external tangents for two intersecting circles -/
def commonExternalTangents (c1 c2 : Circle) : ℕ :=
  sorry

/-- The distance between two points in a 2D plane -/
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sorry

theorem two_intersecting_circles_common_tangents :
  let c1 : Circle := { center := (1, 2), radius := 1 }
  let c2 : Circle := { center := (2, 5), radius := 3 }
  distance c1.center c2.center < c1.radius + c2.radius →
  commonExternalTangents c1 c2 = 2 :=
sorry

end two_intersecting_circles_common_tangents_l1209_120966


namespace oreos_and_cookies_problem_l1209_120995

theorem oreos_and_cookies_problem :
  ∀ (oreos cookies : ℕ) (oreo_price cookie_price : ℚ),
    oreos * 9 = cookies * 4 →
    oreo_price = 2 →
    cookie_price = 3 →
    cookies * cookie_price - oreos * oreo_price = 95 →
    oreos + cookies = 65 := by
  sorry

end oreos_and_cookies_problem_l1209_120995


namespace a_months_is_32_l1209_120941

/-- Represents the pasture rental problem -/
structure PastureRental where
  total_cost : ℕ
  a_horses : ℕ
  b_horses : ℕ
  c_horses : ℕ
  b_months : ℕ
  c_months : ℕ
  b_payment : ℕ

/-- Calculates the number of months a put in the horses -/
def calculate_a_months (p : PastureRental) : ℕ :=
  ((p.total_cost - p.b_payment - p.c_horses * p.c_months) / p.a_horses)

/-- Theorem stating that a put in the horses for 32 months -/
theorem a_months_is_32 (p : PastureRental) 
  (h1 : p.total_cost = 841)
  (h2 : p.a_horses = 12)
  (h3 : p.b_horses = 16)
  (h4 : p.c_horses = 18)
  (h5 : p.b_months = 9)
  (h6 : p.c_months = 6)
  (h7 : p.b_payment = 348) :
  calculate_a_months p = 32 := by
  sorry

#eval calculate_a_months { 
  total_cost := 841, 
  a_horses := 12, 
  b_horses := 16, 
  c_horses := 18, 
  b_months := 9, 
  c_months := 6, 
  b_payment := 348 
}

end a_months_is_32_l1209_120941


namespace folded_paper_area_ratio_l1209_120904

/-- Represents a rectangular piece of paper with specific folding properties -/
structure FoldedPaper where
  width : ℝ
  length : ℝ
  area : ℝ
  foldedArea : ℝ
  lengthIsDoubleWidth : length = 2 * width
  areaIsLengthTimesWidth : area = length * width
  foldedAreaCalculation : foldedArea = area - 2 * (width * width / 4)

/-- Theorem stating that the ratio of folded area to original area is 1/2 -/
theorem folded_paper_area_ratio 
    (paper : FoldedPaper) : paper.foldedArea / paper.area = 1 / 2 := by
  sorry

end folded_paper_area_ratio_l1209_120904


namespace division_of_decimals_l1209_120961

theorem division_of_decimals : (0.08 / 0.002) / 0.04 = 1000 := by
  sorry

end division_of_decimals_l1209_120961


namespace mashed_potatoes_vs_bacon_l1209_120956

theorem mashed_potatoes_vs_bacon (mashed_potatoes bacon : ℕ) 
  (h1 : mashed_potatoes = 408) 
  (h2 : bacon = 42) : 
  mashed_potatoes - bacon = 366 := by
  sorry

end mashed_potatoes_vs_bacon_l1209_120956


namespace dogs_return_simultaneously_l1209_120905

theorem dogs_return_simultaneously
  (L : ℝ)  -- Distance between the two people
  (v V u : ℝ)  -- Speeds of slow person, fast person, and dogs respectively
  (h1 : 0 < v)
  (h2 : v < V)
  (h3 : 0 < u)
  (h4 : V < u) :
  (2 * L * u) / ((u + V) * (u + v)) = (2 * L * u) / ((u + v) * (u + V)) :=
by sorry

end dogs_return_simultaneously_l1209_120905


namespace equilateral_triangle_condition_l1209_120997

theorem equilateral_triangle_condition (a b c : ℂ) :
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  Complex.abs a = 1 →
  Complex.abs b = 1 →
  Complex.abs c = 1 →
  Complex.abs (a + b - c) ^ 2 + Complex.abs (b + c - a) ^ 2 + Complex.abs (c + a - b) ^ 2 = 12 →
  a + b + c = 0 := by
  sorry

end equilateral_triangle_condition_l1209_120997


namespace polar_bear_daily_fish_consumption_l1209_120963

/-- The amount of trout eaten daily by the polar bear in buckets -/
def trout_amount : ℝ := 0.2

/-- The amount of salmon eaten daily by the polar bear in buckets -/
def salmon_amount : ℝ := 0.4

/-- The total amount of fish eaten daily by the polar bear in buckets -/
def total_fish_amount : ℝ := trout_amount + salmon_amount

theorem polar_bear_daily_fish_consumption :
  total_fish_amount = 0.6 := by sorry

end polar_bear_daily_fish_consumption_l1209_120963


namespace find_s_l1209_120958

theorem find_s (r s : ℝ) (hr : r > 1) (hs : s > 1) 
  (h1 : 1/r + 1/s = 1) (h2 : r*s = 9) : 
  s = (9 + 3*Real.sqrt 5) / 2 := by
  sorry

end find_s_l1209_120958


namespace alternative_interest_rate_l1209_120985

/-- Proves that given a principal of 4200 invested for 2 years, if the interest at 18% p.a. 
    is 504 more than the interest at an unknown rate p, then p is equal to 12. -/
theorem alternative_interest_rate (principal : ℝ) (time : ℝ) (known_rate : ℝ) (difference : ℝ) (p : ℝ) : 
  principal = 4200 →
  time = 2 →
  known_rate = 18 →
  difference = 504 →
  principal * known_rate / 100 * time - principal * p / 100 * time = difference →
  p = 12 := by
  sorry

#check alternative_interest_rate

end alternative_interest_rate_l1209_120985


namespace quadratic_translation_l1209_120977

/-- Given a quadratic function f and its translated version g, 
    prove that f has the form -2x^2+1 -/
theorem quadratic_translation (f g : ℝ → ℝ) :
  (∀ x, g x = -2*x^2 + 4*x + 1) →
  (∀ x, g x = f (x - 1) + 2) →
  (∀ x, f x = -2*x^2 + 1) :=
by sorry

end quadratic_translation_l1209_120977


namespace find_x_l1209_120952

theorem find_x (a b c d x : ℝ) 
  (h1 : a ≠ b) 
  (h2 : b ≠ 0) 
  (h3 : (a + x) / (b + x) = 4 * a / (3 * b)) 
  (h4 : c = 4 * a) 
  (h5 : d = 3 * b) :
  x = a * b / (3 * b - 4 * a) :=
by sorry

end find_x_l1209_120952


namespace total_paintable_area_is_1520_l1209_120962

-- Define the parameters of the problem
def num_bedrooms : ℕ := 4
def room_length : ℝ := 14
def room_width : ℝ := 11
def room_height : ℝ := 9
def unpaintable_area : ℝ := 70

-- Calculate the total wall area of one bedroom
def total_wall_area : ℝ := 2 * (room_length * room_height + room_width * room_height)

-- Calculate the paintable area of one bedroom
def paintable_area_per_room : ℝ := total_wall_area - unpaintable_area

-- Theorem statement
theorem total_paintable_area_is_1520 :
  num_bedrooms * paintable_area_per_room = 1520 := by
  sorry

end total_paintable_area_is_1520_l1209_120962


namespace least_positive_angle_theorem_l1209_120931

open Real

/-- The least positive angle θ (in degrees) satisfying cos 10° = sin 20° + sin θ is 40°. -/
theorem least_positive_angle_theorem : 
  (∀ θ : ℝ, 0 < θ ∧ θ < 40 → cos (10 * π / 180) ≠ sin (20 * π / 180) + sin (θ * π / 180)) ∧
  cos (10 * π / 180) = sin (20 * π / 180) + sin (40 * π / 180) :=
sorry

end least_positive_angle_theorem_l1209_120931


namespace model_n_time_proof_l1209_120915

/-- Represents the time (in minutes) taken by a model N computer to complete the task -/
def model_n_time : ℝ := 12

/-- Represents the time (in minutes) taken by a model M computer to complete the task -/
def model_m_time : ℝ := 24

/-- Represents the number of model M computers used -/
def num_model_m : ℕ := 8

/-- Represents the total time (in minutes) taken by both models working together -/
def total_time : ℝ := 1

theorem model_n_time_proof :
  (num_model_m : ℝ) / model_m_time + (num_model_m : ℝ) / model_n_time = 1 / total_time :=
sorry

end model_n_time_proof_l1209_120915


namespace people_at_game_l1209_120939

/-- The number of people who came to a little league game -/
theorem people_at_game (total_seats empty_seats : ℕ) 
  (h1 : total_seats = 92)
  (h2 : empty_seats = 45) :
  total_seats - empty_seats = 47 := by
  sorry

end people_at_game_l1209_120939


namespace v_2015_equals_2_l1209_120906

/-- Function g as defined in the problem -/
def g : ℕ → ℕ
| 1 => 5
| 2 => 3
| 3 => 4
| 4 => 1
| 5 => 2
| _ => 0  -- Default case for completeness

/-- Sequence v defined recursively -/
def v : ℕ → ℕ
| 0 => 3
| n + 1 => g (v n)

/-- Theorem stating that the 2015th term of the sequence is 2 -/
theorem v_2015_equals_2 : v 2015 = 2 := by
  sorry

end v_2015_equals_2_l1209_120906


namespace blaine_win_probability_l1209_120907

/-- Probability of Amelia getting heads -/
def p_amelia : ℚ := 1/4

/-- Probability of Blaine getting heads -/
def p_blaine : ℚ := 1/3

/-- Probability of Charlie getting heads -/
def p_charlie : ℚ := 1/5

/-- The probability that Blaine wins the game -/
def p_blaine_wins : ℚ := 25/36

theorem blaine_win_probability :
  let p_cycle : ℚ := (1 - p_amelia) * (1 - p_blaine) * (1 - p_charlie)
  p_blaine_wins = (1 - p_amelia) * p_blaine / (1 - p_cycle) := by
  sorry

end blaine_win_probability_l1209_120907


namespace unmarked_trees_l1209_120955

def mark_trees (n : ℕ) : Finset ℕ :=
  Finset.filter (fun i => i % 2 = 1 ∨ i % 3 = 1) (Finset.range n)

theorem unmarked_trees :
  (Finset.range 13 \ mark_trees 13).card = 4 := by
  sorry

end unmarked_trees_l1209_120955


namespace initial_workers_correct_l1209_120937

/-- Represents the initial number of workers employed by the contractor -/
def initial_workers : ℕ := 360

/-- Represents the total number of days to complete the wall -/
def total_days : ℕ := 50

/-- Represents the number of days after which progress is measured -/
def days_passed : ℕ := 25

/-- Represents the percentage of work completed after 'days_passed' -/
def work_completed : ℚ := 2/5

/-- Represents the additional workers needed to complete the work on time -/
def additional_workers : ℕ := 90

/-- Theorem stating that the initial number of workers is correct given the conditions -/
theorem initial_workers_correct :
  initial_workers * (total_days : ℚ) = (initial_workers + additional_workers) * 
    (total_days * work_completed) :=
by sorry

end initial_workers_correct_l1209_120937


namespace i_to_2016_equals_1_l1209_120938

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem i_to_2016_equals_1 : i ^ 2016 = 1 := by
  sorry

end i_to_2016_equals_1_l1209_120938


namespace sufficient_not_necessary_l1209_120994

theorem sufficient_not_necessary (x y : ℝ) : 
  (∀ x y : ℝ, x ≥ 2 ∧ y ≥ 2 → x^2 + y^2 ≥ 4) ∧
  (∃ x y : ℝ, x^2 + y^2 ≥ 4 ∧ ¬(x ≥ 2 ∧ y ≥ 2)) :=
by sorry

end sufficient_not_necessary_l1209_120994


namespace cupboard_cost_price_l1209_120918

theorem cupboard_cost_price (selling_price selling_price_with_profit : ℝ) 
  (h1 : selling_price = 0.88 * 6250)
  (h2 : selling_price_with_profit = 1.12 * 6250)
  (h3 : selling_price_with_profit = selling_price + 1500) : 
  6250 = 6250 := by
sorry

end cupboard_cost_price_l1209_120918


namespace geometric_sequence_sum_l1209_120949

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n : ℕ, a (n + 1) = a n * q) →  -- Geometric sequence condition
  q ≠ 1 →  -- Common ratio not equal to 1
  a 1 * a 2 * a 3 = -1/8 →  -- Product of first three terms
  2 * a 4 = a 2 + a 3 →  -- Arithmetic sequence condition
  a 1 + a 2 + a 3 + a 4 = 5/8 := by
  sorry

end geometric_sequence_sum_l1209_120949


namespace coefficient_sum_l1209_120926

theorem coefficient_sum (b₅ b₄ b₃ b₂ b₁ b₀ : ℝ) :
  (∀ x : ℝ, (4*x - 2)^5 = b₅*x^5 + b₄*x^4 + b₃*x^3 + b₂*x^2 + b₁*x + b₀) →
  b₅ + b₄ + b₃ + b₂ + b₁ + b₀ = 32 := by
sorry

end coefficient_sum_l1209_120926


namespace min_value_function_equality_condition_l1209_120944

theorem min_value_function (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x^2 + 12*x + y + 64/(x^2 * y) ≥ 81 :=
by sorry

theorem equality_condition (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x^2 + 12*x + y + 64/(x^2 * y) = 81 ↔ x = 4 ∧ y = 16 :=
by sorry

end min_value_function_equality_condition_l1209_120944


namespace paco_cookies_l1209_120932

theorem paco_cookies (cookies_eaten : ℕ) (cookies_given : ℕ) : 
  cookies_eaten = 14 →
  cookies_given = 13 →
  cookies_eaten = cookies_given + 1 →
  cookies_eaten + cookies_given = 27 :=
by
  sorry

end paco_cookies_l1209_120932


namespace total_flowers_l1209_120981

theorem total_flowers (total_vases : Nat) (vases_with_five : Nat) (flowers_in_four : Nat) (flowers_in_one : Nat) : 
  total_vases = 5 → vases_with_five = 4 → flowers_in_four = 5 → flowers_in_one = 6 → 
  vases_with_five * flowers_in_four + (total_vases - vases_with_five) * flowers_in_one = 26 := by
  sorry

end total_flowers_l1209_120981


namespace gas_bill_payment_l1209_120940

def electricity_bill : ℚ := 60
def gas_bill : ℚ := 40
def water_bill : ℚ := 40
def internet_bill : ℚ := 25

def gas_bill_paid_initially : ℚ := (3 / 4) * gas_bill
def water_bill_paid : ℚ := (1 / 2) * water_bill
def internet_bill_paid : ℚ := 4 * 5

def remaining_to_pay : ℚ := 30

theorem gas_bill_payment (payment : ℚ) : 
  gas_bill + water_bill + internet_bill - 
  (gas_bill_paid_initially + water_bill_paid + internet_bill_paid + payment) = 
  remaining_to_pay → 
  payment = 5 := by sorry

end gas_bill_payment_l1209_120940


namespace fraction_decimal_digits_l1209_120950

/-- The number of digits to the right of the decimal point when a fraction is expressed as a decimal. -/
def decimal_digits (n : ℚ) : ℕ := sorry

/-- The fraction we're considering -/
def fraction : ℚ := 3^6 / (6^4 * 625)

/-- Theorem stating that the number of digits to the right of the decimal point
    in the decimal representation of our fraction is 4 -/
theorem fraction_decimal_digits :
  decimal_digits fraction = 4 := by sorry

end fraction_decimal_digits_l1209_120950


namespace covered_digits_sum_l1209_120920

/-- Represents a five-digit number with some digits possibly covered -/
structure PartialNumber :=
  (d1 d2 d3 d4 d5 : Option Nat)

/-- The sum of the visible digits in a PartialNumber -/
def visibleSum (n : PartialNumber) : Nat :=
  (n.d1.getD 0) * 10000 + (n.d2.getD 0) * 1000 + (n.d3.getD 0) * 100 + (n.d4.getD 0) * 10 + (n.d5.getD 0)

/-- The number of covered digits in a PartialNumber -/
def coveredCount (n : PartialNumber) : Nat :=
  (if n.d1.isNone then 1 else 0) +
  (if n.d2.isNone then 1 else 0) +
  (if n.d3.isNone then 1 else 0) +
  (if n.d4.isNone then 1 else 0) +
  (if n.d5.isNone then 1 else 0)

theorem covered_digits_sum (n1 n2 n3 : PartialNumber) :
  visibleSum n1 + visibleSum n2 + visibleSum n3 = 57263 - 1000 - 200 - 9 ∧
  coveredCount n1 + coveredCount n2 + coveredCount n3 = 3 →
  ∃ (p1 p2 p3 : Nat), 
    (p1 = 1 ∧ p2 = 2 ∧ p3 = 9) ∧
    visibleSum n1 + visibleSum n2 + visibleSum n3 + p1 * 1000 + p2 * 100 + p3 = 57263 :=
by sorry


end covered_digits_sum_l1209_120920


namespace kenny_trumpet_practice_time_l1209_120998

/-- Given Kenny's activities last week, prove the time he spent practicing trumpet. -/
theorem kenny_trumpet_practice_time :
  ∀ (basketball_time running_time trumpet_time : ℕ),
  basketball_time = 10 →
  running_time = 2 * basketball_time →
  trumpet_time = 2 * running_time →
  trumpet_time = 40 :=
by
  sorry

end kenny_trumpet_practice_time_l1209_120998


namespace smallest_n_cookies_l1209_120946

theorem smallest_n_cookies : ∃ (n : ℕ), n > 0 ∧ 16 ∣ (25 * n - 3) ∧ ∀ (m : ℕ), m > 0 ∧ 16 ∣ (25 * m - 3) → n ≤ m := by
  sorry

end smallest_n_cookies_l1209_120946


namespace original_price_satisfies_conditions_l1209_120959

/-- The original price of a dish that satisfies the given conditions --/
def original_price : ℝ := 42

/-- John's payment given the original price --/
def john_payment (price : ℝ) : ℝ := 0.9 * price + 0.15 * price

/-- Jane's payment given the original price --/
def jane_payment (price : ℝ) : ℝ := 0.9 * price + 0.15 * (0.9 * price)

/-- Theorem stating that the original price satisfies the given conditions --/
theorem original_price_satisfies_conditions :
  john_payment original_price - jane_payment original_price = 0.63 :=
sorry

end original_price_satisfies_conditions_l1209_120959


namespace alices_number_l1209_120957

theorem alices_number (n : ℕ) : 
  (∃ k : ℕ, n = 180 * k) → 
  (∃ m : ℕ, n = 240 * m) → 
  2000 < n → 
  n < 5000 → 
  n = 2160 ∨ n = 2880 ∨ n = 3600 ∨ n = 4320 := by
sorry

end alices_number_l1209_120957


namespace area_between_concentric_circles_l1209_120987

theorem area_between_concentric_circles 
  (r : ℝ)  -- radius of inner circle
  (h1 : r > 0)  -- radius is positive
  (h2 : 3*r - r = 3)  -- width of gray region is 3
  : π * (3*r)^2 - π * r^2 = 18 * π := by
  sorry

end area_between_concentric_circles_l1209_120987


namespace two_positive_solutions_l1209_120901

theorem two_positive_solutions (a : ℝ) :
  (∃! x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ 
   (|2*x₁ - 1| - a = 0) ∧ (|2*x₂ - 1| - a = 0)) ↔ 
  (0 < a ∧ a < 1) := by
  sorry

end two_positive_solutions_l1209_120901


namespace jennas_stickers_l1209_120993

/-- Given that the ratio of Kate's stickers to Jenna's stickers is 7:4 and Kate has 21 stickers,
    prove that Jenna has 12 stickers. -/
theorem jennas_stickers (kate_stickers : ℕ) (jenna_stickers : ℕ) : 
  (kate_stickers : ℚ) / jenna_stickers = 7 / 4 → kate_stickers = 21 → jenna_stickers = 12 := by
  sorry

end jennas_stickers_l1209_120993


namespace inequality_solution_range_l1209_120960

theorem inequality_solution_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 3 * a * x + a - 2 < 0) ↔ -8/5 < a ∧ a ≤ 0 := by
  sorry

end inequality_solution_range_l1209_120960


namespace cubic_difference_even_iff_sum_even_l1209_120983

theorem cubic_difference_even_iff_sum_even (p q : ℕ) :
  Even (p^3 - q^3) ↔ Even (p + q) := by sorry

end cubic_difference_even_iff_sum_even_l1209_120983


namespace mod_nine_equiv_l1209_120967

theorem mod_nine_equiv : ∃ (n : ℤ), 0 ≤ n ∧ n < 9 ∧ -1234 ≡ n [ZMOD 9] ∧ n = 8 := by
  sorry

end mod_nine_equiv_l1209_120967


namespace tutorial_time_multiplier_l1209_120986

/-- Represents the time spent on various activities before playing a game --/
structure GamePreparationTime where
  download : ℝ
  install : ℝ
  tutorial : ℝ
  total : ℝ

/-- Theorem: Given the conditions, the tutorial time multiplier is 3 --/
theorem tutorial_time_multiplier (t : GamePreparationTime) : 
  t.download = 10 ∧ 
  t.install = t.download / 2 ∧ 
  t.total = 60 ∧ 
  t.total = t.download + t.install + t.tutorial → 
  t.tutorial = 3 * (t.download + t.install) :=
by sorry

end tutorial_time_multiplier_l1209_120986


namespace three_circles_theorem_l1209_120935

/-- Represents a circle with a center and a radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two circles are externally tangent -/
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 = (c1.radius + c2.radius)^2

/-- The area of the shaded region formed by three externally tangent circles -/
noncomputable def shaded_area (c1 c2 c3 : Circle) : ℝ := sorry

/-- The main theorem -/
theorem three_circles_theorem :
  let c1 : Circle := { center := (0, 0), radius := Real.sqrt 3 - 1 }
  let c2 : Circle := { center := (2, 0), radius := 3 - Real.sqrt 3 }
  let c3 : Circle := { center := (0, 2 * Real.sqrt 3), radius := 1 + Real.sqrt 3 }
  are_externally_tangent c1 c2 ∧
  are_externally_tangent c2 c3 ∧
  are_externally_tangent c3 c1 →
  ∃ (a b c : ℚ),
    shaded_area c1 c2 c3 = a * Real.sqrt 3 + b * Real.pi + c * Real.pi * Real.sqrt 3 ∧
    a + b + c = 3/2 := by
  sorry

end three_circles_theorem_l1209_120935


namespace trapezoid_median_length_l1209_120933

/-- Given a triangle and a trapezoid with the same altitude and equal areas,
    prove that the median of the trapezoid is 24 inches. -/
theorem trapezoid_median_length (h : ℝ) (triangle_area trapezoid_area : ℝ) : 
  triangle_area = (1/2) * 24 * h →
  trapezoid_area = ((15 + 33) / 2) * h →
  triangle_area = trapezoid_area →
  (15 + 33) / 2 = 24 := by
sorry


end trapezoid_median_length_l1209_120933


namespace commute_time_difference_l1209_120968

/-- Given a set of 5 commute times with known average and variance, prove the absolute difference between two unknown times. -/
theorem commute_time_difference (x y : ℝ) : 
  (x + y + 10 + 11 + 9) / 5 = 10 →
  ((x - 10)^2 + (y - 10)^2 + 0^2 + 1^2 + (-1)^2) / 5 = 2 →
  |x - y| = 4 := by
  sorry

end commute_time_difference_l1209_120968


namespace sum_of_even_numbers_l1209_120976

theorem sum_of_even_numbers (n : ℕ) :
  2 * (n * (n + 1) / 2) = n^2 + n :=
sorry

end sum_of_even_numbers_l1209_120976


namespace negation_of_universal_proposition_l1209_120936

theorem negation_of_universal_proposition :
  ¬(∀ x : ℝ, x^2 ≥ Real.log 2) ↔ ∃ x : ℝ, x^2 < Real.log 2 := by
  sorry

end negation_of_universal_proposition_l1209_120936


namespace smallest_obtuse_consecutive_triangle_perimeter_l1209_120909

/-- A triangle with consecutive integer side lengths -/
structure ConsecutiveTriangle where
  a : ℕ
  sides : Fin 3 → ℕ
  consecutive : ∀ i : Fin 2, sides i.succ = sides i + 1
  valid : a > 0 ∧ sides 0 = a

/-- Checks if a triangle is obtuse -/
def isObtuse (t : ConsecutiveTriangle) : Prop :=
  let a := t.sides 0
  let b := t.sides 1
  let c := t.sides 2
  a^2 + b^2 < c^2 ∨ a^2 + c^2 < b^2 ∨ b^2 + c^2 < a^2

/-- The perimeter of a triangle -/
def perimeter (t : ConsecutiveTriangle) : ℕ :=
  t.sides 0 + t.sides 1 + t.sides 2

/-- The main theorem -/
theorem smallest_obtuse_consecutive_triangle_perimeter :
  ∃ (t : ConsecutiveTriangle), isObtuse t ∧
    (∀ (t' : ConsecutiveTriangle), isObtuse t' → perimeter t ≤ perimeter t') ∧
    perimeter t = 9 := by
  sorry

end smallest_obtuse_consecutive_triangle_perimeter_l1209_120909


namespace acme_savings_at_min_shirts_l1209_120972

/-- Acme T-Shirt Company's pricing function -/
def acme_cost (n : ℕ) : ℚ :=
  if n ≤ 20 then 60 + 10 * n
  else (60 + 10 * n) * (9/10)

/-- Beta T-Shirt Company's pricing function -/
def beta_cost (n : ℕ) : ℚ := 15 * n

/-- The minimum number of shirts for which Acme is cheaper than Beta -/
def min_shirts_for_acme_savings : ℕ := 13

theorem acme_savings_at_min_shirts :
  acme_cost min_shirts_for_acme_savings < beta_cost min_shirts_for_acme_savings ∧
  ∀ k : ℕ, k < min_shirts_for_acme_savings →
    acme_cost k ≥ beta_cost k :=
by sorry

end acme_savings_at_min_shirts_l1209_120972


namespace exam_girls_count_l1209_120945

/-- Proves that the number of girls is 1800 given the exam conditions -/
theorem exam_girls_count :
  ∀ (boys girls : ℕ),
  boys + girls = 2000 →
  (34 * boys + 32 * girls : ℚ) = 331 * 20 →
  girls = 1800 := by
sorry

end exam_girls_count_l1209_120945


namespace age_sum_proof_l1209_120975

theorem age_sum_proof (younger_age older_age : ℕ) : 
  older_age - younger_age = 2 →
  older_age = 38 →
  younger_age + older_age = 74 :=
by
  sorry

end age_sum_proof_l1209_120975


namespace notebook_cost_l1209_120930

/-- The total cost of notebooks with given prices and quantities -/
def total_cost (green_price : ℕ) (green_quantity : ℕ) (black_price : ℕ) (pink_price : ℕ) : ℕ :=
  green_price * green_quantity + black_price + pink_price

/-- Theorem: The total cost of 4 notebooks (2 green at $10 each, 1 black at $15, and 1 pink at $10) is $45 -/
theorem notebook_cost : total_cost 10 2 15 10 = 45 := by
  sorry

end notebook_cost_l1209_120930


namespace survey_theorem_l1209_120947

/-- Represents the response of a student to a subject --/
inductive Response
| Yes
| No
| Unsure

/-- Represents a subject with its response counts --/
structure Subject where
  yes_count : Nat
  no_count : Nat
  unsure_count : Nat

/-- The survey results --/
structure SurveyResults where
  total_students : Nat
  subject_m : Subject
  subject_r : Subject
  yes_only_m : Nat

def SurveyResults.students_not_yes_either (results : SurveyResults) : Nat :=
  results.total_students - (results.subject_m.yes_count + results.subject_r.yes_count - results.yes_only_m)

theorem survey_theorem (results : SurveyResults) 
  (h1 : results.total_students = 800)
  (h2 : results.subject_m.yes_count = 500)
  (h3 : results.subject_m.no_count = 200)
  (h4 : results.subject_m.unsure_count = 100)
  (h5 : results.subject_r.yes_count = 400)
  (h6 : results.subject_r.no_count = 100)
  (h7 : results.subject_r.unsure_count = 300)
  (h8 : results.yes_only_m = 170) :
  results.students_not_yes_either = 230 := by
  sorry

#eval SurveyResults.students_not_yes_either {
  total_students := 800,
  subject_m := { yes_count := 500, no_count := 200, unsure_count := 100 },
  subject_r := { yes_count := 400, no_count := 100, unsure_count := 300 },
  yes_only_m := 170
}

end survey_theorem_l1209_120947


namespace teal_survey_result_l1209_120979

theorem teal_survey_result (total : ℕ) (more_green : ℕ) (both : ℕ) (neither : ℕ) (undecided : ℕ) :
  total = 150 →
  more_green = 90 →
  both = 40 →
  neither = 20 →
  undecided = 10 →
  ∃ (more_blue : ℕ), more_blue = 70 ∧ 
    total = more_green + more_blue - both + neither + undecided :=
by sorry

end teal_survey_result_l1209_120979


namespace adams_money_from_mother_l1209_120984

/-- Given Adam's initial savings and final total, prove the amount his mother gave him. -/
theorem adams_money_from_mother (initial_savings final_total : ℕ) 
  (h1 : initial_savings = 79)
  (h2 : final_total = 92)
  : final_total - initial_savings = 13 := by
  sorry

end adams_money_from_mother_l1209_120984


namespace bird_count_l1209_120928

/-- Represents the count of animals in a nature reserve --/
structure AnimalCount where
  birds : ℕ
  mythical : ℕ
  mammals : ℕ

/-- Theorem stating the number of two-legged birds in the nature reserve --/
theorem bird_count (ac : AnimalCount) : 
  ac.birds + ac.mythical + ac.mammals = 300 →
  2 * ac.birds + 3 * ac.mythical + 4 * ac.mammals = 708 →
  ac.birds = 192 := by
  sorry


end bird_count_l1209_120928


namespace basketball_substitutions_remainder_l1209_120912

/-- Represents the number of ways to make substitutions in a basketball game -/
def substitution_ways (total_players starters max_substitutions : ℕ) : ℕ :=
  sorry

/-- The main theorem about the basketball substitutions problem -/
theorem basketball_substitutions_remainder :
  substitution_ways 15 5 4 % 1000 = 301 := by sorry

end basketball_substitutions_remainder_l1209_120912


namespace trigonometric_product_equals_three_fourths_l1209_120974

theorem trigonometric_product_equals_three_fourths :
  let cos30 : ℝ := Real.sqrt 3 / 2
  let sin60 : ℝ := Real.sqrt 3 / 2
  let sin30 : ℝ := 1 / 2
  let cos60 : ℝ := 1 / 2
  (1 - 1 / cos30) * (1 + 1 / sin60) * (1 - 1 / sin30) * (1 + 1 / cos60) = 3 / 4 := by
  sorry

end trigonometric_product_equals_three_fourths_l1209_120974


namespace sally_balloons_l1209_120973

/-- The number of orange balloons Sally has after losing some -/
def remaining_balloons (initial : ℕ) (lost : ℕ) : ℕ := initial - lost

/-- Proof that Sally has 7 orange balloons after losing 2 from her initial 9 -/
theorem sally_balloons : remaining_balloons 9 2 = 7 := by
  sorry

end sally_balloons_l1209_120973


namespace min_value_reciprocal_sum_l1209_120969

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  1 / a + 1 / b ≥ 2 ∧ (1 / a + 1 / b = 2 ↔ a = 1 ∧ b = 1) := by
  sorry

end min_value_reciprocal_sum_l1209_120969


namespace num_cities_with_protests_is_21_l1209_120934

/-- The number of cities experiencing protests given the specified conditions -/
def num_cities_with_protests : ℕ :=
  let protest_days : ℕ := 30
  let arrests_per_day_per_city : ℕ := 10
  let pre_trial_days : ℕ := 4
  let post_trial_days : ℕ := 7  -- half of 2-week sentence
  let total_jail_weeks : ℕ := 9900

  -- Calculate the number of cities
  21

/-- Theorem stating that the number of cities experiencing protests is 21 -/
theorem num_cities_with_protests_is_21 :
  num_cities_with_protests = 21 := by
  sorry

end num_cities_with_protests_is_21_l1209_120934


namespace print_shop_cost_difference_l1209_120970

def print_shop_x_price : ℝ := 1.25
def print_shop_y_price : ℝ := 2.75
def print_shop_x_discount : ℝ := 0.10
def print_shop_y_discount : ℝ := 0.05
def print_shop_x_tax : ℝ := 0.07
def print_shop_y_tax : ℝ := 0.09
def num_copies : ℕ := 40

def calculate_total_cost (base_price discount tax : ℝ) (copies : ℕ) : ℝ :=
  let pre_discount := base_price * copies
  let discounted := pre_discount * (1 - discount)
  discounted * (1 + tax)

theorem print_shop_cost_difference :
  calculate_total_cost print_shop_y_price print_shop_y_discount print_shop_y_tax num_copies -
  calculate_total_cost print_shop_x_price print_shop_x_discount print_shop_x_tax num_copies =
  65.755 := by sorry

end print_shop_cost_difference_l1209_120970


namespace work_completion_l1209_120919

/-- The number of days A takes to complete the work alone -/
def days_A : ℝ := 4

/-- The number of days B takes to complete the work alone -/
def days_B : ℝ := 12

/-- The number of days B takes to finish the remaining work after A leaves -/
def days_B_remaining : ℝ := 4.000000000000001

/-- The number of days A and B work together -/
def days_together : ℝ := 2

theorem work_completion :
  let rate_A := 1 / days_A
  let rate_B := 1 / days_B
  let rate_together := rate_A + rate_B
  rate_together * days_together + rate_B * days_B_remaining = 1 :=
by sorry

end work_completion_l1209_120919


namespace sum_equals_negative_power_l1209_120925

open BigOperators

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the sum
def T : ℤ := ∑ k in Finset.range 25, (-1:ℤ)^k * binomial 50 (2*k)

-- Theorem statement
theorem sum_equals_negative_power : T = -2^25 := by sorry

end sum_equals_negative_power_l1209_120925


namespace cubic_roots_problem_l1209_120980

/-- Given a cubic polynomial x³ - 2x² + 5x - 8 with roots p, q, r,
    and another cubic polynomial x³ + ux² + vx + w with roots p+q, q+r, r+p,
    prove that w = 34 -/
theorem cubic_roots_problem (p q r u v w : ℝ) : 
  (∀ x, x^3 - 2*x^2 + 5*x - 8 = 0 ↔ x = p ∨ x = q ∨ x = r) →
  (∀ x, x^3 + u*x^2 + v*x + w = 0 ↔ x = p+q ∨ x = q+r ∨ x = r+p) →
  w = 34 := by
sorry

end cubic_roots_problem_l1209_120980


namespace largest_additional_plates_is_largest_possible_l1209_120943

/-- Represents the number of choices in each section of a license plate --/
structure LicensePlateChoices where
  section1 : Nat
  section2 : Nat
  section3 : Nat

/-- Calculates the total number of possible license plates --/
def totalPlates (choices : LicensePlateChoices) : Nat :=
  choices.section1 * choices.section2 * choices.section3

/-- The initial number of choices for each section --/
def initialChoices : LicensePlateChoices :=
  { section1 := 5, section2 := 3, section3 := 3 }

/-- The optimal distribution of new letters --/
def optimalChoices : LicensePlateChoices :=
  { section1 := 5, section2 := 5, section3 := 4 }

/-- Theorem stating that the largest number of additional plates is 55 --/
theorem largest_additional_plates :
  totalPlates optimalChoices - totalPlates initialChoices = 55 := by
  sorry

/-- Theorem stating that this is indeed the largest possible number --/
theorem is_largest_possible (newChoices : LicensePlateChoices)
  (h1 : newChoices.section1 + newChoices.section2 + newChoices.section3 = 
        initialChoices.section1 + initialChoices.section2 + initialChoices.section3 + 3) :
  totalPlates newChoices - totalPlates initialChoices ≤ 55 := by
  sorry

end largest_additional_plates_is_largest_possible_l1209_120943


namespace centroid_perpendicular_distance_l1209_120916

-- Define a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a line
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the perpendicular distance from a point to a line
def perpendicularDistance (p : ℝ × ℝ) (l : Line) : ℝ :=
  sorry

-- Define the centroid of a triangle
def centroid (t : Triangle) : ℝ × ℝ :=
  sorry

-- Theorem statement
theorem centroid_perpendicular_distance (t : Triangle) (l : Line) :
  perpendicularDistance (centroid t) l =
    (perpendicularDistance t.A l + perpendicularDistance t.B l + perpendicularDistance t.C l) / 3 :=
  sorry

end centroid_perpendicular_distance_l1209_120916


namespace art_of_passing_through_walls_l1209_120991

theorem art_of_passing_through_walls (n : ℝ) :
  (8 * Real.sqrt (8 / n) = Real.sqrt (8 * (8 / n))) ↔ n = 63 := by
  sorry

end art_of_passing_through_walls_l1209_120991


namespace geometric_sum_first_six_terms_l1209_120964

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_first_six_terms :
  let a : ℚ := 1/3
  let r : ℚ := 1/4
  let n : ℕ := 6
  geometric_sum a r n = 4/3 := by sorry

end geometric_sum_first_six_terms_l1209_120964


namespace circumradius_of_special_triangle_l1209_120951

/-- Represents a triangle with consecutive natural number side lengths and an inscribed circle radius of 4 -/
structure SpecialTriangle where
  n : ℕ
  side_a : ℕ := n - 1
  side_b : ℕ := n
  side_c : ℕ := n + 1
  inradius : ℝ := 4

/-- The radius of the circumcircle of a SpecialTriangle is 65/8 -/
theorem circumradius_of_special_triangle (t : SpecialTriangle) : 
  (t.side_a : ℝ) * t.side_b * t.side_c / (4 * t.inradius * (t.side_a + t.side_b + t.side_c) / 2) = 65 / 8 := by
  sorry

end circumradius_of_special_triangle_l1209_120951


namespace square_circle_area_ratio_l1209_120908

/-- Given a square and a circle that intersect such that each side of the square
    contains a chord of the circle equal in length to twice the radius of the circle,
    the ratio of the area of the square to the area of the circle is 2/π. -/
theorem square_circle_area_ratio (s : Real) (r : Real) (π : Real) :
  s > 0 ∧ r > 0 ∧ π > 0 ∧ 
  (2 * r = s) ∧  -- chord length equals side length
  (π * r^2 = π * r * r) →  -- definition of circle area
  (s^2 / (π * r^2) = 2 / π) :=
sorry

end square_circle_area_ratio_l1209_120908


namespace roots_of_quadratic_equation_l1209_120923

theorem roots_of_quadratic_equation :
  let f : ℝ → ℝ := λ x ↦ x^2 - 3
  ∃ x₁ x₂ : ℝ, x₁ = Real.sqrt 3 ∧ x₂ = -Real.sqrt 3 ∧ f x₁ = 0 ∧ f x₂ = 0 :=
by sorry

end roots_of_quadratic_equation_l1209_120923


namespace geometric_sequence_properties_l1209_120978

def is_geometric_sequence (s : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, s (n + 1) = r * s n

theorem geometric_sequence_properties
  (a b : ℕ → ℝ)
  (ha : is_geometric_sequence a)
  (hb : is_geometric_sequence b) :
  (¬ ∀ s : ℕ → ℝ, (∀ n, s n = a n + b n) → is_geometric_sequence s) ∧
  (∃ s : ℕ → ℝ, (∀ n, s n = a n * b n) ∧ is_geometric_sequence s) :=
sorry

end geometric_sequence_properties_l1209_120978


namespace sum_of_reciprocals_squared_l1209_120948

noncomputable def a : ℝ := Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 6
noncomputable def b : ℝ := -Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 6
noncomputable def c : ℝ := Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 6
noncomputable def d : ℝ := -Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 6

theorem sum_of_reciprocals_squared :
  (1/a + 1/b + 1/c + 1/d)^2 = 16*(11 + 2*Real.sqrt 30) / (11 + 2*Real.sqrt 30 - 3*Real.sqrt 6)^2 :=
by sorry

end sum_of_reciprocals_squared_l1209_120948


namespace planes_parallel_or_intersect_l1209_120929

-- Define a type for 3D space
structure Space3D where
  -- Add necessary fields here
  
-- Define a type for planes in 3D space
structure Plane where
  -- Add necessary fields here

-- Define a type for lines in 3D space
structure Line where
  -- Add necessary fields here

-- Define what it means for a line to be parallel to a plane
def Line.parallelTo (l : Line) (p : Plane) : Prop :=
  sorry

-- Define what it means for a plane to contain a line
def Plane.contains (p : Plane) (l : Line) : Prop :=
  sorry

-- Define what it means for two planes to be parallel
def Plane.parallel (p1 p2 : Plane) : Prop :=
  sorry

-- Define what it means for two planes to intersect
def Plane.intersect (p1 p2 : Plane) : Prop :=
  sorry

-- The main theorem
theorem planes_parallel_or_intersect (p1 p2 : Plane) :
  (∃ (S : Set Line), Set.Infinite S ∧ (∀ l ∈ S, p1.contains l ∧ l.parallelTo p2)) →
  (p1.parallel p2 ∨ p1.intersect p2) :=
sorry

end planes_parallel_or_intersect_l1209_120929


namespace total_triangles_is_200_l1209_120922

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  sideLength : ℕ

/-- Represents the large equilateral triangle -/
def largeTriangle : EquilateralTriangle :=
  { sideLength := 10 }

/-- Represents a small equilateral triangle -/
def smallTriangle : EquilateralTriangle :=
  { sideLength := 1 }

/-- The number of small triangles that fit in the large triangle -/
def numSmallTriangles : ℕ := 100

/-- Counts the number of equilateral triangles of a given side length -/
def countTriangles (sideLength : ℕ) : ℕ :=
  if sideLength = 1 then numSmallTriangles
  else if sideLength > largeTriangle.sideLength then 0
  else largeTriangle.sideLength - sideLength + 1

/-- The total number of equilateral triangles -/
def totalTriangles : ℕ :=
  (List.range largeTriangle.sideLength).map countTriangles |>.sum

theorem total_triangles_is_200 : totalTriangles = 200 := by
  sorry

end total_triangles_is_200_l1209_120922


namespace rice_sale_proof_l1209_120953

/-- Calculates the daily amount of rice to be sold given initial amount, additional amount, and number of days -/
def daily_rice_sale (initial_tons : ℕ) (additional_kg : ℕ) (days : ℕ) : ℕ :=
  (initial_tons * 1000 + additional_kg) / days

/-- Proves that given 4 tons of rice initially, with an additional 4000 kilograms transported in,
    and needing to be sold within 4 days, the amount of rice to be sold each day is 2000 kilograms -/
theorem rice_sale_proof : daily_rice_sale 4 4000 4 = 2000 := by
  sorry

end rice_sale_proof_l1209_120953


namespace equation_solution_l1209_120900

theorem equation_solution (a : ℝ) : 
  (∀ x, 2 * x + a = 3 ↔ x = -1) → a = 5 := by
  sorry

end equation_solution_l1209_120900


namespace study_time_for_target_average_l1209_120902

/-- Calculates the number of minutes needed to study on the 12th day to achieve a given average -/
def minutes_to_study_on_last_day (days_30min : ℕ) (days_45min : ℕ) (target_average : ℕ) : ℕ :=
  let total_days := days_30min + days_45min + 1
  let total_minutes_needed := total_days * target_average
  let minutes_already_studied := days_30min * 30 + days_45min * 45
  total_minutes_needed - minutes_already_studied

/-- Theorem stating that given the specific study pattern, 90 minutes are needed on the 12th day -/
theorem study_time_for_target_average :
  minutes_to_study_on_last_day 7 4 40 = 90 := by
  sorry

end study_time_for_target_average_l1209_120902


namespace bowtie_equation_solution_l1209_120913

-- Define the operation ⋈
noncomputable def bowtie (a b : ℝ) : ℝ :=
  a^2 + Real.sqrt (b + 2 * Real.sqrt (b + 3 * Real.sqrt b))

-- State the theorem
theorem bowtie_equation_solution :
  ∃ x : ℝ, bowtie 3 x = 18 ∧ x = 63 :=
by
  sorry

end bowtie_equation_solution_l1209_120913


namespace chess_game_probability_l1209_120965

/-- The probability of a chess game resulting in a draw -/
def prob_draw : ℚ := 1/2

/-- The probability of player A winning the chess game -/
def prob_a_win : ℚ := 1/3

/-- The probability of player A not losing the chess game -/
def prob_a_not_lose : ℚ := prob_draw + prob_a_win

theorem chess_game_probability : prob_a_not_lose = 5/6 := by
  sorry

end chess_game_probability_l1209_120965


namespace fraction_equality_l1209_120903

theorem fraction_equality (a b : ℝ) (h1 : b ≠ 0) (h2 : 2*a ≠ b) (h3 : a/b = 2/3) : 
  b/(2*a - b) = 3 := by
sorry

end fraction_equality_l1209_120903


namespace function_lower_bound_l1209_120910

open Real

theorem function_lower_bound (a x : ℝ) (ha : a > 0) : 
  let f : ℝ → ℝ := λ x => a * (exp x + a) - x
  f x > 2 * log a + 3/2 := by
  sorry

end function_lower_bound_l1209_120910


namespace subset_iff_m_le_three_l1209_120999

-- Define the sets A and B
def A : Set ℝ := { x | -2 ≤ x ∧ x ≤ 5 }
def B (m : ℝ) : Set ℝ := { x | m + 1 ≤ x ∧ x ≤ 2*m - 1 }

-- State the theorem
theorem subset_iff_m_le_three (m : ℝ) : B m ⊆ A ↔ m ≤ 3 := by
  sorry

end subset_iff_m_le_three_l1209_120999


namespace cubic_three_roots_l1209_120917

-- Define the cubic function
def f (x : ℝ) : ℝ := x^3 - 3*x + 1

-- Theorem statement
theorem cubic_three_roots : ∃ (a b c : ℝ), (∀ x, f x = 0 ↔ x = a ∨ x = b ∨ x = c) ∧ (a ≠ b ∧ b ≠ c ∧ a ≠ c) :=
sorry

end cubic_three_roots_l1209_120917


namespace simplify_expression_l1209_120992

theorem simplify_expression (x : ℝ) : 3*x + 4*x^3 + 2 - (7 - 3*x - 4*x^3) = 8*x^3 + 6*x - 5 := by
  sorry

end simplify_expression_l1209_120992


namespace additional_toothpicks_eq_351_l1209_120921

/-- The number of toothpicks needed for a 3-step staircase -/
def initial_toothpicks : ℕ := 18

/-- The ratio of the geometric progression for toothpick increase -/
def progression_ratio : ℕ := 3

/-- Calculate the total additional toothpicks needed to complete a 6-step staircase -/
def additional_toothpicks : ℕ :=
  let step4_increase := (initial_toothpicks / 2) * progression_ratio
  let step5_increase := step4_increase * progression_ratio
  let step6_increase := step5_increase * progression_ratio
  step4_increase + step5_increase + step6_increase

/-- Theorem stating that the additional toothpicks needed is 351 -/
theorem additional_toothpicks_eq_351 : additional_toothpicks = 351 := by
  sorry

end additional_toothpicks_eq_351_l1209_120921


namespace largest_angle_measure_l1209_120989

/-- A convex heptagon with interior angles as specified -/
structure ConvexHeptagon where
  x : ℝ
  angle1 : ℝ := x + 2
  angle2 : ℝ := 2 * x
  angle3 : ℝ := 3 * x
  angle4 : ℝ := 4 * x
  angle5 : ℝ := 5 * x
  angle6 : ℝ := 6 * x - 2
  angle7 : ℝ := 7 * x - 3

/-- The sum of interior angles of a heptagon is 900 degrees -/
axiom heptagon_angle_sum (h : ConvexHeptagon) : 
  h.angle1 + h.angle2 + h.angle3 + h.angle4 + h.angle5 + h.angle6 + h.angle7 = 900

/-- The measure of the largest angle in the specified convex heptagon is 222.75 degrees -/
theorem largest_angle_measure (h : ConvexHeptagon) : 
  max h.angle1 (max h.angle2 (max h.angle3 (max h.angle4 (max h.angle5 (max h.angle6 h.angle7))))) = 222.75 := by
  sorry

end largest_angle_measure_l1209_120989
