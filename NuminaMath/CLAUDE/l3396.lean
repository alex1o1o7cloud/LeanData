import Mathlib

namespace continuity_at_two_delta_epsilon_relation_l3396_339669

def f (x : ℝ) : ℝ := -5 * x^2 - 8

theorem continuity_at_two :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 2| < δ → |f x - f 2| < ε :=
by
  sorry

theorem delta_epsilon_relation :
  ∀ ε > 0, ∃ δ > 0, δ = ε / 25 ∧
    ∀ x, |x - 2| < δ → |f x - f 2| < ε :=
by
  sorry

end continuity_at_two_delta_epsilon_relation_l3396_339669


namespace x_range_l3396_339645

theorem x_range (x : ℝ) : 
  (6 - 3 * x ≥ 0) ∧ (¬(1 / (x + 1) < 0)) → x ∈ Set.Icc (-1) 2 := by
sorry

end x_range_l3396_339645


namespace horse_cattle_price_problem_l3396_339604

theorem horse_cattle_price_problem (x y : ℚ) :
  (4 * x + 6 * y = 48) ∧ (3 * x + 5 * y = 38) →
  x = 6 ∧ y = 4 := by
sorry

end horse_cattle_price_problem_l3396_339604


namespace set_intersection_equality_l3396_339630

def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {x | ∃ k : ℤ, x = k}

theorem set_intersection_equality :
  (Aᶜ ∩ B ∩ {x : ℝ | -2 ≤ x ∧ x ≤ 0}) = {-1, 0} := by sorry

end set_intersection_equality_l3396_339630


namespace remainder_of_123456789012_div_210_l3396_339643

theorem remainder_of_123456789012_div_210 :
  123456789012 % 210 = 0 := by
  sorry

end remainder_of_123456789012_div_210_l3396_339643


namespace quadratic_roots_difference_range_l3396_339619

theorem quadratic_roots_difference_range (a b c x₁ x₂ : ℝ) :
  a > b →
  b > c →
  a + b + c = 0 →
  a * x₁^2 + 2 * b * x₁ + c = 0 →
  a * x₂^2 + 2 * b * x₂ + c = 0 →
  Real.sqrt 3 < |x₁ - x₂| ∧ |x₁ - x₂| < 2 * Real.sqrt 3 := by
  sorry

end quadratic_roots_difference_range_l3396_339619


namespace fourth_sphere_radius_l3396_339662

/-- A cone with four spheres inside, where three spheres have radius 3 and touch the base. -/
structure ConeFourSpheres where
  /-- Radius of the three identical spheres -/
  r₁ : ℝ
  /-- Radius of the fourth sphere -/
  r₂ : ℝ
  /-- Angle between the slant height and the base of the cone -/
  θ : ℝ
  /-- The three identical spheres touch the base of the cone -/
  touch_base : True
  /-- All spheres touch each other externally -/
  touch_externally : True
  /-- All spheres touch the lateral surface of the cone -/
  touch_lateral : True
  /-- The radius of the three identical spheres is 3 -/
  r₁_eq_3 : r₁ = 3
  /-- The angle between the slant height and the base of the cone is π/3 -/
  θ_eq_pi_div_3 : θ = π / 3

/-- The radius of the fourth sphere in the cone arrangement is 9 - 4√2 -/
theorem fourth_sphere_radius (c : ConeFourSpheres) : c.r₂ = 9 - 4 * Real.sqrt 2 := by
  sorry

end fourth_sphere_radius_l3396_339662


namespace client_phones_dropped_off_kevins_phone_repair_problem_l3396_339615

theorem client_phones_dropped_off (initial_phones : ℕ) (repaired_phones : ℕ) (phones_per_person : ℕ) : ℕ :=
  let remaining_phones := initial_phones - repaired_phones
  let total_phones_to_repair := 2 * phones_per_person
  total_phones_to_repair - remaining_phones

theorem kevins_phone_repair_problem :
  client_phones_dropped_off 15 3 9 = 6 := by
  sorry

end client_phones_dropped_off_kevins_phone_repair_problem_l3396_339615


namespace problem_2021_l3396_339678

theorem problem_2021 : (2021^2 - 2020) / 2021 + 7 = 2027 := by
  sorry

end problem_2021_l3396_339678


namespace bike_sharing_growth_specific_bike_sharing_case_l3396_339627

/-- Represents the growth of shared bicycles over three months -/
theorem bike_sharing_growth 
  (initial_bikes : ℕ) 
  (planned_increase : ℕ) 
  (growth_rate : ℝ) : 
  initial_bikes * (1 + growth_rate)^2 = initial_bikes + planned_increase :=
by
  sorry

/-- The specific case mentioned in the problem -/
theorem specific_bike_sharing_case (x : ℝ) : 
  1000 * (1 + x)^2 = 1000 + 440 :=
by
  sorry

end bike_sharing_growth_specific_bike_sharing_case_l3396_339627


namespace second_pipe_fill_time_l3396_339640

/-- Represents a system of pipes filling or draining a tank -/
structure PipeSystem where
  fill_time_1 : ℝ  -- Time for first pipe to fill the tank
  drain_time : ℝ   -- Time for drain pipe to empty the tank
  combined_time : ℝ -- Time to fill the tank with all pipes open
  fill_time_2 : ℝ  -- Time for second pipe to fill the tank (to be proven)

/-- The theorem stating the relationship between the pipes' fill times -/
theorem second_pipe_fill_time (ps : PipeSystem) 
  (h1 : ps.fill_time_1 = 5)
  (h2 : ps.drain_time = 20)
  (h3 : ps.combined_time = 2.5) : 
  ps.fill_time_2 = 4 := by
  sorry


end second_pipe_fill_time_l3396_339640


namespace book_difference_proof_l3396_339637

def old_town_books : ℕ := 750
def riverview_books : ℕ := 1240
def downtown_books : ℕ := 1800
def eastside_books : ℕ := 1620

def library_books : List ℕ := [old_town_books, riverview_books, downtown_books, eastside_books]

theorem book_difference_proof :
  (List.maximum library_books).get! - (List.minimum library_books).get! = 1050 := by
  sorry

end book_difference_proof_l3396_339637


namespace specific_systematic_sample_l3396_339648

/-- Systematic sampling function -/
def systematicSample (totalItems : ℕ) (sampleSize : ℕ) (firstNumber : ℕ) (nthItem : ℕ) : ℕ :=
  let k := totalItems / sampleSize
  firstNumber + k * (nthItem - 1)

/-- Theorem for the specific systematic sampling problem -/
theorem specific_systematic_sample :
  systematicSample 1000 50 15 40 = 795 := by
  sorry

end specific_systematic_sample_l3396_339648


namespace gumball_problem_solution_l3396_339605

/-- Represents the number of gumballs of each color in the machine -/
structure GumballMachine where
  red : Nat
  white : Nat
  blue : Nat
  green : Nat

/-- 
Given a gumball machine with the specified number of gumballs for each color,
this function returns the minimum number of gumballs one must buy to guarantee
getting four of the same color.
-/
def minGumballsToBuy (machine : GumballMachine) : Nat :=
  sorry

/-- The theorem stating the correct answer for the given problem -/
theorem gumball_problem_solution :
  let machine : GumballMachine := { red := 10, white := 6, blue := 8, green := 9 }
  minGumballsToBuy machine = 13 := by
  sorry

end gumball_problem_solution_l3396_339605


namespace largest_power_of_five_l3396_339653

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_of_factorials (n : ℕ) : ℕ := factorial n + factorial (n + 1) + factorial (n + 2)

theorem largest_power_of_five (n : ℕ) : 
  (∃ k : ℕ, sum_of_factorials 105 = 5^n * k) ∧ 
  (∀ m : ℕ, m > n → ¬∃ k : ℕ, sum_of_factorials 105 = 5^m * k) ↔ 
  n = 25 :=
sorry

end largest_power_of_five_l3396_339653


namespace not_all_greater_than_quarter_l3396_339633

theorem not_all_greater_than_quarter (a b c : ℝ) 
  (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hc : 0 < c ∧ c < 1) : 
  ¬(((1 - a) * b > 1/4) ∧ ((1 - b) * c > 1/4) ∧ ((1 - c) * a > 1/4)) := by
  sorry

end not_all_greater_than_quarter_l3396_339633


namespace power_calculation_l3396_339698

theorem power_calculation : 7^3 - 5*(6^2) + 2^4 = 179 := by
  sorry

end power_calculation_l3396_339698


namespace zero_in_interval_l3396_339667

-- Define the function f(x) = 2x - 5
def f (x : ℝ) : ℝ := 2 * x - 5

-- State the theorem
theorem zero_in_interval :
  (∀ x y, x < y → f x < f y) →  -- f is monotonically increasing
  Continuous f →                -- f is continuous
  ∃ c ∈ Set.Ioo 2 3, f c = 0    -- there exists a c in (2, 3) such that f(c) = 0
:= by sorry

end zero_in_interval_l3396_339667


namespace find_number_multiplied_by_9999_l3396_339636

theorem find_number_multiplied_by_9999 : ∃ x : ℕ, x * 9999 = 724797420 ∧ x = 72480 := by
  sorry

end find_number_multiplied_by_9999_l3396_339636


namespace square_value_l3396_339638

theorem square_value (r : ℝ) (h1 : r + r = 75) (h2 : (r + r) + 2 * r = 143) : r = 41 := by
  sorry

end square_value_l3396_339638


namespace inequality_proofs_l3396_339626

theorem inequality_proofs :
  (∀ (a b : ℝ), a > 0 → b > 0 → (b/a) + (a/b) ≥ 2) ∧
  (∀ (x y : ℝ), x*y < 0 → (x/y) + (y/x) ≤ -2) := by
  sorry

end inequality_proofs_l3396_339626


namespace mark_survival_days_l3396_339618

/- Define the problem parameters -/
def num_astronauts : ℕ := 6
def food_days_per_astronaut : ℕ := 5
def water_per_astronaut : ℝ := 50
def potato_yield_per_sqm : ℝ := 2.5
def water_required_per_sqm : ℝ := 4
def potato_needed_per_day : ℝ := 1.875

/- Define the theorem -/
theorem mark_survival_days :
  let initial_food_days := num_astronauts * food_days_per_astronaut
  let total_water := num_astronauts * water_per_astronaut
  let irrigated_area := total_water / water_required_per_sqm
  let total_potatoes := irrigated_area * potato_yield_per_sqm
  let potato_days := total_potatoes / potato_needed_per_day
  initial_food_days + potato_days = 130 := by
  sorry


end mark_survival_days_l3396_339618


namespace min_value_expression_lower_bound_achievable_l3396_339683

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 4) :
  ((x + 1) * (2*y + 1)) / (x * y) ≥ 9/2 :=
sorry

theorem lower_bound_achievable :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + 2*y = 4 ∧ ((x + 1) * (2*y + 1)) / (x * y) = 9/2 :=
sorry

end min_value_expression_lower_bound_achievable_l3396_339683


namespace cost_for_23_days_l3396_339611

/-- Calculates the cost of staying in a student youth hostel for a given number of days. -/
def hostelCost (days : ℕ) : ℚ :=
  let firstWeekRate : ℚ := 18
  let additionalWeekRate : ℚ := 13
  let firstWeekDays : ℕ := min days 7
  let additionalDays : ℕ := days - firstWeekDays
  firstWeekRate * firstWeekDays + additionalWeekRate * additionalDays

/-- Proves that the cost for a 23-day stay is $334.00 -/
theorem cost_for_23_days : hostelCost 23 = 334 := by
  sorry

#eval hostelCost 23

end cost_for_23_days_l3396_339611


namespace multiple_problem_l3396_339622

theorem multiple_problem (n m : ℝ) : n = 5 → m * n - 15 = 2 * n + 10 → m = 7 := by
  sorry

end multiple_problem_l3396_339622


namespace perimeter_of_20_rectangles_l3396_339610

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Creates a list of rectangles following the given pattern -/
def createRectangles (n : ℕ) : List Rectangle :=
  List.range n |>.map (fun i => ⟨i + 1, i + 2⟩)

/-- Calculates the perimeter of a polygon formed by arranging rectangles -/
def polygonPerimeter (rectangles : List Rectangle) : ℕ :=
  sorry

theorem perimeter_of_20_rectangles :
  let rectangles := createRectangles 20
  polygonPerimeter rectangles = 462 := by
  sorry

end perimeter_of_20_rectangles_l3396_339610


namespace f_inverse_composition_l3396_339652

def f : Fin 6 → Fin 6
| 1 => 4
| 2 => 6
| 3 => 2
| 4 => 5
| 5 => 3
| 6 => 1

theorem f_inverse_composition (h : Function.Bijective f) :
  (Function.invFun f (Function.invFun f (Function.invFun f 6))) = 5 := by
  sorry

end f_inverse_composition_l3396_339652


namespace lottery_probability_l3396_339623

def megaball_count : ℕ := 30
def winnerball_count : ℕ := 50
def ordered_winnerball_count : ℕ := 2
def unordered_winnerball_count : ℕ := 5

def megaball_prob : ℚ := 1 / megaball_count
def ordered_winnerball_prob : ℚ := 1 / (winnerball_count * (winnerball_count - 1))
def unordered_winnerball_prob : ℚ := 1 / (Nat.choose (winnerball_count - ordered_winnerball_count) unordered_winnerball_count)

theorem lottery_probability :
  megaball_prob * ordered_winnerball_prob * unordered_winnerball_prob = 1 / 125703480000 :=
by sorry

end lottery_probability_l3396_339623


namespace least_k_for_error_bound_l3396_339668

def u : ℕ → ℚ
  | 0 => 1/8
  | n + 1 => 3 * u n - 3 * (u n)^2

def L : ℚ := 1/3

theorem least_k_for_error_bound :
  (∀ k < 9, |u k - L| > 1/2^500) ∧
  |u 9 - L| ≤ 1/2^500 := by
  sorry

end least_k_for_error_bound_l3396_339668


namespace arithmetic_square_root_of_nine_l3396_339675

theorem arithmetic_square_root_of_nine : ∃! x : ℝ, x ≥ 0 ∧ x^2 = 9 :=
  by sorry

end arithmetic_square_root_of_nine_l3396_339675


namespace pond_filling_time_l3396_339661

/-- Represents the time needed to fill a pond under specific conditions. -/
def time_to_fill_pond (initial_flow_ratio : ℚ) (initial_fill_ratio : ℚ) (initial_days : ℚ) : ℚ :=
  let total_volume := 18 * initial_flow_ratio * initial_days / initial_fill_ratio
  let remaining_volume := total_volume * (1 - initial_fill_ratio)
  remaining_volume / 1

theorem pond_filling_time :
  let initial_flow_ratio : ℚ := 3/4
  let initial_fill_ratio : ℚ := 2/3
  let initial_days : ℚ := 16
  time_to_fill_pond initial_flow_ratio initial_fill_ratio initial_days = 6 := by
  sorry

#eval time_to_fill_pond (3/4) (2/3) 16

end pond_filling_time_l3396_339661


namespace total_blue_marbles_l3396_339644

-- Define the number of marbles collected by each friend
def jenny_red : ℕ := 30
def jenny_blue : ℕ := 25
def mary_red : ℕ := 2 * jenny_red
def anie_red : ℕ := mary_red + 20
def anie_blue : ℕ := 2 * jenny_blue
def mary_blue : ℕ := anie_blue / 2

-- Theorem to prove
theorem total_blue_marbles :
  jenny_blue + mary_blue + anie_blue = 100 := by
  sorry

end total_blue_marbles_l3396_339644


namespace age_puzzle_solution_l3396_339670

/-- Represents a person's age --/
structure Age :=
  (tens : Nat)
  (ones : Nat)
  (is_valid : tens ≤ 9 ∧ ones ≤ 9)

/-- The age after 10 years --/
def age_after_10_years (a : Age) : Nat :=
  10 * a.tens + a.ones + 10

/-- Helen's age is the reverse of Ellen's age --/
def is_reverse (helen : Age) (ellen : Age) : Prop :=
  helen.tens = ellen.ones ∧ helen.ones = ellen.tens

/-- In 10 years, Helen will be three times as old as Ellen --/
def future_age_relation (helen : Age) (ellen : Age) : Prop :=
  age_after_10_years helen = 3 * age_after_10_years ellen

/-- The current age difference --/
def age_difference (helen : Age) (ellen : Age) : Int :=
  (10 * helen.tens + helen.ones) - (10 * ellen.tens + ellen.ones)

theorem age_puzzle_solution :
  ∃ (helen ellen : Age),
    is_reverse helen ellen ∧
    future_age_relation helen ellen ∧
    age_difference helen ellen = 54 :=
  sorry

end age_puzzle_solution_l3396_339670


namespace union_of_sets_l3396_339684

theorem union_of_sets : 
  let P : Set ℕ := {1, 2, 3, 4}
  let Q : Set ℕ := {3, 4, 5, 6}
  P ∪ Q = {1, 2, 3, 4, 5, 6} := by
sorry

end union_of_sets_l3396_339684


namespace jeans_cost_theorem_l3396_339642

def total_cost : ℕ := 110
def coat_cost : ℕ := 40
def shoes_cost : ℕ := 30
def num_jeans : ℕ := 2

theorem jeans_cost_theorem :
  ∃ (jeans_cost : ℕ),
    jeans_cost * num_jeans + coat_cost + shoes_cost = total_cost ∧
    jeans_cost = 20 :=
by sorry

end jeans_cost_theorem_l3396_339642


namespace prob_B_given_A_value_l3396_339628

/-- Represents the number of balls in the box -/
def total_balls : ℕ := 10

/-- Represents the number of black balls initially in the box -/
def black_balls : ℕ := 8

/-- Represents the number of red balls initially in the box -/
def red_balls : ℕ := 2

/-- Represents the number of balls each player draws -/
def balls_drawn : ℕ := 2

/-- Calculates the probability of player B drawing 2 black balls given that player A has drawn 2 black balls -/
def prob_B_given_A : ℚ :=
  (Nat.choose (black_balls - balls_drawn) balls_drawn) / (Nat.choose total_balls balls_drawn)

theorem prob_B_given_A_value : prob_B_given_A = 15 / 28 := by
  sorry

end prob_B_given_A_value_l3396_339628


namespace julio_has_more_soda_l3396_339659

/-- Calculates the total liters of soda for a person given the number of orange and grape soda bottles -/
def totalSoda (orangeBottles grapeBottles : ℕ) : ℕ := 2 * (orangeBottles + grapeBottles)

theorem julio_has_more_soda : 
  let julioTotal := totalSoda 4 7
  let mateoTotal := totalSoda 1 3
  julioTotal - mateoTotal = 14 := by
  sorry

end julio_has_more_soda_l3396_339659


namespace hemisphere_base_area_l3396_339629

/-- If the surface area of a hemisphere is 9, then the area of its base is 3 -/
theorem hemisphere_base_area (r : ℝ) (h : 3 * Real.pi * r^2 = 9) : Real.pi * r^2 = 3 := by
  sorry

end hemisphere_base_area_l3396_339629


namespace expression_evaluation_l3396_339617

theorem expression_evaluation (x : ℕ) (h : x = 3) : x^2 + x * (x^(x^2)) = 59058 := by
  sorry

end expression_evaluation_l3396_339617


namespace benny_crayons_l3396_339697

theorem benny_crayons (initial : ℕ) (final : ℕ) (added : ℕ) : 
  initial = 9 → final = 12 → added = final - initial → added = 3 := by
  sorry

end benny_crayons_l3396_339697


namespace factorization_sum_l3396_339685

theorem factorization_sum (x y : ℝ) : 
  ∃ (a b c d e f g h j k : ℤ), 
    125 * x^9 - 216 * y^9 = (a*x + b*y) * (c*x^3 + d*x*y^2 + e*y^3) * (f*x + g*y) * (h*x^3 + j*x*y^2 + k*y^3) ∧
    a + b + c + d + e + f + g + h + j + k = 16 :=
by sorry

end factorization_sum_l3396_339685


namespace cost_increase_percentage_l3396_339606

theorem cost_increase_percentage (initial_cost selling_price new_cost : ℝ) : 
  initial_cost > 0 →
  selling_price = 2.5 * initial_cost →
  new_cost > initial_cost →
  (selling_price - new_cost) / selling_price = 0.552 →
  (new_cost - initial_cost) / initial_cost = 0.12 :=
by sorry

end cost_increase_percentage_l3396_339606


namespace fraction_equivalence_l3396_339614

theorem fraction_equivalence : 
  let original_numerator : ℚ := 4
  let original_denominator : ℚ := 7
  let target_numerator : ℚ := 7
  let target_denominator : ℚ := 9
  let n : ℚ := 13/2
  (original_numerator + n) / (original_denominator + n) = target_numerator / target_denominator :=
by sorry

end fraction_equivalence_l3396_339614


namespace base9_to_base3_conversion_l3396_339687

/-- Converts a digit from base 9 to base 4 -/
def base9To4Digit (d : Nat) : Nat :=
  d / 4 * 10 + d % 4

/-- Converts a number from base 9 to base 4 -/
def base9To4 (n : Nat) : Nat :=
  let d1 := n / 81
  let d2 := (n / 9) % 9
  let d3 := n % 9
  base9To4Digit d1 * 10000 + base9To4Digit d2 * 100 + base9To4Digit d3

/-- Converts a digit from base 4 to base 3 -/
def base4To3Digit (d : Nat) : Nat :=
  d / 3 * 10 + d % 3

/-- Converts a number from base 4 to base 3 -/
def base4To3 (n : Nat) : Nat :=
  let d1 := n / 10000
  let d2 := (n / 100) % 100
  let d3 := n % 100
  base4To3Digit d1 * 100000000 + base4To3Digit d2 * 10000 + base4To3Digit d3

theorem base9_to_base3_conversion :
  base4To3 (base9To4 758) = 01101002000 := by
  sorry

end base9_to_base3_conversion_l3396_339687


namespace grocery_to_gym_speed_l3396_339682

/-- Represents Angelina's walking scenario --/
structure WalkingScenario where
  initial_distance : ℝ
  second_distance : ℝ
  initial_speed : ℝ
  second_speed : ℝ
  third_speed : ℝ
  first_second_time_diff : ℝ
  second_third_time_diff : ℝ

/-- Theorem stating the speed from grocery to gym --/
theorem grocery_to_gym_speed (w : WalkingScenario) 
  (h1 : w.initial_distance = 100)
  (h2 : w.second_distance = 180)
  (h3 : w.second_speed = 2 * w.initial_speed)
  (h4 : w.third_speed = 3 * w.initial_speed)
  (h5 : w.initial_distance / w.initial_speed - w.second_distance / w.second_speed = w.first_second_time_diff)
  (h6 : w.first_second_time_diff = 40)
  (h7 : w.second_third_time_diff = 20) :
  w.second_speed = 1/2 := by
  sorry

end grocery_to_gym_speed_l3396_339682


namespace acute_triangles_in_cuboid_l3396_339641

-- Define a rectangular cuboid
structure RectangularCuboid where
  vertices : Finset (ℝ × ℝ × ℝ)
  vertex_count : vertices.card = 8

-- Define a function to count acute triangles
def count_acute_triangles (rc : RectangularCuboid) : ℕ := sorry

-- Theorem statement
theorem acute_triangles_in_cuboid (rc : RectangularCuboid) :
  count_acute_triangles rc = 8 := by sorry

end acute_triangles_in_cuboid_l3396_339641


namespace inequality_proof_l3396_339602

theorem inequality_proof (x y : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : x + y = 2) :
  x^2 * y^2 * (x^2 + y^2) ≤ 2 := by
  sorry

end inequality_proof_l3396_339602


namespace intersection_complement_equal_set_l3396_339607

def M : Set ℝ := {-1, 0, 1, 3}
def N : Set ℝ := {x : ℝ | x^2 - x - 2 ≥ 0}

theorem intersection_complement_equal_set : M ∩ (Set.univ \ N) = {0, 1} := by
  sorry

end intersection_complement_equal_set_l3396_339607


namespace trailingZeros_100_factorial_l3396_339612

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: The number of trailing zeros in 100! is 24 -/
theorem trailingZeros_100_factorial :
  trailingZeros 100 = 24 := by
  sorry

end trailingZeros_100_factorial_l3396_339612


namespace inverse_proposition_l3396_339632

theorem inverse_proposition (a b : ℝ) : 
  (a = 0 → a * b = 0) ↔ (a * b = 0 → a = 0) :=
sorry

end inverse_proposition_l3396_339632


namespace probability_higher_first_lower_second_l3396_339647

def card_set : Finset ℕ := Finset.range 7

def favorable_outcomes : Finset (ℕ × ℕ) :=
  Finset.filter (fun p => p.1 > p.2) (card_set.product card_set)

theorem probability_higher_first_lower_second :
  (favorable_outcomes.card : ℚ) / (card_set.card * card_set.card) = 3 / 7 := by
  sorry

end probability_higher_first_lower_second_l3396_339647


namespace product_scaling_l3396_339671

theorem product_scaling (a b c : ℝ) (h : (268 : ℝ) * 74 = 19732) :
  2.68 * 0.74 = 1.9732 := by
  sorry

end product_scaling_l3396_339671


namespace curve_and_circle_properties_l3396_339621

-- Define the points and vectors
def E : ℝ × ℝ := (-2, 0)
def F : ℝ × ℝ := (2, 0)
def A : ℝ × ℝ := (2, 1)

-- Define the dot product of vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the curve C
def C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the condition for point M on curve C
def M_condition (x y : ℝ) : Prop :=
  dot_product (x + 2, y) (x - 2, y) = -3

-- Define the point P and the tangent condition
def P (a b : ℝ) : ℝ × ℝ := (a, b)
def tangent_condition (a b : ℝ) : Prop :=
  ∃ (x y : ℝ), C x y ∧ (a - x)^2 + (b - y)^2 = (a - 2)^2 + (b - 1)^2

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  (x - 6/5)^2 + (y - 3/5)^2 = (3 * Real.sqrt 5 / 5 - 1)^2

-- State the theorem
theorem curve_and_circle_properties :
  ∀ (x y a b : ℝ),
    C x y ∧
    M_condition x y ∧
    tangent_condition a b →
    (∀ (u v : ℝ), C u v ↔ (u - 1)^2 + v^2 = 1) ∧
    (∀ (r : ℝ), r > 0 → 
      (∀ (u v : ℝ), (u - a)^2 + (v - b)^2 = r^2 → ¬(C u v)) →
      r ≥ 3 * Real.sqrt 5 / 5 - 1) ∧
    circle_equation a b :=
by sorry

end curve_and_circle_properties_l3396_339621


namespace num_shortest_paths_A_to_B_l3396_339639

/-- Represents a point on the grid --/
structure Point where
  x : ℕ
  y : ℕ

/-- Represents the road network --/
def RoadNetwork : Type := Set (Point × Point)

/-- Calculates the number of shortest paths between two points on a given road network --/
def numShortestPaths (start finish : Point) (network : RoadNetwork) : ℕ :=
  sorry

/-- The specific road network described in the problem --/
def specificNetwork : RoadNetwork :=
  sorry

/-- The start point A --/
def pointA : Point :=
  ⟨0, 0⟩

/-- The end point B --/
def pointB : Point :=
  ⟨11, 8⟩

/-- Theorem stating that the number of shortest paths from A to B on the specific network is 22023 --/
theorem num_shortest_paths_A_to_B :
  numShortestPaths pointA pointB specificNetwork = 22023 :=
by sorry

end num_shortest_paths_A_to_B_l3396_339639


namespace parabola_equation_and_range_l3396_339654

/-- A parabola with equation y = x^2 - 2mx + m^2 - 1 -/
def Parabola (m : ℝ) (x y : ℝ) : Prop :=
  y = x^2 - 2*m*x + m^2 - 1

/-- The parabola intersects the y-axis at (0, 3) -/
def IntersectsYAxisAt3 (m : ℝ) : Prop :=
  Parabola m 0 3

/-- The vertex of the parabola is in the fourth quadrant -/
def VertexInFourthQuadrant (m : ℝ) : Prop :=
  let x_vertex := m  -- x-coordinate of vertex is m for this parabola
  let y_vertex := -1  -- y-coordinate of vertex is -1 for this parabola
  x_vertex > 0 ∧ y_vertex < 0

theorem parabola_equation_and_range (m : ℝ) 
  (h1 : IntersectsYAxisAt3 m) 
  (h2 : VertexInFourthQuadrant m) :
  (∀ x y, Parabola m x y ↔ y = x^2 - 4*x + 3) ∧
  (∀ x y, 0 ≤ x ∧ x ≤ 3 ∧ Parabola m x y → -1 ≤ y ∧ y ≤ 3) := by
  sorry

end parabola_equation_and_range_l3396_339654


namespace printing_press_completion_time_l3396_339620

-- Define the start time (9:00 AM)
def start_time : ℕ := 9

-- Define the time when half the order is completed (12:00 PM)
def half_time : ℕ := 12

-- Define the time to complete half the order
def half_duration : ℕ := half_time - start_time

-- Theorem: The printing press will finish the entire order at 3:00 PM
theorem printing_press_completion_time :
  start_time + 2 * half_duration = 15 := by
  sorry

end printing_press_completion_time_l3396_339620


namespace incircle_radius_eq_area_div_semiperimeter_l3396_339655

/-- Triangle DEF with given angles and side length --/
structure TriangleDEF where
  D : ℝ
  E : ℝ
  F : ℝ
  DE : ℝ
  angle_D_eq : D = 75
  angle_E_eq : E = 45
  angle_F_eq : F = 60
  side_DE_eq : DE = 20

/-- The radius of the incircle of triangle DEF --/
def incircle_radius (t : TriangleDEF) : ℝ := sorry

/-- The semi-perimeter of triangle DEF --/
def semi_perimeter (t : TriangleDEF) : ℝ := sorry

/-- The area of triangle DEF --/
def triangle_area (t : TriangleDEF) : ℝ := sorry

/-- Theorem: The radius of the incircle is equal to the area divided by the semi-perimeter --/
theorem incircle_radius_eq_area_div_semiperimeter (t : TriangleDEF) :
  incircle_radius t = triangle_area t / semi_perimeter t := by sorry

end incircle_radius_eq_area_div_semiperimeter_l3396_339655


namespace total_fruits_is_41_l3396_339665

/-- The number of oranges Mike received -/
def mike_oranges : ℕ := 3

/-- The number of apples Matt received -/
def matt_apples : ℕ := 2 * mike_oranges

/-- The number of bananas Mark received -/
def mark_bananas : ℕ := mike_oranges + matt_apples

/-- The number of grapes Mary received -/
def mary_grapes : ℕ := mike_oranges + matt_apples + mark_bananas + 5

/-- The total number of fruits received by all four children -/
def total_fruits : ℕ := mike_oranges + matt_apples + mark_bananas + mary_grapes

theorem total_fruits_is_41 : total_fruits = 41 := by
  sorry

end total_fruits_is_41_l3396_339665


namespace final_position_on_number_line_final_position_is_28_l3396_339613

/-- Given a number line where the distance from 0 to 40 is divided into 10 equal steps,
    if a person moves forward 8 steps and then back 1 step, their final position will be 28. -/
theorem final_position_on_number_line : ℝ → Prop :=
  fun final_position =>
    let total_distance : ℝ := 40
    let total_steps : ℕ := 10
    let step_size : ℝ := total_distance / total_steps
    let forward_steps : ℕ := 8
    let backward_steps : ℕ := 1
    final_position = (forward_steps - backward_steps : ℕ) * step_size

theorem final_position_is_28 : final_position_on_number_line 28 := by
  sorry

#check final_position_is_28

end final_position_on_number_line_final_position_is_28_l3396_339613


namespace sum_of_squares_of_roots_l3396_339603

theorem sum_of_squares_of_roots (x : ℝ) : 
  x^2 + 8*x - 12 = 0 → ∃ r₁ r₂ : ℝ, r₁ + r₂ = -8 ∧ r₁ * r₂ = -12 ∧ r₁^2 + r₂^2 = 88 := by
  sorry

end sum_of_squares_of_roots_l3396_339603


namespace triangle_area_l3396_339674

/-- A triangle with sides of length 9, 40, and 41 has an area of 180. -/
theorem triangle_area (a b c : ℝ) (ha : a = 9) (hb : b = 40) (hc : c = 41) : 
  (1/2) * a * b = 180 := by
  sorry

end triangle_area_l3396_339674


namespace point_on_line_max_product_l3396_339690

/-- Given points A(a,b) and B(4,c) lie on the line y = kx + 3, where k is a constant and k ≠ 0,
    and the maximum value of ab is 9, then c = 2. -/
theorem point_on_line_max_product (k a b c : ℝ) : 
  k ≠ 0 → 
  b = k * a + 3 → 
  c = k * 4 + 3 → 
  (∀ x y, x * y ≤ 9 → a * b ≥ x * y) → 
  c = 2 := by
sorry

end point_on_line_max_product_l3396_339690


namespace rotation_result_l3396_339660

-- Define the shapes
inductive Shape
  | Rectangle
  | SmallCircle
  | Pentagon

-- Define the positions
inductive Position
  | Top
  | LeftBottom
  | RightBottom

-- Define the circular plane
structure CircularPlane where
  shapes : List Shape
  positions : List Position
  arrangement : Shape → Position

-- Define the rotation
def rotate150 (plane : CircularPlane) : CircularPlane := sorry

-- Theorem statement
theorem rotation_result (plane : CircularPlane) :
  plane.arrangement Shape.Rectangle = Position.Top →
  plane.arrangement Shape.SmallCircle = Position.LeftBottom →
  plane.arrangement Shape.Pentagon = Position.RightBottom →
  (rotate150 plane).arrangement Shape.SmallCircle = Position.Top := by
  sorry

end rotation_result_l3396_339660


namespace max_toys_purchasable_l3396_339676

theorem max_toys_purchasable (initial_amount : ℚ) (game_cost : ℚ) (book_cost : ℚ) (toy_cost : ℚ) :
  initial_amount = 57.45 →
  game_cost = 26.89 →
  book_cost = 12.37 →
  toy_cost = 6 →
  ⌊(initial_amount - game_cost - book_cost) / toy_cost⌋ = 3 :=
by
  sorry

end max_toys_purchasable_l3396_339676


namespace smallest_sum_of_reciprocal_sum_l3396_339646

theorem smallest_sum_of_reciprocal_sum (x y : ℕ+) 
  (h1 : x ≠ y) 
  (h2 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 10) : 
  (∀ a b : ℕ+, a ≠ b → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 10 → (x : ℤ) + y ≤ (a : ℤ) + b) ∧ 
  ∃ p q : ℕ+, p ≠ q ∧ (1 : ℚ) / p + (1 : ℚ) / q = (1 : ℚ) / 10 ∧ (p : ℤ) + q = 45 :=
sorry

end smallest_sum_of_reciprocal_sum_l3396_339646


namespace triangle_parallelogram_properties_l3396_339625

/-- A triangle with a parallelogram inscribed in it. -/
structure TriangleWithParallelogram where
  /-- The length of the first side of the triangle -/
  side1 : ℝ
  /-- The length of the second side of the triangle -/
  side2 : ℝ
  /-- The length of the first side of the parallelogram -/
  para_side1 : ℝ
  /-- Assumption that the first side of the triangle is 9 -/
  h1 : side1 = 9
  /-- Assumption that the second side of the triangle is 15 -/
  h2 : side2 = 15
  /-- Assumption that the first side of the parallelogram is 6 -/
  h3 : para_side1 = 6
  /-- Assumption that the parallelogram is inscribed in the triangle -/
  h4 : para_side1 ≤ side1 ∧ para_side1 ≤ side2
  /-- Assumption that the diagonals of the parallelogram are parallel to the sides of the triangle -/
  h5 : True  -- This is a placeholder as we can't directly represent this geometrical property

/-- The theorem stating the properties of the triangle and parallelogram -/
theorem triangle_parallelogram_properties (tp : TriangleWithParallelogram) :
  ∃ (para_side2 triangle_side3 : ℝ),
    para_side2 = 4 * Real.sqrt 2 ∧
    triangle_side3 = 18 :=
  sorry

end triangle_parallelogram_properties_l3396_339625


namespace quadratic_minimum_change_l3396_339600

/-- Given a quadratic polynomial f(x) = ax^2 + bx + c, 
    if adding x^2 to f(x) increases its minimum value by 1
    and subtracting x^2 from f(x) decreases its minimum value by 3,
    then adding 2x^2 to f(x) will increase its minimum value by 3/2. -/
theorem quadratic_minimum_change 
  (f : ℝ → ℝ) 
  (a b c : ℝ) 
  (h_quad : ∀ x, f x = a * x^2 + b * x + c)
  (h_pos : a > 0)
  (h_add : (- b^2 / (4 * (a + 1)) + c) - (- b^2 / (4 * a) + c) = 1)
  (h_sub : (- b^2 / (4 * a) + c) - (- b^2 / (4 * (a - 1)) + c) = 3) :
  (- b^2 / (4 * a) + c) - (- b^2 / (4 * (a + 2)) + c) = 3/2 := by
  sorry


end quadratic_minimum_change_l3396_339600


namespace arctan_sum_special_case_l3396_339680

theorem arctan_sum_special_case (a b : ℝ) : 
  a = 1/3 → (a + 1) * (b + 1) = 5/2 → Real.arctan a + Real.arctan b = Real.arctan (29/17) := by
  sorry

end arctan_sum_special_case_l3396_339680


namespace lost_weights_solution_l3396_339681

/-- Represents the types of weights in the set -/
inductive WeightType
  | Light : WeightType  -- 43g
  | Medium : WeightType -- 57g
  | Heavy : WeightType  -- 70g

/-- The weight in grams for each type -/
def weight_value (w : WeightType) : ℕ :=
  match w with
  | WeightType.Light => 43
  | WeightType.Medium => 57
  | WeightType.Heavy => 70

/-- The total number of each type of weight initially -/
def initial_count : ℕ := sorry

/-- The total weight of all weights initially -/
def initial_total_weight : ℕ := initial_count * (weight_value WeightType.Light + weight_value WeightType.Medium + weight_value WeightType.Heavy)

/-- The remaining weight after some weights were lost -/
def remaining_weight : ℕ := 20172

/-- Represents the number of weights lost for each type -/
structure LostWeights :=
  (light : ℕ)
  (medium : ℕ)
  (heavy : ℕ)

/-- The total number of weights lost -/
def total_lost (lw : LostWeights) : ℕ := lw.light + lw.medium + lw.heavy

/-- The total weight of lost weights -/
def lost_weight (lw : LostWeights) : ℕ :=
  lw.light * weight_value WeightType.Light +
  lw.medium * weight_value WeightType.Medium +
  lw.heavy * weight_value WeightType.Heavy

/-- Theorem stating that the only solution is losing 4 weights of 57g each -/
theorem lost_weights_solution :
  ∃! lw : LostWeights,
    total_lost lw < 5 ∧
    initial_total_weight - lost_weight lw = remaining_weight ∧
    lw.light = 0 ∧ lw.medium = 4 ∧ lw.heavy = 0 :=
  sorry

end lost_weights_solution_l3396_339681


namespace gcf_lcm_problem_l3396_339673

theorem gcf_lcm_problem : Nat.gcd (Nat.lcm 18 30) (Nat.lcm 21 28) = 6 := by
  sorry

end gcf_lcm_problem_l3396_339673


namespace parent_selection_theorem_l3396_339657

def total_parents : ℕ := 12
def num_couples : ℕ := 6
def parents_to_select : ℕ := 4

theorem parent_selection_theorem :
  let ways_to_select_couple := num_couples
  let remaining_parents := total_parents - 2
  let ways_to_select_others := (remaining_parents.choose (parents_to_select - 2))
  ways_to_select_couple * ways_to_select_others = 240 := by
  sorry

end parent_selection_theorem_l3396_339657


namespace butane_molecular_weight_l3396_339663

/-- The molecular weight of Butane in grams per mole. -/
def molecular_weight_butane : ℝ := 65

/-- The number of moles used in the given condition. -/
def given_moles : ℝ := 4

/-- The total molecular weight of the given moles in grams. -/
def given_total_weight : ℝ := 260

/-- Theorem stating that the molecular weight of Butane is 65 grams/mole. -/
theorem butane_molecular_weight : 
  molecular_weight_butane = given_total_weight / given_moles := by
  sorry

end butane_molecular_weight_l3396_339663


namespace max_rooms_needed_l3396_339692

/-- Represents a group of football fans -/
structure FanGroup where
  team : Fin 3
  gender : Bool
  count : Nat

/-- The maximum number of fans that can be accommodated in one room -/
def max_fans_per_room : Nat := 3

/-- The total number of fans -/
def total_fans : Nat := 100

/-- Calculate the number of rooms needed for a group of fans -/
def rooms_needed (group : FanGroup) : Nat :=
  (group.count + max_fans_per_room - 1) / max_fans_per_room

/-- The main theorem stating the maximum number of rooms needed -/
theorem max_rooms_needed (fans : List FanGroup) 
  (h1 : fans.length = 6)
  (h2 : fans.all (λ g ↦ g.count > 0))
  (h3 : (fans.map FanGroup.count).sum = total_fans) :
  (fans.map rooms_needed).sum ≤ 37 := by
  sorry

end max_rooms_needed_l3396_339692


namespace cube_root_of_a_times_sqrt_a_l3396_339664

theorem cube_root_of_a_times_sqrt_a (a : ℝ) (ha : a > 0) : 
  (a * a^(1/2))^(1/3) = a^(1/2) := by
  sorry

end cube_root_of_a_times_sqrt_a_l3396_339664


namespace problem_statement_l3396_339672

/-- Given two expressions A and B in terms of a and b, prove that A + 2B has a specific form
    and that when it's independent of b, a has a specific value. -/
theorem problem_statement (a b : ℝ) : 
  let A := 2*a^2 + 3*a*b - 2*b - 1
  let B := -a^2 - a*b + 1
  (A + 2*B = a*b - 2*b + 1) ∧ 
  (∀ b, A + 2*B = a*b - 2*b + 1 → a = 2) := by
  sorry

end problem_statement_l3396_339672


namespace farm_oxen_count_l3396_339658

/-- Represents the daily fodder consumption of one buffalo -/
def B : ℝ := sorry

/-- Represents the number of oxen on the farm -/
def O : ℕ := sorry

/-- The total amount of fodder available on the farm -/
def total_fodder : ℝ := sorry

theorem farm_oxen_count : O = 8 := by
  have h1 : 3 * B = 4 * (3/4 * B) := sorry
  have h2 : 3 * B = 2 * (3/2 * B) := sorry
  have h3 : total_fodder = (33 * B + 3/2 * O * B) * 48 := sorry
  have h4 : total_fodder = (108 * B + 3/2 * O * B) * 18 := sorry
  sorry

end farm_oxen_count_l3396_339658


namespace book_weight_l3396_339608

theorem book_weight (total_weight : ℝ) (num_books : ℕ) (h1 : total_weight = 42) (h2 : num_books = 14) :
  total_weight / num_books = 3 := by
sorry

end book_weight_l3396_339608


namespace number_line_positions_l3396_339689

theorem number_line_positions (x : ℝ) : 
  (x > 0 → (0 = -4*x + 1/4 * (12*x - (-4*x)) ∧ x = 0 + 1/4 * (4*x - 0))) ∧
  (x < 0 → (0 = 12*x + 3/4 * (-4*x - 12*x) ∧ x = 4*x + 3/4 * (0 - 4*x))) :=
by sorry

end number_line_positions_l3396_339689


namespace inequality_implication_l3396_339631

theorem inequality_implication (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : a - d > b - c := by
  sorry

end inequality_implication_l3396_339631


namespace monkey_feeding_problem_l3396_339601

theorem monkey_feeding_problem :
  ∀ (x : ℝ),
    (3/4 * x + 2 = 4/3 * (x - 2)) →
    (3/4 * x + x = 14) :=
by
  sorry

end monkey_feeding_problem_l3396_339601


namespace pencils_on_desk_l3396_339656

theorem pencils_on_desk (drawer : ℕ) (added : ℕ) (total : ℕ) (initial : ℕ) : 
  drawer = 43 → added = 16 → total = 78 → initial + added + drawer = total → initial = 19 := by
sorry

end pencils_on_desk_l3396_339656


namespace least_sum_of_exponents_for_2023_l3396_339624

def is_sum_of_distinct_powers_of_two (n : ℕ) (exponents : List ℕ) : Prop :=
  n = (exponents.map (fun e => 2^e)).sum ∧ exponents.Nodup

theorem least_sum_of_exponents_for_2023 :
  ∃ (exponents : List ℕ),
    is_sum_of_distinct_powers_of_two 2023 exponents ∧
    ∀ (other_exponents : List ℕ),
      is_sum_of_distinct_powers_of_two 2023 other_exponents →
      exponents.sum ≤ other_exponents.sum ∧
      exponents.sum = 48 :=
sorry

end least_sum_of_exponents_for_2023_l3396_339624


namespace triangle_movement_l3396_339686

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Check if a triangle is isosceles and right-angled at C -/
def isIsoscelesRightTriangle (t : Triangle) : Prop :=
  (t.A.x - t.C.x)^2 + (t.A.y - t.C.y)^2 = (t.B.x - t.C.x)^2 + (t.B.y - t.C.y)^2 ∧
  (t.A.x - t.C.x) * (t.B.x - t.C.x) + (t.A.y - t.C.y) * (t.B.y - t.C.y) = 0

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The main theorem -/
theorem triangle_movement (t : Triangle) (a b : Line) (c : ℝ) :
  isIsoscelesRightTriangle t →
  (∀ (t' : Triangle), isIsoscelesRightTriangle t' →
    pointOnLine t'.A a → pointOnLine t'.B b →
    (t'.A.x - t'.B.x)^2 + (t'.A.y - t'.B.y)^2 = c^2) →
  ∃ (l : Line),
    (l.a = 1 ∧ l.b = 1 ∧ l.c = 0) ∨ (l.a = 1 ∧ l.b = -1 ∧ l.c = 0) ∧
    ∀ (p : Point), pointOnLine p l →
      -c/2 ≤ p.x ∧ p.x ≤ c/2 →
      ∃ (t' : Triangle), isIsoscelesRightTriangle t' ∧
        pointOnLine t'.A a ∧ pointOnLine t'.B b ∧
        t'.C = p :=
sorry

end triangle_movement_l3396_339686


namespace chess_swimming_percentage_l3396_339616

theorem chess_swimming_percentage (total_students : ℕ) 
  (chess_percentage : ℚ) (swimming_students : ℕ) :
  total_students = 1000 →
  chess_percentage = 1/5 →
  swimming_students = 20 →
  (swimming_students : ℚ) / (chess_percentage * total_students) * 100 = 10 :=
by sorry

end chess_swimming_percentage_l3396_339616


namespace imaginary_part_of_complex_fraction_l3396_339649

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := (2 + Complex.I) / Complex.I
  Complex.im z = -2 := by sorry

end imaginary_part_of_complex_fraction_l3396_339649


namespace inequality_system_solution_set_l3396_339694

theorem inequality_system_solution_set :
  let S := {x : ℝ | x + 8 < 4*x - 1 ∧ (1/2)*x ≥ 4 - (3/2)*x}
  S = {x : ℝ | x > 3} :=
by sorry

end inequality_system_solution_set_l3396_339694


namespace log_comparison_l3396_339666

theorem log_comparison : 
  Real.log 6 / Real.log 3 > Real.log 10 / Real.log 5 ∧ 
  Real.log 10 / Real.log 5 > Real.log 14 / Real.log 7 := by
  sorry

end log_comparison_l3396_339666


namespace highest_a_divisible_by_8_l3396_339688

def is_divisible_by_8 (n : ℕ) : Prop := n % 8 = 0

def last_three_digits (n : ℕ) : ℕ := n % 1000

def construct_number (a : ℕ) : ℕ := 365000 + a * 100 + 16

theorem highest_a_divisible_by_8 :
  ∀ a : ℕ, a ≤ 9 →
    (is_divisible_by_8 (construct_number a) ↔ a ≤ 8) ∧
    (a = 8 → is_divisible_by_8 (construct_number a)) ∧
    (a = 9 → ¬is_divisible_by_8 (construct_number a)) :=
sorry

end highest_a_divisible_by_8_l3396_339688


namespace triangle_length_l3396_339635

-- Define the curve y = x^3
def curve (x : ℝ) : ℝ := x^3

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the conditions of the problem
structure ProblemConditions where
  triangle : Triangle
  on_curve : 
    curve triangle.A.1 = triangle.A.2 ∧
    curve triangle.B.1 = triangle.B.2 ∧
    curve triangle.C.1 = triangle.C.2
  A_at_origin : triangle.A = (0, 0)
  BC_parallel_x : triangle.B.2 = triangle.C.2
  area : ℝ

-- Define the theorem
theorem triangle_length (conditions : ProblemConditions) 
  (h : conditions.area = 125) : 
  let BC_length := |conditions.triangle.C.1 - conditions.triangle.B.1|
  BC_length = 10 := by sorry

end triangle_length_l3396_339635


namespace first_day_distance_l3396_339650

theorem first_day_distance (total_distance : ℝ) (days : ℕ) (ratio : ℝ) 
  (h1 : total_distance = 378)
  (h2 : days = 6)
  (h3 : ratio = 1/2) :
  let first_day := total_distance * (1 - ratio) / (1 - ratio^days)
  first_day = 192 := by sorry

end first_day_distance_l3396_339650


namespace chairs_produced_in_six_hours_l3396_339677

/-- The number of chairs produced by a group of workers over a given time period -/
def chairs_produced (num_workers : ℕ) (chairs_per_worker_per_hour : ℕ) (additional_chairs : ℕ) (hours : ℕ) : ℕ :=
  num_workers * chairs_per_worker_per_hour * hours + additional_chairs

/-- Theorem stating that 3 workers producing 4 chairs per hour, with an additional chair every 6 hours, produce 73 chairs in 6 hours -/
theorem chairs_produced_in_six_hours :
  chairs_produced 3 4 1 6 = 73 := by
  sorry

end chairs_produced_in_six_hours_l3396_339677


namespace friend_balloon_count_l3396_339691

theorem friend_balloon_count (my_balloons : ℕ) (difference : ℕ) : my_balloons = 7 → difference = 2 → my_balloons - difference = 5 := by
  sorry

end friend_balloon_count_l3396_339691


namespace car_sale_profit_ratio_l3396_339696

theorem car_sale_profit_ratio (c₁ c₂ : ℝ) (h : c₁ > 0 ∧ c₂ > 0) :
  (1.1 * c₁ + 0.9 * c₂ - (c₁ + c₂)) / (c₁ + c₂) = 0.01 →
  c₂ = (9 / 11) * c₁ := by
  sorry

end car_sale_profit_ratio_l3396_339696


namespace only_proposition_3_true_l3396_339693

theorem only_proposition_3_true : 
  (¬∀ (a b c : ℝ), a > b ∧ c ≠ 0 → a * c > b * c) ∧ 
  (¬∀ (a b c : ℝ), a > b → a * c^2 > b * c^2) ∧ 
  (∀ (a b c : ℝ), a * c^2 > b * c^2 → a > b) ∧ 
  (¬∀ (a b : ℝ), a > b → 1 / a < 1 / b) ∧ 
  (¬∀ (a b c d : ℝ), a > b ∧ b > 0 ∧ c > d → a * c > b * d) :=
by sorry

end only_proposition_3_true_l3396_339693


namespace rhombus_area_l3396_339695

theorem rhombus_area (d₁ d₂ : ℝ) : 
  d₁^2 - 14*d₁ + 48 = 0 → 
  d₂^2 - 14*d₂ + 48 = 0 → 
  d₁ ≠ d₂ →
  (d₁ * d₂) / 2 = 24 := by
sorry

end rhombus_area_l3396_339695


namespace inequality_proof_l3396_339679

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a / (a^4 + b^2) + b / (a^2 + b^4) ≤ 1 / (a * b) := by
  sorry

end inequality_proof_l3396_339679


namespace train_speed_calculation_l3396_339609

-- Define the given constants
def train_length : ℝ := 390  -- in meters
def man_speed : ℝ := 2       -- in km/h
def crossing_time : ℝ := 52  -- in seconds

-- Define the theorem
theorem train_speed_calculation :
  ∃ (train_speed : ℝ),
    train_speed > 0 ∧
    train_speed = 25 ∧
    (train_speed + man_speed) * (crossing_time / 3600) = train_length / 1000 :=
by sorry

end train_speed_calculation_l3396_339609


namespace always_satisfies_condition_l3396_339634

-- Define the set of colors
inductive Color
| Red
| Blue
| Green
| Yellow

-- Define a point with a color
structure ColoredPoint where
  color : Color

-- Define a colored line segment
structure ColoredSegment where
  endpoint1 : ColoredPoint
  endpoint2 : ColoredPoint
  color : Color

-- Define the coloring property for segments
def validSegmentColoring (segment : ColoredSegment) : Prop :=
  segment.color = segment.endpoint1.color ∨ segment.color = segment.endpoint2.color

-- Define the configuration of points and segments
structure Configuration where
  points : Fin 4 → ColoredPoint
  segments : Fin 6 → ColoredSegment
  allColorsUsed : ∀ c : Color, ∃ s : Fin 6, (segments s).color = c
  distinctPointColors : ∀ i j : Fin 4, i ≠ j → (points i).color ≠ (points j).color
  validSegments : ∀ s : Fin 6, validSegmentColoring (segments s)

-- Define the conditions to be satisfied
def satisfiesConditionA (config : Configuration) (p : Fin 4) : Prop :=
  ∃ s1 s2 s3 : Fin 6,
    (config.segments s1).color = Color.Red ∧
    (config.segments s2).color = Color.Blue ∧
    (config.segments s3).color = Color.Green ∧
    ((config.segments s1).endpoint1 = config.points p ∨ (config.segments s1).endpoint2 = config.points p) ∧
    ((config.segments s2).endpoint1 = config.points p ∨ (config.segments s2).endpoint2 = config.points p) ∧
    ((config.segments s3).endpoint1 = config.points p ∨ (config.segments s3).endpoint2 = config.points p)

def satisfiesConditionB (config : Configuration) (p : Fin 4) : Prop :=
  ∃ s1 s2 s3 : Fin 6,
    (config.segments s1).color = Color.Red ∧
    (config.segments s2).color = Color.Blue ∧
    (config.segments s3).color = Color.Green ∧
    (config.segments s1).endpoint1 ≠ config.points p ∧
    (config.segments s1).endpoint2 ≠ config.points p ∧
    (config.segments s2).endpoint1 ≠ config.points p ∧
    (config.segments s2).endpoint2 ≠ config.points p ∧
    (config.segments s3).endpoint1 ≠ config.points p ∧
    (config.segments s3).endpoint2 ≠ config.points p

-- The main theorem
theorem always_satisfies_condition (config : Configuration) :
  ∃ p : Fin 4, satisfiesConditionA config p ∨ satisfiesConditionB config p := by
  sorry

end always_satisfies_condition_l3396_339634


namespace arithmetic_sequence_problem_l3396_339699

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (b : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_a2 : a 2 = 2)
  (h_a4 : a 4 = 4)
  (h_b : ∀ n, b n = 2^(a n)) :
  (∀ n, a n = n) ∧ (b 1 + b 2 + b 3 + b 4 + b 5 = 62) := by
  sorry

end arithmetic_sequence_problem_l3396_339699


namespace infinite_geometric_series_sum_specific_geometric_series_sum_l3396_339651

/-- The sum of an infinite geometric series with first term a and common ratio r is a / (1 - r),
    given that |r| < 1 -/
theorem infinite_geometric_series_sum (a r : ℚ) (h : |r| < 1) :
  (∑' n, a * r^n) = a / (1 - r) :=
sorry

/-- The sum of the specific infinite geometric series is 10/9 -/
theorem specific_geometric_series_sum :
  let a : ℚ := 5/3
  let r : ℚ := -1/2
  (∑' n, a * r^n) = 10/9 := by
  sorry

end infinite_geometric_series_sum_specific_geometric_series_sum_l3396_339651
