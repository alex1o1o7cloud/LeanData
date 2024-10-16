import Mathlib

namespace NUMINAMATH_CALUDE_square_area_proof_l1559_155961

-- Define the length of the longer side of the smaller rectangle
def longer_side : ℝ := 6

-- Define the ratio between longer and shorter sides
def ratio : ℝ := 3

-- Define the area of the square WXYZ
def square_area : ℝ := 144

-- Theorem statement
theorem square_area_proof :
  let shorter_side := longer_side / ratio
  let square_side := 2 * longer_side
  square_side ^ 2 = square_area :=
by sorry

end NUMINAMATH_CALUDE_square_area_proof_l1559_155961


namespace NUMINAMATH_CALUDE_smallest_n_for_roots_of_unity_l1559_155906

/-- The polynomial z^5 - z^3 + z -/
def f (z : ℂ) : ℂ := z^5 - z^3 + z

/-- n-th root of unity -/
def is_nth_root_of_unity (z : ℂ) (n : ℕ) : Prop := z^n = 1

/-- All roots of f are n-th roots of unity -/
def all_roots_are_nth_roots_of_unity (n : ℕ) : Prop :=
  ∀ z : ℂ, f z = 0 → is_nth_root_of_unity z n

/-- 12 is the smallest positive integer n such that all roots of f are n-th roots of unity -/
theorem smallest_n_for_roots_of_unity :
  (all_roots_are_nth_roots_of_unity 12) ∧
  (∀ m : ℕ, 0 < m → m < 12 → ¬(all_roots_are_nth_roots_of_unity m)) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_roots_of_unity_l1559_155906


namespace NUMINAMATH_CALUDE_employee_share_l1559_155942

def total_profit : ℝ := 50
def num_employees : ℕ := 9
def self_percentage : ℝ := 0.1

theorem employee_share : 
  (total_profit - self_percentage * total_profit) / num_employees = 5 := by
sorry

end NUMINAMATH_CALUDE_employee_share_l1559_155942


namespace NUMINAMATH_CALUDE_coefficient_of_x_l1559_155938

/-- The coefficient of x in the simplified expression 5(2x - 3) + 7(10 - 3x^2 + 2x) - 9(4x - 2) is -12 -/
theorem coefficient_of_x (x : ℝ) : 
  let expr := 5*(2*x - 3) + 7*(10 - 3*x^2 + 2*x) - 9*(4*x - 2)
  ∃ (a b c : ℝ), expr = a*x^2 + (-12)*x + b + c := by
sorry

end NUMINAMATH_CALUDE_coefficient_of_x_l1559_155938


namespace NUMINAMATH_CALUDE_bread_cost_l1559_155964

def total_money : ℝ := 60
def celery_cost : ℝ := 5
def cereal_original_cost : ℝ := 12
def cereal_discount : ℝ := 0.5
def milk_original_cost : ℝ := 10
def milk_discount : ℝ := 0.1
def potato_cost : ℝ := 1
def potato_quantity : ℕ := 6
def money_left_for_coffee : ℝ := 26

theorem bread_cost : 
  total_money - 
  (celery_cost + 
   cereal_original_cost * (1 - cereal_discount) + 
   milk_original_cost * (1 - milk_discount) + 
   potato_cost * potato_quantity + 
   money_left_for_coffee) = 8 := by sorry

end NUMINAMATH_CALUDE_bread_cost_l1559_155964


namespace NUMINAMATH_CALUDE_root_in_interval_l1559_155944

def f (x : ℝ) := x^3 - 2*x - 5

theorem root_in_interval :
  (f 2 < 0) →
  (f 3 > 0) →
  (f 2.5 > 0) →
  ∃ x, x ∈ Set.Ioo 2 2.5 ∧ f x = 0 :=
by sorry

end NUMINAMATH_CALUDE_root_in_interval_l1559_155944


namespace NUMINAMATH_CALUDE_geometric_arithmetic_ratio_l1559_155935

/-- A geometric sequence with common ratio q -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

/-- Three terms form an arithmetic sequence -/
def arithmetic_sequence (x y z : ℝ) : Prop :=
  y - x = z - y

theorem geometric_arithmetic_ratio (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q →
  arithmetic_sequence (a 4) (a 5) (a 6) →
  q = 1 ∨ q = -2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_ratio_l1559_155935


namespace NUMINAMATH_CALUDE_repair_cost_is_5000_l1559_155925

/-- Calculates the repair cost of a machine given its purchase price, transportation charges,
    profit percentage, and final selling price. -/
def repair_cost (purchase_price : ℤ) (transportation_charges : ℤ) (profit_percentage : ℚ)
                (selling_price : ℤ) : ℚ :=
  ((selling_price : ℚ) - (1 + profit_percentage) * ((purchase_price + transportation_charges) : ℚ)) /
  (1 + profit_percentage)

/-- Theorem stating that the repair cost is 5000 given the specific conditions -/
theorem repair_cost_is_5000 :
  repair_cost 13000 1000 (1/2) 28500 = 5000 := by
  sorry

end NUMINAMATH_CALUDE_repair_cost_is_5000_l1559_155925


namespace NUMINAMATH_CALUDE_logan_corn_purchase_l1559_155967

/-- Proves that Logan bought 15.0 pounds of corn given the problem conditions -/
theorem logan_corn_purchase 
  (corn_price : ℝ) 
  (bean_price : ℝ) 
  (total_weight : ℝ) 
  (total_cost : ℝ) 
  (h1 : corn_price = 1.20)
  (h2 : bean_price = 0.60)
  (h3 : total_weight = 30)
  (h4 : total_cost = 27.00) : 
  ∃ (corn_weight : ℝ) (bean_weight : ℝ),
    corn_weight + bean_weight = total_weight ∧ 
    corn_price * corn_weight + bean_price * bean_weight = total_cost ∧ 
    corn_weight = 15.0 := by
  sorry

end NUMINAMATH_CALUDE_logan_corn_purchase_l1559_155967


namespace NUMINAMATH_CALUDE_equation_equivalence_l1559_155980

theorem equation_equivalence (x : ℝ) : 2 * (x + 1) = x + 7 ↔ x = 5 := by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l1559_155980


namespace NUMINAMATH_CALUDE_race_time_differences_l1559_155943

def race_distance : ℝ := 10
def john_speed : ℝ := 15
def alice_time : ℝ := 48
def bob_time : ℝ := 52
def charlie_time : ℝ := 55

theorem race_time_differences :
  let john_time := race_distance / john_speed * 60
  (alice_time - john_time = 8) ∧
  (bob_time - john_time = 12) ∧
  (charlie_time - john_time = 15) := by
  sorry

end NUMINAMATH_CALUDE_race_time_differences_l1559_155943


namespace NUMINAMATH_CALUDE_sqrt_calculations_l1559_155911

theorem sqrt_calculations : 
  (2 * Real.sqrt 12 + Real.sqrt 75 - 12 * Real.sqrt (1/3) = 5 * Real.sqrt 3) ∧
  (6 * Real.sqrt (8/5) / (2 * Real.sqrt 2) * (-1/2 * Real.sqrt 60) = -6 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_calculations_l1559_155911


namespace NUMINAMATH_CALUDE_employee_count_l1559_155946

/-- The number of employees in an organization (excluding the manager) -/
def num_employees : ℕ := 20

/-- The average monthly salary of employees (excluding the manager) -/
def avg_salary : ℕ := 1600

/-- The increase in average salary when the manager's salary is added -/
def salary_increase : ℕ := 100

/-- The manager's monthly salary -/
def manager_salary : ℕ := 3700

/-- Theorem stating the number of employees given the salary conditions -/
theorem employee_count :
  (num_employees * avg_salary + manager_salary) / (num_employees + 1) =
  avg_salary + salary_increase :=
by sorry

end NUMINAMATH_CALUDE_employee_count_l1559_155946


namespace NUMINAMATH_CALUDE_sample_size_calculation_l1559_155950

/-- Represents the sample size calculation for three communities --/
theorem sample_size_calculation 
  (pop_A pop_B pop_C : ℕ) 
  (sample_C : ℕ) 
  (h1 : pop_A = 600) 
  (h2 : pop_B = 1200) 
  (h3 : pop_C = 1500) 
  (h4 : sample_C = 15) : 
  ∃ n : ℕ, n * pop_C = sample_C * (pop_A + pop_B + pop_C) ∧ n = 33 :=
sorry

end NUMINAMATH_CALUDE_sample_size_calculation_l1559_155950


namespace NUMINAMATH_CALUDE_s_1010_mod_500_l1559_155948

/-- The polynomial q(x) = x^1010 + x^1009 + x^1008 + ... + x + 1 -/
def q (x : ℕ) : ℕ := Finset.sum (Finset.range 1011) (fun i => x^i)

/-- The polynomial remainder s(x) when q(x) is divided by x^2 - 1 -/
def s (x : ℕ) : ℕ := q x % (x^2 - 1)

/-- Theorem stating that s(1010) modulo 500 equals 55 -/
theorem s_1010_mod_500 : s 1010 % 500 = 55 := by
  sorry

end NUMINAMATH_CALUDE_s_1010_mod_500_l1559_155948


namespace NUMINAMATH_CALUDE_root_transformation_l1559_155999

-- Define the original quadratic equation
def original_equation (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- Define the transformed equation
def transformed_equation (a b c x : ℝ) : Prop := a * (x - 1)^2 + b * (x - 1) + c = 0

-- Theorem statement
theorem root_transformation (a b c : ℝ) :
  (original_equation a b c (-1) ∧ original_equation a b c 2) →
  (transformed_equation a b c 0 ∧ transformed_equation a b c 3) :=
by sorry

end NUMINAMATH_CALUDE_root_transformation_l1559_155999


namespace NUMINAMATH_CALUDE_inverse_exponential_function_l1559_155949

noncomputable def f (a : ℝ) : ℝ → ℝ := fun x => Real.log x / Real.log a

theorem inverse_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f := f a
  (∀ x, f (a^x) = x) ∧ (∀ y, a^(f y) = y) ∧ f 2 = 1 → f = fun x => Real.log x / Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_exponential_function_l1559_155949


namespace NUMINAMATH_CALUDE_at_least_three_correct_guesses_l1559_155983

-- Define the type for colors
inductive Color
| Red | Orange | Yellow | Green | Blue | Indigo | Violet

-- Define the type for dwarves
structure Dwarf where
  id : Fin 6
  seenHats : Finset Color

-- Define the game setup
structure GameSetup where
  allHats : Finset Color
  hiddenHat : Color
  dwarves : Fin 6 → Dwarf

-- Define the guessing strategy
def guessNearestClockwise (d : Dwarf) (allColors : Finset Color) : Color :=
  sorry

-- Theorem statement
theorem at_least_three_correct_guesses 
  (setup : GameSetup)
  (h1 : setup.allHats.card = 7)
  (h2 : ∀ d : Fin 6, (setup.dwarves d).seenHats.card = 5)
  (h3 : ∀ d : Fin 6, (setup.dwarves d).seenHats ⊆ setup.allHats)
  (h4 : setup.hiddenHat ∈ setup.allHats) :
  ∃ (correctGuesses : Finset (Fin 6)), 
    correctGuesses.card ≥ 3 ∧ 
    ∀ d ∈ correctGuesses, guessNearestClockwise (setup.dwarves d) setup.allHats = setup.hiddenHat :=
sorry

end NUMINAMATH_CALUDE_at_least_three_correct_guesses_l1559_155983


namespace NUMINAMATH_CALUDE_part_one_part_two_l1559_155981

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ 1 ∨ x ≥ 2}
def B (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 2}

-- Theorem for part 1
theorem part_one : (Set.univ \ A) ∩ B 1 = {x | 1 < x ∧ x < 2} := by sorry

-- Theorem for part 2
theorem part_two : ∀ a : ℝ, (Set.univ \ A) ∩ B a = ∅ ↔ a ≤ -1 ∨ a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1559_155981


namespace NUMINAMATH_CALUDE_remaining_water_volume_l1559_155968

/-- Given a cup with 2 liters of water, after pouring out x milliliters 4 times, 
    the remaining volume in milliliters is equal to 2000 - 4x. -/
theorem remaining_water_volume (x : ℝ) : 
  2000 - 4 * x = (2 : ℝ) * 1000 - 4 * x := by sorry

end NUMINAMATH_CALUDE_remaining_water_volume_l1559_155968


namespace NUMINAMATH_CALUDE_profit_percentage_l1559_155923

theorem profit_percentage (C P : ℝ) (h : (2/3) * P = 0.9 * C) :
  (P - C) / C = 0.35 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_l1559_155923


namespace NUMINAMATH_CALUDE_range_of_z_l1559_155952

theorem range_of_z (x y : ℝ) (h : x^2 + y^2 = 1) :
  ∃ (z_min z_max : ℝ), z_min = -4/3 ∧ z_max = 0 ∧
  ∀ z, z = (y - 1) / (x + 2) → z_min ≤ z ∧ z ≤ z_max :=
sorry

end NUMINAMATH_CALUDE_range_of_z_l1559_155952


namespace NUMINAMATH_CALUDE_find_a_value_l1559_155959

-- Define the polynomial expansion
def polynomial_expansion (n : ℕ) (a b c : ℤ) (x : ℝ) : Prop :=
  (x + 2) ^ n = x ^ n + a * x ^ (n - 1) + (b * x + c)

-- State the theorem
theorem find_a_value (n : ℕ) (a b c : ℤ) :
  n ≥ 3 →
  polynomial_expansion n a b c x →
  b = 4 * c →
  a = 16 := by
  sorry


end NUMINAMATH_CALUDE_find_a_value_l1559_155959


namespace NUMINAMATH_CALUDE_line_intercept_form_l1559_155913

/-- A line passing through the point (2,3) with slope 2 has the equation x/(1/2) + y/(-1) = 1 in intercept form. -/
theorem line_intercept_form (l : Set (ℝ × ℝ)) : 
  (∀ (x y : ℝ), (x, y) ∈ l ↔ y - 3 = 2 * (x - 2)) →
  (∀ (x y : ℝ), (x, y) ∈ l ↔ x / (1/2) + y / (-1) = 1) :=
by sorry

end NUMINAMATH_CALUDE_line_intercept_form_l1559_155913


namespace NUMINAMATH_CALUDE_unique_square_divisible_by_six_in_range_l1559_155954

theorem unique_square_divisible_by_six_in_range : ∃! x : ℕ, 
  (∃ n : ℕ, x = n^2) ∧ 
  (∃ k : ℕ, x = 6 * k) ∧ 
  50 ≤ x ∧ x ≤ 150 :=
by sorry

end NUMINAMATH_CALUDE_unique_square_divisible_by_six_in_range_l1559_155954


namespace NUMINAMATH_CALUDE_rebeccas_marbles_l1559_155963

/-- Rebecca's egg and marble problem -/
theorem rebeccas_marbles :
  ∀ (num_eggs num_marbles : ℕ),
  num_eggs = 20 →
  num_eggs = num_marbles + 14 →
  num_marbles = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_rebeccas_marbles_l1559_155963


namespace NUMINAMATH_CALUDE_not_necessarily_equal_numbers_l1559_155994

theorem not_necessarily_equal_numbers : ∃ (a b c : ℝ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  (a + b^2 + c^2 = b + a^2 + c^2) ∧
  (a + b^2 + c^2 = c + a^2 + b^2) ∧
  ¬(a = b ∧ b = c) := by
  sorry

end NUMINAMATH_CALUDE_not_necessarily_equal_numbers_l1559_155994


namespace NUMINAMATH_CALUDE_last_two_digits_product_l1559_155921

theorem last_two_digits_product (n : ℤ) : 
  (∃ k : ℤ, n = 6 * k) →  -- n is divisible by 6
  (∃ a b : ℕ, a < 10 ∧ b < 10 ∧ n % 100 = 10 * a + b ∧ a + b = 12) →  -- sum of last two digits is 12
  (∃ x y : ℕ, x < 10 ∧ y < 10 ∧ n % 100 = 10 * x + y ∧ x * y = 32 ∨ x * y = 36) :=
by sorry

end NUMINAMATH_CALUDE_last_two_digits_product_l1559_155921


namespace NUMINAMATH_CALUDE_soda_cans_calculation_correct_l1559_155922

/-- Given that S cans of soda can be purchased for Q dimes, and 1 dollar is worth 10 dimes,
    this function calculates the number of cans that can be purchased for D dollars. -/
def soda_cans_for_dollars (S Q D : ℚ) : ℚ :=
  10 * D * S / Q

/-- Theorem stating that the number of cans that can be purchased for D dollars
    is correctly calculated by the soda_cans_for_dollars function. -/
theorem soda_cans_calculation_correct (S Q D : ℚ) (hS : S > 0) (hQ : Q > 0) (hD : D ≥ 0) :
  soda_cans_for_dollars S Q D = 10 * D * S / Q :=
by sorry

end NUMINAMATH_CALUDE_soda_cans_calculation_correct_l1559_155922


namespace NUMINAMATH_CALUDE_unique_prime_with_same_remainder_l1559_155934

theorem unique_prime_with_same_remainder : 
  ∃! n : ℕ, 
    Prime n ∧ 
    200 < n ∧ 
    n < 300 ∧ 
    ∃ r : ℕ, n % 7 = r ∧ n % 9 = r :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_with_same_remainder_l1559_155934


namespace NUMINAMATH_CALUDE_ship_journey_distance_l1559_155969

/-- The total distance traveled by a ship in three days -/
def ship_total_distance (first_day_distance : ℝ) : ℝ :=
  let second_day_distance := 3 * first_day_distance
  let third_day_distance := second_day_distance + 110
  first_day_distance + second_day_distance + third_day_distance

/-- Theorem stating the total distance traveled by the ship -/
theorem ship_journey_distance : ship_total_distance 100 = 810 := by
  sorry

end NUMINAMATH_CALUDE_ship_journey_distance_l1559_155969


namespace NUMINAMATH_CALUDE_prob_different_topics_is_five_sixths_l1559_155939

/-- The number of essay topics -/
def num_topics : ℕ := 6

/-- The probability that two students select different topics -/
def prob_different_topics : ℚ := 5/6

/-- Theorem stating that the probability of two students selecting different topics
    from 6 available topics is 5/6 -/
theorem prob_different_topics_is_five_sixths :
  prob_different_topics = 5/6 := by sorry

end NUMINAMATH_CALUDE_prob_different_topics_is_five_sixths_l1559_155939


namespace NUMINAMATH_CALUDE_john_total_running_distance_l1559_155926

/-- The number of days from Monday to Saturday, inclusive -/
def days_ran : ℕ := 6

/-- The distance John ran each day in meters -/
def daily_distance : ℕ := 1700

/-- The total distance John ran before getting injured -/
def total_distance : ℕ := days_ran * daily_distance

/-- Theorem stating that the total distance John ran is 10200 meters -/
theorem john_total_running_distance :
  total_distance = 10200 := by sorry

end NUMINAMATH_CALUDE_john_total_running_distance_l1559_155926


namespace NUMINAMATH_CALUDE_teacher_in_middle_girls_not_adjacent_teacher_flanked_by_girls_l1559_155987

-- Define the class composition
def num_boys : ℕ := 4
def num_girls : ℕ := 2
def num_teacher : ℕ := 1

-- Define the total number of people
def total_people : ℕ := num_boys + num_girls + num_teacher

-- Theorem for scenario 1
theorem teacher_in_middle :
  (Nat.factorial (total_people - 1)) = 720 := by sorry

-- Theorem for scenario 2
theorem girls_not_adjacent :
  (Nat.factorial (total_people - num_girls)) * (Nat.factorial (total_people - num_girls - 1)) = 2400 := by sorry

-- Theorem for scenario 3
theorem teacher_flanked_by_girls :
  (Nat.factorial (total_people - num_girls)) * (Nat.factorial num_girls) = 240 := by sorry

end NUMINAMATH_CALUDE_teacher_in_middle_girls_not_adjacent_teacher_flanked_by_girls_l1559_155987


namespace NUMINAMATH_CALUDE_parabola_intersection_point_l1559_155909

-- Define the parabolas
def C₁ (x y : ℝ) : Prop := x = (1/4) * (y - 1)^2 + (Real.sqrt 2 - 1)

def C₂ (x y a b : ℝ) : Prop := y^2 - a*y + x + 2*b = 0

-- Define the perpendicular tangents condition
def perpendicularTangents (x y a : ℝ) : Prop := 
  (1 / (2*y - 2)) * (-1 / (2*y - a)) = -1

-- Theorem statement
theorem parabola_intersection_point 
  (a b : ℝ) 
  (h : ∃ x y, C₁ x y ∧ C₂ x y a b ∧ perpendicularTangents x y a) :
  C₂ (Real.sqrt 2 - 1/2) 1 a b :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_point_l1559_155909


namespace NUMINAMATH_CALUDE_sprint_tournament_races_l1559_155957

/-- Calculates the number of races needed to determine a winner in a sprint tournament. -/
def races_needed (total_athletes : ℕ) (runners_per_race : ℕ) (advancing_per_race : ℕ) : ℕ :=
  sorry

/-- The sprint tournament problem -/
theorem sprint_tournament_races (total_athletes : ℕ) (runners_per_race : ℕ) (advancing_per_race : ℕ) 
  (h1 : total_athletes = 300)
  (h2 : runners_per_race = 8)
  (h3 : advancing_per_race = 2) :
  races_needed total_athletes runners_per_race advancing_per_race = 53 :=
by sorry

end NUMINAMATH_CALUDE_sprint_tournament_races_l1559_155957


namespace NUMINAMATH_CALUDE_single_elimination_tournament_games_tournament_with_23_teams_l1559_155937

/-- In a single-elimination tournament, the number of games played is one less than the number of teams. -/
theorem single_elimination_tournament_games (n : ℕ) (n_pos : n > 0) :
  let teams := n
  let games := n - 1
  games = teams - 1 := by sorry

/-- For a tournament with 23 teams, 22 games are played. -/
theorem tournament_with_23_teams :
  let teams := 23
  let games := teams - 1
  games = 22 := by sorry

end NUMINAMATH_CALUDE_single_elimination_tournament_games_tournament_with_23_teams_l1559_155937


namespace NUMINAMATH_CALUDE_conditional_prob_B_given_A_l1559_155972

-- Define the sample space for a six-sided die
def Ω : Finset Nat := {1, 2, 3, 4, 5, 6}

-- Define event A: odd numbers
def A : Finset Nat := {1, 3, 5}

-- Define event B: getting 3 points
def B : Finset Nat := {3}

-- Define the probability measure
def P (S : Finset Nat) : ℚ := (S.card : ℚ) / (Ω.card : ℚ)

-- Theorem: P(B|A) = 1/3
theorem conditional_prob_B_given_A : 
  P (A ∩ B) / P A = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_conditional_prob_B_given_A_l1559_155972


namespace NUMINAMATH_CALUDE_dog_eaten_cost_l1559_155930

-- Define the ingredients and their costs
def flour_cost : ℝ := 3.20
def sugar_cost : ℝ := 2.10
def butter_cost : ℝ := 5.50
def egg_cost : ℝ := 0.45
def baking_soda_cost : ℝ := 0.60
def baking_powder_cost : ℝ := 1.30
def salt_cost : ℝ := 0.35
def vanilla_extract_cost : ℝ := 1.75
def milk_cost : ℝ := 1.40
def vegetable_oil_cost : ℝ := 2.10

-- Define the quantities of ingredients
def flour_qty : ℝ := 2.5
def sugar_qty : ℝ := 1.5
def butter_qty : ℝ := 0.75
def egg_qty : ℝ := 4
def baking_soda_qty : ℝ := 1
def baking_powder_qty : ℝ := 1
def salt_qty : ℝ := 1
def vanilla_extract_qty : ℝ := 1
def milk_qty : ℝ := 1.25
def vegetable_oil_qty : ℝ := 0.75

-- Define other constants
def sales_tax_rate : ℝ := 0.07
def total_slices : ℕ := 12
def mother_eaten_slices : ℕ := 4

-- Theorem to prove
theorem dog_eaten_cost (total_cost : ℝ) (cost_with_tax : ℝ) (cost_per_slice : ℝ) :
  total_cost = flour_cost * flour_qty + sugar_cost * sugar_qty + butter_cost * butter_qty +
               egg_cost * egg_qty + baking_soda_cost * baking_soda_qty + 
               baking_powder_cost * baking_powder_qty + salt_cost * salt_qty +
               vanilla_extract_cost * vanilla_extract_qty + milk_cost * milk_qty +
               vegetable_oil_cost * vegetable_oil_qty →
  cost_with_tax = total_cost * (1 + sales_tax_rate) →
  cost_per_slice = cost_with_tax / total_slices →
  cost_per_slice * (total_slices - mother_eaten_slices) = 17.44 :=
by sorry

end NUMINAMATH_CALUDE_dog_eaten_cost_l1559_155930


namespace NUMINAMATH_CALUDE_unique_triple_lcm_l1559_155914

theorem unique_triple_lcm : 
  ∃! (a b c : ℕ+), 
    Nat.lcm a b = 1200 ∧ 
    Nat.lcm b c = 1800 ∧ 
    Nat.lcm c a = 2400 := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_lcm_l1559_155914


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l1559_155982

def repeating_decimal_12 : ℚ := 4 / 33
def repeating_decimal_03 : ℚ := 1 / 33
def repeating_decimal_006 : ℚ := 2 / 333

theorem sum_of_repeating_decimals :
  repeating_decimal_12 + repeating_decimal_03 + repeating_decimal_006 = 19041 / 120879 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l1559_155982


namespace NUMINAMATH_CALUDE_factorization_equality_l1559_155996

theorem factorization_equality (m n : ℝ) : m^2*n - 2*m*n + n = n*(m-1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1559_155996


namespace NUMINAMATH_CALUDE_function_not_in_second_quadrant_l1559_155984

/-- The function f(x) = a^x + b does not pass through the second quadrant when a > 1 and b < -1 -/
theorem function_not_in_second_quadrant (a b : ℝ) (ha : a > 1) (hb : b < -1) :
  ∀ x : ℝ, x < 0 → a^x + b ≤ 0 := by sorry

end NUMINAMATH_CALUDE_function_not_in_second_quadrant_l1559_155984


namespace NUMINAMATH_CALUDE_expected_practice_problems_l1559_155932

/-- Represents the number of pairs of shoes -/
def num_pairs : ℕ := 5

/-- Represents the number of days -/
def num_days : ℕ := 5

/-- Represents the probability of selecting two shoes of the same color on a given day -/
def prob_same_color : ℚ := 1 / 9

/-- Represents the expected number of practice problems done in one day -/
def expected_problems_per_day : ℚ := prob_same_color

/-- Theorem stating the expected value of practice problems over 5 days -/
theorem expected_practice_problems :
  (num_days : ℚ) * expected_problems_per_day = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_expected_practice_problems_l1559_155932


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l1559_155973

theorem fraction_sum_equality : 
  (1 : ℚ) / 15 + (2 : ℚ) / 25 + (3 : ℚ) / 35 + (4 : ℚ) / 45 = (506 : ℚ) / 1575 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l1559_155973


namespace NUMINAMATH_CALUDE_digit_150_of_17_70_l1559_155960

/-- The decimal representation of a rational number -/
def decimal_representation (q : ℚ) : ℕ → ℕ := sorry

/-- The repeating sequence in the decimal representation of a rational number -/
def repeating_sequence (q : ℚ) : List ℕ := sorry

theorem digit_150_of_17_70 : 
  decimal_representation (17 / 70) 150 = 7 := by sorry

end NUMINAMATH_CALUDE_digit_150_of_17_70_l1559_155960


namespace NUMINAMATH_CALUDE_starters_count_l1559_155993

/-- The number of ways to select 7 starters from a team of 16 players,
    including a set of twins, with the condition that at least one but
    no more than two twins must be included. -/
def select_starters (total_players : ℕ) (num_twins : ℕ) (num_starters : ℕ) : ℕ :=
  let non_twin_players := total_players - num_twins
  let one_twin := num_twins * Nat.choose non_twin_players (num_starters - 1)
  let both_twins := Nat.choose non_twin_players (num_starters - num_twins)
  one_twin + both_twins

theorem starters_count :
  select_starters 16 2 7 = 8008 := by
  sorry

end NUMINAMATH_CALUDE_starters_count_l1559_155993


namespace NUMINAMATH_CALUDE_sum_at_two_and_neg_two_l1559_155924

/-- A cubic polynomial with specific properties -/
structure CubicPolynomial (k : ℝ) where
  Q : ℝ → ℝ
  is_cubic : ∃ a b c : ℝ, ∀ x, Q x = a * x^3 + b * x^2 + c * x + k
  at_zero : Q 0 = k
  at_one : Q 1 = 3 * k
  at_neg_one : Q (-1) = 4 * k

/-- The sum of the polynomial evaluated at 2 and -2 equals 22k -/
theorem sum_at_two_and_neg_two (k : ℝ) (p : CubicPolynomial k) :
  p.Q 2 + p.Q (-2) = 22 * k := by sorry

end NUMINAMATH_CALUDE_sum_at_two_and_neg_two_l1559_155924


namespace NUMINAMATH_CALUDE_derivative_f_l1559_155920

noncomputable def f (x : ℝ) : ℝ := (3/2) * Real.log (Real.tanh (x/2)) + Real.cosh x - Real.cosh x / (2 * Real.sinh x ^ 2)

theorem derivative_f (x : ℝ) : 
  deriv f x = Real.cosh x ^ 4 / Real.sinh x ^ 3 :=
by sorry

end NUMINAMATH_CALUDE_derivative_f_l1559_155920


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1559_155990

theorem trigonometric_identity : 
  (Real.cos (68 * π / 180) * Real.cos (8 * π / 180) - Real.cos (82 * π / 180) * Real.cos (22 * π / 180)) /
  (Real.cos (53 * π / 180) * Real.cos (23 * π / 180) - Real.cos (67 * π / 180) * Real.cos (37 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1559_155990


namespace NUMINAMATH_CALUDE_product_of_coefficients_l1559_155903

theorem product_of_coefficients (b c : ℤ) : 
  (∀ r : ℝ, r^2 - 2*r - 1 = 0 → r^5 - b*r - c = 0) → 
  b * c = 348 := by
sorry

end NUMINAMATH_CALUDE_product_of_coefficients_l1559_155903


namespace NUMINAMATH_CALUDE_sandys_carrots_l1559_155917

/-- Sandy's carrot problem -/
theorem sandys_carrots (initial_carrots : ℕ) (taken_carrots : ℕ) 
  (h1 : initial_carrots = 6)
  (h2 : taken_carrots = 3) :
  initial_carrots - taken_carrots = 3 := by
  sorry

end NUMINAMATH_CALUDE_sandys_carrots_l1559_155917


namespace NUMINAMATH_CALUDE_cycle_price_proof_l1559_155931

theorem cycle_price_proof (sale_price : ℝ) (gain_percentage : ℝ) 
  (h1 : sale_price = 1440)
  (h2 : gain_percentage = 60) : 
  ∃ original_price : ℝ, 
    original_price = 900 ∧ 
    sale_price = original_price + (gain_percentage / 100) * original_price :=
by
  sorry

end NUMINAMATH_CALUDE_cycle_price_proof_l1559_155931


namespace NUMINAMATH_CALUDE_side_significant_digits_equal_area_significant_digits_l1559_155915

-- Define the area of the square
def square_area : ℝ := 2.3406

-- Define the precision of the area measurement (to the nearest ten-thousandth)
def area_precision : ℝ := 0.0001

-- Define the function to count significant digits
def count_significant_digits (x : ℝ) : ℕ := sorry

-- Theorem statement
theorem side_significant_digits_equal_area_significant_digits :
  count_significant_digits (Real.sqrt square_area) = count_significant_digits square_area :=
sorry

end NUMINAMATH_CALUDE_side_significant_digits_equal_area_significant_digits_l1559_155915


namespace NUMINAMATH_CALUDE_smallest_sum_reciprocals_l1559_155978

theorem smallest_sum_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 15) :
  ∃ (a b : ℕ+), a ≠ b ∧ (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 15 ∧ a.val + b.val = 64 ∧
  ∀ (c d : ℕ+), c ≠ d → (1 : ℚ) / c + (1 : ℚ) / d = (1 : ℚ) / 15 → c.val + d.val ≥ 64 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_reciprocals_l1559_155978


namespace NUMINAMATH_CALUDE_complex_real_condition_l1559_155928

theorem complex_real_condition (m : ℝ) :
  (Complex.I * (m^2 - 2*m - 15) : ℂ).im = 0 → m = 5 ∨ m = -3 := by
  sorry

end NUMINAMATH_CALUDE_complex_real_condition_l1559_155928


namespace NUMINAMATH_CALUDE_airline_owns_five_planes_l1559_155927

/-- The number of airplanes owned by an airline company. -/
def num_airplanes (rows_per_plane : ℕ) (seats_per_row : ℕ) (flights_per_day : ℕ) (total_passengers : ℕ) : ℕ :=
  total_passengers / (rows_per_plane * seats_per_row * flights_per_day)

/-- Theorem stating that the airline company owns 5 airplanes. -/
theorem airline_owns_five_planes :
  num_airplanes 20 7 2 1400 = 5 := by
  sorry

end NUMINAMATH_CALUDE_airline_owns_five_planes_l1559_155927


namespace NUMINAMATH_CALUDE_absolute_value_integral_l1559_155907

theorem absolute_value_integral : ∫ x in (0:ℝ)..(4:ℝ), |x - 2| = 4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_integral_l1559_155907


namespace NUMINAMATH_CALUDE_ellipse_properties_l1559_155975

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0
  h_ecc : (a^2 - b^2) / a^2 = 1/4
  h_max_area : a * b = 2 * Real.sqrt 3

/-- The standard form of the ellipse -/
def standard_form (e : Ellipse) : Prop :=
  ∀ x y : ℝ, x^2/4 + y^2/3 = 1 ↔ x^2/e.a^2 + y^2/e.b^2 = 1

/-- The fixed point property -/
def fixed_point_property (e : Ellipse) : Prop :=
  ∃ D : ℝ × ℝ, 
    D.2 = 0 ∧ 
    D.1 = -11/8 ∧
    ∀ M N : ℝ × ℝ,
      (M.1^2/e.a^2 + M.2^2/e.b^2 = 1) →
      (N.1^2/e.a^2 + N.2^2/e.b^2 = 1) →
      (∃ t : ℝ, M.1 = t * M.2 - 1 ∧ N.1 = t * N.2 - 1) →
      ((M.1 - D.1) * (N.1 - D.1) + (M.2 - D.2) * (N.2 - D.2) = -135/64)

theorem ellipse_properties (e : Ellipse) : 
  standard_form e ∧ fixed_point_property e := by
  sorry

end NUMINAMATH_CALUDE_ellipse_properties_l1559_155975


namespace NUMINAMATH_CALUDE_rationalize_denominator_cube_root_l1559_155908

theorem rationalize_denominator_cube_root :
  ∀ (x : ℝ), x^3 = 3 →
  2 / (x - 2) = -(2 * x + 4) / 5 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_cube_root_l1559_155908


namespace NUMINAMATH_CALUDE_shooting_team_composition_l1559_155940

theorem shooting_team_composition (x y : ℕ) : 
  x > 0 → y > 0 →
  (22 * x + 47 * y) / (x + y) = 41 →
  (y : ℚ) / (x + y) = 19 / 25 := by
sorry

end NUMINAMATH_CALUDE_shooting_team_composition_l1559_155940


namespace NUMINAMATH_CALUDE_cos_alpha_minus_pi_third_l1559_155905

theorem cos_alpha_minus_pi_third (α : ℝ) 
  (h : Real.cos (α - π / 6) + Real.sin α = (4 / 5) * Real.sqrt 3) : 
  Real.cos (α - π / 3) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_minus_pi_third_l1559_155905


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l1559_155958

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x > 0, P x) ↔ (∀ x > 0, ¬ P x) :=
by sorry

theorem negation_of_quadratic_equation :
  (¬ ∃ x > 0, x^2 + 3*x - 5 = 0) ↔ (∀ x > 0, x^2 + 3*x - 5 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l1559_155958


namespace NUMINAMATH_CALUDE_min_sum_is_twelve_l1559_155904

/-- Represents a 3x3 grid of integers -/
def Grid := Matrix (Fin 3) (Fin 3) ℕ

/-- Checks if a grid contains all numbers from 1 to 9 exactly once -/
def isValidGrid (g : Grid) : Prop :=
  ∀ n : ℕ, n ≥ 1 ∧ n ≤ 9 → (∃! (i j : Fin 3), g i j = n)

/-- Calculates the sum of a row in the grid -/
def rowSum (g : Grid) (i : Fin 3) : ℕ :=
  (g i 0) + (g i 1) + (g i 2)

/-- Calculates the sum of a column in the grid -/
def colSum (g : Grid) (j : Fin 3) : ℕ :=
  (g 0 j) + (g 1 j) + (g 2 j)

/-- Checks if all rows and columns in the grid have the same sum -/
def hasEqualSums (g : Grid) : Prop :=
  ∃ s : ℕ, (∀ i : Fin 3, rowSum g i = s) ∧ (∀ j : Fin 3, colSum g j = s)

/-- The main theorem: The minimum sum for a valid grid with equal sums is 12 -/
theorem min_sum_is_twelve :
  ∀ g : Grid, isValidGrid g → hasEqualSums g →
  ∃ s : ℕ, (∀ i : Fin 3, rowSum g i = s) ∧ (∀ j : Fin 3, colSum g j = s) ∧ s ≥ 12 :=
sorry

end NUMINAMATH_CALUDE_min_sum_is_twelve_l1559_155904


namespace NUMINAMATH_CALUDE_dataset_transformation_l1559_155971

theorem dataset_transformation (initial_points : ℕ) : 
  initial_points = 200 →
  let increased_points := initial_points + initial_points / 5
  let final_points := increased_points - increased_points / 4
  final_points = 180 := by
sorry

end NUMINAMATH_CALUDE_dataset_transformation_l1559_155971


namespace NUMINAMATH_CALUDE_odd_function_property_l1559_155976

def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def IsIncreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

def HasMinimumOn (f : ℝ → ℝ) (a b : ℝ) (m : ℝ) : Prop :=
  (∀ x, a ≤ x ∧ x ≤ b → m ≤ f x) ∧ (∃ x, a ≤ x ∧ x ≤ b ∧ f x = m)

def HasMaximumOn (f : ℝ → ℝ) (a b : ℝ) (m : ℝ) : Prop :=
  (∀ x, a ≤ x ∧ x ≤ b → f x ≤ m) ∧ (∃ x, a ≤ x ∧ x ≤ b ∧ f x = m)

theorem odd_function_property (f : ℝ → ℝ) :
  IsOdd f →
  IsIncreasingOn f 1 3 →
  HasMinimumOn f 1 3 7 →
  IsIncreasingOn f (-3) (-1) ∧ HasMaximumOn f (-3) (-1) (-7) := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l1559_155976


namespace NUMINAMATH_CALUDE_temperature_difference_l1559_155988

def highest_temp : ℚ := 10
def lowest_temp : ℚ := -5

theorem temperature_difference :
  highest_temp - lowest_temp = 15 := by sorry

end NUMINAMATH_CALUDE_temperature_difference_l1559_155988


namespace NUMINAMATH_CALUDE_sum_of_distances_constant_l1559_155902

/-- An equilateral triangle -/
structure EquilateralTriangle where
  side : ℝ
  side_pos : side > 0

/-- A point inside an equilateral triangle -/
structure InternalPoint (t : EquilateralTriangle) where
  x : ℝ
  y : ℝ
  inside : x > 0 ∧ y > 0 ∧ x + y < t.side

/-- The sum of perpendicular distances from an internal point to the three sides of an equilateral triangle -/
def sumOfDistances (t : EquilateralTriangle) (p : InternalPoint t) : ℝ :=
  p.x + p.y + (t.side - p.x - p.y)

/-- Theorem: The sum of perpendicular distances from any internal point to the three sides of an equilateral triangle is constant and equal to (√3/2) * side length -/
theorem sum_of_distances_constant (t : EquilateralTriangle) (p : InternalPoint t) :
  sumOfDistances t p = (Real.sqrt 3 / 2) * t.side := by
  sorry


end NUMINAMATH_CALUDE_sum_of_distances_constant_l1559_155902


namespace NUMINAMATH_CALUDE_specific_box_volume_l1559_155998

/-- The volume of an open box constructed from a rectangular sheet of metal -/
def box_volume (length width x : ℝ) : ℝ :=
  (length - 2*x) * (width - 2*x) * x

/-- Theorem: The volume of the specific box described in the problem -/
theorem specific_box_volume (x : ℝ) :
  box_volume 16 12 x = 4*x^3 - 56*x^2 + 192*x :=
by sorry

end NUMINAMATH_CALUDE_specific_box_volume_l1559_155998


namespace NUMINAMATH_CALUDE_system_solution_l1559_155955

theorem system_solution (x y k : ℝ) : 
  4 * x + 3 * y = 1 → 
  k * x + (k - 1) * y = 3 → 
  x = y → 
  k = 11 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l1559_155955


namespace NUMINAMATH_CALUDE_smallest_nonnegative_congruence_l1559_155953

theorem smallest_nonnegative_congruence :
  ∃ n : ℕ, n < 7 ∧ -2222 ≡ n [ZMOD 7] ∧ ∀ m : ℕ, m < 7 → -2222 ≡ m [ZMOD 7] → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_nonnegative_congruence_l1559_155953


namespace NUMINAMATH_CALUDE_min_type_A_buses_l1559_155900

/-- Represents the number of Type A buses -/
def x : ℕ := sorry

/-- The capacity of a Type A bus -/
def capacity_A : ℕ := 45

/-- The capacity of a Type B bus -/
def capacity_B : ℕ := 30

/-- The total number of people to transport -/
def total_people : ℕ := 300

/-- The total number of buses to be rented -/
def total_buses : ℕ := 8

/-- The minimum number of Type A buses needed -/
def min_buses_A : ℕ := 4

theorem min_type_A_buses :
  (∀ n : ℕ, n ≥ min_buses_A →
    capacity_A * n + capacity_B * (total_buses - n) ≥ total_people) ∧
  (∀ m : ℕ, m < min_buses_A →
    capacity_A * m + capacity_B * (total_buses - m) < total_people) :=
by sorry

end NUMINAMATH_CALUDE_min_type_A_buses_l1559_155900


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1559_155965

def i : ℂ := Complex.I

theorem complex_fraction_simplification :
  (1 + 2*i) / i = -2 + i := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1559_155965


namespace NUMINAMATH_CALUDE_wood_length_l1559_155992

/-- The original length of a piece of wood, given the length sawed off and the remaining length -/
theorem wood_length (sawed_off : ℝ) (remaining : ℝ) (h1 : sawed_off = 2.3) (h2 : remaining = 6.6) :
  sawed_off + remaining = 8.9 := by
  sorry

end NUMINAMATH_CALUDE_wood_length_l1559_155992


namespace NUMINAMATH_CALUDE_optimal_profit_l1559_155995

/-- Profit function for n plants per pot -/
def P (n : ℕ) : ℝ := n * (5 - 0.5 * (n - 3))

/-- The optimal number of plants per pot -/
def optimal_plants : ℕ := 5

theorem optimal_profit :
  (P optimal_plants = 20) ∧ 
  (∀ n : ℕ, 3 ≤ n ∧ n ≤ 6 → P n ≤ 20) ∧
  (∀ n : ℕ, 3 ≤ n ∧ n < optimal_plants → P n < 20) ∧
  (∀ n : ℕ, optimal_plants < n ∧ n ≤ 6 → P n < 20) := by
  sorry

#eval P optimal_plants  -- Should output 20

end NUMINAMATH_CALUDE_optimal_profit_l1559_155995


namespace NUMINAMATH_CALUDE_product_increase_theorem_l1559_155962

theorem product_increase_theorem :
  ∃ (a b c d e : ℕ), 
    (((a - 3) * (b - 3) * (c - 3) * (d - 3) * (e - 3)) : ℤ) = 
    15 * (a * b * c * d * e) :=
by sorry

end NUMINAMATH_CALUDE_product_increase_theorem_l1559_155962


namespace NUMINAMATH_CALUDE_pizza_cost_is_80_l1559_155966

/-- The total cost of pizzas given the number of pizzas, pieces per pizza, and cost per piece. -/
def total_cost (num_pizzas : ℕ) (pieces_per_pizza : ℕ) (cost_per_piece : ℕ) : ℕ :=
  num_pizzas * pieces_per_pizza * cost_per_piece

/-- Theorem stating that the total cost of pizzas is $80 under the given conditions. -/
theorem pizza_cost_is_80 :
  total_cost 4 5 4 = 80 := by
  sorry

end NUMINAMATH_CALUDE_pizza_cost_is_80_l1559_155966


namespace NUMINAMATH_CALUDE_frog_climb_time_l1559_155977

/-- Represents the frog's climbing problem in the well -/
structure FrogClimb where
  well_depth : ℕ := 12
  climb_distance : ℕ := 3
  slide_distance : ℕ := 1
  time_to_climb : ℕ := 3
  time_to_slide : ℕ := 1
  time_at_3m_from_top : ℕ := 17

/-- Calculates the total time for the frog to reach the top of the well -/
def total_climb_time (f : FrogClimb) : ℕ :=
  sorry

/-- Theorem stating that the total climb time is 22 minutes -/
theorem frog_climb_time (f : FrogClimb) : total_climb_time f = 22 :=
  sorry

end NUMINAMATH_CALUDE_frog_climb_time_l1559_155977


namespace NUMINAMATH_CALUDE_kate_museum_visits_cost_l1559_155929

/-- Calculates the total amount spent on museum visits over 3 years -/
def total_spent (initial_fee : ℕ) (increased_fee : ℕ) (visits_first_year : ℕ) (visits_per_year_after : ℕ) : ℕ :=
  initial_fee * visits_first_year + increased_fee * visits_per_year_after * 2

/-- Theorem stating the total amount Kate spent on museum visits over 3 years -/
theorem kate_museum_visits_cost :
  let initial_fee := 5
  let increased_fee := 7
  let visits_first_year := 12
  let visits_per_year_after := 4
  total_spent initial_fee increased_fee visits_first_year visits_per_year_after = 116 := by
  sorry

#eval total_spent 5 7 12 4

end NUMINAMATH_CALUDE_kate_museum_visits_cost_l1559_155929


namespace NUMINAMATH_CALUDE_pencil_cost_l1559_155970

/-- The cost of an item when paying with a dollar and receiving change -/
def item_cost (payment : ℚ) (change : ℚ) : ℚ :=
  payment - change

/-- Theorem: Given a purchase where the buyer pays with a one-dollar bill
    and receives 65 cents in change, the cost of the item is 35 cents. -/
theorem pencil_cost :
  let payment : ℚ := 1
  let change : ℚ := 65/100
  item_cost payment change = 35/100 := by
  sorry

end NUMINAMATH_CALUDE_pencil_cost_l1559_155970


namespace NUMINAMATH_CALUDE_pentagonal_faces_count_l1559_155951

/-- A convex polyhedron with pentagon and hexagon faces -/
structure ConvexPolyhedron where
  -- Number of pentagonal faces
  n : ℕ
  -- Number of hexagonal faces
  k : ℕ
  -- The polyhedron is convex
  convex : True
  -- Faces are either pentagons or hexagons
  faces_pentagon_or_hexagon : True
  -- Exactly three edges meet at each vertex
  three_edges_per_vertex : True

/-- The number of pentagonal faces in a convex polyhedron with pentagon and hexagon faces -/
theorem pentagonal_faces_count (p : ConvexPolyhedron) : p.n = 12 := by
  sorry

end NUMINAMATH_CALUDE_pentagonal_faces_count_l1559_155951


namespace NUMINAMATH_CALUDE_largest_2023_digit_prime_squared_minus_one_div_30_l1559_155941

/-- p is the largest prime with 2023 digits -/
def p : Nat := sorry

/-- p^2 - 1 is divisible by 30 -/
theorem largest_2023_digit_prime_squared_minus_one_div_30 : 
  30 ∣ (p^2 - 1) := by sorry

end NUMINAMATH_CALUDE_largest_2023_digit_prime_squared_minus_one_div_30_l1559_155941


namespace NUMINAMATH_CALUDE_bugs_meet_time_l1559_155910

/-- The time (in minutes) it takes for two bugs to meet again at the starting point,
    given they start on two tangent circles with radii 7 and 3 inches,
    crawling at speeds of 4π and 3π inches per minute respectively. -/
def meeting_time : ℝ :=
  let r₁ : ℝ := 7  -- radius of larger circle
  let r₂ : ℝ := 3  -- radius of smaller circle
  let v₁ : ℝ := 4 * Real.pi  -- speed of bug on larger circle
  let v₂ : ℝ := 3 * Real.pi  -- speed of bug on smaller circle
  let t₁ : ℝ := (2 * Real.pi * r₁) / v₁  -- time for full circle on larger circle
  let t₂ : ℝ := (2 * Real.pi * r₂) / v₂  -- time for full circle on smaller circle
  14  -- the actual meeting time

theorem bugs_meet_time :
  meeting_time = 14 := by sorry

end NUMINAMATH_CALUDE_bugs_meet_time_l1559_155910


namespace NUMINAMATH_CALUDE_probability_theorem_l1559_155912

def is_valid_pair (a b : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 60 ∧ 1 ≤ b ∧ b ≤ 60 ∧ a ≠ b

def satisfies_condition (a b : ℕ) : Prop :=
  ∃ (m : ℕ), a * b + a + b = 6 * m - 1

def total_pairs : ℕ := Nat.choose 60 2

def favorable_pairs : ℕ := total_pairs - Nat.choose 50 2

theorem probability_theorem :
  (favorable_pairs : ℚ) / total_pairs = 91 / 295 := by sorry

end NUMINAMATH_CALUDE_probability_theorem_l1559_155912


namespace NUMINAMATH_CALUDE_cube_rotation_theorem_l1559_155916

/-- Represents a cube with numbers on its faces -/
structure Cube where
  left : ℕ
  right : ℕ
  front : ℕ
  back : ℕ
  top : ℕ
  bottom : ℕ

/-- Represents the state of the cube after rotations -/
structure CubeState where
  bottom : ℕ
  front : ℕ
  right : ℕ

/-- Rotates the cube from left to right -/
def rotateLeftRight (c : Cube) : Cube := sorry

/-- Rotates the cube from front to back -/
def rotateFrontBack (c : Cube) : Cube := sorry

/-- Applies multiple rotations to the cube -/
def applyRotations (c : Cube) (leftRightRotations frontBackRotations : ℕ) : Cube := sorry

/-- Theorem stating the final state of the cube after rotations -/
theorem cube_rotation_theorem (c : Cube) 
  (h1 : c.left + c.right = 50)
  (h2 : c.front + c.back = 50)
  (h3 : c.top + c.bottom = 50) :
  let finalCube := applyRotations c 97 98
  CubeState.mk finalCube.bottom finalCube.front finalCube.right = CubeState.mk 13 35 11 := by sorry

end NUMINAMATH_CALUDE_cube_rotation_theorem_l1559_155916


namespace NUMINAMATH_CALUDE_common_ratio_is_negative_two_l1559_155979

def geometric_sequence : ℕ → ℚ
  | 0 => 10
  | 1 => -20
  | 2 => 40
  | 3 => -80
  | _ => 0  -- We only define the first 4 terms as given in the problem

theorem common_ratio_is_negative_two :
  ∀ n : ℕ, n < 3 → geometric_sequence (n + 1) / geometric_sequence n = -2 :=
by
  sorry

#eval geometric_sequence 0
#eval geometric_sequence 1
#eval geometric_sequence 2
#eval geometric_sequence 3

end NUMINAMATH_CALUDE_common_ratio_is_negative_two_l1559_155979


namespace NUMINAMATH_CALUDE_vector_magnitude_problem_l1559_155991

/-- Given vectors a and b in ℝ², where b = (-1, 2) and a + b = (1, 3),
    prove that the magnitude of a - 2b is equal to 5. -/
theorem vector_magnitude_problem (a b : ℝ × ℝ) 
  (h1 : b = (-1, 2))
  (h2 : a + b = (1, 3)) :
  ‖a - 2 • b‖ = 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_problem_l1559_155991


namespace NUMINAMATH_CALUDE_eighteen_men_handshakes_l1559_155936

/-- The maximum number of handshakes without cyclic handshakes for n men -/
def maxHandshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: For 18 men, the maximum number of handshakes without cyclic handshakes is 153 -/
theorem eighteen_men_handshakes :
  maxHandshakes 18 = 153 := by
  sorry

end NUMINAMATH_CALUDE_eighteen_men_handshakes_l1559_155936


namespace NUMINAMATH_CALUDE_morse_code_symbols_l1559_155986

/-- The number of possible symbols for a given sequence length in Morse code -/
def morse_combinations (n : ℕ) : ℕ := 2^n

/-- The total number of distinct Morse code symbols for sequences up to length 5 -/
def total_morse_symbols : ℕ :=
  (morse_combinations 1) + (morse_combinations 2) + (morse_combinations 3) +
  (morse_combinations 4) + (morse_combinations 5)

theorem morse_code_symbols :
  total_morse_symbols = 62 :=
by sorry

end NUMINAMATH_CALUDE_morse_code_symbols_l1559_155986


namespace NUMINAMATH_CALUDE_cistern_wet_area_l1559_155947

/-- Calculates the total wet surface area of a rectangular cistern -/
def total_wet_area (length width depth : Real) : Real :=
  let bottom_area := length * width
  let long_walls_area := 2 * (length * depth)
  let short_walls_area := 2 * (width * depth)
  bottom_area + long_walls_area + short_walls_area

/-- Theorem stating that the total wet surface area of the given cistern is 233 m² -/
theorem cistern_wet_area :
  total_wet_area 12 14 1.25 = 233 := by
  sorry

end NUMINAMATH_CALUDE_cistern_wet_area_l1559_155947


namespace NUMINAMATH_CALUDE_conference_handshakes_l1559_155997

/-- Represents a conference with specific group dynamics -/
structure Conference where
  total_people : Nat
  group_a_size : Nat
  group_b_size : Nat
  exceptions : Nat
  unknown_per_exception : Nat

/-- Calculates the number of handshakes in the conference -/
def handshakes (c : Conference) : Nat :=
  let group_a_b_handshakes := c.group_a_size * c.group_b_size
  let group_b_internal_handshakes := c.group_b_size * (c.group_b_size - 1) / 2
  let exception_handshakes := c.exceptions * c.unknown_per_exception
  group_a_b_handshakes + group_b_internal_handshakes + exception_handshakes

/-- The theorem to be proved -/
theorem conference_handshakes :
  let c := Conference.mk 40 25 15 5 3
  handshakes c = 495 := by
  sorry

#eval handshakes (Conference.mk 40 25 15 5 3)

end NUMINAMATH_CALUDE_conference_handshakes_l1559_155997


namespace NUMINAMATH_CALUDE_inequality_solution_l1559_155945

theorem inequality_solution (x : ℝ) : (1 + x) / 3 < x / 2 ↔ x > 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1559_155945


namespace NUMINAMATH_CALUDE_condition_relationship_l1559_155974

theorem condition_relationship (θ : ℝ) (a : ℝ) : 
  ¬(∀ θ a, (Real.sqrt (1 + Real.sin θ) = a) ↔ (Real.sin (θ/2) + Real.cos (θ/2) = a)) :=
by sorry

end NUMINAMATH_CALUDE_condition_relationship_l1559_155974


namespace NUMINAMATH_CALUDE_number_exists_l1559_155901

theorem number_exists : ∃ x : ℝ, x * 1.6 - (2 * 1.4) / 1.3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_number_exists_l1559_155901


namespace NUMINAMATH_CALUDE_sum_of_circle_areas_l1559_155918

/-- Represents a right triangle with sides 6, 8, and 10, where the vertices
    are centers of mutually externally tangent circles -/
structure TriangleWithCircles where
  /-- Radius of the circle centered at the vertex opposite the side of length 8 -/
  r : ℝ
  /-- Radius of the circle centered at the vertex opposite the side of length 6 -/
  s : ℝ
  /-- Radius of the circle centered at the vertex opposite the side of length 10 -/
  t : ℝ
  /-- The sum of radii of circles centered at vertices adjacent to side 6 equals 6 -/
  adj_6 : r + s = 6
  /-- The sum of radii of circles centered at vertices adjacent to side 8 equals 8 -/
  adj_8 : s + t = 8
  /-- The sum of radii of circles centered at vertices adjacent to side 10 equals 10 -/
  adj_10 : r + t = 10

/-- The sum of the areas of the three circles in a TriangleWithCircles is 56π -/
theorem sum_of_circle_areas (twc : TriangleWithCircles) :
  π * (twc.r^2 + twc.s^2 + twc.t^2) = 56 * π := by
  sorry

end NUMINAMATH_CALUDE_sum_of_circle_areas_l1559_155918


namespace NUMINAMATH_CALUDE_sum_of_digits_power_of_two_l1559_155985

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- Main theorem -/
theorem sum_of_digits_power_of_two : sum_of_digits (sum_of_digits (sum_of_digits (2^2006))) = 4 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_digits_power_of_two_l1559_155985


namespace NUMINAMATH_CALUDE_book_pricing_problem_l1559_155933

-- Define the variables
variable (price_A : ℝ) (price_B : ℝ) (num_A : ℕ) (num_B : ℕ)

-- Define the conditions
def condition1 : Prop := price_A * num_A = 3000
def condition2 : Prop := price_B * num_B = 1600
def condition3 : Prop := price_A = 1.5 * price_B
def condition4 : Prop := num_A = num_B + 20

-- Define the World Book Day purchase
def world_book_day_expenditure : ℝ := 0.8 * (20 * price_A + 25 * price_B)

-- State the theorem
theorem book_pricing_problem 
  (h1 : condition1 price_A num_A)
  (h2 : condition2 price_B num_B)
  (h3 : condition3 price_A price_B)
  (h4 : condition4 num_A num_B) :
  price_A = 30 ∧ price_B = 20 ∧ world_book_day_expenditure price_A price_B = 880 := by
  sorry


end NUMINAMATH_CALUDE_book_pricing_problem_l1559_155933


namespace NUMINAMATH_CALUDE_square_difference_81_49_l1559_155989

theorem square_difference_81_49 : 81^2 - 49^2 = 4160 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_81_49_l1559_155989


namespace NUMINAMATH_CALUDE_investment_principal_l1559_155919

/-- Proves that an investment with a 9% simple annual interest rate yielding $231 monthly interest has a principal of $30,800 --/
theorem investment_principal (monthly_interest : ℝ) (annual_rate : ℝ) :
  monthly_interest = 231 →
  annual_rate = 0.09 →
  (monthly_interest / (annual_rate / 12)) = 30800 := by
  sorry

end NUMINAMATH_CALUDE_investment_principal_l1559_155919


namespace NUMINAMATH_CALUDE_tenth_day_is_monday_l1559_155956

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a schedule for a month -/
structure MonthSchedule where
  numDays : Nat
  firstDay : DayOfWeek
  runningDays : List DayOfWeek
  runningTimePerDay : Nat
  totalRunningTime : Nat

/-- Returns the day of the week for a given day of the month -/
def dayOfMonth (schedule : MonthSchedule) (day : Nat) : DayOfWeek :=
  sorry

theorem tenth_day_is_monday (schedule : MonthSchedule) :
  schedule.numDays = 31 ∧
  schedule.runningDays = [DayOfWeek.Monday, DayOfWeek.Saturday, DayOfWeek.Sunday] ∧
  schedule.runningTimePerDay = 20 ∧
  schedule.totalRunningTime = 5 * 60 →
  dayOfMonth schedule 10 = DayOfWeek.Monday :=
by sorry

end NUMINAMATH_CALUDE_tenth_day_is_monday_l1559_155956
