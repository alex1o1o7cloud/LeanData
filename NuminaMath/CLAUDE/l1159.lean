import Mathlib

namespace combined_work_time_l1159_115951

-- Define the work rates for A and B
def work_rate_A : ℚ := 1 / 4
def work_rate_B : ℚ := 1 / 12

-- Define the combined work rate
def combined_work_rate : ℚ := work_rate_A + work_rate_B

-- Theorem statement
theorem combined_work_time :
  (1 : ℚ) / combined_work_rate = 3 := by sorry

end combined_work_time_l1159_115951


namespace teresas_colored_pencils_l1159_115915

/-- Given information about Teresa's pencils and her siblings, prove the number of colored pencils she has. -/
theorem teresas_colored_pencils 
  (black_pencils : ℕ) 
  (num_siblings : ℕ) 
  (pencils_per_sibling : ℕ) 
  (pencils_kept : ℕ) 
  (h1 : black_pencils = 35)
  (h2 : num_siblings = 3)
  (h3 : pencils_per_sibling = 13)
  (h4 : pencils_kept = 10) :
  black_pencils + (num_siblings * pencils_per_sibling + pencils_kept) - black_pencils = 14 :=
by sorry

end teresas_colored_pencils_l1159_115915


namespace consecutive_integers_sum_l1159_115902

theorem consecutive_integers_sum (a b c : ℤ) : 
  (b = a + 1) → (c = b + 1) → (a * b * c = 336) → (a + b + c = 21) := by
sorry

end consecutive_integers_sum_l1159_115902


namespace green_apples_count_l1159_115970

theorem green_apples_count (total : ℕ) (red_to_green_ratio : ℕ) 
  (h1 : total = 496) 
  (h2 : red_to_green_ratio = 3) : 
  ∃ (green : ℕ), green = 124 ∧ total = green * (red_to_green_ratio + 1) :=
by
  sorry

end green_apples_count_l1159_115970


namespace cubic_meter_to_cubic_centimeters_l1159_115972

/-- Prove that one cubic meter is equal to 1,000,000 cubic centimeters -/
theorem cubic_meter_to_cubic_centimeters :
  (∀ m cm : ℕ, m = 100 * cm → m^3 = 1000000 * cm^3) :=
by sorry

end cubic_meter_to_cubic_centimeters_l1159_115972


namespace kola_sugar_percentage_l1159_115995

/-- Calculates the percentage of sugar in a kola solution after adding ingredients -/
theorem kola_sugar_percentage
  (initial_volume : Real)
  (initial_water_percent : Real)
  (initial_kola_percent : Real)
  (added_sugar : Real)
  (added_water : Real)
  (added_kola : Real)
  (h1 : initial_volume = 340)
  (h2 : initial_water_percent = 88)
  (h3 : initial_kola_percent = 5)
  (h4 : added_sugar = 3.2)
  (h5 : added_water = 10)
  (h6 : added_kola = 6.8) :
  let initial_sugar_percent := 100 - initial_water_percent - initial_kola_percent
  let initial_water := initial_volume * initial_water_percent / 100
  let initial_kola := initial_volume * initial_kola_percent / 100
  let initial_sugar := initial_volume * initial_sugar_percent / 100
  let final_water := initial_water + added_water
  let final_kola := initial_kola + added_kola
  let final_sugar := initial_sugar + added_sugar
  let final_volume := final_water + final_kola + final_sugar
  final_sugar / final_volume * 100 = 7.5 := by
  sorry


end kola_sugar_percentage_l1159_115995


namespace prime_sum_of_squares_and_divisibility_l1159_115971

theorem prime_sum_of_squares_and_divisibility (p : ℕ) : 
  Prime p → 
  (∃ m n : ℤ, (p : ℤ) = m^2 + n^2 ∧ (m^3 + n^3 - 4) % p = 0) → 
  p = 2 ∨ p = 5 := by
sorry

end prime_sum_of_squares_and_divisibility_l1159_115971


namespace apple_price_theorem_l1159_115908

/-- The price of apples with a two-tier pricing system -/
theorem apple_price_theorem 
  (l q : ℝ) 
  (h1 : 30 * l + 3 * q = 360) 
  (h2 : 30 * l + 6 * q = 420) : 
  25 * l = 250 := by
  sorry

end apple_price_theorem_l1159_115908


namespace banana_orange_equivalence_l1159_115964

/-- The cost of fruits at Zoe's Zesty Market -/
structure FruitCost where
  banana : ℕ
  apple : ℕ
  orange : ℕ

/-- The cost relationship between fruits -/
def cost_relationship (fc : FruitCost) : Prop :=
  5 * fc.banana = 4 * fc.apple ∧ 8 * fc.apple = 6 * fc.orange

/-- The theorem stating the equivalence of 40 bananas and 24 oranges in cost -/
theorem banana_orange_equivalence (fc : FruitCost) 
  (h : cost_relationship fc) : 40 * fc.banana = 24 * fc.orange := by
  sorry

#check banana_orange_equivalence

end banana_orange_equivalence_l1159_115964


namespace max_value_theorem_l1159_115904

theorem max_value_theorem (x y z : ℝ) (h : x + y + z = 3) :
  Real.sqrt (2 * x + 13) + (3 * y + 5) ^ (1/3) + (8 * z + 12) ^ (1/4) ≤ 8 := by
sorry

end max_value_theorem_l1159_115904


namespace existence_of_special_sequence_l1159_115976

theorem existence_of_special_sequence : ∃ (a b c : ℝ), 
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧
  (a + b + c = 6) ∧
  (b - a = c - b) ∧
  ((a^2 = b * c) ∨ (b^2 = a * c) ∨ (c^2 = a * b)) := by
  sorry

end existence_of_special_sequence_l1159_115976


namespace proportion_problem_l1159_115917

theorem proportion_problem : ∃ X : ℝ, (8 / 4 = X / 240) ∧ X = 480 := by
  sorry

end proportion_problem_l1159_115917


namespace least_integer_square_36_more_than_triple_l1159_115978

theorem least_integer_square_36_more_than_triple (x : ℤ) :
  (x^2 = 3*x + 36) → (x ≥ -6) :=
by sorry

end least_integer_square_36_more_than_triple_l1159_115978


namespace min_sum_m_n_l1159_115933

theorem min_sum_m_n (m n : ℕ+) (h : 45 * m = n^3) : 
  (∀ m' n' : ℕ+, 45 * m' = n'^3 → m' + n' ≥ m + n) → m + n = 90 := by
sorry

end min_sum_m_n_l1159_115933


namespace inscribed_circle_radius_right_triangle_l1159_115967

theorem inscribed_circle_radius_right_triangle 
  (a b c r : ℝ) 
  (h_right : a^2 + b^2 = c^2) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_inscribed : r > 0 ∧ r * (a + b + c) = a * b) : 
  r = (a + b - c) / 2 := by
sorry

end inscribed_circle_radius_right_triangle_l1159_115967


namespace softball_team_size_l1159_115929

theorem softball_team_size :
  ∀ (men women : ℕ),
  women = men + 2 →
  (men : ℚ) / (women : ℚ) = 7777777777777778 / 10000000000000000 →
  men + women = 16 :=
by
  sorry

end softball_team_size_l1159_115929


namespace solution_set_when_m_is_5_m_range_when_solution_set_is_real_l1159_115939

def f (x m : ℝ) : ℝ := |x + 1| + |x - 2| - m

theorem solution_set_when_m_is_5 :
  {x : ℝ | f x 5 > 0} = {x : ℝ | x < -2 ∨ x > 3} :=
sorry

theorem m_range_when_solution_set_is_real :
  (∀ x : ℝ, f x m ≥ 2) → m ≤ 1 :=
sorry

end solution_set_when_m_is_5_m_range_when_solution_set_is_real_l1159_115939


namespace ellen_painted_17_lilies_l1159_115969

/-- Time in minutes to paint each type of flower or vine -/
def lily_time : ℕ := 5
def rose_time : ℕ := 7
def orchid_time : ℕ := 3
def vine_time : ℕ := 2

/-- Total time spent painting -/
def total_time : ℕ := 213

/-- Number of roses, orchids, and vines painted -/
def roses : ℕ := 10
def orchids : ℕ := 6
def vines : ℕ := 20

/-- Function to calculate the number of lilies painted -/
def lilies_painted : ℕ := 
  (total_time - (roses * rose_time + orchids * orchid_time + vines * vine_time)) / lily_time

theorem ellen_painted_17_lilies : lilies_painted = 17 := by
  sorry

end ellen_painted_17_lilies_l1159_115969


namespace power_equation_solution_l1159_115994

theorem power_equation_solution : ∃ K : ℕ, (4 ^ 5) * (2 ^ 3) = 2 ^ K ∧ K = 13 := by
  sorry

end power_equation_solution_l1159_115994


namespace dice_roll_probability_l1159_115913

def probability_first_die : ℚ := 3 / 8
def probability_second_die : ℚ := 3 / 4

theorem dice_roll_probability :
  probability_first_die * probability_second_die = 9 / 32 := by
  sorry

end dice_roll_probability_l1159_115913


namespace player_current_average_l1159_115977

/-- Represents a cricket player's statistics -/
structure PlayerStats where
  matches_played : ℕ
  current_average : ℝ
  desired_increase : ℝ
  next_match_runs : ℕ

/-- Theorem stating the player's current average given the conditions -/
theorem player_current_average (player : PlayerStats)
  (h1 : player.matches_played = 10)
  (h2 : player.desired_increase = 4)
  (h3 : player.next_match_runs = 78) :
  player.current_average = 34 := by
  sorry

#check player_current_average

end player_current_average_l1159_115977


namespace geometric_series_sum_l1159_115905

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_series_sum :
  let a := 3 / 4
  let r := 3 / 4
  let n := 15
  geometric_sum a r n = 3177884751 / 1073741824 := by sorry

end geometric_series_sum_l1159_115905


namespace tower_configurations_count_l1159_115957

/-- The number of ways to build a tower of 10 cubes high using 3 red cubes, 4 blue cubes, and 5 yellow cubes, where two cubes are not used. -/
def towerConfigurations (red : Nat) (blue : Nat) (yellow : Nat) (towerHeight : Nat) : Nat :=
  sorry

/-- Theorem stating that the number of different tower configurations is 277,200 -/
theorem tower_configurations_count :
  towerConfigurations 3 4 5 10 = 277200 := by
  sorry

end tower_configurations_count_l1159_115957


namespace doll_collection_problem_l1159_115974

theorem doll_collection_problem (original_count : ℕ) : 
  (original_count + 2 : ℚ) = original_count * (1 + 1/4) → 
  original_count + 2 = 10 := by
sorry

end doll_collection_problem_l1159_115974


namespace saras_weekly_savings_l1159_115987

/-- Sara's weekly savings to match Jim's savings after 820 weeks -/
theorem saras_weekly_savings (sara_initial : ℕ) (jim_weekly : ℕ) (weeks : ℕ) : 
  sara_initial = 4100 → jim_weekly = 15 → weeks = 820 →
  ∃ (sara_weekly : ℕ), sara_initial + weeks * sara_weekly = weeks * jim_weekly := by
  sorry

#check saras_weekly_savings

end saras_weekly_savings_l1159_115987


namespace perfect_square_solutions_l1159_115946

theorem perfect_square_solutions : 
  {n : ℤ | ∃ m : ℤ, n^2 + 6*n + 24 = m^2} = {4, -2, -4, -10} := by sorry

end perfect_square_solutions_l1159_115946


namespace unique_balance_point_condition_l1159_115949

/-- A function f has a unique balance point if there exists a unique t such that f(t) = t -/
def has_unique_balance_point (f : ℝ → ℝ) : Prop :=
  ∃! t : ℝ, f t = t

/-- The quadratic function we're analyzing -/
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 - 3 * x + 2 * m

/-- Theorem stating the conditions for a unique balance point -/
theorem unique_balance_point_condition (m : ℝ) :
  has_unique_balance_point (f m) ↔ m = 2 ∨ m = -1 ∨ m = 1 := by sorry

end unique_balance_point_condition_l1159_115949


namespace compound_not_uniquely_determined_l1159_115936

/-- Represents a chemical compound -/
structure Compound where
  elements : List String
  mass_percentages : List Float
  mass_percentage_sum_eq_100 : mass_percentages.sum = 100

/-- A compound contains Cl with a mass percentage of 47.3% -/
def chlorine_compound : Compound := {
  elements := ["Cl", "Unknown"],
  mass_percentages := [47.3, 52.7],
  mass_percentage_sum_eq_100 := by sorry
}

/-- Predicate to check if a compound matches the given chlorine compound -/
def matches_chlorine_compound (c : Compound) : Prop :=
  "Cl" ∈ c.elements ∧ 47.3 ∈ c.mass_percentages

/-- Theorem stating that the compound cannot be uniquely determined -/
theorem compound_not_uniquely_determined :
  ∃ (c1 c2 : Compound), c1 ≠ c2 ∧ matches_chlorine_compound c1 ∧ matches_chlorine_compound c2 :=
by sorry

end compound_not_uniquely_determined_l1159_115936


namespace ellipse_hyperbola_same_foci_l1159_115911

/-- The value of m for which an ellipse and a hyperbola with given equations have the same foci -/
theorem ellipse_hyperbola_same_foci (m : ℝ) : 
  (∀ x y : ℝ, x^2 / 4 + y^2 / m^2 = 1 → ∃ c : ℝ, c^2 = 4 - m^2 ∧ (x = c ∨ x = -c) ∧ y = 0) →
  (∀ x y : ℝ, x^2 / m - y^2 / 2 = 1 → ∃ c : ℝ, c^2 = m + 2 ∧ (x = c ∨ x = -c) ∧ y = 0) →
  (∃ c : ℝ, c^2 = 4 - m^2 ∧ c^2 = m + 2) →
  m = 1 :=
by sorry

end ellipse_hyperbola_same_foci_l1159_115911


namespace mans_running_speed_l1159_115934

/-- Proves that given a man who walks at 8 kmph for 4 hours and 45 minutes,
    and runs the same distance in 120 minutes, his running speed is 19 kmph. -/
theorem mans_running_speed
  (walking_speed : ℝ)
  (walking_time_hours : ℝ)
  (walking_time_minutes : ℝ)
  (running_time_minutes : ℝ)
  (h1 : walking_speed = 8)
  (h2 : walking_time_hours = 4)
  (h3 : walking_time_minutes = 45)
  (h4 : running_time_minutes = 120)
  : (walking_speed * (walking_time_hours + walking_time_minutes / 60)) /
    (running_time_minutes / 60) = 19 := by
  sorry


end mans_running_speed_l1159_115934


namespace mixture_volume_l1159_115922

/-- Proves that the total volume of a mixture of two liquids is 4 liters -/
theorem mixture_volume (weight_a weight_b : ℝ) (ratio_a ratio_b : ℕ) (total_weight : ℝ) :
  weight_a = 950 →
  weight_b = 850 →
  ratio_a = 3 →
  ratio_b = 2 →
  total_weight = 3640 →
  ∃ (vol_a vol_b : ℝ),
    vol_a / vol_b = ratio_a / ratio_b ∧
    total_weight = vol_a * weight_a + vol_b * weight_b ∧
    vol_a + vol_b = 4 :=
by sorry

end mixture_volume_l1159_115922


namespace f_difference_l1159_115923

/-- k(n) is the largest odd divisor of n -/
def k (n : ℕ+) : ℕ+ := sorry

/-- f(n) is the sum of k(i) from i=1 to n -/
def f (n : ℕ+) : ℕ := sorry

/-- Theorem: f(2n) - f(n) = n^2 for any positive integer n -/
theorem f_difference (n : ℕ+) : f (2 * n) - f n = n^2 := by sorry

end f_difference_l1159_115923


namespace arithmetic_equality_l1159_115925

theorem arithmetic_equality : 45 * 52 + 28 * 45 = 3600 := by
  sorry

end arithmetic_equality_l1159_115925


namespace radio_price_calculation_l1159_115921

/-- Given a radio with 7% sales tax, if reducing its price by 161.46 results in a price of 2468,
    then the original price including sales tax is 2629.46. -/
theorem radio_price_calculation (original_price : ℝ) : 
  (original_price - 161.46 = 2468) → original_price = 2629.46 := by
  sorry

end radio_price_calculation_l1159_115921


namespace population_scientific_notation_l1159_115932

def population : ℝ := 1411750000

theorem population_scientific_notation : 
  population = 1.41175 * (10 : ℝ) ^ 9 :=
sorry

end population_scientific_notation_l1159_115932


namespace units_digit_problem_l1159_115968

theorem units_digit_problem : (8 * 18 * 1988 - 8^3) % 10 = 0 := by
  sorry

end units_digit_problem_l1159_115968


namespace k_range_l1159_115900

open Real

/-- The function f(x) = (ln x)/x - kx is increasing on (0, +∞) -/
def f_increasing (k : ℝ) : Prop :=
  ∀ x, x > 0 → Monotone (λ x => (log x) / x - k * x)

/-- The theorem to be proved -/
theorem k_range (k : ℝ) : f_increasing k → k ≤ -1 / (2 * Real.exp 3) := by
  sorry

end k_range_l1159_115900


namespace initial_persons_count_l1159_115998

/-- The number of persons initially in the group. -/
def initial_persons : ℕ := sorry

/-- The average weight increase when a new person joins the group. -/
def avg_weight_increase : ℚ := 7/2

/-- The weight difference between the new person and the replaced person. -/
def weight_difference : ℚ := 28

theorem initial_persons_count : initial_persons = 8 := by
  have h1 : (initial_persons : ℚ) * avg_weight_increase = weight_difference := by sorry
  sorry

end initial_persons_count_l1159_115998


namespace min_face_sum_l1159_115909

-- Define a cube as a set of 8 integers
def Cube := Fin 8 → ℕ

-- Define a face as a set of 4 vertices
def Face := Fin 4 → Fin 8

-- Condition: numbers are from 1 to 8
def valid_cube (c : Cube) : Prop :=
  (∀ i, c i ≥ 1 ∧ c i ≤ 8) ∧ (∀ i j, i ≠ j → c i ≠ c j)

-- Condition: sum of any three vertices on a face is at least 10
def valid_face_sums (c : Cube) (f : Face) : Prop :=
  ∀ i j k, i < j → j < k → c (f i) + c (f j) + c (f k) ≥ 10

-- The sum of numbers on a face
def face_sum (c : Cube) (f : Face) : ℕ :=
  (c (f 0)) + (c (f 1)) + (c (f 2)) + (c (f 3))

-- The theorem to prove
theorem min_face_sum (c : Cube) :
  valid_cube c → (∀ f : Face, valid_face_sums c f) →
  ∃ f : Face, face_sum c f = 16 ∧ ∀ g : Face, face_sum c g ≥ 16 :=
sorry

end min_face_sum_l1159_115909


namespace gcd_of_840_and_1764_l1159_115997

theorem gcd_of_840_and_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end gcd_of_840_and_1764_l1159_115997


namespace sum_mod_thirteen_l1159_115954

theorem sum_mod_thirteen : (5678 + 5679 + 5680 + 5681) % 13 = 4 := by
  sorry

end sum_mod_thirteen_l1159_115954


namespace workshop_sample_size_l1159_115937

/-- Calculates the sample size for a stratum in stratified sampling -/
def stratumSampleSize (totalPopulation : ℕ) (totalSampleSize : ℕ) (stratumSize : ℕ) : ℕ :=
  (totalSampleSize * stratumSize) / totalPopulation

theorem workshop_sample_size :
  let totalProducts : ℕ := 1024
  let sampleSize : ℕ := 64
  let workshopProduction : ℕ := 128
  stratumSampleSize totalProducts sampleSize workshopProduction = 8 := by
  sorry

end workshop_sample_size_l1159_115937


namespace initial_men_count_l1159_115962

/-- The number of men working initially -/
def initial_men : ℕ := 12

/-- The number of hours worked per day by the initial group -/
def initial_hours_per_day : ℕ := 8

/-- The number of days worked by the initial group -/
def initial_days : ℕ := 10

/-- The number of men in the new group -/
def new_men : ℕ := 6

/-- The number of hours worked per day by the new group -/
def new_hours_per_day : ℕ := 20

/-- The number of days worked by the new group -/
def new_days : ℕ := 8

/-- Theorem stating that the initial number of men is 12 -/
theorem initial_men_count : 
  initial_men * initial_hours_per_day * initial_days = 
  new_men * new_hours_per_day * new_days :=
by
  sorry

#check initial_men_count

end initial_men_count_l1159_115962


namespace greene_nursery_flower_count_l1159_115950

theorem greene_nursery_flower_count : 
  let red_roses : ℕ := 1491
  let yellow_carnations : ℕ := 3025
  let white_roses : ℕ := 1768
  red_roses + yellow_carnations + white_roses = 6284 :=
by sorry

end greene_nursery_flower_count_l1159_115950


namespace factorization_of_difference_of_squares_l1159_115958

theorem factorization_of_difference_of_squares (x : ℝ) :
  x^2 - 9 = (x + 3) * (x - 3) := by sorry

end factorization_of_difference_of_squares_l1159_115958


namespace geometric_sequence_decreasing_l1159_115927

def geometric_sequence (a₁ : ℝ) (q : ℝ) : ℕ → ℝ := fun n ↦ a₁ * q^(n-1)

theorem geometric_sequence_decreasing (a₁ q : ℝ) :
  (∀ n : ℕ, geometric_sequence a₁ q (n+1) < geometric_sequence a₁ q n) ↔
  ((a₁ > 0 ∧ 0 < q ∧ q < 1) ∨ (a₁ < 0 ∧ q > 1)) :=
sorry

end geometric_sequence_decreasing_l1159_115927


namespace total_spent_equals_sum_of_items_l1159_115941

/-- The total amount Joan spent on toys and clothes -/
def total_spent_on_toys_and_clothes : ℚ := 60.10

/-- The cost of toy cars -/
def toy_cars_cost : ℚ := 14.88

/-- The cost of the skateboard -/
def skateboard_cost : ℚ := 4.88

/-- The cost of toy trucks -/
def toy_trucks_cost : ℚ := 5.86

/-- The cost of pants -/
def pants_cost : ℚ := 14.55

/-- The cost of the shirt -/
def shirt_cost : ℚ := 7.43

/-- The cost of the hat -/
def hat_cost : ℚ := 12.50

/-- Theorem stating that the sum of the costs of toys and clothes equals the total amount spent -/
theorem total_spent_equals_sum_of_items :
  toy_cars_cost + skateboard_cost + toy_trucks_cost + pants_cost + shirt_cost + hat_cost = total_spent_on_toys_and_clothes :=
by sorry

end total_spent_equals_sum_of_items_l1159_115941


namespace today_is_thursday_l1159_115940

-- Define the days of the week
inductive Day : Type
  | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday

-- Define when A lies
def A_lies (d : Day) : Prop :=
  d = Day.Monday ∨ d = Day.Tuesday ∨ d = Day.Wednesday

-- Define when B lies
def B_lies (d : Day) : Prop :=
  d = Day.Thursday ∨ d = Day.Friday ∨ d = Day.Saturday

-- Define the previous day
def prev_day (d : Day) : Day :=
  match d with
  | Day.Monday => Day.Sunday
  | Day.Tuesday => Day.Monday
  | Day.Wednesday => Day.Tuesday
  | Day.Thursday => Day.Wednesday
  | Day.Friday => Day.Thursday
  | Day.Saturday => Day.Friday
  | Day.Sunday => Day.Saturday

-- Theorem statement
theorem today_is_thursday : 
  ∃ (d : Day), 
    (A_lies (prev_day d) ↔ ¬(A_lies d)) ∧ 
    (B_lies (prev_day d) ↔ ¬(B_lies d)) ∧ 
    d = Day.Thursday := by
  sorry

end today_is_thursday_l1159_115940


namespace sand_pile_volume_l1159_115910

/-- The volume of a conical sand pile -/
theorem sand_pile_volume (d h r : ℝ) : 
  d = 10 →  -- diameter is 10 feet
  h = 0.6 * d →  -- height is 60% of diameter
  r = d / 2 →  -- radius is half of diameter
  (1 / 3) * π * r^2 * h = 50 * π := by
  sorry

end sand_pile_volume_l1159_115910


namespace briannas_books_l1159_115935

/-- Brianna's book reading problem -/
theorem briannas_books :
  let books_per_year : ℕ := 24
  let gift_books : ℕ := 6
  let old_books : ℕ := 4
  let bought_books : ℕ := x
  let borrowed_books : ℕ := x - 2

  gift_books + bought_books + borrowed_books + old_books = books_per_year →
  bought_books = 8 := by
  sorry

end briannas_books_l1159_115935


namespace largest_subarray_sum_l1159_115945

/-- A type representing a 5x5 array of natural numbers -/
def Array5x5 := Fin 5 → Fin 5 → ℕ

/-- Predicate to check if an array contains distinct numbers from 1 to 25 -/
def isValidArray (a : Array5x5) : Prop :=
  ∀ i j, 1 ≤ a i j ∧ a i j ≤ 25 ∧
  ∀ i' j', (i ≠ i' ∨ j ≠ j') → a i j ≠ a i' j'

/-- Sum of a 2x2 subarray starting at position (i, j) -/
def subarraySum (a : Array5x5) (i j : Fin 4) : ℕ :=
  a i j + a i (j + 1) + a (i + 1) j + a (i + 1) (j + 1)

/-- Theorem stating that 45 is the largest N satisfying the given property -/
theorem largest_subarray_sum : 
  (∀ a : Array5x5, isValidArray a → ∀ i j : Fin 4, subarraySum a i j ≥ 45) ∧
  ¬(∀ a : Array5x5, isValidArray a → ∀ i j : Fin 4, subarraySum a i j ≥ 46) :=
sorry

end largest_subarray_sum_l1159_115945


namespace elberta_money_l1159_115965

theorem elberta_money (granny_smith : ℕ) (anjou elberta : ℝ) : 
  granny_smith = 72 →
  anjou = (1 / 4 : ℝ) * granny_smith →
  elberta = anjou + 3 →
  elberta = 21 := by sorry

end elberta_money_l1159_115965


namespace max_circle_sum_is_15_l1159_115996

/-- Represents a configuration of numbers in the circle diagram -/
def CircleConfiguration := Fin 7 → Fin 7

/-- The sum of numbers in a given circle of the configuration -/
def circle_sum (config : CircleConfiguration) (circle : Fin 3) : ℕ :=
  sorry

/-- Checks if a configuration is valid (uses all numbers 0 to 6 exactly once) -/
def is_valid_configuration (config : CircleConfiguration) : Prop :=
  sorry

/-- Checks if all circles in a configuration have the same sum -/
def all_circles_equal_sum (config : CircleConfiguration) : Prop :=
  sorry

/-- The maximum possible sum for each circle -/
def max_circle_sum : ℕ := 15

theorem max_circle_sum_is_15 :
  ∃ (config : CircleConfiguration),
    is_valid_configuration config ∧
    all_circles_equal_sum config ∧
    ∀ (c : Fin 3), circle_sum config c = max_circle_sum ∧
    ∀ (config' : CircleConfiguration),
      is_valid_configuration config' →
      all_circles_equal_sum config' →
      ∀ (c : Fin 3), circle_sum config' c ≤ max_circle_sum :=
sorry

end max_circle_sum_is_15_l1159_115996


namespace arithmetic_geometric_sequence_l1159_115980

theorem arithmetic_geometric_sequence : 
  ∃ (a b c : ℝ), 
    (a > 0 ∧ b > 0 ∧ c > 0) ∧ 
    (b - a = c - b) ∧ 
    (a + b + c = 15) ∧ 
    ((a + 1) * (c + 9) = (b + 3)^2) ∧
    (a = 1 ∧ b = 5 ∧ c = 9) := by
  sorry

end arithmetic_geometric_sequence_l1159_115980


namespace smallest_x_absolute_value_equation_l1159_115912

theorem smallest_x_absolute_value_equation : 
  (∀ x : ℝ, |5*x + 15| = 40 → x ≥ -11) ∧ 
  (|5*(-11) + 15| = 40) := by
  sorry

end smallest_x_absolute_value_equation_l1159_115912


namespace det_linear_combination_zero_l1159_115975

open Matrix

theorem det_linear_combination_zero
  (A B : Matrix (Fin 3) (Fin 3) ℝ)
  (h : A ^ 2 + B ^ 2 = 0) :
  ∀ (a b : ℝ), det (a • A + b • B) = 0 := by
sorry

end det_linear_combination_zero_l1159_115975


namespace makeup_exam_average_score_l1159_115943

/-- Represents the average score of students who took the exam on the make-up date -/
def makeup_avg : ℝ := 90

theorem makeup_exam_average_score 
  (total_students : ℕ) 
  (assigned_day_percent : ℝ) 
  (assigned_day_avg : ℝ) 
  (total_avg : ℝ) 
  (h1 : total_students = 100)
  (h2 : assigned_day_percent = 70)
  (h3 : assigned_day_avg = 60)
  (h4 : total_avg = 69) :
  makeup_avg = 90 := by
  sorry

#check makeup_exam_average_score

end makeup_exam_average_score_l1159_115943


namespace solution_set_inequality_l1159_115990

open Set
open Function
open Real

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- Theorem statement
theorem solution_set_inequality 
  (h_domain : ∀ x, x > 0 → DifferentiableAt ℝ f x)
  (h_derivative : ∀ x, x > 0 → deriv f x = f' x)
  (h_inequality : ∀ x, x > 0 → f x > f' x) :
  {x : ℝ | Real.exp (x + 2) * f (x^2 - x) > Real.exp (x^2) * f 2} = 
  Ioo (-1) 0 ∪ Ioo 1 2 := by
sorry

end solution_set_inequality_l1159_115990


namespace object3_length_is_15_l1159_115956

def longest_tape : ℕ := 5

def object1_length : ℕ := 225
def object2_length : ℕ := 780

def object3_length : ℕ := Nat.gcd object1_length object2_length

theorem object3_length_is_15 :
  longest_tape = 5 ∧
  object1_length = 225 ∧
  object2_length = 780 ∧
  object3_length = Nat.gcd object1_length object2_length →
  object3_length = 15 :=
by sorry

end object3_length_is_15_l1159_115956


namespace rex_cards_left_is_150_l1159_115959

/-- The number of Pokemon cards Rex has left after dividing his collection --/
def rexCardsLeft (nicolesCards : ℕ) : ℕ :=
  let cindysCards := nicolesCards * 2
  let totalCards := nicolesCards + cindysCards
  let rexCards := totalCards / 2
  rexCards / 4

/-- Theorem stating that Rex has 150 cards left --/
theorem rex_cards_left_is_150 : rexCardsLeft 400 = 150 := by
  sorry

end rex_cards_left_is_150_l1159_115959


namespace binomial_divides_lcm_l1159_115907

theorem binomial_divides_lcm (n : ℕ) (h : n ≥ 1) :
  ∃ k : ℕ, k * Nat.choose (2 * n) n = Finset.lcm (Finset.range (2 * n + 1)) id :=
by sorry

end binomial_divides_lcm_l1159_115907


namespace sine_inequality_l1159_115981

theorem sine_inequality (x y : ℝ) (h1 : 0 < x) (h2 : x < y) (h3 : y < 1) :
  0 < Real.sin x ∧ Real.sin x < Real.sin y := by sorry

end sine_inequality_l1159_115981


namespace shirt_price_calculation_l1159_115947

/-- The original price of the shirt -/
def shirt_price : ℝ := 156.52

/-- The original price of the coat -/
def coat_price : ℝ := 3 * shirt_price

/-- The original price of the pants -/
def pants_price : ℝ := 2 * shirt_price

/-- The total cost after discounts -/
def total_cost : ℝ := 900

theorem shirt_price_calculation :
  (shirt_price * 0.9 + coat_price * 0.95 + pants_price) = total_cost :=
by sorry

end shirt_price_calculation_l1159_115947


namespace fraction_zero_implies_a_equals_two_l1159_115944

theorem fraction_zero_implies_a_equals_two (a : ℝ) 
  (h1 : (a^2 - 4) / (a + 2) = 0) 
  (h2 : a + 2 ≠ 0) : 
  a = 2 := by
sorry

end fraction_zero_implies_a_equals_two_l1159_115944


namespace remaining_savings_is_25_70_l1159_115963

/-- Calculates the remaining savings after jewelry purchases and tax --/
def remaining_savings (initial_savings earrings_cost necklace_cost bracelet_cost jewelry_set_cost discount_percent tax_percent : ℚ) : ℚ :=
  let individual_items_cost := earrings_cost + necklace_cost + bracelet_cost
  let discounted_jewelry_set_cost := jewelry_set_cost * (1 - discount_percent / 100)
  let total_before_tax := individual_items_cost + discounted_jewelry_set_cost
  let tax_amount := total_before_tax * (tax_percent / 100)
  let final_total_cost := total_before_tax + tax_amount
  initial_savings - final_total_cost

/-- Theorem stating that the remaining savings are $25.70 --/
theorem remaining_savings_is_25_70 :
  remaining_savings 200 23 48 35 80 25 5 = 25.70 := by
  sorry

end remaining_savings_is_25_70_l1159_115963


namespace chips_sales_problem_l1159_115938

theorem chips_sales_problem (total_sales : ℕ) (first_week : ℕ) (second_week : ℕ) :
  total_sales = 100 →
  first_week = 15 →
  second_week = 3 * first_week →
  ∃ (third_fourth_week : ℕ),
    third_fourth_week * 2 = total_sales - (first_week + second_week) ∧
    third_fourth_week = 20 := by
  sorry

end chips_sales_problem_l1159_115938


namespace abc_equation_solution_l1159_115961

theorem abc_equation_solution (a b c : ℕ+) (h1 : b ≤ c) 
  (h2 : (a * b - 1) * (a * c - 1) = 2023 * b * c) : 
  c = 82 ∨ c = 167 ∨ c = 1034 := by
sorry

end abc_equation_solution_l1159_115961


namespace squares_not_always_congruent_l1159_115983

-- Define a square
structure Square where
  side_length : ℝ
  side_length_pos : side_length > 0

-- Define properties of squares
def Square.is_equiangular (s : Square) : Prop := True
def Square.is_rectangle (s : Square) : Prop := True
def Square.is_regular_polygon (s : Square) : Prop := True
def Square.is_similar_to (s1 s2 : Square) : Prop := True

-- Define congruence for squares
def Square.is_congruent_to (s1 s2 : Square) : Prop :=
  s1.side_length = s2.side_length

-- Theorem statement
theorem squares_not_always_congruent :
  ∃ (s1 s2 : Square),
    s1.is_equiangular ∧
    s1.is_rectangle ∧
    s1.is_regular_polygon ∧
    s2.is_equiangular ∧
    s2.is_rectangle ∧
    s2.is_regular_polygon ∧
    Square.is_similar_to s1 s2 ∧
    ¬ Square.is_congruent_to s1 s2 :=
by
  sorry

end squares_not_always_congruent_l1159_115983


namespace total_schedules_l1159_115979

/-- Represents the number of periods in a day -/
def total_periods : ℕ := 6

/-- Represents the number of morning periods -/
def morning_periods : ℕ := 3

/-- Represents the number of afternoon periods -/
def afternoon_periods : ℕ := 3

/-- Represents the total number of subjects -/
def total_subjects : ℕ := 6

/-- Represents the number of ways to schedule Mathematics in the morning and Art in the afternoon -/
def math_art_schedules : ℕ := morning_periods * afternoon_periods

/-- Represents the number of remaining subjects to be scheduled -/
def remaining_subjects : ℕ := total_subjects - 2

/-- Represents the number of remaining periods to schedule the remaining subjects -/
def remaining_periods : ℕ := total_periods - 2

/-- The main theorem stating the total number of possible schedules -/
theorem total_schedules : 
  math_art_schedules * (Nat.factorial remaining_subjects) = 216 :=
sorry

end total_schedules_l1159_115979


namespace quadratic_roots_sum_product_l1159_115986

theorem quadratic_roots_sum_product (k p : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 - k * x + p = 0 ∧ 3 * y^2 - k * y + p = 0 ∧ x + y = -3 ∧ x * y = -6) →
  k + p = -27 := by
  sorry

end quadratic_roots_sum_product_l1159_115986


namespace absolute_value_equation_range_l1159_115930

theorem absolute_value_equation_range (x : ℝ) : 
  |x - 1| + x - 1 = 0 → x ≤ 1 := by
  sorry

end absolute_value_equation_range_l1159_115930


namespace binomial_17_9_l1159_115984

theorem binomial_17_9 (h1 : Nat.choose 15 6 = 5005) (h2 : Nat.choose 15 8 = 6435) :
  Nat.choose 17 9 = 24310 := by
  sorry

end binomial_17_9_l1159_115984


namespace geometric_arithmetic_sequence_ratio_l1159_115942

theorem geometric_arithmetic_sequence_ratio (x y z : ℝ) 
  (h1 : (4 * y) / (3 * x) = (5 * z) / (4 * y))  -- geometric sequence condition
  (h2 : 1 / y - 1 / x = 1 / z - 1 / y)         -- arithmetic sequence condition
  (h3 : x ≠ 0)
  (h4 : y ≠ 0)
  (h5 : z ≠ 0) :
  x / z + z / x = 34 / 15 := by
sorry

end geometric_arithmetic_sequence_ratio_l1159_115942


namespace quadratic_two_distinct_roots_root_two_implies_k_value_l1159_115973

/-- The quadratic equation k^2*x^2 + 2*(k-1)*x + 1 = 0 -/
def quadratic_equation (k x : ℝ) : Prop :=
  k^2 * x^2 + 2*(k-1)*x + 1 = 0

/-- The discriminant of the quadratic equation -/
def discriminant (k : ℝ) : ℝ :=
  4*(k-1)^2 - 4*k^2

theorem quadratic_two_distinct_roots (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ quadratic_equation k x ∧ quadratic_equation k y) ↔
  (k < 1/2 ∧ k ≠ 0) :=
sorry

theorem root_two_implies_k_value :
  ∀ k : ℝ, quadratic_equation k 2 → k = -3/2 :=
sorry

end quadratic_two_distinct_roots_root_two_implies_k_value_l1159_115973


namespace quadratic_intersection_points_l1159_115992

/-- A quadratic function with at least one y-intercept -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  has_y_intercept : ∃ x, a * x^2 + b * x + c = 0

/-- The number of intersection points between f(x) and -f(-x) -/
def intersection_points_f_u (f : QuadraticFunction) : ℕ := 1

/-- The number of intersection points between f(x) and f(x+1) -/
def intersection_points_f_v (f : QuadraticFunction) : ℕ := 0

/-- The main theorem -/
theorem quadratic_intersection_points (f : QuadraticFunction) :
  7 * (intersection_points_f_u f) + 3 * (intersection_points_f_v f) = 7 := by
  sorry

end quadratic_intersection_points_l1159_115992


namespace soccer_field_kids_l1159_115919

/-- Given an initial number of kids on a soccer field and the number of friends each kid invites,
    calculate the total number of kids on the field after invitations. -/
def total_kids_after_invitations (initial_kids : ℕ) (friends_per_kid : ℕ) : ℕ :=
  initial_kids + initial_kids * friends_per_kid

/-- Theorem: If there are initially 14 kids on a soccer field and each kid invites 3 friends,
    then the total number of kids on the field after invitations is 56. -/
theorem soccer_field_kids : total_kids_after_invitations 14 3 = 56 := by
  sorry

end soccer_field_kids_l1159_115919


namespace ellipse_k_value_l1159_115985

/-- An ellipse with equation 4x² + ky² = 4 and a focus at (0, 1) has k = 2 -/
theorem ellipse_k_value (k : ℝ) : 
  (∀ x y : ℝ, 4 * x^2 + k * y^2 = 4) →  -- Ellipse equation
  (0, 1) ∈ {p : ℝ × ℝ | p.1^2 / 1^2 + p.2^2 / (4/k) = 1} →  -- Focus condition
  k = 2 :=
by sorry

end ellipse_k_value_l1159_115985


namespace triangle_side_length_l1159_115924

theorem triangle_side_length 
  (AB : ℝ) 
  (angle_ADB : ℝ) 
  (sin_A : ℝ) 
  (sin_C : ℝ) 
  (h1 : AB = 30)
  (h2 : angle_ADB = Real.pi / 2)
  (h3 : sin_A = 2/3)
  (h4 : sin_C = 1/4) :
  ∃ (DC : ℝ), DC = 20 * Real.sqrt 15 := by
  sorry

end triangle_side_length_l1159_115924


namespace points_on_parabola_l1159_115952

-- Define the function y = x^2
def f (x : ℝ) : ℝ := x^2

-- Theorem statement
theorem points_on_parabola :
  ∀ t : ℝ, ∃ p₁ p₂ : ℝ × ℝ,
    p₁ = (1, f 1) ∧
    p₂ = (t, f t) ∧
    (p₁.2 = f p₁.1) ∧
    (p₂.2 = f p₂.1) :=
by
  sorry


end points_on_parabola_l1159_115952


namespace parabola_p_value_l1159_115931

/-- The latus rectum of a parabola y^2 = 2px --/
def latus_rectum (p : ℝ) : ℝ := 4 * p

/-- Theorem: For a parabola y^2 = 2px with latus rectum equal to 4, p equals 2 --/
theorem parabola_p_value : ∀ p : ℝ, latus_rectum p = 4 → p = 2 := by
  sorry

end parabola_p_value_l1159_115931


namespace elaine_rent_percentage_l1159_115999

/-- Represents Elaine's earnings and rent expenses over two years -/
structure ElaineFinances where
  lastYearEarnings : ℝ
  lastYearRentPercentage : ℝ
  earningsIncrease : ℝ
  rentIncrease : ℝ
  thisYearRentPercentage : ℝ

/-- The conditions of Elaine's finances -/
def elaineFinancesConditions (e : ElaineFinances) : Prop :=
  e.lastYearRentPercentage = 20 ∧
  e.earningsIncrease = 35 ∧
  e.rentIncrease = 202.5

/-- Theorem stating that given the conditions, Elaine's rent percentage this year is 30% -/
theorem elaine_rent_percentage (e : ElaineFinances) 
  (h : elaineFinancesConditions e) : e.thisYearRentPercentage = 30 := by
  sorry


end elaine_rent_percentage_l1159_115999


namespace asian_games_mascot_sales_l1159_115906

/-- Asian Games Mascot Sales Problem -/
theorem asian_games_mascot_sales 
  (initial_price : ℝ) 
  (cost_price : ℝ) 
  (initial_sales : ℝ) 
  (price_reduction_factor : ℝ) :
  initial_price = 80 ∧ 
  cost_price = 50 ∧ 
  initial_sales = 200 ∧ 
  price_reduction_factor = 20 →
  ∃ (sales_function : ℝ → ℝ) 
    (profit_function : ℝ → ℝ) 
    (optimal_price : ℝ),
    (∀ x, sales_function x = -20 * x + 1800) ∧
    (profit_function 65 = 7500 ∧ profit_function 75 = 7500) ∧
    (optimal_price = 70 ∧ 
     ∀ x, profit_function x ≤ profit_function optimal_price) :=
by sorry

end asian_games_mascot_sales_l1159_115906


namespace furniture_shop_cost_price_l1159_115901

/-- 
Given a furniture shop where the owner charges 25% more than the cost price,
this theorem proves that if a customer pays Rs. 1000 for an item, 
then the cost price of that item is Rs. 800.
-/
theorem furniture_shop_cost_price 
  (markup_percentage : ℝ) 
  (selling_price : ℝ) 
  (h1 : markup_percentage = 25)
  (h2 : selling_price = 1000) :
  let cost_price := selling_price / (1 + markup_percentage / 100)
  cost_price = 800 := by
  sorry

end furniture_shop_cost_price_l1159_115901


namespace circle_constant_l1159_115914

/-- Theorem: For a circle with equation x^2 + 10x + y^2 + 8y + c = 0 and radius 5, the value of c is 16. -/
theorem circle_constant (c : ℝ) : 
  (∀ x y : ℝ, x^2 + 10*x + y^2 + 8*y + c = 0 ↔ (x+5)^2 + (y+4)^2 = 25) → 
  c = 16 := by
sorry

end circle_constant_l1159_115914


namespace cloth_sale_meters_l1159_115928

/-- Proves that the number of meters of cloth sold is 75 given the total selling price,
    profit per meter, and cost price per meter. -/
theorem cloth_sale_meters (total_selling_price : ℕ) (profit_per_meter : ℕ) (cost_price_per_meter : ℕ)
    (h1 : total_selling_price = 4950)
    (h2 : profit_per_meter = 15)
    (h3 : cost_price_per_meter = 51) :
    (total_selling_price / (cost_price_per_meter + profit_per_meter) : ℕ) = 75 := by
  sorry

end cloth_sale_meters_l1159_115928


namespace regular_octagon_diagonal_ratio_l1159_115903

/-- The ratio of the shortest diagonal to the longest diagonal in a regular octagon -/
theorem regular_octagon_diagonal_ratio : 
  ∃ (shortest_diagonal longest_diagonal : ℝ), 
    shortest_diagonal > 0 ∧ 
    longest_diagonal > 0 ∧
    shortest_diagonal / longest_diagonal = Real.sqrt 2 / 2 := by
  sorry

end regular_octagon_diagonal_ratio_l1159_115903


namespace repeating_decimal_equiv_fraction_fraction_in_lowest_terms_l1159_115920

def repeating_decimal_to_fraction (a b c : ℕ) : ℚ :=
  (a : ℚ) + (b * 10 + c : ℚ) / 990

theorem repeating_decimal_equiv_fraction :
  repeating_decimal_to_fraction 4 1 7 = 413 / 990 :=
sorry

theorem fraction_in_lowest_terms : ∀ n : ℕ, n > 1 → n ∣ 413 → n ∣ 990 → False :=
sorry

#eval repeating_decimal_to_fraction 4 1 7

end repeating_decimal_equiv_fraction_fraction_in_lowest_terms_l1159_115920


namespace divisibility_by_seven_l1159_115918

theorem divisibility_by_seven (k : ℕ) : 
  (∀ n : ℕ, n > 0 → ∃ q : ℤ, (3^(6*n - 1) - k * 2^(3*n - 2) + 1 : ℤ) = 7 * q) ↔ 
  (∃ m : ℤ, k = 7 * m + 3) :=
sorry

end divisibility_by_seven_l1159_115918


namespace intersection_value_l1159_115926

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

/-- Represents a curve in polar coordinates -/
structure PolarCurve where
  equation : PolarPoint → Prop

def C₁ : PolarCurve :=
  { equation := fun p => p.ρ * (Real.cos p.θ + Real.sin p.θ) = 1 }

def C₂ (a : ℝ) : PolarCurve :=
  { equation := fun p => p.ρ = a }

def onPolarAxis (p : PolarPoint) : Prop :=
  p.θ = 0 ∨ p.θ = Real.pi

theorem intersection_value (a : ℝ) (h₁ : a > 0) :
  (∃ p : PolarPoint, C₁.equation p ∧ (C₂ a).equation p ∧ onPolarAxis p) →
  a = Real.sqrt 2 / 2 := by
  sorry

end intersection_value_l1159_115926


namespace trapezoid_area_l1159_115993

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  p : Point
  q : Point
  r : Point
  s : Point
  area : ℝ

/-- Represents a trapezoid -/
structure Trapezoid where
  t : Point
  u : Point
  v : Point
  s : Point

/-- Given a rectangle PQRS and points T, U, V forming a trapezoid TUVS, 
    prove that the area of TUVS is 10 square units -/
theorem trapezoid_area 
  (pqrs : Rectangle)
  (t : Point)
  (u : Point)
  (v : Point)
  (h1 : pqrs.area = 20)
  (h2 : t.x - pqrs.p.x = 2)
  (h3 : t.y = pqrs.p.y)
  (h4 : u.x - pqrs.q.x = 2)
  (h5 : u.y = pqrs.r.y)
  (h6 : v.x = pqrs.r.x)
  (h7 : v.y - t.y = pqrs.r.y - pqrs.p.y)
  : ∃ (tuvs : Trapezoid), tuvs.t = t ∧ tuvs.u = u ∧ tuvs.v = v ∧ tuvs.s = pqrs.s ∧ 
    (tuvs.v.x - tuvs.t.x + tuvs.s.x - tuvs.u.x) * (tuvs.u.y - tuvs.t.y) / 2 = 10 :=
by sorry

end trapezoid_area_l1159_115993


namespace player_B_most_stable_l1159_115982

/-- Represents a player in the shooting test -/
inductive Player : Type
  | A
  | B
  | C
  | D

/-- Returns the variance of a given player -/
def variance (p : Player) : ℝ :=
  match p with
  | Player.A => 0.66
  | Player.B => 0.52
  | Player.C => 0.58
  | Player.D => 0.62

/-- Defines what it means for a player to have the most stable performance -/
def has_most_stable_performance (p : Player) : Prop :=
  ∀ q : Player, variance p ≤ variance q

/-- Theorem stating that Player B has the most stable performance -/
theorem player_B_most_stable :
  has_most_stable_performance Player.B := by
  sorry

end player_B_most_stable_l1159_115982


namespace seconds_in_day_scientific_notation_l1159_115916

/-- The number of seconds in a day -/
def seconds_in_day : ℕ := 86400

/-- Scientific notation representation of seconds in a day -/
def scientific_notation : ℝ := 8.64 * (10 ^ 4)

theorem seconds_in_day_scientific_notation :
  (seconds_in_day : ℝ) = scientific_notation := by sorry

end seconds_in_day_scientific_notation_l1159_115916


namespace dry_grapes_weight_l1159_115948

/-- Calculates the weight of dry grapes obtained from fresh grapes -/
theorem dry_grapes_weight
  (fresh_water_content : Real)
  (dry_water_content : Real)
  (fresh_weight : Real)
  (h1 : fresh_water_content = 0.90)
  (h2 : dry_water_content = 0.20)
  (h3 : fresh_weight = 20)
  : Real :=
by
  -- The weight of dry grapes obtained from fresh_weight of fresh grapes
  -- is equal to 2.5
  sorry

#check dry_grapes_weight

end dry_grapes_weight_l1159_115948


namespace stewart_farm_ratio_l1159_115989

/-- The ratio of sheep to horses at Stewart farm -/
theorem stewart_farm_ratio : 
  ∀ (num_sheep num_horses : ℕ) (food_per_horse total_horse_food : ℕ),
  num_sheep = 24 →
  food_per_horse = 230 →
  total_horse_food = 12880 →
  num_horses * food_per_horse = total_horse_food →
  (num_sheep : ℚ) / (num_horses : ℚ) = 3 / 7 := by
  sorry

end stewart_farm_ratio_l1159_115989


namespace bakery_flour_calculation_l1159_115966

/-- Given a bakery that uses wheat flour and white flour, prove that the amount of white flour
    used is equal to the total amount of flour used minus the amount of wheat flour used. -/
theorem bakery_flour_calculation (total_flour white_flour wheat_flour : ℝ) 
    (h1 : total_flour = 0.3)
    (h2 : wheat_flour = 0.2) :
  white_flour = total_flour - wheat_flour := by
  sorry

end bakery_flour_calculation_l1159_115966


namespace grains_per_teaspoon_l1159_115991

/-- Represents the number of grains of rice in one cup -/
def grains_per_cup : ℕ := 480

/-- Represents the number of tablespoons in half a cup -/
def tablespoons_per_half_cup : ℕ := 8

/-- Represents the number of teaspoons in one tablespoon -/
def teaspoons_per_tablespoon : ℕ := 3

/-- Theorem stating that there are 10 grains of rice in a teaspoon -/
theorem grains_per_teaspoon :
  (grains_per_cup / (2 * tablespoons_per_half_cup * teaspoons_per_tablespoon)) = 10 := by
  sorry

end grains_per_teaspoon_l1159_115991


namespace rectangular_field_area_l1159_115953

theorem rectangular_field_area (x y : ℝ) 
  (h1 : x + y = 20) 
  (h2 : y = 9) : 
  x * y = 99 := by
  sorry

end rectangular_field_area_l1159_115953


namespace total_investment_total_investment_is_6647_l1159_115960

/-- The problem of calculating total investments --/
theorem total_investment (raghu_investment : ℕ) : ℕ :=
  let trishul_investment := raghu_investment - raghu_investment / 10
  let vishal_investment := trishul_investment + trishul_investment / 10
  raghu_investment + trishul_investment + vishal_investment

/-- The theorem stating that the total investment is 6647 when Raghu invests 2300 --/
theorem total_investment_is_6647 : total_investment 2300 = 6647 := by
  sorry

end total_investment_total_investment_is_6647_l1159_115960


namespace dividend_calculation_l1159_115988

theorem dividend_calculation (remainder : ℕ) (divisor : ℕ) (quotient : ℕ) :
  remainder = 6 →
  divisor = 5 * quotient →
  divisor = 3 * remainder + 2 →
  divisor * quotient + remainder = 86 := by
  sorry

end dividend_calculation_l1159_115988


namespace full_price_revenue_l1159_115955

/-- Represents the fundraiser scenario -/
structure Fundraiser where
  total_tickets : ℕ
  total_revenue : ℚ
  full_price : ℚ
  full_price_tickets : ℕ

/-- The fundraiser satisfies the given conditions -/
def valid_fundraiser (f : Fundraiser) : Prop :=
  f.total_tickets = 180 ∧
  f.total_revenue = 2600 ∧
  f.full_price > 0 ∧
  f.full_price_tickets ≤ f.total_tickets ∧
  f.full_price_tickets * f.full_price + (f.total_tickets - f.full_price_tickets) * (f.full_price / 3) = f.total_revenue

/-- The theorem stating that the revenue from full-price tickets is $975 -/
theorem full_price_revenue (f : Fundraiser) (h : valid_fundraiser f) : 
  f.full_price_tickets * f.full_price = 975 := by
  sorry

end full_price_revenue_l1159_115955
