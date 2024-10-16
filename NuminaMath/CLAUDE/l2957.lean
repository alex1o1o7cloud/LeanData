import Mathlib

namespace NUMINAMATH_CALUDE_absolute_difference_inequality_characterization_l2957_295790

def absolute_difference_inequality (a : ℝ) : Set ℝ :=
  {x : ℝ | |x - 1| - |x - 2| < a}

theorem absolute_difference_inequality_characterization (a : ℝ) :
  (absolute_difference_inequality a = Set.univ ↔ a > 1) ∧
  (absolute_difference_inequality a ≠ ∅ ↔ a > -1) ∧
  (absolute_difference_inequality a = ∅ ↔ a ≤ -1) :=
sorry

end NUMINAMATH_CALUDE_absolute_difference_inequality_characterization_l2957_295790


namespace NUMINAMATH_CALUDE_percentage_problem_l2957_295700

theorem percentage_problem (x : ℝ) (h : 0.4 * x = 160) : 0.6 * x = 240 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2957_295700


namespace NUMINAMATH_CALUDE_surface_area_between_cylinders_l2957_295732

/-- The total surface area of the space between two concentric cylinders -/
theorem surface_area_between_cylinders (h inner_radius : ℝ) 
  (h_pos : h > 0) (inner_radius_pos : inner_radius > 0) :
  let outer_radius := inner_radius + 1
  2 * π * h * (outer_radius - inner_radius) = 16 * π := by
  sorry

end NUMINAMATH_CALUDE_surface_area_between_cylinders_l2957_295732


namespace NUMINAMATH_CALUDE_batsman_average_after_12th_innings_average_increase_by_2_min_score_before_12th_consecutive_scores_before_12th_l2957_295789

/-- Represents a cricket batsman's performance -/
structure Batsman where
  innings : Nat
  totalRuns : Nat
  minScore : Nat
  consecutiveScores : Nat

/-- Calculates the average score of a batsman -/
def average (b : Batsman) : Rat :=
  b.totalRuns / b.innings

/-- Represents the batsman's performance after 11 innings -/
def initialBatsman : Batsman :=
  { innings := 11
  , totalRuns := 11 * 24  -- 11 * average before 12th innings
  , minScore := 20
  , consecutiveScores := 25 }

/-- Represents the batsman's performance after 12 innings -/
def finalBatsman : Batsman :=
  { innings := 12
  , totalRuns := initialBatsman.totalRuns + 48
  , minScore := 20
  , consecutiveScores := 25 }

theorem batsman_average_after_12th_innings :
  average finalBatsman = 26 := by
  sorry

theorem average_increase_by_2 :
  average finalBatsman - average initialBatsman = 2 := by
  sorry

theorem min_score_before_12th :
  initialBatsman.minScore ≥ 20 := by
  sorry

theorem consecutive_scores_before_12th :
  initialBatsman.consecutiveScores = 25 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_after_12th_innings_average_increase_by_2_min_score_before_12th_consecutive_scores_before_12th_l2957_295789


namespace NUMINAMATH_CALUDE_parabola_transformation_l2957_295725

/-- Represents a parabola in the Cartesian plane -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Shifts a parabola horizontally -/
def shift_horizontal (p : Parabola) (dx : ℝ) : Parabola :=
  { a := p.a, h := p.h - dx, k := p.k }

/-- Shifts a parabola vertically -/
def shift_vertical (p : Parabola) (dy : ℝ) : Parabola :=
  { a := p.a, h := p.h, k := p.k + dy }

/-- The initial parabola y = -(x+1)^2 + 2 -/
def initial_parabola : Parabola :=
  { a := -1, h := 1, k := 2 }

/-- The final parabola y = -(x+2)^2 -/
def final_parabola : Parabola :=
  { a := -1, h := 2, k := 0 }

theorem parabola_transformation :
  (shift_vertical (shift_horizontal initial_parabola 1) (-2)) = final_parabola := by
  sorry

end NUMINAMATH_CALUDE_parabola_transformation_l2957_295725


namespace NUMINAMATH_CALUDE_sock_pairs_calculation_l2957_295729

def calculate_sock_pairs (initial_socks thrown_away_socks new_socks : ℕ) : ℕ :=
  ((initial_socks - thrown_away_socks) + new_socks) / 2

theorem sock_pairs_calculation (initial_socks thrown_away_socks new_socks : ℕ) 
  (h1 : initial_socks ≥ thrown_away_socks) :
  calculate_sock_pairs initial_socks thrown_away_socks new_socks = 
  ((initial_socks - thrown_away_socks) + new_socks) / 2 := by
  sorry

#eval calculate_sock_pairs 28 4 36

end NUMINAMATH_CALUDE_sock_pairs_calculation_l2957_295729


namespace NUMINAMATH_CALUDE_waiter_tables_l2957_295703

/-- Proves that a waiter with 40 customers and tables of 5 women and 3 men each has 5 tables. -/
theorem waiter_tables (total_customers : ℕ) (women_per_table : ℕ) (men_per_table : ℕ) :
  total_customers = 40 →
  women_per_table = 5 →
  men_per_table = 3 →
  total_customers = (women_per_table + men_per_table) * 5 :=
by sorry

end NUMINAMATH_CALUDE_waiter_tables_l2957_295703


namespace NUMINAMATH_CALUDE_book_cost_l2957_295788

theorem book_cost (book bookmark : ℝ) 
  (total_cost : book + bookmark = 2.10)
  (price_difference : book = bookmark + 2) :
  book = 2.05 := by
sorry

end NUMINAMATH_CALUDE_book_cost_l2957_295788


namespace NUMINAMATH_CALUDE_at_most_one_greater_than_one_l2957_295713

theorem at_most_one_greater_than_one (x y : ℝ) (h : x + y < 2) :
  ¬(x > 1 ∧ y > 1) := by
  sorry

end NUMINAMATH_CALUDE_at_most_one_greater_than_one_l2957_295713


namespace NUMINAMATH_CALUDE_female_officers_count_l2957_295756

theorem female_officers_count (total_on_duty : ℕ) (female_ratio_on_duty : ℚ) (female_percentage : ℚ) :
  total_on_duty = 100 →
  female_ratio_on_duty = 1/2 →
  female_percentage = 1/5 →
  (female_ratio_on_duty * total_on_duty : ℚ) / female_percentage = 250 := by
  sorry

end NUMINAMATH_CALUDE_female_officers_count_l2957_295756


namespace NUMINAMATH_CALUDE_oliver_bath_water_usage_l2957_295786

/-- Calculates the weekly water usage for baths given the bucket capacity, 
    number of buckets to fill the tub, number of buckets removed, and days per week -/
def weekly_water_usage (bucket_capacity : ℕ) (buckets_to_fill : ℕ) (buckets_removed : ℕ) (days_per_week : ℕ) : ℕ :=
  (buckets_to_fill - buckets_removed) * bucket_capacity * days_per_week

/-- Theorem stating that given the specific conditions, the weekly water usage is 9240 ounces -/
theorem oliver_bath_water_usage :
  weekly_water_usage 120 14 3 7 = 9240 := by
  sorry

end NUMINAMATH_CALUDE_oliver_bath_water_usage_l2957_295786


namespace NUMINAMATH_CALUDE_total_money_is_140_l2957_295782

/-- Calculates the total money collected from football game tickets -/
def total_money_collected (adult_price child_price : ℚ) (total_attendees adult_attendees : ℕ) : ℚ :=
  adult_price * adult_attendees + child_price * (total_attendees - adult_attendees)

/-- Theorem stating that the total money collected is $140 -/
theorem total_money_is_140 :
  let adult_price : ℚ := 60 / 100
  let child_price : ℚ := 25 / 100
  let total_attendees : ℕ := 280
  let adult_attendees : ℕ := 200
  total_money_collected adult_price child_price total_attendees adult_attendees = 140 / 1 := by
  sorry

end NUMINAMATH_CALUDE_total_money_is_140_l2957_295782


namespace NUMINAMATH_CALUDE_cost_price_calculation_l2957_295752

theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) 
  (h1 : selling_price = 600)
  (h2 : profit_percentage = 25) :
  let cost_price := selling_price / (1 + profit_percentage / 100)
  cost_price = 480 := by
sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l2957_295752


namespace NUMINAMATH_CALUDE_jerrys_breakfast_calories_l2957_295730

/-- Given Jerry's breakfast composition and total calories, prove the calories per pancake. -/
theorem jerrys_breakfast_calories (pancakes : ℕ) (bacon_strips : ℕ) (bacon_calories : ℕ) 
  (cereal_calories : ℕ) (total_calories : ℕ) (calories_per_pancake : ℕ) :
  pancakes = 6 →
  bacon_strips = 2 →
  bacon_calories = 100 →
  cereal_calories = 200 →
  total_calories = 1120 →
  total_calories = pancakes * calories_per_pancake + bacon_strips * bacon_calories + cereal_calories →
  calories_per_pancake = 120 := by
  sorry

end NUMINAMATH_CALUDE_jerrys_breakfast_calories_l2957_295730


namespace NUMINAMATH_CALUDE_fraction_product_l2957_295702

theorem fraction_product : (2 : ℚ) / 9 * (5 : ℚ) / 11 = 10 / 99 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_l2957_295702


namespace NUMINAMATH_CALUDE_greg_trousers_bought_l2957_295755

/-- The cost of a shirt -/
def shirt_cost : ℝ := sorry

/-- The cost of a trouser -/
def trouser_cost : ℝ := sorry

/-- The cost of a tie -/
def tie_cost : ℝ := sorry

/-- The number of trousers bought in the first scenario -/
def trousers_bought : ℕ := sorry

theorem greg_trousers_bought : 
  (3 * shirt_cost + trousers_bought * trouser_cost + 2 * tie_cost = 90) ∧
  (7 * shirt_cost + 2 * trouser_cost + 2 * tie_cost = 50) ∧
  (5 * shirt_cost + 3 * trouser_cost + 2 * tie_cost = 70) →
  trousers_bought = 4 := by
sorry

end NUMINAMATH_CALUDE_greg_trousers_bought_l2957_295755


namespace NUMINAMATH_CALUDE_solve_amusement_park_problem_l2957_295798

def amusement_park_problem (adult_price child_price total_tickets child_tickets : ℕ) : Prop :=
  adult_price = 8 ∧
  child_price = 5 ∧
  total_tickets = 33 ∧
  child_tickets = 21 ∧
  (total_tickets - child_tickets) * adult_price + child_tickets * child_price = 201

theorem solve_amusement_park_problem :
  ∃ (adult_price child_price total_tickets child_tickets : ℕ),
    amusement_park_problem adult_price child_price total_tickets child_tickets :=
by
  sorry

end NUMINAMATH_CALUDE_solve_amusement_park_problem_l2957_295798


namespace NUMINAMATH_CALUDE_first_cyclist_overtakes_second_opposite_P_l2957_295720

/-- Represents the circular runway --/
structure CircularRunway where
  radius : ℝ

/-- Represents a moving entity on the circular runway --/
structure MovingEntity where
  velocity : ℝ

/-- Represents the scenario of cyclists and pedestrian on the circular runway --/
structure RunwayScenario where
  runway : CircularRunway
  cyclist1 : MovingEntity
  cyclist2 : MovingEntity
  pedestrian : MovingEntity

/-- The main theorem stating the point where the first cyclist overtakes the second --/
theorem first_cyclist_overtakes_second_opposite_P (scenario : RunwayScenario) 
  (h1 : scenario.cyclist1.velocity > scenario.cyclist2.velocity)
  (h2 : scenario.pedestrian.velocity = (scenario.cyclist1.velocity + scenario.cyclist2.velocity) / 12)
  (h3 : ∃ t1 t2, t2 - t1 = 91 ∧ 
        t1 = (2 * π * scenario.runway.radius) / (scenario.cyclist1.velocity + scenario.pedestrian.velocity) ∧
        t2 = (2 * π * scenario.runway.radius) / (scenario.cyclist2.velocity + scenario.pedestrian.velocity))
  (h4 : ∃ t3 t4, t4 - t3 = 187 ∧
        t3 = (2 * π * scenario.runway.radius) / (scenario.cyclist1.velocity - scenario.pedestrian.velocity) ∧
        t4 = (2 * π * scenario.runway.radius) / (scenario.cyclist2.velocity - scenario.pedestrian.velocity)) :
  ∃ t : ℝ, t * scenario.cyclist1.velocity = π * scenario.runway.radius ∧
          t * scenario.cyclist2.velocity = π * scenario.runway.radius :=
by sorry

end NUMINAMATH_CALUDE_first_cyclist_overtakes_second_opposite_P_l2957_295720


namespace NUMINAMATH_CALUDE_caitlin_sara_weight_l2957_295780

/-- Given the weights of three people (Annette, Caitlin, and Sara), proves that
    Caitlin and Sara weigh 87 pounds together. -/
theorem caitlin_sara_weight 
  (annette caitlin sara : ℝ) 
  (h1 : annette + caitlin = 95)   -- Annette and Caitlin weigh 95 pounds together
  (h2 : annette = sara + 8) :     -- Annette weighs 8 pounds more than Sara
  caitlin + sara = 87 := by sorry

end NUMINAMATH_CALUDE_caitlin_sara_weight_l2957_295780


namespace NUMINAMATH_CALUDE_katie_miles_run_l2957_295761

/-- Given that Adam ran 125 miles and Adam ran 80 miles more than Katie, prove that Katie ran 45 miles. -/
theorem katie_miles_run (adam_miles : ℕ) (difference : ℕ) (katie_miles : ℕ) 
  (h1 : adam_miles = 125)
  (h2 : adam_miles = katie_miles + difference)
  (h3 : difference = 80) : 
  katie_miles = 45 := by
sorry

end NUMINAMATH_CALUDE_katie_miles_run_l2957_295761


namespace NUMINAMATH_CALUDE_gcd_g_x_l2957_295769

def g (x : ℤ) : ℤ := (5*x+3)*(8*x+2)*(12*x+7)*(3*x+10)

theorem gcd_g_x (x : ℤ) (h : 46800 ∣ x) : 
  Nat.gcd (Int.natAbs (g x)) (Int.natAbs x) = 60 := by
  sorry

end NUMINAMATH_CALUDE_gcd_g_x_l2957_295769


namespace NUMINAMATH_CALUDE_nested_sqrt_value_l2957_295728

theorem nested_sqrt_value :
  ∃ y : ℝ, y = Real.sqrt (3 + y) → y = (1 + Real.sqrt 13) / 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_sqrt_value_l2957_295728


namespace NUMINAMATH_CALUDE_cube_sum_geq_product_sum_l2957_295741

theorem cube_sum_geq_product_sum {a b : ℝ} (ha : a > 0) (hb : b > 0) :
  a^3 + b^3 ≥ a^2*b + a*b^2 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_geq_product_sum_l2957_295741


namespace NUMINAMATH_CALUDE_circle_properties_l2957_295710

/-- Given a circle with area 16π, prove its diameter is 8 and circumference is 8π -/
theorem circle_properties (r : ℝ) (h : π * r^2 = 16 * π) :
  2 * r = 8 ∧ 2 * π * r = 8 * π := by
  sorry


end NUMINAMATH_CALUDE_circle_properties_l2957_295710


namespace NUMINAMATH_CALUDE_max_diagonals_regular_1000_gon_l2957_295773

/-- The number of sides in the regular polygon -/
def n : ℕ := 1000

/-- The number of different diagonal lengths in a regular n-gon -/
def num_diagonal_lengths (n : ℕ) : ℕ := n / 2

/-- The total number of diagonals in a regular n-gon -/
def total_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The number of diagonals of each length in a regular n-gon -/
def diagonals_per_length (n : ℕ) : ℕ := n

/-- The maximum number of diagonals that can be selected such that among any three of the chosen diagonals, at least two have the same length -/
def max_selected_diagonals (n : ℕ) : ℕ := 2 * diagonals_per_length n

theorem max_diagonals_regular_1000_gon :
  max_selected_diagonals n = 2000 :=
sorry

end NUMINAMATH_CALUDE_max_diagonals_regular_1000_gon_l2957_295773


namespace NUMINAMATH_CALUDE_coin_sequences_ten_l2957_295748

/-- The number of distinct sequences when flipping a coin n times -/
def coin_sequences (n : ℕ) : ℕ := 2^n

/-- Theorem: The number of distinct sequences when flipping a coin 10 times is 1024 -/
theorem coin_sequences_ten : coin_sequences 10 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_coin_sequences_ten_l2957_295748


namespace NUMINAMATH_CALUDE_triangle_area_l2957_295768

/-- The area of a triangle with side lengths 10, 10, and 12 is 48 -/
theorem triangle_area (a b c : ℝ) (h1 : a = 10) (h2 : b = 10) (h3 : c = 12) :
  (1 / 2 : ℝ) * c * Real.sqrt (4 * a^2 * b^2 - (a^2 + b^2 - c^2)^2) / (2 * a * b) = 48 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2957_295768


namespace NUMINAMATH_CALUDE_power_of_product_square_l2957_295745

theorem power_of_product_square (a b : ℝ) : (a * b^2)^2 = a^2 * b^4 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_square_l2957_295745


namespace NUMINAMATH_CALUDE_dark_tiles_fraction_l2957_295705

/-- Represents a 4x4 block of tiles on the floor -/
structure Block where
  size : Nat
  dark_tiles : Nat

/-- Represents the entire tiled floor -/
structure Floor where
  block : Block

/-- The fraction of dark tiles in the floor -/
def dark_fraction (f : Floor) : Rat :=
  f.block.dark_tiles / (f.block.size * f.block.size)

theorem dark_tiles_fraction (f : Floor) 
  (h1 : f.block.size = 4)
  (h2 : f.block.dark_tiles = 12) : 
  dark_fraction f = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_dark_tiles_fraction_l2957_295705


namespace NUMINAMATH_CALUDE_three_fractions_inequality_l2957_295749

theorem three_fractions_inequality (a b c : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
  (sum_one : a + b + c = 1) :
  (a - b*c) / (a + b*c) + (b - c*a) / (b + c*a) + (c - a*b) / (c + a*b) ≤ 3/2 := by
sorry

end NUMINAMATH_CALUDE_three_fractions_inequality_l2957_295749


namespace NUMINAMATH_CALUDE_chennys_friends_l2957_295762

theorem chennys_friends (initial_candies : ℕ) (additional_candies : ℕ) (candies_per_friend : ℕ) :
  initial_candies = 10 →
  additional_candies = 4 →
  candies_per_friend = 2 →
  (initial_candies + additional_candies) / candies_per_friend = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_chennys_friends_l2957_295762


namespace NUMINAMATH_CALUDE_ways_to_put_five_balls_three_boxes_l2957_295783

/-- The number of ways to put n distinguishable balls into k distinguishable boxes -/
def ways_to_put_balls (n : ℕ) (k : ℕ) : ℕ := k^n

/-- Theorem: There are 243 ways to put 5 distinguishable balls into 3 distinguishable boxes -/
theorem ways_to_put_five_balls_three_boxes : ways_to_put_balls 5 3 = 243 := by
  sorry

end NUMINAMATH_CALUDE_ways_to_put_five_balls_three_boxes_l2957_295783


namespace NUMINAMATH_CALUDE_solve_linear_equation_l2957_295796

theorem solve_linear_equation (x : ℝ) :
  3 * x - 7 = 11 → x = 6 := by
sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l2957_295796


namespace NUMINAMATH_CALUDE_expected_rolls_2010_l2957_295759

/-- The expected number of rolls for a fair six-sided die to reach or exceed a given sum -/
noncomputable def expected_rolls (n : ℕ) : ℝ :=
  if n = 0 then 0
  else (1 / 6) * (expected_rolls (n - 1) + expected_rolls (n - 2) + expected_rolls (n - 3) +
                  expected_rolls (n - 4) + expected_rolls (n - 5) + expected_rolls (n - 6)) + 1

/-- Theorem stating the expected number of rolls to reach or exceed 2010 -/
theorem expected_rolls_2010 :
  ∃ ε > 0, |expected_rolls 2010 - 574.761904| < ε :=
sorry

end NUMINAMATH_CALUDE_expected_rolls_2010_l2957_295759


namespace NUMINAMATH_CALUDE_correct_num_non_officers_l2957_295737

/-- Represents the number of non-officers in an office -/
def num_non_officers : ℕ := 495

/-- Represents the number of officers in an office -/
def num_officers : ℕ := 15

/-- Represents the average salary of all employees in Rs/month -/
def avg_salary_all : ℚ := 120

/-- Represents the average salary of officers in Rs/month -/
def avg_salary_officers : ℚ := 450

/-- Represents the average salary of non-officers in Rs/month -/
def avg_salary_non_officers : ℚ := 110

/-- Theorem stating that the number of non-officers is correct given the conditions -/
theorem correct_num_non_officers :
  (num_officers * avg_salary_officers + num_non_officers * avg_salary_non_officers) / 
  (num_officers + num_non_officers : ℚ) = avg_salary_all := by
  sorry


end NUMINAMATH_CALUDE_correct_num_non_officers_l2957_295737


namespace NUMINAMATH_CALUDE_consecutive_sequence_unique_l2957_295744

/-- Three consecutive natural numbers forming an arithmetic and geometric sequence -/
def ConsecutiveSequence (a b c : ℕ) : Prop :=
  (b = a + 1) ∧ (c = b + 1) ∧
  (b + 2)^2 = (a + 1) * (c + 5)

theorem consecutive_sequence_unique :
  ∀ a b c : ℕ, ConsecutiveSequence a b c → a = 1 ∧ b = 2 ∧ c = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_consecutive_sequence_unique_l2957_295744


namespace NUMINAMATH_CALUDE_no_given_factors_of_polynomial_l2957_295722

theorem no_given_factors_of_polynomial :
  let p (x : ℝ) := x^4 - 2*x^2 + 9
  let factors := [
    (fun x => x^2 + 3),
    (fun x => x + 1),
    (fun x => x^2 - 3),
    (fun x => x^2 + 2*x - 3)
  ]
  ∀ f ∈ factors, ¬ (∃ q : ℝ → ℝ, ∀ x, p x = f x * q x) :=
by sorry

end NUMINAMATH_CALUDE_no_given_factors_of_polynomial_l2957_295722


namespace NUMINAMATH_CALUDE_bee_colony_reduction_l2957_295739

/-- Prove that a bee colony with given initial size and daily loss rate
    reaches 1/4 of its initial size in the calculated number of days. -/
theorem bee_colony_reduction (initial_size : ℕ) (daily_loss : ℕ) (days : ℕ) :
  initial_size = 80000 →
  daily_loss = 1200 →
  days = 50 →
  initial_size - days * daily_loss = initial_size / 4 :=
by sorry

end NUMINAMATH_CALUDE_bee_colony_reduction_l2957_295739


namespace NUMINAMATH_CALUDE_diophantine_equation_min_max_sum_l2957_295767

theorem diophantine_equation_min_max_sum : 
  ∃ (p q : ℕ), 
    (∀ (x y : ℕ), x > 0 ∧ y > 0 ∧ 6 * x + 7 * y = 2012 → x + y ≥ p) ∧
    (∀ (x y : ℕ), x > 0 ∧ y > 0 ∧ 6 * x + 7 * y = 2012 → x + y ≤ q) ∧
    (∃ (x₁ y₁ x₂ y₂ : ℕ), 
      x₁ > 0 ∧ y₁ > 0 ∧ 6 * x₁ + 7 * y₁ = 2012 ∧ x₁ + y₁ = p ∧
      x₂ > 0 ∧ y₂ > 0 ∧ 6 * x₂ + 7 * y₂ = 2012 ∧ x₂ + y₂ = q) ∧
    p + q = 623 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_min_max_sum_l2957_295767


namespace NUMINAMATH_CALUDE_digit_distribution_l2957_295712

theorem digit_distribution (n : ℕ) 
  (h1 : n > 0)
  (h2 : (n / 2 : ℚ) = n * (1 / 2 : ℚ))
  (h3 : (n / 5 : ℚ) = n * (1 / 5 : ℚ))
  (h4 : (n / 10 : ℚ) = n * (1 / 10 : ℚ))
  (h5 : (1 / 2 : ℚ) + 2 * (1 / 5 : ℚ) + (1 / 10 : ℚ) = 1) :
  n = 10 := by
sorry

end NUMINAMATH_CALUDE_digit_distribution_l2957_295712


namespace NUMINAMATH_CALUDE_gold_percentage_in_first_metal_l2957_295774

theorem gold_percentage_in_first_metal
  (total_weight : Real)
  (desired_gold_percentage : Real)
  (first_metal_weight : Real)
  (second_metal_weight : Real)
  (second_metal_gold_percentage : Real)
  (h1 : total_weight = 12.4)
  (h2 : desired_gold_percentage = 0.5)
  (h3 : first_metal_weight = 6.2)
  (h4 : second_metal_weight = 6.2)
  (h5 : second_metal_gold_percentage = 0.4)
  (h6 : total_weight = first_metal_weight + second_metal_weight) :
  let total_gold := total_weight * desired_gold_percentage
  let second_metal_gold := second_metal_weight * second_metal_gold_percentage
  let first_metal_gold := total_gold - second_metal_gold
  let first_metal_gold_percentage := first_metal_gold / first_metal_weight
  first_metal_gold_percentage = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_gold_percentage_in_first_metal_l2957_295774


namespace NUMINAMATH_CALUDE_photo_arrangements_l2957_295707

/-- Represents the number of people in the photo arrangement --/
def total_people : ℕ := 7

/-- Represents the number of students in the photo arrangement --/
def num_students : ℕ := 6

/-- Represents the position of the teacher in the row --/
def teacher_position : ℕ := 4

/-- Represents the number of positions to the left of the teacher --/
def left_positions : ℕ := 3

/-- Represents the number of positions to the right of the teacher --/
def right_positions : ℕ := 3

/-- Represents the number of positions available for Student A --/
def positions_for_A : ℕ := 5

/-- Represents the number of positions available for Student B --/
def positions_for_B : ℕ := 5

/-- Represents the number of remaining students after placing A and B --/
def remaining_students : ℕ := 4

/-- Theorem stating the number of different arrangements --/
theorem photo_arrangements :
  (positions_for_A * (positions_for_B - 1) * (remaining_students!)) * 2 = 960 := by
  sorry

end NUMINAMATH_CALUDE_photo_arrangements_l2957_295707


namespace NUMINAMATH_CALUDE_sum_of_digits_of_9N_l2957_295765

/-- A function that checks if each digit of a natural number is strictly greater than the digit to its left -/
def is_strictly_increasing_digits (n : ℕ) : Prop :=
  ∀ i j, i < j → (n.digits 10).get i < (n.digits 10).get j

/-- A function that calculates the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

/-- Theorem: For any natural number N where each digit is strictly greater than the digit to its left,
    the sum of the digits of 9N is equal to 9 -/
theorem sum_of_digits_of_9N (N : ℕ) (h : is_strictly_increasing_digits N) :
  sum_of_digits (9 * N) = 9 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_9N_l2957_295765


namespace NUMINAMATH_CALUDE_discounted_price_calculation_l2957_295771

/-- The actual price of the good in Rupees -/
def actual_price : ℝ := 9502.923976608186

/-- The first discount rate -/
def discount1 : ℝ := 0.20

/-- The second discount rate -/
def discount2 : ℝ := 0.10

/-- The third discount rate -/
def discount3 : ℝ := 0.05

/-- The discounted price after applying three successive discounts -/
def discounted_price (p : ℝ) (d1 d2 d3 : ℝ) : ℝ :=
  p * (1 - d1) * (1 - d2) * (1 - d3)

/-- Theorem stating that the discounted price is approximately 6498.40 -/
theorem discounted_price_calculation :
  ∃ ε > 0, abs (discounted_price actual_price discount1 discount2 discount3 - 6498.40) < ε :=
sorry

end NUMINAMATH_CALUDE_discounted_price_calculation_l2957_295771


namespace NUMINAMATH_CALUDE_months_with_average_salary_8900_l2957_295779

def average_salary_jan_to_apr : ℕ := 8000
def average_salary_some_months : ℕ := 8900
def salary_may : ℕ := 6500
def salary_jan : ℕ := 2900

theorem months_with_average_salary_8900 :
  let total_salary_jan_to_apr := average_salary_jan_to_apr * 4
  let total_salary_feb_to_apr := total_salary_jan_to_apr - salary_jan
  let total_salary_feb_to_may := total_salary_feb_to_apr + salary_may
  total_salary_feb_to_may / average_salary_some_months = 4 := by
sorry

end NUMINAMATH_CALUDE_months_with_average_salary_8900_l2957_295779


namespace NUMINAMATH_CALUDE_robin_water_bottles_l2957_295799

/-- The number of additional bottles needed on the last day -/
def additional_bottles (total_bottles : ℕ) (daily_consumption : ℕ) : ℕ :=
  daily_consumption - (total_bottles % daily_consumption)

/-- Theorem stating that given 617 bottles and a daily consumption of 9 bottles, 
    4 additional bottles are needed on the last day -/
theorem robin_water_bottles : additional_bottles 617 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_robin_water_bottles_l2957_295799


namespace NUMINAMATH_CALUDE_lawn_mowing_problem_l2957_295747

theorem lawn_mowing_problem (initial_people : ℕ) (initial_hours : ℕ) (target_hours : ℕ) :
  initial_people = 8 →
  initial_hours = 5 →
  target_hours = 3 →
  ∃ (additional_people : ℕ),
    additional_people = 6 ∧
    (initial_people + additional_people) * target_hours = initial_people * initial_hours :=
by sorry

end NUMINAMATH_CALUDE_lawn_mowing_problem_l2957_295747


namespace NUMINAMATH_CALUDE_inscribed_square_side_length_l2957_295708

/-- Given a right triangle with legs of lengths 6 and 8, and an inscribed square
    sharing the right angle with the triangle, the side length of the square is 24/7. -/
theorem inscribed_square_side_length :
  ∀ (a b c : Real) (s : Real),
    a = 6 →
    b = 8 →
    c^2 = a^2 + b^2 →  -- Pythagorean theorem for right triangle
    s^2 + (a - s) * s + (b - s) * s = (a * b) / 2 →  -- Area equality
    s = 24 / 7 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_square_side_length_l2957_295708


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l2957_295743

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 5/8) (h2 : x - y = 3/8) : x^2 - y^2 = 15/64 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l2957_295743


namespace NUMINAMATH_CALUDE_solutions_to_equation_l2957_295704

theorem solutions_to_equation : 
  ∃ (s : Set ℝ), s = {x : ℝ | x^4 + (2 - x)^4 = 34} ∧ s = {1 + Real.sqrt 2, 1 - Real.sqrt 2} := by
sorry

end NUMINAMATH_CALUDE_solutions_to_equation_l2957_295704


namespace NUMINAMATH_CALUDE_smallest_angle_quadrilateral_l2957_295716

theorem smallest_angle_quadrilateral (a b c d : ℝ) : 
  a > 0 → b > 0 → c > 0 → d > 0 →
  (a + b + c + d = 360) →
  (b = 5/4 * a) → (c = 3/2 * a) → (d = 7/4 * a) →
  a = 720 / 11 := by
sorry

end NUMINAMATH_CALUDE_smallest_angle_quadrilateral_l2957_295716


namespace NUMINAMATH_CALUDE_gum_distribution_l2957_295750

theorem gum_distribution (num_cousins : Nat) (gum_per_cousin : Nat) : 
  num_cousins = 4 → gum_per_cousin = 5 → num_cousins * gum_per_cousin = 20 := by
  sorry

end NUMINAMATH_CALUDE_gum_distribution_l2957_295750


namespace NUMINAMATH_CALUDE_folded_square_FG_length_l2957_295731

/-- A folded square sheet of paper with side length 1 -/
structure FoldedSquare where
  /-- The point where corners B and D meet after folding -/
  E : ℝ × ℝ
  /-- The point F on side AB -/
  F : ℝ × ℝ
  /-- The point G on side AD -/
  G : ℝ × ℝ
  /-- E lies on the diagonal AC -/
  E_on_diagonal : E.1 = E.2
  /-- F is on side AB -/
  F_on_AB : F.2 = 0 ∧ 0 ≤ F.1 ∧ F.1 ≤ 1
  /-- G is on side AD -/
  G_on_AD : G.1 = 0 ∧ 0 ≤ G.2 ∧ G.2 ≤ 1

/-- The theorem stating that the length of FG in a folded unit square is 2√2 - 2 -/
theorem folded_square_FG_length (s : FoldedSquare) : 
  Real.sqrt ((s.F.1 - s.G.1)^2 + (s.F.2 - s.G.2)^2) = 2 * Real.sqrt 2 - 2 := by
  sorry

end NUMINAMATH_CALUDE_folded_square_FG_length_l2957_295731


namespace NUMINAMATH_CALUDE_work_completion_time_l2957_295787

theorem work_completion_time (work : ℝ) (a b : ℝ) 
  (h1 : a + b = work / 6)  -- A and B together complete work in 6 days
  (h2 : a = work / 14)     -- A alone completes work in 14 days
  : b = work / 10.5        -- B alone completes work in 10.5 days
:= by sorry

end NUMINAMATH_CALUDE_work_completion_time_l2957_295787


namespace NUMINAMATH_CALUDE_sum_not_divisible_l2957_295797

theorem sum_not_divisible : ∃ (y : ℤ), 
  y = 42 + 98 + 210 + 333 + 175 + 28 ∧ 
  ¬(∃ (k : ℤ), y = 7 * k) ∧ 
  ¬(∃ (k : ℤ), y = 14 * k) ∧ 
  ¬(∃ (k : ℤ), y = 28 * k) ∧ 
  ¬(∃ (k : ℤ), y = 21 * k) := by
  sorry

end NUMINAMATH_CALUDE_sum_not_divisible_l2957_295797


namespace NUMINAMATH_CALUDE_problem_2005_squared_minus_2003_times_2007_l2957_295794

theorem problem_2005_squared_minus_2003_times_2007 : 2005^2 - 2003 * 2007 = 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_2005_squared_minus_2003_times_2007_l2957_295794


namespace NUMINAMATH_CALUDE_cubic_root_equation_solutions_l2957_295719

theorem cubic_root_equation_solutions :
  let f : ℝ → ℝ := λ x => (18 * x - 2)^(1/3) + (16 * x + 2)^(1/3) + (-72 * x)^(1/3) - 6 * x^(1/3)
  {x : ℝ | f x = 0} = {0, 1/9, -1/8} := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_equation_solutions_l2957_295719


namespace NUMINAMATH_CALUDE_simplify_expression_l2957_295746

theorem simplify_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^3 + y^3 = 3*(x + y)) : x/y + y/x - 3/(x*y) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2957_295746


namespace NUMINAMATH_CALUDE_ginger_wears_size_8_l2957_295715

def anna_size : ℕ := 2

def becky_size (anna : ℕ) : ℕ := 3 * anna

def ginger_size (becky : ℕ) : ℕ := 2 * becky - 4

theorem ginger_wears_size_8 : 
  ginger_size (becky_size anna_size) = 8 := by sorry

end NUMINAMATH_CALUDE_ginger_wears_size_8_l2957_295715


namespace NUMINAMATH_CALUDE_handmade_ornaments_fraction_l2957_295735

theorem handmade_ornaments_fraction (total : ℕ) (handmade_fraction : ℚ) :
  total = 20 →
  (1 : ℚ) / 3 * total = (1 : ℚ) / 2 * (handmade_fraction * total) →
  handmade_fraction = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_handmade_ornaments_fraction_l2957_295735


namespace NUMINAMATH_CALUDE_waiter_tips_l2957_295760

/-- Calculates the total tips earned by a waiter --/
def total_tips (total_customers : ℕ) (non_tipping_customers : ℕ) (tip_amount : ℕ) : ℕ :=
  (total_customers - non_tipping_customers) * tip_amount

/-- Theorem stating the total tips earned by the waiter --/
theorem waiter_tips : total_tips 7 5 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_waiter_tips_l2957_295760


namespace NUMINAMATH_CALUDE_parallel_iff_a_eq_one_perpendicular_iff_a_eq_zero_l2957_295758

-- Define the lines
def line1 (a : ℝ) (x y : ℝ) : Prop := x + a * y - 2 * a - 2 = 0
def line2 (a : ℝ) (x y : ℝ) : Prop := a * x + y - 1 - a = 0

-- Define parallel and perpendicular conditions
def parallel (a : ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ ∀ x y, line1 a x y ↔ line2 a (k * x) (k * y)
def perpendicular (a : ℝ) : Prop := ∃ x1 y1 x2 y2 : ℝ, 
  line1 a x1 y1 ∧ line2 a x2 y2 ∧ (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) ≠ 0 ∧
  (x2 - x1) * (y2 - y1) = 0

-- State the theorems
theorem parallel_iff_a_eq_one : ∀ a : ℝ, parallel a ↔ a = 1 := by sorry

theorem perpendicular_iff_a_eq_zero : ∀ a : ℝ, perpendicular a ↔ a = 0 := by sorry

end NUMINAMATH_CALUDE_parallel_iff_a_eq_one_perpendicular_iff_a_eq_zero_l2957_295758


namespace NUMINAMATH_CALUDE_prob_different_colors_value_l2957_295740

def blue_chips : ℕ := 6
def red_chips : ℕ := 5
def yellow_chips : ℕ := 4
def green_chips : ℕ := 3

def total_chips : ℕ := blue_chips + red_chips + yellow_chips + green_chips

def prob_different_colors : ℚ :=
  let prob_blue : ℚ := blue_chips / total_chips
  let prob_red : ℚ := red_chips / total_chips
  let prob_yellow : ℚ := yellow_chips / total_chips
  let prob_green : ℚ := green_chips / total_chips
  
  let prob_not_blue : ℚ := 1 - prob_blue
  let prob_not_red : ℚ := 1 - prob_red
  let prob_not_yellow : ℚ := 1 - prob_yellow
  let prob_not_green : ℚ := 1 - prob_green
  
  prob_blue * prob_not_blue +
  prob_red * prob_not_red +
  prob_yellow * prob_not_yellow +
  prob_green * prob_not_green

theorem prob_different_colors_value :
  prob_different_colors = 119 / 162 :=
sorry

end NUMINAMATH_CALUDE_prob_different_colors_value_l2957_295740


namespace NUMINAMATH_CALUDE_correct_card_order_l2957_295791

/-- Represents a card in the arrangement --/
inductive Card : Type
  | A | B | C | D | E | F

/-- Represents the relative position of two cards --/
inductive Position
  | Above
  | Below
  | SameLevel

/-- Determines the relative position of two cards based on their overlaps --/
def relative_position (c1 c2 : Card) : Position := sorry

/-- Represents the final ordering of cards --/
def card_order : List Card := sorry

/-- Theorem stating the correct order of cards --/
theorem correct_card_order :
  card_order = [Card.F, Card.E, Card.A, Card.D, Card.C, Card.B] ∧
  relative_position Card.E Card.A = Position.SameLevel ∧
  relative_position Card.A Card.E = Position.SameLevel :=
sorry

end NUMINAMATH_CALUDE_correct_card_order_l2957_295791


namespace NUMINAMATH_CALUDE_log_identity_l2957_295742

theorem log_identity (a : ℝ) (h : a = Real.log 3 / Real.log 4) : 
  2^a + 2^(-a) = 4 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_log_identity_l2957_295742


namespace NUMINAMATH_CALUDE_village_population_equality_l2957_295764

/-- The initial population of Village X -/
def initial_population_X : ℕ := 72000

/-- The yearly decrease in population of Village X -/
def decrease_rate_X : ℕ := 1200

/-- The initial population of Village Y -/
def initial_population_Y : ℕ := 42000

/-- The yearly increase in population of Village Y -/
def increase_rate_Y : ℕ := 800

/-- The number of years after which the populations are equal -/
def years : ℕ := 15

theorem village_population_equality :
  initial_population_X - (decrease_rate_X * years) =
  initial_population_Y + (increase_rate_Y * years) :=
by sorry

end NUMINAMATH_CALUDE_village_population_equality_l2957_295764


namespace NUMINAMATH_CALUDE_velocity_center_of_mass_before_collision_l2957_295793

/-- Velocity of the center of mass of a two-cart system before collision -/
theorem velocity_center_of_mass_before_collision 
  (m : ℝ) -- mass of cart 1
  (v1_initial : ℝ) -- initial velocity of cart 1
  (m2 : ℝ) -- mass of cart 2
  (v2_initial : ℝ) -- initial velocity of cart 2
  (v1_final : ℝ) -- final velocity of cart 1
  (h1 : v1_initial = 12) -- initial velocity of cart 1 is 12 m/s
  (h2 : m2 = 4) -- mass of cart 2 is 4 kg
  (h3 : v2_initial = 0) -- cart 2 is initially at rest
  (h4 : v1_final = -6) -- final velocity of cart 1 is 6 m/s to the left
  (h5 : m > 0) -- mass of cart 1 is positive
  (h6 : m2 > 0) -- mass of cart 2 is positive
  : (m * v1_initial + m2 * v2_initial) / (m + m2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_velocity_center_of_mass_before_collision_l2957_295793


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2957_295792

theorem quadratic_inequality_solution (m : ℝ) :
  (∀ x, mx^2 + 8*m*x + 28 < 0 ↔ -7 < x ∧ x < -1) →
  m = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2957_295792


namespace NUMINAMATH_CALUDE_socks_difference_l2957_295757

/-- The number of pairs of socks Laticia knitted in the first week -/
def first_week : ℕ := 12

/-- The number of pairs of socks Laticia knitted in the second week -/
def second_week : ℕ := sorry

/-- The number of pairs of socks Laticia knitted in the third week -/
def third_week : ℕ := (first_week + second_week) / 2

/-- The number of pairs of socks Laticia knitted in the fourth week -/
def fourth_week : ℕ := third_week - 3

/-- The total number of pairs of socks Laticia knitted -/
def total_socks : ℕ := 57

theorem socks_difference : 
  first_week + second_week + third_week + fourth_week = total_socks ∧ 
  second_week - first_week = 1 := by sorry

end NUMINAMATH_CALUDE_socks_difference_l2957_295757


namespace NUMINAMATH_CALUDE_students_liking_sports_l2957_295784

theorem students_liking_sports (basketball : Finset ℕ) (cricket : Finset ℕ) 
  (h1 : basketball.card = 7)
  (h2 : cricket.card = 5)
  (h3 : (basketball ∩ cricket).card = 3) :
  (basketball ∪ cricket).card = 9 := by
  sorry

end NUMINAMATH_CALUDE_students_liking_sports_l2957_295784


namespace NUMINAMATH_CALUDE_margin_in_terms_of_selling_price_l2957_295723

theorem margin_in_terms_of_selling_price
  (C S M n : ℝ)
  (h1 : M = (2 / n) * C)
  (h2 : S - M = C)
  : M = 2 * S / (n + 2) := by
sorry

end NUMINAMATH_CALUDE_margin_in_terms_of_selling_price_l2957_295723


namespace NUMINAMATH_CALUDE_tangent_sphere_radius_l2957_295738

/-- Given a sphere of radius R and three mutually perpendicular planes drawn through its center,
    the radius x of a sphere that is tangent to all these planes and the given sphere
    is x = (√3 ± 1)R / 2. -/
theorem tangent_sphere_radius (R : ℝ) (R_pos : R > 0) :
  ∃ x : ℝ, x > 0 ∧
  (x = (Real.sqrt 3 + 1) * R / 2 ∨ x = (Real.sqrt 3 - 1) * R / 2) :=
by sorry

end NUMINAMATH_CALUDE_tangent_sphere_radius_l2957_295738


namespace NUMINAMATH_CALUDE_problem_solution_l2957_295706

-- Definition of the relation (x, y) = n
def relation (x y n : ℝ) : Prop := x^n = y

theorem problem_solution :
  -- Part 1
  relation 10 1000 3 ∧
  relation (-5) 25 2 ∧
  -- Part 2
  (∀ x, relation x 16 2 → (x = 4 ∨ x = -4)) ∧
  -- Part 3
  (∀ a b, relation 4 a 2 → relation b 8 3 → relation b a 4) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2957_295706


namespace NUMINAMATH_CALUDE_valve_emission_difference_l2957_295726

/-- The difference in water emission rates between two valves filling a pool -/
theorem valve_emission_difference (pool_capacity : ℝ) (both_valves_time : ℝ) (first_valve_time : ℝ) : 
  pool_capacity > 0 → 
  both_valves_time > 0 → 
  first_valve_time > 0 → 
  pool_capacity / both_valves_time - pool_capacity / first_valve_time = 50 := by
  sorry

#check valve_emission_difference 12000 48 120

end NUMINAMATH_CALUDE_valve_emission_difference_l2957_295726


namespace NUMINAMATH_CALUDE_model_M_completion_time_l2957_295753

/-- The time (in minutes) taken by a model N computer to complete the task -/
def model_N_time : ℝ := 18

/-- The number of model M computers used -/
def num_model_M : ℝ := 12

/-- The time (in minutes) taken to complete the task when using both models -/
def total_time : ℝ := 1

/-- The time (in minutes) taken by a model M computer to complete the task -/
def model_M_time : ℝ := 36

theorem model_M_completion_time :
  (num_model_M / model_M_time + num_model_M / model_N_time) * total_time = num_model_M :=
by sorry

end NUMINAMATH_CALUDE_model_M_completion_time_l2957_295753


namespace NUMINAMATH_CALUDE_smallest_possible_median_l2957_295777

def number_set (x : ℤ) : Finset ℤ := {x, 3*x, 4, 1, 6}

def is_median (m : ℤ) (s : Finset ℤ) : Prop :=
  2 * (s.filter (λ y => y ≤ m)).card ≥ s.card ∧
  2 * (s.filter (λ y => y ≥ m)).card ≥ s.card

theorem smallest_possible_median :
  ∃ (x : ℤ), is_median 1 (number_set x) ∧
  ∀ (y : ℤ) (m : ℤ), is_median m (number_set y) → m ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_smallest_possible_median_l2957_295777


namespace NUMINAMATH_CALUDE_nicki_total_miles_l2957_295785

/-- Represents the number of weeks in a year -/
def weeks_in_year : ℕ := 52

/-- Represents the number of miles Nicki ran per week in the first half of the year -/
def first_half_miles_per_week : ℕ := 20

/-- Represents the number of miles Nicki ran per week in the second half of the year -/
def second_half_miles_per_week : ℕ := 30

/-- Calculates the total miles Nicki ran for the year -/
def total_miles_run : ℕ := 
  (first_half_miles_per_week * (weeks_in_year / 2)) + 
  (second_half_miles_per_week * (weeks_in_year / 2))

/-- Theorem stating that Nicki ran 1300 miles in total for the year -/
theorem nicki_total_miles : total_miles_run = 1300 := by
  sorry

end NUMINAMATH_CALUDE_nicki_total_miles_l2957_295785


namespace NUMINAMATH_CALUDE_school_population_l2957_295770

theorem school_population (b g t a : ℕ) : 
  b = 4 * g ∧ 
  g = 8 * t ∧ 
  t = 2 * a → 
  b + g + t + a = 83 * a :=
by sorry

end NUMINAMATH_CALUDE_school_population_l2957_295770


namespace NUMINAMATH_CALUDE_f_and_g_increasing_l2957_295795

-- Define the functions
def f (x : ℝ) : ℝ := x^3
def g (x : ℝ) : ℝ := x^(1/2)

-- State the theorem
theorem f_and_g_increasing :
  (∀ x y : ℝ, x < y → f x < f y) ∧
  (∀ x y : ℝ, 0 ≤ x ∧ x < y → g x < g y) :=
sorry

end NUMINAMATH_CALUDE_f_and_g_increasing_l2957_295795


namespace NUMINAMATH_CALUDE_test_total_points_l2957_295709

theorem test_total_points (total_questions : ℕ) (two_point_questions : ℕ) : 
  total_questions = 40 → 
  two_point_questions = 30 → 
  (total_questions - two_point_questions) * 4 + two_point_questions * 2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_test_total_points_l2957_295709


namespace NUMINAMATH_CALUDE_same_last_five_digits_l2957_295727

theorem same_last_five_digits (N : ℕ) : N = 3125 ↔ 
  (N > 0) ∧ 
  (∃ (a b c d e : ℕ), 
    a ≠ 0 ∧ 
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧
    N % 100000 = a * 10000 + b * 1000 + c * 100 + d * 10 + e ∧
    (N^2) % 100000 = a * 10000 + b * 1000 + c * 100 + d * 10 + e) ∧
  (∀ M : ℕ, M < N → 
    (M > 0) → 
    (∀ (a b c d e : ℕ), 
      a ≠ 0 → 
      a < 10 → b < 10 → c < 10 → d < 10 → e < 10 →
      M % 100000 = a * 10000 + b * 1000 + c * 100 + d * 10 + e →
      (M^2) % 100000 ≠ a * 10000 + b * 1000 + c * 100 + d * 10 + e)) :=
by sorry

end NUMINAMATH_CALUDE_same_last_five_digits_l2957_295727


namespace NUMINAMATH_CALUDE_gcd_660_924_l2957_295763

theorem gcd_660_924 : Nat.gcd 660 924 = 132 := by
  sorry

end NUMINAMATH_CALUDE_gcd_660_924_l2957_295763


namespace NUMINAMATH_CALUDE_final_number_calculation_l2957_295776

theorem final_number_calculation : 
  let initial_number : ℕ := 9
  let doubled := initial_number * 2
  let added_13 := doubled + 13
  let final_number := added_13 * 3
  final_number = 93 := by sorry

end NUMINAMATH_CALUDE_final_number_calculation_l2957_295776


namespace NUMINAMATH_CALUDE_rose_discount_percentage_l2957_295772

theorem rose_discount_percentage (dozen_count : ℕ) (cost_per_rose : ℕ) (final_amount : ℕ) : 
  dozen_count = 5 → 
  cost_per_rose = 6 → 
  final_amount = 288 → 
  (1 - (final_amount : ℚ) / (dozen_count * 12 * cost_per_rose)) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_rose_discount_percentage_l2957_295772


namespace NUMINAMATH_CALUDE_total_score_approximation_l2957_295714

/-- Represents the types of shots in a basketball game -/
inductive ShotType
  | ThreePoint
  | TwoPoint
  | FreeThrow

/-- Represents the success rate for each shot type -/
def successRate (shot : ShotType) : ℝ :=
  match shot with
  | ShotType.ThreePoint => 0.25
  | ShotType.TwoPoint => 0.50
  | ShotType.FreeThrow => 0.80

/-- Represents the point value for each shot type -/
def pointValue (shot : ShotType) : ℕ :=
  match shot with
  | ShotType.ThreePoint => 3
  | ShotType.TwoPoint => 2
  | ShotType.FreeThrow => 1

/-- The total number of shots attempted -/
def totalShots : ℕ := 40

/-- Calculates the number of attempts for each shot type, assuming equal distribution -/
def attemptsPerType : ℕ := totalShots / 3

/-- Calculates the points scored for a given shot type -/
def pointsScored (shot : ShotType) : ℝ :=
  (successRate shot) * (pointValue shot : ℝ) * (attemptsPerType : ℝ)

/-- Calculates the total points scored across all shot types -/
def totalPointsScored : ℝ :=
  pointsScored ShotType.ThreePoint + pointsScored ShotType.TwoPoint + pointsScored ShotType.FreeThrow

/-- Theorem stating that the total points scored is approximately 33 -/
theorem total_score_approximation :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ |totalPointsScored - 33| < ε :=
sorry

end NUMINAMATH_CALUDE_total_score_approximation_l2957_295714


namespace NUMINAMATH_CALUDE_seating_theorem_l2957_295751

/-- The number of ways to arrange n objects from m choices --/
def permutation (n m : ℕ) : ℕ := 
  if n > m then 0
  else Nat.factorial m / Nat.factorial (m - n)

/-- The number of ways four people can sit in a row of five chairs --/
def seating_arrangements : ℕ := permutation 4 5

theorem seating_theorem : seating_arrangements = 120 := by
  sorry

end NUMINAMATH_CALUDE_seating_theorem_l2957_295751


namespace NUMINAMATH_CALUDE_choose_4_from_10_l2957_295718

theorem choose_4_from_10 : Nat.choose 10 4 = 210 := by
  sorry

end NUMINAMATH_CALUDE_choose_4_from_10_l2957_295718


namespace NUMINAMATH_CALUDE_original_paint_intensity_l2957_295778

-- Define the paint mixing problem
def paint_mixing (original_intensity : ℝ) : Prop :=
  let f : ℝ := 1/3  -- fraction of original paint replaced
  let replacement_intensity : ℝ := 20  -- 20% solution
  let final_intensity : ℝ := 40  -- 40% final intensity
  (1 - f) * original_intensity + f * replacement_intensity = final_intensity

-- Theorem statement
theorem original_paint_intensity :
  ∃ (original_intensity : ℝ), paint_mixing original_intensity ∧ original_intensity = 50 := by
  sorry

end NUMINAMATH_CALUDE_original_paint_intensity_l2957_295778


namespace NUMINAMATH_CALUDE_jennifer_book_fraction_l2957_295766

theorem jennifer_book_fraction (total : ℚ) (sandwich_fraction : ℚ) (museum_fraction : ℚ) (leftover : ℚ) :
  total = 90 →
  sandwich_fraction = 1 / 5 →
  museum_fraction = 1 / 6 →
  leftover = 12 →
  let spent := total - leftover
  let sandwich_cost := total * sandwich_fraction
  let museum_cost := total * museum_fraction
  let book_cost := spent - sandwich_cost - museum_cost
  book_cost / total = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_jennifer_book_fraction_l2957_295766


namespace NUMINAMATH_CALUDE_min_value_quadratic_form_l2957_295734

theorem min_value_quadratic_form (x y : ℝ) : x^2 - x*y + y^2 ≥ 0 ∧ 
  (x^2 - x*y + y^2 = 0 ↔ x = 0 ∧ y = 0) := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_form_l2957_295734


namespace NUMINAMATH_CALUDE_percent_of_x_is_z_l2957_295711

theorem percent_of_x_is_z (x y z w : ℝ) 
  (h1 : 0.45 * z = 0.72 * y)
  (h2 : y = 0.75 * x)
  (h3 : w = 0.60 * z^2)
  (h4 : z = 0.30 * w^(1/3)) :
  z = 1.2 * x := by
sorry

end NUMINAMATH_CALUDE_percent_of_x_is_z_l2957_295711


namespace NUMINAMATH_CALUDE_sum_first_six_terms_eq_54_l2957_295717

/-- An arithmetic sequence with given 3rd, 4th, and 5th terms -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  a 3 = 7 ∧ a 4 = 11 ∧ a 5 = 15 ∧ ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

/-- The sum of the first six terms of the sequence -/
def SumFirstSixTerms (a : ℕ → ℤ) : ℤ :=
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6

/-- Theorem stating that the sum of the first six terms is 54 -/
theorem sum_first_six_terms_eq_54 (a : ℕ → ℤ) (h : ArithmeticSequence a) :
  SumFirstSixTerms a = 54 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_six_terms_eq_54_l2957_295717


namespace NUMINAMATH_CALUDE_sector_central_angle_l2957_295736

theorem sector_central_angle (area : Real) (radius : Real) (central_angle : Real) :
  area = 3 * Real.pi / 8 →
  radius = 1 →
  area = 1 / 2 * central_angle * radius ^ 2 →
  central_angle = 3 * Real.pi / 4 := by
sorry

end NUMINAMATH_CALUDE_sector_central_angle_l2957_295736


namespace NUMINAMATH_CALUDE_area_R_specific_rhombus_l2957_295701

/-- Represents a rhombus ABCD -/
structure Rhombus where
  side_length : ℝ
  angle_B : ℝ

/-- Represents the region R inside the rhombus -/
def region_R (r : Rhombus) : Set (ℝ × ℝ) :=
  sorry

/-- The area of region R in the rhombus -/
def area_R (r : Rhombus) : ℝ :=
  sorry

/-- Theorem stating the area of region R in the specific rhombus -/
theorem area_R_specific_rhombus :
  let r : Rhombus := { side_length := 3, angle_B := 150 * π / 180 }
  area_R r = 9 * (Real.sqrt 6 - Real.sqrt 2) / 8 := by
    sorry

end NUMINAMATH_CALUDE_area_R_specific_rhombus_l2957_295701


namespace NUMINAMATH_CALUDE_weight_replaced_person_correct_l2957_295775

/-- Represents the weight change scenario of a group of people -/
structure WeightChangeScenario where
  initial_count : ℕ
  average_increase : ℝ
  new_person_weight : ℝ

/-- Calculates the weight of the replaced person given a WeightChangeScenario -/
def weight_of_replaced_person (scenario : WeightChangeScenario) : ℝ :=
  scenario.new_person_weight - scenario.initial_count * scenario.average_increase

theorem weight_replaced_person_correct (scenario : WeightChangeScenario) :
  scenario.initial_count = 6 →
  scenario.average_increase = 2.5 →
  scenario.new_person_weight = 80 →
  weight_of_replaced_person scenario = 65 := by
  sorry

end NUMINAMATH_CALUDE_weight_replaced_person_correct_l2957_295775


namespace NUMINAMATH_CALUDE_p_satisfies_equation_l2957_295733

/-- The polynomial p(x) that satisfies the given equation -/
def p (x : ℝ) : ℝ := (x - 2) * (x - 4) * (x - 8) * (x - 16)

/-- Theorem stating that p(x) satisfies the given equation for all real x -/
theorem p_satisfies_equation (x : ℝ) : (x - 16) * p (2 * x) = (16 * x - 16) * p x := by
  sorry

end NUMINAMATH_CALUDE_p_satisfies_equation_l2957_295733


namespace NUMINAMATH_CALUDE_product_of_fractions_l2957_295781

theorem product_of_fractions : (2 : ℚ) / 3 * (3 : ℚ) / 8 = (1 : ℚ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l2957_295781


namespace NUMINAMATH_CALUDE_prob_A_leading_after_three_prob_B_wins_3_2_l2957_295754

-- Define the probability of Team A winning a single game
def p_A_win : ℝ := 0.60

-- Define the probability of Team B winning a single game
def p_B_win : ℝ := 1 - p_A_win

-- Define the number of games needed to win the match
def games_to_win : ℕ := 3

-- Define the total number of games in a full match
def total_games : ℕ := 5

-- Theorem for the probability of Team A leading after the first three games
theorem prob_A_leading_after_three : 
  (Finset.sum (Finset.range 2) (λ k => Nat.choose 3 (3 - k) * p_A_win ^ (3 - k) * p_B_win ^ k)) = 0.648 := by sorry

-- Theorem for the probability of Team B winning the match with a score of 3:2
theorem prob_B_wins_3_2 : 
  (Nat.choose 4 2 * p_A_win ^ 2 * p_B_win ^ 2 * p_B_win) = 0.138 := by sorry

end NUMINAMATH_CALUDE_prob_A_leading_after_three_prob_B_wins_3_2_l2957_295754


namespace NUMINAMATH_CALUDE_function_passes_through_point_l2957_295724

theorem function_passes_through_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∃ y : ℝ, y = a * 2 - 2 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l2957_295724


namespace NUMINAMATH_CALUDE_coin_flip_theorem_l2957_295721

/-- Represents the state of coins on a table -/
structure CoinState where
  total_coins : ℕ
  two_ruble_coins : ℕ
  five_ruble_coins : ℕ
  visible_sum : ℕ

/-- Checks if a CoinState is valid according to the problem conditions -/
def is_valid_state (state : CoinState) : Prop :=
  state.total_coins = 14 ∧
  state.two_ruble_coins + state.five_ruble_coins = state.total_coins ∧
  state.two_ruble_coins > 0 ∧
  state.five_ruble_coins > 0 ∧
  state.visible_sum ≤ 2 * state.two_ruble_coins + 5 * state.five_ruble_coins

/-- Calculates the new visible sum after flipping all coins -/
def flipped_sum (state : CoinState) : ℕ :=
  2 * state.two_ruble_coins + 5 * state.five_ruble_coins - state.visible_sum

/-- The main theorem to prove -/
theorem coin_flip_theorem (state : CoinState) :
  is_valid_state state →
  flipped_sum state = 3 * state.visible_sum →
  state.five_ruble_coins = 4 ∨ state.five_ruble_coins = 8 ∨ state.five_ruble_coins = 12 := by
  sorry


end NUMINAMATH_CALUDE_coin_flip_theorem_l2957_295721
