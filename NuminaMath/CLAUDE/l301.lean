import Mathlib

namespace polynomial_coefficients_l301_30126

/-- Given that (1-2x)^5 = a₀ + a₁x + a₂x² + a₃x³ + a₄x⁴ + a₅x⁵,
    prove the following statements about the coefficients. -/
theorem polynomial_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (1 - 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a₀ = 1 ∧ 
   a₁ + a₂ + a₃ + a₄ + a₅ = -2 ∧
   a₁ + a₃ + a₅ = -122) :=
by sorry

end polynomial_coefficients_l301_30126


namespace angle_between_vectors_l301_30106

/-- Given complex numbers z₁, z₂, z₃ satisfying (z₃ - z₁) / (z₂ - z₁) = ai 
    where a ∈ ℝ and a ≠ 0, the angle between vectors ⃗Z₁Z₂ and ⃗Z₁Z₃ is π/2. -/
theorem angle_between_vectors (z₁ z₂ z₃ : ℂ) (a : ℝ) 
    (h : (z₃ - z₁) / (z₂ - z₁) = Complex.I * a) 
    (ha : a ≠ 0) : 
  Complex.arg ((z₃ - z₁) / (z₂ - z₁)) = π / 2 := by
  sorry

end angle_between_vectors_l301_30106


namespace equation_solution_l301_30130

theorem equation_solution (y : ℚ) : 
  (40 : ℚ) / 60 = Real.sqrt (y / 60) → y = 110 / 3 := by
  sorry

end equation_solution_l301_30130


namespace lawn_mowing_solution_l301_30128

/-- Represents the lawn mowing problem -/
def LawnMowingProblem (lawn_length lawn_width swath_width overlap mowing_speed : ℝ) : Prop :=
  let effective_width := (swath_width - overlap) / 12  -- Convert to feet
  let strips := lawn_width / effective_width
  let total_distance := strips * lawn_length
  let mowing_time := total_distance / mowing_speed
  0.94 < mowing_time ∧ mowing_time < 0.96

/-- Theorem stating the solution to the lawn mowing problem -/
theorem lawn_mowing_solution :
  LawnMowingProblem 72 120 (30/12) (6/12) 4500 :=
sorry

end lawn_mowing_solution_l301_30128


namespace train_length_l301_30168

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 180 → time = 8 → speed * time * (1000 / 3600) = 400 :=
by sorry

end train_length_l301_30168


namespace custom_equation_solution_l301_30143

-- Define the custom operation *
def star (a b : ℚ) : ℚ := 4 * a - 2 * b

-- State the theorem
theorem custom_equation_solution :
  ∃! x : ℚ, star 3 (star 6 x) = -2 ∧ x = 17/2 := by sorry

end custom_equation_solution_l301_30143


namespace triangle_properties_l301_30114

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  (a - c) / (a * Real.cos C + c * Real.cos A) = (b - c) / (a + c) →
  a + b + c ≤ 3 * Real.sqrt 3 →
  A = π / 3 ∧ a / (2 * Real.sin (π / 3)) = 1 := by
  sorry

end triangle_properties_l301_30114


namespace fruit_condition_percentage_l301_30109

theorem fruit_condition_percentage (oranges bananas : ℕ) 
  (rotten_oranges_percent rotten_bananas_percent : ℚ) :
  oranges = 600 →
  bananas = 400 →
  rotten_oranges_percent = 15 / 100 →
  rotten_bananas_percent = 8 / 100 →
  let total_fruits := oranges + bananas
  let rotten_oranges := (rotten_oranges_percent * oranges).num
  let rotten_bananas := (rotten_bananas_percent * bananas).num
  let total_rotten := rotten_oranges + rotten_bananas
  let good_fruits := total_fruits - total_rotten
  (good_fruits : ℚ) / total_fruits * 100 = 87.8 := by
sorry

end fruit_condition_percentage_l301_30109


namespace star_sharing_problem_l301_30163

theorem star_sharing_problem (stars : ℝ) (students_per_star : ℝ) 
  (h1 : stars = 3.0) 
  (h2 : students_per_star = 41.33333333) : 
  ⌊stars * students_per_star⌋ = 124 := by
  sorry

end star_sharing_problem_l301_30163


namespace merry_go_round_ride_times_l301_30104

theorem merry_go_round_ride_times 
  (dave_time : ℝ) 
  (erica_time : ℝ) 
  (erica_longer_percent : ℝ) :
  dave_time = 10 →
  erica_time = 65 →
  erica_longer_percent = 0.30 →
  ∃ (chuck_time : ℝ),
    erica_time = chuck_time * (1 + erica_longer_percent) ∧
    chuck_time / dave_time = 5 :=
by sorry

end merry_go_round_ride_times_l301_30104


namespace lollipop_cost_is_two_l301_30189

/-- The cost of a single lollipop in dollars -/
def lollipop_cost : ℝ := 2

/-- The number of lollipops bought -/
def num_lollipops : ℕ := 4

/-- The number of chocolate packs bought -/
def num_chocolate_packs : ℕ := 6

/-- The number of $10 bills used for payment -/
def num_ten_dollar_bills : ℕ := 6

/-- The amount of change received in dollars -/
def change_received : ℝ := 4

theorem lollipop_cost_is_two :
  lollipop_cost = 2 ∧
  num_lollipops * lollipop_cost + num_chocolate_packs * (4 * lollipop_cost) = 
    num_ten_dollar_bills * 10 - change_received :=
by sorry

end lollipop_cost_is_two_l301_30189


namespace michael_monica_ratio_l301_30161

/-- The ages of three people satisfy certain conditions -/
structure AgesProblem where
  /-- Patrick's age -/
  p : ℕ
  /-- Michael's age -/
  m : ℕ
  /-- Monica's age -/
  mo : ℕ
  /-- The ages of Patrick and Michael are in the ratio of 3:5 -/
  patrick_michael_ratio : 3 * m = 5 * p
  /-- The sum of their ages is 88 -/
  sum_of_ages : p + m + mo = 88
  /-- The difference between Monica and Patrick's ages is 22 -/
  monica_patrick_diff : mo - p = 22

/-- The ratio of Michael's age to Monica's age is 3:4 -/
theorem michael_monica_ratio (prob : AgesProblem) : 3 * prob.mo = 4 * prob.m := by
  sorry

end michael_monica_ratio_l301_30161


namespace boarding_students_count_total_boarding_students_l301_30151

theorem boarding_students_count (total_students : ℕ) (male_students : ℕ) 
  (female_youth_league : ℕ) (female_boarding : ℕ) (non_boarding_youth_league : ℕ) 
  (male_boarding_youth_league : ℕ) (male_non_youth_league_non_boarding : ℕ) 
  (female_non_youth_league_non_boarding : ℕ) : ℕ :=
  sorry

/-- Given the following conditions:
    - There are 50 students in total
    - There are 33 male students
    - There are 7 female members of the Youth League
    - There are 9 female boarding students
    - There are 15 non-boarding members of the Youth League
    - There are 6 male boarding members of the Youth League
    - There are 8 male students who are non-members of the Youth League and non-boarding
    - There are 3 female students who are non-members of the Youth League and non-boarding
    The total number of boarding students is 28. -/
theorem total_boarding_students :
  boarding_students_count 50 33 7 9 15 6 8 3 = 28 := by
  sorry

end boarding_students_count_total_boarding_students_l301_30151


namespace casino_game_max_guaranteed_money_l301_30149

/-- Represents the outcome of a single bet -/
inductive BetOutcome
| Win
| Lose

/-- Represents the state of the game after each bet -/
structure GameState :=
  (money : ℕ)
  (bets_made : ℕ)
  (consecutive_losses : ℕ)

/-- The betting strategy function type -/
def BettingStrategy := GameState → ℕ

/-- The game rules function type -/
def GameRules := GameState → BetOutcome → GameState

theorem casino_game_max_guaranteed_money 
  (initial_money : ℕ) 
  (max_bets : ℕ) 
  (max_bet_amount : ℕ) 
  (consolation_win_threshold : ℕ) 
  (strategy : BettingStrategy) 
  (rules : GameRules) :
  initial_money = 100 →
  max_bets = 5 →
  max_bet_amount = 17 →
  consolation_win_threshold = 4 →
  ∃ (final_money : ℕ), final_money ≥ 98 ∧
    ∀ (outcomes : List BetOutcome), 
      outcomes.length = max_bets →
      let final_state := outcomes.foldl rules { money := initial_money, bets_made := 0, consecutive_losses := 0 }
      final_state.money ≥ final_money :=
by sorry

end casino_game_max_guaranteed_money_l301_30149


namespace set_operations_l301_30156

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x < -2 ∨ x > 5}

def B : Set ℝ := {x | 4 ≤ x ∧ x ≤ 6}

theorem set_operations :
  (Aᶜ : Set ℝ) = {x | -2 ≤ x ∧ x ≤ 5} ∧
  (Bᶜ : Set ℝ) = {x | x < 4 ∨ x > 6} ∧
  (A ∩ B : Set ℝ) = {x | 5 < x ∧ x ≤ 6} ∧
  ((A ∪ B)ᶜ : Set ℝ) = {x | -2 ≤ x ∧ x < 4} :=
by sorry

end set_operations_l301_30156


namespace arrangements_count_is_24_l301_30157

/-- The number of ways to arrange 5 people in a line, where two specific people
    must stand next to each other but not at the ends. -/
def arrangements_count : ℕ :=
  /- Number of ways to arrange A and B together -/
  (2 * 1) *
  /- Number of positions for A and B together (excluding ends) -/
  3 *
  /- Number of ways to arrange the other 3 people -/
  (3 * 2 * 1)

/-- Theorem stating that the number of arrangements is 24 -/
theorem arrangements_count_is_24 : arrangements_count = 24 := by
  sorry

end arrangements_count_is_24_l301_30157


namespace bob_pennies_bob_pennies_proof_l301_30112

theorem bob_pennies : ℕ → ℕ → Prop :=
  fun a b =>
    (b + 1 = 4 * (a - 1)) ∧
    (b - 1 = 3 * (a + 1)) →
    b = 31

-- The proof goes here
theorem bob_pennies_proof : bob_pennies 9 31 := by
  sorry

end bob_pennies_bob_pennies_proof_l301_30112


namespace box_volume_increase_l301_30138

theorem box_volume_increase (l w h : ℝ) 
  (volume : l * w * h = 5000)
  (surface_area : 2 * (l * w + w * h + l * h) = 1800)
  (edge_sum : 4 * (l + w + h) = 240) :
  (l + 2) * (w + 2) * (h + 2) = 7048 := by
  sorry

end box_volume_increase_l301_30138


namespace park_area_theorem_l301_30169

/-- Represents a rectangular park with a given perimeter where the width is one-third of the length -/
structure RectangularPark where
  perimeter : ℝ
  width : ℝ
  length : ℝ
  width_length_relation : width = length / 3
  perimeter_constraint : perimeter = 2 * (width + length)

/-- Calculates the area of a rectangular park -/
def parkArea (park : RectangularPark) : ℝ :=
  park.width * park.length

/-- Theorem stating that a rectangular park with a perimeter of 90 meters and width one-third of its length has an area of 379.6875 square meters -/
theorem park_area_theorem (park : RectangularPark) (h : park.perimeter = 90) : 
  parkArea park = 379.6875 := by
  sorry

end park_area_theorem_l301_30169


namespace rebecca_camping_items_l301_30171

/-- Represents the number of tent stakes Rebecca bought. -/
def tent_stakes : ℕ := sorry

/-- Represents the number of packets of drink mix Rebecca bought. -/
def drink_mix : ℕ := 3 * tent_stakes

/-- Represents the number of bottles of water Rebecca bought. -/
def water_bottles : ℕ := tent_stakes + 2

/-- The total number of items Rebecca bought. -/
def total_items : ℕ := 22

theorem rebecca_camping_items : tent_stakes = 4 :=
  by sorry

end rebecca_camping_items_l301_30171


namespace min_detectors_for_gold_coins_l301_30115

/-- Represents a grid of unit squares -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a subgrid within a larger grid -/
structure Subgrid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents the minimum number of detectors needed -/
def min_detectors (g : Grid) (s : Subgrid) : ℕ := sorry

/-- The main theorem stating the minimum number of detectors needed -/
theorem min_detectors_for_gold_coins (g : Grid) (s : Subgrid) :
  g.rows = 2017 ∧ g.cols = 2017 ∧ s.rows = 1500 ∧ s.cols = 1500 →
  min_detectors g s = 1034 := by sorry

end min_detectors_for_gold_coins_l301_30115


namespace x_value_proof_l301_30100

theorem x_value_proof (x y : ℝ) (h1 : x > y) 
  (h2 : x^2 * y^2 + x^2 + y^2 + 2*x*y = 40) 
  (h3 : x*y + x + y = 8) : x = 3 + Real.sqrt 7 := by
  sorry

end x_value_proof_l301_30100


namespace area_decreasing_map_l301_30124

open Set
open MeasureTheory

-- Define a distance function for ℝ²
noncomputable def distance (x y : ℝ × ℝ) : ℝ := Real.sqrt ((x.1 - y.1)^2 + (x.2 - y.2)^2)

-- Define the properties of function f
def is_distance_decreasing (f : ℝ × ℝ → ℝ × ℝ) : Prop :=
  ∀ x y : ℝ × ℝ, distance x y ≥ distance (f x) (f y)

-- Theorem statement
theorem area_decreasing_map
  (f : ℝ × ℝ → ℝ × ℝ)
  (h_inj : Function.Injective f)
  (h_surj : Function.Surjective f)
  (h_dist : is_distance_decreasing f)
  (A : Set (ℝ × ℝ)) :
  MeasureTheory.volume A ≥ MeasureTheory.volume (f '' A) :=
sorry

end area_decreasing_map_l301_30124


namespace sum_of_products_power_inequality_l301_30131

theorem sum_of_products_power_inequality (a b c : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (sum_one : a + b + c = 1) : 
  (a * b) ^ (5/4 : ℝ) + (b * c) ^ (5/4 : ℝ) + (c * a) ^ (5/4 : ℝ) < 1/4 := by
  sorry

end sum_of_products_power_inequality_l301_30131


namespace sequence_relationship_l301_30150

def y (x : ℕ) : ℕ := x^2 + x + 1

theorem sequence_relationship (x : ℕ) :
  x > 0 →
  (y (x + 1) - y x = 2 * x + 2) ∧
  (y (x + 2) - y (x + 1) = 2 * x + 4) ∧
  (y (x + 3) - y (x + 2) = 2 * x + 6) ∧
  (y (x + 4) - y (x + 3) = 2 * x + 8) :=
by sorry

end sequence_relationship_l301_30150


namespace roots_sum_product_l301_30102

theorem roots_sum_product (p q r : ℂ) : 
  (2 * p ^ 3 - 5 * p ^ 2 + 7 * p - 3 = 0) →
  (2 * q ^ 3 - 5 * q ^ 2 + 7 * q - 3 = 0) →
  (2 * r ^ 3 - 5 * r ^ 2 + 7 * r - 3 = 0) →
  p * q + q * r + r * p = 7 / 2 := by
sorry

end roots_sum_product_l301_30102


namespace equal_roots_quadratic_l301_30182

theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - m * x + 3 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 - m * y + 3 = 0 → y = x) → 
  m = 6 ∨ m = -6 :=
by sorry

end equal_roots_quadratic_l301_30182


namespace largest_angle_is_75_l301_30145

-- Define the angles of the triangle
def triangle_angles (a b c : ℝ) : Prop :=
  -- The sum of all angles in a triangle is 180°
  a + b + c = 180 ∧
  -- Two angles sum to 7/6 of a right angle (90°)
  b + c = 7/6 * 90 ∧
  -- One angle is 10° more than twice the other
  c = 2 * b + 10 ∧
  -- All angles are non-negative
  0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c

-- Theorem statement
theorem largest_angle_is_75 (a b c : ℝ) :
  triangle_angles a b c → max a (max b c) = 75 :=
by sorry

end largest_angle_is_75_l301_30145


namespace circle_symmetry_l301_30110

-- Define the circle C1
def C1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 8*y + 19 = 0

-- Define the line l
def l (x y : ℝ) : Prop := x + 2*y - 5 = 0

-- Define symmetry with respect to a line
def symmetric_wrt_line (C1 C2 l : ℝ → ℝ → Prop) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ), C1 x1 y1 ∧ C2 x2 y2 ∧
  (x1 + x2) / 2 + 2 * ((y1 + y2) / 2) - 5 = 0 ∧
  (y2 - y1) / (x2 - x1) * (-1/2) = -1

-- Define circle C2
def C2 (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Theorem statement
theorem circle_symmetry :
  symmetric_wrt_line C1 C2 l → ∀ x y, C2 x y ↔ x^2 + y^2 = 1 :=
sorry

end circle_symmetry_l301_30110


namespace consecutive_divisible_numbers_l301_30183

theorem consecutive_divisible_numbers :
  ∃ n : ℕ, 100 ≤ n ∧ n < 200 ∧ 
    3 ∣ n ∧ 5 ∣ (n + 1) ∧ 7 ∣ (n + 2) :=
by sorry

end consecutive_divisible_numbers_l301_30183


namespace bob_alice_difference_l301_30164

/-- The difference in final amounts between two investors, given their initial investment
    and respective returns. -/
def investment_difference (initial_investment : ℕ) (alice_multiplier bob_multiplier : ℕ) : ℕ :=
  (initial_investment * bob_multiplier + initial_investment) - (initial_investment * alice_multiplier)

/-- Theorem stating that given the problem conditions, Bob ends up with $8000 more than Alice. -/
theorem bob_alice_difference : investment_difference 2000 2 5 = 8000 := by
  sorry

end bob_alice_difference_l301_30164


namespace weight_of_a_l301_30132

-- Define the people
structure Person where
  weight : ℝ
  height : ℝ
  age : ℝ

-- Define the group of A, B, C
def group_abc (a b c : Person) : Prop :=
  (a.weight + b.weight + c.weight) / 3 = 84 ∧
  (a.height + b.height + c.height) / 3 = 170 ∧
  (a.age + b.age + c.age) / 3 = 30

-- Define the group with D added
def group_abcd (a b c d : Person) : Prop :=
  (a.weight + b.weight + c.weight + d.weight) / 4 = 80 ∧
  (a.height + b.height + c.height + d.height) / 4 = 172 ∧
  (a.age + b.age + c.age + d.age) / 4 = 28

-- Define the group with E replacing A
def group_bcde (b c d e : Person) : Prop :=
  (b.weight + c.weight + d.weight + e.weight) / 4 = 79 ∧
  (b.height + c.height + d.height + e.height) / 4 = 173 ∧
  (b.age + c.age + d.age + e.age) / 4 = 27

-- Define the relationship between D and E
def d_e_relation (d e a : Person) : Prop :=
  e.weight = d.weight + 7 ∧
  e.age = a.age - 3

-- Theorem statement
theorem weight_of_a 
  (a b c d e : Person)
  (h1 : group_abc a b c)
  (h2 : group_abcd a b c d)
  (h3 : group_bcde b c d e)
  (h4 : d_e_relation d e a) :
  a.weight = 79 := by
  sorry

end weight_of_a_l301_30132


namespace parabola_directrix_l301_30160

theorem parabola_directrix (x y : ℝ) :
  y = 4 * x^2 → (∃ (k : ℝ), y = -1/(4*k) ∧ k = 1/4) :=
by sorry

end parabola_directrix_l301_30160


namespace simplify_fraction_1_simplify_fraction_2_l301_30187

-- Problem 1
theorem simplify_fraction_1 (a : ℝ) (h : a ≠ -1) :
  (2 * a^2 - 3) / (a + 1) - (a^2 - 2) / (a + 1) = a - 1 := by
  sorry

-- Problem 2
theorem simplify_fraction_2 (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 2) :
  (x / (x^2 - 4)) / (x / (4 - 2*x)) = -2 / (x + 2) := by
  sorry

end simplify_fraction_1_simplify_fraction_2_l301_30187


namespace equation_solution_l301_30154

theorem equation_solution (y : ℝ) (h : y ≠ 0) : 
  (2 / y) + (3 / y) / (6 / y) = 1.5 → y = 2 :=
by sorry

end equation_solution_l301_30154


namespace greatest_x_value_l301_30173

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem greatest_x_value (x y : ℕ) (a b : ℝ) :
  x > 0 →
  y > 0 →
  is_prime y →
  a > 1 →
  b > 1 →
  a = 2.75 →
  b = 4.26 →
  ((a * x^2) / (y^3 : ℝ)) + b < 800000 →
  Nat.gcd x y = 1 →
  (∀ x' y' : ℕ, x' > 0 → y' > 0 → is_prime y' → Nat.gcd x' y' = 1 →
    ((a * x'^2) / (y'^3 : ℝ)) + b < 800000 → x' + y' < x + y) →
  x ≤ 2801 :=
sorry

end greatest_x_value_l301_30173


namespace additional_workers_for_wall_project_l301_30142

/-- Calculates the number of additional workers needed to complete a project on time -/
def additional_workers_needed (total_days : ℕ) (initial_workers : ℕ) (days_passed : ℕ) (work_completed_percentage : ℚ) : ℕ :=
  let total_work := total_days * initial_workers
  let remaining_work := total_work * (1 - work_completed_percentage)
  let remaining_days := total_days - days_passed
  let work_by_existing := initial_workers * remaining_days
  let additional_work_needed := remaining_work - work_by_existing
  (additional_work_needed / remaining_days).ceil.toNat

/-- Proves that given the initial conditions, 12 additional workers are needed -/
theorem additional_workers_for_wall_project : 
  additional_workers_needed 50 60 25 (2/5) = 12 := by
  sorry

end additional_workers_for_wall_project_l301_30142


namespace quadratic_function_example_l301_30117

theorem quadratic_function_example : ∃ (a b c : ℝ),
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  (f 1 = 0) ∧ (f 5 = 0) ∧ (f 3 = 10) := by
  sorry

end quadratic_function_example_l301_30117


namespace problem_solution_l301_30190

theorem problem_solution (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0)
  (h_eq : 7 * x^2 + 14 * x * y = 2 * x^3 + 4 * x^2 * y + y^3) : x = 7 := by
  sorry

end problem_solution_l301_30190


namespace positive_sum_l301_30113

theorem positive_sum (x y z : ℝ) 
  (hx : 0 < x ∧ x < 0.5) 
  (hy : -0.5 < y ∧ y < 0) 
  (hz : 0.5 < z ∧ z < 1) : 
  y + z > 0 := by
  sorry

end positive_sum_l301_30113


namespace missing_angle_is_zero_l301_30123

/-- Represents a polygon with a missing angle -/
structure PolygonWithMissingAngle where
  n : ℕ                     -- number of sides
  sum_without_missing : ℝ   -- sum of all angles except the missing one
  missing_angle : ℝ         -- the missing angle

/-- The theorem stating that the missing angle is 0° -/
theorem missing_angle_is_zero (p : PolygonWithMissingAngle) 
  (h1 : p.sum_without_missing = 3240)
  (h2 : p.sum_without_missing + p.missing_angle = 180 * (p.n - 2)) :
  p.missing_angle = 0 := by
sorry


end missing_angle_is_zero_l301_30123


namespace unique_poly3_satisfying_conditions_l301_30105

/-- A polynomial function of degree exactly 3 -/
structure Poly3 where
  f : ℝ → ℝ
  degree_3 : ∃ a b c d : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^3 + b * x^2 + c * x + d

/-- The conditions that the polynomial function must satisfy -/
def satisfies_conditions (p : Poly3) : Prop :=
  ∀ x : ℝ, p.f (x^2) = (p.f x)^2 ∧
            p.f (x^2) = p.f (p.f x) ∧
            p.f 1 = 1

/-- Theorem stating the uniqueness of the polynomial function -/
theorem unique_poly3_satisfying_conditions :
  ∃! p : Poly3, satisfies_conditions p :=
sorry

end unique_poly3_satisfying_conditions_l301_30105


namespace remainder_problem_l301_30146

theorem remainder_problem : 123456789012 % 360 = 108 := by
  sorry

end remainder_problem_l301_30146


namespace tangent_line_problem_l301_30139

theorem tangent_line_problem (a b : ℝ) : 
  let f : ℝ → ℝ := λ x => Real.log x + a / x
  let f' : ℝ → ℝ := λ x => 1 / x - a / (x^2)
  (∀ y, 4 * y - 1 - b = 0 ↔ y = f 1 + f' 1 * (1 - 1)) →
  a * b = 3 / 2 := by
  sorry

end tangent_line_problem_l301_30139


namespace shift_cosine_to_sine_l301_30162

theorem shift_cosine_to_sine (x : ℝ) :
  let original := λ x : ℝ => 2 * Real.cos (2 * x)
  let shifted := λ x : ℝ => original (x - π / 8)
  let target := λ x : ℝ => 2 * Real.sin (2 * x + π / 4)
  0 < π / 8 ∧ π / 8 < π / 2 →
  shifted = target := by sorry

end shift_cosine_to_sine_l301_30162


namespace difference_of_squares_l301_30192

theorem difference_of_squares (x y : ℚ) 
  (h1 : x + y = 15/26) 
  (h2 : x - y = 2/65) : 
  x^2 - y^2 = 15/845 := by
sorry

end difference_of_squares_l301_30192


namespace x_4_sufficient_not_necessary_l301_30197

def vector_a (x : ℝ) : Fin 2 → ℝ := ![x, 3]

theorem x_4_sufficient_not_necessary :
  (∀ x : ℝ, x = 4 → ‖vector_a x‖ = 5) ∧
  (∃ y : ℝ, y ≠ 4 ∧ ‖vector_a y‖ = 5) :=
by sorry

end x_4_sufficient_not_necessary_l301_30197


namespace greatest_good_set_l301_30121

def is_good (k : ℕ) (S : Set ℕ) : Prop :=
  ∃ (color : ℕ → Fin k),
    ∀ s ∈ S, ∀ x y : ℕ, x + y = s → color x ≠ color y

theorem greatest_good_set (k : ℕ) (h : k > 1) :
  (∀ a : ℕ, is_good k {x | ∃ t, x = a + t ∧ 1 ≤ t ∧ t ≤ 2*k - 1}) ∧
  ¬(∀ a : ℕ, is_good k {x | ∃ t, x = a + t ∧ 1 ≤ t ∧ t ≤ 2*k}) :=
sorry

end greatest_good_set_l301_30121


namespace digits_left_of_264_divisible_by_4_l301_30125

theorem digits_left_of_264_divisible_by_4 : 
  (∀ n : ℕ, n < 10 → (n * 1000 + 264) % 4 = 0) ∧ 
  (∃ (S : Finset ℕ), S.card = 10 ∧ ∀ n ∈ S, n < 10 ∧ (n * 1000 + 264) % 4 = 0) := by
  sorry

end digits_left_of_264_divisible_by_4_l301_30125


namespace units_digit_of_expression_l301_30178

/-- Converts a number from base 7 to base 10 -/
def toBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 7 -/
def toBase7 (n : ℕ) : ℕ := sorry

/-- Returns the units digit of a number in base 7 -/
def unitsDigitBase7 (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem units_digit_of_expression :
  let a := 43
  let b := 124
  let c := 15
  unitsDigitBase7 ((toBase7 (toBase10 a + toBase10 b)) * c) = 6 := by sorry

end units_digit_of_expression_l301_30178


namespace area_of_original_figure_l301_30199

/-- Represents the properties of an oblique diametric view of a figure -/
structure ObliqueView where
  is_isosceles_trapezoid : Bool
  base_angle : ℝ
  leg_length : ℝ
  top_base_length : ℝ

/-- Calculates the area of the original plane figure given its oblique diametric view -/
def original_area (view : ObliqueView) : ℝ :=
  sorry

/-- Theorem stating the area of the original plane figure given specific oblique view properties -/
theorem area_of_original_figure (view : ObliqueView) 
  (h1 : view.is_isosceles_trapezoid = true)
  (h2 : view.base_angle = π / 4)  -- 45° in radians
  (h3 : view.leg_length = 1)
  (h4 : view.top_base_length = 1) :
  original_area view = 2 + Real.sqrt 2 :=
sorry

end area_of_original_figure_l301_30199


namespace cricketer_score_percentage_l301_30122

/-- Calculates the percentage of runs made by running between the wickets -/
def percentage_runs_by_running (total_runs : ℕ) (boundaries : ℕ) (sixes : ℕ) : ℚ :=
  let runs_from_boundaries := boundaries * 4
  let runs_from_sixes := sixes * 6
  let runs_by_running := total_runs - (runs_from_boundaries + runs_from_sixes)
  (runs_by_running : ℚ) / total_runs * 100

/-- Proves that the percentage of runs made by running between the wickets is approximately 60.53% -/
theorem cricketer_score_percentage :
  let result := percentage_runs_by_running 152 12 2
  ∃ ε > 0, |result - 60.53| < ε :=
sorry

end cricketer_score_percentage_l301_30122


namespace coefficient_x2_implies_a_eq_2_l301_30135

/-- The coefficient of x^2 in the expansion of (x+a)^5 -/
def coefficient_x2 (a : ℝ) : ℝ := 10 * a^3

/-- Theorem stating that if the coefficient of x^2 in (x+a)^5 is 80, then a = 2 -/
theorem coefficient_x2_implies_a_eq_2 :
  coefficient_x2 2 = 80 ∧ (∀ a : ℝ, coefficient_x2 a = 80 → a = 2) :=
sorry

end coefficient_x2_implies_a_eq_2_l301_30135


namespace min_sum_of_fractions_l301_30166

def Digits : Finset Nat := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

theorem min_sum_of_fractions (A B C D : Nat) :
  A ∈ Digits → B ∈ Digits → C ∈ Digits → D ∈ Digits →
  A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
  B ≠ 0 → D ≠ 0 →
  (∀ A' B' C' D' : Nat,
    A' ∈ Digits → B' ∈ Digits → C' ∈ Digits → D' ∈ Digits →
    A' ≠ B' → A' ≠ C' → A' ≠ D' → B' ≠ C' → B' ≠ D' → C' ≠ D' →
    B' ≠ 0 → D' ≠ 0 →
    (A : ℚ) / (B : ℚ) + (C : ℚ) / (D : ℚ) ≤ (A' : ℚ) / (B' : ℚ) + (C' : ℚ) / (D' : ℚ)) →
  (A : ℚ) / (B : ℚ) + (C : ℚ) / (D : ℚ) = 1 / 8 := by
sorry

end min_sum_of_fractions_l301_30166


namespace complex_modulus_squared_l301_30170

theorem complex_modulus_squared (z : ℂ) (h : z * Complex.abs z = 3 + 12*I) : Complex.abs z ^ 2 = 3 := by
  sorry

end complex_modulus_squared_l301_30170


namespace total_weight_is_63_l301_30107

/-- The weight of beeswax used in each candle, in ounces -/
def beeswax_weight : ℕ := 8

/-- The weight of coconut oil used in each candle, in ounces -/
def coconut_oil_weight : ℕ := 1

/-- The number of candles Ethan makes -/
def num_candles : ℕ := 10 - 3

/-- The total weight of all candles made by Ethan, in ounces -/
def total_weight : ℕ := num_candles * (beeswax_weight + coconut_oil_weight)

/-- Theorem stating that the total weight of candles is 63 ounces -/
theorem total_weight_is_63 : total_weight = 63 := by
  sorry

end total_weight_is_63_l301_30107


namespace sanda_exercise_days_l301_30147

theorem sanda_exercise_days 
  (javier_daily_minutes : ℕ) 
  (javier_days : ℕ) 
  (sanda_daily_minutes : ℕ) 
  (total_minutes : ℕ) :
  javier_daily_minutes = 50 →
  javier_days = 7 →
  sanda_daily_minutes = 90 →
  total_minutes = 620 →
  (javier_daily_minutes * javier_days + sanda_daily_minutes * (total_minutes - javier_daily_minutes * javier_days) / sanda_daily_minutes = total_minutes) →
  (total_minutes - javier_daily_minutes * javier_days) / sanda_daily_minutes = 3 :=
by sorry

end sanda_exercise_days_l301_30147


namespace justin_jerseys_l301_30127

theorem justin_jerseys (long_sleeve_cost : ℕ) (striped_cost : ℕ) (long_sleeve_count : ℕ) (total_spent : ℕ) :
  long_sleeve_cost = 15 →
  striped_cost = 10 →
  long_sleeve_count = 4 →
  total_spent = 80 →
  (total_spent - long_sleeve_cost * long_sleeve_count) / striped_cost = 2 :=
by sorry

end justin_jerseys_l301_30127


namespace country_club_cost_l301_30184

/-- Calculates the amount one person pays for the first year of a country club membership,
    given they pay half the total cost for a group. -/
theorem country_club_cost
  (num_people : ℕ)
  (joining_fee : ℕ)
  (monthly_cost : ℕ)
  (months_in_year : ℕ)
  (h_num_people : num_people = 4)
  (h_joining_fee : joining_fee = 4000)
  (h_monthly_cost : monthly_cost = 1000)
  (h_months_in_year : months_in_year = 12) :
  (num_people * joining_fee + num_people * monthly_cost * months_in_year) / 2 = 32000 := by
  sorry

#check country_club_cost

end country_club_cost_l301_30184


namespace symmetry_implies_axis_l301_30101

/-- A function g is symmetric about x = 1.5 if g(x) = g(3-x) for all x -/
def SymmetricAbout1_5 (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = g (3 - x)

/-- The line x = 1.5 is an axis of symmetry for g if
    for all points (x, g(x)), the point (3-x, g(x)) is also on the graph of g -/
def IsAxisOfSymmetry1_5 (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = g (3 - x)

theorem symmetry_implies_axis (g : ℝ → ℝ) :
  SymmetricAbout1_5 g → IsAxisOfSymmetry1_5 g :=
by
  sorry

#check symmetry_implies_axis

end symmetry_implies_axis_l301_30101


namespace total_cost_plates_and_spoons_l301_30186

theorem total_cost_plates_and_spoons :
  let num_plates : ℕ := 9
  let price_per_plate : ℚ := 2
  let num_spoons : ℕ := 4
  let price_per_spoon : ℚ := 3/2
  (num_plates : ℚ) * price_per_plate + (num_spoons : ℚ) * price_per_spoon = 24 := by
  sorry

end total_cost_plates_and_spoons_l301_30186


namespace product_inequality_with_sum_constraint_l301_30196

theorem product_inequality_with_sum_constraint (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) 
  (sum_constraint : x + y + z = 1) :
  (1 + 1/x) * (1 + 1/y) * (1 + 1/z) ≥ 64 ∧
  ((1 + 1/x) * (1 + 1/y) * (1 + 1/z) = 64 ↔ x = 1/3 ∧ y = 1/3 ∧ z = 1/3) :=
by sorry

end product_inequality_with_sum_constraint_l301_30196


namespace divisibility_by_1947_l301_30140

theorem divisibility_by_1947 (n : ℕ) (h : Odd n) :
  (46^n + 296 * 13^n) % 1947 = 0 := by
  sorry

end divisibility_by_1947_l301_30140


namespace principal_is_7000_l301_30129

/-- Calculates the simple interest for a given principal, rate, and time -/
def simpleInterest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  principal * rate * time / 100

/-- Represents the financial transaction described in the problem -/
structure Transaction where
  principal : ℚ
  borrowRate : ℚ
  lendRate : ℚ
  time : ℚ
  gainPerYear : ℚ

/-- Theorem stating that given the conditions, the principal is 7000 -/
theorem principal_is_7000 (t : Transaction) 
  (h1 : t.time = 2)
  (h2 : t.borrowRate = 4)
  (h3 : t.lendRate = 6)
  (h4 : t.gainPerYear = 140)
  (h5 : t.gainPerYear = (simpleInterest t.principal t.lendRate t.time - 
                         simpleInterest t.principal t.borrowRate t.time) / t.time) :
  t.principal = 7000 := by
  sorry

end principal_is_7000_l301_30129


namespace domain_of_f_given_range_l301_30144

-- Define the function f
def f (x : ℝ) : ℝ := x + 1

-- Define the theorem
theorem domain_of_f_given_range :
  (∀ y ∈ Set.Ioo 2 3, ∃ x, f x = y) ∧ f 2 = 3 →
  {x : ℝ | ∃ y ∈ Set.Ioo 2 3, f x = y} ∪ {2} = Set.Ioo 1 2 ∪ {2} :=
by sorry

end domain_of_f_given_range_l301_30144


namespace mango_rate_calculation_l301_30155

theorem mango_rate_calculation (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (total_paid : ℕ) :
  grape_quantity = 8 →
  grape_rate = 70 →
  mango_quantity = 8 →
  total_paid = 1000 →
  (total_paid - grape_quantity * grape_rate) / mango_quantity = 55 := by
sorry

end mango_rate_calculation_l301_30155


namespace hyperbola_k_range_l301_30181

def is_hyperbola (k : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / k + y^2 / (k - 3) = 1 ∧ k ≠ 0 ∧ k ≠ 3

theorem hyperbola_k_range :
  ∀ k : ℝ, is_hyperbola k ↔ 0 < k ∧ k < 3 := by sorry

end hyperbola_k_range_l301_30181


namespace geometric_sequence_increasing_iff_second_greater_first_l301_30188

/-- A geometric sequence with positive first term -/
structure GeometricSequence where
  a : ℕ → ℝ
  first_positive : a 1 > 0
  is_geometric : ∀ n : ℕ, n > 0 → ∃ q : ℝ, a (n + 1) = q * a n

/-- An increasing sequence -/
def IsIncreasing (s : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → s (n + 1) > s n

theorem geometric_sequence_increasing_iff_second_greater_first (seq : GeometricSequence) :
  (seq.a 2 > seq.a 1) ↔ IsIncreasing seq.a :=
sorry

end geometric_sequence_increasing_iff_second_greater_first_l301_30188


namespace range_of_m_l301_30108

def f (x : ℝ) := x^2 - 4*x + 5

theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc (-1) m, f x ≤ 10) ∧
  (∃ x ∈ Set.Icc (-1) m, f x = 10) ∧
  (∀ x ∈ Set.Icc (-1) m, f x ≥ 1) ∧
  (∃ x ∈ Set.Icc (-1) m, f x = 1) →
  m ∈ Set.Icc 2 5 :=
sorry

end range_of_m_l301_30108


namespace second_warehouse_more_profitable_l301_30174

/-- Represents the monthly rent in thousands of rubles -/
def monthly_rent_first : ℝ := 80

/-- Represents the monthly rent in thousands of rubles -/
def monthly_rent_second : ℝ := 20

/-- Represents the probability of the bank repossessing the second warehouse -/
def repossession_probability : ℝ := 0.5

/-- Represents the number of months after which repossession might occur -/
def repossession_month : ℕ := 5

/-- Represents the moving expenses in thousands of rubles -/
def moving_expenses : ℝ := 150

/-- Represents the lease duration in months -/
def lease_duration : ℕ := 12

/-- Calculates the expected cost of renting the second warehouse for one year -/
def expected_cost_second : ℝ :=
  let cost_no_repossession := monthly_rent_second * lease_duration
  let cost_repossession := monthly_rent_second * repossession_month +
                           monthly_rent_first * (lease_duration - repossession_month) +
                           moving_expenses
  (1 - repossession_probability) * cost_no_repossession +
  repossession_probability * cost_repossession

/-- Calculates the cost of renting the first warehouse for one year -/
def cost_first : ℝ := monthly_rent_first * lease_duration

theorem second_warehouse_more_profitable :
  expected_cost_second < cost_first :=
sorry

end second_warehouse_more_profitable_l301_30174


namespace triangle_side_length_l301_30195

theorem triangle_side_length (a b c : ℝ) (A : ℝ) :
  a = 1 → c = Real.sqrt 3 → A = π / 6 →
  (a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A) →
  b = 1 ∨ b = 2 := by
sorry

end triangle_side_length_l301_30195


namespace hyperbola_eccentricity_l301_30111

/-- For a hyperbola with equation x²/9 - y²/m = 1 and eccentricity e = 2, m = 27 -/
theorem hyperbola_eccentricity (x y m : ℝ) (e : ℝ) 
  (h1 : x^2 / 9 - y^2 / m = 1)
  (h2 : e = 2)
  (h3 : e = Real.sqrt (1 + m / 9)) : 
  m = 27 := by
  sorry


end hyperbola_eccentricity_l301_30111


namespace dvd_pack_cost_l301_30158

theorem dvd_pack_cost (total_cost : ℝ) (num_packs : ℕ) (h1 : total_cost = 120) (h2 : num_packs = 6) :
  total_cost / num_packs = 20 := by
sorry

end dvd_pack_cost_l301_30158


namespace log_sum_simplification_l301_30177

theorem log_sum_simplification :
  let f (a b : ℝ) := 1 / (Real.log a / Real.log b + 1)
  f 3 12 + f 2 8 + f 7 9 = 1 - Real.log 7 / Real.log 1008 :=
by sorry

end log_sum_simplification_l301_30177


namespace correct_regression_sequence_l301_30136

/-- Represents the steps in linear regression analysis -/
inductive RegressionStep
  | InterpretEquation
  | CollectData
  | CalculateEquation
  | ComputeCorrelation
  | PlotScatterDiagram

/-- Represents a sequence of regression steps -/
def RegressionSequence := List RegressionStep

/-- The correct sequence of regression steps -/
def correctSequence : RegressionSequence := [
  RegressionStep.CollectData,
  RegressionStep.PlotScatterDiagram,
  RegressionStep.ComputeCorrelation,
  RegressionStep.CalculateEquation,
  RegressionStep.InterpretEquation
]

/-- Predicate to check if a sequence is valid for determining linear relationship -/
def isValidSequence (seq : RegressionSequence) : Prop := 
  seq = correctSequence

/-- Theorem stating that the correct sequence is valid for linear regression analysis -/
theorem correct_regression_sequence : 
  isValidSequence correctSequence := by sorry

end correct_regression_sequence_l301_30136


namespace fruit_salad_composition_l301_30167

/-- Fruit salad composition problem -/
theorem fruit_salad_composition (total : ℕ) (b r g c : ℕ) : 
  total = 360 ∧ 
  r = 3 * b ∧ 
  g = 4 * c ∧ 
  c = 5 * r ∧ 
  total = b + r + g + c → 
  c = 68 := by
sorry

end fruit_salad_composition_l301_30167


namespace calculation_proof_l301_30193

theorem calculation_proof :
  ((-11 : ℤ) + 8 + (-4) = -7) ∧
  (-1^2023 - |1 - (1/3 : ℚ)| * (-3/2)^2 = -5/2) := by
  sorry

end calculation_proof_l301_30193


namespace more_science_than_math_books_l301_30194

def total_budget : ℕ := 500
def math_books : ℕ := 4
def math_book_price : ℕ := 20
def science_book_price : ℕ := 10
def art_book_price : ℕ := 20
def music_book_cost : ℕ := 160

theorem more_science_than_math_books :
  ∃ (science_books : ℕ) (art_books : ℕ),
    science_books > math_books ∧
    art_books = 2 * math_books ∧
    total_budget = math_books * math_book_price + science_books * science_book_price + 
                   art_books * art_book_price + music_book_cost ∧
    science_books - math_books = 6 :=
by sorry

end more_science_than_math_books_l301_30194


namespace investment_ratio_l301_30148

/-- Prove that the investment ratio between A and C is 3:1 --/
theorem investment_ratio (a b c : ℕ) (total_profit c_profit : ℕ) : 
  a = 3 * b → -- A and B invested in a ratio of 3:1
  total_profit = 60000 → -- The total profit was 60000
  c_profit = 20000 → -- C received 20000 from the profit
  3 * c = a := by
  sorry

end investment_ratio_l301_30148


namespace complex_sum_of_parts_l301_30179

theorem complex_sum_of_parts (a b : ℝ) (i : ℂ) (h : i * i = -1) 
  (h1 : (1 : ℂ) + 2 * i = a + b * i) : a + b = 3 := by
  sorry

end complex_sum_of_parts_l301_30179


namespace subset_range_l301_30137

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x - 15 ≤ 0}

-- Define set B parameterized by m
def B (m : ℝ) : Set ℝ := {x | m - 2 < x ∧ x < 2*m - 3}

-- Theorem statement
theorem subset_range (m : ℝ) : B m ⊆ A ↔ m ≤ 4 :=
sorry

end subset_range_l301_30137


namespace imaginary_part_of_complex_fraction_l301_30175

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := (5 - I) / (1 - I)
  Complex.im z = 2 := by sorry

end imaginary_part_of_complex_fraction_l301_30175


namespace inverse_point_theorem_l301_30176

-- Define a function f and its inverse
variable (f : ℝ → ℝ)
variable (f_inv : ℝ → ℝ)

-- State that f_inv is the inverse of f
axiom inverse_relation : ∀ x, f_inv (f x) = x ∧ f (f_inv x) = x

-- Define the condition that f(1) + 1 = 2
axiom condition : f 1 + 1 = 2

-- Theorem to prove
theorem inverse_point_theorem : f_inv 1 - 1 = 0 := by
  sorry

end inverse_point_theorem_l301_30176


namespace range_of_z_l301_30134

-- Define the variables and their constraints
def a : ℝ := sorry
def b : ℝ := sorry

-- Define the function z
def z (a b : ℝ) : ℝ := 2 * a - b

-- State the theorem
theorem range_of_z :
  (2 < a ∧ a < 3) → (-2 < b ∧ b < -1) →
  ∀ z₀ : ℝ, (∃ a₀ b₀ : ℝ, (2 < a₀ ∧ a₀ < 3) ∧ (-2 < b₀ ∧ b₀ < -1) ∧ z a₀ b₀ = z₀) ↔ (5 < z₀ ∧ z₀ < 8) :=
by sorry

end range_of_z_l301_30134


namespace water_needed_for_recipe_l301_30116

/-- Represents the ratio of ingredients in the fruit punch recipe -/
structure PunchRatio where
  water : ℕ
  orange : ℕ
  lemon : ℕ

/-- Calculates the amount of water needed for a given punch recipe and total volume -/
def water_needed (ratio : PunchRatio) (total_gallons : ℚ) (quarts_per_gallon : ℕ) : ℚ :=
  let total_parts := ratio.water + ratio.orange + ratio.lemon
  let water_fraction := ratio.water / total_parts
  water_fraction * total_gallons * quarts_per_gallon

/-- Proves that the amount of water needed for the given recipe and volume is 15/2 quarts -/
theorem water_needed_for_recipe : 
  let recipe := PunchRatio.mk 5 2 1
  let total_gallons := 3
  let quarts_per_gallon := 4
  water_needed recipe total_gallons quarts_per_gallon = 15/2 := by
  sorry


end water_needed_for_recipe_l301_30116


namespace wall_photo_area_l301_30120

theorem wall_photo_area (paper_width paper_length frame_width : ℕ) 
  (hw : paper_width = 8)
  (hl : paper_length = 12)
  (hf : frame_width = 2) : 
  (paper_width + 2 * frame_width) * (paper_length + 2 * frame_width) = 192 := by
  sorry

end wall_photo_area_l301_30120


namespace square_area_difference_l301_30119

theorem square_area_difference : 
  ∀ (smaller_length greater_length : ℝ),
    greater_length = 7 →
    greater_length = smaller_length + 2 →
    (greater_length ^ 2 - smaller_length ^ 2 : ℝ) = 24 := by
  sorry

end square_area_difference_l301_30119


namespace equation_solutions_count_l301_30185

theorem equation_solutions_count : 
  ∃! n : ℕ, n = (Finset.filter 
    (λ (p : ℕ × ℕ) => (p.1 - 4)^2 - 35 = (p.2 - 3)^2) 
    (Finset.product (Finset.range 1000) (Finset.range 1000))).card ∧ n = 3 :=
by sorry

end equation_solutions_count_l301_30185


namespace intersection_sum_l301_30133

/-- Given two lines y = nx + 5 and y = 4x + c that intersect at (8, 9),
    prove that n + c = -22.5 -/
theorem intersection_sum (n c : ℝ) : 
  (∀ x y : ℝ, y = n * x + 5 ∨ y = 4 * x + c) →
  9 = n * 8 + 5 →
  9 = 4 * 8 + c →
  n + c = -22.5 := by
  sorry

end intersection_sum_l301_30133


namespace symmetry_across_origin_l301_30191

/-- Given two points A and B in a 2D plane, where B is symmetrical to A with respect to the origin,
    this theorem proves that if A has coordinates (2, -6), then B has coordinates (-2, 6). -/
theorem symmetry_across_origin (A B : ℝ × ℝ) :
  A = (2, -6) → B = (-A.1, -A.2) → B = (-2, 6) := by
  sorry

end symmetry_across_origin_l301_30191


namespace symmetry_line_probability_l301_30118

/-- Represents a point on a grid --/
structure GridPoint where
  x : Nat
  y : Nat

/-- Represents a rectangle with a uniform grid --/
structure GridRectangle where
  width : Nat
  height : Nat

/-- The total number of points in the grid rectangle --/
def totalPoints (rect : GridRectangle) : Nat :=
  rect.width * rect.height

/-- The center point of the rectangle --/
def centerPoint (rect : GridRectangle) : GridPoint :=
  { x := rect.width / 2, y := rect.height / 2 }

/-- Checks if a given point is on a line of symmetry --/
def isOnSymmetryLine (p : GridPoint) (center : GridPoint) (rect : GridRectangle) : Bool :=
  p.x = center.x ∨ p.y = center.y

/-- Counts the number of points on lines of symmetry, excluding the center --/
def countSymmetryPoints (rect : GridRectangle) : Nat :=
  rect.width + rect.height - 2

/-- The main theorem --/
theorem symmetry_line_probability (rect : GridRectangle) : 
  rect.width = 10 ∧ rect.height = 10 →
  (countSymmetryPoints rect : Rat) / ((totalPoints rect - 1 : Nat) : Rat) = 2 / 11 := by
  sorry


end symmetry_line_probability_l301_30118


namespace banana_difference_l301_30103

theorem banana_difference (total : ℕ) (lydia_bananas : ℕ) (donna_bananas : ℕ)
  (h1 : total = 200)
  (h2 : lydia_bananas = 60)
  (h3 : donna_bananas = 40) :
  total - donna_bananas - lydia_bananas - lydia_bananas = 40 := by
  sorry

end banana_difference_l301_30103


namespace ellipse_a_plus_k_l301_30172

/-- An ellipse with given properties -/
structure Ellipse where
  -- Foci coordinates
  f1 : ℝ × ℝ := (1, 1)
  f2 : ℝ × ℝ := (1, 3)
  -- Point on the ellipse
  p : ℝ × ℝ := (-4, 2)
  -- Constants in the ellipse equation
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ
  -- Positivity of a and b
  a_pos : a > 0
  b_pos : b > 0
  -- Ellipse equation
  eq : ∀ (x y : ℝ), (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1 ↔ (x, y) ∈ {p : ℝ × ℝ | (p.1 - f1.1)^2 + (p.2 - f1.2)^2 + (p.1 - f2.1)^2 + (p.2 - f2.2)^2 = (2*a)^2}

/-- Theorem: For the given ellipse, a + k = 7 -/
theorem ellipse_a_plus_k (e : Ellipse) : e.a + e.k = 7 := by sorry

end ellipse_a_plus_k_l301_30172


namespace expression_value_at_nine_l301_30165

theorem expression_value_at_nine :
  let x : ℝ := 9
  let f (x : ℝ) := (x^9 - 27*x^6 + 19683) / (x^6 - 27)
  f x = 492804 := by
sorry

end expression_value_at_nine_l301_30165


namespace murtha_pebble_collection_l301_30180

/-- Sum of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem murtha_pebble_collection :
  arithmetic_sum 1 1 15 = 120 := by
  sorry

end murtha_pebble_collection_l301_30180


namespace triangle_tangent_sum_l301_30159

theorem triangle_tangent_sum (A B C : Real) : 
  A + B + C = π →  -- angle sum property of triangle
  A + C = 2 * B →  -- given condition
  Real.tan (A / 2) + Real.tan (C / 2) + Real.sqrt 3 * Real.tan (A / 2) * Real.tan (C / 2) = Real.sqrt 3 := by
  sorry

end triangle_tangent_sum_l301_30159


namespace football_team_right_handed_players_l301_30141

theorem football_team_right_handed_players
  (total_players : ℕ)
  (throwers : ℕ)
  (h1 : total_players = 70)
  (h2 : throwers = 28)
  (h3 : throwers ≤ total_players)
  (h4 : (total_players - throwers) % 3 = 0) -- Ensures non-throwers can be divided into thirds
  : (throwers + ((total_players - throwers) * 2) / 3) = 56 := by
  sorry

end football_team_right_handed_players_l301_30141


namespace complex_magnitude_l301_30152

theorem complex_magnitude (z : ℂ) : (1 - 2*Complex.I)*z = 3 + 4*Complex.I → Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_magnitude_l301_30152


namespace parallel_vectors_magnitude_l301_30198

/-- Given two vectors a and b in ℝ², where a is parallel to b, 
    prove that the magnitude of 2a + 3b is 4√5. -/
theorem parallel_vectors_magnitude (m : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![-2, m]
  (∃ (k : ℝ), a = k • b) →
  ‖(2 : ℝ) • a + (3 : ℝ) • b‖ = 4 * Real.sqrt 5 := by
sorry

end parallel_vectors_magnitude_l301_30198


namespace fraction_power_simplification_l301_30153

theorem fraction_power_simplification :
  (77777 : ℕ) = 7 * 11111 →
  (77777 ^ 6 : ℕ) / (11111 ^ 6) = 117649 := by
  sorry

end fraction_power_simplification_l301_30153
