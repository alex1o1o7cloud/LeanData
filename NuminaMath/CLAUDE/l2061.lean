import Mathlib

namespace complex_number_validity_one_plus_i_is_valid_l2061_206176

theorem complex_number_validity : Complex → Prop :=
  fun z => ∃ (a b : ℝ), z = Complex.mk a b

theorem one_plus_i_is_valid : complex_number_validity (1 + Complex.I) := by
  sorry

end complex_number_validity_one_plus_i_is_valid_l2061_206176


namespace solve_farmer_problem_l2061_206164

def farmer_problem (total_cattle : ℕ) (male_percentage : ℚ) (male_count : ℕ) (total_milk : ℚ) : Prop :=
  let female_percentage : ℚ := 1 - male_percentage
  let female_count : ℕ := total_cattle - male_count
  let milk_per_female : ℚ := total_milk / female_count
  (male_percentage * total_cattle = male_count) ∧
  (female_percentage * total_cattle = female_count) ∧
  (milk_per_female = 2)

theorem solve_farmer_problem :
  ∃ (total_cattle : ℕ),
    farmer_problem total_cattle (2/5) 50 150 :=
by
  sorry

end solve_farmer_problem_l2061_206164


namespace lecture_hall_tables_l2061_206122

theorem lecture_hall_tables (total_legs : ℕ) (stools_per_table : ℕ) (stool_legs : ℕ) (table_legs : ℕ) :
  total_legs = 680 →
  stools_per_table = 8 →
  stool_legs = 4 →
  table_legs = 4 →
  (total_legs : ℚ) / ((stools_per_table * stool_legs + table_legs) : ℚ) = 680 / 36 :=
by sorry

end lecture_hall_tables_l2061_206122


namespace some_number_value_l2061_206107

theorem some_number_value : ∃ (some_number : ℝ), 
  |5 - 8 * (3 - some_number)| - |5 - 11| = 71 ∧ some_number = 12 := by
  sorry

end some_number_value_l2061_206107


namespace unique_x_with_three_prime_factors_l2061_206183

theorem unique_x_with_three_prime_factors (x n : ℕ) : 
  x = 7^n + 1 →
  Odd n →
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ p ≠ 11 ∧ q ≠ 11 ∧ x = 2 * 11 * p * q) →
  x = 16808 :=
by sorry

end unique_x_with_three_prime_factors_l2061_206183


namespace stating_count_paths_correct_l2061_206153

/-- 
Counts the number of paths from (0,0) to (m,n) where m < n and 
at every intermediate point (a,b), a < b.
-/
def count_paths (m n : ℕ) : ℕ :=
  if m < n then
    (Nat.factorial (m + n - 1) * (n - m)) / (Nat.factorial m * Nat.factorial n)
  else 0

/-- 
Theorem stating that count_paths gives the correct number of paths
from (0,0) to (m,n) satisfying the given conditions.
-/
theorem count_paths_correct (m n : ℕ) (h : m < n) :
  count_paths m n = ((Nat.factorial (m + n - 1) * (n - m)) / (Nat.factorial m * Nat.factorial n)) :=
by sorry

end stating_count_paths_correct_l2061_206153


namespace dalton_savings_l2061_206163

def jump_rope_cost : ℕ := 7
def board_game_cost : ℕ := 12
def playground_ball_cost : ℕ := 4
def uncle_money : ℕ := 13
def additional_money_needed : ℕ := 4

def total_cost : ℕ := jump_rope_cost + board_game_cost + playground_ball_cost

theorem dalton_savings (savings : ℕ) : 
  savings = total_cost - (uncle_money + additional_money_needed) := by
  sorry

end dalton_savings_l2061_206163


namespace tara_book_sales_l2061_206193

/-- Calculates the total number of books Tara needs to sell to buy a new clarinet and an accessory, given initial savings, clarinet cost, book price, and additional accessory cost. -/
def total_books_sold (initial_savings : ℕ) (clarinet_cost : ℕ) (book_price : ℕ) (accessory_cost : ℕ) : ℕ :=
  let initial_goal := clarinet_cost - initial_savings
  let halfway_books := (initial_goal / 2) / book_price
  let final_goal := initial_goal + accessory_cost
  let final_books := final_goal / book_price
  halfway_books + final_books

/-- Theorem stating that Tara needs to sell 28 books in total to reach her goal. -/
theorem tara_book_sales : total_books_sold 10 90 5 20 = 28 := by
  sorry

end tara_book_sales_l2061_206193


namespace line_and_circle_equations_l2061_206112

-- Define the line l
def line_l (x y : ℝ) : Prop := x - 2*y = 0

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 1

-- Theorem statement
theorem line_and_circle_equations :
  ∀ (x y : ℝ),
  (∃ (t : ℝ), x = 2 + 4*t ∧ y = 1 + 2*t) →  -- Line l passes through (2, 1) and (6, 3)
  (∃ (a : ℝ), line_l (2*a) a ∧ circle_C (2*a) a) →  -- Circle C's center lies on line l
  circle_C 2 0 →  -- Circle C is tangent to x-axis at (2, 0)
  (line_l x y ↔ x - 2*y = 0) ∧  -- Equation of line l
  (circle_C x y ↔ (x - 2)^2 + (y - 1)^2 = 1)  -- Equation of circle C
  := by sorry

end line_and_circle_equations_l2061_206112


namespace log_stack_total_l2061_206113

/-- The sum of an arithmetic sequence with 15 terms, starting at 15 and ending at 1 -/
def log_stack_sum : ℕ := 
  let first_term := 15
  let last_term := 1
  let num_terms := 15
  (num_terms * (first_term + last_term)) / 2

/-- The total number of logs in the stack is 120 -/
theorem log_stack_total : log_stack_sum = 120 := by
  sorry

end log_stack_total_l2061_206113


namespace f_equals_g_l2061_206110

def f (x : ℝ) : ℝ := x^2 - 2*x - 1
def g (t : ℝ) : ℝ := t^2 - 2*t - 1

theorem f_equals_g : f = g := by sorry

end f_equals_g_l2061_206110


namespace mo_tea_consumption_l2061_206173

/-- Represents the drinking habits of Mo --/
structure MoDrinkingHabits where
  n : ℕ  -- number of hot chocolate cups on rainy days
  t : ℕ  -- number of tea cups on non-rainy days
  total_cups : ℕ  -- total cups drunk in a week
  tea_chocolate_diff : ℕ  -- difference between tea and hot chocolate cups
  rainy_days : ℕ  -- number of rainy days in a week

/-- Theorem stating Mo's tea consumption on non-rainy days --/
theorem mo_tea_consumption (mo : MoDrinkingHabits) 
  (h1 : mo.total_cups = 36)
  (h2 : mo.tea_chocolate_diff = 14)
  (h3 : mo.rainy_days = 2)
  (h4 : mo.rainy_days * mo.n + (7 - mo.rainy_days) * mo.t = mo.total_cups)
  (h5 : (7 - mo.rainy_days) * mo.t = mo.rainy_days * mo.n + mo.tea_chocolate_diff) :
  mo.t = 5 := by
  sorry

#check mo_tea_consumption

end mo_tea_consumption_l2061_206173


namespace orangeade_price_day1_l2061_206161

/-- Represents the price and volume data for orangeade sales over two days -/
structure OrangeadeSales where
  orange_juice : ℝ
  water_day1 : ℝ
  water_day2 : ℝ
  price_day2 : ℝ
  revenue : ℝ

/-- Calculates the price per glass on the first day given orangeade sales data -/
def price_day1 (sales : OrangeadeSales) : ℝ :=
  1.5 * sales.price_day2

/-- Theorem stating that under the given conditions, the price on the first day is $0.30 -/
theorem orangeade_price_day1 (sales : OrangeadeSales) 
  (h1 : sales.water_day1 = sales.orange_juice)
  (h2 : sales.water_day2 = 2 * sales.water_day1)
  (h3 : sales.price_day2 = 0.2)
  (h4 : sales.revenue = (sales.orange_juice + sales.water_day1) * (price_day1 sales))
  (h5 : sales.revenue = (sales.orange_juice + sales.water_day2) * sales.price_day2) :
  price_day1 sales = 0.3 := by
  sorry

#eval price_day1 { orange_juice := 1, water_day1 := 1, water_day2 := 2, price_day2 := 0.2, revenue := 0.6 }

end orangeade_price_day1_l2061_206161


namespace fraction_inequality_l2061_206121

theorem fraction_inequality (a b m : ℝ) (ha : 0 < a) (hb : 0 < b) (hm : 0 < m) (hab : a < b) :
  a / b < (a + m) / (b + m) := by
  sorry

end fraction_inequality_l2061_206121


namespace expected_worth_unfair_coin_l2061_206168

/-- The expected worth of an unfair coin flip -/
theorem expected_worth_unfair_coin : 
  let p_heads : ℚ := 2/3
  let p_tails : ℚ := 1/3
  let gain_heads : ℤ := 5
  let loss_tails : ℤ := 6
  p_heads * gain_heads - p_tails * loss_tails = 4/3 := by
  sorry

end expected_worth_unfair_coin_l2061_206168


namespace inequality_system_integer_solutions_l2061_206199

theorem inequality_system_integer_solutions (x : ℤ) : 
  (2 * (1 - x) ≤ 4 ∧ x - 4 < (x - 8) / 3) ↔ (x = -1 ∨ x = 0 ∨ x = 1) :=
by sorry

end inequality_system_integer_solutions_l2061_206199


namespace unique_dissection_solution_l2061_206177

/-- Represents a square dissection into four-cell and five-cell figures -/
structure SquareDissection where
  size : ℕ
  four_cell_count : ℕ
  five_cell_count : ℕ

/-- Checks if a given dissection is valid for a square of size 6 -/
def is_valid_dissection (d : SquareDissection) : Prop :=
  d.size = 6 ∧ 
  d.four_cell_count > 0 ∧ 
  d.five_cell_count > 0 ∧
  d.size * d.size = 4 * d.four_cell_count + 5 * d.five_cell_count

/-- The unique solution to the square dissection problem -/
def unique_solution : SquareDissection :=
  { size := 6
    four_cell_count := 4
    five_cell_count := 4 }

/-- Theorem stating that the unique solution is the only valid dissection -/
theorem unique_dissection_solution :
  ∀ d : SquareDissection, is_valid_dissection d ↔ d = unique_solution :=
by sorry


end unique_dissection_solution_l2061_206177


namespace volume_difference_equals_negative_thirteen_l2061_206175

/-- A rectangular prism with given face perimeters -/
structure RectangularPrism where
  p1 : ℝ
  p2 : ℝ
  p3 : ℝ

/-- Calculate the volume of a rectangular prism given its face perimeters -/
def volume (prism : RectangularPrism) : ℝ := sorry

/-- The difference in volumes between two rectangular prisms -/
def volumeDifference (a b : RectangularPrism) : ℝ :=
  volume a - volume b

theorem volume_difference_equals_negative_thirteen :
  let a : RectangularPrism := ⟨12, 16, 24⟩
  let b : RectangularPrism := ⟨12, 16, 20⟩
  volumeDifference a b = -13 := by sorry

end volume_difference_equals_negative_thirteen_l2061_206175


namespace dark_tile_fraction_is_16_81_l2061_206119

/-- Represents a tiling system with darker tiles along diagonals -/
structure TilingSystem where
  size : Nat
  corner_size : Nat
  dark_tiles_per_corner : Nat

/-- The fraction of darker tiles in the entire floor -/
def dark_tile_fraction (ts : TilingSystem) : Rat :=
  (4 * ts.dark_tiles_per_corner : Rat) / (ts.size^2 : Rat)

/-- The specific tiling system described in the problem -/
def floor_tiling : TilingSystem :=
  { size := 9
  , corner_size := 4
  , dark_tiles_per_corner := 4 }

/-- Theorem: The fraction of darker tiles in the floor is 16/81 -/
theorem dark_tile_fraction_is_16_81 :
  dark_tile_fraction floor_tiling = 16 / 81 := by
  sorry

end dark_tile_fraction_is_16_81_l2061_206119


namespace find_number_given_hcf_lcm_l2061_206129

/-- Given two positive integers with specific HCF and LCM, prove that one is 24 if the other is 169 -/
theorem find_number_given_hcf_lcm (A B : ℕ+) : 
  (Nat.gcd A B = 13) →
  (Nat.lcm A B = 312) →
  (B = 169) →
  A = 24 := by
sorry

end find_number_given_hcf_lcm_l2061_206129


namespace function_and_range_proof_l2061_206120

-- Define the function f
def f (x : ℝ) (b c : ℝ) : ℝ := 2 * x^2 + b * x + c

-- State the theorem
theorem function_and_range_proof :
  ∀ b c : ℝ,
  (∀ x : ℝ, f x b c < 0 ↔ 1 < x ∧ x < 5) →
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → ∃ t : ℝ, f x b c ≤ 2 + t) →
  (∀ x : ℝ, f x b c = 2 * x^2 - 12 * x + 10) ∧
  (∀ t : ℝ, t ≥ -10 ↔ ∃ x : ℝ, 1 ≤ x ∧ x ≤ 3 ∧ f x b c ≤ 2 + t) :=
by sorry

end function_and_range_proof_l2061_206120


namespace expand_product_l2061_206125

theorem expand_product (x : ℝ) : 
  (2 * x^3 - 3 * x + 1) * (x^2 + 4 * x + 3) = 
  2 * x^5 + 8 * x^4 + 3 * x^3 - 11 * x^2 - 5 * x + 3 := by
sorry

end expand_product_l2061_206125


namespace sprint_tournament_races_l2061_206109

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

end sprint_tournament_races_l2061_206109


namespace inequality_conditions_l2061_206162

theorem inequality_conditions (a b c : ℝ) 
  (h : ∀ (x y z : ℝ), a * (x - y) * (x - z) + b * (y - x) * (y - z) + c * (z - x) * (z - y) ≥ 0) : 
  (-a + 2*b + 2*c ≥ 0) ∧ (2*a - b + 2*c ≥ 0) ∧ (2*a + 2*b - c ≥ 0) := by
  sorry

end inequality_conditions_l2061_206162


namespace or_false_sufficient_not_necessary_for_and_false_l2061_206146

theorem or_false_sufficient_not_necessary_for_and_false (p q : Prop) :
  (¬(p ∨ q) → ¬(p ∧ q)) ∧ ¬(¬(p ∧ q) → ¬(p ∨ q)) :=
sorry

end or_false_sufficient_not_necessary_for_and_false_l2061_206146


namespace x_value_l2061_206184

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {1, 3, 9}

theorem x_value (x : ℕ) (h1 : x ∈ A) (h2 : x ∉ B) : x = 2 := by
  sorry

end x_value_l2061_206184


namespace desired_interest_percentage_l2061_206115

/-- Calculates the desired interest percentage for a share investment. -/
theorem desired_interest_percentage
  (face_value : ℝ)
  (dividend_rate : ℝ)
  (market_value : ℝ)
  (h1 : face_value = 20)
  (h2 : dividend_rate = 0.09)
  (h3 : market_value = 15) :
  (dividend_rate * face_value) / market_value = 0.12 := by
  sorry

end desired_interest_percentage_l2061_206115


namespace smallest_N_is_255_l2061_206140

/-- Represents a team in the basketball championship --/
structure Team where
  id : ℕ
  isCalifornian : Bool
  wins : ℕ

/-- Represents the basketball championship --/
structure Championship where
  N : ℕ
  teams : Finset Team
  games : Finset (Team × Team)

/-- The conditions of the championship --/
def ChampionshipConditions (c : Championship) : Prop :=
  -- Total number of teams is 5N
  c.teams.card = 5 * c.N
  -- Every two teams played exactly one game
  ∧ c.games.card = (c.teams.card * (c.teams.card - 1)) / 2
  -- 251 teams are from California
  ∧ (c.teams.filter (λ t => t.isCalifornian)).card = 251
  -- Alcatraz is a Californian team
  ∧ ∃ alcatraz ∈ c.teams, alcatraz.isCalifornian
    -- Alcatraz is the unique Californian champion
    ∧ ∀ t ∈ c.teams, t.isCalifornian → t.wins ≤ alcatraz.wins
    -- Alcatraz is the unique loser of the tournament
    ∧ ∀ t ∈ c.teams, t.id ≠ alcatraz.id → alcatraz.wins < t.wins

/-- The theorem stating that the smallest possible value of N is 255 --/
theorem smallest_N_is_255 :
  ∀ c : Championship, ChampionshipConditions c → c.N ≥ 255 :=
sorry

end smallest_N_is_255_l2061_206140


namespace simplify_expression_l2061_206117

theorem simplify_expression : 20 + (-14) - (-18) + 13 = 37 := by
  sorry

end simplify_expression_l2061_206117


namespace solution_set_equivalence_l2061_206106

-- Define the set of real numbers greater than 1
def greater_than_one : Set ℝ := {x | x > 1}

-- Define the solution set of ax - 1 > 0
def solution_set_linear (a : ℝ) : Set ℝ := {x | a * x - 1 > 0}

-- Define the solution set of (ax - 1)(x + 2) ≥ 0
def solution_set_quadratic (a : ℝ) : Set ℝ := {x | (a * x - 1) * (x + 2) ≥ 0}

-- Define the set (-∞, -2] ∪ [1, +∞)
def target_set : Set ℝ := {x | x ≤ -2 ∨ x ≥ 1}

theorem solution_set_equivalence (a : ℝ) : 
  solution_set_linear a = greater_than_one → solution_set_quadratic a = target_set := by
  sorry

end solution_set_equivalence_l2061_206106


namespace anna_initial_stamps_l2061_206101

theorem anna_initial_stamps (x : ℕ) (alison_stamps : ℕ) : 
  alison_stamps = 28 → 
  x + alison_stamps / 2 = 50 → 
  x = 36 := by
sorry

end anna_initial_stamps_l2061_206101


namespace perpendicular_planes_not_necessarily_parallel_l2061_206194

/-- Represents a plane in 3D space -/
structure Plane3D where
  -- Add necessary fields for a plane

/-- Represents a line in 3D space -/
structure Line3D where
  -- Add necessary fields for a line

/-- Two planes are perpendicular -/
def perpendicular (p1 p2 : Plane3D) : Prop := sorry

/-- Two planes are parallel -/
def parallel (p1 p2 : Plane3D) : Prop := sorry

/-- The statement that if two planes are perpendicular to a third plane, they are parallel to each other is false -/
theorem perpendicular_planes_not_necessarily_parallel (α β γ : Plane3D) :
  ¬(∀ α β γ : Plane3D, perpendicular α β → perpendicular β γ → parallel α γ) := by
  sorry

#check perpendicular_planes_not_necessarily_parallel

end perpendicular_planes_not_necessarily_parallel_l2061_206194


namespace reciprocal_of_negative_2023_l2061_206102

theorem reciprocal_of_negative_2023 : ((-2023)⁻¹ : ℚ) = -1 / 2023 := by sorry

end reciprocal_of_negative_2023_l2061_206102


namespace central_angle_twice_inscribed_l2061_206135

/-- A circle with a diameter and a point on its circumference -/
structure CircleWithDiameterAndPoint where
  /-- The center of the circle -/
  O : ℝ × ℝ
  /-- One end of the diameter -/
  A : ℝ × ℝ
  /-- The other end of the diameter -/
  B : ℝ × ℝ
  /-- An arbitrary point on the circle -/
  C : ℝ × ℝ
  /-- AB is a diameter -/
  diameter : dist O A = dist O B
  /-- C is on the circle -/
  on_circle : dist O C = dist O A

/-- The angle between two vectors -/
def angle (v w : ℝ × ℝ) : ℝ := sorry

/-- The theorem: Central angle COB is twice the inscribed angle CAB -/
theorem central_angle_twice_inscribed 
  (circle : CircleWithDiameterAndPoint) : 
  angle (circle.C - circle.O) (circle.B - circle.O) = 
  2 * angle (circle.C - circle.A) (circle.B - circle.A) := by
  sorry

end central_angle_twice_inscribed_l2061_206135


namespace triangle_theorem_l2061_206137

noncomputable def triangle_proof (a b c : ℝ) (A B C : ℝ) : Prop :=
  let S := (33 : ℝ) / 2
  3 * a = 5 * c * Real.sin A ∧
  Real.cos B = -(5 : ℝ) / 13 ∧
  S = (1 / 2) * a * c * Real.sin B →
  Real.sin A = (33 : ℝ) / 65 ∧
  b = 10

theorem triangle_theorem :
  ∀ (a b c : ℝ) (A B C : ℝ),
  triangle_proof a b c A B C :=
sorry

end triangle_theorem_l2061_206137


namespace parallel_and_perpendicular_relations_l2061_206126

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (line_parallel_to_plane : Line → Plane → Prop)
variable (line_perpendicular_to_plane : Line → Plane → Prop)

-- Define the lines and planes
variable (m n : Line)
variable (α β γ : Plane)

-- State that m and n are different lines
variable (m_ne_n : m ≠ n)

-- State that α, β, and γ are different planes
variable (α_ne_β : α ≠ β)
variable (α_ne_γ : α ≠ γ)
variable (β_ne_γ : β ≠ γ)

-- Define the theorem
theorem parallel_and_perpendicular_relations :
  (∀ (a b c : Plane), parallel_planes a c → parallel_planes b c → parallel_planes a b) ∧
  (∀ (l1 l2 : Line) (p1 p2 : Plane), 
    line_perpendicular_to_plane l1 p1 → 
    line_perpendicular_to_plane l2 p2 → 
    parallel_planes p1 p2 → 
    parallel_lines l1 l2) :=
sorry

end parallel_and_perpendicular_relations_l2061_206126


namespace fraction_equality_l2061_206189

theorem fraction_equality (a b c d e : ℝ) 
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 4)
  (h3 : c / d = 7)
  (h4 : d / e = 1 / 2) :
  e / a = 8 / 35 := by
  sorry

end fraction_equality_l2061_206189


namespace fly_distance_l2061_206185

/-- The distance flown by a fly between two approaching pedestrians -/
theorem fly_distance (d : ℝ) (v_ped : ℝ) (v_fly : ℝ) (h1 : d > 0) (h2 : v_ped > 0) (h3 : v_fly > 0) :
  let t := d / (2 * v_ped)
  v_fly * t = v_fly * d / (2 * v_ped) := by sorry

#check fly_distance

end fly_distance_l2061_206185


namespace jenny_investment_l2061_206170

/-- Jenny's investment problem -/
theorem jenny_investment (total : ℝ) (real_estate : ℝ) (mutual_funds : ℝ) 
  (h1 : total = 200000)
  (h2 : real_estate = 3 * mutual_funds)
  (h3 : total = real_estate + mutual_funds) :
  real_estate = 150000 := by
  sorry

end jenny_investment_l2061_206170


namespace equilibrium_force_l2061_206156

/-- A 2D vector representation --/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Addition of two 2D vectors --/
def Vector2D.add (v w : Vector2D) : Vector2D :=
  ⟨v.x + w.x, v.y + w.y⟩

/-- Negation of a 2D vector --/
def Vector2D.neg (v : Vector2D) : Vector2D :=
  ⟨-v.x, -v.y⟩

/-- Zero 2D vector --/
def Vector2D.zero : Vector2D :=
  ⟨0, 0⟩

theorem equilibrium_force (f₁ f₂ f₃ f₄ : Vector2D) 
    (h₁ : f₁ = ⟨-2, -1⟩) 
    (h₂ : f₂ = ⟨-3, 2⟩)
    (h₃ : f₃ = ⟨4, -3⟩)
    (h₄ : f₄ = ⟨1, 2⟩) :
    Vector2D.add (Vector2D.add (Vector2D.add f₁ f₂) f₃) f₄ = Vector2D.zero := by
  sorry

#check equilibrium_force

end equilibrium_force_l2061_206156


namespace min_squares_128_343_l2061_206154

/-- Represents a rectangle with height and width -/
structure Rectangle where
  height : ℕ
  width : ℕ

/-- Represents a polyomino spanning a rectangle -/
def SpanningPolyomino (r : Rectangle) : Type := Unit

/-- The number of unit squares in a spanning polyomino -/
def num_squares (r : Rectangle) (p : SpanningPolyomino r) : ℕ := sorry

/-- The minimum number of unit squares in any spanning polyomino for a given rectangle -/
def min_spanning_squares (r : Rectangle) : ℕ := sorry

/-- Theorem: The minimum number of unit squares in a spanning polyomino for a 128-by-343 rectangle is 470 -/
theorem min_squares_128_343 :
  let r : Rectangle := { height := 128, width := 343 }
  min_spanning_squares r = 470 := by sorry

end min_squares_128_343_l2061_206154


namespace advertising_customers_l2061_206128

/-- Proves that the number of customers brought to a site by advertising is 100,
    given the cost of advertising, purchase rate, item cost, and profit. -/
theorem advertising_customers (ad_cost profit item_cost : ℝ) (purchase_rate : ℝ) :
  ad_cost = 1000 →
  profit = 1000 →
  item_cost = 25 →
  purchase_rate = 0.8 →
  ∃ (num_customers : ℕ), 
    (↑num_customers : ℝ) * purchase_rate * item_cost = ad_cost + profit ∧
    num_customers = 100 :=
by sorry

end advertising_customers_l2061_206128


namespace quadratic_no_real_roots_l2061_206171

theorem quadratic_no_real_roots (m : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x + m ≠ 0) ↔ m > 1 := by
  sorry

end quadratic_no_real_roots_l2061_206171


namespace probability_power_of_two_four_digit_l2061_206111

/-- A four-digit number is a natural number between 1000 and 9999, inclusive. -/
def FourDigitNumber (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- A number is a power of 2 if its base-2 logarithm is an integer. -/
def IsPowerOfTwo (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

/-- The count of four-digit numbers that are powers of 2. -/
def CountPowersOfTwoFourDigit : ℕ := 4

/-- The total count of four-digit numbers. -/
def TotalFourDigitNumbers : ℕ := 9000

/-- The probability of a randomly chosen four-digit number being a power of 2. -/
def ProbabilityPowerOfTwo : ℚ := CountPowersOfTwoFourDigit / TotalFourDigitNumbers

theorem probability_power_of_two_four_digit :
  ProbabilityPowerOfTwo = 1 / 2250 := by sorry

end probability_power_of_two_four_digit_l2061_206111


namespace line_intercept_form_l2061_206130

/-- A line passing through the point (2,3) with slope 2 has the equation x/(1/2) + y/(-1) = 1 in intercept form. -/
theorem line_intercept_form (l : Set (ℝ × ℝ)) : 
  (∀ (x y : ℝ), (x, y) ∈ l ↔ y - 3 = 2 * (x - 2)) →
  (∀ (x y : ℝ), (x, y) ∈ l ↔ x / (1/2) + y / (-1) = 1) :=
by sorry

end line_intercept_form_l2061_206130


namespace negation_equivalence_l2061_206144

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Doctor : U → Prop)
variable (GoodAtMath : U → Prop)

-- Define the statements
def AllDoctorsGoodAtMath : Prop := ∀ x, Doctor x → GoodAtMath x
def AtLeastOneDoctorBadAtMath : Prop := ∃ x, Doctor x ∧ ¬GoodAtMath x

-- Theorem to prove
theorem negation_equivalence :
  AtLeastOneDoctorBadAtMath U Doctor GoodAtMath ↔ ¬(AllDoctorsGoodAtMath U Doctor GoodAtMath) :=
by sorry

end negation_equivalence_l2061_206144


namespace point_symmetry_l2061_206188

/-- A point is symmetric to another point with respect to the origin if their coordinates sum to zero. -/
def symmetric_wrt_origin (p q : ℝ × ℝ) : Prop :=
  p.1 + q.1 = 0 ∧ p.2 + q.2 = 0

/-- The theorem states that the point (2, -3) is symmetric to the point (-2, 3) with respect to the origin. -/
theorem point_symmetry : symmetric_wrt_origin (-2, 3) (2, -3) := by
  sorry

end point_symmetry_l2061_206188


namespace AlF3_MgCl2_cell_potential_l2061_206116

/-- Standard reduction potential for Al^3+/Al in volts -/
def E_Al : ℝ := -1.66

/-- Standard reduction potential for Mg^2+/Mg in volts -/
def E_Mg : ℝ := -2.37

/-- Calculate the cell potential of an electrochemical cell -/
def cell_potential (E_reduction E_oxidation : ℝ) : ℝ :=
  E_reduction - E_oxidation

/-- Theorem: The cell potential of an electrochemical cell involving 
    Aluminum Fluoride and Magnesium Chloride is 0.71 V -/
theorem AlF3_MgCl2_cell_potential : 
  cell_potential E_Al (-E_Mg) = 0.71 := by
  sorry

end AlF3_MgCl2_cell_potential_l2061_206116


namespace max_value_product_l2061_206169

theorem max_value_product (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 5*x + 2*y < 50) :
  xy*(50 - 5*x - 2*y) ≤ 125000/432 ∧ 
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 5*x₀ + 2*y₀ < 50 ∧ x₀*y₀*(50 - 5*x₀ - 2*y₀) = 125000/432 :=
by sorry

end max_value_product_l2061_206169


namespace max_value_of_a_l2061_206158

theorem max_value_of_a : 
  (∀ x : ℝ, x ≠ 0 → |a - 2| ≤ |x + 1/x|) → 
  ∃ a_max : ℝ, a_max = 4 ∧ ∀ a : ℝ, (∀ x : ℝ, x ≠ 0 → |a - 2| ≤ |x + 1/x|) → a ≤ a_max :=
sorry

end max_value_of_a_l2061_206158


namespace number_percentage_equality_l2061_206181

theorem number_percentage_equality : ∃ x : ℚ, (3 / 10 : ℚ) * x = (2 / 10 : ℚ) * 40 ∧ x = 80 / 3 := by
  sorry

end number_percentage_equality_l2061_206181


namespace negation_at_most_two_solutions_l2061_206198

/-- Negation of "at most n" is "at least n+1" -/
axiom negation_at_most (n : ℕ) : ¬(∀ m : ℕ, m ≤ n) ↔ ∃ m : ℕ, m ≥ n + 1

/-- The negation of "there are at most two solutions" is equivalent to "there are at least three solutions" -/
theorem negation_at_most_two_solutions :
  ¬(∃ S : Set ℕ, (∀ n ∈ S, n ≤ 2)) ↔ ∃ S : Set ℕ, (∃ n ∈ S, n ≥ 3) :=
sorry

end negation_at_most_two_solutions_l2061_206198


namespace ben_win_probability_l2061_206187

theorem ben_win_probability (lose_prob : ℚ) (h1 : lose_prob = 5 / 8) 
  (h2 : ¬ ∃ (draw_prob : ℚ), draw_prob ≠ 0) : 
  1 - lose_prob = 3 / 8 := by
sorry

end ben_win_probability_l2061_206187


namespace quadratic_equation_roots_l2061_206149

theorem quadratic_equation_roots (a b c : ℝ) (h : a = 2 ∧ b = -3 ∧ c = 1) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 :=
sorry

end quadratic_equation_roots_l2061_206149


namespace x_value_l2061_206192

theorem x_value : ∃ x : ℝ, (0.25 * x = 0.12 * 1500 - 15) ∧ (x = 660) := by
  sorry

end x_value_l2061_206192


namespace smallest_number_divisible_by_4_6_8_10_l2061_206157

def is_divisible_by_all (n : ℕ) : Prop :=
  (n % 4 = 0) ∧ (n % 6 = 0) ∧ (n % 8 = 0) ∧ (n % 10 = 0)

theorem smallest_number_divisible_by_4_6_8_10 :
  ∀ n : ℕ, n ≥ 136 → (is_divisible_by_all (n - 16) → n ≥ 136) ∧
  is_divisible_by_all (136 - 16) := by
  sorry

end smallest_number_divisible_by_4_6_8_10_l2061_206157


namespace circle_radius_problem_l2061_206182

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def collinear (a b c : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, b = a + t • (c - a)

def externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

def share_common_tangent (c1 c2 c3 : Circle) : Prop :=
  ∃ (l : ℝ × ℝ → Prop), ∀ (p : ℝ × ℝ),
    l p → (∃ (q : ℝ × ℝ), l q ∧ 
      ((p.1 - c1.center.1)^2 + (p.2 - c1.center.2)^2 = c1.radius^2 ∨
       (p.1 - c2.center.1)^2 + (p.2 - c2.center.2)^2 = c2.radius^2 ∨
       (p.1 - c3.center.1)^2 + (p.2 - c3.center.2)^2 = c3.radius^2))

theorem circle_radius_problem (A B C : Circle) 
  (h1 : collinear A.center B.center C.center)
  (h2 : ∃ t : ℝ, 0 < t ∧ t < 1 ∧ B.center = A.center + t • (C.center - A.center))
  (h3 : externally_tangent A B)
  (h4 : externally_tangent B C)
  (h5 : share_common_tangent A B C)
  (h6 : A.radius = 12)
  (h7 : B.radius = 42) :
  C.radius = 147 := by
  sorry

end circle_radius_problem_l2061_206182


namespace union_and_intersection_conditions_l2061_206147

def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m + 3}

theorem union_and_intersection_conditions (m : ℝ) :
  (A ∪ B m = A ↔ m ∈ Set.Ioi (-2) ∪ Set.Iio (-1/2)) ∧
  (A ∩ B m ≠ ∅ ↔ m ∈ Set.Ioo (-2) 1) := by
  sorry

end union_and_intersection_conditions_l2061_206147


namespace tire_usage_calculation_tire_usage_proof_l2061_206143

/-- Calculates the miles each tire was used given the total distance and tire usage pattern. -/
theorem tire_usage_calculation (total_distance : ℕ) (first_part_distance : ℕ) (second_part_distance : ℕ) 
  (total_tires : ℕ) (tires_used_first_part : ℕ) (tires_used_second_part : ℕ) : ℕ :=
  let total_tire_miles := first_part_distance * tires_used_first_part + second_part_distance * tires_used_second_part
  total_tire_miles / total_tires

/-- Proves that each tire was used for 38,571 miles given the specific conditions of the problem. -/
theorem tire_usage_proof : 
  tire_usage_calculation 50000 40000 10000 7 5 7 = 38571 := by
  sorry

end tire_usage_calculation_tire_usage_proof_l2061_206143


namespace convincing_statement_l2061_206148

-- Define the types of people
inductive Person
| Knight
| Knave

-- Define the wealth status of knights
inductive KnightWealth
| Poor
| Rich

-- Define a function to determine if a person tells the truth
def tellsTruth (p : Person) : Prop :=
  match p with
  | Person.Knight => True
  | Person.Knave => False

-- Define the statement "I am not a poor knight"
def statement (p : Person) (w : KnightWealth) : Prop :=
  p = Person.Knight ∧ w ≠ KnightWealth.Poor

-- Theorem to prove
theorem convincing_statement 
  (p : Person) (w : KnightWealth) : 
  tellsTruth p → statement p w → (p = Person.Knight ∧ w = KnightWealth.Rich) :=
by
  sorry


end convincing_statement_l2061_206148


namespace ryan_sandwiches_l2061_206104

def slices_per_sandwich : ℕ := 3
def total_slices : ℕ := 15

theorem ryan_sandwiches :
  total_slices / slices_per_sandwich = 5 := by sorry

end ryan_sandwiches_l2061_206104


namespace decryption_result_l2061_206124

/-- Represents an encrypted text -/
def EncryptedText := String

/-- Represents a decrypted text -/
def DecryptedText := String

/-- The encryption method used for the original message -/
def encryptionMethod (original : String) (encrypted : EncryptedText) : Prop :=
  encrypted.toList.filter (· ∈ original.toList) = original.toList

/-- The decryption function -/
noncomputable def decrypt (text : EncryptedText) : DecryptedText :=
  sorry

/-- Theorem stating the decryption results -/
theorem decryption_result 
  (text1 text2 text3 : EncryptedText)
  (h1 : encryptionMethod "МОСКВА" "ЙМЫВОТСБЛКЪГВЦАЯЯ")
  (h2 : encryptionMethod "МОСКВА" "УКМАПОЧСРКЩВЗАХ")
  (h3 : encryptionMethod "МОСКВА" "ШМФЭОГЧСЙЪКФЬВЫЕАКК")
  (h4 : text1 = "ТПЕОИРВНТМОЛАРГЕИАНВИЛЕДНМТААГТДЬТКУБЧКГЕИШНЕИАЯРЯ")
  (h5 : text2 = "ЛСИЕМГОРТКРОМИТВАВКНОПКРАСЕОГНАЬЕП")
  (h6 : text3 = "РТПАИОМВСВТИЕОБПРОЕННИГЬКЕЕАМТАЛВТДЬСОУМЧШСЕОНШЬИАЯК") :
  (decrypt text1 = "ПОВТОРЕНИЕМАТЬУЧЕНИЯ" ∧
   decrypt text2 = "С ЧИСТОЙ СОВЕСТЬЮ" ∧
   decrypt text3 = "ПОВТОРЕНИЕМАТЬУЧЕНИЯ") :=
by sorry

end decryption_result_l2061_206124


namespace rectangle_segment_product_l2061_206160

theorem rectangle_segment_product (AB BC CD DE x : ℝ) : 
  AB = 5 →
  BC = 11 →
  CD = 3 →
  DE = 9 →
  0 < x →
  x < DE →
  AB * (AB + BC + CD + x) = x * (DE - x) →
  x = 11.95 := by
sorry

end rectangle_segment_product_l2061_206160


namespace rectangle_to_square_transformation_l2061_206159

theorem rectangle_to_square_transformation (a b : ℝ) : 
  a > 0 → b > 0 → a * b = 54 → (3 * a) * (b / 2) = 9^2 := by
  sorry

end rectangle_to_square_transformation_l2061_206159


namespace xy_value_l2061_206174

theorem xy_value (x y : ℝ) 
  (h1 : (8:ℝ)^x / (4:ℝ)^(x+y) = 16)
  (h2 : (16:ℝ)^(x+y) / (4:ℝ)^(7*y) = 256) : 
  x * y = 48 := by
  sorry

end xy_value_l2061_206174


namespace shelves_per_case_l2061_206114

theorem shelves_per_case (num_cases : ℕ) (records_per_shelf : ℕ) (ridges_per_record : ℕ) 
  (shelf_fullness : ℚ) (total_ridges : ℕ) : ℕ :=
  let shelves_per_case := (total_ridges / (shelf_fullness * records_per_shelf * ridges_per_record)) / num_cases
  3

#check shelves_per_case 4 20 60 (3/5) 8640

/- Proof
sorry
-/

end shelves_per_case_l2061_206114


namespace magazine_cost_l2061_206139

theorem magazine_cost (b m : ℝ) 
  (h1 : 2 * b + 2 * m = 26) 
  (h2 : b + 3 * m = 27) : 
  m = 7 := by
sorry

end magazine_cost_l2061_206139


namespace tenth_day_is_monday_l2061_206108

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

end tenth_day_is_monday_l2061_206108


namespace largest_n_for_seq_containment_l2061_206186

/-- A bi-infinite sequence of natural numbers -/
def BiInfiniteSeq := ℤ → ℕ

/-- A sequence is periodic with period p if it repeats every p elements -/
def IsPeriodic (s : BiInfiniteSeq) (p : ℕ) : Prop :=
  ∀ i : ℤ, s i = s (i + p)

/-- A subsequence of length n starting at index i is contained in another sequence -/
def SubseqContained (sub main : BiInfiniteSeq) (n : ℕ) (i : ℤ) : Prop :=
  ∀ k : ℕ, k < n → ∃ j : ℤ, sub (i + k) = main j

/-- The main theorem stating the largest possible n -/
theorem largest_n_for_seq_containment :
  ∃ (n : ℕ) (A B : BiInfiniteSeq),
    IsPeriodic A 1995 ∧
    ¬ IsPeriodic B 1995 ∧
    (∀ i : ℤ, SubseqContained B A n i) ∧
    (∀ m : ℕ, m > n →
      ¬ ∃ (C D : BiInfiniteSeq),
        IsPeriodic C 1995 ∧
        ¬ IsPeriodic D 1995 ∧
        (∀ i : ℤ, SubseqContained D C m i)) ∧
    n = 1995 :=
  sorry

end largest_n_for_seq_containment_l2061_206186


namespace minimize_constant_term_l2061_206179

/-- The function representing the constant term in the expansion -/
def f (a : ℝ) : ℝ := a^3 - 9*a

/-- Theorem stating that √3 minimizes f(a) for a > 0 -/
theorem minimize_constant_term (a : ℝ) (h : a > 0) :
  f a ≥ f (Real.sqrt 3) :=
sorry

end minimize_constant_term_l2061_206179


namespace square_side_length_l2061_206178

theorem square_side_length (area : ℝ) (side : ℝ) : 
  area = 144 → side ^ 2 = area → side = 12 := by
  sorry

end square_side_length_l2061_206178


namespace unique_triple_lcm_l2061_206131

theorem unique_triple_lcm : 
  ∃! (a b c : ℕ+), 
    Nat.lcm a b = 1200 ∧ 
    Nat.lcm b c = 1800 ∧ 
    Nat.lcm c a = 2400 := by
  sorry

end unique_triple_lcm_l2061_206131


namespace equation_solution_l2061_206136

theorem equation_solution : ∃ x : ℝ, (27 - 5 = 4 + x) ∧ (x = 18) := by sorry

end equation_solution_l2061_206136


namespace grandmas_brownie_pan_l2061_206197

/-- Represents a rectangular brownie pan with cuts -/
structure BrowniePan where
  m : ℕ+  -- length
  n : ℕ+  -- width
  length_cuts : ℕ
  width_cuts : ℕ

/-- Calculates the number of interior pieces -/
def interior_pieces (pan : BrowniePan) : ℕ :=
  (pan.m.val - pan.length_cuts - 1) * (pan.n.val - pan.width_cuts - 1)

/-- Calculates the number of perimeter pieces -/
def perimeter_pieces (pan : BrowniePan) : ℕ :=
  2 * (pan.m.val + pan.n.val) - 4

/-- The main theorem about Grandma's brownie pan -/
theorem grandmas_brownie_pan :
  ∃ (pan : BrowniePan),
    pan.length_cuts = 3 ∧
    pan.width_cuts = 5 ∧
    interior_pieces pan = 2 * perimeter_pieces pan ∧
    pan.m = 6 ∧
    pan.n = 12 := by
  sorry

end grandmas_brownie_pan_l2061_206197


namespace power_product_equality_l2061_206134

theorem power_product_equality : (-0.125)^2022 * 8^2023 = 8 := by
  sorry

end power_product_equality_l2061_206134


namespace college_students_count_l2061_206145

/-- Calculates the total number of students in a college given the ratio of boys to girls and the number of girls. -/
def totalStudents (boyRatio girlRatio numGirls : ℕ) : ℕ :=
  let numBoys := boyRatio * numGirls / girlRatio
  numBoys + numGirls

/-- Theorem: In a college where the ratio of boys to girls is 8:5 and there are 190 girls, the total number of students is 494. -/
theorem college_students_count :
  totalStudents 8 5 190 = 494 := by
  sorry

end college_students_count_l2061_206145


namespace car_fuel_usage_l2061_206190

/-- Proves that a car traveling for 5 hours at 60 mph with a fuel efficiency of 1 gallon per 30 miles
    uses 5/6 of a 12-gallon tank. -/
theorem car_fuel_usage (speed : ℝ) (fuel_efficiency : ℝ) (tank_capacity : ℝ) (travel_time : ℝ) :
  speed = 60 →
  fuel_efficiency = 30 →
  tank_capacity = 12 →
  travel_time = 5 →
  (speed * travel_time / fuel_efficiency) / tank_capacity = 5 / 6 := by
  sorry


end car_fuel_usage_l2061_206190


namespace rectangular_solid_spheres_l2061_206167

/-- A rectangular solid -/
structure RectangularSolid where
  length : ℝ
  width : ℝ
  height : ℝ
  length_pos : length > 0
  width_pos : width > 0
  height_pos : height > 0

/-- A sphere -/
structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- Predicate for a sphere being circumscribed around a rectangular solid -/
def isCircumscribed (s : Sphere) (r : RectangularSolid) : Prop :=
  sorry

/-- Predicate for a sphere being inscribed in a rectangular solid -/
def isInscribed (s : Sphere) (r : RectangularSolid) : Prop :=
  sorry

/-- Theorem: A rectangular solid with a circumscribed sphere does not necessarily have an inscribed sphere -/
theorem rectangular_solid_spheres (r : RectangularSolid) (s : Sphere) :
  isCircumscribed s r → ¬∀ (s' : Sphere), isInscribed s' r :=
sorry

end rectangular_solid_spheres_l2061_206167


namespace length_FG_is_20_l2061_206172

/-- Triangle PQR with points F and G -/
structure TrianglePQR where
  /-- Length of side PQ -/
  PQ : ℝ
  /-- Length of side PR -/
  PR : ℝ
  /-- Length of side QR -/
  QR : ℝ
  /-- Point F on PQ -/
  F : ℝ
  /-- Point G on PR -/
  G : ℝ
  /-- FG is parallel to QR -/
  FG_parallel_QR : Bool
  /-- G divides PR in ratio 2:1 -/
  G_divides_PR : G = (2/3) * PR

/-- The length of FG in the given triangle configuration -/
def length_FG (t : TrianglePQR) : ℝ := sorry

/-- Theorem stating that the length of FG is 20 under the given conditions -/
theorem length_FG_is_20 (t : TrianglePQR) 
  (h1 : t.PQ = 24) 
  (h2 : t.PR = 26) 
  (h3 : t.QR = 30) 
  (h4 : t.FG_parallel_QR = true) : 
  length_FG t = 20 := by sorry

end length_FG_is_20_l2061_206172


namespace smallest_yellow_candy_count_l2061_206100

/-- The cost of a piece of yellow candy in cents -/
def yellow_candy_cost : ℕ := 15

/-- The number of red candies Joe can buy -/
def red_candy_count : ℕ := 10

/-- The number of green candies Joe can buy -/
def green_candy_count : ℕ := 16

/-- The number of blue candies Joe can buy -/
def blue_candy_count : ℕ := 18

theorem smallest_yellow_candy_count :
  ∃ n : ℕ, n > 0 ∧
  (yellow_candy_cost * n) % red_candy_count = 0 ∧
  (yellow_candy_cost * n) % green_candy_count = 0 ∧
  (yellow_candy_cost * n) % blue_candy_count = 0 ∧
  (∀ m : ℕ, m > 0 →
    (yellow_candy_cost * m) % red_candy_count = 0 →
    (yellow_candy_cost * m) % green_candy_count = 0 →
    (yellow_candy_cost * m) % blue_candy_count = 0 →
    m ≥ n) ∧
  n = 48 := by
  sorry

end smallest_yellow_candy_count_l2061_206100


namespace sqrt_27_minus_3tan60_plus_power_equals_1_l2061_206195

theorem sqrt_27_minus_3tan60_plus_power_equals_1 :
  Real.sqrt 27 - 3 * Real.tan (60 * π / 180) + (π - Real.sqrt 2) ^ 0 = 1 := by
  sorry

end sqrt_27_minus_3tan60_plus_power_equals_1_l2061_206195


namespace units_digit_sum_powers_l2061_206141

theorem units_digit_sum_powers : (19^89 + 89^19) % 10 = 8 := by
  sorry

end units_digit_sum_powers_l2061_206141


namespace kath_group_cost_l2061_206155

/-- Calculates the total cost of movie admission for a group, given a regular price, 
    discount amount, and number of people in the group. -/
def total_cost (regular_price discount : ℕ) (group_size : ℕ) : ℕ :=
  (regular_price - discount) * group_size

/-- Proves that the total cost for Kath's group is $30 -/
theorem kath_group_cost : 
  let regular_price : ℕ := 8
  let discount : ℕ := 3
  let kath_siblings : ℕ := 2
  let kath_friends : ℕ := 3
  let group_size : ℕ := 1 + kath_siblings + kath_friends
  total_cost regular_price discount group_size = 30 := by
  sorry

#eval total_cost 8 3 6

end kath_group_cost_l2061_206155


namespace minutes_to_skate_on_ninth_day_l2061_206103

/-- The number of minutes Gage skated each day for the first 6 days -/
def minutes_per_day_first_6 : ℕ := 60

/-- The number of days Gage skated for 60 minutes -/
def days_skating_60_min : ℕ := 6

/-- The number of minutes Gage skated each day for the next 2 days -/
def minutes_per_day_next_2 : ℕ := 120

/-- The number of days Gage skated for 120 minutes -/
def days_skating_120_min : ℕ := 2

/-- The target average number of minutes per day for all 9 days -/
def target_average_minutes : ℕ := 100

/-- The total number of days Gage skated -/
def total_days : ℕ := 9

/-- Theorem stating the number of minutes Gage needs to skate on the 9th day -/
theorem minutes_to_skate_on_ninth_day :
  target_average_minutes * total_days -
  (minutes_per_day_first_6 * days_skating_60_min +
   minutes_per_day_next_2 * days_skating_120_min) = 300 := by
  sorry

end minutes_to_skate_on_ninth_day_l2061_206103


namespace shopping_tax_calculation_l2061_206150

theorem shopping_tax_calculation (total : ℝ) (clothing_percent : ℝ) (food_percent : ℝ) 
  (other_percent : ℝ) (clothing_tax : ℝ) (food_tax : ℝ) (total_tax_percent : ℝ) 
  (h1 : clothing_percent = 0.5)
  (h2 : food_percent = 0.1)
  (h3 : other_percent = 0.4)
  (h4 : clothing_percent + food_percent + other_percent = 1)
  (h5 : clothing_tax = 0.04)
  (h6 : food_tax = 0)
  (h7 : total_tax_percent = 0.052)
  : ∃ other_tax : ℝ, 
    clothing_tax * clothing_percent * total + 
    food_tax * food_percent * total + 
    other_tax * other_percent * total = 
    total_tax_percent * total ∧ 
    other_tax = 0.08 := by
sorry

end shopping_tax_calculation_l2061_206150


namespace negation_equivalence_l2061_206152

theorem negation_equivalence : 
  (¬(x = 3 → x^2 - 2*x - 3 = 0)) ↔ (x ≠ 3 → x^2 - 2*x - 3 ≠ 0) :=
by sorry

end negation_equivalence_l2061_206152


namespace ellipse_properties_l2061_206191

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the foci
def F1 : ℝ × ℝ := sorry
def F2 : ℝ × ℝ := sorry

-- Define a point on the ellipse
def P : ℝ × ℝ := sorry
axiom P_on_ellipse : ellipse P.1 P.2

-- Define the eccentricity
def eccentricity : ℝ := sorry

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the angle between three points
def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem ellipse_properties :
  eccentricity = 1/2 ∧
  ∃ (Q : ℝ × ℝ), ellipse Q.1 Q.2 ∧ distance Q F1 = 3 ∧ ∀ (R : ℝ × ℝ), ellipse R.1 R.2 → distance R F1 ≤ 3 ∧
  0 ≤ angle F1 P F2 ∧ angle F1 P F2 ≤ π/3 :=
sorry

end ellipse_properties_l2061_206191


namespace cortland_apples_l2061_206142

theorem cortland_apples (total : ℝ) (golden : ℝ) (macintosh : ℝ) 
  (h1 : total = 0.67)
  (h2 : golden = 0.17)
  (h3 : macintosh = 0.17) :
  total - (golden + macintosh) = 0.33 := by
  sorry

end cortland_apples_l2061_206142


namespace wanda_blocks_theorem_l2061_206105

/-- The total number of blocks Wanda has after receiving more from Theresa -/
def total_blocks (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem stating that Wanda's total blocks is the sum of her initial blocks and additional blocks -/
theorem wanda_blocks_theorem (initial : ℕ) (additional : ℕ) :
  total_blocks initial additional = initial + additional := by
  sorry

end wanda_blocks_theorem_l2061_206105


namespace arc_length_example_l2061_206166

/-- The length of an arc in a circle, given the radius and central angle -/
def arcLength (radius : ℝ) (centralAngle : ℝ) : ℝ :=
  radius * centralAngle

theorem arc_length_example :
  let radius : ℝ := 10
  let centralAngle : ℝ := 2 * Real.pi / 3
  arcLength radius centralAngle = 20 * Real.pi / 3 := by
sorry


end arc_length_example_l2061_206166


namespace quick_response_solution_l2061_206132

def quick_response_problem (x y z : ℕ) : Prop :=
  5 * x + 4 * y + 3 * z = 15 ∧ 
  (x = 1 ∧ y = 1 ∧ z = 2) ∨ (x = 0 ∧ y = 3 ∧ z = 1)

theorem quick_response_solution :
  ∀ x y z : ℕ, 5 * x + 4 * y + 3 * z = 15 → quick_response_problem x y z :=
by
  sorry

#check quick_response_solution

end quick_response_solution_l2061_206132


namespace equal_roots_quadratic_l2061_206196

theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, x^2 - 3*x + m = 0 ∧ 
   ∀ y : ℝ, y^2 - 3*y + m = 0 → y = x) → 
  m = 9/4 := by
sorry

end equal_roots_quadratic_l2061_206196


namespace complex_equation_solution_l2061_206127

theorem complex_equation_solution (a b : ℝ) :
  (Complex.mk 1 2) / (Complex.mk a b) = Complex.mk 1 1 →
  a = (3 : ℝ) / 2 ∧ b = (1 : ℝ) / 2 := by
  sorry

end complex_equation_solution_l2061_206127


namespace smallest_k_for_sum_of_squares_multiple_of_400_l2061_206165

theorem smallest_k_for_sum_of_squares_multiple_of_400 : 
  ∀ k : ℕ+, k < 800 → ¬(∃ m : ℕ, k * (k + 1) * (2 * k + 1) = 6 * 400 * m) ∧ 
  ∃ m : ℕ, 800 * (800 + 1) * (2 * 800 + 1) = 6 * 400 * m :=
by sorry

end smallest_k_for_sum_of_squares_multiple_of_400_l2061_206165


namespace relationship_between_a_and_b_l2061_206180

theorem relationship_between_a_and_b (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : Real.exp a + 2 * a = Real.exp b + 3 * b) : a > b := by
  sorry

end relationship_between_a_and_b_l2061_206180


namespace smallest_n_for_roots_of_unity_l2061_206118

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

end smallest_n_for_roots_of_unity_l2061_206118


namespace tims_prank_combinations_l2061_206151

/-- Represents the number of choices for each day of the prank --/
structure PrankChoices where
  day1 : Nat
  day2 : Nat
  day3 : Nat
  day4 : Nat → Nat
  day5 : Nat

/-- Calculates the total number of combinations for the prank --/
def totalCombinations (choices : PrankChoices) : Nat :=
  choices.day1 * choices.day2 * choices.day3 * 
  (choices.day3 * choices.day4 1 + choices.day3 * choices.day4 2 + choices.day3 * choices.day4 3) *
  choices.day5

/-- The specific choices for Tim's prank --/
def timsPrankChoices : PrankChoices where
  day1 := 1
  day2 := 2
  day3 := 3
  day4 := fun n => match n with
    | 1 => 2
    | 2 => 3
    | _ => 1
  day5 := 1

theorem tims_prank_combinations :
  totalCombinations timsPrankChoices = 36 := by
  sorry

end tims_prank_combinations_l2061_206151


namespace internal_diagonal_cubes_l2061_206123

def cuboid_diagonal_cubes (x y z : ℕ) : ℕ :=
  x + y + z - (Nat.gcd x y + Nat.gcd y z + Nat.gcd z x) + Nat.gcd x (Nat.gcd y z)

theorem internal_diagonal_cubes :
  cuboid_diagonal_cubes 168 350 390 = 880 := by
  sorry

end internal_diagonal_cubes_l2061_206123


namespace percentage_calculation_l2061_206138

theorem percentage_calculation (y : ℝ) : 
  0.11 * y = 0.3 * (0.7 * y) - 0.1 * y := by sorry

end percentage_calculation_l2061_206138


namespace simplify_expression_l2061_206133

theorem simplify_expression (x y : ℚ) (hx : x = 5) (hy : y = 2) :
  (10 * x * y^3) / (15 * x^2 * y^2) = 4 / 15 := by
  sorry

end simplify_expression_l2061_206133
