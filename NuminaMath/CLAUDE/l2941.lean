import Mathlib

namespace complement_union_theorem_l2941_294181

universe u

def U : Set ℕ := {0, 1, 3, 4, 5, 6, 8}
def A : Set ℕ := {1, 4, 5, 8}
def B : Set ℕ := {2, 6}

theorem complement_union_theorem :
  (U \ A) ∪ B = {0, 2, 3, 6} := by sorry

end complement_union_theorem_l2941_294181


namespace total_spent_is_30_40_l2941_294103

/-- Represents the store's inventory and pricing --/
structure Store where
  barrette_price : ℝ
  comb_price : ℝ
  hairband_price : ℝ
  hair_ties_price : ℝ

/-- Represents a customer's purchase --/
structure Purchase where
  barrettes : ℕ
  combs : ℕ
  hairbands : ℕ
  hair_ties : ℕ

/-- Calculates the total cost of a purchase before discount and tax --/
def purchase_cost (s : Store) (p : Purchase) : ℝ :=
  s.barrette_price * p.barrettes +
  s.comb_price * p.combs +
  s.hairband_price * p.hairbands +
  s.hair_ties_price * p.hair_ties

/-- Applies discount if applicable --/
def apply_discount (cost : ℝ) (item_count : ℕ) : ℝ :=
  if item_count > 5 then cost * 0.85 else cost

/-- Applies sales tax --/
def apply_tax (cost : ℝ) : ℝ :=
  cost * 1.08

/-- Calculates the final cost of a purchase after discount and tax --/
def final_cost (s : Store) (p : Purchase) : ℝ :=
  let initial_cost := purchase_cost s p
  let item_count := p.barrettes + p.combs + p.hairbands + p.hair_ties
  let discounted_cost := apply_discount initial_cost item_count
  apply_tax discounted_cost

/-- The main theorem --/
theorem total_spent_is_30_40 (s : Store) (k_purchase c_purchase : Purchase) :
  s.barrette_price = 4 ∧
  s.comb_price = 2 ∧
  s.hairband_price = 3 ∧
  s.hair_ties_price = 2.5 ∧
  k_purchase = { barrettes := 1, combs := 1, hairbands := 2, hair_ties := 0 } ∧
  c_purchase = { barrettes := 3, combs := 1, hairbands := 0, hair_ties := 2 } →
  final_cost s k_purchase + final_cost s c_purchase = 30.40 := by
  sorry

end total_spent_is_30_40_l2941_294103


namespace inequality_solution_length_l2941_294165

theorem inequality_solution_length (c d : ℝ) : 
  (∀ x : ℝ, d ≤ x^2 + 4*x + 3 ∧ x^2 + 4*x + 3 ≤ c) →
  (∃ a b : ℝ, ∀ x : ℝ, (d ≤ x^2 + 4*x + 3 ∧ x^2 + 4*x + 3 ≤ c) ↔ (a ≤ x ∧ x ≤ b)) →
  (∃ a b : ℝ, b - a = 8 ∧ ∀ x : ℝ, (d ≤ x^2 + 4*x + 3 ∧ x^2 + 4*x + 3 ≤ c) ↔ (a ≤ x ∧ x ≤ b)) →
  c - d = 0 :=
by sorry

end inequality_solution_length_l2941_294165


namespace kyle_weekly_papers_l2941_294132

/-- The number of papers Kyle delivers in a week -/
def weekly_papers (weekday_houses : ℕ) (sunday_skip : ℕ) (sunday_only : ℕ) : ℕ :=
  6 * weekday_houses + (weekday_houses - sunday_skip + sunday_only)

/-- Theorem stating the total number of papers Kyle delivers in a week -/
theorem kyle_weekly_papers :
  weekly_papers 100 10 30 = 720 := by
  sorry

#eval weekly_papers 100 10 30

end kyle_weekly_papers_l2941_294132


namespace simple_random_sampling_prob_std_dev_transformation_l2941_294194

/-- Simple random sampling probability -/
theorem simple_random_sampling_prob (population_size : ℕ) (sample_size : ℕ) :
  population_size = 50 → sample_size = 10 →
  (sample_size : ℝ) / (population_size : ℝ) = 0.2 := by sorry

/-- Standard deviation transformation -/
theorem std_dev_transformation (x : Fin 10 → ℝ) (σ : ℝ) :
  Real.sqrt (Finset.univ.sum (λ i => (x i - Finset.univ.sum x / 10) ^ 2) / 10) = σ →
  Real.sqrt (Finset.univ.sum (λ i => ((2 * x i - 1) - Finset.univ.sum (λ j => 2 * x j - 1) / 10) ^ 2) / 10) = 2 * σ := by sorry

end simple_random_sampling_prob_std_dev_transformation_l2941_294194


namespace flip_invariant_numbers_l2941_294106

/-- A digit that remains unchanged when flipped upside down -/
inductive FlipInvariantDigit : Nat → Prop
  | zero : FlipInvariantDigit 0
  | eight : FlipInvariantDigit 8

/-- A three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  is_valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≥ 0 ∧ tens ≤ 9 ∧ ones ≥ 0 ∧ ones ≤ 9

/-- A three-digit number that remains unchanged when flipped upside down -/
def FlipInvariantNumber (n : ThreeDigitNumber) : Prop :=
  FlipInvariantDigit n.hundreds ∧ FlipInvariantDigit n.tens ∧ FlipInvariantDigit n.ones

theorem flip_invariant_numbers :
  ∀ n : ThreeDigitNumber, FlipInvariantNumber n →
    (n.hundreds = 8 ∧ n.tens = 0 ∧ n.ones = 8) ∨ (n.hundreds = 8 ∧ n.tens = 8 ∧ n.ones = 8) :=
by sorry

end flip_invariant_numbers_l2941_294106


namespace sum_of_cube_roots_bounded_l2941_294112

theorem sum_of_cube_roots_bounded (a₁ a₂ a₃ a₄ : ℝ) 
  (h_pos : a₁ > 0 ∧ a₂ > 0 ∧ a₃ > 0 ∧ a₄ > 0) 
  (h_sum : a₁ + a₂ + a₃ + a₄ = 1) : 
  5 < (7 * a₁ + 1) ^ (1/3) + (7 * a₂ + 1) ^ (1/3) + 
      (7 * a₃ + 1) ^ (1/3) + (7 * a₄ + 1) ^ (1/3) ∧
      (7 * a₁ + 1) ^ (1/3) + (7 * a₂ + 1) ^ (1/3) + 
      (7 * a₃ + 1) ^ (1/3) + (7 * a₄ + 1) ^ (1/3) < 6 := by
  sorry

end sum_of_cube_roots_bounded_l2941_294112


namespace chair_table_price_percentage_l2941_294116

/-- The price of a chair in dollars -/
def chair_price : ℚ := (96 - 84)

/-- The price of a table in dollars -/
def table_price : ℚ := 84

/-- The price of 2 chairs and 1 table -/
def price_2c1t : ℚ := 2 * chair_price + table_price

/-- The price of 1 chair and 2 tables -/
def price_1c2t : ℚ := chair_price + 2 * table_price

/-- The percentage of price_2c1t to price_1c2t -/
def percentage : ℚ := price_2c1t / price_1c2t * 100

theorem chair_table_price_percentage :
  percentage = 60 := by sorry

end chair_table_price_percentage_l2941_294116


namespace xiao_ming_walk_relation_l2941_294182

/-- Represents the relationship between remaining distance and time walked
    for a person walking towards a destination. -/
def distance_time_relation (total_distance : ℝ) (speed : ℝ) (x : ℝ) : ℝ :=
  total_distance - speed * x

/-- Theorem stating the relationship between remaining distance and time walked
    for Xiao Ming's walk to school. -/
theorem xiao_ming_walk_relation :
  ∀ x y : ℝ, y = distance_time_relation 1200 70 x ↔ y = -70 * x + 1200 :=
by sorry

end xiao_ming_walk_relation_l2941_294182


namespace star_four_three_l2941_294134

/-- Definition of the star operation -/
def star (a b : ℤ) : ℤ := a^2 + a*b - b^3

/-- Theorem stating that 4 ⋆ 3 = 1 -/
theorem star_four_three : star 4 3 = 1 := by
  sorry

end star_four_three_l2941_294134


namespace selection_ways_l2941_294179

def club_size : ℕ := 20
def co_presidents : ℕ := 2
def treasurers : ℕ := 1

theorem selection_ways : 
  (club_size.choose co_presidents * (club_size - co_presidents).choose treasurers) = 3420 := by
  sorry

end selection_ways_l2941_294179


namespace defeat_dragon_l2941_294177

/-- Represents the three heroes --/
inductive Hero
| Ilya
| Dobrynya
| Alyosha

/-- Calculates the number of heads removed by a hero's strike --/
def headsRemoved (hero : Hero) (h : ℕ) : ℕ :=
  match hero with
  | Hero.Ilya => (h / 2) + 1
  | Hero.Dobrynya => (h / 3) + 2
  | Hero.Alyosha => (h / 4) + 3

/-- Represents a sequence of strikes by the heroes --/
def Strike := List Hero

/-- Applies a sequence of strikes to the initial number of heads --/
def applyStrikes (initialHeads : ℕ) (strikes : Strike) : ℕ :=
  strikes.foldl (fun remaining hero => remaining - (headsRemoved hero remaining)) initialHeads

/-- Theorem: For any initial number of heads, there exists a sequence of strikes that reduces it to zero --/
theorem defeat_dragon (initialHeads : ℕ) : ∃ (strikes : Strike), applyStrikes initialHeads strikes = 0 :=
sorry


end defeat_dragon_l2941_294177


namespace part_one_part_two_l2941_294139

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.b = 2 ∧ Real.cos t.B = 4/5

-- Part 1: Prove that if A = 30°, then a = 5/3
theorem part_one (t : Triangle) (h : triangle_conditions t) (h_A : t.A = 30 * π / 180) :
  t.a = 5/3 := by sorry

-- Part 2: Prove that the maximum area of the triangle is 3
theorem part_two (t : Triangle) (h : triangle_conditions t) :
  (∃ (max_area : ℝ), max_area = 3 ∧ 
    ∀ (t' : Triangle), triangle_conditions t' → 
      1/2 * t'.a * t'.c * Real.sin t'.B ≤ max_area) := by sorry

end part_one_part_two_l2941_294139


namespace total_cost_theorem_l2941_294173

/-- The cost of items and their relationships -/
structure ItemCosts where
  pencil_cost : ℝ
  pen_cost : ℝ
  notebook_cost : ℝ
  pen_pencil_diff : ℝ
  notebook_pen_ratio : ℝ
  notebook_discount : ℝ
  cad_usd_rate : ℝ

/-- Calculate the total cost in USD -/
def total_cost_usd (costs : ItemCosts) : ℝ :=
  let pen_cost := costs.pencil_cost + costs.pen_pencil_diff
  let notebook_cost := costs.notebook_pen_ratio * pen_cost
  let discounted_notebook_cost := notebook_cost * (1 - costs.notebook_discount)
  let total_cad := costs.pencil_cost + pen_cost + discounted_notebook_cost
  total_cad * costs.cad_usd_rate

/-- Theorem stating the total cost in USD -/
theorem total_cost_theorem (costs : ItemCosts) 
  (h1 : costs.pencil_cost = 2)
  (h2 : costs.pen_pencil_diff = 9)
  (h3 : costs.notebook_pen_ratio = 2)
  (h4 : costs.notebook_discount = 0.15)
  (h5 : costs.cad_usd_rate = 1.25) :
  total_cost_usd costs = 39.63 := by
  sorry

#eval total_cost_usd {
  pencil_cost := 2,
  pen_cost := 11,
  notebook_cost := 22,
  pen_pencil_diff := 9,
  notebook_pen_ratio := 2,
  notebook_discount := 0.15,
  cad_usd_rate := 1.25
}

end total_cost_theorem_l2941_294173


namespace intersection_of_inequalities_l2941_294147

theorem intersection_of_inequalities (m n : ℝ) (h : -1 < m ∧ m < 0 ∧ 0 < n) :
  {x : ℝ | m < x ∧ x < n} ∩ {x : ℝ | -1 < x ∧ x < 0} = {x : ℝ | -1 < x ∧ x < 0} := by
  sorry

end intersection_of_inequalities_l2941_294147


namespace number_equation_l2941_294198

theorem number_equation (x : ℝ) : 3550 - (x / 20.04) = 3500 ↔ x = 1002 := by
  sorry

end number_equation_l2941_294198


namespace solution_set_not_negative_interval_l2941_294162

theorem solution_set_not_negative_interval (a b : ℝ) :
  {x : ℝ | a * x > b} ≠ Set.Iio (-b/a) :=
sorry

end solution_set_not_negative_interval_l2941_294162


namespace carnival_tickets_total_l2941_294152

/-- Represents the number of tickets used for a carnival ride -/
structure RideTickets where
  ferrisWheel : ℕ
  bumperCars : ℕ
  rollerCoaster : ℕ

/-- Calculates the total number of tickets used for a set of rides -/
def totalTickets (rides : RideTickets) (ferrisWheelCost bumperCarsCost rollerCoasterCost : ℕ) : ℕ :=
  rides.ferrisWheel * ferrisWheelCost + rides.bumperCars * bumperCarsCost + rides.rollerCoaster * rollerCoasterCost

/-- Theorem stating the total number of tickets used by Oliver, Emma, and Sophia -/
theorem carnival_tickets_total : 
  let ferrisWheelCost := 7
  let bumperCarsCost := 5
  let rollerCoasterCost := 9
  let oliver := RideTickets.mk 5 4 0
  let emma := RideTickets.mk 0 6 3
  let sophia := RideTickets.mk 3 2 2
  totalTickets oliver ferrisWheelCost bumperCarsCost rollerCoasterCost +
  totalTickets emma ferrisWheelCost bumperCarsCost rollerCoasterCost +
  totalTickets sophia ferrisWheelCost bumperCarsCost rollerCoasterCost = 161 := by
  sorry

end carnival_tickets_total_l2941_294152


namespace marble_arrangement_theorem_l2941_294149

/-- The number of green marbles -/
def green_marbles : ℕ := 6

/-- The maximum number of red marbles that satisfies the arrangement condition -/
def max_red_marbles : ℕ := 18

/-- The total number of marbles in the arrangement -/
def total_marbles : ℕ := green_marbles + max_red_marbles

/-- The number of ways to arrange the marbles -/
def arrangement_count : ℕ := Nat.choose total_marbles green_marbles

theorem marble_arrangement_theorem :
  arrangement_count % 1000 = 564 := by sorry

end marble_arrangement_theorem_l2941_294149


namespace periodic_function_value_l2941_294168

def is_periodic (f : ℝ → ℝ) (period : ℝ) : Prop :=
  ∀ x, f (x + period) = f x

theorem periodic_function_value (f : ℝ → ℝ) :
  is_periodic f 3 →
  (∀ x ∈ Set.Icc (-1) 2, f x = x + 1) →
  f 2017 = 2 := by
sorry

end periodic_function_value_l2941_294168


namespace track_length_is_900_l2941_294180

/-- The length of a circular track where two runners meet again -/
def track_length (v1 v2 t : ℝ) : ℝ :=
  (v1 - v2) * t

/-- Theorem stating the length of the track is 900 meters -/
theorem track_length_is_900 :
  let v1 : ℝ := 30  -- Speed of Bruce in m/s
  let v2 : ℝ := 20  -- Speed of Bhishma in m/s
  let t : ℝ := 90   -- Time in seconds
  track_length v1 v2 t = 900 := by
  sorry

#eval track_length 30 20 90  -- Should output 900

end track_length_is_900_l2941_294180


namespace complex_number_real_condition_l2941_294166

theorem complex_number_real_condition (a b : ℝ) :
  let z : ℂ := Complex.mk (a^2 + b^2) (a + |a|)
  (z.im = 0) ↔ (a ≤ 0) := by sorry

end complex_number_real_condition_l2941_294166


namespace third_set_total_l2941_294122

/-- Represents a set of candies -/
structure CandySet where
  hard : ℕ
  chocolate : ℕ
  gummy : ℕ

/-- The problem setup -/
def candy_problem (set1 set2 set3 : CandySet) : Prop :=
  -- Total number of each type of candy is equal across all sets
  set1.hard + set2.hard + set3.hard = set1.chocolate + set2.chocolate + set3.chocolate ∧
  set1.hard + set2.hard + set3.hard = set1.gummy + set2.gummy + set3.gummy ∧
  -- First set conditions
  set1.chocolate = set1.gummy ∧
  set1.hard = set1.chocolate + 7 ∧
  -- Second set conditions
  set2.hard = set2.chocolate ∧
  set2.gummy = set2.hard - 15 ∧
  -- Third set condition
  set3.hard = 0

/-- The theorem to prove -/
theorem third_set_total (set1 set2 set3 : CandySet) 
  (h : candy_problem set1 set2 set3) : 
  set3.chocolate + set3.gummy = 29 := by
  sorry

end third_set_total_l2941_294122


namespace root_product_theorem_l2941_294192

theorem root_product_theorem (a b m p r : ℝ) : 
  (a^2 - m*a + 3 = 0) →
  (b^2 - m*b + 3 = 0) →
  ((a + 1/b)^2 - p*(a + 1/b) + r = 0) →
  ((b + 1/a + 1)^2 - p*(b + 1/a + 1) + r = 0) →
  r = 19/3 := by sorry

end root_product_theorem_l2941_294192


namespace parallel_perpendicular_lines_l2941_294144

-- Define the lines and points
def l₁ (m : ℝ) := {(x, y) : ℝ × ℝ | (y - m) / (x + 2) = (4 - m) / (m + 2)}
def l₂ := {(x, y) : ℝ × ℝ | 2*x + y - 1 = 0}
def l₃ (n : ℝ) := {(x, y) : ℝ × ℝ | x + n*y + 1 = 0}

def A (m : ℝ) : ℝ × ℝ := (-2, m)
def B (m : ℝ) : ℝ × ℝ := (m, 4)

-- Define the theorem
theorem parallel_perpendicular_lines (m n : ℝ) : 
  (A m ∈ l₁ m) → 
  (B m ∈ l₁ m) → 
  (∀ (x y : ℝ), (x, y) ∈ l₁ m ↔ (x, y) ∈ l₂) → 
  (∀ (x y : ℝ), (x, y) ∈ l₂ → (x, y) ∈ l₃ n → x = y) → 
  m + n = -10 := by
  sorry

#check parallel_perpendicular_lines

end parallel_perpendicular_lines_l2941_294144


namespace laundry_wash_time_l2941_294133

/-- The time it takes to wash clothes in minutes -/
def clothes_time : ℕ := 30

/-- The time it takes to wash towels in minutes -/
def towels_time : ℕ := 2 * clothes_time

/-- The time it takes to wash sheets in minutes -/
def sheets_time : ℕ := towels_time - 15

/-- The total time it takes to wash all laundry in minutes -/
def total_wash_time : ℕ := clothes_time + towels_time + sheets_time

theorem laundry_wash_time : total_wash_time = 135 := by
  sorry

end laundry_wash_time_l2941_294133


namespace hyperbola_real_axis_length_l2941_294190

/-- The length of the real axis of a hyperbola with equation x²/9 - y² = 1 is 6. -/
theorem hyperbola_real_axis_length :
  ∃ (f : ℝ → ℝ → Prop),
    (∀ x y, f x y ↔ x^2/9 - y^2 = 1) →
    (∃ a : ℝ, a > 0 ∧ ∀ x y, f x y ↔ x^2/a^2 - y^2 = 1) →
    2 * Real.sqrt 9 = 6 :=
by sorry

end hyperbola_real_axis_length_l2941_294190


namespace greatest_power_of_three_in_40_factorial_l2941_294169

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def is_factor (a b : ℕ) : Prop := b % a = 0

def count_multiples (n k : ℕ) : ℕ := n / k

theorem greatest_power_of_three_in_40_factorial :
  (∀ m : ℕ, m ≤ 18 → is_factor (3^m) (factorial 40)) ∧
  ¬(is_factor (3^19) (factorial 40)) := by
  sorry

end greatest_power_of_three_in_40_factorial_l2941_294169


namespace fraction_sum_minus_five_equals_negative_four_l2941_294137

theorem fraction_sum_minus_five_equals_negative_four (a b : ℝ) (h : a ≠ b) :
  a / (a - b) + b / (b - a) - 5 = -4 := by
  sorry

end fraction_sum_minus_five_equals_negative_four_l2941_294137


namespace hotel_profit_theorem_l2941_294115

/-- Calculates the hotel's weekly profit given the operations expenses and service percentages --/
def hotel_profit (operations_expenses : ℚ) 
  (meetings_percent : ℚ) (events_percent : ℚ) (rooms_percent : ℚ)
  (meetings_tax : ℚ) (meetings_commission : ℚ)
  (events_tax : ℚ) (events_commission : ℚ)
  (rooms_tax : ℚ) (rooms_commission : ℚ) : ℚ :=
  let meetings_income := meetings_percent * operations_expenses
  let events_income := events_percent * operations_expenses
  let rooms_income := rooms_percent * operations_expenses
  let total_income := meetings_income + events_income + rooms_income
  let meetings_additional := meetings_income * (meetings_tax + meetings_commission)
  let events_additional := events_income * (events_tax + events_commission)
  let rooms_additional := rooms_income * (rooms_tax + rooms_commission)
  let total_additional := meetings_additional + events_additional + rooms_additional
  total_income - operations_expenses - total_additional

/-- The hotel's weekly profit is $1,283.75 given the specified conditions --/
theorem hotel_profit_theorem : 
  hotel_profit 5000 (5/8) (3/10) (11/20) (1/10) (1/20) (2/25) (3/50) (3/25) (3/100) = 1283.75 := by
  sorry


end hotel_profit_theorem_l2941_294115


namespace train_crossing_time_l2941_294123

/-- The time it takes for a train to cross a man walking in the opposite direction -/
theorem train_crossing_time (train_length : Real) (train_speed : Real) (man_speed : Real) :
  train_length = 50 ∧ 
  train_speed = 24.997600191984645 ∧ 
  man_speed = 5 →
  (train_length / ((train_speed + man_speed) * (1000 / 3600))) = 6 := by
  sorry

end train_crossing_time_l2941_294123


namespace two_number_problem_l2941_294193

theorem two_number_problem (x y : ℝ) (h1 : x + y = 40) (h2 : 3 * y - 4 * x = 10) :
  |y - x| = 8.58 := by
sorry

end two_number_problem_l2941_294193


namespace pool_filling_time_l2941_294187

/-- The number of hours it takes for a swimming pool to reach full capacity -/
def full_capacity_hours : ℕ := 8

/-- The factor by which the water volume increases each hour -/
def volume_increase_factor : ℕ := 3

/-- The fraction of the pool's capacity we're interested in -/
def target_fraction : ℚ := 1 / 9

/-- The number of hours it takes to reach the target fraction of capacity -/
def target_hours : ℕ := 6

theorem pool_filling_time :
  (volume_increase_factor ^ (full_capacity_hours - target_hours) : ℚ) = 1 / target_fraction :=
sorry

end pool_filling_time_l2941_294187


namespace range_of_m_l2941_294145

-- Define propositions p and q
def p (x : ℝ) : Prop := x < -2 ∨ x > 10
def q (x m : ℝ) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m^2

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (m : ℝ) : Prop :=
  (∀ x : ℝ, ¬(p x) → q x m) ∧ ¬(∀ x : ℝ, q x m → ¬(p x))

-- Theorem statement
theorem range_of_m :
  ∀ m : ℝ, sufficient_not_necessary m ↔ m > 3 :=
sorry

end range_of_m_l2941_294145


namespace surface_area_difference_l2941_294117

def rectangular_solid_surface_area (l w h : ℝ) : ℝ :=
  2 * (l * w + l * h + w * h)

def cube_surface_area (s : ℝ) : ℝ :=
  6 * s^2

def new_exposed_area (s : ℝ) : ℝ :=
  3 * s^2

theorem surface_area_difference :
  let original_area := rectangular_solid_surface_area 4 5 6
  let removed_area := cube_surface_area 2
  let exposed_area := new_exposed_area 2
  original_area - removed_area + exposed_area = original_area - 12
  := by sorry

end surface_area_difference_l2941_294117


namespace medicine_dose_per_kg_l2941_294178

theorem medicine_dose_per_kg (child_weight : ℝ) (dose_parts : ℕ) (dose_per_part : ℝ) :
  child_weight = 30 →
  dose_parts = 3 →
  dose_per_part = 50 →
  (dose_parts * dose_per_part) / child_weight = 5 := by
  sorry

end medicine_dose_per_kg_l2941_294178


namespace smallest_number_last_three_digits_l2941_294191

def is_divisible_by (n m : ℕ) : Prop := n % m = 0

def consists_of_2_and_7 (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 2 ∨ d = 7

def has_at_least_one_2_and_7 (n : ℕ) : Prop :=
  2 ∈ n.digits 10 ∧ 7 ∈ n.digits 10

def last_three_digits (n : ℕ) : ℕ := n % 1000

theorem smallest_number_last_three_digits :
  ∃ m : ℕ, 
    (∀ k : ℕ, k < m → 
      ¬(is_divisible_by k 6 ∧ 
        is_divisible_by k 8 ∧ 
        consists_of_2_and_7 k ∧ 
        has_at_least_one_2_and_7 k)) ∧
    is_divisible_by m 6 ∧
    is_divisible_by m 8 ∧
    consists_of_2_and_7 m ∧
    has_at_least_one_2_and_7 m ∧
    last_three_digits m = 722 :=
by sorry

end smallest_number_last_three_digits_l2941_294191


namespace max_product_sum_300_l2941_294164

theorem max_product_sum_300 :
  ∃ (x : ℤ), x * (300 - x) = 22500 ∧ ∀ (y : ℤ), y * (300 - y) ≤ 22500 :=
sorry

end max_product_sum_300_l2941_294164


namespace annual_increase_fraction_l2941_294125

theorem annual_increase_fraction (initial_amount final_amount : ℝ) 
  (h1 : initial_amount > 0)
  (h2 : final_amount > initial_amount)
  (h3 : initial_amount * (1 + f)^2 = final_amount)
  (h4 : initial_amount = 57600)
  (h5 : final_amount = 72900) : 
  f = 0.125 := by
sorry

end annual_increase_fraction_l2941_294125


namespace lowest_sale_price_percentage_l2941_294171

theorem lowest_sale_price_percentage (list_price : ℝ) (max_regular_discount : ℝ) (summer_discount : ℝ) :
  list_price = 80 →
  max_regular_discount = 0.7 →
  summer_discount = 0.2 →
  (list_price * (1 - max_regular_discount) - list_price * summer_discount) / list_price = 0.1 := by
  sorry

end lowest_sale_price_percentage_l2941_294171


namespace parabolas_imply_right_triangle_l2941_294188

/-- Two parabolas intersecting the x-axis at the same non-origin point -/
def intersecting_parabolas (a b c : ℝ) : Prop :=
  ∃ x : ℝ, x ≠ 0 ∧ x^2 + 2*a*x + b^2 = 0 ∧ x^2 + 2*c*x - b^2 = 0

/-- The triangle formed by sides a, b, and c is right-angled -/
def right_angled_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2

theorem parabolas_imply_right_triangle (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_neq : a ≠ c) 
  (h_intersect : intersecting_parabolas a b c) : 
  right_angled_triangle a b c := by
  sorry

end parabolas_imply_right_triangle_l2941_294188


namespace smallest_sum_of_squared_ratios_l2941_294121

theorem smallest_sum_of_squared_ratios (c d : ℕ) (hc : c > 0) (hd : d > 0) :
  ∃ (min : ℚ), min = 2 ∧
  (((c + d : ℚ) / (c - d : ℚ))^2 + ((c - d : ℚ) / (c + d : ℚ))^2 ≥ min) ∧
  ∃ (c' d' : ℕ), c' > 0 ∧ d' > 0 ∧
  ((c' + d' : ℚ) / (c' - d' : ℚ))^2 + ((c' - d' : ℚ) / (c' + d' : ℚ))^2 = min :=
sorry

end smallest_sum_of_squared_ratios_l2941_294121


namespace fewer_blue_chairs_than_yellow_l2941_294109

/-- Represents the number of chairs of each color in Rodrigo's classroom -/
structure ClassroomChairs where
  red : ℕ
  yellow : ℕ
  blue : ℕ

def total_chairs (c : ClassroomChairs) : ℕ := c.red + c.yellow + c.blue

theorem fewer_blue_chairs_than_yellow (c : ClassroomChairs) 
  (h1 : c.red = 4)
  (h2 : c.yellow = 2 * c.red)
  (h3 : total_chairs c - 3 = 15) :
  c.yellow - c.blue = 2 := by
  sorry

end fewer_blue_chairs_than_yellow_l2941_294109


namespace geometric_properties_l2941_294196

/-- A geometric figure -/
structure Figure where
  -- Add necessary properties here
  mk :: -- Constructor

/-- Defines when two figures can overlap perfectly -/
def can_overlap (f1 f2 : Figure) : Prop :=
  sorry

/-- Defines congruence between two figures -/
def congruent (f1 f2 : Figure) : Prop :=
  sorry

/-- The area of a figure -/
def area (f : Figure) : ℝ :=
  sorry

/-- The perimeter of a figure -/
def perimeter (f : Figure) : ℝ :=
  sorry

theorem geometric_properties :
  (∀ f1 f2 : Figure, can_overlap f1 f2 → congruent f1 f2) ∧
  (∀ f1 f2 : Figure, congruent f1 f2 → area f1 = area f2) ∧
  (∃ f1 f2 : Figure, area f1 = area f2 ∧ ¬congruent f1 f2) ∧
  (∃ f1 f2 : Figure, perimeter f1 = perimeter f2 ∧ ¬congruent f1 f2) :=
sorry

end geometric_properties_l2941_294196


namespace stratified_sampling_medium_supermarkets_l2941_294101

theorem stratified_sampling_medium_supermarkets 
  (large : ℕ) 
  (medium : ℕ) 
  (small : ℕ) 
  (sample_size : ℕ) 
  (h1 : large = 200) 
  (h2 : medium = 400) 
  (h3 : small = 1400) 
  (h4 : sample_size = 100) :
  (medium : ℚ) * sample_size / (large + medium + small) = 20 := by
  sorry

end stratified_sampling_medium_supermarkets_l2941_294101


namespace buffy_whiskers_l2941_294161

/-- Represents the number of whiskers for each cat -/
structure CatWhiskers where
  juniper : ℕ
  puffy : ℕ
  scruffy : ℕ
  buffy : ℕ

/-- The conditions for the cat whiskers problem -/
def whiskerConditions (c : CatWhiskers) : Prop :=
  c.juniper = 12 ∧
  c.puffy = 3 * c.juniper ∧
  c.scruffy = 2 * c.puffy ∧
  c.buffy = (c.juniper + c.puffy + c.scruffy) / 3

/-- Theorem stating that under the given conditions, Buffy has 40 whiskers -/
theorem buffy_whiskers (c : CatWhiskers) :
  whiskerConditions c → c.buffy = 40 := by
  sorry

end buffy_whiskers_l2941_294161


namespace euler_polynomial_consecutive_composites_l2941_294150

theorem euler_polynomial_consecutive_composites :
  ∃ k : ℤ, ∀ j ∈ Finset.range 40,
    ∃ d : ℤ, d ∣ ((k + j)^2 + (k + j) + 41) ∧ d ≠ 1 ∧ d ≠ ((k + j)^2 + (k + j) + 41) := by
  sorry

end euler_polynomial_consecutive_composites_l2941_294150


namespace given_number_scientific_notation_l2941_294160

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coefficient_range : 1 ≤ coefficient ∧ coefficient < 10

/-- The given number in meters -/
def given_number : ℝ := 0.000000014

/-- The scientific notation representation of the given number -/
def scientific_representation : ScientificNotation :=
  { coefficient := 1.4
    exponent := -8
    coefficient_range := by sorry }

theorem given_number_scientific_notation :
  given_number = scientific_representation.coefficient * (10 : ℝ) ^ scientific_representation.exponent :=
by sorry

end given_number_scientific_notation_l2941_294160


namespace last_week_sales_l2941_294138

def chocolate_sales (week1 week2 week3 week4 week5 : ℕ) : Prop :=
  week1 = 75 ∧ week2 = 67 ∧ week3 = 75 ∧ week4 = 70

theorem last_week_sales (week5 : ℕ) :
  chocolate_sales 75 67 75 70 week5 →
  (75 + 67 + 75 + 70 + week5) / 5 = 71 →
  week5 = 68 := by
  sorry

end last_week_sales_l2941_294138


namespace betty_total_items_betty_total_cost_l2941_294102

/-- The number of slippers Betty ordered -/
def slippers : ℕ := 6

/-- The number of lipsticks Betty ordered -/
def lipsticks : ℕ := 4

/-- The number of hair colors Betty ordered -/
def hair_colors : ℕ := 8

/-- The cost of each slipper -/
def slipper_cost : ℚ := 5/2

/-- The cost of each lipstick -/
def lipstick_cost : ℚ := 5/4

/-- The cost of each hair color -/
def hair_color_cost : ℚ := 3

/-- The total amount Betty paid -/
def total_paid : ℚ := 44

/-- Theorem stating that Betty ordered 18 items in total -/
theorem betty_total_items : slippers + lipsticks + hair_colors = 18 := by
  sorry

/-- Theorem verifying the total cost matches the amount Betty paid -/
theorem betty_total_cost : 
  slippers * slipper_cost + lipsticks * lipstick_cost + hair_colors * hair_color_cost = total_paid := by
  sorry

end betty_total_items_betty_total_cost_l2941_294102


namespace modified_arithmetic_sum_l2941_294170

def arithmetic_sum (a₁ : ℕ) (aₙ : ℕ) (n : ℕ) : ℕ := n * (a₁ + aₙ) / 2

theorem modified_arithmetic_sum :
  3 * (arithmetic_sum 110 119 10) = 3435 :=
by sorry

end modified_arithmetic_sum_l2941_294170


namespace impossible_odd_sum_arrangement_l2941_294174

theorem impossible_odd_sum_arrangement : 
  ¬ ∃ (seq : Fin 2018 → ℕ), 
    (∀ i : Fin 2018, 1 ≤ seq i ∧ seq i ≤ 2018) ∧ 
    (∀ i : Fin 2018, seq i ≠ seq ((i + 1) % 2018)) ∧
    (∀ i : Fin 2018, Odd (seq i + seq ((i + 1) % 2018) + seq ((i + 2) % 2018))) :=
by
  sorry


end impossible_odd_sum_arrangement_l2941_294174


namespace trip_duration_proof_l2941_294185

/-- The battery life in standby mode (in hours) -/
def standby_life : ℝ := 210

/-- The rate at which the battery depletes while talking compared to standby mode -/
def talking_depletion_rate : ℝ := 35

/-- Calculates the total trip duration given the time spent talking -/
def total_trip_duration (talking_time : ℝ) : ℝ := 2 * talking_time

/-- Theorem stating that the total trip duration is 11 hours and 40 minutes -/
theorem trip_duration_proof :
  ∃ (talking_time : ℝ),
    talking_time > 0 ∧
    talking_time ≤ standby_life ∧
    talking_depletion_rate * (standby_life - talking_time) = talking_time ∧
    total_trip_duration talking_time = 11 + 40 / 60 :=
by sorry

end trip_duration_proof_l2941_294185


namespace shiny_igneous_fraction_l2941_294110

/-- Represents Cliff's rock collection -/
structure RockCollection where
  total : ℕ
  sedimentary : ℕ
  igneous : ℕ
  shinyIgneous : ℕ
  shinySedimentary : ℕ

/-- Properties of Cliff's rock collection -/
def isValidCollection (c : RockCollection) : Prop :=
  c.igneous = c.sedimentary / 2 ∧
  c.shinySedimentary = c.sedimentary / 5 ∧
  c.shinyIgneous = 30 ∧
  c.total = 270 ∧
  c.total = c.sedimentary + c.igneous

theorem shiny_igneous_fraction (c : RockCollection) 
  (h : isValidCollection c) : 
  (c.shinyIgneous : ℚ) / c.igneous = 1 / 3 := by
  sorry

end shiny_igneous_fraction_l2941_294110


namespace unique_solution_congruences_l2941_294130

theorem unique_solution_congruences :
  ∃! x : ℕ, x < 120 ∧
    (4 + x) % 8 = 3^2 % 8 ∧
    (6 + x) % 27 = 4^2 % 27 ∧
    (8 + x) % 125 = 6^2 % 125 ∧
    x = 37 := by
  sorry

end unique_solution_congruences_l2941_294130


namespace correct_average_calculation_l2941_294195

theorem correct_average_calculation (n : ℕ) (initial_avg : ℚ) (incorrect_num correct_num : ℚ) :
  n = 10 →
  initial_avg = 16 →
  incorrect_num = 25 →
  correct_num = 35 →
  (n : ℚ) * initial_avg + (correct_num - incorrect_num) = n * 17 := by
  sorry

end correct_average_calculation_l2941_294195


namespace casey_pumping_rate_l2941_294153

def corn_rows : ℕ := 4
def corn_plants_per_row : ℕ := 15
def water_per_corn_plant : ℚ := 1/2
def num_pigs : ℕ := 10
def water_per_pig : ℚ := 4
def num_ducks : ℕ := 20
def water_per_duck : ℚ := 1/4
def pumping_time : ℕ := 25

theorem casey_pumping_rate :
  let total_corn_plants := corn_rows * corn_plants_per_row
  let water_for_corn := (total_corn_plants : ℚ) * water_per_corn_plant
  let water_for_pigs := (num_pigs : ℚ) * water_per_pig
  let water_for_ducks := (num_ducks : ℚ) * water_per_duck
  let total_water := water_for_corn + water_for_pigs + water_for_ducks
  total_water / (pumping_time : ℚ) = 3 := by
  sorry

end casey_pumping_rate_l2941_294153


namespace quadratic_factorization_l2941_294163

theorem quadratic_factorization (x : ℝ) : 4 * x^2 - 8 * x + 4 = 4 * (x - 1)^2 := by
  sorry

end quadratic_factorization_l2941_294163


namespace grasshopper_return_to_origin_l2941_294156

def jump_length (n : ℕ) : ℕ := n

def is_horizontal (n : ℕ) : Bool :=
  n % 2 = 1

theorem grasshopper_return_to_origin :
  let horizontal_jumps := List.range 31 |>.filter is_horizontal |>.map jump_length
  let vertical_jumps := List.range 31 |>.filter (fun n => ¬ is_horizontal n) |>.map jump_length
  (List.sum horizontal_jumps = 0) ∧ (List.sum vertical_jumps = 0) := by
  sorry

end grasshopper_return_to_origin_l2941_294156


namespace seventh_term_of_geometric_sequence_l2941_294128

/-- Given a geometric sequence with first term a and common ratio r,
    the nth term is given by a * r^(n-1) -/
def geometric_sequence (a r : ℚ) (n : ℕ) : ℚ := a * r^(n-1)

/-- The common ratio of a geometric sequence can be found by dividing
    the second term by the first term -/
def common_ratio (a₁ a₂ : ℚ) : ℚ := a₂ / a₁

theorem seventh_term_of_geometric_sequence (a₁ a₂ : ℚ) 
  (h₁ : a₁ = 3)
  (h₂ : a₂ = -3/2) :
  geometric_sequence a₁ (common_ratio a₁ a₂) 7 = 3/64 := by
  sorry


end seventh_term_of_geometric_sequence_l2941_294128


namespace total_value_after_depreciation_l2941_294136

def calculate_depreciated_value (initial_value : ℝ) (depreciation_rate : ℝ) (years : ℕ) : ℝ :=
  initial_value * (1 - depreciation_rate) ^ years

theorem total_value_after_depreciation 
  (machine1_value : ℝ) (machine2_value : ℝ) (machine3_value : ℝ)
  (machine1_rate : ℝ) (machine2_rate : ℝ) (machine3_rate : ℝ)
  (years : ℕ) :
  machine1_value = 2500 →
  machine2_value = 3500 →
  machine3_value = 4500 →
  machine1_rate = 0.05 →
  machine2_rate = 0.07 →
  machine3_rate = 0.04 →
  years = 3 →
  (calculate_depreciated_value machine1_value machine1_rate years +
   calculate_depreciated_value machine2_value machine2_rate years +
   calculate_depreciated_value machine3_value machine3_rate years) = 8940 := by
  sorry

end total_value_after_depreciation_l2941_294136


namespace perfect_square_expression_l2941_294176

theorem perfect_square_expression : ∃ (x : ℝ), (12.86 * 12.86 + 12.86 * 0.28 + 0.14 * 0.14) = x^2 := by
  sorry

end perfect_square_expression_l2941_294176


namespace root_value_theorem_l2941_294159

theorem root_value_theorem (m : ℝ) : 2 * m^2 - 3 * m - 3 = 0 → 4 * m^2 - 6 * m + 2017 = 2023 := by
  sorry

end root_value_theorem_l2941_294159


namespace greatest_common_factor_72_180_270_l2941_294143

theorem greatest_common_factor_72_180_270 : Nat.gcd 72 (Nat.gcd 180 270) = 18 := by
  sorry

end greatest_common_factor_72_180_270_l2941_294143


namespace no_common_root_l2941_294175

theorem no_common_root (a b c d : ℝ) (h : 0 < a ∧ a < b ∧ b < c ∧ c < d) :
  ¬∃ x : ℝ, x^2 + b*x + c = 0 ∧ x^2 + a*x + d = 0 := by
  sorry

end no_common_root_l2941_294175


namespace problem_statement_l2941_294167

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = a * b) :
  (a + b ≥ 4) ∧ (a + 4 * b ≥ 9) ∧ (1 / a^2 + 2 / b^2 ≥ 2 / 3) := by
  sorry

end problem_statement_l2941_294167


namespace no_positive_rational_root_l2941_294113

theorem no_positive_rational_root : ¬∃ (q : ℚ), q > 0 ∧ q^3 - 10*q^2 + q - 2021 = 0 := by
  sorry

end no_positive_rational_root_l2941_294113


namespace prob_three_sixes_is_one_over_216_l2941_294119

/-- The number of faces on a standard die -/
def standard_die_faces : ℕ := 6

/-- The probability of rolling a specific number on a standard die -/
def prob_single_roll (n : ℕ) : ℚ := 1 / standard_die_faces

/-- The number of dice rolled -/
def num_dice : ℕ := 3

/-- The target sum we're aiming for -/
def target_sum : ℕ := 18

/-- The probability of rolling the target sum with the given number of dice -/
def prob_target_sum : ℚ := (prob_single_roll target_sum) ^ num_dice

theorem prob_three_sixes_is_one_over_216 : prob_target_sum = 1 / 216 := by
  sorry

end prob_three_sixes_is_one_over_216_l2941_294119


namespace inequality_proof_l2941_294104

theorem inequality_proof (n : ℕ) : (n - 1 : ℝ)^(n + 1) * (n + 1 : ℝ)^(n - 1) < n^(2 * n) := by
  sorry

end inequality_proof_l2941_294104


namespace mixture_composition_l2941_294199

theorem mixture_composition (water_percent_1 water_percent_2 mixture_percent : ℝ)
  (parts_1 : ℝ) (h1 : water_percent_1 = 0.20)
  (h2 : water_percent_2 = 0.35) (h3 : parts_1 = 10)
  (h4 : mixture_percent = 0.24285714285714285) : ∃ (parts_2 : ℝ),
  parts_2 = 4 ∧ 
  (water_percent_1 * parts_1 + water_percent_2 * parts_2) / (parts_1 + parts_2) = mixture_percent :=
by sorry

end mixture_composition_l2941_294199


namespace cos_alpha_value_l2941_294146

theorem cos_alpha_value (α : Real) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.sin (α - π / 6) = 1 / 3) : 
  Real.cos α = (2 * Real.sqrt 6 - 1) / 6 := by
  sorry

end cos_alpha_value_l2941_294146


namespace student_guinea_pig_difference_l2941_294107

/-- The number of fifth-grade classrooms -/
def num_classrooms : ℕ := 5

/-- The number of students in each classroom -/
def students_per_classroom : ℕ := 20

/-- The number of guinea pigs in each classroom -/
def guinea_pigs_per_classroom : ℕ := 3

/-- The total number of students in all classrooms -/
def total_students : ℕ := num_classrooms * students_per_classroom

/-- The total number of guinea pigs in all classrooms -/
def total_guinea_pigs : ℕ := num_classrooms * guinea_pigs_per_classroom

theorem student_guinea_pig_difference :
  total_students - total_guinea_pigs = 85 := by
  sorry

end student_guinea_pig_difference_l2941_294107


namespace parabola_point_ordering_l2941_294151

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = -2 * (x - 1)^2 + 3

/-- Point A lies on the parabola -/
def point_A (y₁ : ℝ) : Prop := parabola (-3) y₁

/-- Point B lies on the parabola -/
def point_B (y₂ : ℝ) : Prop := parabola 2 y₂

/-- Theorem stating the ordering of y₁, y₂, and 3 -/
theorem parabola_point_ordering (y₁ y₂ : ℝ) 
  (hA : point_A y₁) (hB : point_B y₂) : y₁ < y₂ ∧ y₂ < 3 := by
  sorry

end parabola_point_ordering_l2941_294151


namespace light_glow_duration_l2941_294141

/-- The number of times the light glowed between 1:57:58 and 3:20:47 am -/
def glow_count : ℝ := 292.29411764705884

/-- The total time in seconds between 1:57:58 am and 3:20:47 am -/
def total_time : ℕ := 4969

/-- The duration of each light glow in seconds -/
def glow_duration : ℕ := 17

theorem light_glow_duration :
  Int.floor (total_time / glow_count) = glow_duration := by sorry

end light_glow_duration_l2941_294141


namespace total_spent_equals_1150_l2941_294197

-- Define the quantities of toys
def elder_action_figures : ℕ := 60
def younger_action_figures : ℕ := 3 * elder_action_figures
def cars : ℕ := 20
def stuffed_animals : ℕ := 10

-- Define the prices of toys
def elder_action_figure_price : ℕ := 5
def younger_action_figure_price : ℕ := 4
def car_price : ℕ := 3
def stuffed_animal_price : ℕ := 7

-- Define the total cost function
def total_cost : ℕ :=
  elder_action_figures * elder_action_figure_price +
  younger_action_figures * younger_action_figure_price +
  cars * car_price +
  stuffed_animals * stuffed_animal_price

-- Theorem statement
theorem total_spent_equals_1150 : total_cost = 1150 := by
  sorry

end total_spent_equals_1150_l2941_294197


namespace derivative_lg_l2941_294189

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem derivative_lg (x : ℝ) (h : x > 0) :
  deriv lg x = 1 / (x * Real.log 10) :=
sorry

end derivative_lg_l2941_294189


namespace sufficient_not_necessary_condition_l2941_294118

-- Define the conditions
def p (x : ℝ) : Prop := x^2 - 4*x - 5 ≤ 0
def q (x a : ℝ) : Prop := |x - 3| < a ∧ a > 0

-- Define the solution set of p
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 5}

-- Define the set B
def B (a : ℝ) : Set ℝ := {x | 3 - a < x ∧ x < 3 + a}

-- Theorem statement
theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ x, p x → q x a) ∧ (∃ x, q x a ∧ ¬p x) → a > 4 :=
sorry

end sufficient_not_necessary_condition_l2941_294118


namespace circle_area_through_points_l2941_294126

/-- The area of a circle with center P(2, -5) passing through Q(-7, 6) is 202π. -/
theorem circle_area_through_points :
  let P : ℝ × ℝ := (2, -5)
  let Q : ℝ × ℝ := (-7, 6)
  let r : ℝ := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  (π * r^2) = 202 * π := by sorry

end circle_area_through_points_l2941_294126


namespace inequality_solution_set_l2941_294114

theorem inequality_solution_set (x : ℝ) (h : x ≠ 3) :
  (2 * x - 1) / (x - 3) ≥ 1 ↔ x > 3 ∨ x ≤ -2 := by
  sorry

end inequality_solution_set_l2941_294114


namespace cashew_nut_purchase_l2941_294183

/-- Prove that given the conditions of the nut purchase problem, the number of kilos of cashew nuts bought is 3. -/
theorem cashew_nut_purchase (cashew_price peanut_price peanut_amount total_weight avg_price : ℝ) 
  (h1 : cashew_price = 210)
  (h2 : peanut_price = 130)
  (h3 : peanut_amount = 2)
  (h4 : total_weight = 5)
  (h5 : avg_price = 178) :
  (total_weight - peanut_amount) = 3 := by
  sorry


end cashew_nut_purchase_l2941_294183


namespace lcm_gcf_problem_l2941_294135

theorem lcm_gcf_problem (n m : ℕ+) 
  (h1 : Nat.lcm n m = 56)
  (h2 : Nat.gcd n m = 10)
  (h3 : n = 40) :
  m = 14 := by
  sorry

end lcm_gcf_problem_l2941_294135


namespace solution_set_f_geq_4_min_value_f_l2941_294148

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2*x - 3| + |x - 5|

-- Theorem for the solution set of f(x) ≥ 4
theorem solution_set_f_geq_4 :
  {x : ℝ | f x ≥ 4} = {x : ℝ | x ≥ 2 ∨ x ≤ 4/3} :=
by sorry

-- Theorem for the minimum value of f(x)
theorem min_value_f :
  ∃ (x : ℝ), f x = 7/2 ∧ ∀ (y : ℝ), f y ≥ 7/2 :=
by sorry

end solution_set_f_geq_4_min_value_f_l2941_294148


namespace book_arrangement_combinations_l2941_294100

-- Define the number of each type of book
def geometry_books : ℕ := 4
def number_theory_books : ℕ := 5

-- Define the total number of books
def total_books : ℕ := geometry_books + number_theory_books

-- Define the number of remaining spots after placing the first geometry book
def remaining_spots : ℕ := total_books - 1

-- Define the number of remaining geometry books to place
def remaining_geometry_books : ℕ := geometry_books - 1

-- Theorem statement
theorem book_arrangement_combinations :
  (remaining_spots.choose remaining_geometry_books) = 56 := by
  sorry

end book_arrangement_combinations_l2941_294100


namespace janes_change_is_correct_l2941_294124

/-- The change Jane receives when buying an apple -/
def janes_change (apple_price : ℚ) (paid_amount : ℚ) : ℚ :=
  paid_amount - apple_price

/-- Theorem: Jane receives $4.25 in change -/
theorem janes_change_is_correct : 
  janes_change 0.75 5.00 = 4.25 := by
  sorry

end janes_change_is_correct_l2941_294124


namespace lucille_house_height_difference_l2941_294127

/-- Proves that Lucille's house is 9.32 feet shorter than the average height of all houses. -/
theorem lucille_house_height_difference :
  let lucille_height : ℝ := 80
  let neighbor1_height : ℝ := 70.5
  let neighbor2_height : ℝ := 99.3
  let neighbor3_height : ℝ := 84.2
  let neighbor4_height : ℝ := 112.6
  let total_height : ℝ := lucille_height + neighbor1_height + neighbor2_height + neighbor3_height + neighbor4_height
  let average_height : ℝ := total_height / 5
  average_height - lucille_height = 9.32 := by
  sorry

#eval (80 + 70.5 + 99.3 + 84.2 + 112.6) / 5 - 80

end lucille_house_height_difference_l2941_294127


namespace sqrt_square_abs_sqrt_neg_nine_squared_l2941_294172

theorem sqrt_square_abs (x : ℝ) : Real.sqrt (x^2) = |x| := by sorry

theorem sqrt_neg_nine_squared : Real.sqrt ((-9)^2) = 9 := by sorry

end sqrt_square_abs_sqrt_neg_nine_squared_l2941_294172


namespace traffic_light_probability_l2941_294108

/-- Represents the duration of traffic light phases in seconds -/
structure TrafficLightCycle where
  greenDuration : ℕ
  redDuration : ℕ

/-- Calculates the probability of waiting at least a given time in a traffic light cycle -/
def waitingProbability (cycle : TrafficLightCycle) (minWaitTime : ℕ) : ℚ :=
  let totalDuration := cycle.greenDuration + cycle.redDuration
  let waitInterval := cycle.redDuration - minWaitTime
  waitInterval / totalDuration

theorem traffic_light_probability (cycle : TrafficLightCycle) 
    (h1 : cycle.greenDuration = 40)
    (h2 : cycle.redDuration = 50) :
    waitingProbability cycle 20 = 1/3 := by
  sorry

end traffic_light_probability_l2941_294108


namespace inequality_not_always_satisfied_l2941_294158

theorem inequality_not_always_satisfied :
  ∃ (p q r : ℝ), p < 1 ∧ q < 2 ∧ r < 3 ∧ p^2 + 2*q*r ≥ 5 := by
  sorry

end inequality_not_always_satisfied_l2941_294158


namespace task_force_combinations_l2941_294157

theorem task_force_combinations (independents greens : ℕ) 
  (h1 : independents = 10) (h2 : greens = 7) : 
  (Nat.choose independents 4) * (Nat.choose greens 3) = 7350 := by
  sorry

end task_force_combinations_l2941_294157


namespace find_n_l2941_294105

theorem find_n (m n : ℕ) (h1 : Nat.lcm m n = 690) (h2 : ¬3 ∣ n) (h3 : ¬2 ∣ m) : n = 230 := by
  sorry

end find_n_l2941_294105


namespace cos_135_degrees_l2941_294131

theorem cos_135_degrees : Real.cos (135 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end cos_135_degrees_l2941_294131


namespace remainder_8_pow_2012_mod_10_l2941_294186

/-- Definition of exponentiation --/
def pow (a : ℕ) (n : ℕ) : ℕ := (a : ℕ) ^ n

/-- The remainder when 8^2012 is divided by 10 --/
theorem remainder_8_pow_2012_mod_10 : pow 8 2012 % 10 = 2 := by sorry

end remainder_8_pow_2012_mod_10_l2941_294186


namespace elvins_internet_charge_l2941_294129

/-- Proves that the fixed monthly charge for internet service is $6 given the conditions of Elvin's telephone bills. -/
theorem elvins_internet_charge (january_bill february_bill : ℕ) 
  (h1 : january_bill = 48)
  (h2 : february_bill = 90)
  (fixed_charge : ℕ) (january_calls february_calls : ℕ)
  (h3 : february_calls = 2 * january_calls)
  (h4 : january_bill = fixed_charge + january_calls)
  (h5 : february_bill = fixed_charge + february_calls) :
  fixed_charge = 6 := by
  sorry

end elvins_internet_charge_l2941_294129


namespace incorrect_propositions_l2941_294155

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines and between a line and a plane
variable (parallel_line : Line → Line → Prop)
variable (parallel_plane : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)

-- Define the intersection of two planes
variable (intersection : Plane → Plane → Line)

theorem incorrect_propositions :
  ∃ (l m : Line) (α β : Plane),
    -- Proposition A
    ¬(parallel_line l m ∧ contained_in m α → parallel_plane l α) ∧
    -- Proposition B
    ¬(parallel_plane l α ∧ parallel_plane m α → parallel_line l m) ∧
    -- Proposition C
    ¬(parallel_line l m ∧ parallel_plane m α → parallel_plane l α) :=
by sorry

end incorrect_propositions_l2941_294155


namespace initial_marbles_equation_l2941_294120

/-- The number of marbles Connie had initially -/
def initial_marbles : ℕ := sorry

/-- The number of marbles Connie gave to Juan -/
def marbles_given : ℕ := 73

/-- The number of marbles Connie has left -/
def marbles_left : ℕ := 70

/-- Theorem stating that the initial number of marbles is equal to
    the sum of marbles given away and marbles left -/
theorem initial_marbles_equation : initial_marbles = marbles_given + marbles_left := by
  sorry

end initial_marbles_equation_l2941_294120


namespace common_root_quadratic_equations_l2941_294111

theorem common_root_quadratic_equations (a : ℝ) :
  (∃ x : ℝ, a^2 * x^2 + a * x - 1 = 0 ∧ x^2 - a * x - a^2 = 0) →
  (a = (-1 + Real.sqrt 5) / 2 ∨ a = (-1 - Real.sqrt 5) / 2 ∨
   a = (1 + Real.sqrt 5) / 2 ∨ a = (1 - Real.sqrt 5) / 2) :=
by sorry

end common_root_quadratic_equations_l2941_294111


namespace A_minus_2B_y_value_when_independent_l2941_294140

-- Define the expressions A and B
def A (x y : ℝ) : ℝ := 4 * x^2 - x * y + 2 * y
def B (x y : ℝ) : ℝ := 2 * x^2 - x * y + x

-- Theorem 1: A - 2B = xy - 2x + 2y
theorem A_minus_2B (x y : ℝ) : A x y - 2 * B x y = x * y - 2 * x + 2 * y := by sorry

-- Theorem 2: If A - 2B is independent of x, then y = 2
theorem y_value_when_independent (y : ℝ) : 
  (∀ x : ℝ, A x y - 2 * B x y = A 0 y - 2 * B 0 y) → y = 2 := by sorry

end A_minus_2B_y_value_when_independent_l2941_294140


namespace smallest_x_multiple_of_53_l2941_294142

theorem smallest_x_multiple_of_53 : 
  ∃ (x : ℕ), x > 0 ∧ 
  (∀ (y : ℕ), y > 0 → (3 * y + 28)^2 % 53 = 0 → x ≤ y) ∧ 
  (3 * x + 28)^2 % 53 = 0 ∧ 
  x = 26 := by
sorry

end smallest_x_multiple_of_53_l2941_294142


namespace fraction_equation_solution_l2941_294154

theorem fraction_equation_solution :
  ∃ (x : ℚ), x ≠ 3 ∧ x ≠ -2 ∧ (x + 4) / (x - 3) = (x - 2) / (x + 2) ∧ x = -2 / 11 := by
  sorry

end fraction_equation_solution_l2941_294154


namespace at_least_one_greater_than_one_l2941_294184

theorem at_least_one_greater_than_one (a b : ℝ) :
  a + b > 2 → max a b > 1 := by
  sorry

end at_least_one_greater_than_one_l2941_294184
