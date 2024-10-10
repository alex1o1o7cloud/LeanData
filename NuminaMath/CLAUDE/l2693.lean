import Mathlib

namespace height_comparison_l2693_269331

theorem height_comparison (a b : ℝ) (h : a = b * 0.6) :
  (b - a) / a = 2 / 3 := by
  sorry

end height_comparison_l2693_269331


namespace intersection_when_a_is_one_range_of_a_when_union_is_real_l2693_269346

def A (a : ℝ) : Set ℝ := {x | a - 4 < x ∧ x < a + 4}
def B : Set ℝ := {x | x < -1 ∨ x > 5}

theorem intersection_when_a_is_one :
  A 1 ∩ B = {x | -3 < x ∧ x < -1} := by sorry

theorem range_of_a_when_union_is_real :
  (∃ a, A a ∪ B = Set.univ) ↔ ∃ a, 1 < a ∧ a < 3 := by sorry

end intersection_when_a_is_one_range_of_a_when_union_is_real_l2693_269346


namespace binomial_15_5_l2693_269308

theorem binomial_15_5 : Nat.choose 15 5 = 3003 := by
  sorry

end binomial_15_5_l2693_269308


namespace shaded_area_percentage_l2693_269382

/-- The size of the square grid -/
def gridSize : ℕ := 6

/-- The number of shaded squares -/
def shadedSquares : ℕ := 16

/-- The total number of squares in the grid -/
def totalSquares : ℕ := gridSize * gridSize

/-- The percentage of shaded area -/
def shadedPercentage : ℚ := (shadedSquares : ℚ) / (totalSquares : ℚ) * 100

theorem shaded_area_percentage :
  shadedPercentage = 4444 / 10000 := by sorry

end shaded_area_percentage_l2693_269382


namespace snow_difference_l2693_269361

def mrs_hilt_snow : ℕ := 29
def brecknock_snow : ℕ := 17

theorem snow_difference : mrs_hilt_snow - brecknock_snow = 12 := by
  sorry

end snow_difference_l2693_269361


namespace gcd_of_B_is_two_l2693_269356

def B : Set ℕ := {n | ∃ y : ℕ, n = (y - 1) + y + (y + 1) + (y + 2) ∧ y > 0}

theorem gcd_of_B_is_two : 
  ∃ d : ℕ, d > 0 ∧ (∀ b ∈ B, d ∣ b) ∧ (∀ m : ℕ, m > 0 → (∀ b ∈ B, m ∣ b) → m ≤ d) ∧ d = 2 := by
  sorry

end gcd_of_B_is_two_l2693_269356


namespace pizza_slices_left_l2693_269354

theorem pizza_slices_left (total_slices : ℕ) (eaten_slices : ℕ) (h1 : total_slices = 32) (h2 : eaten_slices = 25) :
  total_slices - eaten_slices = 7 := by
  sorry

end pizza_slices_left_l2693_269354


namespace unique_valid_grid_l2693_269381

/-- Represents a 3x3 grid with letters A, B, and C -/
def Grid := Fin 3 → Fin 3 → Fin 3

/-- Checks if a row contains exactly one of each letter -/
def valid_row (g : Grid) (row : Fin 3) : Prop :=
  ∀ letter : Fin 3, ∃! col : Fin 3, g row col = letter

/-- Checks if a column contains exactly one of each letter -/
def valid_column (g : Grid) (col : Fin 3) : Prop :=
  ∀ letter : Fin 3, ∃! row : Fin 3, g row col = letter

/-- Checks if the primary diagonal contains exactly one of each letter -/
def valid_diagonal (g : Grid) : Prop :=
  ∀ letter : Fin 3, ∃! i : Fin 3, g i i = letter

/-- Checks if A is in the upper left corner -/
def a_in_corner (g : Grid) : Prop := g 0 0 = 0

/-- Checks if the grid is valid according to all conditions -/
def valid_grid (g : Grid) : Prop :=
  (∀ row : Fin 3, valid_row g row) ∧
  (∀ col : Fin 3, valid_column g col) ∧
  valid_diagonal g ∧
  a_in_corner g

/-- The main theorem: there is exactly one valid grid arrangement -/
theorem unique_valid_grid : ∃! g : Grid, valid_grid g :=
  sorry

end unique_valid_grid_l2693_269381


namespace good_number_implies_prime_l2693_269341

/-- A positive integer b is "good for a" if C(an, b) - 1 is divisible by an + 1 for all positive integers n such that an ≥ b -/
def is_good_for (a b : ℕ+) : Prop :=
  ∀ n : ℕ+, a * n ≥ b → (Nat.choose (a * n) b - 1) % (a * n + 1) = 0

theorem good_number_implies_prime (a b : ℕ+) 
  (h1 : is_good_for a b)
  (h2 : ¬ is_good_for a (b + 2)) :
  Nat.Prime (b + 1) :=
sorry

end good_number_implies_prime_l2693_269341


namespace john_guests_correct_l2693_269300

/-- The number of guests John wants for his wedding. -/
def john_guests : ℕ := 50

/-- The venue cost for the wedding. -/
def venue_cost : ℕ := 10000

/-- The cost per guest for the wedding. -/
def cost_per_guest : ℕ := 500

/-- The total cost of the wedding if John's wife gets her way. -/
def total_cost : ℕ := 50000

/-- Theorem stating that the number of guests John wants is correct. -/
theorem john_guests_correct :
  venue_cost + cost_per_guest * (john_guests + (60 * john_guests) / 100) = total_cost :=
by sorry

end john_guests_correct_l2693_269300


namespace money_left_after_distributions_and_donations_l2693_269345

def total_income : ℝ := 1200000

def children_share : ℝ := 0.2
def wife_share : ℝ := 0.3
def donation_rate : ℝ := 0.05
def num_children : ℕ := 3

theorem money_left_after_distributions_and_donations :
  let amount_to_children := children_share * total_income * num_children
  let amount_to_wife := wife_share * total_income
  let remaining_before_donation := total_income - (amount_to_children + amount_to_wife)
  let donation_amount := donation_rate * remaining_before_donation
  total_income - (amount_to_children + amount_to_wife + donation_amount) = 114000 := by
  sorry

end money_left_after_distributions_and_donations_l2693_269345


namespace jersey_profit_is_152_l2693_269304

/-- The amount of money made from selling jerseys during a game -/
def money_from_jerseys (profit_per_jersey : ℕ) (jerseys_sold : ℕ) : ℕ :=
  profit_per_jersey * jerseys_sold

/-- Theorem stating that the money made from selling jerseys is $152 -/
theorem jersey_profit_is_152 :
  let profit_per_jersey : ℕ := 76
  let profit_per_tshirt : ℕ := 204
  let tshirts_sold : ℕ := 158
  let jerseys_sold : ℕ := 2
  money_from_jerseys profit_per_jersey jerseys_sold = 152 := by
sorry

end jersey_profit_is_152_l2693_269304


namespace weighted_average_fish_per_day_l2693_269333

-- Define the daily catch for each person
def aang_catch : List Nat := [5, 7, 9]
def sokka_catch : List Nat := [8, 5, 6]
def toph_catch : List Nat := [10, 12, 8]
def zuko_catch : List Nat := [6, 7, 10]

-- Define the number of people and days
def num_people : Nat := 4
def num_days : Nat := 3

-- Define the total fish caught by the group
def total_fish : Nat := aang_catch.sum + sokka_catch.sum + toph_catch.sum + zuko_catch.sum

-- Define the total days fished by the group
def total_days : Nat := num_people * num_days

-- Theorem to prove
theorem weighted_average_fish_per_day :
  (total_fish : Rat) / total_days = 93/12 := by sorry

end weighted_average_fish_per_day_l2693_269333


namespace imaginary_part_of_complex_expression_l2693_269395

theorem imaginary_part_of_complex_expression : 
  let z : ℂ := 1 - Complex.I
  let expression : ℂ := z^2 + 2/z
  Complex.im expression = -1 := by sorry

end imaginary_part_of_complex_expression_l2693_269395


namespace floor_negative_seven_fourths_l2693_269339

theorem floor_negative_seven_fourths :
  ⌊(-7 : ℝ) / 4⌋ = -2 := by sorry

end floor_negative_seven_fourths_l2693_269339


namespace prove_first_divisor_l2693_269309

def least_number : ℕ := 1394

def first_divisor : ℕ := 6

theorem prove_first_divisor :
  (least_number % first_divisor = 14) ∧
  (2535 % first_divisor = 1929) ∧
  (40 % first_divisor = 34) :=
by sorry

end prove_first_divisor_l2693_269309


namespace staff_age_l2693_269340

theorem staff_age (num_students : ℕ) (student_avg_age : ℝ) (new_avg_age : ℝ) : 
  num_students = 32 →
  student_avg_age = 16 →
  new_avg_age = student_avg_age + 1 →
  (num_students : ℝ) * student_avg_age + (num_students + 1 : ℝ) * new_avg_age = (num_students + 1 : ℝ) * 49 := by
  sorry

end staff_age_l2693_269340


namespace lcm_of_36_and_100_l2693_269316

theorem lcm_of_36_and_100 : Nat.lcm 36 100 = 900 := by
  sorry

end lcm_of_36_and_100_l2693_269316


namespace solve_cost_problem_l2693_269332

def cost_problem (shirt_cost jacket_cost carrie_payment : ℕ) 
                 (num_shirts num_pants num_jackets : ℕ) : Prop :=
  let total_cost := 2 * carrie_payment
  let pants_cost := (total_cost - num_shirts * shirt_cost - num_jackets * jacket_cost) / num_pants
  pants_cost = 18

theorem solve_cost_problem :
  cost_problem 8 60 94 4 2 2 := by
  sorry

end solve_cost_problem_l2693_269332


namespace athlete_heartbeats_l2693_269318

/-- Calculates the total number of heartbeats for an athlete jogging a given distance -/
def total_heartbeats (heart_rate : ℕ) (distance : ℕ) (pace : ℕ) : ℕ :=
  heart_rate * distance * pace

/-- Proves that an athlete jogging 15 miles at 8 minutes per mile with a heart rate of 120 bpm will have 14400 total heartbeats -/
theorem athlete_heartbeats :
  total_heartbeats 120 15 8 = 14400 := by
  sorry

#eval total_heartbeats 120 15 8

end athlete_heartbeats_l2693_269318


namespace diamond_count_l2693_269311

/-- The number of rubies in the chest -/
def rubies : ℕ := 377

/-- The difference between the number of diamonds and rubies -/
def diamond_ruby_difference : ℕ := 44

/-- The number of diamonds in the chest -/
def diamonds : ℕ := rubies + diamond_ruby_difference

theorem diamond_count : diamonds = 421 := by
  sorry

end diamond_count_l2693_269311


namespace stating_investment_plans_count_l2693_269379

/-- Represents the number of cities available for investment --/
def num_cities : ℕ := 4

/-- Represents the number of projects to be distributed --/
def num_projects : ℕ := 3

/-- Represents the maximum number of projects allowed in a single city --/
def max_projects_per_city : ℕ := 2

/-- 
Calculates the number of ways to distribute distinct projects among cities,
with a limit on the number of projects per city.
--/
def investment_plans (cities : ℕ) (projects : ℕ) (max_per_city : ℕ) : ℕ := 
  sorry

/-- 
Theorem stating that the number of investment plans 
for the given conditions is 60.
--/
theorem investment_plans_count : 
  investment_plans num_cities num_projects max_projects_per_city = 60 := by
  sorry

end stating_investment_plans_count_l2693_269379


namespace a_zero_iff_multiple_of_ten_sum_a_1_to_2005_l2693_269355

def a (n : ℕ+) : ℕ :=
  (7 * n.val) % 10

theorem a_zero_iff_multiple_of_ten (n : ℕ+) : a n = 0 ↔ 10 ∣ n.val := by
  sorry

theorem sum_a_1_to_2005 : (Finset.range 2005).sum (λ i => a ⟨i + 1, Nat.succ_pos i⟩) = 9025 := by
  sorry

end a_zero_iff_multiple_of_ten_sum_a_1_to_2005_l2693_269355


namespace cube_volume_from_edge_sum_l2693_269326

/-- Given a cube where the sum of the lengths of all edges is 96 cm, 
    prove that its volume is 512 cubic centimeters. -/
theorem cube_volume_from_edge_sum (edge_sum : ℝ) (volume : ℝ) : 
  edge_sum = 96 → volume = (edge_sum / 12)^3 → volume = 512 := by sorry

end cube_volume_from_edge_sum_l2693_269326


namespace cone_base_diameter_l2693_269302

/-- A cone with surface area 3π and lateral surface that unfolds into a semicircle -/
structure Cone where
  /-- The radius of the base of the cone -/
  radius : ℝ
  /-- The slant height of the cone -/
  slant_height : ℝ
  /-- The lateral surface unfolds into a semicircle -/
  lateral_surface_semicircle : slant_height = 2 * radius
  /-- The surface area of the cone is 3π -/
  surface_area : π * radius^2 + π * radius * slant_height = 3 * π

/-- The diameter of the base of the cone is 2 -/
theorem cone_base_diameter (c : Cone) : 2 * c.radius = 2 := by
  sorry

end cone_base_diameter_l2693_269302


namespace divide_c_by_a_l2693_269327

theorem divide_c_by_a (a b c : ℝ) (h1 : a * b = 3) (h2 : b * c = 8/5) : c / a = 8/15 := by
  sorry

end divide_c_by_a_l2693_269327


namespace scooter_gain_percent_l2693_269386

theorem scooter_gain_percent : 
  let initial_cost : ℚ := 900
  let repair1 : ℚ := 150
  let repair2 : ℚ := 75
  let repair3 : ℚ := 225
  let selling_price : ℚ := 1800
  let total_cost : ℚ := initial_cost + repair1 + repair2 + repair3
  let gain : ℚ := selling_price - total_cost
  let gain_percent : ℚ := (gain / total_cost) * 100
  gain_percent = 33.33 := by sorry

end scooter_gain_percent_l2693_269386


namespace penthouse_units_l2693_269343

theorem penthouse_units (total_floors : ℕ) (regular_units : ℕ) (penthouse_floors : ℕ) (total_units : ℕ)
  (h1 : total_floors = 23)
  (h2 : regular_units = 12)
  (h3 : penthouse_floors = 2)
  (h4 : total_units = 256) :
  (total_units - (total_floors - penthouse_floors) * regular_units) / penthouse_floors = 2 := by
  sorry

end penthouse_units_l2693_269343


namespace function_periodicity_l2693_269335

/-- Given a > 0 and a function f satisfying the specified condition, 
    prove that f is periodic with period 2a -/
theorem function_periodicity 
  (a : ℝ) 
  (ha : a > 0) 
  (f : ℝ → ℝ) 
  (hf : ∀ x : ℝ, f (x + a) = 1/2 + Real.sqrt (f x - (f x)^2)) : 
  ∃ b : ℝ, b > 0 ∧ ∀ x : ℝ, f (x + b) = f x :=
sorry

end function_periodicity_l2693_269335


namespace range_of_a_l2693_269336

-- Define the sets P and Q
def P : Set ℝ := {x | x ≤ -3 ∨ x ≥ 0}
def Q (a : ℝ) : Set ℝ := {x | x ≤ a}

-- Define the conditions
def condition_not_p (x : ℝ) : Prop := -3 < x ∧ x < 0
def condition_not_q (x a : ℝ) : Prop := x > a

-- Define the relationship between q and p
def q_sufficient_not_necessary_for_p (a : ℝ) : Prop :=
  Q a ⊂ P ∧ Q a ≠ P

-- The main theorem
theorem range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, condition_not_p x → condition_not_q x a) →
  q_sufficient_not_necessary_for_p a →
  a ≤ -3 := by sorry

end range_of_a_l2693_269336


namespace egg_packing_problem_l2693_269393

theorem egg_packing_problem (initial_eggs : Nat) (eggs_per_carton : Nat) (broken_eggs : Nat) :
  initial_eggs = 1000 →
  eggs_per_carton = 12 →
  broken_eggs < 12 →
  ∃ (filled_cartons : Nat), (initial_eggs - broken_eggs) = filled_cartons * eggs_per_carton →
  broken_eggs = 4 := by
  sorry

end egg_packing_problem_l2693_269393


namespace abcd_addition_l2693_269358

theorem abcd_addition (a b c d : Nat) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a ≥ 1 ∧ a ≤ 9 ∧ b ≥ 0 ∧ b ≤ 9 ∧ c ≥ 0 ∧ c ≤ 9 ∧ d ≥ 0 ∧ d ≤ 9 →
  (1000 * a + 100 * b + 10 * c + d) + (1000 * a + 100 * b + 10 * c + d) = 5472 →
  d = 6 := by
sorry

end abcd_addition_l2693_269358


namespace hyperbola_center_l2693_269359

/-- The hyperbola is defined by the equation 9x^2 - 54x - 16y^2 + 128y - 400 = 0 -/
def hyperbola_equation (x y : ℝ) : Prop :=
  9 * x^2 - 54 * x - 16 * y^2 + 128 * y - 400 = 0

/-- The center of a hyperbola is the point (h, k) where h and k are the coordinates that make
    the equation symmetric about the vertical and horizontal axes passing through (h, k) -/
def is_center (h k : ℝ) : Prop :=
  ∀ x y, hyperbola_equation x y ↔ hyperbola_equation (2*h - x) y ∧ hyperbola_equation x (2*k - y)

/-- The center of the hyperbola defined by 9x^2 - 54x - 16y^2 + 128y - 400 = 0 is (3, 4) -/
theorem hyperbola_center : is_center 3 4 := by sorry

end hyperbola_center_l2693_269359


namespace log_inequality_condition_l2693_269322

theorem log_inequality_condition (x y : ℝ) :
  (∀ x y, x > 0 ∧ y > 0 ∧ Real.log x > Real.log y → x > y) ∧
  ¬(∀ x y, x > y → Real.log x > Real.log y) :=
by sorry

end log_inequality_condition_l2693_269322


namespace polynomial_factorization_l2693_269387

theorem polynomial_factorization (a b c : ℝ) : 
  a^4 * (b^3 - c^3) + b^4 * (c^3 - a^3) + c^4 * (a^3 - b^3) = 
  (a - b) * (b - c) * (c - a) * (a^6*b + a^6*c + b^6*a + b^6*c + c^6*a + c^6*b) :=
by sorry

end polynomial_factorization_l2693_269387


namespace sequence_representation_l2693_269397

theorem sequence_representation (a : ℕ → ℕ) 
  (h_increasing : ∀ k : ℕ, k ≥ 1 → a k < a (k + 1)) :
  ∀ N : ℕ, ∃ m p q x y : ℕ, 
    m > N ∧ 
    p ≠ q ∧ 
    x > 0 ∧ 
    y > 0 ∧ 
    a m = x * a p + y * a q :=
sorry

end sequence_representation_l2693_269397


namespace stock_price_decrease_l2693_269321

theorem stock_price_decrease (P : ℝ) (X : ℝ) : 
  P > 0 →
  1.20 * P * (1 - X) * 1.35 = 1.215 * P →
  X = 0.25 := by
sorry

end stock_price_decrease_l2693_269321


namespace union_of_A_and_B_l2693_269394

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | 0 < x ∧ x < 3}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | -1 < x ∧ x < 3} := by sorry

end union_of_A_and_B_l2693_269394


namespace f_min_max_values_g_negative_range_l2693_269323

noncomputable section

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := (a - 1/2) * x^2 + Real.log x

def g (a : ℝ) (x : ℝ) : ℝ := f a x - 2 * a * x

-- Define the interval [1/e, e]
def interval : Set ℝ := Set.Icc (1 / Real.exp 1) (Real.exp 1)

-- Theorem 1: Minimum and maximum values of f when a = -1/2
theorem f_min_max_values :
  let f_neg_half (x : ℝ) := f (-1/2) x
  (∀ x ∈ interval, f_neg_half x ≥ 1 - Real.exp 1 ^ 2) ∧
  (∃ x ∈ interval, f_neg_half x = 1 - Real.exp 1 ^ 2) ∧
  (∀ x ∈ interval, f_neg_half x ≤ -1/2 - 1/2 * Real.log 2) ∧
  (∃ x ∈ interval, f_neg_half x = -1/2 - 1/2 * Real.log 2) := by
  sorry

-- Theorem 2: Range of a for which g(x) < 0 holds for all x > 2
theorem g_negative_range :
  {a : ℝ | ∀ x > 2, g a x < 0} = Set.Iic (1/2) := by
  sorry

end

end f_min_max_values_g_negative_range_l2693_269323


namespace problem_solution_l2693_269337

theorem problem_solution (a b c d : ℕ) 
  (h : 342 * (a * b * c * d + a * b + a * d + c * d + 1) = 379 * (b * c * d + b + d)) :
  a * 1000 + b * 100 + c * 10 + d = 1949 := by
  sorry

end problem_solution_l2693_269337


namespace diophantine_equation_solutions_l2693_269380

theorem diophantine_equation_solutions (p q : ℕ) (hp : Prime p) (hq : Prime q) (hpq : p ≠ q) :
  let solutions := {(a, b) : ℕ × ℕ | (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / (p * q)}
  solutions = {
    (1 + p*q, p^2*q^2 + p*q),
    (p*(q + 1), p*q*(q + 1)),
    (q*(p + 1), p*q*(p + 1)),
    (2*p*q, 2*p*q),
    (p^2*q*(p + q), q^2 + p*q),
    (q^2 + p*q, p^2 + p*q),
    (p*q*(p + 1), q*(p + 1)),
    (p*q*(q + 1), p*(q + 1)),
    (p^2*q^2 + p*q, 1 + p*q)
  } := by sorry

end diophantine_equation_solutions_l2693_269380


namespace cookies_problem_l2693_269369

theorem cookies_problem (mona jasmine rachel : ℕ) : 
  mona = 20 →
  jasmine < mona →
  rachel = jasmine + 10 →
  mona + jasmine + rachel = 60 →
  jasmine = 15 := by
sorry

end cookies_problem_l2693_269369


namespace nancy_football_games_l2693_269328

/-- Nancy's football game attendance problem -/
theorem nancy_football_games 
  (total_games : ℕ) 
  (this_month_games : ℕ) 
  (next_month_games : ℕ) 
  (h1 : total_games = 24)
  (h2 : this_month_games = 9)
  (h3 : next_month_games = 7) :
  total_games - this_month_games - next_month_games = 8 := by
  sorry

end nancy_football_games_l2693_269328


namespace mersenne_prime_implies_exponent_prime_l2693_269370

theorem mersenne_prime_implies_exponent_prime (n : ℕ) : 
  Prime (2^n - 1) → Prime n := by
  sorry

end mersenne_prime_implies_exponent_prime_l2693_269370


namespace fraction_B_is_02_l2693_269391

-- Define the fractions of students receiving grades
def fraction_A : ℝ := 0.7
def fraction_A_or_B : ℝ := 0.9

-- Theorem statement
theorem fraction_B_is_02 : 
  fraction_A_or_B - fraction_A = 0.2 := by
  sorry

end fraction_B_is_02_l2693_269391


namespace trajectory_and_symmetry_l2693_269352

-- Define the fixed circle F
def F (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

-- Define the line L that the moving circle is tangent to
def L (x : ℝ) : Prop := x = -1

-- Define the trajectory C of the center P
def C (x y : ℝ) : Prop := y^2 = 8*x

-- Define symmetry about the line y = x - 1
def symmetric_about_line (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (x₁ + x₂)/2 - ((y₁ + y₂)/2 + 1) = 0 ∧ y₁ + y₂ = x₁ + x₂ - 2

theorem trajectory_and_symmetry :
  (∀ x y, C x y ↔ ∃ r, (∀ xf yf, F xf yf → (x - xf)^2 + (y - yf)^2 = (r + 1)^2) ∧
                       (∀ xl, L xl → |x - xl| = r)) ∧
  ¬(∃ x₁ y₁ x₂ y₂, C x₁ y₁ ∧ C x₂ y₂ ∧ symmetric_about_line x₁ y₁ x₂ y₂) :=
sorry

end trajectory_and_symmetry_l2693_269352


namespace arithmetic_sequence_ratio_l2693_269384

/-- For an arithmetic sequence with first term a₁ and common difference d,
    if the sum of the first 10 terms is 4 times the sum of the first 5 terms,
    then a₁/d = 1/2. -/
theorem arithmetic_sequence_ratio (a₁ d : ℝ) :
  let S : ℕ → ℝ := λ n => n * a₁ + (n * (n - 1) / 2) * d
  S 10 = 4 * S 5 → a₁ / d = 1 / 2 := by
  sorry

end arithmetic_sequence_ratio_l2693_269384


namespace store_earnings_calculation_l2693_269383

/-- Represents the earnings from a day's sale of drinks at a country store. -/
def store_earnings (cola_price : ℚ) (juice_price : ℚ) (water_price : ℚ) (sports_drink_price : ℚ)
                   (cola_sold : ℕ) (juice_sold : ℕ) (water_sold : ℕ) (sports_drink_sold : ℕ)
                   (sports_drink_paid : ℕ) : ℚ :=
  cola_price * cola_sold + juice_price * juice_sold + water_price * water_sold +
  sports_drink_price * sports_drink_paid

/-- Theorem stating the total earnings of the store given the specific conditions. -/
theorem store_earnings_calculation :
  let cola_price : ℚ := 3
  let juice_price : ℚ := 3/2
  let water_price : ℚ := 1
  let sports_drink_price : ℚ := 5/2
  let cola_sold : ℕ := 18
  let juice_sold : ℕ := 15
  let water_sold : ℕ := 30
  let sports_drink_sold : ℕ := 44
  let sports_drink_paid : ℕ := 22
  store_earnings cola_price juice_price water_price sports_drink_price
                 cola_sold juice_sold water_sold sports_drink_sold sports_drink_paid = 161.5 := by
  sorry


end store_earnings_calculation_l2693_269383


namespace grape_price_l2693_269315

theorem grape_price (G : ℝ) : 
  (8 * G + 11 * 55 = 1165) → G = 70 := by sorry

end grape_price_l2693_269315


namespace complex_equation_result_l2693_269301

theorem complex_equation_result (a b : ℝ) (i : ℂ) (h1 : i * i = -1) (h2 : (a + 4 * i) * i = b + i) : a - b = 5 := by
  sorry

end complex_equation_result_l2693_269301


namespace tournament_games_theorem_l2693_269349

/-- A single-elimination tournament with no ties -/
structure Tournament :=
  (num_teams : ℕ)
  (no_ties : Bool)

/-- The number of games required to declare a winner in a tournament -/
def games_to_winner (t : Tournament) : ℕ :=
  t.num_teams - 1

/-- Theorem: In a single-elimination tournament with 24 teams and no ties,
    the number of games required to declare a winner is 23 -/
theorem tournament_games_theorem (t : Tournament) 
  (h1 : t.num_teams = 24) (h2 : t.no_ties = true) : 
  games_to_winner t = 23 := by
  sorry

end tournament_games_theorem_l2693_269349


namespace car_uphill_speed_l2693_269375

/-- Proves that the uphill speed of a car is 30 km/hr given the specified conditions -/
theorem car_uphill_speed (V_up : ℝ) : 
  V_up > 0 →
  (100 / V_up + 50 / 60 : ℝ) = 150 / 36 →
  V_up = 30 := by
sorry

end car_uphill_speed_l2693_269375


namespace complex_root_magnitude_l2693_269324

theorem complex_root_magnitude (z : ℂ) (h : z^2 - z + 1 = 0) : Complex.abs z = 1 := by
  sorry

end complex_root_magnitude_l2693_269324


namespace train_length_l2693_269360

/-- The length of a train that passes a stationary man in 8 seconds and crosses a 270-meter platform in 20 seconds is 180 meters. -/
theorem train_length : ℝ → Prop :=
  fun L : ℝ =>
    (L / 8 = (L + 270) / 20) →
    L = 180

/-- Proof of the train length theorem -/
lemma train_length_proof : ∃ L : ℝ, train_length L :=
by
  sorry

end train_length_l2693_269360


namespace stratified_sampling_eleventh_grade_l2693_269371

/-- Represents the ratio of students in grades 10, 11, and 12 -/
def student_ratio : Fin 3 → ℕ
  | 0 => 4  -- 10th grade
  | 1 => 3  -- 11th grade
  | 2 => 3  -- 12th grade

/-- Total number of parts in the ratio -/
def total_ratio : ℕ := (student_ratio 0) + (student_ratio 1) + (student_ratio 2)

/-- Total sample size -/
def sample_size : ℕ := 50

/-- Calculates the number of students drawn from the 11th grade -/
def eleventh_grade_sample : ℕ := 
  (student_ratio 1 * sample_size) / total_ratio

theorem stratified_sampling_eleventh_grade :
  eleventh_grade_sample = 15 :=
sorry

end stratified_sampling_eleventh_grade_l2693_269371


namespace min_value_of_expression_l2693_269303

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^2 + 2*a*b - 3 = 0) :
  ∃ (k : ℝ), k = 2*a + b ∧ k ≥ 4 ∧ ∀ (x y : ℝ), x > 0 → y > 0 → x^2 + 2*x*y - 3 = 0 → 2*x + y ≥ k :=
sorry

end min_value_of_expression_l2693_269303


namespace crayons_in_drawer_l2693_269399

/-- The number of crayons initially in the drawer -/
def initial_crayons : ℕ := 9

/-- The number of crayons Benny added to the drawer -/
def added_crayons : ℕ := 3

/-- The total number of crayons in the drawer after Benny adds more -/
def total_crayons : ℕ := initial_crayons + added_crayons

theorem crayons_in_drawer : total_crayons = 12 := by
  sorry

end crayons_in_drawer_l2693_269399


namespace hyperbola_and_condition_implies_m_range_l2693_269307

/-- Represents a hyperbola equation -/
def is_hyperbola (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (m + 3) + y^2 / (m - 4) = 1

/-- Condition for all real x -/
def condition_for_all_x (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + m*x + m + 3 ≥ 0

/-- The range of m -/
def m_range (m : ℝ) : Prop :=
  -2 ≤ m ∧ m < 4

theorem hyperbola_and_condition_implies_m_range :
  ∀ m : ℝ, is_hyperbola m ∧ condition_for_all_x m → m_range m :=
sorry

end hyperbola_and_condition_implies_m_range_l2693_269307


namespace smallest_slope_for_tangent_circle_l2693_269344

def circle_u₁ (x y : ℝ) : Prop := x^2 + y^2 + 8*x - 18*y - 75 = 0
def circle_u₂ (x y : ℝ) : Prop := x^2 + y^2 - 8*x - 18*y + 135 = 0

def externally_tangent (x y r : ℝ) : Prop := (x - 4)^2 + (y - 9)^2 = (r + 2)^2
def internally_tangent (x y r : ℝ) : Prop := (x + 4)^2 + (y - 9)^2 = (10 - r)^2

def contains_center (b x y : ℝ) : Prop := y = b * x

theorem smallest_slope_for_tangent_circle :
  ∃ (n : ℝ), n > 0 ∧
    (∀ b : ℝ, b > 0 →
      (∃ x y r : ℝ, contains_center b x y ∧ externally_tangent x y r ∧ internally_tangent x y r) →
      b ≥ n) ∧
    n^2 = 61/24 :=
sorry

end smallest_slope_for_tangent_circle_l2693_269344


namespace committee_probability_l2693_269306

/-- The probability of selecting exactly 2 boys in a 6-person committee 
    randomly chosen from a group of 30 members (12 boys and 18 girls) -/
theorem committee_probability (total_members : ℕ) (boys : ℕ) (girls : ℕ) 
  (committee_size : ℕ) (h1 : total_members = 30) (h2 : boys = 12) 
  (h3 : girls = 18) (h4 : committee_size = 6) (h5 : total_members = boys + girls) :
  (Nat.choose boys 2 * Nat.choose girls 4) / Nat.choose total_members committee_size = 8078 / 23751 := by
sorry

end committee_probability_l2693_269306


namespace inequality_holds_l2693_269362

/-- The inequality holds for the given pairs of non-negative integers -/
theorem inequality_holds (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  ∀ k n : ℕ, 
    (1 + y^n / x^k ≥ (1 + y)^n / (1 + x)^k) ↔ 
    ((k = 0 ∧ n ≥ 0) ∨ 
     (k = 1 ∧ n = 0) ∨ 
     (k = 0 ∧ n = 0) ∨ 
     (k ≥ n - 1 ∧ n ≥ 1)) :=
by sorry

end inequality_holds_l2693_269362


namespace solution_to_system_l2693_269377

/-- The system of equations -/
def equation1 (x y : ℝ) : Prop := x^2*y - x*y^2 - 5*x + 5*y + 3 = 0
def equation2 (x y : ℝ) : Prop := x^3*y - x*y^3 - 5*x^2 + 5*y^2 + 15 = 0

/-- The theorem stating that (4, 1) is the solution to the system of equations -/
theorem solution_to_system : equation1 4 1 ∧ equation2 4 1 := by sorry

end solution_to_system_l2693_269377


namespace f_difference_l2693_269368

/-- The function f(x) as defined in the problem -/
def f (x : ℝ) : ℝ := x^6 + x^4 + 3*x^3 + 4*x^2 + 8*x

/-- Theorem stating that f(3) - f(-3) = 210 -/
theorem f_difference : f 3 - f (-3) = 210 := by
  sorry

end f_difference_l2693_269368


namespace butter_theorem_l2693_269392

def butter_problem (total_butter : ℝ) (chocolate_chip : ℝ) (peanut_butter : ℝ) (sugar : ℝ) (oatmeal : ℝ) (spilled : ℝ) : Prop :=
  let used_butter := chocolate_chip * total_butter + peanut_butter * total_butter + sugar * total_butter + oatmeal * total_butter
  let remaining_before_spill := total_butter - used_butter
  let remaining_after_spill := remaining_before_spill - spilled
  remaining_after_spill = 0.375

theorem butter_theorem : 
  butter_problem 15 (2/5) (1/6) (1/8) (1/4) 0.5 := by
  sorry

end butter_theorem_l2693_269392


namespace triangle_properties_l2693_269363

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Theorem statement
theorem triangle_properties (t : Triangle) : 
  -- 1. If A > B, then sin A > sin B
  (t.A > t.B → Real.sin t.A > Real.sin t.B) ∧ 
  -- 2. sin 2A = sin 2B does not necessarily imply isosceles
  ¬(Real.sin (2 * t.A) = Real.sin (2 * t.B) → t.a = t.b) ∧ 
  -- 3. a² + b² = c² does not necessarily imply isosceles
  ¬(t.a^2 + t.b^2 = t.c^2 → t.a = t.b) ∧ 
  -- 4. a² + b² > c² does not necessarily imply largest angle is obtuse
  ¬(t.a^2 + t.b^2 > t.c^2 → t.A > Real.pi/2 ∨ t.B > Real.pi/2 ∨ t.C > Real.pi/2) := by
sorry


end triangle_properties_l2693_269363


namespace sally_cards_total_l2693_269372

/-- The number of cards Sally has now is equal to the sum of her initial cards,
    the cards Dan gave her, and the cards she bought. -/
theorem sally_cards_total
  (initial : ℕ)  -- Sally's initial number of cards
  (from_dan : ℕ) -- Number of cards Dan gave Sally
  (bought : ℕ)   -- Number of cards Sally bought
  (h1 : initial = 27)
  (h2 : from_dan = 41)
  (h3 : bought = 20) :
  initial + from_dan + bought = 88 :=
by sorry

end sally_cards_total_l2693_269372


namespace triple_overlap_is_six_l2693_269385

/-- Represents a rectangular carpet with width and height in meters -/
structure Carpet where
  width : ℝ
  height : ℝ

/-- Represents the auditorium and the placement of carpets -/
structure Auditorium where
  width : ℝ
  height : ℝ
  carpet1 : Carpet
  carpet2 : Carpet
  carpet3 : Carpet

/-- Calculates the area of triple overlap given an auditorium setup -/
def tripleOverlapArea (a : Auditorium) : ℝ :=
  2 * 3

/-- Theorem stating that the triple overlap area is 6 square meters -/
theorem triple_overlap_is_six (a : Auditorium) 
    (h1 : a.width = 10 ∧ a.height = 10)
    (h2 : a.carpet1.width = 6 ∧ a.carpet1.height = 8)
    (h3 : a.carpet2.width = 6 ∧ a.carpet2.height = 6)
    (h4 : a.carpet3.width = 5 ∧ a.carpet3.height = 7) :
  tripleOverlapArea a = 6 := by
  sorry

end triple_overlap_is_six_l2693_269385


namespace f_one_values_l2693_269314

noncomputable section

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
axiom functional_equation : ∀ (x y : ℝ), f (x + y) = f x * f y
axiom not_constant_zero : ∃ (x : ℝ), f x ≠ 0
axiom exists_a : ∃ (a : ℝ), a ≠ 0 ∧ f a = 2

-- Theorem statement
theorem f_one_values : (f 1 = Real.sqrt 2) ∨ (f 1 = -Real.sqrt 2) :=
sorry

end

end f_one_values_l2693_269314


namespace gcf_lcm_sum_of_numbers_l2693_269398

def numbers : List Nat := [16, 32, 48]

theorem gcf_lcm_sum_of_numbers (A B : Nat) 
  (h1 : A = Nat.gcd 16 (Nat.gcd 32 48))
  (h2 : B = Nat.lcm 16 (Nat.lcm 32 48)) : 
  A + B = 112 := by
  sorry

end gcf_lcm_sum_of_numbers_l2693_269398


namespace chess_game_draw_probability_l2693_269348

theorem chess_game_draw_probability (p_win p_not_lose : ℝ) 
  (h_win : p_win = 2/5)
  (h_not_lose : p_not_lose = 9/10) :
  p_not_lose - p_win = 1/2 := by
  sorry

end chess_game_draw_probability_l2693_269348


namespace sum_of_powers_l2693_269347

theorem sum_of_powers : -2^2003 + (-2)^2004 + 2^2005 - 2^2006 = -3 * 2^2003 := by
  sorry

end sum_of_powers_l2693_269347


namespace problem_statement_l2693_269367

theorem problem_statement (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_not_all_equal : ¬(a = b ∧ b = c)) : 
  ((a - b)^2 + (b - c)^2 + (c - a)^2 ≠ 0) ∧ 
  (a > b ∨ a < b ∨ a = b) ∧
  (∃ (x y z : ℝ), x ≠ z ∧ y ≠ z ∧ x ≠ y) :=
by sorry

end problem_statement_l2693_269367


namespace inequalities_hold_l2693_269342

theorem inequalities_hold (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : 0 > c) (h4 : c > d) : 
  (a - d > b - c) ∧ (a * d^2 > b * c^2) := by
  sorry

end inequalities_hold_l2693_269342


namespace trajectory_equation_l2693_269338

-- Define the fixed circle C
def C (x y : ℝ) : Prop := x^2 + (y+3)^2 = 1

-- Define the line that M is tangent to
def L (y : ℝ) : Prop := y = 2

-- Define the moving circle M
def M (x y : ℝ) : Prop := ∃ (r : ℝ), r > 0 ∧ 
  (∀ (x' y' : ℝ), C x' y' → (x - x')^2 + (y - y')^2 = (1 + r)^2) ∧
  (∀ (y' : ℝ), L y' → |y - y'| = r)

-- State the theorem
theorem trajectory_equation :
  ∀ (x y : ℝ), M x y → x^2 = -12*y := by sorry

end trajectory_equation_l2693_269338


namespace books_sold_on_friday_l2693_269305

theorem books_sold_on_friday (initial_stock : ℕ) (monday : ℕ) (tuesday : ℕ) (wednesday : ℕ) (thursday : ℕ) (not_sold : ℕ)
  (h1 : initial_stock = 800)
  (h2 : monday = 60)
  (h3 : tuesday = 10)
  (h4 : wednesday = 20)
  (h5 : thursday = 44)
  (h6 : not_sold = 600) :
  initial_stock - not_sold - (monday + tuesday + wednesday + thursday) = 66 := by
  sorry

end books_sold_on_friday_l2693_269305


namespace CD_possible_values_l2693_269374

-- Define the points on the number line
def A : ℝ := -3
def B : ℝ := 6

-- Define the distances
def AC : ℝ := 8
def BD : ℝ := 2

-- Define the possible positions for C and D
def C1 : ℝ := A + AC
def C2 : ℝ := A - AC
def D1 : ℝ := B + BD
def D2 : ℝ := B - BD

-- Define the set of possible CD values
def CD_values : Set ℝ := {|C1 - D1|, |C1 - D2|, |C2 - D1|, |C2 - D2|}

-- Theorem statement
theorem CD_possible_values : CD_values = {3, 1, 19, 15} := by sorry

end CD_possible_values_l2693_269374


namespace youtube_time_is_17_minutes_l2693_269351

/-- The total time spent on YouTube per day -/
def total_youtube_time (num_videos : ℕ) (video_length : ℕ) (ad_time : ℕ) : ℕ :=
  num_videos * video_length + ad_time

/-- Theorem stating that the total time spent on YouTube is 17 minutes -/
theorem youtube_time_is_17_minutes :
  total_youtube_time 2 7 3 = 17 := by
  sorry

end youtube_time_is_17_minutes_l2693_269351


namespace julie_reading_problem_l2693_269365

/-- The number of pages in Julie's book -/
def total_pages : ℕ := 120

/-- The number of pages Julie read yesterday -/
def pages_yesterday : ℕ := 12

/-- The number of pages Julie read today -/
def pages_today : ℕ := 2 * pages_yesterday

/-- The number of pages remaining after Julie read yesterday and today -/
def remaining_pages : ℕ := total_pages - (pages_yesterday + pages_today)

theorem julie_reading_problem :
  (pages_yesterday = 12) ∧
  (total_pages = 120) ∧
  (pages_today = 2 * pages_yesterday) ∧
  (remaining_pages / 2 = 42) :=
by sorry

end julie_reading_problem_l2693_269365


namespace min_value_a_squared_l2693_269376

/-- In an acute-angled triangle ABC with sides a, b, c opposite angles A, B, C respectively,
    if b^2 * sin(C) = 4√2 * sin(B) and the area of triangle ABC is 8/3,
    then the minimum value of a^2 is 16√2/3. -/
theorem min_value_a_squared (a b c A B C : ℝ) : 
  0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2 →  -- Acute-angled triangle
  b^2 * Real.sin C = 4 * Real.sqrt 2 * Real.sin B →     -- Given condition
  (1/2) * b * c * Real.sin A = 8/3 →                    -- Area of triangle
  ∀ x, x^2 ≥ a^2 → x^2 ≥ (16 * Real.sqrt 2) / 3 :=      -- Minimum value of a^2
by sorry

end min_value_a_squared_l2693_269376


namespace inequality_proof_l2693_269396

theorem inequality_proof (a b c r : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hr : r > 1) : 
  (a / (b + c)) + (b / (c + a)) + (c / (a + b)) ≤ 
  (a^r / (b^r + c^r)) + (b^r / (c^r + a^r)) + (c^r / (a^r + b^r)) := by
  sorry

end inequality_proof_l2693_269396


namespace workshop_efficiency_l2693_269320

theorem workshop_efficiency (x : ℝ) : 
  (1500 / x - 1500 / (2.5 * x) = 18) → x = 50 :=
by
  sorry

end workshop_efficiency_l2693_269320


namespace marks_car_repair_cost_l2693_269390

/-- The total cost of fixing Mark's car -/
def total_cost (part_cost : ℕ) (num_parts : ℕ) (labor_rate : ℚ) (hours_worked : ℕ) : ℚ :=
  (part_cost * num_parts : ℚ) + labor_rate * (hours_worked * 60)

/-- Theorem stating that the total cost of fixing Mark's car is $220 -/
theorem marks_car_repair_cost :
  total_cost 20 2 0.5 6 = 220 := by
  sorry

end marks_car_repair_cost_l2693_269390


namespace fraction_value_l2693_269378

theorem fraction_value (m n : ℝ) (h : 1/m - 1/n = 6) : m * n / (m - n) = -1/6 := by
  sorry

end fraction_value_l2693_269378


namespace complex_sum_equals_i_l2693_269317

theorem complex_sum_equals_i : Complex.I^2 = -1 → (1 : ℂ) + Complex.I + Complex.I^2 = Complex.I :=
by
  sorry

end complex_sum_equals_i_l2693_269317


namespace system_solution_l2693_269325

theorem system_solution : 
  ∃! (s : Set (ℝ × ℝ)), s = {(12, 10), (-10, -12)} ∧ 
    ∀ (x y : ℝ), (x, y) ∈ s ↔ 
      ((3/2 : ℝ)^(x-y) - (2/3 : ℝ)^(x-y) = 65/36 ∧
       x*y - x + y = 118) := by
  sorry

end system_solution_l2693_269325


namespace pascal_triangle_count_l2693_269329

/-- Represents a row in Pascal's Triangle -/
def PascalRow := List Nat

/-- Generates the nth row of Pascal's Triangle -/
def generatePascalRow (n : Nat) : PascalRow :=
  sorry

/-- Counts the number of even integers in a given row -/
def countEvens (row : PascalRow) : Nat :=
  sorry

/-- Counts the number of integers that are multiples of 4 in a given row -/
def countMultiplesOfFour (row : PascalRow) : Nat :=
  sorry

/-- Theorem stating the count of even integers and multiples of 4 in the first 12 rows of Pascal's Triangle -/
theorem pascal_triangle_count :
  let rows := List.range 12
  let evenCount := rows.map (fun n => countEvens (generatePascalRow n)) |>.sum
  let multiples4Count := rows.map (fun n => countMultiplesOfFour (generatePascalRow n)) |>.sum
  ∃ (e m : Nat), evenCount = e ∧ multiples4Count = m :=
by
  sorry

end pascal_triangle_count_l2693_269329


namespace triangle_segment_length_l2693_269310

theorem triangle_segment_length : 
  ∀ (a b c h x : ℝ),
  a = 40 ∧ b = 90 ∧ c = 100 →
  a^2 = x^2 + h^2 →
  b^2 = (c - x)^2 + h^2 →
  c - x = 82.5 :=
by sorry

end triangle_segment_length_l2693_269310


namespace exists_remarkable_polygon_for_n_gt_4_l2693_269357

/-- A remarkable polygon is a grid polygon that is not a rectangle and can form a similar polygon from several of its copies. -/
structure RemarkablePolygon (n : ℕ) where
  cells : ℕ
  not_rectangle : cells ≠ 4
  can_form_similar : True  -- Simplified condition for similarity

/-- For all integers n > 4, there exists a remarkable polygon with n cells. -/
theorem exists_remarkable_polygon_for_n_gt_4 (n : ℕ) (h : n > 4) :
  ∃ (P : RemarkablePolygon n), P.cells = n :=
sorry

end exists_remarkable_polygon_for_n_gt_4_l2693_269357


namespace log_lower_bound_l2693_269364

/-- The number of distinct prime factors of a positive integer -/
def num_distinct_prime_factors (n : ℕ+) : ℕ :=
  (Nat.factors n).toFinset.card

/-- For any positive integer n, log(n) ≥ k * log(2), where k is the number of distinct prime factors of n -/
theorem log_lower_bound (n : ℕ+) :
  Real.log n ≥ (num_distinct_prime_factors n : ℝ) * Real.log 2 := by
  sorry


end log_lower_bound_l2693_269364


namespace p_true_q_false_l2693_269373

-- Define the quadratic equation
def hasRealRoots (m : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + x - m = 0

-- Proposition p
theorem p_true : ∀ m : ℝ, m > 0 → hasRealRoots m :=
sorry

-- Converse of p (proposition q) is false
theorem q_false : ∃ m : ℝ, m ≥ -1 ∧ m ≤ 0 ∧ hasRealRoots m :=
sorry

end p_true_q_false_l2693_269373


namespace quadratic_always_nonnegative_implies_a_range_l2693_269313

theorem quadratic_always_nonnegative_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + 2*a*x + 1 ≥ 0) → -1 ≤ a ∧ a ≤ 1 := by
  sorry

end quadratic_always_nonnegative_implies_a_range_l2693_269313


namespace total_students_l2693_269350

theorem total_students (boys girls : ℕ) (h1 : boys * 5 = girls * 6) (h2 : girls = 200) :
  boys + girls = 440 := by
  sorry

end total_students_l2693_269350


namespace convex_polygon_division_impossibility_l2693_269389

-- Define a polygon
def Polygon : Type := List (ℝ × ℝ)

-- Define a function to check if a polygon is convex
def isConvex (p : Polygon) : Prop := sorry

-- Define a function to check if a quadrilateral is non-convex
def isNonConvexQuadrilateral (q : Polygon) : Prop := sorry

-- Define a function to represent the division of a polygon into quadrilaterals
def divideIntoQuadrilaterals (p : Polygon) (qs : List Polygon) : Prop := sorry

theorem convex_polygon_division_impossibility (p : Polygon) (qs : List Polygon) :
  isConvex p → (∀ q ∈ qs, isNonConvexQuadrilateral q) → divideIntoQuadrilaterals p qs → False :=
by sorry

end convex_polygon_division_impossibility_l2693_269389


namespace expression_evaluation_l2693_269353

theorem expression_evaluation (x y : ℝ) (h : x > y ∧ y > 0) :
  (x^(2*y) * y^x) / (y^(2*x) * x^y) = (x/y)^(y-x) := by
  sorry

end expression_evaluation_l2693_269353


namespace jerrys_debt_problem_jerrys_total_debt_l2693_269312

/-- Jerry's debt payment problem -/
theorem jerrys_debt_problem (payment_two_months_ago : ℕ) 
                            (payment_increase : ℕ) 
                            (remaining_debt : ℕ) : ℕ :=
  let payment_last_month := payment_two_months_ago + payment_increase
  let total_paid := payment_two_months_ago + payment_last_month
  let total_debt := total_paid + remaining_debt
  total_debt

/-- Proof of Jerry's total debt -/
theorem jerrys_total_debt : jerrys_debt_problem 12 3 23 = 50 := by
  sorry

end jerrys_debt_problem_jerrys_total_debt_l2693_269312


namespace max_value_of_b_l2693_269330

theorem max_value_of_b (a b c : ℝ) : 
  (∃ q : ℝ, a = b / q ∧ c = b * q) →  -- geometric sequence condition
  (b + 2 = (a + 6 + c + 1) / 2) →     -- arithmetic sequence condition
  b ≤ 3/4 :=                          -- maximum value of b
by sorry

end max_value_of_b_l2693_269330


namespace shekar_weighted_average_l2693_269319

def weightedAverage (scores : List ℝ) (weights : List ℝ) : ℝ :=
  (List.zip scores weights).map (fun (s, w) => s * w) |> List.sum

theorem shekar_weighted_average :
  let scores : List ℝ := [76, 65, 82, 62, 85]
  let weights : List ℝ := [0.20, 0.15, 0.25, 0.25, 0.15]
  weightedAverage scores weights = 73.7 := by
sorry

end shekar_weighted_average_l2693_269319


namespace intersection_point_on_line_and_x_axis_l2693_269334

/-- The line equation 5y - 3x = 15 -/
def line_equation (x y : ℝ) : Prop := 5 * y - 3 * x = 15

/-- A point is on the x-axis if its y-coordinate is 0 -/
def on_x_axis (x y : ℝ) : Prop := y = 0

/-- The intersection point of the line and the x-axis -/
def intersection_point : ℝ × ℝ := (-5, 0)

theorem intersection_point_on_line_and_x_axis :
  line_equation intersection_point.1 intersection_point.2 ∧
  on_x_axis intersection_point.1 intersection_point.2 := by
  sorry

end intersection_point_on_line_and_x_axis_l2693_269334


namespace expand_product_l2693_269366

theorem expand_product (x : ℝ) : 5 * (x + 2) * (x + 6) * (x - 1) = 5 * x^3 + 35 * x^2 + 20 * x - 60 := by
  sorry

end expand_product_l2693_269366


namespace jose_share_of_profit_l2693_269388

/-- Calculates the share of profit for an investor based on their investment amount, duration, and total profit -/
def calculate_share_of_profit (investment : ℕ) (duration : ℕ) (total_investment_months : ℕ) (total_profit : ℕ) : ℕ :=
  (investment * duration * total_profit) / total_investment_months

theorem jose_share_of_profit (tom_investment : ℕ) (tom_duration : ℕ) (jose_investment : ℕ) (jose_duration : ℕ) (total_profit : ℕ) :
  tom_investment = 30000 →
  tom_duration = 12 →
  jose_investment = 45000 →
  jose_duration = 10 →
  total_profit = 27000 →
  calculate_share_of_profit jose_investment jose_duration (tom_investment * tom_duration + jose_investment * jose_duration) total_profit = 15000 := by
  sorry

#eval calculate_share_of_profit 45000 10 810000 27000

end jose_share_of_profit_l2693_269388
