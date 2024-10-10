import Mathlib

namespace cost_equation_solution_l1309_130965

/-- Given the cost equations for products A and B, prove that the solution (16, 4) satisfies both equations. -/
theorem cost_equation_solution :
  let x : ℚ := 16
  let y : ℚ := 4
  (20 * x + 15 * y = 380) ∧ (15 * x + 10 * y = 280) := by
  sorry

end cost_equation_solution_l1309_130965


namespace first_hour_distance_car_distance_problem_l1309_130971

/-- Given a car with increasing speed, calculate the distance traveled in the first hour -/
theorem first_hour_distance (speed_increase : ℕ → ℕ) (total_distance : ℕ) : ℕ :=
  let first_hour_dist : ℕ := 55
  have speed_increase_def : ∀ n : ℕ, speed_increase n = 2 * n := by sorry
  have total_distance_def : total_distance = 792 := by sorry
  have sum_formula : total_distance = (12 : ℕ) * first_hour_dist + 11 * 12 := by sorry
  first_hour_dist

/-- The main theorem stating the distance traveled in the first hour -/
theorem car_distance_problem : first_hour_distance (λ n => 2 * n) 792 = 55 := by sorry

end first_hour_distance_car_distance_problem_l1309_130971


namespace inequality_solution_implies_a_less_than_one_l1309_130975

theorem inequality_solution_implies_a_less_than_one :
  ∀ a : ℝ, (∀ x : ℝ, (a - 1) * x > 2 ↔ x < 2 / (a - 1)) → a < 1 := by
  sorry

end inequality_solution_implies_a_less_than_one_l1309_130975


namespace cubic_minus_linear_factorization_l1309_130996

theorem cubic_minus_linear_factorization (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) := by
  sorry

end cubic_minus_linear_factorization_l1309_130996


namespace segment_ratio_l1309_130945

/-- Given a line segment GH with points E and F lying on it, 
    where GE is 3 times EH and GF is 7 times FH, 
    prove that EF is 1/8 of GH. -/
theorem segment_ratio (G E F H : Real) : 
  (E - G) = 3 * (H - E) →
  (F - G) = 7 * (H - F) →
  (F - E) = (1/8) * (H - G) := by
  sorry

end segment_ratio_l1309_130945


namespace ratio_fraction_l1309_130959

theorem ratio_fraction (x y : ℚ) (h : x / y = 4 / 5) : (x + y) / (x - y) = -9 := by
  sorry

end ratio_fraction_l1309_130959


namespace fraction_is_positive_l1309_130905

theorem fraction_is_positive
  (a b c d : ℝ)
  (ha : a < 0) (hb : b < 0) (hc : c < 0) (hd : d < 0)
  (x₁ x₂ x₃ x₄ : ℝ)
  (h : |x₁ - a| + |x₂ + b| + |x₃ - c| + |x₄ + d| = 0) :
  (x₁ * x₂) / (x₃ * x₄) > 0 := by
sorry

end fraction_is_positive_l1309_130905


namespace chicken_rabbit_problem_l1309_130983

theorem chicken_rabbit_problem (x y : ℕ) : 
  (x + y = 35 ∧ 2 * x + 4 * y = 94) ↔ 
  (x + y = 35 ∧ x * 2 + y * 4 = 94) := by sorry

end chicken_rabbit_problem_l1309_130983


namespace total_gumballs_l1309_130976

def gumball_problem (total gumballs_todd gumballs_alisha gumballs_bobby remaining : ℕ) : Prop :=
  gumballs_todd = 4 ∧
  gumballs_alisha = 2 * gumballs_todd ∧
  gumballs_bobby = 4 * gumballs_alisha - 5 ∧
  total = gumballs_todd + gumballs_alisha + gumballs_bobby + remaining ∧
  remaining = 6

theorem total_gumballs : ∃ total : ℕ, gumball_problem total 4 8 27 6 ∧ total = 45 :=
  sorry

end total_gumballs_l1309_130976


namespace find_number_l1309_130904

theorem find_number (x y N : ℝ) (h1 : x / (2 * y) = N / 2) (h2 : (7 * x + 2 * y) / (x - 2 * y) = 23) : N = 3 := by
  sorry

end find_number_l1309_130904


namespace square_field_area_l1309_130914

theorem square_field_area (side_length : ℝ) (h : side_length = 5) :
  side_length * side_length = 25 := by sorry

end square_field_area_l1309_130914


namespace complement_A_intersect_B_range_of_a_l1309_130994

-- Define the sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x < 5}
def B (a : ℝ) : Set ℝ := {x | -a < x ∧ x ≤ a + 3}

-- Part 1
theorem complement_A_intersect_B :
  (Set.univ \ A) ∩ B 1 = {x : ℝ | -1 < x ∧ x < 1} := by sorry

-- Part 2
theorem range_of_a (a : ℝ) :
  B a ∩ A = B a → a ≤ -1 := by sorry

end complement_A_intersect_B_range_of_a_l1309_130994


namespace valid_3x3_grid_exists_l1309_130915

/-- Represents a county with a diagonal road -/
inductive County
  | NorthEast
  | SouthWest

/-- Represents a 3x3 grid of counties -/
def Grid := Fin 3 → Fin 3 → County

/-- Checks if two adjacent counties have compatible road directions -/
def compatible (c1 c2 : County) : Bool :=
  match c1, c2 with
  | County.NorthEast, County.SouthWest => true
  | County.SouthWest, County.NorthEast => true
  | _, _ => false

/-- Checks if the grid forms a valid closed path -/
def isValidPath (g : Grid) : Bool :=
  -- Implementation details omitted
  sorry

/-- Theorem stating that a valid 3x3 grid configuration exists -/
theorem valid_3x3_grid_exists : ∃ g : Grid, isValidPath g := by
  sorry

end valid_3x3_grid_exists_l1309_130915


namespace stella_annual_income_after_tax_l1309_130984

/-- Calculates Stella's annual income after tax deduction --/
theorem stella_annual_income_after_tax :
  let base_salary : ℕ := 3500
  let bonuses : List ℕ := [1200, 600, 1500, 900, 1200]
  let paid_months : ℕ := 10
  let tax_rate : ℚ := 1 / 20

  let total_base_salary := base_salary * paid_months
  let total_bonuses := bonuses.sum
  let total_income := total_base_salary + total_bonuses
  let tax_deduction := (total_income : ℚ) * tax_rate
  let annual_income_after_tax := (total_income : ℚ) - tax_deduction

  annual_income_after_tax = 38380 := by
  sorry

end stella_annual_income_after_tax_l1309_130984


namespace average_marks_combined_l1309_130907

theorem average_marks_combined (n₁ n₂ : ℕ) (avg₁ avg₂ : ℚ) 
  (h₁ : n₁ = 12) (h₂ : n₂ = 28) (h₃ : avg₁ = 40) (h₄ : avg₂ = 60) :
  (n₁ * avg₁ + n₂ * avg₂) / (n₁ + n₂ : ℚ) = 54 := by
  sorry

end average_marks_combined_l1309_130907


namespace julie_newspaper_count_l1309_130909

theorem julie_newspaper_count :
  let boxes : ℕ := 2
  let packages_per_box : ℕ := 5
  let sheets_per_package : ℕ := 250
  let sheets_per_newspaper : ℕ := 25
  let total_sheets : ℕ := boxes * packages_per_box * sheets_per_package
  let newspapers : ℕ := total_sheets / sheets_per_newspaper
  newspapers = 100 := by sorry

end julie_newspaper_count_l1309_130909


namespace correct_total_distance_l1309_130992

/-- The total distance to fly from Germany to Russia and then return to Spain -/
def totalDistance (spainRussia : ℕ) (spainGermany : ℕ) : ℕ :=
  (spainRussia - spainGermany) + spainRussia

theorem correct_total_distance :
  totalDistance 7019 1615 = 12423 := by
  sorry

#eval totalDistance 7019 1615

end correct_total_distance_l1309_130992


namespace q_satisfies_conditions_l1309_130960

/-- The quadratic polynomial q(x) that satisfies the given conditions -/
def q (x : ℚ) : ℚ := (17 * x^2 - 8 * x + 21) / 15

/-- Theorem stating that q(x) satisfies the required conditions -/
theorem q_satisfies_conditions :
  q (-2) = 7 ∧ q 1 = 2 ∧ q 3 = 10 := by
  sorry

end q_satisfies_conditions_l1309_130960


namespace sandy_tokens_l1309_130931

theorem sandy_tokens (total_tokens : ℕ) (num_siblings : ℕ) : 
  total_tokens = 1000000 →
  num_siblings = 4 →
  let sandy_share := total_tokens / 2
  let remaining_tokens := total_tokens - sandy_share
  let sibling_share := remaining_tokens / num_siblings
  sandy_share - sibling_share = 375000 :=
by
  sorry

end sandy_tokens_l1309_130931


namespace sequence_equality_l1309_130978

-- Define the sequence S
def S (n : ℕ) : ℕ := (4 * n - 3)^2

-- Define the proposed form of S
def S_proposed (n : ℕ) (a b : ℤ) : ℤ := (4 * n - 3) * (a * n + b)

-- Theorem statement
theorem sequence_equality (a b : ℤ) :
  (∀ n : ℕ, n > 0 → S n = S_proposed n a b) →
  a^2 + b^2 = 25 :=
sorry

end sequence_equality_l1309_130978


namespace matt_total_skips_l1309_130939

/-- The number of skips per second -/
def skips_per_second : ℕ := 3

/-- The duration of jumping in minutes -/
def jump_duration : ℕ := 10

/-- The number of seconds in a minute -/
def seconds_per_minute : ℕ := 60

/-- Theorem: Given the conditions, Matt's total number of skips is 1800 -/
theorem matt_total_skips :
  skips_per_second * jump_duration * seconds_per_minute = 1800 := by
  sorry

end matt_total_skips_l1309_130939


namespace goldbach_132_max_diff_l1309_130981

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem goldbach_132_max_diff :
  ∃ (p q : ℕ), is_prime p ∧ is_prime q ∧ p + q = 132 ∧ p < q ∧
  ∀ (r s : ℕ), is_prime r → is_prime s → r + s = 132 → r < s →
  s - r ≤ q - p ∧
  q - p = 122 :=
sorry

end goldbach_132_max_diff_l1309_130981


namespace parsley_sprigs_theorem_l1309_130922

/-- Calculates the number of parsley sprigs left after decorating plates -/
def sprigs_left (initial_sprigs : ℕ) (whole_sprig_plates : ℕ) (half_sprig_plates : ℕ) : ℕ :=
  initial_sprigs - (whole_sprig_plates + (half_sprig_plates / 2))

/-- Proves that given the specific conditions, 11 sprigs are left -/
theorem parsley_sprigs_theorem :
  sprigs_left 25 8 12 = 11 := by
  sorry

end parsley_sprigs_theorem_l1309_130922


namespace sally_boxes_proof_l1309_130938

/-- The number of boxes Sally sold on Saturday -/
def saturday_boxes : ℕ := 65

/-- The number of boxes Sally sold on Sunday -/
def sunday_boxes : ℕ := (3 * saturday_boxes) / 2

/-- The number of boxes Sally sold on Monday -/
def monday_boxes : ℕ := (13 * sunday_boxes) / 10

theorem sally_boxes_proof :
  saturday_boxes + sunday_boxes + monday_boxes = 290 :=
sorry

end sally_boxes_proof_l1309_130938


namespace complex_equation_solution_l1309_130902

theorem complex_equation_solution (x y : ℝ) :
  (Complex.I * (x + Complex.I) + y = 1 + 2 * Complex.I) →
  x - y = 0 := by
sorry

end complex_equation_solution_l1309_130902


namespace mans_speed_with_current_l1309_130974

/-- 
Given a man's speed against the current and the speed of the current,
this theorem proves the man's speed with the current.
-/
theorem mans_speed_with_current 
  (speed_against_current : ℝ) 
  (current_speed : ℝ) 
  (h1 : speed_against_current = 12.4)
  (h2 : current_speed = 4.3) : 
  speed_against_current + 2 * current_speed = 21 := by
sorry

end mans_speed_with_current_l1309_130974


namespace compound_interest_rate_l1309_130912

theorem compound_interest_rate (ci_2 ci_3 : ℚ) 
  (h1 : ci_2 = 1200)
  (h2 : ci_3 = 1260) : 
  ∃ r : ℚ, r = 0.05 ∧ r * ci_2 = ci_3 - ci_2 :=
sorry

end compound_interest_rate_l1309_130912


namespace smallest_c_plus_d_l1309_130901

theorem smallest_c_plus_d : ∃ (c d : ℕ+), 
  (3^6 * 7^2 : ℕ) = c^(d:ℕ) ∧ 
  (∀ (c' d' : ℕ+), (3^6 * 7^2 : ℕ) = c'^(d':ℕ) → c + d ≤ c' + d') ∧
  c + d = 1325 := by
  sorry

end smallest_c_plus_d_l1309_130901


namespace log_equation_solution_l1309_130946

-- Define the logarithm function (base 10)
noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_equation_solution (a : ℝ) (h : log a - 2 * log 2 = 1) : a = 40 := by
  sorry

end log_equation_solution_l1309_130946


namespace greatest_prime_factor_of_5_pow_7_plus_10_pow_6_l1309_130903

theorem greatest_prime_factor_of_5_pow_7_plus_10_pow_6 :
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ (5^7 + 10^6) ∧ ∀ q : ℕ, Nat.Prime q → q ∣ (5^7 + 10^6) → q ≤ p :=
by sorry

end greatest_prime_factor_of_5_pow_7_plus_10_pow_6_l1309_130903


namespace range_of_a_l1309_130956

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - a * log x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x + 2*x

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc 1 (Real.exp 1), ¬ Monotone (g a))
  ∧ (∀ x ∈ Set.Icc 1 (Real.exp 1), g a x ≤ g a (Real.exp 1))
  ∧ (∀ x ∈ Set.Icc 1 (Real.exp 1), x ≠ Real.exp 1 → g a x < g a (Real.exp 1))
  → 3 < a ∧ a < (Real.exp 1)^2 / 2 + 2 * Real.exp 1 - 5 / 2 :=
by sorry

end range_of_a_l1309_130956


namespace complex_on_ray_unit_circle_l1309_130986

theorem complex_on_ray_unit_circle (z : ℂ) (a b : ℝ) :
  z = a + b * I →
  a = b →
  a ≥ 0 →
  Complex.abs z = 1 →
  z = Complex.mk (Real.sqrt 2 / 2) (Real.sqrt 2 / 2) :=
by sorry

end complex_on_ray_unit_circle_l1309_130986


namespace yogurt_combinations_l1309_130913

theorem yogurt_combinations (flavors : Nat) (toppings : Nat) : 
  flavors = 5 → toppings = 7 → 
  flavors * (1 + toppings.choose 1 + toppings.choose 2) = 145 := by
sorry

end yogurt_combinations_l1309_130913


namespace joes_fast_food_cost_l1309_130963

theorem joes_fast_food_cost : 
  let sandwich_cost : ℕ := 4
  let soda_cost : ℕ := 3
  let num_sandwiches : ℕ := 3
  let num_sodas : ℕ := 5
  (num_sandwiches * sandwich_cost + num_sodas * soda_cost) = 27 := by
sorry

end joes_fast_food_cost_l1309_130963


namespace bumper_car_line_problem_l1309_130999

theorem bumper_car_line_problem (initial_people : ℕ) (people_left : ℕ) (total_people : ℕ) : 
  initial_people = 9 →
  people_left = 6 →
  total_people = 18 →
  total_people - (initial_people - people_left) = 15 := by
sorry

end bumper_car_line_problem_l1309_130999


namespace harriets_age_l1309_130990

theorem harriets_age (peter_age harriet_age : ℕ) : 
  (peter_age + 4 = 2 * (harriet_age + 4)) →  -- Condition 1
  (peter_age = 60 / 2) →                     -- Conditions 2 and 3 combined
  harriet_age = 13 := by
sorry

end harriets_age_l1309_130990


namespace correct_sum_l1309_130919

theorem correct_sum (x y : ℕ+) (h1 : x - y = 4) (h2 : x * y = 132) : x + y = 24 := by
  sorry

end correct_sum_l1309_130919


namespace roxanne_change_l1309_130953

/-- Represents the purchase and payment scenario for Roxanne --/
structure Purchase where
  lemonade_count : ℕ
  lemonade_price : ℚ
  sandwich_count : ℕ
  sandwich_price : ℚ
  paid_amount : ℚ

/-- Calculates the change Roxanne should receive --/
def calculate_change (p : Purchase) : ℚ :=
  p.paid_amount - (p.lemonade_count * p.lemonade_price + p.sandwich_count * p.sandwich_price)

/-- Theorem stating that Roxanne's change should be $11 --/
theorem roxanne_change :
  let p : Purchase := {
    lemonade_count := 2,
    lemonade_price := 2,
    sandwich_count := 2,
    sandwich_price := 2.5,
    paid_amount := 20
  }
  calculate_change p = 11 := by sorry

end roxanne_change_l1309_130953


namespace borrowed_amount_l1309_130998

theorem borrowed_amount (X : ℝ) : 
  (X + 0.1 * X = 110) → X = 100 := by
  sorry

end borrowed_amount_l1309_130998


namespace infinite_non_representable_numbers_l1309_130962

theorem infinite_non_representable_numbers : 
  ∃ S : Set ℕ, Set.Infinite S ∧ 
  ∀ k ∈ S, ∀ n : ℕ, ∀ p : ℕ, 
    Prime p → k ≠ n^2 + p := by
  sorry

end infinite_non_representable_numbers_l1309_130962


namespace negation_of_proposition_l1309_130972

theorem negation_of_proposition (m : ℝ) :
  (¬(m > 0 → ∃ x : ℝ, x^2 + x - m = 0)) ↔ (m ≤ 0 → ∀ x : ℝ, x^2 + x - m ≠ 0) :=
by sorry

end negation_of_proposition_l1309_130972


namespace batsman_average_after_12th_innings_l1309_130952

/-- Represents a batsman's statistics -/
structure Batsman where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an innings -/
def newAverage (b : Batsman) (runsScored : ℕ) : ℚ :=
  (b.totalRuns + runsScored) / (b.innings + 1)

/-- Theorem: A batsman's average after 12 innings is 58, given the conditions -/
theorem batsman_average_after_12th_innings 
  (b : Batsman)
  (h1 : b.innings = 11)
  (h2 : newAverage b 80 = b.average + 2)
  : newAverage b 80 = 58 := by
  sorry

end batsman_average_after_12th_innings_l1309_130952


namespace johns_share_ratio_l1309_130928

theorem johns_share_ratio (total : ℕ) (johns_share : ℕ) 
  (h1 : total = 4800) (h2 : johns_share = 1600) : 
  johns_share / total = 1 / 3 := by
  sorry

end johns_share_ratio_l1309_130928


namespace candy_store_sales_l1309_130969

-- Define the quantities and prices
def fudge_pounds : ℕ := 20
def fudge_price : ℚ := 2.5
def truffle_dozens : ℕ := 5
def truffle_price : ℚ := 1.5
def pretzel_dozens : ℕ := 3
def pretzel_price : ℚ := 2

-- Define the calculation for total sales
def total_sales : ℚ :=
  fudge_pounds * fudge_price +
  truffle_dozens * 12 * truffle_price +
  pretzel_dozens * 12 * pretzel_price

-- Theorem statement
theorem candy_store_sales : total_sales = 212 := by
  sorry

end candy_store_sales_l1309_130969


namespace sqrt_equation_solution_l1309_130918

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (4 + Real.sqrt x) = 4 → x = 144 := by
  sorry

end sqrt_equation_solution_l1309_130918


namespace part_one_part_two_part_three_l1309_130947

-- Define the quadratic equation
def quadratic_equation (m : ℝ) (x : ℝ) : Prop :=
  3 * x^2 + m * x + 2 = 0

-- Part I
theorem part_one (m : ℝ) : quadratic_equation m 2 → m = -7 := by
  sorry

-- Part II
theorem part_two :
  (∃ x₁ x₂ : ℝ, x₁ = 1 ∧ x₂ = 2/3 ∧ quadratic_equation (-5) x₁ ∧ quadratic_equation (-5) x₂) := by
  sorry

-- Part III
theorem part_three (m : ℝ) :
  m ≥ 5 →
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation m x₁ ∧ quadratic_equation m x₂ := by
  sorry

end part_one_part_two_part_three_l1309_130947


namespace carrots_grown_total_l1309_130991

/-- The number of carrots grown by Joan -/
def joans_carrots : ℕ := 29

/-- The number of carrots grown by Jessica -/
def jessicas_carrots : ℕ := 11

/-- The total number of carrots grown by Joan and Jessica -/
def total_carrots : ℕ := joans_carrots + jessicas_carrots

theorem carrots_grown_total : total_carrots = 40 := by
  sorry

end carrots_grown_total_l1309_130991


namespace two_points_theorem_l1309_130958

/-- Represents the three possible states of a point in the bun -/
inductive PointState
  | Type1
  | Type2
  | NoRaisin

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The bun as a bounded 3D space -/
def Bun : Set Point3D :=
  sorry

/-- Function that determines the state of a point in the bun -/
def pointState : Point3D → PointState :=
  sorry

/-- Distance between two points in 3D space -/
def distance (p q : Point3D) : ℝ :=
  sorry

theorem two_points_theorem :
  ∃ (p q : Point3D), p ∈ Bun ∧ q ∈ Bun ∧ distance p q = 1 ∧
    (pointState p = pointState q ∨ (pointState p = PointState.NoRaisin ∧ pointState q = PointState.NoRaisin)) :=
  sorry

end two_points_theorem_l1309_130958


namespace x_value_l1309_130924

theorem x_value (x : ℝ) : x + Real.sqrt 81 = 25 → x = 16 := by
  sorry

end x_value_l1309_130924


namespace k_range_l1309_130942

-- Define the propositions p and q
def p (x k : ℝ) : Prop := x ≥ k
def q (x : ℝ) : Prop := (3 : ℝ) / (x + 1) < 1

-- Define the necessary but not sufficient condition
def necessary_but_not_sufficient (p q : ℝ → Prop) : Prop :=
  (∀ x, q x → p x) ∧ ∃ x, p x ∧ ¬q x

-- Theorem statement
theorem k_range (k : ℝ) : 
  necessary_but_not_sufficient (p k) q ↔ k > 2 :=
sorry

end k_range_l1309_130942


namespace line_equal_intercepts_l1309_130993

/-- A line with equation ax + y - 2 - a = 0 has equal intercepts on the x-axis and y-axis if and only if a = -2 or a = 1 -/
theorem line_equal_intercepts (a : ℝ) : 
  (∃ k : ℝ, k ≠ 0 ∧ 
    (a * k = k - 2 + a) ∧ 
    (k = k - 2 + a)) ↔ 
  (a = -2 ∨ a = 1) :=
sorry

end line_equal_intercepts_l1309_130993


namespace min_sum_reciprocals_l1309_130977

theorem min_sum_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 10) :
  ∃ (a b : ℕ+), a ≠ b ∧ (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 10 ∧ 
  (∀ (c d : ℕ+), c ≠ d → (1 : ℚ) / c + (1 : ℚ) / d = (1 : ℚ) / 10 → a + b ≤ c + d) ∧
  a + b = 45 :=
sorry

end min_sum_reciprocals_l1309_130977


namespace complex_modulus_problem_l1309_130917

theorem complex_modulus_problem (z : ℂ) (h : z * (1 - Complex.I) = 2) :
  Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_modulus_problem_l1309_130917


namespace fraction_order_l1309_130937

theorem fraction_order : 
  (20 : ℚ) / 15 < 25 / 18 ∧ 25 / 18 < 23 / 16 ∧ 23 / 16 < 21 / 14 := by
  sorry

end fraction_order_l1309_130937


namespace executive_committee_formation_l1309_130987

theorem executive_committee_formation (total_members : ℕ) (experienced_members : ℕ) (committee_size : ℕ) : 
  total_members = 30 →
  experienced_members = 8 →
  committee_size = 5 →
  (Finset.sum (Finset.range (Nat.min committee_size experienced_members + 1))
    (λ k => Nat.choose experienced_members k * Nat.choose (total_members - experienced_members) (committee_size - k))) = 116172 := by
  sorry

end executive_committee_formation_l1309_130987


namespace gina_tip_percentage_is_five_percent_l1309_130908

/-- The bill amount in dollars -/
def bill_amount : ℝ := 26

/-- The minimum tip percentage for good tippers -/
def good_tipper_percentage : ℝ := 20

/-- The additional amount in cents Gina needs to tip to be a good tipper -/
def additional_tip_cents : ℝ := 390

/-- Gina's tip percentage -/
def gina_tip_percentage : ℝ := 5

/-- Theorem stating that Gina's tip percentage is 5% given the conditions -/
theorem gina_tip_percentage_is_five_percent :
  (gina_tip_percentage / 100) * bill_amount + (additional_tip_cents / 100) =
  (good_tipper_percentage / 100) * bill_amount :=
by sorry

end gina_tip_percentage_is_five_percent_l1309_130908


namespace monomial_properties_l1309_130921

/-- Represents a monomial in variables a and b -/
structure Monomial where
  coeff : ℤ
  a_exp : ℕ
  b_exp : ℕ

/-- The coefficient of a monomial -/
def coefficient (m : Monomial) : ℤ := m.coeff

/-- The degree of a monomial -/
def degree (m : Monomial) : ℕ := m.a_exp + m.b_exp

/-- The monomial 2a²b -/
def m : Monomial := { coeff := 2, a_exp := 2, b_exp := 1 }

theorem monomial_properties :
  coefficient m = 2 ∧ degree m = 3 := by sorry

end monomial_properties_l1309_130921


namespace homothety_composition_l1309_130929

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a homothety in 3D space
structure Homothety3D where
  center : Point3D
  ratio : ℝ

-- Define a line in 3D space
structure Line3D where
  point : Point3D
  direction : Point3D

-- Function to compose two homotheties
def compose_homotheties (h1 h2 : Homothety3D) : Homothety3D :=
  sorry

-- Function to check if a point lies on a line
def point_on_line (p : Point3D) (l : Line3D) : Prop :=
  sorry

-- Theorem statement
theorem homothety_composition 
  (h1 h2 : Homothety3D) 
  (l : Line3D) :
  let h3 := compose_homotheties h1 h2
  point_on_line h3.center l ∧ 
  h3.ratio = h1.ratio * h2.ratio ∧
  point_on_line h1.center l ∧
  point_on_line h2.center l :=
sorry

end homothety_composition_l1309_130929


namespace eggs_not_eaten_is_six_l1309_130961

/-- Represents the number of eggs not eaten in a week given the following conditions:
  * Rhea buys 2 trays of eggs every week
  * Each tray has 24 eggs
  * Her son and daughter eat 2 eggs every morning
  * Rhea and her husband eat 4 eggs every night
  * There are 7 days in a week
-/
def eggs_not_eaten : ℕ :=
  let trays_per_week : ℕ := 2
  let eggs_per_tray : ℕ := 24
  let children_eggs_per_day : ℕ := 2
  let parents_eggs_per_day : ℕ := 4
  let days_per_week : ℕ := 7
  
  let total_eggs_bought := trays_per_week * eggs_per_tray
  let children_eggs_eaten := children_eggs_per_day * days_per_week
  let parents_eggs_eaten := parents_eggs_per_day * days_per_week
  let total_eggs_eaten := children_eggs_eaten + parents_eggs_eaten
  
  total_eggs_bought - total_eggs_eaten

theorem eggs_not_eaten_is_six : eggs_not_eaten = 6 := by
  sorry

end eggs_not_eaten_is_six_l1309_130961


namespace unique_divisible_by_44_l1309_130995

/-- Represents a six-digit number in the form 5n7264 where n is a single digit -/
def sixDigitNumber (n : Nat) : Nat :=
  500000 + 10000 * n + 7264

/-- Checks if a number is divisible by another number -/
def isDivisibleBy (a b : Nat) : Prop :=
  ∃ k, a = b * k

/-- Theorem stating that 517264 is the only number in the form 5n7264 
    (where n is a single digit) that is divisible by 44 -/
theorem unique_divisible_by_44 : 
  ∀ n : Nat, n < 10 → 
    (isDivisibleBy (sixDigitNumber n) 44 ↔ n = 1) := by
  sorry

#check unique_divisible_by_44

end unique_divisible_by_44_l1309_130995


namespace martin_crayons_l1309_130967

theorem martin_crayons (total_boxes : ℕ) (crayons_per_box : ℕ) (boxes_with_missing : ℕ) (missing_per_box : ℕ) :
  total_boxes = 8 →
  crayons_per_box = 7 →
  boxes_with_missing = 3 →
  missing_per_box = 2 →
  total_boxes * crayons_per_box - boxes_with_missing * missing_per_box = 50 :=
by sorry

end martin_crayons_l1309_130967


namespace sin_thirty_degrees_l1309_130916

/-- Given a point Q on the unit circle 30° counterclockwise from (1,0),
    and E as the foot of the altitude from Q to the x-axis,
    prove that sin(30°) = 1/2 -/
theorem sin_thirty_degrees (Q : ℝ × ℝ) (E : ℝ × ℝ) :
  (Q.1^2 + Q.2^2 = 1) →  -- Q is on the unit circle
  (Q.1 = Real.cos (30 * π / 180)) →  -- Q is 30° counterclockwise from (1,0)
  (Q.2 = Real.sin (30 * π / 180)) →
  (E.1 = Q.1 ∧ E.2 = 0) →  -- E is the foot of the altitude from Q to the x-axis
  Real.sin (30 * π / 180) = 1/2 := by
sorry

end sin_thirty_degrees_l1309_130916


namespace double_root_condition_l1309_130923

/-- For a polynomial of the form A x^(n+1) + B x^n + 1, where n is a natural number,
    x = 1 is a root with multiplicity at least 2 if and only if A = n and B = -(n+1). -/
theorem double_root_condition (n : ℕ) (A B : ℝ) :
  (∀ x : ℝ, A * x^(n+1) + B * x^n + 1 = 0 ∧ 
   (A * (n+1) * x^n + B * n * x^(n-1) = 0)) ↔ 
  (A = n ∧ B = -(n+1)) :=
sorry

end double_root_condition_l1309_130923


namespace race_head_start_l1309_130927

theorem race_head_start (L : ℝ) (Va Vb : ℝ) (h : Va = (22 / 19) * Vb) :
  ∃ H : ℝ, H / L = 3 / 22 ∧ L / Va = (L - H) / Vb :=
by sorry

end race_head_start_l1309_130927


namespace cycle_gain_percent_l1309_130941

/-- Calculates the gain percent when an item is bought and sold at given prices. -/
def gainPercent (costPrice sellingPrice : ℚ) : ℚ :=
  ((sellingPrice - costPrice) / costPrice) * 100

/-- Theorem: The gain percent is 50% when a cycle is bought for Rs. 900 and sold for Rs. 1350. -/
theorem cycle_gain_percent :
  let costPrice : ℚ := 900
  let sellingPrice : ℚ := 1350
  gainPercent costPrice sellingPrice = 50 := by
  sorry

end cycle_gain_percent_l1309_130941


namespace carmichael_function_properties_l1309_130979

variable (a : ℕ)

theorem carmichael_function_properties (ha : a > 2) :
  (∃ n : ℕ, n > 1 ∧ ¬ Nat.Prime n ∧ a^n ≡ 1 [ZMOD n]) ∧
  (∀ p : ℕ, p > 1 → (∀ k : ℕ, 1 < k ∧ k < p → ¬(a^k ≡ 1 [ZMOD k])) → a^p ≡ 1 [ZMOD p] → Nat.Prime p) ∧
  ¬(∃ n : ℕ, n > 1 ∧ 2^n ≡ 1 [ZMOD n]) :=
by sorry

end carmichael_function_properties_l1309_130979


namespace quadratic_roots_existence_l1309_130966

/-- The discriminant of a quadratic equation ax^2 + bx + c = 0 -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- A quadratic equation ax^2 + bx + c = 0 has real roots iff its discriminant is nonnegative -/
def has_real_roots (a b c : ℝ) : Prop := discriminant a b c ≥ 0

theorem quadratic_roots_existence :
  ¬(has_real_roots 1 1 1) ∧
  (has_real_roots 1 2 1) ∧
  (has_real_roots 1 (-2) (-1)) ∧
  (has_real_roots 1 (-1) (-2)) := by sorry

end quadratic_roots_existence_l1309_130966


namespace initial_oranges_count_l1309_130968

/-- The number of oranges initially in the bin -/
def initial_oranges : ℕ := sorry

/-- The number of oranges thrown away -/
def thrown_away : ℕ := 37

/-- The number of new oranges added -/
def new_oranges : ℕ := 7

/-- The final number of oranges in the bin -/
def final_oranges : ℕ := 10

/-- Theorem stating that the initial number of oranges was 40 -/
theorem initial_oranges_count : initial_oranges = 40 := by
  sorry

end initial_oranges_count_l1309_130968


namespace cubic_inequality_solution_l1309_130933

theorem cubic_inequality_solution (x : ℝ) : 
  x^3 - 9*x^2 + 24*x > 0 ↔ (0 < x ∧ x < 3) ∨ (x > 8) := by
  sorry

end cubic_inequality_solution_l1309_130933


namespace intersection_point_determines_k_l1309_130935

/-- Line with slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.on_line (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- Check if two lines are perpendicular -/
def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.slope * l2.slope = -1

theorem intersection_point_determines_k 
  (m n : Line)
  (p : Point)
  (k : ℝ)
  (h1 : m.slope = 4)
  (h2 : m.intercept = 2)
  (h3 : n.slope = k)
  (h4 : n.intercept = 3)
  (h5 : p.x = 1)
  (h6 : p.y = 6)
  (h7 : p.on_line m)
  (h8 : p.on_line n)
  : k = 3 := by
  sorry

#check intersection_point_determines_k

end intersection_point_determines_k_l1309_130935


namespace complex_magnitude_squared_plus_self_l1309_130944

theorem complex_magnitude_squared_plus_self (z : ℂ) (h : z = 1 + I) :
  Complex.abs (z^2 + z) = Real.sqrt 10 := by
  sorry

end complex_magnitude_squared_plus_self_l1309_130944


namespace circle_equation_l1309_130988

/-- A circle C with center (a,b) and radius 1 -/
structure Circle where
  a : ℝ
  b : ℝ
  radius : ℝ := 1

/-- The circle C is in the first quadrant -/
def in_first_quadrant (C : Circle) : Prop :=
  C.a > 0 ∧ C.b > 0

/-- The circle C is tangent to the line 4x-3y=0 -/
def tangent_to_line (C : Circle) : Prop :=
  abs (4 * C.a - 3 * C.b) / 5 = C.radius

/-- The circle C is tangent to the x-axis -/
def tangent_to_x_axis (C : Circle) : Prop :=
  C.b = C.radius

/-- The standard equation of the circle -/
def standard_equation (C : Circle) : Prop :=
  ∀ x y : ℝ, (x - C.a)^2 + (y - C.b)^2 = C.radius^2

theorem circle_equation (C : Circle) 
  (h1 : in_first_quadrant C)
  (h2 : tangent_to_line C)
  (h3 : tangent_to_x_axis C) :
  standard_equation { a := 2, b := 1, radius := 1 } :=
sorry

end circle_equation_l1309_130988


namespace tina_crayon_selection_ways_l1309_130989

/-- The number of different-colored crayons in the box -/
def total_crayons : ℕ := 15

/-- The number of crayons Tina must select -/
def selected_crayons : ℕ := 6

/-- The number of mandatory crayons (red and blue) -/
def mandatory_crayons : ℕ := 2

/-- Computes the number of ways to select k items from n items -/
def combinations (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- The main theorem stating the number of ways Tina can select the crayons -/
theorem tina_crayon_selection_ways :
  combinations (total_crayons - mandatory_crayons) (selected_crayons - mandatory_crayons) = 715 := by
  sorry

end tina_crayon_selection_ways_l1309_130989


namespace minor_premise_identification_l1309_130970

-- Define the type for functions
def Function := Type → Type

-- Define properties
def IsTrigonometric (f : Function) : Prop := sorry
def IsPeriodic (f : Function) : Prop := sorry

-- Define tan function
def tan : Function := sorry

-- Theorem statement
theorem minor_premise_identification :
  (∀ f : Function, IsTrigonometric f → IsPeriodic f) →  -- major premise
  (IsTrigonometric tan) →                               -- minor premise
  (IsPeriodic tan) →                                    -- conclusion
  (IsTrigonometric tan)                                 -- proves minor premise
  := by sorry

end minor_premise_identification_l1309_130970


namespace cubic_solution_sum_l1309_130973

theorem cubic_solution_sum (k : ℝ) (a b c : ℝ) : 
  (a^3 - 6*a^2 + 8*a + k = 0) →
  (b^3 - 6*b^2 + 8*b + k = 0) →
  (c^3 - 6*c^2 + 8*c + k = 0) →
  (k ≠ 0) →
  (a*b/c + b*c/a + c*a/b = 64/k - 12) := by sorry

end cubic_solution_sum_l1309_130973


namespace omi_age_l1309_130936

/-- Given the ages of Kimiko, Arlette, and Omi, prove Omi's age -/
theorem omi_age (kimiko_age : ℕ) (arlette_age : ℕ) (omi_age : ℕ) : 
  kimiko_age = 28 →
  arlette_age = (3 * kimiko_age) / 4 →
  (kimiko_age + arlette_age + omi_age) / 3 = 35 →
  omi_age = 56 := by
sorry

end omi_age_l1309_130936


namespace jessie_weight_before_jogging_l1309_130964

def weight_before_jogging (weight_after_first_week weight_lost_first_week : ℕ) : ℕ :=
  weight_after_first_week + weight_lost_first_week

theorem jessie_weight_before_jogging 
  (weight_after_first_week : ℕ) 
  (weight_lost_first_week : ℕ) 
  (h1 : weight_after_first_week = 36)
  (h2 : weight_lost_first_week = 56) : 
  weight_before_jogging weight_after_first_week weight_lost_first_week = 92 := by
  sorry

end jessie_weight_before_jogging_l1309_130964


namespace croissant_cost_calculation_l1309_130934

/-- Calculates the cost of croissants for a committee luncheon --/
theorem croissant_cost_calculation 
  (people : ℕ) 
  (sandwiches_per_person : ℕ) 
  (croissants_per_dozen : ℕ) 
  (cost_per_dozen : ℚ) : 
  people = 24 → 
  sandwiches_per_person = 2 → 
  croissants_per_dozen = 12 → 
  cost_per_dozen = 8 → 
  (people * sandwiches_per_person / croissants_per_dozen : ℚ) * cost_per_dozen = 32 :=
by sorry

#check croissant_cost_calculation

end croissant_cost_calculation_l1309_130934


namespace area_of_union_S_l1309_130957

/-- A disc D in the 2D plane -/
structure Disc where
  center : ℝ × ℝ
  radius : ℝ

/-- The set S of discs D -/
def S : Set Disc :=
  {D : Disc | D.center.2 = D.center.1^2 - 3/4 ∧ 
              ∀ (x y : ℝ), (x - D.center.1)^2 + (y - D.center.2)^2 < D.radius^2 → y < 0}

/-- The area of the union of all discs in S -/
def unionArea (S : Set Disc) : ℝ := sorry

/-- Theorem stating the area of the union of discs in S -/
theorem area_of_union_S : unionArea S = (2 * Real.pi / 3) + (Real.sqrt 3 / 4) := by sorry

end area_of_union_S_l1309_130957


namespace certain_number_value_l1309_130930

theorem certain_number_value (x y z : ℝ) 
  (h1 : y = 1.10 * z) 
  (h2 : x = 0.90 * y) 
  (h3 : x = 123.75) : 
  z = 125 := by
  sorry

end certain_number_value_l1309_130930


namespace inequality_solution_l1309_130954

theorem inequality_solution (a x : ℝ) :
  (a * x) / (x - 1) < (a - 1) / (x - 1) ↔
  (a > 0 ∧ (a - 1) / a < x ∧ x < 1) ∨
  (a = 0 ∧ x < 1) ∨
  (a < 0 ∧ (x > (a - 1) / a ∨ x < 1)) :=
by sorry

end inequality_solution_l1309_130954


namespace xyz_value_l1309_130910

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 30)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 10) :
  x * y * z = 20 / 3 := by
sorry

end xyz_value_l1309_130910


namespace triangle_problem_l1309_130997

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  A + B + C = π ∧
  Real.cos (B - C) - 2 * Real.sin B * Real.sin C = -1/2 →
  A = π/3 ∧
  (a = 5 ∧ b = 4 →
    a^2 = b^2 + c^2 - 2*b*c*Real.cos A →
    (1/2) * b * c * Real.sin A = 2*Real.sqrt 3 + Real.sqrt 39) :=
by sorry

end triangle_problem_l1309_130997


namespace trigonometric_system_solution_l1309_130940

theorem trigonometric_system_solution (θ : ℝ) (a b : ℝ) 
  (eq1 : Real.sin θ + Real.cos θ = a)
  (eq2 : Real.sin θ - Real.cos θ = b)
  (eq3 : Real.sin θ * Real.sin θ - Real.cos θ * Real.cos θ - Real.sin θ = -b * b) :
  ((a = Real.sqrt 7 / 2 ∧ b = 1 / 2) ∨
   (a = -Real.sqrt 7 / 2 ∧ b = 1 / 2) ∨
   (a = 1 ∧ b = -1) ∨
   (a = -1 ∧ b = 1)) := by
  sorry

end trigonometric_system_solution_l1309_130940


namespace james_quiz_bowl_points_l1309_130925

/-- Calculates the total points earned by a student in a quiz bowl game. -/
def quiz_bowl_points (total_rounds : ℕ) (questions_per_round : ℕ) (points_per_correct : ℕ) 
  (bonus_points : ℕ) (questions_missed : ℕ) : ℕ :=
  let total_questions := total_rounds * questions_per_round
  let correct_answers := total_questions - questions_missed
  let base_points := correct_answers * points_per_correct
  let full_rounds := total_rounds - (questions_missed + questions_per_round - 1) / questions_per_round
  let bonus_total := full_rounds * bonus_points
  base_points + bonus_total

/-- Theorem stating that James earned 64 points in the quiz bowl game. -/
theorem james_quiz_bowl_points : 
  quiz_bowl_points 5 5 2 4 1 = 64 := by
  sorry

end james_quiz_bowl_points_l1309_130925


namespace division_remainder_problem_l1309_130900

theorem division_remainder_problem (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) 
  (h1 : dividend = 161)
  (h2 : divisor = 16)
  (h3 : quotient = 10)
  (h4 : dividend = divisor * quotient + (dividend % divisor)) :
  dividend % divisor = 1 := by
  sorry

end division_remainder_problem_l1309_130900


namespace t_range_max_radius_equation_l1309_130955

-- Define the circle equation
def circle_equation (x y t : ℝ) : Prop := x^2 + y^2 - 2*x + t^2 = 0

-- Theorem for the range of t
theorem t_range : ∀ x y t : ℝ, circle_equation x y t → -1 < t ∧ t < 1 := by sorry

-- Define the maximum radius
def max_radius (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Theorem for the circle equation when radius is maximum
theorem max_radius_equation : 
  (∃ t : ℝ, ∀ x y : ℝ, circle_equation x y t ∧ 
    (∀ t' : ℝ, circle_equation x y t' → 
      (x - 1)^2 + y^2 ≥ (x - 1)^2 + y^2)) → 
  ∀ x y : ℝ, max_radius x y := by sorry

end t_range_max_radius_equation_l1309_130955


namespace semicircle_circumference_from_rectangle_square_l1309_130948

/-- Given a rectangle and a square with equal perimeters, prove the circumference of a semicircle
    whose diameter is equal to the side of the square. -/
theorem semicircle_circumference_from_rectangle_square 
  (rect_length : ℝ) (rect_breadth : ℝ) (square_side : ℝ) :
  rect_length = 8 →
  rect_breadth = 6 →
  2 * (rect_length + rect_breadth) = 4 * square_side →
  ∃ (semicircle_circumference : ℝ), 
    semicircle_circumference = Real.pi * square_side / 2 + square_side :=
by sorry

end semicircle_circumference_from_rectangle_square_l1309_130948


namespace calculation_proof_l1309_130949

theorem calculation_proof : 121 * (13 / 25) + 12 * (21 / 25) = 73 := by
  sorry

end calculation_proof_l1309_130949


namespace gingers_garden_water_usage_l1309_130950

/-- Represents the problem of calculating water usage in Ginger's garden --/
theorem gingers_garden_water_usage 
  (hours_worked : ℕ) 
  (bottle_capacity : ℕ) 
  (total_water_used : ℕ) 
  (h1 : hours_worked = 8)
  (h2 : bottle_capacity = 2)
  (h3 : total_water_used = 26) :
  (total_water_used - hours_worked * bottle_capacity) / bottle_capacity = 5 := by
  sorry

#check gingers_garden_water_usage

end gingers_garden_water_usage_l1309_130950


namespace remaining_amount_is_15_60_l1309_130982

/-- Calculate the remaining amount for a trip given expenses and gifts --/
def calculate_remaining_amount (initial_amount gas_cost lunch_cost gift_cost_per_person num_people extra_gift_cost grandma_gift toll_fee ice_cream_cost : ℚ) : ℚ :=
  let total_spent := gas_cost + lunch_cost + (gift_cost_per_person * num_people) + extra_gift_cost
  let total_received := initial_amount + (grandma_gift * num_people)
  let amount_before_return := total_received - total_spent
  amount_before_return - (toll_fee + ice_cream_cost)

/-- Theorem stating that the remaining amount for the return trip is $15.60 --/
theorem remaining_amount_is_15_60 :
  calculate_remaining_amount 60 12 23.40 5 3 7 10 8 9 = 15.60 := by
  sorry

end remaining_amount_is_15_60_l1309_130982


namespace train_cars_count_l1309_130932

/-- The number of cars counted in the first 15 seconds -/
def initial_cars : ℕ := 9

/-- The time in seconds for the initial count -/
def initial_time : ℕ := 15

/-- The total time in seconds for the train to clear the crossing -/
def total_time : ℕ := 210

/-- The number of cars in the train -/
def train_cars : ℕ := (initial_cars * total_time) / initial_time

theorem train_cars_count : train_cars = 126 := by
  sorry

end train_cars_count_l1309_130932


namespace mary_nickels_l1309_130980

theorem mary_nickels (initial_nickels : ℕ) (dad_gave : ℕ) (total_now : ℕ) : 
  dad_gave = 5 → total_now = 12 → initial_nickels + dad_gave = total_now → initial_nickels = 7 := by
sorry

end mary_nickels_l1309_130980


namespace count_parallelograms_l1309_130906

/-- The number of parallelograms formed in a grid created by intersecting a parallelogram
    with two sets of m lines each (parallel to the parallelogram's sides) -/
def num_parallelograms (m : ℕ) : ℕ :=
  ((m + 1) * (m + 2) / 2) ^ 2

/-- Theorem stating that num_parallelograms correctly calculates the number of parallelograms -/
theorem count_parallelograms (m : ℕ) :
  num_parallelograms m = ((m + 1) * (m + 2) / 2) ^ 2 := by
  sorry

end count_parallelograms_l1309_130906


namespace negation_of_exists_lt_is_forall_ge_l1309_130951

theorem negation_of_exists_lt_is_forall_ge :
  (¬ ∃ x : ℝ, x^2 - x + 1 < 0) ↔ (∀ x : ℝ, x^2 - x + 1 ≥ 0) := by
  sorry

end negation_of_exists_lt_is_forall_ge_l1309_130951


namespace john_car_profit_l1309_130985

/-- The money John made from fixing and racing a car -/
theorem john_car_profit (original_cost repair_discount prize_money prize_keep_percent : ℚ)
  (h1 : original_cost = 20000)
  (h2 : repair_discount = 20)
  (h3 : prize_money = 70000)
  (h4 : prize_keep_percent = 90) :
  let discounted_cost := original_cost * (1 - repair_discount / 100)
  let kept_prize := prize_money * (prize_keep_percent / 100)
  kept_prize - discounted_cost = 47000 :=
by sorry

end john_car_profit_l1309_130985


namespace equation_one_solution_system_of_equations_solution_l1309_130920

-- Equation (1)
theorem equation_one_solution :
  ∃! x : ℚ, (2*x + 5) / 6 - (3*x - 2) / 8 = 1 :=
by sorry

-- System of equations (2)
theorem system_of_equations_solution :
  ∃! (x y : ℚ), (2*x - 1) / 5 + (3*y - 2) / 4 = 2 ∧
                (3*x + 1) / 5 - (3*y + 2) / 4 = 0 :=
by sorry

end equation_one_solution_system_of_equations_solution_l1309_130920


namespace probability_of_sum_17_l1309_130911

/-- The number of faces on each die -/
def numFaces : ℕ := 8

/-- The target sum we're aiming for -/
def targetSum : ℕ := 17

/-- The number of dice rolled -/
def numDice : ℕ := 3

/-- The probability of rolling a specific number on a single die -/
def singleDieProbability : ℚ := 1 / numFaces

/-- The total number of possible outcomes when rolling three dice -/
def totalOutcomes : ℕ := numFaces ^ numDice

/-- The number of favorable outcomes (ways to get a sum of 17) -/
def favorableOutcomes : ℕ := 27

/-- The theorem stating the probability of rolling a sum of 17 with three 8-faced dice -/
theorem probability_of_sum_17 : 
  (favorableOutcomes : ℚ) / totalOutcomes = 27 / 512 :=
sorry

end probability_of_sum_17_l1309_130911


namespace ratio_problem_l1309_130943

theorem ratio_problem (a b c : ℝ) (h1 : b/a = 3) (h2 : c/b = 4) : 
  (a + 2*b) / (b + 2*c) = 7/27 := by
sorry

end ratio_problem_l1309_130943


namespace cars_remaining_l1309_130926

theorem cars_remaining (initial : Nat) (first_group : Nat) (second_group : Nat)
  (h1 : initial = 24)
  (h2 : first_group = 8)
  (h3 : second_group = 6) :
  initial - first_group - second_group = 10 := by
  sorry

end cars_remaining_l1309_130926
