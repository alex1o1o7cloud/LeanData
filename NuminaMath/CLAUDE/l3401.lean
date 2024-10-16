import Mathlib

namespace NUMINAMATH_CALUDE_cubic_stone_weight_l3401_340166

/-- The weight of a cubic stone -/
def stone_weight (edge_length : ℝ) (weight_per_unit : ℝ) : ℝ :=
  edge_length ^ 3 * weight_per_unit

/-- Theorem: The weight of a cubic stone with edge length 8 decimeters,
    where each cubic decimeter weighs 3.5 kilograms, is 1792 kilograms. -/
theorem cubic_stone_weight :
  stone_weight 8 3.5 = 1792 := by
  sorry

end NUMINAMATH_CALUDE_cubic_stone_weight_l3401_340166


namespace NUMINAMATH_CALUDE_largest_vertex_sum_l3401_340115

/-- Represents a parabola passing through specific points -/
structure Parabola (P : ℤ) where
  a : ℤ
  b : ℤ
  c : ℤ
  pass_origin : a * 0 * 0 + b * 0 + c = 0
  pass_3P : a * (3 * P) * (3 * P) + b * (3 * P) + c = 0
  pass_3P_minus_1 : a * (3 * P - 1) * (3 * P - 1) + b * (3 * P - 1) + c = 45

/-- Calculates the sum of coordinates of the vertex of a parabola -/
def vertexSum (P : ℤ) (p : Parabola P) : ℚ :=
  3 * P / 2 - (p.a : ℚ) * (9 * P^2 : ℚ) / 4

/-- Theorem stating the largest possible vertex sum -/
theorem largest_vertex_sum :
  ∀ P : ℤ, P ≠ 0 → ∀ p : Parabola P, vertexSum P p ≤ 138 := by sorry

end NUMINAMATH_CALUDE_largest_vertex_sum_l3401_340115


namespace NUMINAMATH_CALUDE_loan_interest_time_l3401_340145

/-- 
Given:
- A loan of 1000 at 3% per year
- A loan of 1400 at 5% per year
- The total interest is 350

Prove that the number of years required for the total interest to reach 350 is 3.5
-/
theorem loan_interest_time (principal1 principal2 rate1 rate2 total_interest : ℝ) 
  (h1 : principal1 = 1000)
  (h2 : principal2 = 1400)
  (h3 : rate1 = 0.03)
  (h4 : rate2 = 0.05)
  (h5 : total_interest = 350) :
  (total_interest / (principal1 * rate1 + principal2 * rate2)) = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_loan_interest_time_l3401_340145


namespace NUMINAMATH_CALUDE_part_one_part_two_l3401_340185

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x - 1

-- Part (1)
theorem part_one (m n : ℝ) :
  (∀ x, f m x < 0 ↔ -2 < x ∧ x < n) →
  m = 3/2 ∧ n = 1/2 := by sorry

-- Part (2)
theorem part_two (m : ℝ) :
  (∀ x ∈ Set.Icc m (m+1), f m x < 0) →
  m > -Real.sqrt 2 / 2 ∧ m < 0 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3401_340185


namespace NUMINAMATH_CALUDE_print_gift_wrap_price_l3401_340172

/-- The price of print gift wrap per roll -/
def print_price : ℝ := 6

/-- The price of solid color gift wrap per roll -/
def solid_price : ℝ := 4

/-- The total number of rolls sold -/
def total_rolls : ℕ := 480

/-- The total amount of money collected -/
def total_money : ℝ := 2340

/-- The number of print rolls sold -/
def print_rolls : ℕ := 210

theorem print_gift_wrap_price :
  print_price * print_rolls + solid_price * (total_rolls - print_rolls) = total_money :=
sorry

end NUMINAMATH_CALUDE_print_gift_wrap_price_l3401_340172


namespace NUMINAMATH_CALUDE_A_power_difference_l3401_340191

def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 3; 0, 1]

theorem A_power_difference :
  A^20 - 2 • A^19 = !![0, 3; 0, -1] := by sorry

end NUMINAMATH_CALUDE_A_power_difference_l3401_340191


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_l3401_340126

/-- A trapezoid with specific side lengths -/
structure Trapezoid where
  EG : ℝ
  FH : ℝ
  GH : ℝ
  EF : ℝ
  is_trapezoid : EF > GH
  parallel_bases : EF = 2 * GH

/-- The perimeter of a trapezoid -/
def perimeter (t : Trapezoid) : ℝ := t.EG + t.FH + t.GH + t.EF

/-- Theorem: The perimeter of the given trapezoid is 183 units -/
theorem trapezoid_perimeter :
  ∃ t : Trapezoid, t.EG = 35 ∧ t.FH = 40 ∧ t.GH = 36 ∧ perimeter t = 183 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_perimeter_l3401_340126


namespace NUMINAMATH_CALUDE_farm_dogs_left_l3401_340153

/-- Given a farm with dogs and farmhands, calculates the number of dogs left after a morning walk. -/
def dogs_left_after_walk (total_dogs : ℕ) (dog_houses : ℕ) (farmhands : ℕ) (dogs_per_farmhand : ℕ) : ℕ :=
  total_dogs - farmhands * dogs_per_farmhand

/-- Proves that given the specific conditions of the farm, 144 dogs are left after the morning walk. -/
theorem farm_dogs_left : dogs_left_after_walk 156 22 6 2 = 144 := by
  sorry

#eval dogs_left_after_walk 156 22 6 2

end NUMINAMATH_CALUDE_farm_dogs_left_l3401_340153


namespace NUMINAMATH_CALUDE_negative_2023_times_99_l3401_340151

theorem negative_2023_times_99 (p : ℤ) (h : p = (-2023) * 100) : (-2023) * 99 = p + 2023 := by
  sorry

end NUMINAMATH_CALUDE_negative_2023_times_99_l3401_340151


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l3401_340188

theorem simplify_trig_expression :
  Real.sqrt (1 - 2 * Real.sin (Real.pi + 4) * Real.cos (Real.pi + 4)) = Real.cos 4 - Real.sin 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l3401_340188


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l3401_340199

theorem quadratic_equation_solutions :
  (∀ x, 3 * x^2 - 6 * x = 0 ↔ x = 0 ∨ x = 2) ∧
  (∀ x, x^2 + 4 * x - 1 = 0 ↔ x = -2 + Real.sqrt 5 ∨ x = -2 - Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l3401_340199


namespace NUMINAMATH_CALUDE_wednesday_distance_l3401_340157

/-- The number of days Terese runs in a week -/
def running_days : ℕ := 4

/-- The average distance Terese runs per day -/
def average_distance : ℝ := 4

/-- The distance Terese runs on Monday -/
def monday_distance : ℝ := 4.2

/-- The distance Terese runs on Tuesday -/
def tuesday_distance : ℝ := 3.8

/-- The distance Terese runs on Thursday -/
def thursday_distance : ℝ := 4.4

/-- Theorem: Given the running distances on Monday, Tuesday, and Thursday, 
    and the average distance per day, Terese runs 3.6 miles on Wednesday. -/
theorem wednesday_distance : 
  running_days * average_distance - (monday_distance + tuesday_distance + thursday_distance) = 3.6 := by
  sorry

end NUMINAMATH_CALUDE_wednesday_distance_l3401_340157


namespace NUMINAMATH_CALUDE_myrtle_hens_eggs_per_day_l3401_340121

/-- The number of eggs laid by Myrtle's hens -/
theorem myrtle_hens_eggs_per_day :
  ∀ (num_hens : ℕ) (days_gone : ℕ) (eggs_taken : ℕ) (eggs_dropped : ℕ) (eggs_remaining : ℕ),
  num_hens = 3 →
  days_gone = 7 →
  eggs_taken = 12 →
  eggs_dropped = 5 →
  eggs_remaining = 46 →
  ∃ (eggs_per_hen_per_day : ℕ),
    eggs_per_hen_per_day = 3 ∧
    num_hens * eggs_per_hen_per_day * days_gone - eggs_taken - eggs_dropped = eggs_remaining :=
by
  sorry


end NUMINAMATH_CALUDE_myrtle_hens_eggs_per_day_l3401_340121


namespace NUMINAMATH_CALUDE_agreed_period_is_18_months_prove_agreed_period_of_service_l3401_340177

/-- Represents the agreed period of service in months -/
def agreed_period : ℕ := 18

/-- Represents the actual period served in months -/
def actual_period : ℕ := 9

/-- Represents the full payment amount in rupees -/
def full_payment : ℕ := 800

/-- Represents the actual payment received in rupees -/
def actual_payment : ℕ := 400

/-- Theorem stating that the agreed period of service is 18 months -/
theorem agreed_period_is_18_months :
  (actual_payment = full_payment / 2) →
  (actual_period * 2 = agreed_period) :=
by
  sorry

/-- Main theorem proving the agreed period of service -/
theorem prove_agreed_period_of_service :
  agreed_period = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_agreed_period_is_18_months_prove_agreed_period_of_service_l3401_340177


namespace NUMINAMATH_CALUDE_fermat_prime_power_of_two_l3401_340127

theorem fermat_prime_power_of_two (n : ℕ+) : 
  (Nat.Prime (2^(n : ℕ) + 1)) → (∃ k : ℕ, n = 2^k) := by
  sorry

end NUMINAMATH_CALUDE_fermat_prime_power_of_two_l3401_340127


namespace NUMINAMATH_CALUDE_rational_equation_solution_l3401_340190

theorem rational_equation_solution : 
  {x : ℝ | (1 / (x^2 + 9*x - 12) + 1 / (x^2 + 5*x - 14) - 1 / (x^2 - 15*x - 18) = 0) ∧ 
           (x^2 + 9*x - 12 ≠ 0) ∧ (x^2 + 5*x - 14 ≠ 0) ∧ (x^2 - 15*x - 18 ≠ 0)} = 
  {2, -9, 6, -3} := by sorry

end NUMINAMATH_CALUDE_rational_equation_solution_l3401_340190


namespace NUMINAMATH_CALUDE_pigs_joined_l3401_340180

/-- Given an initial number of pigs and a final number of pigs,
    prove that the number of pigs that joined is equal to their difference. -/
theorem pigs_joined (initial final : ℕ) (h : final ≥ initial) :
  final - initial = final - initial :=
by sorry

end NUMINAMATH_CALUDE_pigs_joined_l3401_340180


namespace NUMINAMATH_CALUDE_largest_x_satisfying_equation_l3401_340198

theorem largest_x_satisfying_equation : 
  ∃ (x : ℚ), x = 3/25 ∧ 
  (∀ y : ℚ, y ≥ 0 → Real.sqrt (3 * y) = 5 * y → y ≤ x) ∧
  Real.sqrt (3 * x) = 5 * x := by
  sorry

end NUMINAMATH_CALUDE_largest_x_satisfying_equation_l3401_340198


namespace NUMINAMATH_CALUDE_exists_valid_numbering_9_not_exists_valid_numbering_10_l3401_340124

/-- A convex n-gon with a point inside -/
structure ConvexNGon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  inner_point : ℝ × ℝ
  is_convex : sorry -- Add convexity condition

/-- Numbering of sides and segments -/
def Numbering (n : ℕ) := Fin n → Fin n

/-- Sum of numbers in a triangle -/
def triangle_sum (n : ℕ) (polygon : ConvexNGon n) (numbering : Numbering n) (i : Fin n) : ℕ := sorry

/-- Existence of a valid numbering for n = 9 -/
theorem exists_valid_numbering_9 :
  ∃ (polygon : ConvexNGon 9) (numbering : Numbering 9),
    ∀ (i j : Fin 9), triangle_sum 9 polygon numbering i = triangle_sum 9 polygon numbering j :=
sorry

/-- Non-existence of a valid numbering for n = 10 -/
theorem not_exists_valid_numbering_10 :
  ¬ ∃ (polygon : ConvexNGon 10) (numbering : Numbering 10),
    ∀ (i j : Fin 10), triangle_sum 10 polygon numbering i = triangle_sum 10 polygon numbering j :=
sorry

end NUMINAMATH_CALUDE_exists_valid_numbering_9_not_exists_valid_numbering_10_l3401_340124


namespace NUMINAMATH_CALUDE_arnold_protein_consumption_l3401_340159

/-- Represents the protein content and quantity of a food item -/
structure FoodItem where
  name : String
  proteinPerServing : ℝ
  servings : ℝ

/-- Calculates the total protein from a list of food items -/
def totalProtein (foods : List FoodItem) : ℝ :=
  foods.map (fun f => f.proteinPerServing * f.servings) |>.sum

/-- Arnold's protein consumption theorem -/
theorem arnold_protein_consumption 
  (collagen : FoodItem)
  (protein_powder : FoodItem)
  (steak : FoodItem)
  (h1 : collagen.name = "Collagen Powder")
  (h2 : collagen.proteinPerServing = 9)  -- 18 grams / 2 scoops
  (h3 : collagen.servings = 1)
  (h4 : protein_powder.name = "Protein Powder")
  (h5 : protein_powder.proteinPerServing = 21)
  (h6 : protein_powder.servings = 1)
  (h7 : steak.name = "Steak")
  (h8 : steak.proteinPerServing = 56)
  (h9 : steak.servings = 1) :
  totalProtein [collagen, protein_powder, steak] = 86 := by
  sorry


end NUMINAMATH_CALUDE_arnold_protein_consumption_l3401_340159


namespace NUMINAMATH_CALUDE_team_a_prefers_best_of_five_l3401_340173

/-- Represents the probability of Team A winning a non-deciding game -/
def team_a_win_prob : ℝ := 0.6

/-- Represents the probability of Team B winning a non-deciding game -/
def team_b_win_prob : ℝ := 0.4

/-- Represents the probability of either team winning a deciding game -/
def deciding_game_win_prob : ℝ := 0.5

/-- Calculates the probability of Team A winning a best-of-three series -/
def best_of_three_win_prob : ℝ := 
  team_a_win_prob^2 + 2 * team_a_win_prob * team_b_win_prob * deciding_game_win_prob

/-- Calculates the probability of Team A winning a best-of-five series -/
def best_of_five_win_prob : ℝ := 
  team_a_win_prob^3 + 
  3 * team_a_win_prob^2 * team_b_win_prob + 
  6 * team_a_win_prob^2 * team_b_win_prob^2 * deciding_game_win_prob

/-- Theorem stating that Team A has a higher probability of winning in a best-of-five series -/
theorem team_a_prefers_best_of_five : best_of_five_win_prob > best_of_three_win_prob := by
  sorry

end NUMINAMATH_CALUDE_team_a_prefers_best_of_five_l3401_340173


namespace NUMINAMATH_CALUDE_sammy_gift_wrapping_l3401_340193

/-- The number of gifts Sammy can wrap -/
def num_gifts : ℕ := 8

/-- The length of ribbon required for each gift in meters -/
def ribbon_per_gift : ℝ := 1.5

/-- The total length of Tom's ribbon in meters -/
def total_ribbon : ℝ := 15

/-- The length of ribbon left after wrapping all gifts in meters -/
def ribbon_left : ℝ := 3

/-- Theorem stating that the number of gifts Sammy can wrap is correct -/
theorem sammy_gift_wrapping :
  (↑num_gifts : ℝ) * ribbon_per_gift = total_ribbon - ribbon_left :=
by sorry

end NUMINAMATH_CALUDE_sammy_gift_wrapping_l3401_340193


namespace NUMINAMATH_CALUDE_problem_statement_l3401_340179

theorem problem_statement (a b m : ℝ) : 
  2^a = m ∧ 3^b = m ∧ a * b ≠ 0 ∧ 
  ∃ (k : ℝ), a + k = a * b ∧ a * b + k = b → 
  m = Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3401_340179


namespace NUMINAMATH_CALUDE_optimal_chip_purchase_l3401_340148

/-- Represents the purchase of chips with given constraints -/
structure ChipPurchase where
  priceA : ℕ  -- Unit price of type A chips
  priceB : ℕ  -- Unit price of type B chips
  quantityA : ℕ  -- Quantity of type A chips
  quantityB : ℕ  -- Quantity of type B chips
  total_cost : ℕ  -- Total cost of the purchase

/-- Theorem stating the optimal purchase and minimum cost -/
theorem optimal_chip_purchase :
  ∃ (purchase : ChipPurchase),
    -- Conditions
    purchase.priceB = purchase.priceA + 9 ∧
    purchase.quantityA * purchase.priceA = 3120 ∧
    purchase.quantityB * purchase.priceB = 4200 ∧
    purchase.quantityA = purchase.quantityB ∧
    purchase.quantityA + purchase.quantityB = 200 ∧
    4 * purchase.quantityA ≥ purchase.quantityB ∧
    3 * purchase.quantityA ≤ purchase.quantityB ∧
    -- Correct answer
    purchase.priceA = 26 ∧
    purchase.priceB = 35 ∧
    purchase.quantityA = 50 ∧
    purchase.quantityB = 150 ∧
    purchase.total_cost = 6550 ∧
    -- Minimum cost property
    (∀ (other : ChipPurchase),
      other.priceB = other.priceA + 9 →
      other.quantityA + other.quantityB = 200 →
      4 * other.quantityA ≥ other.quantityB →
      3 * other.quantityA ≤ other.quantityB →
      other.total_cost ≥ purchase.total_cost) :=
by
  sorry


end NUMINAMATH_CALUDE_optimal_chip_purchase_l3401_340148


namespace NUMINAMATH_CALUDE_f_properties_l3401_340169

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + (a + 1) / 2 * x^2 + 1

theorem f_properties (a : ℝ) :
  -- Part 1
  (a = -1/2 →
    ∃ (max min : ℝ),
      (∀ x ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), f a x ≤ max) ∧
      (∃ x ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), f a x = max) ∧
      (∀ x ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), f a x ≥ min) ∧
      (∃ x ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), f a x = min) ∧
      max = 1/2 + (Real.exp 1)^2/4 ∧
      min = 5/4) ∧
  -- Part 2
  (∃ (mono : ℝ → Prop), mono a ↔
    (∀ x y, 0 < x → 0 < y → x < y → (f a x < f a y ∨ f a x > f a y ∨ f a x = f a y))) ∧
  -- Part 3
  (-1 < a → a < 0 →
    (∀ x, x > 0 → f a x > 1 + a/2 * Real.log (-a)) ∧
    (1/Real.exp 1 - 1 < a ∧ a < 0)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3401_340169


namespace NUMINAMATH_CALUDE_quadratic_root_implies_coefficient_l3401_340130

theorem quadratic_root_implies_coefficient 
  (b c : ℝ) 
  (h : ∃ x : ℂ, x^2 + b*x + c = 0 ∧ x = 2 + I) : 
  b = -4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_coefficient_l3401_340130


namespace NUMINAMATH_CALUDE_ashok_subjects_l3401_340101

theorem ashok_subjects (average_all : ℝ) (average_five : ℝ) (sixth_subject : ℝ) 
  (h1 : average_all = 72)
  (h2 : average_five = 74)
  (h3 : sixth_subject = 62) :
  ∃ n : ℕ, n = 6 ∧ n * average_all = 5 * average_five + sixth_subject :=
by
  sorry

end NUMINAMATH_CALUDE_ashok_subjects_l3401_340101


namespace NUMINAMATH_CALUDE_max_quotient_value_l3401_340197

theorem max_quotient_value (a b : ℝ) (ha : 100 ≤ a ∧ a ≤ 500) (hb : 400 ≤ b ∧ b ≤ 1000) :
  (∀ x y, 100 ≤ x ∧ x ≤ 500 → 400 ≤ y ∧ y ≤ 1000 → b / a ≥ y / x) ∧ b / a ≤ 10 :=
sorry

end NUMINAMATH_CALUDE_max_quotient_value_l3401_340197


namespace NUMINAMATH_CALUDE_distance_between_ports_l3401_340131

/-- The distance between two ports given ship and current speeds and time difference -/
theorem distance_between_ports (ship_speed : ℝ) (current_speed : ℝ) (time_diff : ℝ) :
  ship_speed > current_speed →
  ship_speed = 24 →
  current_speed = 3 →
  time_diff = 5 →
  ∃ (distance : ℝ),
    distance / (ship_speed - current_speed) - distance / (ship_speed + current_speed) = time_diff ∧
    distance = 200 / 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_ports_l3401_340131


namespace NUMINAMATH_CALUDE_rational_function_property_l3401_340158

theorem rational_function_property (f : ℚ → ℚ) 
  (h : ∀ x y : ℚ, f (x + y) = 2 * f (x / 2) + 3 * f (y / 3)) :
  ∃ a : ℚ, ∀ x : ℚ, f x = a * x :=
by sorry

end NUMINAMATH_CALUDE_rational_function_property_l3401_340158


namespace NUMINAMATH_CALUDE_angle_bisection_limit_l3401_340150

/-- The limit of an alternating series of angle bisections in a 60° angle -/
theorem angle_bisection_limit (θ : Real) (h : θ = 60) : 
  (∑' n, (-1)^n * (1/2)^(n+1)) * θ = 20 := by
  sorry

end NUMINAMATH_CALUDE_angle_bisection_limit_l3401_340150


namespace NUMINAMATH_CALUDE_odd_function_implies_a_eq_two_l3401_340122

/-- The function f(x) = (x + a - 2)(2x² + a - 1) -/
def f (a : ℝ) (x : ℝ) : ℝ := (x + a - 2) * (2 * x^2 + a - 1)

/-- f is an odd function -/
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_implies_a_eq_two :
  ∀ a : ℝ, is_odd_function (f a) → a = 2 := by sorry

end NUMINAMATH_CALUDE_odd_function_implies_a_eq_two_l3401_340122


namespace NUMINAMATH_CALUDE_chess_game_probability_l3401_340140

theorem chess_game_probability (prob_A_win prob_A_not_lose : ℝ) 
  (h1 : prob_A_win = 0.4)
  (h2 : prob_A_not_lose = 0.8) : 
  1 - prob_A_not_lose = 0.6 :=
by
  sorry


end NUMINAMATH_CALUDE_chess_game_probability_l3401_340140


namespace NUMINAMATH_CALUDE_original_savings_l3401_340110

def lindas_savings : ℚ → Prop :=
  λ s => (1 / 4 : ℚ) * s = 450

theorem original_savings : ∃ s : ℚ, lindas_savings s ∧ s = 1800 :=
  sorry

end NUMINAMATH_CALUDE_original_savings_l3401_340110


namespace NUMINAMATH_CALUDE_perfect_square_primes_no_perfect_square_primes_l3401_340163

theorem perfect_square_primes (p : Nat) : Prime p → (∃ n : Nat, (7^(p-1) - 1) / p = n^2) ↔ p = 3 :=
sorry

theorem no_perfect_square_primes (p : Nat) : Prime p → ¬∃ n : Nat, (11^(p-1) - 1) / p = n^2 :=
sorry

end NUMINAMATH_CALUDE_perfect_square_primes_no_perfect_square_primes_l3401_340163


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l3401_340114

theorem cubic_equation_solution :
  let f (x : ℂ) := (x - 2)^3 + (x - 6)^3
  ∃ (s : Finset ℂ), s.card = 3 ∧ 
    (∀ x ∈ s, f x = 0) ∧
    (∀ x, f x = 0 → x ∈ s) ∧
    (4 ∈ s) ∧ 
    (4 + 2 * Complex.I * Real.sqrt 3 ∈ s) ∧ 
    (4 - 2 * Complex.I * Real.sqrt 3 ∈ s) :=
by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l3401_340114


namespace NUMINAMATH_CALUDE_f_properties_l3401_340129

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 1| + |x + 1|

-- Define the range of f
def range_f : Set ℝ := {y : ℝ | ∃ x, f x = y}

-- Theorem statement
theorem f_properties :
  (range_f = {y : ℝ | y ≥ 3/2}) ∧
  (∀ a : ℝ, a ∈ range_f → |a - 1| + |a + 1| > 3/(2*a) ∧ 3/(2*a) > 7/2 - 2*a) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l3401_340129


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l3401_340112

theorem binomial_coefficient_equality (m : ℕ) : 
  (Nat.choose 17 (3*m - 1) = Nat.choose 17 (2*m + 3)) → (m = 3 ∨ m = 4) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l3401_340112


namespace NUMINAMATH_CALUDE_vasily_expected_salary_l3401_340144

/-- Represents the salary distribution for graduates --/
structure SalaryDistribution where
  high : ℝ  -- Salary for 1/5 of graduates
  medium : ℝ  -- Salary for 1/10 of graduates
  low : ℝ  -- Salary for 1/20 of graduates
  default : ℝ  -- Salary for the rest

/-- Represents the given conditions of the problem --/
structure ProblemConditions where
  total_students : ℕ
  successful_graduates : ℕ
  non_graduate_salary : ℝ
  graduate_salary_dist : SalaryDistribution
  education_duration : ℕ

def expected_salary (conditions : ProblemConditions) : ℝ :=
  sorry

theorem vasily_expected_salary 
  (conditions : ProblemConditions)
  (h1 : conditions.total_students = 300)
  (h2 : conditions.successful_graduates = 270)
  (h3 : conditions.non_graduate_salary = 25000)
  (h4 : conditions.graduate_salary_dist.high = 60000)
  (h5 : conditions.graduate_salary_dist.medium = 80000)
  (h6 : conditions.graduate_salary_dist.low = 25000)
  (h7 : conditions.graduate_salary_dist.default = 40000)
  (h8 : conditions.education_duration = 4) :
  expected_salary conditions = 45025 :=
sorry

end NUMINAMATH_CALUDE_vasily_expected_salary_l3401_340144


namespace NUMINAMATH_CALUDE_isabellas_hair_length_l3401_340194

/-- Isabella's hair length problem -/
theorem isabellas_hair_length :
  ∀ (current_length future_length growth : ℕ),
  future_length = 22 →
  growth = 4 →
  future_length = current_length + growth →
  current_length = 18 := by
  sorry

end NUMINAMATH_CALUDE_isabellas_hair_length_l3401_340194


namespace NUMINAMATH_CALUDE_blue_pill_cost_proof_l3401_340123

def treatment_duration : ℕ := 3 * 7 -- 3 weeks in days

def daily_blue_pills : ℕ := 1
def daily_yellow_pills : ℕ := 1

def total_cost : ℚ := 735

def blue_pill_cost : ℚ := 18.5
def yellow_pill_cost : ℚ := blue_pill_cost - 2

theorem blue_pill_cost_proof :
  blue_pill_cost * (treatment_duration * daily_blue_pills) +
  yellow_pill_cost * (treatment_duration * daily_yellow_pills) = total_cost :=
by sorry

end NUMINAMATH_CALUDE_blue_pill_cost_proof_l3401_340123


namespace NUMINAMATH_CALUDE_smaller_of_reciprocal_and_sine_interval_length_l3401_340108

open Real

theorem smaller_of_reciprocal_and_sine (x : ℝ) :
  (min (1/x) (sin x) > 1/2) ↔ (π/6 < x ∧ x < 5*π/6) :=
sorry

theorem interval_length : 
  (5*π/6 - π/6 : ℝ) = 2*π/3 :=
sorry

end NUMINAMATH_CALUDE_smaller_of_reciprocal_and_sine_interval_length_l3401_340108


namespace NUMINAMATH_CALUDE_line_properties_l3401_340116

-- Define a type for lines in 2D Cartesian plane
structure Line where
  slope : ℝ
  y_intercept : ℝ

-- Define a function to check if two lines are perpendicular
def are_perpendicular (l1 l2 : Line) : Prop :=
  l1.slope * l2.slope = -1

-- Define a function to represent a line passing through (1,1) with slope 1
def line_through_1_1_slope_1 (x : ℝ) : Prop :=
  x ≠ 1 → ∃ y : ℝ, y - 1 = x - 1

-- Theorem stating the properties we want to prove
theorem line_properties :
  ∀ (l1 l2 : Line),
    (are_perpendicular l1 l2 → l1.slope * l2.slope = -1) ∧
    line_through_1_1_slope_1 2 := by
  sorry

end NUMINAMATH_CALUDE_line_properties_l3401_340116


namespace NUMINAMATH_CALUDE_principal_is_800_l3401_340192

/-- Calculates the principal amount given the final amount, interest rate, and time -/
def calculate_principal (amount : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  amount / (1 + rate * time)

/-- Theorem stating that the principal is 800 given the problem conditions -/
theorem principal_is_800 : 
  let amount : ℚ := 896
  let rate : ℚ := 5 / 100
  let time : ℚ := 12 / 5
  calculate_principal amount rate time = 800 := by sorry

end NUMINAMATH_CALUDE_principal_is_800_l3401_340192


namespace NUMINAMATH_CALUDE_tan_3285_degrees_l3401_340132

theorem tan_3285_degrees : Real.tan (3285 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_3285_degrees_l3401_340132


namespace NUMINAMATH_CALUDE_theta_range_l3401_340149

theorem theta_range (θ : Real) : 
  (∀ x : Real, x ∈ Set.Icc 0 1 → x^2 * Real.cos θ - x*(1-x) + (1-x)^2 * Real.sin θ > 0) → 
  π/12 < θ ∧ θ < 5*π/12 := by
sorry

end NUMINAMATH_CALUDE_theta_range_l3401_340149


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3401_340120

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (1 - 4 * x) = 5 → x = -6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3401_340120


namespace NUMINAMATH_CALUDE_function_inequality_l3401_340103

-- Define the interval (3,7)
def openInterval : Set ℝ := {x : ℝ | 3 < x ∧ x < 7}

-- Define the theorem
theorem function_inequality
  (f g : ℝ → ℝ)
  (h_diff_f : DifferentiableOn ℝ f openInterval)
  (h_diff_g : DifferentiableOn ℝ g openInterval)
  (h_deriv : ∀ x ∈ openInterval, deriv f x < deriv g x) :
  ∀ x ∈ openInterval, f x + g 3 < g x + f 3 :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_l3401_340103


namespace NUMINAMATH_CALUDE_jude_current_age_l3401_340133

/-- Heath's age today -/
def heath_age_today : ℕ := 16

/-- Heath's age in 5 years -/
def heath_age_future : ℕ := heath_age_today + 5

/-- Jude's age in 5 years -/
def jude_age_future : ℕ := heath_age_future / 3

/-- Jude's age today -/
def jude_age_today : ℕ := jude_age_future - 5

/-- Theorem stating Jude's age today -/
theorem jude_current_age : jude_age_today = 2 := by
  sorry

end NUMINAMATH_CALUDE_jude_current_age_l3401_340133


namespace NUMINAMATH_CALUDE_equation_has_real_roots_l3401_340160

theorem equation_has_real_roots (a b c p q : ℝ) :
  ∃ x : ℝ, (a^2 / (x - p) + b^2 / (x - q) - c = 0) :=
by sorry

end NUMINAMATH_CALUDE_equation_has_real_roots_l3401_340160


namespace NUMINAMATH_CALUDE_student_tickets_sold_l3401_340125

theorem student_tickets_sold (adult_price student_price total_tickets total_revenue : ℕ) 
  (h1 : adult_price = 6)
  (h2 : student_price = 3)
  (h3 : total_tickets = 846)
  (h4 : total_revenue = 3846) :
  ∃ (adult_tickets student_tickets : ℕ),
    adult_tickets + student_tickets = total_tickets ∧
    adult_price * adult_tickets + student_price * student_tickets = total_revenue ∧
    student_tickets = 410 := by
  sorry

end NUMINAMATH_CALUDE_student_tickets_sold_l3401_340125


namespace NUMINAMATH_CALUDE_sin_alpha_value_l3401_340128

theorem sin_alpha_value (α : Real) (h1 : π/2 < α ∧ α < π) (h2 : 3 * Real.sin (2 * α) = Real.cos α) : 
  Real.sin α = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l3401_340128


namespace NUMINAMATH_CALUDE_marriage_year_proof_l3401_340138

def year_of_marriage : ℕ := 1980
def year_child1_born : ℕ := 1982
def year_child2_born : ℕ := 1984
def reference_year : ℕ := 1986

theorem marriage_year_proof :
  (reference_year - year_child1_born) + (reference_year - year_child2_born) = reference_year - year_of_marriage :=
by sorry

end NUMINAMATH_CALUDE_marriage_year_proof_l3401_340138


namespace NUMINAMATH_CALUDE_units_digit_of_special_three_digit_number_l3401_340184

/-- The product of digits of a three-digit number -/
def P (n : ℕ) : ℕ :=
  (n / 100) * ((n / 10) % 10) * (n % 10)

/-- The sum of digits of a three-digit number -/
def S (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

/-- A three-digit number is between 100 and 999 -/
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem units_digit_of_special_three_digit_number (N : ℕ) 
  (h1 : is_three_digit N) 
  (h2 : N = P N + S N) : 
  N % 10 = 9 := by
sorry

end NUMINAMATH_CALUDE_units_digit_of_special_three_digit_number_l3401_340184


namespace NUMINAMATH_CALUDE_optimal_base_side_l3401_340181

/-- A lidless water tank with square base -/
structure WaterTank where
  volume : ℝ
  baseSide : ℝ
  height : ℝ

/-- The surface area of a lidless water tank -/
def surfaceArea (tank : WaterTank) : ℝ :=
  tank.baseSide ^ 2 + 4 * tank.baseSide * tank.height

/-- The volume constraint for the water tank -/
def volumeConstraint (tank : WaterTank) : Prop :=
  tank.volume = tank.baseSide ^ 2 * tank.height

/-- Theorem: The base side length that minimizes the surface area of a lidless water tank
    with volume 256 cubic units and a square base is 8 units -/
theorem optimal_base_side :
  ∃ (tank : WaterTank),
    tank.volume = 256 ∧
    volumeConstraint tank ∧
    (∀ (other : WaterTank),
      other.volume = 256 →
      volumeConstraint other →
      surfaceArea tank ≤ surfaceArea other) ∧
    tank.baseSide = 8 :=
  sorry

end NUMINAMATH_CALUDE_optimal_base_side_l3401_340181


namespace NUMINAMATH_CALUDE_all_days_equal_availability_l3401_340161

-- Define the days of the week
inductive Day
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday

-- Define the team members
inductive Member
| Alice
| Bob
| Charlie
| Diana

-- Define a function to represent availability
def isAvailable (m : Member) (d : Day) : Bool :=
  match m, d with
  | Member.Alice, Day.Monday => false
  | Member.Alice, Day.Thursday => false
  | Member.Bob, Day.Tuesday => false
  | Member.Bob, Day.Wednesday => false
  | Member.Bob, Day.Friday => false
  | Member.Charlie, Day.Wednesday => false
  | Member.Charlie, Day.Thursday => false
  | Member.Charlie, Day.Friday => false
  | Member.Diana, Day.Monday => false
  | Member.Diana, Day.Tuesday => false
  | _, _ => true

-- Count available members for a given day
def availableCount (d : Day) : Nat :=
  (List.filter (fun m => isAvailable m d) [Member.Alice, Member.Bob, Member.Charlie, Member.Diana]).length

-- Theorem: All days have equal availability
theorem all_days_equal_availability :
  ∀ d1 d2 : Day, availableCount d1 = availableCount d2 :=
sorry

end NUMINAMATH_CALUDE_all_days_equal_availability_l3401_340161


namespace NUMINAMATH_CALUDE_student_hamster_difference_l3401_340117

/-- The number of third-grade classrooms -/
def num_classrooms : ℕ := 5

/-- The number of students in each classroom -/
def students_per_classroom : ℕ := 20

/-- The number of hamsters in each classroom -/
def hamsters_per_classroom : ℕ := 1

/-- The total number of students in all classrooms -/
def total_students : ℕ := num_classrooms * students_per_classroom

/-- The total number of hamsters in all classrooms -/
def total_hamsters : ℕ := num_classrooms * hamsters_per_classroom

theorem student_hamster_difference :
  total_students - total_hamsters = 95 := by
  sorry

end NUMINAMATH_CALUDE_student_hamster_difference_l3401_340117


namespace NUMINAMATH_CALUDE_cycle_price_proof_l3401_340196

theorem cycle_price_proof (selling_price : ℝ) (gain_percent : ℝ) (original_price : ℝ) : 
  selling_price = 1620 → 
  gain_percent = 8 → 
  selling_price = original_price * (1 + gain_percent / 100) → 
  original_price = 1500 := by
  sorry

#check cycle_price_proof

end NUMINAMATH_CALUDE_cycle_price_proof_l3401_340196


namespace NUMINAMATH_CALUDE_edward_money_theorem_l3401_340154

/-- Represents the amount of money Edward had before spending --/
def initial_amount : ℝ := 22

/-- Represents the amount Edward spent on books --/
def spent_amount : ℝ := 16

/-- Represents the amount Edward has left --/
def remaining_amount : ℝ := 6

/-- Represents the number of books Edward bought --/
def number_of_books : ℕ := 92

/-- Theorem stating that the initial amount equals the sum of spent and remaining amounts --/
theorem edward_money_theorem :
  initial_amount = spent_amount + remaining_amount := by sorry

end NUMINAMATH_CALUDE_edward_money_theorem_l3401_340154


namespace NUMINAMATH_CALUDE_science_marks_calculation_l3401_340189

def average_marks : ℝ := 73
def num_subjects : ℕ := 5
def math_marks : ℝ := 76
def social_marks : ℝ := 82
def english_marks : ℝ := 67
def biology_marks : ℝ := 75

theorem science_marks_calculation : 
  ∃ (science_marks : ℝ), 
    average_marks * num_subjects = 
      math_marks + social_marks + english_marks + biology_marks + science_marks ∧
    science_marks = 65 := by
  sorry

end NUMINAMATH_CALUDE_science_marks_calculation_l3401_340189


namespace NUMINAMATH_CALUDE_sum_of_x_solutions_is_zero_l3401_340147

theorem sum_of_x_solutions_is_zero (y : ℝ) (h1 : y = 9) (h2 : ∃ x : ℝ, x^2 + y^2 = 169) :
  ∃ x₁ x₂ : ℝ, x₁^2 + y^2 = 169 ∧ x₂^2 + y^2 = 169 ∧ x₁ + x₂ = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_x_solutions_is_zero_l3401_340147


namespace NUMINAMATH_CALUDE_seeds_per_small_garden_l3401_340106

/-- Proves that given the initial number of seeds, seeds planted in the big garden,
    and the number of small gardens, the number of seeds in each small garden is correct. -/
theorem seeds_per_small_garden 
  (total_seeds : ℕ) 
  (big_garden_seeds : ℕ) 
  (small_gardens : ℕ) 
  (h1 : total_seeds = 56)
  (h2 : big_garden_seeds = 35)
  (h3 : small_gardens = 7)
  : (total_seeds - big_garden_seeds) / small_gardens = 3 := by
  sorry

end NUMINAMATH_CALUDE_seeds_per_small_garden_l3401_340106


namespace NUMINAMATH_CALUDE_pencils_for_classroom_l3401_340176

/-- Given a classroom with 4 children where each child receives 2 pencils,
    prove that the teacher needs to give out 8 pencils in total. -/
theorem pencils_for_classroom (num_children : ℕ) (pencils_per_child : ℕ) 
  (h1 : num_children = 4) (h2 : pencils_per_child = 2) : 
  num_children * pencils_per_child = 8 := by
  sorry

end NUMINAMATH_CALUDE_pencils_for_classroom_l3401_340176


namespace NUMINAMATH_CALUDE_amys_net_earnings_result_l3401_340178

/-- Calculates Amy's net earnings for a week given her daily work details and tax rate -/
def amys_net_earnings (day1_hours day1_rate day1_tips day1_bonus : ℝ)
                      (day2_hours day2_rate day2_tips : ℝ)
                      (day3_hours day3_rate day3_tips : ℝ)
                      (day4_hours day4_rate day4_tips day4_overtime : ℝ)
                      (day5_hours day5_rate day5_tips : ℝ)
                      (tax_rate : ℝ) : ℝ :=
  let day1_earnings := day1_hours * day1_rate + day1_tips + day1_bonus
  let day2_earnings := day2_hours * day2_rate + day2_tips
  let day3_earnings := day3_hours * day3_rate + day3_tips
  let day4_earnings := day4_hours * day4_rate + day4_tips + day4_overtime
  let day5_earnings := day5_hours * day5_rate + day5_tips
  let gross_earnings := day1_earnings + day2_earnings + day3_earnings + day4_earnings + day5_earnings
  let taxes := tax_rate * gross_earnings
  gross_earnings - taxes

/-- Theorem stating that Amy's net earnings for the week are $118.58 -/
theorem amys_net_earnings_result :
  amys_net_earnings 4 3 6 10 6 4 7 3 5 2 5 3.5 8 5 7 4 5 0.15 = 118.58 := by
  sorry

#eval amys_net_earnings 4 3 6 10 6 4 7 3 5 2 5 3.5 8 5 7 4 5 0.15

end NUMINAMATH_CALUDE_amys_net_earnings_result_l3401_340178


namespace NUMINAMATH_CALUDE_difference_of_squares_a_l3401_340171

theorem difference_of_squares_a (a : ℝ) : (a + 1) * (a - 1) = a^2 - 1 := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_a_l3401_340171


namespace NUMINAMATH_CALUDE_discriminant_of_specific_quadratic_l3401_340167

/-- The discriminant of a quadratic equation ax^2 + bx + c -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- Theorem: The discriminant of 5x^2 - 11x + 4 is 41 -/
theorem discriminant_of_specific_quadratic : discriminant 5 (-11) 4 = 41 := by
  sorry

end NUMINAMATH_CALUDE_discriminant_of_specific_quadratic_l3401_340167


namespace NUMINAMATH_CALUDE_eighteen_is_counterexample_l3401_340119

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_counterexample (n : ℕ) : Prop :=
  ¬(is_prime n) ∧ ¬(is_prime (n - 3))

theorem eighteen_is_counterexample : is_counterexample 18 := by
  sorry

end NUMINAMATH_CALUDE_eighteen_is_counterexample_l3401_340119


namespace NUMINAMATH_CALUDE_hexagon_area_is_52_l3401_340102

-- Define the hexagon vertices
def hexagon_vertices : List (ℝ × ℝ) := [
  (0, 0), (2, 4), (5, 4), (7, 0), (5, -4), (2, -4)
]

-- Function to calculate the area of a trapezoid given its four vertices
def trapezoid_area (v1 v2 v3 v4 : ℝ × ℝ) : ℝ := sorry

-- Function to calculate the area of the hexagon
def hexagon_area (vertices : List (ℝ × ℝ)) : ℝ := sorry

-- Theorem stating that the area of the hexagon is 52 square units
theorem hexagon_area_is_52 : hexagon_area hexagon_vertices = 52 := by sorry

end NUMINAMATH_CALUDE_hexagon_area_is_52_l3401_340102


namespace NUMINAMATH_CALUDE_special_sequence_bijective_l3401_340143

/-- An integer sequence with specific properties -/
def SpecialSequence (a : ℕ → ℤ) : Prop :=
  (∀ n : ℕ, ∃ k > n, a k > 0) ∧  -- Infinitely many positive integers
  (∀ n : ℕ, ∃ k > n, a k < 0) ∧  -- Infinitely many negative integers
  (∀ n : ℕ+, Function.Injective (fun i => a i % n))  -- Distinct remainders

/-- The main theorem -/
theorem special_sequence_bijective (a : ℕ → ℤ) (h : SpecialSequence a) :
  Function.Bijective a :=
sorry

end NUMINAMATH_CALUDE_special_sequence_bijective_l3401_340143


namespace NUMINAMATH_CALUDE_frank_cans_total_l3401_340104

/-- The number of cans Frank picked up on Saturday and Sunday combined -/
def total_cans (saturday_bags : ℕ) (sunday_bags : ℕ) (cans_per_bag : ℕ) : ℕ :=
  (saturday_bags + sunday_bags) * cans_per_bag

/-- Theorem stating that Frank picked up 40 cans in total -/
theorem frank_cans_total : total_cans 5 3 5 = 40 := by
  sorry

end NUMINAMATH_CALUDE_frank_cans_total_l3401_340104


namespace NUMINAMATH_CALUDE_power_fraction_equality_l3401_340141

theorem power_fraction_equality : (2^2017 + 2^2013) / (2^2017 - 2^2013) = 17/15 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_equality_l3401_340141


namespace NUMINAMATH_CALUDE_intersection_of_planes_intersects_at_least_one_line_l3401_340139

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the property of two lines being skew
variable (skew : Line → Line → Prop)

-- Define the property of a line being in a plane
variable (in_plane : Line → Plane → Prop)

-- Define the intersection of two planes
variable (plane_intersection : Plane → Plane → Line)

-- Define the property of a line intersecting another line
variable (intersects : Line → Line → Prop)

-- Theorem statement
theorem intersection_of_planes_intersects_at_least_one_line
  (a b l : Line) (α β : Plane)
  (h1 : skew a b)
  (h2 : in_plane a α)
  (h3 : in_plane b β)
  (h4 : plane_intersection α β = l) :
  intersects l a ∨ intersects l b :=
sorry

end NUMINAMATH_CALUDE_intersection_of_planes_intersects_at_least_one_line_l3401_340139


namespace NUMINAMATH_CALUDE_exponent_multiplication_l3401_340168

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l3401_340168


namespace NUMINAMATH_CALUDE_projection_theorem_l3401_340135

def vector_a : ℝ × ℝ := (-2, -4)

theorem projection_theorem (b : ℝ × ℝ) 
  (angle_ab : Real.cos (120 * π / 180) = -1/2)
  (magnitude_b : Real.sqrt (b.1^2 + b.2^2) = Real.sqrt 5) :
  let projection := (Real.sqrt ((vector_a.1)^2 + (vector_a.2)^2)) * 
                    Real.cos (120 * π / 180)
  projection = -Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_projection_theorem_l3401_340135


namespace NUMINAMATH_CALUDE_opposite_teal_is_blue_l3401_340136

-- Define the colors
inductive Color
| Blue | Yellow | Orange | Black | Teal | Violet

-- Define a cube type
structure Cube where
  faces : Fin 6 → Color
  unique_colors : ∀ i j, i ≠ j → faces i ≠ faces j

-- Define the views
def view1 (c : Cube) : Prop :=
  c.faces 0 = Color.Yellow ∧ c.faces 1 = Color.Blue ∧ c.faces 2 = Color.Orange

def view2 (c : Cube) : Prop :=
  c.faces 0 = Color.Yellow ∧ c.faces 1 = Color.Black ∧ c.faces 2 = Color.Orange

def view3 (c : Cube) : Prop :=
  c.faces 0 = Color.Yellow ∧ c.faces 1 = Color.Violet ∧ c.faces 2 = Color.Orange

-- Theorem statement
theorem opposite_teal_is_blue (c : Cube) 
  (h1 : view1 c) (h2 : view2 c) (h3 : view3 c) :
  ∃ i j, i ≠ j ∧ c.faces i = Color.Teal ∧ c.faces j = Color.Blue ∧ 
  (∀ k, k ≠ i → k ≠ j → c.faces k ≠ Color.Teal ∧ c.faces k ≠ Color.Blue) :=
sorry

end NUMINAMATH_CALUDE_opposite_teal_is_blue_l3401_340136


namespace NUMINAMATH_CALUDE_min_value_function_l3401_340164

theorem min_value_function (x : ℝ) (h : x > 0) : 2 + 4*x + 1/x ≥ 6 ∧ ∃ y > 0, 2 + 4*y + 1/y = 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_function_l3401_340164


namespace NUMINAMATH_CALUDE_problem_statement_l3401_340134

theorem problem_statement : 
  (∀ x : ℝ, x^2 - x + 1 > 0) ∨ ¬(∃ x : ℝ, x > 0 ∧ Real.sin x > 1) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3401_340134


namespace NUMINAMATH_CALUDE_square_of_99_l3401_340113

theorem square_of_99 : 99 * 99 = 9801 := by
  sorry

end NUMINAMATH_CALUDE_square_of_99_l3401_340113


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3401_340182

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x - 2| < 1} = {x : ℝ | 1 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3401_340182


namespace NUMINAMATH_CALUDE_negative_m_exponent_division_l3401_340183

theorem negative_m_exponent_division (m : ℝ) : (-m)^6 / (-m)^3 = -m^3 := by sorry

end NUMINAMATH_CALUDE_negative_m_exponent_division_l3401_340183


namespace NUMINAMATH_CALUDE_bus_ticket_savings_l3401_340156

/-- The price of a single bus ticket in dollars -/
def single_ticket_price : ℚ := 1.50

/-- The price of a package of 5 bus tickets in dollars -/
def package_price : ℚ := 5.75

/-- The number of tickets required -/
def required_tickets : ℕ := 40

/-- The number of packages needed to cover the required tickets -/
def packages_needed : ℕ := required_tickets / 5

theorem bus_ticket_savings : 
  (required_tickets : ℚ) * single_ticket_price - 
  (packages_needed : ℚ) * package_price = 14 := by sorry

end NUMINAMATH_CALUDE_bus_ticket_savings_l3401_340156


namespace NUMINAMATH_CALUDE_total_money_after_redistribution_l3401_340165

/-- Represents the money redistribution process among three friends. -/
def moneyRedistribution (initialAmy : ℝ) (initialJan : ℝ) (initialToy : ℝ) : ℝ :=
  let afterAmy := initialAmy - 2 * (initialJan + initialToy) + 3 * initialJan + 3 * initialToy
  let afterJan := 3 * (initialAmy - 2 * (initialJan + initialToy)) + 
                  (3 * initialJan - 2 * (initialAmy - 2 * (initialJan + initialToy) + 3 * initialToy)) + 
                  3 * 3 * initialToy
  let afterToy := 27  -- Given condition
  afterAmy + afterJan + afterToy

/-- Theorem stating that the total amount after redistribution is 243 when Toy starts and ends with 27. -/
theorem total_money_after_redistribution :
  ∀ (initialAmy : ℝ) (initialJan : ℝ),
  moneyRedistribution initialAmy initialJan 27 = 243 :=
by
  sorry

#eval moneyRedistribution 0 0 27  -- For verification

end NUMINAMATH_CALUDE_total_money_after_redistribution_l3401_340165


namespace NUMINAMATH_CALUDE_modified_cube_edge_count_l3401_340155

/-- Represents a cube with a given side length -/
structure Cube where
  sideLength : ℝ

/-- Represents the result of removing smaller cubes from the corners of a larger cube -/
structure ModifiedCube where
  originalCube : Cube
  removedCube : Cube

/-- Calculates the number of edges in the modified cube structure -/
def edgeCount (mc : ModifiedCube) : ℕ :=
  12 + 8 * 6

/-- Theorem stating that removing cubes of side length 5 from each corner of a cube 
    with side length 10 results in a solid with 60 edges -/
theorem modified_cube_edge_count :
  let largeCube := Cube.mk 10
  let smallCube := Cube.mk 5
  let modifiedCube := ModifiedCube.mk largeCube smallCube
  edgeCount modifiedCube = 60 := by
  sorry

end NUMINAMATH_CALUDE_modified_cube_edge_count_l3401_340155


namespace NUMINAMATH_CALUDE_max_value_of_sum_products_l3401_340186

theorem max_value_of_sum_products (a b c d : ℕ) : 
  ({a, b, c, d} : Finset ℕ) = {1, 2, 4, 5} →
  a * b + b * c + c * d + d * a ≤ 36 :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_of_sum_products_l3401_340186


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l3401_340175

theorem nested_fraction_evaluation : 
  (1 : ℚ) / (2 + 1 / (3 + 1 / 4)) = 13 / 30 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l3401_340175


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l3401_340174

theorem complex_fraction_equality (a b : ℝ) (i : ℂ) (h1 : i * i = -1) (h2 : a + b * i = i * (1 - i)) :
  (a + b * i) / (a - b * i) = i := by sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l3401_340174


namespace NUMINAMATH_CALUDE_point_movement_on_number_line_l3401_340170

/-- Given two points A and B on a number line, where A represents -3 and B is obtained by moving 7 units to the right from A, prove that B represents 4. -/
theorem point_movement_on_number_line (A B : ℝ) : A = -3 ∧ B = A + 7 → B = 4 := by
  sorry

end NUMINAMATH_CALUDE_point_movement_on_number_line_l3401_340170


namespace NUMINAMATH_CALUDE_dartboard_central_angle_l3401_340111

theorem dartboard_central_angle (probability : ℝ) (central_angle : ℝ) : 
  probability = 1 / 8 → central_angle = 45 := by
  sorry

end NUMINAMATH_CALUDE_dartboard_central_angle_l3401_340111


namespace NUMINAMATH_CALUDE_mistaken_divisor_problem_l3401_340195

theorem mistaken_divisor_problem (dividend : ℕ) (correct_divisor : ℕ) (correct_quotient : ℕ) (mistaken_quotient : ℕ) 
  (h1 : dividend = correct_divisor * correct_quotient)
  (h2 : correct_divisor = 21)
  (h3 : correct_quotient = 40)
  (h4 : mistaken_quotient = 70)
  (h5 : ∃ (mistaken_divisor : ℕ), dividend = mistaken_divisor * mistaken_quotient) :
  ∃ (mistaken_divisor : ℕ), mistaken_divisor = 12 ∧ dividend = mistaken_divisor * mistaken_quotient := by
  sorry

end NUMINAMATH_CALUDE_mistaken_divisor_problem_l3401_340195


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l3401_340137

theorem rectangle_perimeter (x y : ℝ) 
  (rachel_sum : 2 * x + y = 44)
  (heather_sum : x + 2 * y = 40) : 
  2 * (x + y) = 56 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l3401_340137


namespace NUMINAMATH_CALUDE_square_side_increase_l3401_340142

theorem square_side_increase (p : ℝ) : 
  (1 + p / 100)^2 = 1.96 → p = 40 := by
  sorry

end NUMINAMATH_CALUDE_square_side_increase_l3401_340142


namespace NUMINAMATH_CALUDE_alice_overall_score_approx_80_percent_l3401_340162

/-- Represents a test with a number of questions and a score percentage -/
structure Test where
  questions : ℕ
  score : ℚ
  scoreInRange : 0 ≤ score ∧ score ≤ 1

/-- Alice's test results -/
def aliceTests : List Test := [
  ⟨20, 3/4, by norm_num⟩,
  ⟨50, 17/20, by norm_num⟩,
  ⟨30, 3/5, by norm_num⟩,
  ⟨40, 9/10, by norm_num⟩
]

/-- The total number of questions Alice answered correctly -/
def totalCorrect : ℚ :=
  aliceTests.foldl (fun acc test => acc + test.questions * test.score) 0

/-- The total number of questions across all tests -/
def totalQuestions : ℕ :=
  aliceTests.foldl (fun acc test => acc + test.questions) 0

/-- Alice's overall score as a percentage -/
def overallScore : ℚ := totalCorrect / totalQuestions

theorem alice_overall_score_approx_80_percent :
  abs (overallScore - 4/5) < 1/100 := by
  sorry

end NUMINAMATH_CALUDE_alice_overall_score_approx_80_percent_l3401_340162


namespace NUMINAMATH_CALUDE_min_cost_is_58984_l3401_340105

/-- Represents a travel agency with its pricing structure -/
structure TravelAgency where
  name : String
  young_age_limit : Nat
  young_price : Nat
  adult_price : Nat
  discount_or_commission : Float
  is_discount : Bool

/-- Represents a family member -/
structure FamilyMember where
  age : Nat

/-- Calculates the total cost for a family's vacation with a given travel agency -/
def calculate_total_cost (agency : TravelAgency) (family : List FamilyMember) : Float :=
  sorry

/-- The Dorokhov family -/
def dorokhov_family : List FamilyMember :=
  [⟨35⟩, ⟨35⟩, ⟨5⟩]  -- Assuming parents are 35 years old

/-- Globus travel agency -/
def globus : TravelAgency :=
  ⟨"Globus", 5, 11200, 25400, 0.02, true⟩

/-- Around the World travel agency -/
def around_the_world : TravelAgency :=
  ⟨"Around the World", 6, 11400, 23500, 0.01, false⟩

/-- Theorem: The minimum cost for the Dorokhov family's vacation is 58984 rubles -/
theorem min_cost_is_58984 :
  min (calculate_total_cost globus dorokhov_family)
      (calculate_total_cost around_the_world dorokhov_family) = 58984 :=
  sorry

end NUMINAMATH_CALUDE_min_cost_is_58984_l3401_340105


namespace NUMINAMATH_CALUDE_factorize_3x_minus_12x_squared_factorize_negative_x_squared_plus_6xy_minus_9y_squared_factorize_n_squared_m_minus_2_plus_2_minus_m_factorize_a_squared_plus_4b_squared_squared_minus_16a_squared_b_squared_l3401_340146

-- Problem 1
theorem factorize_3x_minus_12x_squared (x : ℝ) :
  3*x - 12*x^2 = 3*x*(1-4*x) := by sorry

-- Problem 2
theorem factorize_negative_x_squared_plus_6xy_minus_9y_squared (x y : ℝ) :
  -x^2 + 6*x*y - 9*y^2 = -(x-3*y)^2 := by sorry

-- Problem 3
theorem factorize_n_squared_m_minus_2_plus_2_minus_m (m n : ℝ) :
  n^2*(m-2) + (2-m) = (m-2)*(n+1)*(n-1) := by sorry

-- Problem 4
theorem factorize_a_squared_plus_4b_squared_squared_minus_16a_squared_b_squared (a b : ℝ) :
  (a^2 + 4*b^2)^2 - 16*a^2*b^2 = (a+2*b)^2 * (a-2*b)^2 := by sorry

end NUMINAMATH_CALUDE_factorize_3x_minus_12x_squared_factorize_negative_x_squared_plus_6xy_minus_9y_squared_factorize_n_squared_m_minus_2_plus_2_minus_m_factorize_a_squared_plus_4b_squared_squared_minus_16a_squared_b_squared_l3401_340146


namespace NUMINAMATH_CALUDE_optimal_pricing_strategy_l3401_340187

/-- Represents the pricing strategy of a merchant -/
structure MerchantPricing where
  list_price : ℝ
  purchase_discount : ℝ
  marked_price : ℝ
  sale_discount : ℝ
  profit_margin : ℝ

/-- Calculates the purchase price given the list price and purchase discount -/
def purchase_price (m : MerchantPricing) : ℝ :=
  m.list_price * (1 - m.purchase_discount)

/-- Calculates the selling price given the marked price and sale discount -/
def selling_price (m : MerchantPricing) : ℝ :=
  m.marked_price * (1 - m.sale_discount)

/-- Calculates the profit given the selling price and purchase price -/
def profit (m : MerchantPricing) : ℝ :=
  selling_price m - purchase_price m

/-- Theorem stating the optimal pricing strategy for the merchant -/
theorem optimal_pricing_strategy (m : MerchantPricing) 
  (h1 : m.purchase_discount = 0.25)
  (h2 : m.sale_discount = 0.20)
  (h3 : m.profit_margin = 0.25)
  (h4 : profit m = m.profit_margin * selling_price m) :
  m.marked_price = 1.25 * m.list_price := by
  sorry

end NUMINAMATH_CALUDE_optimal_pricing_strategy_l3401_340187


namespace NUMINAMATH_CALUDE_prob_five_heads_five_tails_l3401_340152

/-- Represents the state of the coin after some number of flips. -/
structure CoinState where
  heads : ℕ
  tails : ℕ

/-- The probability of getting heads given the current state of the coin. -/
def prob_heads (state : CoinState) : ℚ :=
  (state.heads + 1) / (state.heads + state.tails + 2)

/-- The probability of a specific sequence of 10 flips resulting in exactly 5 heads and 5 tails. -/
def prob_sequence : ℚ := 1 / 39916800

/-- The number of ways to arrange 5 heads and 5 tails in 10 flips. -/
def num_sequences : ℕ := 252

/-- The theorem stating the probability of getting exactly 5 heads and 5 tails after 10 flips. -/
theorem prob_five_heads_five_tails :
  num_sequences * prob_sequence = 1 / 158760 := by sorry

end NUMINAMATH_CALUDE_prob_five_heads_five_tails_l3401_340152


namespace NUMINAMATH_CALUDE_divisibility_implies_value_l3401_340109

-- Define the polynomials
def f (x : ℝ) : ℝ := x^4 + 4*x^3 + 6*x^2 + 4*x + 1
def g (x p q r s : ℝ) : ℝ := x^5 + 5*x^4 + 10*p*x^3 + 10*q*x^2 + 5*r*x + s

-- State the theorem
theorem divisibility_implies_value (p q r s : ℝ) :
  (∃ h : ℝ → ℝ, ∀ x, g x p q r s = f x * h x) →
  (p + q) * r = -1.5 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implies_value_l3401_340109


namespace NUMINAMATH_CALUDE_inequality_proof_l3401_340118

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 1/x + 1/y + 1/z = 1) :
  Real.sqrt (x + y*z) + Real.sqrt (y + z*x) + Real.sqrt (z + x*y) ≥ 
  Real.sqrt (x*y*z) + Real.sqrt x + Real.sqrt y + Real.sqrt z :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3401_340118


namespace NUMINAMATH_CALUDE_at_least_one_calculation_incorrect_l3401_340100

def is_divisible_by_9 (n : ℕ) : Prop :=
  n % 9 = 0

theorem at_least_one_calculation_incorrect
  (first_num second_num : ℕ)
  (sum_digits_first sum_digits_second : ℕ)
  (h1 : first_num * sum_digits_second = 201320132013)
  (h2 : second_num * sum_digits_first = 201420142014)
  (h3 : is_divisible_by_9 201320132013)
  (h4 : ¬ is_divisible_by_9 201420142014) :
  ¬(∀ (x y : ℕ), x * y = 201320132013 → is_divisible_by_9 (x * y)) ∨
  ¬(∀ (x y : ℕ), x * y = 201420142014 → ¬ is_divisible_by_9 (x * y)) :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_calculation_incorrect_l3401_340100


namespace NUMINAMATH_CALUDE_percentage_of_truth_speakers_l3401_340107

theorem percentage_of_truth_speakers (L B : ℝ) (h1 : L = 0.2) (h2 : B = 0.1) 
  (h3 : L + B + (L + B - B) = 0.4) : L + B - B = 0.3 :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_truth_speakers_l3401_340107
