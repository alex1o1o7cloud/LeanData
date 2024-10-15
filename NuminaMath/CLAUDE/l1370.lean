import Mathlib

namespace NUMINAMATH_CALUDE_trapezoid_upper_side_length_l1370_137087

/-- Theorem: For a trapezoid with a base of 25 cm, a height of 13 cm, and an area of 286 cm²,
    the length of the upper side is 19 cm. -/
theorem trapezoid_upper_side_length 
  (base : ℝ) (height : ℝ) (area : ℝ) (upper_side : ℝ) 
  (h1 : base = 25) 
  (h2 : height = 13) 
  (h3 : area = 286) 
  (h4 : area = (1/2) * (base + upper_side) * height) : 
  upper_side = 19 := by
  sorry

#check trapezoid_upper_side_length

end NUMINAMATH_CALUDE_trapezoid_upper_side_length_l1370_137087


namespace NUMINAMATH_CALUDE_pet_store_cages_l1370_137081

def cages_used (total_puppies : ℕ) (sold_puppies : ℕ) (puppies_per_cage : ℕ) : ℕ :=
  let remaining_puppies := total_puppies - sold_puppies
  (remaining_puppies / puppies_per_cage) + if remaining_puppies % puppies_per_cage > 0 then 1 else 0

theorem pet_store_cages :
  cages_used 1700 621 26 = 42 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_cages_l1370_137081


namespace NUMINAMATH_CALUDE_stamps_bought_theorem_l1370_137035

/-- The total number of stamps bought by Evariste and Sophie -/
def total_stamps (x y : ℕ) : ℕ := x + y

/-- The cost of Evariste's stamps in pence -/
def evariste_cost : ℕ := 110

/-- The cost of Sophie's stamps in pence -/
def sophie_cost : ℕ := 70

/-- The total amount spent in pence -/
def total_spent : ℕ := 1000

theorem stamps_bought_theorem (x y : ℕ) :
  x * evariste_cost + y * sophie_cost = total_spent →
  total_stamps x y = 12 := by
  sorry

#check stamps_bought_theorem

end NUMINAMATH_CALUDE_stamps_bought_theorem_l1370_137035


namespace NUMINAMATH_CALUDE_complex_equation_sum_of_squares_l1370_137076

theorem complex_equation_sum_of_squares (a b : ℝ) (i : ℂ) : 
  i * i = -1 → 
  (a + i) / i = b + i → 
  a^2 + b^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_of_squares_l1370_137076


namespace NUMINAMATH_CALUDE_kelly_initial_apples_l1370_137019

/-- The number of apples Kelly needs to pick -/
def apples_to_pick : ℕ := 49

/-- The total number of apples Kelly will have after picking -/
def total_apples : ℕ := 105

/-- The initial number of apples Kelly had -/
def initial_apples : ℕ := total_apples - apples_to_pick

theorem kelly_initial_apples :
  initial_apples = 56 :=
sorry

end NUMINAMATH_CALUDE_kelly_initial_apples_l1370_137019


namespace NUMINAMATH_CALUDE_board_coverage_uncoverable_boards_l1370_137029

/-- Represents a rectangular board, possibly with one square removed -/
structure Board where
  rows : Nat
  cols : Nat
  removed : Bool

/-- Calculates the total number of squares on a board -/
def Board.totalSquares (b : Board) : Nat :=
  b.rows * b.cols - if b.removed then 1 else 0

/-- Predicate for whether a board can be covered by dominoes -/
def canBeCovered (b : Board) : Prop :=
  b.totalSquares % 2 = 0

/-- Main theorem: A board can be covered iff its total squares is even -/
theorem board_coverage (b : Board) :
  canBeCovered b ↔ b.totalSquares % 2 = 0 := by sorry

/-- Specific boards from the problem -/
def board_3x4 : Board := { rows := 3, cols := 4, removed := false }
def board_3x5 : Board := { rows := 3, cols := 5, removed := false }
def board_4x4_removed : Board := { rows := 4, cols := 4, removed := true }
def board_5x5 : Board := { rows := 5, cols := 5, removed := false }
def board_6x3 : Board := { rows := 6, cols := 3, removed := false }

/-- Theorem about which boards cannot be covered -/
theorem uncoverable_boards :
  ¬(canBeCovered board_3x5) ∧
  ¬(canBeCovered board_4x4_removed) ∧
  ¬(canBeCovered board_5x5) ∧
  (canBeCovered board_3x4) ∧
  (canBeCovered board_6x3) := by sorry

end NUMINAMATH_CALUDE_board_coverage_uncoverable_boards_l1370_137029


namespace NUMINAMATH_CALUDE_problem_solution_l1370_137075

theorem problem_solution (w x y : ℝ) 
  (h1 : 6 / w + 6 / x = 6 / y)
  (h2 : w * x = y)
  (h3 : (w + x) / 2 = 0.5) :
  x = 0.5 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1370_137075


namespace NUMINAMATH_CALUDE_goose_survival_rate_l1370_137091

theorem goose_survival_rate (total_eggs : ℝ) (hatched_fraction : ℝ) (first_year_survival_fraction : ℝ) (first_year_survivors : ℕ) : 
  total_eggs = 550 →
  hatched_fraction = 2/3 →
  first_year_survival_fraction = 2/5 →
  first_year_survivors = 110 →
  ∃ (first_month_survival_fraction : ℝ),
    first_month_survival_fraction * hatched_fraction * first_year_survival_fraction * total_eggs = first_year_survivors ∧
    first_month_survival_fraction = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_goose_survival_rate_l1370_137091


namespace NUMINAMATH_CALUDE_prob_six_odd_in_eight_rolls_l1370_137022

/-- The probability of rolling an odd number on a fair 6-sided die -/
def prob_odd : ℚ := 1 / 2

/-- The number of rolls -/
def num_rolls : ℕ := 8

/-- The number of desired odd rolls -/
def num_odd : ℕ := 6

/-- The probability of getting exactly 6 odd numbers in 8 rolls of a fair 6-sided die -/
theorem prob_six_odd_in_eight_rolls : 
  (Nat.choose num_rolls num_odd : ℚ) * prob_odd ^ num_odd * (1 - prob_odd) ^ (num_rolls - num_odd) = 7 / 64 := by
  sorry

end NUMINAMATH_CALUDE_prob_six_odd_in_eight_rolls_l1370_137022


namespace NUMINAMATH_CALUDE_max_value_fraction_l1370_137007

theorem max_value_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : (2*a + b)^2 = 1 + 6*a*b) :
  (a * b) / (2*a + b + 1) ≤ 1/6 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ (2*a₀ + b₀)^2 = 1 + 6*a₀*b₀ ∧ (a₀ * b₀) / (2*a₀ + b₀ + 1) = 1/6 :=
sorry

end NUMINAMATH_CALUDE_max_value_fraction_l1370_137007


namespace NUMINAMATH_CALUDE_quadratic_unique_solution_l1370_137070

theorem quadratic_unique_solution (c : ℝ) : 
  (c ≠ 0 ∧ 
   ∃! b : ℝ, b > 0 ∧ 
   ∃! x : ℝ, x^2 + (b^2 + 1/b^2) * x + c = 0) ↔ 
  (c = (1 + Real.sqrt 2) / 2 ∨ c = (1 - Real.sqrt 2) / 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_unique_solution_l1370_137070


namespace NUMINAMATH_CALUDE_max_earnings_ali_baba_l1370_137010

/-- Represents the weight of the bag when filled with only diamonds -/
def diamond_full_weight : ℝ := 40

/-- Represents the weight of the bag when filled with only gold -/
def gold_full_weight : ℝ := 200

/-- Represents the maximum weight Ali Baba can carry -/
def max_carry_weight : ℝ := 100

/-- Represents the cost of 1 kg of diamonds in dinars -/
def diamond_cost : ℝ := 60

/-- Represents the cost of 1 kg of gold in dinars -/
def gold_cost : ℝ := 20

/-- Represents the objective function to maximize -/
def objective_function (x y : ℝ) : ℝ := diamond_cost * x + gold_cost * y

/-- Theorem stating that the maximum value of the objective function
    under given constraints is 3000 dinars -/
theorem max_earnings_ali_baba :
  ∃ x y : ℝ,
    x ≥ 0 ∧
    y ≥ 0 ∧
    x + y ≤ max_carry_weight ∧
    (x / diamond_full_weight + y / gold_full_weight) ≤ 1 ∧
    objective_function x y = 3000 ∧
    ∀ x' y' : ℝ,
      x' ≥ 0 →
      y' ≥ 0 →
      x' + y' ≤ max_carry_weight →
      (x' / diamond_full_weight + y' / gold_full_weight) ≤ 1 →
      objective_function x' y' ≤ 3000 :=
by sorry

end NUMINAMATH_CALUDE_max_earnings_ali_baba_l1370_137010


namespace NUMINAMATH_CALUDE_relationship_abc_l1370_137089

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

theorem relationship_abc : 
  let a := base_to_decimal [1, 2] 16
  let b := base_to_decimal [2, 5] 7
  let c := base_to_decimal [3, 3] 4
  c < a ∧ a < b := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l1370_137089


namespace NUMINAMATH_CALUDE_similar_triangles_height_l1370_137049

theorem similar_triangles_height (h_small : ℝ) (area_ratio : ℝ) :
  h_small = 3 →
  area_ratio = 4 →
  ∃ h_large : ℝ, h_large = 6 ∧ h_large / h_small = Real.sqrt area_ratio :=
by sorry

end NUMINAMATH_CALUDE_similar_triangles_height_l1370_137049


namespace NUMINAMATH_CALUDE_inequality_to_interval_l1370_137000

theorem inequality_to_interval : 
  {x : ℝ | -8 ≤ x ∧ x < 15} = Set.Icc (-8) 15 := by sorry

end NUMINAMATH_CALUDE_inequality_to_interval_l1370_137000


namespace NUMINAMATH_CALUDE_rogers_wife_is_anne_l1370_137090

-- Define the set of people
inductive Person : Type
  | Henry | Peter | Louis | Roger | Elizabeth | Jeanne | Mary | Anne

-- Define the relationship of being married
def married : Person → Person → Prop := sorry

-- Define the action of dancing
def dancing : Person → Prop := sorry

-- Define the action of playing an instrument
def playing : Person → String → Prop := sorry

theorem rogers_wife_is_anne :
  -- Conditions
  (∀ p : Person, ∃! q : Person, married p q) →
  (∃ p : Person, married Person.Henry p ∧ dancing p ∧ 
    ∃ q : Person, married q Person.Elizabeth ∧ dancing q) →
  (¬ dancing Person.Roger) →
  (¬ dancing Person.Anne) →
  (playing Person.Peter "trumpet") →
  (playing Person.Mary "piano") →
  (¬ married Person.Anne Person.Peter) →
  -- Conclusion
  married Person.Roger Person.Anne :=
by sorry

end NUMINAMATH_CALUDE_rogers_wife_is_anne_l1370_137090


namespace NUMINAMATH_CALUDE_min_production_avoid_losses_l1370_137060

/-- The minimum daily production of gloves to avoid losses -/
def min_production : ℕ := 800

/-- The total daily production cost (in yuan) as a function of daily production volume (in pairs) -/
def total_cost (x : ℕ) : ℕ := 5 * x + 4000

/-- The factory price per pair of gloves (in yuan) -/
def price_per_pair : ℕ := 10

/-- The daily revenue (in yuan) as a function of daily production volume (in pairs) -/
def revenue (x : ℕ) : ℕ := price_per_pair * x

/-- Theorem stating that the minimum daily production to avoid losses is 800 pairs -/
theorem min_production_avoid_losses :
  ∀ x : ℕ, x ≥ min_production ↔ revenue x ≥ total_cost x :=
sorry

end NUMINAMATH_CALUDE_min_production_avoid_losses_l1370_137060


namespace NUMINAMATH_CALUDE_floor_equation_solution_l1370_137063

theorem floor_equation_solution (x : ℝ) :
  ⌊⌊3 * x⌋ + (1 : ℝ) / 2⌋ = ⌊x + 3⌋ ↔ 4 / 3 ≤ x ∧ x < 2 :=
sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l1370_137063


namespace NUMINAMATH_CALUDE_polynomial_parity_and_divisibility_l1370_137031

theorem polynomial_parity_and_divisibility (p q : ℤ) :
  (∀ x : ℤ, ∃ k : ℤ, x^2 + p*x + q = 2*k ↔ p % 2 = 1 ∧ q % 2 = 0) ∧
  (∀ x : ℤ, ∃ k : ℤ, x^2 + p*x + q = 2*k + 1 ↔ p % 2 = 1 ∧ q % 2 = 1) ∧
  (∀ x : ℤ, ∃ k : ℤ, x^3 + p*x + q = 3*k ↔ q % 3 = 0 ∧ p % 3 = 2) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_parity_and_divisibility_l1370_137031


namespace NUMINAMATH_CALUDE_roses_per_girl_l1370_137020

/-- Proves that each girl planted 3 roses given the conditions of the problem -/
theorem roses_per_girl (total_students : ℕ) (total_plants : ℕ) (birches : ℕ) 
  (h1 : total_students = 24)
  (h2 : total_plants = 24)
  (h3 : birches = 6)
  (h4 : birches * 3 = total_students - (total_students - birches * 3)) :
  (total_plants - birches) / (total_students - birches * 3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_roses_per_girl_l1370_137020


namespace NUMINAMATH_CALUDE_surf_festival_average_l1370_137068

theorem surf_festival_average (day1 : ℕ) (day2 : ℕ) (day3 : ℕ) :
  day1 = 1500 →
  day2 = day1 + 600 →
  day3 = (2 : ℕ) * day1 / 5 →
  (day1 + day2 + day3) / 3 = 1400 := by
  sorry

end NUMINAMATH_CALUDE_surf_festival_average_l1370_137068


namespace NUMINAMATH_CALUDE_number_of_elements_in_set_l1370_137069

theorem number_of_elements_in_set
  (initial_average : ℚ)
  (misread_number : ℚ)
  (correct_number : ℚ)
  (correct_average : ℚ)
  (h1 : initial_average = 18)
  (h2 : misread_number = 26)
  (h3 : correct_number = 36)
  (h4 : correct_average = 19) :
  ∃ (n : ℕ), (n : ℚ) * initial_average - misread_number = (n : ℚ) * correct_average - correct_number ∧ n = 10 :=
by sorry

end NUMINAMATH_CALUDE_number_of_elements_in_set_l1370_137069


namespace NUMINAMATH_CALUDE_unique_n_for_prime_roots_l1370_137066

/-- Determines if a natural number is prime -/
def isPrime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ m : ℕ, m > 0 → m < p → p % m ≠ 0

/-- The quadratic equation as a function of x and n -/
def quadraticEq (x n : ℕ) : ℤ :=
  2 * x^2 - 8*n*x + 10*x - n^2 + 35*n - 76

theorem unique_n_for_prime_roots :
  ∃! n : ℕ, ∃ x₁ x₂ : ℕ,
    x₁ ≠ x₂ ∧
    isPrime x₁ ∧
    isPrime x₂ ∧
    quadraticEq x₁ n = 0 ∧
    quadraticEq x₂ n = 0 ∧
    n = 3 ∧
    x₁ = 2 ∧
    x₂ = 5 :=
sorry

end NUMINAMATH_CALUDE_unique_n_for_prime_roots_l1370_137066


namespace NUMINAMATH_CALUDE_farmer_james_animals_l1370_137009

/-- Represents the number of heads for each animal type -/
def heads : Fin 3 → ℕ
  | 0 => 2  -- Hens
  | 1 => 3  -- Peacocks
  | 2 => 6  -- Zombie hens

/-- Represents the number of legs for each animal type -/
def legs : Fin 3 → ℕ
  | 0 => 8  -- Hens
  | 1 => 9  -- Peacocks
  | 2 => 12 -- Zombie hens

/-- The total number of heads on the farm -/
def total_heads : ℕ := 800

/-- The total number of legs on the farm -/
def total_legs : ℕ := 2018

/-- Calculates the total number of animals on the farm -/
def total_animals : ℕ := (total_legs - total_heads) / 6

theorem farmer_james_animals :
  total_animals = 203 ∧
  (∃ (h p z : ℕ),
    h * heads 0 + p * heads 1 + z * heads 2 = total_heads ∧
    h * legs 0 + p * legs 1 + z * legs 2 = total_legs ∧
    h + p + z = total_animals) :=
by sorry

#eval total_animals

end NUMINAMATH_CALUDE_farmer_james_animals_l1370_137009


namespace NUMINAMATH_CALUDE_initial_alcohol_percentage_l1370_137048

theorem initial_alcohol_percentage 
  (initial_volume : ℝ) 
  (added_alcohol : ℝ) 
  (added_water : ℝ) 
  (final_percentage : ℝ) :
  initial_volume = 40 →
  added_alcohol = 2.5 →
  added_water = 7.5 →
  final_percentage = 9 →
  ∃ (initial_percentage : ℝ),
    initial_percentage * initial_volume / 100 + added_alcohol = 
    final_percentage * (initial_volume + added_alcohol + added_water) / 100 ∧
    initial_percentage = 5 :=
by sorry

end NUMINAMATH_CALUDE_initial_alcohol_percentage_l1370_137048


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1370_137026

/-- Given a line ax - by + 1 = 0 (where a > 0 and b > 0) passing through the center of the circle
    x^2 + y^2 + 2x - 4y + 1 = 0, the minimum value of 1/a + 1/b is 3 + 2√2. -/
theorem min_value_sum_reciprocals (a b : ℝ) (ha : a > 0) (hb : b > 0) 
    (h_line : a * (-1) - b * 2 + 1 = 0) : 
    (∀ (a' b' : ℝ), a' > 0 → b' > 0 → a' * (-1) - b' * 2 + 1 = 0 → 1 / a + 1 / b ≤ 1 / a' + 1 / b') → 
    1 / a + 1 / b = 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1370_137026


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l1370_137097

/-- 
Given that the cost price is 89% of the selling price, 
prove that the profit percentage is (100/89 - 1) * 100.
-/
theorem profit_percentage_calculation (selling_price : ℝ) (cost_price : ℝ) 
  (h : cost_price = 0.89 * selling_price) :
  (selling_price - cost_price) / cost_price * 100 = (100/89 - 1) * 100 := by
  sorry

#eval (100/89 - 1) * 100 -- This will output approximately 12.36

end NUMINAMATH_CALUDE_profit_percentage_calculation_l1370_137097


namespace NUMINAMATH_CALUDE_triangle_circumradius_l1370_137014

/-- Given a triangle ABC where:
  * a, b, c are sides opposite to angles A, B, C respectively
  * a = 2
  * b = 3
  * cos C = 1/3
  Then the radius of the circumcircle is 9√2/8 -/
theorem triangle_circumradius (A B C : ℝ) (a b c : ℝ) :
  a = 2 →
  b = 3 →
  c = (a^2 + b^2 - 2*a*b*(1/3))^(1/2) →
  let r := c / (2 * (1 - (1/3)^2)^(1/2))
  r = 9 * (2^(1/2)) / 8 := by
sorry

end NUMINAMATH_CALUDE_triangle_circumradius_l1370_137014


namespace NUMINAMATH_CALUDE_base_conversion_theorem_l1370_137073

/-- Conversion from base 7 to base 10 -/
def base7ToBase10 (n : Nat) : Nat :=
  (n / 100) * 7^2 + ((n / 10) % 10) * 7 + (n % 10)

/-- Given 563 in base 7 equals xy in base 10, prove that (x+y)/9 = 11/9 -/
theorem base_conversion_theorem :
  let n := 563
  let xy := base7ToBase10 n
  let x := xy / 10
  let y := xy % 10
  (x + y) / 9 = 11 / 9 := by sorry

end NUMINAMATH_CALUDE_base_conversion_theorem_l1370_137073


namespace NUMINAMATH_CALUDE_sin_two_alpha_value_l1370_137058

theorem sin_two_alpha_value (α : Real) 
  (h1 : 0 < α ∧ α < π) 
  (h2 : (1/2) * Real.cos (2*α) = Real.sin (π/4 + α)) : 
  Real.sin (2*α) = -1 := by
  sorry

end NUMINAMATH_CALUDE_sin_two_alpha_value_l1370_137058


namespace NUMINAMATH_CALUDE_circle_radius_l1370_137011

theorem circle_radius (C : ℝ) (h : C = 72 * Real.pi) : C / (2 * Real.pi) = 36 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l1370_137011


namespace NUMINAMATH_CALUDE_race_theorem_l1370_137032

def race_problem (john_speed : ℝ) (race_distance : ℝ) (winning_margin : ℝ) : Prop :=
  let john_time := race_distance / john_speed * 60
  let next_fastest_time := john_time + winning_margin
  next_fastest_time = 23

theorem race_theorem :
  race_problem 15 5 3 := by
  sorry

end NUMINAMATH_CALUDE_race_theorem_l1370_137032


namespace NUMINAMATH_CALUDE_six_digit_divisible_by_nine_l1370_137082

theorem six_digit_divisible_by_nine :
  ∃! d : ℕ, d < 10 ∧ (135790 + d) % 9 = 0 :=
by sorry

end NUMINAMATH_CALUDE_six_digit_divisible_by_nine_l1370_137082


namespace NUMINAMATH_CALUDE_xyz_value_l1370_137039

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 24)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 10) : 
  x * y * z = 14 / 3 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l1370_137039


namespace NUMINAMATH_CALUDE_w_squared_value_l1370_137054

theorem w_squared_value (w : ℝ) (h : (w + 15)^2 = (4*w + 6)*(2*w + 3)) : w^2 = 207/7 := by
  sorry

end NUMINAMATH_CALUDE_w_squared_value_l1370_137054


namespace NUMINAMATH_CALUDE_average_equation_solution_l1370_137086

theorem average_equation_solution (x : ℝ) : 
  ((2 * x + 12 + 5 * x^2 + 3 * x + 1 + 3 * x + 14) / 3 = 6 * x^2 + x - 21) ↔ 
  (x = (5 + Real.sqrt 4705) / 26 ∨ x = (5 - Real.sqrt 4705) / 26) :=
by sorry

end NUMINAMATH_CALUDE_average_equation_solution_l1370_137086


namespace NUMINAMATH_CALUDE_garrison_reinforcement_size_l1370_137078

/-- Calculates the size of reinforcement given garrison provisions information -/
theorem garrison_reinforcement_size
  (initial_size : ℕ)
  (initial_duration : ℕ)
  (initial_consumption : ℚ)
  (time_before_reinforcement : ℕ)
  (new_consumption : ℚ)
  (additional_duration : ℕ)
  (h1 : initial_size = 2000)
  (h2 : initial_duration = 40)
  (h3 : initial_consumption = 3/2)
  (h4 : time_before_reinforcement = 20)
  (h5 : new_consumption = 2)
  (h6 : additional_duration = 10) :
  ∃ (reinforcement_size : ℕ),
    reinforcement_size = 1500 ∧
    (initial_size * initial_consumption * initial_duration : ℚ) =
    (initial_size * initial_consumption * time_before_reinforcement +
     (initial_size * initial_consumption + reinforcement_size * new_consumption) * additional_duration : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_garrison_reinforcement_size_l1370_137078


namespace NUMINAMATH_CALUDE_nadia_walked_18_km_l1370_137005

/-- The distance Hannah walked in kilometers -/
def hannah_distance : ℝ := sorry

/-- The distance Nadia walked in kilometers -/
def nadia_distance : ℝ := 2 * hannah_distance

/-- The total distance walked by both girls in kilometers -/
def total_distance : ℝ := 27

theorem nadia_walked_18_km :
  nadia_distance = 18 ∧ hannah_distance + nadia_distance = total_distance :=
by sorry

end NUMINAMATH_CALUDE_nadia_walked_18_km_l1370_137005


namespace NUMINAMATH_CALUDE_factoring_transformation_l1370_137030

theorem factoring_transformation (y : ℝ) : 4 * y^2 - 4 * y + 1 = (2 * y - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factoring_transformation_l1370_137030


namespace NUMINAMATH_CALUDE_product_remainder_mod_17_l1370_137074

theorem product_remainder_mod_17 : (2003 * 2004 * 2005 * 2006 * 2007) % 17 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_mod_17_l1370_137074


namespace NUMINAMATH_CALUDE_factor_implies_d_value_l1370_137047

/-- The polynomial Q(x) -/
def Q (d : ℝ) (x : ℝ) : ℝ := x^3 - 3*x^2 + d*x - 8

/-- Theorem: If x + 2 is a factor of Q(x), then d = -14 -/
theorem factor_implies_d_value (d : ℝ) :
  (∀ x, Q d x = 0 ↔ x = -2 ∨ (x + 2) * (x^2 - 5*x + 4 - d/2) = 0) →
  d = -14 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_d_value_l1370_137047


namespace NUMINAMATH_CALUDE_obtuse_triangle_side_range_l1370_137099

theorem obtuse_triangle_side_range (a : ℝ) : 
  (∃ (θ : ℝ), 
    -- Triangle inequality
    a > 1 ∧ 
    -- Obtuse triangle condition
    π/2 < θ ∧ 
    -- Largest angle doesn't exceed 120°
    θ ≤ 2*π/3 ∧ 
    -- Cosine law for the largest angle
    Real.cos θ = (a^2 + (a+1)^2 - (a+2)^2) / (2*a*(a+1))) 
  ↔ 
  (3/2 ≤ a ∧ a < 3) :=
by sorry

end NUMINAMATH_CALUDE_obtuse_triangle_side_range_l1370_137099


namespace NUMINAMATH_CALUDE_complex_real_condition_l1370_137083

theorem complex_real_condition (a : ℝ) : 
  (((1 : ℂ) + Complex.I) ^ 2 - a / Complex.I).im = 0 → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_real_condition_l1370_137083


namespace NUMINAMATH_CALUDE_root_inequality_l1370_137055

noncomputable section

open Real

-- Define the functions f and g
def f (x : ℝ) := exp x + x - 2
def g (x : ℝ) := log x + x - 2

-- State the theorem
theorem root_inequality (a b : ℝ) (ha : f a = 0) (hb : g b = 0) :
  f a < f 1 ∧ f 1 < f b := by
  sorry

end

end NUMINAMATH_CALUDE_root_inequality_l1370_137055


namespace NUMINAMATH_CALUDE_opposite_corner_not_always_farthest_l1370_137002

/-- A rectangular parallelepiped (box) -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ
  length_pos : 0 < length
  width_pos : 0 < width
  height_pos : 0 < height

/-- A point on the surface of a box -/
structure SurfacePoint (b : Box) where
  x : ℝ
  y : ℝ
  z : ℝ
  on_surface : (x = 0 ∨ x = b.length) ∨ (y = 0 ∨ y = b.width) ∨ (z = 0 ∨ z = b.height)

/-- The distance between two points on the surface of a box -/
noncomputable def surface_distance (b : Box) (p1 p2 : SurfacePoint b) : ℝ :=
  sorry

/-- The corner opposite to (0, 0, 0) -/
def opposite_corner (b : Box) : SurfacePoint b :=
  { x := b.length, y := b.width, z := b.height,
    on_surface := by simp }

/-- Theorem: The opposite corner is not necessarily the point with the greatest distance from a corner -/
theorem opposite_corner_not_always_farthest (b : Box) :
  ∃ (p : SurfacePoint b), surface_distance b ⟨0, 0, 0, by simp⟩ p > 
                           surface_distance b ⟨0, 0, 0, by simp⟩ (opposite_corner b) :=
sorry

end NUMINAMATH_CALUDE_opposite_corner_not_always_farthest_l1370_137002


namespace NUMINAMATH_CALUDE_floor_sum_eval_l1370_137080

theorem floor_sum_eval : ⌊(-7/4 : ℚ)⌋ + ⌊(-3/2 : ℚ)⌋ - ⌊(-5/3 : ℚ)⌋ = -2 := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_eval_l1370_137080


namespace NUMINAMATH_CALUDE_continuous_at_two_l1370_137084

/-- The function f(x) = -4x^2 - 8 is continuous at x₀ = 2 -/
theorem continuous_at_two :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 2| < δ → |(-4*x^2 - 8) - (-4*2^2 - 8)| < ε :=
by sorry

end NUMINAMATH_CALUDE_continuous_at_two_l1370_137084


namespace NUMINAMATH_CALUDE_cube_of_difference_l1370_137062

theorem cube_of_difference (a b : ℝ) 
  (h1 : a - b = 8) 
  (h2 : a^2 + b^2 = 98) : 
  (a - b)^3 = 512 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_difference_l1370_137062


namespace NUMINAMATH_CALUDE_least_common_multiple_2_3_4_5_6_sixty_divisible_by_2_3_4_5_6_least_number_of_marbles_l1370_137025

theorem least_common_multiple_2_3_4_5_6 : ∀ n : ℕ, n > 0 → (2 ∣ n) ∧ (3 ∣ n) ∧ (4 ∣ n) ∧ (5 ∣ n) ∧ (6 ∣ n) → n ≥ 60 := by
  sorry

theorem sixty_divisible_by_2_3_4_5_6 : (2 ∣ 60) ∧ (3 ∣ 60) ∧ (4 ∣ 60) ∧ (5 ∣ 60) ∧ (6 ∣ 60) := by
  sorry

theorem least_number_of_marbles : ∃! n : ℕ, n > 0 ∧ (2 ∣ n) ∧ (3 ∣ n) ∧ (4 ∣ n) ∧ (5 ∣ n) ∧ (6 ∣ n) ∧ ∀ m : ℕ, m > 0 → (2 ∣ m) ∧ (3 ∣ m) ∧ (4 ∣ m) ∧ (5 ∣ m) ∧ (6 ∣ m) → n ≤ m := by
  sorry

end NUMINAMATH_CALUDE_least_common_multiple_2_3_4_5_6_sixty_divisible_by_2_3_4_5_6_least_number_of_marbles_l1370_137025


namespace NUMINAMATH_CALUDE_gold_pucks_count_gold_pucks_theorem_l1370_137008

theorem gold_pucks_count : ℕ → Prop :=
  fun total_gold : ℕ =>
    ∃ (pucks_per_box : ℕ),
      -- Each box has the same number of pucks
      3 * pucks_per_box = 40 + total_gold ∧
      -- One box contains all black pucks and 1/7 of gold pucks
      pucks_per_box = 40 + total_gold / 7 ∧
      -- The number of gold pucks is 140
      total_gold = 140

-- The proof of the theorem
theorem gold_pucks_theorem : gold_pucks_count 140 := by
  sorry

end NUMINAMATH_CALUDE_gold_pucks_count_gold_pucks_theorem_l1370_137008


namespace NUMINAMATH_CALUDE_max_non_intersecting_points_l1370_137024

/-- A type representing a point in a plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A function that checks if a broken line formed by a list of points intersects itself -/
def is_self_intersecting (points : List Point) : Prop :=
  sorry

/-- The property that any permutation of points forms a non-self-intersecting broken line -/
def all_permutations_non_intersecting (points : List Point) : Prop :=
  ∀ perm : List Point, perm.Perm points → ¬(is_self_intersecting perm)

theorem max_non_intersecting_points :
  ∃ (points : List Point),
    points.length = 4 ∧
    all_permutations_non_intersecting points ∧
    ∀ (larger_set : List Point),
      larger_set.length > 4 →
      ¬(all_permutations_non_intersecting larger_set) :=
sorry

end NUMINAMATH_CALUDE_max_non_intersecting_points_l1370_137024


namespace NUMINAMATH_CALUDE_profit_percentage_is_20_percent_l1370_137013

def selling_price : ℝ := 250
def cost_price : ℝ := 208.33

theorem profit_percentage_is_20_percent :
  (selling_price - cost_price) / cost_price * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_is_20_percent_l1370_137013


namespace NUMINAMATH_CALUDE_rectangle_area_l1370_137015

theorem rectangle_area (square_area : Real) (rectangle_width : Real) (rectangle_length : Real) :
  square_area = 36 →
  rectangle_width = Real.sqrt square_area →
  rectangle_length = 3 * rectangle_width →
  rectangle_width * rectangle_length = 108 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1370_137015


namespace NUMINAMATH_CALUDE_remaining_garden_area_is_48_l1370_137093

/-- The area of a rectangle with given length and width -/
def rectangleArea (length width : ℝ) : ℝ := length * width

/-- The dimensions of the large garden -/
def largeGardenLength : ℝ := 10
def largeGardenWidth : ℝ := 6

/-- The dimensions of the small plot -/
def smallPlotLength : ℝ := 4
def smallPlotWidth : ℝ := 3

/-- The remaining garden area after removing the small plot -/
def remainingGardenArea : ℝ :=
  rectangleArea largeGardenLength largeGardenWidth -
  rectangleArea smallPlotLength smallPlotWidth

theorem remaining_garden_area_is_48 :
  remainingGardenArea = 48 := by sorry

end NUMINAMATH_CALUDE_remaining_garden_area_is_48_l1370_137093


namespace NUMINAMATH_CALUDE_sum_of_digits_3n_l1370_137079

/-- Sum of decimal digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Given a natural number n where the sum of its digits is 100 and
    the sum of digits of 44n is 800, prove that the sum of digits of 3n is 300 -/
theorem sum_of_digits_3n (n : ℕ) 
  (h1 : sumOfDigits n = 100) 
  (h2 : sumOfDigits (44 * n) = 800) : 
  sumOfDigits (3 * n) = 300 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_3n_l1370_137079


namespace NUMINAMATH_CALUDE_course_choice_theorem_l1370_137016

/-- The number of ways to choose courses for 5 students -/
def course_choice_ways : ℕ := 20

/-- The number of students -/
def num_students : ℕ := 5

/-- The number of courses -/
def num_courses : ℕ := 2

/-- The minimum number of students required for each course -/
def min_students_per_course : ℕ := 2

theorem course_choice_theorem :
  ∀ (ways : ℕ),
  ways = course_choice_ways →
  ways = (num_students.choose min_students_per_course) * num_courses.factorial :=
by sorry

end NUMINAMATH_CALUDE_course_choice_theorem_l1370_137016


namespace NUMINAMATH_CALUDE_find_a_l1370_137067

def A : Set ℝ := {x | x^2 + 6*x < 0}

def B (a : ℝ) : Set ℝ := {x | x^2 - (a-2)*x - 2*a < 0}

theorem find_a : 
  A ∪ B a = {x : ℝ | -6 < x ∧ x < 5} → a = 5 := by
sorry

end NUMINAMATH_CALUDE_find_a_l1370_137067


namespace NUMINAMATH_CALUDE_sum_after_removal_l1370_137052

def original_series : List ℚ := [1/2, 1/4, 1/6, 1/8, 1/10, 1/12]

def removed_terms : List ℚ := [1/8, 1/10]

theorem sum_after_removal :
  (original_series.filter (λ x => x ∉ removed_terms)).sum = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_after_removal_l1370_137052


namespace NUMINAMATH_CALUDE_x_fourth_minus_inverse_x_fourth_l1370_137092

theorem x_fourth_minus_inverse_x_fourth (x : ℝ) (h : x - 1/x = 5) : x^4 - 1/x^4 = 727 := by
  sorry

end NUMINAMATH_CALUDE_x_fourth_minus_inverse_x_fourth_l1370_137092


namespace NUMINAMATH_CALUDE_min_max_abs_quadratic_minus_linear_l1370_137065

/-- The minimum value of max_{0 ≤ x ≤ 2} |x^2 - 2xy| over all real y is 4√2 -/
theorem min_max_abs_quadratic_minus_linear :
  (∀ y : ℝ, ∃ x : ℝ, 0 ≤ x ∧ x ≤ 2 ∧ |x^2 - 2*x*y| ≥ 4 * Real.sqrt 2) ∧
  (∃ y : ℝ, ∀ x : ℝ, 0 ≤ x → x ≤ 2 → |x^2 - 2*x*y| ≤ 4 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_min_max_abs_quadratic_minus_linear_l1370_137065


namespace NUMINAMATH_CALUDE_prob_different_colors_specific_l1370_137036

/-- The probability of drawing two chips of different colors from a bag --/
def prob_different_colors (blue red yellow : ℕ) : ℚ :=
  let total := blue + red + yellow
  let prob_blue := blue / total
  let prob_red := red / total
  let prob_yellow := yellow / total
  let prob_not_blue := (red + yellow) / (total - 1)
  let prob_not_red := (blue + yellow) / (total - 1)
  let prob_not_yellow := (blue + red) / (total - 1)
  prob_blue * prob_not_blue + prob_red * prob_not_red + prob_yellow * prob_not_yellow

/-- Theorem: The probability of drawing two chips of different colors from a bag with 7 blue, 6 red, and 5 yellow chips is 122/153 --/
theorem prob_different_colors_specific : prob_different_colors 7 6 5 = 122 / 153 := by
  sorry

end NUMINAMATH_CALUDE_prob_different_colors_specific_l1370_137036


namespace NUMINAMATH_CALUDE_no_integer_n_for_real_nth_power_of_complex_l1370_137021

theorem no_integer_n_for_real_nth_power_of_complex : 
  ¬ ∃ (n : ℤ), (Complex.I + n : ℂ)^5 ∈ Set.range Complex.ofReal := by sorry

end NUMINAMATH_CALUDE_no_integer_n_for_real_nth_power_of_complex_l1370_137021


namespace NUMINAMATH_CALUDE_plant_initial_length_proof_l1370_137038

/-- The daily growth rate of the plant in feet -/
def daily_growth : ℝ := 0.6875

/-- The initial length of the plant in feet -/
def initial_length : ℝ := 11

/-- The length of the plant after n days -/
def plant_length (n : ℕ) : ℝ := initial_length + n * daily_growth

theorem plant_initial_length_proof :
  (plant_length 10 = 1.3 * plant_length 4) →
  initial_length = 11 := by
  sorry

end NUMINAMATH_CALUDE_plant_initial_length_proof_l1370_137038


namespace NUMINAMATH_CALUDE_total_payment_is_195_l1370_137023

def monthly_rate : ℝ := 50

def discount_rate (month : ℕ) : ℝ :=
  match month with
  | 1 => 0.05
  | 2 => 0.07
  | 3 => 0.10
  | 4 => 0.12
  | _ => 0

def late_fee_rate (month : ℕ) : ℝ :=
  match month with
  | 1 => 0.03
  | 2 => 0.02
  | 3 => 0.04
  | 4 => 0.03
  | _ => 0

def payment_amount (month : ℕ) (on_time : Bool) : ℝ :=
  if on_time then
    monthly_rate * (1 - discount_rate month)
  else
    monthly_rate * (1 + late_fee_rate month)

def total_payment : ℝ :=
  payment_amount 1 true +
  payment_amount 2 false +
  payment_amount 3 true +
  payment_amount 4 false

theorem total_payment_is_195 : total_payment = 195 := by
  sorry

end NUMINAMATH_CALUDE_total_payment_is_195_l1370_137023


namespace NUMINAMATH_CALUDE_total_legs_calculation_l1370_137044

theorem total_legs_calculation (total_tables : ℕ) (four_legged_tables : ℕ) 
  (h1 : total_tables = 36)
  (h2 : four_legged_tables = 16)
  (h3 : four_legged_tables ≤ total_tables) :
  four_legged_tables * 4 + (total_tables - four_legged_tables) * 3 = 124 := by
  sorry

#check total_legs_calculation

end NUMINAMATH_CALUDE_total_legs_calculation_l1370_137044


namespace NUMINAMATH_CALUDE_geometric_progression_first_term_is_one_l1370_137004

/-- A geometric progression is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricProgression (a : ℝ → ℝ) (r : ℝ) : Prop :=
  ∀ n : ℕ, a n = a 0 * r ^ n

/-- The product of any two terms in the progression is also a term in the progression. -/
def ProductIsInProgression (a : ℝ → ℝ) : Prop :=
  ∀ i j k : ℕ, ∃ k : ℕ, a i * a j = a k

/-- In a geometric progression where the product of any two terms is also a term in the progression,
    the first term of the progression must be 1. -/
theorem geometric_progression_first_term_is_one
  (a : ℝ → ℝ) (r : ℝ)
  (h1 : IsGeometricProgression a r)
  (h2 : ProductIsInProgression a) :
  a 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_first_term_is_one_l1370_137004


namespace NUMINAMATH_CALUDE_bluegrass_percentage_in_x_l1370_137040

/-- Represents a seed mixture with percentages of different grass types -/
structure SeedMixture where
  ryegrass : ℝ
  bluegrass : ℝ
  fescue : ℝ

/-- The final mixture of X and Y -/
def finalMixture (x y : SeedMixture) (xProportion : ℝ) : SeedMixture :=
  { ryegrass := x.ryegrass * xProportion + y.ryegrass * (1 - xProportion)
  , bluegrass := x.bluegrass * xProportion + y.bluegrass * (1 - xProportion)
  , fescue := x.fescue * xProportion + y.fescue * (1 - xProportion) }

theorem bluegrass_percentage_in_x 
  (x : SeedMixture) 
  (y : SeedMixture)
  (h1 : x.ryegrass = 0.4)
  (h2 : x.ryegrass + x.bluegrass = 1)
  (h3 : y.ryegrass = 0.25)
  (h4 : y.fescue = 0.75)
  (h5 : (finalMixture x y (1/3)).ryegrass = 0.3) :
  x.bluegrass = 0.6 := by
sorry

end NUMINAMATH_CALUDE_bluegrass_percentage_in_x_l1370_137040


namespace NUMINAMATH_CALUDE_valid_regression_equation_l1370_137028

-- Define the linear regression equation
def linear_regression (a b : ℝ) (x : ℝ) : ℝ := a * x + b

-- Define the theorem
theorem valid_regression_equation :
  -- Conditions
  ∀ (x_mean y_mean : ℝ),
  x_mean = 3 →
  y_mean = 3.5 →
  -- The regression equation
  ∃ (a b : ℝ),
  -- Positive correlation
  a > 0 ∧
  -- Equation passes through (x_mean, y_mean)
  linear_regression a b x_mean = y_mean ∧
  -- Specific coefficients
  a = 0.4 ∧
  b = 2.3 :=
by
  sorry


end NUMINAMATH_CALUDE_valid_regression_equation_l1370_137028


namespace NUMINAMATH_CALUDE_linear_function_not_in_first_quadrant_l1370_137033

/-- A linear function that decreases as x increases and satisfies kb > 0 does not pass through the first quadrant -/
theorem linear_function_not_in_first_quadrant
  (k b : ℝ) -- k and b are real numbers
  (h1 : k * b > 0) -- condition: kb > 0
  (h2 : k < 0) -- condition: y decreases as x increases
  : ∀ x y : ℝ, y = k * x + b → ¬(x > 0 ∧ y > 0) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_not_in_first_quadrant_l1370_137033


namespace NUMINAMATH_CALUDE_parallel_vectors_ratio_l1370_137098

theorem parallel_vectors_ratio (θ : ℝ) : 
  let a : ℝ × ℝ := (Real.cos θ, Real.sin θ)
  let b : ℝ × ℝ := (1, 3)
  (∃ (k : ℝ), a.1 = k * b.1 ∧ a.2 = k * b.2) →
  (Real.sin θ + Real.cos θ) / (Real.sin θ - Real.cos θ) = 2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_ratio_l1370_137098


namespace NUMINAMATH_CALUDE_equation_solution_l1370_137072

theorem equation_solution : ∃ x : ℝ, (5*x + 9*x = 350 - 10*(x - 4)) ∧ x = 16.25 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1370_137072


namespace NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l1370_137034

theorem polar_to_rectangular_conversion :
  let r : ℝ := 3
  let θ : ℝ := π / 4
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  x = 3 * Real.sqrt 2 / 2 ∧ y = 3 * Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l1370_137034


namespace NUMINAMATH_CALUDE_method1_is_optimal_l1370_137064

/-- Represents the three methods of division available to the economist. -/
inductive DivisionMethod
  | method1
  | method2
  | method3

/-- Represents a division of coins. -/
structure Division where
  total : ℕ
  part1 : ℕ
  part2 : ℕ
  part3 : ℕ
  part4 : ℕ

/-- The coin division problem. -/
def CoinDivisionProblem (n : ℕ) (method : DivisionMethod) : Prop :=
  -- The total number of coins is odd and greater than 4
  n % 2 = 1 ∧ n > 4 ∧
  -- There exists a valid division of coins
  ∃ (div : Division),
    -- The total number of coins is correct
    div.total = n ∧
    -- Each part has at least one coin
    div.part1 ≥ 1 ∧ div.part2 ≥ 1 ∧ div.part3 ≥ 1 ∧ div.part4 ≥ 1 ∧
    -- The sum of all parts equals the total
    div.part1 + div.part2 + div.part3 + div.part4 = n ∧
    -- The lawyer's initial division results in two parts with at least 2 coins each
    div.part1 + div.part2 ≥ 2 ∧ div.part3 + div.part4 ≥ 2 ∧
    -- Method 1 is the optimal strategy
    (method = DivisionMethod.method1 →
      div.part1 + div.part4 > div.part2 + div.part3 ∧
      div.part1 + div.part4 > div.part1 + div.part2 - 1)

/-- Theorem stating that Method 1 is the optimal strategy for the economist. -/
theorem method1_is_optimal (n : ℕ) :
  CoinDivisionProblem n DivisionMethod.method1 := by
  sorry


end NUMINAMATH_CALUDE_method1_is_optimal_l1370_137064


namespace NUMINAMATH_CALUDE_min_rowers_theorem_l1370_137046

/-- Represents a lyamzik with a weight --/
structure Lyamzik where
  weight : Nat

/-- Represents the boat used for crossing --/
structure Boat where
  maxWeight : Nat

/-- Represents the river crossing scenario --/
structure RiverCrossing where
  lyamziks : List Lyamzik
  boat : Boat
  maxRowsPerLyamzik : Nat

/-- The minimum number of lyamziks required to row --/
def minRowersRequired (rc : RiverCrossing) : Nat :=
  12

theorem min_rowers_theorem (rc : RiverCrossing) 
  (h1 : rc.lyamziks.length = 28)
  (h2 : (rc.lyamziks.filter (fun l => l.weight = 2)).length = 7)
  (h3 : (rc.lyamziks.filter (fun l => l.weight = 3)).length = 7)
  (h4 : (rc.lyamziks.filter (fun l => l.weight = 4)).length = 7)
  (h5 : (rc.lyamziks.filter (fun l => l.weight = 5)).length = 7)
  (h6 : rc.boat.maxWeight = 10)
  (h7 : rc.maxRowsPerLyamzik = 2) :
  minRowersRequired rc ≥ 12 := by
  sorry

#check min_rowers_theorem

end NUMINAMATH_CALUDE_min_rowers_theorem_l1370_137046


namespace NUMINAMATH_CALUDE_max_distance_with_specific_tires_l1370_137003

/-- Represents the maximum distance a car can travel with tire switching -/
def maxDistanceWithTireSwitching (frontTireLife : ℕ) (rearTireLife : ℕ) : ℕ :=
  min frontTireLife rearTireLife

/-- Theorem stating the maximum distance a car can travel with specific tire lifespans -/
theorem max_distance_with_specific_tires :
  maxDistanceWithTireSwitching 42000 56000 = 42000 := by
  sorry

#check max_distance_with_specific_tires

end NUMINAMATH_CALUDE_max_distance_with_specific_tires_l1370_137003


namespace NUMINAMATH_CALUDE_best_of_three_win_probability_l1370_137045

/-- The probability of winning a single game -/
def p : ℚ := 3 / 5

/-- The probability of winning the overall competition in a best-of-three format -/
def win_probability : ℚ :=
  p^2 + 2 * p^2 * (1 - p)

theorem best_of_three_win_probability :
  win_probability = 81 / 125 := by
  sorry

end NUMINAMATH_CALUDE_best_of_three_win_probability_l1370_137045


namespace NUMINAMATH_CALUDE_mork_tax_rate_l1370_137057

/-- Represents the tax rates and incomes of Mork and Mindy -/
structure TaxData where
  mork_income : ℝ
  mork_tax_rate : ℝ
  mindy_tax_rate : ℝ
  combined_tax_rate : ℝ

/-- The conditions of the problem -/
def tax_conditions (data : TaxData) : Prop :=
  data.mork_income > 0 ∧
  data.mindy_tax_rate = 0.25 ∧
  data.combined_tax_rate = 0.29

/-- The theorem stating Mork's tax rate given the conditions -/
theorem mork_tax_rate (data : TaxData) :
  tax_conditions data →
  data.mork_tax_rate = 0.45 := by
  sorry


end NUMINAMATH_CALUDE_mork_tax_rate_l1370_137057


namespace NUMINAMATH_CALUDE_inequality_proof_l1370_137037

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_order : a ≥ b ∧ b ≥ c)
  (h_sum : a + b + c ≤ 1) :
  a^2 + 3*b^2 + 5*c^2 ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1370_137037


namespace NUMINAMATH_CALUDE_share_distribution_l1370_137096

theorem share_distribution (total : ℝ) (y_share : ℝ) (x_to_y_ratio : ℝ) :
  total = 273 →
  y_share = 63 →
  x_to_y_ratio = 0.45 →
  ∃ (x_share z_share : ℝ),
    y_share = x_to_y_ratio * x_share ∧
    total = x_share + y_share + z_share ∧
    z_share / x_share = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_share_distribution_l1370_137096


namespace NUMINAMATH_CALUDE_data_analysis_l1370_137043

def dataset : List ℕ := [10, 8, 6, 9, 8, 7, 8]

def mode (l : List ℕ) : ℕ := sorry

def median (l : List ℕ) : ℚ := sorry

def mean (l : List ℕ) : ℚ := sorry

def variance (l : List ℕ) : ℚ := sorry

theorem data_analysis (l : List ℕ) (h : l = dataset) : 
  (mode l = 8) ∧ 
  (median l = 8) ∧ 
  (mean l = 8) ∧ 
  (variance l ≠ 8) := by sorry

end NUMINAMATH_CALUDE_data_analysis_l1370_137043


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1370_137042

theorem sufficient_but_not_necessary (p q : Prop) 
  (h : (¬p → ¬q) ∧ ¬(¬q → ¬p)) : 
  (p → q) ∧ ¬(q → p) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1370_137042


namespace NUMINAMATH_CALUDE_skittles_distribution_l1370_137085

/-- Given 25 Skittles distributed among 5 people, prove that each person receives 5 Skittles. -/
theorem skittles_distribution (total_skittles : ℕ) (num_people : ℕ) (skittles_per_person : ℕ) :
  total_skittles = 25 →
  num_people = 5 →
  skittles_per_person = total_skittles / num_people →
  skittles_per_person = 5 :=
by sorry

end NUMINAMATH_CALUDE_skittles_distribution_l1370_137085


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l1370_137018

/-- Given a line passing through points (1, -3) and (-1, 3), 
    prove that the sum of its slope and y-intercept is -3 -/
theorem line_slope_intercept_sum (m b : ℝ) : 
  (∀ x y : ℝ, y = m * x + b → 
    ((x = 1 ∧ y = -3) ∨ (x = -1 ∧ y = 3))) → 
  m + b = -3 :=
by sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l1370_137018


namespace NUMINAMATH_CALUDE_problem_statement_l1370_137027

theorem problem_statement (x y : ℝ) 
  (h1 : 1/x + 1/y = 5) 
  (h2 : x*y + x + y = 7) : 
  x^2*y + x*y^2 = 245/36 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1370_137027


namespace NUMINAMATH_CALUDE_monomial_count_l1370_137006

-- Define what a monomial is
def is_monomial (expr : String) : Bool := sorry

-- Define the set of expressions
def expressions : List String := ["(x+a)/2", "-2", "2x^2y", "b", "7x^2+8x-1"]

-- State the theorem
theorem monomial_count : 
  (expressions.filter is_monomial).length = 3 := by sorry

end NUMINAMATH_CALUDE_monomial_count_l1370_137006


namespace NUMINAMATH_CALUDE_max_value_of_complex_expression_l1370_137094

theorem max_value_of_complex_expression (w : ℂ) (h : Complex.abs w = 2) :
  Complex.abs ((w - 2)^2 * (w + 2)) ≤ 24 ∧
  ∃ w₀ : ℂ, Complex.abs w₀ = 2 ∧ Complex.abs ((w₀ - 2)^2 * (w₀ + 2)) = 24 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_complex_expression_l1370_137094


namespace NUMINAMATH_CALUDE_complex_cube_root_l1370_137077

theorem complex_cube_root (a b : ℕ+) (h : (↑a + Complex.I * ↑b) ^ 3 = 2 + 11 * Complex.I) :
  ↑a + Complex.I * ↑b = 2 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_root_l1370_137077


namespace NUMINAMATH_CALUDE_product_real_implies_ratio_l1370_137053

def complex (a b : ℝ) : ℂ := a + b * Complex.I

theorem product_real_implies_ratio (a b : ℝ) (hb : b ≠ 0) :
  let z₁ : ℂ := 2 + 3 * Complex.I
  let z₂ : ℂ := complex a b
  (z₁ * z₂).im = 0 → a / b = -2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_product_real_implies_ratio_l1370_137053


namespace NUMINAMATH_CALUDE_dilution_calculation_l1370_137012

/-- Calculates the amount of water needed to dilute a shaving lotion to a desired alcohol concentration -/
theorem dilution_calculation (initial_volume : ℝ) (initial_concentration : ℝ) (final_concentration : ℝ) :
  initial_volume = 12 →
  initial_concentration = 0.6 →
  final_concentration = 0.45 →
  ∃ (water_volume : ℝ),
    water_volume = 4 ∧
    (initial_volume * initial_concentration) / (initial_volume + water_volume) = final_concentration :=
by sorry

end NUMINAMATH_CALUDE_dilution_calculation_l1370_137012


namespace NUMINAMATH_CALUDE_banker_cannot_guarantee_2kg_l1370_137059

/-- Represents the state of the banker's sand and exchange rates -/
structure SandState where
  g : ℕ -- Exchange rate for gold
  p : ℕ -- Exchange rate for platinum
  G : ℚ -- Amount of gold sand in kg
  P : ℚ -- Amount of platinum sand in kg

/-- Calculates the invariant metric S for a given SandState -/
def calcMetric (state : SandState) : ℚ :=
  state.G * state.p + state.P * state.g

/-- Represents the daily change in exchange rates -/
inductive DailyChange
  | decreaseG
  | decreaseP

/-- Applies a daily change to the SandState -/
def applyDailyChange (state : SandState) (change : DailyChange) : SandState :=
  match change with
  | DailyChange.decreaseG => { state with g := state.g - 1 }
  | DailyChange.decreaseP => { state with p := state.p - 1 }

/-- Theorem stating that the banker cannot guarantee 2 kg of each sand type after 2000 days -/
theorem banker_cannot_guarantee_2kg (initialState : SandState)
  (h_initial_g : initialState.g = 1001)
  (h_initial_p : initialState.p = 1001)
  (h_initial_G : initialState.G = 1)
  (h_initial_P : initialState.P = 1) :
  ¬ ∃ (finalState : SandState),
    (∃ (changes : List DailyChange),
      changes.length = 2000 ∧
      finalState = changes.foldl applyDailyChange initialState ∧
      finalState.g = 1 ∧ finalState.p = 1) ∧
    finalState.G ≥ 2 ∧ finalState.P ≥ 2 :=
  sorry

#check banker_cannot_guarantee_2kg

end NUMINAMATH_CALUDE_banker_cannot_guarantee_2kg_l1370_137059


namespace NUMINAMATH_CALUDE_correlation_coefficient_is_one_l1370_137041

/-- A structure representing a set of sample data points -/
structure SampleData where
  n : ℕ
  x : Fin n → ℝ
  y : Fin n → ℝ
  n_ge_2 : n ≥ 2
  x_not_all_equal : ∃ i j, i ≠ j ∧ x i ≠ x j
  points_on_line : ∀ i, y i = 3 * x i + 1

/-- The correlation coefficient of a set of sample data -/
def correlationCoefficient (data : SampleData) : ℝ :=
  sorry

/-- Theorem stating that the correlation coefficient is 1 for the given conditions -/
theorem correlation_coefficient_is_one (data : SampleData) : 
  correlationCoefficient data = 1 :=
sorry

end NUMINAMATH_CALUDE_correlation_coefficient_is_one_l1370_137041


namespace NUMINAMATH_CALUDE_candidates_per_state_l1370_137050

theorem candidates_per_state (candidates : ℕ) : 
  (candidates * 6 / 100 : ℚ) + 80 = candidates * 7 / 100 → candidates = 8000 := by
  sorry

end NUMINAMATH_CALUDE_candidates_per_state_l1370_137050


namespace NUMINAMATH_CALUDE_gcd_5280_2155_l1370_137056

theorem gcd_5280_2155 : Nat.gcd 5280 2155 = 5 := by
  sorry

end NUMINAMATH_CALUDE_gcd_5280_2155_l1370_137056


namespace NUMINAMATH_CALUDE_unique_number_l1370_137017

def is_valid_number (n : ℕ) : Prop :=
  -- The number is six digits long
  100000 ≤ n ∧ n < 1000000 ∧
  -- The first digit is 2
  (n / 100000 = 2) ∧
  -- Moving the first digit to the last position results in a number that is three times the original number
  (n % 100000 * 10 + 2 = 3 * n)

theorem unique_number : ∃! n : ℕ, is_valid_number n ∧ n = 285714 :=
sorry

end NUMINAMATH_CALUDE_unique_number_l1370_137017


namespace NUMINAMATH_CALUDE_max_bouquets_is_37_l1370_137095

/-- Represents the number of flowers available for each type -/
structure FlowerInventory where
  narcissus : ℕ
  chrysanthemum : ℕ
  tulip : ℕ
  lily : ℕ
  rose : ℕ

/-- Represents the constraints for creating a bouquet -/
structure BouquetConstraints where
  min_narcissus : ℕ
  min_chrysanthemum : ℕ
  min_tulip : ℕ
  max_lily_or_rose : ℕ
  max_total : ℕ

/-- Calculates the maximum number of bouquets that can be made -/
def maxBouquets (inventory : FlowerInventory) (constraints : BouquetConstraints) : ℕ :=
  sorry

/-- Theorem stating that the maximum number of bouquets is 37 -/
theorem max_bouquets_is_37 :
  let inventory := FlowerInventory.mk 75 90 50 45 60
  let constraints := BouquetConstraints.mk 2 1 1 3 10
  maxBouquets inventory constraints = 37 := by sorry

end NUMINAMATH_CALUDE_max_bouquets_is_37_l1370_137095


namespace NUMINAMATH_CALUDE_reach_one_l1370_137071

/-- Represents the two possible operations in the game -/
inductive Operation
  | EraseUnitsDigit
  | MultiplyByTwo

/-- Defines a step in the game as applying an operation to a number -/
def applyOperation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.EraseUnitsDigit => n / 10
  | Operation.MultiplyByTwo => n * 2

/-- Represents a sequence of operations -/
def OperationSequence := List Operation

/-- Applies a sequence of operations to a number -/
def applySequence (n : ℕ) (seq : OperationSequence) : ℕ :=
  seq.foldl applyOperation n

/-- The main theorem stating that for any positive natural number,
    there exists a sequence of operations that transforms it to 1 -/
theorem reach_one (n : ℕ) (h : n > 0) :
  ∃ (seq : OperationSequence), applySequence n seq = 1 := by
  sorry

end NUMINAMATH_CALUDE_reach_one_l1370_137071


namespace NUMINAMATH_CALUDE_sum_of_squares_and_products_l1370_137001

theorem sum_of_squares_and_products (x y z : ℝ) 
  (nonneg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) 
  (sum_of_squares : x^2 + y^2 + z^2 = 52) 
  (sum_of_products : x*y + y*z + z*x = 24) : 
  x + y + z = 10 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_and_products_l1370_137001


namespace NUMINAMATH_CALUDE_hyperbola_equation_correct_l1370_137061

/-- Represents a hyperbola with equation ax² - by² = 1 -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  eq : (x y : ℝ) → a * x^2 - b * y^2 = 1

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

def has_focus (h : Hyperbola) (p : Point) : Prop :=
  ∃ c : ℝ, c^2 = 1 / h.a + 1 / h.b ∧ p.y^2 = c^2

def has_same_asymptotes (h1 h2 : Hyperbola) : Prop :=
  h1.a / h1.b = h2.a / h2.b

theorem hyperbola_equation_correct (h1 h2 : Hyperbola) (p : Point) :
  h1.a = 1/24 ∧ h1.b = 1/12 ∧
  h2.a = 1/2 ∧ h2.b = 1 ∧
  p.x = 0 ∧ p.y = 6 ∧
  has_focus h1 p ∧
  has_same_asymptotes h1 h2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_correct_l1370_137061


namespace NUMINAMATH_CALUDE_perimeter_of_parallelogram_PSTU_l1370_137051

-- Define the triangle PQR
structure Triangle (P Q R : ℝ × ℝ) : Prop where
  pq_eq_pr : dist P Q = dist P R
  pq_eq_15 : dist P Q = 15
  qr_eq_14 : dist Q R = 14

-- Define points S, T, U on the sides of the triangle
def S (P Q : ℝ × ℝ) : ℝ × ℝ := sorry
def T (Q R : ℝ × ℝ) : ℝ × ℝ := sorry
def U (P R : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the parallelism conditions
def parallel (A B C D : ℝ × ℝ) : Prop := sorry

-- Main theorem
theorem perimeter_of_parallelogram_PSTU (P Q R : ℝ × ℝ) 
  (h : Triangle P Q R) 
  (h_st_parallel : parallel (S P Q) (T Q R) P R)
  (h_tu_parallel : parallel (T Q R) (U P R) P Q) : 
  dist P (S P Q) + dist (S P Q) (T Q R) + dist (T Q R) (U P R) + dist (U P R) P = 30 := by
  sorry


end NUMINAMATH_CALUDE_perimeter_of_parallelogram_PSTU_l1370_137051


namespace NUMINAMATH_CALUDE_function_satisfies_conditions_l1370_137088

-- Define the function f
def f (x : ℤ) : ℤ := x^3 - 3*x^2 + 5*x + 9

-- State the theorem
theorem function_satisfies_conditions : 
  f 3 = 12 ∧ f 4 = 22 ∧ f 5 = 36 ∧ f 6 = 54 ∧ f 7 = 76 := by
  sorry

end NUMINAMATH_CALUDE_function_satisfies_conditions_l1370_137088
