import Mathlib

namespace NUMINAMATH_CALUDE_cakes_served_total_l2998_299826

/-- The number of cakes served in a restaurant over two days. -/
def total_cakes (lunch_today dinner_today yesterday : ℕ) : ℕ :=
  lunch_today + dinner_today + yesterday

/-- Theorem stating that the total number of cakes served is 14 -/
theorem cakes_served_total :
  total_cakes 5 6 3 = 14 := by
  sorry

end NUMINAMATH_CALUDE_cakes_served_total_l2998_299826


namespace NUMINAMATH_CALUDE_garden_yield_mr_green_garden_yield_l2998_299892

/-- Calculates the expected potato yield from a rectangular garden after applying fertilizer -/
theorem garden_yield (length_steps width_steps feet_per_step : ℕ) 
  (original_yield_per_sqft : ℚ) (yield_increase_percent : ℕ) : ℚ :=
  let length_feet := length_steps * feet_per_step
  let width_feet := width_steps * feet_per_step
  let area := length_feet * width_feet
  let original_yield := area * original_yield_per_sqft
  let yield_increase_factor := 1 + yield_increase_percent / 100
  original_yield * yield_increase_factor

/-- Proves that Mr. Green's garden will yield 2227.5 pounds of potatoes after fertilizer -/
theorem mr_green_garden_yield :
  garden_yield 18 25 3 (1/2) 10 = 2227.5 := by
  sorry

end NUMINAMATH_CALUDE_garden_yield_mr_green_garden_yield_l2998_299892


namespace NUMINAMATH_CALUDE_equation_solution_l2998_299869

theorem equation_solution : ∃ k : ℝ, (5/9 * (k^2 - 32))^3 = 150 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2998_299869


namespace NUMINAMATH_CALUDE_trig_identity_l2998_299838

theorem trig_identity (α : ℝ) (h : Real.cos (75 * π / 180 + α) = 1/3) :
  Real.sin (60 * π / 180 + 2*α) = 7/9 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2998_299838


namespace NUMINAMATH_CALUDE_maintenance_check_increase_l2998_299870

theorem maintenance_check_increase (original_time : ℝ) (increase_percentage : ℝ) (new_time : ℝ) :
  original_time = 25 →
  increase_percentage = 20 →
  new_time = original_time * (1 + increase_percentage / 100) →
  new_time = 30 := by
  sorry

end NUMINAMATH_CALUDE_maintenance_check_increase_l2998_299870


namespace NUMINAMATH_CALUDE_square_side_length_l2998_299820

theorem square_side_length (d : ℝ) (h : d = 4) : 
  ∃ (s : ℝ), s * s + s * s = d * d ∧ s = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l2998_299820


namespace NUMINAMATH_CALUDE_power_of_product_equality_l2998_299864

theorem power_of_product_equality (a b : ℝ) : (-2 * a * b^2)^3 = -8 * a^3 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_equality_l2998_299864


namespace NUMINAMATH_CALUDE_shopkeeper_profit_calculation_l2998_299893

theorem shopkeeper_profit_calculation 
  (C L S : ℝ)
  (h1 : L = C * (1 + intended_profit_percentage))
  (h2 : S = 0.9 * L)
  (h3 : S = 1.35 * C)
  : intended_profit_percentage = 0.5 :=
by sorry


end NUMINAMATH_CALUDE_shopkeeper_profit_calculation_l2998_299893


namespace NUMINAMATH_CALUDE_circular_garden_radius_l2998_299891

theorem circular_garden_radius (r : ℝ) (h : r > 0) :
  2 * Real.pi * r = (1 / 4) * Real.pi * r^2 → r = 8 := by
  sorry

end NUMINAMATH_CALUDE_circular_garden_radius_l2998_299891


namespace NUMINAMATH_CALUDE_max_min_on_interval_l2998_299813

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 5

-- Define the interval
def interval : Set ℝ := Set.Icc 1 3

-- State the theorem
theorem max_min_on_interval :
  (∃ (x : ℝ), x ∈ interval ∧ ∀ (y : ℝ), y ∈ interval → f y ≤ f x) ∧
  (∃ (x : ℝ), x ∈ interval ∧ ∀ (y : ℝ), y ∈ interval → f x ≤ f y) ∧
  (∀ (x : ℝ), x ∈ interval → 1 ≤ f x ∧ f x ≤ 5) :=
sorry

end NUMINAMATH_CALUDE_max_min_on_interval_l2998_299813


namespace NUMINAMATH_CALUDE_line_equation_proof_l2998_299877

theorem line_equation_proof (m c : ℝ) (h1 : c ≠ 0) (h2 : m = 4 + 2 * Real.sqrt 7) (h3 : c = 2 - 2 * Real.sqrt 7) :
  ∃ k : ℝ, 
    (∀ k' : ℝ, k' ≠ k → 
      (abs ((k'^2 + 4*k' + 3) - (m*k' + c)) ≠ 7 ∨ 
       ¬∃ y1 y2 : ℝ, y1 = k'^2 + 4*k' + 3 ∧ y2 = m*k' + c ∧ y1 ≠ y2)) ∧
    (∃ y1 y2 : ℝ, y1 = k^2 + 4*k + 3 ∧ y2 = m*k + c ∧ y1 ≠ y2 ∧ abs (y1 - y2) = 7) ∧
    m * 1 + c = 6 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_proof_l2998_299877


namespace NUMINAMATH_CALUDE_friday_temperature_l2998_299842

/-- Temperatures for each day of the week -/
structure WeekTemperatures where
  monday : ℝ
  tuesday : ℝ
  wednesday : ℝ
  thursday : ℝ
  friday : ℝ

/-- The theorem stating the temperature on Friday given the conditions -/
theorem friday_temperature (temps : WeekTemperatures)
  (h1 : (temps.monday + temps.tuesday + temps.wednesday + temps.thursday) / 4 = 48)
  (h2 : (temps.tuesday + temps.wednesday + temps.thursday + temps.friday) / 4 = 46)
  (h3 : temps.monday = 41) :
  temps.friday = 33 := by
  sorry

#check friday_temperature

end NUMINAMATH_CALUDE_friday_temperature_l2998_299842


namespace NUMINAMATH_CALUDE_selection_probabilities_correct_l2998_299883

/-- Given a group of 3 boys and 2 girls, this function calculates various probabilities
    when selecting two people from the group. -/
def selection_probabilities (num_boys : ℕ) (num_girls : ℕ) : ℚ × ℚ × ℚ :=
  let total := num_boys + num_girls
  let total_combinations := (total.choose 2 : ℚ)
  let two_boys := (num_boys.choose 2 : ℚ) / total_combinations
  let one_girl := (num_boys * num_girls : ℚ) / total_combinations
  let at_least_one_girl := 1 - two_boys
  (two_boys, one_girl, at_least_one_girl)

theorem selection_probabilities_correct :
  selection_probabilities 3 2 = (3/10, 3/5, 7/10) := by
  sorry

end NUMINAMATH_CALUDE_selection_probabilities_correct_l2998_299883


namespace NUMINAMATH_CALUDE_problem_statement_l2998_299804

theorem problem_statement (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a * b * c ≤ 1/9 ∧ 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (a * b * c)) := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2998_299804


namespace NUMINAMATH_CALUDE_inequality_system_solutions_l2998_299848

theorem inequality_system_solutions (m : ℝ) : 
  (∃ (x₁ x₂ x₃ : ℤ), 
    (∀ (x : ℤ), (x > m ∧ x < 8) ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃)) ∧
    (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃)) ↔
  (4 ≤ m ∧ m < 5) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solutions_l2998_299848


namespace NUMINAMATH_CALUDE_inequality_proof_l2998_299896

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (1 / x^2 - 1) * (1 / y^2 - 1) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2998_299896


namespace NUMINAMATH_CALUDE_at_most_one_negative_l2998_299872

theorem at_most_one_negative (a b c : ℝ) (sum_nonneg : a + b + c ≥ 0) (product_nonpos : a * b * c ≤ 0) :
  ¬(((a < 0 ∧ b < 0) ∨ (a < 0 ∧ c < 0) ∨ (b < 0 ∧ c < 0))) := by
  sorry

end NUMINAMATH_CALUDE_at_most_one_negative_l2998_299872


namespace NUMINAMATH_CALUDE_monomial_sum_implies_expression_l2998_299825

/-- If the sum of two monomials is still a monomial, then a specific expression evaluates to -1 --/
theorem monomial_sum_implies_expression (m n : ℝ) : 
  (∃ (a : ℝ), ∃ (k : ℕ), ∃ (l : ℕ), 3 * (X : ℝ → ℝ → ℝ) k l + (-2) * (X : ℝ → ℝ → ℝ) (2*m+3) 3 = a * (X : ℝ → ℝ → ℝ) k l) →
  (4*m - n)^n = -1 := by
  sorry

/-- Helper function to represent monomials --/
def X (i j : ℕ) : ℝ → ℝ → ℝ := fun x y ↦ x^i * y^j

end NUMINAMATH_CALUDE_monomial_sum_implies_expression_l2998_299825


namespace NUMINAMATH_CALUDE_box_volume_l2998_299873

/-- A rectangular box with specific proportions -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ
  front_half_top : length * width = 0.5 * (length * height)
  top_one_half_side : length * height = 1.5 * (width * height)
  side_area : width * height = 200

/-- The volume of a box is the product of its length, width, and height -/
def volume (b : Box) : ℝ := b.length * b.width * b.height

/-- Theorem stating that a box with the given proportions has a volume of 3000 -/
theorem box_volume (b : Box) : volume b = 3000 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_l2998_299873


namespace NUMINAMATH_CALUDE_largest_number_l2998_299837

theorem largest_number (a b c d e : ℝ) :
  a = (7 * 8)^(1/4)^(1/2) →
  b = (8 * 7^(1/3))^(1/4) →
  c = (7 * 8^(1/4))^(1/2) →
  d = (7 * 8^(1/4))^(1/3) →
  e = (8 * 7^(1/3))^(1/4) →
  d ≥ a ∧ d ≥ b ∧ d ≥ c ∧ d ≥ e := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l2998_299837


namespace NUMINAMATH_CALUDE_sum_congruence_and_parity_l2998_299865

def sum : ℕ := 2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999

theorem sum_congruence_and_parity :
  (sum % 9 = 6) ∧ Even 6 := by sorry

end NUMINAMATH_CALUDE_sum_congruence_and_parity_l2998_299865


namespace NUMINAMATH_CALUDE_rectangle_area_increase_l2998_299878

theorem rectangle_area_increase :
  ∀ (l w : ℝ), l > 0 → w > 0 →
  let new_length := 1.25 * l
  let new_width := 1.15 * w
  let original_area := l * w
  let new_area := new_length * new_width
  (new_area - original_area) / original_area = 0.4375 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_increase_l2998_299878


namespace NUMINAMATH_CALUDE_game_winner_parity_l2998_299886

/-- The game state representing the current rectangle -/
structure GameState where
  width : ℕ
  height : ℕ
  area : ℕ
  h_width : width > 1
  h_height : height > 1
  h_area : area = width * height

/-- The result of the game -/
inductive GameResult
  | FirstPlayerWins
  | SecondPlayerWins

/-- The game rules and win condition -/
def game_rules (initial_state : GameState) : GameResult :=
  if initial_state.area % 2 = 1 then
    GameResult.FirstPlayerWins
  else
    GameResult.SecondPlayerWins

/-- The main theorem stating the winning condition based on initial area parity -/
theorem game_winner_parity (m n : ℕ) (h_m : m > 1) (h_n : n > 1) :
  let initial_state : GameState := {
    width := m,
    height := n,
    area := m * n,
    h_width := h_m,
    h_height := h_n,
    h_area := rfl
  }
  game_rules initial_state =
    if m * n % 2 = 1 then
      GameResult.FirstPlayerWins
    else
      GameResult.SecondPlayerWins :=
sorry

end NUMINAMATH_CALUDE_game_winner_parity_l2998_299886


namespace NUMINAMATH_CALUDE_dog_barks_theorem_l2998_299858

/-- The number of times a single dog barks per minute -/
def single_dog_barks_per_minute : ℕ := 30

/-- The number of dogs -/
def number_of_dogs : ℕ := 2

/-- The duration of barking in minutes -/
def duration : ℕ := 10

/-- The total number of barks from all dogs -/
def total_barks : ℕ := 600

theorem dog_barks_theorem :
  single_dog_barks_per_minute * number_of_dogs * duration = total_barks :=
by sorry

end NUMINAMATH_CALUDE_dog_barks_theorem_l2998_299858


namespace NUMINAMATH_CALUDE_peach_ratio_l2998_299817

/-- Proves the ratio of peaches in knapsack to one cloth bag is 1:2 --/
theorem peach_ratio (total_peaches : ℕ) (knapsack_peaches : ℕ) (num_cloth_bags : ℕ) :
  total_peaches = 5 * 12 →
  knapsack_peaches = 12 →
  num_cloth_bags = 2 →
  (total_peaches - knapsack_peaches) % num_cloth_bags = 0 →
  (knapsack_peaches : ℚ) / ((total_peaches - knapsack_peaches) / num_cloth_bags) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_peach_ratio_l2998_299817


namespace NUMINAMATH_CALUDE_unique_number_ratio_l2998_299802

theorem unique_number_ratio : ∃! x : ℝ, (x + 1) / (x + 5) = (x + 5) / (x + 13) := by
  sorry

end NUMINAMATH_CALUDE_unique_number_ratio_l2998_299802


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l2998_299845

theorem complex_expression_simplification :
  Real.sqrt 2 * (Real.sqrt 6 - Real.sqrt 12) + (Real.sqrt 3 + 1)^2 + 12 / Real.sqrt 6 = 4 + 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l2998_299845


namespace NUMINAMATH_CALUDE_two_numbers_difference_l2998_299830

theorem two_numbers_difference (a b : ℕ) 
  (sum_eq : a + b = 24365)
  (b_div_5 : b % 5 = 0)
  (b_div_10_eq_2a : b / 10 = 2 * a) :
  b - a = 19931 :=
by sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l2998_299830


namespace NUMINAMATH_CALUDE_sqrt_200_simplification_l2998_299839

theorem sqrt_200_simplification : Real.sqrt 200 = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_200_simplification_l2998_299839


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l2998_299832

def U : Set Nat := {1,2,3,4,5,6,7,8}
def M : Set Nat := {1,3,5,7}
def N : Set Nat := {2,5,8}

theorem complement_intersection_theorem : 
  (U \ M) ∩ N = {2,8} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l2998_299832


namespace NUMINAMATH_CALUDE_total_easter_eggs_l2998_299827

def clubHouseEggs : ℕ := 40
def parkEggs : ℕ := 25
def townHallEggs : ℕ := 15

theorem total_easter_eggs : 
  clubHouseEggs + parkEggs + townHallEggs = 80 := by
  sorry

end NUMINAMATH_CALUDE_total_easter_eggs_l2998_299827


namespace NUMINAMATH_CALUDE_product_of_sum_and_sum_of_cubes_l2998_299819

theorem product_of_sum_and_sum_of_cubes (a b : ℝ) 
  (h1 : a + b = 8) 
  (h2 : a^3 + b^3 = 152) : 
  a * b = 15 := by
sorry

end NUMINAMATH_CALUDE_product_of_sum_and_sum_of_cubes_l2998_299819


namespace NUMINAMATH_CALUDE_problem_solution_l2998_299866

def is_product_of_three_primes_less_than_10 (n : ℕ) : Prop :=
  ∃ p q r, p < 10 ∧ q < 10 ∧ r < 10 ∧ Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ n = p * q * r

def all_primes_less_than_10_present (a b : ℕ) : Prop :=
  ∀ p, p < 10 → Nat.Prime p → (p ∣ a ∨ p ∣ b)

theorem problem_solution (a b : ℕ) :
  is_product_of_three_primes_less_than_10 a ∧
  is_product_of_three_primes_less_than_10 b ∧
  all_primes_less_than_10_present a b ∧
  Nat.gcd a b = Nat.gcd (a / 15) b ∧
  Nat.gcd a b = 2 * Nat.gcd a (b / 4) →
  a = 30 ∧ b = 28 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2998_299866


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l2998_299876

theorem negation_of_existence (P : ℝ → Prop) : 
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x₀ : ℝ, x₀^2 - 2*x₀ + 1 < 0) ↔ (∀ x : ℝ, x^2 - 2*x + 1 ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l2998_299876


namespace NUMINAMATH_CALUDE_max_value_function_l2998_299884

theorem max_value_function (a : ℝ) : 
  a > 0 → 
  a ≠ 1 → 
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, a^(2*x) + 2*a^x - 1 ≤ 14) → 
  (∃ x ∈ Set.Icc (-1 : ℝ) 1, a^(2*x) + 2*a^x - 1 = 14) → 
  a = 1/3 ∨ a = 3 := by
sorry

end NUMINAMATH_CALUDE_max_value_function_l2998_299884


namespace NUMINAMATH_CALUDE_abc_sum_theorem_l2998_299856

def is_valid_digit (d : ℕ) : Prop := d > 0 ∧ d < 6

def to_base_6 (n : ℕ) : ℕ := n

theorem abc_sum_theorem (A B C : ℕ) 
  (h_distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (h_valid : is_valid_digit A ∧ is_valid_digit B ∧ is_valid_digit C)
  (h_equation : to_base_6 (A * 36 + B * 6 + C) + to_base_6 (B * 6 + C) = to_base_6 (A * 36 + C * 6 + A)) :
  to_base_6 (A + B + C) = 11 :=
sorry

end NUMINAMATH_CALUDE_abc_sum_theorem_l2998_299856


namespace NUMINAMATH_CALUDE_divided_triangle_area_l2998_299835

/-- Represents a triangle with parallel lines dividing its sides -/
structure DividedTriangle where
  /-- The area of the original triangle -/
  area : ℝ
  /-- The number of equal segments the sides are divided into -/
  num_segments : ℕ
  /-- The area of the largest part after division -/
  largest_part_area : ℝ

/-- Theorem stating the relationship between the area of the largest part
    and the total area of the triangle -/
theorem divided_triangle_area (t : DividedTriangle)
    (h1 : t.num_segments = 10)
    (h2 : t.largest_part_area = 38) :
    t.area = 200 := by
  sorry

end NUMINAMATH_CALUDE_divided_triangle_area_l2998_299835


namespace NUMINAMATH_CALUDE_total_spending_correct_l2998_299846

-- Define the stores and their purchases
structure Store :=
  (items : List (String × Float))
  (discount : Float)
  (accessoryDeal : Option (Float × Float))
  (freeItem : Option Float)
  (shippingFee : Bool)

def stores : List Store := [
  ⟨[("shoes", 200)], 0.3, none, none, false⟩,
  ⟨[("shirts", 160), ("pants", 150)], 0.2, none, none, false⟩,
  ⟨[("jacket", 250), ("tie", 40), ("hat", 60)], 0, some (0.5, 0.5), none, false⟩,
  ⟨[("watch", 120), ("wallet", 49)], 0, none, some 49, true⟩,
  ⟨[("belt", 35), ("scarf", 45)], 0, none, none, true⟩
]

-- Define the overall discount and tax rates
def rewardsDiscount : Float := 0.05
def salesTax : Float := 0.08

-- Define the gift card amount
def giftCardAmount : Float := 50

-- Define the shipping fee
def shippingFee : Float := 5

-- Function to calculate the total spending
noncomputable def calculateTotalSpending (stores : List Store) (rewardsDiscount : Float) (salesTax : Float) (giftCardAmount : Float) (shippingFee : Float) : Float :=
  sorry

-- Theorem to prove
theorem total_spending_correct :
  calculateTotalSpending stores rewardsDiscount salesTax giftCardAmount shippingFee = 854.29 :=
sorry

end NUMINAMATH_CALUDE_total_spending_correct_l2998_299846


namespace NUMINAMATH_CALUDE_min_value_theorem_l2998_299810

theorem min_value_theorem (x y a : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_a : a > 0) :
  (∀ x y, x + 2*y = 1 → (3/x + a/y) ≥ 6*Real.sqrt 3) ∧
  (∃ x y, x + 2*y = 1 ∧ 3/x + a/y = 6*Real.sqrt 3) →
  (∀ x y, 1/x + 2/y = 1 → 3*x + a*y ≥ 6*Real.sqrt 3) ∧
  (∃ x y, 1/x + 2/y = 1 ∧ 3*x + a*y = 6*Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2998_299810


namespace NUMINAMATH_CALUDE_average_is_three_l2998_299863

/-- Given four real numbers A, B, C, and D satisfying certain conditions,
    prove that their average is 3. -/
theorem average_is_three (A B C D : ℝ) 
    (eq1 : 501 * C - 2004 * A = 3006)
    (eq2 : 2502 * B + 6006 * A = 10010)
    (eq3 : D = A + 2) :
    (A + B + C + D) / 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_average_is_three_l2998_299863


namespace NUMINAMATH_CALUDE_no_special_numbers_l2998_299894

/-- A number is prime if it's greater than 1 and has no divisors other than 1 and itself -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

/-- A number is composite if it has more than two factors -/
def isComposite (n : ℕ) : Prop := n > 1 ∧ ∃ d : ℕ, 1 < d ∧ d < n ∧ d ∣ n

/-- A number is a perfect square if it's the square of an integer -/
def isPerfectSquare (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

/-- The set of integers from 1 to 1000 -/
def numberSet : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 1000}

theorem no_special_numbers : ∀ n ∈ numberSet, isPrime n ∨ isComposite n ∨ isPerfectSquare n := by
  sorry

end NUMINAMATH_CALUDE_no_special_numbers_l2998_299894


namespace NUMINAMATH_CALUDE_career_d_degrees_l2998_299880

/-- Represents the ratio of male to female students -/
def maleToFemaleRatio : Rat := 2 / 3

/-- Represents the percentage of males preferring each career -/
def malePreference : Fin 6 → Rat
| 0 => 25 / 100  -- Career A
| 1 => 15 / 100  -- Career B
| 2 => 30 / 100  -- Career C
| 3 => 40 / 100  -- Career D
| 4 => 20 / 100  -- Career E
| 5 => 35 / 100  -- Career F

/-- Represents the percentage of females preferring each career -/
def femalePreference : Fin 6 → Rat
| 0 => 50 / 100  -- Career A
| 1 => 40 / 100  -- Career B
| 2 => 10 / 100  -- Career C
| 3 => 20 / 100  -- Career D
| 4 => 30 / 100  -- Career E
| 5 => 25 / 100  -- Career F

/-- Calculates the degrees in a circle graph for a given career -/
def careerDegrees (careerIndex : Fin 6) : ℚ :=
  let totalStudents := maleToFemaleRatio + 1
  let maleStudents := maleToFemaleRatio
  let femaleStudents := 1
  let studentsPreferringCareer := 
    maleStudents * malePreference careerIndex + femaleStudents * femalePreference careerIndex
  (studentsPreferringCareer / totalStudents) * 360

/-- Theorem stating that Career D should be represented by 100.8 degrees in the circle graph -/
theorem career_d_degrees : careerDegrees 3 = 100.8 := by sorry

end NUMINAMATH_CALUDE_career_d_degrees_l2998_299880


namespace NUMINAMATH_CALUDE_product_multiple_in_consecutive_integers_l2998_299868

theorem product_multiple_in_consecutive_integers (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a < b) :
  ∃ (start : ℤ) (x y : ℤ), 
    x ≠ y ∧ 
    start ≤ x ∧ x < start + b ∧
    start ≤ y ∧ y < start + b ∧
    (x * y) % (a * b) = 0 :=
by sorry

end NUMINAMATH_CALUDE_product_multiple_in_consecutive_integers_l2998_299868


namespace NUMINAMATH_CALUDE_smallest_integer_y_smallest_integer_solution_l2998_299840

theorem smallest_integer_y (y : ℤ) : (7 - 5 * y < 22) ↔ (y > -3) :=
  sorry

theorem smallest_integer_solution : ∃ y : ℤ, (∀ z : ℤ, 7 - 5 * z < 22 → y ≤ z) ∧ (7 - 5 * y < 22) ∧ y = -2 :=
  sorry

end NUMINAMATH_CALUDE_smallest_integer_y_smallest_integer_solution_l2998_299840


namespace NUMINAMATH_CALUDE_taxi_charge_calculation_l2998_299809

/-- Calculates the total charge for a taxi trip -/
def total_charge (initial_fee : ℚ) (charge_per_increment : ℚ) (increment_distance : ℚ) (trip_distance : ℚ) : ℚ :=
  initial_fee + (trip_distance / increment_distance).floor * charge_per_increment

theorem taxi_charge_calculation :
  let initial_fee : ℚ := 9/4  -- $2.25
  let charge_per_increment : ℚ := 7/20  -- $0.35
  let increment_distance : ℚ := 2/5  -- 2/5 mile
  let trip_distance : ℚ := 18/5  -- 3.6 miles
  total_charge initial_fee charge_per_increment increment_distance trip_distance = 27/5  -- $5.40
:= by sorry

end NUMINAMATH_CALUDE_taxi_charge_calculation_l2998_299809


namespace NUMINAMATH_CALUDE_inequality_proof_l2998_299874

theorem inequality_proof (a x : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hx : 0 ≤ x ∧ x ≤ π) :
  (2 * a - 1) * Real.sin x + (1 - a) * Real.sin ((1 - a) * x) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2998_299874


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l2998_299854

/-- A geometric sequence of 5 terms -/
def GeometricSequence (a : Fin 5 → ℝ) : Prop :=
  ∀ i j k, i < j → j < k → a i * a k = a j ^ 2

theorem geometric_sequence_product (a : Fin 5 → ℝ) 
  (h_geom : GeometricSequence a)
  (h_first : a 0 = 1/2)
  (h_last : a 4 = 8) :
  a 1 * a 2 * a 3 = 8 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l2998_299854


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_three_halves_l2998_299841

/-- Given real numbers a, b, and c satisfying the conditions,
    prove that (a/(b+c)) + (b/(c+a)) + (c/(a+b)) = 3/2 -/
theorem sum_of_fractions_equals_three_halves
  (a b c : ℝ)
  (h1 : a^3 + b^3 + c^3 = 3*a*b*c)
  (h2 : Matrix.det !![a, b, c; c, a, b; b, c, a] = 0) :
  a / (b + c) + b / (c + a) + c / (a + b) = 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_three_halves_l2998_299841


namespace NUMINAMATH_CALUDE_calculate_expression_l2998_299833

/-- Proves that 8 * 9(2/5) - 3 = 72(1/5) -/
theorem calculate_expression : 8 * (9 + 2/5) - 3 = 72 + 1/5 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2998_299833


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2998_299853

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_of_M_and_N : M ∩ N = {2, 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2998_299853


namespace NUMINAMATH_CALUDE_expand_and_simplify_l2998_299822

theorem expand_and_simplify (x : ℝ) : 2*x*(x-4) - (2*x-3)*(x+2) = -9*x + 6 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l2998_299822


namespace NUMINAMATH_CALUDE_min_x_value_and_factor_sum_l2998_299807

theorem min_x_value_and_factor_sum (x y : ℕ+) (h : 3 * x^7 = 17 * y^11) :
  ∃ (a b c d : ℕ),
    x = a^c * b^d ∧
    a = 17 ∧ b = 3 ∧ c = 6 ∧ d = 4 ∧
    a + b + c + d = 30 ∧
    (∀ (x' : ℕ+), 3 * x'^7 = 17 * y^11 → x ≤ x') := by
sorry

end NUMINAMATH_CALUDE_min_x_value_and_factor_sum_l2998_299807


namespace NUMINAMATH_CALUDE_triangle_properties_l2998_299857

-- Define the triangle ABC
structure Triangle :=
  (A B C : Real)  -- angles
  (a b c : Real)  -- sides

-- Define the main theorem
theorem triangle_properties (abc : Triangle) 
  (h1 : Real.tan abc.B + Real.tan abc.C = (2 * Real.sin abc.A) / Real.cos abc.C)
  (h2 : abc.a = abc.c + 2)
  (h3 : ∃ θ : Real, θ > π / 2 ∧ (θ = abc.A ∨ θ = abc.B ∨ θ = abc.C)) :
  abc.B = π / 3 ∧ (0 < abc.c ∧ abc.c < 2) :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l2998_299857


namespace NUMINAMATH_CALUDE_lawsuit_probability_difference_l2998_299850

def probability_win_lawsuit1 : ℝ := 0.3
def probability_win_lawsuit2 : ℝ := 0.5

theorem lawsuit_probability_difference :
  (1 - probability_win_lawsuit1) * (1 - probability_win_lawsuit2) - 
  (probability_win_lawsuit1 * probability_win_lawsuit2) = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_lawsuit_probability_difference_l2998_299850


namespace NUMINAMATH_CALUDE_bottles_left_on_shelf_prove_bottles_left_l2998_299890

/-- Calculates the number of bottles left on a shelf after two customers make purchases with specific discounts --/
theorem bottles_left_on_shelf (initial_bottles : ℕ) 
  (jason_bottles : ℕ) (harry_bottles : ℕ) : ℕ :=
  let jason_effective_bottles := jason_bottles
  let harry_effective_bottles := harry_bottles + 1
  initial_bottles - (jason_effective_bottles + harry_effective_bottles)

/-- Proves that given the specific conditions, there are 23 bottles left on the shelf --/
theorem prove_bottles_left : 
  bottles_left_on_shelf 35 5 6 = 23 := by
  sorry

end NUMINAMATH_CALUDE_bottles_left_on_shelf_prove_bottles_left_l2998_299890


namespace NUMINAMATH_CALUDE_power_inequality_specific_power_inequality_l2998_299800

theorem power_inequality (a : ℕ) (h : a ≥ 3) : a^(a+1) > (a+1)^a := by sorry

theorem specific_power_inequality : (2023 : ℕ)^2024 > 2024^2023 := by sorry

end NUMINAMATH_CALUDE_power_inequality_specific_power_inequality_l2998_299800


namespace NUMINAMATH_CALUDE_parabola_directrix_l2998_299805

/-- The equation of a parabola -/
def parabola (x y : ℝ) : Prop := y = 4 * x^2 - 3

/-- The equation of the directrix -/
def directrix (y : ℝ) : Prop := y = -49/16

/-- Theorem stating that the directrix of the given parabola is y = -49/16 -/
theorem parabola_directrix : 
  ∀ x y : ℝ, parabola x y → ∃ d : ℝ, directrix d ∧ 
  (∀ p : ℝ × ℝ, p.1 = x ∧ p.2 = y → 
    (p.2 - d) = (x^2 + (y + 3 - 1/(16:ℝ))^2) / (4 * 1/(16:ℝ))) :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l2998_299805


namespace NUMINAMATH_CALUDE_corresponding_angles_not_always_equal_l2998_299812

-- Define the concept of corresponding angles
def corresponding_angles (l1 l2 t : Line) (a1 a2 : Angle) : Prop :=
  -- We don't provide a specific definition, as it's not given in the problem
  sorry

-- Define the theorem
theorem corresponding_angles_not_always_equal :
  ¬ ∀ (l1 l2 t : Line) (a1 a2 : Angle),
    corresponding_angles l1 l2 t a1 a2 → a1 = a2 :=
by
  sorry

end NUMINAMATH_CALUDE_corresponding_angles_not_always_equal_l2998_299812


namespace NUMINAMATH_CALUDE_common_internal_tangent_length_l2998_299888

theorem common_internal_tangent_length
  (center_distance : ℝ)
  (small_radius : ℝ)
  (large_radius : ℝ)
  (h1 : center_distance = 50)
  (h2 : small_radius = 7)
  (h3 : large_radius = 10) :
  Real.sqrt (center_distance ^ 2 - (small_radius + large_radius) ^ 2) = Real.sqrt 2211 :=
by sorry

end NUMINAMATH_CALUDE_common_internal_tangent_length_l2998_299888


namespace NUMINAMATH_CALUDE_cos_difference_l2998_299885

theorem cos_difference (α : Real) (h : α = 2 * Real.pi / 3) :
  Real.cos (α + Real.pi / 2) - Real.cos (Real.pi + α) = -(Real.sqrt 3 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_difference_l2998_299885


namespace NUMINAMATH_CALUDE_cylinder_height_l2998_299818

theorem cylinder_height (perimeter : Real) (diagonal : Real) (height : Real) : 
  perimeter = 6 → diagonal = 10 → height = 8 → 
  perimeter = 2 * Real.pi * (perimeter / (2 * Real.pi)) ∧ 
  diagonal^2 = perimeter^2 + height^2 :=
by
  sorry

end NUMINAMATH_CALUDE_cylinder_height_l2998_299818


namespace NUMINAMATH_CALUDE_expression_lower_bound_l2998_299834

theorem expression_lower_bound :
  ∃ (L : ℤ), L = 3 ∧
  (∃ (S : Finset ℤ), S.card = 20 ∧
    ∀ n ∈ S, L < 4 * n + 7 ∧ 4 * n + 7 < 80) ∧
  ∀ (n : ℤ), 4 * n + 7 ≥ L :=
by sorry

end NUMINAMATH_CALUDE_expression_lower_bound_l2998_299834


namespace NUMINAMATH_CALUDE_correct_reading_growth_equation_l2998_299897

/-- Represents the growth of average reading amount per student over 2 years -/
def reading_growth (x : ℝ) : Prop :=
  let initial_amount : ℝ := 1
  let final_amount : ℝ := 1.21
  let growth_period : ℕ := 2
  100 * (1 + x)^growth_period = 121

/-- Proves that the equation correctly represents the reading growth -/
theorem correct_reading_growth_equation :
  ∃ x : ℝ, reading_growth x ∧ x > 0 := by sorry

end NUMINAMATH_CALUDE_correct_reading_growth_equation_l2998_299897


namespace NUMINAMATH_CALUDE_no_integer_solutions_l2998_299815

theorem no_integer_solutions : ¬∃ (x y : ℤ), x^3 + 4*x^2 + x = 18*y^3 + 18*y^2 + 6*y + 3 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l2998_299815


namespace NUMINAMATH_CALUDE_quadrilateral_perimeter_l2998_299899

/-- Perimeter of a quadrilateral EFGH with specific properties -/
theorem quadrilateral_perimeter (EF HG FG : ℝ) (h1 : EF = 15) (h2 : HG = 6) (h3 : FG = 20) :
  ∃ (EH : ℝ), EF + FG + HG + EH = 41 + Real.sqrt 481 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_perimeter_l2998_299899


namespace NUMINAMATH_CALUDE_box_height_is_eight_inches_l2998_299801

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  height : ℝ
  width : ℝ
  length : ℝ

/-- Calculates the volume of a rectangular object given its dimensions -/
def volume (d : Dimensions) : ℝ := d.height * d.width * d.length

theorem box_height_is_eight_inches
  (box : Dimensions)
  (block : Dimensions)
  (h1 : box.width = 10)
  (h2 : box.length = 12)
  (h3 : block.height = 3)
  (h4 : block.width = 2)
  (h5 : block.length = 4)
  (h6 : volume box = 40 * volume block) :
  box.height = 8 := by
  sorry

end NUMINAMATH_CALUDE_box_height_is_eight_inches_l2998_299801


namespace NUMINAMATH_CALUDE_monotone_decreasing_implies_m_ge_one_l2998_299821

/-- A function f(x) = x^2 - 2mx + 1 that is monotonically decreasing on (-∞, 1) -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*m*x + 1

/-- The property that f is monotonically decreasing on (-∞, 1) -/
def is_monotone_decreasing (m : ℝ) : Prop :=
  ∀ x y, x < y → y ≤ 1 → f m x > f m y

/-- The theorem stating that if f is monotonically decreasing on (-∞, 1), then m ≥ 1 -/
theorem monotone_decreasing_implies_m_ge_one (m : ℝ) 
  (h : is_monotone_decreasing m) : m ≥ 1 := by
  sorry

#check monotone_decreasing_implies_m_ge_one

end NUMINAMATH_CALUDE_monotone_decreasing_implies_m_ge_one_l2998_299821


namespace NUMINAMATH_CALUDE_sum_is_positive_l2998_299898

theorem sum_is_positive (x y : ℝ) (h1 : x * y < 0) (h2 : x > abs y) : x + y > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_is_positive_l2998_299898


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2998_299811

theorem arithmetic_calculation : (2^3 * 3 * 5) + (18 / 2) = 129 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2998_299811


namespace NUMINAMATH_CALUDE_milk_powder_sampling_l2998_299816

/-- Represents a system sampling method. -/
structure SystemSampling where
  totalItems : ℕ
  sampledItems : ℕ
  firstSampledNumber : ℕ

/-- Calculates the number of the nth sampled item in a system sampling method. -/
def nthSampledNumber (s : SystemSampling) (n : ℕ) : ℕ :=
  s.firstSampledNumber + (n - 1) * (s.totalItems / s.sampledItems)

/-- Theorem stating that for the given system sampling parameters, 
    the 41st sampled item will be numbered 607. -/
theorem milk_powder_sampling :
  let s : SystemSampling := {
    totalItems := 3000,
    sampledItems := 200,
    firstSampledNumber := 7
  }
  nthSampledNumber s 41 = 607 := by
  sorry

end NUMINAMATH_CALUDE_milk_powder_sampling_l2998_299816


namespace NUMINAMATH_CALUDE_hyperbola_vertex_distance_l2998_299855

/-- The equation of the hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  4 * x^2 + 24 * x - 4 * y^2 + 8 * y + 44 = 0

/-- The distance between the vertices of the hyperbola -/
def vertex_distance : ℝ := 2

theorem hyperbola_vertex_distance :
  ∀ x y : ℝ, hyperbola_equation x y → vertex_distance = 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_vertex_distance_l2998_299855


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l2998_299814

theorem smallest_n_congruence : ∃! n : ℕ+, n.val = 20 ∧ 
  (∀ m : ℕ+, m.val < n.val → ¬(5 * m.val ≡ 1826 [ZMOD 26])) ∧
  (5 * n.val ≡ 1826 [ZMOD 26]) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l2998_299814


namespace NUMINAMATH_CALUDE_complex_square_roots_l2998_299867

theorem complex_square_roots (z : ℂ) : z^2 = -77 - 36*I ↔ z = 2 - 9*I ∨ z = -2 + 9*I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_roots_l2998_299867


namespace NUMINAMATH_CALUDE_emily_salary_adjustment_l2998_299859

/-- Calculates Emily's new salary after adjusting employee salaries -/
def emilysNewSalary (initialSalary: ℕ) (numEmployees: ℕ) (initialEmployeeSalary targetEmployeeSalary: ℕ) : ℕ :=
  initialSalary - numEmployees * (targetEmployeeSalary - initialEmployeeSalary)

/-- Proves that Emily's new salary is $850,000 given the initial conditions -/
theorem emily_salary_adjustment :
  emilysNewSalary 1000000 10 20000 35000 = 850000 := by
  sorry

end NUMINAMATH_CALUDE_emily_salary_adjustment_l2998_299859


namespace NUMINAMATH_CALUDE_triangle_circumscribed_circle_radius_l2998_299881

theorem triangle_circumscribed_circle_radius 
  (α : Real) (a b : Real) (R : Real) : 
  α = π / 3 →  -- 60° in radians
  a = 6 → 
  b = 2 → 
  R = (2 * Real.sqrt 21) / 3 → 
  2 * R = (Real.sqrt (a^2 + b^2 - 2*a*b*(Real.cos α))) / (Real.sin α) := by
  sorry

end NUMINAMATH_CALUDE_triangle_circumscribed_circle_radius_l2998_299881


namespace NUMINAMATH_CALUDE_binomial_square_proof_l2998_299895

theorem binomial_square_proof : 
  ∃ (r s : ℚ), (r * X + s) ^ 2 = (196 / 9 : ℚ) * X ^ 2 + 28 * X + 9 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_proof_l2998_299895


namespace NUMINAMATH_CALUDE_probability_of_three_given_sum_fourteen_l2998_299851

-- Define a type for die outcomes
def DieOutcome := Fin 6

-- Define a type for a set of four tosses
def FourTosses := Fin 4 → DieOutcome

-- Function to calculate the sum of four tosses
def sumTosses (tosses : FourTosses) : Nat :=
  (tosses 0).val + (tosses 1).val + (tosses 2).val + (tosses 3).val + 4

-- Function to check if a set of tosses contains at least one 3
def hasThree (tosses : FourTosses) : Prop :=
  ∃ i, (tosses i).val = 2

-- Theorem statement
theorem probability_of_three_given_sum_fourteen (tosses : FourTosses) :
  sumTosses tosses = 14 → hasThree tosses := by
  sorry

#check probability_of_three_given_sum_fourteen

end NUMINAMATH_CALUDE_probability_of_three_given_sum_fourteen_l2998_299851


namespace NUMINAMATH_CALUDE_train_passing_platform_l2998_299849

/-- Calculates the time for a train to pass a platform given its length, time to cross a tree, and platform length -/
theorem train_passing_platform (train_length : ℝ) (time_cross_tree : ℝ) (platform_length : ℝ) :
  train_length = 1200 ∧ time_cross_tree = 120 ∧ platform_length = 1100 →
  (train_length + platform_length) / (train_length / time_cross_tree) = 230 := by
sorry

end NUMINAMATH_CALUDE_train_passing_platform_l2998_299849


namespace NUMINAMATH_CALUDE_arcsin_arctan_equation_solution_l2998_299831

theorem arcsin_arctan_equation_solution :
  ∃ x : ℝ, x ∈ Set.Icc (-1/2 : ℝ) (1/2 : ℝ) ∧ Real.arcsin x + Real.arcsin (2*x) = Real.arctan x :=
by
  sorry

end NUMINAMATH_CALUDE_arcsin_arctan_equation_solution_l2998_299831


namespace NUMINAMATH_CALUDE_square_function_properties_l2998_299875

-- Define the function f(x) = x^2 on (0, +∞)
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem square_function_properties :
  ∀ (x₁ x₂ : ℝ), x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ →
    (f (x₁ * x₂) = f x₁ * f x₂) ∧
    ((f x₁ - f x₂) / (x₁ - x₂) > 0) ∧
    (f ((x₁ + x₂) / 2) < (f x₁ + f x₂) / 2) :=
by sorry

end NUMINAMATH_CALUDE_square_function_properties_l2998_299875


namespace NUMINAMATH_CALUDE_gardening_time_ratio_l2998_299828

/-- Proves that the ratio of time to plant one flower to time to mow one line is 1/4 --/
theorem gardening_time_ratio :
  ∀ (total_time mow_time plant_time : ℕ) 
    (lines flowers_per_row rows : ℕ) 
    (time_per_line : ℕ),
  total_time = 108 →
  lines = 40 →
  time_per_line = 2 →
  flowers_per_row = 7 →
  rows = 8 →
  mow_time = lines * time_per_line →
  plant_time = total_time - mow_time →
  (plant_time : ℚ) / (rows * flowers_per_row : ℚ) / (time_per_line : ℚ) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_gardening_time_ratio_l2998_299828


namespace NUMINAMATH_CALUDE_cindy_earnings_l2998_299823

/-- Calculates the earnings for teaching one math course in a month --/
def earnings_per_course (total_hours_per_week : ℕ) (num_courses : ℕ) (hourly_rate : ℕ) (weeks_per_month : ℕ) : ℕ :=
  (total_hours_per_week / num_courses) * weeks_per_month * hourly_rate

/-- Theorem: Cindy's earnings for one math course in a month --/
theorem cindy_earnings :
  let total_hours_per_week : ℕ := 48
  let num_courses : ℕ := 4
  let hourly_rate : ℕ := 25
  let weeks_per_month : ℕ := 4
  earnings_per_course total_hours_per_week num_courses hourly_rate weeks_per_month = 1200 := by
  sorry

#eval earnings_per_course 48 4 25 4

end NUMINAMATH_CALUDE_cindy_earnings_l2998_299823


namespace NUMINAMATH_CALUDE_ice_chests_filled_example_l2998_299836

/-- Given an ice machine with a total number of ice cubes and a fixed number of ice cubes per chest,
    calculate the number of ice chests that can be filled. -/
def ice_chests_filled (total_ice_cubes : ℕ) (ice_cubes_per_chest : ℕ) : ℕ :=
  total_ice_cubes / ice_cubes_per_chest

/-- Prove that with 294 ice cubes in total and 42 ice cubes per chest, 7 ice chests can be filled. -/
theorem ice_chests_filled_example : ice_chests_filled 294 42 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ice_chests_filled_example_l2998_299836


namespace NUMINAMATH_CALUDE_stratified_sample_size_l2998_299882

theorem stratified_sample_size
  (ratio_A ratio_B ratio_C : ℕ)
  (sample_A : ℕ)
  (h_ratio : ratio_A = 3 ∧ ratio_B = 4 ∧ ratio_C = 7)
  (h_sample_A : sample_A = 15) :
  ∃ n : ℕ, n = sample_A * (ratio_A + ratio_B + ratio_C) / ratio_A ∧ n = 70 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sample_size_l2998_299882


namespace NUMINAMATH_CALUDE_sum_two_smallest_prime_factors_of_120_l2998_299861

theorem sum_two_smallest_prime_factors_of_120 :
  ∃ (p q : Nat), Nat.Prime p ∧ Nat.Prime q ∧ p < q ∧
  p ∣ 120 ∧ q ∣ 120 ∧
  (∀ r : Nat, Nat.Prime r → r ∣ 120 → r = p ∨ r ≥ q) ∧
  p + q = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_two_smallest_prime_factors_of_120_l2998_299861


namespace NUMINAMATH_CALUDE_min_first_prize_l2998_299803

/-- Represents the prize structure and constraints for a competition --/
structure PrizeStructure where
  total_fund : ℕ
  first_prize : ℕ
  second_prize : ℕ
  third_prize : ℕ
  first_winners : ℕ
  second_winners : ℕ
  third_winners : ℕ

/-- Defines the conditions for a valid prize structure --/
def is_valid_structure (p : PrizeStructure) : Prop :=
  p.total_fund = 10800 ∧
  p.first_prize = 3 * p.second_prize ∧
  p.second_prize = 3 * p.third_prize ∧
  p.third_prize * p.third_winners > p.second_prize * p.second_winners ∧
  p.second_prize * p.second_winners > p.first_prize * p.first_winners ∧
  p.first_winners + p.second_winners + p.third_winners ≤ 20 ∧
  p.first_prize * p.first_winners + p.second_prize * p.second_winners + p.third_prize * p.third_winners = p.total_fund

/-- Theorem stating the minimum first prize amount --/
theorem min_first_prize (p : PrizeStructure) (h : is_valid_structure p) : 
  p.first_prize ≥ 2700 := by
  sorry

#check min_first_prize

end NUMINAMATH_CALUDE_min_first_prize_l2998_299803


namespace NUMINAMATH_CALUDE_magic_square_d_plus_e_l2998_299879

/-- Represents a 3x3 magic square with some known values and variables -/
structure MagicSquare where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ
  f : ℤ
  sum : ℤ
  row1_sum : 30 + b + 22 = sum
  row2_sum : 19 + c + d = sum
  row3_sum : a + 28 + f = sum
  col1_sum : 30 + 19 + a = sum
  col2_sum : b + c + 28 = sum
  col3_sum : 22 + d + f = sum
  diag1_sum : 30 + c + f = sum
  diag2_sum : a + c + 22 = sum

/-- The sum of d and e in the magic square is 54 -/
theorem magic_square_d_plus_e (ms : MagicSquare) : ms.d + ms.e = 54 := by
  sorry

end NUMINAMATH_CALUDE_magic_square_d_plus_e_l2998_299879


namespace NUMINAMATH_CALUDE_fraction_sum_non_negative_l2998_299862

theorem fraction_sum_non_negative (a b c : ℝ) (h1 : a > b) (h2 : b > c) : 
  1 / (a - b) + 1 / (b - c) + 4 / (c - a) ≥ 0 := by sorry

end NUMINAMATH_CALUDE_fraction_sum_non_negative_l2998_299862


namespace NUMINAMATH_CALUDE_average_apple_weight_l2998_299843

def apple_weights : List ℝ := [120, 150, 180, 200, 220]

theorem average_apple_weight :
  (apple_weights.sum / apple_weights.length : ℝ) = 174 := by
  sorry

end NUMINAMATH_CALUDE_average_apple_weight_l2998_299843


namespace NUMINAMATH_CALUDE_inequality_proof_l2998_299844

theorem inequality_proof (x y z : ℝ) 
  (hx : x ≠ 1) (hy : y ≠ 1) (hz : z ≠ 1) (hxyz : x * y * z = 1) : 
  (x^2 / (x - 1)^2) + (y^2 / (y - 1)^2) + (z^2 / (z - 1)^2) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2998_299844


namespace NUMINAMATH_CALUDE_all_odd_rolls_probability_l2998_299824

def standard_die_odd_prob : ℚ := 1/2

def roll_count : ℕ := 8

theorem all_odd_rolls_probability :
  (standard_die_odd_prob ^ roll_count : ℚ) = 1/256 := by
  sorry

end NUMINAMATH_CALUDE_all_odd_rolls_probability_l2998_299824


namespace NUMINAMATH_CALUDE_eight_people_seating_theorem_l2998_299871

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def seating_arrangements (n : ℕ) (restricted_pairs : ℕ) : ℕ :=
  factorial n - (2 * restricted_pairs * factorial (n - 1) * 2 - factorial (n - 2) * 4)

theorem eight_people_seating_theorem :
  seating_arrangements 8 2 = 23040 := by sorry

end NUMINAMATH_CALUDE_eight_people_seating_theorem_l2998_299871


namespace NUMINAMATH_CALUDE_inverse_function_property_l2998_299860

def invertible_function (f : ℝ → ℝ) : Prop :=
  ∃ g : ℝ → ℝ, (∀ x, g (f x) = x) ∧ (∀ y, f (g y) = y)

theorem inverse_function_property
  (f : ℝ → ℝ)
  (h_inv : invertible_function f)
  (h_point : 1 - f 1 = 2) :
  ∃ g : ℝ → ℝ, invertible_function g ∧ g = f⁻¹ ∧ g (-1) - (-1) = 2 :=
sorry

end NUMINAMATH_CALUDE_inverse_function_property_l2998_299860


namespace NUMINAMATH_CALUDE_saltwater_volume_l2998_299852

/-- Proves that the initial volume of a saltwater solution is 160 gallons given specific conditions --/
theorem saltwater_volume : ∃ (x : ℝ), 
  (x > 0) ∧ 
  (0.20 * x = x * 0.20) ∧ 
  (0.20 * x + 16 = (1/3) * (3/4 * x + 8 + 16)) ∧ 
  (x = 160) := by
sorry

end NUMINAMATH_CALUDE_saltwater_volume_l2998_299852


namespace NUMINAMATH_CALUDE_sum_of_factorials_last_two_digits_l2998_299808

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def S : ℕ := (List.range 2012).map factorial |>.sum

def last_two_digits (n : ℕ) : ℕ := n % 100

theorem sum_of_factorials_last_two_digits :
  last_two_digits S = 13 := by sorry

end NUMINAMATH_CALUDE_sum_of_factorials_last_two_digits_l2998_299808


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2998_299887

theorem polynomial_simplification (x : ℝ) :
  (2 * x^6 + 3 * x^5 + x^4 + x^3 + 5) - (x^6 + 4 * x^5 + 2 * x^4 - x^3 + 7) =
  x^6 - x^5 - x^4 + 2 * x^3 - 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2998_299887


namespace NUMINAMATH_CALUDE_correct_subtraction_l2998_299889

theorem correct_subtraction (x : ℤ) (h : x - 63 = 8) : x - 36 = 35 := by
  sorry

end NUMINAMATH_CALUDE_correct_subtraction_l2998_299889


namespace NUMINAMATH_CALUDE_vector_sum_zero_parallel_necessary_not_sufficient_l2998_299847

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def parallel (a b : V) : Prop := ∃ (k : ℝ), b = k • a

theorem vector_sum_zero_parallel_necessary_not_sufficient :
  (∀ (a b : V), a ≠ 0 ∧ b ≠ 0 → (a + b = 0 → parallel a b)) ∧
  (∃ (a b : V), a ≠ 0 ∧ b ≠ 0 ∧ parallel a b ∧ a + b ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_vector_sum_zero_parallel_necessary_not_sufficient_l2998_299847


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_squared_l2998_299829

/-- A circle inscribed in a quadrilateral EFGH -/
structure InscribedCircle where
  /-- Radius of the inscribed circle -/
  r : ℝ
  /-- Length of segment ER -/
  er : ℝ
  /-- Length of segment RF -/
  rf : ℝ
  /-- Length of segment GS -/
  gs : ℝ
  /-- Length of segment SH -/
  sh : ℝ
  /-- The circle is tangent to EF at R and to GH at S -/
  tangent_condition : True

/-- The theorem stating that the square of the radius of the inscribed circle is (3225/118)^2 -/
theorem inscribed_circle_radius_squared (c : InscribedCircle)
  (h1 : c.er = 22)
  (h2 : c.rf = 21)
  (h3 : c.gs = 40)
  (h4 : c.sh = 35) :
  c.r^2 = (3225/118)^2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_squared_l2998_299829


namespace NUMINAMATH_CALUDE_prime_extension_l2998_299806

theorem prime_extension (n : ℕ) (h1 : n ≥ 2) :
  (∀ k : ℕ, 0 ≤ k ∧ k ≤ Real.sqrt (n / 3) → Nat.Prime (k^2 + k + n)) →
  (∀ k : ℕ, 0 ≤ k ∧ k ≤ n - 2 → Nat.Prime (k^2 + k + n)) := by
  sorry

end NUMINAMATH_CALUDE_prime_extension_l2998_299806
