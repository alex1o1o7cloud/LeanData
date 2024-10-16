import Mathlib

namespace NUMINAMATH_CALUDE_count_valid_distributions_l778_77898

/-- Represents an envelope containing two cards -/
def Envelope := Fin 6 × Fin 6

/-- Represents a valid distribution of cards into envelopes -/
def ValidDistribution := { d : Fin 3 → Envelope // 
  (∀ i j : Fin 3, i ≠ j → d i ≠ d j) ∧ 
  (∃ i : Fin 3, d i = ⟨1, 2⟩ ∨ d i = ⟨2, 1⟩) }

/-- The number of valid distributions -/
def numValidDistributions : ℕ := sorry

theorem count_valid_distributions : numValidDistributions = 18 := by sorry

end NUMINAMATH_CALUDE_count_valid_distributions_l778_77898


namespace NUMINAMATH_CALUDE_range_of_m_l778_77804

def A (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 3 * m - 1}
def B : Set ℝ := {x | 1 ≤ x ∧ x ≤ 10}

theorem range_of_m (m : ℝ) :
  (A m ∪ B = B) ↔ m ≤ 11/3 := by sorry

end NUMINAMATH_CALUDE_range_of_m_l778_77804


namespace NUMINAMATH_CALUDE_relationship_exists_l778_77825

/-- Represents the contingency table --/
structure ContingencyTable where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  n : ℕ

/-- Calculates K^2 value --/
def calculate_k_squared (ct : ContingencyTable) : ℚ :=
  (ct.n * (ct.a * ct.d - ct.b * ct.c)^2 : ℚ) / 
  ((ct.a + ct.b) * (ct.c + ct.d) * (ct.a + ct.c) * (ct.b + ct.d) : ℚ)

/-- Theorem stating the conditions and the result --/
theorem relationship_exists (a : ℕ) : 
  5 < a ∧ 
  a < 10 ∧
  let ct := ContingencyTable.mk a (15 - a) (20 - a) (30 + a) 65
  calculate_k_squared ct ≥ (6635 : ℚ) / 1000 →
  a = 9 := by
  sorry

#check relationship_exists

end NUMINAMATH_CALUDE_relationship_exists_l778_77825


namespace NUMINAMATH_CALUDE_profit_360_implies_price_increase_4_price_13_implies_profit_350_l778_77837

/-- Represents the daily profit function for a company selling goods -/
def profit_function (x : ℕ) : ℤ := 10 * (x + 2) * (10 - x)

/-- Theorem stating that when the daily profit is 360 yuan, the selling price has increased by 4 yuan -/
theorem profit_360_implies_price_increase_4 :
  ∃ (x : ℕ), 0 ≤ x ∧ x ≤ 10 ∧ profit_function x = 360 → x = 4 := by
  sorry

/-- Theorem stating that when the selling price increases by 3 yuan (to 13 yuan), the profit is 350 yuan -/
theorem price_13_implies_profit_350 :
  profit_function 3 = 350 := by
  sorry

end NUMINAMATH_CALUDE_profit_360_implies_price_increase_4_price_13_implies_profit_350_l778_77837


namespace NUMINAMATH_CALUDE_necklaces_from_three_spools_l778_77835

/-- The number of necklaces that can be made from spools of wire -/
def necklaces_from_wire (num_spools : ℕ) (spool_length : ℕ) (wire_per_necklace : ℕ) : ℕ :=
  (num_spools * spool_length) / wire_per_necklace

/-- Theorem: Given 3 spools of 20-foot wire and 4 feet per necklace, 15 necklaces can be made -/
theorem necklaces_from_three_spools :
  necklaces_from_wire 3 20 4 = 15 := by
  sorry

end NUMINAMATH_CALUDE_necklaces_from_three_spools_l778_77835


namespace NUMINAMATH_CALUDE_specific_group_size_l778_77849

/-- Represents a group of people with language skills -/
structure LanguageGroup where
  latin : ℕ     -- Number of people who can speak Latin
  french : ℕ    -- Number of people who can speak French
  neither : ℕ   -- Number of people who can't speak either Latin or French
  both : ℕ      -- Number of people who can speak both Latin and French

/-- Calculates the total number of people in the group -/
def totalPeople (group : LanguageGroup) : ℕ :=
  (group.latin + group.french - group.both) + group.neither

/-- Theorem: The specific group has 25 people -/
theorem specific_group_size :
  let group : LanguageGroup := {
    latin := 13,
    french := 15,
    neither := 6,
    both := 9
  }
  totalPeople group = 25 := by sorry

end NUMINAMATH_CALUDE_specific_group_size_l778_77849


namespace NUMINAMATH_CALUDE_min_quadratic_value_l778_77800

def is_distinct (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c

def quadratic_function (a b c : ℕ) (x : ℝ) : ℝ := a * (x - b) * (x - c)

theorem min_quadratic_value :
  ∀ a b c : ℕ,
    a > 0 ∧ a < 10 ∧
    b > 0 ∧ b < 10 ∧
    c > 0 ∧ c < 10 ∧
    is_distinct a b c →
    ∃ (min_value : ℝ),
      (∀ x : ℝ, quadratic_function a b c x ≥ min_value) ∧
      (∀ a' b' c' : ℕ,
        a' > 0 ∧ a' < 10 ∧
        b' > 0 ∧ b' < 10 ∧
        c' > 0 ∧ c' < 10 ∧
        is_distinct a' b' c' →
        ∃ (min_value' : ℝ),
          (∀ x : ℝ, quadratic_function a' b' c' x ≥ min_value') ∧
          min_value ≥ -128) :=
sorry

end NUMINAMATH_CALUDE_min_quadratic_value_l778_77800


namespace NUMINAMATH_CALUDE_fish_weight_l778_77815

/-- Represents the weight of a fish with its components -/
structure Fish where
  head : ℝ
  body : ℝ
  tail : ℝ

/-- The fish satisfies the given conditions -/
def validFish (f : Fish) : Prop :=
  f.head = f.tail + f.body / 2 ∧
  f.body = f.head + f.tail ∧
  f.tail = 1

/-- The total weight of the fish -/
def totalWeight (f : Fish) : ℝ :=
  f.head + f.body + f.tail

/-- Theorem stating that a valid fish weighs 8 kg -/
theorem fish_weight (f : Fish) (h : validFish f) : totalWeight f = 8 := by
  sorry

#check fish_weight

end NUMINAMATH_CALUDE_fish_weight_l778_77815


namespace NUMINAMATH_CALUDE_actual_distance_calculation_l778_77827

-- Define the map scale
def map_scale : ℚ := 200

-- Define the measured distance on the map
def map_distance : ℚ := 9/2

-- Theorem to prove
theorem actual_distance_calculation :
  map_scale * map_distance / 100 = 9 := by
  sorry

end NUMINAMATH_CALUDE_actual_distance_calculation_l778_77827


namespace NUMINAMATH_CALUDE_square_sum_equals_36_l778_77887

theorem square_sum_equals_36 (x y z w : ℝ) 
  (eq1 : x^2 / (2^2 - 1^2) + y^2 / (2^2 - 3^2) + z^2 / (2^2 - 5^2) + w^2 / (2^2 - 7^2) = 1)
  (eq2 : x^2 / (4^2 - 1^2) + y^2 / (4^2 - 3^2) + z^2 / (4^2 - 5^2) + w^2 / (4^2 - 7^2) = 1)
  (eq3 : x^2 / (6^2 - 1^2) + y^2 / (6^2 - 3^2) + z^2 / (6^2 - 5^2) + w^2 / (6^2 - 7^2) = 1)
  (eq4 : x^2 / (8^2 - 1^2) + y^2 / (8^2 - 3^2) + z^2 / (8^2 - 5^2) + w^2 / (8^2 - 7^2) = 1) :
  x^2 + y^2 + z^2 + w^2 = 36 := by
sorry


end NUMINAMATH_CALUDE_square_sum_equals_36_l778_77887


namespace NUMINAMATH_CALUDE_base8_653_equals_base10_427_l778_77892

/-- Converts a number from base 8 to base 10 -/
def base8ToBase10 (n : Nat) : Nat :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * 8^2 + tens * 8^1 + ones * 8^0

theorem base8_653_equals_base10_427 :
  base8ToBase10 653 = 427 := by
  sorry

end NUMINAMATH_CALUDE_base8_653_equals_base10_427_l778_77892


namespace NUMINAMATH_CALUDE_absolute_value_relation_l778_77821

theorem absolute_value_relation :
  let p : ℝ → Prop := λ x ↦ |x| = 2
  let q : ℝ → Prop := λ x ↦ x = 2
  (∀ x, q x → p x) ∧ ¬(∀ x, p x → q x) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_relation_l778_77821


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l778_77855

theorem diophantine_equation_solution : ∃ (u v : ℤ), 364 * u + 154 * v = 14 ∧ u = 3 ∧ v = -7 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l778_77855


namespace NUMINAMATH_CALUDE_f_increasing_iff_three_solutions_iff_l778_77813

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x * |x - a| + 2 * x

-- Theorem 1: f(x) is increasing on ℝ iff -2 ≤ a ≤ 2
theorem f_increasing_iff (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ -2 ≤ a ∧ a ≤ 2 :=
sorry

-- Theorem 2: f(x) = bf(a) has three distinct real solutions iff b ∈ (1, 9/8)
theorem three_solutions_iff (a : ℝ) (h : -2 ≤ a ∧ a ≤ 4) :
  (∃ b : ℝ, 1 < b ∧ b < 9/8 ∧ ∃ x y z : ℝ, x < y ∧ y < z ∧
    f a x = b * f a a ∧ f a y = b * f a a ∧ f a z = b * f a a) ↔
  (2 < a ∧ a ≤ 4) :=
sorry

end

end NUMINAMATH_CALUDE_f_increasing_iff_three_solutions_iff_l778_77813


namespace NUMINAMATH_CALUDE_john_paid_with_one_nickel_l778_77896

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The number of quarters John used -/
def quarters_used : ℕ := 4

/-- The number of dimes John used -/
def dimes_used : ℕ := 3

/-- The cost of the candy bar in cents -/
def candy_bar_cost : ℕ := 131

/-- The amount of change John received in cents -/
def change_received : ℕ := 4

/-- The number of nickels John used to pay for the candy bar -/
def nickels_used : ℕ := 1

theorem john_paid_with_one_nickel :
  quarters_used * quarter_value +
  dimes_used * dime_value +
  nickels_used * nickel_value =
  candy_bar_cost + change_received :=
sorry

end NUMINAMATH_CALUDE_john_paid_with_one_nickel_l778_77896


namespace NUMINAMATH_CALUDE_garden_area_l778_77817

/-- Represents a rectangular garden with specific properties. -/
structure Garden where
  width : ℝ
  length : ℝ
  perimeter : ℝ
  lengthCondition : length = 3 * width + 10
  perimeterCondition : perimeter = 2 * (length + width)
  perimeterValue : perimeter = 400

/-- The area of a rectangular garden. -/
def Garden.area (g : Garden) : ℝ := g.length * g.width

/-- Theorem stating the area of the garden with given conditions. -/
theorem garden_area (g : Garden) : g.area = 7243.75 := by
  sorry

end NUMINAMATH_CALUDE_garden_area_l778_77817


namespace NUMINAMATH_CALUDE_uniform_price_uniform_price_is_75_l778_77890

/-- Calculates the price of a uniform given the conditions of a servant's employment --/
theorem uniform_price (full_year_salary : ℕ) (months_worked : ℕ) (payment_received : ℕ) : ℕ :=
  let prorated_salary := full_year_salary * months_worked / 12
  prorated_salary - payment_received

/-- Proves that the price of the uniform is 75 given the specific conditions --/
theorem uniform_price_is_75 :
  uniform_price 500 9 300 = 75 := by
  sorry

end NUMINAMATH_CALUDE_uniform_price_uniform_price_is_75_l778_77890


namespace NUMINAMATH_CALUDE_cube_face_area_l778_77826

theorem cube_face_area (V : ℝ) (h : V = 125) : ∃ (A : ℝ), A = 25 ∧ A = (V ^ (1/3)) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_face_area_l778_77826


namespace NUMINAMATH_CALUDE_coin_age_possibilities_l778_77872

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def coin_digits : List ℕ := [3, 3, 3, 5, 1, 8]

def valid_first_digit (d : ℕ) : Prop := d ∈ coin_digits ∧ is_prime d

def count_valid_ages : ℕ := 40

theorem coin_age_possibilities :
  (∀ d ∈ coin_digits, d ≥ 0 ∧ d ≤ 9) →
  (∃ d ∈ coin_digits, valid_first_digit d) →
  count_valid_ages = 40 := by
  sorry

end NUMINAMATH_CALUDE_coin_age_possibilities_l778_77872


namespace NUMINAMATH_CALUDE_jacks_lifetime_l778_77802

theorem jacks_lifetime (L : ℝ) : 
  L > 0 → 
  (1/6 : ℝ) * L + (1/12 : ℝ) * L + (1/7 : ℝ) * L + 5 + (1/2 : ℝ) * L + 4 = L → 
  L = 84 := by
sorry

end NUMINAMATH_CALUDE_jacks_lifetime_l778_77802


namespace NUMINAMATH_CALUDE_sum_of_gcd_and_lcm_equals_90_l778_77834

def numbers : List Nat := [18, 36, 72]

theorem sum_of_gcd_and_lcm_equals_90 : 
  (numbers.foldl Nat.gcd numbers.head!) + (numbers.foldl Nat.lcm numbers.head!) = 90 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_gcd_and_lcm_equals_90_l778_77834


namespace NUMINAMATH_CALUDE_divisibility_by_power_of_three_l778_77822

def sequence_a (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n ≥ 4 → a n = a (n - 1) - min (a (n - 2)) (a (n - 3))

theorem divisibility_by_power_of_three (a : ℕ → ℤ) (h : sequence_a a) :
  ∀ k : ℕ, k > 0 → ∃ n : ℕ, (3^k : ℤ) ∣ a n :=
sorry

end NUMINAMATH_CALUDE_divisibility_by_power_of_three_l778_77822


namespace NUMINAMATH_CALUDE_ryan_has_30_stickers_l778_77894

/-- The number of stickers Ryan has -/
def ryan_stickers : ℕ := 30

/-- The number of stickers Steven has -/
def steven_stickers : ℕ := 3 * ryan_stickers

/-- The number of stickers Terry has -/
def terry_stickers : ℕ := steven_stickers + 20

/-- The total number of stickers -/
def total_stickers : ℕ := 230

theorem ryan_has_30_stickers :
  ryan_stickers + steven_stickers + terry_stickers = total_stickers ∧
  ryan_stickers = 30 := by sorry

end NUMINAMATH_CALUDE_ryan_has_30_stickers_l778_77894


namespace NUMINAMATH_CALUDE_max_product_with_sum_and_even_l778_77841

theorem max_product_with_sum_and_even (x y : ℤ) : 
  x + y = 280 → (Even x ∨ Even y) → x * y ≤ 19600 := by
  sorry

end NUMINAMATH_CALUDE_max_product_with_sum_and_even_l778_77841


namespace NUMINAMATH_CALUDE_strawberry_harvest_l778_77851

theorem strawberry_harvest (base height : ℝ) (plants_per_sqft : ℕ) (strawberries_per_plant : ℕ) :
  base = 10 →
  height = 12 →
  plants_per_sqft = 5 →
  strawberries_per_plant = 8 →
  (1/2 * base * height * plants_per_sqft * strawberries_per_plant : ℝ) = 2400 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_harvest_l778_77851


namespace NUMINAMATH_CALUDE_log_sum_theorem_l778_77886

theorem log_sum_theorem (a b : ℤ) : 
  a + 1 = b → 
  (a : ℝ) < Real.log 800 / Real.log 2 → 
  (Real.log 800 / Real.log 2 : ℝ) < b → 
  a + b = 19 := by
sorry

end NUMINAMATH_CALUDE_log_sum_theorem_l778_77886


namespace NUMINAMATH_CALUDE_goldbach_negation_equivalence_l778_77829

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def goldbach_conjecture : Prop :=
  ∀ n : ℕ, n > 2 → n % 2 = 0 → ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ n = p + q

theorem goldbach_negation_equivalence :
  (¬ goldbach_conjecture) ↔
  (∃ n : ℕ, n > 2 ∧ n % 2 = 0 ∧ ∀ p q : ℕ, is_prime p → is_prime q → n ≠ p + q) :=
sorry

end NUMINAMATH_CALUDE_goldbach_negation_equivalence_l778_77829


namespace NUMINAMATH_CALUDE_total_cost_squat_rack_and_barbell_l778_77877

/-- The total cost of a squat rack and a barbell, where the squat rack costs $2500
    and the barbell costs 1/10 as much as the squat rack, is $2750. -/
theorem total_cost_squat_rack_and_barbell :
  let squat_rack_cost : ℕ := 2500
  let barbell_cost : ℕ := squat_rack_cost / 10
  squat_rack_cost + barbell_cost = 2750 :=
by sorry

end NUMINAMATH_CALUDE_total_cost_squat_rack_and_barbell_l778_77877


namespace NUMINAMATH_CALUDE_inequalities_proof_l778_77840

theorem inequalities_proof :
  (∃ a b c : ℝ, a > b ∧ a * c^2 ≤ b * c^2) ∧
  (∀ a b : ℝ, a > 0 ∧ 0 > b → a * b < a^2) ∧
  (∃ a b : ℝ, a * b = 4 ∧ a + b < 4) ∧
  (∀ a b c d : ℝ, a > b ∧ c > d → a - d > b - c) :=
sorry

end NUMINAMATH_CALUDE_inequalities_proof_l778_77840


namespace NUMINAMATH_CALUDE_two_digit_number_difference_l778_77869

/-- 
Given a two-digit number where the difference between the original number 
and the number with interchanged digits is 81, the difference between its 
two digits is 9.
-/
theorem two_digit_number_difference (x y : ℕ) : 
  x < 10 → y < 10 → (10 * x + y) - (10 * y + x) = 81 → x - y = 9 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_difference_l778_77869


namespace NUMINAMATH_CALUDE_bacteria_population_after_nine_days_l778_77875

/-- Represents the population of bacteria after a given number of 3-day periods -/
def bacteriaPopulation (initialCount : ℕ) (periods : ℕ) : ℕ :=
  initialCount * (3 ^ periods)

/-- Theorem stating that the bacteria population after 9 days (3 periods) is 36 -/
theorem bacteria_population_after_nine_days :
  bacteriaPopulation 4 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_bacteria_population_after_nine_days_l778_77875


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l778_77836

theorem quadratic_roots_condition (a : ℝ) (x₁ x₂ : ℝ) : 
  x₁ ≠ x₂ →
  x₁^2 + (a - 1) * x₁ + 2 * a - 5 = 0 →
  x₂^2 + (a - 1) * x₂ + 2 * a - 5 = 0 →
  1 / x₁ + 1 / x₂ < -3 / 5 →
  a > 5 / 2 ∧ a < 10 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l778_77836


namespace NUMINAMATH_CALUDE_derivative_f_l778_77824

noncomputable def f (x : ℝ) : ℝ := (1/4) * Real.log ((x-1)/(x+1)) - (1/2) * Real.arctan x

theorem derivative_f (x : ℝ) (h₁ : x ≠ 1) (h₂ : x ≠ -1) :
  deriv f x = 1 / (x^4 - 1) := by sorry

end NUMINAMATH_CALUDE_derivative_f_l778_77824


namespace NUMINAMATH_CALUDE_manny_purchase_theorem_l778_77865

/-- The cost of a plastic chair in dollars -/
def chair_cost : ℚ := 55 / 5

/-- The cost of a portable table in dollars -/
def table_cost : ℚ := 3 * chair_cost

/-- Manny's initial amount in dollars -/
def initial_amount : ℚ := 100

/-- The cost of Manny's purchase (one table and two chairs) in dollars -/
def purchase_cost : ℚ := table_cost + 2 * chair_cost

/-- The amount left after Manny's purchase in dollars -/
def amount_left : ℚ := initial_amount - purchase_cost

theorem manny_purchase_theorem : amount_left = 45 := by
  sorry

end NUMINAMATH_CALUDE_manny_purchase_theorem_l778_77865


namespace NUMINAMATH_CALUDE_max_abs_sum_under_condition_l778_77862

theorem max_abs_sum_under_condition (x y : ℝ) (h : x^2 + y^2 + x*y = 1) :
  |x| + |y| ≤ Real.sqrt (4/3) := by
  sorry

end NUMINAMATH_CALUDE_max_abs_sum_under_condition_l778_77862


namespace NUMINAMATH_CALUDE_probability_all_colors_l778_77801

/-- The probability of selecting 4 balls from a bag containing 2 red balls, 3 white balls, and 4 yellow balls, such that the selection includes balls of all three colors, is equal to 4/7. -/
theorem probability_all_colors (red : ℕ) (white : ℕ) (yellow : ℕ) (total_select : ℕ) :
  red = 2 →
  white = 3 →
  yellow = 4 →
  total_select = 4 →
  (Nat.choose (red + white + yellow) total_select : ℚ) ≠ 0 →
  (↑(Nat.choose red 2 * Nat.choose white 1 * Nat.choose yellow 1 +
     Nat.choose red 1 * Nat.choose white 2 * Nat.choose yellow 1 +
     Nat.choose red 1 * Nat.choose white 1 * Nat.choose yellow 2) /
   Nat.choose (red + white + yellow) total_select : ℚ) = 4 / 7 :=
by sorry

end NUMINAMATH_CALUDE_probability_all_colors_l778_77801


namespace NUMINAMATH_CALUDE_library_crates_l778_77871

theorem library_crates (novels : ℕ) (comics : ℕ) (documentaries : ℕ) (albums : ℕ) 
  (crate_capacity : ℕ) (h1 : novels = 145) (h2 : comics = 271) (h3 : documentaries = 419) 
  (h4 : albums = 209) (h5 : crate_capacity = 9) : 
  ((novels + comics + documentaries + albums) / crate_capacity : ℕ) = 116 := by
  sorry

end NUMINAMATH_CALUDE_library_crates_l778_77871


namespace NUMINAMATH_CALUDE_solution_set_of_equation_l778_77888

def is_solution (x : ℝ) : Prop :=
  -2*x > 0 ∧ 3 - x^2 > 0 ∧ -2*x = 3 - x^2

theorem solution_set_of_equation : 
  {x : ℝ | is_solution x} = {-1} := by sorry

end NUMINAMATH_CALUDE_solution_set_of_equation_l778_77888


namespace NUMINAMATH_CALUDE_product_is_term_iff_first_term_is_power_of_ratio_l778_77830

/-- A geometric progression is defined by its first term and common ratio -/
structure GeometricProgression where
  a : ℝ  -- First term
  q : ℝ  -- Common ratio

/-- The nth term of a geometric progression -/
def GeometricProgression.nthTerm (gp : GeometricProgression) (n : ℕ) : ℝ :=
  gp.a * gp.q ^ n

/-- Condition for product of terms to be another term -/
def productIsTermCondition (gp : GeometricProgression) : Prop :=
  ∃ m : ℤ, gp.a = gp.q ^ m

theorem product_is_term_iff_first_term_is_power_of_ratio (gp : GeometricProgression) :
  (∀ n p k : ℕ, ∃ k : ℕ, gp.nthTerm n * gp.nthTerm p = gp.nthTerm k) ↔
  productIsTermCondition gp :=
sorry

end NUMINAMATH_CALUDE_product_is_term_iff_first_term_is_power_of_ratio_l778_77830


namespace NUMINAMATH_CALUDE_vector_properties_l778_77811

def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (1, -1)
def c (m n : ℝ) : ℝ × ℝ := (m - 2, n)

variable (m n : ℝ)
variable (hm : m > 0)
variable (hn : n > 0)

theorem vector_properties :
  (a.1 * b.1 + a.2 * b.2 = 1) ∧
  ((a.1 - b.1) * (c m n).1 + (a.2 - b.2) * (c m n).2 = 0 → m + 2*n = 2) := by
  sorry

end NUMINAMATH_CALUDE_vector_properties_l778_77811


namespace NUMINAMATH_CALUDE_black_marble_price_is_ten_cents_l778_77809

/-- Represents the marble pricing problem --/
structure MarbleProblem where
  total_marbles : ℕ
  white_percentage : ℚ
  black_percentage : ℚ
  white_price : ℚ
  color_price : ℚ
  total_earnings : ℚ

/-- Calculates the price of each black marble --/
def black_marble_price (p : MarbleProblem) : ℚ :=
  let white_marbles := p.white_percentage * p.total_marbles
  let black_marbles := p.black_percentage * p.total_marbles
  let color_marbles := p.total_marbles - (white_marbles + black_marbles)
  let white_earnings := white_marbles * p.white_price
  let color_earnings := color_marbles * p.color_price
  let black_earnings := p.total_earnings - (white_earnings + color_earnings)
  black_earnings / black_marbles

/-- Theorem stating that the black marble price is $0.10 --/
theorem black_marble_price_is_ten_cents 
  (p : MarbleProblem) 
  (h1 : p.total_marbles = 100)
  (h2 : p.white_percentage = 1/5)
  (h3 : p.black_percentage = 3/10)
  (h4 : p.white_price = 1/20)
  (h5 : p.color_price = 1/5)
  (h6 : p.total_earnings = 14) :
  black_marble_price p = 1/10 := by
  sorry

#eval black_marble_price { 
  total_marbles := 100, 
  white_percentage := 1/5, 
  black_percentage := 3/10, 
  white_price := 1/5, 
  color_price := 1/5, 
  total_earnings := 14 
}

end NUMINAMATH_CALUDE_black_marble_price_is_ten_cents_l778_77809


namespace NUMINAMATH_CALUDE_tasty_candy_identification_l778_77820

/-- Represents a strategy for identifying tasty candies -/
structure TastyStrategy where
  query : (ℕ → Bool) → Finset ℕ → ℕ
  interpret : (Finset ℕ → ℕ) → Finset ℕ

/-- The total number of candies -/
def total_candies : ℕ := 28

/-- A function that determines if a candy is tasty -/
def is_tasty : ℕ → Bool := sorry

/-- The maximum number of queries allowed -/
def max_queries : ℕ := 20

theorem tasty_candy_identification :
  ∃ (s : TastyStrategy),
    (∀ (f : ℕ → Bool),
      let query_count := (Finset.range total_candies).card
      s.interpret (λ subset => s.query f subset) =
        {i | i ∈ Finset.range total_candies ∧ f i}) ∧
    (∀ (f : ℕ → Bool),
      (Finset.range total_candies).card ≤ max_queries) :=
sorry

end NUMINAMATH_CALUDE_tasty_candy_identification_l778_77820


namespace NUMINAMATH_CALUDE_fraction_of_8000_l778_77823

theorem fraction_of_8000 (x : ℝ) : x = 0.1 →
  x * 8000 - (1 / 20) * (1 / 100) * 8000 = 796 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_8000_l778_77823


namespace NUMINAMATH_CALUDE_parking_spots_total_l778_77889

/-- Calculates the total number of open parking spots in a 4-story parking area -/
def total_parking_spots (first_level : ℕ) (second_level_diff : ℕ) (third_level_diff : ℕ) (fourth_level : ℕ) : ℕ :=
  let second_level := first_level + second_level_diff
  let third_level := second_level + third_level_diff
  first_level + second_level + third_level + fourth_level

/-- Theorem: The total number of open parking spots is 46 -/
theorem parking_spots_total : 
  total_parking_spots 4 7 6 14 = 46 := by
  sorry

end NUMINAMATH_CALUDE_parking_spots_total_l778_77889


namespace NUMINAMATH_CALUDE_greatest_integer_y_l778_77810

theorem greatest_integer_y (y : ℕ+) : (y.val : ℝ)^4 / (y.val : ℝ)^2 < 18 ↔ y.val ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_y_l778_77810


namespace NUMINAMATH_CALUDE_jenny_garden_area_l778_77856

/-- Represents a rectangular garden with fence posts. -/
structure Garden where
  total_posts : ℕ
  post_spacing : ℕ
  shorter_side_posts : ℕ
  longer_side_posts : ℕ

/-- Calculates the area of a rectangular garden given its specifications. -/
def garden_area (g : Garden) : ℕ :=
  (g.shorter_side_posts - 1) * g.post_spacing * ((g.longer_side_posts - 1) * g.post_spacing)

/-- Theorem: The area of Jenny's garden is 144 square yards. -/
theorem jenny_garden_area :
  ∀ g : Garden,
    g.total_posts = 24 →
    g.post_spacing = 3 →
    g.longer_side_posts = 3 * g.shorter_side_posts →
    g.total_posts = 2 * (g.shorter_side_posts + g.longer_side_posts - 2) →
    garden_area g = 144 := by
  sorry

#eval garden_area { total_posts := 24, post_spacing := 3, shorter_side_posts := 3, longer_side_posts := 9 }

end NUMINAMATH_CALUDE_jenny_garden_area_l778_77856


namespace NUMINAMATH_CALUDE_solve_table_height_l778_77818

def table_height_problem (initial_measurement : ℝ) (rearranged_measurement : ℝ) 
  (block_width : ℝ) (table_thickness : ℝ) : Prop :=
  ∃ (h l : ℝ),
    l + h - block_width + table_thickness = initial_measurement ∧
    block_width + h - l + table_thickness = rearranged_measurement ∧
    h = 33

theorem solve_table_height :
  table_height_problem 40 34 6 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_table_height_l778_77818


namespace NUMINAMATH_CALUDE_alternating_square_sum_equals_5304_l778_77814

def alternatingSquareSum (n : ℕ) : ℤ :=
  let seq := List.range n |> List.reverse |> List.map (λ i => (101 - i : ℤ)^2)
  seq.enum.foldl (λ acc (i, x) => acc + (if i % 4 < 2 then x else -x)) 0

theorem alternating_square_sum_equals_5304 :
  alternatingSquareSum 100 = 5304 := by
  sorry

end NUMINAMATH_CALUDE_alternating_square_sum_equals_5304_l778_77814


namespace NUMINAMATH_CALUDE_ordering_abc_l778_77876

def a : ℝ := (2 : ℝ) ^ (4/3)
def b : ℝ := (4 : ℝ) ^ (2/5)
def c : ℝ := (25 : ℝ) ^ (1/3)

theorem ordering_abc : b < a ∧ a < c := by sorry

end NUMINAMATH_CALUDE_ordering_abc_l778_77876


namespace NUMINAMATH_CALUDE_interest_rate_problem_l778_77850

/-- Calculates simple interest given principal, rate, and time -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem interest_rate_problem (principal : ℝ) (time : ℝ) (rate : ℝ) 
    (h1 : principal = 15000)
    (h2 : time = 2)
    (h3 : simpleInterest principal rate time = simpleInterest principal 0.12 time + 900) :
  rate = 0.15 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_problem_l778_77850


namespace NUMINAMATH_CALUDE_systematic_sampling_l778_77832

theorem systematic_sampling (total_students : Nat) (num_groups : Nat) (first_group_number : Nat) (target_group : Nat) :
  total_students = 480 →
  num_groups = 30 →
  first_group_number = 5 →
  target_group = 8 →
  (target_group - 1) * (total_students / num_groups) + first_group_number = 117 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_l778_77832


namespace NUMINAMATH_CALUDE_banana_count_l778_77885

theorem banana_count (total : ℕ) (apple_multiplier persimmon_multiplier : ℕ) 
  (h1 : total = 210)
  (h2 : apple_multiplier = 4)
  (h3 : persimmon_multiplier = 3) :
  ∃ (banana_count : ℕ), 
    banana_count * (apple_multiplier + persimmon_multiplier) = total ∧ 
    banana_count = 30 := by
  sorry

end NUMINAMATH_CALUDE_banana_count_l778_77885


namespace NUMINAMATH_CALUDE_beach_seashells_l778_77891

/-- 
Given a person who spends 5 days at the beach and finds 7 seashells each day,
the total number of seashells found during the trip is 35.
-/
theorem beach_seashells : 
  ∀ (days : ℕ) (shells_per_day : ℕ),
  days = 5 → shells_per_day = 7 →
  days * shells_per_day = 35 := by
sorry

end NUMINAMATH_CALUDE_beach_seashells_l778_77891


namespace NUMINAMATH_CALUDE_shopping_tax_theorem_l778_77854

/-- Calculates the total tax percentage given spending percentages and tax rates -/
def total_tax_percentage (clothing_percent : ℝ) (food_percent : ℝ) (other_percent : ℝ)
                         (clothing_tax_rate : ℝ) (food_tax_rate : ℝ) (other_tax_rate : ℝ) : ℝ :=
  (clothing_percent * clothing_tax_rate + food_percent * food_tax_rate + other_percent * other_tax_rate) * 100

/-- Theorem stating that the total tax percentage is 4.8% given the specified conditions -/
theorem shopping_tax_theorem :
  total_tax_percentage 0.6 0.1 0.3 0.04 0 0.08 = 4.8 := by
  sorry

#eval total_tax_percentage 0.6 0.1 0.3 0.04 0 0.08

end NUMINAMATH_CALUDE_shopping_tax_theorem_l778_77854


namespace NUMINAMATH_CALUDE_power_two_greater_than_square_plus_one_l778_77863

theorem power_two_greater_than_square_plus_one (n : ℕ) (h : n ≥ 5) :
  2^n > n^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_power_two_greater_than_square_plus_one_l778_77863


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_m_half_l778_77859

/-- Given two non-zero parallel vectors (m^2-1, m+1) and (1, -2), prove that m = 1/2 -/
theorem parallel_vectors_imply_m_half (m : ℝ) : 
  (m^2 - 1 ≠ 0 ∨ m + 1 ≠ 0) →  -- Vector a is non-zero
  (∃ (k : ℝ), k ≠ 0 ∧ k * (1 : ℝ) = m^2 - 1 ∧ k * (-2 : ℝ) = m + 1) →  -- Vectors are parallel
  m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_m_half_l778_77859


namespace NUMINAMATH_CALUDE_alex_walking_distance_l778_77803

/-- Calculates the distance Alex had to walk after his tire punctured --/
def distance_to_walk (total_distance : ℝ) (flat_speed flat_time : ℝ) (uphill_speed uphill_time : ℝ) (downhill_speed downhill_time : ℝ) : ℝ :=
  total_distance - (flat_speed * flat_time + uphill_speed * uphill_time + downhill_speed * downhill_time)

theorem alex_walking_distance :
  distance_to_walk 164 20 4.5 12 2.5 24 1.5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_alex_walking_distance_l778_77803


namespace NUMINAMATH_CALUDE_reading_time_calculation_l778_77805

theorem reading_time_calculation (pages_book1 pages_book2 : ℕ) 
  (time_book1 time_book2 : ℚ) (pages_to_read1 pages_to_read2 : ℕ) :
  pages_book1 = 4 →
  pages_book2 = 5 →
  time_book1 = 2 →
  time_book2 = 3 →
  pages_to_read1 = 36 →
  pages_to_read2 = 25 →
  (pages_to_read1 * (time_book1 / pages_book1) + 
   pages_to_read2 * (time_book2 / pages_book2)) = 33 :=
by
  sorry

end NUMINAMATH_CALUDE_reading_time_calculation_l778_77805


namespace NUMINAMATH_CALUDE_total_gas_usage_l778_77819

theorem total_gas_usage (adhira_usage : ℕ) (felicity_usage : ℕ) : 
  felicity_usage = 4 * adhira_usage - 5 →
  felicity_usage = 23 →
  felicity_usage + adhira_usage = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_total_gas_usage_l778_77819


namespace NUMINAMATH_CALUDE_vertex_sum_is_ten_l778_77866

/-- Two parabolas intersecting at two points -/
structure IntersectingParabolas where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  h1 : -(3 - a)^2 + b = 6
  h2 : (3 - c)^2 + d = 6
  h3 : -(7 - a)^2 + b = 2
  h4 : (7 - c)^2 + d = 2

/-- The sum of x-coordinates of the vertices of two intersecting parabolas -/
def vertexSum (p : IntersectingParabolas) : ℝ := p.a + p.c

/-- Theorem: The sum of x-coordinates of the vertices of two intersecting parabolas is 10 -/
theorem vertex_sum_is_ten (p : IntersectingParabolas) : vertexSum p = 10 := by
  sorry

end NUMINAMATH_CALUDE_vertex_sum_is_ten_l778_77866


namespace NUMINAMATH_CALUDE_count_perimeters_eq_42_l778_77860

/-- Represents a quadrilateral EFGH with specific properties -/
structure Quadrilateral where
  ef : ℕ+
  fg : ℕ+
  gh : ℕ+
  eh : ℕ+
  perimeter_lt_1200 : ef.val + fg.val + gh.val + eh.val < 1200
  right_angle_f : True
  right_angle_g : True
  ef_eq_3 : ef = 3
  gh_eq_eh : gh = eh

/-- The number of different possible perimeter values -/
def count_perimeters : ℕ := sorry

/-- Main theorem stating the number of different possible perimeter values -/
theorem count_perimeters_eq_42 : count_perimeters = 42 := by sorry

end NUMINAMATH_CALUDE_count_perimeters_eq_42_l778_77860


namespace NUMINAMATH_CALUDE_fraction_integer_values_l778_77883

theorem fraction_integer_values (a : ℤ) :
  (∃ k : ℤ, (a^3 + 1) / (a - 1) = k) ↔ a = -1 ∨ a = 0 ∨ a = 2 ∨ a = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_integer_values_l778_77883


namespace NUMINAMATH_CALUDE_tank_inflow_rate_l778_77807

/-- Given two tanks with equal capacity, prove the inflow rate of the slower-filling tank -/
theorem tank_inflow_rate (capacity : ℝ) (fast_rate slow_rate : ℝ) (time_diff : ℝ) :
  capacity > 0 →
  fast_rate > 0 →
  slow_rate > 0 →
  time_diff > 0 →
  capacity / fast_rate + time_diff = capacity / slow_rate →
  capacity = 20 →
  fast_rate = 4 →
  time_diff = 5 →
  slow_rate = 2 := by sorry

end NUMINAMATH_CALUDE_tank_inflow_rate_l778_77807


namespace NUMINAMATH_CALUDE_fgh_supermarkets_in_us_l778_77808

/-- The number of FGH supermarkets in the US, given the total number of supermarkets
    and the difference between US and Canadian supermarkets. -/
def us_supermarkets (total : ℕ) (difference : ℕ) : ℕ :=
  (total + difference) / 2

theorem fgh_supermarkets_in_us :
  us_supermarkets 60 22 = 41 := by
  sorry

#eval us_supermarkets 60 22

end NUMINAMATH_CALUDE_fgh_supermarkets_in_us_l778_77808


namespace NUMINAMATH_CALUDE_northwest_molded_break_even_l778_77881

/-- Break-even point calculation for Northwest Molded -/
theorem northwest_molded_break_even :
  let fixed_cost : ℝ := 7640
  let variable_cost : ℝ := 0.60
  let selling_price : ℝ := 4.60
  let break_even_point := fixed_cost / (selling_price - variable_cost)
  break_even_point = 1910 := by
  sorry

end NUMINAMATH_CALUDE_northwest_molded_break_even_l778_77881


namespace NUMINAMATH_CALUDE_town_population_males_l778_77895

theorem town_population_males (total_population : ℕ) (num_segments : ℕ) (male_segments : ℕ) :
  total_population = 800 →
  num_segments = 4 →
  male_segments = 1 →
  2 * (total_population / num_segments * male_segments) = total_population →
  total_population / num_segments * male_segments = 400 :=
by sorry

end NUMINAMATH_CALUDE_town_population_males_l778_77895


namespace NUMINAMATH_CALUDE_a_share_is_4080_l778_77852

/-- Calculates the share of profit for an investor in a partnership business. -/
def calculate_share_of_profit (investment_a investment_b investment_c total_profit : ℚ) : ℚ :=
  let total_investment := investment_a + investment_b + investment_c
  let ratio_a := investment_a / total_investment
  ratio_a * total_profit

/-- Theorem stating that A's share of the profit is 4080 given the investments and total profit. -/
theorem a_share_is_4080 
  (investment_a : ℚ) 
  (investment_b : ℚ) 
  (investment_c : ℚ) 
  (total_profit : ℚ) 
  (h1 : investment_a = 6300)
  (h2 : investment_b = 4200)
  (h3 : investment_c = 10500)
  (h4 : total_profit = 13600) :
  calculate_share_of_profit investment_a investment_b investment_c total_profit = 4080 := by
  sorry

#eval calculate_share_of_profit 6300 4200 10500 13600

end NUMINAMATH_CALUDE_a_share_is_4080_l778_77852


namespace NUMINAMATH_CALUDE_square_cardinality_continuum_l778_77873

/-- A square in the 2D plane -/
def Square : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

/-- The unit interval [0, 1] -/
def UnitInterval : Set ℝ :=
  {x | 0 ≤ x ∧ x ≤ 1}

theorem square_cardinality_continuum :
  Cardinal.mk (Square) = Cardinal.mk (UnitInterval) :=
sorry

end NUMINAMATH_CALUDE_square_cardinality_continuum_l778_77873


namespace NUMINAMATH_CALUDE_inequality_range_l778_77893

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, x > 0 → x - Real.log x - a > 0) ↔ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l778_77893


namespace NUMINAMATH_CALUDE_remove_horizontal_eliminates_triangles_l778_77867

/-- Represents a triangular grid constructed with toothpicks -/
structure TriangularGrid where
  toothpicks : ℕ
  rows : ℕ
  columns : ℕ
  triangles : ℕ

/-- The specific triangular grid in the problem -/
def problemGrid : TriangularGrid :=
  { toothpicks := 36
  , rows := 3
  , columns := 5
  , triangles := 35 }

/-- The number of horizontal toothpicks in the grid -/
def horizontalToothpicks (grid : TriangularGrid) : ℕ := grid.rows * grid.columns

/-- Theorem stating that removing all horizontal toothpicks eliminates all triangles -/
theorem remove_horizontal_eliminates_triangles (grid : TriangularGrid) :
  horizontalToothpicks grid = 15 ∧
  horizontalToothpicks grid ≤ grid.toothpicks ∧
  grid.triangles > 35 →
  (∀ n : ℕ, n < 15 → ∃ t : ℕ, t > 0) ∧
  (∀ t : ℕ, t = 0) :=
sorry

#check remove_horizontal_eliminates_triangles problemGrid

end NUMINAMATH_CALUDE_remove_horizontal_eliminates_triangles_l778_77867


namespace NUMINAMATH_CALUDE_sports_parade_children_count_l778_77844

theorem sports_parade_children_count :
  ∃! n : ℕ, 100 ≤ n ∧ n ≤ 150 ∧ n % 8 = 5 ∧ n % 10 = 7 ∧ n = 125 := by
  sorry

end NUMINAMATH_CALUDE_sports_parade_children_count_l778_77844


namespace NUMINAMATH_CALUDE_volunteer_schedule_l778_77884

theorem volunteer_schedule (sasha leo uma kim : ℕ) 
  (h1 : sasha = 5) 
  (h2 : leo = 8) 
  (h3 : uma = 9) 
  (h4 : kim = 10) : 
  Nat.lcm (Nat.lcm (Nat.lcm sasha leo) uma) kim = 360 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_schedule_l778_77884


namespace NUMINAMATH_CALUDE_total_hunt_is_21_l778_77838

/-- The number of animals hunted by Sam in a day -/
def sam_hunt : ℕ := 6

/-- The number of animals hunted by Rob in a day -/
def rob_hunt : ℕ := sam_hunt / 2

/-- The number of animals hunted by Mark in a day -/
def mark_hunt : ℕ := (sam_hunt + rob_hunt) / 3

/-- The number of animals hunted by Peter in a day -/
def peter_hunt : ℕ := 3 * mark_hunt

/-- The total number of animals hunted by all four in a day -/
def total_hunt : ℕ := sam_hunt + rob_hunt + mark_hunt + peter_hunt

theorem total_hunt_is_21 : total_hunt = 21 := by
  sorry

end NUMINAMATH_CALUDE_total_hunt_is_21_l778_77838


namespace NUMINAMATH_CALUDE_prism_volume_l778_77857

/-- The volume of a right rectangular prism with given face areas -/
theorem prism_volume (a b c : ℝ) (h1 : a * b = 56) (h2 : b * c = 63) (h3 : 2 * a * c = 72) :
  a * b * c = 504 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l778_77857


namespace NUMINAMATH_CALUDE_company_ratio_is_9_47_l778_77874

/-- Represents the ratio of managers to non-managers in a company -/
structure ManagerRatio where
  managers : ℕ
  non_managers : ℕ

/-- The company's policy for manager to non-manager ratio -/
axiom company_ratio : ManagerRatio

/-- The ratio is constant across all departments -/
axiom ratio_constant (dept1 dept2 : ManagerRatio) : 
  dept1.managers * dept2.non_managers = dept1.non_managers * dept2.managers

/-- In a department with 9 managers, the maximum number of non-managers is 47 -/
axiom max_non_managers : ∃ (dept : ManagerRatio), dept.managers = 9 ∧ dept.non_managers = 47

/-- The company ratio is equal to 9:47 -/
theorem company_ratio_is_9_47 : company_ratio.managers = 9 ∧ company_ratio.non_managers = 47 := by
  sorry

end NUMINAMATH_CALUDE_company_ratio_is_9_47_l778_77874


namespace NUMINAMATH_CALUDE_velocity_from_similarity_l778_77812

/-- Given a, T, R, L, and x as real numbers, where x represents a distance,
    and assuming the equation (a * T) / (a * T - R) = (L + x) / x holds,
    prove that the velocity of the point described by x is a * (L / R). -/
theorem velocity_from_similarity (a T R L x : ℝ) (h : (a * T) / (a * T - R) = (L + x) / x) :
  x / T = a * L / R := by
  sorry

end NUMINAMATH_CALUDE_velocity_from_similarity_l778_77812


namespace NUMINAMATH_CALUDE_point_in_region_l778_77899

def in_region (x y : ℝ) : Prop := 2 * x + y - 6 ≤ 0

theorem point_in_region :
  in_region 0 6 ∧
  ¬in_region 0 7 ∧
  ¬in_region 5 0 ∧
  ¬in_region 2 3 :=
by sorry

end NUMINAMATH_CALUDE_point_in_region_l778_77899


namespace NUMINAMATH_CALUDE_inequality_proof_l778_77833

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 / (b + c) + 1 / (a + c) + 1 / (a + b) ≥ 9 / (2 * (a + b + c)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l778_77833


namespace NUMINAMATH_CALUDE_hair_length_after_growth_and_cut_l778_77853

theorem hair_length_after_growth_and_cut (x : ℝ) : 
  let initial_length : ℝ := 14
  let growth : ℝ := x
  let cut_length : ℝ := 20
  let final_length : ℝ := initial_length + growth - cut_length
  final_length = x - 6 := by sorry

end NUMINAMATH_CALUDE_hair_length_after_growth_and_cut_l778_77853


namespace NUMINAMATH_CALUDE_line_inclination_angle_l778_77858

/-- The inclination angle of a line is the angle between the positive x-axis and the line, measured counterclockwise. -/
def inclination_angle (a b c : ℝ) : ℝ := sorry

/-- A line is defined by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

theorem line_inclination_angle :
  let l : Line := { a := 1, b := 1, c := 1 }
  inclination_angle l.a l.b l.c = 135 * π / 180 := by sorry

end NUMINAMATH_CALUDE_line_inclination_angle_l778_77858


namespace NUMINAMATH_CALUDE_line_ab_equals_ba_infinite_lines_through_point_ray_ab_not_equal_ba_ray_line_length_incomparable_l778_77861

-- Define basic geometric structures
structure Point : Type :=
  (x : ℝ) (y : ℝ)

structure Line : Type :=
  (p1 : Point) (p2 : Point)

structure Ray : Type :=
  (start : Point) (through : Point)

-- Define equality for lines
def line_eq (l1 l2 : Line) : Prop :=
  (l1.p1 = l2.p1 ∧ l1.p2 = l2.p2) ∨ (l1.p1 = l2.p2 ∧ l1.p2 = l2.p1)

-- Define inequality for rays
def ray_neq (r1 r2 : Ray) : Prop :=
  r1.start ≠ r2.start ∨ r1.through ≠ r2.through

-- Theorem statements
theorem line_ab_equals_ba (A B : Point) : 
  line_eq (Line.mk A B) (Line.mk B A) :=
sorry

theorem infinite_lines_through_point (P : Point) :
  ∀ n : ℕ, ∃ (lines : Fin n → Line), ∀ i : Fin n, (lines i).p1 = P :=
sorry

theorem ray_ab_not_equal_ba (A B : Point) :
  ray_neq (Ray.mk A B) (Ray.mk B A) :=
sorry

theorem ray_line_length_incomparable :
  ¬∃ (f : Ray → ℝ) (g : Line → ℝ), ∀ (r : Ray) (l : Line), f r < g l :=
sorry

end NUMINAMATH_CALUDE_line_ab_equals_ba_infinite_lines_through_point_ray_ab_not_equal_ba_ray_line_length_incomparable_l778_77861


namespace NUMINAMATH_CALUDE_value_not_unique_l778_77845

/-- A quadratic function passing through (0, 1) and (1, 0), and concave down -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  pass_origin : c = 1
  pass_one : a + b + c = 0
  concave_down : a < 0

/-- The value of a - b + c cannot be uniquely determined for all quadratic functions satisfying the given conditions -/
theorem value_not_unique (f g : QuadraticFunction) : ∃ (f g : QuadraticFunction), f.a - f.b + f.c ≠ g.a - g.b + g.c := by
  sorry

end NUMINAMATH_CALUDE_value_not_unique_l778_77845


namespace NUMINAMATH_CALUDE_power_of_three_squared_l778_77828

theorem power_of_three_squared : 3^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_squared_l778_77828


namespace NUMINAMATH_CALUDE_sam_sticker_spending_l778_77816

/-- Given Sam's initial penny count and his spending on toys and candy, 
    calculate the amount spent on stickers. -/
theorem sam_sticker_spending 
  (total : ℕ) 
  (toy_cost : ℕ) 
  (candy_cost : ℕ) 
  (h1 : total = 2476) 
  (h2 : toy_cost = 1145) 
  (h3 : candy_cost = 781) :
  total - (toy_cost + candy_cost) = 550 := by
  sorry

#check sam_sticker_spending

end NUMINAMATH_CALUDE_sam_sticker_spending_l778_77816


namespace NUMINAMATH_CALUDE_daily_savings_amount_l778_77848

/-- Proves that saving the same amount daily for 20 days totaling 2 dimes equals 1 cent per day -/
theorem daily_savings_amount (savings_period : ℕ) (total_saved : ℕ) (daily_amount : ℚ) : 
  savings_period = 20 →
  total_saved = 20 →  -- 2 dimes = 20 cents
  daily_amount * savings_period = total_saved →
  daily_amount = 1 := by
sorry

end NUMINAMATH_CALUDE_daily_savings_amount_l778_77848


namespace NUMINAMATH_CALUDE_overtime_hours_calculation_l778_77831

theorem overtime_hours_calculation 
  (regular_rate : ℝ) 
  (regular_hours : ℝ) 
  (total_pay : ℝ) 
  (h1 : regular_rate = 3)
  (h2 : regular_hours = 40)
  (h3 : total_pay = 192) :
  let overtime_rate := 2 * regular_rate
  let regular_pay := regular_rate * regular_hours
  let overtime_pay := total_pay - regular_pay
  overtime_pay / overtime_rate = 12 := by
sorry

end NUMINAMATH_CALUDE_overtime_hours_calculation_l778_77831


namespace NUMINAMATH_CALUDE_division_multiplication_equivalence_l778_77843

theorem division_multiplication_equivalence : 
  ∀ (x : ℚ), x * (9 / 3) * (5 / 6) = x / (2 / 5) :=
by sorry

end NUMINAMATH_CALUDE_division_multiplication_equivalence_l778_77843


namespace NUMINAMATH_CALUDE_probability_sum_17_l778_77879

def roll_sum_17 : ℕ := 56

def total_outcomes : ℕ := 6^4

theorem probability_sum_17 : 
  (roll_sum_17 : ℚ) / total_outcomes = 7 / 162 :=
sorry

end NUMINAMATH_CALUDE_probability_sum_17_l778_77879


namespace NUMINAMATH_CALUDE_f_properties_l778_77870

noncomputable def f (x : ℝ) : ℝ := -Real.sqrt 3 * Real.sin x ^ 2 + Real.sin x * Real.cos x

theorem f_properties :
  (f (25 * Real.pi / 6) = 0) ∧
  (∀ T > 0, (∀ x, f (x + T) = f x) → T ≥ Real.pi) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ 1 - Real.sqrt 3 / 2) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≥ -Real.sqrt 3) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = 1 - Real.sqrt 3 / 2) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = -Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l778_77870


namespace NUMINAMATH_CALUDE_units_digit_of_3_pow_2011_l778_77864

def units_digit (n : ℕ) : ℕ := n % 10

def power_of_3_units_digit (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 3
  | 2 => 9
  | 3 => 7
  | _ => 0  -- This case should never occur

theorem units_digit_of_3_pow_2011 :
  units_digit (3^2011) = power_of_3_units_digit 2011 :=
by sorry

end NUMINAMATH_CALUDE_units_digit_of_3_pow_2011_l778_77864


namespace NUMINAMATH_CALUDE_multiple_properties_l778_77880

-- Define x and y as integers
variable (x y : ℤ)

-- Define the conditions
variable (h1 : ∃ k : ℤ, x = 8 * k)
variable (h2 : ∃ m : ℤ, y = 12 * m)

-- Theorem to prove
theorem multiple_properties :
  (∃ n : ℤ, y = 4 * n) ∧ (∃ p : ℤ, x - y = 4 * p) :=
by sorry

end NUMINAMATH_CALUDE_multiple_properties_l778_77880


namespace NUMINAMATH_CALUDE_symmetry_line_l778_77878

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 4*y + 4 = 0

-- Define the line
def line_l (x y : ℝ) : Prop := y = x - 2

-- Theorem statement
theorem symmetry_line :
  ∀ (x1 y1 x2 y2 : ℝ),
  circle1 x1 y1 → circle2 x2 y2 →
  ∃ (x y : ℝ), line_l x y ∧
  (x = (x1 + x2) / 2) ∧ (y = (y1 + y2) / 2) ∧
  ((x - x1)^2 + (y - y1)^2 = (x - x2)^2 + (y - y2)^2) :=
sorry

end NUMINAMATH_CALUDE_symmetry_line_l778_77878


namespace NUMINAMATH_CALUDE_intersection_M_N_l778_77897

-- Define the sets M and N
def M : Set ℝ := {x | Real.log (1 - x) < 0}
def N : Set ℝ := {x | x^2 ≤ 1}

-- State the theorem
theorem intersection_M_N : M ∩ N = Set.Ioo 0 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l778_77897


namespace NUMINAMATH_CALUDE_power_algorithm_correct_l778_77847

/-- Algorithm to compute B^N -/
def power_algorithm (B : ℝ) (N : ℕ) : ℝ :=
  if N = 0 then 1
  else
    let rec loop (a b : ℝ) (n : ℕ) : ℝ :=
      if n = 0 then a
      else if n % 2 = 0 then loop a (b * b) (n / 2)
      else loop (a * b) (b * b) (n / 2)
    loop 1 B N

/-- Theorem stating that the algorithm computes B^N -/
theorem power_algorithm_correct (B : ℝ) (N : ℕ) (hB : B > 0) :
  power_algorithm B N = B ^ N := by
  sorry

#check power_algorithm_correct

end NUMINAMATH_CALUDE_power_algorithm_correct_l778_77847


namespace NUMINAMATH_CALUDE_additional_money_needed_mrs_smith_shopping_l778_77868

/-- Calculates the additional money needed for Mrs. Smith's shopping trip --/
theorem additional_money_needed (total_budget : ℚ) (dress_budget : ℚ) (shoe_budget : ℚ) (accessory_budget : ℚ)
  (increase_ratio : ℚ) (discount_rate : ℚ) : ℚ :=
  let total_needed := (dress_budget + shoe_budget + accessory_budget) * (1 + increase_ratio)
  let discounted_total := total_needed * (1 - discount_rate)
  discounted_total - total_budget

/-- Proves that Mrs. Smith needs $95 more --/
theorem mrs_smith_shopping : 
  additional_money_needed 500 300 150 50 (2/5) (15/100) = 95 := by
  sorry

end NUMINAMATH_CALUDE_additional_money_needed_mrs_smith_shopping_l778_77868


namespace NUMINAMATH_CALUDE_sample_size_is_thirteen_l778_77882

/-- Represents a workshop with its production quantity -/
structure Workshop where
  quantity : ℕ

/-- Represents a stratified sampling scenario -/
structure StratifiedSampling where
  workshops : List Workshop
  sampleFromSmallest : ℕ

/-- Calculates the total sample size for a stratified sampling scenario -/
def totalSampleSize (s : StratifiedSampling) : ℕ :=
  sorry

/-- The main theorem stating that for the given scenario, the total sample size is 13 -/
theorem sample_size_is_thirteen :
  let scenario := StratifiedSampling.mk
    [Workshop.mk 120, Workshop.mk 80, Workshop.mk 60]
    3
  totalSampleSize scenario = 13 := by
  sorry

end NUMINAMATH_CALUDE_sample_size_is_thirteen_l778_77882


namespace NUMINAMATH_CALUDE_hansol_weight_l778_77839

/-- Given two people, Hanbyul and Hansol, with the following conditions:
    1. The sum of their weights is 88 kg.
    2. Hanbyul weighs 4 kg more than Hansol.
    Prove that Hansol weighs 42 kg. -/
theorem hansol_weight (hanbyul hansol : ℝ) 
    (sum_weight : hanbyul + hansol = 88)
    (weight_diff : hanbyul = hansol + 4) : 
  hansol = 42 := by
  sorry

end NUMINAMATH_CALUDE_hansol_weight_l778_77839


namespace NUMINAMATH_CALUDE_football_season_length_l778_77806

/-- The number of football games in one month -/
def games_per_month : ℝ := 323.0

/-- The total number of football games in the season -/
def total_games : ℕ := 5491

/-- The number of months in the football season -/
def season_months : ℕ := 17

/-- Theorem stating that the number of months in the season is 17 -/
theorem football_season_length :
  (total_games : ℝ) / games_per_month = season_months := by
  sorry

end NUMINAMATH_CALUDE_football_season_length_l778_77806


namespace NUMINAMATH_CALUDE_stan_typing_speed_l778_77846

theorem stan_typing_speed : 
  -- Define constants
  let pages : ℕ := 5
  let words_per_page : ℕ := 400
  let water_per_hour : ℚ := 15
  let water_needed : ℚ := 10
  -- Calculate total words and time
  let total_words : ℕ := pages * words_per_page
  let time_hours : ℚ := water_needed / water_per_hour
  -- Calculate words per minute
  let words_per_minute : ℚ := total_words / (time_hours * 60)
  -- Prove the result
  words_per_minute = 50 := by sorry

end NUMINAMATH_CALUDE_stan_typing_speed_l778_77846


namespace NUMINAMATH_CALUDE_billy_crayons_l778_77842

theorem billy_crayons (initial remaining eaten : ℕ) 
  (h1 : eaten = 52)
  (h2 : remaining = 10)
  (h3 : initial = remaining + eaten) :
  initial = 62 := by
  sorry

end NUMINAMATH_CALUDE_billy_crayons_l778_77842
