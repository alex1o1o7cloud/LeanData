import Mathlib

namespace infinitely_many_pairs_exist_l3404_340418

theorem infinitely_many_pairs_exist : 
  ∀ n : ℕ, ∃ a b : ℕ+, 
    a.val > n ∧ 
    b.val > n ∧ 
    (a.val * b.val) ∣ (a.val^2 + b.val^2 + a.val + b.val + 1) :=
by sorry

end infinitely_many_pairs_exist_l3404_340418


namespace intersection_point_sum_l3404_340477

theorem intersection_point_sum (a b : ℝ) : 
  (∃ x y : ℝ, x = (1/3) * y + a ∧ y = (1/3) * x + b ∧ x = 3 ∧ y = 3) → 
  a + b = 4 := by
  sorry

end intersection_point_sum_l3404_340477


namespace quadratic_equation_properties_l3404_340474

/-- Properties of a quadratic equation -/
theorem quadratic_equation_properties
  (a b c : ℝ) (h_a : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c
  -- Statement 1
  (∀ x, f x = 0 ↔ x = 1 ∨ x = 2) → 2 * a - c = 0 ∧
  -- Statement 2
  (b = 2 * a + c → b^2 - 4 * a * c > 0) ∧
  -- Statement 3
  (∀ m, f m = 0 → b^2 - 4 * a * c = (2 * a * m + b)^2) := by
  sorry


end quadratic_equation_properties_l3404_340474


namespace min_value_reciprocal_sum_l3404_340401

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + b = 1) :
  (1 / a + 1 / b) ≥ 3 + 2 * Real.sqrt 2 :=
by sorry

end min_value_reciprocal_sum_l3404_340401


namespace sphere_volume_is_10936_l3404_340488

/-- The volume of a small hemisphere container in liters -/
def small_hemisphere_volume : ℝ := 4

/-- The number of small hemisphere containers required -/
def num_hemispheres : ℕ := 2734

/-- The total volume of water in the sphere container in liters -/
def sphere_volume : ℝ := small_hemisphere_volume * num_hemispheres

/-- Theorem stating that the total volume of water in the sphere container is 10936 liters -/
theorem sphere_volume_is_10936 : sphere_volume = 10936 := by
  sorry

end sphere_volume_is_10936_l3404_340488


namespace price_of_49_dozens_l3404_340441

/-- Calculates the price of a given number of dozens of apples at a new price -/
def price_of_apples (initial_price : ℝ) (new_price : ℝ) (dozens : ℕ) : ℝ :=
  dozens * new_price

/-- Theorem: The price of 49 dozens of apples at the new price is 49 times the new price -/
theorem price_of_49_dozens 
  (initial_price : ℝ) 
  (new_price : ℝ) 
  (h1 : initial_price = 1517.25)
  (h2 : new_price = 2499) :
  price_of_apples initial_price new_price 49 = 49 * new_price :=
by sorry

end price_of_49_dozens_l3404_340441


namespace gas_price_increase_l3404_340433

/-- Given two successive price increases in gas, where the second increase is 20%,
    and a driver needs to reduce gas consumption by 35.89743589743589% to keep
    expenditure constant, prove that the first price increase was approximately 30%. -/
theorem gas_price_increase (initial_price : ℝ) (initial_consumption : ℝ) :
  initial_price > 0 →
  initial_consumption > 0 →
  ∃ (first_increase : ℝ),
    (initial_price * initial_consumption =
      initial_price * (1 + first_increase / 100) * 1.20 * initial_consumption * (1 - 35.89743589743589 / 100)) ∧
    (abs (first_increase - 30) < 0.00001) := by
  sorry

end gas_price_increase_l3404_340433


namespace intersection_with_complement_of_reals_l3404_340476

open Set

theorem intersection_with_complement_of_reals (A B : Set ℝ) 
  (hA : A = {x : ℝ | x > 0}) 
  (hB : B = {x : ℝ | x > 1}) : 
  A ∩ (Set.univ \ B) = {x : ℝ | 0 < x ∧ x ≤ 1} := by
  sorry

end intersection_with_complement_of_reals_l3404_340476


namespace twelve_months_probability_l3404_340495

/-- Represents the card game "Twelve Months" -/
structure TwelveMonths where
  /-- Number of columns -/
  n : Nat
  /-- Number of cards per column -/
  m : Nat
  /-- Total number of cards -/
  total_cards : Nat
  /-- Condition: total cards equals m * n -/
  h_total : total_cards = m * n

/-- The probability of all cards being flipped in the "Twelve Months" game -/
def probability_all_flipped (game : TwelveMonths) : ℚ :=
  1 / game.n

/-- Theorem stating the probability of all cards being flipped in the "Twelve Months" game -/
theorem twelve_months_probability (game : TwelveMonths) 
  (h_columns : game.n = 12) 
  (h_cards_per_column : game.m = 4) : 
  probability_all_flipped game = 1 / 12 := by
  sorry

#eval probability_all_flipped ⟨12, 4, 48, rfl⟩

end twelve_months_probability_l3404_340495


namespace no_real_zeros_l3404_340400

theorem no_real_zeros (x : ℝ) : x^6 - x^5 + x^4 - x^3 + x^2 - x + 3/4 ≥ 3/8 := by
  sorry

end no_real_zeros_l3404_340400


namespace triangle_side_expression_simplification_l3404_340460

theorem triangle_side_expression_simplification
  (a b c : ℝ) (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  |a - b - c| + |b - c + a| + |c - a - b| = a + 3*b - c :=
by sorry

end triangle_side_expression_simplification_l3404_340460


namespace range_of_a_l3404_340490

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Ioo 0 2, (a - a^2) * (x^2 + 1) + x ≤ 0) ↔ 
  a ∈ Set.Iic ((1 - Real.sqrt 3) / 2) ∪ Set.Ici ((1 + Real.sqrt 3) / 2) :=
sorry

end range_of_a_l3404_340490


namespace train_length_l3404_340434

/-- The length of a train given its speed and time to pass an observer -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 144 → time_s = 6 → speed_kmh * (1000 / 3600) * time_s = 240 := by
  sorry

end train_length_l3404_340434


namespace smallest_integer_with_remainders_l3404_340457

theorem smallest_integer_with_remainders : ∃ n : ℕ, 
  n > 0 ∧ 
  n % 2 = 1 ∧ 
  n % 3 = 2 ∧ 
  n % 4 = 3 ∧ 
  n % 10 = 9 ∧ 
  ∀ m : ℕ, m > 0 ∧ m % 2 = 1 ∧ m % 3 = 2 ∧ m % 4 = 3 ∧ m % 10 = 9 → n ≤ m :=
by
  sorry

#eval 59 % 2  -- Expected output: 1
#eval 59 % 3  -- Expected output: 2
#eval 59 % 4  -- Expected output: 3
#eval 59 % 10 -- Expected output: 9

end smallest_integer_with_remainders_l3404_340457


namespace barycentric_coords_proportional_to_areas_l3404_340463

-- Define a triangle ABC
variable (A B C : ℝ × ℝ)

-- Define a point P inside the triangle
variable (P : ℝ × ℝ)

-- Define the area function
noncomputable def area (X Y Z : ℝ × ℝ) : ℝ := sorry

-- Define the barycentric coordinates
def barycentric_coords (P A B C : ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

-- State the theorem
theorem barycentric_coords_proportional_to_areas :
  ∃ (k : ℝ), k ≠ 0 ∧ 
    barycentric_coords P A B C = 
      (k * area P B C, k * area P C A, k * area P A B) := by sorry

end barycentric_coords_proportional_to_areas_l3404_340463


namespace unique_root_formula_l3404_340485

/-- A quadratic polynomial with exactly one root -/
class UniqueRootQuadratic (g : ℝ → ℝ) : Prop where
  is_quadratic : ∃ a b c : ℝ, ∀ x, g x = a * x^2 + b * x + c
  unique_root : ∃! x : ℝ, g x = 0

/-- The property that g(ax + b) + g(cx + d) has exactly one root -/
def has_unique_combined_root (g : ℝ → ℝ) (a b c d : ℝ) : Prop :=
  ∃! x : ℝ, g (a * x + b) + g (c * x + d) = 0

theorem unique_root_formula 
  (g : ℝ → ℝ) (a b c d : ℝ) 
  [UniqueRootQuadratic g] 
  (h₁ : has_unique_combined_root g a b c d) 
  (h₂ : a ≠ c) : 
  ∃ x₀ : ℝ, (∀ x, g x = 0 ↔ x = x₀) ∧ x₀ = (a * d - b * c) / (a - c) := by
  sorry

end unique_root_formula_l3404_340485


namespace fraction_equivalence_l3404_340491

theorem fraction_equivalence 
  (a b d k : ℝ) 
  (h1 : d ≠ 0) 
  (h2 : k ≠ 0) : 
  (∀ x, (a * (k * x) + b) / (a * (k * x) + d) = (b * (k * x)) / (d * (k * x))) ↔ b = d :=
sorry

end fraction_equivalence_l3404_340491


namespace polynomial_divisibility_l3404_340435

-- Define the polynomial
def P (x : ℝ) : ℝ := 4*x^4 + 4*x^3 - 11*x^2 - 6*x + 9

-- Define the divisor
def D (x : ℝ) : ℝ := (x - 1)^2

-- Define the quotient
def Q (x : ℝ) : ℝ := 4*x^2 + 12*x + 9

-- Theorem statement
theorem polynomial_divisibility :
  ∀ x : ℝ, P x = D x * Q x :=
sorry

end polynomial_divisibility_l3404_340435


namespace egg_distribution_l3404_340427

def crate_capacity : ℕ := 18
def abigail_eggs : ℕ := 58
def beatrice_eggs : ℕ := 76
def carson_eggs : ℕ := 27

def total_eggs : ℕ := abigail_eggs + beatrice_eggs + carson_eggs
def full_crates : ℕ := total_eggs / crate_capacity
def remaining_eggs : ℕ := total_eggs % crate_capacity

theorem egg_distribution :
  (remaining_eggs / 3 = 5) ∧
  (remaining_eggs % 3 = 2) ∧
  (abigail_eggs + 6 + beatrice_eggs + 6 + carson_eggs + 5 = total_eggs - full_crates * crate_capacity) := by
  sorry

end egg_distribution_l3404_340427


namespace two_digit_numbers_problem_l3404_340486

theorem two_digit_numbers_problem :
  ∃ (x y : ℕ), 10 ≤ x ∧ x < y ∧ y < 100 ∧
  (1000 * y + x) % (100 * x + y) = 590 ∧
  (1000 * y + x) / (100 * x + y) = 2 ∧
  2 * y + 3 * x = 72 := by
  sorry

end two_digit_numbers_problem_l3404_340486


namespace total_earnings_theorem_l3404_340465

/-- Represents the different car models --/
inductive CarModel
| A
| B
| C
| D

/-- Represents the different services offered --/
inductive Service
| OilChange
| Repair
| CarWash
| TireRotation

/-- Returns the price of a service for a given car model --/
def servicePrice (model : CarModel) (service : Service) : ℕ :=
  match model, service with
  | CarModel.A, Service.OilChange => 20
  | CarModel.A, Service.Repair => 30
  | CarModel.A, Service.CarWash => 5
  | CarModel.A, Service.TireRotation => 15
  | CarModel.B, Service.OilChange => 25
  | CarModel.B, Service.Repair => 40
  | CarModel.B, Service.CarWash => 8
  | CarModel.B, Service.TireRotation => 18
  | CarModel.C, Service.OilChange => 30
  | CarModel.C, Service.Repair => 50
  | CarModel.C, Service.CarWash => 10
  | CarModel.C, Service.TireRotation => 20
  | CarModel.D, Service.OilChange => 35
  | CarModel.D, Service.Repair => 60
  | CarModel.D, Service.CarWash => 12
  | CarModel.D, Service.TireRotation => 22

/-- Applies discount if the number of services is 3 or more --/
def applyDiscount (total : ℕ) (numServices : ℕ) : ℕ :=
  if numServices ≥ 3 then
    total - (total * 10 / 100)
  else
    total

/-- Calculates the total price for a car model with given services --/
def totalPrice (model : CarModel) (services : List Service) : ℕ :=
  let total := services.foldl (fun acc service => acc + servicePrice model service) 0
  applyDiscount total services.length

/-- The main theorem to prove --/
theorem total_earnings_theorem :
  let modelA_services := [Service.OilChange, Service.Repair, Service.CarWash]
  let modelB_services := [Service.OilChange, Service.Repair, Service.CarWash, Service.TireRotation]
  let modelC_services := [Service.OilChange, Service.Repair, Service.TireRotation, Service.CarWash]
  let modelD_services := [Service.OilChange, Service.Repair, Service.TireRotation]
  
  5 * (totalPrice CarModel.A modelA_services) +
  3 * (totalPrice CarModel.B modelB_services) +
  2 * (totalPrice CarModel.C modelC_services) +
  4 * (totalPrice CarModel.D modelD_services) = 111240 :=
by sorry


end total_earnings_theorem_l3404_340465


namespace geometric_sequence_sum_l3404_340403

/-- Given a geometric sequence {a_n} where a_1 = 3 and a_4 = 24, 
    prove that a_3 + a_4 + a_5 = 84 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q) →  -- geometric sequence condition
  a 1 = 3 →                                  -- a_1 = 3
  a 4 = 24 →                                 -- a_4 = 24
  a 3 + a 4 + a 5 = 84 := by
sorry

end geometric_sequence_sum_l3404_340403


namespace maria_hardcover_volumes_l3404_340402

/-- Proof that Maria bought 9 hardcover volumes -/
theorem maria_hardcover_volumes :
  ∀ (h p : ℕ), -- h: number of hardcover volumes, p: number of paperback volumes
  h + p = 15 → -- total number of volumes
  10 * p + 30 * h = 330 → -- total cost equation
  h = 9 := by
sorry

end maria_hardcover_volumes_l3404_340402


namespace box_office_growth_l3404_340471

theorem box_office_growth (x : ℝ) : 
  (∃ (initial final : ℝ), 
    initial = 2 ∧ 
    final = 4 ∧ 
    final = initial * (1 + x)^2) ↔ 
  2 * (1 + x)^2 = 4 := by sorry

end box_office_growth_l3404_340471


namespace james_purchase_cost_l3404_340449

/-- Calculates the total cost of James' purchase --/
def totalCost (bedFramePrice bedPrice bedsideTablePrice bedFrameDiscount bedDiscount bedsideTableDiscount salesTax : ℝ) : ℝ :=
  let discountedBedFramePrice := bedFramePrice * (1 - bedFrameDiscount)
  let discountedBedPrice := bedPrice * (1 - bedDiscount)
  let discountedBedsideTablePrice := bedsideTablePrice * (1 - bedsideTableDiscount)
  let totalDiscountedPrice := discountedBedFramePrice + discountedBedPrice + discountedBedsideTablePrice
  totalDiscountedPrice * (1 + salesTax)

/-- Theorem stating the total cost of James' purchase --/
theorem james_purchase_cost :
  totalCost 75 750 120 0.20 0.20 0.15 0.085 = 826.77 := by
  sorry

end james_purchase_cost_l3404_340449


namespace triangle_angle_value_l3404_340450

theorem triangle_angle_value (A : ℝ) (h : 0 < A ∧ A < π) : 
  Real.sqrt 2 * Real.sin A = Real.sqrt (3 * Real.cos A) → A = π / 3 := by
  sorry

end triangle_angle_value_l3404_340450


namespace rotate_180_of_A_l3404_340452

/-- Rotate a point 180 degrees about the origin -/
def rotate_180 (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

/-- The original point A -/
def A : ℝ × ℝ := (-3, 2)

theorem rotate_180_of_A :
  rotate_180 A = (3, -2) := by
  sorry

end rotate_180_of_A_l3404_340452


namespace diamond_equation_solution_l3404_340466

/-- A binary operation on nonzero real numbers satisfying certain properties -/
def diamond (a b : ℝ) : ℝ := sorry

/-- The binary operation satisfies a ♢ (b ♢ c) = (a ♢ b) · c -/
axiom diamond_assoc (a b c : ℝ) : a ≠ 0 → b ≠ 0 → c ≠ 0 → diamond a (diamond b c) = (diamond a b) * c

/-- The binary operation satisfies a ♢ a = 1 -/
axiom diamond_self (a : ℝ) : a ≠ 0 → diamond a a = 1

/-- The equation 1008 ♢ (12 ♢ x) = 50 is satisfied when x = 25/42 -/
theorem diamond_equation_solution :
  1008 ≠ 0 → 12 ≠ 0 → (25 : ℝ) / 42 ≠ 0 → diamond 1008 (diamond 12 ((25 : ℝ) / 42)) = 50 := by sorry

end diamond_equation_solution_l3404_340466


namespace square_remainder_mod_five_l3404_340405

theorem square_remainder_mod_five (n : ℤ) (h : n % 5 = 3) : n^2 % 5 = 4 := by
  sorry

end square_remainder_mod_five_l3404_340405


namespace terms_before_ten_l3404_340478

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + d * (n - 1)

theorem terms_before_ten (a₁ : ℤ) (d : ℤ) (n : ℕ) :
  a₁ = 105 ∧ d = -5 →
  arithmetic_sequence a₁ d 20 = 10 ∧
  ∀ k : ℕ, k < 20 → arithmetic_sequence a₁ d k > 10 :=
by sorry

end terms_before_ten_l3404_340478


namespace equal_fish_time_l3404_340483

def brent_fish (n : ℕ) : ℕ := 9 * 4^n

def gretel_fish (n : ℕ) : ℕ := 243 * 3^n

theorem equal_fish_time : ∃ (n : ℕ), n > 0 ∧ brent_fish n = gretel_fish n ∧ n = 8 := by
  sorry

end equal_fish_time_l3404_340483


namespace intersection_of_M_and_N_l3404_340475

-- Define the sets M and N
def M : Set ℕ := {0, 1, 3}
def N : Set ℕ := {0, 1, 7}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {0, 1} := by
  sorry

end intersection_of_M_and_N_l3404_340475


namespace valid_arrangements_count_l3404_340413

/-- The number of ways to arrange n people in a row -/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n people in a row where k specific people sit together -/
def arrangementsWithGrouped (n k : ℕ) : ℕ :=
  Nat.factorial (n - k + 1) * Nat.factorial k

/-- The number of valid arrangements for 8 people where 3 specific people cannot sit together -/
def validArrangements : ℕ :=
  totalArrangements 8 - arrangementsWithGrouped 8 3

theorem valid_arrangements_count :
  validArrangements = 36000 := by sorry

end valid_arrangements_count_l3404_340413


namespace purchase_cost_l3404_340469

/-- The cost of a hamburger in dollars -/
def hamburger_cost : ℕ := 4

/-- The cost of a milkshake in dollars -/
def milkshake_cost : ℕ := 3

/-- The number of hamburgers purchased -/
def num_hamburgers : ℕ := 7

/-- The number of milkshakes purchased -/
def num_milkshakes : ℕ := 6

/-- The total cost of the purchase -/
def total_cost : ℕ := hamburger_cost * num_hamburgers + milkshake_cost * num_milkshakes

theorem purchase_cost : total_cost = 46 := by
  sorry

end purchase_cost_l3404_340469


namespace closest_to_70_l3404_340468

def A : ℚ := 254 / 5
def B : ℚ := 400 / 6
def C : ℚ := 492 / 7

def target : ℚ := 70

theorem closest_to_70 :
  |C - target| ≤ |A - target| ∧ |C - target| ≤ |B - target| :=
sorry

end closest_to_70_l3404_340468


namespace max_value_inequality_l3404_340436

theorem max_value_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 * b^2 * c^2 * (a + b + c)) / ((a + b)^3 * (b + c)^3) ≤ 1/8 := by
  sorry

end max_value_inequality_l3404_340436


namespace females_wearing_glasses_l3404_340479

/-- In a town with a given population, number of males, and percentage of females wearing glasses,
    calculate the number of females wearing glasses. -/
theorem females_wearing_glasses
  (total_population : ℕ)
  (males : ℕ)
  (female_glasses_percentage : ℚ)
  (h1 : total_population = 5000)
  (h2 : males = 2000)
  (h3 : female_glasses_percentage = 30 / 100) :
  (total_population - males) * female_glasses_percentage = 900 := by
sorry

end females_wearing_glasses_l3404_340479


namespace n_times_n_plus_one_divisible_by_two_l3404_340446

theorem n_times_n_plus_one_divisible_by_two (n : ℕ) (h : 1 ≤ n ∧ n ≤ 99) : 
  2 ∣ (n * (n + 1)) := by
  sorry

end n_times_n_plus_one_divisible_by_two_l3404_340446


namespace light_configurations_l3404_340447

/-- The number of rows and columns in the grid -/
def gridSize : Nat := 6

/-- The number of possible states for each switch (on or off) -/
def switchStates : Nat := 2

/-- The total number of different configurations of lights in the grid -/
def totalConfigurations : Nat := (switchStates ^ gridSize - 1) * (switchStates ^ gridSize - 1) + 1

/-- Theorem stating that the number of different configurations of lights is 3970 -/
theorem light_configurations :
  totalConfigurations = 3970 := by
  sorry

end light_configurations_l3404_340447


namespace sqrt_sum_squares_eq_sum_l3404_340431

theorem sqrt_sum_squares_eq_sum (a b c : ℝ) :
  Real.sqrt (a^2 + b^2 + c^2) = a + b + c ↔ a + b + c ≥ 0 ∧ a*b + a*c + b*c = 0 := by sorry

end sqrt_sum_squares_eq_sum_l3404_340431


namespace smallest_divisor_after_323_l3404_340422

theorem smallest_divisor_after_323 (n : ℕ) (h1 : 1000 ≤ n ∧ n < 10000) 
  (h2 : Even n) (h3 : n % 323 = 0) :
  (∃ k : ℕ, k > 323 ∧ n % k = 0 ∧ ∀ m : ℕ, m > 323 ∧ n % m = 0 → k ≤ m) ∧
  (∀ k : ℕ, k > 323 ∧ n % k = 0 ∧ (∀ m : ℕ, m > 323 ∧ n % m = 0 → k ≤ m) → k = 340) :=
by sorry

end smallest_divisor_after_323_l3404_340422


namespace total_songs_two_days_l3404_340437

-- Define the number of songs listened to yesterday
def songs_yesterday : ℕ := 9

-- Define the relationship between yesterday's and today's songs
def song_relationship (x : ℕ) : Prop :=
  songs_yesterday = 2 * (x.sqrt : ℕ) - 5

-- Theorem to prove
theorem total_songs_two_days (x : ℕ) 
  (h : song_relationship x) : songs_yesterday + x = 58 := by
  sorry

end total_songs_two_days_l3404_340437


namespace first_player_winning_strategy_l3404_340430

/-- A game with vectors in a plane -/
structure VectorGame where
  n : ℕ
  vectors : Fin n → ℝ × ℝ

/-- The result of the game -/
inductive GameResult
  | FirstPlayerWins
  | SecondPlayerWins

/-- A strategy for playing the game -/
def Strategy := (n : ℕ) → (remaining : Finset (Fin n)) → Fin n

/-- The game outcome given a strategy for the first player -/
def playGame (game : VectorGame) (strategy : Strategy) : GameResult :=
  sorry

/-- Theorem: The first player has a winning strategy -/
theorem first_player_winning_strategy (game : VectorGame) 
  (h : game.n = 2010) : 
  ∃ (strategy : Strategy), playGame game strategy = GameResult.FirstPlayerWins :=
sorry

end first_player_winning_strategy_l3404_340430


namespace middle_zero_between_zero_and_one_l3404_340451

/-- The cubic function f(x) = x^3 - 4x + a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 4*x + a

/-- Theorem: For 0 < a < 2, if f(x) has three zeros x₁ < x₂ < x₃, then 0 < x₂ < 1 -/
theorem middle_zero_between_zero_and_one (a : ℝ) (x₁ x₂ x₃ : ℝ)
  (ha : 0 < a ∧ a < 2)
  (hzeros : f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0)
  (horder : x₁ < x₂ ∧ x₂ < x₃) :
  0 < x₂ ∧ x₂ < 1 := by
  sorry


end middle_zero_between_zero_and_one_l3404_340451


namespace sally_grew_five_onions_l3404_340438

/-- The number of onions grown by Sally, given the number of onions grown by Sara and Fred, and the total number of onions. -/
def sallys_onions (sara_onions fred_onions total_onions : ℕ) : ℕ :=
  total_onions - (sara_onions + fred_onions)

/-- Theorem stating that Sally grew 5 onions given the conditions in the problem. -/
theorem sally_grew_five_onions :
  sallys_onions 4 9 18 = 5 := by
  sorry

end sally_grew_five_onions_l3404_340438


namespace min_moves_for_target_vectors_l3404_340455

/-- A tuple of 31 integers -/
def Tuple31 := Fin 31 → ℤ

/-- The set of standard basis vectors -/
def StandardBasis : Set Tuple31 :=
  {v | ∃ i, ∀ j, v j = if i = j then 1 else 0}

/-- The set of target vectors -/
def TargetVectors : Set Tuple31 :=
  {v | ∀ i, v i = if i = 0 then 0 else 1} ∪
  {v | ∀ i, v i = if i = 1 then 0 else 1} ∪
  {v | ∀ i, v i = if i = 30 then 0 else 1}

/-- The operation of adding two vectors -/
def AddVectors (v w : Tuple31) : Tuple31 :=
  λ i => v i + w i

/-- The set of vectors that can be generated in n moves -/
def GeneratedVectors (n : ℕ) : Set Tuple31 :=
  sorry

/-- The theorem statement -/
theorem min_moves_for_target_vectors :
  (∃ n, TargetVectors ⊆ GeneratedVectors n) ∧
  (∀ m, m < 87 → ¬(TargetVectors ⊆ GeneratedVectors m)) :=
sorry

end min_moves_for_target_vectors_l3404_340455


namespace sum_equation_l3404_340454

theorem sum_equation (x y z : ℝ) (h1 : y = 3 * x) (h2 : z = 3 * y) : 
  2 * x + 3 * y + z = 20 * x := by
  sorry

end sum_equation_l3404_340454


namespace symmetry_implies_phi_value_l3404_340417

theorem symmetry_implies_phi_value (φ : Real) :
  φ ∈ Set.Icc 0 Real.pi →
  (∀ x : Real, 3 * Real.cos (x + φ) - 1 = 3 * Real.cos ((2 * Real.pi / 3 - x) + φ) - 1) →
  φ = 2 * Real.pi / 3 := by
sorry

end symmetry_implies_phi_value_l3404_340417


namespace die_throws_probability_l3404_340497

/-- The probability of rolling a number greater than 4 on a single die throw -/
def prob_high : ℚ := 1/3

/-- The probability of rolling a number less than or equal to 4 on a single die throw -/
def prob_low : ℚ := 2/3

/-- The probability of getting at least two numbers greater than 4 in two die throws -/
def prob_at_least_two_high : ℚ := prob_high * prob_high + 2 * prob_high * prob_low

theorem die_throws_probability :
  prob_at_least_two_high = 5/9 := by sorry

end die_throws_probability_l3404_340497


namespace function_properties_l3404_340493

def f (x : ℝ) : ℝ := x^3 + x^2 + x + 1

theorem function_properties : 
  f 0 = 1 ∧ 
  f (-1) = 0 ∧ 
  ∃ ε > 0, |f 1 - 4| < ε := by
  sorry

end function_properties_l3404_340493


namespace largest_integer_with_remainder_ninety_eight_satisfies_conditions_ninety_eight_is_largest_l3404_340472

theorem largest_integer_with_remainder : 
  ∀ n : ℕ, n < 100 ∧ n % 6 = 2 → n ≤ 98 :=
by
  sorry

theorem ninety_eight_satisfies_conditions : 
  98 < 100 ∧ 98 % 6 = 2 :=
by
  sorry

theorem ninety_eight_is_largest :
  ∀ n : ℕ, n < 100 ∧ n % 6 = 2 → n ≤ 98 ∧ 
  ∃ m : ℕ, m < 100 ∧ m % 6 = 2 ∧ m = 98 :=
by
  sorry

end largest_integer_with_remainder_ninety_eight_satisfies_conditions_ninety_eight_is_largest_l3404_340472


namespace solution_and_minimum_value_l3404_340484

-- Define the solution set of |2x-3| < x
def solution_set : Set ℝ := {x | 1 < x ∧ x < 3}

-- Define m and n based on the quadratic equation x^2 - mx + n = 0 with roots 1 and 3
def m : ℝ := 4
def n : ℝ := 3

-- Define the constraint for a, b, c
def abc_constraint (a b c : ℝ) : Prop := 0 < a ∧ a < 1 ∧ 0 < b ∧ b < 1 ∧ 0 < c ∧ c < 1 ∧ a * b + b * c + a * c = 1

theorem solution_and_minimum_value :
  (m - n = 1) ∧
  (∀ a b c : ℝ, abc_constraint a b c → a + b + c ≥ Real.sqrt 3) ∧
  (∃ a b c : ℝ, abc_constraint a b c ∧ a + b + c = Real.sqrt 3) := by
  sorry

end solution_and_minimum_value_l3404_340484


namespace train_journey_time_l3404_340462

theorem train_journey_time (S : ℝ) (x : ℝ) (h1 : x > 0) (h2 : S > 0) :
  (S / (2 * x) + S / (2 * 0.75 * x)) - S / x = 0.5 →
  S / x + 0.5 = 3.5 := by
sorry

end train_journey_time_l3404_340462


namespace ribbon_length_difference_equals_side_length_l3404_340414

/-- Represents the dimensions of a box in centimeters -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the ribbon length for the first wrapping method -/
def ribbonLength1 (box : BoxDimensions) (bowLength : ℝ) : ℝ :=
  2 * box.length + 2 * box.width + 4 * box.height + bowLength

/-- Calculates the ribbon length for the second wrapping method -/
def ribbonLength2 (box : BoxDimensions) (bowLength : ℝ) : ℝ :=
  2 * box.length + 4 * box.width + 2 * box.height + bowLength

/-- Theorem stating that the difference in ribbon lengths equals one side of the box -/
theorem ribbon_length_difference_equals_side_length
  (box : BoxDimensions)
  (bowLength : ℝ)
  (h1 : box.length = 22)
  (h2 : box.width = 22)
  (h3 : box.height = 11)
  (h4 : bowLength = 24) :
  ribbonLength2 box bowLength - ribbonLength1 box bowLength = box.length := by
  sorry

end ribbon_length_difference_equals_side_length_l3404_340414


namespace no_solution_for_system_l3404_340481

theorem no_solution_for_system :
  ¬∃ (x y z : ℝ), 
    (Real.sqrt (2 * x^2 + 2) = y - 1) ∧
    (Real.sqrt (2 * y^2 + 2) = z - 1) ∧
    (Real.sqrt (2 * z^2 + 2) = x - 1) := by
  sorry

end no_solution_for_system_l3404_340481


namespace line_point_k_value_l3404_340482

/-- Given a line containing points (0, 10), (5, k), and (25, 0), prove that k = 8 -/
theorem line_point_k_value (k : ℝ) : 
  (∃ (m b : ℝ), m * 0 + b = 10 ∧ m * 5 + b = k ∧ m * 25 + b = 0) → k = 8 := by
  sorry

end line_point_k_value_l3404_340482


namespace smallest_integer_with_remainders_l3404_340464

theorem smallest_integer_with_remainders (n : ℕ) : 
  (n > 1) → 
  (∀ m : ℕ, m > 1 ∧ m < n → ¬(m % 6 = 1 ∧ m % 7 = 1 ∧ m % 9 = 1)) → 
  (n % 6 = 1 ∧ n % 7 = 1 ∧ n % 9 = 1) → 
  (n = 127 ∧ 120 < n ∧ n < 199) :=
by sorry

end smallest_integer_with_remainders_l3404_340464


namespace odometer_puzzle_l3404_340458

/-- Represents the odometer reading as a triple of digits -/
structure OdometerReading where
  hundreds : Nat
  tens : Nat
  ones : Nat

/-- Represents the trip details -/
structure TripDetails where
  initial : OdometerReading
  final : OdometerReading
  duration : Nat  -- in hours
  avgSpeed : Nat  -- in miles per hour

theorem odometer_puzzle (trip : TripDetails) :
  trip.initial.hundreds ≥ 2 ∧
  trip.initial.hundreds + trip.initial.tens + trip.initial.ones = 9 ∧
  trip.avgSpeed = 60 ∧
  trip.initial.hundreds = trip.final.ones ∧
  trip.initial.tens = trip.final.tens ∧
  trip.initial.ones = trip.final.hundreds →
  trip.initial.hundreds^2 + trip.initial.tens^2 + trip.initial.ones^2 = 33 := by
  sorry

end odometer_puzzle_l3404_340458


namespace players_both_games_l3404_340467

/-- Given a group of players with the following properties:
  * There are 400 players in total
  * 350 players play outdoor games
  * 110 players play indoor games
  This theorem proves that the number of players who play both indoor and outdoor games is 60. -/
theorem players_both_games (total : ℕ) (outdoor : ℕ) (indoor : ℕ) 
  (h_total : total = 400)
  (h_outdoor : outdoor = 350)
  (h_indoor : indoor = 110) :
  ∃ (both : ℕ), both = outdoor + indoor - total ∧ both = 60 := by
  sorry

end players_both_games_l3404_340467


namespace smallest_b_undefined_inverse_b_330_satisfies_conditions_smallest_b_is_330_l3404_340407

theorem smallest_b_undefined_inverse (b : ℕ) : b > 0 ∧ 
  (∀ x : ℕ, x * b % 36 ≠ 1) ∧ 
  (∀ y : ℕ, y * b % 55 ≠ 1) → 
  b ≥ 330 := by
  sorry

theorem b_330_satisfies_conditions : 
  (∀ x : ℕ, x * 330 % 36 ≠ 1) ∧ 
  (∀ y : ℕ, y * 330 % 55 ≠ 1) := by
  sorry

theorem smallest_b_is_330 : 
  ∃ b : ℕ, b > 0 ∧ 
  (∀ x : ℕ, x * b % 36 ≠ 1) ∧ 
  (∀ y : ℕ, y * b % 55 ≠ 1) ∧ 
  b = 330 := by
  sorry

end smallest_b_undefined_inverse_b_330_satisfies_conditions_smallest_b_is_330_l3404_340407


namespace class_average_after_exclusion_l3404_340411

theorem class_average_after_exclusion 
  (total_students : ℕ) 
  (total_average : ℚ) 
  (excluded_students : ℕ) 
  (excluded_average : ℚ) : 
  total_students = 10 → 
  total_average = 70 → 
  excluded_students = 5 → 
  excluded_average = 50 → 
  let remaining_students := total_students - excluded_students
  let remaining_total := total_students * total_average - excluded_students * excluded_average
  remaining_total / remaining_students = 90 := by
  sorry

end class_average_after_exclusion_l3404_340411


namespace garden_length_l3404_340425

/-- Proves that a rectangular garden with perimeter 500 m and breadth 100 m has length 150 m -/
theorem garden_length (perimeter : ℝ) (breadth : ℝ) (length : ℝ) : 
  perimeter = 500 → 
  breadth = 100 → 
  perimeter = 2 * (length + breadth) → 
  length = 150 := by
  sorry

end garden_length_l3404_340425


namespace laura_weekly_driving_distance_l3404_340459

/-- Calculates the total miles driven by Laura per week -/
def total_miles_per_week (
  house_to_school_round_trip : ℕ)
  (supermarket_extra_distance : ℕ)
  (school_trips_per_week : ℕ)
  (supermarket_trips_per_week : ℕ) : ℕ :=
  let school_miles := house_to_school_round_trip * school_trips_per_week
  let supermarket_round_trip := house_to_school_round_trip + 2 * supermarket_extra_distance
  let supermarket_miles := supermarket_round_trip * supermarket_trips_per_week
  school_miles + supermarket_miles

/-- Laura's weekly driving distance theorem -/
theorem laura_weekly_driving_distance :
  total_miles_per_week 20 10 5 2 = 180 := by
  sorry

end laura_weekly_driving_distance_l3404_340459


namespace percentage_less_than_l3404_340426

theorem percentage_less_than (x y : ℝ) (h : x = 12 * y) :
  (x - y) / x * 100 = (11 / 12) * 100 :=
sorry

end percentage_less_than_l3404_340426


namespace second_hole_depth_l3404_340415

/-- Represents the depth of a hole dug by workers -/
def hole_depth (workers : ℕ) (hours : ℕ) (rate : ℚ) : ℚ :=
  (workers * hours : ℚ) * rate

theorem second_hole_depth :
  let initial_workers : ℕ := 45
  let initial_hours : ℕ := 8
  let initial_depth : ℚ := 30
  let extra_workers : ℕ := 65
  let second_hours : ℕ := 6
  
  let total_workers : ℕ := initial_workers + extra_workers
  let digging_rate : ℚ := initial_depth / (initial_workers * initial_hours)
  
  hole_depth total_workers second_hours digging_rate = 55 := by
  sorry


end second_hole_depth_l3404_340415


namespace quadratic_distinct_roots_k_nonzero_l3404_340496

/-- Given a quadratic equation kx^2 - 2x + 1/2 = 0, if it has two distinct real roots, then k ≠ 0 -/
theorem quadratic_distinct_roots_k_nonzero (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ k * x^2 - 2*x + 1/2 = 0 ∧ k * y^2 - 2*y + 1/2 = 0) → k ≠ 0 :=
by sorry

end quadratic_distinct_roots_k_nonzero_l3404_340496


namespace quadratic_factorization_l3404_340440

theorem quadratic_factorization (x : ℝ) : 6*x^2 - 24*x + 18 = 6*(x - 1)*(x - 3) := by
  sorry

end quadratic_factorization_l3404_340440


namespace park_is_square_l3404_340421

/-- A shape with a certain number of 90-degree angles -/
structure Shape :=
  (angles : ℕ)

/-- Definition of a square -/
def is_square (s : Shape) : Prop := s.angles = 4

theorem park_is_square (park : Shape) (square_field : Shape)
  (h1 : is_square square_field)
  (h2 : park.angles + square_field.angles = 8) :
  is_square park :=
sorry

end park_is_square_l3404_340421


namespace picnic_bread_slices_l3404_340444

/-- Calculate the total number of bread slices needed for a picnic --/
theorem picnic_bread_slices :
  let total_people : ℕ := 6
  let pb_people : ℕ := 4
  let tuna_people : ℕ := 3
  let turkey_people : ℕ := 2
  let pb_sandwiches_per_person : ℕ := 2
  let tuna_sandwiches_per_person : ℕ := 3
  let turkey_sandwiches_per_person : ℕ := 1
  let pb_slices_per_sandwich : ℕ := 2
  let tuna_slices_per_sandwich : ℕ := 3
  let turkey_slices_per_sandwich : ℚ := 3/2

  let total_pb_sandwiches := pb_people * pb_sandwiches_per_person
  let total_tuna_sandwiches := tuna_people * tuna_sandwiches_per_person
  let total_turkey_sandwiches := turkey_people * turkey_sandwiches_per_person

  let total_pb_slices := total_pb_sandwiches * pb_slices_per_sandwich
  let total_tuna_slices := total_tuna_sandwiches * tuna_slices_per_sandwich
  let total_turkey_slices := (total_turkey_sandwiches : ℚ) * turkey_slices_per_sandwich

  (total_pb_slices : ℚ) + (total_tuna_slices : ℚ) + total_turkey_slices = 46
  := by sorry

end picnic_bread_slices_l3404_340444


namespace min_pieces_same_color_l3404_340420

theorem min_pieces_same_color (total_pieces : ℕ) (pieces_per_color : ℕ) (h1 : total_pieces = 60) (h2 : pieces_per_color = 15) :
  ∃ (min_pieces : ℕ), 
    (∀ (n : ℕ), n < min_pieces → ∃ (selection : Finset ℕ), selection.card = n ∧ 
      ∀ (i j : ℕ), i ∈ selection → j ∈ selection → i ≠ j → (i / pieces_per_color) ≠ (j / pieces_per_color)) ∧
    (∃ (selection : Finset ℕ), selection.card = min_pieces ∧ 
      ∃ (i j : ℕ), i ∈ selection ∧ j ∈ selection ∧ i ≠ j ∧ (i / pieces_per_color) = (j / pieces_per_color)) ∧
    min_pieces = 5 :=
by sorry

end min_pieces_same_color_l3404_340420


namespace min_abs_z_l3404_340428

theorem min_abs_z (z : ℂ) (h : Complex.abs (z - 7) + Complex.abs (z - 6*I) = 15) :
  ∃ (w : ℂ), Complex.abs w = 14/5 ∧ ∀ (v : ℂ), Complex.abs (v - 7) + Complex.abs (v - 6*I) = 15 → Complex.abs v ≥ Complex.abs w :=
sorry

end min_abs_z_l3404_340428


namespace rational_numbers_four_units_from_origin_l3404_340439

theorem rational_numbers_four_units_from_origin :
  {x : ℚ | |x| = 4} = {-4, 4} := by
  sorry

end rational_numbers_four_units_from_origin_l3404_340439


namespace max_books_borrowed_l3404_340494

theorem max_books_borrowed (total_students : ℕ) (zero_books : ℕ) (one_book : ℕ) (two_books : ℕ) 
  (h1 : total_students = 25)
  (h2 : zero_books = 3)
  (h3 : one_book = 11)
  (h4 : two_books = 6)
  (h5 : (total_students : ℚ) * 2 = (zero_books * 0 + one_book * 1 + two_books * 2 + 
    (total_students - zero_books - one_book - two_books) * 3 + 
    (total_students * 2 - zero_books * 0 - one_book * 1 - two_books * 2 - 
    (total_students - zero_books - one_book - two_books) * 3))) :
  ∃ (max_books : ℕ), max_books = 15 ∧ 
    max_books ≤ total_students * 2 - zero_books * 0 - one_book * 1 - two_books * 2 - 
    (total_students - zero_books - one_book - two_books - 1) * 3 :=
by sorry

end max_books_borrowed_l3404_340494


namespace absolute_value_inequality_l3404_340470

theorem absolute_value_inequality (x : ℝ) : 
  ‖‖x - 2‖ - 1‖ ≤ 1 ↔ 0 ≤ x ∧ x ≤ 4 := by sorry

end absolute_value_inequality_l3404_340470


namespace buns_left_is_two_l3404_340432

/-- The number of buns initially on the plate -/
def initial_buns : ℕ := 15

/-- Karlsson takes three times as many buns as Little Boy -/
def karlsson_multiplier : ℕ := 3

/-- Bimbo takes three times fewer buns than Little Boy -/
def bimbo_divisor : ℕ := 3

/-- The number of buns Bimbo takes -/
def bimbo_buns : ℕ := 1

/-- The number of buns Little Boy takes -/
def little_boy_buns : ℕ := bimbo_buns * bimbo_divisor

/-- The number of buns Karlsson takes -/
def karlsson_buns : ℕ := little_boy_buns * karlsson_multiplier

/-- The total number of buns taken -/
def total_taken : ℕ := bimbo_buns + little_boy_buns + karlsson_buns

/-- The number of buns left on the plate -/
def buns_left : ℕ := initial_buns - total_taken

theorem buns_left_is_two : buns_left = 2 := by
  sorry

end buns_left_is_two_l3404_340432


namespace smallest_four_digit_multiple_of_18_l3404_340445

theorem smallest_four_digit_multiple_of_18 :
  ∀ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ 18 ∣ n → n ≥ 1008 :=
by
  sorry

end smallest_four_digit_multiple_of_18_l3404_340445


namespace polynomial_division_theorem_l3404_340409

theorem polynomial_division_theorem (a b : ℝ) : 
  (∃ (P : ℝ → ℝ), (fun X => a * X^4 + b * X^3 + 1) = fun X => (X - 1)^2 * P X) → 
  a = 3 ∧ b = -4 := by
sorry

end polynomial_division_theorem_l3404_340409


namespace equation_solution_l3404_340453

/-- Given the equation P = s / (1 + k + m)^n, prove that n = log(s/P) / log(1 + k + m) -/
theorem equation_solution (P s k m n : ℝ) (h : P = s / (1 + k + m)^n) :
  n = Real.log (s / P) / Real.log (1 + k + m) :=
by sorry

end equation_solution_l3404_340453


namespace sum_of_first_six_primes_mod_seventh_prime_l3404_340416

def first_six_primes : List Nat := [2, 3, 5, 7, 11, 13]
def seventh_prime : Nat := 17

theorem sum_of_first_six_primes_mod_seventh_prime : 
  (first_six_primes.sum) % seventh_prime = 7 := by
  sorry

end sum_of_first_six_primes_mod_seventh_prime_l3404_340416


namespace birds_in_tree_l3404_340419

/-- Given 179 initial birds in a tree and 38 additional birds joining them,
    the total number of birds in the tree is 217. -/
theorem birds_in_tree (initial_birds additional_birds : ℕ) 
  (h1 : initial_birds = 179)
  (h2 : additional_birds = 38) :
  initial_birds + additional_birds = 217 := by
  sorry

end birds_in_tree_l3404_340419


namespace smallest_non_square_units_digit_l3404_340408

def is_square_units_digit (d : ℕ) : Prop :=
  ∃ n : ℕ, n^2 % 10 = d

theorem smallest_non_square_units_digit :
  (∀ d < 2, is_square_units_digit d) ∧
  ¬(is_square_units_digit 2) ∧
  (∀ d ≥ 2, ¬(is_square_units_digit d) → d ≥ 2) :=
sorry

end smallest_non_square_units_digit_l3404_340408


namespace finite_solutions_egyptian_fraction_l3404_340443

theorem finite_solutions_egyptian_fraction :
  (∃ (S : Set (ℕ+ × ℕ+ × ℕ+)), Finite S ∧
    ∀ (a b c : ℕ+), (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c = (1 : ℚ) / 1983 ↔ (a, b, c) ∈ S) :=
by sorry

end finite_solutions_egyptian_fraction_l3404_340443


namespace one_millionth_digit_of_3_div_41_l3404_340498

/-- The decimal representation of a rational number -/
def decimal_representation (q : ℚ) : ℕ → ℕ := sorry

/-- The nth digit after the decimal point in the decimal representation of q -/
def nth_digit_after_decimal (q : ℚ) (n : ℕ) : ℕ := 
  decimal_representation q n

/-- The one-millionth digit after the decimal point in 3/41 is 7 -/
theorem one_millionth_digit_of_3_div_41 : 
  nth_digit_after_decimal (3/41) 1000000 = 7 := by sorry

end one_millionth_digit_of_3_div_41_l3404_340498


namespace combined_pencil_length_l3404_340480

-- Define the length of a pencil in cubes
def pencil_length : ℕ := 12

-- Define the number of pencils
def num_pencils : ℕ := 2

-- Theorem: The combined length of two pencils is 24 cubes
theorem combined_pencil_length :
  num_pencils * pencil_length = 24 := by
  sorry

end combined_pencil_length_l3404_340480


namespace intersection_of_A_and_B_l3404_340461

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 3}

theorem intersection_of_A_and_B :
  A ∩ B = {x | 0 ≤ x ∧ x ≤ 2} := by sorry

end intersection_of_A_and_B_l3404_340461


namespace james_money_calculation_l3404_340442

/-- Given 3 bills of $20 each and $75 already in a wallet, prove that the total amount is $135 -/
theorem james_money_calculation :
  let bills_count : ℕ := 3
  let bill_value : ℕ := 20
  let initial_wallet_amount : ℕ := 75
  bills_count * bill_value + initial_wallet_amount = 135 :=
by sorry

end james_money_calculation_l3404_340442


namespace gum_pieces_per_package_l3404_340410

theorem gum_pieces_per_package (total_packages : ℕ) (total_pieces : ℕ) 
  (h1 : total_packages = 27) 
  (h2 : total_pieces = 486) : 
  total_pieces / total_packages = 18 := by
  sorry

end gum_pieces_per_package_l3404_340410


namespace bottle_cost_difference_l3404_340499

/-- Represents a bottle of capsules -/
structure Bottle where
  capsules : ℕ
  cost : ℚ

/-- Calculate the cost per capsule for a given bottle -/
def costPerCapsule (b : Bottle) : ℚ :=
  b.cost / b.capsules

/-- The difference in cost per capsule between two bottles -/
def costDifference (b1 b2 : Bottle) : ℚ :=
  costPerCapsule b1 - costPerCapsule b2

theorem bottle_cost_difference :
  let bottleR : Bottle := { capsules := 250, cost := 25/4 }
  let bottleT : Bottle := { capsules := 100, cost := 3 }
  costDifference bottleT bottleR = 1/200 := by
sorry

end bottle_cost_difference_l3404_340499


namespace andreas_living_room_area_l3404_340423

theorem andreas_living_room_area :
  ∀ (floor_area carpet_area : ℝ),
    carpet_area = 4 * 9 →
    0.75 * floor_area = carpet_area →
    floor_area = 48 := by
  sorry

end andreas_living_room_area_l3404_340423


namespace percentage_problem_l3404_340412

theorem percentage_problem (p : ℝ) (x : ℝ) 
  (h1 : (p / 100) * x = 300)
  (h2 : (120 / 100) * x = 1800) : p = 20 := by
  sorry

end percentage_problem_l3404_340412


namespace inserted_numbers_sum_l3404_340429

theorem inserted_numbers_sum : ∃! (a b : ℝ), 
  0 < a ∧ 0 < b ∧ 
  4 < a ∧ a < b ∧ b < 16 ∧ 
  (∃ r : ℝ, 0 < r ∧ a = 4 * r ∧ b = 4 * r^2) ∧
  (∃ d : ℝ, b = a + d ∧ 16 = b + d) ∧
  a + b = 24 := by
sorry

end inserted_numbers_sum_l3404_340429


namespace nails_in_toolshed_l3404_340473

theorem nails_in_toolshed (initial_nails : ℕ) (nails_to_buy : ℕ) (total_nails : ℕ) :
  initial_nails = 247 →
  nails_to_buy = 109 →
  total_nails = 500 →
  total_nails = initial_nails + nails_to_buy + (total_nails - initial_nails - nails_to_buy) →
  total_nails - initial_nails - nails_to_buy = 144 :=
by sorry

end nails_in_toolshed_l3404_340473


namespace movie_only_attendance_l3404_340456

/-- Represents the number of students attending different activities --/
structure ActivityAttendance where
  total : ℕ
  picnic : ℕ
  games : ℕ
  movie_and_picnic : ℕ
  movie_and_games : ℕ
  picnic_and_games : ℕ
  all_activities : ℕ

/-- The given conditions for the problem --/
def given_conditions : ActivityAttendance :=
  { total := 31
  , picnic := 20
  , games := 5
  , movie_and_picnic := 4
  , movie_and_games := 2
  , picnic_and_games := 0
  , all_activities := 2
  }

/-- Theorem stating that the number of students meeting for the movie only is 12 --/
theorem movie_only_attendance (conditions : ActivityAttendance) : 
  conditions.total - (conditions.picnic + conditions.games - conditions.movie_and_picnic - conditions.movie_and_games - conditions.picnic_and_games + conditions.all_activities) = 12 :=
by sorry

end movie_only_attendance_l3404_340456


namespace x_fifth_minus_ten_x_equals_213_l3404_340448

theorem x_fifth_minus_ten_x_equals_213 (x : ℝ) (h : x = 3) : x^5 - 10*x = 213 := by
  sorry

end x_fifth_minus_ten_x_equals_213_l3404_340448


namespace quadratic_function_value_l3404_340487

/-- A quadratic function with specified properties -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_function_value (a b c : ℝ) :
  (∀ x, QuadraticFunction a b c x ≤ 75) ∧ 
  (QuadraticFunction a b c (-3) = 0) ∧ 
  (QuadraticFunction a b c 3 = 0) →
  QuadraticFunction a b c 2 = 125/3 := by
  sorry

end quadratic_function_value_l3404_340487


namespace bowling_tournament_prize_orders_l3404_340424

/-- Represents a bowling tournament with 6 players and a specific playoff structure. -/
structure BowlingTournament :=
  (num_players : Nat)
  (playoff_structure : List (Nat × Nat))

/-- Calculates the number of possible prize order combinations in a bowling tournament. -/
def possiblePrizeOrders (tournament : BowlingTournament) : Nat :=
  2^(tournament.num_players - 1)

/-- Theorem stating that the number of possible prize order combinations
    in the given 6-player bowling tournament is 32. -/
theorem bowling_tournament_prize_orders :
  ∃ (t : BowlingTournament),
    t.num_players = 6 ∧
    t.playoff_structure = [(6, 5), (4, 0), (3, 0), (2, 0), (1, 0)] ∧
    possiblePrizeOrders t = 32 :=
by
  sorry


end bowling_tournament_prize_orders_l3404_340424


namespace exists_number_with_special_quotient_l3404_340492

-- Define a function to check if a number contains all digits from 1 to 8 exactly once
def containsAllDigitsOnce (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ Finset.range 8 → (∃! i : ℕ, i < (Nat.digits 10 n).length ∧ (Nat.digits 10 n).get ⟨i, by sorry⟩ = d + 1)

-- Theorem statement
theorem exists_number_with_special_quotient :
  ∃ N d : ℕ, N > 0 ∧ d > 0 ∧ containsAllDigitsOnce (N / d) :=
sorry

end exists_number_with_special_quotient_l3404_340492


namespace mans_walking_speed_l3404_340404

-- Define the given conditions
def walking_time : ℝ := 8
def running_time : ℝ := 2
def running_speed : ℝ := 36

-- Define the walking speed as a variable
def walking_speed : ℝ := sorry

-- Theorem statement
theorem mans_walking_speed :
  walking_speed * walking_time = running_speed * running_time →
  walking_speed = 9 := by
  sorry

end mans_walking_speed_l3404_340404


namespace cos_sum_of_complex_on_unit_circle_l3404_340406

/-- Given complex numbers on the unit circle represented by their real and imaginary parts,
    prove that the cosine of the sum of their arguments is as specified. -/
theorem cos_sum_of_complex_on_unit_circle
  (γ δ : ℝ)
  (h1 : Complex.exp (Complex.I * γ) = Complex.ofReal (8/17) + Complex.I * (15/17))
  (h2 : Complex.exp (Complex.I * δ) = Complex.ofReal (3/5) - Complex.I * (4/5)) :
  Real.cos (γ + δ) = 84/85 := by
  sorry

end cos_sum_of_complex_on_unit_circle_l3404_340406


namespace number_divided_by_6_multiplied_by_12_equals_13_l3404_340489

theorem number_divided_by_6_multiplied_by_12_equals_13 : ∃ x : ℚ, (x / 6) * 12 = 13 ∧ x = 13/2 := by
  sorry

end number_divided_by_6_multiplied_by_12_equals_13_l3404_340489
