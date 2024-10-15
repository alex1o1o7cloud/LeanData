import Mathlib

namespace NUMINAMATH_CALUDE_goldbach_negation_equivalence_l3564_356424

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def goldbach_conjecture : Prop :=
  ∀ n : ℕ, n > 2 → n % 2 = 0 → ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ n = p + q

theorem goldbach_negation_equivalence :
  (¬ goldbach_conjecture) ↔
  (∃ n : ℕ, n > 2 ∧ n % 2 = 0 ∧ ∀ p q : ℕ, is_prime p → is_prime q → n ≠ p + q) :=
sorry

end NUMINAMATH_CALUDE_goldbach_negation_equivalence_l3564_356424


namespace NUMINAMATH_CALUDE_memory_card_picture_size_l3564_356454

/-- Represents a memory card with a given capacity and picture storage capabilities. -/
structure MemoryCard where
  capacity : ℕ  -- Total capacity in megabytes
  large_pics : ℕ  -- Number of large pictures it can hold
  small_pics : ℕ  -- Number of small pictures it can hold
  small_pic_size : ℕ  -- Size of small pictures in megabytes

/-- Calculates the size of pictures when the card is filled with large pictures. -/
def large_pic_size (card : MemoryCard) : ℕ :=
  card.capacity / card.large_pics

theorem memory_card_picture_size (card : MemoryCard) 
  (h1 : card.small_pics = 3000)
  (h2 : card.large_pics = 4000)
  (h3 : large_pic_size card = 6) :
  card.small_pic_size = 8 := by
  sorry

#check memory_card_picture_size

end NUMINAMATH_CALUDE_memory_card_picture_size_l3564_356454


namespace NUMINAMATH_CALUDE_tangent_line_at_one_l3564_356405

noncomputable def f (x : ℝ) := x^4 - 2*x^3

theorem tangent_line_at_one (x : ℝ) : 
  let p := (1, f 1)
  let m := deriv f 1
  (fun x => m * (x - p.1) + p.2) = (fun x => -2 * x + 1) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_l3564_356405


namespace NUMINAMATH_CALUDE_employee_salary_proof_l3564_356421

/-- Given two employees with a total weekly salary and a salary ratio, prove the salary of one employee. -/
theorem employee_salary_proof (total : ℚ) (ratio : ℚ) (n_salary : ℚ) : 
  total = 583 →
  ratio = 1.2 →
  n_salary + ratio * n_salary = total →
  n_salary = 265 := by
sorry

end NUMINAMATH_CALUDE_employee_salary_proof_l3564_356421


namespace NUMINAMATH_CALUDE_dodecagon_enclosure_l3564_356481

/-- The number of sides of the central polygon -/
def m : ℕ := 12

/-- The number of smaller polygons enclosing the central polygon -/
def num_enclosing : ℕ := 12

/-- The number of smaller polygons meeting at each vertex of the central polygon -/
def num_meeting : ℕ := 3

/-- The number of sides of each smaller polygon -/
def n : ℕ := 12

/-- The interior angle of a regular polygon with m sides -/
def interior_angle (m : ℕ) : ℚ := (m - 2) * 180 / m

/-- The exterior angle of a regular polygon with m sides -/
def exterior_angle (m : ℕ) : ℚ := 180 - interior_angle m

/-- Theorem stating that n must be 12 for the given configuration -/
theorem dodecagon_enclosure :
  exterior_angle m = num_meeting * (exterior_angle n / num_meeting) :=
sorry

end NUMINAMATH_CALUDE_dodecagon_enclosure_l3564_356481


namespace NUMINAMATH_CALUDE_lemonade_sales_calculation_l3564_356450

/-- Calculates the total sales for lemonade glasses sold over two days -/
theorem lemonade_sales_calculation (price_per_glass : ℚ) (saturday_sales sunday_sales : ℕ) :
  price_per_glass = 25 / 100 →
  saturday_sales = 41 →
  sunday_sales = 53 →
  (saturday_sales + sunday_sales : ℚ) * price_per_glass = 2350 / 100 := by
  sorry

#eval (41 + 53 : ℚ) * (25 / 100) -- Optional: to verify the result

end NUMINAMATH_CALUDE_lemonade_sales_calculation_l3564_356450


namespace NUMINAMATH_CALUDE_strawberry_harvest_l3564_356445

theorem strawberry_harvest (base height : ℝ) (plants_per_sqft : ℕ) (strawberries_per_plant : ℕ) :
  base = 10 →
  height = 12 →
  plants_per_sqft = 5 →
  strawberries_per_plant = 8 →
  (1/2 * base * height * plants_per_sqft * strawberries_per_plant : ℝ) = 2400 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_harvest_l3564_356445


namespace NUMINAMATH_CALUDE_mrs_heine_dogs_l3564_356406

/-- Given that Mrs. Heine buys 3 heart biscuits for each dog and needs to buy 6 biscuits in total,
    prove that she has 2 dogs. -/
theorem mrs_heine_dogs :
  ∀ (total_biscuits biscuits_per_dog : ℕ),
    total_biscuits = 6 →
    biscuits_per_dog = 3 →
    total_biscuits / biscuits_per_dog = 2 :=
by sorry

end NUMINAMATH_CALUDE_mrs_heine_dogs_l3564_356406


namespace NUMINAMATH_CALUDE_murtha_pebbles_after_20_days_l3564_356403

def pebbles_collected (n : ℕ) : ℕ := n + 1

def pebbles_given_away (n : ℕ) : ℕ := if n % 5 = 0 then 3 else 0

def total_pebbles (days : ℕ) : ℕ :=
  2 + (Finset.range (days - 1)).sum pebbles_collected - (Finset.range days).sum pebbles_given_away

theorem murtha_pebbles_after_20_days :
  total_pebbles 20 = 218 := by sorry

end NUMINAMATH_CALUDE_murtha_pebbles_after_20_days_l3564_356403


namespace NUMINAMATH_CALUDE_remainder_of_3_power_20_l3564_356420

theorem remainder_of_3_power_20 (a : ℕ) : 
  a = (1 + 2)^20 → a % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_3_power_20_l3564_356420


namespace NUMINAMATH_CALUDE_unique_valid_prism_l3564_356498

/-- A right rectangular prism with integer side lengths -/
structure RectPrism where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  h1 : a ≤ b
  h2 : b ≤ c

/-- Predicate for a valid division of a prism -/
def validDivision (p : RectPrism) : Prop :=
  ∃ (k : ℚ), 0 < k ∧ k < 1 ∧
  ((k * p.a.val = p.a.val ∧ k * p.b.val = p.a.val) ∨
   (k * p.b.val = p.b.val ∧ k * p.c.val = p.b.val) ∨
   (k * p.c.val = p.c.val ∧ k * p.a.val = p.c.val))

theorem unique_valid_prism :
  ∃! (p : RectPrism), p.b = 101 ∧ validDivision p :=
sorry

end NUMINAMATH_CALUDE_unique_valid_prism_l3564_356498


namespace NUMINAMATH_CALUDE_polynomial_expansion_equality_l3564_356411

theorem polynomial_expansion_equality (x y : ℤ) :
  5 * (x + 4) * (x + 7) * (x + 11) * (x + 13) - 4 * (x + y)^2 =
  5*x^4 + 35*x^3 + 960*x^2 + 1649*x + 4000 - 8*x*y - 4*y^2 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_expansion_equality_l3564_356411


namespace NUMINAMATH_CALUDE_lap_time_improvement_is_two_thirds_l3564_356463

/-- Represents running data with number of laps and total time in minutes -/
structure RunningData where
  laps : ℕ
  time : ℚ

/-- Calculates the lap time in minutes for given running data -/
def lapTime (data : RunningData) : ℚ :=
  data.time / data.laps

/-- The initial running data -/
def initialData : RunningData :=
  { laps := 15, time := 45 }

/-- The final running data after training -/
def finalData : RunningData :=
  { laps := 18, time := 42 }

/-- The improvement in lap time -/
def lapTimeImprovement : ℚ :=
  lapTime initialData - lapTime finalData

theorem lap_time_improvement_is_two_thirds :
  lapTimeImprovement = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_lap_time_improvement_is_two_thirds_l3564_356463


namespace NUMINAMATH_CALUDE_quadratic_minimum_l3564_356492

theorem quadratic_minimum (p q : ℝ) : 
  (∃ (y : ℝ → ℝ), (∀ x, y x = x^2 + p*x + q) ∧ 
   (∃ x₀, ∀ x, y x₀ ≤ y x) ∧ 
   (∃ x₁, y x₁ = 0)) →
  q = p^2 / 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l3564_356492


namespace NUMINAMATH_CALUDE_lid_circumference_l3564_356412

theorem lid_circumference (diameter : ℝ) (h : diameter = 2) :
  Real.pi * diameter = 2 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_lid_circumference_l3564_356412


namespace NUMINAMATH_CALUDE_train_speed_calculation_l3564_356460

/-- Calculates the speed of a train given its length, time to cross a man, and the man's speed -/
theorem train_speed_calculation (train_length : ℝ) (crossing_time : ℝ) (man_speed_kmh : ℝ) :
  train_length = 120 →
  crossing_time = 6 →
  man_speed_kmh = 5 →
  ∃ (train_speed_kmh : ℝ), 
    (train_speed_kmh ≥ 66.9) ∧ 
    (train_speed_kmh ≤ 67.1) ∧
    (train_speed_kmh * 1000 / 3600 + man_speed_kmh * 1000 / 3600) * crossing_time = train_length :=
by sorry


end NUMINAMATH_CALUDE_train_speed_calculation_l3564_356460


namespace NUMINAMATH_CALUDE_square_sum_given_sum_and_product_l3564_356486

theorem square_sum_given_sum_and_product (m n : ℝ) 
  (h1 : m + n = 10) (h2 : m * n = 24) : m^2 + n^2 = 52 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_sum_and_product_l3564_356486


namespace NUMINAMATH_CALUDE_pauls_lost_crayons_l3564_356429

/-- Paul's crayon problem -/
theorem pauls_lost_crayons (initial : ℕ) (given_away : ℕ) (remaining : ℕ) :
  initial = 1453 →
  given_away = 563 →
  remaining = 332 →
  initial - given_away - remaining = 558 := by
  sorry

end NUMINAMATH_CALUDE_pauls_lost_crayons_l3564_356429


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_p_and_q_l3564_356436

theorem not_p_sufficient_not_necessary_for_not_p_and_q (p q : Prop) :
  (∀ (h : ¬p), ¬(p ∧ q)) ∧
  ¬(∀ (h : ¬(p ∧ q)), ¬p) :=
sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_p_and_q_l3564_356436


namespace NUMINAMATH_CALUDE_snail_distance_is_20_l3564_356494

def snail_path : List ℤ := [0, 4, -3, 6]

def distance (a b : ℤ) : ℕ := (a - b).natAbs

def total_distance (path : List ℤ) : ℕ :=
  (path.zip path.tail).foldl (fun acc (a, b) => acc + distance a b) 0

theorem snail_distance_is_20 : total_distance snail_path = 20 := by
  sorry

end NUMINAMATH_CALUDE_snail_distance_is_20_l3564_356494


namespace NUMINAMATH_CALUDE_intersection_solutions_l3564_356485

theorem intersection_solutions (α β : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 = 1 ∧ 2*x + 2*y - 1 - Real.sqrt 3 = 0 ∧
   (x = Real.sin α ∧ y = Real.sin (2*β) ∨ x = Real.sin β ∧ y = Real.cos (2*α))) →
  (∃ (n k : ℤ), α = (-1)^n * π/6 + π * (n : ℝ) ∧ β = π/3 + 2*π * (k : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_intersection_solutions_l3564_356485


namespace NUMINAMATH_CALUDE_function_properties_l3564_356495

theorem function_properties (f : ℝ → ℝ) 
  (h1 : ∃ x, f x ≠ 0)
  (h2 : ∀ x y, f x + f y = 2 * f ((x + y) / 2) * f ((x - y) / 2)) :
  (f 0 = 1) ∧ (∀ x, f (-x) = f x) := by
sorry

end NUMINAMATH_CALUDE_function_properties_l3564_356495


namespace NUMINAMATH_CALUDE_monthly_expenses_ratio_l3564_356474

theorem monthly_expenses_ratio (E : ℝ) (rent_percentage : ℝ) (rent_amount : ℝ) (savings : ℝ)
  (h1 : rent_percentage = 0.07)
  (h2 : rent_amount = 133)
  (h3 : savings = 817)
  (h4 : rent_amount = E * rent_percentage) :
  (E - rent_amount - savings) / E = 0.5 := by
sorry

end NUMINAMATH_CALUDE_monthly_expenses_ratio_l3564_356474


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_achievable_l3564_356418

theorem min_value_theorem (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  4 * a^3 + 8 * b^3 + 27 * c^3 + 64 * d^3 + 2 / (a * b * c * d) ≥ 16 * Real.sqrt 3 :=
by
  sorry

theorem min_value_achievable :
  ∃ (a b c d : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    4 * a^3 + 8 * b^3 + 27 * c^3 + 64 * d^3 + 2 / (a * b * c * d) = 16 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_achievable_l3564_356418


namespace NUMINAMATH_CALUDE_daily_savings_amount_l3564_356457

/-- Proves that saving the same amount daily for 20 days totaling 2 dimes equals 1 cent per day -/
theorem daily_savings_amount (savings_period : ℕ) (total_saved : ℕ) (daily_amount : ℚ) : 
  savings_period = 20 →
  total_saved = 20 →  -- 2 dimes = 20 cents
  daily_amount * savings_period = total_saved →
  daily_amount = 1 := by
sorry

end NUMINAMATH_CALUDE_daily_savings_amount_l3564_356457


namespace NUMINAMATH_CALUDE_root_product_bound_l3564_356490

-- Define the equations
def equation1 (x : ℝ) : Prop := Real.log x / Real.log 4 - (1/4)^x = 0
def equation2 (x : ℝ) : Prop := Real.log x / Real.log (1/4) - (1/4)^x = 0

-- State the theorem
theorem root_product_bound 
  (x₁ x₂ : ℝ) 
  (h1 : equation1 x₁) 
  (h2 : equation2 x₂) : 
  0 < x₁ * x₂ ∧ x₁ * x₂ < 1 := by
  sorry

end NUMINAMATH_CALUDE_root_product_bound_l3564_356490


namespace NUMINAMATH_CALUDE_prime_sum_problem_l3564_356444

theorem prime_sum_problem (p q s r : ℕ) : 
  Nat.Prime p → Nat.Prime q → Nat.Prime s → Nat.Prime r →
  p + q + s = r →
  1 < p → p < q → q < s →
  p = 2 := by
sorry

end NUMINAMATH_CALUDE_prime_sum_problem_l3564_356444


namespace NUMINAMATH_CALUDE_train_crossing_time_l3564_356435

theorem train_crossing_time (train_length platform_length platform_crossing_time : ℝ) 
  (h1 : train_length = 900)
  (h2 : platform_length = 1050)
  (h3 : platform_crossing_time = 39)
  : (train_length / ((train_length + platform_length) / platform_crossing_time)) = 18 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l3564_356435


namespace NUMINAMATH_CALUDE_volume_ratio_equal_surface_area_l3564_356469

/-- Given an equilateral cone, an equilateral cylinder, and a sphere, all with equal surface area F,
    their volumes are in the ratio 2 : √6 : 3. -/
theorem volume_ratio_equal_surface_area (F : ℝ) (F_pos : F > 0) :
  ∃ (K₁ K₂ K₃ : ℝ),
    (K₁ > 0 ∧ K₂ > 0 ∧ K₃ > 0) ∧
    (K₁ = F * Real.sqrt F / (9 * Real.sqrt Real.pi)) ∧  -- Volume of equilateral cone
    (K₂ = F * Real.sqrt F * Real.sqrt 6 / (18 * Real.sqrt Real.pi)) ∧  -- Volume of equilateral cylinder
    (K₃ = F * Real.sqrt F / (6 * Real.sqrt Real.pi)) ∧  -- Volume of sphere
    (K₁ / 2 = K₂ / Real.sqrt 6 ∧ K₁ / 2 = K₃ / 3) :=
by sorry

end NUMINAMATH_CALUDE_volume_ratio_equal_surface_area_l3564_356469


namespace NUMINAMATH_CALUDE_ram_birthday_is_19th_l3564_356465

/-- Represents the number of languages learned per day -/
def languages_per_day : ℕ := sorry

/-- Represents the number of languages known on the first day of the month -/
def languages_first_day : ℕ := 820

/-- Represents the number of languages known on the last day of the month -/
def languages_last_day : ℕ := 1100

/-- Represents the number of languages known on the birthday -/
def languages_birthday : ℕ := 1000

/-- Represents the day of the month on which the birthday falls -/
def birthday : ℕ := sorry

theorem ram_birthday_is_19th : 
  birthday = 19 ∧
  languages_per_day * (birthday - 1) + languages_first_day = languages_birthday ∧
  languages_per_day * (31 - 1) + languages_first_day = languages_last_day :=
sorry

end NUMINAMATH_CALUDE_ram_birthday_is_19th_l3564_356465


namespace NUMINAMATH_CALUDE_triangle_angle_problem_l3564_356426

theorem triangle_angle_problem (x z : ℝ) : 
  (2*x + 3*x + x = 180) → 
  (x + z = 180) → 
  z = 150 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_problem_l3564_356426


namespace NUMINAMATH_CALUDE_solution_set_quadratic_equation_l3564_356400

theorem solution_set_quadratic_equation :
  {x : ℝ | x^2 - 3*x + 2 = 0} = {1, 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_equation_l3564_356400


namespace NUMINAMATH_CALUDE_equidistant_function_b_squared_l3564_356464

/-- A complex function that is equidistant from its input and the origin -/
def equidistant_function (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) : ℂ → ℂ := 
  fun z ↦ (a + b * Complex.I) * z

/-- The main theorem -/
theorem equidistant_function_b_squared 
  (a b : ℝ) 
  (h₁ : 0 < a) 
  (h₂ : 0 < b) 
  (h₃ : ∀ z : ℂ, Complex.abs (equidistant_function a b h₁ h₂ z - z) = Complex.abs (equidistant_function a b h₁ h₂ z))
  (h₄ : Complex.abs (a + b * Complex.I) = 10) :
  b^2 = 99.75 := by
sorry

end NUMINAMATH_CALUDE_equidistant_function_b_squared_l3564_356464


namespace NUMINAMATH_CALUDE_expression_evaluation_l3564_356438

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem expression_evaluation (y : ℕ) (x : ℕ) (h1 : y = 2) (h2 : x = y + 1) :
  5 * (factorial y) * (x ^ y) + 3 * (factorial x) * (y ^ x) = 234 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3564_356438


namespace NUMINAMATH_CALUDE_sum_reciprocals_l3564_356468

theorem sum_reciprocals (x y z : ℝ) (ω : ℂ) 
  (hx : x ≠ -1) (hy : y ≠ -1) (hz : z ≠ -1)
  (hω1 : ω^3 = 1) (hω2 : ω ≠ 1)
  (h : 1/(x + ω) + 1/(y + ω) + 1/(z + ω) = ω) :
  1/(x + 1) + 1/(y + 1) + 1/(z + 1) = -1/3 := by
sorry

end NUMINAMATH_CALUDE_sum_reciprocals_l3564_356468


namespace NUMINAMATH_CALUDE_alien_energy_conversion_l3564_356432

def base5_to_base10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

theorem alien_energy_conversion :
  base5_to_base10 [0, 2, 3] = 85 := by
  sorry

end NUMINAMATH_CALUDE_alien_energy_conversion_l3564_356432


namespace NUMINAMATH_CALUDE_fifteen_dogs_like_neither_l3564_356440

/-- Represents the number of dogs in different categories -/
structure DogCounts where
  total : Nat
  likesChicken : Nat
  likesBeef : Nat
  likesBoth : Nat

/-- Calculates the number of dogs that don't like either chicken or beef -/
def dogsLikingNeither (counts : DogCounts) : Nat :=
  counts.total - (counts.likesChicken + counts.likesBeef - counts.likesBoth)

/-- Theorem stating that 15 dogs don't like either chicken or beef -/
theorem fifteen_dogs_like_neither (counts : DogCounts)
  (h1 : counts.total = 75)
  (h2 : counts.likesChicken = 13)
  (h3 : counts.likesBeef = 55)
  (h4 : counts.likesBoth = 8) :
  dogsLikingNeither counts = 15 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_dogs_like_neither_l3564_356440


namespace NUMINAMATH_CALUDE_a_share_is_4080_l3564_356446

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

end NUMINAMATH_CALUDE_a_share_is_4080_l3564_356446


namespace NUMINAMATH_CALUDE_final_cell_count_l3564_356455

/-- Calculates the number of cells after a given number of days, 
    given an initial population and a tripling period. -/
def cell_count (initial_cells : ℕ) (tripling_period : ℕ) (total_days : ℕ) : ℕ :=
  initial_cells * (3 ^ (total_days / tripling_period))

/-- Theorem stating that given 5 initial cells, tripling every 3 days for 9 days, 
    the final cell count is 135. -/
theorem final_cell_count : cell_count 5 3 9 = 135 := by
  sorry

#eval cell_count 5 3 9

end NUMINAMATH_CALUDE_final_cell_count_l3564_356455


namespace NUMINAMATH_CALUDE_no_function_satisfies_conditions_l3564_356462

theorem no_function_satisfies_conditions :
  ¬∃ (f : ℚ → ℝ),
    (f 0 = 0) ∧
    (∀ a : ℚ, a ≠ 0 → f a > 0) ∧
    (∀ x y : ℚ, f (x + y) = f x * f y) ∧
    (∀ x y : ℚ, x ≠ 0 → y ≠ 0 → f (x + y) ≤ max (f x) (f y)) ∧
    (∃ x : ℤ, f x ≠ 1) ∧
    (∀ n : ℕ, n > 0 → ∀ x : ℤ, f (1 + x + x^2 + (x^n - 1) / (x - 1)) = 1) :=
by sorry

end NUMINAMATH_CALUDE_no_function_satisfies_conditions_l3564_356462


namespace NUMINAMATH_CALUDE_inverse_proposition_l3564_356408

theorem inverse_proposition (a b : ℝ) :
  (∀ x y : ℝ, (|x| > |y| → x > y)) →
  (a > b → |a| > |b|) :=
sorry

end NUMINAMATH_CALUDE_inverse_proposition_l3564_356408


namespace NUMINAMATH_CALUDE_concert_attendance_l3564_356480

/-- The number of buses used for the concert. -/
def num_buses : ℕ := 8

/-- The number of students each bus can carry. -/
def students_per_bus : ℕ := 45

/-- The total number of students who went to the concert. -/
def total_students : ℕ := num_buses * students_per_bus

theorem concert_attendance : total_students = 360 := by
  sorry

end NUMINAMATH_CALUDE_concert_attendance_l3564_356480


namespace NUMINAMATH_CALUDE_area_enclosed_by_g_l3564_356459

open Real MeasureTheory

noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x + π / 3)

theorem area_enclosed_by_g : 
  ∫ (x : ℝ) in (0)..(π / 3), g x = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_area_enclosed_by_g_l3564_356459


namespace NUMINAMATH_CALUDE_notebook_ratio_l3564_356499

theorem notebook_ratio (total_students : ℕ) (total_notebooks : ℕ) 
  (h1 : total_students = 28)
  (h2 : total_notebooks = 112)
  (h3 : ∃ (x y : ℕ), x + y = total_students ∧ y = total_students / 2 ∧ 5 * x + 3 * y = total_notebooks) :
  ∃ (x y : ℕ), x = y ∧ x + y = total_students ∧ 5 * x + 3 * y = total_notebooks := by
  sorry

end NUMINAMATH_CALUDE_notebook_ratio_l3564_356499


namespace NUMINAMATH_CALUDE_min_value_of_complex_l3564_356467

open Complex

theorem min_value_of_complex (z : ℂ) (h : abs (z + I) + abs (z - I) = 2) :
  (∀ w : ℂ, abs (w + I) + abs (w - I) = 2 → abs (z + I + 1) ≤ abs (w + I + 1)) ∧
  (∃ z₀ : ℂ, abs (z₀ + I) + abs (z₀ - I) = 2 ∧ abs (z₀ + I + 1) = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_complex_l3564_356467


namespace NUMINAMATH_CALUDE_power_algorithm_correct_l3564_356456

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

end NUMINAMATH_CALUDE_power_algorithm_correct_l3564_356456


namespace NUMINAMATH_CALUDE_larger_number_problem_l3564_356452

theorem larger_number_problem (L S : ℕ) (hL : L > S) : 
  L - S = 1365 → L = 6 * S + 10 → L = 1636 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l3564_356452


namespace NUMINAMATH_CALUDE_symmetric_line_l3564_356409

/-- Given a line L with equation x + 2y - 1 = 0 and a point P(1, -1),
    the line symmetric to L with respect to P has the equation x + 2y - 3 = 0 -/
theorem symmetric_line (x y : ℝ) : 
  (x + 2*y - 1 = 0) → -- original line equation
  (∃ (x' y' : ℝ), (x' = 2 - x ∧ y' = -2 - y) ∧ (x' + 2*y' - 1 = 0)) → -- symmetry condition
  (x + 2*y - 3 = 0) -- symmetric line equation
:= by sorry

end NUMINAMATH_CALUDE_symmetric_line_l3564_356409


namespace NUMINAMATH_CALUDE_distance_between_points_l3564_356448

/-- The distance between two points (-3, -4) and (5, 6) is 2√41 -/
theorem distance_between_points : 
  let a : ℝ × ℝ := (-3, -4)
  let b : ℝ × ℝ := (5, 6)
  Real.sqrt ((b.1 - a.1)^2 + (b.2 - a.2)^2) = 2 * Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l3564_356448


namespace NUMINAMATH_CALUDE_three_digit_base_nine_to_base_three_digit_count_sum_l3564_356493

theorem three_digit_base_nine_to_base_three_digit_count_sum :
  ∀ n : ℕ,
  (3^4 ≤ n ∧ n < 3^6) →
  (∃ e : ℕ, (3^(e-1) ≤ n ∧ n < 3^e) ∧ (e = 5 ∨ e = 6 ∨ e = 7)) ∧
  (5 + 6 + 7 = 18) :=
sorry

end NUMINAMATH_CALUDE_three_digit_base_nine_to_base_three_digit_count_sum_l3564_356493


namespace NUMINAMATH_CALUDE_systematic_sampling_used_l3564_356433

/-- Represents the sampling methods --/
inductive SamplingMethod
  | Lottery
  | RandomNumberTable
  | Systematic
  | Stratified

/-- Represents the auditorium setup and sampling process --/
structure AuditoriumSampling where
  total_seats : Nat
  seats_per_row : Nat
  selected_seat_number : Nat
  num_selected : Nat

/-- Determines the sampling method based on the auditorium setup and selection process --/
def determine_sampling_method (setup : AuditoriumSampling) : SamplingMethod :=
  sorry

/-- Theorem stating that the sampling method used is systematic sampling --/
theorem systematic_sampling_used (setup : AuditoriumSampling) 
  (h1 : setup.total_seats = 25)
  (h2 : setup.seats_per_row = 20)
  (h3 : setup.selected_seat_number = 15)
  (h4 : setup.num_selected = 25) :
  determine_sampling_method setup = SamplingMethod.Systematic :=
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_used_l3564_356433


namespace NUMINAMATH_CALUDE_S_in_quadrants_I_and_II_l3564_356437

-- Define the set of points satisfying the inequalities
def S : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 > 2 * p.1 ∧ p.2 > 4 - p.1}

-- Define quadrants I and II
def quadrantI : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 > 0 ∧ p.2 > 0}
def quadrantII : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 < 0 ∧ p.2 > 0}

-- Theorem stating that S is contained in quadrants I and II
theorem S_in_quadrants_I_and_II : S ⊆ quadrantI ∪ quadrantII := by
  sorry


end NUMINAMATH_CALUDE_S_in_quadrants_I_and_II_l3564_356437


namespace NUMINAMATH_CALUDE_x_twenty_percent_greater_than_52_l3564_356461

theorem x_twenty_percent_greater_than_52 (x : ℝ) : 
  x = 52 * (1 + 20 / 100) → x = 62.4 := by
sorry

end NUMINAMATH_CALUDE_x_twenty_percent_greater_than_52_l3564_356461


namespace NUMINAMATH_CALUDE_snow_probability_l3564_356415

theorem snow_probability (p1 p2 : ℚ) : 
  p1 = 1/4 → p2 = 1/3 → 
  1 - (1 - p1)^4 * (1 - p2)^3 = 68359/100000 := by sorry

end NUMINAMATH_CALUDE_snow_probability_l3564_356415


namespace NUMINAMATH_CALUDE_pecan_mixture_amount_l3564_356407

/-- Prove that the amount of pecans in a mixture is correct given the specified conditions. -/
theorem pecan_mixture_amount 
  (cashew_amount : ℝ) 
  (cashew_price : ℝ) 
  (mixture_price : ℝ) 
  (pecan_amount : ℝ) :
  cashew_amount = 2 ∧ 
  cashew_price = 3.5 ∧ 
  mixture_price = 4.34 ∧
  pecan_amount = 1.33333333333 →
  pecan_amount = 1.33333333333 :=
by sorry

end NUMINAMATH_CALUDE_pecan_mixture_amount_l3564_356407


namespace NUMINAMATH_CALUDE_slope_intercept_form_parallel_lines_a_value_l3564_356476

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The slope-intercept form of a line ax + by + c = 0 is y = (-a/b)x - (c/b) -/
theorem slope_intercept_form {a b c : ℝ} (hb : b ≠ 0) :
  ∀ x y : ℝ, a * x + b * y + c = 0 ↔ y = (-a/b) * x - (c/b) :=
sorry

theorem parallel_lines_a_value :
  (∀ x y : ℝ, a * x + 4 * y + 1 = 0 ↔ 2 * x + y - 2 = 0) → a = 8 :=
sorry

end NUMINAMATH_CALUDE_slope_intercept_form_parallel_lines_a_value_l3564_356476


namespace NUMINAMATH_CALUDE_sum_of_digits_of_power_l3564_356425

-- Define a function to get the tens digit of a natural number
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

-- Define a function to get the ones digit of a natural number
def ones_digit (n : ℕ) : ℕ := n % 10

-- State the theorem
theorem sum_of_digits_of_power : 
  tens_digit ((3 + 4)^17) + ones_digit ((3 + 4)^17) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_power_l3564_356425


namespace NUMINAMATH_CALUDE_total_marbles_l3564_356453

def marbles_bought : ℝ := 5423.6
def marbles_before : ℝ := 12834.9

theorem total_marbles :
  marbles_bought + marbles_before = 18258.5 := by
  sorry

end NUMINAMATH_CALUDE_total_marbles_l3564_356453


namespace NUMINAMATH_CALUDE_mariela_get_well_cards_l3564_356482

theorem mariela_get_well_cards (total : ℝ) (from_home : ℝ) (from_country : ℝ)
  (h1 : total = 403.0)
  (h2 : from_home = 287.0)
  (h3 : total = from_home + from_country) :
  from_country = 116.0 := by
  sorry

end NUMINAMATH_CALUDE_mariela_get_well_cards_l3564_356482


namespace NUMINAMATH_CALUDE_smallest_d_for_3150_perfect_square_l3564_356404

theorem smallest_d_for_3150_perfect_square : 
  ∃ (d : ℕ), d > 0 ∧ d = 14 ∧ 
  (∃ (n : ℕ), 3150 * d = n^2) ∧
  (∀ (k : ℕ), k > 0 → k < d → ¬∃ (m : ℕ), 3150 * k = m^2) := by
  sorry

end NUMINAMATH_CALUDE_smallest_d_for_3150_perfect_square_l3564_356404


namespace NUMINAMATH_CALUDE_infinitely_many_composite_mersenne_numbers_l3564_356458

theorem infinitely_many_composite_mersenne_numbers :
  ∀ k : ℕ, ∃ n : ℕ, 
    Odd n ∧ 
    ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ 2^n - 1 = a * b :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_composite_mersenne_numbers_l3564_356458


namespace NUMINAMATH_CALUDE_fraction_decomposition_l3564_356466

theorem fraction_decomposition (n : ℕ) (hn : n > 0) :
  (∃ (a b : ℕ), a ≠ b ∧ 3 / (5 * n) = 1 / a + 1 / b) ∧
  ((∃ (x : ℤ), 3 / (5 * n) = 1 / x + 1 / x) ↔ ∃ (k : ℕ), n = 3 * k) ∧
  (n > 1 → ∃ (c d : ℕ), 3 / (5 * n) = 1 / c - 1 / d) :=
by sorry

end NUMINAMATH_CALUDE_fraction_decomposition_l3564_356466


namespace NUMINAMATH_CALUDE_triangle_area_l3564_356439

/-- Prove that the area of the triangle formed by the lines x = -5, y = x, and the x-axis is 12.5 -/
theorem triangle_area : 
  let line1 : ℝ → ℝ := λ x => -5
  let line2 : ℝ → ℝ := λ x => x
  let intersection_x : ℝ := -5
  let intersection_y : ℝ := line2 intersection_x
  let base : ℝ := abs intersection_x
  let height : ℝ := abs intersection_y
  (1/2) * base * height = 12.5 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l3564_356439


namespace NUMINAMATH_CALUDE_batsman_sixes_l3564_356401

def total_runs : ℕ := 120
def boundaries : ℕ := 3
def boundary_value : ℕ := 4
def six_value : ℕ := 6

theorem batsman_sixes :
  ∃ (sixes : ℕ),
    sixes * six_value + boundaries * boundary_value + (total_runs / 2) = total_runs ∧
    sixes = 8 := by
  sorry

end NUMINAMATH_CALUDE_batsman_sixes_l3564_356401


namespace NUMINAMATH_CALUDE_inequality_proof_l3564_356477

theorem inequality_proof (x y : ℝ) : 
  5 * x^2 + y^2 + 1 ≥ 4 * x * y + 2 * x ∧ 
  (5 * x^2 + y^2 + 1 = 4 * x * y + 2 * x ↔ x = 1 ∧ y = 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3564_356477


namespace NUMINAMATH_CALUDE_game_cost_proof_l3564_356422

/-- The cost of a video game that Ronald and Max want to buy --/
def game_cost : ℕ := 60

/-- The price of each ice cream --/
def ice_cream_price : ℕ := 5

/-- The number of ice creams they need to sell to afford the game --/
def ice_creams_needed : ℕ := 24

/-- The number of people splitting the cost of the game --/
def people_splitting_cost : ℕ := 2

/-- Theorem stating that the game cost is correct given the conditions --/
theorem game_cost_proof : 
  game_cost = (ice_cream_price * ice_creams_needed) / people_splitting_cost :=
by sorry

end NUMINAMATH_CALUDE_game_cost_proof_l3564_356422


namespace NUMINAMATH_CALUDE_carnival_spending_theorem_l3564_356434

def carnival_spending (initial_amount food_cost : ℕ) : ℕ :=
  let ride_cost := 2 * food_cost
  let game_cost := 2 * food_cost
  initial_amount - (food_cost + ride_cost + game_cost)

theorem carnival_spending_theorem :
  carnival_spending 80 15 = 5 := by
  sorry

end NUMINAMATH_CALUDE_carnival_spending_theorem_l3564_356434


namespace NUMINAMATH_CALUDE_arithmetic_triangle_sum_l3564_356471

-- Define a triangle with angles in arithmetic progression and side lengths 6, 7, and y
structure ArithmeticTriangle where
  y : ℝ
  angle_progression : ℝ → ℝ → ℝ → Prop
  side_lengths : ℝ → ℝ → ℝ → Prop

-- Define the sum of possible y values
def sum_of_y_values (t : ArithmeticTriangle) : ℝ := sorry

-- Define positive integers a, b, and c
def a : ℕ := sorry
def b : ℕ := sorry
def c : ℕ := sorry

-- Theorem statement
theorem arithmetic_triangle_sum :
  ∃ (t : ArithmeticTriangle),
    t.angle_progression 60 60 60 ∧
    t.side_lengths 6 7 t.y ∧
    sum_of_y_values t = a + Real.sqrt b + Real.sqrt c ∧
    a + b + c = 68 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_triangle_sum_l3564_356471


namespace NUMINAMATH_CALUDE_total_gas_usage_l3564_356441

theorem total_gas_usage (adhira_usage : ℕ) (felicity_usage : ℕ) : 
  felicity_usage = 4 * adhira_usage - 5 →
  felicity_usage = 23 →
  felicity_usage + adhira_usage = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_total_gas_usage_l3564_356441


namespace NUMINAMATH_CALUDE_andrei_club_visits_l3564_356489

theorem andrei_club_visits :
  ∀ (d c : ℕ),
  15 * d + 11 * c = 115 →
  d + c = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_andrei_club_visits_l3564_356489


namespace NUMINAMATH_CALUDE_negative_three_times_inequality_l3564_356430

theorem negative_three_times_inequality {a b : ℝ} (h : a < b) : -3 * a > -3 * b := by
  sorry

end NUMINAMATH_CALUDE_negative_three_times_inequality_l3564_356430


namespace NUMINAMATH_CALUDE_largest_gcd_value_l3564_356491

theorem largest_gcd_value (n : ℕ) : 
  ∃ (m : ℕ), (∀ k : ℕ, Nat.gcd (k^2 + 3) ((k + 1)^2 + 3) ≤ m) ∧ 
             (Nat.gcd (n^2 + 3) ((n + 1)^2 + 3) = m) ∧
             m = 13 := by
  sorry

end NUMINAMATH_CALUDE_largest_gcd_value_l3564_356491


namespace NUMINAMATH_CALUDE_forty_percent_of_number_equals_144_l3564_356423

theorem forty_percent_of_number_equals_144 (x : ℝ) : 0.4 * x = 144 → x = 360 := by
  sorry

end NUMINAMATH_CALUDE_forty_percent_of_number_equals_144_l3564_356423


namespace NUMINAMATH_CALUDE_friday_thirteenth_most_common_l3564_356488

/-- Represents a day of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a month in the Gregorian calendar -/
structure Month where
  startDay : DayOfWeek
  length : Nat

/-- Represents a year in the Gregorian calendar -/
structure Year where
  isLeap : Bool
  months : List Month

/-- Calculates the day of week for the 13th of a given month -/
def thirteenthDayOfMonth (m : Month) : DayOfWeek :=
  sorry

/-- Counts the occurrences of each day as the 13th in a 400-year cycle -/
def countThirteenths (years : List Year) : DayOfWeek → Nat :=
  sorry

/-- The Gregorian calendar repeats every 400 years -/
def gregorianCycle : List Year :=
  sorry

/-- Main theorem: Friday is the most common day for the 13th of a month -/
theorem friday_thirteenth_most_common :
  ∀ d : DayOfWeek, d ≠ DayOfWeek.Friday →
    countThirteenths gregorianCycle DayOfWeek.Friday > countThirteenths gregorianCycle d :=
  sorry

end NUMINAMATH_CALUDE_friday_thirteenth_most_common_l3564_356488


namespace NUMINAMATH_CALUDE_distance_to_school_proof_l3564_356475

/-- The distance from Layla's house to the high school -/
def distance_to_school : ℝ := 3

theorem distance_to_school_proof :
  ∀ (total_distance : ℝ),
  (2 * distance_to_school + 4 = total_distance) →
  (total_distance = 10) →
  distance_to_school = 3 := by
sorry

end NUMINAMATH_CALUDE_distance_to_school_proof_l3564_356475


namespace NUMINAMATH_CALUDE_arithmetic_sequence_y_value_l3564_356402

/-- Given an arithmetic sequence with first three terms y + 1, 3y - 2, and 9 - 2y, prove that y = 2 -/
theorem arithmetic_sequence_y_value (y : ℝ) : 
  (∃ d : ℝ, (3*y - 2) - (y + 1) = d ∧ (9 - 2*y) - (3*y - 2) = d) → y = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_y_value_l3564_356402


namespace NUMINAMATH_CALUDE_store_owner_uniforms_l3564_356487

theorem store_owner_uniforms :
  ∃ (U : ℕ), 
    (U > 0) ∧ 
    (∃ (E : ℕ), U + 1 = 2 * E) ∧ 
    (∀ (V : ℕ), V < U → ¬(∃ (F : ℕ), V + 1 = 2 * F)) ∧
    (U = 3) := by
  sorry

end NUMINAMATH_CALUDE_store_owner_uniforms_l3564_356487


namespace NUMINAMATH_CALUDE_distance_to_xy_plane_l3564_356447

/-- The distance from a point (3, 2, -5) to the xy-plane is 5. -/
theorem distance_to_xy_plane : 
  let p : ℝ × ℝ × ℝ := (3, 2, -5)
  abs (p.2) = 5 := by sorry

end NUMINAMATH_CALUDE_distance_to_xy_plane_l3564_356447


namespace NUMINAMATH_CALUDE_train_length_calculation_train_B_length_l3564_356484

/-- The length of a train given the length of another train, their speeds, and the time they take to cross each other when moving in opposite directions. -/
theorem train_length_calculation (length_A : ℝ) (speed_A speed_B : ℝ) (crossing_time : ℝ) : ℝ :=
  let speed_A_ms := speed_A * 1000 / 3600
  let speed_B_ms := speed_B * 1000 / 3600
  let relative_speed := speed_A_ms + speed_B_ms
  let total_distance := relative_speed * crossing_time
  total_distance - length_A

/-- The length of Train B is approximately 299.95 meters. -/
theorem train_B_length : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |train_length_calculation 200 120 80 9 - 299.95| < ε :=
sorry

end NUMINAMATH_CALUDE_train_length_calculation_train_B_length_l3564_356484


namespace NUMINAMATH_CALUDE_certain_number_proof_l3564_356496

theorem certain_number_proof (p q x : ℝ) 
  (h1 : 3 / p = x)
  (h2 : 3 / q = 18)
  (h3 : p - q = 0.20833333333333334) :
  x = 8 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3564_356496


namespace NUMINAMATH_CALUDE_sarah_sock_purchase_l3564_356497

/-- Represents the number of pairs of socks at each price point --/
structure SockCounts where
  two_dollar : ℕ
  four_dollar : ℕ
  five_dollar : ℕ

/-- Checks if the given sock counts satisfy the problem conditions --/
def is_valid_solution (s : SockCounts) : Prop :=
  s.two_dollar + s.four_dollar + s.five_dollar = 15 ∧
  2 * s.two_dollar + 4 * s.four_dollar + 5 * s.five_dollar = 45 ∧
  s.two_dollar ≥ 1 ∧ s.four_dollar ≥ 1 ∧ s.five_dollar ≥ 1

theorem sarah_sock_purchase :
  ∃ (s : SockCounts), is_valid_solution s ∧ (s.two_dollar = 8 ∨ s.two_dollar = 9) :=
sorry

end NUMINAMATH_CALUDE_sarah_sock_purchase_l3564_356497


namespace NUMINAMATH_CALUDE_partial_fraction_sum_zero_l3564_356442

theorem partial_fraction_sum_zero (A B C D E F : ℝ) :
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ -1 ∧ x ≠ -2 ∧ x ≠ -3 ∧ x ≠ -4 ∧ x ≠ -5 →
    1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5)) =
    A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5)) →
  A + B + C + D + E + F = 0 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_sum_zero_l3564_356442


namespace NUMINAMATH_CALUDE_pole_wire_length_l3564_356451

def pole_problem (base_distance : ℝ) (short_pole_height : ℝ) (tall_pole_height : ℝ) (short_pole_elevation : ℝ) : Prop :=
  let effective_short_pole_height : ℝ := short_pole_height + short_pole_elevation
  let vertical_distance : ℝ := tall_pole_height - effective_short_pole_height
  let wire_length : ℝ := Real.sqrt (base_distance^2 + vertical_distance^2)
  wire_length = Real.sqrt 445

theorem pole_wire_length :
  pole_problem 18 6 20 3 :=
by
  sorry

end NUMINAMATH_CALUDE_pole_wire_length_l3564_356451


namespace NUMINAMATH_CALUDE_quadratic_inequality_requires_conditional_branch_l3564_356443

-- Define the type for algorithms
inductive Algorithm
  | ProductOfTwoNumbers
  | DistancePointToLine
  | SolveQuadraticInequality
  | AreaOfTrapezoid

-- Define a function to check if an algorithm requires a conditional branch
def requiresConditionalBranch (a : Algorithm) : Prop :=
  match a with
  | Algorithm.SolveQuadraticInequality => True
  | _ => False

-- State the theorem
theorem quadratic_inequality_requires_conditional_branch :
  ∀ (a : Algorithm),
    requiresConditionalBranch a ↔ a = Algorithm.SolveQuadraticInequality :=
by sorry

#check quadratic_inequality_requires_conditional_branch

end NUMINAMATH_CALUDE_quadratic_inequality_requires_conditional_branch_l3564_356443


namespace NUMINAMATH_CALUDE_roots_quadratic_equation_l3564_356428

theorem roots_quadratic_equation (m n : ℝ) : 
  (m^2 - 8*m + 5 = 0) → 
  (n^2 - 8*n + 5 = 0) → 
  (1 / (m - 1) + 1 / (n - 1) = -3) := by
sorry

end NUMINAMATH_CALUDE_roots_quadratic_equation_l3564_356428


namespace NUMINAMATH_CALUDE_shorter_leg_length_l3564_356431

/-- Represents a 30-60-90 triangle -/
structure Triangle30_60_90 where
  /-- The length of the shorter leg -/
  shorter_leg : ℝ
  /-- The length of the longer leg -/
  longer_leg : ℝ
  /-- The length of the hypotenuse -/
  hypotenuse : ℝ
  /-- The length of the median to the hypotenuse -/
  median_to_hypotenuse : ℝ
  /-- Constraint: The median to the hypotenuse is half the hypotenuse -/
  median_hypotenuse_relation : median_to_hypotenuse = hypotenuse / 2
  /-- Constraint: The shorter leg is half the hypotenuse -/
  shorter_leg_relation : shorter_leg = hypotenuse / 2
  /-- Constraint: The longer leg is √3 times the shorter leg -/
  longer_leg_relation : longer_leg = shorter_leg * Real.sqrt 3

/-- 
Theorem: In a 30-60-90 triangle, if the length of the median to the hypotenuse is 15 units, 
then the length of the shorter leg is 15 units.
-/
theorem shorter_leg_length (t : Triangle30_60_90) (h : t.median_to_hypotenuse = 15) : 
  t.shorter_leg = 15 := by
  sorry

end NUMINAMATH_CALUDE_shorter_leg_length_l3564_356431


namespace NUMINAMATH_CALUDE_remaining_segment_length_l3564_356478

/-- Represents an equilateral triangle with segments drawn from vertices to opposite sides. -/
structure SegmentedEquilateralTriangle where
  /-- Length of the first segment on one side -/
  a : ℝ
  /-- Length of the second segment on one side -/
  b : ℝ
  /-- Length of the third segment on one side -/
  c : ℝ
  /-- Length of the shortest segment on another side -/
  d : ℝ
  /-- Length of the segment adjacent to the shortest segment -/
  e : ℝ
  /-- Assumption that the triangle is equilateral and segments form a complete side -/
  side_length : a + b + c = d + e + (a + b + c - (d + e))

/-- Theorem stating that the remaining segment length is 4 cm given the conditions -/
theorem remaining_segment_length
  (triangle : SegmentedEquilateralTriangle)
  (h1 : triangle.a = 5)
  (h2 : triangle.b = 10)
  (h3 : triangle.c = 2)
  (h4 : triangle.d = 1.5)
  (h5 : triangle.e = 11.5) :
  triangle.a + triangle.b + triangle.c - (triangle.d + triangle.e) = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_remaining_segment_length_l3564_356478


namespace NUMINAMATH_CALUDE_cube_root_opposite_zero_l3564_356410

theorem cube_root_opposite_zero :
  ∀ x : ℝ, (x^(1/3) = -x) ↔ (x = 0) :=
sorry

end NUMINAMATH_CALUDE_cube_root_opposite_zero_l3564_356410


namespace NUMINAMATH_CALUDE_certain_number_proof_l3564_356417

theorem certain_number_proof : 
  ∃ x : ℝ, 0.8 * x = (4 / 5) * 25 + 16 ∧ x = 45 := by
sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3564_356417


namespace NUMINAMATH_CALUDE_division_remainder_l3564_356414

theorem division_remainder (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 11 →
  divisor = 4 →
  quotient = 2 →
  dividend = divisor * quotient + remainder →
  remainder = 3 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_l3564_356414


namespace NUMINAMATH_CALUDE_joan_change_theorem_l3564_356472

def change_received (cat_toy_cost cage_cost amount_paid : ℚ) : ℚ :=
  amount_paid - (cat_toy_cost + cage_cost)

theorem joan_change_theorem (cat_toy_cost cage_cost amount_paid : ℚ) 
  (h1 : cat_toy_cost = 8.77)
  (h2 : cage_cost = 10.97)
  (h3 : amount_paid = 20) :
  change_received cat_toy_cost cage_cost amount_paid = 0.26 := by
  sorry

#eval change_received 8.77 10.97 20

end NUMINAMATH_CALUDE_joan_change_theorem_l3564_356472


namespace NUMINAMATH_CALUDE_chicken_difference_l3564_356416

/-- The number of chickens in the coop -/
def coop_chickens : ℕ := 14

/-- The number of chickens in the run -/
def run_chickens : ℕ := 2 * coop_chickens

/-- The number of chickens free ranging -/
def free_ranging_chickens : ℕ := 52

/-- The difference between double the number of chickens in the run and the number of chickens free ranging -/
theorem chicken_difference : 2 * run_chickens - free_ranging_chickens = 4 := by
  sorry

end NUMINAMATH_CALUDE_chicken_difference_l3564_356416


namespace NUMINAMATH_CALUDE_ab_value_l3564_356413

theorem ab_value (a b : ℝ) (h1 : a - b = 4) (h2 : a^2 + b^2 = 80) : a * b = 32 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l3564_356413


namespace NUMINAMATH_CALUDE_constant_product_percentage_change_l3564_356419

theorem constant_product_percentage_change (x y : ℝ) (C : ℝ) (h : x * y = C) :
  x * (1 + 0.2) * (y * (1 - 1/6)) = C := by sorry

end NUMINAMATH_CALUDE_constant_product_percentage_change_l3564_356419


namespace NUMINAMATH_CALUDE_only_η_hypergeometric_l3564_356473

/-- Represents the total number of balls -/
def total_balls : ℕ := 10

/-- Represents the number of black balls -/
def black_balls : ℕ := 6

/-- Represents the number of white balls -/
def white_balls : ℕ := 4

/-- Represents the number of balls drawn -/
def drawn_balls : ℕ := 4

/-- Represents the score for a black ball -/
def black_score : ℕ := 2

/-- Represents the score for a white ball -/
def white_score : ℕ := 1

/-- Represents the maximum number drawn -/
def X : ℕ → ℕ := sorry

/-- Represents the minimum number drawn -/
def Y : ℕ → ℕ := sorry

/-- Represents the total score of the drawn balls -/
def ξ : ℕ → ℕ := sorry

/-- Represents the number of black balls drawn -/
def η : ℕ → ℕ := sorry

/-- Defines a hypergeometric distribution -/
def is_hypergeometric (f : ℕ → ℕ) : Prop := sorry

theorem only_η_hypergeometric :
  is_hypergeometric η ∧
  ¬is_hypergeometric X ∧
  ¬is_hypergeometric Y ∧
  ¬is_hypergeometric ξ :=
sorry

end NUMINAMATH_CALUDE_only_η_hypergeometric_l3564_356473


namespace NUMINAMATH_CALUDE_sum_of_five_consecutive_even_numbers_l3564_356427

theorem sum_of_five_consecutive_even_numbers (m : ℤ) (h : Even m) :
  m + (m + 2) + (m + 4) + (m + 6) + (m + 8) = 5 * m + 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_five_consecutive_even_numbers_l3564_356427


namespace NUMINAMATH_CALUDE_annulus_chord_circle_area_equality_l3564_356470

theorem annulus_chord_circle_area_equality (R r x : ℝ) (h1 : 0 < r) (h2 : r < R) (h3 : R^2 = r^2 + x^2) :
  π * x^2 = π * (R^2 - r^2) :=
by sorry

end NUMINAMATH_CALUDE_annulus_chord_circle_area_equality_l3564_356470


namespace NUMINAMATH_CALUDE_absolute_value_equation_one_negative_root_l3564_356479

theorem absolute_value_equation_one_negative_root (a : ℝ) : 
  (∃! x : ℝ, x < 0 ∧ |x| = a * x + 1) → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_one_negative_root_l3564_356479


namespace NUMINAMATH_CALUDE_sequence_max_value_l3564_356449

theorem sequence_max_value (n : ℕ+) : 
  let a := λ (k : ℕ+) => (k : ℝ) / ((k : ℝ)^2 + 6)
  (∀ k : ℕ+, a k ≤ 1/5) ∧ (∃ k : ℕ+, a k = 1/5) :=
sorry

end NUMINAMATH_CALUDE_sequence_max_value_l3564_356449


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3564_356483

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_perp : (1 : ℝ) / (2 * b - 3) = -a / (2 * b)) : 
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ (1 : ℝ) / (2 * y - 3) = -x / (2 * y) → 2 * a + 3 * b ≤ 2 * x + 3 * y) ∧
  (2 * a + 3 * b = 25 / 2) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3564_356483
