import Mathlib

namespace NUMINAMATH_CALUDE_equation_solutions_l1487_148728

theorem equation_solutions :
  (∃ x₁ x₂, x₁ = -3/2 ∧ x₂ = 2 ∧ 2 * x₁^2 - x₁ - 6 = 0 ∧ 2 * x₂^2 - x₂ - 6 = 0) ∧
  (∃ y₁ y₂, y₁ = -1 ∧ y₂ = 1/2 ∧ (y₁ - 2)^2 = 9 * y₁^2 ∧ (y₂ - 2)^2 = 9 * y₂^2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1487_148728


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_area_l1487_148742

-- Define an isosceles right triangle
structure IsoscelesRightTriangle where
  -- AB and BC are the legs, AC is the hypotenuse
  AB : ℝ
  BC : ℝ
  AC : ℝ
  -- Angle B is 90°
  angle_B_is_right : AB^2 + BC^2 = AC^2
  -- Triangle is isosceles (AB = BC)
  is_isosceles : AB = BC
  -- Altitude BD is 1 unit
  altitude_BD : ℝ
  altitude_is_one : altitude_BD = 1

-- Theorem statement
theorem isosceles_right_triangle_area
  (t : IsoscelesRightTriangle) : 
  (1/2) * t.AB * t.BC = 1 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_area_l1487_148742


namespace NUMINAMATH_CALUDE_f_derivative_and_extrema_l1487_148758

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 - 4) * (x - a)

theorem f_derivative_and_extrema (a : ℝ) :
  (∀ x, deriv (f a) x = 3 * x^2 - 2 * a * x - 4) ∧
  (deriv (f a) (-1) = 0 → a = 1/2) ∧
  (a = 1/2 → ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc (-2) 2, f a x ≤ max) ∧
    (∃ x ∈ Set.Icc (-2) 2, f a x = max) ∧
    (∀ x ∈ Set.Icc (-2) 2, f a x ≥ min) ∧
    (∃ x ∈ Set.Icc (-2) 2, f a x = min) ∧
    max = 9/2 ∧ min = -50/27) := by
  sorry

end NUMINAMATH_CALUDE_f_derivative_and_extrema_l1487_148758


namespace NUMINAMATH_CALUDE_heartsuit_three_eight_l1487_148738

-- Define the ♥ operation
def heartsuit (x y : ℝ) : ℝ := 4 * x + 6 * y

-- Theorem statement
theorem heartsuit_three_eight : heartsuit 3 8 = 60 := by
  sorry

end NUMINAMATH_CALUDE_heartsuit_three_eight_l1487_148738


namespace NUMINAMATH_CALUDE_first_store_unload_percentage_l1487_148751

def initial_load : ℝ := 50000
def second_store_percentage : ℝ := 0.20
def final_load : ℝ := 36000

theorem first_store_unload_percentage :
  ∃ x : ℝ, 
    x ≥ 0 ∧ x ≤ 1 ∧
    (1 - x) * initial_load * (1 - second_store_percentage) = final_load ∧
    x = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_first_store_unload_percentage_l1487_148751


namespace NUMINAMATH_CALUDE_imaginary_equation_solution_l1487_148771

theorem imaginary_equation_solution (z : ℂ) (b : ℝ) : 
  (z.re = 0) →  -- z is a pure imaginary number
  ((2 - I) * z = 4 - b * (1 + I)^2) →
  b = -4 :=
by sorry

end NUMINAMATH_CALUDE_imaginary_equation_solution_l1487_148771


namespace NUMINAMATH_CALUDE_A_power_101_l1487_148755

def A : Matrix (Fin 3) (Fin 3) ℤ := !![0, 0, 1; 1, 0, 0; 0, 1, 0]

theorem A_power_101 : A ^ 101 = !![0, 1, 0; 0, 0, 1; 1, 0, 0] := by
  sorry

end NUMINAMATH_CALUDE_A_power_101_l1487_148755


namespace NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_l1487_148782

theorem sqrt_x_div_sqrt_y (x y : ℝ) :
  (1/3)^2 + (1/4)^2 = (17*x/60) * ((1/5)^2 + (1/6)^2) →
  Real.sqrt x / Real.sqrt y = (25/2) * Real.sqrt (60/1037) := by
sorry

end NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_l1487_148782


namespace NUMINAMATH_CALUDE_salad_vegetables_count_l1487_148770

theorem salad_vegetables_count :
  ∀ (cucumbers tomatoes total : ℕ),
  cucumbers = 70 →
  tomatoes = 3 * cucumbers →
  total = cucumbers + tomatoes →
  total = 280 :=
by
  sorry

end NUMINAMATH_CALUDE_salad_vegetables_count_l1487_148770


namespace NUMINAMATH_CALUDE_range_of_a_l1487_148715

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x + 5 ≥ a^2 - 3*a) ↔ -1 ≤ a ∧ a ≤ 4 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l1487_148715


namespace NUMINAMATH_CALUDE_min_of_three_exists_l1487_148716

theorem min_of_three_exists : ∃ (f : ℝ → ℝ → ℝ → ℝ), 
  ∀ (a b c : ℝ), f a b c ≤ a ∧ f a b c ≤ b ∧ f a b c ≤ c ∧ 
  (∀ (m : ℝ), m ≤ a ∧ m ≤ b ∧ m ≤ c → f a b c ≥ m) :=
sorry

end NUMINAMATH_CALUDE_min_of_three_exists_l1487_148716


namespace NUMINAMATH_CALUDE_friday_temperature_l1487_148737

/-- Temperatures for Tuesday, Wednesday, Thursday, and Friday --/
structure WeekTemperatures where
  tuesday : ℝ
  wednesday : ℝ
  thursday : ℝ
  friday : ℝ

/-- Theorem stating that Friday's temperature is 53°C given the conditions --/
theorem friday_temperature (t : WeekTemperatures) : t.friday = 53 :=
  by
  have h1 : (t.tuesday + t.wednesday + t.thursday) / 3 = 45 := by sorry
  have h2 : (t.wednesday + t.thursday + t.friday) / 3 = 50 := by sorry
  have h3 : t.tuesday = 38 := by sorry
  have h4 : t.tuesday = 38 ∨ t.wednesday = 53 ∨ t.thursday = 53 ∨ t.friday = 53 := by sorry
  sorry

end NUMINAMATH_CALUDE_friday_temperature_l1487_148737


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1487_148745

theorem arithmetic_sequence_sum : 
  ∀ (a₁ aₙ d n : ℤ), 
  a₁ = -45 → 
  aₙ = -1 → 
  d = 2 → 
  n = (aₙ - a₁) / d + 1 → 
  n * (a₁ + aₙ) / 2 = -529 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1487_148745


namespace NUMINAMATH_CALUDE_smallest_staircase_steps_l1487_148779

theorem smallest_staircase_steps : ∃ n : ℕ,
  n > 15 ∧
  n % 3 = 1 ∧
  n % 5 = 3 ∧
  n % 7 = 1 ∧
  (∀ m : ℕ, m > 15 ∧ m % 3 = 1 ∧ m % 5 = 3 ∧ m % 7 = 1 → m ≥ n) ∧
  n = 73 := by
sorry

end NUMINAMATH_CALUDE_smallest_staircase_steps_l1487_148779


namespace NUMINAMATH_CALUDE_s_five_value_l1487_148725

theorem s_five_value (x : ℝ) (h : x + 1/x = 4) : x^5 + 1/x^5 = 724 := by
  sorry

end NUMINAMATH_CALUDE_s_five_value_l1487_148725


namespace NUMINAMATH_CALUDE_pizza_consumption_l1487_148726

theorem pizza_consumption (rachel_pizza : ℕ) (bella_pizza : ℕ)
  (h1 : rachel_pizza = 598)
  (h2 : bella_pizza = 354) :
  rachel_pizza + bella_pizza = 952 :=
by sorry

end NUMINAMATH_CALUDE_pizza_consumption_l1487_148726


namespace NUMINAMATH_CALUDE_min_squares_15_step_staircase_l1487_148795

/-- Represents a staircase with a given number of steps -/
structure Staircase :=
  (steps : ℕ)

/-- The minimum number of squares required to cover a staircase -/
def min_squares_to_cover (s : Staircase) : ℕ := s.steps

/-- Theorem: The minimum number of squares required to cover a 15-step staircase is 15 -/
theorem min_squares_15_step_staircase :
  ∀ (s : Staircase), s.steps = 15 → min_squares_to_cover s = 15 := by
  sorry

/-- Lemma: Cutting can only be done along the boundaries of the cells -/
lemma cut_along_boundaries (s : Staircase) : True := by
  sorry

/-- Lemma: Each step in the staircase forms a unit square -/
lemma step_is_unit_square (s : Staircase) : True := by
  sorry

end NUMINAMATH_CALUDE_min_squares_15_step_staircase_l1487_148795


namespace NUMINAMATH_CALUDE_xy_inequality_l1487_148781

theorem xy_inequality (x y : ℝ) (h : (x + 1) * (y + 2) = 8) :
  (x * y - 10)^2 ≥ 64 ∧
  ((x * y - 10)^2 = 64 ↔ (x = 1 ∧ y = 2) ∨ (x = -3 ∧ y = -6)) :=
by sorry

end NUMINAMATH_CALUDE_xy_inequality_l1487_148781


namespace NUMINAMATH_CALUDE_no_valid_arrangement_l1487_148768

/-- A sequence of natural numbers from 1 to 10 -/
def Sequence := Fin 10 → ℕ

/-- Predicate to check if a sequence satisfies the integer percentage difference property -/
def IntegerPercentageDifference (s : Sequence) : Prop :=
  ∀ i : Fin 9, ∃ k : ℤ,
    s (i.succ) = s i + (s i * k) / 100 ∨
    s (i.succ) = s i - (s i * k) / 100

/-- Predicate to check if a sequence contains all numbers from 1 to 10 -/
def ContainsAllNumbers (s : Sequence) : Prop :=
  ∀ n : Fin 10, ∃ i : Fin 10, s i = n.val + 1

/-- Theorem stating that it's impossible to arrange numbers 1 to 10 with the given property -/
theorem no_valid_arrangement :
  ¬ ∃ s : Sequence, IntegerPercentageDifference s ∧ ContainsAllNumbers s := by
  sorry

end NUMINAMATH_CALUDE_no_valid_arrangement_l1487_148768


namespace NUMINAMATH_CALUDE_seating_arrangements_l1487_148777

def total_people : ℕ := 12
def people_per_table : ℕ := 6
def num_tables : ℕ := 2

def arrange_people (n : ℕ) (k : ℕ) : ℕ := (n.factorial * 14400) / (k.factorial * k.factorial)

def arrange_couples (n : ℕ) (k : ℕ) : ℕ := (n.factorial * 14400 * 4096) / (k.factorial * k.factorial)

theorem seating_arrangements :
  (arrange_people total_people people_per_table = (total_people.factorial * 14400) / (people_per_table.factorial * people_per_table.factorial)) ∧
  (arrange_couples total_people people_per_table = (total_people.factorial * 14400 * 4096) / (people_per_table.factorial * people_per_table.factorial)) :=
by sorry

end NUMINAMATH_CALUDE_seating_arrangements_l1487_148777


namespace NUMINAMATH_CALUDE_max_product_under_constraint_l1487_148757

theorem max_product_under_constraint :
  ∀ x y : ℕ+, 
  7 * x + 4 * y = 140 → 
  x * y ≤ 168 := by
sorry

end NUMINAMATH_CALUDE_max_product_under_constraint_l1487_148757


namespace NUMINAMATH_CALUDE_police_hat_multiple_l1487_148750

/-- Proves that the multiple of Fire Chief Simpson's hats that Policeman O'Brien had before he lost one is 2 -/
theorem police_hat_multiple :
  let simpson_hats : ℕ := 15
  let obrien_current_hats : ℕ := 34
  let obrien_previous_hats : ℕ := obrien_current_hats + 1
  ∃ x : ℕ, x * simpson_hats + 5 = obrien_previous_hats ∧ x = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_police_hat_multiple_l1487_148750


namespace NUMINAMATH_CALUDE_average_of_abc_l1487_148778

theorem average_of_abc (A B C : ℚ) 
  (eq1 : 2002 * C - 3003 * A = 6006)
  (eq2 : 2002 * B + 4004 * A = 8008)
  (eq3 : B - C = A + 1) :
  (A + B + C) / 3 = 7 / 3 := by
  sorry

end NUMINAMATH_CALUDE_average_of_abc_l1487_148778


namespace NUMINAMATH_CALUDE_smartphones_for_discount_prove_smartphones_for_discount_l1487_148732

/-- Represents the discount rate as a fraction -/
def discount_rate : ℚ := 5 / 100

/-- Represents the cost of an iPhone X in dollars -/
def iphone_cost : ℕ := 600

/-- Represents the savings in dollars when buying together -/
def savings : ℕ := 90

/-- Theorem stating that the number of smartphones needed to be bought at once to get the discount is 3 -/
theorem smartphones_for_discount : ℕ :=
  3

/-- Proof that the number of smartphones needed to be bought at once to get the discount is 3 -/
theorem prove_smartphones_for_discount :
  ∃ (n : ℕ), n * iphone_cost * discount_rate = savings ∧ n = smartphones_for_discount :=
sorry

end NUMINAMATH_CALUDE_smartphones_for_discount_prove_smartphones_for_discount_l1487_148732


namespace NUMINAMATH_CALUDE_assignment_count_correct_l1487_148701

/-- The number of ways to assign 5 students to 3 universities -/
def assignment_count : ℕ := 150

/-- The number of students to be assigned -/
def num_students : ℕ := 5

/-- The number of universities -/
def num_universities : ℕ := 3

/-- Theorem stating that the number of assignment methods is correct -/
theorem assignment_count_correct :
  (∀ (assignment : Fin num_students → Fin num_universities),
    (∀ u : Fin num_universities, ∃ s : Fin num_students, assignment s = u) →
    (∃ (unique_assignment : Fin num_students → Fin num_universities),
      unique_assignment = assignment)) →
  assignment_count = 150 := by
sorry

end NUMINAMATH_CALUDE_assignment_count_correct_l1487_148701


namespace NUMINAMATH_CALUDE_jennifer_additional_tanks_l1487_148789

/-- Represents the number of fish in each type of tank --/
structure TankCapacity where
  goldfish : Nat
  betta : Nat
  guppy : Nat
  clownfish : Nat

/-- Represents the number of tanks for each type of fish --/
structure TankCount where
  goldfish : Nat
  betta : Nat
  guppy : Nat
  clownfish : Nat

/-- Calculates the total number of fish given tank capacities and counts --/
def totalFish (capacity : TankCapacity) (count : TankCount) : Nat :=
  capacity.goldfish * count.goldfish +
  capacity.betta * count.betta +
  capacity.guppy * count.guppy +
  capacity.clownfish * count.clownfish

/-- Calculates the total number of tanks --/
def totalTanks (count : TankCount) : Nat :=
  count.goldfish + count.betta + count.guppy + count.clownfish

/-- Represents Jennifer's aquarium setup --/
def jennifer_setup : Prop :=
  ∃ (capacity : TankCapacity) (existing_count : TankCount) (new_count : TankCount),
    capacity.goldfish = 15 ∧
    capacity.betta = 1 ∧
    capacity.guppy = 5 ∧
    capacity.clownfish = 4 ∧
    existing_count.goldfish = 3 ∧
    existing_count.betta = 0 ∧
    existing_count.guppy = 0 ∧
    existing_count.clownfish = 0 ∧
    totalFish capacity (TankCount.mk
      existing_count.goldfish
      (existing_count.betta + new_count.betta)
      (existing_count.guppy + new_count.guppy)
      (existing_count.clownfish + new_count.clownfish)) = 75 ∧
    new_count.betta + new_count.guppy + new_count.clownfish = 15 ∧
    ∀ (alt_count : TankCount),
      totalFish capacity (TankCount.mk
        existing_count.goldfish
        (existing_count.betta + alt_count.betta)
        (existing_count.guppy + alt_count.guppy)
        (existing_count.clownfish + alt_count.clownfish)) = 75 →
      totalTanks alt_count ≥ totalTanks new_count

theorem jennifer_additional_tanks : jennifer_setup := by
  sorry

end NUMINAMATH_CALUDE_jennifer_additional_tanks_l1487_148789


namespace NUMINAMATH_CALUDE_half_angle_quadrant_l1487_148753

theorem half_angle_quadrant (α : Real) : 
  (∃ k : ℤ, 2 * k * π + π / 2 < α ∧ α < 2 * k * π + π) →
  (∃ m : ℤ, m * π < α / 2 ∧ α / 2 < m * π + π / 2) :=
sorry

end NUMINAMATH_CALUDE_half_angle_quadrant_l1487_148753


namespace NUMINAMATH_CALUDE_atMostOneHead_exactlyTwoHeads_mutually_exclusive_l1487_148740

/-- Represents the outcome of a single coin toss -/
inductive CoinOutcome
| Heads
| Tails

/-- Represents the outcome of tossing two coins simultaneously -/
def TwoCoinsOutcome := (CoinOutcome × CoinOutcome)

/-- The event of getting at most one head when tossing two coins -/
def atMostOneHead (outcome : TwoCoinsOutcome) : Prop :=
  match outcome with
  | (CoinOutcome.Tails, CoinOutcome.Tails) => True
  | (CoinOutcome.Heads, CoinOutcome.Tails) => True
  | (CoinOutcome.Tails, CoinOutcome.Heads) => True
  | (CoinOutcome.Heads, CoinOutcome.Heads) => False

/-- The event of getting exactly two heads when tossing two coins -/
def exactlyTwoHeads (outcome : TwoCoinsOutcome) : Prop :=
  match outcome with
  | (CoinOutcome.Heads, CoinOutcome.Heads) => True
  | _ => False

/-- Theorem stating that "at most one head" and "exactly two heads" are mutually exclusive -/
theorem atMostOneHead_exactlyTwoHeads_mutually_exclusive :
  ∀ (outcome : TwoCoinsOutcome), ¬(atMostOneHead outcome ∧ exactlyTwoHeads outcome) :=
by
  sorry


end NUMINAMATH_CALUDE_atMostOneHead_exactlyTwoHeads_mutually_exclusive_l1487_148740


namespace NUMINAMATH_CALUDE_function_properties_l1487_148792

noncomputable def f (a b x : ℝ) : ℝ := Real.exp x * (a * x + b) + x^2 + 2 * x

theorem function_properties (a b : ℝ) :
  (f a b 0 = 1 ∧ (deriv (f a b)) 0 = 4) →
  (a = 1 ∧ b = 1) ∧
  (∀ k, (∀ x ∈ Set.Icc (-2) (-1), f 1 1 x ≥ x^2 + 2*(k+1)*x + k) ↔ 
        k ≥ (1/4) * Real.exp (-3/2)) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1487_148792


namespace NUMINAMATH_CALUDE_five_digit_multiple_of_three_l1487_148724

def is_multiple_of_three (n : ℕ) : Prop := ∃ k : ℕ, n = 3 * k

theorem five_digit_multiple_of_three :
  ∀ d : ℕ, d < 10 →
    (is_multiple_of_three (56780 + d) ↔ d = 1) :=
by sorry

end NUMINAMATH_CALUDE_five_digit_multiple_of_three_l1487_148724


namespace NUMINAMATH_CALUDE_johns_growth_per_month_l1487_148760

/-- Proves that John's growth per month is 2 inches given his original height, new height, and growth period. -/
theorem johns_growth_per_month 
  (original_height : ℕ) 
  (new_height_feet : ℕ) 
  (growth_period : ℕ) 
  (h1 : original_height = 66)
  (h2 : new_height_feet = 6)
  (h3 : growth_period = 3) :
  (new_height_feet * 12 - original_height) / growth_period = 2 := by
  sorry

#check johns_growth_per_month

end NUMINAMATH_CALUDE_johns_growth_per_month_l1487_148760


namespace NUMINAMATH_CALUDE_lunch_calories_l1487_148713

/-- The total calories for a kid's lunch -/
def total_calories (burger_calories : ℕ) (carrot_stick_calories : ℕ) (cookie_calories : ℕ) : ℕ :=
  burger_calories + 5 * carrot_stick_calories + 5 * cookie_calories

/-- Theorem stating that the total calories for each kid's lunch is 750 -/
theorem lunch_calories :
  total_calories 400 20 50 = 750 := by
  sorry

end NUMINAMATH_CALUDE_lunch_calories_l1487_148713


namespace NUMINAMATH_CALUDE_smallest_n_remainder_l1487_148706

theorem smallest_n_remainder (N : ℕ) : 
  (N > 0) →
  (∃ k : ℕ, 2008 * N = k^2) →
  (∃ m : ℕ, 2007 * N = m^3) →
  (∀ M : ℕ, M < N → (¬∃ k : ℕ, 2008 * M = k^2) ∨ (¬∃ m : ℕ, 2007 * M = m^3)) →
  N % 25 = 17 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_remainder_l1487_148706


namespace NUMINAMATH_CALUDE_diaries_count_l1487_148786

/-- The number of diaries Natalie's sister has after buying and losing some -/
def final_diaries : ℕ :=
  let initial : ℕ := 23
  let bought : ℕ := 5 * initial
  let total : ℕ := initial + bought
  let lost : ℕ := (7 * total) / 9
  total - lost

theorem diaries_count : final_diaries = 31 := by
  sorry

end NUMINAMATH_CALUDE_diaries_count_l1487_148786


namespace NUMINAMATH_CALUDE_average_of_sequence_l1487_148700

theorem average_of_sequence (z : ℝ) : 
  let sequence := [0, 3*z, 6*z, 12*z, 24*z]
  (sequence.sum / sequence.length : ℝ) = 9*z := by
sorry

end NUMINAMATH_CALUDE_average_of_sequence_l1487_148700


namespace NUMINAMATH_CALUDE_matrix_cube_property_l1487_148796

theorem matrix_cube_property (a b c : ℂ) : 
  let M : Matrix (Fin 3) (Fin 3) ℂ := !![a, b, c; b, c, a; c, a, b]
  (M^3 = 1) → (a*b*c = -1) → (a^3 + b^3 + c^3 = 4) := by
sorry

end NUMINAMATH_CALUDE_matrix_cube_property_l1487_148796


namespace NUMINAMATH_CALUDE_count_repetitive_permutations_formula_l1487_148763

/-- The count of n-repetitive permutations formed by a₁, a₂, a₃, a₄, a₅, a₆ 
    where both a₁ and a₃ each appear an even number of times -/
def count_repetitive_permutations (n : ℕ) : ℕ :=
  (6^n - 2 * 5^n + 4^n) / 4

/-- Theorem stating that the count of n-repetitive permutations with the given conditions
    is equal to (6^n - 2 * 5^n + 4^n) / 4 -/
theorem count_repetitive_permutations_formula (n : ℕ) :
  count_repetitive_permutations n = (6^n - 2 * 5^n + 4^n) / 4 := by
  sorry

end NUMINAMATH_CALUDE_count_repetitive_permutations_formula_l1487_148763


namespace NUMINAMATH_CALUDE_smallest_pretty_multiple_of_401_l1487_148741

/-- A positive integer is pretty if for each of its proper divisors d,
    there exist two divisors whose difference is d. -/
def IsPretty (n : ℕ) : Prop :=
  n > 0 ∧ ∀ d : ℕ, d ∣ n → 1 < d → d < n →
    ∃ d₁ d₂ : ℕ, d₁ ∣ n ∧ d₂ ∣ n ∧ 1 ≤ d₁ ∧ d₁ ≤ n ∧ 1 ≤ d₂ ∧ d₂ ≤ n ∧ d₂ - d₁ = d

theorem smallest_pretty_multiple_of_401 :
  ∃ n : ℕ, n > 401 ∧ 401 ∣ n ∧ IsPretty n ∧
    ∀ m : ℕ, m > 401 → 401 ∣ m → IsPretty m → n ≤ m :=
by
  use 160400
  sorry

end NUMINAMATH_CALUDE_smallest_pretty_multiple_of_401_l1487_148741


namespace NUMINAMATH_CALUDE_circle_radii_order_l1487_148721

/-- Given three circles A, B, and C with the following properties:
    - Circle A has a circumference of 6π
    - Circle B has an area of 16π
    - Circle C has a radius of 2
    Prove that the radii of the circles are ordered as r_C < r_A < r_B -/
theorem circle_radii_order (r_A r_B r_C : ℝ) : 
  (2 * π * r_A = 6 * π) →  -- Circumference of A
  (π * r_B^2 = 16 * π) →   -- Area of B
  (r_C = 2) →              -- Radius of C
  r_C < r_A ∧ r_A < r_B := by
sorry

end NUMINAMATH_CALUDE_circle_radii_order_l1487_148721


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l1487_148775

-- Define the quadratic function f(x)
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 2

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f (-12) (-2) x > 0} = {x : ℝ | -1/2 < x ∧ x < 1/3} := by sorry

-- Part 2
theorem range_of_a_part2 :
  ∀ a : ℝ, (∀ x : ℝ, f a (-1) x ≥ 0) ↔ a ≥ 1/8 := by sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l1487_148775


namespace NUMINAMATH_CALUDE_problem_statement_l1487_148767

theorem problem_statement (x : ℝ) : 
  (0.4 * 60 = (4/5) * x + 4) → x = 25 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1487_148767


namespace NUMINAMATH_CALUDE_residue_sum_mod_19_l1487_148731

theorem residue_sum_mod_19 : (8^1356 + 7^1200) % 19 = 10 := by
  sorry

end NUMINAMATH_CALUDE_residue_sum_mod_19_l1487_148731


namespace NUMINAMATH_CALUDE_new_oarsman_weight_l1487_148748

theorem new_oarsman_weight (n : ℕ) (old_weight average_increase : ℝ) :
  n = 10 ∧ old_weight = 53 ∧ average_increase = 1.8 →
  ∃ new_weight : ℝ,
    new_weight = old_weight + n * average_increase ∧
    new_weight = 71 := by
  sorry

end NUMINAMATH_CALUDE_new_oarsman_weight_l1487_148748


namespace NUMINAMATH_CALUDE_black_square_area_proof_l1487_148746

/-- The edge length of the cube in feet -/
def cube_edge : ℝ := 12

/-- The total area covered by yellow paint in square feet -/
def yellow_area : ℝ := 432

/-- The number of faces on a cube -/
def num_faces : ℕ := 6

/-- The area of the black square on each face of the cube in square feet -/
def black_square_area : ℝ := 72

theorem black_square_area_proof :
  let total_surface_area := num_faces * cube_edge ^ 2
  let yellow_area_per_face := yellow_area / num_faces
  black_square_area = cube_edge ^ 2 - yellow_area_per_face := by
  sorry

end NUMINAMATH_CALUDE_black_square_area_proof_l1487_148746


namespace NUMINAMATH_CALUDE_lucy_age_l1487_148756

def FriendGroup : Type := Fin 6 → Nat

def validAges (group : FriendGroup) : Prop :=
  ∃ (perm : Equiv.Perm (Fin 6)), ∀ i, group (perm i) = [4, 6, 8, 10, 12, 14].get i

def skateparkCondition (group : FriendGroup) : Prop :=
  ∃ i j, i ≠ j ∧ group i + group j = 18

def swimmingPoolCondition (group : FriendGroup) : Prop :=
  ∃ i j, i ≠ j ∧ group i < 12 ∧ group j < 12

def libraryCondition (group : FriendGroup) (lucyIndex : Fin 6) : Prop :=
  ∃ i, i ≠ lucyIndex ∧ group i = 6

theorem lucy_age (group : FriendGroup) (lucyIndex : Fin 6) :
  validAges group →
  skateparkCondition group →
  swimmingPoolCondition group →
  libraryCondition group lucyIndex →
  group lucyIndex = 12 :=
by sorry

end NUMINAMATH_CALUDE_lucy_age_l1487_148756


namespace NUMINAMATH_CALUDE_triangle_tan_A_l1487_148704

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a/b = (b + √3c)/a and sin C = 2√3 sin B, then tan A = √3/3 -/
theorem triangle_tan_A (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  A > 0 → A < π → B > 0 → B < π → C > 0 → C < π →
  A + B + C = π →
  (a / b = (b + Real.sqrt 3 * c) / a) →
  (Real.sin C = 2 * Real.sqrt 3 * Real.sin B) →
  Real.tan A = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_tan_A_l1487_148704


namespace NUMINAMATH_CALUDE_remainder_problem_l1487_148730

theorem remainder_problem (d : ℕ) (r : ℕ) (h1 : d > 1) 
  (h2 : 1237 % d = r)
  (h3 : 1694 % d = r)
  (h4 : 2791 % d = r) :
  d - r = 134 := by sorry

end NUMINAMATH_CALUDE_remainder_problem_l1487_148730


namespace NUMINAMATH_CALUDE_mike_percentage_l1487_148773

def phone_cost : ℝ := 1300
def additional_needed : ℝ := 780

theorem mike_percentage : 
  (phone_cost - additional_needed) / phone_cost * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_mike_percentage_l1487_148773


namespace NUMINAMATH_CALUDE_tank_capacity_l1487_148791

theorem tank_capacity (C : ℚ) 
  (h1 : (3/4 : ℚ) * C + 4 = (7/8 : ℚ) * C) : C = 32 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l1487_148791


namespace NUMINAMATH_CALUDE_work_completion_time_l1487_148774

/-- Proves that if A is thrice as fast as B and together they can do a work in 15 days, 
    then A alone can do the work in 20 days. -/
theorem work_completion_time 
  (a b : ℝ) -- Work rates of A and B
  (h1 : a = 3 * b) -- A is thrice as fast as B
  (h2 : (a + b) * 15 = 1) -- Together, A and B can do the work in 15 days
  : a * 20 = 1 := by -- A alone can do the work in 20 days
sorry


end NUMINAMATH_CALUDE_work_completion_time_l1487_148774


namespace NUMINAMATH_CALUDE_arithmetic_sequence_l1487_148736

/-- Given real numbers x, y, and z satisfying the equation (z-x)^2 - 4(x-y)(y-z) = 0,
    prove that 2y = x + z, which implies that x, y, and z form an arithmetic sequence. -/
theorem arithmetic_sequence (x y z : ℝ) (h : (z - x)^2 - 4*(x - y)*(y - z) = 0) :
  2*y = x + z := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_l1487_148736


namespace NUMINAMATH_CALUDE_no_power_of_three_and_five_l1487_148711

def sequence_v : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 8 * sequence_v (n + 1) - sequence_v n

theorem no_power_of_three_and_five :
  ∀ n : ℕ, ¬∃ (a b : ℕ+), sequence_v n = 3^(a:ℕ) * 5^(b:ℕ) := by
  sorry

end NUMINAMATH_CALUDE_no_power_of_three_and_five_l1487_148711


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1487_148734

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  s : ℕ → ℤ  -- The sum of the first n terms
  first_term : a 1 = -7
  third_sum : s 3 = -15

/-- Properties of the arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n : ℕ, seq.a n = 2 * n - 9) ∧
  (∀ n : ℕ, seq.s n = (n - 4)^2 - 16) ∧
  (∀ n : ℕ, seq.s n ≥ -16) ∧
  seq.s 4 = -16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1487_148734


namespace NUMINAMATH_CALUDE_tractors_moved_l1487_148799

/-- Represents the farming field scenario -/
structure FarmingField where
  initialTractors : ℕ
  initialDays : ℕ
  initialHectaresPerDay : ℕ
  remainingTractors : ℕ
  remainingDays : ℕ
  remainingHectaresPerDay : ℕ

/-- The theorem stating the number of tractors moved -/
theorem tractors_moved (field : FarmingField)
  (h1 : field.initialTractors = 6)
  (h2 : field.initialDays = 4)
  (h3 : field.initialHectaresPerDay = 120)
  (h4 : field.remainingTractors = 4)
  (h5 : field.remainingDays = 5)
  (h6 : field.remainingHectaresPerDay = 144)
  (h7 : field.initialTractors * field.initialDays * field.initialHectaresPerDay =
        field.remainingTractors * field.remainingDays * field.remainingHectaresPerDay) :
  field.initialTractors - field.remainingTractors = 2 := by
  sorry

#check tractors_moved

end NUMINAMATH_CALUDE_tractors_moved_l1487_148799


namespace NUMINAMATH_CALUDE_bakery_pies_relation_l1487_148793

/-- The number of pies Mcgee's Bakery sold -/
def mcgees_pies : ℕ := 16

/-- The number of pies Smith's Bakery sold -/
def smiths_pies : ℕ := 70

/-- The difference between Smith's pies and the multiple of Mcgee's pies -/
def difference : ℕ := 6

/-- The multiple of Mcgee's pies related to Smith's pies -/
def multiple : ℕ := 4

theorem bakery_pies_relation :
  multiple * mcgees_pies + difference = smiths_pies :=
by sorry

end NUMINAMATH_CALUDE_bakery_pies_relation_l1487_148793


namespace NUMINAMATH_CALUDE_first_number_is_seven_l1487_148794

/-- A sequence of 8 numbers where each number starting from the third
    is the sum of the two previous numbers. -/
def FibonacciLikeSequence (a : Fin 8 → ℕ) : Prop :=
  ∀ i : Fin 8, i.val ≥ 2 → a i = a (Fin.sub i 1) + a (Fin.sub i 2)

/-- Theorem stating that if the 5th number is 53 and the 8th number is 225
    in a Fibonacci-like sequence of 8 numbers, then the 1st number is 7. -/
theorem first_number_is_seven
  (a : Fin 8 → ℕ)
  (h_seq : FibonacciLikeSequence a)
  (h_fifth : a 4 = 53)
  (h_eighth : a 7 = 225) :
  a 0 = 7 := by
  sorry


end NUMINAMATH_CALUDE_first_number_is_seven_l1487_148794


namespace NUMINAMATH_CALUDE_min_area_with_prime_dimension_l1487_148719

/-- Represents a rectangle with integer dimensions. -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Checks if a number is prime. -/
def isPrime (n : ℕ) : Prop := Nat.Prime n

/-- Calculates the perimeter of a rectangle. -/
def perimeter (r : Rectangle) : ℕ := 2 * (r.length + r.width)

/-- Calculates the area of a rectangle. -/
def area (r : Rectangle) : ℕ := r.length * r.width

/-- Checks if at least one dimension of a rectangle is prime. -/
def hasOnePrimeDimension (r : Rectangle) : Prop := isPrime r.length ∨ isPrime r.width

/-- The main theorem stating the minimum area of a rectangle with given conditions. -/
theorem min_area_with_prime_dimension :
  ∀ r : Rectangle,
    r.length > 0 ∧ r.width > 0 →
    perimeter r = 120 →
    hasOnePrimeDimension r →
    (∀ r' : Rectangle, 
      r'.length > 0 ∧ r'.width > 0 →
      perimeter r' = 120 →
      hasOnePrimeDimension r' →
      area r ≤ area r') →
    area r = 116 :=
sorry

end NUMINAMATH_CALUDE_min_area_with_prime_dimension_l1487_148719


namespace NUMINAMATH_CALUDE_annual_income_calculation_l1487_148712

theorem annual_income_calculation (total : ℝ) (p1 : ℝ) (rate1 : ℝ) (rate2 : ℝ)
  (h1 : total = 2500)
  (h2 : p1 = 500.0000000000002)
  (h3 : rate1 = 0.05)
  (h4 : rate2 = 0.06) :
  let p2 := total - p1
  let income1 := p1 * rate1
  let income2 := p2 * rate2
  income1 + income2 = 145 := by sorry

end NUMINAMATH_CALUDE_annual_income_calculation_l1487_148712


namespace NUMINAMATH_CALUDE_sports_club_tennis_players_l1487_148727

/-- Given a sports club with the following properties:
  * There are 80 total members
  * 48 members play badminton
  * 7 members play neither badminton nor tennis
  * 21 members play both badminton and tennis
  Prove that 46 members play tennis -/
theorem sports_club_tennis_players (total : ℕ) (badminton : ℕ) (neither : ℕ) (both : ℕ) 
  (h1 : total = 80)
  (h2 : badminton = 48)
  (h3 : neither = 7)
  (h4 : both = 21) :
  total - neither - (badminton - both) = 46 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_tennis_players_l1487_148727


namespace NUMINAMATH_CALUDE_average_value_sequence_l1487_148783

theorem average_value_sequence (x : ℝ) : 
  let sequence := [0, 3*x, 6*x, 12*x, 24*x]
  (sequence.sum / sequence.length : ℝ) = 9*x := by
  sorry

end NUMINAMATH_CALUDE_average_value_sequence_l1487_148783


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l1487_148702

/-- Represents the number of people in each age group -/
structure Population :=
  (elderly : ℕ)
  (middleAged : ℕ)
  (young : ℕ)

/-- Represents the number of people to be sampled from each age group -/
structure Sample :=
  (elderly : ℕ)
  (middleAged : ℕ)
  (young : ℕ)

/-- The total number of people in the population -/
def totalPopulation (p : Population) : ℕ :=
  p.elderly + p.middleAged + p.young

/-- The total number of people in the sample -/
def totalSample (s : Sample) : ℕ :=
  s.elderly + s.middleAged + s.young

/-- Checks if the sample is proportionally representative of the population -/
def isProportionalSample (p : Population) (s : Sample) : Prop :=
  s.elderly * totalPopulation p = p.elderly * totalSample s ∧
  s.middleAged * totalPopulation p = p.middleAged * totalSample s ∧
  s.young * totalPopulation p = p.young * totalSample s

theorem stratified_sampling_theorem (p : Population) (s : Sample) :
  p.elderly = 27 →
  p.middleAged = 54 →
  p.young = 81 →
  totalSample s = 42 →
  isProportionalSample p s →
  s.elderly = 7 ∧ s.middleAged = 14 ∧ s.young = 21 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l1487_148702


namespace NUMINAMATH_CALUDE_largest_number_of_cubic_roots_l1487_148785

theorem largest_number_of_cubic_roots (p q r : ℝ) 
  (sum_eq : p + q + r = 3)
  (sum_prod_eq : p * q + p * r + q * r = -6)
  (prod_eq : p * q * r = -8) :
  max p (max q r) = (1 + Real.sqrt 17) / 2 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_of_cubic_roots_l1487_148785


namespace NUMINAMATH_CALUDE_problem_solution_l1487_148710

-- Define the function f
def f (a x : ℝ) : ℝ := |x - 4*a| + |x|

-- Theorem statement
theorem problem_solution :
  ∀ a : ℝ, (∀ x : ℝ, f a x ≥ a^2) →
  (-4 ≤ a ∧ a ≤ 4) ∧
  (∃ min_value : ℝ, min_value = 16/21 ∧
    ∀ x y z : ℝ, 4*x + 2*y + z = 4 →
      (x + y)^2 + y^2 + z^2 ≥ min_value) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1487_148710


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1487_148752

theorem trigonometric_identity (α : ℝ) : 
  (Real.sin α)^2 + (Real.cos (π/6 + α))^2 + (Real.sin α) * (Real.cos (π/6 + α)) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1487_148752


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_problem_l1487_148739

/-- Arithmetic sequence term -/
def a_n (a b n : ℕ) : ℕ := a + (n - 1) * b

/-- Geometric sequence term -/
def b_n (a b n : ℕ) : ℕ := b * a^(n - 1)

/-- C_n sequence term -/
def C_n (a b n : ℕ) : ℕ := a_n a b (n + 1) + b_n a b n

theorem arithmetic_geometric_sequence_problem 
  (a b : ℕ) 
  (h_a_pos : a > 1) 
  (h_b_pos : b > 1) 
  (h_a1_lt_b1 : a < b) 
  (h_b2_lt_a3 : b * a < a + 2 * b) 
  (h_exists_m : ∀ n : ℕ, n > 0 → ∃ m : ℕ, m > 0 ∧ a_n a b m + 3 = b_n a b n) :
  (a = 2 ∧ b = 5) ∧ 
  (b = 4 → 
    ∃ n : ℕ, C_n a b n = 18 ∧ C_n a b (n + 1) = 30 ∧ C_n a b (n + 2) = 50 ∧
    ∀ k : ℕ, k ≠ n → ¬(C_n a b k * C_n a b (k + 2) = (C_n a b (k + 1))^2)) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_problem_l1487_148739


namespace NUMINAMATH_CALUDE_d_squared_plus_5d_l1487_148776

theorem d_squared_plus_5d (d : ℤ) : d = 5 + 6 → d^2 + 5*d = 176 := by
  sorry

end NUMINAMATH_CALUDE_d_squared_plus_5d_l1487_148776


namespace NUMINAMATH_CALUDE_proposition_implications_l1487_148766

def p (a : ℝ) : Prop := 1 ∈ {x : ℝ | x^2 < a}
def q (a : ℝ) : Prop := 2 ∈ {x : ℝ | x^2 < a}

theorem proposition_implications (a : ℝ) :
  ((p a ∨ q a) → a > 1) ∧ ((p a ∧ q a) → a > 4) := by sorry

end NUMINAMATH_CALUDE_proposition_implications_l1487_148766


namespace NUMINAMATH_CALUDE_biased_die_expected_value_l1487_148708

/-- The expected value of winnings for a biased die roll -/
theorem biased_die_expected_value :
  let p_six : ℚ := 1/4  -- Probability of rolling a 6
  let p_other : ℚ := 3/4  -- Probability of rolling any other number
  let win_six : ℚ := 4  -- Winnings for rolling a 6
  let lose_other : ℚ := -1  -- Loss for rolling any other number
  p_six * win_six + p_other * lose_other = 1/4 := by
sorry

end NUMINAMATH_CALUDE_biased_die_expected_value_l1487_148708


namespace NUMINAMATH_CALUDE_perfect_square_condition_l1487_148729

/-- Given a quadratic expression of the form 16x^2 - bx + 9, 
    prove that it is a perfect square trinomial if and only if b = ±24 -/
theorem perfect_square_condition (b : ℝ) : 
  (∃ (k : ℝ), ∀ (x : ℝ), 16 * x^2 - b * x + 9 = (k * x + 3)^2) ↔ (b = 24 ∨ b = -24) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l1487_148729


namespace NUMINAMATH_CALUDE_uki_earnings_l1487_148769

/-- Represents Uki's bakery business -/
structure BakeryBusiness where
  cupcake_price : ℝ
  cookie_price : ℝ
  biscuit_price : ℝ
  daily_cupcakes : ℕ
  daily_cookies : ℕ
  daily_biscuits : ℕ

/-- Calculates the total earnings for a given number of days -/
def total_earnings (b : BakeryBusiness) (days : ℕ) : ℝ :=
  (b.cupcake_price * b.daily_cupcakes + 
   b.cookie_price * b.daily_cookies + 
   b.biscuit_price * b.daily_biscuits) * days

/-- Theorem stating that Uki's total earnings for five days is $350 -/
theorem uki_earnings : ∃ (b : BakeryBusiness), 
  b.cupcake_price = 1.5 ∧ 
  b.cookie_price = 2 ∧ 
  b.biscuit_price = 1 ∧ 
  b.daily_cupcakes = 20 ∧ 
  b.daily_cookies = 10 ∧ 
  b.daily_biscuits = 20 ∧ 
  total_earnings b 5 = 350 := by
  sorry

end NUMINAMATH_CALUDE_uki_earnings_l1487_148769


namespace NUMINAMATH_CALUDE_james_training_sessions_l1487_148797

/-- James' training schedule -/
structure TrainingSchedule where
  hoursPerSession : ℕ
  daysOffPerWeek : ℕ
  totalHoursPerYear : ℕ

/-- Calculate the number of training sessions per day -/
def sessionsPerDay (schedule : TrainingSchedule) : ℚ :=
  let daysPerWeek : ℕ := 7
  let weeksPerYear : ℕ := 52
  let trainingDaysPerYear : ℕ := (daysPerWeek - schedule.daysOffPerWeek) * weeksPerYear
  let hoursPerDay : ℚ := schedule.totalHoursPerYear / trainingDaysPerYear
  hoursPerDay / schedule.hoursPerSession

/-- Theorem: James trains 2 times per day -/
theorem james_training_sessions (james : TrainingSchedule) 
  (h1 : james.hoursPerSession = 4)
  (h2 : james.daysOffPerWeek = 2)
  (h3 : james.totalHoursPerYear = 2080) : 
  sessionsPerDay james = 2 := by
  sorry


end NUMINAMATH_CALUDE_james_training_sessions_l1487_148797


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1487_148790

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - x ≥ 0) ↔ (∃ x₀ : ℝ, x₀^2 - x₀ < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1487_148790


namespace NUMINAMATH_CALUDE_intersection_point_of_linear_system_l1487_148798

theorem intersection_point_of_linear_system (b : ℝ) :
  let eq1 : ℝ → ℝ → Prop := λ x y => x + y - b = 0
  let eq2 : ℝ → ℝ → Prop := λ x y => 3 * x + y - 2 = 0
  let line1 : ℝ → ℝ → Prop := λ x y => y = -x + b
  let line2 : ℝ → ℝ → Prop := λ x y => y = -3 * x + 2
  (∃ m, eq1 (-1) m ∧ eq2 (-1) m) →
  (∃! p : ℝ × ℝ, line1 p.1 p.2 ∧ line2 p.1 p.2 ∧ p = (-1, 5)) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_of_linear_system_l1487_148798


namespace NUMINAMATH_CALUDE_intersection_line_equation_l1487_148733

-- Define the two given lines
def l₁ (x y : ℝ) : Prop := 2 * x - y + 7 = 0
def l₂ (x y : ℝ) : Prop := y = 1 - x

-- Define the intersection point
def intersection_point (x y : ℝ) : Prop := l₁ x y ∧ l₂ x y

-- Define the line passing through the intersection point and origin
def target_line (x y : ℝ) : Prop := 3 * x + 2 * y = 0

-- Theorem statement
theorem intersection_line_equation :
  ∃ (x₀ y₀ : ℝ), intersection_point x₀ y₀ ∧
  ∀ (x y : ℝ), (x = 0 ∧ y = 0) ∨ (x = x₀ ∧ y = y₀) → target_line x y :=
sorry

end NUMINAMATH_CALUDE_intersection_line_equation_l1487_148733


namespace NUMINAMATH_CALUDE_expression_evaluation_l1487_148717

theorem expression_evaluation (x : ℝ) (h : x = 2) : 
  (2*x - 1)^2 + (x + 3)*(x - 3) - 4*(x - 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1487_148717


namespace NUMINAMATH_CALUDE_book_arrangement_count_l1487_148772

/-- Represents the number of books -/
def n : ℕ := 6

/-- Represents the number of ways to arrange books A and B at the ends -/
def end_arrangements : ℕ := 2

/-- Represents the number of ways to order books C and D -/
def cd_orders : ℕ := 2

/-- Represents the number of ways to arrange the C-D pair and the other 2 books in the middle -/
def middle_arrangements : ℕ := 6

/-- The total number of valid arrangements -/
def total_arrangements : ℕ := end_arrangements * cd_orders * middle_arrangements

theorem book_arrangement_count :
  total_arrangements = 24 :=
sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l1487_148772


namespace NUMINAMATH_CALUDE_multiples_of_three_is_closed_l1487_148707

def is_closed (A : Set ℤ) : Prop :=
  ∀ a b : ℤ, a ∈ A → b ∈ A → (a + b) ∈ A ∧ (a - b) ∈ A

def multiples_of_three : Set ℤ :=
  {n : ℤ | ∃ k : ℤ, n = 3 * k}

theorem multiples_of_three_is_closed :
  is_closed multiples_of_three :=
by
  sorry

end NUMINAMATH_CALUDE_multiples_of_three_is_closed_l1487_148707


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1487_148723

/-- An isosceles triangle with two sides of length 12 and a third side of length 17 has a perimeter of 41. -/
theorem isosceles_triangle_perimeter : ℝ → Prop :=
  fun perimeter =>
    ∃ (a b c : ℝ),
      a = 12 ∧ b = 12 ∧ c = 17 ∧  -- Two sides are 12, third side is 17
      (a = b ∨ a = c ∨ b = c) ∧   -- Definition of isosceles triangle
      perimeter = a + b + c ∧     -- Definition of perimeter
      perimeter = 41              -- The perimeter we want to prove

/-- Proof of the theorem -/
lemma proof_isosceles_triangle_perimeter : isosceles_triangle_perimeter 41 := by
  sorry

#check proof_isosceles_triangle_perimeter

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1487_148723


namespace NUMINAMATH_CALUDE_sequence_periodicity_l1487_148720

theorem sequence_periodicity (a : ℕ → ℚ) (h1 : ∀ n : ℕ, |a (n + 1) - 2 * a n| = 2)
    (h2 : ∀ n : ℕ, |a n| ≤ 2) :
  ∃ k l : ℕ, k < l ∧ a k = a l :=
sorry

end NUMINAMATH_CALUDE_sequence_periodicity_l1487_148720


namespace NUMINAMATH_CALUDE_simplify_expression_l1487_148735

theorem simplify_expression (x y : ℝ) (h : y = Real.sqrt (x - 2) + Real.sqrt (2 - x) + 2) :
  |y - Real.sqrt 3| - (x - 2 + Real.sqrt 2)^2 = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1487_148735


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1487_148744

theorem complex_equation_solution (a b : ℝ) (z : ℂ) :
  (∀ x : ℂ, x^2 + (4 + Complex.I) * x + 4 + a * Complex.I = 0 → x.im = 0) →
  z = a + b * Complex.I →
  (b : ℂ)^2 + (4 + Complex.I) * b + 4 + a * Complex.I = 0 →
  z = 2 - 2 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1487_148744


namespace NUMINAMATH_CALUDE_erin_has_90_dollars_l1487_148705

/-- The amount of money Erin has after emptying all machines in her launderette -/
def erins_money_after_emptying (quarters_per_machine : ℕ) (dimes_per_machine : ℕ) (num_machines : ℕ) : ℚ :=
  (quarters_per_machine * (25 : ℚ) / 100 + dimes_per_machine * (10 : ℚ) / 100) * num_machines

/-- Theorem stating that Erin will have $90.00 after emptying all machines -/
theorem erin_has_90_dollars :
  erins_money_after_emptying 80 100 3 = 90 :=
by sorry

end NUMINAMATH_CALUDE_erin_has_90_dollars_l1487_148705


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l1487_148788

theorem quadratic_one_solution (p : ℚ) : 
  (∃! x, 3 * x^2 - 7 * x + p = 0) ↔ p = 49/12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l1487_148788


namespace NUMINAMATH_CALUDE_difference_even_prime_sums_l1487_148709

def sumFirstNEvenNumbers (n : ℕ) : ℕ := n * (n + 1)

def sumFirstNPrimes (n : ℕ) : ℕ := sorry

theorem difference_even_prime_sums : 
  sumFirstNEvenNumbers 3005 - sumFirstNPrimes 3005 = 9039030 - sumFirstNPrimes 3005 := by
  sorry

end NUMINAMATH_CALUDE_difference_even_prime_sums_l1487_148709


namespace NUMINAMATH_CALUDE_solve_system_l1487_148787

theorem solve_system (p q : ℚ) 
  (eq1 : 5 * p + 6 * q = 10) 
  (eq2 : 6 * p + 5 * q = 17) : 
  p = 52 / 11 := by
sorry

end NUMINAMATH_CALUDE_solve_system_l1487_148787


namespace NUMINAMATH_CALUDE_factor_expression_l1487_148718

theorem factor_expression (x : ℝ) : 16 * x^2 + 8 * x = 8 * x * (2 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1487_148718


namespace NUMINAMATH_CALUDE_four_correct_propositions_l1487_148780

theorem four_correct_propositions 
  (a b c : ℝ) : 
  (((a < b → a + c < b + c) ∧                   -- Original proposition
    ((a + c < b + c) → (a < b)) ∧               -- Converse
    ((a ≥ b) → (a + c ≥ b + c)) ∧               -- Inverse
    ((a + c ≥ b + c) → (a ≥ b))) →              -- Contrapositive
   (4 = (Bool.toNat (a < b → a + c < b + c) +
         Bool.toNat ((a + c < b + c) → (a < b)) +
         Bool.toNat ((a ≥ b) → (a + c ≥ b + c)) +
         Bool.toNat ((a + c ≥ b + c) → (a ≥ b))))) :=
by sorry

end NUMINAMATH_CALUDE_four_correct_propositions_l1487_148780


namespace NUMINAMATH_CALUDE_reese_practice_hours_l1487_148762

/-- Calculates the total piano practice hours for Reese over a given number of months -/
def total_practice_hours (months : ℕ) : ℕ :=
  let initial_weekly_hours := 4
  let initial_months := 2
  let increased_weekly_hours := 5
  let workshop_hours := 3
  
  let initial_practice := initial_weekly_hours * 4 * min months initial_months
  let increased_practice := increased_weekly_hours * 4 * max (months - initial_months) 0
  let total_workshops := months * workshop_hours
  
  initial_practice + increased_practice + total_workshops

/-- Theorem stating that Reese's total practice hours after 5 months is 107 -/
theorem reese_practice_hours : total_practice_hours 5 = 107 := by
  sorry

end NUMINAMATH_CALUDE_reese_practice_hours_l1487_148762


namespace NUMINAMATH_CALUDE_smallest_valid_integer_l1487_148743

def decimal_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def binary_sum (n : ℕ) : ℕ :=
  n.digits 2 |>.sum

def is_valid (n : ℕ) : Prop :=
  1000 < n ∧ n < 2000 ∧ decimal_sum n = binary_sum n

theorem smallest_valid_integer : 
  (∀ m, 1000 < m ∧ m < 1101 → ¬(is_valid m)) ∧ is_valid 1101 := by
  sorry

end NUMINAMATH_CALUDE_smallest_valid_integer_l1487_148743


namespace NUMINAMATH_CALUDE_total_walking_hours_l1487_148759

/-- Represents the types of dogs Charlotte walks -/
inductive DogType
  | Poodle
  | Chihuahua
  | Labrador

/-- Represents a day of the week -/
inductive Day
  | Monday
  | Tuesday
  | Wednesday

/-- Returns the number of hours it takes to walk a dog of a given type -/
def walkingTime (d : DogType) : Nat :=
  match d with
  | DogType.Poodle => 2
  | DogType.Chihuahua => 1
  | DogType.Labrador => 3

/-- Returns the number of dogs of a given type walked on a specific day -/
def dogsWalked (day : Day) (dogType : DogType) : Nat :=
  match day, dogType with
  | Day.Monday, DogType.Poodle => 4
  | Day.Monday, DogType.Chihuahua => 2
  | Day.Monday, DogType.Labrador => 0
  | Day.Tuesday, DogType.Poodle => 4
  | Day.Tuesday, DogType.Chihuahua => 2
  | Day.Tuesday, DogType.Labrador => 0
  | Day.Wednesday, DogType.Poodle => 0
  | Day.Wednesday, DogType.Chihuahua => 0
  | Day.Wednesday, DogType.Labrador => 4

/-- Calculates the total hours spent walking dogs on a given day -/
def hoursPerDay (day : Day) : Nat :=
  (dogsWalked day DogType.Poodle * walkingTime DogType.Poodle) +
  (dogsWalked day DogType.Chihuahua * walkingTime DogType.Chihuahua) +
  (dogsWalked day DogType.Labrador * walkingTime DogType.Labrador)

/-- Theorem stating that the total hours for dog-walking this week is 32 -/
theorem total_walking_hours :
  hoursPerDay Day.Monday + hoursPerDay Day.Tuesday + hoursPerDay Day.Wednesday = 32 := by
  sorry


end NUMINAMATH_CALUDE_total_walking_hours_l1487_148759


namespace NUMINAMATH_CALUDE_min_max_sum_l1487_148761

theorem min_max_sum (x y z u v : ℕ+) (h : x + y + z + u + v = 2505) :
  let N := max (x + y) (max (y + z) (max (z + u) (u + v)))
  N ≥ 1253 ∧ ∃ (a b c d e : ℕ+), a + b + c + d + e = 2505 ∧
    max (a + b) (max (b + c) (max (c + d) (d + e))) = 1253 := by
  sorry

end NUMINAMATH_CALUDE_min_max_sum_l1487_148761


namespace NUMINAMATH_CALUDE_system_solution_ratio_l1487_148714

/-- Given a system of linear equations with non-zero solutions x, y, and z,
    prove that xz/y^2 = 175 -/
theorem system_solution_ratio (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
  (eq1 : x + (95/3)*y + 4*z = 0)
  (eq2 : 4*x + (95/3)*y - 3*z = 0)
  (eq3 : 3*x + 5*y - 4*z = 0) :
  x*z/y^2 = 175 := by
sorry


end NUMINAMATH_CALUDE_system_solution_ratio_l1487_148714


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1487_148784

def is_geometric_sequence (α : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, α (n + 1) = α n * r

theorem geometric_sequence_property 
  (α : ℕ → ℝ) 
  (h_geometric : is_geometric_sequence α) 
  (h_product : α 4 * α 5 * α 6 = 27) : 
  α 5 = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1487_148784


namespace NUMINAMATH_CALUDE_second_grade_sample_size_l1487_148703

/-- Given a total sample size and ratios for three grades, calculate the number of students to be drawn from a specific grade -/
def stratified_sample (total_sample : ℕ) (ratio1 ratio2 ratio3 : ℕ) (grade : ℕ) : ℕ :=
  let total_ratio := ratio1 + ratio2 + ratio3
  let grade_ratio := match grade with
    | 1 => ratio1
    | 2 => ratio2
    | 3 => ratio3
    | _ => 0
  (grade_ratio * total_sample) / total_ratio

/-- Theorem stating that for a sample size of 50 and ratios 3:3:4, the second grade should have 15 students -/
theorem second_grade_sample_size :
  stratified_sample 50 3 3 4 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_second_grade_sample_size_l1487_148703


namespace NUMINAMATH_CALUDE_jaron_snickers_needed_l1487_148747

/-- The number of Snickers bars Jaron needs to sell to win the Nintendo Switch -/
def snickers_needed (total_points_needed : ℕ) (bunnies_sold : ℕ) (points_per_bunny : ℕ) (points_per_snickers : ℕ) : ℕ :=
  ((total_points_needed - bunnies_sold * points_per_bunny) + points_per_snickers - 1) / points_per_snickers

theorem jaron_snickers_needed :
  snickers_needed 2000 8 100 25 = 48 := by
  sorry

end NUMINAMATH_CALUDE_jaron_snickers_needed_l1487_148747


namespace NUMINAMATH_CALUDE_vector_perpendicular_condition_l1487_148722

theorem vector_perpendicular_condition (a b : ℝ × ℝ) (m : ℝ) : 
  ‖a‖ = 3 →
  ‖b‖ = 2 →
  a • b = 3 →
  (a - m • b) • a = 0 →
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_vector_perpendicular_condition_l1487_148722


namespace NUMINAMATH_CALUDE_a_4_equals_4_l1487_148749

/-- Given a sequence {aₙ} defined by aₙ = (-1)ⁿ n, prove that a₄ = 4 -/
theorem a_4_equals_4 (a : ℕ → ℤ) (h : ∀ n, a n = (-1)^n * n) : a 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_a_4_equals_4_l1487_148749


namespace NUMINAMATH_CALUDE_composite_sum_of_powers_l1487_148754

-- Define the problem statement
theorem composite_sum_of_powers (a b c d : ℕ+) (h : a * b = c * d) :
  ∃ (m n : ℕ), m > 1 ∧ n > 1 ∧ a^2016 + b^2016 + c^2016 + d^2016 = m * n :=
by sorry


end NUMINAMATH_CALUDE_composite_sum_of_powers_l1487_148754


namespace NUMINAMATH_CALUDE_max_candies_eaten_l1487_148764

/-- Represents the board with numbers -/
structure Board :=
  (numbers : List Nat)

/-- Represents Karlson's candy-eating process -/
def process (b : Board) : Nat :=
  let n := b.numbers.length
  n * (n - 1) / 2

/-- The initial board with 37 ones -/
def initial_board : Board :=
  { numbers := List.replicate 37 1 }

/-- The theorem stating the maximum number of candies Karlson can eat -/
theorem max_candies_eaten (b : Board := initial_board) : 
  process b = 666 := by
  sorry

end NUMINAMATH_CALUDE_max_candies_eaten_l1487_148764


namespace NUMINAMATH_CALUDE_unique_complex_solution_l1487_148765

theorem unique_complex_solution :
  ∃! z : ℂ, Complex.abs z < 20 ∧ Complex.exp z = 1 - z / 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_complex_solution_l1487_148765
