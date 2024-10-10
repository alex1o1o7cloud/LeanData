import Mathlib

namespace bowl_capacity_ratio_l1354_135449

theorem bowl_capacity_ratio :
  ∀ (capacity_1 capacity_2 : ℕ),
    capacity_1 < capacity_2 →
    capacity_2 = 600 →
    capacity_1 + capacity_2 = 1050 →
    (capacity_1 : ℚ) / capacity_2 = 3 / 4 := by
  sorry

end bowl_capacity_ratio_l1354_135449


namespace cars_per_salesperson_per_month_l1354_135411

/-- Proves that given 500 cars for sale, 10 sales professionals, and a 5-month period to sell all cars, each salesperson sells 10 cars per month. -/
theorem cars_per_salesperson_per_month 
  (total_cars : ℕ) 
  (sales_professionals : ℕ) 
  (months_to_sell : ℕ) 
  (h1 : total_cars = 500) 
  (h2 : sales_professionals = 10) 
  (h3 : months_to_sell = 5) :
  total_cars / (sales_professionals * months_to_sell) = 10 :=
by
  sorry

end cars_per_salesperson_per_month_l1354_135411


namespace grid_sum_property_l1354_135436

def Grid := Matrix (Fin 2) (Fin 3) ℕ

def is_valid_grid (g : Grid) : Prop :=
  ∀ i j, 1 ≤ g i j ∧ g i j ≤ 9 ∧
  (∀ i₁ i₂ j, i₁ ≠ i₂ → g i₁ j ≠ g i₂ j) ∧
  (∀ i j₁ j₂, j₁ ≠ j₂ → g i j₁ ≠ g i j₂)

theorem grid_sum_property (g : Grid) (h : is_valid_grid g) :
  (g 0 0 + g 0 1 + g 0 2 = 23) →
  (g 0 0 + g 1 0 = 14) →
  (g 0 1 + g 1 1 = 16) →
  (g 0 2 + g 1 2 = 17) →
  g 1 0 + 2 * g 1 1 + 3 * g 1 2 = 49 := by
sorry

end grid_sum_property_l1354_135436


namespace remaining_work_days_l1354_135493

/-- Given two workers x and y, where x can finish a job in 36 days and y in 24 days,
    prove that x needs 18 days to finish the remaining work after y worked for 12 days. -/
theorem remaining_work_days (x_days y_days y_worked_days : ℕ) 
  (hx : x_days = 36) (hy : y_days = 24) (hw : y_worked_days = 12) : 
  (x_days : ℚ) / 2 = 18 := by
  sorry

#check remaining_work_days

end remaining_work_days_l1354_135493


namespace tangent_problem_l1354_135403

theorem tangent_problem (α β : Real) 
  (h1 : Real.tan (α/2 + β) = 1/2) 
  (h2 : Real.tan (β - α/2) = 1/3) : 
  Real.tan α = 1/7 := by
  sorry

end tangent_problem_l1354_135403


namespace dog_walking_homework_diff_l1354_135440

/-- Represents the time in minutes for various activities -/
structure ActivityTimes where
  total : ℕ
  homework : ℕ
  cleaning : ℕ
  trash : ℕ
  remaining : ℕ

/-- Calculates the time spent walking the dog -/
def walkingTime (t : ActivityTimes) : ℕ :=
  t.total - t.remaining - (t.homework + t.cleaning + t.trash)

/-- Theorem stating the difference between dog walking and homework time -/
theorem dog_walking_homework_diff (t : ActivityTimes) : 
  t.total = 120 ∧ 
  t.homework = 30 ∧ 
  t.cleaning = t.homework / 2 ∧ 
  t.trash = t.homework / 6 ∧ 
  t.remaining = 35 → 
  walkingTime t - t.homework = 5 := by
  sorry


end dog_walking_homework_diff_l1354_135440


namespace two_men_absent_l1354_135477

/-- Represents the work completion scenario -/
structure WorkCompletion where
  total_men : ℕ
  planned_days : ℕ
  actual_days : ℕ

/-- Calculates the number of absent men given the work completion scenario -/
def calculate_absent_men (w : WorkCompletion) : ℕ :=
  w.total_men - (w.total_men * w.planned_days) / w.actual_days

/-- Theorem stating that 2 men became absent in the given scenario -/
theorem two_men_absent (w : WorkCompletion) 
  (h1 : w.total_men = 22)
  (h2 : w.planned_days = 20)
  (h3 : w.actual_days = 22) : 
  calculate_absent_men w = 2 := by
  sorry

#eval calculate_absent_men ⟨22, 20, 22⟩

end two_men_absent_l1354_135477


namespace equation_solution_l1354_135491

theorem equation_solution : ∃ x : ℚ, (2 / 7) * (1 / 3) * x = 14 ∧ x = 147 := by
  sorry

end equation_solution_l1354_135491


namespace system_solution_l1354_135438

theorem system_solution : ∃ (x y z : ℝ), 
  (x + y = 1) ∧ (x + z = 0) ∧ (y + z = -1) ∧ 
  (x = 1) ∧ (y = 0) ∧ (z = -1) := by
  sorry

end system_solution_l1354_135438


namespace complex_number_in_first_quadrant_l1354_135464

theorem complex_number_in_first_quadrant : 
  let z : ℂ := (2 - Complex.I) / (1 - 2 * Complex.I)
  (z.re > 0) ∧ (z.im > 0) := by sorry

end complex_number_in_first_quadrant_l1354_135464


namespace nora_muffin_sales_l1354_135494

/-- The number of cases of muffins Nora needs to sell to raise $120 -/
def cases_needed (packs_per_case : ℕ) (muffins_per_pack : ℕ) (price_per_muffin : ℕ) (target_amount : ℕ) : ℕ :=
  target_amount / (packs_per_case * muffins_per_pack * price_per_muffin)

/-- Proof that Nora needs to sell 5 cases of muffins to raise $120 -/
theorem nora_muffin_sales :
  cases_needed 3 4 2 120 = 5 := by
  sorry

end nora_muffin_sales_l1354_135494


namespace no_perfect_squares_l1354_135486

theorem no_perfect_squares (n : ℕ+) : 
  ¬(∃ (a b c : ℕ), (2 * n^2 + 1 = a^2) ∧ (3 * n^2 + 1 = b^2) ∧ (6 * n^2 + 1 = c^2)) := by
  sorry

end no_perfect_squares_l1354_135486


namespace spinner_probability_l1354_135424

theorem spinner_probability (p_A p_B p_C p_D : ℚ) : 
  p_A = 1/4 → p_B = 1/3 → p_D = 1/6 → p_A + p_B + p_C + p_D = 1 → p_C = 1/4 := by
  sorry

end spinner_probability_l1354_135424


namespace sufficient_but_not_necessary_l1354_135432

theorem sufficient_but_not_necessary (x : ℝ) : 
  (∀ x, x > 2 → x^2 - 3*x + 2 > 0) ∧ 
  (∃ x, x^2 - 3*x + 2 > 0 ∧ ¬(x > 2)) := by
  sorry

end sufficient_but_not_necessary_l1354_135432


namespace least_positive_period_is_36_l1354_135447

-- Define the property of the function
def has_property (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 6) + f (x - 6) = f x

-- Define the concept of a period for a function
def is_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f x

-- State the theorem
theorem least_positive_period_is_36 (f : ℝ → ℝ) (h : has_property f) :
  (∃ p : ℝ, p > 0 ∧ is_period f p) →
  (∀ q : ℝ, q > 0 → is_period f q → q ≥ 36) ∧ is_period f 36 :=
sorry

end least_positive_period_is_36_l1354_135447


namespace square_root_property_l1354_135428

theorem square_root_property (p k : ℕ) (hp : Nat.Prime p) (hp_odd : Odd p) 
  (hk : k > 0) (h_sqrt : ∃ n : ℕ, n > 0 ∧ n^2 = k^2 - p*k) : 
  k = (p + 1)^2 / 4 := by
  sorry

end square_root_property_l1354_135428


namespace marbles_in_larger_container_l1354_135497

/-- Given that a container with a volume of 24 cm³ can hold 75 marbles,
    prove that a container with a volume of 72 cm³ can hold 225 marbles,
    assuming the ratio of marbles to volume is constant. -/
theorem marbles_in_larger_container (v₁ v₂ : ℝ) (m₁ m₂ : ℕ) 
    (h₁ : v₁ = 24) (h₂ : m₁ = 75) (h₃ : v₂ = 72) :
    (m₁ : ℝ) / v₁ = m₂ / v₂ → m₂ = 225 := by
  sorry

end marbles_in_larger_container_l1354_135497


namespace smallest_shift_l1354_135466

-- Define a periodic function g with period 30
def g (x : ℝ) : ℝ := sorry

-- State the periodicity of g
axiom g_periodic (x : ℝ) : g (x + 30) = g x

-- Define the property we want to prove
def property (b : ℝ) : Prop :=
  ∀ x, g ((x - b) / 3) = g (x / 3)

-- State the theorem
theorem smallest_shift :
  (∃ b > 0, property b) ∧ 
  (∀ b > 0, property b → b ≥ 90) ∧
  property 90 := by sorry

end smallest_shift_l1354_135466


namespace unique_periodic_modulus_l1354_135488

/-- The binomial coefficient C(n,k) -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The sequence x_n = C(2n, n) -/
def x_seq (n : ℕ) : ℕ := binomial (2 * n) n

/-- A sequence is eventually periodic modulo m if there exist positive integers N and T
    such that for all n ≥ N, x_(n+T) ≡ x_n (mod m) -/
def eventually_periodic_mod (x : ℕ → ℕ) (m : ℕ) : Prop :=
  ∃ (N T : ℕ), T > 0 ∧ ∀ n ≥ N, x (n + T) % m = x n % m

/-- The main theorem: 2 is the only positive integer h > 1 such that 
    the sequence x_n = C(2n, n) is eventually periodic modulo h -/
theorem unique_periodic_modulus :
  ∀ h : ℕ, h > 1 → (eventually_periodic_mod x_seq h ↔ h = 2) := by sorry

end unique_periodic_modulus_l1354_135488


namespace value_of_a_l1354_135476

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1/x + a*x^3

theorem value_of_a : 
  ∀ a : ℝ, (deriv (f a)) 1 = 5 → a = 2 := by
  sorry

end value_of_a_l1354_135476


namespace basket_replacement_theorem_l1354_135471

/-- The number of people who entered the stadium before the basket needed replacement -/
def people_entered : ℕ :=
  sorry

/-- The number of placards each person takes -/
def placards_per_person : ℕ := 2

/-- The total number of placards the basket can hold -/
def basket_capacity : ℕ := 823

theorem basket_replacement_theorem :
  people_entered = 411 ∧
  people_entered * placards_per_person < basket_capacity ∧
  (people_entered + 1) * placards_per_person > basket_capacity :=
by sorry

end basket_replacement_theorem_l1354_135471


namespace smallest_four_digit_pascal_l1354_135499

/-- Pascal's triangle is represented as a function from row and column to natural number -/
def pascal : ℕ → ℕ → ℕ
  | 0, _ => 1
  | n + 1, 0 => 1
  | n + 1, k + 1 => pascal n k + pascal n (k + 1)

/-- A number is four-digit if it's between 1000 and 9999 inclusive -/
def isFourDigit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem smallest_four_digit_pascal :
  ∃ (r c : ℕ), isFourDigit (pascal r c) ∧
    ∀ (r' c' : ℕ), isFourDigit (pascal r' c') → pascal r c ≤ pascal r' c' :=
  sorry

end smallest_four_digit_pascal_l1354_135499


namespace unique_quadratic_function_l1354_135439

/-- A quadratic function of the form f(x) = x^2 + ax + b -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^2 + a*x + b

/-- The theorem stating the unique quadratic function satisfying the given condition -/
theorem unique_quadratic_function (a b : ℝ) :
  (∀ x, (f a b (f a b x - x)) / (f a b x) = x^2 + 2023*x + 1777) →
  a = 2025 ∧ b = 249 := by
  sorry

end unique_quadratic_function_l1354_135439


namespace pizza_varieties_count_l1354_135446

/-- The number of base pizza flavors -/
def base_flavors : ℕ := 4

/-- The number of topping combinations (including no additional toppings) -/
def topping_combinations : ℕ := 4

/-- Calculates the total number of pizza varieties -/
def total_varieties : ℕ := base_flavors * topping_combinations

theorem pizza_varieties_count :
  total_varieties = 16 := by
  sorry

end pizza_varieties_count_l1354_135446


namespace pure_imaginary_product_l1354_135453

theorem pure_imaginary_product (b : ℝ) : 
  let z₁ : ℂ := 1 + Complex.I
  let z₂ : ℂ := 1 - b * Complex.I
  (z₁ * z₂).re = 0 ∧ (z₁ * z₂).im ≠ 0 → b = -1 := by
  sorry

end pure_imaginary_product_l1354_135453


namespace four_color_theorem_l1354_135406

/-- Represents a map as a planar graph -/
structure Map where
  vertices : Set Nat
  edges : Set (Nat × Nat)
  is_planar : Bool

/-- A coloring of a map -/
def Coloring (m : Map) := Nat → Fin 4

/-- Checks if a coloring is valid for a given map -/
def is_valid_coloring (m : Map) (c : Coloring m) : Prop :=
  ∀ (v₁ v₂ : Nat), (v₁, v₂) ∈ m.edges → c v₁ ≠ c v₂

/-- The Four Color Theorem -/
theorem four_color_theorem (m : Map) (h : m.is_planar = true) :
  ∃ (c : Coloring m), is_valid_coloring m c :=
sorry

end four_color_theorem_l1354_135406


namespace pascal_triangle_first_25_rows_sum_l1354_135451

/-- The number of elements in the nth row of Pascal's Triangle -/
def pascal_row_elements (n : ℕ) : ℕ := n + 1

/-- The sum of elements in the first n rows of Pascal's Triangle -/
def pascal_triangle_sum (n : ℕ) : ℕ := (n + 1) * (n + 2) / 2

theorem pascal_triangle_first_25_rows_sum :
  pascal_triangle_sum 24 = 325 := by
  sorry

end pascal_triangle_first_25_rows_sum_l1354_135451


namespace expression_evaluation_l1354_135480

/-- Proves that the given expression evaluates to -8 when a = 2 and b = -1 -/
theorem expression_evaluation :
  let a : ℤ := 2
  let b : ℤ := -1
  3 * (2 * a^2 * b - 3 * a * b^2 - 1) - 2 * (3 * a^2 * b - 4 * a * b^2 + 1) - 1 = -8 :=
by sorry

end expression_evaluation_l1354_135480


namespace paint_used_approximation_l1354_135454

/-- The amount of paint Joe starts with in gallons -/
def initial_paint : ℝ := 720

/-- The fraction of paint used in the first week -/
def first_week_fraction : ℚ := 2/7

/-- The fraction of remaining paint used in the second week -/
def second_week_fraction : ℚ := 3/8

/-- The fraction of remaining paint used in the third week -/
def third_week_fraction : ℚ := 5/11

/-- The fraction of remaining paint used in the fourth week -/
def fourth_week_fraction : ℚ := 4/13

/-- The total amount of paint used after four weeks -/
def total_paint_used : ℝ :=
  let first_week := initial_paint * (first_week_fraction : ℝ)
  let second_week := (initial_paint - first_week) * (second_week_fraction : ℝ)
  let third_week := (initial_paint - first_week - second_week) * (third_week_fraction : ℝ)
  let fourth_week := (initial_paint - first_week - second_week - third_week) * (fourth_week_fraction : ℝ)
  first_week + second_week + third_week + fourth_week

/-- Theorem stating that the total paint used is approximately 598.620 gallons -/
theorem paint_used_approximation : 
  598.619 < total_paint_used ∧ total_paint_used < 598.621 :=
sorry

end paint_used_approximation_l1354_135454


namespace complex_power_of_four_l1354_135498

theorem complex_power_of_four :
  (3 * Complex.cos (30 * Real.pi / 180) + 3 * Complex.I * Complex.sin (30 * Real.pi / 180)) ^ 4 =
  -40.5 + 40.5 * Complex.I * Real.sqrt 3 := by
  sorry

end complex_power_of_four_l1354_135498


namespace problem_1_problem_2_l1354_135487

-- Problem 1
theorem problem_1 : (-1)^3 + Real.sqrt 4 - (2 - Real.sqrt 2)^0 = 0 := by sorry

-- Problem 2
theorem problem_2 (a : ℝ) : (a + 3) * (a - 3) - a * (a - 2) = 2 * a - 9 := by sorry

end problem_1_problem_2_l1354_135487


namespace min_sum_proof_l1354_135468

/-- The minimum sum of m and n satisfying the conditions -/
def min_sum : ℕ := 106

/-- The value of m in the minimal solution -/
def m_min : ℕ := 3

/-- The value of n in the minimal solution -/
def n_min : ℕ := 103

/-- Checks if two numbers are congruent modulo 1000 -/
def congruent_mod_1000 (a b : ℕ) : Prop :=
  a % 1000 = b % 1000

theorem min_sum_proof :
  ∀ m n : ℕ,
    n > m →
    m ≥ 1 →
    congruent_mod_1000 (1978^n) (1978^m) →
    m + n ≥ min_sum ∧
    (m + n = min_sum → m = m_min ∧ n = n_min) :=
by sorry

end min_sum_proof_l1354_135468


namespace inequalities_equivalence_l1354_135444

theorem inequalities_equivalence (x : ℝ) :
  (2 * (x + 1) - 1 < 3 * x + 2 ↔ x > -1) ∧
  ((x + 3) / 2 - 1 ≥ (2 * x - 3) / 3 ↔ x ≤ 9) :=
by sorry

end inequalities_equivalence_l1354_135444


namespace complex_division_example_l1354_135422

/-- The imaginary unit -/
def i : ℂ := Complex.I

/-- Theorem stating that the complex number 2i/(1+i) equals 1+i -/
theorem complex_division_example : (2 * i) / (1 + i) = 1 + i := by
  sorry

end complex_division_example_l1354_135422


namespace negation_of_existence_negation_of_quadratic_inequality_l1354_135484

theorem negation_of_existence (f : ℝ → Prop) :
  (¬ ∃ x : ℝ, f x) ↔ ∀ x : ℝ, ¬ f x := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 - 2*x + 3 > 0) ↔ (∀ x : ℝ, x^2 - 2*x + 3 ≤ 0) := by sorry

end negation_of_existence_negation_of_quadratic_inequality_l1354_135484


namespace isosceles_right_triangle_area_l1354_135496

theorem isosceles_right_triangle_area (leg : ℝ) (h_leg : leg = 3) :
  let triangle_area := (1 / 2) * leg * leg
  triangle_area = 4.5 := by
  sorry

end isosceles_right_triangle_area_l1354_135496


namespace polynomial_divisibility_l1354_135415

theorem polynomial_divisibility (r s : ℝ) : 
  (∀ x : ℝ, (x - 2) * (x + 1) ∣ (x^6 - x^5 + 3*x^4 - r*x^3 + s*x^2 + 3*x - 7)) ↔ 
  (r = 33/4 ∧ s = -13/4) := by
sorry

end polynomial_divisibility_l1354_135415


namespace expression_simplification_l1354_135462

theorem expression_simplification (x : ℝ) (h : x^2 - 2*x - 2 = 0) :
  ((x - 1) / x - (x - 2) / (x + 1)) / ((2*x^2 - x) / (x^2 + 2*x + 1)) = 1/2 := by
  sorry

end expression_simplification_l1354_135462


namespace andrey_gifts_l1354_135485

theorem andrey_gifts :
  ∃ (n : ℕ) (a : ℕ),
    n > 2 ∧
    n * (n - 2) = a * (n - 1) + 16 ∧
    n = 18 :=
by sorry

end andrey_gifts_l1354_135485


namespace sqrt_ab_is_integer_l1354_135413

theorem sqrt_ab_is_integer (a b n : ℕ+) 
  (h : (a : ℚ) / b = ((a : ℚ)^2 + (n : ℚ)^2) / ((b : ℚ)^2 + (n : ℚ)^2)) : 
  ∃ k : ℕ, k^2 = a * b := by
sorry

end sqrt_ab_is_integer_l1354_135413


namespace infinite_sum_n_over_n4_plus_1_l1354_135457

/-- The infinite sum of n / (n^4 + 1) from n = 1 to infinity equals 1. -/
theorem infinite_sum_n_over_n4_plus_1 : 
  ∑' n : ℕ+, (n : ℝ) / ((n : ℝ)^4 + 1) = 1 :=
sorry

end infinite_sum_n_over_n4_plus_1_l1354_135457


namespace G_of_two_eq_six_l1354_135467

noncomputable def G (x : ℝ) : ℝ :=
  1.2 * Real.sqrt (abs (x + 1.5)) + (7 / Real.pi) * Real.arctan (1.1 * Real.sqrt (abs (x + 1.5)))

theorem G_of_two_eq_six : G 2 = 6 := by sorry

end G_of_two_eq_six_l1354_135467


namespace quadratic_roots_to_coefficients_l1354_135495

theorem quadratic_roots_to_coefficients :
  ∀ (b c : ℝ), 
    (∀ x : ℝ, x^2 + b*x + c = 0 ↔ x = 1 ∨ x = -2) →
    b = 1 ∧ c = -2 := by
  sorry

end quadratic_roots_to_coefficients_l1354_135495


namespace smallest_sum_of_reciprocals_l1354_135419

theorem smallest_sum_of_reciprocals (x y : ℕ+) : 
  x ≠ y → 
  (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 18 → 
  (∀ a b : ℕ+, a ≠ b → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 18 → (x : ℤ) + y ≤ (a : ℤ) + b) →
  (x : ℤ) + y = 75 :=
by sorry

end smallest_sum_of_reciprocals_l1354_135419


namespace sum_of_numbers_l1354_135412

theorem sum_of_numbers (a b : ℕ) (h : (a + b) * (a - b) = 1996) : a + b = 998 := by
  sorry

end sum_of_numbers_l1354_135412


namespace ryan_load_is_correct_l1354_135452

/-- The number of packages Sarah's trucks can carry in one load -/
def sarah_load : ℕ := 18

/-- The total number of packages shipped by both services -/
def total_packages : ℕ := 198

/-- Predicate to check if a number is a valid load size for Ryan's trucks -/
def is_valid_ryan_load (n : ℕ) : Prop :=
  n > sarah_load ∧ total_packages % n = 0

/-- The number of packages Ryan's trucks can carry in one load -/
def ryan_load : ℕ := 22

theorem ryan_load_is_correct : 
  is_valid_ryan_load ryan_load ∧ 
  ∀ (n : ℕ), is_valid_ryan_load n → n ≥ ryan_load :=
sorry

end ryan_load_is_correct_l1354_135452


namespace sqrt_t6_plus_t4_l1354_135465

theorem sqrt_t6_plus_t4 (t : ℝ) : Real.sqrt (t^6 + t^4) = t^2 * Real.sqrt (t^2 + 1) := by
  sorry

end sqrt_t6_plus_t4_l1354_135465


namespace cube_root_of_four_fifth_powers_l1354_135400

theorem cube_root_of_four_fifth_powers (x : ℝ) :
  x = (5^6 + 5^6 + 5^6 + 5^6)^(1/3) → x = 25 * 2^(2/3) := by
  sorry

end cube_root_of_four_fifth_powers_l1354_135400


namespace trig_values_150_degrees_l1354_135427

/-- Given a point P on the unit circle corresponding to an angle of 150°, 
    prove that tan(150°) = -√3 and sin(150°) = √3/2 -/
theorem trig_values_150_degrees : 
  ∀ (P : ℝ × ℝ), 
  (P.1^2 + P.2^2 = 1) →  -- P is on the unit circle
  (P.1 = -1/2 ∧ P.2 = Real.sqrt 3 / 2) →  -- P corresponds to 150°
  (Real.tan (150 * π / 180) = -Real.sqrt 3 ∧ 
   Real.sin (150 * π / 180) = Real.sqrt 3 / 2) := by
  sorry

end trig_values_150_degrees_l1354_135427


namespace diameter_length_l1354_135441

/-- Represents a circle with diameter AB and perpendicular chord CD -/
structure Circle where
  AB : ℕ
  CD : ℕ
  is_two_digit : 10 ≤ AB ∧ AB < 100
  is_reversed : CD = (AB % 10) * 10 + (AB / 10)

/-- The distance OH is rational -/
def rational_OH (c : Circle) : Prop :=
  ∃ (q : ℚ), q > 0 ∧ q^2 * 4 = 99 * (c.AB / 10 - c.AB % 10) * (c.AB / 10 + c.AB % 10)

theorem diameter_length (c : Circle) (h : rational_OH c) : c.AB = 65 :=
sorry

end diameter_length_l1354_135441


namespace probability_of_same_group_l1354_135459

def card_count : ℕ := 20
def people_count : ℕ := 4
def first_drawn : ℕ := 5
def second_drawn : ℕ := 14

def same_group_probability : ℚ := 7 / 51

theorem probability_of_same_group :
  let remaining_cards := card_count - people_count + 2
  let favorable_outcomes := (card_count - second_drawn) * (card_count - second_drawn - 1) +
                            (first_drawn - 1) * (first_drawn - 2)
  let total_outcomes := remaining_cards * (remaining_cards - 1)
  (favorable_outcomes : ℚ) / total_outcomes = same_group_probability :=
sorry

end probability_of_same_group_l1354_135459


namespace hexagram_shell_placement_l1354_135478

def hexagram_arrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

theorem hexagram_shell_placement :
  hexagram_arrangements 12 = 39916800 := by
  sorry

end hexagram_shell_placement_l1354_135478


namespace brads_red_balloons_l1354_135437

/-- Given that Brad has a total of 17 balloons and 9 of them are green,
    prove that he has 8 red balloons. -/
theorem brads_red_balloons (total : ℕ) (green : ℕ) (h1 : total = 17) (h2 : green = 9) :
  total - green = 8 := by
  sorry

end brads_red_balloons_l1354_135437


namespace simplify_and_evaluate_l1354_135433

theorem simplify_and_evaluate (x : ℝ) (h : x = 5) :
  (x + 3) / (x^2 - 4) / (2 - (x + 1) / (x + 2)) = 1 / 3 := by
  sorry

end simplify_and_evaluate_l1354_135433


namespace tv_weight_difference_l1354_135420

/-- The difference in weight between two TVs with given dimensions -/
theorem tv_weight_difference : 
  let bill_length : ℕ := 48
  let bill_width : ℕ := 100
  let bob_length : ℕ := 70
  let bob_width : ℕ := 60
  let weight_per_sq_inch : ℚ := 4 / 1
  let oz_per_pound : ℕ := 16
  let bill_area : ℕ := bill_length * bill_width
  let bob_area : ℕ := bob_length * bob_width
  let bill_weight_oz : ℚ := bill_area * weight_per_sq_inch
  let bob_weight_oz : ℚ := bob_area * weight_per_sq_inch
  let bill_weight_lbs : ℚ := bill_weight_oz / oz_per_pound
  let bob_weight_lbs : ℚ := bob_weight_oz / oz_per_pound
  bill_weight_lbs - bob_weight_lbs = 150
  := by sorry

end tv_weight_difference_l1354_135420


namespace transmission_time_is_128_seconds_l1354_135405

/-- The number of blocks to be sent -/
def num_blocks : ℕ := 80

/-- The number of chunks in each block -/
def chunks_per_block : ℕ := 256

/-- The transmission rate in chunks per second -/
def transmission_rate : ℕ := 160

/-- The time it takes to send all blocks in seconds -/
def transmission_time : ℕ := num_blocks * chunks_per_block / transmission_rate

theorem transmission_time_is_128_seconds : transmission_time = 128 := by
  sorry

end transmission_time_is_128_seconds_l1354_135405


namespace arithmetic_sequence_contains_2017_l1354_135469

/-- An arithmetic sequence containing 25, 41, and 65 also contains 2017 -/
theorem arithmetic_sequence_contains_2017 (a₁ d : ℤ) (k n m : ℕ) 
  (h_pos : d > 0)
  (h_25 : 25 = a₁ + k * d)
  (h_41 : 41 = a₁ + n * d)
  (h_65 : 65 = a₁ + m * d) :
  ∃ l : ℕ, 2017 = a₁ + l * d :=
sorry

end arithmetic_sequence_contains_2017_l1354_135469


namespace original_to_half_ratio_l1354_135483

theorem original_to_half_ratio (x : ℝ) (h : x / 2 = 9) : x / (x / 2) = 2 := by
  sorry

end original_to_half_ratio_l1354_135483


namespace quadratic_polynomial_property_l1354_135461

/-- A quadratic polynomial of the form x^2 - (p+q)x + pq -/
def QuadraticPolynomial (p q : ℝ) : ℝ → ℝ := fun x ↦ x^2 - (p+q)*x + p*q

/-- The composite function p(p(x)) -/
def CompositePolynomial (p q : ℝ) : ℝ → ℝ :=
  fun x ↦ let px := QuadraticPolynomial p q x
          (QuadraticPolynomial p q) px

/-- Predicate that checks if a polynomial has exactly four distinct real roots -/
def HasFourDistinctRealRoots (f : ℝ → ℝ) : Prop :=
  ∃ (a b c d : ℝ), (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧
    (∀ x, f x = 0 ↔ x = a ∨ x = b ∨ x = c ∨ x = d)

/-- The theorem to be proved -/
theorem quadratic_polynomial_property :
  ∃ (p q : ℝ),
    HasFourDistinctRealRoots (CompositePolynomial p q) ∧
    (∀ (p' q' : ℝ),
      HasFourDistinctRealRoots (CompositePolynomial p' q') →
      (let f := QuadraticPolynomial p q
       let f' := QuadraticPolynomial p' q'
       ∀ (a b c d : ℝ),
         f a = a → f b = b → f c = c → f d = d →
         ∀ (a' b' c' d' : ℝ),
           f' a' = a' → f' b' = b' → f' c' = c' → f' d' = d' →
           a * b * c * d ≥ a' * b' * c' * d')) →
    QuadraticPolynomial p q 0 = 0 := by
  sorry

end quadratic_polynomial_property_l1354_135461


namespace largest_package_size_and_cost_l1354_135448

def lucas_notebooks : ℕ := 36
def maria_notebooks : ℕ := 60
def package_cost : ℕ := 3

theorem largest_package_size_and_cost :
  let max_package_size := Nat.gcd lucas_notebooks maria_notebooks
  (max_package_size = 12) ∧ (package_cost = 3) := by
  sorry

end largest_package_size_and_cost_l1354_135448


namespace sphere_radius_in_cube_l1354_135434

/-- The radius of spheres packed in a cube -/
theorem sphere_radius_in_cube (n : ℕ) (side_length : ℝ) (radius : ℝ) : 
  n = 8 →  -- There are 8 spheres
  side_length = 2 →  -- The cube has side length 2
  radius > 0 →  -- The radius is positive
  (2 * radius = side_length / 2 + radius) →  -- Condition for spheres to be tangent
  radius = 1 := by sorry

end sphere_radius_in_cube_l1354_135434


namespace packing_peanuts_calculation_l1354_135475

/-- The amount of packing peanuts (in grams) needed for each large order -/
def large_order_peanuts : ℕ := sorry

/-- The total amount of packing peanuts (in grams) used -/
def total_peanuts : ℕ := 800

/-- The number of large orders -/
def num_large_orders : ℕ := 3

/-- The number of small orders -/
def num_small_orders : ℕ := 4

/-- The amount of packing peanuts (in grams) needed for each small order -/
def small_order_peanuts : ℕ := 50

theorem packing_peanuts_calculation :
  large_order_peanuts * num_large_orders + small_order_peanuts * num_small_orders = total_peanuts ∧
  large_order_peanuts = 200 := by sorry

end packing_peanuts_calculation_l1354_135475


namespace inverse_42_mod_53_l1354_135489

theorem inverse_42_mod_53 (h : (11⁻¹ : ZMod 53) = 31) : (42⁻¹ : ZMod 53) = 22 := by
  sorry

end inverse_42_mod_53_l1354_135489


namespace combined_research_degrees_l1354_135414

def total_percentage : ℝ := 100
def microphotonics_percentage : ℝ := 10
def home_electronics_percentage : ℝ := 24
def food_additives_percentage : ℝ := 15
def genetically_modified_microorganisms_percentage : ℝ := 29
def industrial_lubricants_percentage : ℝ := 8
def nanotechnology_percentage : ℝ := 7

def basic_astrophysics_percentage : ℝ :=
  total_percentage - (microphotonics_percentage + home_electronics_percentage + 
  food_additives_percentage + genetically_modified_microorganisms_percentage + 
  industrial_lubricants_percentage + nanotechnology_percentage)

def combined_percentage : ℝ := basic_astrophysics_percentage + nanotechnology_percentage

def degrees_in_circle : ℝ := 360

theorem combined_research_degrees :
  combined_percentage * (degrees_in_circle / total_percentage) = 50.4 := by
  sorry

end combined_research_degrees_l1354_135414


namespace unique_digit_divisibility_l1354_135435

theorem unique_digit_divisibility : ∃! (B : ℕ), B < 10 ∧ 45 % B = 0 ∧ (451 * 10 + B * 1 + 7) % 3 = 0 := by
  sorry

end unique_digit_divisibility_l1354_135435


namespace smallest_k_for_two_trailing_zeros_l1354_135456

theorem smallest_k_for_two_trailing_zeros : ∃ k : ℕ+, k = 13 ∧ 
  (∀ m : ℕ+, m < k → ¬(100 ∣ Nat.choose (2 * m) m)) ∧ 
  (100 ∣ Nat.choose (2 * k) k) := by
  sorry

end smallest_k_for_two_trailing_zeros_l1354_135456


namespace special_function_value_l1354_135450

/-- A function satisfying f(x + y) = f(x) + f(y) + 2xy for all real x and y -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y + 2 * x * y

theorem special_function_value :
  ∀ f : ℝ → ℝ, special_function f → f 1 = 2 → f (-3) = 6 := by
  sorry

end special_function_value_l1354_135450


namespace fraction_nonnegative_l1354_135417

theorem fraction_nonnegative (x : ℝ) (h : x ≠ 3) : x^2 / (x - 3)^2 ≥ 0 := by
  sorry

end fraction_nonnegative_l1354_135417


namespace chord_with_midpoint_A_no_chord_with_midpoint_B_l1354_135429

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

-- Define a chord of the hyperbola
def is_chord (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  hyperbola x₁ y₁ ∧ hyperbola x₂ y₂

-- Define the midpoint of a chord
def is_midpoint (x y x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x = (x₁ + x₂) / 2 ∧ y = (y₁ + y₂) / 2

-- Theorem 1: Chord with midpoint A(2,1)
theorem chord_with_midpoint_A :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    is_chord x₁ y₁ x₂ y₂ ∧
    is_midpoint 2 1 x₁ y₁ x₂ y₂ ∧
    ∀ (x y : ℝ), y = 6*x - 11 ↔ ((x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂)) :=
sorry

-- Theorem 2: No chord with midpoint B(1,1)
theorem no_chord_with_midpoint_B :
  ¬∃ (x₁ y₁ x₂ y₂ : ℝ),
    is_chord x₁ y₁ x₂ y₂ ∧
    is_midpoint 1 1 x₁ y₁ x₂ y₂ :=
sorry

end chord_with_midpoint_A_no_chord_with_midpoint_B_l1354_135429


namespace intersection_sum_l1354_135431

theorem intersection_sum (a b m : ℝ) : 
  ((-m + a = 8) ∧ (m + b = 8)) → a + b = 16 := by
  sorry

end intersection_sum_l1354_135431


namespace train_crossing_time_l1354_135410

/-- Proves that a train with given length and speed takes the specified time to cross an electric pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 120 →
  train_speed_kmh = 144 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 3 := by
  sorry

#check train_crossing_time

end train_crossing_time_l1354_135410


namespace line_segment_parameter_sum_of_squares_l1354_135423

/-- Given a line segment connecting (1, -3) and (-4, 9), parameterized by x = at + b and y = ct + d
    where 0 ≤ t ≤ 1 and t = 0 corresponds to (1, -3), prove that a^2 + b^2 + c^2 + d^2 = 179 -/
theorem line_segment_parameter_sum_of_squares :
  ∀ (a b c d : ℝ),
  (∀ t : ℝ, 0 ≤ t → t ≤ 1 → ∃ x y : ℝ, x = a * t + b ∧ y = c * t + d) →
  (b = 1 ∧ d = -3) →
  (a + b = -4 ∧ c + d = 9) →
  a^2 + b^2 + c^2 + d^2 = 179 := by
sorry

end line_segment_parameter_sum_of_squares_l1354_135423


namespace magic_square_theorem_l1354_135409

/-- A type representing a 3x3 grid -/
def Grid := Fin 3 → Fin 3 → ℤ

/-- The set of numbers to be used in the grid -/
def GridNumbers : Finset ℤ := {-3, -2, -1, 0, 1, 2, 3, 4, 5}

/-- The sum of each row, column, and diagonal is equal -/
def is_magic (g : Grid) : Prop :=
  let sum := g 0 0 + g 0 1 + g 0 2
  ∀ i j, (i = j → g i 0 + g i 1 + g i 2 = sum) ∧
         (i = j → g 0 j + g 1 j + g 2 j = sum) ∧
         ((i = 0 ∧ j = 0) → g 0 0 + g 1 1 + g 2 2 = sum) ∧
         ((i = 0 ∧ j = 2) → g 0 2 + g 1 1 + g 2 0 = sum)

/-- The theorem to be proved -/
theorem magic_square_theorem (g : Grid) 
  (h1 : g 0 0 = -2)
  (h2 : g 0 2 = 0)
  (h3 : g 2 2 = 4)
  (h4 : is_magic g)
  (h5 : ∀ i j, g i j ∈ GridNumbers)
  (h6 : ∀ x, x ∈ GridNumbers → ∃! i j, g i j = x) :
  ∃ a b c, g 0 1 = a ∧ g 2 1 = b ∧ g 2 0 = c ∧ a - b - c = 4 :=
sorry

end magic_square_theorem_l1354_135409


namespace min_sum_squares_complex_l1354_135407

theorem min_sum_squares_complex (w : ℂ) (h : Complex.abs (w - (3 - 2*I)) = 4) :
  ∃ (min : ℝ), min = 48 ∧ 
  ∀ (z : ℂ), Complex.abs (z - (3 - 2*I)) = 4 →
    Complex.abs (z + (1 + 2*I))^2 + Complex.abs (z - (7 + 2*I))^2 ≥ min :=
by sorry

end min_sum_squares_complex_l1354_135407


namespace sum_of_powers_of_three_l1354_135402

theorem sum_of_powers_of_three : (-3)^4 + (-3)^3 + (-3)^2 + 3^2 + 3^3 + 3^4 = 180 := by
  sorry

end sum_of_powers_of_three_l1354_135402


namespace corner_start_winning_strategy_adjacent_start_winning_strategy_l1354_135408

/-- Represents the players in the game -/
inductive Player
| A
| B

/-- Represents the game state -/
structure GameState where
  n : Nat
  currentPosition : Nat × Nat
  visitedPositions : Set (Nat × Nat)
  currentPlayer : Player

/-- Defines a winning strategy for a player -/
def HasWinningStrategy (player : Player) (initialState : GameState) : Prop :=
  ∃ (strategy : GameState → Nat × Nat),
    ∀ (gameState : GameState),
      gameState.currentPlayer = player →
      (strategy gameState ∉ gameState.visitedPositions) →
      (∃ (nextState : GameState), 
        nextState.currentPosition = strategy gameState ∧
        nextState.visitedPositions = insert gameState.currentPosition gameState.visitedPositions ∧
        nextState.currentPlayer ≠ player)

theorem corner_start_winning_strategy :
  ∀ (n : Nat),
    (n % 2 = 0 → HasWinningStrategy Player.A (GameState.mk n (0, 0) {(0, 0)} Player.A)) ∧
    (n % 2 = 1 → HasWinningStrategy Player.B (GameState.mk n (0, 0) {(0, 0)} Player.A)) :=
sorry

theorem adjacent_start_winning_strategy :
  ∀ (n : Nat) (startPos : Nat × Nat),
    (startPos = (0, 1) ∨ startPos = (1, 0)) →
    HasWinningStrategy Player.A (GameState.mk n startPos {startPos} Player.A) :=
sorry

end corner_start_winning_strategy_adjacent_start_winning_strategy_l1354_135408


namespace consecutive_integers_product_l1354_135472

theorem consecutive_integers_product (a : ℕ) : 
  (a * (a + 1) * (a + 2) * (a + 3) * (a + 4) = 15120) → (a + 4 = 12) :=
by
  sorry

end consecutive_integers_product_l1354_135472


namespace unique_stamp_denomination_l1354_135492

/-- Given a positive integer n, this function checks if a postage value can be formed
    using stamps of denominations 7, n, and n+1 cents. -/
def can_form_postage (n : ℕ+) (postage : ℕ) : Prop :=
  ∃ (a b c : ℕ), postage = 7 * a + n * b + (n + 1) * c

/-- This theorem states that 18 is the unique positive integer n such that,
    given stamps of denominations 7, n, and n+1 cents, 106 cents is the
    greatest postage that cannot be formed. -/
theorem unique_stamp_denomination :
  ∃! (n : ℕ+),
    (¬ can_form_postage n 106) ∧
    (∀ m : ℕ, m > 106 → can_form_postage n m) ∧
    (∀ k : ℕ, k < 106 → ¬ can_form_postage n k → can_form_postage n (k + 1)) :=
by sorry

end unique_stamp_denomination_l1354_135492


namespace tomatoes_left_l1354_135473

/-- Given 21 initial tomatoes and birds eating one-third of them, prove that 14 tomatoes are left -/
theorem tomatoes_left (initial : ℕ) (eaten_fraction : ℚ) (h1 : initial = 21) (h2 : eaten_fraction = 1/3) :
  initial - (initial * eaten_fraction).floor = 14 := by
  sorry

end tomatoes_left_l1354_135473


namespace baseball_league_games_l1354_135418

theorem baseball_league_games (n m : ℕ) : 
  (∃ (g₁ g₂ : Finset (Finset ℕ)), 
    (g₁.card = 4 ∧ g₂.card = 4) ∧ 
    (∀ t₁ ∈ g₁, ∀ t₂ ∈ g₁, t₁ ≠ t₂ → (∃ k : ℕ, k = n)) ∧
    (∀ t₁ ∈ g₁, ∀ t₂ ∈ g₂, (∃ k : ℕ, k = m)) ∧
    n > 2 * m ∧
    m > 4 ∧
    (∃ t ∈ g₁, 3 * n + 4 * m = 76)) →
  n = 48 := by sorry

end baseball_league_games_l1354_135418


namespace maria_oatmeal_cookies_l1354_135482

/-- The number of oatmeal cookies Maria had -/
def num_oatmeal_cookies (cookies_per_bag : ℕ) (num_chocolate_chip : ℕ) (num_baggies : ℕ) : ℕ :=
  num_baggies * cookies_per_bag - num_chocolate_chip

/-- Theorem stating that Maria had 2 oatmeal cookies -/
theorem maria_oatmeal_cookies :
  num_oatmeal_cookies 5 33 7 = 2 := by
  sorry

end maria_oatmeal_cookies_l1354_135482


namespace cost_price_percentage_l1354_135458

theorem cost_price_percentage (cost_price selling_price : ℝ) (profit_percent : ℝ) :
  profit_percent = 150 →
  selling_price = cost_price + (profit_percent / 100) * cost_price →
  (cost_price / selling_price) * 100 = 40 := by
sorry

end cost_price_percentage_l1354_135458


namespace range_of_a_l1354_135445

/-- Definition of the circle D -/
def circle_D (x y : ℝ) : Prop := (x - 3)^2 + (y - 2)^2 = 1

/-- Definition of point B -/
def point_B : ℝ × ℝ := (-1, 0)

/-- Definition of point C -/
def point_C (a : ℝ) : ℝ × ℝ := (a, 0)

/-- Theorem stating the range of a -/
theorem range_of_a (A B C : ℝ × ℝ) (a : ℝ) :
  (∃ x y, A = (x, y) ∧ circle_D x y) →  -- A lies on circle D
  B = point_B →                         -- B is at (-1, 0)
  C = point_C a →                       -- C is at (a, 0)
  (A.1 - B.1) * (A.1 - C.1) + (A.2 - B.2) * (A.2 - C.2) = 0 →  -- Right angle at A
  14/5 ≤ a ∧ a ≤ 16/3 :=                -- Range of a
by sorry

end range_of_a_l1354_135445


namespace quadratic_root_difference_squares_l1354_135421

theorem quadratic_root_difference_squares (b : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁^2 - b*x₁ + 12 = 0 ∧ x₂^2 - b*x₂ + 12 = 0 ∧ x₁^2 - x₂^2 = 7) → 
  b = 7 ∨ b = -7 := by
sorry

end quadratic_root_difference_squares_l1354_135421


namespace difference_of_squares_l1354_135443

theorem difference_of_squares (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) := by
  sorry

end difference_of_squares_l1354_135443


namespace count_valid_m_l1354_135470

theorem count_valid_m : ∃! (S : Finset ℤ), 
  (∀ m ∈ S, (∀ x : ℝ, (3 - 3*x < x - 5 ∧ x - m > -1) ↔ x > 2) ∧ 
             (∃ x : ℕ+, (2*x - m) / 3 = 1)) ∧
  S.card = 3 :=
sorry

end count_valid_m_l1354_135470


namespace volunteer_arrangement_l1354_135474

theorem volunteer_arrangement (n : ℕ) (k : ℕ) : n = 7 ∧ k = 3 → 
  (Nat.choose n k) * (Nat.choose (n - k) k) = 140 := by
  sorry

end volunteer_arrangement_l1354_135474


namespace base_conversion_l1354_135416

theorem base_conversion (b : ℕ) (h1 : b > 0) : 
  (5 * 6 + 2 = b * b + b + 1) → b = 5 := by
  sorry

end base_conversion_l1354_135416


namespace charity_fundraising_contribution_l1354_135490

theorem charity_fundraising_contribution 
  (total_goal : ℝ) 
  (collected : ℝ) 
  (num_people : ℕ) 
  (h1 : total_goal = 2400)
  (h2 : collected = 300)
  (h3 : num_people = 8) :
  (total_goal - collected) / num_people = 262.5 := by
sorry

end charity_fundraising_contribution_l1354_135490


namespace point_in_plane_region_l1354_135460

def in_plane_region (x y : ℝ) : Prop := 2*x + y - 6 < 0

theorem point_in_plane_region :
  in_plane_region 0 1 ∧
  ¬(in_plane_region 5 0) ∧
  ¬(in_plane_region 0 7) ∧
  ¬(in_plane_region 2 3) :=
by sorry

end point_in_plane_region_l1354_135460


namespace mrs_hilt_money_l1354_135425

/-- Mrs. Hilt's pencil purchase problem -/
theorem mrs_hilt_money (pencil_cost remaining_money : ℕ) 
  (h1 : pencil_cost = 11)
  (h2 : remaining_money = 4) : 
  pencil_cost + remaining_money = 15 := by
  sorry

end mrs_hilt_money_l1354_135425


namespace books_read_total_l1354_135426

def total_books (megan kelcie john greg alice : ℝ) : ℝ :=
  megan + kelcie + john + greg + alice

theorem books_read_total :
  ∀ (megan kelcie john greg alice : ℝ),
    megan = 45 →
    kelcie = megan / 3 →
    john = kelcie + 7 →
    greg = 2 * john + 11 →
    alice = 2.5 * greg - 10 →
    total_books megan kelcie john greg alice = 264.5 :=
by
  sorry

end books_read_total_l1354_135426


namespace san_antonio_bound_passes_ten_buses_l1354_135479

/-- Represents the schedule and trip details of buses between Austin and San Antonio -/
structure BusSchedule where
  austin_to_sa_interval : ℕ -- Interval in minutes for Austin to San Antonio buses
  sa_to_austin_interval : ℕ -- Interval in minutes for San Antonio to Austin buses
  sa_to_austin_offset : ℕ   -- Offset in minutes for San Antonio to Austin buses
  trip_duration : ℕ         -- Trip duration in minutes

/-- Calculates the number of buses passed on the highway -/
def buses_passed (schedule : BusSchedule) : ℕ :=
  sorry -- Proof to be implemented

/-- Main theorem: A San Antonio-bound bus passes 10 Austin-bound buses on the highway -/
theorem san_antonio_bound_passes_ten_buses :
  let schedule : BusSchedule := {
    austin_to_sa_interval := 30,
    sa_to_austin_interval := 45,
    sa_to_austin_offset := 15,
    trip_duration := 240  -- 4 hours in minutes
  }
  buses_passed schedule = 10 := by sorry

end san_antonio_bound_passes_ten_buses_l1354_135479


namespace glow_interval_l1354_135455

/-- The time interval between glows of a light, given the total time period and number of glows. -/
theorem glow_interval (total_time : ℕ) (num_glows : ℝ) 
  (h1 : total_time = 4969)
  (h2 : num_glows = 382.2307692307692) :
  ∃ (interval : ℝ), abs (interval - 13) < 0.0000001 ∧ interval = total_time / num_glows :=
sorry

end glow_interval_l1354_135455


namespace inequality_solution_set_l1354_135463

theorem inequality_solution_set (x : ℝ) (h : x ≠ 1) :
  (2 * x - 1) / (x - 1) ≥ 1 ↔ x ≤ 0 ∨ x > 1 :=
by sorry

end inequality_solution_set_l1354_135463


namespace cost_ratio_when_b_tripled_x_halved_l1354_135442

/-- The cost ratio when b is tripled and x is halved in the formula C = at(bx)^6 -/
theorem cost_ratio_when_b_tripled_x_halved (a t b x : ℝ) :
  let original_cost := a * t * (b * x)^6
  let new_cost := a * t * (3 * b * (x / 2))^6
  (new_cost / original_cost) * 100 = 1139.0625 := by
sorry

end cost_ratio_when_b_tripled_x_halved_l1354_135442


namespace fibSum_eq_five_nineteenths_l1354_135404

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- The sum of the infinite series ∑(n=0 to ∞) F_n / 5^n -/
noncomputable def fibSum : ℝ := ∑' n, (fib n : ℝ) / 5^n

/-- Theorem stating that the sum of the infinite series equals 5/19 -/
theorem fibSum_eq_five_nineteenths : fibSum = 5 / 19 := by sorry

end fibSum_eq_five_nineteenths_l1354_135404


namespace trip_savings_l1354_135401

/-- The amount Trip can save by going to the earlier movie. -/
def total_savings (evening_ticket_cost : ℚ) (food_combo_cost : ℚ) 
  (ticket_discount_percent : ℚ) (food_discount_percent : ℚ) : ℚ :=
  (ticket_discount_percent / 100) * evening_ticket_cost + 
  (food_discount_percent / 100) * food_combo_cost

/-- Proof that Trip can save $7 by going to the earlier movie. -/
theorem trip_savings : 
  total_savings 10 10 20 50 = 7 := by
  sorry

end trip_savings_l1354_135401


namespace sequence_problem_l1354_135481

-- Define the arithmetic sequence
def is_arithmetic_sequence (x y z w : ℝ) : Prop :=
  y - x = z - y ∧ z - y = w - z

-- Define the geometric sequence
def is_geometric_sequence (x y z w v : ℝ) : Prop :=
  ∃ r : ℝ, y = x * r ∧ z = y * r ∧ w = z * r ∧ v = w * r

theorem sequence_problem (a b c d e : ℝ) :
  is_arithmetic_sequence (-1) a b (-4) →
  is_geometric_sequence (-1) c d e (-4) →
  c = -1 * Real.sqrt 2 :=
by sorry

end sequence_problem_l1354_135481


namespace reservoir_capacity_proof_l1354_135430

theorem reservoir_capacity_proof (current_amount : ℝ) (normal_level : ℝ) (total_capacity : ℝ) 
  (h1 : current_amount = 14)
  (h2 : current_amount = 2 * normal_level)
  (h3 : current_amount = 0.7 * total_capacity) :
  total_capacity - normal_level = 13 := by
sorry

end reservoir_capacity_proof_l1354_135430
