import Mathlib

namespace NUMINAMATH_CALUDE_binomial_square_derivation_l2326_232607

theorem binomial_square_derivation (x y : ℝ) :
  ∃ (a b : ℝ), (-1/2 * x + y) * (y + 1/2 * x) = a^2 - b^2 :=
sorry

end NUMINAMATH_CALUDE_binomial_square_derivation_l2326_232607


namespace NUMINAMATH_CALUDE_complex_ratio_theorem_l2326_232631

theorem complex_ratio_theorem (z₁ z₂ : ℂ) 
  (h₁ : Complex.abs z₁ = 3) 
  (h₂ : Complex.abs z₂ = 5) 
  (h₃ : Complex.abs (z₁ - z₂) = 7) : 
  z₁ / z₂ = (3 / 5 : ℂ) * (-1 / 2 + Complex.I * Real.sqrt 3 / 2) ∨
  z₁ / z₂ = (3 / 5 : ℂ) * (-1 / 2 - Complex.I * Real.sqrt 3 / 2) :=
sorry

end NUMINAMATH_CALUDE_complex_ratio_theorem_l2326_232631


namespace NUMINAMATH_CALUDE_parallel_and_perpendicular_properties_l2326_232687

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines
variable (parallel : Line → Line → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem parallel_and_perpendicular_properties
  (a b c : Line) (γ : Plane) :
  (parallel a b ∧ parallel b c → parallel a c) ∧
  (perpendicular a γ ∧ perpendicular b γ → parallel a b) :=
sorry

end NUMINAMATH_CALUDE_parallel_and_perpendicular_properties_l2326_232687


namespace NUMINAMATH_CALUDE_minimize_sum_of_squares_l2326_232639

/-- The quadratic equation in x has only integer roots -/
def has_integer_roots (k : ℚ) : Prop :=
  ∃ x₁ x₂ : ℤ, k * x₁^2 + (3 - 3*k) * x₁ + (2*k - 6) = 0 ∧
              k * x₂^2 + (3 - 3*k) * x₂ + (2*k - 6) = 0

/-- The quadratic equation in y has two positive integer roots -/
def has_positive_integer_roots (k t : ℚ) : Prop :=
  ∃ y₁ y₂ : ℕ, (k + 3) * y₁^2 - 15 * y₁ + t = 0 ∧
              (k + 3) * y₂^2 - 15 * y₂ + t = 0 ∧
              y₁ ≠ y₂

theorem minimize_sum_of_squares (k t : ℚ) :
  has_integer_roots k →
  has_positive_integer_roots k t →
  (k = 3/4 ∧ t = 15) →
  ∃ y₁ y₂ : ℕ, (k + 3) * y₁^2 - 15 * y₁ + t = 0 ∧
              (k + 3) * y₂^2 - 15 * y₂ + t = 0 ∧
              y₁^2 + y₂^2 = 8 ∧
              ∀ y₁' y₂' : ℕ, (k + 3) * y₁'^2 - 15 * y₁' + t = 0 →
                             (k + 3) * y₂'^2 - 15 * y₂' + t = 0 →
                             y₁'^2 + y₂'^2 ≥ 8 :=
by sorry

end NUMINAMATH_CALUDE_minimize_sum_of_squares_l2326_232639


namespace NUMINAMATH_CALUDE_circle_tangent_to_line_l2326_232617

theorem circle_tangent_to_line (m : ℝ) (h : m > 0) :
  ∃ (x y : ℝ), x^2 + y^2 = 4*m ∧ x + y = 2*Real.sqrt m ∧
  ∀ (x' y' : ℝ), x'^2 + y'^2 = 4*m → x' + y' = 2*Real.sqrt m →
  (x' - x)^2 + (y' - y)^2 = 0 ∨ (x' - x)^2 + (y' - y)^2 > 0 :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_to_line_l2326_232617


namespace NUMINAMATH_CALUDE_divisibility_problem_l2326_232663

theorem divisibility_problem (a b c : ℤ) (h : 18 ∣ (a^3 + b^3 + c^3)) : 6 ∣ (a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l2326_232663


namespace NUMINAMATH_CALUDE_airline_wireless_internet_percentage_l2326_232648

theorem airline_wireless_internet_percentage
  (snack_percentage : ℝ)
  (both_services_percentage : ℝ)
  (h1 : snack_percentage = 70)
  (h2 : both_services_percentage = 35)
  (h3 : both_services_percentage ≤ snack_percentage) :
  ∃ (wireless_percentage : ℝ),
    wireless_percentage = both_services_percentage ∧
    wireless_percentage ≤ 100 ∧
    wireless_percentage ≥ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_airline_wireless_internet_percentage_l2326_232648


namespace NUMINAMATH_CALUDE_intersection_M_N_l2326_232660

def M : Set ℝ := {x | (x - 3) / (x + 1) ≤ 0}

def N : Set ℝ := {-3, -1, 1, 3, 5}

theorem intersection_M_N : M ∩ N = {1, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2326_232660


namespace NUMINAMATH_CALUDE_math_books_count_l2326_232636

/-- The number of math books on a shelf with 100 total books, 32 history books, and 25 geography books. -/
def math_books (total : ℕ) (history : ℕ) (geography : ℕ) : ℕ :=
  total - history - geography

/-- Theorem stating that there are 43 math books on the shelf. -/
theorem math_books_count : math_books 100 32 25 = 43 := by
  sorry

end NUMINAMATH_CALUDE_math_books_count_l2326_232636


namespace NUMINAMATH_CALUDE_factorization_of_a_squared_plus_3a_l2326_232680

theorem factorization_of_a_squared_plus_3a (a : ℝ) : a^2 + 3*a = a*(a+3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_a_squared_plus_3a_l2326_232680


namespace NUMINAMATH_CALUDE_grocery_theorem_l2326_232619

def grocery_problem (initial_budget bread_cost candy_cost : ℚ) : ℚ :=
  let remaining_after_bread_candy := initial_budget - (bread_cost + candy_cost)
  let turkey_cost := remaining_after_bread_candy / 3
  remaining_after_bread_candy - turkey_cost

theorem grocery_theorem :
  grocery_problem 32 3 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_grocery_theorem_l2326_232619


namespace NUMINAMATH_CALUDE_average_weight_of_class_l2326_232618

theorem average_weight_of_class (group1_count : Nat) (group1_avg : Real) 
  (group2_count : Nat) (group2_avg : Real) :
  group1_count = 22 →
  group2_count = 8 →
  group1_avg = 50.25 →
  group2_avg = 45.15 →
  let total_count := group1_count + group2_count
  let total_weight := group1_count * group1_avg + group2_count * group2_avg
  (total_weight / total_count) = 48.89 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_of_class_l2326_232618


namespace NUMINAMATH_CALUDE_correct_head_start_for_dead_heat_l2326_232606

/-- The fraction of the race length that runner a should give as a head start to runner b -/
def head_start_fraction (speed_ratio : ℚ) : ℚ :=
  1 - (1 / speed_ratio)

/-- Theorem stating the correct head start fraction for the given speed ratio -/
theorem correct_head_start_for_dead_heat (race_length : ℚ) (speed_a speed_b : ℚ) 
  (h_speed : speed_a = 16/15 * speed_b) (h_positive : speed_b > 0) :
  head_start_fraction (speed_a / speed_b) * race_length = 1/16 * race_length :=
by sorry

end NUMINAMATH_CALUDE_correct_head_start_for_dead_heat_l2326_232606


namespace NUMINAMATH_CALUDE_solution_set_and_range_l2326_232666

def f (x : ℝ) : ℝ := |2*x + 1| + 2*|x - 3|

theorem solution_set_and_range :
  (∃ (S : Set ℝ), S = {x : ℝ | f x ≤ 7*x} ∧ S = {x : ℝ | x ≥ 1}) ∧
  (∃ (M : Set ℝ), M = {m : ℝ | ∃ x : ℝ, f x = |m|} ∧ M = {m : ℝ | m ≥ 7 ∨ m ≤ -7}) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_and_range_l2326_232666


namespace NUMINAMATH_CALUDE_multiply_subtract_divide_l2326_232669

theorem multiply_subtract_divide : 4 * 6 * 8 - 24 / 4 = 186 := by
  sorry

end NUMINAMATH_CALUDE_multiply_subtract_divide_l2326_232669


namespace NUMINAMATH_CALUDE_rectangle_area_solution_l2326_232634

/-- A rectangle with dimensions (3x - 4) and (4x + 6) has area 12x^2 + 2x - 24 -/
def rectangle_area (x : ℝ) : ℝ := (3*x - 4) * (4*x + 6)

/-- The solution set for x -/
def solution_set : Set ℝ := {x | x > 4/3}

theorem rectangle_area_solution :
  ∀ x : ℝ, x ∈ solution_set ↔ 
    (rectangle_area x = 12*x^2 + 2*x - 24 ∧ 
     3*x - 4 > 0 ∧ 
     4*x + 6 > 0) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_solution_l2326_232634


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2326_232625

theorem quadratic_equation_solution : 
  ∃! x : ℝ, x^2 + 6*x + 8 = -(x + 4)*(x + 6) :=
by
  -- The unique solution is x = -4
  use -4
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2326_232625


namespace NUMINAMATH_CALUDE_least_number_divisible_by_five_primes_l2326_232657

theorem least_number_divisible_by_five_primes : 
  ∃ (n : ℕ), n > 0 ∧ (∃ (p₁ p₂ p₃ p₄ p₅ : ℕ), 
    Nat.Prime p₁ ∧ Nat.Prime p₂ ∧ Nat.Prime p₃ ∧ Nat.Prime p₄ ∧ Nat.Prime p₅ ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₁ ≠ p₅ ∧
    p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₂ ≠ p₅ ∧
    p₃ ≠ p₄ ∧ p₃ ≠ p₅ ∧
    p₄ ≠ p₅ ∧
    n % p₁ = 0 ∧ n % p₂ = 0 ∧ n % p₃ = 0 ∧ n % p₄ = 0 ∧ n % p₅ = 0) ∧
  (∀ m : ℕ, m > 0 → (∃ (q₁ q₂ q₃ q₄ q₅ : ℕ),
    Nat.Prime q₁ ∧ Nat.Prime q₂ ∧ Nat.Prime q₃ ∧ Nat.Prime q₄ ∧ Nat.Prime q₅ ∧
    q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₁ ≠ q₅ ∧
    q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₂ ≠ q₅ ∧
    q₃ ≠ q₄ ∧ q₃ ≠ q₅ ∧
    q₄ ≠ q₅ ∧
    m % q₁ = 0 ∧ m % q₂ = 0 ∧ m % q₃ = 0 ∧ m % q₄ = 0 ∧ m % q₅ = 0) → m ≥ n) ∧
  n = 2310 := by
sorry

end NUMINAMATH_CALUDE_least_number_divisible_by_five_primes_l2326_232657


namespace NUMINAMATH_CALUDE_eggs_left_for_breakfast_l2326_232641

def total_eggs : ℕ := 3 * 12

def eggs_for_crepes : ℕ := total_eggs / 3

def eggs_after_crepes : ℕ := total_eggs - eggs_for_crepes

def eggs_for_cupcakes : ℕ := (eggs_after_crepes * 3) / 5

def eggs_left : ℕ := eggs_after_crepes - eggs_for_cupcakes

theorem eggs_left_for_breakfast : eggs_left = 10 := by
  sorry

end NUMINAMATH_CALUDE_eggs_left_for_breakfast_l2326_232641


namespace NUMINAMATH_CALUDE_integral_rational_function_l2326_232605

open Real

theorem integral_rational_function (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 3) (h3 : x ≠ -1) :
  (deriv fun x => 2*x + 4*log (abs x) + log (abs (x - 3)) - 2*log (abs (x + 1))) x =
  (2*x^3 - x^2 - 7*x - 12) / (x*(x-3)*(x+1)) :=
by sorry

end NUMINAMATH_CALUDE_integral_rational_function_l2326_232605


namespace NUMINAMATH_CALUDE_samirs_age_in_five_years_l2326_232664

/-- Given that Samir's age is half of Hania's age 10 years ago,
    and Hania will be 45 years old in 5 years,
    prove that Samir will be 20 years old in 5 years. -/
theorem samirs_age_in_five_years
  (samir_current_age : ℕ)
  (hania_current_age : ℕ)
  (samir_age_condition : samir_current_age = (hania_current_age - 10) / 2)
  (hania_future_age_condition : hania_current_age + 5 = 45) :
  samir_current_age + 5 = 20 := by
  sorry


end NUMINAMATH_CALUDE_samirs_age_in_five_years_l2326_232664


namespace NUMINAMATH_CALUDE_profit_percent_calculation_l2326_232658

/-- 
Theorem: Given an article where selling at half of a certain price results in a 20% loss,
the profit percent when selling at the full price is 60%.
-/
theorem profit_percent_calculation (cost_price selling_price : ℝ) : 
  (selling_price / 2 = cost_price * 0.8) →  -- Half price results in 20% loss
  (selling_price - cost_price) / cost_price = 0.6  -- Profit percent is 60%
  := by sorry

end NUMINAMATH_CALUDE_profit_percent_calculation_l2326_232658


namespace NUMINAMATH_CALUDE_problem_solution_l2326_232635

theorem problem_solution : (29.7 + 83.45) - 0.3 = 112.85 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2326_232635


namespace NUMINAMATH_CALUDE_prism_volume_l2326_232612

/-- The volume of a right rectangular prism with given face areas and sum of dimensions -/
theorem prism_volume (a b c : ℝ) 
  (h1 : a * b = 18) 
  (h2 : b * c = 20) 
  (h3 : c * a = 12) 
  (h4 : a + b + c = 11) : 
  a * b * c = 12 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l2326_232612


namespace NUMINAMATH_CALUDE_cosine_sum_equation_solutions_l2326_232645

theorem cosine_sum_equation_solutions (x : Real) (h : x ∈ Set.Icc 0 (2 * Real.pi)) :
  2 * Real.cos (5 * x) + 2 * Real.cos (4 * x) + 2 * Real.cos (3 * x) +
  2 * Real.cos (2 * x) + 2 * Real.cos x + 1 = 0 ↔
  ∃ k : Fin 10, x = (2 * k.val.succ * Real.pi) / 11 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_equation_solutions_l2326_232645


namespace NUMINAMATH_CALUDE_town_population_problem_l2326_232616

theorem town_population_problem (initial_population : ℕ) : 
  let after_changes := initial_population + 100 - 400
  let after_year_1 := after_changes / 2
  let after_year_2 := after_year_1 / 2
  let after_year_3 := after_year_2 / 2
  let after_year_4 := after_year_3 / 2
  after_year_4 = 60 → initial_population = 780 := by
  sorry

end NUMINAMATH_CALUDE_town_population_problem_l2326_232616


namespace NUMINAMATH_CALUDE_sweatshirt_cost_is_15_l2326_232671

def hannah_shopping (sweatshirt_cost : ℝ) : Prop :=
  let num_sweatshirts : ℕ := 3
  let num_tshirts : ℕ := 2
  let tshirt_cost : ℝ := 10
  let total_spent : ℝ := 65
  (num_sweatshirts * sweatshirt_cost) + (num_tshirts * tshirt_cost) = total_spent

theorem sweatshirt_cost_is_15 : 
  ∃ (sweatshirt_cost : ℝ), hannah_shopping sweatshirt_cost ∧ sweatshirt_cost = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_sweatshirt_cost_is_15_l2326_232671


namespace NUMINAMATH_CALUDE_function_is_zero_l2326_232681

/-- A function satisfying the given functional equation is the zero function -/
theorem function_is_zero (f : ℝ → ℝ) 
    (h : ∀ x y : ℝ, f (x * f y + 2 * x) = x * y + 2 * f x) : 
  ∀ x : ℝ, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_is_zero_l2326_232681


namespace NUMINAMATH_CALUDE_rational_roots_of_equation_l2326_232608

theorem rational_roots_of_equation (a b c d : ℝ) :
  ∃ x : ℚ, (a + b)^2 * (x + c^2) * (x + d^2) - (c + d)^2 * (x + a^2) * (x + b^2) = 0 :=
by sorry

end NUMINAMATH_CALUDE_rational_roots_of_equation_l2326_232608


namespace NUMINAMATH_CALUDE_inequality_transformations_l2326_232659

theorem inequality_transformations (a b : ℝ) (h : a < b) :
  (a + 2 < b + 2) ∧ 
  (3 * a < 3 * b) ∧ 
  ((1/2) * a < (1/2) * b) ∧ 
  (-2 * a > -2 * b) :=
by sorry

end NUMINAMATH_CALUDE_inequality_transformations_l2326_232659


namespace NUMINAMATH_CALUDE_tobias_swims_3000_meters_l2326_232637

/-- The number of meters Tobias swims in 3 hours with regular pauses -/
def tobias_swim_distance : ℕ :=
  let total_time : ℕ := 3 * 60  -- 3 hours in minutes
  let swim_pause_cycle : ℕ := 25 + 5  -- 25 min swim + 5 min pause
  let num_cycles : ℕ := total_time / swim_pause_cycle
  let total_swim_time : ℕ := num_cycles * 25  -- Total swimming time in minutes
  let meters_per_5min : ℕ := 100  -- Swims 100 meters every 5 minutes
  total_swim_time / 5 * meters_per_5min

/-- Theorem stating that Tobias swims 3000 meters -/
theorem tobias_swims_3000_meters : tobias_swim_distance = 3000 := by
  sorry

#eval tobias_swim_distance  -- This should output 3000

end NUMINAMATH_CALUDE_tobias_swims_3000_meters_l2326_232637


namespace NUMINAMATH_CALUDE_model1_best_fit_l2326_232670

-- Define the coefficient of determination for each model
def R2_model1 : ℝ := 0.976
def R2_model2 : ℝ := 0.776
def R2_model3 : ℝ := 0.076
def R2_model4 : ℝ := 0.351

-- Define a function to determine if a model has the best fitting effect
def has_best_fit (model_R2 : ℝ) : Prop :=
  model_R2 > R2_model2 ∧ model_R2 > R2_model3 ∧ model_R2 > R2_model4

-- Theorem stating that Model 1 has the best fitting effect
theorem model1_best_fit : has_best_fit R2_model1 := by
  sorry

end NUMINAMATH_CALUDE_model1_best_fit_l2326_232670


namespace NUMINAMATH_CALUDE_expression_values_l2326_232685

theorem expression_values (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  let e := a / |a| + b / |b| + c / |c| + d / |d| + (a * b * c * d) / |a * b * c * d|
  e = 5 ∨ e = 1 ∨ e = -1 ∨ e = -5 :=
by sorry

end NUMINAMATH_CALUDE_expression_values_l2326_232685


namespace NUMINAMATH_CALUDE_toms_age_ratio_l2326_232647

theorem toms_age_ratio (T N : ℕ) : 
  (∃ (x y z : ℕ), T = x + y + z) →  -- T is the sum of three children's ages
  (T - N = 2 * ((T - N) - 3 * N)) →  -- N years ago, Tom's age was twice the sum of his children's ages
  T / N = 5 := by
sorry

end NUMINAMATH_CALUDE_toms_age_ratio_l2326_232647


namespace NUMINAMATH_CALUDE_chemistry_class_gender_difference_l2326_232611

theorem chemistry_class_gender_difference :
  ∀ (boys girls : ℕ),
  (3 : ℕ) * boys = (4 : ℕ) * girls →
  boys + girls = 42 →
  girls - boys = 6 :=
by sorry

end NUMINAMATH_CALUDE_chemistry_class_gender_difference_l2326_232611


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2326_232609

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), a.1 * b.2 = k * a.2 * b.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (2*x, -3)
  are_parallel a b → x = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2326_232609


namespace NUMINAMATH_CALUDE_smallest_non_factor_non_prime_l2326_232696

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem smallest_non_factor_non_prime : 
  ∃ (n : ℕ), 
    n > 0 ∧ 
    ¬(factorial 30 % n = 0) ∧ 
    ¬(Nat.Prime n) ∧
    (∀ m : ℕ, m > 0 ∧ m < n → 
      (factorial 30 % m = 0) ∨ (Nat.Prime m)) ∧
    n = 961 := by
  sorry

end NUMINAMATH_CALUDE_smallest_non_factor_non_prime_l2326_232696


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2326_232600

/-- A monotonic continuous function on the real numbers satisfying f(x)·f(y) = f(x+y) -/
def FunctionalEquationSolution (f : ℝ → ℝ) : Prop :=
  Monotone f ∧ Continuous f ∧ ∀ x y : ℝ, f x * f y = f (x + y)

/-- The solution to the functional equation is of the form f(x) = a^x for some a > 0 and a ≠ 1 -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : FunctionalEquationSolution f) :
  ∃ a : ℝ, a > 0 ∧ a ≠ 1 ∧ ∀ x : ℝ, f x = a^x :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2326_232600


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l2326_232632

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y + 2*x*y = 8) :
  ∀ z, z = x + 2*y → z ≥ 4 ∧ ∃ x₀ y₀, x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 2*y₀ + 2*x₀*y₀ = 8 ∧ x₀ + 2*y₀ = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l2326_232632


namespace NUMINAMATH_CALUDE_race_people_count_l2326_232627

theorem race_people_count (num_cars : ℕ) (initial_people_per_car : ℕ) (people_gained_halfway : ℕ) :
  num_cars = 20 →
  initial_people_per_car = 3 →
  people_gained_halfway = 1 →
  num_cars * (initial_people_per_car + people_gained_halfway) = 80 := by
sorry

end NUMINAMATH_CALUDE_race_people_count_l2326_232627


namespace NUMINAMATH_CALUDE_fifth_number_in_row_51_l2326_232679

/-- Pascal's triangle binomial coefficient -/
def pascal (n k : ℕ) : ℕ :=
  Nat.choose n k

/-- The number of elements in a row of Pascal's triangle -/
def row_size (row : ℕ) : ℕ :=
  row + 1

theorem fifth_number_in_row_51 :
  pascal 50 4 = 22050 :=
by
  sorry

end NUMINAMATH_CALUDE_fifth_number_in_row_51_l2326_232679


namespace NUMINAMATH_CALUDE_sum_of_squares_l2326_232654

theorem sum_of_squares (x y z : ℝ) 
  (eq1 : x^2 + 3*y = 10)
  (eq2 : y^2 + 5*z = -10)
  (eq3 : z^2 + 7*x = -20) :
  x^2 + y^2 + z^2 = 20.75 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_l2326_232654


namespace NUMINAMATH_CALUDE_sphere_volume_ratio_l2326_232683

theorem sphere_volume_ratio (r R : ℝ) (h : R = 4 * r) :
  (4 / 3 * Real.pi * R^3) / (4 / 3 * Real.pi * r^3) = 64 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_ratio_l2326_232683


namespace NUMINAMATH_CALUDE_divisors_of_2_pow_56_minus_1_l2326_232689

theorem divisors_of_2_pow_56_minus_1 :
  ∃ (a b : ℕ), 95 < a ∧ a < 105 ∧ 95 < b ∧ b < 105 ∧
  a ≠ b ∧
  (2^56 - 1) % a = 0 ∧ (2^56 - 1) % b = 0 ∧
  (∀ c : ℕ, 95 < c ∧ c < 105 → (2^56 - 1) % c = 0 → c = a ∨ c = b) ∧
  a = 101 ∧ b = 127 :=
sorry

end NUMINAMATH_CALUDE_divisors_of_2_pow_56_minus_1_l2326_232689


namespace NUMINAMATH_CALUDE_problem_figure_perimeter_l2326_232697

/-- Represents the figure described in the problem -/
structure SquareFigure where
  stackHeight : Nat
  stackGap : Nat
  topSquares : Nat
  bottomSquares : Nat

/-- Calculates the perimeter of the square figure -/
def perimeterOfSquareFigure (fig : SquareFigure) : Nat :=
  let horizontalSegments := fig.topSquares * 2 + fig.bottomSquares * 2
  let verticalSegments := fig.stackHeight * 2 * 2 + fig.topSquares * 2
  horizontalSegments + verticalSegments

/-- The specific figure described in the problem -/
def problemFigure : SquareFigure :=
  { stackHeight := 3
  , stackGap := 1
  , topSquares := 3
  , bottomSquares := 2 }

theorem problem_figure_perimeter :
  perimeterOfSquareFigure problemFigure = 22 := by
  sorry

end NUMINAMATH_CALUDE_problem_figure_perimeter_l2326_232697


namespace NUMINAMATH_CALUDE_rectangle_area_ratio_l2326_232602

/-- Given two rectangles S₁ and S₂ with specific vertices and equal areas, prove that 360x/y = 810 --/
theorem rectangle_area_ratio (x y : ℝ) (hx : x < 9) (hy : y < 4) 
  (h_equal_area : x * (4 - y) = y * (9 - x)) : 
  360 * x / y = 810 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_ratio_l2326_232602


namespace NUMINAMATH_CALUDE_james_delivery_capacity_l2326_232651

/-- Given that James takes 20 trips a day and delivers 1000 bags in 5 days,
    prove that he can carry 10 bags on each trip. -/
theorem james_delivery_capacity
  (trips_per_day : ℕ)
  (total_bags : ℕ)
  (total_days : ℕ)
  (h1 : trips_per_day = 20)
  (h2 : total_bags = 1000)
  (h3 : total_days = 5) :
  total_bags / (trips_per_day * total_days) = 10 := by
  sorry

end NUMINAMATH_CALUDE_james_delivery_capacity_l2326_232651


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l2326_232615

theorem trigonometric_equation_solution (x : ℝ) : 
  (Real.tan (5 * π / 2 + x) - 3 * Real.tan x ^ 2 = (Real.cos (2 * x) - 1) / Real.cos x ^ 2) →
  ∃ k : ℤ, x = π / 4 * (4 * ↑k - 1) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l2326_232615


namespace NUMINAMATH_CALUDE_remainder_21_pow_2051_mod_29_l2326_232653

theorem remainder_21_pow_2051_mod_29 : 21^2051 % 29 = 15 := by
  sorry

end NUMINAMATH_CALUDE_remainder_21_pow_2051_mod_29_l2326_232653


namespace NUMINAMATH_CALUDE_remainder_theorem_l2326_232694

theorem remainder_theorem : (104 * 106 - 8) % 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2326_232694


namespace NUMINAMATH_CALUDE_segment_construction_l2326_232650

/-- A list of 99 natural numbers from 1 to 99 -/
def segments : List ℕ := List.range 99

/-- The sum of all segments -/
def total_length : ℕ := List.sum segments

/-- Predicate to check if a square can be formed -/
def can_form_square (segs : List ℕ) : Prop :=
  ∃ (side : ℕ), 4 * side = List.sum segs

/-- Predicate to check if a rectangle can be formed -/
def can_form_rectangle (segs : List ℕ) : Prop :=
  ∃ (length width : ℕ), length * width = List.sum segs ∧ length ≠ width

/-- Predicate to check if an equilateral triangle can be formed -/
def can_form_equilateral_triangle (segs : List ℕ) : Prop :=
  ∃ (side : ℕ), 3 * side = List.sum segs

theorem segment_construction :
  ¬ can_form_square segments ∧
  can_form_rectangle segments ∧
  can_form_equilateral_triangle segments :=
sorry

end NUMINAMATH_CALUDE_segment_construction_l2326_232650


namespace NUMINAMATH_CALUDE_roots_of_equation_l2326_232656

theorem roots_of_equation (x : ℝ) : (x - 5)^2 = 2*(x - 5) ↔ x = 5 ∨ x = 7 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_equation_l2326_232656


namespace NUMINAMATH_CALUDE_only_cylinder_has_rectangular_front_view_l2326_232604

-- Define the solid figures
inductive SolidFigure
  | Cylinder
  | TriangularPyramid
  | Sphere
  | Cone

-- Define the front view shapes
inductive FrontViewShape
  | Rectangle
  | Triangle
  | Circle

-- Function to determine the front view shape of a solid figure
def frontViewShape (figure : SolidFigure) : FrontViewShape :=
  match figure with
  | SolidFigure.Cylinder => FrontViewShape.Rectangle
  | SolidFigure.TriangularPyramid => FrontViewShape.Triangle
  | SolidFigure.Sphere => FrontViewShape.Circle
  | SolidFigure.Cone => FrontViewShape.Triangle

-- Theorem stating that only the cylinder has a rectangular front view
theorem only_cylinder_has_rectangular_front_view :
  ∀ (figure : SolidFigure),
    frontViewShape figure = FrontViewShape.Rectangle ↔ figure = SolidFigure.Cylinder :=
by sorry

end NUMINAMATH_CALUDE_only_cylinder_has_rectangular_front_view_l2326_232604


namespace NUMINAMATH_CALUDE_complex_equation_sum_l2326_232621

theorem complex_equation_sum (a b : ℝ) : 
  (a : ℂ) + b * Complex.I = (1 + 2 * Complex.I) * (1 - Complex.I) → a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l2326_232621


namespace NUMINAMATH_CALUDE_triple_characterization_l2326_232682

def is_valid_triple (a m n : ℕ) : Prop :=
  a ≥ 2 ∧ m ≥ 2 ∧ (a^n + 203) % (a^m + 1) = 0

def solution_set (k m : ℕ) : Set (ℕ × ℕ × ℕ) :=
  {(2, 2, 4*k+1), (2, 3, 6*k+2), (2, 4, 8*k+8), (2, 6, 12*k+9),
   (3, 2, 4*k+3), (4, 2, 4*k+4), (5, 2, 4*k+1), (8, 2, 4*k+3),
   (10, 2, 4*k+2), (203, m, (2*k+1)*m+1)}

theorem triple_characterization :
  ∀ a m n : ℕ, is_valid_triple a m n ↔ ∃ k : ℕ, (a, m, n) ∈ solution_set k m :=
sorry

end NUMINAMATH_CALUDE_triple_characterization_l2326_232682


namespace NUMINAMATH_CALUDE_top_of_second_column_is_20_l2326_232640

/-- Represents a 7x6 grid of numbers -/
def Grid := Fin 7 → Fin 6 → ℤ

/-- The given grid satisfies the problem conditions -/
def satisfies_conditions (g : Grid) : Prop :=
  -- Row is an arithmetic sequence with first element 15 and common difference 0
  (∀ i : Fin 7, g i 0 = 15) ∧
  -- Third column is an arithmetic sequence containing 10 and 5
  (g 2 1 = 10 ∧ g 2 2 = 5) ∧
  -- Second column's bottom element is -10
  (g 1 5 = -10) ∧
  -- Each column is an arithmetic sequence
  (∀ j : Fin 6, ∃ d : ℤ, ∀ i : Fin 5, g 1 (i + 1) = g 1 i + d) ∧
  (∀ j : Fin 6, ∃ d : ℤ, ∀ i : Fin 5, g 2 (i + 1) = g 2 i + d)

/-- The theorem to be proved -/
theorem top_of_second_column_is_20 (g : Grid) (h : satisfies_conditions g) : g 1 0 = 20 := by
  sorry

end NUMINAMATH_CALUDE_top_of_second_column_is_20_l2326_232640


namespace NUMINAMATH_CALUDE_regular_hexagon_side_length_l2326_232692

/-- The length of a side in a regular hexagon, given the distance between opposite sides -/
theorem regular_hexagon_side_length (distance_between_opposite_sides : ℝ) : 
  distance_between_opposite_sides > 0 →
  ∃ (side_length : ℝ), 
    side_length = (20 * Real.sqrt 3) / 3 * distance_between_opposite_sides / 10 := by
  sorry

end NUMINAMATH_CALUDE_regular_hexagon_side_length_l2326_232692


namespace NUMINAMATH_CALUDE_floor_equation_solutions_l2326_232630

theorem floor_equation_solutions (n : ℤ) : 
  (⌊n^2 / 3⌋ : ℤ) - (⌊n / 3⌋ : ℤ)^2 = 3 ↔ n = -8 ∨ n = -4 := by
  sorry

end NUMINAMATH_CALUDE_floor_equation_solutions_l2326_232630


namespace NUMINAMATH_CALUDE_x_power_twenty_equals_one_l2326_232662

theorem x_power_twenty_equals_one (x : ℝ) (h : x + 1/x = 2) : x^20 = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_power_twenty_equals_one_l2326_232662


namespace NUMINAMATH_CALUDE_ball_probability_theorem_l2326_232699

/-- Represents the two boxes containing balls -/
inductive Box
| A
| B

/-- Represents the color of the balls -/
inductive Color
| Red
| White

/-- Represents the number of balls in each box before transfer -/
def initial_count : Box → Color → ℕ
| Box.A, Color.Red => 4
| Box.A, Color.White => 2
| Box.B, Color.Red => 2
| Box.B, Color.White => 3

/-- Represents the probability space for this problem -/
structure BallProbability where
  /-- The probability of event A (red ball taken from box A) -/
  prob_A : ℝ
  /-- The probability of event B (white ball taken from box A) -/
  prob_B : ℝ
  /-- The probability of event C (red ball taken from box B after transfer) -/
  prob_C : ℝ
  /-- The conditional probability of C given A -/
  prob_C_given_A : ℝ

/-- The main theorem that encapsulates the problem -/
theorem ball_probability_theorem (p : BallProbability) : 
  p.prob_A + p.prob_B = 1 ∧ 
  p.prob_A * p.prob_B = 0 ∧
  p.prob_C_given_A = 1/2 ∧ 
  p.prob_C = 4/9 := by
  sorry


end NUMINAMATH_CALUDE_ball_probability_theorem_l2326_232699


namespace NUMINAMATH_CALUDE_libor_number_theorem_l2326_232661

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def all_digits_odd (n : ℕ) : Prop :=
  ∀ d, d ∈ (n.digits 10) → d % 2 = 1

def no_odd_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ (n.digits 10) → d % 2 = 0

theorem libor_number_theorem :
  ∀ n : ℕ, is_three_digit n ∧ all_digits_odd n ∧ is_three_digit (n + 421) ∧ no_odd_digits (n + 421) →
    n = 179 ∨ n = 199 ∨ n = 379 ∨ n = 399 :=
sorry

end NUMINAMATH_CALUDE_libor_number_theorem_l2326_232661


namespace NUMINAMATH_CALUDE_probability_k_standard_parts_formula_l2326_232601

/-- The probability of selecting exactly k standard parts when randomly choosing m parts from a batch of N parts containing n standard parts. -/
def probability_k_standard_parts (N n m k : ℕ) : ℚ :=
  (Nat.choose n k * Nat.choose (N - n) (m - k)) / Nat.choose N m

/-- Theorem stating that the probability of selecting exactly k standard parts
    when randomly choosing m parts from a batch of N parts containing n standard parts
    is equal to (C_n^k * C_(N-n)^(m-k)) / C_N^m. -/
theorem probability_k_standard_parts_formula
  (N n m k : ℕ)
  (h1 : n ≤ N)
  (h2 : m ≤ N)
  (h3 : k ≤ m)
  (h4 : k ≤ n) :
  probability_k_standard_parts N n m k =
    (Nat.choose n k * Nat.choose (N - n) (m - k)) / Nat.choose N m :=
by
  sorry

#check probability_k_standard_parts_formula

end NUMINAMATH_CALUDE_probability_k_standard_parts_formula_l2326_232601


namespace NUMINAMATH_CALUDE_room_length_proof_l2326_232652

theorem room_length_proof (x : ℝ) 
  (room_width : ℝ) (room_height : ℝ)
  (door_width : ℝ) (door_height : ℝ)
  (large_window_width : ℝ) (large_window_height : ℝ)
  (small_window_width : ℝ) (small_window_height : ℝ)
  (paint_cost_per_sqm : ℝ) (total_paint_cost : ℝ)
  (h1 : room_width = 7)
  (h2 : room_height = 5)
  (h3 : door_width = 1)
  (h4 : door_height = 3)
  (h5 : large_window_width = 2)
  (h6 : large_window_height = 1.5)
  (h7 : small_window_width = 1)
  (h8 : small_window_height = 1.5)
  (h9 : paint_cost_per_sqm = 3)
  (h10 : total_paint_cost = 474)
  (h11 : total_paint_cost = paint_cost_per_sqm * 
    (2 * (x * room_height + room_width * room_height) - 
    2 * (door_width * door_height) - 
    (large_window_width * large_window_height) - 
    2 * (small_window_width * small_window_height))) :
  x = 10 := by
  sorry

end NUMINAMATH_CALUDE_room_length_proof_l2326_232652


namespace NUMINAMATH_CALUDE_tile_border_ratio_l2326_232684

/-- Represents the arrangement of tiles in a square garden -/
structure TileArrangement where
  n : ℕ               -- Number of tiles along one side of the garden
  s : ℝ               -- Side length of each tile in meters
  d : ℝ               -- Width of the border around each tile in meters
  h_positive_s : 0 < s
  h_positive_d : 0 < d

/-- The theorem stating the ratio of border width to tile side length -/
theorem tile_border_ratio (arr : TileArrangement) (h_n : arr.n = 30) 
  (h_coverage : (arr.n^2 * arr.s^2) / ((arr.n * arr.s + 2 * arr.n * arr.d)^2) = 0.81) :
  arr.d / arr.s = 1 / 18 := by
  sorry

end NUMINAMATH_CALUDE_tile_border_ratio_l2326_232684


namespace NUMINAMATH_CALUDE_cats_in_academy_l2326_232668

/-- The number of cats that can jump -/
def jump : ℕ := 40

/-- The number of cats that can fetch -/
def fetch : ℕ := 25

/-- The number of cats that can spin -/
def spin : ℕ := 30

/-- The number of cats that can jump and fetch -/
def jump_fetch : ℕ := 20

/-- The number of cats that can fetch and spin -/
def fetch_spin : ℕ := 10

/-- The number of cats that can jump and spin -/
def jump_spin : ℕ := 15

/-- The number of cats that can do all three tricks -/
def all_tricks : ℕ := 7

/-- The number of cats that can do none of the tricks -/
def no_tricks : ℕ := 5

/-- The total number of cats in the academy -/
def total_cats : ℕ := 62

theorem cats_in_academy :
  total_cats = 
    (jump - jump_fetch - jump_spin + all_tricks) +
    (jump_fetch - all_tricks) +
    (fetch - jump_fetch - fetch_spin + all_tricks) +
    (fetch_spin - all_tricks) +
    (jump_spin - all_tricks) +
    (spin - jump_spin - fetch_spin + all_tricks) +
    all_tricks +
    no_tricks := by sorry

end NUMINAMATH_CALUDE_cats_in_academy_l2326_232668


namespace NUMINAMATH_CALUDE_platform_length_l2326_232675

/-- Given a train of length 300 meters that crosses a platform in 27 seconds
    and a signal pole in 18 seconds, the length of the platform is 150 meters. -/
theorem platform_length (train_length : ℝ) (platform_time : ℝ) (pole_time : ℝ) :
  train_length = 300 →
  platform_time = 27 →
  pole_time = 18 →
  (train_length * platform_time / pole_time) - train_length = 150 := by
sorry

end NUMINAMATH_CALUDE_platform_length_l2326_232675


namespace NUMINAMATH_CALUDE_expression_value_l2326_232622

theorem expression_value (x y : ℝ) (h : 2 * x + y = 6) :
  ((x - y)^2 - (x + y)^2 + y * (2 * x - y)) / (-2 * y) = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2326_232622


namespace NUMINAMATH_CALUDE_cube_sum_ge_squared_product_sum_l2326_232655

theorem cube_sum_ge_squared_product_sum {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^3 + b^3 + c^3 ≥ a^2*b + b^2*c + c^2*a := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_ge_squared_product_sum_l2326_232655


namespace NUMINAMATH_CALUDE_smallest_valid_number_last_four_digits_l2326_232623

def is_valid_digit (d : Nat) : Prop :=
  d = 4 ∨ d = 6 ∨ d = 9

def contains_all_required_digits (n : Nat) : Prop :=
  ∃ (d1 d2 d3 : Nat), d1 ∈ n.digits 10 ∧ d2 ∈ n.digits 10 ∧ d3 ∈ n.digits 10 ∧
  d1 = 4 ∧ d2 = 6 ∧ d3 = 9

def all_digits_valid (n : Nat) : Prop :=
  ∀ d ∈ n.digits 10, is_valid_digit d

def last_four_digits (n : Nat) : Nat :=
  n % 10000

theorem smallest_valid_number_last_four_digits :
  ∃ (n : Nat), 
    (∀ m : Nat, m < n → ¬(m % 6 = 0 ∧ m % 9 = 0 ∧ all_digits_valid m ∧ contains_all_required_digits m)) ∧
    n % 6 = 0 ∧
    n % 9 = 0 ∧
    all_digits_valid n ∧
    contains_all_required_digits n ∧
    last_four_digits n = 4699 :=
  sorry

end NUMINAMATH_CALUDE_smallest_valid_number_last_four_digits_l2326_232623


namespace NUMINAMATH_CALUDE_cost_per_serving_is_one_dollar_l2326_232603

/-- The cost of a serving of spaghetti and meatballs -/
def cost_per_serving (pasta_cost sauce_cost meatballs_cost : ℚ) (num_servings : ℕ) : ℚ :=
  (pasta_cost + sauce_cost + meatballs_cost) / num_servings

/-- Theorem: The cost per serving is $1.00 -/
theorem cost_per_serving_is_one_dollar :
  cost_per_serving 1 2 5 8 = 1 := by sorry

end NUMINAMATH_CALUDE_cost_per_serving_is_one_dollar_l2326_232603


namespace NUMINAMATH_CALUDE_grass_stains_count_l2326_232649

theorem grass_stains_count (grass_stain_time marinara_stain_time total_time : ℕ) 
  (marinara_stain_count : ℕ) (h1 : grass_stain_time = 4) 
  (h2 : marinara_stain_time = 7) (h3 : marinara_stain_count = 1) 
  (h4 : total_time = 19) : 
  ∃ (grass_stain_count : ℕ), 
    grass_stain_count * grass_stain_time + 
    marinara_stain_count * marinara_stain_time = total_time ∧ 
    grass_stain_count = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_grass_stains_count_l2326_232649


namespace NUMINAMATH_CALUDE_complex_number_opposite_parts_l2326_232672

theorem complex_number_opposite_parts (a : ℝ) : 
  let z : ℂ := a / (1 - 2*I) + Complex.abs I
  (Complex.re z = -Complex.im z) → a = -5/3 := by
sorry

end NUMINAMATH_CALUDE_complex_number_opposite_parts_l2326_232672


namespace NUMINAMATH_CALUDE_simplify_polynomial_l2326_232613

theorem simplify_polynomial (r : ℝ) : (2*r^2 + 5*r - 7) - (r^2 + 9*r - 3) = r^2 - 4*r - 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l2326_232613


namespace NUMINAMATH_CALUDE_max_balls_for_five_weighings_impossibility_more_than_243_balls_l2326_232678

/-- The number of weighings required to identify the lighter ball -/
def num_weighings : ℕ := 5

/-- The maximum number of balls that can be tested with the given number of weighings -/
def max_balls : ℕ := 3^num_weighings

/-- Theorem stating that the maximum number of balls is 243 given 5 weighings -/
theorem max_balls_for_five_weighings :
  num_weighings = 5 → max_balls = 243 := by
  sorry

/-- Theorem stating that it's impossible to identify the lighter ball among more than 243 balls with 5 weighings -/
theorem impossibility_more_than_243_balls (n : ℕ) :
  num_weighings = 5 → n > 243 → ¬(∃ strategy : Unit, True) := by
  sorry

end NUMINAMATH_CALUDE_max_balls_for_five_weighings_impossibility_more_than_243_balls_l2326_232678


namespace NUMINAMATH_CALUDE_crane_height_theorem_l2326_232688

/-- Represents the height of a crane and the building it's working on -/
structure CraneBuilding where
  crane_height : ℝ
  building_height : ℝ

/-- The problem setup -/
def construction_problem (crane2_height : ℝ) : Prop :=
  let crane1 : CraneBuilding := ⟨228, 200⟩
  let crane2 : CraneBuilding := ⟨crane2_height, 100⟩
  let crane3 : CraneBuilding := ⟨147, 140⟩
  let cranes : List CraneBuilding := [crane1, crane2, crane3]
  let avg_height_diff : ℝ := (cranes.map (λ c => c.crane_height - c.building_height)).sum / cranes.length
  let avg_building_height : ℝ := (cranes.map (λ c => c.building_height)).sum / cranes.length
  avg_height_diff = 0.13 * avg_building_height

/-- The theorem to be proved -/
theorem crane_height_theorem : 
  ∃ (h : ℝ), construction_problem h ∧ abs (h - 122) < 1 :=
sorry

end NUMINAMATH_CALUDE_crane_height_theorem_l2326_232688


namespace NUMINAMATH_CALUDE_quadratic_transformations_integer_roots_l2326_232614

/-- 
Given a quadratic equation x^2 + px + q = 0, where p and q are integers,
this function returns true if the equation has integer roots.
-/
def has_integer_roots (p q : ℤ) : Prop :=
  ∃ x y : ℤ, x^2 + p*x + q = 0 ∧ y^2 + p*y + q = 0 ∧ x ≠ y

/-- 
This theorem states that there exist initial integer values for p and q
such that the quadratic equation x^2 + px + q = 0 and its nine transformations
(where p and q are increased by 1 each time) all have integer roots.
-/
theorem quadratic_transformations_integer_roots :
  ∃ p q : ℤ, 
    (∀ i : ℕ, i ≤ 9 → has_integer_roots (p + i) (q + i)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_transformations_integer_roots_l2326_232614


namespace NUMINAMATH_CALUDE_homeroom_teacher_selection_count_l2326_232646

/-- The number of ways to arrange k elements from n distinct elements -/
def arrangementCount (n k : ℕ) : ℕ := sorry

/-- The number of valid selection schemes for homeroom teachers -/
def validSelectionCount (maleTotalCount femaleTotalCount selectCount : ℕ) : ℕ :=
  arrangementCount (maleTotalCount + femaleTotalCount) selectCount -
  (arrangementCount maleTotalCount selectCount + arrangementCount femaleTotalCount selectCount)

theorem homeroom_teacher_selection_count :
  validSelectionCount 5 4 3 = 420 := by sorry

end NUMINAMATH_CALUDE_homeroom_teacher_selection_count_l2326_232646


namespace NUMINAMATH_CALUDE_a_gt_2_necessary_not_sufficient_for_a_gt_5_l2326_232624

theorem a_gt_2_necessary_not_sufficient_for_a_gt_5 :
  (∀ a : ℝ, a > 5 → a > 2) ∧ 
  (∃ a : ℝ, a > 2 ∧ ¬(a > 5)) :=
by sorry

end NUMINAMATH_CALUDE_a_gt_2_necessary_not_sufficient_for_a_gt_5_l2326_232624


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l2326_232642

theorem sum_of_two_numbers (x y : ℤ) : x = 18 ∧ y = 2 * x - 3 → x + y = 51 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l2326_232642


namespace NUMINAMATH_CALUDE_no_alpha_sequence_exists_l2326_232620

theorem no_alpha_sequence_exists : ¬∃ (α : ℝ) (a : ℕ → ℝ), 
  (0 < α ∧ α < 1) ∧ 
  (∀ n : ℕ, 0 < a n) ∧
  (∀ n : ℕ, 1 + a (n + 1) ≤ a n + (α / n) * a n) :=
by sorry

end NUMINAMATH_CALUDE_no_alpha_sequence_exists_l2326_232620


namespace NUMINAMATH_CALUDE_min_volume_base_area_is_8d_squared_l2326_232690

/-- Regular quadrilateral pyramid with a plane bisecting the dihedral angle -/
structure RegularPyramid where
  /-- Distance from the base to the intersection point of the bisecting plane and the height -/
  d : ℝ
  /-- The plane bisects the dihedral angle at a side of the base -/
  bisects_dihedral_angle : True

/-- The area of the base that minimizes the volume of the pyramid -/
def min_volume_base_area (p : RegularPyramid) : ℝ := 8 * p.d^2

/-- Theorem stating that the area of the base minimizing the volume is 8d^2 -/
theorem min_volume_base_area_is_8d_squared (p : RegularPyramid) :
  min_volume_base_area p = 8 * p.d^2 := by
  sorry

end NUMINAMATH_CALUDE_min_volume_base_area_is_8d_squared_l2326_232690


namespace NUMINAMATH_CALUDE_sequence_contains_large_number_l2326_232693

theorem sequence_contains_large_number 
  (seq : Fin 20 → ℕ) 
  (distinct : ∀ i j, i ≠ j → seq i ≠ seq j) 
  (consecutive_product_square : ∀ i : Fin 19, ∃ k : ℕ, seq i * seq (i.succ) = k * k) 
  (first_num : seq 0 = 42) :
  ∃ i : Fin 20, seq i > 16000 := by
  sorry

end NUMINAMATH_CALUDE_sequence_contains_large_number_l2326_232693


namespace NUMINAMATH_CALUDE_snail_problem_l2326_232695

/-- The number of snails originally in Centerville -/
def original_snails : ℕ := 11760

/-- The number of snails removed from Centerville -/
def removed_snails : ℕ := 3482

/-- The number of snails remaining in Centerville -/
def remaining_snails : ℕ := original_snails - removed_snails

theorem snail_problem : remaining_snails = 8278 := by
  sorry

end NUMINAMATH_CALUDE_snail_problem_l2326_232695


namespace NUMINAMATH_CALUDE_constant_c_value_l2326_232686

theorem constant_c_value (b c : ℝ) : 
  (∀ x : ℝ, (x + 4) * (x + b) = x^2 + c*x + 12) → c = 7 := by
  sorry

end NUMINAMATH_CALUDE_constant_c_value_l2326_232686


namespace NUMINAMATH_CALUDE_quadratic_function_range_l2326_232628

/-- Represents a quadratic function y = ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- Evaluates the quadratic function at a given x -/
def QuadraticFunction.eval (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

/-- Theorem stating the range of y values for a specific quadratic function -/
theorem quadratic_function_range (f : QuadraticFunction) 
  (h1 : f.eval (-1) = -5)
  (h2 : f.eval 0 = -8)
  (h3 : f.eval 1 = -9)
  (h4 : f.eval 3 = -5)
  (h5 : f.eval 5 = 7) :
  ∀ x, 0 < x → x < 5 → -9 ≤ f.eval x ∧ f.eval x < 7 :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l2326_232628


namespace NUMINAMATH_CALUDE_age_proof_l2326_232638

/-- Proves the ages of Desiree and her cousin given the conditions -/
theorem age_proof (desiree_age : ℝ) (cousin_age : ℝ) : 
  desiree_age = 2.99999835 →
  cousin_age = 1.499999175 →
  desiree_age = 2 * cousin_age →
  desiree_age + 30 = 0.6666666 * (cousin_age + 30) + 14 :=
by sorry

end NUMINAMATH_CALUDE_age_proof_l2326_232638


namespace NUMINAMATH_CALUDE_complex_magnitude_squared_l2326_232643

theorem complex_magnitude_squared (z : ℂ) (h : z^2 = 16 - 30*I) : Complex.abs z = Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_squared_l2326_232643


namespace NUMINAMATH_CALUDE_parabola_intersection_l2326_232665

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2 + 2*x - 3

-- Define line l1
def line_l1 (x m : ℝ) : ℝ := -x + m

-- Define the axis of symmetry of the parabola
def axis_of_symmetry : ℝ := -1

-- Define the property that l2 is symmetric with respect to the axis of symmetry
def l2_symmetric (B D : ℝ × ℝ) : Prop :=
  B.1 + D.1 = 2 * axis_of_symmetry

-- Define the condition that A and D are above x-axis, B and C are below
def points_position (A B C D : ℝ × ℝ) : Prop :=
  A.2 > 0 ∧ D.2 > 0 ∧ B.2 < 0 ∧ C.2 < 0

-- Define the condition AC · BD = 26
def product_condition (A B C D : ℝ × ℝ) : Prop :=
  ((A.1 - C.1)^2 + (A.2 - C.2)^2) * ((B.1 - D.1)^2 + (B.2 - D.2)^2) = 26

-- Theorem statement
theorem parabola_intersection (m : ℝ) 
  (A B C D : ℝ × ℝ) 
  (h1 : ∀ x, parabola x = line_l1 x m → (x = A.1 ∨ x = C.1))
  (h2 : l2_symmetric B D)
  (h3 : points_position A B C D)
  (h4 : product_condition A B C D) :
  m = -2 := by sorry

end NUMINAMATH_CALUDE_parabola_intersection_l2326_232665


namespace NUMINAMATH_CALUDE_cylinder_height_relationship_l2326_232667

theorem cylinder_height_relationship (r1 h1 r2 h2 : ℝ) :
  r1 > 0 ∧ h1 > 0 ∧ r2 > 0 ∧ h2 > 0 →
  r2 = 1.2 * r1 →
  π * r1^2 * h1 = π * r2^2 * h2 →
  h1 = 1.44 * h2 :=
by sorry

end NUMINAMATH_CALUDE_cylinder_height_relationship_l2326_232667


namespace NUMINAMATH_CALUDE_sin_shift_equivalence_l2326_232691

/-- Proves that shifting sin(2x + π/3) right by π/4 results in sin(2x - π/6) -/
theorem sin_shift_equivalence (x : ℝ) :
  Real.sin (2 * (x - π/4) + π/3) = Real.sin (2*x - π/6) := by
  sorry

end NUMINAMATH_CALUDE_sin_shift_equivalence_l2326_232691


namespace NUMINAMATH_CALUDE_prime_squared_minus_one_divisibility_l2326_232674

theorem prime_squared_minus_one_divisibility (p : ℕ) (h_prime : Nat.Prime p) (h_ge_7 : p ≥ 7) :
  (∃ q : ℕ, Nat.Prime q ∧ q ≥ 7 ∧ 40 ∣ (q^2 - 1)) ∧
  (∃ r : ℕ, Nat.Prime r ∧ r ≥ 7 ∧ ¬(40 ∣ (r^2 - 1))) :=
sorry

end NUMINAMATH_CALUDE_prime_squared_minus_one_divisibility_l2326_232674


namespace NUMINAMATH_CALUDE_basketball_max_score_l2326_232673

def max_individual_score (n : ℕ) (total_points : ℕ) (min_points : ℕ) : ℕ :=
  total_points - (n - 1) * min_points

theorem basketball_max_score :
  max_individual_score 12 100 7 = 23 :=
by sorry

end NUMINAMATH_CALUDE_basketball_max_score_l2326_232673


namespace NUMINAMATH_CALUDE_john_bought_three_sodas_l2326_232677

/-- Given a payment, cost per soda, and change received, calculate the number of sodas bought --/
def sodas_bought (payment : ℕ) (cost_per_soda : ℕ) (change : ℕ) : ℕ :=
  (payment - change) / cost_per_soda

/-- Theorem: John bought 3 sodas --/
theorem john_bought_three_sodas :
  sodas_bought 20 2 14 = 3 := by
  sorry

end NUMINAMATH_CALUDE_john_bought_three_sodas_l2326_232677


namespace NUMINAMATH_CALUDE_factor_proof_l2326_232676

theorem factor_proof : 
  (∃ n : ℤ, 65 = 5 * n) ∧ (∃ m : ℤ, 144 = 9 * m) := by sorry

end NUMINAMATH_CALUDE_factor_proof_l2326_232676


namespace NUMINAMATH_CALUDE_sequence_is_arithmetic_l2326_232610

/-- The sum of the first n terms of the sequence -/
def S (a n : ℕ) : ℕ := a * n^2 + n

/-- The n-th term of the sequence -/
def a_n (a n : ℕ) : ℤ := S a n - S a (n-1)

/-- A sequence is arithmetic if the difference between consecutive terms is constant -/
def is_arithmetic_sequence (f : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, f (n+1) - f n = d

theorem sequence_is_arithmetic (a : ℕ) (h : a > 0) :
  is_arithmetic_sequence (a_n a) := by
  sorry

end NUMINAMATH_CALUDE_sequence_is_arithmetic_l2326_232610


namespace NUMINAMATH_CALUDE_unique_assignment_exists_l2326_232626

-- Define the types for our images
inductive Image : Type
| cat
| chicken
| crab
| bear
| goat

-- Define a function type that assigns digits to images
def ImageAssignment := Image → Nat

-- Define the conditions for the row and column sums
def satisfiesRowSums (assignment : ImageAssignment) : Prop :=
  assignment Image.cat + assignment Image.chicken + assignment Image.crab + assignment Image.bear + assignment Image.goat = 15 ∧
  assignment Image.goat + assignment Image.goat + assignment Image.crab + assignment Image.bear + assignment Image.bear = 16 ∧
  assignment Image.chicken + assignment Image.chicken + assignment Image.goat + assignment Image.cat + assignment Image.cat = 15 ∧
  assignment Image.crab + assignment Image.crab + assignment Image.crab + assignment Image.crab + assignment Image.crab = 10 ∧
  assignment Image.bear + assignment Image.bear + assignment Image.chicken + assignment Image.chicken + assignment Image.goat = 21

def satisfiesColumnSums (assignment : ImageAssignment) : Prop :=
  assignment Image.cat + assignment Image.goat + assignment Image.chicken + assignment Image.crab + assignment Image.bear = 15 ∧
  assignment Image.chicken + assignment Image.bear + assignment Image.goat + assignment Image.crab + assignment Image.bear = 13 ∧
  assignment Image.crab + assignment Image.crab + assignment Image.chicken + assignment Image.chicken + assignment Image.goat = 17 ∧
  assignment Image.bear + assignment Image.bear + assignment Image.goat + assignment Image.cat + assignment Image.chicken = 20 ∧
  assignment Image.goat + assignment Image.bear + assignment Image.cat + assignment Image.crab + assignment Image.crab = 11

-- Define the condition for different images having different digits
def differentImagesHaveDifferentDigits (assignment : ImageAssignment) : Prop :=
  assignment Image.cat ≠ assignment Image.chicken ∧
  assignment Image.cat ≠ assignment Image.crab ∧
  assignment Image.cat ≠ assignment Image.bear ∧
  assignment Image.cat ≠ assignment Image.goat ∧
  assignment Image.chicken ≠ assignment Image.crab ∧
  assignment Image.chicken ≠ assignment Image.bear ∧
  assignment Image.chicken ≠ assignment Image.goat ∧
  assignment Image.crab ≠ assignment Image.bear ∧
  assignment Image.crab ≠ assignment Image.goat ∧
  assignment Image.bear ≠ assignment Image.goat

-- The main theorem
theorem unique_assignment_exists : 
  ∃! assignment : ImageAssignment, 
    satisfiesRowSums assignment ∧ 
    satisfiesColumnSums assignment ∧ 
    differentImagesHaveDifferentDigits assignment ∧
    assignment Image.cat = 1 ∧
    assignment Image.chicken = 5 ∧
    assignment Image.crab = 2 ∧
    assignment Image.bear = 4 ∧
    assignment Image.goat = 3 :=
  sorry


end NUMINAMATH_CALUDE_unique_assignment_exists_l2326_232626


namespace NUMINAMATH_CALUDE_min_value_xy_over_x2_plus_y2_l2326_232629

theorem min_value_xy_over_x2_plus_y2 (x y : ℝ) 
  (hx : 1/4 ≤ x ∧ x ≤ 2/3) (hy : 1/5 ≤ y ∧ y ≤ 1/2) :
  ∃ (min_val : ℝ), min_val = 2/5 ∧ 
    ∀ (z w : ℝ), (1/4 ≤ z ∧ z ≤ 2/3) → (1/5 ≤ w ∧ w ≤ 1/2) → 
      z * w / (z^2 + w^2) ≥ min_val := by
  sorry

end NUMINAMATH_CALUDE_min_value_xy_over_x2_plus_y2_l2326_232629


namespace NUMINAMATH_CALUDE_largest_integer_solution_negative_six_is_largest_largest_integer_is_negative_six_l2326_232698

theorem largest_integer_solution (x : ℤ) : (7 - 3 * x > 22) ↔ (x ≤ -6) :=
  sorry

theorem negative_six_is_largest : ∃ (x : ℤ), (7 - 3 * x > 22) ∧ (∀ (y : ℤ), y > x → ¬(7 - 3 * y > 22)) :=
  sorry

theorem largest_integer_is_negative_six : (∃! (x : ℤ), (7 - 3 * x > 22) ∧ (∀ (y : ℤ), y > x → ¬(7 - 3 * y > 22))) ∧ 
  (∀ (x : ℤ), (7 - 3 * x > 22) ∧ (∀ (y : ℤ), y > x → ¬(7 - 3 * y > 22)) → x = -6) :=
  sorry

end NUMINAMATH_CALUDE_largest_integer_solution_negative_six_is_largest_largest_integer_is_negative_six_l2326_232698


namespace NUMINAMATH_CALUDE_earring_percentage_l2326_232633

theorem earring_percentage :
  ∀ (bella_earrings monica_earrings rachel_earrings : ℕ),
    bella_earrings = 10 →
    monica_earrings = 2 * rachel_earrings →
    bella_earrings + monica_earrings + rachel_earrings = 70 →
    (bella_earrings : ℚ) / (monica_earrings : ℚ) * 100 = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_earring_percentage_l2326_232633


namespace NUMINAMATH_CALUDE_james_sales_percentage_l2326_232644

/-- Represents the number of houses visited on the first day -/
def houses_day1 : ℕ := 20

/-- Represents the number of houses visited on the second day -/
def houses_day2 : ℕ := 2 * houses_day1

/-- Represents the number of items sold per house each day -/
def items_per_house : ℕ := 2

/-- Represents the total number of items sold over both days -/
def total_items_sold : ℕ := 104

/-- Calculates the percentage of houses sold to on the second day -/
def percentage_sold_day2 : ℚ :=
  (total_items_sold - houses_day1 * items_per_house) / (2 * houses_day2)

theorem james_sales_percentage :
  percentage_sold_day2 = 4/5 := by sorry

end NUMINAMATH_CALUDE_james_sales_percentage_l2326_232644
