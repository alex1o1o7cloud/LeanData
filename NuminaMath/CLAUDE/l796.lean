import Mathlib

namespace NUMINAMATH_CALUDE_x_asymptotics_l796_79623

/-- The Lambert W function -/
noncomputable def W : ℝ → ℝ := sorry

/-- Asymptotic equivalence -/
def asymptotic_equiv (f g : ℕ → ℝ) : Prop :=
  ∃ (c₁ c₂ : ℝ) (N : ℕ), c₁ > 0 ∧ c₂ > 0 ∧ ∀ n ≥ N, c₁ * g n ≤ f n ∧ f n ≤ c₂ * g n

theorem x_asymptotics (n : ℕ) (x : ℝ) (h : x^x = n) :
  asymptotic_equiv (λ n => x) (λ n => Real.log n / Real.log (Real.log n)) :=
sorry

end NUMINAMATH_CALUDE_x_asymptotics_l796_79623


namespace NUMINAMATH_CALUDE_largest_number_with_constraints_l796_79669

def is_valid_number (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 4 ∨ d = 2

def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem largest_number_with_constraints :
  ∀ n : ℕ, 
    is_valid_number n ∧ 
    digit_sum n = 20 →
    n ≤ 44444 :=
by sorry

end NUMINAMATH_CALUDE_largest_number_with_constraints_l796_79669


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l796_79609

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  a 4 + a 7 = 2 →
  a 5 * a 6 = -8 →
  a 1 + a 10 = -7 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l796_79609


namespace NUMINAMATH_CALUDE_tangent_product_l796_79603

theorem tangent_product (α β : ℝ) (h : 2 * Real.cos (2 * α + β) + 3 * Real.cos β = 0) :
  Real.tan (α + β) * Real.tan α = -5 := by sorry

end NUMINAMATH_CALUDE_tangent_product_l796_79603


namespace NUMINAMATH_CALUDE_ln_x_plus_one_negative_condition_l796_79674

theorem ln_x_plus_one_negative_condition (x : ℝ) :
  (∀ x, (Real.log (x + 1) < 0) → (x < 0)) ∧
  (∃ x, x < 0 ∧ ¬(Real.log (x + 1) < 0)) :=
sorry

end NUMINAMATH_CALUDE_ln_x_plus_one_negative_condition_l796_79674


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l796_79642

theorem pure_imaginary_complex_number (m : ℝ) : 
  (((m^2 - 5*m + 6) : ℂ) + (m^2 - 3*m)*I = (0 : ℂ) + ((m^2 - 3*m) : ℝ)*I) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l796_79642


namespace NUMINAMATH_CALUDE_normal_distribution_probability_l796_79657

-- Define a normal distribution
def normal_distribution (μ σ : ℝ) : Type := ℝ

-- Define the probability function
noncomputable def P {α : Type} (event : Set α) : ℝ := sorry

-- Define the random variable ξ
def ξ : normal_distribution 0 σ := sorry

-- State the theorem
theorem normal_distribution_probability 
  (h : P {x | -2 ≤ x ∧ x ≤ 0} = 0.4) : 
  P {x | x > 2} = 0.1 := by sorry

end NUMINAMATH_CALUDE_normal_distribution_probability_l796_79657


namespace NUMINAMATH_CALUDE_power_function_k_values_l796_79624

/-- A function is a power function if it has the form f(x) = ax^n where a is a non-zero constant and n is a real number. -/
def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ (a n : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^n

/-- The main theorem: if y=(k^2-k-5)x^3 is a power function, then k = 3 or k = -2 -/
theorem power_function_k_values (k : ℝ) :
  is_power_function (λ x => (k^2 - k - 5) * x^3) → k = 3 ∨ k = -2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_k_values_l796_79624


namespace NUMINAMATH_CALUDE_unique_sum_property_l796_79608

theorem unique_sum_property (A : ℕ) : 
  (0 ≤ A * (A - 1999) / 2 ∧ A * (A - 1999) / 2 ≤ 999) ↔ A = 1999 :=
sorry

end NUMINAMATH_CALUDE_unique_sum_property_l796_79608


namespace NUMINAMATH_CALUDE_quadratic_root_k_value_l796_79661

theorem quadratic_root_k_value : ∀ k : ℝ, 
  ((-1 : ℝ)^2 + 3*(-1) + k = 0) → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_k_value_l796_79661


namespace NUMINAMATH_CALUDE_exist_good_coloring_l796_79697

/-- The set of colors --/
inductive Color
| red
| white

/-- The type of coloring functions --/
def Coloring := Fin 2017 → Color

/-- Checks if a sequence is an arithmetic progression --/
def isArithmeticSequence (s : Fin n → Fin 2017) : Prop :=
  ∃ a d : ℕ, ∀ i : Fin n, s i = a + i.val * d

/-- The main theorem --/
theorem exist_good_coloring (n : ℕ) (h : n ≥ 18) :
  ∃ f : Coloring, ∀ s : Fin n → Fin 2017, 
    isArithmeticSequence s → 
    ∃ i j : Fin n, f (s i) ≠ f (s j) :=
sorry

end NUMINAMATH_CALUDE_exist_good_coloring_l796_79697


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l796_79693

/-- The equation of a circle in the form (x - h)^2 + (y - k)^2 = r^2 -/
def CircleEquation (h k r : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

theorem circle_center_and_radius :
  ∃ (h k r : ℝ), 
    (∀ x y : ℝ, CircleEquation h k r x y ↔ (x - 2)^2 + (y + 3)^2 = 2) ∧
    h = 2 ∧ k = -3 ∧ r = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l796_79693


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l796_79689

theorem trigonometric_equation_solution (x : ℝ) :
  8.471 * (3 * Real.tan x - Real.tan x ^ 3) / (2 - 1 / Real.cos x ^ 2) = 
  (4 + 2 * Real.cos (6 * x / 5)) / (Real.cos (3 * x) + Real.cos x) ↔
  ∃ k : ℤ, x = 5 * π / 6 + 10 * π * k / 3 ∧ ¬∃ t : ℤ, k = 2 + 3 * t :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l796_79689


namespace NUMINAMATH_CALUDE_trapezoid_median_theorem_median_is_six_l796_79692

/-- The length of the median of a trapezoid -/
def median_length : ℝ := 6

/-- The length of the longer base of the trapezoid -/
def longer_base : ℝ := 1.5 * median_length

/-- The length of the shorter base of the trapezoid -/
def shorter_base : ℝ := median_length - 3

/-- Theorem: The median of a trapezoid is the average of its bases -/
theorem trapezoid_median_theorem (median : ℝ) (longer_base shorter_base : ℝ) 
  (h1 : longer_base = 1.5 * median) 
  (h2 : shorter_base = median - 3) : 
  median = (longer_base + shorter_base) / 2 := by sorry

/-- Proof that the median length is 6 units -/
theorem median_is_six : 
  median_length = 6 ∧ 
  longer_base = 1.5 * median_length ∧ 
  shorter_base = median_length - 3 ∧
  median_length = (longer_base + shorter_base) / 2 := by sorry

end NUMINAMATH_CALUDE_trapezoid_median_theorem_median_is_six_l796_79692


namespace NUMINAMATH_CALUDE_system_solution_l796_79682

theorem system_solution (x y z b : ℝ) : 
  (3 * x * y * z - x^3 - y^3 - z^3 = b^3) ∧
  (x + y + z = 2 * b) ∧
  (x^2 + y^2 + z^2 = b^2) →
  ((x = 0 ∧ y = -z) ∨ (y = 0 ∧ x = -z)) ∧ b = 0 :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l796_79682


namespace NUMINAMATH_CALUDE_x_with_three_prime_divisors_including_2_l796_79605

theorem x_with_three_prime_divisors_including_2 (x n : ℕ) :
  x = 2^n - 32 ∧ 
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ p ≠ 2 ∧ q ≠ 2 ∧
    (∀ r : ℕ, Prime r → r ∣ x → r = 2 ∨ r = p ∨ r = q)) →
  x = 2016 ∨ x = 16352 := by
sorry

end NUMINAMATH_CALUDE_x_with_three_prime_divisors_including_2_l796_79605


namespace NUMINAMATH_CALUDE_fraction_value_l796_79691

theorem fraction_value (a b : ℝ) (h1 : |a| = 3) (h2 : |b| = 1/2) (h3 : a > b) :
  a / b = 6 ∨ a / b = -6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l796_79691


namespace NUMINAMATH_CALUDE_exponential_decreasing_l796_79630

theorem exponential_decreasing (a b : ℝ) (h1 : a > b) (h2 : b > 0) : (1/2 : ℝ)^a < (1/2 : ℝ)^b := by
  sorry

end NUMINAMATH_CALUDE_exponential_decreasing_l796_79630


namespace NUMINAMATH_CALUDE_f_range_theorem_l796_79618

-- Define the function f(x)
def f (x : ℝ) : ℝ := (x^2 - 1) * (x^2 - 12*x + 35)

-- State the theorem
theorem f_range_theorem :
  (∀ x : ℝ, f (6 - x) = f x) →  -- Symmetry condition
  (∃ y : ℝ, ∀ x : ℝ, f x ≥ y) ∧ -- Lower bound exists
  (∀ y : ℝ, y ≥ -36 → ∃ x : ℝ, f x = y) -- All values ≥ -36 are in the range
  :=
by sorry

end NUMINAMATH_CALUDE_f_range_theorem_l796_79618


namespace NUMINAMATH_CALUDE_qinJiushaoResult_l796_79611

/-- Qin Jiushao's algorithm for polynomial evaluation -/
def qinJiushaoAlgorithm (n : ℕ) (x : ℕ) : ℕ :=
  let rec loop : ℕ → ℕ → ℕ
    | 0, v => v
    | i+1, v => loop i (x * v + 1)
  loop n 1

/-- Theorem stating the result of Qin Jiushao's algorithm for n=5 and x=2 -/
theorem qinJiushaoResult : qinJiushaoAlgorithm 5 2 = 2^5 + 2^4 + 2^3 + 2^2 + 2 + 1 := by
  sorry

#eval qinJiushaoAlgorithm 5 2

end NUMINAMATH_CALUDE_qinJiushaoResult_l796_79611


namespace NUMINAMATH_CALUDE_factorization_equality_l796_79686

theorem factorization_equality (a b c x y : ℝ) :
  -a * (x - y) - b * (y - x) + c * (x - y) = (y - x) * (a - b - c) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l796_79686


namespace NUMINAMATH_CALUDE_rational_solutions_quadratic_l796_79676

theorem rational_solutions_quadratic (k : ℕ+) : 
  (∃ x : ℚ, k * x^2 + 26 * x + k = 0) ↔ k = 5 := by
sorry

end NUMINAMATH_CALUDE_rational_solutions_quadratic_l796_79676


namespace NUMINAMATH_CALUDE_shortest_path_l796_79617

/-- Represents an elevator in the building --/
inductive Elevator
| A | B | C | D | E | F | G | H | I | J

/-- Represents a floor in the building --/
inductive Floor
| First | Second

/-- Represents a location on a floor --/
structure Location where
  floor : Floor
  x : ℕ
  y : ℕ

/-- Defines the building layout --/
def building_layout : List (Elevator × Location) := sorry

/-- Defines the entrance location --/
def entrance : Location := sorry

/-- Defines the exit location --/
def exit : Location := sorry

/-- Determines if an elevator leads to a confined room --/
def is_confined (e : Elevator) : Bool := sorry

/-- Calculates the distance between two locations --/
def distance (l1 l2 : Location) : ℕ := sorry

/-- Determines if a path is valid (uses only non-confined elevators) --/
def is_valid_path (path : List Elevator) : Bool := sorry

/-- Calculates the total distance of a path --/
def path_distance (path : List Elevator) : ℕ := sorry

/-- The theorem to be proved --/
theorem shortest_path :
  let path := [Elevator.B, Elevator.J, Elevator.G]
  is_valid_path path ∧
  (∀ other_path, is_valid_path other_path → path_distance path ≤ path_distance other_path) :=
sorry

end NUMINAMATH_CALUDE_shortest_path_l796_79617


namespace NUMINAMATH_CALUDE_prime_sum_85_product_166_l796_79698

theorem prime_sum_85_product_166 (p q : ℕ) (hp : Prime p) (hq : Prime q) (hsum : p + q = 85) :
  p * q = 166 := by
sorry

end NUMINAMATH_CALUDE_prime_sum_85_product_166_l796_79698


namespace NUMINAMATH_CALUDE_max_value_of_f_l796_79607

theorem max_value_of_f (x : ℝ) : 
  let f := fun (x : ℝ) => 1 / (1 - x * (1 - x))
  f x ≤ 4/3 ∧ ∃ y, f y = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l796_79607


namespace NUMINAMATH_CALUDE_score_statistics_l796_79641

def scores : List ℕ := [42, 43, 43, 43, 44, 44, 45, 45, 46, 46, 46, 46, 46, 47, 47]

def mode (l : List ℕ) : ℕ := sorry

def median (l : List ℕ) : ℚ := sorry

theorem score_statistics :
  mode scores = 46 ∧ median scores = 45 := by sorry

end NUMINAMATH_CALUDE_score_statistics_l796_79641


namespace NUMINAMATH_CALUDE_graph_translation_l796_79604

-- Define the original function
def original_function (x : ℝ) : ℝ := 3 * x - 4

-- Define the transformation (moving up by 2 units)
def transform (f : ℝ → ℝ) : ℝ → ℝ := λ x => f x + 2

-- State the theorem
theorem graph_translation :
  ∀ x : ℝ, transform original_function x = 3 * x - 2 := by
sorry

end NUMINAMATH_CALUDE_graph_translation_l796_79604


namespace NUMINAMATH_CALUDE_integer_solutions_of_equation_l796_79646

theorem integer_solutions_of_equation :
  ∀ m n : ℤ, m^2 + 2*m = n^4 + 20*n^3 + 104*n^2 + 40*n + 2003 →
  ((m = 128 ∧ n = 7) ∨ (m = 128 ∧ n = -17)) := by
sorry

end NUMINAMATH_CALUDE_integer_solutions_of_equation_l796_79646


namespace NUMINAMATH_CALUDE_interest_credited_cents_l796_79655

/-- Represents the interest rate as a decimal (3% per annum) -/
def annual_rate : ℚ := 3 / 100

/-- Represents the time period in years (3 months) -/
def time_period : ℚ := 1 / 4

/-- Represents the final balance after interest is applied -/
def final_balance : ℚ := 310.45

/-- Calculates the initial balance given the final balance, rate, and time -/
def initial_balance : ℚ := final_balance / (1 + annual_rate * time_period)

/-- Calculates the interest credited in dollars -/
def interest_credited : ℚ := final_balance - initial_balance

/-- Theorem stating that the interest credited in cents is 37 -/
theorem interest_credited_cents : 
  (interest_credited * 100).floor = 37 := by sorry

end NUMINAMATH_CALUDE_interest_credited_cents_l796_79655


namespace NUMINAMATH_CALUDE_remainder_equivalence_l796_79637

theorem remainder_equivalence (N : ℤ) (k : ℤ) : 
  N % 18 = 19 → N % 242 = (18 * k + 19) % 242 := by
  sorry

end NUMINAMATH_CALUDE_remainder_equivalence_l796_79637


namespace NUMINAMATH_CALUDE_function_value_at_e_l796_79696

open Real

theorem function_value_at_e (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (∀ x, f x = 2 * (deriv f 1) * log x + x) →
  f (exp 1) = -2 + exp 1 := by
sorry

end NUMINAMATH_CALUDE_function_value_at_e_l796_79696


namespace NUMINAMATH_CALUDE_gcd_437_323_l796_79699

theorem gcd_437_323 : Nat.gcd 437 323 = 19 := by
  sorry

end NUMINAMATH_CALUDE_gcd_437_323_l796_79699


namespace NUMINAMATH_CALUDE_violets_family_ticket_cost_l796_79660

/-- Calculates the total cost of tickets for a family visit to the aquarium. -/
def total_ticket_cost (adult_price child_price : ℕ) (num_adults num_children : ℕ) : ℕ :=
  adult_price * num_adults + child_price * num_children

/-- Proves that the total cost for Violet's family to buy separate tickets is $155. -/
theorem violets_family_ticket_cost :
  total_ticket_cost 35 20 1 6 = 155 := by
  sorry

#eval total_ticket_cost 35 20 1 6

end NUMINAMATH_CALUDE_violets_family_ticket_cost_l796_79660


namespace NUMINAMATH_CALUDE_no_integer_pairs_with_square_diff_150_l796_79690

theorem no_integer_pairs_with_square_diff_150 :
  ¬ ∃ (m n : ℕ), m ≥ n ∧ m^2 - n^2 = 150 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_pairs_with_square_diff_150_l796_79690


namespace NUMINAMATH_CALUDE_sum_of_first_five_terms_l796_79638

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = 2 * a n

theorem sum_of_first_five_terms
  (a : ℕ → ℝ)
  (h1 : geometric_sequence a)
  (h2 : a 1 = 1) :
  a 1 + a 2 + a 3 + a 4 + a 5 = 31 :=
sorry

end NUMINAMATH_CALUDE_sum_of_first_five_terms_l796_79638


namespace NUMINAMATH_CALUDE_fraction_comparison_l796_79627

theorem fraction_comparison (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x^2 - y^2) / (x - y) > (x^2 + y^2) / (x + y) := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l796_79627


namespace NUMINAMATH_CALUDE_factorial_fraction_equality_l796_79671

theorem factorial_fraction_equality : (4 * Nat.factorial 6 + 24 * Nat.factorial 5) / Nat.factorial 7 = 8 / 7 := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_equality_l796_79671


namespace NUMINAMATH_CALUDE_percentage_problem_l796_79679

theorem percentage_problem (x : ℝ) (a : ℝ) (h1 : (x / 100) * 170 = 85) (h2 : a = 170) : x = 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l796_79679


namespace NUMINAMATH_CALUDE_knickknack_weight_is_six_l796_79622

-- Define the given conditions
def bookcase_max_weight : ℝ := 80
def hardcover_count : ℕ := 70
def hardcover_weight : ℝ := 0.5
def textbook_count : ℕ := 30
def textbook_weight : ℝ := 2
def knickknack_count : ℕ := 3
def weight_over_limit : ℝ := 33

-- Define the total weight of the collection
def total_weight : ℝ := bookcase_max_weight + weight_over_limit

-- Define the weight of hardcover books and textbooks
def books_weight : ℝ := hardcover_count * hardcover_weight + textbook_count * textbook_weight

-- Define the total weight of knick-knacks
def knickknacks_total_weight : ℝ := total_weight - books_weight

-- Theorem to prove
theorem knickknack_weight_is_six :
  knickknacks_total_weight / knickknack_count = 6 := by
  sorry

end NUMINAMATH_CALUDE_knickknack_weight_is_six_l796_79622


namespace NUMINAMATH_CALUDE_complex_magnitude_l796_79629

theorem complex_magnitude (z : ℂ) : z = (2 - I) / (1 + I) → Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l796_79629


namespace NUMINAMATH_CALUDE_post_office_mail_theorem_l796_79643

/-- Calculates the total number of pieces of mail handled by a post office in six months -/
def mail_in_six_months (letters_per_day : ℕ) (packages_per_day : ℕ) (days_per_month : ℕ) : ℕ :=
  let total_per_day := letters_per_day + packages_per_day
  let total_per_month := total_per_day * days_per_month
  total_per_month * 6

/-- Theorem stating that a post office receiving 60 letters and 20 packages per day
    handles 14400 pieces of mail in six months, assuming 30-day months -/
theorem post_office_mail_theorem :
  mail_in_six_months 60 20 30 = 14400 := by
  sorry

#eval mail_in_six_months 60 20 30

end NUMINAMATH_CALUDE_post_office_mail_theorem_l796_79643


namespace NUMINAMATH_CALUDE_fraction_inequality_solution_set_l796_79652

theorem fraction_inequality_solution_set (x : ℝ) :
  (x + 1) / (x + 2) < 0 ↔ -2 < x ∧ x < -1 :=
sorry

end NUMINAMATH_CALUDE_fraction_inequality_solution_set_l796_79652


namespace NUMINAMATH_CALUDE_linear_function_slope_condition_l796_79678

/-- Given a linear function y = (m-2)x + 2 + m with two points on its graph,
    prove that if x₁ < x₂ and y₁ > y₂, then m < 2 -/
theorem linear_function_slope_condition (m : ℝ) (x₁ x₂ y₁ y₂ : ℝ)
  (h1 : y₁ = (m - 2) * x₁ + 2 + m)
  (h2 : y₂ = (m - 2) * x₂ + 2 + m)
  (h3 : x₁ < x₂)
  (h4 : y₁ > y₂) :
  m < 2 :=
sorry

end NUMINAMATH_CALUDE_linear_function_slope_condition_l796_79678


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l796_79632

theorem stratified_sampling_theorem (total_population : ℕ) (category_size : ℕ) (sample_size : ℕ) 
  (h1 : total_population = 100)
  (h2 : category_size = 30)
  (h3 : sample_size = 20) :
  (category_size : ℚ) / total_population * sample_size = 6 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l796_79632


namespace NUMINAMATH_CALUDE_x_squared_equals_one_l796_79659

theorem x_squared_equals_one (x : ℝ) (h1 : x > 0) (h2 : Real.sin (Real.arctan x) = 1 / x) : x^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_equals_one_l796_79659


namespace NUMINAMATH_CALUDE_row_swap_property_l796_79628

def row_swap_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![0, 1; 1, 0]

theorem row_swap_property (A : Matrix (Fin 2) (Fin 2) ℝ) :
  row_swap_matrix * A = Matrix.of (λ i j => A (1 - i) j) := by
  sorry

end NUMINAMATH_CALUDE_row_swap_property_l796_79628


namespace NUMINAMATH_CALUDE_equation_represents_ellipse_l796_79633

/-- The equation x^2 + 2y^2 - 6x - 8y + 9 = 0 represents an ellipse -/
theorem equation_represents_ellipse :
  ∃ (h k a b : ℝ), a > 0 ∧ b > 0 ∧
    ∀ (x y : ℝ), x^2 + 2*y^2 - 6*x - 8*y + 9 = 0 ↔
      ((x - h)^2 / a^2) + ((y - k)^2 / b^2) = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_equation_represents_ellipse_l796_79633


namespace NUMINAMATH_CALUDE_carlas_daily_collection_l796_79610

/-- Represents the number of items Carla needs to collect each day for her project -/
def daily_collection_amount (leaves bugs days : ℕ) : ℕ :=
  (leaves + bugs) / days

/-- Proves that Carla needs to collect 5 items per day given the project conditions -/
theorem carlas_daily_collection :
  daily_collection_amount 30 20 10 = 5 := by
  sorry

#eval daily_collection_amount 30 20 10

end NUMINAMATH_CALUDE_carlas_daily_collection_l796_79610


namespace NUMINAMATH_CALUDE_jackpot_probability_6_45_100_l796_79695

/-- Represents the lottery "6 out of 45" -/
structure Lottery :=
  (total_numbers : Nat)
  (numbers_to_choose : Nat)

/-- Represents a player's bet in the lottery -/
structure Bet :=
  (number_of_tickets : Nat)

/-- Calculate the number of combinations for choosing k items from n items -/
def choose (n k : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- Calculate the probability of hitting the jackpot -/
def jackpot_probability (l : Lottery) (b : Bet) : ℚ :=
  b.number_of_tickets / (choose l.total_numbers l.numbers_to_choose)

/-- Theorem: The probability of hitting the jackpot in a "6 out of 45" lottery with 100 unique tickets -/
theorem jackpot_probability_6_45_100 :
  let l : Lottery := ⟨45, 6⟩
  let b : Bet := ⟨100⟩
  jackpot_probability l b = 100 / 8145060 := by sorry

end NUMINAMATH_CALUDE_jackpot_probability_6_45_100_l796_79695


namespace NUMINAMATH_CALUDE_rectangular_prism_sum_l796_79670

/-- A rectangular prism is a three-dimensional shape with rectangular faces. -/
structure RectangularPrism where
  edges : Nat
  corners : Nat
  faces : Nat

/-- The properties of a standard rectangular prism. -/
def standardPrism : RectangularPrism :=
  { edges := 12
  , corners := 8
  , faces := 6 }

/-- The sum of edges, corners, and faces of a rectangular prism is 26. -/
theorem rectangular_prism_sum :
  standardPrism.edges + standardPrism.corners + standardPrism.faces = 26 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_sum_l796_79670


namespace NUMINAMATH_CALUDE_complex_expression_evaluation_l796_79619

theorem complex_expression_evaluation :
  - Real.sqrt 3 * Real.sqrt 6 + abs (1 - Real.sqrt 2) - (1/3)⁻¹ = -4 * Real.sqrt 2 - 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_evaluation_l796_79619


namespace NUMINAMATH_CALUDE_no_real_with_negative_sum_of_abs_and_square_l796_79648

theorem no_real_with_negative_sum_of_abs_and_square :
  ¬ (∃ x : ℝ, abs x + x^2 < 0) := by
sorry

end NUMINAMATH_CALUDE_no_real_with_negative_sum_of_abs_and_square_l796_79648


namespace NUMINAMATH_CALUDE_floor_plus_x_eq_seventeen_fourths_l796_79600

theorem floor_plus_x_eq_seventeen_fourths :
  ∃! (x : ℚ), ⌊x⌋ + x = 17/4 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_floor_plus_x_eq_seventeen_fourths_l796_79600


namespace NUMINAMATH_CALUDE_little_twelve_games_l796_79606

/-- Represents a basketball conference with two divisions -/
structure BasketballConference :=
  (teams_per_division : ℕ)
  (intra_division_games : ℕ)
  (inter_division_games : ℕ)

/-- The Little Twelve Basketball Conference setup -/
def little_twelve : BasketballConference :=
  { teams_per_division := 6,
    intra_division_games := 2,
    inter_division_games := 2 }

/-- Calculate the total number of conference games -/
def total_conference_games (conf : BasketballConference) : ℕ :=
  let total_teams := 2 * conf.teams_per_division
  let games_per_team := (conf.teams_per_division - 1) * conf.intra_division_games +
                        conf.teams_per_division * conf.inter_division_games
  (total_teams * games_per_team) / 2

theorem little_twelve_games :
  total_conference_games little_twelve = 132 := by
  sorry

end NUMINAMATH_CALUDE_little_twelve_games_l796_79606


namespace NUMINAMATH_CALUDE_max_value_theorem_l796_79673

theorem max_value_theorem (a b c : ℝ) (h : a^2 + b^2 + c^2 = 1) :
  ∃ (M : ℝ), M = 3 ∧ ∀ (x y z : ℝ), x^2 + y^2 + z^2 = 1 → 3*x*y - 3*y*z + 2*z^2 ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l796_79673


namespace NUMINAMATH_CALUDE_equilateral_triangle_side_length_l796_79687

/-- 
Given an equilateral triangle with a perimeter of 63 cm, 
prove that the length of one side is 21 cm.
-/
theorem equilateral_triangle_side_length 
  (perimeter : ℝ) 
  (is_equilateral : Bool) :
  perimeter = 63 ∧ is_equilateral = true → 
  ∃ (side_length : ℝ), side_length = 21 ∧ perimeter = 3 * side_length :=
by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_side_length_l796_79687


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l796_79625

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ) (a₁ : ℝ), r ≠ 0 ∧ ∀ n : ℕ, a n = a₁ * r^(n-1)

theorem geometric_sequence_sixth_term 
  (a : ℕ → ℝ) 
  (h_geo : is_geometric_sequence a) 
  (h_sum1 : a 2 + a 4 = 20) 
  (h_sum2 : a 3 + a 5 = 40) : 
  a 6 = 64 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l796_79625


namespace NUMINAMATH_CALUDE_quadratic_vertex_ordinate_l796_79626

theorem quadratic_vertex_ordinate 
  (a b c : ℝ) 
  (d : ℝ) 
  (h1 : a ≠ 0) 
  (h2 : b^2 - 4*a*c > 0) 
  (h3 : d = (Real.sqrt (b^2 - 4*a*c)) / a) :
  ∃! y : ℝ, y = -a * d^2 / 4 ∧ 
    y = a * (-b / (2*a))^2 + b * (-b / (2*a)) + c :=
sorry

end NUMINAMATH_CALUDE_quadratic_vertex_ordinate_l796_79626


namespace NUMINAMATH_CALUDE_tournament_games_l796_79634

/-- Calculates the number of games in a single-elimination tournament. -/
def gamesInTournament (n : ℕ) : ℕ := n - 1

/-- The number of teams in the tournament -/
def numTeams : ℕ := 20

theorem tournament_games :
  gamesInTournament numTeams = 19 := by sorry

end NUMINAMATH_CALUDE_tournament_games_l796_79634


namespace NUMINAMATH_CALUDE_largest_fraction_l796_79635

theorem largest_fraction : 
  let fractions := [5/12, 7/15, 29/58, 151/303, 199/400]
  ∀ x ∈ fractions, (29:ℚ)/58 ≥ x := by sorry

end NUMINAMATH_CALUDE_largest_fraction_l796_79635


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_4_l796_79602

-- Define the motion equation
def s (t : ℝ) : ℝ := 1 - t + t^2

-- Define the instantaneous velocity (derivative of s)
def v (t : ℝ) : ℝ := 2 * t - 1

-- Theorem statement
theorem instantaneous_velocity_at_4 : v 4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_4_l796_79602


namespace NUMINAMATH_CALUDE_slab_cost_l796_79662

/-- The cost of a slab of beef given the conditions of kabob stick production -/
theorem slab_cost (cubes_per_stick : ℕ) (cubes_per_slab : ℕ) (sticks : ℕ) (total_cost : ℕ) : 
  cubes_per_stick = 4 →
  cubes_per_slab = 80 →
  sticks = 40 →
  total_cost = 50 →
  (total_cost : ℚ) / ((sticks * cubes_per_stick : ℕ) / cubes_per_slab : ℚ) = 25 := by
  sorry

end NUMINAMATH_CALUDE_slab_cost_l796_79662


namespace NUMINAMATH_CALUDE_range_of_k_l796_79665

-- Define the equation
def equation (x y k : ℝ) : Prop := x + y - 6 * Real.sqrt (x + y) + 3 * k = 0

-- Define the condition that the equation represents only one line
def represents_one_line (k : ℝ) : Prop :=
  ∀ x y : ℝ, equation x y k → ∃! (x' y' : ℝ), equation x' y' k ∧ x' = x ∧ y' = y

-- Theorem statement
theorem range_of_k (k : ℝ) :
  represents_one_line k ↔ k = 3 ∨ k < 0 := by sorry

end NUMINAMATH_CALUDE_range_of_k_l796_79665


namespace NUMINAMATH_CALUDE_sqrt_a_is_integer_l796_79680

theorem sqrt_a_is_integer (a b : ℕ) (h1 : a > b) (h2 : b > 0)
  (h3 : ∃ k : ℤ, (Real.sqrt (Real.sqrt a + Real.sqrt b) + Real.sqrt (Real.sqrt a - Real.sqrt b)) = k) :
  ∃ n : ℕ, Real.sqrt a = n :=
sorry

end NUMINAMATH_CALUDE_sqrt_a_is_integer_l796_79680


namespace NUMINAMATH_CALUDE_additional_painters_eq_four_l796_79656

/-- The number of additional people required to paint a fence in 2 hours,
    given that 8 people can paint it in 3 hours and all people paint at the same rate. -/
def additional_painters : ℕ :=
  let initial_painters : ℕ := 8
  let initial_time : ℕ := 3
  let new_time : ℕ := 2
  let total_work : ℕ := initial_painters * initial_time
  let new_painters : ℕ := total_work / new_time
  new_painters - initial_painters

theorem additional_painters_eq_four :
  additional_painters = 4 :=
sorry

end NUMINAMATH_CALUDE_additional_painters_eq_four_l796_79656


namespace NUMINAMATH_CALUDE_log_158489_between_integers_l796_79640

theorem log_158489_between_integers : ∃ p q : ℤ,
  (p : ℝ) < Real.log 158489 / Real.log 10 ∧
  Real.log 158489 / Real.log 10 < (q : ℝ) ∧
  q = p + 1 ∧
  p + q = 11 := by
  sorry

end NUMINAMATH_CALUDE_log_158489_between_integers_l796_79640


namespace NUMINAMATH_CALUDE_number_of_slips_with_three_l796_79663

/-- Given a bag of slips with numbers 3 or 8, prove the number of slips with 3 --/
theorem number_of_slips_with_three
  (total_slips : ℕ)
  (number_options : Fin 2 → ℕ)
  (expected_value : ℚ)
  (h_total : total_slips = 15)
  (h_options : number_options = ![3, 8])
  (h_expected : expected_value = 5) :
  ∃ (slips_with_three : ℕ),
    slips_with_three = 9 ∧
    (slips_with_three : ℚ) / total_slips * number_options 0 +
    ((total_slips - slips_with_three) : ℚ) / total_slips * number_options 1 = expected_value :=
by sorry

end NUMINAMATH_CALUDE_number_of_slips_with_three_l796_79663


namespace NUMINAMATH_CALUDE_simplify_expression_l796_79667

theorem simplify_expression (x y : ℝ) : (3 * x + 22) + (150 * y + 22) = 3 * x + 150 * y + 44 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l796_79667


namespace NUMINAMATH_CALUDE_additional_employees_hired_l796_79616

/-- Calculates the number of additional employees hired by a company --/
theorem additional_employees_hired (
  initial_employees : ℕ)
  (hourly_wage : ℚ)
  (hours_per_day : ℕ)
  (days_per_week : ℕ)
  (weeks_per_month : ℕ)
  (new_total_wages : ℚ)
  (h1 : initial_employees = 500)
  (h2 : hourly_wage = 12)
  (h3 : hours_per_day = 10)
  (h4 : days_per_week = 5)
  (h5 : weeks_per_month = 4)
  (h6 : new_total_wages = 1680000) :
  (new_total_wages - (initial_employees * hourly_wage * hours_per_day * days_per_week * weeks_per_month)) / 
  (hourly_wage * hours_per_day * days_per_week * weeks_per_month) = 200 := by
  sorry

#check additional_employees_hired

end NUMINAMATH_CALUDE_additional_employees_hired_l796_79616


namespace NUMINAMATH_CALUDE_chicken_egg_problem_l796_79668

theorem chicken_egg_problem (initial_eggs : ℕ) (used_eggs : ℕ) (eggs_per_chicken : ℕ) (final_eggs : ℕ) : 
  initial_eggs = 10 →
  used_eggs = 5 →
  eggs_per_chicken = 3 →
  final_eggs = 11 →
  (final_eggs - (initial_eggs - used_eggs)) / eggs_per_chicken = 2 :=
by sorry

end NUMINAMATH_CALUDE_chicken_egg_problem_l796_79668


namespace NUMINAMATH_CALUDE_profit_percent_l796_79636

/-- Given an article with cost price C and selling price P, 
    where selling at 2/3 of P results in a 14% loss,
    prove that selling at P results in a 29% profit -/
theorem profit_percent (C P : ℝ) (h : (2/3) * P = 0.86 * C) :
  (P - C) / C * 100 = 29 := by
  sorry

end NUMINAMATH_CALUDE_profit_percent_l796_79636


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l796_79653

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 2 - x) ↔ x ≤ 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l796_79653


namespace NUMINAMATH_CALUDE_nine_fourth_equals_three_two_m_l796_79631

theorem nine_fourth_equals_three_two_m (m : ℕ) : 9^4 = 3^(2*m) → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_nine_fourth_equals_three_two_m_l796_79631


namespace NUMINAMATH_CALUDE_sum_of_products_negative_max_greater_or_equal_cube_root_four_l796_79649

-- Define the conditions
def sum_zero (a b c : ℝ) : Prop := a + b + c = 0
def product_one (a b c : ℝ) : Prop := a * b * c = 1

-- Define the theorems to prove
theorem sum_of_products_negative (a b c : ℝ) (h1 : sum_zero a b c) (h2 : product_one a b c) :
  a * b + b * c + c * a < 0 :=
sorry

theorem max_greater_or_equal_cube_root_four (a b c : ℝ) (h1 : sum_zero a b c) (h2 : product_one a b c) :
  max a (max b c) ≥ (4 : ℝ) ^ (1/3) :=
sorry

end NUMINAMATH_CALUDE_sum_of_products_negative_max_greater_or_equal_cube_root_four_l796_79649


namespace NUMINAMATH_CALUDE_f_negative_a_equals_negative_three_l796_79644

noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x + Real.sqrt (4 * x^2 + 1)) - 2 / (2^x + 1)

theorem f_negative_a_equals_negative_three (a : ℝ) (h : f a = 1) : f (-a) = -3 := by
  sorry

end NUMINAMATH_CALUDE_f_negative_a_equals_negative_three_l796_79644


namespace NUMINAMATH_CALUDE_flag_design_count_l796_79675

/-- The number of possible colors for each stripe -/
def num_colors : ℕ := 3

/-- The number of stripes on the flag -/
def num_stripes : ℕ := 3

/-- The number of possible flag designs -/
def num_flag_designs : ℕ := num_colors ^ num_stripes

theorem flag_design_count :
  num_flag_designs = 27 :=
sorry

end NUMINAMATH_CALUDE_flag_design_count_l796_79675


namespace NUMINAMATH_CALUDE_virginia_april_rainfall_l796_79613

/-- Calculates the rainfall in April given the rainfall in other months and the average -/
def april_rainfall (march may june july average : ℝ) : ℝ :=
  5 * average - (march + may + june + july)

/-- Theorem stating that given the specified rainfall amounts and average, April's rainfall was 4.5 inches -/
theorem virginia_april_rainfall :
  let march : ℝ := 3.79
  let may : ℝ := 3.95
  let june : ℝ := 3.09
  let july : ℝ := 4.67
  let average : ℝ := 4
  april_rainfall march may june july average = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_virginia_april_rainfall_l796_79613


namespace NUMINAMATH_CALUDE_faye_bought_48_books_l796_79639

/-- The number of coloring books Faye initially had -/
def initial_books : ℕ := 34

/-- The number of coloring books Faye gave away -/
def books_given_away : ℕ := 3

/-- The total number of coloring books Faye had after buying more -/
def final_total : ℕ := 79

/-- The number of coloring books Faye bought -/
def books_bought : ℕ := final_total - (initial_books - books_given_away)

theorem faye_bought_48_books : books_bought = 48 := by
  sorry

end NUMINAMATH_CALUDE_faye_bought_48_books_l796_79639


namespace NUMINAMATH_CALUDE_parabola_focus_l796_79650

-- Define the parabola
def parabola (x y : ℝ) : Prop := 2 * x^2 = -y

-- Define the focus of a parabola
def focus (f : ℝ × ℝ) (p : (ℝ → ℝ → Prop)) : Prop :=
  ∀ x y, p x y → (x - f.1)^2 = 4 * f.2 * (y - f.2)

-- Theorem statement
theorem parabola_focus :
  focus (0, -1/8) parabola :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_l796_79650


namespace NUMINAMATH_CALUDE_savings_problem_l796_79658

theorem savings_problem (S : ℝ) : 
  (S * 1.1 * (2 / 10) = 44) → S = 200 := by sorry

end NUMINAMATH_CALUDE_savings_problem_l796_79658


namespace NUMINAMATH_CALUDE_total_staff_weekdays_and_weekends_l796_79651

def weekday_chefs : ℕ := 16
def weekday_waiters : ℕ := 16
def weekday_busboys : ℕ := 10
def weekday_hostesses : ℕ := 5

def weekend_additional_chefs : ℕ := 5
def weekend_additional_hostesses : ℕ := 2

def chef_leave_percentage : ℚ := 25 / 100
def waiter_leave_percentage : ℚ := 20 / 100
def busboy_leave_percentage : ℚ := 30 / 100
def hostess_leave_percentage : ℚ := 15 / 100

theorem total_staff_weekdays_and_weekends :
  let weekday_chefs_left := weekday_chefs - Int.floor (chef_leave_percentage * weekday_chefs)
  let weekday_waiters_left := weekday_waiters - Int.floor (waiter_leave_percentage * weekday_waiters)
  let weekday_busboys_left := weekday_busboys - Int.floor (busboy_leave_percentage * weekday_busboys)
  let weekday_hostesses_left := weekday_hostesses - Int.floor (hostess_leave_percentage * weekday_hostesses)
  
  let weekday_total := weekday_chefs_left + weekday_waiters_left + weekday_busboys_left + weekday_hostesses_left
  
  let weekend_chefs := weekday_chefs + weekend_additional_chefs
  let weekend_waiters := weekday_waiters_left
  let weekend_busboys := weekday_busboys_left
  let weekend_hostesses := weekday_hostesses + weekend_additional_hostesses
  
  let weekend_total := weekend_chefs + weekend_waiters + weekend_busboys + weekend_hostesses
  
  weekday_total + weekend_total = 84 := by
    sorry

end NUMINAMATH_CALUDE_total_staff_weekdays_and_weekends_l796_79651


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l796_79647

theorem boat_speed_in_still_water 
  (current_speed : ℝ) 
  (downstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (h1 : current_speed = 5) 
  (h2 : downstream_distance = 6.25) 
  (h3 : downstream_time = 0.25) : 
  ∃ (still_water_speed : ℝ), 
    still_water_speed = 20 ∧ 
    downstream_distance = (still_water_speed + current_speed) * downstream_time :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l796_79647


namespace NUMINAMATH_CALUDE_average_sequence_l796_79694

theorem average_sequence (x : ℚ) : 
  (List.sum (List.range 149) + x) / 150 = 50 * x → x = 11175 / 7499 := by
  sorry

end NUMINAMATH_CALUDE_average_sequence_l796_79694


namespace NUMINAMATH_CALUDE_system_of_equations_result_l796_79666

theorem system_of_equations_result (x y : ℝ) 
  (eq1 : 5 * x + y = 19) 
  (eq2 : x + 3 * y = 1) : 
  3 * x + 2 * y = 10 := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_result_l796_79666


namespace NUMINAMATH_CALUDE_p_neither_necessary_nor_sufficient_l796_79672

-- Define the propositions p and q
def p (x y : ℝ) : Prop := x + y ≠ -2
def q (x : ℝ) : Prop := x ≠ 0  -- We use x ≠ 0 to represent a non-trivial condition for x

-- Theorem statement
theorem p_neither_necessary_nor_sufficient (y : ℝ) (h : y ≠ -1) :
  ¬(∀ x, q x → p x y) ∧ ¬(∀ x, p x y → q x) :=
sorry

end NUMINAMATH_CALUDE_p_neither_necessary_nor_sufficient_l796_79672


namespace NUMINAMATH_CALUDE_alcohol_concentration_reduction_specific_alcohol_reduction_l796_79677

/-- Calculates the percentage reduction in alcohol concentration when water is added to a solution. -/
theorem alcohol_concentration_reduction 
  (initial_volume : ℝ) 
  (initial_concentration : ℝ) 
  (water_added : ℝ) : ℝ :=
  let initial_alcohol := initial_volume * initial_concentration
  let final_volume := initial_volume + water_added
  let final_concentration := initial_alcohol / final_volume
  let reduction_percentage := (initial_concentration - final_concentration) / initial_concentration * 100
  by
    -- Proof goes here
    sorry

/-- The specific problem statement -/
theorem specific_alcohol_reduction : 
  alcohol_concentration_reduction 15 0.20 25 = 62.5 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_alcohol_concentration_reduction_specific_alcohol_reduction_l796_79677


namespace NUMINAMATH_CALUDE_determinant_of_specific_matrix_l796_79621

theorem determinant_of_specific_matrix :
  let A : Matrix (Fin 3) (Fin 3) ℤ := !![3, 1, -2; 8, 5, -4; 3, 3, 6]
  Matrix.det A = 48 := by sorry

end NUMINAMATH_CALUDE_determinant_of_specific_matrix_l796_79621


namespace NUMINAMATH_CALUDE_angle_between_vectors_l796_79601

/-- Given vectors a and b in ℝ², prove that the angle between them is π -/
theorem angle_between_vectors (a b : ℝ × ℝ) : 
  a + 2 • b = (2, -4) → 
  3 • a - b = (-8, 16) → 
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) = π :=
by sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l796_79601


namespace NUMINAMATH_CALUDE_student_union_selections_l796_79688

/-- Represents the number of students in each grade of the student union -/
structure StudentUnion where
  freshmen : Nat
  sophomores : Nat
  juniors : Nat

/-- Calculates the number of ways to select one person as president -/
def selectPresident (su : StudentUnion) : Nat :=
  su.freshmen + su.sophomores + su.juniors

/-- Calculates the number of ways to select one person from each grade for the standing committee -/
def selectStandingCommittee (su : StudentUnion) : Nat :=
  su.freshmen * su.sophomores * su.juniors

/-- Calculates the number of ways to select two people from different grades for a city activity -/
def selectCityActivity (su : StudentUnion) : Nat :=
  su.freshmen * su.sophomores + su.sophomores * su.juniors + su.juniors * su.freshmen

theorem student_union_selections (su : StudentUnion) 
  (h1 : su.freshmen = 5) 
  (h2 : su.sophomores = 6) 
  (h3 : su.juniors = 4) : 
  selectPresident su = 15 ∧ 
  selectStandingCommittee su = 120 ∧ 
  selectCityActivity su = 74 := by
  sorry

#eval selectPresident ⟨5, 6, 4⟩
#eval selectStandingCommittee ⟨5, 6, 4⟩
#eval selectCityActivity ⟨5, 6, 4⟩

end NUMINAMATH_CALUDE_student_union_selections_l796_79688


namespace NUMINAMATH_CALUDE_minimum_bushes_for_zucchinis_l796_79664

def containers_per_bush : ℕ := 10
def containers_per_zucchini : ℕ := 3
def desired_zucchinis : ℕ := 36

def minimum_bushes : ℕ :=
  (desired_zucchinis * containers_per_zucchini + containers_per_bush - 1) / containers_per_bush

theorem minimum_bushes_for_zucchinis :
  minimum_bushes = 11 ∧
  minimum_bushes * containers_per_bush ≥ desired_zucchinis * containers_per_zucchini ∧
  (minimum_bushes - 1) * containers_per_bush < desired_zucchinis * containers_per_zucchini :=
by sorry

end NUMINAMATH_CALUDE_minimum_bushes_for_zucchinis_l796_79664


namespace NUMINAMATH_CALUDE_total_coins_l796_79683

/-- Given a number of stacks and coins per stack, proves that the total number of coins
    is equal to the product of these two quantities. -/
theorem total_coins (num_stacks : ℕ) (coins_per_stack : ℕ) :
  num_stacks * coins_per_stack = num_stacks * coins_per_stack := by
  sorry

/-- Calculates the total number of coins Maria has. -/
def maria_coins : ℕ :=
  let num_stacks : ℕ := 10
  let coins_per_stack : ℕ := 6
  num_stacks * coins_per_stack

#eval maria_coins

end NUMINAMATH_CALUDE_total_coins_l796_79683


namespace NUMINAMATH_CALUDE_robot_gather_time_l796_79685

/-- The time (in minutes) it takes a robot to create a battery -/
def create_time : ℕ := 9

/-- The number of robots working simultaneously -/
def num_robots : ℕ := 10

/-- The number of batteries manufactured in 5 hours -/
def batteries_produced : ℕ := 200

/-- The time (in hours) taken to manufacture the batteries -/
def production_time : ℕ := 5

/-- The time (in minutes) it takes a robot to gather materials for a battery -/
def gather_time : ℕ := 6

theorem robot_gather_time :
  gather_time = 6 ∧
  create_time = 9 ∧
  num_robots = 10 ∧
  batteries_produced = 200 ∧
  production_time = 5 →
  num_robots * batteries_produced * (gather_time + create_time) = production_time * 60 :=
by sorry

end NUMINAMATH_CALUDE_robot_gather_time_l796_79685


namespace NUMINAMATH_CALUDE_trapezoid_circle_theorem_l796_79654

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a circle -/
structure Circle :=
  (center : Point)
  (radius : ℝ)

/-- Represents a trapezoid -/
structure Trapezoid :=
  (A B C D : Point)

/-- States that two lines are parallel -/
def parallel (p1 p2 p3 p4 : Point) : Prop := sorry

/-- States that a point lies on a line segment -/
def on_segment (p q r : Point) : Prop := sorry

/-- States that a circle is inscribed in an angle -/
def inscribed_in_angle (c : Circle) (p1 p2 p3 : Point) : Prop := sorry

/-- States that a circle touches a line segment -/
def touches_segment (c : Circle) (p q : Point) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p q : Point) : ℝ := sorry

/-- Calculates the area of a trapezoid -/
def area (t : Trapezoid) : ℝ := sorry

theorem trapezoid_circle_theorem (ABCD : Trapezoid) (Ω : Circle) (E : Point) :
  parallel ABCD.A ABCD.D ABCD.B ABCD.C →
  distance ABCD.A ABCD.D > distance ABCD.B ABCD.C →
  inscribed_in_angle Ω ABCD.B ABCD.A ABCD.D →
  touches_segment Ω ABCD.B ABCD.C →
  on_segment E ABCD.C ABCD.D →
  distance ABCD.C E = 9 →
  distance E ABCD.D = 7 →
  Ω.radius = 6 ∧ area ABCD = 96 + 24 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_circle_theorem_l796_79654


namespace NUMINAMATH_CALUDE_missing_number_equation_l796_79681

theorem missing_number_equation (x : ℤ) : 10111 - 10 * 2 * x = 10011 ↔ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_equation_l796_79681


namespace NUMINAMATH_CALUDE_f_composite_negative_two_l796_79612

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 1 - Real.sqrt x else 2^x

theorem f_composite_negative_two :
  f (f (-2)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_f_composite_negative_two_l796_79612


namespace NUMINAMATH_CALUDE_larger_number_twice_smaller_l796_79645

theorem larger_number_twice_smaller (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : a - b/2 = 3 * (b - b/2)) : a = 2 * b := by
  sorry

end NUMINAMATH_CALUDE_larger_number_twice_smaller_l796_79645


namespace NUMINAMATH_CALUDE_min_value_theorem_l796_79684

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 1

-- State the theorem
theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_f1 : f a b 1 = 2) :
  (1 / a + 4 / b) ≥ 9 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l796_79684


namespace NUMINAMATH_CALUDE_total_wheels_in_parking_lot_l796_79614

/-- Calculates the total number of wheels in a parking lot with cars and motorcycles -/
theorem total_wheels_in_parking_lot 
  (num_cars : ℕ) 
  (num_motorcycles : ℕ) 
  (wheels_per_car : ℕ) 
  (wheels_per_motorcycle : ℕ) 
  (h1 : num_cars = 19) 
  (h2 : num_motorcycles = 11) 
  (h3 : wheels_per_car = 5) 
  (h4 : wheels_per_motorcycle = 2) : 
  num_cars * wheels_per_car + num_motorcycles * wheels_per_motorcycle = 117 := by
sorry

end NUMINAMATH_CALUDE_total_wheels_in_parking_lot_l796_79614


namespace NUMINAMATH_CALUDE_square_sum_inequality_equality_condition_l796_79620

theorem square_sum_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a^2 + b^2 + c^2 + d^2)^2 ≥ (a+b)*(b+c)*(c+d)*(d+a) :=
by sorry

theorem equality_condition (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a^2 + b^2 + c^2 + d^2)^2 = (a+b)*(b+c)*(c+d)*(d+a) ↔ a = b ∧ b = c ∧ c = d :=
by sorry

end NUMINAMATH_CALUDE_square_sum_inequality_equality_condition_l796_79620


namespace NUMINAMATH_CALUDE_greatest_abcba_divisible_by_13_l796_79615

def is_valid_abcba (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000 ∧
  ∃ (a b c : ℕ),
    a < 10 ∧ b < 10 ∧ c < 10 ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    n = 10000 * a + 1000 * b + 100 * c + 10 * b + a

theorem greatest_abcba_divisible_by_13 :
  ∀ n : ℕ, is_valid_abcba n → n % 13 = 0 → n ≤ 83638 :=
by sorry

end NUMINAMATH_CALUDE_greatest_abcba_divisible_by_13_l796_79615
