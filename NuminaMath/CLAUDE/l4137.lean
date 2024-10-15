import Mathlib

namespace NUMINAMATH_CALUDE_complex_number_equality_l4137_413771

theorem complex_number_equality : ∀ (i : ℂ), i^2 = -1 →
  (2 * i) / (2 + i) = 2/5 + 4/5 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equality_l4137_413771


namespace NUMINAMATH_CALUDE_max_d_value_l4137_413731

def a (n : ℕ+) : ℕ := 100 + 2 * n ^ 2

def d (n : ℕ+) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_value :
  (∃ k : ℕ+, d k = 49) ∧ (∀ n : ℕ+, d n ≤ 49) :=
sorry

end NUMINAMATH_CALUDE_max_d_value_l4137_413731


namespace NUMINAMATH_CALUDE_consecutive_integers_divisibility_l4137_413786

theorem consecutive_integers_divisibility (a₁ a₂ a₃ : ℕ) 
  (h1 : a₁ + 1 = a₂) 
  (h2 : a₂ + 1 = a₃) 
  (h3 : 0 < a₁) : 
  a₂^3 ∣ (a₁ * a₂ * a₃ + a₂) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_divisibility_l4137_413786


namespace NUMINAMATH_CALUDE_floor_plus_self_eq_seventeen_fourths_l4137_413768

theorem floor_plus_self_eq_seventeen_fourths :
  ∃! (y : ℚ), ⌊y⌋ + y = 17 / 4 :=
by sorry

end NUMINAMATH_CALUDE_floor_plus_self_eq_seventeen_fourths_l4137_413768


namespace NUMINAMATH_CALUDE_g_monotone_decreasing_iff_a_in_range_l4137_413783

/-- The function g(x) defined as ax³ + 2(1-a)x² - 3ax -/
def g (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 2 * (1 - a) * x^2 - 3 * a * x

/-- g(x) is monotonically decreasing in the interval (-∞, a/3) if and only if -1 ≤ a ≤ 0 -/
theorem g_monotone_decreasing_iff_a_in_range (a : ℝ) :
  (∀ x y, x < y → y < a/3 → g a x > g a y) ↔ -1 ≤ a ∧ a ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_g_monotone_decreasing_iff_a_in_range_l4137_413783


namespace NUMINAMATH_CALUDE_three_slice_toast_l4137_413767

/-- Represents a slice of bread with two sides -/
structure Bread :=
  (side1 : Bool)
  (side2 : Bool)

/-- Represents the state of the toaster -/
structure ToasterState :=
  (slot1 : Option Bread)
  (slot2 : Option Bread)

/-- Represents the toasting process -/
def toast (initial : List Bread) (time : Nat) : List Bread → Prop :=
  sorry

theorem three_slice_toast :
  ∀ (initial : List Bread),
    initial.length = 3 →
    ∀ (b : Bread), b ∈ initial → ¬b.side1 ∧ ¬b.side2 →
    ∃ (final : List Bread),
      toast initial 3 final ∧
      final.length = 3 ∧
      ∀ (b : Bread), b ∈ final → b.side1 ∧ b.side2 :=
by sorry

end NUMINAMATH_CALUDE_three_slice_toast_l4137_413767


namespace NUMINAMATH_CALUDE_math_contest_theorem_l4137_413711

theorem math_contest_theorem (n m k : ℕ) (h_n : n = 200) (h_m : m = 6) (h_k : k = 120)
  (solved : Fin n → Fin m → Prop)
  (h_solved : ∀ j : Fin m, ∃ S : Finset (Fin n), S.card ≥ k ∧ ∀ i ∈ S, solved i j) :
  ∃ i₁ i₂ : Fin n, i₁ ≠ i₂ ∧ ∀ j : Fin m, solved i₁ j ∨ solved i₂ j := by
  sorry

end NUMINAMATH_CALUDE_math_contest_theorem_l4137_413711


namespace NUMINAMATH_CALUDE_boy_age_problem_l4137_413733

theorem boy_age_problem (total_boys : Nat) (avg_age_all : Nat) (avg_age_first_six : Nat) (avg_age_last_six : Nat)
  (h1 : total_boys = 11)
  (h2 : avg_age_all = 50)
  (h3 : avg_age_first_six = 49)
  (h4 : avg_age_last_six = 52) :
  total_boys * avg_age_all = 6 * avg_age_first_six + 6 * avg_age_last_six - 56 := by
  sorry

#check boy_age_problem

end NUMINAMATH_CALUDE_boy_age_problem_l4137_413733


namespace NUMINAMATH_CALUDE_car_distribution_l4137_413798

theorem car_distribution (total_cars : ℕ) (first_supplier : ℕ) (fourth_fifth_each : ℕ) :
  total_cars = 5650000 →
  first_supplier = 1000000 →
  fourth_fifth_each = 325000 →
  ∃ (second_supplier : ℕ),
    second_supplier + first_supplier + (second_supplier + first_supplier) + 2 * fourth_fifth_each = total_cars ∧
    second_supplier = first_supplier + 500000 := by
  sorry

end NUMINAMATH_CALUDE_car_distribution_l4137_413798


namespace NUMINAMATH_CALUDE_function_equals_identity_l4137_413712

theorem function_equals_identity (f : ℝ → ℝ) :
  (Continuous f) →
  (f 0 = 0) →
  (f 1 = 1) →
  (∀ x ∈ (Set.Ioo 0 1), ∃ h : ℝ, 
    0 ≤ x - h ∧ x + h ≤ 1 ∧ 
    f x = (f (x - h) + f (x + h)) / 2) →
  (∀ x ∈ (Set.Icc 0 1), f x = x) := by
sorry

end NUMINAMATH_CALUDE_function_equals_identity_l4137_413712


namespace NUMINAMATH_CALUDE_radical_simplification_l4137_413758

theorem radical_simplification (q : ℝ) (hq : q > 0) :
  Real.sqrt (42 * q) * Real.sqrt (7 * q) * Real.sqrt (3 * q) = 21 * q * Real.sqrt (2 * q) := by
  sorry

end NUMINAMATH_CALUDE_radical_simplification_l4137_413758


namespace NUMINAMATH_CALUDE_loss_percentage_is_29_percent_l4137_413722

-- Define the markup percentage
def markup : ℝ := 0.40

-- Define the discount percentage
def discount : ℝ := 0.07857142857142857

-- Define the loss percentage we want to prove
def target_loss_percentage : ℝ := 0.29

-- Theorem statement
theorem loss_percentage_is_29_percent (cost_price : ℝ) (cost_price_positive : cost_price > 0) :
  let marked_price := cost_price * (1 + markup)
  let selling_price := marked_price * (1 - discount)
  let loss := cost_price - selling_price
  let loss_percentage := loss / cost_price
  loss_percentage = target_loss_percentage :=
by sorry

end NUMINAMATH_CALUDE_loss_percentage_is_29_percent_l4137_413722


namespace NUMINAMATH_CALUDE_min_sum_squares_l4137_413703

theorem min_sum_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 8) :
  ∃ (m : ℝ), (∀ a b c : ℝ, a^3 + b^3 + c^3 - 3*a*b*c = 8 → a^2 + b^2 + c^2 ≥ m) ∧
             (x^2 + y^2 + z^2 = m) ∧
             (m = 40/7) :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l4137_413703


namespace NUMINAMATH_CALUDE_x_plus_q_in_terms_of_q_l4137_413732

theorem x_plus_q_in_terms_of_q (x q : ℝ) (h1 : |x - 5| = q) (h2 : x > 5) : x + q = 2*q + 5 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_q_in_terms_of_q_l4137_413732


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l4137_413700

def M : Set ℤ := {0, 1, 2, 3}
def N : Set ℤ := {-1, 1}

theorem intersection_of_M_and_N : M ∩ N = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l4137_413700


namespace NUMINAMATH_CALUDE_range_of_k_value_of_k_with_condition_l4137_413750

-- Define the quadratic equation
def quadratic (k x : ℝ) : ℝ := x^2 + (2*k - 1)*x + k^2 - 1

-- Define the condition for two real roots
def has_two_real_roots (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic k x₁ = 0 ∧ quadratic k x₂ = 0

-- Define the condition for the sum of squares
def sum_of_squares_condition (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, quadratic k x₁ = 0 ∧ quadratic k x₂ = 0 ∧ x₁^2 + x₂^2 = 16 + x₁*x₂

-- Theorem for the range of k
theorem range_of_k :
  ∀ k : ℝ, has_two_real_roots k → k ≤ 5/4 :=
sorry

-- Theorem for the value of k when sum of squares condition is satisfied
theorem value_of_k_with_condition :
  ∀ k : ℝ, has_two_real_roots k → sum_of_squares_condition k → k = -2 :=
sorry

end NUMINAMATH_CALUDE_range_of_k_value_of_k_with_condition_l4137_413750


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l4137_413776

theorem smallest_integer_satisfying_inequality : 
  (∀ y : ℤ, y < 8 → (y : ℚ) / 4 + 3 / 7 ≤ 9 / 4) ∧ 
  (8 : ℚ) / 4 + 3 / 7 > 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l4137_413776


namespace NUMINAMATH_CALUDE_prob_all_players_odd_sum_l4137_413738

/-- The number of tiles --/
def n : ℕ := 12

/-- The number of odd tiles --/
def odd_tiles : ℕ := n / 2

/-- The number of even tiles --/
def even_tiles : ℕ := n / 2

/-- The number of tiles each player selects --/
def tiles_per_player : ℕ := 4

/-- The number of players --/
def num_players : ℕ := 3

/-- The probability of all players getting an odd sum --/
def prob_all_odd_sum : ℚ := 800 / 963

/-- Theorem stating the probability of all players getting an odd sum --/
theorem prob_all_players_odd_sum :
  let total_distributions := Nat.choose n tiles_per_player * 
                             Nat.choose (n - tiles_per_player) tiles_per_player * 
                             Nat.choose (n - 2 * tiles_per_player) tiles_per_player
  let odd_sum_distributions := (Nat.choose odd_tiles 3 * Nat.choose even_tiles 1)^num_players / 
                               Nat.factorial num_players
  (odd_sum_distributions : ℚ) / total_distributions = prob_all_odd_sum := by
  sorry

end NUMINAMATH_CALUDE_prob_all_players_odd_sum_l4137_413738


namespace NUMINAMATH_CALUDE_students_without_glasses_l4137_413788

theorem students_without_glasses (total : ℕ) (with_glasses_percent : ℚ) 
  (h1 : total = 325) 
  (h2 : with_glasses_percent = 40 / 100) : 
  ↑total * (1 - with_glasses_percent) = 195 := by
  sorry

end NUMINAMATH_CALUDE_students_without_glasses_l4137_413788


namespace NUMINAMATH_CALUDE_remaining_safe_caffeine_l4137_413765

/-- The maximum safe amount of caffeine that can be consumed per day in milligrams. -/
def max_safe_caffeine : ℕ := 500

/-- The amount of caffeine in each energy drink in milligrams. -/
def caffeine_per_drink : ℕ := 120

/-- The number of energy drinks Brandy consumes. -/
def drinks_consumed : ℕ := 4

/-- The remaining safe amount of caffeine Brandy can consume that day in milligrams. -/
theorem remaining_safe_caffeine : 
  max_safe_caffeine - (caffeine_per_drink * drinks_consumed) = 20 := by
  sorry

end NUMINAMATH_CALUDE_remaining_safe_caffeine_l4137_413765


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l4137_413787

-- Define the concept of a function being even or odd
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the condition that both f and g are either odd or even
def BothEvenOrOdd (f g : ℝ → ℝ) : Prop :=
  (IsEven f ∧ IsEven g) ∨ (IsOdd f ∧ IsOdd g)

-- Define the property that the product of f and g is even
def ProductIsEven (f g : ℝ → ℝ) : Prop :=
  IsEven (fun x ↦ f x * g x)

-- Theorem statement
theorem sufficient_not_necessary (f g : ℝ → ℝ) :
  (BothEvenOrOdd f g → ProductIsEven f g) ∧
  ¬(ProductIsEven f g → BothEvenOrOdd f g) := by
  sorry


end NUMINAMATH_CALUDE_sufficient_not_necessary_l4137_413787


namespace NUMINAMATH_CALUDE_orange_price_theorem_l4137_413759

/-- The cost of fruits and the discount policy at a store --/
structure FruitStore where
  apple_cost : ℚ
  banana_cost : ℚ
  discount_per_five : ℚ

/-- A customer's purchase of fruits --/
structure Purchase where
  apples : ℕ
  oranges : ℕ
  bananas : ℕ

/-- Calculate the total cost of a purchase given the store's prices and an orange price --/
def totalCost (store : FruitStore) (purchase : Purchase) (orange_price : ℚ) : ℚ :=
  store.apple_cost * purchase.apples +
  orange_price * purchase.oranges +
  store.banana_cost * purchase.bananas -
  store.discount_per_five * ((purchase.apples + purchase.oranges + purchase.bananas) / 5)

/-- The theorem stating the price of oranges based on Mary's purchase --/
theorem orange_price_theorem (store : FruitStore) (purchase : Purchase) :
  store.apple_cost = 1 →
  store.banana_cost = 3 →
  store.discount_per_five = 1 →
  purchase.apples = 5 →
  purchase.oranges = 3 →
  purchase.bananas = 2 →
  totalCost store purchase (8/3) = 15 :=
by sorry

end NUMINAMATH_CALUDE_orange_price_theorem_l4137_413759


namespace NUMINAMATH_CALUDE_pythagorean_proof_depends_on_parallel_postulate_l4137_413725

-- Define Euclidean geometry
class EuclideanGeometry where
  -- Assume the existence of parallel postulate
  parallel_postulate : Prop

-- Define the concept of a direct proof of the Pythagorean theorem
class PythagoreanProof (E : EuclideanGeometry) where
  -- The proof uses similarity of triangles
  uses_triangle_similarity : Prop
  -- The proof uses equivalency of areas
  uses_area_equivalence : Prop

-- Theorem statement
theorem pythagorean_proof_depends_on_parallel_postulate 
  (E : EuclideanGeometry) 
  (P : PythagoreanProof E) : 
  E.parallel_postulate → 
  (P.uses_triangle_similarity ∨ P.uses_area_equivalence) → 
  -- The proof depends on the parallel postulate
  Prop :=
sorry

end NUMINAMATH_CALUDE_pythagorean_proof_depends_on_parallel_postulate_l4137_413725


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l4137_413726

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : Real.log 2 * x + Real.log 4 * y = Real.log 2) :
  (1 / x + 1 / y) ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l4137_413726


namespace NUMINAMATH_CALUDE_basketball_card_price_l4137_413702

/-- The price of each pack of basketball cards given Nina's shopping details -/
theorem basketball_card_price (toy_price shirt_price total_spent : ℚ)
  (num_toys num_shirts num_card_packs : ℕ)
  (h1 : toy_price = 10)
  (h2 : shirt_price = 6)
  (h3 : num_toys = 3)
  (h4 : num_shirts = 5)
  (h5 : num_card_packs = 2)
  (h6 : total_spent = 70)
  (h7 : total_spent = toy_price * num_toys + shirt_price * num_shirts + num_card_packs * card_price) :
  card_price = 5 :=
by
  sorry

#check basketball_card_price

end NUMINAMATH_CALUDE_basketball_card_price_l4137_413702


namespace NUMINAMATH_CALUDE_divisible_by_three_l4137_413745

theorem divisible_by_three (n : ℕ) : 
  (3 ∣ n * 2^n + 1) ↔ (n % 3 = 1 ∨ n % 3 = 2) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_three_l4137_413745


namespace NUMINAMATH_CALUDE_factor_probability_l4137_413791

/-- The number of consecutive natural numbers in the set -/
def n : ℕ := 120

/-- The factorial we're considering -/
def f : ℕ := 5

/-- The number of factors of f! -/
def num_factors : ℕ := 16

/-- The probability of selecting a factor of f! from the set of n consecutive natural numbers -/
def probability : ℚ := num_factors / n

theorem factor_probability : probability = 2 / 15 := by
  sorry

end NUMINAMATH_CALUDE_factor_probability_l4137_413791


namespace NUMINAMATH_CALUDE_sequence_formulas_correct_l4137_413756

def sequence1 (n : ℕ) : ℚ := 1 / (n * (n + 1))

def sequence2 (n : ℕ) : ℕ := 2^(n - 1)

def sequence3 (n : ℕ) : ℚ := 4 / (3 * n + 2)

theorem sequence_formulas_correct :
  (∀ n : ℕ, n > 0 → sequence1 n = 1 / (n * (n + 1))) ∧
  (∀ n : ℕ, n > 0 → sequence2 n = 2^(n - 1)) ∧
  (∀ n : ℕ, n > 0 → sequence3 n = 4 / (3 * n + 2)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_formulas_correct_l4137_413756


namespace NUMINAMATH_CALUDE_printing_task_theorem_l4137_413728

/-- Represents the printing task -/
structure PrintingTask where
  totalPages : ℕ
  printerATime : ℕ
  printerBExtraRate : ℕ

/-- Calculates the time taken for both printers to complete the task together -/
def timeTakenTogether (task : PrintingTask) : ℚ :=
  (task.totalPages : ℚ) * (task.printerATime : ℚ) / (task.totalPages + task.printerATime * task.printerBExtraRate : ℚ)

/-- Theorem stating that for the given conditions, the time taken is (35 * 60) / 430 minutes -/
theorem printing_task_theorem (task : PrintingTask) 
  (h1 : task.totalPages = 35)
  (h2 : task.printerATime = 60)
  (h3 : task.printerBExtraRate = 6) : 
  timeTakenTogether task = 35 * 60 / 430 := by
  sorry

#eval timeTakenTogether { totalPages := 35, printerATime := 60, printerBExtraRate := 6 }

end NUMINAMATH_CALUDE_printing_task_theorem_l4137_413728


namespace NUMINAMATH_CALUDE_hyperbola_axis_ratio_l4137_413746

/-- For a hyperbola with equation x^2 - my^2 = 1, if the length of the imaginary axis
    is three times the length of the real axis, then m = 1/9 -/
theorem hyperbola_axis_ratio (m : ℝ) :
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
    (∀ (x y : ℝ), x^2 - m*y^2 = 1 ↔ (x/a)^2 - (y/b)^2 = 1) ∧
    b = 3*a) →
  m = 1/9 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_axis_ratio_l4137_413746


namespace NUMINAMATH_CALUDE_newspaper_price_calculation_l4137_413755

/-- The price of each Wednesday, Thursday, and Friday edition of the newspaper -/
def weekday_price : ℚ := 1/2

theorem newspaper_price_calculation :
  let weeks : ℕ := 8
  let weekday_editions_per_week : ℕ := 3
  let sunday_price : ℚ := 2
  let total_spent : ℚ := 28
  weekday_price = (total_spent - (sunday_price * weeks)) / (weekday_editions_per_week * weeks) :=
by
  sorry

#eval weekday_price

end NUMINAMATH_CALUDE_newspaper_price_calculation_l4137_413755


namespace NUMINAMATH_CALUDE_sum_of_abs_sum_and_diff_lt_two_l4137_413749

theorem sum_of_abs_sum_and_diff_lt_two (a b : ℝ) : 
  (|a| < 1) → (|b| < 1) → (|a + b| + |a - b| < 2) := by
sorry

end NUMINAMATH_CALUDE_sum_of_abs_sum_and_diff_lt_two_l4137_413749


namespace NUMINAMATH_CALUDE_function_local_extrema_l4137_413742

/-- The function f(x) = (x^2 + ax + 2)e^x has both a local maximum and a local minimum
    if and only if a > 2 or a < -2 -/
theorem function_local_extrema (a : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    IsLocalMax (fun x => (x^2 + a*x + 2) * Real.exp x) x₁ ∧
    IsLocalMin (fun x => (x^2 + a*x + 2) * Real.exp x) x₂) ↔
  (a > 2 ∨ a < -2) :=
sorry

end NUMINAMATH_CALUDE_function_local_extrema_l4137_413742


namespace NUMINAMATH_CALUDE_complex_equation_solution_l4137_413735

theorem complex_equation_solution (z : ℂ) :
  (z - 2) * (1 + Complex.I) = 1 - Complex.I → z = 2 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l4137_413735


namespace NUMINAMATH_CALUDE_factor_into_sqrt_l4137_413743

theorem factor_into_sqrt (a b : ℝ) (h : a < b) :
  (a - b) * Real.sqrt (-1 / (a - b)) = -Real.sqrt (b - a) := by
  sorry

end NUMINAMATH_CALUDE_factor_into_sqrt_l4137_413743


namespace NUMINAMATH_CALUDE_log_base_1024_integer_count_l4137_413773

theorem log_base_1024_integer_count :
  ∃! (S : Finset ℕ+), 
    (∀ b ∈ S, ∃ n : ℕ+, (b : ℝ) ^ (n : ℝ) = 1024) ∧ 
    (∀ b : ℕ+, (∃ n : ℕ+, (b : ℝ) ^ (n : ℝ) = 1024) → b ∈ S) ∧
    S.card = 4 :=
by sorry

end NUMINAMATH_CALUDE_log_base_1024_integer_count_l4137_413773


namespace NUMINAMATH_CALUDE_inscribed_square_side_length_l4137_413704

/-- A right triangle with sides 5, 12, and 13 -/
structure RightTriangle where
  pq : ℝ
  qr : ℝ
  pr : ℝ
  is_right : pq^2 + qr^2 = pr^2
  pq_eq : pq = 5
  qr_eq : qr = 12
  pr_eq : pr = 13

/-- A square inscribed in the right triangle -/
structure InscribedSquare (t : RightTriangle) where
  side_length : ℝ
  on_hypotenuse : side_length ≤ t.pr
  on_legs : side_length ≤ t.pq ∧ side_length ≤ t.qr

/-- The side length of the inscribed square is 156/25 -/
theorem inscribed_square_side_length (t : RightTriangle) (s : InscribedSquare t) :
  s.side_length = 156 / 25 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_side_length_l4137_413704


namespace NUMINAMATH_CALUDE_intersection_when_a_is_one_intersection_equals_B_iff_l4137_413729

-- Define the sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def B (a : ℝ) : Set ℝ := {x | 2*a < x ∧ x < a+3}

-- Statement I
theorem intersection_when_a_is_one :
  (Set.univ \ A) ∩ (B 1) = {x | 3 < x ∧ x < 4} := by sorry

-- Statement II
theorem intersection_equals_B_iff (a : ℝ) :
  (Set.univ \ A) ∩ (B a) = B a ↔ a ≤ -2 ∨ a ≥ 3/2 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_one_intersection_equals_B_iff_l4137_413729


namespace NUMINAMATH_CALUDE_white_squares_in_row_l4137_413717

/-- Represents a modified stair-step figure where each row begins and ends with a black square
    and has alternating white and black squares. -/
structure ModifiedStairStep where
  /-- The number of squares in the nth row is 2n -/
  squares_in_row : ℕ → ℕ
  /-- Each row begins and ends with a black square -/
  begins_ends_black : ∀ n : ℕ, squares_in_row n ≥ 2
  /-- The number of squares in each row is even -/
  even_squares : ∀ n : ℕ, Even (squares_in_row n)

/-- The number of white squares in the nth row of a modified stair-step figure is equal to n -/
theorem white_squares_in_row (figure : ModifiedStairStep) (n : ℕ) :
  (figure.squares_in_row n) / 2 = n := by
  sorry

end NUMINAMATH_CALUDE_white_squares_in_row_l4137_413717


namespace NUMINAMATH_CALUDE_three_dollar_two_l4137_413747

-- Define the custom operation $
def dollar (a b : ℕ) : ℕ := a^2 * (b + 1) + a * b

-- Theorem statement
theorem three_dollar_two : dollar 3 2 = 33 := by
  sorry

end NUMINAMATH_CALUDE_three_dollar_two_l4137_413747


namespace NUMINAMATH_CALUDE_correct_small_glasses_l4137_413777

/-- Calculates the number of small drinking glasses given the following conditions:
  * 50 jelly beans fill a large glass
  * 25 jelly beans fill a small glass
  * There are 5 large glasses
  * A total of 325 jelly beans are used
-/
def number_of_small_glasses (large_glass_beans : ℕ) (small_glass_beans : ℕ) 
  (num_large_glasses : ℕ) (total_beans : ℕ) : ℕ :=
  (total_beans - large_glass_beans * num_large_glasses) / small_glass_beans

theorem correct_small_glasses : 
  number_of_small_glasses 50 25 5 325 = 3 := by
  sorry

end NUMINAMATH_CALUDE_correct_small_glasses_l4137_413777


namespace NUMINAMATH_CALUDE_remainder_9_1995_mod_7_l4137_413736

theorem remainder_9_1995_mod_7 : 9^1995 % 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_9_1995_mod_7_l4137_413736


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l4137_413793

theorem quadratic_inequality_solution_range (k : ℝ) :
  (k > 0) →
  (∃ x : ℝ, x^2 - 8*x + k < 0) ↔ (k < 16) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l4137_413793


namespace NUMINAMATH_CALUDE_infections_exceed_threshold_l4137_413769

/-- The number of people infected after two rounds of infection -/
def infected_after_two_rounds : ℕ := 81

/-- The average number of people infected by one person in each round -/
def average_infections_per_round : ℕ := 8

/-- The threshold number of infections we want to exceed after three rounds -/
def infection_threshold : ℕ := 700

/-- Theorem stating that the number of infected people after three rounds exceeds the threshold -/
theorem infections_exceed_threshold : 
  infected_after_two_rounds * (1 + average_infections_per_round) > infection_threshold := by
  sorry


end NUMINAMATH_CALUDE_infections_exceed_threshold_l4137_413769


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l4137_413744

/-- A geometric sequence with specific properties -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n) ∧
  (a 1 + a 3 = 8) ∧
  (a 5 + a 7 = 4)

/-- The sum of specific terms in the geometric sequence equals 3 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h : GeometricSequence a) :
  a 9 + a 11 + a 13 + a 15 = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l4137_413744


namespace NUMINAMATH_CALUDE_sequence_properties_l4137_413762

/-- Given a sequence {a_n} where S_n is the sum of the first n terms -/
def S (n : ℕ+) : ℚ := 3 * n.val^2 + 4 * n.val

/-- The nth term of the sequence -/
def a (n : ℕ+) : ℚ := S n - S (n - 1)

theorem sequence_properties :
  (∀ n : ℕ+, a n = 6 * n.val + 1) ∧
  (∀ n : ℕ+, n ≥ 2 → a n - a (n - 1) = 6) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l4137_413762


namespace NUMINAMATH_CALUDE_tangent_line_and_monotonicity_l4137_413720

def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*a*x^2 + 3

theorem tangent_line_and_monotonicity (a : ℝ) :
  (a = 1 → ∃ (m b : ℝ), m = 9 ∧ b = 8 ∧ 
    ∀ x y, y = f 1 x → (x = -1 → y = m*x + b)) ∧
  (a = 0 → ∀ x₁ x₂, x₁ < x₂ → f a x₁ < f a x₂) ∧
  (a < 0 → (∀ x₁ x₂, x₁ < x₂ ∧ x₂ < 2*a → f a x₁ > f a x₂) ∧
           (∀ x₁ x₂, 2*a < x₁ ∧ x₁ < x₂ ∧ x₂ < 0 → f a x₁ < f a x₂) ∧
           (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂)) ∧
  (a > 0 → (∀ x₁ x₂, x₁ < x₂ ∧ x₂ < 0 → f a x₁ < f a x₂) ∧
           (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 2*a → f a x₁ > f a x₂) ∧
           (∀ x₁ x₂, 2*a < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_and_monotonicity_l4137_413720


namespace NUMINAMATH_CALUDE_return_speed_l4137_413774

/-- Given two towns and a person's travel speeds, calculate the return speed -/
theorem return_speed (d : ℝ) (v_xy v_total : ℝ) (h1 : v_xy = 54) (h2 : v_total = 43.2) :
  let v_yx := 2 * v_total * v_xy / (2 * v_xy - v_total)
  v_yx = 36 := by sorry

end NUMINAMATH_CALUDE_return_speed_l4137_413774


namespace NUMINAMATH_CALUDE_number_of_ambiguous_dates_l4137_413772

/-- The number of days that cannot be uniquely determined by date notation -/
def ambiguous_dates : ℕ :=
  let total_possible_ambiguous := 12 * 12  -- Days 1-12 for each of the 12 months
  let non_ambiguous := 12  -- Dates where day and month are the same (e.g., 1.1, 2.2, ..., 12.12)
  total_possible_ambiguous - non_ambiguous

/-- Theorem stating that the number of ambiguous dates is 132 -/
theorem number_of_ambiguous_dates : ambiguous_dates = 132 := by
  sorry


end NUMINAMATH_CALUDE_number_of_ambiguous_dates_l4137_413772


namespace NUMINAMATH_CALUDE_infinitely_many_fantastic_triplets_l4137_413781

/-- Definition of a fantastic triplet -/
def is_fantastic_triplet (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  (∃ k : ℚ, b = k * a ∧ c = k * b) ∧
  (∃ d : ℤ, b + 1 - a = d ∧ c - (b + 1) = d)

/-- There exist infinitely many fantastic triplets -/
theorem infinitely_many_fantastic_triplets :
  ∀ i : ℕ, ∃ a b c : ℕ,
    is_fantastic_triplet a b c ∧
    a = 2^(2*i+1) ∧
    b = 2^(2*i+1) + 2^i ∧
    c = 2^(2*i+1) + 2^(i+2) + 2 :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_fantastic_triplets_l4137_413781


namespace NUMINAMATH_CALUDE_p_squared_plus_20_not_prime_l4137_413708

theorem p_squared_plus_20_not_prime (p : ℕ) (h : Prime p) : ¬ Prime (p^2 + 20) := by
  sorry

end NUMINAMATH_CALUDE_p_squared_plus_20_not_prime_l4137_413708


namespace NUMINAMATH_CALUDE_largest_d_for_negative_five_in_range_l4137_413761

-- Define the function g
def g (x d : ℝ) : ℝ := x^2 + 5*x + d

-- State the theorem
theorem largest_d_for_negative_five_in_range :
  (∃ (d : ℝ), ∀ (d' : ℝ), 
    (∃ (x : ℝ), g x d = -5) → 
    (∃ (x : ℝ), g x d' = -5) → 
    d' ≤ d) ∧
  (∃ (x : ℝ), g x (5/4) = -5) :=
sorry

end NUMINAMATH_CALUDE_largest_d_for_negative_five_in_range_l4137_413761


namespace NUMINAMATH_CALUDE_familyReunionHandshakesCount_l4137_413713

/-- Represents the number of handshakes at a family reunion --/
def familyReunionHandshakes : ℕ :=
  let quadrupletSets : ℕ := 12
  let quintupletSets : ℕ := 4
  let quadrupletsPerSet : ℕ := 4
  let quintupletsPerSet : ℕ := 5
  let totalQuadruplets : ℕ := quadrupletSets * quadrupletsPerSet
  let totalQuintuplets : ℕ := quintupletSets * quintupletsPerSet
  let quadrupletHandshakes : ℕ := totalQuadruplets * (totalQuadruplets - quadrupletsPerSet)
  let quintupletHandshakes : ℕ := totalQuintuplets * (totalQuintuplets - quintupletsPerSet)
  let crossHandshakes : ℕ := totalQuadruplets * 7 + totalQuintuplets * 12
  (quadrupletHandshakes + quintupletHandshakes + crossHandshakes) / 2

/-- Theorem stating that the number of handshakes at the family reunion is 1494 --/
theorem familyReunionHandshakesCount : familyReunionHandshakes = 1494 := by
  sorry

end NUMINAMATH_CALUDE_familyReunionHandshakesCount_l4137_413713


namespace NUMINAMATH_CALUDE_angle_D_is_120_l4137_413770

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : ℝ)
  (sum_360 : A + B + C + D = 360)
  (all_positive : A > 0 ∧ B > 0 ∧ C > 0 ∧ D > 0)

-- Define the ratio condition
def ratio_condition (q : Quadrilateral) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ q.A = k ∧ q.B = 2*k ∧ q.C = k ∧ q.D = 2*k

-- Theorem statement
theorem angle_D_is_120 (q : Quadrilateral) (h : ratio_condition q) : q.D = 120 := by
  sorry

end NUMINAMATH_CALUDE_angle_D_is_120_l4137_413770


namespace NUMINAMATH_CALUDE_xy_reciprocal_and_ratio_l4137_413706

theorem xy_reciprocal_and_ratio (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : x * y = 1) (h4 : x / y = 36) : y = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_xy_reciprocal_and_ratio_l4137_413706


namespace NUMINAMATH_CALUDE_empty_solution_set_implies_a_range_l4137_413794

theorem empty_solution_set_implies_a_range :
  (∀ x : ℝ, |x - 1| - |x - 2| ≤ 1) →
  (∀ x : ℝ, |x - 1| - |x - 2| < a^2 + a + 1) →
  a ∈ Set.Iio (-1) ∪ Set.Ioi 0 := by
  sorry

end NUMINAMATH_CALUDE_empty_solution_set_implies_a_range_l4137_413794


namespace NUMINAMATH_CALUDE_inverse_f_84_l4137_413764

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^3 + 3

-- State the theorem
theorem inverse_f_84 : 
  ∃ (y : ℝ), f y = 84 ∧ y = 3 := by sorry

end NUMINAMATH_CALUDE_inverse_f_84_l4137_413764


namespace NUMINAMATH_CALUDE_lcm_of_1400_and_1050_l4137_413716

theorem lcm_of_1400_and_1050 : Nat.lcm 1400 1050 = 4200 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_1400_and_1050_l4137_413716


namespace NUMINAMATH_CALUDE_whipped_cream_cans_needed_l4137_413799

/-- The number of pies Billie bakes per day -/
def pies_per_day : ℕ := 3

/-- The number of days Billie bakes pies -/
def baking_days : ℕ := 11

/-- The number of cans of whipped cream needed to cover one pie -/
def cans_per_pie : ℕ := 2

/-- The number of pies Tiffany eats -/
def pies_eaten : ℕ := 4

/-- The total number of pies Billie bakes -/
def total_pies : ℕ := pies_per_day * baking_days

/-- The number of pies remaining after Tiffany eats -/
def remaining_pies : ℕ := total_pies - pies_eaten

/-- The number of cans of whipped cream needed to cover the remaining pies -/
def cans_needed : ℕ := remaining_pies * cans_per_pie

theorem whipped_cream_cans_needed : cans_needed = 58 := by
  sorry

end NUMINAMATH_CALUDE_whipped_cream_cans_needed_l4137_413799


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l4137_413782

theorem solution_set_quadratic_inequality :
  {x : ℝ | x^2 - 3*x - 4 ≤ 0} = {x : ℝ | -1 ≤ x ∧ x ≤ 4} := by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l4137_413782


namespace NUMINAMATH_CALUDE_quadratic_root_theorem_l4137_413721

-- Define the quadratic polynomial f(x) = ax² + bx + c
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the condition for f(x) to have exactly one root
def has_one_root (a b c : ℝ) : Prop :=
  ∃! x, f a b c x = 0

-- Define g(x) = f(3x + 2) - 2f(2x - 1)
def g (a b c x : ℝ) : ℝ := f a b c (3*x + 2) - 2 * f a b c (2*x - 1)

-- Theorem statement
theorem quadratic_root_theorem (a b c : ℝ) :
  a ≠ 0 →
  has_one_root a b c →
  has_one_root 1 (20 - b) (2 + 4*b - b^2/4) →
  ∃ x, f a b c x = 0 ∧ x = -7 :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_theorem_l4137_413721


namespace NUMINAMATH_CALUDE_rectangular_plot_roots_l4137_413719

theorem rectangular_plot_roots (length width r s : ℝ) : 
  length^2 - 3*length + 2 = 0 →
  width^2 - 3*width + 2 = 0 →
  (1/length)^2 - r*(1/length) + s = 0 →
  (1/width)^2 - r*(1/width) + s = 0 →
  r*s = 0.75 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_roots_l4137_413719


namespace NUMINAMATH_CALUDE_mlb_game_hits_and_misses_l4137_413766

theorem mlb_game_hits_and_misses (hits misses : ℕ) : 
  misses = 3 * hits → 
  misses = 50 → 
  hits + misses = 200 := by
  sorry

end NUMINAMATH_CALUDE_mlb_game_hits_and_misses_l4137_413766


namespace NUMINAMATH_CALUDE_absolute_value_simplification_l4137_413796

theorem absolute_value_simplification : |(-4^2 + (5 - 2))| = 13 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_simplification_l4137_413796


namespace NUMINAMATH_CALUDE_recurring_decimal_to_fraction_l4137_413730

theorem recurring_decimal_to_fraction : 
  (0.3 : ℚ) + (23 : ℚ) / 99 = 527 / 990 := by sorry

end NUMINAMATH_CALUDE_recurring_decimal_to_fraction_l4137_413730


namespace NUMINAMATH_CALUDE_mary_overtime_pay_increase_l4137_413792

/-- Represents Mary's work schedule and pay structure -/
structure WorkSchedule where
  maxHours : Nat
  regularHours : Nat
  regularRate : ℚ
  totalEarnings : ℚ

/-- Calculates the percentage increase in overtime pay given a work schedule -/
def overtimePayIncrease (schedule : WorkSchedule) : ℚ :=
  let regularEarnings := schedule.regularHours * schedule.regularRate
  let overtimeEarnings := schedule.totalEarnings - regularEarnings
  let overtimeHours := schedule.maxHours - schedule.regularHours
  let overtimeRate := overtimeEarnings / overtimeHours
  ((overtimeRate - schedule.regularRate) / schedule.regularRate) * 100

/-- Theorem stating that Mary's overtime pay increase is 25% -/
theorem mary_overtime_pay_increase :
  let mary_schedule : WorkSchedule := {
    maxHours := 45,
    regularHours := 20,
    regularRate := 8,
    totalEarnings := 410
  }
  overtimePayIncrease mary_schedule = 25 := by
  sorry


end NUMINAMATH_CALUDE_mary_overtime_pay_increase_l4137_413792


namespace NUMINAMATH_CALUDE_units_digit_of_2_power_10_l4137_413763

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The function to calculate 2 to the power of n -/
def powerOfTwo (n : ℕ) : ℕ := 2^n

theorem units_digit_of_2_power_10 : unitsDigit (powerOfTwo 10) = 4 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_2_power_10_l4137_413763


namespace NUMINAMATH_CALUDE_minimum_trips_for_5000_rubles_l4137_413707

theorem minimum_trips_for_5000_rubles :
  ∀ (x y : ℕ),
  31 * x + 32 * y = 5000 →
  x + y ≥ 157 :=
by
  sorry

end NUMINAMATH_CALUDE_minimum_trips_for_5000_rubles_l4137_413707


namespace NUMINAMATH_CALUDE_least_months_to_triple_l4137_413709

def initial_amount : ℝ := 1500
def monthly_interest_rate : ℝ := 0.06

def compound_factor (t : ℕ) : ℝ := (1 + monthly_interest_rate) ^ t

theorem least_months_to_triple :
  ∀ n : ℕ, n < 20 → compound_factor n ≤ 3 ∧
  compound_factor 20 > 3 :=
by sorry

end NUMINAMATH_CALUDE_least_months_to_triple_l4137_413709


namespace NUMINAMATH_CALUDE_eds_pets_l4137_413748

/-- The number of pets Ed has -/
def total_pets (dogs cats : ℕ) : ℕ :=
  let fish := 2 * (dogs + cats)
  let birds := dogs * cats
  dogs + cats + fish + birds

/-- Theorem stating the total number of Ed's pets -/
theorem eds_pets : total_pets 2 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_eds_pets_l4137_413748


namespace NUMINAMATH_CALUDE_triangle_determinant_zero_l4137_413754

theorem triangle_determinant_zero (A B C : Real) 
  (h_triangle : A + B + C = Real.pi) : 
  let matrix : Matrix (Fin 3) (Fin 3) Real := 
    ![![Real.cos A ^ 2, Real.tan A, 1],
      ![Real.cos B ^ 2, Real.tan B, 1],
      ![Real.cos C ^ 2, Real.tan C, 1]]
  Matrix.det matrix = 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_determinant_zero_l4137_413754


namespace NUMINAMATH_CALUDE_greatest_x_value_l4137_413757

theorem greatest_x_value (x : ℤ) (h : (2.134 : ℝ) * (10 : ℝ) ^ (x : ℝ) < 210000) :
  x ≤ 4 ∧ ∃ y : ℤ, y > 4 → (2.134 : ℝ) * (10 : ℝ) ^ (y : ℝ) ≥ 210000 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_value_l4137_413757


namespace NUMINAMATH_CALUDE_equipment_value_after_three_years_l4137_413741

/-- The value of equipment after n years, given an initial value and annual depreciation rate. -/
def equipment_value (initial_value : ℝ) (depreciation_rate : ℝ) (years : ℕ) : ℝ :=
  initial_value * (1 - depreciation_rate) ^ years

/-- Theorem: The value of equipment initially worth 10,000 yuan, depreciating by 50% annually, will be 1,250 yuan after 3 years. -/
theorem equipment_value_after_three_years :
  equipment_value 10000 0.5 3 = 1250 := by
  sorry

end NUMINAMATH_CALUDE_equipment_value_after_three_years_l4137_413741


namespace NUMINAMATH_CALUDE_on_time_passengers_l4137_413778

theorem on_time_passengers (total : ℕ) (late : ℕ) (on_time : ℕ) : 
  total = 14720 → late = 213 → on_time = total - late → on_time = 14507 := by
  sorry

end NUMINAMATH_CALUDE_on_time_passengers_l4137_413778


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l4137_413740

theorem absolute_value_inequality (x : ℝ) : 
  |2*x - 1| - |x + 1| < 1 ↔ -1/3 < x ∧ x < 3 :=
sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l4137_413740


namespace NUMINAMATH_CALUDE_integer_pair_solution_l4137_413734

theorem integer_pair_solution (m n : ℤ) :
  (m - n)^2 = 4 * m * n / (m + n - 1) →
  ∃ k : ℕ, k ≠ 1 ∧
    ((m = (k^2 + k) / 2 ∧ n = (k^2 - k) / 2) ∨
     (m = (k^2 - k) / 2 ∧ n = (k^2 + k) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_integer_pair_solution_l4137_413734


namespace NUMINAMATH_CALUDE_bernardo_wins_game_l4137_413724

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem bernardo_wins_game :
  ∃ N : ℕ,
    N = 32 ∧
    16 * N + 1400 < 2000 ∧
    16 * N + 1500 ≥ 2000 ∧
    sum_of_digits N = 5 ∧
    ∀ m : ℕ, m < N →
      ¬(16 * m + 1400 < 2000 ∧
        16 * m + 1500 ≥ 2000 ∧
        sum_of_digits m = 5) :=
by sorry

end NUMINAMATH_CALUDE_bernardo_wins_game_l4137_413724


namespace NUMINAMATH_CALUDE_sequence_sum_l4137_413737

theorem sequence_sum (a : ℕ → ℝ) (a_pos : ∀ n, a n > 0) 
  (h1 : a 1 = 2) (h2 : a 2 = 3) (h3 : a 3 = 4) (h5 : a 5 = 6) :
  ∃ (a_val t : ℝ), a_val > 0 ∧ t > 0 ∧ a_val = a 5 ∧ t = a_val^2 - 1 ∧ a_val + t = 41 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l4137_413737


namespace NUMINAMATH_CALUDE_perpendicular_line_value_l4137_413715

theorem perpendicular_line_value (θ : Real) (h : Real.tan θ = -3) :
  2 / (3 * Real.sin θ ^ 2 - Real.cos θ ^ 2) = 10/13 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_value_l4137_413715


namespace NUMINAMATH_CALUDE_initial_bird_families_l4137_413784

theorem initial_bird_families (flew_away left_now : ℕ) 
  (h1 : flew_away = 27) 
  (h2 : left_now = 14) : 
  flew_away + left_now = 41 := by
  sorry

end NUMINAMATH_CALUDE_initial_bird_families_l4137_413784


namespace NUMINAMATH_CALUDE_empty_set_proof_l4137_413751

theorem empty_set_proof : {x : ℝ | x^2 + x + 1 = 0} = ∅ := by
  sorry

end NUMINAMATH_CALUDE_empty_set_proof_l4137_413751


namespace NUMINAMATH_CALUDE_f_min_value_f_at_3_l4137_413710

-- Define the function f
def f (x : ℝ) : ℝ := 7 * x^2 - 28 * x + 2003

-- Theorem for the minimum value of f
theorem f_min_value : ∃ (min : ℝ), min = 1975 ∧ ∀ (x : ℝ), f x ≥ min :=
sorry

-- Theorem for the value of f(3)
theorem f_at_3 : f 3 = 1982 :=
sorry

end NUMINAMATH_CALUDE_f_min_value_f_at_3_l4137_413710


namespace NUMINAMATH_CALUDE_exists_equilateral_DEF_l4137_413701

open Real

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- Checks if a triangle is acute-angled -/
def isAcuteAngled (t : Triangle) : Prop := sorry

/-- Checks if a point is inside a circle -/
def isInside (p : Point) (c : Circle) : Prop := sorry

/-- Gets the circumcircle of a triangle -/
def circumcircle (t : Triangle) : Circle := sorry

/-- Gets the intersection points of a ray from a point through another point with a circle -/
def rayIntersection (start : Point) (through : Point) (c : Circle) : Point := sorry

/-- Checks if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop := sorry

/-- Main theorem -/
theorem exists_equilateral_DEF (ABC : Triangle) (c : Circle) :
  isAcuteAngled ABC →
  c = circumcircle ABC →
  ∃ P : Point,
    isInside P c ∧
    let D := rayIntersection A P c
    let E := rayIntersection B P c
    let F := rayIntersection C P c
    isEquilateral (Triangle.mk D E F) :=
by sorry

end NUMINAMATH_CALUDE_exists_equilateral_DEF_l4137_413701


namespace NUMINAMATH_CALUDE_last_digit_of_large_prime_l4137_413780

theorem last_digit_of_large_prime (n : ℕ) (h : n = 859433) :
  (2^n - 1) % 10 = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_large_prime_l4137_413780


namespace NUMINAMATH_CALUDE_rectangle_area_transformation_l4137_413727

theorem rectangle_area_transformation (A : ℝ) : 
  (12 + 3) * (12 - A) = 120 → A = 4 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_transformation_l4137_413727


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l4137_413785

theorem quadratic_equation_solution (p q : ℤ) :
  (∃ x : ℝ, 36 * x^2 - 4 * (p^2 + 11) * x + 135 * (p + q) + 576 = 0) →
  p + q = 20 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l4137_413785


namespace NUMINAMATH_CALUDE_jones_trip_time_comparison_l4137_413795

/-- Proves that the time taken for the third trip is three times the time taken for the first trip
    given the conditions of Jones' three trips. -/
theorem jones_trip_time_comparison 
  (v : ℝ) -- Original speed
  (h1 : v > 0) -- Assumption that speed is positive
  (d1 : ℝ) (h2 : d1 = 40) -- Distance of first trip
  (d2 : ℝ) (h3 : d2 = 200) -- Distance of second trip
  (d3 : ℝ) (h4 : d3 = 480) -- Distance of third trip
  (v2 : ℝ) (h5 : v2 = 2 * v) -- Speed of second trip
  (v3 : ℝ) (h6 : v3 = 2 * v2) -- Speed of third trip
  : (d3 / v3) = 3 * (d1 / v) := by
  sorry

end NUMINAMATH_CALUDE_jones_trip_time_comparison_l4137_413795


namespace NUMINAMATH_CALUDE_correct_num_tripodasauruses_l4137_413779

/-- Represents the number of tripodasauruses in a flock -/
def num_tripodasauruses : ℕ := 5

/-- Represents the number of legs a tripodasaurus has -/
def legs_per_tripodasaurus : ℕ := 3

/-- Represents the number of heads a tripodasaurus has -/
def heads_per_tripodasaurus : ℕ := 1

/-- Represents the total number of heads and legs in the flock -/
def total_heads_and_legs : ℕ := 20

/-- Theorem stating that the number of tripodasauruses in the flock is correct -/
theorem correct_num_tripodasauruses : 
  num_tripodasauruses * (legs_per_tripodasaurus + heads_per_tripodasaurus) = total_heads_and_legs :=
by sorry

end NUMINAMATH_CALUDE_correct_num_tripodasauruses_l4137_413779


namespace NUMINAMATH_CALUDE_combination_5_choose_3_l4137_413753

/-- The number of combinations of n things taken k at a time -/
def combination (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Proof that C(5,3) equals 10 -/
theorem combination_5_choose_3 : combination 5 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_combination_5_choose_3_l4137_413753


namespace NUMINAMATH_CALUDE_scarf_cost_l4137_413760

theorem scarf_cost (initial_amount : ℕ) (toy_car_cost : ℕ) (num_toy_cars : ℕ) (beanie_cost : ℕ) (remaining_amount : ℕ) : 
  initial_amount = 53 →
  toy_car_cost = 11 →
  num_toy_cars = 2 →
  beanie_cost = 14 →
  remaining_amount = 7 →
  initial_amount - (num_toy_cars * toy_car_cost + beanie_cost + remaining_amount) = 10 := by
sorry

end NUMINAMATH_CALUDE_scarf_cost_l4137_413760


namespace NUMINAMATH_CALUDE_oil_weight_in_salad_dressing_salad_dressing_oil_weight_l4137_413790

/-- Calculates the weight of oil per ml in a salad dressing mixture --/
theorem oil_weight_in_salad_dressing 
  (bowl_capacity : ℝ) 
  (oil_proportion : ℝ) 
  (vinegar_proportion : ℝ) 
  (vinegar_weight : ℝ) 
  (total_weight : ℝ) : ℝ :=
  let oil_volume := bowl_capacity * oil_proportion
  let vinegar_volume := bowl_capacity * vinegar_proportion
  let vinegar_total_weight := vinegar_volume * vinegar_weight
  let oil_total_weight := total_weight - vinegar_total_weight
  oil_total_weight / oil_volume

/-- Proves that the weight of oil in the given salad dressing mixture is 5 g/ml --/
theorem salad_dressing_oil_weight :
  oil_weight_in_salad_dressing 150 (2/3) (1/3) 4 700 = 5 := by
  sorry

end NUMINAMATH_CALUDE_oil_weight_in_salad_dressing_salad_dressing_oil_weight_l4137_413790


namespace NUMINAMATH_CALUDE_weekend_reading_l4137_413723

/-- The number of pages Bekah needs to read for history class -/
def total_pages : ℕ := 408

/-- The number of days left to finish reading -/
def days_left : ℕ := 5

/-- The number of pages Bekah needs to read each day for the remaining days -/
def pages_per_day : ℕ := 59

/-- The number of pages Bekah read over the weekend -/
def pages_read_weekend : ℕ := total_pages - (days_left * pages_per_day)

theorem weekend_reading :
  pages_read_weekend = 113 := by sorry

end NUMINAMATH_CALUDE_weekend_reading_l4137_413723


namespace NUMINAMATH_CALUDE_b_55_mod_55_eq_zero_l4137_413752

/-- The integer obtained by writing all the integers from 1 to n from left to right -/
def b (n : ℕ) : ℕ :=
  sorry

/-- Theorem: The remainder when b₅₅ is divided by 55 is 0 -/
theorem b_55_mod_55_eq_zero : b 55 % 55 = 0 := by
  sorry

end NUMINAMATH_CALUDE_b_55_mod_55_eq_zero_l4137_413752


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l4137_413789

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3 * b = 1) :
  (1 / a + 1 / b) ≥ 4 + 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l4137_413789


namespace NUMINAMATH_CALUDE_exponent_fraction_equality_l4137_413797

theorem exponent_fraction_equality : (3^2015 + 3^2013) / (3^2015 - 3^2013) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_exponent_fraction_equality_l4137_413797


namespace NUMINAMATH_CALUDE_min_weighings_for_ten_coins_l4137_413718

/-- Represents a weighing on a balance scale -/
inductive Weighing
  | Equal : Weighing
  | LeftLighter : Weighing
  | RightLighter : Weighing

/-- Represents the state of knowledge about the coins -/
structure CoinState where
  total : Nat
  genuine : Nat
  counterfeit : Nat

/-- A function that performs a weighing and updates the coin state -/
def performWeighing (state : CoinState) (w : Weighing) : CoinState :=
  sorry

/-- The minimum number of weighings required to find the counterfeit coin -/
def minWeighings (state : CoinState) : Nat :=
  sorry

/-- Theorem stating that the minimum number of weighings for 10 coins with 1 counterfeit is 3 -/
theorem min_weighings_for_ten_coins :
  let initialState : CoinState := ⟨10, 9, 1⟩
  minWeighings initialState = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_weighings_for_ten_coins_l4137_413718


namespace NUMINAMATH_CALUDE_geometric_sequence_a12_l4137_413714

/-- A geometric sequence (aₙ) -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_a12 (a : ℕ → ℝ) :
  geometric_sequence a → a 4 = 4 → a 8 = 8 → a 12 = 16 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a12_l4137_413714


namespace NUMINAMATH_CALUDE_peanut_butter_servings_l4137_413775

/-- The amount of peanut butter in the jar in tablespoons -/
def jar_amount : ℚ := 45 + 2/3

/-- The size of one serving of peanut butter in tablespoons -/
def serving_size : ℚ := 1 + 1/3

/-- The number of servings in the jar -/
def servings : ℚ := jar_amount / serving_size

theorem peanut_butter_servings : servings = 34 + 1/4 := by
  sorry

end NUMINAMATH_CALUDE_peanut_butter_servings_l4137_413775


namespace NUMINAMATH_CALUDE_range_of_a_l4137_413739

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x - a| + |x - 1| ≤ 3) → -2 ≤ a ∧ a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l4137_413739


namespace NUMINAMATH_CALUDE_quadratic_sum_l4137_413705

def quadratic_function (a b c : ℤ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_sum (a b c : ℤ) : 
  (quadratic_function a b c 0 = 2) → 
  (∀ x, quadratic_function a b c x ≥ quadratic_function a b c 1) →
  (quadratic_function a b c 1 = -1) →
  a - b + c = 11 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l4137_413705
