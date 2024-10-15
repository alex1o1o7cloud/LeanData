import Mathlib

namespace NUMINAMATH_CALUDE_return_probability_after_2012_moves_chessboard_return_probability_l1768_176889

/-- Represents the size of the chessboard -/
def boardSize : ℕ := 8

/-- Represents the total number of moves -/
def totalMoves : ℕ := 2012

/-- Represents the probability of returning to the original position after a given number of moves -/
noncomputable def returnProbability (n : ℕ) : ℚ :=
  ((1 + 2^(n / 2 - 1)) / 2^(n / 2 + 1))^2

/-- Theorem stating the probability of returning to the original position after 2012 moves -/
theorem return_probability_after_2012_moves :
  returnProbability totalMoves = ((1 + 2^1005) / 2^1007)^2 := by
  sorry

/-- Theorem stating that the calculated probability is correct for the given chessboard and moves -/
theorem chessboard_return_probability :
  boardSize = 8 →
  totalMoves = 2012 →
  returnProbability totalMoves = ((1 + 2^1005) / 2^1007)^2 := by
  sorry

end NUMINAMATH_CALUDE_return_probability_after_2012_moves_chessboard_return_probability_l1768_176889


namespace NUMINAMATH_CALUDE_engagement_treats_value_l1768_176820

def hotel_nights : ℕ := 2
def hotel_cost_per_night : ℕ := 4000
def car_value : ℕ := 30000

def total_treat_value : ℕ :=
  hotel_nights * hotel_cost_per_night + car_value + 4 * car_value

theorem engagement_treats_value :
  total_treat_value = 158000 := by
  sorry

end NUMINAMATH_CALUDE_engagement_treats_value_l1768_176820


namespace NUMINAMATH_CALUDE_no_zeros_in_larger_interval_l1768_176838

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property of having a unique zero in the given intervals
def has_unique_zero_in_intervals (f : ℝ → ℝ) : Prop :=
  ∃! x, f x = 0 ∧ 
    0 < x ∧ x < 16 ∧
    0 < x ∧ x < 8 ∧
    0 < x ∧ x < 4 ∧
    0 < x ∧ x < 2

-- State the theorem
theorem no_zeros_in_larger_interval 
  (h : has_unique_zero_in_intervals f) : 
  ∀ x ∈ Set.Icc 2 16, f x ≠ 0 := by
  sorry


end NUMINAMATH_CALUDE_no_zeros_in_larger_interval_l1768_176838


namespace NUMINAMATH_CALUDE_base6_addition_l1768_176888

-- Define a function to convert from base 6 to base 10
def base6ToBase10 (n : ℕ) : ℕ := sorry

-- Define a function to convert from base 10 to base 6
def base10ToBase6 (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem base6_addition : 
  base10ToBase6 (base6ToBase10 25 + base6ToBase10 35) = 104 := by sorry

end NUMINAMATH_CALUDE_base6_addition_l1768_176888


namespace NUMINAMATH_CALUDE_inverse_fraction_minus_abs_diff_l1768_176805

theorem inverse_fraction_minus_abs_diff : (1/3)⁻¹ - |Real.sqrt 3 - 3| = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_fraction_minus_abs_diff_l1768_176805


namespace NUMINAMATH_CALUDE_calculate_initial_weight_l1768_176886

/-- Calculates the initial weight of a person on a constant weight loss diet -/
theorem calculate_initial_weight 
  (current_weight : ℝ) 
  (future_weight : ℝ) 
  (months_to_future : ℝ) 
  (months_on_diet : ℝ) 
  (h1 : current_weight > future_weight) 
  (h2 : months_to_future > 0) 
  (h3 : months_on_diet > 0) :
  ∃ (initial_weight : ℝ),
    initial_weight = current_weight + (current_weight - future_weight) / months_to_future * months_on_diet :=
by
  sorry

#check calculate_initial_weight

end NUMINAMATH_CALUDE_calculate_initial_weight_l1768_176886


namespace NUMINAMATH_CALUDE_complex_multiplication_l1768_176851

theorem complex_multiplication (i : ℂ) : i * i = -1 → i * (3 + 4*i) = -4 + 3*i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l1768_176851


namespace NUMINAMATH_CALUDE_quadratic_root_implies_coefficients_l1768_176871

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the quadratic equation
def is_root (x a b : ℂ) : Prop := x^2 + a*x + b = 0

-- State the theorem
theorem quadratic_root_implies_coefficients :
  ∀ (a b : ℝ), is_root (1 - i) a b → a = -2 ∧ b = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_coefficients_l1768_176871


namespace NUMINAMATH_CALUDE_cubic_polynomial_theorem_l1768_176830

/-- Given a, b, c are roots of x³ + 4x² + 6x + 8 = 0 -/
def cubic_roots (a b c : ℝ) : Prop :=
  a^3 + 4*a^2 + 6*a + 8 = 0 ∧
  b^3 + 4*b^2 + 6*b + 8 = 0 ∧
  c^3 + 4*c^2 + 6*c + 8 = 0

/-- Q is a cubic polynomial satisfying the given conditions -/
def Q_conditions (Q : ℝ → ℝ) (a b c : ℝ) : Prop :=
  (∃ p q r s : ℝ, ∀ x, Q x = p*x^3 + q*x^2 + r*x + s) ∧
  Q a = b + c ∧
  Q b = a + c ∧
  Q c = a + b ∧
  Q (a + b + c) = -20

theorem cubic_polynomial_theorem (a b c : ℝ) (Q : ℝ → ℝ) 
  (h1 : cubic_roots a b c) (h2 : Q_conditions Q a b c) :
  ∀ x, Q x = 5/4*x^3 + 4*x^2 + 17/4*x + 2 :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_theorem_l1768_176830


namespace NUMINAMATH_CALUDE_complex_equation_sum_l1768_176810

theorem complex_equation_sum (a b : ℝ) : 
  (Complex.mk a b = (2 * Complex.I) / (1 + Complex.I)) → a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l1768_176810


namespace NUMINAMATH_CALUDE_constant_speed_walking_time_l1768_176862

/-- Represents the time taken to walk a certain distance at a constant speed -/
structure WalkingTime where
  distance : ℝ
  time : ℝ

/-- Given a constant walking speed, prove that if it takes 30 minutes to walk 4 kilometers,
    then it will take 15 minutes to walk 2 kilometers -/
theorem constant_speed_walking_time 
  (speed : ℝ) 
  (library : WalkingTime) 
  (school : WalkingTime) 
  (h1 : speed > 0)
  (h2 : library.distance = 4)
  (h3 : library.time = 30)
  (h4 : school.distance = 2)
  (h5 : library.distance / library.time = speed)
  (h6 : school.distance / school.time = speed) :
  school.time = 15 := by
  sorry

end NUMINAMATH_CALUDE_constant_speed_walking_time_l1768_176862


namespace NUMINAMATH_CALUDE_train_length_train_length_is_120_l1768_176860

/-- The length of a train given specific conditions -/
theorem train_length (jogger_speed : ℝ) (train_speed : ℝ) (initial_distance : ℝ) (passing_time : ℝ) : ℝ :=
  let jogger_speed_ms := jogger_speed * 1000 / 3600
  let train_speed_ms := train_speed * 1000 / 3600
  let relative_speed := train_speed_ms - jogger_speed_ms
  relative_speed * passing_time - initial_distance

/-- Proof that the train length is 120 meters under given conditions -/
theorem train_length_is_120 :
  train_length 9 45 120 24 = 120 := by
  sorry

end NUMINAMATH_CALUDE_train_length_train_length_is_120_l1768_176860


namespace NUMINAMATH_CALUDE_smallest_integer_above_root_sum_sixth_power_l1768_176829

theorem smallest_integer_above_root_sum_sixth_power :
  ∃ n : ℕ, n = 3323 ∧ (∀ m : ℕ, m < n → (m : ℝ) ≤ (Real.sqrt 5 + Real.sqrt 3)^6) ∧
  n > (Real.sqrt 5 + Real.sqrt 3)^6 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_above_root_sum_sixth_power_l1768_176829


namespace NUMINAMATH_CALUDE_custard_pie_problem_l1768_176824

theorem custard_pie_problem (price_per_slice : ℚ) (slices_per_pie : ℕ) (total_revenue : ℚ) :
  price_per_slice = 3 →
  slices_per_pie = 10 →
  total_revenue = 180 →
  (total_revenue / (price_per_slice * slices_per_pie : ℚ)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_custard_pie_problem_l1768_176824


namespace NUMINAMATH_CALUDE_smallest_a_for_parabola_l1768_176866

/-- The smallest possible value of a for a parabola with given conditions -/
theorem smallest_a_for_parabola (a b c : ℝ) : 
  a > 0 → 
  (∃ n : ℤ, a + b + c = n) →
  (∀ x y : ℝ, y = a * x^2 + b * x + c ↔ y + 2 = a * (x - 3/4)^2) →
  (∀ a' : ℝ, a' > 0 ∧ 
    (∃ b' c' : ℝ, (∃ n : ℤ, a' + b' + c' = n) ∧
    (∀ x y : ℝ, y = a' * x^2 + b' * x + c' ↔ y + 2 = a' * (x - 3/4)^2)) →
    a ≤ a') →
  a = 16 :=
by sorry

end NUMINAMATH_CALUDE_smallest_a_for_parabola_l1768_176866


namespace NUMINAMATH_CALUDE_number_problem_l1768_176870

theorem number_problem (x : ℝ) : (0.6 * x = 0.3 * 10 + 27) → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1768_176870


namespace NUMINAMATH_CALUDE_existence_of_monochromatic_triangle_l1768_176842

/-- A point in the six-pointed star --/
structure Point :=
  (index : Fin 13)

/-- The color of a point --/
inductive Color
| Red
| Green

/-- A coloring of the points in the star --/
def Coloring := Point → Color

/-- Predicate to check if three points form an equilateral triangle --/
def IsEquilateralTriangle (p q r : Point) : Prop := sorry

/-- The main theorem --/
theorem existence_of_monochromatic_triangle (coloring : Coloring) :
  ∃ (p q r : Point), coloring p = coloring q ∧ coloring q = coloring r ∧ IsEquilateralTriangle p q r :=
sorry

end NUMINAMATH_CALUDE_existence_of_monochromatic_triangle_l1768_176842


namespace NUMINAMATH_CALUDE_newer_train_distance_calculation_l1768_176821

/-- The distance traveled by the older train in miles -/
def older_train_distance : ℝ := 300

/-- The percentage increase in distance for the newer train -/
def percentage_increase : ℝ := 0.30

/-- The distance traveled by the newer train in miles -/
def newer_train_distance : ℝ := older_train_distance * (1 + percentage_increase)

theorem newer_train_distance_calculation : newer_train_distance = 390 := by
  sorry

end NUMINAMATH_CALUDE_newer_train_distance_calculation_l1768_176821


namespace NUMINAMATH_CALUDE_range_a_theorem_l1768_176813

-- Define propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x + 4 > 0
def q (a : ℝ) : Prop := a < 1 ∧ a ≠ 0

-- Define the range of a
def range_of_a (a : ℝ) : Prop := (1 ≤ a ∧ a < 2) ∨ a ≤ -2 ∨ a = 0

-- Theorem statement
theorem range_a_theorem (a : ℝ) : 
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → range_of_a a :=
sorry

end NUMINAMATH_CALUDE_range_a_theorem_l1768_176813


namespace NUMINAMATH_CALUDE_melody_reading_fraction_l1768_176811

theorem melody_reading_fraction (english : ℕ) (science : ℕ) (civics : ℕ) (chinese : ℕ) 
  (total_pages : ℕ) (h1 : english = 20) (h2 : science = 16) (h3 : civics = 8) (h4 : chinese = 12) 
  (h5 : total_pages = 14) :
  ∃ (f : ℚ), f * (english + science + civics + chinese : ℚ) = total_pages ∧ f = 1/4 := by
sorry

end NUMINAMATH_CALUDE_melody_reading_fraction_l1768_176811


namespace NUMINAMATH_CALUDE_ramsey_theorem_l1768_176861

-- Define a type for people
variable (Person : Type)

-- Define the acquaintance relation
variable (knows : Person → Person → Prop)

-- Axiom: The acquaintance relation is symmetric (mutual)
axiom knows_symmetric : ∀ (a b : Person), knows a b ↔ knows b a

-- Define a group of 6 people
variable (group : Finset Person)
axiom group_size : group.card = 6

-- Main theorem
theorem ramsey_theorem :
  ∃ (subset : Finset Person),
    subset.card = 3 ∧
    subset ⊆ group ∧
    (∀ (a b : Person), a ∈ subset → b ∈ subset → a ≠ b → knows a b) ∨
    (∀ (a b : Person), a ∈ subset → b ∈ subset → a ≠ b → ¬knows a b) :=
sorry

end NUMINAMATH_CALUDE_ramsey_theorem_l1768_176861


namespace NUMINAMATH_CALUDE_unique_integer_between_sqrt5_and_sqrt15_l1768_176857

theorem unique_integer_between_sqrt5_and_sqrt15 : 
  ∃! n : ℤ, (↑n : ℝ) > Real.sqrt 5 ∧ (↑n : ℝ) < Real.sqrt 15 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_integer_between_sqrt5_and_sqrt15_l1768_176857


namespace NUMINAMATH_CALUDE_gift_cost_l1768_176867

theorem gift_cost (half_cost : ℝ) (h : half_cost = 14) : 
  2 * half_cost = 28 := by
  sorry

end NUMINAMATH_CALUDE_gift_cost_l1768_176867


namespace NUMINAMATH_CALUDE_product_of_smallest_primes_l1768_176849

def smallest_one_digit_primes : List Nat := [2, 3]
def smallest_two_digit_prime : Nat := 11

theorem product_of_smallest_primes : 
  (smallest_one_digit_primes.prod * smallest_two_digit_prime) = 66 := by
  sorry

end NUMINAMATH_CALUDE_product_of_smallest_primes_l1768_176849


namespace NUMINAMATH_CALUDE_sine_equality_implies_equal_coefficients_l1768_176825

theorem sine_equality_implies_equal_coefficients 
  (α β γ δ : ℝ) 
  (h_positive : α > 0 ∧ β > 0 ∧ γ > 0 ∧ δ > 0) 
  (h_equality : ∀ x : ℝ, Real.sin (α * x) + Real.sin (β * x) = Real.sin (γ * x) + Real.sin (δ * x)) : 
  α = γ ∨ α = δ := by
sorry

end NUMINAMATH_CALUDE_sine_equality_implies_equal_coefficients_l1768_176825


namespace NUMINAMATH_CALUDE_paperclips_exceed_500_l1768_176831

def paperclips (n : ℕ) : ℕ := 5 * 4^n

theorem paperclips_exceed_500 : 
  (∃ k, paperclips k > 500) ∧ 
  (∀ j, j < 3 → paperclips j ≤ 500) ∧
  (paperclips 3 > 500) := by
  sorry

end NUMINAMATH_CALUDE_paperclips_exceed_500_l1768_176831


namespace NUMINAMATH_CALUDE_sine_fraction_simplification_l1768_176839

theorem sine_fraction_simplification (b : Real) (h : b = 2 * Real.pi / 13) :
  (Real.sin (4 * b) * Real.sin (8 * b) * Real.sin (10 * b) * Real.sin (12 * b) * Real.sin (14 * b)) /
  (Real.sin b * Real.sin (2 * b) * Real.sin (4 * b) * Real.sin (6 * b) * Real.sin (10 * b)) =
  Real.sin (10 * Real.pi / 13) / Real.sin (4 * Real.pi / 13) := by
  sorry

end NUMINAMATH_CALUDE_sine_fraction_simplification_l1768_176839


namespace NUMINAMATH_CALUDE_peanuts_in_box_l1768_176855

/-- The number of peanuts in a box after adding more -/
def total_peanuts (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem: If a box initially contains 4 peanuts and 2 more are added, the total is 6 -/
theorem peanuts_in_box : total_peanuts 4 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_peanuts_in_box_l1768_176855


namespace NUMINAMATH_CALUDE_mean_equality_implies_x_value_l1768_176840

theorem mean_equality_implies_x_value :
  let mean1 := (3 + 7 + 15) / 3
  let mean2 := (x + 10) / 2
  mean1 = mean2 → x = 20 / 3 :=
by sorry

end NUMINAMATH_CALUDE_mean_equality_implies_x_value_l1768_176840


namespace NUMINAMATH_CALUDE_exchange_rate_scaling_l1768_176887

theorem exchange_rate_scaling (x : ℝ) :
  2994 * 14.5 = 177 → 29.94 * 1.45 = 0.177 := by
  sorry

end NUMINAMATH_CALUDE_exchange_rate_scaling_l1768_176887


namespace NUMINAMATH_CALUDE_books_on_shelf_l1768_176826

theorem books_on_shelf (initial_books : ℕ) (added_books : ℕ) (removed_books : ℕ) 
  (h1 : initial_books = 38) 
  (h2 : added_books = 10) 
  (h3 : removed_books = 5) : 
  initial_books + added_books - removed_books = 43 := by
  sorry

end NUMINAMATH_CALUDE_books_on_shelf_l1768_176826


namespace NUMINAMATH_CALUDE_certain_number_equation_l1768_176864

theorem certain_number_equation : ∃ x : ℝ, 
  (5 * x - (2 * 1.4) / 1.3 = 4) ∧ 
  (abs (x - 1.23076923077) < 0.00000000001) := by
  sorry

end NUMINAMATH_CALUDE_certain_number_equation_l1768_176864


namespace NUMINAMATH_CALUDE_inequality_proof_l1768_176827

theorem inequality_proof (a b c m n p : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 1) 
  (h2 : m^2 + n^2 + p^2 = 1) : 
  (|a*m + b*n + c*p| ≤ 1) ∧ 
  (a*b*c ≠ 0 → m^4/a^2 + n^4/b^2 + p^4/c^2 ≥ 1) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1768_176827


namespace NUMINAMATH_CALUDE_consecutive_pages_sum_l1768_176836

theorem consecutive_pages_sum (n : ℕ) : 
  n * (n + 1) * (n + 2) = 136080 → n + (n + 1) + (n + 2) = 144 := by
sorry

end NUMINAMATH_CALUDE_consecutive_pages_sum_l1768_176836


namespace NUMINAMATH_CALUDE_range_of_a_l1768_176817

theorem range_of_a (a : ℝ) : 
  (∀ x, (a - 4 < x ∧ x < a + 4) → (x - 2) * (x - 3) > 0) →
  (a ≤ -2 ∨ a ≥ 7) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1768_176817


namespace NUMINAMATH_CALUDE_modulo_graph_intercepts_l1768_176844

theorem modulo_graph_intercepts (x₀ y₀ : ℕ) : 
  x₀ < 29 → y₀ < 29 → 
  (5 * x₀ ≡ 3 [ZMOD 29]) → 
  (2 * y₀ ≡ 26 [ZMOD 29]) → 
  x₀ + y₀ = 31 := by sorry

end NUMINAMATH_CALUDE_modulo_graph_intercepts_l1768_176844


namespace NUMINAMATH_CALUDE_loot_box_cost_l1768_176852

/-- Proves that the cost of each loot box is $5 given the specified conditions -/
theorem loot_box_cost (avg_value : ℝ) (total_spent : ℝ) (total_loss : ℝ) :
  avg_value = 3.5 →
  total_spent = 40 →
  total_loss = 12 →
  ∃ (cost : ℝ), cost = 5 ∧ cost * (total_spent - total_loss) / total_spent = avg_value :=
by
  sorry


end NUMINAMATH_CALUDE_loot_box_cost_l1768_176852


namespace NUMINAMATH_CALUDE_square_of_difference_three_minus_sqrt_two_l1768_176823

theorem square_of_difference_three_minus_sqrt_two : (3 - Real.sqrt 2)^2 = 11 - 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_difference_three_minus_sqrt_two_l1768_176823


namespace NUMINAMATH_CALUDE_maggie_plant_books_l1768_176872

/-- The number of books about plants Maggie bought -/
def num_plant_books : ℕ := 9

/-- The number of books about fish Maggie bought -/
def num_fish_books : ℕ := 1

/-- The number of science magazines Maggie bought -/
def num_magazines : ℕ := 10

/-- The cost of each book in dollars -/
def book_cost : ℕ := 15

/-- The cost of each magazine in dollars -/
def magazine_cost : ℕ := 2

/-- The total amount Maggie spent in dollars -/
def total_spent : ℕ := 170

theorem maggie_plant_books :
  num_plant_books * book_cost + num_fish_books * book_cost + num_magazines * magazine_cost = total_spent :=
by sorry

end NUMINAMATH_CALUDE_maggie_plant_books_l1768_176872


namespace NUMINAMATH_CALUDE_parabola_circle_tangency_l1768_176874

-- Define the parabola C
def parabola_C (x y : ℝ) : Prop := y^2 = x

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

-- Define a point on the parabola
def point_on_parabola (p : ℝ × ℝ) : Prop := parabola_C p.1 p.2

-- Define tangency of a line to the circle
def line_tangent_to_circle (p q : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), circle_M (p.1 + t * (q.1 - p.1)) (p.2 + t * (q.2 - p.2)) ∧
             ∀ (s : ℝ), s ≠ t → ¬circle_M (p.1 + s * (q.1 - p.1)) (p.2 + s * (q.2 - p.2))

theorem parabola_circle_tangency 
  (A₁ A₂ A₃ : ℝ × ℝ) 
  (h₁ : point_on_parabola A₁) 
  (h₂ : point_on_parabola A₂) 
  (h₃ : point_on_parabola A₃) 
  (h₄ : line_tangent_to_circle A₁ A₂) 
  (h₅ : line_tangent_to_circle A₁ A₃) : 
  line_tangent_to_circle A₂ A₃ := by
  sorry

end NUMINAMATH_CALUDE_parabola_circle_tangency_l1768_176874


namespace NUMINAMATH_CALUDE_consecutive_numbers_sum_l1768_176832

theorem consecutive_numbers_sum (n : ℕ) : 
  (n + (n + 1) + (n + 2) = 60) → ((n + 2) + (n + 3) + (n + 4) = 66) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_numbers_sum_l1768_176832


namespace NUMINAMATH_CALUDE_solve_equation_l1768_176878

theorem solve_equation (x : ℝ) : 3 * x = (26 - x) + 14 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1768_176878


namespace NUMINAMATH_CALUDE_inverse_difference_equals_negative_one_l1768_176803

theorem inverse_difference_equals_negative_one 
  (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = x * y) : 
  1 / x - 1 / y = -1 :=
sorry

end NUMINAMATH_CALUDE_inverse_difference_equals_negative_one_l1768_176803


namespace NUMINAMATH_CALUDE_problem_solution_l1768_176848

theorem problem_solution (x y : ℝ) (h : |x - 3| + Real.sqrt (x - y + 1) = 0) :
  Real.sqrt (x^2 * y + x * y^2 + 1/4 * y^3) = 10 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1768_176848


namespace NUMINAMATH_CALUDE_minimum_groups_l1768_176837

theorem minimum_groups (n : Nat) (h : n = 29) : 
  Nat.ceil (n / 4 : ℚ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_minimum_groups_l1768_176837


namespace NUMINAMATH_CALUDE_mr_zhang_birthday_l1768_176814

-- Define the possible dates
inductive Date
| feb5 | feb7 | feb9
| may5 | may8
| aug4 | aug7
| sep4 | sep6 | sep9

def Date.month : Date → Nat
| .feb5 | .feb7 | .feb9 => 2
| .may5 | .may8 => 5
| .aug4 | .aug7 => 8
| .sep4 | .sep6 | .sep9 => 9

def Date.day : Date → Nat
| .feb5 => 5
| .feb7 => 7
| .feb9 => 9
| .may5 => 5
| .may8 => 8
| .aug4 => 4
| .aug7 => 7
| .sep4 => 4
| .sep6 => 6
| .sep9 => 9

-- Define the statements made by A and B
def A_statement1 (d : Date) : Prop := 
  ∃ d' : Date, d.month = d'.month ∧ d ≠ d'

def B_statement (d : Date) : Prop :=
  ∀ d' : Date, A_statement1 d' → d.day ≠ d'.day

def A_statement2 (d : Date) : Prop :=
  ∀ d' : Date, A_statement1 d' ∧ B_statement d' → d = d'

-- Theorem to prove
theorem mr_zhang_birthday : 
  ∃! d : Date, A_statement1 d ∧ B_statement d ∧ A_statement2 d ∧ d = Date.aug4 := by
  sorry

end NUMINAMATH_CALUDE_mr_zhang_birthday_l1768_176814


namespace NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l1768_176800

theorem sum_of_solutions_is_zero : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ (6 * x₁ = 150 / x₁) ∧ (6 * x₂ = 150 / x₂) ∧ (x₁ + x₂ = 0) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l1768_176800


namespace NUMINAMATH_CALUDE_amoeba_population_l1768_176877

/-- The number of amoebas in the puddle after n days -/
def amoebas (n : ℕ) : ℕ :=
  3^n

/-- The number of days the amoeba population grows -/
def days : ℕ := 10

theorem amoeba_population : amoebas days = 59049 := by
  sorry

end NUMINAMATH_CALUDE_amoeba_population_l1768_176877


namespace NUMINAMATH_CALUDE_final_dog_count_l1768_176893

def initial_dogs : ℕ := 80
def adoption_rate : ℚ := 40 / 100
def returned_dogs : ℕ := 5

theorem final_dog_count : 
  initial_dogs - (initial_dogs * adoption_rate).floor + returned_dogs = 53 := by
  sorry

end NUMINAMATH_CALUDE_final_dog_count_l1768_176893


namespace NUMINAMATH_CALUDE_height_of_specific_prism_l1768_176883

/-- A right triangular prism with base PQR -/
structure RightTriangularPrism where
  /-- Length of side PQ of the base triangle -/
  pq : ℝ
  /-- Length of side PR of the base triangle -/
  pr : ℝ
  /-- Volume of the prism -/
  volume : ℝ

/-- Theorem: The height of a specific right triangular prism is 10 -/
theorem height_of_specific_prism (prism : RightTriangularPrism)
  (h_pq : prism.pq = Real.sqrt 5)
  (h_pr : prism.pr = Real.sqrt 5)
  (h_vol : prism.volume = 25.000000000000004) :
  (2 * prism.volume) / (prism.pq * prism.pr) = 10 := by
  sorry


end NUMINAMATH_CALUDE_height_of_specific_prism_l1768_176883


namespace NUMINAMATH_CALUDE_beth_crayons_l1768_176884

theorem beth_crayons (initial_packs : ℚ) : 
  (initial_packs / 10 + 6 = 6.4) → initial_packs = 4 := by
  sorry

end NUMINAMATH_CALUDE_beth_crayons_l1768_176884


namespace NUMINAMATH_CALUDE_number_division_problem_l1768_176856

theorem number_division_problem : ∃ x : ℚ, x / 11 + 156 = 178 ∧ x = 242 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l1768_176856


namespace NUMINAMATH_CALUDE_angle_B_measure_l1768_176859

theorem angle_B_measure :
  ∀ (A B : ℝ),
  A + B = 180 →  -- complementary angles sum to 180°
  B = 4 * A →    -- B is 4 times A
  B = 144 :=     -- B measures 144°
by
  sorry

end NUMINAMATH_CALUDE_angle_B_measure_l1768_176859


namespace NUMINAMATH_CALUDE_quadratic_equation_distinct_roots_l1768_176876

theorem quadratic_equation_distinct_roots (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ k * x^2 - 2 * x - 1 = 0 ∧ k * y^2 - 2 * y - 1 = 0) ↔ 
  (k > -1 ∧ k ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_distinct_roots_l1768_176876


namespace NUMINAMATH_CALUDE_average_pens_sold_per_day_l1768_176822

theorem average_pens_sold_per_day 
  (bundles_sold : ℕ) 
  (days : ℕ) 
  (pens_per_bundle : ℕ) 
  (h1 : bundles_sold = 15) 
  (h2 : days = 5) 
  (h3 : pens_per_bundle = 40) : 
  (bundles_sold * pens_per_bundle) / days = 120 := by
  sorry

end NUMINAMATH_CALUDE_average_pens_sold_per_day_l1768_176822


namespace NUMINAMATH_CALUDE_three_distinct_roots_condition_l1768_176892

theorem three_distinct_roots_condition (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    (|x^3 - a^3| = x - a) ∧
    (|y^3 - a^3| = y - a) ∧
    (|z^3 - a^3| = z - a)) ↔
  (-2 / Real.sqrt 3 < a ∧ a < -1 / Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_three_distinct_roots_condition_l1768_176892


namespace NUMINAMATH_CALUDE_part_one_part_two_l1768_176816

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |3 * x + m|

-- Define the function g
def g (m : ℝ) (x : ℝ) : ℝ := f m x - 2 * |x - 1|

-- Part I
theorem part_one (m : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc (-1) 3 ↔ f m x - m ≤ 9) → m = -3 := by sorry

-- Part II
theorem part_two (m : ℝ) (h_m_pos : m > 0) :
  (∃ A B C : ℝ × ℝ, 
    A.2 = 0 ∧ B.2 = 0 ∧ C.2 = g m C.1 ∧
    C.1 ∈ Set.Ioo A.1 B.1 ∧
    (1/2) * |B.1 - A.1| * |C.2| > 60) →
  m > 12 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1768_176816


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l1768_176863

/-- Represents a repeating decimal where the digit repeats indefinitely after the decimal point. -/
def RepeatingDecimal (digit : ℕ) : ℚ :=
  (digit : ℚ) / 9

/-- The sum of 0.4444... and 0.7777... is equal to 11/9. -/
theorem repeating_decimal_sum :
  RepeatingDecimal 4 + RepeatingDecimal 7 = 11 / 9 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l1768_176863


namespace NUMINAMATH_CALUDE_team_a_score_l1768_176880

theorem team_a_score (total_points team_b_points team_c_points : ℕ) 
  (h1 : team_b_points = 9)
  (h2 : team_c_points = 4)
  (h3 : total_points = 15)
  : total_points - (team_b_points + team_c_points) = 2 := by
  sorry

end NUMINAMATH_CALUDE_team_a_score_l1768_176880


namespace NUMINAMATH_CALUDE_rectangle_area_proof_l1768_176868

/-- Rectangle with known side length and area -/
structure Rectangle1 where
  side : ℝ
  area : ℝ

/-- Rectangle similar to Rectangle1 with known diagonal -/
structure Rectangle2 where
  diagonal : ℝ

/-- The area of Rectangle2 given the properties of Rectangle1 and Rectangle2 -/
def area_rectangle2 (r1 : Rectangle1) (r2 : Rectangle2) : ℝ :=
  160

theorem rectangle_area_proof (r1 : Rectangle1) (r2 : Rectangle2) 
  (h1 : r1.side = 4)
  (h2 : r1.area = 32)
  (h3 : r2.diagonal = 20) :
  area_rectangle2 r1 r2 = 160 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_proof_l1768_176868


namespace NUMINAMATH_CALUDE_garden_perimeter_l1768_176853

/-- A rectangular garden with equal length and breadth of 150 meters has a perimeter of 600 meters. -/
theorem garden_perimeter :
  ∀ (length breadth : ℝ),
  length > 0 →
  breadth > 0 →
  (length = 150 ∧ breadth = 150) →
  4 * length = 600 := by
sorry

end NUMINAMATH_CALUDE_garden_perimeter_l1768_176853


namespace NUMINAMATH_CALUDE_hypotenuse_length_l1768_176807

theorem hypotenuse_length (x y : ℝ) : 
  2 * x^2 - 8 * x + 7 = 0 →
  2 * y^2 - 8 * y + 7 = 0 →
  x ≠ y →
  x > 0 →
  y > 0 →
  x^2 + y^2 = 3^2 :=
by sorry

end NUMINAMATH_CALUDE_hypotenuse_length_l1768_176807


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1768_176802

def A : Set ℝ := {-1, 1, 2, 4}
def B : Set ℝ := {x : ℝ | |x - 1| ≤ 1}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1768_176802


namespace NUMINAMATH_CALUDE_power_tower_mod_1000_l1768_176845

theorem power_tower_mod_1000 : 5^(5^(5^5)) % 1000 = 125 := by sorry

end NUMINAMATH_CALUDE_power_tower_mod_1000_l1768_176845


namespace NUMINAMATH_CALUDE_oneRedOneWhite_bothWhite_mutually_exclusive_not_contradictory_l1768_176835

/-- Represents the color of a ball -/
inductive BallColor
| White
| Red

/-- Represents the outcome of drawing two balls -/
structure DrawOutcome :=
  (first second : BallColor)

/-- The bag containing balls -/
def bag : Multiset BallColor := 2 • {BallColor.White} + 3 • {BallColor.Red}

/-- Event: One red ball and one white ball -/
def oneRedOneWhite (outcome : DrawOutcome) : Prop :=
  (outcome.first = BallColor.Red ∧ outcome.second = BallColor.White) ∨
  (outcome.first = BallColor.White ∧ outcome.second = BallColor.Red)

/-- Event: Both balls are white -/
def bothWhite (outcome : DrawOutcome) : Prop :=
  outcome.first = BallColor.White ∧ outcome.second = BallColor.White

/-- Two events are mutually exclusive -/
def mutuallyExclusive (e1 e2 : DrawOutcome → Prop) : Prop :=
  ∀ outcome, ¬(e1 outcome ∧ e2 outcome)

/-- Two events are contradictory -/
def contradictory (e1 e2 : DrawOutcome → Prop) : Prop :=
  mutuallyExclusive e1 e2 ∧ ∀ outcome, e1 outcome ∨ e2 outcome

theorem oneRedOneWhite_bothWhite_mutually_exclusive_not_contradictory :
  mutuallyExclusive oneRedOneWhite bothWhite ∧
  ¬contradictory oneRedOneWhite bothWhite :=
sorry

end NUMINAMATH_CALUDE_oneRedOneWhite_bothWhite_mutually_exclusive_not_contradictory_l1768_176835


namespace NUMINAMATH_CALUDE_fathers_age_ratio_l1768_176896

theorem fathers_age_ratio (father_age ronit_age : ℕ) : 
  (father_age + 8 = (ronit_age + 8) * 5 / 2) →
  (father_age + 16 = (ronit_age + 16) * 2) →
  father_age = ronit_age * 4 := by
sorry

end NUMINAMATH_CALUDE_fathers_age_ratio_l1768_176896


namespace NUMINAMATH_CALUDE_range_of_a_l1768_176865

-- Define the propositions p and q
def p (x : ℝ) : Prop := x > 1 ∨ x < -3
def q (x a : ℝ) : Prop := x > a

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x, q x a → p x) ∧ (∃ x, p x ∧ ¬q x a)

-- Theorem statement
theorem range_of_a (a : ℝ) :
  sufficient_not_necessary a → a ∈ Set.Ici 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1768_176865


namespace NUMINAMATH_CALUDE_e_neg_4i_in_second_quadrant_l1768_176873

-- Define the complex exponential function
noncomputable def cexp (z : ℂ) : ℂ := Real.exp z.re * (Complex.cos z.im + Complex.I * Complex.sin z.im)

-- Define the quadrants of the complex plane
def in_second_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im > 0

-- Theorem statement
theorem e_neg_4i_in_second_quadrant : 
  in_second_quadrant (cexp (-4 * Complex.I)) :=
sorry

end NUMINAMATH_CALUDE_e_neg_4i_in_second_quadrant_l1768_176873


namespace NUMINAMATH_CALUDE_greatest_five_digit_with_product_90_l1768_176818

/-- A function that returns true if n is a five-digit number -/
def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999

/-- A function that returns the product of digits of a natural number -/
def digit_product (n : ℕ) : ℕ :=
  sorry

/-- A function that returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ :=
  sorry

/-- The greatest five-digit number whose digits have a product of 90 -/
def M : ℕ :=
  sorry

theorem greatest_five_digit_with_product_90 :
  is_five_digit M ∧
  digit_product M = 90 ∧
  (∀ n : ℕ, is_five_digit n → digit_product n = 90 → n ≤ M) ∧
  digit_sum M = 18 :=
sorry

end NUMINAMATH_CALUDE_greatest_five_digit_with_product_90_l1768_176818


namespace NUMINAMATH_CALUDE_mans_usual_time_to_office_l1768_176809

/-- Proves that if a man walks at 3/4 of his usual pace and arrives at his office 20 minutes late, 
    his usual time to reach the office is 80 minutes. -/
theorem mans_usual_time_to_office (usual_pace : ℝ) (usual_time : ℝ) 
    (h1 : usual_pace > 0) (h2 : usual_time > 0) : 
    (3 / 4 * usual_pace) * (usual_time + 20) = usual_pace * usual_time → 
    usual_time = 80 := by
  sorry


end NUMINAMATH_CALUDE_mans_usual_time_to_office_l1768_176809


namespace NUMINAMATH_CALUDE_tangent_line_at_x_1_l1768_176806

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 2*x + 1

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 2

-- Theorem statement
theorem tangent_line_at_x_1 :
  ∃ (m b : ℝ), 
    (∀ x y : ℝ, y = m*x + b ↔ x - y - 1 = 0) ∧
    m = f' 1 ∧
    f 1 = m*1 + b :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_x_1_l1768_176806


namespace NUMINAMATH_CALUDE_min_value_of_f_l1768_176882

noncomputable def f (x : ℝ) : ℝ := (x^2 - 4*x + 5) / (2*x - 4)

theorem min_value_of_f (x : ℝ) (h : x ≥ 5/2) : f x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1768_176882


namespace NUMINAMATH_CALUDE_marble_combinations_l1768_176897

-- Define the number of marbles
def total_marbles : ℕ := 9

-- Define the number of marbles to choose
def chosen_marbles : ℕ := 4

-- Define the combination function
def combination (n k : ℕ) : ℕ := Nat.choose n k

-- Theorem statement
theorem marble_combinations : combination total_marbles chosen_marbles = 126 := by
  sorry

end NUMINAMATH_CALUDE_marble_combinations_l1768_176897


namespace NUMINAMATH_CALUDE_birds_in_marsh_end_of_day_l1768_176819

/-- Calculates the total number of birds in the marsh at the end of the day -/
def total_birds_end_of_day (initial_geese initial_ducks geese_departed swans_arrived herons_arrived : ℕ) : ℕ :=
  (initial_geese - geese_departed) + initial_ducks + swans_arrived + herons_arrived

/-- Theorem stating the total number of birds at the end of the day -/
theorem birds_in_marsh_end_of_day :
  total_birds_end_of_day 58 37 15 22 2 = 104 := by
  sorry

end NUMINAMATH_CALUDE_birds_in_marsh_end_of_day_l1768_176819


namespace NUMINAMATH_CALUDE_lottery_probabilities_l1768_176850

/-- Represents the outcome of a customer's lottery participation -/
inductive LotteryResult
  | Gold
  | Silver
  | NoWin

/-- Models the lottery promotion scenario -/
structure LotteryPromotion where
  totalTickets : Nat
  surveySize : Nat
  noWinRatio : Rat
  silverRatioAmongWinners : Rat

/-- Calculates the probability of at least one gold prize winner among 3 randomly selected customers -/
def probAtLeastOneGold (lp : LotteryPromotion) : Rat :=
  sorry

/-- Calculates the probability that the number of gold prize winners is not more than 
    the number of silver prize winners among 3 randomly selected customers -/
def probGoldNotMoreThanSilver (lp : LotteryPromotion) : Rat :=
  sorry

/-- The main theorem stating the probabilities for the given lottery promotion scenario -/
theorem lottery_probabilities (lp : LotteryPromotion) 
  (h1 : lp.totalTickets = 2000)
  (h2 : lp.surveySize = 30)
  (h3 : lp.noWinRatio = 2/3)
  (h4 : lp.silverRatioAmongWinners = 3/5) :
  probAtLeastOneGold lp = 73/203 ∧ 
  probGoldNotMoreThanSilver lp = 157/203 := by
  sorry

end NUMINAMATH_CALUDE_lottery_probabilities_l1768_176850


namespace NUMINAMATH_CALUDE_outfit_combinations_l1768_176841

theorem outfit_combinations (short_sleeve : ℕ) (long_sleeve : ℕ) (jeans : ℕ) (formal_trousers : ℕ) :
  short_sleeve = 5 →
  long_sleeve = 3 →
  jeans = 6 →
  formal_trousers = 2 →
  (short_sleeve + long_sleeve) * (jeans + formal_trousers) = 64 :=
by
  sorry

end NUMINAMATH_CALUDE_outfit_combinations_l1768_176841


namespace NUMINAMATH_CALUDE_grapes_boxes_count_l1768_176804

def asparagus_bundles : ℕ := 60
def asparagus_price : ℚ := 3
def grape_price : ℚ := 2.5
def apple_count : ℕ := 700
def apple_price : ℚ := 0.5
def total_worth : ℚ := 630

theorem grapes_boxes_count :
  ∃ (grape_boxes : ℕ),
    grape_boxes * grape_price +
    asparagus_bundles * asparagus_price +
    apple_count * apple_price = total_worth ∧
    grape_boxes = 40 := by sorry

end NUMINAMATH_CALUDE_grapes_boxes_count_l1768_176804


namespace NUMINAMATH_CALUDE_ned_video_game_earnings_l1768_176875

/-- Calculates the total money earned from selling video games --/
def totalEarnings (totalGames : ℕ) (nonWorkingGames : ℕ) 
                  (firstGroupSize : ℕ) (firstGroupPrice : ℕ)
                  (secondGroupSize : ℕ) (secondGroupPrice : ℕ)
                  (remainingPrice : ℕ) : ℕ :=
  let workingGames := totalGames - nonWorkingGames
  let remainingGames := workingGames - firstGroupSize - secondGroupSize
  firstGroupSize * firstGroupPrice + 
  secondGroupSize * secondGroupPrice + 
  remainingGames * remainingPrice

/-- Theorem stating the total earnings from selling the working games --/
theorem ned_video_game_earnings : 
  totalEarnings 25 8 5 9 7 12 15 = 204 := by
  sorry

end NUMINAMATH_CALUDE_ned_video_game_earnings_l1768_176875


namespace NUMINAMATH_CALUDE_inscribed_dodecagon_radius_inscribed_dodecagon_radius_proof_l1768_176847

/-- The radius of a circle circumscribing a convex dodecagon with alternating side lengths of √2 and √24 is √38. -/
theorem inscribed_dodecagon_radius : ℝ → ℝ → ℝ → Prop :=
  fun (r : ℝ) (side1 : ℝ) (side2 : ℝ) =>
    side1 = Real.sqrt 2 ∧
    side2 = Real.sqrt 24 ∧
    r = Real.sqrt 38

/-- Proof of the theorem -/
theorem inscribed_dodecagon_radius_proof :
  ∃ (r : ℝ), inscribed_dodecagon_radius r (Real.sqrt 2) (Real.sqrt 24) :=
by
  sorry

#check inscribed_dodecagon_radius
#check inscribed_dodecagon_radius_proof

end NUMINAMATH_CALUDE_inscribed_dodecagon_radius_inscribed_dodecagon_radius_proof_l1768_176847


namespace NUMINAMATH_CALUDE_community_center_chairs_l1768_176869

/-- Converts a number from base 6 to base 10 -/
def base6ToBase10 (a b c : ℕ) : ℕ := a * 6^2 + b * 6 + c

/-- Calculates the number of chairs needed given the total people and people per chair -/
def calculateChairs (totalPeople : ℕ) (peoplePerChair : ℕ) : ℚ :=
  (totalPeople : ℚ) / peoplePerChair

theorem community_center_chairs :
  let seatingCapacity := base6ToBase10 2 3 1
  let peoplePerChair := 3
  calculateChairs seatingCapacity peoplePerChair = 30.33 := by sorry

end NUMINAMATH_CALUDE_community_center_chairs_l1768_176869


namespace NUMINAMATH_CALUDE_clock_angles_at_3_and_6_l1768_176899

/-- The angle between the hour hand and minute hand of a clock at a given time -/
def clock_angle (hour : ℕ) (minute : ℕ) : ℝ :=
  sorry

theorem clock_angles_at_3_and_6 :
  (clock_angle 3 0 = 90) ∧ (clock_angle 6 0 = 180) := by
  sorry

end NUMINAMATH_CALUDE_clock_angles_at_3_and_6_l1768_176899


namespace NUMINAMATH_CALUDE_petrol_price_increase_l1768_176801

theorem petrol_price_increase (original_price original_consumption : ℝ) 
  (h_positive_price : original_price > 0) 
  (h_positive_consumption : original_consumption > 0) : 
  let consumption_reduction := 23.076923076923073 / 100
  let new_consumption := original_consumption * (1 - consumption_reduction)
  let new_price := original_price * original_consumption / new_consumption
  (new_price - original_price) / original_price = 0.3 := by
sorry

end NUMINAMATH_CALUDE_petrol_price_increase_l1768_176801


namespace NUMINAMATH_CALUDE_sum_of_min_max_FGH_is_23_l1768_176858

/-- Represents a single digit (0-9) -/
def SingleDigit : Type := { n : ℕ // n < 10 }

/-- Represents a number in the form F861G20H -/
def NumberFGH (F G H : SingleDigit) : ℕ := 
  F.1 * 100000000 + 861 * 100000 + G.1 * 10000 + 20 * 100 + H.1

/-- Condition that F861G20H is divisible by 11 -/
def IsDivisibleBy11 (F G H : SingleDigit) : Prop :=
  NumberFGH F G H % 11 = 0

theorem sum_of_min_max_FGH_is_23 :
  ∃ (Fmin Gmin Hmin Fmax Gmax Hmax : SingleDigit),
    (∀ F G H : SingleDigit, IsDivisibleBy11 F G H →
      Fmin.1 + Gmin.1 + Hmin.1 ≤ F.1 + G.1 + H.1 ∧
      F.1 + G.1 + H.1 ≤ Fmax.1 + Gmax.1 + Hmax.1) ∧
    Fmin.1 + Gmin.1 + Hmin.1 + Fmax.1 + Gmax.1 + Hmax.1 = 23 :=
sorry

end NUMINAMATH_CALUDE_sum_of_min_max_FGH_is_23_l1768_176858


namespace NUMINAMATH_CALUDE_race_head_start_l1768_176890

theorem race_head_start (Va Vb D H : ℝ) :
  Va = (30 / 17) * Vb →
  D / Va = (D - H) / Vb →
  H = (13 / 30) * D :=
by sorry

end NUMINAMATH_CALUDE_race_head_start_l1768_176890


namespace NUMINAMATH_CALUDE_earth_moon_distance_scientific_notation_l1768_176898

/-- Represents the distance from Earth to Moon in kilometers -/
def earth_moon_distance : ℝ := 384401

/-- Converts a real number to scientific notation with given significant figures -/
def to_scientific_notation (x : ℝ) (sig_figs : ℕ) : ℝ × ℤ :=
  sorry

theorem earth_moon_distance_scientific_notation :
  to_scientific_notation earth_moon_distance 3 = (3.84, 5) :=
sorry

end NUMINAMATH_CALUDE_earth_moon_distance_scientific_notation_l1768_176898


namespace NUMINAMATH_CALUDE_coach_a_basketballs_l1768_176808

/-- The number of basketballs Coach A bought -/
def num_basketballs : ℕ := 10

/-- The cost of each basketball in dollars -/
def basketball_cost : ℚ := 29

/-- The total cost of Coach B's purchases in dollars -/
def coach_b_cost : ℚ := 14 * 2.5 + 18

/-- The difference in cost between Coach A and Coach B's purchases in dollars -/
def cost_difference : ℚ := 237

theorem coach_a_basketballs :
  basketball_cost * num_basketballs = coach_b_cost + cost_difference := by
  sorry


end NUMINAMATH_CALUDE_coach_a_basketballs_l1768_176808


namespace NUMINAMATH_CALUDE_largest_n_for_sin_cos_inequality_l1768_176834

theorem largest_n_for_sin_cos_inequality : 
  (∀ n : ℕ, n > 8 → ∃ x : ℝ, (Real.sin x)^n + (Real.cos x)^n < 1 / (2 * n)) ∧ 
  (∀ x : ℝ, (Real.sin x)^8 + (Real.cos x)^8 ≥ 1 / 16) := by
  sorry

end NUMINAMATH_CALUDE_largest_n_for_sin_cos_inequality_l1768_176834


namespace NUMINAMATH_CALUDE_books_borrowed_after_lunch_l1768_176812

theorem books_borrowed_after_lunch (initial_books : ℕ) (borrowed_by_lunch : ℕ) (added_after_lunch : ℕ) (remaining_by_evening : ℕ) : 
  initial_books = 100 →
  borrowed_by_lunch = 50 →
  added_after_lunch = 40 →
  remaining_by_evening = 60 →
  initial_books - borrowed_by_lunch + added_after_lunch - remaining_by_evening = 30 := by
sorry

end NUMINAMATH_CALUDE_books_borrowed_after_lunch_l1768_176812


namespace NUMINAMATH_CALUDE_calculation_proof_l1768_176833

theorem calculation_proof :
  (5 / (-5/3) * (-2) = 6) ∧
  (-(1^2) + 3 * (-2)^2 + (-9) / (-1/3)^2 = -70) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1768_176833


namespace NUMINAMATH_CALUDE_quadratic_composition_no_roots_l1768_176894

/-- A quadratic function f(x) = ax^2 + bx + c -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- The statement that f(x) = x has no real roots -/
def NoRealRoots (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x ≠ x

theorem quadratic_composition_no_roots (a b c : ℝ) (ha : a ≠ 0) :
  let f := QuadraticFunction a b c
  NoRealRoots f → NoRealRoots (f ∘ f) := by
  sorry

#check quadratic_composition_no_roots

end NUMINAMATH_CALUDE_quadratic_composition_no_roots_l1768_176894


namespace NUMINAMATH_CALUDE_curve_line_tangent_l1768_176815

/-- The curve y = √(4 - x²) and the line y = m have exactly one common point if and only if m = 2 -/
theorem curve_line_tangent (m : ℝ) : 
  (∃! p : ℝ × ℝ, p.2 = Real.sqrt (4 - p.1^2) ∧ p.2 = m) ↔ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_curve_line_tangent_l1768_176815


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l1768_176891

/-- Given a point M that is the midpoint of AB, and point A,
    prove that the sum of coordinates of B is as expected. -/
theorem midpoint_coordinate_sum (M A B : ℝ × ℝ) : 
  M = (3, 5) →  -- M has coordinates (3,5)
  A = (6, 8) →  -- A has coordinates (6,8)
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →  -- M is the midpoint of AB
  B.1 + B.2 = 2 :=  -- The sum of B's coordinates is 2
by
  sorry

#check midpoint_coordinate_sum

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l1768_176891


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1768_176879

theorem arithmetic_sequence_sum : ∀ (a₁ aₙ d n : ℕ),
  a₁ = 1 →
  aₙ = 21 →
  d = 2 →
  n * (a₁ + aₙ) = (aₙ - a₁ + d) * (aₙ - a₁ + d) →
  n * (a₁ + aₙ) / 2 = 121 :=
by
  sorry

#check arithmetic_sequence_sum

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1768_176879


namespace NUMINAMATH_CALUDE_f_2_eq_137_60_l1768_176885

def f (n : ℕ+) : ℚ :=
  Finset.sum (Finset.range (2 * n + 2)) (fun i => 1 / (i + 1 : ℚ))

theorem f_2_eq_137_60 : f 2 = 137 / 60 := by
  sorry

end NUMINAMATH_CALUDE_f_2_eq_137_60_l1768_176885


namespace NUMINAMATH_CALUDE_quadrilateral_area_l1768_176828

theorem quadrilateral_area (d h₁ h₂ : ℝ) (hd : d = 26) (hh₁ : h₁ = 9) (hh₂ : h₂ = 6) :
  (1/2) * d * (h₁ + h₂) = 195 :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l1768_176828


namespace NUMINAMATH_CALUDE_investment_satisfies_profit_ratio_q_investment_is_correct_l1768_176846

/-- Represents the investment amounts and profit ratio of two business partners -/
structure BusinessInvestment where
  p_investment : ℝ
  q_investment : ℝ
  p_profit_ratio : ℝ
  q_profit_ratio : ℝ

/-- The business investment scenario described in the problem -/
def problem_investment : BusinessInvestment where
  p_investment := 50000
  q_investment := 66666.67
  p_profit_ratio := 3
  q_profit_ratio := 4

/-- Theorem stating that the given investment amounts satisfy the profit ratio condition -/
theorem investment_satisfies_profit_ratio (bi : BusinessInvestment) :
  bi.p_investment / bi.q_investment = bi.p_profit_ratio / bi.q_profit_ratio →
  bi = problem_investment :=
by
  sorry

/-- Main theorem proving that q's investment is correct given the conditions -/
theorem q_investment_is_correct :
  ∃ (bi : BusinessInvestment),
    bi.p_investment = 50000 ∧
    bi.p_profit_ratio = 3 ∧
    bi.q_profit_ratio = 4 ∧
    bi.p_investment / bi.q_investment = bi.p_profit_ratio / bi.q_profit_ratio ∧
    bi.q_investment = 66666.67 :=
by
  sorry

end NUMINAMATH_CALUDE_investment_satisfies_profit_ratio_q_investment_is_correct_l1768_176846


namespace NUMINAMATH_CALUDE_monkey_banana_theorem_l1768_176895

/-- Represents the monkey's banana transportation problem -/
structure BananaProblem where
  total_bananas : ℕ
  distance : ℕ
  max_carry : ℕ
  eat_rate : ℕ

/-- Calculates the maximum number of bananas the monkey can bring home -/
def max_bananas_home (problem : BananaProblem) : ℕ :=
  sorry

/-- Theorem stating that for the given problem, the maximum number of bananas brought home is 25 -/
theorem monkey_banana_theorem (problem : BananaProblem) 
  (h1 : problem.total_bananas = 100)
  (h2 : problem.distance = 50)
  (h3 : problem.max_carry = 50)
  (h4 : problem.eat_rate = 1) :
  max_bananas_home problem = 25 := by
  sorry

end NUMINAMATH_CALUDE_monkey_banana_theorem_l1768_176895


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l1768_176881

theorem smallest_solution_of_equation :
  let f (x : ℝ) := 2 * x / (x - 2) + (2 * x^2 - 24) / x - 11
  ∃ (y : ℝ), y = (1 - Real.sqrt 65) / 4 ∧ f y = 0 ∧ ∀ (z : ℝ), f z = 0 → y ≤ z :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l1768_176881


namespace NUMINAMATH_CALUDE_quadratic_radical_combination_l1768_176854

theorem quadratic_radical_combination (x : ℝ) : 
  (∃ y : ℝ, y^2 = 2*x - 1 ∧ y = Real.sqrt 3) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_radical_combination_l1768_176854


namespace NUMINAMATH_CALUDE_number_of_boys_l1768_176843

theorem number_of_boys (total : ℕ) (girls : ℕ) (boys : ℕ) : 
  girls = (60 : ℕ) * total / 100 →
  girls = 450 →
  boys = total - girls →
  boys = 300 := by
sorry

end NUMINAMATH_CALUDE_number_of_boys_l1768_176843
