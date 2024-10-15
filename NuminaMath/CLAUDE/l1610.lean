import Mathlib

namespace NUMINAMATH_CALUDE_complex_exp_form_l1610_161096

theorem complex_exp_form (z : ℂ) : z = 1 + Complex.I * Real.sqrt 3 → Complex.arg z = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_exp_form_l1610_161096


namespace NUMINAMATH_CALUDE_triangle_perimeter_l1610_161001

theorem triangle_perimeter (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  c * Real.cos B + b * Real.cos C = 2 * a * Real.cos A →
  a = 2 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 →
  a + b + c = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l1610_161001


namespace NUMINAMATH_CALUDE_projection_property_l1610_161094

/-- A projection that takes (3, -3) to (75/26, -15/26) -/
def projection (v : ℝ × ℝ) : ℝ × ℝ :=
  sorry

theorem projection_property :
  projection (3, -3) = (75/26, -15/26) →
  projection ((5, 7) + (-3, -4)) = (35/26, -7/26) :=
by
  sorry

end NUMINAMATH_CALUDE_projection_property_l1610_161094


namespace NUMINAMATH_CALUDE_cubic_root_equality_l1610_161042

theorem cubic_root_equality (p q r : ℝ) : 
  (∀ x : ℝ, x^3 - 3*p*x^2 + 3*q^2*x - r^3 = 0 ↔ (x = p ∨ x = q ∨ x = r)) →
  p = q ∧ q = r :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_equality_l1610_161042


namespace NUMINAMATH_CALUDE_expression_evaluation_l1610_161099

theorem expression_evaluation :
  (5^1003 + 7^1004)^2 - (5^1003 - 7^1004)^2 = 28 * 35^1003 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1610_161099


namespace NUMINAMATH_CALUDE_max_tan_A_l1610_161054

theorem max_tan_A (A B : Real) (h1 : 0 < A) (h2 : A < π/2) (h3 : 0 < B) (h4 : B < π/2)
  (h5 : 3 * Real.sin A = Real.cos (A + B) * Real.sin B) :
  ∃ (max_tan_A : Real), ∀ (A' B' : Real),
    0 < A' → A' < π/2 → 0 < B' → B' < π/2 →
    3 * Real.sin A' = Real.cos (A' + B') * Real.sin B' →
    Real.tan A' ≤ max_tan_A ∧
    max_tan_A = Real.sqrt 3 / 12 := by
  sorry

end NUMINAMATH_CALUDE_max_tan_A_l1610_161054


namespace NUMINAMATH_CALUDE_sawing_time_l1610_161032

/-- Given that sawing a steel bar into 2 pieces takes 2 minutes,
    this theorem proves that sawing the same bar into 6 pieces takes 10 minutes. -/
theorem sawing_time (time_for_two_pieces : ℕ) (pieces : ℕ) : 
  time_for_two_pieces = 2 → pieces = 6 → (pieces - 1) * (time_for_two_pieces / (2 - 1)) = 10 := by
  sorry

end NUMINAMATH_CALUDE_sawing_time_l1610_161032


namespace NUMINAMATH_CALUDE_probability_no_defective_bulbs_l1610_161059

def total_bulbs : ℕ := 10
def defective_bulbs : ℕ := 4
def selected_bulbs : ℕ := 4

theorem probability_no_defective_bulbs :
  (Nat.choose (total_bulbs - defective_bulbs) selected_bulbs) /
  (Nat.choose total_bulbs selected_bulbs) = 1 / 14 :=
by sorry

end NUMINAMATH_CALUDE_probability_no_defective_bulbs_l1610_161059


namespace NUMINAMATH_CALUDE_arccos_one_half_equals_pi_third_l1610_161083

theorem arccos_one_half_equals_pi_third : Real.arccos (1/2) = π/3 := by
  sorry

end NUMINAMATH_CALUDE_arccos_one_half_equals_pi_third_l1610_161083


namespace NUMINAMATH_CALUDE_quadratic_monotonicity_l1610_161043

/-- A function f is monotonically increasing on (a, +∞) if for all x, y > a, x < y implies f(x) < f(y) -/
def MonoIncreasing (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x y, x > a → y > a → x < y → f x < f y

/-- The quadratic function f(x) = x^2 + mx - 2 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x - 2

theorem quadratic_monotonicity (m : ℝ) :
  MonoIncreasing (f m) 2 → m ≥ -4 := by sorry

end NUMINAMATH_CALUDE_quadratic_monotonicity_l1610_161043


namespace NUMINAMATH_CALUDE_integral_sin_plus_sqrt_one_minus_x_squared_l1610_161081

theorem integral_sin_plus_sqrt_one_minus_x_squared : 
  ∫ x in (-1)..1, (Real.sin x + Real.sqrt (1 - x^2)) = π / 2 := by sorry

end NUMINAMATH_CALUDE_integral_sin_plus_sqrt_one_minus_x_squared_l1610_161081


namespace NUMINAMATH_CALUDE_retail_price_problem_l1610_161060

/-- The retail price problem -/
theorem retail_price_problem
  (wholesale_price : ℝ)
  (discount_rate : ℝ)
  (profit_rate : ℝ)
  (h1 : wholesale_price = 90)
  (h2 : discount_rate = 0.1)
  (h3 : profit_rate = 0.2)
  (retail_price : ℝ) :
  retail_price = 120 ↔
    retail_price * (1 - discount_rate) = 
      wholesale_price * (1 + profit_rate) :=
by sorry

end NUMINAMATH_CALUDE_retail_price_problem_l1610_161060


namespace NUMINAMATH_CALUDE_perfect_squares_between_50_and_250_l1610_161070

theorem perfect_squares_between_50_and_250 : 
  (Finset.filter (fun n => 50 < n * n ∧ n * n < 250) (Finset.range 16)).card = 8 := by
  sorry

end NUMINAMATH_CALUDE_perfect_squares_between_50_and_250_l1610_161070


namespace NUMINAMATH_CALUDE_x2y_plus_1_is_third_degree_binomial_l1610_161038

/-- A binomial is a polynomial with exactly two terms. -/
def is_binomial (p : Polynomial ℝ) : Prop :=
  p.support.card = 2

/-- The degree of a polynomial is the highest degree of any of its terms. -/
def polynomial_degree (p : Polynomial ℝ) : ℕ := p.natDegree

/-- A third-degree polynomial has a degree of 3. -/
def is_third_degree (p : Polynomial ℝ) : Prop :=
  polynomial_degree p = 3

theorem x2y_plus_1_is_third_degree_binomial :
  let p : Polynomial ℝ := X^2 * Y + 1
  is_binomial p ∧ is_third_degree p :=
sorry

end NUMINAMATH_CALUDE_x2y_plus_1_is_third_degree_binomial_l1610_161038


namespace NUMINAMATH_CALUDE_quadrilateral_area_l1610_161084

/-- Quadrilateral PQRS with given side lengths -/
structure Quadrilateral :=
  (PS : ℝ)
  (SR : ℝ)
  (PQ : ℝ)
  (RQ : ℝ)

/-- The area of the quadrilateral PQRS is 36 -/
theorem quadrilateral_area (q : Quadrilateral) 
  (h1 : q.PS = 3)
  (h2 : q.SR = 4)
  (h3 : q.PQ = 13)
  (h4 : q.RQ = 12) : 
  ∃ (area : ℝ), area = 36 := by
  sorry

#check quadrilateral_area

end NUMINAMATH_CALUDE_quadrilateral_area_l1610_161084


namespace NUMINAMATH_CALUDE_inequality_proof_l1610_161008

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / Real.sqrt (a^2 + 8*b*c)) + (b / Real.sqrt (b^2 + 8*c*a)) + (c / Real.sqrt (c^2 + 8*a*b)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1610_161008


namespace NUMINAMATH_CALUDE_kim_test_probability_l1610_161045

theorem kim_test_probability (p : ℚ) (h : p = 5/8) : 1 - p = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_kim_test_probability_l1610_161045


namespace NUMINAMATH_CALUDE_ice_cream_scoops_left_l1610_161025

/-- Represents the flavors of ice cream --/
inductive Flavor
  | Chocolate
  | Strawberry
  | Vanilla

/-- Represents a person --/
inductive Person
  | Ethan
  | Lucas
  | Danny
  | Connor
  | Olivia
  | Shannon

/-- The number of scoops in each carton --/
def scoops_per_carton : ℕ := 10

/-- The initial number of scoops for each flavor --/
def initial_scoops (f : Flavor) : ℕ := scoops_per_carton

/-- The number of scoops a person wants for each flavor --/
def scoops_wanted (p : Person) (f : Flavor) : ℕ :=
  match p, f with
  | Person.Ethan, Flavor.Chocolate => 1
  | Person.Ethan, Flavor.Vanilla => 1
  | Person.Lucas, Flavor.Chocolate => 2
  | Person.Danny, Flavor.Chocolate => 2
  | Person.Connor, Flavor.Chocolate => 2
  | Person.Olivia, Flavor.Strawberry => 1
  | Person.Olivia, Flavor.Vanilla => 1
  | Person.Shannon, Flavor.Strawberry => 2
  | Person.Shannon, Flavor.Vanilla => 2
  | _, _ => 0

/-- The total number of scoops taken for each flavor --/
def total_scoops_taken (f : Flavor) : ℕ :=
  (scoops_wanted Person.Ethan f) +
  (scoops_wanted Person.Lucas f) +
  (scoops_wanted Person.Danny f) +
  (scoops_wanted Person.Connor f) +
  (scoops_wanted Person.Olivia f) +
  (scoops_wanted Person.Shannon f)

/-- The number of scoops left for each flavor --/
def scoops_left (f : Flavor) : ℕ :=
  initial_scoops f - total_scoops_taken f

/-- The total number of scoops left --/
def total_scoops_left : ℕ :=
  (scoops_left Flavor.Chocolate) +
  (scoops_left Flavor.Strawberry) +
  (scoops_left Flavor.Vanilla)

theorem ice_cream_scoops_left : total_scoops_left = 16 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_scoops_left_l1610_161025


namespace NUMINAMATH_CALUDE_intersection_nonempty_iff_a_in_range_union_equals_interval_iff_a_equals_two_l1610_161055

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | 2 < x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x : ℝ | a < x ∧ x < 3*a}

-- Theorem for part (1)
theorem intersection_nonempty_iff_a_in_range (a : ℝ) :
  (A ∩ B a).Nonempty ↔ (4/3 ≤ a ∧ a < 4) :=
sorry

-- Theorem for part (2)
theorem union_equals_interval_iff_a_equals_two (a : ℝ) :
  A ∪ B a = {x : ℝ | 2 < x ∧ x < 6} ↔ a = 2 :=
sorry

end NUMINAMATH_CALUDE_intersection_nonempty_iff_a_in_range_union_equals_interval_iff_a_equals_two_l1610_161055


namespace NUMINAMATH_CALUDE_sum_of_roots_is_6_l1610_161069

-- Define a quadratic function
variable (f : ℝ → ℝ)

-- Define the symmetry property
def is_symmetric_about_3 (f : ℝ → ℝ) : Prop :=
  ∀ x, f (3 + x) = f (3 - x)

-- Define the property of having two real roots
def has_two_real_roots (f : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0

-- Theorem statement
theorem sum_of_roots_is_6 (f : ℝ → ℝ) 
  (h_sym : is_symmetric_about_3 f) 
  (h_roots : has_two_real_roots f) :
  ∃ x₁ x₂ : ℝ, has_two_real_roots f ∧ x₁ + x₂ = 6 :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_is_6_l1610_161069


namespace NUMINAMATH_CALUDE_equal_water_amounts_l1610_161073

theorem equal_water_amounts (hot_fill_time cold_fill_time : ℝ) 
  (h_hot : hot_fill_time = 23)
  (h_cold : cold_fill_time = 19) :
  let delay := 2
  let hot_rate := 1 / hot_fill_time
  let cold_rate := 1 / cold_fill_time
  let total_time := hot_fill_time / 2 + delay
  hot_rate * total_time = cold_rate * (total_time - delay) :=
by sorry

end NUMINAMATH_CALUDE_equal_water_amounts_l1610_161073


namespace NUMINAMATH_CALUDE_sam_container_capacity_l1610_161015

/-- Represents a rectangular container with dimensions and marble capacity. -/
structure Container where
  length : ℝ
  width : ℝ
  height : ℝ
  capacity : ℕ

/-- Calculates the volume of a container. -/
def containerVolume (c : Container) : ℝ :=
  c.length * c.width * c.height

/-- Theorem: Given Ellie's container dimensions and capacity, and the relative dimensions
    of Sam's container, Sam's container holds 1200 marbles. -/
theorem sam_container_capacity
  (ellie : Container)
  (h_ellie_dims : ellie.length = 2 ∧ ellie.width = 3 ∧ ellie.height = 4)
  (h_ellie_capacity : ellie.capacity = 200)
  (sam : Container)
  (h_sam_dims : sam.length = ellie.length ∧ 
                sam.width = 2 * ellie.width ∧ 
                sam.height = 3 * ellie.height) :
  sam.capacity = 1200 := by
sorry


end NUMINAMATH_CALUDE_sam_container_capacity_l1610_161015


namespace NUMINAMATH_CALUDE_principal_calculation_l1610_161051

/-- The compound interest formula for yearly compounding -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- The problem statement -/
theorem principal_calculation (final_amount : ℝ) (rate : ℝ) (time : ℕ)
  (h_final : final_amount = 3087)
  (h_rate : rate = 0.05)
  (h_time : time = 2) :
  ∃ principal : ℝ, 
    compound_interest principal rate time = final_amount ∧ 
    principal = 2800 := by
  sorry

end NUMINAMATH_CALUDE_principal_calculation_l1610_161051


namespace NUMINAMATH_CALUDE_floor_e_squared_l1610_161068

theorem floor_e_squared : ⌊Real.exp 1 ^ 2⌋ = 7 := by sorry

end NUMINAMATH_CALUDE_floor_e_squared_l1610_161068


namespace NUMINAMATH_CALUDE_fraction_ratio_equality_l1610_161064

theorem fraction_ratio_equality : 
  ∃ (x y : ℚ), x / y = (240 : ℚ) / 1547 ∧ 
  x / y / ((2 : ℚ) / 13) = ((5 : ℚ) / 34) / ((7 : ℚ) / 48) := by
  sorry

end NUMINAMATH_CALUDE_fraction_ratio_equality_l1610_161064


namespace NUMINAMATH_CALUDE_equal_rectangle_count_l1610_161029

def count_rectangles (perimeter : ℕ) : ℕ :=
  (perimeter / 2 - 1) / 2

theorem equal_rectangle_count :
  count_rectangles 1996 = count_rectangles 1998 :=
by sorry

end NUMINAMATH_CALUDE_equal_rectangle_count_l1610_161029


namespace NUMINAMATH_CALUDE_perfect_square_condition_l1610_161000

theorem perfect_square_condition (n : ℕ) : 
  ∃ k : ℕ, n^2 + 3*n = k^2 ↔ n = 1 := by sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l1610_161000


namespace NUMINAMATH_CALUDE_worker_production_equations_l1610_161013

/-- Represents the daily production of workers in a company -/
structure WorkerProduction where
  novice : ℕ
  experienced : ℕ

/-- The conditions of the worker production problem -/
class WorkerProductionProblem (w : WorkerProduction) where
  experience_difference : w.experienced - w.novice = 30
  total_production : w.novice + 2 * w.experienced = 180

/-- The theorem stating the correct system of equations for the worker production problem -/
theorem worker_production_equations (w : WorkerProduction) [WorkerProductionProblem w] :
  (w.experienced - w.novice = 30) ∧ (w.novice + 2 * w.experienced = 180) := by
  sorry

end NUMINAMATH_CALUDE_worker_production_equations_l1610_161013


namespace NUMINAMATH_CALUDE_books_from_first_shop_is_32_l1610_161046

/-- Represents the number of books bought from the first shop -/
def books_from_first_shop : ℕ := sorry

/-- The total amount spent on books from the first shop in Rs -/
def amount_first_shop : ℕ := 1500

/-- The number of books bought from the second shop -/
def books_from_second_shop : ℕ := 60

/-- The total amount spent on books from the second shop in Rs -/
def amount_second_shop : ℕ := 340

/-- The average price per book for all books in Rs -/
def average_price : ℕ := 20

/-- Theorem stating that the number of books bought from the first shop is 32 -/
theorem books_from_first_shop_is_32 : books_from_first_shop = 32 := by
  sorry

end NUMINAMATH_CALUDE_books_from_first_shop_is_32_l1610_161046


namespace NUMINAMATH_CALUDE_best_shooter_D_l1610_161005

structure Shooter where
  name : String
  average_score : ℝ
  variance : ℝ

def is_best_shooter (s : Shooter) (shooters : List Shooter) : Prop :=
  (∀ t ∈ shooters, s.average_score ≥ t.average_score) ∧
  (∀ t ∈ shooters, s.average_score = t.average_score → s.variance ≤ t.variance)

theorem best_shooter_D :
  let shooters := [
    ⟨"A", 9, 1.2⟩,
    ⟨"B", 8, 0.4⟩,
    ⟨"C", 9, 1.8⟩,
    ⟨"D", 9, 0.4⟩
  ]
  let D := ⟨"D", 9, 0.4⟩
  is_best_shooter D shooters := by
  sorry

#check best_shooter_D

end NUMINAMATH_CALUDE_best_shooter_D_l1610_161005


namespace NUMINAMATH_CALUDE_fraction_power_four_l1610_161028

theorem fraction_power_four : (5 / 6 : ℚ) ^ 4 = 625 / 1296 := by
  sorry

end NUMINAMATH_CALUDE_fraction_power_four_l1610_161028


namespace NUMINAMATH_CALUDE_malcolm_route_ratio_l1610_161071

/-- Malcolm's route to school problem -/
theorem malcolm_route_ratio : 
  ∀ (r : ℝ), 
  (6 + 6*r + (1/3)*(6 + 6*r) + 18 = 42) → 
  r = 17/4 := by
sorry

end NUMINAMATH_CALUDE_malcolm_route_ratio_l1610_161071


namespace NUMINAMATH_CALUDE_prime_sum_probability_l1610_161016

/-- A function that returns true if a number is prime, false otherwise -/
def isPrime (n : ℕ) : Bool := sorry

/-- The number of dice being rolled -/
def numDice : ℕ := 7

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The total number of possible outcomes when rolling numDice dice -/
def totalOutcomes : ℕ := numFaces ^ numDice

/-- The number of outcomes that result in a prime sum -/
def primeOutcomes : ℕ := 80425

/-- The probability of rolling numDice dice and obtaining a prime sum -/
def primeProbability : ℚ := primeOutcomes / totalOutcomes

theorem prime_sum_probability :
  primeProbability = 26875 / 93312 := by sorry

end NUMINAMATH_CALUDE_prime_sum_probability_l1610_161016


namespace NUMINAMATH_CALUDE_train_bridge_time_l1610_161086

/-- Time for a train to pass a bridge -/
theorem train_bridge_time (train_length bridge_length : ℝ) (train_speed_kmh : ℝ) :
  train_length = 360 →
  bridge_length = 140 →
  train_speed_kmh = 45 →
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 40 := by
  sorry

end NUMINAMATH_CALUDE_train_bridge_time_l1610_161086


namespace NUMINAMATH_CALUDE_triangle_inequality_two_points_l1610_161037

/-- Triangle inequality for two points in the plane of a triangle -/
theorem triangle_inequality_two_points (A B C P₁ P₂ : ℝ × ℝ) 
  (a : ℝ) (b : ℝ) (c : ℝ) 
  (a₁ : ℝ) (b₁ : ℝ) (c₁ : ℝ) 
  (a₂ : ℝ) (b₂ : ℝ) (c₂ : ℝ) 
  (ha : a = dist B C) 
  (hb : b = dist A C) 
  (hc : c = dist A B) 
  (ha₁ : a₁ = dist P₁ A) 
  (hb₁ : b₁ = dist P₁ B) 
  (hc₁ : c₁ = dist P₁ C) 
  (ha₂ : a₂ = dist P₂ A) 
  (hb₂ : b₂ = dist P₂ B) 
  (hc₂ : c₂ = dist P₂ C) : 
  a * a₁ * a₂ + b * b₁ * b₂ + c * c₁ * c₂ ≥ a * b * c :=
sorry

#check triangle_inequality_two_points

end NUMINAMATH_CALUDE_triangle_inequality_two_points_l1610_161037


namespace NUMINAMATH_CALUDE_factorization_difference_of_squares_l1610_161007

theorem factorization_difference_of_squares (a b : ℝ) : a^2 * b^2 - 9 = (a*b + 3) * (a*b - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_difference_of_squares_l1610_161007


namespace NUMINAMATH_CALUDE_regular_polygon_perimeter_l1610_161079

/-- A regular polygon with side length 7 and exterior angle 90 degrees has a perimeter of 28 units. -/
theorem regular_polygon_perimeter (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) : 
  n > 0 ∧ 
  side_length = 7 ∧ 
  exterior_angle = 90 ∧ 
  (360 : ℝ) / n = exterior_angle → 
  n * side_length = 28 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_perimeter_l1610_161079


namespace NUMINAMATH_CALUDE_changed_number_proof_l1610_161090

theorem changed_number_proof (a b c d e : ℝ) : 
  (a + b + c + d + e) / 5 = 8 →
  (8 + b + c + d + e) / 5 = 9 →
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_changed_number_proof_l1610_161090


namespace NUMINAMATH_CALUDE_special_triangle_side_lengths_l1610_161091

/-- Triangle with consecutive integer side lengths and perpendicular median and angle bisector -/
structure SpecialTriangle where
  -- Side lengths
  a : ℕ
  b : ℕ
  c : ℕ
  -- Consecutive integer side lengths
  consecutive_sides : c = b + 1 ∧ b = a + 1
  -- Median from A
  median_a : ℝ × ℝ
  -- Angle bisector from B
  bisector_b : ℝ × ℝ
  -- Perpendicularity condition
  perpendicular : median_a.1 * bisector_b.1 + median_a.2 * bisector_b.2 = 0

/-- The side lengths of a special triangle are 2, 3, and 4 -/
theorem special_triangle_side_lengths (t : SpecialTriangle) : t.a = 2 ∧ t.b = 3 ∧ t.c = 4 :=
sorry

end NUMINAMATH_CALUDE_special_triangle_side_lengths_l1610_161091


namespace NUMINAMATH_CALUDE_zeros_after_one_in_10000_to_50_l1610_161058

theorem zeros_after_one_in_10000_to_50 : 
  (∃ n : ℕ, 10000^50 = 10^n ∧ n = 200) :=
by sorry

end NUMINAMATH_CALUDE_zeros_after_one_in_10000_to_50_l1610_161058


namespace NUMINAMATH_CALUDE_uncle_gift_amount_l1610_161052

/-- The amount of money Geoffrey's uncle gave him --/
def uncle_gift (grandmother_gift aunt_gift total_after_gifts spent_on_games money_left : ℕ) : ℕ :=
  total_after_gifts - grandmother_gift - aunt_gift - money_left

/-- Theorem stating the amount of money Geoffrey's uncle gave him --/
theorem uncle_gift_amount : 
  uncle_gift 20 25 125 105 20 = 60 := by
  sorry

#eval uncle_gift 20 25 125 105 20

end NUMINAMATH_CALUDE_uncle_gift_amount_l1610_161052


namespace NUMINAMATH_CALUDE_age_sum_is_21_l1610_161011

/-- Given two people p and q, where 6 years ago p was half the age of q,
    and the ratio of their present ages is 3:4, prove that the sum of
    their present ages is 21 years. -/
theorem age_sum_is_21 (p q : ℕ) : 
  (p - 6 = (q - 6) / 2) →  -- 6 years ago, p was half of q in age
  (p : ℚ) / q = 3 / 4 →    -- The ratio of their present ages is 3:4
  p + q = 21 :=            -- The sum of their present ages is 21
by sorry

end NUMINAMATH_CALUDE_age_sum_is_21_l1610_161011


namespace NUMINAMATH_CALUDE_lattice_points_on_segment_l1610_161041

/-- The number of lattice points on a line segment -/
def latticePointCount (x1 y1 x2 y2 : ℤ) : ℕ :=
  sorry

/-- Theorem stating the number of lattice points on the given line segment -/
theorem lattice_points_on_segment : latticePointCount 5 23 60 353 = 56 := by
  sorry

end NUMINAMATH_CALUDE_lattice_points_on_segment_l1610_161041


namespace NUMINAMATH_CALUDE_fly_journey_l1610_161082

theorem fly_journey (r : ℝ) (s : ℝ) (h1 : r = 65) (h2 : s = 90) :
  let d := 2 * r
  let b := Real.sqrt (d^2 - s^2)
  d + s + b = 314 :=
by sorry

end NUMINAMATH_CALUDE_fly_journey_l1610_161082


namespace NUMINAMATH_CALUDE_smallest_sum_prime_set_l1610_161023

/-- A set of natural numbers uses each digit exactly once -/
def uses_each_digit_once (s : Finset ℕ) : Prop :=
  ∃ (digits : Finset ℕ), digits.card = 10 ∧
    ∀ d ∈ digits, 0 ≤ d ∧ d < 10 ∧
    ∀ n ∈ s, ∀ k, 0 ≤ k ∧ k < 10 → (n / 10^k % 10) ∈ digits

/-- The sum of a set of natural numbers -/
def set_sum (s : Finset ℕ) : ℕ := s.sum id

/-- The theorem to be proved -/
theorem smallest_sum_prime_set :
  ∃ (s : Finset ℕ),
    (∀ n ∈ s, Nat.Prime n) ∧
    uses_each_digit_once s ∧
    set_sum s = 4420 ∧
    (∀ t : Finset ℕ, (∀ n ∈ t, Nat.Prime n) → uses_each_digit_once t → set_sum s ≤ set_sum t) :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_prime_set_l1610_161023


namespace NUMINAMATH_CALUDE_time_to_paint_one_house_l1610_161030

/-- Given that 9 houses can be painted in 3 hours, prove that one house can be painted in 20 minutes. -/
theorem time_to_paint_one_house : 
  ∀ (total_houses : ℕ) (total_hours : ℕ) (minutes_per_hour : ℕ),
  total_houses = 9 →
  total_hours = 3 →
  minutes_per_hour = 60 →
  (total_hours * minutes_per_hour) / total_houses = 20 := by
  sorry

end NUMINAMATH_CALUDE_time_to_paint_one_house_l1610_161030


namespace NUMINAMATH_CALUDE_transformed_quadratic_roots_l1610_161080

theorem transformed_quadratic_roots (α β : ℂ) : 
  (3 * α^2 + 2 * α + 1 = 0) → 
  (3 * β^2 + 2 * β + 1 = 0) → 
  ((3 * α + 2)^2 + 4 = 0) ∧ ((3 * β + 2)^2 + 4 = 0) := by
sorry

end NUMINAMATH_CALUDE_transformed_quadratic_roots_l1610_161080


namespace NUMINAMATH_CALUDE_fraction_equality_implies_c_geq_one_l1610_161056

theorem fraction_equality_implies_c_geq_one
  (a b : ℕ+) (c : ℝ)
  (h_c_pos : c > 0)
  (h_eq : (a + 1) / (b + c) = b / a) :
  c ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_c_geq_one_l1610_161056


namespace NUMINAMATH_CALUDE_first_thrilling_thursday_is_correct_l1610_161089

/-- Represents a date with a day and a month -/
structure Date where
  day : Nat
  month : Nat

/-- Represents a day of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Returns true if the given date is a Thursday -/
def isThursday (d : Date) (startDate : Date) (startDay : DayOfWeek) : Bool :=
  sorry

/-- Returns true if the given date is a Thrilling Thursday -/
def isThrillingThursday (d : Date) (startDate : Date) (startDay : DayOfWeek) : Bool :=
  sorry

/-- The date of school start -/
def schoolStartDate : Date :=
  { day := 12, month := 9 }

/-- The day of the week when school starts -/
def schoolStartDay : DayOfWeek :=
  DayOfWeek.Tuesday

/-- The date of the first Thrilling Thursday -/
def firstThrillingThursday : Date :=
  { day := 26, month := 10 }

theorem first_thrilling_thursday_is_correct :
  isThrillingThursday firstThrillingThursday schoolStartDate schoolStartDay ∧
  ∀ d, d.month ≥ schoolStartDate.month ∧ 
       (d.month > schoolStartDate.month ∨ (d.month = schoolStartDate.month ∧ d.day ≥ schoolStartDate.day)) ∧
       isThrillingThursday d schoolStartDate schoolStartDay →
       (d.month > firstThrillingThursday.month ∨ 
        (d.month = firstThrillingThursday.month ∧ d.day ≥ firstThrillingThursday.day)) :=
  sorry

end NUMINAMATH_CALUDE_first_thrilling_thursday_is_correct_l1610_161089


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1610_161047

theorem geometric_sequence_problem (b : ℝ) (h1 : b > 0) :
  (∃ r : ℝ, 36 * r = b ∧ b * r = 2 / 9) → b = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1610_161047


namespace NUMINAMATH_CALUDE_ab_plus_cd_value_l1610_161003

theorem ab_plus_cd_value (a b c d : ℝ) 
  (eq1 : a + b + c = 5)
  (eq2 : a + b + d = -3)
  (eq3 : a + c + d = 10)
  (eq4 : b + c + d = -1) :
  a * b + c * d = -346 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ab_plus_cd_value_l1610_161003


namespace NUMINAMATH_CALUDE_little_red_final_score_l1610_161097

/-- Calculates the final score for the "Sunshine Sports" competition --/
def final_score (running_score fancy_jump_rope_score jump_rope_score : ℝ)
  (running_weight fancy_jump_rope_weight jump_rope_weight : ℝ) : ℝ :=
  running_score * running_weight +
  fancy_jump_rope_score * fancy_jump_rope_weight +
  jump_rope_score * jump_rope_weight

/-- Theorem stating that Little Red's final score is 83 --/
theorem little_red_final_score :
  final_score 90 80 70 0.5 0.3 0.2 = 83 := by
  sorry

#eval final_score 90 80 70 0.5 0.3 0.2

end NUMINAMATH_CALUDE_little_red_final_score_l1610_161097


namespace NUMINAMATH_CALUDE_kitty_vacuuming_time_l1610_161014

/-- Represents the weekly cleaning routine for a living room -/
structure LivingRoomCleaning where
  pickup_time : ℕ
  window_time : ℕ
  dusting_time : ℕ
  total_time_4weeks : ℕ

/-- Calculates the time spent vacuuming per week -/
def vacuuming_time_per_week (cleaning : LivingRoomCleaning) : ℕ :=
  let other_tasks_time := cleaning.pickup_time + cleaning.window_time + cleaning.dusting_time
  let total_other_tasks_4weeks := other_tasks_time * 4
  let total_vacuuming_4weeks := cleaning.total_time_4weeks - total_other_tasks_4weeks
  total_vacuuming_4weeks / 4

/-- Theorem stating that Kitty spends 20 minutes vacuuming per week -/
theorem kitty_vacuuming_time (cleaning : LivingRoomCleaning)
    (h1 : cleaning.pickup_time = 5)
    (h2 : cleaning.window_time = 15)
    (h3 : cleaning.dusting_time = 10)
    (h4 : cleaning.total_time_4weeks = 200) :
    vacuuming_time_per_week cleaning = 20 := by
  sorry

end NUMINAMATH_CALUDE_kitty_vacuuming_time_l1610_161014


namespace NUMINAMATH_CALUDE_arccos_cos_seven_l1610_161048

theorem arccos_cos_seven : Real.arccos (Real.cos 7) = 7 - 2 * Real.pi := by sorry

end NUMINAMATH_CALUDE_arccos_cos_seven_l1610_161048


namespace NUMINAMATH_CALUDE_m_intersect_n_equals_open_interval_l1610_161065

-- Define set M
def M : Set ℝ := {x : ℝ | x^2 + 5*x - 14 < 0}

-- Define set N
def N : Set ℝ := {x : ℝ | 1 < x ∧ x < 4}

-- Theorem statement
theorem m_intersect_n_equals_open_interval :
  M ∩ N = Set.Ioo 1 2 := by sorry

end NUMINAMATH_CALUDE_m_intersect_n_equals_open_interval_l1610_161065


namespace NUMINAMATH_CALUDE_polynomial_with_triple_roots_l1610_161022

def p : ℝ → ℝ := fun x ↦ 12 * x^5 - 30 * x^4 + 20 * x^3 - 1

theorem polynomial_with_triple_roots :
  (∀ x : ℝ, (∃ q : ℝ → ℝ, p x + 1 = x^3 * q x)) ∧
  (∀ x : ℝ, (∃ r : ℝ → ℝ, p x - 1 = (x - 1)^3 * r x)) →
  ∀ x : ℝ, p x = 12 * x^5 - 30 * x^4 + 20 * x^3 - 1 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_with_triple_roots_l1610_161022


namespace NUMINAMATH_CALUDE_abs_inequality_equivalence_l1610_161085

theorem abs_inequality_equivalence (x : ℝ) :
  (1 ≤ |x - 2| ∧ |x - 2| ≤ 7) ↔ ((-5 ≤ x ∧ x ≤ 1) ∨ (3 ≤ x ∧ x ≤ 9)) :=
sorry

end NUMINAMATH_CALUDE_abs_inequality_equivalence_l1610_161085


namespace NUMINAMATH_CALUDE_green_blue_difference_after_two_borders_l1610_161072

/-- Calculates the number of tiles in a border of a hexagonal figure -/
def border_tiles (side_length : ℕ) : ℕ := 6 * side_length

/-- Represents a hexagonal figure with blue and green tiles -/
structure HexFigure where
  blue_tiles : ℕ
  green_tiles : ℕ

/-- Adds a border of green tiles to a hexagonal figure -/
def add_border (fig : HexFigure) (border_size : ℕ) : HexFigure :=
  { blue_tiles := fig.blue_tiles,
    green_tiles := fig.green_tiles + border_tiles border_size }

theorem green_blue_difference_after_two_borders :
  let initial_figure : HexFigure := { blue_tiles := 14, green_tiles := 8 }
  let first_border := add_border initial_figure 3
  let second_border := add_border first_border 5
  second_border.green_tiles - second_border.blue_tiles = 42 := by
  sorry

end NUMINAMATH_CALUDE_green_blue_difference_after_two_borders_l1610_161072


namespace NUMINAMATH_CALUDE_min_cost_29_disks_l1610_161077

/-- Represents the cost of a package of disks -/
structure Package where
  quantity : Nat
  price : Nat

/-- Calculates the minimum cost to buy at least n disks given a list of packages -/
def minCost (packages : List Package) (n : Nat) : Nat :=
  sorry

/-- The available packages -/
def availablePackages : List Package :=
  [{ quantity := 1, price := 20 },
   { quantity := 10, price := 111 },
   { quantity := 25, price := 265 }]

theorem min_cost_29_disks :
  minCost availablePackages 29 = 333 :=
sorry

end NUMINAMATH_CALUDE_min_cost_29_disks_l1610_161077


namespace NUMINAMATH_CALUDE_ratio_squares_equality_l1610_161074

theorem ratio_squares_equality : (1625^2 - 1612^2) / (1631^2 - 1606^2) = 13 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ratio_squares_equality_l1610_161074


namespace NUMINAMATH_CALUDE_partnership_profit_l1610_161087

/-- Represents the investment and profit distribution in a partnership --/
structure Partnership where
  /-- A's investment ratio relative to B --/
  a_ratio : ℚ
  /-- B's investment ratio relative to C --/
  b_ratio : ℚ
  /-- B's share of the profit --/
  b_share : ℕ

/-- Calculates the total profit given the partnership details --/
def calculate_total_profit (p : Partnership) : ℕ :=
  sorry

/-- Theorem stating that given the specified partnership details, the total profit is 7700 --/
theorem partnership_profit (p : Partnership) 
  (h1 : p.a_ratio = 3)
  (h2 : p.b_ratio = 2/3)
  (h3 : p.b_share = 1400) :
  calculate_total_profit p = 7700 :=
sorry

end NUMINAMATH_CALUDE_partnership_profit_l1610_161087


namespace NUMINAMATH_CALUDE_geometric_sequence_condition_l1610_161050

/-- The sum of the first n terms of the sequence a_n -/
def S (n : ℕ) (c : ℝ) : ℝ := 2^n + c

/-- The n-th term of the sequence a_n -/
def a (n : ℕ) (c : ℝ) : ℝ := S n c - S (n-1) c

/-- Predicate to check if a sequence is geometric -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, n ≥ 2 → a (n+1) = r * a n

theorem geometric_sequence_condition (c : ℝ) :
  is_geometric_sequence (a · c) ↔ c = -1 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_condition_l1610_161050


namespace NUMINAMATH_CALUDE_log_ratio_squared_l1610_161024

theorem log_ratio_squared (x y : ℝ) 
  (hx_pos : x > 0) (hy_pos : y > 0) 
  (hx_neq_one : x ≠ 1) (hy_neq_one : y ≠ 1) 
  (h_log : Real.log x / Real.log 3 = Real.log 81 / Real.log y) 
  (h_prod : x * y^2 = 243) : 
  (Real.log (x/y) / Real.log 3)^2 = 49/36 := by
sorry

end NUMINAMATH_CALUDE_log_ratio_squared_l1610_161024


namespace NUMINAMATH_CALUDE_part_one_part_two_l1610_161033

-- Define the sets A, B, and C
def A (a : ℝ) : Set ℝ := {x | x^2 - a*x + a^2 - 12 = 0}
def B : Set ℝ := {x | x^2 - 2*x - 8 = 0}
def C (m : ℝ) : Set ℝ := {x | m*x + 1 = 0}

-- Part I: Prove that if A = B, then a = 2
theorem part_one : A 2 = B :=
sorry

-- Part II: Prove that if B ∪ C = B, then m ∈ {-1/4, 0, 1/2}
theorem part_two (m : ℝ) : B ∪ C m = B → m ∈ ({-1/4, 0, 1/2} : Set ℝ) :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1610_161033


namespace NUMINAMATH_CALUDE_units_digit_of_product_l1610_161053

/-- The units digit of a natural number -/
def units_digit (n : ℕ) : ℕ := n % 10

/-- The property that the units digit of any power of 5 is 5 -/
axiom units_digit_power_of_five (k : ℕ) : units_digit (5^k) = 5

/-- The main theorem: The units digit of 5^11 * 2^3 is 0 -/
theorem units_digit_of_product : units_digit (5^11 * 2^3) = 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_product_l1610_161053


namespace NUMINAMATH_CALUDE_smallest_value_theorem_l1610_161063

theorem smallest_value_theorem (a : ℝ) (h : 8 * a^3 + 6 * a^2 + 7 * a + 5 = 4) :
  ∃ (min_val : ℝ), min_val = (1 : ℝ) / 2 ∧ ∀ (x : ℝ), 8 * x^3 + 6 * x^2 + 7 * x + 5 = 4 → 3 * x + 2 ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_theorem_l1610_161063


namespace NUMINAMATH_CALUDE_square_units_digit_l1610_161044

theorem square_units_digit (n : ℤ) : 
  (n^2 / 10) % 10 = 7 → n^2 % 10 = 6 := by sorry

end NUMINAMATH_CALUDE_square_units_digit_l1610_161044


namespace NUMINAMATH_CALUDE_carries_cucumber_harvest_l1610_161039

/-- Represents the dimensions of a rectangular garden -/
structure GardenDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the expected cucumber harvest given garden dimensions and planting parameters -/
def expected_harvest (garden : GardenDimensions) (plants_per_sqft : ℝ) (cucumbers_per_plant : ℝ) : ℝ :=
  garden.length * garden.width * plants_per_sqft * cucumbers_per_plant

/-- Theorem stating that Carrie's garden will yield 9000 cucumbers -/
theorem carries_cucumber_harvest :
  let garden := GardenDimensions.mk 10 12
  let plants_per_sqft := 5
  let cucumbers_per_plant := 15
  expected_harvest garden plants_per_sqft cucumbers_per_plant = 9000 := by
  sorry


end NUMINAMATH_CALUDE_carries_cucumber_harvest_l1610_161039


namespace NUMINAMATH_CALUDE_symmetry_yOz_correct_l1610_161004

/-- Given a point (x, y, z) in 3D space, this function returns its symmetrical point
    with respect to the yOz plane -/
def symmetry_yOz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-x, y, z)

theorem symmetry_yOz_correct :
  symmetry_yOz (1, 2, 1) = (-1, 2, 1) := by
  sorry

end NUMINAMATH_CALUDE_symmetry_yOz_correct_l1610_161004


namespace NUMINAMATH_CALUDE_range_of_a_l1610_161012

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc (-2) 2 → x^2 + a*x + 3 ≥ a) → 
  a ∈ Set.Icc (-7) 2 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l1610_161012


namespace NUMINAMATH_CALUDE_valid_selection_count_l1610_161088

/-- Represents a dad in the TV show -/
structure Dad :=
  (id : Nat)

/-- Represents a kid in the TV show -/
structure Kid :=
  (id : Nat)
  (isGirl : Bool)
  (dad : Dad)

/-- Represents the selection of one dad and three kids -/
structure Selection :=
  (selectedDad : Dad)
  (selectedKids : Finset Kid)

/-- The set of all dads -/
def allDads : Finset Dad := sorry

/-- The set of all kids -/
def allKids : Finset Kid := sorry

/-- Kimi is a boy -/
def kimi : Kid := sorry

/-- Stone is a boy -/
def stone : Kid := sorry

/-- Predicate to check if a selection is valid -/
def isValidSelection (s : Selection) : Prop :=
  s.selectedKids.card = 3 ∧
  (∃ k ∈ s.selectedKids, k.isGirl) ∧
  (kimi ∈ s.selectedKids ↔ kimi.dad = s.selectedDad) ∧
  (stone ∈ s.selectedKids ↔ stone.dad ≠ s.selectedDad)

/-- The set of all possible valid selections -/
def allValidSelections : Finset Selection :=
  sorry

theorem valid_selection_count :
  allValidSelections.card = 12 :=
sorry

end NUMINAMATH_CALUDE_valid_selection_count_l1610_161088


namespace NUMINAMATH_CALUDE_quadratic_form_nonnegative_l1610_161078

theorem quadratic_form_nonnegative
  (a b c x y z : ℝ)
  (sum_xyz : x + y + z = 0)
  (sum_abc_nonneg : a + b + c ≥ 0)
  (sum_products_nonneg : a * b + b * c + c * a ≥ 0) :
  a * x^2 + b * y^2 + c * z^2 ≥ 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_form_nonnegative_l1610_161078


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l1610_161020

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, x₁^2 - 1 = 0 ∧ x₂^2 - 1 = 0 ∧ x₁ = 1 ∧ x₂ = -1) ∧
  (∃ x₁ x₂ : ℝ, x₁^2 - 3*x₁ + 1 = 0 ∧ x₂^2 - 3*x₂ + 1 = 0 ∧
    x₁ = (3 + Real.sqrt 5) / 2 ∧ x₂ = (3 - Real.sqrt 5) / 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l1610_161020


namespace NUMINAMATH_CALUDE_cricket_innings_problem_l1610_161021

theorem cricket_innings_problem (initial_average : ℝ) (next_innings_runs : ℝ) (average_increase : ℝ) :
  initial_average = 32 →
  next_innings_runs = 137 →
  average_increase = 5 →
  ∃ n : ℕ, (n : ℝ) * initial_average + next_innings_runs = (n + 1 : ℝ) * (initial_average + average_increase) ∧ n = 20 :=
by sorry

end NUMINAMATH_CALUDE_cricket_innings_problem_l1610_161021


namespace NUMINAMATH_CALUDE_train_speed_l1610_161092

/-- The speed of a train crossing a bridge -/
theorem train_speed (train_length : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 100 →
  bridge_length = 300 →
  crossing_time = 36 →
  ∃ (speed : ℝ), abs (speed - (train_length + bridge_length) / crossing_time) < 0.01 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_l1610_161092


namespace NUMINAMATH_CALUDE_quadratic_root_implies_u_value_l1610_161098

theorem quadratic_root_implies_u_value (u : ℝ) : 
  (3 * (((-15 - Real.sqrt 205) / 6) ^ 2) + 15 * ((-15 - Real.sqrt 205) / 6) + u = 0) → 
  u = 5/3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_u_value_l1610_161098


namespace NUMINAMATH_CALUDE_reflected_rays_angle_l1610_161049

theorem reflected_rays_angle 
  (α β : Real) 
  (h_α : 0 < α ∧ α < π/2) 
  (h_β : 0 < β ∧ β < π/2) : 
  ∃ θ : Real, θ = Real.arccos (1 - 2 * Real.sin α ^ 2 * Real.sin β ^ 2) := by
sorry

end NUMINAMATH_CALUDE_reflected_rays_angle_l1610_161049


namespace NUMINAMATH_CALUDE_electronic_items_loss_percentage_l1610_161010

/-- Calculate the overall loss percentage for three electronic items -/
theorem electronic_items_loss_percentage :
  let cost_prices : List ℚ := [1500, 2500, 800]
  let sale_prices : List ℚ := [1275, 2300, 700]
  let total_cost := cost_prices.sum
  let total_sale := sale_prices.sum
  let loss := total_cost - total_sale
  let loss_percentage := (loss / total_cost) * 100
  loss_percentage = 10.9375 := by
  sorry

end NUMINAMATH_CALUDE_electronic_items_loss_percentage_l1610_161010


namespace NUMINAMATH_CALUDE_share_calculation_l1610_161019

theorem share_calculation (total_amount : ℕ) (ratio_parts : List ℕ) : 
  total_amount = 4800 → 
  ratio_parts = [2, 4, 6] → 
  (total_amount / (ratio_parts.sum)) * (ratio_parts.head!) = 800 := by
  sorry

end NUMINAMATH_CALUDE_share_calculation_l1610_161019


namespace NUMINAMATH_CALUDE_pentagonal_prism_sum_l1610_161009

/-- Definition of a pentagonal prism -/
structure PentagonalPrism where
  bases : ℕ := 2
  connecting_faces : ℕ := 5
  edges_per_base : ℕ := 5
  vertices_per_base : ℕ := 5

/-- Theorem: The sum of faces, edges, and vertices of a pentagonal prism is 32 -/
theorem pentagonal_prism_sum (p : PentagonalPrism) : 
  (p.bases + p.connecting_faces) + 
  (p.edges_per_base * 2 + p.edges_per_base) + 
  (p.vertices_per_base * 2) = 32 := by
  sorry

#check pentagonal_prism_sum

end NUMINAMATH_CALUDE_pentagonal_prism_sum_l1610_161009


namespace NUMINAMATH_CALUDE_four_plus_five_result_l1610_161034

/-- Define the sequence operation for two consecutive integers -/
def seqOperation (a b : ℕ) : ℕ := (a + b)^2 + 1

/-- Theorem stating that 4 + 5 results in 82 in the given sequence -/
theorem four_plus_five_result :
  seqOperation 4 5 = 82 :=
by sorry

end NUMINAMATH_CALUDE_four_plus_five_result_l1610_161034


namespace NUMINAMATH_CALUDE_six_digit_divisibility_l1610_161066

theorem six_digit_divisibility (a b c : ℕ) 
  (h1 : a ≥ 1 ∧ a ≤ 9) 
  (h2 : b ≥ 0 ∧ b ≤ 9) 
  (h3 : c ≥ 0 ∧ c ≤ 9) : 
  (100000 * a + 10000 * b + 1000 * c + 100 * a + 10 * b + c) % 1001 = 0 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_divisibility_l1610_161066


namespace NUMINAMATH_CALUDE_course_selection_count_l1610_161040

/-- The number of different course selection schemes for 3 students choosing from 3 elective courses -/
def course_selection_schemes : ℕ := 18

/-- The number of elective courses -/
def num_courses : ℕ := 3

/-- The number of students -/
def num_students : ℕ := 3

/-- Proposition that each student chooses only one course -/
axiom one_course_per_student : True

/-- Proposition that exactly one course has no students -/
axiom one_empty_course : True

/-- Theorem stating that the number of different course selection schemes is 18 -/
theorem course_selection_count : course_selection_schemes = 18 := by
  sorry

end NUMINAMATH_CALUDE_course_selection_count_l1610_161040


namespace NUMINAMATH_CALUDE_mod_equivalence_proof_l1610_161057

theorem mod_equivalence_proof : ∃ (n : ℤ), 0 ≤ n ∧ n ≤ 7 ∧ n ≡ -4573 [ZMOD 8] → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_proof_l1610_161057


namespace NUMINAMATH_CALUDE_kim_cousins_count_l1610_161002

theorem kim_cousins_count (gum_per_cousin : ℕ) (total_gum : ℕ) (h1 : gum_per_cousin = 5) (h2 : total_gum = 20) :
  total_gum / gum_per_cousin = 4 := by
  sorry

end NUMINAMATH_CALUDE_kim_cousins_count_l1610_161002


namespace NUMINAMATH_CALUDE_fence_painting_rate_l1610_161095

theorem fence_painting_rate (num_fences : ℕ) (fence_length : ℕ) (total_earnings : ℚ) :
  num_fences = 50 →
  fence_length = 500 →
  total_earnings = 5000 →
  total_earnings / (num_fences * fence_length : ℚ) = 0.20 := by
  sorry

end NUMINAMATH_CALUDE_fence_painting_rate_l1610_161095


namespace NUMINAMATH_CALUDE_travel_agency_choice_l1610_161061

-- Define the cost functions for each travel agency
def costA (x : ℝ) : ℝ := 2000 * x * 0.75
def costB (x : ℝ) : ℝ := 2000 * (x - 1) * 0.8

-- Define the theorem
theorem travel_agency_choice (x : ℝ) (h1 : 10 ≤ x) (h2 : x ≤ 25) :
  (10 ≤ x ∧ x < 16 → costB x < costA x) ∧
  (x = 16 → costA x = costB x) ∧
  (16 < x ∧ x ≤ 25 → costA x < costB x) :=
sorry

end NUMINAMATH_CALUDE_travel_agency_choice_l1610_161061


namespace NUMINAMATH_CALUDE_garden_walkway_area_l1610_161035

/-- Represents the configuration of a garden with flower beds and walkways -/
structure Garden where
  bed_width : ℕ
  bed_height : ℕ
  walkway_width : ℕ
  rows : ℕ
  beds_in_first_row : ℕ
  beds_in_other_rows : ℕ

/-- Calculates the total area of walkways in the garden -/
def walkway_area (g : Garden) : ℕ :=
  let total_width := g.bed_width * g.beds_in_first_row + (g.beds_in_first_row + 1) * g.walkway_width
  let total_height := g.bed_height * g.rows + (g.rows + 1) * g.walkway_width
  let total_area := total_width * total_height
  let bed_area := (g.bed_width * g.bed_height) * (g.beds_in_first_row + (g.rows - 1) * g.beds_in_other_rows)
  total_area - bed_area

/-- The theorem stating that for the given garden configuration, the walkway area is 488 square feet -/
theorem garden_walkway_area :
  let g : Garden := {
    bed_width := 8,
    bed_height := 3,
    walkway_width := 2,
    rows := 4,
    beds_in_first_row := 3,
    beds_in_other_rows := 2
  }
  walkway_area g = 488 := by sorry

end NUMINAMATH_CALUDE_garden_walkway_area_l1610_161035


namespace NUMINAMATH_CALUDE_area_of_triangle_area_value_l1610_161093

theorem area_of_triangle : ℝ → Prop :=
  fun area =>
    ∃ (line1 line2 : ℝ → ℝ → Prop) (x_axis : ℝ → ℝ → Prop),
      (∀ x y, line1 x y ↔ y = x) ∧
      (∀ x y, line2 x y ↔ x = -7) ∧
      (∀ x y, x_axis x y ↔ y = 0) ∧
      (∃ x y, line1 x y ∧ line2 x y) ∧
      (let base := 7
       let height := 7
       area = (1/2) * base * height)

theorem area_value : area_of_triangle 24.5 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_area_value_l1610_161093


namespace NUMINAMATH_CALUDE_cubic_integer_bound_l1610_161027

theorem cubic_integer_bound (a b c d : ℝ) (ha : a > 4/3) :
  ∃ (S : Finset ℤ), (∀ x : ℤ, x ∈ S ↔ |a * x^3 + b * x^2 + c * x + d| ≤ 1) ∧ Finset.card S ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_integer_bound_l1610_161027


namespace NUMINAMATH_CALUDE_average_y_value_l1610_161017

def linear_regression (x : ℝ) : ℝ := 1.5 * x + 45

def x_values : List ℝ := [1, 7, 5, 13, 19]

theorem average_y_value (x_avg : ℝ) (h : x_avg = (List.sum x_values) / (List.length x_values)) :
  linear_regression x_avg = 58.5 := by
  sorry

end NUMINAMATH_CALUDE_average_y_value_l1610_161017


namespace NUMINAMATH_CALUDE_sequence_properties_l1610_161026

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, b (n + 1) = b n * q

theorem sequence_properties
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (ha : arithmetic_sequence a)
  (hb : geometric_sequence b)
  (ha_cond : 2 * a 5 - a 3 = 3)
  (hb_2 : b 2 = 1)
  (hb_4 : b 4 = 4) :
  a 7 = 3 ∧ b 6 = 16 ∧ (∃ q : ℝ, (q = 2 ∨ q = -2) ∧ ∀ n : ℕ, b (n + 1) = b n * q) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l1610_161026


namespace NUMINAMATH_CALUDE_cyclic_sum_minimum_l1610_161036

theorem cyclic_sum_minimum (a b c d : ℝ) 
  (non_neg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) 
  (sum_eq_four : a + b + c + d = 4) : 
  ((b + 3) / (a^2 + 4) + 
   (c + 3) / (b^2 + 4) + 
   (d + 3) / (c^2 + 4) + 
   (a + 3) / (d^2 + 4)) ≥ 3 ∧ 
  ∃ a b c d, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ 
    a + b + c + d = 4 ∧ 
    ((b + 3) / (a^2 + 4) + 
     (c + 3) / (b^2 + 4) + 
     (d + 3) / (c^2 + 4) + 
     (a + 3) / (d^2 + 4)) = 3 :=
by sorry

end NUMINAMATH_CALUDE_cyclic_sum_minimum_l1610_161036


namespace NUMINAMATH_CALUDE_max_triangle_side_l1610_161062

theorem max_triangle_side (a : ℕ) : 
  (3 + 8 > a ∧ 3 + a > 8 ∧ 8 + a > 3) → a ≤ 10 :=
by sorry

end NUMINAMATH_CALUDE_max_triangle_side_l1610_161062


namespace NUMINAMATH_CALUDE_base_six_addition_l1610_161067

/-- Given a base-6 addition 4AB₆ + 41₆ = 53A₆, prove that A + B = 9 in base 10 -/
theorem base_six_addition (A B : ℕ) : 
  (4 * 6^2 + A * 6 + B) + (4 * 6 + 1) = 5 * 6^2 + 3 * 6 + A → A + B = 9 :=
by sorry

end NUMINAMATH_CALUDE_base_six_addition_l1610_161067


namespace NUMINAMATH_CALUDE_product_of_multiples_of_three_l1610_161031

theorem product_of_multiples_of_three : ∃ (a b : ℕ), 
  a = 22 * 3 ∧ 
  b = 23 * 3 ∧ 
  a < 100 ∧ 
  b < 100 ∧ 
  a * b = 4554 := by
  sorry

end NUMINAMATH_CALUDE_product_of_multiples_of_three_l1610_161031


namespace NUMINAMATH_CALUDE_t_shape_area_is_12_l1610_161075

def square_area (side : ℝ) : ℝ := side * side

def t_shape_area (outer_side : ℝ) (inner_side1 : ℝ) (inner_side2 : ℝ) : ℝ :=
  square_area outer_side - (2 * square_area inner_side1 + square_area inner_side2)

theorem t_shape_area_is_12 :
  t_shape_area 6 2 4 = 12 := by sorry

end NUMINAMATH_CALUDE_t_shape_area_is_12_l1610_161075


namespace NUMINAMATH_CALUDE_quadratic_function_and_area_l1610_161006

-- Define the quadratic function f
def f : ℝ → ℝ := fun x ↦ x^2 + 2*x + 1

-- Theorem statement
theorem quadratic_function_and_area :
  (∀ x, (deriv f) x = 2*x + 2) ∧ 
  (∃! x, f x = 0) ∧
  (∫ x in (-3)..0, ((-x^2 - 4*x + 1) - f x)) = 9 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_and_area_l1610_161006


namespace NUMINAMATH_CALUDE_used_car_seller_problem_l1610_161076

theorem used_car_seller_problem (num_clients : ℕ) (cars_per_client : ℕ) (selections_per_car : ℕ) 
  (h1 : num_clients = 24)
  (h2 : cars_per_client = 2)
  (h3 : selections_per_car = 3) :
  (num_clients * cars_per_client) / selections_per_car = 16 := by
  sorry

end NUMINAMATH_CALUDE_used_car_seller_problem_l1610_161076


namespace NUMINAMATH_CALUDE_renata_final_balance_l1610_161018

/-- Represents Renata's financial transactions throughout the day -/
def renata_transactions : ℤ → ℤ
| 0 => 10                  -- Initial amount
| 1 => -4                  -- Charity ticket donation
| 2 => 90                  -- Charity draw winnings
| 3 => -50                 -- First slot machine loss
| 4 => -10                 -- Second slot machine loss
| 5 => -5                  -- Third slot machine loss
| 6 => -1                  -- Water bottle purchase
| 7 => -1                  -- Lottery ticket purchase
| 8 => 65                  -- Lottery winnings
| _ => 0                   -- No more transactions

/-- The final balance after all transactions -/
def final_balance : ℤ := (List.range 9).foldl (· + renata_transactions ·) 0

/-- Theorem stating that Renata's final balance is $94 -/
theorem renata_final_balance : final_balance = 94 := by
  sorry

end NUMINAMATH_CALUDE_renata_final_balance_l1610_161018
