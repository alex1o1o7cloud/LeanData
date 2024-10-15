import Mathlib

namespace NUMINAMATH_CALUDE_max_rectangle_division_ratio_l1920_192066

/-- The number of ways to divide a rectangle with side lengths a and b into smaller rectangles with integer side lengths -/
def D (a b : ℕ+) : ℕ := sorry

/-- The theorem stating that D(a,b)/(2(a+b)) ≤ 3/8 for all positive integers a and b, 
    with equality if and only if a = b = 2 -/
theorem max_rectangle_division_ratio 
  (a b : ℕ+) : 
  (D a b : ℚ) / (2 * ((a:ℚ) + (b:ℚ))) ≤ 3/8 ∧ 
  ((D a b : ℚ) / (2 * ((a:ℚ) + (b:ℚ))) = 3/8 ↔ a = 2 ∧ b = 2) := by
  sorry

end NUMINAMATH_CALUDE_max_rectangle_division_ratio_l1920_192066


namespace NUMINAMATH_CALUDE_triangle_with_prime_angles_exists_l1920_192095

-- Define what it means for a number to be prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

-- Theorem statement
theorem triangle_with_prime_angles_exists : ∃ p q r : ℕ, 
  isPrime p ∧ isPrime q ∧ isPrime r ∧ p + q + r = 180 := by
  sorry

end NUMINAMATH_CALUDE_triangle_with_prime_angles_exists_l1920_192095


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l1920_192035

theorem fractional_equation_solution :
  ∀ x : ℝ, (4 / x = 2 / (x + 1)) ↔ (x = -2) :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l1920_192035


namespace NUMINAMATH_CALUDE_estate_area_calculation_l1920_192087

/-- Represents the side length of the square on the map in inches -/
def map_side_length : ℝ := 12

/-- Represents the scale of the map in miles per inch -/
def map_scale : ℝ := 100

/-- Calculates the actual side length of the estate in miles -/
def actual_side_length : ℝ := map_side_length * map_scale

/-- Calculates the actual area of the estate in square miles -/
def actual_area : ℝ := actual_side_length ^ 2

/-- Theorem stating that the actual area of the estate is 1440000 square miles -/
theorem estate_area_calculation : actual_area = 1440000 := by
  sorry

end NUMINAMATH_CALUDE_estate_area_calculation_l1920_192087


namespace NUMINAMATH_CALUDE_roberto_outfits_l1920_192076

/-- Represents the number of trousers Roberto has -/
def num_trousers : ℕ := 4

/-- Represents the number of shirts Roberto has -/
def num_shirts : ℕ := 7

/-- Represents the number of jackets Roberto has -/
def num_jackets : ℕ := 5

/-- Represents the number of hat options Roberto has (wear or not wear) -/
def num_hat_options : ℕ := 2

/-- Calculates the total number of outfit combinations -/
def total_outfits : ℕ := num_trousers * num_shirts * num_jackets * num_hat_options

/-- Theorem stating that the total number of outfits Roberto can create is 280 -/
theorem roberto_outfits : total_outfits = 280 := by
  sorry

end NUMINAMATH_CALUDE_roberto_outfits_l1920_192076


namespace NUMINAMATH_CALUDE_equality_or_opposite_equality_l1920_192062

theorem equality_or_opposite_equality (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  a^2 + b^3/a = b^2 + a^3/b → a = b ∨ a = -b := by
  sorry

end NUMINAMATH_CALUDE_equality_or_opposite_equality_l1920_192062


namespace NUMINAMATH_CALUDE_rectangle_dimension_change_l1920_192030

theorem rectangle_dimension_change (original_length original_width : ℝ) 
  (new_length new_width : ℝ) (h_positive : original_length > 0 ∧ original_width > 0) :
  new_width = 1.5 * original_width ∧ 
  original_length * original_width = new_length * new_width →
  (original_length - new_length) / original_length = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_rectangle_dimension_change_l1920_192030


namespace NUMINAMATH_CALUDE_updated_mean_after_decrement_l1920_192021

theorem updated_mean_after_decrement (n : ℕ) (original_mean : ℚ) (decrement : ℚ) :
  n > 0 →
  n = 50 →
  original_mean = 200 →
  decrement = 9 →
  (n : ℚ) * original_mean - n * decrement = n * 191 := by
  sorry

end NUMINAMATH_CALUDE_updated_mean_after_decrement_l1920_192021


namespace NUMINAMATH_CALUDE_ben_age_l1920_192000

def Ages : List ℕ := [6, 8, 10, 12, 14]

def ParkPair (a b : ℕ) : Prop := a + b = 18 ∧ a ∈ Ages ∧ b ∈ Ages ∧ a ≠ b

def LibraryPair (a b : ℕ) : Prop := a + b < 20 ∧ a ∈ Ages ∧ b ∈ Ages ∧ a ≠ b

def RemainingAges (park1 park2 lib1 lib2 : ℕ) : List ℕ :=
  Ages.filter (λ x => x ∉ [park1, park2, lib1, lib2])

theorem ben_age :
  ∀ park1 park2 lib1 lib2 youngest_home,
    ParkPair park1 park2 →
    LibraryPair lib1 lib2 →
    youngest_home = (RemainingAges park1 park2 lib1 lib2).minimum →
    10 ∈ RemainingAges park1 park2 lib1 lib2 →
    10 ≠ youngest_home →
    10 = (RemainingAges park1 park2 lib1 lib2).maximum :=
by sorry

end NUMINAMATH_CALUDE_ben_age_l1920_192000


namespace NUMINAMATH_CALUDE_unique_consecutive_digit_square_swap_l1920_192029

/-- A function that checks if a number is formed by four consecutive digits -/
def is_consecutive_digits (n : ℕ) : Prop :=
  ∃ a : ℕ, n = 1000 * a + 100 * (a + 1) + 10 * (a + 2) + (a + 3)

/-- A function that swaps the first two digits of a four-digit number -/
def swap_first_two_digits (n : ℕ) : ℕ :=
  let d1 := n / 1000
  let d2 := (n / 100) % 10
  let last_two := n % 100
  1000 * d2 + 100 * d1 + last_two

/-- The main theorem stating that 3456 is the only number satisfying the conditions -/
theorem unique_consecutive_digit_square_swap :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 →
    (is_consecutive_digits n ∧ ∃ m : ℕ, swap_first_two_digits n = m ^ 2) ↔ n = 3456 :=
sorry

end NUMINAMATH_CALUDE_unique_consecutive_digit_square_swap_l1920_192029


namespace NUMINAMATH_CALUDE_subset_condition_disjoint_condition_l1920_192091

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x : ℝ | m + 1 ≤ x ∧ x ≤ 2*m - 1}

-- Theorem 1: B ⊆ A ⇔ m ∈ (-∞, 3]
theorem subset_condition (m : ℝ) : B m ⊆ A ↔ m ≤ 3 := by sorry

-- Theorem 2: A ∩ B = ∅ ⇔ m ∈ (-∞, 2) ∪ (4, +∞)
theorem disjoint_condition (m : ℝ) : A ∩ B m = ∅ ↔ m < 2 ∨ m > 4 := by sorry

end NUMINAMATH_CALUDE_subset_condition_disjoint_condition_l1920_192091


namespace NUMINAMATH_CALUDE_pedestrian_speed_problem_l1920_192078

/-- The problem of two pedestrians traveling between points A and B -/
theorem pedestrian_speed_problem (x : ℝ) :
  x > 0 →  -- The speed must be positive
  (11 / (x + 1.5) + 8 / x ≥ 12 / (x + 2) + 2) →  -- The inequality from the problem
  x ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_pedestrian_speed_problem_l1920_192078


namespace NUMINAMATH_CALUDE_chef_inventory_solution_l1920_192040

def chef_inventory (initial_apples initial_flour initial_sugar initial_butter : ℕ) : Prop :=
  let used_apples : ℕ := 15
  let used_flour : ℕ := 6
  let used_sugar : ℕ := 14  -- 10 initially + 4 from newly bought
  let used_butter : ℕ := 3
  let remaining_apples : ℕ := 4
  let remaining_flour : ℕ := 3
  let remaining_sugar : ℕ := 13
  let remaining_butter : ℕ := 2
  let given_away_apples : ℕ := 2
  (initial_apples = used_apples + given_away_apples + remaining_apples) ∧
  (initial_flour = 2 * (used_flour + remaining_flour)) ∧
  (initial_sugar = used_sugar + remaining_sugar) ∧
  (initial_butter = used_butter + remaining_butter)

theorem chef_inventory_solution :
  ∃ (initial_apples initial_flour initial_sugar initial_butter : ℕ),
    chef_inventory initial_apples initial_flour initial_sugar initial_butter ∧
    initial_apples = 21 ∧
    initial_flour = 18 ∧
    initial_sugar = 27 ∧
    initial_butter = 5 ∧
    initial_apples + initial_flour + initial_sugar + initial_butter = 71 :=
by sorry

end NUMINAMATH_CALUDE_chef_inventory_solution_l1920_192040


namespace NUMINAMATH_CALUDE_georgie_enter_exit_ways_l1920_192061

/-- The number of windows in the haunted mansion -/
def num_windows : ℕ := 8

/-- The number of ways Georgie can enter and exit the mansion -/
def num_ways : ℕ := num_windows * (num_windows - 1)

/-- Theorem stating that the number of ways Georgie can enter and exit is 56 -/
theorem georgie_enter_exit_ways : num_ways = 56 := by
  sorry

end NUMINAMATH_CALUDE_georgie_enter_exit_ways_l1920_192061


namespace NUMINAMATH_CALUDE_hat_number_sum_l1920_192010

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem hat_number_sum : 
  ∀ (A B : ℕ),
  (A ≥ 2 ∧ A ≤ 49) →  -- Alice's number is between 2 and 49
  (B > 10 ∧ is_prime B) →  -- Bob's number is prime and greater than 10
  (∀ k : ℕ, k ≥ 2 ∧ k ≤ 49 → k ≠ A → ¬(k > B)) →  -- Alice can't tell who has the larger number
  (∀ k : ℕ, k ≥ 2 ∧ k ≤ 49 → k ≠ A → (k > B ∨ B > k)) →  -- Bob can tell who has the larger number
  (∃ (k : ℕ), 50 * B + A = k * k) →  -- The result is a perfect square
  A + B = 37 :=
by sorry

end NUMINAMATH_CALUDE_hat_number_sum_l1920_192010


namespace NUMINAMATH_CALUDE_xy_inequality_l1920_192047

theorem xy_inequality (x y : ℝ) (h : x^2 + y^2 - x*y = 1) :
  (-2 : ℝ) ≤ x + y ∧ x + y ≤ 2 ∧ 2/3 ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_xy_inequality_l1920_192047


namespace NUMINAMATH_CALUDE_ship_journey_distance_l1920_192057

theorem ship_journey_distance : 
  let day1_distance : ℕ := 100
  let day2_distance : ℕ := 3 * day1_distance
  let day3_distance : ℕ := day2_distance + 110
  let total_distance : ℕ := day1_distance + day2_distance + day3_distance
  total_distance = 810 := by sorry

end NUMINAMATH_CALUDE_ship_journey_distance_l1920_192057


namespace NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l1920_192023

theorem regular_polygon_exterior_angle (n : ℕ) (n_pos : 0 < n) :
  (360 : ℝ) / n = 72 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l1920_192023


namespace NUMINAMATH_CALUDE_daughter_weight_l1920_192065

/-- Represents the weights of family members -/
structure FamilyWeights where
  grandmother : ℝ
  daughter : ℝ
  child : ℝ

/-- The conditions of the family weight problem -/
def FamilyWeightProblem (w : FamilyWeights) : Prop :=
  w.grandmother + w.daughter + w.child = 110 ∧
  w.daughter + w.child = 60 ∧
  w.child = (1 / 5) * w.grandmother

/-- The theorem stating that given the conditions, the daughter's weight is 50 kg -/
theorem daughter_weight (w : FamilyWeights) : 
  FamilyWeightProblem w → w.daughter = 50 := by
  sorry


end NUMINAMATH_CALUDE_daughter_weight_l1920_192065


namespace NUMINAMATH_CALUDE_ratio_a_to_c_l1920_192020

theorem ratio_a_to_c (a b c d : ℚ) 
  (h1 : a / b = 5 / 4)
  (h2 : c / d = 4 / 3)
  (h3 : d / b = 1 / 5) :
  a / c = 75 / 16 := by
sorry

end NUMINAMATH_CALUDE_ratio_a_to_c_l1920_192020


namespace NUMINAMATH_CALUDE_latus_rectum_of_parabola_l1920_192026

/-- The latus rectum of a parabola x^2 = -2y is y = 1/2 -/
theorem latus_rectum_of_parabola (x y : ℝ) :
  x^2 = -2*y → (∃ (x₀ : ℝ), x₀^2 = -2*(1/2) ∧ x₀ ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_latus_rectum_of_parabola_l1920_192026


namespace NUMINAMATH_CALUDE_triangle_problem_l1920_192009

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
def Triangle (a b c : ℝ) (A B C : ℝ) := True

theorem triangle_problem 
  (a b c : ℝ) (A B C : ℝ) 
  (h_triangle : Triangle a b c A B C)
  (h_eq : 3 * a * Real.sin C = c * Real.cos A)
  (h_B : B = π / 4)
  (h_area : 1 / 2 * a * c * Real.sin B = 9) :
  Real.sin A = Real.sqrt 10 / 10 ∧ a = 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l1920_192009


namespace NUMINAMATH_CALUDE_baseball_cost_l1920_192022

def marbles_cost : ℚ := 9.05
def football_cost : ℚ := 4.95
def total_cost : ℚ := 20.52

theorem baseball_cost : total_cost - (marbles_cost + football_cost) = 6.52 := by
  sorry

end NUMINAMATH_CALUDE_baseball_cost_l1920_192022


namespace NUMINAMATH_CALUDE_soap_cost_l1920_192034

/-- The total cost of soap given the number of bars, weight per bar, and price per pound -/
theorem soap_cost (num_bars : ℕ) (weight_per_bar : ℝ) (price_per_pound : ℝ) :
  num_bars = 20 →
  weight_per_bar = 1.5 →
  price_per_pound = 0.5 →
  num_bars * weight_per_bar * price_per_pound = 15 := by
  sorry

end NUMINAMATH_CALUDE_soap_cost_l1920_192034


namespace NUMINAMATH_CALUDE_quadratic_rational_root_even_coefficient_l1920_192080

theorem quadratic_rational_root_even_coefficient
  (a b c : ℤ) (h_a : a ≠ 0)
  (h_root : ∃ (x : ℚ), a * x^2 + b * x + c = 0) :
  Even a ∨ Even b ∨ Even c :=
sorry

end NUMINAMATH_CALUDE_quadratic_rational_root_even_coefficient_l1920_192080


namespace NUMINAMATH_CALUDE_max_value_theorem_equality_condition_l1920_192094

theorem max_value_theorem (x : ℝ) (h : x > 0) : 2 - x - 4 / x ≤ -2 :=
sorry

theorem equality_condition (x : ℝ) (h : x > 0) : 2 - x - 4 / x = -2 ↔ x = 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_equality_condition_l1920_192094


namespace NUMINAMATH_CALUDE_lowest_class_size_class_size_120_lowest_class_size_is_120_l1920_192043

theorem lowest_class_size (n : ℕ) : n > 0 ∧ 6 ∣ n ∧ 8 ∣ n ∧ 12 ∣ n ∧ 15 ∣ n → n ≥ 120 := by
  sorry

theorem class_size_120 : 6 ∣ 120 ∧ 8 ∣ 120 ∧ 12 ∣ 120 ∧ 15 ∣ 120 := by
  sorry

theorem lowest_class_size_is_120 : ∃! n : ℕ, n > 0 ∧ 6 ∣ n ∧ 8 ∣ n ∧ 12 ∣ n ∧ 15 ∣ n ∧ ∀ m : ℕ, (m > 0 ∧ 6 ∣ m ∧ 8 ∣ m ∧ 12 ∣ m ∧ 15 ∣ m) → n ≤ m := by
  sorry

end NUMINAMATH_CALUDE_lowest_class_size_class_size_120_lowest_class_size_is_120_l1920_192043


namespace NUMINAMATH_CALUDE_f_properties_l1920_192001

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 1 / (2^x + 1)

theorem f_properties (a : ℝ) :
  (∀ x₁ x₂, x₁ < x₂ → f a x₁ < f a x₂) ∧
  (∀ x, f a x = -f a (-x) ↔ a = 1/2) ∧
  (a = 1/2 →
    (∀ y, -1/2 < y ∧ y < 1/2 → ∃ x, f a x = y) ∧
    (∀ x, -1/2 < f a x ∧ f a x < 1/2)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1920_192001


namespace NUMINAMATH_CALUDE_power_of_power_l1920_192072

theorem power_of_power (a : ℝ) : (a^3)^2 = a^6 := by sorry

end NUMINAMATH_CALUDE_power_of_power_l1920_192072


namespace NUMINAMATH_CALUDE_equations_represent_problem_l1920_192084

/-- Represents the money each person brought -/
structure Money where
  a : ℝ  -- Amount A brought
  b : ℝ  -- Amount B brought

/-- Checks if the given equations satisfy the conditions of the problem -/
def satisfies_conditions (m : Money) : Prop :=
  (m.a + (1/2) * m.b = 50) ∧ (m.b + (2/3) * m.a = 50)

/-- Theorem stating that the equations correctly represent the problem -/
theorem equations_represent_problem :
  ∃ (m : Money), satisfies_conditions m :=
sorry

end NUMINAMATH_CALUDE_equations_represent_problem_l1920_192084


namespace NUMINAMATH_CALUDE_initial_stock_value_l1920_192086

/-- Represents the daily change in stock value -/
def daily_change : ℤ := 1

/-- Represents the number of days until the stock reaches $200 -/
def days_to_target : ℕ := 100

/-- Represents the target value of the stock -/
def target_value : ℤ := 200

/-- Theorem stating that the initial stock value is $101 -/
theorem initial_stock_value (V : ℤ) :
  V + (days_to_target - 1) * daily_change = target_value →
  V = 101 := by
  sorry

end NUMINAMATH_CALUDE_initial_stock_value_l1920_192086


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l1920_192085

theorem trigonometric_simplification :
  (Real.sin (15 * π / 180) + Real.sin (30 * π / 180) + Real.sin (45 * π / 180) + 
   Real.sin (60 * π / 180) + Real.sin (75 * π / 180)) / 
  (Real.cos (10 * π / 180) * Real.cos (20 * π / 180) * Real.cos (30 * π / 180)) = 
  (Real.sqrt 2 * (4 * Real.cos (22.5 * π / 180) * Real.cos (7.5 * π / 180) + 1)) / 
  (Real.cos (10 * π / 180) * Real.cos (20 * π / 180) * Real.cos (30 * π / 180)) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l1920_192085


namespace NUMINAMATH_CALUDE_smallest_r_is_two_l1920_192052

theorem smallest_r_is_two :
  ∃ (r : ℝ), r > 0 ∧ r = 2 ∧
  (∀ (a : ℝ), a > 0 →
    ∃ (x : ℝ), (2 - a * r ≤ x) ∧ (x ≤ 2) ∧ (a * x^3 + x^2 - 4 = 0)) ∧
  (∀ (r' : ℝ), r' > 0 →
    (∀ (a : ℝ), a > 0 →
      ∃ (x : ℝ), (2 - a * r' ≤ x) ∧ (x ≤ 2) ∧ (a * x^3 + x^2 - 4 = 0)) →
    r' ≥ r) :=
by sorry

end NUMINAMATH_CALUDE_smallest_r_is_two_l1920_192052


namespace NUMINAMATH_CALUDE_quadratic_sum_l1920_192051

theorem quadratic_sum (a b c : ℝ) : 
  (∀ x, 5 * x^2 - 30 * x - 45 = a * (x + b)^2 + c) → 
  a + b + c = -88 := by
sorry

end NUMINAMATH_CALUDE_quadratic_sum_l1920_192051


namespace NUMINAMATH_CALUDE_simultaneous_equations_solution_l1920_192064

/-- The simultaneous equations y = kx + 5 and y = (3k - 2)x + 6 have at least one solution
    in terms of real numbers (x, y) if and only if k ≠ 1 -/
theorem simultaneous_equations_solution (k : ℝ) :
  (∃ x y : ℝ, y = k * x + 5 ∧ y = (3 * k - 2) * x + 6) ↔ k ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_simultaneous_equations_solution_l1920_192064


namespace NUMINAMATH_CALUDE_poker_loss_l1920_192049

theorem poker_loss (initial_amount winnings debt : ℤ) : 
  initial_amount = 100 → winnings = 65 → debt = 50 → 
  (initial_amount + winnings + debt) = 215 := by
sorry

end NUMINAMATH_CALUDE_poker_loss_l1920_192049


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l1920_192039

/-- Given a line L1 with equation x - 2y + 3 = 0 and a point A(1, 2),
    the line L2 passing through A and perpendicular to L1 has the equation 2x + y - 4 = 0 -/
theorem perpendicular_line_equation (x y : ℝ) : 
  let L1 : ℝ → ℝ → Prop := λ x y ↦ x - 2*y + 3 = 0
  let A : ℝ × ℝ := (1, 2)
  let L2 : ℝ → ℝ → Prop := λ x y ↦ 2*x + y - 4 = 0
  (∀ x y, L1 x y ↔ x - 2*y + 3 = 0) →
  (L2 A.1 A.2) →
  (∀ x₁ y₁ x₂ y₂, L1 x₁ y₁ → L1 x₂ y₂ → L2 x₁ y₁ → L2 x₂ y₂ → 
    (x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁) ≠ 0 →
    ((x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁)) * ((x₂ - x₁) * (x₂ - x₁)) = 0) →
  ∀ x y, L2 x y ↔ 2*x + y - 4 = 0 := by
sorry


end NUMINAMATH_CALUDE_perpendicular_line_equation_l1920_192039


namespace NUMINAMATH_CALUDE_gcd_180_270_l1920_192069

theorem gcd_180_270 : Nat.gcd 180 270 = 90 := by
  sorry

end NUMINAMATH_CALUDE_gcd_180_270_l1920_192069


namespace NUMINAMATH_CALUDE_student_number_proof_l1920_192005

theorem student_number_proof : 
  ∃ x : ℝ, (2 * x - 138 = 102) ∧ (x = 120) := by
  sorry

end NUMINAMATH_CALUDE_student_number_proof_l1920_192005


namespace NUMINAMATH_CALUDE_fence_cost_circular_plot_l1920_192083

/-- The cost of building a fence around a circular plot -/
theorem fence_cost_circular_plot (area : ℝ) (price_per_foot : ℝ) : 
  area = 289 → price_per_foot = 58 → 
  (2 * Real.sqrt area * price_per_foot : ℝ) = 1972 := by
  sorry

#check fence_cost_circular_plot

end NUMINAMATH_CALUDE_fence_cost_circular_plot_l1920_192083


namespace NUMINAMATH_CALUDE_sector_area_l1920_192046

/-- The area of a sector with central angle 2π/3 and radius 3 is 3π. -/
theorem sector_area (θ : Real) (r : Real) (h1 : θ = 2 * Real.pi / 3) (h2 : r = 3) :
  (θ / (2 * Real.pi)) * Real.pi * r^2 = 3 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l1920_192046


namespace NUMINAMATH_CALUDE_custom_op_two_three_l1920_192013

-- Define the custom operation
def customOp (x y : ℕ) : ℕ := x + y^2

-- Theorem statement
theorem custom_op_two_three : customOp 2 3 = 11 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_two_three_l1920_192013


namespace NUMINAMATH_CALUDE_hundred_to_fifty_equals_ten_to_hundred_l1920_192075

theorem hundred_to_fifty_equals_ten_to_hundred : 100 ^ 50 = 10 ^ 100 := by
  sorry

end NUMINAMATH_CALUDE_hundred_to_fifty_equals_ten_to_hundred_l1920_192075


namespace NUMINAMATH_CALUDE_abcd_hex_binary_digits_l1920_192008

-- Define the hexadecimal number ABCD₁₆ as its decimal equivalent
def abcd_hex : ℕ := 43981

-- Theorem stating that the binary representation of ABCD₁₆ requires 16 bits
theorem abcd_hex_binary_digits : 
  (Nat.log 2 abcd_hex).succ = 16 := by sorry

end NUMINAMATH_CALUDE_abcd_hex_binary_digits_l1920_192008


namespace NUMINAMATH_CALUDE_car_speed_decrease_l1920_192017

/-- Proves that the speed decrease per interval is 3 mph given the conditions of the problem -/
theorem car_speed_decrease (initial_speed : ℝ) (distance_fifth : ℝ) (interval_duration : ℝ) :
  initial_speed = 45 →
  distance_fifth = 4.4 →
  interval_duration = 8 / 60 →
  ∃ (speed_decrease : ℝ),
    speed_decrease = 3 ∧
    initial_speed - 4 * speed_decrease = distance_fifth / interval_duration :=
by sorry

end NUMINAMATH_CALUDE_car_speed_decrease_l1920_192017


namespace NUMINAMATH_CALUDE_train_length_calculation_l1920_192088

/-- Calculates the length of a train given its speed, tunnel length, and time to pass through the tunnel. -/
theorem train_length_calculation (train_speed : ℝ) (tunnel_length : ℝ) (passing_time : ℝ) :
  train_speed = 72 →
  tunnel_length = 1.7 →
  passing_time = 1.5 / 60 →
  (train_speed * passing_time) - tunnel_length = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l1920_192088


namespace NUMINAMATH_CALUDE_circle_line_distance_range_l1920_192099

/-- Given a circle x^2 + y^2 = 16 and a line y = x + b, if there are at least three points
    on the circle with a distance of 1 from the line, then -3√2 ≤ b ≤ 3√2 -/
theorem circle_line_distance_range (b : ℝ) :
  (∃ (p q r : ℝ × ℝ),
    p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
    (p.1^2 + p.2^2 = 16) ∧ (q.1^2 + q.2^2 = 16) ∧ (r.1^2 + r.2^2 = 16) ∧
    (abs (p.1 - p.2 + b) / Real.sqrt 2 = 1) ∧
    (abs (q.1 - q.2 + b) / Real.sqrt 2 = 1) ∧
    (abs (r.1 - r.2 + b) / Real.sqrt 2 = 1)) →
  -3 * Real.sqrt 2 ≤ b ∧ b ≤ 3 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_circle_line_distance_range_l1920_192099


namespace NUMINAMATH_CALUDE_pyramid_volume_is_four_thirds_l1920_192079

-- Define the cube IJKLMNO
structure Cube where
  volume : ℝ

-- Define the pyramid IJMO
structure Pyramid where
  base : Cube

-- Define the volume of the pyramid
def pyramid_volume (p : Pyramid) : ℝ := sorry

-- Theorem statement
theorem pyramid_volume_is_four_thirds (c : Cube) (p : Pyramid) 
  (h1 : c.volume = 8) 
  (h2 : p.base = c) : 
  pyramid_volume p = 4/3 := by sorry

end NUMINAMATH_CALUDE_pyramid_volume_is_four_thirds_l1920_192079


namespace NUMINAMATH_CALUDE_tangent_equality_solution_l1920_192032

theorem tangent_equality_solution (x : Real) : 
  0 < x ∧ x < 360 →
  Real.tan ((150 - x) * π / 180) = 
    (Real.sin (150 * π / 180) - Real.sin (x * π / 180)) / 
    (Real.cos (150 * π / 180) - Real.cos (x * π / 180)) →
  x = 100 ∨ x = 220 := by
sorry

end NUMINAMATH_CALUDE_tangent_equality_solution_l1920_192032


namespace NUMINAMATH_CALUDE_angle_F_is_60_l1920_192037

/-- A trapezoid with specific angle relationships -/
structure SpecialTrapezoid where
  -- Angles of the trapezoid
  angleE : ℝ
  angleF : ℝ
  angleG : ℝ
  angleH : ℝ
  -- Conditions given in the problem
  parallel_sides : True  -- Represents that EF and GH are parallel
  angle_E_triple_H : angleE = 3 * angleH
  angle_G_double_F : angleG = 2 * angleF
  -- Properties of a trapezoid
  sum_angles : angleE + angleF + angleG + angleH = 360
  opposite_angles_sum : angleF + angleG = 180

/-- Theorem stating that in the special trapezoid, angle F measures 60 degrees -/
theorem angle_F_is_60 (t : SpecialTrapezoid) : t.angleF = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_F_is_60_l1920_192037


namespace NUMINAMATH_CALUDE_exists_special_function_l1920_192055

/-- The closed interval [0, 1] -/
def ClosedUnitInterval : Set ℝ := { x | 0 ≤ x ∧ x ≤ 1 }

/-- A continuous function from [0, 1] to [0, 1] -/
def ContinuousUnitFunction (f : ℝ → ℝ) : Prop :=
  Continuous f ∧ ∀ x ∈ ClosedUnitInterval, f x ∈ ClosedUnitInterval

/-- A line in ℝ² -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The number of intersections between a function and a line -/
def NumberOfIntersections (f : ℝ → ℝ) (l : Line) : ℕ := sorry

/-- The existence of a function with the required properties -/
theorem exists_special_function :
  ∃ g : ℝ → ℝ,
    ContinuousUnitFunction g ∧
    (∀ l : Line, NumberOfIntersections g l < ω) ∧
    (∀ n : ℕ, ∃ l : Line, NumberOfIntersections g l > n) :=
sorry

end NUMINAMATH_CALUDE_exists_special_function_l1920_192055


namespace NUMINAMATH_CALUDE_min_max_values_l1920_192096

/-- Given positive real numbers x and y satisfying x² + y² = x + y,
    prove that the minimum value of 1/x + 1/y is 2 and the maximum value of x + y is 2 -/
theorem min_max_values (x y : ℝ) (h_pos : x > 0 ∧ y > 0) (h_eq : x^2 + y^2 = x + y) :
  (∀ a b : ℝ, a > 0 → b > 0 → a^2 + b^2 = a + b → 1/x + 1/y ≤ 1/a + 1/b) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → a^2 + b^2 = a + b → x + y ≥ a + b) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a^2 + b^2 = a + b ∧ 1/a + 1/b = 2) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a^2 + b^2 = a + b ∧ a + b = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_max_values_l1920_192096


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1920_192004

/-- A regular polygon with side length 10 and perimeter 60 has 6 sides -/
theorem regular_polygon_sides (s : ℕ) (side_length perimeter : ℝ) : 
  side_length = 10 → perimeter = 60 → s * side_length = perimeter → s = 6 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1920_192004


namespace NUMINAMATH_CALUDE_percentage_d_grades_l1920_192006

def scores : List ℕ := [89, 65, 55, 96, 73, 93, 82, 70, 77, 65, 81, 79, 67, 85, 88, 61, 84, 71, 73, 90]

def is_d_grade (score : ℕ) : Bool :=
  65 ≤ score ∧ score ≤ 75

def count_d_grades (scores : List ℕ) : ℕ :=
  scores.filter is_d_grade |>.length

theorem percentage_d_grades :
  (count_d_grades scores : ℚ) / scores.length * 100 = 35 := by
  sorry

end NUMINAMATH_CALUDE_percentage_d_grades_l1920_192006


namespace NUMINAMATH_CALUDE_allison_sewing_time_l1920_192090

/-- The time it takes Al to sew dresses individually -/
def al_time : ℝ := 12

/-- The time Allison and Al work together -/
def joint_work_time : ℝ := 3

/-- The additional time Allison works alone after Al leaves -/
def allison_extra_time : ℝ := 3.75

/-- The time it takes Allison to sew dresses individually -/
def allison_time : ℝ := 9

theorem allison_sewing_time : 
  (joint_work_time / allison_time + joint_work_time / al_time + allison_extra_time / allison_time) = 1 := by
  sorry

#check allison_sewing_time

end NUMINAMATH_CALUDE_allison_sewing_time_l1920_192090


namespace NUMINAMATH_CALUDE_log_ratio_squared_l1920_192028

theorem log_ratio_squared (x y : ℝ) (hx : x > 0) (hy : y > 0) (hx1 : x ≠ 1) (hy1 : y ≠ 1) 
  (h1 : Real.log x / Real.log 3 = Real.log 81 / Real.log y) (h2 : x * y = 243) :
  (Real.log (x / y) / Real.log 3)^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_log_ratio_squared_l1920_192028


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1920_192025

/-- A triangle with two equal sides and side lengths of 3 and 5 has a perimeter of either 11 or 13 -/
theorem isosceles_triangle_perimeter : ∀ (a b : ℝ), 
  a = 3 ∧ b = 5 →
  (∃ (p : ℝ), (p = 11 ∨ p = 13) ∧ 
   ((2 * a + b = p ∧ a + a > b) ∨ (2 * b + a = p ∧ b + b > a))) :=
by sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1920_192025


namespace NUMINAMATH_CALUDE_slope_of_line_l1920_192015

/-- The slope of a line represented by the equation 3y = 4x + 9 is 4/3 -/
theorem slope_of_line (x y : ℝ) : 3 * y = 4 * x + 9 → (y - 3) / x = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_slope_of_line_l1920_192015


namespace NUMINAMATH_CALUDE_jolene_earnings_180_l1920_192014

/-- Represents Jolene's earnings from various jobs -/
structure JoleneEarnings where
  babysitting_families : ℕ
  babysitting_rate : ℕ
  car_washing_jobs : ℕ
  car_washing_rate : ℕ

/-- Calculates Jolene's total earnings -/
def total_earnings (e : JoleneEarnings) : ℕ :=
  e.babysitting_families * e.babysitting_rate + e.car_washing_jobs * e.car_washing_rate

/-- Theorem stating that Jolene's total earnings are $180 -/
theorem jolene_earnings_180 :
  ∃ (e : JoleneEarnings),
    e.babysitting_families = 4 ∧
    e.babysitting_rate = 30 ∧
    e.car_washing_jobs = 5 ∧
    e.car_washing_rate = 12 ∧
    total_earnings e = 180 := by
  sorry

end NUMINAMATH_CALUDE_jolene_earnings_180_l1920_192014


namespace NUMINAMATH_CALUDE_base_conversion_and_operation_l1920_192007

-- Define the base conversions
def base9_to_10 (n : ℕ) : ℕ := n

def base4_to_10 (n : ℕ) : ℕ := n

def base8_to_10 (n : ℕ) : ℕ := n

-- Define the operation
def operation (a b c d : ℕ) : ℕ := a / b - c + d

-- Theorem statement
theorem base_conversion_and_operation :
  operation (base9_to_10 1357) (base4_to_10 100) (base8_to_10 2460) (base9_to_10 5678) = 2938 := by
  sorry

-- Additional lemmas for individual base conversions
lemma base9_1357 : base9_to_10 1357 = 1024 := by sorry
lemma base4_100 : base4_to_10 100 = 16 := by sorry
lemma base8_2460 : base8_to_10 2460 = 1328 := by sorry
lemma base9_5678 : base9_to_10 5678 = 4202 := by sorry

end NUMINAMATH_CALUDE_base_conversion_and_operation_l1920_192007


namespace NUMINAMATH_CALUDE_jerrys_debt_l1920_192041

theorem jerrys_debt (total_debt : ℕ) (first_payment : ℕ) (additional_payment : ℕ) 
  (h1 : total_debt = 50)
  (h2 : first_payment = 12)
  (h3 : additional_payment = 3) : 
  total_debt - (first_payment + (first_payment + additional_payment)) = 23 :=
by sorry

end NUMINAMATH_CALUDE_jerrys_debt_l1920_192041


namespace NUMINAMATH_CALUDE_geometric_progression_common_ratio_l1920_192070

/-- A geometric progression where each term is positive and any given term 
    is equal to the sum of the next three following terms. -/
structure GeometricProgression where
  a : ℝ  -- First term
  r : ℝ  -- Common ratio
  a_pos : 0 < a  -- Each term is positive
  r_pos : 0 < r  -- Common ratio is positive (to ensure all terms are positive)
  sum_property : ∀ n : ℕ, a * r^n = a * r^(n+1) + a * r^(n+2) + a * r^(n+3)

theorem geometric_progression_common_ratio 
  (gp : GeometricProgression) : 
  gp.r^3 + gp.r^2 + gp.r - 1 = 0 ∧ 
  abs (gp.r - 0.5437) < 0.0001 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_common_ratio_l1920_192070


namespace NUMINAMATH_CALUDE_chord_length_l1920_192056

/-- Theorem: The length of a chord in a circle is √(2ar), where r is the radius of the circle
    and a is the distance from one end of the chord to the tangent drawn through its other end. -/
theorem chord_length (r a : ℝ) (hr : r > 0) (ha : a > 0) :
  ∃ (chord_length : ℝ), chord_length = Real.sqrt (2 * a * r) := by
  sorry

end NUMINAMATH_CALUDE_chord_length_l1920_192056


namespace NUMINAMATH_CALUDE_maria_average_sales_l1920_192063

/-- The average number of kilograms of apples sold per hour by Maria at the market -/
def average_apples_sold (first_hour_sales second_hour_sales : ℕ) (total_hours : ℕ) : ℚ :=
  (first_hour_sales + second_hour_sales : ℚ) / total_hours

/-- Theorem stating that Maria's average apple sales per hour is 6 kg/hour -/
theorem maria_average_sales :
  average_apples_sold 10 2 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_maria_average_sales_l1920_192063


namespace NUMINAMATH_CALUDE_root_product_equals_21_l1920_192073

theorem root_product_equals_21 (x₁ x₂ x₃ : ℝ) :
  x₁ < x₂ ∧ x₂ < x₃ ∧
  (∀ x, Real.sqrt 100 * x^3 - 210 * x^2 + 3 = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) →
  x₂ * (x₁ + x₃) = 21 := by
sorry

end NUMINAMATH_CALUDE_root_product_equals_21_l1920_192073


namespace NUMINAMATH_CALUDE_mildreds_oranges_l1920_192033

/-- The number of oranges Mildred's father gave her -/
def oranges_given (initial final : ℕ) : ℕ := final - initial

theorem mildreds_oranges : oranges_given 77 79 = 2 := by
  sorry

end NUMINAMATH_CALUDE_mildreds_oranges_l1920_192033


namespace NUMINAMATH_CALUDE_f_value_at_3_l1920_192031

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^5 - b * x^3 + c * x - 3

-- State the theorem
theorem f_value_at_3 (a b c : ℝ) : f a b c (-3) = 7 → f a b c 3 = -13 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_3_l1920_192031


namespace NUMINAMATH_CALUDE_new_shoes_cost_proof_l1920_192048

/-- The cost of repairing used shoes -/
def repair_cost : ℝ := 10.50

/-- The duration (in years) that repaired used shoes last -/
def repair_duration : ℝ := 1

/-- The duration (in years) that new shoes last -/
def new_duration : ℝ := 2

/-- The percentage increase in average cost per year of new shoes compared to repaired shoes -/
def cost_increase_percentage : ℝ := 42.857142857142854

/-- The cost of purchasing new shoes -/
def new_shoes_cost : ℝ := 30.00

/-- Theorem stating that the cost of new shoes is $30.00 given the problem conditions -/
theorem new_shoes_cost_proof :
  new_shoes_cost = (repair_cost / repair_duration + cost_increase_percentage / 100 * repair_cost) * new_duration :=
by sorry

end NUMINAMATH_CALUDE_new_shoes_cost_proof_l1920_192048


namespace NUMINAMATH_CALUDE_batsman_highest_score_l1920_192016

theorem batsman_highest_score 
  (total_innings : ℕ) 
  (overall_average : ℚ) 
  (score_difference : ℕ) 
  (average_without_extremes : ℚ) 
  (h : total_innings = 46)
  (i : overall_average = 62)
  (j : score_difference = 150)
  (k : average_without_extremes = 58) :
  ∃ (highest_score lowest_score : ℕ),
    highest_score - lowest_score = score_difference ∧
    highest_score + lowest_score = total_innings * overall_average - (total_innings - 2) * average_without_extremes ∧
    highest_score = 221 :=
by sorry

end NUMINAMATH_CALUDE_batsman_highest_score_l1920_192016


namespace NUMINAMATH_CALUDE_hyperbola_focal_distance_l1920_192027

-- Define the hyperbola
def is_on_hyperbola (x y : ℝ) : Prop :=
  x^2 / 9 - y^2 / 16 = 1

-- Define the foci
def left_focus : ℝ × ℝ := sorry
def right_focus : ℝ × ℝ := sorry

-- Define the distance function
def distance (p q : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem hyperbola_focal_distance 
  (P : ℝ × ℝ) 
  (h_on_hyperbola : is_on_hyperbola P.1 P.2) 
  (h_left_distance : distance P left_focus = 3) : 
  distance P right_focus = 9 := by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_distance_l1920_192027


namespace NUMINAMATH_CALUDE_sign_determination_l1920_192011

theorem sign_determination (a b : ℝ) (h1 : a * b > 0) (h2 : a + b < 0) : a < 0 ∧ b < 0 := by
  sorry

end NUMINAMATH_CALUDE_sign_determination_l1920_192011


namespace NUMINAMATH_CALUDE_lights_at_top_point_l1920_192050

/-- Represents the number of layers in the structure -/
def num_layers : ℕ := 7

/-- Represents the common ratio of the geometric sequence -/
def common_ratio : ℕ := 2

/-- Represents the total number of lights -/
def total_lights : ℕ := 381

/-- Theorem stating that the number of lights at the topmost point is 3 -/
theorem lights_at_top_point : 
  ∃ (a : ℕ), a * (common_ratio ^ num_layers - 1) / (common_ratio - 1) = total_lights ∧ a = 3 :=
sorry

end NUMINAMATH_CALUDE_lights_at_top_point_l1920_192050


namespace NUMINAMATH_CALUDE_find_b_value_l1920_192045

theorem find_b_value (a b : ℚ) (eq1 : 3 * a + 3 = 0) (eq2 : 2 * b - a = 4) : b = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_find_b_value_l1920_192045


namespace NUMINAMATH_CALUDE_range_of_x_minus_y_l1920_192018

theorem range_of_x_minus_y (x y : ℝ) (hx : -1 < x ∧ x < 4) (hy : 2 < y ∧ y < 3) :
  ∃ a b : ℝ, a = -4 ∧ b = 2 ∧ a < x - y ∧ x - y < b :=
sorry

end NUMINAMATH_CALUDE_range_of_x_minus_y_l1920_192018


namespace NUMINAMATH_CALUDE_sqrt_three_squared_five_fourth_l1920_192082

theorem sqrt_three_squared_five_fourth (x : ℝ) : 
  x = Real.sqrt (3^2 * 5^4) → x = 75 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_squared_five_fourth_l1920_192082


namespace NUMINAMATH_CALUDE_optimal_fence_dimensions_l1920_192081

/-- Represents the dimensions of a rectangular plot -/
structure PlotDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the total fence length for a given plot -/
def totalFenceLength (d : PlotDimensions) : ℝ :=
  3 * d.length + 2 * d.width

/-- Theorem stating the optimal dimensions for minimal fence length -/
theorem optimal_fence_dimensions :
  ∃ (d : PlotDimensions),
    d.length * d.width = 294 ∧
    d.length = 14 ∧
    d.width = 21 ∧
    ∀ (d' : PlotDimensions),
      d'.length * d'.width = 294 →
      totalFenceLength d ≤ totalFenceLength d' := by
  sorry

end NUMINAMATH_CALUDE_optimal_fence_dimensions_l1920_192081


namespace NUMINAMATH_CALUDE_pentagon_smallest_angle_l1920_192071

theorem pentagon_smallest_angle 
  (angles : Fin 5 → ℝ)
  (arithmetic_sequence : ∀ i : Fin 4, angles (i + 1) - angles i = angles (i + 2) - angles (i + 1))
  (largest_angle : angles 4 = 150)
  (angle_sum : angles 0 + angles 1 + angles 2 + angles 3 + angles 4 = 540) :
  angles 0 = 66 := by
sorry

end NUMINAMATH_CALUDE_pentagon_smallest_angle_l1920_192071


namespace NUMINAMATH_CALUDE_pascal_triangle_row20_element5_l1920_192093

theorem pascal_triangle_row20_element5 : Nat.choose 20 4 = 4845 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_row20_element5_l1920_192093


namespace NUMINAMATH_CALUDE_count_valid_pairs_l1920_192097

-- Define ω as a complex number that is a nonreal root of z^4 = 1
def ω : ℂ := sorry

-- Define the property for the ordered pairs we're looking for
def validPair (a b : ℤ) : Prop :=
  Complex.abs (a • ω + b) = 1

-- State the theorem
theorem count_valid_pairs :
  ∃! (n : ℕ), ∃ (S : Finset (ℤ × ℤ)), 
    S.card = n ∧ 
    (∀ (p : ℤ × ℤ), p ∈ S ↔ validPair p.1 p.2) ∧
    n = 4 := by sorry

end NUMINAMATH_CALUDE_count_valid_pairs_l1920_192097


namespace NUMINAMATH_CALUDE_hot_dogs_remainder_l1920_192024

theorem hot_dogs_remainder : 25197638 % 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_hot_dogs_remainder_l1920_192024


namespace NUMINAMATH_CALUDE_proof_by_contradiction_method_l1920_192038

-- Define what proof by contradiction means
def proof_by_contradiction (P : Prop) : Prop :=
  ∃ (proof : ¬P → False), P

-- State the theorem
theorem proof_by_contradiction_method :
  ¬(∀ (P Q : Prop), proof_by_contradiction P ↔ (¬P ∧ ¬Q → False)) :=
sorry

end NUMINAMATH_CALUDE_proof_by_contradiction_method_l1920_192038


namespace NUMINAMATH_CALUDE_dental_cleaning_theorem_l1920_192092

/-- Represents the number of teeth for different animals --/
structure AnimalTeeth where
  dog : ℕ
  cat : ℕ
  pig : ℕ

/-- Represents the number of animals to be cleaned --/
structure AnimalsToClean where
  dogs : ℕ
  cats : ℕ
  pigs : ℕ

/-- Calculates the total number of teeth cleaned --/
def totalTeethCleaned (teeth : AnimalTeeth) (animals : AnimalsToClean) : ℕ :=
  teeth.dog * animals.dogs + teeth.cat * animals.cats + teeth.pig * animals.pigs

/-- Theorem stating that given the conditions, 5 dogs result in 706 teeth cleaned --/
theorem dental_cleaning_theorem (teeth : AnimalTeeth) (animals : AnimalsToClean) :
  teeth.dog = 42 →
  teeth.cat = 30 →
  teeth.pig = 28 →
  animals.cats = 10 →
  animals.pigs = 7 →
  totalTeethCleaned teeth { dogs := 5, cats := animals.cats, pigs := animals.pigs } = 706 :=
by
  sorry

#check dental_cleaning_theorem

end NUMINAMATH_CALUDE_dental_cleaning_theorem_l1920_192092


namespace NUMINAMATH_CALUDE_compute_expression_l1920_192019

theorem compute_expression : 8 * (1/3)^4 = 8/81 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l1920_192019


namespace NUMINAMATH_CALUDE_productivity_increase_l1920_192042

/-- Represents the productivity level during a work shift -/
structure Productivity where
  planned : ℝ
  reduced : ℝ

/-- Represents a work shift with its duration and productivity levels -/
structure Shift where
  duration : ℝ
  plannedHours : ℝ
  productivity : Productivity

/-- Calculates the total work done during a shift -/
def totalWork (s : Shift) : ℝ :=
  s.plannedHours * s.productivity.planned +
  (s.duration - s.plannedHours) * s.productivity.reduced

/-- Theorem stating the productivity increase when extending the workday -/
theorem productivity_increase
  (initialShift : Shift)
  (extendedShift : Shift)
  (h1 : initialShift.duration = 8)
  (h2 : initialShift.plannedHours = 6)
  (h3 : initialShift.productivity.planned = 1)
  (h4 : initialShift.productivity.reduced = 0.75)
  (h5 : extendedShift.duration = 9)
  (h6 : extendedShift.plannedHours = 6)
  (h7 : extendedShift.productivity.planned = 1)
  (h8 : extendedShift.productivity.reduced = 0.7) :
  (totalWork extendedShift - totalWork initialShift) / totalWork initialShift = 0.08 := by
  sorry


end NUMINAMATH_CALUDE_productivity_increase_l1920_192042


namespace NUMINAMATH_CALUDE_right_angled_triangle_l1920_192077

theorem right_angled_triangle (h₁ h₂ h₃ : ℝ) (h_positive : h₁ > 0 ∧ h₂ > 0 ∧ h₃ > 0)
  (h_altitudes : h₁ = 12 ∧ h₂ = 15 ∧ h₃ = 20) :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a * h₁ = 2 * (b * c / 2)) ∧
    (b * h₂ = 2 * (a * c / 2)) ∧
    (c * h₃ = 2 * (a * b / 2)) ∧
    a^2 + b^2 = c^2 :=
by sorry

end NUMINAMATH_CALUDE_right_angled_triangle_l1920_192077


namespace NUMINAMATH_CALUDE_axis_of_symmetry_l1920_192074

noncomputable def f (ω φ x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

theorem axis_of_symmetry 
  (ω φ : ℝ) 
  (h_ω : ω > 0) 
  (h_φ : 0 ≤ φ ∧ φ < Real.pi) 
  (h_even : ∀ x, f ω φ x = f ω φ (-x)) 
  (h_distance : ∃ (a b : ℝ), b - a = 4 * Real.sqrt 2 ∧ f ω φ b = f ω φ a) :
  ∃ (x : ℝ), x = 4 ∧ ∀ y, f ω φ (x + y) = f ω φ (x - y) :=
sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_l1920_192074


namespace NUMINAMATH_CALUDE_sum_of_digits_of_4444_power_4444_l1920_192054

-- Define the sum of digits function
def S (n : ℕ) : ℕ := sorry

-- State the theorem
theorem sum_of_digits_of_4444_power_4444 :
  ∃ (S : ℕ → ℕ),
    (∀ n : ℕ, S n % 9 = n % 9) →
    S (S (S (4444^4444))) = 7 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_4444_power_4444_l1920_192054


namespace NUMINAMATH_CALUDE_smallest_root_of_quadratic_l1920_192089

theorem smallest_root_of_quadratic (x : ℝ) :
  (4 * x^2 - 20 * x + 24 = 0) → (x ≥ 2) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_root_of_quadratic_l1920_192089


namespace NUMINAMATH_CALUDE_parabola_and_intersection_l1920_192068

-- Define the parabola C
def parabola_C (x y : ℝ) : Prop := y^2 = 8*x

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2/2 - y^2/2 = 1

-- Define the line passing through P(3,1) with slope 1
def line (x y : ℝ) : Prop := y = x - 2

-- Theorem statement
theorem parabola_and_intersection :
  -- Conditions
  (∀ x y, parabola_C x y → (x = 0 ∧ y = 0) → True) → -- Vertex at origin
  (∀ x, parabola_C x 0 → True) → -- Axis of symmetry is coordinate axis
  (∃ x₀, x₀ = -2 ∧ (∀ x y, parabola_C x y → |x - x₀| = y^2/(4*x₀))) → -- Directrix passes through left focus of hyperbola
  -- Conclusions
  (∀ x y, parabola_C x y ↔ y^2 = 8*x) ∧ -- Equation of parabola C
  (∃ x₁ y₁ x₂ y₂, 
    parabola_C x₁ y₁ ∧ parabola_C x₂ y₂ ∧ 
    line x₁ y₁ ∧ line x₂ y₂ ∧ 
    ((x₁ - x₂)^2 + (y₁ - y₂)^2)^(1/2 : ℝ) = 16) -- Length of MN is 16
  := by sorry

end NUMINAMATH_CALUDE_parabola_and_intersection_l1920_192068


namespace NUMINAMATH_CALUDE_pencil_distribution_l1920_192060

theorem pencil_distribution (initial_pencils : ℕ) (kept_pencils : ℕ) (extra_to_nilo : ℕ) 
  (h1 : initial_pencils = 50)
  (h2 : kept_pencils = 20)
  (h3 : extra_to_nilo = 10) : 
  ∃ (pencils_to_manny : ℕ), 
    pencils_to_manny + (pencils_to_manny + extra_to_nilo) = initial_pencils - kept_pencils ∧ 
    pencils_to_manny = 10 := by
  sorry

end NUMINAMATH_CALUDE_pencil_distribution_l1920_192060


namespace NUMINAMATH_CALUDE_sin_18_cos_36_eq_quarter_l1920_192067

theorem sin_18_cos_36_eq_quarter : Real.sin (18 * π / 180) * Real.cos (36 * π / 180) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_sin_18_cos_36_eq_quarter_l1920_192067


namespace NUMINAMATH_CALUDE_rotated_square_height_l1920_192044

theorem rotated_square_height (square_side : Real) (rotation_angle : Real) : 
  square_side = 2 ∧ rotation_angle = π / 6 →
  let diagonal := square_side * Real.sqrt 2
  let height_above_center := (diagonal / 2) * Real.sin rotation_angle
  let initial_center_height := square_side / 2
  initial_center_height + height_above_center = 1 + Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_rotated_square_height_l1920_192044


namespace NUMINAMATH_CALUDE_m_range_l1920_192003

def f (x : ℝ) : ℝ := x^2 - 2*x + 3

theorem m_range (m : ℝ) : 
  (∀ x ∈ Set.Icc 0 m, f x ≤ 3) ∧ 
  (∃ x ∈ Set.Icc 0 m, f x = 3) ∧
  (∀ x ∈ Set.Icc 0 m, f x ≥ 2) ∧
  (∃ x ∈ Set.Icc 0 m, f x = 2) →
  m ∈ Set.Icc 1 2 :=
by sorry

end NUMINAMATH_CALUDE_m_range_l1920_192003


namespace NUMINAMATH_CALUDE_largest_angle_hexagon_l1920_192036

/-- The largest interior angle of a convex hexagon with six consecutive integer angles -/
def largest_hexagon_angle : ℝ := 122.5

/-- A hexagon with six consecutive integer angles -/
structure ConsecutiveAngleHexagon where
  -- The smallest angle of the hexagon
  base_angle : ℝ
  -- Predicate ensuring the angles are consecutive integers
  consecutive_integers : ∀ i : Fin 6, (base_angle + i) = ↑(⌊base_angle⌋ + i)

/-- Theorem stating that the largest angle in a convex hexagon with six consecutive integer angles is 122.5° -/
theorem largest_angle_hexagon (h : ConsecutiveAngleHexagon) : 
  (h.base_angle + 5) = largest_hexagon_angle := by
  sorry

/-- The sum of interior angles of a hexagon is 720° -/
axiom sum_hexagon_angles : ∀ (h : ConsecutiveAngleHexagon), 
  (h.base_angle * 6 + 15) = 720

end NUMINAMATH_CALUDE_largest_angle_hexagon_l1920_192036


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1920_192098

theorem polynomial_divisibility (a b : ℕ) (h1 : a ≥ 2 * b) (h2 : b > 1) :
  ∃ P : Polynomial ℕ, (Polynomial.degree P > 0) ∧ 
    (∀ i, Polynomial.coeff P i < b) ∧
    (P.eval a % P.eval b = 0) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l1920_192098


namespace NUMINAMATH_CALUDE_correct_selection_count_l1920_192059

/-- Represents a basketball team with twins -/
structure BasketballTeam where
  total_players : Nat
  twin_sets : Nat
  non_twins : Nat

/-- Calculates the number of ways to select players for a game -/
def select_players (team : BasketballTeam) (to_select : Nat) : Nat :=
  sorry

/-- The specific basketball team from the problem -/
def our_team : BasketballTeam := {
  total_players := 16,
  twin_sets := 3,
  non_twins := 10
}

/-- Theorem stating the correct number of ways to select players -/
theorem correct_selection_count :
  select_players our_team 7 = 1380 := by sorry

end NUMINAMATH_CALUDE_correct_selection_count_l1920_192059


namespace NUMINAMATH_CALUDE_greatest_line_segment_length_l1920_192058

/-- The greatest possible length of a line segment joining two points on a circle -/
theorem greatest_line_segment_length (r : ℝ) (h : r = 4) : 
  ∃ (d : ℝ), d = 2 * r ∧ ∀ (l : ℝ), l ≤ d := by sorry

end NUMINAMATH_CALUDE_greatest_line_segment_length_l1920_192058


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1920_192012

theorem algebraic_expression_value (m n : ℝ) 
  (h1 : m - n = -2) 
  (h2 : m * n = 3) : 
  -m^3*n + 2*m^2*n^2 - m*n^3 = -12 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1920_192012


namespace NUMINAMATH_CALUDE_M_equals_set_l1920_192053

def M : Set ℕ := {m : ℕ | m > 0 ∧ (∃ k : ℤ, 10 = k * (m + 1))}

theorem M_equals_set : M = {1, 4, 9} := by
  sorry

end NUMINAMATH_CALUDE_M_equals_set_l1920_192053


namespace NUMINAMATH_CALUDE_t_shape_perimeter_l1920_192002

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

/-- Represents a T-shape formed by two rectangles -/
structure TShape where
  top : Rectangle
  bottom : Rectangle

/-- Calculates the perimeter of a T-shape -/
def TShape.perimeter (t : TShape) : ℝ :=
  t.top.perimeter + t.bottom.perimeter - 2 * t.top.width

theorem t_shape_perimeter : 
  let t : TShape := {
    top := { width := 1, height := 4 },
    bottom := { width := 5, height := 2 }
  }
  TShape.perimeter t = 20 := by sorry

end NUMINAMATH_CALUDE_t_shape_perimeter_l1920_192002
