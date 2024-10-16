import Mathlib

namespace NUMINAMATH_CALUDE_distance_calculation_l1637_163725

/-- Conversion factor from meters to kilometers -/
def meters_to_km : ℝ := 1000

/-- Distance from Xiaoqing's home to the park in meters -/
def total_distance : ℝ := 6000

/-- Distance Xiaoqing has already walked in meters -/
def walked_distance : ℝ := 1200

/-- Theorem stating the conversion of total distance to kilometers and the remaining distance to the park -/
theorem distance_calculation :
  (total_distance / meters_to_km = 6) ∧
  (total_distance - walked_distance = 4800) := by
  sorry

end NUMINAMATH_CALUDE_distance_calculation_l1637_163725


namespace NUMINAMATH_CALUDE_distance_between_vertices_l1637_163754

/-- The distance between the vertices of two quadratic functions -/
theorem distance_between_vertices (a b c d e f : ℝ) : 
  let vertex1 := (- b / (2 * a), a * (- b / (2 * a))^2 + b * (- b / (2 * a)) + c)
  let vertex2 := (- e / (2 * d), d * (- e / (2 * d))^2 + e * (- e / (2 * d)) + f)
  let distance := Real.sqrt ((vertex1.1 - vertex2.1)^2 + (vertex1.2 - vertex2.2)^2)
  (a = 1 ∧ b = -4 ∧ c = 7 ∧ d = 1 ∧ e = 6 ∧ f = 20) → distance = Real.sqrt 89 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_vertices_l1637_163754


namespace NUMINAMATH_CALUDE_tens_digit_of_7_pow_35_l1637_163789

-- Define a function to get the tens digit of a natural number
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

-- State the theorem
theorem tens_digit_of_7_pow_35 : tens_digit (7^35) = 4 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_7_pow_35_l1637_163789


namespace NUMINAMATH_CALUDE_convergence_of_beta_series_l1637_163755

theorem convergence_of_beta_series (α : ℕ → ℝ) (β : ℕ → ℝ) :
  (∀ n : ℕ, α n > 0) →
  (∀ n : ℕ, β n = (α n * n) / (n + 1)) →
  Summable α →
  Summable β := by
sorry

end NUMINAMATH_CALUDE_convergence_of_beta_series_l1637_163755


namespace NUMINAMATH_CALUDE_gcf_of_120_180_300_l1637_163710

theorem gcf_of_120_180_300 : Nat.gcd 120 (Nat.gcd 180 300) = 60 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_120_180_300_l1637_163710


namespace NUMINAMATH_CALUDE_marley_louis_orange_ratio_l1637_163798

theorem marley_louis_orange_ratio :
  let louis_oranges : ℕ := 5
  let samantha_apples : ℕ := 7
  let marley_apples : ℕ := 3 * samantha_apples
  let marley_total_fruits : ℕ := 31
  let marley_oranges : ℕ := marley_total_fruits - marley_apples
  (marley_oranges : ℚ) / louis_oranges = 2 := by sorry

end NUMINAMATH_CALUDE_marley_louis_orange_ratio_l1637_163798


namespace NUMINAMATH_CALUDE_age_difference_l1637_163796

theorem age_difference (A B C : ℕ) : 
  (∃ k : ℕ, A = B + k) →  -- A is some years older than B
  B = 2 * C →             -- B is twice as old as C
  A + B + C = 27 →        -- Total of ages is 27
  B = 10 →                -- B is 10 years old
  A = B + 2 :=            -- A is 2 years older than B
by sorry

end NUMINAMATH_CALUDE_age_difference_l1637_163796


namespace NUMINAMATH_CALUDE_rolling_coin_curve_length_l1637_163741

/-- The length of the curve traced by the center of a rolling coin -/
theorem rolling_coin_curve_length 
  (coin_circumference : ℝ) 
  (quadrilateral_perimeter : ℝ) : 
  coin_circumference = 5 →
  quadrilateral_perimeter = 20 →
  (curve_length : ℝ) = quadrilateral_perimeter + coin_circumference →
  curve_length = 25 :=
by sorry

end NUMINAMATH_CALUDE_rolling_coin_curve_length_l1637_163741


namespace NUMINAMATH_CALUDE_total_cartons_packed_l1637_163747

/-- Proves the total number of cartons packed given the conditions -/
theorem total_cartons_packed 
  (cans_per_carton : ℕ) 
  (loaded_cartons : ℕ) 
  (remaining_cans : ℕ) : 
  cans_per_carton = 20 → 
  loaded_cartons = 40 → 
  remaining_cans = 200 → 
  loaded_cartons + (remaining_cans / cans_per_carton) = 50 := by
  sorry

end NUMINAMATH_CALUDE_total_cartons_packed_l1637_163747


namespace NUMINAMATH_CALUDE_closest_integer_to_sqrt_6_l1637_163797

theorem closest_integer_to_sqrt_6 : 
  ∃ (n : ℤ), ∀ (m : ℤ), |n - Real.sqrt 6| ≤ |m - Real.sqrt 6| ∧ n = 2 :=
sorry

end NUMINAMATH_CALUDE_closest_integer_to_sqrt_6_l1637_163797


namespace NUMINAMATH_CALUDE_moe_has_least_money_l1637_163756

-- Define the set of people
inductive Person : Type
  | Bo | Coe | Flo | Jo | Moe | Zoe

-- Define the "has more money than" relation
def has_more_money (p1 p2 : Person) : Prop := sorry

-- Define the conditions
axiom different_amounts : ∀ (p1 p2 : Person), p1 ≠ p2 → has_more_money p1 p2 ∨ has_more_money p2 p1
axiom flo_bo_zoe : has_more_money Person.Flo Person.Bo ∧ has_more_money Person.Zoe Person.Flo
axiom zoe_coe : has_more_money Person.Zoe Person.Coe
axiom bo_coe_moe : has_more_money Person.Bo Person.Moe ∧ has_more_money Person.Coe Person.Moe
axiom jo_moe_zoe : has_more_money Person.Jo Person.Moe ∧ has_more_money Person.Zoe Person.Jo

-- Define the "has least money" property
def has_least_money (p : Person) : Prop :=
  ∀ (other : Person), other ≠ p → has_more_money other p

-- Theorem statement
theorem moe_has_least_money : has_least_money Person.Moe := by
  sorry

end NUMINAMATH_CALUDE_moe_has_least_money_l1637_163756


namespace NUMINAMATH_CALUDE_quadratic_equation_condition_l1637_163786

theorem quadratic_equation_condition (m : ℝ) : 
  (|m - 1| = 2 ∧ m + 1 ≠ 0) ↔ m = 3 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_condition_l1637_163786


namespace NUMINAMATH_CALUDE_fraction_simplification_l1637_163727

theorem fraction_simplification : (10^8) / (10 * 10^5) = 100 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1637_163727


namespace NUMINAMATH_CALUDE_gravitational_force_at_new_distance_l1637_163780

/-- Gravitational force calculation -/
theorem gravitational_force_at_new_distance
  (f1 : ℝ) (d1 : ℝ) (d2 : ℝ)
  (h1 : f1 = 480)
  (h2 : d1 = 5000)
  (h3 : d2 = 300000)
  (h4 : ∀ (f d : ℝ), f * d^2 = f1 * d1^2) :
  ∃ (f2 : ℝ), f2 = 1 / 75 ∧ f2 * d2^2 = f1 * d1^2 := by
  sorry

end NUMINAMATH_CALUDE_gravitational_force_at_new_distance_l1637_163780


namespace NUMINAMATH_CALUDE_cos_4theta_from_complex_exp_l1637_163795

theorem cos_4theta_from_complex_exp (θ : ℝ) :
  Complex.exp (θ * Complex.I) = (3 + Complex.I * Real.sqrt 8) / 4 →
  Real.cos (4 * θ) = -287 / 256 := by
  sorry

end NUMINAMATH_CALUDE_cos_4theta_from_complex_exp_l1637_163795


namespace NUMINAMATH_CALUDE_f_odd_and_increasing_l1637_163718

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + x

-- State the theorem
theorem f_odd_and_increasing :
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x y, x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_f_odd_and_increasing_l1637_163718


namespace NUMINAMATH_CALUDE_tickets_to_be_sold_l1637_163779

def total_tickets : ℕ := 100
def jude_sales : ℕ := 16

def andrea_sales (jude_sales : ℕ) : ℕ := 2 * jude_sales

def sandra_sales (jude_sales : ℕ) : ℕ := jude_sales / 2 + 4

def total_sold (jude_sales : ℕ) : ℕ :=
  jude_sales + andrea_sales jude_sales + sandra_sales jude_sales

theorem tickets_to_be_sold :
  total_tickets - total_sold jude_sales = 40 :=
by sorry

end NUMINAMATH_CALUDE_tickets_to_be_sold_l1637_163779


namespace NUMINAMATH_CALUDE_squirrels_in_tree_l1637_163700

theorem squirrels_in_tree (nuts : ℕ) (squirrels : ℕ) 
  (h1 : nuts = 2)
  (h2 : squirrels - nuts = 2) :
  squirrels = 4 := by
  sorry

end NUMINAMATH_CALUDE_squirrels_in_tree_l1637_163700


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1637_163785

theorem arithmetic_calculation : 4 * (8 - 3) - 7 = 13 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1637_163785


namespace NUMINAMATH_CALUDE_odd_function_product_negative_l1637_163706

-- Define an odd function
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Theorem statement
theorem odd_function_product_negative
  (f : ℝ → ℝ)
  (h_odd : IsOdd f)
  (h_nonzero : ∀ x, f x ≠ 0) :
  ∀ x, f x * f (-x) < 0 :=
by sorry

end NUMINAMATH_CALUDE_odd_function_product_negative_l1637_163706


namespace NUMINAMATH_CALUDE_max_value_quadratic_l1637_163777

theorem max_value_quadratic :
  ∃ (M : ℝ), M = 19 ∧ ∀ p : ℝ, -3 * p^2 + 18 * p - 8 ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l1637_163777


namespace NUMINAMATH_CALUDE_julia_age_l1637_163764

-- Define the ages of the individuals
def Grace : ℕ := 20
def Helen : ℕ := Grace + 4
def Ian : ℕ := Helen - 5
def Julia : ℕ := Ian + 2

-- Theorem to prove
theorem julia_age : Julia = 21 := by
  sorry

end NUMINAMATH_CALUDE_julia_age_l1637_163764


namespace NUMINAMATH_CALUDE_product_base8_units_digit_l1637_163769

theorem product_base8_units_digit (a b : ℕ) (ha : a = 256) (hb : b = 72) :
  (a * b) % 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_base8_units_digit_l1637_163769


namespace NUMINAMATH_CALUDE_difference_divisible_by_nine_l1637_163791

def reverse (n : ℕ) : ℕ := sorry

theorem difference_divisible_by_nine (n : ℕ) : 
  ∃ k : ℤ, n - reverse n = 9 * k := by sorry

end NUMINAMATH_CALUDE_difference_divisible_by_nine_l1637_163791


namespace NUMINAMATH_CALUDE_ellipse_equation_and_product_constant_l1637_163730

noncomputable section

-- Define the ellipse
def Ellipse (a b : ℝ) := {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

-- Define eccentricity
def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - (b^2 / a^2))

-- Define the x-intercept of line QM
def x_intercept_QM (x₁ y₁ x₂ y₂ : ℝ) : ℝ := (x₂ * y₁ - x₁ * y₂) / (y₁ + y₂)

-- Define the y-intercept of line QN
def y_intercept_QN (x₁ y₁ x₂ y₂ : ℝ) : ℝ := (x₁ * y₂ + x₂ * y₁) / (x₁ - x₂)

-- Define the slope of line OR
def slope_OR (x₁ y₁ x₂ y₂ : ℝ) : ℝ := (y₁ + y₂) / (x₁ + x₂)

theorem ellipse_equation_and_product_constant (a b : ℝ) 
  (h₁ : a > b) (h₂ : b > 0) (h₃ : eccentricity a b = 1/2) :
  (∃ (x y : ℝ), Ellipse 2 (Real.sqrt 3) (x, y)) ∧ 
  (∀ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ → y₁ ≠ y₂ → 
    Ellipse a b (x₁, y₁) → Ellipse a b (x₂, y₂) →
    (x_intercept_QM x₁ y₁ x₂ y₂) * (y_intercept_QN x₁ y₁ x₂ y₂) * (slope_OR x₁ y₁ x₂ y₂) = 0) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_and_product_constant_l1637_163730


namespace NUMINAMATH_CALUDE_area_of_removed_triangles_l1637_163760

theorem area_of_removed_triangles (side_length : ℝ) (hypotenuse : ℝ) : 
  side_length = 16 → hypotenuse = 8 → 
  4 * (1/2 * (hypotenuse^2 / 2)) = 64 := by
  sorry

end NUMINAMATH_CALUDE_area_of_removed_triangles_l1637_163760


namespace NUMINAMATH_CALUDE_cos_180_eq_neg_one_l1637_163744

-- Define the rotation function
def rotate_180 (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

-- Define the cosine of 180 degrees
def cos_180 : ℝ := (rotate_180 (1, 0)).1

-- Theorem statement
theorem cos_180_eq_neg_one : cos_180 = -1 := by
  sorry

end NUMINAMATH_CALUDE_cos_180_eq_neg_one_l1637_163744


namespace NUMINAMATH_CALUDE_max_discount_theorem_l1637_163707

theorem max_discount_theorem (C : ℝ) : 
  C > 0 →                           -- Cost price is positive
  1.8 * C = 360 →                   -- Selling price is 80% above cost price and equals 360
  ∀ x : ℝ, 
    360 - x ≥ 1.3 * C →             -- Price after discount is at least 130% of cost price
    x ≤ 100 :=                      -- Maximum discount is 100
by
  sorry

end NUMINAMATH_CALUDE_max_discount_theorem_l1637_163707


namespace NUMINAMATH_CALUDE_aubree_animal_count_l1637_163768

def total_animals (initial_beavers : ℕ) (initial_chipmunks : ℕ) : ℕ :=
  let final_beavers := 2 * initial_beavers
  let final_chipmunks := initial_chipmunks - 10
  initial_beavers + initial_chipmunks + final_beavers + final_chipmunks

theorem aubree_animal_count :
  total_animals 20 40 = 130 := by
  sorry

end NUMINAMATH_CALUDE_aubree_animal_count_l1637_163768


namespace NUMINAMATH_CALUDE_milk_volume_is_ten_l1637_163729

/-- The total volume of milk sold by Josephine -/
def total_milk_volume : ℝ :=
  3 * 2 + 2 * 0.75 + 5 * 0.5

/-- Theorem stating that the total volume of milk sold is 10 liters -/
theorem milk_volume_is_ten : total_milk_volume = 10 := by
  sorry

end NUMINAMATH_CALUDE_milk_volume_is_ten_l1637_163729


namespace NUMINAMATH_CALUDE_smallest_number_with_2020_divisors_l1637_163767

/-- The number of divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- n is the smallest natural number with exactly k distinct divisors -/
def is_smallest_with_divisors (n k : ℕ) : Prop :=
  num_divisors n = k ∧ ∀ m < n, num_divisors m ≠ k

theorem smallest_number_with_2020_divisors :
  is_smallest_with_divisors (2^100 * 3^4 * 5 * 7) 2020 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_2020_divisors_l1637_163767


namespace NUMINAMATH_CALUDE_three_numbers_sum_l1637_163745

theorem three_numbers_sum (x y z : ℝ) : 
  x ≤ y ∧ y ≤ z →  -- x is the least, z is the greatest
  y = 9 →  -- median is 9
  (x + y + z) / 3 = x + 20 →  -- mean is 20 more than least
  (x + y + z) / 3 = z - 18 →  -- mean is 18 less than greatest
  x + y + z = 21 := by sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l1637_163745


namespace NUMINAMATH_CALUDE_riddle_solvable_l1637_163732

/-- Represents a jar filled with water --/
structure Jar :=
  (capacity : ℕ)
  (filled : Bool)

/-- Represents an attempt to "break" the jar --/
inductive Attempt
  | Think
  | Throw

/-- Represents the outcome of an attempt --/
inductive Outcome
  | Unbroken
  | Broken

/-- Represents the riddle solution --/
def solve_riddle (jar : Jar) (attempts : List Attempt) : Prop :=
  jar.capacity = 3 ∧ 
  jar.filled = true ∧
  attempts.length ≤ 3 ∧
  ∃ (outcome : Outcome), outcome = Outcome.Broken

/-- The main theorem stating that the riddle can be solved --/
theorem riddle_solvable : 
  ∃ (jar : Jar) (attempts : List Attempt), 
    solve_riddle jar attempts :=
sorry

end NUMINAMATH_CALUDE_riddle_solvable_l1637_163732


namespace NUMINAMATH_CALUDE_broadcast_methods_count_l1637_163740

/-- The number of different advertisements -/
def total_ads : ℕ := 5

/-- The number of commercial advertisements -/
def commercial_ads : ℕ := 3

/-- The number of Olympic promotional advertisements -/
def olympic_ads : ℕ := 2

/-- A function that calculates the number of ways to arrange the advertisements -/
def arrangement_count : ℕ :=
  Nat.factorial commercial_ads * Nat.choose 4 2

/-- Theorem stating that the number of different broadcasting methods is 36 -/
theorem broadcast_methods_count :
  arrangement_count = 36 :=
by sorry

end NUMINAMATH_CALUDE_broadcast_methods_count_l1637_163740


namespace NUMINAMATH_CALUDE_unique_f_two_l1637_163782

def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x * f y - f (x + y) = x * y

theorem unique_f_two (f : ℝ → ℝ) (h : functional_equation f) : f 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_f_two_l1637_163782


namespace NUMINAMATH_CALUDE_smallest_positive_difference_l1637_163773

/-- Vovochka's sum method for two three-digit numbers -/
def vovochkaSum (a b c d e f : ℕ) : ℕ :=
  1000 * (a + d) + 100 * (b + e) + (c + f)

/-- Correct sum for two three-digit numbers -/
def correctSum (a b c d e f : ℕ) : ℕ :=
  100 * (a + d) + 10 * (b + e) + (c + f)

/-- The difference between Vovochka's sum and the correct sum -/
def sumDifference (a b c d e f : ℕ) : ℤ :=
  (vovochkaSum a b c d e f : ℤ) - (correctSum a b c d e f : ℤ)

theorem smallest_positive_difference :
  ∀ a b c d e f : ℕ,
    a < 10 → b < 10 → c < 10 → d < 10 → e < 10 → f < 10 →
    (∃ a' b' c' d' e' f' : ℕ,
      a' < 10 ∧ b' < 10 ∧ c' < 10 ∧ d' < 10 ∧ e' < 10 ∧ f' < 10 ∧
      sumDifference a' b' c' d' e' f' > 0 ∧
      sumDifference a' b' c' d' e' f' ≤ sumDifference a b c d e f) →
    (∃ a' b' c' d' e' f' : ℕ,
      a' < 10 ∧ b' < 10 ∧ c' < 10 ∧ d' < 10 ∧ e' < 10 ∧ f' < 10 ∧
      sumDifference a' b' c' d' e' f' = 1800 ∧
      sumDifference a' b' c' d' e' f' > 0 ∧
      ∀ x y z u v w : ℕ,
        x < 10 → y < 10 → z < 10 → u < 10 → v < 10 → w < 10 →
        sumDifference x y z u v w > 0 →
        sumDifference a' b' c' d' e' f' ≤ sumDifference x y z u v w) :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_difference_l1637_163773


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l1637_163719

theorem cubic_equation_roots (p q : ℝ) : 
  (∃ a b c : ℕ+, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    (∀ x : ℝ, x^3 - 9*x^2 + p*x - q = 0 ↔ (x = a ∨ x = b ∨ x = c))) →
  p + q = 38 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l1637_163719


namespace NUMINAMATH_CALUDE_cube_root_equation_l1637_163759

theorem cube_root_equation (x : ℝ) : 
  (x * (x^5)^(1/4))^(1/3) = 5 ↔ x = 5 * 5^(1/3) :=
sorry

end NUMINAMATH_CALUDE_cube_root_equation_l1637_163759


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1637_163784

theorem partial_fraction_decomposition (A B C : ℚ) :
  (∀ x : ℚ, x^2 - 20 = A*(x+2)*(x-3) + B*(x-2)*(x-3) + C*(x-2)*(x+2)) →
  A * B * C = 2816 / 35 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1637_163784


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1637_163757

theorem max_value_of_expression (a b c : ℝ) 
  (nonneg_a : 0 ≤ a) (nonneg_b : 0 ≤ b) (nonneg_c : 0 ≤ c)
  (sum_squares : a^2 + b^2 + c^2 = 1) :
  2 * a * b + 2 * b * c * Real.sqrt 2 ≤ Real.sqrt (3/2) ∧ 
  ∃ a₀ b₀ c₀ : ℝ, 0 ≤ a₀ ∧ 0 ≤ b₀ ∧ 0 ≤ c₀ ∧ 
    a₀^2 + b₀^2 + c₀^2 = 1 ∧
    2 * a₀ * b₀ + 2 * b₀ * c₀ * Real.sqrt 2 = Real.sqrt (3/2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1637_163757


namespace NUMINAMATH_CALUDE_pair_probability_after_removal_l1637_163761

/-- Represents a deck of cards -/
structure Deck :=
  (size : ℕ)
  (num_fives : ℕ)
  (num_threes : ℕ)

/-- Calculates the number of ways to choose 2 cards from a deck -/
def choose_two (d : Deck) : ℕ := Nat.choose d.size 2

/-- Calculates the number of ways to form pairs in a deck -/
def num_pairs (d : Deck) : ℕ := d.num_fives * Nat.choose 5 2 + d.num_threes * Nat.choose 3 2

/-- The probability of selecting a pair from the deck -/
def pair_probability (d : Deck) : ℚ := (num_pairs d : ℚ) / (choose_two d : ℚ)

theorem pair_probability_after_removal :
  let d : Deck := ⟨46, 4, 2⟩
  pair_probability d = 46 / 1035 :=
sorry

end NUMINAMATH_CALUDE_pair_probability_after_removal_l1637_163761


namespace NUMINAMATH_CALUDE_min_value_theorem_l1637_163704

theorem min_value_theorem (a b : ℝ) (h : 2 * a - 3 * b + 6 = 0) :
  ∃ (min_val : ℝ), min_val = (1 / 4 : ℝ) ∧ ∀ (x : ℝ), 4^a + (1 / 8^b) ≥ x → x ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1637_163704


namespace NUMINAMATH_CALUDE_certain_number_problem_l1637_163749

theorem certain_number_problem (x : ℝ) (h : 5 * x - 28 = 232) : x = 52 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l1637_163749


namespace NUMINAMATH_CALUDE_treaty_signing_day_l1637_163746

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

def advance_day (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

def advance_days (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | Nat.succ m => advance_day (advance_days d m)

theorem treaty_signing_day :
  advance_days DayOfWeek.Wednesday 1566 = DayOfWeek.Wednesday :=
by sorry


end NUMINAMATH_CALUDE_treaty_signing_day_l1637_163746


namespace NUMINAMATH_CALUDE_no_linear_term_implies_m_equals_two_l1637_163726

theorem no_linear_term_implies_m_equals_two (m : ℝ) :
  (∀ x : ℝ, ∃ a b : ℝ, (x + m) * (2 - x) = a * x^2 + b) →
  m = 2 := by
  sorry

end NUMINAMATH_CALUDE_no_linear_term_implies_m_equals_two_l1637_163726


namespace NUMINAMATH_CALUDE_sin_135_degrees_l1637_163778

theorem sin_135_degrees : Real.sin (135 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_135_degrees_l1637_163778


namespace NUMINAMATH_CALUDE_count_numbers_satisfying_conditions_l1637_163799

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def starts_with_nine (n : ℕ) : Prop := ∃ (a b : ℕ), n = 900 + 90 * a + b ∧ 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9

def digit_sum (n : ℕ) : ℕ := 
  (n / 100) + ((n / 10) % 10) + (n % 10)

def satisfies_conditions (n : ℕ) : Prop :=
  is_three_digit n ∧
  starts_with_nine n ∧
  digit_sum n = 27 ∧
  Even n

theorem count_numbers_satisfying_conditions : 
  ∃! (s : Finset ℕ), (∀ n ∈ s, satisfies_conditions n) ∧ s.card = 3 :=
sorry

end NUMINAMATH_CALUDE_count_numbers_satisfying_conditions_l1637_163799


namespace NUMINAMATH_CALUDE_mowing_time_c_l1637_163775

-- Define the work rates
def work_rate (days : ℚ) : ℚ := 1 / days

-- Define the given conditions
def condition1 (a b : ℚ) : Prop := a + b = work_rate 28
def condition2 (a b c : ℚ) : Prop := a + b + c = work_rate 21

-- Theorem statement
theorem mowing_time_c (a b c : ℚ) 
  (h1 : condition1 a b) (h2 : condition2 a b c) : c = work_rate 84 := by
  sorry

end NUMINAMATH_CALUDE_mowing_time_c_l1637_163775


namespace NUMINAMATH_CALUDE_date_relationship_l1637_163766

/-- Represents the data of dates in a leap year -/
def dates : Finset ℕ := sorry

/-- The mean of the dates -/
def μ : ℚ := sorry

/-- The median of the dates -/
def M : ℕ := sorry

/-- The modes of the dates -/
def modes : Finset ℕ := sorry

/-- The median of the modes -/
def d : ℕ := sorry

/-- The number of occurrences for each date -/
def occurrences (n : ℕ) : ℕ := 
  if n ≤ 29 then 12
  else if n = 30 then 11
  else if n = 31 then 7
  else 0

theorem date_relationship : d < M ∧ (M : ℚ) < μ := by sorry

end NUMINAMATH_CALUDE_date_relationship_l1637_163766


namespace NUMINAMATH_CALUDE_symmetry_implies_line_equation_l1637_163734

-- Define the circles
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y + 4 = 0

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y + 2 = 0

-- Define symmetry with respect to a line
def symmetric_wrt_line (circle1 circle2 line : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ), circle1 x1 y1 ∧ circle2 x2 y2 ∧ 
  (∃ (x y : ℝ), line x y ∧ 
    ((x - x1)^2 + (y - y1)^2 = (x - x2)^2 + (y - y2)^2) ∧
    ((x1 + x2) / 2 = x) ∧ ((y1 + y2) / 2 = y))

-- Theorem statement
theorem symmetry_implies_line_equation : 
  symmetric_wrt_line circle_O circle_C line_l :=
sorry

end NUMINAMATH_CALUDE_symmetry_implies_line_equation_l1637_163734


namespace NUMINAMATH_CALUDE_abc_ordering_l1637_163701

theorem abc_ordering (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hdistinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) (heq : a^2 + c^2 = 2*b*c) :
  (b > a ∧ a > c) ∨ (c > b ∧ b > a) :=
sorry

end NUMINAMATH_CALUDE_abc_ordering_l1637_163701


namespace NUMINAMATH_CALUDE_inscribed_square_probability_l1637_163781

theorem inscribed_square_probability (r : ℝ) (h : r > 0) :
  let circle_area := π * r^2
  let square_side := r * Real.sqrt 2
  let square_area := square_side^2
  square_area / circle_area = 2 / π := by sorry

end NUMINAMATH_CALUDE_inscribed_square_probability_l1637_163781


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1637_163743

theorem min_value_of_expression (a : ℝ) (ha : a > 0) :
  (a + 1)^2 / a ≥ 4 ∧ ((a + 1)^2 / a = 4 ↔ a = 1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1637_163743


namespace NUMINAMATH_CALUDE_peregrine_falcon_problem_l1637_163790

/-- The percentage of pigeons eaten by peregrines -/
def percentage_eaten (initial_pigeons : ℕ) (chicks_per_pigeon : ℕ) (remaining_pigeons : ℕ) : ℚ :=
  let total_pigeons := initial_pigeons + initial_pigeons * chicks_per_pigeon
  let eaten_pigeons := total_pigeons - remaining_pigeons
  (eaten_pigeons : ℚ) / (total_pigeons : ℚ) * 100

theorem peregrine_falcon_problem :
  percentage_eaten 40 6 196 = 30 := by
  sorry

end NUMINAMATH_CALUDE_peregrine_falcon_problem_l1637_163790


namespace NUMINAMATH_CALUDE_larger_integer_value_l1637_163776

theorem larger_integer_value (a b : ℕ+) 
  (h_quotient : (a : ℚ) / (b : ℚ) = 7 / 3)
  (h_product : (a : ℕ) * (b : ℕ) = 189) :
  max a b = 21 := by
sorry

end NUMINAMATH_CALUDE_larger_integer_value_l1637_163776


namespace NUMINAMATH_CALUDE_prob_at_least_one_black_correct_l1637_163720

/-- The number of white balls in the bag -/
def white_balls : ℕ := 3

/-- The number of black balls in the bag -/
def black_balls : ℕ := 2

/-- The total number of balls in the bag -/
def total_balls : ℕ := white_balls + black_balls

/-- The number of balls drawn from the bag -/
def drawn_balls : ℕ := 2

/-- The probability of drawing at least one black ball -/
def prob_at_least_one_black : ℚ := 7 / 10

theorem prob_at_least_one_black_correct :
  (1 : ℚ) - (Nat.choose white_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ) = prob_at_least_one_black :=
by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_black_correct_l1637_163720


namespace NUMINAMATH_CALUDE_lindas_furniture_spending_l1637_163724

theorem lindas_furniture_spending (savings : ℚ) (tv_cost : ℚ) (furniture_fraction : ℚ) :
  savings = 840 →
  tv_cost = 210 →
  furniture_fraction * savings + tv_cost = savings →
  furniture_fraction = 3/4 := by
sorry

end NUMINAMATH_CALUDE_lindas_furniture_spending_l1637_163724


namespace NUMINAMATH_CALUDE_find_n_l1637_163723

theorem find_n (n : ℕ) : 
  (Nat.lcm n 14 = 56) → (Nat.gcd n 14 = 12) → n = 48 := by
  sorry

end NUMINAMATH_CALUDE_find_n_l1637_163723


namespace NUMINAMATH_CALUDE_quadratic_roots_to_coefficients_l1637_163787

theorem quadratic_roots_to_coefficients (b c : ℝ) : 
  (∀ x : ℝ, x^2 + b*x + c = 0 ↔ x = 2 ∨ x = -3) → 
  b = 1 ∧ c = -6 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_to_coefficients_l1637_163787


namespace NUMINAMATH_CALUDE_train_length_l1637_163728

/-- Given a train that can cross an electric pole in 120 seconds while traveling at 90 km/h,
    prove that its length is 3000 meters. -/
theorem train_length (crossing_time : ℝ) (speed_kmh : ℝ) (length : ℝ) : 
  crossing_time = 120 →
  speed_kmh = 90 →
  length = speed_kmh * (1000 / 3600) * crossing_time →
  length = 3000 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1637_163728


namespace NUMINAMATH_CALUDE_soccer_balls_per_class_l1637_163794

theorem soccer_balls_per_class 
  (num_schools : ℕ)
  (elementary_classes_per_school : ℕ)
  (middle_classes_per_school : ℕ)
  (total_soccer_balls : ℕ)
  (h1 : num_schools = 2)
  (h2 : elementary_classes_per_school = 4)
  (h3 : middle_classes_per_school = 5)
  (h4 : total_soccer_balls = 90) :
  total_soccer_balls / (num_schools * (elementary_classes_per_school + middle_classes_per_school)) = 5 :=
by sorry

end NUMINAMATH_CALUDE_soccer_balls_per_class_l1637_163794


namespace NUMINAMATH_CALUDE_brain_teaser_game_l1637_163765

theorem brain_teaser_game (total_participants : Nat) (total_questions : Nat) 
  (total_correct_answers : Nat) (exactly_six_correct : Nat) (exactly_eight_correct : Nat) :
  total_participants = 60 →
  total_questions = 10 →
  total_correct_answers = 452 →
  exactly_six_correct = 21 →
  exactly_eight_correct = 12 →
  (∀ participant, participant ≥ 6) →
  ∃ (exactly_seven_correct : Nat) (exactly_nine_correct : Nat) (exactly_ten_correct : Nat),
    exactly_seven_correct = exactly_nine_correct ∧
    exactly_six_correct + exactly_seven_correct + exactly_eight_correct + 
      exactly_nine_correct + exactly_ten_correct = total_participants ∧
    6 * exactly_six_correct + 7 * exactly_seven_correct + 8 * exactly_eight_correct +
      9 * exactly_nine_correct + 10 * exactly_ten_correct = total_correct_answers ∧
    exactly_ten_correct = 7 :=
by sorry

end NUMINAMATH_CALUDE_brain_teaser_game_l1637_163765


namespace NUMINAMATH_CALUDE_min_value_3x_plus_4y_l1637_163751

theorem min_value_3x_plus_4y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3*y = 5*x*y) :
  3*x + 4*y ≥ 5 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 3*y₀ = 5*x₀*y₀ ∧ 3*x₀ + 4*y₀ = 5 :=
sorry

end NUMINAMATH_CALUDE_min_value_3x_plus_4y_l1637_163751


namespace NUMINAMATH_CALUDE_max_ab_value_l1637_163735

/-- A function f with a parameter a and b -/
def f (a b x : ℝ) : ℝ := 4 * x^3 - a * x^2 - 2 * b * x + 2

/-- The derivative of f with respect to x -/
def f' (a b x : ℝ) : ℝ := 12 * x^2 - 2 * a * x - 2 * b

theorem max_ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_extremum : f' a b 1 = 0) : 
  ab ≤ 9 ∧ ∃ a b, a > 0 ∧ b > 0 ∧ f' a b 1 = 0 ∧ a * b = 9 := by
  sorry

end NUMINAMATH_CALUDE_max_ab_value_l1637_163735


namespace NUMINAMATH_CALUDE_sin_45_degrees_l1637_163708

theorem sin_45_degrees :
  Real.sin (π / 4) = 1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_45_degrees_l1637_163708


namespace NUMINAMATH_CALUDE_emily_candy_distribution_l1637_163722

/-- Given that Emily has 34 pieces of candy and 5 friends, prove that she needs to remove 4 pieces
    to distribute the remaining candies equally among her friends. -/
theorem emily_candy_distribution (total_candy : Nat) (num_friends : Nat) 
    (h1 : total_candy = 34) (h2 : num_friends = 5) :
    ∃ (removed : Nat) (distributed : Nat),
      removed = 4 ∧
      distributed * num_friends = total_candy - removed ∧
      ∀ r, r < removed → ¬∃ d, d * num_friends = total_candy - r :=
by sorry

end NUMINAMATH_CALUDE_emily_candy_distribution_l1637_163722


namespace NUMINAMATH_CALUDE_sunflower_majority_on_friday_friday_first_sunflower_majority_l1637_163792

/-- Represents the amount of sunflower seeds in the feeder on a given day -/
def sunflower_seeds (day : ℕ) : ℝ :=
  0.9 + (0.63 ^ (day - 1)) * 0.9

/-- Represents the total amount of seeds in the feeder on a given day -/
def total_seeds (day : ℕ) : ℝ :=
  3 * day

/-- The theorem states that on the 5th day, sunflower seeds exceed half of the total seeds -/
theorem sunflower_majority_on_friday :
  sunflower_seeds 5 > (total_seeds 5) / 2 := by
  sorry

/-- Helper function to check if sunflower seeds exceed half of total seeds on a given day -/
def is_sunflower_majority (day : ℕ) : Prop :=
  sunflower_seeds day > (total_seeds day) / 2

/-- The theorem states that Friday (day 5) is the first day when sunflower seeds exceed half of the total seeds -/
theorem friday_first_sunflower_majority :
  is_sunflower_majority 5 ∧ 
  (∀ d : ℕ, d < 5 → ¬is_sunflower_majority d) := by
  sorry

end NUMINAMATH_CALUDE_sunflower_majority_on_friday_friday_first_sunflower_majority_l1637_163792


namespace NUMINAMATH_CALUDE_lindsey_squat_weight_l1637_163714

/-- The total weight Lindsey will squat given exercise bands and a dumbbell -/
theorem lindsey_squat_weight (num_bands : ℕ) (band_resistance : ℕ) (dumbbell_weight : ℕ) : 
  num_bands = 2 →
  band_resistance = 5 →
  dumbbell_weight = 10 →
  num_bands * band_resistance + dumbbell_weight = 20 := by
  sorry

end NUMINAMATH_CALUDE_lindsey_squat_weight_l1637_163714


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1637_163712

theorem quadratic_equation_solution (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) 
  (h1 : c^2 + c*c + 2*d = 0) (h2 : d^2 + c*d + 2*d = 0) : 
  c = 2 ∧ d = -4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1637_163712


namespace NUMINAMATH_CALUDE_integer_expression_multiple_of_three_l1637_163758

theorem integer_expression_multiple_of_three (n k : ℕ) (h1 : 1 ≤ k) (h2 : k < n) (h3 : ∃ m : ℕ, n = 3 * m) :
  ∃ z : ℤ, (2 * n - 3 * k - 2) * (n.choose k) = (k + 2) * z := by
  sorry

end NUMINAMATH_CALUDE_integer_expression_multiple_of_three_l1637_163758


namespace NUMINAMATH_CALUDE_parabola_hyperbola_focus_coincide_l1637_163788

/-- The value of p for which the focus of the parabola x^2 = 2py (p > 0) 
    coincides with the focus of the hyperbola y^2/3 - x^2 = 1 -/
theorem parabola_hyperbola_focus_coincide : 
  ∃ p : ℝ, p > 0 ∧ 
  (∀ x y : ℝ, x^2 = 2*p*y ↔ (x, y) ∈ {(x, y) | x^2 = 2*p*y}) ∧
  (∀ x y : ℝ, y^2/3 - x^2 = 1 ↔ (x, y) ∈ {(x, y) | y^2/3 - x^2 = 1}) ∧
  (0, p/2) = (0, 2) ∧
  p = 4 :=
sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_focus_coincide_l1637_163788


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l1637_163742

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 - 2*x + 5 ≤ 0) ↔ (∃ x : ℝ, x^2 - 2*x + 5 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l1637_163742


namespace NUMINAMATH_CALUDE_tony_temp_day5_l1637_163763

-- Define the illnesses and their effects
structure Illness where
  duration : ℕ
  tempChange : ℤ
  startDay : ℕ

-- Define Tony's normal temperature and fever threshold
def normalTemp : ℕ := 95
def feverThreshold : ℕ := 100

-- Define the illnesses
def illnessA : Illness := ⟨7, 10, 1⟩
def illnessB : Illness := ⟨5, 4, 3⟩
def illnessC : Illness := ⟨3, -2, 5⟩

-- Function to calculate temperature change on a given day
def tempChangeOnDay (day : ℕ) : ℤ :=
  let baseChange := 
    (if day ≥ illnessA.startDay then illnessA.tempChange else 0) +
    (if day ≥ illnessB.startDay then 
      (if day ≥ illnessA.startDay then 2 * illnessB.tempChange else illnessB.tempChange)
    else 0) +
    (if day ≥ illnessC.startDay then illnessC.tempChange else 0)
  let synergisticEffect := if day = 5 then -3 else 0
  baseChange + synergisticEffect

-- Theorem to prove
theorem tony_temp_day5 : 
  (normalTemp : ℤ) + tempChangeOnDay 5 = 108 ∧ 
  (normalTemp : ℤ) + tempChangeOnDay 5 - feverThreshold = 8 := by
  sorry

end NUMINAMATH_CALUDE_tony_temp_day5_l1637_163763


namespace NUMINAMATH_CALUDE_calculation_proof_l1637_163705

theorem calculation_proof :
  (- (2^3 / 8) - (1/4 * (-2)^2) = -2) ∧
  ((-1/12 - 1/16 + 3/4 - 1/6) * (-48) = -21) := by
sorry

end NUMINAMATH_CALUDE_calculation_proof_l1637_163705


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l1637_163737

theorem geometric_sequence_product (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 → -- Ensuring a, b, c are positive
  (1 : ℝ) < a ∧ a < b ∧ b < c ∧ c < 256 → -- Ensuring the order of the sequence
  (b / a = a / 1) ∧ (c / b = b / a) ∧ (256 / c = c / b) → -- Geometric sequence condition
  1 * a * b * c * 256 = 2^20 := by
sorry


end NUMINAMATH_CALUDE_geometric_sequence_product_l1637_163737


namespace NUMINAMATH_CALUDE_sqrt_of_sqrt_plus_one_over_two_is_integer_l1637_163736

theorem sqrt_of_sqrt_plus_one_over_two_is_integer (n : ℕ+) 
  (h : ∃ (x : ℕ+), x^2 = 12 * n^2 + 1) :
  ∃ (q : ℕ+), q^2 = (Nat.sqrt (12 * n^2 + 1) + 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_sqrt_of_sqrt_plus_one_over_two_is_integer_l1637_163736


namespace NUMINAMATH_CALUDE_dubblefud_game_l1637_163783

theorem dubblefud_game (red_value blue_value green_value : ℕ)
  (total_product : ℕ) (red blue green : ℕ) :
  red_value = 3 →
  blue_value = 7 →
  green_value = 11 →
  total_product = 5764801 →
  blue = green →
  (red_value ^ red) * (blue_value ^ blue) * (green_value ^ green) = total_product →
  red = 7 := by
  sorry

end NUMINAMATH_CALUDE_dubblefud_game_l1637_163783


namespace NUMINAMATH_CALUDE_B_squared_equals_451_l1637_163772

/-- The function g defined as g(x) = √31 + 105/x -/
noncomputable def g (x : ℝ) : ℝ := Real.sqrt 31 + 105 / x

/-- The equation from the problem -/
def problem_equation (x : ℝ) : Prop :=
  x = g (g (g (g (g x))))

/-- The sum of absolute values of roots of the equation -/
noncomputable def B : ℝ :=
  abs ((Real.sqrt 31 + Real.sqrt 451) / 2) +
  abs ((Real.sqrt 31 - Real.sqrt 451) / 2)

/-- Theorem stating that B^2 equals 451 -/
theorem B_squared_equals_451 : B^2 = 451 := by
  sorry

end NUMINAMATH_CALUDE_B_squared_equals_451_l1637_163772


namespace NUMINAMATH_CALUDE_total_cups_is_twenty_l1637_163770

/-- Represents the number of cups of tea drunk by each merchant -/
structure Merchants where
  sosipatra : ℕ
  olympiada : ℕ
  poliksena : ℕ

/-- Defines the conditions given in the problem -/
def satisfies_conditions (m : Merchants) : Prop :=
  m.sosipatra + m.olympiada = 11 ∧
  m.olympiada + m.poliksena = 15 ∧
  m.sosipatra + m.poliksena = 14

/-- Theorem stating that the total number of cups is 20 -/
theorem total_cups_is_twenty (m : Merchants) (h : satisfies_conditions m) :
  m.sosipatra + m.olympiada + m.poliksena = 20 := by
  sorry

end NUMINAMATH_CALUDE_total_cups_is_twenty_l1637_163770


namespace NUMINAMATH_CALUDE_prop1_false_prop2_true_prop3_true_prop4_true_l1637_163750

-- Define the basic geometric objects
variable (Line Plane : Type)

-- Define the geometric relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Plane → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (line_parallel_to_plane : Line → Plane → Prop)
variable (line_perpendicular_to_plane : Line → Plane → Prop)
variable (lines_perpendicular : Line → Line → Prop)
variable (line_of_intersection : Plane → Plane → Line)

-- Proposition 1 (false)
theorem prop1_false :
  ¬(∀ (p1 p2 p3 : Plane) (l1 l2 : Line),
    line_in_plane l1 p1 → line_in_plane l2 p1 →
    line_parallel_to_plane l1 p2 → line_parallel_to_plane l2 p2 →
    parallel p1 p2) := sorry

-- Proposition 2 (true)
theorem prop2_true :
  ∀ (p1 p2 : Plane) (l : Line),
    line_perpendicular_to_plane l p1 →
    line_in_plane l p2 →
    perpendicular p1 p2 := sorry

-- Proposition 3 (true)
theorem prop3_true :
  ∀ (l1 l2 l3 : Line),
    lines_perpendicular l1 l3 →
    lines_perpendicular l2 l3 →
    lines_perpendicular l1 l2 := sorry

-- Proposition 4 (true)
theorem prop4_true :
  ∀ (p1 p2 : Plane) (l : Line),
    perpendicular p1 p2 →
    line_in_plane l p1 →
    ¬(lines_perpendicular l (line_of_intersection p1 p2)) →
    ¬(line_perpendicular_to_plane l p2) := sorry

end NUMINAMATH_CALUDE_prop1_false_prop2_true_prop3_true_prop4_true_l1637_163750


namespace NUMINAMATH_CALUDE_battery_life_comparison_l1637_163713

/-- Represents the charge of a battery -/
structure BatteryCharge where
  charge : ℝ
  positive : charge > 0

/-- Represents a clock powered by batteries -/
structure Clock where
  batteries : ℕ
  batteryType : BatteryCharge

/-- The problem statement -/
theorem battery_life_comparison 
  (battery_a battery_b : BatteryCharge)
  (clock_1 clock_2 : Clock)
  (h1 : battery_a.charge = 6 * battery_b.charge)
  (h2 : clock_1.batteries = 4 ∧ clock_1.batteryType = battery_a)
  (h3 : clock_2.batteries = 3 ∧ clock_2.batteryType = battery_b)
  (h4 : (clock_2.batteries : ℝ) * clock_2.batteryType.charge = 2)
  : (clock_1.batteries : ℝ) * clock_1.batteryType.charge - 
    (clock_2.batteries : ℝ) * clock_2.batteryType.charge = 14 := by
  sorry

#check battery_life_comparison

end NUMINAMATH_CALUDE_battery_life_comparison_l1637_163713


namespace NUMINAMATH_CALUDE_three_lines_two_intersections_l1637_163762

-- Define the lines
def line1 (x y : ℝ) : Prop := x + y + 1 = 0
def line2 (x y : ℝ) : Prop := 2*x - y + 8 = 0
def line3 (a x y : ℝ) : Prop := a*x + 3*y - 5 = 0

-- Define what it means for two points to be distinct
def distinct (p1 p2 : ℝ × ℝ) : Prop := p1 ≠ p2

-- Define what it means for a point to be on a line
def on_line1 (p : ℝ × ℝ) : Prop := line1 p.1 p.2
def on_line2 (p : ℝ × ℝ) : Prop := line2 p.1 p.2
def on_line3 (a : ℝ) (p : ℝ × ℝ) : Prop := line3 a p.1 p.2

-- Theorem statement
theorem three_lines_two_intersections (a : ℝ) :
  (∃ p1 p2 : ℝ × ℝ, distinct p1 p2 ∧ 
    on_line1 p1 ∧ on_line1 p2 ∧ 
    on_line2 p1 ∧ on_line2 p2 ∧ 
    on_line3 a p1 ∧ on_line3 a p2 ∧
    (∀ p3 : ℝ × ℝ, on_line1 p3 ∧ on_line2 p3 ∧ on_line3 a p3 → p3 = p1 ∨ p3 = p2)) →
  a = 3 ∨ a = -6 :=
sorry

end NUMINAMATH_CALUDE_three_lines_two_intersections_l1637_163762


namespace NUMINAMATH_CALUDE_village_population_proof_l1637_163753

theorem village_population_proof (P : ℕ) : 
  (0.85 : ℝ) * ((0.90 : ℝ) * P) = 6514 → P = 8518 := by sorry

end NUMINAMATH_CALUDE_village_population_proof_l1637_163753


namespace NUMINAMATH_CALUDE_cheryl_material_usage_l1637_163702

theorem cheryl_material_usage
  (material1 : ℚ)
  (material2 : ℚ)
  (leftover : ℚ)
  (h1 : material1 = 4 / 19)
  (h2 : material2 = 2 / 13)
  (h3 : leftover = 4 / 26)
  : material1 + material2 - leftover = 52 / 247 :=
by sorry

end NUMINAMATH_CALUDE_cheryl_material_usage_l1637_163702


namespace NUMINAMATH_CALUDE_chastity_initial_money_l1637_163711

def lollipop_cost : ℚ := 1.5
def gummies_cost : ℚ := 2
def lollipops_bought : ℕ := 4
def gummies_packs_bought : ℕ := 2
def money_left : ℚ := 5

def initial_money : ℚ := 15

theorem chastity_initial_money :
  initial_money = 
    (lollipop_cost * lollipops_bought + gummies_cost * gummies_packs_bought + money_left) :=
by sorry

end NUMINAMATH_CALUDE_chastity_initial_money_l1637_163711


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1637_163739

theorem imaginary_part_of_z (z : ℂ) : z * (1 - Complex.I) = Complex.abs (1 - Complex.I) + Complex.I →
  z.im = (Real.sqrt 2 + 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1637_163739


namespace NUMINAMATH_CALUDE_goblet_sphere_max_radius_l1637_163717

theorem goblet_sphere_max_radius :
  let goblet_cross_section := fun (x : ℝ) => x^4
  let sphere_in_goblet := fun (r : ℝ) (x y : ℝ) => y ≥ goblet_cross_section x ∧ (y - r)^2 + x^2 = r^2
  ∃ (max_r : ℝ), max_r = 3 / Real.rpow 2 (1/3) ∧
    (∀ r, r > 0 → sphere_in_goblet r 0 0 → r ≤ max_r) ∧
    sphere_in_goblet max_r 0 0 :=
sorry

end NUMINAMATH_CALUDE_goblet_sphere_max_radius_l1637_163717


namespace NUMINAMATH_CALUDE_unique_solution_logarithmic_equation_l1637_163721

theorem unique_solution_logarithmic_equation :
  ∃! (x y : ℝ), x > 0 ∧ y > 0 ∧ Real.log (x^3 + (1/3) * y^3 + 1/9) = Real.log x + Real.log y := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_logarithmic_equation_l1637_163721


namespace NUMINAMATH_CALUDE_sum_even_integers_100_to_200_l1637_163771

-- Define the sum of the first n positive even integers
def sumFirstNEvenIntegers (n : ℕ) : ℕ := n * (n + 1)

-- Define the sum of even integers from a to b inclusive
def sumEvenIntegersFromTo (a b : ℕ) : ℕ :=
  let n := (b - a) / 2 + 1
  n * (a + b) / 2

-- Theorem statement
theorem sum_even_integers_100_to_200 :
  sumFirstNEvenIntegers 50 = 2550 →
  sumEvenIntegersFromTo 100 200 = 7650 := by
  sorry

end NUMINAMATH_CALUDE_sum_even_integers_100_to_200_l1637_163771


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_probability_l1637_163752

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Calculate the area of a triangle -/
def triangleArea (t : Triangle) : ℝ :=
  sorry

/-- Check if a point is inside a triangle -/
def isInside (p : Point) (t : Triangle) : Prop :=
  sorry

/-- Calculate the probability of a point being in a specific region of a triangle -/
def probabilityInRegion (t : Triangle) (condition : Point → Bool) : ℝ :=
  sorry

/-- The main theorem to be proved -/
theorem isosceles_right_triangle_probability :
  let A : Point := ⟨0, 8⟩
  let B : Point := ⟨0, 0⟩
  let C : Point := ⟨8, 0⟩
  let ABC : Triangle := ⟨A, B, C⟩
  probabilityInRegion ABC (fun P => 
    triangleArea ⟨P, B, C⟩ < (1/3) * triangleArea ABC) = 7/32 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_probability_l1637_163752


namespace NUMINAMATH_CALUDE_soccer_league_games_l1637_163748

theorem soccer_league_games (n : ℕ) (h : n = 14) : (n * (n - 1)) / 2 = 91 := by
  sorry

end NUMINAMATH_CALUDE_soccer_league_games_l1637_163748


namespace NUMINAMATH_CALUDE_isabella_read_250_pages_l1637_163733

/-- The number of pages Isabella read in a week -/
def total_pages (pages_first_three : ℕ) (pages_next_three : ℕ) (pages_last_day : ℕ) : ℕ :=
  3 * pages_first_three + 3 * pages_next_three + pages_last_day

/-- Theorem stating that Isabella read 250 pages in total -/
theorem isabella_read_250_pages : 
  total_pages 36 44 10 = 250 := by
  sorry

#check isabella_read_250_pages

end NUMINAMATH_CALUDE_isabella_read_250_pages_l1637_163733


namespace NUMINAMATH_CALUDE_two_digit_reverse_difference_cube_l1637_163774

/-- A two-digit number -/
def TwoDigitNumber (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

/-- The reversed digits of a two-digit number -/
def reverseDigits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

/-- Predicate for a number being a positive perfect cube -/
def isPositivePerfectCube (n : ℕ) : Prop :=
  ∃ k : ℕ, k > 0 ∧ n = k^3

/-- The main theorem -/
theorem two_digit_reverse_difference_cube :
  ∃! (s : Finset ℕ),
    (∀ n ∈ s, TwoDigitNumber n ∧ isPositivePerfectCube (n - reverseDigits n)) ∧
    Finset.card s = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_reverse_difference_cube_l1637_163774


namespace NUMINAMATH_CALUDE_curve_through_center_l1637_163738

-- Define a square
structure Square where
  side : ℝ
  center : ℝ × ℝ

-- Define a curve
structure Curve where
  path : ℝ → ℝ × ℝ

-- Define the property of dividing the square into equal areas
def divides_equally (γ : Curve) (s : Square) : Prop :=
  ∃ (area1 area2 : ℝ), area1 = area2 ∧ area1 + area2 = s.side * s.side

-- Define the property of a line segment passing through a point
def passes_through (a b c : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ c = (1 - t) • a + t • b

-- The main theorem
theorem curve_through_center (s : Square) (γ : Curve) 
  (h : divides_equally γ s) :
  ∃ (a b : ℝ × ℝ), (∃ (t1 t2 : ℝ), γ.path t1 = a ∧ γ.path t2 = b) ∧ 
    passes_through a b s.center :=
sorry

end NUMINAMATH_CALUDE_curve_through_center_l1637_163738


namespace NUMINAMATH_CALUDE_line_intercepts_and_point_l1637_163709

/-- Given a line 3x + 5y + c = 0 where the sum of x- and y-intercepts is 16,
    prove that c = -30 and the point (2, 24/5) lies on the line. -/
theorem line_intercepts_and_point (c : ℝ) : 
  (∃ (x y : ℝ), 3*x + 5*y + c = 0 ∧ x + y = 16) → 
  (c = -30 ∧ 3*2 + 5*(24/5) + c = 0) :=
by sorry

end NUMINAMATH_CALUDE_line_intercepts_and_point_l1637_163709


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1637_163715

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁ = 6 ∧ x₂ = -2) ∧ 
  (x₁^2 - 4*x₁ = 12) ∧ 
  (x₂^2 - 4*x₂ = 12) ∧
  (∀ x : ℝ, x^2 - 4*x = 12 → x = x₁ ∨ x = x₂) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1637_163715


namespace NUMINAMATH_CALUDE_x_value_l1637_163793

theorem x_value : ∃ x : ℝ, (3 * x = (20 - x) + 20) ∧ (x = 10) := by
  sorry

end NUMINAMATH_CALUDE_x_value_l1637_163793


namespace NUMINAMATH_CALUDE_equidistant_function_b_squared_l1637_163703

/-- A complex function that is equidistant from z and the origin -/
def equidistant_function (a b : ℝ) : ℂ → ℂ := fun z ↦ (a + b * Complex.I) * z

/-- The property that f(z) is equidistant from z and the origin for all z -/
def is_equidistant (f : ℂ → ℂ) : Prop :=
  ∀ z : ℂ, Complex.abs (f z - z) = Complex.abs (f z)

theorem equidistant_function_b_squared
  (a b : ℝ)
  (h_pos_a : a > 0)
  (h_pos_b : b > 0)
  (h_equidistant : is_equidistant (equidistant_function a b))
  (h_norm : Complex.abs (a + b * Complex.I) = 5) :
  b^2 = 99/4 := by
  sorry

end NUMINAMATH_CALUDE_equidistant_function_b_squared_l1637_163703


namespace NUMINAMATH_CALUDE_lapis_share_is_correct_l1637_163731

-- Define the problem parameters
def total_treasure : ℚ := 900000
def fonzie_contribution : ℚ := 7000
def aunt_bee_contribution : ℚ := 8000
def lapis_contribution : ℚ := 9000

-- Define Lapis's share calculation
def lapis_share : ℚ :=
  (lapis_contribution / (fonzie_contribution + aunt_bee_contribution + lapis_contribution)) * total_treasure

-- Theorem statement
theorem lapis_share_is_correct : lapis_share = 337500 := by
  sorry

end NUMINAMATH_CALUDE_lapis_share_is_correct_l1637_163731


namespace NUMINAMATH_CALUDE_exists_circuit_with_rational_resistance_l1637_163716

/-- Represents an electrical circuit composed of unit resistances -/
inductive Circuit
  | unit : Circuit
  | series : Circuit → Circuit → Circuit
  | parallel : Circuit → Circuit → Circuit

/-- Calculates the resistance of a circuit -/
def resistance : Circuit → ℚ
  | Circuit.unit => 1
  | Circuit.series c1 c2 => resistance c1 + resistance c2
  | Circuit.parallel c1 c2 => 1 / (1 / resistance c1 + 1 / resistance c2)

/-- Theorem: For any rational number a/b (where a and b are positive integers),
    there exists an electrical circuit composed of unit resistances
    whose total resistance is equal to a/b -/
theorem exists_circuit_with_rational_resistance (a b : ℕ) (h : b > 0) :
  ∃ c : Circuit, resistance c = a / b := by sorry

end NUMINAMATH_CALUDE_exists_circuit_with_rational_resistance_l1637_163716
