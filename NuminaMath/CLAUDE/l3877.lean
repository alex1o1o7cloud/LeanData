import Mathlib

namespace NUMINAMATH_CALUDE_eustace_age_in_three_years_l3877_387785

/-- Proves that Eustace will be 39 years old in 3 years, given the conditions -/
theorem eustace_age_in_three_years
  (eustace_age : ℕ)
  (milford_age : ℕ)
  (h1 : eustace_age = 2 * milford_age)
  (h2 : milford_age + 3 = 21) :
  eustace_age + 3 = 39 := by
  sorry

end NUMINAMATH_CALUDE_eustace_age_in_three_years_l3877_387785


namespace NUMINAMATH_CALUDE_solve_bowling_problem_l3877_387761

def bowling_problem (score1 score2 average : ℕ) : Prop :=
  ∃ score3 : ℕ, 
    (score1 + score2 + score3) / 3 = average ∧
    score3 = 3 * average - score1 - score2

theorem solve_bowling_problem : 
  bowling_problem 113 85 106 → ∃ score3 : ℕ, score3 = 120 := by
  sorry

#check solve_bowling_problem

end NUMINAMATH_CALUDE_solve_bowling_problem_l3877_387761


namespace NUMINAMATH_CALUDE_mans_age_to_sons_age_ratio_l3877_387773

/-- Proves that the ratio of a man's age to his son's age in two years is 2:1,
    given that the man is 37 years older than his son and the son's current age is 35. -/
theorem mans_age_to_sons_age_ratio :
  let sons_current_age : ℕ := 35
  let mans_current_age : ℕ := sons_current_age + 37
  let sons_age_in_two_years : ℕ := sons_current_age + 2
  let mans_age_in_two_years : ℕ := mans_current_age + 2
  (mans_age_in_two_years : ℚ) / (sons_age_in_two_years : ℚ) = 2 := by
sorry

end NUMINAMATH_CALUDE_mans_age_to_sons_age_ratio_l3877_387773


namespace NUMINAMATH_CALUDE_four_integers_average_l3877_387740

theorem four_integers_average (a b c d : ℕ+) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (a + b + c + d : ℚ) / 4 = 5 →
  ∀ w x y z : ℕ+, w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z →
  (w + x + y + z : ℚ) / 4 = 5 →
  (max a (max b (max c d)) - min a (min b (min c d)) : ℤ) ≥ 
  (max w (max x (max y z)) - min w (min x (min y z)) : ℤ) →
  (a + b + c + d - max a (max b (max c d)) - min a (min b (min c d)) : ℚ) / 2 = 5/2 :=
by sorry

end NUMINAMATH_CALUDE_four_integers_average_l3877_387740


namespace NUMINAMATH_CALUDE_estate_distribution_l3877_387709

theorem estate_distribution (a b c d : ℝ) : 
  a > 0 ∧ 
  b = 1.20 * a ∧ 
  c = 1.20 * b ∧ 
  d = 1.20 * c ∧ 
  d - a = 19520 →
  (b = 32176 ∨ c = 32176 ∨ d = 32176) := by
sorry

end NUMINAMATH_CALUDE_estate_distribution_l3877_387709


namespace NUMINAMATH_CALUDE_two_books_from_different_genres_l3877_387700

/-- The number of ways to choose two books from different genres -/
def choose_two_books (mystery : ℕ) (fantasy : ℕ) (biography : ℕ) : ℕ :=
  mystery * fantasy + mystery * biography + fantasy * biography

/-- Theorem: Given 5 mystery novels, 3 fantasy novels, and 2 biographies,
    the number of ways to choose 2 books from different genres is 31 -/
theorem two_books_from_different_genres :
  choose_two_books 5 3 2 = 31 := by
  sorry

end NUMINAMATH_CALUDE_two_books_from_different_genres_l3877_387700


namespace NUMINAMATH_CALUDE_total_distance_right_triangle_l3877_387798

/-- The total distance traveled in a right-angled triangle XYZ -/
theorem total_distance_right_triangle (XZ YZ XY : ℝ) : 
  XZ = 4000 →
  XY = 5000 →
  XZ^2 + YZ^2 = XY^2 →
  XZ + YZ + XY = 12000 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_right_triangle_l3877_387798


namespace NUMINAMATH_CALUDE_tan_beta_value_l3877_387704

theorem tan_beta_value (α β : Real) 
  (h1 : (Real.sin α * Real.cos α) / (1 - Real.cos (2 * α)) = 1)
  (h2 : Real.tan (α - β) = 1/3) : 
  Real.tan β = 1/7 := by
sorry

end NUMINAMATH_CALUDE_tan_beta_value_l3877_387704


namespace NUMINAMATH_CALUDE_continued_fraction_result_l3877_387755

-- Define the continued fraction representation of x
noncomputable def x : ℝ := 2 + Real.sqrt 3 / (2 + Real.sqrt 3 / (2 + Real.sqrt 3 / (2 + Real.sqrt 3 / 2)))

-- State the theorem
theorem continued_fraction_result :
  1 / ((x + 1) * (x - 3)) = (3 + Real.sqrt 3) / (-6) :=
sorry

end NUMINAMATH_CALUDE_continued_fraction_result_l3877_387755


namespace NUMINAMATH_CALUDE_range_of_a_l3877_387745

-- Define the conditions P and Q
def P (a : ℝ) : Prop := ∀ x y : ℝ, ∃ k : ℝ, k > 0 ∧ x^2 / (3 - a) + y^2 / (1 + a) = k

def Q (a : ℝ) : Prop := ∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0

-- Theorem statement
theorem range_of_a (a : ℝ) (hP : P a) (hQ : Q a) : 
  -1 < a ∧ a ≤ 2 ∧ a ≠ 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3877_387745


namespace NUMINAMATH_CALUDE_root_magnitude_of_quadratic_l3877_387735

theorem root_magnitude_of_quadratic (z : ℂ) : z^2 + z + 1 = 0 → Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_root_magnitude_of_quadratic_l3877_387735


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_equals_one_l3877_387779

-- Define the lines l₁ and l₂
def l₁ (a : ℝ) : ℝ × ℝ → Prop := λ p => a * p.1 + p.2 + 2 = 0
def l₂ (a : ℝ) : ℝ × ℝ → Prop := λ p => p.1 + (a - 2) * p.2 + 1 = 0

-- Define perpendicularity of lines
def perpendicular (f g : ℝ × ℝ → Prop) : Prop :=
  ∃ (m₁ m₂ : ℝ), (∀ p, f p ↔ p.2 = m₁ * p.1 + 0) ∧
                 (∀ p, g p ↔ p.2 = m₂ * p.1 + 0) ∧
                 m₁ * m₂ = -1

-- State the theorem
theorem perpendicular_lines_a_equals_one :
  perpendicular (l₁ a) (l₂ a) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_equals_one_l3877_387779


namespace NUMINAMATH_CALUDE_draws_calculation_l3877_387702

def total_games : ℕ := 14
def wins : ℕ := 2
def losses : ℕ := 2

theorem draws_calculation : total_games - (wins + losses) = 10 := by
  sorry

end NUMINAMATH_CALUDE_draws_calculation_l3877_387702


namespace NUMINAMATH_CALUDE_set_intersection_equality_l3877_387796

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | |x - 1| > 2}

-- Define set B
def B : Set ℝ := {x | x^2 - 6*x + 8 < 0}

-- Define the open interval (2, 3]
def open_interval : Set ℝ := Set.Ioc 2 3

-- Theorem statement
theorem set_intersection_equality :
  (Set.compl A ∩ B) = open_interval :=
sorry

end NUMINAMATH_CALUDE_set_intersection_equality_l3877_387796


namespace NUMINAMATH_CALUDE_log_relation_l3877_387713

theorem log_relation (y : ℝ) (m : ℝ) : 
  Real.log 5 / Real.log 9 = y → Real.log 125 / Real.log 3 = m * y → m = 6 := by
  sorry

end NUMINAMATH_CALUDE_log_relation_l3877_387713


namespace NUMINAMATH_CALUDE_like_terms_power_l3877_387764

theorem like_terms_power (m n : ℕ) : 
  (∃ (x y : ℝ), 2 * x^(m-1) * y^2 = -2 * x^2 * y^n) → 
  (-m : ℤ)^n = 9 := by
sorry

end NUMINAMATH_CALUDE_like_terms_power_l3877_387764


namespace NUMINAMATH_CALUDE_bake_sale_theorem_l3877_387727

/-- Represents the types of baked goods sold at the bake sale -/
inductive BakedGood
  | Cookie
  | Brownie
  | Cupcake
  | CakeSlice

/-- Represents the bake sale data -/
structure BakeSale where
  totalItems : Nat
  cookiesSold : Nat
  browniesSold : Nat
  cupcakesSold : Nat
  cookiePrice : Rat
  browniePrice : Rat
  cupcakePrice : Rat
  cakeSlicePrice : Rat

def bakeSaleData : BakeSale :=
  { totalItems := 250
  , cookiesSold := 50
  , browniesSold := 80
  , cupcakesSold := 60
  , cookiePrice := 3/2
  , browniePrice := 2
  , cupcakePrice := 5/2
  , cakeSlicePrice := 3
  }

/-- Calculates the number of cake slices sold -/
def cakeSlicesSold (sale : BakeSale) : Nat :=
  sale.totalItems - (sale.cookiesSold + sale.browniesSold + sale.cupcakesSold)

/-- Calculates the revenue for a specific baked good -/
def revenue (sale : BakeSale) (good : BakedGood) : Rat :=
  match good with
  | BakedGood.Cookie => sale.cookiesSold * sale.cookiePrice
  | BakedGood.Brownie => sale.browniesSold * sale.browniePrice
  | BakedGood.Cupcake => sale.cupcakesSold * sale.cupcakePrice
  | BakedGood.CakeSlice => cakeSlicesSold sale * sale.cakeSlicePrice

/-- Theorem stating the ratio of items sold and the revenue for each type -/
theorem bake_sale_theorem (sale : BakeSale) : 
  (sale.cookiesSold : Rat) / 10 = 5 ∧
  (sale.browniesSold : Rat) / 10 = 8 ∧
  (sale.cupcakesSold : Rat) / 10 = 6 ∧
  (cakeSlicesSold sale : Rat) / 10 = 6 ∧
  revenue sale BakedGood.Cookie = 75 ∧
  revenue sale BakedGood.Brownie = 160 ∧
  revenue sale BakedGood.Cupcake = 150 ∧
  revenue sale BakedGood.CakeSlice = 180 :=
by sorry

end NUMINAMATH_CALUDE_bake_sale_theorem_l3877_387727


namespace NUMINAMATH_CALUDE_travelers_checks_worth_l3877_387754

/-- Represents the total worth of travelers checks -/
def total_worth (num_50 : ℕ) (num_100 : ℕ) : ℕ :=
  50 * num_50 + 100 * num_100

/-- Represents the average value of remaining checks after spending some $50 checks -/
def average_remaining (num_50 : ℕ) (num_100 : ℕ) (spent_50 : ℕ) : ℚ :=
  (50 * (num_50 - spent_50) + 100 * num_100) / (num_50 + num_100 - spent_50)

theorem travelers_checks_worth :
  ∀ (num_50 num_100 : ℕ),
    num_50 + num_100 = 30 →
    average_remaining num_50 num_100 15 = 70 →
    total_worth num_50 num_100 = 1800 :=
by sorry

end NUMINAMATH_CALUDE_travelers_checks_worth_l3877_387754


namespace NUMINAMATH_CALUDE_x_intercept_of_parallel_lines_l3877_387759

/-- Two lines are parallel if their slopes are equal -/
def parallel (m1 m2 : ℚ) : Prop := m1 = m2

/-- Line l1 with slope m1 and y-intercept b1 -/
def line1 (x y : ℚ) (m1 b1 : ℚ) : Prop := y = m1 * x + b1

/-- Line l2 with slope m2 and y-intercept b2 -/
def line2 (x y : ℚ) (m2 b2 : ℚ) : Prop := y = m2 * x + b2

/-- The x-intercept of a line with slope m and y-intercept b -/
def x_intercept (m b : ℚ) : ℚ := -b / m

theorem x_intercept_of_parallel_lines 
  (a : ℚ) 
  (h_parallel : parallel (-(a+2)/3) (-(a-1)/2)) : 
  x_intercept (-(a+2)/3) (5/3) = 5/9 := by sorry

end NUMINAMATH_CALUDE_x_intercept_of_parallel_lines_l3877_387759


namespace NUMINAMATH_CALUDE_sqrt_three_irrational_l3877_387714

theorem sqrt_three_irrational : Irrational (Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_irrational_l3877_387714


namespace NUMINAMATH_CALUDE_total_cost_is_18_l3877_387728

/-- The cost of items in Sean's purchase --/
def CostCalculation (soda_price : ℝ) : Prop :=
  let soda_quantity : ℕ := 3
  let soup_quantity : ℕ := 2
  let sandwich_quantity : ℕ := 1
  let soup_price : ℝ := soda_price * soda_quantity
  let sandwich_price : ℝ := 3 * soup_price
  let total_cost : ℝ := soda_price * soda_quantity + soup_price * soup_quantity + sandwich_price * sandwich_quantity
  total_cost = 18

/-- Theorem stating that the total cost is $18 given the conditions --/
theorem total_cost_is_18 : CostCalculation 1 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_18_l3877_387728


namespace NUMINAMATH_CALUDE_divisible_by_3_or_5_count_l3877_387708

def count_divisible (n : Nat) : Nat :=
  (n / 3) + (n / 5) - (n / 15)

theorem divisible_by_3_or_5_count : count_divisible 46 = 21 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_3_or_5_count_l3877_387708


namespace NUMINAMATH_CALUDE_smallest_iteration_for_three_l3877_387791

def f (x : ℕ) : ℕ :=
  if x % 2 = 0 ∧ x % 7 = 0 then x / 14
  else if x % 7 = 0 then 2 * x
  else if x % 2 = 0 then 7 * x
  else x + 2

def f_iter (n : ℕ) (x : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n + 1 => f (f_iter n x)

theorem smallest_iteration_for_three :
  (∀ a : ℕ, 1 < a → a < 6 → f_iter a 3 ≠ f 3) ∧
  f_iter 6 3 = f 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_iteration_for_three_l3877_387791


namespace NUMINAMATH_CALUDE_complex_symmetry_division_l3877_387719

/-- Two complex numbers are symmetric about the imaginary axis if their real parts are negatives of each other and their imaginary parts are equal. -/
def symmetric_about_imaginary_axis (z₁ z₂ : ℂ) : Prop :=
  z₁.re = -z₂.re ∧ z₁.im = z₂.im

/-- The main theorem: If z₁ and z₂ are symmetric about the imaginary axis and z₁ = -1 + i, then z₁ / z₂ = i -/
theorem complex_symmetry_division (z₁ z₂ : ℂ) 
  (h_sym : symmetric_about_imaginary_axis z₁ z₂) 
  (h_z₁ : z₁ = -1 + Complex.I) : 
  z₁ / z₂ = Complex.I :=
sorry

end NUMINAMATH_CALUDE_complex_symmetry_division_l3877_387719


namespace NUMINAMATH_CALUDE_optimal_probability_l3877_387712

-- Define the probability of success for a single shot
variable (p : ℝ)

-- Define the number of successful shots as a random variable
def X : ℕ → ℝ
  | n => p^n * (1 - p)

-- Define the probability of making between 35 and 69 shots
def P_35_to_69 (p : ℝ) : ℝ :=
  p^35 - p^70

-- State the theorem
theorem optimal_probability :
  ∃ (p : ℝ), p > 0 ∧ p < 1 ∧
  (∀ q : ℝ, q > 0 → q < 1 → P_35_to_69 q ≤ P_35_to_69 p) ∧
  p = (1/2)^(1/35) :=
sorry

end NUMINAMATH_CALUDE_optimal_probability_l3877_387712


namespace NUMINAMATH_CALUDE_described_loop_is_while_loop_l3877_387741

/-- Represents a generic loop structure -/
structure LoopStructure :=
  (condition_evaluation : Bool)
  (execution_order : Bool)

/-- Defines a While loop structure -/
def is_while_loop (loop : LoopStructure) : Prop :=
  loop.condition_evaluation ∧ loop.execution_order

/-- Theorem stating that the described loop structure is a While loop -/
theorem described_loop_is_while_loop :
  ∀ (loop : LoopStructure),
  loop.condition_evaluation = true ∧
  loop.execution_order = true →
  is_while_loop loop :=
by
  sorry

#check described_loop_is_while_loop

end NUMINAMATH_CALUDE_described_loop_is_while_loop_l3877_387741


namespace NUMINAMATH_CALUDE_combined_mpg_l3877_387749

/-- The combined miles per gallon of two cars given their individual mpg and relative distances driven -/
theorem combined_mpg (ray_mpg tom_mpg : ℝ) (h1 : ray_mpg = 48) (h2 : tom_mpg = 24) : 
  let s : ℝ := 1  -- Tom's distance (arbitrary non-zero value)
  let ray_distance := 2 * s
  let tom_distance := s
  let total_distance := ray_distance + tom_distance
  let total_fuel := ray_distance / ray_mpg + tom_distance / tom_mpg
  total_distance / total_fuel = 36 := by
sorry


end NUMINAMATH_CALUDE_combined_mpg_l3877_387749


namespace NUMINAMATH_CALUDE_complex_on_imaginary_axis_l3877_387707

theorem complex_on_imaginary_axis (a : ℝ) : 
  let z : ℂ := (a^2 - 2*a : ℝ) + (a^2 - a - 2 : ℝ) * I
  (z.re = 0) → (a = 0 ∨ a = 2) :=
by sorry

end NUMINAMATH_CALUDE_complex_on_imaginary_axis_l3877_387707


namespace NUMINAMATH_CALUDE_pyramid_volume_from_rectangle_l3877_387756

/-- The volume of a pyramid formed from a rectangle with specific dimensions -/
theorem pyramid_volume_from_rectangle (AB BC : ℝ) (h : AB = 15 * Real.sqrt 2 ∧ BC = 17 * Real.sqrt 2) :
  let P : ℝ × ℝ × ℝ := (15 * Real.sqrt 2 / 2, 17 * Real.sqrt 2 / 2, Real.sqrt 257)
  let base_area : ℝ := (1 / 2) * AB * BC
  let volume : ℝ := (1 / 3) * base_area * P.2.2
  volume = 85 * Real.sqrt 257 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_volume_from_rectangle_l3877_387756


namespace NUMINAMATH_CALUDE_square_difference_l3877_387763

theorem square_difference : (30 : ℕ)^2 - (29 : ℕ)^2 = 59 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l3877_387763


namespace NUMINAMATH_CALUDE_valid_coloring_iff_odd_l3877_387716

/-- A valid coloring of an n-gon satisfies the given conditions --/
def ValidColoring (n : ℕ) (P : Set (Fin n)) (coloring : Fin n → Fin n → Fin n) : Prop :=
  -- P represents the vertices of the n-gon
  (∀ i j : Fin n, coloring i j < n) ∧ 
  -- For any three distinct colors, there exists a triangle with those colors
  (∀ c₁ c₂ c₃ : Fin n, c₁ ≠ c₂ ∧ c₂ ≠ c₃ ∧ c₁ ≠ c₃ → 
    ∃ i j k : Fin n, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
      coloring i j = c₁ ∧ coloring j k = c₂ ∧ coloring i k = c₃)

/-- A valid coloring of an n-gon exists if and only if n is odd --/
theorem valid_coloring_iff_odd (n : ℕ) :
  (∃ P : Set (Fin n), ∃ coloring : Fin n → Fin n → Fin n, ValidColoring n P coloring) ↔ Odd n :=
sorry

end NUMINAMATH_CALUDE_valid_coloring_iff_odd_l3877_387716


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3877_387751

theorem pure_imaginary_complex_number (m : ℝ) : 
  (m^2 - 3*m = 0) ∧ (m^2 - 5*m + 6 ≠ 0) → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3877_387751


namespace NUMINAMATH_CALUDE_greatest_possible_large_chips_l3877_387747

/-- Represents the number of chips in the box -/
def total_chips : ℕ := 60

/-- Represents the number of large chips -/
def large_chips : ℕ := 29

/-- Represents the number of small chips -/
def small_chips : ℕ := total_chips - large_chips

/-- Represents the difference between small and large chips -/
def difference : ℕ := small_chips - large_chips

theorem greatest_possible_large_chips :
  (total_chips = small_chips + large_chips) ∧
  (∃ p : ℕ, Nat.Prime p ∧ small_chips = large_chips + p ∧ p ∣ large_chips) ∧
  (∀ l : ℕ, l > large_chips →
    ¬(∃ p : ℕ, Nat.Prime p ∧ (total_chips - l) = l + p ∧ p ∣ l)) :=
by sorry

#eval large_chips -- Should output 29
#eval small_chips -- Should output 31
#eval difference -- Should output 2

end NUMINAMATH_CALUDE_greatest_possible_large_chips_l3877_387747


namespace NUMINAMATH_CALUDE_operation_result_l3877_387790

theorem operation_result (x : ℝ) : 40 + 5 * x / (180 / 3) = 41 → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_operation_result_l3877_387790


namespace NUMINAMATH_CALUDE_water_percentage_in_dried_grapes_l3877_387757

/-- 
Given:
- Fresh grapes contain 90% water by weight
- 25 kg of fresh grapes yield 3.125 kg of dried grapes

Prove that the percentage of water in dried grapes is 20%
-/
theorem water_percentage_in_dried_grapes :
  let fresh_grape_weight : ℝ := 25
  let dried_grape_weight : ℝ := 3.125
  let fresh_water_percentage : ℝ := 90
  let dried_water_percentage : ℝ := (dried_grape_weight - (fresh_grape_weight * (1 - fresh_water_percentage / 100))) / dried_grape_weight * 100
  dried_water_percentage = 20 := by sorry

end NUMINAMATH_CALUDE_water_percentage_in_dried_grapes_l3877_387757


namespace NUMINAMATH_CALUDE_oz_language_lost_words_l3877_387789

/-- Represents the number of letters in the Oz alphabet -/
def alphabet_size : ℕ := 65

/-- Represents the number of letters in a word (either 1 or 2) -/
def word_length : Fin 2 → ℕ
| 0 => 1
| 1 => 2

/-- Calculates the number of words lost when one letter is forbidden -/
def words_lost (n : ℕ) : ℕ :=
  1 + n + n

/-- Theorem stating that forbidding one letter in the Oz language results in 131 lost words -/
theorem oz_language_lost_words :
  words_lost alphabet_size = 131 := by
  sorry

end NUMINAMATH_CALUDE_oz_language_lost_words_l3877_387789


namespace NUMINAMATH_CALUDE_triangle_angle_C_l3877_387720

theorem triangle_angle_C (a b : ℝ) (A : ℝ) :
  a = 1 →
  b = Real.sqrt 2 →
  2 * Real.sin A * (Real.cos (π / 4))^2 + Real.cos A * Real.sin (π / 2) - Real.sin A = 3 / 2 →
  ∃ (C : ℝ), (C = 7 * π / 12 ∨ C = π / 12) ∧ 
  (∃ (B : ℝ), A + B + C = π ∧ Real.sin A / a = Real.sin B / b) :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_C_l3877_387720


namespace NUMINAMATH_CALUDE_school_play_girls_l3877_387780

/-- The number of girls in the school play -/
def num_girls : ℕ := 6

/-- The number of boys in the school play -/
def num_boys : ℕ := 8

/-- The total number of parents attending the premiere -/
def total_parents : ℕ := 28

/-- The number of parents per child -/
def parents_per_child : ℕ := 2

theorem school_play_girls :
  num_girls = 6 ∧
  num_boys * parents_per_child + num_girls * parents_per_child = total_parents :=
sorry

end NUMINAMATH_CALUDE_school_play_girls_l3877_387780


namespace NUMINAMATH_CALUDE_sin_585_degrees_l3877_387775

theorem sin_585_degrees : Real.sin (585 * π / 180) = - Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_585_degrees_l3877_387775


namespace NUMINAMATH_CALUDE_grisha_remaining_money_l3877_387734

-- Define the given constants
def initial_money : ℕ := 5000
def bunny_price : ℕ := 45
def bag_price : ℕ := 30
def bunnies_per_bag : ℕ := 30

-- Define the function to calculate the remaining money
def remaining_money : ℕ :=
  let full_bag_cost := bag_price + bunnies_per_bag * bunny_price
  let full_bags := initial_money / full_bag_cost
  let money_after_full_bags := initial_money - full_bags * full_bag_cost
  let additional_bag_cost := bag_price
  let money_for_extra_bunnies := money_after_full_bags - additional_bag_cost
  let extra_bunnies := money_for_extra_bunnies / bunny_price
  initial_money - (full_bags * full_bag_cost + additional_bag_cost + extra_bunnies * bunny_price)

-- The theorem to prove
theorem grisha_remaining_money :
  remaining_money = 20 := by sorry

end NUMINAMATH_CALUDE_grisha_remaining_money_l3877_387734


namespace NUMINAMATH_CALUDE_inequality_proof_l3877_387705

theorem inequality_proof (a b c α : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a * b * c * (a^α + b^α + c^α) ≥ a^(α+2) * (-a + b + c) + b^(α+2) * (a - b + c) + c^(α+2) * (a + b - c) ∧
  (a * b * c * (a^α + b^α + c^α) = a^(α+2) * (-a + b + c) + b^(α+2) * (a - b + c) + c^(α+2) * (a + b - c) ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3877_387705


namespace NUMINAMATH_CALUDE_holly_weekly_pill_count_l3877_387725

/-- Calculates the total number of pills Holly takes in a week -/
def weekly_pill_count (insulin_per_day : ℕ) (blood_pressure_per_day : ℕ) : ℕ :=
  let anticonvulsants_per_day := 2 * blood_pressure_per_day
  let daily_total := insulin_per_day + blood_pressure_per_day + anticonvulsants_per_day
  7 * daily_total

/-- Proves that Holly takes 77 pills in a week given her daily requirements -/
theorem holly_weekly_pill_count : 
  weekly_pill_count 2 3 = 77 := by
  sorry

end NUMINAMATH_CALUDE_holly_weekly_pill_count_l3877_387725


namespace NUMINAMATH_CALUDE_f_value_at_negative_five_pi_thirds_l3877_387742

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem f_value_at_negative_five_pi_thirds 
  (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_period : has_period f π)
  (h_cos : ∀ x ∈ Set.Icc (-π/2) 0, f x = Real.cos x) :
  f (-5*π/3) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_negative_five_pi_thirds_l3877_387742


namespace NUMINAMATH_CALUDE_curve_equation_proof_l3877_387752

theorem curve_equation_proof :
  let x : ℝ → ℝ := λ t => 3 * Real.cos t - 2 * Real.sin t
  let y : ℝ → ℝ := λ t => 3 * Real.sin t
  let a : ℝ := 1 / 9
  let b : ℝ := -4 / 27
  let c : ℝ := 5 / 81
  let d : ℝ := 0
  let e : ℝ := 1 / 3
  ∀ t : ℝ, a * (x t)^2 + b * (x t) * (y t) + c * (y t)^2 + d * (x t) + e * (y t) = 1 :=
by sorry

end NUMINAMATH_CALUDE_curve_equation_proof_l3877_387752


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3877_387797

/-- Given a complex number z defined in terms of a real number m, 
    prove that when z is a pure imaginary number, m = -3 -/
theorem pure_imaginary_complex_number (m : ℝ) : 
  let z : ℂ := (m^2 + m - 6)/m + (m^2 - 2*m)*I
  (z.re = 0 ∧ z.im ≠ 0) → m = -3 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3877_387797


namespace NUMINAMATH_CALUDE_parabola_tangent_angle_sine_l3877_387738

/-- Given a parabola x^2 = 4y with focus F(0, 1), and a point A on the parabola where the tangent line has slope 2, 
    prove that the sine of the angle between AF and the tangent line at A is √5/5. -/
theorem parabola_tangent_angle_sine (A : ℝ × ℝ) : 
  let (x, y) := A
  (x^2 = 4*y) →                   -- A is on the parabola
  ((1/2)*x = 2) →                 -- Slope of tangent at A is 2
  let F := (0, 1)                 -- Focus of the parabola
  let slope_AF := (y - 1) / (x - 0)
  let tan_theta := |((1/2)*x - slope_AF) / (1 + (1/2)*x * slope_AF)|
  Real.sqrt (tan_theta^2 / (1 + tan_theta^2)) = Real.sqrt 5 / 5 :=
by sorry

end NUMINAMATH_CALUDE_parabola_tangent_angle_sine_l3877_387738


namespace NUMINAMATH_CALUDE_arcsin_equation_solution_l3877_387721

theorem arcsin_equation_solution :
  ∀ x : ℝ, Real.arcsin x + Real.arcsin (3 * x) = π / 2 →
  x = 1 / Real.sqrt 10 ∨ x = -(1 / Real.sqrt 10) := by
sorry

end NUMINAMATH_CALUDE_arcsin_equation_solution_l3877_387721


namespace NUMINAMATH_CALUDE_cake_muffin_buyers_l3877_387760

theorem cake_muffin_buyers (cake_buyers : ℕ) (muffin_buyers : ℕ) (both_buyers : ℕ) 
  (prob_neither : ℚ) (h1 : cake_buyers = 50) (h2 : muffin_buyers = 40) 
  (h3 : both_buyers = 19) (h4 : prob_neither = 29/100) : 
  ∃ total_buyers : ℕ, 
    total_buyers = 100 ∧ 
    (cake_buyers + muffin_buyers - both_buyers : ℚ) + prob_neither * total_buyers = total_buyers :=
by sorry

end NUMINAMATH_CALUDE_cake_muffin_buyers_l3877_387760


namespace NUMINAMATH_CALUDE_banquet_food_consumption_l3877_387774

/-- Represents the football banquet scenario -/
structure FootballBanquet where
  /-- The maximum amount of food (in pounds) consumed by any individual guest -/
  max_food_per_guest : ℝ
  /-- The minimum number of guests that attended the banquet -/
  min_guests : ℕ
  /-- The total amount of food (in pounds) consumed at the banquet -/
  total_food_consumed : ℝ

/-- Theorem stating that the total food consumed at the banquet is at least 326 pounds -/
theorem banquet_food_consumption (banquet : FootballBanquet)
  (h1 : banquet.max_food_per_guest ≤ 2)
  (h2 : banquet.min_guests ≥ 163)
  : banquet.total_food_consumed ≥ 326 := by
  sorry

#check banquet_food_consumption

end NUMINAMATH_CALUDE_banquet_food_consumption_l3877_387774


namespace NUMINAMATH_CALUDE_extra_fruits_calculation_l3877_387767

theorem extra_fruits_calculation (red_ordered green_ordered oranges_ordered : ℕ)
                                 (red_chosen green_chosen oranges_chosen : ℕ)
                                 (h1 : red_ordered = 43)
                                 (h2 : green_ordered = 32)
                                 (h3 : oranges_ordered = 25)
                                 (h4 : red_chosen = 7)
                                 (h5 : green_chosen = 5)
                                 (h6 : oranges_chosen = 4) :
  (red_ordered - red_chosen) + (green_ordered - green_chosen) + (oranges_ordered - oranges_chosen) = 84 :=
by sorry

end NUMINAMATH_CALUDE_extra_fruits_calculation_l3877_387767


namespace NUMINAMATH_CALUDE_burger_lovers_l3877_387776

theorem burger_lovers (total : ℕ) (pizza_lovers : ℕ) (both_lovers : ℕ) 
    (h1 : total = 200)
    (h2 : pizza_lovers = 125)
    (h3 : both_lovers = 40)
    (h4 : both_lovers ≤ pizza_lovers)
    (h5 : pizza_lovers ≤ total) :
  total - (pizza_lovers - both_lovers) - both_lovers = 115 := by
  sorry

end NUMINAMATH_CALUDE_burger_lovers_l3877_387776


namespace NUMINAMATH_CALUDE_unique_representation_of_nonnegative_integers_l3877_387799

theorem unique_representation_of_nonnegative_integers (n : ℕ) :
  ∃! (x y : ℕ), n = ((x + y)^2 + 3*x + y) / 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_representation_of_nonnegative_integers_l3877_387799


namespace NUMINAMATH_CALUDE_trigonometric_equality_l3877_387744

theorem trigonometric_equality (α : ℝ) :
  (2 * Real.cos (π/6 - 2*α) - Real.sqrt 3 * Real.sin (5*π/2 - 2*α)) /
  (Real.cos (9*π/2 - 2*α) + 2 * Real.cos (π/6 + 2*α)) =
  Real.tan (2*α) / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equality_l3877_387744


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l3877_387732

/-- Given two vectors a and b in ℝ², where a = (1, 2) and b = (2x, -3),
    and a is parallel to b, prove that x = -3/4 -/
theorem parallel_vectors_x_value (x : ℝ) :
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![2*x, -3]
  (∃ (k : ℝ), k ≠ 0 ∧ b = k • a) →
  x = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l3877_387732


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3877_387722

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (d : ℝ)
  (h_arithmetic : arithmetic_sequence a d)
  (h_even_sum : a 2 + a 4 + a 6 + a 8 + a 10 = 30)
  (h_odd_sum : a 1 + a 3 + a 5 + a 7 + a 9 = 25) :
  d = 1 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3877_387722


namespace NUMINAMATH_CALUDE_birthday_cake_cost_l3877_387701

/-- Proves that the cost of the birthday cake is $25 given the conditions of Erika and Rick's gift-buying scenario. -/
theorem birthday_cake_cost (gift_cost : ℝ) (erika_savings : ℝ) (rick_savings : ℝ) (leftover : ℝ) :
  gift_cost = 250 →
  erika_savings = 155 →
  rick_savings = gift_cost / 2 →
  leftover = 5 →
  erika_savings + rick_savings - gift_cost - leftover = 25 := by
  sorry

#check birthday_cake_cost

end NUMINAMATH_CALUDE_birthday_cake_cost_l3877_387701


namespace NUMINAMATH_CALUDE_jane_usable_a4_sheets_l3877_387729

/-- Represents the different types of paper sheets -/
inductive SheetType
  | BrownA4
  | YellowA4
  | YellowA3
  | PinkA2

/-- Calculates the number of usable sheets given the total and damaged counts -/
def usableSheets (total : ℕ) (damaged : ℕ) : ℕ :=
  total - damaged + (damaged / 2)

/-- Theorem: Jane has 40 total usable A4 sheets for sketching -/
theorem jane_usable_a4_sheets :
  let brown_a4_total := 28
  let yellow_a4_total := 18
  let yellow_a3_total := 9
  let pink_a2_total := 10
  let brown_a4_damaged := 3
  let yellow_a4_damaged := 5
  let yellow_a3_damaged := 2
  let pink_a2_damaged := 2
  let brown_a4_usable := usableSheets brown_a4_total brown_a4_damaged
  let yellow_a4_usable := usableSheets yellow_a4_total yellow_a4_damaged
  brown_a4_usable + yellow_a4_usable = 40 := by
    sorry


end NUMINAMATH_CALUDE_jane_usable_a4_sheets_l3877_387729


namespace NUMINAMATH_CALUDE_jacket_sale_profit_l3877_387784

/-- Calculates the merchant's gross profit for a jacket sale --/
theorem jacket_sale_profit (purchase_price : ℝ) (markup_percentage : ℝ) (discount_percentage : ℝ) : 
  purchase_price = 42 ∧ 
  markup_percentage = 0.3 ∧ 
  discount_percentage = 0.2 → 
  let selling_price := purchase_price / (1 - markup_percentage)
  let discounted_price := selling_price * (1 - discount_percentage)
  discounted_price - purchase_price = 6 := by
  sorry

end NUMINAMATH_CALUDE_jacket_sale_profit_l3877_387784


namespace NUMINAMATH_CALUDE_reflect_x_of_P_l3877_387792

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The x-axis reflection of a point -/
def reflect_x (p : Point) : Point :=
  { x := p.x, y := -p.y }

/-- The given point P -/
def P : Point :=
  { x := -1, y := 2 }

/-- Theorem: The x-axis reflection of P(-1, 2) is (-1, -2) -/
theorem reflect_x_of_P : reflect_x P = { x := -1, y := -2 } := by
  sorry

end NUMINAMATH_CALUDE_reflect_x_of_P_l3877_387792


namespace NUMINAMATH_CALUDE_smallest_n_with_forty_percent_leftmost_one_l3877_387746

/-- Returns true if the leftmost digit of n is 1 -/
def leftmost_digit_is_one (n : ℕ) : Bool := sorry

/-- Returns the count of numbers from 1 to n (inclusive) with leftmost digit 1 -/
def count_leftmost_one (n : ℕ) : ℕ := sorry

theorem smallest_n_with_forty_percent_leftmost_one :
  ∀ N : ℕ,
    N > 2017 →
    (count_leftmost_one N : ℚ) / N = 2 / 5 →
    N ≥ 1481480 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_with_forty_percent_leftmost_one_l3877_387746


namespace NUMINAMATH_CALUDE_license_plate_difference_l3877_387794

def florida_combinations : ℕ := 26^4 * 10^3
def georgia_combinations : ℕ := 26^3 * 10^3

theorem license_plate_difference : 
  florida_combinations - georgia_combinations = 439400000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_difference_l3877_387794


namespace NUMINAMATH_CALUDE_average_first_20_even_numbers_l3877_387769

theorem average_first_20_even_numbers : 
  let first_20_even : List ℕ := List.range 20 |>.map (fun i => 2 * (i + 1))
  (first_20_even.sum / first_20_even.length : ℚ) = 21 := by
  sorry

end NUMINAMATH_CALUDE_average_first_20_even_numbers_l3877_387769


namespace NUMINAMATH_CALUDE_infinite_possibilities_for_A_squared_l3877_387781

/-- Given a 3x3 matrix A with real entries such that A^4 = 0, 
    there are infinitely many possible matrices that A^2 can be. -/
theorem infinite_possibilities_for_A_squared 
  (A : Matrix (Fin 3) (Fin 3) ℝ) 
  (h : A ^ 4 = 0) : 
  ∃ S : Set (Matrix (Fin 3) (Fin 3) ℝ), 
    (∀ B ∈ S, ∃ A : Matrix (Fin 3) (Fin 3) ℝ, A ^ 4 = 0 ∧ A ^ 2 = B) ∧ 
    Set.Infinite S :=
by sorry

end NUMINAMATH_CALUDE_infinite_possibilities_for_A_squared_l3877_387781


namespace NUMINAMATH_CALUDE_negation_of_proposition_l3877_387730

theorem negation_of_proposition (p : Prop) :
  (¬ (∀ x : ℝ, x^2 - x + 1 > 0)) ↔ (∃ x : ℝ, x^2 - x + 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l3877_387730


namespace NUMINAMATH_CALUDE_linear_function_decreases_l3877_387731

/-- A linear function with a negative slope decreases as x increases -/
theorem linear_function_decreases (m b : ℝ) (h : m < 0) :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → (m * x₁ + b) > (m * x₂ + b) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_decreases_l3877_387731


namespace NUMINAMATH_CALUDE_sum_of_special_integers_l3877_387753

theorem sum_of_special_integers (n : ℕ) : 
  (∃ (S : Finset ℕ), 
    (∀ m ∈ S, m < 100 ∧ m > 0 ∧ ∃ k : ℤ, 5 * m^2 + 3 * m - 5 = 15 * k) ∧ 
    (∀ m : ℕ, m < 100 ∧ m > 0 ∧ (∃ k : ℤ, 5 * m^2 + 3 * m - 5 = 15 * k) → m ∈ S) ∧
    (Finset.sum S id = 635)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_special_integers_l3877_387753


namespace NUMINAMATH_CALUDE_yellow_balls_count_l3877_387703

theorem yellow_balls_count (total : ℕ) (white green red purple : ℕ) (prob : ℚ) :
  total = 100 ∧
  white = 50 ∧
  green = 30 ∧
  red = 7 ∧
  purple = 3 ∧
  prob = 9/10 ∧
  prob = (white + green + (total - white - green - red - purple)) / total →
  total - white - green - red - purple = 10 := by
  sorry

end NUMINAMATH_CALUDE_yellow_balls_count_l3877_387703


namespace NUMINAMATH_CALUDE_tank_insulation_problem_l3877_387783

theorem tank_insulation_problem (x : ℝ) : 
  x > 0 →  -- Ensure x is positive
  (14 * x + 20) * 20 = 1520 → 
  x = 4 := by
sorry

end NUMINAMATH_CALUDE_tank_insulation_problem_l3877_387783


namespace NUMINAMATH_CALUDE_system_solutions_l3877_387770

/-- The system of equations -/
def system (x y : ℝ) : Prop :=
  y^2 = x^3 - 3*x^2 + 2*x ∧ x^2 = y^3 - 3*y^2 + 2*y

/-- The set of solutions -/
def solutions : Set (ℝ × ℝ) :=
  {(0, 0), (2 + Real.sqrt 2, 2 + Real.sqrt 2), (2 - Real.sqrt 2, 2 - Real.sqrt 2)}

/-- Theorem stating that the solutions are correct and complete -/
theorem system_solutions :
  ∀ (x y : ℝ), system x y ↔ (x, y) ∈ solutions :=
sorry

end NUMINAMATH_CALUDE_system_solutions_l3877_387770


namespace NUMINAMATH_CALUDE_four_digit_divisible_by_18_l3877_387772

theorem four_digit_divisible_by_18 : 
  ∀ n : ℕ, n < 10 → (4150 + n) % 18 = 0 ↔ n = 8 := by sorry

end NUMINAMATH_CALUDE_four_digit_divisible_by_18_l3877_387772


namespace NUMINAMATH_CALUDE_power_sum_zero_l3877_387717

theorem power_sum_zero : (-2 : ℤ) ^ (3^2) + 2 ^ (3^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_zero_l3877_387717


namespace NUMINAMATH_CALUDE_sandwich_count_l3877_387737

def num_meats : ℕ := 12
def num_cheeses : ℕ := 8
def num_toppings : ℕ := 5

def sandwich_combinations : ℕ := num_meats * (num_cheeses.choose 2) * num_toppings

theorem sandwich_count : sandwich_combinations = 1680 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_count_l3877_387737


namespace NUMINAMATH_CALUDE_floor_sqrt_99_l3877_387736

theorem floor_sqrt_99 : ⌊Real.sqrt 99⌋ = 9 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_99_l3877_387736


namespace NUMINAMATH_CALUDE_trapezium_side_length_l3877_387771

theorem trapezium_side_length (a b h area : ℝ) : 
  b = 20 → h = 12 → area = 228 → area = (a + b) * h / 2 → a = 18 := by
  sorry

end NUMINAMATH_CALUDE_trapezium_side_length_l3877_387771


namespace NUMINAMATH_CALUDE_largest_valid_number_l3877_387766

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n ≤ 9999 ∧
  (n % 10 = (n / 10 % 10 + n / 100 % 10) % 10) ∧
  (n / 10 % 10 = (n / 100 % 10 + n / 1000 % 10) % 10)

theorem largest_valid_number : 
  is_valid_number 9099 ∧ ∀ m : ℕ, is_valid_number m → m ≤ 9099 :=
by sorry

end NUMINAMATH_CALUDE_largest_valid_number_l3877_387766


namespace NUMINAMATH_CALUDE_circle_tangents_theorem_no_single_common_tangent_l3877_387786

/-- Represents the number of common tangents between two circles -/
inductive CommonTangents
  | zero
  | two
  | three
  | four

/-- Represents the configuration of two circles -/
structure CircleConfiguration where
  r1 : ℝ  -- radius of the first circle
  r2 : ℝ  -- radius of the second circle
  d : ℝ   -- distance between the centers of the circles

/-- Function to determine the number of common tangents based on circle configuration -/
def numberOfCommonTangents (config : CircleConfiguration) : CommonTangents :=
  sorry

/-- Theorem stating that two circles with radii 10 and 4 can have 0, 2, 3, or 4 common tangents -/
theorem circle_tangents_theorem :
  ∀ (d : ℝ),
  let config := CircleConfiguration.mk 10 4 d
  (numberOfCommonTangents config = CommonTangents.zero) ∨
  (numberOfCommonTangents config = CommonTangents.two) ∨
  (numberOfCommonTangents config = CommonTangents.three) ∨
  (numberOfCommonTangents config = CommonTangents.four) :=
by sorry

/-- Theorem stating that two circles with radii 10 and 4 cannot have exactly 1 common tangent -/
theorem no_single_common_tangent :
  ∀ (d : ℝ),
  let config := CircleConfiguration.mk 10 4 d
  numberOfCommonTangents config ≠ CommonTangents.zero :=
by sorry

end NUMINAMATH_CALUDE_circle_tangents_theorem_no_single_common_tangent_l3877_387786


namespace NUMINAMATH_CALUDE_jason_career_percentage_increase_l3877_387795

/-- Represents the career progression of a military person --/
structure MilitaryCareer where
  join_age : ℕ
  years_to_chief : ℕ
  retirement_age : ℕ
  years_after_master_chief : ℕ

/-- Calculates the percentage increase in time from chief to master chief
    compared to the time to become a chief --/
def percentage_increase (career : MilitaryCareer) : ℚ :=
  let total_years := career.retirement_age - career.join_age
  let years_chief_to_retirement := total_years - career.years_to_chief
  let years_to_master_chief := years_chief_to_retirement - career.years_after_master_chief
  (years_to_master_chief - career.years_to_chief) / career.years_to_chief * 100

/-- Theorem stating that for Jason's career, the percentage increase is 25% --/
theorem jason_career_percentage_increase :
  let jason_career := MilitaryCareer.mk 18 8 46 10
  percentage_increase jason_career = 25 := by
  sorry

end NUMINAMATH_CALUDE_jason_career_percentage_increase_l3877_387795


namespace NUMINAMATH_CALUDE_complement_intersect_theorem_l3877_387718

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 2, 3}
def B : Set Nat := {2, 5}

theorem complement_intersect_theorem :
  (U \ B) ∩ A = {1, 3} := by sorry

end NUMINAMATH_CALUDE_complement_intersect_theorem_l3877_387718


namespace NUMINAMATH_CALUDE_lines_concurrent_l3877_387768

-- Define the types for points and lines
variable (Point Line : Type)

-- Define the incidence relation
variable (lies_on : Point → Line → Prop)

-- Define the intersection of two lines
variable (intersect : Line → Line → Point)

-- Define the line passing through two points
variable (line_through : Point → Point → Line)

variable (A B C D E F P X Y Z W Q : Point)

-- Define the quadrilateral ABCD
variable (is_quadrilateral : Prop)

-- Define the conditions for E, F, P, X, Y, Z, W
variable (E_def : E = intersect (line_through A B) (line_through C D))
variable (F_def : F = intersect (line_through B C) (line_through D A))
variable (not_on_EF : ¬ lies_on P (line_through E F))
variable (X_def : X = intersect (line_through P A) (line_through E F))
variable (Y_def : Y = intersect (line_through P B) (line_through E F))
variable (Z_def : Z = intersect (line_through P C) (line_through E F))
variable (W_def : W = intersect (line_through P D) (line_through E F))

-- The theorem to prove
theorem lines_concurrent :
  ∃ Q : Point,
    lies_on Q (line_through A Z) ∧
    lies_on Q (line_through B W) ∧
    lies_on Q (line_through C X) ∧
    lies_on Q (line_through D Y) :=
sorry

end NUMINAMATH_CALUDE_lines_concurrent_l3877_387768


namespace NUMINAMATH_CALUDE_audrey_dream_fraction_l3877_387711

theorem audrey_dream_fraction (total_sleep : ℝ) (not_dreaming : ℝ) 
  (h1 : total_sleep = 10)
  (h2 : not_dreaming = 6) :
  (total_sleep - not_dreaming) / total_sleep = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_audrey_dream_fraction_l3877_387711


namespace NUMINAMATH_CALUDE_problem_solution_l3877_387739

theorem problem_solution (x y : ℝ) (h1 : x > y) 
  (h2 : x^2*y^2 + x^2 + y^2 + 2*x*y = 40) (h3 : x*y + x + y = 8) : 
  x = 3 + Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3877_387739


namespace NUMINAMATH_CALUDE_monotone_increasing_condition_l3877_387724

/-- The function f(x) = kx - ln x is monotonically increasing on (1, +∞) if and only if k ≥ 1 -/
theorem monotone_increasing_condition (k : ℝ) :
  (∀ x > 1, Monotone (fun x => k * x - Real.log x)) ↔ k ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_monotone_increasing_condition_l3877_387724


namespace NUMINAMATH_CALUDE_wang_li_final_score_l3877_387750

/-- Calculates the weighted average score given individual scores and weights -/
def weightedAverage (writtenScore demonstrationScore interviewScore : ℚ) 
  (writtenWeight demonstrationWeight interviewWeight : ℚ) : ℚ :=
  (writtenScore * writtenWeight + demonstrationScore * demonstrationWeight + interviewScore * interviewWeight) /
  (writtenWeight + demonstrationWeight + interviewWeight)

/-- Theorem stating that Wang Li's final score is 94 given the specified scores and weights -/
theorem wang_li_final_score :
  weightedAverage 96 90 95 5 3 2 = 94 := by
  sorry


end NUMINAMATH_CALUDE_wang_li_final_score_l3877_387750


namespace NUMINAMATH_CALUDE_max_ab_squared_l3877_387788

theorem max_ab_squared (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 3) :
  a * b^2 ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_max_ab_squared_l3877_387788


namespace NUMINAMATH_CALUDE_exponent_multiplication_l3877_387748

theorem exponent_multiplication (a : ℝ) : a^3 * a^4 = a^7 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l3877_387748


namespace NUMINAMATH_CALUDE_largest_reciprocal_l3877_387758

theorem largest_reciprocal (a b c d e : ℚ) : 
  a = 1/8 → b = 3/4 → c = 1/2 → d = 10 → e = -2 →
  (1/a > 1/b ∧ 1/a > 1/c ∧ 1/a > 1/d ∧ 1/a > 1/e) := by
  sorry

end NUMINAMATH_CALUDE_largest_reciprocal_l3877_387758


namespace NUMINAMATH_CALUDE_parabola_intersection_theorem_l3877_387782

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the line
def line (x y : ℝ) : Prop := y = Real.sqrt 3 * x - Real.sqrt 3

-- Define point M
def point_M : ℝ × ℝ := (1, 0)

-- Define the midpoint condition
def is_midpoint (m a b : ℝ × ℝ) : Prop :=
  m.1 = (a.1 + b.1) / 2 ∧ m.2 = (a.2 + b.2) / 2

-- Main theorem
theorem parabola_intersection_theorem :
  ∃ (p : ℝ) (a b : ℝ × ℝ),
    parabola p b.1 b.2 ∧
    line b.1 b.2 ∧
    is_midpoint point_M a b →
    p = 2 :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_theorem_l3877_387782


namespace NUMINAMATH_CALUDE_modulo_23_equivalence_l3877_387787

theorem modulo_23_equivalence :
  ∃! n : ℕ, n < 23 ∧ 47582 % 23 = n :=
by
  use 3
  constructor
  · simp
    sorry
  · intro m ⟨hm, hmeq⟩
    sorry

#check modulo_23_equivalence

end NUMINAMATH_CALUDE_modulo_23_equivalence_l3877_387787


namespace NUMINAMATH_CALUDE_negative_three_triangle_four_equals_seven_l3877_387762

-- Define the ▲ operation
def triangle (a b : ℚ) : ℚ := -a + b

-- Theorem statement
theorem negative_three_triangle_four_equals_seven :
  triangle (-3) 4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_negative_three_triangle_four_equals_seven_l3877_387762


namespace NUMINAMATH_CALUDE_odd_sum_probability_l3877_387726

/-- Represents a tile with a number from 1 to 12 -/
def Tile := Fin 12

/-- Represents a player's selection of 4 tiles -/
def PlayerSelection := Finset Tile

/-- The set of all possible tile selections -/
def AllSelections : Finset (PlayerSelection × PlayerSelection × PlayerSelection) :=
  sorry

/-- Checks if a player's selection sum is odd -/
def isOddSum (selection : PlayerSelection) : Bool :=
  sorry

/-- The set of selections where all players have odd sums -/
def OddSumSelections : Finset (PlayerSelection × PlayerSelection × PlayerSelection) :=
  sorry

/-- The probability of all players obtaining an odd sum -/
theorem odd_sum_probability :
  (Finset.card OddSumSelections : ℚ) / (Finset.card AllSelections : ℚ) = 16 / 385 :=
sorry

end NUMINAMATH_CALUDE_odd_sum_probability_l3877_387726


namespace NUMINAMATH_CALUDE_animal_permutations_l3877_387743

/-- The number of animals excluding Rat and Snake -/
def n : ℕ := 4

/-- The total number of animals -/
def total_animals : ℕ := 6

/-- Theorem stating that the number of permutations of n distinct objects
    is equal to n factorial, where n is the number of animals excluding
    Rat and Snake -/
theorem animal_permutations :
  (Finset.range n).card.factorial = 24 :=
sorry

end NUMINAMATH_CALUDE_animal_permutations_l3877_387743


namespace NUMINAMATH_CALUDE_median_sum_lower_bound_l3877_387765

/-- Given a triangle ABC with sides a, b, c and medians ma, mb, mc, 
    the sum of the lengths of the medians is at least three quarters of its perimeter. -/
theorem median_sum_lower_bound (a b c ma mb mc : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_pos_ma : ma > 0) (h_pos_mb : mb > 0) (h_pos_mc : mc > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_ma : ma^2 = (2*b^2 + 2*c^2 - a^2) / 4)
  (h_mb : mb^2 = (2*c^2 + 2*a^2 - b^2) / 4)
  (h_mc : mc^2 = (2*a^2 + 2*b^2 - c^2) / 4) :
  ma + mb + mc ≥ 3/4 * (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_median_sum_lower_bound_l3877_387765


namespace NUMINAMATH_CALUDE_part_one_part_two_l3877_387710

-- Part 1
theorem part_one : Real.sqrt 9 + 2 * Real.sin (30 * π / 180) - 1 = 3 := by sorry

-- Part 2
theorem part_two : 
  ∀ x : ℝ, (2*x - 3)^2 = 2*(2*x - 3) ↔ x = 3/2 ∨ x = 5/2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3877_387710


namespace NUMINAMATH_CALUDE_inverse_function_sum_l3877_387723

/-- Given a function g and its inverse g⁻¹, prove that c + d = 3 * (2^(1/3)) -/
theorem inverse_function_sum (c d : ℝ) 
  (g : ℝ → ℝ) (g_inv : ℝ → ℝ)
  (hg : ∀ x, g x = c * x + d)
  (hg_inv : ∀ x, g_inv x = d * x - 2 * c)
  (h_inverse : ∀ x, g (g_inv x) = x) :
  c + d = 3 * Real.rpow 2 (1/3) := by
sorry

end NUMINAMATH_CALUDE_inverse_function_sum_l3877_387723


namespace NUMINAMATH_CALUDE_good_price_after_discounts_l3877_387733

theorem good_price_after_discounts (P : ℝ) : 
  P * (1 - 0.20) * (1 - 0.10) * (1 - 0.05) = 6700 → P = 9798.25 := by
  sorry

end NUMINAMATH_CALUDE_good_price_after_discounts_l3877_387733


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l3877_387715

/-- 
Given two infinite geometric series:
- First series with first term a₁ = 8 and second term b₁ = 2
- Second series with first term a₂ = 8 and second term b₂ = 2 + m
If the sum of the second series is three times the sum of the first series,
then m = 4.
-/
theorem geometric_series_ratio (m : ℝ) : 
  let a₁ : ℝ := 8
  let b₁ : ℝ := 2
  let a₂ : ℝ := 8
  let b₂ : ℝ := 2 + m
  let r₁ : ℝ := b₁ / a₁
  let r₂ : ℝ := b₂ / a₂
  let s₁ : ℝ := a₁ / (1 - r₁)
  let s₂ : ℝ := a₂ / (1 - r₂)
  s₂ = 3 * s₁ → m = 4 := by
  sorry


end NUMINAMATH_CALUDE_geometric_series_ratio_l3877_387715


namespace NUMINAMATH_CALUDE_no_rational_solution_l3877_387778

theorem no_rational_solution : ¬∃ (p q r : ℚ), p + q + r = 0 ∧ p * q * r = 1 := by
  sorry

end NUMINAMATH_CALUDE_no_rational_solution_l3877_387778


namespace NUMINAMATH_CALUDE_scott_sales_theorem_scott_total_sales_l3877_387706

/-- Calculates the total money made from selling items at given prices and quantities -/
def total_money_made (smoothie_price cake_price : ℕ) (smoothie_qty cake_qty : ℕ) : ℕ :=
  smoothie_price * smoothie_qty + cake_price * cake_qty

/-- Theorem stating that the total money made is equal to the sum of products of prices and quantities -/
theorem scott_sales_theorem (smoothie_price cake_price : ℕ) (smoothie_qty cake_qty : ℕ) :
  total_money_made smoothie_price cake_price smoothie_qty cake_qty =
  smoothie_price * smoothie_qty + cake_price * cake_qty :=
by
  sorry

/-- Verifies that Scott's total sales match the calculated amount -/
theorem scott_total_sales :
  total_money_made 3 2 40 18 = 156 :=
by
  sorry

end NUMINAMATH_CALUDE_scott_sales_theorem_scott_total_sales_l3877_387706


namespace NUMINAMATH_CALUDE_union_membership_intersection_membership_positive_product_l3877_387777

-- Statement 1
theorem union_membership (A B : Set α) (x : α) : x ∈ A ∪ B → x ∈ A ∨ x ∈ B := by sorry

-- Statement 2
theorem intersection_membership (A B : Set α) (x : α) : x ∈ A ∩ B → x ∈ A ∧ x ∈ B := by sorry

-- Statement 3
theorem positive_product (a b : ℝ) : a > 0 ∧ b > 0 → a * b > 0 := by sorry

end NUMINAMATH_CALUDE_union_membership_intersection_membership_positive_product_l3877_387777


namespace NUMINAMATH_CALUDE_total_subjects_l3877_387793

/-- Given the number of subjects taken by Monica, prove the total number of subjects taken by all four students. -/
theorem total_subjects (monica : ℕ) (h1 : monica = 10) : ∃ (marius millie michael : ℕ),
  marius = monica + 4 ∧
  millie = marius + 3 ∧
  michael = 2 * millie ∧
  monica + marius + millie + michael = 75 :=
by sorry

end NUMINAMATH_CALUDE_total_subjects_l3877_387793
