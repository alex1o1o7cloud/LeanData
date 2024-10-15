import Mathlib

namespace NUMINAMATH_CALUDE_union_of_A_and_B_l8_818

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -2 < x ∧ x < 1}
def B : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}

-- Theorem statement
theorem union_of_A_and_B :
  A ∪ B = {x : ℝ | -2 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l8_818


namespace NUMINAMATH_CALUDE_fruit_sales_problem_l8_800

/-- Fruit sales problem -/
theorem fruit_sales_problem 
  (purchase_price : ℝ)
  (price_10 : ℝ)
  (sales_10 : ℝ)
  (price_13 : ℝ)
  (profit_13 : ℝ)
  (h1 : purchase_price = 8)
  (h2 : price_10 = 10)
  (h3 : sales_10 = 300)
  (h4 : price_13 = 13)
  (h5 : profit_13 = 750)
  (y : ℝ → ℝ)
  (h6 : ∀ x > 0, ∃ k b : ℝ, y x = k * x + b) :
  (∃ k b : ℝ, ∀ x > 0, y x = k * x + b ∧ k = -50 ∧ b = 800) ∧
  (∃ max_price : ℝ, max_price = 12 ∧ 
    ∀ x > 0, (y x) * (x - purchase_price) ≤ (y max_price) * (max_price - purchase_price)) ∧
  (∃ max_profit : ℝ, max_profit = 800 ∧
    max_profit = (y 12) * (12 - purchase_price)) :=
by sorry

end NUMINAMATH_CALUDE_fruit_sales_problem_l8_800


namespace NUMINAMATH_CALUDE_mango_seller_loss_percentage_l8_808

/-- Calculates the percentage of loss for a fruit seller selling mangoes -/
theorem mango_seller_loss_percentage
  (selling_price : ℝ)
  (profit_price : ℝ)
  (h1 : selling_price = 16)
  (h2 : profit_price = 21.818181818181817)
  (h3 : profit_price = 1.2 * (profit_price / 1.2)) :
  (((profit_price / 1.2) - selling_price) / (profit_price / 1.2)) * 100 = 12 :=
by sorry

end NUMINAMATH_CALUDE_mango_seller_loss_percentage_l8_808


namespace NUMINAMATH_CALUDE_fruitBaskets_eq_96_l8_892

/-- The number of ways to choose k items from n identical items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of fruit baskets with at least 3 pieces of fruit,
    given 7 apples and 12 oranges -/
def fruitBaskets : ℕ :=
  let totalBaskets := (choose 7 0 + choose 7 1 + choose 7 2 + choose 7 3 +
                       choose 7 4 + choose 7 5 + choose 7 6 + choose 7 7) *
                      (choose 12 0 + choose 12 1 + choose 12 2 + choose 12 3 +
                       choose 12 4 + choose 12 5 + choose 12 6 + choose 12 7 +
                       choose 12 8 + choose 12 9 + choose 12 10 + choose 12 11 +
                       choose 12 12)
  let invalidBaskets := choose 7 0 * choose 12 0 +
                        choose 7 0 * choose 12 1 +
                        choose 7 0 * choose 12 2 +
                        choose 7 1 * choose 12 0 +
                        choose 7 1 * choose 12 1 +
                        choose 7 2 * choose 12 0
  totalBaskets - invalidBaskets

theorem fruitBaskets_eq_96 : fruitBaskets = 96 := by sorry

end NUMINAMATH_CALUDE_fruitBaskets_eq_96_l8_892


namespace NUMINAMATH_CALUDE_rectangle_area_breadth_ratio_l8_896

/-- Theorem: For a rectangular plot with breadth 14 meters and length 10 meters greater than its breadth, the ratio of its area to its breadth is 24:1. -/
theorem rectangle_area_breadth_ratio :
  ∀ (length breadth area : ℝ),
  breadth = 14 →
  length = breadth + 10 →
  area = length * breadth →
  area / breadth = 24 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_breadth_ratio_l8_896


namespace NUMINAMATH_CALUDE_bayswater_volleyball_club_members_l8_880

theorem bayswater_volleyball_club_members : 
  let knee_pad_cost : ℕ := 6
  let jersey_cost : ℕ := knee_pad_cost + 7
  let member_cost : ℕ := 2 * (knee_pad_cost + jersey_cost)
  let total_expenditure : ℕ := 3120
  total_expenditure / member_cost = 82 :=
by sorry

end NUMINAMATH_CALUDE_bayswater_volleyball_club_members_l8_880


namespace NUMINAMATH_CALUDE_farm_problem_solution_l8_886

/-- Represents the farm ploughing problem -/
structure FarmProblem where
  planned_daily_area : ℕ  -- Planned area to plough per day
  actual_daily_area : ℕ   -- Actual area ploughed per day
  extra_days : ℕ          -- Extra days worked
  total_field_area : ℕ    -- Total area of the farm field

/-- Calculates the area left to plough -/
def area_left_to_plough (fp : FarmProblem) : ℕ :=
  let planned_days := fp.total_field_area / fp.planned_daily_area
  let actual_days := planned_days + fp.extra_days
  let ploughed_area := fp.actual_daily_area * actual_days
  fp.total_field_area - ploughed_area

/-- Theorem stating the correct result for the given problem -/
theorem farm_problem_solution :
  let fp : FarmProblem := {
    planned_daily_area := 340,
    actual_daily_area := 85,
    extra_days := 2,
    total_field_area := 280
  }
  area_left_to_plough fp = 25 := by
  sorry

end NUMINAMATH_CALUDE_farm_problem_solution_l8_886


namespace NUMINAMATH_CALUDE_divisible_by_nine_l8_878

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

theorem divisible_by_nine (N : ℕ) : 
  sum_of_digits N = sum_of_digits (5 * N) → N % 9 = 0 := by sorry

end NUMINAMATH_CALUDE_divisible_by_nine_l8_878


namespace NUMINAMATH_CALUDE_ages_sum_after_three_years_l8_875

/-- Given four persons a, b, c, and d with the following age relationships:
    - The sum of their present ages is S
    - a's age is twice b's age
    - c's age is half of a's age
    - d's age is the difference between a's and c's ages
    This theorem proves that the sum of their ages after 3 years is S + 12 -/
theorem ages_sum_after_three_years
  (S : ℝ) -- Sum of present ages
  (a b c d : ℝ) -- Present ages of individuals
  (h1 : a + b + c + d = S) -- Sum of present ages is S
  (h2 : a = 2 * b) -- a's age is twice b's age
  (h3 : c = a / 2) -- c's age is half of a's age
  (h4 : d = a - c) -- d's age is the difference between a's and c's ages
  : (a + 3) + (b + 3) + (c + 3) + (d + 3) = S + 12 := by
  sorry


end NUMINAMATH_CALUDE_ages_sum_after_three_years_l8_875


namespace NUMINAMATH_CALUDE_complex_multiplication_l8_879

theorem complex_multiplication (i : ℂ) : i * i = -1 → (1 + i) * (1 - i) = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l8_879


namespace NUMINAMATH_CALUDE_furniture_shop_cost_price_l8_821

/-- 
Given a furniture shop where the owner charges 25% more than the cost price,
this theorem proves that if a customer pays Rs. 1000 for an item, 
then the cost price of that item is Rs. 800.
-/
theorem furniture_shop_cost_price 
  (markup_percentage : ℝ) 
  (selling_price : ℝ) 
  (h1 : markup_percentage = 25)
  (h2 : selling_price = 1000) :
  let cost_price := selling_price / (1 + markup_percentage / 100)
  cost_price = 800 := by
  sorry

end NUMINAMATH_CALUDE_furniture_shop_cost_price_l8_821


namespace NUMINAMATH_CALUDE_pond_volume_calculation_l8_835

/-- The volume of a rectangular prism with given dimensions -/
def pond_volume (length width depth : ℝ) : ℝ :=
  length * width * depth

/-- Theorem stating that the volume of the pond is 1200 cubic meters -/
theorem pond_volume_calculation :
  pond_volume 20 12 5 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_pond_volume_calculation_l8_835


namespace NUMINAMATH_CALUDE_football_club_balance_l8_852

/-- Represents the balance and transactions of a football club --/
structure FootballClub where
  initialBalance : ℝ
  playersSold : ℕ
  sellingPrice : ℝ
  playerAPrice : ℝ
  playerBPrice : ℝ
  playerCPrice : ℝ
  playerDPrice : ℝ
  eurToUsd : ℝ
  gbpToUsd : ℝ
  jpyToUsd : ℝ

/-- Calculates the final balance of the football club after transactions --/
def finalBalance (club : FootballClub) : ℝ :=
  club.initialBalance +
  club.playersSold * club.sellingPrice -
  (club.playerAPrice * club.eurToUsd +
   club.playerBPrice * club.gbpToUsd +
   club.playerCPrice * club.jpyToUsd +
   club.playerDPrice * club.eurToUsd)

/-- Theorem stating that the final balance of the football club is 71.4 million USD --/
theorem football_club_balance (club : FootballClub)
  (h1 : club.initialBalance = 100)
  (h2 : club.playersSold = 2)
  (h3 : club.sellingPrice = 10)
  (h4 : club.playerAPrice = 12)
  (h5 : club.playerBPrice = 8)
  (h6 : club.playerCPrice = 1000)
  (h7 : club.playerDPrice = 9)
  (h8 : club.eurToUsd = 1.3)
  (h9 : club.gbpToUsd = 1.6)
  (h10 : club.jpyToUsd = 0.0085) :
  finalBalance club = 71.4 := by
  sorry

end NUMINAMATH_CALUDE_football_club_balance_l8_852


namespace NUMINAMATH_CALUDE_linear_function_not_in_third_quadrant_l8_872

/-- A linear function y = (2k-1)x + k does not pass through the third quadrant
    if and only if 0 ≤ k < 1/2 -/
theorem linear_function_not_in_third_quadrant (k : ℝ) :
  (∀ x y : ℝ, y = (2*k - 1)*x + k → ¬(x < 0 ∧ y < 0)) ↔ (0 ≤ k ∧ k < 1/2) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_not_in_third_quadrant_l8_872


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l8_844

theorem sum_of_squares_of_roots (p q r : ℝ) : 
  (3 * p^3 - 2 * p^2 + 6 * p - 9 = 0) → 
  (3 * q^3 - 2 * q^2 + 6 * q - 9 = 0) → 
  (3 * r^3 - 2 * r^2 + 6 * r - 9 = 0) → 
  p^2 + q^2 + r^2 = -32/9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l8_844


namespace NUMINAMATH_CALUDE_nine_women_eighteen_tea_l8_831

/-- The time (in minutes) it takes for a given number of women to drink a given amount of tea,
    given that 1.5 women drink 1.5 tea in 1.5 minutes. -/
def drinking_time (women : ℚ) (tea : ℚ) : ℚ :=
  1.5 * tea / women

/-- Theorem stating that if 1.5 women drink 1.5 tea in 1.5 minutes,
    then 9 women can drink 18 tea in 3 minutes. -/
theorem nine_women_eighteen_tea :
  drinking_time 9 18 = 3 := by
  sorry

end NUMINAMATH_CALUDE_nine_women_eighteen_tea_l8_831


namespace NUMINAMATH_CALUDE_magnitude_BC_l8_839

/-- Given two points A and C in ℝ², and a vector AB, prove that the magnitude of BC is √29 -/
theorem magnitude_BC (A C B : ℝ × ℝ) (h1 : A = (2, -1)) (h2 : C = (0, 2)) 
  (h3 : B.1 - A.1 = 3 ∧ B.2 - A.2 = 5) : 
  Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = Real.sqrt 29 := by
  sorry

#check magnitude_BC

end NUMINAMATH_CALUDE_magnitude_BC_l8_839


namespace NUMINAMATH_CALUDE_ben_dogs_difference_l8_801

/-- The number of dogs Teddy has -/
def teddy_dogs : ℕ := 7

/-- The number of cats Teddy has -/
def teddy_cats : ℕ := 8

/-- The number of cats Dave has -/
def dave_cats : ℕ := teddy_cats + 13

/-- The number of dogs Dave has -/
def dave_dogs : ℕ := teddy_dogs - 5

/-- The total number of pets all three have -/
def total_pets : ℕ := 54

/-- The number of dogs Ben has -/
def ben_dogs : ℕ := total_pets - (teddy_dogs + teddy_cats + dave_dogs + dave_cats)

theorem ben_dogs_difference : ben_dogs - teddy_dogs = 9 := by
  sorry

end NUMINAMATH_CALUDE_ben_dogs_difference_l8_801


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l8_873

theorem inequality_and_equality_condition (a b c d : ℝ)
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_one : a + b + c + d = 1) :
  (a^2 / (a + b) + b^2 / (b + c) + c^2 / (c + d) + d^2 / (d + a) ≥ 1/2) ∧
  (a^2 / (a + b) + b^2 / (b + c) + c^2 / (c + d) + d^2 / (d + a) = 1/2 ↔ a = b ∧ b = c ∧ c = d) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l8_873


namespace NUMINAMATH_CALUDE_quadratic_root_implies_s_value_l8_836

theorem quadratic_root_implies_s_value (p s : ℝ) :
  (∃ (x : ℂ), 3 * x^2 + p * x + s = 0 ∧ x = 4 + 3*I) →
  s = 75 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_s_value_l8_836


namespace NUMINAMATH_CALUDE_least_n_square_and_cube_n_144_satisfies_least_n_is_144_l8_862

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

theorem least_n_square_and_cube :
  ∀ n : ℕ, n > 0 →
    (is_perfect_square (9*n) ∧ is_perfect_cube (12*n)) →
    n ≥ 144 :=
by sorry

theorem n_144_satisfies :
  is_perfect_square (9*144) ∧ is_perfect_cube (12*144) :=
by sorry

theorem least_n_is_144 :
  ∀ n : ℕ, n > 0 →
    (is_perfect_square (9*n) ∧ is_perfect_cube (12*n)) →
    n = 144 :=
by sorry

end NUMINAMATH_CALUDE_least_n_square_and_cube_n_144_satisfies_least_n_is_144_l8_862


namespace NUMINAMATH_CALUDE_translation_result_l8_865

def point_translation (x y dx dy : ℝ) : ℝ × ℝ :=
  (x + dx, y - dy)

theorem translation_result :
  point_translation (-2) 3 3 1 = (1, 2) := by
  sorry

end NUMINAMATH_CALUDE_translation_result_l8_865


namespace NUMINAMATH_CALUDE_base_conversion_subtraction_l8_848

/-- Converts a number from base 8 to base 10 -/
def base8ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 9 to base 10 -/
def base9ToBase10 (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem base_conversion_subtraction :
  base8ToBase10 52103 - base9ToBase10 1452 = 20471 := by sorry

end NUMINAMATH_CALUDE_base_conversion_subtraction_l8_848


namespace NUMINAMATH_CALUDE_k_range_l8_820

open Real

/-- The function f(x) = (ln x)/x - kx is increasing on (0, +∞) -/
def f_increasing (k : ℝ) : Prop :=
  ∀ x, x > 0 → Monotone (λ x => (log x) / x - k * x)

/-- The theorem to be proved -/
theorem k_range (k : ℝ) : f_increasing k → k ≤ -1 / (2 * Real.exp 3) := by
  sorry

end NUMINAMATH_CALUDE_k_range_l8_820


namespace NUMINAMATH_CALUDE_radio_price_calculation_l8_826

/-- Given a radio with 7% sales tax, if reducing its price by 161.46 results in a price of 2468,
    then the original price including sales tax is 2629.46. -/
theorem radio_price_calculation (original_price : ℝ) : 
  (original_price - 161.46 = 2468) → original_price = 2629.46 := by
  sorry

end NUMINAMATH_CALUDE_radio_price_calculation_l8_826


namespace NUMINAMATH_CALUDE_picnic_task_division_l8_809

theorem picnic_task_division (n : ℕ) (k : ℕ) (h1 : n = 6) (h2 : k = 3) :
  Nat.choose n k = 20 := by
  sorry

end NUMINAMATH_CALUDE_picnic_task_division_l8_809


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l8_884

/-- The quadratic function f(x) with parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x + 9

/-- Predicate indicating if f has a root for a given m -/
def has_root (m : ℝ) : Prop := ∃ x, f m x = 0

theorem sufficient_not_necessary :
  (∀ m, m > 7 → has_root m) ∧ 
  (∃ m, has_root m ∧ m ≤ 7) := by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l8_884


namespace NUMINAMATH_CALUDE_mans_running_speed_l8_804

/-- Proves that given a man who walks at 8 kmph for 4 hours and 45 minutes,
    and runs the same distance in 120 minutes, his running speed is 19 kmph. -/
theorem mans_running_speed
  (walking_speed : ℝ)
  (walking_time_hours : ℝ)
  (walking_time_minutes : ℝ)
  (running_time_minutes : ℝ)
  (h1 : walking_speed = 8)
  (h2 : walking_time_hours = 4)
  (h3 : walking_time_minutes = 45)
  (h4 : running_time_minutes = 120)
  : (walking_speed * (walking_time_hours + walking_time_minutes / 60)) /
    (running_time_minutes / 60) = 19 := by
  sorry


end NUMINAMATH_CALUDE_mans_running_speed_l8_804


namespace NUMINAMATH_CALUDE_correct_proposition_l8_860

def p : Prop := ∀ x : ℝ, x^2 - x + 2 < 0

def q : Prop := ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 ≥ 1

theorem correct_proposition : ¬p ∧ q := by
  sorry

end NUMINAMATH_CALUDE_correct_proposition_l8_860


namespace NUMINAMATH_CALUDE_quadratic_root_sum_equality_l8_894

theorem quadratic_root_sum_equality (b₁ b₂ b₃ c₁ c₂ c₃ : ℝ) 
  (h₁ : b₁^2 - 4*c₁ = 1)
  (h₂ : b₂^2 - 4*c₂ = 4)
  (h₃ : b₃^2 - 4*c₃ = 9) :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    (x₁^2 + b₁*x₁ + c₁ = 0) ∧
    (y₁^2 + b₁*y₁ + c₁ = 0) ∧
    (x₂^2 + b₂*x₂ + c₂ = 0) ∧
    (y₂^2 + b₂*y₂ + c₂ = 0) ∧
    (x₃^2 + b₃*x₃ + c₃ = 0) ∧
    (y₃^2 + b₃*y₃ + c₃ = 0) ∧
    (x₁ + x₂ + y₃ = y₁ + y₂ + x₃) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_equality_l8_894


namespace NUMINAMATH_CALUDE_problem_1_l8_888

theorem problem_1 : (-8) + (-7) - (-6) + 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l8_888


namespace NUMINAMATH_CALUDE_parabola_translation_l8_837

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  f : ℝ → ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (h v : ℝ) : Parabola :=
  { f := fun x => p.f (x - h) + v }

/-- The original parabola y = x^2 - 2 -/
def original_parabola : Parabola :=
  { f := fun x => x^2 - 2 }

/-- The translated parabola -/
def translated_parabola : Parabola :=
  translate original_parabola 1 3

theorem parabola_translation :
  translated_parabola.f = fun x => (x - 1)^2 + 1 := by
  sorry


end NUMINAMATH_CALUDE_parabola_translation_l8_837


namespace NUMINAMATH_CALUDE_unit_vector_AB_l8_816

def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (4, -1)

theorem unit_vector_AB : 
  let AB := (B.1 - A.1, B.2 - A.2)
  let magnitude := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let unit_vector := (AB.1 / magnitude, AB.2 / magnitude)
  unit_vector = (3/5, -4/5) :=
by sorry

end NUMINAMATH_CALUDE_unit_vector_AB_l8_816


namespace NUMINAMATH_CALUDE_inscribed_equal_angles_is_regular_circumscribed_equal_sides_is_regular_l8_828

-- Define a polygon type
structure Polygon where
  vertices : List ℝ × ℝ
  sides : Nat
  is_odd : Odd sides

-- Define properties for inscribed and circumscribed polygons
def is_inscribed (p : Polygon) : Prop := sorry
def is_circumscribed (p : Polygon) : Prop := sorry

-- Define properties for equal angles and equal sides
def has_equal_angles (p : Polygon) : Prop := sorry
def has_equal_sides (p : Polygon) : Prop := sorry

-- Define what it means for a polygon to be regular
def is_regular (p : Polygon) : Prop := sorry

-- Theorem for part a
theorem inscribed_equal_angles_is_regular (p : Polygon) 
  (h_inscribed : is_inscribed p) (h_equal_angles : has_equal_angles p) : 
  is_regular p := by sorry

-- Theorem for part b
theorem circumscribed_equal_sides_is_regular (p : Polygon) 
  (h_circumscribed : is_circumscribed p) (h_equal_sides : has_equal_sides p) : 
  is_regular p := by sorry

end NUMINAMATH_CALUDE_inscribed_equal_angles_is_regular_circumscribed_equal_sides_is_regular_l8_828


namespace NUMINAMATH_CALUDE_linear_function_problem_l8_812

/-- A linear function satisfying certain conditions -/
def f (x : ℝ) : ℝ := sorry

/-- The inverse of f -/
def f_inv (x : ℝ) : ℝ := sorry

theorem linear_function_problem :
  (∀ x y : ℝ, ∃ a b : ℝ, f x = a * x + b) →  -- f is linear
  (∀ x : ℝ, f x = 4 * f_inv x + 8) →         -- f(x) = 4f^(-1)(x) + 8
  (f 1 = 5) →                                -- f(1) = 5
  (f 2 = 20 / 3) :=                          -- f(2) = 20/3
by sorry

end NUMINAMATH_CALUDE_linear_function_problem_l8_812


namespace NUMINAMATH_CALUDE_salary_percentage_difference_l8_887

theorem salary_percentage_difference (a b : ℝ) (h : b = 1.25 * a) :
  (b - a) / b = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_salary_percentage_difference_l8_887


namespace NUMINAMATH_CALUDE_f_is_odd_and_decreasing_l8_874

def f (x : ℝ) : ℝ := -x^3

theorem f_is_odd_and_decreasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f y < f x) :=
sorry

end NUMINAMATH_CALUDE_f_is_odd_and_decreasing_l8_874


namespace NUMINAMATH_CALUDE_today_is_thursday_l8_825

-- Define the days of the week
inductive Day : Type
  | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday

-- Define when A lies
def A_lies (d : Day) : Prop :=
  d = Day.Monday ∨ d = Day.Tuesday ∨ d = Day.Wednesday

-- Define when B lies
def B_lies (d : Day) : Prop :=
  d = Day.Thursday ∨ d = Day.Friday ∨ d = Day.Saturday

-- Define the previous day
def prev_day (d : Day) : Day :=
  match d with
  | Day.Monday => Day.Sunday
  | Day.Tuesday => Day.Monday
  | Day.Wednesday => Day.Tuesday
  | Day.Thursday => Day.Wednesday
  | Day.Friday => Day.Thursday
  | Day.Saturday => Day.Friday
  | Day.Sunday => Day.Saturday

-- Theorem statement
theorem today_is_thursday : 
  ∃ (d : Day), 
    (A_lies (prev_day d) ↔ ¬(A_lies d)) ∧ 
    (B_lies (prev_day d) ↔ ¬(B_lies d)) ∧ 
    d = Day.Thursday := by
  sorry

end NUMINAMATH_CALUDE_today_is_thursday_l8_825


namespace NUMINAMATH_CALUDE_fruit_drink_volume_l8_832

/-- Represents a fruit drink composed of orange, watermelon, and grape juice -/
structure FruitDrink where
  total : ℝ
  orange_percent : ℝ
  watermelon_percent : ℝ
  grape_ounces : ℝ

/-- Theorem stating the total volume of the fruit drink -/
theorem fruit_drink_volume (drink : FruitDrink)
  (h1 : drink.orange_percent = 0.25)
  (h2 : drink.watermelon_percent = 0.40)
  (h3 : drink.grape_ounces = 70)
  (h4 : drink.orange_percent + drink.watermelon_percent + drink.grape_ounces / drink.total = 1) :
  drink.total = 200 := by
  sorry

end NUMINAMATH_CALUDE_fruit_drink_volume_l8_832


namespace NUMINAMATH_CALUDE_briannas_books_l8_805

/-- Brianna's book reading problem -/
theorem briannas_books :
  let books_per_year : ℕ := 24
  let gift_books : ℕ := 6
  let old_books : ℕ := 4
  let bought_books : ℕ := x
  let borrowed_books : ℕ := x - 2

  gift_books + bought_books + borrowed_books + old_books = books_per_year →
  bought_books = 8 := by
  sorry

end NUMINAMATH_CALUDE_briannas_books_l8_805


namespace NUMINAMATH_CALUDE_inequality_holds_l8_899

theorem inequality_holds (a b c : ℝ) (h1 : a > 0) (h2 : 0 > b) (h3 : b > c) :
  a / c^2 > b / c^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_l8_899


namespace NUMINAMATH_CALUDE_driver_net_pay_rate_l8_889

/-- Calculates the net rate of pay for a driver given specific conditions. -/
theorem driver_net_pay_rate 
  (travel_time : ℝ) 
  (speed : ℝ) 
  (fuel_efficiency : ℝ) 
  (pay_per_mile : ℝ) 
  (gas_price : ℝ)
  (h1 : travel_time = 3)
  (h2 : speed = 50)
  (h3 : fuel_efficiency = 25)
  (h4 : pay_per_mile = 0.60)
  (h5 : gas_price = 2.50) :
  (pay_per_mile * speed * travel_time - (speed * travel_time / fuel_efficiency) * gas_price) / travel_time = 25 :=
by sorry

end NUMINAMATH_CALUDE_driver_net_pay_rate_l8_889


namespace NUMINAMATH_CALUDE_zero_area_quadrilateral_l8_849

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the area of a quadrilateral given its four vertices in 3D space -/
def quadrilateralArea (A B C D : Point3D) : ℝ :=
  sorry

/-- The main theorem stating that the area of the quadrilateral with given vertices is 0 -/
theorem zero_area_quadrilateral :
  let A : Point3D := ⟨2, 4, 6⟩
  let B : Point3D := ⟨7, 9, 11⟩
  let C : Point3D := ⟨1, 3, 5⟩
  let D : Point3D := ⟨6, 8, 10⟩
  quadrilateralArea A B C D = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_area_quadrilateral_l8_849


namespace NUMINAMATH_CALUDE_music_talent_sample_l8_843

/-- Represents the number of students selected in a stratified sampling -/
def stratified_sample (total_population : ℕ) (group_size : ℕ) (sample_size : ℕ) : ℕ :=
  (group_size * sample_size) / total_population

/-- Proves that in a stratified sampling of 40 students from a population of 100 students,
    where 40 students have music talent, the number of music-talented students selected is 16 -/
theorem music_talent_sample :
  stratified_sample 100 40 40 = 16 := by
  sorry

end NUMINAMATH_CALUDE_music_talent_sample_l8_843


namespace NUMINAMATH_CALUDE_subset_from_intersection_l8_866

theorem subset_from_intersection (M N : Set α) : M ∩ N = M → M ⊆ N := by
  sorry

end NUMINAMATH_CALUDE_subset_from_intersection_l8_866


namespace NUMINAMATH_CALUDE_flight_cost_calculation_l8_846

def trip_expenses (initial_savings hotel_cost food_cost remaining_money : ℕ) : Prop :=
  ∃ flight_cost : ℕ, 
    initial_savings = hotel_cost + food_cost + flight_cost + remaining_money

theorem flight_cost_calculation (initial_savings hotel_cost food_cost remaining_money : ℕ) 
  (h : trip_expenses initial_savings hotel_cost food_cost remaining_money) :
  ∃ flight_cost : ℕ, flight_cost = 1200 :=
by
  sorry

end NUMINAMATH_CALUDE_flight_cost_calculation_l8_846


namespace NUMINAMATH_CALUDE_integer_solutions_count_l8_815

/-- The number of distinct integer values of a for which x^2 + ax + 9a = 0 has integer solutions for x -/
theorem integer_solutions_count : 
  (∃ (S : Finset ℤ), (∀ a : ℤ, (∃ x : ℤ, x^2 + a*x + 9*a = 0) ↔ a ∈ S) ∧ Finset.card S = 5) :=
by sorry

end NUMINAMATH_CALUDE_integer_solutions_count_l8_815


namespace NUMINAMATH_CALUDE_spade_operation_result_l8_870

def spade (a b : ℝ) : ℝ := |a - b|

theorem spade_operation_result : spade 3 (spade 5 (spade 8 12)) = 2 := by sorry

end NUMINAMATH_CALUDE_spade_operation_result_l8_870


namespace NUMINAMATH_CALUDE_smallest_positive_integer_congruence_l8_841

theorem smallest_positive_integer_congruence (x : ℕ) : x = 29 ↔ 
  x > 0 ∧
  (5 * x) % 20 = 25 % 20 ∧
  (3 * x + 1) % 7 = 4 % 7 ∧
  (2 * x - 3) % 13 = x % 13 ∧
  ∀ y : ℕ, y > 0 → 
    ((5 * y) % 20 = 25 % 20 ∧
     (3 * y + 1) % 7 = 4 % 7 ∧
     (2 * y - 3) % 13 = y % 13) → 
    x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_congruence_l8_841


namespace NUMINAMATH_CALUDE_students_playing_neither_sport_l8_890

theorem students_playing_neither_sport (total : ℕ) (football : ℕ) (tennis : ℕ) (both : ℕ)
  (h_total : total = 38)
  (h_football : football = 26)
  (h_tennis : tennis = 20)
  (h_both : both = 17) :
  total - (football + tennis - both) = 9 :=
by sorry

end NUMINAMATH_CALUDE_students_playing_neither_sport_l8_890


namespace NUMINAMATH_CALUDE_positive_integers_relation_l8_854

theorem positive_integers_relation (a b : ℕ) : 
  a > 0 → b > 0 → (a, b) ≠ (1, 1) → (a * b - 1) ∣ (a^2 + b^2) → a^2 + b^2 = 5 * a * b - 5 := by
  sorry

end NUMINAMATH_CALUDE_positive_integers_relation_l8_854


namespace NUMINAMATH_CALUDE_complex_number_properties_l8_817

theorem complex_number_properties : ∃ (z : ℂ), 
  z = 2 / (1 - Complex.I) ∧ 
  Complex.abs z = Real.sqrt 2 ∧
  z^2 = 2 * Complex.I ∧
  z^2 - 2*z + 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_properties_l8_817


namespace NUMINAMATH_CALUDE_unique_balance_point_condition_l8_810

/-- A function f has a unique balance point if there exists a unique t such that f(t) = t -/
def has_unique_balance_point (f : ℝ → ℝ) : Prop :=
  ∃! t : ℝ, f t = t

/-- The quadratic function we're analyzing -/
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 - 3 * x + 2 * m

/-- Theorem stating the conditions for a unique balance point -/
theorem unique_balance_point_condition (m : ℝ) :
  has_unique_balance_point (f m) ↔ m = 2 ∨ m = -1 ∨ m = 1 := by sorry

end NUMINAMATH_CALUDE_unique_balance_point_condition_l8_810


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l8_822

theorem consecutive_integers_sum (a b c : ℤ) : 
  (b = a + 1) → (c = b + 1) → (a * b * c = 336) → (a + b + c = 21) := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l8_822


namespace NUMINAMATH_CALUDE_finite_valid_hexagon_angles_l8_867

/-- Represents a sequence of interior angles of a hexagon -/
structure HexagonAngles where
  x : ℕ
  d : ℕ

/-- Checks if a given HexagonAngles satisfies the required conditions -/
def isValidHexagonAngles (angles : HexagonAngles) : Prop :=
  angles.x > 30 ∧
  angles.x + 5 * angles.d < 150 ∧
  2 * angles.x + 5 * angles.d = 240

/-- The set of all valid HexagonAngles -/
def validHexagonAnglesSet : Set HexagonAngles :=
  {angles | isValidHexagonAngles angles}

theorem finite_valid_hexagon_angles : Set.Finite validHexagonAnglesSet := by
  sorry

end NUMINAMATH_CALUDE_finite_valid_hexagon_angles_l8_867


namespace NUMINAMATH_CALUDE_polynomial_simplification_l8_898

theorem polynomial_simplification (x : ℝ) :
  (3*x^2 - 2*x + 5)*(x - 2) - (x - 2)*(2*x^2 + 5*x - 8) + (2*x - 3)*(x - 2)*(x + 4) = 3*x^3 - 8*x^2 + 5*x - 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l8_898


namespace NUMINAMATH_CALUDE_rectangular_field_area_l8_823

theorem rectangular_field_area (x y : ℝ) 
  (h1 : x + y = 20) 
  (h2 : y = 9) : 
  x * y = 99 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l8_823


namespace NUMINAMATH_CALUDE_fraction_zero_implies_a_equals_two_l8_803

theorem fraction_zero_implies_a_equals_two (a : ℝ) 
  (h1 : (a^2 - 4) / (a + 2) = 0) 
  (h2 : a + 2 ≠ 0) : 
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_a_equals_two_l8_803


namespace NUMINAMATH_CALUDE_quadratic_solution_l8_864

theorem quadratic_solution : ∃! x : ℝ, x > 0 ∧ 3 * x^2 + 13 * x - 10 = 0 :=
by
  use 2/3
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l8_864


namespace NUMINAMATH_CALUDE_f_increasing_on_2_3_l8_845

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x
def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y
def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

-- State the theorem
theorem f_increasing_on_2_3 (heven : is_even f) (hperiodic : is_periodic f 2) 
  (hdecr : is_decreasing_on f (-1) 0) : is_increasing_on f 2 3 := by sorry

end NUMINAMATH_CALUDE_f_increasing_on_2_3_l8_845


namespace NUMINAMATH_CALUDE_extra_time_at_reduced_speed_l8_838

theorem extra_time_at_reduced_speed 
  (usual_time : ℝ) 
  (speed_ratio : ℝ) 
  (h1 : usual_time = 72.00000000000001)
  (h2 : speed_ratio = 0.75) : 
  (usual_time / speed_ratio) - usual_time = 24 := by
sorry

end NUMINAMATH_CALUDE_extra_time_at_reduced_speed_l8_838


namespace NUMINAMATH_CALUDE_polynomial_simplification_l8_861

theorem polynomial_simplification (q : ℝ) :
  (5 * q^4 - 4 * q^3 + 7 * q - 8) + (3 - 5 * q^2 + q^3 - 2 * q) =
  5 * q^4 - 3 * q^3 - 5 * q^2 + 5 * q - 5 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l8_861


namespace NUMINAMATH_CALUDE_optimal_greening_arrangement_l8_882

/-- Represents a construction team with daily greening area and cost -/
structure Team where
  daily_area : ℝ
  daily_cost : ℝ

/-- The optimal greening arrangement problem -/
def OptimalGreeningArrangement (total_area : ℝ) (max_days : ℕ) (team_a team_b : Team) : Prop :=
  -- Team A is 1.8 times more efficient than Team B
  team_a.daily_area = 1.8 * team_b.daily_area ∧
  -- Team A takes 4 days less than Team B for 450 m²
  (450 / team_a.daily_area) + 4 = 450 / team_b.daily_area ∧
  -- Optimal arrangement
  ∃ (days_a days_b : ℕ),
    -- Total area constraint
    team_a.daily_area * days_a + team_b.daily_area * days_b ≥ total_area ∧
    -- Time constraint
    days_a + days_b ≤ max_days ∧
    -- Optimal solution
    days_a = 30 ∧ days_b = 18 ∧
    -- Minimum cost
    team_a.daily_cost * days_a + team_b.daily_cost * days_b = 40.5

/-- Theorem stating the optimal greening arrangement -/
theorem optimal_greening_arrangement :
  OptimalGreeningArrangement 3600 48
    (Team.mk 90 1.05)
    (Team.mk 50 0.5) := by
  sorry

end NUMINAMATH_CALUDE_optimal_greening_arrangement_l8_882


namespace NUMINAMATH_CALUDE_walnut_trees_before_planting_l8_881

theorem walnut_trees_before_planting 
  (total_after : ℕ) 
  (planted : ℕ) 
  (h1 : total_after = 55) 
  (h2 : planted = 33) :
  total_after - planted = 22 := by
  sorry

end NUMINAMATH_CALUDE_walnut_trees_before_planting_l8_881


namespace NUMINAMATH_CALUDE_makeup_exam_average_score_l8_802

/-- Represents the average score of students who took the exam on the make-up date -/
def makeup_avg : ℝ := 90

theorem makeup_exam_average_score 
  (total_students : ℕ) 
  (assigned_day_percent : ℝ) 
  (assigned_day_avg : ℝ) 
  (total_avg : ℝ) 
  (h1 : total_students = 100)
  (h2 : assigned_day_percent = 70)
  (h3 : assigned_day_avg = 60)
  (h4 : total_avg = 69) :
  makeup_avg = 90 := by
  sorry

#check makeup_exam_average_score

end NUMINAMATH_CALUDE_makeup_exam_average_score_l8_802


namespace NUMINAMATH_CALUDE_min_crooks_proof_l8_829

/-- Represents the total number of ministers -/
def total_ministers : ℕ := 100

/-- Represents the size of any subgroup of ministers that must contain at least one crook -/
def subgroup_size : ℕ := 10

/-- Represents the property that any subgroup of ministers contains at least one crook -/
def at_least_one_crook (num_crooks : ℕ) : Prop :=
  ∀ (subgroup : Finset ℕ), subgroup.card = subgroup_size → 
    (total_ministers - num_crooks < subgroup.card)

/-- The minimum number of crooks in the cabinet -/
def min_crooks : ℕ := total_ministers - (subgroup_size - 1)

theorem min_crooks_proof :
  (at_least_one_crook min_crooks) ∧ 
  (∀ k < min_crooks, ¬(at_least_one_crook k)) :=
sorry

end NUMINAMATH_CALUDE_min_crooks_proof_l8_829


namespace NUMINAMATH_CALUDE_min_value_A_over_C_l8_877

theorem min_value_A_over_C (x A C : ℝ) (hx : x > 0) (hA : A > 0) (hC : C > 0)
  (h1 : x^2 + 1/x^2 = A) (h2 : x - 1/x = C) (h3 : C = Real.sqrt 3) :
  A / C ≥ 5 * Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_min_value_A_over_C_l8_877


namespace NUMINAMATH_CALUDE_max_sum_of_goods_l8_869

theorem max_sum_of_goods (m n : ℕ) : m > 0 ∧ n > 0 ∧ 5 * m + 17 * n = 203 → m + n ≤ 31 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_goods_l8_869


namespace NUMINAMATH_CALUDE_red_balls_count_l8_851

/-- The number of blue balls in the bin -/
def blue_balls : ℕ := 7

/-- The amount won when drawing a blue ball -/
def blue_win : ℤ := 3

/-- The amount lost when drawing a red ball -/
def red_loss : ℤ := 1

/-- The expected value of the game -/
def expected_value : ℚ := 1

/-- The number of red balls in the bin -/
def red_balls : ℕ := sorry

theorem red_balls_count : red_balls = 7 := by sorry

end NUMINAMATH_CALUDE_red_balls_count_l8_851


namespace NUMINAMATH_CALUDE_combined_work_time_l8_806

-- Define the work rates for A and B
def work_rate_A : ℚ := 1 / 4
def work_rate_B : ℚ := 1 / 12

-- Define the combined work rate
def combined_work_rate : ℚ := work_rate_A + work_rate_B

-- Theorem statement
theorem combined_work_time :
  (1 : ℚ) / combined_work_rate = 3 := by sorry

end NUMINAMATH_CALUDE_combined_work_time_l8_806


namespace NUMINAMATH_CALUDE_greene_nursery_flower_count_l8_811

theorem greene_nursery_flower_count : 
  let red_roses : ℕ := 1491
  let yellow_carnations : ℕ := 3025
  let white_roses : ℕ := 1768
  red_roses + yellow_carnations + white_roses = 6284 :=
by sorry

end NUMINAMATH_CALUDE_greene_nursery_flower_count_l8_811


namespace NUMINAMATH_CALUDE_five_digit_palindromes_count_l8_850

/-- A function that returns the number of positive five-digit palindromic integers -/
def count_five_digit_palindromes : ℕ :=
  9 * 10 * 10

/-- Theorem stating that the number of positive five-digit palindromic integers is 900 -/
theorem five_digit_palindromes_count :
  count_five_digit_palindromes = 900 := by
  sorry

#eval count_five_digit_palindromes

end NUMINAMATH_CALUDE_five_digit_palindromes_count_l8_850


namespace NUMINAMATH_CALUDE_die_roll_probabilities_l8_830

-- Define the type for a single die roll
def DieRoll : Type := Fin 6

-- Define the sample space for two die rolls
def SampleSpace : Type := DieRoll × DieRoll

-- Define the probability measure
noncomputable def prob : Set SampleSpace → ℝ := sorry

-- Define the event "sum is 5"
def sum_is_5 (roll : SampleSpace) : Prop :=
  roll.1.val + roll.2.val + 2 = 5

-- Define the event "at least one roll is odd"
def at_least_one_odd (roll : SampleSpace) : Prop :=
  roll.1.val % 2 = 0 ∨ roll.2.val % 2 = 0

-- State the theorem
theorem die_roll_probabilities :
  (prob {roll : SampleSpace | sum_is_5 roll} = 1/9) ∧
  (prob {roll : SampleSpace | at_least_one_odd roll} = 3/4) := by sorry

end NUMINAMATH_CALUDE_die_roll_probabilities_l8_830


namespace NUMINAMATH_CALUDE_intersection_value_l8_834

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

/-- Represents a curve in polar coordinates -/
structure PolarCurve where
  equation : PolarPoint → Prop

def C₁ : PolarCurve :=
  { equation := fun p => p.ρ * (Real.cos p.θ + Real.sin p.θ) = 1 }

def C₂ (a : ℝ) : PolarCurve :=
  { equation := fun p => p.ρ = a }

def onPolarAxis (p : PolarPoint) : Prop :=
  p.θ = 0 ∨ p.θ = Real.pi

theorem intersection_value (a : ℝ) (h₁ : a > 0) :
  (∃ p : PolarPoint, C₁.equation p ∧ (C₂ a).equation p ∧ onPolarAxis p) →
  a = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_value_l8_834


namespace NUMINAMATH_CALUDE_total_spent_equals_sum_of_items_l8_814

/-- The total amount Joan spent on toys and clothes -/
def total_spent_on_toys_and_clothes : ℚ := 60.10

/-- The cost of toy cars -/
def toy_cars_cost : ℚ := 14.88

/-- The cost of the skateboard -/
def skateboard_cost : ℚ := 4.88

/-- The cost of toy trucks -/
def toy_trucks_cost : ℚ := 5.86

/-- The cost of pants -/
def pants_cost : ℚ := 14.55

/-- The cost of the shirt -/
def shirt_cost : ℚ := 7.43

/-- The cost of the hat -/
def hat_cost : ℚ := 12.50

/-- Theorem stating that the sum of the costs of toys and clothes equals the total amount spent -/
theorem total_spent_equals_sum_of_items :
  toy_cars_cost + skateboard_cost + toy_trucks_cost + pants_cost + shirt_cost + hat_cost = total_spent_on_toys_and_clothes :=
by sorry

end NUMINAMATH_CALUDE_total_spent_equals_sum_of_items_l8_814


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l8_857

theorem negation_of_universal_statement (S : Set ℝ) :
  (¬ ∀ x ∈ S, |x - 1| ≠ 2) ↔ (∃ x ∈ S, |x - 1| = 2) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l8_857


namespace NUMINAMATH_CALUDE_elevator_weight_average_l8_840

theorem elevator_weight_average (initial_people : Nat) (initial_avg_weight : ℝ) (new_person_weight : ℝ) :
  initial_people = 6 →
  initial_avg_weight = 156 →
  new_person_weight = 121 →
  let total_weight := initial_people * initial_avg_weight + new_person_weight
  let new_people_count := initial_people + 1
  let new_avg_weight := total_weight / new_people_count
  new_avg_weight = 151 := by
sorry

end NUMINAMATH_CALUDE_elevator_weight_average_l8_840


namespace NUMINAMATH_CALUDE_bryden_received_value_l8_885

/-- The face value of a state quarter in dollars -/
def quarter_value : ℚ := 1/4

/-- The number of state quarters Bryden has -/
def bryden_quarters : ℕ := 10

/-- The percentage of face value the collector offers -/
def collector_offer_percentage : ℕ := 1500

/-- The amount Bryden will receive for his quarters in dollars -/
def bryden_received : ℚ := (bryden_quarters : ℚ) * quarter_value * (collector_offer_percentage : ℚ) / 100

theorem bryden_received_value : bryden_received = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_bryden_received_value_l8_885


namespace NUMINAMATH_CALUDE_simplify_expression_l8_891

theorem simplify_expression (x : ℝ) : 2*x + 3 - 4*x - 5 + 6*x + 7 - 8*x - 9 = -4*x - 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l8_891


namespace NUMINAMATH_CALUDE_lower_variance_less_volatile_l8_868

/-- Represents a shooter's performance --/
structure ShooterPerformance where
  average_score : ℝ
  variance : ℝ
  num_shots : ℕ

/-- Defines volatility based on variance --/
def less_volatile (a b : ShooterPerformance) : Prop :=
  a.variance < b.variance

/-- Theorem: Given two shooters with the same average score but different variances,
    the shooter with the lower variance has less volatile performance --/
theorem lower_variance_less_volatile (a b : ShooterPerformance) 
  (h1 : a.average_score = b.average_score)
  (h2 : a.variance ≠ b.variance)
  (h3 : a.num_shots = b.num_shots)
  : less_volatile (if a.variance < b.variance then a else b) (if a.variance > b.variance then a else b) :=
by
  sorry

end NUMINAMATH_CALUDE_lower_variance_less_volatile_l8_868


namespace NUMINAMATH_CALUDE_lansing_new_students_average_l8_855

/-- The average number of new students per school in Lansing -/
def average_new_students_per_school (total_schools : Float) (total_new_students : Float) : Float :=
  total_new_students / total_schools

/-- Theorem: The average number of new students per school in Lansing is 9.88 -/
theorem lansing_new_students_average :
  let total_schools : Float := 25.0
  let total_new_students : Float := 247.0
  average_new_students_per_school total_schools total_new_students = 9.88 := by
  sorry

end NUMINAMATH_CALUDE_lansing_new_students_average_l8_855


namespace NUMINAMATH_CALUDE_sum_of_common_divisors_l8_897

def numbers : List Int := [36, 72, -24, 120, 96]

def is_common_divisor (d : Nat) : Bool :=
  numbers.all (fun n => n % d = 0)

def common_divisors : List Nat :=
  (List.range 37).filter is_common_divisor

theorem sum_of_common_divisors :
  common_divisors.sum = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_common_divisors_l8_897


namespace NUMINAMATH_CALUDE_highway_speed_l8_847

/-- Prove that given the conditions, the average speed on the highway is 87 km/h -/
theorem highway_speed (total_distance : ℝ) (total_time : ℝ) (highway_time : ℝ) (city_time : ℝ) (city_speed : ℝ) :
  total_distance = 59 →
  total_time = 1 →
  highway_time = 1/3 →
  city_time = 2/3 →
  city_speed = 45 →
  (total_distance - city_speed * city_time) / highway_time = 87 := by
sorry

end NUMINAMATH_CALUDE_highway_speed_l8_847


namespace NUMINAMATH_CALUDE_dress_price_calculation_l8_813

def calculate_final_price (original_price : ℝ) (initial_discount : ℝ) (additional_discount : ℝ) (store_credit : ℝ) (sales_tax : ℝ) : ℝ :=
  let price_after_initial_discount := original_price * (1 - initial_discount)
  let price_after_additional_discount := price_after_initial_discount * (1 - additional_discount)
  let price_after_credit := price_after_additional_discount - store_credit
  let final_price := price_after_credit * (1 + sales_tax)
  final_price

theorem dress_price_calculation :
  calculate_final_price 50 0.3 0.2 10 0.075 = 19.35 := by
  sorry

end NUMINAMATH_CALUDE_dress_price_calculation_l8_813


namespace NUMINAMATH_CALUDE_students_on_south_side_l8_858

theorem students_on_south_side (total : ℕ) (difference : ℕ) (south : ℕ) : 
  total = 41 → difference = 3 → south = total / 2 + difference / 2 → south = 22 := by
  sorry

end NUMINAMATH_CALUDE_students_on_south_side_l8_858


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l8_893

/-- Given a quadratic equation x^2 + (2k-1)x + k^2 - k = 0 where x = 2 is one of the roots,
    prove that it has two distinct real roots and the value of -2k^2 - 6k - 5 is -1 -/
theorem quadratic_equation_properties (k : ℝ) :
  (∃ x : ℝ, x^2 + (2*k - 1)*x + k^2 - k = 0) →
  (2^2 + (2*k - 1)*2 + k^2 - k = 0) →
  (∃ x y : ℝ, x ≠ y ∧ x^2 + (2*k - 1)*x + k^2 - k = 0 ∧ y^2 + (2*k - 1)*y + k^2 - k = 0) ∧
  (-2*k^2 - 6*k - 5 = -1) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l8_893


namespace NUMINAMATH_CALUDE_red_rose_theatre_ticket_sales_l8_833

theorem red_rose_theatre_ticket_sales 
  (price_low : ℝ) 
  (price_high : ℝ) 
  (total_sales : ℝ) 
  (low_price_tickets : ℕ) 
  (h1 : price_low = 4.5)
  (h2 : price_high = 6)
  (h3 : total_sales = 1972.5)
  (h4 : low_price_tickets = 205) :
  ∃ (high_price_tickets : ℕ),
    (low_price_tickets : ℝ) * price_low + (high_price_tickets : ℝ) * price_high = total_sales ∧
    low_price_tickets + high_price_tickets = 380 :=
by sorry

end NUMINAMATH_CALUDE_red_rose_theatre_ticket_sales_l8_833


namespace NUMINAMATH_CALUDE_add_twenty_four_thirty_six_l8_859

theorem add_twenty_four_thirty_six : 24 + 36 = 60 := by
  sorry

end NUMINAMATH_CALUDE_add_twenty_four_thirty_six_l8_859


namespace NUMINAMATH_CALUDE_sun_division_l8_853

theorem sun_division (x y z : ℚ) : 
  (∀ r : ℚ, y = (45/100) * r → z = (30/100) * r) →  -- For each rupee x gets, y gets 45 paisa and z gets 30 paisa
  y = 54 →                                          -- Y's share is Rs. 54
  x + y + z = 210                                   -- Total amount is Rs. 210
  := by sorry

end NUMINAMATH_CALUDE_sun_division_l8_853


namespace NUMINAMATH_CALUDE_average_weight_section_B_l8_871

/-- Given a class with two sections A and B, prove the average weight of section B. -/
theorem average_weight_section_B 
  (students_A : ℕ) 
  (students_B : ℕ) 
  (avg_weight_A : ℝ) 
  (avg_weight_total : ℝ) 
  (h1 : students_A = 60)
  (h2 : students_B = 70)
  (h3 : avg_weight_A = 60)
  (h4 : avg_weight_total = 70.77) :
  ∃ avg_weight_B : ℝ, abs (avg_weight_B - 79.99) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_section_B_l8_871


namespace NUMINAMATH_CALUDE_nonagon_diagonal_intersections_l8_876

/-- The number of vertices in a regular nonagon -/
def n : ℕ := 9

/-- The number of distinct intersection points of diagonals in the interior of a regular nonagon -/
def intersection_points (n : ℕ) : ℕ := Nat.choose n 4

/-- Theorem: The number of distinct intersection points of diagonals in the interior of a regular nonagon is 126 -/
theorem nonagon_diagonal_intersections :
  intersection_points n = 126 := by sorry

end NUMINAMATH_CALUDE_nonagon_diagonal_intersections_l8_876


namespace NUMINAMATH_CALUDE_points_on_parabola_l8_807

-- Define the function y = x^2
def f (x : ℝ) : ℝ := x^2

-- Theorem statement
theorem points_on_parabola :
  ∀ t : ℝ, ∃ p₁ p₂ : ℝ × ℝ,
    p₁ = (1, f 1) ∧
    p₂ = (t, f t) ∧
    (p₁.2 = f p₁.1) ∧
    (p₂.2 = f p₂.1) :=
by
  sorry


end NUMINAMATH_CALUDE_points_on_parabola_l8_807


namespace NUMINAMATH_CALUDE_evaluate_expression_l8_863

theorem evaluate_expression : (3 + 2) * (3^2 + 2^2) * (3^4 + 2^4) = 6255 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l8_863


namespace NUMINAMATH_CALUDE_seconds_in_day_scientific_notation_l8_819

/-- The number of seconds in a day -/
def seconds_in_day : ℕ := 86400

/-- Scientific notation representation of seconds in a day -/
def scientific_notation : ℝ := 8.64 * (10 ^ 4)

theorem seconds_in_day_scientific_notation :
  (seconds_in_day : ℝ) = scientific_notation := by sorry

end NUMINAMATH_CALUDE_seconds_in_day_scientific_notation_l8_819


namespace NUMINAMATH_CALUDE_range_of_a_l8_842

-- Define the propositions p and q
def p (x : ℝ) : Prop := 1 ≤ x ∧ x < 3
def q (x a : ℝ) : Prop := x^2 - a*x ≤ x - a

-- Define the range of a
def a_range (a : ℝ) : Prop := 1 ≤ a ∧ a < 3

-- State the theorem
theorem range_of_a :
  (∀ x a : ℝ, (¬(p x) → ¬(q x a))) →
  (∀ a : ℝ, a_range a) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l8_842


namespace NUMINAMATH_CALUDE_first_month_sale_is_5400_l8_895

/-- Calculates the sale in the first month given the sales for the next 5 months and the average sale for 6 months -/
def first_month_sale (sale2 sale3 sale4 sale5 sale6 average : ℕ) : ℕ :=
  6 * average - (sale2 + sale3 + sale4 + sale5 + sale6)

/-- Theorem stating that the sale in the first month is 5400 given the specific sales figures -/
theorem first_month_sale_is_5400 :
  first_month_sale 9000 6300 7200 4500 1200 5600 = 5400 := by
  sorry

#eval first_month_sale 9000 6300 7200 4500 1200 5600

end NUMINAMATH_CALUDE_first_month_sale_is_5400_l8_895


namespace NUMINAMATH_CALUDE_custom_operation_result_l8_883

def custom_operation (A B : Set ℕ) : Set ℕ :=
  {x | x ∈ A ∪ B ∧ x ∉ A ∩ B}

theorem custom_operation_result :
  let A : Set ℕ := {1, 2, 3, 4, 5}
  let B : Set ℕ := {4, 5, 6}
  custom_operation A B = {1, 2, 3, 6} := by
  sorry

end NUMINAMATH_CALUDE_custom_operation_result_l8_883


namespace NUMINAMATH_CALUDE_problem_solution_l8_856

theorem problem_solution (m n : ℝ) (hm : m^2 - 2*m = 1) (hn : n^2 - 2*n = 1) (hne : m ≠ n) :
  (m + n) - (m * n) = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l8_856


namespace NUMINAMATH_CALUDE_sum_mod_thirteen_l8_824

theorem sum_mod_thirteen : (5678 + 5679 + 5680 + 5681) % 13 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_thirteen_l8_824


namespace NUMINAMATH_CALUDE_mixture_volume_l8_827

/-- Proves that the total volume of a mixture of two liquids is 4 liters -/
theorem mixture_volume (weight_a weight_b : ℝ) (ratio_a ratio_b : ℕ) (total_weight : ℝ) :
  weight_a = 950 →
  weight_b = 850 →
  ratio_a = 3 →
  ratio_b = 2 →
  total_weight = 3640 →
  ∃ (vol_a vol_b : ℝ),
    vol_a / vol_b = ratio_a / ratio_b ∧
    total_weight = vol_a * weight_a + vol_b * weight_b ∧
    vol_a + vol_b = 4 :=
by sorry

end NUMINAMATH_CALUDE_mixture_volume_l8_827
