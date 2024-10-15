import Mathlib

namespace NUMINAMATH_CALUDE_no_solution_in_range_l3816_381666

theorem no_solution_in_range (x y : ℕ+) (h : 3 * x^2 + x = 4 * y^2 + y) :
  x - y ≠ 2013 ∧ x - y ≠ 2014 ∧ x - y ≠ 2015 ∧ x - y ≠ 2016 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_in_range_l3816_381666


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3816_381630

theorem inequality_solution_set (x : ℝ) : 
  (((2 * x - 3) / (x + 4) > (4 * x + 1) / (3 * x - 2)) ↔ 
  (x > -4 ∧ x < (17 - Real.sqrt 201) / 4) ∨ 
  (x > (17 + Real.sqrt 201) / 4 ∧ x < 2 / 3)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3816_381630


namespace NUMINAMATH_CALUDE_goods_train_speed_l3816_381623

/-- Calculates the speed of a goods train given the conditions of the problem. -/
theorem goods_train_speed
  (man_train_speed : ℝ)
  (passing_time : ℝ)
  (goods_train_length : ℝ)
  (h1 : man_train_speed = 20)
  (h2 : passing_time = 9)
  (h3 : goods_train_length = 280) :
  ∃ (goods_train_speed : ℝ),
    goods_train_speed = 92 ∧
    (man_train_speed + goods_train_speed) * (1 / 3.6) = goods_train_length / passing_time :=
by sorry

end NUMINAMATH_CALUDE_goods_train_speed_l3816_381623


namespace NUMINAMATH_CALUDE_percentage_problem_l3816_381658

theorem percentage_problem (N : ℝ) (P : ℝ) (h1 : N = 140) 
  (h2 : (P / 100) * N = (4 / 5) * N - 21) : P = 65 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3816_381658


namespace NUMINAMATH_CALUDE_percentage_difference_l3816_381687

theorem percentage_difference (x y z : ℝ) (h1 : y = 1.2 * z) (h2 : z = 150) (h3 : x + y + z = 555) :
  (x - y) / y * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l3816_381687


namespace NUMINAMATH_CALUDE_zark_game_threshold_l3816_381626

/-- The score for dropping n zarks -/
def drop_score (n : ℕ) : ℕ := n^2

/-- The score for eating n zarks -/
def eat_score (n : ℕ) : ℕ := 15 * n

/-- 16 is the smallest positive integer n for which dropping n zarks scores more than eating them -/
theorem zark_game_threshold : ∀ n : ℕ, n > 0 → (drop_score n > eat_score n ↔ n ≥ 16) :=
by sorry

end NUMINAMATH_CALUDE_zark_game_threshold_l3816_381626


namespace NUMINAMATH_CALUDE_slope_at_negative_five_l3816_381698

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem slope_at_negative_five
  (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_diff : Differentiable ℝ f)
  (h_der_one : deriv f 1 = 1)
  (h_period : has_period f 4) :
  deriv f (-5) = -1 := by
  sorry

end NUMINAMATH_CALUDE_slope_at_negative_five_l3816_381698


namespace NUMINAMATH_CALUDE_absolute_difference_of_U_coordinates_l3816_381681

/-- Triangle PQR with vertices P(0,10), Q(5,0), and R(10,0) -/
def P : ℝ × ℝ := (0, 10)
def Q : ℝ × ℝ := (5, 0)
def R : ℝ × ℝ := (10, 0)

/-- V is on QR and 3 units away from Q -/
def V : ℝ × ℝ := (2, 0)

/-- U is on PR and has the same x-coordinate as V -/
def U : ℝ × ℝ := (2, 8)

/-- The theorem to be proved -/
theorem absolute_difference_of_U_coordinates : 
  |U.2 - U.1| = 6 := by sorry

end NUMINAMATH_CALUDE_absolute_difference_of_U_coordinates_l3816_381681


namespace NUMINAMATH_CALUDE_sahara_temperature_difference_l3816_381613

/-- The maximum temperature difference in the Sahara Desert --/
theorem sahara_temperature_difference (highest_temp lowest_temp : ℤ) 
  (h_highest : highest_temp = 58)
  (h_lowest : lowest_temp = -34) :
  highest_temp - lowest_temp = 92 := by
  sorry

end NUMINAMATH_CALUDE_sahara_temperature_difference_l3816_381613


namespace NUMINAMATH_CALUDE_three_digit_cube_divisible_by_16_l3816_381680

theorem three_digit_cube_divisible_by_16 : 
  ∃! n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ ∃ m : ℕ, n = m^3 ∧ 16 ∣ n :=
by sorry

end NUMINAMATH_CALUDE_three_digit_cube_divisible_by_16_l3816_381680


namespace NUMINAMATH_CALUDE_board_piece_difference_l3816_381616

def board_length : ℝ := 20
def shorter_piece : ℝ := 8

theorem board_piece_difference : 
  let longer_piece := board_length - shorter_piece
  2 * shorter_piece - longer_piece = 4 := by
  sorry

end NUMINAMATH_CALUDE_board_piece_difference_l3816_381616


namespace NUMINAMATH_CALUDE_ap_sum_70_l3816_381644

def arithmetic_progression (a d : ℚ) (n : ℕ) : ℚ :=
  n / 2 * (2 * a + (n - 1) * d)

theorem ap_sum_70 (a d : ℚ) :
  arithmetic_progression a d 20 = 150 →
  arithmetic_progression a d 50 = 20 →
  arithmetic_progression a d 70 = -910/3 := by
  sorry

end NUMINAMATH_CALUDE_ap_sum_70_l3816_381644


namespace NUMINAMATH_CALUDE_inequalities_and_minimum_l3816_381694

theorem inequalities_and_minimum (a b : ℝ) :
  (a > b ∧ b > 0 → a - 1/a > b - 1/b) ∧
  (a > 0 ∧ b > 0 ∧ 2*a + b = 1 → 2/a + 1/b ≥ 9) :=
by sorry

end NUMINAMATH_CALUDE_inequalities_and_minimum_l3816_381694


namespace NUMINAMATH_CALUDE_tangent_point_exists_l3816_381636

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

theorem tangent_point_exists : ∃ (x₀ y₀ : ℝ), 
  f x₀ = y₀ ∧ 
  f' x₀ = 4 ∧ 
  x₀ = -1 ∧ 
  y₀ = -4 :=
sorry

end NUMINAMATH_CALUDE_tangent_point_exists_l3816_381636


namespace NUMINAMATH_CALUDE_sum_of_cubes_of_roots_l3816_381614

theorem sum_of_cubes_of_roots (x₁ x₂ x₃ : ℂ) : 
  x₁^3 + x₂^3 + x₃^3 = 0 → x₁ + x₂ + x₃ = -2 → x₁*x₂ + x₂*x₃ + x₃*x₁ = 1 → x₁*x₂*x₃ = 3 → 
  x₁^3 + x₂^3 + x₃^3 = 7 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_cubes_of_roots_l3816_381614


namespace NUMINAMATH_CALUDE_custom_mult_theorem_l3816_381609

/-- Custom multiplication operation -/
def custom_mult (a b : ℝ) : ℝ := (a - b) ^ 2

/-- Theorem stating that ((x-y)^2 + 1) * ((y-x)^2 + 1) = 0 for the custom multiplication -/
theorem custom_mult_theorem (x y : ℝ) : 
  custom_mult ((x - y) ^ 2 + 1) ((y - x) ^ 2 + 1) = 0 := by
  sorry


end NUMINAMATH_CALUDE_custom_mult_theorem_l3816_381609


namespace NUMINAMATH_CALUDE_de_moivres_formula_l3816_381606

theorem de_moivres_formula (n : ℕ) (φ : ℝ) :
  (Complex.cos φ + Complex.I * Complex.sin φ) ^ n = Complex.cos (n * φ) + Complex.I * Complex.sin (n * φ) := by
  sorry

end NUMINAMATH_CALUDE_de_moivres_formula_l3816_381606


namespace NUMINAMATH_CALUDE_part_one_part_two_l3816_381683

-- Define the inequality
def inequality (a b x : ℝ) : Prop := a * x^2 - b ≥ 2 * x - a * x

-- Define the solution set
def solution_set (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ -1

-- Theorem for part (1)
theorem part_one (a b : ℝ) : 
  (∀ x, inequality a b x ↔ solution_set x) → a = -1 ∧ b = 2 := by sorry

-- Define the second inequality
def inequality_two (a x : ℝ) : Prop := (a * x - 2) * (x + 1) ≥ 0

-- Define the solution sets for part (2)
def solution_set_one (a x : ℝ) : Prop := 2 / a ≤ x ∧ x ≤ -1
def solution_set_two (x : ℝ) : Prop := x = -1
def solution_set_three (a x : ℝ) : Prop := -1 ≤ x ∧ x ≤ 2 / a

-- Theorem for part (2)
theorem part_two (a : ℝ) (h : a < 0) :
  (∀ x, inequality_two a x ↔ 
    ((-2 < a ∧ a < 0 ∧ solution_set_one a x) ∨
     (a = -2 ∧ solution_set_two x) ∨
     (a < -2 ∧ solution_set_three a x))) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3816_381683


namespace NUMINAMATH_CALUDE_starting_number_of_range_l3816_381601

theorem starting_number_of_range (n : ℕ) (h1 : n ≤ 31) (h2 : n % 3 = 0) 
  (h3 : ∀ k, n - 18 ≤ k ∧ k ≤ n → k % 3 = 0) : n - 18 = 12 := by
  sorry

end NUMINAMATH_CALUDE_starting_number_of_range_l3816_381601


namespace NUMINAMATH_CALUDE_daphnes_collection_height_l3816_381633

/-- Represents the height of a book collection in inches and pages -/
structure BookCollection where
  inches : ℝ
  pages : ℝ
  pages_per_inch : ℝ

/-- The problem statement -/
theorem daphnes_collection_height 
  (miles : BookCollection)
  (daphne : BookCollection)
  (longest_collection_pages : ℝ)
  (h1 : miles.pages_per_inch = 5)
  (h2 : daphne.pages_per_inch = 50)
  (h3 : miles.inches = 240)
  (h4 : longest_collection_pages = 1250)
  (h5 : longest_collection_pages ≥ miles.pages)
  (h6 : longest_collection_pages ≥ daphne.pages)
  (h7 : daphne.pages = longest_collection_pages) :
  daphne.inches = 25 := by
sorry

end NUMINAMATH_CALUDE_daphnes_collection_height_l3816_381633


namespace NUMINAMATH_CALUDE_negation_of_statement_l3816_381611

def S : Set Int := {1, -1, 0}

theorem negation_of_statement :
  (¬ ∀ x ∈ S, 2 * x + 1 > 0) ↔ (∃ x ∈ S, 2 * x + 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_statement_l3816_381611


namespace NUMINAMATH_CALUDE_equilateral_triangle_intersection_l3816_381653

theorem equilateral_triangle_intersection (a : ℝ) : 
  (∃ (A B : ℝ × ℝ), 
    (A.1 + A.2 = Real.sqrt 3 * a ∧ A.1^2 + A.2^2 = a^2 + (a-1)^2) ∧
    (B.1 + B.2 = Real.sqrt 3 * a ∧ B.1^2 + B.2^2 = a^2 + (a-1)^2) ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = A.1^2 + A.2^2 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = B.1^2 + B.2^2) →
  a = 1/2 := by
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_intersection_l3816_381653


namespace NUMINAMATH_CALUDE_operation_on_original_number_l3816_381682

theorem operation_on_original_number : ∃ (f : ℝ → ℝ), 
  (3 * (f 4 + 9) = 51) ∧ (f 4 = 2 * 4) := by
  sorry

end NUMINAMATH_CALUDE_operation_on_original_number_l3816_381682


namespace NUMINAMATH_CALUDE_bryans_bookshelves_l3816_381645

theorem bryans_bookshelves (total_books : ℕ) (books_per_shelf : ℕ) (h1 : total_books = 38) (h2 : books_per_shelf = 2) :
  total_books / books_per_shelf = 19 := by
  sorry

end NUMINAMATH_CALUDE_bryans_bookshelves_l3816_381645


namespace NUMINAMATH_CALUDE_point_distance_3d_l3816_381685

/-- Given two points A(m, 2, 3) and B(1, -1, 1) in 3D space with distance √13 between them, m = 1 -/
theorem point_distance_3d (m : ℝ) : 
  let A : ℝ × ℝ × ℝ := (m, 2, 3)
  let B : ℝ × ℝ × ℝ := (1, -1, 1)
  (m - 1)^2 + 3^2 + 2^2 = 13 → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_point_distance_3d_l3816_381685


namespace NUMINAMATH_CALUDE_ab_value_l3816_381634

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 29) : a * b = 10 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l3816_381634


namespace NUMINAMATH_CALUDE_book_pages_count_l3816_381673

theorem book_pages_count :
  ∀ (P : ℕ),
  (P / 2 : ℕ) = P / 2 →  -- Half of the pages are filled with images
  (P - (P / 2 + 11)) / 2 = 19 →  -- Remaining pages after images and intro, half of which are text
  P = 98 :=
by sorry

end NUMINAMATH_CALUDE_book_pages_count_l3816_381673


namespace NUMINAMATH_CALUDE_solve_equation_l3816_381628

theorem solve_equation (x : ℝ) (h : 3 * x + 15 = (1 / 3) * (6 * x + 45)) : x = 0 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3816_381628


namespace NUMINAMATH_CALUDE_tv_ad_sequences_l3816_381656

/-- Represents the number of different broadcast sequences for advertisements -/
def num_broadcast_sequences (total_ads : ℕ) (commercial_ads : ℕ) (public_service_ads : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of different broadcast sequences for the given conditions -/
theorem tv_ad_sequences :
  let total_ads := 5
  let commercial_ads := 3
  let public_service_ads := 2
  num_broadcast_sequences total_ads commercial_ads public_service_ads = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_tv_ad_sequences_l3816_381656


namespace NUMINAMATH_CALUDE_password_identification_l3816_381638

def is_valid_password (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ n % 9 = 0 ∧ n / 1000 = 5

def alice_knows (n : ℕ) : Prop :=
  ∃ a b : ℕ, n / 100 % 10 = a ∧ n / 10 % 10 = b

def bob_knows (n : ℕ) : Prop :=
  ∃ b c : ℕ, n / 10 % 10 = b ∧ n % 10 = c

def initially_unknown (n : ℕ) : Prop :=
  ∃ m : ℕ, m ≠ n ∧ is_valid_password m ∧ alice_knows m ∧ bob_knows m

theorem password_identification :
  ∃ n : ℕ,
    is_valid_password n ∧
    alice_knows n ∧
    bob_knows n ∧
    initially_unknown n ∧
    (∀ m : ℕ, is_valid_password m ∧ alice_knows m ∧ bob_knows m ∧ initially_unknown m → m ≤ n) ∧
    n = 5940 :=
  sorry

end NUMINAMATH_CALUDE_password_identification_l3816_381638


namespace NUMINAMATH_CALUDE_min_value_of_f_l3816_381654

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + a

-- Define the interval
def I : Set ℝ := Set.Icc (-1) 2

-- State the theorem
theorem min_value_of_f (a : ℝ) :
  (∃ x ∈ I, ∀ y ∈ I, f a y ≤ f a x) ∧ (f a 2 = 20) →
  (∃ x ∈ I, ∀ y ∈ I, f a x ≤ f a y) ∧ (f a (-1) = -7) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3816_381654


namespace NUMINAMATH_CALUDE_largest_possible_BD_l3816_381642

/-- A cyclic quadrilateral with side lengths that are distinct primes less than 20 -/
structure CyclicQuadrilateral where
  AB : ℕ
  BC : ℕ
  CD : ℕ
  DA : ℕ
  cyclic : Bool
  distinct_primes : AB ≠ BC ∧ AB ≠ CD ∧ AB ≠ DA ∧ BC ≠ CD ∧ BC ≠ DA ∧ CD ≠ DA
  all_prime : Nat.Prime AB ∧ Nat.Prime BC ∧ Nat.Prime CD ∧ Nat.Prime DA
  all_less_than_20 : AB < 20 ∧ BC < 20 ∧ CD < 20 ∧ DA < 20
  AB_is_11 : AB = 11
  product_condition : BC * CD = AB * DA

/-- The diagonal BD of the cyclic quadrilateral -/
def diagonal_BD (q : CyclicQuadrilateral) : ℝ := sorry

theorem largest_possible_BD (q : CyclicQuadrilateral) :
  ∃ (max_bd : ℝ), diagonal_BD q ≤ max_bd ∧ max_bd = Real.sqrt 290 := by
  sorry

end NUMINAMATH_CALUDE_largest_possible_BD_l3816_381642


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3816_381689

/-- The equation of a hyperbola passing through (6, √3) with asymptotes y = ±x/3 -/
theorem hyperbola_equation (x y : ℝ) :
  (∀ k : ℝ, k * x = 3 * y → k = 1 ∨ k = -1) →  -- asymptotes condition
  6^2 / 9 - (Real.sqrt 3)^2 = 1 →               -- point condition
  x^2 / 9 - y^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3816_381689


namespace NUMINAMATH_CALUDE_factor_expression_l3816_381690

theorem factor_expression (y z : ℝ) : 64 - 16 * y^2 * z^2 = 16 * (2 - y*z) * (2 + y*z) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3816_381690


namespace NUMINAMATH_CALUDE_distance_XY_is_24_l3816_381641

/-- The distance between points X and Y in miles. -/
def distance_XY : ℝ := 24

/-- Yolanda's walking rate in miles per hour. -/
def yolanda_rate : ℝ := 3

/-- Bob's walking rate in miles per hour. -/
def bob_rate : ℝ := 4

/-- The distance Bob has walked when they meet, in miles. -/
def bob_distance : ℝ := 12

/-- The time difference between Yolanda and Bob's start, in hours. -/
def time_difference : ℝ := 1

theorem distance_XY_is_24 : 
  distance_XY = yolanda_rate * (bob_distance / bob_rate + time_difference) + bob_distance :=
sorry

end NUMINAMATH_CALUDE_distance_XY_is_24_l3816_381641


namespace NUMINAMATH_CALUDE_absolute_value_not_positive_l3816_381607

theorem absolute_value_not_positive (y : ℚ) : |5 * y - 3| ≤ 0 ↔ y = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_not_positive_l3816_381607


namespace NUMINAMATH_CALUDE_maggie_bouncy_balls_l3816_381663

/-- The number of packs of red bouncy balls -/
def red_packs : ℕ := 4

/-- The number of balls in each pack of red bouncy balls -/
def red_per_pack : ℕ := 12

/-- The number of packs of yellow bouncy balls -/
def yellow_packs : ℕ := 8

/-- The number of balls in each pack of yellow bouncy balls -/
def yellow_per_pack : ℕ := 10

/-- The number of packs of green bouncy balls -/
def green_packs : ℕ := 4

/-- The number of balls in each pack of green bouncy balls -/
def green_per_pack : ℕ := 14

/-- The number of packs of blue bouncy balls -/
def blue_packs : ℕ := 6

/-- The number of balls in each pack of blue bouncy balls -/
def blue_per_pack : ℕ := 8

/-- The total number of bouncy balls Maggie bought -/
def total_balls : ℕ := red_packs * red_per_pack + yellow_packs * yellow_per_pack + 
                        green_packs * green_per_pack + blue_packs * blue_per_pack

theorem maggie_bouncy_balls : total_balls = 232 := by
  sorry

end NUMINAMATH_CALUDE_maggie_bouncy_balls_l3816_381663


namespace NUMINAMATH_CALUDE_no_solution_for_all_a_b_l3816_381650

theorem no_solution_for_all_a_b : ∃ (a b : ℤ), a ≠ 0 ∧ b ≠ 0 ∧
  ¬∃ (x y : ℝ), (Real.tan (13 * x) * Real.tan (a * y) = 1) ∧
                (Real.tan (21 * x) * Real.tan (b * y) = 1) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_for_all_a_b_l3816_381650


namespace NUMINAMATH_CALUDE_cubic_divisibility_l3816_381664

theorem cubic_divisibility (t : ℤ) : (((125 * t - 12) ^ 3 + 2 * (125 * t - 12) + 2) % 125 = 0) := by
  sorry

end NUMINAMATH_CALUDE_cubic_divisibility_l3816_381664


namespace NUMINAMATH_CALUDE_sin_cos_theorem_l3816_381635

theorem sin_cos_theorem (θ : ℝ) (z : ℂ) : 
  z = (Real.sin θ - 2 * Real.cos θ) + (Real.sin θ + 2 * Real.cos θ) * Complex.I →
  z.re = 0 →
  z.im ≠ 0 →
  Real.sin θ * Real.cos θ = 2/5 :=
by sorry

end NUMINAMATH_CALUDE_sin_cos_theorem_l3816_381635


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l3816_381643

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c
  (∃ x y : ℝ, f x = 0 ∧ f y = 0 ∧ x ≠ y) →
  (∃ s : ℝ, s = -(b / a) ∧ s = x + y) :=
by sorry

theorem sum_of_roots_specific_equation :
  let f : ℝ → ℝ := λ x ↦ x^2 + 2023 * x - 2024
  (∃ x y : ℝ, f x = 0 ∧ f y = 0 ∧ x ≠ y) →
  (∃ s : ℝ, s = -2023 ∧ s = x + y) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l3816_381643


namespace NUMINAMATH_CALUDE_line_slope_hyperbola_intersection_l3816_381660

/-- A line intersecting a hyperbola x^2 - y^2 = 1 at two points has a slope of 2 
    if the midpoint of the line segment between these points is (2,1) -/
theorem line_slope_hyperbola_intersection (A B : ℝ × ℝ) : 
  (A.1^2 - A.2^2 = 1) →  -- A is on the hyperbola
  (B.1^2 - B.2^2 = 1) →  -- B is on the hyperbola
  ((A.1 + B.1) / 2 = 2 ∧ (A.2 + B.2) / 2 = 1) →  -- Midpoint is (2,1)
  (B.2 - A.2) / (B.1 - A.1) = 2 :=  -- Slope is 2
by sorry

end NUMINAMATH_CALUDE_line_slope_hyperbola_intersection_l3816_381660


namespace NUMINAMATH_CALUDE_minimum_distance_point_to_curve_l3816_381637

open Real

theorem minimum_distance_point_to_curve (t m : ℝ) : 
  (∃ (P : ℝ × ℝ), 
    P.2 = exp P.1 ∧ 
    (∀ (Q : ℝ × ℝ), Q.2 = exp Q.1 → (t - P.1)^2 + P.2^2 ≤ (t - Q.1)^2 + Q.2^2) ∧
    (t - P.1)^2 + P.2^2 = 12) →
  t = 3 + log 3 / 2 := by
sorry


end NUMINAMATH_CALUDE_minimum_distance_point_to_curve_l3816_381637


namespace NUMINAMATH_CALUDE_batsman_matches_l3816_381652

theorem batsman_matches (total_matches : ℕ) (last_matches : ℕ) (last_avg : ℚ) (overall_avg : ℚ) :
  total_matches = 35 →
  last_matches = 13 →
  last_avg = 15 →
  overall_avg = 23.17142857142857 →
  total_matches - last_matches = 22 :=
by sorry

end NUMINAMATH_CALUDE_batsman_matches_l3816_381652


namespace NUMINAMATH_CALUDE_consecutive_integers_product_sum_l3816_381661

theorem consecutive_integers_product_sum (x : ℕ) : 
  x > 0 ∧ x * (x + 1) = 812 → x + (x + 1) = 57 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_sum_l3816_381661


namespace NUMINAMATH_CALUDE_equilateral_triangle_x_value_l3816_381665

/-- An equilateral triangle with side lengths expressed in terms of x -/
structure EquilateralTriangle where
  x : ℝ
  side_length : ℝ
  eq_sides : side_length = 4 * x ∧ side_length = x + 12

theorem equilateral_triangle_x_value (t : EquilateralTriangle) : t.x = 4 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_x_value_l3816_381665


namespace NUMINAMATH_CALUDE_point_on_line_l3816_381674

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem point_on_line (p1 p2 p3 p4 : Point) :
  p1 = Point.mk 2 5 →
  p2 = Point.mk 4 11 →
  p3 = Point.mk 6 17 →
  p4 = Point.mk 15 44 →
  collinear p1 p2 p3 →
  collinear p1 p2 p4 :=
by
  sorry


end NUMINAMATH_CALUDE_point_on_line_l3816_381674


namespace NUMINAMATH_CALUDE_dhoni_leftover_percentage_l3816_381662

/-- The percentage of earnings Dhoni had left over after spending on rent and a dishwasher -/
theorem dhoni_leftover_percentage : ℝ := by
  -- Define the percentage spent on rent
  let rent_percentage : ℝ := 20
  -- Define the percentage spent on dishwasher (5% less than rent)
  let dishwasher_percentage : ℝ := rent_percentage - 5
  -- Define the total percentage spent
  let total_spent_percentage : ℝ := rent_percentage + dishwasher_percentage
  -- Define the leftover percentage
  let leftover_percentage : ℝ := 100 - total_spent_percentage
  -- Prove that the leftover percentage is 65%
  have : leftover_percentage = 65 := by sorry
  -- Return the result
  exact leftover_percentage

end NUMINAMATH_CALUDE_dhoni_leftover_percentage_l3816_381662


namespace NUMINAMATH_CALUDE_lewis_speed_l3816_381669

/-- Proves that Lewis's speed is 80 mph given the problem conditions -/
theorem lewis_speed (john_speed : ℝ) (total_distance : ℝ) (meeting_distance : ℝ) :
  john_speed = 40 ∧ 
  total_distance = 240 ∧ 
  meeting_distance = 160 →
  (total_distance + (total_distance - meeting_distance)) / (meeting_distance / john_speed) = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_lewis_speed_l3816_381669


namespace NUMINAMATH_CALUDE_yellow_balls_count_l3816_381604

theorem yellow_balls_count (total : ℕ) (white green red purple : ℕ) (prob : ℚ) :
  total = 100 →
  white = 50 →
  green = 30 →
  red = 9 →
  purple = 3 →
  prob = 88/100 →
  prob = (white + green + (total - white - green - red - purple)) / total →
  total - white - green - red - purple = 8 :=
by sorry

end NUMINAMATH_CALUDE_yellow_balls_count_l3816_381604


namespace NUMINAMATH_CALUDE_survey_result_l3816_381655

theorem survey_result (total : ℕ) (migraines insomnia anxiety : ℕ)
  (migraines_insomnia migraines_anxiety insomnia_anxiety : ℕ)
  (all_three : ℕ) :
  total = 150 →
  migraines = 90 →
  insomnia = 60 →
  anxiety = 30 →
  migraines_insomnia = 20 →
  migraines_anxiety = 10 →
  insomnia_anxiety = 15 →
  all_three = 5 →
  total - (migraines + insomnia + anxiety - migraines_insomnia - migraines_anxiety - insomnia_anxiety + all_three) = 40 := by
  sorry

#check survey_result

end NUMINAMATH_CALUDE_survey_result_l3816_381655


namespace NUMINAMATH_CALUDE_even_blue_faces_count_l3816_381677

/-- Represents a cube with a certain number of blue faces -/
structure PaintedCube where
  blueFaces : Nat

/-- Represents the wooden block -/
structure WoodenBlock where
  length : Nat
  width : Nat
  height : Nat
  paintedSides : Nat

/-- Function to generate the list of cubes from a wooden block -/
def generateCubes (block : WoodenBlock) : List PaintedCube :=
  sorry

/-- Function to count cubes with even number of blue faces -/
def countEvenBlueFaces (cubes : List PaintedCube) : Nat :=
  sorry

/-- Main theorem -/
theorem even_blue_faces_count (block : WoodenBlock) 
    (h1 : block.length = 5)
    (h2 : block.width = 3)
    (h3 : block.height = 1)
    (h4 : block.paintedSides = 5) :
  countEvenBlueFaces (generateCubes block) = 5 := by
  sorry

end NUMINAMATH_CALUDE_even_blue_faces_count_l3816_381677


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3816_381629

theorem fraction_to_decimal (n d : ℕ) (h : d ≠ 0) :
  (n : ℚ) / d = 0.650 :=
by
  sorry

#check fraction_to_decimal 13 320

end NUMINAMATH_CALUDE_fraction_to_decimal_l3816_381629


namespace NUMINAMATH_CALUDE_system_two_solutions_l3816_381624

/-- The system of equations has exactly two solutions if and only if a = 1 or a = 25 -/
theorem system_two_solutions (a : ℝ) :
  (∃! (x₁ y₁ x₂ y₂ : ℝ), 
    (abs (y₁ - 3 - x₁) + abs (y₁ - 3 + x₁) = 6 ∧
     (abs x₁ - 4)^2 + (abs y₁ - 3)^2 = a) ∧
    (abs (y₂ - 3 - x₂) + abs (y₂ - 3 + x₂) = 6 ∧
     (abs x₂ - 4)^2 + (abs y₂ - 3)^2 = a) ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂)) ↔ 
  (a = 1 ∨ a = 25) :=
by sorry

end NUMINAMATH_CALUDE_system_two_solutions_l3816_381624


namespace NUMINAMATH_CALUDE_earphone_cost_l3816_381699

def mean_expenditure : ℝ := 500
def mon_expenditure : ℝ := 450
def tue_expenditure : ℝ := 600
def wed_expenditure : ℝ := 400
def thu_expenditure : ℝ := 500
def sat_expenditure : ℝ := 550
def sun_expenditure : ℝ := 300
def pen_cost : ℝ := 30
def notebook_cost : ℝ := 50
def num_days : ℕ := 7

theorem earphone_cost :
  let total_expenditure := mean_expenditure * num_days
  let known_expenditures := mon_expenditure + tue_expenditure + wed_expenditure + 
                            thu_expenditure + sat_expenditure + sun_expenditure
  let friday_expenditure := total_expenditure - known_expenditures
  let other_items_cost := pen_cost + notebook_cost
  friday_expenditure - other_items_cost = 620 := by
sorry

end NUMINAMATH_CALUDE_earphone_cost_l3816_381699


namespace NUMINAMATH_CALUDE_sum_and_double_l3816_381691

theorem sum_and_double : (142 + 29 + 26 + 14) * 2 = 422 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_double_l3816_381691


namespace NUMINAMATH_CALUDE_ab_and_c_values_l3816_381671

theorem ab_and_c_values (a b c : ℤ) 
  (h1 : a - b = 3) 
  (h2 : a^2 + b^2 = 29) 
  (h3 : a + b + c = 10) : 
  a * b = 10 ∧ (c = 3 ∨ c = 17) := by
sorry

end NUMINAMATH_CALUDE_ab_and_c_values_l3816_381671


namespace NUMINAMATH_CALUDE_hannahs_speed_l3816_381672

/-- Proves that Hannah's speed is 15 km/h given the problem conditions --/
theorem hannahs_speed (glen_speed : ℝ) (distance : ℝ) (time : ℝ) 
  (h_glen_speed : glen_speed = 37)
  (h_distance : distance = 130)
  (h_time : time = 5) :
  ∃ hannah_speed : ℝ, hannah_speed = 15 ∧ 
  2 * distance = (glen_speed + hannah_speed) * time :=
by sorry

end NUMINAMATH_CALUDE_hannahs_speed_l3816_381672


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l3816_381670

theorem tan_alpha_plus_pi_fourth (α : Real) 
  (h1 : α > 0) (h2 : α < Real.pi / 2) 
  (h3 : Real.cos (2 * α) + Real.cos α ^ 2 = 0) : 
  Real.tan (α + Real.pi / 4) = -3 - 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l3816_381670


namespace NUMINAMATH_CALUDE_complex_real_condition_l3816_381617

theorem complex_real_condition (m : ℝ) : 
  (((m : ℂ) + Complex.I) / (1 - Complex.I)).im = 0 → m = -1 :=
by sorry

end NUMINAMATH_CALUDE_complex_real_condition_l3816_381617


namespace NUMINAMATH_CALUDE_complex_number_range_angle_between_vectors_l3816_381647

-- Problem 1
theorem complex_number_range (Z : ℂ) (a : ℝ) 
  (h1 : (Z + 2*I).im = 0)
  (h2 : ((Z / (2 - I)).im = 0))
  (h3 : ((Z + a*I)^2).re > 0)
  (h4 : ((Z + a*I)^2).im > 0) :
  2 < a ∧ a < 6 := by sorry

-- Problem 2
theorem angle_between_vectors (z₁ z₂ : ℂ) 
  (h1 : z₁ = 3)
  (h2 : z₂ = -5 + 5*I) :
  Real.arccos ((z₁.re * z₂.re + z₁.im * z₂.im) / (Complex.abs z₁ * Complex.abs z₂)) = 3 * Real.pi / 4 := by sorry

end NUMINAMATH_CALUDE_complex_number_range_angle_between_vectors_l3816_381647


namespace NUMINAMATH_CALUDE_integral_equals_three_implies_k_equals_four_l3816_381618

theorem integral_equals_three_implies_k_equals_four (k : ℝ) : 
  (∫ x in (0:ℝ)..(1:ℝ), 3 * x^2 + k * x) = 3 → k = 4 := by
sorry

end NUMINAMATH_CALUDE_integral_equals_three_implies_k_equals_four_l3816_381618


namespace NUMINAMATH_CALUDE_ball_distribution_theorem_l3816_381600

/-- The number of ways to distribute indistinguishable balls into boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  Nat.choose (num_balls + num_boxes - 1) num_balls

/-- The number of ways to distribute balls of three colors into boxes -/
def distribute_three_colors (num_balls_per_color : ℕ) (num_boxes : ℕ) : ℕ :=
  (distribute_balls num_balls_per_color num_boxes) ^ 3

theorem ball_distribution_theorem (num_balls_per_color num_boxes : ℕ) 
  (h1 : num_balls_per_color = 4) 
  (h2 : num_boxes = 6) : 
  distribute_three_colors num_balls_per_color num_boxes = (Nat.choose 9 4) ^ 3 := by
  sorry

#eval distribute_three_colors 4 6

end NUMINAMATH_CALUDE_ball_distribution_theorem_l3816_381600


namespace NUMINAMATH_CALUDE_max_lateral_area_triangular_prism_l3816_381697

/-- The maximum lateral area of a triangular prism inscribed in a sphere -/
theorem max_lateral_area_triangular_prism (r : ℝ) (h : r = 2) :
  ∃ (a h : ℝ),
    -- Condition: prism inscribed in sphere
    4 * a^2 + 3 * h^2 = 48 ∧
    -- Condition: lateral area
    (3 : ℝ) * a * h ≤ 12 * Real.sqrt 3 ∧
    -- Condition: maximum value
    ∀ (a' h' : ℝ), 4 * a'^2 + 3 * h'^2 = 48 → (3 : ℝ) * a' * h' ≤ 12 * Real.sqrt 3 :=
by
  sorry


end NUMINAMATH_CALUDE_max_lateral_area_triangular_prism_l3816_381697


namespace NUMINAMATH_CALUDE_computer_upgrade_cost_l3816_381632

/-- Calculates the total amount spent on a computer after upgrading the video card -/
def totalSpent (initialCost salePrice newCardCost : ℕ) : ℕ :=
  initialCost + newCardCost - salePrice

/-- Theorem stating the total amount spent on the computer -/
theorem computer_upgrade_cost :
  ∀ (initialCost salePrice newCardCost : ℕ),
    initialCost = 1200 →
    salePrice = 300 →
    newCardCost = 500 →
    totalSpent initialCost salePrice newCardCost = 1400 :=
by
  sorry

end NUMINAMATH_CALUDE_computer_upgrade_cost_l3816_381632


namespace NUMINAMATH_CALUDE_exists_growth_rate_unique_growth_rate_l3816_381605

/-- Represents the average annual growth rate of Fujian's regional GDP from 2020 to 2022 -/
def average_annual_growth_rate (x : ℝ) : Prop :=
  43903.89 * (1 + x)^2 = 53109.85

/-- The initial GDP of Fujian in 2020 (in billion yuan) -/
def initial_gdp : ℝ := 43903.89

/-- The GDP of Fujian in 2022 (in billion yuan) -/
def final_gdp : ℝ := 53109.85

/-- Theorem stating that there exists an average annual growth rate satisfying the equation -/
theorem exists_growth_rate : ∃ x : ℝ, average_annual_growth_rate x :=
  sorry

/-- Theorem stating that the average annual growth rate is unique -/
theorem unique_growth_rate : ∀ x y : ℝ, average_annual_growth_rate x → average_annual_growth_rate y → x = y :=
  sorry

end NUMINAMATH_CALUDE_exists_growth_rate_unique_growth_rate_l3816_381605


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l3816_381651

theorem system_of_equations_solution (x y m : ℝ) : 
  (3 * x + 5 * y = m + 2) → 
  (2 * x + 3 * y = m) → 
  (x + y = -10) → 
  (m^2 - 2*m + 1 = 81) := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l3816_381651


namespace NUMINAMATH_CALUDE_unique_solution_l3816_381603

/-- Represents the number of children in each class -/
structure ClassSizes where
  judo : ℕ
  agriculture : ℕ
  math : ℕ

/-- Checks if the given class sizes satisfy all conditions -/
def satisfiesConditions (sizes : ClassSizes) : Prop :=
  sizes.judo + sizes.agriculture + sizes.math = 32 ∧
  sizes.judo > 0 ∧ sizes.agriculture > 0 ∧ sizes.math > 0 ∧
  sizes.judo / 2 + sizes.agriculture / 4 + sizes.math / 8 = 6

/-- The theorem stating that the unique solution satisfying all conditions is (4, 4, 24) -/
theorem unique_solution : 
  ∃! sizes : ClassSizes, satisfiesConditions sizes ∧ 
  sizes.judo = 4 ∧ sizes.agriculture = 4 ∧ sizes.math = 24 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l3816_381603


namespace NUMINAMATH_CALUDE_sum_to_k_is_triangular_square_k_values_l3816_381631

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

def sum_to_k (k : ℕ) : ℕ := k * (k + 1) / 2

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

theorem sum_to_k_is_triangular_square (k : ℕ) : Prop :=
  ∃ n : ℕ, sum_to_k k = n^2 ∧ n < 150 ∧ is_perfect_square (triangular_number n)

theorem k_values : {k : ℕ | sum_to_k_is_triangular_square k} = {1, 8, 39, 92, 168} := by
  sorry

end NUMINAMATH_CALUDE_sum_to_k_is_triangular_square_k_values_l3816_381631


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_set_l3816_381608

def number_set : List ℝ := [16, 23, 38, 11.5]

theorem arithmetic_mean_of_set : 
  (number_set.sum / number_set.length : ℝ) = 22.125 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_set_l3816_381608


namespace NUMINAMATH_CALUDE_complex_absolute_value_l3816_381675

theorem complex_absolute_value (ω : ℂ) : ω = 7 + 3*I → Complex.abs (ω^2 + 8*ω + 98) = Real.sqrt 41605 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_l3816_381675


namespace NUMINAMATH_CALUDE_cars_given_to_sister_l3816_381622

/- Define the problem parameters -/
def initial_cars : ℕ := 14
def bought_cars : ℕ := 28
def birthday_cars : ℕ := 12
def cars_to_vinnie : ℕ := 3
def cars_left : ℕ := 43

/- Define the theorem -/
theorem cars_given_to_sister :
  ∃ (cars_to_sister : ℕ),
    initial_cars + bought_cars + birthday_cars
    = cars_to_sister + cars_to_vinnie + cars_left ∧
    cars_to_sister = 8 := by
  sorry

end NUMINAMATH_CALUDE_cars_given_to_sister_l3816_381622


namespace NUMINAMATH_CALUDE_ellipse_sum_theorem_l3816_381679

/-- Represents an ellipse with center (h, k), semi-major axis a, and semi-minor axis b -/
structure Ellipse where
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ

/-- The sum of h, k, a, and b for a specific ellipse -/
def ellipse_sum (e : Ellipse) : ℝ :=
  e.h + e.k + e.a + e.b

/-- Theorem: For an ellipse centered at (3, -5) with vertical semi-major axis 8 and semi-minor axis 4,
    the sum of h, k, a, and b equals 10 -/
theorem ellipse_sum_theorem (e : Ellipse)
    (h_center : e.h = 3 ∧ e.k = -5)
    (h_axes : e.a = 8 ∧ e.b = 4)
    (h_vertical : e.a > e.b) :
    ellipse_sum e = 10 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_sum_theorem_l3816_381679


namespace NUMINAMATH_CALUDE_alice_prob_is_nine_twentyfifths_l3816_381627

/-- Represents the person holding the ball -/
inductive Person : Type
| Alice : Person
| Bob : Person

/-- The probability of tossing the ball to the other person -/
def toss_prob (p : Person) : ℚ :=
  match p with
  | Person.Alice => 3/5
  | Person.Bob => 1/3

/-- The probability of keeping the ball -/
def keep_prob (p : Person) : ℚ :=
  1 - toss_prob p

/-- The probability that Alice has the ball after two turns, given she starts with it -/
def prob_alice_after_two_turns : ℚ :=
  toss_prob Person.Alice * toss_prob Person.Bob +
  keep_prob Person.Alice * keep_prob Person.Alice

theorem alice_prob_is_nine_twentyfifths :
  prob_alice_after_two_turns = 9/25 := by
  sorry

end NUMINAMATH_CALUDE_alice_prob_is_nine_twentyfifths_l3816_381627


namespace NUMINAMATH_CALUDE_largest_positive_root_l3816_381688

theorem largest_positive_root (a₀ a₁ a₂ a₃ : ℝ) 
  (h₀ : |a₀| ≤ 3) (h₁ : |a₁| ≤ 3) (h₂ : |a₂| ≤ 3) (h₃ : |a₃| ≤ 3) :
  ∃ (r : ℝ), r = 3 ∧ 
  (∀ (x : ℝ), x > r → ∀ (b₀ b₁ b₂ b₃ : ℝ), 
    |b₀| ≤ 3 → |b₁| ≤ 3 → |b₂| ≤ 3 → |b₃| ≤ 3 → 
    x^4 + b₃*x^3 + b₂*x^2 + b₁*x + b₀ ≠ 0) ∧
  (∃ (c₀ c₁ c₂ c₃ : ℝ), 
    |c₀| ≤ 3 ∧ |c₁| ≤ 3 ∧ |c₂| ≤ 3 ∧ |c₃| ≤ 3 ∧ 
    r^4 + c₃*r^3 + c₂*r^2 + c₁*r + c₀ = 0) :=
by sorry

end NUMINAMATH_CALUDE_largest_positive_root_l3816_381688


namespace NUMINAMATH_CALUDE_summer_course_duration_l3816_381692

/-- The number of days required for a summer course with the given conditions. -/
def summer_course_days (n k : ℕ) : ℕ :=
  (n.choose 2) / (k.choose 2)

/-- Theorem stating the number of days for the summer course. -/
theorem summer_course_duration :
  summer_course_days 15 3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_summer_course_duration_l3816_381692


namespace NUMINAMATH_CALUDE_ratio_a_to_b_l3816_381657

/-- An arithmetic sequence with specific terms -/
structure ArithmeticSequence where
  a : ℝ
  b : ℝ
  y : ℝ
  first_term : ℝ := 2 * a
  second_term : ℝ := y
  third_term : ℝ := 3 * b
  fourth_term : ℝ := 4 * y
  is_arithmetic : ∃ (d : ℝ), second_term - first_term = d ∧ 
                              third_term - second_term = d ∧ 
                              fourth_term - third_term = d

/-- The ratio of a to b in the arithmetic sequence is -1/5 -/
theorem ratio_a_to_b (seq : ArithmeticSequence) : seq.a / seq.b = -1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_a_to_b_l3816_381657


namespace NUMINAMATH_CALUDE_quadratic_polynomial_remainder_l3816_381640

theorem quadratic_polynomial_remainder (m n : ℚ) : 
  let P : ℚ → ℚ := λ x => x^2 + m*x + n
  (P m = m ∧ P n = n) → 
  ((m = 0 ∧ n = 0) ∨ (m = 1/2 ∧ n = 0) ∨ (m = 1 ∧ n = -1)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_remainder_l3816_381640


namespace NUMINAMATH_CALUDE_coin_sum_impossibility_l3816_381693

def nickel : ℕ := 5
def dime : ℕ := 10
def quarter : ℕ := 25

def is_valid_sum (sum : ℕ) : Prop :=
  ∃ (n d q : ℕ), n + d + q = 6 ∧ n * nickel + d * dime + q * quarter = sum

theorem coin_sum_impossibility :
  is_valid_sum 40 ∧
  is_valid_sum 50 ∧
  is_valid_sum 60 ∧
  is_valid_sum 70 ∧
  ¬ is_valid_sum 30 :=
sorry

end NUMINAMATH_CALUDE_coin_sum_impossibility_l3816_381693


namespace NUMINAMATH_CALUDE_fraction_less_than_one_l3816_381648

theorem fraction_less_than_one (a b : ℝ) (h1 : a > b) (h2 : b > 0) : b / a < 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_less_than_one_l3816_381648


namespace NUMINAMATH_CALUDE_candy_packaging_remainder_l3816_381621

theorem candy_packaging_remainder : 38759863 % 6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_candy_packaging_remainder_l3816_381621


namespace NUMINAMATH_CALUDE_descending_order_exists_l3816_381610

theorem descending_order_exists (x y z : ℤ) : ∃ (a b c : ℤ), 
  ({a, b, c} : Finset ℤ) = {x, y, z} ∧ a ≥ b ∧ b ≥ c := by sorry

end NUMINAMATH_CALUDE_descending_order_exists_l3816_381610


namespace NUMINAMATH_CALUDE_shoe_donation_percentage_l3816_381668

theorem shoe_donation_percentage (initial_shoes : ℕ) (final_shoes : ℕ) (purchased_shoes : ℕ) : 
  initial_shoes = 80 → 
  final_shoes = 62 → 
  purchased_shoes = 6 → 
  (initial_shoes - (final_shoes - purchased_shoes)) / initial_shoes * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_shoe_donation_percentage_l3816_381668


namespace NUMINAMATH_CALUDE_bus_interval_theorem_l3816_381615

/-- The interval between buses on a circular route -/
def interval (num_buses : ℕ) (total_time : ℕ) : ℕ :=
  total_time / num_buses

/-- The theorem stating the relationship between intervals for 2 and 3 buses -/
theorem bus_interval_theorem (total_time : ℕ) :
  interval 2 total_time = 21 → interval 3 total_time = 14 :=
by
  sorry

#check bus_interval_theorem

end NUMINAMATH_CALUDE_bus_interval_theorem_l3816_381615


namespace NUMINAMATH_CALUDE_sweet_potatoes_sold_l3816_381676

theorem sweet_potatoes_sold (total harvested : ℕ) (sold_to_lenon : ℕ) (unsold : ℕ) 
  (h1 : total = 80)
  (h2 : sold_to_lenon = 15)
  (h3 : unsold = 45) :
  total - sold_to_lenon - unsold = 20 :=
by sorry

end NUMINAMATH_CALUDE_sweet_potatoes_sold_l3816_381676


namespace NUMINAMATH_CALUDE_hawks_score_l3816_381602

/-- 
Given the total points scored and the winning margin in a basketball game,
this theorem proves the score of the losing team.
-/
theorem hawks_score (total_points winning_margin : ℕ) 
  (h1 : total_points = 42)
  (h2 : winning_margin = 6) : 
  (total_points - winning_margin) / 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_hawks_score_l3816_381602


namespace NUMINAMATH_CALUDE_race_inequality_l3816_381646

theorem race_inequality (x : ℝ) : 
  (∀ (race_length : ℝ) (initial_speed : ℝ) (ming_speed : ℝ) (li_speed : ℝ) (distance_ahead : ℝ),
    race_length = 10000 ∧ 
    initial_speed = 200 ∧ 
    ming_speed = 250 ∧ 
    li_speed = 300 ∧ 
    distance_ahead = 200 ∧ 
    x > 0 ∧ 
    x < 50 ∧  -- This ensures Xiao Ming doesn't finish before encountering Xiao Li
    (race_length - initial_speed * x - distance_ahead) / ming_speed < 
      (race_length - initial_speed * x) / li_speed) →
  (10000 - 200 * x - 200) / 250 > (10000 - 200 * x) / 300 :=
by sorry

end NUMINAMATH_CALUDE_race_inequality_l3816_381646


namespace NUMINAMATH_CALUDE_perfect_game_score_l3816_381639

/-- Given that a perfect score is 21 points, prove that the total points after 3 perfect games is 63. -/
theorem perfect_game_score (perfect_score : ℕ) (h : perfect_score = 21) :
  3 * perfect_score = 63 := by
  sorry

end NUMINAMATH_CALUDE_perfect_game_score_l3816_381639


namespace NUMINAMATH_CALUDE_intersecting_circles_B_coords_l3816_381659

/-- Two circles with centers on the line y = 1 - x, intersecting at points A and B -/
structure IntersectingCircles where
  O₁ : ℝ × ℝ
  O₂ : ℝ × ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  centers_on_line : ∀ c, c = O₁ ∨ c = O₂ → c.2 = 1 - c.1
  A_coords : A = (-7, 9)

/-- The theorem stating that point B has coordinates (-8, 8) -/
theorem intersecting_circles_B_coords (circles : IntersectingCircles) : 
  circles.B = (-8, 8) := by
  sorry

end NUMINAMATH_CALUDE_intersecting_circles_B_coords_l3816_381659


namespace NUMINAMATH_CALUDE_arccos_sqrt3_over_2_l3816_381667

theorem arccos_sqrt3_over_2 : Real.arccos (Real.sqrt 3 / 2) = π / 6 := by sorry

end NUMINAMATH_CALUDE_arccos_sqrt3_over_2_l3816_381667


namespace NUMINAMATH_CALUDE_door_height_problem_l3816_381684

theorem door_height_problem (pole_length width height diagonal : ℝ) : 
  pole_length > 0 ∧
  width > 0 ∧
  height > 0 ∧
  pole_length = width + 4 ∧
  pole_length = height + 2 ∧
  pole_length = diagonal ∧
  diagonal^2 = width^2 + height^2
  → height = 8 := by
  sorry

end NUMINAMATH_CALUDE_door_height_problem_l3816_381684


namespace NUMINAMATH_CALUDE_boat_distance_difference_l3816_381695

/-- The difference in distance traveled between two boats, one traveling downstream
    and one upstream, is 30 km. -/
theorem boat_distance_difference
  (a : ℝ)  -- Speed of both boats in still water (km/h)
  (h : a > 5)  -- Assumption that the boat speed is greater than the water flow speed
  : (3 * (a + 5)) - (3 * (a - 5)) = 30 :=
by sorry

end NUMINAMATH_CALUDE_boat_distance_difference_l3816_381695


namespace NUMINAMATH_CALUDE_henry_bought_two_fireworks_l3816_381678

/-- The number of fireworks Henry bought -/
def henrys_fireworks (total : ℕ) (last_year : ℕ) (friends : ℕ) : ℕ :=
  total - last_year - friends

/-- Proof that Henry bought 2 fireworks -/
theorem henry_bought_two_fireworks :
  henrys_fireworks 11 6 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_henry_bought_two_fireworks_l3816_381678


namespace NUMINAMATH_CALUDE_brothers_multiple_l3816_381620

/-- Given that Aaron has 4 brothers and Bennett has 6 brothers, 
    prove that the multiple relating their number of brothers is 2. -/
theorem brothers_multiple (aaron_brothers : ℕ) (bennett_brothers : ℕ) : 
  aaron_brothers = 4 → bennett_brothers = 6 → ∃ x : ℕ, x * aaron_brothers - 2 = bennett_brothers ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_brothers_multiple_l3816_381620


namespace NUMINAMATH_CALUDE_function_growth_l3816_381612

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- State that f' is the derivative of f
variable (hf' : ∀ x, HasDerivAt f (f' x) x)

-- State the condition that f'(x) > f(x) for all x
variable (h : ∀ x, f' x > f x)

-- Theorem statement
theorem function_growth (f f' : ℝ → ℝ) (hf' : ∀ x, HasDerivAt f (f' x) x) (h : ∀ x, f' x > f x) :
  f 2012 > Real.exp 2012 * f 0 := by
  sorry

end NUMINAMATH_CALUDE_function_growth_l3816_381612


namespace NUMINAMATH_CALUDE_x_value_proof_l3816_381625

theorem x_value_proof : ∀ x : ℝ, x + Real.sqrt 25 = Real.sqrt 36 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l3816_381625


namespace NUMINAMATH_CALUDE_distance_to_charlie_l3816_381619

/-- The vertical distance Annie and Barbara walk together to reach Charlie -/
theorem distance_to_charlie 
  (annie_x annie_y barbara_x barbara_y charlie_x charlie_y : ℚ) : 
  annie_x = 6 → 
  annie_y = -20 → 
  barbara_x = 1 → 
  barbara_y = 14 → 
  charlie_x = 7/2 → 
  charlie_y = 2 → 
  charlie_y - (annie_y + barbara_y) / 2 = 5 := by
sorry

end NUMINAMATH_CALUDE_distance_to_charlie_l3816_381619


namespace NUMINAMATH_CALUDE_solution_characterization_l3816_381696

def system_equations (n : ℕ) (x : ℕ → ℝ) : Prop :=
  (∀ i ∈ Finset.range n, 1 - x i * x ((i + 1) % n) = 0)

theorem solution_characterization (n : ℕ) (x : ℕ → ℝ) (hn : n > 0) :
  system_equations n x →
  (n % 2 = 1 ∧ (∀ i ∈ Finset.range n, x i = 1 ∨ x i = -1)) ∨
  (n % 2 = 0 ∧ ∃ a : ℝ, a ≠ 0 ∧
    x 0 = a ∧ x 1 = 1 / a ∧
    ∀ i ∈ Finset.range (n - 2), x (i + 2) = x i) :=
by sorry

end NUMINAMATH_CALUDE_solution_characterization_l3816_381696


namespace NUMINAMATH_CALUDE_vector_simplification_l3816_381686

variable {V : Type*} [AddCommGroup V]

theorem vector_simplification (P M N : V) : 
  (P - M) - (P - N) + (M - N) = (0 : V) := by sorry

end NUMINAMATH_CALUDE_vector_simplification_l3816_381686


namespace NUMINAMATH_CALUDE_triangle_properties_l3816_381649

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : Real.cos t.A * Real.cos t.C + Real.sin t.A * Real.sin t.C + Real.cos t.B = 3/2)
  (h2 : t.b^2 = t.a * t.c)  -- Geometric progression condition
  (h3 : t.a / Real.tan t.A + t.c / Real.tan t.C = 2 * t.b / Real.tan t.B) :
  t.B = π/3 ∧ t.A = π/3 ∧ t.C = π/3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3816_381649
