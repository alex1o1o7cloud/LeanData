import Mathlib

namespace NUMINAMATH_CALUDE_correct_negation_of_existential_statement_l958_95860

theorem correct_negation_of_existential_statement :
  (¬ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 > 0) :=
by sorry

end NUMINAMATH_CALUDE_correct_negation_of_existential_statement_l958_95860


namespace NUMINAMATH_CALUDE_train_length_l958_95896

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 180 → time = 7 → speed * time * (5 / 18) = 350 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l958_95896


namespace NUMINAMATH_CALUDE_lcm_gcd_problem_l958_95897

theorem lcm_gcd_problem (a b : ℕ+) : 
  Nat.lcm a b = 2310 →
  Nat.gcd a b = 55 →
  a = 210 →
  b = 605 := by
sorry

end NUMINAMATH_CALUDE_lcm_gcd_problem_l958_95897


namespace NUMINAMATH_CALUDE_arithmetic_sequence_10th_term_l958_95867

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_10th_term
  (a : ℕ → ℚ)
  (h_arith : arithmetic_sequence a)
  (h_a1 : a 1 = 2)
  (h_a4 : a 4 = 6) :
  a 10 = 14 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_10th_term_l958_95867


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l958_95826

/-- A right triangle with perimeter 40 and area 30 has a hypotenuse of length 74/4 -/
theorem right_triangle_hypotenuse : ∀ a b c : ℝ,
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →  -- right triangle condition
  a + b + c = 40 →   -- perimeter condition
  a * b / 2 = 30 →   -- area condition
  c = 74 / 4 := by
    sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l958_95826


namespace NUMINAMATH_CALUDE_reduction_after_four_trials_l958_95809

/-- The reduction factor for the 0.618 method -/
def golden_ratio_inverse : ℝ := 0.618

/-- The number of trials -/
def num_trials : ℕ := 4

/-- The reduction factor after n trials using the 0.618 method -/
def reduction_factor (n : ℕ) : ℝ := golden_ratio_inverse ^ (n - 1)

/-- Theorem stating the reduction factor after 4 trials -/
theorem reduction_after_four_trials :
  reduction_factor num_trials = golden_ratio_inverse ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_reduction_after_four_trials_l958_95809


namespace NUMINAMATH_CALUDE_ferry_speed_difference_l958_95848

theorem ferry_speed_difference (speed_p time_p distance_q_factor time_difference : ℝ) 
  (h1 : speed_p = 6)
  (h2 : time_p = 3)
  (h3 : distance_q_factor = 3)
  (h4 : time_difference = 3) : 
  let distance_p := speed_p * time_p
  let distance_q := distance_q_factor * distance_p
  let time_q := time_p + time_difference
  let speed_q := distance_q / time_q
  speed_q - speed_p = 3 := by sorry

end NUMINAMATH_CALUDE_ferry_speed_difference_l958_95848


namespace NUMINAMATH_CALUDE_polynomial_simplification_l958_95800

theorem polynomial_simplification : 
  2021^4 - 4 * 2023^4 + 6 * 2025^4 - 4 * 2027^4 + 2029^4 = 384 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l958_95800


namespace NUMINAMATH_CALUDE_tommy_books_l958_95887

/-- The number of books Tommy wants to buy -/
def num_books (book_cost savings_needed current_money : ℕ) : ℕ :=
  (savings_needed + current_money) / book_cost

/-- Proof that Tommy wants to buy 8 books -/
theorem tommy_books : num_books 5 27 13 = 8 := by
  sorry

end NUMINAMATH_CALUDE_tommy_books_l958_95887


namespace NUMINAMATH_CALUDE_complement_intersection_equals_set_l958_95802

-- Define the universal set U
def U : Set Int := {-1, 1, 2, 3}

-- Define set A
def A : Set Int := {-1, 2}

-- Define set B
def B : Set Int := {x : Int | x^2 - 2*x - 3 = 0}

-- Theorem statement
theorem complement_intersection_equals_set : 
  (U \ (A ∩ B)) = {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_set_l958_95802


namespace NUMINAMATH_CALUDE_blue_balloons_l958_95815

theorem blue_balloons (total red green purple : ℕ) (h1 : total = 135) (h2 : red = 45) (h3 : green = 27) (h4 : purple = 32) :
  total - (red + green + purple) = 31 := by
  sorry

end NUMINAMATH_CALUDE_blue_balloons_l958_95815


namespace NUMINAMATH_CALUDE_race_result_l958_95895

/-- Represents the difference in meters between two runners at the end of a 1000-meter race. -/
def finish_difference (runner1 runner2 : ℕ) : ℝ := sorry

theorem race_result (A B C : ℕ) :
  finish_difference A C = 200 →
  finish_difference B C = 120.87912087912093 →
  finish_difference A B = 79.12087912087907 :=
by sorry

end NUMINAMATH_CALUDE_race_result_l958_95895


namespace NUMINAMATH_CALUDE_negation_of_at_least_one_even_l958_95828

theorem negation_of_at_least_one_even (a b c : ℕ) :
  (¬ (Even a ∨ Even b ∨ Even c)) ↔ (Odd a ∧ Odd b ∧ Odd c) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_at_least_one_even_l958_95828


namespace NUMINAMATH_CALUDE_x_minus_y_positive_l958_95888

theorem x_minus_y_positive (x y a : ℝ) 
  (h1 : x + y > 0) 
  (h2 : a < 0) 
  (h3 : a * y > 0) : 
  x - y > 0 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_positive_l958_95888


namespace NUMINAMATH_CALUDE_bookstore_revenue_theorem_l958_95849

structure BookStore where
  total_books : ℕ
  novels : ℕ
  biographies : ℕ
  science_books : ℕ
  novel_price : ℚ
  biography_price : ℚ
  science_book_price : ℚ
  novel_discount : ℚ
  biography_discount : ℚ
  science_book_discount : ℚ
  remaining_novels : ℕ
  remaining_biographies : ℕ
  remaining_science_books : ℕ
  sales_tax : ℚ

def calculate_total_revenue (store : BookStore) : ℚ :=
  let sold_novels := store.novels - store.remaining_novels
  let sold_biographies := store.biographies - store.remaining_biographies
  let sold_science_books := store.science_books - store.remaining_science_books
  let novel_revenue := (sold_novels : ℚ) * store.novel_price * (1 - store.novel_discount)
  let biography_revenue := (sold_biographies : ℚ) * store.biography_price * (1 - store.biography_discount)
  let science_book_revenue := (sold_science_books : ℚ) * store.science_book_price * (1 - store.science_book_discount)
  let total_discounted_revenue := novel_revenue + biography_revenue + science_book_revenue
  total_discounted_revenue * (1 + store.sales_tax)

theorem bookstore_revenue_theorem (store : BookStore) 
  (h1 : store.total_books = 500)
  (h2 : store.novels + store.biographies + store.science_books = store.total_books)
  (h3 : store.novels - store.remaining_novels = (3 * store.novels) / 5)
  (h4 : store.biographies - store.remaining_biographies = (2 * store.biographies) / 3)
  (h5 : store.science_books - store.remaining_science_books = (7 * store.science_books) / 10)
  (h6 : store.novel_price = 8)
  (h7 : store.biography_price = 12)
  (h8 : store.science_book_price = 10)
  (h9 : store.novel_discount = 1/4)
  (h10 : store.biography_discount = 3/10)
  (h11 : store.science_book_discount = 1/5)
  (h12 : store.remaining_novels = 60)
  (h13 : store.remaining_biographies = 65)
  (h14 : store.remaining_science_books = 50)
  (h15 : store.sales_tax = 1/20)
  : calculate_total_revenue store = 2696.4 := by
  sorry

end NUMINAMATH_CALUDE_bookstore_revenue_theorem_l958_95849


namespace NUMINAMATH_CALUDE_min_value_of_expression_l958_95812

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a + 2) * (1 / b + 2) ≥ 16 ∧
  ((1 / a + 2) * (1 / b + 2) = 16 ↔ a = 1/2 ∧ b = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l958_95812


namespace NUMINAMATH_CALUDE_custom_mul_ab_equals_nine_l958_95873

/-- Custom multiplication operation -/
def custom_mul (a b : ℝ) (x y : ℝ) : ℝ := a * x + b * y - 1

/-- Theorem stating that under given conditions, a*b = 9 -/
theorem custom_mul_ab_equals_nine
  (a b : ℝ)
  (h1 : custom_mul a b 1 2 = 4)
  (h2 : custom_mul a b (-2) 3 = 10) :
  custom_mul a b a b = 9 :=
sorry

end NUMINAMATH_CALUDE_custom_mul_ab_equals_nine_l958_95873


namespace NUMINAMATH_CALUDE_max_value_of_expression_l958_95862

theorem max_value_of_expression (a b : ℝ) 
  (h : 17 * (a^2 + b^2) - 30 * a * b - 16 = 0) : 
  ∃ (x : ℝ), x = Real.sqrt (16 * a^2 + 4 * b^2 - 16 * a * b - 12 * a + 6 * b + 9) ∧ 
  x ≤ 7 ∧ 
  ∃ (a₀ b₀ : ℝ), 17 * (a₀^2 + b₀^2) - 30 * a₀ * b₀ - 16 = 0 ∧ 
    Real.sqrt (16 * a₀^2 + 4 * b₀^2 - 16 * a₀ * b₀ - 12 * a₀ + 6 * b₀ + 9) = 7 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l958_95862


namespace NUMINAMATH_CALUDE_max_square_plots_for_given_field_l958_95816

/-- Represents the dimensions of a rectangular field -/
structure FieldDimensions where
  width : ℕ
  length : ℕ

/-- Represents the available fencing and field dimensions -/
structure FencingProblem where
  field : FieldDimensions
  internalFencing : ℕ

/-- Calculates the maximum number of square plots given a fencing problem -/
def maxSquarePlots (problem : FencingProblem) : ℕ :=
  sorry

/-- The main theorem stating the solution to the specific problem -/
theorem max_square_plots_for_given_field :
  let problem : FencingProblem := {
    field := { width := 40, length := 60 },
    internalFencing := 2400
  }
  maxSquarePlots problem = 6 := by
  sorry

end NUMINAMATH_CALUDE_max_square_plots_for_given_field_l958_95816


namespace NUMINAMATH_CALUDE_first_digit_base_nine_of_2121122_base_three_l958_95854

def base_three_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

def first_digit_base_nine (n : Nat) : Nat :=
  Nat.log 9 n

theorem first_digit_base_nine_of_2121122_base_three :
  let y : Nat := base_three_to_decimal [2, 2, 1, 1, 2, 1, 2]
  first_digit_base_nine y = 2 := by
  sorry

end NUMINAMATH_CALUDE_first_digit_base_nine_of_2121122_base_three_l958_95854


namespace NUMINAMATH_CALUDE_square_is_self_product_l958_95894

theorem square_is_self_product (b : ℚ) : b^2 = b * b := by
  sorry

end NUMINAMATH_CALUDE_square_is_self_product_l958_95894


namespace NUMINAMATH_CALUDE_isosceles_triangle_l958_95823

theorem isosceles_triangle (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧          -- Sum of angles in a triangle
  c = 2 * a * Real.cos B → -- Given condition
  A = B                    -- Conclusion: triangle is isosceles
  := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_l958_95823


namespace NUMINAMATH_CALUDE_smallest_num_with_digit_sum_2017_properties_first_digit_times_num_digits_l958_95835

/-- The smallest natural number with digit sum 2017 -/
def smallest_num_with_digit_sum_2017 : ℕ :=
  1 * 10^224 + (10^224 - 1)

/-- The digit sum of a natural number -/
def digit_sum (n : ℕ) : ℕ :=
  sorry

/-- The number of digits in a natural number -/
def num_digits (n : ℕ) : ℕ :=
  sorry

theorem smallest_num_with_digit_sum_2017_properties :
  digit_sum smallest_num_with_digit_sum_2017 = 2017 ∧
  num_digits smallest_num_with_digit_sum_2017 = 225 ∧
  smallest_num_with_digit_sum_2017 < 10^225 ∧
  ∀ m : ℕ, m < smallest_num_with_digit_sum_2017 → digit_sum m ≠ 2017 :=
by sorry

theorem first_digit_times_num_digits :
  (smallest_num_with_digit_sum_2017 / 10^224) * num_digits smallest_num_with_digit_sum_2017 = 225 :=
by sorry

end NUMINAMATH_CALUDE_smallest_num_with_digit_sum_2017_properties_first_digit_times_num_digits_l958_95835


namespace NUMINAMATH_CALUDE_pet_store_puppies_l958_95857

/-- The number of puppies sold --/
def puppies_sold : ℕ := 39

/-- The number of cages used --/
def cages_used : ℕ := 3

/-- The number of puppies per cage --/
def puppies_per_cage : ℕ := 2

/-- The initial number of puppies in the pet store --/
def initial_puppies : ℕ := puppies_sold + cages_used * puppies_per_cage

theorem pet_store_puppies : initial_puppies = 45 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_puppies_l958_95857


namespace NUMINAMATH_CALUDE_inequality_properties_l958_95851

theorem inequality_properties (a b : ℝ) (h : 1/a < 1/b ∧ 1/b < 0) :
  a^2 < b^2 ∧ a*b < b^2 ∧ a/b + b/a > 2 ∧ |a| + |b| = |a + b| := by
  sorry

end NUMINAMATH_CALUDE_inequality_properties_l958_95851


namespace NUMINAMATH_CALUDE_configuration_permutations_l958_95852

/-- The number of distinct arrangements of the letters in "CONFIGURATION" -/
def configuration_arrangements : ℕ := 389188800

/-- The total number of letters in "CONFIGURATION" -/
def total_letters : ℕ := 13

/-- The number of times each of O, I, N, and U appears in "CONFIGURATION" -/
def repeated_letter_count : ℕ := 2

/-- The number of letters that repeat in "CONFIGURATION" -/
def repeating_letters : ℕ := 4

theorem configuration_permutations :
  configuration_arrangements = (Nat.factorial total_letters) / (Nat.factorial repeated_letter_count ^ repeating_letters) :=
sorry

end NUMINAMATH_CALUDE_configuration_permutations_l958_95852


namespace NUMINAMATH_CALUDE_percentage_boys_playing_soccer_l958_95853

theorem percentage_boys_playing_soccer 
  (total_students : ℕ) 
  (boys : ℕ) 
  (playing_soccer : ℕ) 
  (girls_not_playing : ℕ) 
  (h1 : total_students = 420)
  (h2 : boys = 296)
  (h3 : playing_soccer = 250)
  (h4 : girls_not_playing = 89)
  : (boys - (total_students - boys - girls_not_playing)) / playing_soccer * 100 = 86 := by
  sorry

end NUMINAMATH_CALUDE_percentage_boys_playing_soccer_l958_95853


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_squared_l958_95838

theorem sum_of_reciprocals_squared (a b c d : ℝ) : 
  a = 2 * Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 35 →
  b = -2 * Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 35 →
  c = 2 * Real.sqrt 5 - Real.sqrt 7 + Real.sqrt 35 →
  d = -2 * Real.sqrt 5 - Real.sqrt 7 + Real.sqrt 35 →
  (1/a + 1/b + 1/c + 1/d)^2 = 560 / 155432121 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_squared_l958_95838


namespace NUMINAMATH_CALUDE_cos_angle_F₁PF₂_l958_95842

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 2 = 1

-- Define the origin
def O : ℝ × ℝ := (0, 0)

-- Define the foci (we don't know their exact coordinates, so we leave them abstract)
variables (F₁ F₂ : ℝ × ℝ)

-- Define point P on the ellipse
variable (P : ℝ × ℝ)

-- State that P is on the ellipse
axiom P_on_ellipse : ellipse P.1 P.2

-- Define the distance between O and P
axiom OP_distance : Real.sqrt ((P.1 - O.1)^2 + (P.2 - O.2)^2) = Real.sqrt 3

-- Theorem to prove
theorem cos_angle_F₁PF₂ : 
  ∃ (F₁ F₂ : ℝ × ℝ), 
    (F₁ ≠ F₂) ∧ 
    (∀ Q : ℝ × ℝ, ellipse Q.1 Q.2 → 
      Real.sqrt ((Q.1 - F₁.1)^2 + (Q.2 - F₁.2)^2) +
      Real.sqrt ((Q.1 - F₂.1)^2 + (Q.2 - F₂.2)^2) = 
      Real.sqrt ((F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2)) →
    ((P.1 - F₁.1) * (P.1 - F₂.1) + (P.2 - F₁.2) * (P.2 - F₂.2)) / 
    (Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) * 
     Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2)) = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_cos_angle_F₁PF₂_l958_95842


namespace NUMINAMATH_CALUDE_min_cut_length_40x30_paper_10x5_rect_l958_95855

/-- Represents a rectangular piece of paper -/
structure Paper where
  width : ℕ
  height : ℕ

/-- Represents a rectangle to be cut out -/
structure CutRectangle where
  width : ℕ
  height : ℕ

/-- Calculates the minimum cut length required to extract a rectangle from a paper -/
def minCutLength (paper : Paper) (rect : CutRectangle) : ℕ :=
  sorry

/-- Theorem stating the minimum cut length for the given problem -/
theorem min_cut_length_40x30_paper_10x5_rect :
  let paper := Paper.mk 40 30
  let rect := CutRectangle.mk 10 5
  minCutLength paper rect = 40 := by sorry

end NUMINAMATH_CALUDE_min_cut_length_40x30_paper_10x5_rect_l958_95855


namespace NUMINAMATH_CALUDE_total_spent_on_presents_l958_95811

def leonard_wallets : ℕ := 3
def leonard_wallet_price : ℚ := 35.50
def leonard_sneakers : ℕ := 2
def leonard_sneaker_price : ℚ := 120.75
def leonard_belt_price : ℚ := 44.25
def leonard_discount_rate : ℚ := 0.10

def michael_backpack_price : ℚ := 89.50
def michael_jeans : ℕ := 3
def michael_jeans_price : ℚ := 54.50
def michael_tie_price : ℚ := 24.75
def michael_discount_rate : ℚ := 0.15

def emily_shirts : ℕ := 2
def emily_shirt_price : ℚ := 69.25
def emily_books : ℕ := 4
def emily_book_price : ℚ := 14.80
def emily_tax_rate : ℚ := 0.08

theorem total_spent_on_presents (leonard_total michael_total emily_total : ℚ) :
  leonard_total = (1 - leonard_discount_rate) * (leonard_wallets * leonard_wallet_price + leonard_sneakers * leonard_sneaker_price + leonard_belt_price) →
  michael_total = (1 - michael_discount_rate) * (michael_backpack_price + michael_jeans * michael_jeans_price + michael_tie_price) →
  emily_total = (1 + emily_tax_rate) * (emily_shirts * emily_shirt_price + emily_books * emily_book_price) →
  leonard_total + michael_total + emily_total = 802.64 :=
by sorry

end NUMINAMATH_CALUDE_total_spent_on_presents_l958_95811


namespace NUMINAMATH_CALUDE_three_digit_number_property_l958_95868

theorem three_digit_number_property : 
  ∃ (N : ℕ), 
    (100 ≤ N ∧ N < 1000) ∧ 
    (N % 11 = 0) ∧ 
    (N / 11 = (N / 100)^2 + ((N / 10) % 10)^2 + (N % 10)^2) ∧
    (N = 550 ∨ N = 803) ∧
    (∀ (M : ℕ), 
      (100 ≤ M ∧ M < 1000) ∧ 
      (M % 11 = 0) ∧ 
      (M / 11 = (M / 100)^2 + ((M / 10) % 10)^2 + (M % 10)^2) →
      (M = 550 ∨ M = 803)) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_number_property_l958_95868


namespace NUMINAMATH_CALUDE_jeff_ninja_stars_l958_95875

/-- The number of ninja throwing stars each person has -/
structure NinjaStars where
  eric : ℕ
  chad : ℕ
  jeff : ℕ

/-- The conditions of the problem -/
def ninja_star_problem (stars : NinjaStars) : Prop :=
  stars.eric = 4 ∧
  stars.chad = 2 * stars.eric ∧
  stars.eric + stars.chad + stars.jeff = 16 ∧
  stars.chad = (2 * stars.eric) - 2

theorem jeff_ninja_stars :
  ∃ (stars : NinjaStars), ninja_star_problem stars ∧ stars.jeff = 6 := by
  sorry

end NUMINAMATH_CALUDE_jeff_ninja_stars_l958_95875


namespace NUMINAMATH_CALUDE_parabola_vertex_vertex_of_specific_parabola_l958_95893

/-- The vertex of a parabola y = ax^2 + bx + c is (h, k) where h = -b/(2a) and k = f(h) -/
theorem parabola_vertex (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  let vertex_x : ℝ := -b / (2 * a)
  let vertex_y : ℝ := f vertex_x
  (∀ x, f x ≥ vertex_y) ∨ (∀ x, f x ≤ vertex_y) :=
sorry

theorem vertex_of_specific_parabola :
  let f : ℝ → ℝ := λ x => 3 * x^2 - 6 * x + 5
  let vertex : ℝ × ℝ := (1, 2)
  (∀ x, f x ≥ 2) ∧ f 1 = 2 :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_vertex_of_specific_parabola_l958_95893


namespace NUMINAMATH_CALUDE_solution_set_f_exp_pos_l958_95836

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the solution set of f(x) < 0
def solution_set_f_neg (x : ℝ) : Prop := x < -1 ∨ x > 1/3

-- Theorem statement
theorem solution_set_f_exp_pos :
  (∀ x, f x < 0 ↔ solution_set_f_neg x) →
  (∀ x, f (Real.exp x) > 0 ↔ x < -Real.log 3) :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_exp_pos_l958_95836


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l958_95864

def OA : ℝ × ℝ := (2, 2)
def OB : ℝ × ℝ := (5, 3)

def AB : ℝ × ℝ := (OB.1 - OA.1, OB.2 - OA.2)

theorem vector_difference_magnitude : 
  Real.sqrt ((2 * OA.1 - OB.1)^2 + (2 * OA.2 - OB.2)^2) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_vector_difference_magnitude_l958_95864


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l958_95824

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a₁ * r^(n - 1)

theorem geometric_sequence_sum (a₁ r : ℝ) (h₁ : a₁ = 1) (h₂ : r = -2) :
  let a := geometric_sequence a₁ r
  (a 1) + |a 2| + (a 3) + |a 4| = 15 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l958_95824


namespace NUMINAMATH_CALUDE_tetrahedron_probabilities_l958_95886

/-- A regular tetrahedron with numbers 1, 2, 3, 4 on its faces -/
structure Tetrahedron :=
  (faces : Fin 4 → Fin 4)
  (bijective : Function.Bijective faces)

/-- The probability space of throwing the tetrahedron twice -/
def TetrahedronThrows := Tetrahedron × Tetrahedron

/-- Event A: 2 or 3 facing down on first throw -/
def event_A (t : TetrahedronThrows) : Prop :=
  t.1.faces 0 = 2 ∨ t.1.faces 0 = 3

/-- Event B: sum of numbers facing down is odd -/
def event_B (t : TetrahedronThrows) : Prop :=
  (t.1.faces 0 + t.2.faces 0) % 2 = 1

/-- Event C: sum of numbers facing up is not less than 15 -/
def event_C (t : TetrahedronThrows) : Prop :=
  (10 - t.1.faces 0 - t.2.faces 0) ≥ 15

/-- The probability measure on TetrahedronThrows -/
noncomputable def P : Set TetrahedronThrows → ℝ := sorry

theorem tetrahedron_probabilities :
  (P {t : TetrahedronThrows | event_A t} * P {t : TetrahedronThrows | event_B t} =
   P {t : TetrahedronThrows | event_A t ∧ event_B t}) ∧
  (P {t : TetrahedronThrows | event_A t ∨ event_B t} = 3/4) ∧
  (P {t : TetrahedronThrows | event_C t} = 5/8) :=
sorry

end NUMINAMATH_CALUDE_tetrahedron_probabilities_l958_95886


namespace NUMINAMATH_CALUDE_square_area_ratio_l958_95850

/-- The ratio of the area of a square with side length 2y to the area of a square with side length 8y is 1/16 -/
theorem square_area_ratio (y : ℝ) (y_pos : y > 0) : 
  (2 * y)^2 / (8 * y)^2 = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l958_95850


namespace NUMINAMATH_CALUDE_economics_class_question_l958_95807

theorem economics_class_question (total_students : ℕ) 
  (q2_correct : ℕ) (not_taken : ℕ) (both_correct : ℕ) :
  total_students = 40 →
  q2_correct = 29 →
  not_taken = 10 →
  both_correct = 29 →
  ∃ (q1_correct : ℕ), q1_correct ≥ 29 :=
by sorry

end NUMINAMATH_CALUDE_economics_class_question_l958_95807


namespace NUMINAMATH_CALUDE_danivan_drugstore_inventory_l958_95876

def calculate_final_inventory (starting_inventory : ℕ) (daily_sales : List ℕ) (deliveries : List ℕ) : ℕ :=
  let daily_changes := List.zipWith (λ s d => d - s) daily_sales deliveries
  starting_inventory + daily_changes.sum

theorem danivan_drugstore_inventory : 
  let starting_inventory : ℕ := 4500
  let daily_sales : List ℕ := [1277, 2124, 679, 854, 535, 1073, 728]
  let deliveries : List ℕ := [2250, 0, 980, 750, 0, 1345, 0]
  calculate_final_inventory starting_inventory daily_sales deliveries = 2555 := by
  sorry

#eval calculate_final_inventory 4500 [1277, 2124, 679, 854, 535, 1073, 728] [2250, 0, 980, 750, 0, 1345, 0]

end NUMINAMATH_CALUDE_danivan_drugstore_inventory_l958_95876


namespace NUMINAMATH_CALUDE_fruit_shop_apples_l958_95869

/-- Given a ratio of fruits and the number of mangoes, calculate the number of apples -/
theorem fruit_shop_apples (ratio_mangoes ratio_oranges ratio_apples : ℕ) 
  (num_mangoes : ℕ) (h1 : ratio_mangoes = 10) (h2 : ratio_oranges = 2) 
  (h3 : ratio_apples = 3) (h4 : num_mangoes = 120) : 
  (num_mangoes / ratio_mangoes) * ratio_apples = 36 := by
  sorry

end NUMINAMATH_CALUDE_fruit_shop_apples_l958_95869


namespace NUMINAMATH_CALUDE_geometry_algebra_properties_l958_95898

-- Define supplementary angles
def supplementary (α β : Real) : Prop := α + β = 180

-- Define congruent angles
def congruent (α β : Real) : Prop := α = β

-- Define vertical angles
def vertical (α β : Real) : Prop := α = β

-- Define perpendicular lines
def perpendicular (l₁ l₂ : Line) : Prop := sorry

-- Define parallel lines
def parallel (l₁ l₂ : Line) : Prop := sorry

theorem geometry_algebra_properties :
  (∃ α β : Real, supplementary α β ∧ ¬congruent α β) ∧
  (∀ α β : Real, vertical α β → α = β) ∧
  ((-1 : Real)^(1/3) = -1) ∧
  (∀ l₁ l₂ l₃ : Line, perpendicular l₁ l₃ → perpendicular l₂ l₃ → parallel l₁ l₂) :=
sorry

end NUMINAMATH_CALUDE_geometry_algebra_properties_l958_95898


namespace NUMINAMATH_CALUDE_red_balloons_count_l958_95830

/-- Proves that the total number of red balloons after destruction is 40 -/
theorem red_balloons_count (fred_balloons sam_balloons dan_destroyed : ℝ) 
  (h1 : fred_balloons = 10)
  (h2 : sam_balloons = 46)
  (h3 : dan_destroyed = 16) :
  fred_balloons + sam_balloons - dan_destroyed = 40 := by
  sorry

#check red_balloons_count

end NUMINAMATH_CALUDE_red_balloons_count_l958_95830


namespace NUMINAMATH_CALUDE_universal_rook_program_exists_l958_95833

/-- Represents a position on the 8x8 chessboard --/
structure Position :=
  (row : Fin 8)
  (col : Fin 8)

/-- Represents a command for moving the rook --/
inductive Command
  | RIGHT
  | LEFT
  | UP
  | DOWN

/-- Represents a maze configuration on the 8x8 chessboard --/
def Maze := Set (Position × Position)

/-- Represents a program as a finite sequence of commands --/
def Program := List Command

/-- Function to determine if a square is accessible from a given position in a maze --/
def isAccessible (maze : Maze) (start finish : Position) : Prop := sorry

/-- Function to determine if a program visits all accessible squares from a given start position --/
def visitsAllAccessible (maze : Maze) (start : Position) (program : Program) : Prop := sorry

/-- The main theorem stating that there exists a program that works for all mazes and start positions --/
theorem universal_rook_program_exists :
  ∃ (program : Program),
    ∀ (maze : Maze) (start : Position),
      visitsAllAccessible maze start program := by sorry

end NUMINAMATH_CALUDE_universal_rook_program_exists_l958_95833


namespace NUMINAMATH_CALUDE_probability_theorem_l958_95877

/-- The probability of selecting three distinct integers between 1 and 100 (inclusive) 
    such that their product is odd and a multiple of 5 -/
def probability_odd_multiple_of_five : ℚ := 3 / 125

/-- The set of integers from 1 to 100, inclusive -/
def integer_set : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 100}

/-- A function that determines if a natural number is odd -/
def is_odd (n : ℕ) : Prop := n % 2 = 1

/-- A function that determines if a natural number is a multiple of 5 -/
def is_multiple_of_five (n : ℕ) : Prop := n % 5 = 0

/-- The main theorem stating that the probability of selecting three distinct integers 
    between 1 and 100 (inclusive) such that their product is odd and a multiple of 5 
    is equal to 3/125 -/
theorem probability_theorem : 
  ∀ (a b c : ℕ), a ∈ integer_set → b ∈ integer_set → c ∈ integer_set → 
  a ≠ b → b ≠ c → a ≠ c →
  (is_odd a ∧ is_odd b ∧ is_odd c ∧ (is_multiple_of_five a ∨ is_multiple_of_five b ∨ is_multiple_of_five c)) →
  probability_odd_multiple_of_five = 3 / 125 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l958_95877


namespace NUMINAMATH_CALUDE_circle_radius_is_sqrt_21_25_l958_95821

-- Define the circle Ω
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define points P and Q
def P : ℝ × ℝ := (9, 17)
def Q : ℝ × ℝ := (18, 15)

-- Define the line y = 2
def line_y_2 (x : ℝ) : ℝ := 2

-- Theorem statement
theorem circle_radius_is_sqrt_21_25 (Ω : Circle) :
  P ∈ {p : ℝ × ℝ | (p.1 - Ω.center.1)^2 + (p.2 - Ω.center.2)^2 = Ω.radius^2} →
  Q ∈ {p : ℝ × ℝ | (p.1 - Ω.center.1)^2 + (p.2 - Ω.center.2)^2 = Ω.radius^2} →
  (∃ x : ℝ, (x, line_y_2 x) ∈ {p : ℝ × ℝ | ∃ t : ℝ, p = (P.1 + t * (P.2 - Ω.center.2), P.2 - t * (P.1 - Ω.center.1))} ∩
                               {p : ℝ × ℝ | ∃ t : ℝ, p = (Q.1 + t * (Q.2 - Ω.center.2), Q.2 - t * (Q.1 - Ω.center.1))}) →
  Ω.radius = Real.sqrt 21.25 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_is_sqrt_21_25_l958_95821


namespace NUMINAMATH_CALUDE_smallest_x_for_g_equality_l958_95863

-- Define the function g
def g : ℝ → ℝ := sorry

-- State the theorem
theorem smallest_x_for_g_equality (g : ℝ → ℝ) : 
  (∀ (x : ℝ), x > 0 → g (4 * x) = 4 * g x) →
  (∀ (x : ℝ), 2 ≤ x ∧ x ≤ 4 → g x = 1 - |x - 3|) →
  (∀ (x : ℝ), x ≥ 0 ∧ g x = g 2048 → x ≥ 2) ∧
  g 2 = g 2048 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_for_g_equality_l958_95863


namespace NUMINAMATH_CALUDE_purely_imaginary_value_l958_95841

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem purely_imaginary_value (a : ℝ) :
  let z : ℂ := Complex.mk (a^2 + a - 2) (a^2 - 1)
  is_purely_imaginary z → a = -2 :=
by sorry

end NUMINAMATH_CALUDE_purely_imaginary_value_l958_95841


namespace NUMINAMATH_CALUDE_water_bills_theorem_l958_95829

/-- Water pricing structure -/
def water_price (usage : ℕ) : ℚ :=
  if usage ≤ 10 then 0.45 * usage
  else if usage ≤ 20 then 0.45 * 10 + 0.80 * (usage - 10)
  else 0.45 * 10 + 0.80 * 10 + 1.50 * (usage - 20)

/-- Theorem stating the water bills for households A, B, and C -/
theorem water_bills_theorem :
  ∃ (usage_A usage_B usage_C : ℕ),
    usage_A > 20 ∧ 
    10 < usage_B ∧ usage_B ≤ 20 ∧
    usage_C ≤ 10 ∧
    water_price usage_A - water_price usage_B = 7.10 ∧
    water_price usage_B - water_price usage_C = 3.75 ∧
    water_price usage_A = 14 ∧
    water_price usage_B = 6.9 ∧
    water_price usage_C = 3.15 :=
by sorry

end NUMINAMATH_CALUDE_water_bills_theorem_l958_95829


namespace NUMINAMATH_CALUDE_binomial_expansion_sum_l958_95834

theorem binomial_expansion_sum (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x : ℝ, (3*x - 2)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  a₀ + a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ + 6*a₆ = 82 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_sum_l958_95834


namespace NUMINAMATH_CALUDE_group_size_proof_l958_95846

theorem group_size_proof (total_paise : ℕ) (n : ℕ) : 
  total_paise = 7744 →
  n * n = total_paise →
  n = 88 := by
  sorry

end NUMINAMATH_CALUDE_group_size_proof_l958_95846


namespace NUMINAMATH_CALUDE_linear_equation_equivalence_l958_95861

theorem linear_equation_equivalence (x y : ℝ) (h : x + 3 * y = 3) : 
  (x = 3 - 3 * y) ∧ (y = (3 - x) / 3) := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_equivalence_l958_95861


namespace NUMINAMATH_CALUDE_first_complete_coverage_l958_95820

/-- Triangular number function -/
def triangular (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Function to check if all remainders modulo 12 have been covered -/
def allRemaindersCovered (n : ℕ) : Prop :=
  ∀ r : Fin 12, ∃ k ≤ n, triangular k % 12 = r

/-- The main theorem -/
theorem first_complete_coverage :
  (allRemaindersCovered 19 ∧ ∀ m < 19, ¬allRemaindersCovered m) :=
sorry

end NUMINAMATH_CALUDE_first_complete_coverage_l958_95820


namespace NUMINAMATH_CALUDE_triangle_is_right_angle_l958_95831

theorem triangle_is_right_angle (a b c : ℝ) : 
  a = 3 ∧ b = 4 ∧ c^2 - 10*c + 25 = 0 → c^2 = a^2 + b^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_is_right_angle_l958_95831


namespace NUMINAMATH_CALUDE_circle_through_two_points_tangent_to_line_l958_95832

-- Define the basic geometric objects
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

structure Circle where
  center : Point
  radius : ℝ

-- Define the condition for a point to be on a line
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Define the condition for a circle to pass through a point
def circlePassesThroughPoint (c : Circle) (p : Point) : Prop :=
  (c.center.x - p.x)^2 + (c.center.y - p.y)^2 = c.radius^2

-- Define the condition for a circle to be tangent to a line
def circleTangentToLine (c : Circle) (l : Line) : Prop :=
  ∃ p : Point, pointOnLine p l ∧ circlePassesThroughPoint c p ∧
  ∀ q : Point, pointOnLine q l → (c.center.x - q.x)^2 + (c.center.y - q.y)^2 ≥ c.radius^2

-- Theorem statement
theorem circle_through_two_points_tangent_to_line 
  (A B : Point) (l : Line) : 
  ∃ c : Circle, circlePassesThroughPoint c A ∧ 
                circlePassesThroughPoint c B ∧ 
                circleTangentToLine c l :=
sorry

end NUMINAMATH_CALUDE_circle_through_two_points_tangent_to_line_l958_95832


namespace NUMINAMATH_CALUDE_popcorn_profit_l958_95858

def buying_price : ℝ := 4
def selling_price : ℝ := 8
def bags_sold : ℕ := 30

def profit_per_bag : ℝ := selling_price - buying_price
def total_profit : ℝ := bags_sold * profit_per_bag

theorem popcorn_profit : total_profit = 120 := by
  sorry

end NUMINAMATH_CALUDE_popcorn_profit_l958_95858


namespace NUMINAMATH_CALUDE_triangle_area_l958_95806

theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  (0 < a) → (0 < b) → (0 < c) →
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (A + B + C = π) →
  (b * Real.sin C + c * Real.sin B = 4 * a * Real.sin B * Real.sin C) →
  (b^2 + c^2 - a^2 = 8) →
  (1/2 * b * c * Real.sin A = 4 * Real.sqrt 3 / 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l958_95806


namespace NUMINAMATH_CALUDE_sum_maximized_at_11_or_12_l958_95814

/-- The sequence term defined as a function of n -/
def a (n : ℕ) : ℤ := 24 - 2 * n

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ) : ℤ := n * (24 - n)

/-- Theorem stating that the sum is maximized when n is 11 or 12 -/
theorem sum_maximized_at_11_or_12 :
  ∀ k : ℕ, k > 0 → S k ≤ max (S 11) (S 12) :=
sorry

end NUMINAMATH_CALUDE_sum_maximized_at_11_or_12_l958_95814


namespace NUMINAMATH_CALUDE_c_months_correct_l958_95817

/-- The number of months c put his oxen for grazing -/
def c_months : ℝ :=
  let a_oxen := 10
  let a_months := 7
  let b_oxen := 12
  let b_months := 5
  let c_oxen := 15
  let total_rent := 210
  let c_share := 53.99999999999999
  3

/-- Theorem stating that c_months is correct given the problem conditions -/
theorem c_months_correct :
  let a_oxen := 10
  let a_months := 7
  let b_oxen := 12
  let b_months := 5
  let c_oxen := 15
  let total_rent := 210
  let c_share := 53.99999999999999
  let total_ox_months := a_oxen * a_months + b_oxen * b_months + c_oxen * c_months
  c_share = (c_oxen * c_months / total_ox_months) * total_rent :=
by sorry

#eval c_months

end NUMINAMATH_CALUDE_c_months_correct_l958_95817


namespace NUMINAMATH_CALUDE_max_discount_rate_l958_95890

/-- Represents the maximum discount rate problem --/
theorem max_discount_rate 
  (cost_price : ℝ) 
  (selling_price : ℝ) 
  (min_profit_margin : ℝ) 
  (h1 : cost_price = 4)
  (h2 : selling_price = 5)
  (h3 : min_profit_margin = 0.1) :
  ∃ (max_discount : ℝ),
    max_discount = 0.12 ∧
    ∀ (discount : ℝ),
      discount ≤ max_discount →
      (selling_price * (1 - discount) - cost_price) / cost_price ≥ min_profit_margin :=
sorry

end NUMINAMATH_CALUDE_max_discount_rate_l958_95890


namespace NUMINAMATH_CALUDE_difference_equals_three_44ths_l958_95805

/-- The decimal representation of 0.overline{81} -/
def repeating_decimal : ℚ := 9/11

/-- The decimal representation of 0.75 -/
def decimal_75 : ℚ := 3/4

/-- The theorem stating that the difference between 0.overline{81} and 0.75 is 3/44 -/
theorem difference_equals_three_44ths : 
  repeating_decimal - decimal_75 = 3/44 := by sorry

end NUMINAMATH_CALUDE_difference_equals_three_44ths_l958_95805


namespace NUMINAMATH_CALUDE_sum_four_digit_distinct_remainder_l958_95827

def T : ℕ := sorry

theorem sum_four_digit_distinct_remainder (T : ℕ) : T % 1000 = 960 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_four_digit_distinct_remainder_l958_95827


namespace NUMINAMATH_CALUDE_petya_ran_less_than_two_minutes_l958_95882

/-- Represents the race between Petya and Vasya -/
structure Race where
  distance : ℝ
  petyaSpeed : ℝ
  petyaTime : ℝ
  vasyaStartDelay : ℝ

/-- Conditions of the race -/
def raceConditions (r : Race) : Prop :=
  r.distance > 0 ∧
  r.petyaSpeed > 0 ∧
  r.petyaTime > 0 ∧
  r.vasyaStartDelay = 1 ∧
  r.distance = r.petyaSpeed * r.petyaTime ∧
  r.petyaTime < r.distance / (2 * r.petyaSpeed) + r.vasyaStartDelay

/-- Theorem: Under the given conditions, Petya ran the distance in less than two minutes -/
theorem petya_ran_less_than_two_minutes (r : Race) (h : raceConditions r) : r.petyaTime < 2 := by
  sorry

end NUMINAMATH_CALUDE_petya_ran_less_than_two_minutes_l958_95882


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l958_95843

theorem arithmetic_mean_of_fractions :
  let a := 8 / 12
  let b := 5 / 6
  let c := 9 / 12
  c = (a + b) / 2 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l958_95843


namespace NUMINAMATH_CALUDE_polar_to_rectangular_equation_l958_95881

/-- The rectangular coordinate equation of the curve ρ = sin θ - 3cos θ -/
theorem polar_to_rectangular_equation :
  ∀ (x y ρ θ : ℝ),
  (ρ = Real.sin θ - 3 * Real.cos θ) →
  (x = ρ * Real.cos θ) →
  (y = ρ * Real.sin θ) →
  (x^2 - 3*x + y^2 - y = 0) := by
  sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_equation_l958_95881


namespace NUMINAMATH_CALUDE_cubic_factorization_l958_95884

theorem cubic_factorization (m : ℝ) : m^3 - 4*m = m*(m + 2)*(m - 2) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l958_95884


namespace NUMINAMATH_CALUDE_complex_modulus_one_l958_95889

theorem complex_modulus_one (a : ℝ) :
  let z : ℂ := (a - 1) + a * Complex.I
  Complex.abs z = 1 → a = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_one_l958_95889


namespace NUMINAMATH_CALUDE_final_number_is_365_l958_95845

/-- Represents the skipping pattern for a single student -/
def skip_pattern (n : ℕ) : Bool :=
  n % 4 ≠ 2

/-- Applies the skipping pattern for a given number of students -/
def apply_skip_pattern (students : ℕ) (n : ℕ) : Bool :=
  match students with
  | 0 => true
  | k + 1 => skip_pattern n && apply_skip_pattern k (((n - 1) / 4) + 1)

/-- The main theorem stating that after 8 students apply the skipping pattern, 365 is the only number remaining -/
theorem final_number_is_365 :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 1100 → (apply_skip_pattern 8 n ↔ n = 365) :=
by sorry

end NUMINAMATH_CALUDE_final_number_is_365_l958_95845


namespace NUMINAMATH_CALUDE_balloon_radius_increase_l958_95866

theorem balloon_radius_increase (C₁ C₂ r₁ r₂ Δr : ℝ) : 
  C₁ = 20 → 
  C₂ = 25 → 
  C₁ = 2 * Real.pi * r₁ → 
  C₂ = 2 * Real.pi * r₂ → 
  Δr = r₂ - r₁ → 
  Δr = 5 / (2 * Real.pi) := by
sorry

end NUMINAMATH_CALUDE_balloon_radius_increase_l958_95866


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l958_95865

theorem right_triangle_hypotenuse (L M N : ℝ) : 
  -- LMN is a right triangle with right angle at M
  -- sin N = 3/5
  -- LM = 18
  Real.sin N = 3/5 → LM = 18 → LN = 30 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l958_95865


namespace NUMINAMATH_CALUDE_smallest_n_for_probability_condition_l958_95847

theorem smallest_n_for_probability_condition : 
  (∃ n : ℕ+, (1 : ℚ) / (n * (n + 1)) < 1 / 2020 ∧ 
    ∀ m : ℕ+, m < n → (1 : ℚ) / (m * (m + 1)) ≥ 1 / 2020) ∧
  (∀ n : ℕ+, (1 : ℚ) / (n * (n + 1)) < 1 / 2020 ∧ 
    ∀ m : ℕ+, m < n → (1 : ℚ) / (m * (m + 1)) ≥ 1 / 2020 → n = 45) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_probability_condition_l958_95847


namespace NUMINAMATH_CALUDE_change_in_expression_l958_95837

theorem change_in_expression (x a : ℝ) (k : ℝ) (h : k > 0) :
  let f := fun x => 3 * x^2 - k
  (f (x + a) - f x = 6 * a * x + 3 * a^2) ∧
  (f (x - a) - f x = -6 * a * x + 3 * a^2) :=
by sorry

end NUMINAMATH_CALUDE_change_in_expression_l958_95837


namespace NUMINAMATH_CALUDE_two_numbers_sum_2014_l958_95874

theorem two_numbers_sum_2014 : ∃ (x y : ℕ), x > y ∧ x + y = 2014 ∧ 3 * (x / 100) = y + 6 ∧ y = 51 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_sum_2014_l958_95874


namespace NUMINAMATH_CALUDE_second_quadrant_angles_l958_95856

-- Define a function to check if an angle is in the second quadrant
def is_in_second_quadrant (angle : ℝ) : Prop :=
  90 < angle % 360 ∧ angle % 360 ≤ 180

-- Define the given angles
def angle1 : ℝ := -120
def angle2 : ℝ := -240
def angle3 : ℝ := 180
def angle4 : ℝ := 495

-- Theorem statement
theorem second_quadrant_angles :
  is_in_second_quadrant angle2 ∧
  is_in_second_quadrant angle4 ∧
  ¬is_in_second_quadrant angle1 ∧
  ¬is_in_second_quadrant angle3 :=
sorry

end NUMINAMATH_CALUDE_second_quadrant_angles_l958_95856


namespace NUMINAMATH_CALUDE_expected_value_is_500_l958_95871

/-- Represents the prize structure for a game activity -/
structure PrizeStructure where
  firstPrize : ℝ
  commonDifference : ℝ

/-- Represents the probability distribution for winning prizes -/
structure ProbabilityDistribution where
  firstTerm : ℝ
  commonRatio : ℝ

/-- Calculates the expected value of the prize -/
def expectedValue (ps : PrizeStructure) (pd : ProbabilityDistribution) : ℝ :=
  let secondPrize := ps.firstPrize + ps.commonDifference
  let thirdPrize := ps.firstPrize + 2 * ps.commonDifference
  let secondProb := pd.firstTerm * pd.commonRatio
  let thirdProb := pd.firstTerm * pd.commonRatio * pd.commonRatio
  ps.firstPrize * pd.firstTerm + secondPrize * secondProb + thirdPrize * thirdProb

/-- The main theorem stating that the expected value is 500 yuan -/
theorem expected_value_is_500 
  (ps : PrizeStructure) 
  (pd : ProbabilityDistribution) 
  (h1 : ps.firstPrize = 700)
  (h2 : ps.commonDifference = -140)
  (h3 : pd.commonRatio = 2)
  (h4 : pd.firstTerm + pd.firstTerm * pd.commonRatio + pd.firstTerm * pd.commonRatio * pd.commonRatio = 1) :
  expectedValue ps pd = 500 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_is_500_l958_95871


namespace NUMINAMATH_CALUDE_sum_congruence_l958_95879

theorem sum_congruence : (2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999) % 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_congruence_l958_95879


namespace NUMINAMATH_CALUDE_population_increase_rate_l958_95808

def birth_rate : ℚ := 32 / 1000
def death_rate : ℚ := 11 / 1000

theorem population_increase_rate : 
  (birth_rate - death_rate) * 100 = 2.1 := by sorry

end NUMINAMATH_CALUDE_population_increase_rate_l958_95808


namespace NUMINAMATH_CALUDE_birthday_candles_l958_95844

/-- Represents the ages of the seven children --/
structure ChildrenAges where
  youngest : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ
  fifth : ℕ
  sixth : ℕ
  seventh : ℕ

/-- The problem statement --/
theorem birthday_candles (ages : ChildrenAges) : 
  ages.youngest = 1 →
  ages.second = 2 →
  ages.third = 3 →
  ages.fourth = 4 →
  ages.fifth = 5 →
  ages.sixth = ages.seventh →
  (ages.youngest + ages.second + ages.third + ages.fourth + ages.fifth + ages.sixth + ages.seventh) = 
    2 * (ages.second - 1 + ages.third - 1 + ages.fourth - 1 + ages.fifth - 1 + ages.sixth - 1 + ages.seventh - 1) + 2 →
  (ages.youngest + ages.second + ages.third + ages.fourth + ages.fifth + ages.sixth + ages.seventh) = 27 := by
  sorry


end NUMINAMATH_CALUDE_birthday_candles_l958_95844


namespace NUMINAMATH_CALUDE_factorization_of_polynomial_l958_95870

theorem factorization_of_polynomial (x : ℝ) :
  x^2 - 6*x + 9 - 64*x^4 = (-8*x^2 + x - 3)*(8*x^2 + x - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_polynomial_l958_95870


namespace NUMINAMATH_CALUDE_unique_tiling_l958_95872

/-- A set is bounded below or above -/
def BoundedBelowOrAbove (A : Set ℝ) : Prop :=
  (∃ l, ∀ a ∈ A, l ≤ a) ∨ (∃ u, ∀ a ∈ A, a ≤ u)

/-- S tiles A -/
def Tiles (S A : Set ℝ) : Prop :=
  ∃ (I : Type) (f : I → Set ℝ), (∀ i, f i ⊆ S) ∧ (∀ i j, i ≠ j → f i ∩ f j = ∅) ∧ (⋃ i, f i) = A

/-- Unique tiling -/
def UniqueTiling (S A : Set ℝ) : Prop :=
  ∀ (I J : Type) (f : I → Set ℝ) (g : J → Set ℝ),
    (∀ i, f i ⊆ S) ∧ (∀ i j, i ≠ j → f i ∩ f j = ∅) ∧ (⋃ i, f i) = A →
    (∀ i, g i ⊆ S) ∧ (∀ i j, i ≠ j → g i ∩ g j = ∅) ∧ (⋃ i, g i) = A →
    ∃ (h : I ≃ J), ∀ i, f i = g (h i)

theorem unique_tiling (A : Set ℝ) (S : Set ℝ) :
  BoundedBelowOrAbove A → Tiles S A → UniqueTiling S A := by
  sorry

end NUMINAMATH_CALUDE_unique_tiling_l958_95872


namespace NUMINAMATH_CALUDE_WXYZ_perimeter_l958_95840

/-- Represents a rectangle with a perimeter --/
structure Rectangle where
  perimeter : ℕ

/-- Represents the large rectangle WXYZ --/
def WXYZ : Rectangle := sorry

/-- The four smaller rectangles that WXYZ is divided into --/
def smallRectangles : Fin 4 → Rectangle := sorry

/-- The sum of perimeters of diagonally opposite rectangles equals the perimeter of WXYZ --/
axiom perimeter_sum (i j : Fin 4) (h : i.val + j.val = 3) : 
  (smallRectangles i).perimeter + (smallRectangles j).perimeter = WXYZ.perimeter

/-- The perimeters of three of the smaller rectangles --/
axiom known_perimeters : 
  ∃ (i j k : Fin 4) (h : i ≠ j ∧ j ≠ k ∧ i ≠ k),
    (smallRectangles i).perimeter = 11 ∧
    (smallRectangles j).perimeter = 16 ∧
    (smallRectangles k).perimeter = 19

/-- The perimeter of the fourth rectangle is between 11 and 19 --/
axiom fourth_perimeter :
  ∃ (l : Fin 4), ∀ (i : Fin 4), 
    (smallRectangles i).perimeter ≠ 11 → 
    (smallRectangles i).perimeter ≠ 16 → 
    (smallRectangles i).perimeter ≠ 19 →
    11 < (smallRectangles l).perimeter ∧ (smallRectangles l).perimeter < 19

/-- The perimeter of WXYZ is 30 --/
theorem WXYZ_perimeter : WXYZ.perimeter = 30 := by sorry

end NUMINAMATH_CALUDE_WXYZ_perimeter_l958_95840


namespace NUMINAMATH_CALUDE_eggs_per_tray_l958_95804

theorem eggs_per_tray (total_trays : ℕ) (total_eggs : ℕ) (eggs_per_tray : ℕ) : 
  total_trays = 7 →
  total_eggs = 70 →
  total_eggs = total_trays * eggs_per_tray →
  eggs_per_tray = 10 := by
sorry

end NUMINAMATH_CALUDE_eggs_per_tray_l958_95804


namespace NUMINAMATH_CALUDE_focus_coordinates_l958_95880

/-- A parabola with equation x^2 = 2py where p > 0 -/
structure Parabola where
  p : ℝ
  p_pos : p > 0

/-- The directrix of a parabola -/
def directrix (par : Parabola) : ℝ → ℝ → Prop :=
  fun x y => y = -2

/-- The focus of a parabola -/
def focus (par : Parabola) : ℝ × ℝ :=
  (0, par.p)

/-- Theorem stating that if the directrix of a parabola passes through (0, -2),
    then its focus is at (0, 2) -/
theorem focus_coordinates (par : Parabola) :
  directrix par 0 (-2) → focus par = (0, 2) := by
  sorry

end NUMINAMATH_CALUDE_focus_coordinates_l958_95880


namespace NUMINAMATH_CALUDE_calculation_proof_l958_95822

theorem calculation_proof : 1 - (1/2)⁻¹ * Real.sin (π/3) + |2^0 - Real.sqrt 3| = 0 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l958_95822


namespace NUMINAMATH_CALUDE_y_in_terms_of_x_l958_95878

theorem y_in_terms_of_x (p : ℝ) (x y : ℝ) 
  (hx : x = 1 + 3^p) (hy : y = 1 + 3^(-p)) : 
  y = x / (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_y_in_terms_of_x_l958_95878


namespace NUMINAMATH_CALUDE_ratio_a_to_c_l958_95885

theorem ratio_a_to_c (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 4 / 3)
  (hdb : d / b = 1 / 7) :
  a / c = 105 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ratio_a_to_c_l958_95885


namespace NUMINAMATH_CALUDE_remainder_eight_power_2002_mod_9_l958_95803

theorem remainder_eight_power_2002_mod_9 : 8^2002 % 9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_eight_power_2002_mod_9_l958_95803


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l958_95819

/-- An arithmetic sequence with positive terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  positive : ∀ n, a n > 0
  arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

/-- The theorem statement -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence) 
  (h1 : seq.a 1 = 3)
  (h2 : seq.a 1 + seq.a 2 + seq.a 3 = 21) :
  seq.a 4 + seq.a 5 + seq.a 6 = 57 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l958_95819


namespace NUMINAMATH_CALUDE_intersection_condition_chord_length_condition_l958_95825

-- Define the ellipse and line
def ellipse (x y : ℝ) : Prop := 4 * x^2 + y^2 = 1
def line (x y m : ℝ) : Prop := y = x + m

-- Theorem for intersection condition
theorem intersection_condition (m : ℝ) :
  (∃ x y : ℝ, ellipse x y ∧ line x y m) ↔ -Real.sqrt 5 / 2 ≤ m ∧ m ≤ Real.sqrt 5 / 2 :=
sorry

-- Theorem for chord length condition
theorem chord_length_condition (m : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧ 
    line x₁ y₁ m ∧ line x₂ y₂ m ∧
    Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 4 * Real.sqrt 2 / 5) →
  m = 1/2 ∨ m = -1/2 :=
sorry

end NUMINAMATH_CALUDE_intersection_condition_chord_length_condition_l958_95825


namespace NUMINAMATH_CALUDE_geralds_apples_count_l958_95859

/-- Given that Pam has 10 bags of apples, 1200 apples in total, and each of her bags
    contains 3 times the number of apples in each of Gerald's bags, prove that
    each of Gerald's bags contains 40 apples. -/
theorem geralds_apples_count (pam_bags : ℕ) (pam_total_apples : ℕ) (gerald_apples : ℕ) 
  (h1 : pam_bags = 10)
  (h2 : pam_total_apples = 1200)
  (h3 : pam_total_apples = pam_bags * (3 * gerald_apples)) :
  gerald_apples = 40 := by
  sorry

end NUMINAMATH_CALUDE_geralds_apples_count_l958_95859


namespace NUMINAMATH_CALUDE_problem_statement_l958_95839

theorem problem_statement (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (abs (x + 2*y) + abs (x - y) ≤ 5/2 ↔ 1/6 ≤ x ∧ x < 1) ∧
  ((1/x^2 - 1) * (1/y^2 - 1) ≥ 9) := by sorry

end NUMINAMATH_CALUDE_problem_statement_l958_95839


namespace NUMINAMATH_CALUDE_yellow_ball_count_l958_95813

theorem yellow_ball_count (r b g y : ℕ) : 
  r = 2 * b →
  b = 2 * g →
  y > 7 →
  r + b + g + y = 27 →
  y = 20 := by
sorry

end NUMINAMATH_CALUDE_yellow_ball_count_l958_95813


namespace NUMINAMATH_CALUDE_mountain_hike_l958_95899

theorem mountain_hike (rate_up : ℝ) (time : ℝ) (rate_down_factor : ℝ) : 
  rate_up = 3 →
  time = 2 →
  rate_down_factor = 1.5 →
  (rate_up * time) * rate_down_factor = 9 := by
  sorry

end NUMINAMATH_CALUDE_mountain_hike_l958_95899


namespace NUMINAMATH_CALUDE_veridux_female_employees_l958_95892

/-- Proves that the number of female employees at Veridux Corporation is 90 -/
theorem veridux_female_employees :
  let total_employees : ℕ := 250
  let total_managers : ℕ := 40
  let male_associates : ℕ := 160
  let female_managers : ℕ := 40
  let total_associates : ℕ := total_employees - total_managers
  let female_associates : ℕ := total_associates - male_associates
  let female_employees : ℕ := female_managers + female_associates
  female_employees = 90 := by
  sorry


end NUMINAMATH_CALUDE_veridux_female_employees_l958_95892


namespace NUMINAMATH_CALUDE_find_set_C_l958_95883

def A : Set ℝ := {x | x^2 - 5*x + 6 = 0}
def B (a : ℝ) : Set ℝ := {x | a*x - 6 = 0}

theorem find_set_C : 
  ∃ C : Set ℝ, 
    (C = {0, 2, 3}) ∧ 
    (∀ a : ℝ, a ∈ C ↔ (A ∪ B a = A)) :=
by sorry

end NUMINAMATH_CALUDE_find_set_C_l958_95883


namespace NUMINAMATH_CALUDE_ordered_pairs_count_l958_95891

theorem ordered_pairs_count : 
  ∃! (pairs : List (ℕ × ℕ)), 
    (∀ (m n : ℕ), (m, n) ∈ pairs ↔ m > 0 ∧ n > 0 ∧ 6 / m + 3 / n = 1) ∧
    pairs.length = 6 := by
  sorry

end NUMINAMATH_CALUDE_ordered_pairs_count_l958_95891


namespace NUMINAMATH_CALUDE_heptagon_angles_l958_95818

/-- The number of sides in a heptagon -/
def n : ℕ := 7

/-- The measure of an interior angle of a regular heptagon -/
def interior_angle : ℚ := (5 * 180) / n

/-- The measure of an exterior angle of a regular heptagon -/
def exterior_angle : ℚ := 180 - interior_angle

theorem heptagon_angles :
  (interior_angle = (5 * 180) / n) ∧
  (exterior_angle = 180 - ((5 * 180) / n)) := by
  sorry

end NUMINAMATH_CALUDE_heptagon_angles_l958_95818


namespace NUMINAMATH_CALUDE_exponent_calculation_l958_95810

theorem exponent_calculation : (((18^15 / 18^14)^3 * 8^3) / 4^5) = 2916 := by
  sorry

end NUMINAMATH_CALUDE_exponent_calculation_l958_95810


namespace NUMINAMATH_CALUDE_meeting_point_distance_l958_95801

theorem meeting_point_distance (total_distance : ℝ) (speed1 speed2 : ℝ) 
  (h1 : total_distance = 36)
  (h2 : speed1 = 2)
  (h3 : speed2 = 4) :
  speed1 * (total_distance / (speed1 + speed2)) = 12 :=
by sorry

end NUMINAMATH_CALUDE_meeting_point_distance_l958_95801
