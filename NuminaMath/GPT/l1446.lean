import Mathlib

namespace NUMINAMATH_GPT_largest_prime_factor_among_numbers_l1446_144695

-- Definitions of the numbers with their prime factors
def num1 := 39
def num2 := 51
def num3 := 77
def num4 := 91
def num5 := 121

def prime_factors (n : ℕ) : List ℕ := sorry  -- Placeholder for the prime factors function

-- Prime factors for the given numbers
def factors_num1 := prime_factors num1
def factors_num2 := prime_factors num2
def factors_num3 := prime_factors num3
def factors_num4 := prime_factors num4
def factors_num5 := prime_factors num5

-- Extract the largest prime factor from a list of factors
def largest_prime_factor (factors : List ℕ) : ℕ := sorry  -- Placeholder for the largest_prime_factor function

-- Largest prime factors for each number
def largest_prime_factor_num1 := largest_prime_factor factors_num1
def largest_prime_factor_num2 := largest_prime_factor factors_num2
def largest_prime_factor_num3 := largest_prime_factor factors_num3
def largest_prime_factor_num4 := largest_prime_factor factors_num4
def largest_prime_factor_num5 := largest_prime_factor factors_num5

theorem largest_prime_factor_among_numbers :
  largest_prime_factor_num2 = 17 ∧
  largest_prime_factor_num1 = 13 ∧
  largest_prime_factor_num3 = 11 ∧
  largest_prime_factor_num4 = 13 ∧
  largest_prime_factor_num5 = 11 ∧
  (largest_prime_factor_num2 > largest_prime_factor_num1) ∧
  (largest_prime_factor_num2 > largest_prime_factor_num3) ∧
  (largest_prime_factor_num2 > largest_prime_factor_num4) ∧
  (largest_prime_factor_num2 > largest_prime_factor_num5)
:= by
  -- skeleton proof, details to be filled in
  sorry

end NUMINAMATH_GPT_largest_prime_factor_among_numbers_l1446_144695


namespace NUMINAMATH_GPT_sum_of_possible_values_N_l1446_144680

variable (a b c N : ℕ)

theorem sum_of_possible_values_N :
  (N = a * b * c) ∧ (N = 8 * (a + b + c)) ∧ (c = 2 * a + b) → N = 136 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_possible_values_N_l1446_144680


namespace NUMINAMATH_GPT_find_a_l1446_144625

noncomputable def g (x : ℝ) := 5 * x - 7

theorem find_a (a : ℝ) (h : g a = 0) : a = 7 / 5 :=
sorry

end NUMINAMATH_GPT_find_a_l1446_144625


namespace NUMINAMATH_GPT_quad_sin_theorem_l1446_144639

-- Define the necessary entities in Lean
structure Quadrilateral (A B C D : Type) :=
(angleB : ℝ)
(angleD : ℝ)
(angleA : ℝ)

-- Define the main theorem
theorem quad_sin_theorem {A B C D : Type} (quad : Quadrilateral A B C D) (AC AD : ℝ) (α : ℝ) :
  quad.angleB = 90 ∧ quad.angleD = 90 ∧ quad.angleA = α → AD = AC * Real.sin α := 
sorry

end NUMINAMATH_GPT_quad_sin_theorem_l1446_144639


namespace NUMINAMATH_GPT_train_speed_kmh_l1446_144668

-- Definitions based on the conditions
variables (L V : ℝ)
variable (h1 : L = 10 * V)
variable (h2 : L + 600 = 30 * V)

-- The proof statement, no solution steps, just the conclusion
theorem train_speed_kmh : (V * 3.6) = 108 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_kmh_l1446_144668


namespace NUMINAMATH_GPT_range_of_a_l1446_144615

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, a*x^2 - 3*x + 2 = 0) ∧ 
  (∀ x y : ℝ, a*x^2 - 3*x + 2 = 0 ∧ a*y^2 - 3*y + 2 = 0 → x = y) 
  ↔ (a = 0 ∨ a = 9 / 8) := by sorry

end NUMINAMATH_GPT_range_of_a_l1446_144615


namespace NUMINAMATH_GPT_number_of_classes_l1446_144682

theorem number_of_classes (x : ℕ) (h : x * (x - 1) / 2 = 28) : x = 8 := by
  sorry

end NUMINAMATH_GPT_number_of_classes_l1446_144682


namespace NUMINAMATH_GPT_find_x_coordinate_l1446_144627

theorem find_x_coordinate :
  ∃ x : ℝ, (∃ m b : ℝ, (∀ y x : ℝ, y = m * x + b) ∧ 
                     ((3 = m * 10 + b) ∧ 
                      (0 = m * 4 + b)
                     ) ∧ 
                     (-3 = m * x + b) ∧ 
                     (x = -2)) :=
sorry

end NUMINAMATH_GPT_find_x_coordinate_l1446_144627


namespace NUMINAMATH_GPT_g_of_3_eq_seven_over_two_l1446_144696

theorem g_of_3_eq_seven_over_two :
  ∀ f g : ℝ → ℝ,
  (∀ x, f x = (2 * x + 3) / (x - 1)) →
  (∀ x, g x = (x + 4) / (x - 1)) →
  g 3 = 7 / 2 :=
by
  sorry

end NUMINAMATH_GPT_g_of_3_eq_seven_over_two_l1446_144696


namespace NUMINAMATH_GPT_ball_price_equation_l1446_144617

structure BallPrices where
  (x : Real) -- price of each soccer ball in yuan
  (condition1 : ∀ (x : Real), (1500 / (x + 20) - 800 / x = 5))

/-- Prove that the equation follows from the given conditions. -/
theorem ball_price_equation (b : BallPrices) : 1500 / (b.x + 20) - 800 / b.x = 5 := 
by sorry

end NUMINAMATH_GPT_ball_price_equation_l1446_144617


namespace NUMINAMATH_GPT_Merry_sold_470_apples_l1446_144600

-- Define the conditions
def boxes_on_Saturday : Nat := 50
def boxes_on_Sunday : Nat := 25
def apples_per_box : Nat := 10
def boxes_left : Nat := 3

-- Define the question as the number of apples sold
theorem Merry_sold_470_apples :
  (boxes_on_Saturday - boxes_on_Sunday) * apples_per_box +
  (boxes_on_Sunday - boxes_left) * apples_per_box = 470 := by
  sorry

end NUMINAMATH_GPT_Merry_sold_470_apples_l1446_144600


namespace NUMINAMATH_GPT_sum_invested_7000_l1446_144629

-- Define the conditions
def interest_15 (P : ℝ) : ℝ := P * 0.15 * 2
def interest_12 (P : ℝ) : ℝ := P * 0.12 * 2

-- Main statement to prove
theorem sum_invested_7000 (P : ℝ) (h : interest_15 P - interest_12 P = 420) : P = 7000 := by
  sorry

end NUMINAMATH_GPT_sum_invested_7000_l1446_144629


namespace NUMINAMATH_GPT_closest_ratio_l1446_144690

theorem closest_ratio
  (a_0 : ℝ)
  (h_pos : a_0 > 0)
  (a_10 : ℝ)
  (h_eq : a_10 = a_0 * (1 + 0.05) ^ 10) :
  abs ((a_10 / a_0) - 1.6) ≤ abs ((a_10 / a_0) - 1.5) ∧
  abs ((a_10 / a_0) - 1.6) ≤ abs ((a_10 / a_0) - 1.7) ∧
  abs ((a_10 / a_0) - 1.6) ≤ abs ((a_10 / a_0) - 1.8) := 
sorry

end NUMINAMATH_GPT_closest_ratio_l1446_144690


namespace NUMINAMATH_GPT_accurate_to_ten_thousandth_l1446_144644

/-- Define the original number --/
def original_number : ℕ := 580000

/-- Define the accuracy of the number represented by 5.8 * 10^5 --/
def is_accurate_to_ten_thousandth_place (n : ℕ) : Prop :=
  n = 5 * 100000 + 8 * 10000

/-- The statement to be proven --/
theorem accurate_to_ten_thousandth : is_accurate_to_ten_thousandth_place original_number :=
by
  sorry

end NUMINAMATH_GPT_accurate_to_ten_thousandth_l1446_144644


namespace NUMINAMATH_GPT_circle_radius_given_circumference_l1446_144616

theorem circle_radius_given_circumference (C : ℝ) (hC : C = 3.14) : ∃ r : ℝ, C = 2 * Real.pi * r ∧ r = 0.5 := 
by
  sorry

end NUMINAMATH_GPT_circle_radius_given_circumference_l1446_144616


namespace NUMINAMATH_GPT_collinear_vectors_l1446_144655

theorem collinear_vectors (x : ℝ) :
  (∃ k : ℝ, (2, 4) = (k * 2, k * 4) ∧ (k * 2 = x ∧ k * 4 = 6)) → x = 3 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_collinear_vectors_l1446_144655


namespace NUMINAMATH_GPT_danny_wrappers_more_than_soda_cans_l1446_144693

theorem danny_wrappers_more_than_soda_cans :
  (67 - 22 = 45) := sorry

end NUMINAMATH_GPT_danny_wrappers_more_than_soda_cans_l1446_144693


namespace NUMINAMATH_GPT_average_speed_of_entire_trip_l1446_144673

/-- Conditions -/
def distance_local : ℝ := 40  -- miles
def speed_local : ℝ := 20  -- mph
def distance_highway : ℝ := 180  -- miles
def speed_highway : ℝ := 60  -- mph

/-- Average speed proof statement -/
theorem average_speed_of_entire_trip :
  let total_distance := distance_local + distance_highway
  let total_time := distance_local / speed_local + distance_highway / speed_highway
  total_distance / total_time = 44 :=
by
  sorry

end NUMINAMATH_GPT_average_speed_of_entire_trip_l1446_144673


namespace NUMINAMATH_GPT_range_of_a_l1446_144643

theorem range_of_a 
  (a : ℝ):
  (∀ x : ℝ, |x + 2| + |x - 1| > a^2 - 2 * a) ↔ (-1 < a ∧ a < 3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1446_144643


namespace NUMINAMATH_GPT_correct_conclusions_l1446_144694

def pos_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n > 0

def sum_of_n_terms (S a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n+1) * S (n+1) = 9

def second_term_less_than_3 (a S : ℕ → ℝ) : Prop :=
  a 1 < 3

def is_decreasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) < a n

def exists_term_less_than_1_over_100 (a : ℕ → ℝ) : Prop :=
  ∃ n : ℕ, a n < 1/100

theorem correct_conclusions (a S : ℕ → ℝ) :
  pos_sequence a → sum_of_n_terms S a →
  second_term_less_than_3 a S ∧ (¬(∀ q : ℝ, ∃ r : ℝ, ∀ n : ℕ, a n = r * q ^ n)) ∧ is_decreasing_sequence a ∧ exists_term_less_than_1_over_100 a :=
sorry

end NUMINAMATH_GPT_correct_conclusions_l1446_144694


namespace NUMINAMATH_GPT_decimal_to_binary_thirteen_l1446_144632

theorem decimal_to_binary_thirteen : (13 : ℕ) = 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0 :=
by
  sorry

end NUMINAMATH_GPT_decimal_to_binary_thirteen_l1446_144632


namespace NUMINAMATH_GPT_smallest_value_of_Q_l1446_144658

noncomputable def Q (x : ℝ) : ℝ := x^4 - 4*x^3 + 7*x^2 - 2*x + 10

theorem smallest_value_of_Q :
  min (Q 1) (min (10 : ℝ) (min (4 : ℝ) (min (1 - 4 + 7 - 2 + 10 : ℝ) (2.5 : ℝ)))) = 2.5 :=
by sorry

end NUMINAMATH_GPT_smallest_value_of_Q_l1446_144658


namespace NUMINAMATH_GPT_total_number_of_sheep_l1446_144605

theorem total_number_of_sheep (a₁ a₂ a₃ a₄ a₅ a₆ a₇ d : ℤ)
    (h1 : a₂ = a₁ + d)
    (h2 : a₃ = a₁ + 2 * d)
    (h3 : a₄ = a₁ + 3 * d)
    (h4 : a₅ = a₁ + 4 * d)
    (h5 : a₆ = a₁ + 5 * d)
    (h6 : a₇ = a₁ + 6 * d)
    (h_sum : a₁ + a₂ + a₃ = 33)
    (h_seven: 2 * a₂ + 9 = a₇) :
    a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 133 := sorry

end NUMINAMATH_GPT_total_number_of_sheep_l1446_144605


namespace NUMINAMATH_GPT_longest_side_of_enclosure_l1446_144646

theorem longest_side_of_enclosure
  (l w : ℝ)
  (h1 : 2 * l + 2 * w = 180)
  (h2 : l * w = 1440) :
  l = 72 ∨ w = 72 :=
by {
  sorry
}

end NUMINAMATH_GPT_longest_side_of_enclosure_l1446_144646


namespace NUMINAMATH_GPT_inequality_solution_sets_l1446_144624

variable (a x : ℝ)

theorem inequality_solution_sets:
    ({x | 12 * x^2 - a * x > a^2} =
        if a > 0 then {x | x < -a/4} ∪ {x | x > a/3}
        else if a = 0 then {x | x ≠ 0}
        else {x | x < a/3} ∪ {x | x > -a/4}) :=
by sorry

end NUMINAMATH_GPT_inequality_solution_sets_l1446_144624


namespace NUMINAMATH_GPT_inequality_proof_l1446_144607

-- Defining the conditions
variable (a b : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) (cond : 1 / a + 1 / b = 1)

-- Defining the theorem to be proved
theorem inequality_proof (n : ℕ) : 
  (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n+1) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1446_144607


namespace NUMINAMATH_GPT_solve_for_x_and_y_l1446_144612

theorem solve_for_x_and_y (x y : ℚ) (h : (1 / 6) + (6 / x) = (14 / x) + (1 / 14) + y) : x = 84 ∧ y = 0 :=
sorry

end NUMINAMATH_GPT_solve_for_x_and_y_l1446_144612


namespace NUMINAMATH_GPT_find_x_l1446_144640

def is_mean_twice_mode (l : List ℕ) (mean eq_mode : ℕ) : Prop :=
  l.sum / l.length = eq_mode * 2

theorem find_x (x : ℕ) (h1 : x > 0) (h2 : x ≤ 100)
  (h3 : is_mean_twice_mode [20, x, x, x, x] x (x * 2)) : x = 10 :=
sorry

end NUMINAMATH_GPT_find_x_l1446_144640


namespace NUMINAMATH_GPT_area_AKM_less_than_area_ABC_l1446_144619

-- Define the rectangle ABCD
structure Rectangle :=
(A B C D : ℝ) -- Four vertices of the rectangle
(side_AB : ℝ) (side_BC : ℝ) (side_CD : ℝ) (side_DA : ℝ)

-- Define the arbitrary points K and M on sides BC and CD respectively
variables (B C D K M : ℝ)

-- Define the area of triangle function and area of rectangle function
def area_triangle (A B C : ℝ) : ℝ := sorry -- Assuming a function calculating area of triangle given 3 vertices
def area_rectangle (A B C D : ℝ) : ℝ := sorry -- Assuming a function calculating area of rectangle given 4 vertices

-- Assuming the conditions given in the problem statement
variables (A : ℝ) (rect : Rectangle)

-- Prove that the area of triangle AKM is less than the area of triangle ABC
theorem area_AKM_less_than_area_ABC : 
  ∀ (K M : ℝ), K ∈ [B,C] → M ∈ [C,D] →
    area_triangle A K M < area_triangle A B C := sorry

end NUMINAMATH_GPT_area_AKM_less_than_area_ABC_l1446_144619


namespace NUMINAMATH_GPT_pictures_left_l1446_144666

def initial_zoo_pics : ℕ := 49
def initial_museum_pics : ℕ := 8
def deleted_pics : ℕ := 38

theorem pictures_left (total_pics : ℕ) :
  total_pics = initial_zoo_pics + initial_museum_pics →
  total_pics - deleted_pics = 19 :=
by
  intro h1
  rw [h1]
  sorry

end NUMINAMATH_GPT_pictures_left_l1446_144666


namespace NUMINAMATH_GPT_katie_five_dollar_bills_l1446_144609

theorem katie_five_dollar_bills (x y : ℕ) (h1 : x + y = 12) (h2 : 5 * x + 10 * y = 80) : x = 8 :=
by
  sorry

end NUMINAMATH_GPT_katie_five_dollar_bills_l1446_144609


namespace NUMINAMATH_GPT_growth_operation_two_operations_growth_operation_four_operations_l1446_144653

noncomputable def growth_operation_perimeter (initial_side_length : ℕ) (growth_operations : ℕ) := 
  initial_side_length * 3 * (4/3 : ℚ)^(growth_operations + 1)

theorem growth_operation_two_operations :
  growth_operation_perimeter 9 2 = 48 := by sorry

theorem growth_operation_four_operations :
  growth_operation_perimeter 9 4 = 256 / 3 := by sorry

end NUMINAMATH_GPT_growth_operation_two_operations_growth_operation_four_operations_l1446_144653


namespace NUMINAMATH_GPT_sequence_50th_term_l1446_144654

def sequence_term (n : ℕ) : ℕ × ℕ :=
  (5 + (n - 1), n - 1)

theorem sequence_50th_term :
  sequence_term 50 = (54, 49) :=
by
  sorry

end NUMINAMATH_GPT_sequence_50th_term_l1446_144654


namespace NUMINAMATH_GPT_rectangular_reconfiguration_l1446_144670

theorem rectangular_reconfiguration (k : ℕ) (n : ℕ) (h₁ : k - 5 > 0) (h₂ : k ≥ 6) (h₃ : k ≤ 9) :
  (k * (k - 5) = n^2) → (n = 6) :=
by {
  sorry  -- proof is omitted
}

end NUMINAMATH_GPT_rectangular_reconfiguration_l1446_144670


namespace NUMINAMATH_GPT_family_reunion_handshakes_l1446_144671

theorem family_reunion_handshakes (married_couples : ℕ) (participants : ℕ) (allowed_handshakes : ℕ) (total_handshakes : ℕ) :
  married_couples = 8 →
  participants = married_couples * 2 →
  allowed_handshakes = participants - 1 - 1 - 6 →
  total_handshakes = (participants * allowed_handshakes) / 2 →
  total_handshakes = 64 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_family_reunion_handshakes_l1446_144671


namespace NUMINAMATH_GPT_TV_cost_difference_l1446_144661

def cost_per_square_inch_difference :=
  let first_TV_width := 24
  let first_TV_height := 16
  let first_TV_original_cost_euros := 840
  let first_TV_discount_percent := 0.10
  let first_TV_tax_percent := 0.05
  let exchange_rate_first := 1.20
  let first_TV_area := first_TV_width * first_TV_height

  let discounted_price_first_TV := first_TV_original_cost_euros * (1 - first_TV_discount_percent)
  let total_cost_euros_first_TV := discounted_price_first_TV * (1 + first_TV_tax_percent)
  let total_cost_dollars_first_TV := total_cost_euros_first_TV * exchange_rate_first
  let cost_per_square_inch_first_TV := total_cost_dollars_first_TV / first_TV_area

  let new_TV_width := 48
  let new_TV_height := 32
  let new_TV_original_cost_dollars := 1800
  let new_TV_first_discount_percent := 0.20
  let new_TV_second_discount_percent := 0.15
  let new_TV_tax_percent := 0.08
  let new_TV_area := new_TV_width * new_TV_height

  let price_after_first_discount := new_TV_original_cost_dollars * (1 - new_TV_first_discount_percent)
  let price_after_second_discount := price_after_first_discount * (1 - new_TV_second_discount_percent)
  let total_cost_dollars_new_TV := price_after_second_discount * (1 + new_TV_tax_percent)
  let cost_per_square_inch_new_TV := total_cost_dollars_new_TV / new_TV_area

  let cost_difference_per_square_inch := cost_per_square_inch_first_TV - cost_per_square_inch_new_TV
  cost_difference_per_square_inch

theorem TV_cost_difference :
  cost_per_square_inch_difference = 1.62 := by
  sorry

end NUMINAMATH_GPT_TV_cost_difference_l1446_144661


namespace NUMINAMATH_GPT_relationship_among_three_numbers_l1446_144662

theorem relationship_among_three_numbers :
  let a := -0.3
  let b := 0.3^2
  let c := 2^0.3
  b < a ∧ a < c :=
by
  let a := -0.3
  let b := 0.3^2
  let c := 2^0.3
  sorry

end NUMINAMATH_GPT_relationship_among_three_numbers_l1446_144662


namespace NUMINAMATH_GPT_black_greater_than_gray_by_103_l1446_144649

def a := 12
def b := 9
def c := 7
def d := 3

def area (side: ℕ) := side * side

def black_area_sum : ℕ := area a + area c
def gray_area_sum : ℕ := area b + area d

theorem black_greater_than_gray_by_103 :
  black_area_sum - gray_area_sum = 103 := by
  sorry

end NUMINAMATH_GPT_black_greater_than_gray_by_103_l1446_144649


namespace NUMINAMATH_GPT_frustum_shortest_distance_l1446_144672

open Real

noncomputable def shortest_distance (R1 R2 : ℝ) (AB : ℝ) (string_from_midpoint : Bool) : ℝ :=
  if R1 = 5 ∧ R2 = 10 ∧ AB = 20 ∧ string_from_midpoint = true then 4 else 0

theorem frustum_shortest_distance : 
  shortest_distance 5 10 20 true = 4 :=
by sorry

end NUMINAMATH_GPT_frustum_shortest_distance_l1446_144672


namespace NUMINAMATH_GPT_Morse_code_distinct_symbols_count_l1446_144633

theorem Morse_code_distinct_symbols_count :
  let count (n : ℕ) := 2 ^ n
  count 1 + count 2 + count 3 + count 4 + count 5 = 62 :=
by
  sorry

end NUMINAMATH_GPT_Morse_code_distinct_symbols_count_l1446_144633


namespace NUMINAMATH_GPT_ratio_3_7_not_possible_l1446_144611

theorem ratio_3_7_not_possible (n : ℕ) (h : 30 < n ∧ n < 40) :
  ¬ (∃ k : ℕ, n = 10 * k) :=
by {
  sorry
}

end NUMINAMATH_GPT_ratio_3_7_not_possible_l1446_144611


namespace NUMINAMATH_GPT_icing_two_sides_on_Jack_cake_l1446_144634

noncomputable def Jack_cake_icing_two_sides (cake_size : ℕ) : ℕ :=
  let side_cubes := 4 * (cake_size - 2) * 3
  let vertical_edge_cubes := 4 * (cake_size - 2)
  side_cubes + vertical_edge_cubes

-- The statement to be proven
theorem icing_two_sides_on_Jack_cake : Jack_cake_icing_two_sides 5 = 96 :=
by
  sorry

end NUMINAMATH_GPT_icing_two_sides_on_Jack_cake_l1446_144634


namespace NUMINAMATH_GPT_vector_BC_l1446_144698

/-- Given points A (0,1), B (3,2) and vector AC (-4,-3), prove that BC = (-7, -4) -/
theorem vector_BC
  (A B : ℝ × ℝ)
  (AC : ℝ × ℝ)
  (hA : A = (0, 1))
  (hB : B = (3, 2))
  (hAC : AC = (-4, -3)) :
  (AC - (B - A)) = (-7, -4) :=
by
  sorry

end NUMINAMATH_GPT_vector_BC_l1446_144698


namespace NUMINAMATH_GPT_raft_travel_time_l1446_144688

noncomputable def downstream_speed (x y : ℝ) : ℝ := x + y
noncomputable def upstream_speed (x y : ℝ) : ℝ := x - y

theorem raft_travel_time {x y : ℝ} 
  (h1 : 7 * upstream_speed x y = 5 * downstream_speed x y) : (35 : ℝ) = (downstream_speed x y) * 7 / 4 := by sorry

end NUMINAMATH_GPT_raft_travel_time_l1446_144688


namespace NUMINAMATH_GPT_infinite_hexagons_exist_l1446_144665

theorem infinite_hexagons_exist :
  ∃ (a1 a2 a3 a4 a5 a6 : ℤ), 
  (a1 + a2 + a3 + a4 + a5 + a6 = 20) ∧
  (a1 ≤ a2) ∧ (a1 + a2 ≤ a3) ∧ (a2 + a3 ≤ a4) ∧
  (a3 + a4 ≤ a5) ∧ (a4 + a5 ≤ a6) ∧ (a1 + a2 + a3 + a4 + a5 > a6) :=
sorry

end NUMINAMATH_GPT_infinite_hexagons_exist_l1446_144665


namespace NUMINAMATH_GPT_base_eight_to_ten_l1446_144635

theorem base_eight_to_ten (n : Nat) (h : n = 52) : 8 * 5 + 2 = 42 :=
by
  -- Proof will be written here.
  sorry

end NUMINAMATH_GPT_base_eight_to_ten_l1446_144635


namespace NUMINAMATH_GPT_charlie_and_dana_proof_l1446_144681

noncomputable def charlie_and_dana_ways 
    (cookies : ℕ) (smoothies : ℕ) (total_items : ℕ) 
    (distinct_charlie : ℕ) 
    (repeatable_dana : ℕ) : ℕ :=
    if cookies = 8 ∧ smoothies = 5 ∧ total_items = 5 ∧ distinct_charlie = 0 
       ∧ repeatable_dana = 0 then 27330 else 0

theorem charlie_and_dana_proof :
  charlie_and_dana_ways 8 5 5 0 0 = 27330 := 
  sorry

end NUMINAMATH_GPT_charlie_and_dana_proof_l1446_144681


namespace NUMINAMATH_GPT_min_value_quadratic_l1446_144641

theorem min_value_quadratic :
  ∀ (x : ℝ), (2 * x^2 - 8 * x + 15) ≥ 7 :=
by
  -- We need to show that 2x^2 - 8x + 15 has a minimum value of 7
  sorry

end NUMINAMATH_GPT_min_value_quadratic_l1446_144641


namespace NUMINAMATH_GPT_triangle_inequality_inequality_l1446_144626

-- Define a helper function to describe the triangle inequality
def triangle_inequality (a b c : ℝ) : Prop :=
a + b > c ∧ a + c > b ∧ b + c > a

-- Define the main statement
theorem triangle_inequality_inequality (a b c : ℝ) (h_triangle : triangle_inequality a b c):
  a * (b - c) ^ 2 + b * (c - a) ^ 2 + c * (a - b) ^ 2 + 4 * a * b * c > a ^ 3 + b ^ 3 + c ^ 3 :=
sorry

end NUMINAMATH_GPT_triangle_inequality_inequality_l1446_144626


namespace NUMINAMATH_GPT_abc_sum_is_12_l1446_144636

theorem abc_sum_is_12
  (a b c : ℕ)
  (h : 28 * a + 30 * b + 31 * c = 365) :
  a + b + c = 12 :=
by
  sorry

end NUMINAMATH_GPT_abc_sum_is_12_l1446_144636


namespace NUMINAMATH_GPT_valid_parameterizations_l1446_144678

noncomputable def line_equation (x y : ℝ) : Prop := y = (5/3) * x + 1

def parametrize_A (t : ℝ) : Prop :=
  ∃ (x y : ℝ), (x, y) = (3 + t * 3, 6 + t * 5) ∧ line_equation x y

def parametrize_D (t : ℝ) : Prop :=
  ∃ (x y : ℝ), (x, y) = (-1 + t * 3, -2/3 + t * 5) ∧ line_equation x y

theorem valid_parameterizations : parametrize_A t ∧ parametrize_D t :=
by
  -- Proof steps are skipped
  sorry

end NUMINAMATH_GPT_valid_parameterizations_l1446_144678


namespace NUMINAMATH_GPT_part1_part2_l1446_144652

noncomputable def f (a x : ℝ) := a * x^2 - (a + 1) * x + 1

theorem part1 (a : ℝ) (h1 : a ≠ 0) :
  (∀ x : ℝ, f a x ≤ 2) ↔ (-3 - 2 * Real.sqrt 2 ≤ a ∧ a ≤ -3 + 2 * Real.sqrt 2) :=
sorry

theorem part2 (a : ℝ) (h1 : a ≠ 0) (x : ℝ) :
  (f a x < 0) ↔
    ((0 < a ∧ a < 1 ∧ 1 < x ∧ x < 1 / a) ∨
     (a = 1 ∧ false) ∨
     (a > 1 ∧ 1 / a < x ∧ x < 1) ∨
     (a < 0 ∧ (x < 1 / a ∨ x > 1))) :=
sorry

end NUMINAMATH_GPT_part1_part2_l1446_144652


namespace NUMINAMATH_GPT_Andrew_is_19_l1446_144669

-- Define individuals and their relationships
def Andrew_age (Bella_age : ℕ) : ℕ := Bella_age - 5
def Bella_age (Carlos_age : ℕ) : ℕ := Carlos_age + 4
def Carlos_age : ℕ := 20

-- Formulate the problem statement
theorem Andrew_is_19 : Andrew_age (Bella_age Carlos_age) = 19 :=
by
  sorry

end NUMINAMATH_GPT_Andrew_is_19_l1446_144669


namespace NUMINAMATH_GPT_swan_count_l1446_144660

theorem swan_count (total_birds : ℕ) (fraction_ducks : ℚ):
  fraction_ducks = 5 / 6 →
  total_birds = 108 →
  ∃ (num_swans : ℕ), num_swans = 18 :=
by
  intro h_fraction_ducks h_total_birds
  sorry

end NUMINAMATH_GPT_swan_count_l1446_144660


namespace NUMINAMATH_GPT_unit_circle_sector_arc_length_l1446_144631

theorem unit_circle_sector_arc_length (r S l : ℝ) (h1 : r = 1) (h2 : S = 1) (h3 : S = 1 / 2 * l * r) : l = 2 :=
by
  sorry

end NUMINAMATH_GPT_unit_circle_sector_arc_length_l1446_144631


namespace NUMINAMATH_GPT_point_B_in_first_quadrant_l1446_144679

def is_first_quadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 > 0

theorem point_B_in_first_quadrant : is_first_quadrant (1, 2) :=
by
  sorry

end NUMINAMATH_GPT_point_B_in_first_quadrant_l1446_144679


namespace NUMINAMATH_GPT_total_distance_from_A_through_B_to_C_l1446_144675

noncomputable def distance_A_B_map : ℝ := 120
noncomputable def distance_B_C_map : ℝ := 70
noncomputable def map_scale : ℝ := 10 -- km per cm

noncomputable def distance_A_B := distance_A_B_map * map_scale -- Distance from City A to City B in km
noncomputable def distance_B_C := distance_B_C_map * map_scale -- Distance from City B to City C in km
noncomputable def total_distance := distance_A_B + distance_B_C -- Total distance in km

theorem total_distance_from_A_through_B_to_C :
  total_distance = 1900 := by
  sorry

end NUMINAMATH_GPT_total_distance_from_A_through_B_to_C_l1446_144675


namespace NUMINAMATH_GPT_total_cost_first_3_years_l1446_144637

def monthly_fee : ℕ := 12
def down_payment : ℕ := 50
def years : ℕ := 3

theorem total_cost_first_3_years :
  (years * 12 * monthly_fee + down_payment) = 482 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_first_3_years_l1446_144637


namespace NUMINAMATH_GPT_total_surface_area_of_cuboid_l1446_144663

variables (l w h : ℝ)
variables (lw_area wh_area lh_area : ℝ)

def box_conditions :=
  lw_area = l * w ∧
  wh_area = w * h ∧
  lh_area = l * h

theorem total_surface_area_of_cuboid (hc : box_conditions l w h 120 72 60) :
  2 * (120 + 72 + 60) = 504 :=
sorry

end NUMINAMATH_GPT_total_surface_area_of_cuboid_l1446_144663


namespace NUMINAMATH_GPT_charlie_pennies_l1446_144614

variable (a c : ℕ)

theorem charlie_pennies (h1 : c + 1 = 4 * (a - 1)) (h2 : c - 1 = 3 * (a + 1)) : c = 31 := 
by
  sorry

end NUMINAMATH_GPT_charlie_pennies_l1446_144614


namespace NUMINAMATH_GPT_area_of_dodecagon_l1446_144657

theorem area_of_dodecagon (r : ℝ) : 
  ∃ A : ℝ, (∃ n : ℕ, n = 12) ∧ (A = 3 * r^2) := 
by
  sorry

end NUMINAMATH_GPT_area_of_dodecagon_l1446_144657


namespace NUMINAMATH_GPT_inverse_f_neg_3_l1446_144628

def f (x : ℝ) : ℝ := 5 - 2 * x

theorem inverse_f_neg_3 : (∃ x : ℝ, f x = -3) ∧ (f 4 = -3) :=
by
  sorry

end NUMINAMATH_GPT_inverse_f_neg_3_l1446_144628


namespace NUMINAMATH_GPT_find_theta_l1446_144659

theorem find_theta
  (θ : ℝ)
  (hθ : 0 ≤ θ ∧ θ < 2 * Real.pi)
  (ha : ∃ k, (2 * Real.cos θ, 2 * Real.sin θ) = (k * 3, k * Real.sqrt 3)) :
  θ = Real.pi / 6 ∨ θ = 7 * Real.pi / 6 :=
by
  sorry

end NUMINAMATH_GPT_find_theta_l1446_144659


namespace NUMINAMATH_GPT_radius_moon_scientific_notation_l1446_144608

def scientific_notation := 1738000 = 1.738 * 10^6

theorem radius_moon_scientific_notation : scientific_notation := 
sorry

end NUMINAMATH_GPT_radius_moon_scientific_notation_l1446_144608


namespace NUMINAMATH_GPT_cos_sum_to_product_l1446_144687

theorem cos_sum_to_product (x : ℝ) : 
  (∃ a b c d : ℕ, a * Real.cos (b * x) * Real.cos (c * x) * Real.cos (d * x) =
  Real.cos (2 * x) + Real.cos (6 * x) + Real.cos (10 * x) + Real.cos (14 * x) 
  ∧ a + b + c + d = 18) :=
sorry

end NUMINAMATH_GPT_cos_sum_to_product_l1446_144687


namespace NUMINAMATH_GPT_real_root_exists_l1446_144697

theorem real_root_exists (a b c : ℝ) :
  (∃ x : ℝ, x^2 + (a - b) * x + (b - c) = 0) ∨ 
  (∃ x : ℝ, x^2 + (b - c) * x + (c - a) = 0) ∨ 
  (∃ x : ℝ, x^2 + (c - a) * x + (a - b) = 0) :=
by {
  sorry
}

end NUMINAMATH_GPT_real_root_exists_l1446_144697


namespace NUMINAMATH_GPT_find_y_from_eqns_l1446_144642

theorem find_y_from_eqns (x y : ℝ) (h1 : x^2 - x + 6 = y + 2) (h2 : x = -5) : y = 34 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_y_from_eqns_l1446_144642


namespace NUMINAMATH_GPT_hexagon_inequality_l1446_144664

variable {Point : Type*} [MetricSpace Point]

-- Define points A1, A2, A3, A4, A5, A6 in a Metric Space
variables (A1 A2 A3 A4 A5 A6 O : Point)

-- Conditions
def angle_condition (O A1 A2 A3 A4 A5 A6 : Point) : Prop :=
  -- Points form a hexagon where each side is visible from O at 60 degrees
  -- We assume MetricSpace has a function measuring angles such as angle O x y = 60
  true -- A simplified condition; the actual angle measurement needs more geometry setup

def distance_condition_odd (O A1 A3 A5 : Point) : Prop := dist O A1 > dist O A3 ∧ dist O A3 > dist O A5
def distance_condition_even (O A2 A4 A6 : Point) : Prop := dist O A2 > dist O A4 ∧ dist O A4 > dist O A6

-- Question to prove
theorem hexagon_inequality 
  (hc : angle_condition O A1 A2 A3 A4 A5 A6) 
  (ho : distance_condition_odd O A1 A3 A5)
  (he : distance_condition_even O A2 A4 A6) : 
  dist A1 A2 + dist A3 A4 + dist A5 A6 < dist A2 A3 + dist A4 A5 + dist A6 A1 := 
sorry

end NUMINAMATH_GPT_hexagon_inequality_l1446_144664


namespace NUMINAMATH_GPT_min_value_of_quadratic_l1446_144613

theorem min_value_of_quadratic :
  ∃ x : ℝ, (∀ y : ℝ, y = x^2 - 8 * x + 15 → y ≥ -1) ∧ (∃ x₀ : ℝ, x₀ = 4 ∧ (x₀^2 - 8 * x₀ + 15 = -1)) :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_quadratic_l1446_144613


namespace NUMINAMATH_GPT_probability_sum_18_two_12_sided_dice_l1446_144692

theorem probability_sum_18_two_12_sided_dice :
  let total_outcomes := 12 * 12
  let successful_outcomes := 7
  successful_outcomes / total_outcomes = 7 / 144 := by
sorry

end NUMINAMATH_GPT_probability_sum_18_two_12_sided_dice_l1446_144692


namespace NUMINAMATH_GPT_square_of_product_of_third_sides_l1446_144691

-- Given data for triangles P1 and P2
variables {a b c d : ℝ}

-- Areas of triangles P1 and P2
def area_P1_pos (a b : ℝ) : Prop := a * b / 2 = 3
def area_P2_pos (a d : ℝ) : Prop := a * d / 2 = 6

-- Condition that b = d / 2
def side_ratio (b d : ℝ) : Prop := b = d / 2

-- Pythagorean theorem applied to both triangles
def pythagorean_P1 (a b c : ℝ) : Prop := a^2 + b^2 = c^2
def pythagorean_P2 (a d c : ℝ) : Prop := a^2 + d^2 = c^2

-- The goal is to prove (cd)^2 = 120
theorem square_of_product_of_third_sides (a b c d : ℝ)
  (h_area_P1: area_P1_pos a b) 
  (h_area_P2: area_P2_pos a d) 
  (h_side_ratio: side_ratio b d) 
  (h_pythagorean_P1: pythagorean_P1 a b c) 
  (h_pythagorean_P2: pythagorean_P2 a d c) :
  (c * d)^2 = 120 := 
sorry

end NUMINAMATH_GPT_square_of_product_of_third_sides_l1446_144691


namespace NUMINAMATH_GPT_range_of_a_l1446_144618

noncomputable def A := { x : ℝ | 0 < x ∧ x < 2 }
noncomputable def B (a : ℝ) := { x : ℝ | 0 < x ∧ x < (2 / a) }

theorem range_of_a (a : ℝ) (h : 0 < a) : (A ∩ (B a)) = A → 0 < a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_GPT_range_of_a_l1446_144618


namespace NUMINAMATH_GPT_coverable_hook_l1446_144648

def is_coverable (m n : ℕ) : Prop :=
  ∃ a b : ℕ, (m = 3 * a ∧ n = 4 * b) ∨ (m = 12 * a ∧ n = b ∧ b ≠ 1 ∧ b ≠ 2 ∧ b ≠ 5)

theorem coverable_hook (m n : ℕ) : (∃ a b : ℕ, (m = 3 * a ∧ n = 4 * b) ∨ (m = 12 * a ∧ n = b ∧ b ≠ 1 ∧ b ≠ 2 ∧ b ≠ 5))
  ↔ is_coverable m n :=
by
  sorry

end NUMINAMATH_GPT_coverable_hook_l1446_144648


namespace NUMINAMATH_GPT_decreasing_condition_l1446_144606

variable (m : ℝ)

def quadratic_fn (x : ℝ) : ℝ := x^2 + m * x + 1

theorem decreasing_condition (h : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 5 → (deriv (quadratic_fn m) x ≤ 0)) :
    m ≤ -10 := 
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_decreasing_condition_l1446_144606


namespace NUMINAMATH_GPT_gcd_ab_is_22_l1446_144645

def a : ℕ := 198
def b : ℕ := 308

theorem gcd_ab_is_22 : Nat.gcd a b = 22 := 
by { sorry }

end NUMINAMATH_GPT_gcd_ab_is_22_l1446_144645


namespace NUMINAMATH_GPT_range_of_a4_l1446_144621

noncomputable def geometric_sequence (a1 a2 a3 : ℝ) (q : ℝ) (a4 : ℝ) : Prop :=
  ∃ (a1 q : ℝ), 0 < a1 ∧ a1 < 1 ∧ 
                1 < a1 * q ∧ a1 * q < 2 ∧ 
                2 < a1 * q^2 ∧ a1 * q^2 < 4 ∧ 
                a4 = (a1 * q^2) * q ∧ 
                2 * Real.sqrt 2 < a4 ∧ a4 < 16

theorem range_of_a4 (a1 a2 a3 a4 : ℝ) (q : ℝ) (h1 : 0 < a1) (h2 : a1 < 1) 
  (h3 : 1 < a2) (h4 : a2 < 2) (h5 : a2 = a1 * q)
  (h6 : 2 < a3) (h7 : a3 < 4) (h8 : a3 = a1 * q^2) :
  2 * Real.sqrt 2 < a4 ∧ a4 < 16 :=
by
  have hq1 : 2 * q^2 < 1 := sorry    -- Placeholder for necessary inequalities
  have hq2: 1 < q ∧ q < 4 := sorry   -- Placeholder for necessary inequalities
  sorry

end NUMINAMATH_GPT_range_of_a4_l1446_144621


namespace NUMINAMATH_GPT_integer_solution_for_x_l1446_144656

theorem integer_solution_for_x (x : ℤ) : 
  (∃ y z : ℤ, x = 7 * y + 3 ∧ x = 5 * z + 2) ↔ 
  (∃ t : ℤ, x = 35 * t + 17) :=
by
  sorry

end NUMINAMATH_GPT_integer_solution_for_x_l1446_144656


namespace NUMINAMATH_GPT_units_digit_product_l1446_144684

theorem units_digit_product : (3^5 * 2^3) % 10 = 4 := 
sorry

end NUMINAMATH_GPT_units_digit_product_l1446_144684


namespace NUMINAMATH_GPT_compound_interest_rate_l1446_144699

theorem compound_interest_rate
  (P : ℝ)  -- Principal amount
  (r : ℝ)  -- Annual interest rate in decimal
  (A2 A3 : ℝ)  -- Amounts after 2 and 3 years
  (h2 : A2 = P * (1 + r)^2)
  (h3 : A3 = P * (1 + r)^3) :
  A2 = 17640 → A3 = 22932 → r = 0.3 := by
  sorry

end NUMINAMATH_GPT_compound_interest_rate_l1446_144699


namespace NUMINAMATH_GPT_fraction_power_four_l1446_144604

theorem fraction_power_four :
  (5 / 6) ^ 4 = 625 / 1296 :=
by sorry

end NUMINAMATH_GPT_fraction_power_four_l1446_144604


namespace NUMINAMATH_GPT_ratio_equality_l1446_144620

theorem ratio_equality (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z)
  (h4 : x ≠ y) (h5 : y ≠ z) (h6 : z ≠ x)
  (h7 : (y + 1) / (x + z) = (x + y + 2) / (z + 1))
  (h8 : (x + 1) / y = (y + 1) / (x + z)) :
  (x + 1) / y = 1 :=
by
  sorry

end NUMINAMATH_GPT_ratio_equality_l1446_144620


namespace NUMINAMATH_GPT_total_pieces_10_rows_l1446_144677

-- Define the conditions for the rods
def rod_seq (n : ℕ) : ℕ := 3 * n

-- Define the sum of the arithmetic sequence for rods
def sum_rods (n : ℕ) : ℕ := 3 * (n * (n + 1)) / 2

-- Define the conditions for the connectors
def connector_seq (n : ℕ) : ℕ := n + 1

-- Define the sum of the arithmetic sequence for connectors
def sum_connectors (n : ℕ) : ℕ := (n * (n + 1)) / 2

-- Define the total pieces calculation
def total_pieces (n : ℕ) : ℕ := sum_rods n + sum_connectors (n + 1)

-- The target statement
theorem total_pieces_10_rows : total_pieces 10 = 231 :=
by
  sorry

end NUMINAMATH_GPT_total_pieces_10_rows_l1446_144677


namespace NUMINAMATH_GPT_negation_exists_x_squared_leq_abs_x_l1446_144686

theorem negation_exists_x_squared_leq_abs_x :
  (¬ ∃ x : ℝ, x ∈ Set.Ioo (-1 : ℝ) (0 : ℝ) ∧ x^2 ≤ |x|) ↔ (∀ x : ℝ, x ∈ Set.Ioo (-1 : ℝ) (0 : ℝ) → x^2 > |x|) :=
by
  sorry

end NUMINAMATH_GPT_negation_exists_x_squared_leq_abs_x_l1446_144686


namespace NUMINAMATH_GPT_records_given_l1446_144603

theorem records_given (X : ℕ) (started_with : ℕ) (bought : ℕ) (days_per_record : ℕ) (total_days : ℕ)
  (h1 : started_with = 8) (h2 : bought = 30) (h3 : days_per_record = 2) (h4 : total_days = 100) :
  X = 12 := by
  sorry

end NUMINAMATH_GPT_records_given_l1446_144603


namespace NUMINAMATH_GPT_apples_given_by_Susan_l1446_144667

theorem apples_given_by_Susan (x y final_apples : ℕ) (h1 : y = 9) (h2 : final_apples = 17) (h3: final_apples = y + x) : x = 8 := by
  sorry

end NUMINAMATH_GPT_apples_given_by_Susan_l1446_144667


namespace NUMINAMATH_GPT_tom_seashells_l1446_144602

theorem tom_seashells (days_at_beach : ℕ) (seashells_per_day : ℕ) (total_seashells : ℕ) 
  (h1 : days_at_beach = 5) (h2 : seashells_per_day = 7) : total_seashells = 35 := 
by 
  sorry

end NUMINAMATH_GPT_tom_seashells_l1446_144602


namespace NUMINAMATH_GPT_total_collisions_100_balls_l1446_144683

def num_of_collisions (n: ℕ) : ℕ :=
  n * (n - 1) / 2

theorem total_collisions_100_balls :
  num_of_collisions 100 = 4950 :=
by
  sorry

end NUMINAMATH_GPT_total_collisions_100_balls_l1446_144683


namespace NUMINAMATH_GPT_max_a_for_three_solutions_l1446_144610

-- Define the equation as a Lean function
def equation (x a : ℝ) : ℝ :=
  (|x-2| + 2 * a)^2 - 3 * (|x-2| + 2 * a) + 4 * a * (3 - 4 * a)

-- Statement of the proof problem
theorem max_a_for_three_solutions :
  (∃ (a : ℝ), (∀ x : ℝ, equation x a = 0) ∧
  (∀ (b : ℝ), (∀ x : ℝ, equation x b = 0) → b ≤ 0.5)) :=
sorry

end NUMINAMATH_GPT_max_a_for_three_solutions_l1446_144610


namespace NUMINAMATH_GPT_henry_games_total_l1446_144623

theorem henry_games_total
    (wins : ℕ)
    (losses : ℕ)
    (draws : ℕ)
    (hw : wins = 2)
    (hl : losses = 2)
    (hd : draws = 10) :
  wins + losses + draws = 14 :=
by
  -- The proof is omitted.
  sorry

end NUMINAMATH_GPT_henry_games_total_l1446_144623


namespace NUMINAMATH_GPT_trigonometric_comparison_l1446_144676

noncomputable def a : ℝ := 2 * Real.sin (13 * Real.pi / 180) * Real.cos (13 * Real.pi / 180)
noncomputable def b : ℝ := 2 * Real.tan (76 * Real.pi / 180) / (1 + Real.tan (76 * Real.pi / 180)^2)
noncomputable def c : ℝ := Real.sqrt ((1 - Real.cos (50 * Real.pi / 180)) / 2)

theorem trigonometric_comparison : b > a ∧ a > c := by
  sorry

end NUMINAMATH_GPT_trigonometric_comparison_l1446_144676


namespace NUMINAMATH_GPT_find_asymptote_slope_l1446_144647

theorem find_asymptote_slope :
  (∀ x y : ℝ, (x^2 / 144 - y^2 / 81 = 0) → (y = 3/4 * x ∨ y = -3/4 * x)) :=
by
  sorry

end NUMINAMATH_GPT_find_asymptote_slope_l1446_144647


namespace NUMINAMATH_GPT_alix_has_15_more_chocolates_than_nick_l1446_144622

-- Definitions based on the problem conditions
def nick_chocolates : ℕ := 10
def alix_initial_chocolates : ℕ := 3 * nick_chocolates
def chocolates_taken_by_mom : ℕ := 5
def alix_chocolates_after_mom_took_some : ℕ := alix_initial_chocolates - chocolates_taken_by_mom

-- Statement of the theorem to prove
theorem alix_has_15_more_chocolates_than_nick :
  alix_chocolates_after_mom_took_some - nick_chocolates = 15 :=
sorry

end NUMINAMATH_GPT_alix_has_15_more_chocolates_than_nick_l1446_144622


namespace NUMINAMATH_GPT_product_of_m_and_u_l1446_144650

noncomputable def g : ℝ → ℝ := sorry

axiom g_conditions : (∀ x y : ℝ, g (x^2 - y^2) = (x - y) * ((g x) ^ 3 + (g y) ^ 3)) ∧ (g 1 = 1)

def m : ℕ := sorry
def u : ℝ := sorry

theorem product_of_m_and_u : m * u = 3 :=
by 
  -- all conditions about 'g' are assumed as axioms and not directly included in the proof steps
  exact sorry

end NUMINAMATH_GPT_product_of_m_and_u_l1446_144650


namespace NUMINAMATH_GPT_fractional_equation_solution_l1446_144638

theorem fractional_equation_solution (m : ℝ) (x : ℝ) :
  (m + 3) / (x - 1) = 1 → x > 0 → m > -4 ∧ m ≠ -3 :=
by
  sorry

end NUMINAMATH_GPT_fractional_equation_solution_l1446_144638


namespace NUMINAMATH_GPT_math_problem_l1446_144601

variable (x y : ℝ)

theorem math_problem (h1 : x^2 - 3 * x * y + 2 * y^2 + x - y = 0) (h2 : x^2 - 2 * x * y + y^2 - 5 * x + 7 * y = 0) :
  x * y - 12 * x + 15 * y = 0 :=
  sorry

end NUMINAMATH_GPT_math_problem_l1446_144601


namespace NUMINAMATH_GPT_minuend_is_not_integer_l1446_144689

theorem minuend_is_not_integer (M S D : ℚ) (h1 : M + S + D = 555) (h2 : M - S = D) : ¬ ∃ n : ℤ, M = n := 
by
  sorry

end NUMINAMATH_GPT_minuend_is_not_integer_l1446_144689


namespace NUMINAMATH_GPT_height_percentage_difference_l1446_144674

theorem height_percentage_difference (A B : ℝ) (h : B = A * (4/3)) : 
  (A * (1/3) / B) * 100 = 25 := by
  sorry

end NUMINAMATH_GPT_height_percentage_difference_l1446_144674


namespace NUMINAMATH_GPT_distinct_sums_is_98_l1446_144630

def arithmetic_sequence_distinct_sums (a_n : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ) :=
  (∀ n : ℕ, S n = (n * (2 * a_n 0 + (n - 1) * d)) / 2) ∧
  S 5 = 0 ∧
  d ≠ 0 →
  (∃ distinct_count : ℕ, distinct_count = 98 ∧
   ∀ i j : ℕ, 1 ≤ i ∧ i ≤ 100 ∧ 1 ≤ j ∧ j ≤ 100 ∧ S i = S j → i = j)

theorem distinct_sums_is_98 (a_n : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ) (h : arithmetic_sequence_distinct_sums a_n S d) :
  ∃ distinct_count : ℕ, distinct_count = 98 :=
sorry

end NUMINAMATH_GPT_distinct_sums_is_98_l1446_144630


namespace NUMINAMATH_GPT_flower_stones_per_bracelet_l1446_144685

theorem flower_stones_per_bracelet (total_stones : ℝ) (bracelets : ℝ)  (H_total: total_stones = 88.0) (H_bracelets: bracelets = 8.0) :
  (total_stones / bracelets = 11.0) :=
by
  rw [H_total, H_bracelets]
  norm_num

end NUMINAMATH_GPT_flower_stones_per_bracelet_l1446_144685


namespace NUMINAMATH_GPT_perfect_square_mod_3_l1446_144651

theorem perfect_square_mod_3 (k : ℤ) (hk : ∃ m : ℤ, k = m^2) : k % 3 = 0 ∨ k % 3 = 1 :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_mod_3_l1446_144651
