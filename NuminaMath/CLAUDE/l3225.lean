import Mathlib

namespace NUMINAMATH_CALUDE_triangle_cosine_sum_less_than_two_l3225_322530

theorem triangle_cosine_sum_less_than_two (α β γ : Real) 
  (h_triangle : α + β + γ = Real.pi) 
  (h_positive : 0 < α ∧ 0 < β ∧ 0 < γ) : 
  Real.cos α + Real.cos β + Real.cos γ < 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_sum_less_than_two_l3225_322530


namespace NUMINAMATH_CALUDE_num_arrangements_equals_5040_l3225_322593

/-- The number of candidates --/
def n : ℕ := 8

/-- The number of volunteers to be selected --/
def k : ℕ := 5

/-- The number of days --/
def days : ℕ := 5

/-- Function to calculate the number of arrangements --/
def num_arrangements (n k : ℕ) : ℕ :=
  let only_one := 2 * (n - 2).choose (k - 1) * k.factorial
  let both := (n - 2).choose (k - 2) * (k - 2).factorial * 2 * (k - 1)
  only_one + both

/-- Theorem stating the number of arrangements --/
theorem num_arrangements_equals_5040 :
  num_arrangements n k = 5040 := by sorry

end NUMINAMATH_CALUDE_num_arrangements_equals_5040_l3225_322593


namespace NUMINAMATH_CALUDE_equation_solution_l3225_322524

theorem equation_solution :
  let f (x : ℝ) := x * ((6 - x) / (x + 1)) * ((6 - x) / (x + 1) + x)
  ∀ x : ℝ, f x = 8 ↔ x = 2 + Real.sqrt 2 ∨ x = 2 - Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3225_322524


namespace NUMINAMATH_CALUDE_product_17_reciprocal_squares_sum_l3225_322521

theorem product_17_reciprocal_squares_sum :
  ∀ a b : ℕ+, 
  (a * b : ℕ+) = 17 →
  (1 : ℚ) / (a * a : ℚ) + (1 : ℚ) / (b * b : ℚ) = 290 / 289 := by
  sorry

end NUMINAMATH_CALUDE_product_17_reciprocal_squares_sum_l3225_322521


namespace NUMINAMATH_CALUDE_power_of_power_l3225_322579

theorem power_of_power (a : ℝ) : (a^5)^3 = a^15 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l3225_322579


namespace NUMINAMATH_CALUDE_line_symmetry_l3225_322559

/-- Given two lines in the plane and a point, this theorem states that
    the lines are symmetric about the point. -/
theorem line_symmetry (x y : ℝ) : 
  (2 * x + 3 * y - 6 = 0) → 
  (2 * ((2 : ℝ) - x) + 3 * ((2 : ℝ) - y) - 6 = 0) →
  (2 * x + 3 * y - 4 = 0) := by
  sorry

end NUMINAMATH_CALUDE_line_symmetry_l3225_322559


namespace NUMINAMATH_CALUDE_average_of_w_and_x_l3225_322500

theorem average_of_w_and_x (w x y : ℝ) 
  (h1 : 2 / w + 2 / x = 2 / y) 
  (h2 : w * x = y) : 
  (w + x) / 2 = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_average_of_w_and_x_l3225_322500


namespace NUMINAMATH_CALUDE_sqrt_20_minus_1_range_l3225_322532

theorem sqrt_20_minus_1_range : 3 < Real.sqrt 20 - 1 ∧ Real.sqrt 20 - 1 < 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_20_minus_1_range_l3225_322532


namespace NUMINAMATH_CALUDE_book_weight_l3225_322501

theorem book_weight (num_books : ℕ) (total_weight : ℝ) (bag_weight : ℝ) :
  num_books = 14 →
  total_weight = 11.14 →
  bag_weight = 0.5 →
  (total_weight - bag_weight) / num_books = 0.76 := by
  sorry

end NUMINAMATH_CALUDE_book_weight_l3225_322501


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3225_322576

def A : Set ℝ := {x | x < -1 ∨ (2 ≤ x ∧ x < 3)}
def B : Set ℝ := {x | -2 ≤ x ∧ x < 4}

theorem union_of_A_and_B : A ∪ B = {x : ℝ | x < 4} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3225_322576


namespace NUMINAMATH_CALUDE_factorization_equality_l3225_322540

theorem factorization_equality (a b : ℝ) : 6 * a^2 * b - 3 * a * b = 3 * a * b * (2 * a - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3225_322540


namespace NUMINAMATH_CALUDE_exists_quadratic_with_2n_roots_l3225_322517

/-- Definition of function iteration -/
def iterate (f : ℝ → ℝ) : ℕ → (ℝ → ℝ)
  | 0 => id
  | n + 1 => f ∘ (iterate f n)

/-- A quadratic polynomial -/
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- Theorem stating the existence of a quadratic polynomial with the desired property -/
theorem exists_quadratic_with_2n_roots :
  ∃ (a b c : ℝ), ∀ (n : ℕ), n > 0 →
    (∃ (roots : Finset ℝ), roots.card = 2^n ∧
      (∀ x : ℝ, x ∈ roots ↔ iterate (quadratic a b c) n x = 0) ∧
      (∀ x y : ℝ, x ∈ roots → y ∈ roots → x ≠ y → x ≠ y)) :=
sorry

end NUMINAMATH_CALUDE_exists_quadratic_with_2n_roots_l3225_322517


namespace NUMINAMATH_CALUDE_range_of_p_l3225_322528

open Set

noncomputable def f (x : ℝ) : ℝ := x^3 - x^2 - 10*x

def A : Set ℝ := {x | (deriv f) x ≤ 0}

def B (p : ℝ) : Set ℝ := {x | p + 1 ≤ x ∧ x ≤ 2*p - 1}

theorem range_of_p (p : ℝ) (h : A ∪ B p = A) : p ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_p_l3225_322528


namespace NUMINAMATH_CALUDE_adjacent_even_sum_l3225_322510

/-- A circular arrangement of seven natural numbers -/
def CircularArrangement := Fin 7 → ℕ

/-- Two numbers in a circular arrangement are adjacent if their indices differ by 1 (mod 7) -/
def adjacent (arr : CircularArrangement) (i j : Fin 7) : Prop :=
  (i.val + 1) % 7 = j.val ∨ (j.val + 1) % 7 = i.val

/-- The main theorem: In any circular arrangement of seven natural numbers,
    there exist two adjacent numbers whose sum is even -/
theorem adjacent_even_sum (arr : CircularArrangement) :
  ∃ (i j : Fin 7), adjacent arr i j ∧ Even (arr i + arr j) := by
  sorry

end NUMINAMATH_CALUDE_adjacent_even_sum_l3225_322510


namespace NUMINAMATH_CALUDE_largest_minus_smallest_is_52_l3225_322546

def digits : Finset Nat := {8, 3, 4, 6}

def is_two_digit (n : Nat) : Prop := 10 ≤ n ∧ n ≤ 99

def valid_number (n : Nat) : Prop :=
  is_two_digit n ∧ ∃ (a b : Nat), a ∈ digits ∧ b ∈ digits ∧ a ≠ b ∧ n = 10 * a + b

theorem largest_minus_smallest_is_52 :
  ∃ (max min : Nat),
    valid_number max ∧
    valid_number min ∧
    (∀ n, valid_number n → n ≤ max) ∧
    (∀ n, valid_number n → min ≤ n) ∧
    max - min = 52 := by
  sorry

end NUMINAMATH_CALUDE_largest_minus_smallest_is_52_l3225_322546


namespace NUMINAMATH_CALUDE_lower_limit_of_b_l3225_322582

theorem lower_limit_of_b (a b : ℤ) (h1 : 8 < a ∧ a < 15) (h2 : b < 21) 
  (h3 : (14 : ℚ) / b - (9 : ℚ) / b = (155 : ℚ) / 100) : 4 ≤ b := by
  sorry

end NUMINAMATH_CALUDE_lower_limit_of_b_l3225_322582


namespace NUMINAMATH_CALUDE_binary_multiplication_l3225_322507

/-- Converts a list of binary digits to a natural number. -/
def binaryToNat (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Converts a natural number to a list of binary digits. -/
def natToBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
    let rec toBinary (m : ℕ) : List Bool :=
      if m = 0 then [] else (m % 2 = 1) :: toBinary (m / 2)
    toBinary n

theorem binary_multiplication :
  let a := [true, false, true, true]  -- 1101₂
  let b := [true, true, true]         -- 111₂
  let c := [true, true, true, true, false, false, true]  -- 1001111₂
  binaryToNat a * binaryToNat b = binaryToNat c := by
sorry

end NUMINAMATH_CALUDE_binary_multiplication_l3225_322507


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3225_322592

/-- Given a > 0 and a ≠ 1, if f(x) = ax is decreasing on ℝ, then g(x) = (2-a)x³ is increasing on ℝ, 
    but the converse is not always true. -/
theorem sufficient_but_not_necessary (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x y : ℝ, x < y → a * x < a * y) →
  (∀ x y : ℝ, x < y → (2 - a) * x^3 < (2 - a) * y^3) ∧
  ¬(∀ x y : ℝ, x < y → (2 - a) * x^3 < (2 - a) * y^3 → a * x < a * y) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3225_322592


namespace NUMINAMATH_CALUDE_total_litter_weight_l3225_322529

/-- The amount of litter collected by Gina and her neighborhood --/
def litterCollection (gina_bags : ℕ) (neighborhood_multiplier : ℕ) (weight_per_bag : ℕ) : ℕ :=
  let total_bags := gina_bags + gina_bags * neighborhood_multiplier
  total_bags * weight_per_bag

/-- Theorem stating the total weight of litter collected --/
theorem total_litter_weight :
  litterCollection 2 82 4 = 664 := by
  sorry

end NUMINAMATH_CALUDE_total_litter_weight_l3225_322529


namespace NUMINAMATH_CALUDE_division_result_l3225_322564

theorem division_result (x : ℚ) : x / 5000 = 0.0114 → x = 57 := by
  sorry

end NUMINAMATH_CALUDE_division_result_l3225_322564


namespace NUMINAMATH_CALUDE_pizza_dough_flour_calculation_l3225_322516

theorem pizza_dough_flour_calculation 
  (original_doughs : ℕ) 
  (original_flour_per_dough : ℚ) 
  (new_doughs : ℕ) 
  (total_flour : ℚ) 
  (h1 : original_doughs = 45)
  (h2 : original_flour_per_dough = 1/9)
  (h3 : new_doughs = 15)
  (h4 : total_flour = original_doughs * original_flour_per_dough)
  (h5 : total_flour = new_doughs * (total_flour / new_doughs)) :
  total_flour / new_doughs = 1/3 := by
sorry

end NUMINAMATH_CALUDE_pizza_dough_flour_calculation_l3225_322516


namespace NUMINAMATH_CALUDE_ham_bread_percentage_l3225_322508

theorem ham_bread_percentage (bread_cost ham_cost cake_cost : ℚ) 
  (h1 : bread_cost = 50)
  (h2 : ham_cost = 150)
  (h3 : cake_cost = 200) :
  (bread_cost + ham_cost) / (bread_cost + ham_cost + cake_cost) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ham_bread_percentage_l3225_322508


namespace NUMINAMATH_CALUDE_water_consumption_l3225_322569

theorem water_consumption (morning_amount : ℝ) (afternoon_multiplier : ℝ) : 
  morning_amount = 1.5 → 
  afternoon_multiplier = 3 → 
  morning_amount + (afternoon_multiplier * morning_amount) = 6 := by
  sorry

end NUMINAMATH_CALUDE_water_consumption_l3225_322569


namespace NUMINAMATH_CALUDE_regular_price_calculation_l3225_322520

/-- Represents the promotional offer and total paid for tires -/
structure TireOffer where
  regularPrice : ℝ  -- Regular price of one tire
  totalPaid : ℝ     -- Total amount paid for four tires
  fourthTirePrice : ℝ -- Price of the fourth tire in the offer

/-- The promotional offer satisfies the given conditions -/
def validOffer (offer : TireOffer) : Prop :=
  offer.totalPaid = 3 * offer.regularPrice + offer.fourthTirePrice

/-- The theorem to prove -/
theorem regular_price_calculation (offer : TireOffer) 
  (h1 : offer.totalPaid = 310)
  (h2 : offer.fourthTirePrice = 5)
  (h3 : validOffer offer) :
  offer.regularPrice = 101.67 := by
  sorry


end NUMINAMATH_CALUDE_regular_price_calculation_l3225_322520


namespace NUMINAMATH_CALUDE_least_common_multiple_of_pack_sizes_l3225_322570

theorem least_common_multiple_of_pack_sizes (tulip_pack_size daffodil_pack_size : ℕ) 
  (h1 : tulip_pack_size = 15) 
  (h2 : daffodil_pack_size = 16) : 
  Nat.lcm tulip_pack_size daffodil_pack_size = 240 := by
  sorry

end NUMINAMATH_CALUDE_least_common_multiple_of_pack_sizes_l3225_322570


namespace NUMINAMATH_CALUDE_determine_a_l3225_322519

-- Define the universal set U
def U (a : ℝ) : Set ℝ := {2, 4, 1-a}

-- Define set A
def A (a : ℝ) : Set ℝ := {2, a^2 - a + 2}

-- Theorem statement
theorem determine_a : 
  ∀ a : ℝ, (U a \ A a = {-1}) → a = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_determine_a_l3225_322519


namespace NUMINAMATH_CALUDE_money_distribution_l3225_322598

theorem money_distribution (total : ℕ) (a b c d : ℕ) : 
  a + b + c + d = total →
  5 * b = 2 * a →
  5 * c = 4 * a →
  5 * d = 3 * a →
  c = d + 500 →
  a = 2500 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l3225_322598


namespace NUMINAMATH_CALUDE_fourth_term_max_coefficient_l3225_322534

def has_max_fourth_term (n : ℕ) : Prop :=
  ∀ k, 0 ≤ k ∧ k ≤ n → Nat.choose n 3 ≥ Nat.choose n k

theorem fourth_term_max_coefficient (n : ℕ) :
  has_max_fourth_term n ↔ n = 5 ∨ n = 6 ∨ n = 7 := by sorry

end NUMINAMATH_CALUDE_fourth_term_max_coefficient_l3225_322534


namespace NUMINAMATH_CALUDE_homework_problem_l3225_322585

theorem homework_problem (total_problems : ℕ) (finished_problems : ℕ) (remaining_pages : ℕ) 
  (x y : ℕ) (h1 : total_problems = 450) (h2 : finished_problems = 185) (h3 : remaining_pages = 15) :
  ∃ (odd_pages even_pages : ℕ), 
    odd_pages + even_pages = remaining_pages ∧ 
    odd_pages * x + even_pages * y = total_problems - finished_problems :=
by sorry

end NUMINAMATH_CALUDE_homework_problem_l3225_322585


namespace NUMINAMATH_CALUDE_julia_math_contest_julia_math_contest_proof_l3225_322560

theorem julia_math_contest (total_problems : ℕ) (correct_points : ℤ) (incorrect_points : ℤ) 
  (julia_score : ℤ) (julia_correct : ℕ) : Prop :=
  total_problems = 12 →
  correct_points = 6 →
  incorrect_points = -3 →
  julia_score = 27 →
  julia_correct = 7 →
  (julia_correct : ℤ) * correct_points + (total_problems - julia_correct : ℤ) * incorrect_points = julia_score

theorem julia_math_contest_proof : 
  ∃ (total_problems : ℕ) (correct_points incorrect_points julia_score : ℤ) (julia_correct : ℕ),
    julia_math_contest total_problems correct_points incorrect_points julia_score julia_correct :=
by
  sorry

end NUMINAMATH_CALUDE_julia_math_contest_julia_math_contest_proof_l3225_322560


namespace NUMINAMATH_CALUDE_key_arrangement_count_l3225_322595

/-- The number of ways to arrange n distinct objects in a circular permutation -/
def circularPermutations (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- The number of boxes -/
def numBoxes : ℕ := 6

theorem key_arrangement_count :
  circularPermutations numBoxes = 120 :=
by sorry

end NUMINAMATH_CALUDE_key_arrangement_count_l3225_322595


namespace NUMINAMATH_CALUDE_min_distance_complex_circle_l3225_322584

open Complex

theorem min_distance_complex_circle (Z : ℂ) (h : abs (Z + 2 - 2*I) = 1) :
  ∃ (min_val : ℝ), min_val = 3 ∧ ∀ (W : ℂ), abs (W + 2 - 2*I) = 1 → abs (W - 2 - 2*I) ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_distance_complex_circle_l3225_322584


namespace NUMINAMATH_CALUDE_least_three_digit_product_24_l3225_322574

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_product (n : ℕ) : ℕ :=
  (n / 100) * ((n / 10) % 10) * (n % 10)

theorem least_three_digit_product_24 :
  ∀ n : ℕ, is_three_digit n → digit_product n = 24 → 234 ≤ n :=
sorry

end NUMINAMATH_CALUDE_least_three_digit_product_24_l3225_322574


namespace NUMINAMATH_CALUDE_shirts_produced_l3225_322553

theorem shirts_produced (shirts_per_minute : ℕ) (minutes_worked : ℕ) : 
  shirts_per_minute = 2 → minutes_worked = 4 → shirts_per_minute * minutes_worked = 8 := by
  sorry

end NUMINAMATH_CALUDE_shirts_produced_l3225_322553


namespace NUMINAMATH_CALUDE_trees_in_yard_l3225_322589

/-- The number of trees planted along a yard. -/
def number_of_trees (yard_length : ℕ) (tree_distance : ℕ) : ℕ :=
  (yard_length / tree_distance) + 1

/-- Theorem: Given a yard 441 metres long with trees planted at equal distances,
    one tree at each end, and 21 metres between consecutive trees,
    there are 22 trees planted along the yard. -/
theorem trees_in_yard :
  let yard_length : ℕ := 441
  let tree_distance : ℕ := 21
  number_of_trees yard_length tree_distance = 22 := by
  sorry

end NUMINAMATH_CALUDE_trees_in_yard_l3225_322589


namespace NUMINAMATH_CALUDE_probability_of_pink_flower_l3225_322537

-- Define the contents of the bags
def bag_A_red : ℕ := 6
def bag_A_pink : ℕ := 3
def bag_B_red : ℕ := 2
def bag_B_pink : ℕ := 7

-- Define the total number of flowers in each bag
def total_A : ℕ := bag_A_red + bag_A_pink
def total_B : ℕ := bag_B_red + bag_B_pink

-- Define the probability of choosing a pink flower from each bag
def prob_pink_A : ℚ := bag_A_pink / total_A
def prob_pink_B : ℚ := bag_B_pink / total_B

-- Theorem statement
theorem probability_of_pink_flower :
  (prob_pink_A + prob_pink_B) / 2 = 5 / 9 := by sorry

end NUMINAMATH_CALUDE_probability_of_pink_flower_l3225_322537


namespace NUMINAMATH_CALUDE_abs_S_value_l3225_322568

/-- The imaginary unit i -/
def i : ℂ := Complex.I

/-- The complex number S -/
def S : ℂ := (1 + 2*i)^12 - (1 - i)^12

/-- Theorem stating the absolute value of S -/
theorem abs_S_value : Complex.abs S = 15689 := by sorry

end NUMINAMATH_CALUDE_abs_S_value_l3225_322568


namespace NUMINAMATH_CALUDE_bing_position_guimao_in_cycle_l3225_322554

-- Define the cyclic arrangement
def heavenly_stems := 10
def earthly_branches := 12
def cycle_length := 60

-- Define the position of 丙 (bǐng)
def bing_first_appearance := 3

-- Define the function for the nth appearance of 丙 (bǐng)
def bing_column (n : ℕ) : ℕ := 10 * n - 7

-- Define the position of 癸卯 (guǐ mǎo) in the cycle
def guimao_position := 40

-- Theorem for the position of 丙 (bǐng)
theorem bing_position (n : ℕ) : 
  bing_column n ≡ bing_first_appearance [MOD cycle_length] :=
sorry

-- Theorem for the position of 癸卯 (guǐ mǎo)
theorem guimao_in_cycle : 
  guimao_position > 0 ∧ guimao_position ≤ cycle_length :=
sorry

end NUMINAMATH_CALUDE_bing_position_guimao_in_cycle_l3225_322554


namespace NUMINAMATH_CALUDE_unique_solution_l3225_322542

-- Define the recursive sequence
def s (x : ℤ) : ℕ → ℤ
  | 0 => 0
  | n + 1 => (s x n ^ 2 + x).sqrt

-- Define the equation with 1998 square roots
def equation (x y : ℤ) : Prop :=
  s x 1998 = y

-- Theorem statement
theorem unique_solution :
  ∀ x y : ℤ, equation x y → x = 0 ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3225_322542


namespace NUMINAMATH_CALUDE_equivalent_operations_l3225_322583

theorem equivalent_operations (x : ℝ) : (x * (4/5)) / (4/7) = x * (7/5) := by
  sorry

end NUMINAMATH_CALUDE_equivalent_operations_l3225_322583


namespace NUMINAMATH_CALUDE_f_positive_implies_a_range_l3225_322565

open Real

/-- The function f(x) defined in terms of parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := (a - 1) * x^2 + (a - 1) * x + 1

/-- Theorem stating that if f(x) > 0 for all real x, then 1 ≤ a < 5 -/
theorem f_positive_implies_a_range (a : ℝ) :
  (∀ x : ℝ, f a x > 0) → 1 ≤ a ∧ a < 5 := by
  sorry

end NUMINAMATH_CALUDE_f_positive_implies_a_range_l3225_322565


namespace NUMINAMATH_CALUDE_max_square_plots_l3225_322535

/-- Represents the field dimensions and available fencing -/
structure FieldData where
  width : ℝ
  length : ℝ
  fence : ℝ

/-- Calculates the number of square plots given the number of plots along the width -/
def numPlots (n : ℕ) : ℕ := n * (2 * n)

/-- Calculates the length of fence used given the number of plots along the width -/
def fenceUsed (n : ℕ) : ℝ := 120 * n - 90

/-- The main theorem stating the maximum number of square plots -/
theorem max_square_plots (field : FieldData) 
    (h_width : field.width = 30)
    (h_length : field.length = 60)
    (h_fence : field.fence = 2268) : 
  (∃ (n : ℕ), numPlots n = 722 ∧ 
              fenceUsed n ≤ field.fence ∧ 
              ∀ (m : ℕ), m > n → fenceUsed m > field.fence) :=
sorry

end NUMINAMATH_CALUDE_max_square_plots_l3225_322535


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3225_322594

/-- Given a geometric sequence {a_n} where the sum of the first n terms S_n
    is defined as S_n = x · 3^n + 1, this theorem states that x = -1. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (x : ℝ) :
  (∀ n, S n = x * 3^n + 1) →
  (∀ n, a (n+1) / a n = a (n+2) / a (n+1)) →
  x = -1 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3225_322594


namespace NUMINAMATH_CALUDE_sixth_term_value_l3225_322578

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the properties of a₄ and a₈
def roots_property (a : ℕ → ℝ) : Prop :=
  a 4 ^ 2 - 34 * a 4 + 64 = 0 ∧ a 8 ^ 2 - 34 * a 8 + 64 = 0

-- Theorem statement
theorem sixth_term_value (a : ℕ → ℝ) 
  (h1 : geometric_sequence a) 
  (h2 : roots_property a) : 
  a 6 = 8 := by sorry

end NUMINAMATH_CALUDE_sixth_term_value_l3225_322578


namespace NUMINAMATH_CALUDE_geometric_sequence_roots_property_l3225_322503

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the property of a_4 and a_12 being roots of x^2 + 3x + 1 = 0
def roots_property (a : ℕ → ℝ) : Prop :=
  a 4 + a 12 = -3 ∧ a 4 * a 12 = 1

theorem geometric_sequence_roots_property (a : ℕ → ℝ) :
  geometric_sequence a →
  (roots_property a → a 8 = -1) ∧
  ¬(a 8 = -1 → roots_property a) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_roots_property_l3225_322503


namespace NUMINAMATH_CALUDE_sequence_representation_l3225_322548

theorem sequence_representation (a : ℕ → ℝ) 
  (h0 : a 0 = 4)
  (h1 : a 1 = 22)
  (h_rec : ∀ n : ℕ, n ≥ 2 → a n - 6 * a (n - 1) + a (n - 2) = 0) :
  ∃ x y : ℕ → ℕ, ∀ n : ℕ, a n = (y n ^ 2 + 7) / (x n - y n) := by
sorry

end NUMINAMATH_CALUDE_sequence_representation_l3225_322548


namespace NUMINAMATH_CALUDE_pie_eating_contest_l3225_322538

theorem pie_eating_contest (a b c : ℚ) 
  (ha : a = 5/6) (hb : b = 2/3) (hc : c = 3/4) : 
  (max a (max b c) - min a (min b c)) = 1/6 :=
sorry

end NUMINAMATH_CALUDE_pie_eating_contest_l3225_322538


namespace NUMINAMATH_CALUDE_prob_open_door_third_attempt_l3225_322557

/-- Probability of opening a door on the third attempt given 5 keys with only one correct key -/
theorem prob_open_door_third_attempt (total_keys : ℕ) (correct_keys : ℕ) (attempt : ℕ) :
  total_keys = 5 →
  correct_keys = 1 →
  attempt = 3 →
  (1 : ℚ) / total_keys = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_prob_open_door_third_attempt_l3225_322557


namespace NUMINAMATH_CALUDE_number_separation_l3225_322586

theorem number_separation (a b : ℝ) (h1 : a = 50) (h2 : 0.40 * a = 0.625 * b + 10) : a + b = 66 := by
  sorry

end NUMINAMATH_CALUDE_number_separation_l3225_322586


namespace NUMINAMATH_CALUDE_tommy_calculation_l3225_322543

theorem tommy_calculation (x : ℚ) : (x - 7) / 5 = 23 → (x - 5) / 7 = 16 := by
  sorry

end NUMINAMATH_CALUDE_tommy_calculation_l3225_322543


namespace NUMINAMATH_CALUDE_lamp_sales_theorem_l3225_322502

/-- The monthly average growth rate of lamp sales -/
def monthly_growth_rate : ℝ := 0.2

/-- The price of lamps in April to achieve the target profit -/
def april_price : ℝ := 38

/-- Initial sales volume in January -/
def january_sales : ℕ := 400

/-- Sales volume in March -/
def march_sales : ℕ := 576

/-- Purchase cost per lamp -/
def purchase_cost : ℝ := 30

/-- Initial selling price -/
def initial_price : ℝ := 40

/-- Increase in sales volume per 0.5 yuan price reduction -/
def sales_increase_per_half_yuan : ℕ := 6

/-- Target profit in April -/
def target_profit : ℝ := 4800

/-- Theorem stating the correctness of the monthly growth rate and April price -/
theorem lamp_sales_theorem :
  (january_sales * (1 + monthly_growth_rate)^2 = march_sales) ∧
  ((april_price - purchase_cost) *
    (march_sales + 2 * sales_increase_per_half_yuan * (initial_price - april_price)) = target_profit) := by
  sorry


end NUMINAMATH_CALUDE_lamp_sales_theorem_l3225_322502


namespace NUMINAMATH_CALUDE_distance_ratio_l3225_322527

-- Define the speeds and time for both cars
def speed_A : ℝ := 70
def speed_B : ℝ := 35
def time : ℝ := 10

-- Define the distances traveled by each car
def distance_A : ℝ := speed_A * time
def distance_B : ℝ := speed_B * time

-- Theorem to prove the ratio of distances
theorem distance_ratio :
  distance_A / distance_B = 2 := by sorry

end NUMINAMATH_CALUDE_distance_ratio_l3225_322527


namespace NUMINAMATH_CALUDE_cube_sum_inequality_l3225_322522

theorem cube_sum_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^3 / (x + y)) + (y^3 / (y + z)) + (z^3 / (z + x)) ≥ (x*y + y*z + z*x) / 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_inequality_l3225_322522


namespace NUMINAMATH_CALUDE_cubic_function_derivative_condition_l3225_322561

/-- Given a function f(x) = x^3 - mx + 3, if f'(1) = 0, then m = 3 -/
theorem cubic_function_derivative_condition (m : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^3 - m*x + 3
  (∀ x, (deriv f) x = 3*x^2 - m) → (deriv f) 1 = 0 → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_derivative_condition_l3225_322561


namespace NUMINAMATH_CALUDE_radius_of_circle_Q_l3225_322573

/-- Triangle ABC with given side lengths -/
structure Triangle :=
  (AB : ℝ)
  (AC : ℝ)
  (BC : ℝ)

/-- Circle with center and radius -/
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

/-- Tangency between a circle and a line segment -/
def IsTangent (c : Circle) (p q : ℝ × ℝ) : Prop := sorry

/-- External tangency between two circles -/
def IsExternallyTangent (c1 c2 : Circle) : Prop := sorry

/-- Circle lies inside a triangle -/
def CircleInsideTriangle (c : Circle) (t : Triangle) : Prop := sorry

/-- Main theorem -/
theorem radius_of_circle_Q (t : Triangle) (p q : Circle) :
  t.AB = 144 ∧ t.AC = 144 ∧ t.BC = 80 ∧
  p.radius = 24 ∧
  IsTangent p (0, 0) (t.AC, 0) ∧
  IsTangent p (t.BC, 0) (0, 0) ∧
  IsExternallyTangent p q ∧
  IsTangent q (0, 0) (t.AB, 0) ∧
  IsTangent q (t.BC, 0) (0, 0) ∧
  CircleInsideTriangle q t →
  q.radius = 64 - 12 * Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_radius_of_circle_Q_l3225_322573


namespace NUMINAMATH_CALUDE_angle_inequality_equivalence_l3225_322526

theorem angle_inequality_equivalence (θ : Real) : 
  (∀ x : Real, 0 ≤ x ∧ x ≤ 1 → 
    x^3 * Real.sin θ + x^2 * Real.cos θ - x * (1 - x) + (1 - x)^2 * Real.sin θ > 0) ↔ 
  (π / 12 < θ ∧ θ < 5 * π / 12) :=
by sorry

end NUMINAMATH_CALUDE_angle_inequality_equivalence_l3225_322526


namespace NUMINAMATH_CALUDE_triangle_base_length_l3225_322533

theorem triangle_base_length 
  (area : ℝ) 
  (height : ℝ) 
  (h1 : area = 10) 
  (h2 : height = 5) : 
  area = (height * 4) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_base_length_l3225_322533


namespace NUMINAMATH_CALUDE_revenue_maximized_at_optimal_price_l3225_322599

/-- Revenue function for the bookstore --/
def R (p : ℝ) : ℝ := p * (145 - 7 * p)

/-- The optimal price that maximizes revenue --/
def optimal_price : ℕ := 10

theorem revenue_maximized_at_optimal_price :
  ∀ p : ℕ, p ≤ 30 → R p ≤ R optimal_price := by
  sorry

end NUMINAMATH_CALUDE_revenue_maximized_at_optimal_price_l3225_322599


namespace NUMINAMATH_CALUDE_flower_pattern_perimeter_l3225_322518

/-- The perimeter of a "flower" pattern formed by removing a 45° sector from a circle --/
theorem flower_pattern_perimeter (r : ℝ) (h : r = 3) : 
  let circumference := 2 * π * r
  let arc_length := (315 / 360) * circumference
  let straight_edges := 2 * r
  arc_length + straight_edges = (21 / 4) * π + 6 := by
  sorry

end NUMINAMATH_CALUDE_flower_pattern_perimeter_l3225_322518


namespace NUMINAMATH_CALUDE_negation_equivalence_l3225_322504

variable (a : ℝ)

def original_proposition : Prop :=
  ∀ x : ℝ, x^2 + (a - 1) * x + 1 ≥ 0

theorem negation_equivalence :
  (¬ original_proposition a) ↔ (∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3225_322504


namespace NUMINAMATH_CALUDE_trivia_team_points_l3225_322575

/-- Calculates the total points scored by a trivia team -/
def total_points (total_members : ℕ) (absent_members : ℕ) (points_per_member : ℕ) : ℕ :=
  (total_members - absent_members) * points_per_member

/-- Proves that the total points scored by the trivia team is 64 -/
theorem trivia_team_points :
  total_points 12 4 8 = 64 := by
  sorry

end NUMINAMATH_CALUDE_trivia_team_points_l3225_322575


namespace NUMINAMATH_CALUDE_calculate_expression_l3225_322549

theorem calculate_expression : ((18^18 / 18^17)^3 * 8^3) / 2^9 = 5832 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3225_322549


namespace NUMINAMATH_CALUDE_even_pairs_ge_odd_pairs_l3225_322591

/-- A sequence of zeros and ones -/
def BinarySequence := List Bool

/-- Count pairs (1,0) with even number of digits between them -/
def countEvenPairs (seq : BinarySequence) : ℕ := sorry

/-- Count pairs (1,0) with odd number of digits between them -/
def countOddPairs (seq : BinarySequence) : ℕ := sorry

/-- Theorem: In any binary sequence, the number of (1,0) pairs with even digits between
    is greater than or equal to the number of (1,0) pairs with odd digits between -/
theorem even_pairs_ge_odd_pairs (seq : BinarySequence) :
  countEvenPairs seq ≥ countOddPairs seq := by
  sorry

end NUMINAMATH_CALUDE_even_pairs_ge_odd_pairs_l3225_322591


namespace NUMINAMATH_CALUDE_speaker_combinations_l3225_322531

/-- Represents the number of representatives for each company -/
def company_reps : List ℕ := [2, 1, 1, 1, 1]

/-- The total number of companies -/
def num_companies : ℕ := company_reps.length

/-- The number of speakers required -/
def num_speakers : ℕ := 3

/-- Calculates the number of ways to choose speakers from different companies -/
def choose_speakers (reps : List ℕ) (k : ℕ) : ℕ := sorry

theorem speaker_combinations :
  choose_speakers company_reps num_speakers = 16 := by sorry

end NUMINAMATH_CALUDE_speaker_combinations_l3225_322531


namespace NUMINAMATH_CALUDE_circle_op_five_three_l3225_322556

-- Define the operation ∘
def circle_op (a b : ℕ) : ℕ := 4*a + 6*b + 1

-- State the theorem
theorem circle_op_five_three : circle_op 5 3 = 39 := by sorry

end NUMINAMATH_CALUDE_circle_op_five_three_l3225_322556


namespace NUMINAMATH_CALUDE_triangle_area_is_13_5_l3225_322566

/-- The area of a triangular region bounded by the two coordinate axes and the line 3x + y = 9 -/
def triangleArea : ℝ := 13.5

/-- The equation of the line bounding the triangular region -/
def lineEquation (x y : ℝ) : Prop := 3 * x + y = 9

/-- Theorem stating that the area of the triangular region is 13.5 square units -/
theorem triangle_area_is_13_5 :
  triangleArea = 13.5 := by sorry

end NUMINAMATH_CALUDE_triangle_area_is_13_5_l3225_322566


namespace NUMINAMATH_CALUDE_binomial_coefficient_identity_l3225_322509

theorem binomial_coefficient_identity (n k : ℕ) (h : k ≤ n) :
  k * Nat.choose n k = n * Nat.choose (n - 1) (k - 1) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_identity_l3225_322509


namespace NUMINAMATH_CALUDE_cricket_run_rate_problem_l3225_322551

/-- Calculates the required run rate for the remaining overs in a cricket game -/
def required_run_rate (total_overs : ℕ) (first_overs : ℕ) (first_run_rate : ℚ) (target : ℕ) : ℚ :=
  let remaining_overs := total_overs - first_overs
  let runs_scored := first_run_rate * first_overs
  let runs_needed := target - runs_scored
  runs_needed / remaining_overs

/-- Theorem stating the required run rate for the given cricket game scenario -/
theorem cricket_run_rate_problem :
  required_run_rate 50 10 (32/10) 272 = 6 := by
  sorry

end NUMINAMATH_CALUDE_cricket_run_rate_problem_l3225_322551


namespace NUMINAMATH_CALUDE_final_breath_holding_time_l3225_322506

def breath_holding_progress (initial_time : ℝ) : ℝ :=
  let week1 := initial_time * 2
  let week2 := week1 * 2
  let week3 := week2 * 1.5
  week3

theorem final_breath_holding_time :
  breath_holding_progress 10 = 60 := by
  sorry

end NUMINAMATH_CALUDE_final_breath_holding_time_l3225_322506


namespace NUMINAMATH_CALUDE_money_distribution_l3225_322572

/-- Given a distribution of money in the ratio 3:5:7 among three people,
    where the second person's share is 1500,
    prove that the difference between the first and third person's shares is 1200. -/
theorem money_distribution (total : ℕ) (share1 share2 share3 : ℕ) :
  share1 + share2 + share3 = total →
  3 * share2 = 5 * share1 →
  7 * share1 = 3 * share3 →
  share2 = 1500 →
  share3 - share1 = 1200 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l3225_322572


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l3225_322571

/-- Proves that the speed of a boat in still water is 24 km/hr, given the conditions -/
theorem boat_speed_in_still_water :
  let stream_speed : ℝ := 4
  let downstream_distance : ℝ := 84
  let downstream_time : ℝ := 3
  let boat_speed : ℝ := (downstream_distance / downstream_time) - stream_speed
  boat_speed = 24 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l3225_322571


namespace NUMINAMATH_CALUDE_smallest_c_for_inequality_l3225_322552

theorem smallest_c_for_inequality : ∃ c : ℕ, c = 9 ∧ (∀ k : ℕ, 27 ^ k > 3 ^ 24 → k ≥ c) := by
  sorry

end NUMINAMATH_CALUDE_smallest_c_for_inequality_l3225_322552


namespace NUMINAMATH_CALUDE_reflection_across_x_axis_l3225_322541

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define reflection across x-axis
def reflect_x_axis (p : Point) : Point :=
  (p.1, -p.2)

-- Theorem statement
theorem reflection_across_x_axis (p : Point) :
  p = (-2, 1) → reflect_x_axis p = (-2, -1) := by
  sorry

end NUMINAMATH_CALUDE_reflection_across_x_axis_l3225_322541


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l3225_322544

/-- A geometric sequence with first term 2 and satisfying a₄a₆ = 4a₇² has a₃ = 1 -/
theorem geometric_sequence_problem (a : ℕ → ℝ) (h1 : a 1 = 2) 
  (h2 : a 4 * a 6 = 4 * (a 7)^2) (h3 : ∀ n : ℕ, n ≥ 1 → ∃ q : ℝ, a (n + 1) = a n * q) : 
  a 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l3225_322544


namespace NUMINAMATH_CALUDE_extrema_and_tangent_line_l3225_322596

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3*x

-- State the theorem
theorem extrema_and_tangent_line :
  -- Local extrema conditions
  (∃ ε > 0, ∀ x ∈ Set.Ioo (-1 - ε) (-1 + ε), f x ≤ f (-1)) ∧
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≥ f 1) ∧
  -- Tangent line condition
  (∃ x₀ : ℝ, 9*x₀ - f x₀ + 16 = 0 ∧
    ∀ x : ℝ, 9*x - f x + 16 = 0 → x = x₀) :=
by sorry

end NUMINAMATH_CALUDE_extrema_and_tangent_line_l3225_322596


namespace NUMINAMATH_CALUDE_fourth_month_sale_l3225_322590

theorem fourth_month_sale
  (sale1 sale2 sale3 sale5 sale6 : ℕ)
  (average_sale : ℕ)
  (h1 : sale1 = 6435)
  (h2 : sale2 = 6927)
  (h3 : sale3 = 6855)
  (h5 : sale5 = 6562)
  (h6 : sale6 = 6191)
  (h_avg : average_sale = 6700)
  (h_total : average_sale * 6 = sale1 + sale2 + sale3 + sale5 + sale6 + sale4) :
  sale4 = 7230 :=
by
  sorry

end NUMINAMATH_CALUDE_fourth_month_sale_l3225_322590


namespace NUMINAMATH_CALUDE_initial_overs_played_l3225_322513

/-- Proves that the number of overs played initially is 10, given the specified conditions --/
theorem initial_overs_played (total_target : ℝ) (initial_run_rate : ℝ) (remaining_overs : ℝ) (required_run_rate : ℝ)
  (h1 : total_target = 282)
  (h2 : initial_run_rate = 3.8)
  (h3 : remaining_overs = 40)
  (h4 : required_run_rate = 6.1)
  : ∃ (x : ℝ), x = 10 ∧ initial_run_rate * x + required_run_rate * remaining_overs = total_target :=
by
  sorry

end NUMINAMATH_CALUDE_initial_overs_played_l3225_322513


namespace NUMINAMATH_CALUDE_nonDefectiveEnginesCount_l3225_322550

/-- Given a number of batches and engines per batch, calculates the number of non-defective engines
    when one fourth of the total engines are defective. -/
def nonDefectiveEngines (batches : ℕ) (enginesPerBatch : ℕ) : ℕ :=
  let totalEngines := batches * enginesPerBatch
  let defectiveEngines := totalEngines / 4
  totalEngines - defectiveEngines

/-- Proves that given 5 batches of 80 engines each, with one fourth being defective,
    the number of non-defective engines is 300. -/
theorem nonDefectiveEnginesCount :
  nonDefectiveEngines 5 80 = 300 := by
  sorry

#eval nonDefectiveEngines 5 80

end NUMINAMATH_CALUDE_nonDefectiveEnginesCount_l3225_322550


namespace NUMINAMATH_CALUDE_number_ratio_l3225_322512

theorem number_ratio (x : ℝ) (h : x + 5 = 17) : x / (2 * x) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_number_ratio_l3225_322512


namespace NUMINAMATH_CALUDE_trig_identity_on_line_l3225_322588

/-- If the terminal side of angle α lies on the line y = 2x, 
    then sin²α - cos²α + sin α * cos α = 1 -/
theorem trig_identity_on_line (α : Real) 
  (h : Real.tan α = 2) : 
  Real.sin α ^ 2 - Real.cos α ^ 2 + Real.sin α * Real.cos α = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_on_line_l3225_322588


namespace NUMINAMATH_CALUDE_solve_equation_l3225_322577

theorem solve_equation (x y : ℝ) :
  3 * x - 5 * y = 7 → y = (3 * x - 7) / 5 := by
sorry

end NUMINAMATH_CALUDE_solve_equation_l3225_322577


namespace NUMINAMATH_CALUDE_tenth_minus_ninth_square_diff_l3225_322580

/-- The number of tiles in the nth square of the sequence -/
def tiles_in_square (n : ℕ) : ℕ := n^2

/-- The theorem stating the difference in tiles between the 10th and 9th squares -/
theorem tenth_minus_ninth_square_diff : tiles_in_square 10 - tiles_in_square 9 = 19 := by
  sorry

end NUMINAMATH_CALUDE_tenth_minus_ninth_square_diff_l3225_322580


namespace NUMINAMATH_CALUDE_garden_length_l3225_322555

/-- Given a rectangular garden with perimeter 1200 meters and breadth 240 meters, 
    prove that its length is 360 meters. -/
theorem garden_length (perimeter : ℝ) (breadth : ℝ) (length : ℝ) : 
  perimeter = 1200 ∧ 
  breadth = 240 ∧ 
  perimeter = 2 * (length + breadth) →
  length = 360 := by
sorry

end NUMINAMATH_CALUDE_garden_length_l3225_322555


namespace NUMINAMATH_CALUDE_opposite_minus_six_l3225_322563

theorem opposite_minus_six (a : ℤ) : a = -(-6) → 1 - a = -5 := by
  sorry

end NUMINAMATH_CALUDE_opposite_minus_six_l3225_322563


namespace NUMINAMATH_CALUDE_equilateral_triangle_25_division_equilateral_triangle_5_equal_parts_l3225_322539

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- Represents a division of an equilateral triangle -/
structure TriangleDivision where
  original : EquilateralTriangle
  num_divisions : ℕ
  num_divisions_pos : num_divisions > 0

/-- Theorem: An equilateral triangle can be divided into 25 smaller equilateral triangles -/
theorem equilateral_triangle_25_division (t : EquilateralTriangle) :
  ∃ (d : TriangleDivision), d.original = t ∧ d.num_divisions = 25 :=
sorry

/-- Represents a grouping of the divided triangles -/
structure TriangleGrouping where
  division : TriangleDivision
  num_groups : ℕ
  num_groups_pos : num_groups > 0
  triangles_per_group : ℕ
  triangles_per_group_pos : triangles_per_group > 0
  valid_grouping : division.num_divisions = num_groups * triangles_per_group

/-- Theorem: The 25 smaller triangles can be grouped into 5 equal parts -/
theorem equilateral_triangle_5_equal_parts (t : EquilateralTriangle) :
  ∃ (g : TriangleGrouping), g.division.original = t ∧ g.num_groups = 5 ∧ g.triangles_per_group = 5 :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_25_division_equilateral_triangle_5_equal_parts_l3225_322539


namespace NUMINAMATH_CALUDE_non_working_video_games_l3225_322562

theorem non_working_video_games (total : ℕ) (price : ℕ) (earnings : ℕ) : 
  total = 10 → price = 6 → earnings = 12 → total - (earnings / price) = 8 := by
  sorry

end NUMINAMATH_CALUDE_non_working_video_games_l3225_322562


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_l3225_322511

theorem floor_ceiling_sum (x y : ℝ) (hx : 1 < x ∧ x < 2) (hy : 3 < y ∧ y < 4) :
  ⌊x⌋ + ⌈y⌉ = 5 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_l3225_322511


namespace NUMINAMATH_CALUDE_sum_of_max_min_l3225_322581

theorem sum_of_max_min (a b c d : ℝ) (ha : a = 0.11) (hb : b = 0.98) (hc : c = 3/4) (hd : d = 2/3) :
  (max a (max b (max c d))) + (min a (min b (min c d))) = 1.09 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_max_min_l3225_322581


namespace NUMINAMATH_CALUDE_second_polygon_sides_l3225_322505

theorem second_polygon_sides (perimeter : ℝ) (side_length_second : ℝ) : 
  perimeter > 0 → side_length_second > 0 →
  perimeter = 50 * (3 * side_length_second) →
  perimeter = 150 * side_length_second := by
  sorry

end NUMINAMATH_CALUDE_second_polygon_sides_l3225_322505


namespace NUMINAMATH_CALUDE_same_parity_iff_square_sum_l3225_322545

theorem same_parity_iff_square_sum (a b : ℤ) :
  (∃ k : ℤ, a - b = 2 * k) ↔ (∃ c d : ℤ, a^2 + b^2 + c^2 + 1 = d^2) := by sorry

end NUMINAMATH_CALUDE_same_parity_iff_square_sum_l3225_322545


namespace NUMINAMATH_CALUDE_cube_face_sum_l3225_322597

/-- Represents the numbers on the faces of a cube -/
structure CubeFaces where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  d : ℕ+
  e : ℕ+
  f : ℕ+

/-- The sum of vertex products for a given set of cube faces -/
def vertexProductSum (faces : CubeFaces) : ℕ :=
  faces.a * faces.b * faces.c +
  faces.a * faces.e * faces.c +
  faces.a * faces.b * faces.f +
  faces.a * faces.e * faces.f +
  faces.d * faces.b * faces.c +
  faces.d * faces.e * faces.c +
  faces.d * faces.b * faces.f +
  faces.d * faces.e * faces.f

/-- The sum of all face values -/
def faceSum (faces : CubeFaces) : ℕ :=
  faces.a + faces.b + faces.c + faces.d + faces.e + faces.f

theorem cube_face_sum (faces : CubeFaces) :
  vertexProductSum faces = 1008 → faceSum faces = 173 := by
  sorry


end NUMINAMATH_CALUDE_cube_face_sum_l3225_322597


namespace NUMINAMATH_CALUDE_sum_of_three_element_subset_sums_l3225_322547

def A : Finset ℕ := Finset.range 10

def three_element_subsets (s : Finset ℕ) : Finset (Finset ℕ) :=
  s.powerset.filter (λ subset => subset.card = 3)

def subset_sum (subset : Finset ℕ) : ℕ :=
  subset.sum id

theorem sum_of_three_element_subset_sums : 
  (three_element_subsets A).sum subset_sum = 1980 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_element_subset_sums_l3225_322547


namespace NUMINAMATH_CALUDE_palm_tree_count_l3225_322558

theorem palm_tree_count (desert forest : ℕ) 
  (h1 : desert = (2 : ℚ) / 5 * forest)  -- Desert has 2/5 the trees of the forest
  (h2 : desert + forest = 7000)         -- Total trees in both locations
  : forest = 5000 := by
  sorry

end NUMINAMATH_CALUDE_palm_tree_count_l3225_322558


namespace NUMINAMATH_CALUDE_football_players_l3225_322587

theorem football_players (total : ℕ) (cricket : ℕ) (neither : ℕ) (both : ℕ) 
  (h1 : total = 410)
  (h2 : cricket = 175)
  (h3 : neither = 50)
  (h4 : both = 140) :
  total - cricket + both - neither = 375 := by
  sorry

end NUMINAMATH_CALUDE_football_players_l3225_322587


namespace NUMINAMATH_CALUDE_value_of_a_l3225_322514

theorem value_of_a (a : ℝ) (S : Set ℝ) : 
  S = {x : ℝ | 3 * x + a = 0} → 
  (1 : ℝ) ∈ S → 
  a = -3 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_l3225_322514


namespace NUMINAMATH_CALUDE_ratio_problem_l3225_322515

theorem ratio_problem (x y : ℚ) (h : (3*x - 2*y) / (2*x + y) = 5/4) : x / y = 13/2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l3225_322515


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l3225_322536

theorem solution_set_quadratic_inequality :
  {x : ℝ | 4 - x^2 < 0} = Set.Ioi 2 ∪ Set.Iio (-2) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l3225_322536


namespace NUMINAMATH_CALUDE_isosceles_triangle_is_convex_l3225_322525

-- Define an isosceles triangle
structure IsoscelesTriangle where
  sides : Fin 3 → ℝ
  is_isosceles : ∃ (i j : Fin 3), i ≠ j ∧ sides i = sides j

-- Define a convex polygon
def is_convex (polygon : Fin n → ℝ × ℝ) : Prop :=
  ∀ i j : Fin n, ∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 →
    ∃ k : Fin n, polygon k = (1 - t) • (polygon i) + t • (polygon j)

-- Theorem statement
theorem isosceles_triangle_is_convex (T : IsoscelesTriangle) :
  is_convex (λ i : Fin 3 => sorry) :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_is_convex_l3225_322525


namespace NUMINAMATH_CALUDE_triangles_in_3x4_grid_l3225_322523

/-- Represents a rectangular grid with diagonals --/
structure RectangularGridWithDiagonals where
  rows : Nat
  columns : Nat

/-- Calculates the number of triangles in a rectangular grid with diagonals --/
def count_triangles (grid : RectangularGridWithDiagonals) : Nat :=
  let basic_triangles := 2 * grid.rows * grid.columns
  let row_triangles := grid.rows * (grid.columns - 1) * grid.columns / 2
  let diagonal_triangles := 2
  basic_triangles + row_triangles + diagonal_triangles

/-- Theorem: The number of triangles in a 3x4 grid with diagonals is 44 --/
theorem triangles_in_3x4_grid :
  count_triangles ⟨3, 4⟩ = 44 := by
  sorry

#eval count_triangles ⟨3, 4⟩

end NUMINAMATH_CALUDE_triangles_in_3x4_grid_l3225_322523


namespace NUMINAMATH_CALUDE_fraction_equality_l3225_322567

theorem fraction_equality : (1 : ℝ) / (2 - Real.sqrt 3) = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3225_322567
