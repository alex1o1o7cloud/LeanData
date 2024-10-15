import Mathlib

namespace NUMINAMATH_GPT_pentagonal_grid_toothpicks_l2285_228579

theorem pentagonal_grid_toothpicks :
  ∀ (base toothpicks per sides toothpicks per joint : ℕ),
    base = 10 → 
    sides = 4 → 
    toothpicks_per_side = 8 → 
    joints = 5 → 
    toothpicks_per_joint = 1 → 
    (base + sides * toothpicks_per_side + joints * toothpicks_per_joint = 47) :=
by
  intros base sides toothpicks_per_side joints toothpicks_per_joint
  sorry

end NUMINAMATH_GPT_pentagonal_grid_toothpicks_l2285_228579


namespace NUMINAMATH_GPT_calculate_perimeter_l2285_228541

-- Definitions based on conditions
def num_posts : ℕ := 36
def post_width : ℕ := 2
def gap_width : ℕ := 4
def sides : ℕ := 4

-- Computations inferred from the conditions (not using solution steps directly)
def posts_per_side : ℕ := num_posts / sides
def gaps_per_side : ℕ := posts_per_side - 1
def side_length : ℕ := posts_per_side * post_width + gaps_per_side * gap_width

-- Theorem statement, proving the perimeter is 200 feet
theorem calculate_perimeter : 4 * side_length = 200 := by
  sorry

end NUMINAMATH_GPT_calculate_perimeter_l2285_228541


namespace NUMINAMATH_GPT_percentage_of_same_grade_is_48_l2285_228548

def students_with_same_grade (grades : ℕ × ℕ → ℕ) : ℕ :=
  grades (0, 0) + grades (1, 1) + grades (2, 2) + grades (3, 3) + grades (4, 4)

theorem percentage_of_same_grade_is_48
  (grades : ℕ × ℕ → ℕ)
  (h : grades (0, 0) = 3 ∧ grades (1, 1) = 6 ∧ grades (2, 2) = 8 ∧ grades (3, 3) = 4 ∧ grades (4, 4) = 3)
  (total_students : ℕ) (h_students : total_students = 50) :
  (students_with_same_grade grades / 50 : ℚ) * 100 = 48 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_same_grade_is_48_l2285_228548


namespace NUMINAMATH_GPT_find_y_of_x_pow_l2285_228564

theorem find_y_of_x_pow (x y : ℝ) (h1 : x = 2) (h2 : x^(3*y - 1) = 8) : y = 4 / 3 :=
by
  -- skipping proof
  sorry

end NUMINAMATH_GPT_find_y_of_x_pow_l2285_228564


namespace NUMINAMATH_GPT_quadrilateral_area_lt_one_l2285_228555

theorem quadrilateral_area_lt_one 
  (a b c d : ℝ) 
  (h_a : a < 1) 
  (h_b : b < 1) 
  (h_c : c < 1) 
  (h_d : d < 1) 
  (h_pos_a : 0 ≤ a)
  (h_pos_b : 0 ≤ b)
  (h_pos_c : 0 ≤ c)
  (h_pos_d : 0 ≤ d) :
  ∃ (area : ℝ), area < 1 :=
by
  sorry

end NUMINAMATH_GPT_quadrilateral_area_lt_one_l2285_228555


namespace NUMINAMATH_GPT_alex_ahead_of_max_after_even_l2285_228551

theorem alex_ahead_of_max_after_even (x : ℕ) (h1 : x - 200 + 170 + 440 = 1110) : x = 300 :=
sorry

end NUMINAMATH_GPT_alex_ahead_of_max_after_even_l2285_228551


namespace NUMINAMATH_GPT_smallest_k_for_64k_greater_than_6_l2285_228581

theorem smallest_k_for_64k_greater_than_6 : ∃ (k : ℕ), 64 ^ k > 6 ∧ ∀ m : ℕ, m < k → 64 ^ m ≤ 6 :=
by
  use 1
  sorry

end NUMINAMATH_GPT_smallest_k_for_64k_greater_than_6_l2285_228581


namespace NUMINAMATH_GPT_final_price_is_correct_l2285_228594

def cost_cucumber : ℝ := 5
def cost_tomato : ℝ := cost_cucumber - 0.2 * cost_cucumber
def cost_bell_pepper : ℝ := cost_cucumber + 0.5 * cost_cucumber
def total_cost_before_discount : ℝ := 2 * cost_tomato + 3 * cost_cucumber + 4 * cost_bell_pepper
def final_price : ℝ := total_cost_before_discount - 0.1 * total_cost_before_discount

theorem final_price_is_correct : final_price = 47.7 := sorry

end NUMINAMATH_GPT_final_price_is_correct_l2285_228594


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l2285_228505

theorem arithmetic_sequence_common_difference (a_1 a_5 d : ℝ) 
  (h1 : a_5 = a_1 + 4 * d) 
  (h2 : a_1 + (a_1 + d) + (a_1 + 2 * d) = 6) : 
  d = 2 := 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l2285_228505


namespace NUMINAMATH_GPT_tenfold_largest_two_digit_number_l2285_228589

def largest_two_digit_number : ℕ := 99

theorem tenfold_largest_two_digit_number :
  10 * largest_two_digit_number = 990 :=
by
  sorry

end NUMINAMATH_GPT_tenfold_largest_two_digit_number_l2285_228589


namespace NUMINAMATH_GPT_ratio_cost_to_marked_price_l2285_228576

theorem ratio_cost_to_marked_price (p : ℝ) (hp : p > 0) :
  let selling_price := (3 / 4) * p
  let cost_price := (5 / 6) * selling_price
  cost_price / p = 5 / 8 :=
by 
  sorry

end NUMINAMATH_GPT_ratio_cost_to_marked_price_l2285_228576


namespace NUMINAMATH_GPT_sum_of_remainders_mod_15_l2285_228527

theorem sum_of_remainders_mod_15 (a b c : ℕ) (h1 : a % 15 = 11) (h2 : b % 15 = 13) (h3 : c % 15 = 14) :
  (a + b + c) % 15 = 8 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_remainders_mod_15_l2285_228527


namespace NUMINAMATH_GPT_no_common_interior_points_l2285_228538

open Metric

-- Define the distance conditions for two convex polygons F1 and F2
variables {F1 F2 : Set (EuclideanSpace ℝ (Fin 2))}

-- F1 is a convex polygon
def is_convex (S : Set (EuclideanSpace ℝ (Fin 2))) : Prop :=
  ∀ {x y : EuclideanSpace ℝ (Fin 2)} {a b : ℝ},
    x ∈ S → y ∈ S → 0 ≤ a → 0 ≤ b → a + b = 1 → a • x + b • y ∈ S

-- Conditions provided in the problem
def condition1 (F : Set (EuclideanSpace ℝ (Fin 2))) : Prop :=
  ∀ {x y : EuclideanSpace ℝ (Fin 2)}, x ∈ F → y ∈ F → dist x y ≤ 1

def condition2 (F1 F2 : Set (EuclideanSpace ℝ (Fin 2))) : Prop :=
  ∀ {x : EuclideanSpace ℝ (Fin 2)} {y : EuclideanSpace ℝ (Fin 2)}, x ∈ F1 → y ∈ F2 → dist x y > 1 / Real.sqrt 2

-- The theorem to prove
theorem no_common_interior_points (h1 : is_convex F1) (h2 : is_convex F2) 
  (h3 : condition1 F1) (h4 : condition1 F2) (h5 : condition2 F1 F2) :
  ∀ p ∈ interior F1, ∀ q ∈ interior F2, p ≠ q :=
sorry

end NUMINAMATH_GPT_no_common_interior_points_l2285_228538


namespace NUMINAMATH_GPT_angles_equal_l2285_228568

theorem angles_equal (α θ γ : Real) (hα : 0 < α ∧ α < π / 2) (hθ : 0 < θ ∧ θ < π / 2) (hγ : 0 < γ ∧ γ < π / 2)
  (h : Real.sin (α + γ) * Real.tan α = Real.sin (θ + γ) * Real.tan θ) : α = θ :=
by
  sorry

end NUMINAMATH_GPT_angles_equal_l2285_228568


namespace NUMINAMATH_GPT_no_valid_partition_of_nat_l2285_228539

-- Definitions of the sets A, B, and C as nonempty subsets of positive integers
variable (A B C : Set ℕ)

-- Definition to capture the key condition in the problem
def valid_partition (A B C : Set ℕ) : Prop :=
  (∀ x ∈ A, ∀ y ∈ B, (x^2 - x * y + y^2) ∈ C) 

-- The main theorem stating that such a partition is impossible
theorem no_valid_partition_of_nat : 
  (∃ A B C : Set ℕ, A ≠ ∅ ∧ B ≠ ∅ ∧ C ≠ ∅ ∧ (∀ x ∈ A, ∀ y ∈ B, (x^2 - x * y + y^2) ∈ C)) → False :=
by
  sorry

end NUMINAMATH_GPT_no_valid_partition_of_nat_l2285_228539


namespace NUMINAMATH_GPT_percentage_cut_away_in_second_week_l2285_228597

theorem percentage_cut_away_in_second_week :
  ∃(x : ℝ), (x / 100) * 142.5 * 0.9 = 109.0125 ∧ x = 15 :=
by
  sorry

end NUMINAMATH_GPT_percentage_cut_away_in_second_week_l2285_228597


namespace NUMINAMATH_GPT_divisible_by_primes_l2285_228513

theorem divisible_by_primes (x y z : ℕ) (hx : x < 10) (hy : y < 10) (hz : z < 10) :
  (100100 * x + 10010 * y + 1001 * z) % 7 = 0 ∧ 
  (100100 * x + 10010 * y + 1001 * z) % 11 = 0 ∧ 
  (100100 * x + 10010 * y + 1001 * z) % 13 = 0 := 
by
  sorry

end NUMINAMATH_GPT_divisible_by_primes_l2285_228513


namespace NUMINAMATH_GPT_problem_part1_29_13_problem_part2_mn_problem_part3_k_36_problem_part4_min_val_l2285_228574

def is_perfect_number (n : ℕ) : Prop :=
  ∃ a b : ℤ, n = a^2 + b^2

theorem problem_part1_29_13 : is_perfect_number 29 ∧ is_perfect_number 13 := by
  sorry

theorem problem_part2_mn : 
  ∃ m n : ℤ, (∀ a : ℤ, a^2 - 4 * a + 8 = (a - m)^2 + n^2) ∧ (m * n = 4 ∨ m * n = -4) := by
  sorry

theorem problem_part3_k_36 (a b : ℤ) : 
  ∃ k : ℤ, (∀ k : ℤ, a^2 + 4*a*b + 5*b^2 - 12*b + k = (a + 2*b)^2 + (b-6)^2) ∧ k = 36 := by
  sorry

theorem problem_part4_min_val (a b : ℝ) : 
  (∀ (a b : ℝ), -a^2 + 5*a + b - 7 = 0 → ∃ a' b', (a + b = (a'-2)^2 + 3) ∧ a' + b' = 3) := by
  sorry

end NUMINAMATH_GPT_problem_part1_29_13_problem_part2_mn_problem_part3_k_36_problem_part4_min_val_l2285_228574


namespace NUMINAMATH_GPT_value_at_minus_two_l2285_228533

def f (x : ℝ) : ℝ := x^2 + 3 * x - 5

theorem value_at_minus_two : f (-2) = -7 := by
  sorry

end NUMINAMATH_GPT_value_at_minus_two_l2285_228533


namespace NUMINAMATH_GPT_sin_45_degree_l2285_228501

noncomputable section

open Real

theorem sin_45_degree : sin (π / 4) = sqrt 2 / 2 := sorry

end NUMINAMATH_GPT_sin_45_degree_l2285_228501


namespace NUMINAMATH_GPT_least_number_to_add_l2285_228598

theorem least_number_to_add (n : ℕ) : 
  (∀ k : ℕ, n = 1 + k * 425 ↔ n + 1019 % 425 = 0) → n = 256 := 
sorry

end NUMINAMATH_GPT_least_number_to_add_l2285_228598


namespace NUMINAMATH_GPT_beaker_water_division_l2285_228529

-- Given conditions
variable (buckets : ℕ) (bucket_capacity : ℕ) (remaining_water : ℝ)
  (total_buckets : ℕ := 2) (capacity : ℕ := 120) (remaining : ℝ := 2.4)

-- Theorem statement
theorem beaker_water_division (h1 : buckets = total_buckets)
                             (h2 : bucket_capacity = capacity)
                             (h3 : remaining_water = remaining) :
                             (total_water : ℝ := buckets * bucket_capacity + remaining_water ) → 
                             (water_per_beaker : ℝ := total_water / 3) →
                             water_per_beaker = 80.8 :=
by
  -- Skipping the proof steps here, will use sorry
  sorry

end NUMINAMATH_GPT_beaker_water_division_l2285_228529


namespace NUMINAMATH_GPT_book_page_count_l2285_228582

theorem book_page_count (pages_per_night : ℝ) (nights : ℝ) : pages_per_night = 120.0 → nights = 10.0 → pages_per_night * nights = 1200.0 :=
by
  sorry

end NUMINAMATH_GPT_book_page_count_l2285_228582


namespace NUMINAMATH_GPT_contest_correct_answers_l2285_228577

/-- 
In a mathematics contest with ten problems, a student gains 
5 points for a correct answer and loses 2 points for an 
incorrect answer. If Olivia answered every problem 
and her score was 29, how many correct answers did she have?
-/
theorem contest_correct_answers (c w : ℕ) (h1 : c + w = 10) (h2 : 5 * c - 2 * w = 29) : c = 7 :=
by 
  sorry

end NUMINAMATH_GPT_contest_correct_answers_l2285_228577


namespace NUMINAMATH_GPT_school_election_votes_l2285_228532

theorem school_election_votes (E S R L : ℕ)
  (h1 : E = 2 * S)
  (h2 : E = 4 * R)
  (h3 : S = 5 * R)
  (h4 : S = 3 * L)
  (h5 : R = 16) :
  E = 64 ∧ S = 80 ∧ R = 16 ∧ L = 27 := by
  sorry

end NUMINAMATH_GPT_school_election_votes_l2285_228532


namespace NUMINAMATH_GPT_workers_time_together_l2285_228547

theorem workers_time_together (T : ℝ) (h1 : ∀ t : ℝ, (T + 8) = t → 1 / t = 1 / (T + 8))
                                (h2 : ∀ t : ℝ, (T + 4.5) = t → 1 / t = 1 / (T + 4.5))
                                (h3 : 1 / (T + 8) + 1 / (T + 4.5) = 1 / T) : T = 6 :=
sorry

end NUMINAMATH_GPT_workers_time_together_l2285_228547


namespace NUMINAMATH_GPT_hyperbola_equation_l2285_228557

-- Lean 4 statement
theorem hyperbola_equation (a b : ℝ) (hpos_a : a > 0) (hpos_b : b > 0)
    (length_imag_axis : 2 * b = 2)
    (asymptote : ∃ (k : ℝ), ∀ x : ℝ, y = k * x ↔ y = (1 / 2) * x) :
  (x y : ℝ) → (x^2 / a^2) - (y^2 / b^2) = 1 ↔ (x^2 / 4) - (y^2 / 1) = 1 :=
by 
  intros
  sorry

end NUMINAMATH_GPT_hyperbola_equation_l2285_228557


namespace NUMINAMATH_GPT_range_of_a_l2285_228570

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2 * Real.exp 1 * x - Real.log x / x + a

theorem range_of_a (a : ℝ) :
  (∃ x > 0, f x a = 0) → a ≤ Real.exp 2 + 1 / Real.exp 1 := by
  sorry

end NUMINAMATH_GPT_range_of_a_l2285_228570


namespace NUMINAMATH_GPT_product_mnp_l2285_228561

theorem product_mnp (m n p : ℕ) (b x z c : ℂ) (h1 : b^8 * x * z - b^7 * z - b^6 * x = b^5 * (c^5 - 1)) 
  (h2 : (b^m * x - b^n) * (b^p * z - b^3) = b^5 * c^5) : m * n * p = 30 :=
sorry

end NUMINAMATH_GPT_product_mnp_l2285_228561


namespace NUMINAMATH_GPT_iterate_g_eq_2_l2285_228516

def g (n : ℕ) : ℕ :=
if n % 2 = 1 then n^2 - 2*n + 2 else 2*n

theorem iterate_g_eq_2 {n : ℕ} (hn : 1 ≤ n ∧ n ≤ 100): 
  (∃ m : ℕ, (Nat.iterate g m n) = 2) ↔ n = 1 :=
by
sorry

end NUMINAMATH_GPT_iterate_g_eq_2_l2285_228516


namespace NUMINAMATH_GPT_f_when_x_lt_4_l2285_228531

noncomputable def f : ℝ → ℝ := sorry

theorem f_when_x_lt_4 (x : ℝ) (h1 : ∀ y : ℝ, y > 4 → f y = 2^(y-1)) (h2 : ∀ y : ℝ, f (4-y) = f (4+y)) (hx : x < 4) : f x = 2^(7-x) :=
by
  sorry

end NUMINAMATH_GPT_f_when_x_lt_4_l2285_228531


namespace NUMINAMATH_GPT_quadratic_expression_l2285_228565

-- Definitions of roots and their properties
def quadratic_roots (r s : ℚ) : Prop :=
  (r + s = 5 / 3) ∧ (r * s = -8 / 3)

theorem quadratic_expression (r s : ℚ) (h : quadratic_roots r s) :
  (9 * r^2 - 9 * s^2) / (r - s) = 15 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_expression_l2285_228565


namespace NUMINAMATH_GPT_sequence_formula_l2285_228595

noncomputable def a (n : ℕ) : ℕ :=
if n = 0 then 1 else (a (n - 1)) + 2^(n-1)

theorem sequence_formula (n : ℕ) (h : n > 0) : 
    a n = 2^n - 1 := 
sorry

end NUMINAMATH_GPT_sequence_formula_l2285_228595


namespace NUMINAMATH_GPT_box_volume_correct_l2285_228572

variables (length width height : ℕ)

def volume_of_box (length width height : ℕ) : ℕ :=
  length * width * height

theorem box_volume_correct :
  volume_of_box 20 15 10 = 3000 :=
by
  -- This is where the proof would go
  sorry 

end NUMINAMATH_GPT_box_volume_correct_l2285_228572


namespace NUMINAMATH_GPT_can_transform_1220_to_2012_cannot_transform_1220_to_2021_l2285_228500

def can_transform (abcd : ℕ) (wxyz : ℕ) : Prop :=
  ∀ a b c d w x y z, 
  abcd = a*1000 + b*100 + c*10 + d ∧ 
  wxyz = w*1000 + x*100 + y*10 + z →
  (∃ (k : ℕ) (m : ℕ), 
    (k = a ∧ a ≠ d  ∧ m = c  ∧ c ≠ w ∧ 
     w = b + (k - b) ∧ x = c + (m - c)) ∨
    (k = w ∧ w ≠ x  ∧ m = y  ∧ y ≠ z ∧ 
     z = a + (k - a) ∧ x = d + (m - d)))
          
theorem can_transform_1220_to_2012 : can_transform 1220 2012 :=
sorry

theorem cannot_transform_1220_to_2021 : ¬ can_transform 1220 2021 :=
sorry

end NUMINAMATH_GPT_can_transform_1220_to_2012_cannot_transform_1220_to_2021_l2285_228500


namespace NUMINAMATH_GPT_no_real_solution_l2285_228559

theorem no_real_solution :
  ¬ ∃ x : ℝ, (3 * x ^ 2 / (x - 2) - (5 * x + 4) / 4 + (10 - 9 * x) / (x - 2) + 2 = 0) :=
sorry

end NUMINAMATH_GPT_no_real_solution_l2285_228559


namespace NUMINAMATH_GPT_fiveLetterWordsWithAtLeastOneVowel_l2285_228558

-- Definitions for the given conditions
def letters : List Char := ['A', 'B', 'C', 'D', 'E', 'F']
def vowels : List Char := ['A', 'E']
def consonants : List Char := ['B', 'C', 'D', 'F']

-- Total number of 5-letter words with no restrictions
def totalWords := 6^5

-- Total number of 5-letter words containing no vowels
def noVowelWords := 4^5

-- Prove that the number of 5-letter words with at least one vowel is 6752
theorem fiveLetterWordsWithAtLeastOneVowel : (totalWords - noVowelWords) = 6752 := by
  sorry

end NUMINAMATH_GPT_fiveLetterWordsWithAtLeastOneVowel_l2285_228558


namespace NUMINAMATH_GPT_minimum_spend_on_boxes_l2285_228566

noncomputable def box_length : ℕ := 20
noncomputable def box_width : ℕ := 20
noncomputable def box_height : ℕ := 12
noncomputable def cost_per_box : ℝ := 0.40
noncomputable def total_volume : ℕ := 2400000

theorem minimum_spend_on_boxes : 
  (total_volume / (box_length * box_width * box_height)) * cost_per_box = 200 :=
by
  sorry

end NUMINAMATH_GPT_minimum_spend_on_boxes_l2285_228566


namespace NUMINAMATH_GPT_sum_of_values_of_m_l2285_228504

-- Define the inequality conditions
def condition1 (x m : ℝ) : Prop := (x - m) / 2 ≥ 0
def condition2 (x : ℝ) : Prop := x + 3 < 3 * (x - 1)

-- Define the equation constraint for y
def fractional_equation (y m : ℝ) : Prop := (3 - y) / (2 - y) + m / (y - 2) = 3

-- Sum function for the values of m
def sum_of_m (m1 m2 m3 : ℝ) : ℝ := m1 + m2 + m3

-- Main theorem
theorem sum_of_values_of_m : sum_of_m 3 (-3) (-1) = -1 := 
by { sorry }

end NUMINAMATH_GPT_sum_of_values_of_m_l2285_228504


namespace NUMINAMATH_GPT_jen_profit_is_960_l2285_228573

def buying_price : ℕ := 80
def selling_price : ℕ := 100
def num_candy_bars_bought : ℕ := 50
def num_candy_bars_sold : ℕ := 48

def profit_per_candy_bar := selling_price - buying_price
def total_profit := profit_per_candy_bar * num_candy_bars_sold

theorem jen_profit_is_960 : total_profit = 960 := by
  sorry

end NUMINAMATH_GPT_jen_profit_is_960_l2285_228573


namespace NUMINAMATH_GPT_algebraic_expression_value_l2285_228578

variable (a : ℝ)

theorem algebraic_expression_value (h : a = Real.sqrt 2) :
  (a / (a - 1)^2) / (1 + 1 / (a - 1)) = Real.sqrt 2 + 1 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l2285_228578


namespace NUMINAMATH_GPT_find_ellipse_l2285_228591

noncomputable def standard_equation_ellipse (x y : ℝ) : Prop :=
  (x^2 / 9 + y^2 / 3 = 1)
  ∨ (x^2 / 18 + y^2 / 9 = 1)
  ∨ (y^2 / (45 / 2) + x^2 / (45 / 4) = 1)

variables 
  (P1 P2 : ℝ × ℝ)
  (focus : ℝ × ℝ)
  (a b : ℝ)

def passes_through_points (P1 P2 : ℝ × ℝ) : Prop :=
  ∀ equation : (ℝ → ℝ → Prop), 
    equation P1.1 P1.2 ∧ equation P2.1 P2.2

def focus_conditions (focus : ℝ × ℝ) : Prop :=
  -- Condition indicating focus, relationship with the minor axis etc., will be precisely defined here
  true -- Placeholder, needs correct mathematical condition

theorem find_ellipse : 
  passes_through_points P1 P2 
  → focus_conditions focus 
  → standard_equation_ellipse x y :=
sorry

end NUMINAMATH_GPT_find_ellipse_l2285_228591


namespace NUMINAMATH_GPT_part1_part2_l2285_228502

noncomputable def f (m x : ℝ) : ℝ := m - |x - 1| - |x + 1|

theorem part1 (x : ℝ) : -3 / 2 < x ∧ x < 3 / 2 ↔ f 5 x > 2 := by
  sorry

theorem part2 (m : ℝ) : (∀ x : ℝ, ∃ y : ℝ, x^2 + 2 * x + 3 = f m y) ↔ 4 ≤ m := by
  sorry

end NUMINAMATH_GPT_part1_part2_l2285_228502


namespace NUMINAMATH_GPT_landA_area_and_ratio_l2285_228569

/-
  a = 3, b = 5, c = 6
  p = 1/2 * (a + b + c)
  S = sqrt(p * (p - a) * (p - b) * (p - c))
  S_A = 2 * sqrt(14)
  S_B = 3/2 * sqrt(14)
  S_A / S_B = 4 / 3
-/
theorem landA_area_and_ratio :
  let a := 3
  let b := 5
  let c := 6
  let p := (a + b + c) / 2
  let S_A := Real.sqrt (p * (p - a) * (p - b) * (p - c))
  let S_B := 3 / 2 * Real.sqrt 14
  S_A = 2 * Real.sqrt 14 ∧ S_A / S_B = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_landA_area_and_ratio_l2285_228569


namespace NUMINAMATH_GPT_winning_margin_l2285_228546

theorem winning_margin (total_votes : ℝ) (winning_votes : ℝ) (winning_percent : ℝ) (losing_percent : ℝ) 
  (win_votes_eq: winning_votes = winning_percent * total_votes)
  (perc_eq: winning_percent + losing_percent = 1)
  (win_votes_given: winning_votes = 550)
  (winning_percent_given: winning_percent = 0.55)
  (losing_percent_given: losing_percent = 0.45) :
  winning_votes - (losing_percent * total_votes) = 100 := 
by
  sorry

end NUMINAMATH_GPT_winning_margin_l2285_228546


namespace NUMINAMATH_GPT_exists_xyz_l2285_228575

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem exists_xyz :
  ∃ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧
  (x + sum_of_digits x = y + sum_of_digits y ∧ y + sum_of_digits y = z + sum_of_digits z) :=
by {
  sorry
}

end NUMINAMATH_GPT_exists_xyz_l2285_228575


namespace NUMINAMATH_GPT_sum_of_possible_values_of_k_l2285_228584

theorem sum_of_possible_values_of_k :
  (∀ j k : ℕ, 0 < j ∧ 0 < k → (1 / (j:ℚ)) + (1 / (k:ℚ)) = (1 / 5) → k = 6 ∨ k = 10 ∨ k = 30) ∧ 
  (46 = 6 + 10 + 30) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_possible_values_of_k_l2285_228584


namespace NUMINAMATH_GPT_Jenny_recycling_l2285_228521

theorem Jenny_recycling:
  let bottle_weight := 6
  let can_weight := 2
  let glass_jar_weight := 8
  let max_weight := 100
  let num_cans := 20
  let bottle_value := 10
  let can_value := 3
  let glass_jar_value := 12
  let total_money := (num_cans * can_value) + (7 * glass_jar_value) + (0 * bottle_value)
  total_money = 144 ∧ num_cans = 20 ∧ glass_jars = 7 ∧ bottles = 0 := by sorry

end NUMINAMATH_GPT_Jenny_recycling_l2285_228521


namespace NUMINAMATH_GPT_y_gets_per_rupee_l2285_228599

theorem y_gets_per_rupee (a p : ℝ) (ha : a * p = 63) (htotal : p + a * p + 0.3 * p = 245) : a = 0.63 :=
by
  sorry

end NUMINAMATH_GPT_y_gets_per_rupee_l2285_228599


namespace NUMINAMATH_GPT_find_a_and_solve_inequality_l2285_228544

theorem find_a_and_solve_inequality :
  (∀ x : ℝ, |x^2 - 4 * x + a| + |x - 3| ≤ 5 → x ≤ 3) →
  a = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_a_and_solve_inequality_l2285_228544


namespace NUMINAMATH_GPT_staircase_problem_l2285_228563

theorem staircase_problem :
  ∃ (n : ℕ), (n > 20) ∧ (n % 5 = 4) ∧ (n % 6 = 3) ∧ (n % 7 = 5) ∧ n = 159 :=
by sorry

end NUMINAMATH_GPT_staircase_problem_l2285_228563


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l2285_228550

def A : Set ℝ := {x : ℝ | -1 < x ∧ x ≤ 2}
def B : Set ℝ := {-1, 0, 1}
def intersection : Set ℝ := {0, 1}

theorem intersection_of_A_and_B : A ∩ B = intersection := 
by sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l2285_228550


namespace NUMINAMATH_GPT_value_of_f_2012_l2285_228526

def odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

noncomputable def f : ℝ → ℝ := sorry

axiom odd_fn : odd_function f
axiom f_at_2 : f 2 = 0
axiom functional_eq : ∀ x : ℝ, f (x + 4) = f x + f 4

theorem value_of_f_2012 : f 2012 = 0 :=
by
  sorry

end NUMINAMATH_GPT_value_of_f_2012_l2285_228526


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_for_q_l2285_228536

def proposition_p (a : ℝ) := (1 / a) > (1 / 4)
def proposition_q (a : ℝ) := ∀ x : ℝ, (a * x^2 + a * x + 1) > 0

theorem sufficient_but_not_necessary_condition_for_q (a : ℝ) :
  proposition_p a → proposition_q a → (∃ a : ℝ, 0 < a ∧ a < 4) ∧ (∃ a : ℝ, 0 < a ∧ a < 4 ∧ ¬ proposition_p a) 
  := sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_for_q_l2285_228536


namespace NUMINAMATH_GPT_find_f_sqrt2_l2285_228520

noncomputable def f : ℝ → ℝ := sorry

axiom f_domain : ∀ x, x > 0 → (∃ y, f y = x ∨ y = x)

axiom f_multiplicative : ∀ x y : ℝ, x > 0 → y > 0 → f (x * y) = f x + f y
axiom f_at_8 : f 8 = 3

-- Define the problem statement
theorem find_f_sqrt2 : f (Real.sqrt 2) = 1 / 2 := sorry

end NUMINAMATH_GPT_find_f_sqrt2_l2285_228520


namespace NUMINAMATH_GPT_smaller_solution_of_quadratic_eq_l2285_228542

theorem smaller_solution_of_quadratic_eq : 
  (exists x y : ℝ, x < y ∧ x^2 - 13 * x + 36 = 0 ∧ y^2 - 13 * y + 36 = 0 ∧ x = 4) :=
by sorry

end NUMINAMATH_GPT_smaller_solution_of_quadratic_eq_l2285_228542


namespace NUMINAMATH_GPT_greatest_matching_pairs_left_l2285_228508

-- Define the initial number of pairs and lost individual shoes
def initial_pairs : ℕ := 26
def lost_ind_shoes : ℕ := 9

-- The statement to be proved
theorem greatest_matching_pairs_left : 
  (initial_pairs * 2 - lost_ind_shoes) / 2 + (initial_pairs - (initial_pairs * 2 - lost_ind_shoes) / 2) / 1 = 17 := 
by 
  sorry

end NUMINAMATH_GPT_greatest_matching_pairs_left_l2285_228508


namespace NUMINAMATH_GPT_find_X_l2285_228519

theorem find_X (X : ℚ) (h : (1/3 : ℚ) * (1/4 : ℚ) * X = (1/4 : ℚ) * (1/6 : ℚ) * 120) : X = 60 := 
sorry

end NUMINAMATH_GPT_find_X_l2285_228519


namespace NUMINAMATH_GPT_lunch_break_duration_l2285_228523

theorem lunch_break_duration (m a : ℝ) (L : ℝ) :
  (9 - L) * (m + a) = 0.6 → 
  (7 - L) * a = 0.3 → 
  (5 - L) * m = 0.1 → 
  L = 42 / 60 :=
by sorry

end NUMINAMATH_GPT_lunch_break_duration_l2285_228523


namespace NUMINAMATH_GPT_area_of_triangle_8_9_9_l2285_228562

noncomputable def triangle_area (a b c : ℕ) : Real :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem area_of_triangle_8_9_9 : triangle_area 8 9 9 = 4 * Real.sqrt 65 :=
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_8_9_9_l2285_228562


namespace NUMINAMATH_GPT_ruth_started_with_89_apples_l2285_228522

theorem ruth_started_with_89_apples 
  (initial_apples : ℕ)
  (shared_apples : ℕ)
  (remaining_apples : ℕ)
  (h1 : shared_apples = 5)
  (h2 : remaining_apples = 84)
  (h3 : remaining_apples = initial_apples - shared_apples) : 
  initial_apples = 89 :=
by
  sorry

end NUMINAMATH_GPT_ruth_started_with_89_apples_l2285_228522


namespace NUMINAMATH_GPT_stingrays_count_l2285_228545

theorem stingrays_count (Sh S : ℕ) (h1 : Sh = 2 * S) (h2 : S + Sh = 84) : S = 28 :=
by
  -- Proof will be filled here
  sorry

end NUMINAMATH_GPT_stingrays_count_l2285_228545


namespace NUMINAMATH_GPT_find_a_l2285_228509

theorem find_a (a : ℝ) (A B : ℝ × ℝ × ℝ) (hA : A = (-1, 1, -a)) (hB : B = (-a, 3, -1)) (hAB : dist A B = 2) : a = -1 := by
  sorry

end NUMINAMATH_GPT_find_a_l2285_228509


namespace NUMINAMATH_GPT_bus_driver_total_hours_l2285_228585

theorem bus_driver_total_hours
  (reg_rate : ℝ := 16)
  (ot_rate : ℝ := 28)
  (total_hours : ℝ)
  (total_compensation : ℝ := 920)
  (h : total_compensation = reg_rate * 40 + ot_rate * (total_hours - 40)) :
  total_hours = 50 := 
by 
  sorry

end NUMINAMATH_GPT_bus_driver_total_hours_l2285_228585


namespace NUMINAMATH_GPT_find_c_l2285_228507

variable (x y c : ℝ)

def condition1 : Prop := 2 * x + 5 * y = 3
def condition2 : Prop := c = Real.sqrt (4^(x + 1/2) * 32^y)

theorem find_c (h1 : condition1 x y) (h2 : condition2 x y c) : c = 4 := by
  sorry

end NUMINAMATH_GPT_find_c_l2285_228507


namespace NUMINAMATH_GPT_cupcakes_total_l2285_228549

theorem cupcakes_total (initially_made : ℕ) (sold : ℕ) (newly_made : ℕ) (initially_made_eq : initially_made = 42) (sold_eq : sold = 22) (newly_made_eq : newly_made = 39) : initially_made - sold + newly_made = 59 :=
by
  sorry

end NUMINAMATH_GPT_cupcakes_total_l2285_228549


namespace NUMINAMATH_GPT_ratio_of_triangles_in_octagon_l2285_228596

-- Conditions
def regular_octagon_division : Prop := 
  let L := 1 -- Area of each small congruent right triangle
  let ABJ := 2 * L -- Area of triangle ABJ
  let ADE := 6 * L -- Area of triangle ADE
  (ABJ / ADE = (1:ℝ) / 3)

-- Statement
theorem ratio_of_triangles_in_octagon : regular_octagon_division := by
  sorry

end NUMINAMATH_GPT_ratio_of_triangles_in_octagon_l2285_228596


namespace NUMINAMATH_GPT_solve_for_a_l2285_228528

theorem solve_for_a (f : ℝ → ℝ) (a : ℝ)
  (h1 : ∀ x, f (x^3) = Real.log x / Real.log a)
  (h2 : f 8 = 1) :
  a = 2 :=
sorry

end NUMINAMATH_GPT_solve_for_a_l2285_228528


namespace NUMINAMATH_GPT_gcd_of_polynomials_l2285_228524

theorem gcd_of_polynomials (b : ℤ) (h : ∃ k : ℤ, b = 2 * 5959 * k) :
  Int.gcd (4 * b^2 + 73 * b + 156) (4 * b + 15) = 1 :=
by
  sorry

end NUMINAMATH_GPT_gcd_of_polynomials_l2285_228524


namespace NUMINAMATH_GPT_factorial_division_l2285_228535

open Nat

theorem factorial_division : 12! / 11! = 12 := sorry

end NUMINAMATH_GPT_factorial_division_l2285_228535


namespace NUMINAMATH_GPT_initial_eggs_count_l2285_228583

theorem initial_eggs_count (harry_adds : ℕ) (total_eggs : ℕ) (initial_eggs : ℕ) :
  harry_adds = 5 → total_eggs = 52 → initial_eggs = total_eggs - harry_adds → initial_eggs = 47 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_initial_eggs_count_l2285_228583


namespace NUMINAMATH_GPT_distance_between_feet_of_perpendiculars_eq_area_over_radius_l2285_228530
noncomputable def area (ABC : Type) : ℝ := sorry
noncomputable def circumradius (ABC : Type) : ℝ := sorry

theorem distance_between_feet_of_perpendiculars_eq_area_over_radius
  (ABC : Type)
  (area_ABC : ℝ)
  (R : ℝ)
  (h_area : area ABC = area_ABC)
  (h_radius : circumradius ABC = R) :
  ∃ (m : ℝ), m = area_ABC / R := sorry

end NUMINAMATH_GPT_distance_between_feet_of_perpendiculars_eq_area_over_radius_l2285_228530


namespace NUMINAMATH_GPT_lines_through_same_quadrants_l2285_228537

theorem lines_through_same_quadrants (k b : ℝ) (hk : k ≠ 0):
    ∃ n, n ≥ 7 ∧ ∀ (f : Fin n → ℝ × ℝ), ∃ (i j : Fin n), i ≠ j ∧ 
    ((f i).1 > 0 ∧ (f j).1 > 0 ∧ (f i).2 > 0 ∧ (f j).2 > 0 ∨
     (f i).1 > 0 ∧ (f j).1 > 0 ∧ (f i).2 < 0 ∧ (f j).2 < 0 ∨
     (f i).1 > 0 ∧ (f j).1 > 0 ∧ (f i).2 = 0 ∧ (f j).2 = 0 ∨
     (f i).1 < 0 ∧ (f j).1 < 0 ∧ (f i).2 > 0 ∧ (f j).2 > 0 ∨
     (f i).1 < 0 ∧ (f j).1 < 0 ∧ (f i).2 < 0 ∧ (f j).2 < 0 ∨
     (f i).1 < 0 ∧ (f j).1 < 0 ∧ (f i).2 = 0 ∧ (f j).2 = 0) :=
by sorry

end NUMINAMATH_GPT_lines_through_same_quadrants_l2285_228537


namespace NUMINAMATH_GPT_gain_percentage_is_five_percent_l2285_228554

variables (CP SP New_SP Loss Loss_Percentage Gain Gain_Percentage : ℝ)
variables (H1 : Loss_Percentage = 10)
variables (H2 : CP = 933.33)
variables (H3 : Loss = (Loss_Percentage / 100) * CP)
variables (H4 : SP = CP - Loss)
variables (H5 : New_SP = SP + 140)
variables (H6 : Gain = New_SP - CP)
variables (H7 : Gain_Percentage = (Gain / CP) * 100)

theorem gain_percentage_is_five_percent :
  Gain_Percentage = 5 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_gain_percentage_is_five_percent_l2285_228554


namespace NUMINAMATH_GPT_cargo_to_passenger_ratio_l2285_228587

def total_cars : Nat := 71
def passenger_cars : Nat := 44
def engine_and_caboose : Nat := 2
def cargo_cars : Nat := total_cars - passenger_cars - engine_and_caboose

theorem cargo_to_passenger_ratio : cargo_cars = 25 ∧ passenger_cars = 44 →
  cargo_cars.toFloat / passenger_cars.toFloat = 25.0 / 44.0 :=
by
  intros h
  rw [h.1]
  rw [h.2]
  sorry

end NUMINAMATH_GPT_cargo_to_passenger_ratio_l2285_228587


namespace NUMINAMATH_GPT_range_of_f_on_interval_l2285_228556

-- Definition of the function
def f (x : ℝ) : ℝ := x^2 - 2 * x + 2

-- Definition of the interval
def domain (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 2

-- The main statement
theorem range_of_f_on_interval : 
  ∀ y, (∃ x, domain x ∧ f x = y) ↔ (1 ≤ y ∧ y ≤ 10) :=
by
  sorry

end NUMINAMATH_GPT_range_of_f_on_interval_l2285_228556


namespace NUMINAMATH_GPT_first_player_can_ensure_distinct_rational_roots_l2285_228592

theorem first_player_can_ensure_distinct_rational_roots :
  ∃ (a b c : ℚ), a + b + c = 0 ∧ (∀ x : ℚ, x^2 + (b/a) * x + (c/a) = 0 → False) :=
by
  sorry

end NUMINAMATH_GPT_first_player_can_ensure_distinct_rational_roots_l2285_228592


namespace NUMINAMATH_GPT_total_lives_l2285_228512

-- Definitions of given conditions
def original_friends : Nat := 2
def lives_per_player : Nat := 6
def additional_players : Nat := 2

-- Proof statement to show the total number of lives
theorem total_lives :
  (original_friends * lives_per_player) + (additional_players * lives_per_player) = 24 := by
  sorry

end NUMINAMATH_GPT_total_lives_l2285_228512


namespace NUMINAMATH_GPT_apples_not_sold_correct_l2285_228553

-- Define the constants and conditions
def boxes_ordered_per_week : ℕ := 10
def apples_per_box : ℕ := 300
def fraction_sold : ℚ := 3 / 4

-- Define the total number of apples ordered in a week
def total_apples_ordered : ℕ := boxes_ordered_per_week * apples_per_box

-- Define the total number of apples sold in a week
def apples_sold : ℚ := fraction_sold * total_apples_ordered

-- Define the total number of apples not sold in a week
def apples_not_sold : ℚ := total_apples_ordered - apples_sold

-- Lean statement to prove the total number of apples not sold is 750
theorem apples_not_sold_correct :
  apples_not_sold = 750 := 
sorry

end NUMINAMATH_GPT_apples_not_sold_correct_l2285_228553


namespace NUMINAMATH_GPT_payment_first_trip_payment_second_trip_l2285_228517

-- Define conditions and questions
variables {x y : ℝ}

-- Conditions: discounts and expenditure
def discount_1st_trip (x : ℝ) := 0.9 * x
def discount_2nd_trip (y : ℝ) := 300 * 0.9 + (y - 300) * 0.8

def combined_discount (x y : ℝ) := 300 * 0.9 + (x + y - 300) * 0.8

-- Given conditions as equations
axiom eq1 : discount_1st_trip x + discount_2nd_trip y - combined_discount x y = 19
axiom eq2 : x + y - (discount_1st_trip x + discount_2nd_trip y) = 67

-- The proof statements
theorem payment_first_trip : discount_1st_trip 190 = 171 := by sorry

theorem payment_second_trip : discount_2nd_trip 390 = 342 := by sorry

end NUMINAMATH_GPT_payment_first_trip_payment_second_trip_l2285_228517


namespace NUMINAMATH_GPT_problem1_problem2_l2285_228515

-- Definitions for conditions
def p (x : ℝ) : Prop := x^2 - 7 * x + 10 < 0
def q (x : ℝ) (m : ℝ) : Prop := x^2 - 4 * m * x + 3 * m^2 < 0

-- Problem 1: For m = 4, p ∧ q implies 4 < x < 5
theorem problem1 (x : ℝ) (h : 4 < x ∧ x < 5) : 
  p x ∧ q x 4 :=
sorry

-- Problem 2: ∃ m, m > 0, m ≤ 2, and 3m ≥ 5 implies (5/3 ≤ m ≤ 2)
theorem problem2 (m : ℝ) (h1 : m > 0) (h2 : m ≤ 2) (h3 : 3 * m ≥ 5) : 
  5 / 3 ≤ m ∧ m ≤ 2 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l2285_228515


namespace NUMINAMATH_GPT_find_a4_l2285_228514

theorem find_a4 (a : ℕ → ℕ) 
  (h1 : ∀ n, (a n + 1) / (a (n + 1) + 1) = 1 / 2) 
  (h2 : a 2 = 2) : 
  a 4 = 11 :=
sorry

end NUMINAMATH_GPT_find_a4_l2285_228514


namespace NUMINAMATH_GPT_pears_sold_in_afternoon_l2285_228588

theorem pears_sold_in_afternoon (m a total : ℕ) (h1 : a = 2 * m) (h2 : m = 120) (h3 : m + a = total) (h4 : total = 360) :
  a = 240 :=
by
  sorry

end NUMINAMATH_GPT_pears_sold_in_afternoon_l2285_228588


namespace NUMINAMATH_GPT_simplify_fraction_l2285_228518

theorem simplify_fraction (b : ℕ) (hb : b = 2) : (15 * b ^ 4) / (45 * b ^ 3) = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l2285_228518


namespace NUMINAMATH_GPT_swan_populations_after_10_years_l2285_228534

noncomputable def swan_population_rita (R : ℝ) : ℝ :=
  480 * (1 - R / 100) ^ 10

noncomputable def swan_population_sarah (S : ℝ) : ℝ :=
  640 * (1 - S / 100) ^ 10

noncomputable def swan_population_tom (T : ℝ) : ℝ :=
  800 * (1 - T / 100) ^ 10

theorem swan_populations_after_10_years 
  (R S T : ℝ) :
  swan_population_rita R = 480 * (1 - R / 100) ^ 10 ∧
  swan_population_sarah S = 640 * (1 - S / 100) ^ 10 ∧
  swan_population_tom T = 800 * (1 - T / 100) ^ 10 := 
by sorry

end NUMINAMATH_GPT_swan_populations_after_10_years_l2285_228534


namespace NUMINAMATH_GPT_maximum_value_attains_maximum_value_l2285_228590

theorem maximum_value
  (a b c : ℝ)
  (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c)
  (h₃ : a + b + c = 1) :
  (ab / (a + b)) + (ac / (a + c)) + (bc / (b + c)) ≤ 1 / 2 :=
sorry

theorem attains_maximum_value :
  ∃ (a b c : ℝ), 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a + b + c = 1 ∧
  (ab / (a + b)) + (ac / (a + c)) + (bc / (b + c)) = 1 / 2 :=
sorry

end NUMINAMATH_GPT_maximum_value_attains_maximum_value_l2285_228590


namespace NUMINAMATH_GPT_number_of_five_digit_numbers_without_repeating_digits_with_two_adjacent_odds_is_72_l2285_228593

def five_digit_number_count : Nat :=
  -- Number of ways to select and arrange odd digits in two groups
  let group_odd_digits := (Nat.choose 3 2) * (Nat.factorial 2)
  -- Number of ways to arrange the even digits
  let arrange_even_digits := Nat.factorial 2
  -- Number of ways to insert two groups of odd digits into the gaps among even digits
  let insert_odd_groups := (Nat.factorial 3)
  -- Total ways
  group_odd_digits * arrange_even_digits * arrange_even_digits * insert_odd_groups

theorem number_of_five_digit_numbers_without_repeating_digits_with_two_adjacent_odds_is_72 :
  five_digit_number_count = 72 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_number_of_five_digit_numbers_without_repeating_digits_with_two_adjacent_odds_is_72_l2285_228593


namespace NUMINAMATH_GPT_scientific_notation_of_100000000_l2285_228571

theorem scientific_notation_of_100000000 :
  100000000 = 1 * 10^8 :=
sorry

end NUMINAMATH_GPT_scientific_notation_of_100000000_l2285_228571


namespace NUMINAMATH_GPT_tan_monotonic_increasing_interval_l2285_228567

noncomputable def monotonic_increasing_interval (k : ℤ) : Set ℝ :=
  {x : ℝ | 2 * k * Real.pi - (5 * Real.pi) / 3 < x ∧ x < 2 * k * Real.pi + Real.pi / 3 }

theorem tan_monotonic_increasing_interval (k : ℤ) :
  ∀ x : ℝ, (y = Real.tan ((x / 2) + (Real.pi / 3))) → 
           x ∈ monotonic_increasing_interval k :=
sorry

end NUMINAMATH_GPT_tan_monotonic_increasing_interval_l2285_228567


namespace NUMINAMATH_GPT_solution_set_of_inequality_l2285_228525

theorem solution_set_of_inequality :
  { x : ℝ | ∃ (h : x ≠ 1), 1 / (x - 1) ≥ -1 } = { x : ℝ | x ≤ 0 ∨ 1 < x } :=
by sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l2285_228525


namespace NUMINAMATH_GPT_math_problem_l2285_228503

open Real

variable (x : ℝ)
variable (h : x + 1 / x = sqrt 3)

theorem math_problem : x^7 - 3 * x^5 + x^2 = -5 * x + 4 * sqrt 3 :=
by sorry

end NUMINAMATH_GPT_math_problem_l2285_228503


namespace NUMINAMATH_GPT_find_a_l2285_228586

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a * x^2 + x
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := 3 * x^2 + 2 * a * x + 1

theorem find_a (a : ℝ) : f' a 1 = 6 → a = 1 :=
by
  intro h
  have h_f_prime : 3 * (1 : ℝ) ^ 2 + 2 * a * (1 : ℝ) + 1 = 6 := h
  sorry

end NUMINAMATH_GPT_find_a_l2285_228586


namespace NUMINAMATH_GPT_reciprocal_of_neg_5_l2285_228580

theorem reciprocal_of_neg_5 : (∃ r : ℚ, -5 * r = 1) ∧ r = -1 / 5 :=
by sorry

end NUMINAMATH_GPT_reciprocal_of_neg_5_l2285_228580


namespace NUMINAMATH_GPT_maia_daily_client_requests_l2285_228511

theorem maia_daily_client_requests (daily_requests : ℕ) (remaining_requests : ℕ) (days : ℕ) 
  (received_requests : ℕ) (total_requests : ℕ) (worked_requests : ℕ) :
  (daily_requests = 6) →
  (remaining_requests = 10) →
  (days = 5) →
  (received_requests = daily_requests * days) →
  (total_requests = received_requests - remaining_requests) →
  (worked_requests = total_requests / days) →
  worked_requests = 4 :=
by
  sorry

end NUMINAMATH_GPT_maia_daily_client_requests_l2285_228511


namespace NUMINAMATH_GPT_box_surface_area_correct_l2285_228540

-- Define the dimensions of the original cardboard.
def original_length : ℕ := 25
def original_width : ℕ := 40

-- Define the size of the squares removed from each corner.
def square_side : ℕ := 8

-- Define the surface area function.
def surface_area (length width : ℕ) (square_side : ℕ) : ℕ :=
  let area_remaining := (length * width) - 4 * (square_side * square_side)
  area_remaining

-- The theorem statement to prove
theorem box_surface_area_correct : surface_area original_length original_width square_side = 744 :=
by
  sorry

end NUMINAMATH_GPT_box_surface_area_correct_l2285_228540


namespace NUMINAMATH_GPT_hypotenuse_length_l2285_228543

theorem hypotenuse_length (a b c : ℝ) (h_right_angled : c^2 = a^2 + b^2) (h_sum_of_squares : a^2 + b^2 + c^2 = 2500) : c = 25 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_hypotenuse_length_l2285_228543


namespace NUMINAMATH_GPT_total_plates_l2285_228506

-- Define the initial conditions
def flower_plates_initial : ℕ := 4
def checked_plates : ℕ := 8
def polka_dotted_plates := 2 * checked_plates
def flower_plates_remaining := flower_plates_initial - 1

-- Prove the total number of plates Jack has left
theorem total_plates : flower_plates_remaining + polka_dotted_plates + checked_plates = 27 :=
by
  sorry

end NUMINAMATH_GPT_total_plates_l2285_228506


namespace NUMINAMATH_GPT_frequencies_of_first_class_products_confidence_in_difference_of_quality_l2285_228560

theorem frequencies_of_first_class_products 
  (aA bA aB bB : ℕ) 
  (total_A total_B total_products : ℕ) 
  (hA : aA = 150) (hB : bA = 50) (hC : aB = 120) (hD : bB = 80) 
  (hE : total_A = 200) (hF : total_B = 200) (h_total : total_products = total_A + total_B) : 
  aA / total_A = 3 / 4 ∧ aB / total_B = 3 / 5 := 
by
  sorry

theorem confidence_in_difference_of_quality 
  (n a b c d : ℕ) 
  (h_n : n = 400) (hA : a = 150) (hB : b = 50) (hC : c = 120) (hD : d = 80) 
  (K2 : ℝ) 
  (hK2 : K2 = n * ((a * d - b * c) ^ 2) / ((a + b) * (c + d) * (a + c) * (b + d))) : 
  K2 > 6.635 ∧ K2 < 10.828 := 
by
  sorry

end NUMINAMATH_GPT_frequencies_of_first_class_products_confidence_in_difference_of_quality_l2285_228560


namespace NUMINAMATH_GPT_octadecagon_identity_l2285_228510

theorem octadecagon_identity (a r : ℝ) (h : a = 2 * r * Real.sin (π / 18)) :
  a^3 + r^3 = 3 * r^2 * a :=
sorry

end NUMINAMATH_GPT_octadecagon_identity_l2285_228510


namespace NUMINAMATH_GPT_labourer_saving_after_debt_clearance_l2285_228552

variable (averageExpenditureFirst6Months : ℕ)
variable (monthlyIncome : ℕ)
variable (reducedMonthlyExpensesNext4Months : ℕ)

theorem labourer_saving_after_debt_clearance (h1 : averageExpenditureFirst6Months = 90)
                                              (h2 : monthlyIncome = 81)
                                              (h3 : reducedMonthlyExpensesNext4Months = 60) :
    (monthlyIncome * 4) - ((reducedMonthlyExpensesNext4Months * 4) + 
    ((averageExpenditureFirst6Months * 6) - (monthlyIncome * 6))) = 30 := by
  sorry

end NUMINAMATH_GPT_labourer_saving_after_debt_clearance_l2285_228552
