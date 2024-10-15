import Mathlib

namespace NUMINAMATH_CALUDE_max_inequality_value_zero_is_max_l2258_225811

theorem max_inequality_value (x : ℝ) (h : x > -1) : x + 1 / (x + 1) - 1 ≥ 0 :=
sorry

theorem zero_is_max (a : ℝ) (h : ∀ x > -1, x + 1 / (x + 1) - 1 ≥ a) : a ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_max_inequality_value_zero_is_max_l2258_225811


namespace NUMINAMATH_CALUDE_readers_both_sf_and_lit_l2258_225871

/-- Represents the number of readers who read both science fiction and literary works. -/
def readers_both (total readers_sf readers_lit : ℕ) : ℕ :=
  readers_sf + readers_lit - total

/-- 
Given a group of 400 readers, where 250 read science fiction and 230 read literary works,
proves that 80 readers read both science fiction and literary works.
-/
theorem readers_both_sf_and_lit : 
  readers_both 400 250 230 = 80 := by
  sorry

end NUMINAMATH_CALUDE_readers_both_sf_and_lit_l2258_225871


namespace NUMINAMATH_CALUDE_sqrt_three_plus_sqrt_two_times_sqrt_three_minus_sqrt_two_l2258_225817

theorem sqrt_three_plus_sqrt_two_times_sqrt_three_minus_sqrt_two : 
  (Real.sqrt 3 + Real.sqrt 2) * (Real.sqrt 3 - Real.sqrt 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_plus_sqrt_two_times_sqrt_three_minus_sqrt_two_l2258_225817


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2258_225825

theorem simplify_and_evaluate (x : ℝ) (h : x = 4) : 
  (1 + 1 / (x + 1)) * ((x + 1) / (x^2 + 4*x + 4)) = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2258_225825


namespace NUMINAMATH_CALUDE_rice_mixture_price_l2258_225826

/-- Given two types of rice mixed together, prove the price of the first type --/
theorem rice_mixture_price (price2 : ℚ) (weight1 weight2 : ℚ) (mixture_price : ℚ) 
  (h1 : price2 = 960 / 100)  -- Rs. 9.60 converted to a rational number
  (h2 : weight1 = 49)
  (h3 : weight2 = 56)
  (h4 : mixture_price = 820 / 100)  -- Rs. 8.20 converted to a rational number
  : ∃ (price1 : ℚ), price1 = 660 / 100 ∧  -- Rs. 6.60 converted to a rational number
    (weight1 * price1 + weight2 * price2) / (weight1 + weight2) = mixture_price :=
by sorry

end NUMINAMATH_CALUDE_rice_mixture_price_l2258_225826


namespace NUMINAMATH_CALUDE_tan_plus_pi_fourth_implies_cos_double_l2258_225843

theorem tan_plus_pi_fourth_implies_cos_double (θ : Real) : 
  Real.tan (θ + Real.pi / 4) = 3 → Real.cos (2 * θ) = 3 / 5 :=
by sorry

end NUMINAMATH_CALUDE_tan_plus_pi_fourth_implies_cos_double_l2258_225843


namespace NUMINAMATH_CALUDE_blue_balls_removed_l2258_225860

theorem blue_balls_removed (total_initial : ℕ) (blue_initial : ℕ) (prob_after : ℚ) : 
  total_initial = 18 → 
  blue_initial = 6 → 
  prob_after = 1/5 → 
  ∃ (removed : ℕ), 
    removed ≤ blue_initial ∧ 
    (blue_initial - removed : ℚ) / (total_initial - removed : ℚ) = prob_after ∧
    removed = 3 := by
  sorry

end NUMINAMATH_CALUDE_blue_balls_removed_l2258_225860


namespace NUMINAMATH_CALUDE_blue_segments_count_l2258_225893

/-- Represents a 10x10 grid with colored points and segments -/
structure ColoredGrid :=
  (red_points : Nat)
  (red_corners : Nat)
  (red_edges : Nat)
  (green_segments : Nat)

/-- Calculates the number of blue segments in the grid -/
def blue_segments (grid : ColoredGrid) : Nat :=
  -- The actual calculation would go here
  sorry

/-- Theorem stating the number of blue segments in the given conditions -/
theorem blue_segments_count (grid : ColoredGrid) :
  grid.red_points = 52 →
  grid.red_corners = 2 →
  grid.red_edges = 16 →
  grid.green_segments = 98 →
  blue_segments grid = 37 := by
  sorry

end NUMINAMATH_CALUDE_blue_segments_count_l2258_225893


namespace NUMINAMATH_CALUDE_population_percentage_l2258_225845

theorem population_percentage (men women : ℝ) (h : women = 0.9 * men) :
  (men / women) * 100 = (1 / 0.9) * 100 := by
  sorry

end NUMINAMATH_CALUDE_population_percentage_l2258_225845


namespace NUMINAMATH_CALUDE_wax_needed_l2258_225809

def total_wax : ℕ := 166
def current_wax : ℕ := 20

theorem wax_needed : total_wax - current_wax = 146 := by
  sorry

end NUMINAMATH_CALUDE_wax_needed_l2258_225809


namespace NUMINAMATH_CALUDE_divisible_by_eleven_l2258_225827

theorem divisible_by_eleven (a : ℝ) : 
  (∃ k : ℤ, (2 * 10^10 + a : ℝ) = 11 * k) → 
  0 ≤ a → 
  a < 11 → 
  a = 9 := by
sorry

end NUMINAMATH_CALUDE_divisible_by_eleven_l2258_225827


namespace NUMINAMATH_CALUDE_boys_on_slide_l2258_225838

theorem boys_on_slide (initial_boys additional_boys : ℕ) 
  (h1 : initial_boys = 22)
  (h2 : additional_boys = 13) :
  initial_boys + additional_boys = 35 := by
sorry

end NUMINAMATH_CALUDE_boys_on_slide_l2258_225838


namespace NUMINAMATH_CALUDE_group_size_from_average_age_change_l2258_225818

theorem group_size_from_average_age_change (N : ℕ) (T : ℕ) : 
  N > 0 → 
  (T : ℚ) / N - 3 = (T - 42 + 12 : ℚ) / N → 
  N = 10 := by
sorry

end NUMINAMATH_CALUDE_group_size_from_average_age_change_l2258_225818


namespace NUMINAMATH_CALUDE_wire_service_reporters_l2258_225814

theorem wire_service_reporters (total : ℝ) (h_total : total > 0) :
  let local_politics := 0.28 * total
  let all_politics := local_politics / 0.7
  let non_politics := total - all_politics
  non_politics / total = 0.6 :=
by
  sorry

end NUMINAMATH_CALUDE_wire_service_reporters_l2258_225814


namespace NUMINAMATH_CALUDE_existence_of_N_l2258_225858

theorem existence_of_N : ∃ N : ℝ, (0.47 * N - 0.36 * 1412) + 63 = 3 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_N_l2258_225858


namespace NUMINAMATH_CALUDE_solution_satisfies_system_l2258_225810

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := ((2 * x - y) ^ (2 / x)) ^ (1 / 2) = 2
def equation2 (x y : ℝ) : Prop := (2 * x - y) * (5 ^ (x / 4)) = 1000

-- Theorem statement
theorem solution_satisfies_system :
  ∃ (x y : ℝ), x = 12 ∧ y = 16 ∧ equation1 x y ∧ equation2 x y :=
by sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_l2258_225810


namespace NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_17_l2258_225844

theorem greatest_two_digit_multiple_of_17 : ∃ n : ℕ, n = 85 ∧ 
  (∀ m : ℕ, m % 17 = 0 → 10 ≤ m → m ≤ 99 → m ≤ n) := by
  sorry

end NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_17_l2258_225844


namespace NUMINAMATH_CALUDE_product_of_sums_and_differences_l2258_225872

theorem product_of_sums_and_differences (P Q R S : ℝ) : 
  P = (Real.sqrt 2010 + Real.sqrt 2011) →
  Q = (-Real.sqrt 2010 - Real.sqrt 2011) →
  R = (Real.sqrt 2010 - Real.sqrt 2011) →
  S = (Real.sqrt 2011 - Real.sqrt 2010) →
  P * Q * R * S = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_and_differences_l2258_225872


namespace NUMINAMATH_CALUDE_area_integral_solution_l2258_225876

theorem area_integral_solution (k : ℝ) : k > 0 →
  (∫ x in Set.Icc k (1/2), 1/x) = 2 * Real.log 2 ∨
  (∫ x in Set.Icc (1/2) k, 1/x) = 2 * Real.log 2 →
  k = 1/8 ∨ k = 2 := by
sorry

end NUMINAMATH_CALUDE_area_integral_solution_l2258_225876


namespace NUMINAMATH_CALUDE_dictionary_chunk_pages_l2258_225897

def is_permutation (a b : ℕ) : Prop := sorry

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

theorem dictionary_chunk_pages (first_page last_page : ℕ) :
  first_page = 213 →
  is_permutation first_page last_page →
  is_even last_page →
  ∀ p, is_permutation first_page p ∧ is_even p → p ≤ last_page →
  last_page - first_page + 1 = 100 :=
by sorry

end NUMINAMATH_CALUDE_dictionary_chunk_pages_l2258_225897


namespace NUMINAMATH_CALUDE_baseball_card_pages_l2258_225812

theorem baseball_card_pages (cards_per_page new_cards old_cards : ℕ) 
  (h1 : cards_per_page = 3)
  (h2 : new_cards = 3)
  (h3 : old_cards = 9) :
  (new_cards + old_cards) / cards_per_page = 4 := by
  sorry

end NUMINAMATH_CALUDE_baseball_card_pages_l2258_225812


namespace NUMINAMATH_CALUDE_quadratic_equation_m_range_l2258_225831

theorem quadratic_equation_m_range (m : ℝ) :
  (∀ x, ∃ a b c : ℝ, a ≠ 0 ∧ (m^2 - 4) * x^2 + (2 - m) * x + 1 = a * x^2 + b * x + c) ↔
  m ≠ 2 ∧ m ≠ -2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_m_range_l2258_225831


namespace NUMINAMATH_CALUDE_function_properties_l2258_225822

def f (a b x : ℝ) : ℝ := a * x^2 + b * x - 2 * b

theorem function_properties :
  ∀ (a b : ℝ),
  (a > 0 ∧ a = b → {x : ℝ | f a b x < 0} = Set.Ioo (-2) 1) ∧
  (a = 1 ∧ (∀ x < 2, f a b x ≥ 1) → b ≤ 2 * Real.sqrt 3 - 4) ∧
  (|f a b (-1)| ≤ 1 ∧ |f a b 1| ≤ 3 → 5/3 ≤ |a| + |b + 2| ∧ |a| + |b + 2| ≤ 9) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2258_225822


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_two_l2258_225847

theorem sqrt_sum_equals_two (x y θ : ℝ) 
  (h1 : x + y = 3 - Real.cos (4 * θ)) 
  (h2 : x - y = 4 * Real.sin (2 * θ)) : 
  Real.sqrt x + Real.sqrt y = 2 := by
sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_two_l2258_225847


namespace NUMINAMATH_CALUDE_surrounding_circles_radius_l2258_225895

theorem surrounding_circles_radius (R : ℝ) (n : ℕ) (r : ℝ) :
  R = 2 ∧ n = 6 ∧ r > 0 →
  (R + r)^2 = (2 * r)^2 + (2 * r)^2 - 2 * (2 * r) * (2 * r) * Real.cos (2 * Real.pi / n) →
  r = (2 + 2 * Real.sqrt 11) / 11 := by
  sorry

end NUMINAMATH_CALUDE_surrounding_circles_radius_l2258_225895


namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l2258_225848

theorem boys_to_girls_ratio : 
  ∀ (boys girls : ℕ), 
    boys = 80 →
    girls = boys + 128 →
    ∃ (a b : ℕ), a = 5 ∧ b = 13 ∧ a * girls = b * boys :=
by
  sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l2258_225848


namespace NUMINAMATH_CALUDE_kenny_basketball_time_l2258_225885

/-- 
Given that:
- Kenny played basketball last week
- He ran for twice as long as he played basketball
- He practiced on the trumpet for twice as long as he ran
- He practiced on the trumpet for 40 hours

Prove that Kenny played basketball for 10 hours last week.
-/
theorem kenny_basketball_time (trumpet_time : ℕ) (h1 : trumpet_time = 40) :
  let run_time := trumpet_time / 2
  let basketball_time := run_time / 2
  basketball_time = 10 := by
  sorry

end NUMINAMATH_CALUDE_kenny_basketball_time_l2258_225885


namespace NUMINAMATH_CALUDE_problem_statement_l2258_225863

theorem problem_statement (m n N k : ℕ) :
  (n^2 + 1)^(2^k) * (44*n^3 + 11*n^2 + 10*n + 2) = N^m →
  m = 1 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2258_225863


namespace NUMINAMATH_CALUDE_book_cost_price_l2258_225832

theorem book_cost_price (cost : ℝ) : 
  (1.15 * cost - 1.10 * cost = 120) → cost = 2400 :=
by sorry

end NUMINAMATH_CALUDE_book_cost_price_l2258_225832


namespace NUMINAMATH_CALUDE_five_sixteenths_decimal_l2258_225823

theorem five_sixteenths_decimal : (5 : ℚ) / 16 = 0.3125 := by sorry

end NUMINAMATH_CALUDE_five_sixteenths_decimal_l2258_225823


namespace NUMINAMATH_CALUDE_square_1444_product_l2258_225884

theorem square_1444_product (x : ℤ) (h : x^2 = 1444) : (x + 1) * (x - 1) = 1443 := by
  sorry

end NUMINAMATH_CALUDE_square_1444_product_l2258_225884


namespace NUMINAMATH_CALUDE_half_sum_of_consecutive_odd_primes_is_composite_l2258_225898

/-- Two natural numbers are consecutive primes if they are both prime and there are no primes between them. -/
def ConsecutivePrimes (p p' : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime p' ∧ p < p' ∧ ∀ q, p < q → q < p' → ¬Nat.Prime q

/-- A natural number is composite if it's greater than 1 and not prime. -/
def Composite (n : ℕ) : Prop :=
  n > 1 ∧ ¬Nat.Prime n

theorem half_sum_of_consecutive_odd_primes_is_composite
  (p p' : ℕ) (h : ConsecutivePrimes p p') (hp_odd : Odd p) (hp'_odd : Odd p') (hp_ge_3 : p ≥ 3) :
  Composite ((p + p') / 2) :=
sorry

end NUMINAMATH_CALUDE_half_sum_of_consecutive_odd_primes_is_composite_l2258_225898


namespace NUMINAMATH_CALUDE_unique_x_for_real_sqrt_l2258_225806

theorem unique_x_for_real_sqrt (y : ℝ) : ∃! x : ℝ, ∃ z : ℝ, z^2 = -(x + 2*y)^2 := by
  sorry

end NUMINAMATH_CALUDE_unique_x_for_real_sqrt_l2258_225806


namespace NUMINAMATH_CALUDE_exists_special_expression_l2258_225840

/-- Represents an arithmetic expression with ones and alternating operations -/
inductive Expression
  | one : Expression
  | add : Expression → Expression → Expression
  | mul : Expression → Expression → Expression

/-- Evaluates an expression -/
def evaluate : Expression → ℕ
  | Expression.one => 1
  | Expression.add e1 e2 => evaluate e1 + evaluate e2
  | Expression.mul e1 e2 => evaluate e1 * evaluate e2

/-- Swaps addition and multiplication operations in an expression -/
def swap_operations : Expression → Expression
  | Expression.one => Expression.one
  | Expression.add e1 e2 => Expression.mul (swap_operations e1) (swap_operations e2)
  | Expression.mul e1 e2 => Expression.add (swap_operations e1) (swap_operations e2)

/-- Theorem stating the existence of an expression satisfying the problem conditions -/
theorem exists_special_expression : 
  ∃ e : Expression, evaluate e = 2014 ∧ evaluate (swap_operations e) = 2014 := by
  sorry


end NUMINAMATH_CALUDE_exists_special_expression_l2258_225840


namespace NUMINAMATH_CALUDE_shyam_weight_increase_l2258_225874

theorem shyam_weight_increase 
  (ram_original : ℝ) 
  (shyam_original : ℝ) 
  (ram_shyam_ratio : ram_original / shyam_original = 4 / 5)
  (ram_increase : ℝ) 
  (ram_increase_percent : ram_increase = 0.1 * ram_original)
  (total_new : ℝ) 
  (total_new_value : total_new = 82.8)
  (total_increase : ℝ) 
  (total_increase_percent : total_increase = 0.15 * (ram_original + shyam_original))
  (total_new_eq : total_new = ram_original + ram_increase + shyam_original + (shyam_original * x))
  : x = 0.19 := by
  sorry

#check shyam_weight_increase

end NUMINAMATH_CALUDE_shyam_weight_increase_l2258_225874


namespace NUMINAMATH_CALUDE_odd_integer_divisibility_l2258_225861

theorem odd_integer_divisibility (n : ℤ) (h : Odd n) :
  ∃ x : ℤ, (n^2 : ℤ) ∣ (x^2 - n*x - 1) := by sorry

end NUMINAMATH_CALUDE_odd_integer_divisibility_l2258_225861


namespace NUMINAMATH_CALUDE_even_function_quadratic_l2258_225867

def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 2

theorem even_function_quadratic (a b : ℝ) :
  (∀ x ∈ Set.Icc (1 + a) 1, f a b x = f a b (-x)) →
  a + 2 * b = -2 := by
  sorry

end NUMINAMATH_CALUDE_even_function_quadratic_l2258_225867


namespace NUMINAMATH_CALUDE_reciprocals_proportional_l2258_225886

/-- If x and y are directly proportional, then their reciprocals are also directly proportional -/
theorem reciprocals_proportional {x y : ℝ} (h : ∃ k : ℝ, y = k * x) :
  ∃ c : ℝ, (1 / y) = c * (1 / x) :=
by sorry

end NUMINAMATH_CALUDE_reciprocals_proportional_l2258_225886


namespace NUMINAMATH_CALUDE_origin_on_circle_M_l2258_225864

-- Define the parabola C: y² = 2x
def parabola_C (x y : ℝ) : Prop := y^2 = 2*x

-- Define the line l passing through (2,0)
def line_l (x y : ℝ) (k : ℝ) : Prop := y = k*(x - 2)

-- Define points A and B as intersections of line l and parabola C
def point_A (k : ℝ) : ℝ × ℝ := sorry
def point_B (k : ℝ) : ℝ × ℝ := sorry

-- Define circle M with diameter AB
def circle_M (x y : ℝ) (k : ℝ) : Prop :=
  let A := point_A k
  let B := point_B k
  let center := ((A.1 + B.1)/2, (A.2 + B.2)/2)
  let radius := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) / 2
  (x - center.1)^2 + (y - center.2)^2 = radius^2

-- Theorem: The origin O(0,0) lies on circle M
theorem origin_on_circle_M (k : ℝ) : circle_M 0 0 k :=
  sorry


end NUMINAMATH_CALUDE_origin_on_circle_M_l2258_225864


namespace NUMINAMATH_CALUDE_part_one_part_two_l2258_225830

-- Define the function f
def f (x m : ℝ) : ℝ := |x + m| + |2 * x - 3|

-- Part I
theorem part_one : 
  {x : ℝ | f x (-3) < 9} = {x : ℝ | -1 < x ∧ x < 5} :=
sorry

-- Part II
theorem part_two :
  (∃ x ∈ Set.Icc 2 4, f x m ≤ 3) → m ∈ Set.Icc (-2) 2 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2258_225830


namespace NUMINAMATH_CALUDE_average_marks_of_failed_candidates_l2258_225842

theorem average_marks_of_failed_candidates
  (total_candidates : ℕ)
  (overall_average : ℚ)
  (passed_average : ℚ)
  (passed_candidates : ℕ)
  (h1 : total_candidates = 120)
  (h2 : overall_average = 35)
  (h3 : passed_average = 39)
  (h4 : passed_candidates = 100) :
  (total_candidates * overall_average - passed_candidates * passed_average) / (total_candidates - passed_candidates) = 15 :=
by sorry

end NUMINAMATH_CALUDE_average_marks_of_failed_candidates_l2258_225842


namespace NUMINAMATH_CALUDE_valid_numbers_l2258_225892

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 100000 ∧ n < 1000000) ∧
  (∀ d : ℕ, d ∈ [1, 2, 3, 4, 5, 6] → ∃! p : ℕ, p < 6 ∧ (n / 10^p) % 10 = d) ∧
  (n / 10000) % 2 = 0 ∧
  (n / 10000) % 3 = 0 ∧
  (n / 100) % 4 = 0 ∧
  (n / 10) % 5 = 0 ∧
  n % 6 = 0

theorem valid_numbers : 
  {n : ℕ | is_valid_number n} = {123654, 321654} :=
sorry

end NUMINAMATH_CALUDE_valid_numbers_l2258_225892


namespace NUMINAMATH_CALUDE_polar_line_through_point_parallel_to_axis_l2258_225836

/-- 
Represents a line in polar coordinates passing through a given point and parallel to the polar axis.
-/
def polar_line_parallel_to_axis (r : ℝ) (θ : ℝ) : Prop :=
  ∀ ρ θ', ρ * Real.sin θ' = r * Real.sin θ

theorem polar_line_through_point_parallel_to_axis 
  (r : ℝ) (θ : ℝ) (h_r : r = 2) (h_θ : θ = π/4) :
  polar_line_parallel_to_axis r θ ↔ 
  ∀ ρ θ', ρ * Real.sin θ' = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_polar_line_through_point_parallel_to_axis_l2258_225836


namespace NUMINAMATH_CALUDE_world_expo_visitors_l2258_225850

def cost_per_person (n : ℕ) : ℕ :=
  if n ≤ 30 then 120
  else max 90 (120 - 2 * (n - 30))

def total_cost (n : ℕ) : ℕ :=
  n * cost_per_person n

theorem world_expo_visitors :
  ∃ n : ℕ, n > 30 ∧ total_cost n = 4000 ∧
  ∀ m : ℕ, m ≠ n → total_cost m ≠ 4000 :=
sorry

end NUMINAMATH_CALUDE_world_expo_visitors_l2258_225850


namespace NUMINAMATH_CALUDE_missed_bus_time_l2258_225813

theorem missed_bus_time (usual_time : ℝ) (speed_ratio : ℝ) (h1 : usual_time = 16) (h2 : speed_ratio = 4/5) :
  (usual_time / speed_ratio) - usual_time = 4 := by
  sorry

end NUMINAMATH_CALUDE_missed_bus_time_l2258_225813


namespace NUMINAMATH_CALUDE_root_property_l2258_225846

theorem root_property (a : ℝ) : 3 * a^2 - 5 * a - 2 = 0 → 6 * a^2 - 10 * a = 4 := by
  sorry

end NUMINAMATH_CALUDE_root_property_l2258_225846


namespace NUMINAMATH_CALUDE_right_angled_triangle_l2258_225841

theorem right_angled_triangle (A B C : ℝ) (h_triangle : A + B + C = π) 
  (h_cosine_sum : Real.cos A ^ 2 + Real.cos B ^ 2 + Real.cos C ^ 2 = 1) : 
  A = π / 2 ∨ B = π / 2 ∨ C = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_right_angled_triangle_l2258_225841


namespace NUMINAMATH_CALUDE_bells_toll_together_l2258_225899

def bell_intervals : List Nat := [2, 4, 6, 8, 10, 12]
def period_minutes : Nat := 30
def period_seconds : Nat := period_minutes * 60

def lcm_list (list : List Nat) : Nat :=
  list.foldl Nat.lcm 1

theorem bells_toll_together : 
  (period_seconds / lcm_list bell_intervals) + 1 = 16 := by
  sorry

end NUMINAMATH_CALUDE_bells_toll_together_l2258_225899


namespace NUMINAMATH_CALUDE_fraction_simplification_l2258_225834

theorem fraction_simplification : 
  (1 / (1 + 1 / (3 + 1 / 4))) = 13 / 17 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2258_225834


namespace NUMINAMATH_CALUDE_festival_attendance_l2258_225880

theorem festival_attendance (total : ℕ) (d1 d2 d3 d4 : ℕ) : 
  total = 3600 →
  d2 = d1 / 2 →
  d3 = 3 * d1 →
  d4 = 2 * d2 →
  d1 + d2 + d3 + d4 = total →
  (total : ℚ) / 4 = 900 := by
  sorry

end NUMINAMATH_CALUDE_festival_attendance_l2258_225880


namespace NUMINAMATH_CALUDE_simplify_square_roots_l2258_225890

theorem simplify_square_roots : 
  Real.sqrt 18 * Real.sqrt 72 + Real.sqrt 200 = 36 + 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l2258_225890


namespace NUMINAMATH_CALUDE_smallest_prime_after_seven_nonprimes_l2258_225804

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_consecutive_nonprime (start : ℕ) : Prop :=
  ∀ i : ℕ, i < 7 → ¬(is_prime (start + i))

theorem smallest_prime_after_seven_nonprimes :
  ∃ start : ℕ,
    is_consecutive_nonprime start ∧
    is_prime 97 ∧
    (∀ p : ℕ, p < 97 → ¬(is_prime p ∧ p > start + 6)) :=
  sorry

end NUMINAMATH_CALUDE_smallest_prime_after_seven_nonprimes_l2258_225804


namespace NUMINAMATH_CALUDE_square_remainder_is_square_l2258_225889

theorem square_remainder_is_square (N : ℤ) : ∃ (a b : ℤ), 
  ((N = 8 * a + b ∨ N = 8 * a - b) ∧ (b = 0 ∨ b = 1 ∨ b = 2 ∨ b = 3 ∨ b = 4)) →
  ∃ (k : ℤ), N^2 % 16 = k^2 := by
sorry

end NUMINAMATH_CALUDE_square_remainder_is_square_l2258_225889


namespace NUMINAMATH_CALUDE_largest_valid_number_l2258_225839

def is_valid_number (n : ℕ) : Prop :=
  (n > 0) ∧
  (∀ d₁ d₂, d₁ ∈ n.digits 10 → d₂ ∈ n.digits 10 → d₁ ≠ d₂ → d₁ ≠ 0 ∧ d₂ ≠ 0) ∧
  (n.digits 10).sum = 17

theorem largest_valid_number :
  ∀ m : ℕ, is_valid_number m → m ≤ 98 :=
by sorry

end NUMINAMATH_CALUDE_largest_valid_number_l2258_225839


namespace NUMINAMATH_CALUDE_soup_kettle_capacity_l2258_225878

theorem soup_kettle_capacity (current_percentage : ℚ) (current_servings : ℕ) : 
  current_percentage = 55 / 100 →
  current_servings = 88 →
  (current_servings : ℚ) / current_percentage = 160 :=
by sorry

end NUMINAMATH_CALUDE_soup_kettle_capacity_l2258_225878


namespace NUMINAMATH_CALUDE_solve_equation_l2258_225857

theorem solve_equation : 
  ∃ y : ℚ, (40 : ℚ) / 60 = Real.sqrt ((y / 60) - (10 : ℚ) / 60) → y = 110 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2258_225857


namespace NUMINAMATH_CALUDE_helga_shoe_ratio_l2258_225882

/-- Represents the number of shoe pairs tried on at each store --/
structure ShoeTrials where
  store1 : ℕ
  store2 : ℕ
  store3 : ℕ
  store4 : ℕ

/-- Calculates the ratio of shoes tried on at the fourth store to the first three stores combined --/
def shoeRatio (trials : ShoeTrials) : Rat :=
  trials.store4 / (trials.store1 + trials.store2 + trials.store3)

/-- Theorem stating the shoe trial ratio for Helga's shopping trip --/
theorem helga_shoe_ratio :
  let trials : ShoeTrials := {
    store1 := 7,
    store2 := 9,
    store3 := 0,
    store4 := 48 - (7 + 9 + 0)
  }
  shoeRatio trials = 2 / 1 := by sorry

end NUMINAMATH_CALUDE_helga_shoe_ratio_l2258_225882


namespace NUMINAMATH_CALUDE_proportion_solution_l2258_225821

/-- Given a proportion 0.75 : x :: 5 : 11, prove that x = 1.65 -/
theorem proportion_solution (x : ℝ) (h : 0.75 / x = 5 / 11) : x = 1.65 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l2258_225821


namespace NUMINAMATH_CALUDE_cone_volume_l2258_225803

/-- Given a cone with slant height 5 and lateral area 20π, its volume is 16π -/
theorem cone_volume (s : ℝ) (l : ℝ) (v : ℝ) 
  (h_slant : s = 5)
  (h_lateral : l = 20 * Real.pi) :
  v = 16 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_l2258_225803


namespace NUMINAMATH_CALUDE_student_sums_problem_l2258_225820

theorem student_sums_problem (total : ℕ) (correct : ℕ) (incorrect : ℕ) 
  (h1 : total = 96)
  (h2 : incorrect = 3 * correct)
  (h3 : total = correct + incorrect) :
  correct = 24 := by
  sorry

end NUMINAMATH_CALUDE_student_sums_problem_l2258_225820


namespace NUMINAMATH_CALUDE_first_year_growth_rate_is_15_percent_l2258_225888

def initial_investment : ℝ := 80
def additional_investment : ℝ := 28
def second_year_growth_rate : ℝ := 0.10
def final_portfolio_value : ℝ := 132

theorem first_year_growth_rate_is_15_percent :
  ∃ r : ℝ, 
    (initial_investment * (1 + r) + additional_investment) * (1 + second_year_growth_rate) = final_portfolio_value ∧
    r = 0.15 := by
  sorry

end NUMINAMATH_CALUDE_first_year_growth_rate_is_15_percent_l2258_225888


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l2258_225835

/-- Given line l1 with equation 4x + 5y - 8 = 0 -/
def l1 : ℝ → ℝ → Prop :=
  λ x y => 4*x + 5*y - 8 = 0

/-- Point A with coordinates (3, 2) -/
def A : ℝ × ℝ := (3, 2)

/-- The perpendicular line l2 passing through A -/
def l2 : ℝ → ℝ → Prop :=
  λ x y => 5*x - 4*y - 7 = 0

theorem perpendicular_line_through_point :
  (∀ x y, l2 x y ↔ 5*x - 4*y - 7 = 0) ∧
  l2 A.1 A.2 ∧
  (∀ x1 y1 x2 y2, l1 x1 y1 → l1 x2 y2 → l2 x1 y1 → l2 x2 y2 →
    (x2 - x1) * (4) + (y2 - y1) * (5) = 0) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l2258_225835


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l2258_225815

theorem sqrt_product_equality : Real.sqrt 12 * Real.sqrt 8 = 4 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l2258_225815


namespace NUMINAMATH_CALUDE_column_with_most_shaded_boxes_l2258_225807

def divisors (n : ℕ) : Finset ℕ :=
  Finset.filter (· ∣ n) (Finset.range (n + 1))

def number_of_divisors (n : ℕ) : ℕ :=
  (divisors n).card

theorem column_with_most_shaded_boxes :
  ∀ n ∈ ({144, 120, 150, 96, 100} : Finset ℕ),
    number_of_divisors n ≤ number_of_divisors 120 :=
by sorry

end NUMINAMATH_CALUDE_column_with_most_shaded_boxes_l2258_225807


namespace NUMINAMATH_CALUDE_vector_angle_constraint_l2258_225859

def a (k : ℝ) : Fin 2 → ℝ := ![-k, 4]
def b (k : ℝ) : Fin 2 → ℝ := ![k, k+3]

def dot_product (v w : Fin 2 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1)

def is_acute_angle (v w : Fin 2 → ℝ) : Prop :=
  dot_product v w > 0 ∧ v ≠ w

theorem vector_angle_constraint (k : ℝ) :
  is_acute_angle (a k) (b k) → k ∈ Set.Ioo (-2 : ℝ) 0 ∪ Set.Ioo 0 6 :=
sorry

end NUMINAMATH_CALUDE_vector_angle_constraint_l2258_225859


namespace NUMINAMATH_CALUDE_coffee_order_total_cost_l2258_225862

theorem coffee_order_total_cost : 
  let drip_coffee_price : ℝ := 2.25
  let drip_coffee_discount : ℝ := 0.1
  let espresso_price : ℝ := 3.50
  let espresso_tax : ℝ := 0.15
  let latte_price : ℝ := 4.00
  let vanilla_syrup_price : ℝ := 0.50
  let vanilla_syrup_tax : ℝ := 0.20
  let cold_brew_price : ℝ := 2.50
  let cold_brew_discount : ℝ := 1.00
  let cappuccino_price : ℝ := 3.50
  let cappuccino_tip : ℝ := 0.05

  let drip_coffee_cost := 2 * drip_coffee_price * (1 - drip_coffee_discount)
  let espresso_cost := espresso_price * (1 + espresso_tax)
  let latte_cost := latte_price + (latte_price / 2) + (vanilla_syrup_price * (1 + vanilla_syrup_tax))
  let cold_brew_cost := 2 * cold_brew_price - cold_brew_discount
  let cappuccino_cost := cappuccino_price * (1 + cappuccino_tip)

  let total_cost := drip_coffee_cost + espresso_cost + latte_cost + cold_brew_cost + cappuccino_cost

  total_cost = 22.35 := by sorry

end NUMINAMATH_CALUDE_coffee_order_total_cost_l2258_225862


namespace NUMINAMATH_CALUDE_circumcenter_on_median_l2258_225833

variable {A B C O H P Q : ℂ}

/-- The triangle ABC is acute -/
def is_acute_triangle (A B C : ℂ) : Prop := sorry

/-- O is the circumcenter of triangle ABC -/
def is_circumcenter (O A B C : ℂ) : Prop := sorry

/-- H is the orthocenter of triangle ABC -/
def is_orthocenter (H A B C : ℂ) : Prop := sorry

/-- P is the intersection of OA and the altitude from B -/
def is_P_intersection (P O A B C : ℂ) : Prop := sorry

/-- Q is the intersection of OA and the altitude from C -/
def is_Q_intersection (Q O A B C : ℂ) : Prop := sorry

/-- X is the circumcenter of triangle PQH -/
def is_PQH_circumcenter (X P Q H : ℂ) : Prop := sorry

/-- M is the midpoint of BC -/
def is_midpoint_BC (M B C : ℂ) : Prop := sorry

/-- Three points are collinear -/
def collinear (X Y Z : ℂ) : Prop := sorry

theorem circumcenter_on_median 
  (h_acute : is_acute_triangle A B C)
  (h_O : is_circumcenter O A B C)
  (h_H : is_orthocenter H A B C)
  (h_P : is_P_intersection P O A B C)
  (h_Q : is_Q_intersection Q O A B C) :
  ∃ (X M : ℂ), is_PQH_circumcenter X P Q H ∧ 
               is_midpoint_BC M B C ∧ 
               collinear A X M :=
sorry

end NUMINAMATH_CALUDE_circumcenter_on_median_l2258_225833


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2258_225883

theorem absolute_value_inequality (a b : ℝ) (h : a > b) : |a| > b := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2258_225883


namespace NUMINAMATH_CALUDE_new_ratio_is_one_to_two_l2258_225896

/-- Represents the coin collection --/
structure CoinCollection where
  gold : ℕ
  silver : ℕ

/-- The initial state of the coin collection --/
def initial_collection : CoinCollection :=
  { gold := 0, silver := 0 }

/-- The condition that initially there is one gold coin for every 3 silver coins --/
axiom initial_ratio (c : CoinCollection) : c.gold * 3 = c.silver

/-- The operation of adding 15 gold coins to the collection --/
def add_gold_coins (c : CoinCollection) : CoinCollection :=
  { gold := c.gold + 15, silver := c.silver }

/-- The total number of coins after adding 15 gold coins is 135 --/
axiom total_coins_after_addition (c : CoinCollection) :
  (add_gold_coins c).gold + (add_gold_coins c).silver = 135

/-- Theorem stating that the new ratio of gold to silver coins is 1:2 --/
theorem new_ratio_is_one_to_two (c : CoinCollection) :
  2 * (add_gold_coins c).gold = (add_gold_coins c).silver :=
sorry

end NUMINAMATH_CALUDE_new_ratio_is_one_to_two_l2258_225896


namespace NUMINAMATH_CALUDE_six_legged_creatures_count_l2258_225866

/-- Represents the number of creatures with 6 legs -/
def creatures_with_6_legs : ℕ := sorry

/-- Represents the number of creatures with 10 legs -/
def creatures_with_10_legs : ℕ := sorry

/-- The total number of creatures -/
def total_creatures : ℕ := 20

/-- The total number of legs -/
def total_legs : ℕ := 156

/-- Theorem stating that the number of creatures with 6 legs is 11 -/
theorem six_legged_creatures_count : 
  creatures_with_6_legs = 11 ∧ 
  creatures_with_6_legs + creatures_with_10_legs = total_creatures ∧
  6 * creatures_with_6_legs + 10 * creatures_with_10_legs = total_legs := by
  sorry

end NUMINAMATH_CALUDE_six_legged_creatures_count_l2258_225866


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l2258_225819

theorem complex_number_quadrant : ∃ (z : ℂ), z = (2 - Complex.I)^2 ∧ Complex.re z > 0 ∧ Complex.im z < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l2258_225819


namespace NUMINAMATH_CALUDE_trajectory_of_intersecting_lines_l2258_225801

/-- The trajectory of point P given two intersecting lines through A(-1,0) and B(1,0) with slope product -1 -/
theorem trajectory_of_intersecting_lines (x y : ℝ) :
  let k_AP := y / (x + 1)
  let k_BP := y / (x - 1)
  (k_AP * k_BP = -1) → (x ≠ -1 ∧ x ≠ 1) → (x^2 + y^2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_trajectory_of_intersecting_lines_l2258_225801


namespace NUMINAMATH_CALUDE_remainder_is_neg_one_l2258_225849

/-- The polynomial x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 -/
def f (x : ℂ) : ℂ := x^6 + x^5 + x^4 + x^3 + x^2 + x + 1

/-- The polynomial x^100 + x^75 + x^50 + x^25 + 1 -/
def g (x : ℂ) : ℂ := x^100 + x^75 + x^50 + x^25 + 1

/-- The theorem stating that the remainder of g(x) divided by f(x) is -1 -/
theorem remainder_is_neg_one : ∃ q : ℂ → ℂ, ∀ x : ℂ, g x = f x * q x + (-1) := by
  sorry

end NUMINAMATH_CALUDE_remainder_is_neg_one_l2258_225849


namespace NUMINAMATH_CALUDE_exam_failure_count_l2258_225816

theorem exam_failure_count (total : ℕ) (pass_percentage : ℚ) (fail_count : ℕ) : 
  total = 400 → pass_percentage = 35 / 100 → fail_count = total - (pass_percentage * total).floor → fail_count = 260 := by
  sorry

end NUMINAMATH_CALUDE_exam_failure_count_l2258_225816


namespace NUMINAMATH_CALUDE_no_two_digit_double_square_sum_l2258_225851

theorem no_two_digit_double_square_sum :
  ¬ ∃ (N : ℕ), 
    (10 ≤ N ∧ N ≤ 99) ∧ 
    (∃ (k : ℕ), N + (10 * (N % 10) + N / 10) = 2 * k^2) :=
sorry

end NUMINAMATH_CALUDE_no_two_digit_double_square_sum_l2258_225851


namespace NUMINAMATH_CALUDE_parallelogram_area_l2258_225828

/-- The area of a parallelogram with base 20 and height 4 is 80 -/
theorem parallelogram_area : 
  ∀ (base height : ℝ), 
  base = 20 → 
  height = 4 → 
  base * height = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l2258_225828


namespace NUMINAMATH_CALUDE_arithmetic_mean_sqrt2_l2258_225879

theorem arithmetic_mean_sqrt2 : 
  (Real.sqrt 2 + 1 + (Real.sqrt 2 - 1)) / 2 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_sqrt2_l2258_225879


namespace NUMINAMATH_CALUDE_base_conversion_problem_l2258_225873

theorem base_conversion_problem (c d : ℕ) :
  (c ≤ 9 ∧ d ≤ 9) →  -- c and d are base-10 digits
  (5 * 8^2 + 4 * 8^1 + 3 * 8^0 = 300 + 10 * c + d) →  -- 543₈ = 3cd₁₀
  (c * d) / 12 = 5 / 4 := by
sorry

end NUMINAMATH_CALUDE_base_conversion_problem_l2258_225873


namespace NUMINAMATH_CALUDE_P_intersect_M_l2258_225824

def P : Set ℝ := {y | ∃ x, y = x^2 - 3*x + 1}

def M : Set ℝ := {y | ∃ x ∈ Set.Icc (-2) 5, y = Real.sqrt (x + 2) * Real.sqrt (5 - x)}

theorem P_intersect_M : P ∩ M = Set.Icc (-5/4) 5 := by sorry

end NUMINAMATH_CALUDE_P_intersect_M_l2258_225824


namespace NUMINAMATH_CALUDE_age_difference_proof_l2258_225877

/-- Given the ages of Milena, her grandmother, and her grandfather, prove the age difference between Milena and her grandfather. -/
theorem age_difference_proof (milena_age : ℕ) (grandmother_age_factor : ℕ) (grandfather_age_difference : ℕ) 
  (h1 : milena_age = 7)
  (h2 : grandmother_age_factor = 9)
  (h3 : grandfather_age_difference = 2) :
  grandfather_age_difference + grandmother_age_factor * milena_age - milena_age = 58 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_proof_l2258_225877


namespace NUMINAMATH_CALUDE_time_to_see_again_is_120_l2258_225808

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a person walking -/
structure Walker where
  position : Point
  speed : ℝ

/-- The scenario of Jenny and Kenny walking -/
structure WalkingScenario where
  jenny : Walker
  kenny : Walker
  buildingRadius : ℝ
  pathDistance : ℝ

/-- The time when Jenny and Kenny can see each other again -/
def timeToSeeAgain (scenario : WalkingScenario) : ℝ :=
  sorry

/-- The theorem stating that the time to see again is 120 seconds -/
theorem time_to_see_again_is_120 (scenario : WalkingScenario) :
  scenario.jenny.speed = 2 →
  scenario.kenny.speed = 4 →
  scenario.pathDistance = 300 →
  scenario.buildingRadius = 75 →
  scenario.jenny.position = Point.mk (-75) (-150) →
  scenario.kenny.position = Point.mk (-75) 150 →
  timeToSeeAgain scenario = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_time_to_see_again_is_120_l2258_225808


namespace NUMINAMATH_CALUDE_angle_ADC_measure_l2258_225854

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ)
  (sum_angles : A + B + C = 180)

-- Define the point D
structure PointD (t : Triangle) :=
  (on_angle_bisector : True)  -- D is on the angle bisector of ∠ABC
  (on_perp_bisector : True)   -- D is on the perpendicular bisector of AC

-- Theorem statement
theorem angle_ADC_measure (t : Triangle) (d : PointD t) 
  (h1 : t.A = 44) (h2 : t.B = 66) (h3 : t.C = 70) : 
  ∃ (angle_ADC : ℝ), angle_ADC = 114 := by
  sorry

end NUMINAMATH_CALUDE_angle_ADC_measure_l2258_225854


namespace NUMINAMATH_CALUDE_wilsons_theorem_and_square_l2258_225875

theorem wilsons_theorem_and_square (p : Nat) (hp : p > 1) :
  (((Nat.factorial (p - 1) + 1) % p = 0) ↔ Nat.Prime p) ∧
  (Nat.Prime p → (Nat.factorial (p - 1))^2 % p = 1) ∧
  (¬Nat.Prime p → (Nat.factorial (p - 1))^2 % p = 0) := by
  sorry

end NUMINAMATH_CALUDE_wilsons_theorem_and_square_l2258_225875


namespace NUMINAMATH_CALUDE_not_divides_power_minus_one_l2258_225837

theorem not_divides_power_minus_one (n : ℕ) (h : n > 1) : ¬(n ∣ 2^n - 1) := by
  sorry

end NUMINAMATH_CALUDE_not_divides_power_minus_one_l2258_225837


namespace NUMINAMATH_CALUDE_remaining_money_is_63_10_l2258_225868

/-- Calculates the remaining money after hotel stays --/
def remaining_money (initial_amount : ℝ) 
  (hotel1_night_rate hotel1_morning_rate : ℝ)
  (hotel1_night_hours hotel1_morning_hours : ℝ)
  (hotel2_night_rate hotel2_morning_rate : ℝ)
  (hotel2_night_hours hotel2_morning_hours : ℝ)
  (tax_rate service_fee : ℝ) : ℝ :=
  let hotel1_subtotal := hotel1_night_rate * hotel1_night_hours + hotel1_morning_rate * hotel1_morning_hours
  let hotel2_subtotal := hotel2_night_rate * hotel2_night_hours + hotel2_morning_rate * hotel2_morning_hours
  let hotel1_total := hotel1_subtotal * (1 + tax_rate) + service_fee
  let hotel2_total := hotel2_subtotal * (1 + tax_rate) + service_fee
  initial_amount - (hotel1_total + hotel2_total)

/-- Theorem stating that the remaining money after hotel stays is $63.10 --/
theorem remaining_money_is_63_10 :
  remaining_money 160 2.5 3 6 4 3.5 4 8 6 0.1 5 = 63.1 := by
  sorry

end NUMINAMATH_CALUDE_remaining_money_is_63_10_l2258_225868


namespace NUMINAMATH_CALUDE_estimate_red_balls_l2258_225829

theorem estimate_red_balls (num_black : ℕ) (total_draws : ℕ) (black_draws : ℕ) :
  num_black = 4 →
  total_draws = 100 →
  black_draws = 40 →
  ∃ (num_red : ℕ),
    (num_black : ℚ) / (num_black + num_red : ℚ) = (black_draws : ℚ) / (total_draws : ℚ) ∧
    num_red = 6 := by
  sorry

end NUMINAMATH_CALUDE_estimate_red_balls_l2258_225829


namespace NUMINAMATH_CALUDE_unique_prime_with_few_divisors_l2258_225869

theorem unique_prime_with_few_divisors :
  ∃! p : ℕ, Nat.Prime p ∧ (Nat.card (Nat.divisors (p^2 + 11)) < 11) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_prime_with_few_divisors_l2258_225869


namespace NUMINAMATH_CALUDE_rhombus_area_70_l2258_225800

/-- The area of a rhombus with given vertices -/
theorem rhombus_area_70 : 
  let vertices : List (ℝ × ℝ) := [(0, 3.5), (10, 0), (0, -3.5), (-10, 0)]
  let diag1 : ℝ := |3.5 - (-3.5)|
  let diag2 : ℝ := |10 - (-10)|
  (diag1 * diag2) / 2 = 70 := by sorry

end NUMINAMATH_CALUDE_rhombus_area_70_l2258_225800


namespace NUMINAMATH_CALUDE_sin_translation_problem_l2258_225855

open Real

theorem sin_translation_problem (φ : ℝ) : 
  (0 < φ) → (φ < π / 2) →
  (∃ x₁ x₂ : ℝ, |sin (2 * x₁) - sin (2 * x₂ - 2 * φ)| = 2) →
  (∀ x₁ x₂ : ℝ, |sin (2 * x₁) - sin (2 * x₂ - 2 * φ)| = 2 → |x₁ - x₂| ≥ π / 3) →
  (∃ x₁ x₂ : ℝ, |sin (2 * x₁) - sin (2 * x₂ - 2 * φ)| = 2 ∧ |x₁ - x₂| = π / 3) →
  φ = π / 6 := by
sorry

end NUMINAMATH_CALUDE_sin_translation_problem_l2258_225855


namespace NUMINAMATH_CALUDE_equal_distribution_payout_l2258_225891

def earnings : List ℝ := [30, 35, 45, 55, 65]

theorem equal_distribution_payout (earnings : List ℝ) : 
  earnings = [30, 35, 45, 55, 65] →
  List.length earnings = 5 →
  List.sum earnings / 5 = 46 →
  65 - (List.sum earnings / 5) = 19 :=
by
  sorry

end NUMINAMATH_CALUDE_equal_distribution_payout_l2258_225891


namespace NUMINAMATH_CALUDE_candy_cost_l2258_225802

def coin_value (c : Nat) : Nat :=
  if c = 1 then 25
  else if c = 2 then 10
  else if c = 3 then 5
  else 1

def is_valid_change (coins : List Nat) : Prop :=
  coins.length = 4 ∧ coins.all (λ c => c ∈ [1, 2, 3, 4])

def change_value (coins : List Nat) : Nat :=
  coins.map coin_value |>.sum

theorem candy_cost (coins : List Nat) :
  is_valid_change coins →
  (∀ other_coins, is_valid_change other_coins → change_value other_coins ≤ change_value coins) →
  100 - change_value coins = 55 := by
sorry

end NUMINAMATH_CALUDE_candy_cost_l2258_225802


namespace NUMINAMATH_CALUDE_ordering_abc_l2258_225887

theorem ordering_abc : 
  let a : ℝ := Real.exp (Real.sqrt 2)
  let b : ℝ := 2 + Real.sqrt 2
  let c : ℝ := Real.log (12 + 6 * Real.sqrt 2)
  a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_ordering_abc_l2258_225887


namespace NUMINAMATH_CALUDE_architecture_better_than_logistics_l2258_225881

structure Industry where
  name : String
  applicants : ℕ
  openings : ℕ

def employment_ratio (i : Industry) : ℚ :=
  (i.applicants : ℚ) / (i.openings : ℚ)

theorem architecture_better_than_logistics 
  (architecture logistics : Industry)
  (h1 : architecture.name = "Architecture")
  (h2 : logistics.name = "Logistics")
  (h3 : architecture.applicants < logistics.applicants)
  (h4 : architecture.openings > logistics.openings) :
  employment_ratio architecture < employment_ratio logistics :=
by sorry

end NUMINAMATH_CALUDE_architecture_better_than_logistics_l2258_225881


namespace NUMINAMATH_CALUDE_pizza_slices_left_per_person_l2258_225856

theorem pizza_slices_left_per_person 
  (small_pizza : ℕ) 
  (large_pizza : ℕ) 
  (people : ℕ) 
  (slices_eaten_per_person : ℕ) 
  (h1 : small_pizza = 8) 
  (h2 : large_pizza = 14) 
  (h3 : people = 2) 
  (h4 : slices_eaten_per_person = 9) : 
  ((small_pizza + large_pizza) - (people * slices_eaten_per_person)) / people = 2 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_left_per_person_l2258_225856


namespace NUMINAMATH_CALUDE_fruit_prices_l2258_225853

theorem fruit_prices (x y z f : ℝ) 
  (h1 : x + y + z + f = 45)
  (h2 : f = 3 * x)
  (h3 : z = x + y) :
  y + z = 9 := by
sorry

end NUMINAMATH_CALUDE_fruit_prices_l2258_225853


namespace NUMINAMATH_CALUDE_area_inscribed_triangle_in_octagon_l2258_225865

/-- An equilateral octagon -/
structure EquilateralOctagon where
  side_length : ℝ
  is_positive : 0 < side_length

/-- An equilateral triangle formed by three diagonals of an equilateral octagon -/
def InscribedEquilateralTriangle (octagon : EquilateralOctagon) : Set (ℝ × ℝ) :=
  sorry

/-- The area of an inscribed equilateral triangle in an equilateral octagon -/
def area_inscribed_triangle (octagon : EquilateralOctagon) : ℝ :=
  sorry

/-- Theorem: The area of an inscribed equilateral triangle in an equilateral octagon
    with side length 60 is 900 -/
theorem area_inscribed_triangle_in_octagon :
  ∀ (octagon : EquilateralOctagon),
    octagon.side_length = 60 →
    area_inscribed_triangle octagon = 900 :=
by sorry

end NUMINAMATH_CALUDE_area_inscribed_triangle_in_octagon_l2258_225865


namespace NUMINAMATH_CALUDE_power_of_three_mod_thirteen_l2258_225894

theorem power_of_three_mod_thirteen : 3^3021 % 13 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_mod_thirteen_l2258_225894


namespace NUMINAMATH_CALUDE_hdha_ratio_is_zero_l2258_225870

/-- A triangle with sides of lengths 8, 15, and 17 -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 8
  hb : b = 15
  hc : c = 17

/-- The orthocenter (intersection of altitudes) of the triangle -/
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

/-- The foot of the altitude from A to side BC -/
def altitudeFoot (t : Triangle) : ℝ × ℝ := sorry

/-- The vertex A of the triangle -/
def vertexA (t : Triangle) : ℝ × ℝ := sorry

/-- The ratio of HD to HA, where H is the orthocenter and D is the foot of the altitude from A -/
def hdhaRatio (t : Triangle) : ℝ := sorry

theorem hdha_ratio_is_zero (t : Triangle) : hdhaRatio t = 0 := by
  sorry

end NUMINAMATH_CALUDE_hdha_ratio_is_zero_l2258_225870


namespace NUMINAMATH_CALUDE_remaining_money_calculation_l2258_225852

def euro_to_dollar : ℝ := 1.183
def pound_to_dollar : ℝ := 1.329
def yen_to_dollar : ℝ := 0.009
def real_to_dollar : ℝ := 0.193

def conversion_fee_rate : ℝ := 0.015
def sales_tax_rate : ℝ := 0.08
def transportation_fee : ℝ := 12
def gift_wrapping_fee : ℝ := 7.5
def spending_fraction : ℝ := 0.75

def initial_euro : ℝ := 25
def initial_pound : ℝ := 50
def initial_dollar : ℝ := 35
def initial_yen : ℝ := 8000
def initial_real : ℝ := 60
def initial_savings : ℝ := 105

theorem remaining_money_calculation :
  let total_dollars := initial_euro * euro_to_dollar +
                       initial_pound * pound_to_dollar +
                       initial_dollar +
                       initial_yen * yen_to_dollar +
                       initial_real * real_to_dollar +
                       initial_savings
  let after_conversion_fee := total_dollars * (1 - conversion_fee_rate)
  let after_fixed_fees := after_conversion_fee - transportation_fee - gift_wrapping_fee
  let spent_amount := after_fixed_fees * spending_fraction
  let tax_amount := spent_amount * sales_tax_rate
  let remaining_amount := after_fixed_fees - (spent_amount + tax_amount)
  remaining_amount = 73.82773125 := by sorry

end NUMINAMATH_CALUDE_remaining_money_calculation_l2258_225852


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2258_225805

def M : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 3}
def N : Set ℝ := {1, 2, 3}

theorem intersection_of_M_and_N : M ∩ N = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2258_225805
