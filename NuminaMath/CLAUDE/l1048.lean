import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_roots_equals_negative_eight_l1048_104819

/-- An odd function satisfying f(x-4) = -f(x) -/
def OddPeriodicFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (x - 4) = -f x)

theorem sum_of_roots_equals_negative_eight
  (f : ℝ → ℝ) (m : ℝ) (x₁ x₂ x₃ x₄ : ℝ)
  (hf : OddPeriodicFunction f)
  (hm : m > 0)
  (h_roots : x₁ ∈ Set.Icc (-8 : ℝ) 8 ∧
             x₂ ∈ Set.Icc (-8 : ℝ) 8 ∧
             x₃ ∈ Set.Icc (-8 : ℝ) 8 ∧
             x₄ ∈ Set.Icc (-8 : ℝ) 8)
  (h_distinct : x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄)
  (h_eq : f x₁ = m ∧ f x₂ = m ∧ f x₃ = m ∧ f x₄ = m) :
  x₁ + x₂ + x₃ + x₄ = -8 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_roots_equals_negative_eight_l1048_104819


namespace NUMINAMATH_CALUDE_inverse_function_property_l1048_104888

-- Define a function f and its inverse f_inv
variable (f : ℝ → ℝ) (f_inv : ℝ → ℝ)

-- Define the property of f and f_inv being inverse functions
def are_inverse (f : ℝ → ℝ) (f_inv : ℝ → ℝ) : Prop :=
  ∀ x, f (f_inv x) = x ∧ f_inv (f x) = x

-- State the theorem
theorem inverse_function_property
  (h1 : are_inverse f f_inv)
  (h2 : f 2 = -1) :
  f_inv (-1) = 2 :=
sorry

end NUMINAMATH_CALUDE_inverse_function_property_l1048_104888


namespace NUMINAMATH_CALUDE_combinations_count_l1048_104801

/-- Represents the cost of a pencil in cents -/
def pencil_cost : ℕ := 5

/-- Represents the cost of an eraser in cents -/
def eraser_cost : ℕ := 10

/-- Represents the cost of a notebook in cents -/
def notebook_cost : ℕ := 20

/-- Represents the total amount Mrs. Hilt has in cents -/
def total_amount : ℕ := 50

/-- Counts the number of valid combinations of items that can be purchased -/
def count_combinations : ℕ :=
  (Finset.filter (fun t : ℕ × ℕ × ℕ =>
    pencil_cost * t.1 + eraser_cost * t.2.1 + notebook_cost * t.2.2 = total_amount)
    (Finset.product (Finset.range (total_amount / pencil_cost + 1))
      (Finset.product (Finset.range (total_amount / eraser_cost + 1))
        (Finset.range (total_amount / notebook_cost + 1))))).card

theorem combinations_count :
  count_combinations = 12 := by sorry

end NUMINAMATH_CALUDE_combinations_count_l1048_104801


namespace NUMINAMATH_CALUDE_morning_travel_time_l1048_104887

/-- Proves that the time taken to move in the morning is 1 hour -/
theorem morning_travel_time (v_morning v_afternoon : ℝ) (time_diff : ℝ) 
  (h1 : v_morning = 20)
  (h2 : v_afternoon = 10)
  (h3 : time_diff = 1) :
  ∃ (t_morning : ℝ), t_morning = 1 ∧ t_morning * v_morning = (t_morning + time_diff) * v_afternoon :=
by sorry

end NUMINAMATH_CALUDE_morning_travel_time_l1048_104887


namespace NUMINAMATH_CALUDE_monster_count_monster_count_proof_l1048_104857

theorem monster_count : ℕ → Prop :=
  fun m : ℕ =>
    ∃ s : ℕ,
      s = 4 * m + 3 ∧
      s = 5 * m - 6 →
      m = 9

-- The proof is omitted
theorem monster_count_proof : monster_count 9 := by
  sorry

end NUMINAMATH_CALUDE_monster_count_monster_count_proof_l1048_104857


namespace NUMINAMATH_CALUDE_monster_hunt_l1048_104843

theorem monster_hunt (x : ℕ) : 
  (x + 2*x + 4*x + 8*x + 16*x = 62) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_monster_hunt_l1048_104843


namespace NUMINAMATH_CALUDE_prize_selection_ways_l1048_104847

/-- The number of ways to select prize winners from finalists -/
def select_winners (n : ℕ) : ℕ :=
  n * (n - 1).choose 2

/-- Theorem stating that selecting 1 first prize, 2 second prizes, and 3 third prizes
    from 6 finalists can be done in 60 ways -/
theorem prize_selection_ways : select_winners 6 = 60 := by
  sorry

end NUMINAMATH_CALUDE_prize_selection_ways_l1048_104847


namespace NUMINAMATH_CALUDE_factors_of_180_l1048_104858

/-- The number of distinct positive factors of 180 -/
def num_factors_180 : ℕ := sorry

/-- Theorem stating that the number of distinct positive factors of 180 is 18 -/
theorem factors_of_180 : num_factors_180 = 18 := by sorry

end NUMINAMATH_CALUDE_factors_of_180_l1048_104858


namespace NUMINAMATH_CALUDE_total_brownies_l1048_104820

/-- The number of brownies Tina ate per day -/
def tina_daily : ℕ := 2

/-- The number of days Tina ate brownies -/
def days : ℕ := 5

/-- The number of brownies Tina's husband ate per day -/
def husband_daily : ℕ := 1

/-- The number of brownies shared with guests -/
def shared : ℕ := 4

/-- The number of brownies left -/
def left : ℕ := 5

/-- Theorem stating the total number of brownie pieces -/
theorem total_brownies : 
  tina_daily * days + husband_daily * days + shared + left = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_brownies_l1048_104820


namespace NUMINAMATH_CALUDE_tangent_line_problem_l1048_104802

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x + x^2
def g (b : ℝ) (x : ℝ) : ℝ := Real.sin x + b * x

def is_tangent_at (l : ℝ → ℝ) (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  l x₀ = f x₀ ∧ (deriv l) x₀ = (deriv f) x₀

theorem tangent_line_problem (a b : ℝ) (l : ℝ → ℝ) :
  is_tangent_at l (f a) 0 →
  is_tangent_at l (g b) (Real.pi / 2) →
  (a = 1 ∧ b = 1) ∧
  (∀ x, l x = x + 1) ∧
  (∀ x, Real.exp x + x^2 - x - Real.sin x > 0) :=
sorry

end

end NUMINAMATH_CALUDE_tangent_line_problem_l1048_104802


namespace NUMINAMATH_CALUDE_three_digit_square_ends_with_itself_l1048_104885

theorem three_digit_square_ends_with_itself (A : ℕ) : 
  (100 ≤ A ∧ A ≤ 999) → (A^2 ≡ A [ZMOD 1000]) ↔ (A = 376 ∨ A = 625) := by
  sorry

end NUMINAMATH_CALUDE_three_digit_square_ends_with_itself_l1048_104885


namespace NUMINAMATH_CALUDE_exists_special_quadratic_trinomial_l1048_104855

/-- A quadratic trinomial function -/
def QuadraticTrinomial := ℝ → ℝ

/-- The n-th composition of a function with itself -/
def compose_n_times (f : ℝ → ℝ) : ℕ → (ℝ → ℝ)
| 0 => id
| n + 1 => f ∘ (compose_n_times f n)

/-- The number of distinct real roots of a function -/
noncomputable def num_distinct_real_roots (f : ℝ → ℝ) : ℕ := sorry

/-- The main theorem statement -/
theorem exists_special_quadratic_trinomial :
  ∃ (f : QuadraticTrinomial),
    ∀ (n : ℕ), num_distinct_real_roots (compose_n_times f n) = 2 * n :=
sorry

end NUMINAMATH_CALUDE_exists_special_quadratic_trinomial_l1048_104855


namespace NUMINAMATH_CALUDE_like_terms_imply_zero_power_l1048_104808

theorem like_terms_imply_zero_power (n : ℕ) : 
  (∃ x y, -x^(2*n-1) * y = 3 * x^8 * y) → (2*n - 9)^2013 = 0 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_imply_zero_power_l1048_104808


namespace NUMINAMATH_CALUDE_cubic_function_property_l1048_104832

theorem cubic_function_property (p q r s : ℝ) : 
  let g := fun (x : ℝ) => p * x^3 + q * x^2 + r * x + s
  (g (-1) = 2) → (g (-2) = -1) → (g 1 = -2) → 
  (9*p - 3*q + 3*r - s = -2) := by
sorry

end NUMINAMATH_CALUDE_cubic_function_property_l1048_104832


namespace NUMINAMATH_CALUDE_max_product_given_sum_l1048_104860

theorem max_product_given_sum (a b : ℝ) : 
  a > 0 → b > 0 → a + b = 2 → ∀ x y : ℝ, x > 0 → y > 0 → x + y = 2 → a * b ≥ x * y :=
sorry

end NUMINAMATH_CALUDE_max_product_given_sum_l1048_104860


namespace NUMINAMATH_CALUDE_square_sum_given_difference_and_product_l1048_104896

theorem square_sum_given_difference_and_product (a b : ℝ) 
  (h1 : a - b = 10) 
  (h2 : a * b = 55) : 
  a^2 + b^2 = 210 :=
by sorry

end NUMINAMATH_CALUDE_square_sum_given_difference_and_product_l1048_104896


namespace NUMINAMATH_CALUDE_inequality_solution_l1048_104829

theorem inequality_solution (a x : ℝ) : 
  a * x^2 - 2 ≥ 2 * x - a * x ↔ 
  (a = 0 ∧ x ≤ -1) ∨
  (a > 0 ∧ (x ≥ 2/a ∨ x ≤ -1)) ∨
  (-2 < a ∧ a < 0 ∧ 2/a ≤ x ∧ x ≤ -1) ∨
  (a = -2 ∧ x = -1) ∨
  (a < -2 ∧ -1 ≤ x ∧ x ≤ 2/a) := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_l1048_104829


namespace NUMINAMATH_CALUDE_smallest_b_for_equation_exists_solution_unique_smallest_solution_l1048_104807

theorem smallest_b_for_equation (A B : ℕ) : 
  (360 / (A * A * A / B) = 5) → B ≥ 3 :=
by
  sorry

theorem exists_solution : 
  ∃ (A B : ℕ), (360 / (A * A * A / B) = 5) ∧ B = 3 :=
by
  sorry

theorem unique_smallest_solution (A B : ℕ) : 
  (360 / (A * A * A / B) = 5) → B ≥ 3 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_b_for_equation_exists_solution_unique_smallest_solution_l1048_104807


namespace NUMINAMATH_CALUDE_inequality_solution_set_a_range_for_inequality_l1048_104873

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x + 3|
def g (x : ℝ) : ℝ := |2*x - 1|

-- Statement for the first part of the problem
theorem inequality_solution_set :
  {x : ℝ | f x < g x} = {x : ℝ | x < -2/3 ∨ x > 4} :=
sorry

-- Statement for the second part of the problem
theorem a_range_for_inequality (a : ℝ) :
  (∀ x : ℝ, 2 * f x + g x > a * x + 4) ↔ -1 < a ∧ a ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_a_range_for_inequality_l1048_104873


namespace NUMINAMATH_CALUDE_sum_of_composite_functions_l1048_104886

def p (x : ℝ) : ℝ := |x| - 3

def q (x : ℝ) : ℝ := -|x|

def x_values : List ℝ := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

theorem sum_of_composite_functions :
  (x_values.map (λ x => q (p x))).sum = -15 := by sorry

end NUMINAMATH_CALUDE_sum_of_composite_functions_l1048_104886


namespace NUMINAMATH_CALUDE_largest_share_is_18000_l1048_104876

/-- Represents the profit share of a partner -/
structure Share where
  ratio : Nat
  amount : Nat

/-- Calculates the largest share given a total profit and a list of ratios -/
def largest_share (total_profit : Nat) (ratios : List Nat) : Nat :=
  let sum_ratios := ratios.sum
  let part_value := total_profit / sum_ratios
  (ratios.maximum.getD 0) * part_value

/-- The theorem stating that the largest share is $18,000 -/
theorem largest_share_is_18000 :
  largest_share 48000 [1, 2, 3, 4, 6] = 18000 := by
  sorry

end NUMINAMATH_CALUDE_largest_share_is_18000_l1048_104876


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l1048_104892

theorem quadratic_rewrite (b : ℝ) (m : ℝ) : 
  (b > 0) → 
  (∀ x, x^2 + b*x + 36 = (x + m)^2 + 4) → 
  b = 8 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l1048_104892


namespace NUMINAMATH_CALUDE_decreasing_linear_function_condition_l1048_104865

/-- A linear function y = (m-3)x + 5 where y decreases as x increases -/
def decreasingLinearFunction (m : ℝ) : ℝ → ℝ := fun x ↦ (m - 3) * x + 5

/-- Theorem: If y decreases as x increases for the linear function y = (m-3)x + 5, then m < 3 -/
theorem decreasing_linear_function_condition (m : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → decreasingLinearFunction m x₁ > decreasingLinearFunction m x₂) →
  m < 3 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_linear_function_condition_l1048_104865


namespace NUMINAMATH_CALUDE_wallpaper_overlap_l1048_104881

theorem wallpaper_overlap (total_area : ℝ) (large_wall_area : ℝ) (two_layer_area : ℝ) (three_layer_area : ℝ) (four_layer_area : ℝ) 
  (h1 : total_area = 500)
  (h2 : large_wall_area = 280)
  (h3 : two_layer_area = 54)
  (h4 : three_layer_area = 28)
  (h5 : four_layer_area = 14) :
  ∃ (six_layer_area : ℝ), 
    six_layer_area = 9 ∧ 
    total_area = (large_wall_area - two_layer_area - three_layer_area) + 
                 2 * two_layer_area + 
                 3 * three_layer_area + 
                 4 * four_layer_area + 
                 6 * six_layer_area :=
by sorry

end NUMINAMATH_CALUDE_wallpaper_overlap_l1048_104881


namespace NUMINAMATH_CALUDE_bobs_deli_cost_l1048_104866

/-- The total cost for a customer at Bob's Deli -/
def total_cost (sandwich_price soda_price : ℕ) (sandwich_quantity soda_quantity : ℕ) (discount_threshold discount_amount : ℕ) : ℕ :=
  let initial_total := sandwich_price * sandwich_quantity + soda_price * soda_quantity
  if initial_total > discount_threshold then
    initial_total - discount_amount
  else
    initial_total

/-- The theorem stating that the customer will pay $55 in total -/
theorem bobs_deli_cost : total_cost 5 3 7 10 50 10 = 55 := by
  sorry

end NUMINAMATH_CALUDE_bobs_deli_cost_l1048_104866


namespace NUMINAMATH_CALUDE_tiles_along_width_l1048_104863

theorem tiles_along_width (area : ℝ) (tile_size : ℝ) : 
  area = 360 → tile_size = 9 → (8 : ℝ) * Real.sqrt 5 = (12 * Real.sqrt (area / 2)) / tile_size := by
  sorry

end NUMINAMATH_CALUDE_tiles_along_width_l1048_104863


namespace NUMINAMATH_CALUDE_michael_truck_meetings_l1048_104813

/-- Represents the problem of Michael and the garbage truck --/
structure GarbageTruckProblem where
  michael_speed : ℝ
  michael_delay : ℝ
  pail_spacing : ℝ
  truck_speed : ℝ
  truck_stop_duration : ℝ
  initial_distance : ℝ

/-- Calculates the number of times Michael and the truck meet --/
def number_of_meetings (problem : GarbageTruckProblem) : ℕ :=
  sorry

/-- The specific problem instance --/
def our_problem : GarbageTruckProblem :=
  { michael_speed := 3
  , michael_delay := 20
  , pail_spacing := 300
  , truck_speed := 12
  , truck_stop_duration := 45
  , initial_distance := 300 }

/-- Theorem stating that Michael and the truck meet exactly 6 times --/
theorem michael_truck_meetings :
  number_of_meetings our_problem = 6 := by
  sorry

end NUMINAMATH_CALUDE_michael_truck_meetings_l1048_104813


namespace NUMINAMATH_CALUDE_range_of_a_l1048_104811

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

-- State the theorem
theorem range_of_a (a : ℝ) : (¬(p a) ∧ q a) → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1048_104811


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_bound_l1048_104899

theorem sum_of_reciprocals_bound (a b c d : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_sum : a + b + c + d = 1) : 
  1 / (4*a + 3*b + c) + 1 / (3*a + b + 4*d) + 
  1 / (a + 4*c + 3*d) + 1 / (4*b + 3*c + d) ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_bound_l1048_104899


namespace NUMINAMATH_CALUDE_min_value_of_parallel_lines_l1048_104889

theorem min_value_of_parallel_lines (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_parallel : a * (b - 3) - 2 * b = 0) : 
  (∀ x y : ℝ, 2 * a + 3 * b ≥ 25) ∧ (∃ x y : ℝ, 2 * a + 3 * b = 25) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_parallel_lines_l1048_104889


namespace NUMINAMATH_CALUDE_garbage_collection_theorem_l1048_104828

/-- The amount of garbage collected by four people given specific relationships between their collections. -/
def total_garbage_collected (daliah_amount : ℝ) : ℝ := 
  let dewei_amount := daliah_amount - 2
  let zane_amount := 4 * dewei_amount
  let bela_amount := zane_amount + 3.75
  daliah_amount + dewei_amount + zane_amount + bela_amount

/-- Theorem stating that the total amount of garbage collected is 160.75 pounds when Daliah collects 17.5 pounds. -/
theorem garbage_collection_theorem : 
  total_garbage_collected 17.5 = 160.75 := by
  sorry

#eval total_garbage_collected 17.5

end NUMINAMATH_CALUDE_garbage_collection_theorem_l1048_104828


namespace NUMINAMATH_CALUDE_circle_center_l1048_104848

/-- A circle passes through (0,1) and is tangent to y = (x-1)^2 at (3,4). Its center is (-2, 15/2). -/
theorem circle_center (c : ℝ × ℝ) : 
  (∀ (x y : ℝ), (x - c.1)^2 + (y - c.2)^2 = (c.1 - 3)^2 + (c.2 - 4)^2 → 
    (x = 0 ∧ y = 1) ∨ (x = 3 ∧ y = 4)) →
  (∀ (x : ℝ), (x - 3)^2 + (((x - 1)^2 - 4)^2) / 16 = (c.1 - 3)^2 + (c.2 - 4)^2) →
  c = (-2, 15/2) := by
sorry

end NUMINAMATH_CALUDE_circle_center_l1048_104848


namespace NUMINAMATH_CALUDE_inequality_proof_equality_condition_l1048_104821

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a^2 + b^2 + c^2 + d^2)^2 ≥ (a+b)*(b+c)*(c+d)*(d+a) :=
by sorry

theorem equality_condition (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a^2 + b^2 + c^2 + d^2)^2 = (a+b)*(b+c)*(c+d)*(d+a) ↔ a = b ∧ b = c ∧ c = d :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_equality_condition_l1048_104821


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l1048_104878

/-- Given a geometric sequence of positive numbers where the fourth term is 16
    and the ninth term is 8, the sixth term is equal to 16 * (4^(1/5)) -/
theorem geometric_sequence_sixth_term
  (a : ℝ → ℝ)  -- The sequence
  (r : ℝ)      -- Common ratio
  (h_positive : ∀ n, a n > 0)  -- All terms are positive
  (h_geometric : ∀ n, a (n + 1) = a n * r)  -- It's a geometric sequence
  (h_fourth : a 4 = 16)  -- The fourth term is 16
  (h_ninth : a 9 = 8)  -- The ninth term is 8
  : a 6 = 16 * (4^(1/5)) := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l1048_104878


namespace NUMINAMATH_CALUDE_x_twelfth_power_l1048_104827

theorem x_twelfth_power (x : ℂ) (h : x + 1/x = 2 * Real.sqrt 2) : x^12 = 14449 := by
  sorry

end NUMINAMATH_CALUDE_x_twelfth_power_l1048_104827


namespace NUMINAMATH_CALUDE_reading_rate_difference_l1048_104844

-- Define the given information
def songhee_pages : ℕ := 288
def songhee_days : ℕ := 12
def eunju_pages : ℕ := 243
def eunju_days : ℕ := 9

-- Define the daily reading rates
def songhee_rate : ℚ := songhee_pages / songhee_days
def eunju_rate : ℚ := eunju_pages / eunju_days

-- Theorem statement
theorem reading_rate_difference : eunju_rate - songhee_rate = 3 := by
  sorry

end NUMINAMATH_CALUDE_reading_rate_difference_l1048_104844


namespace NUMINAMATH_CALUDE_evening_pages_read_l1048_104850

/-- Given a person who reads books with the following conditions:
  * Reads twice a day (morning and evening)
  * Reads 5 pages in the morning
  * Reads at this rate for a week (7 days)
  * Reads a total of 105 pages in a week
This theorem proves that the number of pages read in the evening is 10. -/
theorem evening_pages_read (morning_pages : ℕ) (total_pages : ℕ) (days : ℕ) :
  morning_pages = 5 →
  days = 7 →
  total_pages = 105 →
  ∃ (evening_pages : ℕ), 
    days * (morning_pages + evening_pages) = total_pages ∧ 
    evening_pages = 10 := by
  sorry

end NUMINAMATH_CALUDE_evening_pages_read_l1048_104850


namespace NUMINAMATH_CALUDE_geometric_distribution_variance_l1048_104840

/-- A random variable following a geometric distribution with parameter p -/
def GeometricDistribution (p : ℝ) := { X : ℝ → ℝ // 0 < p ∧ p ≤ 1 }

/-- The variance of a random variable -/
def variance (X : ℝ → ℝ) : ℝ := sorry

/-- The theorem stating that the variance of a geometric distribution is (1-p)/p^2 -/
theorem geometric_distribution_variance (p : ℝ) (X : GeometricDistribution p) :
  variance X.val = (1 - p) / p^2 := by sorry

end NUMINAMATH_CALUDE_geometric_distribution_variance_l1048_104840


namespace NUMINAMATH_CALUDE_oddSumProbability_l1048_104805

/-- Represents an unfair die where even numbers are 5 times as likely as odd numbers -/
structure UnfairDie where
  /-- Probability of rolling an odd number -/
  oddProb : ℝ
  /-- Probability of rolling an even number -/
  evenProb : ℝ
  /-- Even probability is 5 times odd probability -/
  evenOddRatio : evenProb = 5 * oddProb
  /-- Total probability is 1 -/
  totalProb : oddProb + evenProb = 1

/-- The probability of rolling an odd sum with two rolls of the unfair die -/
def oddSumProb (d : UnfairDie) : ℝ :=
  2 * d.oddProb * d.evenProb

theorem oddSumProbability (d : UnfairDie) : oddSumProb d = 5 / 18 := by
  sorry


end NUMINAMATH_CALUDE_oddSumProbability_l1048_104805


namespace NUMINAMATH_CALUDE_same_color_probability_is_59_225_l1048_104891

/-- Represents a 30-sided die with colored sides -/
structure ColoredDie :=
  (blue : Nat)
  (yellow : Nat)
  (green : Nat)
  (purple : Nat)
  (total : Nat)
  (side_sum : blue + yellow + green + purple = total)

/-- The probability of two dice showing the same color -/
def same_color_probability (d : ColoredDie) : Rat :=
  let blue_prob := (d.blue * d.blue : Rat) / (d.total * d.total)
  let yellow_prob := (d.yellow * d.yellow : Rat) / (d.total * d.total)
  let green_prob := (d.green * d.green : Rat) / (d.total * d.total)
  let purple_prob := (d.purple * d.purple : Rat) / (d.total * d.total)
  blue_prob + yellow_prob + green_prob + purple_prob

/-- The specific 30-sided die described in the problem -/
def problem_die : ColoredDie :=
  { blue := 6
    yellow := 8
    green := 10
    purple := 6
    total := 30
    side_sum := by norm_num }

/-- Theorem stating the probability of two problem dice showing the same color -/
theorem same_color_probability_is_59_225 :
  same_color_probability problem_die = 59 / 225 := by
  sorry


end NUMINAMATH_CALUDE_same_color_probability_is_59_225_l1048_104891


namespace NUMINAMATH_CALUDE_mutually_exclusive_head_l1048_104835

-- Define the set of people
variable (People : Type)

-- Define the property of standing at the head of the line
variable (stands_at_head : People → Prop)

-- Define A and B as specific people
variable (A B : People)

-- Axiom: A and B are distinct people
axiom A_neq_B : A ≠ B

-- Axiom: Only one person can stand at the head of the line
axiom one_at_head : ∀ (x y : People), stands_at_head x ∧ stands_at_head y → x = y

-- Theorem: The events "A stands at the head of the line" and "B stands at the head of the line" are mutually exclusive
theorem mutually_exclusive_head : 
  ¬(stands_at_head A ∧ stands_at_head B) :=
sorry

end NUMINAMATH_CALUDE_mutually_exclusive_head_l1048_104835


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l1048_104882

theorem arithmetic_sequence_length :
  ∀ (a₁ d n : ℤ),
    a₁ = -48 →
    d = 8 →
    a₁ + (n - 1) * d = 80 →
    n = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l1048_104882


namespace NUMINAMATH_CALUDE_quadrilateral_inequalities_l1048_104826

-- Define a structure for a convex quadrilateral
structure ConvexQuadrilateral :=
  (a b c d t : ℝ)
  (area_positive : t > 0)
  (sides_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)

-- Define what it means for a quadrilateral to be cyclic
def is_cyclic (q : ConvexQuadrilateral) : Prop := sorry

-- State the theorem
theorem quadrilateral_inequalities (q : ConvexQuadrilateral) :
  (2 * q.t ≤ q.a * q.b + q.c * q.d) ∧
  (2 * q.t ≤ q.a * q.c + q.b * q.d) ∧
  ((2 * q.t = q.a * q.b + q.c * q.d) ∨ (2 * q.t = q.a * q.c + q.b * q.d) → is_cyclic q) :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_inequalities_l1048_104826


namespace NUMINAMATH_CALUDE_coles_return_speed_coles_return_speed_is_120_l1048_104825

/-- Calculates the average speed of the return trip given the conditions of Cole's journey --/
theorem coles_return_speed (speed_to_work : ℝ) (total_time : ℝ) (time_to_work : ℝ) : ℝ :=
  let distance_to_work := speed_to_work * time_to_work
  let time_to_return := total_time - time_to_work
  distance_to_work / time_to_return

/-- Proves that Cole's average speed driving back home is 120 km/h --/
theorem coles_return_speed_is_120 :
  coles_return_speed 80 2 (72 / 60) = 120 := by
  sorry

end NUMINAMATH_CALUDE_coles_return_speed_coles_return_speed_is_120_l1048_104825


namespace NUMINAMATH_CALUDE_jelly_bean_distribution_l1048_104841

theorem jelly_bean_distribution (total_jelly_beans : ℕ) (remaining_jelly_beans : ℕ) (boy_girl_difference : ℕ) : 
  total_jelly_beans = 500 →
  remaining_jelly_beans = 10 →
  boy_girl_difference = 4 →
  ∃ (girls boys : ℕ),
    girls + boys = 32 ∧
    boys = girls + boy_girl_difference ∧
    girls * girls + boys * boys = total_jelly_beans - remaining_jelly_beans :=
by sorry

end NUMINAMATH_CALUDE_jelly_bean_distribution_l1048_104841


namespace NUMINAMATH_CALUDE_more_larger_boxes_l1048_104853

/-- Represents the number of glasses in a small box -/
def small_box : ℕ := 12

/-- Represents the number of glasses in a large box -/
def large_box : ℕ := 16

/-- Represents the average number of glasses per box -/
def average_glasses : ℕ := 15

/-- Represents the total number of glasses -/
def total_glasses : ℕ := 480

theorem more_larger_boxes (s l : ℕ) : 
  s * small_box + l * large_box = total_glasses →
  (s + l : ℚ) = (total_glasses : ℚ) / average_glasses →
  l > s →
  l - s = 16 := by
  sorry

end NUMINAMATH_CALUDE_more_larger_boxes_l1048_104853


namespace NUMINAMATH_CALUDE_money_distribution_l1048_104861

theorem money_distribution (a b c : ℕ) 
  (total : a + b + c = 450)
  (ac_sum : a + c = 200)
  (bc_sum : b + c = 350) :
  c = 100 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l1048_104861


namespace NUMINAMATH_CALUDE_oil_price_reduction_l1048_104830

/-- Calculates the percentage reduction in oil price given the original amount, additional amount, total cost, and reduced price. -/
theorem oil_price_reduction 
  (X : ℝ)              -- Original amount of oil in kg
  (additional : ℝ)     -- Additional amount of oil in kg
  (total_cost : ℝ)     -- Total cost in Rs
  (reduced_price : ℝ)  -- Reduced price per kg in Rs
  (h1 : additional = 5)
  (h2 : total_cost = 600)
  (h3 : reduced_price = 30)
  (h4 : X + additional = total_cost / reduced_price)
  (h5 : X = total_cost / (total_cost / X))
  : (1 - reduced_price / (total_cost / X)) * 100 = 25 := by
  sorry


end NUMINAMATH_CALUDE_oil_price_reduction_l1048_104830


namespace NUMINAMATH_CALUDE_x_squared_plus_reciprocal_l1048_104831

theorem x_squared_plus_reciprocal (x : ℝ) (h : 54 = x^4 + 1/x^4) :
  x^2 + 1/x^2 = Real.sqrt 56 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_reciprocal_l1048_104831


namespace NUMINAMATH_CALUDE_certain_number_proof_l1048_104862

theorem certain_number_proof : ∃ x : ℝ, (3889 + 12.952 - x = 3854.002) ∧ (x = 47.95) := by sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1048_104862


namespace NUMINAMATH_CALUDE_pure_imaginary_ratio_l1048_104884

theorem pure_imaginary_ratio (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : ∃ (y : ℝ), (3 - 4*I) * (a + b*I) = y*I) : 
  a / b = -4 / 3 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_ratio_l1048_104884


namespace NUMINAMATH_CALUDE_original_mixture_volume_l1048_104814

theorem original_mixture_volume 
  (original_alcohol_percentage : ℝ)
  (added_water : ℝ)
  (new_alcohol_percentage : ℝ)
  (h1 : original_alcohol_percentage = 0.25)
  (h2 : added_water = 3)
  (h3 : new_alcohol_percentage = 0.20833333333333336)
  : ∃ (original_volume : ℝ),
    original_volume * original_alcohol_percentage / (original_volume + added_water) = new_alcohol_percentage ∧
    original_volume = 15 :=
by sorry

end NUMINAMATH_CALUDE_original_mixture_volume_l1048_104814


namespace NUMINAMATH_CALUDE_ships_initial_distance_l1048_104822

/-- The initial distance between two ships moving towards a port -/
def initial_distance : ℝ := 240

/-- The distance traveled by the second ship when a right triangle is formed -/
def right_triangle_distance : ℝ := 80

/-- The remaining distance for the second ship when the first ship reaches the port -/
def remaining_distance : ℝ := 120

theorem ships_initial_distance :
  ∃ (v₁ v₂ : ℝ), v₁ > 0 ∧ v₂ > 0 ∧
  (initial_distance - v₁ * (right_triangle_distance / v₂))^2 + right_triangle_distance^2 = initial_distance^2 ∧
  (initial_distance / v₁) * v₂ = initial_distance - remaining_distance :=
by sorry

#check ships_initial_distance

end NUMINAMATH_CALUDE_ships_initial_distance_l1048_104822


namespace NUMINAMATH_CALUDE_increasing_symmetric_function_inequality_l1048_104869

theorem increasing_symmetric_function_inequality 
  (f : ℝ → ℝ) 
  (h_increasing : ∀ x y, x ≥ 2 → y ≥ 2 → x < y → f x < f y) 
  (h_symmetric : ∀ x, f (2 + x) = f (2 - x)) 
  (h_inequality : f (1 - 2 * x^2) < f (1 + 2*x - x^2)) :
  -2 < x ∧ x < 0 :=
by sorry

end NUMINAMATH_CALUDE_increasing_symmetric_function_inequality_l1048_104869


namespace NUMINAMATH_CALUDE_book_sale_gain_percentage_l1048_104800

/-- Calculates the desired gain percentage for a book sale --/
theorem book_sale_gain_percentage 
  (loss_price : ℝ) 
  (loss_percentage : ℝ) 
  (desired_price : ℝ) : 
  loss_price = 800 ∧ 
  loss_percentage = 20 ∧ 
  desired_price = 1100 → 
  (desired_price - loss_price / (1 - loss_percentage / 100)) / 
  (loss_price / (1 - loss_percentage / 100)) * 100 = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_book_sale_gain_percentage_l1048_104800


namespace NUMINAMATH_CALUDE_sin_sum_alpha_beta_l1048_104877

theorem sin_sum_alpha_beta (α β : ℝ) 
  (h1 : Real.sin α + Real.cos β = 1) 
  (h2 : Real.cos α + Real.sin β = 0) : 
  Real.sin (α + β) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_alpha_beta_l1048_104877


namespace NUMINAMATH_CALUDE_rotate_D_180_about_origin_l1048_104815

def rotate_180_about_origin (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

theorem rotate_D_180_about_origin :
  let D : ℝ × ℝ := (-6, 2)
  rotate_180_about_origin D = (6, -2) := by
  sorry

end NUMINAMATH_CALUDE_rotate_D_180_about_origin_l1048_104815


namespace NUMINAMATH_CALUDE_mass_of_man_on_boat_l1048_104897

/-- The mass of a man who causes a boat to sink by a certain amount. -/
def mass_of_man (boat_length boat_breadth boat_sinking water_density : ℝ) : ℝ :=
  boat_length * boat_breadth * boat_sinking * water_density

/-- Theorem stating the mass of the man in the given scenario. -/
theorem mass_of_man_on_boat :
  let boat_length : ℝ := 4
  let boat_breadth : ℝ := 3
  let boat_sinking : ℝ := 0.01  -- 1 cm in meters
  let water_density : ℝ := 1000 -- kg/m³
  mass_of_man boat_length boat_breadth boat_sinking water_density = 120 := by
  sorry

end NUMINAMATH_CALUDE_mass_of_man_on_boat_l1048_104897


namespace NUMINAMATH_CALUDE_doctor_team_count_l1048_104868

/-- The number of ways to choose a team of doctors under specific conditions -/
def choose_doctor_team (total_doctors : ℕ) (pediatricians surgeons general_practitioners : ℕ) 
  (team_size : ℕ) : ℕ :=
  (pediatricians.choose 1) * (surgeons.choose 1) * (general_practitioners.choose 1) * 
  ((total_doctors - 3).choose (team_size - 3))

/-- Theorem stating the number of ways to choose a team of 5 doctors from 25 doctors, 
    with specific specialty requirements -/
theorem doctor_team_count : 
  choose_doctor_team 25 5 10 10 5 = 115500 := by
  sorry

end NUMINAMATH_CALUDE_doctor_team_count_l1048_104868


namespace NUMINAMATH_CALUDE_log_inequality_l1048_104803

theorem log_inequality (c a b : ℝ) (h1 : 0 < c) (h2 : c < 1) (h3 : b > 1) (h4 : a > b) :
  Real.log c / Real.log a > Real.log c / Real.log b := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l1048_104803


namespace NUMINAMATH_CALUDE_travel_ratio_l1048_104810

-- Define variables for the number of countries each person traveled to
def george_countries : ℕ := 6
def zack_countries : ℕ := 18

-- Define functions for other travelers based on the given conditions
def joseph_countries : ℕ := george_countries / 2
def patrick_countries : ℕ := zack_countries / 2

-- Define the ratio function
def ratio (a b : ℕ) : ℚ := a / b

-- Theorem statement
theorem travel_ratio : ratio patrick_countries joseph_countries = 3 := by sorry

end NUMINAMATH_CALUDE_travel_ratio_l1048_104810


namespace NUMINAMATH_CALUDE_truck_problem_l1048_104871

theorem truck_problem (T b c : ℝ) (hT : T > 0) (hb : b > 0) (hc : c > 0) :
  let x := (b * c + Real.sqrt (b^2 * c^2 + 4 * b * c * T)) / (2 * c)
  x * (x - b) * c = T * x ∧ (x - b) * (T / x + c) = T :=
by sorry

end NUMINAMATH_CALUDE_truck_problem_l1048_104871


namespace NUMINAMATH_CALUDE_no_interchange_possible_l1048_104839

/-- Represents the circular arrangement of three tiles -/
inductive CircularArrangement
  | ABC
  | BCA
  | CAB

/-- Represents a move that slides a tile to an adjacent vacant space -/
inductive Move
  | Left
  | Right

/-- Applies a move to a circular arrangement -/
def applyMove (arr : CircularArrangement) (m : Move) : CircularArrangement :=
  match arr, m with
  | CircularArrangement.ABC, Move.Right => CircularArrangement.BCA
  | CircularArrangement.BCA, Move.Right => CircularArrangement.CAB
  | CircularArrangement.CAB, Move.Right => CircularArrangement.ABC
  | CircularArrangement.ABC, Move.Left => CircularArrangement.CAB
  | CircularArrangement.BCA, Move.Left => CircularArrangement.ABC
  | CircularArrangement.CAB, Move.Left => CircularArrangement.BCA

/-- Applies a sequence of moves to a circular arrangement -/
def applyMoves (arr : CircularArrangement) (moves : List Move) : CircularArrangement :=
  match moves with
  | [] => arr
  | m :: ms => applyMoves (applyMove arr m) ms

/-- Theorem stating that it's impossible to interchange 1 and 3 -/
theorem no_interchange_possible (moves : List Move) :
  applyMoves CircularArrangement.ABC moves ≠ CircularArrangement.BCA :=
sorry

end NUMINAMATH_CALUDE_no_interchange_possible_l1048_104839


namespace NUMINAMATH_CALUDE_right_triangle_area_l1048_104859

theorem right_triangle_area (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  a + b = 24 →
  c = 24 →
  a^2 + c^2 = (24 + b)^2 →
  (1/2) * a * c = 216 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1048_104859


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1048_104816

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_sum : a 1 + 2 * a 8 + a 15 = 96) :
  2 * a 9 - a 10 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1048_104816


namespace NUMINAMATH_CALUDE_smallest_n_same_last_two_digits_l1048_104854

theorem smallest_n_same_last_two_digits : ∃ n : ℕ+, 
  (∀ m : ℕ+, m < n → ¬(107 * m ≡ m [ZMOD 100])) ∧ 
  (107 * n ≡ n [ZMOD 100]) ∧
  n = 50 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_same_last_two_digits_l1048_104854


namespace NUMINAMATH_CALUDE_claire_profit_is_60_l1048_104864

def claire_profit (total_loaves : ℕ) (morning_price afternoon_price late_price cost_per_loaf fixed_cost : ℚ) : ℚ :=
  let morning_sales := total_loaves / 3
  let afternoon_sales := (total_loaves - morning_sales) / 2
  let late_sales := total_loaves - morning_sales - afternoon_sales
  let total_revenue := morning_sales * morning_price + afternoon_sales * afternoon_price + late_sales * late_price
  let total_cost := total_loaves * cost_per_loaf + fixed_cost
  total_revenue - total_cost

theorem claire_profit_is_60 :
  claire_profit 60 3 2 (3/2) 1 10 = 60 := by
  sorry

end NUMINAMATH_CALUDE_claire_profit_is_60_l1048_104864


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l1048_104812

theorem product_of_three_numbers (a b c : ℝ) 
  (sum_eq : a + b + c = 210)
  (rel_b : 5 * a = b - 11)
  (rel_c : 5 * a = c + 11) :
  a * b * c = 168504 := by
sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l1048_104812


namespace NUMINAMATH_CALUDE_necessary_implies_sufficient_l1048_104894

-- Define what it means for q to be a necessary condition for p
def necessary_condition (p q : Prop) : Prop :=
  p → q

-- Define what it means for p to be a sufficient condition for q
def sufficient_condition (p q : Prop) : Prop :=
  p → q

-- Theorem statement
theorem necessary_implies_sufficient (p q : Prop) 
  (h : necessary_condition p q) : sufficient_condition p q :=
by
  sorry


end NUMINAMATH_CALUDE_necessary_implies_sufficient_l1048_104894


namespace NUMINAMATH_CALUDE_tan_beta_value_l1048_104879

theorem tan_beta_value (α β : Real) 
  (h1 : Real.tan α = 3) 
  (h2 : Real.tan (α + β) = 2) : 
  Real.tan β = -1/7 := by sorry

end NUMINAMATH_CALUDE_tan_beta_value_l1048_104879


namespace NUMINAMATH_CALUDE_sin_cos_sum_equals_negative_one_l1048_104893

theorem sin_cos_sum_equals_negative_one :
  Real.sin (11 / 6 * π) + Real.cos (10 / 3 * π) = -1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equals_negative_one_l1048_104893


namespace NUMINAMATH_CALUDE_sqrt_one_plus_cos_alpha_l1048_104849

theorem sqrt_one_plus_cos_alpha (α : Real) (h : π < α ∧ α < 2*π) :
  Real.sqrt (1 + Real.cos α) = -Real.sqrt 2 * Real.cos (α/2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_one_plus_cos_alpha_l1048_104849


namespace NUMINAMATH_CALUDE_bush_height_after_two_years_l1048_104842

def bush_height (initial_height : ℝ) (years : ℕ) : ℝ :=
  initial_height * (3 ^ years)

theorem bush_height_after_two_years 
  (h : bush_height 1 5 = 81) : 
  bush_height 1 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_bush_height_after_two_years_l1048_104842


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1048_104809

def set_A : Set ℝ := {y | ∃ x, y = Real.log x}
def set_B : Set ℝ := {x | x ≥ 0}

theorem intersection_of_A_and_B : set_A ∩ set_B = Set.Ici (0 : ℝ) := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1048_104809


namespace NUMINAMATH_CALUDE_farmer_ploughing_problem_l1048_104895

theorem farmer_ploughing_problem (planned_rate : ℕ) (actual_rate : ℕ) (extra_days : ℕ) (area_left : ℕ) (total_area : ℕ) :
  planned_rate = 120 →
  actual_rate = 85 →
  extra_days = 2 →
  area_left = 40 →
  total_area = 720 →
  ∃ (planned_days : ℕ), 
    planned_days * planned_rate = total_area ∧
    (planned_days + extra_days) * actual_rate + area_left = total_area ∧
    planned_days = 6 :=
by sorry

end NUMINAMATH_CALUDE_farmer_ploughing_problem_l1048_104895


namespace NUMINAMATH_CALUDE_sin_equation_solution_l1048_104880

theorem sin_equation_solution (n : ℤ) :
  -180 ≤ n ∧ n ≤ 180 →
  Real.sin (n * π / 180) = Real.sin (680 * π / 180) →
  n = 40 ∨ n = 140 := by
sorry

end NUMINAMATH_CALUDE_sin_equation_solution_l1048_104880


namespace NUMINAMATH_CALUDE_not_perfect_square_factorial_product_l1048_104883

theorem not_perfect_square_factorial_product (n : ℕ) (h : n ∈ ({19, 20, 21, 22, 23} : Set ℕ)) :
  ¬ ∃ m : ℕ, (n.factorial * (n + 1).factorial) / 2 = m ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_factorial_product_l1048_104883


namespace NUMINAMATH_CALUDE_a_value_l1048_104852

/-- Custom operation @ for positive integers -/
def custom_op (k j : ℕ+) : ℕ+ :=
  sorry

/-- The value of b -/
def b : ℕ := 2120

/-- The ratio q -/
def q : ℚ := 1/2

/-- The value of a -/
def a : ℕ := 1060

/-- Theorem stating that a = 1060 given the conditions -/
theorem a_value : a = 1060 :=
  sorry

end NUMINAMATH_CALUDE_a_value_l1048_104852


namespace NUMINAMATH_CALUDE_right_triangle_complex_roots_l1048_104851

theorem right_triangle_complex_roots : 
  ∃! (s : Finset ℂ), 
    (∀ z ∈ s, z ≠ 0 ∧ (z.re * (z^3).re + z.im * (z^3).im = 0)) ∧ 
    s.card = 2 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_complex_roots_l1048_104851


namespace NUMINAMATH_CALUDE_quartic_roots_l1048_104874

theorem quartic_roots : 
  let f : ℝ → ℝ := λ x ↦ 3*x^4 + 2*x^3 - 8*x^2 + 2*x + 3
  ∃ (a b c d : ℝ), 
    (a = (1 - Real.sqrt 43 + 2*Real.sqrt 34) / 6) ∧
    (b = (1 - Real.sqrt 43 - 2*Real.sqrt 34) / 6) ∧
    (c = (1 + Real.sqrt 43 + 2*Real.sqrt 34) / 6) ∧
    (d = (1 + Real.sqrt 43 - 2*Real.sqrt 34) / 6) ∧
    (∀ x : ℝ, f x = 0 ↔ (x = a ∨ x = b ∨ x = c ∨ x = d)) :=
by sorry

end NUMINAMATH_CALUDE_quartic_roots_l1048_104874


namespace NUMINAMATH_CALUDE_base8_237_equals_base10_159_l1048_104818

/-- Converts a three-digit number from base 8 to base 10 -/
def base8ToBase10 (hundreds : Nat) (tens : Nat) (ones : Nat) : Nat :=
  hundreds * 8^2 + tens * 8^1 + ones * 8^0

/-- The base 8 number 237 is equal to 159 in base 10 -/
theorem base8_237_equals_base10_159 : base8ToBase10 2 3 7 = 159 := by
  sorry

end NUMINAMATH_CALUDE_base8_237_equals_base10_159_l1048_104818


namespace NUMINAMATH_CALUDE_complex_number_modulus_l1048_104867

theorem complex_number_modulus : ∃ (z : ℂ), z = (2 * Complex.I) / (1 + Complex.I) ∧ Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_modulus_l1048_104867


namespace NUMINAMATH_CALUDE_set_equality_implies_a_value_l1048_104833

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {a, a^2}
def B (b : ℝ) : Set ℝ := {1, b}

-- State the theorem
theorem set_equality_implies_a_value (a b : ℝ) :
  A a = B b → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_a_value_l1048_104833


namespace NUMINAMATH_CALUDE_absolute_value_equation_l1048_104870

theorem absolute_value_equation (x : ℝ) (h : |2 - x| = 2 + |x|) : |2 - x| = 2 - x := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_l1048_104870


namespace NUMINAMATH_CALUDE_side_length_eq_twice_radius_l1048_104834

/-- A square with a circle inscribed such that the circle is tangent to two adjacent sides
    and passes through one vertex of the square. -/
structure InscribedCircleSquare where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The side length of the square -/
  s : ℝ
  /-- The circle is tangent to two adjacent sides of the square -/
  tangent_to_sides : True
  /-- The circle passes through one vertex of the square -/
  passes_through_vertex : True

/-- The side length of a square with an inscribed circle tangent to two adjacent sides
    and passing through one vertex is equal to twice the radius of the circle. -/
theorem side_length_eq_twice_radius (square : InscribedCircleSquare) :
  square.s = 2 * square.r := by
  sorry

end NUMINAMATH_CALUDE_side_length_eq_twice_radius_l1048_104834


namespace NUMINAMATH_CALUDE_equation_solution_l1048_104898

theorem equation_solution (x : ℝ) (h1 : x > 0) (h2 : (x - 5) / 10 = 5 / (x - 10)) : x = 15 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1048_104898


namespace NUMINAMATH_CALUDE_reeya_second_subject_score_l1048_104838

/-- Given Reeya's scores in 4 subjects and her average score, prove the score of the second subject. -/
theorem reeya_second_subject_score (score1 score2 score3 score4 : ℕ) (average : ℚ) :
  score1 = 55 →
  score3 = 82 →
  score4 = 55 →
  average = 67 →
  (score1 + score2 + score3 + score4 : ℚ) / 4 = average →
  score2 = 76 := by
sorry

end NUMINAMATH_CALUDE_reeya_second_subject_score_l1048_104838


namespace NUMINAMATH_CALUDE_kenya_peanuts_l1048_104890

theorem kenya_peanuts (jose_peanuts : ℕ) (kenya_extra : ℕ) : 
  jose_peanuts = 85 → kenya_extra = 48 → jose_peanuts + kenya_extra = 133 := by
  sorry

end NUMINAMATH_CALUDE_kenya_peanuts_l1048_104890


namespace NUMINAMATH_CALUDE_line_through_point_l1048_104804

theorem line_through_point (a : ℝ) (h : a ≠ 0) :
  (∀ x y : ℝ, y = 5 * x + a → (x = a → y = a^2)) → a = 6 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l1048_104804


namespace NUMINAMATH_CALUDE_triangle_tiling_exists_quadrilateral_tiling_exists_hexagon_tiling_exists_l1048_104824

/-- A polygon in the plane -/
structure Polygon :=
  (vertices : List (ℝ × ℝ))

/-- A tiling of the plane using a given polygon -/
def Tiling (p : Polygon) := 
  List (ℝ × ℝ) → Prop

/-- Predicate for a centrally symmetric hexagon -/
def IsCentrallySymmetricHexagon (p : Polygon) : Prop :=
  p.vertices.length = 6 ∧ 
  ∃ center : ℝ × ℝ, ∀ v ∈ p.vertices, 
    ∃ v' ∈ p.vertices, v' = (2 * center.1 - v.1, 2 * center.2 - v.2)

/-- Theorem stating that any triangle can tile the plane -/
theorem triangle_tiling_exists (t : Polygon) (h : t.vertices.length = 3) :
  ∃ tiling : Tiling t, True :=
sorry

/-- Theorem stating that any quadrilateral can tile the plane -/
theorem quadrilateral_tiling_exists (q : Polygon) (h : q.vertices.length = 4) :
  ∃ tiling : Tiling q, True :=
sorry

/-- Theorem stating that any centrally symmetric hexagon can tile the plane -/
theorem hexagon_tiling_exists (h : Polygon) (symmetric : IsCentrallySymmetricHexagon h) :
  ∃ tiling : Tiling h, True :=
sorry

end NUMINAMATH_CALUDE_triangle_tiling_exists_quadrilateral_tiling_exists_hexagon_tiling_exists_l1048_104824


namespace NUMINAMATH_CALUDE_y_divisibility_l1048_104846

def y : ℕ := 72 + 108 + 144 + 180 + 324 + 396 + 3600

theorem y_divisibility :
  (∃ k : ℕ, y = 6 * k) ∧
  (∃ k : ℕ, y = 12 * k) ∧
  (∃ k : ℕ, y = 18 * k) ∧
  (∃ k : ℕ, y = 36 * k) := by
  sorry

end NUMINAMATH_CALUDE_y_divisibility_l1048_104846


namespace NUMINAMATH_CALUDE_coprime_condition_l1048_104806

theorem coprime_condition (a b c d : ℤ) (h : Nat.gcd a.natAbs b.natAbs = 1 ∧ 
  Nat.gcd c.natAbs d.natAbs = 1 ∧ Nat.gcd a.natAbs c.natAbs = 1) : 
  (∀ (p : ℕ), Nat.Prime p → p ∣ (a * d - b * c).natAbs → (p ∣ a.natAbs ∨ p ∣ c.natAbs)) ↔ 
  (∀ (n : ℤ), Nat.gcd (a * n + b).natAbs (c * n + d).natAbs = 1) := by
sorry

end NUMINAMATH_CALUDE_coprime_condition_l1048_104806


namespace NUMINAMATH_CALUDE_inequality_proof_l1048_104837

theorem inequality_proof (x : ℝ) (h1 : x > -1) (h2 : x ≠ 0) :
  (2 * |x|) / (2 + x) < |Real.log (1 + x)| ∧ |Real.log (1 + x)| < |x| / Real.sqrt (1 + x) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1048_104837


namespace NUMINAMATH_CALUDE_smaller_number_in_ratio_l1048_104875

theorem smaller_number_in_ratio (p q k : ℝ) (hp : p > 0) (hq : q > 0) : 
  p / q = 3 / 5 → p^2 + q^2 = 2 * k → min p q = 3 * Real.sqrt (k / 17) := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_in_ratio_l1048_104875


namespace NUMINAMATH_CALUDE_polynomial_remainder_l1048_104836

theorem polynomial_remainder (s : ℤ) : (s^11 + 1) % (s + 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l1048_104836


namespace NUMINAMATH_CALUDE_square_and_cube_roots_problem_l1048_104856

theorem square_and_cube_roots_problem (a b : ℝ) : 
  (∃ (x : ℝ), x > 0 ∧ (3*a - 14)^2 = x ∧ (a + 2)^2 = x) → 
  (b + 11)^(1/3) = -3 → 
  a = 3 ∧ b = -38 ∧ (1 - (a + b))^(1/2) = 6 ∨ (1 - (a + b))^(1/2) = -6 :=
by sorry

end NUMINAMATH_CALUDE_square_and_cube_roots_problem_l1048_104856


namespace NUMINAMATH_CALUDE_millionaire_hat_sale_l1048_104823

/-- Proves that the fraction of hats sold is 2/3 given the conditions of the problem -/
theorem millionaire_hat_sale (H : ℝ) (h1 : H > 0) : 
  let brown_hats := (1/4 : ℝ) * H
  let sold_brown_hats := (4/5 : ℝ) * brown_hats
  let remaining_hats := H - sold_brown_hats - ((3/4 : ℝ) * H - (1/5 : ℝ) * brown_hats)
  let remaining_brown_hats := brown_hats - sold_brown_hats
  (remaining_brown_hats / remaining_hats) = (15/100 : ℝ) →
  (sold_brown_hats + ((3/4 : ℝ) * H - (1/5 : ℝ) * brown_hats)) / H = (2/3 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_millionaire_hat_sale_l1048_104823


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1048_104817

theorem arithmetic_calculation : 8 / 4 - 3^2 + 4 * 5 = 13 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1048_104817


namespace NUMINAMATH_CALUDE_distance_at_least_diameter_time_l1048_104845

/-- Represents a circular track -/
structure Track where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a car on a track -/
structure Car where
  track : Track
  clockwise : Bool
  position : ℝ → ℝ × ℝ

/-- The setup of the problem -/
def problem_setup : ℝ × Track × Track × Car × Car := sorry

/-- The time during which the distance between the cars is at least the diameter of each track -/
def time_at_least_diameter (setup : ℝ × Track × Track × Car × Car) : ℝ := sorry

/-- The main theorem stating that the time during which the distance between the cars 
    is at least the diameter of each track is 1/2 hour -/
theorem distance_at_least_diameter_time 
  (setup : ℝ × Track × Track × Car × Car) 
  (h_setup : setup = problem_setup) : 
  time_at_least_diameter setup = 1/2 := by sorry

end NUMINAMATH_CALUDE_distance_at_least_diameter_time_l1048_104845


namespace NUMINAMATH_CALUDE_division_multiplication_result_l1048_104872

theorem division_multiplication_result : (9 / 6) * 12 = 18 := by
  sorry

end NUMINAMATH_CALUDE_division_multiplication_result_l1048_104872
