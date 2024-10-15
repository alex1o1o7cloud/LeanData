import Mathlib

namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3383_338358

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 > 0) ↔ (∃ x : ℝ, x^2 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3383_338358


namespace NUMINAMATH_CALUDE_product_evaluation_l3383_338338

theorem product_evaluation (n : ℕ) (h : n = 3) :
  (n - 2) * (n - 1) * n * (n + 1) * (n + 2) * (n + 3) = 720 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l3383_338338


namespace NUMINAMATH_CALUDE_eight_possible_values_for_d_l3383_338368

def is_digit (n : ℕ) : Prop := n < 10

def distinct_digits (a b c d e : ℕ) : Prop :=
  is_digit a ∧ is_digit b ∧ is_digit c ∧ is_digit d ∧ is_digit e ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

def valid_subtraction (a b c d e : ℕ) : Prop :=
  10000 * a + 1000 * b + 100 * b + 10 * c + b -
  (10000 * b + 1000 * c + 100 * a + 10 * e + a) =
  10000 * d + 1000 * b + 100 * d + 10 * d + d

theorem eight_possible_values_for_d :
  ∃ (s : Finset ℕ), s.card = 8 ∧
  (∀ d, d ∈ s ↔ ∃ (a b c e : ℕ), distinct_digits a b c d e ∧ valid_subtraction a b c d e) :=
sorry

end NUMINAMATH_CALUDE_eight_possible_values_for_d_l3383_338368


namespace NUMINAMATH_CALUDE_cellphone_cost_correct_l3383_338380

/-- The cost of a single cellphone before discount -/
def cellphone_cost : ℝ := 800

/-- The number of cellphones purchased -/
def num_cellphones : ℕ := 2

/-- The discount rate applied to the total cost -/
def discount_rate : ℝ := 0.05

/-- The final price paid after the discount -/
def final_price : ℝ := 1520

/-- Theorem stating that the given cellphone cost satisfies the conditions -/
theorem cellphone_cost_correct : 
  (num_cellphones : ℝ) * cellphone_cost * (1 - discount_rate) = final_price := by
  sorry

end NUMINAMATH_CALUDE_cellphone_cost_correct_l3383_338380


namespace NUMINAMATH_CALUDE_matrix_product_equality_l3383_338335

def A : Matrix (Fin 2) (Fin 2) ℤ := !![3, -4; 6, 2]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![0, 3; -2, 1]

theorem matrix_product_equality :
  A * B = !![8, 5; -4, 20] := by sorry

end NUMINAMATH_CALUDE_matrix_product_equality_l3383_338335


namespace NUMINAMATH_CALUDE_line_intercept_sum_l3383_338310

/-- Given a line mx + 3y - 12 = 0 where m is a real number,
    if the sum of its intercepts on the x and y axes is 7,
    then m = 4. -/
theorem line_intercept_sum (m : ℝ) : 
  (∃ x y : ℝ, m * x + 3 * y - 12 = 0 ∧ 
   (x = 0 ∨ y = 0) ∧
   (∃ x₀ y₀ : ℝ, m * x₀ + 3 * y₀ - 12 = 0 ∧ 
    x₀ = 0 ∧ y₀ = 0 ∧ x + y₀ = 7)) → 
  m = 4 := by
sorry

end NUMINAMATH_CALUDE_line_intercept_sum_l3383_338310


namespace NUMINAMATH_CALUDE_mod_equivalence_2023_l3383_338365

theorem mod_equivalence_2023 : ∃! n : ℕ, n ≤ 11 ∧ n ≡ -2023 [ZMOD 12] ∧ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_2023_l3383_338365


namespace NUMINAMATH_CALUDE_floor_sqrt_eight_count_l3383_338361

theorem floor_sqrt_eight_count : 
  (Finset.filter (fun x : ℕ => ⌊Real.sqrt x⌋ = 8) (Finset.range 81)).card = 17 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_eight_count_l3383_338361


namespace NUMINAMATH_CALUDE_fixed_point_on_graph_l3383_338364

theorem fixed_point_on_graph (k : ℝ) : 
  let f := fun (x : ℝ) => 5 * x^2 + k * x - 3 * k
  f 3 = 45 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_graph_l3383_338364


namespace NUMINAMATH_CALUDE_perfect_square_power_of_two_plus_65_l3383_338392

theorem perfect_square_power_of_two_plus_65 (n : ℕ+) :
  (∃ (x : ℕ), 2^n.val + 65 = x^2) ↔ n.val = 4 ∨ n.val = 10 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_power_of_two_plus_65_l3383_338392


namespace NUMINAMATH_CALUDE_power_calculation_l3383_338315

theorem power_calculation : (8^5 / 8^2) * 2^10 / 2^3 = 65536 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l3383_338315


namespace NUMINAMATH_CALUDE_no_double_application_function_exists_l3383_338381

theorem no_double_application_function_exists :
  ¬ ∃ (f : ℕ → ℕ), ∀ (n : ℕ), f (f n) = n + 1987 := by
sorry

end NUMINAMATH_CALUDE_no_double_application_function_exists_l3383_338381


namespace NUMINAMATH_CALUDE_james_total_cost_l3383_338360

/-- Calculates the total cost of James' vehicle purchases, registrations, and maintenance packages --/
def total_cost : ℕ :=
  let dirt_bike_cost : ℕ := 3 * 150
  let off_road_cost : ℕ := 4 * 300
  let atv_cost : ℕ := 2 * 450
  let moped_cost : ℕ := 5 * 200
  let scooter_cost : ℕ := 3 * 100

  let dirt_bike_reg : ℕ := 3 * 25
  let off_road_reg : ℕ := 4 * 25
  let atv_reg : ℕ := 2 * 30
  let moped_reg : ℕ := 5 * 15
  let scooter_reg : ℕ := 3 * 20

  let dirt_bike_maint : ℕ := 3 * 50
  let off_road_maint : ℕ := 4 * 75
  let atv_maint : ℕ := 2 * 100
  let moped_maint : ℕ := 5 * 60

  dirt_bike_cost + off_road_cost + atv_cost + moped_cost + scooter_cost +
  dirt_bike_reg + off_road_reg + atv_reg + moped_reg + scooter_reg +
  dirt_bike_maint + off_road_maint + atv_maint + moped_maint

theorem james_total_cost : total_cost = 5170 := by
  sorry

end NUMINAMATH_CALUDE_james_total_cost_l3383_338360


namespace NUMINAMATH_CALUDE_adult_ticket_price_l3383_338344

/-- Represents the price of tickets and sales data for a theater --/
structure TheaterSales where
  adult_price : ℚ
  child_price : ℚ
  total_revenue : ℚ
  total_tickets : ℕ
  adult_tickets : ℕ

/-- Theorem stating that the adult ticket price is $10.50 given the conditions --/
theorem adult_ticket_price (sale : TheaterSales)
  (h1 : sale.child_price = 5)
  (h2 : sale.total_revenue = 236)
  (h3 : sale.total_tickets = 34)
  (h4 : sale.adult_tickets = 12)
  : sale.adult_price = 21/2 := by
  sorry

#eval (21 : ℚ) / 2  -- To verify that 21/2 is indeed 10.50

end NUMINAMATH_CALUDE_adult_ticket_price_l3383_338344


namespace NUMINAMATH_CALUDE_b_has_property_P_l3383_338395

-- Define property P for a sequence
def has_property_P (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, (a (n + 1) + a (n + 2)) = q * (a n + a (n + 1))

-- Define the sequence b_n
def b (n : ℕ) : ℝ := 2^n + (-1)^n

-- Theorem statement
theorem b_has_property_P : has_property_P b := by
  sorry

end NUMINAMATH_CALUDE_b_has_property_P_l3383_338395


namespace NUMINAMATH_CALUDE_no_a_in_either_subject_l3383_338336

theorem no_a_in_either_subject (total_students : ℕ) (a_in_chemistry : ℕ) (a_in_physics : ℕ) (a_in_both : ℕ) :
  total_students = 40 →
  a_in_chemistry = 10 →
  a_in_physics = 18 →
  a_in_both = 6 →
  total_students - (a_in_chemistry + a_in_physics - a_in_both) = 18 :=
by sorry

end NUMINAMATH_CALUDE_no_a_in_either_subject_l3383_338336


namespace NUMINAMATH_CALUDE_jacket_discount_percentage_l3383_338384

/-- Proves that the discount percentage is 20% given the specified conditions --/
theorem jacket_discount_percentage (purchase_price selling_price discount_price : ℝ) 
  (h1 : purchase_price = 54)
  (h2 : selling_price = purchase_price + 0.4 * selling_price)
  (h3 : discount_price - purchase_price = 18) : 
  (selling_price - discount_price) / selling_price = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_jacket_discount_percentage_l3383_338384


namespace NUMINAMATH_CALUDE_tammy_second_day_speed_l3383_338305

/-- Represents Tammy's mountain climbing over two days -/
structure MountainClimb where
  total_time : ℝ
  speed_increase : ℝ
  time_decrease : ℝ
  uphill_speed_decrease : ℝ
  downhill_speed_increase : ℝ
  total_distance : ℝ

/-- Calculates Tammy's average speed on the second day -/
def second_day_speed (climb : MountainClimb) : ℝ :=
  -- Definition to be proved
  4

/-- Theorem stating that Tammy's average speed on the second day was 4 km/h -/
theorem tammy_second_day_speed (climb : MountainClimb) 
  (h1 : climb.total_time = 14)
  (h2 : climb.speed_increase = 0.5)
  (h3 : climb.time_decrease = 2)
  (h4 : climb.uphill_speed_decrease = 1)
  (h5 : climb.downhill_speed_increase = 1)
  (h6 : climb.total_distance = 52) :
  second_day_speed climb = 4 := by
  sorry

end NUMINAMATH_CALUDE_tammy_second_day_speed_l3383_338305


namespace NUMINAMATH_CALUDE_subset_condition_1_subset_condition_2_l3383_338391

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x - 8 = 0}

-- Define set B
def B (a : ℝ) : Set ℝ := {x | x^2 + a*x + a^2 - 12 = 0}

-- Theorem for part 1
theorem subset_condition_1 : A ⊆ B a → a = -2 := by sorry

-- Theorem for part 2
theorem subset_condition_2 : B a ⊆ A → a ≥ 4 ∨ a < -4 ∨ a = -2 := by sorry

end NUMINAMATH_CALUDE_subset_condition_1_subset_condition_2_l3383_338391


namespace NUMINAMATH_CALUDE_max_value_x_plus_2y_l3383_338329

theorem max_value_x_plus_2y (x y : ℝ) (h : 3 * (x^2 + y^2) = x + y) :
  x + 2*y ≤ Real.sqrt (5/18) + 1/2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_x_plus_2y_l3383_338329


namespace NUMINAMATH_CALUDE_sum_reciprocal_product_20_l3383_338311

/-- Given a sequence {a_n} where the sum of its first n terms is S_n = 6n - n^2,
    this function returns the sum of the first k terms of the sequence {1/(a_n * a_{n+1})} -/
def sum_reciprocal_product (k : ℕ) : ℚ :=
  let S : ℕ → ℚ := λ n => 6 * n - n^2
  let a : ℕ → ℚ := λ n => S n - S (n-1)
  let term : ℕ → ℚ := λ n => 1 / (a n * a (n+1))
  (Finset.range k).sum term

/-- The main theorem stating that the sum of the first 20 terms of the 
    sequence {1/(a_n * a_{n+1})} is equal to -4/35 -/
theorem sum_reciprocal_product_20 : sum_reciprocal_product 20 = -4/35 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocal_product_20_l3383_338311


namespace NUMINAMATH_CALUDE_ratio_as_percent_l3383_338314

theorem ratio_as_percent (first_part second_part : ℕ) (h1 : first_part = 25) (h2 : second_part = 50) :
  ∃ (p : ℚ), abs (p - 100 * (first_part : ℚ) / (first_part + second_part)) < 0.01 ∧ p = 33.33 := by
  sorry

end NUMINAMATH_CALUDE_ratio_as_percent_l3383_338314


namespace NUMINAMATH_CALUDE_probability_proof_l3383_338366

def total_balls : ℕ := 6
def white_balls : ℕ := 3
def black_balls : ℕ := 3
def drawn_balls : ℕ := 2

def probability_at_most_one_black : ℚ := 4/5

theorem probability_proof :
  (Nat.choose total_balls drawn_balls - Nat.choose black_balls drawn_balls) / Nat.choose total_balls drawn_balls = probability_at_most_one_black :=
sorry

end NUMINAMATH_CALUDE_probability_proof_l3383_338366


namespace NUMINAMATH_CALUDE_cricketer_average_score_l3383_338307

theorem cricketer_average_score (total_matches : ℕ) (first_matches : ℕ) (last_matches : ℕ)
  (first_avg : ℚ) (last_avg : ℚ) :
  total_matches = first_matches + last_matches →
  total_matches = 12 →
  first_matches = 8 →
  last_matches = 4 →
  first_avg = 40 →
  last_avg = 64 →
  (first_avg * first_matches + last_avg * last_matches) / total_matches = 48 := by
  sorry

#check cricketer_average_score

end NUMINAMATH_CALUDE_cricketer_average_score_l3383_338307


namespace NUMINAMATH_CALUDE_negation_of_all_children_good_l3383_338317

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Child : U → Prop)
variable (GoodAtMusic : U → Prop)

-- Define the original statement and its negation
def AllChildrenGood : Prop := ∀ x, Child x → GoodAtMusic x
def AllChildrenPoor : Prop := ∀ x, Child x → ¬GoodAtMusic x

-- Theorem statement
theorem negation_of_all_children_good :
  AllChildrenPoor U Child GoodAtMusic ↔ ¬AllChildrenGood U Child GoodAtMusic :=
sorry

end NUMINAMATH_CALUDE_negation_of_all_children_good_l3383_338317


namespace NUMINAMATH_CALUDE_geometric_sequence_increasing_iff_l3383_338394

/-- A geometric sequence with positive first term -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q ∧ a 1 > 0

/-- An increasing sequence -/
def IncreasingSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) > a n

theorem geometric_sequence_increasing_iff (a : ℕ → ℝ) :
  GeometricSequence a → (a 2 > a 1 ↔ IncreasingSequence a) := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_increasing_iff_l3383_338394


namespace NUMINAMATH_CALUDE_distribution_ratio_l3383_338321

/-- Represents the distribution of money among four people --/
structure Distribution where
  p : ℚ  -- Amount received by P
  q : ℚ  -- Amount received by Q
  r : ℚ  -- Amount received by R
  s : ℚ  -- Amount received by S

/-- Theorem stating the ratio of P's amount to Q's amount --/
theorem distribution_ratio (d : Distribution) : 
  d.p + d.q + d.r + d.s = 1000 →  -- Total amount condition
  d.s = 4 * d.r →                 -- S gets 4 times R's amount
  d.q = d.r →                     -- Q and R receive equal amounts
  d.s - d.p = 250 →               -- Difference between S and P
  d.p / d.q = 2 / 1 := by          -- Ratio of P's amount to Q's amount
sorry


end NUMINAMATH_CALUDE_distribution_ratio_l3383_338321


namespace NUMINAMATH_CALUDE_probability_of_snow_l3383_338345

/-- The probability of snow on at least one day out of four, given specific conditions --/
theorem probability_of_snow (p : ℝ) (q : ℝ) : 
  p = 3/4 →  -- probability of snow on each of the first three days
  q = 4/5 →  -- probability of snow on the last day if it snowed before
  (1 - (1 - p)^3 * (1 - p) - (1 - (1 - p)^3) * (1 - q)) = 1023/1280 :=
by sorry

end NUMINAMATH_CALUDE_probability_of_snow_l3383_338345


namespace NUMINAMATH_CALUDE_largest_consecutive_sum_of_3_12_l3383_338342

theorem largest_consecutive_sum_of_3_12 :
  (∃ (k : ℕ), k > 486 ∧ 
    (∃ (n : ℕ), 3^12 = (Finset.range k).sum (λ i => n + i + 1))) →
  False :=
sorry

end NUMINAMATH_CALUDE_largest_consecutive_sum_of_3_12_l3383_338342


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3383_338356

/-- A quadratic function with real coefficients -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^2 + a*x + b

/-- The theorem stating the value of c given the conditions -/
theorem quadratic_inequality_solution (a b m : ℝ) :
  (∀ x, f a b x ≥ 0) →
  (∃ c, ∀ x, f a b x < c ↔ m < x ∧ x < m + 6) →
  ∃ c, (∀ x, f a b x < c ↔ m < x ∧ x < m + 6) ∧ c = 9 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3383_338356


namespace NUMINAMATH_CALUDE_solution_set_when_t_3_non_negative_for_all_x_iff_t_1_l3383_338373

-- Define the function f
def f (t x : ℝ) : ℝ := x^2 - (t+1)*x + t

-- Theorem for part 1
theorem solution_set_when_t_3 :
  {x : ℝ | f 3 x > 0} = {x : ℝ | x < 1 ∨ x > 3} := by sorry

-- Theorem for part 2
theorem non_negative_for_all_x_iff_t_1 :
  (∀ x : ℝ, f t x ≥ 0) ↔ t = 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_t_3_non_negative_for_all_x_iff_t_1_l3383_338373


namespace NUMINAMATH_CALUDE_identity_proof_l3383_338367

-- Define the necessary functions and series
def infiniteProduct (f : ℕ → ℝ) : ℝ := sorry

def infiniteSum (f : ℤ → ℝ) : ℝ := sorry

-- State the theorem
theorem identity_proof (x : ℝ) (h : |x| < 1) :
  -- First identity
  (infiniteProduct (λ m => (1 - x^(2*m - 1))^2)) =
  (1 / infiniteProduct (λ m => (1 - x^(2*m)))) *
  (infiniteSum (λ k => (-1)^k * x^(k^2)))
  ∧
  -- Second identity
  (infiniteProduct (λ m => (1 - x^m))) =
  (infiniteProduct (λ m => (1 + x^m))) *
  (infiniteSum (λ k => (-1)^k * x^(k^2))) :=
by sorry

end NUMINAMATH_CALUDE_identity_proof_l3383_338367


namespace NUMINAMATH_CALUDE_kim_laura_difference_l3383_338369

/-- Proves that Kim paints 3 fewer tiles per minute than Laura -/
theorem kim_laura_difference (don ken laura kim : ℕ) : 
  don = 3 →  -- Don paints 3 tiles per minute
  ken = don + 2 →  -- Ken paints 2 more tiles than Don per minute
  laura = 2 * ken →  -- Laura paints twice as many tiles as Ken per minute
  don + ken + laura + kim = 25 →  -- They paint 375 tiles in 15 minutes (375 / 15 = 25)
  laura - kim = 3 := by  -- Kim paints 3 fewer tiles than Laura per minute
sorry

end NUMINAMATH_CALUDE_kim_laura_difference_l3383_338369


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3383_338324

-- Define the sets A and B
def A : Set ℝ := {1, 2, 1/2}
def B : Set ℝ := {y | ∃ x ∈ A, y = x^2}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3383_338324


namespace NUMINAMATH_CALUDE_skittles_taken_away_l3383_338383

def initial_skittles : ℕ := 25
def remaining_skittles : ℕ := 18

theorem skittles_taken_away : initial_skittles - remaining_skittles = 7 := by
  sorry

end NUMINAMATH_CALUDE_skittles_taken_away_l3383_338383


namespace NUMINAMATH_CALUDE_tangent_line_at_one_monotonicity_intervals_existence_of_positive_value_l3383_338343

-- Define the function f(x) = ln x + ax
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * x

-- Theorem for the tangent line equation
theorem tangent_line_at_one (a : ℝ) :
  a = 1 → ∃ (m b : ℝ), ∀ x y, y = f 1 x → (2 : ℝ) * x - y - 1 = 0 := by sorry

-- Theorem for monotonicity intervals
theorem monotonicity_intervals (a : ℝ) :
  (a ≥ 0 → ∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂) ∧
  (a < 0 → (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < -1/a → f a x₁ < f a x₂) ∧
           (∀ x₁ x₂, -1/a < x₁ ∧ x₁ < x₂ → f a x₁ > f a x₂)) := by sorry

-- Theorem for the range of a where f(x₀) > 0 exists
theorem existence_of_positive_value (a : ℝ) :
  (∃ x₀, 0 < x₀ ∧ f a x₀ > 0) ↔ a > -1 / Real.exp 1 := by sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_monotonicity_intervals_existence_of_positive_value_l3383_338343


namespace NUMINAMATH_CALUDE_f_max_value_l3383_338386

-- Define the function f(x) = x³ + 3x² - 4
def f (x : ℝ) : ℝ := x^3 + 3*x^2 - 4

-- State the theorem about the maximum value of f
theorem f_max_value :
  ∃ (M : ℝ), M = 0 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_f_max_value_l3383_338386


namespace NUMINAMATH_CALUDE_compare_squares_and_products_l3383_338320

theorem compare_squares_and_products 
  (x a b : ℝ) 
  (h1 : x < a) 
  (h2 : a < b) 
  (h3 : b < 0) : 
  x^2 > a * x ∧ 
  a * x > b * x ∧ 
  x^2 > a^2 ∧ 
  a^2 > b^2 := by
sorry

end NUMINAMATH_CALUDE_compare_squares_and_products_l3383_338320


namespace NUMINAMATH_CALUDE_no_four_distinct_numbers_l3383_338319

theorem no_four_distinct_numbers : 
  ¬ ∃ (a b c d : ℝ), (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧ 
    (a^11 - a = b^11 - b) ∧ 
    (a^11 - a = c^11 - c) ∧ 
    (a^11 - a = d^11 - d) := by
  sorry

end NUMINAMATH_CALUDE_no_four_distinct_numbers_l3383_338319


namespace NUMINAMATH_CALUDE_whoosit_count_2_l3383_338347

def worker_count_1 : ℕ := 150
def widget_count_1 : ℕ := 450
def whoosit_count_1 : ℕ := 300
def hours_1 : ℕ := 1

def worker_count_2 : ℕ := 90
def widget_count_2 : ℕ := 540
def hours_2 : ℕ := 3

def worker_count_3 : ℕ := 75
def widget_count_3 : ℕ := 300
def whoosit_count_3 : ℕ := 400
def hours_3 : ℕ := 4

def widget_production_rate_1 : ℚ := widget_count_1 / (worker_count_1 * hours_1)
def whoosit_production_rate_1 : ℚ := whoosit_count_1 / (worker_count_1 * hours_1)

def widget_production_rate_3 : ℚ := widget_count_3 / (worker_count_3 * hours_3)
def whoosit_production_rate_3 : ℚ := whoosit_count_3 / (worker_count_3 * hours_3)

theorem whoosit_count_2 (h : 2 * whoosit_production_rate_3 = widget_production_rate_3) :
  ∃ n : ℕ, n = 360 ∧ n / (worker_count_2 * hours_2) = whoosit_production_rate_3 :=
by sorry

end NUMINAMATH_CALUDE_whoosit_count_2_l3383_338347


namespace NUMINAMATH_CALUDE_unwatered_rosebushes_l3383_338350

/-- The number of unwatered rosebushes in Anna and Vitya's garden -/
theorem unwatered_rosebushes 
  (total : ℕ) 
  (vitya_watered : ℕ) 
  (anna_watered : ℕ) 
  (both_watered : ℕ)
  (h1 : total = 2006)
  (h2 : vitya_watered = total / 2)
  (h3 : anna_watered = total / 2)
  (h4 : both_watered = 3) :
  total - (vitya_watered + anna_watered - both_watered) = 3 :=
by sorry

end NUMINAMATH_CALUDE_unwatered_rosebushes_l3383_338350


namespace NUMINAMATH_CALUDE_min_value_of_function_equality_condition_l3383_338382

theorem min_value_of_function (x : ℝ) (h : x > 0) : (x^2 + 1) / x ≥ 2 :=
  sorry

theorem equality_condition (x : ℝ) (h : x > 0) : (x^2 + 1) / x = 2 ↔ x = 1 :=
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_equality_condition_l3383_338382


namespace NUMINAMATH_CALUDE_floor_product_equation_l3383_338398

theorem floor_product_equation : ∃! (x : ℝ), x > 0 ∧ (⌊x⌋ : ℝ) * x = 70 ∧ x = (70 : ℝ) / 8 := by sorry

end NUMINAMATH_CALUDE_floor_product_equation_l3383_338398


namespace NUMINAMATH_CALUDE_rectangle_area_with_inscribed_circle_l3383_338396

/-- Given a rectangle with an inscribed circle of radius 6 and a length-to-width ratio of 3:1,
    prove that the area of the rectangle is 432. -/
theorem rectangle_area_with_inscribed_circle (r : ℝ) (ratio : ℝ) :
  r = 6 →
  ratio = 3 →
  let width := 2 * r
  let length := ratio * width
  width * length = 432 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_with_inscribed_circle_l3383_338396


namespace NUMINAMATH_CALUDE_special_subset_contains_all_rationals_l3383_338312

def is_special_subset (S : Set ℚ) : Prop :=
  (1/2 ∈ S) ∧ 
  (∀ x ∈ S, x/2 ∈ S) ∧ 
  (∀ x ∈ S, 1/(x+1) ∈ S)

theorem special_subset_contains_all_rationals (S : Set ℚ) 
  (h : is_special_subset S) :
  ∀ q ∈ Set.Ioo (0 : ℚ) 1, q ∈ S :=
by
  sorry

end NUMINAMATH_CALUDE_special_subset_contains_all_rationals_l3383_338312


namespace NUMINAMATH_CALUDE_penguin_colony_size_l3383_338306

/-- Represents the number of penguins in a colony over time -/
structure PenguinColony where
  initial : ℕ
  current : ℕ

/-- Calculates the current number of penguins based on initial conditions -/
def calculate_current_penguins (initial : ℕ) : ℕ :=
  6 * initial + 129

/-- Theorem stating the current number of penguins in the colony -/
theorem penguin_colony_size (colony : PenguinColony) : 
  colony.initial * 3/2 = 237 → colony.current = 1077 := by
  sorry

#check penguin_colony_size

end NUMINAMATH_CALUDE_penguin_colony_size_l3383_338306


namespace NUMINAMATH_CALUDE_probability_sum_16_three_dice_rolls_l3383_338372

theorem probability_sum_16_three_dice_rolls :
  let die_faces : ℕ := 6
  let total_outcomes : ℕ := die_faces ^ 3
  let favorable_outcomes : ℕ := 6
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 36 :=
by sorry

end NUMINAMATH_CALUDE_probability_sum_16_three_dice_rolls_l3383_338372


namespace NUMINAMATH_CALUDE_fifth_root_unity_product_l3383_338318

theorem fifth_root_unity_product (r : ℂ) (h1 : r^5 = 1) (h2 : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) = 5 := by
  sorry

end NUMINAMATH_CALUDE_fifth_root_unity_product_l3383_338318


namespace NUMINAMATH_CALUDE_inequality_proof_l3383_338370

theorem inequality_proof (n : ℕ) (hn : n > 1) : 
  let a : ℚ := 1 / n
  (a^2 : ℚ) < a ∧ a < (1 : ℚ) / a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3383_338370


namespace NUMINAMATH_CALUDE_parallelogram_to_rhombus_l3383_338330

structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

def is_convex (Q : Quadrilateral) : Prop := sorry

def is_parallelogram (Q : Quadrilateral) : Prop := sorry

def is_rhombus (Q : Quadrilateral) : Prop := sorry

def is_similar_not_congruent (Q1 Q2 : Quadrilateral) : Prop := sorry

def perpendicular_move (Q : Quadrilateral) : Quadrilateral := sorry

theorem parallelogram_to_rhombus (P : Quadrilateral) 
  (h_convex : is_convex P) 
  (h_initial : is_parallelogram P) 
  (h_final : ∃ (P_final : Quadrilateral), 
    (∃ (n : ℕ), n > 0 ∧ P_final = (perpendicular_move^[n] P)) ∧ 
    is_similar_not_congruent P P_final) :
  is_rhombus P := by sorry

end NUMINAMATH_CALUDE_parallelogram_to_rhombus_l3383_338330


namespace NUMINAMATH_CALUDE_polygon_area_theorem_l3383_338387

/-- The area of a polygon with given vertices -/
def polygonArea (vertices : List (ℤ × ℤ)) : ℚ :=
  sorry

/-- The number of integer points strictly inside a polygon -/
def interiorPoints (vertices : List (ℤ × ℤ)) : ℕ :=
  sorry

/-- The number of integer points on the boundary of a polygon -/
def boundaryPoints (vertices : List (ℤ × ℤ)) : ℕ :=
  sorry

theorem polygon_area_theorem :
  let vertices : List (ℤ × ℤ) := [(0, 1), (1, 2), (3, 2), (4, 1), (2, 0)]
  polygonArea vertices = 15/2 ∧
  interiorPoints vertices = 6 ∧
  boundaryPoints vertices = 5 :=
by sorry

end NUMINAMATH_CALUDE_polygon_area_theorem_l3383_338387


namespace NUMINAMATH_CALUDE_no_real_roots_implies_no_real_roots_composition_l3383_338385

/-- A quadratic function f(x) = ax^2 + bx + c -/
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

/-- Theorem: If f(x) = x has no real roots, then f(f(x)) = x has no real roots -/
theorem no_real_roots_implies_no_real_roots_composition
  (a b c : ℝ) :
  (∀ x : ℝ, f a b c x ≠ x) →
  (∀ x : ℝ, f a b c (f a b c x) ≠ x) :=
by sorry

end NUMINAMATH_CALUDE_no_real_roots_implies_no_real_roots_composition_l3383_338385


namespace NUMINAMATH_CALUDE_unique_prime_sum_30_l3383_338376

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

theorem unique_prime_sum_30 :
  ∃! (A B C : ℕ), 
    isPrime A ∧ isPrime B ∧ isPrime C ∧
    A < 20 ∧ B < 20 ∧ C < 20 ∧
    A + B + C = 30 ∧
    A = 2 ∧ B = 11 ∧ C = 17 :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_sum_30_l3383_338376


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_cube_sum_l3383_338389

theorem smallest_prime_divisor_cube_sum (n : ℕ) : n ≥ 2 → (∃ (a d : ℕ), Prime a ∧ d > 0 ∧ d ∣ n ∧ (∀ p : ℕ, Prime p → p ∣ n → a ≤ p) ∧ n = a^3 + d^3) → (n = 16 ∨ n = 72 ∨ n = 520) :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_cube_sum_l3383_338389


namespace NUMINAMATH_CALUDE_molly_has_three_brothers_l3383_338353

/-- Represents the problem of determining Molly's number of brothers --/
def MollysBrothers (cost_per_package : ℕ) (num_parents : ℕ) (total_cost : ℕ) : Prop :=
  ∃ (num_brothers : ℕ),
    cost_per_package * (num_parents + num_brothers + num_brothers + 2 * num_brothers) = total_cost

/-- Theorem stating that Molly has 3 brothers given the problem conditions --/
theorem molly_has_three_brothers :
  MollysBrothers 5 2 70 → ∃ (num_brothers : ℕ), num_brothers = 3 := by
  sorry

end NUMINAMATH_CALUDE_molly_has_three_brothers_l3383_338353


namespace NUMINAMATH_CALUDE_rectangle_perimeter_bound_l3383_338328

/-- The curve W defined by y = x^2 + 1/4 -/
def W : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1^2 + 1/4}

/-- A rectangle with vertices as points in ℝ × ℝ -/
structure Rectangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- Perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ :=
  2 * (dist r.A r.B + dist r.B r.C)

/-- Three vertices of the rectangle are on W -/
def three_vertices_on_W (r : Rectangle) : Prop :=
  (r.A ∈ W ∧ r.B ∈ W ∧ r.C ∈ W) ∨
  (r.A ∈ W ∧ r.B ∈ W ∧ r.D ∈ W) ∨
  (r.A ∈ W ∧ r.C ∈ W ∧ r.D ∈ W) ∨
  (r.B ∈ W ∧ r.C ∈ W ∧ r.D ∈ W)

theorem rectangle_perimeter_bound (r : Rectangle) 
  (h : three_vertices_on_W r) : 
  perimeter r > 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_bound_l3383_338328


namespace NUMINAMATH_CALUDE_min_expression_l3383_338322

theorem min_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (x * y / 2 + 18 / (x * y) ≥ 6) ∧ 
  ((x * y / 2 + 18 / (x * y) = 6) → (y / 2 + x / 3 ≥ 2)) ∧
  ((x * y / 2 + 18 / (x * y) = 6) ∧ (y / 2 + x / 3 = 2) → (x = 3 ∧ y = 2)) := by
sorry

end NUMINAMATH_CALUDE_min_expression_l3383_338322


namespace NUMINAMATH_CALUDE_quadratic_root_range_l3383_338390

theorem quadratic_root_range (t : ℝ) :
  (∃ α β : ℝ, (3*t*α^2 + (3-7*t)*α + 2 = 0) ∧
              (3*t*β^2 + (3-7*t)*β + 2 = 0) ∧
              (0 < α) ∧ (α < 1) ∧ (1 < β) ∧ (β < 2)) →
  (5/4 < t) ∧ (t < 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l3383_338390


namespace NUMINAMATH_CALUDE_team_B_is_better_l3383_338309

/-- Represents the expected cost of drug development for Team A -/
def expected_cost_A (p : ℝ) (m : ℝ) : ℝ :=
  -2 * m * p^2 + 6 * m

/-- Represents the expected cost of drug development for Team B -/
def expected_cost_B (q : ℝ) (n : ℝ) : ℝ :=
  6 * n * q^3 - 9 * n * q^2 + 6 * n

/-- Theorem stating that Team B's expected cost is less than Team A's when n = 2/3m and p = q -/
theorem team_B_is_better (p q m n : ℝ) 
  (h1 : 0 < p ∧ p < 1) 
  (h2 : m > 0) 
  (h3 : n = 2/3 * m) 
  (h4 : p = q) : 
  expected_cost_B q n < expected_cost_A p m :=
sorry

end NUMINAMATH_CALUDE_team_B_is_better_l3383_338309


namespace NUMINAMATH_CALUDE_total_points_is_238_l3383_338326

/-- Represents a player's statistics in the basketball game -/
structure PlayerStats :=
  (two_pointers : ℕ)
  (three_pointers : ℕ)
  (free_throws : ℕ)
  (steals : ℕ)
  (rebounds : ℕ)
  (fouls : ℕ)

/-- Calculates the total points for a player given their stats -/
def calculate_points (stats : PlayerStats) : ℤ :=
  2 * stats.two_pointers + 3 * stats.three_pointers + stats.free_throws +
  stats.steals + 2 * stats.rebounds - 5 * stats.fouls

/-- The main theorem to prove -/
theorem total_points_is_238 :
  let sam := PlayerStats.mk 20 10 5 4 6 2
  let alex := PlayerStats.mk 15 8 5 6 3 3
  let jake := PlayerStats.mk 10 6 3 7 5 4
  let lily := PlayerStats.mk 16 4 7 3 7 1
  calculate_points sam + calculate_points alex + calculate_points jake + calculate_points lily = 238 := by
  sorry

end NUMINAMATH_CALUDE_total_points_is_238_l3383_338326


namespace NUMINAMATH_CALUDE_min_value_expression_l3383_338327

theorem min_value_expression (x y : ℝ) (hx : x > 1) (hy : y > 1) :
  (x^3 / (y - 1)) + (y^3 / (x - 1)) ≥ 16 ∧
  ∃ x y, x > 1 ∧ y > 1 ∧ (x^3 / (y - 1)) + (y^3 / (x - 1)) = 16 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3383_338327


namespace NUMINAMATH_CALUDE_exactly_one_black_ball_remains_l3383_338331

/-- Represents the color of a ball -/
inductive Color
| Black
| Gray
| White

/-- Represents the state of the box -/
structure BoxState :=
  (black : Nat)
  (gray : Nat)
  (white : Nat)

/-- Simulates drawing two balls from the box -/
def drawTwoBalls (state : BoxState) : BoxState :=
  sorry

/-- Checks if the given state has exactly two balls remaining -/
def hasTwoballsRemaining (state : BoxState) : Bool :=
  state.black + state.gray + state.white = 2

/-- Represents the final state of the box after the procedure -/
def finalState (initialState : BoxState) : BoxState :=
  sorry

/-- The main theorem to be proved -/
theorem exactly_one_black_ball_remains :
  let initialState : BoxState := ⟨105, 89, 5⟩
  let finalState := finalState initialState
  hasTwoballsRemaining finalState ∧ finalState.black = 1 :=
by sorry

end NUMINAMATH_CALUDE_exactly_one_black_ball_remains_l3383_338331


namespace NUMINAMATH_CALUDE_jakes_weight_l3383_338341

theorem jakes_weight (jake_weight sister_weight : ℕ) : 
  jake_weight - 33 = 2 * sister_weight →
  jake_weight + sister_weight = 153 →
  jake_weight = 113 := by
sorry

end NUMINAMATH_CALUDE_jakes_weight_l3383_338341


namespace NUMINAMATH_CALUDE_trailing_zeros_of_500_power_150_l3383_338397

-- Define 500 as 5 * 10^2
def five_hundred : ℕ := 5 * 10^2

-- Define the exponent
def exponent : ℕ := 150

-- Define the function to count trailing zeros
def trailing_zeros (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem trailing_zeros_of_500_power_150 :
  trailing_zeros (five_hundred ^ exponent) = 300 := by sorry

end NUMINAMATH_CALUDE_trailing_zeros_of_500_power_150_l3383_338397


namespace NUMINAMATH_CALUDE_sum_within_range_l3383_338399

/-- Converts a decimal number to its representation in a given base -/
def toBase (n : ℕ) (base : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in a given base to its decimal value -/
def fromBase (digits : List ℕ) (base : ℕ) : ℕ :=
  sorry

/-- Checks if a number is within the valid range -/
def isValidNumber (n : ℕ) : Prop :=
  n ≥ 3577 ∧ n ≤ 3583

/-- Calculates the sum of base conversions -/
def sumOfBaseConversions (n : ℕ) : ℕ :=
  fromBase (toBase n 7) 10 + fromBase (toBase n 8) 10 + fromBase (toBase n 9) 10

/-- Theorem: The sum of base conversions for valid numbers is within 0.5% of 25,000 -/
theorem sum_within_range (n : ℕ) (h : isValidNumber n) :
  (sumOfBaseConversions n : ℝ) > 24875 ∧ (sumOfBaseConversions n : ℝ) < 25125 :=
  sorry

end NUMINAMATH_CALUDE_sum_within_range_l3383_338399


namespace NUMINAMATH_CALUDE_smallest_multiple_l3383_338346

theorem smallest_multiple (n : ℕ) : n = 459 ↔ 
  (∃ k : ℕ, n = 17 * k) ∧ 
  (∃ m : ℕ, n = 76 * m + 3) ∧ 
  (∀ x : ℕ, x < n → ¬(∃ k : ℕ, x = 17 * k) ∨ ¬(∃ m : ℕ, x = 76 * m + 3)) := by
sorry

end NUMINAMATH_CALUDE_smallest_multiple_l3383_338346


namespace NUMINAMATH_CALUDE_lenas_collage_friends_l3383_338308

/-- Given the conditions of Lena's collage project, prove the number of friends' pictures glued. -/
theorem lenas_collage_friends (clippings_per_friend : ℕ) (glue_per_clipping : ℕ) (total_glue : ℕ) 
  (h1 : clippings_per_friend = 3)
  (h2 : glue_per_clipping = 6)
  (h3 : total_glue = 126) :
  total_glue / (clippings_per_friend * glue_per_clipping) = 7 := by
  sorry

end NUMINAMATH_CALUDE_lenas_collage_friends_l3383_338308


namespace NUMINAMATH_CALUDE_parabola_translation_theorem_l3383_338355

/-- Represents a parabola in the form y = ax² + bx + c --/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola vertically --/
def translate_vertical (p : Parabola) (d : ℝ) : Parabola :=
  { a := p.a, b := p.b, c := p.c - d }

/-- Translates a parabola horizontally --/
def translate_horizontal (p : Parabola) (d : ℝ) : Parabola :=
  { a := p.a, b := -2 * p.a * d + p.b, c := p.a * d^2 - p.b * d + p.c }

theorem parabola_translation_theorem :
  let original := Parabola.mk 3 0 0
  let down_3 := translate_vertical original 3
  let right_2 := translate_horizontal down_3 2
  right_2 = Parabola.mk 3 (-12) 9 := by sorry

end NUMINAMATH_CALUDE_parabola_translation_theorem_l3383_338355


namespace NUMINAMATH_CALUDE_bank_withdrawal_bill_value_l3383_338333

theorem bank_withdrawal_bill_value (x n : ℕ) (h1 : x = 300) (h2 : n = 30) :
  (2 * x) / n = 20 := by
  sorry

end NUMINAMATH_CALUDE_bank_withdrawal_bill_value_l3383_338333


namespace NUMINAMATH_CALUDE_runs_by_running_percentage_l3383_338374

def total_runs : ℕ := 120
def boundaries : ℕ := 6
def sixes : ℕ := 4
def runs_per_boundary : ℕ := 4
def runs_per_six : ℕ := 6

theorem runs_by_running_percentage :
  let runs_from_boundaries := boundaries * runs_per_boundary
  let runs_from_sixes := sixes * runs_per_six
  let runs_without_running := runs_from_boundaries + runs_from_sixes
  let runs_by_running := total_runs - runs_without_running
  (runs_by_running : ℚ) / total_runs * 100 = 60 := by sorry

end NUMINAMATH_CALUDE_runs_by_running_percentage_l3383_338374


namespace NUMINAMATH_CALUDE_removed_carrots_average_weight_l3383_338379

/-- Proves that the average weight of 4 removed carrots is 190 grams -/
theorem removed_carrots_average_weight
  (total_weight : ℝ)
  (remaining_carrots : ℕ)
  (removed_carrots : ℕ)
  (remaining_average : ℝ)
  (h1 : total_weight = 3.64)
  (h2 : remaining_carrots = 16)
  (h3 : removed_carrots = 4)
  (h4 : remaining_average = 180)
  (h5 : remaining_carrots + removed_carrots = 20) :
  (total_weight * 1000 - remaining_carrots * remaining_average) / removed_carrots = 190 :=
by sorry

end NUMINAMATH_CALUDE_removed_carrots_average_weight_l3383_338379


namespace NUMINAMATH_CALUDE_factorial_fraction_l3383_338354

theorem factorial_fraction (N : ℕ) :
  (Nat.factorial (N + 2)) / (Nat.factorial (N + 3) - Nat.factorial (N + 2)) = 1 / (N + 2) := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_l3383_338354


namespace NUMINAMATH_CALUDE_system_solution_proof_single_equation_solution_proof_l3383_338323

-- System of equations
theorem system_solution_proof (x y : ℝ) : 
  x = 1 ∧ y = 2 → 2*x + 3*y = 8 ∧ 3*x - 5*y = -7 := by sorry

-- Single equation
theorem single_equation_solution_proof (x : ℝ) :
  x = -1 → (x-2)/(x+2) - 12/(x^2-4) = 1 := by sorry

end NUMINAMATH_CALUDE_system_solution_proof_single_equation_solution_proof_l3383_338323


namespace NUMINAMATH_CALUDE_square_remainders_l3383_338301

theorem square_remainders (n : ℤ) : 
  (∃ r : ℤ, r ∈ ({0, 1} : Set ℤ) ∧ n^2 ≡ r [ZMOD 3]) ∧
  (∃ r : ℤ, r ∈ ({0, 1} : Set ℤ) ∧ n^2 ≡ r [ZMOD 4]) ∧
  (∃ r : ℤ, r ∈ ({0, 1, 4} : Set ℤ) ∧ n^2 ≡ r [ZMOD 5]) ∧
  (∃ r : ℤ, r ∈ ({0, 1, 4} : Set ℤ) ∧ n^2 ≡ r [ZMOD 8]) :=
by sorry

end NUMINAMATH_CALUDE_square_remainders_l3383_338301


namespace NUMINAMATH_CALUDE_smallest_k_for_triangular_l3383_338300

/-- A positive integer T is triangular if there exists a positive integer n such that T = n * (n + 1) / 2 -/
def IsTriangular (T : ℕ) : Prop :=
  ∃ n : ℕ, n > 0 ∧ T = n * (n + 1) / 2

/-- The smallest positive integer k such that for any triangular number T, 81T + k is also triangular -/
theorem smallest_k_for_triangular : ∃! k : ℕ, 
  k > 0 ∧ 
  (∀ T : ℕ, IsTriangular T → IsTriangular (81 * T + k)) ∧
  (∀ k' : ℕ, k' > 0 → k' < k → 
    ∃ T : ℕ, IsTriangular T ∧ ¬IsTriangular (81 * T + k')) :=
sorry

end NUMINAMATH_CALUDE_smallest_k_for_triangular_l3383_338300


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3383_338393

theorem complex_fraction_simplification (x y : ℚ) (hx : x = 3) (hy : y = 4) :
  (1 / y) / (1 / x) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3383_338393


namespace NUMINAMATH_CALUDE_d_eq_l_l3383_338357

/-- The number of partitions of n into distinct summands -/
def d (n : ℕ) : ℕ := sorry

/-- The number of partitions of n into odd summands -/
def l (n : ℕ) : ℕ := sorry

/-- The generating function for d(n) -/
noncomputable def d_gen_fun (x : ℝ) : ℝ := ∑' n, d n * x^n

/-- The generating function for l(n) -/
noncomputable def l_gen_fun (x : ℝ) : ℝ := ∑' n, l n * x^n

/-- The product representation of d_gen_fun -/
noncomputable def d_prod (x : ℝ) : ℝ := ∏' k, (1 + x^k)

/-- The product representation of l_gen_fun -/
noncomputable def l_prod (x : ℝ) : ℝ := ∏' k, (1 - x^(2*k+1))⁻¹

/-- The main theorem: d(n) = l(n) for all n -/
theorem d_eq_l : ∀ n : ℕ, d n = l n := by sorry

/-- d(0) = l(0) = 1 -/
axiom d_zero : d 0 = 1
axiom l_zero : l 0 = 1

/-- The generating functions are equal to their product representations -/
axiom d_gen_fun_eq_prod : d_gen_fun = d_prod
axiom l_gen_fun_eq_prod : l_gen_fun = l_prod

end NUMINAMATH_CALUDE_d_eq_l_l3383_338357


namespace NUMINAMATH_CALUDE_square_of_negative_sqrt_two_equals_two_l3383_338371

theorem square_of_negative_sqrt_two_equals_two :
  ((-Real.sqrt 2) ^ 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_sqrt_two_equals_two_l3383_338371


namespace NUMINAMATH_CALUDE_hawkeye_battery_charge_cost_l3383_338316

theorem hawkeye_battery_charge_cost 
  (budget : ℝ) 
  (num_charges : ℕ) 
  (remaining : ℝ) 
  (h1 : budget = 20)
  (h2 : num_charges = 4)
  (h3 : remaining = 6) : 
  (budget - remaining) / num_charges = 3.50 := by
  sorry

end NUMINAMATH_CALUDE_hawkeye_battery_charge_cost_l3383_338316


namespace NUMINAMATH_CALUDE_white_spotted_mushrooms_count_l3383_338349

/-- The number of red mushrooms Bill gathered -/
def red_mushrooms : ℕ := 12

/-- The number of brown mushrooms Bill gathered -/
def brown_mushrooms : ℕ := 6

/-- The number of green mushrooms Ted gathered -/
def green_mushrooms : ℕ := 14

/-- The number of blue mushrooms Ted gathered -/
def blue_mushrooms : ℕ := 6

/-- The fraction of blue mushrooms with white spots -/
def blue_spotted_fraction : ℚ := 1/2

/-- The fraction of red mushrooms with white spots -/
def red_spotted_fraction : ℚ := 2/3

/-- The fraction of brown mushrooms with white spots -/
def brown_spotted_fraction : ℚ := 1

theorem white_spotted_mushrooms_count : 
  ⌊blue_spotted_fraction * blue_mushrooms⌋ + 
  ⌊red_spotted_fraction * red_mushrooms⌋ + 
  ⌊brown_spotted_fraction * brown_mushrooms⌋ = 17 := by
  sorry

end NUMINAMATH_CALUDE_white_spotted_mushrooms_count_l3383_338349


namespace NUMINAMATH_CALUDE_car_sale_profit_l3383_338377

theorem car_sale_profit (P : ℝ) (h : P > 0) :
  let discount_rate := 0.2
  let profit_rate := 0.28000000000000004
  let purchase_price := P * (1 - discount_rate)
  let selling_price := P * (1 + profit_rate)
  let increase_rate := (selling_price - purchase_price) / purchase_price
  increase_rate = 0.6 := by sorry

end NUMINAMATH_CALUDE_car_sale_profit_l3383_338377


namespace NUMINAMATH_CALUDE_clock_strike_time_l3383_338340

/-- If a clock takes 42 seconds to strike 7 times, it takes 60 seconds to strike 10 times. -/
theorem clock_strike_time (strike_time : ℕ → ℝ) 
  (h : strike_time 7 = 42) : strike_time 10 = 60 := by
  sorry

end NUMINAMATH_CALUDE_clock_strike_time_l3383_338340


namespace NUMINAMATH_CALUDE_inequality_proof_l3383_338363

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x^2 + y^2 + z^2 = x + y + z) :
  x + y + z + 3 ≥ 6 * ((xy + yz + zx) / 3)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3383_338363


namespace NUMINAMATH_CALUDE_candy_mixture_cost_l3383_338337

/-- Given a mixture of two types of candy, prove the cost of the second type. -/
theorem candy_mixture_cost
  (first_candy_weight : ℝ)
  (first_candy_cost : ℝ)
  (second_candy_weight : ℝ)
  (mixture_cost : ℝ)
  (h1 : first_candy_weight = 20)
  (h2 : first_candy_cost = 10)
  (h3 : second_candy_weight = 80)
  (h4 : mixture_cost = 6)
  : (((first_candy_weight + second_candy_weight) * mixture_cost
     - first_candy_weight * first_candy_cost) / second_candy_weight) = 5 := by
  sorry

#check candy_mixture_cost

end NUMINAMATH_CALUDE_candy_mixture_cost_l3383_338337


namespace NUMINAMATH_CALUDE_certain_number_proof_l3383_338359

theorem certain_number_proof (x : ℝ) :
  (1.12 * x) / 4.98 = 528.0642570281125 → x = 2350 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3383_338359


namespace NUMINAMATH_CALUDE_items_deleted_l3383_338378

theorem items_deleted (initial : ℕ) (remaining : ℕ) (deleted : ℕ) : 
  initial = 100 → remaining = 20 → deleted = initial - remaining → deleted = 80 :=
by sorry

end NUMINAMATH_CALUDE_items_deleted_l3383_338378


namespace NUMINAMATH_CALUDE_geometric_sequence_150th_term_l3383_338313

/-- Given a geometric sequence with first term 8 and second term -4, 
    the 150th term is equal to -8 * (1/2)^149 -/
theorem geometric_sequence_150th_term : 
  ∀ (a : ℕ → ℝ), 
    (∀ n, a (n + 1) = a n * (-1/2)) → 
    a 1 = 8 → 
    a 2 = -4 → 
    a 150 = -8 * (1/2)^149 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_150th_term_l3383_338313


namespace NUMINAMATH_CALUDE_markup_percentage_of_selling_price_l3383_338302

theorem markup_percentage_of_selling_price 
  (cost selling_price markup : ℝ) 
  (h1 : markup = 0.1 * cost) 
  (h2 : selling_price = cost + markup) :
  markup / selling_price = 100 / 11 / 100 := by
sorry

end NUMINAMATH_CALUDE_markup_percentage_of_selling_price_l3383_338302


namespace NUMINAMATH_CALUDE_unique_real_root_of_system_l3383_338325

theorem unique_real_root_of_system : 
  ∃! x : ℝ, x^3 + 9 = 0 ∧ x + 3 = 0 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_real_root_of_system_l3383_338325


namespace NUMINAMATH_CALUDE_baseball_team_wins_l3383_338304

theorem baseball_team_wins (total_games : ℕ) (ratio : ℚ) (wins : ℕ) : 
  total_games = 10 → 
  ratio = 2 → 
  ratio = total_games / (total_games - wins) → 
  wins = 5 := by
sorry

end NUMINAMATH_CALUDE_baseball_team_wins_l3383_338304


namespace NUMINAMATH_CALUDE_shuai_fen_solution_l3383_338352

/-- Represents the "Shuai Fen" distribution system -/
structure ShuaiFen where
  a : ℝ
  x : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  h_a_pos : a > 0
  h_c : c = 36
  h_bd : b + d = 75
  h_shuai_fen_b : (b - c) / b = x
  h_shuai_fen_c : (c - d) / c = x
  h_shuai_fen_a : (a - b) / a = x
  h_total : a = b + c + d

/-- The "Shuai Fen" problem solution -/
theorem shuai_fen_solution (sf : ShuaiFen) : sf.x = 0.25 ∧ sf.a = 175 := by
  sorry

end NUMINAMATH_CALUDE_shuai_fen_solution_l3383_338352


namespace NUMINAMATH_CALUDE_sticker_distribution_theorem_l3383_338362

/-- The number of ways to distribute n indistinguishable items into k distinguishable boxes --/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of ways to distribute stickers among sheets --/
def distribute_stickers (total_stickers sheets : ℕ) : ℕ :=
  distribute (total_stickers - sheets) sheets

theorem sticker_distribution_theorem :
  distribute_stickers 12 5 = 330 := by sorry

end NUMINAMATH_CALUDE_sticker_distribution_theorem_l3383_338362


namespace NUMINAMATH_CALUDE_lemons_for_twenty_gallons_l3383_338303

/-- Calculates the number of lemons needed for a given volume of lemonade -/
def lemons_needed (base_lemons : ℕ) (base_gallons : ℕ) (target_gallons : ℕ) : ℕ :=
  let base_ratio := base_lemons / base_gallons
  let base_lemons_needed := base_ratio * target_gallons
  let additional_lemons := target_gallons / 10
  base_lemons_needed + additional_lemons

theorem lemons_for_twenty_gallons :
  lemons_needed 40 50 20 = 18 := by
  sorry

#eval lemons_needed 40 50 20

end NUMINAMATH_CALUDE_lemons_for_twenty_gallons_l3383_338303


namespace NUMINAMATH_CALUDE_max_value_on_circle_l3383_338334

theorem max_value_on_circle (x y z : ℝ) (h : x^2 + y^2 - 2*x + 2*y - 1 = 0) :
  ∃ (M : ℝ), M = 2 * Real.sqrt 2 + Real.sqrt 3 ∧ 
  ∀ (w : ℝ), w = (x + 1) * Real.sin z + (y - 1) * Real.cos z → w ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_on_circle_l3383_338334


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l3383_338351

theorem min_value_trig_expression (x : ℝ) : 
  Real.sin x ^ 4 + 2 * Real.cos x ^ 4 + Real.sin x ^ 2 * Real.cos x ^ 2 ≥ 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l3383_338351


namespace NUMINAMATH_CALUDE_regular_dinosaur_count_l3383_338348

theorem regular_dinosaur_count :
  ∀ (barney_weight : ℕ) (regular_dino_weight : ℕ) (total_weight : ℕ) (num_regular_dinos : ℕ),
    regular_dino_weight = 800 →
    barney_weight = regular_dino_weight * num_regular_dinos + 1500 →
    total_weight = barney_weight + regular_dino_weight * num_regular_dinos →
    total_weight = 9500 →
    num_regular_dinos = 5 := by
sorry

end NUMINAMATH_CALUDE_regular_dinosaur_count_l3383_338348


namespace NUMINAMATH_CALUDE_hyperbola_parallel_line_intersection_l3383_338339

-- Define a hyperbola
structure Hyperbola where
  A : ℝ
  B : ℝ
  hAB : A ≠ 0 ∧ B ≠ 0

-- Define a line parallel to the asymptote
structure ParallelLine where
  m : ℝ
  hm : m ≠ 0

-- Theorem statement
theorem hyperbola_parallel_line_intersection (h : Hyperbola) (l : ParallelLine) :
  ∃! p : ℝ × ℝ, 
    (h.A * p.1)^2 - (h.B * p.2)^2 = 1 ∧ 
    h.A * p.1 - h.B * p.2 = l.m :=
sorry

end NUMINAMATH_CALUDE_hyperbola_parallel_line_intersection_l3383_338339


namespace NUMINAMATH_CALUDE_unique_solution_l3383_338332

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x - 3

-- State the theorem
theorem unique_solution :
  ∃! x : ℝ, 2 * (f x) - 21 = f (x - 4) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3383_338332


namespace NUMINAMATH_CALUDE_solution_correctness_l3383_338375

-- First system of equations
def system1 (x y : ℝ) : Prop :=
  3 * x + 2 * y = 6 ∧ y = x - 2

-- Second system of equations
def system2 (m n : ℝ) : Prop :=
  m + 2 * n = 7 ∧ -3 * m + 5 * n = 1

theorem solution_correctness :
  (∃ x y, system1 x y) ∧ (∃ m n, system2 m n) ∧
  (∀ x y, system1 x y → x = 2 ∧ y = 0) ∧
  (∀ m n, system2 m n → m = 3 ∧ n = 2) := by
  sorry

end NUMINAMATH_CALUDE_solution_correctness_l3383_338375


namespace NUMINAMATH_CALUDE_jumping_contest_l3383_338388

theorem jumping_contest (G F M K : ℤ) : 
  G = 39 ∧ 
  G = F + 19 ∧ 
  M = F - 12 ∧ 
  K = 2 * F - 5 →
  K = 35 :=
by sorry

end NUMINAMATH_CALUDE_jumping_contest_l3383_338388
