import Mathlib

namespace NUMINAMATH_CALUDE_grocery_store_inventory_l997_99719

/-- The total number of bottles and fruits in a grocery store -/
def total_items (regular_soda diet_soda sparkling_water orange_juice cranberry_juice apples oranges bananas pears : ℕ) : ℕ :=
  regular_soda + diet_soda + sparkling_water + orange_juice + cranberry_juice + apples + oranges + bananas + pears

/-- Theorem stating the total number of items in the grocery store -/
theorem grocery_store_inventory : 
  total_items 130 88 65 47 27 102 88 74 45 = 666 := by
  sorry

end NUMINAMATH_CALUDE_grocery_store_inventory_l997_99719


namespace NUMINAMATH_CALUDE_train_delay_l997_99731

/-- Proves that a train moving at 6/7 of its usual speed will be 30 minutes late on a journey that usually takes 3 hours. -/
theorem train_delay (usual_speed : ℝ) (usual_time : ℝ) (h1 : usual_time = 3) :
  let current_speed := (6/7) * usual_speed
  let current_time := usual_speed * usual_time / current_speed
  (current_time - usual_time) * 60 = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_delay_l997_99731


namespace NUMINAMATH_CALUDE_fair_special_savings_l997_99791

/-- Calculates the percentage saved when buying three pairs of sandals under the "fair special" --/
theorem fair_special_savings : 
  let regular_price : ℝ := 60
  let second_pair_discount : ℝ := 0.4
  let third_pair_discount : ℝ := 0.25
  let total_regular_price : ℝ := 3 * regular_price
  let discounted_price : ℝ := regular_price + 
                              (1 - second_pair_discount) * regular_price + 
                              (1 - third_pair_discount) * regular_price
  let savings : ℝ := total_regular_price - discounted_price
  let percentage_saved : ℝ := (savings / total_regular_price) * 100
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ |percentage_saved - 22| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_fair_special_savings_l997_99791


namespace NUMINAMATH_CALUDE_two_digit_number_sum_l997_99706

theorem two_digit_number_sum (a b : ℕ) : 
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 →
  (10 * a + b) - (10 * b + a) = 3 * (a + b) →
  (10 * a + b) + (10 * b + a) = 33 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_sum_l997_99706


namespace NUMINAMATH_CALUDE_parallel_lines_m_value_l997_99726

/-- Two lines are parallel if and only if they have the same slope -/
axiom parallel_lines_same_slope {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The equation of line l₁ -/
def line_l₁ (m : ℝ) (x y : ℝ) : Prop := m * x + y - 2 = 0

/-- The equation of line l₂ -/
def line_l₂ (x y : ℝ) : Prop := y = 2 * x - 1

theorem parallel_lines_m_value :
  (∀ x y : ℝ, line_l₁ m x y ↔ line_l₂ x y) → m = -2 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_m_value_l997_99726


namespace NUMINAMATH_CALUDE_trig_identity_l997_99774

theorem trig_identity (α β : ℝ) : 
  Real.sin α ^ 2 + Real.sin β ^ 2 - Real.sin α ^ 2 * Real.sin β ^ 2 + Real.cos α ^ 2 * Real.cos β ^ 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l997_99774


namespace NUMINAMATH_CALUDE_fishing_result_l997_99722

/-- The number of fishes Will and Henry have after fishing -/
def total_fishes (will_catfish : ℕ) (will_eels : ℕ) (henry_trout_per_catfish : ℕ) : ℕ :=
  let will_total := will_catfish + will_eels
  let henry_total := will_catfish * henry_trout_per_catfish
  let henry_keeps := henry_total / 2
  will_total + henry_keeps

/-- Theorem stating the total number of fishes Will and Henry have -/
theorem fishing_result : total_fishes 16 10 3 = 50 := by
  sorry

end NUMINAMATH_CALUDE_fishing_result_l997_99722


namespace NUMINAMATH_CALUDE_max_tangent_circles_is_three_l997_99789

/-- An annulus with inner radius 1 and outer radius 9 -/
structure Annulus where
  inner_radius : ℝ := 1
  outer_radius : ℝ := 9

/-- A circle tangent to both the inner and outer circles of the annulus -/
structure TangentCircle (A : Annulus) where
  radius : ℝ
  center_distance : ℝ
  tangent_inner : center_distance = A.inner_radius + radius
  tangent_outer : center_distance = A.outer_radius - radius

/-- The maximum number of non-overlapping tangent circles in the annulus -/
def max_tangent_circles (A : Annulus) : ℕ :=
  sorry

/-- The theorem stating that the maximum number of non-overlapping tangent circles is 3 -/
theorem max_tangent_circles_is_three (A : Annulus) : max_tangent_circles A = 3 := by
  sorry

end NUMINAMATH_CALUDE_max_tangent_circles_is_three_l997_99789


namespace NUMINAMATH_CALUDE_white_balls_count_l997_99740

def total_balls : ℕ := 40
def red_frequency : ℚ := 15 / 100
def black_frequency : ℚ := 45 / 100

theorem white_balls_count :
  ∃ (white_balls : ℕ),
    white_balls = total_balls * (1 - red_frequency - black_frequency) := by
  sorry

end NUMINAMATH_CALUDE_white_balls_count_l997_99740


namespace NUMINAMATH_CALUDE_initial_value_theorem_l997_99708

theorem initial_value_theorem (y : ℕ) (h : y > 0) :
  ∃ x : ℤ, (x : ℤ) + 49 = y^2 ∧ x = y^2 - 49 :=
by sorry

end NUMINAMATH_CALUDE_initial_value_theorem_l997_99708


namespace NUMINAMATH_CALUDE_unbounded_fraction_over_primes_l997_99721

-- Define the ord_p function
def ord_p (a p : ℕ) : ℕ := sorry

-- State the theorem
theorem unbounded_fraction_over_primes (a : ℕ) (h : a > 1) :
  ∀ M : ℕ, ∃ p : ℕ, Prime p ∧ (p - 1) / ord_p a p > M :=
sorry

end NUMINAMATH_CALUDE_unbounded_fraction_over_primes_l997_99721


namespace NUMINAMATH_CALUDE_other_person_money_l997_99779

/-- If Mia has $110 and this amount is $20 more than twice as much money as someone else, then that person has $45. -/
theorem other_person_money (mia_money : ℕ) (other_money : ℕ) : 
  mia_money = 110 → mia_money = 2 * other_money + 20 → other_money = 45 := by
  sorry

end NUMINAMATH_CALUDE_other_person_money_l997_99779


namespace NUMINAMATH_CALUDE_jacks_paycheck_l997_99707

theorem jacks_paycheck (paycheck : ℝ) : 
  (paycheck * 0.8 * 0.2 = 20) → paycheck = 125 := by
  sorry

end NUMINAMATH_CALUDE_jacks_paycheck_l997_99707


namespace NUMINAMATH_CALUDE_max_side_length_11_l997_99772

/-- Represents a triangle with integer side lengths -/
structure Triangle where
  a : ℕ
  b : ℕ
  c : ℕ
  different : a ≠ b ∧ b ≠ c ∧ a ≠ c
  perimeter_24 : a + b + c = 24
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

/-- The maximum length of any side in a triangle with integer side lengths and perimeter 24 is 11 -/
theorem max_side_length_11 (t : Triangle) : t.a ≤ 11 ∧ t.b ≤ 11 ∧ t.c ≤ 11 := by
  sorry

end NUMINAMATH_CALUDE_max_side_length_11_l997_99772


namespace NUMINAMATH_CALUDE_square_difference_theorem_l997_99751

theorem square_difference_theorem (a b M : ℝ) : 
  (a + 2*b)^2 = (a - 2*b)^2 + M → M = 8*a*b := by
sorry

end NUMINAMATH_CALUDE_square_difference_theorem_l997_99751


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequalities_l997_99738

theorem smallest_integer_satisfying_inequalities :
  ∀ x : ℤ, x < 3 * x - 12 ∧ x > 0 → x ≥ 7 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequalities_l997_99738


namespace NUMINAMATH_CALUDE_certain_number_proof_l997_99771

theorem certain_number_proof (h1 : 213 * 16 = 3408) (h2 : 0.16 * x = 0.3408) : x = 2.13 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l997_99771


namespace NUMINAMATH_CALUDE_work_completion_time_l997_99730

theorem work_completion_time (renu_rate suma_rate : ℚ) 
  (h1 : renu_rate = 1 / 8)
  (h2 : suma_rate = 1 / (24 / 5))
  : (1 / (renu_rate + suma_rate) : ℚ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l997_99730


namespace NUMINAMATH_CALUDE_subject_selection_methods_l997_99792

/-- The number of subjects excluding the mandatory subject -/
def n : ℕ := 5

/-- The number of subjects to be chosen from the remaining subjects -/
def k : ℕ := 2

/-- Combination formula -/
def combination (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem subject_selection_methods :
  combination n k = 10 :=
by sorry

end NUMINAMATH_CALUDE_subject_selection_methods_l997_99792


namespace NUMINAMATH_CALUDE_canoe_row_probability_value_l997_99716

def oar_probability : ℚ := 3/5

/-- The probability of being able to row a canoe with two independent oars -/
def canoe_row_probability : ℚ :=
  oar_probability * oar_probability +  -- Both oars work
  oar_probability * (1 - oar_probability) +  -- Left works, right breaks
  (1 - oar_probability) * oar_probability  -- Left breaks, right works

theorem canoe_row_probability_value :
  canoe_row_probability = 21/25 := by
  sorry

end NUMINAMATH_CALUDE_canoe_row_probability_value_l997_99716


namespace NUMINAMATH_CALUDE_f_2002_equals_96_l997_99711

/-- A function satisfying the given property -/
def special_function (f : ℕ → ℝ) : Prop :=
  ∀ (a b n : ℕ), a > 0 → b > 0 → n > 0 → a + b = 2^n → f a + f b = n^2

/-- The theorem to be proved -/
theorem f_2002_equals_96 (f : ℕ → ℝ) (h : special_function f) : f 2002 = 96 := by
  sorry

end NUMINAMATH_CALUDE_f_2002_equals_96_l997_99711


namespace NUMINAMATH_CALUDE_f_inequality_part1_f_inequality_part2_l997_99759

def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| - |a * x - 1|

theorem f_inequality_part1 :
  ∀ x : ℝ, f 1 x > 1 ↔ x > 1/2 := by sorry

theorem f_inequality_part2 :
  ∀ a : ℝ, (∀ x ∈ Set.Ioo 0 1, f a x > x) ↔ a ∈ Set.Ioc 0 2 := by sorry

end NUMINAMATH_CALUDE_f_inequality_part1_f_inequality_part2_l997_99759


namespace NUMINAMATH_CALUDE_product_inequality_l997_99747

theorem product_inequality (a b x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1)
  (hx₁ : 0 < x₁) (hx₂ : 0 < x₂) (hx₃ : 0 < x₃) (hx₄ : 0 < x₄) (hx₅ : 0 < x₅)
  (hx_prod : x₁ * x₂ * x₃ * x₄ * x₅ = 1) :
  (a * x₁ + b) * (a * x₂ + b) * (a * x₃ + b) * (a * x₄ + b) * (a * x₅ + b) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l997_99747


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l997_99766

theorem smallest_n_congruence (n : ℕ) : 
  (∀ k : ℕ, 0 < k ∧ k < 5 → ¬(1031 * k ≡ 1067 * k [ZMOD 30])) ∧ 
  (1031 * 5 ≡ 1067 * 5 [ZMOD 30]) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l997_99766


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l997_99745

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (1 - 2*x)^5 = a₀ + 2*a₁*x + 4*a₂*x^2 + 8*a₃*x^3 + 16*a₄*x^4 + 32*a₅*x^5) →
  a₁ + a₂ + a₃ + a₄ + a₅ = -1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l997_99745


namespace NUMINAMATH_CALUDE_linear_function_properties_l997_99775

/-- A linear function passing through two given points -/
def linear_function (k b : ℝ) : ℝ → ℝ := λ x => k * x + b

theorem linear_function_properties :
  ∃ (k b : ℝ),
    (linear_function k b (-4) = -9) ∧
    (linear_function k b 3 = 5) ∧
    (k = 2 ∧ b = -1) ∧
    (∃ x, linear_function k b x = 0 ∧ x = 1/2) ∧
    (linear_function k b 0 = -1) ∧
    (1/2 * 1/2 * 1 = 1/4) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_properties_l997_99775


namespace NUMINAMATH_CALUDE_weight_plate_problem_l997_99767

/-- Given 10 weight plates that feel 20% heavier when lowered, prove that each plate weighs 30 pounds if the total weight feels like 360 pounds when lowered. -/
theorem weight_plate_problem (num_plates : ℕ) (perceived_weight : ℝ) (weight_increase : ℝ) :
  num_plates = 10 →
  weight_increase = 0.2 →
  perceived_weight = 360 →
  (perceived_weight / (1 + weight_increase)) / num_plates = 30 := by
sorry

end NUMINAMATH_CALUDE_weight_plate_problem_l997_99767


namespace NUMINAMATH_CALUDE_students_not_in_biology_l997_99737

theorem students_not_in_biology (total_students : ℕ) (biology_percentage : ℚ) 
  (h1 : total_students = 880) 
  (h2 : biology_percentage = 275 / 1000) : 
  total_students - (total_students * biology_percentage).floor = 638 := by
  sorry

end NUMINAMATH_CALUDE_students_not_in_biology_l997_99737


namespace NUMINAMATH_CALUDE_min_sum_perfect_squares_l997_99703

theorem min_sum_perfect_squares (x y : ℤ) (h : x^2 - y^2 = 221) :
  ∃ (a b : ℤ), a^2 - b^2 = 221 ∧ a^2 + b^2 ≤ x^2 + y^2 ∧ a^2 + b^2 = 229 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_perfect_squares_l997_99703


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l997_99735

/-- An isosceles trapezoid with specific dimensions and inscribed circles. -/
structure IsoscelesTrapezoidWithCircles where
  /-- The length of side AB -/
  ab : ℝ
  /-- The length of sides BC and DA -/
  bc_da : ℝ
  /-- The length of side CD -/
  cd : ℝ
  /-- The radius of circles centered at A and B -/
  outer_radius_large : ℝ
  /-- The radius of circles centered at C and D -/
  outer_radius_small : ℝ
  /-- Constraint: AB = 8 -/
  ab_eq : ab = 8
  /-- Constraint: BC = DA = 6 -/
  bc_da_eq : bc_da = 6
  /-- Constraint: CD = 5 -/
  cd_eq : cd = 5
  /-- Constraint: Radius of circles at A and B is 4 -/
  outer_radius_large_eq : outer_radius_large = 4
  /-- Constraint: Radius of circles at C and D is 3 -/
  outer_radius_small_eq : outer_radius_small = 3

/-- The theorem stating the radius of the inscribed circle tangent to all four outer circles. -/
theorem inscribed_circle_radius (t : IsoscelesTrapezoidWithCircles) :
  ∃ r : ℝ, r = (-105 + 4 * Real.sqrt 141) / 13 ∧
    r > 0 ∧
    ∃ (x y : ℝ),
      x^2 + y^2 = (r + t.outer_radius_large)^2 ∧
      x^2 + (t.bc_da - y)^2 = (r + t.outer_radius_small)^2 ∧
      2*x = t.ab - 2*t.outer_radius_large :=
by sorry


end NUMINAMATH_CALUDE_inscribed_circle_radius_l997_99735


namespace NUMINAMATH_CALUDE_wednesday_water_intake_l997_99795

/-- Represents the water intake for a week -/
structure WeeklyWaterIntake where
  total : ℕ
  mon_thu_sat : ℕ
  tue_fri_sun : ℕ
  wed : ℕ

/-- The water intake on Wednesday can be determined from the other data -/
theorem wednesday_water_intake (w : WeeklyWaterIntake)
  (h_total : w.total = 60)
  (h_mon_thu_sat : w.mon_thu_sat = 9)
  (h_tue_fri_sun : w.tue_fri_sun = 8)
  (h_balance : w.total = 3 * w.mon_thu_sat + 3 * w.tue_fri_sun + w.wed) :
  w.wed = 9 := by
  sorry

#check wednesday_water_intake

end NUMINAMATH_CALUDE_wednesday_water_intake_l997_99795


namespace NUMINAMATH_CALUDE_smallest_side_range_l997_99786

theorem smallest_side_range (c : ℝ) (a b d : ℝ) (h1 : c > 0) (h2 : a > 0) (h3 : b > 0) (h4 : d > 0) 
  (h5 : a + b + d = c) (h6 : d = 2 * a) (h7 : a ≤ b) (h8 : a ≤ d) : 
  c / 6 < a ∧ a < c / 4 := by
sorry

end NUMINAMATH_CALUDE_smallest_side_range_l997_99786


namespace NUMINAMATH_CALUDE_olympic_year_zodiac_l997_99788

/-- The zodiac signs in order -/
inductive ZodiacSign
| Rat | Ox | Tiger | Rabbit | Dragon | Snake | Horse | Goat | Monkey | Rooster | Dog | Pig

/-- Function to get the zodiac sign for a given year -/
def getZodiacSign (year : Int) : ZodiacSign :=
  match (year - 1) % 12 with
  | 0 => ZodiacSign.Rooster
  | 1 => ZodiacSign.Dog
  | 2 => ZodiacSign.Pig
  | 3 => ZodiacSign.Rat
  | 4 => ZodiacSign.Ox
  | 5 => ZodiacSign.Tiger
  | 6 => ZodiacSign.Rabbit
  | 7 => ZodiacSign.Dragon
  | 8 => ZodiacSign.Snake
  | 9 => ZodiacSign.Horse
  | 10 => ZodiacSign.Goat
  | _ => ZodiacSign.Monkey

theorem olympic_year_zodiac :
  getZodiacSign 2008 = ZodiacSign.Rabbit :=
by sorry

end NUMINAMATH_CALUDE_olympic_year_zodiac_l997_99788


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l997_99778

theorem cubic_equation_solution (x : ℝ) (hx : x^3 + 6 * (x / (x - 3))^3 = 135) :
  let y := ((x - 3)^3 * (x + 4)) / (3 * x - 4)
  y = 0 ∨ y = 23382 / 122 := by
sorry


end NUMINAMATH_CALUDE_cubic_equation_solution_l997_99778


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l997_99739

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

/-- Given a geometric sequence {a_n} where a_1 + a_3 = 1 and a_2 + a_4 = 2, 
    the sum of the 5th, 6th, 7th, and 8th terms equals 48. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) 
    (h_geometric : is_geometric_sequence a)
    (h_sum1 : a 1 + a 3 = 1)
    (h_sum2 : a 2 + a 4 = 2) :
    a 5 + a 6 + a 7 + a 8 = 48 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l997_99739


namespace NUMINAMATH_CALUDE_tens_digit_of_3_power_2010_l997_99713

theorem tens_digit_of_3_power_2010 : ∃ k : ℕ, 3^2010 = 100 * k + 40 + (3^2010 % 10) :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_3_power_2010_l997_99713


namespace NUMINAMATH_CALUDE_value_of_expression_l997_99763

theorem value_of_expression (x y : ℝ) (hx : x = 12) (hy : y = 7) :
  (x - y) * (x + y) = 95 := by sorry

end NUMINAMATH_CALUDE_value_of_expression_l997_99763


namespace NUMINAMATH_CALUDE_T_eight_three_l997_99712

def T (a b : ℤ) : ℤ := 4*a + 5*b - 1

theorem T_eight_three : T 8 3 = 46 := by sorry

end NUMINAMATH_CALUDE_T_eight_three_l997_99712


namespace NUMINAMATH_CALUDE_ratio_difference_bound_l997_99783

theorem ratio_difference_bound (a : Fin 5 → ℝ) (h : ∀ i, a i > 0) :
  ∃ i j k l : Fin 5, i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧
  |a i / a j - a k / a l| < (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_difference_bound_l997_99783


namespace NUMINAMATH_CALUDE_filter_price_calculation_l997_99718

/-- Proves that the price of each of the remaining 2 filters is $22.55 -/
theorem filter_price_calculation (kit_price : ℝ) (filter1_price : ℝ) (filter2_price : ℝ) 
  (discount_percentage : ℝ) :
  kit_price = 72.50 →
  filter1_price = 12.45 →
  filter2_price = 11.50 →
  discount_percentage = 0.1103448275862069 →
  ∃ (x : ℝ), 
    x = 22.55 ∧
    kit_price = (1 - discount_percentage) * (2 * filter1_price + 2 * x + filter2_price) := by
  sorry

end NUMINAMATH_CALUDE_filter_price_calculation_l997_99718


namespace NUMINAMATH_CALUDE_peanut_butter_servings_l997_99736

-- Define the initial amount of peanut butter in tablespoons
def initial_amount : ℚ := 35 + 2/3

-- Define the amount used for the recipe in tablespoons
def amount_used : ℚ := 5 + 1/3

-- Define the serving size in tablespoons
def serving_size : ℚ := 3

-- Theorem to prove
theorem peanut_butter_servings :
  ⌊(initial_amount - amount_used) / serving_size⌋ = 10 := by
  sorry

end NUMINAMATH_CALUDE_peanut_butter_servings_l997_99736


namespace NUMINAMATH_CALUDE_bonsai_cost_proof_l997_99761

/-- The cost of a small bonsai -/
def small_bonsai_cost : ℝ := 30

/-- The cost of a big bonsai -/
def big_bonsai_cost : ℝ := 20

/-- The number of small bonsai sold -/
def small_bonsai_sold : ℕ := 3

/-- The number of big bonsai sold -/
def big_bonsai_sold : ℕ := 5

/-- The total earnings -/
def total_earnings : ℝ := 190

theorem bonsai_cost_proof :
  small_bonsai_cost * small_bonsai_sold + big_bonsai_cost * big_bonsai_sold = total_earnings :=
by sorry

end NUMINAMATH_CALUDE_bonsai_cost_proof_l997_99761


namespace NUMINAMATH_CALUDE_only_one_divides_power_minus_one_l997_99732

theorem only_one_divides_power_minus_one :
  ∀ n : ℕ, n ≥ 1 → (n ∣ 2^n - 1 ↔ n = 1) := by
  sorry

end NUMINAMATH_CALUDE_only_one_divides_power_minus_one_l997_99732


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l997_99755

theorem arithmetic_geometric_sequence_ratio 
  (a : ℕ → ℝ) 
  (d : ℝ) 
  (h1 : d ≠ 0) 
  (h2 : ∀ n, a (n + 1) = a n + d) 
  (h3 : (a 5 - a 1) * (a 17 - a 5) = (a 5 - a 1)^2) : 
  a 5 / a 1 = 9 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l997_99755


namespace NUMINAMATH_CALUDE_inequality_holds_l997_99782

theorem inequality_holds (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y > 2) :
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_l997_99782


namespace NUMINAMATH_CALUDE_incorrect_statement_C_l997_99756

theorem incorrect_statement_C : 
  (∀ x : ℝ, x > 0 → (∃ y : ℝ, y > 0 ∧ y^2 = x) ∧ (∃! y : ℝ, y > 0 ∧ y^2 = x)) ∧ 
  (∀ x : ℝ, x^3 < 0 → ∃ y : ℝ, y < 0 ∧ y^3 = x) ∧
  (∃ x : ℝ, x^(1/3) = x^(1/2)) ∧
  ¬(∃ y : ℝ, y ≠ 0 ∧ (y^2 = 81 → (y = 9 ∨ y = -9))) :=
by sorry

end NUMINAMATH_CALUDE_incorrect_statement_C_l997_99756


namespace NUMINAMATH_CALUDE_average_monthly_bill_l997_99727

/-- The average monthly bill for a family over 6 months, given the average for the first 4 months and the last 2 months. -/
theorem average_monthly_bill (avg_first_four : ℝ) (avg_last_two : ℝ) : 
  avg_first_four = 30 → avg_last_two = 24 → 
  (4 * avg_first_four + 2 * avg_last_two) / 6 = 28 := by
  sorry

end NUMINAMATH_CALUDE_average_monthly_bill_l997_99727


namespace NUMINAMATH_CALUDE_cyclic_win_sets_count_l997_99762

/-- A round-robin tournament with the given conditions -/
structure Tournament :=
  (num_teams : ℕ)
  (wins_per_team : ℕ)
  (losses_per_team : ℕ)
  (h_round_robin : wins_per_team + losses_per_team = num_teams - 1)
  (h_no_ties : True)

/-- The number of sets of three teams with cyclic wins -/
def cyclic_win_sets (t : Tournament) : ℕ := sorry

/-- The theorem to be proved -/
theorem cyclic_win_sets_count 
  (t : Tournament) 
  (h_num_teams : t.num_teams = 20) 
  (h_wins : t.wins_per_team = 12) 
  (h_losses : t.losses_per_team = 7) : 
  cyclic_win_sets t = 570 := by sorry

end NUMINAMATH_CALUDE_cyclic_win_sets_count_l997_99762


namespace NUMINAMATH_CALUDE_quadratic_equation_problem_l997_99781

theorem quadratic_equation_problem (a : ℝ) : 2 * (5 - a) * (6 + a) = 100 → a^2 + a + 1 = -19 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_problem_l997_99781


namespace NUMINAMATH_CALUDE_rectangle_ratio_l997_99796

/-- Given a rectangle with width 5 inches and area 100 square inches, 
    prove that the ratio of length to width is 4:1 -/
theorem rectangle_ratio (width : ℝ) (length : ℝ) (area : ℝ) :
  width = 5 →
  area = 100 →
  area = length * width →
  length / width = 4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l997_99796


namespace NUMINAMATH_CALUDE_similar_right_triangles_leg_length_l997_99750

/-- Two similar right triangles with legs 12 and 9 in one, and x and 6 in the other, have x = 8 -/
theorem similar_right_triangles_leg_length : ∀ x : ℝ,
  (12 : ℝ) / x = 9 / 6 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_similar_right_triangles_leg_length_l997_99750


namespace NUMINAMATH_CALUDE_angle_C_sides_a_b_max_area_l997_99709

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.c = 2 ∧ Real.sqrt 3 * t.a = 2 * t.c * Real.sin t.A

-- Theorem 1: Angle C
theorem angle_C (t : Triangle) (h : triangle_conditions t) : t.C = π/3 := by
  sorry

-- Theorem 2: Sides a and b
theorem sides_a_b (t : Triangle) (h : triangle_conditions t) 
  (area : (1/2) * t.a * t.b * Real.sin t.C = Real.sqrt 3) : 
  t.a = 2 ∧ t.b = 2 := by
  sorry

-- Theorem 3: Maximum area
theorem max_area (t : Triangle) (h : triangle_conditions t) :
  ∀ (s : Triangle), triangle_conditions s → 
    (1/2) * s.a * s.b * Real.sin s.C ≤ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_sides_a_b_max_area_l997_99709


namespace NUMINAMATH_CALUDE_largest_consecutive_integers_sum_sixty_consecutive_integers_sum_largest_n_is_sixty_l997_99717

theorem largest_consecutive_integers_sum (n : ℕ) : 
  (∃ a : ℕ, a > 0 ∧ n * (2 * a + n - 1) = 4020) → n ≤ 60 :=
by
  sorry

theorem sixty_consecutive_integers_sum : 
  ∃ a : ℕ, a > 0 ∧ 60 * (2 * a + 60 - 1) = 4020 :=
by
  sorry

theorem largest_n_is_sixty : 
  ∀ n : ℕ, (∃ a : ℕ, a > 0 ∧ n * (2 * a + n - 1) = 4020) → n ≤ 60 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_consecutive_integers_sum_sixty_consecutive_integers_sum_largest_n_is_sixty_l997_99717


namespace NUMINAMATH_CALUDE_daniels_purchase_l997_99768

/-- Given the costs of a magazine and a pencil, and a coupon discount,
    calculate the total amount spent. -/
def total_spent (magazine_cost pencil_cost coupon_discount : ℚ) : ℚ :=
  magazine_cost + pencil_cost - coupon_discount

/-- Theorem stating that given the specific costs and discount,
    the total amount spent is $1.00. -/
theorem daniels_purchase :
  total_spent 0.85 0.50 0.35 = 1.00 := by
  sorry

end NUMINAMATH_CALUDE_daniels_purchase_l997_99768


namespace NUMINAMATH_CALUDE_sin_315_degrees_l997_99758

theorem sin_315_degrees : 
  Real.sin (315 * π / 180) = -Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_315_degrees_l997_99758


namespace NUMINAMATH_CALUDE_equation_solutions_l997_99754

theorem equation_solutions : 
  let f (x : ℝ) := (x - 1) * (x - 3) * (x - 5) * (x - 7) * (x - 3) * (x - 5) * (x - 1)
  let g (x : ℝ) := (x - 3) * (x - 7) * (x - 3)
  let S := {x : ℝ | x ≠ 3 ∧ x ≠ 7 ∧ f x / g x = 1}
  S = {3 + Real.sqrt 3, 3 - Real.sqrt 3, 3 + Real.sqrt 5, 3 - Real.sqrt 5} := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l997_99754


namespace NUMINAMATH_CALUDE_square_of_sum_fifteen_three_l997_99793

theorem square_of_sum_fifteen_three : 15^2 + 2*(15*3) + 3^2 = 324 := by
  sorry

end NUMINAMATH_CALUDE_square_of_sum_fifteen_three_l997_99793


namespace NUMINAMATH_CALUDE_bags_found_next_day_l997_99741

theorem bags_found_next_day 
  (initial_bags : ℕ) 
  (total_bags : ℕ) 
  (h : initial_bags ≤ total_bags) :
  total_bags - initial_bags = total_bags - initial_bags :=
by sorry

end NUMINAMATH_CALUDE_bags_found_next_day_l997_99741


namespace NUMINAMATH_CALUDE_optimal_sampling_theorem_l997_99710

/-- Represents the different blood types --/
inductive BloodType
| O
| A
| B
| AB

/-- Represents the available sampling methods --/
inductive SamplingMethod
| Random
| Systematic
| Stratified

structure School :=
  (total_students : Nat)
  (blood_type_counts : BloodType → Nat)
  (sample_size : Nat)
  (soccer_team_size : Nat)
  (soccer_sample_size : Nat)

def optimal_sampling_method (school : School) (is_blood_type_study : Bool) : SamplingMethod :=
  if is_blood_type_study then
    SamplingMethod.Stratified
  else
    SamplingMethod.Random

theorem optimal_sampling_theorem (school : School) :
  (school.total_students = 500) →
  (school.blood_type_counts BloodType.O = 200) →
  (school.blood_type_counts BloodType.A = 125) →
  (school.blood_type_counts BloodType.B = 125) →
  (school.blood_type_counts BloodType.AB = 50) →
  (school.sample_size = 20) →
  (school.soccer_team_size = 11) →
  (school.soccer_sample_size = 2) →
  (optimal_sampling_method school true = SamplingMethod.Stratified) ∧
  (optimal_sampling_method school false = SamplingMethod.Random) :=
by
  sorry

end NUMINAMATH_CALUDE_optimal_sampling_theorem_l997_99710


namespace NUMINAMATH_CALUDE_weight_of_six_moles_of_compound_l997_99725

/-- The weight of a given number of moles of a compound -/
def weight (moles : ℝ) (molecular_weight : ℝ) : ℝ :=
  moles * molecular_weight

/-- Proof that the weight of 6 moles of a compound with molecular weight 1404 is 8424 -/
theorem weight_of_six_moles_of_compound (molecular_weight : ℝ) 
  (h : molecular_weight = 1404) : weight 6 molecular_weight = 8424 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_six_moles_of_compound_l997_99725


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l997_99704

theorem solution_set_of_inequality (x : ℝ) : 
  (2 * x - x^2 > 0) ↔ (x > 0 ∧ x < 2) := by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l997_99704


namespace NUMINAMATH_CALUDE_isosceles_triangle_angle_measure_l997_99780

/-- 
An isosceles triangle with one angle 20% smaller than a right angle 
has its two largest angles measuring 54 degrees each.
-/
theorem isosceles_triangle_angle_measure : 
  ∀ (a b c : ℝ),
  -- The triangle is isosceles
  a = b →
  -- The sum of angles in a triangle is 180°
  a + b + c = 180 →
  -- One angle (c) is 20% smaller than a right angle (90°)
  c = 0.8 * 90 →
  -- Each of the two largest angles (a and b) measures 54°
  a = 54 ∧ b = 54 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_angle_measure_l997_99780


namespace NUMINAMATH_CALUDE_range_of_m_l997_99790

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the property of f being increasing on [-2, 2]
def is_increasing_on_interval (f : ℝ → ℝ) : Prop :=
  ∀ x y, -2 ≤ x ∧ x < y ∧ y ≤ 2 → f x < f y

-- Define the theorem
theorem range_of_m (h1 : is_increasing_on_interval f) (h2 : ∀ m, f (1 - m) < f m) :
  ∀ m, m ∈ Set.Ioo (1/2) 2 ↔ -2 ≤ 1 - m ∧ 1 - m < m ∧ m ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l997_99790


namespace NUMINAMATH_CALUDE_equal_roots_cubic_l997_99715

theorem equal_roots_cubic (m n : ℝ) (h : n ≠ 0) :
  ∃ (x : ℝ), (x^3 + m*x - n = 0 ∧ n*x^3 - 2*m^2*x^2 - 5*m*n*x - 2*m^3 - n^2 = 0) →
  ∃ (a : ℝ), a = (n/2)^(1/3) ∧ 
  (∀ y : ℝ, y^3 + m*y - n = 0 ↔ y = a ∨ y = a ∨ y = -2*a) :=
by sorry

end NUMINAMATH_CALUDE_equal_roots_cubic_l997_99715


namespace NUMINAMATH_CALUDE_stratified_sampling_proportion_l997_99744

/-- Represents the number of students to be selected in a stratified sampling -/
def total_sample : ℕ := 45

/-- Represents the total number of male students -/
def male_population : ℕ := 500

/-- Represents the total number of female students -/
def female_population : ℕ := 400

/-- Represents the number of male students selected in the sample -/
def male_sample : ℕ := 25

/-- Calculates the number of female students to be selected in the sample -/
def female_sample : ℕ := (male_sample * female_population) / male_population

/-- Proves that the calculated female sample size maintains the stratified sampling proportion -/
theorem stratified_sampling_proportion :
  female_sample = 20 ∧
  (male_sample : ℚ) / male_population = (female_sample : ℚ) / female_population ∧
  male_sample + female_sample = total_sample :=
sorry

end NUMINAMATH_CALUDE_stratified_sampling_proportion_l997_99744


namespace NUMINAMATH_CALUDE_ellipse_a_plus_k_eq_eight_l997_99724

/-- An ellipse with given properties -/
structure Ellipse where
  -- Foci coordinates
  f1 : ℝ × ℝ := (1, 1)
  f2 : ℝ × ℝ := (1, 5)
  -- Point on the ellipse
  p : ℝ × ℝ := (-4, 3)
  -- Constants in the equation (x-h)^2/a^2 + (y-k)^2/b^2 = 1
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ
  -- Ensure a and b are positive
  ha : a > 0
  hb : b > 0
  -- Ensure the point p satisfies the equation
  heq : (p.1 - h)^2 / a^2 + (p.2 - k)^2 / b^2 = 1

/-- Theorem stating that a + k = 8 for the given ellipse -/
theorem ellipse_a_plus_k_eq_eight (e : Ellipse) : e.a + e.k = 8 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_a_plus_k_eq_eight_l997_99724


namespace NUMINAMATH_CALUDE_president_vice_president_count_l997_99773

/-- The number of ways to select a president and vice president from 5 people -/
def president_vice_president_selections : ℕ := 20

/-- The number of people to choose from -/
def total_people : ℕ := 5

/-- The number of positions to fill -/
def positions_to_fill : ℕ := 2

theorem president_vice_president_count :
  president_vice_president_selections = total_people * (total_people - 1) :=
sorry

end NUMINAMATH_CALUDE_president_vice_president_count_l997_99773


namespace NUMINAMATH_CALUDE_washing_machine_capacity_l997_99723

theorem washing_machine_capacity 
  (shirts : ℕ) 
  (sweaters : ℕ) 
  (loads : ℕ) 
  (h1 : shirts = 43) 
  (h2 : sweaters = 2) 
  (h3 : loads = 9) : 
  (shirts + sweaters) / loads = 5 := by
  sorry

end NUMINAMATH_CALUDE_washing_machine_capacity_l997_99723


namespace NUMINAMATH_CALUDE_total_hours_worked_l997_99746

theorem total_hours_worked (saturday_hours sunday_hours : ℕ) 
  (h1 : saturday_hours = 6) 
  (h2 : sunday_hours = 4) : 
  saturday_hours + sunday_hours = 10 := by
  sorry

end NUMINAMATH_CALUDE_total_hours_worked_l997_99746


namespace NUMINAMATH_CALUDE_unique_N_for_210_terms_l997_99752

/-- The number of terms in the expansion of (a+b+c+d+1)^n that contain all four variables
    a, b, c, and d, each to some positive power -/
def numTermsWithAllVars (n : ℕ) : ℕ := Nat.choose n 4

theorem unique_N_for_210_terms :
  ∃! N : ℕ, N > 0 ∧ numTermsWithAllVars N = 210 := by sorry

end NUMINAMATH_CALUDE_unique_N_for_210_terms_l997_99752


namespace NUMINAMATH_CALUDE_eight_power_fifteen_div_sixtyfour_power_six_l997_99729

theorem eight_power_fifteen_div_sixtyfour_power_six : 8^15 / 64^6 = 512 := by
  sorry

end NUMINAMATH_CALUDE_eight_power_fifteen_div_sixtyfour_power_six_l997_99729


namespace NUMINAMATH_CALUDE_suitcase_problem_l997_99765

structure SuitcaseScenario where
  total_suitcases : ℕ
  business_suitcases : ℕ
  placement_interval : ℕ

def scenario : SuitcaseScenario :=
  { total_suitcases := 200
  , business_suitcases := 10
  , placement_interval := 2 }

def probability_last_suitcase_at_two_minutes (s : SuitcaseScenario) : ℚ :=
  (Nat.choose 59 9 : ℚ) / (Nat.choose s.total_suitcases s.business_suitcases : ℚ)

def expected_waiting_time (s : SuitcaseScenario) : ℚ :=
  4020 / 11

theorem suitcase_problem (s : SuitcaseScenario) 
  (h1 : s.total_suitcases = 200) 
  (h2 : s.business_suitcases = 10) 
  (h3 : s.placement_interval = 2) :
  probability_last_suitcase_at_two_minutes s = (Nat.choose 59 9 : ℚ) / (Nat.choose 200 10 : ℚ) ∧ 
  expected_waiting_time s = 4020 / 11 := by
  sorry

#eval probability_last_suitcase_at_two_minutes scenario
#eval expected_waiting_time scenario

end NUMINAMATH_CALUDE_suitcase_problem_l997_99765


namespace NUMINAMATH_CALUDE_queue_waiting_times_l997_99702

/-- Represents a queue with Slowpokes and Quickies -/
structure Queue where
  m : ℕ  -- number of Slowpokes
  n : ℕ  -- number of Quickies
  a : ℕ  -- time taken by Quickies
  b : ℕ  -- time taken by Slowpokes

/-- Calculates the minimum total waiting time for a given queue -/
def min_waiting_time (q : Queue) : ℕ :=
  q.a * (q.n.choose 2) + q.a * q.m * q.n + q.b * (q.m.choose 2)

/-- Calculates the maximum total waiting time for a given queue -/
def max_waiting_time (q : Queue) : ℕ :=
  q.a * (q.n.choose 2) + q.b * q.m * q.n + q.b * (q.m.choose 2)

/-- Calculates the expected total waiting time for a given queue -/
def expected_waiting_time (q : Queue) : ℚ :=
  (q.m + q.n).choose 2 * (q.b * q.m + q.a * q.n) / (q.m + q.n)

/-- Theorem stating the properties of the queue waiting times -/
theorem queue_waiting_times (q : Queue) :
  (min_waiting_time q ≤ max_waiting_time q) ∧
  (↑(min_waiting_time q) ≤ expected_waiting_time q) ∧
  (expected_waiting_time q ≤ max_waiting_time q) :=
sorry

end NUMINAMATH_CALUDE_queue_waiting_times_l997_99702


namespace NUMINAMATH_CALUDE_bakery_combinations_l997_99714

/-- The number of ways to distribute n items among k categories, 
    with each category receiving at least m items. -/
def distribute (n k m : ℕ) : ℕ := sorry

/-- There are exactly 3 ways to distribute 7 items among 3 categories, 
    with each category receiving at least 2 items. -/
theorem bakery_combinations : distribute 7 3 2 = 3 := by sorry

end NUMINAMATH_CALUDE_bakery_combinations_l997_99714


namespace NUMINAMATH_CALUDE_store_sales_l997_99743

-- Define the prices and quantities of each pencil type
def eraser_price : ℚ := 0.8
def regular_price : ℚ := 0.5
def short_price : ℚ := 0.4
def mechanical_price : ℚ := 1.2
def novelty_price : ℚ := 1.5

def eraser_quantity : ℕ := 200
def regular_quantity : ℕ := 40
def short_quantity : ℕ := 35
def mechanical_quantity : ℕ := 25
def novelty_quantity : ℕ := 15

-- Define the total sales function
def total_sales : ℚ :=
  eraser_price * eraser_quantity +
  regular_price * regular_quantity +
  short_price * short_quantity +
  mechanical_price * mechanical_quantity +
  novelty_price * novelty_quantity

-- Theorem statement
theorem store_sales : total_sales = 246.5 := by
  sorry

end NUMINAMATH_CALUDE_store_sales_l997_99743


namespace NUMINAMATH_CALUDE_unique_valid_subset_l997_99787

def original_set : Finset ℕ := {1, 3, 4, 5, 7, 8, 9, 11, 12, 14}

def is_valid_subset (s : Finset ℕ) : Prop :=
  s.card = 2 ∧ 
  s ⊆ original_set ∧
  (Finset.sum (original_set \ s) id) / (original_set.card - 2 : ℚ) = 7

theorem unique_valid_subset : ∃! s : Finset ℕ, is_valid_subset s := by
  sorry

end NUMINAMATH_CALUDE_unique_valid_subset_l997_99787


namespace NUMINAMATH_CALUDE_triangle_base_length_l997_99728

/-- Given a triangle with area 615 and height 10, prove its base is 123 -/
theorem triangle_base_length (area : ℝ) (height : ℝ) (base : ℝ) 
  (h_area : area = 615) 
  (h_height : height = 10) 
  (h_triangle_area : area = (base * height) / 2) : base = 123 := by
  sorry

end NUMINAMATH_CALUDE_triangle_base_length_l997_99728


namespace NUMINAMATH_CALUDE_hall_volume_l997_99733

/-- A rectangular hall with specific dimensions and area properties -/
structure RectangularHall where
  length : ℝ
  width : ℝ
  height : ℝ
  area_equality : 2 * (length * width) = 2 * (length * height + width * height)

/-- The volume of a rectangular hall is 900 cubic meters -/
theorem hall_volume (hall : RectangularHall) 
  (h_length : hall.length = 15)
  (h_width : hall.width = 10) : 
  hall.length * hall.width * hall.height = 900 := by
  sorry

#check hall_volume

end NUMINAMATH_CALUDE_hall_volume_l997_99733


namespace NUMINAMATH_CALUDE_expand_product_l997_99794

theorem expand_product (y : ℝ) : 4 * (y - 3) * (y^2 + 2*y + 4) = 4*y^3 - 4*y^2 - 8*y - 48 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l997_99794


namespace NUMINAMATH_CALUDE_projection_magnitude_l997_99705

theorem projection_magnitude (a b : ℝ × ℝ) :
  (a.1 * b.1 + a.2 * b.2 = -2) →
  (b = (1, Real.sqrt 3)) →
  let c := ((a.1 * b.1 + a.2 * b.2) / (b.1 ^ 2 + b.2 ^ 2)) • b
  (b.1 - c.1) ^ 2 + (b.2 - c.2) ^ 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_projection_magnitude_l997_99705


namespace NUMINAMATH_CALUDE_twentieth_term_of_specific_sequence_l997_99797

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem twentieth_term_of_specific_sequence :
  arithmetic_sequence 2 5 20 = 97 := by
sorry

end NUMINAMATH_CALUDE_twentieth_term_of_specific_sequence_l997_99797


namespace NUMINAMATH_CALUDE_hyperbola_transverse_axis_length_l997_99720

/-- Given a hyperbola with equation x²/a² - y² = 1 where a > 0 and eccentricity = 2,
    prove that the length of its transverse axis is 2√3/3 -/
theorem hyperbola_transverse_axis_length (a : ℝ) (h1 : a > 0) :
  let e := 2  -- eccentricity
  let c := Real.sqrt (a^2 + 1)  -- focal distance
  e = c / a →  -- definition of eccentricity
  2 * a = 2 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_transverse_axis_length_l997_99720


namespace NUMINAMATH_CALUDE_cube_difference_l997_99734

theorem cube_difference (x : ℝ) (h : x - 1/x = 5) : x^3 - 1/x^3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_l997_99734


namespace NUMINAMATH_CALUDE_circle_equation_l997_99776

/-- The equation of a circle with center (1, 1) and radius 1 -/
theorem circle_equation : 
  ∀ (x y : ℝ), (x - 1)^2 + (y - 1)^2 = 1 ↔ 
  ((x, y) : ℝ × ℝ) ∈ {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 1)^2 = 1} :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l997_99776


namespace NUMINAMATH_CALUDE_adult_ticket_cost_is_correct_l997_99784

/-- The cost of an adult ticket in dollars -/
def adult_ticket_cost : ℕ := 19

/-- The cost of a child ticket in dollars -/
def child_ticket_cost : ℕ := adult_ticket_cost - 6

/-- The total number of tickets -/
def total_tickets : ℕ := 5

/-- The number of adult tickets -/
def adult_tickets : ℕ := 2

/-- The number of child tickets -/
def child_tickets : ℕ := 3

/-- The total cost of all tickets in dollars -/
def total_cost : ℕ := 77

theorem adult_ticket_cost_is_correct : 
  adult_tickets * adult_ticket_cost + child_tickets * child_ticket_cost = total_cost :=
by sorry

end NUMINAMATH_CALUDE_adult_ticket_cost_is_correct_l997_99784


namespace NUMINAMATH_CALUDE_min_packages_for_scooter_l997_99757

/-- The minimum number of packages to recover the cost of a scooter -/
def min_packages (scooter_cost : ℕ) (earning_per_package : ℕ) (fuel_cost : ℕ) : ℕ :=
  (scooter_cost + (earning_per_package - fuel_cost - 1)) / (earning_per_package - fuel_cost)

/-- Theorem stating the minimum number of packages needed to recover the scooter cost -/
theorem min_packages_for_scooter :
  min_packages 3200 15 4 = 291 :=
by sorry

end NUMINAMATH_CALUDE_min_packages_for_scooter_l997_99757


namespace NUMINAMATH_CALUDE_even_number_of_even_scores_l997_99769

/-- Represents a team's score in the basketball competition -/
structure TeamScore where
  wins : ℕ
  losses : ℕ
  draws : ℕ

/-- The total number of teams in the competition -/
def numTeams : ℕ := 10

/-- The number of games each team plays -/
def gamesPerTeam : ℕ := numTeams - 1

/-- Calculate the total score for a team -/
def totalScore (ts : TeamScore) : ℕ :=
  2 * ts.wins + ts.draws

/-- The scores of all teams in the competition -/
def allTeamScores : Finset TeamScore :=
  sorry

/-- The sum of all team scores is equal to the total number of games multiplied by 2 -/
axiom total_score_sum : 
  (allTeamScores.sum totalScore) = (numTeams * gamesPerTeam)

/-- Theorem: There must be an even number of teams with an even total score -/
theorem even_number_of_even_scores : 
  Even (Finset.filter (fun ts => Even (totalScore ts)) allTeamScores).card :=
sorry

end NUMINAMATH_CALUDE_even_number_of_even_scores_l997_99769


namespace NUMINAMATH_CALUDE_quadratic_solutions_l997_99749

/-- Given a quadratic function f(x) = x^2 + mx, if its axis of symmetry is x = 3,
    then the solutions to x^2 + mx = 0 are 0 and 6. -/
theorem quadratic_solutions (m : ℝ) :
  (∀ x, x^2 + m*x = (x - 3)^2 + k) →
  (∃ k, ∀ x, x^2 + m*x = (x - 3)^2 + k) →
  (∀ x, x^2 + m*x = 0 ↔ x = 0 ∨ x = 6) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_solutions_l997_99749


namespace NUMINAMATH_CALUDE_power_function_range_l997_99701

def power_function (x : ℝ) (m : ℕ+) : ℝ := x^(3*m.val - 9)

theorem power_function_range (m : ℕ+) 
  (h1 : ∀ (x : ℝ), x > 0 → ∀ (y : ℝ), y > x → power_function y m < power_function x m)
  (h2 : ∀ (x : ℝ), power_function x m = power_function (-x) m) :
  {a : ℝ | (a + 1)^(m.val/3) < (3 - 2*a)^(m.val/3)} = {a : ℝ | a < 2/3} := by
sorry

end NUMINAMATH_CALUDE_power_function_range_l997_99701


namespace NUMINAMATH_CALUDE_carnival_total_cost_l997_99798

def carnival_cost (bumper_rides_mara : ℕ) (space_rides_riley : ℕ) (ferris_rides_each : ℕ)
  (bumper_cost : ℕ) (space_cost : ℕ) (ferris_cost : ℕ) : ℕ :=
  bumper_rides_mara * bumper_cost +
  space_rides_riley * space_cost +
  2 * ferris_rides_each * ferris_cost

theorem carnival_total_cost :
  carnival_cost 2 4 3 2 4 5 = 50 := by
  sorry

end NUMINAMATH_CALUDE_carnival_total_cost_l997_99798


namespace NUMINAMATH_CALUDE_book_selection_theorem_l997_99770

theorem book_selection_theorem (n m : ℕ) (h1 : n = 8) (h2 : m = 5) :
  (Nat.choose (n - 1) (m - 1)) = 35 := by
  sorry

end NUMINAMATH_CALUDE_book_selection_theorem_l997_99770


namespace NUMINAMATH_CALUDE_determine_friendship_graph_l997_99760

/-- Represents the friendship graph among apprentices -/
def FriendshipGraph := Fin 10 → Fin 10 → Prop

/-- Represents a duty assignment for a single day -/
def DutyAssignment := Fin 10 → Bool

/-- Calculates the number of missing pastries for a given duty assignment and friendship graph -/
def missingPastries (duty : DutyAssignment) (friends : FriendshipGraph) : ℕ :=
  sorry

/-- Theorem: The chef can determine the friendship graph after 45 days -/
theorem determine_friendship_graph 
  (friends : FriendshipGraph) :
  ∃ (assignments : Fin 45 → DutyAssignment),
    ∀ (other_friends : FriendshipGraph),
      (∀ (day : Fin 45), missingPastries (assignments day) friends = 
                          missingPastries (assignments day) other_friends) →
      friends = other_friends :=
sorry

end NUMINAMATH_CALUDE_determine_friendship_graph_l997_99760


namespace NUMINAMATH_CALUDE_student_transfer_fraction_l997_99748

theorem student_transfer_fraction (initial_students new_students final_students : ℕ) : 
  initial_students = 160 →
  new_students = 20 →
  final_students = 120 →
  (initial_students + new_students - final_students) / (initial_students + new_students) = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_student_transfer_fraction_l997_99748


namespace NUMINAMATH_CALUDE_positive_sequence_existence_l997_99777

theorem positive_sequence_existence :
  ∃ (a : ℕ → ℝ) (a₁ : ℝ), 
    (∀ n, a n > 0) ∧
    (∀ n, a (n + 2) = a n - a (n + 1)) ∧
    (a₁ > 0) ∧
    (∀ n, a n = a₁ * ((Real.sqrt 5 - 1) / 2) ^ (n - 1)) :=
by sorry

end NUMINAMATH_CALUDE_positive_sequence_existence_l997_99777


namespace NUMINAMATH_CALUDE_coin_diameter_is_14_l997_99785

/-- The diameter of a coin given its radius -/
def coin_diameter (radius : ℝ) : ℝ := 2 * radius

/-- Theorem: The diameter of a coin with radius 7 cm is 14 cm -/
theorem coin_diameter_is_14 : coin_diameter 7 = 14 := by
  sorry

end NUMINAMATH_CALUDE_coin_diameter_is_14_l997_99785


namespace NUMINAMATH_CALUDE_no_nonperiodic_function_satisfies_equation_l997_99753

theorem no_nonperiodic_function_satisfies_equation :
  ¬∃ f : ℝ → ℝ, (∀ x : ℝ, f (x + 1) = f x * (f x + 1)) ∧ (¬∃ p : ℝ, p ≠ 0 ∧ ∀ x : ℝ, f (x + p) = f x) :=
by sorry

end NUMINAMATH_CALUDE_no_nonperiodic_function_satisfies_equation_l997_99753


namespace NUMINAMATH_CALUDE_solution_set_a_eq_one_range_of_a_l997_99799

-- Define the function f(x, a)
def f (x a : ℝ) : ℝ := |x + a| + |x|

-- Theorem 1: Solution set when a = 1
theorem solution_set_a_eq_one :
  {x : ℝ | f x 1 < 3} = {x : ℝ | -2 < x ∧ x < 1} := by sorry

-- Theorem 2: Range of a when f(x) < 3 has a solution
theorem range_of_a (a : ℝ) :
  (∃ x, f x a < 3) ↔ -3 < a ∧ a < 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_a_eq_one_range_of_a_l997_99799


namespace NUMINAMATH_CALUDE_hillary_friday_reading_time_l997_99742

/-- The total assigned reading time in minutes -/
def total_assigned_time : ℕ := 60

/-- The number of minutes Hillary read on Saturday -/
def saturday_reading_time : ℕ := 28

/-- The number of minutes Hillary needs to read on Sunday -/
def sunday_reading_time : ℕ := 16

/-- The number of minutes Hillary read on Friday night -/
def friday_reading_time : ℕ := total_assigned_time - (saturday_reading_time + sunday_reading_time)

theorem hillary_friday_reading_time :
  friday_reading_time = 16 := by sorry

end NUMINAMATH_CALUDE_hillary_friday_reading_time_l997_99742


namespace NUMINAMATH_CALUDE_geometric_series_relation_l997_99764

/-- Given two infinite geometric series with the specified conditions, prove that n = 20/3 -/
theorem geometric_series_relation (a₁ b₁ a₂ b₂ n : ℝ) : 
  a₁ = 15 ∧ b₁ = 5 ∧ a₂ = 15 ∧ b₂ = 5 + n ∧ 
  (a₁ / (1 - b₁ / a₁)) * 3 = a₂ / (1 - b₂ / a₂) → 
  n = 20 / 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_relation_l997_99764


namespace NUMINAMATH_CALUDE_tan_105_degrees_l997_99700

theorem tan_105_degrees : Real.tan (105 * π / 180) = -2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_105_degrees_l997_99700
