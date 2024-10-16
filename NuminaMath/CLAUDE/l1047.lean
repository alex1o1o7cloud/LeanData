import Mathlib

namespace NUMINAMATH_CALUDE_oneTwentiethOf80_l1047_104755

-- Define the given condition
def oneNinthOf60 : ℚ := 5

-- Define the function to calculate a fraction of a number
def fractionOf (numerator denominator value : ℚ) : ℚ :=
  (numerator / denominator) * value

-- Theorem statement
theorem oneTwentiethOf80 : fractionOf 1 20 80 = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_oneTwentiethOf80_l1047_104755


namespace NUMINAMATH_CALUDE_right_triangle_third_side_product_l1047_104776

theorem right_triangle_third_side_product (a b c : ℝ) : 
  a = 4 → b = 5 → (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) → 
  (c = 3 ∨ c = Real.sqrt 41) → 
  3 * Real.sqrt 41 = (if c = 3 then 3 else Real.sqrt 41) * (if c = Real.sqrt 41 then Real.sqrt 41 else 3) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_product_l1047_104776


namespace NUMINAMATH_CALUDE_pond_length_l1047_104754

/-- Given a rectangular field with length 20 m and width 10 m, containing a square pond
    whose area is 1/8 of the field's area, the length of the pond is 5 m. -/
theorem pond_length (field_length field_width pond_area : ℝ) : 
  field_length = 20 →
  field_width = 10 →
  field_length = 2 * field_width →
  pond_area = (1 / 8) * (field_length * field_width) →
  Real.sqrt pond_area = 5 := by
  sorry


end NUMINAMATH_CALUDE_pond_length_l1047_104754


namespace NUMINAMATH_CALUDE_kolya_purchase_l1047_104784

/-- Represents the price of an item in kopecks -/
def ItemPrice (rubles : ℕ) : ℕ := 100 * rubles + 99

/-- Represents Kolya's total purchase in kopecks -/
def TotalPurchase : ℕ := 200 * 100 + 83

/-- Predicate to check if a given number of items satisfies the purchase conditions -/
def ValidPurchase (n : ℕ) : Prop :=
  ∃ (rubles : ℕ), n * ItemPrice rubles = TotalPurchase

theorem kolya_purchase :
  {n : ℕ | ValidPurchase n} = {17, 117} := by sorry

end NUMINAMATH_CALUDE_kolya_purchase_l1047_104784


namespace NUMINAMATH_CALUDE_cos_240_degrees_l1047_104780

theorem cos_240_degrees : Real.cos (240 * π / 180) = -(1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_cos_240_degrees_l1047_104780


namespace NUMINAMATH_CALUDE_unique_function_solution_l1047_104751

-- Define the property that f should satisfy
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * y - 1) + f x * f y = 2 * x * y - 1

-- State the theorem
theorem unique_function_solution :
  ∀ f : ℝ → ℝ, SatisfiesEquation f → (∀ x : ℝ, f x = x) :=
by
  sorry

end NUMINAMATH_CALUDE_unique_function_solution_l1047_104751


namespace NUMINAMATH_CALUDE_cricket_team_age_difference_l1047_104794

theorem cricket_team_age_difference (team_size : ℕ) (captain_age : ℕ) (team_avg_age : ℚ) :
  team_size = 11 →
  captain_age = 27 →
  team_avg_age = 24 →
  ∃ (wicket_keeper_age : ℕ),
    (team_avg_age * team_size - captain_age - wicket_keeper_age) / (team_size - 2) = team_avg_age - 1 →
    wicket_keeper_age = captain_age + 3 :=
by sorry

end NUMINAMATH_CALUDE_cricket_team_age_difference_l1047_104794


namespace NUMINAMATH_CALUDE_max_cube_in_tetrahedron_l1047_104710

/-- The maximum edge length of a cube that can rotate freely inside a regular tetrahedron -/
def max_cube_edge_length (tetrahedron_edge : ℝ) : ℝ :=
  2

/-- Theorem: The maximum edge length of a cube that can rotate freely inside a regular tetrahedron
    with edge length 6√2 is equal to 2 -/
theorem max_cube_in_tetrahedron :
  max_cube_edge_length (6 * Real.sqrt 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_max_cube_in_tetrahedron_l1047_104710


namespace NUMINAMATH_CALUDE_shirts_sold_proof_l1047_104723

/-- The number of shirts sold by Sab and Dane -/
def num_shirts : ℕ := 18

/-- The number of pairs of shoes sold -/
def num_shoes : ℕ := 6

/-- The price of each pair of shoes in dollars -/
def price_shoes : ℕ := 3

/-- The price of each shirt in dollars -/
def price_shirts : ℕ := 2

/-- The earnings of each person (Sab and Dane) in dollars -/
def earnings_per_person : ℕ := 27

theorem shirts_sold_proof : 
  num_shirts = 18 ∧ 
  num_shoes * price_shoes + num_shirts * price_shirts = 2 * earnings_per_person := by
  sorry

end NUMINAMATH_CALUDE_shirts_sold_proof_l1047_104723


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1047_104730

theorem polynomial_factorization (x y z : ℝ) :
  x^3 * (y^2 - z^2) + y^3 * (z^2 - x^2) + z^3 * (x^2 - y^2) =
  -(x - y) * (y - z) * (z - x) * (x*y + x*z + y*z) := by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1047_104730


namespace NUMINAMATH_CALUDE_cubic_root_product_l1047_104779

theorem cubic_root_product (a b c : ℝ) : 
  (a^3 - 18*a^2 + 20*a - 8 = 0) ∧ 
  (b^3 - 18*b^2 + 20*b - 8 = 0) ∧ 
  (c^3 - 18*c^2 + 20*c - 8 = 0) →
  (2+a)*(2+b)*(2+c) = 128 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_product_l1047_104779


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1047_104738

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_properties
  (a : ℕ → ℤ)
  (h_arith : arithmetic_sequence a)
  (h1 : a 5 = -1)
  (h2 : a 8 = 2)
  (m n : ℕ+)
  (h3 : m ≠ n)
  (h4 : a m = n)
  (h5 : a n = m) :
  (a 1 = -5 ∧ ∃ d : ℤ, d = 1 ∧ ∀ k : ℕ, a (k + 1) = a k + d) ∧
  a (m + n) = 0 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1047_104738


namespace NUMINAMATH_CALUDE_line_equation_l1047_104763

/-- The circle with center (-1, 2) and radius √(5-a) --/
def Circle (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 + 1)^2 + (p.2 - 2)^2 = 5 - a}

/-- The line l --/
def Line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 - p.2 + 1 = 0}

/-- The midpoint of the chord AB --/
def M : ℝ × ℝ := (0, 1)

theorem line_equation (a : ℝ) (h : a < 3) :
  ∃ A B : ℝ × ℝ, A ∈ Circle a ∧ B ∈ Circle a ∧
  A ∈ Line ∧ B ∈ Line ∧
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  Line = {p : ℝ × ℝ | p.1 - p.2 + 1 = 0} := by
sorry

end NUMINAMATH_CALUDE_line_equation_l1047_104763


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1047_104737

theorem quadratic_inequality_solution_set :
  {x : ℝ | (x + 3) * (2 - x) < 0} = {x : ℝ | x < -3 ∨ x > 2} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1047_104737


namespace NUMINAMATH_CALUDE_translated_linear_function_range_l1047_104701

theorem translated_linear_function_range (x : ℝ) :
  let f : ℝ → ℝ := fun x ↦ x + 2
  f x > 0 → x > -2 := by
sorry

end NUMINAMATH_CALUDE_translated_linear_function_range_l1047_104701


namespace NUMINAMATH_CALUDE_unique_positive_solution_l1047_104717

def f (x : ℝ) : ℝ := x^11 + 9*x^10 + 19*x^9 + 2023*x^8 - 1421*x^7 + 5

theorem unique_positive_solution :
  ∃! x : ℝ, x > 0 ∧ f x = 0 := by sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l1047_104717


namespace NUMINAMATH_CALUDE_complex_equation_sum_l1047_104729

theorem complex_equation_sum (a t : ℝ) (i : ℂ) : 
  i * i = -1 → 
  a + i = (1 + 2*i) * (t*i) → 
  t + a = -1 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l1047_104729


namespace NUMINAMATH_CALUDE_triple_equation_solution_l1047_104747

theorem triple_equation_solution (a b c : ℝ) :
  (a * (b^2 + c) = c * (c + a * b)) ∧
  (b * (c^2 + a) = a * (a + b * c)) ∧
  (c * (a^2 + b) = b * (b + c * a)) →
  (∃ x : ℝ, a = x ∧ b = x ∧ c = x) ∨
  (b = 0 ∧ c = 0) :=
sorry

end NUMINAMATH_CALUDE_triple_equation_solution_l1047_104747


namespace NUMINAMATH_CALUDE_carl_typing_words_l1047_104727

/-- Calculates the total number of words typed given a typing speed, daily typing duration, and number of days. -/
def total_words_typed (typing_speed : ℕ) (hours_per_day : ℕ) (days : ℕ) : ℕ :=
  typing_speed * 60 * hours_per_day * days

/-- Proves that given the specified conditions, Carl types 84000 words in 7 days. -/
theorem carl_typing_words :
  total_words_typed 50 4 7 = 84000 := by
  sorry

end NUMINAMATH_CALUDE_carl_typing_words_l1047_104727


namespace NUMINAMATH_CALUDE_symmetry_condition_l1047_104706

/-- A function f is symmetric about a line x = k if f(k + t) = f(k - t) for all t. -/
def IsSymmetricAbout (f : ℝ → ℝ) (k : ℝ) : Prop :=
  ∀ t, f (k + t) = f (k - t)

/-- The main theorem stating the condition for symmetry of the given function. -/
theorem symmetry_condition (a : ℝ) :
  IsSymmetricAbout (fun x => a * Real.sin x + Real.cos (x + π/6)) (π/3) ↔ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_condition_l1047_104706


namespace NUMINAMATH_CALUDE_parabolas_intersection_l1047_104739

-- Define the two parabola functions
def f (x : ℝ) : ℝ := 2 * x^2 + 5 * x - 3
def g (x : ℝ) : ℝ := x^2 + 2

-- Theorem statement
theorem parabolas_intersection :
  (∃ (x y : ℝ), f x = g x ∧ y = f x) ↔
  (∃ (x y : ℝ), (x = -5 ∧ y = 27) ∨ (x = 1 ∧ y = 3)) :=
by sorry

end NUMINAMATH_CALUDE_parabolas_intersection_l1047_104739


namespace NUMINAMATH_CALUDE_evaluate_expression_l1047_104795

theorem evaluate_expression : 3000 * (3000 ^ 3000) = 3000 ^ 3001 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1047_104795


namespace NUMINAMATH_CALUDE_correct_chest_contents_l1047_104762

-- Define the types of coins
inductive CoinType
| Gold
| Silver
| Copper

-- Define the chests
structure Chest where
  label : CoinType
  content : CoinType

-- Define the problem setup
def setup : List Chest := [
  { label := CoinType.Gold, content := CoinType.Silver },
  { label := CoinType.Silver, content := CoinType.Gold },
  { label := CoinType.Gold, content := CoinType.Copper }
]

-- Theorem statement
theorem correct_chest_contents :
  ∀ (chests : List Chest),
  (chests.length = 3) →
  (∃! c, c ∈ chests ∧ c.content = CoinType.Gold) →
  (∃! c, c ∈ chests ∧ c.content = CoinType.Silver) →
  (∃! c, c ∈ chests ∧ c.content = CoinType.Copper) →
  (∀ c ∈ chests, c.label ≠ c.content) →
  (chests = setup) :=
by sorry

end NUMINAMATH_CALUDE_correct_chest_contents_l1047_104762


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_4_eq_neg_3_implies_cos_2alpha_plus_2sin_2alpha_eq_1_l1047_104767

theorem tan_alpha_plus_pi_4_eq_neg_3_implies_cos_2alpha_plus_2sin_2alpha_eq_1 
  (α : ℝ) (h : Real.tan (α + π/4) = -3) : 
  Real.cos (2*α) + 2 * Real.sin (2*α) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_4_eq_neg_3_implies_cos_2alpha_plus_2sin_2alpha_eq_1_l1047_104767


namespace NUMINAMATH_CALUDE_vasya_fish_count_l1047_104756

/-- Represents the number of fish Vasya caught -/
def total_fish : ℕ := 10

/-- Represents the weight of the three largest fish as a fraction of the total catch -/
def largest_fish_weight_fraction : ℚ := 35 / 100

/-- Represents the weight of the three smallest fish as a fraction of the remaining catch -/
def smallest_fish_weight_fraction : ℚ := 5 / 13

/-- Represents the number of largest fish -/
def num_largest_fish : ℕ := 3

/-- Represents the number of smallest fish -/
def num_smallest_fish : ℕ := 3

theorem vasya_fish_count :
  ∃ (x : ℕ),
    total_fish = num_largest_fish + x + num_smallest_fish ∧
    (1 - largest_fish_weight_fraction) * smallest_fish_weight_fraction = 
      (25 : ℚ) / 100 ∧
    (35 : ℚ) / 3 ≤ (40 : ℚ) / x ∧
    (40 : ℚ) / x ≤ (25 : ℚ) / 3 :=
sorry

end NUMINAMATH_CALUDE_vasya_fish_count_l1047_104756


namespace NUMINAMATH_CALUDE_divisibility_by_three_l1047_104720

theorem divisibility_by_three (n : ℕ) : 
  (3 ∣ n * 2^n + 1) ↔ (n % 6 = 1 ∨ n % 6 = 2) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_by_three_l1047_104720


namespace NUMINAMATH_CALUDE_train_speed_l1047_104731

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed (train_length : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 135 →
  bridge_length = 240.03 →
  crossing_time = 30 →
  (((train_length + bridge_length) / crossing_time) * 3.6) = 45.0036 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l1047_104731


namespace NUMINAMATH_CALUDE_distinct_arrangements_l1047_104748

def word_length : ℕ := 6
def freq_letter1 : ℕ := 1
def freq_letter2 : ℕ := 2
def freq_letter3 : ℕ := 3

theorem distinct_arrangements :
  (word_length.factorial) / (freq_letter1.factorial * freq_letter2.factorial * freq_letter3.factorial) = 60 := by
  sorry

end NUMINAMATH_CALUDE_distinct_arrangements_l1047_104748


namespace NUMINAMATH_CALUDE_lost_to_initial_ratio_l1047_104715

/-- Represents the number of black socks Andy initially had -/
def initial_black_socks : ℕ := 6

/-- Represents the number of white socks Andy initially had -/
def initial_white_socks : ℕ := 4 * initial_black_socks

/-- Represents the number of white socks Andy has after losing some -/
def remaining_white_socks : ℕ := initial_black_socks + 6

/-- Represents the number of white socks Andy lost -/
def lost_white_socks : ℕ := initial_white_socks - remaining_white_socks

/-- Theorem stating that the ratio of lost white socks to initial white socks is 1/2 -/
theorem lost_to_initial_ratio :
  (lost_white_socks : ℚ) / initial_white_socks = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_lost_to_initial_ratio_l1047_104715


namespace NUMINAMATH_CALUDE_equal_cost_at_20_minutes_unique_solution_l1047_104728

/-- The base rate for United Telephone service -/
def united_base_rate : ℝ := 11

/-- The per-minute charge for United Telephone -/
def united_per_minute : ℝ := 0.25

/-- The base rate for Atlantic Call service -/
def atlantic_base_rate : ℝ := 12

/-- The per-minute charge for Atlantic Call -/
def atlantic_per_minute : ℝ := 0.20

/-- The total cost for United Telephone service for m minutes -/
def united_cost (m : ℝ) : ℝ := united_base_rate + united_per_minute * m

/-- The total cost for Atlantic Call service for m minutes -/
def atlantic_cost (m : ℝ) : ℝ := atlantic_base_rate + atlantic_per_minute * m

/-- Theorem stating that the costs are equal at 20 minutes -/
theorem equal_cost_at_20_minutes : 
  united_cost 20 = atlantic_cost 20 :=
by sorry

/-- Theorem stating that 20 minutes is the unique solution -/
theorem unique_solution (m : ℝ) :
  united_cost m = atlantic_cost m ↔ m = 20 :=
by sorry

end NUMINAMATH_CALUDE_equal_cost_at_20_minutes_unique_solution_l1047_104728


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l1047_104761

-- Define the given line
def given_line (x y : ℝ) : Prop := 2 * x + y - 5 = 0

-- Define the point A
def point_A : ℝ × ℝ := (2, -3)

-- Define the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := x - 2 * y - 8 = 0

-- Theorem statement
theorem perpendicular_line_through_point :
  ∀ (x y : ℝ),
    (perpendicular_line x y ∧ (x, y) = point_A) →
    (∀ (x' y' : ℝ), given_line x' y' → (x - x') * (x' - x) + (y - y') * (y' - y) = 0) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l1047_104761


namespace NUMINAMATH_CALUDE_problem_cube_surface_area_l1047_104722

/-- Represents the structure of the cube after modifications -/
structure ModifiedCube where
  initial_size : Nat
  small_cube_size : Nat
  small_cube_count : Nat
  center_face_removed : Bool
  center_small_cube_removed : Bool
  small_cube_faces_removed : Bool

/-- Calculates the surface area of the modified cube structure -/
def surface_area (c : ModifiedCube) : Nat :=
  sorry

/-- The specific cube structure from the problem -/
def problem_cube : ModifiedCube :=
  { initial_size := 12
  , small_cube_size := 3
  , small_cube_count := 64
  , center_face_removed := true
  , center_small_cube_removed := true
  , small_cube_faces_removed := true }

/-- Theorem stating that the surface area of the problem_cube is 4344 -/
theorem problem_cube_surface_area :
  surface_area problem_cube = 4344 :=
sorry

end NUMINAMATH_CALUDE_problem_cube_surface_area_l1047_104722


namespace NUMINAMATH_CALUDE_lemonade_sales_difference_l1047_104705

/-- Calculates the total difference in lemonade sales between siblings --/
def total_lemonade_sales_difference (
  stanley_cups_per_hour : ℕ)
  (stanley_price_per_cup : ℚ)
  (carl_cups_per_hour : ℕ)
  (carl_price_per_cup : ℚ)
  (lucy_cups_per_hour : ℕ)
  (lucy_price_per_cup : ℚ)
  (hours : ℕ) : ℚ :=
  let stanley_total := (stanley_cups_per_hour * hours : ℚ) * stanley_price_per_cup
  let carl_total := (carl_cups_per_hour * hours : ℚ) * carl_price_per_cup
  let lucy_total := (lucy_cups_per_hour * hours : ℚ) * lucy_price_per_cup
  |carl_total - stanley_total| + |lucy_total - stanley_total| + |carl_total - lucy_total|

theorem lemonade_sales_difference :
  total_lemonade_sales_difference 4 (3/2) 7 (13/10) 5 (9/5) 3 = 93/5 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_sales_difference_l1047_104705


namespace NUMINAMATH_CALUDE_prob_at_least_one_value_l1047_104726

/-- The probability of event A occurring -/
def prob_A : ℝ := 0.4

/-- The probability of event B occurring -/
def prob_B : ℝ := 0.5

/-- Events A and B are independent -/
axiom events_independent : True

/-- The probability of at least one of the events A or B occurring -/
def prob_at_least_one : ℝ := 1 - (1 - prob_A) * (1 - prob_B)

/-- Theorem: The probability of at least one of the events A or B occurring is 0.7 -/
theorem prob_at_least_one_value : prob_at_least_one = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_value_l1047_104726


namespace NUMINAMATH_CALUDE_z_sum_zero_implies_x_squared_minus_y_squared_eq_neg_three_z_times_one_plus_i_purely_imaginary_implies_modulus_eq_two_sqrt_two_l1047_104741

-- Define complex numbers z₁ and z₂
def z₁ (x : ℝ) : ℂ := (2 * x + 1) + 2 * Complex.I
def z₂ (x y : ℝ) : ℂ := -x - y * Complex.I

-- Theorem 1
theorem z_sum_zero_implies_x_squared_minus_y_squared_eq_neg_three
  (x y : ℝ) (h : z₁ x + z₂ x y = 0) :
  x^2 - y^2 = -3 := by sorry

-- Theorem 2
theorem z_times_one_plus_i_purely_imaginary_implies_modulus_eq_two_sqrt_two
  (x : ℝ) (h : (Complex.I + 1) * z₁ x = Complex.I * (Complex.im ((Complex.I + 1) * z₁ x))) :
  Complex.abs (z₁ x) = 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_z_sum_zero_implies_x_squared_minus_y_squared_eq_neg_three_z_times_one_plus_i_purely_imaginary_implies_modulus_eq_two_sqrt_two_l1047_104741


namespace NUMINAMATH_CALUDE_range_of_m_l1047_104719

theorem range_of_m (a b m : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b - a * b = 0)
  (h_log : ∀ a b, 0 < a → 0 < b → a + b - a * b = 0 → Real.log (m ^ 2 / (a + b)) ≤ 0) :
  -2 ≤ m ∧ m ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1047_104719


namespace NUMINAMATH_CALUDE_passing_percentage_is_fifty_l1047_104783

def student_score : ℕ := 200
def failed_by : ℕ := 20
def max_marks : ℕ := 440

def passing_score : ℕ := student_score + failed_by

def passing_percentage : ℚ := (passing_score : ℚ) / (max_marks : ℚ) * 100

theorem passing_percentage_is_fifty : passing_percentage = 50 := by
  sorry

end NUMINAMATH_CALUDE_passing_percentage_is_fifty_l1047_104783


namespace NUMINAMATH_CALUDE_no_two_digit_multiple_of_3_5_7_l1047_104712

theorem no_two_digit_multiple_of_3_5_7 : 
  ∀ n : ℕ, 10 ≤ n ∧ n ≤ 99 → ¬(3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n) := by
  sorry

end NUMINAMATH_CALUDE_no_two_digit_multiple_of_3_5_7_l1047_104712


namespace NUMINAMATH_CALUDE_boy_walking_time_l1047_104716

/-- Given a boy who walks at 4/3 of his usual rate and arrives at school 4 minutes early,
    his usual time to reach school is 16 minutes. -/
theorem boy_walking_time (usual_rate : ℝ) (usual_time : ℝ) 
    (h1 : usual_rate > 0) 
    (h2 : usual_time > 0) 
    (h3 : usual_rate * usual_time = (4/3 * usual_rate) * (usual_time - 4)) : 
  usual_time = 16 := by
sorry

end NUMINAMATH_CALUDE_boy_walking_time_l1047_104716


namespace NUMINAMATH_CALUDE_fourth_root_16_times_sixth_root_9_l1047_104750

theorem fourth_root_16_times_sixth_root_9 : 
  (16 : ℝ) ^ (1/4) * (9 : ℝ) ^ (1/6) = 2 * (3 : ℝ) ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_16_times_sixth_root_9_l1047_104750


namespace NUMINAMATH_CALUDE_inequality_solution_range_l1047_104718

theorem inequality_solution_range (k : ℝ) : 
  (1 : ℝ) ∈ {x : ℝ | k^2 * x^2 - 6*k*x + 8 ≥ 0} ↔ k ≥ 4 ∨ k ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l1047_104718


namespace NUMINAMATH_CALUDE_rocky_knockouts_l1047_104769

theorem rocky_knockouts (total_fights : ℕ) (knockout_percentage : ℚ) (first_round_percentage : ℚ) :
  total_fights = 190 →
  knockout_percentage = 1/2 →
  first_round_percentage = 1/5 →
  (↑total_fights * knockout_percentage * first_round_percentage : ℚ) = 19 := by
sorry

end NUMINAMATH_CALUDE_rocky_knockouts_l1047_104769


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1047_104749

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt (5 * x + 9) = 12 → x = 27 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1047_104749


namespace NUMINAMATH_CALUDE_min_value_problem_l1047_104734

theorem min_value_problem (a : ℝ) (h_a : a > 0) :
  (∃ x y : ℝ, x ≥ 1 ∧ x + y ≤ 3 ∧ y ≥ a * (x - 3) ∧
    (∀ x' y' : ℝ, x' ≥ 1 → x' + y' ≤ 3 → y' ≥ a * (x' - 3) → 2 * x' + y' ≥ 2 * x + y) ∧
    2 * x + y = 1) →
  a = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_problem_l1047_104734


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l1047_104788

open Real

theorem trigonometric_equation_solution (x : ℝ) :
  (∀ K : ℤ, x ≠ π * K / 3) →
  (cos x)^2 = (sin (2 * x))^2 + cos (3 * x) / sin (3 * x) →
  (∃ n : ℤ, x = π / 2 + π * n) ∨ (∃ k : ℤ, x = π / 6 + π * k) ∨ (∃ k : ℤ, x = -π / 6 + π * k) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l1047_104788


namespace NUMINAMATH_CALUDE_emilys_spending_l1047_104714

/-- Emily's spending problem -/
theorem emilys_spending (X : ℝ) : 
  X + 2 * X + 3 * X = 120 → X = 20 := by
  sorry

end NUMINAMATH_CALUDE_emilys_spending_l1047_104714


namespace NUMINAMATH_CALUDE_collinear_points_a_value_l1047_104709

-- Define the points
def A : ℝ × ℝ := (0, 2)
def B : ℝ × ℝ := (3, -1)
def C : ℝ → ℝ × ℝ := λ a => (a, 0)

-- Define collinearity
def collinear (p q r : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, r.1 - p.1 = t * (q.1 - p.1) ∧ r.2 - p.2 = t * (q.2 - p.2)

-- Theorem statement
theorem collinear_points_a_value :
  collinear A B (C a) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_a_value_l1047_104709


namespace NUMINAMATH_CALUDE_problem_statement_l1047_104766

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - a / 2 * x^2

def l (k : ℤ) (x : ℝ) : ℝ := (k - 2 : ℝ) * x - k + 1

theorem problem_statement :
  (∀ a : ℝ, (∃ x₀ : ℝ, x₀ ∈ Set.Icc (Real.exp 1) (Real.exp 2) ∧ f a x₀ > 0) →
    a < 2 / Real.exp 1) ∧
  (∃ k : ℤ, k = 4 ∧
    ∀ k' : ℤ, (∀ x : ℝ, x > 1 → f 0 x > l k' x) → k' ≤ k) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l1047_104766


namespace NUMINAMATH_CALUDE_function_bounds_l1047_104742

-- Define the function f
def f (x y z : ℝ) : ℝ := 7*x + 5*y - 2*z

-- State the theorem
theorem function_bounds (x y z : ℝ) 
  (h1 : -1 ≤ 2*x + y - z ∧ 2*x + y - z ≤ 8)
  (h2 : 2 ≤ x - y + z ∧ x - y + z ≤ 9)
  (h3 : -3 ≤ x + 2*y - z ∧ x + 2*y - z ≤ 7) :
  -6 ≤ f x y z ∧ f x y z ≤ 47 := by
  sorry

end NUMINAMATH_CALUDE_function_bounds_l1047_104742


namespace NUMINAMATH_CALUDE_no_solution_for_equation_l1047_104793

theorem no_solution_for_equation : ¬∃ (x : ℝ), (12 / (x^2 - 9) - 2 / (x - 3) = 1 / (x + 3)) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_equation_l1047_104793


namespace NUMINAMATH_CALUDE_equation_solutions_l1047_104721

theorem equation_solutions :
  (∀ x : ℝ, x^2 - 2*x - 8 = 0 ↔ x = -2 ∨ x = 4) ∧
  (∀ x : ℝ, (x + 1)^2 = 4*x^2 ↔ x = -1/3 ∨ x = 1) := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l1047_104721


namespace NUMINAMATH_CALUDE_min_distance_scaled_circle_to_line_l1047_104743

/-- The minimum distance from a point on the scaled circle to a line -/
theorem min_distance_scaled_circle_to_line :
  let C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}
  let l : Set (ℝ × ℝ) := {p | p.1 + Real.sqrt 3 * p.2 - 6 = 0}
  let C' : Set (ℝ × ℝ) := {p | (p.1^2 / 9) + p.2^2 = 1}
  ∃ (d : ℝ), d = 3 - Real.sqrt 3 ∧ 
    ∀ (p : ℝ × ℝ), p ∈ C' → 
      ∀ (q : ℝ × ℝ), q ∈ l → 
        d ≤ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_scaled_circle_to_line_l1047_104743


namespace NUMINAMATH_CALUDE_loan_duration_l1047_104703

theorem loan_duration (principal_B principal_C interest_rate total_interest : ℚ) 
  (duration_C : ℕ) : 
  principal_B = 5000 →
  principal_C = 3000 →
  duration_C = 4 →
  interest_rate = 15 / 100 →
  total_interest = 3300 →
  principal_B * interest_rate * (duration_B : ℚ) + principal_C * interest_rate * (duration_C : ℚ) = total_interest →
  duration_B = 2 := by
  sorry

#check loan_duration

end NUMINAMATH_CALUDE_loan_duration_l1047_104703


namespace NUMINAMATH_CALUDE_inverse_proportion_inequality_l1047_104713

theorem inverse_proportion_inequality (k : ℝ) (y₁ y₂ y₃ : ℝ) :
  k < 0 →
  y₁ = k / (-3) →
  y₂ = k / (-2) →
  y₃ = k / 3 →
  y₃ < y₁ ∧ y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_inequality_l1047_104713


namespace NUMINAMATH_CALUDE_parallel_segments_length_l1047_104700

/-- Given a quadrilateral ABYZ where AB is parallel to YZ, this theorem proves
    that if AZ = 54, BQ = 18, and QY = 36, then QZ = 36. -/
theorem parallel_segments_length (A B Y Z Q : ℝ × ℝ) : 
  (∃ k : ℝ, k ≠ 0 ∧ B - A = k • (Z - Y)) →  -- AB parallel to YZ
  dist A Z = 54 →
  dist B Q = 18 →
  dist Q Y = 36 →
  dist Q Z = 36 := by
sorry


end NUMINAMATH_CALUDE_parallel_segments_length_l1047_104700


namespace NUMINAMATH_CALUDE_cylinder_volume_formula_l1047_104711

/-- Given a cylinder and a plane passing through its element, prove the formula for the cylinder's volume. -/
theorem cylinder_volume_formula (l α β : ℝ) (h_α_acute : 0 < α ∧ α < π / 2) (h_β_acute : 0 < β ∧ β < π / 2) :
  ∃ V : ℝ, V = (π * l^3 * Real.sin (2 * β) * Real.cos β) / (8 * (Real.cos α)^2) :=
by sorry

end NUMINAMATH_CALUDE_cylinder_volume_formula_l1047_104711


namespace NUMINAMATH_CALUDE_product_of_two_greatest_unattainable_scores_l1047_104764

/-- A score is attainable if it can be expressed as a non-negative integer combination of 19, 9, and 8. -/
def IsAttainable (n : ℕ) : Prop :=
  ∃ a b c : ℕ, n = 19 * a + 9 * b + 8 * c

/-- The set of all attainable scores. -/
def AttainableScores : Set ℕ :=
  {n : ℕ | IsAttainable n}

/-- The set of all unattainable scores. -/
def UnattainableScores : Set ℕ :=
  {n : ℕ | ¬IsAttainable n}

/-- The two greatest unattainable scores. -/
def TwoGreatestUnattainableScores : Fin 2 → ℕ :=
  fun i => if i = 0 then 39 else 31

theorem product_of_two_greatest_unattainable_scores :
  (TwoGreatestUnattainableScores 0) * (TwoGreatestUnattainableScores 1) = 1209 ∧
  (∀ n : ℕ, n ∈ UnattainableScores → n ≤ (TwoGreatestUnattainableScores 0)) ∧
  (∀ n : ℕ, n ∈ UnattainableScores ∧ n ≠ (TwoGreatestUnattainableScores 0) → n ≤ (TwoGreatestUnattainableScores 1)) :=
by sorry

end NUMINAMATH_CALUDE_product_of_two_greatest_unattainable_scores_l1047_104764


namespace NUMINAMATH_CALUDE_absolute_value_plus_exponent_zero_l1047_104770

theorem absolute_value_plus_exponent_zero : 
  |(-4 : ℝ)| + (3 - Real.pi)^(0 : ℝ) = 5 := by sorry

end NUMINAMATH_CALUDE_absolute_value_plus_exponent_zero_l1047_104770


namespace NUMINAMATH_CALUDE_infinitely_many_skew_lines_l1047_104785

/-- A line in 3D space -/
structure Line3D where
  -- Define properties of a line

/-- A plane in 3D space -/
structure Plane3D where
  -- Define properties of a plane

/-- Predicate to check if a line intersects a plane -/
def intersects (l : Line3D) (α : Plane3D) : Prop :=
  sorry

/-- Predicate to check if a line is perpendicular to a plane -/
def perpendicular (l : Line3D) (α : Plane3D) : Prop :=
  sorry

/-- Predicate to check if two lines are skew -/
def skew (l1 l2 : Line3D) : Prop :=
  sorry

/-- Predicate to check if a line is within a plane -/
def within_plane (l : Line3D) (α : Plane3D) : Prop :=
  sorry

/-- The main theorem -/
theorem infinitely_many_skew_lines 
  (l : Line3D) (α : Plane3D) 
  (h1 : intersects l α) 
  (h2 : ¬perpendicular l α) :
  ∃ S : Set Line3D, (∀ l' ∈ S, within_plane l' α ∧ skew l l') ∧ Set.Infinite S :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_skew_lines_l1047_104785


namespace NUMINAMATH_CALUDE_x_eq_y_sufficient_not_necessary_l1047_104704

theorem x_eq_y_sufficient_not_necessary :
  (∀ x y : ℝ, x = y → |x| = |y|) ∧
  (∃ x y : ℝ, |x| = |y| ∧ x ≠ y) :=
by sorry

end NUMINAMATH_CALUDE_x_eq_y_sufficient_not_necessary_l1047_104704


namespace NUMINAMATH_CALUDE_fencing_requirement_l1047_104759

theorem fencing_requirement (area : ℝ) (uncovered_side : ℝ) (fencing : ℝ) : 
  area = 880 →
  uncovered_side = 25 →
  fencing = uncovered_side + 2 * (area / uncovered_side) →
  fencing = 95.4 := by
sorry

end NUMINAMATH_CALUDE_fencing_requirement_l1047_104759


namespace NUMINAMATH_CALUDE_greatest_multiple_of_30_l1047_104765

/-- A function that checks if a list of digits represents a valid arrangement
    according to the problem conditions -/
def is_valid_arrangement (digits : List Nat) : Prop :=
  digits.length = 6 ∧
  digits.toFinset = {1, 3, 4, 6, 8, 9} ∧
  (digits.foldl (fun acc d => acc * 10 + d) 0) % 30 = 0

/-- The claim that 986310 is the greatest possible number satisfying the conditions -/
theorem greatest_multiple_of_30 :
  ∀ (digits : List Nat),
    is_valid_arrangement digits →
    (digits.foldl (fun acc d => acc * 10 + d) 0) ≤ 986310 :=
by sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_30_l1047_104765


namespace NUMINAMATH_CALUDE_rain_duration_l1047_104797

theorem rain_duration (total_hours : ℕ) (no_rain_hours : ℕ) 
  (h1 : total_hours = 8) (h2 : no_rain_hours = 6) : 
  total_hours - no_rain_hours = 2 := by
  sorry

end NUMINAMATH_CALUDE_rain_duration_l1047_104797


namespace NUMINAMATH_CALUDE_successive_discounts_equivalence_l1047_104786

theorem successive_discounts_equivalence :
  let discount1 : ℝ := 0.15
  let discount2 : ℝ := 0.10
  let discount3 : ℝ := 0.05
  let equivalent_single_discount : ℝ := 1 - (1 - discount1) * (1 - discount2) * (1 - discount3)
  equivalent_single_discount = 0.27325 :=
by sorry

end NUMINAMATH_CALUDE_successive_discounts_equivalence_l1047_104786


namespace NUMINAMATH_CALUDE_initial_people_count_initial_people_count_proof_l1047_104771

theorem initial_people_count : ℕ → Prop :=
  fun n => 
    (n / 3 : ℚ) / 2 = 15 → n = 90

-- The proof goes here
theorem initial_people_count_proof : initial_people_count 90 := by
  sorry

end NUMINAMATH_CALUDE_initial_people_count_initial_people_count_proof_l1047_104771


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1047_104773

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 > x - 1) ↔ (∃ x : ℝ, x^2 ≤ x - 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1047_104773


namespace NUMINAMATH_CALUDE_complex_square_sum_l1047_104789

theorem complex_square_sum (a b : ℝ) : (1 + Complex.I) ^ 2 = a + b * Complex.I → a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_square_sum_l1047_104789


namespace NUMINAMATH_CALUDE_power_of_fraction_five_sixths_fourth_l1047_104733

theorem power_of_fraction_five_sixths_fourth : (5 / 6 : ℚ) ^ 4 = 625 / 1296 := by
  sorry

end NUMINAMATH_CALUDE_power_of_fraction_five_sixths_fourth_l1047_104733


namespace NUMINAMATH_CALUDE_triangle_perimeter_l1047_104791

/-- An equilateral triangle with inscribed circles and square -/
structure TriangleWithInscriptions where
  /-- Side length of the equilateral triangle -/
  side : ℝ
  /-- Radius of each inscribed circle -/
  circle_radius : ℝ
  /-- Side length of the inscribed square -/
  square_side : ℝ
  /-- The circle radius is 4 -/
  h_circle_radius : circle_radius = 4
  /-- The square side is equal to the triangle side minus twice the diameter of two circles -/
  h_square_side : square_side = side - 4 * circle_radius
  /-- The triangle side is composed of two parts touching the circles and the diameter of two circles -/
  h_side : side = 2 * (circle_radius * Real.sqrt 3) + 2 * circle_radius

/-- The perimeter of the triangle is 24 + 24√3 -/
theorem triangle_perimeter (t : TriangleWithInscriptions) : 
  3 * t.side = 24 + 24 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l1047_104791


namespace NUMINAMATH_CALUDE_hens_count_l1047_104707

/-- Represents the number of hens in the farm -/
def num_hens : ℕ := sorry

/-- Represents the number of cows in the farm -/
def num_cows : ℕ := sorry

/-- The total number of heads in the farm -/
def total_heads : ℕ := 48

/-- The total number of feet in the farm -/
def total_feet : ℕ := 140

/-- Each hen has 1 head and 2 feet -/
def hen_head_feet : ℕ × ℕ := (1, 2)

/-- Each cow has 1 head and 4 feet -/
def cow_head_feet : ℕ × ℕ := (1, 4)

theorem hens_count : num_hens = 26 :=
  by sorry

end NUMINAMATH_CALUDE_hens_count_l1047_104707


namespace NUMINAMATH_CALUDE_sector_shape_area_l1047_104760

theorem sector_shape_area (r : ℝ) (h : r = 12) : 
  let circle_area := π * r^2
  let sector_90 := (90 / 360) * circle_area
  let sector_120 := (120 / 360) * circle_area
  sector_90 + sector_120 = 84 * π := by
sorry

end NUMINAMATH_CALUDE_sector_shape_area_l1047_104760


namespace NUMINAMATH_CALUDE_same_color_prob_eq_half_l1047_104724

/-- The probability of drawing two balls of the same color from an urn -/
def same_color_prob (n : ℕ) : ℚ :=
  (1 / (n + 5))^2 + (4 / (n + 5))^2 + (n / (n + 5))^2

/-- Theorem: The probability of drawing two balls of the same color is 1/2 iff n = 1 or n = 9 -/
theorem same_color_prob_eq_half (n : ℕ) :
  same_color_prob n = 1/2 ↔ n = 1 ∨ n = 9 := by
  sorry

#eval same_color_prob 1  -- Should output 1/2
#eval same_color_prob 9  -- Should output 1/2

end NUMINAMATH_CALUDE_same_color_prob_eq_half_l1047_104724


namespace NUMINAMATH_CALUDE_circle_intersection_theorem_l1047_104740

-- Define the curve C
def C (x y : ℝ) : Prop := x^2 + y^2 - 3*x = 0 ∧ 5/3 < x ∧ x ≤ 3

-- Define the line L
def L (k x y : ℝ) : Prop := y = k*(x - 4)

-- Theorem statement
theorem circle_intersection_theorem :
  -- Part 1: Center of the circle
  (∃! center : ℝ × ℝ, center.1 = 3/2 ∧ center.2 = 0 ∧
    ∀ x y : ℝ, C x y → (x - center.1)^2 + (y - center.2)^2 = (3/2)^2) ∧
  -- Part 2: Intersection conditions
  (∀ k : ℝ, (∃! p : ℝ × ℝ, C p.1 p.2 ∧ L k p.1 p.2) ↔
    k ∈ Set.Icc (-2*Real.sqrt 5/7) (2*Real.sqrt 5/7) ∪ {-3/4, 3/4}) :=
by sorry

end NUMINAMATH_CALUDE_circle_intersection_theorem_l1047_104740


namespace NUMINAMATH_CALUDE_amanda_hourly_rate_l1047_104799

/-- Amanda's cleaning service hourly rate calculation -/
theorem amanda_hourly_rate :
  let monday_hours : ℝ := 7.5
  let tuesday_hours : ℝ := 3
  let thursday_hours : ℝ := 4
  let saturday_hours : ℝ := 6
  let total_hours : ℝ := monday_hours + tuesday_hours + thursday_hours + saturday_hours
  let total_earnings : ℝ := 410
  total_earnings / total_hours = 20 := by
sorry

end NUMINAMATH_CALUDE_amanda_hourly_rate_l1047_104799


namespace NUMINAMATH_CALUDE_quadratic_other_intercept_l1047_104752

/-- A quadratic function with vertex (2, 9) and one x-intercept at (3, 0) has its other x-intercept at x = 1 -/
theorem quadratic_other_intercept (a b c : ℝ) : 
  (∀ x, a * x^2 + b * x + c = 9 - a * (x - 2)^2) →  -- vertex form with vertex (2, 9)
  a * 3^2 + b * 3 + c = 0 →                         -- x-intercept at (3, 0)
  a * 1^2 + b * 1 + c = 0 :=                        -- other x-intercept at (1, 0)
by sorry

end NUMINAMATH_CALUDE_quadratic_other_intercept_l1047_104752


namespace NUMINAMATH_CALUDE_triangle_inequality_range_l1047_104778

/-- The triangle operation on real numbers -/
def triangle (x y : ℝ) : ℝ := x * (2 - y)

/-- Theorem stating the range of m for which (x + m) △ x < 1 holds for all real x -/
theorem triangle_inequality_range (m : ℝ) :
  (∀ x : ℝ, triangle (x + m) x < 1) ↔ m ∈ Set.Ioo (-4 : ℝ) 0 := by sorry

end NUMINAMATH_CALUDE_triangle_inequality_range_l1047_104778


namespace NUMINAMATH_CALUDE_power_of_power_l1047_104736

theorem power_of_power (x : ℝ) : (x^3)^2 = x^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l1047_104736


namespace NUMINAMATH_CALUDE_sin_arctan_x_equals_x_l1047_104757

theorem sin_arctan_x_equals_x (x : ℝ) :
  x > 0 →
  Real.sin (Real.arctan x) = x →
  x^4 = (3 - Real.sqrt 5) / 2 := by
sorry

end NUMINAMATH_CALUDE_sin_arctan_x_equals_x_l1047_104757


namespace NUMINAMATH_CALUDE_fifth_number_21st_row_l1047_104772

/-- The number of odd numbers in the nth row of the triangular arrangement -/
def odd_numbers_in_row (n : ℕ) : ℕ := 2 * n - 1

/-- The sum of odd numbers in the first n rows -/
def sum_odd_numbers (n : ℕ) : ℕ :=
  (odd_numbers_in_row n + 1) * n / 2

/-- The nth positive odd number -/
def nth_odd_number (n : ℕ) : ℕ := 2 * n - 1

theorem fifth_number_21st_row :
  let total_before := sum_odd_numbers 20
  let position := total_before + 5
  nth_odd_number position = 809 := by sorry

end NUMINAMATH_CALUDE_fifth_number_21st_row_l1047_104772


namespace NUMINAMATH_CALUDE_sqrt_x6_plus_x4_l1047_104745

theorem sqrt_x6_plus_x4 (x : ℝ) : Real.sqrt (x^6 + x^4) = x^2 * Real.sqrt (x^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x6_plus_x4_l1047_104745


namespace NUMINAMATH_CALUDE_square_side_difference_sum_l1047_104708

theorem square_side_difference_sum (a b c : ℝ) 
  (ha : a^2 = 25) (hb : b^2 = 81) (hc : c^2 = 64) : 
  (b - a) + (b - c) = 5 := by sorry

end NUMINAMATH_CALUDE_square_side_difference_sum_l1047_104708


namespace NUMINAMATH_CALUDE_exhibition_survey_l1047_104702

/-- The percentage of visitors who liked the first part of the exhibition -/
def first_part_percentage : ℝ := 25

/-- The percentage of visitors who liked the second part of the exhibition -/
def second_part_percentage : ℝ := 40

theorem exhibition_survey (total_visitors : ℝ) (h_total_positive : total_visitors > 0) :
  let visitors_first_part := (first_part_percentage / 100) * total_visitors
  let visitors_second_part := (second_part_percentage / 100) * total_visitors
  (96 / 100 * visitors_first_part = 60 / 100 * visitors_second_part) ∧
  (59 / 100 * total_visitors = total_visitors - (visitors_first_part + visitors_second_part - 96 / 100 * visitors_first_part)) →
  first_part_percentage = 25 := by
sorry


end NUMINAMATH_CALUDE_exhibition_survey_l1047_104702


namespace NUMINAMATH_CALUDE_sunglasses_hat_probability_l1047_104777

theorem sunglasses_hat_probability (total_sunglasses : ℕ) (total_hats : ℕ) 
  (prob_hat_and_sunglasses : ℚ) :
  total_sunglasses = 80 →
  total_hats = 60 →
  prob_hat_and_sunglasses = 1/3 →
  (prob_hat_and_sunglasses * total_hats) / total_sunglasses = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_sunglasses_hat_probability_l1047_104777


namespace NUMINAMATH_CALUDE_unique_solution_l1047_104787

def is_valid_number (a b : ℕ) : Prop :=
  a < 10 ∧ b < 10 ∧
  (18600 + 10 * a + b) % 3 = 2 ∧
  (18600 + 10 * a + b) % 5 = 3 ∧
  (18600 + 10 * a + b) % 11 = 0

theorem unique_solution :
  ∃! (a b : ℕ), is_valid_number a b ∧ a = 2 ∧ b = 3 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l1047_104787


namespace NUMINAMATH_CALUDE_divisibility_problem_l1047_104744

theorem divisibility_problem (a b : ℕ) :
  (∃ k : ℕ, a = k * (b + 1)) ∧
  (∃ m : ℕ, 43 = m * (a + b)) →
  ((a = 22 ∧ b = 21) ∨
   (a = 33 ∧ b = 10) ∨
   (a = 40 ∧ b = 3) ∨
   (a = 42 ∧ b = 1)) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_problem_l1047_104744


namespace NUMINAMATH_CALUDE_triangle_property_l1047_104796

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- State the theorem
theorem triangle_property (t : Triangle) 
  (h : 2 * Real.sin t.B * Real.sin t.C * Real.cos t.A = 1 - Real.cos (2 * t.A)) :
  (t.b^2 + t.c^2) / t.a^2 = 3 ∧ 
  (∀ (t' : Triangle), Real.sin t'.A ≤ Real.sqrt 5 / 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_property_l1047_104796


namespace NUMINAMATH_CALUDE_initial_investment_interest_rate_l1047_104746

/-- Given an initial investment and an additional investment with their respective interest rates,
    proves that the interest rate of the initial investment is 5% when the total annual income
    equals 6% of the entire investment. -/
theorem initial_investment_interest_rate
  (initial_investment : ℝ)
  (additional_investment : ℝ)
  (additional_rate : ℝ)
  (total_rate : ℝ)
  (h1 : initial_investment = 3000)
  (h2 : additional_investment = 1499.9999999999998)
  (h3 : additional_rate = 0.08)
  (h4 : total_rate = 0.06)
  (h5 : ∃ r : ℝ, initial_investment * r + additional_investment * additional_rate =
                 (initial_investment + additional_investment) * total_rate) :
  ∃ r : ℝ, r = 0.05 ∧
    initial_investment * r + additional_investment * additional_rate =
    (initial_investment + additional_investment) * total_rate :=
sorry

end NUMINAMATH_CALUDE_initial_investment_interest_rate_l1047_104746


namespace NUMINAMATH_CALUDE_prime_iff_totient_and_divisor_sum_condition_l1047_104735

/-- Euler's totient function -/
def φ : ℕ → ℕ := sorry

/-- Divisor sum function -/
def σ : ℕ → ℕ := sorry

/-- An integer n ≥ 2 is prime if and only if φ(n) divides (n - 1) and (n + 1) divides σ(n) -/
theorem prime_iff_totient_and_divisor_sum_condition (n : ℕ) (h : n ≥ 2) :
  Nat.Prime n ↔ (φ n ∣ n - 1) ∧ (n + 1 ∣ σ n) := by sorry

end NUMINAMATH_CALUDE_prime_iff_totient_and_divisor_sum_condition_l1047_104735


namespace NUMINAMATH_CALUDE_air_quality_probability_l1047_104798

theorem air_quality_probability (p_one_day : ℝ) (p_two_days : ℝ) 
  (h1 : p_one_day = 0.8) 
  (h2 : p_two_days = 0.6) : 
  p_two_days / p_one_day = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_air_quality_probability_l1047_104798


namespace NUMINAMATH_CALUDE_ratio_problem_l1047_104775

theorem ratio_problem (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 4 / 3)
  (hdb : d / b = 1 / 5) :
  a / c = 75 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l1047_104775


namespace NUMINAMATH_CALUDE_students_playing_both_sports_l1047_104732

theorem students_playing_both_sports (total : ℕ) (football : ℕ) (cricket : ℕ) (neither : ℕ) : 
  total = 420 → football = 325 → cricket = 175 → neither = 50 →
  football + cricket - (total - neither) = 130 := by
sorry

end NUMINAMATH_CALUDE_students_playing_both_sports_l1047_104732


namespace NUMINAMATH_CALUDE_ab_value_l1047_104753

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 29) : a * b = 10 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l1047_104753


namespace NUMINAMATH_CALUDE_prime_sum_divides_cube_diff_l1047_104774

theorem prime_sum_divides_cube_diff (p q : ℕ) : 
  Prime p → Prime q → (p + q) ∣ (p^3 - q^3) → p = q := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_divides_cube_diff_l1047_104774


namespace NUMINAMATH_CALUDE_toaster_sales_l1047_104792

/-- Represents the inverse proportionality between number of customers and toaster cost -/
def inverse_proportional (customers : ℕ) (cost : ℝ) (k : ℝ) : Prop :=
  (customers : ℝ) * cost = k

/-- Proves that if 12 customers buy a $500 toaster, then 8 customers will buy a $750 toaster,
    given the inverse proportionality relationship -/
theorem toaster_sales (k : ℝ) :
  inverse_proportional 12 500 k →
  inverse_proportional 8 750 k :=
by
  sorry

end NUMINAMATH_CALUDE_toaster_sales_l1047_104792


namespace NUMINAMATH_CALUDE_folded_circle_cut_theorem_l1047_104768

/-- Represents a circular paper folded along its diameter. -/
structure FoldedCircle :=
  (diameter : ℝ)

/-- Represents a straight line drawn on the folded circular paper. -/
structure Line :=
  (angle : ℝ)  -- Angle with respect to the diameter

/-- Calculates the number of pieces resulting from cutting a folded circular paper along given lines. -/
def num_pieces (circle : FoldedCircle) (lines : List Line) : ℕ :=
  sorry

/-- Theorem stating the minimum and maximum number of pieces when cutting a folded circular paper with 3 lines. -/
theorem folded_circle_cut_theorem (circle : FoldedCircle) :
  (∃ (lines : List Line), lines.length = 3 ∧ num_pieces circle lines = 4) ∧
  (∃ (lines : List Line), lines.length = 3 ∧ num_pieces circle lines = 7) ∧
  (∀ (lines : List Line), lines.length = 3 → 4 ≤ num_pieces circle lines ∧ num_pieces circle lines ≤ 7) :=
sorry

end NUMINAMATH_CALUDE_folded_circle_cut_theorem_l1047_104768


namespace NUMINAMATH_CALUDE_greatest_integer_in_set_l1047_104790

/-- A set of consecutive even integers -/
def ConsecutiveEvenSet : Type := List Nat

/-- The median of a list of numbers -/
def median (l : List Nat) : Nat :=
  sorry

/-- Check if a list contains only even numbers -/
def allEven (l : List Nat) : Prop :=
  sorry

/-- Check if a list contains consecutive even integers -/
def isConsecutiveEven (l : List Nat) : Prop :=
  sorry

theorem greatest_integer_in_set (s : ConsecutiveEvenSet) 
  (h1 : median s = 150)
  (h2 : s.head! = 140)
  (h3 : allEven s)
  (h4 : isConsecutiveEven s) :
  s.getLast! = 152 :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_in_set_l1047_104790


namespace NUMINAMATH_CALUDE_abs_x_minus_one_l1047_104758

theorem abs_x_minus_one (x : ℚ) (h : |1 - x| = 1 + |x|) : |x - 1| = 1 - x := by
  sorry

end NUMINAMATH_CALUDE_abs_x_minus_one_l1047_104758


namespace NUMINAMATH_CALUDE_slower_time_to_top_l1047_104725

/-- The time taken by the slower of two people to reach the top of a building --/
def time_to_top (stories : ℕ) (run_time : ℕ) (elevator_time : ℕ) (stop_time : ℕ) : ℕ :=
  max
    (stories * run_time)
    (stories * elevator_time + (stories - 1) * stop_time)

/-- Theorem stating that the slower person takes 217 seconds to reach the top floor --/
theorem slower_time_to_top :
  time_to_top 20 10 8 3 = 217 := by
  sorry

end NUMINAMATH_CALUDE_slower_time_to_top_l1047_104725


namespace NUMINAMATH_CALUDE_largest_inexpressible_number_l1047_104782

def is_expressible (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 5 * a + 6 * b

theorem largest_inexpressible_number : 
  (∀ k : ℕ, k > 19 ∧ k ≤ 50 → is_expressible k) ∧
  ¬is_expressible 19 :=
sorry

end NUMINAMATH_CALUDE_largest_inexpressible_number_l1047_104782


namespace NUMINAMATH_CALUDE_unique_prime_double_squares_l1047_104781

theorem unique_prime_double_squares : 
  ∃! (p : ℕ), 
    Prime p ∧ 
    (∃ (x : ℕ), p + 7 = 2 * x^2) ∧ 
    (∃ (y : ℕ), p^2 + 7 = 2 * y^2) ∧ 
    p = 11 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_double_squares_l1047_104781
