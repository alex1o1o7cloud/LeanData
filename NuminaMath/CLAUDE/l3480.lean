import Mathlib

namespace NUMINAMATH_CALUDE_problem_statement_l3480_348086

theorem problem_statement (a b : ℝ) : 
  ({a, b/a, 1} : Set ℝ) = {a^2, a+b, 0} → a^2014 + b^2013 = 1 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3480_348086


namespace NUMINAMATH_CALUDE_vector_decomposition_l3480_348079

/-- Prove that the given vector x can be decomposed in terms of vectors p, q, and r -/
theorem vector_decomposition (x p q r : ℝ × ℝ × ℝ) : 
  x = (15, -20, -1) → 
  p = (0, 2, 1) → 
  q = (0, 1, -1) → 
  r = (5, -3, 2) → 
  x = (-6 : ℝ) • p + (1 : ℝ) • q + (3 : ℝ) • r :=
by sorry

end NUMINAMATH_CALUDE_vector_decomposition_l3480_348079


namespace NUMINAMATH_CALUDE_cos_330_degrees_l3480_348017

theorem cos_330_degrees : Real.cos (330 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_330_degrees_l3480_348017


namespace NUMINAMATH_CALUDE_range_of_a_l3480_348021

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - x + 2 * a

theorem range_of_a (a : ℝ) :
  (∀ y : ℝ, (∃ x : ℝ, f a x = y) ↔ (∃ x : ℝ, f a (f a x) = y)) →
  a ∈ Set.Ioo (1/2) 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3480_348021


namespace NUMINAMATH_CALUDE_sam_hunts_seven_l3480_348055

/-- The number of animals hunted by Sam, Rob, Mark, and Peter in a day -/
def total_animals : ℕ := 21

/-- Sam's hunt count -/
def sam_hunt : ℕ := 7

/-- Rob's hunt count in terms of Sam's -/
def rob_hunt (s : ℕ) : ℚ := s / 2

/-- Mark's hunt count in terms of Sam's -/
def mark_hunt (s : ℕ) : ℚ := (1 / 3) * (s + rob_hunt s)

/-- Peter's hunt count in terms of Sam's -/
def peter_hunt (s : ℕ) : ℚ := 3 * mark_hunt s

/-- Theorem stating that Sam hunts 7 animals given the conditions -/
theorem sam_hunts_seven :
  sam_hunt + rob_hunt sam_hunt + mark_hunt sam_hunt + peter_hunt sam_hunt = total_animals := by
  sorry

#eval sam_hunt

end NUMINAMATH_CALUDE_sam_hunts_seven_l3480_348055


namespace NUMINAMATH_CALUDE_special_offer_cost_l3480_348052

/-- Represents the cost of a T-shirt in pence -/
def TShirtCost : ℕ := 1650

/-- Represents the savings per T-shirt in pence -/
def SavingsPerShirt : ℕ := 550

/-- Represents the number of T-shirts in the offer -/
def NumShirts : ℕ := 3

/-- Represents the number of T-shirts paid for in the offer -/
def PaidShirts : ℕ := 2

theorem special_offer_cost :
  PaidShirts * TShirtCost = 3300 := by sorry

end NUMINAMATH_CALUDE_special_offer_cost_l3480_348052


namespace NUMINAMATH_CALUDE_cube_sum_equals_linear_sum_l3480_348087

theorem cube_sum_equals_linear_sum (a b : ℝ) 
  (h : a / (1 + b) + b / (1 + a) = 1) : 
  a^3 + b^3 = a + b := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_equals_linear_sum_l3480_348087


namespace NUMINAMATH_CALUDE_total_ladybugs_count_l3480_348051

/-- The number of ladybugs with spots -/
def ladybugs_with_spots : ℕ := 12170

/-- The number of ladybugs without spots -/
def ladybugs_without_spots : ℕ := 54912

/-- The total number of ladybugs -/
def total_ladybugs : ℕ := ladybugs_with_spots + ladybugs_without_spots

theorem total_ladybugs_count : total_ladybugs = 67082 := by
  sorry

end NUMINAMATH_CALUDE_total_ladybugs_count_l3480_348051


namespace NUMINAMATH_CALUDE_savings_calculation_l3480_348006

theorem savings_calculation (income expenditure savings : ℕ) : 
  (income : ℚ) / expenditure = 10 / 4 →
  income = 19000 →
  savings = income - expenditure →
  savings = 11400 := by
sorry

end NUMINAMATH_CALUDE_savings_calculation_l3480_348006


namespace NUMINAMATH_CALUDE_system_solution_unique_l3480_348050

theorem system_solution_unique : 
  ∃! (x y : ℝ), x + y = 2 ∧ x + 2*y = 3 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l3480_348050


namespace NUMINAMATH_CALUDE_subcommittee_count_l3480_348074

theorem subcommittee_count (n m k t : ℕ) (h1 : n = 12) (h2 : m = 5) (h3 : k = 5) (h4 : t = 5) :
  (Nat.choose n k) - (Nat.choose (n - t) k) = 771 :=
by sorry

end NUMINAMATH_CALUDE_subcommittee_count_l3480_348074


namespace NUMINAMATH_CALUDE_anns_age_l3480_348007

theorem anns_age (ann barbara : ℕ) : 
  ann + barbara = 62 →  -- Sum of their ages is 62
  ann = 2 * (barbara - (ann - barbara)) →  -- Ann's age relation
  ann = 50 :=
by sorry

end NUMINAMATH_CALUDE_anns_age_l3480_348007


namespace NUMINAMATH_CALUDE_cafeteria_red_apples_l3480_348043

/-- The number of red apples ordered by the cafeteria -/
def red_apples : ℕ := sorry

/-- The number of green apples ordered by the cafeteria -/
def green_apples : ℕ := 17

/-- The number of students who took apples -/
def students_took_apples : ℕ := 10

/-- The number of extra apples left -/
def extra_apples : ℕ := 32

/-- The total number of apples ordered by the cafeteria -/
def total_apples : ℕ := red_apples + green_apples

theorem cafeteria_red_apples :
  red_apples = 25 :=
by sorry

end NUMINAMATH_CALUDE_cafeteria_red_apples_l3480_348043


namespace NUMINAMATH_CALUDE_hot_air_balloon_theorem_l3480_348036

def hot_air_balloon_problem (initial_balloons : ℕ) : ℕ :=
  let after_first_30_min := initial_balloons - initial_balloons / 5
  let after_next_hour := after_first_30_min - (after_first_30_min * 3) / 10
  let durable_balloons := after_next_hour / 10
  let regular_balloons := after_next_hour - durable_balloons
  let blown_up_regular := min regular_balloons (2 * (initial_balloons - after_next_hour))
  durable_balloons

theorem hot_air_balloon_theorem :
  hot_air_balloon_problem 200 = 11 := by
  sorry

end NUMINAMATH_CALUDE_hot_air_balloon_theorem_l3480_348036


namespace NUMINAMATH_CALUDE_dinner_time_l3480_348095

theorem dinner_time (total_time homework_time cleaning_time trash_time dishwasher_time : ℕ)
  (h1 : total_time = 120)
  (h2 : homework_time = 30)
  (h3 : cleaning_time = 30)
  (h4 : trash_time = 5)
  (h5 : dishwasher_time = 10) :
  total_time - (homework_time + cleaning_time + trash_time + dishwasher_time) = 45 := by
  sorry

end NUMINAMATH_CALUDE_dinner_time_l3480_348095


namespace NUMINAMATH_CALUDE_translation_theorem_l3480_348029

def f (x : ℝ) : ℝ := -2 * x^2 + 4 * x + 1

def g (x : ℝ) : ℝ := -2 * (x + 2)^2 + 4 * (x + 2) + 4

theorem translation_theorem :
  ∀ x : ℝ, g x = f (x + 2) + 3 :=
by
  sorry

end NUMINAMATH_CALUDE_translation_theorem_l3480_348029


namespace NUMINAMATH_CALUDE_alexs_class_size_l3480_348035

theorem alexs_class_size :
  ∃! b : ℕ, 100 < b ∧ b < 150 ∧
  (∃ k : ℕ, b = 4 * k - 2) ∧
  (∃ m : ℕ, b = 5 * m - 3) ∧
  (∃ n : ℕ, b = 6 * n - 4) := by
sorry

end NUMINAMATH_CALUDE_alexs_class_size_l3480_348035


namespace NUMINAMATH_CALUDE_square_side_length_average_l3480_348073

theorem square_side_length_average (a b c : ℝ) 
  (ha : a = 36) (hb : b = 64) (hc : c = 144) : 
  (Real.sqrt a + Real.sqrt b + Real.sqrt c) / 3 = 26 / 3 := by
sorry

end NUMINAMATH_CALUDE_square_side_length_average_l3480_348073


namespace NUMINAMATH_CALUDE_book_profit_calculation_l3480_348058

/-- Calculate the overall percent profit for two books with given costs, markups, and discounts -/
theorem book_profit_calculation (cost_a cost_b : ℝ) (markup_a markup_b : ℝ) (discount_a discount_b : ℝ) :
  cost_a = 50 →
  cost_b = 70 →
  markup_a = 0.4 →
  markup_b = 0.6 →
  discount_a = 0.15 →
  discount_b = 0.2 →
  let marked_price_a := cost_a * (1 + markup_a)
  let marked_price_b := cost_b * (1 + markup_b)
  let sale_price_a := marked_price_a * (1 - discount_a)
  let sale_price_b := marked_price_b * (1 - discount_b)
  let total_cost := cost_a + cost_b
  let total_sale_price := sale_price_a + sale_price_b
  let total_profit := total_sale_price - total_cost
  let percent_profit := (total_profit / total_cost) * 100
  percent_profit = 24.25 := by sorry

end NUMINAMATH_CALUDE_book_profit_calculation_l3480_348058


namespace NUMINAMATH_CALUDE_three_parallel_lines_planes_l3480_348060

-- Define a type for lines in 3D space
structure Line3D where
  -- Add necessary fields to represent a line in 3D space
  -- This is a placeholder and may need to be adjusted based on Lean's geometry libraries

-- Define a predicate for parallel lines
def parallel (l1 l2 : Line3D) : Prop :=
  sorry -- Definition of parallel lines

-- Define a predicate for coplanar lines
def coplanar (l1 l2 l3 : Line3D) : Prop :=
  sorry -- Definition of coplanar lines

-- Define a function to count planes through two lines
def count_planes_through_two_lines (l1 l2 : Line3D) : ℕ :=
  sorry -- Definition to count planes through two lines

-- Theorem statement
theorem three_parallel_lines_planes (a b c : Line3D) :
  parallel a b ∧ parallel b c ∧ parallel a c ∧ ¬coplanar a b c →
  (count_planes_through_two_lines a b +
   count_planes_through_two_lines b c +
   count_planes_through_two_lines a c) = 3 :=
by sorry

end NUMINAMATH_CALUDE_three_parallel_lines_planes_l3480_348060


namespace NUMINAMATH_CALUDE_cylindrical_glass_volume_l3480_348047

/-- The volume of a cylindrical glass with specific straw conditions -/
theorem cylindrical_glass_volume : 
  ∀ (h r : ℝ),
  h > 0 → 
  r > 0 →
  h = 8 →
  r = 6 →
  h^2 + r^2 = 10^2 →
  (π : ℝ) = 3.14 →
  π * r^2 * h = 226.08 :=
by sorry

end NUMINAMATH_CALUDE_cylindrical_glass_volume_l3480_348047


namespace NUMINAMATH_CALUDE_largest_valid_number_l3480_348009

def is_valid (n : ℕ) : Prop :=
  (n ≥ 1000000 ∧ n ≤ 9999999) ∧
  ∀ i : ℕ, i ∈ [0, 1, 2, 3, 4] →
    (((n / 10^i) % 1000) % 11 = 0 ∨ ((n / 10^i) % 1000) % 13 = 0)

theorem largest_valid_number :
  is_valid 9884737 ∧ ∀ m : ℕ, is_valid m → m ≤ 9884737 :=
sorry

end NUMINAMATH_CALUDE_largest_valid_number_l3480_348009


namespace NUMINAMATH_CALUDE_power_sums_l3480_348076

variable (x y p q : ℝ)

def sum_condition : Prop := x + y = -p
def product_condition : Prop := x * y = q

theorem power_sums (h1 : sum_condition x y p) (h2 : product_condition x y q) :
  (x^2 + y^2 = p^2 - 2*q) ∧
  (x^3 + y^3 = -p^3 + 3*p*q) ∧
  (x^4 + y^4 = p^4 - 4*p^2*q + 2*q^2) := by
  sorry

end NUMINAMATH_CALUDE_power_sums_l3480_348076


namespace NUMINAMATH_CALUDE_woogle_threshold_l3480_348037

/-- The score for dropping n woogles -/
def drop_score (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The score for eating n woogles -/
def eat_score (n : ℕ) : ℕ := 15 * n

/-- 30 is the smallest positive integer for which dropping woogles scores more than eating them -/
theorem woogle_threshold : ∀ k : ℕ, k < 30 → drop_score k ≤ eat_score k ∧ drop_score 30 > eat_score 30 := by
  sorry

end NUMINAMATH_CALUDE_woogle_threshold_l3480_348037


namespace NUMINAMATH_CALUDE_alberts_expression_l3480_348098

theorem alberts_expression (p q r s t u : ℚ) : 
  p = 2 ∧ q = 3 ∧ r = 4 ∧ s = 5 ∧ t = 6 →
  p - (q - (r - (s * (t + u)))) = p - q - r - s * t + u →
  u = 4/3 := by sorry

end NUMINAMATH_CALUDE_alberts_expression_l3480_348098


namespace NUMINAMATH_CALUDE_scientific_notation_of_41600_l3480_348014

theorem scientific_notation_of_41600 :
  ∃ (a : ℝ) (n : ℤ), 41600 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 4.16 ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_41600_l3480_348014


namespace NUMINAMATH_CALUDE_marble_probability_theorem_l3480_348063

/-- Represents a box containing marbles -/
structure Box where
  total : ℕ
  red : ℕ
  blue : ℕ
  hSum : red + blue = total

/-- The probability of drawing a red marble from a box -/
def redProb (b : Box) : ℚ :=
  b.red / b.total

/-- The probability of drawing a blue marble from a box -/
def blueProb (b : Box) : ℚ :=
  b.blue / b.total

/-- The main theorem -/
theorem marble_probability_theorem
  (box1 box2 : Box)
  (hTotal : box1.total + box2.total = 30)
  (hRedProb : redProb box1 * redProb box2 = 2/3) :
  blueProb box1 * blueProb box2 = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_theorem_l3480_348063


namespace NUMINAMATH_CALUDE_geometric_sequence_proof_l3480_348066

theorem geometric_sequence_proof :
  let a : ℚ := 3
  let r : ℚ := 8 / 27
  let sequence : ℕ → ℚ := λ n => a * r ^ (n - 1)
  (sequence 1 = 3) ∧ 
  (sequence 2 = 8 / 9) ∧ 
  (sequence 3 = 32 / 81) :=
by
  sorry

#check geometric_sequence_proof

end NUMINAMATH_CALUDE_geometric_sequence_proof_l3480_348066


namespace NUMINAMATH_CALUDE_orange_eaters_ratio_l3480_348045

/-- Represents a family gathering with a specific number of people and orange eaters. -/
structure FamilyGathering where
  total_people : ℕ
  orange_eaters : ℕ
  h_orange_eaters : orange_eaters = total_people - 10

/-- The ratio of orange eaters to total people in a family gathering is 1:2. -/
theorem orange_eaters_ratio (gathering : FamilyGathering) 
    (h_total : gathering.total_people = 20) : 
    (gathering.orange_eaters : ℚ) / gathering.total_people = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_orange_eaters_ratio_l3480_348045


namespace NUMINAMATH_CALUDE_coordinates_wrt_origin_l3480_348057

-- Define a point in a 2D Cartesian coordinate system
def Point := ℝ × ℝ

-- Define the given point
def givenPoint : Point := (-2, 3)

-- Theorem stating that the coordinates of the given point with respect to the origin are (-2, 3)
theorem coordinates_wrt_origin (p : Point) (h : p = givenPoint) : p = (-2, 3) := by
  sorry

end NUMINAMATH_CALUDE_coordinates_wrt_origin_l3480_348057


namespace NUMINAMATH_CALUDE_cubic_function_one_zero_l3480_348041

/-- Given a cubic function f(x) = -x^3 - x on the interval [m, n] where f(m) * f(n) < 0,
    f(x) has exactly one zero in the open interval (m, n). -/
theorem cubic_function_one_zero (m n : ℝ) (hm : m < n)
  (f : ℝ → ℝ) (hf : ∀ x, f x = -x^3 - x)
  (h_neg : f m * f n < 0) :
  ∃! x, m < x ∧ x < n ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_cubic_function_one_zero_l3480_348041


namespace NUMINAMATH_CALUDE_coupon1_best_discount_l3480_348096

/-- Represents the discount offered by Coupon 1 -/
def coupon1_discount (x : ℝ) : ℝ := 0.15 * x

/-- Represents the discount offered by Coupon 2 -/
def coupon2_discount : ℝ := 30

/-- Represents the discount offered by Coupon 3 -/
def coupon3_discount (x : ℝ) : ℝ := 0.22 * (x - 150)

/-- Theorem stating the condition for Coupon 1 to offer the greatest discount -/
theorem coupon1_best_discount (x : ℝ) :
  (coupon1_discount x > coupon2_discount ∧ coupon1_discount x > coupon3_discount x) ↔
  (200 < x ∧ x < 471.43) :=
sorry

end NUMINAMATH_CALUDE_coupon1_best_discount_l3480_348096


namespace NUMINAMATH_CALUDE_lcm_of_54_96_120_150_l3480_348093

theorem lcm_of_54_96_120_150 : Nat.lcm 54 (Nat.lcm 96 (Nat.lcm 120 150)) = 21600 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_54_96_120_150_l3480_348093


namespace NUMINAMATH_CALUDE_triangle_altitude_l3480_348030

/-- Given a triangle with area 600 square feet and base length 30 feet,
    prove that its altitude is 40 feet. -/
theorem triangle_altitude (A : ℝ) (b : ℝ) (h : ℝ) 
    (area_eq : A = 600)
    (base_eq : b = 30)
    (area_formula : A = (1/2) * b * h) : h = 40 := by
  sorry

end NUMINAMATH_CALUDE_triangle_altitude_l3480_348030


namespace NUMINAMATH_CALUDE_equation_has_real_roots_l3480_348024

theorem equation_has_real_roots (K : ℝ) : ∃ x : ℝ, x = K^2 * (x - 1) * (x + 2) :=
sorry

end NUMINAMATH_CALUDE_equation_has_real_roots_l3480_348024


namespace NUMINAMATH_CALUDE_school_club_profit_l3480_348089

/-- Calculates the profit for a school club selling cookies -/
def cookie_profit (num_cookies : ℕ) (buy_rate : ℚ) (sell_price : ℚ) (handling_fee : ℚ) : ℚ :=
  let cost := (num_cookies : ℚ) / buy_rate + handling_fee
  let revenue := (num_cookies : ℚ) * sell_price
  revenue - cost

/-- The profit for the school club selling cookies is $190 -/
theorem school_club_profit :
  cookie_profit 1200 3 (1/2) 10 = 190 := by
  sorry

end NUMINAMATH_CALUDE_school_club_profit_l3480_348089


namespace NUMINAMATH_CALUDE_new_car_travel_distance_l3480_348031

/-- The distance traveled by the older car in miles -/
def older_car_distance : ℝ := 150

/-- The percentage increase in distance for the new car -/
def percentage_increase : ℝ := 0.30

/-- The distance traveled by the new car in miles -/
def new_car_distance : ℝ := older_car_distance * (1 + percentage_increase)

theorem new_car_travel_distance : new_car_distance = 195 := by
  sorry

end NUMINAMATH_CALUDE_new_car_travel_distance_l3480_348031


namespace NUMINAMATH_CALUDE_theater_seats_count_l3480_348090

/-- Represents a theater with an arithmetic progression of seats per row -/
structure Theater where
  first_row_seats : ℕ
  seat_increase : ℕ
  last_row_seats : ℕ

/-- Calculates the total number of seats in the theater -/
def total_seats (t : Theater) : ℕ :=
  let n := (t.last_row_seats - t.first_row_seats) / t.seat_increase + 1
  n * (t.first_row_seats + t.last_row_seats) / 2

/-- Theorem stating that a theater with given conditions has 770 seats -/
theorem theater_seats_count :
  ∀ t : Theater,
    t.first_row_seats = 14 →
    t.seat_increase = 2 →
    t.last_row_seats = 56 →
    total_seats t = 770 :=
by
  sorry


end NUMINAMATH_CALUDE_theater_seats_count_l3480_348090


namespace NUMINAMATH_CALUDE_unit_digit_product_l3480_348005

theorem unit_digit_product : (3^68 * 6^59 * 7^71) % 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_unit_digit_product_l3480_348005


namespace NUMINAMATH_CALUDE_triangle_properties_l3480_348075

theorem triangle_properties (A B C : Real) (a b c : Real) :
  -- Given conditions
  (b + b * Real.cos A = Real.sqrt 3 * Real.sin B) →
  (a = Real.sqrt 21) →
  (b = 4) →
  -- Conclusions to prove
  (A = π / 3) ∧
  (1/2 * b * c * Real.sin A = 5 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3480_348075


namespace NUMINAMATH_CALUDE_fruit_problem_solution_l3480_348048

def fruit_problem (cost_A cost_B : ℝ) (weight_diff : ℝ) (total_weight : ℝ) 
  (selling_price_A selling_price_B : ℝ) : Prop :=
  ∃ (weight_A weight_B cost_per_kg_A cost_per_kg_B : ℝ),
    cost_A = weight_A * cost_per_kg_A ∧
    cost_B = weight_B * cost_per_kg_B ∧
    cost_per_kg_B = 1.5 * cost_per_kg_A ∧
    weight_A = weight_B + weight_diff ∧
    (∀ a b, a + b = total_weight ∧ a ≥ 3 * b →
      (13 - cost_per_kg_A) * a + (20 - cost_per_kg_B) * b ≤
      (13 - cost_per_kg_A) * 75 + (20 - cost_per_kg_B) * 25) ∧
    cost_per_kg_A = 10 ∧
    cost_per_kg_B = 15

theorem fruit_problem_solution :
  fruit_problem 300 300 10 100 13 20 :=
sorry

end NUMINAMATH_CALUDE_fruit_problem_solution_l3480_348048


namespace NUMINAMATH_CALUDE_inequality_one_integer_solution_l3480_348034

theorem inequality_one_integer_solution (a : ℝ) :
  (∃! (x : ℤ), 2 * a * (x : ℝ)^2 - 4 * (x : ℝ) < a * (x : ℝ) - 2) ↔ 1 ≤ a ∧ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_one_integer_solution_l3480_348034


namespace NUMINAMATH_CALUDE_limits_zero_l3480_348070

open Real

theorem limits_zero : 
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |n / (10 : ℝ)^n| < ε) ∧ 
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |log n / n| < ε) := by
  sorry

end NUMINAMATH_CALUDE_limits_zero_l3480_348070


namespace NUMINAMATH_CALUDE_largest_of_seven_consecutive_integers_l3480_348026

theorem largest_of_seven_consecutive_integers (a : ℕ) : 
  (∃ (x : ℕ), x > 0 ∧ 
    (x + (x + 1) + (x + 2) + (x + 3) + (x + 4) + (x + 5) + (x + 6) = 3020) ∧
    (∀ (y : ℕ), y > 0 → 
      (y + (y + 1) + (y + 2) + (y + 3) + (y + 4) + (y + 5) + (y + 6) = 3020) → 
      y = x)) →
  a = 434 ∧
  (∃ (x : ℕ), x > 0 ∧ 
    (x + (x + 1) + (x + 2) + (x + 3) + (x + 4) + (x + 5) + a = 3020) ∧
    (∀ (y : ℕ), y > 0 → 
      (y + (y + 1) + (y + 2) + (y + 3) + (y + 4) + (y + 5) + a = 3020) → 
      y = x)) :=
by sorry

end NUMINAMATH_CALUDE_largest_of_seven_consecutive_integers_l3480_348026


namespace NUMINAMATH_CALUDE_star_arrangement_exists_l3480_348010

/-- A type representing a star-like configuration with 11 positions --/
structure StarConfiguration :=
  (positions : Fin 11 → ℕ)

/-- The sum of numbers from 1 to 11 --/
def sum_1_to_11 : ℕ := (11 * 12) / 2

/-- The segments of the star configuration --/
def segments : List (Fin 11 × Fin 11 × Fin 11) := sorry

/-- The condition that all numbers from 1 to 11 are used exactly once --/
def valid_arrangement (config : StarConfiguration) : Prop :=
  (∀ n : Fin 11, ∃ p : Fin 11, config.positions p = n.val + 1) ∧
  (∀ p q : Fin 11, p ≠ q → config.positions p ≠ config.positions q)

/-- The condition that the sum of each segment is 18 --/
def segment_sum_18 (config : StarConfiguration) : Prop :=
  ∀ seg ∈ segments, 
    config.positions seg.1 + config.positions seg.2.1 + config.positions seg.2.2 = 18

/-- The main theorem: there exists a valid arrangement with segment sum 18 --/
theorem star_arrangement_exists : 
  ∃ (config : StarConfiguration), valid_arrangement config ∧ segment_sum_18 config := by
  sorry

end NUMINAMATH_CALUDE_star_arrangement_exists_l3480_348010


namespace NUMINAMATH_CALUDE_escalator_standing_time_l3480_348082

/-- Represents the time it takes to travel an escalator under different conditions -/
def EscalatorTime (normal_time twice_normal_time : ℝ) : Prop :=
  ∃ (x u : ℝ),
    x > 0 ∧ u > 0 ∧
    (u + x) * normal_time = (u + 2*x) * twice_normal_time ∧
    u * (normal_time * 1.5) = (u + x) * normal_time

theorem escalator_standing_time 
  (h : EscalatorTime 40 30) : 
  ∃ (standing_time : ℝ), standing_time = 60 :=
by sorry

end NUMINAMATH_CALUDE_escalator_standing_time_l3480_348082


namespace NUMINAMATH_CALUDE_measurement_error_probability_l3480_348099

/-- The standard deviation of the measurement errors -/
def σ : ℝ := 10

/-- The maximum allowed absolute error -/
def δ : ℝ := 15

/-- The cumulative distribution function of the standard normal distribution -/
noncomputable def Φ : ℝ → ℝ := sorry

/-- The probability that the absolute error is less than δ -/
noncomputable def P (δ : ℝ) (σ : ℝ) : ℝ := 2 * Φ (δ / σ)

theorem measurement_error_probability :
  ∃ ε > 0, |P δ σ - 0.8664| < ε :=
sorry

end NUMINAMATH_CALUDE_measurement_error_probability_l3480_348099


namespace NUMINAMATH_CALUDE_rectangle_cutout_equals_square_area_l3480_348078

theorem rectangle_cutout_equals_square_area : 
  (10 * 7 - 1 * 6 : ℕ) = 8 * 8 := by sorry

end NUMINAMATH_CALUDE_rectangle_cutout_equals_square_area_l3480_348078


namespace NUMINAMATH_CALUDE_basketball_points_per_basket_l3480_348053

theorem basketball_points_per_basket 
  (matthew_points : ℕ) 
  (shawn_points : ℕ) 
  (total_baskets : ℕ) 
  (h1 : matthew_points = 9) 
  (h2 : shawn_points = 6) 
  (h3 : total_baskets = 5) : 
  (matthew_points + shawn_points) / total_baskets = 3 := by
sorry

end NUMINAMATH_CALUDE_basketball_points_per_basket_l3480_348053


namespace NUMINAMATH_CALUDE_buckingham_palace_visitors_l3480_348042

theorem buckingham_palace_visitors (current_day_visitors previous_day_visitors : ℕ) 
  (h1 : current_day_visitors = 661) 
  (h2 : previous_day_visitors = 600) : 
  current_day_visitors - previous_day_visitors = 61 := by
  sorry

end NUMINAMATH_CALUDE_buckingham_palace_visitors_l3480_348042


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_achieved_l3480_348023

theorem min_value_theorem (x : ℝ) (h1 : 0 < x) (h2 : x < 1) :
  (1 / x) + (2 * x / (1 - x)) ≥ 1 + 2 * Real.sqrt 2 :=
by sorry

theorem min_value_achieved (x : ℝ) (h1 : 0 < x) (h2 : x < 1) :
  ∃ x₀, 0 < x₀ ∧ x₀ < 1 ∧
    (1 / x₀) + (2 * x₀ / (1 - x₀)) = 1 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_achieved_l3480_348023


namespace NUMINAMATH_CALUDE_intersection_coordinate_sum_zero_l3480_348002

/-- Two lines in the coordinate plane -/
structure Line where
  slope : ℝ
  intercept : ℝ
  is_x_intercept : Bool

/-- The point of intersection of two lines -/
def intersection (l1 l2 : Line) : ℝ × ℝ := sorry

/-- Theorem: The sum of coordinates of the intersection point is 0 -/
theorem intersection_coordinate_sum_zero :
  let line_a : Line := ⟨-1, 2, true⟩
  let line_b : Line := ⟨5, -10, false⟩
  let (a, b) := intersection line_a line_b
  a + b = 0 := by sorry

end NUMINAMATH_CALUDE_intersection_coordinate_sum_zero_l3480_348002


namespace NUMINAMATH_CALUDE_soap_brand_usage_l3480_348018

/-- Given a survey of households and their soap usage, prove the number using both brands --/
theorem soap_brand_usage (total : ℕ) (neither : ℕ) (only_A : ℕ) (both : ℕ) :
  total = 300 →
  neither = 80 →
  only_A = 60 →
  total = neither + only_A + both + 3 * both →
  both = 40 := by
sorry

end NUMINAMATH_CALUDE_soap_brand_usage_l3480_348018


namespace NUMINAMATH_CALUDE_quadrupled_base_and_exponent_l3480_348027

theorem quadrupled_base_and_exponent (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0) :
  (4 * a)^(4 * b) = a^b * x^(2 * b) → x = 16 * a^(3/2) := by
  sorry

end NUMINAMATH_CALUDE_quadrupled_base_and_exponent_l3480_348027


namespace NUMINAMATH_CALUDE_horse_food_bags_l3480_348069

/-- Calculates the number of food bags needed for horses over a period of time. -/
theorem horse_food_bags 
  (num_horses : ℕ) 
  (feedings_per_day : ℕ) 
  (food_per_feeding : ℕ) 
  (days : ℕ) 
  (bag_weight_in_pounds : ℕ) : 
  num_horses = 25 → 
  feedings_per_day = 2 → 
  food_per_feeding = 20 → 
  days = 60 → 
  bag_weight_in_pounds = 1000 → 
  (num_horses * feedings_per_day * food_per_feeding * days) / bag_weight_in_pounds = 60 := by
  sorry

#check horse_food_bags

end NUMINAMATH_CALUDE_horse_food_bags_l3480_348069


namespace NUMINAMATH_CALUDE_isosceles_triangle_side_lengths_l3480_348004

/-- An isosceles triangle with centroid on the inscribed circle -/
structure IsoscelesTriangleWithCentroidOnIncircle where
  -- The lengths of the sides of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- The triangle is isosceles with sides a and b equal
  isosceles : a = b
  -- The perimeter of the triangle is 60
  perimeter : a + b + c = 60
  -- The centroid lies on the inscribed circle
  centroid_on_incircle : True  -- We represent this condition as always true for simplicity

/-- The theorem stating the side lengths of the triangle -/
theorem isosceles_triangle_side_lengths 
  (t : IsoscelesTriangleWithCentroidOnIncircle) : 
  t.a = 25 ∧ t.b = 25 ∧ t.c = 10 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_side_lengths_l3480_348004


namespace NUMINAMATH_CALUDE_octagon_diagonal_relation_l3480_348033

/-- In a regular octagon, a is the side length, b is the diagonal spanning two sides, and d is the diagonal spanning three sides. -/
structure RegularOctagon where
  a : ℝ
  b : ℝ
  d : ℝ
  a_pos : 0 < a

/-- The relation between side length and diagonals in a regular octagon -/
theorem octagon_diagonal_relation (oct : RegularOctagon) : oct.d^2 = oct.a^2 + oct.b^2 := by
  sorry

end NUMINAMATH_CALUDE_octagon_diagonal_relation_l3480_348033


namespace NUMINAMATH_CALUDE_jaime_score_l3480_348049

theorem jaime_score (n : ℕ) (avg_without : ℚ) (avg_with : ℚ) (jaime_score : ℚ) :
  n = 20 →
  avg_without = 85 →
  avg_with = 86 →
  (n - 1) * avg_without + jaime_score = n * avg_with →
  jaime_score = 105 :=
by
  sorry

end NUMINAMATH_CALUDE_jaime_score_l3480_348049


namespace NUMINAMATH_CALUDE_problem_statement_l3480_348077

theorem problem_statement (a b c : ℝ) (h : a^3 + a*b + a*c < 0) : b^5 - 4*a*c > 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3480_348077


namespace NUMINAMATH_CALUDE_mickeys_jaydens_difference_l3480_348084

theorem mickeys_jaydens_difference (mickey jayden coraline : ℕ) : 
  (∃ d : ℕ, mickey = jayden + d) →
  jayden = coraline - 40 →
  coraline = 80 →
  mickey + jayden + coraline = 180 →
  ∃ d : ℕ, mickey = jayden + d ∧ d = 20 := by
  sorry

end NUMINAMATH_CALUDE_mickeys_jaydens_difference_l3480_348084


namespace NUMINAMATH_CALUDE_gcd_840_1764_l3480_348088

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end NUMINAMATH_CALUDE_gcd_840_1764_l3480_348088


namespace NUMINAMATH_CALUDE_middle_part_of_proportional_division_l3480_348072

theorem middle_part_of_proportional_division (total : ℚ) (a b c : ℚ) 
  (h_total : total = 120)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_prop : a = 2 * b ∧ c = (1/2) * b) : 
  b = 240/7 := by
  sorry

end NUMINAMATH_CALUDE_middle_part_of_proportional_division_l3480_348072


namespace NUMINAMATH_CALUDE_regression_line_intercept_l3480_348068

-- Define the number of data points
def n : ℕ := 8

-- Define the slope of the regression line
def m : ℚ := 1/3

-- Define the sum of x values
def sum_x : ℚ := 3

-- Define the sum of y values
def sum_y : ℚ := 5

-- Define the mean of x values
def mean_x : ℚ := sum_x / n

-- Define the mean of y values
def mean_y : ℚ := sum_y / n

-- Theorem statement
theorem regression_line_intercept :
  ∃ (a : ℚ), mean_y = m * mean_x + a ∧ a = 1/2 := by sorry

end NUMINAMATH_CALUDE_regression_line_intercept_l3480_348068


namespace NUMINAMATH_CALUDE_overlapping_squares_area_l3480_348038

/-- Represents a square sheet of paper -/
structure Square :=
  (side : ℝ)

/-- Represents the rotation of a square -/
inductive Rotation
  | NoRotation
  | Rotation45
  | Rotation90

/-- Represents a stack of rotated squares -/
structure RotatedSquares :=
  (bottom : Square)
  (middle : Square)
  (top : Square)
  (middleRotation : Rotation)
  (topRotation : Rotation)

/-- Calculates the area of the resulting shape formed by overlapping rotated squares -/
def resultingArea (rs : RotatedSquares) : ℝ :=
  sorry

theorem overlapping_squares_area :
  ∀ (rs : RotatedSquares),
    rs.bottom.side = 8 ∧
    rs.middle.side = 8 ∧
    rs.top.side = 8 ∧
    rs.middleRotation = Rotation.Rotation45 ∧
    rs.topRotation = Rotation.Rotation90 →
    resultingArea rs = 192 :=
  sorry

end NUMINAMATH_CALUDE_overlapping_squares_area_l3480_348038


namespace NUMINAMATH_CALUDE_inequality_proof_l3480_348003

theorem inequality_proof (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  (1/2) * (a + b)^2 + (1/4) * (a + b) ≥ a * Real.sqrt b + b * Real.sqrt a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3480_348003


namespace NUMINAMATH_CALUDE_jo_bob_balloon_ride_l3480_348016

/-- The problem of Jo-Bob's hot air balloon ride -/
theorem jo_bob_balloon_ride (rise_rate : ℝ) (fall_rate : ℝ) (second_pull_time : ℝ) 
  (fall_time : ℝ) (max_height : ℝ) :
  rise_rate = 50 →
  fall_rate = 10 →
  second_pull_time = 15 →
  fall_time = 10 →
  max_height = 1400 →
  ∃ (first_pull_time : ℝ),
    first_pull_time * rise_rate - fall_time * fall_rate + second_pull_time * rise_rate = max_height ∧
    first_pull_time = 15 := by
  sorry

#check jo_bob_balloon_ride

end NUMINAMATH_CALUDE_jo_bob_balloon_ride_l3480_348016


namespace NUMINAMATH_CALUDE_arrange_five_books_two_identical_l3480_348028

/-- The number of ways to arrange books on a shelf -/
def arrange_books (total : ℕ) (identical : ℕ) : ℕ :=
  (Nat.factorial total) / (Nat.factorial identical)

/-- Theorem: The number of ways to arrange 5 books, where 2 are identical, is 60 -/
theorem arrange_five_books_two_identical :
  arrange_books 5 2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_arrange_five_books_two_identical_l3480_348028


namespace NUMINAMATH_CALUDE_peter_has_winning_strategy_l3480_348039

/-- Represents the possible moves in the game -/
inductive Move
  | Single : Nat → Nat → Move  -- 1x1
  | HorizontalRect : Nat → Nat → Move  -- 1x2
  | VerticalRect : Nat → Nat → Move  -- 2x1
  | Square : Nat → Nat → Move  -- 2x2

/-- Represents the game state -/
structure GameState where
  board : Matrix (Fin 8) (Fin 8) Bool
  currentPlayer : Bool  -- true for Peter, false for Victor

/-- Checks if a move is valid in the current game state -/
def isValidMove (state : GameState) (move : Move) : Bool :=
  sorry

/-- Applies a move to the current game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  sorry

/-- Checks if the game is over (no valid moves left) -/
def isGameOver (state : GameState) : Bool :=
  sorry

/-- Represents a strategy for playing the game -/
def Strategy := GameState → Move

/-- The symmetry strategy for Peter -/
def symmetryStrategy : Strategy :=
  sorry

/-- Theorem: Peter has a winning strategy -/
theorem peter_has_winning_strategy :
  ∃ (strategy : Strategy),
    ∀ (game : GameState),
      game.currentPlayer = true →  -- Peter's turn
      ¬(isGameOver game) →
      ∃ (move : Move),
        isValidMove game move ∧
        ¬(isGameOver (applyMove game move)) ∧
        ∀ (victor_move : Move),
          isValidMove (applyMove game move) victor_move →
          ¬(isGameOver (applyMove (applyMove game move) victor_move)) →
          ∃ (peter_response : Move),
            isValidMove (applyMove (applyMove game move) victor_move) peter_response ∧
            strategy (applyMove (applyMove game move) victor_move) = peter_response :=
  sorry

end NUMINAMATH_CALUDE_peter_has_winning_strategy_l3480_348039


namespace NUMINAMATH_CALUDE_sum_abc_is_zero_l3480_348065

theorem sum_abc_is_zero (a b c : ℝ) 
  (h1 : (a + b) / c = (b + c) / a) 
  (h2 : (b + c) / a = (a + c) / b) 
  (h3 : b ≠ c) : 
  a + b + c = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_abc_is_zero_l3480_348065


namespace NUMINAMATH_CALUDE_jewels_total_gain_l3480_348015

/-- Represents the problem of calculating Jewel's total gain from selling magazines --/
def jewels_magazines_problem (cheap_magazines : ℕ) (expensive_magazines : ℕ) 
  (cheap_buy_price : ℚ) (expensive_buy_price : ℚ)
  (cheap_sell_price : ℚ) (expensive_sell_price : ℚ)
  (cheap_discount_percent : ℚ) (expensive_discount_percent : ℚ)
  (cheap_discount_on : ℕ) (expensive_discount_on : ℕ) : Prop :=
let total_cost := cheap_magazines * cheap_buy_price + expensive_magazines * expensive_buy_price
let total_sell := cheap_magazines * cheap_sell_price + expensive_magazines * expensive_sell_price
let cheap_discount := cheap_sell_price * cheap_discount_percent
let expensive_discount := expensive_sell_price * expensive_discount_percent
let total_discount := cheap_discount + expensive_discount
let total_gain := total_sell - total_discount - total_cost
total_gain = 5.1875

/-- Theorem stating that Jewel's total gain is $5.1875 under the given conditions --/
theorem jewels_total_gain :
  jewels_magazines_problem 5 5 3 4 3.5 4.75 0.1 0.15 2 4 := by
  sorry

end NUMINAMATH_CALUDE_jewels_total_gain_l3480_348015


namespace NUMINAMATH_CALUDE_certain_instrument_count_l3480_348046

/-- The number of the certain instrument Charlie owns -/
def x : ℕ := sorry

/-- Charlie's flutes -/
def charlie_flutes : ℕ := 1

/-- Charlie's horns -/
def charlie_horns : ℕ := 2

/-- Carli's flutes -/
def carli_flutes : ℕ := 2 * charlie_flutes

/-- Carli's horns -/
def carli_horns : ℕ := charlie_horns / 2

/-- Total number of instruments owned by Charlie and Carli -/
def total_instruments : ℕ := 7

theorem certain_instrument_count : 
  charlie_flutes + charlie_horns + x + carli_flutes + carli_horns = total_instruments ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_certain_instrument_count_l3480_348046


namespace NUMINAMATH_CALUDE_cards_in_same_envelope_probability_l3480_348008

/-- The number of cards -/
def num_cards : ℕ := 6

/-- The number of envelopes -/
def num_envelopes : ℕ := 3

/-- The number of cards per envelope -/
def cards_per_envelope : ℕ := 2

/-- The set of all possible distributions of cards into envelopes -/
def all_distributions : Finset (Fin num_cards → Fin num_envelopes) :=
  sorry

/-- The set of distributions where cards 1 and 2 are in the same envelope -/
def favorable_distributions : Finset (Fin num_cards → Fin num_envelopes) :=
  sorry

/-- The probability of cards 1 and 2 being in the same envelope -/
def prob_same_envelope : ℚ :=
  (favorable_distributions.card : ℚ) / (all_distributions.card : ℚ)

theorem cards_in_same_envelope_probability :
  prob_same_envelope = 1 / 5 :=
sorry

end NUMINAMATH_CALUDE_cards_in_same_envelope_probability_l3480_348008


namespace NUMINAMATH_CALUDE_rectangle_length_equals_two_l3480_348001

theorem rectangle_length_equals_two (square_side : ℝ) (rect_width : ℝ) (rect_length : ℝ) :
  square_side = 4 →
  rect_width = 8 →
  square_side * square_side = rect_width * rect_length →
  rect_length = 2 := by
sorry

end NUMINAMATH_CALUDE_rectangle_length_equals_two_l3480_348001


namespace NUMINAMATH_CALUDE_kenya_peanuts_l3480_348032

theorem kenya_peanuts (jose_peanuts : ℕ) (kenya_additional_peanuts : ℕ) 
  (h1 : jose_peanuts = 85)
  (h2 : kenya_additional_peanuts = 48) :
  jose_peanuts + kenya_additional_peanuts = 133 :=
by sorry

end NUMINAMATH_CALUDE_kenya_peanuts_l3480_348032


namespace NUMINAMATH_CALUDE_vertical_distance_theorem_l3480_348059

def f (x : ℝ) := |x|
def g (x : ℝ) := -x^2 - 4*x - 3

def solution_set : Set ℝ := {(-5 + Real.sqrt 29)/2, (-5 - Real.sqrt 29)/2, (-3 + Real.sqrt 13)/2, (-3 - Real.sqrt 13)/2}

theorem vertical_distance_theorem :
  ∀ x : ℝ, (f x - g x = 4 ∨ g x - f x = 4) ↔ x ∈ solution_set := by sorry

end NUMINAMATH_CALUDE_vertical_distance_theorem_l3480_348059


namespace NUMINAMATH_CALUDE_sqrt_seven_decimal_part_l3480_348054

theorem sqrt_seven_decimal_part (a : ℝ) : 
  (2 < Real.sqrt 7 ∧ Real.sqrt 7 < 3) → 
  (a = Real.sqrt 7 - 2) → 
  ((Real.sqrt 7 + 2) * a = 3) := by
sorry

end NUMINAMATH_CALUDE_sqrt_seven_decimal_part_l3480_348054


namespace NUMINAMATH_CALUDE_geometric_sequence_second_term_l3480_348081

theorem geometric_sequence_second_term (a₁ a₃ : ℝ) (h₁ : a₁ = 180) (h₃ : a₃ = 75 / 32) :
  ∃ b : ℝ, b > 0 ∧ b^2 = 421.875 ∧ ∃ r : ℝ, a₁ * r = b ∧ b * r = a₃ :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_second_term_l3480_348081


namespace NUMINAMATH_CALUDE_highest_power_of_two_dividing_difference_of_sixth_powers_l3480_348083

theorem highest_power_of_two_dividing_difference_of_sixth_powers :
  ∃ k : ℕ, 2^k = (Nat.gcd (15^6 - 9^6) (2^64)) ∧ k = 4 := by
  sorry

end NUMINAMATH_CALUDE_highest_power_of_two_dividing_difference_of_sixth_powers_l3480_348083


namespace NUMINAMATH_CALUDE_brothers_age_difference_l3480_348071

theorem brothers_age_difference (a b : ℕ) : 
  a > 0 → b > 0 → a + b = 60 → 3 * b = 2 * a → a - b = 12 := by
  sorry

end NUMINAMATH_CALUDE_brothers_age_difference_l3480_348071


namespace NUMINAMATH_CALUDE_divisibility_problem_l3480_348020

theorem divisibility_problem (N : ℕ) (h1 : N % 44 = 0) (h2 : N % 30 = 18) : N / 44 = 3 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l3480_348020


namespace NUMINAMATH_CALUDE_symmetric_points_on_parabola_l3480_348013

/-- Given two points on a parabola that are symmetric about a line, prove the value of m -/
theorem symmetric_points_on_parabola (x₁ x₂ y₁ y₂ m : ℝ) : 
  y₁ = 2 * x₁^2 →                   -- A is on the parabola
  y₂ = 2 * x₂^2 →                   -- B is on the parabola
  (y₁ + y₂) / 2 = (x₁ + x₂) / 2 + m →  -- Midpoint of A and B is on y = x + m
  (y₂ - y₁) / (x₂ - x₁) = -1 →      -- Slope of AB is perpendicular to y = x + m
  x₁ * x₂ = -1/2 →                  -- Given condition
  m = 3/2 := by
sorry

end NUMINAMATH_CALUDE_symmetric_points_on_parabola_l3480_348013


namespace NUMINAMATH_CALUDE_parallel_vectors_x_values_l3480_348085

/-- Given two vectors a and b in ℝ², prove that if they are parallel and have the specified components, then x must be 2 or -1. -/
theorem parallel_vectors_x_values (x : ℝ) :
  let a : ℝ × ℝ := (1, x)
  let b : ℝ × ℝ := (x - 1, 2)
  (∃ (k : ℝ), a = k • b) → x = 2 ∨ x = -1 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_values_l3480_348085


namespace NUMINAMATH_CALUDE_tree_height_differences_l3480_348062

def pine_height : ℚ := 14 + 1/4
def birch_height : ℚ := 18 + 1/2
def cedar_height : ℚ := 20 + 5/8

theorem tree_height_differences :
  (cedar_height - pine_height = 6 + 3/8) ∧
  (cedar_height - birch_height = 2 + 1/8) := by
  sorry

end NUMINAMATH_CALUDE_tree_height_differences_l3480_348062


namespace NUMINAMATH_CALUDE_sports_stars_arrangement_l3480_348019

/-- The number of ways to arrange players from multiple teams in a row, where teammates must sit together -/
def arrangement_count (team_sizes : List Nat) : Nat :=
  (Nat.factorial team_sizes.length) * (team_sizes.map Nat.factorial).prod

/-- Theorem: The number of ways to arrange 10 players from 4 teams (with 3, 3, 2, and 2 players respectively) in a row, where teammates must sit together, is 3456 -/
theorem sports_stars_arrangement :
  arrangement_count [3, 3, 2, 2] = 3456 := by
  sorry

#eval arrangement_count [3, 3, 2, 2]

end NUMINAMATH_CALUDE_sports_stars_arrangement_l3480_348019


namespace NUMINAMATH_CALUDE_x_value_l3480_348011

theorem x_value (y : ℝ) (x : ℝ) : 
  y = 125 * (1 + 0.1) → 
  x = y * (1 - 0.1) → 
  x = 123.75 := by
sorry

end NUMINAMATH_CALUDE_x_value_l3480_348011


namespace NUMINAMATH_CALUDE_parallel_lines_distance_l3480_348061

/-- Given a circle intersected by three equally spaced parallel lines creating chords of lengths 36, 36, and 40, the distance between two adjacent parallel lines is 4√19/3 -/
theorem parallel_lines_distance (r : ℝ) (d : ℝ) : 
  (36 * r^2 = 648 + (9/4) * d^2) ∧ 
  (40 * r^2 = 800 + (45/4) * d^2) →
  d = (4 * Real.sqrt 19) / 3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_distance_l3480_348061


namespace NUMINAMATH_CALUDE_correct_distribution_ways_l3480_348064

/-- The number of ways to distribute four distinct balls into two boxes -/
def distribution_ways : ℕ := 10

/-- The number of distinct balls -/
def num_balls : ℕ := 4

/-- The number of boxes -/
def num_boxes : ℕ := 2

/-- The minimum number of balls required in box 1 -/
def min_box1 : ℕ := 1

/-- The minimum number of balls required in box 2 -/
def min_box2 : ℕ := 2

/-- A function that calculates the number of ways to distribute balls -/
def calculate_distribution_ways (n : ℕ) (k : ℕ) (min1 : ℕ) (min2 : ℕ) : ℕ := sorry

/-- Theorem stating that the number of ways to distribute the balls is correct -/
theorem correct_distribution_ways :
  calculate_distribution_ways num_balls num_boxes min_box1 min_box2 = distribution_ways := by sorry

end NUMINAMATH_CALUDE_correct_distribution_ways_l3480_348064


namespace NUMINAMATH_CALUDE_am_gm_for_even_sum_l3480_348080

theorem am_gm_for_even_sum (a b : ℕ) (ha : a > 0) (hb : b > 0) (hsum : Even (a + b)) :
  (a + b : ℝ) / 2 ≥ Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_am_gm_for_even_sum_l3480_348080


namespace NUMINAMATH_CALUDE_compute_expression_l3480_348022

theorem compute_expression : 10 + 8 * (2 - 9)^2 = 402 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l3480_348022


namespace NUMINAMATH_CALUDE_income_comparison_l3480_348044

theorem income_comparison (juan tim mary : ℝ) 
  (h1 : tim = juan * 0.7)
  (h2 : mary = juan * 1.12) :
  (mary - tim) / tim = 0.6 :=
sorry

end NUMINAMATH_CALUDE_income_comparison_l3480_348044


namespace NUMINAMATH_CALUDE_village_lasts_five_weeks_l3480_348056

/-- The number of weeks a village lasts given supernatural predators -/
def village_duration (village_population : ℕ) 
  (lead_vampire_drain : ℕ) (vampire_group_size : ℕ) (vampire_group_drain : ℕ)
  (alpha_werewolf_eat : ℕ) (werewolf_pack_size : ℕ) (werewolf_pack_eat : ℕ)
  (ghost_feed : ℕ) : ℕ :=
  let total_consumed_per_week := 
    lead_vampire_drain + 
    (vampire_group_size * vampire_group_drain) + 
    alpha_werewolf_eat + 
    (werewolf_pack_size * werewolf_pack_eat) + 
    ghost_feed
  village_population / total_consumed_per_week

theorem village_lasts_five_weeks :
  village_duration 200 5 3 5 7 2 5 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_village_lasts_five_weeks_l3480_348056


namespace NUMINAMATH_CALUDE_sum_at_13th_position_l3480_348092

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℕ
  is_permutation : Function.Bijective vertices

/-- The sum of numbers in a specific position across all rotations of a regular polygon -/
def sum_at_position (p : RegularPolygon 100) (pos : ℕ) : ℕ :=
  (Finset.range 100).sum (λ i => p.vertices ((i + pos - 1) % 100 : Fin 100))

/-- The main theorem -/
theorem sum_at_13th_position (p : RegularPolygon 100) 
  (h_vertices : ∀ i : Fin 100, p.vertices i = i.val + 1) : 
  sum_at_position p 13 = 10100 := by
  sorry

end NUMINAMATH_CALUDE_sum_at_13th_position_l3480_348092


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3480_348094

/-- An isosceles triangle with side lengths 6 and 9 has a perimeter of either 21 or 24 -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  (a = 6 ∨ a = 9) →
  (b = 6 ∨ b = 9) →
  (a = b) →
  (c = 6 ∨ c = 9) →
  (a + b + c = 21 ∨ a + b + c = 24) := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3480_348094


namespace NUMINAMATH_CALUDE_no_solution_for_fermat_like_equation_l3480_348000

theorem no_solution_for_fermat_like_equation :
  ∀ (x y z k : ℕ), x < k → y < k → x^k + y^k ≠ z^k := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_fermat_like_equation_l3480_348000


namespace NUMINAMATH_CALUDE_min_cost_22_bottles_l3480_348067

/-- Calculates the minimum cost to buy a given number of bottles -/
def min_cost (single_price : ℚ) (box_price : ℚ) (bottles_needed : ℕ) : ℚ :=
  let box_size := 6
  let full_boxes := bottles_needed / box_size
  let remaining_bottles := bottles_needed % box_size
  full_boxes * box_price + remaining_bottles * single_price

/-- The minimum cost to buy 22 bottles is R$ 56.20 -/
theorem min_cost_22_bottles :
  min_cost (280 / 100) (1500 / 100) 22 = 5620 / 100 := by
  sorry

end NUMINAMATH_CALUDE_min_cost_22_bottles_l3480_348067


namespace NUMINAMATH_CALUDE_prime_1993_equations_l3480_348012

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

theorem prime_1993_equations (h : isPrime 1993) :
  (∃ x y : ℕ, x^2 - y^2 = 1993) ∧
  (¬∃ x y : ℕ, x^3 - y^3 = 1993) ∧
  (¬∃ x y : ℕ, x^4 - y^4 = 1993) :=
by sorry

end NUMINAMATH_CALUDE_prime_1993_equations_l3480_348012


namespace NUMINAMATH_CALUDE_find_k_l3480_348025

theorem find_k : ∃ k : ℝ, ∀ x : ℝ, -x^2 - (k + 7)*x - 8 = -(x - 2)*(x - 4) → k = -13 := by
  sorry

end NUMINAMATH_CALUDE_find_k_l3480_348025


namespace NUMINAMATH_CALUDE_problem_statement_l3480_348097

theorem problem_statement (a b : ℝ) (h : |a + 2| + Real.sqrt (b - 4) = 0) : a / b = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3480_348097


namespace NUMINAMATH_CALUDE_james_beef_pork_ratio_l3480_348040

/-- Proves that the ratio of beef to pork James bought is 2:1 given the problem conditions --/
theorem james_beef_pork_ratio :
  ∀ (beef pork : ℝ) (meals : ℕ),
    beef = 20 →
    meals * 20 = 400 →
    meals * 1.5 = beef + pork →
    beef / pork = 2 := by
  sorry

end NUMINAMATH_CALUDE_james_beef_pork_ratio_l3480_348040


namespace NUMINAMATH_CALUDE_parabola_focal_chord_angle_l3480_348091

/-- Given a parabola y^2 = 2px and a focal chord AB of length 8p, 
    the angle of inclination θ of AB satisfies sin θ = ±1/2 -/
theorem parabola_focal_chord_angle (p : ℝ) (θ : ℝ) : 
  (∀ x y : ℝ, y^2 = 2*p*x) →  -- parabola equation
  (8*p = 2*p / (Real.sin θ)^2) →  -- focal chord length formula
  (Real.sin θ = 1/2 ∨ Real.sin θ = -1/2) :=
sorry

end NUMINAMATH_CALUDE_parabola_focal_chord_angle_l3480_348091
