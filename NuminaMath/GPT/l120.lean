import Mathlib

namespace NUMINAMATH_GPT_find_n_l120_12086

-- Define the arithmetic series sums
def s1 (n : ℕ) : ℕ := (5 * n^2 + 5 * n) / 2
def s2 (n : ℕ) : ℕ := n^2 + n

-- The theorem to be proved
theorem find_n : ∃ n : ℕ, s1 n + s2 n = 156 ∧ n = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l120_12086


namespace NUMINAMATH_GPT_yogurt_cost_l120_12039

-- Define the conditions given in the problem
def total_cost_ice_cream : ℕ := 20 * 6
def spent_difference : ℕ := 118

theorem yogurt_cost (y : ℕ) 
  (h1 : total_cost_ice_cream = 2 * y + spent_difference) : 
  y = 1 :=
  sorry

end NUMINAMATH_GPT_yogurt_cost_l120_12039


namespace NUMINAMATH_GPT_complement_union_l120_12036

def A := { x : ℤ | ∃ k : ℤ, x = 3 * k + 1 }
def B := { x : ℤ | ∃ k : ℤ, x = 3 * k + 2 }
def U := ℤ

theorem complement_union :
  { x : ℤ | x ∉ (A ∪ B) } = { x : ℤ | ∃ k : ℤ, x = 3 * k } := 
by 
  sorry

end NUMINAMATH_GPT_complement_union_l120_12036


namespace NUMINAMATH_GPT_total_wheels_at_station_l120_12064

/--
There are 4 trains at a train station.
Each train has 4 carriages.
Each carriage has 3 rows of wheels.
Each row of wheels has 5 wheels.
The total number of wheels at the train station is 240.
-/
theorem total_wheels_at_station : 
    let number_of_trains := 4
    let carriages_per_train := 4
    let rows_per_carriage := 3
    let wheels_per_row := 5
    number_of_trains * carriages_per_train * rows_per_carriage * wheels_per_row = 240 := 
by
    sorry

end NUMINAMATH_GPT_total_wheels_at_station_l120_12064


namespace NUMINAMATH_GPT_find_d_l120_12045

theorem find_d (d : ℚ) (h : ∀ x : ℚ, 4*x^3 + 17*x^2 + d*x + 28 = 0 → x = -4/3) : d = 155 / 9 :=
sorry

end NUMINAMATH_GPT_find_d_l120_12045


namespace NUMINAMATH_GPT_sequence_difference_l120_12044

theorem sequence_difference (a : ℕ → ℤ) (h : ∀ n, a (n + 1) = a n + n) : a 2017 - a 2016 = 2016 :=
sorry

end NUMINAMATH_GPT_sequence_difference_l120_12044


namespace NUMINAMATH_GPT_at_least_one_zero_of_product_zero_l120_12042

theorem at_least_one_zero_of_product_zero (a b c : ℝ) (h : a * b * c = 0) : a = 0 ∨ b = 0 ∨ c = 0 := by
  sorry

end NUMINAMATH_GPT_at_least_one_zero_of_product_zero_l120_12042


namespace NUMINAMATH_GPT_rectangle_x_satisfy_l120_12043

theorem rectangle_x_satisfy (x : ℝ) (h1 : 3 * x = 3 * x) (h2 : x + 5 = x + 5) (h3 : (3 * x) * (x + 5) = 2 * (3 * x) + 2 * (x + 5)) : x = 1 :=
sorry

end NUMINAMATH_GPT_rectangle_x_satisfy_l120_12043


namespace NUMINAMATH_GPT_abc_sum_l120_12084

theorem abc_sum (a b c : ℤ) 
  (h1 : ∀ x : ℤ, (x + a) * (x + b) = x^2 + 21 * x + 110)
  (h2 : ∀ x : ℤ, (x - b) * (x - c) = x^2 - 19 * x + 88) : 
  a + b + c = 29 := 
by
  sorry

end NUMINAMATH_GPT_abc_sum_l120_12084


namespace NUMINAMATH_GPT_average_speed_return_trip_l120_12072

/--
A train travels from Albany to Syracuse, a distance of 120 miles,
at an average rate of 50 miles per hour. The train then continues
to Rochester, which is 90 miles from Syracuse, before returning
to Albany. On its way to Rochester, the train's average speed is
60 miles per hour. Finally, the train travels back to Albany from
Rochester, with the total travel time of the train, including all
three legs of the journey, being 9 hours and 15 minutes. What was
the average rate of speed of the train on the return trip from
Rochester to Albany?
-/
theorem average_speed_return_trip :
  let dist_Albany_Syracuse := 120 -- miles
  let speed_Albany_Syracuse := 50 -- miles per hour
  let dist_Syracuse_Rochester := 90 -- miles
  let speed_Syracuse_Rochester := 60 -- miles per hour
  let total_travel_time := 9.25 -- hours (9 hours 15 minutes)
  let time_Albany_Syracuse := dist_Albany_Syracuse / speed_Albany_Syracuse
  let time_Syracuse_Rochester := dist_Syracuse_Rochester / speed_Syracuse_Rochester
  let total_time_so_far := time_Albany_Syracuse + time_Syracuse_Rochester
  let time_return_trip := total_travel_time - total_time_so_far
  let dist_return_trip := dist_Albany_Syracuse + dist_Syracuse_Rochester
  let average_speed_return := dist_return_trip / time_return_trip
  average_speed_return = 39.25 :=
by
  -- sorry placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_average_speed_return_trip_l120_12072


namespace NUMINAMATH_GPT_cupboard_cost_price_l120_12005

theorem cupboard_cost_price (C : ℝ) 
  (h1 : ∀ C₀, C = C₀ → C₀ * 0.88 + 1500 = C₀ * 1.12) :
  C = 6250 := by
  sorry

end NUMINAMATH_GPT_cupboard_cost_price_l120_12005


namespace NUMINAMATH_GPT_find_x_l120_12029

theorem find_x (x : ℕ) (a : ℕ) (h₁: a = 450) (h₂: (15^x * 8^3) / 256 = a) : x = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l120_12029


namespace NUMINAMATH_GPT_sin_double_alpha_zero_l120_12074

noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.cos x

theorem sin_double_alpha_zero (α : ℝ) (h : f α = 1) : Real.sin (2 * α) = 0 :=
by 
  -- Proof would go here, but we're using sorry
  sorry

end NUMINAMATH_GPT_sin_double_alpha_zero_l120_12074


namespace NUMINAMATH_GPT_ratio_part_to_whole_l120_12055

/-- One part of one third of two fifth of a number is 17, and 40% of that number is 204. 
Prove that the ratio of the part to the whole number is 1:30. -/
theorem ratio_part_to_whole 
  (N : ℝ)
  (h1 : (1 / 1) * (1 / 3) * (2 / 5) * N = 17) 
  (h2 : 0.40 * N = 204) : 
  17 / N = 1 / 30 :=
  sorry

end NUMINAMATH_GPT_ratio_part_to_whole_l120_12055


namespace NUMINAMATH_GPT_repave_today_l120_12017

theorem repave_today (total_repaved : ℕ) (repaved_before_today : ℕ) (repaved_today : ℕ) :
  total_repaved = 4938 → repaved_before_today = 4133 → repaved_today = total_repaved - repaved_before_today → repaved_today = 805 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3


end NUMINAMATH_GPT_repave_today_l120_12017


namespace NUMINAMATH_GPT_like_terms_exponents_equal_l120_12095

theorem like_terms_exponents_equal (a b : ℤ) :
  (∀ x y : ℝ, 2 * x^a * y^2 = -3 * x^3 * y^(b+3) → a = 3 ∧ b = -1) :=
by
  sorry

end NUMINAMATH_GPT_like_terms_exponents_equal_l120_12095


namespace NUMINAMATH_GPT_valid_numbers_l120_12078

-- Define the conditions for three-digit numbers
def isThreeDigitNumber (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000

-- Define the splitting cases and the required property
def satisfiesFirstCase (n : ℕ) : Prop :=
  ∃ a b c : ℕ, a ≠ 0 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧
  n = 100 * a + 10 * b + c ∧
  3 * ((10 * a + b) * c) = n

def satisfiesSecondCase (n : ℕ) : Prop :=
  ∃ a b c : ℕ, a ≠ 0 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧
  n = 100 * a + 10 * b + c ∧
  3 * (a * (10 * b + c)) = n

-- Define the main proposition
def validThreeDigitNumber (n : ℕ) : Prop :=
  isThreeDigitNumber n ∧ (satisfiesFirstCase n ∨ satisfiesSecondCase n)

-- The theorem statement which we need to prove
theorem valid_numbers : ∀ n : ℕ, validThreeDigitNumber n ↔ n = 150 ∨ n = 240 ∨ n = 735 :=
by
  sorry

end NUMINAMATH_GPT_valid_numbers_l120_12078


namespace NUMINAMATH_GPT_water_left_after_four_hours_l120_12046

def initial_water : ℕ := 40
def water_loss_per_hour : ℕ := 2
def added_water_hour3 : ℕ := 1
def added_water_hour4 : ℕ := 3
def total_hours : ℕ := 4

theorem water_left_after_four_hours :
  initial_water - (water_loss_per_hour * total_hours) + (added_water_hour3 + added_water_hour4) = 36 := by
  sorry

end NUMINAMATH_GPT_water_left_after_four_hours_l120_12046


namespace NUMINAMATH_GPT_coal_removal_date_l120_12025

theorem coal_removal_date (m n : ℝ) (h1 : 0 < m) (h2 : 0 < n)
  (h3 : 25 * m + 9 * n = 0.5)
  (h4 : ∃ z : ℝ,  z * (n + m) = 0.5)
  (h5 : ∀ z : ℝ, z = 12 → (16 + z) * m = (9 + z) * n):
  ∃ t : ℝ, t = 28 := 
by 
{
  sorry
}

end NUMINAMATH_GPT_coal_removal_date_l120_12025


namespace NUMINAMATH_GPT_purple_sequins_each_row_l120_12097

theorem purple_sequins_each_row (x : ℕ) : 
  (6 * 8) + (9 * 6) + (5 * x) = 162 → x = 12 :=
by 
  sorry

end NUMINAMATH_GPT_purple_sequins_each_row_l120_12097


namespace NUMINAMATH_GPT_area_of_given_triangle_is_32_l120_12033

noncomputable def area_of_triangle : ℕ :=
  let A := (-8, 0)
  let B := (0, 8)
  let C := (0, 0)
  1 / 2 * (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2) : ℤ).natAbs

theorem area_of_given_triangle_is_32 : area_of_triangle = 32 := 
  sorry

end NUMINAMATH_GPT_area_of_given_triangle_is_32_l120_12033


namespace NUMINAMATH_GPT_price_per_unit_l120_12026

theorem price_per_unit (x y : ℝ) 
    (h1 : 2 * x + 3 * y = 690) 
    (h2 : x + 4 * y = 720) : 
    x = 120 ∧ y = 150 := 
by 
    sorry

end NUMINAMATH_GPT_price_per_unit_l120_12026


namespace NUMINAMATH_GPT_smallest_x_l120_12003

noncomputable def digitSum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_x (x a : ℕ) (h1 : a = 100 * x + 4950)
  (h2 : digitSum a = 50) :
  x = 99950 :=
by sorry

end NUMINAMATH_GPT_smallest_x_l120_12003


namespace NUMINAMATH_GPT_range_of_a_l120_12053

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, a * x^2 + 2 * a * x - 4 < 0) : -4 < a ∧ a ≤ 0 := 
sorry

end NUMINAMATH_GPT_range_of_a_l120_12053


namespace NUMINAMATH_GPT_min_holiday_days_l120_12080

theorem min_holiday_days 
  (rained_days : ℕ) 
  (sunny_mornings : ℕ)
  (sunny_afternoons : ℕ) 
  (condition1 : rained_days = 7) 
  (condition2 : sunny_mornings = 5) 
  (condition3 : sunny_afternoons = 6) :
  ∃ (days : ℕ), days = 9 :=
by
  -- The specific steps of the proof are omitted as per the instructions
  sorry

end NUMINAMATH_GPT_min_holiday_days_l120_12080


namespace NUMINAMATH_GPT_solve_for_x_l120_12066

theorem solve_for_x (x y : ℚ) (h1 : 3 * x - 2 * y = 8) (h2 : x + 3 * y = 7) : x = 38 / 11 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l120_12066


namespace NUMINAMATH_GPT_circles_intersect_l120_12065

section PositionalRelationshipCircles

-- Define the first circle O1 with center (1, 0) and radius 1
def Circle1 (p : ℝ × ℝ) : Prop := (p.1 - 1)^2 + p.2^2 = 1

-- Define the second circle O2 with center (0, 3) and radius 3
def Circle2 (p : ℝ × ℝ) : Prop := p.1^2 + (p.2 - 3)^2 = 9

-- Prove that the positional relationship between Circle1 and Circle2 is intersecting
theorem circles_intersect : 
  ∃ p : ℝ × ℝ, Circle1 p ∧ Circle2 p :=
sorry

end PositionalRelationshipCircles

end NUMINAMATH_GPT_circles_intersect_l120_12065


namespace NUMINAMATH_GPT_quadratic_equality_l120_12049

theorem quadratic_equality (x : ℝ) 
  (h : 14*x + 5 - 21*x^2 = -2) : 
  6*x^2 - 4*x + 5 = 7 := 
by
  sorry

end NUMINAMATH_GPT_quadratic_equality_l120_12049


namespace NUMINAMATH_GPT_luke_initial_money_l120_12076

def initial_amount (X : ℤ) : Prop :=
  let spent := 11
  let received := 21
  let current_amount := 58
  X - spent + received = current_amount

theorem luke_initial_money : ∃ (X : ℤ), initial_amount X ∧ X = 48 :=
by
  sorry

end NUMINAMATH_GPT_luke_initial_money_l120_12076


namespace NUMINAMATH_GPT_moles_of_BeOH2_l120_12059

-- Definitions based on the given conditions
def balanced_chemical_equation (xBe2C xH2O xBeOH2 xCH4 : ℕ) : Prop :=
  xBe2C = 1 ∧ xH2O = 4 ∧ xBeOH2 = 2 ∧ xCH4 = 1

def initial_conditions (yBe2C yH2O : ℕ) : Prop :=
  yBe2C = 1 ∧ yH2O = 4

-- Lean statement to prove the number of moles of Beryllium hydroxide formed
theorem moles_of_BeOH2 (xBe2C xH2O xBeOH2 xCH4 yBe2C yH2O : ℕ) (h1 : balanced_chemical_equation xBe2C xH2O xBeOH2 xCH4) (h2 : initial_conditions yBe2C yH2O) :
  xBeOH2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_moles_of_BeOH2_l120_12059


namespace NUMINAMATH_GPT_bookstore_purchase_prices_equal_l120_12031

variable (x : ℝ)

theorem bookstore_purchase_prices_equal
  (h1 : 500 > 0)
  (h2 : 700 > 0)
  (h3 : x > 0)
  (h4 : x + 4 > 0)
  (h5 : ∃ p₁ p₂ : ℝ, p₁ = 500 / x ∧ p₂ = 700 / (x + 4) ∧ p₁ = p₂) :
  500 / x = 700 / (x + 4) :=
by
  sorry

end NUMINAMATH_GPT_bookstore_purchase_prices_equal_l120_12031


namespace NUMINAMATH_GPT_middle_school_students_count_l120_12022

def split_equally (m h : ℕ) : Prop := m = h
def percent_middle (M m : ℕ) : Prop := m = M / 5
def percent_high (H h : ℕ) : Prop := h = 3 * H / 10
def total_students (M H : ℕ) : Prop := M + H = 50
def number_of_middle_school_students (M: ℕ) := M

theorem middle_school_students_count (M H m h : ℕ) 
  (hm_eq : split_equally m h) 
  (hm_percent : percent_middle M m) 
  (hh_percent : percent_high H h) 
  (htotal : total_students M H) : 
  number_of_middle_school_students M = 30 :=
by
  sorry

end NUMINAMATH_GPT_middle_school_students_count_l120_12022


namespace NUMINAMATH_GPT_only_integer_solution_l120_12048

theorem only_integer_solution (a b c d : ℤ) (h : a^2 + b^2 = 3 * (c^2 + d^2)) : 
  a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 := 
by
  sorry

end NUMINAMATH_GPT_only_integer_solution_l120_12048


namespace NUMINAMATH_GPT_number_of_pairs_101_l120_12019

theorem number_of_pairs_101 :
  (∃ n : ℕ, (∀ a b : ℕ, (a > 0) → (b > 0) → (a + b = 101) → (b > a) → (n = 50))) :=
sorry

end NUMINAMATH_GPT_number_of_pairs_101_l120_12019


namespace NUMINAMATH_GPT_new_oranges_added_l120_12071
-- Import the necessary library

-- Define the constants and conditions
def initial_oranges : ℕ := 5
def thrown_away : ℕ := 2
def total_oranges_now : ℕ := 31

-- Define new_oranges as the variable we want to prove
def new_oranges (x : ℕ) : Prop := x = 28

-- The theorem to prove how many new oranges were added
theorem new_oranges_added :
  ∃ (x : ℕ), new_oranges x ∧ total_oranges_now = initial_oranges - thrown_away + x :=
by
  sorry

end NUMINAMATH_GPT_new_oranges_added_l120_12071


namespace NUMINAMATH_GPT_valid_domain_of_x_l120_12077

theorem valid_domain_of_x (x : ℝ) : 
  (x + 1 ≥ 0 ∧ x ≠ 0) ↔ (x ≥ -1 ∧ x ≠ 0) :=
by sorry

end NUMINAMATH_GPT_valid_domain_of_x_l120_12077


namespace NUMINAMATH_GPT_discount_percentage_l120_12083

theorem discount_percentage (M C S : ℝ) (hC : C = 0.64 * M) (hS : S = C * 1.28125) :
  ((M - S) / M) * 100 = 18.08 := 
by
  sorry

end NUMINAMATH_GPT_discount_percentage_l120_12083


namespace NUMINAMATH_GPT_determine_n_l120_12085

theorem determine_n (n : ℕ) (h : 3^n = 27 * 81^3 / 9^4) : n = 7 := by
  sorry

end NUMINAMATH_GPT_determine_n_l120_12085


namespace NUMINAMATH_GPT_phoenix_flight_l120_12012

theorem phoenix_flight : ∃ n : ℕ, 3 ^ n > 6560 ∧ ∀ m < n, 3 ^ m ≤ 6560 :=
by sorry

end NUMINAMATH_GPT_phoenix_flight_l120_12012


namespace NUMINAMATH_GPT_divides_or_l120_12023

-- Definitions
variables {m n : ℕ} -- using natural numbers (non-negative integers) for simplicity in Lean

-- Hypothesis: m ∨ n + m ∧ n = m + n
theorem divides_or (h : Nat.lcm m n + Nat.gcd m n = m + n) : m ∣ n ∨ n ∣ m :=
sorry

end NUMINAMATH_GPT_divides_or_l120_12023


namespace NUMINAMATH_GPT_find_a_plus_b_l120_12006

variables (a b c d x : ℝ)

def conditions (a b c d x : ℝ) : Prop :=
  (a + b = x) ∧
  (b + c = 9) ∧
  (c + d = 3) ∧
  (a + d = 5)

theorem find_a_plus_b (a b c d x : ℝ) (h : conditions a b c d x) : a + b = 11 :=
by
  have h1 : a + b = x := h.1
  have h2 : b + c = 9 := h.2.1
  have h3 : c + d = 3 := h.2.2.1
  have h4 : a + d = 5 := h.2.2.2
  sorry

end NUMINAMATH_GPT_find_a_plus_b_l120_12006


namespace NUMINAMATH_GPT_average_is_0_1667X_plus_3_l120_12020

noncomputable def average_of_three_numbers (X Y Z : ℝ) : ℝ := (X + Y + Z) / 3

theorem average_is_0_1667X_plus_3 (X Y Z : ℝ) 
  (h1 : 2001 * Z - 4002 * X = 8008) 
  (h2 : 2001 * Y + 5005 * X = 10010) : 
  average_of_three_numbers X Y Z = 0.1667 * X + 3 := 
sorry

end NUMINAMATH_GPT_average_is_0_1667X_plus_3_l120_12020


namespace NUMINAMATH_GPT_tangent_line_circle_l120_12057

theorem tangent_line_circle (m : ℝ) (h : ∀ x y : ℝ,  (x + y + m = 0) → (x^2 + y^2 = m) → m = 2) : m = 2 :=
sorry

end NUMINAMATH_GPT_tangent_line_circle_l120_12057


namespace NUMINAMATH_GPT_max_mow_time_l120_12091

-- Define the conditions
def timeToMow (x : ℕ) : Prop := 
  let timeToFertilize := 2 * x
  x + timeToFertilize = 120

-- State the theorem
theorem max_mow_time (x : ℕ) (h : timeToMow x) : x = 40 := by
  sorry

end NUMINAMATH_GPT_max_mow_time_l120_12091


namespace NUMINAMATH_GPT_stock_worth_is_100_l120_12052

-- Define the number of puppies and kittens
def num_puppies : ℕ := 2
def num_kittens : ℕ := 4

-- Define the cost per puppy and kitten
def cost_per_puppy : ℕ := 20
def cost_per_kitten : ℕ := 15

-- Define the total stock worth function
def stock_worth (num_puppies num_kittens cost_per_puppy cost_per_kitten : ℕ) : ℕ :=
  (num_puppies * cost_per_puppy) + (num_kittens * cost_per_kitten)

-- The theorem to prove that the stock worth is $100
theorem stock_worth_is_100 :
  stock_worth num_puppies num_kittens cost_per_puppy cost_per_kitten = 100 :=
by
  sorry

end NUMINAMATH_GPT_stock_worth_is_100_l120_12052


namespace NUMINAMATH_GPT_period_of_f_minimum_value_zero_then_a_eq_one_maximum_value_of_f_axis_of_symmetry_l120_12016

noncomputable def f (x a : ℝ) := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x) ^ 2 + a

theorem period_of_f : ∀ a : ℝ, ∀ x : ℝ, f (x + π) a = f x a := 
by sorry

theorem minimum_value_zero_then_a_eq_one : (∀ x : ℝ, f x a ≥ 0) → a = 1 := 
by sorry

theorem maximum_value_of_f : a = 1 → (∀ x : ℝ, f x 1 ≤ 4) :=
by sorry

theorem axis_of_symmetry : a = 1 → ∃ k : ℤ, ∀ x : ℝ, 2 * x + π / 6 = k * π + π / 2 ↔ f x 1 = f 0 1 :=
by sorry

end NUMINAMATH_GPT_period_of_f_minimum_value_zero_then_a_eq_one_maximum_value_of_f_axis_of_symmetry_l120_12016


namespace NUMINAMATH_GPT_find_a_l120_12047

noncomputable def collinear (A B C : ℝ × ℝ) : Prop :=
  (B.2 - A.2) * (C.1 - A.1) = (C.2 - A.2) * (B.1 - A.1)

theorem find_a (a : ℕ) (h : collinear (a, 0) (0, a + 4) (1, 3)) : a = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l120_12047


namespace NUMINAMATH_GPT_car_mass_nearest_pound_l120_12099

def mass_of_car_kg : ℝ := 1500
def kg_to_pounds : ℝ := 0.4536

theorem car_mass_nearest_pound :
  (↑(Int.floor ((mass_of_car_kg / kg_to_pounds) + 0.5))) = 3307 :=
by
  sorry

end NUMINAMATH_GPT_car_mass_nearest_pound_l120_12099


namespace NUMINAMATH_GPT_negation_of_p_l120_12051

open Classical

variable (p : Prop)

theorem negation_of_p (h : ∀ x : ℝ, x^3 + 2 < 0) : 
  ∃ x : ℝ, x^3 + 2 ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_p_l120_12051


namespace NUMINAMATH_GPT_math_problem_l120_12015

variable (f : ℝ → ℝ)

-- Conditions
axiom condition1 : f 1 = 1
axiom condition2 : ∀ x y : ℝ, f (x + y) + f (x - y) = f x * f y

-- Proof goals
theorem math_problem :
  (f 0 = 2) ∧
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ x : ℝ, f (x + 6) = f x) :=
by 
  sorry

end NUMINAMATH_GPT_math_problem_l120_12015


namespace NUMINAMATH_GPT_trailing_zeros_300_factorial_l120_12050

-- Definition to count the number of times a prime factor divides the factorial of n
def count_factors (n : ℕ) (p : ℕ) : ℕ :=
  Nat.div (n / p) 1 + Nat.div (n / p^2) 1 + Nat.div (n / p^3) 1 + Nat.div (n / p^4) 1

-- Theorem stating the number of trailing zeros in 300! is 74
theorem trailing_zeros_300_factorial : count_factors 300 5 = 74 := by
  sorry

end NUMINAMATH_GPT_trailing_zeros_300_factorial_l120_12050


namespace NUMINAMATH_GPT_subset_A_implies_a_subset_B_implies_range_a_l120_12061

variable (a : ℝ)

def A : Set ℝ := {x | x^2 - 2*x - 8 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + a*x + a^2 - 12 = 0}

theorem subset_A_implies_a (h : A ⊆ B a) : a = -2 := 
sorry

theorem subset_B_implies_range_a (h : B a ⊆ A) : a >= 4 ∨ a < -4 ∨ a = -2 := 
sorry

end NUMINAMATH_GPT_subset_A_implies_a_subset_B_implies_range_a_l120_12061


namespace NUMINAMATH_GPT_find_c_l120_12027

   variable {a b c : ℝ}
   
   theorem find_c (h1 : 4 * a - 3 * b + c = 0)
     (h2 : (a - 1)^2 + (b - 1)^2 = 4) :
     c = 9 ∨ c = -11 := 
   by
     sorry
   
end NUMINAMATH_GPT_find_c_l120_12027


namespace NUMINAMATH_GPT_union_sets_l120_12079

def M : Set ℕ := {0, 1, 3}
def N : Set ℕ := {x | ∃ a ∈ M, x = 3 * a}

theorem union_sets : M ∪ N = {0, 1, 3, 9} := by
  sorry

end NUMINAMATH_GPT_union_sets_l120_12079


namespace NUMINAMATH_GPT_estimate_sqrt_interval_l120_12035

theorem estimate_sqrt_interval : 4 < 2 * Real.sqrt 5 ∧ 2 * Real.sqrt 5 < 5 :=
by
  sorry

end NUMINAMATH_GPT_estimate_sqrt_interval_l120_12035


namespace NUMINAMATH_GPT_music_students_count_l120_12010

open Nat

theorem music_students_count (total_students : ℕ) (art_students : ℕ) (both_music_art : ℕ) 
      (neither_music_art : ℕ) (M : ℕ) :
    total_students = 500 →
    art_students = 10 →
    both_music_art = 10 →
    neither_music_art = 470 →
    (total_students - neither_music_art) = 30 →
    (M + (art_students - both_music_art)) = 30 →
    M = 30 :=
by
  intros h_total h_art h_both h_neither h_music_art_total h_music_count
  sorry

end NUMINAMATH_GPT_music_students_count_l120_12010


namespace NUMINAMATH_GPT_find_rate_l120_12004

noncomputable def national_bank_interest_rate (total_income: ℚ) (investment_national: ℚ) (investment_additional: ℚ) (additional_rate: ℚ) (total_investment_rate: ℚ): ℚ :=
  (total_income - (investment_additional * additional_rate)) / investment_national

theorem find_rate (total_income: ℚ) (investment_national: ℚ) (investment_additional: ℚ) (additional_rate: ℚ) (total_investment_rate: ℚ) (correct_rate: ℚ):
  investment_national = 2400 → investment_additional = 600 → additional_rate = 0.10 → total_investment_rate = 0.06 → total_income = total_investment_rate * (investment_national + investment_additional) → correct_rate = 0.05 → national_bank_interest_rate total_income investment_national investment_additional additional_rate total_investment_rate = correct_rate :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end NUMINAMATH_GPT_find_rate_l120_12004


namespace NUMINAMATH_GPT_smaller_two_digit_product_l120_12014

theorem smaller_two_digit_product (a b : ℕ) (h1: 10 ≤ a) (h2: a ≤ 99) (h3: 10 ≤ b) (h4: b ≤ 99) (h_prod: a * b = 4536) : a = 54 ∨ b = 54 :=
by sorry

end NUMINAMATH_GPT_smaller_two_digit_product_l120_12014


namespace NUMINAMATH_GPT_rhombus_new_perimeter_l120_12067

theorem rhombus_new_perimeter (d1 d2 : ℝ) (scale : ℝ) (h_d1 : d1 = 10) (h_d2 : d2 = 24) (h_scale : scale = 0.5) : 
  4 * (scale * (Real.sqrt ((d1/2)^2 + (d2/2)^2))) = 26 := 
by
  sorry

end NUMINAMATH_GPT_rhombus_new_perimeter_l120_12067


namespace NUMINAMATH_GPT_cubic_roots_relations_l120_12008

theorem cubic_roots_relations 
    (a b c d : ℚ) 
    (x1 x2 x3 : ℚ) 
    (h : a ≠ 0)
    (hroots : a * x1^3 + b * x1^2 + c * x1 + d = 0 
      ∧ a * x2^3 + b * x2^2 + c * x2 + d = 0 
      ∧ a * x3^3 + b * x3^2 + c * x3 + d = 0) 
    :
    (x1 + x2 + x3 = -b / a) 
    ∧ (x1 * x2 + x1 * x3 + x2 * x3 = c / a) 
    ∧ (x1 * x2 * x3 = -d / a) := 
sorry

end NUMINAMATH_GPT_cubic_roots_relations_l120_12008


namespace NUMINAMATH_GPT_convex_pentagons_l120_12034

theorem convex_pentagons (P : Finset ℝ) (h : P.card = 15) : 
  (P.card.choose 5) = 3003 := 
by
  sorry

end NUMINAMATH_GPT_convex_pentagons_l120_12034


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l120_12082

theorem sufficient_but_not_necessary (a b : ℝ) : (ab >= 2) -> a^2 + b^2 >= 4 ∧ ∃ a b : ℝ, a^2 + b^2 >= 4 ∧ ab < 2 := by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l120_12082


namespace NUMINAMATH_GPT_correct_solution_l120_12013

theorem correct_solution : 
  ∀ (x y a b : ℚ), (a = 1) → (b = 1 / 2) → 
  (a * x + y = 2) → (2 * x - b * y = 1) → 
  (x = 4 / 5 ∧ y = 6 / 5) := 
by
  intros x y a b ha hb h1 h2
  sorry

end NUMINAMATH_GPT_correct_solution_l120_12013


namespace NUMINAMATH_GPT_bases_with_final_digit_one_in_360_l120_12009

theorem bases_with_final_digit_one_in_360 (b : ℕ) (h : 2 ≤ b ∧ b ≤ 9) : ¬(b ∣ 359) :=
by
  sorry

end NUMINAMATH_GPT_bases_with_final_digit_one_in_360_l120_12009


namespace NUMINAMATH_GPT_problem_statement_l120_12092

theorem problem_statement (x : ℝ) (h₀ : 0 < x) (h₁ : x + 1 / x ≥ 2) (h₂ : x + 4 / x ^ 2 ≥ 3) (h₃ : x + 27 / x ^ 3 ≥ 4) :
  ∀ a : ℝ, (x + a / x ^ 4 ≥ 5) → a = 4 ^ 4 := 
by 
  sorry

end NUMINAMATH_GPT_problem_statement_l120_12092


namespace NUMINAMATH_GPT_prod_eq_one_l120_12002

noncomputable def is_parity_equal (A : Finset ℝ) (a : ℝ) : Prop :=
  (A.filter (fun x => x > a)).card % 2 = (A.filter (fun x => x < 1/a)).card % 2

theorem prod_eq_one
  (A : Finset ℝ)
  (hA : ∀ (a : ℝ), 0 < a → is_parity_equal A a)
  (hA_pos : ∀ x ∈ A, 0 < x) :
  A.prod id = 1 :=
sorry

end NUMINAMATH_GPT_prod_eq_one_l120_12002


namespace NUMINAMATH_GPT_max_f_value_inequality_m_n_l120_12068

section
variable (x : ℝ)

def f (x : ℝ) := abs (x - 1) - 2 * abs (x + 1)

theorem max_f_value : ∃ k, (∀ x : ℝ, f x ≤ k) ∧ (∃ x₀ : ℝ, f x₀ = k) ∧ k = 2 := 
by sorry

theorem inequality_m_n (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : 1 / m + 1 / (2 * n) = 2) :
  m + 2 * n ≥ 2 :=
by sorry

end

end NUMINAMATH_GPT_max_f_value_inequality_m_n_l120_12068


namespace NUMINAMATH_GPT_initial_friends_online_l120_12060

theorem initial_friends_online (F : ℕ) 
  (h1 : 8 + F = 13) 
  (h2 : 6 * F = 30) : 
  F = 5 :=
by
  sorry

end NUMINAMATH_GPT_initial_friends_online_l120_12060


namespace NUMINAMATH_GPT_problem_1_problem_2_l120_12087

def f (x : ℝ) : ℝ := |2 * x - 1|

theorem problem_1 : {x : ℝ | f x > 2} = {x : ℝ | x < -1 / 2 ∨ x > 3 / 2} := sorry

theorem problem_2 (m : ℝ) : (∀ x : ℝ, f x + |2 * (x + 3)| - 4 > m * x) → m ≤ -11 := sorry

end NUMINAMATH_GPT_problem_1_problem_2_l120_12087


namespace NUMINAMATH_GPT_matilda_fathers_chocolate_bars_l120_12088

/-- Matilda had 20 chocolate bars and shared them evenly amongst herself and her 4 sisters.
    When her father got home, he was upset that they did not put aside any chocolates for him.
    They felt bad, so they each gave up half of their chocolate bars for their father.
    Their father then gave 3 chocolate bars to their mother and ate some.
    Matilda's father had 5 chocolate bars left.
    Prove that Matilda's father ate 2 chocolate bars. -/
theorem matilda_fathers_chocolate_bars:
  ∀ (total_chocolates initial_people chocolates_per_person given_to_father chocolates_left chocolates_eaten: ℕ ),
    total_chocolates = 20 →
    initial_people = 5 →
    chocolates_per_person = total_chocolates / initial_people →
    given_to_father = (chocolates_per_person / 2) * initial_people →
    chocolates_left = given_to_father - 3 →
    chocolates_left - 5 = chocolates_eaten →
    chocolates_eaten = 2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_matilda_fathers_chocolate_bars_l120_12088


namespace NUMINAMATH_GPT_price_ratio_l120_12094

-- Definitions based on the provided conditions
variables (x y : ℕ) -- number of ballpoint pens and gel pens respectively
variables (b g T : ℝ) -- price of ballpoint pen, gel pen, and total amount paid respectively

-- The two given conditions
def cond1 (x y : ℕ) (b g T : ℝ) : Prop := 
  (x + y) * g = 4 * (x * b + y * g)

def cond2 (x y : ℕ) (b g T : ℝ) : Prop := 
  (x + y) * b = (x * b + y * g) / 2

-- The goal to prove
theorem price_ratio (x y : ℕ) (b g T : ℝ) (h1 : cond1 x y b g T) (h2 : cond2 x y b g T) : 
  g = 8 * b :=
sorry

end NUMINAMATH_GPT_price_ratio_l120_12094


namespace NUMINAMATH_GPT_cone_volume_l120_12018

theorem cone_volume (r h l : ℝ) (π := Real.pi)
  (slant_height : l = 5)
  (lateral_area : π * r * l = 20 * π) :
  (1 / 3) * π * r^2 * h = 16 * π :=
by
  -- Definitions based on conditions
  let slant_height_definition := slant_height
  let lateral_area_definition := lateral_area
  
  -- Need actual proof steps which are omitted using sorry
  sorry

end NUMINAMATH_GPT_cone_volume_l120_12018


namespace NUMINAMATH_GPT_regular_polygon_sides_l120_12040

theorem regular_polygon_sides (n : ℕ) (h1 : ∃ a : ℝ, a = 120 ∧ ∀ i < n, 120 = a) : n = 6 :=
by
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_l120_12040


namespace NUMINAMATH_GPT_chris_money_before_birthday_l120_12037

variables {x : ℕ} -- Assuming we are working with natural numbers (non-negative integers)

-- Conditions
def grandmother_money : ℕ := 25
def aunt_uncle_money : ℕ := 20
def parents_money : ℕ := 75
def total_money_now : ℕ := 279

-- Question
theorem chris_money_before_birthday : x = total_money_now - (grandmother_money + aunt_uncle_money + parents_money) :=
by
  sorry

end NUMINAMATH_GPT_chris_money_before_birthday_l120_12037


namespace NUMINAMATH_GPT_salad_dressing_vinegar_percentage_l120_12090

-- Define the initial conditions
def percentage_in_vinegar_in_Q : ℝ := 10
def percentage_of_vinegar_in_combined : ℝ := 12
def percentage_of_dressing_P_in_combined : ℝ := 0.10
def percentage_of_dressing_Q_in_combined : ℝ := 0.90
def percentage_of_vinegar_in_P (V : ℝ) : ℝ := V

-- The statement to prove
theorem salad_dressing_vinegar_percentage (V : ℝ) 
  (hQ : percentage_in_vinegar_in_Q = 10)
  (hCombined : percentage_of_vinegar_in_combined = 12)
  (hP_combined : percentage_of_dressing_P_in_combined = 0.10)
  (hQ_combined : percentage_of_dressing_Q_in_combined = 0.90)
  (hV_combined : 0.10 * percentage_of_vinegar_in_P V + 0.90 * percentage_in_vinegar_in_Q = 12) :
  V = 30 :=
by 
  sorry

end NUMINAMATH_GPT_salad_dressing_vinegar_percentage_l120_12090


namespace NUMINAMATH_GPT_vector_dot_product_l120_12081

-- Definitions of the vectors
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-1, 2)

-- Definition of the dot product for 2D vectors
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Main statement to prove
theorem vector_dot_product :
  dot_product (a.1 + b.1, a.2 + b.2) (a.1 - b.1, a.2 - b.2) = 0 :=
by
  sorry

end NUMINAMATH_GPT_vector_dot_product_l120_12081


namespace NUMINAMATH_GPT_find_f_neg2_l120_12062

noncomputable def f (x : ℝ) : ℝ := -2 * (x + 1) + 1

theorem find_f_neg2 : f (-2) = 3 := by
  sorry

end NUMINAMATH_GPT_find_f_neg2_l120_12062


namespace NUMINAMATH_GPT_solution_set_of_inequality_l120_12058

-- We define the inequality condition
def inequality (x : ℝ) : Prop := (x - 3) * (x + 2) < 0

-- We need to state that for all real numbers x, iff x satisfies the inequality,
-- then x must be within the interval (-2, 3).
theorem solution_set_of_inequality :
  ∀ x : ℝ, inequality x ↔ -2 < x ∧ x < 3 :=
by {
   sorry
}

end NUMINAMATH_GPT_solution_set_of_inequality_l120_12058


namespace NUMINAMATH_GPT_Anya_loss_games_l120_12093

noncomputable def game_indices_Anya_lost (Anya Bella Valya Galya Dasha : ℕ) : Set ℕ :=
  {4, 8, 12, 16}

theorem Anya_loss_games
  (Anya Bella Valya Galya Dasha total_games : ℕ)
  (h1 : Anya = 4)
  (h2 : Bella = 6)
  (h3 : Valya = 7)
  (h4 : Galya = 10)
  (h5 : Dasha = 11)
  (h6 : total_games = (Anya + Bella + Valya + Galya + Dasha) / 2)
  (h7 : total_games = 19) :
  game_indices_Anya_lost Anya Bella Valya Galya Dasha = {4, 8, 12, 16} :=
  by sorry

end NUMINAMATH_GPT_Anya_loss_games_l120_12093


namespace NUMINAMATH_GPT_smallest_integral_value_k_l120_12001

-- Define the discriminant of the quadratic equation
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Define the quadratic equation
def quadratic (k : ℝ) (x : ℝ) : ℝ := 3 * x * (k * x - 5) - x^2 + 4

-- Define the condition for the quadratic equation having no real roots
def no_real_roots (k : ℝ) : Prop :=
  let a := 3 * k - 1
  let b := -15
  let c := 4
  discriminant a b c < 0

-- The Lean 4 statement to find the smallest integral value of k such that the quadratic has no real roots
theorem smallest_integral_value_k : ∃ (k : ℤ), no_real_roots k ∧ (∀ (m : ℤ), no_real_roots m → k ≤ m) :=
  sorry

end NUMINAMATH_GPT_smallest_integral_value_k_l120_12001


namespace NUMINAMATH_GPT_juice_packs_in_box_l120_12096

theorem juice_packs_in_box 
  (W_box L_box H_box W_juice_pack L_juice_pack H_juice_pack : ℕ)
  (hW_box : W_box = 24) (hL_box : L_box = 15) (hH_box : H_box = 28)
  (hW_juice_pack : W_juice_pack = 4) (hL_juice_pack : L_juice_pack = 5) (hH_juice_pack : H_juice_pack = 7) : 
  (W_box * L_box * H_box) / (W_juice_pack * L_juice_pack * H_juice_pack) = 72 :=
by
  sorry

end NUMINAMATH_GPT_juice_packs_in_box_l120_12096


namespace NUMINAMATH_GPT_remainder_of_2356912_div_8_l120_12041

theorem remainder_of_2356912_div_8 : 912 % 8 = 0 := 
by 
  sorry

end NUMINAMATH_GPT_remainder_of_2356912_div_8_l120_12041


namespace NUMINAMATH_GPT_z_sum_of_squares_eq_101_l120_12021

open Complex

noncomputable def z_distances_sum_of_squares (z : ℂ) (h : abs (z - (3 + -3 * I)) = 3) : ℝ :=
  abs (z - (1 + 1 * I)) ^ 2 + abs (z - (5 - 5 * I)) ^ 2

theorem z_sum_of_squares_eq_101 (z : ℂ) (h : abs (z - (3 + -3 * I)) = 3) : 
  z_distances_sum_of_squares z h = 101 :=
by
  sorry

end NUMINAMATH_GPT_z_sum_of_squares_eq_101_l120_12021


namespace NUMINAMATH_GPT_express_y_in_terms_of_x_l120_12073

theorem express_y_in_terms_of_x (x y : ℝ) (h : y - 2 * x = 6) : y = 2 * x + 6 :=
by
  sorry

end NUMINAMATH_GPT_express_y_in_terms_of_x_l120_12073


namespace NUMINAMATH_GPT_find_f2_l120_12056

noncomputable def f (a b x : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

theorem find_f2 (a b : ℝ) (h : f a b (-2) = 10) : f a b 2 = -26 :=
by
  sorry

end NUMINAMATH_GPT_find_f2_l120_12056


namespace NUMINAMATH_GPT_number_of_zeros_of_g_is_4_l120_12032

noncomputable def f (x : ℝ) : ℝ := 
  if x < 0 then x + 1/x else Real.log x

noncomputable def g (x : ℝ) : ℝ := 
  f (f x + 2) + 2

theorem number_of_zeros_of_g_is_4 : 
  ∃ S : Finset ℝ, S.card = 4 ∧ ∀ x ∈ S, g x = 0 :=
sorry

end NUMINAMATH_GPT_number_of_zeros_of_g_is_4_l120_12032


namespace NUMINAMATH_GPT_printingTime_l120_12075

def printerSpeed : ℝ := 23
def pauseTime : ℝ := 2
def totalPages : ℝ := 350

theorem printingTime : (totalPages / printerSpeed) + ((totalPages / 50 - 1) * pauseTime) = 27 := by 
  sorry

end NUMINAMATH_GPT_printingTime_l120_12075


namespace NUMINAMATH_GPT_sum_of_fractions_l120_12054

theorem sum_of_fractions :
  (1 / (3 * 4) + 1 / (4 * 5) + 1 / (5 * 6) + 1 / (6 * 7) + 1 / (7 * 8) + 1 / (8 * 9)) = (2 / 9) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_fractions_l120_12054


namespace NUMINAMATH_GPT_ratio_white_to_remaining_l120_12024

def total_beans : ℕ := 572

def red_beans (total : ℕ) : ℕ := total / 4

def remaining_beans_after_red (total : ℕ) (red : ℕ) : ℕ := total - red

def green_beans : ℕ := 143

def remaining_beans_after_green (remaining : ℕ) (green : ℕ) : ℕ := remaining - green

def white_beans (remaining : ℕ) : ℕ := remaining / 2

theorem ratio_white_to_remaining (total : ℕ) (red : ℕ) (remaining : ℕ) (green : ℕ) (white : ℕ) 
  (H_total : total = 572)
  (H_red : red = red_beans total)
  (H_remaining : remaining = remaining_beans_after_red total red)
  (H_green : green = 143)
  (H_remaining_after_green : remaining_beans_after_green remaining green = white)
  (H_white : white = white_beans remaining) :
  (white : ℚ) / (remaining : ℚ) = (1 : ℚ) / 2 := 
by sorry

end NUMINAMATH_GPT_ratio_white_to_remaining_l120_12024


namespace NUMINAMATH_GPT_total_players_l120_12089

theorem total_players (kabaddi : ℕ) (only_kho_kho : ℕ) (both_games : ℕ) 
  (h_kabaddi : kabaddi = 10) (h_only_kho_kho : only_kho_kho = 15) 
  (h_both_games : both_games = 5) : (kabaddi - both_games) + only_kho_kho + both_games = 25 :=
by
  sorry

end NUMINAMATH_GPT_total_players_l120_12089


namespace NUMINAMATH_GPT_Morgan_first_SAT_score_l120_12098

variable (S : ℝ) -- Morgan's first SAT score
variable (improved_score : ℝ := 1100) -- Improved score on second attempt
variable (improvement_rate : ℝ := 0.10) -- Improvement rate

theorem Morgan_first_SAT_score:
  improved_score = S * (1 + improvement_rate) → S = 1000 := 
by 
  sorry

end NUMINAMATH_GPT_Morgan_first_SAT_score_l120_12098


namespace NUMINAMATH_GPT_multiple_of_regular_rate_is_1_5_l120_12028

-- Definitions
def hourly_rate := 5.50
def regular_hours := 7.5
def total_hours := 10.5
def total_earnings := 66.0
def excess_hours := total_hours - regular_hours
def regular_earnings := regular_hours * hourly_rate
def excess_earnings := total_earnings - regular_earnings
def rate_per_excess_hour := excess_earnings / excess_hours
def multiple_of_regular_rate := rate_per_excess_hour / hourly_rate

-- Statement of the problem
theorem multiple_of_regular_rate_is_1_5 : multiple_of_regular_rate = 1.5 :=
by
  -- Note: The proof is not required, hence sorry is used.
  sorry

end NUMINAMATH_GPT_multiple_of_regular_rate_is_1_5_l120_12028


namespace NUMINAMATH_GPT_problem_l120_12011

variable (x y : ℝ)

theorem problem (h1 : (x + y)^2 = 81) (h2 : x * y = 18) : (x - y)^2 = 9 :=
by
  sorry

end NUMINAMATH_GPT_problem_l120_12011


namespace NUMINAMATH_GPT_fraction_n_p_l120_12007

theorem fraction_n_p (m n p : ℝ) (r1 r2 : ℝ)
  (h1 : r1 * r2 = m)
  (h2 : -(r1 + r2) = p)
  (h3 : m ≠ 0)
  (h4 : n ≠ 0)
  (h5 : p ≠ 0)
  (h6 : m = - (r1 + r2) / 2)
  (h7 : n = r1 * r2 / 4) :
  n / p = 1 / 8 :=
by
  sorry

end NUMINAMATH_GPT_fraction_n_p_l120_12007


namespace NUMINAMATH_GPT_complementary_angle_measure_l120_12070

theorem complementary_angle_measure (A S C : ℝ) (h1 : A = 45) (h2 : A + S = 180) (h3 : A + C = 90) (h4 : S = 3 * C) : C = 45 :=
by
  sorry

end NUMINAMATH_GPT_complementary_angle_measure_l120_12070


namespace NUMINAMATH_GPT_max_area_triangle_l120_12000

noncomputable def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 2

noncomputable def line_eq (x y : ℝ) : Prop := 2 * Real.sqrt 2 * x - y - 1 = 0

theorem max_area_triangle (x1 y1 x2 y2 xp yp : ℝ) (h1 : circle_eq x1 y1) (h2 : circle_eq x2 y2) (h3 : circle_eq xp yp)
  (h4 : line_eq x1 y1) (h5 : line_eq x2 y2) (h6 : (xp, yp) ≠ (x1, y1)) (h7 : (xp, yp) ≠ (x2, y2)) :
  ∃ S : ℝ, S = 10 * Real.sqrt 5 / 9 :=
by
  sorry

end NUMINAMATH_GPT_max_area_triangle_l120_12000


namespace NUMINAMATH_GPT_phil_won_more_games_than_charlie_l120_12030

theorem phil_won_more_games_than_charlie :
  ∀ (P D C Ph : ℕ),
  (P = D + 5) → (C = D - 2) → (Ph = 12) → (P = Ph + 4) →
  Ph - C = 3 :=
by
  intros P D C Ph hP hC hPh hPPh
  sorry

end NUMINAMATH_GPT_phil_won_more_games_than_charlie_l120_12030


namespace NUMINAMATH_GPT_solution_of_equation_l120_12069

theorem solution_of_equation (m : ℝ) :
  (∃ x : ℝ, x = (4 - 3 * m) / 2 ∧ x > 0) ↔ m < 4 / 3 ∧ m ≠ 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_solution_of_equation_l120_12069


namespace NUMINAMATH_GPT_gcd_lcm_8951_4267_l120_12063

theorem gcd_lcm_8951_4267 :
  gcd 8951 4267 = 1 ∧ lcm 8951 4267 = 38212917 :=
by
  sorry

end NUMINAMATH_GPT_gcd_lcm_8951_4267_l120_12063


namespace NUMINAMATH_GPT_minimum_expenses_for_Nikifor_to_win_maximum_F_value_l120_12038

noncomputable def number_of_voters := 35
noncomputable def sellable_voters := 14 -- 40% of 35
noncomputable def preference_voters := 21 -- 60% of 35
noncomputable def minimum_votes_to_win := 18 -- 50% of 35 + 1
noncomputable def cost_per_vote := 9

def vote_supply_function (P : ℕ) : ℕ :=
  if P = 0 then 10
  else if 1 ≤ P ∧ P ≤ 14 then 10 + P
  else 24


theorem minimum_expenses_for_Nikifor_to_win :
  ∃ P : ℕ, P * cost_per_vote = 162 ∧ vote_supply_function P ≥ minimum_votes_to_win := 
sorry

theorem maximum_F_value (F : ℕ) : 
  F = 3 :=
sorry

end NUMINAMATH_GPT_minimum_expenses_for_Nikifor_to_win_maximum_F_value_l120_12038
