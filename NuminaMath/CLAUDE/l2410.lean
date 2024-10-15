import Mathlib

namespace NUMINAMATH_CALUDE_masud_siblings_count_l2410_241073

theorem masud_siblings_count :
  ∀ (janet_siblings masud_siblings carlos_siblings : ℕ),
    janet_siblings = 4 * masud_siblings - 60 →
    carlos_siblings = 3 * masud_siblings / 4 →
    janet_siblings = carlos_siblings + 135 →
    masud_siblings = 60 := by
  sorry

end NUMINAMATH_CALUDE_masud_siblings_count_l2410_241073


namespace NUMINAMATH_CALUDE_single_element_condition_intersection_condition_l2410_241005

-- Define set A
def A (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 + 2 * x + 3 = 0}

-- Define set B
def B : Set ℝ := {x : ℝ | x^2 - 2 * x - 3 = 0}

-- Theorem for the first part of the problem
theorem single_element_condition (a : ℝ) :
  (∃! x, x ∈ A a) ↔ (a = 0 ∨ a = 1/3) := by sorry

-- Theorem for the second part of the problem
theorem intersection_condition (a : ℝ) :
  A a ∩ B = A a ↔ (a > 1/3 ∨ a = -1) := by sorry

end NUMINAMATH_CALUDE_single_element_condition_intersection_condition_l2410_241005


namespace NUMINAMATH_CALUDE_inverse_proportion_l2410_241076

/-- Given that y is inversely proportional to x and when x = 2, y = -3,
    this theorem proves the relationship between y and x, and the value of x when y = 2. -/
theorem inverse_proportion (x y : ℝ) : 
  (∃ k : ℝ, ∀ x ≠ 0, y = k / x) →  -- y is inversely proportional to x
  (2 : ℝ) * (-3 : ℝ) = y * x →     -- when x = 2, y = -3
  y = -6 / x ∧                     -- the function relationship
  (y = 2 → x = -3)                 -- when y = 2, x = -3
  := by sorry

end NUMINAMATH_CALUDE_inverse_proportion_l2410_241076


namespace NUMINAMATH_CALUDE_relatively_prime_power_sums_l2410_241029

theorem relatively_prime_power_sums (a n m : ℕ) (h_odd : Odd a) (h_pos_n : n > 0) (h_pos_m : m > 0) (h_neq : n ≠ m) :
  Nat.gcd (a^(2^m) + 2^(2^m)) (a^(2^n) + 2^(2^n)) = 1 := by
sorry

end NUMINAMATH_CALUDE_relatively_prime_power_sums_l2410_241029


namespace NUMINAMATH_CALUDE_log_expression_equals_one_l2410_241096

-- Define the base-10 logarithm
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Theorem statement
theorem log_expression_equals_one :
  (log10 2)^2 + log10 20 * log10 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equals_one_l2410_241096


namespace NUMINAMATH_CALUDE_f_lower_bound_a_range_l2410_241094

def f (x : ℝ) : ℝ := |x - 2| + |x + 1| + 2 * |x + 2|

theorem f_lower_bound : ∀ x : ℝ, f x ≥ 5 := by sorry

theorem a_range (a : ℝ) : 
  (∀ x : ℝ, 15 - 2 * (f x) < a^2 + 9 / (a^2 + 1)) → 
  a ≠ Real.sqrt 2 ∧ a ≠ -Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_f_lower_bound_a_range_l2410_241094


namespace NUMINAMATH_CALUDE_root_sum_bound_implies_m_range_l2410_241025

theorem root_sum_bound_implies_m_range (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁^2 - 2*x₁ + m + 2 = 0 ∧
               x₂^2 - 2*x₂ + m + 2 = 0 ∧
               x₁ ≠ x₂ ∧
               |x₁| + |x₂| ≤ 3) →
  -13/4 ≤ m ∧ m ≤ -1 :=
by sorry

end NUMINAMATH_CALUDE_root_sum_bound_implies_m_range_l2410_241025


namespace NUMINAMATH_CALUDE_stationery_box_sheets_l2410_241092

/-- Represents a stationery box with sheets of paper and envelopes -/
structure StationeryBox where
  sheets : ℕ
  envelopes : ℕ

/-- Joe's usage of the stationery box -/
def joe_usage (box : StationeryBox) : Prop :=
  box.sheets - box.envelopes = 70

/-- Lily's usage of the stationery box -/
def lily_usage (box : StationeryBox) : Prop :=
  4 * (box.envelopes - 20) = box.sheets

theorem stationery_box_sheets : 
  ∃ (box : StationeryBox), joe_usage box ∧ lily_usage box ∧ box.sheets = 120 := by
  sorry


end NUMINAMATH_CALUDE_stationery_box_sheets_l2410_241092


namespace NUMINAMATH_CALUDE_circle_center_polar_coordinates_l2410_241008

-- Define the polar equation of the circle
def circle_equation (ρ θ : Real) : Prop := ρ = 2 * Real.sin θ ∧ 0 ≤ θ ∧ θ < 2 * Real.pi

-- Define the center in polar coordinates
def is_center (ρ θ : Real) : Prop := 
  (ρ = 1 ∧ θ = Real.pi / 2) ∨ (ρ = 1 ∧ θ = 3 * Real.pi / 2)

-- Theorem statement
theorem circle_center_polar_coordinates :
  ∀ ρ θ : Real, circle_equation ρ θ → 
  ∃ ρ_c θ_c : Real, is_center ρ_c θ_c ∧ 
  (ρ - ρ_c * Real.cos (θ - θ_c))^2 + (ρ * Real.sin θ - ρ_c * Real.sin θ_c)^2 = ρ_c^2 :=
sorry

end NUMINAMATH_CALUDE_circle_center_polar_coordinates_l2410_241008


namespace NUMINAMATH_CALUDE_odd_selections_from_eleven_l2410_241064

theorem odd_selections_from_eleven (n : ℕ) (h : n = 11) :
  (Finset.range n).sum (fun k => if k % 2 = 1 then Nat.choose n k else 0) = 2^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_odd_selections_from_eleven_l2410_241064


namespace NUMINAMATH_CALUDE_total_tickets_is_84_l2410_241097

-- Define the prices of items in tickets
def hat_price : ℕ := 2
def stuffed_animal_price : ℕ := 10
def yoyo_price : ℕ := 2
def keychain_price : ℕ := 3
def poster_price : ℕ := 7
def toy_car_price : ℕ := 5
def puzzle_price : ℕ := 8
def tshirt_price : ℕ := 15
def novelty_pen_price : ℕ := 4

-- Define the special offer price for two posters
def two_posters_special_price : ℕ := 10

-- Define the function to calculate the total tickets spent
def total_tickets_spent : ℕ :=
  -- First trip
  hat_price + stuffed_animal_price + yoyo_price +
  -- Second trip
  keychain_price + poster_price + toy_car_price +
  -- Third trip
  puzzle_price + tshirt_price + novelty_pen_price +
  -- Fourth trip (special offer for posters)
  two_posters_special_price + stuffed_animal_price +
  -- Fifth trip (50% off sale)
  (tshirt_price / 2) + (toy_car_price / 2)

-- Theorem to prove
theorem total_tickets_is_84 : total_tickets_spent = 84 := by
  sorry

end NUMINAMATH_CALUDE_total_tickets_is_84_l2410_241097


namespace NUMINAMATH_CALUDE_rectangular_plot_breadth_l2410_241043

/-- Proves that for a rectangular plot where the length is thrice the breadth
    and the area is 432 sq m, the breadth is 12 m. -/
theorem rectangular_plot_breadth : 
  ∀ (breadth length area : ℝ),
    length = 3 * breadth →
    area = length * breadth →
    area = 432 →
    breadth = 12 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_breadth_l2410_241043


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2410_241061

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 2*x < 0}
def N : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x | 0 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2410_241061


namespace NUMINAMATH_CALUDE_remainder_theorem_l2410_241003

theorem remainder_theorem (n : ℤ) (k : ℤ) (h : n = 100 * k - 1) : (n^2 - n + 4) % 100 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2410_241003


namespace NUMINAMATH_CALUDE_S_2n_plus_one_not_div_by_three_l2410_241042

/-- 
For a non-negative integer n, S_n is defined as the sum of squares 
of the coefficients of the polynomial (1+x)^n
-/
def S (n : ℕ) : ℕ := (Finset.range (n + 1)).sum (fun k => (Nat.choose n k) ^ 2)

/-- 
For any non-negative integer n, S(2n) + 1 is not divisible by 3
-/
theorem S_2n_plus_one_not_div_by_three (n : ℕ) : ¬ (3 ∣ (S (2 * n) + 1)) := by
  sorry

end NUMINAMATH_CALUDE_S_2n_plus_one_not_div_by_three_l2410_241042


namespace NUMINAMATH_CALUDE_complex_product_equality_l2410_241051

theorem complex_product_equality (x : ℂ) (h : x = Complex.exp (2 * Real.pi * Complex.I / 9)) : 
  (3 * x + x^3) * (3 * x^3 + x^9) * (3 * x^6 + x^18) = 
  22 - 9 * x^5 - 9 * x^2 + 3 * x^6 + 4 * x^3 + 3 * x :=
by sorry

end NUMINAMATH_CALUDE_complex_product_equality_l2410_241051


namespace NUMINAMATH_CALUDE_smallest_sum_B_plus_c_l2410_241072

theorem smallest_sum_B_plus_c : ∃ (B c : ℕ),
  (B < 5) ∧                        -- B is a digit in base 5
  (c > 7) ∧                        -- c is a base greater than 7
  (31 * B = 4 * c + 4) ∧           -- BBB_5 = 44_c
  (∀ (B' c' : ℕ),                  -- For all other valid B' and c'
    (B' < 5) →
    (c' > 7) →
    (31 * B' = 4 * c' + 4) →
    (B + c ≤ B' + c')) ∧
  (B + c = 25)                     -- The smallest sum is 25
  := by sorry

end NUMINAMATH_CALUDE_smallest_sum_B_plus_c_l2410_241072


namespace NUMINAMATH_CALUDE_solution_set_part1_integer_a_part2_l2410_241089

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| - |2*x - a|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 5 x ≥ 0} = {x : ℝ | 2 ≤ x ∧ x ≤ 4} :=
sorry

-- Part 2
theorem integer_a_part2 (a : ℤ) :
  (f a 5 ≥ 3 ∧ f a 6 < 3) → a = 9 :=
sorry

end NUMINAMATH_CALUDE_solution_set_part1_integer_a_part2_l2410_241089


namespace NUMINAMATH_CALUDE_deshaun_summer_reading_l2410_241082

theorem deshaun_summer_reading 
  (summer_break_days : ℕ) 
  (avg_pages_per_book : ℕ) 
  (closest_person_percentage : ℚ) 
  (second_person_pages_per_day : ℕ) 
  (h1 : summer_break_days = 80)
  (h2 : avg_pages_per_book = 320)
  (h3 : closest_person_percentage = 3/4)
  (h4 : second_person_pages_per_day = 180) :
  ∃ (books_read : ℕ), books_read = 60 ∧ 
    (books_read * avg_pages_per_book : ℚ) = 
      (second_person_pages_per_day * summer_break_days : ℚ) / closest_person_percentage :=
by sorry

end NUMINAMATH_CALUDE_deshaun_summer_reading_l2410_241082


namespace NUMINAMATH_CALUDE_train_length_calculation_l2410_241018

/-- The length of a train given its speed, the speed of a man running in the opposite direction, and the time it takes for the train to pass the man. -/
theorem train_length_calculation (train_speed : ℝ) (man_speed : ℝ) (passing_time : ℝ) : 
  train_speed = 60 →
  man_speed = 6 →
  passing_time = 5.999520038396929 →
  ∃ (train_length : ℝ), abs (train_length - 110) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l2410_241018


namespace NUMINAMATH_CALUDE_negation_of_exists_exp_leq_zero_l2410_241046

theorem negation_of_exists_exp_leq_zero :
  (¬ ∃ x : ℝ, Real.exp x ≤ 0) ↔ (∀ x : ℝ, Real.exp x > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_exists_exp_leq_zero_l2410_241046


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l2410_241088

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (∃ a b c : ℝ, (4*x + 7)*(3*x - 5) = 15 ∧ a*x^2 + b*x + c = 0) → 
  (∃ x₁ x₂ : ℝ, (4*x₁ + 7)*(3*x₁ - 5) = 15 ∧ 
                (4*x₂ + 7)*(3*x₂ - 5) = 15 ∧ 
                x₁ + x₂ = -1/12) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l2410_241088


namespace NUMINAMATH_CALUDE_fundraising_shortfall_l2410_241034

def goal : ℚ := 500
def pizza_price : ℚ := 12
def fries_price : ℚ := 0.3
def soda_price : ℚ := 2
def pizza_sold : ℕ := 15
def fries_sold : ℕ := 40
def soda_sold : ℕ := 25

theorem fundraising_shortfall :
  goal - (pizza_price * pizza_sold + fries_price * fries_sold + soda_price * soda_sold) = 258 := by
  sorry

end NUMINAMATH_CALUDE_fundraising_shortfall_l2410_241034


namespace NUMINAMATH_CALUDE_age_difference_l2410_241083

/-- Given Billy's current age and the ratio of my age to Billy's, 
    prove the difference between our ages. -/
theorem age_difference (billy_age : ℕ) (age_ratio : ℕ) : 
  billy_age = 4 → age_ratio = 4 → age_ratio * billy_age - billy_age = 12 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l2410_241083


namespace NUMINAMATH_CALUDE_min_x_value_l2410_241062

theorem min_x_value (x y : ℕ+) (h : (100 : ℚ)/151 < (y : ℚ)/(x : ℚ) ∧ (y : ℚ)/(x : ℚ) < (200 : ℚ)/251) :
  ∀ z : ℕ+, z < x → ¬∃ w : ℕ+, (100 : ℚ)/151 < (w : ℚ)/(z : ℚ) ∧ (w : ℚ)/(z : ℚ) < (200 : ℚ)/251 :=
by sorry

end NUMINAMATH_CALUDE_min_x_value_l2410_241062


namespace NUMINAMATH_CALUDE_cake_measuring_l2410_241040

theorem cake_measuring (flour_needed : ℚ) (milk_needed : ℚ) (cup_capacity : ℚ) : 
  flour_needed = 10/3 ∧ milk_needed = 3/2 ∧ cup_capacity = 1/3 → 
  Int.ceil (flour_needed / cup_capacity) + Int.ceil (milk_needed / cup_capacity) = 15 := by
sorry

end NUMINAMATH_CALUDE_cake_measuring_l2410_241040


namespace NUMINAMATH_CALUDE_base3_to_base10_conversion_l2410_241067

/-- Converts a list of digits in base 3 to a natural number in base 10 -/
def base3ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

/-- The base-3 representation of the number -/
def base3Number : List Nat := [2, 0, 1, 0, 2, 1]

theorem base3_to_base10_conversion :
  base3ToBase10 base3Number = 416 := by
  sorry

end NUMINAMATH_CALUDE_base3_to_base10_conversion_l2410_241067


namespace NUMINAMATH_CALUDE_books_remaining_pauls_remaining_books_l2410_241021

theorem books_remaining (initial books_given books_sold : ℕ) :
  initial ≥ books_given + books_sold →
  initial - (books_given + books_sold) = initial - books_given - books_sold :=
by
  sorry

theorem pauls_remaining_books :
  134 - (39 + 27) = 68 :=
by
  sorry

end NUMINAMATH_CALUDE_books_remaining_pauls_remaining_books_l2410_241021


namespace NUMINAMATH_CALUDE_smallest_sum_of_squares_l2410_241060

theorem smallest_sum_of_squares (x y : ℕ) : 
  x^2 - y^2 = 187 → ∀ a b : ℕ, a^2 - b^2 = 187 → x^2 + y^2 ≤ a^2 + b^2 → x^2 + y^2 = 205 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_squares_l2410_241060


namespace NUMINAMATH_CALUDE_direct_proportion_second_fourth_quadrants_l2410_241014

/-- A function f(x) = ax^b is a direct proportion function if and only if b = 1 -/
def is_direct_proportion (a : ℝ) (b : ℝ) : Prop :=
  b = 1

/-- A function f(x) = ax^b has its graph in the second and fourth quadrants if and only if a < 0 -/
def in_second_and_fourth_quadrants (a : ℝ) : Prop :=
  a < 0

/-- The main theorem stating that for y=(m+1)x^(m^2-3) to be a direct proportion function
    with its graph in the second and fourth quadrants, m must be -2 -/
theorem direct_proportion_second_fourth_quadrants :
  ∀ m : ℝ, is_direct_proportion (m + 1) (m^2 - 3) ∧ 
            in_second_and_fourth_quadrants (m + 1) →
            m = -2 :=
by sorry

end NUMINAMATH_CALUDE_direct_proportion_second_fourth_quadrants_l2410_241014


namespace NUMINAMATH_CALUDE_peaches_theorem_l2410_241075

def peaches_problem (peaches_per_basket : ℕ) (num_baskets : ℕ) (peaches_eaten : ℕ) (peaches_per_box : ℕ) : Prop :=
  let total_peaches := peaches_per_basket * num_baskets
  let remaining_peaches := total_peaches - peaches_eaten
  remaining_peaches / peaches_per_box = 8

theorem peaches_theorem : 
  peaches_problem 25 5 5 15 := by sorry

end NUMINAMATH_CALUDE_peaches_theorem_l2410_241075


namespace NUMINAMATH_CALUDE_travel_problem_solution_l2410_241010

/-- Represents the speeds and distance in the problem -/
structure TravelData where
  pedestrian_speed : ℝ
  cyclist_speed : ℝ
  rider_speed : ℝ
  distance_AB : ℝ

/-- The conditions of the problem -/
def problem_conditions (data : TravelData) : Prop :=
  data.cyclist_speed = 2 * data.pedestrian_speed ∧
  2 * data.cyclist_speed + 2 * data.rider_speed = data.distance_AB ∧
  2.8 * data.pedestrian_speed + 2.8 * data.rider_speed = data.distance_AB ∧
  2 * data.rider_speed = data.distance_AB / 2 - 3 ∧
  2 * data.cyclist_speed = data.distance_AB / 2 + 3

/-- The theorem to prove -/
theorem travel_problem_solution :
  ∃ (data : TravelData),
    problem_conditions data ∧
    data.pedestrian_speed = 6 ∧
    data.cyclist_speed = 12 ∧
    data.rider_speed = 9 ∧
    data.distance_AB = 42 :=
by
  sorry

end NUMINAMATH_CALUDE_travel_problem_solution_l2410_241010


namespace NUMINAMATH_CALUDE_house_painting_and_window_washing_l2410_241098

/-- Represents the number of people needed to complete a task in a given number of days -/
structure WorkForce :=
  (people : ℕ)
  (days : ℕ)

/-- Calculates the total person-days for a given workforce -/
def personDays (w : WorkForce) : ℕ := w.people * w.days

theorem house_painting_and_window_washing 
  (paint_initial : WorkForce) 
  (paint_target : WorkForce) 
  (wash_initial : WorkForce) 
  (wash_target : WorkForce) :
  paint_initial.people = 8 →
  paint_initial.days = 5 →
  paint_target.days = 3 →
  wash_initial.people = paint_initial.people →
  wash_initial.days = 4 →
  wash_target.people = wash_initial.people + 4 →
  personDays paint_initial = personDays paint_target →
  personDays wash_initial = personDays wash_target →
  paint_target.people = 14 ∧ wash_target.days = 3 := by
  sorry

#check house_painting_and_window_washing

end NUMINAMATH_CALUDE_house_painting_and_window_washing_l2410_241098


namespace NUMINAMATH_CALUDE_arcsin_three_fifths_cos_tan_l2410_241091

theorem arcsin_three_fifths_cos_tan :
  (Real.cos (Real.arcsin (3/5)) = 4/5) ∧ 
  (Real.tan (Real.arcsin (3/5)) = 3/4) := by
sorry

end NUMINAMATH_CALUDE_arcsin_three_fifths_cos_tan_l2410_241091


namespace NUMINAMATH_CALUDE_unique_four_digit_solution_l2410_241002

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_equation (a b c d : ℕ) : Prop :=
  1000 * a + 100 * b + 10 * c + d - (100 * a + 10 * b + c) - (10 * a + b) - a = 1995

theorem unique_four_digit_solution :
  ∃! (abcd : ℕ), is_four_digit abcd ∧ 
    ∃ (a b c d : ℕ), abcd = 1000 * a + 100 * b + 10 * c + d ∧ digit_equation a b c d ∧
    a ≠ 0 := by sorry

end NUMINAMATH_CALUDE_unique_four_digit_solution_l2410_241002


namespace NUMINAMATH_CALUDE_inscribed_square_area_l2410_241058

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 - 6*x + 8

/-- The side length of the inscribed square -/
noncomputable def s : ℝ := -2 + 2 * Real.sqrt 2

/-- Theorem: The area of the inscribed square is 12 - 8√2 -/
theorem inscribed_square_area :
  let square_area := s^2
  square_area = 12 - 8 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l2410_241058


namespace NUMINAMATH_CALUDE_total_rainfall_2012_l2410_241019

-- Define the average monthly rainfall for each year
def rainfall_2010 : ℝ := 37.2
def rainfall_2011 : ℝ := rainfall_2010 + 3.5
def rainfall_2012 : ℝ := rainfall_2011 - 1.2

-- Define the number of months in a year
def months_in_year : ℕ := 12

-- Theorem statement
theorem total_rainfall_2012 : 
  rainfall_2012 * months_in_year = 474 := by sorry

end NUMINAMATH_CALUDE_total_rainfall_2012_l2410_241019


namespace NUMINAMATH_CALUDE_ratio_sum_theorem_l2410_241045

theorem ratio_sum_theorem (w x y : ℝ) (hw_x : w / x = 1 / 3) (hw_y : w / y = 2 / 3) :
  (x + y) / y = 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_sum_theorem_l2410_241045


namespace NUMINAMATH_CALUDE_gcd_228_1995_l2410_241057

theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := by
  sorry

end NUMINAMATH_CALUDE_gcd_228_1995_l2410_241057


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l2410_241037

theorem quadratic_form_sum (x : ℝ) : ∃ (a b c : ℝ),
  (6 * x^2 + 72 * x + 432 = a * (x + b)^2 + c) ∧ (a + b + c = 228) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l2410_241037


namespace NUMINAMATH_CALUDE_symmetric_difference_of_A_and_B_l2410_241044

-- Define the sets A and B
def A : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2 - 3*x}
def B : Set ℝ := {y : ℝ | ∃ x : ℝ, y = -2^x}

-- Define the symmetric difference operation
def symmetricDifference (X Y : Set ℝ) : Set ℝ := (X \ Y) ∪ (Y \ X)

-- State the theorem
theorem symmetric_difference_of_A_and_B :
  symmetricDifference A B = {y : ℝ | y < -9/4 ∨ y ≥ 0} := by sorry

end NUMINAMATH_CALUDE_symmetric_difference_of_A_and_B_l2410_241044


namespace NUMINAMATH_CALUDE_race_speed_ratio_l2410_241053

/-- Proves that A runs 4 times faster than B given the race conditions --/
theorem race_speed_ratio (v_B : ℝ) (k : ℝ) : 
  (k > 0) →  -- A is faster than B
  (88 / (k * v_B) = (88 - 66) / v_B) →  -- They finish at the same time
  (k = 4) :=
by sorry

end NUMINAMATH_CALUDE_race_speed_ratio_l2410_241053


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2410_241032

theorem min_value_of_expression (x y : ℝ) 
  (h1 : x > -1) 
  (h2 : y > 0) 
  (h3 : x + 2*y = 2) : 
  ∃ (m : ℝ), m = 3 ∧ ∀ (a b : ℝ), a > -1 → b > 0 → a + 2*b = 2 → 1/(a+1) + 2/b ≥ m :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2410_241032


namespace NUMINAMATH_CALUDE_profit_ratio_theorem_l2410_241084

/-- Given two partners p and q with investment ratio 7:5, where p invests for 10 months
    and q invests for 20 months, prove that the ratio of their profits is 7:10 -/
theorem profit_ratio_theorem (p q : ℕ) (investment_p investment_q : ℝ) 
  (time_p time_q : ℕ) (profit_p profit_q : ℝ) :
  investment_p / investment_q = 7 / 5 →
  time_p = 10 →
  time_q = 20 →
  profit_p = investment_p * time_p →
  profit_q = investment_q * time_q →
  profit_p / profit_q = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_profit_ratio_theorem_l2410_241084


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l2410_241048

/-- 
Given a boat traveling downstream in a stream, this theorem proves that 
the speed of the boat in still water is 5 km/hr, based on the given conditions.
-/
theorem boat_speed_in_still_water 
  (stream_speed : ℝ) 
  (downstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (h1 : stream_speed = 5)
  (h2 : downstream_distance = 100)
  (h3 : downstream_time = 10)
  (h4 : downstream_distance = (boat_speed + stream_speed) * downstream_time) :
  boat_speed = 5 := by
  sorry


end NUMINAMATH_CALUDE_boat_speed_in_still_water_l2410_241048


namespace NUMINAMATH_CALUDE_smallest_square_multiplier_l2410_241068

def y : ℕ := 2^4 * 3^3 * 5^4 * 7^2 * 6^7 * 8^3 * 9^10

theorem smallest_square_multiplier (n : ℕ) : 
  (∃ m : ℕ, n * y = m^2) ∧ (∀ k : ℕ, 0 < k ∧ k < n → ¬∃ m : ℕ, k * y = m^2) → n = 1 :=
by sorry

end NUMINAMATH_CALUDE_smallest_square_multiplier_l2410_241068


namespace NUMINAMATH_CALUDE_fraction_equality_l2410_241011

theorem fraction_equality : 
  (1 * 2 * 4 + 2 * 4 * 8 + 3 * 6 * 12 + 4 * 8 * 16) / 
  (1 * 3 * 9 + 2 * 6 * 18 + 3 * 9 * 27 + 4 * 12 * 36) = 8 / 27 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2410_241011


namespace NUMINAMATH_CALUDE_inequality_proof_l2410_241035

theorem inequality_proof (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, f x > deriv f x) (a b : ℝ) (hab : a > b) : 
  Real.exp a * f b > Real.exp b * f a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2410_241035


namespace NUMINAMATH_CALUDE_firefighter_solution_l2410_241056

/-- Represents the problem of calculating the number of firefighters needed to put out a fire. -/
def FirefighterProblem (hose_rate : ℚ) (water_needed : ℚ) (time_taken : ℚ) : Prop :=
  ∃ (num_firefighters : ℚ),
    num_firefighters * hose_rate * time_taken = water_needed ∧
    num_firefighters = 5

/-- Theorem stating that given the specific conditions of the problem, 
    the number of firefighters required is 5. -/
theorem firefighter_solution :
  FirefighterProblem 20 4000 40 := by
  sorry

end NUMINAMATH_CALUDE_firefighter_solution_l2410_241056


namespace NUMINAMATH_CALUDE_sum_of_five_integers_l2410_241006

theorem sum_of_five_integers (a b c d e : ℕ) :
  a ∈ Finset.range 20 ∧ 
  b ∈ Finset.range 20 ∧ 
  c ∈ Finset.range 20 ∧ 
  d ∈ Finset.range 20 ∧ 
  e ∈ Finset.range 20 ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
  c ≠ d ∧ c ≠ e ∧ 
  d ≠ e →
  15 ≤ a + b + c + d + e ∧ a + b + c + d + e ≤ 90 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_five_integers_l2410_241006


namespace NUMINAMATH_CALUDE_pants_price_l2410_241001

theorem pants_price (total_cost : ℝ) (shirt_price : ℝ → ℝ) (shoes_price : ℝ → ℝ) 
  (h1 : total_cost = 340)
  (h2 : ∀ p, shirt_price p = 3/4 * p)
  (h3 : ∀ p, shoes_price p = p + 10) :
  ∃ p, p = 120 ∧ total_cost = shirt_price p + p + shoes_price p :=
sorry

end NUMINAMATH_CALUDE_pants_price_l2410_241001


namespace NUMINAMATH_CALUDE_ellipse_points_equiv_target_set_l2410_241078

/-- An ellipse passing through (2,1) with the given conditions -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h1 : a > b
  h2 : b > 0
  h3 : 4 / a^2 + 1 / b^2 = 1

/-- The set of points on the ellipse satisfying |y| > 1 -/
def ellipse_points (e : Ellipse) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / e.a^2 + p.2^2 / e.b^2 = 1 ∧ |p.2| > 1}

/-- The target set -/
def target_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 < 5 ∧ |p.2| > 1}

/-- The main theorem -/
theorem ellipse_points_equiv_target_set (e : Ellipse) :
  ellipse_points e = target_set := by sorry

end NUMINAMATH_CALUDE_ellipse_points_equiv_target_set_l2410_241078


namespace NUMINAMATH_CALUDE_adams_farm_animals_l2410_241039

theorem adams_farm_animals (cows sheep pigs : ℕ) : 
  sheep = 2 * cows →
  pigs = 3 * sheep →
  cows + sheep + pigs = 108 →
  cows = 12 := by
sorry

end NUMINAMATH_CALUDE_adams_farm_animals_l2410_241039


namespace NUMINAMATH_CALUDE_max_magic_triangle_sum_l2410_241079

def MagicTriangle : Type := Fin 6 → Nat

def isValidTriangle (t : MagicTriangle) : Prop :=
  (∀ i : Fin 6, t i ≥ 4 ∧ t i ≤ 9) ∧
  (∀ i j : Fin 6, i ≠ j → t i ≠ t j)

def sumS (t : MagicTriangle) : Nat :=
  3 * t 0 + 2 * t 1 + 2 * t 2 + t 3 + t 4

def isBalanced (t : MagicTriangle) : Prop :=
  sumS t = 2 * t 2 + t 3 + 2 * t 4 ∧
  sumS t = 2 * t 4 + t 5 + 2 * t 1

theorem max_magic_triangle_sum :
  ∀ t : MagicTriangle, isValidTriangle t → isBalanced t →
  sumS t ≤ 40 :=
sorry

end NUMINAMATH_CALUDE_max_magic_triangle_sum_l2410_241079


namespace NUMINAMATH_CALUDE_forty_percent_value_l2410_241024

theorem forty_percent_value (x : ℝ) : (0.6 * x = 240) → (0.4 * x = 160) := by
  sorry

end NUMINAMATH_CALUDE_forty_percent_value_l2410_241024


namespace NUMINAMATH_CALUDE_lucy_money_problem_l2410_241017

theorem lucy_money_problem (x : ℝ) : 
  let doubled := 2 * x
  let after_giving := doubled * (4/5)
  let after_losing := after_giving * (2/3)
  let after_spending := after_losing * (3/4)
  after_spending = 15 → x = 18.75 := by sorry

end NUMINAMATH_CALUDE_lucy_money_problem_l2410_241017


namespace NUMINAMATH_CALUDE_skating_time_calculation_l2410_241090

theorem skating_time_calculation (distance : ℝ) (speed : ℝ) (time : ℝ) :
  distance = 150 →
  speed = 12 →
  time = distance / speed →
  time = 12.5 :=
by sorry

end NUMINAMATH_CALUDE_skating_time_calculation_l2410_241090


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2410_241036

/-- Two arithmetic sequences and their sums -/
structure ArithmeticSequencePair where
  a : ℕ → ℚ  -- First arithmetic sequence
  b : ℕ → ℚ  -- Second arithmetic sequence
  S : ℕ → ℚ  -- Sum of first n terms of a
  T : ℕ → ℚ  -- Sum of first n terms of b

/-- The main theorem -/
theorem arithmetic_sequence_ratio 
  (seq : ArithmeticSequencePair)
  (h : ∀ n : ℕ, seq.S n / seq.T n = (n + 3) / (2 * n + 1)) :
  seq.a 6 / seq.b 6 = 14 / 23 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2410_241036


namespace NUMINAMATH_CALUDE_rice_cost_difference_l2410_241012

/-- Represents the rice purchase and distribution scenario -/
structure RiceScenario where
  total_rice : ℝ
  price1 : ℝ
  price2 : ℝ
  price3 : ℝ
  quantity1 : ℝ
  quantity2 : ℝ
  quantity3 : ℝ
  kept_ratio : ℝ

/-- Calculates the cost difference between kept and given rice -/
def cost_difference (scenario : RiceScenario) : ℝ :=
  let total_cost := scenario.price1 * scenario.quantity1 + 
                    scenario.price2 * scenario.quantity2 + 
                    scenario.price3 * scenario.quantity3
  let kept_quantity := scenario.kept_ratio * scenario.total_rice
  let given_quantity := scenario.total_rice - kept_quantity
  let kept_cost := scenario.price1 * scenario.quantity1 + 
                   scenario.price2 * (kept_quantity - scenario.quantity1) + 
                   scenario.price3 * (kept_quantity - scenario.quantity1 - scenario.quantity2)
  let given_cost := total_cost - kept_cost
  kept_cost - given_cost

/-- The main theorem stating the cost difference for the given scenario -/
theorem rice_cost_difference : 
  let scenario : RiceScenario := {
    total_rice := 50,
    price1 := 1.2,
    price2 := 1.5,
    price3 := 2,
    quantity1 := 20,
    quantity2 := 25,
    quantity3 := 5,
    kept_ratio := 0.7
  }
  cost_difference scenario = 41.5 := by sorry


end NUMINAMATH_CALUDE_rice_cost_difference_l2410_241012


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l2410_241059

theorem quadratic_root_relation (p q : ℝ) : 
  (∀ x : ℝ, x^2 - p^2*x + p*q = 0 ↔ (∃ y : ℝ, y^2 + p*y + q = 0 ∧ x = y + 1)) →
  (p = 1 ∨ (p = -2 ∧ q = -1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l2410_241059


namespace NUMINAMATH_CALUDE_roots_of_x_squared_equals_16_l2410_241038

theorem roots_of_x_squared_equals_16 :
  let f : ℝ → ℝ := λ x ↦ x^2 - 16
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ = 4 ∧ x₂ = -4 :=
by sorry

end NUMINAMATH_CALUDE_roots_of_x_squared_equals_16_l2410_241038


namespace NUMINAMATH_CALUDE_largest_two_digit_multiple_minus_one_l2410_241007

theorem largest_two_digit_multiple_minus_one : ∃ (n : ℕ), n = 83 ∧ 
  (∀ m : ℕ, m ≥ 10 ∧ m < 100 ∧ 
    (∃ k : ℕ, m + 1 = 3 * k) ∧ 
    (∃ k : ℕ, m + 1 = 4 * k) ∧ 
    (∃ k : ℕ, m + 1 = 5 * k) ∧ 
    (∃ k : ℕ, m + 1 = 7 * k) → 
  m ≤ n) := by
  sorry

end NUMINAMATH_CALUDE_largest_two_digit_multiple_minus_one_l2410_241007


namespace NUMINAMATH_CALUDE_square_perimeter_from_rectangle_perimeter_l2410_241020

/-- Given a square divided into four congruent rectangles, if the perimeter of each rectangle is 28 inches, then the perimeter of the square is 44.8 inches. -/
theorem square_perimeter_from_rectangle_perimeter : 
  ∀ (s : ℝ), 
  s > 0 → -- side length of the square is positive
  (5 * s / 2 = 28) → -- perimeter of each rectangle is 28 inches
  (4 * s = 44.8) -- perimeter of the square is 44.8 inches
:= by sorry

end NUMINAMATH_CALUDE_square_perimeter_from_rectangle_perimeter_l2410_241020


namespace NUMINAMATH_CALUDE_shaded_region_circle_diameter_l2410_241052

/-- Given two concentric circles with radii 24 and 36 units, the diameter of a new circle
    whose diameter is equal to the area of the shaded region between the two circles
    is 720π units. -/
theorem shaded_region_circle_diameter :
  let r₁ : ℝ := 24
  let r₂ : ℝ := 36
  let shaded_area := π * (r₂^2 - r₁^2)
  let new_circle_diameter := shaded_area
  new_circle_diameter = 720 * π :=
by sorry

end NUMINAMATH_CALUDE_shaded_region_circle_diameter_l2410_241052


namespace NUMINAMATH_CALUDE_raffle_tickets_sold_l2410_241028

/-- Given that a school sold $620 worth of raffle tickets at $4 per ticket,
    prove that the number of tickets sold is 155. -/
theorem raffle_tickets_sold (total_money : ℕ) (ticket_cost : ℕ) (num_tickets : ℕ)
  (h1 : total_money = 620)
  (h2 : ticket_cost = 4)
  (h3 : total_money = ticket_cost * num_tickets) :
  num_tickets = 155 := by
  sorry

end NUMINAMATH_CALUDE_raffle_tickets_sold_l2410_241028


namespace NUMINAMATH_CALUDE_horse_fertilizer_production_l2410_241065

-- Define the given constants
def num_horses : ℕ := 80
def total_acres : ℕ := 20
def gallons_per_acre : ℕ := 400
def acres_per_day : ℕ := 4
def total_days : ℕ := 25

-- Define the function to calculate daily fertilizer production per horse
def daily_fertilizer_per_horse : ℚ :=
  (total_acres * gallons_per_acre : ℚ) / (num_horses * total_days)

-- Theorem statement
theorem horse_fertilizer_production :
  daily_fertilizer_per_horse = 20 := by
  sorry

end NUMINAMATH_CALUDE_horse_fertilizer_production_l2410_241065


namespace NUMINAMATH_CALUDE_remainder_problem_l2410_241054

theorem remainder_problem (n : ℕ) (h : n > 0) (h1 : (n + 1) % 6 = 4) : n % 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2410_241054


namespace NUMINAMATH_CALUDE_total_charts_brought_l2410_241015

/-- Represents the number of associate professors -/
def associate_profs : ℕ := 2

/-- Represents the number of assistant professors -/
def assistant_profs : ℕ := 7

/-- Represents the total number of people present -/
def total_people : ℕ := 9

/-- Represents the total number of pencils brought -/
def total_pencils : ℕ := 11

/-- Represents the number of pencils each associate professor brings -/
def pencils_per_associate : ℕ := 2

/-- Represents the number of pencils each assistant professor brings -/
def pencils_per_assistant : ℕ := 1

/-- Represents the number of charts each associate professor brings -/
def charts_per_associate : ℕ := 1

/-- Represents the number of charts each assistant professor brings -/
def charts_per_assistant : ℕ := 2

theorem total_charts_brought : 
  associate_profs * charts_per_associate + assistant_profs * charts_per_assistant = 16 :=
by sorry

end NUMINAMATH_CALUDE_total_charts_brought_l2410_241015


namespace NUMINAMATH_CALUDE_pencil_count_l2410_241031

theorem pencil_count (initial_pencils additional_pencils : ℕ) 
  (h1 : initial_pencils = 27)
  (h2 : additional_pencils = 45) :
  initial_pencils + additional_pencils = 72 :=
by sorry

end NUMINAMATH_CALUDE_pencil_count_l2410_241031


namespace NUMINAMATH_CALUDE_ordering_of_powers_l2410_241087

theorem ordering_of_powers : 5^15 < 3^20 ∧ 3^20 < 2^30 := by
  sorry

end NUMINAMATH_CALUDE_ordering_of_powers_l2410_241087


namespace NUMINAMATH_CALUDE_greatest_sum_on_circle_l2410_241074

theorem greatest_sum_on_circle (x y : ℤ) (h : x^2 + y^2 = 50) : x + y ≤ 10 := by
  sorry

end NUMINAMATH_CALUDE_greatest_sum_on_circle_l2410_241074


namespace NUMINAMATH_CALUDE_simultaneous_inequalities_l2410_241093

theorem simultaneous_inequalities (x : ℝ) :
  x^3 - 11*x^2 + 10*x < 0 ∧ x^3 - 12*x^2 + 32*x > 0 → (1 < x ∧ x < 4) ∨ (8 < x ∧ x < 10) :=
by sorry

end NUMINAMATH_CALUDE_simultaneous_inequalities_l2410_241093


namespace NUMINAMATH_CALUDE_exam_students_count_l2410_241004

theorem exam_students_count :
  ∀ (N : ℕ) (T : ℝ),
    N > 0 →
    T = 80 * N →
    (T - 350) / (N - 5 : ℝ) = 90 →
    N = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_exam_students_count_l2410_241004


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2410_241050

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) -- The geometric sequence
  (S : ℕ → ℝ) -- The sum function
  (h_geom : ∀ n, a (n + 1) = a n * (a 1 / a 0)) -- Condition for geometric sequence
  (h_sum : ∀ n, S n = (a 0) * (1 - (a 1 / a 0)^n) / (1 - (a 1 / a 0))) -- Sum formula
  (h_eq : 8 * S 6 = 7 * S 3) -- Given equation
  : a 1 / a 0 = -1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2410_241050


namespace NUMINAMATH_CALUDE_no_solution_iff_m_leq_two_l2410_241047

/-- The system of inequalities has no solution if and only if m ≤ 2 -/
theorem no_solution_iff_m_leq_two (m : ℝ) :
  (∀ x : ℝ, ¬(x - 2 < 3*x - 6 ∧ x < m)) ↔ m ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_iff_m_leq_two_l2410_241047


namespace NUMINAMATH_CALUDE_highway_vehicle_ratio_l2410_241099

theorem highway_vehicle_ratio (total_vehicles : ℕ) (num_trucks : ℕ) : 
  total_vehicles = 300 → 
  num_trucks = 100 → 
  ∃ (k : ℕ), k * num_trucks = total_vehicles - num_trucks → 
  (total_vehicles - num_trucks) / num_trucks = 2 := by
  sorry

end NUMINAMATH_CALUDE_highway_vehicle_ratio_l2410_241099


namespace NUMINAMATH_CALUDE_square_side_length_l2410_241009

/-- Given an arrangement of rectangles and squares forming a larger rectangle,
    this theorem proves that the side length of square S2 is 900 units. -/
theorem square_side_length (r : ℕ) : 
  (2 * r + 900 = 2800) ∧ (2 * r + 3 * 900 = 4600) → 900 = 900 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l2410_241009


namespace NUMINAMATH_CALUDE_tom_rides_l2410_241033

/-- Given the total number of tickets, tickets spent, and cost per ride,
    calculate the number of rides Tom can go on. -/
def number_of_rides (total_tickets spent_tickets cost_per_ride : ℕ) : ℕ :=
  (total_tickets - spent_tickets) / cost_per_ride

/-- Theorem stating that Tom can go on 3 rides given the specific conditions. -/
theorem tom_rides : number_of_rides 40 28 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_tom_rides_l2410_241033


namespace NUMINAMATH_CALUDE_product_mod_600_l2410_241026

theorem product_mod_600 : (2537 * 1985) % 600 = 145 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_600_l2410_241026


namespace NUMINAMATH_CALUDE_water_bottles_taken_out_l2410_241081

theorem water_bottles_taken_out (red : ℕ) (black : ℕ) (blue : ℕ) (remaining : ℕ) :
  red = 2 → black = 3 → blue = 4 → remaining = 4 →
  red + black + blue - remaining = 5 :=
by sorry

end NUMINAMATH_CALUDE_water_bottles_taken_out_l2410_241081


namespace NUMINAMATH_CALUDE_right_triangle_probability_l2410_241069

/-- A 3x3 grid of nine unit squares -/
structure Grid :=
  (vertices : Fin 16 → ℝ × ℝ)

/-- Three vertices selected from the grid -/
structure SelectedVertices :=
  (v1 v2 v3 : Fin 16)

/-- Predicate to check if three vertices form a right triangle -/
def is_right_triangle (g : Grid) (sv : SelectedVertices) : Prop :=
  sorry

/-- The total number of ways to select three vertices from 16 -/
def total_selections : ℕ := Nat.choose 16 3

/-- The number of right triangles that can be formed -/
def right_triangle_count (g : Grid) : ℕ :=
  sorry

/-- The main theorem stating the probability -/
theorem right_triangle_probability (g : Grid) :
  (right_triangle_count g : ℚ) / total_selections = 5 / 14 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_probability_l2410_241069


namespace NUMINAMATH_CALUDE_unattainable_value_of_function_l2410_241086

theorem unattainable_value_of_function (x : ℝ) (y : ℝ) : 
  x ≠ -4/3 → 
  y = (2-x) / (3*x+4) → 
  y ≠ -1/3 := by
sorry

end NUMINAMATH_CALUDE_unattainable_value_of_function_l2410_241086


namespace NUMINAMATH_CALUDE_jennifer_grooming_time_l2410_241055

/-- The time it takes Jennifer to groom one dog, in minutes. -/
def grooming_time : ℕ := 20

/-- The number of dogs Jennifer has. -/
def num_dogs : ℕ := 2

/-- The number of days in the given period. -/
def num_days : ℕ := 30

/-- The total time Jennifer spends grooming her dogs in the given period, in hours. -/
def total_grooming_time : ℕ := 20

theorem jennifer_grooming_time :
  grooming_time * num_dogs * num_days = total_grooming_time * 60 :=
sorry

end NUMINAMATH_CALUDE_jennifer_grooming_time_l2410_241055


namespace NUMINAMATH_CALUDE_quadratic_square_of_binomial_l2410_241095

/-- Given a quadratic expression of the form bx^2 + 16x + 16,
    if it is the square of a binomial, then b = 4. -/
theorem quadratic_square_of_binomial (b : ℝ) :
  (∃ t u : ℝ, ∀ x : ℝ, bx^2 + 16*x + 16 = (t*x + u)^2) →
  b = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_square_of_binomial_l2410_241095


namespace NUMINAMATH_CALUDE_current_wax_amount_l2410_241085

theorem current_wax_amount (total_required : ℕ) (additional_needed : ℕ) 
  (h1 : total_required = 492)
  (h2 : additional_needed = 481) :
  total_required - additional_needed = 11 := by
  sorry

end NUMINAMATH_CALUDE_current_wax_amount_l2410_241085


namespace NUMINAMATH_CALUDE_max_acute_angles_in_hexagon_l2410_241013

/-- A convex hexagon is a polygon with 6 sides where all interior points are on the same side of any line through two vertices. -/
structure ConvexHexagon where
  -- We don't need to define the full structure, just declare it exists
  dummy : Unit

/-- An angle is acute if it is less than 90 degrees. -/
def is_acute (angle : ℝ) : Prop := angle > 0 ∧ angle < 90

/-- The sum of interior angles of a hexagon is 720 degrees. -/
axiom hexagon_angle_sum (h : ConvexHexagon) : 
  ∃ (a₁ a₂ a₃ a₄ a₅ a₆ : ℝ), a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = 720

/-- The theorem stating that the maximum number of acute angles in a convex hexagon is 3. -/
theorem max_acute_angles_in_hexagon (h : ConvexHexagon) :
  ∃ (a₁ a₂ a₃ a₄ a₅ a₆ : ℝ),
    (is_acute a₁ ∧ is_acute a₂ ∧ is_acute a₃) ∧
    a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = 720 ∧
    ¬∃ (b₁ b₂ b₃ b₄ : ℝ),
      (is_acute b₁ ∧ is_acute b₂ ∧ is_acute b₃ ∧ is_acute b₄) ∧
      ∃ (b₅ b₆ : ℝ), b₁ + b₂ + b₃ + b₄ + b₅ + b₆ = 720 :=
by
  sorry


end NUMINAMATH_CALUDE_max_acute_angles_in_hexagon_l2410_241013


namespace NUMINAMATH_CALUDE_repeating_decimal_difference_l2410_241071

theorem repeating_decimal_difference : 
  (6 : ℚ) / 11 - 54 / 100 = 6 / 1100 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_difference_l2410_241071


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2410_241000

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x ≥ 1}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 1 ≤ x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2410_241000


namespace NUMINAMATH_CALUDE_odd_function_has_zero_l2410_241063

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Theorem statement
theorem odd_function_has_zero (f : ℝ → ℝ) (h : OddFunction f) : 
  ∃ x : ℝ, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_has_zero_l2410_241063


namespace NUMINAMATH_CALUDE_waiter_income_fraction_l2410_241080

theorem waiter_income_fraction (salary tips income : ℚ) : 
  income = salary + tips → 
  tips = (5 : ℚ) / 4 * salary → 
  tips / income = (5 : ℚ) / 9 := by
  sorry

end NUMINAMATH_CALUDE_waiter_income_fraction_l2410_241080


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2410_241041

theorem rationalize_denominator :
  (1 : ℝ) / (Real.rpow 3 (1/3) + Real.rpow 27 (1/3)) = 
  (Real.rpow 9 (1/3)) / (3 * (Real.rpow 9 (1/3) + 1)) := by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l2410_241041


namespace NUMINAMATH_CALUDE_initial_men_count_l2410_241077

/-- Represents the work scenario with given parameters -/
structure WorkScenario where
  men : ℕ
  hoursPerDay : ℕ
  depth : ℕ

/-- Calculates the work done in a given scenario -/
def workDone (scenario : WorkScenario) : ℕ :=
  scenario.men * scenario.hoursPerDay * scenario.depth

theorem initial_men_count : ∃ (initialMen : ℕ),
  let scenario1 := WorkScenario.mk initialMen 8 30
  let scenario2 := WorkScenario.mk (initialMen + 55) 6 50
  workDone scenario1 = workDone scenario2 ∧ initialMen = 275 := by
  sorry

#check initial_men_count

end NUMINAMATH_CALUDE_initial_men_count_l2410_241077


namespace NUMINAMATH_CALUDE_f_inequality_solution_set_f_inequality_a_range_l2410_241022

def f (x : ℝ) : ℝ := |x - 1| - |2*x + 3|

theorem f_inequality_solution_set :
  {x : ℝ | f x > 2} = {x : ℝ | -2 < x ∧ x < -4/3} :=
sorry

theorem f_inequality_a_range :
  {a : ℝ | ∃ x, f x ≤ 3/2 * a^2 - a} = {a : ℝ | a ≥ 5/3 ∨ a ≤ -1} :=
sorry

end NUMINAMATH_CALUDE_f_inequality_solution_set_f_inequality_a_range_l2410_241022


namespace NUMINAMATH_CALUDE_smallest_n_is_34_l2410_241016

/-- Given a natural number n ≥ 16, this function represents the set {16, 17, ..., n} -/
def S (n : ℕ) : Set ℕ := {x | 16 ≤ x ∧ x ≤ n}

/-- This function checks if a sequence of 15 natural numbers satisfies the required conditions -/
def valid_sequence (n : ℕ) (a : Fin 15 → ℕ) : Prop :=
  (∀ i : Fin 15, a i ∈ S n) ∧
  (∀ i : Fin 15, (i.val + 1) ∣ a i) ∧
  (∀ i j : Fin 15, i ≠ j → a i ≠ a j)

/-- The main theorem stating that 34 is the smallest n satisfying the conditions -/
theorem smallest_n_is_34 :
  (∃ a : Fin 15 → ℕ, valid_sequence 34 a) ∧
  (∀ m : ℕ, m < 34 → ¬∃ a : Fin 15 → ℕ, valid_sequence m a) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_is_34_l2410_241016


namespace NUMINAMATH_CALUDE_focal_radius_circle_tangent_y_axis_l2410_241030

/-- Represents a parabola y^2 = 2px where p > 0 -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Represents a circle with diameter equal to the focal radius of a parabola -/
structure FocalRadiusCircle (para : Parabola) where
  center : ℝ × ℝ
  radius : ℝ

/-- The circle with diameter equal to the focal radius of the parabola y^2 = 2px (p > 0) is tangent to the y-axis -/
theorem focal_radius_circle_tangent_y_axis (para : Parabola) :
  ∃ (c : FocalRadiusCircle para), c.center.1 = c.radius := by
  sorry

end NUMINAMATH_CALUDE_focal_radius_circle_tangent_y_axis_l2410_241030


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2410_241049

theorem purely_imaginary_complex_number (m : ℝ) : 
  let z : ℂ := (m^2 - m) + m * Complex.I
  (∃ (y : ℝ), z = y * Complex.I) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2410_241049


namespace NUMINAMATH_CALUDE_prob_at_least_two_dice_less_than_10_l2410_241070

/-- The probability of a single 20-sided die showing a number less than 10 -/
def p_less_than_10 : ℚ := 9 / 20

/-- The probability of a single 20-sided die showing a number 10 or above -/
def p_10_or_above : ℚ := 11 / 20

/-- The number of dice rolled -/
def n : ℕ := 5

/-- The probability of exactly k dice showing a number less than 10 -/
def prob_k (k : ℕ) : ℚ :=
  (n.choose k) * (p_less_than_10 ^ k) * (p_10_or_above ^ (n - k))

/-- The probability of at least two dice showing a number less than 10 -/
def prob_at_least_two : ℚ :=
  prob_k 2 + prob_k 3 + prob_k 4 + prob_k 5

theorem prob_at_least_two_dice_less_than_10 :
  prob_at_least_two = 157439 / 20000 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_two_dice_less_than_10_l2410_241070


namespace NUMINAMATH_CALUDE_tigers_season_games_l2410_241066

def total_games (games_won : ℕ) (games_lost : ℕ) : ℕ :=
  games_won + games_lost

theorem tigers_season_games :
  let games_won : ℕ := 18
  let games_lost : ℕ := games_won + 21
  total_games games_won games_lost = 57 := by
  sorry

end NUMINAMATH_CALUDE_tigers_season_games_l2410_241066


namespace NUMINAMATH_CALUDE_mollys_brothers_children_l2410_241023

/-- The number of children each of Molly's brothers has -/
def children_per_brother : ℕ := 2

theorem mollys_brothers_children :
  let cost_per_package : ℕ := 5
  let num_parents : ℕ := 2
  let num_brothers : ℕ := 3
  let total_cost : ℕ := 70
  let immediate_family : ℕ := num_parents + num_brothers + num_brothers -- includes spouses
  (cost_per_package * (immediate_family + num_brothers * children_per_brother) = total_cost) ∧
  (children_per_brother > 0) :=
by sorry

end NUMINAMATH_CALUDE_mollys_brothers_children_l2410_241023


namespace NUMINAMATH_CALUDE_exists_valid_arrangement_l2410_241027

/-- Represents the positions on the square --/
inductive Position
  | TopLeft
  | TopRight
  | BottomLeft
  | BottomRight
  | Center

/-- Defines whether two positions are connected --/
def connected (p1 p2 : Position) : Prop :=
  (p1 = Position.Center ∨ p2 = Position.Center) ∧ p1 ≠ p2

/-- Defines an arrangement of numbers on the square --/
def Arrangement := Position → ℕ

/-- Checks if the arrangement satisfies the required conditions --/
def valid_arrangement (arr : Arrangement) : Prop :=
  (∀ p1 p2, connected p1 p2 → ∃ d > 1, d ∣ arr p1 ∧ d ∣ arr p2) ∧
  (∀ p1 p2, ¬connected p1 p2 → Nat.gcd (arr p1) (arr p2) = 1)

/-- The main theorem stating the existence of a valid arrangement --/
theorem exists_valid_arrangement : ∃ arr : Arrangement, valid_arrangement arr := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_arrangement_l2410_241027
