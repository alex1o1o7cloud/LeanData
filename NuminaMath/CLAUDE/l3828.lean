import Mathlib

namespace NUMINAMATH_CALUDE_total_apples_calculation_l3828_382804

/-- The number of apples given to each person -/
def apples_per_person : ℝ := 15.0

/-- The number of people who received apples -/
def number_of_people : ℝ := 3.0

/-- The total number of apples given -/
def total_apples : ℝ := apples_per_person * number_of_people

theorem total_apples_calculation : total_apples = 45.0 := by
  sorry

end NUMINAMATH_CALUDE_total_apples_calculation_l3828_382804


namespace NUMINAMATH_CALUDE_birds_and_storks_on_fence_l3828_382875

theorem birds_and_storks_on_fence (initial_birds initial_storks additional_birds final_total : ℕ) :
  initial_birds = 3 →
  additional_birds = 5 →
  final_total = 10 →
  initial_birds + initial_storks + additional_birds = final_total →
  initial_storks = 2 := by
  sorry

end NUMINAMATH_CALUDE_birds_and_storks_on_fence_l3828_382875


namespace NUMINAMATH_CALUDE_donut_selections_l3828_382812

theorem donut_selections :
  (Nat.choose 9 3) = 84 := by
  sorry

end NUMINAMATH_CALUDE_donut_selections_l3828_382812


namespace NUMINAMATH_CALUDE_min_value_quadratic_l3828_382846

theorem min_value_quadratic (x y : ℝ) : 
  y = x^2 + 16*x + 10 → (∀ z, y ≤ z → z = x^2 + 16*x + 10) → y = -54 :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l3828_382846


namespace NUMINAMATH_CALUDE_pen_wholesale_price_l3828_382800

/-- The wholesale price of a pen -/
def wholesale_price : ℝ := 2.5

/-- The retail price of one pen -/
def retail_price_one : ℝ := 5

/-- The retail price of three pens -/
def retail_price_three : ℝ := 10

/-- Theorem stating that the wholesale price of a pen is 2.5 rubles -/
theorem pen_wholesale_price : 
  (retail_price_one - wholesale_price = retail_price_three - 3 * wholesale_price) → 
  wholesale_price = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_pen_wholesale_price_l3828_382800


namespace NUMINAMATH_CALUDE_f_2_3_4_equals_59_l3828_382835

def f (x y z : ℝ) : ℝ := 2 * x^3 + 3 * y^2 + z^2

theorem f_2_3_4_equals_59 : f 2 3 4 = 59 := by
  sorry

end NUMINAMATH_CALUDE_f_2_3_4_equals_59_l3828_382835


namespace NUMINAMATH_CALUDE_x_range_l3828_382801

theorem x_range (x : ℝ) : 
  (6 - 3 * x ≥ 0) ∧ (¬(1 / (x + 1) < 0)) → x ∈ Set.Icc (-1) 2 := by
sorry

end NUMINAMATH_CALUDE_x_range_l3828_382801


namespace NUMINAMATH_CALUDE_rectangle_ratio_l3828_382856

/-- Configuration of rectangles around a square -/
structure RectangleConfig where
  /-- Side length of the smaller square -/
  s : ℝ
  /-- Shorter side of the rectangle -/
  y : ℝ
  /-- Longer side of the rectangle -/
  x : ℝ
  /-- The side length of the smaller square is 1 -/
  h1 : s = 1
  /-- The side length of the larger square is s + 2y -/
  h2 : s + 2*y = 2*s
  /-- The side length of the larger square is also x + s -/
  h3 : x + s = 2*s

/-- The ratio of the longer side to the shorter side of the rectangle is 2 -/
theorem rectangle_ratio (config : RectangleConfig) : x / y = 2 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l3828_382856


namespace NUMINAMATH_CALUDE_two_roots_of_f_l3828_382897

/-- The function f(x) = 2^x - 3x has exactly two real roots -/
theorem two_roots_of_f : ∃! (n : ℕ), n = 2 ∧ (∃ (S : Set ℝ), S = {x : ℝ | 2^x - 3*x = 0} ∧ Finite S ∧ Nat.card S = n) := by
  sorry

end NUMINAMATH_CALUDE_two_roots_of_f_l3828_382897


namespace NUMINAMATH_CALUDE_farm_animals_count_l3828_382824

theorem farm_animals_count (total_animals : ℕ) (total_legs : ℕ) 
  (h1 : total_animals = 300) 
  (h2 : total_legs = 688) : 
  ∃ (ducks cows : ℕ), 
    ducks + cows = total_animals ∧ 
    2 * ducks + 4 * cows = total_legs ∧ 
    ducks = 256 := by
  sorry

end NUMINAMATH_CALUDE_farm_animals_count_l3828_382824


namespace NUMINAMATH_CALUDE_total_pokemon_cards_l3828_382871

/-- The number of people with Pokemon cards -/
def num_people : ℕ := 6

/-- The number of Pokemon cards each person has -/
def cards_per_person : ℕ := 100

/-- Theorem: The total number of Pokemon cards for 6 people, each having 100 cards, is equal to 600 -/
theorem total_pokemon_cards : num_people * cards_per_person = 600 := by
  sorry

end NUMINAMATH_CALUDE_total_pokemon_cards_l3828_382871


namespace NUMINAMATH_CALUDE_complement_of_union_is_two_four_l3828_382885

-- Define the universal set U
def U : Set ℕ := {x | x > 0 ∧ x < 6}

-- Define sets A and B
def A : Set ℕ := {1, 3}
def B : Set ℕ := {3, 5}

-- State the theorem
theorem complement_of_union_is_two_four :
  (U \ (A ∪ B)) = {2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_is_two_four_l3828_382885


namespace NUMINAMATH_CALUDE_frac_one_over_x_is_fraction_l3828_382868

-- Define what a fraction is
def is_fraction (expr : ℚ → ℚ) : Prop :=
  ∃ (n d : ℚ → ℚ), ∀ x, expr x = (n x) / (d x)

-- State the theorem
theorem frac_one_over_x_is_fraction :
  is_fraction (λ x => 1 / x) :=
sorry

end NUMINAMATH_CALUDE_frac_one_over_x_is_fraction_l3828_382868


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3828_382899

-- Problem 1
theorem problem_1 : 
  |(-2 : ℝ)| + Real.sqrt 2 * Real.tan (45 * π / 180) - Real.sqrt 8 - (2023 - Real.pi) ^ (0 : ℝ) = 1 - Real.sqrt 2 := by
  sorry

-- Problem 2
theorem problem_2 : 
  ∀ x : ℝ, x ≠ 2 → ((2 * x - 3) / (x - 2) - 1 / (2 - x) = 1 ↔ x = 0) := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3828_382899


namespace NUMINAMATH_CALUDE_julian_comic_frames_l3828_382826

/-- Calculates the total number of frames in Julian's comic book --/
def total_frames (total_pages : Nat) (avg_frames : Nat) (pages_305 : Nat) (pages_250 : Nat) : Nat :=
  let frames_305 := pages_305 * 305
  let frames_250 := pages_250 * 250
  let remaining_pages := total_pages - pages_305 - pages_250
  let frames_avg := remaining_pages * avg_frames
  frames_305 + frames_250 + frames_avg

/-- Proves that the total number of frames in Julian's comic book is 7040 --/
theorem julian_comic_frames :
  total_frames 25 280 10 7 = 7040 := by
  sorry

end NUMINAMATH_CALUDE_julian_comic_frames_l3828_382826


namespace NUMINAMATH_CALUDE_unique_positive_number_l3828_382888

theorem unique_positive_number : ∃! x : ℝ, x > 0 ∧ x + 17 = 60 * (1/x) := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_number_l3828_382888


namespace NUMINAMATH_CALUDE_remaining_water_l3828_382807

/-- Calculates the remaining amount of water after an experiment -/
theorem remaining_water (initial : ℚ) (used : ℚ) (remaining : ℚ) : 
  initial = 3 → used = 5/4 → remaining = initial - used → remaining = 7/4 := by
  sorry

end NUMINAMATH_CALUDE_remaining_water_l3828_382807


namespace NUMINAMATH_CALUDE_zeroth_power_of_nonzero_rational_l3828_382862

theorem zeroth_power_of_nonzero_rational (x : ℚ) (h : x ≠ 0) : x^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_zeroth_power_of_nonzero_rational_l3828_382862


namespace NUMINAMATH_CALUDE_problem_distribution_l3828_382815

def distribute_problems (n : ℕ) (m : ℕ) : ℕ :=
  n * (n - 1) * n^(m - 2)

theorem problem_distribution :
  distribute_problems 12 5 = 228096 := by
  sorry

end NUMINAMATH_CALUDE_problem_distribution_l3828_382815


namespace NUMINAMATH_CALUDE_fraction_multiplication_l3828_382889

theorem fraction_multiplication : (2 : ℚ) / 5 * (5 : ℚ) / 9 * (1 : ℚ) / 2 = (1 : ℚ) / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l3828_382889


namespace NUMINAMATH_CALUDE_nadia_flower_cost_l3828_382838

/-- The total cost of flowers bought by Nadia -/
def total_cost (num_roses : ℕ) (rose_price : ℚ) : ℚ :=
  let num_lilies : ℚ := (3 / 4) * num_roses
  let lily_price : ℚ := 2 * rose_price
  num_roses * rose_price + num_lilies * lily_price

/-- Theorem stating the total cost of flowers for Nadia's purchase -/
theorem nadia_flower_cost : total_cost 20 5 = 250 := by
  sorry

end NUMINAMATH_CALUDE_nadia_flower_cost_l3828_382838


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l3828_382893

/-- A line tangent to a circle and passing through a point -/
theorem tangent_line_to_circle (x y : ℝ) : 
  -- The line equation
  (y - 4 = 3/4 * (x + 3)) →
  -- The line passes through (-3, 4)
  ((-3 : ℝ), (4 : ℝ)) ∈ {(x, y) | y - 4 = 3/4 * (x + 3)} →
  -- The line is tangent to the circle
  (∃! (p : ℝ × ℝ), p ∈ {(x, y) | x^2 + y^2 = 25} ∩ {(x, y) | y - 4 = 3/4 * (x + 3)}) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l3828_382893


namespace NUMINAMATH_CALUDE_second_term_of_geometric_series_l3828_382841

theorem second_term_of_geometric_series (a : ℝ) (r : ℝ) (S : ℝ) : 
  r = (1 : ℝ) / 4 →
  S = 16 →
  S = a / (1 - r) →
  a * r = 3 :=
by sorry

end NUMINAMATH_CALUDE_second_term_of_geometric_series_l3828_382841


namespace NUMINAMATH_CALUDE_bodyguard_hours_theorem_l3828_382854

/-- The number of hours per day Tim hires bodyguards -/
def hours_per_day (num_bodyguards : ℕ) (hourly_rate : ℕ) (weekly_payment : ℕ) (days_per_week : ℕ) : ℕ :=
  weekly_payment / (num_bodyguards * hourly_rate * days_per_week)

/-- Theorem stating that Tim hires bodyguards for 8 hours per day -/
theorem bodyguard_hours_theorem (num_bodyguards : ℕ) (hourly_rate : ℕ) (weekly_payment : ℕ) (days_per_week : ℕ)
  (h1 : num_bodyguards = 2)
  (h2 : hourly_rate = 20)
  (h3 : weekly_payment = 2240)
  (h4 : days_per_week = 7) :
  hours_per_day num_bodyguards hourly_rate weekly_payment days_per_week = 8 := by
  sorry

end NUMINAMATH_CALUDE_bodyguard_hours_theorem_l3828_382854


namespace NUMINAMATH_CALUDE_bubble_pass_probability_specific_l3828_382816

/-- The probability that in a sequence of n distinct terms,
    the kth term ends up in the mth position after one bubble pass -/
def bubble_pass_probability (n k m : ℕ) : ℚ :=
  if k ≤ m ∧ m < n then
    (1 : ℚ) / k * (1 : ℚ) / (m - k + 1) * (1 : ℚ) / (n - m)
  else 0

/-- The main theorem stating the probability for the specific case -/
theorem bubble_pass_probability_specific :
  bubble_pass_probability 50 20 40 = 1 / 4000 := by
  sorry

#eval bubble_pass_probability 50 20 40

end NUMINAMATH_CALUDE_bubble_pass_probability_specific_l3828_382816


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3828_382878

theorem inequality_equivalence (y : ℝ) : 
  3/20 + |y - 7/40| < 1/4 ↔ 3/40 < y ∧ y < 11/40 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3828_382878


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3828_382840

theorem inequality_solution_set (x : ℝ) :
  (x^2 + x - 6 ≤ 0) ↔ (-3 ≤ x ∧ x ≤ 2) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3828_382840


namespace NUMINAMATH_CALUDE_pizza_pasta_price_difference_l3828_382869

/-- The price difference between a Pizza and a Pasta -/
def price_difference (pizza_price chilli_price pasta_price : ℚ) : ℚ :=
  pizza_price - pasta_price

/-- The total cost of the Smith family's purchase -/
def smith_purchase (pizza_price chilli_price pasta_price : ℚ) : ℚ :=
  2 * pizza_price + 3 * chilli_price + 4 * pasta_price

/-- The total cost of the Patel family's purchase -/
def patel_purchase (pizza_price chilli_price pasta_price : ℚ) : ℚ :=
  5 * pizza_price + 6 * chilli_price + 7 * pasta_price

theorem pizza_pasta_price_difference 
  (pizza_price chilli_price pasta_price : ℚ) 
  (h1 : smith_purchase pizza_price chilli_price pasta_price = 53)
  (h2 : patel_purchase pizza_price chilli_price pasta_price = 107) :
  price_difference pizza_price chilli_price pasta_price = 1 := by
  sorry

end NUMINAMATH_CALUDE_pizza_pasta_price_difference_l3828_382869


namespace NUMINAMATH_CALUDE_smallest_sum_with_lcm_2012_l3828_382836

theorem smallest_sum_with_lcm_2012 (a b c d e f g : ℕ) : 
  (Nat.lcm a (Nat.lcm b (Nat.lcm c (Nat.lcm d (Nat.lcm e (Nat.lcm f g)))))) = 2012 → 
  a + b + c + d + e + f + g ≥ 512 := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_with_lcm_2012_l3828_382836


namespace NUMINAMATH_CALUDE_square_field_area_l3828_382882

theorem square_field_area (wire_cost_per_meter : ℝ) (total_cost : ℝ) (gate_width : ℝ) (num_gates : ℕ) :
  wire_cost_per_meter = 3.5 →
  total_cost = 2331 →
  gate_width = 1 →
  num_gates = 2 →
  ∃ (side_length : ℝ),
    side_length > 0 ∧
    (4 * side_length - num_gates * gate_width) * wire_cost_per_meter = total_cost ∧
    side_length^2 = 27889 :=
by sorry

end NUMINAMATH_CALUDE_square_field_area_l3828_382882


namespace NUMINAMATH_CALUDE_negative_negative_eight_properties_l3828_382806

theorem negative_negative_eight_properties :
  let x : ℤ := -8
  let y : ℤ := -(-x)
  (y = -x) ∧ 
  (y = -1 * x) ∧ 
  (y = |x|) ∧ 
  (y = 8) := by sorry

end NUMINAMATH_CALUDE_negative_negative_eight_properties_l3828_382806


namespace NUMINAMATH_CALUDE_equal_area_point_on_diagonal_l3828_382853

/-- A point inside a rectangle where lines through it parallel to the sides create equal-area subrectangles -/
structure EqualAreaPoint (a b : ℝ) where
  x : ℝ
  y : ℝ
  x_bounds : 0 < x ∧ x < a
  y_bounds : 0 < y ∧ y < b
  equal_areas : x * y = (a - x) * y ∧ x * (b - y) = (a - x) * (b - y)

/-- The diagonals of a rectangle -/
def rectangleDiagonals (a b : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ (p = (t * a, t * b) ∨ p = ((1 - t) * a, t * b))}

/-- Theorem: Points satisfying the equal area condition lie on the diagonals of the rectangle -/
theorem equal_area_point_on_diagonal (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
    (p : EqualAreaPoint a b) : (p.x, p.y) ∈ rectangleDiagonals a b := by
  sorry

end NUMINAMATH_CALUDE_equal_area_point_on_diagonal_l3828_382853


namespace NUMINAMATH_CALUDE_box_number_equation_l3828_382876

theorem box_number_equation (x : ℝ) : 
  (x > 0 ∧ 8 + 7 / x + 3 / 1000 = 8.073) ↔ x = 100 := by
sorry

end NUMINAMATH_CALUDE_box_number_equation_l3828_382876


namespace NUMINAMATH_CALUDE_max_reciprocal_sum_l3828_382883

theorem max_reciprocal_sum (m n : ℝ) (h1 : m * n > 0) (h2 : m + n = -1) :
  1 / m + 1 / n ≤ 4 := by
sorry

end NUMINAMATH_CALUDE_max_reciprocal_sum_l3828_382883


namespace NUMINAMATH_CALUDE_function_inequality_implies_a_range_l3828_382825

theorem function_inequality_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, x > 0 → x * (2 * Real.log a - Real.log x) ≤ a) →
  (0 < a ∧ a ≤ Real.exp (-1)) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_a_range_l3828_382825


namespace NUMINAMATH_CALUDE_number_solution_l3828_382839

theorem number_solution : ∃ x : ℝ, (35 - 3 * x = 14) ∧ (x = 7) := by sorry

end NUMINAMATH_CALUDE_number_solution_l3828_382839


namespace NUMINAMATH_CALUDE_dividend_rate_calculation_l3828_382866

/-- Dividend calculation problem -/
theorem dividend_rate_calculation
  (preferred_shares : ℕ)
  (common_shares : ℕ)
  (par_value : ℚ)
  (common_dividend_rate : ℚ)
  (total_annual_dividend : ℚ)
  (h1 : preferred_shares = 1200)
  (h2 : common_shares = 3000)
  (h3 : par_value = 50)
  (h4 : common_dividend_rate = 7/200)  -- 3.5% converted to a fraction
  (h5 : total_annual_dividend = 16500) :
  let preferred_dividend_rate := (total_annual_dividend - 2 * common_shares * par_value * common_dividend_rate) / (preferred_shares * par_value)
  preferred_dividend_rate = 1/10 := by sorry

end NUMINAMATH_CALUDE_dividend_rate_calculation_l3828_382866


namespace NUMINAMATH_CALUDE_largest_prime_fermat_like_under_300_l3828_382864

def fermat_like (n : ℕ) : ℕ := 2^n + 1

theorem largest_prime_fermat_like_under_300 :
  ∃ (p : ℕ), 
    Nat.Prime p ∧ 
    (∃ (n : ℕ), Nat.Prime n ∧ fermat_like n = p) ∧
    p < 300 ∧
    (∀ (q : ℕ), 
      Nat.Prime q → 
      (∃ (m : ℕ), Nat.Prime m ∧ fermat_like m = q) → 
      q < 300 → 
      q ≤ p) ∧
    p = 5 :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_fermat_like_under_300_l3828_382864


namespace NUMINAMATH_CALUDE_inequality_solution_l3828_382894

/-- A quadratic function f(x) = ax^2 - bx + c where f(x) > 0 for x in (1, 3) -/
def f (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 - b * x + c

/-- The solution set of f(x) > 0 is (1, 3) -/
def f_positive_interval (a b c : ℝ) : Prop :=
  ∀ x, f a b c x > 0 ↔ 1 < x ∧ x < 3

theorem inequality_solution (a b c : ℝ) (h : f_positive_interval a b c) :
  ∀ t : ℝ, f a b c (|t| + 8) < f a b c (2 + t^2) ↔ -3 < t ∧ t < 3 ∧ t ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l3828_382894


namespace NUMINAMATH_CALUDE_fraction_addition_l3828_382820

theorem fraction_addition (m n : ℚ) (h : m / n = 3 / 7) : (m + n) / n = 10 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l3828_382820


namespace NUMINAMATH_CALUDE_function_composition_l3828_382834

/-- Given f(x) = 2x + 3 and g(x + 2) = f(x), prove that g(x) = 2x - 1 -/
theorem function_composition (f g : ℝ → ℝ) 
  (hf : ∀ x, f x = 2 * x + 3)
  (hg : ∀ x, g (x + 2) = f x) :
  ∀ x, g x = 2 * x - 1 := by sorry

end NUMINAMATH_CALUDE_function_composition_l3828_382834


namespace NUMINAMATH_CALUDE_smallest_sum_of_reciprocal_sum_l3828_382802

theorem smallest_sum_of_reciprocal_sum (x y : ℕ+) 
  (h1 : x ≠ y) 
  (h2 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 10) : 
  (∀ a b : ℕ+, a ≠ b → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 10 → (x : ℤ) + y ≤ (a : ℤ) + b) ∧ 
  ∃ p q : ℕ+, p ≠ q ∧ (1 : ℚ) / p + (1 : ℚ) / q = (1 : ℚ) / 10 ∧ (p : ℤ) + q = 45 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_reciprocal_sum_l3828_382802


namespace NUMINAMATH_CALUDE_power_calculation_l3828_382808

theorem power_calculation : 7^3 - 5*(6^2) + 2^4 = 179 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l3828_382808


namespace NUMINAMATH_CALUDE_jills_net_salary_l3828_382858

/-- Calculates the net monthly salary given the discretionary income percentage and the amount left after allocations -/
def calculate_net_salary (discretionary_income_percentage : ℚ) (amount_left : ℚ) : ℚ :=
  (amount_left / (discretionary_income_percentage * (1 - 0.3 - 0.2 - 0.35))) * 100

/-- Proves that under the given conditions, Jill's net monthly salary is $3700 -/
theorem jills_net_salary :
  let discretionary_income_percentage : ℚ := 1/5
  let amount_left : ℚ := 111
  calculate_net_salary discretionary_income_percentage amount_left = 3700 := by
  sorry

#eval calculate_net_salary (1/5) 111

end NUMINAMATH_CALUDE_jills_net_salary_l3828_382858


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3828_382873

def A : Set Nat := {3, 5, 6, 8}
def B : Set Nat := {4, 5, 7, 8}

theorem intersection_of_A_and_B : A ∩ B = {5, 8} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3828_382873


namespace NUMINAMATH_CALUDE_zero_division_not_always_zero_l3828_382805

theorem zero_division_not_always_zero : ¬ (∀ a : ℝ, a ≠ 0 → 0 / a = 0) :=
sorry

end NUMINAMATH_CALUDE_zero_division_not_always_zero_l3828_382805


namespace NUMINAMATH_CALUDE_unique_prime_sum_diff_l3828_382861

theorem unique_prime_sum_diff : ∃! p : ℕ, 
  Prime p ∧ 
  (∃ q s r t : ℕ, Prime q ∧ Prime s ∧ Prime r ∧ Prime t ∧ 
    p = q + s ∧ p = r - t) ∧ 
  p = 5 := by
sorry

end NUMINAMATH_CALUDE_unique_prime_sum_diff_l3828_382861


namespace NUMINAMATH_CALUDE_sum_of_extremes_even_integers_l3828_382832

theorem sum_of_extremes_even_integers (n : ℕ) (z : ℚ) (h_odd : Odd n) (h_pos : 0 < n) :
  ∃ b : ℤ,
    (∀ i : ℕ, i < n → Even (b + 2 * ↑i)) ∧
    z = (↑n * b + ↑n * (↑n - 1)) / ↑n →
    (b + (b + 2 * ↑(n - 1))) = 2 * ↑⌊z⌋ :=
by sorry

end NUMINAMATH_CALUDE_sum_of_extremes_even_integers_l3828_382832


namespace NUMINAMATH_CALUDE_fruit_weights_l3828_382870

/-- Represents the fruits on the table -/
inductive Fruit
| orange
| banana
| mandarin
| peach
| apple

/-- Assigns weights to fruits -/
def weight : Fruit → ℕ
| Fruit.orange => 280
| Fruit.banana => 170
| Fruit.mandarin => 100
| Fruit.peach => 200
| Fruit.apple => 150

/-- The set of all possible weights -/
def weights : Set ℕ := {100, 150, 170, 200, 280}

theorem fruit_weights :
  (∀ f : Fruit, weight f ∈ weights) ∧
  (weight Fruit.peach < weight Fruit.orange) ∧
  (weight Fruit.apple < weight Fruit.banana) ∧
  (weight Fruit.banana < weight Fruit.peach) ∧
  (weight Fruit.mandarin < weight Fruit.banana) ∧
  (weight Fruit.apple + weight Fruit.banana > weight Fruit.orange) ∧
  (∀ w : ℕ, w ∈ weights → ∃! f : Fruit, weight f = w) :=
by sorry

end NUMINAMATH_CALUDE_fruit_weights_l3828_382870


namespace NUMINAMATH_CALUDE_kilee_age_l3828_382811

/-- 
Given:
- Cornelia is currently 80 years old
- In 10 years, Cornelia will be three times as old as Kilee
Prove that Kilee's current age is 20 years
-/
theorem kilee_age (cornelia_age : ℕ) (kilee_age : ℕ) : 
  cornelia_age = 80 →
  cornelia_age + 10 = 3 * (kilee_age + 10) →
  kilee_age = 20 := by
sorry

end NUMINAMATH_CALUDE_kilee_age_l3828_382811


namespace NUMINAMATH_CALUDE_commercial_break_duration_l3828_382810

theorem commercial_break_duration 
  (num_five_min_commercials : ℕ) 
  (num_two_min_commercials : ℕ) 
  (five_min_duration : ℕ) 
  (two_min_duration : ℕ) : 
  num_five_min_commercials = 3 → 
  num_two_min_commercials = 11 → 
  five_min_duration = 5 → 
  two_min_duration = 2 → 
  num_five_min_commercials * five_min_duration + 
  num_two_min_commercials * two_min_duration = 37 :=
by
  sorry

#check commercial_break_duration

end NUMINAMATH_CALUDE_commercial_break_duration_l3828_382810


namespace NUMINAMATH_CALUDE_f_monotonicity_and_inequality_l3828_382814

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a / (Real.exp 1 * x)

theorem f_monotonicity_and_inequality (a : ℝ) :
  (∀ x y, 0 < x ∧ x < y → (a ≤ 0 → f a x < f a y)) ∧
  (∀ x y, 0 < x ∧ x < y ∧ y < a / Real.exp 1 → (a > 0 → f a y < f a x)) ∧
  (∀ x y, a / Real.exp 1 < x ∧ x < y → (a > 0 → f a x < f a y)) ∧
  (∀ x, x > 0 → f 2 x > Real.exp (-x)) :=
sorry

end

end NUMINAMATH_CALUDE_f_monotonicity_and_inequality_l3828_382814


namespace NUMINAMATH_CALUDE_equation_solution_unique_l3828_382847

theorem equation_solution_unique :
  ∃! (x y : ℝ), x ≥ 2 ∧ y ≥ 1 ∧
  36 * Real.sqrt (x - 2) + 4 * Real.sqrt (y - 1) = 28 - 4 * Real.sqrt (x - 2) - Real.sqrt (y - 1) ∧
  x = 5 ∧ y = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_unique_l3828_382847


namespace NUMINAMATH_CALUDE_frequency_converges_to_probability_l3828_382857

-- Define a random experiment
def RandomExperiment : Type := Unit

-- Define an event in the experiment
def Event (e : RandomExperiment) : Type := Unit

-- Define the probability of an event
def probability (e : RandomExperiment) (A : Event e) : ℝ := sorry

-- Define the frequency of an event after n trials
def frequency (e : RandomExperiment) (A : Event e) (n : ℕ) : ℝ := sorry

-- Theorem: As the number of trials approaches infinity, 
-- the frequency converges to the probability
theorem frequency_converges_to_probability 
  (e : RandomExperiment) (A : Event e) : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, 
  |frequency e A n - probability e A| < ε :=
sorry

end NUMINAMATH_CALUDE_frequency_converges_to_probability_l3828_382857


namespace NUMINAMATH_CALUDE_parallelogram_base_l3828_382895

/-- The base of a parallelogram with area 960 square cm and height 16 cm is 60 cm -/
theorem parallelogram_base (area : ℝ) (height : ℝ) (base : ℝ) : 
  area = 960 ∧ height = 16 ∧ area = base * height → base = 60 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_base_l3828_382895


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3828_382877

def sequence_u (n : ℕ) : ℝ :=
  sorry

theorem sum_of_coefficients :
  (∃ (a b c : ℝ), ∀ (n : ℕ), sequence_u n = a * n^2 + b * n + c) →
  (sequence_u 1 = 7) →
  (∀ (n : ℕ), sequence_u (n + 1) - sequence_u n = 5 + 3 * (n - 1)) →
  (∃ (a b c : ℝ), 
    (∀ (n : ℕ), sequence_u n = a * n^2 + b * n + c) ∧
    (a + b + c = 7)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3828_382877


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3828_382809

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (b : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_a2 : a 2 = 2)
  (h_a4 : a 4 = 4)
  (h_b : ∀ n, b n = 2^(a n)) :
  (∀ n, a n = n) ∧ (b 1 + b 2 + b 3 + b 4 + b 5 = 62) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3828_382809


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l3828_382855

/-- The equation of a hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  y^2 / 4 - x^2 / 9 = 1

/-- The equation of the asymptotes -/
def asymptote_equation (x y : ℝ) : Prop :=
  y = 2/3 * x ∨ y = -2/3 * x

/-- Theorem: The asymptotes of the given hyperbola are y = ±(2/3)x -/
theorem hyperbola_asymptotes :
  ∀ x y : ℝ, hyperbola_equation x y → asymptote_equation x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l3828_382855


namespace NUMINAMATH_CALUDE_order_of_exponents_l3828_382886

theorem order_of_exponents : 5^(1/5) > 0.5^(1/5) ∧ 0.5^(1/5) > 0.5^2 := by
  sorry

end NUMINAMATH_CALUDE_order_of_exponents_l3828_382886


namespace NUMINAMATH_CALUDE_line_symmetrical_to_itself_l3828_382874

/-- A line in the 2D plane represented by its slope-intercept form y = mx + b -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- The line of symmetry -/
def lineOfSymmetry : Line :=
  { slope := 1, intercept := -2 }

/-- The original line -/
def originalLine : Line :=
  { slope := 3, intercept := 3 }

/-- Find the symmetric point of a given point with respect to the line of symmetry -/
def symmetricPoint (p : Point) : Point :=
  { x := p.x, y := p.y }

theorem line_symmetrical_to_itself :
  ∀ (p : Point), pointOnLine p originalLine →
  pointOnLine (symmetricPoint p) originalLine :=
sorry

end NUMINAMATH_CALUDE_line_symmetrical_to_itself_l3828_382874


namespace NUMINAMATH_CALUDE_exterior_bisector_theorem_l3828_382872

/-- Represents a triangle with angles given in degrees -/
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  sum_180 : angle1 + angle2 + angle3 = 180

/-- The triangle formed by the exterior angle bisectors -/
def exterior_bisector_triangle : Triangle :=
  { angle1 := 52
    angle2 := 61
    angle3 := 67
    sum_180 := by norm_num }

/-- The original triangle whose exterior angle bisectors form the given triangle -/
def original_triangle : Triangle :=
  { angle1 := 76
    angle2 := 58
    angle3 := 46
    sum_180 := by norm_num }

theorem exterior_bisector_theorem (t : Triangle) :
  t = exterior_bisector_triangle →
  ∃ (orig : Triangle), orig = original_triangle ∧
    (90 - orig.angle2 / 2) + (90 - orig.angle3 / 2) = t.angle1 ∧
    (90 - orig.angle1 / 2) + (90 - orig.angle3 / 2) = t.angle2 ∧
    (90 - orig.angle1 / 2) + (90 - orig.angle2 / 2) = t.angle3 :=
by
  sorry

end NUMINAMATH_CALUDE_exterior_bisector_theorem_l3828_382872


namespace NUMINAMATH_CALUDE_factory_wage_problem_l3828_382817

/-- Proves that the hourly rate for the remaining employees is $17 given the problem conditions -/
theorem factory_wage_problem (total_employees : ℕ) (employees_at_12 : ℕ) (employees_at_14 : ℕ)
  (shift_length : ℕ) (total_cost : ℕ) :
  total_employees = 300 →
  employees_at_12 = 200 →
  employees_at_14 = 40 →
  shift_length = 8 →
  total_cost = 31840 →
  let remaining_employees := total_employees - (employees_at_12 + employees_at_14)
  let remaining_cost := total_cost - (employees_at_12 * 12 * shift_length + employees_at_14 * 14 * shift_length)
  remaining_cost / (remaining_employees * shift_length) = 17 := by
  sorry

#check factory_wage_problem

end NUMINAMATH_CALUDE_factory_wage_problem_l3828_382817


namespace NUMINAMATH_CALUDE_midpoint_coordinate_product_l3828_382829

/-- The product of the coordinates of the midpoint of a line segment
    with endpoints (4, -1) and (-8, 7) is -6. -/
theorem midpoint_coordinate_product : 
  let x1 : ℝ := 4
  let y1 : ℝ := -1
  let x2 : ℝ := -8
  let y2 : ℝ := 7
  let midpoint_x : ℝ := (x1 + x2) / 2
  let midpoint_y : ℝ := (y1 + y2) / 2
  midpoint_x * midpoint_y = -6 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_product_l3828_382829


namespace NUMINAMATH_CALUDE_constant_term_equality_l3828_382842

theorem constant_term_equality (a : ℝ) : 
  (∃ k : ℕ, (Nat.choose 9 k) * 2^k = 84 * 64 ∧ 
   18 - 3 * k = 0) →
  (∃ r : ℕ, (Nat.choose 9 r) * a^r = 84 * 64 ∧
   9 - 3 * r = 0) →
  a = 4 := by sorry

end NUMINAMATH_CALUDE_constant_term_equality_l3828_382842


namespace NUMINAMATH_CALUDE_interval_equivalence_l3828_382822

def interval_condition (x : ℝ) : Prop :=
  1 < 3 * x ∧ 3 * x < 2 ∧ 1 < 5 * x ∧ 5 * x < 2

theorem interval_equivalence : 
  {x : ℝ | interval_condition x} = {x : ℝ | 1/3 < x ∧ x < 2/5} :=
by sorry

end NUMINAMATH_CALUDE_interval_equivalence_l3828_382822


namespace NUMINAMATH_CALUDE_kaleb_restaurant_bill_l3828_382803

/-- Calculates the total bill for a group at Kaleb's Restaurant -/
def total_bill (num_adults : ℕ) (num_children : ℕ) (adult_meal_cost : ℕ) (child_meal_cost : ℕ) (soda_cost : ℕ) : ℕ :=
  num_adults * adult_meal_cost + num_children * child_meal_cost + (num_adults + num_children) * soda_cost

/-- Theorem: The total bill for a group of 6 adults and 2 children at Kaleb's Restaurant is $60 -/
theorem kaleb_restaurant_bill :
  total_bill 6 2 6 4 2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_kaleb_restaurant_bill_l3828_382803


namespace NUMINAMATH_CALUDE_point_same_side_l3828_382898

def line (x y : ℝ) : ℝ := x + y - 1

def same_side (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (line x₁ y₁ > 0 ∧ line x₂ y₂ > 0) ∨ (line x₁ y₁ < 0 ∧ line x₂ y₂ < 0)

theorem point_same_side : same_side 1 2 (-1) 3 := by
  sorry

end NUMINAMATH_CALUDE_point_same_side_l3828_382898


namespace NUMINAMATH_CALUDE_jason_percentage_more_than_zachary_l3828_382891

/-- Proves that Jason received 30% more money than Zachary from selling video games -/
theorem jason_percentage_more_than_zachary 
  (zachary_games : ℕ) 
  (zachary_price : ℚ) 
  (ryan_extra : ℚ) 
  (total_amount : ℚ) 
  (h1 : zachary_games = 40)
  (h2 : zachary_price = 5)
  (h3 : ryan_extra = 50)
  (h4 : total_amount = 770)
  (h5 : zachary_games * zachary_price + 2 * (zachary_games * zachary_price + ryan_extra) / 2 = total_amount) :
  (((zachary_games * zachary_price + ryan_extra) / 2 - zachary_games * zachary_price) / (zachary_games * zachary_price)) * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_jason_percentage_more_than_zachary_l3828_382891


namespace NUMINAMATH_CALUDE_grasshopper_trap_l3828_382849

/-- Represents the position of the grasshopper on the number line -/
structure Position :=
  (value : ℚ)

/-- Represents a move of the grasshopper -/
inductive Move
  | Left  : ℕ+ → Move
  | Right : ℕ+ → Move

/-- Defines the result of applying a move to a position -/
def apply_move (p : Position) (m : Move) : Position :=
  match m with
  | Move.Left n  => ⟨p.value - n.val⟩
  | Move.Right n => ⟨p.value + n.val⟩

/-- Theorem stating that for any binary-rational position between 0 and 1,
    there exists a sequence of moves that leads to either 0 or 1 -/
theorem grasshopper_trap (a k : ℕ) (h1 : 0 < a) (h2 : a < 2^k) :
  ∃ (moves : List Move), 
    let final_pos := (moves.foldl apply_move ⟨a / 2^k⟩).value
    final_pos = 0 ∨ final_pos = 1 :=
sorry

end NUMINAMATH_CALUDE_grasshopper_trap_l3828_382849


namespace NUMINAMATH_CALUDE_senior_ticket_cost_l3828_382887

theorem senior_ticket_cost 
  (total_tickets : ℕ) 
  (adult_price : ℕ) 
  (total_receipts : ℕ) 
  (senior_tickets : ℕ) 
  (h1 : total_tickets = 510) 
  (h2 : adult_price = 21) 
  (h3 : total_receipts = 8748) 
  (h4 : senior_tickets = 327) :
  ∃ (senior_price : ℕ), 
    senior_price * senior_tickets + adult_price * (total_tickets - senior_tickets) = total_receipts ∧ 
    senior_price = 15 := by
sorry

end NUMINAMATH_CALUDE_senior_ticket_cost_l3828_382887


namespace NUMINAMATH_CALUDE_power_of_seven_mod_thousand_l3828_382851

theorem power_of_seven_mod_thousand : 7^1984 ≡ 401 [ZMOD 1000] := by sorry

end NUMINAMATH_CALUDE_power_of_seven_mod_thousand_l3828_382851


namespace NUMINAMATH_CALUDE_b_plus_c_equals_six_l3828_382859

theorem b_plus_c_equals_six (a b c d : ℝ) 
  (h1 : a + b = 5) 
  (h2 : c + d = 3) 
  (h3 : a + d = 2) : 
  b + c = 6 := by
  sorry

end NUMINAMATH_CALUDE_b_plus_c_equals_six_l3828_382859


namespace NUMINAMATH_CALUDE_dan_gave_41_cards_l3828_382837

/-- Given the initial number of cards, the number of cards bought, and the final number of cards,
    calculate the number of cards given by Dan. -/
def cards_given_by_dan (initial_cards : ℕ) (bought_cards : ℕ) (final_cards : ℕ) : ℕ :=
  final_cards - initial_cards - bought_cards

/-- Theorem stating that Dan gave Sally 41 cards -/
theorem dan_gave_41_cards :
  cards_given_by_dan 27 20 88 = 41 := by
  sorry

end NUMINAMATH_CALUDE_dan_gave_41_cards_l3828_382837


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a12_l3828_382890

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a12 (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_sum : a 7 + a 9 = 16)
  (h_a4 : a 4 = 1) :
  a 12 = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a12_l3828_382890


namespace NUMINAMATH_CALUDE_length_of_BD_l3828_382831

-- Define the triangles and their properties
def right_triangle_ABC (b c : ℝ) : Prop :=
  ∃ (A B C : ℝ × ℝ), 
    (B.1 - A.1) * (C.2 - A.2) = (C.1 - A.1) * (B.2 - A.2) ∧ 
    (C.1 - B.1)^2 + (C.2 - B.2)^2 = b^2 ∧
    (A.1 - C.1)^2 + (A.2 - C.2)^2 = c^2

def right_triangle_ABD (b c : ℝ) : Prop :=
  ∃ (A B D : ℝ × ℝ), 
    (B.1 - A.1) * (D.2 - A.2) = (D.1 - A.1) * (B.2 - A.2) ∧
    (A.1 - D.1)^2 + (A.2 - D.2)^2 = 9 ∧
    (B.1 - A.1)^2 + (B.2 - A.2)^2 = b^2 + c^2

-- The theorem to prove
theorem length_of_BD (b c : ℝ) (h1 : b > 0) (h2 : c > 0) 
  (h3 : right_triangle_ABC b c) (h4 : right_triangle_ABD b c) :
  ∃ (B D : ℝ × ℝ), (B.1 - D.1)^2 + (B.2 - D.2)^2 = b^2 + c^2 - 9 := by
  sorry

end NUMINAMATH_CALUDE_length_of_BD_l3828_382831


namespace NUMINAMATH_CALUDE_right_triangle_area_l3828_382827

theorem right_triangle_area (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  a^2 + b^2 = 10^2 → a + b + 10 = 24 → (1/2) * a * b = 24 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3828_382827


namespace NUMINAMATH_CALUDE_sum_of_powers_divisible_by_ten_l3828_382879

theorem sum_of_powers_divisible_by_ten (n : ℕ) (h : ¬ (4 ∣ n)) :
  10 ∣ (1^n + 2^n + 3^n + 4^n) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_divisible_by_ten_l3828_382879


namespace NUMINAMATH_CALUDE_pauls_chickens_l3828_382852

theorem pauls_chickens (neighbor_sale quick_sale remaining : ℕ) 
  (h1 : neighbor_sale = 12)
  (h2 : quick_sale = 25)
  (h3 : remaining = 43) :
  neighbor_sale + quick_sale + remaining = 80 :=
by sorry

end NUMINAMATH_CALUDE_pauls_chickens_l3828_382852


namespace NUMINAMATH_CALUDE_gcd_of_256_162_720_l3828_382844

theorem gcd_of_256_162_720 : Nat.gcd 256 (Nat.gcd 162 720) = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_256_162_720_l3828_382844


namespace NUMINAMATH_CALUDE_defective_units_shipped_l3828_382896

theorem defective_units_shipped (total_units : ℝ) (h1 : total_units > 0) : 
  let defective_units := 0.07 * total_units
  let defective_shipped := 0.0035 * total_units
  (defective_shipped / defective_units) * 100 = 5 := by
sorry

end NUMINAMATH_CALUDE_defective_units_shipped_l3828_382896


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3828_382819

theorem simplify_and_evaluate (x : ℝ) (h : x = 2) : 
  (1 + 1 / (x + 1)) / ((x + 2) / (x^2 - 1)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3828_382819


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l3828_382892

/-- A three-digit number is represented as 100 * a + 10 * b + c, where a, b, c are single digits -/
def three_digit_number (a b c : ℕ) : Prop :=
  a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10

/-- The number is 12 times the sum of its digits -/
def twelve_times_sum (a b c : ℕ) : Prop :=
  100 * a + 10 * b + c = 12 * (a + b + c)

theorem unique_three_digit_number :
  ∃! (a b c : ℕ), three_digit_number a b c ∧ twelve_times_sum a b c ∧
    100 * a + 10 * b + c = 108 :=
by sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l3828_382892


namespace NUMINAMATH_CALUDE_color_combinations_l3828_382850

/-- Represents the number of color choices for each dot -/
def num_colors : ℕ := 3

/-- Represents the number of ways to color a single triangle -/
def ways_per_triangle : ℕ := 6

/-- Represents the number of triangles in the figure -/
def num_triangles : ℕ := 3

/-- Represents the total number of dots in the figure -/
def total_dots : ℕ := 10

/-- Theorem stating the number of ways to color the figure -/
theorem color_combinations : 
  (ways_per_triangle ^ num_triangles : ℕ) = 216 :=
sorry

end NUMINAMATH_CALUDE_color_combinations_l3828_382850


namespace NUMINAMATH_CALUDE_parabola_c_value_l3828_382880

/-- A parabola passing through two given points has a specific c-value -/
theorem parabola_c_value (b c : ℝ) : 
  (∀ x, 2 = x^2 + b*x + c → x = 1 ∨ x = 5) →
  c = 7 := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l3828_382880


namespace NUMINAMATH_CALUDE_parabola_intersection_count_l3828_382821

/-- The parabola is defined by the function f(x) = 2x^2 - 4x + 1 --/
def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 1

/-- The number of intersection points between the parabola and the coordinate axes --/
def intersection_count : ℕ := 3

/-- Theorem stating that the parabola intersects the coordinate axes at exactly 3 points --/
theorem parabola_intersection_count :
  (∃! y, y = f 0) ∧ 
  (∃! x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) ∧
  intersection_count = 3 :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_count_l3828_382821


namespace NUMINAMATH_CALUDE_train_speed_l3828_382860

/-- The speed of a train given its length and time to cross a fixed point. -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 280) (h2 : time = 20) :
  length / time = 14 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l3828_382860


namespace NUMINAMATH_CALUDE_theorem_1_theorem_2_l3828_382884

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the functional equation
def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x + f y = 2 * f ((x + y) / 2) * f ((x - y) / 2)

-- Theorem 1: If f(1) = 1/2, then f(2) = -1/2
theorem theorem_1 (h : functional_equation f) (h1 : f 1 = 1/2) : f 2 = -1/2 := by
  sorry

-- Theorem 2: If f(1) = 0, then f(11/2) + f(15/2) + f(19/2) + ... + f(2019/2) + f(2023/2) = 0
theorem theorem_2 (h : functional_equation f) (h1 : f 1 = 0) :
  f (11/2) + f (15/2) + f (19/2) + f (2019/2) + f (2023/2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_theorem_1_theorem_2_l3828_382884


namespace NUMINAMATH_CALUDE_smallest_integer_quadratic_inequality_l3828_382828

theorem smallest_integer_quadratic_inequality :
  ∃ (n : ℤ), n^2 - 15*n + 56 ≤ 0 ∧ ∀ (m : ℤ), m^2 - 15*m + 56 ≤ 0 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_quadratic_inequality_l3828_382828


namespace NUMINAMATH_CALUDE_coefficient_x6_in_x_plus_2_to_8_l3828_382813

theorem coefficient_x6_in_x_plus_2_to_8 :
  (Finset.range 9).sum (fun k => (Nat.choose 8 k * 2^(8 - k)) * (if k = 6 then 1 else 0)) = 112 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x6_in_x_plus_2_to_8_l3828_382813


namespace NUMINAMATH_CALUDE_log_xy_value_l3828_382867

-- Define a real-valued logarithm function
noncomputable def log : ℝ → ℝ := sorry

-- State the theorem
theorem log_xy_value (x y : ℝ) (h1 : log (x^2 * y^3) = 2) (h2 : log (x^3 * y^2) = 2) :
  log (x * y) = 4/5 := by sorry

end NUMINAMATH_CALUDE_log_xy_value_l3828_382867


namespace NUMINAMATH_CALUDE_largest_exponent_inequality_l3828_382881

theorem largest_exponent_inequality (n : ℕ) : 64^8 > 4^n ↔ n ≤ 23 := by
  sorry

end NUMINAMATH_CALUDE_largest_exponent_inequality_l3828_382881


namespace NUMINAMATH_CALUDE_six_balls_two_boxes_l3828_382833

/-- The number of ways to distribute n indistinguishable balls into 2 indistinguishable boxes -/
def distribution_count (n : ℕ) : ℕ := sorry

/-- Theorem: The number of ways to distribute 6 indistinguishable balls into 2 indistinguishable boxes is 4 -/
theorem six_balls_two_boxes : distribution_count 6 = 4 := by sorry

end NUMINAMATH_CALUDE_six_balls_two_boxes_l3828_382833


namespace NUMINAMATH_CALUDE_complement_union_and_intersection_l3828_382845

-- Define sets A and B
def A : Set ℝ := {x : ℝ | 3 ≤ x ∧ x < 5}
def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 9}

-- Define the complement of a set in ℝ
def complement (S : Set ℝ) : Set ℝ := {x : ℝ | x ∉ S}

-- State the theorem
theorem complement_union_and_intersection :
  (complement (A ∪ B) = {x : ℝ | x ≤ 2 ∨ x ≥ 9}) ∧
  (complement (A ∩ B) = {x : ℝ | x < 3 ∨ x ≥ 5}) := by
  sorry

end NUMINAMATH_CALUDE_complement_union_and_intersection_l3828_382845


namespace NUMINAMATH_CALUDE_inequality_proof_l3828_382830

theorem inequality_proof (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (a^2 - b^2) / (a^2 + b^2) > (a - b) / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3828_382830


namespace NUMINAMATH_CALUDE_pizza_topping_distribution_l3828_382863

/-- Pizza topping distribution problem -/
theorem pizza_topping_distribution 
  (pepperoni : ℕ) 
  (ham : ℕ) 
  (sausage : ℕ) 
  (slices : ℕ) :
  pepperoni = 30 →
  ham = 2 * pepperoni →
  sausage = pepperoni + 12 →
  slices = 6 →
  (pepperoni + ham + sausage) / slices = 22 := by
  sorry

#check pizza_topping_distribution

end NUMINAMATH_CALUDE_pizza_topping_distribution_l3828_382863


namespace NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l3828_382823

/-- The complex number z = i(1+i) is located in the second quadrant of the complex plane. -/
theorem complex_number_in_second_quadrant : 
  let z : ℂ := Complex.I * (1 + Complex.I)
  (z.re < 0) ∧ (z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l3828_382823


namespace NUMINAMATH_CALUDE_tims_score_is_2352_l3828_382818

-- Define the first 8 prime numbers
def first_8_primes : List Nat := [2, 3, 5, 7, 11, 13, 17, 19]

-- Define the product of the first 8 prime numbers
def prime_product : Nat := first_8_primes.prod

-- Define the sum of digits function
def sum_of_digits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

-- Define N as the sum of digits in the product of the first 8 prime numbers
def N : Nat := sum_of_digits prime_product

-- Define Tim's score as the sum of the first N even numbers
def tims_score : Nat := N * (N + 1)

-- The theorem to prove
theorem tims_score_is_2352 : tims_score = 2352 := by sorry

end NUMINAMATH_CALUDE_tims_score_is_2352_l3828_382818


namespace NUMINAMATH_CALUDE_computers_produced_per_month_l3828_382865

/-- Represents the number of computers produced per 30-minute interval -/
def computers_per_interval : ℕ := 4

/-- Represents the number of days in a month -/
def days_per_month : ℕ := 28

/-- Represents the number of 30-minute intervals in a day -/
def intervals_per_day : ℕ := 48

/-- Calculates the total number of computers produced in a month -/
def computers_per_month : ℕ :=
  computers_per_interval * days_per_month * intervals_per_day

/-- Theorem stating that the number of computers produced per month is 5376 -/
theorem computers_produced_per_month :
  computers_per_month = 5376 := by sorry

end NUMINAMATH_CALUDE_computers_produced_per_month_l3828_382865


namespace NUMINAMATH_CALUDE_tims_soda_cans_l3828_382848

theorem tims_soda_cans (x : ℕ) : 
  x - 6 + (x - 6) / 2 = 24 → x = 22 := by sorry

end NUMINAMATH_CALUDE_tims_soda_cans_l3828_382848


namespace NUMINAMATH_CALUDE_donuts_for_class_l3828_382843

theorem donuts_for_class (total_students : ℕ) (donut_likers_percentage : ℚ) (donuts_per_student : ℕ) : 
  total_students = 30 →
  donut_likers_percentage = 4/5 →
  donuts_per_student = 2 →
  (↑total_students * donut_likers_percentage * ↑donuts_per_student) / 12 = 4 := by
  sorry

end NUMINAMATH_CALUDE_donuts_for_class_l3828_382843
