import Mathlib

namespace NUMINAMATH_CALUDE_exactly_one_fail_probability_l463_46338

/-- The probability that exactly one item fails the inspection when one item is taken from each of two types of products with pass rates of 0.90 and 0.95 respectively is 0.14. -/
theorem exactly_one_fail_probability (pass_rate1 pass_rate2 : ℝ) 
  (h1 : pass_rate1 = 0.90) (h2 : pass_rate2 = 0.95) : 
  pass_rate1 * (1 - pass_rate2) + (1 - pass_rate1) * pass_rate2 = 0.14 := by
  sorry

end NUMINAMATH_CALUDE_exactly_one_fail_probability_l463_46338


namespace NUMINAMATH_CALUDE_sum_interior_angles_theorem_l463_46335

/-- The sum of interior angles of an n-gon -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- Theorem: The sum of the interior angles of any n-gon is (n-2) * 180° -/
theorem sum_interior_angles_theorem (n : ℕ) (h : n ≥ 3) :
  sum_interior_angles n = (n - 2) * 180 := by
  sorry

end NUMINAMATH_CALUDE_sum_interior_angles_theorem_l463_46335


namespace NUMINAMATH_CALUDE_jonahs_calorie_burn_l463_46316

/-- Calculates the difference in calories burned between two running durations -/
def calorie_difference (rate : ℕ) (duration1 duration2 : ℕ) : ℕ :=
  rate * duration2 - rate * duration1

/-- The problem statement -/
theorem jonahs_calorie_burn :
  let rate : ℕ := 30
  let short_duration : ℕ := 2
  let long_duration : ℕ := 5
  calorie_difference rate short_duration long_duration = 90 := by
  sorry

end NUMINAMATH_CALUDE_jonahs_calorie_burn_l463_46316


namespace NUMINAMATH_CALUDE_distance_between_cities_l463_46352

/-- The distance between City A and City B in miles -/
def distance : ℝ := 427.5

/-- The travel time from City A to City B in hours -/
def time_A_to_B : ℝ := 6

/-- The travel time from City B to City A in hours -/
def time_B_to_A : ℝ := 4.5

/-- The time saved on each trip in hours -/
def time_saved : ℝ := 0.5

/-- The average speed for the round trip if time were saved, in miles per hour -/
def average_speed : ℝ := 90

theorem distance_between_cities :
  distance = 427.5 ∧
  (2 * distance) / (time_A_to_B + time_B_to_A - 2 * time_saved) = average_speed :=
sorry

end NUMINAMATH_CALUDE_distance_between_cities_l463_46352


namespace NUMINAMATH_CALUDE_edward_lost_lives_l463_46311

theorem edward_lost_lives (initial_lives : ℕ) (remaining_lives : ℕ) (lost_lives : ℕ) : 
  initial_lives = 15 → remaining_lives = 7 → lost_lives = initial_lives - remaining_lives → lost_lives = 8 := by
  sorry

end NUMINAMATH_CALUDE_edward_lost_lives_l463_46311


namespace NUMINAMATH_CALUDE_abs_gt_one_iff_square_minus_one_gt_zero_l463_46350

theorem abs_gt_one_iff_square_minus_one_gt_zero :
  ∀ x : ℝ, |x| > 1 ↔ x^2 - 1 > 0 := by sorry

end NUMINAMATH_CALUDE_abs_gt_one_iff_square_minus_one_gt_zero_l463_46350


namespace NUMINAMATH_CALUDE_snow_probability_l463_46317

theorem snow_probability (p1 p2 p3 : ℚ) (n1 n2 n3 : ℕ) : 
  p1 = 1/3 →
  p2 = 1/4 →
  p3 = 1/2 →
  n1 = 3 →
  n2 = 4 →
  n3 = 3 →
  1 - (1 - p1)^n1 * (1 - p2)^n2 * (1 - p3)^n3 = 2277/2304 := by
sorry

end NUMINAMATH_CALUDE_snow_probability_l463_46317


namespace NUMINAMATH_CALUDE_smallest_marble_count_l463_46369

def is_valid_marble_count (n : ℕ) : Prop :=
  n > 1 ∧ n % 5 = 2 ∧ n % 7 = 2 ∧ n % 9 = 2

theorem smallest_marble_count :
  ∃ (n : ℕ), is_valid_marble_count n ∧
  ∀ (m : ℕ), is_valid_marble_count m → n ≤ m :=
by
  use 317
  sorry

end NUMINAMATH_CALUDE_smallest_marble_count_l463_46369


namespace NUMINAMATH_CALUDE_five_numbers_product_1000_l463_46344

theorem five_numbers_product_1000 :
  ∃ (a b c d e : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
                     b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
                     c ≠ d ∧ c ≠ e ∧
                     d ≠ e ∧
                     a * b * c * d * e = 1000 :=
by sorry

end NUMINAMATH_CALUDE_five_numbers_product_1000_l463_46344


namespace NUMINAMATH_CALUDE_line_parallel_to_x_axis_l463_46359

/-- 
A line through two points (x₁, y₁) and (x₂, y₂) is parallel to the x-axis 
if and only if y₁ = y₂.
-/
def parallel_to_x_axis (x₁ y₁ x₂ y₂ : ℝ) : Prop := y₁ = y₂

/-- 
The value of k for which the line through the points (3, 2k+1) and (8, 4k-5) 
is parallel to the x-axis.
-/
theorem line_parallel_to_x_axis (k : ℝ) : 
  parallel_to_x_axis 3 (2*k+1) 8 (4*k-5) ↔ k = 3 := by
  sorry

#check line_parallel_to_x_axis

end NUMINAMATH_CALUDE_line_parallel_to_x_axis_l463_46359


namespace NUMINAMATH_CALUDE_garden_walkway_area_l463_46378

/-- Calculates the area of walkways in a garden with vegetable beds -/
theorem garden_walkway_area (rows : Nat) (cols : Nat) (bed_width : Nat) (bed_height : Nat) (walkway_width : Nat) : 
  rows = 4 → cols = 3 → bed_width = 8 → bed_height = 3 → walkway_width = 2 →
  (rows * cols * bed_width * bed_height + 
   (rows + 1) * walkway_width * (cols * bed_width + (cols + 1) * walkway_width) + 
   rows * (cols + 1) * walkway_width * bed_height) - 
  (rows * cols * bed_width * bed_height) = 416 := by
sorry

end NUMINAMATH_CALUDE_garden_walkway_area_l463_46378


namespace NUMINAMATH_CALUDE_right_triangle_circumcenter_angles_l463_46347

theorem right_triangle_circumcenter_angles (α : Real) (h1 : α = 25 * π / 180) :
  let β := π / 2 - α
  let θ₁ := 2 * α
  let θ₂ := 2 * β
  θ₁ = 50 * π / 180 ∧ θ₂ = 130 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_circumcenter_angles_l463_46347


namespace NUMINAMATH_CALUDE_two_number_problem_l463_46337

theorem two_number_problem (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x < y) 
  (h4 : 4 * y = 6 * x) (h5 : x + y = 36) : y = 21.6 := by
  sorry

end NUMINAMATH_CALUDE_two_number_problem_l463_46337


namespace NUMINAMATH_CALUDE_lunch_change_calculation_l463_46379

/-- Calculates the change received when buying lunch items --/
theorem lunch_change_calculation (hamburger_cost onion_rings_cost smoothie_cost amount_paid : ℕ) :
  hamburger_cost = 4 →
  onion_rings_cost = 2 →
  smoothie_cost = 3 →
  amount_paid = 20 →
  amount_paid - (hamburger_cost + onion_rings_cost + smoothie_cost) = 11 := by
  sorry

end NUMINAMATH_CALUDE_lunch_change_calculation_l463_46379


namespace NUMINAMATH_CALUDE_nitin_rank_l463_46382

theorem nitin_rank (total_students : ℕ) (rank_from_last : ℕ) (rank_from_first : ℕ) : 
  total_students = 58 → rank_from_last = 34 → rank_from_first = 25 :=
by sorry

end NUMINAMATH_CALUDE_nitin_rank_l463_46382


namespace NUMINAMATH_CALUDE_eighth_grade_students_l463_46341

theorem eighth_grade_students (total : ℕ) (girls : ℕ) (boys : ℕ) : 
  total = 68 → 
  girls = 28 → 
  boys < 2 * girls → 
  boys = total - girls → 
  2 * girls - boys = 16 := by
sorry

end NUMINAMATH_CALUDE_eighth_grade_students_l463_46341


namespace NUMINAMATH_CALUDE_fiona_finished_tenth_l463_46318

/-- Represents a racer in the competition -/
inductive Racer
| Alice
| Ben
| Carlos
| Diana
| Emma
| Fiona

/-- The type of finishing positions -/
def Position := Fin 15

/-- The finishing order of the race -/
def FinishingOrder := Racer → Position

/-- Defines the relative positions of racers -/
def PlacesAhead (fo : FinishingOrder) (r1 r2 : Racer) (n : ℕ) : Prop :=
  (fo r1).val + n = (fo r2).val

/-- Defines the absolute position of a racer -/
def FinishedIn (fo : FinishingOrder) (r : Racer) (p : Position) : Prop :=
  fo r = p

theorem fiona_finished_tenth (fo : FinishingOrder) :
  PlacesAhead fo Racer.Emma Racer.Diana 4 →
  PlacesAhead fo Racer.Carlos Racer.Alice 2 →
  PlacesAhead fo Racer.Diana Racer.Ben 3 →
  PlacesAhead fo Racer.Carlos Racer.Fiona 3 →
  PlacesAhead fo Racer.Emma Racer.Fiona 2 →
  FinishedIn fo Racer.Ben ⟨7, by norm_num⟩ →
  FinishedIn fo Racer.Fiona ⟨10, by norm_num⟩ := by
  sorry

end NUMINAMATH_CALUDE_fiona_finished_tenth_l463_46318


namespace NUMINAMATH_CALUDE_tan_half_sum_angles_l463_46351

theorem tan_half_sum_angles (x y : ℝ) 
  (h1 : Real.cos x + Real.cos y = 3/5)
  (h2 : Real.sin x + Real.sin y = 8/17) : 
  Real.tan ((x + y)/2) = 40/51 := by
  sorry

end NUMINAMATH_CALUDE_tan_half_sum_angles_l463_46351


namespace NUMINAMATH_CALUDE_arcsin_equation_solutions_l463_46302

theorem arcsin_equation_solutions :
  let f (x : ℝ) := Real.arcsin (2 * x / Real.sqrt 15) + Real.arcsin (3 * x / Real.sqrt 15) = Real.arcsin (4 * x / Real.sqrt 15)
  let valid (x : ℝ) := abs (2 * x / Real.sqrt 15) ≤ 1 ∧ abs (3 * x / Real.sqrt 15) ≤ 1 ∧ abs (4 * x / Real.sqrt 15) ≤ 1
  ∀ x : ℝ, valid x → (f x ↔ x = 0 ∨ x = 15 / 16 ∨ x = -15 / 16) :=
by sorry

end NUMINAMATH_CALUDE_arcsin_equation_solutions_l463_46302


namespace NUMINAMATH_CALUDE_square_of_binomial_condition_l463_46396

theorem square_of_binomial_condition (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 9*x^2 - 18*x + a = (3*x + b)^2) → a = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_of_binomial_condition_l463_46396


namespace NUMINAMATH_CALUDE_inverse_iff_horizontal_line_test_l463_46304

-- Define a type for our functions
def Function2D := ℝ → ℝ

-- Define the horizontal line test
def passes_horizontal_line_test (f : Function2D) : Prop :=
  ∀ y : ℝ, ∀ x₁ x₂ : ℝ, f x₁ = y ∧ f x₂ = y → x₁ = x₂

-- Define what it means for a function to have an inverse
def has_inverse (f : Function2D) : Prop :=
  ∃ g : Function2D, (∀ x : ℝ, g (f x) = x) ∧ (∀ y : ℝ, f (g y) = y)

-- Theorem statement
theorem inverse_iff_horizontal_line_test (f : Function2D) :
  has_inverse f ↔ passes_horizontal_line_test f :=
sorry

end NUMINAMATH_CALUDE_inverse_iff_horizontal_line_test_l463_46304


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l463_46363

theorem quadratic_one_solution (q : ℝ) : 
  (q ≠ 0 ∧ ∃! x : ℝ, q * x^2 - 16 * x + 9 = 0) ↔ q = 64/9 := by sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l463_46363


namespace NUMINAMATH_CALUDE_min_value_xyz_l463_46372

theorem min_value_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 18) :
  x^2 + 4*x*y + y^2 + 3*z^2 ≥ 63 ∧ ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x * y * z = 18 ∧ x^2 + 4*x*y + y^2 + 3*z^2 = 63 :=
by sorry

end NUMINAMATH_CALUDE_min_value_xyz_l463_46372


namespace NUMINAMATH_CALUDE_multiply_after_subtract_l463_46323

theorem multiply_after_subtract (n : ℝ) (x : ℝ) : n = 12 → 4 * n - 3 = (n - 7) * x → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_multiply_after_subtract_l463_46323


namespace NUMINAMATH_CALUDE_spelling_bee_contest_l463_46333

theorem spelling_bee_contest (initial_students : ℕ) : 
  (initial_students : ℚ) * (1 - 0.66) * (1 - 3/4) = 30 →
  initial_students = 120 := by
sorry

end NUMINAMATH_CALUDE_spelling_bee_contest_l463_46333


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l463_46331

/-- A point is in the fourth quadrant if its x-coordinate is positive and its y-coordinate is negative -/
def is_in_fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

/-- The point (3, -2) is in the fourth quadrant of the Cartesian coordinate system -/
theorem point_in_fourth_quadrant : is_in_fourth_quadrant 3 (-2) := by
  sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l463_46331


namespace NUMINAMATH_CALUDE_complex_function_equality_l463_46362

-- Define the complex function f
def f : ℂ → ℂ := fun z ↦ 2 * (1 - z) - Complex.I

-- State the theorem
theorem complex_function_equality :
  (1 + Complex.I) * f (1 - Complex.I) = -1 + Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_function_equality_l463_46362


namespace NUMINAMATH_CALUDE_parabola_properties_l463_46314

-- Define the parabola C
def C (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x

-- Define the line on which the focus lies
def focus_line (x y : ℝ) : Prop := x + 2*y - 4 = 0

-- Theorem statement
theorem parabola_properties :
  ∃ (p : ℝ), 
    (∃ (x y : ℝ), C p x y ∧ focus_line x y) →
    (p = 8 ∧ 
     ∀ (x : ℝ), (x = -4) ↔ (∃ (y : ℝ), C p x y ∧ ∀ (x' y' : ℝ), C p x' y' → (x - x')^2 + (y - y')^2 ≥ (x + 4)^2)) :=
sorry

end NUMINAMATH_CALUDE_parabola_properties_l463_46314


namespace NUMINAMATH_CALUDE_cos_210_degrees_l463_46315

theorem cos_210_degrees : Real.cos (210 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_210_degrees_l463_46315


namespace NUMINAMATH_CALUDE_find_m_l463_46353

theorem find_m : ∃ m : ℕ, 62519 * m = 624877405 ∧ m = 9995 := by
  sorry

end NUMINAMATH_CALUDE_find_m_l463_46353


namespace NUMINAMATH_CALUDE_sum_of_roots_l463_46356

theorem sum_of_roots (h b x₁ x₂ : ℝ) (hx : x₁ ≠ x₂) 
  (eq₁ : 4 * x₁^2 - h * x₁ = b) (eq₂ : 4 * x₂^2 - h * x₂ = b) : 
  x₁ + x₂ = h / 4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l463_46356


namespace NUMINAMATH_CALUDE_sample_product_l463_46375

/-- Given a sample of five numbers (7, 8, 9, x, y) with an average of 8 
    and a standard deviation of √2, prove that xy = 60 -/
theorem sample_product (x y : ℝ) : 
  (7 + 8 + 9 + x + y) / 5 = 8 → 
  Real.sqrt (((7 - 8)^2 + (8 - 8)^2 + (9 - 8)^2 + (x - 8)^2 + (y - 8)^2) / 5) = Real.sqrt 2 →
  x * y = 60 := by
  sorry

end NUMINAMATH_CALUDE_sample_product_l463_46375


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_ten_l463_46334

theorem sum_of_fractions_equals_ten : 
  (1 / 10 : ℚ) + (2 / 10 : ℚ) + (3 / 10 : ℚ) + (4 / 10 : ℚ) + (5 / 10 : ℚ) + 
  (6 / 10 : ℚ) + (7 / 10 : ℚ) + (8 / 10 : ℚ) + (9 / 10 : ℚ) + (55 / 10 : ℚ) = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_ten_l463_46334


namespace NUMINAMATH_CALUDE_articles_sold_at_cost_price_l463_46385

theorem articles_sold_at_cost_price :
  ∀ (X : ℕ) (C S : ℝ),
  X * C = 32 * S →
  S = C * (1 + 0.5625) →
  X = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_articles_sold_at_cost_price_l463_46385


namespace NUMINAMATH_CALUDE_even_periodic_function_monotonicity_l463_46399

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def monotone_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

theorem even_periodic_function_monotonicity
  (f : ℝ → ℝ) (h_even : is_even f) (h_period : has_period f 2) :
  (monotone_on f (Set.Icc 0 1)) ↔ (∀ x ∈ Set.Icc 3 4, ∀ y ∈ Set.Icc 3 4, x ≤ y → f y ≤ f x) :=
sorry

end NUMINAMATH_CALUDE_even_periodic_function_monotonicity_l463_46399


namespace NUMINAMATH_CALUDE_exactly_three_rainy_days_probability_l463_46358

/-- The probability of exactly k successes in n independent trials with probability p of success in each trial -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- The number of days in the period -/
def num_days : ℕ := 4

/-- The number of rainy days we're interested in -/
def num_rainy_days : ℕ := 3

/-- The probability of rain on any given day -/
def rain_probability : ℝ := 0.5

theorem exactly_three_rainy_days_probability :
  binomial_probability num_days num_rainy_days rain_probability = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_exactly_three_rainy_days_probability_l463_46358


namespace NUMINAMATH_CALUDE_total_flowers_l463_46340

theorem total_flowers (roses : ℕ) (lilies : ℕ) (tulips : ℕ) : 
  roses = 34 →
  lilies = roses + 13 →
  tulips = lilies - 23 →
  roses + lilies + tulips = 105 := by
sorry

end NUMINAMATH_CALUDE_total_flowers_l463_46340


namespace NUMINAMATH_CALUDE_triangle_segment_proof_l463_46328

theorem triangle_segment_proof (a b c h x : ℝ) : 
  a = 40 ∧ b = 75 ∧ c = 100 ∧ 
  a^2 = x^2 + h^2 ∧
  b^2 = (c - x)^2 + h^2 →
  c - x = 70.125 := by
sorry

end NUMINAMATH_CALUDE_triangle_segment_proof_l463_46328


namespace NUMINAMATH_CALUDE_smallest_n_equality_l463_46392

def C (n : ℕ) : ℚ := 512 * (1 - (1/4)^n) / (1 - 1/4)

def D (n : ℕ) : ℚ := 3072 * (1 - (1/(-3))^n) / (1 + 1/3)

theorem smallest_n_equality :
  ∃ (n : ℕ), n ≥ 1 ∧ C n = D n ∧ ∀ (m : ℕ), m ≥ 1 ∧ m < n → C m ≠ D m :=
sorry

end NUMINAMATH_CALUDE_smallest_n_equality_l463_46392


namespace NUMINAMATH_CALUDE_golden_ratio_expression_l463_46376

theorem golden_ratio_expression (R : ℝ) (h1 : R^2 + R - 1 = 0) (h2 : R > 0) :
  R^(R^(R^2 + 1/R) + 1/R) + 1/R = 2 := by
  sorry

end NUMINAMATH_CALUDE_golden_ratio_expression_l463_46376


namespace NUMINAMATH_CALUDE_birthday_box_crayons_l463_46332

/-- The number of crayons Paul gave away -/
def crayons_given : ℕ := 571

/-- The number of crayons Paul lost -/
def crayons_lost : ℕ := 161

/-- The difference between crayons given away and lost -/
def crayons_difference : ℕ := 410

/-- Theorem: The number of crayons in Paul's birthday box is 732 -/
theorem birthday_box_crayons :
  crayons_given + crayons_lost = 732 ∧
  crayons_given - crayons_lost = crayons_difference :=
by sorry

end NUMINAMATH_CALUDE_birthday_box_crayons_l463_46332


namespace NUMINAMATH_CALUDE_smallest_value_theorem_l463_46388

theorem smallest_value_theorem (n : ℕ+) : 
  (n : ℝ) / 2 + 18 / (n : ℝ) ≥ 6 ∧ 
  ((6 : ℕ+) : ℝ) / 2 + 18 / ((6 : ℕ+) : ℝ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_smallest_value_theorem_l463_46388


namespace NUMINAMATH_CALUDE_a_lt_neg_four_sufficient_not_necessary_l463_46348

-- Define the function f(x) = ax + 3
def f (a : ℝ) (x : ℝ) : ℝ := a * x + 3

-- Define the interval [-1, 1]
def interval : Set ℝ := Set.Icc (-1) 1

-- Define what it means for f to have a zero in the interval
def has_zero_in_interval (a : ℝ) : Prop :=
  ∃ x ∈ interval, f a x = 0

-- State the theorem
theorem a_lt_neg_four_sufficient_not_necessary :
  (∀ a : ℝ, a < -4 → has_zero_in_interval a) ∧
  ¬(∀ a : ℝ, has_zero_in_interval a → a < -4) :=
sorry

end NUMINAMATH_CALUDE_a_lt_neg_four_sufficient_not_necessary_l463_46348


namespace NUMINAMATH_CALUDE_last_two_average_l463_46371

theorem last_two_average (list : List ℝ) : 
  list.length = 7 →
  (list.sum / 7 : ℝ) = 63 →
  ((list.take 3).sum / 3 : ℝ) = 58 →
  ((list.drop 3).take 2).sum / 2 = 70 →
  ((list.drop 5).sum / 2 : ℝ) = 63.5 := by
  sorry

end NUMINAMATH_CALUDE_last_two_average_l463_46371


namespace NUMINAMATH_CALUDE_square_modification_l463_46397

theorem square_modification (x : ℝ) : 
  x > 0 →
  x^2 = (x - 2) * (1.2 * x) →
  x = 12 :=
by sorry

end NUMINAMATH_CALUDE_square_modification_l463_46397


namespace NUMINAMATH_CALUDE_min_sum_of_four_primes_l463_46301

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem min_sum_of_four_primes :
  ∀ a b c d s : ℕ,
  is_prime a → is_prime b → is_prime c → is_prime d → is_prime s →
  s = a + b + c + d →
  s ≥ 11 :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_four_primes_l463_46301


namespace NUMINAMATH_CALUDE_common_chord_of_intersecting_circles_l463_46386

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y - 8 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 10*y - 24 = 0

-- Define the common chord
def common_chord (x y : ℝ) : Prop := x - 2*y + 4 = 0

-- Theorem statement
theorem common_chord_of_intersecting_circles :
  ∀ x y : ℝ, (circle1 x y ∧ circle2 x y) → common_chord x y :=
by sorry

end NUMINAMATH_CALUDE_common_chord_of_intersecting_circles_l463_46386


namespace NUMINAMATH_CALUDE_percentage_difference_l463_46381

theorem percentage_difference : (0.60 * 50) - (0.45 * 30) = 16.5 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l463_46381


namespace NUMINAMATH_CALUDE_journey_problem_l463_46389

theorem journey_problem (total_distance : ℝ) (days : ℕ) (q : ℝ) :
  total_distance = 378 ∧ days = 6 ∧ q = 1/2 →
  ∃ a : ℝ, a * (1 - q^days) / (1 - q) = total_distance ∧ a * q^(days - 1) = 6 := by
  sorry

end NUMINAMATH_CALUDE_journey_problem_l463_46389


namespace NUMINAMATH_CALUDE_circle_value_l463_46346

theorem circle_value (circle triangle : ℕ) 
  (eq1 : circle + circle + circle + circle = triangle + triangle + circle)
  (eq2 : triangle = 63) : 
  circle = 42 := by
  sorry

end NUMINAMATH_CALUDE_circle_value_l463_46346


namespace NUMINAMATH_CALUDE_lemonade_pitchers_sum_l463_46395

theorem lemonade_pitchers_sum : 
  let first_intermission : ℚ := 0.25
  let second_intermission : ℚ := 0.4166666666666667
  let third_intermission : ℚ := 0.25
  first_intermission + second_intermission + third_intermission = 0.9166666666666667 := by
sorry

end NUMINAMATH_CALUDE_lemonade_pitchers_sum_l463_46395


namespace NUMINAMATH_CALUDE_toy_problem_solution_l463_46361

/-- Represents the toy purchase and sale problem -/
structure ToyProblem where
  first_purchase_cost : ℝ
  second_purchase_cost : ℝ
  cost_increase_rate : ℝ
  quantity_decrease : ℕ
  min_profit : ℝ

/-- Calculates the cost per item for the first purchase -/
def first_item_cost (p : ToyProblem) : ℝ :=
  50

/-- Calculates the minimum selling price to achieve the desired profit -/
def min_selling_price (p : ToyProblem) : ℝ :=
  70

/-- Theorem stating the correctness of the calculated values -/
theorem toy_problem_solution (p : ToyProblem)
  (h1 : p.first_purchase_cost = 3000)
  (h2 : p.second_purchase_cost = 3000)
  (h3 : p.cost_increase_rate = 0.2)
  (h4 : p.quantity_decrease = 10)
  (h5 : p.min_profit = 1700) :
  first_item_cost p = 50 ∧
  min_selling_price p = 70 ∧
  (min_selling_price p * (p.first_purchase_cost / first_item_cost p +
    p.second_purchase_cost / (first_item_cost p * (1 + p.cost_increase_rate))) -
    (p.first_purchase_cost + p.second_purchase_cost) ≥ p.min_profit) :=
  sorry

end NUMINAMATH_CALUDE_toy_problem_solution_l463_46361


namespace NUMINAMATH_CALUDE_range_of_m_l463_46310

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x + 5

-- State the theorem
theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) m, f x ≤ 10) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) m, f x ≥ 1) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) m, f x = 10) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) m, f x = 1) →
  m ∈ Set.Icc 2 5 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l463_46310


namespace NUMINAMATH_CALUDE_pages_to_read_tomorrow_l463_46308

/-- Given a book and Julie's reading progress, calculate the number of pages to read tomorrow --/
theorem pages_to_read_tomorrow (total_pages yesterday_pages : ℕ) : 
  total_pages = 120 →
  yesterday_pages = 12 →
  (total_pages - (yesterday_pages + 2 * yesterday_pages)) / 2 = 42 := by
sorry

end NUMINAMATH_CALUDE_pages_to_read_tomorrow_l463_46308


namespace NUMINAMATH_CALUDE_exam_average_marks_l463_46390

theorem exam_average_marks (total_boys : ℕ) (total_avg : ℚ) (passed_avg : ℚ) (passed_boys : ℕ) :
  total_boys = 120 →
  total_avg = 35 →
  passed_avg = 39 →
  passed_boys = 100 →
  let failed_boys := total_boys - passed_boys
  let total_marks := total_avg * total_boys
  let passed_marks := passed_avg * passed_boys
  let failed_marks := total_marks - passed_marks
  (failed_marks / failed_boys : ℚ) = 15 := by
  sorry

end NUMINAMATH_CALUDE_exam_average_marks_l463_46390


namespace NUMINAMATH_CALUDE_value_of_expression_l463_46327

theorem value_of_expression (x y : ℝ) (hx : x = 2) (hy : y = 1) : 2 * x - 3 * y = 1 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l463_46327


namespace NUMINAMATH_CALUDE_problem_statement_l463_46324

theorem problem_statement (a b : ℝ) (h : a + b = 3) : 2*a^2 + 4*a*b + 2*b^2 - 4 = 14 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l463_46324


namespace NUMINAMATH_CALUDE_loss_percentage_calculation_l463_46321

def cost_price : ℝ := 120
def selling_price : ℝ := 102
def gain_price : ℝ := 144
def gain_percentage : ℝ := 20

theorem loss_percentage_calculation :
  (cost_price - selling_price) / cost_price * 100 = 15 := by
  sorry

end NUMINAMATH_CALUDE_loss_percentage_calculation_l463_46321


namespace NUMINAMATH_CALUDE_license_plate_count_l463_46312

/-- The number of possible digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of possible letters (A-Z) -/
def num_letters : ℕ := 26

/-- The number of digits in a license plate -/
def digits_in_plate : ℕ := 5

/-- The number of letters in a license plate -/
def letters_in_plate : ℕ := 2

/-- The number of positions where the letter block can be placed -/
def letter_block_positions : ℕ := digits_in_plate + 1

/-- The number of distinct license plates possible -/
def num_license_plates : ℕ := 
  letter_block_positions * num_digits^digits_in_plate * num_letters^letters_in_plate

theorem license_plate_count : num_license_plates = 40560000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l463_46312


namespace NUMINAMATH_CALUDE_pizza_order_proof_l463_46300

/-- The number of slices in each pizza -/
def slices_per_pizza : ℕ := 12

/-- The number of slices Dean ate -/
def dean_slices : ℕ := slices_per_pizza / 2

/-- The number of slices Frank ate -/
def frank_slices : ℕ := 3

/-- The number of slices Sammy ate -/
def sammy_slices : ℕ := slices_per_pizza / 3

/-- The number of slices left over -/
def leftover_slices : ℕ := 11

/-- The total number of pizzas Dean ordered -/
def total_pizzas : ℕ := 2

theorem pizza_order_proof :
  (dean_slices + frank_slices + sammy_slices + leftover_slices) / slices_per_pizza = total_pizzas :=
by sorry

end NUMINAMATH_CALUDE_pizza_order_proof_l463_46300


namespace NUMINAMATH_CALUDE_total_gain_percentage_five_articles_l463_46307

/-- Calculate the total gain percentage for five articles --/
theorem total_gain_percentage_five_articles 
  (cp1 cp2 cp3 cp4 cp5 : ℝ)
  (sp1 sp2 sp3 sp4 sp5 : ℝ)
  (h1 : cp1 = 18.50)
  (h2 : cp2 = 25.75)
  (h3 : cp3 = 42.60)
  (h4 : cp4 = 29.90)
  (h5 : cp5 = 56.20)
  (h6 : sp1 = 22.50)
  (h7 : sp2 = 32.25)
  (h8 : sp3 = 49.60)
  (h9 : sp4 = 36.40)
  (h10 : sp5 = 65.80) :
  let total_cp := cp1 + cp2 + cp3 + cp4 + cp5
  let total_sp := sp1 + sp2 + sp3 + sp4 + sp5
  let total_gain := total_sp - total_cp
  let gain_percentage := (total_gain / total_cp) * 100
  gain_percentage = 19.35 :=
by
  sorry


end NUMINAMATH_CALUDE_total_gain_percentage_five_articles_l463_46307


namespace NUMINAMATH_CALUDE_intersection_line_is_correct_l463_46349

/-- The canonical equations of a line that is the intersection of two planes. -/
def is_intersection_line (p₁ p₂ : ℝ → ℝ → ℝ → Prop) (l : ℝ → ℝ → ℝ → Prop) : Prop :=
  ∀ x y z, l x y z ↔ (p₁ x y z ∧ p₂ x y z)

/-- The first plane equation -/
def plane1 (x y z : ℝ) : Prop := 6 * x - 7 * y - z - 2 = 0

/-- The second plane equation -/
def plane2 (x y z : ℝ) : Prop := x + 7 * y - 4 * z - 5 = 0

/-- The canonical equations of the line -/
def line (x y z : ℝ) : Prop := (x - 1) / 35 = (y - 4/7) / 23 ∧ (x - 1) / 35 = z / 49

theorem intersection_line_is_correct :
  is_intersection_line plane1 plane2 line := by sorry

end NUMINAMATH_CALUDE_intersection_line_is_correct_l463_46349


namespace NUMINAMATH_CALUDE_positive_sum_inequality_l463_46303

theorem positive_sum_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y > 2) :
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 := by
  sorry

end NUMINAMATH_CALUDE_positive_sum_inequality_l463_46303


namespace NUMINAMATH_CALUDE_system_solution_l463_46322

theorem system_solution : ∃ (x y : ℝ), x + 3 * y = 7 ∧ y = 2 * x ∧ x = 1 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l463_46322


namespace NUMINAMATH_CALUDE_mutter_lagaan_payment_l463_46309

theorem mutter_lagaan_payment (total_lagaan : ℝ) (mutter_percentage : ℝ) :
  total_lagaan = 344000 →
  mutter_percentage = 0.23255813953488372 →
  mutter_percentage / 100 * total_lagaan = 800 := by
sorry

end NUMINAMATH_CALUDE_mutter_lagaan_payment_l463_46309


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l463_46393

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 1 - x) ↔ x ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l463_46393


namespace NUMINAMATH_CALUDE_chocolate_pieces_per_box_l463_46313

theorem chocolate_pieces_per_box 
  (total_boxes : ℕ) 
  (given_away : ℕ) 
  (remaining_pieces : ℕ) 
  (h1 : total_boxes = 12) 
  (h2 : given_away = 7) 
  (h3 : remaining_pieces = 30) : 
  remaining_pieces / (total_boxes - given_away) = 6 :=
by sorry

end NUMINAMATH_CALUDE_chocolate_pieces_per_box_l463_46313


namespace NUMINAMATH_CALUDE_triangle_max_area_l463_46368

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C,
    if (3+b)(sin A - sin B) = (c-b)sin C and a = 3,
    then the maximum area of triangle ABC is 9√3/4 -/
theorem triangle_max_area (a b c A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  (3 + b) * (Real.sin A - Real.sin B) = (c - b) * Real.sin C ∧
  a = 3 →
  (∃ (S : ℝ), S = (1/2) * b * c * Real.sin A ∧ 
    ∀ (S' : ℝ), S' = (1/2) * b * c * Real.sin A → S' ≤ S) ∧
  (1/2) * b * c * Real.sin A ≤ (9 * Real.sqrt 3) / 4 :=
by sorry


end NUMINAMATH_CALUDE_triangle_max_area_l463_46368


namespace NUMINAMATH_CALUDE_solution_difference_l463_46306

theorem solution_difference (p q : ℝ) : 
  p ≠ q →
  (p - 5) * (p + 3) = 24 * p - 72 →
  (q - 5) * (q + 3) = 24 * q - 72 →
  p > q →
  p - q = 20 := by
  sorry

end NUMINAMATH_CALUDE_solution_difference_l463_46306


namespace NUMINAMATH_CALUDE_not_prime_if_perfect_square_l463_46360

theorem not_prime_if_perfect_square (n : ℕ) (h1 : n > 0) 
  (h2 : ∃ k : ℕ, n * (n + 2013) = k^2) : ¬ Prime n := by
  sorry

end NUMINAMATH_CALUDE_not_prime_if_perfect_square_l463_46360


namespace NUMINAMATH_CALUDE_parallelogram_secant_minimum_sum_l463_46391

/-- Given a parallelogram ABCD with side lengths a and b, and a secant through
    vertex B intersecting extensions of sides DA and DC at points P and Q
    respectively, the sum of segments PA and CQ is minimized when PA = CQ = √(ab). -/
theorem parallelogram_secant_minimum_sum (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) :
  let f : ℝ → ℝ := λ x => x + (a * b) / x
  ∃ (x : ℝ), x > 0 ∧ f x = Real.sqrt (a * b) + Real.sqrt (a * b) ∧
    ∀ (y : ℝ), y > 0 → f y ≥ f x :=
  sorry

end NUMINAMATH_CALUDE_parallelogram_secant_minimum_sum_l463_46391


namespace NUMINAMATH_CALUDE_sphere_volume_fraction_l463_46305

theorem sphere_volume_fraction (R : ℝ) (h : R > 0) :
  let sphereVolume := (4 / 3) * Real.pi * R^3
  let capVolume := Real.pi * R^3 * ((2 / 3) - (5 * Real.sqrt 2) / 12)
  capVolume / sphereVolume = (8 - 5 * Real.sqrt 2) / 16 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_fraction_l463_46305


namespace NUMINAMATH_CALUDE_remaining_amount_after_expenses_l463_46374

def bonus : ℚ := 1496
def kitchen_fraction : ℚ := 1 / 22
def holiday_fraction : ℚ := 1 / 4
def gift_fraction : ℚ := 1 / 8

theorem remaining_amount_after_expenses : 
  bonus - (bonus * kitchen_fraction + bonus * holiday_fraction + bonus * gift_fraction) = 867 := by
  sorry

end NUMINAMATH_CALUDE_remaining_amount_after_expenses_l463_46374


namespace NUMINAMATH_CALUDE_complement_M_inter_N_eq_one_two_l463_46367

def U : Finset ℕ := {1, 2, 3, 4, 5}
def M : Finset ℕ := {3, 4, 5}
def N : Finset ℕ := {1, 2, 5}

theorem complement_M_inter_N_eq_one_two :
  (U \ M) ∩ N = {1, 2} := by sorry

end NUMINAMATH_CALUDE_complement_M_inter_N_eq_one_two_l463_46367


namespace NUMINAMATH_CALUDE_largest_number_with_given_hcf_and_lcm_factors_l463_46380

theorem largest_number_with_given_hcf_and_lcm_factors (a b : ℕ+) 
  (h_hcf : Nat.gcd a b = 52)
  (h_lcm : Nat.lcm a b = 52 * 11 * 12) :
  max a b = 624 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_with_given_hcf_and_lcm_factors_l463_46380


namespace NUMINAMATH_CALUDE_right_triangle_area_l463_46326

/-- The area of a right triangle with legs of 30 inches and 45 inches is 675 square inches. -/
theorem right_triangle_area (a b : ℝ) (h1 : a = 30) (h2 : b = 45) : 
  (1/2) * a * b = 675 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l463_46326


namespace NUMINAMATH_CALUDE_greatest_value_is_product_of_zeros_l463_46319

def Q (x : ℝ) : ℝ := x^4 + 2*x^3 - x^2 - 4*x + 4

theorem greatest_value_is_product_of_zeros :
  let product_of_zeros : ℝ := 4
  let q_of_one : ℝ := Q 1
  let sum_of_coefficients : ℝ := 1 + 2 - 1 - 4 + 4
  let sum_of_real_zeros : ℝ := 0  -- Assumption based on estimated real zeros
  product_of_zeros > q_of_one ∧
  product_of_zeros > sum_of_coefficients ∧
  product_of_zeros > sum_of_real_zeros :=
by sorry

end NUMINAMATH_CALUDE_greatest_value_is_product_of_zeros_l463_46319


namespace NUMINAMATH_CALUDE_coffee_pod_box_cost_l463_46354

/-- Calculates the cost of a box of coffee pods given vacation details --/
theorem coffee_pod_box_cost
  (vacation_days : ℕ)
  (daily_pods : ℕ)
  (pods_per_box : ℕ)
  (total_spending : ℚ)
  (h1 : vacation_days = 40)
  (h2 : daily_pods = 3)
  (h3 : pods_per_box = 30)
  (h4 : total_spending = 32)
  : (total_spending / (vacation_days * daily_pods / pods_per_box : ℚ)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_coffee_pod_box_cost_l463_46354


namespace NUMINAMATH_CALUDE_cistern_fill_time_l463_46373

theorem cistern_fill_time (fill_time : ℝ) (empty_time : ℝ) (h1 : fill_time = 10) (h2 : empty_time = 12) :
  let net_fill_rate := 1 / fill_time - 1 / empty_time
  1 / net_fill_rate = 60 := by sorry

end NUMINAMATH_CALUDE_cistern_fill_time_l463_46373


namespace NUMINAMATH_CALUDE_power_calculation_l463_46384

theorem power_calculation : 3000 * (3000^3000)^2 = 3000^6001 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l463_46384


namespace NUMINAMATH_CALUDE_cars_in_ten_hours_l463_46343

-- Define the time interval between cars (in minutes)
def time_interval : ℕ := 20

-- Define the total duration (in hours)
def total_duration : ℕ := 10

-- Define the function to calculate the number of cars
def num_cars (interval : ℕ) (duration : ℕ) : ℕ :=
  (duration * 60) / interval

-- Theorem to prove
theorem cars_in_ten_hours :
  num_cars time_interval total_duration = 30 := by
  sorry

end NUMINAMATH_CALUDE_cars_in_ten_hours_l463_46343


namespace NUMINAMATH_CALUDE_sum_of_powers_equals_99_l463_46394

theorem sum_of_powers_equals_99 :
  3^4 + (-3)^3 + (-3)^2 + (-3)^1 + 3^1 + 3^2 + 3^3 = 99 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_equals_99_l463_46394


namespace NUMINAMATH_CALUDE_root_sum_square_l463_46383

theorem root_sum_square (α β : ℝ) : 
  α^2 + 2*α - 2021 = 0 → 
  β^2 + 2*β - 2021 = 0 → 
  α^2 + 3*α + β = 2019 :=
by
  sorry

end NUMINAMATH_CALUDE_root_sum_square_l463_46383


namespace NUMINAMATH_CALUDE_distinct_lines_count_l463_46377

/-- Represents a 4-by-4 grid of lattice points -/
def Grid := Fin 4 × Fin 4

/-- A line in the grid is defined by two distinct points it passes through -/
def Line := { pair : Grid × Grid // pair.1 ≠ pair.2 }

/-- Counts the number of distinct lines passing through at least two points in the grid -/
def countDistinctLines : Nat :=
  sorry

/-- The main theorem stating that the number of distinct lines is 84 -/
theorem distinct_lines_count : countDistinctLines = 84 :=
  sorry

end NUMINAMATH_CALUDE_distinct_lines_count_l463_46377


namespace NUMINAMATH_CALUDE_money_distribution_l463_46366

/-- Given three people A, B, and C with money, prove that B and C together have 320 Rs. -/
theorem money_distribution (A B C : ℕ) 
  (total : A + B + C = 500)
  (ac_sum : A + C = 200)
  (c_amount : C = 20) : 
  B + C = 320 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l463_46366


namespace NUMINAMATH_CALUDE_right_triangle_cosine_l463_46330

theorem right_triangle_cosine (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : a = 7) (h3 : c = 25) :
  b / c = 24 / 25 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_cosine_l463_46330


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l463_46339

theorem min_value_trig_expression (α β : ℝ) :
  (3 * Real.cos α + 4 * Real.sin β - 7)^2 + (3 * Real.sin α + 4 * Real.cos β - 12)^2 ≥ 36 ∧
  ∃ α₀ β₀ : ℝ, (3 * Real.cos α₀ + 4 * Real.sin β₀ - 7)^2 + (3 * Real.sin α₀ + 4 * Real.cos β₀ - 12)^2 = 36 :=
by sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l463_46339


namespace NUMINAMATH_CALUDE_partnership_profit_calculation_l463_46355

/-- Calculates the total profit of a partnership given the investments and one partner's share --/
def calculate_total_profit (invest_a invest_b invest_c c_share : ℕ) : ℕ :=
  let total_parts := invest_a + invest_b + invest_c
  let c_parts := invest_c
  let profit_per_part := c_share / c_parts
  profit_per_part * total_parts

/-- Theorem stating that given the specific investments and C's share, the total profit is 90000 --/
theorem partnership_profit_calculation :
  calculate_total_profit 30000 45000 50000 36000 = 90000 := by
  sorry

#eval calculate_total_profit 30000 45000 50000 36000

end NUMINAMATH_CALUDE_partnership_profit_calculation_l463_46355


namespace NUMINAMATH_CALUDE_tan_three_expression_zero_l463_46329

theorem tan_three_expression_zero (θ : Real) (h : Real.tan θ = 3) :
  (1 - Real.cos θ) / Real.sin θ - Real.sin θ / (1 + Real.cos θ) = 0 := by
  sorry

end NUMINAMATH_CALUDE_tan_three_expression_zero_l463_46329


namespace NUMINAMATH_CALUDE_tan_60_plus_inv_sqrt_3_l463_46320

theorem tan_60_plus_inv_sqrt_3 :
  Real.tan (π / 3) + (Real.sqrt 3)⁻¹ = (4 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_60_plus_inv_sqrt_3_l463_46320


namespace NUMINAMATH_CALUDE_otimes_result_l463_46365

/-- Definition of the ⊗ operation -/
def otimes (a b : ℚ) (x y : ℚ) : ℚ := a^2 * x + b * y - 3

/-- Theorem stating that 2 ⊗ (-6) = 7 given 1 ⊗ (-3) = 2 -/
theorem otimes_result (a b : ℚ) (h : otimes a b 1 (-3) = 2) : otimes a b 2 (-6) = 7 := by
  sorry

end NUMINAMATH_CALUDE_otimes_result_l463_46365


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l463_46387

/-- The distance between the foci of an ellipse given by 4x^2 - 16x + y^2 + 10y + 5 = 0 is 6√3 -/
theorem ellipse_foci_distance :
  ∃ (h k a b : ℝ),
    (∀ x y : ℝ, 4*x^2 - 16*x + y^2 + 10*y + 5 = 0 ↔ (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1) →
    a > b →
    2 * Real.sqrt (a^2 - b^2) = 6 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l463_46387


namespace NUMINAMATH_CALUDE_parallel_lines_slope_l463_46357

theorem parallel_lines_slope (a : ℝ) : 
  (∃ (b c : ℝ), (∀ x y : ℝ, y = (a^2 - a) * x + 2 ↔ y = 6 * x + 3)) → 
  (a = -2 ∨ a = 3) := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_slope_l463_46357


namespace NUMINAMATH_CALUDE_smallest_x_satisfying_equation_l463_46342

theorem smallest_x_satisfying_equation : 
  ∃ (x : ℝ), x > 0 ∧ x = 131/11 ∧ 
  (∀ y : ℝ, y > 0 → ⌊y^2⌋ - y * ⌊y⌋ = 10 → x ≤ y) ∧
  (⌊x^2⌋ - x * ⌊x⌋ = 10) := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_satisfying_equation_l463_46342


namespace NUMINAMATH_CALUDE_product_of_D_coordinates_l463_46336

-- Define the points
def C : ℝ × ℝ := (6, -1)
def N : ℝ × ℝ := (4, 3)

-- Define D as a variable point
variable (D : ℝ × ℝ)

-- Define the midpoint condition
def is_midpoint (M A B : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

-- State the theorem
theorem product_of_D_coordinates :
  is_midpoint N C D → D.1 * D.2 = 14 := by sorry

end NUMINAMATH_CALUDE_product_of_D_coordinates_l463_46336


namespace NUMINAMATH_CALUDE_fraction_inequality_implies_inequality_l463_46370

theorem fraction_inequality_implies_inequality (a b c : ℝ) :
  c ≠ 0 → (a / c^2 < b / c^2) → a < b := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_implies_inequality_l463_46370


namespace NUMINAMATH_CALUDE_odd_number_divisibility_l463_46364

theorem odd_number_divisibility (a : ℤ) (h : ∃ n : ℤ, a = 2*n + 1) :
  ∃ k : ℤ, a^4 + 9*(9 - 2*a^2) = 16*k := by
sorry

end NUMINAMATH_CALUDE_odd_number_divisibility_l463_46364


namespace NUMINAMATH_CALUDE_ellipse_k_range_l463_46325

/-- An ellipse with equation x^2 + ky^2 = 2 and foci on the y-axis -/
structure EllipseOnYAxis where
  k : ℝ
  is_ellipse : k > 0
  foci_on_y_axis : k < 1

/-- The range of k for an ellipse with equation x^2 + ky^2 = 2 and foci on the y-axis is (0, 1) -/
theorem ellipse_k_range (e : EllipseOnYAxis) : 0 < e.k ∧ e.k < 1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l463_46325


namespace NUMINAMATH_CALUDE_S_max_l463_46398

/-- The general term of the sequence -/
def a (n : ℕ) : ℤ := 26 - 2 * n

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ) : ℤ := n * (26 - n)

/-- The theorem stating that S is maximized when n is 12 or 13 -/
theorem S_max : ∀ k : ℕ, S k ≤ max (S 12) (S 13) :=
sorry

end NUMINAMATH_CALUDE_S_max_l463_46398


namespace NUMINAMATH_CALUDE_nonagon_diagonals_l463_46345

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A regular nine-sided polygon has 27 diagonals -/
theorem nonagon_diagonals : num_diagonals 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l463_46345
