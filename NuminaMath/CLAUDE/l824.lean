import Mathlib

namespace NUMINAMATH_CALUDE_sin_cos_sum_equals_quarter_l824_82422

theorem sin_cos_sum_equals_quarter : 
  Real.sin (20 * π / 180) * Real.cos (70 * π / 180) + 
  Real.sin (10 * π / 180) * Real.sin (50 * π / 180) = 1/4 := by sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equals_quarter_l824_82422


namespace NUMINAMATH_CALUDE_max_ab_line_circle_intersection_l824_82472

/-- Given a line ax + by - 8 = 0 (where a > 0 and b > 0) intersecting the circle x² + y² - 2x - 4y = 0
    with a chord length of 2√5, the maximum value of ab is 8. -/
theorem max_ab_line_circle_intersection (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x y : ℝ, a * x + b * y - 8 = 0 → x^2 + y^2 - 2*x - 4*y = 0) →
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    a * x₁ + b * y₁ - 8 = 0 ∧ 
    a * x₂ + b * y₂ - 8 = 0 ∧
    x₁^2 + y₁^2 - 2*x₁ - 4*y₁ = 0 ∧
    x₂^2 + y₂^2 - 2*x₂ - 4*y₂ = 0 ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 20) →
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → a' * b' ≤ a * b) →
  a * b = 8 :=
by sorry

end NUMINAMATH_CALUDE_max_ab_line_circle_intersection_l824_82472


namespace NUMINAMATH_CALUDE_polynomial_root_property_l824_82463

/-- Given a polynomial x^3 + ax^2 + bx + 18b with nonzero integer coefficients a and b,
    if it has two coinciding integer roots and all three roots are integers,
    then |ab| = 1440 -/
theorem polynomial_root_property (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∃ r s : ℤ, (∀ x : ℝ, x^3 + a*x^2 + b*x + 18*b = (x - r)^2 * (x - s)) ∧
              r ≠ s) →
  |a * b| = 1440 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_root_property_l824_82463


namespace NUMINAMATH_CALUDE_min_k_value_l824_82419

theorem min_k_value (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ k : ℝ, 1 / a + 1 / b + k / (a + b) ≥ 0) → 
  ∀ k : ℝ, k ≥ -4 ∧ ∃ k₀ : ℝ, k₀ = -4 ∧ 1 / a + 1 / b + k₀ / (a + b) ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_min_k_value_l824_82419


namespace NUMINAMATH_CALUDE_miles_to_tie_l824_82414

/-- The number of miles Billy runs each day from Sunday to Friday -/
def billy_daily_miles : ℚ := 1

/-- The number of miles Tiffany runs each day from Sunday to Tuesday -/
def tiffany_daily_miles_sun_to_tue : ℚ := 2

/-- The number of miles Tiffany runs each day from Wednesday to Friday -/
def tiffany_daily_miles_wed_to_fri : ℚ := 1/3

/-- The number of days Billy and Tiffany run from Sunday to Tuesday -/
def days_sun_to_tue : ℕ := 3

/-- The number of days Billy and Tiffany run from Wednesday to Friday -/
def days_wed_to_fri : ℕ := 3

theorem miles_to_tie : 
  (tiffany_daily_miles_sun_to_tue * days_sun_to_tue + 
   tiffany_daily_miles_wed_to_fri * days_wed_to_fri) - 
  (billy_daily_miles * (days_sun_to_tue + days_wed_to_fri)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_miles_to_tie_l824_82414


namespace NUMINAMATH_CALUDE_roses_per_bouquet_l824_82407

/-- Proves that the number of roses in each bouquet is 12 given the problem conditions -/
theorem roses_per_bouquet (total_bouquets : ℕ) (rose_bouquets : ℕ) (daisy_bouquets : ℕ) 
  (total_flowers : ℕ) (daisies_per_bouquet : ℕ) :
  total_bouquets = 20 →
  rose_bouquets = 10 →
  daisy_bouquets = 10 →
  rose_bouquets + daisy_bouquets = total_bouquets →
  total_flowers = 190 →
  daisies_per_bouquet = 7 →
  (total_flowers - daisy_bouquets * daisies_per_bouquet) / rose_bouquets = 12 :=
by sorry

end NUMINAMATH_CALUDE_roses_per_bouquet_l824_82407


namespace NUMINAMATH_CALUDE_blue_parrots_count_l824_82458

theorem blue_parrots_count (total_parrots : ℕ) (green_fraction : ℚ) (blue_parrots : ℕ) : 
  total_parrots = 92 →
  green_fraction = 3/4 →
  blue_parrots = total_parrots - (green_fraction * total_parrots).num →
  blue_parrots = 23 := by
sorry

end NUMINAMATH_CALUDE_blue_parrots_count_l824_82458


namespace NUMINAMATH_CALUDE_smallest_height_l824_82457

/-- Represents a rectangular box with square base -/
structure Box where
  x : ℝ  -- side length of the square base
  h : ℝ  -- height of the box
  area : ℝ -- surface area of the box

/-- The height of the box is twice the side length plus one -/
def height_constraint (b : Box) : Prop :=
  b.h = 2 * b.x + 1

/-- The surface area of the box is at least 150 square units -/
def area_constraint (b : Box) : Prop :=
  b.area ≥ 150

/-- The surface area is calculated as 2x^2 + 4x(2x + 1) -/
def surface_area_calc (b : Box) : Prop :=
  b.area = 2 * b.x^2 + 4 * b.x * (2 * b.x + 1)

/-- Main theorem: The smallest possible integer height is 9 units -/
theorem smallest_height (b : Box) 
  (h1 : height_constraint b) 
  (h2 : area_constraint b) 
  (h3 : surface_area_calc b) : 
  ∃ (min_height : ℕ), min_height = 9 ∧ 
    ∀ (h : ℕ), (∃ (b' : Box), height_constraint b' ∧ area_constraint b' ∧ surface_area_calc b' ∧ b'.h = h) → 
      h ≥ min_height :=
sorry

end NUMINAMATH_CALUDE_smallest_height_l824_82457


namespace NUMINAMATH_CALUDE_decimal_expansion_of_one_forty_ninth_l824_82452

/-- The repeating sequence in the decimal expansion of 1/49 -/
def repeating_sequence : List Nat :=
  [0, 2, 0, 4, 0, 8, 1, 6, 3, 2, 6, 5, 3, 0, 6, 1, 2, 2, 4, 4, 8, 9, 7, 9,
   5, 9, 1, 8, 3, 6, 7, 3, 4, 6, 9, 3, 8, 7, 7, 5, 5, 1]

/-- The length of the repeating sequence -/
def sequence_length : Nat := 42

/-- Theorem stating that the decimal expansion of 1/49 has the given repeating sequence -/
theorem decimal_expansion_of_one_forty_ninth :
  ∃ (n : Nat), (1 : ℚ) / 49 = (n : ℚ) / (10^sequence_length - 1) ∧
  repeating_sequence = (n.digits 10).reverse.take sequence_length :=
sorry

end NUMINAMATH_CALUDE_decimal_expansion_of_one_forty_ninth_l824_82452


namespace NUMINAMATH_CALUDE_inequality_theorem_l824_82415

theorem inequality_theorem (p q r : ℝ) (n : ℕ) 
  (h_pos_p : p > 0) (h_pos_q : q > 0) (h_pos_r : r > 0) (h_pqr : p * q * r = 1) : 
  (1 / (p^n + q^n + 1)) + (1 / (q^n + r^n + 1)) + (1 / (r^n + p^n + 1)) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l824_82415


namespace NUMINAMATH_CALUDE_perpendicular_tangents_circles_l824_82497

/-- Two circles with perpendicular tangents at intersection points -/
theorem perpendicular_tangents_circles (a : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + 4*y = 0 ∧ x^2 + y^2 + 2*(a-1)*x + 2*y + a^2 = 0 →
    ∃ m n : ℝ, 
      m^2 + n^2 + 4*n = 0 ∧
      2*(a-1)*m - 2*n + a^2 = 0 ∧
      (n + 2) / m * (n + 1) / (m - (1 - a)) = -1) →
  a = -2 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_tangents_circles_l824_82497


namespace NUMINAMATH_CALUDE_probability_at_least_two_defective_l824_82434

/-- The probability of selecting at least 2 defective items from a batch of products -/
theorem probability_at_least_two_defective (total : Nat) (good : Nat) (defective : Nat) 
  (selected : Nat) (h1 : total = good + defective) (h2 : total = 10) (h3 : good = 6) 
  (h4 : defective = 4) (h5 : selected = 3) : 
  (Nat.choose defective 2 * Nat.choose good 1 + Nat.choose defective 3) / 
  Nat.choose total selected = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_two_defective_l824_82434


namespace NUMINAMATH_CALUDE_max_value_of_expression_l824_82488

theorem max_value_of_expression :
  (∀ x : ℝ, |x - 1| - |x + 4| - 5 ≤ 0) ∧
  (∃ x : ℝ, |x - 1| - |x + 4| - 5 = 0) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l824_82488


namespace NUMINAMATH_CALUDE_y_derivative_l824_82430

noncomputable def y (x : ℝ) : ℝ := Real.sqrt x + (1/3) * Real.arctan (Real.sqrt x) + (8/3) * Real.arctan (Real.sqrt x / 2)

theorem y_derivative (x : ℝ) (h : x > 0) : 
  deriv y x = (3 * x^2 + 16 * x + 32) / (6 * Real.sqrt x * (x + 1) * (x + 4)) :=
sorry

end NUMINAMATH_CALUDE_y_derivative_l824_82430


namespace NUMINAMATH_CALUDE_research_budget_allocation_l824_82474

theorem research_budget_allocation (microphotonics home_electronics gmo industrial_lubricants basic_astrophysics food_additives : ℝ) : 
  microphotonics = 10 →
  home_electronics = 24 →
  gmo = 29 →
  industrial_lubricants = 8 →
  basic_astrophysics / 100 = 50.4 / 360 →
  microphotonics + home_electronics + gmo + industrial_lubricants + basic_astrophysics + food_additives = 100 →
  food_additives = 15 := by
  sorry

end NUMINAMATH_CALUDE_research_budget_allocation_l824_82474


namespace NUMINAMATH_CALUDE_total_hair_product_usage_l824_82498

/-- Represents the daily usage of hair products and calculates the total usage over 14 days. -/
def HairProductUsage (S C H R : ℚ) : Prop :=
  S = 1 ∧
  C = 1/2 * S ∧
  H = 2/3 * S ∧
  R = 1/4 * C ∧
  S * 14 = 14 ∧
  C * 14 = 7 ∧
  H * 14 = 28/3 ∧
  R * 14 = 7/4

/-- Theorem stating the total usage of hair products over 14 days. -/
theorem total_hair_product_usage (S C H R : ℚ) :
  HairProductUsage S C H R →
  S * 14 = 14 ∧ C * 14 = 7 ∧ H * 14 = 28/3 ∧ R * 14 = 7/4 :=
by sorry

end NUMINAMATH_CALUDE_total_hair_product_usage_l824_82498


namespace NUMINAMATH_CALUDE_product_of_primes_sum_99_l824_82404

theorem product_of_primes_sum_99 (p q : ℕ) : 
  Nat.Prime p → Nat.Prime q → p + q = 99 → p * q = 194 := by sorry

end NUMINAMATH_CALUDE_product_of_primes_sum_99_l824_82404


namespace NUMINAMATH_CALUDE_supplement_not_always_greater_l824_82466

/-- The supplement of an angle (in degrees) -/
def supplement (x : ℝ) : ℝ := 180 - x

/-- Theorem stating that the statement "The supplement of an angle is always greater than the angle itself" is false -/
theorem supplement_not_always_greater (x : ℝ) : ¬ (∀ x, supplement x > x) := by
  sorry

end NUMINAMATH_CALUDE_supplement_not_always_greater_l824_82466


namespace NUMINAMATH_CALUDE_inequality_range_difference_l824_82446

-- Define g as a strictly increasing function
variable (g : ℝ → ℝ)

-- Define the property of g being strictly increasing
def StrictlyIncreasing (g : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → g x < g y

-- Define the theorem
theorem inequality_range_difference
  (h1 : StrictlyIncreasing g)
  (h2 : ∀ x, x ≥ 0 → g x ≠ 0)
  (h3 : ∃ a b, ∀ t, (g (2*t^2 + t + 5) < g (t^2 - 3*t + 2)) ↔ (b < t ∧ t < a)) :
  ∃ a b, (∀ t, (g (2*t^2 + t + 5) < g (t^2 - 3*t + 2)) ↔ (b < t ∧ t < a)) ∧ a - b = 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_difference_l824_82446


namespace NUMINAMATH_CALUDE_no_valid_operation_l824_82403

-- Define the set of standard arithmetic operations
inductive ArithOp
  | Add
  | Sub
  | Mul
  | Div

-- Define a function to apply an arithmetic operation
def applyOp (op : ArithOp) (a b : ℤ) : ℚ :=
  match op with
  | ArithOp.Add => a + b
  | ArithOp.Sub => a - b
  | ArithOp.Mul => a * b
  | ArithOp.Div => (a : ℚ) / (b : ℚ)

-- Theorem statement
theorem no_valid_operation : ∀ (op : ArithOp), 
  (applyOp op 9 3 : ℚ) + 5 - (4 - 2) ≠ 7 := by
  sorry

end NUMINAMATH_CALUDE_no_valid_operation_l824_82403


namespace NUMINAMATH_CALUDE_field_division_fraction_l824_82432

theorem field_division_fraction (total_area smaller_area larger_area : ℝ) : 
  total_area = 500 →
  smaller_area = 225 →
  larger_area = total_area - smaller_area →
  (larger_area - smaller_area) / ((smaller_area + larger_area) / 2) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_field_division_fraction_l824_82432


namespace NUMINAMATH_CALUDE_sarah_homework_problem_l824_82431

/-- The number of homework problems Sarah had initially -/
def total_problems (finished_problems : ℕ) (remaining_pages : ℕ) (problems_per_page : ℕ) : ℕ :=
  finished_problems + remaining_pages * problems_per_page

/-- Theorem stating that Sarah had 60 homework problems initially -/
theorem sarah_homework_problem :
  total_problems 20 5 8 = 60 := by
  sorry

end NUMINAMATH_CALUDE_sarah_homework_problem_l824_82431


namespace NUMINAMATH_CALUDE_harry_snakes_l824_82469

/-- The number of snakes Harry owns -/
def num_snakes : ℕ := sorry

/-- The number of geckos Harry owns -/
def num_geckos : ℕ := 3

/-- The number of iguanas Harry owns -/
def num_iguanas : ℕ := 2

/-- Monthly feeding cost per snake in dollars -/
def snake_cost : ℕ := 10

/-- Monthly feeding cost per iguana in dollars -/
def iguana_cost : ℕ := 5

/-- Monthly feeding cost per gecko in dollars -/
def gecko_cost : ℕ := 15

/-- Total yearly feeding cost for all pets in dollars -/
def total_yearly_cost : ℕ := 1140

theorem harry_snakes :
  num_snakes = 4 ∧
  (12 * (num_geckos * gecko_cost + num_iguanas * iguana_cost + num_snakes * snake_cost) = total_yearly_cost) :=
by sorry

end NUMINAMATH_CALUDE_harry_snakes_l824_82469


namespace NUMINAMATH_CALUDE_original_number_is_nine_l824_82401

theorem original_number_is_nine (N : ℕ) : (N - 4) % 5 = 0 → N = 9 := by
  sorry

end NUMINAMATH_CALUDE_original_number_is_nine_l824_82401


namespace NUMINAMATH_CALUDE_fourier_expansion_arccos_plus_one_l824_82413

-- Define the Chebyshev polynomials
noncomputable def T (n : ℕ) (x : ℝ) : ℝ := Real.cos (n * Real.arccos x)

-- Define the function to be expanded
noncomputable def f (x : ℝ) : ℝ := Real.arccos x + 1

-- Define the interval
def I : Set ℝ := Set.Ioo (-1) 1

-- Define the Fourier coefficient
noncomputable def a (n : ℕ) : ℝ :=
  if n = 0
  then (Real.pi + 2) / 2
  else 2 / Real.pi * ((-1)^n - 1) / (n^2 : ℝ)

-- State the theorem
theorem fourier_expansion_arccos_plus_one :
  ∀ x ∈ I, f x = (Real.pi + 2) / 2 + ∑' n, a n * T n x :=
sorry

end NUMINAMATH_CALUDE_fourier_expansion_arccos_plus_one_l824_82413


namespace NUMINAMATH_CALUDE_joan_sold_26_books_l824_82467

/-- The number of books Joan sold in the yard sale -/
def books_sold (initial_books : ℕ) (remaining_books : ℕ) : ℕ :=
  initial_books - remaining_books

/-- Proof that Joan sold 26 books -/
theorem joan_sold_26_books (initial_books : ℕ) (remaining_books : ℕ) 
  (h1 : initial_books = 33) (h2 : remaining_books = 7) : 
  books_sold initial_books remaining_books = 26 := by
  sorry

#eval books_sold 33 7

end NUMINAMATH_CALUDE_joan_sold_26_books_l824_82467


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l824_82461

def A : Set ℝ := {x | x^2 - 4*x - 5 < 0}

def B : Set ℝ := {x | |x - 1| > 1}

theorem intersection_of_A_and_B :
  A ∩ B = {x | (-1 < x ∧ x < 0) ∨ (2 < x ∧ x < 5)} :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l824_82461


namespace NUMINAMATH_CALUDE_temperature_comparison_l824_82443

theorem temperature_comparison : -3 < -0.3 := by
  sorry

end NUMINAMATH_CALUDE_temperature_comparison_l824_82443


namespace NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l824_82453

theorem sqrt_3_times_sqrt_12 : Real.sqrt 3 * Real.sqrt 12 = 6 := by sorry

end NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l824_82453


namespace NUMINAMATH_CALUDE_oil_mixture_price_l824_82441

/-- Given two types of oil mixed together, calculate the price of the second oil. -/
theorem oil_mixture_price (volume1 volume2 total_volume : ℚ) (price1 mixture_price : ℚ) :
  volume1 = 10 →
  volume2 = 5 →
  total_volume = volume1 + volume2 →
  price1 = 54 →
  mixture_price = 58 →
  ∃ price2 : ℚ, 
    price2 = 66 ∧
    volume1 * price1 + volume2 * price2 = total_volume * mixture_price :=
by sorry

end NUMINAMATH_CALUDE_oil_mixture_price_l824_82441


namespace NUMINAMATH_CALUDE_linear_function_values_l824_82484

/-- A linear function y = kx + b passing through (-1, 0) and (2, 1/2) -/
def linear_function (x : ℚ) : ℚ :=
  let k : ℚ := 6
  let b : ℚ := -1
  k * x + b

theorem linear_function_values :
  linear_function 0 = -1 ∧
  linear_function (1/2) = 2 ∧
  linear_function (-1/2) = -4 := by
  sorry


end NUMINAMATH_CALUDE_linear_function_values_l824_82484


namespace NUMINAMATH_CALUDE_gcd_of_powers_of_two_minus_one_l824_82411

theorem gcd_of_powers_of_two_minus_one :
  Nat.gcd (2^1015 - 1) (2^1020 - 1) = 1 := by sorry

end NUMINAMATH_CALUDE_gcd_of_powers_of_two_minus_one_l824_82411


namespace NUMINAMATH_CALUDE_rational_sum_theorem_l824_82428

theorem rational_sum_theorem (x y : ℚ) 
  (hx : |x| = 5) 
  (hy : |y| = 2) 
  (hxy : |x - y| = x - y) : 
  x + y = 7 ∨ x + y = 3 := by
sorry

end NUMINAMATH_CALUDE_rational_sum_theorem_l824_82428


namespace NUMINAMATH_CALUDE_stream_speed_l824_82496

theorem stream_speed (downstream_speed upstream_speed : ℝ) 
  (h1 : downstream_speed = 11)
  (h2 : upstream_speed = 8) :
  let stream_speed := (downstream_speed - upstream_speed) / 2
  stream_speed = 1.5 := by sorry

end NUMINAMATH_CALUDE_stream_speed_l824_82496


namespace NUMINAMATH_CALUDE_deceased_member_income_l824_82408

/-- Proves that given a family of 3 earning members with an average monthly income of Rs. 735,
    if one member dies and the new average income becomes Rs. 650,
    then the income of the deceased member was Rs. 905. -/
theorem deceased_member_income
  (total_income : ℕ)
  (remaining_income : ℕ)
  (h1 : total_income / 3 = 735)
  (h2 : remaining_income / 2 = 650)
  (h3 : total_income > remaining_income) :
  total_income - remaining_income = 905 := by
sorry

end NUMINAMATH_CALUDE_deceased_member_income_l824_82408


namespace NUMINAMATH_CALUDE_new_bill_total_l824_82400

/-- Calculates the new bill total after substitutions and additional charges -/
def calculate_new_bill (original_order : ℝ) 
                       (tomato_old : ℝ) (tomato_new : ℝ)
                       (lettuce_old : ℝ) (lettuce_new : ℝ)
                       (celery_old : ℝ) (celery_new : ℝ)
                       (delivery_tip : ℝ) : ℝ :=
  original_order + (tomato_new - tomato_old) + (lettuce_new - lettuce_old) + 
  (celery_new - celery_old) + delivery_tip

/-- Theorem stating that the new bill total is $35.00 -/
theorem new_bill_total : 
  calculate_new_bill 25 0.99 2.20 1.00 1.75 1.96 2.00 8.00 = 35 := by
  sorry

end NUMINAMATH_CALUDE_new_bill_total_l824_82400


namespace NUMINAMATH_CALUDE_smallest_four_digit_negative_congruent_to_one_mod_37_l824_82420

theorem smallest_four_digit_negative_congruent_to_one_mod_37 :
  ∀ x : ℤ, x < 0 ∧ x ≥ -9999 ∧ x ≡ 1 [ZMOD 37] → x ≥ -1034 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_negative_congruent_to_one_mod_37_l824_82420


namespace NUMINAMATH_CALUDE_line_through_intersection_and_parallel_line_through_intersection_with_equal_intercepts_l824_82478

-- Define the lines
def line1 (x y : ℝ) := x + 2*y - 5 = 0
def line2 (x y : ℝ) := 3*x - y - 1 = 0
def line3 (x y : ℝ) := 5*x - y + 100 = 0
def line4 (x y : ℝ) := 2*x + y - 8 = 0
def line5 (x y : ℝ) := x - 2*y + 1 = 0

-- Define the result lines
def result_line1 (x y : ℝ) := 5*x - y - 3 = 0
def result_line2a (x y : ℝ) := 2*x - 3*y = 0
def result_line2b (x y : ℝ) := x + y - 5 = 0

-- Theorem for the first part
theorem line_through_intersection_and_parallel :
  ∃ (x₀ y₀ : ℝ), line1 x₀ y₀ ∧ line2 x₀ y₀ →
  ∀ (x y : ℝ), (y - y₀ = (5 : ℝ) * (x - x₀)) ↔ result_line1 x y :=
sorry

-- Theorem for the second part
theorem line_through_intersection_with_equal_intercepts :
  ∃ (x₀ y₀ : ℝ), line4 x₀ y₀ ∧ line5 x₀ y₀ →
  ∀ (x y : ℝ), (∃ (a : ℝ), x = a ∧ y = a) →
  (result_line2a x y ∨ result_line2b x y) :=
sorry

end NUMINAMATH_CALUDE_line_through_intersection_and_parallel_line_through_intersection_with_equal_intercepts_l824_82478


namespace NUMINAMATH_CALUDE_expression_value_l824_82424

theorem expression_value (x y : ℝ) (h : x - 2*y = -1) : 6 + 2*x - 4*y = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l824_82424


namespace NUMINAMATH_CALUDE_short_story_booklets_l824_82486

theorem short_story_booklets (pages_per_booklet : ℕ) (total_pages : ℕ) (h1 : pages_per_booklet = 9) (h2 : total_pages = 441) :
  total_pages / pages_per_booklet = 49 := by
  sorry

end NUMINAMATH_CALUDE_short_story_booklets_l824_82486


namespace NUMINAMATH_CALUDE_function_increasing_on_interval_l824_82464

/-- The function f(x) = 3x - x^3 is monotonically increasing on the interval [-1, 1]. -/
theorem function_increasing_on_interval (x : ℝ) :
  (∀ x₁ x₂ : ℝ, -1 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 1 → (3 * x₁ - x₁^3) < (3 * x₂ - x₂^3)) :=
by sorry

end NUMINAMATH_CALUDE_function_increasing_on_interval_l824_82464


namespace NUMINAMATH_CALUDE_george_candy_count_l824_82406

/-- The number of bags of candy -/
def num_bags : ℕ := 8

/-- The number of candy pieces in each bag -/
def pieces_per_bag : ℕ := 81

/-- The total number of candy pieces -/
def total_pieces : ℕ := num_bags * pieces_per_bag

theorem george_candy_count : total_pieces = 648 := by
  sorry

end NUMINAMATH_CALUDE_george_candy_count_l824_82406


namespace NUMINAMATH_CALUDE_pythagorean_triple_properties_l824_82473

theorem pythagorean_triple_properties (a b c : ℕ) (h : a^2 + b^2 = c^2) :
  (Even a ∨ Even b) ∧
  (3 ∣ a ∨ 3 ∣ b) ∧
  (5 ∣ a ∨ 5 ∣ b ∨ 5 ∣ c) := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_triple_properties_l824_82473


namespace NUMINAMATH_CALUDE_percentage_difference_l824_82435

theorem percentage_difference (x y z : ℝ) (hx : x = 5 * y) (hz : z = 1.2 * y) :
  (z - y) / x = 0.04 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l824_82435


namespace NUMINAMATH_CALUDE_inequality_proof_l824_82460

theorem inequality_proof (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x^3 + y^3 + z^3 = 1) :
  (x^2 / (1 - x^2)) + (y^2 / (1 - y^2)) + (z^2 / (1 - z^2)) ≥ (3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l824_82460


namespace NUMINAMATH_CALUDE_acid_dilution_l824_82495

/-- Given m ounces of an m% acid solution, when x ounces of water are added,
    a new solution of (m-20)% concentration is formed. Assuming m > 25,
    prove that x = 20m / (m-20). -/
theorem acid_dilution (m : ℝ) (x : ℝ) (h : m > 25) :
  (m * m / 100 = (m - 20) / 100 * (m + x)) → x = 20 * m / (m - 20) := by
  sorry

end NUMINAMATH_CALUDE_acid_dilution_l824_82495


namespace NUMINAMATH_CALUDE_binary_10111_equals_23_l824_82405

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldr (fun (i, bit) acc => acc + if bit then 2^i else 0) 0

theorem binary_10111_equals_23 : 
  binary_to_decimal [true, true, true, false, true] = 23 := by
  sorry

end NUMINAMATH_CALUDE_binary_10111_equals_23_l824_82405


namespace NUMINAMATH_CALUDE_M_squared_equals_36_50_times_144_36_and_sum_of_digits_75_l824_82499

/-- The sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Definition of M -/
def M : ℕ := sorry

theorem M_squared_equals_36_50_times_144_36_and_sum_of_digits_75 :
  M^2 = 36^50 * 144^36 ∧ sum_of_digits M = 75 := by
  sorry

end NUMINAMATH_CALUDE_M_squared_equals_36_50_times_144_36_and_sum_of_digits_75_l824_82499


namespace NUMINAMATH_CALUDE_julia_tag_total_l824_82439

/-- The number of kids Julia played tag with on Monday -/
def monday_kids : ℕ := 7

/-- The number of kids Julia played tag with on Tuesday -/
def tuesday_kids : ℕ := 13

/-- The total number of kids Julia played tag with -/
def total_kids : ℕ := monday_kids + tuesday_kids

theorem julia_tag_total : total_kids = 20 := by
  sorry

end NUMINAMATH_CALUDE_julia_tag_total_l824_82439


namespace NUMINAMATH_CALUDE_exists_angle_leq_90_degrees_l824_82409

-- Define a type for rays in space
def Ray : Type := ℝ → ℝ × ℝ × ℝ

-- Define a function to calculate the angle between two rays
def angle_between_rays (r1 r2 : Ray) : ℝ := sorry

-- State the theorem
theorem exists_angle_leq_90_degrees (rays : Fin 5 → Ray) 
  (h_distinct : ∀ i j, i ≠ j → rays i ≠ rays j) : 
  ∃ i j, i ≠ j ∧ angle_between_rays (rays i) (rays j) ≤ 90 := by sorry

end NUMINAMATH_CALUDE_exists_angle_leq_90_degrees_l824_82409


namespace NUMINAMATH_CALUDE_solution_existence_l824_82456

/-- Given a positive real number a, prove the existence conditions for real solutions
    of the system of equations y = mx ± a and 1/x - 1/y = 1/a for different values of m. -/
theorem solution_existence (a : ℝ) (ha : a > 0) :
  (∀ m : ℝ, ∃ x y : ℝ, y = m * x + a ∧ 1 / x - 1 / y = 1 / a) ∧
  (∀ m : ℝ, (∃ x y : ℝ, y = m * x - a ∧ 1 / x - 1 / y = 1 / a) ↔ m ≤ 0 ∨ m ≥ 4) :=
by sorry

end NUMINAMATH_CALUDE_solution_existence_l824_82456


namespace NUMINAMATH_CALUDE_greatest_third_side_proof_l824_82418

/-- The greatest integer length of the third side of a triangle with two sides of 7 cm and 15 cm -/
def greatest_third_side : ℕ := 21

/-- Triangle inequality theorem for our specific case -/
axiom triangle_inequality (a b c : ℝ) : 
  (a = 7 ∧ b = 15) → (c < a + b ∧ c > |a - b|)

theorem greatest_third_side_proof : 
  ∀ c : ℝ, (c < 22 ∧ c > 8) → c ≤ greatest_third_side := by sorry

end NUMINAMATH_CALUDE_greatest_third_side_proof_l824_82418


namespace NUMINAMATH_CALUDE_secure_app_theorem_l824_82471

/-- Represents an online store application -/
structure OnlineStoreApp where
  paymentGateway : Bool
  dataEncryption : Bool
  transitEncryption : Bool
  codeObfuscation : Bool
  rootedDeviceRestriction : Bool
  antivirusAgent : Bool

/-- Defines the security level of an application -/
def securityLevel (app : OnlineStoreApp) : ℕ :=
  (if app.paymentGateway then 1 else 0) +
  (if app.dataEncryption then 1 else 0) +
  (if app.transitEncryption then 1 else 0) +
  (if app.codeObfuscation then 1 else 0) +
  (if app.rootedDeviceRestriction then 1 else 0) +
  (if app.antivirusAgent then 1 else 0)

/-- Defines a secure application -/
def isSecure (app : OnlineStoreApp) : Prop :=
  securityLevel app = 6

/-- Theorem: An online store app with all security measures implemented is secure -/
theorem secure_app_theorem (app : OnlineStoreApp) 
  (h1 : app.paymentGateway = true)
  (h2 : app.dataEncryption = true)
  (h3 : app.transitEncryption = true)
  (h4 : app.codeObfuscation = true)
  (h5 : app.rootedDeviceRestriction = true)
  (h6 : app.antivirusAgent = true) : 
  isSecure app :=
by
  sorry


end NUMINAMATH_CALUDE_secure_app_theorem_l824_82471


namespace NUMINAMATH_CALUDE_als_original_portion_l824_82449

theorem als_original_portion :
  ∀ (a b c : ℝ),
    a + b + c = 2000 →
    a ≠ b ∧ b ≠ c ∧ a ≠ c →
    a - 150 + 3 * b + 2 * c = 3500 →
    a = 500 := by
  sorry

end NUMINAMATH_CALUDE_als_original_portion_l824_82449


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_l824_82459

-- Define the types for planes and lines
variable (Plane : Type) (Line : Type)

-- Define the relationships between planes and lines
variable (is_perpendicular_line_plane : Line → Plane → Prop)
variable (is_perpendicular_plane_plane : Plane → Plane → Prop)
variable (is_perpendicular_line_line : Line → Line → Prop)
variable (are_distinct : Plane → Plane → Prop)
variable (are_non_intersecting : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_from_perpendicular_planes 
  (α β : Plane) (l m : Line)
  (h_distinct : are_distinct α β)
  (h_non_intersecting : are_non_intersecting l m)
  (h1 : is_perpendicular_line_plane l α)
  (h2 : is_perpendicular_line_plane m β)
  (h3 : is_perpendicular_plane_plane α β) :
  is_perpendicular_line_line l m :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_l824_82459


namespace NUMINAMATH_CALUDE_jacksons_decorations_l824_82442

/-- Given that Mrs. Jackson has 4 boxes of Christmas decorations with 15 decorations in each box
    and she used 35 decorations, prove that she gave 25 decorations to her neighbor. -/
theorem jacksons_decorations (num_boxes : ℕ) (decorations_per_box : ℕ) (used_decorations : ℕ)
    (h1 : num_boxes = 4)
    (h2 : decorations_per_box = 15)
    (h3 : used_decorations = 35) :
    num_boxes * decorations_per_box - used_decorations = 25 := by
  sorry

end NUMINAMATH_CALUDE_jacksons_decorations_l824_82442


namespace NUMINAMATH_CALUDE_smallest_gcd_qr_l824_82410

theorem smallest_gcd_qr (p q r : ℕ+) (h1 : Nat.gcd p q = 210) (h2 : Nat.gcd p r = 770) :
  ∃ (q' r' : ℕ+), Nat.gcd q' r' = 10 ∧ ∀ (q'' r'' : ℕ+), Nat.gcd q'' r'' ≥ 10 :=
by sorry

end NUMINAMATH_CALUDE_smallest_gcd_qr_l824_82410


namespace NUMINAMATH_CALUDE_jaguar_snake_consumption_l824_82490

theorem jaguar_snake_consumption 
  (beetles_per_bird : ℕ) 
  (birds_per_snake : ℕ) 
  (total_jaguars : ℕ) 
  (total_beetles_eaten : ℕ) 
  (h1 : beetles_per_bird = 12)
  (h2 : birds_per_snake = 3)
  (h3 : total_jaguars = 6)
  (h4 : total_beetles_eaten = 1080) :
  total_beetles_eaten / total_jaguars / beetles_per_bird / birds_per_snake = 5 := by
  sorry

end NUMINAMATH_CALUDE_jaguar_snake_consumption_l824_82490


namespace NUMINAMATH_CALUDE_double_arccos_three_fifths_equals_arcsin_twentyfour_twentyfifths_l824_82489

theorem double_arccos_three_fifths_equals_arcsin_twentyfour_twentyfifths :
  2 * Real.arccos (3/5) = Real.arcsin (24/25) := by
  sorry

end NUMINAMATH_CALUDE_double_arccos_three_fifths_equals_arcsin_twentyfour_twentyfifths_l824_82489


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l824_82417

theorem quadratic_inequality_solution (a : ℝ) : 
  (∀ x, x ≠ -(1/a) → a*x^2 + 2*x + a > 0) ∧ 
  (∃ x, a*x^2 + 2*x + a ≤ 0) → 
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l824_82417


namespace NUMINAMATH_CALUDE_distinguishable_triangles_l824_82427

/-- Represents the number of available colors for small triangles -/
def num_colors : ℕ := 8

/-- Represents the number of small triangles in the inner part of the large triangle -/
def inner_triangles : ℕ := 3

/-- Represents the number of small triangles in the outer part of the large triangle -/
def outer_triangles : ℕ := 3

/-- Represents the total number of small triangles in the large triangle -/
def total_triangles : ℕ := inner_triangles + outer_triangles

/-- Calculates the number of ways to color the inner triangle -/
def inner_colorings : ℕ := 
  num_colors + (num_colors * (num_colors - 1)) + (num_colors.choose inner_triangles * inner_triangles.factorial)

/-- Calculates the number of ways to color the outer triangle -/
def outer_colorings : ℕ := 
  num_colors + (num_colors * (num_colors - 1)) + (num_colors.choose outer_triangles * outer_triangles.factorial)

/-- Theorem stating the total number of distinguishable large triangles -/
theorem distinguishable_triangles : 
  inner_colorings * outer_colorings = 116096 := by sorry

end NUMINAMATH_CALUDE_distinguishable_triangles_l824_82427


namespace NUMINAMATH_CALUDE_min_A_over_B_l824_82485

theorem min_A_over_B (x A B : ℝ) (hx : x > 0) (hA : A > 0) (hB : B > 0)
  (h1 : x^2 + 1/x^2 = A) (h2 : x - 1/x = B + 3) :
  A / B ≥ 6 + 2 * Real.sqrt 11 ∧
  (A / B = 6 + 2 * Real.sqrt 11 ↔ B = Real.sqrt 11) :=
sorry

end NUMINAMATH_CALUDE_min_A_over_B_l824_82485


namespace NUMINAMATH_CALUDE_functional_equation_solution_l824_82426

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 4 * f x * f y = f (x + y) + f (x - y)

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h1 : FunctionalEquation f) (h2 : f 1 = 1/4) : f 2014 = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l824_82426


namespace NUMINAMATH_CALUDE_exam_comparison_l824_82468

/-- Given a 50-question exam where Sylvia has one-fifth of incorrect answers
    and Sergio has 4 incorrect answers, prove that Sergio has 6 more correct
    answers than Sylvia. -/
theorem exam_comparison (total_questions : ℕ) (sylvia_incorrect_ratio : ℚ)
    (sergio_incorrect : ℕ) (h1 : total_questions = 50)
    (h2 : sylvia_incorrect_ratio = 1 / 5)
    (h3 : sergio_incorrect = 4) :
    (total_questions - (sylvia_incorrect_ratio * total_questions).num) -
    (total_questions - sergio_incorrect) = 6 := by
  sorry

end NUMINAMATH_CALUDE_exam_comparison_l824_82468


namespace NUMINAMATH_CALUDE_cube_surface_area_from_prism_volume_l824_82476

/-- Given a rectangular prism with dimensions 16, 4, and 24 inches,
    prove that a cube with the same volume has a surface area of
    approximately 798 square inches. -/
theorem cube_surface_area_from_prism_volume :
  let prism_length : ℝ := 16
  let prism_width : ℝ := 4
  let prism_height : ℝ := 24
  let prism_volume : ℝ := prism_length * prism_width * prism_height
  let cube_edge : ℝ := prism_volume ^ (1/3)
  let cube_surface_area : ℝ := 6 * cube_edge ^ 2
  ∃ ε > 0, |cube_surface_area - 798| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_cube_surface_area_from_prism_volume_l824_82476


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l824_82470

theorem triangle_abc_properties (A B C : Real) (a b c : Real) :
  B = π / 4 →
  c = Real.sqrt 6 →
  C = π / 3 →
  A + B + C = π →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  A = 5 * π / 12 ∧
  a = 1 + Real.sqrt 3 ∧
  b = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l824_82470


namespace NUMINAMATH_CALUDE_combined_average_age_l824_82440

theorem combined_average_age (room_a_count : ℕ) (room_b_count : ℕ) 
  (room_a_avg : ℚ) (room_b_avg : ℚ) :
  room_a_count = 8 →
  room_b_count = 6 →
  room_a_avg = 35 →
  room_b_avg = 30 →
  let total_count := room_a_count + room_b_count
  let total_age := room_a_count * room_a_avg + room_b_count * room_b_avg
  (total_age / total_count : ℚ) = 32.86 := by
  sorry

#eval (8 * 35 + 6 * 30) / (8 + 6)

end NUMINAMATH_CALUDE_combined_average_age_l824_82440


namespace NUMINAMATH_CALUDE_correct_calculation_l824_82482

theorem correct_calculation (a : ℝ) : (-2 * a^3)^2 = 4 * a^6 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l824_82482


namespace NUMINAMATH_CALUDE_prob_at_most_one_for_given_probabilities_l824_82444

/-- The probability that at most one of two independent events occurs, given their individual probabilities -/
def prob_at_most_one (p_a p_b : ℝ) : ℝ :=
  1 - p_a * p_b

theorem prob_at_most_one_for_given_probabilities :
  let p_a := 0.6
  let p_b := 0.7
  prob_at_most_one p_a p_b = 0.58 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_most_one_for_given_probabilities_l824_82444


namespace NUMINAMATH_CALUDE_triangle_existence_and_area_l824_82425

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a = Real.sqrt 6 ∧
  Real.sin t.B ^ 2 + Real.sin t.C ^ 2 = Real.sin t.A ^ 2 + (2 * Real.sqrt 3 / 3) * Real.sin t.A * Real.sin t.B * Real.sin t.C

-- Define the theorem
theorem triangle_existence_and_area (t : Triangle) :
  triangle_conditions t → t.b + t.c = 2 * Real.sqrt 3 →
  ∃ (area : ℝ), area = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_existence_and_area_l824_82425


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l824_82492

/-- An isosceles triangle with sides a, b, and c, where a = b -/
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  isosceles : a = b
  positive_a : 0 < a
  positive_b : 0 < b
  positive_c : 0 < c
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The perimeter of a triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ := t.a + t.b + t.c

/-- Theorem: The perimeter of an isosceles triangle with sides 2 and 5 is 12 -/
theorem isosceles_triangle_perimeter :
  ∃ (t : IsoscelesTriangle), t.a = 2 ∧ t.c = 5 ∧ perimeter t = 12 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l824_82492


namespace NUMINAMATH_CALUDE_savings_calculation_l824_82462

/-- Calculates the total savings given a daily savings amount and number of days -/
def totalSavings (dailySavings : ℕ) (days : ℕ) : ℕ :=
  dailySavings * days

/-- Theorem: If a person saves $24 every day for 365 days, the total savings will be $8,760 -/
theorem savings_calculation :
  totalSavings 24 365 = 8760 := by
  sorry

end NUMINAMATH_CALUDE_savings_calculation_l824_82462


namespace NUMINAMATH_CALUDE_circle_radius_for_equal_areas_l824_82438

/-- The radius of a circle satisfying the given conditions for a right-angled triangle --/
theorem circle_radius_for_equal_areas (a b c : ℝ) (h_right_triangle : a^2 + b^2 = c^2)
  (h_side_lengths : a = 6 ∧ b = 8 ∧ c = 10) : 
  ∃ r : ℝ, r^2 = 24 / Real.pi ∧ 
    (π * r^2 = a * b / 2) ∧
    (π * r^2 - a * b / 2 = a * b / 2 - π * r^2) :=
sorry

end NUMINAMATH_CALUDE_circle_radius_for_equal_areas_l824_82438


namespace NUMINAMATH_CALUDE_cos_difference_from_sum_l824_82455

theorem cos_difference_from_sum (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 1.5) 
  (h2 : Real.cos A + Real.cos B = 1) : 
  Real.cos (A - B) = 0.625 := by
sorry

end NUMINAMATH_CALUDE_cos_difference_from_sum_l824_82455


namespace NUMINAMATH_CALUDE_pure_imaginary_quotient_l824_82416

theorem pure_imaginary_quotient (a : ℝ) : 
  let z₁ : ℂ := a + 2*I
  let z₂ : ℂ := 3 - 4*I
  (∃ (b : ℝ), z₁ / z₂ = b*I) → a = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_quotient_l824_82416


namespace NUMINAMATH_CALUDE_inequality_proof_l824_82454

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b)) ≥ 
  3/2 + 1/4 * (a * (c - b)^2 / (c + b) + b * (c - a)^2 / (c + a) + c * (b - a)^2 / (b + a)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l824_82454


namespace NUMINAMATH_CALUDE_integer_solutions_of_equation_l824_82436

theorem integer_solutions_of_equation :
  ∀ x y : ℤ, x^2 = 1 + 4*y^3*(y + 2) ↔ 
    (x = 1 ∧ y = 0) ∨ (x = 1 ∧ y = -2) ∨ (x = -1 ∧ y = 0) ∨ (x = -1 ∧ y = -2) :=
by sorry

end NUMINAMATH_CALUDE_integer_solutions_of_equation_l824_82436


namespace NUMINAMATH_CALUDE_roots_distance_bound_l824_82487

theorem roots_distance_bound (v w : ℂ) : 
  v ≠ w → 
  (v^401 = 1) → 
  (w^401 = 1) → 
  Complex.abs (v + w) < Real.sqrt (3 + Real.sqrt 5) := by
sorry

end NUMINAMATH_CALUDE_roots_distance_bound_l824_82487


namespace NUMINAMATH_CALUDE_orange_juice_proportion_l824_82465

theorem orange_juice_proportion (oranges : ℝ) (quarts : ℝ) :
  oranges / quarts = 36 / 48 →
  quarts = 6 →
  oranges = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_orange_juice_proportion_l824_82465


namespace NUMINAMATH_CALUDE_ellipse_properties_l824_82491

/-- An ellipse with focal length 2 passing through the point (3/2, √6) -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0
  focal_length : a^2 - b^2 = 1
  passes_through : (3/2)^2 / a^2 + 6 / b^2 = 1

/-- The standard equation of the ellipse -/
def standard_equation (e : Ellipse) : Prop :=
  ∀ x y : ℝ, x^2 / 9 + y^2 / 8 = 1 ↔ x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The trajectory of point E -/
def trajectory_equation (e : Ellipse) : Prop :=
  ∀ x y : ℝ, x ≠ 3 ∧ x ≠ -3 → (x^2 / 9 - y^2 / 8 = 1 ↔
    ∃ x₁ y₁ : ℝ, x₁^2 / e.a^2 + y₁^2 / e.b^2 = 1 ∧ x₁ ≠ 0 ∧ |x₁| < e.a ∧
      y / y₁ = (x + e.a) / (x₁ + e.a) ∧
      y / (-y₁) = (x - e.a) / (x₁ - e.a))

/-- The main theorem to be proved -/
theorem ellipse_properties (e : Ellipse) :
  standard_equation e ∧ trajectory_equation e :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l824_82491


namespace NUMINAMATH_CALUDE_brochure_distribution_l824_82412

theorem brochure_distribution (total_brochures : ℕ) (num_boxes : ℕ) 
  (h1 : total_brochures = 5000) 
  (h2 : num_boxes = 5) : 
  (total_brochures / num_boxes : ℚ) / total_brochures = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_brochure_distribution_l824_82412


namespace NUMINAMATH_CALUDE_square_of_99_l824_82483

theorem square_of_99 : (99 : ℕ) ^ 2 = 9801 := by
  sorry

end NUMINAMATH_CALUDE_square_of_99_l824_82483


namespace NUMINAMATH_CALUDE_count_distinct_sums_of_special_fractions_l824_82451

def IsSpecialFraction (a b : ℕ+) : Prop := a + b = 17

def SumOfSpecialFractions (n : ℕ) : Prop :=
  ∃ (a₁ b₁ a₂ b₂ : ℕ+),
    IsSpecialFraction a₁ b₁ ∧
    IsSpecialFraction a₂ b₂ ∧
    n = (a₁ : ℚ) / b₁ + (a₂ : ℚ) / b₂

theorem count_distinct_sums_of_special_fractions :
  ∃! (s : Finset ℕ),
    (∀ n ∈ s, SumOfSpecialFractions n) ∧
    (∀ n, SumOfSpecialFractions n → n ∈ s) ∧
    s.card = 2 :=
sorry

end NUMINAMATH_CALUDE_count_distinct_sums_of_special_fractions_l824_82451


namespace NUMINAMATH_CALUDE_f_range_l824_82450

/-- The function f(x) = |x-1| + |x-2| -/
def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

/-- The range of f is [1, +∞) -/
theorem f_range :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ y ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_f_range_l824_82450


namespace NUMINAMATH_CALUDE_ellipse_and_line_theorem_l824_82437

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2/3 + y^2 = 1

-- Define the line l
def line_l (m : ℝ) (x y : ℝ) : Prop := x = m*y + 1

-- Define the circle with diameter AB passing through origin
def circle_AB_origin (xA yA xB yB : ℝ) : Prop := xA*xB + yA*yB = 0

theorem ellipse_and_line_theorem :
  -- Given conditions
  let a : ℝ := Real.sqrt 3
  let e : ℝ := Real.sqrt 6 / 3
  let c : ℝ := e * a

  -- Part 1: Prove the standard equation of ellipse C
  (∀ x y : ℝ, ellipse_C x y ↔ x^2/3 + y^2 = 1) ∧

  -- Part 2: Prove the equation of line l
  (∃ m : ℝ, m = Real.sqrt 3 / 3 ∨ m = -Real.sqrt 3 / 3) ∧
  (∀ m : ℝ, (m = Real.sqrt 3 / 3 ∨ m = -Real.sqrt 3 / 3) →
    (∃ xA yA xB yB : ℝ,
      ellipse_C xA yA ∧ ellipse_C xB yB ∧
      line_l m xA yA ∧ line_l m xB yB ∧
      circle_AB_origin xA yA xB yB)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_and_line_theorem_l824_82437


namespace NUMINAMATH_CALUDE_max_B_at_125_l824_82480

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The term B_k in the binomial expansion of (1 + 0.3)^500 -/
def B (k : ℕ) : ℝ := (binomial 500 k) * (0.3 ^ k)

/-- Theorem stating that B_k is largest when k = 125 -/
theorem max_B_at_125 : ∀ k : ℕ, k ≤ 500 → B k ≤ B 125 := by sorry

end NUMINAMATH_CALUDE_max_B_at_125_l824_82480


namespace NUMINAMATH_CALUDE_cloth_sale_price_l824_82479

/-- Calculates the total selling price of cloth given the quantity, profit per meter, and cost price per meter. -/
def totalSellingPrice (quantity : ℕ) (profitPerMeter : ℕ) (costPricePerMeter : ℕ) : ℕ :=
  quantity * (costPricePerMeter + profitPerMeter)

/-- Proves that the total selling price of 85 meters of cloth with a profit of $15 per meter and a cost price of $90 per meter is $8925. -/
theorem cloth_sale_price : totalSellingPrice 85 15 90 = 8925 := by
  sorry

end NUMINAMATH_CALUDE_cloth_sale_price_l824_82479


namespace NUMINAMATH_CALUDE_no_solution_when_m_zero_infinite_solutions_when_m_neg_three_unique_solution_when_m_not_zero_and_not_neg_three_l824_82481

-- Define the system of linear equations
def system (m x y : ℝ) : Prop :=
  m * x + y = -1 ∧ 3 * m * x - m * y = 2 * m + 3

-- Define the determinant of the coefficient matrix
def det_coeff (m : ℝ) : ℝ := -m * (m + 3)

-- Define the determinants for x and y
def det_x (m : ℝ) : ℝ := -m - 3
def det_y (m : ℝ) : ℝ := 2 * m * (m + 3)

-- Theorem for the case when m = 0
theorem no_solution_when_m_zero :
  ¬∃ x y : ℝ, system 0 x y :=
sorry

-- Theorem for the case when m = -3
theorem infinite_solutions_when_m_neg_three :
  ∃ x y : ℝ, system (-3) x y ∧ ∀ t : ℝ, system (-3) (x + t) (y - 3*t) :=
sorry

-- Theorem for the case when m ≠ 0 and m ≠ -3
theorem unique_solution_when_m_not_zero_and_not_neg_three (m : ℝ) (hm : m ≠ 0 ∧ m ≠ -3) :
  ∃! x y : ℝ, system m x y ∧ x = 1/m ∧ y = -2 :=
sorry

end NUMINAMATH_CALUDE_no_solution_when_m_zero_infinite_solutions_when_m_neg_three_unique_solution_when_m_not_zero_and_not_neg_three_l824_82481


namespace NUMINAMATH_CALUDE_system_solution_l824_82429

theorem system_solution :
  ∃! (x y : ℝ),
    Real.sqrt (2016.5 + x) + Real.sqrt (2016.5 + y) = 114 ∧
    Real.sqrt (2016.5 - x) + Real.sqrt (2016.5 - y) = 56 ∧
    x = 1232.5 ∧ y = 1232.5 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l824_82429


namespace NUMINAMATH_CALUDE_typing_job_solution_l824_82448

/-- Represents the time taken by two typists to complete a typing job -/
structure TypingJob where
  combined_time : ℝ  -- Time taken when working together
  sequential_time : ℝ  -- Time taken when working sequentially (half each)
  first_typist_time : ℝ  -- Time for first typist to complete job alone
  second_typist_time : ℝ  -- Time for second typist to complete job alone

/-- Theorem stating the solution to the typing job problem -/
theorem typing_job_solution (job : TypingJob) 
  (h1 : job.combined_time = 12)
  (h2 : job.sequential_time = 25)
  (h3 : job.first_typist_time + job.second_typist_time = 50)
  (h4 : job.first_typist_time * job.second_typist_time = 600) :
  job.first_typist_time = 20 ∧ job.second_typist_time = 30 := by
  sorry

#check typing_job_solution

end NUMINAMATH_CALUDE_typing_job_solution_l824_82448


namespace NUMINAMATH_CALUDE_fruits_left_after_selling_l824_82445

def initial_oranges : ℕ := 40
def initial_apples : ℕ := 70
def orange_sold_fraction : ℚ := 1/4
def apple_sold_fraction : ℚ := 1/2

theorem fruits_left_after_selling :
  (initial_oranges - orange_sold_fraction * initial_oranges) +
  (initial_apples - apple_sold_fraction * initial_apples) = 65 :=
by sorry

end NUMINAMATH_CALUDE_fruits_left_after_selling_l824_82445


namespace NUMINAMATH_CALUDE_three_team_soccer_game_total_score_l824_82475

/-- Represents the score of a team in a half of the game -/
structure HalfScore where
  regular : ℕ
  penalties : ℕ

/-- Represents the score of a team for the whole game -/
structure GameScore where
  first_half : HalfScore
  second_half : HalfScore

/-- Calculate the total score for a team -/
def total_score (score : GameScore) : ℕ :=
  score.first_half.regular + score.first_half.penalties +
  score.second_half.regular + score.second_half.penalties

theorem three_team_soccer_game_total_score :
  let team_a : GameScore := {
    first_half := { regular := 8, penalties := 0 },
    second_half := { regular := 8, penalties := 1 }
  }
  let team_b : GameScore := {
    first_half := { regular := 4, penalties := 0 },
    second_half := { regular := 8, penalties := 2 }
  }
  let team_c : GameScore := {
    first_half := { regular := 8, penalties := 0 },
    second_half := { regular := 11, penalties := 0 }
  }
  team_b.first_half.regular = team_a.first_half.regular / 2 →
  team_c.first_half.regular = 2 * team_b.first_half.regular →
  team_a.second_half.regular = team_c.first_half.regular →
  team_b.second_half.regular = team_a.first_half.regular →
  team_c.second_half.regular = team_b.second_half.regular + 3 →
  total_score team_a + total_score team_b + total_score team_c = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_three_team_soccer_game_total_score_l824_82475


namespace NUMINAMATH_CALUDE_repetend_of_5_17_l824_82402

/-- The repetend of a fraction is the repeating sequence of digits in its decimal representation. -/
def is_repetend (n : ℕ) (d : ℕ) (r : ℕ) : Prop :=
  ∃ (k : ℕ), 10^6 * (10 * n - d * r) = d * (10^k - 1)

/-- The 6-digit repetend in the decimal representation of 5/17 is 294117. -/
theorem repetend_of_5_17 : is_repetend 5 17 294117 := by
  sorry

end NUMINAMATH_CALUDE_repetend_of_5_17_l824_82402


namespace NUMINAMATH_CALUDE_trig_identity_l824_82477

theorem trig_identity (x y : ℝ) :
  Real.sin (2 * x - y) * Real.cos (3 * y) + Real.cos (2 * x - y) * Real.sin (3 * y) = Real.sin (2 * x + 2 * y) := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l824_82477


namespace NUMINAMATH_CALUDE_value_of_expression_l824_82433

theorem value_of_expression (a : ℝ) (h : a^2 + 2*a + 1 = 0) : 2*a^2 + 4*a - 3 = -5 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l824_82433


namespace NUMINAMATH_CALUDE_sum_of_edges_l824_82493

/-- The number of edges in a triangular pyramid -/
def triangular_pyramid_edges : ℕ := 6

/-- The number of edges in a triangular prism -/
def triangular_prism_edges : ℕ := 9

/-- The sum of edges in a triangular pyramid and a triangular prism -/
theorem sum_of_edges : triangular_pyramid_edges + triangular_prism_edges = 15 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_edges_l824_82493


namespace NUMINAMATH_CALUDE_g_of_x_plus_3_l824_82447

/-- Given a function g(x) = x(x+3)/3, prove that g(x+3) = (x^2 + 9x + 18) / 3 -/
theorem g_of_x_plus_3 (x : ℝ) : 
  let g : ℝ → ℝ := fun x => x * (x + 3) / 3
  g (x + 3) = (x^2 + 9*x + 18) / 3 := by
sorry

end NUMINAMATH_CALUDE_g_of_x_plus_3_l824_82447


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l824_82421

theorem least_subtraction_for_divisibility : 
  ∃! x : ℕ, x ≤ 15 ∧ (899830 - x) % 16 = 0 ∧ ∀ y : ℕ, y < x → (899830 - y) % 16 ≠ 0 :=
by
  use 6
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l824_82421


namespace NUMINAMATH_CALUDE_donovan_candles_count_l824_82494

/-- The number of candles Donovan brought in -/
def donovans_candles : ℕ := 20

/-- The number of candles in Kalani's bedroom -/
def bedroom_candles : ℕ := 20

/-- The number of candles in the living room -/
def living_room_candles : ℕ := bedroom_candles / 2

/-- The total number of candles in the house -/
def total_candles : ℕ := 50

theorem donovan_candles_count :
  donovans_candles = total_candles - bedroom_candles - living_room_candles :=
by sorry

end NUMINAMATH_CALUDE_donovan_candles_count_l824_82494


namespace NUMINAMATH_CALUDE_f_properties_l824_82423

/-- The function f(x) = x³ - 5x² + 3x -/
def f (x : ℝ) : ℝ := x^3 - 5*x^2 + 3*x

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 - 10*x + 3

theorem f_properties :
  (f' 3 = 0) ∧ 
  (f 1 = -1) ∧
  (∀ x ∈ Set.Icc 2 4, f x ≥ -9) ∧
  (f 3 = -9) ∧
  (∀ x ∈ Set.Icc 2 4, f x ≤ -4) ∧
  (f 4 = -4) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l824_82423
