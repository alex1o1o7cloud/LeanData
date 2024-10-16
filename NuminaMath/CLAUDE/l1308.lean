import Mathlib

namespace NUMINAMATH_CALUDE_work_rate_comparison_l1308_130805

theorem work_rate_comparison (x : ℝ) (work : ℝ) : 
  x > 0 →
  (x + 1) * 21 = x * 28 →
  x = 3 := by
sorry

end NUMINAMATH_CALUDE_work_rate_comparison_l1308_130805


namespace NUMINAMATH_CALUDE_sqrt_two_simplification_l1308_130889

theorem sqrt_two_simplification : 3 * Real.sqrt 2 - Real.sqrt 2 = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_simplification_l1308_130889


namespace NUMINAMATH_CALUDE_circle_radius_condition_l1308_130869

theorem circle_radius_condition (c : ℝ) : 
  (∀ x y : ℝ, x^2 + 4*x + y^2 + 8*y + c = 0 ↔ (x + 2)^2 + (y + 4)^2 = 25) → 
  c = -5 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_condition_l1308_130869


namespace NUMINAMATH_CALUDE_simplify_fraction_l1308_130838

theorem simplify_fraction (x : ℝ) (h : x ≠ 2) : (x^2 / (x - 2)) - (2*x / (x - 2)) = x := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1308_130838


namespace NUMINAMATH_CALUDE_max_pages_for_15_dollars_l1308_130884

/-- The cost in cents to copy 4 pages -/
def cost_per_4_pages : ℕ := 7

/-- The number of pages that can be copied for 4 cents -/
def pages_per_4_cents : ℕ := 4

/-- The amount in dollars available for copying -/
def available_dollars : ℕ := 15

/-- Converts dollars to cents -/
def dollars_to_cents (dollars : ℕ) : ℕ := dollars * 100

/-- Calculates the maximum number of whole pages that can be copied -/
def max_pages : ℕ := 
  (dollars_to_cents available_dollars * pages_per_4_cents) / cost_per_4_pages

theorem max_pages_for_15_dollars : max_pages = 857 := by
  sorry

end NUMINAMATH_CALUDE_max_pages_for_15_dollars_l1308_130884


namespace NUMINAMATH_CALUDE_problem_solution_l1308_130844

theorem problem_solution :
  ∀ (a b : ℝ),
  let A := 2 * a^2 + 3 * a * b - 2 * a - 1
  let B := -a^2 + a * b + a + 3
  (a = -1 ∧ b = 10 → 4 * A - (3 * A - 2 * B) = -45) ∧
  (a * b = 1 → 4 * A - (3 * A - 2 * B) = 10) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1308_130844


namespace NUMINAMATH_CALUDE_semicircle_area_with_inscribed_rectangle_l1308_130887

/-- The area of a semicircle that circumscribes a 2 × 3 rectangle with the longer side on the diameter -/
theorem semicircle_area_with_inscribed_rectangle : 
  ∀ (semicircle_area : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ),
    rectangle_width = 2 →
    rectangle_length = 3 →
    semicircle_area = (9 * Real.pi) / 4 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_area_with_inscribed_rectangle_l1308_130887


namespace NUMINAMATH_CALUDE_vector_magnitude_proof_l1308_130843

open Real

theorem vector_magnitude_proof (a b : ℝ × ℝ) :
  let angle := 60 * π / 180
  norm a = 2 ∧ norm b = 5 ∧ 
  a.1 * b.1 + a.2 * b.2 = norm a * norm b * cos angle →
  norm (2 • a - b) = sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_proof_l1308_130843


namespace NUMINAMATH_CALUDE_pasta_bins_l1308_130871

theorem pasta_bins (soup_bins vegetables_bins total_bins : ℝ)
  (h1 : soup_bins = 0.125)
  (h2 : vegetables_bins = 0.125)
  (h3 : total_bins = 0.75) :
  total_bins - (soup_bins + vegetables_bins) = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_pasta_bins_l1308_130871


namespace NUMINAMATH_CALUDE_additional_cats_needed_prove_additional_cats_l1308_130816

theorem additional_cats_needed (total_mice : ℕ) (initial_cats : ℕ) (initial_days : ℕ) (total_days : ℕ) : ℕ :=
  let initial_work := total_mice / 2
  let remaining_work := total_mice - initial_work
  let initial_rate := initial_work / (initial_cats * initial_days)
  let remaining_days := total_days - initial_days
  let additional_cats := (remaining_work / (initial_rate * remaining_days)) - initial_cats
  additional_cats

theorem prove_additional_cats :
  additional_cats_needed 100 2 5 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_additional_cats_needed_prove_additional_cats_l1308_130816


namespace NUMINAMATH_CALUDE_parallel_line_y_intercept_l1308_130861

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

theorem parallel_line_y_intercept
  (b : Line)
  (given_line : Line)
  (p : Point) :
  parallel b given_line →
  given_line.slope = -3 →
  given_line.intercept = 6 →
  p.x = 3 →
  p.y = -1 →
  pointOnLine p b →
  b.intercept = 8 := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_y_intercept_l1308_130861


namespace NUMINAMATH_CALUDE_hexagon_arithmetic_progression_angle_l1308_130812

theorem hexagon_arithmetic_progression_angle (a d : ℝ) :
  (∀ i : Fin 6, 0 ≤ i.val → i.val < 6 → 0 < a + i.val * d) →
  (6 * a + 15 * d = 720) →
  ∃ i : Fin 6, a + i.val * d = 240 :=
sorry

end NUMINAMATH_CALUDE_hexagon_arithmetic_progression_angle_l1308_130812


namespace NUMINAMATH_CALUDE_number_ratio_problem_l1308_130860

theorem number_ratio_problem (x y z : ℝ) : 
  x = 18 →  -- The smallest number is 18
  y = 4 * x →  -- The second number is 4 times the first
  ∃ k : ℝ, z = k * y →  -- The third number is some multiple of the second
  (x + y + z) / 3 = 78 →  -- Their average is 78
  z / y = 2 :=  -- The ratio of the third to the second is 2
by sorry

end NUMINAMATH_CALUDE_number_ratio_problem_l1308_130860


namespace NUMINAMATH_CALUDE_zeros_in_square_of_nines_zeros_count_in_2019_nines_squared_l1308_130857

theorem zeros_in_square_of_nines (n : ℕ) : 
  (10^n - 1)^2 % 10^(n-1) = 1 ∧ (10^n - 1)^2 % 10^n ≠ 0 := by
  sorry

theorem zeros_count_in_2019_nines_squared : 
  ∃ k : ℕ, (10^2019 - 1)^2 = k * 10^2018 + 1 ∧ k % 10 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_zeros_in_square_of_nines_zeros_count_in_2019_nines_squared_l1308_130857


namespace NUMINAMATH_CALUDE_generalized_distributive_laws_l1308_130810

variable {α : Type*}
variable {I : Type*}
variable (𝔍 : I → Type*)
variable (A : (i : I) → 𝔍 i → Set α)

def paths (𝔍 : I → Type*) := (i : I) → 𝔍 i

theorem generalized_distributive_laws :
  (⋃ i, ⋂ j, A i j) = (⋂ f : paths 𝔍, ⋃ i, A i (f i)) ∧
  (⋂ i, ⋃ j, A i j) = (⋃ f : paths 𝔍, ⋂ i, A i (f i)) :=
sorry

end NUMINAMATH_CALUDE_generalized_distributive_laws_l1308_130810


namespace NUMINAMATH_CALUDE_boys_playing_both_sports_l1308_130829

theorem boys_playing_both_sports (total : ℕ) (basketball : ℕ) (football : ℕ) (neither : ℕ) :
  total = 30 →
  basketball = 18 →
  football = 21 →
  neither = 4 →
  basketball + football - (total - neither) = 13 := by
  sorry

end NUMINAMATH_CALUDE_boys_playing_both_sports_l1308_130829


namespace NUMINAMATH_CALUDE_unique_solution_iff_prime_l1308_130858

theorem unique_solution_iff_prime (n : ℕ+) :
  (∃! a : ℕ, a < n.val.factorial ∧ (n.val.factorial ∣ a^n.val + 1)) ↔ Nat.Prime n.val := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_iff_prime_l1308_130858


namespace NUMINAMATH_CALUDE_increasing_sequence_condition_sufficient_condition_not_necessary_condition_l1308_130809

theorem increasing_sequence_condition (k : ℝ) : 
  (∀ n : ℕ, n > 1 → (n^2 + k*n) < ((n+1)^2 + k*(n+1))) ↔ k > -3 :=
by sorry

theorem sufficient_condition (k : ℝ) :
  k ≥ -2 → ∀ n : ℕ, n > 1 → (n^2 + k*n) < ((n+1)^2 + k*(n+1)) :=
by sorry

theorem not_necessary_condition :
  ∃ k : ℝ, k < -2 ∧ (∀ n : ℕ, n > 1 → (n^2 + k*n) < ((n+1)^2 + k*(n+1))) :=
by sorry

end NUMINAMATH_CALUDE_increasing_sequence_condition_sufficient_condition_not_necessary_condition_l1308_130809


namespace NUMINAMATH_CALUDE_all_numbers_even_l1308_130818

theorem all_numbers_even 
  (A B C D E : ℤ) 
  (h1 : Even (A + B + C))
  (h2 : Even (A + B + D))
  (h3 : Even (A + B + E))
  (h4 : Even (A + C + D))
  (h5 : Even (A + C + E))
  (h6 : Even (A + D + E))
  (h7 : Even (B + C + D))
  (h8 : Even (B + C + E))
  (h9 : Even (B + D + E))
  (h10 : Even (C + D + E)) :
  Even A ∧ Even B ∧ Even C ∧ Even D ∧ Even E := by
  sorry

#check all_numbers_even

end NUMINAMATH_CALUDE_all_numbers_even_l1308_130818


namespace NUMINAMATH_CALUDE_system_inconsistent_l1308_130873

-- Define the coefficient matrix A
def A : Matrix (Fin 4) (Fin 5) ℚ :=
  !![1, 2, -1, 3, -1;
     2, -1, 3, 1, -1;
     1, -1, 1, 2, 0;
     4, 0, 3, 6, -2]

-- Define the augmented matrix Â
def A_hat : Matrix (Fin 4) (Fin 6) ℚ :=
  !![1, 2, -1, 3, -1, 0;
     2, -1, 3, 1, -1, -1;
     1, -1, 1, 2, 0, 2;
     4, 0, 3, 6, -2, 5]

-- Theorem statement
theorem system_inconsistent :
  Matrix.rank A < Matrix.rank A_hat :=
sorry

end NUMINAMATH_CALUDE_system_inconsistent_l1308_130873


namespace NUMINAMATH_CALUDE_alcohol_concentration_is_40_percent_l1308_130839

-- Define the ratios of water to alcohol in solutions A and B
def waterToAlcoholRatioA : Rat := 4 / 1
def waterToAlcoholRatioB : Rat := 2 / 3

-- Define the amount of each solution mixed (assuming 1 unit each)
def amountA : Rat := 1
def amountB : Rat := 1

-- Define the function to calculate the alcohol concentration in the mixed solution
def alcoholConcentration (ratioA ratioB amountA amountB : Rat) : Rat :=
  let waterA := amountA * (ratioA / (ratioA + 1))
  let alcoholA := amountA * (1 / (ratioA + 1))
  let waterB := amountB * (ratioB / (ratioB + 1))
  let alcoholB := amountB * (1 / (ratioB + 1))
  let totalAlcohol := alcoholA + alcoholB
  let totalMixture := waterA + alcoholA + waterB + alcoholB
  totalAlcohol / totalMixture

-- Theorem statement
theorem alcohol_concentration_is_40_percent :
  alcoholConcentration waterToAlcoholRatioA waterToAlcoholRatioB amountA amountB = 2/5 := by
  sorry


end NUMINAMATH_CALUDE_alcohol_concentration_is_40_percent_l1308_130839


namespace NUMINAMATH_CALUDE_complex_power_problem_l1308_130862

theorem complex_power_problem (z : ℂ) : 
  (1 + z) / (1 - z) = Complex.I → z^2023 = -Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_power_problem_l1308_130862


namespace NUMINAMATH_CALUDE_smallest_equal_hotdogs_and_buns_l1308_130828

theorem smallest_equal_hotdogs_and_buns :
  ∃ (n : ℕ), n > 0 ∧ (∃ (m : ℕ), m > 0 ∧ 5 * n = 7 * m) ∧
  (∀ (k : ℕ), k > 0 → (∃ (j : ℕ), j > 0 ∧ 5 * k = 7 * j) → k ≥ n) ∧
  n = 7 := by
sorry

end NUMINAMATH_CALUDE_smallest_equal_hotdogs_and_buns_l1308_130828


namespace NUMINAMATH_CALUDE_exactly_two_girls_together_probability_l1308_130819

/-- The number of boys in the group -/
def num_boys : ℕ := 2

/-- The number of girls in the group -/
def num_girls : ℕ := 3

/-- The total number of students -/
def total_students : ℕ := num_boys + num_girls

/-- The number of ways to arrange all students -/
def total_arrangements : ℕ := Nat.factorial total_students

/-- The number of ways to arrange students with exactly 2 girls together -/
def favorable_arrangements : ℕ := 
  Nat.choose 3 2 * Nat.factorial 2 * Nat.factorial 3

/-- The probability of exactly 2 out of 3 girls standing next to each other -/
def probability : ℚ := favorable_arrangements / total_arrangements

theorem exactly_two_girls_together_probability : 
  probability = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_exactly_two_girls_together_probability_l1308_130819


namespace NUMINAMATH_CALUDE_age_problem_l1308_130852

theorem age_problem (kolya_last_year : ℝ) :
  let vera_last_year := 2 * kolya_last_year
  let victor_last_year := 3.5 * kolya_last_year
  let kolya_now := kolya_last_year + 1
  let vera_now := vera_last_year + 1
  let victor_now := victor_last_year + 1
  let years_until_double := victor_now
  let kolya_future := kolya_now + years_until_double
  let vera_future := vera_now + years_until_double
  (vera_future - kolya_future = 4) →
  (kolya_now = 5 ∧ vera_now = 9 ∧ victor_now = 15) :=
by sorry

end NUMINAMATH_CALUDE_age_problem_l1308_130852


namespace NUMINAMATH_CALUDE_mary_pies_count_l1308_130842

theorem mary_pies_count (apples_per_pie : ℕ) (harvested_apples : ℕ) (apples_to_buy : ℕ) :
  apples_per_pie = 8 →
  harvested_apples = 50 →
  apples_to_buy = 30 →
  (harvested_apples + apples_to_buy) / apples_per_pie = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_mary_pies_count_l1308_130842


namespace NUMINAMATH_CALUDE_bus_stop_problem_l1308_130817

/-- The number of children who got on the bus at a stop -/
def children_got_on (initial : ℕ) (final : ℕ) : ℕ :=
  final - initial

theorem bus_stop_problem :
  let initial_children : ℕ := 64
  let final_children : ℕ := 78
  children_got_on initial_children final_children = 14 := by
  sorry

end NUMINAMATH_CALUDE_bus_stop_problem_l1308_130817


namespace NUMINAMATH_CALUDE_parabola_sum_of_coefficients_l1308_130866

/-- A quadratic function with coefficients p, q, and r -/
def quadratic_function (p q r : ℝ) (x : ℝ) : ℝ := p * x^2 + q * x + r

theorem parabola_sum_of_coefficients 
  (p q r : ℝ) 
  (h_vertex : quadratic_function p q r 3 = 4)
  (h_symmetry : ∀ (x : ℝ), quadratic_function p q r (3 + x) = quadratic_function p q r (3 - x))
  (h_point1 : quadratic_function p q r 1 = 10)
  (h_point2 : quadratic_function p q r (-1) = 14) :
  p + q + r = 10 := by
  sorry

end NUMINAMATH_CALUDE_parabola_sum_of_coefficients_l1308_130866


namespace NUMINAMATH_CALUDE_distribute_five_books_three_students_l1308_130823

/-- The number of ways to distribute n different books among k students,
    with each student receiving at least one book -/
def distribute_books (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 different books among 3 students,
    with each student receiving at least one book, is 150 -/
theorem distribute_five_books_three_students :
  distribute_books 5 3 = 150 := by sorry

end NUMINAMATH_CALUDE_distribute_five_books_three_students_l1308_130823


namespace NUMINAMATH_CALUDE_equation_solutions_count_l1308_130879

theorem equation_solutions_count :
  let f : ℝ → ℝ := λ x => 2 * Real.sqrt 2 * (Real.sin (π * x / 4))^3 - Real.cos (π * (1 - x) / 4)
  ∃! (s : Finset ℝ), 
    (∀ x ∈ s, f x = 0 ∧ 0 ≤ x ∧ x ≤ 2020) ∧
    (∀ x, f x = 0 ∧ 0 ≤ x ∧ x ≤ 2020 → x ∈ s) ∧
    Finset.card s = 505 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_count_l1308_130879


namespace NUMINAMATH_CALUDE_range_of_x_l1308_130892

theorem range_of_x (x : ℝ) : (x^2 - 2*x - 3 ≥ 0) ∧ ¬(|1 - x/2| < 1) ↔ x ≤ -1 ∨ x ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l1308_130892


namespace NUMINAMATH_CALUDE_largest_divisor_of_difference_of_squares_l1308_130876

theorem largest_divisor_of_difference_of_squares (m n : ℕ) : 
  Even m → Even n → n < m → 
  (∃ (k : ℕ), ∀ (a b : ℕ), Even a → Even b → b < a → 
    k ∣ (a^2 - b^2) ∧ k = 16 ∧ ∀ (l : ℕ), (∀ (x y : ℕ), Even x → Even y → y < x → l ∣ (x^2 - y^2)) → l ≤ k) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_difference_of_squares_l1308_130876


namespace NUMINAMATH_CALUDE_minimum_newspapers_to_recover_cost_l1308_130894

/-- The cost of Mary's scooter in dollars -/
def scooter_cost : ℕ := 3000

/-- The amount Mary earns per newspaper delivered in dollars -/
def earning_per_newspaper : ℕ := 8

/-- The transportation cost per newspaper delivery in dollars -/
def transport_cost_per_newspaper : ℕ := 4

/-- The net earning per newspaper in dollars -/
def net_earning_per_newspaper : ℕ := earning_per_newspaper - transport_cost_per_newspaper

theorem minimum_newspapers_to_recover_cost :
  ∃ n : ℕ, n * net_earning_per_newspaper ≥ scooter_cost ∧
  ∀ m : ℕ, m * net_earning_per_newspaper ≥ scooter_cost → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_minimum_newspapers_to_recover_cost_l1308_130894


namespace NUMINAMATH_CALUDE_triangle_side_length_l1308_130835

theorem triangle_side_length (a b x : ℝ) : 
  a = 2 → 
  b = 6 → 
  x^2 - 10*x + 21 = 0 → 
  x > 0 → 
  a + x > b → 
  b + x > a → 
  a + b > x → 
  x = 7 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1308_130835


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1308_130847

theorem sufficient_not_necessary (a b : ℝ) :
  (∀ a b : ℝ, a > 1 ∧ b > 2 → a + b > 3 ∧ a * b > 2) ∧
  (∃ a b : ℝ, a + b > 3 ∧ a * b > 2 ∧ ¬(a > 1 ∧ b > 2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1308_130847


namespace NUMINAMATH_CALUDE_cubic_sum_equals_twenty_l1308_130867

theorem cubic_sum_equals_twenty (x y z : ℝ) (h1 : x = y + z) (h2 : x = 2) :
  x^3 + 3*y^2 + 3*z^2 + 3*x*y*z = 20 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_equals_twenty_l1308_130867


namespace NUMINAMATH_CALUDE_squat_rack_cost_squat_rack_cost_proof_l1308_130897

/-- The cost of a squat rack, given that the barbell costs 1/10 as much and the total is $2750 -/
theorem squat_rack_cost : ℝ → ℝ → Prop :=
  fun (squat_rack_cost barbell_cost : ℝ) =>
    barbell_cost = squat_rack_cost / 10 ∧
    squat_rack_cost + barbell_cost = 2750 →
    squat_rack_cost = 2500

/-- Proof of the squat rack cost theorem -/
theorem squat_rack_cost_proof : squat_rack_cost 2500 250 := by
  sorry

end NUMINAMATH_CALUDE_squat_rack_cost_squat_rack_cost_proof_l1308_130897


namespace NUMINAMATH_CALUDE_widget_purchase_l1308_130808

theorem widget_purchase (W : ℝ) (h1 : 6 * W = 8 * (W - 2)) : 6 * W = 48 := by
  sorry

end NUMINAMATH_CALUDE_widget_purchase_l1308_130808


namespace NUMINAMATH_CALUDE_min_cube_volume_for_pyramid_l1308_130832

/-- Represents a square-based pyramid -/
structure Pyramid where
  height : ℝ
  baseLength : ℝ

/-- Represents a cube -/
structure Cube where
  sideLength : ℝ

/-- Calculates the volume of a cube -/
def cubeVolume (c : Cube) : ℝ := c.sideLength ^ 3

/-- Checks if a pyramid fits inside a cube -/
def pyramidFitsInCube (p : Pyramid) (c : Cube) : Prop :=
  c.sideLength ≥ p.height ∧ c.sideLength ≥ p.baseLength

theorem min_cube_volume_for_pyramid (p : Pyramid) (h1 : p.height = 18) (h2 : p.baseLength = 15) :
  ∃ (c : Cube), pyramidFitsInCube p c ∧ cubeVolume c = 5832 ∧
  ∀ (c' : Cube), pyramidFitsInCube p c' → cubeVolume c' ≥ 5832 := by
  sorry

end NUMINAMATH_CALUDE_min_cube_volume_for_pyramid_l1308_130832


namespace NUMINAMATH_CALUDE_solution_difference_l1308_130853

theorem solution_difference (r s : ℝ) : 
  (((5 * r - 15) / (r^2 + 3*r - 18) = r + 3) ∧
   ((5 * s - 15) / (s^2 + 3*s - 18) = s + 3) ∧
   (r ≠ s) ∧ (r > s)) →
  r - s = 13 := by
sorry

end NUMINAMATH_CALUDE_solution_difference_l1308_130853


namespace NUMINAMATH_CALUDE_unique_sum_of_squares_l1308_130836

def is_sum_of_squares (n : ℕ) (k : ℕ) : Prop :=
  ∃ (a₁ a₂ a₃ a₄ a₅ : ℕ), n = a₁^2 + a₂^2 + a₃^2 + a₄^2 + a₅^2 ∧ (a₁ ≠ 0 → k ≥ 1) ∧
    (a₂ ≠ 0 → k ≥ 2) ∧ (a₃ ≠ 0 → k ≥ 3) ∧ (a₄ ≠ 0 → k ≥ 4) ∧ (a₅ ≠ 0 → k = 5)

def has_unique_representation (n : ℕ) : Prop :=
  ∃! (k : ℕ) (a₁ a₂ a₃ a₄ a₅ : ℕ), k ≤ 5 ∧ is_sum_of_squares n k ∧
    n = a₁^2 + a₂^2 + a₃^2 + a₄^2 + a₅^2

theorem unique_sum_of_squares :
  {n : ℕ | has_unique_representation n} = {1, 2, 3, 6, 7, 15} := by sorry

end NUMINAMATH_CALUDE_unique_sum_of_squares_l1308_130836


namespace NUMINAMATH_CALUDE_exists_nonnegative_product_polynomial_l1308_130821

theorem exists_nonnegative_product_polynomial (f : Polynomial ℝ) 
  (h_no_nonneg_root : ∀ x : ℝ, x ≥ 0 → f.eval x ≠ 0) :
  ∃ h : Polynomial ℝ, ∀ i : ℕ, (f * h).coeff i ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_nonnegative_product_polynomial_l1308_130821


namespace NUMINAMATH_CALUDE_isosceles_similar_triangle_perimeter_l1308_130877

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  positive_a : a > 0
  positive_b : b > 0
  positive_c : c > 0
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- Defines similarity between two triangles -/
def similar (t1 t2 : Triangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ 
    t2.a = k * t1.a ∧
    t2.b = k * t1.b ∧
    t2.c = k * t1.c

/-- Calculates the perimeter of a triangle -/
def perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

theorem isosceles_similar_triangle_perimeter :
  ∀ (t1 t2 : Triangle),
    t1.a = 15 ∧ t1.b = 30 ∧ t1.c = 30 →
    similar t1 t2 →
    t2.a = 75 →
    perimeter t2 = 375 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_similar_triangle_perimeter_l1308_130877


namespace NUMINAMATH_CALUDE_abc_ordering_l1308_130880

theorem abc_ordering :
  let a := (3/5: ℝ) ^ (2/5: ℝ)
  let b := (2/5: ℝ) ^ (3/5: ℝ)
  let c := (2/5: ℝ) ^ (2/5: ℝ)
  b < c ∧ c < a := by
  sorry

end NUMINAMATH_CALUDE_abc_ordering_l1308_130880


namespace NUMINAMATH_CALUDE_lcm_36_100_l1308_130888

theorem lcm_36_100 : Nat.lcm 36 100 = 900 := by
  sorry

end NUMINAMATH_CALUDE_lcm_36_100_l1308_130888


namespace NUMINAMATH_CALUDE_factorial_prime_factorization_l1308_130896

theorem factorial_prime_factorization :
  ∃ (i k m p : ℕ+),
    (8 : ℕ).factorial = 2^(i.val) * 3^(k.val) * 5^(m.val) * 7^(p.val) ∧
    i.val + k.val + m.val + p.val = 11 := by
  sorry

end NUMINAMATH_CALUDE_factorial_prime_factorization_l1308_130896


namespace NUMINAMATH_CALUDE_souvenir_discount_equation_l1308_130830

/-- Proves that for a souvenir with given original and final prices after two consecutive discounts, 
    the equation relating these prices and the discount percentage is correct. -/
theorem souvenir_discount_equation (a : ℝ) : 
  let original_price : ℝ := 168
  let final_price : ℝ := 128
  original_price * (1 - a / 100)^2 = final_price := by
  sorry

end NUMINAMATH_CALUDE_souvenir_discount_equation_l1308_130830


namespace NUMINAMATH_CALUDE_sale_increase_percentage_l1308_130841

theorem sale_increase_percentage
  (original_fee : ℝ)
  (fee_reduction_percentage : ℝ)
  (visitor_increase_percentage : ℝ)
  (h1 : original_fee = 1)
  (h2 : fee_reduction_percentage = 25)
  (h3 : visitor_increase_percentage = 60) :
  let new_fee := original_fee * (1 - fee_reduction_percentage / 100)
  let visitor_multiplier := 1 + visitor_increase_percentage / 100
  let sale_increase_percentage := (new_fee * visitor_multiplier - 1) * 100
  sale_increase_percentage = 20 :=
by sorry

end NUMINAMATH_CALUDE_sale_increase_percentage_l1308_130841


namespace NUMINAMATH_CALUDE_other_solution_of_quadratic_l1308_130885

theorem other_solution_of_quadratic (x₁ : ℚ) :
  x₁ = 3/5 →
  (30 * x₁^2 + 13 = 47 * x₁ - 2) →
  ∃ x₂ : ℚ, x₂ ≠ x₁ ∧ x₂ = 5/6 ∧ 30 * x₂^2 + 13 = 47 * x₂ - 2 := by
  sorry

end NUMINAMATH_CALUDE_other_solution_of_quadratic_l1308_130885


namespace NUMINAMATH_CALUDE_solution_of_equation_l1308_130826

theorem solution_of_equation (x : ℝ) : (5 / (x + 1) - 4 / x = 0) ↔ (x = 4) :=
by sorry

end NUMINAMATH_CALUDE_solution_of_equation_l1308_130826


namespace NUMINAMATH_CALUDE_cube_sum_implies_sum_bound_l1308_130895

theorem cube_sum_implies_sum_bound (p q : ℝ) (h : p^3 + q^3 = 2) : p + q ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_implies_sum_bound_l1308_130895


namespace NUMINAMATH_CALUDE_student_score_l1308_130883

theorem student_score (num_questions num_correct_answers points_per_question : ℕ) 
  (h1 : num_questions = 5)
  (h2 : num_correct_answers = 3)
  (h3 : points_per_question = 2) :
  num_correct_answers * points_per_question = 6 := by sorry

end NUMINAMATH_CALUDE_student_score_l1308_130883


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1308_130803

/-- An arithmetic sequence with first term a₁ and common difference d -/
def arithmeticSequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ :=
  λ n => a₁ + (n - 1 : ℝ) * d

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) (a₁ d : ℝ) 
  (h_arith : a = arithmeticSequence a₁ d)
  (h_first : a 1 = 5)
  (h_sum : a 6 + a 8 = 58) :
  d = 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1308_130803


namespace NUMINAMATH_CALUDE_max_triangle_area_ellipse_circle_intersection_l1308_130870

/-- Given an ellipse E and a line x = t intersecting it, this theorem proves
    the maximum area of triangle ABC formed by the intersection of a circle
    with the y-axis, where the circle's diameter is the chord of the ellipse. -/
theorem max_triangle_area_ellipse_circle_intersection
  (a : ℝ) (t : ℝ) 
  (ha : a > Real.sqrt 3) 
  (ht : t > 0) 
  (he : Real.sqrt (a^2 - 3) / a = 1/2) :
  let E := {p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / 3 = 1}
  let M := (t, Real.sqrt ((1 - t^2 / a^2) * 3))
  let N := (t, -Real.sqrt ((1 - t^2 / a^2) * 3))
  let C := {p : ℝ × ℝ | (p.1 - t)^2 + p.2^2 = ((M.2 - N.2) / 2)^2}
  let A := (0, Real.sqrt ((M.2 - N.2)^2 / 4 - t^2))
  let B := (0, -Real.sqrt ((M.2 - N.2)^2 / 4 - t^2))
  ∃ (tmax : ℝ), tmax > 0 ∧ 
    (∀ t' > 0, t' * Real.sqrt (12 - 7 * t'^2) / 2 ≤ tmax * Real.sqrt (12 - 7 * tmax^2) / 2) ∧
    tmax * Real.sqrt (12 - 7 * tmax^2) / 2 = 3 * Real.sqrt 7 / 7 :=
by sorry

end NUMINAMATH_CALUDE_max_triangle_area_ellipse_circle_intersection_l1308_130870


namespace NUMINAMATH_CALUDE_f_unique_solution_l1308_130874

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2^(-x) else Real.log x / Real.log 81

theorem f_unique_solution :
  ∃! x, f x = 1/4 ∧ x ∈ Set.univ := by sorry

end NUMINAMATH_CALUDE_f_unique_solution_l1308_130874


namespace NUMINAMATH_CALUDE_decagon_diagonals_from_vertex_l1308_130827

/-- The number of sides in a regular decagon -/
def decagon_sides : ℕ := 10

/-- The number of diagonals from one vertex of a polygon with n sides -/
def diagonals_from_vertex (n : ℕ) : ℕ := n - 3

/-- Theorem: The number of diagonals from one vertex of a regular decagon is 7 -/
theorem decagon_diagonals_from_vertex : 
  diagonals_from_vertex decagon_sides = 7 := by sorry

end NUMINAMATH_CALUDE_decagon_diagonals_from_vertex_l1308_130827


namespace NUMINAMATH_CALUDE_min_value_and_ellipse_l1308_130811

theorem min_value_and_ellipse (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : ∀ x : ℝ, x > 0 → (a + b) * x - 1 ≤ x^2) :
  (∀ c d : ℝ, c > 0 → d > 0 → 1 / c + 1 / d ≥ 2) ∧
  (1 / a^2 + 1 / b^2 > 1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_and_ellipse_l1308_130811


namespace NUMINAMATH_CALUDE_probability_one_doctor_one_nurse_l1308_130833

/-- The probability of selecting exactly 1 doctor and 1 nurse from a group of 3 doctors and 2 nurses, when choosing 2 people randomly. -/
theorem probability_one_doctor_one_nurse :
  let total_people : ℕ := 5
  let doctors : ℕ := 3
  let nurses : ℕ := 2
  let selection : ℕ := 2
  Nat.choose total_people selection ≠ 0 →
  (Nat.choose doctors 1 * Nat.choose nurses 1 : ℚ) / Nat.choose total_people selection = 3/5 :=
by sorry

end NUMINAMATH_CALUDE_probability_one_doctor_one_nurse_l1308_130833


namespace NUMINAMATH_CALUDE_smallest_x_absolute_value_equation_l1308_130800

theorem smallest_x_absolute_value_equation :
  let f : ℝ → ℝ := λ x ↦ |2 * x + 5|
  ∃ x : ℝ, f x = 18 ∧ ∀ y : ℝ, f y = 18 → x ≤ y :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_x_absolute_value_equation_l1308_130800


namespace NUMINAMATH_CALUDE_adam_purchase_cost_l1308_130865

-- Define the quantities of each item (in kg)
def almond_qty : Real := 1.5
def walnut_qty : Real := 1
def cashew_qty : Real := 0.5
def raisin_qty : Real := 1
def apricot_qty : Real := 1.5

-- Define the prices of each item (in $/kg)
def almond_price : Real := 12
def walnut_price : Real := 10
def cashew_price : Real := 20
def raisin_price : Real := 8
def apricot_price : Real := 6

-- Define the total cost function
def total_cost : Real :=
  almond_qty * almond_price +
  walnut_qty * walnut_price +
  cashew_qty * cashew_price +
  raisin_qty * raisin_price +
  apricot_qty * apricot_price

-- Theorem statement
theorem adam_purchase_cost : total_cost = 55 := by
  sorry

end NUMINAMATH_CALUDE_adam_purchase_cost_l1308_130865


namespace NUMINAMATH_CALUDE_triangle_inradius_l1308_130893

/-- Given a triangle with perimeter 28 cm and area 28 cm², prove that its inradius is 2 cm -/
theorem triangle_inradius (p A r : ℝ) (h1 : p = 28) (h2 : A = 28) (h3 : A = r * p / 2) : r = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inradius_l1308_130893


namespace NUMINAMATH_CALUDE_shaded_area_is_half_l1308_130882

/-- Represents a rectangle with a given area -/
structure Rectangle where
  area : ℝ

/-- Represents the shaded region after transformation -/
structure ShadedRegion where
  rectangle : Rectangle
  -- The rectangle is cut in two by a vertical cut joining the midpoints of its longer edges
  is_cut_in_half : Bool
  -- The right-hand half is given a quarter turn (90 degrees) about its center
  is_quarter_turned : Bool

/-- The area of the shaded region is half the area of the original rectangle -/
theorem shaded_area_is_half (r : Rectangle) (s : ShadedRegion) 
  (h1 : s.rectangle = r)
  (h2 : s.is_cut_in_half = true)
  (h3 : s.is_quarter_turned = true) :
  (s.rectangle.area / 2 : ℝ) = r.area / 2 :=
by sorry

#check shaded_area_is_half

end NUMINAMATH_CALUDE_shaded_area_is_half_l1308_130882


namespace NUMINAMATH_CALUDE_problem_statement_l1308_130834

/-- Given real numbers a and b satisfying a + 2b = 9, prove:
    1. If |9 - 2b| + |a + 1| < 3, then -2 < a < 1.
    2. If a > 0, b > 0, and z = ab^2, then the maximum value of z is 27. -/
theorem problem_statement (a b : ℝ) (h1 : a + 2*b = 9) :
  (|9 - 2*b| + |a + 1| < 3 → -2 < a ∧ a < 1) ∧
  (a > 0 → b > 0 → ∃ z : ℝ, z = a*b^2 ∧ ∀ w : ℝ, w = a*b^2 → w ≤ 27) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l1308_130834


namespace NUMINAMATH_CALUDE_store_price_reduction_l1308_130845

theorem store_price_reduction (original_price : ℝ) (h_positive : original_price > 0) :
  let first_reduction := 0.12
  let final_percentage := 0.792
  let price_after_first := original_price * (1 - first_reduction)
  let second_reduction := 1 - (final_percentage / (1 - first_reduction))
  second_reduction = 0.1 := by
sorry

end NUMINAMATH_CALUDE_store_price_reduction_l1308_130845


namespace NUMINAMATH_CALUDE_jessica_age_l1308_130856

theorem jessica_age :
  ∀ (j g : ℚ),
  g = 15 * j →
  g - j = 60 →
  j = 30 / 7 :=
by
  sorry

end NUMINAMATH_CALUDE_jessica_age_l1308_130856


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_1953_l1308_130824

theorem smallest_prime_factor_of_1953 : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 1953 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 1953 → p ≤ q :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_1953_l1308_130824


namespace NUMINAMATH_CALUDE_second_caterer_cheaper_l1308_130846

/-- Represents the cost function for a caterer -/
structure CatererCost where
  basicFee : ℕ
  perPersonFee : ℕ

/-- Calculates the total cost for a caterer given the number of people -/
def totalCost (c : CatererCost) (people : ℕ) : ℕ :=
  c.basicFee + c.perPersonFee * people

/-- The first caterer's pricing model -/
def caterer1 : CatererCost := { basicFee := 150, perPersonFee := 18 }

/-- The second caterer's pricing model -/
def caterer2 : CatererCost := { basicFee := 250, perPersonFee := 14 }

/-- Theorem stating that for 26 or more people, the second caterer is less expensive -/
theorem second_caterer_cheaper (n : ℕ) (h : n ≥ 26) :
  totalCost caterer2 n < totalCost caterer1 n := by
  sorry


end NUMINAMATH_CALUDE_second_caterer_cheaper_l1308_130846


namespace NUMINAMATH_CALUDE_expression_evaluation_l1308_130820

theorem expression_evaluation (c k : ℕ) (h1 : c = 4) (h2 : k = 2) :
  ((c^c - c*(c-1)^c + k)^c : ℕ) = 18974736 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1308_130820


namespace NUMINAMATH_CALUDE_tree_height_problem_l1308_130848

theorem tree_height_problem (h₁ h₂ : ℝ) : 
  h₁ = h₂ + 20 →  -- One tree is 20 feet taller than the other
  h₂ / h₁ = 5 / 7 →  -- The heights are in the ratio 5:7
  h₁ = 70 :=  -- The height of the taller tree is 70 feet
by sorry

end NUMINAMATH_CALUDE_tree_height_problem_l1308_130848


namespace NUMINAMATH_CALUDE_triangle_side_length_l1308_130807

def isOnParabola (p : ℝ × ℝ) : Prop := p.2 = -(p.1^2)

def isIsoscelesRightTriangle (p q : ℝ × ℝ) : Prop :=
  p.1^2 + p.2^2 = q.1^2 + q.2^2 ∧ p.1 * q.1 + p.2 * q.2 = 0

theorem triangle_side_length
  (p q : ℝ × ℝ)
  (h1 : isOnParabola p)
  (h2 : isOnParabola q)
  (h3 : isIsoscelesRightTriangle p q)
  : Real.sqrt (p.1^2 + p.2^2) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1308_130807


namespace NUMINAMATH_CALUDE_black_cows_exceeding_half_l1308_130854

theorem black_cows_exceeding_half (total_cows : ℕ) (non_black_cows : ℕ) : 
  total_cows = 18 → non_black_cows = 4 → 
  (total_cows - non_black_cows) - (total_cows / 2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_black_cows_exceeding_half_l1308_130854


namespace NUMINAMATH_CALUDE_number_problem_l1308_130849

theorem number_problem : ∃ x : ℝ, x = 580 ∧ 0.2 * x = 0.3 * 120 + 80 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1308_130849


namespace NUMINAMATH_CALUDE_original_equals_scientific_l1308_130804

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be expressed in scientific notation -/
def original_number : ℕ := 274000000

/-- The scientific notation representation of the original number -/
def scientific_representation : ScientificNotation :=
  { coefficient := 2.74
    exponent := 8
    is_valid := by sorry }

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific :
  (original_number : ℝ) = scientific_representation.coefficient * (10 : ℝ) ^ scientific_representation.exponent :=
by sorry

end NUMINAMATH_CALUDE_original_equals_scientific_l1308_130804


namespace NUMINAMATH_CALUDE_range_of_a_l1308_130859

/-- Given real numbers a, b, c satisfying a system of equations, 
    prove that the range of values for a is [1, 9]. -/
theorem range_of_a (a b c : ℝ) 
  (eq1 : a^2 - b*c - 8*a + 7 = 0)
  (eq2 : b^2 + c^2 + b*c - 6*a + 6 = 0) :
  a ∈ Set.Icc 1 9 := by
  sorry


end NUMINAMATH_CALUDE_range_of_a_l1308_130859


namespace NUMINAMATH_CALUDE_work_completion_theorem_l1308_130831

/-- Given that 36 men can complete a piece of work in 18 days,
    and a smaller group can complete the same work in 72 days,
    prove that the smaller group consists of 9 men. -/
theorem work_completion_theorem :
  ∀ (total_work : ℕ) (smaller_group : ℕ),
  total_work = 36 * 18 →
  total_work = smaller_group * 72 →
  smaller_group = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_work_completion_theorem_l1308_130831


namespace NUMINAMATH_CALUDE_white_balls_count_l1308_130855

/-- Given a bag of balls with the following properties:
  * The total number of balls is 40
  * The probability of drawing a red ball is 0.15
  * The probability of drawing a black ball is 0.45
  * The remaining balls are white
  
  This theorem proves that the number of white balls in the bag is 16. -/
theorem white_balls_count (total : ℕ) (p_red p_black : ℝ) :
  total = 40 →
  p_red = 0.15 →
  p_black = 0.45 →
  (total : ℝ) * (1 - p_red - p_black) = 16 := by
  sorry

end NUMINAMATH_CALUDE_white_balls_count_l1308_130855


namespace NUMINAMATH_CALUDE_bus_capacity_l1308_130899

theorem bus_capacity (rows : ℕ) (sections_per_row : ℕ) (students_per_section : ℕ) :
  rows = 13 →
  sections_per_row = 2 →
  students_per_section = 2 →
  rows * sections_per_row * students_per_section = 52 := by
  sorry

end NUMINAMATH_CALUDE_bus_capacity_l1308_130899


namespace NUMINAMATH_CALUDE_total_weight_is_20_2_l1308_130822

-- Define the capacities of the jugs
def jug1_capacity : ℝ := 2
def jug2_capacity : ℝ := 3
def jug3_capacity : ℝ := 4

-- Define the fill percentages
def jug1_fill_percent : ℝ := 0.7
def jug2_fill_percent : ℝ := 0.6
def jug3_fill_percent : ℝ := 0.5

-- Define the sand densities
def jug1_density : ℝ := 5
def jug2_density : ℝ := 4
def jug3_density : ℝ := 3

-- Calculate the weight of sand in each jug
def jug1_weight : ℝ := jug1_capacity * jug1_fill_percent * jug1_density
def jug2_weight : ℝ := jug2_capacity * jug2_fill_percent * jug2_density
def jug3_weight : ℝ := jug3_capacity * jug3_fill_percent * jug3_density

-- Total weight of sand in all jugs
def total_weight : ℝ := jug1_weight + jug2_weight + jug3_weight

theorem total_weight_is_20_2 : total_weight = 20.2 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_is_20_2_l1308_130822


namespace NUMINAMATH_CALUDE_remainder_theorem_l1308_130875

def polynomial (x : ℝ) : ℝ := 5*x^8 - 3*x^7 + 4*x^6 - 9*x^4 + 3*x^3 - 5*x^2 + 8

def divisor (x : ℝ) : ℝ := 3*x - 6

theorem remainder_theorem :
  ∃ (q : ℝ → ℝ), ∀ (x : ℝ),
    polynomial x = (divisor x) * (q x) + polynomial (2 : ℝ) ∧
    polynomial (2 : ℝ) = 1020 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1308_130875


namespace NUMINAMATH_CALUDE_mechanic_work_hours_l1308_130837

/-- Proves that a mechanic works 8 hours a day given the specified conditions -/
theorem mechanic_work_hours 
  (hourly_rate : ℕ) 
  (days_worked : ℕ) 
  (parts_cost : ℕ) 
  (total_paid : ℕ) 
  (h : hourly_rate = 60)
  (d : days_worked = 14)
  (p : parts_cost = 2500)
  (t : total_paid = 9220) :
  ∃ (hours_per_day : ℕ), 
    hours_per_day = 8 ∧ 
    hourly_rate * hours_per_day * days_worked + parts_cost = total_paid :=
by sorry

end NUMINAMATH_CALUDE_mechanic_work_hours_l1308_130837


namespace NUMINAMATH_CALUDE_a_greater_than_c_greater_than_b_l1308_130813

theorem a_greater_than_c_greater_than_b :
  let a := 0.6 * Real.exp 0.4
  let b := 2 - Real.log 4
  let c := Real.exp 1 - 2
  a > c ∧ c > b :=
by sorry

end NUMINAMATH_CALUDE_a_greater_than_c_greater_than_b_l1308_130813


namespace NUMINAMATH_CALUDE_perpendicular_lines_k_value_l1308_130890

/-- Theorem: Given two lines with direction vectors perpendicular to each other, 
    we can determine the value of k in the second line equation. -/
theorem perpendicular_lines_k_value (k : ℝ) 
  (line1 : ℝ × ℝ → Prop) 
  (line2 : ℝ × ℝ → Prop)
  (dir1 : ℝ × ℝ) 
  (dir2 : ℝ × ℝ) :
  (∀ x y, line1 (x, y) ↔ x + 3*y - 7 = 0) →
  (∀ x y, line2 (x, y) ↔ k*x - y - 2 = 0) →
  (dir1 = (1, -3)) →  -- Direction vector of line1
  (dir2 = (k, 1))  →  -- Direction vector of line2
  (dir1.1 * dir2.1 + dir1.2 * dir2.2 = 0) →  -- Dot product = 0
  k = 3 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_k_value_l1308_130890


namespace NUMINAMATH_CALUDE_imaginary_part_of_product_l1308_130851

theorem imaginary_part_of_product : Complex.im ((1 - Complex.I) * (2 + 4 * Complex.I)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_product_l1308_130851


namespace NUMINAMATH_CALUDE_zero_lt_m_lt_one_necessary_not_sufficient_l1308_130898

-- Define the quadratic equation
def quadratic_eq (m : ℝ) (x : ℝ) : Prop := x^2 + x + m^2 - 1 = 0

-- Define the condition for two real roots with different signs
def has_two_real_roots_diff_signs (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ * x₂ < 0 ∧ quadratic_eq m x₁ ∧ quadratic_eq m x₂

-- Theorem stating that 0 < m < 1 is a necessary but not sufficient condition
theorem zero_lt_m_lt_one_necessary_not_sufficient :
  (∀ m : ℝ, has_two_real_roots_diff_signs m → 0 < m ∧ m < 1) ∧
  (∃ m : ℝ, 0 < m ∧ m < 1 ∧ ¬has_two_real_roots_diff_signs m) :=
sorry

end NUMINAMATH_CALUDE_zero_lt_m_lt_one_necessary_not_sufficient_l1308_130898


namespace NUMINAMATH_CALUDE_cubic_inequality_l1308_130881

theorem cubic_inequality (p q x : ℝ) : x^3 + p*x + q = 0 → 4*q*x ≤ p^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l1308_130881


namespace NUMINAMATH_CALUDE_bills_age_l1308_130850

/-- Bill's current age -/
def b : ℕ := 24

/-- Tracy's current age -/
def t : ℕ := 18

/-- Bill's age is one third larger than Tracy's age -/
axiom bill_tracy_relation : b = (4 * t) / 3

/-- In 30 years, Bill's age will be one eighth larger than Tracy's age -/
axiom future_relation : b + 30 = (9 * (t + 30)) / 8

/-- Theorem: Given the age relations between Bill and Tracy, Bill's current age is 24 -/
theorem bills_age : b = 24 := by sorry

end NUMINAMATH_CALUDE_bills_age_l1308_130850


namespace NUMINAMATH_CALUDE_mikes_net_spent_l1308_130825

/-- The net amount Mike spent at the music store -/
def net_amount (trumpet_cost song_book_price : ℚ) : ℚ :=
  trumpet_cost - song_book_price

/-- Theorem stating the net amount Mike spent at the music store -/
theorem mikes_net_spent :
  let trumpet_cost : ℚ := 145.16
  let song_book_price : ℚ := 5.84
  net_amount trumpet_cost song_book_price = 139.32 := by
  sorry

end NUMINAMATH_CALUDE_mikes_net_spent_l1308_130825


namespace NUMINAMATH_CALUDE_simplify_G_l1308_130864

noncomputable def F (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

noncomputable def G (x : ℝ) : ℝ := F ((2 * x + x^2) / (1 + 2 * x))

theorem simplify_G (x : ℝ) (h : x ≠ -1/2 ∧ x ≠ 1) : 
  G x = 2 * Real.log (1 + 2 * x) - F x :=
by sorry

end NUMINAMATH_CALUDE_simplify_G_l1308_130864


namespace NUMINAMATH_CALUDE_bing_dwen_dwen_practice_time_l1308_130814

/-- Calculates the practice time given start time, end time, and break duration -/
def practice_time (start_time end_time break_duration : ℕ) : ℕ :=
  end_time - start_time - break_duration

/-- Proves that the practice time is 6 hours given the specified conditions -/
theorem bing_dwen_dwen_practice_time :
  let start_time := 8  -- 8 AM
  let end_time := 16   -- 4 PM (16 in 24-hour format)
  let break_duration := 2
  practice_time start_time end_time break_duration = 6 := by
sorry

#eval practice_time 8 16 2  -- Should output 6

end NUMINAMATH_CALUDE_bing_dwen_dwen_practice_time_l1308_130814


namespace NUMINAMATH_CALUDE_angle_sum_identity_l1308_130806

theorem angle_sum_identity (α β γ : Real) (h : α + β + γ = Real.pi) :
  Real.cos α ^ 2 + Real.cos β ^ 2 + Real.cos γ ^ 2 + 2 * Real.cos α * Real.cos β * Real.cos γ = 1 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_identity_l1308_130806


namespace NUMINAMATH_CALUDE_complementary_event_equivalence_l1308_130872

/-- The number of products in the sample -/
def sample_size : ℕ := 10

/-- Event A: at least 2 defective products -/
def event_A (defective : ℕ) : Prop := defective ≥ 2

/-- Complementary event of A -/
def comp_A (defective : ℕ) : Prop := ¬(event_A defective)

/-- At most 1 defective product -/
def at_most_one_defective (defective : ℕ) : Prop := defective ≤ 1

/-- At least 2 non-defective products -/
def at_least_two_non_defective (defective : ℕ) : Prop := sample_size - defective ≥ 2

theorem complementary_event_equivalence :
  ∀ defective : ℕ, defective ≤ sample_size →
    (comp_A defective ↔ at_most_one_defective defective) ∧
    (comp_A defective ↔ at_least_two_non_defective defective) :=
by sorry

end NUMINAMATH_CALUDE_complementary_event_equivalence_l1308_130872


namespace NUMINAMATH_CALUDE_binomial_coeff_not_arithmetic_sequence_l1308_130815

theorem binomial_coeff_not_arithmetic_sequence (n r : ℕ) (h : r + 3 ≤ n) :
  ¬∃ (d : ℚ), 
    (Nat.choose n (r + 1) : ℚ) - (Nat.choose n r : ℚ) = d ∧ 
    (Nat.choose n (r + 2) : ℚ) - (Nat.choose n (r + 1) : ℚ) = d ∧ 
    (Nat.choose n (r + 3) : ℚ) - (Nat.choose n (r + 2) : ℚ) = d :=
sorry

end NUMINAMATH_CALUDE_binomial_coeff_not_arithmetic_sequence_l1308_130815


namespace NUMINAMATH_CALUDE_mango_ratio_proof_l1308_130801

/-- Proves that the ratio of mangoes sold at the market to total mangoes harvested is 1:2 -/
theorem mango_ratio_proof (total_mangoes : ℕ) (num_neighbors : ℕ) (mangoes_per_neighbor : ℕ)
  (h1 : total_mangoes = 560)
  (h2 : num_neighbors = 8)
  (h3 : mangoes_per_neighbor = 35) :
  (total_mangoes - num_neighbors * mangoes_per_neighbor) / total_mangoes = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_mango_ratio_proof_l1308_130801


namespace NUMINAMATH_CALUDE_same_color_probability_l1308_130868

/-- The probability of drawing two balls of the same color from a bag with replacement -/
theorem same_color_probability (total : ℕ) (blue : ℕ) (yellow : ℕ) 
  (h_total : total = blue + yellow)
  (h_blue : blue = 5)
  (h_yellow : yellow = 5) :
  (blue / total) * (blue / total) + (yellow / total) * (yellow / total) = 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_same_color_probability_l1308_130868


namespace NUMINAMATH_CALUDE_first_purchase_correct_max_profit_correct_l1308_130840

/-- Represents the types of dolls -/
inductive DollType
| A
| B

/-- Represents the purchase and selling prices of dolls -/
def price (t : DollType) : ℕ × ℕ :=
  match t with
  | DollType.A => (20, 25)
  | DollType.B => (15, 18)

/-- The total number of dolls purchased -/
def total_dolls : ℕ := 100

/-- The total cost of the first purchase -/
def total_cost : ℕ := 1650

/-- Calculates the number of each type of doll in the first purchase -/
def first_purchase : ℕ × ℕ := sorry

/-- Calculates the profit for a given number of A dolls in the second purchase -/
def profit (x : ℕ) : ℕ := sorry

/-- Finds the maximum profit and corresponding number of dolls for the second purchase -/
def max_profit : ℕ × ℕ × ℕ := sorry

theorem first_purchase_correct :
  first_purchase = (30, 70) := by sorry

theorem max_profit_correct :
  max_profit = (366, 33, 67) := by sorry

end NUMINAMATH_CALUDE_first_purchase_correct_max_profit_correct_l1308_130840


namespace NUMINAMATH_CALUDE_sets_A_B_properties_l1308_130891

theorem sets_A_B_properties (p q : ℝ) (h : p * q ≠ 0) :
  (∀ x₀ : ℝ, 9^x₀ + p * 3^x₀ + q = 0 → q * 9^(-x₀) + p * 3^(-x₀) + 1 = 0) ∧
  (∃ p q : ℝ, 
    (∃ x : ℝ, 9^x + p * 3^x + q = 0 ∧ q * 9^x + p * 3^x + 1 = 0) ∧
    (∀ x : ℝ, x ≠ 1 → 9^x + p * 3^x + q = 0 → q * 9^x + p * 3^x + 1 ≠ 0) ∧
    (9^1 + p * 3^1 + q = 0 ∧ q * 9^1 + p * 3^1 + 1 = 0) ∧
    p = -4 ∧ q = 3) :=
by sorry

end NUMINAMATH_CALUDE_sets_A_B_properties_l1308_130891


namespace NUMINAMATH_CALUDE_specific_rhombus_area_l1308_130802

/-- Represents a rhombus with given properties -/
structure Rhombus where
  side_length : ℝ
  diagonal_difference : ℝ
  diagonals_perpendicular : Bool

/-- Calculates the area of a rhombus given its properties -/
def rhombus_area (r : Rhombus) : ℝ :=
  sorry

/-- Theorem stating the area of a specific rhombus -/
theorem specific_rhombus_area :
  let r : Rhombus := { 
    side_length := Real.sqrt 145,
    diagonal_difference := 8,
    diagonals_perpendicular := true
  }
  rhombus_area r = (Real.sqrt 274 * (Real.sqrt 274 - 4)) / 4 := by
  sorry

end NUMINAMATH_CALUDE_specific_rhombus_area_l1308_130802


namespace NUMINAMATH_CALUDE_total_travel_time_l1308_130878

def station_distance : ℕ := 2 -- hours
def break_time : ℕ := 30 -- minutes

theorem total_travel_time :
  let travel_time_between_stations := station_distance * 60 -- convert hours to minutes
  let total_travel_time := 2 * travel_time_between_stations + break_time
  total_travel_time = 270 := by
sorry

end NUMINAMATH_CALUDE_total_travel_time_l1308_130878


namespace NUMINAMATH_CALUDE_present_value_exponent_l1308_130886

theorem present_value_exponent 
  (Q r j m n : ℝ) 
  (hQ : Q > 0) 
  (hr : r > 0) 
  (hjm : j + m > -1) 
  (heq : Q = r / (1 + j + m) ^ n) : 
  n = Real.log (r / Q) / Real.log (1 + j + m) := by
sorry

end NUMINAMATH_CALUDE_present_value_exponent_l1308_130886


namespace NUMINAMATH_CALUDE_correct_height_order_l1308_130863

-- Define the friends
inductive Friend : Type
| David : Friend
| Emma : Friend
| Fiona : Friend

-- Define the height comparison relation
def taller_than : Friend → Friend → Prop := sorry

-- Define the conditions
axiom different_heights :
  ∀ (a b : Friend), a ≠ b → (taller_than a b ∨ taller_than b a)

axiom transitive :
  ∀ (a b c : Friend), taller_than a b → taller_than b c → taller_than a c

axiom asymmetric :
  ∀ (a b : Friend), taller_than a b → ¬taller_than b a

axiom exactly_one_true :
  (¬(taller_than Friend.Fiona Friend.Emma) ∧
   ¬(taller_than Friend.David Friend.Fiona) ∧
   taller_than Friend.David Friend.Emma) ∨
  (taller_than Friend.Fiona Friend.David ∧
   taller_than Friend.Fiona Friend.Emma) ∨
  (¬(taller_than Friend.David Friend.Emma) ∧
   ¬(taller_than Friend.David Friend.Fiona))

-- Theorem to prove
theorem correct_height_order :
  taller_than Friend.David Friend.Emma ∧
  taller_than Friend.Emma Friend.Fiona ∧
  taller_than Friend.David Friend.Fiona :=
sorry

end NUMINAMATH_CALUDE_correct_height_order_l1308_130863
