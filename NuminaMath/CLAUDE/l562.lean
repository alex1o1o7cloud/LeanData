import Mathlib

namespace probability_all_selected_l562_56226

/-- The probability of Ram being selected -/
def p_ram : ℚ := 6/7

/-- The initial probability of Ravi being selected -/
def p_ravi_initial : ℚ := 1/5

/-- The probability of Ravi being selected given Ram is selected -/
def p_ravi_given_ram : ℚ := 2/5

/-- The initial probability of Rajesh being selected -/
def p_rajesh_initial : ℚ := 2/3

/-- The probability of Rajesh being selected given Ravi is selected -/
def p_rajesh_given_ravi : ℚ := 1/2

/-- The theorem stating the probability of all three brothers being selected -/
theorem probability_all_selected : 
  p_ram * p_ravi_given_ram * p_rajesh_given_ravi = 6/35 := by
  sorry

end probability_all_selected_l562_56226


namespace charity_book_donation_l562_56227

theorem charity_book_donation (initial_books : ℕ) (books_per_donation : ℕ) 
  (borrowed_books : ℕ) (final_books : ℕ) : 
  initial_books = 300 →
  books_per_donation = 5 →
  borrowed_books = 140 →
  final_books = 210 →
  (final_books + borrowed_books - initial_books) / books_per_donation = 10 :=
by
  sorry

end charity_book_donation_l562_56227


namespace unique_triangle_arrangement_l562_56296

-- Define the structure of the triangle
structure Triangle :=
  (A B C D : ℕ)
  (side1 side2 side3 : ℕ)

-- Define the conditions of the problem
def validTriangle (t : Triangle) : Prop :=
  t.A ∈ ({6, 7, 8, 9} : Set ℕ) ∧
  t.B ∈ ({6, 7, 8, 9} : Set ℕ) ∧
  t.C ∈ ({6, 7, 8, 9} : Set ℕ) ∧
  t.D ∈ ({6, 7, 8, 9} : Set ℕ) ∧
  t.A ≠ t.B ∧ t.A ≠ t.C ∧ t.A ≠ t.D ∧
  t.B ≠ t.C ∧ t.B ≠ t.D ∧
  t.C ≠ t.D ∧
  t.side1 = 1 + t.B + 5 ∧
  t.side2 = 3 + 4 + t.D ∧
  t.side3 = 2 + t.A + 4 ∧
  t.side1 = t.side2 ∧ t.side2 = t.side3

-- Theorem statement
theorem unique_triangle_arrangement :
  ∃! t : Triangle, validTriangle t ∧ t.A = 6 ∧ t.B = 8 ∧ t.C = 7 ∧ t.D = 9 :=
sorry

end unique_triangle_arrangement_l562_56296


namespace jim_toads_difference_l562_56282

theorem jim_toads_difference (tim_toads sarah_toads : ℕ) 
  (h1 : tim_toads = 30)
  (h2 : sarah_toads = 100)
  (h3 : sarah_toads = 2 * jim_toads)
  (h4 : jim_toads > tim_toads) : 
  jim_toads - tim_toads = 20 := by
sorry

end jim_toads_difference_l562_56282


namespace mirror_side_length_l562_56251

/-- Given a rectangular wall and a square mirror, proves that the mirror's side length is 34 inches -/
theorem mirror_side_length (wall_width wall_length mirror_area : ℝ) : 
  wall_width = 54 →
  wall_length = 42.81481481481482 →
  mirror_area = (wall_width * wall_length) / 2 →
  Real.sqrt mirror_area = 34 := by
  sorry

end mirror_side_length_l562_56251


namespace fraction_greater_than_one_necessary_not_sufficient_l562_56278

theorem fraction_greater_than_one_necessary_not_sufficient :
  (∀ a b : ℝ, a > b ∧ b > 0 → a / b > 1) ∧
  (∃ a b : ℝ, a / b > 1 ∧ ¬(a > b ∧ b > 0)) := by
  sorry

end fraction_greater_than_one_necessary_not_sufficient_l562_56278


namespace pond_to_field_area_ratio_l562_56240

theorem pond_to_field_area_ratio :
  ∀ (field_length field_width pond_side : ℝ),
    field_length = 2 * field_width →
    field_length = 112 →
    pond_side = 8 →
    (pond_side^2) / (field_length * field_width) = 1 / 98 := by
  sorry

end pond_to_field_area_ratio_l562_56240


namespace subgroup_samples_is_ten_l562_56239

/-- Represents a stratified sampling scenario -/
structure StratifiedSample where
  total_population : ℕ
  subgroup_size : ℕ
  total_samples : ℕ
  subgroup_samples : ℕ

/-- Calculates the number of samples from a subgroup in stratified sampling -/
def calculate_subgroup_samples (s : StratifiedSample) : ℚ :=
  s.total_samples * (s.subgroup_size : ℚ) / s.total_population

/-- Theorem stating that for the given scenario, the number of subgroup samples is 10 -/
theorem subgroup_samples_is_ten : 
  let s : StratifiedSample := {
    total_population := 1200,
    subgroup_size := 200,
    total_samples := 60,
    subgroup_samples := 10
  }
  calculate_subgroup_samples s = 10 := by
  sorry


end subgroup_samples_is_ten_l562_56239


namespace ratio_problem_l562_56249

theorem ratio_problem (x y : ℝ) (h : (3*x - 2*y) / (2*x + y) = 5/4) : y / x = 2/13 := by
  sorry

end ratio_problem_l562_56249


namespace complex_fraction_power_l562_56206

theorem complex_fraction_power (i : ℂ) : i * i = -1 → (((1 + i) / (1 - i)) ^ 2014 : ℂ) = -1 := by
  sorry

end complex_fraction_power_l562_56206


namespace constant_sequence_l562_56260

theorem constant_sequence (a : ℕ → ℝ) : 
  (∀ (b : ℕ → ℕ), (∀ n : ℕ, b n ≠ b (n + 1) ∧ (b n ∣ b (n + 1))) → 
    ∃ (d : ℝ), ∀ n : ℕ, a (b (n + 1)) - a (b n) = d) →
  ∃ (c : ℝ), ∀ n : ℕ, a n = c :=
by sorry

end constant_sequence_l562_56260


namespace total_spider_legs_l562_56217

/-- The number of spiders in Ivy's room -/
def num_spiders : ℕ := 4

/-- The number of legs each spider has -/
def legs_per_spider : ℕ := 8

/-- Theorem: The total number of spider legs in Ivy's room is 32 -/
theorem total_spider_legs : num_spiders * legs_per_spider = 32 := by
  sorry

end total_spider_legs_l562_56217


namespace original_number_l562_56287

theorem original_number (N : ℤ) : 
  (∃ k : ℤ, N + 4 = 25 * k) ∧ 
  (∀ m : ℤ, m < 4 → ¬(∃ j : ℤ, N + m = 25 * j)) →
  N = 21 := by
sorry

end original_number_l562_56287


namespace work_completion_time_l562_56243

/-- The time taken for A to complete the work alone -/
def time_A : ℝ := 10

/-- The time taken for A and B to complete the work together -/
def time_AB : ℝ := 4.444444444444445

/-- The time taken for B to complete the work alone -/
def time_B : ℝ := 8

/-- Theorem stating that given the time for A alone and A and B together, 
    the time for B alone is 8 days -/
theorem work_completion_time : 
  (1 / time_A + 1 / time_B = 1 / time_AB) → time_B = 8 := by
  sorry

end work_completion_time_l562_56243


namespace sum_smallest_largest_prime_1_to_50_l562_56236

theorem sum_smallest_largest_prime_1_to_50 : 
  ∃ (p q : Nat), 
    Prime p ∧ Prime q ∧ 
    p ≤ 50 ∧ q ≤ 50 ∧
    (∀ r, Prime r ∧ r ≤ 50 → p ≤ r) ∧
    (∀ r, Prime r ∧ r ≤ 50 → r ≤ q) ∧
    p + q = 49 := by
  sorry

end sum_smallest_largest_prime_1_to_50_l562_56236


namespace height_relation_l562_56228

/-- Two right circular cylinders with equal volume and related radii -/
structure CylinderPair where
  r₁ : ℝ  -- radius of the first cylinder
  h₁ : ℝ  -- height of the first cylinder
  r₂ : ℝ  -- radius of the second cylinder
  h₂ : ℝ  -- height of the second cylinder
  r₁_pos : 0 < r₁  -- r₁ is positive
  h₁_pos : 0 < h₁  -- h₁ is positive
  r₂_pos : 0 < r₂  -- r₂ is positive
  h₂_pos : 0 < h₂  -- h₂ is positive
  volume_eq : π * r₁^2 * h₁ = π * r₂^2 * h₂  -- volumes are equal
  radius_relation : r₂ = 1.2 * r₁  -- r₂ is 20% more than r₁

/-- The theorem stating the relationship between the heights of the cylinders -/
theorem height_relation (cp : CylinderPair) : cp.h₁ = 1.44 * cp.h₂ := by
  sorry

#check height_relation

end height_relation_l562_56228


namespace count_odd_numbers_300_to_600_l562_56269

theorem count_odd_numbers_300_to_600 : 
  (Finset.filter (fun n => n % 2 = 1) (Finset.Icc 300 600)).card = 150 := by
  sorry

end count_odd_numbers_300_to_600_l562_56269


namespace g_solutions_l562_56252

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- Define the property that g must satisfy
def g_property (g : ℝ → ℝ) : Prop :=
  ∀ x, f (g x) = 9 * x^2 - 6 * x + 1

-- State the theorem
theorem g_solutions :
  ∀ g : ℝ → ℝ, g_property g →
    (∀ x, g x = 3 * x - 1) ∨ (∀ x, g x = -3 * x + 1) :=
by sorry

end g_solutions_l562_56252


namespace square_of_complex_l562_56220

theorem square_of_complex (z : ℂ) : z = 2 - 3*I → z^2 = -5 - 12*I := by
  sorry

end square_of_complex_l562_56220


namespace woman_lawyer_probability_l562_56237

/-- Represents a study group with given proportions of women and women lawyers -/
structure StudyGroup where
  totalMembers : ℕ
  womenPercentage : ℚ
  womenLawyerPercentage : ℚ

/-- Calculates the probability of selecting a woman lawyer at random from the study group -/
def probWomanLawyer (group : StudyGroup) : ℚ :=
  group.womenPercentage * group.womenLawyerPercentage

theorem woman_lawyer_probability (group : StudyGroup) 
  (h1 : group.womenPercentage = 9/10)
  (h2 : group.womenLawyerPercentage = 6/10) :
  probWomanLawyer group = 54/100 := by
  sorry

#eval probWomanLawyer { totalMembers := 100, womenPercentage := 9/10, womenLawyerPercentage := 6/10 }

end woman_lawyer_probability_l562_56237


namespace inscribed_squares_side_length_l562_56268

/-- Right triangle ABC with two inscribed squares -/
structure RightTriangleWithSquares where
  /-- Length of side AB -/
  ab : ℝ
  /-- Length of side BC -/
  bc : ℝ
  /-- Length of side AC (hypotenuse) -/
  ac : ℝ
  /-- Side length of the inscribed squares -/
  s : ℝ
  /-- AB = 6 -/
  ab_eq : ab = 6
  /-- BC = 8 -/
  bc_eq : bc = 8
  /-- AC = 10 -/
  ac_eq : ac = 10
  /-- Pythagorean theorem holds -/
  pythagorean : ab ^ 2 + bc ^ 2 = ac ^ 2
  /-- The two squares do not overlap -/
  non_overlapping : 2 * s ≤ (ab * bc) / ac

/-- The side length of each inscribed square is 2.4 -/
theorem inscribed_squares_side_length (t : RightTriangleWithSquares) : t.s = 2.4 := by
  sorry

end inscribed_squares_side_length_l562_56268


namespace weekly_earnings_theorem_l562_56232

/-- Represents the shop's T-shirt sales and operating conditions -/
structure ShopConditions where
  women_tshirt_interval : ℕ := 30 -- minutes between women's T-shirt sales
  women_tshirt_price : ℕ := 18 -- price of women's T-shirt
  men_tshirt_interval : ℕ := 40 -- minutes between men's T-shirt sales
  men_tshirt_price : ℕ := 15 -- price of men's T-shirt
  daily_operating_minutes : ℕ := 720 -- minutes of operation per day (12 hours)
  days_per_week : ℕ := 7 -- number of operating days per week

/-- Calculates the weekly earnings from T-shirt sales given the shop conditions -/
def calculate_weekly_earnings (conditions : ShopConditions) : ℕ :=
  let women_daily_sales := conditions.daily_operating_minutes / conditions.women_tshirt_interval
  let men_daily_sales := conditions.daily_operating_minutes / conditions.men_tshirt_interval
  let daily_earnings := women_daily_sales * conditions.women_tshirt_price +
                        men_daily_sales * conditions.men_tshirt_price
  daily_earnings * conditions.days_per_week

/-- Theorem stating that the weekly earnings from T-shirt sales is $4914 -/
theorem weekly_earnings_theorem (shop : ShopConditions) :
  calculate_weekly_earnings shop = 4914 := by
  sorry


end weekly_earnings_theorem_l562_56232


namespace least_integer_with_divisibility_condition_l562_56215

def is_divisible (a b : ℕ) : Prop := b ≠ 0 ∧ a % b = 0

def consecutive_pair (a b : ℕ) : Prop := b = a + 1

theorem least_integer_with_divisibility_condition :
  ∃ (n : ℕ) (a : ℕ),
    n = 2329089562800 ∧
    a ≥ 1 ∧ a < 30 ∧
    (∀ k : ℕ, 1 ≤ k ∧ k ≤ 30 → (k = a ∨ k = a + 1 ∨ is_divisible n k)) ∧
    consecutive_pair a (a + 1) ∧
    ¬(is_divisible n a) ∧
    ¬(is_divisible n (a + 1)) ∧
    (∀ m : ℕ, m < n →
      ¬(∃ b : ℕ, b ≥ 1 ∧ b < 30 ∧
        (∀ k : ℕ, 1 ≤ k ∧ k ≤ 30 → (k = b ∨ k = b + 1 ∨ is_divisible m k)) ∧
        consecutive_pair b (b + 1) ∧
        ¬(is_divisible m b) ∧
        ¬(is_divisible m (b + 1)))) :=
sorry

end least_integer_with_divisibility_condition_l562_56215


namespace not_fourth_power_prime_minus_four_l562_56214

theorem not_fourth_power_prime_minus_four (p : ℕ) (h_prime : Nat.Prime p) (h_gt_five : p > 5) :
  ¬ ∃ (a : ℕ), p - 4 = a^4 := by
  sorry

end not_fourth_power_prime_minus_four_l562_56214


namespace polygon_with_40_degree_exterior_angles_has_9_sides_l562_56205

/-- A polygon with exterior angles of 40° has 9 sides -/
theorem polygon_with_40_degree_exterior_angles_has_9_sides :
  ∀ (n : ℕ), 
  (n > 2) →
  (360 / n = 40) →
  n = 9 := by
sorry

end polygon_with_40_degree_exterior_angles_has_9_sides_l562_56205


namespace sum_odd_plus_even_l562_56283

def sum_odd_integers (n : ℕ) : ℕ :=
  (n + 1) * n

def sum_even_integers (n : ℕ) : ℕ :=
  n * (n + 1)

def m : ℕ := sum_odd_integers 56

def t : ℕ := sum_even_integers 25

theorem sum_odd_plus_even : m + t = 3786 := by
  sorry

end sum_odd_plus_even_l562_56283


namespace stratified_sample_category_a_l562_56208

/-- Represents the number of students in each school category -/
structure SchoolCategories where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Calculates the number of students to be sampled from Category A schools
    using stratified sampling -/
def sampleSizeA (categories : SchoolCategories) (totalSample : ℕ) : ℕ :=
  (categories.a * totalSample) / (categories.a + categories.b + categories.c)

/-- Theorem stating that for the given school categories and sample size,
    the number of students to be selected from Category A is 200 -/
theorem stratified_sample_category_a 
  (categories : SchoolCategories)
  (h1 : categories.a = 2000)
  (h2 : categories.b = 3000)
  (h3 : categories.c = 4000)
  (totalSample : ℕ)
  (h4 : totalSample = 900) :
  sampleSizeA categories totalSample = 200 := by
  sorry


end stratified_sample_category_a_l562_56208


namespace department_store_sales_multiple_l562_56257

theorem department_store_sales_multiple (M : ℝ) :
  (∀ (A : ℝ), A > 0 →
    M * A = 0.15384615384615385 * (11 * A + M * A)) →
  M = 2 := by
sorry

end department_store_sales_multiple_l562_56257


namespace popped_kernel_probability_l562_56246

theorem popped_kernel_probability (white yellow blue : ℝ)
  (white_pop yellow_pop blue_pop : ℝ) :
  white = 1/2 →
  yellow = 1/4 →
  blue = 1/4 →
  white_pop = 1/3 →
  yellow_pop = 3/4 →
  blue_pop = 2/3 →
  (white * white_pop) / (white * white_pop + yellow * yellow_pop + blue * blue_pop) = 2/11 := by
  sorry

end popped_kernel_probability_l562_56246


namespace arithmetic_sequence_common_difference_l562_56203

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) (h : is_arithmetic_sequence a) 
  (h2015_2013 : a 2015 = a 2013 + 6) : 
  ∃ d : ℝ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ d = 3 := by
  sorry

end arithmetic_sequence_common_difference_l562_56203


namespace ball_count_theorem_l562_56244

theorem ball_count_theorem (n : ℕ) : 
  (18 : ℝ) / (18 + 9 + n) = (30 : ℝ) / 100 → n = 42 := by
  sorry

end ball_count_theorem_l562_56244


namespace remove_fifteen_for_average_seven_point_five_l562_56211

theorem remove_fifteen_for_average_seven_point_five :
  let sequence := List.range 15
  let sum := sequence.sum
  let removed := 15
  let remaining_sum := sum - removed
  let remaining_count := sequence.length - 1
  (remaining_sum : ℚ) / remaining_count = 15/2 := by
    sorry

end remove_fifteen_for_average_seven_point_five_l562_56211


namespace solution_pairs_l562_56248

theorem solution_pairs (x y a n m : ℕ) (h1 : x + y = a^n) (h2 : x^2 + y^2 = a^m) :
  ∃ k : ℕ, x = 2^k ∧ y = 2^k := by
  sorry

end solution_pairs_l562_56248


namespace twelve_ways_to_choose_l562_56242

/-- The number of ways to choose one female student from a group of 4
    and one male student from a group of 3 -/
def waysToChoose (female_count male_count : ℕ) : ℕ :=
  female_count * male_count

/-- Theorem stating that there are 12 ways to choose one female student
    from a group of 4 and one male student from a group of 3 -/
theorem twelve_ways_to_choose :
  waysToChoose 4 3 = 12 := by
  sorry

end twelve_ways_to_choose_l562_56242


namespace non_zero_digits_count_l562_56233

def expression : ℚ := 180 / (2^4 * 5^6 * 3^2)

def count_non_zero_decimal_digits (q : ℚ) : ℕ :=
  sorry

theorem non_zero_digits_count : count_non_zero_decimal_digits expression = 1 := by
  sorry

end non_zero_digits_count_l562_56233


namespace paint_bought_l562_56294

theorem paint_bought (total_needed paint_existing paint_still_needed : ℕ) 
  (h1 : total_needed = 70)
  (h2 : paint_existing = 36)
  (h3 : paint_still_needed = 11) :
  total_needed - paint_existing - paint_still_needed = 23 :=
by sorry

end paint_bought_l562_56294


namespace tangent_line_at_x_1_l562_56213

-- Define the function f(x) = x³ + x
def f (x : ℝ) : ℝ := x^3 + x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Theorem statement
theorem tangent_line_at_x_1 :
  ∃ (m c : ℝ), 
    (∀ x y : ℝ, y = m * x + c ↔ m * x - y + c = 0) ∧
    (m = f' 1) ∧
    (f 1 = m * 1 + c) ∧
    (m * x - y + c = 0 ↔ 4 * x - y - 2 = 0) :=
sorry

end tangent_line_at_x_1_l562_56213


namespace dragon_rope_problem_l562_56200

theorem dragon_rope_problem (a b c : ℕ) (h_prime : Nat.Prime c) :
  let tower_radius : ℝ := 10
  let rope_length : ℝ := 25
  let height_difference : ℝ := 3
  let rope_touching_tower : ℝ := (a - Real.sqrt b) / c
  (tower_radius > 0 ∧ rope_length > tower_radius ∧ height_difference > 0 ∧
   rope_touching_tower > 0 ∧ rope_touching_tower < rope_length) →
  a + b + c = 352 :=
by sorry

end dragon_rope_problem_l562_56200


namespace even_function_extension_l562_56253

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

-- State the theorem
theorem even_function_extension
  (f : ℝ → ℝ)
  (h_even : EvenFunction f)
  (h_nonneg : ∀ x : ℝ, x ≥ 0 → f x = 2^x + 1) :
  ∀ x : ℝ, x < 0 → f x = 2^(-x) + 1 :=
sorry

end even_function_extension_l562_56253


namespace sum_of_reciprocals_of_quadratic_roots_l562_56262

theorem sum_of_reciprocals_of_quadratic_roots :
  ∀ p q : ℝ, p^2 - 10*p + 3 = 0 → q^2 - 10*q + 3 = 0 → p ≠ q →
  1/p + 1/q = 10/3 := by
sorry

end sum_of_reciprocals_of_quadratic_roots_l562_56262


namespace rectangle_area_difference_l562_56250

theorem rectangle_area_difference (A B a b : ℕ) 
  (h1 : A = 20) (h2 : B = 30) (h3 : a = 4) (h4 : b = 7) : 
  (A * B - a * b) - ((A - a) * B + A * (B - b)) = 20 := by
  sorry

end rectangle_area_difference_l562_56250


namespace problem_figure_perimeter_l562_56279

/-- Represents a figure made of unit squares -/
structure UnitSquareFigure where
  bottom_row : Nat
  left_column : Nat
  top_row : Nat
  right_column : Nat

/-- The specific figure described in the problem -/
def problem_figure : UnitSquareFigure :=
  { bottom_row := 3
  , left_column := 2
  , top_row := 4
  , right_column := 3 }

/-- Calculates the perimeter of a UnitSquareFigure -/
def perimeter (figure : UnitSquareFigure) : Nat :=
  figure.bottom_row + figure.left_column + figure.top_row + figure.right_column

theorem problem_figure_perimeter : perimeter problem_figure = 12 := by
  sorry

#eval perimeter problem_figure

end problem_figure_perimeter_l562_56279


namespace multiply_three_six_and_quarter_l562_56284

theorem multiply_three_six_and_quarter : 3.6 * 0.25 = 0.9 := by
  sorry

end multiply_three_six_and_quarter_l562_56284


namespace optimal_arrangement_l562_56292

/-- Represents the capacity and cost of a truck type -/
structure TruckType where
  water_capacity : ℕ
  vegetable_capacity : ℕ
  cost : ℕ

/-- Represents the donation quantities and truck types -/
structure DonationProblem where
  total_donation : ℕ
  water_vegetable_diff : ℕ
  type_a : TruckType
  type_b : TruckType
  total_trucks : ℕ

def problem : DonationProblem :=
  { total_donation := 120
  , water_vegetable_diff := 12
  , type_a := { water_capacity := 5, vegetable_capacity := 8, cost := 400 }
  , type_b := { water_capacity := 6, vegetable_capacity := 6, cost := 360 }
  , total_trucks := 10
  }

def water_amount (p : DonationProblem) : ℕ :=
  (p.total_donation - p.water_vegetable_diff) / 2

def vegetable_amount (p : DonationProblem) : ℕ :=
  p.total_donation - water_amount p

def is_valid_arrangement (p : DonationProblem) (type_a_count : ℕ) : Prop :=
  let type_b_count := p.total_trucks - type_a_count
  type_a_count * p.type_a.water_capacity + type_b_count * p.type_b.water_capacity ≥ water_amount p ∧
  type_a_count * p.type_a.vegetable_capacity + type_b_count * p.type_b.vegetable_capacity ≥ vegetable_amount p

def transportation_cost (p : DonationProblem) (type_a_count : ℕ) : ℕ :=
  type_a_count * p.type_a.cost + (p.total_trucks - type_a_count) * p.type_b.cost

theorem optimal_arrangement (p : DonationProblem) :
  ∃ (type_a_count : ℕ),
    type_a_count = 3 ∧
    is_valid_arrangement p type_a_count ∧
    ∀ (other_count : ℕ),
      is_valid_arrangement p other_count →
      transportation_cost p type_a_count ≤ transportation_cost p other_count :=
sorry

#eval transportation_cost problem 3  -- Should evaluate to 3720

end optimal_arrangement_l562_56292


namespace x_squared_plus_y_squared_l562_56216

theorem x_squared_plus_y_squared (x y : ℝ) :
  |x - 1/2| + (2*y + 1)^2 = 0 → x^2 + y^2 = 1/2 := by
  sorry

end x_squared_plus_y_squared_l562_56216


namespace kelvin_winning_strategy_l562_56258

/-- Represents a player in the game -/
inductive Player
| Kelvin
| Alex

/-- Represents a single move in the game -/
structure Move where
  digit : Nat
  position : Nat

/-- Represents the state of the game -/
structure GameState where
  number : Nat
  currentPlayer : Player

/-- A strategy for Kelvin -/
def KelvinStrategy := GameState → Move

/-- Applies a move to the current game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  sorry

/-- Checks if a number is a perfect square -/
def isPerfectSquare (n : Nat) : Bool :=
  sorry

/-- Plays the game given Kelvin's strategy and Alex's moves -/
def playGame (strategy : KelvinStrategy) (alexMoves : List Move) : Bool :=
  sorry

/-- Theorem stating that Kelvin has a winning strategy -/
theorem kelvin_winning_strategy :
  ∃ (strategy : KelvinStrategy),
    ∀ (alexMoves : List Move),
      ¬(playGame strategy alexMoves) :=
sorry

end kelvin_winning_strategy_l562_56258


namespace polynomial_root_k_value_l562_56238

theorem polynomial_root_k_value :
  ∀ k : ℚ, (3 : ℚ)^4 + k * (3 : ℚ)^2 - 26 = 0 → k = -55/9 := by
  sorry

end polynomial_root_k_value_l562_56238


namespace min_sum_of_squares_l562_56259

theorem min_sum_of_squares (x y : ℝ) (h : (x + 5) * (y - 5) = 0) :
  ∃ (m : ℝ), (∀ a b : ℝ, (a + 5) * (b - 5) = 0 → x^2 + y^2 ≤ a^2 + b^2) ∧ m = 50 := by
  sorry

end min_sum_of_squares_l562_56259


namespace f_inequality_solution_f_max_negative_l562_56235

def f (x : ℝ) := |x - 1| + |x + 1|

theorem f_inequality_solution (x : ℝ) :
  f x ≤ 4 ↔ x ∈ Set.Icc (-2) 2 :=
sorry

theorem f_max_negative (b : ℝ) (hb : b ≠ 0) :
  (∀ x, f x ≥ (|2*b + 1| + |1 - b|) / |b|) →
  (∃ x, x < 0 ∧ f x ≥ (|2*b + 1| + |1 - b|) / |b| ∧
    ∀ y, y < 0 → f y ≥ (|2*b + 1| + |1 - b|) / |b| → y ≤ x) →
  x = -1.5 :=
sorry

end f_inequality_solution_f_max_negative_l562_56235


namespace quadratic_inequality_range_l562_56271

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, (x - a) * (x + 1 - a) ≥ 0 → x ≠ 1) → 
  a > 1 ∧ a < 2 := by
sorry

end quadratic_inequality_range_l562_56271


namespace college_students_count_l562_56274

theorem college_students_count (boys girls : ℕ) (h1 : boys * 5 = girls * 6) (h2 : girls = 200) :
  boys + girls = 440 := by
  sorry

end college_students_count_l562_56274


namespace vector_operation_result_l562_56201

/-- Proves that the given vector operation results in (4, -7) -/
theorem vector_operation_result : 
  4 • !![3, -9] - 3 • !![2, -7] + 2 • !![-1, 4] = !![4, -7] := by
  sorry

end vector_operation_result_l562_56201


namespace possible_values_of_a_l562_56285

theorem possible_values_of_a (a b c d : ℕ+) 
  (h1 : a > b ∧ b > c ∧ c > d)
  (h2 : a + b + c + d = 2020)
  (h3 : a^2 - b^2 + c^2 - d^2 = 2020) :
  ∃! s : Finset ℕ+, s.card = 501 ∧ ∀ x, x ∈ s ↔ ∃ b' c' d' : ℕ+, 
    x > b' ∧ b' > c' ∧ c' > d' ∧
    x + b' + c' + d' = 2020 ∧
    x^2 - b'^2 + c'^2 - d'^2 = 2020 :=
by sorry

end possible_values_of_a_l562_56285


namespace tangent_points_parallel_to_line_y_coordinates_correct_tangent_points_are_correct_l562_56261

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 + 1

-- Theorem statement
theorem tangent_points_parallel_to_line (x : ℝ) :
  (f' x = 4) ↔ (x = 1 ∨ x = -1) :=
sorry

-- Theorem to verify the y-coordinates
theorem y_coordinates_correct :
  f 1 = 0 ∧ f (-1) = -4 :=
sorry

-- Main theorem combining both conditions
theorem tangent_points_are_correct :
  ∀ x y : ℝ, (f' x = 4 ∧ f x = y) ↔ ((x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = -4)) :=
sorry

end tangent_points_parallel_to_line_y_coordinates_correct_tangent_points_are_correct_l562_56261


namespace intersection_and_conditions_l562_56297

-- Define the intersection point M
def M : ℝ × ℝ := (-1, 2)

-- Define the given lines
def line1 (x y : ℝ) : Prop := 3 * x + 4 * y - 5 = 0
def line2 (x y : ℝ) : Prop := 2 * x - 3 * y + 8 = 0
def line3 (x y : ℝ) : Prop := 2 * x + y + 5 = 0

-- Define the resulting lines
def result_line1 (x y : ℝ) : Prop := x = -1
def result_line2 (x y : ℝ) : Prop := x - 2 * y + 5 = 0

theorem intersection_and_conditions :
  -- M is the intersection point of line1 and line2
  (line1 M.1 M.2 ∧ line2 M.1 M.2) ∧
  -- result_line1 passes through M and (-1, 0)
  (result_line1 M.1 M.2 ∧ result_line1 (-1) 0) ∧
  -- result_line2 passes through M
  result_line2 M.1 M.2 ∧
  -- result_line2 is perpendicular to line3
  (∃ (k : ℝ), k ≠ 0 ∧ 1 * 2 + (-2) * 1 = -k * k) :=
by sorry

end intersection_and_conditions_l562_56297


namespace odd_function_negative_domain_l562_56295

-- Define an odd function f on ℝ
def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the condition for f when x ≥ 0
def fPositive (f : ℝ → ℝ) : Prop :=
  ∀ x, x ≥ 0 → f x = x * (1 + x)

-- Theorem statement
theorem odd_function_negative_domain
  (f : ℝ → ℝ) (odd : isOddFunction f) (pos : fPositive f) :
  ∀ x, x < 0 → f x = x * (1 - x) := by
  sorry

end odd_function_negative_domain_l562_56295


namespace unique_assignment_l562_56276

-- Define the polyhedron structure
structure Polyhedron :=
  (faces : Fin 2022 → ℝ)
  (adjacent : Fin 2022 → Finset (Fin 2022))
  (adjacent_symmetric : ∀ i j, j ∈ adjacent i ↔ i ∈ adjacent j)

-- Define the property of being a valid number assignment
def ValidAssignment (p : Polyhedron) : Prop :=
  ∀ i, p.faces i = if i = 0 then 26
                   else if i = 1 then 4
                   else if i = 2 then 2022
                   else (p.adjacent i).sum p.faces / (p.adjacent i).card

-- Theorem statement
theorem unique_assignment (p : Polyhedron) :
  ∃! f : Fin 2022 → ℝ, ValidAssignment { faces := f, adjacent := p.adjacent, adjacent_symmetric := p.adjacent_symmetric } :=
sorry

end unique_assignment_l562_56276


namespace nested_sqrt_twelve_l562_56218

theorem nested_sqrt_twelve (x : ℝ) : x > 0 ∧ x = Real.sqrt (12 + x) → x = 4 := by
  sorry

end nested_sqrt_twelve_l562_56218


namespace bookshelf_theorem_l562_56266

/-- Given that:
    - A algebra books and H geometry books fill a bookshelf
    - S algebra books and M geometry books fill the same bookshelf
    - E algebra books alone fill the same bookshelf
    - A, H, S, M, E are different positive integers
    - Geometry books are thicker than algebra books
    Prove that E = (A * M - S * H) / (M - H) -/
theorem bookshelf_theorem (A H S M E : ℕ) 
    (hA : A > 0) (hH : H > 0) (hS : S > 0) (hM : M > 0) (hE : E > 0)
    (hDiff : A ≠ H ∧ A ≠ S ∧ A ≠ M ∧ A ≠ E ∧ 
             H ≠ S ∧ H ≠ M ∧ H ≠ E ∧ 
             S ≠ M ∧ S ≠ E ∧ 
             M ≠ E)
    (hFillAH : ∃ d e : ℚ, d > 0 ∧ e > 0 ∧ e > d ∧ A * d + H * e = E * d)
    (hFillSM : ∃ d e : ℚ, d > 0 ∧ e > 0 ∧ e > d ∧ S * d + M * e = E * d) :
  E = (A * M - S * H) / (M - H) := by
sorry

end bookshelf_theorem_l562_56266


namespace seashell_count_l562_56280

theorem seashell_count (sam_shells joan_shells : ℕ) 
  (h1 : sam_shells = 35) 
  (h2 : joan_shells = 18) : 
  sam_shells + joan_shells = 53 := by
  sorry

end seashell_count_l562_56280


namespace ellipse_equation_l562_56229

theorem ellipse_equation (a b : ℝ) (M : ℝ × ℝ) :
  a > b ∧ b > 0 ∧
  (M.1^2 / a^2 + M.2^2 / b^2 = 1) ∧
  (∃ A B : ℝ × ℝ, 
    A.1 = 0 ∧ B.1 = 0 ∧
    (M.1 - A.1)^2 + (M.2 - A.2)^2 = (2 * Real.sqrt 6 / 3)^2 ∧
    (M.1 - B.1)^2 + (M.2 - B.2)^2 = (2 * Real.sqrt 6 / 3)^2 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = (2 * Real.sqrt 6 / 3)^2) →
  a^2 = 6 ∧ b^2 = 4 := by
sorry

end ellipse_equation_l562_56229


namespace reflection_of_M_across_x_axis_l562_56267

/-- The reflection of a point across the x-axis --/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The original point M --/
def M : ℝ × ℝ := (1, 2)

theorem reflection_of_M_across_x_axis :
  reflect_x M = (1, -2) := by sorry

end reflection_of_M_across_x_axis_l562_56267


namespace arc_length_for_60_degrees_l562_56225

/-- Given a circle with radius 10 cm and a central angle of 60°, 
    the length of the corresponding arc is 10π/3 cm. -/
theorem arc_length_for_60_degrees (r : ℝ) (θ : ℝ) (l : ℝ) : 
  r = 10 → θ = 60 * π / 180 → l = r * θ → l = 10 * π / 3 := by
  sorry

end arc_length_for_60_degrees_l562_56225


namespace y_relationship_l562_56264

theorem y_relationship : ∀ y₁ y₂ y₃ : ℝ,
  y₁ = (0.5 : ℝ) ^ (1/4 : ℝ) →
  y₂ = (0.6 : ℝ) ^ (1/4 : ℝ) →
  y₃ = (0.6 : ℝ) ^ (1/5 : ℝ) →
  y₁ < y₂ ∧ y₂ < y₃ := by
  sorry

end y_relationship_l562_56264


namespace work_completion_time_l562_56234

/-- Given that:
    1. Ravi can do a piece of work in 15 days
    2. Ravi and another person together can do the work in 10 days
    Prove that the other person can do the work alone in 30 days -/
theorem work_completion_time (ravi_time : ℝ) (joint_time : ℝ) (other_time : ℝ) :
  ravi_time = 15 →
  joint_time = 10 →
  (1 / ravi_time + 1 / other_time = 1 / joint_time) →
  other_time = 30 := by
  sorry

#check work_completion_time

end work_completion_time_l562_56234


namespace ratio_percentage_difference_l562_56212

theorem ratio_percentage_difference (A B : ℝ) (h : A / B = 5 / 8) :
  (B - A) / B = 37.5 / 100 ∧ (B - A) / A = 60 / 100 := by
  sorry

end ratio_percentage_difference_l562_56212


namespace rhombus_area_l562_56210

theorem rhombus_area (d₁ d₂ : ℝ) : 
  d₁^2 - 6*d₁ + 8 = 0 → 
  d₂^2 - 6*d₂ + 8 = 0 → 
  d₁ ≠ d₂ →
  (1/2) * d₁ * d₂ = 4 := by
  sorry


end rhombus_area_l562_56210


namespace all_less_than_one_l562_56207

theorem all_less_than_one (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a^2 < b) (hbc : b^2 < c) (hca : c^2 < a) :
  a < 1 ∧ b < 1 ∧ c < 1 := by
  sorry

end all_less_than_one_l562_56207


namespace p_neither_necessary_nor_sufficient_l562_56245

-- Define p and q as propositions depending on x and y
def p (x y : ℝ) : Prop := x + y ≠ -2
def q (x : ℝ) : Prop := x ≠ 0  -- Assuming 'x' means x is non-zero

-- Define the theorem
theorem p_neither_necessary_nor_sufficient :
  ∃ (x y : ℝ), y ≠ -1 ∧
  (q x ∧ ¬(p x y)) ∧  -- p is not necessary
  (p x y ∧ ¬(q x))    -- p is not sufficient
  := by sorry

end p_neither_necessary_nor_sufficient_l562_56245


namespace fraction_equality_l562_56231

theorem fraction_equality : (45 : ℚ) / (8 - 3 / 7) = 315 / 53 := by sorry

end fraction_equality_l562_56231


namespace intersection_and_subset_condition_l562_56204

def A : Set ℝ := {x | ∃ y, y = Real.sqrt (6 + 5*x - x^2)}
def B (m : ℝ) : Set ℝ := {x | (x - 1 + m) * (x - 1 - m) ≤ 0}

theorem intersection_and_subset_condition :
  (∃ m : ℝ, m = 3 ∧ A ∩ B m = {x | -1 ≤ x ∧ x ≤ 4}) ∧
  (∀ m : ℝ, m > 0 → (A ⊆ B m → m ≥ 5)) := by sorry

end intersection_and_subset_condition_l562_56204


namespace diagonal_intersection_y_value_l562_56286

/-- A square in the coordinate plane with specific properties -/
structure Square where
  vertex : ℝ × ℝ
  diagonal_intersection_x : ℝ
  area : ℝ

/-- The y-coordinate of the diagonal intersection point of the square -/
def diagonal_intersection_y (s : Square) : ℝ :=
  s.vertex.2 + (s.diagonal_intersection_x - s.vertex.1)

/-- Theorem stating the y-coordinate of the diagonal intersection point -/
theorem diagonal_intersection_y_value (s : Square) 
  (h1 : s.vertex = (-6, -4))
  (h2 : s.diagonal_intersection_x = 3)
  (h3 : s.area = 324) :
  diagonal_intersection_y s = 5 := by
  sorry


end diagonal_intersection_y_value_l562_56286


namespace saras_sister_notebooks_l562_56293

theorem saras_sister_notebooks (initial final ordered lost : ℕ) : 
  final = 8 → ordered = 6 → lost = 2 → initial + ordered - lost = final → initial = 4 := by
  sorry

end saras_sister_notebooks_l562_56293


namespace p_shape_points_count_l562_56222

/-- Represents the "П" shape formed from a square --/
structure PShape :=
  (side_length : ℕ)

/-- Calculates the number of points along the "П" shape --/
def count_points (p : PShape) : ℕ :=
  3 * (p.side_length + 1) - 2

theorem p_shape_points_count :
  ∀ (p : PShape), p.side_length = 10 → count_points p = 31 :=
by
  sorry

end p_shape_points_count_l562_56222


namespace line_product_l562_56247

/-- Given a line y = mx + b passing through points (0, -3) and (3, 6), prove that mb = -9 -/
theorem line_product (m b : ℝ) : 
  (0 : ℝ) = m * 0 + b ∧ 
  (-3 : ℝ) = m * 0 + b ∧ 
  (6 : ℝ) = m * 3 + b → 
  m * b = -9 := by
  sorry

end line_product_l562_56247


namespace cubic_root_equation_solution_l562_56202

theorem cubic_root_equation_solution :
  ∃ x : ℝ, (((3 - x) ^ (1/3 : ℝ)) + ((x - 1) ^ (1/2 : ℝ)) = 2) ∧ (x = 2) :=
by sorry

end cubic_root_equation_solution_l562_56202


namespace marbles_redistribution_l562_56273

/-- The number of marbles Tyrone initially had -/
def tyrone_initial : ℕ := 120

/-- The number of marbles Eric initially had -/
def eric_initial : ℕ := 18

/-- The ratio of Tyrone's marbles to Eric's after redistribution -/
def final_ratio : ℕ := 3

/-- The number of marbles Tyrone gave to Eric -/
def marbles_given : ℚ := 16.5

theorem marbles_redistribution :
  let tyrone_final := tyrone_initial - marbles_given
  let eric_final := eric_initial + marbles_given
  tyrone_final = final_ratio * eric_final := by sorry

end marbles_redistribution_l562_56273


namespace circle_diameter_points_exist_l562_56255

/-- Represents a point on the circumference of a circle -/
structure CirclePoint where
  angle : ℝ
  property : 0 ≤ angle ∧ angle < 2 * Real.pi

/-- Represents an arc on the circumference of a circle -/
structure Arc where
  start : CirclePoint
  length : ℝ
  property : 0 < length ∧ length ≤ 2 * Real.pi

/-- The main theorem statement -/
theorem circle_diameter_points_exist (k : ℕ) (points : Finset CirclePoint) (arcs : Finset Arc) :
  points.card = 3 * k →
  arcs.card = 3 * k →
  (∃ (s₁ : Finset Arc), s₁.card = k ∧ ∀ a ∈ s₁, a.length = 1) →
  (∃ (s₂ : Finset Arc), s₂.card = k ∧ ∀ a ∈ s₂, a.length = 2) →
  (∃ (s₃ : Finset Arc), s₃.card = k ∧ ∀ a ∈ s₃, a.length = 3) →
  ∃ (p₁ p₂ : CirclePoint), p₁ ∈ points ∧ p₂ ∈ points ∧ abs (p₁.angle - p₂.angle) = Real.pi :=
by sorry

end circle_diameter_points_exist_l562_56255


namespace original_fraction_proof_l562_56290

theorem original_fraction_proof (N D : ℚ) :
  (N > 0) →
  (D > 0) →
  ((1.15 * N) / (0.92 * D) = 15 / 16) →
  (N / D = 4 / 3) :=
by
  sorry

end original_fraction_proof_l562_56290


namespace managers_salary_l562_56289

theorem managers_salary (num_employees : ℕ) (avg_salary : ℚ) (avg_increase : ℚ) :
  num_employees = 20 ∧ 
  avg_salary = 1600 ∧ 
  avg_increase = 100 →
  (num_employees * avg_salary + (avg_salary + avg_increase) * (num_employees + 1) - num_employees * avg_salary) = 3700 :=
by sorry

end managers_salary_l562_56289


namespace martha_savings_l562_56270

def daily_allowance : ℚ := 12
def normal_saving_rate : ℚ := 1/2
def exception_saving_rate : ℚ := 1/4
def days_in_week : ℕ := 7
def normal_saving_days : ℕ := 6
def exception_saving_days : ℕ := 1

theorem martha_savings : 
  (normal_saving_days : ℚ) * (daily_allowance * normal_saving_rate) + 
  (exception_saving_days : ℚ) * (daily_allowance * exception_saving_rate) = 39 := by
  sorry

end martha_savings_l562_56270


namespace keaton_orange_harvest_frequency_l562_56219

/-- Represents Keaton's farm earnings and harvest information -/
structure FarmData where
  yearly_earnings : ℕ
  apple_harvest_interval : ℕ
  apple_harvest_value : ℕ
  orange_harvest_value : ℕ

/-- Calculates the frequency of orange harvests in months -/
def orange_harvest_frequency (data : FarmData) : ℕ :=
  12 / (data.yearly_earnings - (12 / data.apple_harvest_interval * data.apple_harvest_value)) / data.orange_harvest_value

/-- Theorem stating that Keaton's orange harvest frequency is 2 months -/
theorem keaton_orange_harvest_frequency :
  orange_harvest_frequency ⟨420, 3, 30, 50⟩ = 2 := by
  sorry

end keaton_orange_harvest_frequency_l562_56219


namespace bridget_apples_l562_56281

theorem bridget_apples (x : ℕ) : 
  (x : ℚ) / 3 + 4 + 6 = x → x = 15 := by
  sorry

end bridget_apples_l562_56281


namespace cos_22_5_squared_minus_sin_22_5_squared_l562_56224

theorem cos_22_5_squared_minus_sin_22_5_squared : 
  Real.cos (22.5 * π / 180) ^ 2 - Real.sin (22.5 * π / 180) ^ 2 = Real.sqrt 2 / 2 := by
  sorry

end cos_22_5_squared_minus_sin_22_5_squared_l562_56224


namespace centipede_dressing_sequences_l562_56265

/-- The number of legs a centipede has -/
def num_legs : ℕ := 10

/-- The total number of items (socks and shoes) -/
def total_items : ℕ := 2 * num_legs

/-- The number of valid sequences for a centipede to wear its socks and shoes -/
def valid_sequences : ℕ := Nat.factorial total_items / (2^ num_legs)

/-- Theorem stating the number of valid sequences for a centipede to wear its socks and shoes -/
theorem centipede_dressing_sequences :
  valid_sequences = Nat.factorial total_items / (2 ^ num_legs) :=
by sorry

end centipede_dressing_sequences_l562_56265


namespace nonnegative_integer_solutions_x_squared_eq_6x_l562_56230

theorem nonnegative_integer_solutions_x_squared_eq_6x :
  ∃! n : ℕ, (∃ s : Finset ℕ, s.card = n ∧
    ∀ x : ℕ, x ∈ s ↔ x^2 = 6*x) ∧ n = 2 := by sorry

end nonnegative_integer_solutions_x_squared_eq_6x_l562_56230


namespace max_guaranteed_amount_l562_56299

/-- Represents a set of bank cards with values from 1 to n rubles -/
def BankCards (n : ℕ) := Finset (Fin n)

/-- The strategy function takes a number of cards and returns the optimal request amount -/
def strategy (n : ℕ) : ℕ := n / 2

/-- Calculates the guaranteed amount for a given strategy on a set of cards -/
def guaranteedAmount (cards : BankCards 100) (s : ℕ → ℕ) : ℕ :=
  (cards.filter (λ i => i.val + 1 ≥ s 100)).card * s 100

theorem max_guaranteed_amount :
  ∀ (cards : BankCards 100),
    ∀ (s : ℕ → ℕ),
      guaranteedAmount cards s ≤ guaranteedAmount cards strategy ∧
      guaranteedAmount cards strategy = 2550 := by
  sorry

#eval strategy 100  -- Should output 50

end max_guaranteed_amount_l562_56299


namespace garden_trees_l562_56277

/-- The number of trees in a garden with given specifications -/
def num_trees (yard_length : ℕ) (tree_distance : ℕ) : ℕ :=
  yard_length / tree_distance + 1

/-- Theorem: The number of trees in a 500-metre garden with 20-metre spacing is 26 -/
theorem garden_trees : num_trees 500 20 = 26 := by
  sorry

end garden_trees_l562_56277


namespace g_comp_three_roots_l562_56298

/-- A quadratic function g(x) = x^2 + 4x + d where d is a real parameter -/
def g (d : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + d

/-- The composition of g with itself -/
def g_comp (d : ℝ) (x : ℝ) : ℝ := g d (g d x)

/-- Theorem stating that g(g(x)) has exactly 3 distinct real roots iff d = 0 -/
theorem g_comp_three_roots (d : ℝ) : 
  (∃! (s : Finset ℝ), s.card = 3 ∧ ∀ x ∈ s, g_comp d x = 0) ↔ d = 0 := by
  sorry

end g_comp_three_roots_l562_56298


namespace peanuts_added_l562_56256

theorem peanuts_added (initial_peanuts final_peanuts : ℕ) 
  (h1 : initial_peanuts = 4)
  (h2 : final_peanuts = 10) :
  final_peanuts - initial_peanuts = 6 := by
  sorry

end peanuts_added_l562_56256


namespace greatest_number_l562_56263

theorem greatest_number : 
  let a := 1000 + 0.01
  let b := 1000 * 0.01
  let c := 1000 / 0.01
  let d := 0.01 / 1000
  let e := 1000 - 0.01
  (c > a) ∧ (c > b) ∧ (c > d) ∧ (c > e) := by
  sorry

end greatest_number_l562_56263


namespace gear_speed_proportion_l562_56275

/-- Represents a gear with a number of teeth and angular speed -/
structure Gear where
  teeth : ℕ
  speed : ℝ

/-- Represents a system of four meshed gears -/
structure GearSystem where
  A : Gear
  B : Gear
  C : Gear
  D : Gear

/-- The proportion of angular speeds for a system of four meshed gears -/
def angular_speed_proportion (g : GearSystem) : Prop :=
  ∃ (k : ℝ), k > 0 ∧
    g.A.speed = k * g.B.teeth * g.C.teeth * g.D.teeth ∧
    g.B.speed = k * g.A.teeth * g.C.teeth * g.D.teeth ∧
    g.C.speed = k * g.A.teeth * g.B.teeth * g.D.teeth ∧
    g.D.speed = k * g.A.teeth * g.B.teeth * g.C.teeth

theorem gear_speed_proportion (g : GearSystem) :
  angular_speed_proportion g → True :=
by
  sorry

end gear_speed_proportion_l562_56275


namespace tim_income_percentage_l562_56223

theorem tim_income_percentage (tim juan mary : ℝ) 
  (h1 : mary = 1.7 * tim) 
  (h2 : mary = 1.02 * juan) : 
  (juan - tim) / juan = 0.4 := by
sorry

end tim_income_percentage_l562_56223


namespace minimum_guests_l562_56254

theorem minimum_guests (total_food : ℝ) (max_per_guest : ℝ) (h1 : total_food = 411) (h2 : max_per_guest = 2.5) :
  ⌈total_food / max_per_guest⌉ = 165 := by
  sorry

end minimum_guests_l562_56254


namespace a_values_in_A_l562_56241

def A : Set ℝ := {2, 4, 6}

theorem a_values_in_A : {a : ℝ | a ∈ A ∧ (6 - a) ∈ A} = {2, 4} := by
  sorry

end a_values_in_A_l562_56241


namespace difference_of_squares_625_575_l562_56221

theorem difference_of_squares_625_575 : 625^2 - 575^2 = 60000 := by
  sorry

end difference_of_squares_625_575_l562_56221


namespace angle_A_in_special_triangle_l562_56272

theorem angle_A_in_special_triangle (A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C →  -- Ensuring positive angles
  B = A + 10 →             -- Given condition
  C = B + 10 →             -- Given condition
  A + B + C = 180 →        -- Sum of angles in a triangle
  A = 50 := by sorry

end angle_A_in_special_triangle_l562_56272


namespace min_perimeter_cross_section_min_perimeter_problem_l562_56288

/-- Regular triangular pyramid with given dimensions -/
structure RegularTriangularPyramid where
  base_side_length : ℝ
  lateral_edge_length : ℝ

/-- Intersection plane for the pyramid -/
structure IntersectionPlane where
  base_point : Point
  intersection_point1 : Point
  intersection_point2 : Point

/-- Theorem stating the minimum perimeter of the cross-sectional triangle -/
theorem min_perimeter_cross_section 
  (pyramid : RegularTriangularPyramid) 
  (plane : IntersectionPlane) : ℝ :=
  sorry

/-- Main theorem proving the minimum perimeter for the given problem -/
theorem min_perimeter_problem : 
  ∀ (pyramid : RegularTriangularPyramid) 
    (plane : IntersectionPlane),
  pyramid.base_side_length = 4 ∧ 
  pyramid.lateral_edge_length = 8 →
  min_perimeter_cross_section pyramid plane = 11 :=
sorry

end min_perimeter_cross_section_min_perimeter_problem_l562_56288


namespace classroom_chairs_l562_56209

theorem classroom_chairs (blue_chairs : ℕ) (green_chairs : ℕ) (white_chairs : ℕ) :
  blue_chairs = 10 →
  green_chairs = 3 * blue_chairs →
  white_chairs = blue_chairs + green_chairs - 13 →
  blue_chairs + green_chairs + white_chairs = 67 := by
sorry

end classroom_chairs_l562_56209


namespace sufficient_not_necessary_l562_56291

theorem sufficient_not_necessary (a b c d : ℝ) :
  (a > b ∧ c > d → a * c + b * d > b * c + a * d) ∧
  ∃ a b c d : ℝ, a * c + b * d > b * c + a * d ∧ ¬(a > b ∧ c > d) :=
sorry

end sufficient_not_necessary_l562_56291
