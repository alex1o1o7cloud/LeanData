import Mathlib

namespace NUMINAMATH_CALUDE_digits_until_2014_l2320_232090

def odd_sequence (n : ℕ) : ℕ := 2 * n - 1

def digit_count (n : ℕ) : ℕ :=
  if n < 10 then 1
  else if n < 100 then 2
  else if n < 1000 then 3
  else 4

def total_digits (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ i => digit_count (odd_sequence (i + 1)))

theorem digits_until_2014 :
  ∃ n : ℕ, odd_sequence n > 2014 ∧ total_digits (n - 1) = 7850 := by sorry

end NUMINAMATH_CALUDE_digits_until_2014_l2320_232090


namespace NUMINAMATH_CALUDE_x_power_2187_minus_reciprocal_l2320_232044

theorem x_power_2187_minus_reciprocal (x : ℂ) (h : x - 1/x = Complex.I * Real.sqrt 3) :
  x^2187 - 1/x^2187 = Complex.I * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_x_power_2187_minus_reciprocal_l2320_232044


namespace NUMINAMATH_CALUDE_vegetable_growth_rate_equation_l2320_232065

theorem vegetable_growth_rate_equation 
  (initial_production final_production : ℝ) 
  (growth_years : ℕ) 
  (x : ℝ) 
  (h1 : initial_production = 800)
  (h2 : final_production = 968)
  (h3 : growth_years = 2)
  (h4 : final_production = initial_production * (1 + x) ^ growth_years) :
  800 * (1 + x)^2 = 968 := by
sorry

end NUMINAMATH_CALUDE_vegetable_growth_rate_equation_l2320_232065


namespace NUMINAMATH_CALUDE_equation_solution_l2320_232001

theorem equation_solution : 
  ∃ x : ℚ, (x ≠ 2) ∧ (7 * x / (x - 2) + 4 / (x - 2) = 6 / (x - 2)) → x = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2320_232001


namespace NUMINAMATH_CALUDE_floor_product_twenty_l2320_232078

theorem floor_product_twenty (x : ℝ) : 
  ⌊x * ⌊x⌋⌋ = 20 ↔ 5 ≤ x ∧ x < 5.25 := by sorry

end NUMINAMATH_CALUDE_floor_product_twenty_l2320_232078


namespace NUMINAMATH_CALUDE_journey_solution_l2320_232005

/-- Represents the problem of Xiaogang and Xiaoqiang's journey --/
structure JourneyProblem where
  total_distance : ℝ
  meeting_time : ℝ
  xiaogang_extra_distance : ℝ
  xiaogang_remaining_time : ℝ

/-- Represents the solution to the journey problem --/
structure JourneySolution where
  xiaogang_speed : ℝ
  xiaoqiang_speed : ℝ
  xiaoqiang_remaining_time : ℝ

/-- Theorem stating the correct solution for the given problem --/
theorem journey_solution (p : JourneyProblem) 
  (h1 : p.meeting_time = 2)
  (h2 : p.xiaogang_extra_distance = 24)
  (h3 : p.xiaogang_remaining_time = 0.5) :
  ∃ (s : JourneySolution),
    s.xiaogang_speed = 16 ∧
    s.xiaoqiang_speed = 4 ∧
    s.xiaoqiang_remaining_time = 8 ∧
    p.total_distance = s.xiaogang_speed * (p.meeting_time + p.xiaogang_remaining_time) ∧
    p.total_distance = (s.xiaogang_speed * p.meeting_time - p.xiaogang_extra_distance) + (s.xiaoqiang_speed * s.xiaoqiang_remaining_time) :=
by
  sorry


end NUMINAMATH_CALUDE_journey_solution_l2320_232005


namespace NUMINAMATH_CALUDE_line_not_in_fourth_quadrant_l2320_232060

theorem line_not_in_fourth_quadrant 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b < 0) 
  (hc : c > 0) : 
  ∀ x y : ℝ, a * x + b * y + c = 0 → ¬(x > 0 ∧ y < 0) := by
sorry

end NUMINAMATH_CALUDE_line_not_in_fourth_quadrant_l2320_232060


namespace NUMINAMATH_CALUDE_girls_from_clay_middle_school_l2320_232064

theorem girls_from_clay_middle_school
  (total_students : ℕ)
  (total_boys : ℕ)
  (total_girls : ℕ)
  (jonas_students : ℕ)
  (clay_students : ℕ)
  (hart_students : ℕ)
  (jonas_boys : ℕ)
  (h1 : total_students = 150)
  (h2 : total_boys = 90)
  (h3 : total_girls = 60)
  (h4 : jonas_students = 50)
  (h5 : clay_students = 70)
  (h6 : hart_students = 30)
  (h7 : jonas_boys = 25)
  (h8 : total_students = total_boys + total_girls)
  (h9 : total_students = jonas_students + clay_students + hart_students)
  : ∃ clay_girls : ℕ, clay_girls = 30 ∧ clay_girls ≤ clay_students :=
by sorry

end NUMINAMATH_CALUDE_girls_from_clay_middle_school_l2320_232064


namespace NUMINAMATH_CALUDE_divisor_of_infinite_set_l2320_232029

theorem divisor_of_infinite_set (A : Set ℕ+) 
  (h_infinite : Set.Infinite A)
  (h_finite_subset : ∀ B : Set ℕ+, B ⊆ A → Set.Finite B → 
    ∃ b : ℕ+, b > 1 ∧ ∀ x ∈ B, b ∣ x) :
  ∃ d : ℕ+, d > 1 ∧ ∀ x ∈ A, d ∣ x :=
sorry

end NUMINAMATH_CALUDE_divisor_of_infinite_set_l2320_232029


namespace NUMINAMATH_CALUDE_one_student_passes_probability_l2320_232092

/-- The probability that exactly one out of three students passes, given their individual passing probabilities -/
theorem one_student_passes_probability
  (p_jia p_yi p_bing : ℚ)
  (h_jia : p_jia = 4 / 5)
  (h_yi : p_yi = 3 / 5)
  (h_bing : p_bing = 7 / 10) :
  (p_jia * (1 - p_yi) * (1 - p_bing)) +
  ((1 - p_jia) * p_yi * (1 - p_bing)) +
  ((1 - p_jia) * (1 - p_yi) * p_bing) =
  47 / 250 := by
  sorry

end NUMINAMATH_CALUDE_one_student_passes_probability_l2320_232092


namespace NUMINAMATH_CALUDE_intersection_value_l2320_232075

theorem intersection_value (m : ℝ) (B : Set ℝ) : 
  ({1, m - 2} : Set ℝ) ∩ B = {2} → m = 4 := by
sorry

end NUMINAMATH_CALUDE_intersection_value_l2320_232075


namespace NUMINAMATH_CALUDE_fence_cost_square_plot_l2320_232077

/-- The cost of building a fence around a square plot -/
theorem fence_cost_square_plot (area : ℝ) (price_per_foot : ℝ) (cost : ℝ) : 
  area = 289 → price_per_foot = 54 → cost = 4 * Real.sqrt area * price_per_foot → cost = 3672 := by
  sorry

end NUMINAMATH_CALUDE_fence_cost_square_plot_l2320_232077


namespace NUMINAMATH_CALUDE_cos_alpha_plus_17pi_12_l2320_232012

theorem cos_alpha_plus_17pi_12 (α : ℝ) (h : Real.sin (α - π / 12) = 1 / 3) :
  Real.cos (α + 17 * π / 12) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_plus_17pi_12_l2320_232012


namespace NUMINAMATH_CALUDE_f_composition_value_l2320_232018

def f (x : ℚ) : ℚ := x⁻¹ + x⁻¹ / (1 + x⁻¹)

theorem f_composition_value : f (f (-3)) = 24/5 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_value_l2320_232018


namespace NUMINAMATH_CALUDE_special_line_equation_l2320_232037

/-- A line passing through (5, 2) with y-intercept twice the x-intercept -/
structure SpecialLine where
  -- Slope-intercept form: y = mx + b
  m : ℝ
  b : ℝ
  -- The line passes through (5, 2)
  point_condition : 2 = m * 5 + b
  -- The y-intercept is twice the x-intercept
  intercept_condition : b = -2 * (b / m)

/-- The equation of the special line is either 2x + y - 12 = 0 or 2x - 5y = 0 -/
theorem special_line_equation (l : SpecialLine) :
  (∀ x y, 2 * x + y - 12 = 0 ↔ y = l.m * x + l.b) ∨
  (∀ x y, 2 * x - 5 * y = 0 ↔ y = l.m * x + l.b) :=
sorry

end NUMINAMATH_CALUDE_special_line_equation_l2320_232037


namespace NUMINAMATH_CALUDE_slope_of_line_l2320_232016

theorem slope_of_line (x y : ℝ) :
  4 * x + 7 * y = 28 → (y - 4) / x = -4 / 7 :=
by sorry

end NUMINAMATH_CALUDE_slope_of_line_l2320_232016


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l2320_232007

theorem complex_magnitude_problem (z : ℂ) : 
  (Complex.I / (1 - Complex.I)) * z = 1 → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l2320_232007


namespace NUMINAMATH_CALUDE_cube_surface_area_equals_prism_volume_l2320_232072

/-- The surface area of a cube with volume equal to a rectangular prism of dimensions 12 × 3 × 18 is equal to the volume of the prism. -/
theorem cube_surface_area_equals_prism_volume :
  let prism_length : ℝ := 12
  let prism_width : ℝ := 3
  let prism_height : ℝ := 18
  let prism_volume := prism_length * prism_width * prism_height
  let cube_edge := (prism_volume) ^ (1/3 : ℝ)
  let cube_surface_area := 6 * cube_edge ^ 2
  cube_surface_area = prism_volume := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_equals_prism_volume_l2320_232072


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2320_232046

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x, (1 - 2*x)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  a₀ + a₁ + a₂ + a₃ = -15 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2320_232046


namespace NUMINAMATH_CALUDE_min_value_of_f_l2320_232024

/-- The function f(x) = x^2 + 8x + 25 -/
def f (x : ℝ) : ℝ := x^2 + 8*x + 25

/-- The minimum value of f(x) is 9 -/
theorem min_value_of_f : ∃ (m : ℝ), ∀ (x : ℝ), f x ≥ m ∧ ∃ (x₀ : ℝ), f x₀ = m ∧ m = 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2320_232024


namespace NUMINAMATH_CALUDE_car_distance_theorem_l2320_232042

-- Define the length of the line of soldiers
def line_length : ℝ := 1

-- Define the distance each soldier marches
def soldier_distance : ℝ := 15

-- Define the speed ratio between the car and soldiers
def speed_ratio : ℝ := 2

-- Theorem statement
theorem car_distance_theorem :
  let car_distance := soldier_distance * speed_ratio * line_length
  car_distance = 30 := by sorry

end NUMINAMATH_CALUDE_car_distance_theorem_l2320_232042


namespace NUMINAMATH_CALUDE_cattle_selling_price_l2320_232006

/-- Proves that the selling price per pound for cattle is $1.60 given the specified conditions --/
theorem cattle_selling_price
  (num_cattle : ℕ)
  (purchase_price : ℝ)
  (feed_cost_percentage : ℝ)
  (weight_per_cattle : ℝ)
  (profit : ℝ)
  (h1 : num_cattle = 100)
  (h2 : purchase_price = 40000)
  (h3 : feed_cost_percentage = 0.20)
  (h4 : weight_per_cattle = 1000)
  (h5 : profit = 112000)
  : ∃ (selling_price_per_pound : ℝ),
    selling_price_per_pound = 1.60 ∧
    selling_price_per_pound * (num_cattle * weight_per_cattle) =
      purchase_price + (feed_cost_percentage * purchase_price) + profit :=
by
  sorry

end NUMINAMATH_CALUDE_cattle_selling_price_l2320_232006


namespace NUMINAMATH_CALUDE_cakes_served_total_l2320_232087

/-- The number of cakes served during lunch today -/
def lunch_cakes : ℕ := 5

/-- The number of cakes served during dinner today -/
def dinner_cakes : ℕ := 6

/-- The number of cakes served yesterday -/
def yesterday_cakes : ℕ := 3

/-- The total number of cakes served over two days -/
def total_cakes : ℕ := lunch_cakes + dinner_cakes + yesterday_cakes

theorem cakes_served_total :
  total_cakes = 14 := by sorry

end NUMINAMATH_CALUDE_cakes_served_total_l2320_232087


namespace NUMINAMATH_CALUDE_team_selection_count_l2320_232097

def boys : ℕ := 7
def girls : ℕ := 9
def team_size : ℕ := 7
def boys_in_team : ℕ := 4
def girls_in_team : ℕ := 3

theorem team_selection_count :
  (Nat.choose boys boys_in_team) * (Nat.choose girls girls_in_team) = 2940 := by
  sorry

end NUMINAMATH_CALUDE_team_selection_count_l2320_232097


namespace NUMINAMATH_CALUDE_ratio_limit_is_27_l2320_232076

/-- The ratio of the largest element to the sum of other elements in the geometric series -/
def ratio (n : ℕ) : ℚ :=
  let a := 3
  let r := 10
  (a * r^n) / (a * (r^n - 1) / (r - 1))

/-- The limit of the ratio as n approaches infinity is 27 -/
theorem ratio_limit_is_27 : ∀ ε > 0, ∃ N, ∀ n ≥ N, |ratio n - 27| < ε :=
sorry

end NUMINAMATH_CALUDE_ratio_limit_is_27_l2320_232076


namespace NUMINAMATH_CALUDE_candy_bar_problem_l2320_232008

/-- Given the candy bar distribution problem, prove that 40% of Jacqueline's candy bars is 120 -/
theorem candy_bar_problem :
  let fred_candy : ℕ := 12
  let bob_candy : ℕ := fred_candy + 6
  let total_candy : ℕ := fred_candy + bob_candy
  let jacqueline_candy : ℕ := 10 * total_candy
  (40 : ℚ) / 100 * jacqueline_candy = 120 := by sorry

end NUMINAMATH_CALUDE_candy_bar_problem_l2320_232008


namespace NUMINAMATH_CALUDE_exam_students_count_l2320_232084

theorem exam_students_count (total_average : ℝ) (excluded_average : ℝ) (new_average : ℝ) 
  (excluded_count : ℕ) (h1 : total_average = 80) (h2 : excluded_average = 30) 
  (h3 : new_average = 90) (h4 : excluded_count = 5) :
  ∃ (N : ℕ), N > 0 ∧ 
  (N * total_average - excluded_count * excluded_average) / (N - excluded_count) = new_average :=
by
  sorry

end NUMINAMATH_CALUDE_exam_students_count_l2320_232084


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l2320_232063

theorem smallest_n_congruence : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → m < n → ¬(725 * m ≡ 1275 * m [ZMOD 35])) ∧ 
  (725 * n ≡ 1275 * n [ZMOD 35]) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l2320_232063


namespace NUMINAMATH_CALUDE_equation_transformations_correct_l2320_232025

theorem equation_transformations_correct 
  (a b c x y : ℝ) : 
  (a = b → a * c = b * c) ∧ 
  (a * (x^2 + 1) = b * (x^2 + 1) → a = b) ∧ 
  (a = b → a / c^2 = b / c^2) ∧ 
  (x = y → x - 3 = y - 3) := by
  sorry

end NUMINAMATH_CALUDE_equation_transformations_correct_l2320_232025


namespace NUMINAMATH_CALUDE_probability_five_green_marbles_l2320_232070

/-- The probability of drawing exactly k successes in n trials 
    with probability p for each success -/
def binomialProbability (n k : ℕ) (p : ℚ) : ℚ :=
  Nat.choose n k * p^k * (1 - p)^(n - k)

/-- The number of green marbles in the bag -/
def greenMarbles : ℕ := 9

/-- The number of purple marbles in the bag -/
def purpleMarbles : ℕ := 6

/-- The total number of marbles in the bag -/
def totalMarbles : ℕ := greenMarbles + purpleMarbles

/-- The probability of drawing a green marble -/
def pGreen : ℚ := greenMarbles / totalMarbles

/-- The number of marbles drawn -/
def numDraws : ℕ := 8

/-- The number of green marbles we want to draw -/
def numGreenDraws : ℕ := 5

theorem probability_five_green_marbles :
  binomialProbability numDraws numGreenDraws pGreen = 108864 / 390625 := by
  sorry

end NUMINAMATH_CALUDE_probability_five_green_marbles_l2320_232070


namespace NUMINAMATH_CALUDE_combination_18_choose_4_l2320_232030

theorem combination_18_choose_4 : Nat.choose 18 4 = 3060 := by
  sorry

end NUMINAMATH_CALUDE_combination_18_choose_4_l2320_232030


namespace NUMINAMATH_CALUDE_equation_solution_and_condition_l2320_232058

theorem equation_solution_and_condition :
  ∃ x : ℝ,
    (8 * x^(1/3) - 4 * (x / x^(2/3)) = 12 + 2 * x^(1/3)) ∧
    (x ≥ Real.sqrt 144) ∧
    (x = 216) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_and_condition_l2320_232058


namespace NUMINAMATH_CALUDE_same_odd_dice_probability_l2320_232050

/-- The number of faces on each die -/
def num_faces : ℕ := 8

/-- The number of odd faces on each die -/
def num_odd_faces : ℕ := 4

/-- The number of dice rolled -/
def num_dice : ℕ := 4

/-- The probability of rolling the same odd number on all dice -/
def prob_same_odd : ℚ := 1 / 1024

theorem same_odd_dice_probability :
  (num_odd_faces : ℚ) / num_faces * (1 / num_faces) ^ (num_dice - 1) = prob_same_odd :=
by sorry

end NUMINAMATH_CALUDE_same_odd_dice_probability_l2320_232050


namespace NUMINAMATH_CALUDE_arithmetic_evaluation_l2320_232080

theorem arithmetic_evaluation : (9 - 2) - (4 - 1) = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_evaluation_l2320_232080


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2320_232017

/-- An isosceles triangle with congruent sides of 5 cm and perimeter of 17 cm has a base of 7 cm. -/
theorem isosceles_triangle_base_length : ∀ (base : ℝ),
  base > 0 →
  5 + 5 + base = 17 →
  base = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2320_232017


namespace NUMINAMATH_CALUDE_mr_green_potato_yield_l2320_232031

/-- Represents the dimensions of a rectangular garden in steps -/
structure GardenDimensions where
  length : ℕ
  width : ℕ

/-- Calculates the expected potato yield from a rectangular garden -/
def expected_potato_yield (garden : GardenDimensions) (step_length : ℝ) (yield_per_sqft : ℝ) : ℝ :=
  (garden.length : ℝ) * step_length * (garden.width : ℝ) * step_length * yield_per_sqft

/-- Theorem stating the expected potato yield for Mr. Green's garden -/
theorem mr_green_potato_yield :
  let garden := GardenDimensions.mk 18 25
  let step_length := 2.5
  let yield_per_sqft := 0.75
  expected_potato_yield garden step_length yield_per_sqft = 2109.375 := by
  sorry

end NUMINAMATH_CALUDE_mr_green_potato_yield_l2320_232031


namespace NUMINAMATH_CALUDE_gcd_of_75_and_100_l2320_232004

theorem gcd_of_75_and_100 : Nat.gcd 75 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_75_and_100_l2320_232004


namespace NUMINAMATH_CALUDE_propositions_relationship_l2320_232073

theorem propositions_relationship (x : ℝ) :
  (∀ x, x < 3 → x < 5) ↔ (∀ x, x ≥ 5 → x ≥ 3) :=
by sorry

end NUMINAMATH_CALUDE_propositions_relationship_l2320_232073


namespace NUMINAMATH_CALUDE_stratified_sampling_male_athletes_l2320_232069

/-- Represents the number of male athletes to be drawn in a stratified sampling -/
def male_athletes_drawn (total_athletes : ℕ) (male_athletes : ℕ) (sample_size : ℕ) : ℚ :=
  (male_athletes : ℚ) * (sample_size : ℚ) / (total_athletes : ℚ)

/-- Theorem stating that in the given scenario, 4 male athletes should be drawn -/
theorem stratified_sampling_male_athletes :
  male_athletes_drawn 30 20 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_male_athletes_l2320_232069


namespace NUMINAMATH_CALUDE_spadesuit_inequality_not_always_true_l2320_232036

def spadesuit (x y : ℝ) : ℝ := x^2 - y^2

theorem spadesuit_inequality_not_always_true :
  ¬ (∀ x y : ℝ, x ≥ y → spadesuit x y ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_spadesuit_inequality_not_always_true_l2320_232036


namespace NUMINAMATH_CALUDE_three_non_adjacent_from_ten_l2320_232098

/-- The number of ways to choose 3 non-adjacent items from a set of 10 items. -/
def non_adjacent_choices (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose (n - k + 1) k

/-- Theorem: There are 56 ways to choose 3 non-adjacent items from a set of 10 items. -/
theorem three_non_adjacent_from_ten : non_adjacent_choices 10 3 = 56 := by
  sorry

end NUMINAMATH_CALUDE_three_non_adjacent_from_ten_l2320_232098


namespace NUMINAMATH_CALUDE_lollipops_left_over_l2320_232055

/-- The number of lollipops Winnie has left after distributing them evenly among her friends -/
theorem lollipops_left_over (total : ℕ) (friends : ℕ) (h1 : total = 505) (h2 : friends = 14) :
  total % friends = 1 := by
  sorry

end NUMINAMATH_CALUDE_lollipops_left_over_l2320_232055


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l2320_232010

theorem consecutive_integers_sum (n : ℕ) : 
  n > 0 ∧ n * (n + 1) = 2800 → n + (n + 1) = 105 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l2320_232010


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l2320_232041

theorem sqrt_product_equality : Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l2320_232041


namespace NUMINAMATH_CALUDE_min_value_shifted_function_l2320_232013

noncomputable def f (x : ℝ) (c : ℝ) : ℝ := x^2 + 4*x + 5 - c

theorem min_value_shifted_function 
  (c : ℝ) 
  (h : ∃ (m : ℝ), ∀ (x : ℝ), f x c ≥ m ∧ ∃ (x₀ : ℝ), f x₀ c = m) 
  (h_min : ∃ (x₀ : ℝ), f x₀ c = 2) :
  ∃ (m : ℝ), (∀ (x : ℝ), f (x - 3) c ≥ m) ∧ (∃ (x₁ : ℝ), f (x₁ - 3) c = m) ∧ m = 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_shifted_function_l2320_232013


namespace NUMINAMATH_CALUDE_derivative_f_at_3_l2320_232062

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 3

theorem derivative_f_at_3 : 
  deriv f 3 = 1 / (3 * Real.log 3) := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_3_l2320_232062


namespace NUMINAMATH_CALUDE_quadratic_roots_close_existence_l2320_232068

theorem quadratic_roots_close_existence :
  ∃ (a b c : ℕ), a ≤ 2019 ∧ b ≤ 2019 ∧ c ≤ 2019 ∧
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧
  (a * x₁^2 + b * x₁ + c = 0) ∧
  (a * x₂^2 + b * x₂ + c = 0) ∧
  |x₁ - x₂| < 0.01 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_close_existence_l2320_232068


namespace NUMINAMATH_CALUDE_decreasing_power_function_m_values_l2320_232038

theorem decreasing_power_function_m_values (m : ℤ) :
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ → x₁^(m^2 - m - 2) > x₂^(m^2 - m - 2)) →
  m = 0 ∨ m = 1 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_power_function_m_values_l2320_232038


namespace NUMINAMATH_CALUDE_max_value_inequality_l2320_232033

theorem max_value_inequality (x y : ℝ) (hx : |x - 1| ≤ 1) (hy : |y - 2| ≤ 1) :
  |x - y + 1| ≤ 2 ∧ ∃ (x₀ y₀ : ℝ), |x₀ - 1| ≤ 1 ∧ |y₀ - 2| ≤ 1 ∧ |x₀ - y₀ + 1| = 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_inequality_l2320_232033


namespace NUMINAMATH_CALUDE_sum_of_roots_l2320_232028

theorem sum_of_roots (k m : ℝ) (x₁ x₂ : ℝ) 
  (h1 : x₁ ≠ x₂) 
  (h2 : 4 * x₁^2 - k * x₁ = m) 
  (h3 : 4 * x₂^2 - k * x₂ = m) : 
  x₁ + x₂ = k / 4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l2320_232028


namespace NUMINAMATH_CALUDE_angle_calculation_l2320_232061

/-- Two angles are complementary if their sum is 90 degrees -/
def complementary (a b : ℝ) : Prop := a + b = 90

/-- Two angles are supplementary if their sum is 180 degrees -/
def supplementary (a b : ℝ) : Prop := a + b = 180

/-- Given that angle1 and angle2 are complementary, angle2 and angle3 are supplementary, 
    and angle1 is 20 degrees, prove that angle3 is 110 degrees -/
theorem angle_calculation (angle1 angle2 angle3 : ℝ) 
    (h1 : complementary angle1 angle2)
    (h2 : supplementary angle2 angle3)
    (h3 : angle1 = 20) : 
  angle3 = 110 := by sorry

end NUMINAMATH_CALUDE_angle_calculation_l2320_232061


namespace NUMINAMATH_CALUDE_firewood_sacks_l2320_232043

theorem firewood_sacks (total_wood : ℕ) (wood_per_sack : ℕ) (h1 : total_wood = 80) (h2 : wood_per_sack = 20) :
  total_wood / wood_per_sack = 4 :=
by sorry

end NUMINAMATH_CALUDE_firewood_sacks_l2320_232043


namespace NUMINAMATH_CALUDE_expression_value_l2320_232015

theorem expression_value : (0.5 : ℝ)^3 - (0.1 : ℝ)^3 / (0.5 : ℝ)^2 + 0.05 + (0.1 : ℝ)^2 = 0.181 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2320_232015


namespace NUMINAMATH_CALUDE_hexagon_perimeter_is_24_l2320_232051

/-- Represents the side length of the larger equilateral triangle -/
def large_triangle_side : ℝ := 4

/-- Represents the side length of the regular hexagon -/
def hexagon_side : ℝ := large_triangle_side

/-- The number of sides in a regular hexagon -/
def hexagon_sides : ℕ := 6

/-- Calculates the perimeter of the regular hexagon -/
def hexagon_perimeter : ℝ := hexagon_side * hexagon_sides

theorem hexagon_perimeter_is_24 : hexagon_perimeter = 24 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_perimeter_is_24_l2320_232051


namespace NUMINAMATH_CALUDE_modified_morse_code_symbols_l2320_232019

/-- The number of distinct symbols for a given sequence length -/
def symbolCount (n : Nat) : Nat :=
  2^n

/-- The total number of distinct symbols for sequences of length 1 to 5 -/
def totalSymbols : Nat :=
  (symbolCount 1) + (symbolCount 2) + (symbolCount 3) + (symbolCount 4) + (symbolCount 5)

/-- Theorem: The total number of distinct symbols in modified Morse code for sequences
    of length 1 to 5 is 62 -/
theorem modified_morse_code_symbols :
  totalSymbols = 62 := by
  sorry

end NUMINAMATH_CALUDE_modified_morse_code_symbols_l2320_232019


namespace NUMINAMATH_CALUDE_no_solution_exists_l2320_232083

theorem no_solution_exists : ¬∃ (x : ℕ), (42 + x = 3 * (8 + x)) ∧ (42 + x = 2 * (10 + x)) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l2320_232083


namespace NUMINAMATH_CALUDE_hamburger_cost_theorem_l2320_232034

def total_hamburgers : ℕ := 50
def single_burger_cost : ℚ := 1
def double_burger_cost : ℚ := 1.5
def double_burgers : ℕ := 41

theorem hamburger_cost_theorem :
  let single_burgers := total_hamburgers - double_burgers
  let total_cost := (single_burgers : ℚ) * single_burger_cost + (double_burgers : ℚ) * double_burger_cost
  total_cost = 70.5 := by sorry

end NUMINAMATH_CALUDE_hamburger_cost_theorem_l2320_232034


namespace NUMINAMATH_CALUDE_ratio_equality_solution_l2320_232047

theorem ratio_equality_solution : 
  ∃! x : ℝ, (3 * x + 1) / (5 * x + 2) = (6 * x + 4) / (10 * x + 7) :=
by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_solution_l2320_232047


namespace NUMINAMATH_CALUDE_experiment_duration_in_seconds_l2320_232032

/-- Converts hours to seconds -/
def hoursToSeconds (hours : ℕ) : ℕ := hours * 3600

/-- Converts minutes to seconds -/
def minutesToSeconds (minutes : ℕ) : ℕ := minutes * 60

/-- Represents the duration of an experiment -/
structure ExperimentDuration where
  hours : ℕ
  minutes : ℕ
  seconds : ℕ

/-- Calculates the total seconds of an experiment duration -/
def totalSeconds (duration : ExperimentDuration) : ℕ :=
  hoursToSeconds duration.hours + minutesToSeconds duration.minutes + duration.seconds

/-- Theorem stating that the experiment lasting 2 hours, 45 minutes, and 30 seconds is equivalent to 9930 seconds -/
theorem experiment_duration_in_seconds :
  totalSeconds { hours := 2, minutes := 45, seconds := 30 } = 9930 := by
  sorry


end NUMINAMATH_CALUDE_experiment_duration_in_seconds_l2320_232032


namespace NUMINAMATH_CALUDE_factorial_sum_equals_4926_l2320_232027

theorem factorial_sum_equals_4926 : 6 * Nat.factorial 6 + 5 * Nat.factorial 5 + Nat.factorial 3 = 4926 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_equals_4926_l2320_232027


namespace NUMINAMATH_CALUDE_volleyball_team_combinations_l2320_232091

theorem volleyball_team_combinations : Nat.choose 16 7 = 11440 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_team_combinations_l2320_232091


namespace NUMINAMATH_CALUDE_Diamond_evaluation_l2320_232066

-- Define the Diamond operation
def Diamond (a b : ℝ) : ℝ := a * b^2 - b + 1

-- Theorem statement
theorem Diamond_evaluation : Diamond (Diamond 2 3) 4 = 253 := by
  sorry

end NUMINAMATH_CALUDE_Diamond_evaluation_l2320_232066


namespace NUMINAMATH_CALUDE_toy_pickup_time_l2320_232081

/-- The time required to put all toys in the box -/
def time_to_fill_box (total_toys : ℕ) (toys_in_per_cycle : ℕ) (toys_out_per_cycle : ℕ) (cycle_time : ℚ) : ℚ :=
  let net_toys_per_cycle := toys_in_per_cycle - toys_out_per_cycle
  let full_cycles := (total_toys - toys_in_per_cycle) / net_toys_per_cycle
  let full_cycles_time := full_cycles * cycle_time
  let final_cycle_time := cycle_time
  (full_cycles_time + final_cycle_time) / 60

/-- The problem statement -/
theorem toy_pickup_time :
  time_to_fill_box 50 4 3 (45 / 60) = 36.75 := by
  sorry

end NUMINAMATH_CALUDE_toy_pickup_time_l2320_232081


namespace NUMINAMATH_CALUDE_jimmy_cards_theorem_l2320_232085

def jimmy_cards_problem (initial_cards : ℕ) (cards_to_bob : ℕ) : Prop :=
  let cards_after_bob := initial_cards - cards_to_bob
  let cards_to_mary := 2 * cards_to_bob
  let final_cards := cards_after_bob - cards_to_mary
  initial_cards = 18 ∧ cards_to_bob = 3 → final_cards = 9

theorem jimmy_cards_theorem : jimmy_cards_problem 18 3 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_cards_theorem_l2320_232085


namespace NUMINAMATH_CALUDE_compute_expression_l2320_232021

theorem compute_expression : 18 * (200 / 3 + 50 / 6 + 16 / 18 + 2) = 1402 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l2320_232021


namespace NUMINAMATH_CALUDE_even_function_implies_a_equals_one_l2320_232054

/-- Given that f(x) = x³(a·2ˣ - 2⁻ˣ) is an even function, prove that a = 1 -/
theorem even_function_implies_a_equals_one (a : ℝ) :
  (∀ x : ℝ, x^3 * (a * 2^x - 2^(-x)) = (-x)^3 * (a * 2^(-x) - 2^x)) →
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_even_function_implies_a_equals_one_l2320_232054


namespace NUMINAMATH_CALUDE_train_passes_jogger_l2320_232009

/-- Prove that a train passes a jogger in 35 seconds given the specified conditions -/
theorem train_passes_jogger (jogger_speed : ℝ) (train_speed : ℝ) (train_length : ℝ) (initial_distance : ℝ) :
  jogger_speed = 9 * (1000 / 3600) →
  train_speed = 45 * (1000 / 3600) →
  train_length = 120 →
  initial_distance = 230 →
  (initial_distance + train_length) / (train_speed - jogger_speed) = 35 := by
  sorry

end NUMINAMATH_CALUDE_train_passes_jogger_l2320_232009


namespace NUMINAMATH_CALUDE_tv_tower_height_l2320_232020

/-- The height of a TV tower given specific angle measurements and distances -/
theorem tv_tower_height (angle_A : Real) (angle_B : Real) (angle_southwest : Real) (distance_AB : Real) :
  angle_A = π / 3 →  -- 60 degrees in radians
  angle_B = π / 4 →  -- 45 degrees in radians
  angle_southwest = π / 6 →  -- 30 degrees in radians
  distance_AB = 35 →
  ∃ (height : Real), height = 5 * Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_tv_tower_height_l2320_232020


namespace NUMINAMATH_CALUDE_isosceles_triangle_side_length_l2320_232074

/-- Given an equilateral triangle with side length 2 and three right-angled isosceles triangles
    constructed on its sides, if the total area of the three right-angled isosceles triangles
    equals the area of the equilateral triangle, then the length of the congruent sides of
    one right-angled isosceles triangle is √(6√3)/3. -/
theorem isosceles_triangle_side_length :
  let equilateral_side : ℝ := 2
  let equilateral_area : ℝ := Real.sqrt 3 / 4 * equilateral_side^2
  let isosceles_area : ℝ := equilateral_area / 3
  let isosceles_side : ℝ := Real.sqrt (2 * isosceles_area)
  isosceles_side = Real.sqrt (6 * Real.sqrt 3) / 3 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_side_length_l2320_232074


namespace NUMINAMATH_CALUDE_total_amount_after_three_years_l2320_232059

/-- Calculates the compound interest for a given principal, rate, and time --/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- The original bill amount --/
def initial_amount : ℝ := 350

/-- The interest rate for the first year --/
def first_year_rate : ℝ := 0.03

/-- The interest rate for the second and third years --/
def later_years_rate : ℝ := 0.05

/-- The total time period in years --/
def total_years : ℕ := 3

theorem total_amount_after_three_years :
  let amount_after_first_year := compound_interest initial_amount first_year_rate 1
  let final_amount := compound_interest amount_after_first_year later_years_rate 2
  ∃ ε > 0, |final_amount - 397.45| < ε :=
sorry

end NUMINAMATH_CALUDE_total_amount_after_three_years_l2320_232059


namespace NUMINAMATH_CALUDE_multiply_subtract_distribute_compute_expression_l2320_232095

theorem multiply_subtract_distribute (a b c : ℕ) : a * c - b * c = (a - b) * c := by sorry

theorem compute_expression : 65 * 1313 - 25 * 1313 = 52520 := by sorry

end NUMINAMATH_CALUDE_multiply_subtract_distribute_compute_expression_l2320_232095


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2320_232088

-- Define the function f(x) = x^2 + 2x - 3
def f (x : ℝ) : ℝ := x^2 + 2*x - 3

-- State the theorem
theorem inequality_solution_set :
  {x : ℝ | f x < 0} = {x : ℝ | -3 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2320_232088


namespace NUMINAMATH_CALUDE_quadratic_roots_l2320_232082

theorem quadratic_roots (a b c : ℝ) (h : b^2 - 4*a*c > 0) :
  (2*(a + b))^2 - 4*3*a*(b + c) > 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_l2320_232082


namespace NUMINAMATH_CALUDE_rose_cost_l2320_232039

/-- The cost of each red rose, given the conditions of Jezebel's flower purchase. -/
theorem rose_cost (num_roses : ℕ) (num_sunflowers : ℕ) (sunflower_cost : ℚ) (total_cost : ℚ) :
  num_roses = 24 →
  num_sunflowers = 3 →
  sunflower_cost = 3 →
  total_cost = 45 →
  (total_cost - num_sunflowers * sunflower_cost) / num_roses = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_rose_cost_l2320_232039


namespace NUMINAMATH_CALUDE_world_expo_allocation_schemes_l2320_232003

theorem world_expo_allocation_schemes :
  let n_volunteers : ℕ := 6
  let n_pavilions : ℕ := 4
  let n_groups_of_two : ℕ := 2
  let n_groups_of_one : ℕ := 2

  -- Number of ways to choose 2 groups of 2 people from 6 volunteers
  let ways_to_choose_groups_of_two : ℕ := Nat.choose n_volunteers 2 * Nat.choose (n_volunteers - 2) 2

  -- Number of ways to allocate the groups to pavilions
  let ways_to_allocate_pavilions : ℕ := Nat.factorial n_pavilions

  -- Total number of allocation schemes
  ways_to_choose_groups_of_two * Nat.choose n_pavilions n_groups_of_two * ways_to_allocate_pavilions = 1080 :=
by sorry

end NUMINAMATH_CALUDE_world_expo_allocation_schemes_l2320_232003


namespace NUMINAMATH_CALUDE_two_digit_number_property_l2320_232089

/-- 
For a two-digit number n where the unit's digit exceeds the 10's digit by 2, 
and n = 24, the product of n and the sum of its digits is 144.
-/
theorem two_digit_number_property : 
  ∀ (a b : ℕ), 
    (10 * a + b = 24) → 
    (b = a + 2) → 
    24 * (a + b) = 144 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_property_l2320_232089


namespace NUMINAMATH_CALUDE_zero_function_theorem_l2320_232014

-- Define the function type
def NonNegativeFunction := { f : ℝ → ℝ // ∀ x ≥ 0, f x ≥ 0 }

-- State the theorem
theorem zero_function_theorem (f : NonNegativeFunction) 
  (h_diff : Differentiable ℝ (fun x => f.val x))
  (h_initial : f.val 0 = 0)
  (h_deriv : ∀ x ≥ 0, (deriv f.val) (x^2) = f.val x) :
  ∀ x ≥ 0, f.val x = 0 := by
sorry

end NUMINAMATH_CALUDE_zero_function_theorem_l2320_232014


namespace NUMINAMATH_CALUDE_expanded_polynomial_has_four_nonzero_terms_l2320_232048

/-- The polynomial obtained from expanding (x+5)(3x^2+2x+4)-4(x^3-x^2+3x) -/
def expanded_polynomial (x : ℝ) : ℝ := -x^3 + 21*x^2 + 2*x + 20

/-- The number of nonzero terms in the expanded polynomial -/
def nonzero_term_count : ℕ := 4

theorem expanded_polynomial_has_four_nonzero_terms :
  (∃ a b c d : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
    ∀ x : ℝ, expanded_polynomial x = a*x^3 + b*x^2 + c*x + d) ∧
  (∀ a b c d e : ℝ, ∀ i j k l m : ℕ,
    (i ≠ j ∨ a = 0) ∧ (i ≠ k ∨ a = 0) ∧ (i ≠ l ∨ a = 0) ∧ (i ≠ m ∨ a = 0) ∧
    (j ≠ k ∨ b = 0) ∧ (j ≠ l ∨ b = 0) ∧ (j ≠ m ∨ b = 0) ∧
    (k ≠ l ∨ c = 0) ∧ (k ≠ m ∨ c = 0) ∧
    (l ≠ m ∨ d = 0) →
    (∀ x : ℝ, expanded_polynomial x = a*x^i + b*x^j + c*x^k + d*x^l + e*x^m) →
    e = 0) :=
by sorry

#check expanded_polynomial_has_four_nonzero_terms

end NUMINAMATH_CALUDE_expanded_polynomial_has_four_nonzero_terms_l2320_232048


namespace NUMINAMATH_CALUDE_appended_number_theorem_l2320_232099

theorem appended_number_theorem (a x : ℕ) (ha : 0 < a) (hx : x ≤ 9) :
  (10 * a + x - a^2 = (11 - x) * a) ↔ (x = a) := by
sorry

end NUMINAMATH_CALUDE_appended_number_theorem_l2320_232099


namespace NUMINAMATH_CALUDE_function_inequality_l2320_232026

open Set

theorem function_inequality (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x, x > 0 → f x ≥ 0) →
  (∀ x, x > 0 → HasDerivAt f (f x) x) →
  (∀ x, x > 0 → x * (deriv f x) + f x < 0) →
  0 < a → 0 < b → a < b →
  b * f b < a * f a := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l2320_232026


namespace NUMINAMATH_CALUDE_unit_digit_of_sum_factorials_100_l2320_232000

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem unit_digit_of_sum_factorials_100 :
  sum_factorials 100 % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_unit_digit_of_sum_factorials_100_l2320_232000


namespace NUMINAMATH_CALUDE_expansion_coefficient_l2320_232011

/-- The coefficient of x^5 in the expansion of (ax^2 + 1/√x)^5 -/
def coefficient_x5 (a : ℝ) : ℝ := 10 * a^3

theorem expansion_coefficient (a : ℝ) : 
  coefficient_x5 a = -80 → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_expansion_coefficient_l2320_232011


namespace NUMINAMATH_CALUDE_movie_ticket_distribution_l2320_232086

theorem movie_ticket_distribution (n : ℕ) (k : ℕ) (h1 : n = 10) (h2 : k = 3) :
  (n.descFactorial k) = 720 := by
  sorry

end NUMINAMATH_CALUDE_movie_ticket_distribution_l2320_232086


namespace NUMINAMATH_CALUDE_min_value_a_l2320_232079

theorem min_value_a (a : ℝ) (h : a > 0) : 
  (∀ x y : ℝ, x > 0 → y > 0 → (x + y) * (1/x + a/y) ≥ 9) → a ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_l2320_232079


namespace NUMINAMATH_CALUDE_university_volunteer_selection_l2320_232056

theorem university_volunteer_selection (undergrad : ℕ) (masters : ℕ) (doctoral : ℕ) 
  (selected_doctoral : ℕ) (h1 : undergrad = 4400) (h2 : masters = 400) (h3 : doctoral = 200) 
  (h4 : selected_doctoral = 10) :
  (undergrad + masters + doctoral) * selected_doctoral / doctoral = 250 := by
  sorry

end NUMINAMATH_CALUDE_university_volunteer_selection_l2320_232056


namespace NUMINAMATH_CALUDE_equation_solution_l2320_232071

theorem equation_solution : 
  {x : ℝ | (12 - 3*x)^2 = x^2} = {3, 6} := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2320_232071


namespace NUMINAMATH_CALUDE_sin_cos_identity_l2320_232035

theorem sin_cos_identity (α : ℝ) : 
  Real.sin α ^ 6 + Real.cos α ^ 6 + 3 * (Real.sin α ^ 2) * (Real.cos α ^ 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l2320_232035


namespace NUMINAMATH_CALUDE_park_fencing_cost_l2320_232057

/-- The cost of fencing a rectangular park -/
theorem park_fencing_cost 
  (length width : ℝ) 
  (area : ℝ) 
  (fencing_cost_paise : ℝ) : 
  length / width = 3 / 2 →
  area = length * width →
  area = 2400 →
  fencing_cost_paise = 50 →
  2 * (length + width) * (fencing_cost_paise / 100) = 100 := by
  sorry


end NUMINAMATH_CALUDE_park_fencing_cost_l2320_232057


namespace NUMINAMATH_CALUDE_other_endpoint_coordinates_l2320_232093

/-- Given a line segment with midpoint (3, 1) and one endpoint at (7, -3),
    prove that the other endpoint is at (-1, 5). -/
theorem other_endpoint_coordinates :
  ∀ (x y : ℝ),
  (3 = (7 + x) / 2) →
  (1 = (-3 + y) / 2) →
  x = -1 ∧ y = 5 := by
sorry

end NUMINAMATH_CALUDE_other_endpoint_coordinates_l2320_232093


namespace NUMINAMATH_CALUDE_deck_size_l2320_232023

theorem deck_size (r b u : ℕ) : 
  r + b + u > 0 →
  r / (r + b + u : ℚ) = 1 / 5 →
  r / ((r + b + u + 3) : ℚ) = 1 / 6 →
  r + b + u = 15 := by
sorry

end NUMINAMATH_CALUDE_deck_size_l2320_232023


namespace NUMINAMATH_CALUDE_matrix_equality_l2320_232096

theorem matrix_equality (A B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h1 : A + B = A * B) 
  (h2 : A * B = ![![10, 6], ![-4, 2]]) : 
  B * A = ![![10, 6], ![-4, 2]] := by
  sorry

end NUMINAMATH_CALUDE_matrix_equality_l2320_232096


namespace NUMINAMATH_CALUDE_domain_of_linear_function_domain_of_rational_function_domain_of_square_root_function_domain_of_reciprocal_square_root_function_domain_of_rational_function_with_linear_denominator_domain_of_arcsin_function_l2320_232052

-- Function 1: z = 4 - x - 2y
theorem domain_of_linear_function (x y : ℝ) :
  ∃ z : ℝ, z = 4 - x - 2*y :=
sorry

-- Function 2: p = 3 / (x^2 + y^2)
theorem domain_of_rational_function (x y : ℝ) :
  (x ≠ 0 ∨ y ≠ 0) → ∃ p : ℝ, p = 3 / (x^2 + y^2) :=
sorry

-- Function 3: z = √(1 - x^2 - y^2)
theorem domain_of_square_root_function (x y : ℝ) :
  x^2 + y^2 ≤ 1 → ∃ z : ℝ, z = Real.sqrt (1 - x^2 - y^2) :=
sorry

-- Function 4: q = 1 / √(xy)
theorem domain_of_reciprocal_square_root_function (x y : ℝ) :
  (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0) → ∃ q : ℝ, q = 1 / Real.sqrt (x*y) :=
sorry

-- Function 5: u = (x^2 * y) / (2x + 1 - y)
theorem domain_of_rational_function_with_linear_denominator (x y : ℝ) :
  2*x + 1 - y ≠ 0 → ∃ u : ℝ, u = (x^2 * y) / (2*x + 1 - y) :=
sorry

-- Function 6: v = arcsin(x + y)
theorem domain_of_arcsin_function (x y : ℝ) :
  -1 ≤ x + y ∧ x + y ≤ 1 → ∃ v : ℝ, v = Real.arcsin (x + y) :=
sorry

end NUMINAMATH_CALUDE_domain_of_linear_function_domain_of_rational_function_domain_of_square_root_function_domain_of_reciprocal_square_root_function_domain_of_rational_function_with_linear_denominator_domain_of_arcsin_function_l2320_232052


namespace NUMINAMATH_CALUDE_impossible_friendship_configuration_l2320_232049

/-- A graph representing friendships in a class -/
structure FriendshipGraph where
  vertices : Finset Nat
  edges : Finset (Nat × Nat)
  sym : ∀ {a b}, (a, b) ∈ edges → (b, a) ∈ edges
  irrefl : ∀ a, (a, a) ∉ edges

/-- The degree of a vertex in the graph -/
def degree (G : FriendshipGraph) (v : Nat) : Nat :=
  (G.edges.filter (λ e => e.1 = v ∨ e.2 = v)).card

/-- Theorem: It's impossible to have a friendship graph with 30 students where
    9 have 3 friends, 11 have 4 friends, and 10 have 5 friends -/
theorem impossible_friendship_configuration (G : FriendshipGraph) :
  G.vertices.card = 30 →
  (∃ S₁ S₂ S₃ : Finset Nat,
    S₁.card = 9 ∧ S₂.card = 11 ∧ S₃.card = 10 ∧
    S₁ ∪ S₂ ∪ S₃ = G.vertices ∧
    (∀ v ∈ S₁, degree G v = 3) ∧
    (∀ v ∈ S₂, degree G v = 4) ∧
    (∀ v ∈ S₃, degree G v = 5)) →
  False := by
  sorry

end NUMINAMATH_CALUDE_impossible_friendship_configuration_l2320_232049


namespace NUMINAMATH_CALUDE_ed_doug_marble_difference_l2320_232040

theorem ed_doug_marble_difference :
  ∀ (ed_initial : ℕ) (ed_lost : ℕ) (ed_current : ℕ) (doug : ℕ),
    ed_initial > doug →
    ed_lost = 20 →
    ed_current = 17 →
    doug = 5 →
    ed_initial = ed_current + ed_lost →
    ed_initial - doug = 32 := by
  sorry

end NUMINAMATH_CALUDE_ed_doug_marble_difference_l2320_232040


namespace NUMINAMATH_CALUDE_power_multiplication_calculate_power_l2320_232067

theorem power_multiplication (a : ℕ) (m n : ℕ) : a * (a ^ n) = a ^ (n + 1) := by sorry

theorem calculate_power : 5000 * (5000 ^ 3000) = 5000 ^ 3001 := by sorry

end NUMINAMATH_CALUDE_power_multiplication_calculate_power_l2320_232067


namespace NUMINAMATH_CALUDE_sophia_lost_pawns_l2320_232002

theorem sophia_lost_pawns (initial_pawns : ℕ) (chloe_lost : ℕ) (pawns_left : ℕ) : 
  initial_pawns = 8 → chloe_lost = 1 → pawns_left = 10 → 
  initial_pawns - (pawns_left - (initial_pawns - chloe_lost)) = 5 := by
sorry

end NUMINAMATH_CALUDE_sophia_lost_pawns_l2320_232002


namespace NUMINAMATH_CALUDE_product_of_repeating_decimal_and_five_l2320_232053

-- Define the repeating decimal 0.456̄
def repeating_decimal : ℚ := 456 / 999

-- State the theorem
theorem product_of_repeating_decimal_and_five :
  repeating_decimal * 5 = 760 / 333 := by
  sorry

end NUMINAMATH_CALUDE_product_of_repeating_decimal_and_five_l2320_232053


namespace NUMINAMATH_CALUDE_edward_good_games_l2320_232094

def games_from_friend : ℕ := 41
def games_from_garage_sale : ℕ := 14
def non_working_games : ℕ := 31

theorem edward_good_games :
  games_from_friend + games_from_garage_sale - non_working_games = 24 := by
  sorry

end NUMINAMATH_CALUDE_edward_good_games_l2320_232094


namespace NUMINAMATH_CALUDE_person_a_speed_l2320_232022

theorem person_a_speed (v_a v_b : ℝ) : 
  v_a > v_b →
  8 * (v_a + v_b) = 6 * (v_a + v_b + 4) →
  6 * ((v_a + 2) - (v_b + 2)) = 6 →
  v_a = 6.5 := by
sorry

end NUMINAMATH_CALUDE_person_a_speed_l2320_232022


namespace NUMINAMATH_CALUDE_composition_equation_solution_l2320_232045

theorem composition_equation_solution (f g : ℝ → ℝ) (a : ℝ) 
  (hf : ∀ x, f x = x / 3 + 2)
  (hg : ∀ x, g x = 5 - 2 * x)
  (h : f (g a) = 4) :
  a = -1/2 := by sorry

end NUMINAMATH_CALUDE_composition_equation_solution_l2320_232045
