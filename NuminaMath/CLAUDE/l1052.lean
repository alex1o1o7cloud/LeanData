import Mathlib

namespace NUMINAMATH_CALUDE_probability_zero_l1052_105261

def P (x : ℝ) : ℝ := x^3 - 4*x^2 - x + 20

theorem probability_zero :
  ∀ x : ℝ, 3 ≤ x ∧ x ≤ 10 →
  (⌊(P x)^(1/3)⌋ : ℝ) ≠ (P ⌊x⌋)^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_probability_zero_l1052_105261


namespace NUMINAMATH_CALUDE_complex_modulus_example_l1052_105297

theorem complex_modulus_example : 
  let z : ℂ := 1 - (5/4)*I
  Complex.abs z = Real.sqrt 41 / 4 := by sorry

end NUMINAMATH_CALUDE_complex_modulus_example_l1052_105297


namespace NUMINAMATH_CALUDE_sqrt_twelve_minus_neg_one_power_zero_plus_abs_sqrt_three_minus_one_l1052_105291

theorem sqrt_twelve_minus_neg_one_power_zero_plus_abs_sqrt_three_minus_one :
  Real.sqrt 12 - (-1)^(0 : ℕ) + |Real.sqrt 3 - 1| = 3 * Real.sqrt 3 - 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_twelve_minus_neg_one_power_zero_plus_abs_sqrt_three_minus_one_l1052_105291


namespace NUMINAMATH_CALUDE_saplings_count_l1052_105292

theorem saplings_count (total_trees : ℕ) (ancient_oaks : ℕ) (fir_trees : ℕ) : 
  total_trees = 96 → ancient_oaks = 15 → fir_trees = 23 → 
  total_trees - (ancient_oaks + fir_trees) = 58 :=
by sorry

end NUMINAMATH_CALUDE_saplings_count_l1052_105292


namespace NUMINAMATH_CALUDE_water_hyacinth_demonstrates_interconnection_and_diversity_l1052_105287

/-- Represents the introduction and effects of water hyacinth -/
structure WaterHyacinthIntroduction where
  introduced_as_fodder : Prop
  rapid_spread : Prop
  decrease_native_species : Prop
  water_pollution : Prop
  increase_mosquitoes : Prop

/-- Represents the philosophical conclusions drawn from the water hyacinth case -/
structure PhilosophicalConclusions where
  universal_interconnection : Prop
  diverse_connections : Prop

/-- Theorem stating that the introduction of water hyacinth demonstrates universal interconnection and diverse connections -/
theorem water_hyacinth_demonstrates_interconnection_and_diversity 
  (wh : WaterHyacinthIntroduction) : PhilosophicalConclusions :=
by sorry

end NUMINAMATH_CALUDE_water_hyacinth_demonstrates_interconnection_and_diversity_l1052_105287


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l1052_105244

theorem quadratic_two_distinct_roots (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 + 3*x + k = 0 ∧ y^2 + 3*y + k = 0) ↔ k < 9/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l1052_105244


namespace NUMINAMATH_CALUDE_two_sin_45_equals_sqrt_2_l1052_105205

theorem two_sin_45_equals_sqrt_2 (α : Real) (h : α = Real.pi / 4) : 
  2 * Real.sin α = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_two_sin_45_equals_sqrt_2_l1052_105205


namespace NUMINAMATH_CALUDE_points_difference_l1052_105276

def basketball_game (jon_points jack_points tom_points : ℕ) : Prop :=
  (jack_points = jon_points + 5) ∧
  (jon_points + jack_points + tom_points = 18) ∧
  (tom_points < jon_points + jack_points)

theorem points_difference (jon_points jack_points tom_points : ℕ) :
  basketball_game jon_points jack_points tom_points →
  jon_points = 3 →
  (jon_points + jack_points) - tom_points = 4 := by
  sorry

end NUMINAMATH_CALUDE_points_difference_l1052_105276


namespace NUMINAMATH_CALUDE_roots_in_specific_intervals_roots_in_unit_interval_l1052_105281

-- Define the quadratic function
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + 2*m*x + 2*m + 1

-- Part I
theorem roots_in_specific_intervals (m : ℝ) :
  (∃ x₁ x₂, x₁ ∈ Set.Ioo (-1 : ℝ) 0 ∧ 
             x₂ ∈ Set.Ioo 1 2 ∧ 
             f m x₁ = 0 ∧ 
             f m x₂ = 0) →
  m ∈ Set.Ioo (-5/6 : ℝ) (-1/2) :=
sorry

-- Part II
theorem roots_in_unit_interval (m : ℝ) :
  (∃ x₁ x₂, x₁ ∈ Set.Ioo 0 1 ∧ 
             x₂ ∈ Set.Ioo 0 1 ∧ 
             f m x₁ = 0 ∧ 
             f m x₂ = 0) →
  m ∈ Set.Ico (-1/2 : ℝ) (1 - Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_roots_in_specific_intervals_roots_in_unit_interval_l1052_105281


namespace NUMINAMATH_CALUDE_weight_loss_difference_equals_303_l1052_105209

/-- Calculates the total weight loss difference between Luca and Kim combined, and Barbi -/
def weight_loss_difference : ℝ :=
  let barbi_monthly_loss : ℝ := 1.5
  let barbi_months : ℕ := 2 * 12
  let luca_yearly_loss : ℝ := 9
  let luca_years : ℕ := 15
  let kim_first_year_monthly_loss : ℝ := 2
  let kim_remaining_monthly_loss : ℝ := 3
  let kim_remaining_months : ℕ := 5 * 12

  let barbi_total_loss := barbi_monthly_loss * barbi_months
  let luca_total_loss := luca_yearly_loss * luca_years
  let kim_first_year_loss := kim_first_year_monthly_loss * 12
  let kim_remaining_loss := kim_remaining_monthly_loss * kim_remaining_months
  let kim_total_loss := kim_first_year_loss + kim_remaining_loss

  (luca_total_loss + kim_total_loss) - barbi_total_loss

theorem weight_loss_difference_equals_303 : weight_loss_difference = 303 := by
  sorry

end NUMINAMATH_CALUDE_weight_loss_difference_equals_303_l1052_105209


namespace NUMINAMATH_CALUDE_sculpture_third_week_cut_percentage_l1052_105234

/-- Calculates the percentage of marble cut away in the third week of sculpting. -/
theorem sculpture_third_week_cut_percentage
  (initial_weight : ℝ)
  (first_week_cut : ℝ)
  (second_week_cut : ℝ)
  (final_weight : ℝ)
  (h1 : initial_weight = 190)
  (h2 : first_week_cut = 0.25)
  (h3 : second_week_cut = 0.15)
  (h4 : final_weight = 109.0125) :
  let weight_after_first_week := initial_weight * (1 - first_week_cut)
  let weight_after_second_week := weight_after_first_week * (1 - second_week_cut)
  let third_week_cut_percentage := 1 - (final_weight / weight_after_second_week)
  ∃ ε > 0, |third_week_cut_percentage - 0.0999| < ε :=
by sorry

end NUMINAMATH_CALUDE_sculpture_third_week_cut_percentage_l1052_105234


namespace NUMINAMATH_CALUDE_remainder_b91_mod_50_l1052_105258

theorem remainder_b91_mod_50 : ∃ k : ℤ, 7^91 + 9^91 = 50 * k + 16 := by sorry

end NUMINAMATH_CALUDE_remainder_b91_mod_50_l1052_105258


namespace NUMINAMATH_CALUDE_rachels_homework_l1052_105271

/-- Rachel's homework problem -/
theorem rachels_homework (reading_pages : ℕ) (math_pages : ℕ) : 
  reading_pages = 2 → math_pages = reading_pages + 3 → math_pages = 5 :=
by sorry

end NUMINAMATH_CALUDE_rachels_homework_l1052_105271


namespace NUMINAMATH_CALUDE_binomial_coefficient_equation_l1052_105265

theorem binomial_coefficient_equation (x : ℕ) : 
  (Nat.choose 10 x = Nat.choose 8 (x - 2) + Nat.choose 8 (x - 1) + Nat.choose 9 (2 * x - 3)) → 
  (x = 3 ∨ x = 4) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equation_l1052_105265


namespace NUMINAMATH_CALUDE_lemonade_mixture_l1052_105298

theorem lemonade_mixture (L : ℝ) : 
  -- First solution composition
  let first_lemonade : ℝ := 20
  let first_carbonated : ℝ := 80
  -- Second solution composition
  let second_lemonade : ℝ := L
  let second_carbonated : ℝ := 55
  -- Mixture composition
  let mixture_carbonated : ℝ := 60
  let mixture_first_solution : ℝ := 20
  -- Conditions
  first_lemonade + first_carbonated = 100 →
  second_lemonade + second_carbonated = 100 →
  mixture_first_solution * first_carbonated / 100 + 
    (100 - mixture_first_solution) * second_carbonated / 100 = mixture_carbonated →
  -- Conclusion
  L = 45 := by
sorry

end NUMINAMATH_CALUDE_lemonade_mixture_l1052_105298


namespace NUMINAMATH_CALUDE_min_n_for_sqrt_12n_integer_l1052_105250

theorem min_n_for_sqrt_12n_integer (n : ℕ) (h1 : n > 0) (h2 : ∃ k : ℕ, k > 0 ∧ k^2 = 12*n) :
  ∀ m : ℕ, m > 0 → (∃ j : ℕ, j > 0 ∧ j^2 = 12*m) → m ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_min_n_for_sqrt_12n_integer_l1052_105250


namespace NUMINAMATH_CALUDE_inequality_proof_l1052_105256

theorem inequality_proof (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (hmax : d = max a (max b c)) : 
  a * (d - b) + b * (d - c) + c * (d - a) ≤ d^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1052_105256


namespace NUMINAMATH_CALUDE_total_seeds_planted_l1052_105225

theorem total_seeds_planted (num_flowerbeds : ℕ) (seeds_per_bed : ℕ) (total_seeds : ℕ) :
  num_flowerbeds = 9 →
  seeds_per_bed = 5 →
  total_seeds = num_flowerbeds * seeds_per_bed →
  total_seeds = 45 := by
  sorry

end NUMINAMATH_CALUDE_total_seeds_planted_l1052_105225


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1052_105243

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 8 → b = 15 → c^2 = a^2 + b^2 → c = 17 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1052_105243


namespace NUMINAMATH_CALUDE_cookies_left_l1052_105229

/-- The number of cookies in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens John bought -/
def dozens_bought : ℕ := 2

/-- The number of cookies John ate -/
def cookies_eaten : ℕ := 3

/-- Theorem: John has 21 cookies left -/
theorem cookies_left : dozens_bought * dozen - cookies_eaten = 21 := by
  sorry

end NUMINAMATH_CALUDE_cookies_left_l1052_105229


namespace NUMINAMATH_CALUDE_regular_polygon_with_150_degree_angles_has_12_sides_l1052_105273

/-- A regular polygon with interior angles of 150 degrees has 12 sides -/
theorem regular_polygon_with_150_degree_angles_has_12_sides :
  ∀ n : ℕ,
  n > 2 →
  (∀ angle : ℝ, angle = 150) →
  (180 * (n - 2) : ℝ) = (150 * n : ℝ) →
  n = 12 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_with_150_degree_angles_has_12_sides_l1052_105273


namespace NUMINAMATH_CALUDE_sum_of_squares_coefficients_l1052_105239

/-- The sum of squares of coefficients in the simplified form of 6(x³-2x²+x-3)-5(x⁴-4x²+3x+2) is 990 -/
theorem sum_of_squares_coefficients : 
  let expression := fun x : ℝ => 6 * (x^3 - 2*x^2 + x - 3) - 5 * (x^4 - 4*x^2 + 3*x + 2)
  let coefficients := [-5, 6, 8, -9, -28]
  (coefficients.map (fun c => c^2)).sum = 990 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_coefficients_l1052_105239


namespace NUMINAMATH_CALUDE_wednesday_rainfall_l1052_105218

/-- Rainfall recorded over three days -/
def total_rainfall : ℝ := 0.67

/-- Rainfall recorded on Monday -/
def monday_rainfall : ℝ := 0.17

/-- Rainfall recorded on Tuesday -/
def tuesday_rainfall : ℝ := 0.42

/-- Theorem stating that the rainfall on Wednesday is 0.08 cm -/
theorem wednesday_rainfall : 
  total_rainfall - (monday_rainfall + tuesday_rainfall) = 0.08 := by
  sorry

end NUMINAMATH_CALUDE_wednesday_rainfall_l1052_105218


namespace NUMINAMATH_CALUDE_sum_of_roots_equals_one_l1052_105255

theorem sum_of_roots_equals_one :
  let f : ℝ → ℝ := λ x => (x + 3) * (x - 4) - 21
  ∃ x₁ x₂ : ℝ, (f x₁ = 0 ∧ f x₂ = 0) ∧ x₁ + x₂ = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_equals_one_l1052_105255


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1052_105299

/-- Given an arithmetic sequence {1/aₙ} where a₁ = 1 and a₄ = 4, prove that a₁₀ = -4/5 -/
theorem arithmetic_sequence_problem (a : ℕ → ℚ) :
  (∃ d : ℚ, ∀ n : ℕ, 1 / a (n + 1) - 1 / a n = d) →
  a 1 = 1 →
  a 4 = 4 →
  a 10 = -4/5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1052_105299


namespace NUMINAMATH_CALUDE_five_lines_sixteen_sections_l1052_105215

/-- The number of sections created by drawing n properly intersecting line segments in a rectangle --/
def sections (n : ℕ) : ℕ :=
  if n = 0 then 1 else sections (n - 1) + n

/-- Theorem: Drawing 5 properly intersecting line segments in a rectangle creates 16 sections --/
theorem five_lines_sixteen_sections : sections 5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_five_lines_sixteen_sections_l1052_105215


namespace NUMINAMATH_CALUDE_sum_of_factors_l1052_105294

theorem sum_of_factors (a b c : ℤ) : 
  (∀ x, x^2 + 20*x + 96 = (x + a) * (x + b)) →
  (∀ x, x^2 + 18*x + 81 = (x - b) * (x + c)) →
  a + b + c = 30 := by
sorry

end NUMINAMATH_CALUDE_sum_of_factors_l1052_105294


namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l1052_105278

/-- Given a circle with equation x^2 - 16x + y^2 + 6y = -75, 
    prove that the sum of its center coordinates and radius is 5 + √2 -/
theorem circle_center_radius_sum :
  ∃ (a b r : ℝ), 
    (∀ x y : ℝ, x^2 - 16*x + y^2 + 6*y = -75 ↔ (x - a)^2 + (y - b)^2 = r^2) ∧
    a + b + r = 5 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_radius_sum_l1052_105278


namespace NUMINAMATH_CALUDE_monotonicity_not_algorithmic_l1052_105267

-- Define the concept of an algorithm
def Algorithm : Type := Unit

-- Define the problems
def SumProblem : Type := Unit
def LinearSystemProblem : Type := Unit
def CircleAreaProblem : Type := Unit
def MonotonicityProblem : Type := Unit

-- Define solvability by algorithm
def SolvableByAlgorithm (p : Type) : Prop := ∃ (a : Algorithm), True

-- State the theorem
theorem monotonicity_not_algorithmic :
  SolvableByAlgorithm SumProblem ∧
  SolvableByAlgorithm LinearSystemProblem ∧
  SolvableByAlgorithm CircleAreaProblem ∧
  ¬SolvableByAlgorithm MonotonicityProblem :=
sorry

end NUMINAMATH_CALUDE_monotonicity_not_algorithmic_l1052_105267


namespace NUMINAMATH_CALUDE_roger_used_crayons_l1052_105277

/-- The number of used crayons Roger had -/
def used_crayons : ℕ := 14 - 2 - 8

/-- The total number of crayons Roger had -/
def total_crayons : ℕ := 14

/-- The number of new crayons Roger had -/
def new_crayons : ℕ := 2

/-- The number of broken crayons Roger had -/
def broken_crayons : ℕ := 8

theorem roger_used_crayons : 
  used_crayons + new_crayons + broken_crayons = total_crayons ∧ used_crayons = 4 := by
  sorry

end NUMINAMATH_CALUDE_roger_used_crayons_l1052_105277


namespace NUMINAMATH_CALUDE_mary_sold_at_least_12_boxes_l1052_105201

/-- The number of cases Mary needs to deliver -/
def cases : ℕ := 2

/-- The number of boxes in each case -/
def boxes_per_case : ℕ := 6

/-- The minimum number of boxes Mary sold -/
def min_boxes_sold : ℕ := cases * boxes_per_case

/-- Mary has some extra boxes (number unspecified) -/
axiom has_extra_boxes : ∃ n : ℕ, n > 0

theorem mary_sold_at_least_12_boxes :
  min_boxes_sold ≥ 12 ∧ ∃ total : ℕ, total > min_boxes_sold :=
sorry

end NUMINAMATH_CALUDE_mary_sold_at_least_12_boxes_l1052_105201


namespace NUMINAMATH_CALUDE_max_sum_xy_l1052_105285

def max_value_xy (x y : ℝ) : Prop :=
  (Real.log y / Real.log ((x^2 + y^2) / 2) ≥ 1) ∧
  ((x, y) ≠ (0, 0)) ∧
  (x^2 + y^2 ≠ 2)

theorem max_sum_xy :
  ∀ x y : ℝ, max_value_xy x y → x + y ≤ 1 + Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_xy_l1052_105285


namespace NUMINAMATH_CALUDE_triangle_inequality_theorem_not_necessary_condition_l1052_105228

/-- Proposition P: segments of lengths a, b, c can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- Proposition Q: a² + b² + c² < 2(ab + bc + ca) -/
def inequality_holds (a b c : ℝ) : Prop :=
  a^2 + b^2 + c^2 < 2*(a*b + b*c + c*a)

theorem triangle_inequality_theorem (a b c : ℝ) :
  can_form_triangle a b c → inequality_holds a b c :=
sorry

theorem not_necessary_condition :
  ∃ a b c : ℝ, inequality_holds a b c ∧ ¬can_form_triangle a b c :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_theorem_not_necessary_condition_l1052_105228


namespace NUMINAMATH_CALUDE_factorization_x4_minus_9_factorization_quadratic_in_a_and_b_l1052_105253

-- Problem 1
theorem factorization_x4_minus_9 (x : ℝ) : 
  x^4 - 9 = (x^2 + 3) * (x + Real.sqrt 3) * (x - Real.sqrt 3) := by sorry

-- Problem 2
theorem factorization_quadratic_in_a_and_b (a b : ℝ) :
  -a^2*b + 2*a*b - b = -b*(a-1)^2 := by sorry

end NUMINAMATH_CALUDE_factorization_x4_minus_9_factorization_quadratic_in_a_and_b_l1052_105253


namespace NUMINAMATH_CALUDE_exam_score_l1052_105269

theorem exam_score (total_questions : ℕ) (correct_answers : ℕ) 
  (marks_per_correct : ℕ) (marks_lost_per_wrong : ℕ) : 
  total_questions = 80 → 
  correct_answers = 40 → 
  marks_per_correct = 4 → 
  marks_lost_per_wrong = 1 → 
  (correct_answers * marks_per_correct) - 
    ((total_questions - correct_answers) * marks_lost_per_wrong) = 120 := by
  sorry

#check exam_score

end NUMINAMATH_CALUDE_exam_score_l1052_105269


namespace NUMINAMATH_CALUDE_factorial_divisibility_l1052_105296

theorem factorial_divisibility (m : ℕ) (h : m > 1) :
  (m - 1).factorial % m = 0 ↔ ¬ Nat.Prime m := by sorry

end NUMINAMATH_CALUDE_factorial_divisibility_l1052_105296


namespace NUMINAMATH_CALUDE_fifteenth_entry_is_29_l1052_105212

/-- r₁₁(m) is the remainder when m is divided by 11 -/
def r₁₁ (m : ℕ) : ℕ := m % 11

/-- List of nonnegative integers n that satisfy r₁₁(7n) ≤ 5 -/
def satisfying_list : List ℕ :=
  (List.range (100 : ℕ)).filter (fun n => r₁₁ (7 * n) ≤ 5)

theorem fifteenth_entry_is_29 : satisfying_list[14] = 29 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_entry_is_29_l1052_105212


namespace NUMINAMATH_CALUDE_river_speed_is_6_l1052_105268

/-- Proves that the speed of the river is 6 km/h given the conditions of the boat problem -/
theorem river_speed_is_6 (total_distance : ℝ) (downstream_distance : ℝ) (still_water_speed : ℝ)
  (h1 : total_distance = 150)
  (h2 : downstream_distance = 90)
  (h3 : still_water_speed = 30)
  (h4 : downstream_distance / (still_water_speed + 6) = (total_distance - downstream_distance) / (still_water_speed - 6)) :
  6 = 6 := by
sorry

end NUMINAMATH_CALUDE_river_speed_is_6_l1052_105268


namespace NUMINAMATH_CALUDE_winter_olympics_volunteers_l1052_105259

/-- The number of ways to distribute n indistinguishable objects into k distinguishable boxes,
    with each box containing at least one object. -/
def distribute (n k : ℕ) : ℕ :=
  (k - 1).choose (n - 1) * k.factorial

/-- The problem statement -/
theorem winter_olympics_volunteers : distribute 5 4 = 240 := by
  sorry

end NUMINAMATH_CALUDE_winter_olympics_volunteers_l1052_105259


namespace NUMINAMATH_CALUDE_sod_area_calculation_l1052_105282

-- Define the dimensions
def yard_width : ℕ := 200
def yard_depth : ℕ := 50
def sidewalk_width : ℕ := 3
def sidewalk_length : ℕ := 50
def front_flowerbed_depth : ℕ := 4
def front_flowerbed_length : ℕ := 25
def third_flowerbed_width : ℕ := 10
def third_flowerbed_length : ℕ := 12
def fourth_flowerbed_width : ℕ := 7
def fourth_flowerbed_length : ℕ := 8

-- Define the theorem
theorem sod_area_calculation : 
  let total_area := yard_width * yard_depth
  let sidewalk_area := sidewalk_width * sidewalk_length
  let front_flowerbeds_area := 2 * (front_flowerbed_depth * front_flowerbed_length)
  let third_flowerbed_area := third_flowerbed_width * third_flowerbed_length
  let fourth_flowerbed_area := fourth_flowerbed_width * fourth_flowerbed_length
  let non_sod_area := sidewalk_area + front_flowerbeds_area + third_flowerbed_area + fourth_flowerbed_area
  total_area - non_sod_area = 9474 := by
sorry

end NUMINAMATH_CALUDE_sod_area_calculation_l1052_105282


namespace NUMINAMATH_CALUDE_problem_grid_square_count_l1052_105246

/-- Represents the grid structure in the problem -/
structure GridStructure where
  width : Nat
  height : Nat
  largeSquares : Nat
  topRowExtraSquares : Nat
  bottomRowExtraSquares : Nat

/-- Counts the number of squares of a given size in the grid -/
def countSquares (g : GridStructure) (size : Nat) : Nat :=
  match size with
  | 1 => g.largeSquares * 6 + g.topRowExtraSquares + g.bottomRowExtraSquares
  | 2 => g.largeSquares * 4 + 4
  | 3 => g.largeSquares * 2 + 1
  | _ => 0

/-- The total number of squares in the grid -/
def totalSquares (g : GridStructure) : Nat :=
  (countSquares g 1) + (countSquares g 2) + (countSquares g 3)

/-- The specific grid structure from the problem -/
def problemGrid : GridStructure := {
  width := 5
  height := 5
  largeSquares := 2
  topRowExtraSquares := 5
  bottomRowExtraSquares := 4
}

theorem problem_grid_square_count :
  totalSquares problemGrid = 38 := by
  sorry

end NUMINAMATH_CALUDE_problem_grid_square_count_l1052_105246


namespace NUMINAMATH_CALUDE_biff_kenneth_race_l1052_105233

/-- Biff and Kenneth's rowboat race problem -/
theorem biff_kenneth_race (race_distance : ℝ) (kenneth_speed : ℝ) (kenneth_extra_distance : ℝ) :
  race_distance = 500 →
  kenneth_speed = 51 →
  kenneth_extra_distance = 10 →
  ∃ (biff_speed : ℝ),
    biff_speed = 50 ∧
    biff_speed * (race_distance + kenneth_extra_distance) / kenneth_speed = race_distance :=
by sorry

end NUMINAMATH_CALUDE_biff_kenneth_race_l1052_105233


namespace NUMINAMATH_CALUDE_circle_diameter_endpoint_l1052_105249

/-- Given a circle with center (5, -4) and one endpoint of a diameter at (0, -9),
    the other endpoint of the diameter is at (10, 1). -/
theorem circle_diameter_endpoint :
  ∀ (P : ℝ × ℝ) (A : ℝ × ℝ) (Q : ℝ × ℝ),
  P = (5, -4) →  -- Center of the circle
  A = (0, -9) →  -- One endpoint of the diameter
  (P.1 - A.1)^2 + (P.2 - A.2)^2 = (Q.1 - P.1)^2 + (Q.2 - P.2)^2 →  -- A and Q are equidistant from P
  P.1 - A.1 = Q.1 - P.1 ∧ P.2 - A.2 = Q.2 - P.2 →  -- A, P, and Q are collinear
  Q = (10, 1) :=
by sorry

end NUMINAMATH_CALUDE_circle_diameter_endpoint_l1052_105249


namespace NUMINAMATH_CALUDE_optimal_bus_rental_l1052_105289

/-- Represents the optimal bus rental problem --/
theorem optimal_bus_rental
  (total_passengers : ℕ)
  (capacity_A capacity_B : ℕ)
  (cost_A cost_B : ℕ)
  (max_total_buses : ℕ)
  (max_B_minus_A : ℕ)
  (h_total_passengers : total_passengers = 900)
  (h_capacity_A : capacity_A = 36)
  (h_capacity_B : capacity_B = 60)
  (h_cost_A : cost_A = 1600)
  (h_cost_B : cost_B = 2400)
  (h_max_total_buses : max_total_buses = 21)
  (h_max_B_minus_A : max_B_minus_A = 7) :
  ∃ (x y : ℕ),
    x = 5 ∧ y = 12 ∧
    capacity_A * x + capacity_B * y ≥ total_passengers ∧
    x + y ≤ max_total_buses ∧
    y ≤ x + max_B_minus_A ∧
    ∀ (a b : ℕ),
      capacity_A * a + capacity_B * b ≥ total_passengers →
      a + b ≤ max_total_buses →
      b ≤ a + max_B_minus_A →
      cost_A * x + cost_B * y ≤ cost_A * a + cost_B * b :=
by sorry

end NUMINAMATH_CALUDE_optimal_bus_rental_l1052_105289


namespace NUMINAMATH_CALUDE_milk_purchase_theorem_l1052_105272

/-- The cost of one bag of milk in yuan -/
def milk_cost : ℕ := 3

/-- The number of bags paid for in the offer -/
def offer_paid : ℕ := 5

/-- The total number of bags received in the offer -/
def offer_total : ℕ := 6

/-- The amount of money mom has in yuan -/
def mom_money : ℕ := 20

/-- The maximum number of bags mom can buy -/
def max_bags : ℕ := 7

/-- The amount of money left after buying the maximum number of bags -/
def money_left : ℕ := 2

theorem milk_purchase_theorem :
  milk_cost * (offer_paid * (max_bags / offer_total) + max_bags % offer_total) ≤ mom_money ∧
  mom_money - milk_cost * (offer_paid * (max_bags / offer_total) + max_bags % offer_total) = money_left :=
by sorry

end NUMINAMATH_CALUDE_milk_purchase_theorem_l1052_105272


namespace NUMINAMATH_CALUDE_white_area_is_42_l1052_105270

/-- The area of a rectangle -/
def rectangle_area (width : ℕ) (height : ℕ) : ℕ := width * height

/-- The area of the letter C -/
def c_area : ℕ := 2 * (6 * 1) + 1 * 4

/-- The area of the letter O -/
def o_area : ℕ := 2 * (6 * 1) + 2 * 4

/-- The area of the letter L -/
def l_area : ℕ := 1 * (6 * 1) + 1 * 4

/-- The total black area of the word COOL -/
def cool_area : ℕ := c_area + 2 * o_area + l_area

/-- The width of the sign -/
def sign_width : ℕ := 18

/-- The height of the sign -/
def sign_height : ℕ := 6

theorem white_area_is_42 : 
  rectangle_area sign_width sign_height - cool_area = 42 := by
  sorry

end NUMINAMATH_CALUDE_white_area_is_42_l1052_105270


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1052_105222

/-- The remainder when x^3 + 3x^2 is divided by x^2 - 7x + 2 is 68x - 20 -/
theorem polynomial_division_remainder : ∃ q : Polynomial ℚ, 
  x^3 + 3*x^2 = (x^2 - 7*x + 2) * q + (68*x - 20) := by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1052_105222


namespace NUMINAMATH_CALUDE_floor_tiles_count_l1052_105224

/-- Represents a rectangular floor covered with square tiles. -/
structure TiledFloor where
  width : ℕ
  length : ℕ
  diagonalTiles : ℕ

/-- The floor is twice as long as it is wide. -/
def isDoubleLength (floor : TiledFloor) : Prop :=
  floor.length = 2 * floor.width

/-- The total number of tiles on the floor. -/
def totalTiles (floor : TiledFloor) : ℕ :=
  floor.width * floor.length

/-- The main theorem to be proved. -/
theorem floor_tiles_count (floor : TiledFloor) 
  (h1 : isDoubleLength floor) 
  (h2 : floor.diagonalTiles = 39) : 
  totalTiles floor = 722 := by
  sorry

end NUMINAMATH_CALUDE_floor_tiles_count_l1052_105224


namespace NUMINAMATH_CALUDE_ivan_purchase_cost_l1052_105286

/-- Calculates the total cost of a discounted purchase -/
def discounted_purchase_cost (original_price discount quantity : ℕ) : ℕ :=
  (original_price - discount) * quantity

/-- Proves that the total cost for Ivan's purchase is $100 -/
theorem ivan_purchase_cost :
  discounted_purchase_cost 12 2 10 = 100 := by
  sorry

end NUMINAMATH_CALUDE_ivan_purchase_cost_l1052_105286


namespace NUMINAMATH_CALUDE_team_loss_percentage_l1052_105290

theorem team_loss_percentage
  (win_loss_ratio : ℚ)
  (total_games : ℕ)
  (h1 : win_loss_ratio = 8 / 5)
  (h2 : total_games = 52) :
  (loss_percentage : ℚ) →
  loss_percentage = 38 / 100 :=
by sorry

end NUMINAMATH_CALUDE_team_loss_percentage_l1052_105290


namespace NUMINAMATH_CALUDE_f_neg_one_eq_neg_two_l1052_105279

/- Define an odd function f -/
def f (x : ℝ) : ℝ := sorry

/- State the properties of f -/
axiom f_odd : ∀ x, f (-x) = -f x
axiom f_positive : ∀ x > 0, f x = x^2 + 1/x

/- Theorem to prove -/
theorem f_neg_one_eq_neg_two : f (-1) = -2 := by sorry

end NUMINAMATH_CALUDE_f_neg_one_eq_neg_two_l1052_105279


namespace NUMINAMATH_CALUDE_words_with_vowels_count_l1052_105284

def alphabet : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}
def vowels : Finset Char := {'A', 'E'}
def word_length : Nat := 5

def total_words : Nat := alphabet.card ^ word_length
def words_without_vowels : Nat := (alphabet \ vowels).card ^ word_length

theorem words_with_vowels_count :
  total_words - words_without_vowels = 6752 :=
sorry

end NUMINAMATH_CALUDE_words_with_vowels_count_l1052_105284


namespace NUMINAMATH_CALUDE_police_catches_thief_l1052_105257

-- Define the square courtyard
def side_length : ℝ := 340

-- Define speeds
def police_speed : ℝ := 85
def thief_speed : ℝ := 75

-- Define the time to catch
def time_to_catch : ℝ := 44

-- Theorem statement
theorem police_catches_thief :
  let time_to_sight : ℝ := (4 * side_length) / (police_speed - thief_speed)
  let police_distance : ℝ := police_speed * time_to_sight
  let thief_distance : ℝ := thief_speed * time_to_sight
  let remaining_side : ℝ := side_length - (thief_distance % side_length)
  let chase_time : ℝ := Real.sqrt ((remaining_side^2) / (police_speed^2 - thief_speed^2))
  time_to_sight + chase_time = time_to_catch :=
by sorry

end NUMINAMATH_CALUDE_police_catches_thief_l1052_105257


namespace NUMINAMATH_CALUDE_line_through_points_eq_target_l1052_105211

/-- The equation of a line passing through two points -/
def line_equation (x1 y1 x2 y2 : ℝ) (x y : ℝ) : Prop :=
  (y - y1) * (x2 - x1) = (x - x1) * (y2 - y1)

/-- The specific points given in the problem -/
def point1 : ℝ × ℝ := (-1, 0)
def point2 : ℝ × ℝ := (0, 1)

/-- The equation we want to prove -/
def target_equation (x y : ℝ) : Prop := x - y + 1 = 0

/-- Theorem stating that the line equation passing through the given points
    is equivalent to the target equation -/
theorem line_through_points_eq_target :
  ∀ x y : ℝ, line_equation point1.1 point1.2 point2.1 point2.2 x y ↔ target_equation x y :=
by sorry

end NUMINAMATH_CALUDE_line_through_points_eq_target_l1052_105211


namespace NUMINAMATH_CALUDE_average_speed_proof_l1052_105248

/-- Prove that the average speed of a trip with given conditions is 40 miles per hour -/
theorem average_speed_proof (total_distance : ℝ) (first_half_speed : ℝ) (second_half_time_factor : ℝ) :
  total_distance = 640 →
  first_half_speed = 80 →
  second_half_time_factor = 3 →
  (total_distance / (total_distance / (2 * first_half_speed) * (1 + second_half_time_factor))) = 40 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_proof_l1052_105248


namespace NUMINAMATH_CALUDE_diagonals_25_sided_polygon_l1052_105223

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 25 sides has 275 diagonals -/
theorem diagonals_25_sided_polygon : num_diagonals 25 = 275 := by
  sorry

end NUMINAMATH_CALUDE_diagonals_25_sided_polygon_l1052_105223


namespace NUMINAMATH_CALUDE_work_efficiency_increase_l1052_105262

theorem work_efficiency_increase (original_months : ℕ) (actual_months : ℕ) (x : ℚ) : 
  original_months = 20 →
  actual_months = 18 →
  (1 : ℚ) / actual_months = (1 : ℚ) / original_months * (1 + x / 100) :=
by sorry

end NUMINAMATH_CALUDE_work_efficiency_increase_l1052_105262


namespace NUMINAMATH_CALUDE_gathering_attendees_l1052_105260

theorem gathering_attendees (n : ℕ) : 
  (n * (n - 1) / 2 = 55) → n = 11 := by
  sorry

end NUMINAMATH_CALUDE_gathering_attendees_l1052_105260


namespace NUMINAMATH_CALUDE_root_sum_pow_l1052_105245

theorem root_sum_pow (p q : ℝ) : 
  p^2 - 7*p + 12 = 0 → 
  q^2 - 7*q + 12 = 0 → 
  p^3 + p^4*q^2 + p^2*q^4 + q^3 = 3691 := by
sorry

end NUMINAMATH_CALUDE_root_sum_pow_l1052_105245


namespace NUMINAMATH_CALUDE_integer_roots_of_cubic_l1052_105220

def f (x : ℤ) : ℤ := x^3 - 3*x^2 - 13*x + 15

theorem integer_roots_of_cubic :
  {x : ℤ | f x = 0} = {-3, 1, 5} := by sorry

end NUMINAMATH_CALUDE_integer_roots_of_cubic_l1052_105220


namespace NUMINAMATH_CALUDE_min_books_proof_l1052_105237

def scooter_cost : ℕ := 3000
def earning_per_book : ℕ := 15
def transport_cost_per_book : ℕ := 4

def min_books_to_earn_back : ℕ := 273

theorem min_books_proof :
  min_books_to_earn_back = (
    let profit_per_book := earning_per_book - transport_cost_per_book
    (scooter_cost + profit_per_book - 1) / profit_per_book
  ) :=
by sorry

end NUMINAMATH_CALUDE_min_books_proof_l1052_105237


namespace NUMINAMATH_CALUDE_binomial_12_11_l1052_105235

theorem binomial_12_11 : (12 : ℕ).choose 11 = 12 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_11_l1052_105235


namespace NUMINAMATH_CALUDE_population_size_l1052_105251

/-- Given a population with specified birth and death rates, and a net growth rate,
    prove that the initial population size is 3000. -/
theorem population_size (birth_rate death_rate net_growth_rate : ℝ) 
  (h1 : birth_rate = 52)
  (h2 : death_rate = 16)
  (h3 : net_growth_rate = 0.012)
  (h4 : birth_rate - death_rate = net_growth_rate * 100) : 
  (birth_rate - death_rate) / net_growth_rate = 3000 := by
  sorry

end NUMINAMATH_CALUDE_population_size_l1052_105251


namespace NUMINAMATH_CALUDE_closest_point_l1052_105232

def v (s : ℝ) : Fin 3 → ℝ := fun i => 
  match i with
  | 0 => 3 + 5*s
  | 1 => -2 + 3*s
  | 2 => -4 - 2*s

def a : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 5
  | 1 => 5
  | 2 => 6

def direction : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 5
  | 1 => 3
  | 2 => -2

theorem closest_point (s : ℝ) : 
  (∀ t : ℝ, ‖v t - a‖ ≥ ‖v s - a‖) ↔ s = 11/38 :=
sorry

end NUMINAMATH_CALUDE_closest_point_l1052_105232


namespace NUMINAMATH_CALUDE_inequality_proof_l1052_105204

theorem inequality_proof (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h4 : a + b + c = 1) 
  (h5 : ∀ x : ℝ, |x - a| + |x - 1| ≥ (a^2 + b^2 + c^2) / (b + c)) : 
  a ≤ Real.sqrt 2 - 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1052_105204


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l1052_105242

/-- A line in 2D space represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are perpendicular -/
def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem perpendicular_line_equation (P : Point) (L : Line) :
  P.x = 4 ∧ P.y = -1 ∧ L.a = 3 ∧ L.b = -4 ∧ L.c = 6 →
  ∃ (M : Line), M.a = 4 ∧ M.b = 3 ∧ M.c = -13 ∧ P.liesOn M ∧ M.perpendicular L := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l1052_105242


namespace NUMINAMATH_CALUDE_graphs_intersect_once_l1052_105230

/-- The value of b for which the graphs of y = bx^2 + 5x + 3 and y = -2x - 3 intersect at exactly one point -/
def b : ℚ := 49 / 24

/-- The first equation -/
def f (x : ℝ) : ℝ := b * x^2 + 5 * x + 3

/-- The second equation -/
def g (x : ℝ) : ℝ := -2 * x - 3

/-- Theorem stating that the graphs intersect at exactly one point when b = 49/24 -/
theorem graphs_intersect_once :
  ∃! x : ℝ, f x = g x :=
sorry

end NUMINAMATH_CALUDE_graphs_intersect_once_l1052_105230


namespace NUMINAMATH_CALUDE_slope_theorem_l1052_105214

/-- Given two points A(-3, 8) and B(5, y) in a coordinate plane, 
    if the slope of the line through A and B is -1/2, then y = 4. -/
theorem slope_theorem (y : ℝ) : 
  let A : ℝ × ℝ := (-3, 8)
  let B : ℝ × ℝ := (5, y)
  let slope := (B.2 - A.2) / (B.1 - A.1)
  slope = -1/2 → y = 4 := by
sorry


end NUMINAMATH_CALUDE_slope_theorem_l1052_105214


namespace NUMINAMATH_CALUDE_problem_solution_l1052_105219

theorem problem_solution (x : ℕ) (h : x = 36) : 
  2 * ((((x + 10) * 2) / 2) - 2) = 88 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1052_105219


namespace NUMINAMATH_CALUDE_cube_surface_area_l1052_105216

/-- The surface area of a cube given the sum of its edge lengths -/
theorem cube_surface_area (edge_sum : ℝ) (h : edge_sum = 72) : 
  6 * (edge_sum / 12)^2 = 216 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l1052_105216


namespace NUMINAMATH_CALUDE_pencils_indeterminate_l1052_105213

/-- Represents the contents of a drawer -/
structure Drawer where
  initial_crayons : ℕ
  added_crayons : ℕ
  final_crayons : ℕ
  pencils : ℕ

/-- Theorem stating that the number of pencils cannot be determined -/
theorem pencils_indeterminate (d : Drawer) 
  (h1 : d.initial_crayons = 41)
  (h2 : d.added_crayons = 12)
  (h3 : d.final_crayons = 53)
  : ¬ ∃ (n : ℕ), ∀ (d' : Drawer), 
    d'.initial_crayons = d.initial_crayons ∧ 
    d'.added_crayons = d.added_crayons ∧ 
    d'.final_crayons = d.final_crayons → 
    d'.pencils = n :=
by
  sorry

end NUMINAMATH_CALUDE_pencils_indeterminate_l1052_105213


namespace NUMINAMATH_CALUDE_jerrys_pool_length_l1052_105210

/-- Represents the problem of calculating Jerry's pool length --/
theorem jerrys_pool_length :
  ∀ (total_water drinking_cooking_water shower_water num_showers pool_width pool_height : ℝ),
    total_water = 1000 →
    drinking_cooking_water = 100 →
    shower_water = 20 →
    num_showers = 15 →
    pool_width = 10 →
    pool_height = 6 →
    ∃ (pool_length : ℝ),
      pool_length = 10 ∧
      pool_length * pool_width * pool_height = 
        total_water - (drinking_cooking_water + shower_water * num_showers) :=
by sorry

end NUMINAMATH_CALUDE_jerrys_pool_length_l1052_105210


namespace NUMINAMATH_CALUDE_abs_diff_gt_cube_root_product_l1052_105240

theorem abs_diff_gt_cube_root_product (a b : ℤ) : 
  a ≠ b → (a^2 + a*b + b^2) ∣ (a*b*(a + b)) → |a - b| > (a*b : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_abs_diff_gt_cube_root_product_l1052_105240


namespace NUMINAMATH_CALUDE_paths_in_4x4_grid_l1052_105247

/-- Number of paths in a grid -/
def num_paths (m n : ℕ) : ℕ :=
  if m = 0 ∨ n = 0 then 1
  else num_paths (m - 1) n + num_paths m (n - 1)

/-- Theorem: The number of paths from (0,0) to (3,3) in a 4x4 grid is 23 -/
theorem paths_in_4x4_grid : num_paths 3 3 = 23 := by
  sorry

end NUMINAMATH_CALUDE_paths_in_4x4_grid_l1052_105247


namespace NUMINAMATH_CALUDE_f_period_and_g_max_l1052_105236

noncomputable def f (x : ℝ) : ℝ := Real.sin (x + Real.pi / 2) * Real.cos (x - Real.pi / 3)

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6)

theorem f_period_and_g_max :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∃ (M : ℝ), M = 2 ∧ ∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 3) → g x ≤ M) :=
sorry

end NUMINAMATH_CALUDE_f_period_and_g_max_l1052_105236


namespace NUMINAMATH_CALUDE_work_problem_l1052_105275

/-- Proves that 8 men became absent given the conditions of the work problem -/
theorem work_problem (total_men : ℕ) (original_days : ℕ) (actual_days : ℕ) 
  (h1 : total_men = 42)
  (h2 : original_days = 17)
  (h3 : actual_days = 21)
  (h4 : total_men * original_days = (total_men - absent_men) * actual_days) :
  absent_men = 8 := by
  sorry

#check work_problem

end NUMINAMATH_CALUDE_work_problem_l1052_105275


namespace NUMINAMATH_CALUDE_percentage_problem_l1052_105280

theorem percentage_problem : 
  ∃ p : ℝ, p * 24 = 0.12 ∧ p = 0.005 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1052_105280


namespace NUMINAMATH_CALUDE_gcd_of_powers_of_97_l1052_105226

theorem gcd_of_powers_of_97 :
  Nat.gcd (97^10 + 1) (97^10 + 97^3 + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_powers_of_97_l1052_105226


namespace NUMINAMATH_CALUDE_equal_lengths_k_value_l1052_105207

theorem equal_lengths_k_value (AB AC : ℝ) (k : ℝ) : 
  AB = AC → AB = 8 → AC = 5 - k → k = -3 := by
  sorry

end NUMINAMATH_CALUDE_equal_lengths_k_value_l1052_105207


namespace NUMINAMATH_CALUDE_largest_of_five_consecutive_even_integers_l1052_105238

/-- Sum of the first n positive integers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Sum of the first n positive even integers -/
def sum_first_n_even (n : ℕ) : ℕ := 2 * sum_first_n n

/-- Sum of five consecutive even integers -/
def sum_five_consecutive_even (n : ℕ) : ℕ := 5 * n - 20

theorem largest_of_five_consecutive_even_integers :
  ∃ (n : ℕ), sum_five_consecutive_even n = sum_first_n_even 30 ∧
             n = 190 ∧
             ∀ (m : ℕ), sum_five_consecutive_even m = sum_first_n_even 30 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_of_five_consecutive_even_integers_l1052_105238


namespace NUMINAMATH_CALUDE_spongebob_fry_price_l1052_105264

/-- Calculates the price of each large fry given the number of burgers sold, 
    price per burger, number of large fries sold, and total earnings. -/
def price_of_large_fry (num_burgers : ℕ) (price_per_burger : ℚ) 
                       (num_fries : ℕ) (total_earnings : ℚ) : ℚ :=
  (total_earnings - num_burgers * price_per_burger) / num_fries

/-- Theorem stating that the price of each large fry is $1.50 
    given Spongebob's sales information. -/
theorem spongebob_fry_price : 
  price_of_large_fry 30 2 12 78 = (3/2) := by
  sorry

end NUMINAMATH_CALUDE_spongebob_fry_price_l1052_105264


namespace NUMINAMATH_CALUDE_modulus_of_z_squared_l1052_105241

theorem modulus_of_z_squared (i : ℂ) (h : i^2 = -1) : 
  let z := (2 - i)^2
  Complex.abs z = 5 := by
sorry

end NUMINAMATH_CALUDE_modulus_of_z_squared_l1052_105241


namespace NUMINAMATH_CALUDE_martin_goldfish_l1052_105288

/-- Calculates the number of goldfish after a given number of weeks -/
def goldfish_after_weeks (initial : ℕ) (die_per_week : ℕ) (buy_per_week : ℕ) (weeks : ℕ) : ℤ :=
  initial - (die_per_week * weeks : ℕ) + (buy_per_week * weeks : ℕ)

/-- Theorem stating the number of goldfish Martin will have after 7 weeks -/
theorem martin_goldfish : goldfish_after_weeks 18 5 3 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_martin_goldfish_l1052_105288


namespace NUMINAMATH_CALUDE_jack_and_jill_meeting_point_l1052_105227

/-- Represents the problem of Jack and Jill running up and down a hill -/
structure HillRun where
  total_distance : ℝ
  uphill_distance : ℝ
  jack_head_start : ℝ
  jack_uphill_speed : ℝ
  jack_downhill_speed : ℝ
  jill_uphill_speed : ℝ
  jill_downhill_speed : ℝ

/-- Calculates the meeting point of Jack and Jill -/
def meeting_point (run : HillRun) : ℝ :=
  sorry

/-- The main theorem stating the distance from the top where Jack and Jill meet -/
theorem jack_and_jill_meeting_point (run : HillRun) 
  (h1 : run.total_distance = 12)
  (h2 : run.uphill_distance = 6)
  (h3 : run.jack_head_start = 1/6)  -- 10 minutes in hours
  (h4 : run.jack_uphill_speed = 15)
  (h5 : run.jack_downhill_speed = 20)
  (h6 : run.jill_uphill_speed = 18)
  (h7 : run.jill_downhill_speed = 24) :
  run.uphill_distance - meeting_point run = 33/19 :=
sorry

end NUMINAMATH_CALUDE_jack_and_jill_meeting_point_l1052_105227


namespace NUMINAMATH_CALUDE_expression_evaluation_l1052_105254

theorem expression_evaluation :
  let x : ℝ := Real.sqrt 2 + 1
  ((x + 1) / (x - 1) + 1 / (x^2 - 2*x + 1)) / (x / (x - 1)) = 1 + Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1052_105254


namespace NUMINAMATH_CALUDE_equation_solution_l1052_105283

theorem equation_solution :
  let f : ℝ → ℝ := λ x => -x^2 * (x + 5) - (5*x - 2)
  ∀ x : ℝ, f x = 0 ↔ x = -2 ∨ x = (-3 + Real.sqrt 5) / 2 ∨ x = (-3 - Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1052_105283


namespace NUMINAMATH_CALUDE_romeo_profit_l1052_105252

/-- Calculates the profit for selling chocolate bars -/
def chocolate_profit (num_bars : ℕ) (cost_per_bar : ℕ) (selling_price : ℕ) (packaging_cost : ℕ) : ℕ :=
  selling_price - (num_bars * cost_per_bar + num_bars * packaging_cost)

/-- Theorem: Romeo's profit is $55 -/
theorem romeo_profit : 
  chocolate_profit 5 5 90 2 = 55 := by
  sorry

end NUMINAMATH_CALUDE_romeo_profit_l1052_105252


namespace NUMINAMATH_CALUDE_circle_and_m_value_l1052_105208

-- Define the circle C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the line equation ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the problem conditions
def problem_conditions (C : Circle) (m : ℝ) : Prop :=
  -- Center C is on the line 2x-y-7=0
  2 * C.center.1 - C.center.2 - 7 = 0 ∧
  -- Circle intersects y-axis at (0, -4) and (0, -2)
  (0 - C.center.1)^2 + (-4 - C.center.2)^2 = C.radius^2 ∧
  (0 - C.center.1)^2 + (-2 - C.center.2)^2 = C.radius^2 ∧
  -- Line x+2y+m=0 intersects circle C
  ∃ (A B : ℝ × ℝ), 
    (A.1 + 2*A.2 + m = 0) ∧
    (B.1 + 2*B.2 + m = 0) ∧
    (A.1 - C.center.1)^2 + (A.2 - C.center.2)^2 = C.radius^2 ∧
    (B.1 - C.center.1)^2 + (B.2 - C.center.2)^2 = C.radius^2 ∧
  -- Parallelogram ACBD with CA and CB as adjacent sides, D on circle C
  ∃ (D : ℝ × ℝ),
    (D.1 - C.center.1)^2 + (D.2 - C.center.2)^2 = C.radius^2

-- Theorem statement
theorem circle_and_m_value (C : Circle) (m : ℝ) :
  problem_conditions C m →
  (C.center = (2, -3) ∧ C.radius^2 = 5) ∧ (m = 3/2 ∨ m = 13/2) :=
sorry

end NUMINAMATH_CALUDE_circle_and_m_value_l1052_105208


namespace NUMINAMATH_CALUDE_blocks_eaten_l1052_105206

theorem blocks_eaten (initial_blocks remaining_blocks : ℕ) 
  (h1 : initial_blocks = 55)
  (h2 : remaining_blocks = 26) :
  initial_blocks - remaining_blocks = 29 := by
  sorry

end NUMINAMATH_CALUDE_blocks_eaten_l1052_105206


namespace NUMINAMATH_CALUDE_apples_handed_out_to_students_l1052_105231

theorem apples_handed_out_to_students 
  (initial_apples : ℕ) 
  (pies_made : ℕ) 
  (apples_per_pie : ℕ) 
  (h1 : initial_apples = 62)
  (h2 : pies_made = 6)
  (h3 : apples_per_pie = 9) :
  initial_apples - (pies_made * apples_per_pie) = 8 := by
sorry

end NUMINAMATH_CALUDE_apples_handed_out_to_students_l1052_105231


namespace NUMINAMATH_CALUDE_min_value_expression_l1052_105263

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 1) : 
  (6 * z) / (2 * x + y) + (6 * x) / (y + 2 * z) + (4 * y) / (x + z) ≥ (5.5 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l1052_105263


namespace NUMINAMATH_CALUDE_real_roots_range_l1052_105293

theorem real_roots_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + 4*a*x - 4*a + 3 = 0) ∨ 
  (∃ x : ℝ, x^2 + (a-1)*x + a^2 = 0) ∨ 
  (∃ x : ℝ, x^2 + 2*a*x - 2*a = 0) → 
  a ≤ -3/2 ∨ a ≥ -1 := by
sorry

end NUMINAMATH_CALUDE_real_roots_range_l1052_105293


namespace NUMINAMATH_CALUDE_average_of_remaining_numbers_l1052_105203

theorem average_of_remaining_numbers
  (n : ℕ)
  (total : ℕ)
  (subset_sum : ℕ)
  (h1 : n = 5)
  (h2 : total = n * 20)
  (h3 : subset_sum = 48)
  (h4 : subset_sum < total) :
  (total - subset_sum) / 2 = 26 := by
sorry

end NUMINAMATH_CALUDE_average_of_remaining_numbers_l1052_105203


namespace NUMINAMATH_CALUDE_inscribed_tetrahedron_volume_l1052_105221

/-- A regular tetrahedron inscribed in a unit sphere -/
structure InscribedTetrahedron where
  /-- The tetrahedron is regular -/
  regular : Bool
  /-- All vertices lie on the surface of the sphere -/
  vertices_on_sphere : Bool
  /-- The sphere has radius 1 -/
  sphere_radius : ℝ
  /-- Three vertices of the base are on a great circle of the sphere -/
  base_on_great_circle : Bool

/-- The volume of the inscribed regular tetrahedron -/
def tetrahedron_volume (t : InscribedTetrahedron) : ℝ :=
  sorry

/-- Theorem stating the volume of the inscribed regular tetrahedron -/
theorem inscribed_tetrahedron_volume 
  (t : InscribedTetrahedron) 
  (h1 : t.regular = true) 
  (h2 : t.vertices_on_sphere = true) 
  (h3 : t.sphere_radius = 1) 
  (h4 : t.base_on_great_circle = true) : 
  tetrahedron_volume t = Real.sqrt 3 / 4 :=
sorry

end NUMINAMATH_CALUDE_inscribed_tetrahedron_volume_l1052_105221


namespace NUMINAMATH_CALUDE_conic_section_eccentricity_l1052_105295

theorem conic_section_eccentricity (m : ℝ) : 
  (m^2 = 2 * 8) →
  (∃ (e : ℝ), (e = Real.sqrt 3 / 2 ∨ e = Real.sqrt 5) ∧
    ((m > 0 → e = Real.sqrt 3 / 2) ∧
     (m < 0 → e = Real.sqrt 5)) ∧
    (∀ (x y : ℝ), x^2 + y^2 / m = 1 → 
      (m > 0 → e^2 = 1 - (1 / m)) ∧
      (m < 0 → e^2 = 1 + (1 / m)))) :=
by sorry

end NUMINAMATH_CALUDE_conic_section_eccentricity_l1052_105295


namespace NUMINAMATH_CALUDE_multiple_valid_scenarios_exist_l1052_105274

/-- Represents the ticket sales scenario for the Red Rose Theatre -/
structure TicketSales where
  total_tickets : ℕ
  total_sales : ℚ
  tickets_at_price1 : ℕ
  price1 : ℚ
  price2 : ℚ

/-- Checks if the given ticket sales scenario is valid -/
def is_valid_scenario (s : TicketSales) : Prop :=
  s.total_tickets = s.tickets_at_price1 + (s.total_tickets - s.tickets_at_price1) ∧
  s.total_sales = s.tickets_at_price1 * s.price1 + (s.total_tickets - s.tickets_at_price1) * s.price2

/-- States that multiple valid scenarios can exist for the same input data -/
theorem multiple_valid_scenarios_exist (total_tickets : ℕ) (total_sales : ℚ) (tickets_at_price1 : ℕ) :
  ∃ (s1 s2 : TicketSales),
    s1.total_tickets = total_tickets ∧
    s1.total_sales = total_sales ∧
    s1.tickets_at_price1 = tickets_at_price1 ∧
    s2.total_tickets = total_tickets ∧
    s2.total_sales = total_sales ∧
    s2.tickets_at_price1 = tickets_at_price1 ∧
    is_valid_scenario s1 ∧
    is_valid_scenario s2 ∧
    s1.price1 ≠ s2.price1 :=
  sorry

#check multiple_valid_scenarios_exist 380 1972.5 205

end NUMINAMATH_CALUDE_multiple_valid_scenarios_exist_l1052_105274


namespace NUMINAMATH_CALUDE_smallest_n_for_divisible_sum_l1052_105200

theorem smallest_n_for_divisible_sum (n : ℕ) : n ≥ 4 → (
  (∀ S : Finset ℤ, S.card = n → ∃ a b c d : ℤ, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 20 ∣ (a + b - c - d))
  ↔ n ≥ 9
) := by sorry

end NUMINAMATH_CALUDE_smallest_n_for_divisible_sum_l1052_105200


namespace NUMINAMATH_CALUDE_hostel_expenditure_equation_l1052_105202

/-- Represents the average expenditure calculation for a student hostel with varying group costs. -/
theorem hostel_expenditure_equation 
  (A B C : ℕ) -- Original number of students in each group
  (a b c : ℕ) -- New students in each group
  (X Y Z : ℝ) -- Average expenditure for each group
  (h1 : A + B + C = 35) -- Total original students
  (h2 : a + b + c = 7)  -- Total new students
  : (A * X + B * Y + C * Z) / 35 - 1 = 
    ((A + a) * X + (B + b) * Y + (C + c) * Z + 42) / 42 := by
  sorry

#check hostel_expenditure_equation

end NUMINAMATH_CALUDE_hostel_expenditure_equation_l1052_105202


namespace NUMINAMATH_CALUDE_jason_music_store_spending_l1052_105217

/-- The price of the flute Jason bought -/
def flute_price : ℚ := 142.46

/-- The price of the music stand Jason bought -/
def stand_price : ℚ := 8.89

/-- The price of the song book Jason bought -/
def book_price : ℚ := 7

/-- The total amount Jason spent at the music store -/
def total_spent : ℚ := flute_price + stand_price + book_price

/-- Theorem stating that the total amount Jason spent is $158.35 -/
theorem jason_music_store_spending :
  total_spent = 158.35 := by sorry

end NUMINAMATH_CALUDE_jason_music_store_spending_l1052_105217


namespace NUMINAMATH_CALUDE_hexagon_area_l1052_105266

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The hexagon defined by its vertices -/
def hexagon : List Point := [
  ⟨0, 3⟩, ⟨3, 3⟩, ⟨4, 0⟩, ⟨3, -3⟩, ⟨0, -3⟩, ⟨-1, 0⟩
]

/-- Calculate the area of a polygon given its vertices -/
def polygonArea (vertices : List Point) : ℝ := sorry

/-- Theorem: The area of the specified hexagon is 18 square units -/
theorem hexagon_area : polygonArea hexagon = 18 := by sorry

end NUMINAMATH_CALUDE_hexagon_area_l1052_105266
