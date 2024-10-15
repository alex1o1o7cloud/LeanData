import Mathlib

namespace NUMINAMATH_CALUDE_solve_equation_l3398_339888

theorem solve_equation (X : ℝ) : (X^3).sqrt = 81 * (81^(1/9)) → X = 3^(80/27) := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3398_339888


namespace NUMINAMATH_CALUDE_square_of_one_minus_i_l3398_339862

theorem square_of_one_minus_i (i : ℂ) : i^2 = -1 → (1 - i)^2 = -2*i := by
  sorry

end NUMINAMATH_CALUDE_square_of_one_minus_i_l3398_339862


namespace NUMINAMATH_CALUDE_remainder_b91_mod_50_l3398_339814

theorem remainder_b91_mod_50 : ∃ k : ℤ, 7^91 + 9^91 = 50 * k + 16 := by sorry

end NUMINAMATH_CALUDE_remainder_b91_mod_50_l3398_339814


namespace NUMINAMATH_CALUDE_second_difference_quadratic_l3398_339851

theorem second_difference_quadratic 
  (f : ℕ → ℝ) 
  (h : ∀ n : ℕ, f (n + 2) - 2 * f (n + 1) + f n = 1) : 
  ∃ a b : ℝ, ∀ n : ℕ, f n = (1/2) * n^2 + a * n + b := by
sorry

end NUMINAMATH_CALUDE_second_difference_quadratic_l3398_339851


namespace NUMINAMATH_CALUDE_words_with_vowels_count_l3398_339827

def alphabet : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}
def vowels : Finset Char := {'A', 'E'}
def word_length : Nat := 5

def total_words : Nat := alphabet.card ^ word_length
def words_without_vowels : Nat := (alphabet \ vowels).card ^ word_length

theorem words_with_vowels_count :
  total_words - words_without_vowels = 6752 :=
sorry

end NUMINAMATH_CALUDE_words_with_vowels_count_l3398_339827


namespace NUMINAMATH_CALUDE_largest_x_equation_l3398_339821

theorem largest_x_equation (x : ℝ) : 
  (((14 * x^3 - 40 * x^2 + 20 * x - 4) / (4 * x - 3) + 6 * x = 8 * x - 3) ↔ 
  (14 * x^3 - 48 * x^2 + 38 * x - 13 = 0)) ∧ 
  (∀ y : ℝ, ((14 * y^3 - 40 * y^2 + 20 * y - 4) / (4 * y - 3) + 6 * y = 8 * y - 3) → y ≤ x) := by
  sorry

end NUMINAMATH_CALUDE_largest_x_equation_l3398_339821


namespace NUMINAMATH_CALUDE_xy_gt_one_necessary_not_sufficient_l3398_339855

theorem xy_gt_one_necessary_not_sufficient :
  (∀ x y : ℝ, x > 1 ∧ y > 1 → x * y > 1) ∧
  (∃ x y : ℝ, x * y > 1 ∧ ¬(x > 1 ∧ y > 1)) :=
by sorry

end NUMINAMATH_CALUDE_xy_gt_one_necessary_not_sufficient_l3398_339855


namespace NUMINAMATH_CALUDE_min_voters_for_tall_to_win_l3398_339834

/-- Represents the voting structure and rules of the giraffe beauty contest --/
structure GiraffeContest where
  total_voters : Nat
  num_districts : Nat
  sections_per_district : Nat
  voters_per_section : Nat
  (total_voters_eq : total_voters = num_districts * sections_per_district * voters_per_section)
  (num_districts_pos : num_districts > 0)
  (sections_per_district_pos : sections_per_district > 0)
  (voters_per_section_pos : voters_per_section > 0)

/-- Calculates the minimum number of voters required to win the contest --/
def minVotersToWin (contest : GiraffeContest) : Nat :=
  let districts_to_win := contest.num_districts / 2 + 1
  let sections_to_win := contest.sections_per_district / 2 + 1
  let voters_to_win_section := contest.voters_per_section / 2 + 1
  districts_to_win * sections_to_win * voters_to_win_section

/-- The main theorem stating the minimum number of voters required for Tall to win --/
theorem min_voters_for_tall_to_win (contest : GiraffeContest)
  (h_total : contest.total_voters = 105)
  (h_districts : contest.num_districts = 5)
  (h_sections : contest.sections_per_district = 7)
  (h_voters : contest.voters_per_section = 3) :
  minVotersToWin contest = 24 := by
  sorry


end NUMINAMATH_CALUDE_min_voters_for_tall_to_win_l3398_339834


namespace NUMINAMATH_CALUDE_cube_surface_area_l3398_339858

/-- The surface area of a cube given the sum of its edge lengths -/
theorem cube_surface_area (edge_sum : ℝ) (h : edge_sum = 72) : 
  6 * (edge_sum / 12)^2 = 216 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l3398_339858


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l3398_339848

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

end NUMINAMATH_CALUDE_perpendicular_line_equation_l3398_339848


namespace NUMINAMATH_CALUDE_abs_inequality_iff_l3398_339810

theorem abs_inequality_iff (a b : ℝ) : a > b ↔ a * |a| > b * |b| := by sorry

end NUMINAMATH_CALUDE_abs_inequality_iff_l3398_339810


namespace NUMINAMATH_CALUDE_triangle_properties_l3398_339895

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.a * Real.sin t.B - t.b * Real.cos t.A = 0)
  (h2 : t.a = 2 * Real.sqrt 5)
  (h3 : t.b = 2) :
  t.A = π / 4 ∧ (1/2 * t.b * t.c * Real.sin t.A = 4) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3398_339895


namespace NUMINAMATH_CALUDE_college_entrance_exam_scoring_l3398_339884

theorem college_entrance_exam_scoring (total_questions raw_score questions_answered correct_answers : ℕ)
  (h1 : total_questions = 85)
  (h2 : questions_answered = 82)
  (h3 : correct_answers = 70)
  (h4 : raw_score = 67)
  (h5 : questions_answered ≤ total_questions)
  (h6 : correct_answers ≤ questions_answered) :
  ∃ (points_subtracted : ℚ),
    points_subtracted = 1/4 ∧
    (correct_answers : ℚ) - (questions_answered - correct_answers) * points_subtracted = raw_score := by
sorry

end NUMINAMATH_CALUDE_college_entrance_exam_scoring_l3398_339884


namespace NUMINAMATH_CALUDE_intersection_M_N_l3398_339869

def M : Set ℝ := {x | (x - 1)^2 < 4}

def N : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_M_N : M ∩ N = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3398_339869


namespace NUMINAMATH_CALUDE_distributive_property_negative_l3398_339845

theorem distributive_property_negative (a b : ℝ) : -3 * (a - b) = -3 * a + 3 * b := by
  sorry

end NUMINAMATH_CALUDE_distributive_property_negative_l3398_339845


namespace NUMINAMATH_CALUDE_consecutive_product_plus_one_is_square_l3398_339896

theorem consecutive_product_plus_one_is_square : ∃ m : ℕ, 
  2017 * 2018 * 2019 * 2020 + 1 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_product_plus_one_is_square_l3398_339896


namespace NUMINAMATH_CALUDE_david_spent_half_ben_spent_more_total_spent_is_48_l3398_339847

/-- The amount Ben spent at the bagel store -/
def ben_spent : ℝ := 32

/-- The amount David spent at the bagel store -/
def david_spent : ℝ := 16

/-- For every dollar Ben spent, David spent 50 cents less -/
theorem david_spent_half : david_spent = ben_spent / 2 := by sorry

/-- Ben paid $16.00 more than David -/
theorem ben_spent_more : ben_spent = david_spent + 16 := by sorry

/-- The total amount spent by Ben and David -/
def total_spent : ℝ := ben_spent + david_spent

theorem total_spent_is_48 : total_spent = 48 := by sorry

end NUMINAMATH_CALUDE_david_spent_half_ben_spent_more_total_spent_is_48_l3398_339847


namespace NUMINAMATH_CALUDE_jack_and_jill_meeting_point_l3398_339872

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

end NUMINAMATH_CALUDE_jack_and_jill_meeting_point_l3398_339872


namespace NUMINAMATH_CALUDE_min_gumballs_for_four_is_eleven_l3398_339836

/-- Represents the number of gumballs of each color in the machine -/
structure GumballMachine where
  red : Nat
  white : Nat
  blue : Nat

/-- Represents the minimum number of gumballs needed to guarantee four of the same color -/
def minGumballsForFour (machine : GumballMachine) : Nat :=
  11

/-- Theorem stating that for the given gumball machine, 11 is the minimum number of gumballs 
    needed to guarantee four of the same color -/
theorem min_gumballs_for_four_is_eleven (machine : GumballMachine) 
    (h1 : machine.red = 10) 
    (h2 : machine.white = 7) 
    (h3 : machine.blue = 6) : 
  minGumballsForFour machine = 11 := by
  sorry

#eval minGumballsForFour { red := 10, white := 7, blue := 6 }

end NUMINAMATH_CALUDE_min_gumballs_for_four_is_eleven_l3398_339836


namespace NUMINAMATH_CALUDE_tea_cost_price_l3398_339818

/-- The cost price of 80 kg of tea per kg -/
def C : ℝ := 15

/-- The theorem stating the cost price of 80 kg of tea per kg -/
theorem tea_cost_price :
  -- 80 kg of tea is mixed with 20 kg of tea at cost price of 20 per kg
  -- The sale price of the mixed tea is 20 per kg
  -- The trader wants to earn a profit of 25%
  (80 * C + 20 * 20) * 1.25 = 100 * 20 :=
by
  sorry

end NUMINAMATH_CALUDE_tea_cost_price_l3398_339818


namespace NUMINAMATH_CALUDE_sandy_puppies_l3398_339873

def total_puppies (initial : ℝ) (additional : ℝ) : ℝ :=
  initial + additional

theorem sandy_puppies : total_puppies 8 4 = 12 := by
  sorry

end NUMINAMATH_CALUDE_sandy_puppies_l3398_339873


namespace NUMINAMATH_CALUDE_multiplication_result_l3398_339865

theorem multiplication_result : (935421 * 625) = 584638125 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_result_l3398_339865


namespace NUMINAMATH_CALUDE_reverse_divisibility_implies_divides_99_l3398_339800

-- Define a function to reverse the digits of a natural number
def reverse_digits (n : ℕ) : ℕ := sorry

-- Define the property of k
def has_reverse_divisibility_property (k : ℕ) : Prop :=
  ∀ n : ℕ, k ∣ n → k ∣ reverse_digits n

-- Theorem statement
theorem reverse_divisibility_implies_divides_99 (k : ℕ) :
  has_reverse_divisibility_property k → k ∣ 99 := by sorry

end NUMINAMATH_CALUDE_reverse_divisibility_implies_divides_99_l3398_339800


namespace NUMINAMATH_CALUDE_three_times_work_days_l3398_339842

/-- The number of days Aarti needs to complete one piece of work -/
def base_work_days : ℕ := 9

/-- The number of times the work is multiplied -/
def work_multiplier : ℕ := 3

/-- Theorem: The time required to complete three times the work is 27 days -/
theorem three_times_work_days : base_work_days * work_multiplier = 27 := by
  sorry

end NUMINAMATH_CALUDE_three_times_work_days_l3398_339842


namespace NUMINAMATH_CALUDE_age_difference_l3398_339881

theorem age_difference : ∀ (a b : ℕ), 
  (a < 10 ∧ b < 10) →  -- a and b are single digits
  (10 * a + b + 5 = 3 * (10 * b + a + 5)) →  -- In 5 years, Rachel's age will be three times Sam's age
  ((10 * a + b) - (10 * b + a) = 63) :=  -- The difference in their current ages is 63
by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3398_339881


namespace NUMINAMATH_CALUDE_min_distance_to_line_l3398_339879

theorem min_distance_to_line (x y : ℝ) :
  8 * x + 15 * y = 120 →
  ∃ (min_val : ℝ), min_val = 120 / 17 ∧
    ∀ (x' y' : ℝ), 8 * x' + 15 * y' = 120 →
      Real.sqrt (x'^2 + y'^2) ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l3398_339879


namespace NUMINAMATH_CALUDE_min_stamps_for_50_cents_l3398_339876

/-- Represents the number of stamps of each denomination -/
structure StampCombination where
  threes : ℕ
  fours : ℕ
  sixes : ℕ

/-- Calculates the total value of stamps in cents -/
def totalValue (s : StampCombination) : ℕ :=
  3 * s.threes + 4 * s.fours + 6 * s.sixes

/-- Calculates the total number of stamps -/
def totalStamps (s : StampCombination) : ℕ :=
  s.threes + s.fours + s.sixes

/-- Checks if a stamp combination is valid (totals 50 cents) -/
def isValid (s : StampCombination) : Prop :=
  totalValue s = 50

/-- Theorem: The minimum number of stamps to make 50 cents is 10 -/
theorem min_stamps_for_50_cents :
  (∃ (s : StampCombination), isValid s ∧ totalStamps s = 10) ∧
  (∀ (s : StampCombination), isValid s → totalStamps s ≥ 10) :=
by sorry

end NUMINAMATH_CALUDE_min_stamps_for_50_cents_l3398_339876


namespace NUMINAMATH_CALUDE_fourth_term_is_sixty_l3398_339892

/-- Represents a stratified sample drawn from an arithmetic sequence of questionnaires. -/
structure StratifiedSample where
  total_questionnaires : ℕ
  sample_size : ℕ
  second_term : ℕ
  h_total : total_questionnaires = 1000
  h_sample : sample_size = 150
  h_second : second_term = 30

/-- The number of questionnaires drawn from the fourth term of the sequence. -/
def fourth_term (s : StratifiedSample) : ℕ := 60

/-- Theorem stating that the fourth term of the stratified sample is 60. -/
theorem fourth_term_is_sixty (s : StratifiedSample) : fourth_term s = 60 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_is_sixty_l3398_339892


namespace NUMINAMATH_CALUDE_circle_equation_l3398_339813

/-- The equation of a circle passing through (0,0), (4,0), and (-1,1) -/
theorem circle_equation (x y : ℝ) : 
  (x^2 + y^2 - 4*x - 6*y = 0) ↔ 
  ((x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1)) := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_l3398_339813


namespace NUMINAMATH_CALUDE_modified_car_distance_increase_l3398_339820

/-- Proves the increase in travel distance after modifying a car's fuel efficiency -/
theorem modified_car_distance_increase
  (original_efficiency : ℝ)
  (tank_capacity : ℝ)
  (fuel_reduction_factor : ℝ)
  (h1 : original_efficiency = 32)
  (h2 : tank_capacity = 12)
  (h3 : fuel_reduction_factor = 0.8)
  : (tank_capacity * (original_efficiency / fuel_reduction_factor) - tank_capacity * original_efficiency) = 76.8 := by
  sorry

end NUMINAMATH_CALUDE_modified_car_distance_increase_l3398_339820


namespace NUMINAMATH_CALUDE_compare_x_y_z_l3398_339833

open Real

theorem compare_x_y_z (x y z : ℝ) (hx : x = log π) (hy : y = log 2 / log 5) (hz : z = exp (-1/2)) :
  y < z ∧ z < x := by sorry

end NUMINAMATH_CALUDE_compare_x_y_z_l3398_339833


namespace NUMINAMATH_CALUDE_farm_land_allocation_l3398_339856

theorem farm_land_allocation (total_land : ℕ) (reserved : ℕ) (cattle : ℕ) (crops : ℕ) 
  (h1 : total_land = 150)
  (h2 : reserved = 15)
  (h3 : cattle = 40)
  (h4 : crops = 70) :
  total_land - reserved - cattle - crops = 25 := by
  sorry

end NUMINAMATH_CALUDE_farm_land_allocation_l3398_339856


namespace NUMINAMATH_CALUDE_min_sum_squares_l3398_339878

/-- Given five real numbers satisfying certain conditions, 
    the sum of their squares has a minimum value. -/
theorem min_sum_squares (a₁ a₂ a₃ a₄ a₅ : ℝ) 
    (h1 : a₁*a₂ + a₂*a₃ + a₃*a₄ + a₄*a₅ + a₅*a₁ = 20)
    (h2 : a₁*a₃ + a₂*a₄ + a₃*a₅ + a₄*a₁ + a₅*a₂ = 22) :
    ∃ (m : ℝ), m = a₁^2 + a₂^2 + a₃^2 + a₄^2 + a₅^2 ∧ 
    ∀ (b₁ b₂ b₃ b₄ b₅ : ℝ), 
    (b₁*b₂ + b₂*b₃ + b₃*b₄ + b₄*b₅ + b₅*b₁ = 20) →
    (b₁*b₃ + b₂*b₄ + b₃*b₅ + b₄*b₁ + b₅*b₂ = 22) →
    m ≤ b₁^2 + b₂^2 + b₃^2 + b₄^2 + b₅^2 ∧
    m = 21 + Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_l3398_339878


namespace NUMINAMATH_CALUDE_pythagorean_triple_double_l3398_339839

theorem pythagorean_triple_double (a b c : ℤ) :
  (a^2 + b^2 = c^2) → ((2*a)^2 + (2*b)^2 = (2*c)^2) := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_triple_double_l3398_339839


namespace NUMINAMATH_CALUDE_simplify_expression_l3398_339883

theorem simplify_expression : 
  ((3 + 5 + 6 + 2) / 3) + ((2 * 3 + 4 * 2 + 5) / 4) = 121 / 12 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3398_339883


namespace NUMINAMATH_CALUDE_fencing_cost_per_meter_l3398_339890

/-- Proves that for a rectangular plot with given dimensions and total fencing cost,
    the cost per meter of fencing is as calculated. -/
theorem fencing_cost_per_meter
  (length : ℝ) (breadth : ℝ) (total_cost : ℝ)
  (h1 : length = 60)
  (h2 : breadth = 40)
  (h3 : total_cost = 5300) :
  total_cost / (2 * (length + breadth)) = 26.5 := by
  sorry

end NUMINAMATH_CALUDE_fencing_cost_per_meter_l3398_339890


namespace NUMINAMATH_CALUDE_cistern_wet_surface_area_l3398_339880

/-- Calculates the total wet surface area of a rectangular cistern -/
def total_wet_surface_area (length width depth : Real) : Real :=
  length * width + 2 * length * depth + 2 * width * depth

/-- Theorem stating the total wet surface area of a specific cistern -/
theorem cistern_wet_surface_area :
  total_wet_surface_area 8 6 1.85 = 99.8 := by
  sorry

end NUMINAMATH_CALUDE_cistern_wet_surface_area_l3398_339880


namespace NUMINAMATH_CALUDE_problem_solution_l3398_339877

theorem problem_solution (x y : ℚ) : 
  x / y = 12 / 5 → y = 25 → x = 60 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l3398_339877


namespace NUMINAMATH_CALUDE_expression_value_at_four_l3398_339808

theorem expression_value_at_four :
  let x : ℚ := 4
  (x^2 - 3*x - 10) / (x - 5) = 6 := by sorry

end NUMINAMATH_CALUDE_expression_value_at_four_l3398_339808


namespace NUMINAMATH_CALUDE_ratio_of_linear_system_l3398_339832

theorem ratio_of_linear_system (x y c d : ℝ) 
  (eq1 : 9 * x - 6 * y = c)
  (eq2 : 15 * x - 10 * y = d)
  (h1 : d ≠ 0)
  (h2 : x ≠ 0)
  (h3 : y ≠ 0) :
  c / d = -2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_linear_system_l3398_339832


namespace NUMINAMATH_CALUDE_equation_solution_l3398_339826

theorem equation_solution :
  let f : ℝ → ℝ := λ x => -x^2 * (x + 5) - (5*x - 2)
  ∀ x : ℝ, f x = 0 ↔ x = -2 ∨ x = (-3 + Real.sqrt 5) / 2 ∨ x = (-3 - Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3398_339826


namespace NUMINAMATH_CALUDE_number_square_puzzle_l3398_339807

theorem number_square_puzzle : ∃ x : ℝ, x^2 + 95 = (x - 20)^2 ∧ x = 7.625 := by
  sorry

end NUMINAMATH_CALUDE_number_square_puzzle_l3398_339807


namespace NUMINAMATH_CALUDE_tangent_line_value_l3398_339894

/-- Proves that if a line is tangent to both y = ln x and x² = ay at the same point, then a = 2e -/
theorem tangent_line_value (a : ℝ) (h : a > 0) : 
  (∃ (x y : ℝ), x > 0 ∧ 
    y = Real.log x ∧ 
    x^2 = a * y ∧ 
    (1 / x) = (2 / a) * x) → 
  a = 2 * Real.exp 1 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_value_l3398_339894


namespace NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l3398_339860

theorem smallest_number_with_given_remainders : 
  ∃ (a : ℕ), a = 74 ∧ 
  (∀ (n : ℕ), n < a → 
    (n % 3 ≠ 2 ∨ n % 5 ≠ 4 ∨ n % 7 ≠ 4)) ∧
  74 % 3 = 2 ∧ 74 % 5 = 4 ∧ 74 % 7 = 4 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l3398_339860


namespace NUMINAMATH_CALUDE_max_perimeter_is_29_l3398_339867

/-- Represents a triangle with two fixed sides of length 7 and 8, and a variable third side y --/
structure Triangle where
  y : ℤ
  is_valid : 0 < y ∧ y < 7 + 8 ∧ 7 < y + 8 ∧ 8 < y + 7

/-- The perimeter of the triangle --/
def perimeter (t : Triangle) : ℤ := 7 + 8 + t.y

/-- Theorem stating that the maximum perimeter is 29 --/
theorem max_perimeter_is_29 :
  ∀ t : Triangle, perimeter t ≤ 29 ∧ ∃ t' : Triangle, perimeter t' = 29 := by
  sorry

#check max_perimeter_is_29

end NUMINAMATH_CALUDE_max_perimeter_is_29_l3398_339867


namespace NUMINAMATH_CALUDE_alice_bracelet_profit_l3398_339863

/-- Calculates Alice's profit from selling friendship bracelets -/
theorem alice_bracelet_profit :
  let total_design_a : ℕ := 30
  let total_design_b : ℕ := 22
  let cost_design_a : ℚ := 2
  let cost_design_b : ℚ := 4.5
  let given_away_design_a : ℕ := 5
  let given_away_design_b : ℕ := 3
  let bulk_price_design_a : ℚ := 0.2
  let bulk_price_design_b : ℚ := 0.4
  let total_cost := total_design_a * cost_design_a + total_design_b * cost_design_b
  let remaining_design_a := total_design_a - given_away_design_a
  let remaining_design_b := total_design_b - given_away_design_b
  let total_revenue := remaining_design_a * bulk_price_design_a + remaining_design_b * bulk_price_design_b
  let profit := total_revenue - total_cost
  profit = -146.4 := by sorry

end NUMINAMATH_CALUDE_alice_bracelet_profit_l3398_339863


namespace NUMINAMATH_CALUDE_zeros_of_f_l3398_339891

def f (x : ℝ) : ℝ := x^3 - 2*x^2 - x + 2

theorem zeros_of_f :
  ∀ x : ℝ, f x = 0 ↔ x = -1 ∨ x = 1 ∨ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_zeros_of_f_l3398_339891


namespace NUMINAMATH_CALUDE_x_varies_as_sixth_power_of_z_l3398_339854

/-- If x varies as the square of y, and y varies as the cube of z,
    then x varies as the 6th power of z. -/
theorem x_varies_as_sixth_power_of_z
  (x y z : ℝ)
  (k j : ℝ)
  (h1 : x = k * y^2)
  (h2 : y = j * z^3) :
  ∃ m : ℝ, x = m * z^6 := by
sorry

end NUMINAMATH_CALUDE_x_varies_as_sixth_power_of_z_l3398_339854


namespace NUMINAMATH_CALUDE_subway_fare_cost_l3398_339805

/-- The cost of the subway fare each way given Brian's spending and constraints -/
theorem subway_fare_cost (apple_cost : ℚ) (kiwi_cost : ℚ) (banana_cost : ℚ) 
  (initial_money : ℚ) (max_apples : ℕ) :
  apple_cost = 14 / 12 →
  kiwi_cost = 10 →
  banana_cost = 5 →
  initial_money = 50 →
  max_apples = 24 →
  (initial_money - kiwi_cost - banana_cost - (↑max_apples * apple_cost)) / 2 = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_subway_fare_cost_l3398_339805


namespace NUMINAMATH_CALUDE_name_is_nika_l3398_339849

-- Define a cube face
inductive Face
| Front
| Back
| Left
| Right
| Top
| Bottom

-- Define a letter
inductive Letter
| N
| I
| K
| A
| T

-- Define a cube
structure Cube where
  faces : Face → Letter

-- Define the arrangement of cubes
def CubeArrangement := List Cube

-- Define the function to get the front-facing letters
def getFrontLetters (arrangement : CubeArrangement) : List Letter :=
  arrangement.map (λ cube => cube.faces Face.Front)

-- Theorem statement
theorem name_is_nika (arrangement : CubeArrangement) 
  (h1 : arrangement.length = 4)
  (h2 : getFrontLetters arrangement = [Letter.N, Letter.I, Letter.K, Letter.A]) :
  "Ника" = "Ника" :=
by sorry

end NUMINAMATH_CALUDE_name_is_nika_l3398_339849


namespace NUMINAMATH_CALUDE_optimal_bus_rental_l3398_339835

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

end NUMINAMATH_CALUDE_optimal_bus_rental_l3398_339835


namespace NUMINAMATH_CALUDE_probability_under_20_l3398_339819

theorem probability_under_20 (total : ℕ) (over_30 : ℕ) (under_20 : ℕ) :
  total = 100 →
  over_30 = 90 →
  under_20 = total - over_30 →
  (under_20 : ℚ) / (total : ℚ) = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_under_20_l3398_339819


namespace NUMINAMATH_CALUDE_tan_alpha_value_l3398_339840

theorem tan_alpha_value (α : ℝ) (h : (Real.sin α - 2 * Real.cos α) / (Real.sin α + Real.cos α) = -1) :
  Real.tan α = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l3398_339840


namespace NUMINAMATH_CALUDE_power_difference_equals_multiple_of_thirty_power_l3398_339875

theorem power_difference_equals_multiple_of_thirty_power : 
  (5^1002 + 6^1001)^2 - (5^1002 - 6^1001)^2 = 24 * 30^1001 := by
sorry

end NUMINAMATH_CALUDE_power_difference_equals_multiple_of_thirty_power_l3398_339875


namespace NUMINAMATH_CALUDE_distinct_prime_factors_count_l3398_339823

def product : ℕ := 77 * 79 * 81 * 83

theorem distinct_prime_factors_count :
  (Nat.factors product).toFinset.card = 5 := by sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_count_l3398_339823


namespace NUMINAMATH_CALUDE_triangular_prism_is_pentahedron_l3398_339822

-- Define the polyhedra types
inductive Polyhedron
| TriangularPyramid
| TriangularPrism
| QuadrangularPrism
| PentagonalPyramid

-- Define the function that returns the number of faces for each polyhedron
def numFaces (p : Polyhedron) : Nat :=
  match p with
  | Polyhedron.TriangularPyramid => 4    -- tetrahedron
  | Polyhedron.TriangularPrism => 5      -- pentahedron
  | Polyhedron.QuadrangularPrism => 6    -- hexahedron
  | Polyhedron.PentagonalPyramid => 6    -- hexahedron

-- Theorem: A triangular prism is a pentahedron
theorem triangular_prism_is_pentahedron :
  numFaces Polyhedron.TriangularPrism = 5 := by sorry

end NUMINAMATH_CALUDE_triangular_prism_is_pentahedron_l3398_339822


namespace NUMINAMATH_CALUDE_seven_awards_four_students_l3398_339809

/-- The number of ways to distribute n different awards among k students,
    where each student receives at least one award. -/
def distribute_awards (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem stating that distributing 7 awards among 4 students results in 920 ways -/
theorem seven_awards_four_students :
  distribute_awards 7 4 = 920 := by sorry

end NUMINAMATH_CALUDE_seven_awards_four_students_l3398_339809


namespace NUMINAMATH_CALUDE_siblings_ages_l3398_339838

-- Define the ages of the siblings
def hans_age : ℕ := 8
def annika_age : ℕ := 25
def emil_age : ℕ := 5
def frida_age : ℕ := 20

-- Define the conditions
def condition1 : Prop := annika_age + 4 = 3 * (hans_age + 4)
def condition2 : Prop := emil_age + 4 = 2 * (hans_age + 4)
def condition3 : Prop := (emil_age + 4) - (hans_age + 4) = (frida_age + 4) / 2
def condition4 : Prop := hans_age + annika_age + emil_age + frida_age = 58
def condition5 : Prop := frida_age + 5 = annika_age

-- Theorem statement
theorem siblings_ages :
  condition1 ∧ condition2 ∧ condition3 ∧ condition4 ∧ condition5 →
  annika_age = 25 ∧ emil_age = 5 ∧ frida_age = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_siblings_ages_l3398_339838


namespace NUMINAMATH_CALUDE_ratio_sum_theorem_l3398_339893

theorem ratio_sum_theorem (a b c d : ℝ) : 
  b = 2 * a ∧ c = 4 * a ∧ d = 5 * a ∧ 
  a^2 + b^2 + c^2 + d^2 = 2460 →
  abs ((a + b + c + d) - 87.744) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ratio_sum_theorem_l3398_339893


namespace NUMINAMATH_CALUDE_roger_used_crayons_l3398_339803

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

end NUMINAMATH_CALUDE_roger_used_crayons_l3398_339803


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_reciprocal_l3398_339812

theorem quadratic_roots_sum_reciprocal (a b : ℝ) : 
  a^2 - 6*a - 5 = 0 → 
  b^2 - 6*b - 5 = 0 → 
  a ≠ 0 → 
  b ≠ 0 → 
  1/a + 1/b = -6/5 := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_reciprocal_l3398_339812


namespace NUMINAMATH_CALUDE_percentage_problem_l3398_339868

theorem percentage_problem : 
  ∃ p : ℝ, p * 24 = 0.12 ∧ p = 0.005 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3398_339868


namespace NUMINAMATH_CALUDE_locus_of_points_m_l3398_339816

-- Define the given circle
structure GivenCircle where
  O : ℝ × ℝ  -- Center of the circle
  R : ℝ      -- Radius of the circle
  h : R > 0  -- Radius is positive

-- Define the point A on the given circle
def PointOnCircle (c : GivenCircle) (A : ℝ × ℝ) : Prop :=
  (A.1 - c.O.1)^2 + (A.2 - c.O.2)^2 = c.R^2

-- Define the tangent line at point A
def TangentLine (c : GivenCircle) (A : ℝ × ℝ) : ℝ × ℝ → Prop :=
  λ M => (M.1 - A.1) * (A.1 - c.O.1) + (M.2 - A.2) * (A.2 - c.O.2) = 0

-- Define the segment AM with length a
def SegmentAM (A M : ℝ × ℝ) (a : ℝ) : Prop :=
  (M.1 - A.1)^2 + (M.2 - A.2)^2 = a^2

-- Theorem: The locus of points M forms a circle concentric with the given circle
theorem locus_of_points_m (c : GivenCircle) (a : ℝ) (h : a > 0) :
  ∀ A M : ℝ × ℝ,
    PointOnCircle c A →
    TangentLine c A M →
    SegmentAM A M a →
    (M.1 - c.O.1)^2 + (M.2 - c.O.2)^2 = c.R^2 + a^2 :=
  sorry

end NUMINAMATH_CALUDE_locus_of_points_m_l3398_339816


namespace NUMINAMATH_CALUDE_sum_of_roots_equals_one_l3398_339852

theorem sum_of_roots_equals_one :
  let f : ℝ → ℝ := λ x => (x + 3) * (x - 4) - 21
  ∃ x₁ x₂ : ℝ, (f x₁ = 0 ∧ f x₂ = 0) ∧ x₁ + x₂ = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_equals_one_l3398_339852


namespace NUMINAMATH_CALUDE_p_true_and_q_false_l3398_339831

-- Define proposition p
def p : Prop := ∀ x y : ℝ, x > y → -x < -y

-- Define proposition q
def q : Prop := ∀ x y : ℝ, x ≠ 0 ∧ y ≠ 0 → (1 / x > 1 / y → x < y)

-- Theorem to prove
theorem p_true_and_q_false : p ∧ ¬q := by
  sorry

end NUMINAMATH_CALUDE_p_true_and_q_false_l3398_339831


namespace NUMINAMATH_CALUDE_negation_of_universal_positive_square_not_equal_l3398_339871

theorem negation_of_universal_positive_square_not_equal (P : ℝ → Prop) :
  (¬ ∀ x : ℝ, x > 0 → x^2 ≠ x) ↔ (∃ x : ℝ, x > 0 ∧ x^2 = x) :=
sorry

end NUMINAMATH_CALUDE_negation_of_universal_positive_square_not_equal_l3398_339871


namespace NUMINAMATH_CALUDE_min_value_complex_condition_l3398_339804

theorem min_value_complex_condition (x y : ℝ) :
  Complex.abs (Complex.mk x y - Complex.I * 4) = Complex.abs (Complex.mk x y + 2) →
  2^x + 4^y ≥ 4 * Real.sqrt 2 ∧ 
  ∃ (x₀ y₀ : ℝ), Complex.abs (Complex.mk x₀ y₀ - Complex.I * 4) = Complex.abs (Complex.mk x₀ y₀ + 2) ∧
                 2^x₀ + 4^y₀ = 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_complex_condition_l3398_339804


namespace NUMINAMATH_CALUDE_line_segment_length_l3398_339815

/-- Given two points M(-2, a) and N(a, 4) on a line with slope -1/2,
    prove that the distance between M and N is 6√3. -/
theorem line_segment_length (a : ℝ) : 
  (4 - a) / (a + 2) = -1/2 →
  Real.sqrt ((a + 2)^2 + (4 - a)^2) = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_length_l3398_339815


namespace NUMINAMATH_CALUDE_jake_final_bitcoins_l3398_339841

/-- Represents the number of bitcoins Jake has at each step -/
def bitcoin_transactions (initial : ℕ) (donation1 : ℕ) (donation2 : ℕ) : ℕ :=
  let after_donation1 := initial - donation1
  let after_brother := after_donation1 / 2
  let after_triple := after_brother * 3
  after_triple - donation2

/-- Theorem stating that Jake ends up with 80 bitcoins after all transactions -/
theorem jake_final_bitcoins :
  bitcoin_transactions 80 20 10 = 80 := by
  sorry

end NUMINAMATH_CALUDE_jake_final_bitcoins_l3398_339841


namespace NUMINAMATH_CALUDE_reading_time_difference_l3398_339837

/-- Proves that the difference in reading time between Lee and Kai is 150 minutes -/
theorem reading_time_difference 
  (kai_speed : ℝ) 
  (lee_speed : ℝ) 
  (book_pages : ℝ) 
  (h1 : kai_speed = 120) 
  (h2 : lee_speed = 60) 
  (h3 : book_pages = 300) : 
  (book_pages / lee_speed - book_pages / kai_speed) * 60 = 150 := by
  sorry

end NUMINAMATH_CALUDE_reading_time_difference_l3398_339837


namespace NUMINAMATH_CALUDE_line_parameterization_specific_line_parameterization_l3398_339897

/-- A parameterization of a line is valid if it satisfies the line equation and has a correct direction vector -/
def IsValidParameterization (a b : ℝ) (p : ℝ × ℝ) (v : ℝ × ℝ) : Prop :=
  (∀ t, (p.1 + t * v.1, p.2 + t * v.2) ∈ {(x, y) | y = a * x + b}) ∧
  ∃ k ≠ 0, v = (k, a * k)

theorem line_parameterization (a b : ℝ) :
  let line := fun x y ↦ y = a * x + b
  IsValidParameterization a b (1, 8) (1, 3) ∧
  IsValidParameterization a b (2, 11) (-1/3, -1) ∧
  IsValidParameterization a b (-1.5, 0.5) (1, 3) ∧
  ¬IsValidParameterization a b (-5/3, 0) (3, 9) ∧
  ¬IsValidParameterization a b (0, 5) (6, 2) :=
by sorry

/-- The specific line y = 3x + 5 has valid parameterizations A, D, and E, but not B and C -/
theorem specific_line_parameterization :
  let line := fun x y ↦ y = 3 * x + 5
  IsValidParameterization 3 5 (1, 8) (1, 3) ∧
  IsValidParameterization 3 5 (2, 11) (-1/3, -1) ∧
  IsValidParameterization 3 5 (-1.5, 0.5) (1, 3) ∧
  ¬IsValidParameterization 3 5 (-5/3, 0) (3, 9) ∧
  ¬IsValidParameterization 3 5 (0, 5) (6, 2) :=
by sorry

end NUMINAMATH_CALUDE_line_parameterization_specific_line_parameterization_l3398_339897


namespace NUMINAMATH_CALUDE_problem_1_l3398_339886

theorem problem_1 : Real.sqrt 32 - 3 * Real.sqrt (1/2) + Real.sqrt 2 = (7/2) * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l3398_339886


namespace NUMINAMATH_CALUDE_inequality_proof_l3398_339853

theorem inequality_proof (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (hmax : d = max a (max b c)) : 
  a * (d - b) + b * (d - c) + c * (d - a) ≤ d^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3398_339853


namespace NUMINAMATH_CALUDE_tangent_line_at_zero_zero_conditions_l3398_339828

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (1 + x) + a * x * Real.exp (-x)

-- Part 1: Tangent line when a = 1
theorem tangent_line_at_zero (x : ℝ) :
  ∃ m b : ℝ, (deriv (f 1)) 0 = m ∧ f 1 0 = b ∧ m = 2 ∧ b = 0 := by sorry

-- Part 2: Range of a for exactly one zero in each interval
theorem zero_conditions (a : ℝ) :
  (∃! x : ℝ, x ∈ Set.Ioo (-1) 0 ∧ f a x = 0) ∧
  (∃! x : ℝ, x ∈ Set.Ioi 0 ∧ f a x = 0) ↔
  a ∈ Set.Iio (-1) := by sorry

end NUMINAMATH_CALUDE_tangent_line_at_zero_zero_conditions_l3398_339828


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3398_339874

theorem complex_fraction_simplification :
  (1 + 2*Complex.I) / (3 - 4*Complex.I) = -1/5 + (2/5)*Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3398_339874


namespace NUMINAMATH_CALUDE_accidental_addition_correction_l3398_339811

theorem accidental_addition_correction (x : ℤ) (h : x + 5 = 43) : 5 * x = 190 := by
  sorry

end NUMINAMATH_CALUDE_accidental_addition_correction_l3398_339811


namespace NUMINAMATH_CALUDE_marble_weight_problem_l3398_339844

theorem marble_weight_problem (weight_piece1 weight_piece2 total_weight : ℝ) 
  (h1 : weight_piece1 = 0.33)
  (h2 : weight_piece2 = 0.33)
  (h3 : total_weight = 0.75) :
  total_weight - (weight_piece1 + weight_piece2) = 0.09 := by
  sorry

end NUMINAMATH_CALUDE_marble_weight_problem_l3398_339844


namespace NUMINAMATH_CALUDE_points_difference_l3398_339802

def basketball_game (jon_points jack_points tom_points : ℕ) : Prop :=
  (jack_points = jon_points + 5) ∧
  (jon_points + jack_points + tom_points = 18) ∧
  (tom_points < jon_points + jack_points)

theorem points_difference (jon_points jack_points tom_points : ℕ) :
  basketball_game jon_points jack_points tom_points →
  jon_points = 3 →
  (jon_points + jack_points) - tom_points = 4 := by
  sorry

end NUMINAMATH_CALUDE_points_difference_l3398_339802


namespace NUMINAMATH_CALUDE_power_of_power_l3398_339861

theorem power_of_power (a : ℝ) : (a^4)^2 = a^8 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l3398_339861


namespace NUMINAMATH_CALUDE_triangle_side_length_l3398_339898

theorem triangle_side_length 
  (A B C : ℝ) 
  (a b c : ℝ) 
  (h_area : (1/2) * a * c * (Real.sqrt 3 / 2) = Real.sqrt 3)
  (h_angle : B = 60 * π / 180)
  (h_sides : a^2 + c^2 = 3*a*c) :
  b = 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3398_339898


namespace NUMINAMATH_CALUDE_subtraction_puzzle_l3398_339843

theorem subtraction_puzzle (A B : Nat) : 
  A ≠ B → 
  A < 10 → 
  B < 10 → 
  2000 + 100 * A + 32 - (100 * B + 10 * B + B) = 1000 + 100 * B + 10 * B + B → 
  B - A = 3 := by
sorry

end NUMINAMATH_CALUDE_subtraction_puzzle_l3398_339843


namespace NUMINAMATH_CALUDE_f_at_negative_two_equals_six_l3398_339825

-- Define the functions f and g
def f (a b c x : ℝ) : ℝ := a * x^2 + 2 * b * x + c
def g (a b c x : ℝ) : ℝ := (a + 1) * x^2 + 2 * (b + 2) * x + (c + 4)

-- Define the discriminant function
def discriminant (a b c : ℝ) : ℝ := 4 * b^2 - 4 * a * c

-- State the theorem
theorem f_at_negative_two_equals_six (a b c : ℝ) :
  discriminant a b c - discriminant (a + 1) (b + 2) (c + 4) = 24 →
  f a b c (-2) = 6 := by
  sorry


end NUMINAMATH_CALUDE_f_at_negative_two_equals_six_l3398_339825


namespace NUMINAMATH_CALUDE_sculpture_third_week_cut_percentage_l3398_339870

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

end NUMINAMATH_CALUDE_sculpture_third_week_cut_percentage_l3398_339870


namespace NUMINAMATH_CALUDE_circle_symmetry_theorem_l3398_339846

/-- The equation of the circle C -/
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 + 4*x + a*y - 5 = 0

/-- The equation of the line -/
def line_equation (x y : ℝ) : Prop :=
  2*x + y - 1 = 0

/-- The theorem stating the relationship between the circle and the line -/
theorem circle_symmetry_theorem (a : ℝ) : 
  (∀ x y : ℝ, circle_equation x y a → 
    ∃ x' y' : ℝ, circle_equation x' y' a ∧ 
    ((x + x')/2, (y + y')/2) ∈ {(x, y) | line_equation x y}) →
  a = -10 := by
  sorry


end NUMINAMATH_CALUDE_circle_symmetry_theorem_l3398_339846


namespace NUMINAMATH_CALUDE_parabola_focus_on_line_l3398_339885

-- Define the line on which the focus lies
def focus_line (x y : ℝ) : Prop := x - 2*y - 4 = 0

-- Define the standard equations of parabolas
def parabola_eq1 (x y : ℝ) : Prop := y^2 = 16*x
def parabola_eq2 (x y : ℝ) : Prop := x^2 = -8*y

-- Theorem statement
theorem parabola_focus_on_line :
  ∀ (x y : ℝ), focus_line x y →
  (∃ (a b : ℝ), parabola_eq1 a b) ∨ (∃ (c d : ℝ), parabola_eq2 c d) :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_on_line_l3398_339885


namespace NUMINAMATH_CALUDE_five_lines_sixteen_sections_l3398_339857

/-- The number of sections created by drawing n properly intersecting line segments in a rectangle --/
def sections (n : ℕ) : ℕ :=
  if n = 0 then 1 else sections (n - 1) + n

/-- Theorem: Drawing 5 properly intersecting line segments in a rectangle creates 16 sections --/
theorem five_lines_sixteen_sections : sections 5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_five_lines_sixteen_sections_l3398_339857


namespace NUMINAMATH_CALUDE_symmetric_distribution_theorem_l3398_339829

/-- A symmetric distribution with mean m and standard deviation d. -/
structure SymmetricDistribution where
  m : ℝ  -- mean
  d : ℝ  -- standard deviation
  symmetric : Bool
  within_one_std_dev : ℝ

/-- The percentage of the distribution less than m + d -/
def percent_less_than_m_plus_d (dist : SymmetricDistribution) : ℝ := sorry

theorem symmetric_distribution_theorem (dist : SymmetricDistribution) 
  (h_symmetric : dist.symmetric = true)
  (h_within_one_std_dev : dist.within_one_std_dev = 84) :
  percent_less_than_m_plus_d dist = 42 := by sorry

end NUMINAMATH_CALUDE_symmetric_distribution_theorem_l3398_339829


namespace NUMINAMATH_CALUDE_a_range_l3398_339866

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∃ x : ℝ, x ∈ Set.Icc 0 1 ∧ a ≤ Real.exp x

def q (a : ℝ) : Prop := ∀ x : ℝ, x^2 + x + a > 0

-- State the theorem
theorem a_range (a : ℝ) (h : p a ∧ q a) : 1/4 < a ∧ a ≤ Real.exp 1 := by
  sorry


end NUMINAMATH_CALUDE_a_range_l3398_339866


namespace NUMINAMATH_CALUDE_volume_of_T_l3398_339899

/-- The solid T in ℝ³ -/
def T : Set (ℝ × ℝ × ℝ) :=
  {p | let (x, y, z) := p
       (|x| + |y| ≤ 2) ∧ (|x| + |z| ≤ 2) ∧ (|y| + |z| ≤ 2)}

/-- The volume of a set in ℝ³ -/
noncomputable def volume (S : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

/-- The volume of T is 64/9 -/
theorem volume_of_T : volume T = 64/9 := by sorry

end NUMINAMATH_CALUDE_volume_of_T_l3398_339899


namespace NUMINAMATH_CALUDE_gardner_brownies_l3398_339887

theorem gardner_brownies :
  ∀ (students cookies cupcakes brownies total_treats : ℕ),
    students = 20 →
    cookies = 20 →
    cupcakes = 25 →
    total_treats = students * 4 →
    total_treats = cookies + cupcakes + brownies →
    brownies = 35 := by
  sorry

end NUMINAMATH_CALUDE_gardner_brownies_l3398_339887


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l3398_339801

theorem quadratic_inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, 1 < x ∧ x < 4 ∧ 2 * x^2 - 8 * x - 4 - a > 0) ↔ 
  (-16 < a ∧ a < -8) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l3398_339801


namespace NUMINAMATH_CALUDE_hexagon_walk_distance_hexagon_walk_distance_proof_l3398_339850

/-- The distance of a point from its starting position after moving 7 km along the perimeter of a regular hexagon with side length 3 km -/
theorem hexagon_walk_distance : ℝ :=
  let side_length : ℝ := 3
  let walk_distance : ℝ := 7
  let hexagon_angle : ℝ := 2 * Real.pi / 6
  let end_position : ℝ × ℝ := (1, Real.sqrt 3)
  2

theorem hexagon_walk_distance_proof :
  hexagon_walk_distance = 2 := by sorry

end NUMINAMATH_CALUDE_hexagon_walk_distance_hexagon_walk_distance_proof_l3398_339850


namespace NUMINAMATH_CALUDE_tyson_basketball_scores_l3398_339889

/-- Represents the number of times Tyson scored points in each category -/
structure BasketballScores where
  threePointers : Nat
  twoPointers : Nat
  onePointers : Nat

/-- Calculates the total points scored given a BasketballScores structure -/
def totalPoints (scores : BasketballScores) : Nat :=
  3 * scores.threePointers + 2 * scores.twoPointers + scores.onePointers

theorem tyson_basketball_scores :
  ∃ (scores : BasketballScores),
    scores.threePointers = 15 ∧
    scores.twoPointers = 12 ∧
    scores.onePointers % 2 = 0 ∧
    totalPoints scores = 75 ∧
    scores.onePointers = 6 := by
  sorry

end NUMINAMATH_CALUDE_tyson_basketball_scores_l3398_339889


namespace NUMINAMATH_CALUDE_smallest_constant_inequality_l3398_339817

theorem smallest_constant_inequality (x y : ℝ) :
  ∃ (C : ℝ), C = 4/3 ∧ (∀ (x y : ℝ), 1 + (x + y)^2 ≤ C * (1 + x^2) * (1 + y^2)) ∧
  (∀ (D : ℝ), (∀ (x y : ℝ), 1 + (x + y)^2 ≤ D * (1 + x^2) * (1 + y^2)) → C ≤ D) :=
sorry

end NUMINAMATH_CALUDE_smallest_constant_inequality_l3398_339817


namespace NUMINAMATH_CALUDE_john_weight_loss_l3398_339864

/-- The number of calories John burns per day -/
def calories_burned_per_day : ℕ := 2300

/-- The number of calories needed to lose 1 pound -/
def calories_per_pound : ℕ := 4000

/-- The number of days it takes John to lose 10 pounds -/
def days_to_lose_10_pounds : ℕ := 80

/-- The number of pounds John wants to lose -/
def pounds_to_lose : ℕ := 10

/-- The number of calories John eats per day -/
def calories_eaten_per_day : ℕ := 1800

theorem john_weight_loss :
  calories_eaten_per_day =
    calories_burned_per_day -
    (pounds_to_lose * calories_per_pound) / days_to_lose_10_pounds :=
by
  sorry

end NUMINAMATH_CALUDE_john_weight_loss_l3398_339864


namespace NUMINAMATH_CALUDE_domino_distribution_l3398_339859

theorem domino_distribution (total_dominoes : Nat) (num_players : Nat) 
  (h1 : total_dominoes = 28) 
  (h2 : num_players = 4) : 
  total_dominoes / num_players = 7 := by
  sorry

end NUMINAMATH_CALUDE_domino_distribution_l3398_339859


namespace NUMINAMATH_CALUDE_rectangles_4x2_grid_l3398_339806

/-- The number of rectangles that can be formed on a grid of dots -/
def num_rectangles (cols : ℕ) (rows : ℕ) : ℕ :=
  (cols.choose 2) * (rows.choose 2)

/-- Theorem: The number of rectangles on a 4x2 grid is 6 -/
theorem rectangles_4x2_grid : num_rectangles 4 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_rectangles_4x2_grid_l3398_339806


namespace NUMINAMATH_CALUDE_nth_number_in_set_l3398_339882

theorem nth_number_in_set (n : ℕ) : 
  (n + 1) * 19 + 13 = (499 * 19 + 13) → n = 498 := by
  sorry

#check nth_number_in_set

end NUMINAMATH_CALUDE_nth_number_in_set_l3398_339882


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l3398_339830

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x ∈ Set.Ioo 0 2, P x) ↔ (∀ x ∈ Set.Ioo 0 2, ¬P x) :=
by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x ∈ Set.Ioo 0 2, x^2 + 2*x + 2 ≤ 0) ↔
  (∀ x ∈ Set.Ioo 0 2, x^2 + 2*x + 2 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l3398_339830


namespace NUMINAMATH_CALUDE_unique_solution_when_a_is_one_l3398_339824

-- Define the equation
def equation (a x : ℝ) : Prop :=
  (5 : ℝ) ^ (x^2 + 2*a*x + a^2) = a*x^2 + 2*a^2*x + a^3 + a^2 - 6*a + 6

-- Theorem statement
theorem unique_solution_when_a_is_one :
  ∃! a : ℝ, ∃! x : ℝ, equation a x :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_when_a_is_one_l3398_339824
