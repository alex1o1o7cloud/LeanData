import Mathlib

namespace NUMINAMATH_CALUDE_ten_point_square_impossibility_l1617_161726

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A set of ten points in a plane -/
def TenPoints := Fin 10 → Point

/-- Predicate to check if four points lie on the boundary of some square -/
def FourPointsOnSquare (p₁ p₂ p₃ p₄ : Point) : Prop := sorry

/-- Predicate to check if all points in a set lie on the boundary of some square -/
def AllPointsOnSquare (points : TenPoints) : Prop := sorry

/-- The main theorem -/
theorem ten_point_square_impossibility (points : TenPoints) 
  (h : ∀ (a b c d : Fin 10), a ≠ b → b ≠ c → c ≠ d → d ≠ a → 
    FourPointsOnSquare (points a) (points b) (points c) (points d)) :
  ¬ AllPointsOnSquare points :=
sorry

end NUMINAMATH_CALUDE_ten_point_square_impossibility_l1617_161726


namespace NUMINAMATH_CALUDE_expression_simplification_l1617_161750

theorem expression_simplification (x y : ℝ) : 1 - (2 - (3 - (4 - (5 - x)))) - y = 3 - (x + y) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1617_161750


namespace NUMINAMATH_CALUDE_jan_skipping_speed_ratio_jan_skipping_speed_ratio_is_two_l1617_161721

/-- The ratio of Jan's skipping speed after training to her speed before training -/
theorem jan_skipping_speed_ratio : ℝ :=
  let speed_before : ℝ := 70  -- skips per minute
  let total_skips_after : ℝ := 700
  let total_minutes_after : ℝ := 5
  let speed_after : ℝ := total_skips_after / total_minutes_after
  speed_after / speed_before

/-- Proof that the ratio of Jan's skipping speed after training to her speed before training is 2 -/
theorem jan_skipping_speed_ratio_is_two :
  jan_skipping_speed_ratio = 2 := by
  sorry

end NUMINAMATH_CALUDE_jan_skipping_speed_ratio_jan_skipping_speed_ratio_is_two_l1617_161721


namespace NUMINAMATH_CALUDE_jennifer_book_expense_l1617_161793

theorem jennifer_book_expense (total : ℚ) (sandwich_fraction : ℚ) (ticket_fraction : ℚ) (leftover : ℚ) :
  total = 180 →
  sandwich_fraction = 1 / 5 →
  ticket_fraction = 1 / 6 →
  leftover = 24 →
  ∃ (book_fraction : ℚ),
    book_fraction = 1 / 2 ∧
    total * sandwich_fraction + total * ticket_fraction + total * book_fraction + leftover = total :=
by sorry

end NUMINAMATH_CALUDE_jennifer_book_expense_l1617_161793


namespace NUMINAMATH_CALUDE_sarahs_lemonade_profit_l1617_161773

/-- Calculates the total profit for Sarah's lemonade stand --/
theorem sarahs_lemonade_profit
  (total_days : ℕ)
  (hot_days : ℕ)
  (cups_per_day : ℕ)
  (cost_per_cup : ℚ)
  (hot_day_price : ℚ)
  (hot_day_markup : ℚ)
  (h1 : total_days = 10)
  (h2 : hot_days = 3)
  (h3 : cups_per_day = 32)
  (h4 : cost_per_cup = 3/4)
  (h5 : hot_day_price = 1.6351744186046513)
  (h6 : hot_day_markup = 5/4) :
  ∃ (profit : ℚ), profit = 210.2265116279069 :=
by sorry

end NUMINAMATH_CALUDE_sarahs_lemonade_profit_l1617_161773


namespace NUMINAMATH_CALUDE_rosie_pie_production_l1617_161788

/-- Given that Rosie can make 3 pies out of 12 apples, prove that she can make 9 pies with 36 apples. -/
theorem rosie_pie_production (apples_per_batch : ℕ) (pies_per_batch : ℕ) (total_apples : ℕ) 
  (h1 : apples_per_batch = 12) 
  (h2 : pies_per_batch = 3) 
  (h3 : total_apples = 36) :
  (total_apples / (apples_per_batch / pies_per_batch)) = 9 := by
  sorry

end NUMINAMATH_CALUDE_rosie_pie_production_l1617_161788


namespace NUMINAMATH_CALUDE_systematic_sampling_probability_l1617_161778

theorem systematic_sampling_probability 
  (total_population : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_population = 120) 
  (h2 : sample_size = 20) :
  (sample_size : ℚ) / total_population = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_probability_l1617_161778


namespace NUMINAMATH_CALUDE_roses_cut_equals_difference_l1617_161799

/-- The number of roses Jessica cut from her garden -/
def roses_cut : ℕ := sorry

/-- The initial number of roses in the vase -/
def initial_roses : ℕ := 7

/-- The final number of roses in the vase -/
def final_roses : ℕ := 23

/-- Theorem stating that the number of roses Jessica cut is equal to the difference between the final and initial number of roses in the vase -/
theorem roses_cut_equals_difference : roses_cut = final_roses - initial_roses := by sorry

end NUMINAMATH_CALUDE_roses_cut_equals_difference_l1617_161799


namespace NUMINAMATH_CALUDE_infinite_solutions_condition_l1617_161730

theorem infinite_solutions_condition (b : ℝ) : 
  (∀ x : ℝ, 4 * (3 * x - b) = 3 * (4 * x + 16)) ↔ b = -12 := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_condition_l1617_161730


namespace NUMINAMATH_CALUDE_rays_remaining_nickels_l1617_161725

/-- Calculates the number of nickels Ray has left after giving money to Peter and Randi -/
theorem rays_remaining_nickels 
  (initial_cents : ℕ) 
  (cents_to_peter : ℕ) 
  (nickel_value : ℕ) 
  (h1 : initial_cents = 95)
  (h2 : cents_to_peter = 25)
  (h3 : nickel_value = 5) :
  (initial_cents - cents_to_peter - 2 * cents_to_peter) / nickel_value = 4 :=
by sorry

end NUMINAMATH_CALUDE_rays_remaining_nickels_l1617_161725


namespace NUMINAMATH_CALUDE_problem_statement_l1617_161789

theorem problem_statement (x y z : ℝ) (h : x^4 + y^4 + z^4 + x*y*z = 4) :
  x ≤ 2 ∧ Real.sqrt (2 - x) ≥ (y + z) / 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1617_161789


namespace NUMINAMATH_CALUDE_coin_flip_probability_difference_l1617_161720

-- Define a fair coin
def fair_coin_prob : ℚ := 1/2

-- Define the number of flips
def num_flips : ℕ := 4

-- Define the probability of exactly 3 heads in 4 flips
def prob_3_heads : ℚ := Nat.choose num_flips 3 * fair_coin_prob^3 * (1 - fair_coin_prob)^(num_flips - 3)

-- Define the probability of 4 heads in 4 flips
def prob_4_heads : ℚ := fair_coin_prob^num_flips

-- Theorem statement
theorem coin_flip_probability_difference : 
  |prob_3_heads - prob_4_heads| = 7/16 := by sorry

end NUMINAMATH_CALUDE_coin_flip_probability_difference_l1617_161720


namespace NUMINAMATH_CALUDE_xy_is_zero_l1617_161747

theorem xy_is_zero (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 162) : x * y = 0 := by
  sorry

end NUMINAMATH_CALUDE_xy_is_zero_l1617_161747


namespace NUMINAMATH_CALUDE_vector_sum_proof_l1617_161723

def a : ℝ × ℝ := (2, 8)
def b : ℝ × ℝ := (-7, 2)

theorem vector_sum_proof : a + 2 • b = (-12, 12) := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_proof_l1617_161723


namespace NUMINAMATH_CALUDE_right_triangle_sides_l1617_161776

/-- A right triangle with perimeter 60 and height to hypotenuse 12 has sides 15, 20, and 35 -/
theorem right_triangle_sides (a b c : ℝ) (h : ℝ) : 
  a > 0 → b > 0 → c > 0 → h > 0 →
  a + b + c = 60 →
  a^2 + b^2 = c^2 →
  a * b = 12 * c →
  h = 12 →
  (a = 15 ∧ b = 20 ∧ c = 35) ∨ (a = 20 ∧ b = 15 ∧ c = 35) :=
by sorry


end NUMINAMATH_CALUDE_right_triangle_sides_l1617_161776


namespace NUMINAMATH_CALUDE_work_completion_time_l1617_161775

/-- Given workers a, b, and c, and their work rates, prove that b alone completes the work in 48 days -/
theorem work_completion_time (a b c : ℝ) 
  (h1 : a + b = 1 / 16)  -- a and b together finish in 16 days
  (h2 : a = 1 / 24)      -- a alone finishes in 24 days
  (h3 : c = 1 / 48)      -- c alone finishes in 48 days
  : b = 1 / 48 :=        -- b alone finishes in 48 days
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l1617_161775


namespace NUMINAMATH_CALUDE_desk_length_l1617_161764

/-- Given a rectangular desk with width 9 cm and perimeter 46 cm, prove its length is 14 cm. -/
theorem desk_length (width : ℝ) (perimeter : ℝ) (length : ℝ) : 
  width = 9 → perimeter = 46 → 2 * (length + width) = perimeter → length = 14 := by
  sorry

end NUMINAMATH_CALUDE_desk_length_l1617_161764


namespace NUMINAMATH_CALUDE_pablo_puzzle_completion_time_l1617_161761

/-- The number of days it takes Pablo to complete all his puzzles -/
def days_to_complete_puzzles (pieces_per_hour : ℕ) (max_hours_per_day : ℕ) 
  (num_puzzles_300 : ℕ) (num_puzzles_500 : ℕ) : ℕ :=
  let total_pieces := num_puzzles_300 * 300 + num_puzzles_500 * 500
  let pieces_per_day := pieces_per_hour * max_hours_per_day
  (total_pieces + pieces_per_day - 1) / pieces_per_day

/-- Theorem stating that it takes Pablo 7 days to complete all his puzzles -/
theorem pablo_puzzle_completion_time :
  days_to_complete_puzzles 100 7 8 5 = 7 := by
  sorry

#eval days_to_complete_puzzles 100 7 8 5

end NUMINAMATH_CALUDE_pablo_puzzle_completion_time_l1617_161761


namespace NUMINAMATH_CALUDE_not_prime_sum_l1617_161719

theorem not_prime_sum (a b c d : ℕ+) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) 
  (h_int : ∃ (n : ℤ), (a / (a + b) : ℚ) + (b / (b + c) : ℚ) + (c / (c + d) : ℚ) + (d / (d + a) : ℚ) = n) : 
  ¬ Nat.Prime (a + b + c + d) := by
sorry

end NUMINAMATH_CALUDE_not_prime_sum_l1617_161719


namespace NUMINAMATH_CALUDE_sum_of_squares_l1617_161736

theorem sum_of_squares (n : ℕ) (h1 : n > 2) 
  (h2 : ∃ m : ℕ, n^2 = (m + 1)^3 - m^3) : 
  ∃ a b : ℕ, n = a^2 + b^2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1617_161736


namespace NUMINAMATH_CALUDE_line_passes_through_point_l1617_161760

/-- Given a line y = 2x + b passing through the point (1, 2), prove that b = 0 -/
theorem line_passes_through_point (b : ℝ) : 
  (∀ x y : ℝ, y = 2 * x + b) → 2 = 2 * 1 + b → b = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_point_l1617_161760


namespace NUMINAMATH_CALUDE_arithmetic_seq_fifth_term_l1617_161733

/-- An arithmetic sequence -/
def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_seq_fifth_term 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_seq a) 
  (h_sum : a 3 + a 8 = 22) 
  (h_sixth : a 6 = 8) : 
  a 5 = 14 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_seq_fifth_term_l1617_161733


namespace NUMINAMATH_CALUDE_minimum_score_condition_l1617_161714

/-- Represents a knowledge competition with specific scoring rules. -/
structure KnowledgeCompetition where
  totalQuestions : Nat
  correctPoints : Int
  incorrectDeduction : Int
  minimumScore : Int

/-- Calculates the score based on the number of correct answers. -/
def calculateScore (comp : KnowledgeCompetition) (correctAnswers : Nat) : Int :=
  comp.correctPoints * correctAnswers - comp.incorrectDeduction * (comp.totalQuestions - correctAnswers)

/-- Theorem stating the condition to achieve the minimum score. -/
theorem minimum_score_condition (comp : KnowledgeCompetition) 
  (h1 : comp.totalQuestions = 20)
  (h2 : comp.correctPoints = 5)
  (h3 : comp.incorrectDeduction = 1)
  (h4 : comp.minimumScore = 88) :
  ∃ x : Nat, calculateScore comp x ≥ comp.minimumScore ∧ 
    ∀ y : Nat, y < x → calculateScore comp y < comp.minimumScore :=
by
  sorry

end NUMINAMATH_CALUDE_minimum_score_condition_l1617_161714


namespace NUMINAMATH_CALUDE_cog_production_90_workers_2_hours_l1617_161792

/-- Represents the production capabilities of workers in a factory --/
structure ProductionRate where
  gears_per_hour : ℝ
  cogs_per_hour : ℝ

/-- Calculates the total production for a given number of workers, hours, and production rate --/
def total_production (workers : ℝ) (hours : ℝ) (rate : ProductionRate) : ProductionRate :=
  { gears_per_hour := workers * hours * rate.gears_per_hour,
    cogs_per_hour := workers * hours * rate.cogs_per_hour }

/-- Theorem stating the production of cogs by 90 workers in 2 hours --/
theorem cog_production_90_workers_2_hours 
  (rate : ProductionRate)
  (h1 : total_production 150 1 rate = { gears_per_hour := 450, cogs_per_hour := 300 })
  (h2 : total_production 100 1.5 rate = { gears_per_hour := 300, cogs_per_hour := 375 })
  (h3 : (total_production 90 2 rate).gears_per_hour = 360) :
  (total_production 90 2 rate).cogs_per_hour = 180 := by
  sorry

#check cog_production_90_workers_2_hours

end NUMINAMATH_CALUDE_cog_production_90_workers_2_hours_l1617_161792


namespace NUMINAMATH_CALUDE_increase_by_percentage_increase_75_by_150_percent_l1617_161756

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) :
  initial * (1 + percentage / 100) = initial + (initial * percentage / 100) := by sorry

theorem increase_75_by_150_percent :
  75 * (1 + 150 / 100) = 187.5 := by sorry

end NUMINAMATH_CALUDE_increase_by_percentage_increase_75_by_150_percent_l1617_161756


namespace NUMINAMATH_CALUDE_water_one_tenth_after_pourings_l1617_161735

/-- The fraction of water remaining after n pourings -/
def waterRemaining (n : ℕ) : ℚ :=
  3 / (n + 3)

/-- The number of pourings required to reach one-tenth of the original volume -/
def pouringsToOneTenth : ℕ := 27

theorem water_one_tenth_after_pourings :
  waterRemaining pouringsToOneTenth = 1 / 10 := by
  sorry

#eval waterRemaining pouringsToOneTenth

end NUMINAMATH_CALUDE_water_one_tenth_after_pourings_l1617_161735


namespace NUMINAMATH_CALUDE_planes_lines_parallelism_l1617_161700

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the necessary relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (line_parallel : Line → Line → Prop)

-- State the theorem
theorem planes_lines_parallelism 
  (α β : Plane) (m n : Line) 
  (h_not_coincident : α ≠ β)
  (h_different_lines : m ≠ n)
  (h_parallel_planes : parallel α β)
  (h_n_perp_α : perpendicular n α)
  (h_m_perp_β : perpendicular m β) :
  line_parallel m n :=
sorry

end NUMINAMATH_CALUDE_planes_lines_parallelism_l1617_161700


namespace NUMINAMATH_CALUDE_area_ratio_theorem_l1617_161780

def total_area : ℝ := 700
def smaller_area : ℝ := 315

theorem area_ratio_theorem :
  let larger_area := total_area - smaller_area
  let difference := larger_area - smaller_area
  let average := (larger_area + smaller_area) / 2
  difference / average = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_theorem_l1617_161780


namespace NUMINAMATH_CALUDE_det_dilation_matrix_5_l1617_161737

/-- A 2x2 matrix representing a dilation with scale factor k centered at the origin -/
def dilation_matrix (k : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![k, 0; 0, k]

/-- Theorem: The determinant of a 2x2 dilation matrix with scale factor 5 is 25 -/
theorem det_dilation_matrix_5 :
  let E := dilation_matrix 5
  Matrix.det E = 25 := by sorry

end NUMINAMATH_CALUDE_det_dilation_matrix_5_l1617_161737


namespace NUMINAMATH_CALUDE_sum_of_squared_coefficients_is_2395_l1617_161702

/-- The expression to be simplified -/
def expression (x : ℝ) : ℝ := 3 * (x^2 - 3*x + 3) - 8 * (x^3 - 2*x^2 + 4*x - 1)

/-- The sum of squares of coefficients of the simplified expression -/
def sum_of_squared_coefficients : ℝ := 2395

theorem sum_of_squared_coefficients_is_2395 :
  sum_of_squared_coefficients = 2395 := by sorry

end NUMINAMATH_CALUDE_sum_of_squared_coefficients_is_2395_l1617_161702


namespace NUMINAMATH_CALUDE_shift_direct_proportion_l1617_161767

def original_function (x : ℝ) : ℝ := -2 * x

def shift_right (f : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ :=
  fun x => f (x - shift)

def resulting_function (x : ℝ) : ℝ := -2 * x + 6

theorem shift_direct_proportion :
  shift_right original_function 3 = resulting_function := by
  sorry

end NUMINAMATH_CALUDE_shift_direct_proportion_l1617_161767


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_factorial_sum_l1617_161758

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem largest_prime_divisor_of_factorial_sum :
  ∃ (p : ℕ), p.Prime ∧ p ∣ (factorial 13 + factorial 14) ∧
  ∀ (q : ℕ), q.Prime → q ∣ (factorial 13 + factorial 14) → q ≤ p :=
by
  sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_factorial_sum_l1617_161758


namespace NUMINAMATH_CALUDE_shelter_ratio_change_l1617_161783

/-- Proves that given an initial ratio of dogs to cats of 15:7, 60 dogs in the shelter,
    and 16 additional cats taken in, the new ratio of dogs to cats is 15:11. -/
theorem shelter_ratio_change (initial_dogs : ℕ) (initial_cats : ℕ) (additional_cats : ℕ) :
  initial_dogs = 60 →
  initial_dogs / initial_cats = 15 / 7 →
  additional_cats = 16 →
  initial_dogs / (initial_cats + additional_cats) = 15 / 11 := by
  sorry

end NUMINAMATH_CALUDE_shelter_ratio_change_l1617_161783


namespace NUMINAMATH_CALUDE_decreasing_interval_of_even_function_l1617_161784

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m - 2) * x^2 + m * x + 4

theorem decreasing_interval_of_even_function (m : ℝ) :
  (∀ x : ℝ, f m x = f m (-x)) →
  {x : ℝ | ∀ y, x ≤ y → f m x ≥ f m y} = Set.Ici (0 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_decreasing_interval_of_even_function_l1617_161784


namespace NUMINAMATH_CALUDE_missing_number_proof_l1617_161791

theorem missing_number_proof : ∃ (x : ℤ), |7 - 8 * (x - 12)| - |5 - 11| = 73 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l1617_161791


namespace NUMINAMATH_CALUDE_cube_root_of_negative_eight_squared_l1617_161759

theorem cube_root_of_negative_eight_squared :
  ((-8^2 : ℝ) ^ (1/3 : ℝ)) = -4 := by sorry

end NUMINAMATH_CALUDE_cube_root_of_negative_eight_squared_l1617_161759


namespace NUMINAMATH_CALUDE_inequality_range_l1617_161748

theorem inequality_range (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ a ∈ Set.Ioc (-2) 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l1617_161748


namespace NUMINAMATH_CALUDE_line_segment_does_not_intersect_staircase_l1617_161742

/-- Represents a step in the staircase -/
structure Step where
  width : Nat
  height : Nat

/-- Represents the staircase -/
def Staircase : List Step := List.range 2019 |>.map (fun i => { width := i + 1, height := 1 })

/-- The line segment from (0,0) to (2019,2019) -/
def LineSegment : Set (ℝ × ℝ) :=
  {p | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p = (2019 * t, 2019 * t)}

/-- Checks if a point is on a step -/
def onStep (p : ℝ × ℝ) (s : Step) : Prop :=
  (s.width - 1 : ℝ) ≤ p.1 ∧ p.1 < s.width ∧
  (s.height - 1 : ℝ) ≤ p.2 ∧ p.2 < s.height

theorem line_segment_does_not_intersect_staircase :
  ∀ p ∈ LineSegment, ∀ s ∈ Staircase, ¬ onStep p s := by
  sorry

end NUMINAMATH_CALUDE_line_segment_does_not_intersect_staircase_l1617_161742


namespace NUMINAMATH_CALUDE_frog_population_difference_l1617_161751

/-- Theorem: Given the conditions about frog populations in two lakes, prove that the percentage difference is 20%. -/
theorem frog_population_difference (lassie_frogs : ℕ) (total_frogs : ℕ) (P : ℝ) : 
  lassie_frogs = 45 →
  total_frogs = 81 →
  total_frogs = lassie_frogs + (lassie_frogs - P / 100 * lassie_frogs) →
  P = 20 := by
  sorry

end NUMINAMATH_CALUDE_frog_population_difference_l1617_161751


namespace NUMINAMATH_CALUDE_smallest_x_with_given_remainders_l1617_161790

theorem smallest_x_with_given_remainders :
  ∃ x : ℕ,
    x > 0 ∧
    x % 6 = 5 ∧
    x % 7 = 6 ∧
    x % 8 = 7 ∧
    ∀ y : ℕ, y > 0 → y % 6 = 5 → y % 7 = 6 → y % 8 = 7 → x ≤ y ∧
    x = 167 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_with_given_remainders_l1617_161790


namespace NUMINAMATH_CALUDE_average_monthly_growth_rate_correct_l1617_161712

/-- The average monthly growth rate of a factory's production volume -/
def average_monthly_growth_rate (a : ℝ) : ℝ := a^(1/11) - 1

/-- Theorem stating that the average monthly growth rate is correct -/
theorem average_monthly_growth_rate_correct (a : ℝ) (h : a > 0) :
  (1 + average_monthly_growth_rate a)^11 = a :=
by sorry

end NUMINAMATH_CALUDE_average_monthly_growth_rate_correct_l1617_161712


namespace NUMINAMATH_CALUDE_garden_width_l1617_161740

/-- A rectangular garden with specific dimensions. -/
structure RectangularGarden where
  width : ℝ
  length : ℝ
  perimeter_eq : width + length = 30
  length_eq : length = width + 8

/-- The width of the garden is 11 feet. -/
theorem garden_width (g : RectangularGarden) : g.width = 11 := by
  sorry

#check garden_width

end NUMINAMATH_CALUDE_garden_width_l1617_161740


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1617_161768

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 1 + a 3 + a 9 = 20 →
  4 * a 5 - a 7 = 20 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1617_161768


namespace NUMINAMATH_CALUDE_smallest_common_multiple_of_8_and_6_l1617_161786

theorem smallest_common_multiple_of_8_and_6 :
  ∃ (n : ℕ), n > 0 ∧ 8 ∣ n ∧ 6 ∣ n ∧ ∀ (m : ℕ), m > 0 ∧ 8 ∣ m ∧ 6 ∣ m → n ≤ m :=
by
  use 24
  sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_of_8_and_6_l1617_161786


namespace NUMINAMATH_CALUDE_fathers_savings_l1617_161762

theorem fathers_savings (total : ℝ) : 
  (total / 2 - (total / 2) * 0.6) = 2000 → total = 10000 := by
  sorry

end NUMINAMATH_CALUDE_fathers_savings_l1617_161762


namespace NUMINAMATH_CALUDE_triangle_inequality_bound_bound_is_tight_l1617_161708

theorem triangle_inequality_bound (a b c : ℝ) (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) (h_cond : a ≥ (b + c) / 3) :
  (a * c + b * c - c^2) / (a^2 + b^2 + 3 * c^2 + 2 * a * b - 4 * b * c) ≤ (2 * Real.sqrt 2 + 1) / 7 :=
sorry

theorem bound_is_tight :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b ∧
  a ≥ (b + c) / 3 ∧
  (a * c + b * c - c^2) / (a^2 + b^2 + 3 * c^2 + 2 * a * b - 4 * b * c) = (2 * Real.sqrt 2 + 1) / 7 :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_bound_bound_is_tight_l1617_161708


namespace NUMINAMATH_CALUDE_solution_to_equation_l1617_161779

-- Define the custom operation
def custom_op (a b : ℝ) : ℝ := a * b + a + b + 2

-- State the theorem
theorem solution_to_equation :
  ∃ x : ℝ, custom_op x 3 = 1 ∧ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_equation_l1617_161779


namespace NUMINAMATH_CALUDE_petya_win_probability_is_1_256_l1617_161705

/-- The "Heap of Stones" game --/
structure HeapOfStones where
  initial_stones : Nat
  max_stones_per_turn : Nat

/-- A player in the game --/
inductive Player
  | Petya
  | Computer

/-- The game state --/
structure GameState where
  stones_left : Nat
  current_player : Player

/-- The result of a game --/
inductive GameResult
  | PetyaWins
  | ComputerWins

/-- The strategy for the computer player --/
def computer_strategy : GameState → Nat := sorry

/-- The random strategy for Petya --/
def petya_random_strategy : GameState → Nat := sorry

/-- Play a single game --/
def play_game (game : HeapOfStones) : GameResult := sorry

/-- Calculate the probability of Petya winning --/
def petya_win_probability (game : HeapOfStones) : ℚ := sorry

/-- The main theorem --/
theorem petya_win_probability_is_1_256 :
  let game : HeapOfStones := ⟨16, 4⟩
  petya_win_probability game = 1 / 256 := by sorry

end NUMINAMATH_CALUDE_petya_win_probability_is_1_256_l1617_161705


namespace NUMINAMATH_CALUDE_fraction_inequality_l1617_161722

theorem fraction_inequality (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) 
  (h3 : c < d) (h4 : d < 0) : 
  a / d < b / c := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l1617_161722


namespace NUMINAMATH_CALUDE_inspection_decision_l1617_161709

/-- Represents the probability of an item being defective -/
def p : Real := 0.1

/-- Total number of items in a box -/
def totalItems : Nat := 200

/-- Number of items in the initial sample -/
def sampleSize : Nat := 20

/-- Number of defective items found in the sample -/
def defectivesInSample : Nat := 2

/-- Cost of inspecting one item -/
def inspectionCost : Real := 2

/-- Compensation fee for one defective item -/
def compensationFee : Real := 25

/-- Expected number of defective items in the remaining items -/
def expectedDefectives : Real := (totalItems - sampleSize) * p

/-- Expected cost without further inspection -/
def expectedCostWithoutInspection : Real :=
  sampleSize * inspectionCost + expectedDefectives * compensationFee

/-- Cost of inspecting all items -/
def costOfInspectingAll : Real := totalItems * inspectionCost

theorem inspection_decision :
  expectedCostWithoutInspection > costOfInspectingAll :=
sorry

end NUMINAMATH_CALUDE_inspection_decision_l1617_161709


namespace NUMINAMATH_CALUDE_picture_on_wall_l1617_161711

theorem picture_on_wall (wall_width picture_width : ℝ) 
  (hw : wall_width = 26) (hp : picture_width = 4) :
  let distance := (wall_width - picture_width) / 2
  distance = 11 := by
  sorry

end NUMINAMATH_CALUDE_picture_on_wall_l1617_161711


namespace NUMINAMATH_CALUDE_product_of_roots_l1617_161739

theorem product_of_roots (x y : ℝ) : 
  x = 16^(1/4) → y = 64^(1/2) → x * y = 16 := by sorry

end NUMINAMATH_CALUDE_product_of_roots_l1617_161739


namespace NUMINAMATH_CALUDE_festival_attendance_l1617_161781

/-- Proves the attendance on the second day of a three-day festival -/
theorem festival_attendance (total : ℕ) (day1 day2 day3 : ℕ) : 
  total = 2700 →
  day2 = day1 / 2 →
  day3 = 3 * day1 →
  total = day1 + day2 + day3 →
  day2 = 300 := by
  sorry

end NUMINAMATH_CALUDE_festival_attendance_l1617_161781


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1617_161729

theorem min_value_of_expression (x y : ℝ) (h : 2 * x - y = 4) :
  ∃ (m : ℝ), m = 8 ∧ ∀ (a b : ℝ), 2 * a - b = 4 → 4^a + (1/2)^b ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1617_161729


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l1617_161718

theorem pure_imaginary_condition (m : ℝ) : 
  (∃ z : ℂ, z = m * (m - 1) + (m - 1) * I ∧ z.re = 0 ∧ z.im ≠ 0) → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l1617_161718


namespace NUMINAMATH_CALUDE_simplify_fraction_l1617_161749

theorem simplify_fraction : 45 * (14 / 25) * (1 / 18) * (5 / 11) = 7 / 11 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1617_161749


namespace NUMINAMATH_CALUDE_john_crab_baskets_l1617_161785

/-- The number of crab baskets John reels in per week -/
def num_baskets : ℕ := sorry

/-- The number of crabs per basket -/
def crabs_per_basket : ℕ := 4

/-- The price of each crab in dollars -/
def price_per_crab : ℕ := 3

/-- John's total earnings in dollars -/
def total_earnings : ℕ := 72

theorem john_crab_baskets :
  num_baskets = 6 :=
by sorry

end NUMINAMATH_CALUDE_john_crab_baskets_l1617_161785


namespace NUMINAMATH_CALUDE_largest_prime_divisor_factorial_sum_l1617_161728

theorem largest_prime_divisor_factorial_sum : 
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ (Nat.factorial 13 + Nat.factorial 14) ∧ 
  ∀ q : ℕ, Nat.Prime q → q ∣ (Nat.factorial 13 + Nat.factorial 14) → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_factorial_sum_l1617_161728


namespace NUMINAMATH_CALUDE_fraction_problem_l1617_161706

theorem fraction_problem :
  ∃ x : ℚ, x * 180 = 18 ∧ x < 0.15 → x = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l1617_161706


namespace NUMINAMATH_CALUDE_decimal_11_is_binary_1011_l1617_161744

def decimal_to_binary (n : Nat) : List Nat :=
  if n = 0 then [0]
  else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else aux (m / 2) ((m % 2) :: acc)
    aux n []

theorem decimal_11_is_binary_1011 : decimal_to_binary 11 = [1, 0, 1, 1] := by
  sorry

end NUMINAMATH_CALUDE_decimal_11_is_binary_1011_l1617_161744


namespace NUMINAMATH_CALUDE_specific_tunnel_length_l1617_161713

/-- Calculates the length of a tunnel given train and travel parameters. -/
def tunnel_length (train_length : ℚ) (train_speed : ℚ) (exit_time : ℚ) : ℚ :=
  train_speed * exit_time / 60 - train_length

/-- Theorem stating the length of the tunnel given specific parameters. -/
theorem specific_tunnel_length :
  let train_length : ℚ := 2
  let train_speed : ℚ := 40
  let exit_time : ℚ := 5
  tunnel_length train_length train_speed exit_time = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_specific_tunnel_length_l1617_161713


namespace NUMINAMATH_CALUDE_cost_price_is_60_l1617_161727

/-- The cost price of a single ball, given the selling price of multiple balls and the loss incurred. -/
def cost_price_of_ball (selling_price : ℕ) (num_balls_sold : ℕ) (num_balls_loss : ℕ) : ℕ :=
  selling_price / (num_balls_sold - num_balls_loss)

/-- Theorem stating that the cost price of a ball is 60 under given conditions. -/
theorem cost_price_is_60 :
  cost_price_of_ball 720 17 5 = 60 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_is_60_l1617_161727


namespace NUMINAMATH_CALUDE_circle_properties_l1617_161710

/-- Given a circle with area 81π cm², prove its radius is 9 cm and circumference is 18π cm. -/
theorem circle_properties (A : ℝ) (h : A = 81 * Real.pi) :
  ∃ (r C : ℝ), r = 9 ∧ C = 18 * Real.pi ∧ A = Real.pi * r^2 ∧ C = 2 * Real.pi * r := by
  sorry

end NUMINAMATH_CALUDE_circle_properties_l1617_161710


namespace NUMINAMATH_CALUDE_smallest_special_number_l1617_161755

def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, n = k^3

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

theorem smallest_special_number : 
  ∀ n : ℕ, n > 0 → n % 20 = 0 → is_perfect_cube (n^2) → is_perfect_square (n^3) → 
  n ≥ 1000000 :=
sorry

end NUMINAMATH_CALUDE_smallest_special_number_l1617_161755


namespace NUMINAMATH_CALUDE_sum_of_divisors_360_l1617_161715

-- Define the sum of positive divisors function
def sumOfDivisors (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem sum_of_divisors_360 (i j : ℕ) :
  sumOfDivisors (2^i * 3^j) = 360 → i = 3 ∧ j = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisors_360_l1617_161715


namespace NUMINAMATH_CALUDE_surface_area_ratio_l1617_161770

/-- A regular tetrahedron with its inscribed sphere -/
structure RegularTetrahedronWithInscribedSphere where
  /-- The surface area of the regular tetrahedron -/
  S₁ : ℝ
  /-- The surface area of the inscribed sphere -/
  S₂ : ℝ
  /-- The surface area of the tetrahedron is positive -/
  h_S₁_pos : 0 < S₁
  /-- The surface area of the sphere is positive -/
  h_S₂_pos : 0 < S₂

/-- The ratio of the surface area of a regular tetrahedron to its inscribed sphere -/
theorem surface_area_ratio (t : RegularTetrahedronWithInscribedSphere) :
  t.S₁ / t.S₂ = 6 * Real.sqrt 3 / Real.pi := by sorry

end NUMINAMATH_CALUDE_surface_area_ratio_l1617_161770


namespace NUMINAMATH_CALUDE_travel_time_calculation_l1617_161769

theorem travel_time_calculation (total_distance : ℝ) (speed1 : ℝ) (speed2 : ℝ) :
  total_distance = 300 →
  speed1 = 30 →
  speed2 = 25 →
  (total_distance / 2 / speed1) + (total_distance / 2 / speed2) = 11 := by
  sorry

end NUMINAMATH_CALUDE_travel_time_calculation_l1617_161769


namespace NUMINAMATH_CALUDE_promotion_savings_difference_l1617_161753

/-- Calculates the total cost of two pairs of shoes under a given promotion --/
def promotionCost (regularPrice : ℝ) (discountPercent : ℝ) : ℝ :=
  regularPrice + (regularPrice * (1 - discountPercent))

/-- Represents the difference in savings between two promotions --/
def savingsDifference (regularPrice : ℝ) (discountA : ℝ) (discountB : ℝ) : ℝ :=
  promotionCost regularPrice discountB - promotionCost regularPrice discountA

theorem promotion_savings_difference :
  savingsDifference 50 0.3 0.2 = 5 := by sorry

end NUMINAMATH_CALUDE_promotion_savings_difference_l1617_161753


namespace NUMINAMATH_CALUDE_projectile_max_height_l1617_161745

/-- The height function of the projectile -/
def h (t : ℝ) : ℝ := -12 * t^2 + 72 * t + 45

/-- Theorem: The maximum height reached by the projectile is 153 feet -/
theorem projectile_max_height :
  ∃ (t_max : ℝ), ∀ (t : ℝ), h t ≤ h t_max ∧ h t_max = 153 :=
sorry

end NUMINAMATH_CALUDE_projectile_max_height_l1617_161745


namespace NUMINAMATH_CALUDE_pictures_on_front_l1617_161765

theorem pictures_on_front (total : ℕ) (on_back : ℕ) (h1 : total = 15) (h2 : on_back = 9) :
  total - on_back = 6 := by
  sorry

end NUMINAMATH_CALUDE_pictures_on_front_l1617_161765


namespace NUMINAMATH_CALUDE_sum_of_digits_of_square_1111_l1617_161798

def repeat_digit (d : Nat) (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | m + 1 => d + 10 * (repeat_digit d m)

def sum_of_digits (n : Nat) : Nat :=
  if n < 10 then n
  else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_of_square_1111 :
  sum_of_digits ((repeat_digit 1 4) ^ 2) = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_square_1111_l1617_161798


namespace NUMINAMATH_CALUDE_quadratic_radical_problem_l1617_161707

/-- A number is a simplest quadratic radical if it cannot be further simplified -/
def IsSimplestQuadraticRadical (x : ℝ) : Prop :=
  ∃ n : ℕ, x = Real.sqrt n ∧ ∀ m : ℕ, m < n → ¬∃ k : ℕ, n = k^2 * m

/-- Two quadratic radicals are of the same type if their radicands have the same squarefree part -/
def SameTypeRadical (x y : ℝ) : Prop :=
  ∃ a b : ℕ, x = Real.sqrt a ∧ y = Real.sqrt b ∧ ∃ k m n : ℕ, k ≠ 0 ∧ m.Coprime n ∧ a = k * m ∧ b = k * n

theorem quadratic_radical_problem (a : ℝ) :
  IsSimplestQuadraticRadical (Real.sqrt (2 * a + 1)) →
  SameTypeRadical (Real.sqrt (2 * a + 1)) (Real.sqrt 48) →
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_radical_problem_l1617_161707


namespace NUMINAMATH_CALUDE_unique_divisible_digit_l1617_161795

def is_divisible (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

def seven_digit_number (A : ℕ) : ℕ := 3538080 + A

theorem unique_divisible_digit :
  ∃! A : ℕ,
    A < 10 ∧
    is_divisible (seven_digit_number A) 2 ∧
    is_divisible (seven_digit_number A) 3 ∧
    is_divisible (seven_digit_number A) 4 ∧
    is_divisible (seven_digit_number A) 5 ∧
    is_divisible (seven_digit_number A) 6 ∧
    is_divisible (seven_digit_number A) 8 ∧
    is_divisible (seven_digit_number A) 9 ∧
    A = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_divisible_digit_l1617_161795


namespace NUMINAMATH_CALUDE_triangle_side_difference_triangle_side_difference_is_12_l1617_161732

theorem triangle_side_difference : ℕ → Prop :=
  fun d =>
    ∃ (x_min x_max : ℤ),
      (∀ x : ℤ, (x > x_min ∧ x < x_max) → (x + 7 > 10 ∧ x + 10 > 7 ∧ 7 + 10 > x)) ∧
      (∀ x : ℤ, (x ≤ x_min ∨ x ≥ x_max) → ¬(x + 7 > 10 ∧ x + 10 > 7 ∧ 7 + 10 > x)) ∧
      (x_max - x_min = d + 1)

theorem triangle_side_difference_is_12 : triangle_side_difference 12 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_difference_triangle_side_difference_is_12_l1617_161732


namespace NUMINAMATH_CALUDE_mandy_shirts_total_l1617_161754

theorem mandy_shirts_total (black_packs yellow_packs : ℕ) 
  (black_per_pack yellow_per_pack : ℕ) : 
  black_packs = 3 → 
  yellow_packs = 3 → 
  black_per_pack = 5 → 
  yellow_per_pack = 2 → 
  black_packs * black_per_pack + yellow_packs * yellow_per_pack = 21 := by
  sorry

#check mandy_shirts_total

end NUMINAMATH_CALUDE_mandy_shirts_total_l1617_161754


namespace NUMINAMATH_CALUDE_frog_final_position_l1617_161741

def frog_jumps (n : ℕ) : ℕ := n * (n + 1) / 2

theorem frog_final_position :
  ∀ (total_positions : ℕ) (num_jumps : ℕ),
    total_positions = 6 →
    num_jumps = 20 →
    frog_jumps num_jumps % total_positions = 1 := by
  sorry

end NUMINAMATH_CALUDE_frog_final_position_l1617_161741


namespace NUMINAMATH_CALUDE_optimal_price_theorem_l1617_161774

-- Define the problem parameters
def initial_price : ℝ := 60
def initial_sales : ℝ := 300
def cost_price : ℝ := 40
def target_profit : ℝ := 6080
def price_sales_ratio : ℝ := 20

-- Define the profit function
def profit (price : ℝ) : ℝ :=
  (price - cost_price) * (initial_sales + price_sales_ratio * (initial_price - price))

-- State the theorem
theorem optimal_price_theorem :
  ∃ (optimal_price : ℝ),
    profit optimal_price = target_profit ∧
    optimal_price < initial_price ∧
    ∀ (p : ℝ), p < optimal_price → profit p < target_profit :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_optimal_price_theorem_l1617_161774


namespace NUMINAMATH_CALUDE_solve_for_c_l1617_161757

theorem solve_for_c (p q : ℝ → ℝ) (c : ℝ) : 
  (∀ x, p x = 4 * x - 5) →
  (∀ x, q x = 5 * x - c) →
  p (q 3) = 27 →
  c = 7 := by
sorry

end NUMINAMATH_CALUDE_solve_for_c_l1617_161757


namespace NUMINAMATH_CALUDE_max_a_is_maximum_l1617_161763

/-- The polynomial function f(x) = ax^2 - ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - a * x + 1

/-- The condition that |f(x)| ≤ 1 for all x in [0, 1] -/
def condition (a : ℝ) : Prop :=
  ∀ x : ℝ, x ∈ Set.Icc 0 1 → |f a x| ≤ 1

/-- The maximum value of a that satisfies the condition -/
def max_a : ℝ := 8

/-- Theorem stating that max_a is the maximum value satisfying the condition -/
theorem max_a_is_maximum :
  (condition max_a) ∧ (∀ a : ℝ, a > max_a → ¬(condition a)) :=
sorry

end NUMINAMATH_CALUDE_max_a_is_maximum_l1617_161763


namespace NUMINAMATH_CALUDE_min_value_of_function_l1617_161738

theorem min_value_of_function (t : ℝ) (h : t > 0) :
  (t^2 - 4*t + 1) / t ≥ -2 ∧ 
  ∀ ε > 0, ∃ t₀ > 0, (t₀^2 - 4*t₀ + 1) / t₀ < -2 + ε :=
sorry

end NUMINAMATH_CALUDE_min_value_of_function_l1617_161738


namespace NUMINAMATH_CALUDE_drop_is_negative_of_rise_is_positive_l1617_161794

/-- Represents the change in water level -/
structure WaterLevelChange where
  magnitude : ℝ
  isRise : Bool

/-- Records a water level change as a signed real number -/
def recordChange (change : WaterLevelChange) : ℝ :=
  if change.isRise then change.magnitude else -change.magnitude

theorem drop_is_negative_of_rise_is_positive 
  (h : ∀ (rise : WaterLevelChange), rise.isRise → recordChange rise = rise.magnitude) :
  ∀ (drop : WaterLevelChange), ¬drop.isRise → recordChange drop = -drop.magnitude :=
by sorry

end NUMINAMATH_CALUDE_drop_is_negative_of_rise_is_positive_l1617_161794


namespace NUMINAMATH_CALUDE_journey_fraction_l1617_161787

theorem journey_fraction (total_journey : ℝ) (bus_fraction : ℝ) (foot_distance : ℝ)
  (h1 : total_journey = 130)
  (h2 : bus_fraction = 17 / 20)
  (h3 : foot_distance = 6.5) :
  (total_journey - bus_fraction * total_journey - foot_distance) / total_journey = 1 / 10 :=
by
  sorry

end NUMINAMATH_CALUDE_journey_fraction_l1617_161787


namespace NUMINAMATH_CALUDE_common_intersection_theorem_l1617_161772

/-- The common intersection point of a family of lines -/
def common_intersection_point : ℝ × ℝ := (-1, 1)

/-- The equation of the family of lines -/
def line_equation (a b d x y : ℝ) : Prop :=
  (b + d) * x + b * y = b + 2 * d

theorem common_intersection_theorem :
  ∀ (a b d : ℝ), line_equation a b d (common_intersection_point.1) (common_intersection_point.2) ∧
  (∀ (x y : ℝ), (∀ a b d : ℝ, line_equation a b d x y) → (x, y) = common_intersection_point) :=
sorry

end NUMINAMATH_CALUDE_common_intersection_theorem_l1617_161772


namespace NUMINAMATH_CALUDE_count_factorizable_pairs_eq_325_l1617_161704

/-- Counts the number of ordered pairs (a,b) satisfying the factorization condition -/
def count_factorizable_pairs : ℕ :=
  (Finset.range 50).sum (λ a => (a + 1) / 2)

/-- The main theorem stating that the count of factorizable pairs is 325 -/
theorem count_factorizable_pairs_eq_325 : count_factorizable_pairs = 325 := by
  sorry


end NUMINAMATH_CALUDE_count_factorizable_pairs_eq_325_l1617_161704


namespace NUMINAMATH_CALUDE_consecutive_non_primes_l1617_161731

theorem consecutive_non_primes (n : ℕ) : ∃ (k : ℕ), ∀ (i : ℕ), i < n → ¬ Nat.Prime (k + i) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_non_primes_l1617_161731


namespace NUMINAMATH_CALUDE_parabola_directrix_l1617_161717

/-- The equation of a parabola -/
def parabola (x y : ℝ) : Prop := y = 3 * x^2 + 6 * x + 2

/-- The equation of the directrix -/
def directrix (y : ℝ) : Prop := y = -11/12

/-- Theorem: The directrix of the given parabola is y = -11/12 -/
theorem parabola_directrix :
  ∀ x y : ℝ, parabola x y → ∃ d : ℝ, directrix d ∧ 
  (∀ p : ℝ × ℝ, p.1 = x ∧ p.2 = y → 
    ∃ f : ℝ × ℝ, (f.2 - p.2)^2 = 4 * 3 * ((p.1 - f.1)^2 + (p.2 - d)^2)) :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l1617_161717


namespace NUMINAMATH_CALUDE_product_equals_nine_l1617_161752

theorem product_equals_nine : 
  (1 + 1/1) * (1 + 1/2) * (1 + 1/3) * (1 + 1/4) * 
  (1 + 1/5) * (1 + 1/6) * (1 + 1/7) * (1 + 1/8) = 9 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_nine_l1617_161752


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1617_161777

theorem polynomial_division_remainder : 
  ∃ q : Polynomial ℝ, 
    5 * X^6 + 3 * X^4 - 2 * X^3 + 7 * X^2 + 4 = 
    (X^2 + 2 * X + 1) * q + (-38 * X - 29) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1617_161777


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1617_161743

theorem solution_set_inequality (x : ℝ) :
  (1 / x ≤ 2) ↔ (x ∈ Set.Iic 0 ∪ Set.Ici (1/2)) :=
sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1617_161743


namespace NUMINAMATH_CALUDE_truck_dirt_speed_l1617_161716

/-- Represents the speed of a truck on different road types -/
structure TruckSpeed where
  dirt : ℝ
  paved : ℝ

/-- Represents the duration and distance of a truck's journey -/
structure TruckJourney where
  speed : TruckSpeed
  dirt_time : ℝ
  paved_time : ℝ
  total_distance : ℝ

/-- The conditions of the truck's journey -/
def journey_conditions (j : TruckJourney) : Prop :=
  j.paved_time = 2 ∧
  j.dirt_time = 3 ∧
  j.total_distance = 200 ∧
  j.speed.paved = j.speed.dirt + 20

/-- The theorem stating the speed of the truck on the dirt road -/
theorem truck_dirt_speed (j : TruckJourney) :
  journey_conditions j → j.speed.dirt = 32 := by
  sorry


end NUMINAMATH_CALUDE_truck_dirt_speed_l1617_161716


namespace NUMINAMATH_CALUDE_train_length_calculation_l1617_161766

/-- The length of a train that crosses a platform of equal length in one minute at 90 km/hr is 750 meters. -/
theorem train_length_calculation (train_length platform_length : ℝ) (speed : ℝ) (time : ℝ) :
  train_length = platform_length →
  speed = 90 →
  time = 1 / 60 →
  train_length = 750 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l1617_161766


namespace NUMINAMATH_CALUDE_expenditure_increase_percentage_l1617_161724

theorem expenditure_increase_percentage
  (initial_expenditure : ℝ)
  (initial_savings : ℝ)
  (initial_income : ℝ)
  (h_ratio : initial_expenditure / initial_savings = 3 / 2)
  (h_income : initial_expenditure + initial_savings = initial_income)
  (h_new_income : ℝ)
  (h_income_increase : h_new_income = initial_income * 1.15)
  (h_new_savings : ℝ)
  (h_savings_increase : h_new_savings = initial_savings * 1.06)
  (h_new_expenditure : ℝ)
  (h_new_balance : h_new_expenditure + h_new_savings = h_new_income) :
  (h_new_expenditure - initial_expenditure) / initial_expenditure = 0.21 :=
sorry

end NUMINAMATH_CALUDE_expenditure_increase_percentage_l1617_161724


namespace NUMINAMATH_CALUDE_quadratic_function_k_value_l1617_161782

/-- A quadratic function with integer coefficients -/
def QuadraticFunction (a b c : ℤ) : ℝ → ℝ := fun x ↦ (a * x^2 : ℝ) + (b * x : ℝ) + (c : ℝ)

theorem quadratic_function_k_value
  (a b c k : ℤ)
  (h1 : QuadraticFunction a b c 1 = 0)
  (h2 : 60 < QuadraticFunction a b c 7 ∧ QuadraticFunction a b c 7 < 70)
  (h3 : 80 < QuadraticFunction a b c 8 ∧ QuadraticFunction a b c 8 < 90)
  (h4 : (2000 : ℝ) * (k : ℝ) < QuadraticFunction a b c 50 ∧
        QuadraticFunction a b c 50 < (2000 : ℝ) * ((k + 1) : ℝ)) :
  k = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_k_value_l1617_161782


namespace NUMINAMATH_CALUDE_cards_added_l1617_161771

theorem cards_added (initial_cards final_cards : ℕ) 
  (h1 : initial_cards = 9) 
  (h2 : final_cards = 13) : 
  final_cards - initial_cards = 4 := by
  sorry

end NUMINAMATH_CALUDE_cards_added_l1617_161771


namespace NUMINAMATH_CALUDE_infinite_functions_satisfying_condition_l1617_161703

theorem infinite_functions_satisfying_condition :
  ∃ (S : Set (ℝ → ℝ)), (Set.Infinite S) ∧ 
  (∀ f ∈ S, 2 * f 3 - 10 = f 1) := by
sorry

end NUMINAMATH_CALUDE_infinite_functions_satisfying_condition_l1617_161703


namespace NUMINAMATH_CALUDE_average_salary_non_technicians_l1617_161797

/-- Proves that the average salary of non-technician workers is 6000 given the conditions --/
theorem average_salary_non_technicians (total_workers : ℕ) (avg_salary_all : ℕ) 
  (num_technicians : ℕ) (avg_salary_technicians : ℕ) :
  total_workers = 21 →
  avg_salary_all = 8000 →
  num_technicians = 7 →
  avg_salary_technicians = 12000 →
  (total_workers - num_technicians) * 
    ((total_workers * avg_salary_all - num_technicians * avg_salary_technicians) / 
     (total_workers - num_technicians)) = 6000 * (total_workers - num_technicians) :=
by
  sorry

#check average_salary_non_technicians

end NUMINAMATH_CALUDE_average_salary_non_technicians_l1617_161797


namespace NUMINAMATH_CALUDE_indistinguishable_ball_sequences_l1617_161701

/-- The number of different sequences when drawing indistinguishable balls -/
def number_of_sequences (total : ℕ) (white : ℕ) (black : ℕ) : ℕ :=
  Nat.choose total white

theorem indistinguishable_ball_sequences :
  number_of_sequences 13 8 5 = 1287 := by
  sorry

end NUMINAMATH_CALUDE_indistinguishable_ball_sequences_l1617_161701


namespace NUMINAMATH_CALUDE_trig_expression_equals_sqrt_three_l1617_161746

theorem trig_expression_equals_sqrt_three :
  (2 * Real.cos (10 * π / 180) - Real.sin (20 * π / 180)) / Real.sin (70 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_sqrt_three_l1617_161746


namespace NUMINAMATH_CALUDE_min_perimeter_isosceles_triangles_l1617_161734

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  leg : ℕ
  base : ℕ

/-- Check if two isosceles triangles are noncongruent -/
def noncongruent (t1 t2 : IsoscelesTriangle) : Prop :=
  t1.leg ≠ t2.leg ∨ t1.base ≠ t2.base

/-- Calculate the perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℕ :=
  2 * t.leg + t.base

/-- Calculate the area of an isosceles triangle -/
def area (t : IsoscelesTriangle) : ℚ :=
  (t.base : ℚ) * (((t.leg : ℚ) ^ 2 - ((t.base : ℚ) / 2) ^ 2).sqrt) / 2

/-- The theorem statement -/
theorem min_perimeter_isosceles_triangles :
  ∃ (t1 t2 : IsoscelesTriangle),
    noncongruent t1 t2 ∧
    perimeter t1 = perimeter t2 ∧
    area t1 = area t2 ∧
    10 * t1.base = 9 * t2.base ∧
    perimeter t1 = 362 ∧
    (∀ (s1 s2 : IsoscelesTriangle),
      noncongruent s1 s2 →
      perimeter s1 = perimeter s2 →
      area s1 = area s2 →
      10 * s1.base = 9 * s2.base →
      perimeter s1 ≥ 362) :=
sorry

end NUMINAMATH_CALUDE_min_perimeter_isosceles_triangles_l1617_161734


namespace NUMINAMATH_CALUDE_gaussian_function_properties_l1617_161796

-- Define the Gaussian function (floor function)
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Theorem statement
theorem gaussian_function_properties :
  -- 1. The range of floor is ℤ
  (∀ n : ℤ, ∃ x : ℝ, floor x = n) ∧
  -- 2. floor is not an odd function
  (∃ x : ℝ, floor (-x) ≠ -floor x) ∧
  -- 3. x - floor x is periodic with period 1
  (∀ x : ℝ, x - floor x = (x + 1) - floor (x + 1)) ∧
  -- 4. floor is not monotonically increasing on ℝ
  (∃ x y : ℝ, x < y ∧ floor x > floor y) :=
by sorry

end NUMINAMATH_CALUDE_gaussian_function_properties_l1617_161796
