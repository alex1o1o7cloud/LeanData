import Mathlib

namespace NUMINAMATH_CALUDE_perimeter_after_adding_tiles_l342_34220

/-- A figure composed of square tiles -/
structure TiledFigure where
  tiles : ℕ
  perimeter : ℕ

/-- Adds tiles to a figure, each sharing at least one side with the original figure -/
def add_tiles (figure : TiledFigure) (new_tiles : ℕ) : TiledFigure :=
  { tiles := figure.tiles + new_tiles,
    perimeter := figure.perimeter + 2 * new_tiles }

theorem perimeter_after_adding_tiles (initial_figure : TiledFigure) :
  initial_figure.tiles = 10 →
  initial_figure.perimeter = 16 →
  (add_tiles initial_figure 4).perimeter = 20 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_after_adding_tiles_l342_34220


namespace NUMINAMATH_CALUDE_labourer_absence_solution_l342_34292

/-- Represents the problem of calculating a labourer's absence days --/
def LabourerAbsence (total_days work_pay absence_fine total_received : ℚ) : Prop :=
  ∃ (days_worked days_absent : ℚ),
    days_worked + days_absent = total_days ∧
    work_pay * days_worked - absence_fine * days_absent = total_received ∧
    days_absent = 5

/-- Theorem stating the solution to the labourer absence problem --/
theorem labourer_absence_solution :
  LabourerAbsence 25 2 (1/2) (75/2) :=
sorry

end NUMINAMATH_CALUDE_labourer_absence_solution_l342_34292


namespace NUMINAMATH_CALUDE_speed_of_sound_calculation_l342_34289

/-- The speed of sound in meters per second -/
def speed_of_sound : ℝ := 330

/-- The time between hearing the first and second blast in seconds -/
def time_between_blasts : ℝ := 30 * 60 + 24

/-- The time between the occurrence of the first and second blast in seconds -/
def time_between_blast_occurrences : ℝ := 30 * 60

/-- The distance from the blast site when hearing the second blast in meters -/
def distance_at_second_blast : ℝ := 7920

/-- Theorem stating that the speed of sound is 330 m/s given the problem conditions -/
theorem speed_of_sound_calculation :
  speed_of_sound = distance_at_second_blast / (time_between_blasts - time_between_blast_occurrences) :=
by sorry

end NUMINAMATH_CALUDE_speed_of_sound_calculation_l342_34289


namespace NUMINAMATH_CALUDE_smallest_n_congruence_eight_satisfies_congruence_eight_is_smallest_smallest_positive_integer_congruence_l342_34286

theorem smallest_n_congruence (n : ℕ) : 
  (n > 0 ∧ 19 * n ≡ 5678 [ZMOD 11]) → n ≥ 8 :=
by sorry

theorem eight_satisfies_congruence : 19 * 8 ≡ 5678 [ZMOD 11] :=
by sorry

theorem eight_is_smallest : 
  ∀ m : ℕ, m > 0 ∧ m < 8 → ¬(19 * m ≡ 5678 [ZMOD 11]) :=
by sorry

theorem smallest_positive_integer_congruence : 
  ∃! n : ℕ, n > 0 ∧ 19 * n ≡ 5678 [ZMOD 11] ∧ 
  ∀ m : ℕ, m > 0 ∧ 19 * m ≡ 5678 [ZMOD 11] → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_eight_satisfies_congruence_eight_is_smallest_smallest_positive_integer_congruence_l342_34286


namespace NUMINAMATH_CALUDE_correct_multiplication_result_l342_34202

theorem correct_multiplication_result (a : ℕ) : 
  (153 * a ≠ 102325 ∧ 153 * a < 102357 ∧ 102357 - 153 * a < 153) → 
  153 * a = 102357 :=
by sorry

end NUMINAMATH_CALUDE_correct_multiplication_result_l342_34202


namespace NUMINAMATH_CALUDE_inequality_proof_l342_34275

theorem inequality_proof (a b : ℝ) (ha : a < 0) (hb : -1 < b ∧ b < 0) : a < a * b^2 ∧ a * b^2 < a * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l342_34275


namespace NUMINAMATH_CALUDE_vanessa_deleted_files_l342_34256

/-- Calculates the number of deleted files given the initial number of music files,
    initial number of video files, and the number of remaining files. -/
def deleted_files (initial_music : ℕ) (initial_video : ℕ) (remaining : ℕ) : ℕ :=
  initial_music + initial_video - remaining

/-- Theorem stating that Vanessa deleted 30 files from her flash drive. -/
theorem vanessa_deleted_files :
  deleted_files 16 48 34 = 30 := by
  sorry

end NUMINAMATH_CALUDE_vanessa_deleted_files_l342_34256


namespace NUMINAMATH_CALUDE_seven_twelfths_decimal_l342_34297

theorem seven_twelfths_decimal : 
  (7 : ℚ) / 12 = 0.5833333333333333 := by sorry

end NUMINAMATH_CALUDE_seven_twelfths_decimal_l342_34297


namespace NUMINAMATH_CALUDE_product_of_four_primes_l342_34252

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- Theorem stating the properties of the product of four specific primes -/
theorem product_of_four_primes (A B : ℕ) 
  (hA : isPrime A) 
  (hB : isPrime B) 
  (hAminusB : isPrime (A - B)) 
  (hAplusB : isPrime (A + B)) : 
  ∃ (p : ℕ), p = A * B * (A - B) * (A + B) ∧ Even p ∧ p % 3 = 0 := by
  sorry


end NUMINAMATH_CALUDE_product_of_four_primes_l342_34252


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l342_34274

/-- An arithmetic sequence {a_n} -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) (h : arithmetic_sequence a) :
  a 6 + a 8 = 16 → a 4 = 1 → a 10 = 15 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l342_34274


namespace NUMINAMATH_CALUDE_least_number_of_cookies_cookies_solution_mohan_cookies_l342_34263

theorem least_number_of_cookies (x : ℕ) : 
  (x % 6 = 5) ∧ (x % 9 = 3) ∧ (x % 11 = 7) → x ≥ 83 :=
by sorry

theorem cookies_solution : 
  (83 % 6 = 5) ∧ (83 % 9 = 3) ∧ (83 % 11 = 7) :=
by sorry

theorem mohan_cookies : 
  ∃ (x : ℕ), (x % 6 = 5) ∧ (x % 9 = 3) ∧ (x % 11 = 7) ∧ 
  (∀ (y : ℕ), (y % 6 = 5) ∧ (y % 9 = 3) ∧ (y % 11 = 7) → x ≤ y) ∧
  x = 83 :=
by sorry

end NUMINAMATH_CALUDE_least_number_of_cookies_cookies_solution_mohan_cookies_l342_34263


namespace NUMINAMATH_CALUDE_measure_union_ge_sum_measures_l342_34291

open MeasureTheory Set

-- Define the algebra structure
variable {α : Type*} [MeasurableSpace α]

-- Define the measure
variable (μ : Measure α)

-- Define the sequence of sets
variable (A : ℕ → Set α)

-- State the theorem
theorem measure_union_ge_sum_measures
  (h_algebra : ∀ n, MeasurableSet (A n))
  (h_disjoint : Pairwise (Disjoint on A))
  (h_union : MeasurableSet (⋃ n, A n)) :
  μ (⋃ n, A n) ≥ ∑' n, μ (A n) :=
sorry

end NUMINAMATH_CALUDE_measure_union_ge_sum_measures_l342_34291


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_two_l342_34287

theorem reciprocal_of_negative_two :
  ((-2 : ℝ)⁻¹ = -1/2) := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_two_l342_34287


namespace NUMINAMATH_CALUDE_modular_inverse_17_mod_19_l342_34215

theorem modular_inverse_17_mod_19 :
  ∃ x : ℕ, x ≤ 18 ∧ (17 * x) % 19 = 1 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_modular_inverse_17_mod_19_l342_34215


namespace NUMINAMATH_CALUDE_first_discount_percentage_l342_34296

def original_price : ℝ := 345
def final_price : ℝ := 227.70
def second_discount : ℝ := 0.25

theorem first_discount_percentage :
  ∃ (d : ℝ), d ≥ 0 ∧ d ≤ 1 ∧
  original_price * (1 - d) * (1 - second_discount) = final_price ∧
  d = 0.12 :=
sorry

end NUMINAMATH_CALUDE_first_discount_percentage_l342_34296


namespace NUMINAMATH_CALUDE_characterization_of_C_l342_34211

def A : Set ℝ := {x | x^2 - 6*x + 8 = 0}
def B (m : ℝ) : Set ℝ := {x | m*x - 4 = 0}
def C : Set ℝ := {m | B m ∩ A = B m}

theorem characterization_of_C : C = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_characterization_of_C_l342_34211


namespace NUMINAMATH_CALUDE_sam_total_wins_l342_34203

theorem sam_total_wins (first_period : Nat) (second_period : Nat)
  (first_win_rate : Rat) (second_win_rate : Rat) :
  first_period = 100 →
  second_period = 100 →
  first_win_rate = 1/2 →
  second_win_rate = 3/5 →
  (first_period * first_win_rate + second_period * second_win_rate : Rat) = 110 := by
  sorry

end NUMINAMATH_CALUDE_sam_total_wins_l342_34203


namespace NUMINAMATH_CALUDE_sock_profit_percentage_l342_34290

/-- Calculates the percentage profit on 4 pairs of socks given the following conditions:
  * 9 pairs of socks were bought
  * Each pair costs $2
  * $0.2 profit is made on 5 pairs
  * Total profit is $3
-/
theorem sock_profit_percentage 
  (total_pairs : Nat) 
  (cost_per_pair : ℚ) 
  (profit_on_five : ℚ) 
  (total_profit : ℚ) 
  (h1 : total_pairs = 9)
  (h2 : cost_per_pair = 2)
  (h3 : profit_on_five = 5 * (1 / 5))
  (h4 : total_profit = 3) :
  let remaining_pairs := total_pairs - 5
  let remaining_profit := total_profit - profit_on_five
  let remaining_cost := remaining_pairs * cost_per_pair
  (remaining_profit / remaining_cost) * 100 = 25 := by
sorry

end NUMINAMATH_CALUDE_sock_profit_percentage_l342_34290


namespace NUMINAMATH_CALUDE_erica_ride_time_l342_34222

/-- The time Dave can ride the merry-go-round in minutes -/
def dave_time : ℝ := 10

/-- The factor by which Chuck can ride longer than Dave -/
def chuck_factor : ℝ := 5

/-- The percentage longer Erica can stay compared to Chuck -/
def erica_percentage : ℝ := 0.3

/-- The time Chuck can ride the merry-go-round in minutes -/
def chuck_time : ℝ := dave_time * chuck_factor

/-- The time Erica can ride the merry-go-round in minutes -/
def erica_time : ℝ := chuck_time * (1 + erica_percentage)

theorem erica_ride_time : erica_time = 65 := by
  sorry

end NUMINAMATH_CALUDE_erica_ride_time_l342_34222


namespace NUMINAMATH_CALUDE_tops_count_l342_34224

-- Define the number of marbles for each person
def dennis_marbles : ℕ := 70
def kurt_marbles : ℕ := dennis_marbles - 45
def laurie_marbles : ℕ := kurt_marbles + 12
def jessica_marbles : ℕ := laurie_marbles + 25

-- Define the number of tops for each person
def laurie_tops : ℕ := laurie_marbles * 2
def kurt_tops : ℕ := kurt_marbles - 3
def dennis_tops : ℕ := dennis_marbles + 8
def jessica_tops : ℕ := jessica_marbles - 10

theorem tops_count :
  laurie_tops = 74 ∧
  kurt_tops = 22 ∧
  dennis_tops = 78 ∧
  jessica_tops = 52 := by sorry

end NUMINAMATH_CALUDE_tops_count_l342_34224


namespace NUMINAMATH_CALUDE_prob_D_is_one_fourth_l342_34201

/-- A spinner with four regions -/
structure Spinner :=
  (probA : ℚ)
  (probB : ℚ)
  (probC : ℚ)
  (probD : ℚ)

/-- The properties of our specific spinner -/
def spinner : Spinner :=
  { probA := 1/4
  , probB := 1/3
  , probC := 1/6
  , probD := 1/4 }

/-- The sum of probabilities in a spinner must equal 1 -/
axiom probability_sum (s : Spinner) : s.probA + s.probB + s.probC + s.probD = 1

/-- Theorem: Given the probabilities of A, B, and C, the probability of D is 1/4 -/
theorem prob_D_is_one_fourth :
  spinner.probA = 1/4 → spinner.probB = 1/3 → spinner.probC = 1/6 →
  spinner.probD = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_prob_D_is_one_fourth_l342_34201


namespace NUMINAMATH_CALUDE_product_of_specific_primes_l342_34225

def smallest_one_digit_primes : List Nat := [2, 3]
def largest_two_digit_prime : Nat := 97

theorem product_of_specific_primes :
  (smallest_one_digit_primes.prod * largest_two_digit_prime) = 582 := by
  sorry

end NUMINAMATH_CALUDE_product_of_specific_primes_l342_34225


namespace NUMINAMATH_CALUDE_square_root_sum_l342_34241

theorem square_root_sum (a b : ℕ+) : 
  (Real.sqrt (7 + a / b) = 7 * Real.sqrt (a / b)) → a + b = 55 := by
  sorry

end NUMINAMATH_CALUDE_square_root_sum_l342_34241


namespace NUMINAMATH_CALUDE_kindergarten_boys_count_l342_34298

/-- Given a kindergarten with a 2:3 ratio of boys to girls and 18 girls, prove there are 12 boys -/
theorem kindergarten_boys_count (total_girls : ℕ) (boys_to_girls_ratio : ℚ) : 
  total_girls = 18 → boys_to_girls_ratio = 2/3 → 
  (total_girls : ℚ) * boys_to_girls_ratio = 12 := by
sorry

end NUMINAMATH_CALUDE_kindergarten_boys_count_l342_34298


namespace NUMINAMATH_CALUDE_fifth_term_zero_l342_34247

/-- An arithmetic sequence {a_n} -/
def arithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n - d

/-- The sequence is decreasing -/
def decreasingSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) < a n

theorem fifth_term_zero
  (a : ℕ → ℝ)
  (h_arith : arithmeticSequence a)
  (h_decr : decreasingSequence a)
  (h_eq : a 1 ^ 2 = a 9 ^ 2) :
  a 5 = 0 :=
sorry

end NUMINAMATH_CALUDE_fifth_term_zero_l342_34247


namespace NUMINAMATH_CALUDE_work_completion_l342_34207

theorem work_completion (days_group1 : ℕ) (men_group2 : ℕ) (days_group2 : ℕ) :
  days_group1 = 18 →
  men_group2 = 27 →
  days_group2 = 24 →
  ∃ men_group1 : ℕ, men_group1 * days_group1 = men_group2 * days_group2 ∧ men_group1 = 36 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_l342_34207


namespace NUMINAMATH_CALUDE_five_balls_four_boxes_l342_34229

/-- The number of ways to distribute indistinguishable balls into indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 6 ways to distribute 5 indistinguishable balls into 4 indistinguishable boxes -/
theorem five_balls_four_boxes : distribute_balls 5 4 = 6 := by sorry

end NUMINAMATH_CALUDE_five_balls_four_boxes_l342_34229


namespace NUMINAMATH_CALUDE_tammy_orange_trees_l342_34281

/-- The number of oranges Tammy can pick from each tree per day -/
def oranges_per_tree_per_day : ℕ := 12

/-- The price of a 6-pack of oranges in dollars -/
def price_per_6pack : ℕ := 2

/-- The total earnings in dollars after 3 weeks -/
def total_earnings : ℕ := 840

/-- The number of days in 3 weeks -/
def days_in_3_weeks : ℕ := 21

/-- The number of orange trees Tammy has -/
def number_of_trees : ℕ := 10

theorem tammy_orange_trees :
  number_of_trees * oranges_per_tree_per_day * days_in_3_weeks =
  (total_earnings / price_per_6pack) * 6 :=
sorry

end NUMINAMATH_CALUDE_tammy_orange_trees_l342_34281


namespace NUMINAMATH_CALUDE_parallel_planes_from_skew_lines_l342_34235

-- Define the types for lines and planes in 3D space
variable (Line Plane : Type)

-- Define the parallel relation between lines and planes
variable (parallel : Line → Plane → Prop)

-- Define the parallel relation between planes
variable (planeParallel : Plane → Plane → Prop)

-- Define the skew relation between lines
variable (skew : Line → Line → Prop)

-- Theorem statement
theorem parallel_planes_from_skew_lines 
  (m n : Line) (α β : Plane) 
  (h_skew : skew m n) 
  (h_m_α : parallel m α) (h_n_α : parallel n α) 
  (h_m_β : parallel m β) (h_n_β : parallel n β) : 
  planeParallel α β :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_from_skew_lines_l342_34235


namespace NUMINAMATH_CALUDE_exponential_function_sum_of_extrema_l342_34254

theorem exponential_function_sum_of_extrema (a : ℝ) : 
  a > 0 → 
  a ≠ 1 → 
  (max (a^1) (a^2) + min (a^1) (a^2) = 6) → 
  a = 2 := by sorry

end NUMINAMATH_CALUDE_exponential_function_sum_of_extrema_l342_34254


namespace NUMINAMATH_CALUDE_chocolate_bar_distribution_l342_34251

theorem chocolate_bar_distribution (total_bars : ℕ) (total_boxes : ℕ) (bars_per_box : ℕ) :
  total_bars = 640 →
  total_boxes = 20 →
  total_bars = total_boxes * bars_per_box →
  bars_per_box = 32 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bar_distribution_l342_34251


namespace NUMINAMATH_CALUDE_shaded_area_semicircles_l342_34248

/-- The area of the shaded region formed by semicircles in a given pattern -/
theorem shaded_area_semicircles (diameter : Real) (pattern_length_feet : Real) : 
  diameter = 3 →
  pattern_length_feet = 1.5 →
  (pattern_length_feet * 12 / diameter) * (π * (diameter / 2)^2 / 2) = 13.5 * π := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_semicircles_l342_34248


namespace NUMINAMATH_CALUDE_special_triangle_base_l342_34264

/-- A triangle with specific side length properties -/
structure SpecialTriangle where
  left : ℝ
  right : ℝ
  base : ℝ
  sum_of_sides : left + right + base = 50
  right_longer : right = left + 2
  left_value : left = 12

/-- The base of a SpecialTriangle is 24 -/
theorem special_triangle_base (t : SpecialTriangle) : t.base = 24 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_base_l342_34264


namespace NUMINAMATH_CALUDE_two_special_integers_under_million_l342_34239

theorem two_special_integers_under_million : 
  ∃! (S : Finset Nat), 
    (∀ n ∈ S, n < 1000000 ∧ 
      ∃ a b : Nat, n = 2 * a^2 ∧ n = 3 * b^3) ∧ 
    S.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_special_integers_under_million_l342_34239


namespace NUMINAMATH_CALUDE_dans_work_time_l342_34253

theorem dans_work_time (D : ℝ) : 
  (∀ (annie_rate : ℝ), annie_rate = 1 / 10 →
   ∀ (dan_rate : ℝ), dan_rate = 1 / D →
   6 * dan_rate + 6 * annie_rate = 1) →
  D = 15 := by
sorry

end NUMINAMATH_CALUDE_dans_work_time_l342_34253


namespace NUMINAMATH_CALUDE_words_per_page_l342_34204

theorem words_per_page (total_pages : Nat) (max_words_per_page : Nat) (total_words_mod : Nat) :
  total_pages = 136 →
  max_words_per_page = 100 →
  total_words_mod = 184 →
  ∃ (words_per_page : Nat),
    words_per_page ≤ max_words_per_page ∧
    (total_pages * words_per_page) % 203 = total_words_mod ∧
    words_per_page = 73 := by
  sorry

end NUMINAMATH_CALUDE_words_per_page_l342_34204


namespace NUMINAMATH_CALUDE_rectangular_solid_diagonal_l342_34258

theorem rectangular_solid_diagonal (a b c : ℝ) 
  (h1 : 2 * (a * b + b * c + a * c) = 30)
  (h2 : 4 * (a + b + c) = 28) :
  (a^2 + b^2 + c^2).sqrt = (19 : ℝ).sqrt := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_diagonal_l342_34258


namespace NUMINAMATH_CALUDE_hyperbola_solutions_l342_34270

-- Define the hyperbola equation
def hyperbola (x y : ℤ) : Prop := x^2 - y^2 = 2500^2

-- Define a function to count the number of integer solutions
def count_solutions : ℕ := sorry

-- Theorem stating that the number of solutions is 70
theorem hyperbola_solutions : count_solutions = 70 := by sorry

end NUMINAMATH_CALUDE_hyperbola_solutions_l342_34270


namespace NUMINAMATH_CALUDE_intersection_slope_l342_34223

/-- Given two lines m and n that intersect at (-4, 0), prove that the slope of line n is -9/4 -/
theorem intersection_slope (k : ℚ) : 
  (∀ x y, y = 2 * x + 8 → y = k * x - 9 → x = -4 ∧ y = 0) → 
  k = -9/4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_slope_l342_34223


namespace NUMINAMATH_CALUDE_cheapest_for_second_caterer_l342_34242

-- Define the pricing functions for both caterers
def first_caterer (x : ℕ) : ℕ := 150 + 18 * x

def second_caterer (x : ℕ) : ℕ :=
  if x ≤ 30 then 250 + 15 * x
  else 400 + 10 * x

-- Define a function to compare the prices
def second_cheaper (x : ℕ) : Prop :=
  second_caterer x < first_caterer x

-- Theorem statement
theorem cheapest_for_second_caterer :
  ∀ n : ℕ, n < 32 → ¬(second_cheaper n) ∧ second_cheaper 32 :=
sorry

end NUMINAMATH_CALUDE_cheapest_for_second_caterer_l342_34242


namespace NUMINAMATH_CALUDE_solve_square_root_equation_l342_34285

theorem solve_square_root_equation (x : ℝ) : 
  Real.sqrt (5 * x + 9) = 12 → x = 27 := by sorry

end NUMINAMATH_CALUDE_solve_square_root_equation_l342_34285


namespace NUMINAMATH_CALUDE_solution_of_equation_l342_34218

theorem solution_of_equation (x : ℝ) : (1 / (3 * x) = 2 / (x + 5)) ↔ x = 1 :=
by sorry

end NUMINAMATH_CALUDE_solution_of_equation_l342_34218


namespace NUMINAMATH_CALUDE_position_function_correct_l342_34260

/-- The velocity function --/
def v (t : ℝ) : ℝ := 3 * t^2 - 1

/-- The position function --/
def s (t : ℝ) : ℝ := t^3 - t + 0.05

/-- Theorem stating that s is the correct position function --/
theorem position_function_correct :
  (∀ t, (deriv s) t = v t) ∧ s 0 = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_position_function_correct_l342_34260


namespace NUMINAMATH_CALUDE_garden_walkway_area_l342_34212

/-- Calculates the total area of walkways in a garden with specified dimensions and layout. -/
def walkway_area (rows : ℕ) (columns : ℕ) (bed_width : ℕ) (bed_height : ℕ) (walkway_width : ℕ) : ℕ :=
  let total_width := columns * bed_width + (columns + 1) * walkway_width
  let total_height := rows * bed_height + (rows + 1) * walkway_width
  let total_area := total_width * total_height
  let bed_area := rows * columns * bed_width * bed_height
  total_area - bed_area

theorem garden_walkway_area :
  walkway_area 4 3 8 3 2 = 416 := by
  sorry

end NUMINAMATH_CALUDE_garden_walkway_area_l342_34212


namespace NUMINAMATH_CALUDE_max_min_product_l342_34284

theorem max_min_product (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c)
  (h4 : a + b + c = 8) (h5 : a * b + b * c + c * a = 16) :
  ∃ m : ℝ, m = min (a * b) (min (b * c) (c * a)) ∧ m ≤ 16 / 9 ∧
  ∃ a' b' c' : ℝ, 0 ≤ a' ∧ 0 ≤ b' ∧ 0 ≤ c' ∧
  a' + b' + c' = 8 ∧ a' * b' + b' * c' + c' * a' = 16 ∧
  min (a' * b') (min (b' * c') (c' * a')) = 16 / 9 := by
  sorry

end NUMINAMATH_CALUDE_max_min_product_l342_34284


namespace NUMINAMATH_CALUDE_divisors_of_600_l342_34250

theorem divisors_of_600 : Nat.card {d : ℕ | d ∣ 600} = 24 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_600_l342_34250


namespace NUMINAMATH_CALUDE_highest_score_calculation_l342_34259

theorem highest_score_calculation (scores : Finset ℕ) (lowest highest : ℕ) :
  Finset.card scores = 15 →
  (Finset.sum scores id) / 15 = 90 →
  ((Finset.sum scores id) - lowest - highest) / 13 = 92 →
  lowest = 65 →
  highest = 89 := by
  sorry

end NUMINAMATH_CALUDE_highest_score_calculation_l342_34259


namespace NUMINAMATH_CALUDE_sin_cos_difference_l342_34273

theorem sin_cos_difference (θ₁ θ₂ θ₃ θ₄ : Real) 
  (h₁ : θ₁ = 17 * π / 180)
  (h₂ : θ₂ = 47 * π / 180)
  (h₃ : θ₃ = 73 * π / 180)
  (h₄ : θ₄ = 43 * π / 180) : 
  Real.sin θ₁ * Real.cos θ₂ - Real.sin θ₃ * Real.cos θ₄ = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_difference_l342_34273


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l342_34228

theorem arithmetic_sequence_length : 
  ∀ (a d last : ℕ), 
  a = 3 → d = 3 → last = 198 → 
  ∃ n : ℕ, n = 66 ∧ last = a + (n - 1) * d :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l342_34228


namespace NUMINAMATH_CALUDE_square_sum_fifteen_l342_34280

theorem square_sum_fifteen (x y : ℝ) 
  (h1 : y + 4 = (x - 2)^2) 
  (h2 : x + 4 = (y - 2)^2) 
  (h3 : x ≠ y) : 
  x^2 + y^2 = 15 := by
sorry

end NUMINAMATH_CALUDE_square_sum_fifteen_l342_34280


namespace NUMINAMATH_CALUDE_player_b_wins_l342_34232

/-- Represents a chessboard --/
def Chessboard := Fin 8 → Fin 8 → Option Bool

/-- Represents a position on the chessboard --/
def Position := Fin 8 × Fin 8

/-- Checks if a bishop can be captured at a given position --/
def canBeCaptured (board : Chessboard) (pos : Position) : Prop :=
  sorry

/-- Represents a valid move in the game --/
def ValidMove (board : Chessboard) (pos : Position) : Prop :=
  board pos.1 pos.2 = none ∧ ¬canBeCaptured board pos

/-- Represents the state of the game --/
structure GameState where
  board : Chessboard
  playerATurn : Bool

/-- Represents a strategy for a player --/
def Strategy := GameState → Position

/-- Checks if a strategy is winning for Player B --/
def isWinningStrategyForB (s : Strategy) : Prop :=
  sorry

/-- The main theorem stating that Player B has a winning strategy --/
theorem player_b_wins : ∃ s : Strategy, isWinningStrategyForB s :=
  sorry

end NUMINAMATH_CALUDE_player_b_wins_l342_34232


namespace NUMINAMATH_CALUDE_arithmetic_sequence_solution_l342_34267

theorem arithmetic_sequence_solution (x : ℝ) (h1 : x ≠ 0) (h2 : x < 4) : 
  (⌊x⌋ + 1 - x + ⌊x⌋ = x + 1 - (⌊x⌋ + 1)) → 
  (x = 0.5 ∨ x = 1.5 ∨ x = 2.5 ∨ x = 3.5) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_solution_l342_34267


namespace NUMINAMATH_CALUDE_bryan_collected_from_four_continents_l342_34257

/-- The number of books Bryan collected per continent -/
def books_per_continent : ℕ := 122

/-- The total number of books Bryan collected -/
def total_books : ℕ := 488

/-- The number of continents Bryan collected books from -/
def num_continents : ℕ := total_books / books_per_continent

theorem bryan_collected_from_four_continents :
  num_continents = 4 := by sorry

end NUMINAMATH_CALUDE_bryan_collected_from_four_continents_l342_34257


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l342_34276

-- Define the sets A and B
def A : Set ℝ := {x | x - 2 < 3}
def B : Set ℝ := {x | 2*x - 3 < 3*x - 2}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | -1 < x ∧ x < 5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l342_34276


namespace NUMINAMATH_CALUDE_complex_power_difference_abs_l342_34214

def i : ℂ := Complex.I

theorem complex_power_difference_abs : 
  Complex.abs ((2 + i)^18 - (2 - i)^18) = 19531250 := by sorry

end NUMINAMATH_CALUDE_complex_power_difference_abs_l342_34214


namespace NUMINAMATH_CALUDE_toms_flying_robots_l342_34246

theorem toms_flying_robots (michael_robots : ℕ) (tom_robots : ℕ) : 
  michael_robots = 12 →
  michael_robots = 4 * tom_robots →
  tom_robots = 3 := by
sorry

end NUMINAMATH_CALUDE_toms_flying_robots_l342_34246


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l342_34277

/-- The sum of a geometric sequence with 6 terms, initial term 10, and common ratio 3 is 3640 -/
theorem geometric_sequence_sum : 
  let a : ℕ := 10  -- initial term
  let r : ℕ := 3   -- common ratio
  let n : ℕ := 6   -- number of terms
  a * (r^n - 1) / (r - 1) = 3640 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l342_34277


namespace NUMINAMATH_CALUDE_periodic_function_l342_34278

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Real.sin (Real.pi * x + φ)

theorem periodic_function (φ : ℝ) :
  ∃ t : ℝ, t ≠ 0 ∧ ∀ x : ℝ, f (x + t) φ = f x φ :=
by
  use 2
  sorry

end NUMINAMATH_CALUDE_periodic_function_l342_34278


namespace NUMINAMATH_CALUDE_extreme_value_difference_l342_34237

noncomputable def f (a b x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*b*x

theorem extreme_value_difference (a b : ℝ) :
  (∃ x, x = 2 ∧ (deriv (f a b)) x = 0) →
  (deriv (f a b)) 1 = -3 →
  ∃ max min, (∀ x, f a b x ≤ max) ∧ 
              (∀ x, f a b x ≥ min) ∧ 
              max - min = 4 :=
by sorry

end NUMINAMATH_CALUDE_extreme_value_difference_l342_34237


namespace NUMINAMATH_CALUDE_distance_between_roots_l342_34234

/-- The distance between the roots of x^2 - 2x - 3 = 0 is 4 -/
theorem distance_between_roots : ∃ x₁ x₂ : ℝ, 
  x₁^2 - 2*x₁ - 3 = 0 ∧ 
  x₂^2 - 2*x₂ - 3 = 0 ∧ 
  x₁ ≠ x₂ ∧
  |x₁ - x₂| = 4 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_roots_l342_34234


namespace NUMINAMATH_CALUDE_triangle_inequalities_triangle_equality_condition_l342_34226

/-- Triangle properties -/
structure Triangle :=
  (a b c : ℝ)
  (r R : ℝ)
  (h_a h_b h_c : ℝ)
  (β_a β_b β_c : ℝ)
  (m_a m_b m_c : ℝ)
  (r_a r_b r_c : ℝ)
  (p : ℝ)

/-- Main theorem -/
theorem triangle_inequalities (t : Triangle) :
  (9 * t.r ≤ t.h_a + t.h_b + t.h_c) ∧
  (t.h_a + t.h_b + t.h_c ≤ t.β_a + t.β_b + t.β_c) ∧
  (t.β_a + t.β_b + t.β_c ≤ t.m_a + t.m_b + t.m_c) ∧
  (t.m_a + t.m_b + t.m_c ≤ 9/2 * t.R) ∧
  (t.β_a + t.β_b + t.β_c ≤ Real.sqrt (t.r_a * t.r_b) + Real.sqrt (t.r_b * t.r_c) + Real.sqrt (t.r_c * t.r_a)) ∧
  (Real.sqrt (t.r_a * t.r_b) + Real.sqrt (t.r_b * t.r_c) + Real.sqrt (t.r_c * t.r_a) ≤ t.p * Real.sqrt 3) ∧
  (t.p * Real.sqrt 3 ≤ t.r_a + t.r_b + t.r_c) ∧
  (t.r_a + t.r_b + t.r_c = t.r + 4 * t.R) ∧
  (27 * t.r^2 ≤ t.h_a^2 + t.h_b^2 + t.h_c^2) ∧
  (t.h_a^2 + t.h_b^2 + t.h_c^2 ≤ t.β_a^2 + t.β_b^2 + t.β_c^2) ∧
  (t.β_a^2 + t.β_b^2 + t.β_c^2 ≤ t.p^2) ∧
  (t.p^2 ≤ t.m_a^2 + t.m_b^2 + t.m_c^2) ∧
  (t.m_a^2 + t.m_b^2 + t.m_c^2 = 3/4 * (t.a^2 + t.b^2 + t.c^2)) ∧
  (3/4 * (t.a^2 + t.b^2 + t.c^2) ≤ 27/4 * t.R^2) ∧
  (1/t.r = 1/t.r_a + 1/t.r_b + 1/t.r_c) ∧
  (1/t.r = 1/t.h_a + 1/t.h_b + 1/t.h_c) ∧
  (1/t.h_a + 1/t.h_b + 1/t.h_c ≥ 1/t.β_a + 1/t.β_b + 1/t.β_c) ∧
  (1/t.β_a + 1/t.β_b + 1/t.β_c ≥ 1/t.m_a + 1/t.m_b + 1/t.m_c) ∧
  (1/t.m_a + 1/t.m_b + 1/t.m_c ≥ 2/t.R) :=
sorry

/-- Equality condition -/
theorem triangle_equality_condition (t : Triangle) :
  (9 * t.r = t.h_a + t.h_b + t.h_c) ∧
  (t.h_a + t.h_b + t.h_c = t.β_a + t.β_b + t.β_c) ∧
  (t.β_a + t.β_b + t.β_c = t.m_a + t.m_b + t.m_c) ∧
  (t.m_a + t.m_b + t.m_c = 9/2 * t.R) ∧
  (t.β_a + t.β_b + t.β_c = Real.sqrt (t.r_a * t.r_b) + Real.sqrt (t.r_b * t.r_c) + Real.sqrt (t.r_c * t.r_a)) ∧
  (Real.sqrt (t.r_a * t.r_b) + Real.sqrt (t.r_b * t.r_c) + Real.sqrt (t.r_c * t.r_a) = t.p * Real.sqrt 3) ∧
  (t.p * Real.sqrt 3 = t.r_a + t.r_b + t.r_c) ∧
  (27 * t.r^2 = t.h_a^2 + t.h_b^2 + t.h_c^2) ∧
  (t.h_a^2 + t.h_b^2 + t.h_c^2 = t.β_a^2 + t.β_b^2 + t.β_c^2) ∧
  (t.β_a^2 + t.β_b^2 + t.β_c^2 = t.p^2) ∧
  (t.p^2 = t.m_a^2 + t.m_b^2 + t.m_c^2) ∧
  (3/4 * (t.a^2 + t.b^2 + t.c^2) = 27/4 * t.R^2) ∧
  (1/t.h_a + 1/t.h_b + 1/t.h_c = 1/t.β_a + 1/t.β_b + 1/t.β_c) ∧
  (1/t.β_a + 1/t.β_b + 1/t.β_c = 1/t.m_a + 1/t.m_b + 1/t.m_c) ∧
  (1/t.m_a + 1/t.m_b + 1/t.m_c = 2/t.R) ↔
  (t.a = t.b ∧ t.b = t.c) :=
sorry

end NUMINAMATH_CALUDE_triangle_inequalities_triangle_equality_condition_l342_34226


namespace NUMINAMATH_CALUDE_exponential_function_property_l342_34283

theorem exponential_function_property (a : ℝ) (ha : a > 0 ∧ a ≠ 1) :
  ∀ x y : ℝ, (fun x => a^x) (x + y) = (fun x => a^x) x * (fun x => a^x) y :=
by sorry

end NUMINAMATH_CALUDE_exponential_function_property_l342_34283


namespace NUMINAMATH_CALUDE_sum_of_fifty_eights_l342_34205

theorem sum_of_fifty_eights : (List.replicate 50 8).sum = 400 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fifty_eights_l342_34205


namespace NUMINAMATH_CALUDE_solution_pair_l342_34206

theorem solution_pair : ∃ (x y : ℝ), 
  (2 * x + 3 * y = (7 - x) + (7 - y)) ∧ 
  (x - 2 * y = (x - 3) + (y - 3)) ∧ 
  x = 2 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_pair_l342_34206


namespace NUMINAMATH_CALUDE_softball_players_count_l342_34268

theorem softball_players_count (cricket hockey football total : ℕ) 
  (h1 : cricket = 16)
  (h2 : hockey = 12)
  (h3 : football = 18)
  (h4 : total = 59) :
  total - (cricket + hockey + football) = 13 := by
  sorry

end NUMINAMATH_CALUDE_softball_players_count_l342_34268


namespace NUMINAMATH_CALUDE_f_of_3_eq_5_l342_34221

/-- The function f defined on ℝ -/
def f : ℝ → ℝ := fun x ↦ 2 * x - 1

/-- Theorem: f(3) = 5 -/
theorem f_of_3_eq_5 : f 3 = 5 := by sorry

end NUMINAMATH_CALUDE_f_of_3_eq_5_l342_34221


namespace NUMINAMATH_CALUDE_exists_pair_satisfying_condition_l342_34262

theorem exists_pair_satisfying_condition (r : Fin 5 → ℝ) : 
  ∃ (i j : Fin 5), i ≠ j ∧ 0 ≤ (r i - r j) / (1 + r i * r j) ∧ (r i - r j) / (1 + r i * r j) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_exists_pair_satisfying_condition_l342_34262


namespace NUMINAMATH_CALUDE_exponential_inequality_l342_34236

theorem exponential_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (1/2 : ℝ)^a < (1/2 : ℝ)^b := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l342_34236


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_l342_34209

theorem necessary_not_sufficient (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ a b, e^a + 2*a = e^b + 3*b → a > b) ∧
  (∃ a b, a > b ∧ e^a + 2*a ≠ e^b + 3*b) :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_l342_34209


namespace NUMINAMATH_CALUDE_car_travel_distance_l342_34279

-- Define the initial conditions
def initial_distance : ℝ := 180
def initial_time : ℝ := 4
def next_time : ℝ := 3

-- Define the theorem
theorem car_travel_distance :
  let speed := initial_distance / initial_time
  speed * next_time = 135 := by
  sorry

end NUMINAMATH_CALUDE_car_travel_distance_l342_34279


namespace NUMINAMATH_CALUDE_select_students_l342_34208

theorem select_students (n_boys : ℕ) (n_girls : ℕ) (n_select : ℕ) : 
  n_boys = 4 → n_girls = 2 → n_select = 4 →
  (Nat.choose n_boys (n_select - 1) * Nat.choose n_girls 1 + 
   Nat.choose n_boys (n_select - 2) * Nat.choose n_girls 2) = 14 := by
sorry

end NUMINAMATH_CALUDE_select_students_l342_34208


namespace NUMINAMATH_CALUDE_part_one_part_two_l342_34269

/-- The set A -/
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 3}

/-- The set B -/
def B (m : ℝ) : Set ℝ := {x | 2 - m ≤ x ∧ x ≤ 2*m - 3}

/-- Part 1: p is sufficient but not necessary for q -/
theorem part_one (m : ℝ) : (∀ x, x ∈ A → x ∈ B m) ∧ (∃ x, x ∈ B m ∧ x ∉ A) → m ≥ 4 := by
  sorry

/-- Part 2: A ∪ B = A -/
theorem part_two (m : ℝ) : A ∪ B m = A → m ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l342_34269


namespace NUMINAMATH_CALUDE_expression_evaluation_l342_34255

theorem expression_evaluation (x y z : ℝ) 
  (hz : z = y - 11)
  (hy : y = x + 3)
  (hx : x = 5) :
  ((x + 3) / (x + 2)) * ((y - 2) / (y - 3)) * ((z + 9) / (z + 7)) = 72 / 35 ∧
  x + 2 ≠ 0 ∧ y - 3 ≠ 0 ∧ z + 7 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l342_34255


namespace NUMINAMATH_CALUDE_fast_food_cost_l342_34240

theorem fast_food_cost (burger shake cola : ℝ) : 
  (3 * burger + 7 * shake + cola = 120) →
  (4 * burger + 10 * shake + cola = 160.5) →
  (burger + shake + cola = 39) :=
by sorry

end NUMINAMATH_CALUDE_fast_food_cost_l342_34240


namespace NUMINAMATH_CALUDE_tank_capacity_l342_34294

theorem tank_capacity : 
  ∀ (capacity : ℝ),
  (capacity / 4 + 150 = 2 * capacity / 3) →
  capacity = 360 := by
sorry

end NUMINAMATH_CALUDE_tank_capacity_l342_34294


namespace NUMINAMATH_CALUDE_chocolate_box_problem_l342_34265

/-- Calculates the number of additional boxes needed to store chocolates --/
def additional_boxes_needed (total_chocolates : ℕ) (chocolates_not_in_box : ℕ) (existing_boxes : ℕ) (friend_chocolates : ℕ) : ℕ :=
  let chocolates_in_boxes := total_chocolates - chocolates_not_in_box
  let total_chocolates_to_box := chocolates_in_boxes + friend_chocolates
  let chocolates_per_box := chocolates_in_boxes / existing_boxes
  let total_boxes_needed := (total_chocolates_to_box + chocolates_per_box - 1) / chocolates_per_box
  total_boxes_needed - existing_boxes

theorem chocolate_box_problem :
  additional_boxes_needed 50 5 3 25 = 2 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_box_problem_l342_34265


namespace NUMINAMATH_CALUDE_real_part_of_z_l342_34200

theorem real_part_of_z (z : ℂ) (h1 : Complex.abs z = 1) (h2 : Complex.abs (z - 1.45) = 1.05) :
  z.re = 20 / 29 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_z_l342_34200


namespace NUMINAMATH_CALUDE_total_sheets_required_l342_34282

/-- The number of letters in the English alphabet -/
def alphabet_size : ℕ := 26

/-- The number of times each letter needs to be written -/
def writing_times : ℕ := 3

/-- The number of sheets needed for one writing of a letter -/
def sheets_per_writing : ℕ := 1

/-- Theorem: The total number of sheets required to write each letter of the English alphabet
    three times (uppercase, lowercase, and cursive script) is 78. -/
theorem total_sheets_required :
  alphabet_size * writing_times * sheets_per_writing = 78 := by sorry

end NUMINAMATH_CALUDE_total_sheets_required_l342_34282


namespace NUMINAMATH_CALUDE_min_value_theorem_l342_34245

theorem min_value_theorem (x y : ℝ) (h : x * y > 0) :
  ∃ m : ℝ, m = 4 - 2 * Real.sqrt 2 ∧
    ∀ z : ℝ, z = y / (x + y) + 2 * x / (2 * x + y) → z ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l342_34245


namespace NUMINAMATH_CALUDE_power_function_through_point_value_l342_34210

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x ^ b

theorem power_function_through_point_value :
  ∀ f : ℝ → ℝ,
  isPowerFunction f →
  f 2 = 8 →
  f 3 = 27 := by
sorry

end NUMINAMATH_CALUDE_power_function_through_point_value_l342_34210


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l342_34293

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (∃ r s : ℝ, (36 - 18 * x - x^2 = 0 ↔ x = r ∨ x = s) ∧ r + s = 18) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l342_34293


namespace NUMINAMATH_CALUDE_ellipse_focus_d_value_l342_34272

/-- Definition of an ellipse with given properties -/
structure Ellipse where
  /-- The ellipse is in the first quadrant -/
  first_quadrant : Bool
  /-- The ellipse is tangent to both x-axis and y-axis -/
  tangent_to_axes : Bool
  /-- One focus of the ellipse -/
  focus1 : ℝ × ℝ
  /-- The other focus of the ellipse -/
  focus2 : ℝ × ℝ

/-- Theorem stating the value of d for the given ellipse -/
theorem ellipse_focus_d_value (e : Ellipse) : 
  e.first_quadrant = true ∧ 
  e.tangent_to_axes = true ∧ 
  e.focus1 = (4, 10) ∧ 
  e.focus2.1 = e.focus2.2 ∧ 
  e.focus2.2 = 10 → 
  e.focus2.1 = 25 := by
sorry

end NUMINAMATH_CALUDE_ellipse_focus_d_value_l342_34272


namespace NUMINAMATH_CALUDE_trig_simplification_l342_34217

theorem trig_simplification (α : ℝ) : 
  (1 + Real.sin (4 * α) - Real.cos (4 * α)) / (1 + Real.sin (4 * α) + Real.cos (4 * α)) = Real.tan (2 * α) := by
  sorry

end NUMINAMATH_CALUDE_trig_simplification_l342_34217


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l342_34299

-- Define the repeating decimal 6.8181...
def repeating_decimal : ℚ := 6 + 81 / 99

-- Theorem stating that the repeating decimal equals 75/11
theorem repeating_decimal_equals_fraction : repeating_decimal = 75 / 11 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l342_34299


namespace NUMINAMATH_CALUDE_age_difference_l342_34230

/-- The difference in total ages of (A,B) and (B,C) given C is 10 years younger than A -/
theorem age_difference (A B C : ℕ) (h : C = A - 10) : A + B - (B + C) = 10 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l342_34230


namespace NUMINAMATH_CALUDE_card_numbers_proof_l342_34295

def is_valid_sequence (seq : List ℕ) : Prop :=
  seq.length = 9 ∧
  (∀ n, n ∈ seq → 1 ≤ n ∧ n ≤ 9) ∧
  (∀ i, i < seq.length - 2 → 
    ¬(seq[i]! < seq[i+1]! ∧ seq[i+1]! < seq[i+2]!) ∧
    ¬(seq[i]! > seq[i+1]! ∧ seq[i+1]! > seq[i+2]!))

def visible_sequence : List ℕ := [1, 3, 4, 6, 7, 8]

theorem card_numbers_proof :
  ∀ (seq : List ℕ),
  is_valid_sequence seq →
  seq.take 1 = [1] →
  seq.drop 1 = 3 :: visible_sequence.drop 2 →
  seq[1]! = 5 ∧ seq[4]! = 2 ∧ seq[5]! = 9 :=
sorry

end NUMINAMATH_CALUDE_card_numbers_proof_l342_34295


namespace NUMINAMATH_CALUDE_total_wage_calculation_l342_34231

/-- Represents the number of days it takes for a worker to complete the job alone -/
structure WorkerSpeed :=
  (days : ℕ)

/-- Calculates the daily work rate of a worker -/
def dailyRate (w : WorkerSpeed) : ℚ :=
  1 / w.days

/-- Represents the wage distribution between two workers -/
structure WageDistribution :=
  (worker_a : ℚ)
  (total : ℚ)

theorem total_wage_calculation 
  (speed_a : WorkerSpeed)
  (speed_b : WorkerSpeed)
  (wage_dist : WageDistribution)
  (h1 : speed_a.days = 10)
  (h2 : speed_b.days = 15)
  (h3 : wage_dist.worker_a = 1980)
  : wage_dist.total = 3300 :=
sorry

end NUMINAMATH_CALUDE_total_wage_calculation_l342_34231


namespace NUMINAMATH_CALUDE_elliptical_track_distance_l342_34266

/-- Represents the properties of an elliptical track with two objects moving on it. -/
structure EllipticalTrack where
  /-- Half the circumference of the track in yards -/
  half_circumference : ℝ
  /-- Distance traveled by object B at first meeting in yards -/
  first_meeting_distance : ℝ
  /-- Distance object A is short of completing a lap at second meeting in yards -/
  second_meeting_shortfall : ℝ

/-- Theorem stating the total distance around the track given specific conditions -/
theorem elliptical_track_distance 
  (track : EllipticalTrack)
  (h1 : track.first_meeting_distance = 150)
  (h2 : track.second_meeting_shortfall = 90)
  (h3 : (track.first_meeting_distance) / (track.half_circumference - track.first_meeting_distance) = 
        (track.half_circumference + track.second_meeting_shortfall) / 
        (2 * track.half_circumference - track.second_meeting_shortfall)) :
  2 * track.half_circumference = 720 := by
  sorry


end NUMINAMATH_CALUDE_elliptical_track_distance_l342_34266


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l342_34227

theorem sqrt_x_minus_one_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 1) ↔ x ≥ 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l342_34227


namespace NUMINAMATH_CALUDE_ellipse_dimensions_l342_34216

/-- Given an ellipse with semi-major axis a and semi-minor axis b,
    where the intersection of lines AB and CF is at (3a, 16),
    prove that a = 5 and b = 4 -/
theorem ellipse_dimensions (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (∃ c : ℝ, a^2 = b^2 + c^2) →
  (∃ x y : ℝ, x = 3*a ∧ y = 16 ∧ x/(-a) + y/b = 1 ∧ x/c + y/(-b) = 1) →
  a = 5 ∧ b = 4 := by sorry

end NUMINAMATH_CALUDE_ellipse_dimensions_l342_34216


namespace NUMINAMATH_CALUDE_city_distance_proof_l342_34219

/-- Calculates the actual distance between two cities given the map distance and scale. -/
def actual_distance (map_distance : ℝ) (scale : ℝ) : ℝ :=
  map_distance * scale

/-- Proves that the actual distance between two cities is 2400 km given the map conditions. -/
theorem city_distance_proof (map_distance : ℝ) (scale : ℝ) 
  (h1 : map_distance = 120)
  (h2 : scale = 20) : 
  actual_distance map_distance scale = 2400 := by
  sorry

#check city_distance_proof

end NUMINAMATH_CALUDE_city_distance_proof_l342_34219


namespace NUMINAMATH_CALUDE_tan_alpha_eq_one_l342_34249

theorem tan_alpha_eq_one (α : Real) 
  (h : (Real.sin α + Real.cos α) / (2 * Real.sin α - Real.cos α) = 2) : 
  Real.tan α = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_eq_one_l342_34249


namespace NUMINAMATH_CALUDE_tangent_line_cubic_curve_l342_34213

/-- Given a cubic function f(x) = x³ + ax + b and a line g(x) = kx + 1 tangent to f at x = 1,
    prove that 2a + b = 1 when f(1) = 3. -/
theorem tangent_line_cubic_curve (a b k : ℝ) : 
  (∀ x, (x^3 + a*x + b) = 3 * x^2 + a) →  -- Derivative condition
  (1^3 + a*1 + b = 3) →                   -- Point (1, 3) lies on the curve
  (k*1 + 1 = 3) →                         -- Point (1, 3) lies on the line
  (k = 3*1^2 + a) →                       -- Slope of tangent equals derivative at x = 1
  (2*a + b = 1) := by
sorry

end NUMINAMATH_CALUDE_tangent_line_cubic_curve_l342_34213


namespace NUMINAMATH_CALUDE_visiting_students_theorem_l342_34288

/-- Represents a set of students visiting each other's homes -/
structure VisitingStudents where
  n : ℕ  -- number of students
  d : ℕ  -- number of days
  assignment : Fin n → Finset (Fin d)

/-- A valid assignment means no subset is contained within another subset -/
def ValidAssignment (vs : VisitingStudents) : Prop :=
  ∀ i j : Fin vs.n, i ≠ j → ¬(vs.assignment i ⊆ vs.assignment j)

theorem visiting_students_theorem :
  (¬∃ vs : VisitingStudents, vs.n = 30 ∧ vs.d = 4 ∧ ValidAssignment vs) ∧
  (¬∃ vs : VisitingStudents, vs.n = 30 ∧ vs.d = 5 ∧ ValidAssignment vs) ∧
  (∃ vs : VisitingStudents, vs.n = 30 ∧ vs.d = 7 ∧ ValidAssignment vs) ∧
  (∃ vs : VisitingStudents, vs.n = 30 ∧ vs.d = 10 ∧ ValidAssignment vs) :=
by sorry

end NUMINAMATH_CALUDE_visiting_students_theorem_l342_34288


namespace NUMINAMATH_CALUDE_initial_geese_count_l342_34243

/-- Given that 28 geese flew away and 23 geese remain in a field,
    prove that there were initially 51 geese in the field. -/
theorem initial_geese_count (flew_away : ℕ) (remaining : ℕ) : 
  flew_away = 28 → remaining = 23 → flew_away + remaining = 51 := by
  sorry

end NUMINAMATH_CALUDE_initial_geese_count_l342_34243


namespace NUMINAMATH_CALUDE_total_crayons_l342_34261

def new_crayons : ℕ := 2
def used_crayons : ℕ := 4
def broken_crayons : ℕ := 8

theorem total_crayons : new_crayons + used_crayons + broken_crayons = 14 := by
  sorry

end NUMINAMATH_CALUDE_total_crayons_l342_34261


namespace NUMINAMATH_CALUDE_reliability_comparison_l342_34244

/-- Probability of a 3-member system making a correct decision -/
def prob_3_correct (p : ℝ) : ℝ := 3 * p^2 * (1 - p) + p^3

/-- Probability of a 5-member system making a correct decision -/
def prob_5_correct (p : ℝ) : ℝ := 10 * p^3 * (1 - p)^2 + 5 * p^4 * (1 - p) + p^5

/-- A 5-member system is more reliable than a 3-member system -/
def more_reliable (p : ℝ) : Prop := prob_5_correct p > prob_3_correct p

theorem reliability_comparison (p : ℝ) (h1 : 0 < p) (h2 : p < 1) :
  more_reliable p ↔ p > (1/2 : ℝ) := by sorry

end NUMINAMATH_CALUDE_reliability_comparison_l342_34244


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l342_34233

theorem imaginary_part_of_z (i : ℂ) (h : i^2 = -1) : 
  let z : ℂ := 2 / (-1 + i)
  Complex.im z = -1 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l342_34233


namespace NUMINAMATH_CALUDE_vector_dot_product_cos_2x_l342_34238

theorem vector_dot_product_cos_2x (x : ℝ) : 
  let a := (Real.sqrt 3 * Real.sin x, Real.cos x)
  let b := (Real.cos x, -Real.cos x)
  x ∈ Set.Ioo (7 * Real.pi / 12) (5 * Real.pi / 6) →
  a.1 * b.1 + a.2 * b.2 = -5/4 →
  Real.cos (2 * x) = (3 - Real.sqrt 21) / 8 := by
    sorry

end NUMINAMATH_CALUDE_vector_dot_product_cos_2x_l342_34238


namespace NUMINAMATH_CALUDE_fraction_equality_l342_34271

theorem fraction_equality (p q s u : ℚ) 
  (h1 : p / q = 3 / 5) 
  (h2 : s / u = 8 / 11) : 
  (4 * p * s - 3 * q * u) / (5 * q * u - 8 * p * s) = -69 / 83 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l342_34271
