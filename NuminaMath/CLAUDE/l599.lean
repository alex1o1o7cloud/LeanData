import Mathlib

namespace NUMINAMATH_CALUDE_cube_difference_negative_l599_59929

theorem cube_difference_negative {a b : ℝ} (ha : a ≠ 0) (hb : b ≠ 0) (h : a < b) : a^3 - b^3 < 0 := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_negative_l599_59929


namespace NUMINAMATH_CALUDE_expression_evaluation_l599_59946

theorem expression_evaluation (c : ℕ) (h : c = 4) :
  (c^c - c*(c-1)^(c-1))^(c-1) = 3241792 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l599_59946


namespace NUMINAMATH_CALUDE_min_value_on_circle_l599_59908

theorem min_value_on_circle (x y : ℝ) :
  (x - 1)^2 + (y + 2)^2 = 4 →
  ∃ (S : ℝ), S = 3*x - y ∧ S ≥ -5 - 2*Real.sqrt 10 ∧
  ∀ (S' : ℝ), (∃ (x' y' : ℝ), (x' - 1)^2 + (y' + 2)^2 = 4 ∧ S' = 3*x' - y') →
  S' ≥ -5 - 2*Real.sqrt 10 :=
sorry

end NUMINAMATH_CALUDE_min_value_on_circle_l599_59908


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l599_59993

/-- The polynomial to be divided -/
def P (z : ℝ) : ℝ := 4*z^4 - 3*z^3 + 2*z^2 - 16*z + 9

/-- The divisor polynomial -/
def D (z : ℝ) : ℝ := 4*z + 6

/-- The theorem stating that the remainder of P(z) divided by D(z) is 173/12 -/
theorem polynomial_division_remainder :
  ∃ Q : ℝ → ℝ, ∀ z : ℝ, P z = D z * Q z + 173/12 :=
sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l599_59993


namespace NUMINAMATH_CALUDE_distinct_prime_factors_of_divisor_sum_360_l599_59909

/-- The sum of positive divisors of a natural number n -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- The number of distinct prime factors of a natural number n -/
def num_distinct_prime_factors (n : ℕ) : ℕ := sorry

/-- Theorem: The number of distinct prime factors of the sum of positive divisors of 360 is 4 -/
theorem distinct_prime_factors_of_divisor_sum_360 : 
  num_distinct_prime_factors (sum_of_divisors 360) = 4 := by sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_of_divisor_sum_360_l599_59909


namespace NUMINAMATH_CALUDE_polynomial_consecutive_integers_l599_59988

/-- A polynomial P(n) = (n^5 + a) / b takes integer values for three consecutive integers
    if and only if (a, b) = (k, 1) or (11k ± 1, 11) for some integer k. -/
theorem polynomial_consecutive_integers (a b : ℕ+) :
  (∃ n : ℤ, ∀ i ∈ ({0, 1, 2} : Set ℤ), ∃ k : ℤ, (n + i)^5 + a = b * k) ↔
  (∃ k : ℤ, (a = k ∧ b = 1) ∨ (a = 11 * k + 1 ∧ b = 11) ∨ (a = 11 * k - 1 ∧ b = 11)) :=
sorry

end NUMINAMATH_CALUDE_polynomial_consecutive_integers_l599_59988


namespace NUMINAMATH_CALUDE_one_and_two_red_mutually_exclusive_not_opposing_l599_59905

/-- Represents the number of red balls drawn -/
inductive RedBallsDrawn
  | zero
  | one
  | two
  | three

/-- The probability of drawing exactly one red ball -/
def prob_one_red : ℝ := sorry

/-- The probability of drawing exactly two red balls -/
def prob_two_red : ℝ := sorry

/-- The total number of balls in the bag -/
def total_balls : ℕ := 8

/-- The number of red balls in the bag -/
def red_balls : ℕ := 5

/-- The number of white balls in the bag -/
def white_balls : ℕ := 3

/-- The number of balls drawn -/
def drawn_balls : ℕ := 3

theorem one_and_two_red_mutually_exclusive_not_opposing :
  (prob_one_red * prob_two_red = 0) ∧ (prob_one_red + prob_two_red < 1) := by
  sorry

end NUMINAMATH_CALUDE_one_and_two_red_mutually_exclusive_not_opposing_l599_59905


namespace NUMINAMATH_CALUDE_fraction_simplification_l599_59986

theorem fraction_simplification (a b : ℝ) (h : b ≠ 0) :
  b / (a * b + b) = 1 / (a + 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l599_59986


namespace NUMINAMATH_CALUDE_capital_ratio_l599_59927

-- Define the partners' capitals
variable (a b c : ℝ)

-- Define the total profit and b's share
def total_profit : ℝ := 16500
def b_share : ℝ := 6000

-- State the theorem
theorem capital_ratio (h1 : b = 4 * c) (h2 : b_share / total_profit = b / (a + b + c)) :
  a / b = 17 / 4 := by
  sorry

end NUMINAMATH_CALUDE_capital_ratio_l599_59927


namespace NUMINAMATH_CALUDE_tan_beta_value_l599_59957

theorem tan_beta_value (α β : Real) 
  (h1 : Real.tan α = -3/4)
  (h2 : Real.tan (α + β) = 1) : 
  Real.tan β = 7 := by
sorry

end NUMINAMATH_CALUDE_tan_beta_value_l599_59957


namespace NUMINAMATH_CALUDE_fruit_boxes_distribution_l599_59966

/-- Given 22 boxes distributed among 3 types of fruits, 
    prove that there must be at least 8 boxes of one type of fruit. -/
theorem fruit_boxes_distribution (boxes : ℕ) (fruit_types : ℕ) 
  (h1 : boxes = 22) (h2 : fruit_types = 3) : 
  ∃ (type : ℕ), type ≤ fruit_types ∧ 
  ∃ (boxes_of_type : ℕ), boxes_of_type ≥ 8 ∧ 
  boxes_of_type ≤ boxes := by
  sorry

end NUMINAMATH_CALUDE_fruit_boxes_distribution_l599_59966


namespace NUMINAMATH_CALUDE_functional_equation_properties_l599_59976

theorem functional_equation_properties 
  (f g h : ℝ → ℝ)
  (hf : ∀ x y : ℝ, f (x * y) = x * f y)
  (hg : ∀ x y : ℝ, g (x * y) = x * g y)
  (hh : ∀ x y : ℝ, h (x * y) = x * h y) :
  (∀ x y : ℝ, f (x + y) = f x + f y) ∧
  (∀ x y : ℝ, (g ∘ h) (x * y) = x * (g ∘ h) y) ∧
  (g ∘ h = h ∘ g) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_properties_l599_59976


namespace NUMINAMATH_CALUDE_movie_original_length_l599_59920

/-- The original length of a movie, given the length of a cut scene and the final length -/
def original_length (cut_scene_length final_length : ℕ) : ℕ :=
  final_length + cut_scene_length

/-- Theorem: The original length of the movie is 60 minutes -/
theorem movie_original_length : original_length 6 54 = 60 := by
  sorry

end NUMINAMATH_CALUDE_movie_original_length_l599_59920


namespace NUMINAMATH_CALUDE_product_sum_division_l599_59958

theorem product_sum_division : (10 * 19 * 20 * 53 * 100 + 601) / 13 = 1549277 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_division_l599_59958


namespace NUMINAMATH_CALUDE_train_length_proof_l599_59930

theorem train_length_proof (platform_crossing_time : ℝ) (pole_crossing_time : ℝ) (platform_length : ℝ) :
  platform_crossing_time = 39 →
  pole_crossing_time = 18 →
  platform_length = 1050 →
  ∃ train_length : ℝ, train_length = 900 ∧
    train_length / pole_crossing_time = (train_length + platform_length) / platform_crossing_time :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_proof_l599_59930


namespace NUMINAMATH_CALUDE_unique_n_solution_l599_59970

theorem unique_n_solution : ∃! (n : ℕ+), 
  Real.cos (π / (2 * n.val)) - Real.sin (π / (2 * n.val)) = Real.sqrt n.val / 2 :=
by
  -- The unique solution is n = 4
  use 4
  constructor
  -- Proof that n = 4 satisfies the equation
  sorry
  -- Proof of uniqueness
  sorry

end NUMINAMATH_CALUDE_unique_n_solution_l599_59970


namespace NUMINAMATH_CALUDE_polynomial_root_sum_l599_59926

theorem polynomial_root_sum (a b : ℝ) : 
  (Complex.I * Real.sqrt 2 + 2 : ℂ) ^ 3 + a * (Complex.I * Real.sqrt 2 + 2) + b = 0 → 
  a + b = 14 := by sorry

end NUMINAMATH_CALUDE_polynomial_root_sum_l599_59926


namespace NUMINAMATH_CALUDE_infant_weight_at_four_months_l599_59951

/-- Represents the weight of an infant in grams at a given age in months. -/
def infantWeight (birthWeight : ℝ) (ageMonths : ℝ) : ℝ :=
  birthWeight + 700 * ageMonths

/-- Theorem stating that an infant with a birth weight of 3000 grams will weigh 5800 grams at 4 months. -/
theorem infant_weight_at_four_months :
  infantWeight 3000 4 = 5800 := by
  sorry

end NUMINAMATH_CALUDE_infant_weight_at_four_months_l599_59951


namespace NUMINAMATH_CALUDE_correct_calculation_l599_59933

theorem correct_calculation (a : ℝ) : 2 * a^4 * 3 * a^5 = 6 * a^9 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l599_59933


namespace NUMINAMATH_CALUDE_compute_expression_l599_59996

/-- Operation Δ: a Δ b = a × 100...00 (b zeros) + b -/
def delta (a b : ℕ) : ℕ := a * (10^b) + b

/-- Operation □: a □ b = a × 10 + b -/
def square (a b : ℕ) : ℕ := a * 10 + b

/-- Theorem: 2018 □ (123 Δ 4) = 1250184 -/
theorem compute_expression : square 2018 (delta 123 4) = 1250184 := by sorry

end NUMINAMATH_CALUDE_compute_expression_l599_59996


namespace NUMINAMATH_CALUDE_simple_interest_rate_percent_l599_59947

theorem simple_interest_rate_percent : 
  ∀ (principal interest time rate : ℝ),
  principal = 800 →
  interest = 176 →
  time = 4 →
  interest = principal * rate * time / 100 →
  rate = 5.5 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_rate_percent_l599_59947


namespace NUMINAMATH_CALUDE_cube_sum_given_sum_and_square_sum_l599_59912

theorem cube_sum_given_sum_and_square_sum (x y : ℝ) 
  (h1 : x + y = 5) 
  (h2 : x^2 + y^2 = 17) : 
  x^3 + y^3 = 65 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_given_sum_and_square_sum_l599_59912


namespace NUMINAMATH_CALUDE_negative_expression_l599_59904

theorem negative_expression : 
  (|(-4)| > 0) ∧ (-(-4) > 0) ∧ ((-4)^2 > 0) ∧ (-4^2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negative_expression_l599_59904


namespace NUMINAMATH_CALUDE_expression_equality_l599_59906

theorem expression_equality : (8 : ℕ)^6 * 27^6 * 8^18 * 27^18 = 216^24 := by sorry

end NUMINAMATH_CALUDE_expression_equality_l599_59906


namespace NUMINAMATH_CALUDE_x_power_expression_l599_59948

theorem x_power_expression (x : ℝ) (h : x + 1/x = 3) :
  x^7 - 5*x^5 + 3*x^3 = 126*x - 48 := by
  sorry

end NUMINAMATH_CALUDE_x_power_expression_l599_59948


namespace NUMINAMATH_CALUDE_plane_perpendicularity_l599_59971

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (parallelLP : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicularP : Plane → Plane → Prop)

-- State the theorem
theorem plane_perpendicularity 
  (m n : Line) (α β : Plane) 
  (h1 : m ≠ n) 
  (h2 : α ≠ β) 
  (h3 : parallel m n) 
  (h4 : parallelLP m α) 
  (h5 : perpendicular n β) : 
  perpendicularP α β := by sorry

end NUMINAMATH_CALUDE_plane_perpendicularity_l599_59971


namespace NUMINAMATH_CALUDE_train_passing_time_l599_59969

/-- The time taken for a train to pass a man moving in the same direction -/
theorem train_passing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) :
  train_length = 375 →
  train_speed = 72 * (1000 / 3600) →
  man_speed = 12 * (1000 / 3600) →
  (train_length / (train_speed - man_speed)) = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_time_l599_59969


namespace NUMINAMATH_CALUDE_stream_speed_l599_59942

/-- Given a boat with speed 78 kmph in still water, if the time taken upstream is twice
    the time taken downstream, then the speed of the stream is 26 kmph. -/
theorem stream_speed (D : ℝ) (D_pos : D > 0) : 
  let boat_speed : ℝ := 78
  let stream_speed : ℝ := 26
  (D / (boat_speed - stream_speed) = 2 * (D / (boat_speed + stream_speed))) →
  stream_speed = 26 := by
sorry

end NUMINAMATH_CALUDE_stream_speed_l599_59942


namespace NUMINAMATH_CALUDE_initial_men_count_l599_59994

/-- Given a group of men with provisions lasting 18 days, prove that the initial number of men is 1000 
    when 400 more men join and the provisions then last 12.86 days, assuming the total amount of provisions remains constant. -/
theorem initial_men_count (initial_days : ℝ) (final_days : ℝ) (additional_men : ℕ) :
  initial_days = 18 →
  final_days = 12.86 →
  additional_men = 400 →
  ∃ (initial_men : ℕ), 
    initial_men * initial_days = (initial_men + additional_men) * final_days ∧
    initial_men = 1000 := by
  sorry

end NUMINAMATH_CALUDE_initial_men_count_l599_59994


namespace NUMINAMATH_CALUDE_hanks_total_reading_time_l599_59936

/-- Represents Hank's weekly reading schedule --/
structure ReadingSchedule where
  newspaper_days : Nat
  newspaper_time : Nat
  magazine_time : Nat
  novel_days : Nat
  novel_time : Nat
  novel_friday_time : Nat
  novel_saturday_multiplier : Nat
  novel_sunday_multiplier : Nat
  scientific_journal_time : Nat
  nonfiction_time : Nat

/-- Calculates the total reading time for a week given a reading schedule --/
def total_reading_time (schedule : ReadingSchedule) : Nat :=
  let newspaper_total := schedule.newspaper_days * schedule.newspaper_time
  let magazine_total := schedule.magazine_time
  let novel_weekday_total := (schedule.novel_days - 3) * schedule.novel_time
  let novel_friday_total := schedule.novel_friday_time
  let novel_saturday_total := schedule.novel_time * schedule.novel_saturday_multiplier
  let novel_sunday_total := schedule.novel_time * schedule.novel_sunday_multiplier
  let scientific_journal_total := schedule.scientific_journal_time
  let nonfiction_total := schedule.nonfiction_time
  newspaper_total + magazine_total + novel_weekday_total + novel_friday_total +
  novel_saturday_total + novel_sunday_total + scientific_journal_total + nonfiction_total

/-- Hank's actual reading schedule --/
def hanks_schedule : ReadingSchedule :=
  { newspaper_days := 5
  , newspaper_time := 30
  , magazine_time := 15
  , novel_days := 5
  , novel_time := 60
  , novel_friday_time := 90
  , novel_saturday_multiplier := 2
  , novel_sunday_multiplier := 3
  , scientific_journal_time := 45
  , nonfiction_time := 40
  }

/-- Theorem stating that Hank's total reading time in a week is 760 minutes --/
theorem hanks_total_reading_time :
  total_reading_time hanks_schedule = 760 := by
  sorry

end NUMINAMATH_CALUDE_hanks_total_reading_time_l599_59936


namespace NUMINAMATH_CALUDE_donation_change_l599_59979

def original_donations : List ℕ := [5, 3, 6, 5, 10]

def median (l : List ℕ) : ℕ := sorry

def mode (l : List ℕ) : List ℕ := sorry

def new_donations (a : ℕ) : List ℕ :=
  original_donations.map (fun x => if x = 3 then x + a else x)

theorem donation_change (a : ℕ) :
  (median (new_donations a) = median original_donations ∧
   mode (new_donations a) = mode original_donations) ↔ (a = 1 ∨ a = 2) := by sorry

end NUMINAMATH_CALUDE_donation_change_l599_59979


namespace NUMINAMATH_CALUDE_greatest_two_digit_with_digit_product_12_l599_59928

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ := (n / 10) * (n % 10)

theorem greatest_two_digit_with_digit_product_12 :
  ∀ n : ℕ, is_two_digit n → digit_product n = 12 → n ≤ 62 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_two_digit_with_digit_product_12_l599_59928


namespace NUMINAMATH_CALUDE_B_power_six_eq_84B_minus_44I_l599_59925

def B : Matrix (Fin 2) (Fin 2) ℤ := !![0, 3; 2, -1]

theorem B_power_six_eq_84B_minus_44I :
  B^6 = 84 • B - 44 • (1 : Matrix (Fin 2) (Fin 2) ℤ) := by
  sorry

end NUMINAMATH_CALUDE_B_power_six_eq_84B_minus_44I_l599_59925


namespace NUMINAMATH_CALUDE_g_composition_l599_59959

def g (x : ℝ) : ℝ := 3 * x + 2

theorem g_composition : g (g (g 3)) = 107 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_l599_59959


namespace NUMINAMATH_CALUDE_marble_distribution_l599_59907

theorem marble_distribution (total_marbles : ℕ) (additional_people : ℕ) : 
  total_marbles = 220 →
  additional_people = 2 →
  ∃ (x : ℕ), 
    (x > 0) ∧ 
    (total_marbles / x - 1 = total_marbles / (x + additional_people)) ∧
    x = 20 :=
by sorry

end NUMINAMATH_CALUDE_marble_distribution_l599_59907


namespace NUMINAMATH_CALUDE_frustum_volume_l599_59955

/-- The volume of a frustum of a right square pyramid inscribed in a sphere -/
theorem frustum_volume (R : ℝ) (β : ℝ) : 
  (R > 0) → (π/4 < β) → (β < π/2) →
  ∃ V : ℝ, V = (2/3) * R^3 * Real.sin (2*β) * (1 + Real.cos (2*β)^2 - Real.cos (2*β)) :=
by sorry

end NUMINAMATH_CALUDE_frustum_volume_l599_59955


namespace NUMINAMATH_CALUDE_cannot_be_equation_l599_59921

-- Define the linear function
def linear_function (k b : ℝ) (x : ℝ) : ℝ := k * x + b

-- Define the condition that the function passes through (-1, -3)
def passes_through_point (k b : ℝ) : Prop :=
  linear_function k b (-1) = -3

-- Define the condition that the distances from intercepts to origin are equal
def equal_intercept_distances (k b : ℝ) : Prop :=
  abs (b / k) = abs b

-- Theorem statement
theorem cannot_be_equation (k b : ℝ) 
  (h1 : passes_through_point k b) 
  (h2 : equal_intercept_distances k b) :
  ¬(k = -3 ∧ b = -6) :=
sorry

end NUMINAMATH_CALUDE_cannot_be_equation_l599_59921


namespace NUMINAMATH_CALUDE_smallest_b_in_arithmetic_sequence_l599_59952

/-- Given four positive terms in an arithmetic sequence with their product equal to 256,
    the smallest possible value of the second term is 4. -/
theorem smallest_b_in_arithmetic_sequence (a b c d : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d →  -- all terms are positive
  ∃ (r : ℝ), a = b - r ∧ c = b + r ∧ d = b + 2*r →  -- arithmetic sequence
  a * b * c * d = 256 →  -- product is 256
  b ≥ 4 ∧ ∃ (r : ℝ), 4 - r > 0 ∧ 4 * (4 - r) * (4 + r) * (4 + 2*r) = 256 :=  -- b ≥ 4 and there exists a valid r for b = 4
by sorry

end NUMINAMATH_CALUDE_smallest_b_in_arithmetic_sequence_l599_59952


namespace NUMINAMATH_CALUDE_midline_tetrahedra_volume_ratio_l599_59910

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  volume : ℝ

/-- A tetrahedron formed by connecting midpoints of midlines to vertices -/
structure MidlineTetrahedron where
  volume : ℝ

/-- The common part of three MidlineTetrahedra -/
structure CommonTetrahedron where
  volume : ℝ

/-- Given a regular tetrahedron, construct three MidlineTetrahedra -/
def construct_midline_tetrahedra (t : RegularTetrahedron) : 
  (MidlineTetrahedron × MidlineTetrahedron × MidlineTetrahedron) :=
  sorry

/-- Find the common part of three MidlineTetrahedra -/
def find_common_part (t1 t2 t3 : MidlineTetrahedron) : CommonTetrahedron :=
  sorry

/-- The theorem to be proved -/
theorem midline_tetrahedra_volume_ratio 
  (t : RegularTetrahedron) 
  (t1 t2 t3 : MidlineTetrahedron) 
  (c : CommonTetrahedron) :
  t1 = (construct_midline_tetrahedra t).1 ∧
  t2 = (construct_midline_tetrahedra t).2.1 ∧
  t3 = (construct_midline_tetrahedra t).2.2 ∧
  c = find_common_part t1 t2 t3 →
  c.volume = t.volume / 10 :=
by sorry

end NUMINAMATH_CALUDE_midline_tetrahedra_volume_ratio_l599_59910


namespace NUMINAMATH_CALUDE_slope_range_l599_59918

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := 4 * x^2 + 25 * y^2 = 100

-- Define the line equation
def line (m : ℝ) (x y : ℝ) : Prop := y = m * x - 3

-- Define the intersection condition
def intersects (m : ℝ) : Prop := ∃ x y : ℝ, ellipse x y ∧ line m x y

-- State the theorem
theorem slope_range (m : ℝ) : 
  intersects m ↔ m ≤ -Real.sqrt (1/5) ∨ m ≥ Real.sqrt (1/5) :=
sorry

end NUMINAMATH_CALUDE_slope_range_l599_59918


namespace NUMINAMATH_CALUDE_smallest_n_square_fifth_power_l599_59932

theorem smallest_n_square_fifth_power : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (∃ (k : ℕ), 2 * n = k^2) ∧ 
  (∃ (m : ℕ), 5 * n = m^5) ∧ 
  (∀ (x : ℕ), x > 0 → x < n → 
    (¬∃ (y : ℕ), 2 * x = y^2) ∨ 
    (¬∃ (z : ℕ), 5 * x = z^5)) ∧
  n = 5000 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_square_fifth_power_l599_59932


namespace NUMINAMATH_CALUDE_garden_perimeter_l599_59940

/-- A rectangular garden with given length and breadth has a specific perimeter. -/
theorem garden_perimeter (length breadth : ℝ) (h1 : length = 360) (h2 : breadth = 240) :
  2 * (length + breadth) = 1200 :=
by sorry

end NUMINAMATH_CALUDE_garden_perimeter_l599_59940


namespace NUMINAMATH_CALUDE_factor_expression_l599_59987

theorem factor_expression (x : ℝ) : 75 * x^11 + 200 * x^22 = 25 * x^11 * (3 + 8 * x^11) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l599_59987


namespace NUMINAMATH_CALUDE_tangent_line_correct_l599_59937

-- Define the curve
def curve (x : ℝ) : ℝ := -x^2 + 6*x

-- Define the derivative of the curve
def curve_derivative (x : ℝ) : ℝ := -2*x + 6

-- Define the proposed tangent line
def tangent_line (x : ℝ) : ℝ := 6*x

theorem tangent_line_correct :
  -- The tangent line passes through the origin
  tangent_line 0 = 0 ∧
  -- The tangent line touches the curve at some point
  ∃ x : ℝ, curve x = tangent_line x ∧
  -- The slope of the tangent line equals the derivative of the curve at the point of tangency
  curve_derivative x = 6 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_correct_l599_59937


namespace NUMINAMATH_CALUDE_square_roots_problem_l599_59943

theorem square_roots_problem (n : ℝ) (a : ℝ) (h1 : n > 0) 
  (h2 : (a + 3) ^ 2 = n) (h3 : (2 * a + 3) ^ 2 = n) : n = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_problem_l599_59943


namespace NUMINAMATH_CALUDE_hyperbola_focus_l599_59963

/-- The hyperbola equation -/
def hyperbola_eq (x y : ℝ) : Prop :=
  2 * x^2 - y^2 + 8 * x - 6 * y - 8 = 0

/-- Definition of a focus for this hyperbola -/
def is_focus (x y : ℝ) : Prop :=
  ∃ (sign : ℝ), sign = 1 ∨ sign = -1 ∧
  x = -2 + sign * Real.sqrt 10.5 ∧
  y = -3

/-- Theorem stating that (-2 + √10.5, -3) is a focus of the hyperbola -/
theorem hyperbola_focus :
  ∃ (x y : ℝ), hyperbola_eq x y ∧ is_focus x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_focus_l599_59963


namespace NUMINAMATH_CALUDE_hydropolis_aquaville_rainfall_difference_l599_59975

/-- The difference in total rainfall between two cities over a year, given their average monthly rainfalls and the number of months. -/
def rainfall_difference (avg_rainfall_city1 avg_rainfall_city2 : ℝ) (months : ℕ) : ℝ :=
  (avg_rainfall_city1 - avg_rainfall_city2) * months

/-- Theorem stating the difference in total rainfall between Hydropolis and Aquaville in 2011 -/
theorem hydropolis_aquaville_rainfall_difference :
  let hydropolis_2010 : ℝ := 36.5
  let rainfall_increase : ℝ := 3.5
  let hydropolis_2011 : ℝ := hydropolis_2010 + rainfall_increase
  let aquaville_2011 : ℝ := hydropolis_2011 - 1.5
  let months : ℕ := 12
  rainfall_difference hydropolis_2011 aquaville_2011 months = 18.0 := by
  sorry

#eval rainfall_difference 40.0 38.5 12

end NUMINAMATH_CALUDE_hydropolis_aquaville_rainfall_difference_l599_59975


namespace NUMINAMATH_CALUDE_tan_half_sum_l599_59964

theorem tan_half_sum (p q : Real) 
  (h1 : Real.cos p + Real.cos q = 1/3)
  (h2 : Real.sin p + Real.sin q = 4/9) : 
  Real.tan ((p + q) / 2) = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_half_sum_l599_59964


namespace NUMINAMATH_CALUDE_two_possible_values_l599_59991

def triangle (a b : ℕ) : ℕ := min a b

def nabla (a b : ℕ) : ℕ := max a b

theorem two_possible_values (x : ℕ) : 
  ∃ (s : Finset ℕ), (s.card = 2) ∧ 
  (triangle 6 (nabla 4 (triangle x 5)) ∈ s) := by
  sorry

end NUMINAMATH_CALUDE_two_possible_values_l599_59991


namespace NUMINAMATH_CALUDE_number_problem_l599_59919

theorem number_problem : 
  ∃ (n : ℝ), n - (102 / 20.4) = 5095 ∧ n = 5100 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l599_59919


namespace NUMINAMATH_CALUDE_difference_of_squares_l599_59999

theorem difference_of_squares (x y : ℝ) : 
  x > 0 → y > 0 → x < y → 
  Real.sqrt x + Real.sqrt y = 1 → 
  Real.sqrt (x / y) + Real.sqrt (y / x) = 10 / 3 → 
  y - x = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_difference_of_squares_l599_59999


namespace NUMINAMATH_CALUDE_students_neither_activity_l599_59923

theorem students_neither_activity (total : ℕ) (table_tennis : ℕ) (chess : ℕ) (both : ℕ) :
  total = 12 →
  table_tennis = 5 →
  chess = 8 →
  both = 3 →
  total - (table_tennis + chess - both) = 2 :=
by sorry

end NUMINAMATH_CALUDE_students_neither_activity_l599_59923


namespace NUMINAMATH_CALUDE_distance_between_A_and_B_l599_59944

def point_A : Fin 3 → ℝ := ![1, 2, 3]
def point_B : Fin 3 → ℝ := ![-1, 3, -2]

theorem distance_between_A_and_B : 
  Real.sqrt (((point_B 0 - point_A 0)^2 + (point_B 1 - point_A 1)^2 + (point_B 2 - point_A 2)^2 : ℝ)) = Real.sqrt 30 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_A_and_B_l599_59944


namespace NUMINAMATH_CALUDE_alBr3_weight_calculation_l599_59903

/-- Calculates the weight of AlBr3 given moles and isotope data -/
def weightAlBr3 (moles : ℝ) (alMass : ℝ) (br79Mass br81Mass : ℝ) (br79Abundance br81Abundance : ℝ) : ℝ :=
  let brAvgMass := br79Mass * br79Abundance + br81Mass * br81Abundance
  let molarMass := alMass + 3 * brAvgMass
  moles * molarMass

/-- The weight of 4 moles of AlBr3 is approximately 1067.2344 grams -/
theorem alBr3_weight_calculation :
  let moles : ℝ := 4
  let alMass : ℝ := 27
  let br79Mass : ℝ := 79
  let br81Mass : ℝ := 81
  let br79Abundance : ℝ := 0.5069
  let br81Abundance : ℝ := 0.4931
  ∃ ε > 0, |weightAlBr3 moles alMass br79Mass br81Mass br79Abundance br81Abundance - 1067.2344| < ε :=
by sorry

end NUMINAMATH_CALUDE_alBr3_weight_calculation_l599_59903


namespace NUMINAMATH_CALUDE_rachel_brownies_l599_59997

def brownies_baked (brought_to_school left_at_home : ℕ) : ℕ :=
  brought_to_school + left_at_home

theorem rachel_brownies : 
  brownies_baked 16 24 = 40 := by
  sorry

end NUMINAMATH_CALUDE_rachel_brownies_l599_59997


namespace NUMINAMATH_CALUDE_chandler_total_rolls_l599_59983

/-- The total number of rolls Chandler needs to sell for the school fundraiser -/
def total_rolls_to_sell : ℕ :=
  let grandmother_rolls := 3
  let uncle_rolls := 4
  let neighbor_rolls := 3
  let additional_rolls := 2
  grandmother_rolls + uncle_rolls + neighbor_rolls + additional_rolls

/-- Theorem stating that Chandler needs to sell 12 rolls in total -/
theorem chandler_total_rolls : total_rolls_to_sell = 12 := by
  sorry

end NUMINAMATH_CALUDE_chandler_total_rolls_l599_59983


namespace NUMINAMATH_CALUDE_identity_function_satisfies_equation_l599_59945

theorem identity_function_satisfies_equation (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x - f (x - y)) + x = f (x + y)) →
  (∀ x : ℝ, f x = x) :=
by sorry

end NUMINAMATH_CALUDE_identity_function_satisfies_equation_l599_59945


namespace NUMINAMATH_CALUDE_first_division_percentage_l599_59934

theorem first_division_percentage 
  (total_students : ℕ) 
  (second_division_percentage : ℚ)
  (just_passed_count : ℕ) :
  total_students = 300 →
  second_division_percentage = 54/100 →
  just_passed_count = 51 →
  ∃ (first_division_percentage : ℚ),
    first_division_percentage = 29/100 ∧
    first_division_percentage + second_division_percentage + (just_passed_count : ℚ) / total_students = 1 :=
by sorry

end NUMINAMATH_CALUDE_first_division_percentage_l599_59934


namespace NUMINAMATH_CALUDE_inequality_proof_l599_59916

open Real

theorem inequality_proof (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x > 0, f x = Real.log x - 3 * x) →
  (∀ x > 0, f x ≤ x * (a * Real.exp x - 4) + b) →
  a + b ≥ 0 := by
    sorry

end NUMINAMATH_CALUDE_inequality_proof_l599_59916


namespace NUMINAMATH_CALUDE_x_plus_y_value_l599_59915

theorem x_plus_y_value (x y : ℝ) 
  (h1 : |x| + x + y = 10) 
  (h2 : x + |y| - y = 12) : 
  x + y = 18/5 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l599_59915


namespace NUMINAMATH_CALUDE_share_premium_percentage_l599_59938

/-- Calculates the premium percentage on shares given investment details -/
theorem share_premium_percentage
  (total_investment : ℝ)
  (face_value : ℝ)
  (dividend_rate : ℝ)
  (total_dividend : ℝ)
  (h1 : total_investment = 14400)
  (h2 : face_value = 100)
  (h3 : dividend_rate = 0.07)
  (h4 : total_dividend = 840) :
  (total_investment / (total_dividend / (dividend_rate * face_value)) - face_value) / face_value * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_share_premium_percentage_l599_59938


namespace NUMINAMATH_CALUDE_product_greater_than_sum_plus_two_l599_59941

theorem product_greater_than_sum_plus_two 
  (a b c : ℝ) 
  (ha : a > 1) 
  (hb : b > 1) 
  (hc : c > 1) 
  (hab : a * b > a + b) 
  (hbc : b * c > b + c) 
  (hac : a * c > a + c) : 
  a * b * c > a + b + c + 2 := by
sorry

end NUMINAMATH_CALUDE_product_greater_than_sum_plus_two_l599_59941


namespace NUMINAMATH_CALUDE_expression_value_l599_59980

theorem expression_value : 
  (1/2) * (Real.log 12 / Real.log 3) - (Real.log 2 / Real.log 3) + 
  (Real.log 3 / Real.log 2) * (Real.log 4 / Real.log 3) + 
  ((-2)^4)^(1/4) + (Real.sqrt 3 - 1)^0 = 11/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l599_59980


namespace NUMINAMATH_CALUDE_square_side_length_l599_59931

theorem square_side_length (area : ℝ) (side : ℝ) (h1 : area = 12) (h2 : area = side ^ 2) :
  side = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l599_59931


namespace NUMINAMATH_CALUDE_change_received_l599_59950

-- Define the number of apples
def num_apples : ℕ := 5

-- Define the cost per apple in cents
def cost_per_apple : ℕ := 30

-- Define the amount paid in dollars
def amount_paid : ℚ := 10

-- Define the function to calculate change
def calculate_change (num_apples : ℕ) (cost_per_apple : ℕ) (amount_paid : ℚ) : ℚ :=
  amount_paid - (num_apples * cost_per_apple : ℚ) / 100

-- Theorem statement
theorem change_received :
  calculate_change num_apples cost_per_apple amount_paid = 8.5 := by
  sorry

end NUMINAMATH_CALUDE_change_received_l599_59950


namespace NUMINAMATH_CALUDE_largest_angle_in_3_4_5_ratio_triangle_l599_59939

theorem largest_angle_in_3_4_5_ratio_triangle : 
  ∀ (a b c : ℝ), 
    a > 0 → b > 0 → c > 0 →
    b = (4/3) * a →
    c = (5/3) * a →
    a + b + c = 180 →
    c = 75 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_in_3_4_5_ratio_triangle_l599_59939


namespace NUMINAMATH_CALUDE_expected_terms_is_ten_l599_59914

/-- A fair tetrahedral die with faces numbered 1 to 4 -/
structure TetrahedralDie :=
  (faces : Finset Nat)
  (fair : faces = {1, 2, 3, 4})

/-- The state of the sequence -/
inductive SequenceState
| Zero  : SequenceState  -- No distinct numbers seen
| One   : SequenceState  -- One distinct number seen
| Two   : SequenceState  -- Two distinct numbers seen
| Three : SequenceState  -- Three distinct numbers seen
| Four  : SequenceState  -- All four numbers seen

/-- Expected number of terms to complete the sequence from a given state -/
noncomputable def expectedTerms (s : SequenceState) : ℝ :=
  match s with
  | SequenceState.Zero  => sorry
  | SequenceState.One   => sorry
  | SequenceState.Two   => sorry
  | SequenceState.Three => sorry
  | SequenceState.Four  => 0

/-- Main theorem: The expected number of terms in the sequence is 10 -/
theorem expected_terms_is_ten (d : TetrahedralDie) : 
  expectedTerms SequenceState.Zero = 10 := by sorry

end NUMINAMATH_CALUDE_expected_terms_is_ten_l599_59914


namespace NUMINAMATH_CALUDE_rational_root_uniqueness_l599_59924

/-- A natural number is valid if it's a square and each of its prime divisors has an even number of decimal digits. -/
def IsValidNumber (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^2 ∧ ∀ p : ℕ, Prime p → p ∣ n → Even (Nat.digits 10 p).length

/-- The polynomial P(x) = x^n - 1987x -/
def P (n : ℕ) (x : ℚ) : ℚ := x^n - 1987*x

theorem rational_root_uniqueness (n : ℕ) (h : IsValidNumber n) :
  ∀ x y : ℚ, P n x = P n y → x = y := by
  sorry

end NUMINAMATH_CALUDE_rational_root_uniqueness_l599_59924


namespace NUMINAMATH_CALUDE_triangle_square_side_ratio_l599_59949

theorem triangle_square_side_ratio (t s : ℝ) : 
  t > 0 ∧ s > 0 ∧ 3 * t = 12 ∧ 4 * s = 12 → t / s = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_square_side_ratio_l599_59949


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l599_59968

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3 * b = 1) :
  (1 / a + 1 / b) ≥ 4 + 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l599_59968


namespace NUMINAMATH_CALUDE_power_calculation_l599_59961

theorem power_calculation : 16^4 * 8^2 / 4^12 = (1 : ℚ) / 4 := by sorry

end NUMINAMATH_CALUDE_power_calculation_l599_59961


namespace NUMINAMATH_CALUDE_complex_determinant_solution_l599_59984

-- Define the determinant operation
def det (a b c d : ℂ) : ℂ := a * d - b * c

-- Define the complex number i
def i : ℂ := Complex.I

-- Theorem statement
theorem complex_determinant_solution :
  ∀ z : ℂ, det 1 (-1) z (z * i) = 2 → z = 1 - i := by
  sorry

end NUMINAMATH_CALUDE_complex_determinant_solution_l599_59984


namespace NUMINAMATH_CALUDE_probability_same_color_is_240_11970_l599_59962

/-- The number of red marbles in the bag -/
def red_marbles : ℕ := 6

/-- The number of white marbles in the bag -/
def white_marbles : ℕ := 7

/-- The number of blue marbles in the bag -/
def blue_marbles : ℕ := 8

/-- The total number of marbles in the bag -/
def total_marbles : ℕ := red_marbles + white_marbles + blue_marbles

/-- The number of marbles drawn -/
def marbles_drawn : ℕ := 4

/-- The probability of drawing four marbles of the same color -/
def probability_same_color : ℚ :=
  (red_marbles * (red_marbles - 1) * (red_marbles - 2) * (red_marbles - 3) +
   white_marbles * (white_marbles - 1) * (white_marbles - 2) * (white_marbles - 3) +
   blue_marbles * (blue_marbles - 1) * (blue_marbles - 2) * (blue_marbles - 3)) /
  (total_marbles * (total_marbles - 1) * (total_marbles - 2) * (total_marbles - 3))

theorem probability_same_color_is_240_11970 :
  probability_same_color = 240 / 11970 := by
  sorry

end NUMINAMATH_CALUDE_probability_same_color_is_240_11970_l599_59962


namespace NUMINAMATH_CALUDE_factorization_equality_l599_59900

theorem factorization_equality (x y : ℝ) : -4*x^2 + 12*x*y - 9*y^2 = -(2*x - 3*y)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l599_59900


namespace NUMINAMATH_CALUDE_parabola_roots_difference_l599_59935

def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem parabola_roots_difference (a b c : ℝ) :
  (∃ (y : ℝ → ℝ), ∀ x, y x = parabola a b c x) →
  (∃ h k, ∀ x, parabola a b c x = a * (x - h)^2 + k) →
  (parabola a b c 2 = -4) →
  (parabola a b c 4 = 12) →
  (∃ m n, m > n ∧ parabola a b c m = 0 ∧ parabola a b c n = 0) →
  (∃ m n, m > n ∧ parabola a b c m = 0 ∧ parabola a b c n = 0 ∧ m - n = 2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_roots_difference_l599_59935


namespace NUMINAMATH_CALUDE_binary_10101_is_21_l599_59956

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_10101_is_21 :
  binary_to_decimal [true, false, true, false, true] = 21 := by
  sorry

end NUMINAMATH_CALUDE_binary_10101_is_21_l599_59956


namespace NUMINAMATH_CALUDE_area_ratio_hexagon_triangle_l599_59990

/-- Regular hexagon with vertices ABCDEF -/
structure RegularHexagon where
  vertices : Fin 6 → ℝ × ℝ
  is_regular : sorry

/-- Triangle ACE within the regular hexagon -/
def TriangleACE (h : RegularHexagon) : Set (ℝ × ℝ) :=
  {p | ∃ (i : Fin 3), p = h.vertices (2 * i)}

/-- Area of a regular hexagon -/
def area_hexagon (h : RegularHexagon) : ℝ := sorry

/-- Area of triangle ACE -/
def area_triangle (h : RegularHexagon) : ℝ := sorry

/-- The ratio of the area of triangle ACE to the area of the regular hexagon is 1/6 -/
theorem area_ratio_hexagon_triangle (h : RegularHexagon) :
  area_triangle h / area_hexagon h = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_area_ratio_hexagon_triangle_l599_59990


namespace NUMINAMATH_CALUDE_quadratic_c_value_l599_59998

/-- The quadratic function f(x) = -x^2 + cx + 8 is positive only on the open interval (2,6) -/
def quadratic_positive_on_interval (c : ℝ) : Prop :=
  ∀ x : ℝ, (-x^2 + c*x + 8 > 0) ↔ (2 < x ∧ x < 6)

/-- The value of c for which the quadratic function is positive only on (2,6) is 8 -/
theorem quadratic_c_value : ∃! c : ℝ, quadratic_positive_on_interval c ∧ c = 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_c_value_l599_59998


namespace NUMINAMATH_CALUDE_smallest_two_digit_with_digit_product_12_l599_59977

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

theorem smallest_two_digit_with_digit_product_12 :
  ∃ (n : ℕ), is_two_digit n ∧ digit_product n = 12 ∧
  ∀ (m : ℕ), is_two_digit m → digit_product m = 12 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_two_digit_with_digit_product_12_l599_59977


namespace NUMINAMATH_CALUDE_phillips_remaining_money_l599_59901

theorem phillips_remaining_money
  (initial_amount : ℕ)
  (spent_oranges : ℕ)
  (spent_apples : ℕ)
  (spent_candy : ℕ)
  (h1 : initial_amount = 95)
  (h2 : spent_oranges = 14)
  (h3 : spent_apples = 25)
  (h4 : spent_candy = 6) :
  initial_amount - (spent_oranges + spent_apples + spent_candy) = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_phillips_remaining_money_l599_59901


namespace NUMINAMATH_CALUDE_unique_negative_zero_l599_59992

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * x^2 + 1

-- State the theorem
theorem unique_negative_zero (a : ℝ) :
  (∃! x₀ : ℝ, f a x₀ = 0 ∧ x₀ < 0) ↔ a > 3/2 := by sorry

end NUMINAMATH_CALUDE_unique_negative_zero_l599_59992


namespace NUMINAMATH_CALUDE_train_length_l599_59922

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 60 → time = 15 → ∃ (length : ℝ), abs (length - 250.05) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_train_length_l599_59922


namespace NUMINAMATH_CALUDE_monotonic_increasing_condition_l599_59973

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 1

theorem monotonic_increasing_condition (a : ℝ) :
  (∀ x y, x ∈ Set.Icc 1 2 → y ∈ Set.Icc 1 2 → x ≤ y → f a x ≤ f a y) →
  a ≥ -1 :=
by sorry

end NUMINAMATH_CALUDE_monotonic_increasing_condition_l599_59973


namespace NUMINAMATH_CALUDE_pizza_order_theorem_l599_59954

/-- The number of pizzas needed for a group of people -/
def pizzas_needed (num_people : ℕ) (slices_per_person : ℕ) (slices_per_pizza : ℕ) : ℕ :=
  (num_people * slices_per_person + slices_per_pizza - 1) / slices_per_pizza

/-- Theorem: The number of pizzas needed for 18 people, where each person gets 3 slices
    and each pizza has 9 slices, is equal to 6 -/
theorem pizza_order_theorem :
  pizzas_needed 18 3 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_pizza_order_theorem_l599_59954


namespace NUMINAMATH_CALUDE_angle_bisector_relation_l599_59967

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point is on the angle bisector of the second and fourth quadrants -/
def isOnAngleBisector (p : Point) : Prop :=
  (p.x < 0 ∧ p.y > 0) ∨ (p.x > 0 ∧ p.y < 0) ∨ p = ⟨0, 0⟩

/-- Theorem stating that for any point on the angle bisector of the second and fourth quadrants, 
    its x-coordinate is the negative of its y-coordinate -/
theorem angle_bisector_relation (p : Point) (h : isOnAngleBisector p) : p.x = -p.y := by
  sorry

end NUMINAMATH_CALUDE_angle_bisector_relation_l599_59967


namespace NUMINAMATH_CALUDE_sin_seven_halves_pi_plus_theta_l599_59917

theorem sin_seven_halves_pi_plus_theta (θ : Real) 
  (h : Real.cos (3 * Real.pi + θ) = -(2 * Real.sqrt 2) / 3) : 
  Real.sin ((7 / 2) * Real.pi + θ) = -(2 * Real.sqrt 2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_seven_halves_pi_plus_theta_l599_59917


namespace NUMINAMATH_CALUDE_average_payment_is_657_l599_59911

/-- Represents the payment structure for a debt over a year -/
structure DebtPayment where
  base : ℕ  -- Base payment amount
  increment1 : ℕ  -- Increment for second segment
  increment2 : ℕ  -- Increment for third segment
  increment3 : ℕ  -- Increment for fourth segment
  increment4 : ℕ  -- Increment for fifth segment

/-- Calculates the average payment given the debt payment structure -/
def averagePayment (dp : DebtPayment) : ℚ :=
  let total := 
    20 * dp.base + 
    30 * (dp.base + dp.increment1) + 
    40 * (dp.base + dp.increment1 + dp.increment2) + 
    50 * (dp.base + dp.increment1 + dp.increment2 + dp.increment3) + 
    60 * (dp.base + dp.increment1 + dp.increment2 + dp.increment3 + dp.increment4)
  total / 200

/-- Theorem stating that the average payment for the given structure is $657 -/
theorem average_payment_is_657 (dp : DebtPayment) 
    (h1 : dp.base = 450)
    (h2 : dp.increment1 = 80)
    (h3 : dp.increment2 = 65)
    (h4 : dp.increment3 = 105)
    (h5 : dp.increment4 = 95) : 
  averagePayment dp = 657 := by
  sorry

end NUMINAMATH_CALUDE_average_payment_is_657_l599_59911


namespace NUMINAMATH_CALUDE_intersection_probability_is_four_sevenths_l599_59913

/-- A rectangular prism with dimensions 3, 4, and 5 units -/
structure RectangularPrism where
  length : ℕ := 3
  width : ℕ := 4
  height : ℕ := 5

/-- The probability that a plane determined by three randomly chosen distinct vertices
    intersects the interior of the prism -/
def intersection_probability (prism : RectangularPrism) : ℚ :=
  4/7

/-- Theorem stating that the probability of intersection is 4/7 -/
theorem intersection_probability_is_four_sevenths (prism : RectangularPrism) :
  intersection_probability prism = 4/7 := by
  sorry

end NUMINAMATH_CALUDE_intersection_probability_is_four_sevenths_l599_59913


namespace NUMINAMATH_CALUDE_ghost_entrance_exit_ways_l599_59982

/-- The number of windows in the haunted mansion -/
def num_windows : ℕ := 8

/-- The number of ways Georgie can enter and exit the mansion -/
def num_ways : ℕ := num_windows * (num_windows - 1)

/-- Theorem: The number of ways Georgie can enter and exit the mansion is 56 -/
theorem ghost_entrance_exit_ways : num_ways = 56 := by
  sorry

end NUMINAMATH_CALUDE_ghost_entrance_exit_ways_l599_59982


namespace NUMINAMATH_CALUDE_truck_catches_bus_l599_59965

-- Define the vehicles
structure Vehicle :=
  (speed : ℝ)

-- Define the initial positions
def initial_position (bus truck car : Vehicle) : Prop :=
  truck.speed > car.speed ∧ bus.speed > truck.speed

-- Define the time when car catches up with truck
def car_catches_truck (t : ℝ) : Prop := t = 10

-- Define the time when car catches up with bus
def car_catches_bus (t : ℝ) : Prop := t = 15

-- Theorem to prove
theorem truck_catches_bus 
  (bus truck car : Vehicle) 
  (h1 : initial_position bus truck car)
  (h2 : car_catches_truck 10)
  (h3 : car_catches_bus 15) :
  ∃ (t : ℝ), t = 15 ∧ 
    (truck.speed * (15 + t) = bus.speed * 15) :=
sorry

end NUMINAMATH_CALUDE_truck_catches_bus_l599_59965


namespace NUMINAMATH_CALUDE_adjacent_sides_equal_not_imply_rhombus_l599_59995

-- Define a quadrilateral
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

-- Define a rhombus
def is_rhombus (q : Quadrilateral) : Prop :=
  ∀ i j : Fin 4, dist (q.vertices i) (q.vertices j) = dist (q.vertices 0) (q.vertices 1)

-- Define adjacent sides
def adjacent_sides_equal (q : Quadrilateral) : Prop :=
  ∃ i : Fin 4, dist (q.vertices i) (q.vertices (i + 1)) = dist (q.vertices ((i + 1) % 4)) (q.vertices ((i + 2) % 4))

-- Theorem to prove
theorem adjacent_sides_equal_not_imply_rhombus :
  ¬(∀ q : Quadrilateral, adjacent_sides_equal q → is_rhombus q) :=
sorry

end NUMINAMATH_CALUDE_adjacent_sides_equal_not_imply_rhombus_l599_59995


namespace NUMINAMATH_CALUDE_sin_75_degrees_l599_59960

theorem sin_75_degrees : Real.sin (75 * π / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_75_degrees_l599_59960


namespace NUMINAMATH_CALUDE_square_value_l599_59981

theorem square_value (p : ℝ) (h1 : p + p = 75) (h2 : (p + p) + 2*p = 149) : p = 38 := by
  sorry

end NUMINAMATH_CALUDE_square_value_l599_59981


namespace NUMINAMATH_CALUDE_triangle_area_is_36_l599_59902

/-- The area of a triangle formed by three lines in a 2D plane -/
def triangleArea (line1 line2 line3 : ℝ → ℝ) : ℝ :=
  sorry

theorem triangle_area_is_36 :
  let line1 := fun (x : ℝ) ↦ 8
  let line2 := fun (x : ℝ) ↦ 2 + x
  let line3 := fun (x : ℝ) ↦ 2 - x
  triangleArea line1 line2 line3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_is_36_l599_59902


namespace NUMINAMATH_CALUDE_equation_one_solutions_l599_59972

theorem equation_one_solutions (x : ℝ) :
  (x - 1)^2 = 4 ↔ x = 3 ∨ x = -1 := by sorry

end NUMINAMATH_CALUDE_equation_one_solutions_l599_59972


namespace NUMINAMATH_CALUDE_path_length_in_60_degree_angle_l599_59978

/-- The total path length for a point inside a 60° angle -/
theorem path_length_in_60_degree_angle (a b : ℝ) (h : 0 < a ∧ a < b) :
  ∃ (path_length : ℝ),
    path_length = a + 2 * Real.sqrt ((a^2 + a*b + b^2) / 3) :=
by sorry

end NUMINAMATH_CALUDE_path_length_in_60_degree_angle_l599_59978


namespace NUMINAMATH_CALUDE_subset_of_A_l599_59953

def A : Set ℕ := {x | x ≤ 4}

theorem subset_of_A : {3} ⊆ A := by
  sorry

end NUMINAMATH_CALUDE_subset_of_A_l599_59953


namespace NUMINAMATH_CALUDE_price_increase_theorem_l599_59974

theorem price_increase_theorem (original_price : ℝ) (original_price_pos : original_price > 0) :
  let first_increase := original_price * 1.2
  let second_increase := first_increase * 1.15
  let total_increase := second_increase - original_price
  (total_increase / original_price) * 100 = 38 := by
sorry

end NUMINAMATH_CALUDE_price_increase_theorem_l599_59974


namespace NUMINAMATH_CALUDE_quadratic_two_real_roots_l599_59989

theorem quadratic_two_real_roots (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 + 4*x + k = 0 ∧ y^2 + 4*y + k = 0) → k ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_real_roots_l599_59989


namespace NUMINAMATH_CALUDE_initial_ball_count_is_three_l599_59985

def bat_cost : ℕ := 500
def ball_cost : ℕ := 100

def initial_purchase_cost : ℕ := 3800
def initial_bat_count : ℕ := 7

def second_purchase_cost : ℕ := 1750
def second_bat_count : ℕ := 3
def second_ball_count : ℕ := 5

theorem initial_ball_count_is_three : 
  ∃ (x : ℕ), 
    initial_bat_count * bat_cost + x * ball_cost = initial_purchase_cost ∧
    second_bat_count * bat_cost + second_ball_count * ball_cost = second_purchase_cost ∧
    x = 3 := by
  sorry

end NUMINAMATH_CALUDE_initial_ball_count_is_three_l599_59985
